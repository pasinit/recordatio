"""Tests for the nightly read + feedback API.

Covers the store read methods (user-scoped unit listing / raw search with
as_of), the MemoryFeedbackStore lifecycle, and the four HTTP endpoints via a
TestClient over a real (temp) SQLiteStore.
"""
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit
from tpasini.recordatio.proxy.feedback_endpoints import register_feedback_routes
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore

_AUTH = {"authorization": "Bearer test-key"}


def _app(store: SQLiteStore, fb: MemoryFeedbackStore, *, auto_apply_safe_ops: bool = False) -> FastAPI:
    app = FastAPI()
    register_feedback_routes(
        app, get_store=lambda: store, memory_feedback_store=fb, user_label="User",
        auto_apply_safe_ops=auto_apply_safe_ops,
    )
    return app


async def _client(app: FastAPI, *, auth: bool = True) -> httpx.AsyncClient:
    """In-loop ASGI client — runs the app in the test's own event loop
    thread, so the single-connection stores are touched from one thread
    exactly as they are under uvicorn (async endpoints on the loop thread)."""
    transport = httpx.ASGITransport(app=app)
    headers = dict(_AUTH) if auth else {}
    return httpx.AsyncClient(transport=transport, base_url="http://test", headers=headers)


def _unit(content: str, *, asserted: datetime, uid: str | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=uid or content.replace(" ", "-"),
        type=KnowledgeType.SEMANTIC,
        content=content,
        asserted_at=asserted,
        valid_from=asserted,
        provenance="test",
    )


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteStore:
    s = SQLiteStore(tmp_path / "mem.db")
    d1 = datetime(2026, 7, 30, 9, 0, tzinfo=UTC)
    d2 = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)
    # Two of Sam's scopes + another user, to prove user-scoping.
    s.add(_unit("Sam is a senior data analyst", asserted=d1, uid="u-old"), "sam/scout")
    s.add(_unit("Sam wants Springfield roles", asserted=d2, uid="u-new"), "sam/_user")
    s.add(_unit("Tommaso wants a X1 flat", asserted=d2, uid="u-other"), "tommaso/helper")
    return s


# ---------------------------------------------------------------------------
# Store read methods
# ---------------------------------------------------------------------------


def test_list_units_for_user_spans_scopes_and_excludes_other_users(store: SQLiteStore) -> None:
    pairs = store.list_units_for_user("sam")
    ids = {u.id for u, _ in pairs}
    assert ids == {"u-old", "u-new"}
    scopes = {agent_id for _, agent_id in pairs}
    assert scopes == {"sam/scout", "sam/_user"}


def test_list_units_for_user_filters_by_day(store: SQLiteStore) -> None:
    after = datetime(2026, 7, 31).isoformat()
    before = datetime(2026, 8, 1).isoformat()
    pairs = store.list_units_for_user("sam", asserted_after=after, asserted_before=before)
    assert {u.id for u, _ in pairs} == {"u-new"}


def test_raw_search_respects_as_of(store: SQLiteStore) -> None:
    # The old unit exists...
    assert store.raw_search_units_for_user("sam", "analyst")
    # ...but not as of before it was asserted.
    as_of = datetime(2026, 7, 30, 8, 0, tzinfo=UTC).isoformat()
    assert store.raw_search_units_for_user("sam", "analyst", as_of=as_of) == []


def test_raw_search_matches_multi_term_queries_in_any_order(store: SQLiteStore) -> None:
    """Regression: the probe used to bind the whole query as ONE contiguous
    LIKE substring, so any multi-term existence probe returned 0 hits and the
    nightly analyst fell back to production replay. Every content word must
    match somewhere in the unit, in any order."""
    pairs = store.raw_search_units_for_user("sam", "analyst senior data")
    assert [u.id for u, _ in pairs] == ["u-old"]
    # Word order and interleaving stopwords must not matter.
    assert store.raw_search_units_for_user("sam", "is Sam a data analyst?")


def test_raw_search_surfaces_units_missing_some_probe_terms(store: SQLiteStore) -> None:
    """Regression: a natural-language existence probe carrying an extra token
    the unit doesn't contain ("Acme Foundation") used to AND every term together and
    return 0, hiding a fact that is definitely present. Overlap semantics: the
    unit still surfaces because its salient terms match."""
    pairs = store.raw_search_units_for_user("sam", "analyst at Acme Foundation")
    assert [u.id for u, _ in pairs] == ["u-old"]


def test_raw_search_ranks_by_term_overlap(store: SQLiteStore) -> None:
    """Units matching more of the probe's content terms rank ahead of units
    matching fewer — ranked overlap, not a conjunction."""
    pairs = store.raw_search_units_for_user("sam", "senior data analyst Springfield")
    # u-old matches senior/data/analyst (3); u-new matches Springfield (1).
    assert [u.id for u, _ in pairs] == ["u-old", "u-new"]


def test_raw_search_matches_inflected_terms_via_lemma(store: SQLiteStore) -> None:
    """A plural/inflected probe term ("analysts") must still find the singular
    surface form in content — the lemma is accepted as an alternative."""
    pairs = store.raw_search_units_for_user("sam", "senior data analysts")
    assert [u.id for u, _ in pairs] == ["u-old"]


def test_get_unit_for_user_scopes(store: SQLiteStore) -> None:
    assert store.get_unit_for_user("sam", "u-new") is not None
    # Other user's unit is invisible under sam's scope.
    assert store.get_unit_for_user("sam", "u-other") is None


# ---------------------------------------------------------------------------
# MemoryFeedbackStore
# ---------------------------------------------------------------------------


def test_memory_feedback_record_and_ops(tmp_path: Path) -> None:
    fb = MemoryFeedbackStore(tmp_path / "fb.db")
    fid = fb.record(
        user_id="sam",
        day="2026-07-31",
        summary="two duplicate units",
        findings=[{"kind": "duplicate", "unit_ids": ["a", "b"]}],
        ops=[
            {"op_type": "tombstone", "target": {"unit_id": "a"}, "reason": "exact dup of b"},
            {"op_type": "merge", "target": {"canonical_id": "b", "duplicate_ids": ["a"]}, "reason": "same fact"},
        ],
    )
    ops = fb.list_ops(fid)
    assert [o.op_type for o in ops] == ["tombstone", "merge"]
    assert all(o.status == "pending" for o in ops)

    fb.mark_op(ops[0].id, status="applied", result="valid_until set")
    assert fb.get_op(ops[0].id).status == "applied"


def test_memory_feedback_rejects_unknown_op(tmp_path: Path) -> None:
    fb = MemoryFeedbackStore(tmp_path / "fb.db")
    with pytest.raises(ValueError, match="unknown op_type"):
        fb.record(user_id="u", day="2026-07-31", summary="x", findings=[], ops=[{"op_type": "delete"}])


def test_memory_feedback_rerun_replaces_pending_keeps_decided(tmp_path: Path) -> None:
    fb = MemoryFeedbackStore(tmp_path / "fb.db")
    fid1 = fb.record(
        user_id="u", day="2026-07-31", summary="v1", findings=[],
        ops=[{"op_type": "tombstone", "target": {"unit_id": "a"}, "reason": "r"},
             {"op_type": "tombstone", "target": {"unit_id": "b"}, "reason": "r"}],
    )
    ops1 = fb.list_ops(fid1)
    fb.mark_op(ops1[0].id, status="applied")  # decide one; the other stays pending

    # Re-run the same day: pending op dropped, applied op preserved as audit.
    fb.record(user_id="u", day="2026-07-31", summary="v2", findings=[],
              ops=[{"op_type": "tombstone", "target": {"unit_id": "c"}, "reason": "r"}])
    records = fb.list_feedback("u")
    # The applied op's parent record survives + the new record = 2 records.
    all_ops = [op for rec in records for op in fb.list_ops(rec.id)]
    statuses = sorted(op.status for op in all_ops)
    assert statuses == ["applied", "pending"]  # dropped the old pending, kept applied + added new pending
    assert any(o.target.get("unit_id") == "c" for o in all_ops)
    assert any(o.target.get("unit_id") == "a" and o.status == "applied" for o in all_ops)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
async def client(store: SQLiteStore, tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    fb = MemoryFeedbackStore(tmp_path / "fb.db")
    async with await _client(_app(store, fb)) as c:
        yield c


async def test_units_endpoint_requires_auth(store: SQLiteStore, tmp_path: Path) -> None:
    fb = MemoryFeedbackStore(tmp_path / "fb.db")
    async with await _client(_app(store, fb), auth=False) as c:
        assert (await c.get("/v1/users/sam/units")).status_code == 401


async def test_units_endpoint_returns_day(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/users/sam/units", params={"day": "2026-07-31"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["units"][0]["id"] == "u-new"
    assert body["units"][0]["agent_id"] == "sam/_user"


async def test_units_endpoint_bad_day(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/users/sam/units", params={"day": "31-07-2026"})).status_code == 400


async def test_memory_search_raw_mode_and_as_of(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/users/sam/memory-search", params={"query": "analyst", "mode": "raw"})
    assert resp.status_code == 200
    assert resp.json()["count"] == 1

    early = "2026-07-30T08:00:00+00:00"
    resp = await client.get(
        "/v1/users/sam/memory-search", params={"query": "analyst", "mode": "raw", "as_of": early}
    )
    assert resp.json()["count"] == 0


async def test_memory_search_raw_mode_multi_term(client: httpx.AsyncClient) -> None:
    """The analyst-style multi-term existence probe must surface a present fact
    even when the probe carries an extra token ("status") the unit does not
    contain — overlap ranking, not conjunction."""
    resp = await client.get(
        "/v1/users/sam/memory-search",
        params={"query": "Sam senior data analyst status", "mode": "raw"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1  # "status" absent, but the analyst fact still surfaces
    assert body["results"][0]["id"] == "u-old"  # best overlap ranks first

    resp = await client.get(
        "/v1/users/sam/memory-search",
        params={"query": "Sam senior data analyst", "mode": "raw"},
    )
    assert resp.json()["count"] >= 1


async def test_memory_search_production_mode_returns_shape(client: httpx.AsyncClient) -> None:
    resp = await client.get(
        "/v1/users/sam/memory-search",
        params={"query": "Springfield roles", "mode": "production", "slug": "scout"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "production"
    assert body["scope"] == "sam/scout"
    assert "counterfactual-current-index" in body["caveat"]


async def test_memory_search_validation(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/users/sam/memory-search", params={"query": ""})).status_code == 400
    assert (await client.get(
        "/v1/users/sam/memory-search", params={"query": "x", "mode": "bogus"}
    )).status_code == 400
    assert (await client.get(
        "/v1/users/sam/memory-search", params={"query": "x", "as_of": "not-a-date"}
    )).status_code == 400


async def test_memory_feedback_post_then_get(client: httpx.AsyncClient) -> None:
    post = await client.post(
        "/v1/users/sam/memory-feedback",
        json={
            "day": "2026-07-31",
            "summary": "one stale unit",
            "findings": [{"kind": "stale", "unit_id": "u-old"}],
            "ops": [{"op_type": "correct", "target": {"unit_id": "u-old", "new_content": "..."}, "reason": "outdated"}],
        },
    )
    assert post.status_code == 200
    assert isinstance(post.json()["id"], int)

    got = await client.get("/v1/users/sam/memory-feedback")
    assert got.status_code == 200
    records = got.json()["records"]
    assert len(records) == 1
    assert records[0]["day"] == "2026-07-31"
    assert records[0]["ops"][0]["op_type"] == "correct"
    assert records[0]["ops"][0]["status"] == "pending"


async def test_memory_feedback_post_rejects_bad_op(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/v1/users/sam/memory-feedback",
        json={"day": "2026-07-31", "summary": "x", "ops": [{"op_type": "nuke"}]},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Auto-drain of the safe-class op queue on POST (opt-in)
# ---------------------------------------------------------------------------


async def test_post_auto_applies_safe_ops_when_enabled(tmp_path: Path) -> None:
    """Regression: the nightly op queue was never drained automatically — the
    only apply path was a manual dashboard click, so analyst-proposed reversible
    ops sat ``pending`` forever and contradicted memory stayed live. With
    ``auto_apply_safe_ops`` on, the reversible safe classes (tombstone/merge)
    are applied the moment the analyst POSTs the record; ``correct`` stays
    pending for human review."""
    import uuid

    at = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)
    dup_id = str(uuid.uuid4())
    canonical_id = str(uuid.uuid4())
    stale_id = str(uuid.uuid4())
    correct_id = str(uuid.uuid4())
    store = SQLiteStore(tmp_path / "mem.db")
    for uid, content in [
        (dup_id, "Palace Court unit already applied"),
        (canonical_id, "canonical Palace Court unit"),
        (stale_id, "Gloucester Terrace permanently excluded"),
        (correct_id, "Sam is a junior analyst"),
    ]:
        store.add(_unit(content, asserted=at, uid=uid), "sam/scout")
    fb = MemoryFeedbackStore(tmp_path / "fb.db")

    async with await _client(_app(store, fb, auto_apply_safe_ops=True)) as c:
        resp = await c.post(
            "/v1/users/sam/memory-feedback",
            json={
                "day": "2026-07-31",
                "summary": "drain the queue",
                "findings": [],
                "ops": [
                    {"op_type": "tombstone", "target": {"unit_id": stale_id}, "reason": "stale exclusion"},
                    {"op_type": "merge",
                     "target": {"canonical_id": canonical_id, "duplicate_ids": [dup_id]}, "reason": "dup"},
                    {"op_type": "correct",
                     "target": {"unit_id": correct_id, "new_content": "Sam is a senior analyst"}, "reason": "title"},
                ],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["auto_applied"] == 2  # tombstone + merge
    assert body["auto_rejected"] == 0

    # The reversible ops actually mutated memory...
    active = {u.id for u, _ in store.list_units_for_user("sam")}
    assert stale_id not in active  # tombstoned
    assert dup_id not in active    # merged away
    assert canonical_id in active  # canonical untouched
    # ...and the ambiguous `correct` op was left for a human.
    statuses = {o.op_type: o.status for rec in fb.list_feedback("sam") for o in fb.list_ops(rec.id)}
    assert statuses == {"tombstone": "applied", "merge": "applied", "correct": "pending"}


async def test_post_leaves_ops_pending_when_auto_apply_disabled(tmp_path: Path) -> None:
    """Default (opt-out) behaviour is unchanged: ops stay pending for the
    dashboard, and the response carries no auto-apply counters."""
    import uuid

    at = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)
    stale_id = str(uuid.uuid4())
    store = SQLiteStore(tmp_path / "mem.db")
    store.add(_unit("stale", asserted=at, uid=stale_id), "sam/scout")
    fb = MemoryFeedbackStore(tmp_path / "fb.db")
    async with await _client(_app(store, fb)) as c:  # default: disabled
        resp = await c.post(
            "/v1/users/sam/memory-feedback",
            json={"day": "2026-07-31", "summary": "x", "findings": [],
                  "ops": [{"op_type": "tombstone", "target": {"unit_id": stale_id}, "reason": "stale"}]},
        )
    assert resp.status_code == 200
    assert "auto_applied" not in resp.json()
    assert {u.id for u, _ in store.list_units_for_user("sam")} == {stale_id}  # untouched
    statuses = {o.status for rec in fb.list_feedback("sam") for o in fb.list_ops(rec.id)}
    assert statuses == {"pending"}


async def test_post_auto_apply_rejects_malformed_safe_op(tmp_path: Path) -> None:
    """The auto-drain reuses apply_all_pending's safety envelope: a malformed
    op (truncated uuid) is auto-rejected, never guessed at or left dangling."""
    import uuid

    at = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)
    good_id = str(uuid.uuid4())
    store = SQLiteStore(tmp_path / "mem.db")
    store.add(_unit("good", asserted=at, uid=good_id), "sam/scout")
    fb = MemoryFeedbackStore(tmp_path / "fb.db")
    async with await _client(_app(store, fb, auto_apply_safe_ops=True)) as c:
        resp = await c.post(
            "/v1/users/sam/memory-feedback",
            json={"day": "2026-07-31", "summary": "x", "findings": [],
                  "ops": [{"op_type": "tombstone", "target": {"unit_id": good_id[:8]}, "reason": "truncated"}]},
        )
    body = resp.json()
    assert body["auto_applied"] == 0
    assert body["auto_rejected"] == 1
    assert {u.id for u, _ in store.list_units_for_user("sam")} == {good_id}  # untouched
    statuses = {o.status for rec in fb.list_feedback("sam") for o in fb.list_ops(rec.id)}
    assert statuses == {"dismissed"}
