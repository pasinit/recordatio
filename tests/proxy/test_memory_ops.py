"""Tests for applying approved memory-feedback ops.

Unit ids are real uuid4 strings (as in production) — the apply path now
requires COMPLETE uuids and treats truncated/unknown ids as invalid
(auto-rejected), so short test ids like "u1" would (correctly) be refused.
"""
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit
from tpasini.recordatio.proxy.feedback_endpoints import register_feedback_routes
from tpasini.recordatio.proxy.memory_ops import InvalidOpError, apply_memory_op
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore

_AUTH = {"authorization": "Bearer test-key"}

U1 = str(uuid.uuid4())
U2 = str(uuid.uuid4())
U3 = str(uuid.uuid4())
OTHER = str(uuid.uuid4())


def _unit(uid: str, content: str) -> KnowledgeUnit:
    at = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)
    return KnowledgeUnit(id=uid, type=KnowledgeType.SEMANTIC, content=content,
                         asserted_at=at, valid_from=at, provenance="test")


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteStore:
    s = SQLiteStore(tmp_path / "mem.db")
    s.add(_unit(U1, "Sam is a analyst"), "sam/scout")
    s.add(_unit(U2, "Sam is a analyst (dup)"), "sam/scout")
    s.add(_unit(U3, "Sam lives in X1"), "sam/_user")
    s.add(_unit(OTHER, "Tommaso wants a flat"), "tommaso/helper")
    return s


def _active_ids(store: SQLiteStore, user: str) -> set[str]:
    return {u.id for u, _ in store.list_units_for_user(user)}


# ---------------------------------------------------------------------------
# apply_memory_op
# ---------------------------------------------------------------------------


def test_tombstone_sets_valid_until(store: SQLiteStore) -> None:
    res = apply_memory_op(store, "sam", "tombstone", {"unit_id": U2})
    assert f"tombstoned {U2}" in res
    assert U2 not in _active_ids(store, "sam")
    assert {U1, U3} <= _active_ids(store, "sam")


def test_merge_tombstones_duplicates_keeps_canonical(store: SQLiteStore) -> None:
    res = apply_memory_op(store, "sam", "merge", {"canonical_id": U1, "duplicate_ids": [U2]})
    assert "merged 1 duplicate" in res
    active = _active_ids(store, "sam")
    assert U1 in active and U2 not in active


def test_correct_supersedes(store: SQLiteStore) -> None:
    target = {"unit_id": U3, "new_content": "Sam lives in Riverside X1"}
    res = apply_memory_op(store, "sam", "correct", target)
    assert f"superseded {U3}" in res
    active = store.list_units_for_user("sam")
    contents = {u.content for u, _ in active}
    assert "Sam lives in Riverside X1" in contents
    assert U3 not in {u.id for u, _ in active}  # old tombstoned
    new = next(u for u, _ in active if u.content == "Sam lives in Riverside X1")
    assert new.provenance == f"correction of {U3}"


def _corrections_of(store: SQLiteStore, user: str, source_id: str) -> list[KnowledgeUnit]:
    return [u for u, _ in store.list_units_for_user(user) if u.provenance == f"correction of {source_id}"]


def test_correcting_same_source_twice_leaves_one_active_descendant(store: SQLiteStore) -> None:
    """The analyst re-corrects the ORIGINAL source id each night. Two such
    corrections must not accumulate sibling replacements — exactly one active
    unit may descend from a given source (the duplicate-cluster regression)."""
    first = {"unit_id": U3, "new_content": "Sam lives in Riverside X1"}
    apply_memory_op(store, "sam", "correct", first)
    second = {"unit_id": U3, "new_content": "Sam lives in Hillcrest X1"}
    apply_memory_op(store, "sam", "correct", second)

    live = _corrections_of(store, "sam", U3)
    assert len(live) == 1
    assert live[0].content == "Sam lives in Hillcrest X1"
    assert U3 not in _active_ids(store, "sam")  # original stays superseded


def test_correcting_same_source_many_times_leaves_one_active_descendant(store: SQLiteStore) -> None:
    for i in range(5):
        apply_memory_op(store, "sam", "correct", {"unit_id": U3, "new_content": f"X1 repricing #{i}"})
    live = _corrections_of(store, "sam", U3)
    assert len(live) == 1
    assert live[0].content == "X1 repricing #4"


def test_op_refuses_out_of_scope_unit(store: SQLiteStore) -> None:
    with pytest.raises(InvalidOpError, match="not found"):
        apply_memory_op(store, "sam", "tombstone", {"unit_id": OTHER})


def test_op_refuses_incomplete_uuid(store: SQLiteStore) -> None:
    # A truncated id (an 8-char prefix of a real unit) is invalid, not resolved.
    with pytest.raises(InvalidOpError, match="not a complete uuid"):
        apply_memory_op(store, "sam", "tombstone", {"unit_id": U2[:8]})


def test_op_validation_errors(store: SQLiteStore) -> None:
    with pytest.raises(InvalidOpError, match="unknown op_type"):
        apply_memory_op(store, "sam", "nuke", {})
    with pytest.raises(InvalidOpError, match="new_content"):
        apply_memory_op(store, "sam", "correct", {"unit_id": U1})
    with pytest.raises(InvalidOpError, match="duplicate_ids"):
        apply_memory_op(store, "sam", "merge", {"canonical_id": U1})


# ---------------------------------------------------------------------------
# apply/dismiss endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def env(store: SQLiteStore, tmp_path: Path):
    fb = MemoryFeedbackStore(tmp_path / "fb.db")
    app = FastAPI()
    register_feedback_routes(app, get_store=lambda: store, memory_feedback_store=fb, user_label="User")
    return app, fb, store


async def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", headers=_AUTH)


def _seed_op(fb: MemoryFeedbackStore, user: str, op_type: str, target: dict) -> int:
    fid = fb.record(user_id=user, day="2026-07-31", summary="s", findings=[],
                    ops=[{"op_type": op_type, "target": target, "reason": "r"}])
    return fb.list_ops(fid)[0].id


async def test_apply_endpoint_tombstones_and_marks(env) -> None:
    app, fb, store = env
    op_id = _seed_op(fb, "sam", "tombstone", {"unit_id": U2})
    async with await _client(app) as c:
        resp = await c.post(f"/v1/users/sam/memory-feedback/ops/{op_id}/apply")
    assert resp.status_code == 200 and resp.json()["status"] == "applied"
    assert U2 not in _active_ids(store, "sam")
    assert fb.get_op(op_id).status == "applied"


async def test_apply_is_idempotent_guarded(env) -> None:
    app, fb, store = env
    op_id = _seed_op(fb, "sam", "tombstone", {"unit_id": U2})
    async with await _client(app) as c:
        assert (await c.post(f"/v1/users/sam/memory-feedback/ops/{op_id}/apply")).status_code == 200
        again = await c.post(f"/v1/users/sam/memory-feedback/ops/{op_id}/apply")
    assert again.status_code == 409  # already applied


async def test_apply_invalid_op_auto_rejects(env) -> None:
    app, fb, store = env
    # An incomplete-uuid target is invalid → auto-rejected (dismissed), not left pending.
    op_id = _seed_op(fb, "sam", "tombstone", {"unit_id": U2[:8]})
    async with await _client(app) as c:
        resp = await c.post(f"/v1/users/sam/memory-feedback/ops/{op_id}/apply")
    assert resp.status_code == 200 and resp.json()["status"] == "rejected"
    assert fb.get_op(op_id).status == "dismissed"
    assert "auto-rejected" in fb.get_op(op_id).decided_result


async def test_apply_scoped_to_owner(env) -> None:
    app, fb, store = env
    op_id = _seed_op(fb, "sam", "tombstone", {"unit_id": U2})
    async with await _client(app) as c:
        resp = await c.post(f"/v1/users/tommaso/memory-feedback/ops/{op_id}/apply")
    assert resp.status_code == 404
    assert fb.get_op(op_id).status == "pending"


async def test_dismiss_endpoint(env) -> None:
    app, fb, store = env
    op_id = _seed_op(fb, "sam", "tombstone", {"unit_id": U2})
    async with await _client(app) as c:
        resp = await c.post(f"/v1/users/sam/memory-feedback/ops/{op_id}/dismiss",
                            json={"reason": "false positive"})
    assert resp.status_code == 200 and resp.json()["status"] == "dismissed"
    assert fb.get_op(op_id).status == "dismissed"
    assert U2 in _active_ids(store, "sam")  # nothing mutated


async def test_apply_unknown_op_id_404(env) -> None:
    app, fb, store = env
    async with await _client(app) as c:
        assert (await c.post("/v1/users/sam/memory-feedback/ops/999/apply")).status_code == 404


async def test_apply_requires_auth(env) -> None:
    app, fb, store = env
    op_id = _seed_op(fb, "sam", "tombstone", {"unit_id": U2})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.post(f"/v1/users/sam/memory-feedback/ops/{op_id}/apply")).status_code == 401
