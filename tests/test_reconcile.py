"""Tests for the judge-driven memory reconcile (candidates → verdicts → finish).

Verdicts are written by the tests (the judge is an external agent), including
wrong or overreaching ones, so the tests pin down what the deterministic
verification lets through. All fixtures are synthetic.
"""
import asyncio
import json
import sqlite3
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import EntityMention, KnowledgeType, KnowledgeUnit, Relation
from tpasini.recordatio.proxy import embeddings, network_guard
from tpasini.recordatio.proxy.reconcile_routes import register_reconcile_routes
from tpasini.recordatio.reconcile import ReconcileError, ReconcileSettings, RunFinishedError, build_service
from tpasini.recordatio.reconcile.harness import run_harness
from tpasini.recordatio.reconcile.judge import JudgeFact, JudgeVerdict, verify_verdict
from tpasini.recordatio.reconcile.service import CORRECT_DISMISS_REASON, ReconcileService
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore
from tpasini.recordatio.storage.reconcile_log import ReconcileLog

USER_SCOPE = "sam/_user"
AGENT_SCOPE = "sam/scout"
SCOPE = USER_SCOPE
RUN = "night-1"
SAME_HOME = [("Sam lives in Springfield", "Sam lives in Springfield", "same")]
T0 = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)


def _unit(
    content: str,
    *,
    at: datetime = T0,
    kind: KnowledgeType = KnowledgeType.SEMANTIC,
    uid: str | None = None,
    provenance: str = "test",
    valid_until: datetime | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=uid or str(uuid.uuid4()),
        type=kind,
        content=content,
        asserted_at=at,
        valid_from=at,
        valid_until=valid_until,
        provenance=provenance,
    )


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "mem.db")


@pytest.fixture()
def feedback(tmp_path: Path) -> MemoryFeedbackStore:
    return MemoryFeedbackStore(tmp_path / "memory_feedback.db")


def _settings(**overrides: object) -> ReconcileSettings:
    return ReconcileSettings(snapshot_keep=2, **overrides)


def _service(store: SQLiteStore, feedback: MemoryFeedbackStore | None = None, **cfg: object) -> ReconcileService:
    return build_service(store, settings=_settings(**cfg), feedback=feedback)


def _units(store: SQLiteStore) -> dict[str, str]:
    rows = store._conn.execute("SELECT id, content FROM knowledge_units").fetchall()
    return {r["id"]: r["content"] for r in rows}


def _pair(svc: ReconcileService, x: KnowledgeUnit, y: KnowledgeUnit, *, user: str = "sam", run: str = RUN) -> dict:
    """The offered pair holding both units (fails when it is not offered)."""
    for p in svc.candidates(user, run, limit=25)["pairs"]:
        if {p["a"]["id"], p["b"]["id"]} == {x.id, y.id}:
            return p
    raise AssertionError(f"pair {x.content!r} / {y.content!r} not offered")


def verdict(pair: dict, relation: str, *, keep: KnowledgeUnit | None = None, drop: KnowledgeUnit | None = None,
            facts: list[tuple[str, str, str]] = ()) -> dict:
    label = {pair["a"]["id"]: "A", pair["b"]["id"]: "B"}
    return {
        "relation": relation,
        "keep": label[keep.id] if keep else None,
        "drop": label[drop.id] if drop else None,
        "facts": [{"dropped_fact": d, "kept_span": k, "status": s} for d, k, s in facts],
        "evidence": "test",
    }


def _judge(svc: ReconcileService, keep: KnowledgeUnit, drop: KnowledgeUnit, relation: str,
           facts: list[tuple[str, str, str]], **kw: object) -> dict:
    p = _pair(svc, keep, drop)
    return svc.apply("sam", RUN, p["pair_token"], verdict(p, relation, keep=keep, drop=drop, facts=facts), **kw)


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


def test_duplicate_subsumed_is_deleted_and_keeper_verbatim(store: SQLiteStore) -> None:
    keep = _unit("Sam lives in Springfield and works as a data analyst")
    drop = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    store.add(keep, SCOPE)
    store.add(drop, SCOPE)
    svc = _service(store)

    out = _judge(svc, keep, drop, "duplicate_subsumed", SAME_HOME)

    assert out["outcome"] == "deleted" and out["dropped_unit_id"] == drop.id
    assert _units(store) == {keep.id: keep.content}


def test_partial_overlap_with_extra_fact_is_kept_and_not_reoffered(store: SQLiteStore) -> None:
    keep = _unit("Sam lives in Springfield")
    other = _unit("Sam lives in Springfield near the river since 2019", at=T0 + timedelta(days=1))
    store.add(keep, SCOPE)
    store.add(other, SCOPE)
    svc = _service(store)

    # An overreaching verdict: claims subsumption but cannot quote the extra fact.
    out = _judge(svc, keep, other, "duplicate_subsumed", SAME_HOME)

    assert out["outcome"] == "kept" and "not accounted for" in out["reason"]
    assert set(_units(store)) == {keep.id, other.id}
    assert svc.candidates("sam", "night-2", limit=25)["pairs"] == []  # cached as kept


def test_contradiction_with_newer_claim_deletes_older(store: SQLiteStore) -> None:
    old = _unit("Sam lives in Springfield")
    new = _unit("Sam lives in Shelbyville", at=T0 + timedelta(days=30))
    store.add(old, SCOPE)
    store.add(new, SCOPE)
    svc = _service(store)

    out = _judge(svc, new, old, "supersedes", [("Sam lives in Springfield", "Sam lives in Shelbyville", "replaced")])

    assert out["outcome"] == "deleted"
    assert _units(store) == {new.id: new.content}


def test_contradiction_where_kept_is_actually_older_deletes_nothing(store: SQLiteStore) -> None:
    old = _unit("Sam lives in Springfield")
    new = _unit("Sam lives in Shelbyville", at=T0 + timedelta(days=30))
    store.add(old, SCOPE)
    store.add(new, SCOPE)
    svc = _service(store)

    out = _judge(svc, old, new, "supersedes", [("Sam lives in Shelbyville", "Sam lives in Springfield", "replaced")])

    assert out["outcome"] == "kept" and "not newer" in out["reason"]
    assert set(_units(store)) == {old.id, new.id}


@pytest.mark.parametrize("relation", ["unrelated", "compatible", "unsure"])
def test_non_deleting_relations_keep_both(store: SQLiteStore, relation: str) -> None:
    a = _unit("Sam likes hiking in Springfield")
    b = _unit("Sam likes cycling in Springfield", at=T0 + timedelta(days=1))
    store.add(a, SCOPE)
    store.add(b, SCOPE)
    svc = _service(store)
    p = _pair(svc, a, b)

    out = svc.apply("sam", RUN, p["pair_token"], verdict(p, relation))

    assert out["outcome"] == "kept" and out["reason"] == relation
    assert set(_units(store)) == {a.id, b.id}


def test_episodic_same_text_on_different_days_is_kept(store: SQLiteStore) -> None:
    a = _unit("Sam went to the gym on Monday morning", kind=KnowledgeType.EPISODIC)
    b = _unit("Sam went to the gym", kind=KnowledgeType.EPISODIC, at=T0 + timedelta(days=3))
    store.add(a, SCOPE)
    store.add(b, SCOPE)
    svc = _service(store)

    out = _judge(svc, a, b, "duplicate_subsumed", [("Sam went to the gym", "Sam went to the gym", "same")])

    assert out["outcome"] == "kept" and "episodic" in out["reason"]


# ---------------------------------------------------------------------------
# Tokens: idempotency, staleness, users
# ---------------------------------------------------------------------------


def test_apply_replay_returns_original_outcome_without_second_effect(tmp_path: Path, store: SQLiteStore) -> None:
    keep = _unit("Sam lives in Springfield and works as a data analyst")
    drop = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    store.add(keep, SCOPE)
    store.add(drop, SCOPE)
    svc = _service(store)
    p = _pair(svc, keep, drop)
    v = verdict(p, "duplicate_subsumed", keep=keep, drop=drop,
                facts=SAME_HOME)

    first = svc.apply("sam", RUN, p["pair_token"], v)
    second = svc.apply("sam", RUN, p["pair_token"], v)
    # Even a different verdict on a replayed token cannot change the outcome.
    third = svc.apply("sam", RUN, p["pair_token"], verdict(p, "unsure"))

    assert first["outcome"] == "deleted" and first["replayed"] is False
    assert second["outcome"] == third["outcome"] == "deleted" and second["replayed"] and third["replayed"]
    assert len(ReconcileLog(tmp_path / "memory_reconcile.db").deletions(RUN)) == 1


def test_stale_pair_token_is_refused(store: SQLiteStore) -> None:
    keep = _unit("Sam lives in Springfield and works as a data analyst")
    drop = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    store.add(keep, SCOPE)
    store.add(drop, SCOPE)
    svc = _service(store)
    p = _pair(svc, keep, drop)
    # The kept unit's text changes after the judge saw it.
    store.add(_unit("Sam lives in Springfield, works remotely", uid=keep.id), SCOPE)

    out = svc.apply("sam", RUN, p["pair_token"], verdict(
        p, "duplicate_subsumed", keep=keep, drop=drop,
        facts=[("Sam lives in Springfield", "Sam lives in Springfield", "same")],
    ))

    assert out["outcome"] == "stale"
    assert drop.id in _units(store)


def test_never_crosses_users(store: SQLiteStore) -> None:
    mine = _unit("Sam lives in Springfield")
    theirs = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    store.add(mine, SCOPE)
    store.add(theirs, "alex/_user")
    svc = _service(store)

    assert svc.candidates("sam", RUN, limit=25)["pairs"] == []
    forged = f"{mine.id}~{theirs.id}~-~0~{'0' * 32}"
    out = svc.apply("sam", RUN, forged, {"relation": "duplicate_subsumed", "keep": "A", "drop": "B", "facts": [
        {"dropped_fact": "Sam lives in Springfield", "kept_span": "Sam lives in Springfield", "status": "same"}]})
    assert out["outcome"] == "stale"
    assert svc.finish("sam", RUN)["deleted"] == 0
    assert set(_units(store)) == {mine.id, theirs.id}


def test_verdict_never_creates_or_rewrites_content(store: SQLiteStore) -> None:
    units = [_unit(f"Sam fact number {i} about Springfield", at=T0 + timedelta(hours=i)) for i in range(4)]
    for u in units:
        store.add(u, SCOPE)
    svc = _service(store)
    before = _units(store)
    p = _pair(svc, units[0], units[1])
    bad = verdict(p, "unsure")
    bad["new_content"] = "Sam moved to Shelbyville"

    with pytest.raises(ReconcileError):
        svc.apply("sam", RUN, p["pair_token"], bad)
    _judge(svc, units[1], units[0], "supersedes",
           [("Sam fact number 0 about Springfield", "Sam fact number 1 about Springfield", "replaced")])
    svc.finish("sam", RUN)

    after = _units(store)
    assert len(after) <= len(before)
    assert all(before.get(uid) == content for uid, content in after.items())


# ---------------------------------------------------------------------------
# Cross-scope
# ---------------------------------------------------------------------------


def test_cross_scope_duplicate_keeps_user_copy(store: SQLiteStore) -> None:
    shared = _unit("Sam lives in Springfield and works as a data analyst")
    agent_copy = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    store.add(shared, USER_SCOPE)
    store.add(agent_copy, AGENT_SCOPE)
    svc = _service(store)
    p = _pair(svc, shared, agent_copy)
    assert {p["a"]["scope"], p["b"]["scope"]} == {USER_SCOPE, AGENT_SCOPE}

    out = svc.apply("sam", RUN, p["pair_token"], verdict(
        p, "duplicate_subsumed", keep=shared, drop=agent_copy,
        facts=[("Sam lives in Springfield", "Sam lives in Springfield", "same")],
    ))

    assert out["outcome"] == "deleted"
    assert _units(store) == {shared.id: shared.content}


def test_cross_scope_never_drops_the_user_copy(store: SQLiteStore) -> None:
    shared = _unit("Sam lives in Springfield")
    agent_copy = _unit("Sam lives in Springfield and works as a data analyst", at=T0 + timedelta(days=1))
    store.add(shared, USER_SCOPE)
    store.add(agent_copy, AGENT_SCOPE)
    svc = _service(store)
    p = _pair(svc, shared, agent_copy)

    out = svc.apply("sam", RUN, p["pair_token"], verdict(
        p, "duplicate_subsumed", keep=agent_copy, drop=shared,
        facts=[("Sam lives in Springfield", "Sam lives in Springfield", "same")],
    ))

    assert out["outcome"] == "kept" and "shared copy" in out["reason"]
    assert set(_units(store)) == {shared.id, agent_copy.id}


def test_two_agent_scopes_never_pair(store: SQLiteStore) -> None:
    a = _unit("Sam lives in Springfield")
    b = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    store.add(a, "sam/scout")
    store.add(b, "sam/helper")
    svc = _service(store)
    assert svc.candidates("sam", RUN, limit=25)["pairs"] == []
    assert svc.finish("sam", RUN)["deleted"] == 0


def test_exact_cross_scope_duplicate_resolved_at_finish(store: SQLiteStore) -> None:
    shared = _unit("Sam prefers tea over coffee.")
    agent_copy = _unit("sam prefers tea over coffee", at=T0 + timedelta(days=2))
    same_scope = _unit("Sam prefers tea over coffee", at=T0 + timedelta(days=3))
    store.add(shared, USER_SCOPE)
    store.add(agent_copy, AGENT_SCOPE)
    store.add(same_scope, USER_SCOPE)
    svc = _service(store)

    assert svc.candidates("sam", RUN, limit=25)["pairs"] == []  # exact duplicates need no judge
    summary = svc.finish("sam", RUN)

    assert _units(store) == {shared.id: shared.content}
    assert summary["by_source"] == {"exact-match": 2}


# ---------------------------------------------------------------------------
# Cascade, dry run, snapshot, run lifecycle
# ---------------------------------------------------------------------------


def test_deletion_cascades_and_keeps_shared_entities(store: SQLiteStore) -> None:
    keep = _unit("Sam works at Acme Labs in Springfield")
    drop = _unit("Sam works at Acme Labs", at=T0 + timedelta(days=1))
    store.add(keep, SCOPE)
    store.add(drop, SCOPE)
    acme = Entity(id=str(uuid.uuid4()), label="Acme Labs", asserted_at=T0)
    city = Entity(id=str(uuid.uuid4()), label="Springfield", asserted_at=T0)
    store.add_entity(acme, SCOPE)
    store.add_entity(city, SCOPE)
    for unit in (keep, drop):
        store.add_relation(Relation(
            id="", subject_id=unit.id, predicate="mentions", object_id=acme.id,
            asserted_at=T0, valid_from=T0, valid_until=None, provenance="auto",
        ), SCOPE)
        store.upsert_entity_mention(EntityMention(
            entity_id=acme.id, source_kind="knowledge_unit", source_id=unit.id, asserted_at=T0,
        ), SCOPE)
        store.upsert_item_embedding(
            item_id=unit.id, agent_id=SCOPE, kind="unit", surface_hash=unit.id,
            embedding=embeddings.pack_embedding([0.0] * 4), model="m", asserted_at=T0.isoformat(),
        )
    store.add_relation(Relation(
        id="", subject_id=acme.id, predicate="located_in", object_id=city.id,
        asserted_at=T0, valid_from=T0, valid_until=None, provenance="auto",
    ), SCOPE)
    store.tag_unit(drop.id, store.list_topics()[0].id, SCOPE)
    svc = _service(store)

    _judge(svc, keep, drop, "duplicate_subsumed", [("Sam works at Acme Labs", "Sam works at Acme Labs", "same")])

    c = store._conn
    assert _units(store) == {keep.id: keep.content}
    for sql in (
        "SELECT COUNT(*) FROM memfts.memento_fts WHERE item_id = ?",
        "SELECT COUNT(*) FROM mememb.item_embeddings WHERE item_id = ?",
        "SELECT COUNT(*) FROM unit_topics WHERE unit_id = ?",
        "SELECT COUNT(*) FROM entity_mentions WHERE source_id = ?",
        "SELECT COUNT(*) FROM relations WHERE subject_id = ? OR object_id = ?",
    ):
        assert c.execute(sql, (drop.id,) * sql.count("?")).fetchone()[0] == 0, sql
    orphans = c.execute(
        "SELECT COUNT(*) FROM memfts.memento_fts f WHERE f.kind = 'unit' "
        "AND NOT EXISTS (SELECT 1 FROM knowledge_units k WHERE k.id = f.item_id)"
    ).fetchone()[0]
    assert orphans == 0
    assert store.get_entity_by_id(acme.id, SCOPE) is not None
    assert store.get_entity_by_id(city.id, SCOPE) is not None
    assert [r.object_id for r in store.get_relations(keep.id, SCOPE)] == [acme.id]
    assert [r.object_id for r in store.get_relations(acme.id, SCOPE)] == [city.id]
    assert store.find_entity_mentions_by_source("knowledge_unit", keep.id, SCOPE)


def _dump(path: Path) -> list[tuple]:
    conn = sqlite3.connect(str(path))
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        return [(t, sorted(map(tuple, conn.execute(f"SELECT * FROM {t}").fetchall()))) for t in tables]  # noqa: S608
    finally:
        conn.close()


def test_dry_run_mutates_nothing(tmp_path: Path, store: SQLiteStore, feedback: MemoryFeedbackStore) -> None:
    keep = _unit("Sam lives in Springfield and works as a data analyst")
    drop = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    exact = _unit("sam lives in springfield and works as a data analyst", at=T0 + timedelta(days=2))
    for u in (keep, drop, exact):
        store.add(u, SCOPE)
    feedback.record(user_id="sam", day="2026-09-02", summary="s", findings=[], ops=[
        {"op_type": "correct", "target": {"unit_id": drop.id, "new_content": "x"}, "reason": "r"},
    ])
    svc = _service(store, feedback)
    before = [_dump(tmp_path / n) for n in ("mem.db", "memory_feedback.db", "memory_reconcile.db")]

    out = _judge(svc, keep, drop, "duplicate_subsumed",
                 [("Sam lives in Springfield", "Sam lives in Springfield", "same")], dry_run=True)
    remaining = svc.candidates("sam", RUN, limit=25)["pairs"]
    preview = svc.finish("sam", RUN, dry_run=True)

    assert out["outcome"] == "would_delete"
    assert remaining == []  # the previewed pair is not offered again in this run
    assert [d["unit_id"] for d in preview["would_delete"]] == [exact.id]
    assert [o["result"] for o in preview["ops"]] == [CORRECT_DISMISS_REASON]
    assert [_dump(tmp_path / n) for n in ("mem.db", "memory_feedback.db", "memory_reconcile.db")] == before
    assert not (tmp_path / "reconcile-snapshots").exists()


def test_snapshot_taken_before_first_mutation_and_retained(tmp_path: Path, store: SQLiteStore) -> None:
    svc = _service(store)
    for n in range(3):
        keep = _unit(f"Sam owns {n} bikes and a car")
        drop = _unit(f"Sam owns {n} bikes", at=T0 + timedelta(days=1))
        store.add(keep, SCOPE)
        store.add(drop, SCOPE)
        run = f"night-{n}"
        p = _pair(svc, keep, drop, run=run)
        svc.apply("sam", run, p["pair_token"], verdict(
            p, "duplicate_subsumed", keep=keep, drop=drop, facts=[(drop.content, drop.content, "same")]))
        summary = svc.finish("sam", run)
        snap = Path(summary["snapshot"])
        snapped = sqlite3.connect(str(snap / "mem.db"))
        ids = {r[0] for r in snapped.execute("SELECT id FROM knowledge_units")}
        snapped.close()
        assert {keep.id, drop.id} <= ids  # the snapshot predates the deletion
        assert (snap / "mem.db.emb").exists()
        assert summary["vacuumed"] is True
    assert len(list((tmp_path / "reconcile-snapshots").iterdir())) == 2  # snapshot_keep=2


def test_finished_run_refuses_more_verdicts_and_replays_summary(store: SQLiteStore) -> None:
    a = _unit("Sam likes hiking in Springfield")
    b = _unit("Sam likes cycling in Springfield", at=T0 + timedelta(days=1))
    store.add(a, SCOPE)
    store.add(b, SCOPE)
    svc = _service(store)
    p = _pair(svc, a, b)
    first = svc.finish("sam", RUN)

    assert svc.finish("sam", RUN) == {**first, "replayed": True}
    assert svc.candidates("sam", RUN, limit=5)["finished"] is True
    with pytest.raises(RunFinishedError):
        svc.apply("sam", RUN, p["pair_token"], verdict(p, "unsure"))
    assert first["snapshot"] is None and first["vacuumed"] is False


def test_budget_caps_pairs_per_run(store: SQLiteStore) -> None:
    for i in range(6):
        store.add(_unit(f"Sam likes Springfield park number {i}"), SCOPE)
    svc = _service(store, max_pairs_per_run=3)
    batch = svc.candidates("sam", RUN, limit=25)
    assert len(batch["pairs"]) == 3 and batch["budget_left"] == 0 and batch["more"] is True
    for p in batch["pairs"]:
        svc.apply("sam", RUN, p["pair_token"], verdict(p, "compatible"))
    assert svc.candidates("sam", RUN, limit=25)["pairs"] == []
    assert svc.candidates("sam", "night-2", limit=25)["pairs"]  # the next run continues


def test_embedding_signal_proposes_pairs(store: SQLiteStore) -> None:
    a = _unit("Sam resides in Springfield")
    b = _unit("Sam's home town is Springfield", at=T0 + timedelta(days=1))
    store.add(a, SCOPE)
    store.add(b, SCOPE)
    vec = embeddings.pack_embedding([1.0, 0.0, 0.0, 0.0])
    for u in (a, b):
        store.upsert_item_embedding(item_id=u.id, agent_id=SCOPE, kind="unit", surface_hash=u.id,
                                    embedding=vec, model="m", asserted_at=T0.isoformat())
    svc = _service(store, fts_neighbours=0)
    pairs = svc.candidates("sam", RUN, limit=25)["pairs"]
    assert [(p["signal"], {p["a"]["id"], p["b"]["id"]}) for p in pairs] == [("embedding", {a.id, b.id})]


# ---------------------------------------------------------------------------
# Backlog of memory-feedback ops + legacy rows
# ---------------------------------------------------------------------------


def _ops(feedback: MemoryFeedbackStore) -> dict[str, tuple[str, str | None]]:
    return {
        op.reason: (op.status, op.decided_result)
        for rec in feedback.list_feedback("sam", limit=100)
        for op in feedback.list_ops(rec.id)
    }


def test_backlog_recheck(store: SQLiteStore, feedback: MemoryFeedbackStore) -> None:
    old = _unit("Sam lives in Springfield")
    new = _unit("Sam lives in Shelbyville", at=T0 + timedelta(days=30))
    canonical = _unit("Sam drives a blue hatchback to work")
    subsumed = _unit("Sam drives a blue hatchback", at=T0 + timedelta(days=1))
    richer = _unit("Sam drives a blue hatchback bought in 2021", at=T0 + timedelta(days=2))
    for u in (old, new, canonical, subsumed, richer):
        store.add(u, SCOPE)
    feedback.record(user_id="sam", day="2026-09-02", summary="s", findings=[], ops=[
        {"op_type": "tombstone", "target": {"unit_id": old.id}, "reason": "tomb-pass"},
        {"op_type": "merge", "target": {"canonical_id": canonical.id, "duplicate_ids": [subsumed.id]},
         "reason": "merge-pass"},
        {"op_type": "merge", "target": {"canonical_id": canonical.id, "duplicate_ids": [richer.id]},
         "reason": "merge-fail"},
        {"op_type": "correct", "target": {"unit_id": new.id, "new_content": "Sam moved"}, "reason": "correct"},
        {"op_type": "tombstone", "target": {"unit_id": new.id[:8]}, "reason": "truncated"},
    ])
    svc = _service(store, feedback)
    batch = svc.candidates("sam", RUN, limit=25)["pairs"]
    op_pairs = [p for p in batch if p["kind"] == "op"]
    assert op_pairs and all(p["may_drop_only"] for p in op_pairs)
    for p in batch:
        ids = {p["a"]["id"], p["b"]["id"]}
        if ids == {old.id, new.id}:
            v = verdict(p, "supersedes", keep=new, drop=old,
                        facts=[("Sam lives in Springfield", "Sam lives in Shelbyville", "replaced")])
        elif ids == {canonical.id, subsumed.id}:
            v = verdict(p, "duplicate_subsumed", keep=canonical, drop=subsumed,
                        facts=[("Sam drives a blue hatchback", "Sam drives a blue hatchback", "same")])
        else:
            v = verdict(p, "compatible")
        svc.apply("sam", RUN, p["pair_token"], v)

    summary = svc.finish("sam", RUN)

    ops = _ops(feedback)
    assert ops["tomb-pass"][0] == "applied"
    assert ops["merge-pass"][0] == "applied"
    assert ops["merge-fail"][0] == "dismissed" and "compatible" in ops["merge-fail"][1]
    assert ops["correct"] == ("dismissed", CORRECT_DISMISS_REASON)
    assert ops["truncated"][0] == "dismissed" and "auto-rejected" in ops["truncated"][1]
    assert set(_units(store)) == {new.id, canonical.id, richer.id}
    assert (summary["ops_applied"], summary["ops_dismissed"]) == (2, 3)


def test_op_pair_may_only_drop_its_target(store: SQLiteStore, feedback: MemoryFeedbackStore) -> None:
    canonical = _unit("Sam drives a blue hatchback")
    dup = _unit("Sam drives a blue hatchback to work", at=T0 + timedelta(days=1))
    store.add(canonical, SCOPE)
    store.add(dup, SCOPE)
    feedback.record(user_id="sam", day="2026-09-02", summary="s", findings=[], ops=[
        {"op_type": "merge", "target": {"canonical_id": canonical.id, "duplicate_ids": [dup.id]}, "reason": "m"},
    ])
    svc = _service(store, feedback)
    p = next(p for p in svc.candidates("sam", RUN, limit=25)["pairs"] if p["kind"] == "op")

    out = svc.apply("sam", RUN, p["pair_token"], verdict(
        p, "duplicate_subsumed", keep=dup, drop=canonical,
        facts=[("Sam drives a blue hatchback", "Sam drives a blue hatchback", "same")]))

    assert out["outcome"] == "kept" and "may only drop" in out["reason"]
    assert set(_units(store)) == {canonical.id, dup.id}


def test_undecided_op_stays_pending(store: SQLiteStore, feedback: MemoryFeedbackStore) -> None:
    canonical = _unit("Sam drives a blue hatchback to work")
    dup = _unit("Sam drives a blue hatchback", at=T0 + timedelta(days=1))
    store.add(canonical, SCOPE)
    store.add(dup, SCOPE)
    feedback.record(user_id="sam", day="2026-09-02", summary="s", findings=[], ops=[
        {"op_type": "merge", "target": {"canonical_id": canonical.id, "duplicate_ids": [dup.id]}, "reason": "m"},
    ])
    svc = _service(store, feedback)
    summary = svc.finish("sam", RUN)  # the judge never answered (failed or ran out of budget)
    assert _ops(feedback)["m"][0] == "pending" and summary["ops_pending"] == 1
    assert set(_units(store)) == {canonical.id, dup.id}


def test_active_unit_never_dropped_for_an_inactive_keeper(store: SQLiteStore, feedback: MemoryFeedbackStore) -> None:
    canonical = _unit("Sam drives a blue hatchback to work", valid_until=T0)  # already tombstoned
    dup = _unit("Sam drives a blue hatchback", at=T0 + timedelta(days=1))
    store.add(canonical, SCOPE)
    store.add(dup, SCOPE)
    feedback.record(user_id="sam", day="2026-09-02", summary="s", findings=[], ops=[
        {"op_type": "merge", "target": {"canonical_id": canonical.id, "duplicate_ids": [dup.id]}, "reason": "m"},
    ])
    svc = _service(store, feedback)
    p = next(p for p in svc.candidates("sam", RUN, limit=25)["pairs"] if p["kind"] == "op")
    out = svc.apply("sam", RUN, p["pair_token"], verdict(
        p, "duplicate_subsumed", keep=canonical, drop=dup,
        facts=[("Sam drives a blue hatchback", "Sam drives a blue hatchback", "same")]))
    assert out["outcome"] == "kept" and "inactive" in out["reason"]
    assert dup.id in _units(store)


def test_legacy_purges_only_provably_superseded(store: SQLiteStore) -> None:
    past = T0 - timedelta(days=10)
    corrected = _unit("Sam lives in Springfield", valid_until=past)
    correction = _unit("Sam lives in Shelbyville", provenance=f"correction of {corrected.id}")
    tombstoned = _unit("Sam likes jazz", valid_until=past)
    expired = _unit("Sam is on holiday this week", kind=KnowledgeType.EPISODIC, valid_until=past)
    for u in (corrected, correction, tombstoned, expired):
        store.add(u, SCOPE)

    summary = _service(store).finish("sam", RUN)

    assert set(_units(store)) == {correction.id, tombstoned.id, expired.id}
    assert summary["legacy_unclassified_kept"] == 2
    assert summary["by_source"] == {"legacy": 1}


# ---------------------------------------------------------------------------
# Verification details
# ---------------------------------------------------------------------------


def test_verify_rejects_non_verbatim_quotes_and_number_mismatch() -> None:
    keep = _unit("Sam pays 1200 GBP rent per month")
    drop = _unit("Sam pays 1100 GBP rent per month")
    bad_quote = JudgeVerdict(relation="duplicate_subsumed", keep="A", drop="B", facts=[
        JudgeFact(dropped_fact="Sam pays rent monthly", kept_span="Sam pays 1200 GBP rent", status="same"),
    ])
    assert verify_verdict(bad_quote, keep, drop)[0] is None
    bad_number = JudgeVerdict(relation="duplicate_subsumed", keep="A", drop="B", facts=[
        JudgeFact(dropped_fact=drop.content, kept_span=keep.content, status="same"),
    ])
    ok, reason = verify_verdict(bad_number, keep, drop)
    assert ok is None and "number" in reason


# ---------------------------------------------------------------------------
# HTTP + network guard
# ---------------------------------------------------------------------------


def test_http_flow(store: SQLiteStore, feedback: MemoryFeedbackStore) -> None:
    keep = _unit("Sam lives in Springfield and works as a data analyst")
    drop = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    store.add(keep, SCOPE)
    store.add(drop, SCOPE)
    svc = _service(store, feedback)
    app = FastAPI()
    register_reconcile_routes(app, get_service=lambda: svc)
    base = "/v1/users/sam/memory-reconcile"

    async def go() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as anon:
            assert (await anon.get(f"{base}/candidates", params={"run_id": RUN})).status_code == 401
        headers = {"authorization": "Bearer test-key"}
        async with httpx.AsyncClient(transport=transport, base_url="http://test", headers=headers) as c:
            assert (await c.get(f"{base}/candidates", params={"run_id": "bad id!"})).status_code == 422
            pair = (await c.get(f"{base}/candidates", params={"run_id": RUN})).json()["pairs"][0]
            v = verdict(pair, "duplicate_subsumed", keep=keep, drop=drop,
                        facts=SAME_HOME)
            token = pair["pair_token"]
            assert (await c.get(f"{base}/verdicts/{token}")).status_code == 404
            dry = (await c.post(f"{base}/verdicts", json={"run_id": RUN, "pair_token": token, "verdict": v,
                                                          "dry_run": True})).json()
            assert dry["outcome"] == "would_delete" and drop.id in _units(store)
            real = (await c.post(f"{base}/verdicts", json={"run_id": RUN, "pair_token": token, "verdict": v})).json()
            assert real["outcome"] == "deleted"
            assert (await c.get(f"{base}/verdicts/{token}")).json()["outcome"] == "deleted"
            fin = (await c.post(f"{base}/runs/{RUN}/finish")).json()
            assert fin["deleted"] == 1
            again = await c.post(f"{base}/verdicts", json={"run_id": RUN, "pair_token": "x~y~-~0~" + "0" * 32,
                                                           "verdict": {"relation": "unsure"}})
            assert again.status_code == 409
            assert (await c.get(f"{base}/runs/{RUN}")).json()["summary"]["deleted"] == 1

    asyncio.run(go())


def test_reconcile_key_may_only_call_reconcile_routes() -> None:
    import hashlib

    key = "reconcile-secret"
    digest = [hashlib.sha256(key.encode()).hexdigest()]
    nets = network_guard.parse_networks(["10.1.0.0/16"])

    def decide(method: str, path: str, client: str = "10.1.0.3", api_key: str | None = key) -> int | None:
        return network_guard.decide(client, method, path, api_key, nets, [], digest)

    assert decide("GET", "/v1/users/sam/memory-reconcile/candidates") is None
    assert decide("POST", "/v1/users/sam/memory-reconcile/verdicts") is None
    assert decide("GET", "/v1/users/sam/memory-reconcile/verdicts/a~b~-~0~" + "0" * 32) is None
    assert decide("POST", "/v1/users/sam/memory-reconcile/runs/night-1/finish") is None
    assert decide("GET", "/v1/users/sam/memory-search") == 403
    assert decide("POST", "/v1/users/sam/memory-feedback/ops/1/apply") == 403
    assert decide("POST", "/v1/chat/completions", client="127.0.0.1") == 403
    assert decide("GET", "/v1/users/sam/memory-reconcile/candidates", api_key="other") == 401


# ---------------------------------------------------------------------------
# Local judge harness / CLI
# ---------------------------------------------------------------------------


def test_harness_runs_the_loop_with_a_judge(store: SQLiteStore) -> None:
    keep = _unit("Sam lives in Springfield and works as a data analyst")
    drop = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    store.add(keep, SCOPE)
    store.add(drop, SCOPE)
    prompts: list[str] = []

    def judge(prompt: str) -> str:
        prompts.append(prompt)
        pairs = json.loads(prompt.split("## Pairs\n\n", 1)[1].split("\n\n## Reply format", 1)[0])
        out = []
        for p in pairs:
            k, d = (keep, drop) if p["a"]["id"] == keep.id or p["b"]["id"] == keep.id else (None, None)
            out.append({"pair_token": p["pair_token"], "verdict": verdict(
                p, "duplicate_subsumed", keep=k, drop=d,
                facts=[("Sam lives in Springfield", "Sam lives in Springfield", "same")])})
        return "```json\n" + json.dumps(out) + "\n```"

    report = run_harness(_service(store), ["sam"], judge=judge, rules="RULES", run_id=RUN, batch=5, dry_run=False)

    assert prompts[0].startswith("RULES")
    assert report["users"]["sam"]["outcomes"] == {"deleted": 1}
    assert report["users"]["sam"]["finish"]["deleted"] == 1
    assert _units(store) == {keep.id: keep.content}


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    from tpasini.recordatio import cli

    db = tmp_path / "copy" / "memories.db"
    seed = SQLiteStore(db)
    keep = _unit("Sam lives in Springfield and works as a data analyst")
    drop = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    seed.add(keep, SCOPE)
    seed.add(drop, SCOPE)
    seed._conn.close()
    before = {p.name: p.read_bytes() for p in db.parent.iterdir()}
    rules = tmp_path / "rules.md"
    rules.write_text("Judge strictly.")
    # A judge command that answers "unsure" for every pair it is shown.
    judge = tmp_path / "judge.py"
    judge.write_text(
        "import json, sys\n"
        "text = sys.stdin.read()\n"
        "pairs = json.loads(text.split('## Pairs\\n\\n', 1)[1].split('\\n\\n## Reply format', 1)[0])\n"
        "print(json.dumps([{'pair_token': p['pair_token'], 'verdict': {'relation': 'unsure'}} for p in pairs]))\n"
    )
    out = tmp_path / "report.json"

    assert cli.reconcile_main([
        "--db", str(db), "--judge-cmd", f"{sys.executable} {judge}", "--judge-prompt", str(rules),
        "--output", str(out),
    ]) == 0

    assert {p.name: p.read_bytes() for p in db.parent.iterdir()} == before
    report = json.loads(out.read_text())
    assert report["dry_run"] is True
    assert report["users"]["sam"]["outcomes"] == {"kept": 1}
