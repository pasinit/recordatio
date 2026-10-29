"""Tests for the nightly memory reconcile / compaction job.

The judge is a scripted fake (no LLM): each test declares, per unit pair,
the verdict the "model" returns — including wrong or overreaching verdicts —
so the tests pin down what the deterministic verification lets through.
All fixtures are synthetic.
"""
import asyncio
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import EntityMention, KnowledgeType, KnowledgeUnit, Relation
from tpasini.recordatio.proxy import embeddings
from tpasini.recordatio.proxy.reconcile_routes import register_reconcile_routes
from tpasini.recordatio.reconcile import ReconcileSettings, build_reconciler
from tpasini.recordatio.reconcile.judge import JudgeCall, JudgeFact, JudgeVerdict, LLMJudge, verify_verdict
from tpasini.recordatio.reconcile.runner import CORRECT_DISMISS_REASON
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore
from tpasini.recordatio.storage.reconcile_log import ReconcileLog

SCOPE = "sam/_user"
T0 = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)

type Script = Callable[[KnowledgeUnit, KnowledgeUnit], JudgeVerdict | Exception]


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


def verdict(
    relation: str,
    *,
    keep: KnowledgeUnit | None = None,
    drop: KnowledgeUnit | None = None,
    facts: list[tuple[str, str, str]] = (),
) -> dict:
    """A verdict expressed with real units; the fake judge maps them to A/B."""
    return {"relation": relation, "keep": keep, "drop": drop, "facts": list(facts)}


class FakeJudge:
    """Returns scripted verdicts per unordered unit pair; ``default`` for the rest."""

    def __init__(self, default: str = "unrelated") -> None:
        self.script: dict[frozenset[str], dict | Exception] = {}
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def on(self, a: KnowledgeUnit, b: KnowledgeUnit, v: dict | Exception) -> None:
        self.script[frozenset((a.id, b.id))] = v

    async def judge(self, a: KnowledgeUnit, b: KnowledgeUnit) -> JudgeCall:
        self.calls.append((a.id, b.id))
        v = self.script.get(frozenset((a.id, b.id)), verdict(self.default))
        if isinstance(v, Exception):
            return JudgeCall(verdict=None, error=str(v))
        label = {a.id: "A", b.id: "B"}
        return JudgeCall(
            verdict=JudgeVerdict(
                relation=v["relation"],
                keep=label[v["keep"].id] if v["keep"] else None,
                drop=label[v["drop"].id] if v["drop"] else None,
                facts=[JudgeFact(dropped_fact=d, kept_span=k, status=s) for d, k, s in v["facts"]],
                evidence="scripted",
            ),
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.001,
        )


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "mem.db")


@pytest.fixture()
def feedback(tmp_path: Path) -> MemoryFeedbackStore:
    return MemoryFeedbackStore(tmp_path / "memory_feedback.db")


def _settings(**overrides: object) -> ReconcileSettings:
    return ReconcileSettings(model="anthropic/test-model", snapshot_keep=2, **overrides)


def _reconciler(store: SQLiteStore, judge: FakeJudge, feedback: MemoryFeedbackStore | None = None, **cfg: object):
    return build_reconciler(store, judge=judge, settings=_settings(**cfg), feedback=feedback)


def _units(store: SQLiteStore) -> dict[str, str]:
    rows = store._conn.execute("SELECT id, content FROM knowledge_units").fetchall()
    return {r["id"]: r["content"] for r in rows}


def run(coro):  # noqa: ANN001, ANN201
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


def test_duplicate_subsumed_is_deleted_and_keeper_verbatim(store: SQLiteStore) -> None:
    keep = _unit("Sam lives in Springfield and works as a data analyst")
    drop = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    store.add(keep, SCOPE)
    store.add(drop, SCOPE)
    judge = FakeJudge()
    judge.on(keep, drop, verdict(
        "duplicate_subsumed", keep=keep, drop=drop,
        facts=[("Sam lives in Springfield", "Sam lives in Springfield", "same")],
    ))

    report = run(_reconciler(store, judge).run(["sam"]))

    assert _units(store) == {keep.id: keep.content}
    assert report["totals"]["deleted"] == 1
    assert report["users"]["sam"]["by_relation"] == {"duplicate_subsumed": 1}


def test_partial_overlap_with_extra_fact_is_kept(store: SQLiteStore) -> None:
    keep = _unit("Sam lives in Springfield")
    other = _unit("Sam lives in Springfield near the river since 2019", at=T0 + timedelta(days=1))
    store.add(keep, SCOPE)
    store.add(other, SCOPE)
    judge = FakeJudge()
    # An overreaching verdict: claims subsumption but cannot quote the extra fact.
    judge.on(keep, other, verdict(
        "duplicate_subsumed", keep=keep, drop=other,
        facts=[("Sam lives in Springfield", "Sam lives in Springfield", "same")],
    ))

    report = run(_reconciler(store, judge).run(["sam"]))

    assert set(_units(store)) == {keep.id, other.id}
    assert report["totals"]["deleted"] == 0
    assert report["totals"]["kept_rejected"] == 1


def test_contradiction_with_newer_claim_deletes_older(store: SQLiteStore) -> None:
    old = _unit("Sam lives in Springfield")
    new = _unit("Sam lives in Shelbyville", at=T0 + timedelta(days=30))
    store.add(old, SCOPE)
    store.add(new, SCOPE)
    judge = FakeJudge()
    judge.on(old, new, verdict(
        "supersedes", keep=new, drop=old,
        facts=[("Sam lives in Springfield", "Sam lives in Shelbyville", "replaced")],
    ))

    report = run(_reconciler(store, judge).run(["sam"]))

    assert _units(store) == {new.id: new.content}
    assert report["users"]["sam"]["by_relation"] == {"supersedes": 1}


def test_contradiction_where_kept_is_actually_older_deletes_nothing(store: SQLiteStore) -> None:
    old = _unit("Sam lives in Springfield")
    new = _unit("Sam lives in Shelbyville", at=T0 + timedelta(days=30))
    store.add(old, SCOPE)
    store.add(new, SCOPE)
    judge = FakeJudge()
    judge.on(old, new, verdict(
        "supersedes", keep=old, drop=new,
        facts=[("Sam lives in Shelbyville", "Sam lives in Springfield", "replaced")],
    ))

    report = run(_reconciler(store, judge).run(["sam"]))

    assert set(_units(store)) == {old.id, new.id}
    assert report["totals"]["deleted"] == 0


@pytest.mark.parametrize("relation", ["unrelated", "compatible", "unsure"])
def test_non_deleting_relations_keep_both(store: SQLiteStore, relation: str) -> None:
    a = _unit("Sam likes hiking in Springfield")
    b = _unit("Sam likes cycling in Springfield", at=T0 + timedelta(days=1))
    store.add(a, SCOPE)
    store.add(b, SCOPE)
    judge = FakeJudge(default=relation)

    report = run(_reconciler(store, judge).run(["sam"]))

    assert judge.calls, "the pair should have been proposed and judged"
    assert set(_units(store)) == {a.id, b.id}
    assert report["totals"][f"kept_{relation}"] >= 1


def test_exact_duplicates_resolve_without_judge(store: SQLiteStore) -> None:
    a = _unit("Sam prefers tea over coffee.")
    b = _unit("sam prefers tea over coffee", at=T0 + timedelta(days=2))
    store.add(a, SCOPE)
    store.add(b, SCOPE)
    store._conn.execute("UPDATE knowledge_units SET extraction_count = 5 WHERE id = ?", (b.id,))
    store._conn.commit()
    judge = FakeJudge()

    run(_reconciler(store, judge).run(["sam"]))

    assert _units(store) == {b.id: b.content}  # the more-reinforced row survives
    assert all(b.id not in pair or a.id not in pair for pair in judge.calls)


def test_episodic_same_text_on_different_days_is_kept(store: SQLiteStore) -> None:
    a = _unit("Sam went to the gym", kind=KnowledgeType.EPISODIC)
    b = _unit("Sam went to the gym", kind=KnowledgeType.EPISODIC, at=T0 + timedelta(days=3))
    store.add(a, SCOPE)
    store.add(b, SCOPE)
    judge = FakeJudge()
    judge.on(a, b, verdict(
        "duplicate_subsumed", keep=b, drop=a, facts=[("Sam went to the gym", "Sam went to the gym", "same")],
    ))

    run(_reconciler(store, judge).run(["sam"]))

    assert set(_units(store)) == {a.id, b.id}


def test_other_scopes_are_never_paired(store: SQLiteStore) -> None:
    a = _unit("Sam lives in Springfield")
    b = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    store.add(a, "sam/_user")
    store.add(b, "sam/scout")
    judge = FakeJudge()

    run(_reconciler(store, judge).run(["sam"]))

    assert set(_units(store)) == {a.id, b.id}
    assert judge.calls == []


# ---------------------------------------------------------------------------
# Invariants: no new content, cascade, dry run, snapshot
# ---------------------------------------------------------------------------


def test_run_never_creates_units_or_changes_content(store: SQLiteStore, feedback: MemoryFeedbackStore) -> None:
    units = [_unit(f"Sam fact number {i} about Springfield", at=T0 + timedelta(hours=i)) for i in range(6)]
    for u in units:
        store.add(u, SCOPE)
    feedback.record(user_id="sam", day="2026-09-02", summary="s", findings=[], ops=[
        {"op_type": "correct", "target": {"unit_id": units[0].id, "new_content": "rewritten"}, "reason": "r"},
    ])
    judge = FakeJudge()
    judge.on(units[0], units[1], verdict(
        "supersedes", keep=units[1], drop=units[0],
        facts=[("Sam fact number 0 about Springfield", "Sam fact number 1 about Springfield", "replaced")],
    ))
    before = _units(store)

    run(_reconciler(store, judge, feedback).run(["sam"]))

    after = _units(store)
    assert len(after) <= len(before)
    assert all(before.get(uid) == content for uid, content in after.items())
    assert "rewritten" not in after.values()


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
    topic_id = store.list_topics()[0].id
    store.tag_unit(drop.id, topic_id, SCOPE)
    judge = FakeJudge()
    judge.on(keep, drop, verdict(
        "duplicate_subsumed", keep=keep, drop=drop,
        facts=[("Sam works at Acme Labs", "Sam works at Acme Labs", "same")],
    ))

    run(_reconciler(store, judge).run(["sam"]))

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
    # Orphan check: every indexed unit row still has its canonical row.
    orphans = c.execute(
        "SELECT COUNT(*) FROM memfts.memento_fts f WHERE f.kind = 'unit' "
        "AND NOT EXISTS (SELECT 1 FROM knowledge_units k WHERE k.id = f.item_id)"
    ).fetchone()[0]
    assert orphans == 0
    # The kept unit's links and the shared entities survive.
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
    store.add(keep, SCOPE)
    store.add(drop, SCOPE)
    feedback.record(user_id="sam", day="2026-09-02", summary="s", findings=[], ops=[
        {"op_type": "correct", "target": {"unit_id": drop.id, "new_content": "x"}, "reason": "r"},
    ])
    judge = FakeJudge()
    judge.on(keep, drop, verdict(
        "duplicate_subsumed", keep=keep, drop=drop,
        facts=[("Sam lives in Springfield", "Sam lives in Springfield", "same")],
    ))
    mem_before, fb_before = _dump(tmp_path / "mem.db"), _dump(tmp_path / "memory_feedback.db")
    log = ReconcileLog(tmp_path / "memory_reconcile.db")
    log_before = _dump(tmp_path / "memory_reconcile.db")

    report = build_reconciler(store, judge=judge, settings=_settings(), feedback=feedback, log=log).run(
        ["sam"], dry_run=True
    )
    report = run(report)

    assert _dump(tmp_path / "mem.db") == mem_before
    assert _dump(tmp_path / "memory_feedback.db") == fb_before
    assert _dump(tmp_path / "memory_reconcile.db") == log_before
    assert not (tmp_path / "reconcile-snapshots").exists()
    assert report["dry_run"] is True and report["snapshot"] is None
    assert [d["unit_id"] for d in report["deletions"]] == [drop.id]
    assert report["deletions"][0]["kept_unit_id"] == keep.id
    assert [o["action"] for o in report["ops"]] == ["dismiss"]


def test_snapshot_taken_before_mutation_and_retained(tmp_path: Path, store: SQLiteStore) -> None:
    for n in range(3):
        a = _unit(f"Sam owns {n} bikes")
        b = _unit(f"sam owns {n} bikes", at=T0 + timedelta(days=1))
        store.add(a, SCOPE)
        store.add(b, SCOPE)
        report = run(_reconciler(store, FakeJudge()).run(["sam"]))
        snap = Path(report["snapshot"])
        snapped = sqlite3.connect(str(snap / "mem.db"))
        ids = {r[0] for r in snapped.execute("SELECT id FROM knowledge_units")}
        snapped.close()
        assert {a.id, b.id} <= ids  # the snapshot predates the deletion
        assert (snap / "mem.db.emb").exists()
    snapshots = sorted((tmp_path / "reconcile-snapshots").iterdir())
    assert len(snapshots) == 2  # snapshot_keep=2


def test_no_snapshot_or_vacuum_when_nothing_to_do(tmp_path: Path, store: SQLiteStore) -> None:
    store.add(_unit("Sam likes tea"), SCOPE)
    report = run(_reconciler(store, FakeJudge()).run(["sam"]))
    assert report["snapshot"] is None and report["vacuumed"] is False


def test_audit_row_per_deletion(tmp_path: Path, store: SQLiteStore) -> None:
    old = _unit("Sam lives in Springfield")
    new = _unit("Sam lives in Shelbyville", at=T0 + timedelta(days=30))
    store.add(old, SCOPE)
    store.add(new, SCOPE)
    judge = FakeJudge()
    judge.on(old, new, verdict(
        "supersedes", keep=new, drop=old,
        facts=[("Sam lives in Springfield", "Sam lives in Shelbyville", "replaced")],
    ))
    report = run(_reconciler(store, judge).run(["sam"]))

    rows = ReconcileLog(tmp_path / "memory_reconcile.db").deletions(report["run_id"])
    assert len(rows) == 1
    row = rows[0]
    assert (row["unit_id"], row["kept_unit_id"], row["agent_id"], row["user_id"], row["relation"]) == (
        old.id, new.id, SCOPE, "sam", "supersedes",
    )
    assert row["evidence"]["facts"][0]["kept_span"] == "Sam lives in Shelbyville"


# ---------------------------------------------------------------------------
# Judge failures
# ---------------------------------------------------------------------------


def test_judge_failure_deletes_nothing(store: SQLiteStore, feedback: MemoryFeedbackStore) -> None:
    keep = _unit("Sam lives in Springfield and works as a data analyst")
    drop = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    store.add(keep, SCOPE)
    store.add(drop, SCOPE)
    feedback.record(user_id="sam", day="2026-09-02", summary="s", findings=[], ops=[
        {"op_type": "merge", "target": {"canonical_id": keep.id, "duplicate_ids": [drop.id]}, "reason": "dup"},
    ])
    judge = FakeJudge()
    judge.on(keep, drop, TimeoutError("judge timed out"))

    report = run(_reconciler(store, judge, feedback).run(["sam"]))

    assert set(_units(store)) == {keep.id, drop.id}
    assert report["totals"]["judge_errors"] >= 1
    op = feedback.list_ops(feedback.list_feedback("sam")[0].id)[0]
    assert op.status == "pending"  # retried next run


def test_llm_judge_timeout_and_garbage_yield_no_verdict() -> None:
    judge = LLMJudge(model="test", backend="anthropic", api_key="k", base_url=None, max_tokens=10, timeout_s=0.01)

    async def slow(_rendered: str) -> tuple[str, int, int]:
        await asyncio.sleep(1)
        return "{}", 0, 0

    async def garbage(_rendered: str) -> tuple[str, int, int]:
        return "I think they are the same memory.", 10, 5

    a, b = _unit("Sam likes tea"), _unit("Sam likes tea a lot")
    judge._stream = slow
    assert run(judge.judge(a, b)).verdict is None
    judge._stream = garbage
    assert run(judge.judge(a, b)).verdict is None


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
# Backlog of memory-feedback ops
# ---------------------------------------------------------------------------


def _ops(feedback: MemoryFeedbackStore) -> dict[str, tuple[str, str | None]]:
    out = {}
    for rec in feedback.list_feedback("sam", limit=100):
        for op in feedback.list_ops(rec.id):
            out[op.reason] = (op.status, op.decided_result)
    return out


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
    judge = FakeJudge()
    judge.on(old, new, verdict(
        "supersedes", keep=new, drop=old,
        facts=[("Sam lives in Springfield", "Sam lives in Shelbyville", "replaced")],
    ))
    judge.on(canonical, subsumed, verdict(
        "duplicate_subsumed", keep=canonical, drop=subsumed,
        facts=[("Sam drives a blue hatchback", "Sam drives a blue hatchback", "same")],
    ))
    judge.on(canonical, richer, verdict("compatible"))

    report = run(_reconciler(store, judge, feedback).run(["sam"]))

    ops = _ops(feedback)
    assert ops["tomb-pass"][0] == "applied"
    assert ops["merge-pass"][0] == "applied"
    assert ops["merge-fail"][0] == "dismissed" and "kept" in ops["merge-fail"][1]
    assert ops["correct"] == ("dismissed", CORRECT_DISMISS_REASON)
    assert ops["truncated"][0] == "dismissed" and "auto-rejected" in ops["truncated"][1]
    assert set(_units(store)) == {new.id, canonical.id, richer.id}
    assert report["totals"]["ops_apply"] == 2 and report["totals"]["ops_dismiss"] == 3


def test_backlog_tombstone_without_superseding_memory_is_dismissed(
    store: SQLiteStore, feedback: MemoryFeedbackStore
) -> None:
    lone = _unit("Sam once visited Springfield museum")
    store.add(lone, SCOPE)
    feedback.record(user_id="sam", day="2026-09-02", summary="s", findings=[], ops=[
        {"op_type": "tombstone", "target": {"unit_id": lone.id}, "reason": "stale"},
    ])
    run(_reconciler(store, FakeJudge(), feedback).run(["sam"]))
    assert _ops(feedback)["stale"][0] == "dismissed"
    assert lone.id in _units(store)


# ---------------------------------------------------------------------------
# Legacy invalidated rows
# ---------------------------------------------------------------------------


def test_legacy_purges_only_provably_superseded(store: SQLiteStore) -> None:
    past = T0 - timedelta(days=10)
    corrected = _unit("Sam lives in Springfield", valid_until=past)
    correction = _unit("Sam lives in Shelbyville", provenance=f"correction of {corrected.id}")
    tombstoned = _unit("Sam likes jazz", valid_until=past)
    expired = _unit("Sam is on holiday this week", kind=KnowledgeType.EPISODIC, valid_until=past)
    for u in (corrected, correction, tombstoned, expired):
        store.add(u, SCOPE)

    report = run(_reconciler(store, FakeJudge()).run(["sam"]))

    assert set(_units(store)) == {correction.id, tombstoned.id, expired.id}
    assert report["users"]["sam"]["legacy_unclassified_kept"] == 2
    assert report["users"]["sam"]["by_source"] == {"legacy": 1}


# ---------------------------------------------------------------------------
# Incremental behaviour
# ---------------------------------------------------------------------------


def test_kept_pairs_are_not_rejudged(store: SQLiteStore) -> None:
    a = _unit("Sam likes hiking in Springfield")
    b = _unit("Sam likes cycling in Springfield", at=T0 + timedelta(days=1))
    store.add(a, SCOPE)
    store.add(b, SCOPE)
    judge = FakeJudge(default="compatible")
    rec = _reconciler(store, judge)
    run(rec.run(["sam"]))
    first = len(judge.calls)
    run(rec.run(["sam"]))
    assert first >= 1 and len(judge.calls) == first


def test_judge_budget_caps_calls(store: SQLiteStore) -> None:
    for i in range(6):
        store.add(_unit(f"Sam likes Springfield park number {i}"), SCOPE)
    judge = FakeJudge(default="compatible")
    report = run(_reconciler(store, judge, max_judge_calls_per_user=2).run(["sam"]))
    assert len(judge.calls) == 2
    assert report["totals"]["budget_exhausted"] == 1


def test_embedding_signal_proposes_pairs(store: SQLiteStore) -> None:
    a = _unit("Sam resides in Springfield")
    b = _unit("Sam's home town is Springfield", at=T0 + timedelta(days=1))
    store.add(a, SCOPE)
    store.add(b, SCOPE)
    vec = embeddings.pack_embedding([1.0, 0.0, 0.0, 0.0])
    for u in (a, b):
        store.upsert_item_embedding(item_id=u.id, agent_id=SCOPE, kind="unit", surface_hash=u.id,
                                    embedding=vec, model="m", asserted_at=T0.isoformat())
    judge = FakeJudge()
    run(_reconciler(store, judge, fts_neighbours=0).run(["sam"]))
    assert {frozenset(c) for c in judge.calls} == {frozenset((a.id, b.id))}


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------


def test_endpoint_auth_and_dry_run(store: SQLiteStore, feedback: MemoryFeedbackStore) -> None:
    keep = _unit("Sam lives in Springfield and works as a data analyst")
    drop = _unit("Sam lives in Springfield", at=T0 + timedelta(days=1))
    store.add(keep, SCOPE)
    store.add(drop, SCOPE)
    judge = FakeJudge()
    judge.on(keep, drop, verdict(
        "duplicate_subsumed", keep=keep, drop=drop,
        facts=[("Sam lives in Springfield", "Sam lives in Springfield", "same")],
    ))
    app = FastAPI()
    register_reconcile_routes(
        app, get_store=lambda: store, memory_feedback_store=feedback,
        settings_factory=_settings, judge_factory=lambda _s, _k: judge,
    )

    async def go() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as anon:
            assert (await anon.post("/v1/memory-reconcile")).status_code == 401
        headers = {"authorization": "Bearer test-key"}
        async with httpx.AsyncClient(transport=transport, base_url="http://test", headers=headers) as c:
            dry = (await c.post("/v1/users/sam/memory-reconcile", params={"dry_run": "true"})).json()
            assert dry["dry_run"] is True and dry["totals"]["would_delete"] == 1
            assert set(_units(store)) == {keep.id, drop.id}
            real = (await c.post("/v1/memory-reconcile")).json()
            assert real["totals"]["deleted"] == 1 and "deletions" not in real

    asyncio.run(go())
    assert set(_units(store)) == {keep.id}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tpasini.recordatio import cli, reconcile

    db = tmp_path / "copy" / "memories.db"
    seed = SQLiteStore(db)
    keep = _unit("Sam lives in Springfield and works as a data analyst")
    drop = _unit("sam lives in springfield and works as a data analyst", at=T0 + timedelta(days=1))
    seed.add(keep, SCOPE)
    seed.add(drop, SCOPE)
    seed._conn.close()
    before = {p.name: p.read_bytes() for p in db.parent.iterdir()}
    monkeypatch.setattr(reconcile, "make_judge", lambda _settings: FakeJudge())
    monkeypatch.setenv("RECORDATIO_RECONCILE_MODEL", "anthropic/test-model")
    out = tmp_path / "report.json"

    assert cli.reconcile_main(["--db", str(db), "--dry-run", "--output", str(out)]) == 0

    assert {p.name: p.read_bytes() for p in db.parent.iterdir()} == before
    report = out.read_text()
    assert drop.id in report or keep.id in report
    assert '"would_delete": 1' in report


def test_active_unit_never_dropped_for_an_inactive_keeper(store: SQLiteStore, feedback: MemoryFeedbackStore) -> None:
    canonical = _unit("Sam drives a blue hatchback to work", valid_until=T0)  # already tombstoned
    dup = _unit("Sam drives a blue hatchback", at=T0 + timedelta(days=1))
    store.add(canonical, SCOPE)
    store.add(dup, SCOPE)
    feedback.record(user_id="sam", day="2026-09-02", summary="s", findings=[], ops=[
        {"op_type": "merge", "target": {"canonical_id": canonical.id, "duplicate_ids": [dup.id]}, "reason": "m"},
    ])
    judge = FakeJudge()
    judge.on(canonical, dup, verdict(
        "duplicate_subsumed", keep=canonical, drop=dup,
        facts=[("Sam drives a blue hatchback", "Sam drives a blue hatchback", "same")],
    ))
    run(_reconciler(store, judge, feedback).run(["sam"]))
    assert dup.id in _units(store)
    assert _ops(feedback)["m"][0] == "dismissed"
