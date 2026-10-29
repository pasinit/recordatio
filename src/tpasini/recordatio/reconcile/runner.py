"""Nightly reconcile + compaction of the memory store.

Deletes — physically — only superseded memories, and never writes memory
text. A unit is superseded when, in the SAME scope:
- a kept unit states every one of its facts (``duplicate_subsumed``), or
- a strictly newer unit replaces its claim (``supersedes``), or
- it was already invalidated because it was superseded, provably: a human
  approved ``correct`` op left a unit with ``provenance = "correction of
  <id>"`` (or a newer sibling correction), or an applied ``merge`` op named
  it as a duplicate AND the judge verifies that today.
Everything else stays: stale-looking, low-value and naturally expired
(``valid_until`` from its own temporal validity) units are never touched.

One run, per user, in this order (one shared judge budget per user):
1. backlog — every pending memory-feedback op is re-checked through the same
   judge + verification: a passing tombstone/merge is applied AS a deletion
   and marked ``applied``; a failing one is ``dismissed`` with the reason;
   every ``correct`` op is dismissed (content-rewriting ops are not
   applied); a judge failure leaves the op ``pending`` for the next run.
2. legacy — already-invalidated units, purged only when provably superseded
   (see above); the rest are counted as unclassified and kept.
3. candidates — exact duplicates (no judge), then embedding / FTS neighbour
   pairs of the units new since the last run plus a rolling slice of older
   ones, each judged and verified (``reconcile.judge``).

Nothing is mutated until planning is complete. A dry run returns the plan
and writes nothing anywhere. A real run takes a consistent snapshot (sqlite
backup API) before the first mutation, keeps the last N, purges with a full
cascade (``SQLiteStore.purge_units``), writes one audit row per deleted unit
(``storage.reconcile_log``), marks the ops, advances cursors, then VACUUMs.
"""
import logging
import re
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from tpasini.recordatio.models import KnowledgeUnit
from tpasini.recordatio.proxy.memory_ops import InvalidOpError, _require_unit
from tpasini.recordatio.reconcile.candidates import (
    candidate_pairs,
    content_hash,
    exact_duplicates,
    pair_key,
)
from tpasini.recordatio.reconcile.judge import PairJudge, VerifiedDrop, recency, verify_verdict
from tpasini.recordatio.reconcile.settings import ReconcileSettings
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackOp, MemoryFeedbackStore
from tpasini.recordatio.storage.reconcile_log import DeletionRecord, ReconcileLog, ScopeCursor

logger = logging.getLogger(__name__)

CORRECT_DISMISS_REASON = "content-rewriting ops are not applied"
_SNAPSHOT_DIR_RE = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{8}$")


def _is_active(unit: KnowledgeUnit) -> bool:
    if unit.valid_until is None:
        return True
    until = unit.valid_until if unit.valid_until.tzinfo else unit.valid_until.replace(tzinfo=UTC)
    return until > datetime.now(UTC)


@dataclass
class PlannedDeletion:
    unit: KnowledgeUnit
    agent_id: str
    kept: KnowledgeUnit | None
    relation: str
    source: str
    evidence: dict[str, Any]


@dataclass
class OpDecision:
    op: MemoryFeedbackOp
    action: Literal["apply", "dismiss", "pending"]
    result: str
    deletions: list[PlannedDeletion] = field(default_factory=list)


@dataclass
class _UserPass:
    """Everything one user's planning produced; applied (or reported) after."""

    user_id: str
    judge_calls_left: int
    pairs_left: int
    stats: Counter = field(default_factory=Counter)
    cost_usd: float = 0.0
    deletions: list[PlannedDeletion] = field(default_factory=list)
    ops: list[OpDecision] = field(default_factory=list)
    dropped: set[str] = field(default_factory=set)
    kept: set[str] = field(default_factory=set)
    judged_cache: list[tuple[str, str]] = field(default_factory=list)
    cursors: dict[str, ScopeCursor] = field(default_factory=dict)

    def claim(self, d: PlannedDeletion) -> bool:
        """Reserve a deletion. Refused when it would chain within this run (the
        drop already served as a keep, or the keep is itself being dropped) —
        such pairs resolve on a later run against the then-current store — and
        when an active unit would be dropped in favour of an inactive one (its
        facts would vanish from recall)."""
        if d.unit.id in self.dropped or d.unit.id in self.kept or (d.kept and d.kept.id in self.dropped):
            return False
        if d.kept is not None and _is_active(d.unit) and not _is_active(d.kept):
            self.stats["refused_inactive_keeper"] += 1
            return False
        self.dropped.add(d.unit.id)
        if d.kept is not None:
            self.kept.add(d.kept.id)
        return True


type _JudgeOutcome = VerifiedDrop | Literal["kept", "error", "budget", "cached"]


class Reconciler:
    def __init__(
        self,
        store: SQLiteStore,
        log: ReconcileLog,
        *,
        judge: PairJudge,
        settings: ReconcileSettings,
        feedback: MemoryFeedbackStore | None = None,
        snapshot_root: Path | None = None,
    ) -> None:
        self._store = store
        self._log = log
        self._pair_judge = judge
        self._cfg = settings
        self._feedback = feedback
        self._snapshot_root = snapshot_root or store.db_path.parent / "reconcile-snapshots"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(
        self, user_ids: list[str] | None = None, *, dry_run: bool = False, details: bool = True
    ) -> dict[str, Any]:
        started = datetime.now(UTC)
        run_id = uuid.uuid4().hex
        users = user_ids if user_ids is not None else self._all_users()
        passes = [await self._plan_user(u) for u in users]

        snapshot: str | None = None
        vacuumed = False
        if not dry_run:
            self._log.start_run(run_id, ",".join(users) if user_ids is not None else "*")
            if any(p.deletions or any(o.action != "pending" for o in p.ops) for p in passes):
                snapshot = str(self._snapshot(run_id, started))
                self._log.set_snapshot(run_id, snapshot)
            deleted = sum([self._apply_user(run_id, p) for p in passes])
            if deleted:
                self._store.vacuum()
                vacuumed = True

        report = self._report(run_id, started, passes, dry_run=dry_run, snapshot=snapshot, vacuumed=vacuumed)
        if not dry_run:
            self._log.finish_run(run_id, {k: v for k, v in report.items() if k not in ("deletions", "ops")})
        logger.info("memory reconcile %s (dry_run=%s): %s", run_id, dry_run, report["totals"])
        if not details:
            report.pop("deletions")
            report.pop("ops")
        return report

    def _all_users(self) -> list[str]:
        users = set(self._store.list_user_ids())
        if self._feedback is not None:
            users |= {rec.user_id for rec in self._feedback.list_all(limit=100_000)}
        return sorted(users)

    # ------------------------------------------------------------------
    # Planning (read-only)
    # ------------------------------------------------------------------

    async def _plan_user(self, user_id: str) -> _UserPass:
        up = _UserPass(
            user_id=user_id,
            judge_calls_left=self._cfg.max_judge_calls_per_user,
            pairs_left=self._cfg.max_pairs_per_user,
        )
        await self._plan_backlog(up)
        for scope in self._store.user_scopes(user_id):
            await self._plan_legacy(up, scope)
            await self._plan_candidates(up, scope)
        return up

    async def _judge(
        self, up: _UserPass, a: KnowledgeUnit, b: KnowledgeUnit, *, want_drop: str | None
    ) -> _JudgeOutcome:
        """Judge one pair under the user's budget; ``want_drop`` restricts a
        deleting verdict to that unit (ops name which unit they target)."""
        key = pair_key(a, b)
        if self._log.judged(key) is not None:
            up.stats["pairs_cached"] += 1
            return "cached"
        if up.judge_calls_left <= 0:
            up.stats["budget_exhausted"] = 1
            return "budget"
        up.judge_calls_left -= 1
        up.stats["judge_calls"] += 1
        call = await self._pair_judge.judge(a, b)
        up.stats["judge_input_tokens"] += call.input_tokens
        up.stats["judge_output_tokens"] += call.output_tokens
        if call.cost_usd is not None:
            up.cost_usd += call.cost_usd
        if call.verdict is None:
            up.stats["judge_errors"] += 1
            logger.warning("reconcile judge failed for %s/%s: %s", a.id, b.id, call.error)
            return "error"
        verified, reason = verify_verdict(call.verdict, a, b)
        if verified is not None and want_drop is not None and verified.drop.id != want_drop:
            verified, reason = None, "rejected: verdict drops the other unit"
        if verified is None:
            bucket = reason if reason in ("compatible", "unrelated", "unsure") else "rejected"
            up.stats[f"kept_{bucket}"] += 1
            up.judged_cache.append((key, bucket))
            return "kept"
        return verified

    def _deletion(self, v: VerifiedDrop, agent_id: str, source: str) -> PlannedDeletion:
        return PlannedDeletion(
            unit=v.drop, agent_id=agent_id, kept=v.keep, relation=v.relation, source=source, evidence=v.evidence
        )

    # -- 1. backlog ----------------------------------------------------------

    async def _plan_backlog(self, up: _UserPass) -> None:
        if self._feedback is None:
            return
        for rec in self._feedback.list_feedback(up.user_id, limit=100_000):
            for op in self._feedback.list_ops(rec.id):
                if op.status == "pending":
                    up.ops.append(await self._decide_op(up, op))

    async def _decide_op(self, up: _UserPass, op: MemoryFeedbackOp) -> OpDecision:
        if op.op_type == "correct":
            up.stats["ops_correct_dismissed"] += 1
            return OpDecision(op, "dismiss", CORRECT_DISMISS_REASON)
        try:
            if op.op_type == "tombstone":
                return await self._decide_tombstone(up, op)
            if op.op_type == "merge":
                return await self._decide_merge(up, op)
        except InvalidOpError as exc:
            return OpDecision(op, "dismiss", f"auto-rejected: {exc}")
        return OpDecision(op, "dismiss", f"auto-rejected: unknown op_type {op.op_type!r}")

    async def _decide_tombstone(self, up: _UserPass, op: MemoryFeedbackOp) -> OpDecision:
        target, scope = _require_unit(self._store, up.user_id, str(op.target.get("unit_id", "")))
        active = [u for u, _ in self._store.scope_units(scope, active=True)]
        pool = active if any(u.id == target.id for u in active) else [*active, target]
        neighbours = candidate_pairs(
            self._store, scope, pool, [target],
            cosine_threshold=self._cfg.cosine_threshold,
            fts_neighbours=self._cfg.tombstone_neighbours,
            exclude=up.dropped - {target.id},
        )[: self._cfg.tombstone_neighbours]
        retry = False
        for pair in neighbours:
            outcome = await self._judge(up, pair.a, pair.b, want_drop=target.id)
            if isinstance(outcome, VerifiedDrop):
                d = self._deletion(outcome, scope, f"backlog-op:{op.id}")
                if up.claim(d):
                    return OpDecision(op, "apply", f"deleted {target.id} ({d.relation}; kept {d.kept.id})", [d])
                retry = True
            elif outcome in ("error", "budget"):
                retry = True
        if retry:
            return OpDecision(op, "pending", "not decided this run (judge unavailable or budget spent)")
        return OpDecision(op, "dismiss", "kept: no memory in its scope verifiably supersedes or subsumes it")

    async def _decide_merge(self, up: _UserPass, op: MemoryFeedbackOp) -> OpDecision:
        canonical, scope = _require_unit(self._store, up.user_id, str(op.target.get("canonical_id", "")))
        dup_ids = op.target.get("duplicate_ids") or []
        if not isinstance(dup_ids, list) or not dup_ids:
            raise InvalidOpError("merge needs a non-empty duplicate_ids list")
        dups = [_require_unit(self._store, up.user_id, str(d)) for d in dup_ids if str(d) != canonical.id]
        planned: list[PlannedDeletion] = []
        kept_notes: list[str] = []
        for dup, dup_scope in dups:
            if dup_scope != scope:
                kept_notes.append(f"{dup.id}: kept (different scope from the canonical)")
                continue
            if _is_active(dup) and not _is_active(canonical):
                kept_notes.append(f"{dup.id}: kept (the canonical is no longer active)")
                continue
            outcome = await self._judge(up, canonical, dup, want_drop=dup.id)
            if outcome in ("error", "budget"):
                return OpDecision(op, "pending", "not decided this run (judge unavailable or budget spent)")
            if isinstance(outcome, VerifiedDrop):
                planned.append(self._deletion(outcome, scope, f"backlog-op:{op.id}"))
            else:
                kept_notes.append(f"{dup.id}: kept (not verified as subsumed by the canonical)")
        claimed = [d for d in planned if up.claim(d)]
        if len(claimed) != len(planned):
            for d in claimed:  # release; the whole op retries next run
                up.dropped.discard(d.unit.id)
            return OpDecision(op, "pending", "not decided this run (overlaps another deletion in this run)")
        if not claimed:
            return OpDecision(op, "dismiss", "; ".join(kept_notes) or "kept: nothing verified")
        result = f"deleted {', '.join(d.unit.id for d in claimed)} (kept {canonical.id})"
        if kept_notes:
            result += "; " + "; ".join(kept_notes)
        return OpDecision(op, "apply", result, claimed)

    # -- 2. legacy invalidated rows --------------------------------------------

    def _applied_merge_dups(self, user_id: str) -> dict[str, str]:
        """``{duplicate_id: canonical_id}`` from merge ops a human applied."""
        out: dict[str, str] = {}
        if self._feedback is None:
            return out
        for rec in self._feedback.list_feedback(user_id, limit=100_000):
            for op in self._feedback.list_ops(rec.id):
                if op.op_type != "merge" or op.status != "applied":
                    continue
                canonical = str(op.target.get("canonical_id", ""))
                for dup in op.target.get("duplicate_ids") or []:
                    if str(dup) != canonical:
                        out[str(dup)] = canonical
        return out

    async def _plan_legacy(self, up: _UserPass, scope: str) -> None:
        expired = [u for u, _ in self._store.scope_units(scope, active=False)]
        if not expired:
            return
        merges = self._applied_merge_dups(up.user_id)
        for unit in expired:
            if unit.id in up.dropped:
                continue
            successors = self._store.units_by_provenance(scope, f"correction of {unit.id}")
            if not successors and unit.provenance.startswith("correction of "):
                successors = [
                    s for s in self._store.units_by_provenance(scope, unit.provenance)
                    if s.id != unit.id and recency(s) > recency(unit)
                ]
            if successors:
                keep = max(successors, key=recency)
                d = PlannedDeletion(
                    unit=unit, agent_id=scope, kept=keep, relation="supersedes", source="legacy:correction",
                    evidence={"provenance": keep.provenance},
                )
                if up.claim(d):
                    up.deletions.append(d)
                continue
            canonical_id = merges.get(unit.id)
            if canonical_id is not None:
                pair = self._store.get_unit_for_user(up.user_id, canonical_id)
                if pair is not None and pair[1] == scope:
                    outcome = await self._judge(up, pair[0], unit, want_drop=unit.id)
                    if isinstance(outcome, VerifiedDrop):
                        d = self._deletion(outcome, scope, "legacy:applied-merge")
                        if up.claim(d):
                            up.deletions.append(d)
                            continue
                up.stats["legacy_merge_unverified"] += 1
                continue
            up.stats["legacy_unclassified_kept"] += 1

    # -- 3. candidate pairs ----------------------------------------------------

    def _focus(self, scope: str, rows: list[tuple[KnowledgeUnit, int]]) -> tuple[list[KnowledgeUnit], ScopeCursor]:
        """Units new since the last run + a rolling slice of older ones, and
        the cursor to persist once this scope is fully processed."""
        cursor = self._log.get_cursor(scope)
        max_rowid = max((r for _, r in rows), default=cursor.new_watermark)
        new = [u for u, r in rows if r > cursor.new_watermark]
        older = [(u, r) for u, r in rows if r <= cursor.new_watermark]
        size = self._cfg.rolling_slice
        window = [(u, r) for u, r in older if r > cursor.rolling_after][:size]
        if len(window) < size:  # wrap around to the oldest rows
            taken = {u.id for u, _ in window}
            window += [(u, r) for u, r in older if u.id not in taken][: size - len(window)]
        rolling_after = window[-1][1] if window else 0
        return new + [u for u, _ in window], ScopeCursor(new_watermark=max_rowid, rolling_after=rolling_after)

    async def _plan_candidates(self, up: _UserPass, scope: str) -> None:
        rows = self._store.scope_units(scope, active=True)
        units = [u for u, _ in rows if u.id not in up.dropped]
        up.stats["units_scanned"] += len(units)
        for dup in exact_duplicates(units):
            d = PlannedDeletion(
                unit=dup.drop, agent_id=scope, kept=dup.keep, relation="duplicate_subsumed", source="exact-match",
                evidence={"signal": "identical normalised content"},
            )
            if up.claim(d):
                up.deletions.append(d)
        focus, next_cursor = self._focus(scope, rows)
        up.stats["focus_units"] += len(focus)
        complete = True
        for pair in candidate_pairs(
            self._store, scope, units, focus,
            cosine_threshold=self._cfg.cosine_threshold,
            fts_neighbours=self._cfg.fts_neighbours,
        ):
            if {pair.a.id, pair.b.id} & up.dropped:
                continue
            if self._log.judged(pair_key(pair.a, pair.b)) is not None:
                up.stats["pairs_cached"] += 1
                continue
            if up.pairs_left <= 0:
                complete = False
                break
            up.pairs_left -= 1
            up.stats["candidate_pairs"] += 1
            up.stats[f"signal_{pair.signal}"] += 1
            outcome = await self._judge(up, pair.a, pair.b, want_drop=None)
            if outcome == "budget":
                complete = False
                break
            if outcome == "error":
                complete = False
            elif isinstance(outcome, VerifiedDrop):
                d = self._deletion(outcome, scope, f"judge:{pair.signal}")
                if up.claim(d):
                    up.deletions.append(d)
                else:
                    complete = False  # chained — revisit next run
        if complete:
            up.cursors[scope] = next_cursor

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _snapshot(self, run_id: str, started: datetime) -> Path:
        target = self._snapshot_root / f"{started.strftime('%Y%m%dT%H%M%S%fZ')}-{run_id[:8]}"
        self._store.backup_to(target)
        if self._feedback is not None:
            self._feedback.backup_to(target / "memory_feedback.db")
        keep = max(1, self._cfg.snapshot_keep)
        ours = sorted(
            p for p in self._snapshot_root.iterdir()
            if p.is_dir() and p != target and _SNAPSHOT_DIR_RE.match(p.name)
        )
        for old in ours[: max(0, len(ours) - (keep - 1))]:
            shutil.rmtree(old, ignore_errors=True)
        return target

    def _still_as_judged(self, user_id: str, unit: KnowledgeUnit, scope: str) -> bool:
        pair = self._store.get_unit_for_user(user_id, unit.id)
        return pair is not None and pair[1] == scope and content_hash(pair[0].content) == content_hash(unit.content)

    def _apply_user(self, run_id: str, up: _UserPass) -> int:
        """Execute one user's plan; return the number of units deleted."""
        all_deletions = [*up.deletions, *(d for o in up.ops if o.action == "apply" for d in o.deletions)]
        done: set[str] = set()
        by_scope: dict[str, list[PlannedDeletion]] = {}
        for d in all_deletions:
            # The store is live: re-check both sides before deleting anything.
            if not self._still_as_judged(up.user_id, d.unit, d.agent_id) or (
                d.kept is not None and not self._still_as_judged(up.user_id, d.kept, d.agent_id)
            ):
                up.stats["skipped_changed_since_judged"] += 1
                continue
            by_scope.setdefault(d.agent_id, []).append(d)
        for scope, items in by_scope.items():
            counts = self._store.purge_units([d.unit.id for d in items], scope)
            for k, v in counts.items():
                up.stats[f"purged_{k}"] += v
            for d in items:
                done.add(d.unit.id)
                self._log.record_deletion(DeletionRecord(
                    run_id=run_id, unit_id=d.unit.id, agent_id=scope, user_id=up.user_id,
                    kept_unit_id=d.kept.id if d.kept else None, relation=d.relation, source=d.source,
                    evidence=d.evidence,
                ))
        if self._feedback is not None:
            for o in up.ops:
                if o.action == "dismiss":
                    self._feedback.mark_op(o.op.id, status="dismissed", result=o.result)
                elif o.action == "apply":
                    if all(d.unit.id in done for d in o.deletions):
                        self._feedback.mark_op(o.op.id, status="applied", result=f"reconcile: {o.result}")
                    else:
                        o.action, o.result = "pending", "not applied: a unit changed since it was judged"
        for key, relation in up.judged_cache:
            self._log.remember_judged(key, relation)
        for scope, cursor in up.cursors.items():
            self._log.set_cursor(scope, cursor)
        up.stats["deleted"] = len(done)
        up.deletions = [d for d in up.deletions if d.unit.id in done]
        return len(done)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _report(
        self,
        run_id: str,
        started: datetime,
        passes: list[_UserPass],
        *,
        dry_run: bool,
        snapshot: str | None,
        vacuumed: bool,
    ) -> dict[str, Any]:
        users: dict[str, Any] = {}
        totals: Counter = Counter()
        deletions: list[dict[str, Any]] = []
        ops: list[dict[str, Any]] = []
        total_cost = 0.0
        for up in passes:
            planned = [*up.deletions, *(d for o in up.ops if o.action == "apply" for d in o.deletions)]
            by_relation = Counter(d.relation for d in planned)
            by_source = Counter(d.source.split(":", 1)[0] for d in planned)
            op_actions = Counter(f"ops_{o.action}" for o in up.ops)
            stats = Counter(up.stats)
            stats.update(op_actions)
            stats["would_delete" if dry_run else "deleted"] = len(planned) if dry_run else up.stats["deleted"]
            users[up.user_id] = {
                **dict(sorted(stats.items())),
                "by_relation": dict(by_relation),
                "by_source": dict(by_source),
                "judge_cost_usd": round(up.cost_usd, 6),
            }
            totals.update(stats)
            total_cost += up.cost_usd
            for d in planned:
                deletions.append({
                    "user_id": up.user_id,
                    "agent_id": d.agent_id,
                    "unit_id": d.unit.id,
                    "content": d.unit.content,
                    "kept_unit_id": d.kept.id if d.kept else None,
                    "kept_content": d.kept.content if d.kept else None,
                    "relation": d.relation,
                    "source": d.source,
                    "evidence": d.evidence,
                })
            for o in up.ops:
                ops.append({
                    "user_id": up.user_id, "op_id": o.op.id, "op_type": o.op.op_type,
                    "action": o.action, "result": o.result,
                })
        return {
            "run_id": run_id,
            "dry_run": dry_run,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "snapshot": snapshot,
            "vacuumed": vacuumed,
            "totals": {**dict(sorted(totals.items())), "judge_cost_usd": round(total_cost, 6)},
            "users": users,
            "deletions": deletions,
            "ops": ops,
        }


__all__ = ["CORRECT_DISMISS_REASON", "OpDecision", "PlannedDeletion", "Reconciler"]
