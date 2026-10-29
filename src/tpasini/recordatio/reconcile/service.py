"""Memory reconcile + compaction, driven by an external judge.

Recordatio never calls a model here. A judge agent (the nightly Cardea
routine, or the ``agentmem reconcile`` harness) loops over three calls:

1. ``candidates`` (read-only) — a bounded batch of unit pairs for one user,
   each with both units' text and a ``pair_token`` binding exactly what was
   shown (ids, scopes, content, validity, the op it came from).
2. ``apply`` — one structured verdict per pair token. Recordatio re-checks the
   token against the live store (a changed unit ⇒ ``stale``, nothing
   happens), runs ``verify_verdict`` plus the scope/op/activity rules, and
   physically deletes the dropped unit only if everything passes. The outcome
   is recorded per token: a replay returns it and never has a second effect.
   Kept verdicts go into the verdict cache so the pair is not offered again.
3. ``finish`` — deterministic work (exact duplicates, units a human-approved
   correction already superseded), settling pending memory-feedback ops,
   advancing the incremental cursors, VACUUM, and the run summary. Replays
   return the stored summary.

A unit is deleted only when superseded in its user's memory:
- ``duplicate_subsumed`` — a kept unit states every one of its facts;
- ``supersedes`` — a strictly newer unit replaces its claim;
always within one scope, or across the user's shared scope
(``{user}/_user``) and one of their agent scopes with the shared copy kept.
Never across users, never in favour of an inactive unit. Nothing else is
deleted: stale-looking, low-value and naturally expired units stay.

Pending feedback ops become pairs in ``candidates`` (a tombstone's target
against its neighbours, a merge's canonical against each duplicate, the
target fixed as the unit that may be dropped). ``finish`` marks an op
``applied`` once one of its pairs deleted, ``dismissed`` once all were kept
(with the reasons), and dismisses every ``correct`` op ("content-rewriting
ops are not applied") and every invalid one.

A run id (chosen by the caller, one per user per night) scopes the per-run
budget. The first mutation of a run takes a consistent snapshot (sqlite
backup API; the last N are kept). Dry runs write nothing anywhere: ``apply``
verifies only, ``finish`` reports what it would do.
"""
import hashlib
import json
import logging
import re
import shutil
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tpasini.recordatio.identity import RESERVED_USER_SLUG
from tpasini.recordatio.models import KnowledgeUnit
from tpasini.recordatio.proxy.memory_ops import InvalidOpError, _require_unit
from tpasini.recordatio.reconcile.candidates import (
    candidate_pairs,
    content_hash,
    exact_duplicates,
    pair_key,
)
from tpasini.recordatio.reconcile.judge import parse_verdict, recency, verify_verdict
from tpasini.recordatio.reconcile.settings import ReconcileSettings
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackOp, MemoryFeedbackStore
from tpasini.recordatio.storage.reconcile_log import DeletionRecord, ReconcileLog, ScopeCursor

logger = logging.getLogger(__name__)

CORRECT_DISMISS_REASON = "content-rewriting ops are not applied"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SNAPSHOT_DIR_RE = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{8}$")
_ID = r"[0-9A-Za-z_.:-]{1,64}"
_TOKEN_RE = re.compile(rf"^({_ID})~({_ID})~({_ID}|-)~(\d{{1,12}})~([0-9a-f]{{32}})$")
_BIG = 100_000


class ReconcileError(ValueError):
    """A malformed request (bad run id, token or verdict). Nothing happened."""


class RunFinishedError(ReconcileError):
    """The run already finished; start a new run id."""


def _is_active(unit: KnowledgeUnit) -> bool:
    if unit.valid_until is None:
        return True
    until = unit.valid_until if unit.valid_until.tzinfo else unit.valid_until.replace(tzinfo=UTC)
    return until > datetime.now(UTC)


@dataclass(frozen=True)
class Pair:
    a: KnowledgeUnit
    a_scope: str
    b: KnowledgeUnit
    b_scope: str
    kind: str  # "op" | "legacy-merge" | "candidate"
    signal: str
    op_id: int | None = None
    want_drop: str | None = None  # the only unit this pair may delete


def _unit_sig(u: KnowledgeUnit, scope: str) -> list[object]:
    return [u.id, scope, u.type.value, content_hash(u.content), u.valid_until.isoformat() if u.valid_until else None]


def pair_token(user_id: str, p: Pair) -> str:
    """``a~b~want~op~digest``: the ids to reload plus a digest of everything the
    judge was shown, so a verdict on changed units is refused as stale."""
    digest = hashlib.sha256(
        json.dumps([user_id, p.op_id, p.want_drop, _unit_sig(p.a, p.a_scope), _unit_sig(p.b, p.b_scope)]).encode()
    ).hexdigest()[:32]
    return f"{p.a.id}~{p.b.id}~{p.want_drop or '-'}~{p.op_id or 0}~{digest}"


def unit_view(u: KnowledgeUnit, scope: str) -> dict[str, Any]:
    return {
        "id": u.id,
        "scope": scope,
        "type": u.type.value,
        "content": u.content,
        "valid_from": u.valid_from.isoformat(),
        "asserted_at": u.asserted_at.isoformat(),
        "valid_until": u.valid_until.isoformat() if u.valid_until else None,
    }


@dataclass
class _Deletion:
    unit: KnowledgeUnit
    scope: str
    kept: KnowledgeUnit | None
    relation: str
    source: str
    evidence: dict[str, Any] = field(default_factory=dict)


class ReconcileService:
    def __init__(
        self,
        store: SQLiteStore,
        log: ReconcileLog,
        *,
        settings: ReconcileSettings,
        feedback: MemoryFeedbackStore | None = None,
        snapshot_root: Path | None = None,
    ) -> None:
        self._store = store
        self._log = log
        self._cfg = settings
        self._feedback = feedback
        self._snapshot_root = snapshot_root or store.db_path.parent / "reconcile-snapshots"
        # Dry-run verdicts are never recorded; this in-memory set only stops
        # ``candidates`` offering a previewed pair again within the same run.
        self._previewed: dict[tuple[str, str], set[str]] = {}

    def users(self) -> list[str]:
        """Users with units, or with feedback ops still to settle."""
        users = set(self._store.list_user_ids())
        if self._feedback is not None:
            users |= {rec.user_id for rec in self._feedback.list_all(limit=_BIG)}
        return sorted(users)

    # ------------------------------------------------------------------
    # candidates
    # ------------------------------------------------------------------

    def candidates(self, user_id: str, run_id: str, *, limit: int) -> dict[str, Any]:
        _check_run_id(run_id)
        run = self._log.get_run(user_id, run_id)
        if run is not None and run.finished_at:
            return {"user_id": user_id, "run_id": run_id, "pairs": [], "budget_left": 0, "more": False,
                    "finished": True}
        previewed = self._previewed.get((user_id, run_id), set())
        used = len(self._log.run_verdicts(user_id, run_id)) + len(previewed)
        budget_left = max(0, self._cfg.max_pairs_per_run - used)
        want = max(0, min(limit, self._cfg.max_batch, budget_left))
        out: list[dict[str, Any]] = []
        more = False
        for p in self._open_pairs(user_id, skip=previewed):
            if len(out) >= want:
                more = True
                break
            out.append({
                "pair_token": pair_token(user_id, p),
                "kind": p.kind,
                "signal": p.signal,
                "op_id": p.op_id,
                "may_drop_only": ("A" if p.want_drop == p.a.id else "B") if p.want_drop else None,
                "a": unit_view(p.a, p.a_scope),
                "b": unit_view(p.b, p.b_scope),
            })
        return {"user_id": user_id, "run_id": run_id, "pairs": out, "budget_left": budget_left - len(out),
                "more": more, "finished": False}

    def _open_pairs(self, user_id: str, *, skip: set[str] = frozenset()) -> Iterator[Pair]:
        """Undecided pairs, best first: backlog ops, applied merges to purge, candidates."""
        for p in self._all_pairs(user_id):
            token = pair_token(user_id, p)
            if token in skip or self._log.get_verdict(user_id, token) is not None:
                continue
            if self._log.judged(pair_key(p.a, p.b)) is not None:
                continue
            yield p

    def _all_pairs(self, user_id: str) -> Iterator[Pair]:
        exact_drops = {d.unit.id for d in self._exact_plan(user_id)}
        seen: set[frozenset[str]] = set()

        def fresh(p: Pair) -> bool:
            key = frozenset((p.a.id, p.b.id))
            if key in seen or {p.a.id, p.b.id} & exact_drops:
                return False
            seen.add(key)
            return True

        for op in self._pending_ops(user_id):
            for p in self._op_pairs(user_id, op):
                if fresh(p):
                    yield p
        for p in self._legacy_merge_pairs(user_id):
            if fresh(p):
                yield p
        for p in self._candidate_pairs(user_id, exclude=exact_drops):
            if fresh(p):
                yield p

    # -- pair sources --------------------------------------------------------

    def _user_scope(self, user_id: str) -> str:
        return f"{user_id}/{RESERVED_USER_SLUG}"

    def _active(self, scope: str) -> list[KnowledgeUnit]:
        return [u for u, _ in self._store.scope_units(scope, active=True)]

    def _pending_ops(self, user_id: str) -> list[MemoryFeedbackOp]:
        if self._feedback is None:
            return []
        return [
            op
            for rec in self._feedback.list_feedback(user_id, limit=_BIG)
            for op in self._feedback.list_ops(rec.id)
            if op.status == "pending"
        ]

    def _op_pairs(self, user_id: str, op: MemoryFeedbackOp) -> list[Pair]:
        """The pairs that decide one pending op; [] for correct/invalid ops."""
        try:
            if op.op_type == "tombstone":
                target, scope = _require_unit(self._store, user_id, str(op.target.get("unit_id", "")))
                pools = [scope]
                if scope != self._user_scope(user_id):
                    pools.append(self._user_scope(user_id))
                pairs: list[Pair] = []
                for pool_scope in pools:
                    found = candidate_pairs(
                        self._store, [target], self._active(pool_scope),
                        focus_scope=scope, pool_scope=pool_scope,
                        cosine_threshold=self._cfg.cosine_threshold, fts_neighbours=self._cfg.tombstone_neighbours,
                    )
                    pairs += [
                        Pair(target, scope, c.b, pool_scope, "op", "op:tombstone", op.id, target.id) for c in found
                    ]
                return pairs[: self._cfg.tombstone_neighbours]
            if op.op_type == "merge":
                canonical, scope = _require_unit(self._store, user_id, str(op.target.get("canonical_id", "")))
                dup_ids = op.target.get("duplicate_ids") or []
                if not isinstance(dup_ids, list):
                    return []
                out = []
                for dup_id in dup_ids:
                    if str(dup_id) == canonical.id:
                        continue
                    dup, dup_scope = _require_unit(self._store, user_id, str(dup_id))
                    out.append(Pair(canonical, scope, dup, dup_scope, "op", "op:merge", op.id, dup.id))
                return out
        except InvalidOpError:
            return []
        return []

    def _applied_merges(self, user_id: str) -> list[MemoryFeedbackOp]:
        if self._feedback is None:
            return []
        return [
            op
            for rec in self._feedback.list_feedback(user_id, limit=_BIG)
            for op in self._feedback.list_ops(rec.id)
            if op.op_type == "merge" and op.status == "applied"
        ]

    def _legacy_merge_pairs(self, user_id: str) -> Iterator[Pair]:
        """Duplicates a human merged (soft-invalidated) earlier: re-judged today
        before they are purged."""
        for op in self._applied_merges(user_id):
            canonical = self._store.get_unit_for_user(user_id, str(op.target.get("canonical_id", "")))
            if canonical is None:
                continue
            for dup_id in op.target.get("duplicate_ids") or []:
                dup = self._store.get_unit_for_user(user_id, str(dup_id))
                if dup is None or dup[0].id == canonical[0].id or _is_active(dup[0]):
                    continue
                yield Pair(
                    canonical[0], canonical[1], dup[0], dup[1], "legacy-merge", "applied-merge", op.id, dup[0].id
                )

    def _focus(self, scope: str) -> tuple[list[KnowledgeUnit], list[KnowledgeUnit], ScopeCursor]:
        """(active units, focus units, cursor to persist once the scope is done).

        Focus = units new since the last finished run + a rolling slice of
        older ones, so a large store converges over several runs.
        """
        rows = self._store.scope_units(scope, active=True)
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
        return (
            [u for u, _ in rows],
            new + [u for u, _ in window],
            ScopeCursor(new_watermark=max_rowid, rolling_after=rolling_after),
        )

    def _scope_pairs(self, user_id: str, scope: str, *, exclude: set[str]) -> Iterator[Pair]:
        """Candidate pairs whose focus side is in ``scope``: within the scope,
        and against the user's shared scope (both directions are covered as
        each side takes its turn as ``scope``)."""
        units, focus, _cursor = self._focus(scope)
        user_scope = self._user_scope(user_id)
        pools = [(scope, units)]
        if scope != user_scope:
            pools.append((user_scope, self._active(user_scope)))
        else:
            pools += [(s, self._active(s)) for s in self._store.user_scopes(user_id) if s.startswith(f"{user_id}/")
                      and s != user_scope]
        for pool_scope, pool in pools:
            for c in candidate_pairs(
                self._store, focus, pool, focus_scope=scope, pool_scope=pool_scope,
                cosine_threshold=self._cfg.cosine_threshold, fts_neighbours=self._cfg.fts_neighbours, exclude=exclude,
            ):
                yield Pair(c.a, scope, c.b, pool_scope, "candidate", c.signal)

    def _candidate_pairs(self, user_id: str, *, exclude: set[str]) -> Iterator[Pair]:
        for scope in self._store.user_scopes(user_id):
            yield from self._scope_pairs(user_id, scope, exclude=exclude)

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    def apply(
        self, user_id: str, run_id: str, token: str, verdict: object, *, dry_run: bool = False
    ) -> dict[str, Any]:
        """Verify one verdict and delete the dropped unit if it passes.

        Returns ``{"outcome": "deleted" | "kept" | "stale" | "would_delete", ...}``.

        Raises:
            ReconcileError: malformed run id, token or verdict (nothing happened).
            RunFinishedError: the run already finished.
        """
        _check_run_id(run_id)
        prior = self._log.get_verdict(user_id, token)
        if prior is not None:
            return {**_outcome_view(prior), "replayed": True}
        run = self._log.get_run(user_id, run_id)
        if run is not None and run.finished_at:
            raise RunFinishedError(f"run {run_id!r} already finished")
        match = _TOKEN_RE.match(token)
        if match is None:
            raise ReconcileError("malformed pair_token")
        a_id, b_id, want, op_raw = match.group(1), match.group(2), match.group(3), int(match.group(4))
        try:
            parsed = parse_verdict(verdict)
        except ValueError as exc:
            raise ReconcileError(str(exc)) from exc

        base = {"pair_token": token, "relation": parsed.relation, "replayed": False}
        a = self._store.get_unit_for_user(user_id, a_id)
        b = self._store.get_unit_for_user(user_id, b_id)
        if a is None or b is None:
            return {**base, "outcome": "stale", "reason": "a unit in this pair no longer exists"}
        pair = Pair(a[0], a[1], b[0], b[1], "", "", op_raw or None, None if want == "-" else want)
        if pair_token(user_id, pair) != token:
            return {**base, "outcome": "stale", "reason": "a unit changed since the pair was offered"}

        verified, reason = verify_verdict(parsed, pair.a, pair.b)
        if verified is not None:
            scopes = {pair.a.id: pair.a_scope, pair.b.id: pair.b_scope}
            keep_scope, drop_scope = scopes[verified.keep.id], scopes[verified.drop.id]
            if pair.want_drop is not None and verified.drop.id != pair.want_drop:
                verified, reason = None, "rejected: this pair may only drop " + ("A" if pair.want_drop == a_id else "B")
            elif (why := self._scope_rule(user_id, keep_scope, drop_scope)) is not None:
                verified, reason = None, why
            elif _is_active(verified.drop) and not _is_active(verified.keep):
                verified, reason = None, "rejected: an active memory is never dropped for an inactive one"

        if verified is None:
            outcome = {**base, "outcome": "kept", "reason": reason}
            if not dry_run:
                self._log.record_verdict(
                    user_id=user_id, pair_token=token, run_id=run_id, op_id=pair.op_id, outcome="kept",
                    relation=parsed.relation, reason=reason, kept_unit_id=None, dropped_unit_id=None,
                )
                bucket = reason if reason in ("compatible", "unrelated", "unsure") else "rejected"
                self._log.remember_judged(pair_key(pair.a, pair.b), bucket)
            else:
                self._previewed.setdefault((user_id, run_id), set()).add(token)
            return outcome

        result = {
            **base,
            "kept_unit_id": verified.keep.id,
            "dropped_unit_id": verified.drop.id,
            "reason": "verified",
        }
        if dry_run:
            self._previewed.setdefault((user_id, run_id), set()).add(token)
            return {**result, "outcome": "would_delete"}
        self._prepare_mutation(user_id, run_id)
        self._delete(user_id, run_id, [_Deletion(
            unit=verified.drop, scope=drop_scope, kept=verified.keep, relation=verified.relation,
            source=f"op:{pair.op_id}" if pair.op_id else "judge", evidence=verified.evidence,
        )])
        self._log.record_verdict(
            user_id=user_id, pair_token=token, run_id=run_id, op_id=pair.op_id, outcome="deleted",
            relation=verified.relation, reason="verified", kept_unit_id=verified.keep.id,
            dropped_unit_id=verified.drop.id,
        )
        return {**result, "outcome": "deleted"}

    def verdict(self, user_id: str, token: str) -> dict[str, Any] | None:
        """The recorded outcome for a pair token (the effect reconciler's lookup)."""
        prior = self._log.get_verdict(user_id, token)
        return _outcome_view(prior) if prior else None

    def _scope_rule(self, user_id: str, keep_scope: str, drop_scope: str) -> str | None:
        if keep_scope == drop_scope:
            return None
        user_scope = self._user_scope(user_id)
        if keep_scope == user_scope and drop_scope.startswith(f"{user_id}/"):
            return None
        return "rejected: across scopes only an agent-scope copy may be dropped, in favour of the shared copy"

    # ------------------------------------------------------------------
    # finish
    # ------------------------------------------------------------------

    def finish(self, user_id: str, run_id: str, *, dry_run: bool = False) -> dict[str, Any]:
        """Deterministic deletions, op settlement, cursors, VACUUM, summary.

        Raises:
            ReconcileError: malformed run id.
        """
        _check_run_id(run_id)
        run = self._log.get_run(user_id, run_id)
        if run is not None and run.finished_at and run.summary is not None:
            return {**run.summary, "replayed": True}

        deletions = self._deterministic_plan(user_id)
        ops = self._settle_ops(user_id)
        exclude = {d.unit.id for d in deletions}
        open_tokens = {pair_token(user_id, p) for p in self._open_pairs(user_id)}
        cursors: dict[str, ScopeCursor] = {}
        for scope in self._store.user_scopes(user_id):
            scope_open = any(
                pair_token(user_id, p) in open_tokens for p in self._scope_pairs(user_id, scope, exclude=exclude)
            )
            if not scope_open:
                cursors[scope] = self._focus(scope)[2]
        unclassified = self._legacy_unclassified(user_id, {d.unit.id for d in deletions})

        if dry_run:
            self._previewed.pop((user_id, run_id), None)
            return {
                "user_id": user_id, "run_id": run_id, "dry_run": True,
                "would_delete": [_deletion_view(d) for d in deletions],
                "ops": [{"op_id": o.id, "action": a, "result": r} for o, a, r in ops if a != "pending"],
                "ops_pending": sum(1 for _o, a, _r in ops if a == "pending"),
                "legacy_unclassified_kept": unclassified,
                "open_pairs": len(open_tokens),
            }

        if deletions or any(a != "pending" for _o, a, _r in ops):
            self._prepare_mutation(user_id, run_id)
        self._delete(user_id, run_id, deletions)
        if self._feedback is not None:
            for op, action, result in ops:
                if action == "apply":
                    self._feedback.mark_op(op.id, status="applied", result=f"reconcile: {result}")
                elif action == "dismiss":
                    self._feedback.mark_op(op.id, status="dismissed", result=result)
        for scope, cursor in cursors.items():
            self._log.set_cursor(scope, cursor)

        deleted = self._log.deletions(run_id)
        deleted = [d for d in deleted if d["user_id"] == user_id]
        vacuumed = False
        if deleted:
            self._store.vacuum()
            vacuumed = True
        verdicts = self._log.run_verdicts(user_id, run_id)
        run = self._log.ensure_run(user_id, run_id)
        summary = {
            "user_id": user_id,
            "run_id": run_id,
            "dry_run": False,
            "deleted": len(deleted),
            "by_relation": dict(Counter(d["relation"] for d in deleted)),
            "by_source": dict(Counter(d["source"].split(":", 1)[0] for d in deleted)),
            "verdicts": dict(Counter(v["outcome"] for v in verdicts)),
            "kept_reasons": dict(Counter(_bucket(v["reason"]) for v in verdicts if v["outcome"] == "kept")),
            "ops_applied": sum(1 for _o, a, _r in ops if a == "apply"),
            "ops_dismissed": sum(1 for _o, a, _r in ops if a == "dismiss"),
            "ops_pending": sum(1 for _o, a, _r in ops if a == "pending"),
            "legacy_unclassified_kept": unclassified,
            "open_pairs": len(open_tokens),
            "snapshot": run.snapshot_path,
            "vacuumed": vacuumed,
        }
        self._log.finish_run(user_id, run_id, summary)
        logger.info("memory reconcile %s/%s finished: %s", user_id, run_id, summary)
        return {**summary, "replayed": False}

    def run_summary(self, user_id: str, run_id: str) -> dict[str, Any] | None:
        run = self._log.get_run(user_id, run_id)
        if run is None:
            return None
        return {"user_id": user_id, "run_id": run_id, "started_at": run.started_at,
                "finished_at": run.finished_at, "snapshot": run.snapshot_path, "summary": run.summary}

    def _exact_plan(self, user_id: str) -> list[_Deletion]:
        scoped = [(u, s) for s in self._store.user_scopes(user_id) for u in self._active(s)]
        return [
            _Deletion(unit=d.drop, scope=d.drop_scope, kept=d.keep, relation="duplicate_subsumed", source="exact-match",
                      evidence={"signal": "identical normalised content", "kept_scope": d.keep_scope})
            for d in exact_duplicates(scoped, self._user_scope(user_id))
        ]

    def _correction_plan(self, user_id: str) -> list[_Deletion]:
        """Invalidated units a human-approved ``correct`` op provably superseded:
        a unit recorded as ``correction of <id>`` exists, or a newer sibling
        correction of the same source does."""
        out: list[_Deletion] = []
        for scope in self._store.user_scopes(user_id):
            for unit, _ in self._store.scope_units(scope, active=False):
                successors = self._store.units_by_provenance(scope, f"correction of {unit.id}")
                if not successors and unit.provenance.startswith("correction of "):
                    successors = [
                        s for s in self._store.units_by_provenance(scope, unit.provenance)
                        if s.id != unit.id and recency(s) > recency(unit)
                    ]
                if successors:
                    keep = max(successors, key=recency)
                    out.append(_Deletion(unit=unit, scope=scope, kept=keep, relation="supersedes",
                                         source="legacy:correction", evidence={"provenance": keep.provenance}))
        return out

    def _deterministic_plan(self, user_id: str) -> list[_Deletion]:
        plan: list[_Deletion] = []
        dropped: set[str] = set()
        kept: set[str] = set()
        for d in [*self._exact_plan(user_id), *self._correction_plan(user_id)]:
            if d.unit.id in dropped or d.unit.id in kept or (d.kept and d.kept.id in dropped):
                continue
            dropped.add(d.unit.id)
            if d.kept is not None:
                kept.add(d.kept.id)
            plan.append(d)
        return plan

    def _legacy_unclassified(self, user_id: str, planned: set[str]) -> int:
        """Invalidated units the store cannot prove were superseded (kept)."""
        merged = {
            str(d) for op in self._applied_merges(user_id) for d in (op.target.get("duplicate_ids") or [])
        }
        return sum(
            1
            for scope in self._store.user_scopes(user_id)
            for unit, _ in self._store.scope_units(scope, active=False)
            if unit.id not in planned and unit.id not in merged
        )

    def _settle_ops(self, user_id: str) -> list[tuple[MemoryFeedbackOp, str, str]]:
        """``(op, "apply" | "dismiss" | "pending", result)`` for every pending op."""
        out: list[tuple[MemoryFeedbackOp, str, str]] = []
        for op in self._pending_ops(user_id):
            if op.op_type == "correct":
                out.append((op, "dismiss", CORRECT_DISMISS_REASON))
                continue
            deleted = [v for v in self._log.op_verdicts(user_id, op.id) if v["outcome"] == "deleted"]
            if deleted:
                ids = ", ".join(v["dropped_unit_id"] for v in deleted)
                out.append((op, "apply", f"deleted {ids} (kept {deleted[0]['kept_unit_id']})"))
                continue
            try:
                self._validate_op(user_id, op)
            except InvalidOpError as exc:
                gone = [str(t) for t in _op_unit_ids(op) if self._log.deletions(unit_id=str(t))]
                if gone:
                    out.append((op, "apply", f"already deleted by reconcile: {', '.join(gone)}"))
                else:
                    out.append((op, "dismiss", f"auto-rejected: {exc}"))
                continue
            pairs = self._op_pairs(user_id, op)
            undecided = [
                p for p in pairs
                if self._log.get_verdict(user_id, pair_token(user_id, p)) is None
                and self._log.judged(pair_key(p.a, p.b)) is None
            ]
            if undecided:
                out.append((op, "pending", f"{len(undecided)} pair(s) not judged yet"))
                continue
            reasons = sorted({
                (self._log.get_verdict(user_id, pair_token(user_id, p)) or {}).get("reason")
                or f"previously judged {self._log.judged(pair_key(p.a, p.b))}"
                for p in pairs
            })
            note = "; ".join(reasons) if reasons else "no memory in its scope supersedes or subsumes it"
            out.append((op, "dismiss", f"kept: {note}"))
        return out

    def _validate_op(self, user_id: str, op: MemoryFeedbackOp) -> None:
        if op.op_type == "tombstone":
            _require_unit(self._store, user_id, str(op.target.get("unit_id", "")))
        elif op.op_type == "merge":
            _require_unit(self._store, user_id, str(op.target.get("canonical_id", "")))
            dups = op.target.get("duplicate_ids") or []
            if not isinstance(dups, list) or not dups:
                raise InvalidOpError("merge needs a non-empty duplicate_ids list")
            for dup in dups:
                _require_unit(self._store, user_id, str(dup))
        else:
            raise InvalidOpError(f"unknown op_type {op.op_type!r}")

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------

    def _prepare_mutation(self, user_id: str, run_id: str) -> None:
        """Record the run and snapshot the store before its first mutation."""
        run = self._log.ensure_run(user_id, run_id)
        if run.snapshot_path:
            return
        started = datetime.now(UTC)
        tag = hashlib.sha256(f"{user_id}/{run_id}".encode()).hexdigest()[:8]
        target = self._snapshot_root / f"{started.strftime('%Y%m%dT%H%M%S%fZ')}-{tag}"
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
        self._log.set_snapshot(user_id, run_id, str(target))

    def _delete(self, user_id: str, run_id: str, deletions: list[_Deletion]) -> None:
        by_scope: dict[str, list[_Deletion]] = {}
        for d in deletions:
            by_scope.setdefault(d.scope, []).append(d)
        for scope, items in by_scope.items():
            self._store.purge_units([d.unit.id for d in items], scope)
            for d in items:
                self._log.record_deletion(DeletionRecord(
                    run_id=run_id, unit_id=d.unit.id, agent_id=scope, user_id=user_id,
                    kept_unit_id=d.kept.id if d.kept else None, relation=d.relation, source=d.source,
                    evidence=d.evidence,
                ))


def _check_run_id(run_id: str) -> None:
    if not RUN_ID_RE.match(run_id or ""):
        raise ReconcileError("run_id must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def _op_unit_ids(op: MemoryFeedbackOp) -> list[object]:
    if op.op_type == "tombstone":
        return [op.target.get("unit_id")]
    return [*(op.target.get("duplicate_ids") or [])]


def _bucket(reason: str) -> str:
    return reason if reason in ("compatible", "unrelated", "unsure") else "rejected"


def _outcome_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_token": row["pair_token"],
        "outcome": row["outcome"],
        "relation": row["relation"],
        "reason": row["reason"],
        "kept_unit_id": row["kept_unit_id"],
        "dropped_unit_id": row["dropped_unit_id"],
        "run_id": row["run_id"],
    }


def _deletion_view(d: _Deletion) -> dict[str, Any]:
    return {
        "unit_id": d.unit.id,
        "scope": d.scope,
        "content": d.unit.content,
        "kept_unit_id": d.kept.id if d.kept else None,
        "kept_content": d.kept.content if d.kept else None,
        "relation": d.relation,
        "source": d.source,
    }


__all__ = ["CORRECT_DISMISS_REASON", "ReconcileError", "ReconcileService", "RunFinishedError", "pair_token"]
