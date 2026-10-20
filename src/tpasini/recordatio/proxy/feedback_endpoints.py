"""Read + feedback endpoints for the nightly feedback analyst.

Four routes, all per-user and auth-gated like the agent endpoints:

- ``GET  /v1/users/{user_id}/units``          — a day's extracted units
- ``GET  /v1/users/{user_id}/memory-search``  — counterfactual recall probe
- ``POST /v1/users/{user_id}/memory-feedback``— store a night's memory feedback
- ``GET  /v1/users/{user_id}/memory-feedback``— read feedback + op lifecycle

``memory-search`` has two modes. ``production`` replays the real briefing
retrieval (BM25 ⊕ embedding fusion + traversal + budget) so the analyst can
ask "would this have been recalled if ranking/budget had done its job?";
``raw`` is a tokenized LIKE scan (every content word must appear somewhere in
the unit, in any order) that asks "does this fact exist at all?". Both
honour ``as_of`` so a counterfactual never surfaces a memory that post-dates
the analysed turn — though ``production`` replays against the *current* graph,
so its ranking verdict is flagged counterfactual-current-index.
"""
import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tpasini.recordatio.identity import RESERVED_USER_SLUG
from tpasini.recordatio.models import KnowledgeUnit
from tpasini.recordatio.proxy import retrieval
from tpasini.recordatio.proxy.agent_endpoints import _require_auth
from tpasini.recordatio.proxy.memory_ops import InvalidOpError, MemoryOpError, apply_memory_op
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore

logger = logging.getLogger(__name__)

# Reversible, low-risk op classes safe to apply without human review: a
# `tombstone` only sets valid_until (soft delete), and a `merge` only
# tombstones the duplicates while leaving the canonical untouched. The
# `correct` class is deliberately excluded — it supersedes content and can be
# genuinely ambiguous, so it always waits for a human decision.
_AUTO_APPLY_OP_TYPES = frozenset({"tombstone", "merge"})


def _auto_apply_safe_ops(
    store: SQLiteStore,
    memory_feedback_store: MemoryFeedbackStore,
    user_id: str,
    feedback_id: int,
) -> dict[str, int]:
    """Immediately apply the reversible safe-class ops of a freshly recorded
    feedback record, mirroring the dashboard's ``apply_all_pending`` safety
    envelope: a malformed op (bad/truncated uuid, out of scope) is auto-rejected
    via ``InvalidOpError`` and any other ``MemoryOpError`` leaves the op pending.
    ``correct`` ops and anything that errors are left for human review.

    Returns a small counter summary for the response/logs.
    """
    applied = rejected = 0
    for op in memory_feedback_store.list_ops(feedback_id):
        if op.status != "pending" or op.op_type not in _AUTO_APPLY_OP_TYPES:
            continue
        try:
            result = apply_memory_op(store, user_id, op.op_type, op.target)
        except InvalidOpError as exc:
            memory_feedback_store.mark_op(op.id, status="dismissed", result=f"auto-rejected: {exc}")
            rejected += 1
            continue
        except MemoryOpError:
            # Transient/non-malformed failure — leave pending for the dashboard.
            continue
        memory_feedback_store.mark_op(op.id, status="applied", result=result)
        applied += 1
    return {"applied": applied, "rejected": rejected}


def _unit_dict(unit: KnowledgeUnit, agent_id: str | None = None) -> dict[str, object]:
    """Serialise a unit for the wire (drops nothing the analyst needs)."""
    out: dict[str, object] = {
        "id": unit.id,
        "type": unit.type.value,
        "content": unit.content,
        "asserted_at": unit.asserted_at.isoformat(),
        "valid_from": unit.valid_from.isoformat(),
        "valid_until": unit.valid_until.isoformat() if unit.valid_until else None,
        "provenance": unit.provenance,
        "extraction_count": unit.extraction_count,
        "retrieval_count": unit.retrieval_count,
    }
    if agent_id is not None:
        out["agent_id"] = agent_id
    return out


def _day_bounds(day: str) -> tuple[str, str]:
    """Return ``[start, end)`` ISO bounds for a ``YYYY-MM-DD`` local-naive day.

    Raises ``ValueError`` on a malformed date.
    """
    d = date.fromisoformat(day)
    start = datetime(d.year, d.month, d.day)
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def _agent_scope(user_id: str, slug: str | None) -> str:
    """Resolve the retrieval scope: a named agent, else the user scope."""
    if slug:
        return f"{user_id}/{slug}"
    return f"{user_id}/{RESERVED_USER_SLUG}"


def register_feedback_routes(
    app: FastAPI,
    *,
    get_store: Callable[[], SQLiteStore],
    memory_feedback_store: MemoryFeedbackStore,
    user_label: str,
    auto_apply_safe_ops: bool = False,
) -> None:
    """Register the nightly read + feedback routes on ``app``.

    ``get_store`` yields the canonical memory store; ``user_label`` is the
    entity label the production retrieval path is seeded with (same value the
    server passes to ``retrieval.retrieve``).

    When ``auto_apply_safe_ops`` is set, the reversible safe-class ops
    (tombstone / merge) of each recorded feedback record are applied as soon as
    the analyst POSTs it, so the op queue is drained automatically instead of
    waiting for a manual dashboard click. ``correct`` ops are always left
    pending for human review.
    """

    @app.get("/v1/users/{user_id}/units")
    async def list_units(user_id: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        day = request.query_params.get("day")
        try:
            after, before = _day_bounds(day) if day else (None, None)
        except ValueError:
            return JSONResponse({"error": "day must be YYYY-MM-DD"}, status_code=400)
        try:
            limit = min(int(request.query_params.get("limit", "200")), 1000)
        except ValueError:
            return JSONResponse({"error": "limit must be an integer"}, status_code=400)
        pairs = get_store().list_units_for_user(
            user_id, asserted_after=after, asserted_before=before, limit=limit
        )
        return JSONResponse(
            {
                "user_id": user_id,
                "day": day,
                "count": len(pairs),
                "units": [_unit_dict(u, agent_id) for u, agent_id in pairs],
            }
        )

    @app.get("/v1/users/{user_id}/memory-search")
    async def memory_search(user_id: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        query = request.query_params.get("query", "").strip()
        if not query:
            return JSONResponse({"error": "query is required"}, status_code=400)
        mode = request.query_params.get("mode", "production")
        if mode not in ("production", "raw"):
            return JSONResponse({"error": "mode must be production|raw"}, status_code=400)
        slug = request.query_params.get("slug") or None
        as_of = request.query_params.get("as_of") or None
        if as_of is not None:
            try:
                datetime.fromisoformat(as_of)
            except ValueError:
                return JSONResponse({"error": "as_of must be ISO-8601"}, status_code=400)
        try:
            k = min(int(request.query_params.get("k", "10")), 100)
        except ValueError:
            return JSONResponse({"error": "k must be an integer"}, status_code=400)

        store = get_store()
        scope = _agent_scope(user_id, slug)
        caveat: str | None = None
        if mode == "raw":
            pairs = store.raw_search_units_for_user(user_id, query, limit=k, as_of=as_of)
            results = [_unit_dict(u, agent_id) for u, agent_id in pairs]
        else:
            result = await retrieval.retrieve(query, scope, store, logger, user_label)
            units: list[KnowledgeUnit] = list(result.units)
            if as_of is not None:
                units = [u for u in units if u.asserted_at.isoformat() <= as_of]
            results = [_unit_dict(u, scope) for u in units[:k]]
            caveat = "counterfactual-current-index: production ranking replays against the current graph"
        return JSONResponse(
            {
                "user_id": user_id,
                "query": query,
                "mode": mode,
                "scope": scope,
                "as_of": as_of,
                "count": len(results),
                "caveat": caveat,
                "results": results,
            }
        )

    @app.post("/v1/users/{user_id}/memory-feedback")
    async def post_memory_feedback(user_id: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        day = body.get("day")
        if not isinstance(day, str):
            return JSONResponse({"error": "day is required"}, status_code=400)
        try:
            date.fromisoformat(day)
        except ValueError:
            return JSONResponse({"error": "day must be YYYY-MM-DD"}, status_code=400)
        summary = body.get("summary")
        if not isinstance(summary, str):
            return JSONResponse({"error": "summary is required"}, status_code=400)
        findings = body.get("findings", [])
        ops = body.get("ops", [])
        if not isinstance(findings, list) or not isinstance(ops, list):
            return JSONResponse({"error": "findings and ops must be arrays"}, status_code=400)
        try:
            feedback_id = memory_feedback_store.record(
                user_id=user_id, day=day, summary=summary, findings=findings, ops=ops
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        payload: dict[str, object] = {"id": feedback_id}
        if auto_apply_safe_ops:
            counts = _auto_apply_safe_ops(get_store(), memory_feedback_store, user_id, feedback_id)
            payload["auto_applied"] = counts["applied"]
            payload["auto_rejected"] = counts["rejected"]
            if counts["applied"] or counts["rejected"]:
                logger.info(
                    "auto-applied safe memory-feedback ops for %s/%s: applied=%d rejected=%d",
                    user_id, day, counts["applied"], counts["rejected"],
                )
        return JSONResponse(payload)

    @app.get("/v1/users/{user_id}/memory-feedback")
    async def get_memory_feedback(user_id: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        records = memory_feedback_store.list_feedback(user_id)
        payload = []
        for rec in records:
            ops = memory_feedback_store.list_ops(rec.id)
            payload.append(
                {
                    "id": rec.id,
                    "day": rec.day,
                    "created_at": rec.created_at,
                    "summary": rec.summary,
                    "findings": rec.findings,
                    "ops": [
                        {
                            "id": op.id,
                            "op_type": op.op_type,
                            "target": op.target,
                            "reason": op.reason,
                            "status": op.status,
                            "decided_at": op.decided_at,
                            "decided_result": op.decided_result,
                        }
                        for op in ops
                    ],
                }
            )
        return JSONResponse({"user_id": user_id, "records": payload})

    def _load_pending_op(user_id: str, op_id: int) -> tuple[object, JSONResponse | None]:
        """Fetch a pending op that belongs to ``user_id``; on any problem
        return ``(None, error_response)`` with the right status code."""
        op = memory_feedback_store.get_op(op_id)
        if op is None:
            return None, JSONResponse({"error": "op not found"}, status_code=404)
        rec = memory_feedback_store.get_feedback(op.feedback_id)
        if rec is None or rec.user_id != user_id:
            # Don't leak another user's op ids — 404, not 403.
            return None, JSONResponse({"error": "op not found"}, status_code=404)
        if op.status != "pending":
            return None, JSONResponse(
                {"error": f"op already {op.status}", "status": op.status}, status_code=409
            )
        return op, None

    @app.post("/v1/users/{user_id}/memory-feedback/ops/{op_id}/apply")
    async def apply_op(user_id: str, op_id: int, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        op, err = _load_pending_op(user_id, op_id)
        if err is not None:
            return err
        try:
            result = apply_memory_op(get_store(), user_id, op.op_type, op.target)
        except InvalidOpError as exc:
            # Invalid suggestion (incomplete uuid, unknown unit, malformed) —
            # can never be applied, so auto-reject rather than leave it pending.
            memory_feedback_store.mark_op(op_id, status="dismissed", result=f"auto-rejected: {exc}")
            return JSONResponse({"ok": True, "op_id": op_id, "status": "rejected", "reason": str(exc)})
        except MemoryOpError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        memory_feedback_store.mark_op(op_id, status="applied", result=result)
        return JSONResponse({"ok": True, "op_id": op_id, "status": "applied", "result": result})

    @app.post("/v1/users/{user_id}/memory-feedback/ops/{op_id}/dismiss")
    async def dismiss_op(user_id: str, op_id: int, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        op, err = _load_pending_op(user_id, op_id)
        if err is not None:
            return err
        reason = None
        try:
            body = await request.json()
            if isinstance(body, dict) and isinstance(body.get("reason"), str):
                reason = body["reason"]
        except Exception:
            pass  # dismiss carries no required body
        memory_feedback_store.mark_op(op_id, status="dismissed", result=reason)
        return JSONResponse({"ok": True, "op_id": op_id, "status": "dismissed"})


__all__ = ["register_feedback_routes"]
