"""HTTP trigger for the nightly memory reconcile (``reconcile.runner``).

- ``POST /v1/memory-reconcile``                   — every user
- ``POST /v1/users/{user_id}/memory-reconcile``   — one user

Auth-gated exactly like the feedback endpoints. Query flags:
``dry_run=true`` reports what would be deleted/applied/dismissed and why
without mutating anything; ``details`` (default: on for dry runs, off for
real runs) includes the per-unit deletions and per-op decisions in the
response — a real run's summary otherwise carries counts only, the detail
being in the audit table. One run at a time: a concurrent call gets 409.

The judge's API key follows extraction: the configured reconcile/extraction
key, else the caller's pass-through key.
"""
import asyncio
import logging
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tpasini.recordatio.proxy.agent_endpoints import _api_key_from_request, _require_auth
from tpasini.recordatio.reconcile import ReconcileSettings, build_reconciler, make_judge, reconcile_log_path
from tpasini.recordatio.reconcile.judge import PairJudge
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore
from tpasini.recordatio.storage.reconcile_log import ReconcileLog

logger = logging.getLogger(__name__)

_TRUE = frozenset({"1", "true", "yes", "on"})


def _flag(request: Request, name: str, default: bool) -> bool:
    raw = request.query_params.get(name)
    return default if raw is None else raw.strip().lower() in _TRUE


def register_reconcile_routes(
    app: FastAPI,
    *,
    get_store: Callable[[], SQLiteStore],
    memory_feedback_store: MemoryFeedbackStore,
    settings_factory: Callable[[], ReconcileSettings] = ReconcileSettings,
    judge_factory: Callable[[ReconcileSettings, str | None], PairJudge] | None = None,
) -> None:
    """Register the reconcile trigger routes on ``app``.

    ``judge_factory(settings, caller_key)`` builds the judge (tests inject a
    fake); the default is the configured LLM judge.
    """
    lock = asyncio.Lock()
    logs: dict[str, ReconcileLog] = {}

    def _log_for(store: SQLiteStore) -> ReconcileLog:
        path = reconcile_log_path(store.db_path)
        if str(path) not in logs:
            logs[str(path)] = ReconcileLog(path)
        return logs[str(path)]

    async def _run(request: Request, user_ids: list[str] | None) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        if lock.locked():
            return JSONResponse({"error": "a reconcile run is already in progress"}, status_code=409)
        dry_run = _flag(request, "dry_run", False)
        details = _flag(request, "details", dry_run)
        settings = settings_factory()
        try:
            judge = (judge_factory or (lambda s, k: make_judge(s, api_key=k)))(
                settings, _api_key_from_request(request)
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        async with lock:
            store = get_store()
            reconciler = build_reconciler(
                store, judge=judge, settings=settings, feedback=memory_feedback_store, log=_log_for(store)
            )
            try:
                report = await reconciler.run(user_ids, dry_run=dry_run, details=details)
            except Exception as exc:
                logger.exception("memory reconcile failed")
                return JSONResponse({"error": f"{exc.__class__.__name__}: {exc}"}, status_code=500)
        return JSONResponse(report)

    @app.post("/v1/memory-reconcile")
    async def reconcile_all(request: Request) -> JSONResponse:
        return await _run(request, None)

    @app.post("/v1/users/{user_id}/memory-reconcile")
    async def reconcile_user(user_id: str, request: Request) -> JSONResponse:
        return await _run(request, [user_id])


__all__ = ["register_reconcile_routes"]
