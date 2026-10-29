"""HTTP API for the memory reconcile (``reconcile.service``), driven by a judge agent.

All routes are per user and auth-gated like the feedback endpoints; a
reconcile key (``proxy/network_guard.py``) may call only these.

- ``GET  /v1/users/{user_id}/memory-reconcile/candidates?run_id=…&limit=…``
  — read-only: the next batch of undecided pairs, each with a ``pair_token``.
- ``POST /v1/users/{user_id}/memory-reconcile/verdicts``
  ``{"run_id", "pair_token", "verdict", "dry_run"?}`` — verify one verdict and
  delete the dropped unit if it passes. Idempotent per pair token: a replay
  returns the recorded outcome (``replayed: true``). ``dry_run`` verifies only.
- ``GET  /v1/users/{user_id}/memory-reconcile/verdicts/{pair_token}`` — the
  recorded outcome, 404 ``verdict_not_found`` when none (the effect
  reconciler's lookup).
- ``POST /v1/users/{user_id}/memory-reconcile/runs/{run_id}/finish[?dry_run=1]``
  — deterministic deletions, op settlement, cursors, VACUUM, summary.
  Idempotent per run id.
- ``GET  /v1/users/{user_id}/memory-reconcile/runs/{run_id}`` — the run row,
  404 ``run_not_found`` when the run never mutated or finished.
"""
import logging
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tpasini.recordatio.proxy.agent_endpoints import _require_auth
from tpasini.recordatio.reconcile import ReconcileError, ReconcileService, RunFinishedError

logger = logging.getLogger(__name__)

_TRUE = frozenset({"1", "true", "yes", "on"})


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in _TRUE


def register_reconcile_routes(app: FastAPI, *, get_service: Callable[[], ReconcileService]) -> None:
    """Register the reconcile routes; ``get_service`` yields the long-lived service."""

    def _error(exc: ReconcileError) -> JSONResponse:
        if isinstance(exc, RunFinishedError):
            return JSONResponse({"error": str(exc), "code": "run_finished"}, status_code=409)
        return JSONResponse({"error": str(exc), "code": "invalid_request"}, status_code=422)

    @app.get("/v1/users/{user_id}/memory-reconcile/candidates")
    async def reconcile_candidates(user_id: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        try:
            limit = int(request.query_params.get("limit", "10"))
        except ValueError:
            return JSONResponse({"error": "limit must be an integer", "code": "invalid_request"}, status_code=422)
        try:
            return JSONResponse(get_service().candidates(user_id, request.query_params.get("run_id", ""), limit=limit))
        except ReconcileError as exc:
            return _error(exc)

    @app.post("/v1/users/{user_id}/memory-reconcile/verdicts")
    async def reconcile_verdict(user_id: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body", "code": "invalid_request"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object", "code": "invalid_request"}, status_code=400)
        dry_run = _truthy(body.get("dry_run")) or _truthy(request.query_params.get("dry_run"))
        try:
            outcome = get_service().apply(
                user_id, str(body.get("run_id", "")), str(body.get("pair_token", "")), body.get("verdict"),
                dry_run=dry_run,
            )
        except ReconcileError as exc:
            return _error(exc)
        return JSONResponse({**outcome, "dry_run": dry_run})

    @app.get("/v1/users/{user_id}/memory-reconcile/verdicts/{pair_token}")
    async def reconcile_verdict_lookup(user_id: str, pair_token: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        found = get_service().verdict(user_id, pair_token)
        if found is None:
            return JSONResponse({"error": "no verdict recorded for this pair token", "code": "verdict_not_found"},
                                status_code=404)
        return JSONResponse(found)

    @app.post("/v1/users/{user_id}/memory-reconcile/runs/{run_id}/finish")
    async def reconcile_finish(user_id: str, run_id: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        try:
            dry_run = _truthy(request.query_params.get("dry_run"))
            return JSONResponse(get_service().finish(user_id, run_id, dry_run=dry_run))
        except ReconcileError as exc:
            return _error(exc)

    @app.get("/v1/users/{user_id}/memory-reconcile/runs/{run_id}")
    async def reconcile_run(user_id: str, run_id: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        found = get_service().run_summary(user_id, run_id)
        if found is None:
            return JSONResponse({"error": "no such run", "code": "run_not_found"}, status_code=404)
        return JSONResponse(found)


__all__ = ["register_reconcile_routes"]
