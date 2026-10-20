"""Memory endpoints for agent runs executed outside Recordatio's runner.

Two routes let an external orchestrator give an agent the same memory the
in-process agent runner does, without routing the turn through Recordatio:

- ``GET  /v1/users/{user_id}/briefing?slug=<slug>&query=<text>`` — the exact
  briefing text the agent briefing hook would inject for that agent scope and
  query, as ``{"briefing": str, "units": int}``. Read-only: no briefing-event
  row, no ``retrieval_count`` bumps. Recall keys may call it (see
  ``network_guard``).
- ``POST /v1/agents/{user_id}/{slug}/transcripts`` — hand over an already
  finished exchange (``input``/``output``) for the same background extraction
  the agent extraction hook runs. Idempotent on ``idempotency_key``: the first
  submission answers ``202`` (extraction scheduled), a repeat answers ``200``
  with ``"duplicate": true`` and schedules nothing. Write path, so recall keys
  and restricted-network callers are refused by ``network_guard``.

The briefing and extraction callables are injected by ``server.py`` so both
routes share the hooks' code rather than re-implementing it.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tpasini.recordatio.proxy.agent_endpoints import _require_auth, _reserved_slug_response
from tpasini.recordatio.proxy.transcript_log import TranscriptLog

# (user_id, slug, query) -> (briefing text or None, number of units rendered)
type BriefingPreview = Callable[[str, str, str], Awaitable[tuple[str | None, int]]]
# (user_id, slug, transcript) -> None; same signature as the agent extraction hook
type TranscriptExtraction = Callable[[str, str, dict], Awaitable[None]]

_TRANSCRIPT_FIELDS = ("input", "output", "occurred_at", "source", "idempotency_key")


def register_agent_memory_routes(
    app: FastAPI,
    *,
    preview_briefing: BriefingPreview,
    extract_transcript: TranscriptExtraction,
    transcript_log: TranscriptLog,
) -> None:
    """Register the briefing-preview and transcript-ingest routes on ``app``."""

    @app.get("/v1/users/{user_id}/briefing")
    async def get_briefing(user_id: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        slug = request.query_params.get("slug", "").strip()
        if not slug:
            return JSONResponse({"error": "slug is required"}, status_code=400)
        if (reserved := _reserved_slug_response(slug)) is not None:
            return reserved
        query = request.query_params.get("query", "").strip()
        if not query:
            return JSONResponse({"error": "query is required"}, status_code=400)
        briefing, units = await preview_briefing(user_id, slug, query)
        return JSONResponse({"briefing": briefing or "", "units": units})

    @app.post("/v1/agents/{user_id}/{slug}/transcripts")
    async def post_transcript(user_id: str, slug: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        if (reserved := _reserved_slug_response(slug)) is not None:
            return reserved
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        for name in _TRANSCRIPT_FIELDS:
            if not isinstance(body.get(name), str):
                return JSONResponse({"error": f"{name} must be a string"}, status_code=400)
        for name in ("occurred_at", "source", "idempotency_key"):
            if not body[name].strip():
                return JSONResponse({"error": f"{name} must not be empty"}, status_code=400)
        if not (body["input"].strip() or body["output"].strip()):
            return JSONResponse({"error": "input and output are both empty"}, status_code=400)
        try:
            datetime.fromisoformat(body["occurred_at"])
        except ValueError:
            return JSONResponse({"error": "occurred_at must be ISO-8601"}, status_code=400)

        key = body["idempotency_key"]
        if not transcript_log.claim(user_id, slug, key, source=body["source"], occurred_at=body["occurred_at"]):
            return JSONResponse({"idempotency_key": key, "duplicate": True}, status_code=200)
        # Same transcript shape the agent runner hands the extraction hook.
        transcript = {
            "messages": [{"role": "user", "content": body["input"]}],
            "response_text": body["output"],
        }
        try:
            await extract_transcript(user_id, slug, transcript)
        except Exception:
            transcript_log.release(user_id, slug, key)
            raise
        return JSONResponse({"idempotency_key": key, "duplicate": False}, status_code=202)
