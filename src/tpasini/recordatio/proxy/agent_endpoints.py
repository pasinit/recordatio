"""FastAPI router for the new agent endpoints.

Three endpoints, all per (user_id, slug):

- ``POST /v1/agents/{user_id}/{slug}/messages`` — SSE stream of TaskEvents.
- ``GET  /v1/agents/{user_id}/{slug}/config``   — current AgentConfig.
- ``PUT  /v1/agents/{user_id}/{slug}/config``   — update; 409 on version mismatch.

The router is built via ``register_agent_routes()`` and mounted from
``server.py`` so the heavy module-level construction (Pontifex factory,
briefing hook closure, extraction hook) stays where it belongs and the
router itself is reusable in tests with a stub runner.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import suppress

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tpasini.recordatio.agents.config_store import AgentConfig, AgentConfigStore, AgentConfigVersionMismatch
from tpasini.recordatio.agents.events import TaskEvent, serialize_sse
from tpasini.recordatio.agents.runner import AgentRunner
from tpasini.recordatio.identity import RESERVED_USER_SLUG, ensure_principal, is_reserved_slug
from tpasini.recordatio.storage.base import SQLiteStore

logger = logging.getLogger(__name__)

# Seconds of TaskEvent silence before the SSE stream emits a ``: keepalive``
# comment line. Long turns have legitimately silent stretches — the model
# generating tool-call inputs or subagent work translates to no
# client-visible event — during which clients with per-read timeouts
# conclude the stream is dead and abort mid-turn (telos did, at 600s,
# truncating routine emails to the pre-tool intro text). SSE parsers skip
# comment lines by spec, so the keepalive is invisible to consumers; it
# only resets their read timers and keeps intermediaries from idling out
# the connection.
_KEEPALIVE_INTERVAL_S = 15.0


def _reserved_slug_response(slug: str) -> JSONResponse | None:
    """The ``_user`` slug names the user-level identity store
    (IDENTITY_DESIGN.md §1) — it is never a real agent."""
    if is_reserved_slug(slug):
        return JSONResponse(
            {"error": f"agent slug {RESERVED_USER_SLUG!r} is reserved for the user-level store"},
            status_code=400,
        )
    return None


def register_user_routes(app: FastAPI, *, get_store: Callable[[], SQLiteStore]) -> None:
    """``PUT /v1/users/{user_id}/principal`` — the login-time ensure seam
    (IDENTITY_DESIGN.md §5). Idempotent: creates the principal only if absent;
    upserts display_name/aliases through the collision-guarded alias path on
    every call. Telos calls this when it mints the user and on login; the
    extraction ensure-on-write converges on the same code path."""

    @app.put("/v1/users/{user_id}/principal")
    async def put_principal(user_id: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        body_bytes = await request.body()
        if body_bytes:
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        else:
            body = {}
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
        display_name = body.get("display_name")
        if display_name is not None and not isinstance(display_name, str):
            return JSONResponse({"error": "display_name must be a string"}, status_code=400)
        aliases = body.get("aliases")
        if aliases is not None and (
            not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases)
        ):
            return JSONResponse({"error": "aliases must be a list of strings"}, status_code=400)
        principal, created = ensure_principal(
            get_store(), user_id, display_name=display_name, aliases=aliases
        )
        return JSONResponse(
            {
                "entity_id": principal.id,
                "user_scope": f"{user_id}/{RESERVED_USER_SLUG}",
                "created": created,
            }
        )


def register_agent_routes(
    app: FastAPI,
    *,
    runner: AgentRunner,
    config_store: AgentConfigStore,
) -> None:
    """Register the three Phase 1 agent endpoints on ``app``.

    No prefix wrapping — we want the routes to appear at the spec'd
    paths verbatim. The router accesses ``runner`` and ``config_store``
    via the closures captured below, so the caller is free to swap
    either of them at construction time without modifying this module.

    All three endpoints require a caller-supplied API key (``x-api-key``
    header or ``Authorization: Bearer …``). This mirrors the chat
    endpoint's contract: Recordatio is a credential pass-through, so an
    unauthenticated request is necessarily one that could never have
    served work anyway — fail fast at the edge so unauthenticated
    callers can't read or mutate agent configs either.
    """

    @app.get("/v1/agents/{user_id}/{slug}/config")
    async def get_config(user_id: str, slug: str, request: Request) -> JSONResponse:
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        config = config_store.get(user_id, slug)
        if config is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(config.to_dict())

    @app.put("/v1/agents/{user_id}/{slug}/config")
    async def put_config(user_id: str, slug: str, request: Request) -> JSONResponse:
        if (reserved := _reserved_slug_response(slug)) is not None:
            return reserved
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

        config = AgentConfig.from_dict(body)
        try:
            stored = config_store.put(user_id, slug, config)
        except AgentConfigVersionMismatch as exc:
            # 409 Conflict carries both versions so the improver can
            # re-fetch and re-apply its diff against the current row
            # without a second round-trip.
            return JSONResponse(
                {
                    "error": "version mismatch",
                    "expected": exc.expected,
                    "actual": exc.actual,
                },
                status_code=409,
            )
        return JSONResponse(stored.to_dict())

    @app.post("/v1/agents/{user_id}/{slug}/messages", response_model=None)
    async def post_messages(user_id: str, slug: str, request: Request) -> StreamingResponse | JSONResponse:
        if (reserved := _reserved_slug_response(slug)) is not None:
            return reserved
        if (unauthorized := _require_auth(request)) is not None:
            return unauthorized
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

        content = body.get("content")
        if not isinstance(content, str) or not content:
            return JSONResponse({"error": "content is required"}, status_code=400)

        # Optional full conversation history (OpenAI-style role/content
        # dicts, latest user message last). Telos ships this alongside the
        # flat ``content`` field so multi-turn context survives the
        # per-turn fresh SDK session — ignoring it was the per-turn-amnesia
        # P0. Absent ⇒ legacy single-turn behaviour, unchanged. Present but
        # malformed ⇒ 400 (a caller that bothers to send history deserves
        # to learn it was garbage rather than silently losing memory).
        messages = body.get("messages")
        if messages is not None:
            error = _validate_messages(messages)
            if error is not None:
                return JSONResponse({"error": error}, status_code=400)

        # Call-lineage fields — preserved verbatim and threaded through to
        # the extraction hook so cross-agent provenance survives. Mirrors
        # the ``x-recordatio-source-agent-id`` header behaviour on the chat
        # path; we accept them in the JSON body here because the agent
        # endpoint is invoked by other agents (curia, sibylla) that
        # already serialize their request id and origin agent slug.
        raw_source = body.get("source_agent_id")
        source_agent_id = raw_source if isinstance(raw_source, str) and raw_source else None
        raw_parent = body.get("parent_request_id")
        parent_request_id = raw_parent if isinstance(raw_parent, str) and raw_parent else None

        # Per-query config: telos sends the authoritative AgentConfig inline
        # on every turn (system prompt, model, mcp servers, allowed_tools) so
        # the agent's live state can never drift from a seeded copy. When
        # present it wins; absent, fall back to the stored config (legacy /
        # other callers). The inline path REQUIRES an explicit allowed_tools —
        # we never fall back to the SDK's wide-open default for a
        # caller-supplied config, so a bug that omits the tool list fails
        # loudly here instead of silently handing the agent every native tool.
        inline_config = body.get("agent_config")
        if inline_config is not None:
            if not isinstance(inline_config, dict):
                return JSONResponse({"error": "agent_config must be an object"}, status_code=400)
            if inline_config.get("allowed_tools") is None:
                return JSONResponse(
                    {"error": "agent_config.allowed_tools is required (explicit tool list)"},
                    status_code=400,
                )
            try:
                config = AgentConfig.from_dict(inline_config)
            except Exception as exc:  # malformed shape
                return JSONResponse({"error": f"invalid agent_config: {exc}"}, status_code=400)
        else:
            config = config_store.get(user_id, slug)
            if config is None:
                return JSONResponse(
                    {"error": "agent config not found — PUT /v1/agents/{user_id}/{slug}/config first"},
                    status_code=404,
                )

        api_key = _api_key_from_request(request)

        async def stream() -> AsyncIterator[bytes]:
            # The runner generator must be iterated by ONE task for its whole
            # life: ``runner.run`` holds an anyio task group (a cancel scope)
            # open across its yields, and anyio requires a scope to exit in
            # the task that entered it. Driving ``anext()`` from a fresh task
            # per keepalive window tripped "Attempted to exit cancel scope in
            # a different task than it was entered in" on long multi-window
            # turns. So a single drain task consumes the generator into a
            # queue, and the keepalive timeout applies to the queue read.
            queue: asyncio.Queue[TaskEvent | None] = asyncio.Queue()

            async def _drain() -> None:
                try:
                    async for event in runner.run(
                        user_id=user_id,
                        slug=slug,
                        content=content,
                        config=config,
                        api_key=api_key,
                        source_agent_id=source_agent_id,
                        parent_request_id=parent_request_id,
                        messages=messages,
                    ):
                        await queue.put(event)
                finally:
                    # Sentinel wakes the reader on normal end, error, and
                    # cancellation alike. Unbounded queue ⇒ put() has no
                    # await point, so it is safe in this finally.
                    await queue.put(None)

            drain = asyncio.create_task(_drain())
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_INTERVAL_S)
                    except TimeoutError:
                        yield b": keepalive\n\n"
                        continue
                    if event is None:
                        # Drain ended — awaiting it re-raises a runner error
                        # so the connection aborts exactly as the pre-queue
                        # ``async for`` did (the client sees truncation).
                        await drain
                        return
                    yield serialize_sse(event)
            finally:
                # Client disconnects cancel this generator mid-await; the
                # drain task (which holds the runner's SDK turn) must not
                # outlive it. Cancelling it unwinds runner.run inside the
                # drain task itself, keeping the cancel scope single-task.
                if not drain.done():
                    drain.cancel()
                    with suppress(asyncio.CancelledError):
                        await drain

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def _validate_messages(messages: object) -> str | None:
    """Return an error string when ``messages`` is malformed, else ``None``.

    Accepted shape (mirrors what telos's ``stream_task_events`` sends):
    a list of dicts each carrying a string ``role`` and a ``content``
    that is either a string or a block list (image turns store content
    as ``[{"type": ..., ...}, ...]``). Blocks themselves are validated
    only as dicts — the transports degrade unknown block types to an
    attachment note rather than rejecting the turn.
    """
    if not isinstance(messages, list):
        return "messages must be a list"
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return f"messages[{idx}] must be an object"
        role = msg.get("role")
        if not isinstance(role, str) or not role:
            return f"messages[{idx}].role must be a non-empty string"
        content = msg.get("content")
        if isinstance(content, str):
            continue
        if isinstance(content, list):
            if all(isinstance(block, dict) for block in content):
                continue
            return f"messages[{idx}].content blocks must be objects"
        return f"messages[{idx}].content must be a string or a list of blocks"
    return None


def _require_auth(request: Request) -> JSONResponse | None:
    """Return a 401 ``JSONResponse`` when no caller credential is present.

    Mirrors the chat endpoint's gate: Recordatio is a credential
    pass-through, so the inbound API key (either ``x-api-key`` or
    ``Authorization: Bearer …``) is the auth signal. Missing key ⇒
    request can't possibly serve real work ⇒ 401 at the edge so the
    config endpoints can't be probed unauthenticated.
    """
    if _api_key_from_request(request) is None:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


def _api_key_from_request(request: Request) -> str | None:
    """Pull a fallback API key from headers. Mirrors the existing chat
    endpoint so a caller can authorise both endpoints uniformly."""
    key = request.headers.get("x-api-key")
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth[7:]
    return key or None
