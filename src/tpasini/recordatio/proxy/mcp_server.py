"""MCP server exposing Recordatio as a tool surface for any MCP-capable agent.

Mounted on the main FastAPI app at ``/mcp`` (see ``server.py``). Two tools:

- ``recollect(query)`` — return memories relevant to ``query`` for the agent
  namespace identified by the ``X-Agent-Id`` request header.
- ``extract(messages, response_text, background=False)`` — run the extraction
  LLM on both sides of a conversation turn, persist the results under the same
  ``X-Agent-Id`` namespace, and return a summary of what was saved.

``agent_id`` deliberately comes from a header (not a tool argument): every
common MCP client (Claude Code, Claude Desktop, Cursor, the SDK clients) lets
the operator configure static per-server headers at registration time, so the
calling LLM cannot address another agent's namespace by spoofing an argument.

``server.py`` builds this FastMCP via :func:`create_mcp`, passing in the
helpers it needs (``_retrieve``, ``_extract_and_store``, ``_background``,
``settings``). That dependency-injection wiring keeps this module from
back-referencing ``server.py`` and avoids both a circular import and the
need for in-function lazy imports.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, ConfigDict

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.extraction.extractor import ConversationTurn, ExtractionResult
from tpasini.recordatio.models import KnowledgeUnit, RenderedRelation
from tpasini.recordatio.proxy.retrieval import RetrievalResult


class RecollectResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    topic_hint: str | None
    units: list[KnowledgeUnit]
    entities: list[Entity]
    relations: list[RenderedRelation]


class ExtractResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    persisted: bool
    queued: bool = False
    input: ExtractionResult | None = None
    output: ExtractionResult | None = None


class _Settings(Protocol):
    recordatio_extraction_api_key: str | None


def _agent_id_from(ctx: Context) -> str:
    request = ctx.request_context.request
    if request is None:
        raise ValueError("MCP tool invoked without an HTTP request context")
    agent_id = request.headers.get("x-agent-id")
    if not agent_id:
        raise ValueError("X-Agent-Id header is required")
    return agent_id


def _source_agent_id_from(ctx: Context, agent_id: str) -> str:
    request = ctx.request_context.request
    if request is None:
        raise ValueError("MCP tool invoked without an HTTP request context")
    return request.headers.get("x-recordatio-source-agent-id") or agent_id


def create_mcp(
    *,
    retrieve: Callable[[str, str], Awaitable[RetrievalResult]],
    extract_and_store: Callable[..., Awaitable[dict[str, ExtractionResult | None]]],
    schedule_background: Callable[[Awaitable], object],
    settings: _Settings,
    allowed_hosts: list[str] | None = None,
) -> FastMCP:
    """Build the FastMCP server with tools wired to the given proxy helpers.

    Helpers are injected (not imported) so this module does not need to
    reach into ``server.py``. ``server.py`` calls this from the bottom of
    its file once all the helpers exist.
    """
    mcp = FastMCP("recordatio", stateless_http=True, transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts or ["127.0.0.1:*", "localhost:*", "[::1]:*"],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
    ))

    @mcp.tool()
    async def recollect(query: str, ctx: Context) -> RecollectResponse:
        """Return memories relevant to ``query`` for the calling agent's namespace.

        The agent namespace is taken from the ``X-Agent-Id`` request header
        (configured once per agent in the MCP client's server registration).
        """
        agent_id = _agent_id_from(ctx)
        result = await retrieve(query, agent_id)
        return RecollectResponse(
            topic_hint=result.topic_hint,
            units=result.units,
            entities=result.entities,
            relations=result.rendered_relations,
        )

    @mcp.tool()
    async def extract(
        messages: list[dict],
        response_text: str,
        ctx: Context,
        background: bool = False,
    ) -> ExtractResponse:
        """Mine knowledge from a conversation turn and persist it to memory.

        Runs extraction on both the input side (user/system/prior-assistant
        messages) and the output side (the final assistant response),
        persists the resulting units, entity mentions, attributes, relation
        candidates, topics, and schema proposals to the agent's namespace,
        and records the LLM cost in the cost log.

        With ``background=False`` (default) the call awaits the full
        pipeline and returns the saved payload. With ``background=True`` it
        schedules the work fire-and-forget and returns immediately —
        useful when the caller doesn't need the saved summary in the same
        turn.
        """
        agent_id = _agent_id_from(ctx)
        source_agent_id = _source_agent_id_from(ctx, agent_id)
        headers = ctx.request_context.request.headers
        api_key = headers.get("x-recordatio-api-key") or settings.recordatio_extraction_api_key
        user_id = headers.get("x-user-id")
        turn = ConversationTurn(messages=messages, response_text=response_text)

        if background:
            schedule_background(
                extract_and_store(
                    turn,
                    api_key,
                    agent_id,
                    user_id=user_id,
                    parent_request_id=None,
                    source_agent_id=source_agent_id,
                )
            )
            return ExtractResponse(persisted=False, queued=True)

        per_side = await extract_and_store(
            turn,
            api_key,
            agent_id,
            user_id=user_id,
            parent_request_id=None,
            source_agent_id=source_agent_id,
        )
        return ExtractResponse(
            persisted=True,
            input=per_side.get("input"),
            output=per_side.get("output"),
        )

    return mcp


def mount_mcp(app: FastAPI, mcp: FastMCP) -> None:
    """Start the mounted MCP manager in the parent app's lifespan.

    ASGI does not run mounted subapp lifespans. Compose with the existing
    parent lifespan so its startup, shutdown and warm-pool cleanup still run.
    """
    http_app = mcp.streamable_http_app()
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[dict | None]:
        async with previous_lifespan(application) as state, mcp.session_manager.run():
            yield state

    app.router.lifespan_context = lifespan
    app.mount("/mcp", http_app)
