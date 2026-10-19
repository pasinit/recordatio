"""agentmem proxy server — memory-augmented LLM proxy.

Exposes POST /v1/chat/completions (OpenAI-compatible).
Injects a [MEMORY] briefing into the system prompt, routes the request through
Pontifex, streams the response back, and schedules background knowledge extraction.

Model routing uses a "provider/model" convention in the request's model field:
  anthropic/claude-opus-4-6     → Anthropic backend
  openai/gpt-4o                 → OpenAI backend
  ollama/qwen2.5:14b            → OpenAI-compatible, http://localhost:11434/v1
  gemini/gemini-2.0-flash       → OpenAI-compatible, Google's endpoint
  lmstudio/llama-3              → OpenAI-compatible, http://localhost:1234/v1
  gpt-4o (no prefix)            → OpenAI backend (fallback)

Configuration (env vars):
  RECORDATIO_EXTRACTION_MODEL  — extraction model in provider/model format (required)
  RECORDATIO_EXTRACTION_API_KEY — API key for extraction provider (if different from upstream)
  AGENTMEM_DB               — SQLite DB path (default: ~/.agentmem/memories.db)
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Coroutine
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pontifex import Pontifex

from tpasini.recordatio import identity
from tpasini.recordatio.agents import (
    AgentConfigStore,
    AgentRunner,
    ApiKeyTransport,
    RateLimitStateStore,
    WarmPool,
)
from tpasini.recordatio.agents.providers import api_fallback_enabled, subscription_transport
from tpasini.recordatio.entity import Entity
from tpasini.recordatio.extraction.extractor import (
    ConversationTurn,
    ExtractionResult,
    ExtractionSide,
    extract_knowledge,
    transcript_to_turn,
)
from tpasini.recordatio.models import (
    Relation,
    RenderedRelation,
)
from tpasini.recordatio.observability import init_tracing, mark_error, span
from tpasini.recordatio.proxy import extraction_persistence, extraction_pipeline, network_guard, prompting, retrieval
from tpasini.recordatio.proxy import runtime as _runtime
from tpasini.recordatio.proxy.agent_endpoints import register_agent_routes, register_user_routes
from tpasini.recordatio.proxy.briefing_log import BriefingLog
from tpasini.recordatio.proxy.briefing_routes import register_briefing_event_routes
from tpasini.recordatio.proxy.cost_log import CostLog
from tpasini.recordatio.proxy.dashboard_routes import register_dashboard_routes
from tpasini.recordatio.proxy.feedback_endpoints import register_feedback_routes
from tpasini.recordatio.proxy.pricing import lookup_cost as _lookup_cost
from tpasini.recordatio.proxy.runtime import (
    ExtractionUsageBuffer,
    Settings,
)
from tpasini.recordatio.proxy.runtime import (
    count_messages_tokens as _count_messages_tokens,
)
from tpasini.recordatio.proxy.runtime import (
    count_tokens as _count_tokens,
)
from tpasini.recordatio.proxy.runtime import (
    parse_upstream_usage as _parse_upstream_usage,
)
from tpasini.recordatio.proxy.runtime import (
    provider_tag as _provider_tag,
)
from tpasini.recordatio.proxy.runtime import (
    resolve_model as _resolve_model,
)
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore
from tpasini.recordatio.wikidata import resolve_entity

logger = logging.getLogger(__name__)

type _ExtractionStoreResult = dict[str, ExtractionResult | None]
_ExtractionUsage = _runtime.ExtractionUsage

settings = Settings()


_extraction_bare_model, _EXTRACTION_BACKEND, _EXTRACTION_BASE_URL = _resolve_model(
    settings.recordatio_extraction_model
)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_DB_PATH = Path(settings.agentmem_db)
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
store = SQLiteStore(_DB_PATH)

_COST_DB_PATH = _DB_PATH.parent / "cost_events.db"
cost_log = CostLog(_COST_DB_PATH)

# Per-turn briefing events — separate file so cost dashboards and briefing
# lookups never compete for the same write lock.
_BRIEFING_DB_PATH = _DB_PATH.parent / "briefing_events.db"
briefing_log = BriefingLog(_BRIEFING_DB_PATH)

# Phase 1: agent-runner state. Same threading model as the other sidecar
# SQLite stores — one connection + a threading.Lock + WAL.
_AGENT_CONFIG_DB_PATH = _DB_PATH.parent / "agent_configs.db"
_AGENT_RATE_LIMIT_DB_PATH = _DB_PATH.parent / (
    "agent_rate_limits.db" if settings.recordatio_subscription_provider == "claude" else "agent_rate_limits_codex.db"
)
agent_config_store = AgentConfigStore(_AGENT_CONFIG_DB_PATH)
agent_rate_state = RateLimitStateStore(_AGENT_RATE_LIMIT_DB_PATH)

# Nightly memory-feedback records + their apply/dismiss op lifecycle.
_MEMORY_FEEDBACK_DB_PATH = _DB_PATH.parent / "memory_feedback.db"
memory_feedback_store = MemoryFeedbackStore(_MEMORY_FEEDBACK_DB_PATH)


_USER_ENTITY_LABEL = "User"
_bootstrapped_agents: set[str] = set()


# ---------------------------------------------------------------------------
# Extraction-cost buffer (piggybacked onto chat responses)
# ---------------------------------------------------------------------------

_extraction_usage_buffer = ExtractionUsageBuffer()
_unreported_extractions = _extraction_usage_buffer.pending
_unreported_extractions_lock = _extraction_usage_buffer.lock


async def _drain_extraction_usage(agent_id: str | None) -> dict | None:
    return await _extraction_usage_buffer.drain(agent_id)


def _ensure_user_entity(agent_id: str) -> None:
    """Ensure-on-write safety net (IDENTITY_DESIGN.md §5): the principal lives
    in the user-level store, never as a bare agent-scope ``User`` entity."""
    user_id = agent_id.partition("/")[0]
    if (
        agent_id in _bootstrapped_agents
        and identity.get_principal(store, identity.user_scope(agent_id)) is not None
    ):
        return
    identity.ensure_principal(store, user_id)
    _bootstrapped_agents.add(agent_id)


# ---------------------------------------------------------------------------
# App + background task helpers
# ---------------------------------------------------------------------------

app = FastAPI(title="agentmem")
init_tracing(app, settings.otel_exporter_otlp_endpoint, settings.otel_service_name)
network_guard.install(app, settings.recordatio_restricted_networks, settings.recordatio_recall_key_sha256)

_background_tasks: set[asyncio.Task] = set()


def _background[T](coro: Coroutine[None, None, T]) -> asyncio.Task[T]:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _bump_retrieval_counts(selected_ids: prompting.SelectedIds | None, agent_id: str) -> None:
    """Bump read-side ``retrieval_count`` for items rendered into a briefing.

    Off the turn's critical path (scheduled via ``_background``). Bumps exactly
    the SELECTED/rendered ids — units, entities, relation ROW ids — scoped to
    the retrieval ``agent_id``. Dropped candidates and pruned relations are not
    present in ``selected_ids`` and so are never bumped.
    """
    if selected_ids is None or selected_ids.is_empty:
        return
    try:
        for kind, ids in (
            ("unit", selected_ids.units),
            ("entity", selected_ids.entities),
            ("relation", selected_ids.relations),
        ):
            if ids:
                store.increment_retrieval_count(kind, ids, agent_id)
    except Exception:
        logger.exception("failed to bump retrieval_count for agent %s", agent_id)


# Serialise extraction calls to avoid rate-limit bursts.
_extraction_sem = asyncio.Semaphore(1)


# ---------------------------------------------------------------------------
# Built-in routes
# ---------------------------------------------------------------------------

@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}



_dashboard_route_handlers = register_dashboard_routes(
    app,
    get_store=lambda: store,
    get_cost_log=lambda: cost_log,
    get_feedback_store=lambda: memory_feedback_store,
)
memories_ui = _dashboard_route_handlers.memories_ui
costs_ui = _dashboard_route_handlers.costs_ui
costs_data = _dashboard_route_handlers.costs_data
memories_agents = _dashboard_route_handlers.memories_agents
memories = _dashboard_route_handlers.memories
register_briefing_event_routes(
    app,
    get_store=lambda: store,
    get_briefing_log=lambda: briefing_log,
    logger=logger,
)



# ---------------------------------------------------------------------------
# Briefing injection
# ---------------------------------------------------------------------------

def _extract_system(messages: list[dict]) -> prompting.ExtractedSystem:
    return prompting.extract_system(messages)


def _strip_reminders(text: str) -> str:
    return prompting.strip_reminders(text)


def _last_user_text(messages: list[dict]) -> str | None:
    return prompting.last_user_text(messages)


def _infer_topic_hint(query: str) -> str | None:
    return retrieval.infer_topic_hint(query, store)


def _get_entity_by_id(entity_id: str, agent_id: str) -> Entity | None:
    return retrieval.get_entity_by_id(entity_id, agent_id, store)


def _get_unit_content(unit_id: str, fallback_units: list, agent_id: str) -> str | None:
    return retrieval.get_unit_content(unit_id, fallback_units, agent_id, store)


def _seed_entities_local(query: str, agent_id: str) -> list[Entity]:
    return retrieval.seed_entities_local(query, agent_id, store, _USER_ENTITY_LABEL)


def _clean_query(query: str) -> str:
    return retrieval.clean_query(query)


async def _wikidata_seed_entities(query: str, exclude_ids: set[str], agent_id: str) -> list[Entity]:
    return await retrieval.wikidata_seed_entities(query, exclude_ids, agent_id, store, logger)


def _traverse(seed_entities: list[Entity], units: list, agent_id: str) -> retrieval.TraversalResult:
    return retrieval.traverse(seed_entities, units, agent_id, store, logger)


def _render_relations_from(
    relations: list[Relation],
    entity_by_id: dict[str, Entity],
    units: list,
    agent_id: str,
) -> list[RenderedRelation]:
    return retrieval.render_relations_from(relations, entity_by_id, units, agent_id, store)


async def _retrieve(
    query: str,
    agent_id: str,
    source_agent_id: str | None = None,
) -> retrieval.RetrievalResult:
    return await retrieval.retrieve(query, agent_id, store, logger, _USER_ENTITY_LABEL, source_agent_id)


async def _retrieve_profile(
    agent_id: str,
    *,
    now: datetime | None = None,
) -> retrieval.RetrievalResult:
    return await retrieval.retrieve_profile(
        agent_id, store, _USER_ENTITY_LABEL, logger, now=now,
    )


async def _build_system(
    existing_system: str | None,
    query: str,
    agent_id: str,
    backend: str = "openai",
    call_id: str | None = None,
    source_agent_id: str | None = None,
) -> prompting.SystemBuildResult:
    async def retrieve_with_source(
        retrieve_query: str,
        retrieve_agent_id: str,
        retrieve_source_agent_id: str | None,
    ) -> retrieval.RetrievalResult:
        return await _retrieve(retrieve_query, retrieve_agent_id, retrieve_source_agent_id)

    return await prompting.build_system(
        existing_system=existing_system,
        query=query,
        agent_id=agent_id,
        backend=backend,
        call_id=call_id,
        source_agent_id=source_agent_id,
        retrieve=retrieve_with_source,
        retrieve_profile=_retrieve_profile,
        briefing_log=briefing_log,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Chat completions endpoint
# ---------------------------------------------------------------------------

def _api_key_from_request(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth[7:]
    return key or None


def _extraction_messages(messages: list[dict], extract_hint: dict | None) -> list[dict]:
    last = extract_hint.get("last_user_message") if extract_hint else None
    if isinstance(last, str) and last:
        return [{"role": "user", "content": last}]
    return messages


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    try:
        body: dict = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    agent_id: str | None = request.headers.get("x-agent-id", "").strip() or None
    # The ``_user`` slug is the reserved user-level store (IDENTITY_DESIGN.md
    # §1): only the routing layer, the ensure API, and the migration write it.
    if agent_id is not None and identity.is_user_scope(agent_id):
        return JSONResponse(
            {"error": f"agent slug {identity.RESERVED_USER_SLUG!r} is reserved for the user-level store"},
            status_code=400,
        )
    source_agent_id: str | None = request.headers.get("x-recordatio-source-agent-id", "").strip() or agent_id
    user_id: str | None = request.headers.get("x-user-id", "").strip() or None

    # Recordatio-only fields — strip before routing upstream.
    extract_hint: dict | None = body.pop("x-recordatio-extract-from", None)
    no_briefing: bool = bool(body.pop("x-recordatio-no-briefing", False))
    skip_input_extraction: bool = bool(body.pop("x-recordatio-skip-input-extraction", False))
    skip_output_extraction: bool = bool(body.pop("x-recordatio-skip-output-extraction", False))

    upstream_api_key = _api_key_from_request(request)
    # Extraction key: use the dedicated setting, or fall back to the upstream
    # key when both providers are the same.
    extraction_api_key: str | None = settings.recordatio_extraction_api_key or upstream_api_key

    # Separate system from conversation messages.
    raw_messages: list[dict] = body.get("messages") or []
    extracted_system = _extract_system(raw_messages)
    existing_system = extracted_system.system
    messages = extracted_system.messages

    # Resolve backend before constructing Pontifex.
    raw_model: str | None = body.get("model") or None
    if not raw_model:
        return JSONResponse({"error": "model is required"}, status_code=400)
    model, backend, base_url = _resolve_model(raw_model)

    # Generate the call_id up front so _build_system can key the persisted
    # briefing row by it, and so the chat-row + extraction-row that follow
    # share the same request_id.
    call_id = f"recordatio-{uuid.uuid4().hex[:12]}"
    provider_tag = _provider_tag(backend, base_url)

    # Build briefing-augmented system prompt.
    system: str | None = existing_system
    briefing_data: dict | None = None
    if not no_briefing and agent_id:
        query = _last_user_text(messages)
        if query:
            system_result = await _build_system(
                existing_system,
                query,
                agent_id,
                backend=backend,
                call_id=call_id,
                source_agent_id=source_agent_id,
            )
            system = system_result.system
            briefing_data = system_result.briefing_data
            # Off-critical-path: bump retrieval_count for the items that
            # actually made it into the rendered briefing (scoped to the
            # retrieval agent_id).
            _background(_bump_retrieval_counts(system_result.selected_ids, agent_id))

    max_tokens: int = body.get("max_tokens", 8096)
    tools: list[dict] | None = body.get("tools")
    is_streaming: bool = bool(body.get("stream", False))

    pf = Pontifex(
        model=model,
        backend=backend,
        base_url=base_url,
        api_key=upstream_api_key,
    )

    logger.info(
        "chat_completions agent=%s source_agent=%s backend=%s model=%s streaming=%s",
        agent_id, source_agent_id, backend, model, is_streaming,
    )

    # Request usage from OpenAI-compatible upstreams via stream_options.
    extra_body: dict | None = None
    if backend == "openai" and is_streaming:
        extra_body = {"stream_options": {"include_usage": True}}

    if is_streaming:
        return StreamingResponse(
            _stream_sse(pf, messages, system, tools, max_tokens,
                        raw_messages, extraction_api_key, agent_id, user_id,
                        extract_hint, briefing_data, extra_body,
                        call_id, provider_tag,
                        skip_input_extraction, skip_output_extraction,
                        source_agent_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return await _non_streaming_response(
        pf, messages, system, tools, max_tokens,
        raw_messages, extraction_api_key, agent_id, user_id,
        extract_hint, call_id, provider_tag,
        skip_input_extraction, skip_output_extraction,
        source_agent_id,
    )


async def _stream_sse(
    pf: Pontifex,
    messages: list[dict],
    system: str | None,
    tools: list[dict] | None,
    max_tokens: int,
    orig_messages: list[dict],
    extraction_api_key: str | None,
    agent_id: str | None,
    user_id: str | None,
    extract_hint: dict | None,
    briefing_data: dict | None,
    extra_body: dict | None = None,
    call_id: str | None = None,
    provider_tag: str | None = None,
    skip_input_extraction: bool = False,
    skip_output_extraction: bool = False,
    source_agent_id: str | None = None,
) -> AsyncIterator[bytes]:
    if briefing_data is not None:
        payload = json.dumps({"type": "agentmem.briefing", **briefing_data})
        yield f"event: agentmem.briefing\ndata: {payload}\n\n".encode()

    if call_id is None:
        call_id = f"recordatio-{uuid.uuid4().hex[:12]}"
    text_chunks: list[str] = []
    upstream_usage: dict | None = None

    with span(
        "pontifex.stream",
        **{
            "llm.model": pf.model,
            "llm.provider": provider_tag,
            "recordatio.request_id": call_id,
            "recordatio.agent_id": agent_id,
            "recordatio.source_agent_id": source_agent_id,
            "recordatio.stream": True,
        },
    ) as sp:
        try:
            async for chunk in pf.stream(
                messages=messages,
                system=system,
                tools=tools,
                max_tokens=max_tokens,
                extra_body=extra_body,
            ):
                for choice in chunk.get("choices") or []:
                    text = choice.get("delta", {}).get("content")
                    if text:
                        text_chunks.append(text)
                # Capture usage from the final chunk (OpenAI sends it when stream_options is set).
                if chunk.get("usage"):
                    upstream_usage = chunk["usage"]
                chunk.setdefault("id", call_id)
                chunk.setdefault("object", "chat.completion.chunk")
                chunk.setdefault("model", pf.model)
                yield f"data: {json.dumps(chunk)}\n\n".encode()
        except Exception as exc:
            mark_error(sp, str(exc), exc)
            logger.exception("stream error")

    # Build usage metadata.
    full_text = "".join(text_chunks)
    if upstream_usage:
        input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens = (
            _parse_upstream_usage(upstream_usage)
        )
    else:
        input_tokens = _count_messages_tokens(messages, system, pf.model)
        output_tokens = _count_tokens(full_text, pf.model)
        cache_creation_tokens = 0
        cache_read_tokens = 0
    cost = _lookup_cost(
        pf.model, input_tokens, output_tokens,
        cache_creation_tokens, cache_read_tokens,
    )
    usage_block: dict = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache_creation_tokens:
        usage_block["cache_creation_tokens"] = cache_creation_tokens
    if cache_read_tokens:
        usage_block["cache_read_tokens"] = cache_read_tokens
    if cost is not None:
        usage_block["cost_usd"] = cost
    # Piggyback the previous turn's extraction cost(s) for this agent — the
    # buffer is filled by `_extract_and_store` after a prior chat completed.
    extraction_fragment = await _drain_extraction_usage(agent_id)
    if extraction_fragment:
        usage_block.update(extraction_fragment)
    usage_data = {"type": "agentmem.usage", "usage": usage_block}
    # Keep the legacy top-level cost_usd field for backward compatibility.
    if cost is not None:
        usage_data["cost_usd"] = cost
    yield f"event: agentmem.usage\ndata: {json.dumps(usage_data)}\n\n".encode()

    logger.info(
        "stream complete: %d chars, input=%d output=%d tokens%s",
        len(full_text), input_tokens, output_tokens,
        f", cost=${cost:.6f}" if cost is not None else "",
    )

    try:
        cost_log.record_chat(
            model=pf.model,
            provider=provider_tag,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_usd=cost,
            agent_id=agent_id,
            source_agent_id=source_agent_id,
            user_id=user_id,
            request_id=call_id,
        )
    except Exception:
        logger.exception("cost-log write failed; continuing")

    yield b"data: [DONE]\n\n"

    turn = ConversationTurn(
        messages=_extraction_messages(orig_messages, extract_hint),
        response_text=full_text,
    )
    _background(_extract_and_store(
        turn, extraction_api_key, agent_id, user_id, call_id,
        skip_input_extraction=skip_input_extraction,
        skip_output_extraction=skip_output_extraction,
        source_agent_id=source_agent_id,
    ))


async def _non_streaming_response(
    pf: Pontifex,
    messages: list[dict],
    system: str | None,
    tools: list[dict] | None,
    max_tokens: int,
    orig_messages: list[dict],
    extraction_api_key: str | None,
    agent_id: str | None,
    user_id: str | None,
    extract_hint: dict | None,
    call_id: str | None = None,
    provider_tag: str | None = None,
    skip_input_extraction: bool = False,
    skip_output_extraction: bool = False,
    source_agent_id: str | None = None,
) -> JSONResponse:
    text_chunks: list[str] = []
    tool_calls: list[dict] = []
    finish_reason = "stop"
    upstream_usage: dict | None = None

    with span(
        "pontifex.stream",
        **{
            "llm.model": pf.model,
            "llm.provider": provider_tag,
            "recordatio.request_id": call_id,
            "recordatio.agent_id": agent_id,
            "recordatio.source_agent_id": source_agent_id,
            "recordatio.stream": False,
        },
    ) as sp:
        try:
            async for chunk in pf.stream(messages=messages, system=system, tools=tools, max_tokens=max_tokens):
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text_chunks.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        tool_calls.append(tc)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                if chunk.get("usage"):
                    upstream_usage = chunk["usage"]
        except Exception as exc:
            mark_error(sp, str(exc), exc)
            logger.exception("non-streaming error")
            return JSONResponse({"error": "upstream error"}, status_code=502)

    full_text = "".join(text_chunks)
    if upstream_usage:
        input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens = (
            _parse_upstream_usage(upstream_usage)
        )
    else:
        input_tokens = _count_messages_tokens(messages, system, pf.model)
        output_tokens = _count_tokens(full_text, pf.model)
        cache_creation_tokens = 0
        cache_read_tokens = 0
    cost = _lookup_cost(
        pf.model, input_tokens, output_tokens,
        cache_creation_tokens, cache_read_tokens,
    )

    message: dict = {"role": "assistant", "content": full_text}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage: dict = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache_creation_tokens:
        usage["cache_creation_tokens"] = cache_creation_tokens
    if cache_read_tokens:
        usage["cache_read_tokens"] = cache_read_tokens
    if cost is not None:
        usage["cost_usd"] = cost
    # Piggyback the previous turn's extraction cost(s) for this agent — the
    # buffer is filled by `_extract_and_store` after a prior chat completed.
    extraction_fragment = await _drain_extraction_usage(agent_id)
    if extraction_fragment:
        usage.update(extraction_fragment)
    if call_id is None:
        call_id = f"recordatio-{uuid.uuid4().hex[:12]}"
    response: dict = {
        "id": call_id,
        "object": "chat.completion",
        "model": pf.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }
    # Keep the legacy top-level cost_usd field for backward compatibility.
    if cost is not None:
        response["cost_usd"] = cost

    logger.info(
        "non-streaming complete: %d chars, input=%d output=%d tokens%s",
        len(full_text), input_tokens, output_tokens,
        f", cost=${cost:.6f}" if cost is not None else "",
    )

    turn = ConversationTurn(
        messages=_extraction_messages(orig_messages, extract_hint),
        response_text=full_text,
    )
    _background(_extract_and_store(
        turn, extraction_api_key, agent_id, user_id, call_id,
        skip_input_extraction=skip_input_extraction,
        skip_output_extraction=skip_output_extraction,
        source_agent_id=source_agent_id,
    ))

    try:
        cost_log.record_chat(
            model=pf.model,
            provider=provider_tag,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_usd=cost,
            agent_id=agent_id,
            source_agent_id=source_agent_id,
            user_id=user_id,
            request_id=call_id,
        )
    except Exception:
        logger.exception("cost-log write failed; continuing")

    return JSONResponse(response)


# ---------------------------------------------------------------------------
# Extraction + storage
# ---------------------------------------------------------------------------

def _extraction_pipeline_context() -> extraction_pipeline.ExtractionPipelineContext:
    return extraction_pipeline.ExtractionPipelineContext(
        store=store,
        cost_log=cost_log,
        extraction_sem=_extraction_sem,
        unreported_extractions=_unreported_extractions,
        unreported_extractions_lock=_unreported_extractions_lock,
        extraction_bare_model=_extraction_bare_model,
        extraction_backend=_EXTRACTION_BACKEND,
        extraction_base_url=_EXTRACTION_BASE_URL,
        user_entity_label=_USER_ENTITY_LABEL,
        bootstrapped_agents=_bootstrapped_agents,
        extract_knowledge=extract_knowledge,
        resolve_entity=resolve_entity,
        ensure_user_entity=_ensure_user_entity,
    )


def _norm_content(text: str) -> str:
    return extraction_pipeline._norm_content(text)


def _merge_cross_side(primary: ExtractionResult, secondary: ExtractionResult) -> ExtractionResult:
    return extraction_pipeline._merge_cross_side(primary, secondary)


async def _extract_and_store(
    turn: ConversationTurn,
    api_key: str | None,
    agent_id: str | None,
    user_id: str | None = None,
    parent_request_id: str | None = None,
    source_agent_id: str | None = None,
    *,
    skip_input_extraction: bool = False,
    skip_output_extraction: bool = False,
) -> dict[str, ExtractionResult | None]:
    return await extraction_pipeline._extract_and_store(
        _extraction_pipeline_context(),
        turn,
        api_key,
        agent_id,
        user_id,
        parent_request_id,
        source_agent_id,
        skip_input_extraction=skip_input_extraction,
        skip_output_extraction=skip_output_extraction,
    )


def _record_skipped_extraction_side(
    *,
    side: ExtractionSide,
    agent_id: str | None,
    user_id: str | None,
    parent_request_id: str | None,
    source_agent_id: str | None = None,
) -> None:
    return extraction_pipeline._record_skipped_extraction_side(
        _extraction_pipeline_context(),
        side=side,
        agent_id=agent_id,
        user_id=user_id,
        parent_request_id=parent_request_id,
        source_agent_id=source_agent_id,
    )


async def _extract_side(
    turn: ConversationTurn,
    api_key: str | None,
    agent_id: str,
    user_id: str | None,
    parent_request_id: str | None,
    *,
    side: ExtractionSide,
    source_agent_id: str | None = None,
) -> ExtractionResult | None:
    return await extraction_pipeline._extract_side(
        _extraction_pipeline_context(),
        turn,
        api_key,
        agent_id,
        user_id,
        parent_request_id,
        source_agent_id,
        side=side,
    )



async def _persist_side(
    result: ExtractionResult,
    agent_id: str,
    source_agent_id: str | None = None,
) -> None:
    await extraction_persistence.persist_side(
        _extraction_pipeline_context(),
        result,
        agent_id,
        source_agent_id=source_agent_id,
    )



# ---------------------------------------------------------------------------
# Phase 1: agent-runner mount.
# Wired at module bottom because the briefing + extraction hooks below need
# to close over helpers defined earlier (_build_system, _extract_and_store).
# ---------------------------------------------------------------------------


async def _agent_briefing_hook(
    user_id: str,
    slug: str,
    content: str,
    base_system_prompt: str,
) -> tuple[str | None, dict | None]:
    """Hydrate the per-turn memory briefing for the agent runner.

    Mirrors the chat-completions path: hydrate the briefing for the
    last user message and persist a briefing-events row keyed by an
    agent-flavoured request id. Unlike the chat path the briefing TEXT
    is returned bare (not combined into the system prompt): the runner
    threads it into the rendered prompt so the system prompt stays
    static and pre-warmed SDK sessions — sealed at connect — stay valid.

    The agent_id is the composite ``f"{user_id}/{slug}"`` — exactly the
    ``x-agent-id`` header telos sends on the chat-completions path
    (``telos/agent.py``: ``self._agent_id = f"{user_id}/{self._slug}"``) —
    so the agent path reads/writes the same per-agent, per-user memory
    scope as legacy chat history. ``source_agent_id`` falls back to that
    same composite, mirroring the chat path's
    ``x-recordatio-source-agent-id or agent_id`` default for a direct user
    turn.
    """
    agent_id = f"{user_id}/{slug}"
    call_id = f"recordatio-agent-{uuid.uuid4().hex[:12]}"
    system_result = await _build_system(
        base_system_prompt or None,
        content,
        agent_id,
        backend="anthropic",
        call_id=call_id,
        source_agent_id=agent_id,
    )
    # Off-critical-path: bump retrieval_count for the items that actually made
    # it into the rendered briefing (scoped to the same composite agent_id used
    # for retrieval).
    _background(_bump_retrieval_counts(system_result.selected_ids, agent_id))
    briefing_data = system_result.briefing_data
    if briefing_data is not None:
        # telos keys the recalled-memories panel off this id: it persists it
        # on the turn row and later hydrates from /v1/briefing-events/{id}.
        # The legacy chat path gets the id from completion-chunk ids; the
        # agent path has no chunk ids, so the briefing event carries it.
        briefing_data = {**briefing_data, "request_id": call_id}
    return system_result.briefing_text, briefing_data


async def _agent_extraction_hook(user_id: str, slug: str, transcript: dict) -> None:
    """Schedule background knowledge extraction from a completed agent turn.

    Reuses ``_extract_and_store`` (the same code path the chat-completions
    endpoint uses) so dedup / persistence logic stays in one place. The
    transcript dict shape is adapted to ``ConversationTurn`` via the
    extractor's ``transcript_to_turn`` helper.

    Lifts ``source_agent_id`` and ``parent_request_id`` off the transcript
    (the runner copies them off the inbound request body) so cross-agent
    call lineage is preserved end-to-end — matches the
    ``x-recordatio-source-agent-id`` header behaviour on the chat path,
    including its fallback to the agent's own id when absent.

    The agent_id is the composite ``f"{user_id}/{slug}"``, matching the
    ``x-agent-id`` header convention telos uses on the chat path — see
    ``_agent_briefing_hook``.
    """
    turn = transcript_to_turn(transcript)
    agent_id = f"{user_id}/{slug}"
    source_agent_id = transcript.get("source_agent_id") or agent_id
    parent_request_id = transcript.get("parent_request_id")
    _background(_extract_and_store(
        turn,
        None,  # api_key — extraction uses its dedicated configured key
        agent_id,
        user_id,
        parent_request_id,
        source_agent_id,
    ))


def _agent_pontifex_factory(model: str, api_key: str | None) -> Pontifex:
    """Build a Pontifex client for the ApiKeyTransport fallback path.

    Routes the agent's ``model`` field (e.g. ``"anthropic/claude-opus-4-7"``)
    through the same provider registry the chat-completions endpoint uses.
    """
    bare_model, backend, base_url = _resolve_model(model)
    return Pontifex(model=bare_model, backend=backend, base_url=base_url, api_key=api_key)


# Warm-spare pool: pre-connected SDK sessions keyed by (user, slug, config
# version) so an active conversation skips the per-turn subprocess spawn +
# control init + MCP handshakes. Sizing/TTL via RECORDATIO_WARM_POOL_MAX /
# RECORDATIO_WARM_POOL_TTL_S (set MAX to 0 to disable warming entirely).
agent_warm_pool = WarmPool()
_agent_subscription_transport = subscription_transport(
    settings.recordatio_subscription_provider,
    rate_state=agent_rate_state,
    warm_pool=agent_warm_pool,
)
_agent_api_key_transport = ApiKeyTransport(pontifex_factory=_agent_pontifex_factory)
agent_runner = AgentRunner(
    subscription=_agent_subscription_transport,
    api_key=_agent_api_key_transport,
    rate_state=agent_rate_state,
    briefing_hook=_agent_briefing_hook,
    extraction_hook=_agent_extraction_hook,
    allow_api_fallback=api_fallback_enabled(
        settings.recordatio_subscription_provider, settings.recordatio_agent_api_fallback,
    ),
)
register_agent_routes(app, runner=agent_runner, config_store=agent_config_store)
register_user_routes(app, get_store=lambda: store)
register_feedback_routes(
    app,
    get_store=lambda: store,
    memory_feedback_store=memory_feedback_store,
    user_label=_USER_ENTITY_LABEL,
)


async def _close_agent_warm_pool() -> None:
    """Server shutdown: disconnect every pooled CLI subprocess.

    Registered as a Starlette shutdown handler (the app predates a
    lifespan context manager, and Starlette 1.0 keeps event handlers on
    the router); without it, warm spares would outlive the Recordatio
    process as zombie CLIs.
    """
    await agent_warm_pool.close_all()


app.router.add_event_handler("shutdown", _close_agent_warm_pool)


# ---------------------------------------------------------------------------
# MCP server mount — wired here at module bottom so all helper functions
# (_retrieve, _extract_and_store, _background, settings) exist by the time
# they are passed into create_mcp(). The mcp_server module accepts them via
# DI rather than importing them, which keeps the import graph acyclic.
# ---------------------------------------------------------------------------
from tpasini.recordatio.proxy.mcp_server import create_mcp, mount_mcp  # noqa: E402

mcp = create_mcp(
    retrieve=_retrieve,
    extract_and_store=_extract_and_store,
    schedule_background=_background,
    settings=settings,
    allowed_hosts=settings.recordatio_mcp_allowed_hosts,
)
mount_mcp(app, mcp)
