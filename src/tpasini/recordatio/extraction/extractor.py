"""Extract typed knowledge units from completed conversation turns."""

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pontifex import Pontifex
from pydantic import BaseModel, ConfigDict, Field

from tpasini.recordatio.extraction.contracts import (
    AtomicMemoryUnit,
    AttributeExtraction,
    ConversationTurn,
    EntityMention,
    EvidenceReference,
    ExtractionPayload,
    ExtractionResult,
    ExtractionSide,
    RelationExtraction,
    SchemaProposal,
    SchemaProposalKind,
    _is_valid_relation_predicate,
)
from tpasini.recordatio.extraction.prompt import (
    ATOMIC_MEMORY_PROMPT_TEMPLATE,
    EXTRACTION_TARGET_RULES,
    ONTOLOGY_FACTS_PROMPT_TEMPLATE,
    SCHEMA_PROPOSAL_PROMPT_TEMPLATE,
)
from tpasini.recordatio.extraction.settings import EXTRACTION_SETTINGS
from tpasini.recordatio.models import KnowledgeUnit
from tpasini.recordatio.observability import mark_error, span
from tpasini.recordatio.storage.base import Store

logger = logging.getLogger(__name__)

__all__ = [
    "AtomicMemoryUnit",
    "AttributeExtraction",
    "ConversationTurn",
    "EntityMention",
    "EvidenceReference",
    "ExtractionResult",
    "ExtractionSide",
    "RelationExtraction",
    "SchemaProposal",
    "SchemaProposalKind",
    "extract_knowledge",
    "transcript_to_turn",
]


def transcript_to_turn(transcript: dict) -> ConversationTurn:
    """Adapt a transcript dict to the ``ConversationTurn`` the extractor expects.

    Added in Phase 1 so the SDK-driven agent runner can hand the
    extractor the same input shape it already understands. The
    transcript dict is the lightweight stand-in for an OpenAI response
    body that ``agents/runner.py`` assembles from the SDK message
    stream — it carries:

    - ``messages``: the conversation turn(s) the model saw (currently
      a single ``user`` message, but kept as a list so multi-turn
      runs are forward-compatible).
    - ``response_text``: the assistant text the SDK produced.

    Missing fields default to the empty form so a partial transcript
    doesn't raise. Extraction logic itself is unchanged; this is a
    pure shape adapter.
    """
    messages = transcript.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    response_text = transcript.get("response_text") or ""
    if not isinstance(response_text, str):
        response_text = str(response_text)
    return ConversationTurn(messages=list(messages), response_text=response_text)


EXTRACTION_MODEL = EXTRACTION_SETTINGS.model
EXTRACTION_BACKEND = EXTRACTION_SETTINGS.backend
MAX_TOKENS = EXTRACTION_SETTINGS.max_tokens
# Only send the most recent messages to the extractor to keep token cost low.
# Older turns are already persisted in the store.
_EXTRACTION_WINDOW = EXTRACTION_SETTINGS.window  # messages (not counting the final assistant response)


@dataclass(frozen=True)
class _ExtractionPassResult:
    payload: ExtractionPayload
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


class _StreamModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _PromptTokenDetails(_StreamModel):
    cached_tokens: int | None = None


class _CacheCreationUsage(_StreamModel):
    ephemeral_5m_input_tokens: int | None = None


class _ExtractionStreamUsage(_StreamModel):
    """Usage payload emitted by supported streaming extraction providers."""

    prompt_tokens: int | None = None
    input_tokens: int | None = None
    completion_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_creation: _CacheCreationUsage | None = None
    cache_read_input_tokens: int | None = None
    prompt_tokens_details: _PromptTokenDetails | None = None

    def token_counts(self) -> tuple[int, int, int, int]:
        input_tokens = _int_or_zero(self.prompt_tokens if self.prompt_tokens is not None else self.input_tokens)
        output_tokens = _int_or_zero(
            self.completion_tokens if self.completion_tokens is not None else self.output_tokens
        )
        cache_creation_tokens = _int_or_zero(
            self.cache_creation_input_tokens
            if self.cache_creation_input_tokens is not None
            else (
                self.cache_creation.ephemeral_5m_input_tokens
                if self.cache_creation is not None
                else None
            )
        )
        cache_read_tokens = _int_or_zero(
            self.cache_read_input_tokens
            if self.cache_read_input_tokens is not None
            else (
                self.prompt_tokens_details.cached_tokens
                if self.prompt_tokens_details is not None
                else None
            )
        )
        return (input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens)


class _ExtractionStreamDelta(_StreamModel):
    content: str | None = None


class _ExtractionStreamChoice(_StreamModel):
    delta: _ExtractionStreamDelta = Field(default_factory=_ExtractionStreamDelta)


class _ExtractionStreamChunk(_StreamModel):
    choices: list[_ExtractionStreamChoice] = Field(default_factory=list)
    usage: _ExtractionStreamUsage | None = None


def _int_or_zero(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _estimate_tokens(text: str) -> int:
    """Cheap tokens-from-chars estimator used when the backend doesn't surface usage.

    Mirrors the fallback the proxy chat path uses (proxy/server.py:_count_tokens):
    roughly four characters per token, with a minimum of one.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


_XML_BLOCK_RE = re.compile(r"<[a-zA-Z][^>]*>.*?</[a-zA-Z][^>]*>", re.DOTALL)
_XML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_xml(text: str) -> str:
    text = _XML_BLOCK_RE.sub("", text)
    return _XML_TAG_RE.sub("", text).strip()


def _message_to_block(msg: dict[str, object]) -> str | None:
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    if isinstance(content, list):
        text_chunks: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_chunks.append(block.get("text", ""))
                elif "text" in block:
                    text_chunks.append(str(block["text"]))
            else:
                text_chunks.append(str(block))
        content = "\n".join(text_chunks)
    content = _strip_xml(content)
    if not content:
        return None
    return f"[{role}]\n{content}"


_CONTEXT_HEADER = "[context — for disambiguation only, do NOT extract from these]"
_TARGET_HEADER = "[target — extract knowledge from this section only]"


def _render_messages(messages: list[dict[str, object]]) -> list[str]:
    blocks: list[str] = []
    for msg in messages:
        block = _message_to_block(msg)
        if block:
            blocks.append(block)
    return blocks


def _join_with_headers(context_blocks: list[str], target_blocks: list[str]) -> str:
    sections: list[str] = []
    if context_blocks:
        sections.append(_CONTEXT_HEADER + "\n" + "\n\n".join(context_blocks))
    if target_blocks:
        sections.append(_TARGET_HEADER + "\n" + "\n\n".join(target_blocks))
    return "\n\n".join(sections)


def _render_input(turn: ConversationTurn) -> str:
    """Render the user/system/prior-assistant messages, splitting them into
    a [context] block (older messages, for disambiguation only) and a
    [target] block (the most recent message — the only one to extract from).
    """
    messages = turn.messages[-_EXTRACTION_WINDOW:]
    if not messages:
        return ""
    context_blocks = _render_messages(messages[:-1])
    target_blocks = _render_messages(messages[-1:])
    return _join_with_headers(context_blocks, target_blocks)


def _render_output(turn: ConversationTurn) -> str:
    """Render the final assistant response as [target], with the preceding
    user/assistant turns as [context] for disambiguation.
    """
    context_blocks = _render_messages(turn.messages[-_EXTRACTION_WINDOW:])
    target_blocks = [f"[assistant:final]\n{turn.response_text}"]
    return _join_with_headers(context_blocks, target_blocks)


def _format_named_list(items: list[tuple[str, str]]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {name}: {description}" for name, description in items)


def _format_attribute_type_list(items: list[tuple[str, str, str]]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {name} ({value_type}): {description}" for name, description, value_type in items)


def _load_entity_types(store: Store, agent_id: str) -> list[tuple[str, str]]:
    try:
        rows = store.list_entity_types(agent_id)
    except Exception as e:
        logger.warning("failed to load entity types: %s", e)
        return []
    return [(r.name, r.description) for r in rows]


def _load_attribute_types(store: Store, agent_id: str) -> list[tuple[str, str, str]]:
    try:
        rows = store.list_attribute_types(agent_id)
    except Exception as e:
        logger.warning("failed to load attribute types: %s", e)
        return []
    return [(r.name, r.description, r.value_type) for r in rows]


def _load_relation_types(store: Store) -> list[tuple[str, str]]:
    try:
        rows = store.list_relation_types()
    except Exception as e:
        logger.warning("failed to load relation types: %s", e)
        return []
    return [(r.name, r.description) for r in rows]


def _load_topics(store: Store) -> list[tuple[str, str]]:
    try:
        rows = store.list_topics()
    except Exception as e:
        logger.warning("failed to load topics: %s", e)
        return []
    return [(r.name, r.description) for r in rows]


def _to_unit(candidate: AtomicMemoryUnit, now: datetime) -> KnowledgeUnit:
    return KnowledgeUnit(
        type=candidate.type,
        content=candidate.content,
        asserted_at=now,
        valid_from=now,
        provenance=candidate.provenance,
    )


def _coerce_extraction_result(
    data: dict[str, object] | ExtractionPayload,
    *,
    now: datetime,
    relation_types: list[tuple[str, str]],
    topics: list[tuple[str, str]],
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    model: str = "",
    side: ExtractionSide | None = None,
) -> ExtractionResult:
    extraction = data if isinstance(data, ExtractionPayload) else ExtractionPayload.model_validate(data)
    memory_units = extraction.units
    units = [_to_unit(candidate, now) for candidate in memory_units]

    existing_topic_names = {name for name, _ in topics}
    existing_relation_type_names = {name for name, _ in relation_types}
    topic_names = [topic for topic in extraction.topics if topic in existing_topic_names]
    entity_mentions = extraction.entity_mentions
    attributes = extraction.attributes
    relation_candidates = extraction.relations
    used_predicates = {relation.predicate for relation in relation_candidates}
    schema_proposals = [
        proposal for proposal in extraction.schema_proposals
        if proposal.kind != SchemaProposalKind.RELATION_TYPE
        or (
            _is_valid_relation_predicate(proposal.name)
            and proposal.name in used_predicates
            and proposal.name not in existing_relation_type_names
        )
    ]

    return ExtractionResult(
        memory_units=memory_units,
        units=units,
        entity_mentions=entity_mentions,
        attributes=attributes,
        relation_candidates=relation_candidates,
        schema_proposals=schema_proposals,
        topics=topic_names,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        model=model,
        side=side,
    )


def _merge_payloads(payloads: list[ExtractionPayload]) -> ExtractionPayload:
    return ExtractionPayload(
        units=_dedupe((item for payload in payloads for item in payload.units), _unit_key),
        entity_mentions=_dedupe(
            (item for payload in payloads for item in payload.entity_mentions),
            _entity_mention_key,
        ),
        attributes=_dedupe((item for payload in payloads for item in payload.attributes), _attribute_key),
        relations=_dedupe((item for payload in payloads for item in payload.relations), _relation_key),
        schema_proposals=_dedupe(
            (item for payload in payloads for item in payload.schema_proposals),
            _schema_proposal_key,
        ),
        topics=_dedupe((item for payload in payloads for item in payload.topics), _topic_key),
    )


def _dedupe[T](items: Iterable[T], key: Callable[[T], object]) -> list[T]:
    deduped: list[T] = []
    seen: set[object] = set()
    for item in items:
        marker = key(item)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


def _unit_key(unit: AtomicMemoryUnit) -> object:
    return unit.id or (unit.type.value, unit.content.casefold())


def _entity_mention_key(mention: EntityMention) -> object:
    return mention.id or mention.label.casefold()


def _attribute_key(attribute: AttributeExtraction) -> object:
    return attribute.id or (
        attribute.subject_id or attribute.subject.casefold(),
        attribute.attribute,
        attribute.value.casefold(),
    )


def _relation_key(relation: RelationExtraction) -> object:
    return relation.id or (
        relation.subject_id or relation.subject.casefold(),
        relation.predicate,
        relation.object_id or relation.object.casefold(),
    )


def _schema_proposal_key(proposal: SchemaProposal) -> object:
    return proposal.id or (proposal.kind.value, proposal.name.casefold())


def _topic_key(topic: str) -> object:
    return topic.casefold()


def _sum_pass_usage(results: list[_ExtractionPassResult]) -> tuple[int, int, int, int]:
    return (
        sum(result.input_tokens for result in results),
        sum(result.output_tokens for result in results),
        sum(result.cache_creation_tokens for result in results),
        sum(result.cache_read_tokens for result in results),
    )


async def _run_extraction_pass(
    gw: Pontifex,
    *,
    rendered: str,
    system_prompt: str,
    pass_name: str,
    backend: str,
    model: str,
    side: ExtractionSide,
) -> _ExtractionPassResult:
    chunks: list[str] = []
    upstream_usage: _ExtractionStreamUsage | None = None
    extra_body: dict[str, object] | None = None
    if backend == "openai":
        extra_body = {"stream_options": {"include_usage": True}}

    with span(
        "pontifex.extraction",
        **{
            "llm.model": model,
            "llm.provider": backend,
            "recordatio.extraction.side": side.value,
            "recordatio.extraction.pass": pass_name,
        },
    ) as sp:
        try:
            async for chunk in gw.stream(
                messages=[{"role": "user", "content": rendered}],
                system=system_prompt,
                max_tokens=MAX_TOKENS,
                extra_body=extra_body,
            ):
                parsed_chunk = _ExtractionStreamChunk.model_validate(chunk)
                for choice in parsed_chunk.choices:
                    text = choice.delta.content
                    if text:
                        chunks.append(text)
                if parsed_chunk.usage is not None:
                    upstream_usage = parsed_chunk.usage
        except Exception as exc:
            mark_error(sp, str(exc), exc)
            raise

    raw = "".join(chunks)
    if upstream_usage:
        input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens = upstream_usage.token_counts()
    else:
        input_tokens = _estimate_tokens(rendered) + _estimate_tokens(system_prompt)
        output_tokens = _estimate_tokens(raw)
        cache_creation_tokens = 0
        cache_read_tokens = 0

    try:
        payload = ExtractionPayload.from_json(raw)
    except ValueError as e:
        logger.warning("knowledge extraction parse failed [pass=%s]: %s", pass_name, e)
        payload = ExtractionPayload()

    return _ExtractionPassResult(
        payload=payload,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
    )


async def extract_knowledge(
    turn: ConversationTurn,
    api_key: str | None,
    store: Store,
    *,
    side: ExtractionSide,
    agent_id: str,
    user_entity_label: str = "User",
    model: str = EXTRACTION_MODEL,
    backend: str = EXTRACTION_BACKEND,
    base_url: str | None = None,
) -> ExtractionResult:
    """Extract knowledge units, entity facts, relation candidates, and topics from one side of a turn.

    ``side`` selects which half of the conversation the extractor sees:
    - ``ExtractionSide.INPUT``  → user / system / prior assistant messages
    - ``ExtractionSide.OUTPUT`` → only the final assistant response

    Each call runs narrow LLM passes for atomic memories, ontology facts, and
    schema proposals/topics. The proxy normally invokes this twice per chat
    turn (once per side); callers can suppress either side independently via
    the ``x-recordatio-skip-{input,output}-extraction`` body flags. Returns an
    empty ``ExtractionResult`` (no LLM call) when the rendered content is empty.

    ``api_key`` is forwarded to the extraction model. For Anthropic backends
    it should be an Anthropic key; for OpenAI backends an OpenAI key, etc.
    Clients that use a different upstream can pass a dedicated extraction key
    via the ``x-recordatio-api-key`` request header.

    ``model`` and ``backend`` select which model runs the extraction — they
    default to ``EXTRACTION_MODEL`` / ``EXTRACTION_BACKEND`` but can be
    overridden per-request via ``x-recordatio-extraction-model`` and
    ``x-recordatio-extraction-backend`` headers.

    Captures token usage from the streamed response (when the upstream
    surfaces it) and writes it into the returned ``ExtractionResult``,
    falling back to a character-based estimator when no ``usage`` chunk
    is emitted.

    Behaviour on errors:
    - JSON/value errors while parsing the model's text response are caught
      and an empty ``ExtractionResult`` (with usage still populated) is
      returned — the LLM call itself succeeded and incurred cost, the
      output just wasn't parseable.
    - All other exceptions (transport, auth, rate-limit, etc.) propagate
      to the caller so it can record a failure event.
    """
    rendered = _render_input(turn) if side is ExtractionSide.INPUT else _render_output(turn)

    if not rendered.strip():
        # Nothing to extract from — short-circuit without an LLM call.
        return ExtractionResult(model=model, side=side)

    entity_types = _load_entity_types(store, agent_id)
    attribute_types = _load_attribute_types(store, agent_id)
    relation_types = _load_relation_types(store)
    topics = _load_topics(store)
    prompt_args = {
        "target_rules": EXTRACTION_TARGET_RULES,
        "user_entity_label": user_entity_label,
        "entity_types_block": _format_named_list(entity_types),
        "attribute_types_block": _format_attribute_type_list(attribute_types),
        "relation_types_block": _format_named_list(relation_types),
        "topics_block": _format_named_list(topics),
    }
    extraction_passes = [
        ("atomic_memory", ATOMIC_MEMORY_PROMPT_TEMPLATE.format(**prompt_args)),
        ("ontology_facts", ONTOLOGY_FACTS_PROMPT_TEMPLATE.format(**prompt_args)),
        ("schema_proposals", SCHEMA_PROPOSAL_PROMPT_TEMPLATE.format(**prompt_args)),
    ]

    if backend == "codex":
        from pontifex.codex import CodexSession
        from pontifex.codex_completion import CodexCompletion

        from tpasini.recordatio.agents.codex_client import CodexClient

        gw = CodexCompletion(model=model, session=CodexSession(client_factory=CodexClient))
    else:
        gw = Pontifex(backend=backend, model=model, api_key=api_key, base_url=base_url)

    # API-level failures (network, auth, rate-limit) bubble up so the caller
    # can record an `ok=0` event; parse failures below are caught locally.
    pass_results = [
        await _run_extraction_pass(
            gw,
            rendered=rendered,
            system_prompt=pass_prompt,
            pass_name=pass_name,
            backend=backend,
            model=model,
            side=side,
        )
        for pass_name, pass_prompt in extraction_passes
    ]
    input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens = _sum_pass_usage(pass_results)
    extraction = _merge_payloads([result.payload for result in pass_results])

    return _coerce_extraction_result(
        extraction,
        now=datetime.now(UTC),
        relation_types=relation_types,
        topics=topics,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        model=model,
        side=side,
    )
