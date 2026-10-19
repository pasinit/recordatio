"""Extraction execution and persistence pipeline for the proxy."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.entity_identity import judge_from_env
from tpasini.recordatio.entity_resolver import (
    EntityMention as ResolverEntityMention,
)
from tpasini.recordatio.entity_resolver import (
    EntityResolutionStatus,
    EntityResolver,
    SQLiteEntityResolverRepository,
    stable_user_mention,
)
from tpasini.recordatio.extraction.contracts import (
    AtomicMemoryUnit,
    AttributeExtraction,
    EvidenceReference,
    RelationExtraction,
    SchemaProposal,
    SchemaProposalKind,
)
from tpasini.recordatio.extraction.contracts import (
    EntityMention as ExtractionEntityMention,
)
from tpasini.recordatio.extraction.extractor import (
    ConversationTurn,
    ExtractionResult,
    ExtractionSide,
    extract_knowledge,
)
from tpasini.recordatio.identity import (
    add_principal_alias,
    ensure_mirror_entity,
    ensure_principal,
    get_principal,
    mention_binds_principal,
    principal_binding_labels,
    user_scope,
)
from tpasini.recordatio.models import (
    AttributeType,
    EntityAttribute,
    KnowledgeType,
    KnowledgeUnit,
    Relation,
    RelationType,
    Topic,
)
from tpasini.recordatio.models import (
    EntityMention as StoredEntityMention,
)
from tpasini.recordatio.proxy import embeddings
from tpasini.recordatio.proxy.cost_log import CostLog
from tpasini.recordatio.proxy.pricing import lookup_cost as _lookup_cost
from tpasini.recordatio.proxy.usage import ExtractionUsage as _ExtractionUsage
from tpasini.recordatio.proxy.usage import provider_tag as _provider_tag
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.wikidata import resolve_entity

logger = logging.getLogger(__name__)

type _ResolveEntityFn = Callable[[str], Awaitable[Entity]]

_MIN_FACT_CONFIDENCE = 0.55
_MIN_SCHEMA_CONFIDENCE = 0.85
_MIN_DISCOVERED_SCALAR_CONFIDENCE = 0.85


class _ExtractKnowledgeFn(Protocol):
    def __call__(
        self,
        turn: ConversationTurn,
        api_key: str | None,
        store: SQLiteStore,
        *,
        side: ExtractionSide,
        agent_id: str,
        user_entity_label: str = "User",
        model: str = "",
        backend: str = "",
        base_url: str | None = None,
    ) -> Awaitable[ExtractionResult]:
        """Run one extraction pass."""
        ...


class _EntityMentionSourceKind(StrEnum):
    CONTEXT = "context"
    EXTRACTION_CANDIDATE = "extraction_candidate"
    KNOWLEDGE_UNIT = "knowledge_unit"
    TARGET = "target"


@dataclass(frozen=True)
class _PersistedMentionSource:
    kind: _EntityMentionSourceKind
    source_id: str
    snippet: str | None


@dataclass(frozen=True)
class _ResolvedRelation:
    predicate: str
    subject_id: str
    object_id: str
    subject_label: str
    object_label: str


@dataclass
class ExtractionPipelineContext:
    """Runtime dependencies for extraction and persistence."""

    store: SQLiteStore
    cost_log: CostLog
    extraction_sem: asyncio.Semaphore
    unreported_extractions: MutableMapping[str, list[_ExtractionUsage]]
    unreported_extractions_lock: asyncio.Lock
    extraction_bare_model: str
    extraction_backend: str
    extraction_base_url: str | None
    user_entity_label: str = "User"
    bootstrapped_agents: set[str] = field(default_factory=set)
    extract_knowledge: _ExtractKnowledgeFn = extract_knowledge
    resolve_entity: _ResolveEntityFn = resolve_entity
    ensure_user_entity: Callable[[str], None] | None = None


@dataclass(frozen=True)
class _EntityMentionKey:
    label: str
    type: str
    canonical_key: str | None


@dataclass(frozen=True)
class _AttributeKey:
    subject: str
    attribute: str
    value: str
    subject_id: str | None


@dataclass(frozen=True)
class _RelationCandidateKey:
    subject: str
    predicate: str
    object: str
    subject_id: str | None
    object_id: str | None


def _ensure_user_entity(ctx: ExtractionPipelineContext, agent_id: str) -> None:
    """Ensure-on-write safety net (IDENTITY_DESIGN.md §5): extraction never
    *creates* identity in the agent scope — it delegates to the same idempotent
    ensure-principal path the login API uses, against the user scope."""
    if ctx.ensure_user_entity is not None:
        ctx.ensure_user_entity(agent_id)
        return
    if agent_id in ctx.bootstrapped_agents:
        return
    ensure_principal(ctx.store, agent_id.partition("/")[0])
    ctx.bootstrapped_agents.add(agent_id)


@dataclass(frozen=True)
class _IdentityRouting:
    """Per-persist routing context (IDENTITY_DESIGN.md §2).

    ``binding_labels`` maps normalized label -> original text for every label
    the E-RULE binds to the principal (the configured user label + the
    principal's standing alias set). ``principal_id`` is ``None`` when no
    principal exists (legacy stores / injected test ensure hooks), in which
    case everything routes agent-scoped exactly as before.
    """

    agent_id: str
    user_scope: str
    principal_id: str | None
    binding_labels: dict[str, str]

    def binds(self, label: str, mention_type: str | None) -> bool:
        return self.principal_id is not None and mention_binds_principal(label, mention_type, self.binding_labels)


def _identity_routing(ctx: ExtractionPipelineContext, agent_id: str) -> _IdentityRouting:
    uscope = user_scope(agent_id)
    principal = None
    try:
        principal = get_principal(ctx.store, uscope)
    except Exception:
        logger.exception("principal lookup failed for %s", uscope)
    if principal is None:
        return _IdentityRouting(agent_id=agent_id, user_scope=uscope, principal_id=None, binding_labels={})
    return _IdentityRouting(
        agent_id=agent_id,
        user_scope=uscope,
        principal_id=principal.id,
        binding_labels=principal_binding_labels(ctx.store, uscope, ctx.user_entity_label),
    )


def _unit_routes_user_level(
    unit: KnowledgeUnit,
    candidate_id: str | None,
    routing: _IdentityRouting,
    principal_evidence_unit_ids: set[str],
) -> bool:
    """§2.2: a *semantic* unit with the principal in its resolved mention set
    is user-anchored; episodic/procedural units stay agent-scoped — the agent
    that lived them owns them. Units are atomic: routed whole, never split."""
    if routing.principal_id is None or unit.type.value != "semantic":
        return False
    if candidate_id and candidate_id in principal_evidence_unit_ids:
        return True
    content = unit.content.lower()
    return any(original.lower() in content for original in routing.binding_labels.values())


# ---------------------------------------------------------------------------

def _norm_content(text: str) -> str:
    """Lowercase + collapse whitespace. Cheap cross-side dedup key."""
    return " ".join(text.lower().split())


# Conversational opt-out tag. When a turn's text carries this marker, that
# side's extraction is suppressed exactly as the explicit
# ``x-recordatio-skip-{input,output}-extraction`` body flags do (a zero-cost
# "skipped" cost-log row is still recorded). This lets a human — or an upstream
# agent — mark a turn ephemeral inline (e.g. synthetic capability tests) without
# having to plumb a request flag.
_IGNORE_FOR_MEMORY_TAG = "[please ignore for memory]"


def _message_content_text(content: object) -> str:
    """Flatten a chat message ``content`` (plain str or content-block list) to
    text, ignoring non-text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _messages_text(messages: list[dict]) -> str:
    """Concatenate the text of every message on a turn's input side."""
    return "\n".join(_message_content_text(m.get("content")) for m in messages)


def _has_ignore_for_memory_tag(text: str | None) -> bool:
    return bool(text) and _IGNORE_FOR_MEMORY_TAG in text.lower()


def _confidence_is_acceptable(confidence: float | None, threshold: float) -> bool:
    return confidence is None or confidence >= threshold


def _confidence_meets_required_threshold(confidence: float | None, threshold: float) -> bool:
    return confidence is not None and confidence >= threshold


def _has_target_evidence(
    evidence: list[EvidenceReference],
    *,
    allow_unit: bool = False,
    allow_missing: bool = True,
) -> bool:
    if not evidence:
        return allow_missing
    for ref in evidence:
        source = ref.source.casefold()
        if source == _EntityMentionSourceKind.TARGET.value and ref.quote:
            return True
        if allow_unit and source in {"unit", _EntityMentionSourceKind.KNOWLEDGE_UNIT.value} and (
            ref.source_id or ref.quote
        ):
            return True
    return False


def _candidate_has_quality(
    *,
    confidence: float | None,
    evidence: list[EvidenceReference],
    threshold: float = _MIN_FACT_CONFIDENCE,
    allow_unit_evidence: bool = False,
) -> bool:
    return (
        _confidence_is_acceptable(confidence, threshold)
        and _has_target_evidence(evidence, allow_unit=allow_unit_evidence)
    )


def _memory_unit_has_quality(candidate: AtomicMemoryUnit | None, unit: KnowledgeUnit) -> bool:
    if candidate is None:
        return True
    return _candidate_has_quality(
        confidence=getattr(candidate, "confidence", None),
        evidence=getattr(candidate, "evidence", []),
    )


def _entity_mention_has_quality(mention: ExtractionEntityMention) -> bool:
    return _candidate_has_quality(
        confidence=mention.confidence,
        evidence=mention.evidence,
        allow_unit_evidence=True,
    )


def _attribute_has_quality(attribute: AttributeExtraction) -> bool:
    return _candidate_has_quality(confidence=attribute.confidence, evidence=attribute.evidence)


def _relation_has_quality(relation: RelationExtraction) -> bool:
    return _candidate_has_quality(
        confidence=relation.confidence,
        evidence=relation.evidence,
        allow_unit_evidence=True,
    )


def _filter_extraction_result(result: ExtractionResult) -> ExtractionResult:
    units: list[KnowledgeUnit] = []
    memory_units: list[AtomicMemoryUnit] = []
    dropped = 0
    for index, unit in enumerate(result.units):
        candidate = result.memory_units[index] if index < len(result.memory_units) else None
        if not _memory_unit_has_quality(candidate, unit):
            logger.debug("  dropping low-quality unit: %r", unit.content[:100])
            dropped += 1
            continue
        units.append(unit)
        if candidate is not None:
            memory_units.append(candidate)

    # Observability: an extraction that produced units but had *every* one
    # discarded by the quality gate is the signature of a systematic
    # evidence-contract mismatch (e.g. the extractor emitting evidence with a
    # non-"target" source or without a quote), not ordinary per-unit filtering.
    # Because extraction runs fire-and-forget every turn, that failure is
    # otherwise silent — the LLM call still logs ok=1 with cost while zero units
    # persist for days. Surface it above DEBUG so a zero-extraction run is
    # visible without a metrics backend.
    if dropped and not units:
        logger.warning(
            "extraction quality gate dropped all %d extracted unit(s) "
            "(side=%s); persisting zero units this turn",
            dropped,
            getattr(result.side, "value", result.side),
        )

    return ExtractionResult(
        memory_units=memory_units,
        units=units,
        entity_mentions=[
            mention for mention in result.entity_mentions
            if _entity_mention_has_quality(mention)
        ],
        attributes=[
            attribute for attribute in result.attributes
            if _attribute_has_quality(attribute)
        ],
        relation_candidates=[
            relation for relation in result.relation_candidates
            if _relation_has_quality(relation)
        ],
        schema_proposals=list(result.schema_proposals),
        topics=list(result.topics),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_creation_tokens=result.cache_creation_tokens,
        cache_read_tokens=result.cache_read_tokens,
        model=result.model,
        side=result.side,
    )


def _schema_candidate_has_quality(proposal: SchemaProposal) -> bool:
    return (
        _confidence_meets_required_threshold(proposal.confidence, _MIN_SCHEMA_CONFIDENCE)
        and _has_target_evidence(proposal.evidence, allow_missing=False)
    )


def _filter_schema_topics(result: ExtractionResult) -> ExtractionResult:
    return ExtractionResult(
        memory_units=list(result.memory_units),
        units=list(result.units),
        entity_mentions=list(result.entity_mentions),
        attributes=list(result.attributes),
        relation_candidates=list(result.relation_candidates),
        schema_proposals=[
            proposal for proposal in result.schema_proposals
            if _schema_candidate_has_quality(proposal)
        ],
        topics=list(result.topics),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_creation_tokens=result.cache_creation_tokens,
        cache_read_tokens=result.cache_read_tokens,
        model=result.model,
        side=result.side,
    )


def _bump_extraction_count(
    ctx: ExtractionPipelineContext,
    table_kind: str,
    existing_id: str,
    agent_id: str,
) -> None:
    """Best-effort reinforcement bump when extraction re-asserts an existing fact.

    "The world confirmed this fact again" → +1 to the matched row's
    ``extraction_count``. Mirrors the error discipline of the rest of
    ``_persist_side`` (runs in ``_background``): a store hiccup is logged, not
    raised, so a single counter failure can't abort the persistence pass.
    """
    try:
        ctx.store.increment_extraction_count(table_kind, [existing_id], agent_id)
    except Exception:
        logger.exception("failed to bump extraction_count for %s %s", table_kind, existing_id)


# Default cosine at or above which a re-extracted unit is treated as a
# near-duplicate of an existing active same-type unit (reinforce, don't insert).
# Conservative by design — a wrong merge is irreversible — and env-overridable
# without a rebuild, matching the tunable-constant convention the briefing
# budgeter uses. Only consulted when F2 embeddings are enabled.
_UNIT_NEARDUP_DEFAULT_COSINE = 0.97

# Global near-dup knob (all types), kept for backwards compatibility.
_UNIT_NEARDUP_GLOBAL_ENV = "RECORDATIO_UNIT_NEARDUP_COSINE"

# Per-type overrides of the near-dup cutoff. Durable semantic and procedural
# units are the ones the LLM extractor paraphrases most between runs (its own
# docstring says so), so a re-extraction of a standing fact routinely lands just
# below the conservative global 0.97 and inserts a near-identical twin instead of
# reinforcing the canonical. This makes the single global knob type-aware so ops
# can loosen the cutoff for the paraphrase-prone durable types WITHOUT touching
# episodic units, whose exact wording is more load-bearing and where a wrong
# merge is costlier. Each per-type knob falls back to the global knob, which
# falls back to the type's built-in default (below) — so an operator can still
# override any of them without a rebuild.
_UNIT_NEARDUP_ENV_BY_TYPE: dict[KnowledgeType, str] = {
    KnowledgeType.SEMANTIC: "RECORDATIO_UNIT_NEARDUP_COSINE_SEMANTIC",
    KnowledgeType.PROCEDURAL: "RECORDATIO_UNIT_NEARDUP_COSINE_PROCEDURAL",
    KnowledgeType.EPISODIC: "RECORDATIO_UNIT_NEARDUP_COSINE_EPISODIC",
}

# Per-type built-in default cutoffs. SEMANTIC and PROCEDURAL are the durable
# "standing fact" types the extractor paraphrases most between daily runs, so at
# the conservative global 0.97 a re-extraction lands *just under* the cutoff and
# mints a near-identical twin every day instead of reinforcing the canonical —
# the duplicate bloat that dilutes retrieval and starves the episodic records
# (the nightly analyst then merges the clusters by hand). Give those two a
# slightly looser built-in default so the reinforcement path (a NON-destructive
# extraction_count bump — the incoming paraphrase is dropped, no row is
# superseded or deleted) catches them by default. EPISODIC is deliberately
# absent: its exact wording is load-bearing (dated pipeline snapshots,
# per-application tracker records) and a wrong merge there is costlier, so it
# keeps the conservative ``_UNIT_NEARDUP_DEFAULT_COSINE``. Still fully
# env-overridable per type or globally.
_UNIT_NEARDUP_DEFAULT_BY_TYPE: dict[KnowledgeType, float] = {
    KnowledgeType.SEMANTIC: 0.94,
    KnowledgeType.PROCEDURAL: 0.94,
}


def _read_cosine_env(name: str, default: float) -> float:
    """Read a cosine cutoff from ``name``, falling back to ``default`` if the var
    is unset or unparseable."""
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _unit_neardup_builtin_default(unit_type: KnowledgeType | None) -> float:
    """Built-in near-dup default for ``unit_type``: the durable types
    (SEMANTIC/PROCEDURAL) get the looser ``_UNIT_NEARDUP_DEFAULT_BY_TYPE`` value;
    everything else (EPISODIC, or no type) keeps the conservative global
    ``_UNIT_NEARDUP_DEFAULT_COSINE``."""
    if unit_type is None:
        return _UNIT_NEARDUP_DEFAULT_COSINE
    return _UNIT_NEARDUP_DEFAULT_BY_TYPE.get(unit_type, _UNIT_NEARDUP_DEFAULT_COSINE)


def _unit_neardup_threshold(unit_type: KnowledgeType | None = None) -> float:
    """Near-duplicate cosine cutoff, read per call so ops can flip it via env.

    Resolution order, most specific first:
      1. the per-type knob for ``unit_type`` (e.g.
         ``RECORDATIO_UNIT_NEARDUP_COSINE_SEMANTIC``), when a type is given;
      2. the single global knob ``RECORDATIO_UNIT_NEARDUP_COSINE``;
      3. the type's built-in default (``_UNIT_NEARDUP_DEFAULT_BY_TYPE`` for the
         durable SEMANTIC/PROCEDURAL types, else ``_UNIT_NEARDUP_DEFAULT_COSINE``).
    """
    global_threshold = _read_cosine_env(
        _UNIT_NEARDUP_GLOBAL_ENV, _unit_neardup_builtin_default(unit_type)
    )
    if unit_type is None:
        return global_threshold
    per_type_env = _UNIT_NEARDUP_ENV_BY_TYPE.get(unit_type)
    if per_type_env is None:
        return global_threshold
    return _read_cosine_env(per_type_env, global_threshold)


def _near_duplicate_unit_id(
    ctx: ExtractionPipelineContext,
    unit: KnowledgeUnit,
    agent_id: str,
    candidate_ids: list[str],
) -> str | None:
    """Embedding-cosine near-duplicate fallback for exact-match unit dedup.

    The extractor is an LLM that paraphrases a standing fact slightly
    differently each time it re-encounters it, so a re-extraction almost never
    matches an existing unit byte-for-byte under ``_norm_content``. Without this
    fallback every paraphrase inserts a fresh unit (extraction_count=1) instead
    of reinforcing the canonical, growing the near-identical clusters the
    nightly analyst then has to merge by hand.

    Returns the id of the most-similar active same-type unit whose cosine to the
    incoming content is at or above ``_unit_neardup_threshold()``, else ``None``.
    Uses the SAME store-text surface and L2-normalized bi-encoder the retrieval
    path already computes (``embeddings``), so "near-duplicate" here means
    exactly what "similar" means at query time.

    Only active when F2 embeddings are enabled and the bi-encoder loads; a
    missing model, un-embedded candidates, or any store hiccup degrades to
    ``None`` (no merge) — this must never raise into the persistence pass.
    """
    if not candidate_ids or not embeddings.enabled():
        return None
    try:
        stored = ctx.store.load_item_embeddings(candidate_ids, (agent_id,))
        if not stored:
            return None
        vecs = embeddings.embed_documents([embeddings.unit_surface(unit.content)])
        if not vecs:
            return None
        query_vec = vecs[0]
        best_id: str | None = None
        best_cos = 0.0
        for candidate_id, blob in stored.items():
            cos = embeddings.cosine(query_vec, embeddings.unpack_embedding(blob))
            if cos > best_cos:
                best_id, best_cos = candidate_id, cos
        if best_id is not None and best_cos >= _unit_neardup_threshold(unit.type):
            return best_id
        return None
    except Exception:
        logger.exception("near-duplicate unit check failed (non-fatal)")
        return None


def _active_unit_id_for_content(
    ctx: ExtractionPipelineContext,
    unit: KnowledgeUnit,
    agent_id: str,
) -> str | None:
    rows = ctx.store._conn.execute(
        """
        SELECT id, content
        FROM knowledge_units
        WHERE agent_id = ? AND type = ? AND valid_until IS NULL
        """,
        (agent_id, unit.type.value),
    ).fetchall()
    content_key = _norm_content(unit.content)
    candidate_ids: list[str] = []
    for row in rows:
        if _norm_content(row["content"]) == content_key:
            return row["id"]
        candidate_ids.append(row["id"])
    # No byte-for-byte match: fall back to an embedding near-duplicate check so a
    # paraphrased re-extraction reinforces the canonical instead of inserting a
    # near-identical twin (no-op unless F2 embeddings are enabled).
    return _near_duplicate_unit_id(ctx, unit, agent_id, candidate_ids)


def _active_attribute_exists(
    ctx: ExtractionPipelineContext,
    *,
    entity_id: str,
    attribute_type: str,
    value: str,
    agent_id: str,
) -> bool:
    row = ctx.store._conn.execute(
        """
        SELECT 1
        FROM entity_attributes
        WHERE agent_id = ?
          AND entity_id = ?
          AND attribute_type = ?
          AND LOWER(value) = LOWER(?)
          AND valid_until IS NULL
        LIMIT 1
        """,
        (agent_id, entity_id, attribute_type, value),
    ).fetchone()
    return row is not None


def _entity_mention_key(mention: ExtractionEntityMention) -> _EntityMentionKey:
    return _EntityMentionKey(
        label=_norm_content(mention.label),
        type=mention.type or "",
        canonical_key=mention.canonical_key,
    )


def _attribute_key(attribute: AttributeExtraction) -> _AttributeKey:
    return _AttributeKey(
        subject=_norm_content(attribute.subject),
        attribute=attribute.attribute,
        value=_norm_content(attribute.value),
        subject_id=attribute.subject_id,
    )


def _relation_candidate_key(
    relation: RelationExtraction,
) -> _RelationCandidateKey:
    return _RelationCandidateKey(
        subject=_norm_content(relation.subject),
        predicate=relation.predicate,
        object=_norm_content(relation.object),
        subject_id=relation.subject_id,
        object_id=relation.object_id,
    )


def _schema_proposal_key(proposal: SchemaProposal) -> tuple[str, str]:
    return (proposal.kind.value, proposal.name.casefold())


def _merge_cross_side(
    primary: ExtractionResult,
    secondary: ExtractionResult,
) -> ExtractionResult:
    """Return a new ``ExtractionResult`` that is the deduplicated union of two
    same-turn extraction passes (INPUT and OUTPUT).

    Both sides looked at the same conversation, so when they agree on a unit,
    entity, or relation that is one duplicate fact — not two assertions. The
    primary side wins ties; entries unique to ``secondary`` are appended after
    primary's. ``primary`` and ``secondary`` are left untouched.

    Metadata fields (token counts, model, side) are not relevant to
    persistence — the per-side cost-log rows have already been written by
    ``_extract_side`` — so they are left at their defaults on the returned
    result.
    """
    unit_keys = {_norm_content(unit.content) for unit in primary.units}
    extra_units = []
    extra_memory_units = []
    for index, unit in enumerate(secondary.units):
        if _norm_content(unit.content) in unit_keys:
            continue
        extra_units.append(unit)
        if index < len(secondary.memory_units):
            extra_memory_units.append(secondary.memory_units[index])

    mention_keys = {_entity_mention_key(mention) for mention in primary.entity_mentions}
    extra_mentions = [
        mention for mention in secondary.entity_mentions
        if _entity_mention_key(mention) not in mention_keys
    ]

    attribute_keys = {_attribute_key(attribute) for attribute in primary.attributes}
    extra_attributes = [
        attribute for attribute in secondary.attributes
        if _attribute_key(attribute) not in attribute_keys
    ]

    relation_candidate_keys = {
        _relation_candidate_key(relation) for relation in primary.relation_candidates
    }
    extra_relation_candidates = [
        relation for relation in secondary.relation_candidates
        if _relation_candidate_key(relation) not in relation_candidate_keys
    ]

    topic_set = set(primary.topics)
    extra_topics = [t for t in secondary.topics if t not in topic_set]

    schema_proposal_keys = {_schema_proposal_key(proposal) for proposal in primary.schema_proposals}
    extra_schema_proposals = [
        proposal for proposal in secondary.schema_proposals
        if _schema_proposal_key(proposal) not in schema_proposal_keys
    ]

    return ExtractionResult(
        units=list(primary.units) + extra_units,
        memory_units=list(primary.memory_units) + extra_memory_units,
        entity_mentions=list(primary.entity_mentions) + extra_mentions,
        attributes=list(primary.attributes) + extra_attributes,
        relation_candidates=list(primary.relation_candidates) + extra_relation_candidates,
        schema_proposals=list(primary.schema_proposals) + extra_schema_proposals,
        topics=list(primary.topics) + extra_topics,
    )


async def _extract_and_store(
    ctx: ExtractionPipelineContext,
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
    """Run extraction on both sides, merge the results, persist once.

    The return value is a dict keyed by side name (``"input"`` / ``"output"``)
    whose values are the per-side ``ExtractionResult`` or ``None`` if that side
    was skipped / crashed. Proxy callers ignore it; the MCP ``extract`` tool
    surfaces it as its synchronous response, so the per-side breakdown is
    preserved here for visibility.

    Order of operations: collect both ``ExtractionResult``s *before* any
    ``store.*`` call, build their deduplicated union, then persist that union
    in a single pass. A fact emitted by both INPUT and OUTPUT reaches the
    store exactly once.
    """
    results: dict[str, ExtractionResult | None] = {"input": None, "output": None}
    if not agent_id:
        logger.warning("skipping knowledge extraction: no agent_id")
        return results
    # Honor an inline "[please ignore for memory]" opt-out. A tag on the USER
    # turn marks the whole exchange ephemeral — the assistant's reply answers an
    # ignored prompt and does not echo the tag, so mining it via the OUTPUT pass
    # would persist supposedly-ephemeral content — therefore a user-side tag
    # suppresses BOTH sides (the stated intent: let a human mark a turn ephemeral
    # inline, e.g. synthetic capability tests). A tag on the response text is an
    # independent trigger that suppresses only the OUTPUT side.
    if not skip_input_extraction and _has_ignore_for_memory_tag(_messages_text(turn.messages)):
        logger.info(
            "skip extraction (both sides): '%s' tag in user turn marks the exchange ephemeral [agent=%s]",
            _IGNORE_FOR_MEMORY_TAG, agent_id,
        )
        skip_input_extraction = True
        skip_output_extraction = True
    if not skip_output_extraction and _has_ignore_for_memory_tag(turn.response_text):
        logger.info("skip_output_extraction: '%s' tag in response [agent=%s]", _IGNORE_FOR_MEMORY_TAG, agent_id)
        skip_output_extraction = True
    logger.info(
        "extraction starting [agent=%s] response=%d chars skip_input=%s skip_output=%s",
        agent_id, len(turn.response_text or ""),
        skip_input_extraction, skip_output_extraction,
    )
    _ensure_user_entity(ctx, agent_id)

    for side, skipped in (
        (ExtractionSide.INPUT, skip_input_extraction),
        (ExtractionSide.OUTPUT, skip_output_extraction),
    ):
        if skipped:
            _record_skipped_extraction_side(ctx, side, agent_id, user_id, parent_request_id, source_agent_id)
            continue
        try:
            results[side.value] = await _extract_side(
                ctx, turn, api_key, agent_id, user_id, parent_request_id, source_agent_id, side,
            )
        except Exception:
            # A failure on one side (LLM error, cost-log mishap, …) must not
            # block the other side's pass — log and keep going.
            logger.exception("extraction crashed [side=%s]", side.value)

    input_result = results["input"]
    output_result = results["output"]
    if input_result is not None and output_result is not None:
        merged = _merge_cross_side(input_result, output_result)
        dropped = (
            len(input_result.units) + len(output_result.units) - len(merged.units),
            len(input_result.entity_mentions) + len(output_result.entity_mentions) - len(merged.entity_mentions),
            (
                len(input_result.relation_candidates)
                + len(output_result.relation_candidates)
                - len(merged.relation_candidates)
            ),
        )
        if any(dropped):
            logger.info(
                "cross-side merge dedup'd %d units, %d entity mentions, %d relation candidates",
                *dropped,
            )
    else:
        merged = input_result or output_result

    if merged is not None:
        try:
            await _persist_side(ctx, merged, agent_id, source_agent_id=source_agent_id)
        except Exception:
            logger.exception("persistence crashed")

    return results


def _record_skipped_extraction_side(
    ctx: ExtractionPipelineContext,
    side: ExtractionSide,
    agent_id: str,
    user_id: str | None,
    parent_request_id: str | None,
    source_agent_id: str | None = None,
) -> None:
    """Log a zero-cost cost-log row for a suppressed extraction pass.

    No ``_unreported_extractions`` entry is added — the call never happened,
    so ``extraction_calls`` on the next chat response shouldn't count it.
    """
    extraction_provider = _provider_tag(ctx.extraction_backend, ctx.extraction_base_url)
    try:
        ctx.cost_log.record_extraction(
            model=ctx.extraction_bare_model,
            provider=extraction_provider,
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            cost_usd=0.0,
            agent_id=agent_id,
            source_agent_id=source_agent_id,
            user_id=user_id,
            request_id=parent_request_id,
            ok=True,
            extraction_side=side,
            skipped=True,
        )
    except Exception:
        logger.exception("cost-log write failed; continuing")


async def _extract_side(
    ctx: ExtractionPipelineContext,
    turn: ConversationTurn,
    api_key: str | None,
    agent_id: str,
    user_id: str | None,
    parent_request_id: str | None,
    source_agent_id: str | None,
    side: ExtractionSide,
) -> ExtractionResult | None:
    """Run a single-sided extraction LLM call and record its cost. No store writes.

    Returns the ``ExtractionResult`` (units/entity mentions/relation candidates not yet persisted).
    Returns ``None`` if the LLM call itself failed — the persistence side-effects
    in that case are limited to a cost-log row.
    """
    extraction_provider = _provider_tag(ctx.extraction_backend, ctx.extraction_base_url)
    async with ctx.extraction_sem:
        try:
            result: ExtractionResult = await ctx.extract_knowledge(
                turn, api_key, ctx.store,
                side=side,
                agent_id=agent_id,
                user_entity_label=ctx.user_entity_label,
                model=ctx.extraction_bare_model,
                backend=ctx.extraction_backend,
                base_url=ctx.extraction_base_url,
            )
        except Exception:
            logger.exception("knowledge extraction failed [side=%s]", side.value)
            # Record the failure (0/0 tokens, $0) so `extraction_calls` on the
            # next response reflects calls that errored at the API level.
            async with ctx.unreported_extractions_lock:
                ctx.unreported_extractions[agent_id].append(_ExtractionUsage(
                    model=ctx.extraction_bare_model,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                ))
            try:
                ctx.cost_log.record_extraction(
                    model=ctx.extraction_bare_model,
                    provider=extraction_provider,
                    input_tokens=0,
                    output_tokens=0,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    cost_usd=0.0,
                    agent_id=agent_id,
                    source_agent_id=source_agent_id,
                    user_id=user_id,
                    request_id=parent_request_id,
                    ok=False,
                    extraction_side=side,
                )
            except Exception:
                logger.exception("cost-log write failed; continuing")
            return None

        # Successful API call (parse may still have failed → empty result with usage populated).
        cost = 0.0 if ctx.extraction_backend == "codex" else _lookup_cost(
            result.model or ctx.extraction_bare_model,
            result.input_tokens, result.output_tokens,
            result.cache_creation_tokens, result.cache_read_tokens,
        ) or 0.0
        async with ctx.unreported_extractions_lock:
            ctx.unreported_extractions[agent_id].append(_ExtractionUsage(
                model=result.model or ctx.extraction_bare_model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=cost,
                cache_creation_tokens=result.cache_creation_tokens,
                cache_read_tokens=result.cache_read_tokens,
            ))
        try:
            ctx.cost_log.record_extraction(
                model=result.model or ctx.extraction_bare_model,
                provider=extraction_provider,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cache_creation_tokens=result.cache_creation_tokens,
                cache_read_tokens=result.cache_read_tokens,
                cost_usd=cost,
                agent_id=agent_id,
                source_agent_id=source_agent_id,
                user_id=user_id,
                request_id=parent_request_id,
                ok=True,
                extraction_side=side,
            )
        except Exception:
            logger.exception("cost-log write failed; continuing")

    logger.info(
        "extraction result [agent=%s side=%s]: %d units, %d entity mentions, %d relation candidates, topics=%s",
        agent_id, side.value, len(result.units), len(result.entity_mentions), len(result.relation_candidates),
        result.topics or [],
    )
    return result


def _mention_source_from_evidence(
    ref: EvidenceReference,
    unit_ids_by_local_id: dict[str, str],
) -> _PersistedMentionSource | None:
    source_id = ref.source_id
    if ref.source == "unit" and source_id in unit_ids_by_local_id:
        return _PersistedMentionSource(
            kind=_EntityMentionSourceKind.KNOWLEDGE_UNIT,
            source_id=unit_ids_by_local_id[source_id],
            snippet=ref.quote,
        )
    if not source_id:
        return None
    try:
        source_kind = _EntityMentionSourceKind(ref.source)
    except ValueError:
        logger.debug("skipping entity mention evidence with unknown source kind %r", ref.source)
        return None
    return _PersistedMentionSource(kind=source_kind, source_id=source_id, snippet=ref.quote)


def _persist_entity_mention_evidence(
    ctx: ExtractionPipelineContext,
    *,
    entity_id: str,
    label: str,
    evidence: list[EvidenceReference],
    units: list[KnowledgeUnit],
    unit_ids_by_local_id: dict[str, str],
    candidate_id: str | None,
    now: datetime,
    agent_id: str,
) -> None:
    sources: set[_PersistedMentionSource] = set()
    for ref in evidence:
        source = _mention_source_from_evidence(ref, unit_ids_by_local_id)
        if source is not None:
            sources.add(source)

    if not sources:
        for unit in units:
            if label.lower() in unit.content.lower():
                sources.add(_PersistedMentionSource(
                    kind=_EntityMentionSourceKind.KNOWLEDGE_UNIT,
                    source_id=unit.id,
                    snippet=label,
                ))

    if not sources and candidate_id:
        sources.add(_PersistedMentionSource(
            kind=_EntityMentionSourceKind.EXTRACTION_CANDIDATE,
            source_id=candidate_id,
            snippet=None,
        ))

    for source in sources:
        ctx.store.upsert_entity_mention(
            StoredEntityMention(
                entity_id=entity_id,
                source_kind=source.kind.value,
                source_id=source.source_id,
                snippet=source.snippet,
                asserted_at=now,
            ),
            agent_id,
        )


def _persist_entity_attribute(
    ctx: ExtractionPipelineContext,
    *,
    entity_id: str,
    attribute_type: str,
    value: str,
    provenance: str,
    functional_predicates: set[str],
    now: datetime,
    agent_id: str,
    allow_attribute_type_creation: bool = True,
) -> None:
    if ctx.store.get_attribute_type_by_name(attribute_type, agent_id) is None:
        if not allow_attribute_type_creation:
            logger.debug("  skipping attribute with unknown type %r", attribute_type)
            return
        ctx.store.upsert_attribute_type(
            AttributeType(
                name=attribute_type,
                description=f"Discovered scalar attribute {attribute_type}",
                value_type="text",
                source="extraction",
                created_at=now,
            ),
            agent_id,
        )
    if _active_attribute_exists(
        ctx,
        entity_id=entity_id,
        attribute_type=attribute_type,
        value=value,
        agent_id=agent_id,
    ):
        logger.debug("  skipping duplicate active attribute %s=%r", attribute_type, value)
        return
    if attribute_type in functional_predicates:
        invalidated = ctx.store.supersede_entity_attributes(entity_id, attribute_type, now, agent_id)
        if invalidated:
            logger.debug("  superseded %d stale %r attribute(s) for entity %s", invalidated, attribute_type, entity_id)
    ctx.store.upsert_entity_attribute(
        EntityAttribute(
            entity_id=entity_id,
            attribute_type=attribute_type,
            value=value,
            asserted_at=now,
            valid_from=now,
            provenance=provenance,
        ),
        agent_id,
    )


def _remember_entity_id(
    *,
    entity_ids: dict[str, str],
    entity_ids_by_local_id: dict[str, str],
    label: str,
    entity_id: str,
    local_id: str | None = None,
) -> None:
    entity_ids[label] = entity_id
    entity_ids.setdefault(label.casefold(), entity_id)
    if local_id:
        entity_ids_by_local_id[local_id] = entity_id


def _resolve_entity_id(
    *,
    entity_ids: dict[str, str],
    entity_ids_by_local_id: dict[str, str],
    label: str,
    local_id: str | None = None,
) -> str | None:
    if local_id and local_id in entity_ids_by_local_id:
        return entity_ids_by_local_id[local_id]
    return entity_ids.get(label) or entity_ids.get(label.casefold())


def _tag_entity_topics(
    ctx: ExtractionPipelineContext,
    entity_id: str,
    topic_names: list[str],
    agent_id: str,
) -> None:
    for topic_name in topic_names:
        try:
            ctx.store.tag_entity(entity_id, topic_name, agent_id)
        except Exception:
            logger.exception("failed to tag entity %s with topic %s", entity_id, topic_name)


def _schema_topic_proposals(proposals: list[SchemaProposal]) -> list[SchemaProposal]:
    return [proposal for proposal in proposals if proposal.kind == SchemaProposalKind.TOPIC]


def _persist_units_and_topics(
    ctx: ExtractionPipelineContext,
    result: ExtractionResult,
    now: datetime,
    agent_id: str,
    source_agent_id: str | None = None,
    routing: _IdentityRouting | None = None,
    principal_evidence_unit_ids: frozenset[str] = frozenset(),
) -> tuple[list[str], dict[str, str]]:
    """Persist units to their §2.2 home scope; returns (topic names, the final
    unit id -> home scope map). Dedup/reinforcement runs against the home
    scope, so two agents re-asserting the same user-anchored fact reinforce
    one user-store row (§3.1)."""
    unit_home_by_id: dict[str, str] = {}
    # Observability: count the two write outcomes so the insert-vs-reinforce
    # ratio is greppable per turn. The near-dup reinforcement fallback is gated
    # on `embeddings.enabled()` and silently degrades to "insert a twin" when F2
    # embeddings are off or candidates are un-embedded — in that configuration
    # EVERY paraphrased re-assertion inserts a fresh unit and the duplicate
    # clusters the nightly analyst merges by hand accumulate invisibly. Surfacing
    # the per-turn tally (and whether the embedding path was even available)
    # makes that silent no-op visible without hand-auditing the store — the same
    # "visible without a metrics backend" discipline the quality gate uses above.
    inserted = 0
    reinforced = 0
    candidates = list(result.memory_units) + [None] * max(0, len(result.units) - len(result.memory_units))
    for unit, candidate in zip(result.units, candidates, strict=False):
        home = agent_id
        if routing is not None and _unit_routes_user_level(
            unit, candidate.id if candidate is not None else None, routing, set(principal_evidence_unit_ids)
        ):
            home = routing.user_scope
        try:
            existing_id = _active_unit_id_for_content(ctx, unit, home)
            if existing_id is None and home != agent_id:
                # Pre-migration transition safety: the same active fact may
                # still live in the agent scope (the migration lifts it later).
                # Reinforce it in place rather than duplicating it into the
                # user store.
                existing_id = _active_unit_id_for_content(ctx, unit, agent_id)
                if existing_id is not None:
                    home = agent_id
            if existing_id is not None:
                unit.id = existing_id
                unit_home_by_id[unit.id] = home
                # Re-assertion of an already-stored fact: bump the write-side
                # reinforcement counter instead of inserting a duplicate row.
                # On a user-store row this counts assertions across all of the
                # user's agents (§3.1).
                _bump_extraction_count(ctx, "unit", existing_id, home)
                reinforced += 1
                logger.debug("  reinforced duplicate active unit [%s]: %r", unit.type.value, unit.content[:100])
                continue
            # User-level rows carry the full originating agent_id (§3.2).
            ctx.store.add(unit, home, source_agent_id=agent_id if home != agent_id else source_agent_id)
            unit_home_by_id[unit.id] = home
            inserted += 1
            logger.debug("  stored unit [%s -> %s]: %r", unit.type.value, home, unit.content[:100])
        except Exception:
            logger.exception("failed to persist knowledge unit")

    if inserted or reinforced:
        logger.info(
            "persisted knowledge units: %d inserted, %d reinforced "
            "(near-dup embeddings enabled=%s)",
            inserted,
            reinforced,
            embeddings.enabled(),
        )

    new_topic_proposals = _schema_topic_proposals(result.schema_proposals)
    for topic in new_topic_proposals:
        try:
            ctx.store.add_topic(
                Topic(
                    id="",
                    name=topic.name,
                    description=topic.description,
                    source="discovered",
                    created_at=now,
                )
            )
            logger.debug("  stored topic %r", topic.name)
        except Exception:
            logger.exception("failed to persist topic %r", topic.name)

    topic_names = list(result.topics) + [topic.name for topic in new_topic_proposals]
    for unit in result.units:
        for topic_name in topic_names:
            try:
                ctx.store.tag_unit(unit.id, topic_name, unit_home_by_id.get(unit.id, agent_id))
            except Exception:
                logger.exception("failed to tag unit %s with topic %s", unit.id, topic_name)
    return topic_names, unit_home_by_id


def _unit_ids_by_local_id(result: ExtractionResult) -> dict[str, str]:
    return {
        candidate.id: unit.id
        for candidate, unit in zip(result.memory_units, result.units, strict=False)
        if candidate.id
    }


def _persist_extracted_entity_mentions(
    ctx: ExtractionPipelineContext,
    *,
    result: ExtractionResult,
    resolver: EntityResolver,
    topic_names: list[str],
    unit_ids_by_local_id: dict[str, str],
    entity_ids: dict[str, str],
    entity_ids_by_local_id: dict[str, str],
    now: datetime,
    agent_id: str,
    routing: _IdentityRouting | None = None,
) -> None:
    for mention in result.entity_mentions:
        try:
            # E-RULE (IDENTITY_DESIGN.md §2.1): a mention that names the
            # principal is rewritten to the stable user mention BEFORE
            # resolution — it binds at the deterministic canonical-key stage,
            # ahead of alias and fuzzy, and resolves IN THE USER SCOPE.
            # Full-label normalized equality only; typed mentions don't bind.
            binds = routing is not None and routing.binds(mention.label, mention.type)
            scope = routing.user_scope if binds else agent_id
            if binds:
                stable = stable_user_mention(routing.user_scope)
                resolver_mention = ResolverEntityMention(
                    label=stable.label,
                    type=stable.type,
                    canonical_key=stable.canonical_key,
                    # Wikidata ids are never propagated to the principal — they
                    # are misattributions of a private person (§2.3); other
                    # identifiers (email, …) are principal material.
                    identifiers={
                        scheme: value
                        for scheme, value in mention.identifiers.items()
                        if scheme not in ("wikidata", "wikidata_id")
                    },
                )
            else:
                resolver_mention = ResolverEntityMention(
                    label=mention.label,
                    type=mention.type,
                    canonical_key=mention.canonical_key,
                    aliases=mention.aliases,
                    identifiers=mention.identifiers,
                )
            resolution = resolver.resolve(resolver_mention, scope)
            if resolution.entity is None:
                logger.debug("  skipping ambiguous entity mention %r", mention.label)
                continue
            if resolution.status == EntityResolutionStatus.RESOLVED:
                # Mention maps to an already-stored entity (canonical key /
                # wikidata / alias / fuzzy match) — no new insert. Count the
                # re-assertion. A genuinely CREATED entity keeps the insert
                # default of 1.
                _bump_extraction_count(ctx, "entity", resolution.entity.id, scope)
            if binds:
                # Mention-supplied aliases extend the principal's standing
                # alias set through the collision-guarded path only (§2.1) —
                # never through the resolver's unguarded metadata merge.
                for alias in mention.aliases:
                    add_principal_alias(
                        ctx.store,
                        resolution.entity.id,
                        alias,
                        routing.user_scope,
                        agent_scope=agent_id,
                        source="extraction",
                        now=now,
                    )
            _remember_entity_id(
                entity_ids=entity_ids,
                entity_ids_by_local_id=entity_ids_by_local_id,
                label=mention.label,
                entity_id=resolution.entity.id,
                local_id=mention.id,
            )
            # The principal's mentions are user-store material (§2.2 row 1);
            # their source ids may reference agent-scope units — cross-scope
            # edges are legal under the composed read.
            _persist_entity_mention_evidence(
                ctx,
                entity_id=resolution.entity.id,
                label=mention.label,
                evidence=mention.evidence,
                units=result.units,
                unit_ids_by_local_id=unit_ids_by_local_id,
                candidate_id=mention.id,
                now=now,
                agent_id=scope,
            )
            logger.debug("  stored entity mention %r", mention.label)
            _tag_entity_topics(ctx, resolution.entity.id, topic_names, scope)
        except Exception:
            logger.exception("failed to resolve/persist entity mention %r", mention.label)


def _persist_extracted_attributes(
    ctx: ExtractionPipelineContext,
    *,
    attributes: list[AttributeExtraction],
    entity_ids: dict[str, str],
    entity_ids_by_local_id: dict[str, str],
    functional_predicates: set[str],
    now: datetime,
    agent_id: str,
    routing: _IdentityRouting | None = None,
) -> None:
    for attribute in attributes:
        subject_id = _resolve_entity_id(
            entity_ids=entity_ids,
            entity_ids_by_local_id=entity_ids_by_local_id,
            label=attribute.subject,
            local_id=attribute.subject_id,
        )
        if subject_id is None:
            logger.debug("  skipping attribute (unresolved subject): %s.%s", attribute.subject, attribute.attribute)
            continue
        # The principal's attributes live in the user store (§2.2 row 1).
        home = agent_id
        if routing is not None and routing.principal_id == subject_id:
            home = routing.user_scope
        _persist_entity_attribute(
            ctx,
            entity_id=subject_id,
            attribute_type=attribute.attribute,
            value=attribute.value,
            provenance=attribute.provenance,
            functional_predicates=functional_predicates,
            now=now,
            agent_id=home,
        )


def _resolve_relation_candidates(
    ctx: ExtractionPipelineContext,
    *,
    relation_candidates: list[RelationExtraction],
    entity_ids: dict[str, str],
    entity_ids_by_local_id: dict[str, str],
    functional_predicates: set[str],
    now: datetime,
    agent_id: str,
    routing: _IdentityRouting | None = None,
) -> list[_ResolvedRelation]:
    resolved_relations: list[_ResolvedRelation] = []
    for rel in relation_candidates:
        subject_id = _resolve_entity_id(
            entity_ids=entity_ids,
            entity_ids_by_local_id=entity_ids_by_local_id,
            label=rel.subject,
            local_id=rel.subject_id,
        )
        if subject_id is None:
            logger.debug("  skipping relation (unresolved subject): %s -[%s]-> %s",
                         rel.subject, rel.predicate, rel.object)
            continue
        object_id = _resolve_entity_id(
            entity_ids=entity_ids,
            entity_ids_by_local_id=entity_ids_by_local_id,
            label=rel.object,
            local_id=rel.object_id,
        )
        if object_id is None:
            attribute_type_exists = ctx.store.get_attribute_type_by_name(rel.predicate, agent_id) is not None
            can_discover_scalar_attribute = (
                _confidence_meets_required_threshold(rel.confidence, _MIN_DISCOVERED_SCALAR_CONFIDENCE)
                and _has_target_evidence(rel.evidence, allow_missing=False)
            )
            if not attribute_type_exists and not can_discover_scalar_attribute:
                logger.debug(
                    "  skipping unresolved relation object instead of creating arbitrary attribute: %s -[%s]-> %s",
                    rel.subject,
                    rel.predicate,
                    rel.object,
                )
                continue
            attr_home = agent_id
            if routing is not None and routing.principal_id == subject_id:
                attr_home = routing.user_scope
            _persist_entity_attribute(
                ctx,
                entity_id=subject_id,
                attribute_type=rel.predicate,
                value=rel.object,
                provenance=rel.provenance,
                functional_predicates=functional_predicates,
                now=now,
                agent_id=attr_home,
                allow_attribute_type_creation=can_discover_scalar_attribute,
            )
            continue
        resolved_relations.append(_ResolvedRelation(
            predicate=rel.predicate,
            subject_id=subject_id,
            object_id=object_id,
            subject_label=rel.subject,
            object_label=rel.object,
        ))
    return resolved_relations


def _persist_relation_types(
    ctx: ExtractionPipelineContext,
    schema_proposals: list[SchemaProposal],
    resolved_relations: list[_ResolvedRelation],
    now: datetime,
) -> None:
    used_relation_predicates = {relation.predicate for relation in resolved_relations}
    for proposal in schema_proposals:
        if proposal.kind != SchemaProposalKind.RELATION_TYPE:
            continue
        if proposal.name not in used_relation_predicates:
            logger.debug("  skipping unused relation type %r", proposal.name)
            continue
        try:
            ctx.store.add_relation_type(
                RelationType(
                    id="",
                    name=proposal.name,
                    description=proposal.description,
                    source="discovered",
                    created_at=now,
                )
            )
            logger.debug("  stored relation type %r", proposal.name)
        except Exception:
            logger.exception("failed to persist relation type %r", proposal.name)


def _active_relation_id(
    ctx: ExtractionPipelineContext,
    *,
    subject_id: str,
    predicate: str,
    object_id: str,
    agent_id: str,
) -> str | None:
    """Id of the active relation matching the unique-index key, if any.

    Mirrors ``idx_relations_unique_active`` (agent_id, subject_id, predicate,
    object_id) WHERE valid_until IS NULL — i.e. the row that ``add_relation``'s
    INSERT OR IGNORE would silently collide with.
    """
    row = ctx.store._conn.execute(
        """
        SELECT id
        FROM relations
        WHERE agent_id = ?
          AND subject_id = ?
          AND predicate = ?
          AND object_id = ?
          AND valid_until IS NULL
        LIMIT 1
        """,
        (agent_id, subject_id, predicate, object_id),
    ).fetchone()
    return row["id"] if row is not None else None


def _user_level_endpoint(
    ctx: ExtractionPipelineContext,
    endpoint_id: str,
    routing: _IdentityRouting,
) -> str:
    """Endpoint id a user-level relation must reference (§2.3): the principal
    itself, an entity already in the user scope, or a deterministic mirror of
    the agent-scoped endpoint."""
    if endpoint_id == routing.principal_id:
        return endpoint_id
    if ctx.store.get_entity_by_id(endpoint_id, routing.user_scope) is not None:
        return endpoint_id
    source = ctx.store.get_entity_by_id(endpoint_id, routing.agent_id)
    if source is None:
        return endpoint_id  # not an entity in this scope; keep the cross-scope edge
    return ensure_mirror_entity(ctx.store, source, routing.user_scope, source_agent_id=routing.agent_id)


def _persist_resolved_relations(
    ctx: ExtractionPipelineContext,
    resolved_relations: list[_ResolvedRelation],
    functional_predicates: set[str],
    now: datetime,
    agent_id: str,
    source_agent_id: str | None = None,
    routing: _IdentityRouting | None = None,
) -> None:
    for relation in resolved_relations:
        try:
            subject_id, object_id = relation.subject_id, relation.object_id
            home = agent_id
            relation_source = source_agent_id
            if routing is not None and routing.principal_id in (subject_id, object_id):
                # §2.2: a relation with the principal as an endpoint is the
                # canonical user-anchored fact — it lives in the user store,
                # with the non-principal endpoint mirrored (§2.3) and the full
                # originating agent_id as provenance (§3.2).
                home = routing.user_scope
                relation_source = agent_id
                subject_id = _user_level_endpoint(ctx, subject_id, routing)
                object_id = _user_level_endpoint(ctx, object_id, routing)
            # Re-assertion of an already-active relation (same scope / subject /
            # predicate / object) is the conflict path ``add_relation``'s
            # INSERT OR IGNORE would silently drop. Count it as reinforcement
            # instead, and skip the supersede+re-insert churn that would
            # otherwise reset the row (and its counter). On a user-store row
            # this counts assertions across all of the user's agents (§3.1).
            existing_id = _active_relation_id(
                ctx,
                subject_id=subject_id,
                predicate=relation.predicate,
                object_id=object_id,
                agent_id=home,
            )
            if existing_id is not None:
                _bump_extraction_count(ctx, "relation", existing_id, home)
                logger.debug(
                    "  reinforced relation: %r -[%s]-> %r",
                    relation.subject_label,
                    relation.predicate,
                    relation.object_label,
                )
                continue
            if relation.predicate in functional_predicates:
                # In the user scope this is the LIVE cross-agent supersession
                # (§3.3): the newest assertion of a functional fact closes all
                # active rows, including rows other agents asserted.
                invalidated = ctx.store.supersede_relations(
                    subject_id,
                    relation.predicate,
                    now,
                    home,
                )
                if invalidated:
                    logger.debug(
                        "  superseded %d stale %r relation(s) for %r",
                        invalidated,
                        relation.predicate,
                        relation.subject_label,
                    )
            ctx.store.add_relation(
                Relation(
                    id="",
                    subject_id=subject_id,
                    predicate=relation.predicate,
                    object_id=object_id,
                    asserted_at=now,
                    valid_from=now,
                    valid_until=None,
                    provenance="extraction",
                ),
                home,
                source_agent_id=relation_source,
            )
            # Standing-alias learning (§2.1): a has_name assertion with the
            # principal as subject extends the alias set, collision-guarded.
            if (
                routing is not None
                and routing.principal_id is not None
                and relation.predicate == "has_name"
                and relation.subject_id == routing.principal_id
            ):
                add_principal_alias(
                    ctx.store,
                    routing.principal_id,
                    relation.object_label,
                    routing.user_scope,
                    agent_scope=agent_id,
                    source="has_name",
                    now=now,
                )
            logger.debug(
                "  stored relation: %r -[%s]-> %r",
                relation.subject_label,
                relation.predicate,
                relation.object_label,
            )
        except Exception:
            logger.exception(
                "failed to persist relation %s -[%s]-> %s",
                relation.subject_label,
                relation.predicate,
                relation.object_label,
            )


def _auto_link_unit_mentions(
    ctx: ExtractionPipelineContext,
    *,
    units: list[KnowledgeUnit],
    entity_ids: dict[str, str],
    resolved_relations: list[_ResolvedRelation],
    now: datetime,
    agent_id: str,
    source_agent_id: str | None = None,
    routing: _IdentityRouting | None = None,
    unit_home_by_id: dict[str, str] | None = None,
) -> None:
    """Auto-link units to mentioned entities. The link is bookkeeping: it
    follows the UNIT's home (§2.2). A user-homed unit's links point at
    user-scope rows (mirroring §2.3); an agent-homed unit may legally link
    cross-scope to the user-scope principal — the composed read resolves it."""
    existing_pairs: set[tuple[str, str]] = {
        (relation.subject_id, relation.object_id)
        for relation in resolved_relations
    }
    homes = unit_home_by_id or {}
    for label, entity_id in entity_ids.items():
        if label.casefold() == label:
            continue
        for unit in units:
            if label.lower() not in unit.content.lower():
                continue
            home = homes.get(unit.id, agent_id)
            target_id = entity_id
            if routing is not None and home == routing.user_scope:
                target_id = _user_level_endpoint(ctx, entity_id, routing)
            pair = (unit.id, target_id)
            if pair in existing_pairs or pair[0] == pair[1]:
                continue
            existing_pairs.add(pair)
            try:
                ctx.store.add_relation(
                    Relation(
                        id="",
                        subject_id=unit.id,
                        predicate="mentions",
                        object_id=target_id,
                        asserted_at=now,
                        valid_from=now,
                        valid_until=None,
                        provenance="auto",
                    ),
                    home,
                    source_agent_id=agent_id if home != agent_id else source_agent_id,
                )
            except Exception:
                logger.exception("failed to auto-link unit %s to entity %s", unit.id, target_id)


async def _persist_side(
    ctx: ExtractionPipelineContext,
    result: ExtractionResult,
    agent_id: str,
    source_agent_id: str | None = None,
) -> None:
    """Persist an ``ExtractionResult`` to the store.

    Split out of the prior ``_extract_and_store_side`` so cross-side dedup
    can run between extraction and persistence — see ``_extract_and_store``.
    """
    result = _filter_extraction_result(result)
    result = _filter_schema_topics(result)
    # Ensure-on-write safety net (§5): the principal exists before any routing
    # decision — an extraction arriving before any login call loses nothing.
    _ensure_user_entity(ctx, agent_id)
    routing = _identity_routing(ctx, agent_id)
    now = datetime.now(UTC)
    # Unit ids referenced as evidence by principal-binding mentions: those
    # units carry the principal in their resolved mention set (§2.2).
    principal_evidence_unit_ids = frozenset(
        ref.source_id
        for mention in result.entity_mentions
        if routing.binds(mention.label, mention.type)
        for ref in mention.evidence
        if ref.source in ("unit", _EntityMentionSourceKind.KNOWLEDGE_UNIT.value) and ref.source_id
    )
    topic_names, unit_home_by_id = _persist_units_and_topics(
        ctx, result, now, agent_id, source_agent_id,
        routing=routing, principal_evidence_unit_ids=principal_evidence_unit_ids,
    )
    # judge is None unless RECORDATIO_ENTITY_JUDGE is set — then borderline mentions
    # are adjudicated by the zero-shot identity judge instead of a tailored key.
    resolver = EntityResolver(SQLiteEntityResolverRepository(ctx.store), judge=judge_from_env())
    entity_ids: dict[str, str] = {}
    entity_ids_by_local_id: dict[str, str] = {}

    _persist_extracted_entity_mentions(
        ctx,
        result=result,
        resolver=resolver,
        topic_names=topic_names,
        unit_ids_by_local_id=_unit_ids_by_local_id(result),
        entity_ids=entity_ids,
        entity_ids_by_local_id=entity_ids_by_local_id,
        now=now,
        agent_id=agent_id,
        routing=routing,
    )
    functional_predicates = {rt.name for rt in ctx.store.list_relation_types() if rt.is_functional}
    _persist_extracted_attributes(
        ctx,
        attributes=result.attributes,
        entity_ids=entity_ids,
        entity_ids_by_local_id=entity_ids_by_local_id,
        functional_predicates=functional_predicates,
        now=now,
        agent_id=agent_id,
        routing=routing,
    )
    resolved_relations = _resolve_relation_candidates(
        ctx,
        relation_candidates=result.relation_candidates,
        entity_ids=entity_ids,
        entity_ids_by_local_id=entity_ids_by_local_id,
        functional_predicates=functional_predicates,
        now=now,
        agent_id=agent_id,
        routing=routing,
    )
    _persist_relation_types(ctx, result.schema_proposals, resolved_relations, now)
    _persist_resolved_relations(
        ctx,
        resolved_relations,
        {rt.name for rt in ctx.store.list_relation_types() if rt.is_functional},
        now,
        agent_id,
        source_agent_id=source_agent_id,
        routing=routing,
    )
    _auto_link_unit_mentions(
        ctx,
        units=result.units,
        entity_ids=entity_ids,
        resolved_relations=resolved_relations,
        now=now,
        agent_id=agent_id,
        source_agent_id=source_agent_id,
        routing=routing,
        unit_home_by_id=unit_home_by_id,
    )
    _embed_persisted_items(ctx, unit_home_by_id, entity_ids_by_local_id, now)


def _embed_persisted_items(
    ctx: ExtractionPipelineContext,
    unit_home_by_id: dict[str, str],
    entity_ids_by_local_id: dict[str, str],
    now: datetime,
) -> None:
    """Write-time F2 hook: embed + cache new/changed unit & entity surfaces.

    No-op unless ``RECORDATIO_F2_EMBEDDINGS`` is on and the bi-encoder loads
    (graceful — the store stays un-embedded and retrieval degrades to BM25).
    Surfaces are read back from canonical rows so the embedded text is exactly
    what retrieval will surface (RR4); the ``surface_hash`` guard skips re-embeds
    of unchanged surfaces.  Failures are swallowed: an embedding hiccup must
    never fail extraction persistence.
    """
    if not embeddings.enabled():
        return
    try:
        unit_ids = list(unit_home_by_id)
        entity_ids = list(dict.fromkeys(entity_ids_by_local_id.values()))
        items = list(ctx.store.item_surfaces_for_embedding(unit_ids, entity_ids))
        # Relations touching any just-written unit/entity: their rendered-triple
        # surface is embeddable now (endpoint labels are canonical at write
        # time).  Caching them here keeps the query path off the on-the-fly
        # relation embed (the latency hot spot).
        rel_ids = ctx.store.relation_ids_referencing(unit_ids + entity_ids)
        items += list(ctx.store.relation_surfaces_for_embedding(rel_ids))
        embeddings.embed_and_store(ctx.store, items, now_iso=now.isoformat())
    except Exception:
        logger.exception("F2: write-time embedding failed (non-fatal)")
