"""Typed contracts for knowledge extraction results."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from tpasini.recordatio.extraction.settings import CONTRACT_SETTINGS
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit


class ExtractionSide(StrEnum):
    """Which side of a conversation turn an extraction pass covers.

    ``StrEnum`` so the SQLite cost-log column can store ``.value`` directly
    while Python callers stay typed.
    """

    INPUT = "input"
    OUTPUT = "output"


class SchemaProposalKind(StrEnum):
    """Ontology/schema category proposed by extraction."""

    ENTITY_TYPE = "entity_type"
    ATTRIBUTE_TYPE = "attribute_type"
    RELATION_TYPE = "relation_type"
    TOPIC = "topic"


_MAX_RELATION_ENDPOINT_CHARS = CONTRACT_SETTINGS.max_relation_endpoint_chars
_MAX_RELATION_ENDPOINT_WORDS = CONTRACT_SETTINGS.max_relation_endpoint_words
_MAX_RELATION_PREDICATE_CHARS = CONTRACT_SETTINGS.max_relation_predicate_chars
_MAX_MEMORY_CONTENT_CHARS = CONTRACT_SETTINGS.max_memory_content_chars
_MAX_EVIDENCE_QUOTE_CHARS = CONTRACT_SETTINGS.max_evidence_quote_chars
_MAX_CANDIDATE_ID_CHARS = CONTRACT_SETTINGS.max_candidate_id_chars
_RELATION_PREDICATE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LOCAL_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")


@dataclass
class ConversationTurn:
    messages: list[dict]
    response_text: str


def _json_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _json_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _string_list(value: object, *, max_length: int) -> list[str]:
    items: list[str] = []
    for item in _json_sequence(value):
        label: object = item
        mapping = _json_mapping(item)
        if mapping is not None:
            label = mapping.get("label") or mapping.get("name")
        text = str(label).strip() if label is not None else ""
        if text and len(text) <= max_length:
            items.append(text)
    return items


def _string_map(value: object, *, max_key_length: int, max_value_length: int) -> dict[str, str]:
    mapping = _json_mapping(value)
    if mapping is None:
        return {}

    out: dict[str, str] = {}
    for key, raw in mapping.items():
        key_text = str(key).strip()
        value_text = str(raw).strip() if raw is not None else ""
        if key_text and value_text and len(key_text) <= max_key_length and len(value_text) <= max_value_length:
            out[key_text] = value_text
    return out


def _is_atomic_relation_endpoint(value: str) -> bool:
    """Return True when *value* is plausible as a relation endpoint."""

    text = value.strip()
    if not text:
        return False
    if "\n" in text or "\r" in text:
        return False
    if len(text) > _MAX_RELATION_ENDPOINT_CHARS:
        return False
    if len(re.findall(r"\S+", text)) > _MAX_RELATION_ENDPOINT_WORDS:
        return False
    return not any(token in text for token in ("; ", ". ", "? ", "! "))


def _is_valid_relation_predicate(value: str) -> bool:
    predicate = value.strip()
    return (
        bool(predicate)
        and len(predicate) <= _MAX_RELATION_PREDICATE_CHARS
        and bool(_RELATION_PREDICATE_RE.fullmatch(predicate))
    )


def _normalize_local_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > _MAX_CANDIDATE_ID_CHARS or not _LOCAL_ID_RE.fullmatch(text):
        raise ValueError("local id must be a short identifier")
    return text


def _strip_required_text(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError("value must not be empty")
    return text


def _strip_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _default_source_text(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text or "target"


def _validate_atomic_endpoint(value: str) -> str:
    if not _is_atomic_relation_endpoint(value):
        raise ValueError("value must be a short atomic endpoint")
    return value


def _validate_predicate_name(value: str) -> str:
    if not _is_valid_relation_predicate(value):
        raise ValueError("value must be a relation-style predicate")
    return value


OptionalText = Annotated[str | None, BeforeValidator(_strip_optional_text)]
LocalCandidateId = Annotated[
    str | None,
    BeforeValidator(_normalize_local_id),
    Field(default=None, max_length=_MAX_CANDIDATE_ID_CHARS),
]
AtomicEndpoint = Annotated[
    str,
    BeforeValidator(_strip_required_text),
    Field(min_length=1, max_length=_MAX_RELATION_ENDPOINT_CHARS),
    AfterValidator(_validate_atomic_endpoint),
]
PredicateName = Annotated[
    str,
    BeforeValidator(_strip_required_text),
    Field(min_length=1, max_length=_MAX_RELATION_PREDICATE_CHARS),
    AfterValidator(_validate_predicate_name),
]


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class EvidenceReference(_ContractModel):
    """Pointer back to text or an extracted unit for later persistence work."""

    source: Annotated[
        str,
        BeforeValidator(_default_source_text),
        Field(default="target", min_length=1, max_length=32),
    ]
    source_id: Annotated[
        str | None,
        BeforeValidator(_strip_optional_text),
        Field(default=None, max_length=_MAX_CANDIDATE_ID_CHARS),
    ]
    quote: Annotated[
        str | None,
        BeforeValidator(_strip_optional_text),
        Field(default=None, max_length=_MAX_EVIDENCE_QUOTE_CHARS),
    ]


class _CandidateModel(_ContractModel):
    id: LocalCandidateId = None

    @field_validator("confidence", mode="before", check_fields=False)
    @classmethod
    def _coerce_confidence(cls, value: object) -> object:
        if value is None:
            return None
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        if 0.0 <= confidence <= 1.0:
            return confidence
        return None


class AtomicMemoryUnit(_CandidateModel):
    """Structured extraction contract for one atomic memory unit candidate."""

    type: KnowledgeType
    content: Annotated[
        str,
        BeforeValidator(_strip_required_text),
        Field(min_length=1, max_length=_MAX_MEMORY_CONTENT_CHARS),
    ]
    provenance: Annotated[
        str,
        BeforeValidator(_strip_required_text),
        Field(default="extraction", min_length=1, max_length=64),
    ]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class EntityMention(_CandidateModel):
    """Entity mention emitted by extraction before resolver canonicalization.

    ``type`` and ``subtype`` are optional extractor hints. The model may infer
    them from wording now; #79 will provide known ontology labels in a dedicated
    briefing so future extraction prefers existing schema names.
    """

    label: AtomicEndpoint
    type: Annotated[OptionalText, Field(default=None, max_length=64)] = None
    subtype: Annotated[OptionalText, Field(default=None, max_length=64)] = None
    aliases: list[str] = Field(default_factory=list)
    identifiers: dict[str, str] = Field(default_factory=dict)
    canonical_key: Annotated[OptionalText, Field(max_length=160)] = Field(
        default=None,
        description="Optional stable resolver key when the extractor can derive one confidently.",
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(default_factory=list)

    @field_validator("aliases", mode="before")
    @classmethod
    def _coerce_aliases(cls, value: object) -> list[str]:
        return _string_list(value, max_length=_MAX_RELATION_ENDPOINT_CHARS)

    @field_validator("identifiers", mode="before")
    @classmethod
    def _coerce_identifiers(cls, value: object) -> dict[str, str]:
        return _string_map(value, max_key_length=64, max_value_length=256)


class AttributeExtraction(_CandidateModel):
    """Scalar attribute candidate about an entity mention."""

    subject: AtomicEndpoint
    subject_id: LocalCandidateId = None
    attribute: PredicateName
    value: AtomicEndpoint
    provenance: Annotated[
        str,
        BeforeValidator(_strip_required_text),
        Field(default="extraction", min_length=1, max_length=64),
    ]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class RelationExtraction(_CandidateModel):
    """Relation candidate with optional local entity mention references.

    The extractor cannot know persisted entity UUIDs yet. ``subject_id`` and
    ``object_id`` refer to local ``EntityMention.id`` values from the same
    extraction result; #69 resolves those local refs to stored entity IDs.
    ``subject`` and ``object`` remain as labels/scalars for resolver input
    when local IDs are absent or point to ambiguous mentions.
    """

    subject: AtomicEndpoint
    subject_id: LocalCandidateId = None
    predicate: PredicateName
    object: AtomicEndpoint
    object_id: LocalCandidateId = None
    provenance: Annotated[
        str,
        BeforeValidator(_strip_required_text),
        Field(default="extraction", min_length=1, max_length=64),
    ]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class SchemaProposal(_CandidateModel):
    """Ontology proposal kept separate from accepted extracted facts."""

    kind: SchemaProposalKind
    name: Annotated[
        str,
        BeforeValidator(_strip_required_text),
        Field(min_length=1, max_length=96),
    ]
    description: Annotated[
        str,
        BeforeValidator(lambda value: str(value or "").strip()),
        Field(default="", max_length=512),
    ]
    applies_to: Annotated[OptionalText, Field(default=None, max_length=64)] = None
    provenance: Annotated[
        str,
        BeforeValidator(_strip_required_text),
        Field(default="extraction", min_length=1, max_length=64),
    ]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name_for_kind(cls, value: str, info: ValidationInfo) -> str:
        kind = info.data.get("kind")
        if kind in {SchemaProposalKind.ATTRIBUTE_TYPE, SchemaProposalKind.RELATION_TYPE}:
            return _validate_predicate_name(value)
        return value


_MEMORY_UNIT_ADAPTER = TypeAdapter(AtomicMemoryUnit)
_ENTITY_MENTION_ADAPTER = TypeAdapter(EntityMention)
_ATTRIBUTE_ADAPTER = TypeAdapter(AttributeExtraction)
_RELATION_ADAPTER = TypeAdapter(RelationExtraction)
_SCHEMA_PROPOSAL_ADAPTER = TypeAdapter(SchemaProposal)


def _validate_candidate[T](adapter: TypeAdapter[T], value: object) -> T | None:
    try:
        return adapter.validate_python(value)
    except ValidationError:
        return None


def _candidate_list[T](value: object, adapter: TypeAdapter[T]) -> list[T]:
    candidates: list[T] = []
    for item in _json_sequence(value):
        candidate = _validate_candidate(adapter, item)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _entity_mention_list(*values: object) -> list[EntityMention]:
    mentions: list[EntityMention] = []
    seen_labels: set[str] = set()
    for value in values:
        for item in _json_sequence(value):
            payload: object = {"label": item} if isinstance(item, str) else item
            mapping = _json_mapping(payload)
            if mapping is not None and "label" not in mapping and "name" in mapping:
                payload = {**mapping, "label": mapping["name"]}
                mapping = _json_mapping(payload)
            if mapping is not None and "type" not in mapping and "entity_type" in mapping:
                payload = {**mapping, "type": mapping["entity_type"]}
                mapping = _json_mapping(payload)
            if mapping is not None and "canonical_key" not in mapping and "canonical_key_hint" in mapping:
                payload = {**mapping, "canonical_key": mapping["canonical_key_hint"]}
            mention = _validate_candidate(_ENTITY_MENTION_ADAPTER, payload)
            if mention is None:
                continue
            label_key = mention.label.casefold()
            if label_key in seen_labels:
                continue
            seen_labels.add(label_key)
            mentions.append(mention)
    return mentions


def _attribute_list(value: object) -> list[AttributeExtraction]:
    attributes: list[AttributeExtraction] = []
    for item in _json_sequence(value):
        mapping = _json_mapping(item)
        payload: object = item
        if mapping is not None and "attribute" not in mapping and "name" in mapping:
            payload = {**mapping, "attribute": mapping["name"]}
        attribute = _validate_candidate(_ATTRIBUTE_ADAPTER, payload)
        if attribute is not None:
            attributes.append(attribute)
    return attributes


class ExtractionPayload(_ContractModel):
    """Raw structured payload returned by the extraction model.

    ``ExtractionResult`` is the application-facing result after this payload
    is converted into persisted knowledge-unit objects and extraction usage
    metadata.
    """

    units: list[AtomicMemoryUnit] = Field(default_factory=list)
    entity_mentions: list[EntityMention] = Field(default_factory=list)
    attributes: list[AttributeExtraction] = Field(default_factory=list)
    relations: list[RelationExtraction] = Field(default_factory=list)
    schema_proposals: list[SchemaProposal] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)

    @classmethod
    def from_json(cls, text: str) -> Self:
        text = text.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        return cls.model_validate_json(text)

    @field_validator("units", mode="before")
    @classmethod
    def _coerce_units(cls, value: object) -> list[AtomicMemoryUnit]:
        return _candidate_list(value, _MEMORY_UNIT_ADAPTER)

    @field_validator("entity_mentions", mode="before")
    @classmethod
    def _coerce_entity_mentions(cls, value: object) -> list[EntityMention]:
        return _entity_mention_list(value)

    @field_validator("attributes", mode="before")
    @classmethod
    def _coerce_attributes(cls, value: object) -> list[AttributeExtraction]:
        return _attribute_list(value)

    @field_validator("relations", mode="before")
    @classmethod
    def _coerce_relations(cls, value: object) -> list[RelationExtraction]:
        return _candidate_list(value, _RELATION_ADAPTER)

    @field_validator("schema_proposals", mode="before")
    @classmethod
    def _coerce_schema_proposals(cls, value: object) -> list[SchemaProposal]:
        return _candidate_list(value, _SCHEMA_PROPOSAL_ADAPTER)

    @field_validator("topics", mode="before")
    @classmethod
    def _coerce_topics(cls, value: object) -> list[str]:
        return _string_list(value, max_length=96)


@dataclass
class ExtractionResult:
    memory_units: list[AtomicMemoryUnit] = field(default_factory=list)
    units: list[KnowledgeUnit] = field(default_factory=list)
    entity_mentions: list[EntityMention] = field(default_factory=list)
    attributes: list[AttributeExtraction] = field(default_factory=list)
    relation_candidates: list[RelationExtraction] = field(default_factory=list)
    schema_proposals: list[SchemaProposal] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    # Usage metrics for the underlying extraction LLM call.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = ""           # actual model used, e.g. "claude-haiku-4-5-20251001"
    side: ExtractionSide | None = None  # which side of the turn this result covers
