from datetime import UTC, datetime
from typing import Any

from tpasini.recordatio.extraction.contracts import (
    ExtractionPayload,
    ExtractionResult,
    ExtractionSide,
    SchemaProposalKind,
)
from tpasini.recordatio.extraction.extractor import _coerce_extraction_result
from tpasini.recordatio.extraction.prompt import (
    ATOMIC_MEMORY_PROMPT_TEMPLATE,
    ONTOLOGY_FACTS_PROMPT_TEMPLATE,
    SCHEMA_PROPOSAL_PROMPT_TEMPLATE,
)
from tpasini.recordatio.models import KnowledgeType


def _coerce(payload: dict[str, Any]) -> ExtractionResult:
    return _coerce_extraction_result(
        payload,
        now=datetime(2026, 1, 2, tzinfo=UTC),
        relation_types=[("works_at", "employment relationship")],
        topics=[("career", "career and work")],
        input_tokens=11,
        output_tokens=7,
        model="test-model",
        side=ExtractionSide.INPUT,
    )


def test_coerces_ontology_aware_payload() -> None:
    result = _coerce(
        {
            "units": [
                {
                    "id": "unit:1",
                    "type": "semantic",
                    "content": "Alice works at Acme.",
                    "provenance": "extraction",
                    "confidence": "0.91",
                    "evidence": [{"source": "target", "quote": "Alice works at Acme"}],
                }
            ],
            "entity_mentions": [
                {
                    "id": "entity:alice",
                    "label": "Alice",
                    "type": "person",
                    "subtype": "engineer",
                    "aliases": ["A. Smith"],
                    "identifiers": {"email": "alice@example.com"},
                    "canonical_key": "person:alice-smith",
                    "confidence": 0.88,
                    "evidence": [{"source": "target", "quote": "Alice"}],
                },
                {"label": "Acme", "type": "organization"},
            ],
            "attributes": [
                {
                    "id": "attr:alice-job",
                    "subject": "Alice",
                    "subject_id": "entity:alice",
                    "attribute": "job_title",
                    "value": "engineer",
                    "provenance": "extraction",
                    "confidence": 0.77,
                    "evidence": [{"source": "unit", "source_id": "u-1"}],
                }
            ],
            "relations": [
                {
                    "id": "rel:alice-acme",
                    "subject": "Alice",
                    "subject_id": "entity:alice",
                    "predicate": "works_at",
                    "object": "Acme",
                    "object_id": "entity:acme",
                    "confidence": 0.82,
                    "evidence": [{"source": "target", "quote": "works at Acme"}],
                }
            ],
            "schema_proposals": [
                {
                    "kind": "attribute_type",
                    "name": "job_title",
                    "description": "Current job title",
                    "applies_to": "person",
                    "confidence": 0.72,
                }
            ],
            "topics": ["career"],
        }
    )

    assert result.memory_units[0].type is KnowledgeType.SEMANTIC
    assert result.memory_units[0].id == "unit:1"
    assert result.memory_units[0].confidence == 0.91
    assert result.memory_units[0].evidence[0].quote == "Alice works at Acme"
    assert result.units[0].content == "Alice works at Acme."

    mention = result.entity_mentions[0]
    assert mention.id == "entity:alice"
    assert mention.type == "person"
    assert mention.subtype == "engineer"
    assert mention.aliases == ["A. Smith"]
    assert mention.identifiers == {"email": "alice@example.com"}
    assert mention.canonical_key == "person:alice-smith"
    assert mention.confidence == 0.88

    assert [mention.label for mention in result.entity_mentions] == ["Alice", "Acme"]
    assert result.attributes[0].attribute == "job_title"
    assert result.attributes[0].id == "attr:alice-job"
    assert result.attributes[0].subject_id == "entity:alice"
    assert result.attributes[0].evidence[0].source_id == "u-1"
    assert result.relation_candidates[0].predicate == "works_at"
    assert result.relation_candidates[0].id == "rel:alice-acme"
    assert result.relation_candidates[0].subject_id == "entity:alice"
    assert result.relation_candidates[0].object_id == "entity:acme"
    assert result.relation_candidates[0].evidence[0].quote == "works at Acme"
    assert result.schema_proposals[0].kind is SchemaProposalKind.ATTRIBUTE_TYPE
    assert result.topics == ["career"]
    assert result.input_tokens == 11
    assert result.model == "test-model"
    assert result.side is ExtractionSide.INPUT


def test_missing_optional_fields_default_to_empty_or_none() -> None:
    result = _coerce(
        {
            "units": [{"type": "episodic", "content": "Alice met Bob."}],
            "entity_mentions": [{"label": "Alice"}],
            "attributes": [{"subject": "Alice", "attribute": "timezone", "value": "UTC"}],
            "relations": [{"subject": "Alice", "predicate": "met", "object": "Bob"}],
            "topics": [],
            "schema_proposals": [],
        }
    )

    assert result.memory_units[0].provenance == "extraction"
    assert result.memory_units[0].confidence is None
    assert result.memory_units[0].evidence == []
    assert result.entity_mentions[0].aliases == []
    assert result.entity_mentions[0].identifiers == {}
    assert result.entity_mentions[0].canonical_key is None
    assert result.attributes[0].provenance == "extraction"
    assert result.relation_candidates[0].confidence is None
    assert result.schema_proposals == []


def test_extraction_payload_from_json_accepts_fenced_payload() -> None:
    extraction = ExtractionPayload.from_json(
        """
        ```json
        {
          "units": [{"type": "semantic", "content": "Alice works at Acme."}],
          "entity_mentions": [{"label": "Alice", "type": "person"}],
          "relations": [{"subject": "Alice", "predicate": "works_at", "object": "Acme"}]
        }
        ```
        """
    )

    assert extraction.units[0].content == "Alice works at Acme."
    assert extraction.entity_mentions[0].label == "Alice"
    assert extraction.entity_mentions[0].type == "person"
    assert extraction.relations[0].predicate == "works_at"


def test_rejects_malformed_candidates_without_dropping_valid_ones() -> None:
    long_endpoint = (
        "Alice works at Acme and this sentence is intentionally too descriptive "
        "to be a graph endpoint because it contains a complete contextual claim."
    )
    result = _coerce(
        {
            "units": [
                {"type": "semantic", "content": "Alice works at Acme."},
                {"type": "unsupported", "content": "bad"},
                {"type": "semantic", "content": "   "},
            ],
            "entity_mentions": [
                {"label": "Alice", "confidence": 1.2, "identifiers": {"wikidata": "Q1", "empty": ""}},
                {"name": ""},
                {"name": "Alice"},
                3,
            ],
            "attributes": [
                {"subject": "Alice", "attribute": "role", "value": "engineer"},
                {"subject": "Alice", "attribute": "bad predicate", "value": "engineer"},
                {"subject": long_endpoint, "attribute": "role", "value": "engineer"},
            ],
            "relations": [
                {"subject": "Alice", "predicate": "works_at", "object": "Acme"},
                {"subject": "Alice", "predicate": "bad predicate", "object": "Acme"},
                {"subject": long_endpoint, "predicate": "works_at", "object": "Acme"},
                {"subject": "Alice", "predicate": "works_at", "object": ""},
            ],
            "schema_proposals": [
                {"kind": "relation_type", "name": "bad predicate"},
                {"kind": "unknown", "name": "x"},
            ],
            "topics": ["unknown"],
        }
    )

    assert [unit.content for unit in result.memory_units] == ["Alice works at Acme."]
    assert len(result.entity_mentions) == 1
    assert result.entity_mentions[0].confidence is None
    assert result.entity_mentions[0].identifiers == {"wikidata": "Q1"}
    assert [attribute.attribute for attribute in result.attributes] == ["role"]
    assert [(relation.subject, relation.predicate, relation.object) for relation in result.relation_candidates] == [
        ("Alice", "works_at", "Acme")
    ]
    assert result.schema_proposals == []
    assert result.topics == []


def test_rejects_oversized_memory_content_and_evidence_quotes() -> None:
    result = _coerce(
        {
            "units": [
                {"type": "semantic", "content": "x" * 900},
                {
                    "type": "semantic",
                    "content": "Small fact.",
                    "evidence": [{"source": "target", "quote": "x" * 700}],
                },
            ],
            "entity_mentions": [],
            "attributes": [],
            "relations": [],
            "schema_proposals": [],
            "topics": [],
        }
    )

    assert result.memory_units == []
    assert result.units == []


def test_schema_proposals_remain_separate_from_accepted_facts() -> None:
    result = _coerce(
        {
            "units": [],
            "entity_mentions": [],
            "attributes": [],
            "relations": [{"subject": "Alice", "predicate": "attends", "object": "Conference"}],
            "schema_proposals": [
                {"kind": "entity_type", "name": "venue", "description": "Physical event venue"},
                {"kind": "relation_type", "name": "attends", "description": "Attendance"},
                {"kind": "topic", "name": "events", "description": "Events and venues"},
            ],
            "topics": [],
        }
    )

    assert [proposal.kind for proposal in result.schema_proposals] == [
        SchemaProposalKind.ENTITY_TYPE,
        SchemaProposalKind.RELATION_TYPE,
        SchemaProposalKind.TOPIC,
    ]
    assert result.attributes == []
    assert result.relation_candidates[0].predicate == "attends"


def test_schema_proposals_are_isolated_to_schema_prompt() -> None:
    assert "schema_proposals" not in ATOMIC_MEMORY_PROMPT_TEMPLATE
    assert "schema_proposals" not in ONTOLOGY_FACTS_PROMPT_TEMPLATE
    assert "schema_proposals" in SCHEMA_PROPOSAL_PROMPT_TEMPLATE


def test_atomic_prompt_carves_out_per_entity_state_from_reports() -> None:
    """A tracker/recap's per-entity STATE must be extracted as durable, not
    dropped as report scaffolding.

    The atomic prompt tells the extractor to omit operational/status-report
    scaffolding. Without an explicit positive carve-out, a daily job/property
    brief — which is structurally a status report — reads as fully droppable, so
    the deliverable's tracker rows never persist and night-to-night continuity is
    lost. Assert the prompt keeps BOTH instructions: omit scaffolding, but keep
    per-entity state records.
    """
    prompt = ATOMIC_MEMORY_PROMPT_TEMPLATE.casefold()
    # The omit-scaffolding instruction is still present.
    assert "report scaffolding" in prompt
    # The positive carve-out for durable per-entity state is present.
    assert "per-entity state" in prompt
    assert "durable" in prompt
    # Current status of a tracked entity is explicitly called out as extractable.
    assert "current status" in prompt or "pipeline status" in prompt
