"""Core data models for agentmem."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class KnowledgeType(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


@dataclass
class KnowledgeUnit:
    type: KnowledgeType
    content: str
    asserted_at: datetime
    valid_from: datetime
    provenance: str
    valid_until: datetime | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    extraction_count: int = 1  # times asserted/re-asserted by extraction (write-side reinforcement)
    retrieval_count: int = 0  # times surfaced in a briefing (read-side usage)


@dataclass
class RetrievalContext:
    query: str
    top_k: int = 10
    topic_hint: str | None = None


@dataclass
class RelationType:
    id: str
    name: str
    description: str
    source: str  # "seed" or "discovered"
    created_at: datetime
    is_functional: bool = False  # True → at most one active value per (subject, predicate)


@dataclass
class EntityAlias:
    entity_id: str
    alias: str
    source: str
    created_at: datetime
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class EntityIdentifier:
    entity_id: str
    namespace: str
    value: str
    source: str
    created_at: datetime
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class EntityAttribute:
    entity_id: str
    attribute_type: str
    value: str
    asserted_at: datetime
    valid_from: datetime
    provenance: str
    valid_until: datetime | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class EntityMention:
    entity_id: str
    source_kind: str
    source_id: str
    asserted_at: datetime
    snippet: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class EntityType:
    name: str
    description: str
    source: str
    created_at: datetime
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class AttributeType:
    name: str
    description: str
    value_type: str
    source: str
    created_at: datetime
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Topic:
    id: str
    name: str
    description: str
    source: str  # "seed" or "discovered"
    created_at: datetime


@dataclass
class Relation:
    id: str
    subject_id: str  # entity or knowledge_unit id
    predicate: str  # relation type name
    object_id: str  # entity or knowledge_unit id
    asserted_at: datetime
    valid_from: datetime
    valid_until: datetime | None
    provenance: str
    extraction_count: int = 1  # times asserted/re-asserted by extraction (write-side reinforcement)
    retrieval_count: int = 0  # times surfaced in a briefing (read-side usage)


@dataclass
class Briefing:
    units: list[KnowledgeUnit]
    entities: list["Entity"] = field(default_factory=list)
    relations: list["RenderedRelation"] = field(default_factory=list)
    topic_names: list[str] = field(default_factory=list)

    def format(self) -> str:
        sections: list[str] = []

        if self.topic_names:
            sections.append("Topics: " + ", ".join(self.topic_names))

        if self.entities:
            entity_lines = ["Entities:"]
            for e in self.entities:
                qid = f" ({e.wikidata_id})" if e.wikidata_id else ""
                desc = f": {e.description}" if e.description else ""
                entity_lines.append(f"- {e.label}{qid}{desc}")
            sections.append("\n".join(entity_lines))

        if self.units:
            fact_lines = ["Facts:"]
            for u in self.units:
                fact_lines.append(f"- ({u.type.value}) {u.content}")
            sections.append("\n".join(fact_lines))

        if self.relations:
            rel_lines = ["Relations:"]
            for r in self.relations:
                rel_lines.append(f"- {r.subject} -[{r.predicate}]-> {r.object}")
            sections.append("\n".join(rel_lines))

        if not sections:
            return "[MEMORY]\n[/MEMORY]"

        body = "\n\n".join(sections)
        note = (
            "Note: memory has been pre-fetched for the current query. "
            "Only call memory_search for substantively different topics not covered above."
        )
        return f"[MEMORY]\n{note}\n\n{body}\n[/MEMORY]"


@dataclass
class RenderedRelation:
    """A relation formatted for briefing display with resolved endpoint labels."""

    subject: str
    predicate: str
    object: str


# Avoid a hard import cycle by importing Entity lazily for type hints.
from tpasini.recordatio.entity import Entity  # noqa: E402
