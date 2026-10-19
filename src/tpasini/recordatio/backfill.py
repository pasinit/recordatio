"""Dry-run entity backfill and audit tooling."""

import argparse
import json
import re
import shutil
import sqlite3
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import (
    AttributeType,
    EntityAlias,
    EntityAttribute,
    EntityIdentifier,
    EntityMention,
)
from tpasini.recordatio.storage.base import SQLiteStore


@dataclass(frozen=True)
class UnitRow:
    id: str
    type: str
    content: str
    asserted_at: str


@dataclass(frozen=True)
class EntityRow:
    id: str
    label: str
    wikidata_id: str | None
    wikipedia_url: str | None
    description: str | None
    asserted_at: str


@dataclass(frozen=True)
class RelationRow:
    id: str
    subject_id: str
    predicate: str
    object_id: str
    asserted_at: str
    valid_until: str | None


@dataclass(frozen=True)
class AliasProposal:
    label: str
    confidence: float
    source: str
    unit_id: str | None = None


@dataclass(frozen=True)
class IdentifierProposal:
    kind: str
    value: str
    confidence: float
    source: str


@dataclass(frozen=True)
class AttributeProposal:
    predicate: str
    value: str
    confidence: float
    source_relation_id: str


@dataclass(frozen=True)
class RelationProposal:
    predicate: str
    object_label: str
    confidence: float
    source_relation_id: str


@dataclass(frozen=True)
class MentionProposal:
    unit_id: str
    confidence: float
    matched_text: str
    source: str


@dataclass(frozen=True)
class ConflictProposal:
    kind: str
    message: str
    confidence: float
    evidence: list[str]


@dataclass
class EntityProposal:
    canonical_label: str
    confidence: float
    existing_entity_ids: list[str] = field(default_factory=list)
    aliases: list[AliasProposal] = field(default_factory=list)
    identifiers: list[IdentifierProposal] = field(default_factory=list)
    attributes: list[AttributeProposal] = field(default_factory=list)
    relations: list[RelationProposal] = field(default_factory=list)
    mentions: list[MentionProposal] = field(default_factory=list)
    source_unit_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BackfillReport:
    agent_id: str
    generated_at: str
    mode: str
    proposed_entities: list[EntityProposal]
    conflicts: list[ConflictProposal]
    notes: list[str]


@dataclass(frozen=True)
class ApplyResult:
    backup_path: str
    audit_path: str
    inserted_entities: int
    inserted_aliases: int
    inserted_identifiers: int
    inserted_attributes: int
    inserted_attribute_types: int
    inserted_entity_mentions: int
    inserted_mentions: int
    skipped_aliases: int
    skipped_identifiers: int
    skipped_attributes: int
    skipped_relations: int


_MAX_ONTOLOGY_VALUE_CHARS = 160
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'_-]*|[A-Z]{2,})"
    r"(?:\s+(?:[A-Z][A-Za-z0-9'_-]*|[A-Z]{2,}|\d+[A-Za-z]?)){0,5}\b"
)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s)>\]\"']+")
_USER_IDENTITY_PATTERNS = (
    re.compile(
        r"\b(?:the\s+user's|user's|my)\s+name\s+is\s+"
        r"([A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*){0,4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+user|user|I)\s+"
        r"(?:am|is|am\s+called|is\s+called|go(?:es)?\s+by)\s+"
        r"([A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*){0,4})\b",
        re.IGNORECASE,
    ),
)
_ALIAS_STOPWORDS = {
    "A",
    "An",
    "As",
    "At",
    "For",
    "He",
    "I",
    "In",
    "It",
    "She",
    "The",
    "They",
    "This",
    "We",
    "When",
    "Where",
    "User",
}


def audit_db(db_path: Path, agent_id: str, *, generated_at: datetime | None = None) -> BackfillReport:
    """Read *db_path* without mutation and return proposed entity backfills."""
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = _connect_readonly(db_path)
    try:
        units = _load_units(conn, agent_id)
        entities = _load_entities(conn, agent_id)
        relations = _load_relations(conn, agent_id)
        functional_predicates = _load_functional_predicates(conn)
    finally:
        conn.close()

    proposal_by_key = _seed_entity_proposals(entities)
    _add_text_candidates(proposal_by_key, units)
    _add_identity_aliases(proposal_by_key, units)
    _add_entity_mentions(proposal_by_key, units)

    entity_by_id = {entity.id: entity for entity in entities}
    key_by_entity_id = {
        entity.id: _canonical_key(entity.label)
        for entity in entities
    }
    _add_relation_proposals(proposal_by_key, relations, entity_by_id, key_by_entity_id, functional_predicates)

    conflicts = _find_conflicts(
        proposal_by_key,
        relations,
        entity_by_id,
        key_by_entity_id,
        functional_predicates,
    )
    proposed_entities = sorted(
        proposal_by_key.values(),
        key=lambda proposal: (-proposal.confidence, proposal.canonical_label.lower()),
    )
    _dedupe_nested(proposed_entities)

    generated = generated_at or datetime.now(UTC)
    notes = [
        "Dry-run is read-only and does not mutate the database.",
        "Apply mode creates ontology metadata rows through the storage APIs and keeps legacy rows intact.",
        "Legacy mentions relations are still written for retrieval compatibility until #71 removes that dependency.",
    ]
    return BackfillReport(
        agent_id=agent_id,
        generated_at=generated.isoformat(),
        mode="dry-run",
        proposed_entities=proposed_entities,
        conflicts=conflicts,
        notes=notes,
    )


def apply_existing_schema_backfill(
    db_path: Path,
    agent_id: str,
    report: BackfillReport,
    *,
    audit_path: Path | None = None,
    backup_path: Path | None = None,
    entity_confidence_threshold: float = 0.9,
    mention_confidence_threshold: float = 0.9,
    alias_confidence_threshold: float = 0.75,
    identifier_confidence_threshold: float = 0.7,
    attribute_confidence_threshold: float = 0.85,
) -> ApplyResult:
    """Apply high-confidence ontology backfills while preserving legacy rows."""
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    backup = backup_path or db_path.with_suffix(db_path.suffix + f".bak-{stamp}")
    audit = audit_path or db_path.with_suffix(db_path.suffix + f".backfill-audit-{stamp}.json")
    shutil.copy2(db_path, backup)
    audit.write_text(report_to_json(report), encoding="utf-8")

    store = SQLiteStore(db_path)
    conn = store._conn
    try:
        entity_ids = _current_entity_ids_by_label(conn, agent_id)
        unit_ids = _current_active_unit_ids(conn, agent_id)
        active_relation_ids = _current_active_relation_ids(conn, agent_id)
        _ensure_mentions_relation_type(conn)

        inserted_entities = 0
        inserted_aliases = 0
        inserted_identifiers = 0
        inserted_attributes = 0
        inserted_attribute_types = 0
        inserted_entity_mentions = 0
        inserted_mentions = 0
        skipped_aliases = 0
        skipped_identifiers = 0
        skipped_attributes = 0
        now_iso = datetime.now(UTC).isoformat()
        now = datetime.fromisoformat(now_iso)

        for proposal in report.proposed_entities:
            entity_id = entity_ids.get(_canonical_key(proposal.canonical_label))
            if entity_id is None and proposal.confidence >= entity_confidence_threshold:
                entity = Entity(
                    id=str(uuid.uuid4()),
                    label=proposal.canonical_label,
                    asserted_at=now,
                    type="concept",
                    canonical_label=proposal.canonical_label,
                    canonical_key=f"concept:{_canonical_key(proposal.canonical_label).replace(' ', '-')}",
                )
                store.add_entity(entity, agent_id)
                entity_ids[_canonical_key(proposal.canonical_label)] = entity.id
                entity_id = entity.id
                inserted_entities += 1
            if entity_id is None:
                skipped_aliases += len(proposal.aliases)
                skipped_identifiers += len(proposal.identifiers)
                skipped_attributes += len(proposal.attributes)
                continue

            for alias in proposal.aliases:
                if alias.confidence < alias_confidence_threshold or _canonical_key(alias.label) == _canonical_key(
                    proposal.canonical_label
                ):
                    skipped_aliases += 1
                    continue
                if not _entity_alias_exists(store, alias.label, entity_id, agent_id):
                    inserted_aliases += 1
                store.upsert_entity_alias(
                    EntityAlias(
                        entity_id=entity_id,
                        alias=alias.label,
                        source=f"backfill:{alias.source}",
                        created_at=now,
                    ),
                    agent_id,
                )

            for identifier in proposal.identifiers:
                if identifier.confidence < identifier_confidence_threshold:
                    skipped_identifiers += 1
                    continue
                if not _entity_identifier_exists(store, identifier.kind, identifier.value, entity_id, agent_id):
                    inserted_identifiers += 1
                store.upsert_entity_identifier(
                    EntityIdentifier(
                        entity_id=entity_id,
                        namespace=identifier.kind,
                        value=identifier.value,
                        source=f"backfill:{identifier.source}",
                        created_at=now,
                    ),
                    agent_id,
                )

            for attribute in proposal.attributes:
                if (
                    attribute.confidence < attribute_confidence_threshold
                    or attribute.source_relation_id not in active_relation_ids
                    or len(attribute.value) > _MAX_ONTOLOGY_VALUE_CHARS
                ):
                    skipped_attributes += 1
                    continue
                if not _attribute_type_exists(store, attribute.predicate, agent_id):
                    inserted_attribute_types += 1
                    store.upsert_attribute_type(
                        AttributeType(
                            name=attribute.predicate,
                            description=f"Backfilled from legacy relation {attribute.predicate}",
                            value_type="text",
                            source="backfill:audit",
                            created_at=now,
                        ),
                        agent_id,
                    )
                if not _entity_attribute_exists(store, entity_id, attribute.predicate, attribute.value, agent_id):
                    inserted_attributes += 1
                store.upsert_entity_attribute(
                    EntityAttribute(
                        entity_id=entity_id,
                        attribute_type=attribute.predicate,
                        value=attribute.value,
                        asserted_at=now,
                        valid_from=now,
                        provenance=f"backfill:relation:{attribute.source_relation_id}",
                    ),
                    agent_id,
                )

            for mention in proposal.mentions:
                if mention.confidence < mention_confidence_threshold or mention.unit_id not in unit_ids:
                    continue
                if not _entity_mention_exists(store, entity_id, mention.unit_id, agent_id):
                    inserted_entity_mentions += 1
                store.upsert_entity_mention(
                    EntityMention(
                        entity_id=entity_id,
                        source_kind="knowledge_unit",
                        source_id=mention.unit_id,
                        snippet=mention.matched_text,
                        asserted_at=now,
                    ),
                    agent_id,
                )
                if _active_relation_exists(conn, agent_id, mention.unit_id, "mentions", entity_id):
                    continue
                conn.execute(
                    """
                    INSERT INTO relations
                        (id, subject_id, predicate, object_id, asserted_at,
                         valid_from, valid_until, provenance, agent_id)
                    VALUES (?, ?, 'mentions', ?, ?, ?, NULL, 'backfill:audit', ?)
                    """,
                    (str(uuid.uuid4()), mention.unit_id, entity_id, now_iso, now_iso, agent_id),
                )
                conn.commit()
                inserted_mentions += 1
    finally:
        store._conn.close()

    return ApplyResult(
        backup_path=str(backup),
        audit_path=str(audit),
        inserted_entities=inserted_entities,
        inserted_aliases=inserted_aliases,
        inserted_identifiers=inserted_identifiers,
        inserted_attributes=inserted_attributes,
        inserted_attribute_types=inserted_attribute_types,
        inserted_entity_mentions=inserted_entity_mentions,
        inserted_mentions=inserted_mentions,
        skipped_aliases=skipped_aliases,
        skipped_identifiers=skipped_identifiers,
        skipped_attributes=skipped_attributes,
        skipped_relations=sum(len(proposal.relations) for proposal in report.proposed_entities),
    )


def report_to_json(report: BackfillReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _load_units(conn: sqlite3.Connection, agent_id: str) -> list[UnitRow]:
    rows = conn.execute(
        """
        SELECT id, type, content, asserted_at
        FROM knowledge_units
        WHERE agent_id = ? AND valid_until IS NULL
        ORDER BY asserted_at, rowid
        """,
        (agent_id,),
    ).fetchall()
    return [
        UnitRow(id=row["id"], type=row["type"], content=row["content"], asserted_at=row["asserted_at"])
        for row in rows
    ]


def _load_entities(conn: sqlite3.Connection, agent_id: str) -> list[EntityRow]:
    rows = conn.execute(
        """
        SELECT id, label, wikidata_id, wikipedia_url, description, asserted_at
        FROM entities
        WHERE agent_id = ?
        ORDER BY asserted_at, rowid
        """,
        (agent_id,),
    ).fetchall()
    return [
        EntityRow(
            id=row["id"],
            label=row["label"],
            wikidata_id=row["wikidata_id"],
            wikipedia_url=row["wikipedia_url"],
            description=row["description"],
            asserted_at=row["asserted_at"],
        )
        for row in rows
    ]


def _load_relations(conn: sqlite3.Connection, agent_id: str) -> list[RelationRow]:
    rows = conn.execute(
        """
        SELECT id, subject_id, predicate, object_id, asserted_at, valid_until
        FROM relations
        WHERE agent_id = ?
        ORDER BY asserted_at, rowid
        """,
        (agent_id,),
    ).fetchall()
    return [
        RelationRow(
            id=row["id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            asserted_at=row["asserted_at"],
            valid_until=row["valid_until"],
        )
        for row in rows
    ]


def _load_functional_predicates(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM relation_types WHERE is_functional = 1").fetchall()
    return {row["name"] for row in rows}


def _seed_entity_proposals(entities: list[EntityRow]) -> dict[str, EntityProposal]:
    proposal_by_key: dict[str, EntityProposal] = {}
    for entity in entities:
        key = _canonical_key(entity.label)
        proposal = proposal_by_key.setdefault(
            key,
            EntityProposal(canonical_label=_canonical_label(entity.label), confidence=1.0),
        )
        proposal.existing_entity_ids.append(entity.id)
        if entity.label != proposal.canonical_label:
            proposal.aliases.append(AliasProposal(entity.label, 1.0, "existing_entity", None))
        if entity.wikidata_id:
            proposal.identifiers.append(
                IdentifierProposal("wikidata_id", entity.wikidata_id, 1.0, "entities.wikidata_id")
            )
        if entity.wikipedia_url:
            proposal.identifiers.append(
                IdentifierProposal("wikipedia_url", entity.wikipedia_url, 1.0, "entities.wikipedia_url")
            )
    return proposal_by_key


def _add_text_candidates(proposal_by_key: dict[str, EntityProposal], units: list[UnitRow]) -> None:
    mentions_by_label: dict[str, list[str]] = defaultdict(list)
    display_by_key: dict[str, str] = {}
    for unit in units:
        for match in _CAPITALIZED_PHRASE_RE.finditer(unit.content):
            label = match.group(0).strip()
            if label in _ALIAS_STOPWORDS or len(label) < 3:
                continue
            key = _canonical_key(label)
            mentions_by_label[key].append(unit.id)
            display_by_key.setdefault(key, _canonical_label(label))

    for key, unit_ids in mentions_by_label.items():
        unique_unit_ids = sorted(set(unit_ids))
        if len(unique_unit_ids) < 2:
            continue
        proposal = proposal_by_key.setdefault(
            key,
            EntityProposal(canonical_label=display_by_key[key], confidence=0.55),
        )
        proposal.confidence = max(proposal.confidence, 0.55)
        proposal.source_unit_ids.extend(unique_unit_ids)
        for unit_id in unique_unit_ids:
            proposal.mentions.append(MentionProposal(unit_id, 0.55, display_by_key[key], "repeated_text_candidate"))


def _add_identity_aliases(proposal_by_key: dict[str, EntityProposal], units: list[UnitRow]) -> None:
    user_key = _canonical_key("User")
    for unit in units:
        for pattern in _USER_IDENTITY_PATTERNS:
            for match in pattern.finditer(unit.content):
                alias = _clean_alias(match.group(1))
                if not alias:
                    continue
                proposal = proposal_by_key.setdefault(
                    user_key,
                    EntityProposal(canonical_label="User", confidence=0.8),
                )
                proposal.confidence = max(proposal.confidence, 0.8)
                proposal.source_unit_ids.append(unit.id)
                proposal.aliases.append(AliasProposal(alias, 0.8, "identity_text_pattern", unit.id))


def _add_entity_mentions(proposal_by_key: dict[str, EntityProposal], units: list[UnitRow]) -> None:
    for proposal in proposal_by_key.values():
        labels = {proposal.canonical_label}
        labels.update(alias.label for alias in proposal.aliases)
        for label in labels:
            if len(label) < 2:
                continue
            pattern = _label_pattern(label)
            for unit in units:
                if pattern.search(unit.content):
                    proposal.mentions.append(MentionProposal(unit.id, 0.95, label, "label_text_match"))
                    proposal.source_unit_ids.append(unit.id)
                    for identifier in _extract_identifiers_near_label(unit.content, label):
                        proposal.identifiers.append(identifier)


def _add_relation_proposals(
    proposal_by_key: dict[str, EntityProposal],
    relations: list[RelationRow],
    entity_by_id: dict[str, EntityRow],
    key_by_entity_id: dict[str, str],
    functional_predicates: set[str],
) -> None:
    for relation in relations:
        if relation.valid_until is not None or relation.subject_id not in key_by_entity_id:
            continue
        proposal = proposal_by_key.get(key_by_entity_id[relation.subject_id])
        if proposal is None:
            continue
        obj = entity_by_id.get(relation.object_id)
        value = obj.label if obj else relation.object_id
        confidence = 0.9 if obj else 0.65
        proposal.relations.append(RelationProposal(relation.predicate, value, confidence, relation.id))
        if relation.predicate in functional_predicates:
            proposal.attributes.append(AttributeProposal(relation.predicate, value, confidence, relation.id))


def _find_conflicts(
    proposal_by_key: dict[str, EntityProposal],
    relations: list[RelationRow],
    entity_by_id: dict[str, EntityRow],
    key_by_entity_id: dict[str, str],
    functional_predicates: set[str],
) -> list[ConflictProposal]:
    conflicts: list[ConflictProposal] = []
    conflicts.extend(_identifier_conflicts(proposal_by_key))
    conflicts.extend(_functional_relation_conflicts(relations, entity_by_id, key_by_entity_id, functional_predicates))
    return sorted(conflicts, key=lambda conflict: (conflict.kind, conflict.message))


def _identifier_conflicts(proposal_by_key: dict[str, EntityProposal]) -> list[ConflictProposal]:
    seen: dict[tuple[str, str], list[str]] = defaultdict(list)
    for proposal in proposal_by_key.values():
        for identifier in proposal.identifiers:
            seen[(identifier.kind, identifier.value)].append(proposal.canonical_label)

    conflicts: list[ConflictProposal] = []
    for (kind, value), labels in seen.items():
        unique_labels = sorted(set(labels))
        if len(unique_labels) <= 1:
            continue
        conflicts.append(
            ConflictProposal(
                kind="identifier_collision",
                message=f"{kind} {value!r} appears on multiple proposed entities",
                confidence=0.35,
                evidence=unique_labels,
            )
        )
    return conflicts


def _functional_relation_conflicts(
    relations: list[RelationRow],
    entity_by_id: dict[str, EntityRow],
    key_by_entity_id: dict[str, str],
    functional_predicates: set[str],
) -> list[ConflictProposal]:
    grouped: dict[tuple[str, str], list[RelationRow]] = defaultdict(list)
    for relation in relations:
        if relation.valid_until is not None or relation.predicate not in functional_predicates:
            continue
        if relation.subject_id not in key_by_entity_id:
            continue
        grouped[(relation.subject_id, relation.predicate)].append(relation)

    conflicts: list[ConflictProposal] = []
    for (subject_id, predicate), rels in grouped.items():
        object_ids = {relation.object_id for relation in rels}
        if len(object_ids) <= 1:
            continue
        subject = entity_by_id[subject_id].label
        values = sorted(entity_by_id[obj].label if obj in entity_by_id else obj for obj in object_ids)
        conflicts.append(
            ConflictProposal(
                kind="functional_attribute_conflict",
                message=f"{subject!r} has multiple active values for {predicate!r}",
                confidence=0.25,
                evidence=values,
            )
        )
    return conflicts


def _dedupe_nested(proposals: list[EntityProposal]) -> None:
    for proposal in proposals:
        proposal.existing_entity_ids = sorted(set(proposal.existing_entity_ids))
        proposal.source_unit_ids = sorted(set(proposal.source_unit_ids))
        proposal.aliases = sorted(set(proposal.aliases), key=lambda alias: (-alias.confidence, alias.label.lower()))
        proposal.identifiers = sorted(
            set(proposal.identifiers),
            key=lambda identifier: (identifier.kind, identifier.value),
        )
        proposal.attributes = sorted(
            set(proposal.attributes),
            key=lambda attribute: (attribute.predicate, attribute.value, attribute.source_relation_id),
        )
        proposal.relations = sorted(
            set(proposal.relations),
            key=lambda relation: (relation.predicate, relation.object_label, relation.source_relation_id),
        )
        proposal.mentions = sorted(
            set(proposal.mentions),
            key=lambda mention: (mention.unit_id, -mention.confidence, mention.matched_text.lower()),
        )


def _canonical_key(label: str) -> str:
    return " ".join(_WORD_RE.findall(label.casefold()))


def _canonical_label(label: str) -> str:
    return " ".join(label.strip().split())


def _clean_alias(value: str) -> str:
    alias = _canonical_label(value).strip(" .,:;!?'\"")
    if not alias or alias in _ALIAS_STOPWORDS:
        return ""
    return alias


def _label_pattern(label: str) -> re.Pattern:
    escaped = re.escape(label)
    return re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", re.IGNORECASE)


def _extract_identifiers_near_label(content: str, label: str) -> list[IdentifierProposal]:
    if not _label_pattern(label).search(content):
        return []
    identifiers: list[IdentifierProposal] = []
    for email in _EMAIL_RE.findall(content):
        identifiers.append(IdentifierProposal("email", email, 0.75, "unit_text_near_label"))
    for url in _URL_RE.findall(content):
        identifiers.append(IdentifierProposal("url", url, 0.7, "unit_text_near_label"))
    return identifiers


def _current_entity_ids_by_label(conn: sqlite3.Connection, agent_id: str) -> dict[str, str]:
    rows = conn.execute("SELECT id, label FROM entities WHERE agent_id = ?", (agent_id,)).fetchall()
    return {_canonical_key(row["label"]): row["id"] for row in rows}


def _current_active_unit_ids(conn: sqlite3.Connection, agent_id: str) -> set[str]:
    rows = conn.execute(
        "SELECT id FROM knowledge_units WHERE agent_id = ? AND valid_until IS NULL",
        (agent_id,),
    ).fetchall()
    return {row["id"] for row in rows}


def _current_active_relation_ids(conn: sqlite3.Connection, agent_id: str) -> set[str]:
    rows = conn.execute(
        "SELECT id FROM relations WHERE agent_id = ? AND valid_until IS NULL",
        (agent_id,),
    ).fetchall()
    return {row["id"] for row in rows}


def _ensure_mentions_relation_type(conn: sqlite3.Connection) -> None:
    now_iso = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT OR IGNORE INTO relation_types
            (id, name, description, source, created_at, is_functional)
        VALUES (?, 'mentions', 'unit references an entity in passing', 'seed', ?, 0)
        """,
        (str(uuid.uuid4()), now_iso),
    )


def _active_relation_exists(
    conn: sqlite3.Connection,
    agent_id: str,
    subject_id: str,
    predicate: str,
    object_id: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM relations
        WHERE agent_id = ? AND subject_id = ? AND predicate = ? AND object_id = ?
          AND valid_until IS NULL
        LIMIT 1
        """,
        (agent_id, subject_id, predicate, object_id),
    ).fetchone()
    return row is not None


def _entity_alias_exists(store: SQLiteStore, alias: str, entity_id: str, agent_id: str) -> bool:
    return any(item.entity_id == entity_id for item in store.find_entity_aliases(alias, agent_id))


def _entity_identifier_exists(
    store: SQLiteStore,
    namespace: str,
    value: str,
    entity_id: str,
    agent_id: str,
) -> bool:
    return any(item.entity_id == entity_id for item in store.find_entity_identifiers(namespace, value, agent_id))


def _entity_attribute_exists(
    store: SQLiteStore,
    entity_id: str,
    attribute_type: str,
    value: str,
    agent_id: str,
) -> bool:
    return any(
        item.attribute_type == attribute_type and item.value == value
        for item in store.list_entity_attributes(entity_id, agent_id, attribute_type, active_only=True)
    )


def _attribute_type_exists(store: SQLiteStore, name: str, agent_id: str) -> bool:
    return store.get_attribute_type_by_name(name, agent_id) is not None


def _entity_mention_exists(store: SQLiteStore, entity_id: str, unit_id: str, agent_id: str) -> bool:
    return any(
        item.entity_id == entity_id
        for item in store.find_entity_mentions_by_source("knowledge_unit", unit_id, agent_id)
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit and backfill canonical entities from an agentmem DB.")
    parser.add_argument("--db", type=Path, default=Path.home() / ".agentmem" / "memories.db")
    parser.add_argument("--agent-id", required=True, help="agent namespace to audit")
    parser.add_argument("--output", type=Path, help="write dry-run report JSON to this path")
    parser.add_argument("--apply", action="store_true", help="apply high-confidence ontology backfills")
    parser.add_argument(
        "--enable-ontology-apply",
        action="store_true",
        help="required with --apply; acknowledges the migration writes ontology metadata and keeps legacy rows",
    )
    parser.add_argument(
        "--enable-existing-schema-apply",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--audit-file", type=Path, help="audit JSON path for --apply")
    args = parser.parse_args(argv)

    if args.apply and not (args.enable_ontology_apply or args.enable_existing_schema_apply):
        parser.error("--apply requires --enable-ontology-apply")

    report = audit_db(args.db, args.agent_id)
    report_json = report_to_json(report)
    if args.output:
        args.output.write_text(report_json, encoding="utf-8")
    elif not args.apply:
        print(report_json, end="")

    if not args.apply:
        return
    result = apply_existing_schema_backfill(args.db, args.agent_id, report, audit_path=args.audit_file)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
