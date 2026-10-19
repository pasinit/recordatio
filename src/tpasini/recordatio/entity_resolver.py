"""Deterministic-first entity resolution."""

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Protocol

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.entity_identity import EntityIdentityJudge, EntityRef
from tpasini.recordatio.models import EntityAlias, EntityIdentifier
from tpasini.recordatio.storage.base import Store

type IdentifierMap = dict[str, str]

_WORD_RE = re.compile(r"[^\w]+")

# A qualifier (postcode, containing place, parenthetical gloss) adds few tokens;
# beyond this gap a longer label is a different, more specific name, not a
# variant. Used only to BLOCK candidates for the judge — never to decide.
_JUDGE_MAX_TOKEN_GAP = 2


class EntityMatchKind(StrEnum):
    EXACT_IDENTIFIER = "exact_identifier"
    CANONICAL_KEY = "canonical_key"
    ALIAS = "alias"
    JUDGE = "judge"
    FUZZY = "fuzzy"
    CREATED = "created"
    AMBIGUOUS = "ambiguous"


class EntityResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    CREATED = "created"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class EntityMention:
    """Structured mention supplied to the resolver boundary.

    ``identifiers`` are external identity keys already observed for this
    mention, keyed by scheme (for example ``{"wikidata": "Q42"}``). The
    generic resolver only consumes them; populating and normalizing identifier
    fields from model output belongs to the extraction contract work in #67.
    """

    label: str
    type: str | None = None
    canonical_key: str | None = None
    aliases: list[str] = field(default_factory=list)
    identifiers: IdentifierMap = field(default_factory=dict)

    def labels_for_matching(self) -> list[str]:
        labels: list[str] = []
        for label in [self.label, *self.aliases]:
            normalized = normalize_label(label)
            if normalized and normalized not in {normalize_label(item) for item in labels}:
                labels.append(label)
        return labels


@dataclass(frozen=True)
class EntityRecord:
    id: str
    label: str
    type: str | None = None
    canonical_key: str | None = None
    aliases: list[str] = field(default_factory=list)
    identifiers: IdentifierMap = field(default_factory=dict)

    def labels_for_matching(self) -> list[str]:
        labels: list[str] = []
        for label in [self.label, *self.aliases]:
            normalized = normalize_label(label)
            if normalized and normalized not in {normalize_label(item) for item in labels}:
                labels.append(label)
        return labels


@dataclass(frozen=True)
class EntityCandidateAudit:
    entity_id: str
    label: str
    type: str | None
    match: EntityMatchKind
    score: float
    canonical_key: str | None = None


@dataclass(frozen=True)
class EntityResolutionAudit:
    match: EntityMatchKind
    reason: str
    candidates: list[EntityCandidateAudit] = field(default_factory=list)


@dataclass(frozen=True)
class EntityResolutionResult:
    status: EntityResolutionStatus
    entity: EntityRecord | None
    confidence: float
    audit: EntityResolutionAudit


class EntityResolverRepository(Protocol):
    def find_by_identifier(
        self, agent_id: str, entity_type: str | None, scheme: str, value: str
    ) -> list[EntityRecord]: ...

    def find_by_canonical_key(
        self, agent_id: str, entity_type: str | None, canonical_key: str
    ) -> list[EntityRecord]: ...

    def find_by_alias(self, agent_id: str, entity_type: str | None, alias: str) -> list[EntityRecord]: ...

    def list_by_type(self, agent_id: str, entity_type: str | None) -> list[EntityRecord]: ...

    def create(self, agent_id: str, mention: EntityMention, canonical_key: str | None) -> EntityRecord: ...

    def merge_metadata(
        self,
        agent_id: str,
        entity_id: str,
        aliases: list[str],
        identifiers: IdentifierMap,
        canonical_key: str | None,
    ) -> EntityRecord: ...


class EntityResolver:
    def __init__(
        self,
        repository: EntityResolverRepository,
        *,
        fuzzy_threshold: float = 0.86,
        ambiguity_margin: float = 0.03,
        judge: EntityIdentityJudge | None = None,
        judge_recall_floor: float = 0.5,
        judge_top_k: int = 3,
    ) -> None:
        self.repository = repository
        self.fuzzy_threshold = fuzzy_threshold
        self.ambiguity_margin = ambiguity_margin
        # When a judge is injected, a borderline mention (below the fuzzy
        # auto-merge bar but token-overlapping an existing entity) is adjudicated
        # by the general identity judge instead of the old tailored dedup key.
        self.judge = judge
        self.judge_recall_floor = judge_recall_floor
        self.judge_top_k = judge_top_k

    def resolve(self, mention: EntityMention, agent_id: str) -> EntityResolutionResult:
        canonical_key = mention.canonical_key or default_canonical_key(mention.type, mention.label)

        return (
            self._resolve_by_identifier(mention, agent_id, canonical_key)
            or self._resolve_by_canonical_key(mention, agent_id, canonical_key)
            or self._resolve_by_alias(mention, agent_id, canonical_key)
            or self._resolve_by_fuzzy_match(mention, agent_id, canonical_key)
            or self._resolve_by_judge(mention, agent_id, canonical_key)
            or self._create_new(mention, agent_id, canonical_key)
        )

    def _resolve_by_identifier(
        self, mention: EntityMention, agent_id: str, canonical_key: str | None
    ) -> EntityResolutionResult | None:
        for scheme, value in mention.identifiers.items():
            matches = self.repository.find_by_identifier(agent_id, mention.type, scheme, value)
            if matches:
                return self._resolve_deterministic(
                    matches,
                    mention,
                    agent_id,
                    EntityMatchKind.EXACT_IDENTIFIER,
                    f"identifier {scheme} matched",
                    canonical_key,
                )
        return None

    def _resolve_by_canonical_key(
        self, mention: EntityMention, agent_id: str, canonical_key: str | None
    ) -> EntityResolutionResult | None:
        if canonical_key:
            matches = self.repository.find_by_canonical_key(agent_id, mention.type, canonical_key)
            if matches:
                return self._resolve_deterministic(
                    matches,
                    mention,
                    agent_id,
                    EntityMatchKind.CANONICAL_KEY,
                    "canonical key matched",
                    canonical_key,
                )
        return None

    def _resolve_by_alias(
        self, mention: EntityMention, agent_id: str, canonical_key: str | None
    ) -> EntityResolutionResult | None:
        alias_matches: list[EntityRecord] = []
        for alias in mention.labels_for_matching():
            alias_matches.extend(self.repository.find_by_alias(agent_id, mention.type, alias))
        if alias_matches:
            return self._resolve_deterministic(
                alias_matches,
                mention,
                agent_id,
                EntityMatchKind.ALIAS,
                "alias matched within type",
                canonical_key,
            )
        return None

    def _resolve_by_judge(
        self, mention: EntityMention, agent_id: str, canonical_key: str | None
    ) -> EntityResolutionResult | None:
        """Attach a borderline mention to an existing entity the general identity
        judge deems the same real-world thing (``Elmway X1`` -> ``Elmway``),
        recording the incoming surface as an alias.

        Replaces the hand-tuned ``canonical_dedup_key`` stage. Runs only when a
        judge is injected and only AFTER fuzzy (so the cheap deterministic stages
        and a confident fuzzy match win first); it adjudicates the *borderline*
        band fuzzy left unmerged — candidates that token-overlap or are loosely
        similar but below the auto-merge bar. The first confident SAME binds; on
        no usable judge signal nothing merges and the mention creates a new
        entity (a missed merge is harmless; the dedup migration is the net).
        """
        if self.judge is None:
            return None
        candidates = self._judge_candidates(mention, agent_id)
        m_ref = EntityRef(label=mention.label, type=mention.type)
        for record in candidates:
            verdict = self.judge.same_entity(m_ref, EntityRef(label=record.label, type=record.type))
            if not verdict.same:
                continue
            merged = self.repository.merge_metadata(
                agent_id,
                record.id,
                mention.labels_for_matching(),
                mention.identifiers,
                canonical_key,
            )
            return EntityResolutionResult(
                status=EntityResolutionStatus.RESOLVED,
                entity=merged,
                confidence=verdict.confidence,
                audit=EntityResolutionAudit(
                    match=EntityMatchKind.JUDGE,
                    reason=f"identity judge: same entity ({verdict.reason})",
                    candidates=[_candidate(record, EntityMatchKind.JUDGE, verdict.confidence)],
                ),
            )
        return None

    def _judge_candidates(self, mention: EntityMention, agent_id: str) -> list[EntityRecord]:
        """The borderline existing entities worth asking the judge about: same
        scope+type, token-subset (within a small gap) or loosely similar, but not
        an identical surface (that bound earlier). Top-K by similarity."""
        m_norm = normalize_label(mention.label)
        m_tokens = frozenset(m_norm.split())
        scored: list[tuple[float, EntityRecord]] = []
        for record in self.repository.list_by_type(agent_id, mention.type):
            r_norm = normalize_label(record.label)
            if not r_norm or r_norm == m_norm:
                continue
            r_tokens = frozenset(r_norm.split())
            subset = bool(
                m_tokens
                and r_tokens
                and (m_tokens <= r_tokens or r_tokens <= m_tokens)
                and abs(len(m_tokens) - len(r_tokens)) <= _JUDGE_MAX_TOKEN_GAP
            )
            ratio = SequenceMatcher(a=m_norm, b=r_norm).ratio()
            if subset or ratio >= self.judge_recall_floor:
                scored.append((ratio, record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _ratio, record in scored[: self.judge_top_k]]

    def _resolve_by_fuzzy_match(
        self, mention: EntityMention, agent_id: str, canonical_key: str | None
    ) -> EntityResolutionResult | None:
        fuzzy = self._fuzzy_candidates(mention, agent_id)
        if not fuzzy:
            return None

        best = fuzzy[0]
        if best.score < self.fuzzy_threshold:
            return None

        close = [
            candidate
            for candidate in fuzzy
            if best.score - candidate.score <= self.ambiguity_margin
            and candidate.score >= self.fuzzy_threshold
        ]
        if len({candidate.entity_id for candidate in close}) > 1:
            return self._ambiguous(
                EntityMatchKind.FUZZY,
                "multiple fuzzy candidates were too close to choose safely",
                close,
            )

        record = self.repository.merge_metadata(
            agent_id,
            best.entity_id,
            mention.labels_for_matching(),
            mention.identifiers,
            canonical_key,
        )
        return EntityResolutionResult(
            status=EntityResolutionStatus.RESOLVED,
            entity=record,
            confidence=best.score,
            audit=EntityResolutionAudit(
                match=EntityMatchKind.FUZZY,
                reason="single fuzzy candidate above threshold",
                candidates=fuzzy[:5],
            ),
        )

    def _create_new(
        self, mention: EntityMention, agent_id: str, canonical_key: str | None
    ) -> EntityResolutionResult:
        created = self.repository.create(agent_id, mention, canonical_key)
        return EntityResolutionResult(
            status=EntityResolutionStatus.CREATED,
            entity=created,
            confidence=1.0,
            audit=EntityResolutionAudit(
                match=EntityMatchKind.CREATED,
                reason="no deterministic or fuzzy match found",
            ),
        )

    def _resolve_deterministic(
        self,
        matches: list[EntityRecord],
        mention: EntityMention,
        agent_id: str,
        match: EntityMatchKind,
        reason: str,
        canonical_key: str | None,
    ) -> EntityResolutionResult:
        unique = _unique_records(matches)
        audits = [_candidate(record, match, 1.0) for record in unique]
        if len(unique) > 1:
            return self._ambiguous(match, f"{reason}, but multiple entities matched", audits)

        record = self.repository.merge_metadata(
            agent_id,
            unique[0].id,
            mention.labels_for_matching(),
            mention.identifiers,
            canonical_key,
        )
        return EntityResolutionResult(
            status=EntityResolutionStatus.RESOLVED,
            entity=record,
            confidence=1.0,
            audit=EntityResolutionAudit(match=match, reason=reason, candidates=audits),
        )

    def _fuzzy_candidates(self, mention: EntityMention, agent_id: str) -> list[EntityCandidateAudit]:
        mention_labels = [normalize_label(label) for label in mention.labels_for_matching()]
        candidates: list[EntityCandidateAudit] = []
        for record in self.repository.list_by_type(agent_id, mention.type):
            candidate_labels = [normalize_label(label) for label in record.labels_for_matching()]
            score = max(
                (
                    SequenceMatcher(a=mention_label, b=candidate_label).ratio()
                    for mention_label in mention_labels
                    for candidate_label in candidate_labels
                    if mention_label and candidate_label
                ),
                default=0.0,
            )
            if score:
                candidates.append(_candidate(record, EntityMatchKind.FUZZY, score))
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates

    @staticmethod
    def _ambiguous(
        match: EntityMatchKind, reason: str, candidates: list[EntityCandidateAudit]
    ) -> EntityResolutionResult:
        return EntityResolutionResult(
            status=EntityResolutionStatus.AMBIGUOUS,
            entity=None,
            confidence=0.0,
            audit=EntityResolutionAudit(match=EntityMatchKind.AMBIGUOUS, reason=reason, candidates=candidates),
        )


class SQLiteEntityResolverRepository:
    """Resolver adapter backed by the ontology-aware SQLite store."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def find_by_identifier(
        self, agent_id: str, entity_type: str | None, scheme: str, value: str
    ) -> list[EntityRecord]:
        records = [
            self._record_from_entity_id(identifier.entity_id, agent_id, entity_type)
            for identifier in self.store.find_entity_identifiers(scheme, value, agent_id, entity_type)
        ]
        records = [record for record in records if record is not None]
        if records:
            return records
        if normalize_label(scheme) in {"wikidata", "wikidata_id"}:
            entity = self.store.get_entity_by_wikidata_id(value, agent_id)
            return [_record_from_entity(entity, entity_type)] if entity else []
        return []

    def find_by_canonical_key(
        self, agent_id: str, entity_type: str | None, canonical_key: str
    ) -> list[EntityRecord]:
        entity = self.store.get_entity_by_canonical_key(canonical_key, agent_id, entity_type)
        return [self._record_from_entity(entity, agent_id)] if entity else []

    def find_by_alias(self, agent_id: str, entity_type: str | None, alias: str) -> list[EntityRecord]:
        records = [
            self._record_from_entity_id(row.entity_id, agent_id, entity_type)
            for row in self.store.find_entity_aliases(alias, agent_id, entity_type)
        ]
        records = [record for record in records if record is not None]
        if records:
            return records
        entity = self.store.get_entity_by_label(alias, agent_id)
        return [self._record_from_entity(entity, agent_id)] if entity else []

    def list_by_type(self, agent_id: str, entity_type: str | None) -> list[EntityRecord]:
        return [
            self._record_from_entity(entity, agent_id)
            for entity in self.store.list_entities(agent_id, entity_type)
        ]

    def create(self, agent_id: str, mention: EntityMention, canonical_key: str | None) -> EntityRecord:
        entity = Entity(
            label=mention.label,
            type=mention.type or "concept",
            canonical_label=mention.label,
            canonical_key=canonical_key,
            wikidata_id=_wikidata_identifier(mention.identifiers),
            asserted_at=datetime.now(UTC),
        )
        self.store.add_entity(entity, agent_id)
        stored = self.store.get_entity_by_label(mention.label, agent_id) or entity
        return self.merge_metadata(
            agent_id,
            stored.id,
            mention.labels_for_matching(),
            mention.identifiers,
            canonical_key,
        )

    def merge_metadata(
        self,
        agent_id: str,
        entity_id: str,
        aliases: list[str],
        identifiers: IdentifierMap,
        canonical_key: str | None,
    ) -> EntityRecord:
        entity = self.store.get_entity_by_id(entity_id, agent_id)
        if entity is not None:
            now = datetime.now(UTC)
            for alias in aliases:
                if normalize_label(alias) != normalize_label(entity.label):
                    self.store.upsert_entity_alias(
                        EntityAlias(entity_id=entity_id, alias=alias, source="extraction", created_at=now),
                        agent_id,
                    )
            for scheme, value in identifiers.items():
                self.store.upsert_entity_identifier(
                    EntityIdentifier(
                        entity_id=entity_id,
                        namespace=scheme,
                        value=value,
                        source="extraction",
                        created_at=now,
                    ),
                    agent_id,
                )
            return self._record_from_entity(entity, agent_id, canonical_key)
        raise LookupError(f"entity {entity_id!r} not found for agent {agent_id!r}")

    def _record_from_entity_id(
        self,
        entity_id: str,
        agent_id: str,
        entity_type: str | None = None,
    ) -> EntityRecord | None:
        entity = self.store.get_entity_by_id(entity_id, agent_id)
        if entity is None or (entity_type is not None and entity.type != entity_type):
            return None
        return self._record_from_entity(entity, agent_id)

    def _record_from_entity(
        self,
        entity: Entity,
        agent_id: str,
        canonical_key: str | None = None,
    ) -> EntityRecord:
        aliases = [alias.alias for alias in self.store.list_entity_aliases(entity.id, agent_id)]
        identifiers = {
            identifier.namespace: identifier.value
            for identifier in self.store.list_entity_identifiers(entity.id, agent_id)
        }
        if entity.wikidata_id:
            identifiers.setdefault("wikidata", entity.wikidata_id)
        return EntityRecord(
            id=entity.id,
            label=entity.label,
            type=entity.type,
            canonical_key=canonical_key or entity.canonical_key or default_canonical_key(entity.type, entity.label),
            aliases=aliases,
            identifiers=identifiers,
        )


def stable_user_mention(agent_id: str, *, label: str = "User", aliases: list[str] | None = None) -> EntityMention:
    all_aliases = aliases if aliases is not None else []
    return EntityMention(
        label=label,
        type="person",
        canonical_key=f"agent_user:{agent_id}",
        aliases=all_aliases,
    )


def default_canonical_key(entity_type: str | None, label: str) -> str | None:
    normalized = normalize_label(label).replace(" ", "-")
    if not normalized:
        return None
    type_prefix = normalize_label(entity_type).replace(" ", "-") if entity_type else "entity"
    return f"{type_prefix}:{normalized}"


def normalize_label(value: str | None) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return _WORD_RE.sub(" ", normalized).strip()


def _unique_records(records: list[EntityRecord]) -> list[EntityRecord]:
    by_id: dict[str, EntityRecord] = {}
    for record in records:
        by_id.setdefault(record.id, record)
    return list(by_id.values())


def _candidate(record: EntityRecord, match: EntityMatchKind, score: float) -> EntityCandidateAudit:
    return EntityCandidateAudit(
        entity_id=record.id,
        label=record.label,
        type=record.type,
        match=match,
        score=score,
        canonical_key=record.canonical_key,
    )


def _record_from_entity(entity: Entity, entity_type: str | None) -> EntityRecord:
    return EntityRecord(
        id=entity.id,
        label=entity.label,
        type=entity_type,
        canonical_key=default_canonical_key(entity_type, entity.label),
        identifiers={"wikidata": entity.wikidata_id} if entity.wikidata_id else {},
    )


def _wikidata_identifier(identifiers: IdentifierMap) -> str | None:
    for scheme, value in identifiers.items():
        if normalize_label(scheme) in {"wikidata", "wikidata_id"}:
            return value
    return None


def _merge_labels(existing: list[str], incoming: list[str]) -> list[str]:
    merged = list(existing)
    normalized_existing = {normalize_label(item) for item in merged}
    for label in incoming:
        normalized = normalize_label(label)
        if normalized and normalized not in normalized_existing:
            merged.append(label)
            normalized_existing.add(normalized)
    return merged
