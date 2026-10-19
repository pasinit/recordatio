from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from tpasini.recordatio.entity_identity import EntityRef, IdentityRelation, IdentityVerdict
from tpasini.recordatio.entity_resolver import (
    EntityMention,
    EntityRecord,
    EntityResolutionStatus,
    EntityResolver,
    EntityResolverRepository,
    SQLiteEntityResolverRepository,
    normalize_label,
    stable_user_mention,
)
from tpasini.recordatio.storage.base import SQLiteStore


class StubJudge:
    """A predicate-driven identity judge for resolver wiring tests: ``predicate``
    decides SAME from the two surface labels. Counts calls so a test can assert
    the judge was (or was not) consulted."""

    def __init__(self, predicate: Callable[[str, str], bool]) -> None:
        self.predicate = predicate
        self.calls = 0

    def same_entity(self, a: EntityRef, b: EntityRef) -> IdentityVerdict:
        self.calls += 1
        same = self.predicate(a.label, b.label)
        relation = IdentityRelation.SAME if same else IdentityRelation.DIFFERENT
        return IdentityVerdict(relation, 0.9, "stub", "stub")


def _subset_judge() -> StubJudge:
    """SAME iff one label's tokens are a subset of the other's — a generic
    qualifier-variant stand-in (Elmway ⊂ Elmway X1)."""
    def predicate(a: str, b: str) -> bool:
        ta, tb = set(normalize_label(a).split()), set(normalize_label(b).split())
        return bool(ta and tb and (ta <= tb or tb <= ta))
    return StubJudge(predicate)


class InMemoryEntityRepository(EntityResolverRepository):
    def __init__(self, records: list[EntityRecord] | None = None) -> None:
        self.records: dict[str, EntityRecord] = {record.id: record for record in records or []}
        self.created = 0

    def find_by_identifier(
        self, agent_id: str, entity_type: str | None, scheme: str, value: str
    ) -> list[EntityRecord]:
        return [
            record
            for record in self._scope(entity_type)
            if record.identifiers.get(scheme) == value
        ]

    def find_by_canonical_key(
        self, agent_id: str, entity_type: str | None, canonical_key: str
    ) -> list[EntityRecord]:
        return [
            record
            for record in self._scope(entity_type)
            if record.canonical_key == canonical_key
        ]

    def find_by_alias(self, agent_id: str, entity_type: str | None, alias: str) -> list[EntityRecord]:
        wanted = normalize_label(alias)
        return [
            record
            for record in self._scope(entity_type)
            if wanted in {normalize_label(label) for label in record.labels_for_matching()}
        ]

    def list_by_type(self, agent_id: str, entity_type: str | None) -> list[EntityRecord]:
        return self._scope(entity_type)

    def create(self, agent_id: str, mention: EntityMention, canonical_key: str | None) -> EntityRecord:
        self.created += 1
        record = EntityRecord(
            id=f"created-{self.created}",
            label=mention.label,
            type=mention.type,
            canonical_key=canonical_key,
            aliases=mention.aliases,
            identifiers=mention.identifiers,
        )
        self.records[record.id] = record
        return record

    def merge_metadata(
        self,
        agent_id: str,
        entity_id: str,
        aliases: list[str],
        identifiers: dict[str, str],
        canonical_key: str | None,
    ) -> EntityRecord:
        record = self.records[entity_id]
        merged_aliases = list(record.aliases)
        seen = {normalize_label(alias) for alias in merged_aliases}
        for alias in aliases:
            normalized = normalize_label(alias)
            if normalized and normalized != normalize_label(record.label) and normalized not in seen:
                merged_aliases.append(alias)
                seen.add(normalized)
        merged = replace(
            record,
            aliases=merged_aliases,
            identifiers={**record.identifiers, **identifiers},
            canonical_key=record.canonical_key or canonical_key,
        )
        self.records[entity_id] = merged
        return merged

    def _scope(self, entity_type: str | None) -> list[EntityRecord]:
        return [
            record
            for record in self.records.values()
            if entity_type is None or record.type is None or record.type == entity_type
        ]


def test_resolves_by_exact_identifier_and_merges_metadata() -> None:
    repository = InMemoryEntityRepository([
        EntityRecord(id="e-1", label="Acme", type="organization", identifiers={"wikidata": "Q123"})
    ])
    result = EntityResolver(repository).resolve(
        EntityMention(
            label="ACME plc",
            type="organization",
            aliases=["Acme Inc."],
            identifiers={"wikidata": "Q123", "lei": "5493001KJTIIGC8Y1R12"},
        ),
        "agent-a",
    )

    assert result.status == EntityResolutionStatus.RESOLVED
    assert result.entity is not None
    assert result.entity.id == "e-1"
    assert "ACME plc" in result.entity.aliases
    assert result.entity.identifiers["lei"] == "5493001KJTIIGC8Y1R12"
    assert result.audit.match == "exact_identifier"


def test_resolves_by_canonical_key_before_aliases() -> None:
    repository = InMemoryEntityRepository([
        EntityRecord(id="e-1", label="Flat 17", type="listing", canonical_key="listing:flat-17"),
        EntityRecord(id="e-2", label="Flat 17 archive", type="listing", aliases=["Flat 17"]),
    ])
    result = EntityResolver(repository).resolve(
        EntityMention(label="Flat 17", type="listing", canonical_key="listing:flat-17"),
        "agent-a",
    )

    assert result.status == EntityResolutionStatus.RESOLVED
    assert result.entity is not None
    assert result.entity.id == "e-1"
    assert result.audit.match == "canonical_key"


def test_resolves_alias_within_type() -> None:
    repository = InMemoryEntityRepository([
        EntityRecord(id="person-1", label="Tommaso Pasini", type="person", aliases=["Tommaso"]),
        EntityRecord(id="org-1", label="Tommaso Labs", type="organization", aliases=["Tommaso"]),
    ])
    result = EntityResolver(repository).resolve(
        EntityMention(label="Tommaso", type="person"),
        "agent-a",
    )

    assert result.status == EntityResolutionStatus.RESOLVED
    assert result.entity is not None
    assert result.entity.id == "person-1"
    assert result.audit.match == "alias"


def test_resolves_constrained_fuzzy_match() -> None:
    repository = InMemoryEntityRepository([
        EntityRecord(id="person-1", label="Tommaso Pasini", type="person"),
        EntityRecord(id="person-2", label="Ada Lovelace", type="person"),
    ])
    result = EntityResolver(repository).resolve(
        EntityMention(label="Tomaso Pasini", type="person"),
        "agent-a",
    )

    assert result.status == EntityResolutionStatus.RESOLVED
    assert result.entity is not None
    assert result.entity.id == "person-1"
    assert result.audit.match == "fuzzy"
    assert result.confidence >= 0.86


def test_creates_new_entity_when_no_match_exists() -> None:
    repository = InMemoryEntityRepository()
    result = EntityResolver(repository).resolve(
        EntityMention(label="Project Atlas", type="task"),
        "agent-a",
    )

    assert result.status == EntityResolutionStatus.CREATED
    assert result.entity is not None
    assert result.entity.label == "Project Atlas"
    assert result.entity.canonical_key == "task:project-atlas"
    assert result.audit.match == "created"


def test_returns_ambiguity_when_alias_matches_multiple_entities() -> None:
    repository = InMemoryEntityRepository([
        EntityRecord(id="person-1", label="Sam Rivera", type="person", aliases=["Sam"]),
        EntityRecord(id="person-2", label="Samir Khan", type="person", aliases=["Sam"]),
    ])
    result = EntityResolver(repository).resolve(
        EntityMention(label="Sam", type="person"),
        "agent-a",
    )

    assert result.status == EntityResolutionStatus.AMBIGUOUS
    assert result.entity is None
    assert result.audit.match == "ambiguous"
    assert len(result.audit.candidates) == 2


def test_stable_user_identity_resolves_user_and_name_to_one_person() -> None:
    repository = InMemoryEntityRepository()
    resolver = EntityResolver(repository)

    by_name = resolver.resolve(
        stable_user_mention("agent-a", label="Tommaso", aliases=["User"]),
        "agent-a",
    )
    by_user = resolver.resolve(
        stable_user_mention("agent-a", label="User", aliases=["Tommaso"]),
        "agent-a",
    )

    assert by_name.status == EntityResolutionStatus.CREATED
    assert by_user.status == EntityResolutionStatus.RESOLVED
    assert by_name.entity is not None
    assert by_user.entity is not None
    assert by_user.entity.id == by_name.entity.id
    assert by_user.entity.canonical_key == "agent_user:agent-a"


def test_judge_attaches_borderline_variant_to_existing_base() -> None:
    # "Elmway" already exists; a new "Elmway X1" mention the judge deems
    # the SAME entity must NOT create a second entity — it attaches and records
    # the variant as an alias.
    repository = InMemoryEntityRepository([
        EntityRecord(id="loc-1", label="Elmway", type="location"),
    ])
    judge = _subset_judge()
    result = EntityResolver(repository, judge=judge).resolve(
        EntityMention(label="Elmway X1", type="location"),
        "agent-a",
    )

    assert result.status == EntityResolutionStatus.RESOLVED
    assert result.entity is not None
    assert result.entity.id == "loc-1"
    assert result.audit.match == "judge"
    assert "Elmway X1" in result.entity.aliases
    assert repository.created == 0
    assert judge.calls == 1


def test_without_judge_a_variant_creates_a_new_entity() -> None:
    # The judge stage is opt-in: with no judge injected, a borderline variant is
    # NOT auto-merged at runtime (the dedup migration is the net) — it creates a
    # new entity rather than relying on a tailored key.
    repository = InMemoryEntityRepository([
        EntityRecord(id="loc-1", label="Elmway", type="location"),
    ])
    result = EntityResolver(repository).resolve(
        EntityMention(label="Elmway X1", type="location"),
        "agent-a",
    )

    assert result.status == EntityResolutionStatus.CREATED
    assert repository.created == 1


def test_judge_respects_type_scope() -> None:
    # different type -> the candidate is not in scope, the judge is never asked,
    # a new entity is created.
    repository = InMemoryEntityRepository([
        EntityRecord(id="loc-1", label="Elmway", type="location"),
    ])
    judge = _subset_judge()
    result = EntityResolver(repository, judge=judge).resolve(
        EntityMention(label="Elmway X1", type="organization"),
        "agent-a",
    )

    assert result.status == EntityResolutionStatus.CREATED
    assert repository.created == 1
    assert judge.calls == 0  # no same-type candidate to adjudicate


def test_judge_keeps_content_distinct_entities_apart() -> None:
    # the judge says DIFFERENT for EU vs EU institutions -> a new entity.
    repository = InMemoryEntityRepository([
        EntityRecord(id="org-1", label="EU", type="organization"),
    ])
    judge = StubJudge(lambda _a, _b: False)
    result = EntityResolver(repository, judge=judge).resolve(
        EntityMention(label="EU institutions", type="organization"),
        "agent-a",
    )

    assert result.status == EntityResolutionStatus.CREATED
    assert judge.calls == 1  # consulted, but declined to merge


def test_judge_runs_after_alias() -> None:
    # an exact alias hit must win over the judge path (and report "alias"); the
    # judge is never consulted.
    repository = InMemoryEntityRepository([
        EntityRecord(id="loc-1", label="Elmway", type="location", aliases=["Elmway X1"]),
        EntityRecord(id="loc-2", label="Elmway Road", type="location"),
    ])
    judge = _subset_judge()
    result = EntityResolver(repository, judge=judge).resolve(
        EntityMention(label="Elmway X1", type="location"),
        "agent-a",
    )

    assert result.status == EntityResolutionStatus.RESOLVED
    assert result.entity is not None
    assert result.entity.id == "loc-1"
    assert result.audit.match == "alias"
    assert judge.calls == 0


def test_judge_attaches_variant_through_sqlite_adapter(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "dedup.db")
    resolver = EntityResolver(SQLiteEntityResolverRepository(store), judge=_subset_judge())

    base = resolver.resolve(EntityMention(label="Sunnydale", type="location"), "agent-a")
    variant = resolver.resolve(EntityMention(label="Sunnydale California", type="location"), "agent-a")

    assert base.status == EntityResolutionStatus.CREATED
    assert variant.status == EntityResolutionStatus.RESOLVED
    assert variant.entity is not None
    assert variant.entity.id == base.entity.id
    assert variant.audit.match == "judge"
    # only one entity of this type exists in the scope
    assert len(store.list_entities("agent-a", "location")) == 1


def test_sqlite_adapter_uses_current_store_without_ontology_tables(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.db")
    resolver = EntityResolver(SQLiteEntityResolverRepository(store))

    created = resolver.resolve(
        EntityMention(label="Local Widget", type="asset", identifiers={"wikidata": "Q999"}),
        "agent-a",
    )
    resolved = resolver.resolve(
        EntityMention(label="Widget", type="asset", identifiers={"wikidata": "Q999"}),
        "agent-a",
    )

    assert created.status == EntityResolutionStatus.CREATED
    assert resolved.status == EntityResolutionStatus.RESOLVED
    assert created.entity is not None
    assert resolved.entity is not None
    assert resolved.entity.id == created.entity.id
