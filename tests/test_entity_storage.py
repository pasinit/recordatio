import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import (
    AttributeType,
    EntityAlias,
    EntityAttribute,
    EntityIdentifier,
    EntityMention,
    EntityType,
)
from tpasini.recordatio.storage.base import SQLiteStore


def _now() -> datetime:
    return datetime.now(UTC)


def _entity(label: str, **kwargs: str) -> Entity:
    return Entity(label=label, asserted_at=_now(), **kwargs)


def test_legacy_entities_migrate_to_generic_entity_shape(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE entities (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            wikidata_id TEXT,
            description TEXT,
            wikipedia_url TEXT,
            asserted_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO entities
            (id, label, wikidata_id, description, wikipedia_url, asserted_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("e-legacy", "Alice", "Q1", "legacy person", None, "2024-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(db_path)

    columns = {row["name"] for row in store._conn.execute("PRAGMA table_info(entities)").fetchall()}
    assert {"agent_id", "type", "subtype", "canonical_label", "canonical_key"} <= columns

    entity = store.get_entity_by_label("alice", "default")
    assert entity is not None
    assert entity.id == "e-legacy"
    assert entity.type == "concept"
    assert entity.canonical_label == "Alice"
    assert entity.wikidata_id == "Q1"


def test_generic_entity_schema_setup_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "idem.db"
    SQLiteStore(db_path)._conn.close()
    store = SQLiteStore(db_path)

    expected_tables = {
        "entity_aliases",
        "entity_identifiers",
        "entity_attributes",
        "entity_mentions",
        "entity_types",
        "attribute_types",
    }
    tables = {
        row["name"] for row in store._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    assert expected_tables <= tables

    seeded_names = {entity_type.name for entity_type in store.list_entity_types("alice")}
    assert seeded_names == {
        "person",
        "organization",
        "place",
        "listing",
        "asset",
        "task",
        "preference_profile",
        "event",
        "document",
        "concept",
    }


def test_legacy_entity_metadata_tables_gain_agent_scope(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_metadata.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE entities (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            wikidata_id TEXT,
            description TEXT,
            wikipedia_url TEXT,
            asserted_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE entity_aliases (
            id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            alias TEXT NOT NULL,
            alias_key TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO entities
            (id, label, wikidata_id, description, wikipedia_url, asserted_at)
        VALUES
            (?, ?, ?, ?, ?, ?),
            (?, ?, ?, ?, ?, ?)
        """,
        (
            "e-default",
            "Alice",
            None,
            None,
            None,
            "2024-01-01T00:00:00+00:00",
            "e-bob",
            "Bob",
            None,
            None,
            None,
            "2024-01-01T00:00:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO entity_aliases
            (id, entity_id, alias, alias_key, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("a-default", "e-default", "Al", "al", "legacy", "2024-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(db_path)

    columns = {row["name"] for row in store._conn.execute("PRAGMA table_info(entity_aliases)").fetchall()}
    assert "agent_id" in columns

    created_at = _now()
    store.upsert_entity_alias(
        EntityAlias(entity_id="e-bob", alias="Al", source="test", created_at=created_at),
        "bob",
    )
    store.upsert_entity_alias(
        EntityAlias(entity_id="e-default", alias="Al", source="updated", created_at=created_at),
        "default",
    )

    rows = store._conn.execute(
        """
        SELECT agent_id, entity_id, source
        FROM entity_aliases
        WHERE alias_key = ?
        ORDER BY agent_id
        """,
        ("al",),
    ).fetchall()
    assert [(row["agent_id"], row["entity_id"], row["source"]) for row in rows] == [
        ("bob", "e-bob", "test"),
        ("default", "e-default", "updated"),
    ]


def test_generic_entity_writes_are_agent_scoped(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "scoped.db")
    alice_entity = _entity("Alice", type="person", canonical_key="person:alice")
    alice_duplicate = _entity("Alicia", type="person", canonical_key="person:alice")
    bob_entity = _entity("Alice", type="person", canonical_key="person:alice")
    store.add_entity(alice_entity, "alice")
    store.add_entity(alice_duplicate, "alice")
    store.add_entity(bob_entity, "bob")

    entity_rows = store._conn.execute(
        "SELECT agent_id, canonical_key FROM entities WHERE canonical_key = ? ORDER BY agent_id",
        ("person:alice",),
    ).fetchall()
    assert [(row["agent_id"], row["canonical_key"]) for row in entity_rows] == [
        ("alice", "person:alice"),
        ("bob", "person:alice"),
    ]

    created_at = _now()
    for agent_id, entity_id in (("alice", alice_entity.id), ("bob", bob_entity.id)):
        store.upsert_entity_alias(
            EntityAlias(entity_id=entity_id, alias="Al", source="test", created_at=created_at),
            agent_id,
        )
        store.upsert_entity_identifier(
            EntityIdentifier(
                entity_id=entity_id,
                namespace="wikidata",
                value="Q42",
                source="test",
                created_at=created_at,
            ),
            agent_id,
        )
        store.upsert_entity_attribute(
            EntityAttribute(
                entity_id=entity_id,
                attribute_type="role",
                value="engineer",
                asserted_at=created_at,
                valid_from=created_at,
                provenance="test",
            ),
            agent_id,
        )
        store.upsert_entity_mention(
            EntityMention(
                entity_id=entity_id,
                source_kind="knowledge_unit",
                source_id="u-1",
                snippet="Alice works here",
                asserted_at=created_at,
            ),
            agent_id,
        )
        store.upsert_entity_type(
            EntityType(name="custom", description="custom type", source="test", created_at=created_at),
            agent_id,
        )
        store.upsert_attribute_type(
            AttributeType(
                name="role",
                description="current role",
                value_type="text",
                source="test",
                created_at=created_at,
            ),
            agent_id,
        )

    assert len(store.list_entity_aliases(alice_entity.id, "alice")) == 1
    assert len(store.list_entity_aliases(alice_entity.id, "bob")) == 0
    assert len(store.list_entity_aliases(bob_entity.id, "bob")) == 1
    assert len(store.list_entity_identifiers(alice_entity.id, "alice")) == 1
    assert len(store.list_entity_identifiers(bob_entity.id, "bob")) == 1
    assert len(store.list_entity_attributes(alice_entity.id, "alice")) == 1
    assert len(store.list_entity_mentions(bob_entity.id, "bob")) == 1
    assert any(entity_type.name == "custom" for entity_type in store.list_entity_types("alice"))
    assert any(attribute_type.name == "role" for attribute_type in store.list_attribute_types("bob"))


def test_entity_read_helpers_are_scoped(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "entities.db")
    alice = _entity("Alice", type="person", canonical_key="person:alice", wikidata_id="Q1")
    bob = _entity("Alice", type="organization", canonical_key="organization:alice", wikidata_id="Q2")
    store.add_entity(alice, "agent-a")
    store.add_entity(bob, "agent-b")

    stored_alice = store.get_entity_by_id(alice.id, "agent-a")
    assert stored_alice is not None
    assert stored_alice.id == alice.id
    assert stored_alice.canonical_label == "Alice"
    assert store.get_entity_by_id(alice.id, "agent-b") is None
    assert store.get_entity_by_canonical_key("person:alice", "agent-a", "person") == stored_alice
    assert store.get_entity_by_canonical_key("person:alice", "agent-b", "person") is None
    assert store.list_entities("agent-a", "person") == [stored_alice]
    assert store.list_entities("agent-b", "person") == []


def test_entity_alias_upsert_read_and_delete_are_idempotent_and_scoped(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "aliases.db")
    created_at = _now()
    first = EntityAlias(id="a-1", entity_id="entity-a", alias="Alice A.", source="first", created_at=created_at)
    second = EntityAlias(id="a-2", entity_id="entity-a", alias=" alice   a. ", source="second", created_at=created_at)
    other_agent = EntityAlias(id="a-3", entity_id="entity-b", alias="Alice A.", source="other", created_at=created_at)

    store.upsert_entity_alias(first, "agent-a")
    store.upsert_entity_alias(second, "agent-a")
    store.upsert_entity_alias(other_agent, "agent-b")

    agent_aliases = store.find_entity_aliases("ALICE A.", "agent-a")
    assert len(agent_aliases) == 1
    assert agent_aliases[0].entity_id == "entity-a"
    assert agent_aliases[0].source == "second"
    assert len(store.find_entity_aliases("Alice A.", "agent-b")) == 1
    assert store.get_entity_alias(agent_aliases[0].id, "agent-a") == agent_aliases[0]

    store.delete_entity_alias(agent_aliases[0].id, "agent-b")
    assert len(store.find_entity_aliases("Alice A.", "agent-a")) == 1
    store.delete_entity_alias(agent_aliases[0].id, "agent-a")
    assert store.find_entity_aliases("Alice A.", "agent-a") == []


def test_entity_identifier_upsert_read_and_delete_are_idempotent_and_scoped(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "identifiers.db")
    created_at = _now()
    first = EntityIdentifier(
        id="i-1",
        entity_id="entity-a",
        namespace="wikidata",
        value="Q42",
        source="first",
        created_at=created_at,
    )
    second = EntityIdentifier(
        id="i-2",
        entity_id="entity-a",
        namespace="wikidata",
        value="Q42",
        source="second",
        created_at=created_at,
    )
    other_agent = EntityIdentifier(
        id="i-3",
        entity_id="entity-b",
        namespace="wikidata",
        value="Q42",
        source="other",
        created_at=created_at,
    )

    store.upsert_entity_identifier(first, "agent-a")
    store.upsert_entity_identifier(second, "agent-a")
    store.upsert_entity_identifier(other_agent, "agent-b")

    identifiers = store.find_entity_identifiers("wikidata", "Q42", "agent-a")
    assert len(identifiers) == 1
    assert identifiers[0].entity_id == "entity-a"
    assert identifiers[0].source == "second"
    assert len(store.find_entity_identifiers("wikidata", "Q42", "agent-b")) == 1
    assert store.get_entity_identifier(identifiers[0].id, "agent-a") == identifiers[0]

    store.delete_entity_identifier(identifiers[0].id, "agent-b")
    assert len(store.find_entity_identifiers("wikidata", "Q42", "agent-a")) == 1
    store.delete_entity_identifier(identifiers[0].id, "agent-a")
    assert store.find_entity_identifiers("wikidata", "Q42", "agent-a") == []


def test_entity_attribute_upsert_read_and_delete_are_idempotent_and_scoped(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "attributes.db")
    created_at = _now()
    first = EntityAttribute(
        id="attr-1",
        entity_id="entity-a",
        attribute_type="role",
        value="engineer",
        asserted_at=created_at,
        valid_from=created_at,
        provenance="first",
    )
    second = EntityAttribute(
        id="attr-2",
        entity_id="entity-a",
        attribute_type="role",
        value="engineer",
        asserted_at=created_at,
        valid_from=created_at,
        provenance="second",
    )
    inactive = EntityAttribute(
        id="attr-3",
        entity_id="entity-a",
        attribute_type="role",
        value="manager",
        asserted_at=created_at,
        valid_from=created_at,
        valid_until=created_at,
        provenance="old",
    )
    other_agent = EntityAttribute(
        id="attr-4",
        entity_id="entity-b",
        attribute_type="role",
        value="engineer",
        asserted_at=created_at,
        valid_from=created_at,
        provenance="other",
    )

    store.upsert_entity_attribute(first, "agent-a")
    store.upsert_entity_attribute(second, "agent-a")
    store.upsert_entity_attribute(inactive, "agent-a")
    store.upsert_entity_attribute(other_agent, "agent-b")

    active = store.find_entity_attributes("role", "agent-a", "engineer")
    assert len(active) == 1
    assert active[0].provenance == "second"
    assert len(store.find_entity_attributes("role", "agent-b", "engineer")) == 1
    assert [attribute.value for attribute in store.list_entity_attributes("entity-a", "agent-a", active_only=True)] == [
        "engineer"
    ]
    assert {attribute.value for attribute in store.list_entity_attributes("entity-a", "agent-a")} == {
        "engineer",
        "manager",
    }
    assert store.get_entity_attribute(active[0].id, "agent-a") == active[0]

    store.delete_entity_attribute(active[0].id, "agent-b")
    assert len(store.find_entity_attributes("role", "agent-a", "engineer")) == 1
    store.delete_entity_attribute(active[0].id, "agent-a")
    assert store.find_entity_attributes("role", "agent-a", "engineer") == []


def test_entity_mention_upsert_read_and_delete_are_idempotent_and_scoped(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "mentions.db")
    created_at = _now()
    first = EntityMention(
        id="m-1",
        entity_id="entity-a",
        source_kind="knowledge_unit",
        source_id="unit-1",
        snippet="first",
        asserted_at=created_at,
    )
    second = EntityMention(
        id="m-2",
        entity_id="entity-a",
        source_kind="knowledge_unit",
        source_id="unit-1",
        snippet="second",
        asserted_at=created_at,
    )
    other_agent = EntityMention(
        id="m-3",
        entity_id="entity-b",
        source_kind="knowledge_unit",
        source_id="unit-1",
        snippet="other",
        asserted_at=created_at,
    )

    store.upsert_entity_mention(first, "agent-a")
    store.upsert_entity_mention(second, "agent-a")
    store.upsert_entity_mention(other_agent, "agent-b")

    mentions = store.find_entity_mentions_by_source("knowledge_unit", "unit-1", "agent-a")
    assert len(mentions) == 1
    assert mentions[0].entity_id == "entity-a"
    assert mentions[0].snippet == "second"
    assert len(store.find_entity_mentions_by_source("knowledge_unit", "unit-1", "agent-b")) == 1
    assert store.get_entity_mention(mentions[0].id, "agent-a") == mentions[0]

    store.delete_entity_mention(mentions[0].id, "agent-b")
    assert len(store.find_entity_mentions_by_source("knowledge_unit", "unit-1", "agent-a")) == 1
    store.delete_entity_mention(mentions[0].id, "agent-a")
    assert store.find_entity_mentions_by_source("knowledge_unit", "unit-1", "agent-a") == []


def test_entity_type_upsert_read_and_delete_are_idempotent_and_scoped(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "entity_types.db")
    created_at = _now()

    store.upsert_entity_type(
        EntityType(id="et-1", name="Project", description="first", source="first", created_at=created_at),
        "agent-a",
    )
    store.upsert_entity_type(
        EntityType(id="et-2", name=" project ", description="second", source="second", created_at=created_at),
        "agent-a",
    )
    store.upsert_entity_type(
        EntityType(id="et-3", name="Project", description="other", source="other", created_at=created_at),
        "agent-b",
    )

    entity_type = store.get_entity_type_by_name("PROJECT", "agent-a")
    assert entity_type is not None
    assert entity_type.description == "second"
    assert store.get_entity_type(entity_type.id, "agent-a") == entity_type
    assert store.get_entity_type_by_name("Project", "agent-b").description == "other"
    assert sum(1 for item in store.list_entity_types("agent-a") if item.name.strip().lower() == "project") == 1

    store.delete_entity_type(entity_type.id, "agent-b")
    assert store.get_entity_type_by_name("Project", "agent-a") is not None
    store.delete_entity_type(entity_type.id, "agent-a")
    assert store.get_entity_type_by_name("Project", "agent-a") is None


def test_attribute_type_upsert_read_and_delete_are_idempotent_and_scoped(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "attribute_types.db")
    created_at = _now()

    store.upsert_attribute_type(
        AttributeType(
            id="at-1",
            name="Favorite Color",
            description="first",
            value_type="text",
            source="first",
            created_at=created_at,
        ),
        "agent-a",
    )
    store.upsert_attribute_type(
        AttributeType(
            id="at-2",
            name=" favorite   color ",
            description="second",
            value_type="text",
            source="second",
            created_at=created_at,
        ),
        "agent-a",
    )
    store.upsert_attribute_type(
        AttributeType(
            id="at-3",
            name="Favorite Color",
            description="other",
            value_type="text",
            source="other",
            created_at=created_at,
        ),
        "agent-b",
    )

    attribute_type = store.get_attribute_type_by_name("FAVORITE COLOR", "agent-a")
    assert attribute_type is not None
    assert attribute_type.description == "second"
    assert store.get_attribute_type(attribute_type.id, "agent-a") == attribute_type
    assert store.get_attribute_type_by_name("Favorite Color", "agent-b").description == "other"
    attribute_type_names = [" ".join(item.name.casefold().split()) for item in store.list_attribute_types("agent-a")]
    assert attribute_type_names.count("favorite color") == 1

    store.delete_attribute_type(attribute_type.id, "agent-b")
    assert store.get_attribute_type_by_name("Favorite Color", "agent-a") is not None
    store.delete_attribute_type(attribute_type.id, "agent-a")
    assert store.get_attribute_type_by_name("Favorite Color", "agent-a") is None
