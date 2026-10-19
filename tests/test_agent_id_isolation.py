"""Tests for agent_id namespacing: verifies that data written under one agent_id
is not visible to queries under a different agent_id."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit, Relation, RetrievalContext
from tpasini.recordatio.storage.base import SQLiteStore


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "test.db")


def _unit(content: str) -> KnowledgeUnit:
    now = datetime.now(UTC)
    return KnowledgeUnit(
        id="",
        type=KnowledgeType.SEMANTIC,
        content=content,
        asserted_at=now,
        valid_from=now,
        provenance="test",
    )


def _entity(label: str) -> Entity:
    return Entity(label=label, asserted_at=datetime.now(UTC))


def test_units_are_isolated(store: SQLiteStore) -> None:
    unit = _unit("Alice loves Python")
    store.add(unit, "alice")

    results_alice = store.search(RetrievalContext(query="Alice", top_k=10), "alice")
    results_bob = store.search(RetrievalContext(query="Alice", top_k=10), "bob")

    assert len(results_alice) == 1
    assert len(results_bob) == 0


def test_units_can_filter_by_source_agent_within_shared_namespace(store: SQLiteStore) -> None:
    weather = _unit("Alice prefers rain alerts")
    weather.id = "weather"
    finance = _unit("Alice tracks AAPL earnings")
    finance.id = "finance"
    store.add(weather, "alice", source_agent_id="alice/weather")
    store.add(finance, "alice", source_agent_id="alice/finance")

    all_results = store.search(RetrievalContext(query="Alice", top_k=10), "alice")
    weather_results = store.search(
        RetrievalContext(query="Alice", top_k=10),
        "alice",
        source_agent_id="alice/weather",
    )

    assert {u.content for u in all_results} == {"Alice prefers rain alerts", "Alice tracks AAPL earnings"}
    assert [u.content for u in weather_results] == ["Alice prefers rain alerts"]


def test_entities_are_isolated(store: SQLiteStore) -> None:
    store.add_entity(_entity("Paris"), "alice")

    assert store.get_entity_by_label("Paris", "alice") is not None
    assert store.get_entity_by_label("Paris", "bob") is None


def test_same_entity_label_in_different_agents(store: SQLiteStore) -> None:
    e_alice = _entity("User")
    e_bob = _entity("User")
    store.add_entity(e_alice, "alice")
    store.add_entity(e_bob, "bob")

    alice_user = store.get_entity_by_label("User", "alice")
    bob_user = store.get_entity_by_label("User", "bob")

    assert alice_user is not None
    assert bob_user is not None
    # They should be different rows (different UUIDs).
    assert alice_user.id != bob_user.id


def test_relations_are_isolated(store: SQLiteStore) -> None:
    entity = _entity("Paris")
    store.add_entity(entity, "alice")
    alice_ent = store.get_entity_by_label("Paris", "alice")
    assert alice_ent is not None

    unit = _unit("Paris is the capital of France")
    store.add(unit, "alice")

    now = datetime.now(UTC)
    store.add_relation(
        Relation(
            id="",
            subject_id=unit.id,
            predicate="about",
            object_id=alice_ent.id,
            asserted_at=now,
            valid_from=now,
            valid_until=None,
            provenance="test",
        ),
        "alice",
    )

    assert len(store.get_relations(unit.id, "alice")) == 1
    assert len(store.get_relations(unit.id, "bob")) == 0


def test_supersede_relations_scoped(store: SQLiteStore) -> None:
    entity = _entity("Bob")
    store.add_entity(entity, "alice")
    alice_ent = store.get_entity_by_label("Bob", "alice")
    assert alice_ent is not None

    loc_entity = _entity("Springfield")
    store.add_entity(loc_entity, "alice")
    loc_ent = store.get_entity_by_label("Springfield", "alice")
    assert loc_ent is not None

    now = datetime.now(UTC)
    store.add_relation(
        Relation(
            id="",
            subject_id=alice_ent.id,
            predicate="lives_in",
            object_id=loc_ent.id,
            asserted_at=now,
            valid_from=now,
            valid_until=None,
            provenance="test",
        ),
        "alice",
    )

    # supersede under alice should not affect bob's relations
    invalidated_bob = store.supersede_relations(alice_ent.id, "lives_in", now, "bob")
    assert invalidated_bob == 0

    invalidated_alice = store.supersede_relations(alice_ent.id, "lives_in", now, "alice")
    assert invalidated_alice == 1


def test_default_agent_migration(tmp_path: Path) -> None:
    """Existing data written without agent_id should default to 'default' namespace."""
    db_path = tmp_path / "legacy.db"
    # Bootstrap the DB through the store (creates schema + runs migrations).
    store = SQLiteStore(db_path)

    columns = {row["name"] for row in store._conn.execute("PRAGMA table_info(knowledge_units)").fetchall()}
    assert {"agent_id", "source_agent_id"} <= columns

    # Insert directly with a known id so we can verify migration default.
    store._conn.execute(
        """
        INSERT OR REPLACE INTO knowledge_units
            (id, type, content, asserted_at, valid_from, valid_until, provenance, agent_id)
        VALUES ('legacy-id', 'semantic', 'Legacy fact', '2024-01-01T00:00:00', '2024-01-01T00:00:00',
                NULL, 'test', 'default')
        """
    )
    store._conn.commit()

    results = store.search(RetrievalContext(query="Legacy", top_k=10), "default")
    assert any(u.content == "Legacy fact" for u in results)

    results_other = store.search(RetrievalContext(query="Legacy", top_k=10), "other-agent")
    assert not any(u.content == "Legacy fact" for u in results_other)
