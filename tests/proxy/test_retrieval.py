"""Tests for query-independent profile retrieval (W1b)."""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit, Relation
from tpasini.recordatio.proxy.retrieval import retrieve_profile
from tpasini.recordatio.storage.base import SQLiteStore

AGENT = "agent-x"
USER_LABEL = "Tommaso"

_logger = logging.getLogger("test")


def _now() -> datetime:
    return datetime.now(UTC)


def _add_entity(store: SQLiteStore, label: str, *, type_: str = "concept") -> Entity:
    entity = Entity(label=label, asserted_at=_now(), type=type_)
    store.add_entity(entity, AGENT)
    stored = store.get_entity_by_label(label, AGENT)
    assert stored is not None
    return stored


def _add_relation(
    store: SQLiteStore,
    subject_id: str,
    predicate: str,
    object_id: str,
    *,
    valid_until: datetime | None = None,
) -> None:
    now = _now()
    store.add_relation(
        Relation(
            id="",
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            asserted_at=now,
            valid_from=now,
            valid_until=valid_until,
            provenance="test",
        ),
        AGENT,
    )


def _add_unit(
    store: SQLiteStore,
    content: str,
    *,
    type_: KnowledgeType = KnowledgeType.SEMANTIC,
    valid_until: datetime | None = None,
) -> KnowledgeUnit:
    now = _now()
    unit = KnowledgeUnit(
        type=type_,
        content=content,
        asserted_at=now,
        valid_from=now,
        valid_until=valid_until,
        provenance="test",
    )
    store.add(unit, AGENT)
    return unit


@pytest.fixture()
def seeded_store(tmp_path: Path) -> SQLiteStore:
    """User entity + active/inactive relations + units + unconnected distractors."""
    store = SQLiteStore(tmp_path / "profile.db")

    user = _add_entity(store, USER_LABEL, type_="person")
    springfield = _add_entity(store, "Springfield", type_="place")
    globex = _add_entity(store, "Globex", type_="organization")
    old_employer = _add_entity(store, "OldCorp", type_="organization")

    # Distractor entity with its own relation, not connected to the user.
    paris = _add_entity(store, "Paris", type_="place")
    france = _add_entity(store, "France", type_="place")
    _add_relation(store, paris.id, "located_in", france.id)

    # Active user relations.
    _add_relation(store, user.id, "lives_in", springfield.id)
    _add_relation(store, user.id, "works_at", globex.id)
    # Superseded (inactive) user relation — must be excluded.
    _add_relation(
        store, user.id, "works_at", old_employer.id, valid_until=_now() - timedelta(days=1)
    )

    # Units about the user (active) and a distractor / inactive unit.
    _add_unit(store, f"{USER_LABEL} is saving for a flat with a budget of 500k.")
    _add_unit(store, f"{USER_LABEL} prefers quiet neighbourhoods.")
    _add_unit(store, "Paris is the capital of France.")  # distractor: no user mention
    _add_unit(
        store,
        f"{USER_LABEL} used to live in Ogdenville.",
        valid_until=_now() - timedelta(days=1),
    )  # inactive
    _add_unit(
        store, f"{USER_LABEL} ran an episodic errand.", type_=KnowledgeType.EPISODIC
    )  # wrong type

    return store


async def test_returns_user_neighbourhood(seeded_store: SQLiteStore) -> None:
    result = await retrieve_profile(AGENT, seeded_store, USER_LABEL, _logger)

    labels = {e.label for e in result.entities}
    assert USER_LABEL in labels
    assert "Springfield" in labels
    assert "Globex" in labels

    predicates = {(r.predicate, r.object) for r in result.rendered_relations}
    assert ("lives_in", "Springfield") in predicates
    assert ("works_at", "Globex") in predicates

    contents = {u.content for u in result.units}
    assert any("budget of 500k" in c for c in contents)
    assert any("quiet neighbourhoods" in c for c in contents)


async def test_excludes_distractors_and_inactive(seeded_store: SQLiteStore) -> None:
    result = await retrieve_profile(AGENT, seeded_store, USER_LABEL, _logger)

    labels = {e.label for e in result.entities}
    # Unconnected distractor entities are not pulled in.
    assert "Paris" not in labels
    assert "France" not in labels
    # Superseded relation endpoint is not reachable.
    assert "OldCorp" not in labels

    # No superseded relations survive.
    assert all(r.valid_until is None for r in result.raw_relations)
    objects = {r.object for r in result.rendered_relations}
    assert "OldCorp" not in objects

    contents = {u.content for u in result.units}
    assert all("Paris is the capital" not in c for c in contents)  # distractor unit
    assert all("Ogdenville" not in c for c in contents)  # inactive unit
    assert all("episodic errand" not in c for c in contents)  # non-semantic unit
    assert all(u.valid_until is None for u in result.units)
    assert all(u.type == KnowledgeType.SEMANTIC for u in result.units)


async def test_missing_user_entity_returns_empty(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "empty.db")
    # Seed an unrelated entity so the store is non-empty but has no user node.
    _add_entity(store, "Springfield", type_="place")

    result = await retrieve_profile(AGENT, store, USER_LABEL, _logger)

    assert result.entities == []
    assert result.units == []
    assert result.raw_relations == []
    assert result.rendered_relations == []
    assert result.topic_hint is None
