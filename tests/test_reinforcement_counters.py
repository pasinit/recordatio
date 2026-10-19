"""Tests for reinforcement counters (extraction_count / retrieval_count).

Covers schema defaults on fresh DBs, idempotent migration of legacy column-less
DBs, loader population from rows, and the batched, agent-scoped increment API.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.models import KnowledgeType, KnowledgeUnit, Relation, RetrievalContext
from tpasini.recordatio.storage.base import SQLiteStore

_AGENT = "default"
_OTHER = "other-agent"

_REINFORCEMENT_TABLES = ("knowledge_units", "entities", "relations")


def _now() -> datetime:
    return datetime.now(UTC)


def _columns(store: SQLiteStore, table: str) -> set[str]:
    return {row["name"] for row in store._conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _unit(content: str = "the sky is blue") -> KnowledgeUnit:
    now = _now()
    return KnowledgeUnit(
        type=KnowledgeType.SEMANTIC,
        content=content,
        asserted_at=now,
        valid_from=now,
        provenance="test",
    )


# --- fresh-DB schema defaults ------------------------------------------------


def test_fresh_db_has_counter_columns_with_defaults(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "fresh.db")
    for table in _REINFORCEMENT_TABLES:
        cols = _columns(store, table)
        assert {"extraction_count", "retrieval_count"} <= cols, table

    unit = _unit()
    store.add(unit, _AGENT)
    loaded = store.search(RetrievalContext(query="sky"), _AGENT)
    assert len(loaded) == 1
    assert loaded[0].extraction_count == 1
    assert loaded[0].retrieval_count == 0


# --- migration of legacy column-less DBs -------------------------------------


def _make_legacy_db(db_path: Path) -> None:
    """Create knowledge_units/entities/relations WITHOUT the counter columns."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE knowledge_units (
            id TEXT PRIMARY KEY, type TEXT NOT NULL, content TEXT NOT NULL,
            asserted_at TEXT NOT NULL, valid_from TEXT NOT NULL, valid_until TEXT,
            provenance TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE entities (
            id TEXT PRIMARY KEY, label TEXT NOT NULL, wikidata_id TEXT,
            description TEXT, wikipedia_url TEXT, asserted_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE relations (
            id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, predicate TEXT NOT NULL,
            object_id TEXT NOT NULL, asserted_at TEXT NOT NULL, valid_from TEXT NOT NULL,
            valid_until TEXT, provenance TEXT NOT NULL
        )
        """
    )
    ts = "2024-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO knowledge_units (id, type, content, asserted_at, valid_from, valid_until, provenance) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("u-legacy", "semantic", "legacy fact", ts, ts, None, "legacy"),
    )
    conn.execute(
        "INSERT INTO entities (id, label, wikidata_id, description, wikipedia_url, asserted_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("e-legacy", "Alice", None, None, None, ts),
    )
    conn.execute(
        "INSERT INTO relations "
        "(id, subject_id, predicate, object_id, asserted_at, valid_from, valid_until, provenance) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("r-legacy", "e-legacy", "lives_in", "e-other", ts, ts, None, "legacy"),
    )
    conn.commit()
    conn.close()


def test_migration_adds_counter_columns_with_defaults_to_existing_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    _make_legacy_db(db_path)

    store = SQLiteStore(db_path)

    for table in _REINFORCEMENT_TABLES:
        assert {"extraction_count", "retrieval_count"} <= _columns(store, table), table

    # Pre-existing rows get the defaults.
    unit_row = store._conn.execute(
        "SELECT extraction_count, retrieval_count FROM knowledge_units WHERE id = ?", ("u-legacy",)
    ).fetchone()
    assert unit_row["extraction_count"] == 1
    assert unit_row["retrieval_count"] == 0

    entity = store.get_entity_by_id("e-legacy", _AGENT)
    assert entity is not None
    assert entity.extraction_count == 1
    assert entity.retrieval_count == 0

    rels = store.get_relations("e-legacy", _AGENT)
    assert len(rels) == 1
    assert rels[0].extraction_count == 1
    assert rels[0].retrieval_count == 0


def test_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    _make_legacy_db(db_path)

    # Open repeatedly; must not raise and columns stay single.
    SQLiteStore(db_path)
    store = SQLiteStore(db_path)
    store._ensure_reinforcement_count_columns()
    store._ensure_reinforcement_count_columns()

    for table in _REINFORCEMENT_TABLES:
        cols = [row["name"] for row in store._conn.execute(f"PRAGMA table_info({table})").fetchall()]
        assert cols.count("extraction_count") == 1, table
        assert cols.count("retrieval_count") == 1, table


# --- loaders populate fields from rows ---------------------------------------


def test_loaders_read_nondefault_counts_from_rows(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "load.db")
    unit = _unit("loadable unit")
    store.add(unit, _AGENT)
    entity = Entity(label="Bob", asserted_at=_now())
    store.add_entity(entity, _AGENT)
    relation = Relation(
        id="r1", subject_id=entity.id, predicate="lives_in", object_id="o1",
        asserted_at=_now(), valid_from=_now(), valid_until=None, provenance="test",
    )
    store.add_relation(relation, _AGENT)

    # Force non-default counts directly in the DB.
    store._conn.execute("UPDATE knowledge_units SET extraction_count = 7, retrieval_count = 3 WHERE id = ?", (unit.id,))
    store._conn.execute("UPDATE entities SET extraction_count = 5, retrieval_count = 2 WHERE id = ?", (entity.id,))
    store._conn.execute("UPDATE relations SET extraction_count = 9, retrieval_count = 4 WHERE id = ?", (relation.id,))
    store._conn.commit()

    loaded_unit = store.search(RetrievalContext(query="loadable"), _AGENT)[0]
    assert (loaded_unit.extraction_count, loaded_unit.retrieval_count) == (7, 3)

    loaded_entity = store.get_entity_by_id(entity.id, _AGENT)
    assert loaded_entity is not None
    assert (loaded_entity.extraction_count, loaded_entity.retrieval_count) == (5, 2)

    loaded_rel = store.get_relations(entity.id, _AGENT)[0]
    assert (loaded_rel.extraction_count, loaded_rel.retrieval_count) == (9, 4)


# --- increment API -----------------------------------------------------------


def test_increment_methods_bump_scoped_rows(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "inc.db")
    mine = _unit("mine")
    theirs = _unit("theirs")
    store.add(mine, _AGENT)
    store.add(theirs, _OTHER)

    store.increment_extraction_count("unit", [mine.id, theirs.id], _AGENT)
    store.increment_retrieval_count("unit", [mine.id], _AGENT, amount=5)

    mine_row = store._conn.execute(
        "SELECT extraction_count, retrieval_count FROM knowledge_units WHERE id = ?", (mine.id,)
    ).fetchone()
    theirs_row = store._conn.execute(
        "SELECT extraction_count, retrieval_count FROM knowledge_units WHERE id = ?", (theirs.id,)
    ).fetchone()

    # mine bumped; theirs untouched because agent_id scope excludes it.
    assert mine_row["extraction_count"] == 2
    assert mine_row["retrieval_count"] == 5
    assert theirs_row["extraction_count"] == 1
    assert theirs_row["retrieval_count"] == 0


def test_increment_supports_entities_and_relations(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "inc2.db")
    entity = Entity(label="Carol", asserted_at=_now())
    store.add_entity(entity, _AGENT)
    relation = Relation(
        id="r2", subject_id=entity.id, predicate="works_at", object_id="o2",
        asserted_at=_now(), valid_from=_now(), valid_until=None, provenance="test",
    )
    store.add_relation(relation, _AGENT)

    store.increment_extraction_count("entity", [entity.id], _AGENT, amount=3)
    store.increment_retrieval_count("relation", [relation.id], _AGENT)

    loaded_entity = store.get_entity_by_id(entity.id, _AGENT)
    assert loaded_entity is not None
    assert loaded_entity.extraction_count == 4  # 1 default + 3
    loaded_rel = store.get_relations(entity.id, _AGENT)[0]
    assert loaded_rel.retrieval_count == 1


def test_increment_is_batched_over_999_id_limit(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "batch.db")
    ids: list[str] = []
    for i in range(1500):  # exceeds the 999 SQLite variable limit -> must chunk
        u = _unit(f"fact {i}")
        store.add(u, _AGENT)
        ids.append(u.id)

    # A single un-chunked IN (...) of 1500 ids would raise; chunking must avoid it.
    store.increment_extraction_count("unit", ids, _AGENT, amount=2)

    counts = store._conn.execute(
        "SELECT DISTINCT extraction_count FROM knowledge_units WHERE agent_id = ?", (_AGENT,)
    ).fetchall()
    assert [row["extraction_count"] for row in counts] == [3]


def test_increment_noops_on_empty_or_zero(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "noop.db")
    unit = _unit()
    store.add(unit, _AGENT)
    store.increment_extraction_count("unit", [], _AGENT)
    store.increment_retrieval_count("unit", [unit.id], _AGENT, amount=0)
    row = store._conn.execute(
        "SELECT extraction_count, retrieval_count FROM knowledge_units WHERE id = ?", (unit.id,)
    ).fetchone()
    assert row["extraction_count"] == 1
    assert row["retrieval_count"] == 0


def test_increment_rejects_unknown_table_kind(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "bad.db")
    try:
        store.increment_extraction_count("nonsense", ["x"], _AGENT)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown table_kind")
