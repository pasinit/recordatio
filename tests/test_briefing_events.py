"""Accounting for current-store hydration of historical briefing IDs."""

import logging
import sqlite3

import pytest

from tpasini.recordatio.proxy.briefing_events import hydrate_briefing_row_with_conn

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def conn():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE knowledge_units (id TEXT, agent_id TEXT, type TEXT, content TEXT, asserted_at TEXT);
        CREATE TABLE entities (id TEXT, agent_id TEXT, label TEXT, wikidata_id TEXT, description TEXT);
        CREATE TABLE relations (id TEXT, agent_id TEXT, subject_id TEXT, predicate TEXT, object_id TEXT);
        INSERT INTO knowledge_units VALUES ('u1', 'alice', 'semantic', 'current content', 'now');
        INSERT INTO entities VALUES ('e1', 'alice', 'current label', NULL, NULL);
        INSERT INTO relations VALUES ('r1', 'alice', 'e1', 'knows', 'u1');
    """)
    yield db
    db.close()


@pytest.fixture
def row():
    return {
        "id": 1, "ts": "then", "request_id": "request", "agent_id": "alice",
        "recalled_unit_ids": ["u1"], "recalled_entity_ids": ["e1"], "recalled_relation_ids": ["r1"],
        "count_units": 1, "count_entities": 1, "count_relations": 1,
    }


def test_complete_hydration(conn, row):
    result = hydrate_briefing_row_with_conn(row, conn, LOGGER)
    assert result["content_source"] == "current_store"
    assert result["hydration_complete"] is True
    for kind, singular in (("units", "unit"), ("entities", "entity"), ("relations", "relation")):
        assert result[f"recalled_{singular}_ids"] == row[f"recalled_{singular}_ids"]
        assert result[f"count_{kind}"] == 1
        assert result["hydration"][kind] == {
            "hydrated_count": 1, "unresolved_ids": [], "errors": [], "complete": True,
        }
    assert result["units"][0]["content"] == "current content"
    assert result["relations"][0]["object"] == "current content"


@pytest.mark.parametrize("other_tenant", [False, True])
def test_unresolved_records_are_not_query_errors(conn, row, other_tenant):
    for table in ("knowledge_units", "entities", "relations"):
        if other_tenant:
            conn.execute(f"UPDATE {table} SET agent_id = 'bob'")
        else:
            conn.execute(f"DELETE FROM {table}")
    result = hydrate_briefing_row_with_conn(row, conn, LOGGER)
    assert result["hydration_complete"] is False
    for kind, id_ in (("units", "u1"), ("entities", "e1"), ("relations", "r1")):
        assert result[kind] == []
        assert result[f"count_{kind}"] == 1
        assert result["hydration"][kind] == {
            "hydrated_count": 0, "unresolved_ids": [id_], "errors": [], "complete": False,
        }


@pytest.mark.parametrize("kind,table,id_", [
    ("units", "knowledge_units", "u1"),
    ("entities", "entities", "e1"),
    ("relations", "relations", "r1"),
])
def test_query_errors_are_explicit(conn, row, kind, table, id_):
    conn.execute(f"DROP TABLE {table}")
    result = hydrate_briefing_row_with_conn(row, conn, LOGGER)
    assert result["hydration_complete"] is False
    assert result["hydration"][kind] == {
        "hydrated_count": 0, "unresolved_ids": [id_], "errors": ["query_failed"], "complete": False,
    }
    if kind != "relations":
        assert result["hydration"]["relations"]["errors"] == ["endpoint_query_failed"]
        assert result["hydration"]["relations"]["complete"] is False


def test_missing_relation_endpoint_is_unresolved(conn, row):
    conn.execute("DELETE FROM knowledge_units")
    result = hydrate_briefing_row_with_conn(row, conn, LOGGER)
    assert result["hydration"]["relations"] == {
        "hydrated_count": 0, "unresolved_ids": ["r1"], "errors": [], "complete": False,
    }


def test_missing_tenant_never_hydrates(conn, row):
    row["agent_id"] = None
    result = hydrate_briefing_row_with_conn(row, conn, LOGGER)
    for kind in ("units", "entities", "relations"):
        assert result[kind] == []
        assert result["hydration"][kind]["errors"] == ["missing_agent_id"]
        assert result["hydration"][kind]["complete"] is False


def test_empty_briefing_is_complete_without_queries(conn, row):
    conn.close()
    for kind, singular in (("units", "unit"), ("entities", "entity"), ("relations", "relation")):
        row[f"recalled_{singular}_ids"] = []
        row[f"count_{kind}"] = 0
    assert hydrate_briefing_row_with_conn(row, conn, LOGGER)["hydration_complete"] is True
