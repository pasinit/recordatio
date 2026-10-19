"""Tests for ``tpasini.recordatio.proxy.briefing_log``.

Schema, write, read behaviour of the sidecar briefing-events store.
Mirrors ``test_cost_log`` — same shape of fixture, same kind of
roundtrip + filter coverage.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tpasini.recordatio.proxy.briefing_log import BriefingEvent, BriefingLog


@pytest.fixture()
def log(tmp_path: Path) -> BriefingLog:
    return BriefingLog(tmp_path / "briefing.db")


def test_record_and_get_by_request_id_roundtrip(log: BriefingLog) -> None:
    log.record(BriefingEvent(
        ts=datetime.now(UTC),
        request_id="recordatio-abc",
        agent_id="alice",
        source_agent_id="alice/weather",
        query_text="what do I prefer?",
        topic_hint="preferences",
        recalled_unit_ids=["u1", "u2"],
        recalled_entity_ids=["e1"],
        recalled_relation_ids=[],
    ))

    row = log.get_by_request_id("recordatio-abc")
    assert row is not None
    assert row["request_id"] == "recordatio-abc"
    assert row["agent_id"] == "alice"
    assert row["source_agent_id"] == "alice/weather"
    assert row["query_text"] == "what do I prefer?"
    assert row["topic_hint"] == "preferences"
    assert row["recalled_unit_ids"] == ["u1", "u2"]
    assert row["recalled_entity_ids"] == ["e1"]
    assert row["recalled_relation_ids"] == []
    assert row["count_units"] == 2
    assert row["count_entities"] == 1
    assert row["count_relations"] == 0


def test_get_by_request_id_returns_none_when_missing(log: BriefingLog) -> None:
    assert log.get_by_request_id("nope") is None


def test_record_briefing_helper_sets_counts(log: BriefingLog) -> None:
    log.record_briefing(
        request_id="r1", agent_id="alice",
        query_text="q", topic_hint=None,
        recalled_unit_ids=["u1"],
        recalled_entity_ids=["e1", "e2"],
        recalled_relation_ids=["r1", "r2", "r3"],
    )
    row = log.get_by_request_id("r1")
    assert row is not None
    assert row["count_units"] == 1
    assert row["count_entities"] == 2
    assert row["count_relations"] == 3


def test_list_by_agent_filters_and_orders_newest_first(log: BriefingLog) -> None:
    # Insertion order = chronological; list_by_agent should reverse it.
    log.record_briefing(
        request_id="r1", agent_id="alice", query_text="q1", topic_hint=None,
        recalled_unit_ids=[], recalled_entity_ids=[], recalled_relation_ids=[],
    )
    log.record_briefing(
        request_id="r2", agent_id="bob", query_text="q2", topic_hint=None,
        recalled_unit_ids=[], recalled_entity_ids=[], recalled_relation_ids=[],
    )
    log.record_briefing(
        request_id="r3", agent_id="alice", query_text="q3", topic_hint=None,
        recalled_unit_ids=[], recalled_entity_ids=[], recalled_relation_ids=[],
    )

    alice_rows = log.list_by_agent(agent_id="alice")
    assert [r["request_id"] for r in alice_rows] == ["r3", "r1"]

    bob_rows = log.list_by_agent(agent_id="bob")
    assert [r["request_id"] for r in bob_rows] == ["r2"]

    all_rows = log.list_by_agent()
    assert [r["request_id"] for r in all_rows] == ["r3", "r2", "r1"]


def test_list_by_agent_can_filter_by_source_agent(log: BriefingLog) -> None:
    log.record_briefing(
        request_id="r-weather", agent_id="alice", source_agent_id="alice/weather",
        query_text="q1", topic_hint=None,
        recalled_unit_ids=[], recalled_entity_ids=[], recalled_relation_ids=[],
    )
    log.record_briefing(
        request_id="r-finance", agent_id="alice", source_agent_id="alice/finance",
        query_text="q2", topic_hint=None,
        recalled_unit_ids=[], recalled_entity_ids=[], recalled_relation_ids=[],
    )

    rows = log.list_by_agent(agent_id="alice", source_agent_id="alice/weather")

    assert [r["request_id"] for r in rows] == ["r-weather"]


def test_list_by_agent_respects_limit_and_offset(log: BriefingLog) -> None:
    for i in range(5):
        log.record_briefing(
            request_id=f"r{i}", agent_id="alice",
            query_text=f"q{i}", topic_hint=None,
            recalled_unit_ids=[], recalled_entity_ids=[], recalled_relation_ids=[],
        )

    # Newest-first: r4, r3, r2, r1, r0
    page1 = log.list_by_agent(agent_id="alice", limit=2, offset=0)
    assert [r["request_id"] for r in page1] == ["r4", "r3"]
    page2 = log.list_by_agent(agent_id="alice", limit=2, offset=2)
    assert [r["request_id"] for r in page2] == ["r2", "r1"]
    page3 = log.list_by_agent(agent_id="alice", limit=2, offset=4)
    assert [r["request_id"] for r in page3] == ["r0"]


def test_persistence_survives_restart(tmp_path: Path) -> None:
    db = tmp_path / "briefing.db"
    log = BriefingLog(db)
    log.record_briefing(
        request_id="r1", agent_id="alice",
        query_text="q", topic_hint=None,
        recalled_unit_ids=["u1"], recalled_entity_ids=[], recalled_relation_ids=[],
    )
    log.close()

    log2 = BriefingLog(db)
    row = log2.get_by_request_id("r1")
    assert row is not None
    assert row["recalled_unit_ids"] == ["u1"]


def test_get_by_request_id_returns_latest_for_collisions(log: BriefingLog) -> None:
    """If a request_id is somehow reused, return the latest row, not a
    random match. (We use ``ORDER BY id DESC`` which is a stable proxy
    for ``ts DESC`` without timestamp-tie ambiguity.)"""
    log.record_briefing(
        request_id="dup", agent_id="alice", query_text="first", topic_hint=None,
        recalled_unit_ids=[], recalled_entity_ids=[], recalled_relation_ids=[],
    )
    log.record_briefing(
        request_id="dup", agent_id="alice", query_text="second", topic_hint=None,
        recalled_unit_ids=[], recalled_entity_ids=[], recalled_relation_ids=[],
    )
    row = log.get_by_request_id("dup")
    assert row is not None
    assert row["query_text"] == "second"
