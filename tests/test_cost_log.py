"""Tests for ``tpasini.recordatio.proxy.cost_log``.

Schema, write, read, and aggregation behaviour of the sidecar cost-events
SQLite store.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tpasini.recordatio.extraction.extractor import ExtractionSide
from tpasini.recordatio.proxy.cost_log import CostEvent, CostLog


def _mk_event(
    *,
    ts: datetime,
    kind: str = "chat",
    model: str = "claude-opus-4-6",
    provider: str | None = "anthropic",
    input_tokens: int = 1000,
    output_tokens: int = 100,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    cost_usd: float | None = 0.01,
    agent_id: str | None = "alice",
    source_agent_id: str | None = None,
    user_id: str | None = None,
    request_id: str | None = "req-1",
    ok: bool = True,
) -> CostEvent:
    return CostEvent(
        ts=ts, kind=kind, model=model, provider=provider,
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens, cache_read_tokens=cache_read_tokens,
        cost_usd=cost_usd, agent_id=agent_id, source_agent_id=source_agent_id, user_id=user_id,
        request_id=request_id, ok=ok,
    )


@pytest.fixture()
def log(tmp_path: Path) -> CostLog:
    return CostLog(tmp_path / "cost.db")


# ---------------------------------------------------------------------------
# Roundtrip + persistence
# ---------------------------------------------------------------------------


def test_record_and_query_roundtrip(log: CostLog) -> None:
    now = datetime.now(UTC)
    log.record(_mk_event(ts=now, input_tokens=1000, output_tokens=100, cost_usd=0.01))
    log.record(_mk_event(
        ts=now + timedelta(seconds=1),
        kind="extraction", model="claude-haiku-4-5", input_tokens=500,
        output_tokens=50, cost_usd=0.0005, request_id="req-1",
    ))

    events = log.query_events()
    assert len(events) == 2
    # Most recent first.
    assert events[0]["kind"] == "extraction"
    assert events[1]["kind"] == "chat"

    totals = log.query_totals()
    assert totals["event_count"] == 2
    assert totals["input_tokens"] == 1500
    assert totals["output_tokens"] == 150
    assert totals["cost_usd"] == pytest.approx(0.0105, abs=1e-9)


def test_persistence_survives_restart(tmp_path: Path) -> None:
    db = tmp_path / "cost.db"
    log = CostLog(db)
    now = datetime.now(UTC)
    log.record(_mk_event(ts=now))
    log.record(_mk_event(ts=now + timedelta(seconds=1), kind="extraction"))
    log.close()

    # Reopen on the same path, count rows.
    log2 = CostLog(db)
    assert len(log2.query_events()) == 2
    assert log2.query_totals()["event_count"] == 2


# ---------------------------------------------------------------------------
# Bucketing for query_series
# ---------------------------------------------------------------------------


def test_query_series_hour_bucketing(log: CostLog) -> None:
    base = datetime(2026, 5, 9, 14, 0, 0, tzinfo=UTC)
    log.record(_mk_event(ts=base + timedelta(minutes=5), cost_usd=0.10))
    log.record(_mk_event(ts=base + timedelta(minutes=30), cost_usd=0.20))
    log.record(_mk_event(ts=base + timedelta(hours=1, minutes=5), cost_usd=0.40))

    series = log.query_series(granularity="hour")
    assert len(series) == 2
    assert series[0]["bucket"] == "2026-05-09T14:00:00Z"
    assert series[0]["cost_usd"] == pytest.approx(0.30, abs=1e-9)
    assert series[1]["bucket"] == "2026-05-09T15:00:00Z"
    assert series[1]["cost_usd"] == pytest.approx(0.40, abs=1e-9)


def test_query_series_day_bucketing(log: CostLog) -> None:
    log.record(_mk_event(ts=datetime(2026, 5, 9, 1, 0, tzinfo=UTC), cost_usd=0.10))
    log.record(_mk_event(ts=datetime(2026, 5, 9, 23, 0, tzinfo=UTC), cost_usd=0.20))
    log.record(_mk_event(ts=datetime(2026, 5, 10, 5, 0, tzinfo=UTC), cost_usd=0.40))

    series = log.query_series(granularity="day")
    assert len(series) == 2
    assert series[0]["bucket"] == "2026-05-09T00:00:00Z"
    assert series[0]["cost_usd"] == pytest.approx(0.30, abs=1e-9)
    assert series[1]["bucket"] == "2026-05-10T00:00:00Z"
    assert series[1]["cost_usd"] == pytest.approx(0.40, abs=1e-9)


def test_query_series_chat_extraction_split(log: CostLog) -> None:
    base = datetime(2026, 5, 9, 14, 5, 0, tzinfo=UTC)
    log.record(_mk_event(ts=base, kind="chat",       cost_usd=0.10))
    log.record(_mk_event(ts=base, kind="extraction", cost_usd=0.02))
    log.record(_mk_event(ts=base, kind="chat",       cost_usd=0.05))

    series = log.query_series(granularity="hour")
    assert len(series) == 1
    bucket = series[0]
    assert bucket["chat_cost_usd"] == pytest.approx(0.15, abs=1e-9)
    assert bucket["extraction_cost_usd"] == pytest.approx(0.02, abs=1e-9)
    assert bucket["cost_usd"] == pytest.approx(0.17, abs=1e-9)


def test_query_series_invalid_granularity(log: CostLog) -> None:
    with pytest.raises(ValueError):
        log.query_series(granularity="week")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_query_aggregate_by_kind(log: CostLog) -> None:
    now = datetime.now(UTC)
    log.record(_mk_event(ts=now, kind="chat",       cost_usd=0.10))
    log.record(_mk_event(ts=now, kind="chat",       cost_usd=0.05))
    log.record(_mk_event(ts=now, kind="extraction", cost_usd=0.02))

    groups = log.query_aggregate(group_by="kind")
    by_key = {g["key"]: g for g in groups}
    assert by_key["chat"]["cost_usd"] == pytest.approx(0.15, abs=1e-9)
    assert by_key["chat"]["event_count"] == 2
    assert by_key["extraction"]["cost_usd"] == pytest.approx(0.02, abs=1e-9)
    assert by_key["extraction"]["event_count"] == 1


def test_query_aggregate_by_model_with_top_child_for_agent(log: CostLog) -> None:
    now = datetime.now(UTC)
    # Alice: opus dominates.
    log.record(_mk_event(ts=now, agent_id="alice", model="claude-opus-4-6",  cost_usd=0.50))
    log.record(_mk_event(ts=now, agent_id="alice", model="claude-haiku-4-5", cost_usd=0.10))
    # Bob: haiku dominates.
    log.record(_mk_event(ts=now, agent_id="bob",   model="claude-opus-4-6",  cost_usd=0.05))
    log.record(_mk_event(ts=now, agent_id="bob",   model="claude-haiku-4-5", cost_usd=0.40))

    # group_by=model has no top_child (only meaningful for agent grouping).
    by_model = log.query_aggregate(group_by="model")
    by_key = {g["key"]: g for g in by_model}
    assert by_key["claude-opus-4-6"]["cost_usd"] == pytest.approx(0.55, abs=1e-9)
    assert by_key["claude-haiku-4-5"]["cost_usd"] == pytest.approx(0.50, abs=1e-9)
    assert all(g["top_child"] is None for g in by_model)

    # group_by=agent populates top_child.
    by_agent = log.query_aggregate(group_by="agent")
    agents = {g["key"]: g for g in by_agent}
    assert agents["alice"]["top_child"] == "claude-opus-4-6"
    assert agents["bob"]["top_child"] == "claude-haiku-4-5"


def test_query_aggregate_invalid_group_by(log: CostLog) -> None:
    with pytest.raises(ValueError):
        log.query_aggregate(group_by="not-a-column")


# ---------------------------------------------------------------------------
# list_distinct
# ---------------------------------------------------------------------------


def test_list_distinct(log: CostLog) -> None:
    now = datetime.now(UTC)
    log.record(_mk_event(
        ts=now, agent_id="alice", source_agent_id="alice/weather",
        model="claude-opus-4-6", user_id="u1",
    ))
    log.record(_mk_event(ts=now, agent_id="bob",   model="claude-opus-4-6", user_id=None))
    log.record(_mk_event(
        ts=now, agent_id="alice", source_agent_id="alice/finance",
        model="gpt-4o", user_id="u2",
    ))

    assert log.list_distinct("agent_id") == ["alice", "bob"]
    assert log.list_distinct("source_agent_id") == ["alice/finance", "alice/weather"]
    assert log.list_distinct("model") == ["claude-opus-4-6", "gpt-4o"]
    assert log.list_distinct("user_id") == ["u1", "u2"]

    with pytest.raises(ValueError):
        log.list_distinct("not-a-column")


# ---------------------------------------------------------------------------
# Filter composition
# ---------------------------------------------------------------------------


def test_query_filters_compose(log: CostLog) -> None:
    base = datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)
    # Alice / u1 / chat / claude
    log.record(_mk_event(
        ts=base, agent_id="alice", source_agent_id="alice/weather", user_id="u1", kind="chat",
        model="claude-opus-4-6", cost_usd=1.0,
    ))
    # Alice / u1 / extraction / claude  (filtered out by kind)
    log.record(_mk_event(
        ts=base + timedelta(minutes=1), agent_id="alice", source_agent_id="alice/weather", user_id="u1",
        kind="extraction", model="claude-haiku-4-5", cost_usd=0.5,
    ))
    # Alice / u2 / chat                  (filtered out by user_id)
    log.record(_mk_event(
        ts=base + timedelta(minutes=2), agent_id="alice", source_agent_id="alice/weather", user_id="u2",
        kind="chat", model="claude-opus-4-6", cost_usd=0.3,
    ))
    # Bob   / u1 / chat                  (filtered out by agent_id)
    log.record(_mk_event(
        ts=base + timedelta(minutes=3), agent_id="bob", user_id="u1",
        kind="chat", model="claude-opus-4-6", cost_usd=0.2,
    ))
    # Alice / u1 / chat — but OUTSIDE the time range
    log.record(_mk_event(
        ts=base + timedelta(hours=2), agent_id="alice", source_agent_id="alice/weather", user_id="u1",
        kind="chat", model="claude-opus-4-6", cost_usd=99.0,
    ))
    # Alice / u1 / chat but another source agent (filtered out by source_agent_id).
    log.record(_mk_event(
        ts=base + timedelta(minutes=4), agent_id="alice", source_agent_id="alice/finance", user_id="u1",
        kind="chat", model="claude-opus-4-6", cost_usd=0.7,
    ))

    totals = log.query_totals(
        agent_id="alice",
        source_agent_id="alice/weather",
        user_id="u1",
        kinds=["chat"],
        since=base,
        until=base + timedelta(minutes=30),
    )
    assert totals["event_count"] == 1
    assert totals["cost_usd"] == pytest.approx(1.0, abs=1e-9)


def test_query_filters_models_in_clause(log: CostLog) -> None:
    now = datetime.now(UTC)
    log.record(_mk_event(ts=now, model="claude-opus-4-6", cost_usd=1.0))
    log.record(_mk_event(ts=now, model="claude-haiku-4-5", cost_usd=0.5))
    log.record(_mk_event(ts=now, model="gpt-4o",           cost_usd=0.3))

    totals = log.query_totals(models=["claude-opus-4-6", "gpt-4o"])
    assert totals["event_count"] == 2
    assert totals["cost_usd"] == pytest.approx(1.3, abs=1e-9)


# ---------------------------------------------------------------------------
# record_chat / record_extraction convenience methods
# ---------------------------------------------------------------------------


def test_record_chat_writes_kind_chat(log: CostLog) -> None:
    log.record_chat(
        model="claude-opus-4-6", provider="anthropic",
        input_tokens=1000, output_tokens=100,
        cache_creation_tokens=0, cache_read_tokens=0,
        cost_usd=0.01, agent_id="alice", source_agent_id="alice/weather",
        user_id=None, request_id="r1",
    )
    events = log.query_events()
    assert len(events) == 1
    assert events[0]["kind"] == "chat"
    assert events[0]["ok"] is True
    assert events[0]["request_id"] == "r1"
    assert events[0]["source_agent_id"] == "alice/weather"


def test_record_extraction_writes_kind_extraction_and_ok_flag(log: CostLog) -> None:
    log.record_extraction(
        model="claude-haiku-4-5", provider="anthropic",
        input_tokens=0, output_tokens=0,
        cache_creation_tokens=0, cache_read_tokens=0,
        cost_usd=0.0, agent_id="alice", source_agent_id="alice/weather",
        user_id=None, request_id="r1", ok=False,
        extraction_side=ExtractionSide.INPUT,
    )
    events = log.query_events()
    assert len(events) == 1
    assert events[0]["kind"] == "extraction"
    assert events[0]["ok"] is False
    assert events[0]["extraction_side"] is ExtractionSide.INPUT
    assert events[0]["skipped"] is False
    assert events[0]["source_agent_id"] == "alice/weather"


def test_record_extraction_roundtrips_each_side(log: CostLog) -> None:
    for side in (ExtractionSide.INPUT, ExtractionSide.OUTPUT):
        log.record_extraction(
            model="claude-haiku-4-5", provider="anthropic",
            input_tokens=10, output_tokens=2,
            cache_creation_tokens=0, cache_read_tokens=0,
            cost_usd=0.0001, agent_id="alice", user_id=None,
            request_id=f"r-{side.value}", ok=True,
            extraction_side=side,
        )
    events = sorted(log.query_events(), key=lambda e: e["request_id"])
    assert [e["extraction_side"] for e in events] == [
        ExtractionSide.INPUT, ExtractionSide.OUTPUT,
    ]
    assert all(e["skipped"] is False for e in events)


def test_record_extraction_with_skipped_flag(log: CostLog) -> None:
    log.record_extraction(
        model="claude-haiku-4-5", provider="anthropic",
        input_tokens=0, output_tokens=0,
        cache_creation_tokens=0, cache_read_tokens=0,
        cost_usd=0.0, agent_id="alice", user_id=None, request_id="r1", ok=True,
        extraction_side=ExtractionSide.INPUT,
        skipped=True,
    )
    events = log.query_events()
    assert len(events) == 1
    assert events[0]["skipped"] is True
    assert events[0]["ok"] is True
    assert events[0]["input_tokens"] == 0
    assert events[0]["cost_usd"] == 0.0


def test_record_invalid_kind_raises(log: CostLog) -> None:
    with pytest.raises(ValueError):
        log.record(_mk_event(ts=datetime.now(UTC), kind="bogus"))
