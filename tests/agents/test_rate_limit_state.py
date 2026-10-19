"""RateLimitStateStore: upsert behaviour and fall-back detection."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tpasini.recordatio.agents.rate_limit_state import RateLimitSnapshot, RateLimitStateStore


@pytest.fixture()
def store(tmp_path: Path) -> RateLimitStateStore:
    return RateLimitStateStore(tmp_path / "rate.db")


def _upsert(store: RateLimitStateStore, **overrides: object) -> RateLimitSnapshot:
    defaults: dict = {
        "user_id": "alice",
        "slug": "weather",
        "status": "allowed",
        "rate_limit_type": "five_hour",
        "overage_status": "rejected",
        "overage_disabled_reason": "out_of_credits",
        "utilization": 0.42,
        "resets_at": "2026-06-07T20:35:00Z",
        "overage_resets_at": None,
    }
    defaults.update(overrides)
    return store.upsert(**defaults)


def test_get_returns_none_when_missing(store: RateLimitStateStore) -> None:
    assert store.get("alice", "weather") is None


def test_upsert_roundtrip(store: RateLimitStateStore) -> None:
    now = datetime.now(UTC)
    snapshot = _upsert(store, observed_at=now)
    assert snapshot.status == "allowed"
    assert snapshot.rate_limit_type == "five_hour"
    assert snapshot.overage_status == "rejected"
    assert snapshot.overage_disabled_reason == "out_of_credits"
    assert snapshot.utilization == pytest.approx(0.42)
    assert snapshot.observed_at.tzinfo is not None


def test_upsert_replaces_existing_row(store: RateLimitStateStore) -> None:
    _upsert(store, status="allowed", utilization=0.1)
    _upsert(store, status="exceeded", utilization=0.99)
    fetched = store.get("alice", "weather")
    assert fetched is not None
    assert fetched.status == "exceeded"
    assert fetched.utilization == pytest.approx(0.99)


def test_clear_drops_row(store: RateLimitStateStore) -> None:
    _upsert(store)
    assert store.get("alice", "weather") is not None
    store.clear("alice", "weather")
    assert store.get("alice", "weather") is None


# ---------------------------------------------------------------------------
# Fall-back detection
# ---------------------------------------------------------------------------


def test_should_fall_back_when_status_exceeded() -> None:
    snap = RateLimitSnapshot(
        user_id="u", slug="s", status="exceeded", rate_limit_type=None,
        overage_status=None, overage_disabled_reason=None, utilization=None,
        resets_at=None, overage_resets_at=None,
        observed_at=datetime.now(UTC),
    )
    assert snap.should_fall_back is True


def test_overage_rejected_policy_alone_does_not_fall_back() -> None:
    """``overage=rejected`` rides on every healthy event for accounts without
    pay-per-use overage — it's a policy descriptor, not a health signal.
    Treating it as one routed all post-first-snapshot turns to the fallback
    while the subscription sat at 20% utilization (prod, 2026-06-10)."""
    snap = RateLimitSnapshot(
        user_id="u", slug="s", status="allowed", rate_limit_type="five_hour",
        overage_status="rejected", overage_disabled_reason="out_of_credits",
        utilization=0.1, resets_at=None, overage_resets_at=None,
        observed_at=datetime.now(UTC),
    )
    assert snap.should_fall_back is False


def test_exceeded_with_passed_reset_recovers() -> None:
    """Once resets_at passes, retry the subscription directly — the fallback
    path never writes subscription snapshots, so waiting for a healthy one
    would pin us to the fallback forever."""
    past_epoch = str(datetime.now(UTC).timestamp() - 60)
    snap = RateLimitSnapshot(
        user_id="u", slug="s", status="exceeded", rate_limit_type="five_hour",
        overage_status="rejected", overage_disabled_reason="out_of_credits",
        utilization=1.0, resets_at=past_epoch, overage_resets_at=None,
        observed_at=datetime.now(UTC),
    )
    assert snap.should_fall_back is False


def test_exceeded_with_future_reset_falls_back() -> None:
    future_epoch = str(datetime.now(UTC).timestamp() + 3600)
    snap = RateLimitSnapshot(
        user_id="u", slug="s", status="exceeded", rate_limit_type="five_hour",
        overage_status="rejected", overage_disabled_reason="out_of_credits",
        utilization=1.0, resets_at=future_epoch, overage_resets_at=None,
        observed_at=datetime.now(UTC),
    )
    assert snap.should_fall_back is True


def test_exceeded_with_malformed_reset_falls_back() -> None:
    snap = RateLimitSnapshot(
        user_id="u", slug="s", status="exceeded", rate_limit_type="five_hour",
        overage_status=None, overage_disabled_reason=None,
        utilization=None, resets_at="not-a-number", overage_resets_at=None,
        observed_at=datetime.now(UTC),
    )
    assert snap.should_fall_back is True


def test_should_not_fall_back_when_healthy() -> None:
    snap = RateLimitSnapshot(
        user_id="u", slug="s", status="allowed", rate_limit_type="five_hour",
        overage_status="allowed", overage_disabled_reason=None,
        utilization=0.1, resets_at=None, overage_resets_at=None,
        observed_at=datetime.now(UTC),
    )
    assert snap.should_fall_back is False
