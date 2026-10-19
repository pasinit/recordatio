"""SQLite-backed cache of the most recent ``RateLimitEvent`` per (user_id, slug).

Per ``MIGRATION_PLAN.md`` Phase 1, the runner uses the cached snapshot to
choose between ``SubscriptionTransport`` and ``ApiKeyTransport``:

- If the last seen event had ``status == "exceeded"`` (and ``resets_at``
  has not passed yet), the next task takes the API-key path.
- Recovery is time-based: once ``resets_at`` passes, the subscription is
  retried directly. ``overage_status == "rejected"`` is a static policy
  descriptor (no pay-per-use overage on this account), not a health signal
  — see ``RateLimitSnapshot.should_fall_back``.

We keep a tiny schema — one row per (user_id, slug), upserted on every
subscription call. The connection pooling matches ``briefing_log`` and
``cost_log``: a single connection with ``check_same_thread=False`` plus a
``threading.Lock`` around writes; WAL journaling lets readers proceed
lock-free.
"""

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_rate_limit_state (
    user_id                  TEXT NOT NULL,
    slug                     TEXT NOT NULL,
    status                   TEXT,
    rate_limit_type          TEXT,
    overage_status           TEXT,
    overage_disabled_reason  TEXT,
    utilization              REAL,
    resets_at                TEXT,
    overage_resets_at        TEXT,
    observed_at              TEXT NOT NULL,
    PRIMARY KEY (user_id, slug)
);
"""


@dataclass(frozen=True)
class RateLimitSnapshot:
    """A flattened view of the SDK's ``RateLimitInfo`` plus when it was seen.

    Frozen so it can be passed across the runner's selection logic without
    risk of accidental mutation. ``None`` for any field means the SDK didn't
    surface that key on the most recent observation.
    """

    user_id: str
    slug: str
    status: str | None
    rate_limit_type: str | None
    overage_status: str | None
    overage_disabled_reason: str | None
    utilization: float | None
    resets_at: str | None
    overage_resets_at: str | None
    observed_at: datetime

    @property
    def should_fall_back(self) -> bool:
        """True when the cached snapshot says the subscription path is unusable.

        Only ``status == "exceeded"`` (the 5-hour window is depleted) trips
        this — and only until ``resets_at`` passes, after which the next turn
        retries the subscription without waiting for a healthy snapshot
        (the fallback path never records subscription snapshots, so waiting
        for one would leave us stuck on the fallback forever).

        ``overage_status == "rejected"`` is deliberately NOT a signal: it is
        a static *policy* descriptor present on every healthy event for
        accounts without pay-per-use overage ("if you exhaust the window,
        you get rejected" — the Phase 0 finding, MIGRATION_NOTES.md gotcha
        #4), not a statement about current health. Treating it as a state
        signal routed every post-first-snapshot turn to the API fallback
        while the subscription sat at 20% utilization (prod, 2026-06-10).
        """
        if (self.status or "").lower() != "exceeded":
            return False
        if self.resets_at:
            try:
                resets_epoch = float(self.resets_at)
            except ValueError:
                return True
            if datetime.now(UTC).timestamp() >= resets_epoch:
                return False
        return True


class RateLimitStateStore:
    """Per-(user_id, slug) cache of the most recent ``RateLimitInfo``."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, user_id: str, slug: str) -> RateLimitSnapshot | None:
        row = self._conn.execute(
            "SELECT * FROM agent_rate_limit_state WHERE user_id = ? AND slug = ?",
            (user_id, slug),
        ).fetchone()
        if row is None:
            return None
        return RateLimitSnapshot(
            user_id=user_id,
            slug=slug,
            status=row["status"],
            rate_limit_type=row["rate_limit_type"],
            overage_status=row["overage_status"],
            overage_disabled_reason=row["overage_disabled_reason"],
            utilization=row["utilization"],
            resets_at=row["resets_at"],
            overage_resets_at=row["overage_resets_at"],
            observed_at=datetime.fromisoformat(row["observed_at"]),
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert(
        self,
        *,
        user_id: str,
        slug: str,
        status: str | None,
        rate_limit_type: str | None,
        overage_status: str | None,
        overage_disabled_reason: str | None,
        utilization: float | None,
        resets_at: str | None,
        overage_resets_at: str | None,
        observed_at: datetime | None = None,
    ) -> RateLimitSnapshot:
        """Insert or replace the snapshot for (user_id, slug)."""
        observed_iso = (observed_at or datetime.now(UTC)).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO agent_rate_limit_state
                    (user_id, slug, status, rate_limit_type, overage_status,
                     overage_disabled_reason, utilization, resets_at,
                     overage_resets_at, observed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, slug) DO UPDATE SET
                    status                  = excluded.status,
                    rate_limit_type         = excluded.rate_limit_type,
                    overage_status          = excluded.overage_status,
                    overage_disabled_reason = excluded.overage_disabled_reason,
                    utilization             = excluded.utilization,
                    resets_at               = excluded.resets_at,
                    overage_resets_at       = excluded.overage_resets_at,
                    observed_at             = excluded.observed_at
                """,
                (
                    user_id,
                    slug,
                    status,
                    rate_limit_type,
                    overage_status,
                    overage_disabled_reason,
                    utilization,
                    resets_at,
                    overage_resets_at,
                    observed_iso,
                ),
            )
            self._conn.commit()
        snapshot = self.get(user_id, slug)
        # Cannot be None — we just wrote it under the same lock.
        assert snapshot is not None
        return snapshot

    def clear(self, user_id: str, slug: str) -> None:
        """Drop the cached snapshot. Used by tests and admin tooling."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM agent_rate_limit_state WHERE user_id = ? AND slug = ?",
                (user_id, slug),
            )
            self._conn.commit()
