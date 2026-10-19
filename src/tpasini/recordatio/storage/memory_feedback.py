"""Persistence for the nightly memory-feedback loop.

The nightly analyst (in telos) POSTs a per-day memory-feedback record: a
prose ``summary``, structured ``findings``, and a list of candidate *ops*
— reversible mutations a human can later Apply or Dismiss from the Recordatio
dashboard (tombstone / merge / correct). This store keeps that record and
the per-op lifecycle; the actual apply logic lives with the canonical
:class:`SQLiteStore` (a memory mutation) and is wired in the dashboard
routes. Kept in its own sidecar DB (like ``briefing_events.db`` /
``cost_events.db``) so feedback churn never touches the frozen memory DB.

Contract:
- ``record(...)`` inserts one feedback row + its ops (all ``pending``) and
  returns the feedback id. Idempotent per ``(user_id, day)``: re-recording a
  day replaces the prior record and its still-pending ops (applied/dismissed
  ops are preserved as an audit trail).
- ``list_feedback`` / ``get_feedback`` / ``list_ops`` are read paths for the
  dashboard.
- ``mark_op`` transitions one op to ``applied`` or ``dismissed`` with an
  audit stamp; it never mutates memory itself.
"""
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_OP_TYPES = frozenset({"tombstone", "merge", "correct"})
_OP_STATUSES = frozenset({"pending", "applied", "dismissed"})


@dataclass(frozen=True)
class MemoryFeedbackOp:
    """One candidate memory mutation proposed by the analyst."""

    id: int
    feedback_id: int
    op_type: str
    target: dict[str, Any]
    reason: str
    status: str
    decided_at: str | None
    decided_result: str | None


@dataclass(frozen=True)
class MemoryFeedbackRecord:
    """A single night's memory-module feedback for one user."""

    id: int
    user_id: str
    day: str
    created_at: str
    summary: str
    findings: list[dict[str, Any]]


class MemoryFeedbackStore:
    """SQLite-backed store for nightly memory feedback and its ops."""

    def __init__(self, db_path: Path) -> None:
        # Same threading model as the other sidecar stores (config_store,
        # cost_log): one connection opened with ``check_same_thread=False`` +
        # a ``threading.Lock`` around writes, WAL so reads don't block. Lets
        # the async request thread and any background thread share it safely.
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._setup()

    def _setup(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                day TEXT NOT NULL,
                created_at TEXT NOT NULL,
                summary TEXT NOT NULL,
                findings_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS memory_feedback_user_day
                ON memory_feedback(user_id, day);
            CREATE TABLE IF NOT EXISTS memory_feedback_op (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id INTEGER NOT NULL,
                op_type TEXT NOT NULL,
                target_json TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                decided_at TEXT,
                decided_result TEXT
            );
            CREATE INDEX IF NOT EXISTS memory_feedback_op_feedback
                ON memory_feedback_op(feedback_id);
            CREATE INDEX IF NOT EXISTS memory_feedback_op_status
                ON memory_feedback_op(status);
            """
        )
        self._conn.commit()

    def record(
        self,
        *,
        user_id: str,
        day: str,
        summary: str,
        findings: list[dict[str, Any]],
        ops: list[dict[str, Any]],
    ) -> int:
        """Insert a feedback record + its candidate ops; return the id.

        Idempotent per ``(user_id, day)``: any prior record for that day and
        its still-``pending`` ops are dropped first, so a re-run replaces
        undecided proposals while applied/dismissed ops survive as audit.

        Raises:
            ValueError: an op has an unknown ``op_type``.
        """
        for op in ops:
            if op.get("op_type") not in _OP_TYPES:
                raise ValueError(f"unknown op_type {op.get('op_type')!r}; expected one of {sorted(_OP_TYPES)}")
        now = datetime.now(UTC).isoformat()
        with self._lock, self._conn:
            prior = [
                int(r["id"])
                for r in self._conn.execute(
                    "SELECT id FROM memory_feedback WHERE user_id = ? AND day = ?",
                    (user_id, day),
                )
            ]
            for fid in prior:
                self._conn.execute(
                    "DELETE FROM memory_feedback_op WHERE feedback_id = ? AND status = 'pending'",
                    (fid,),
                )
                # Keep the parent row only if it still has decided ops to audit.
                remaining = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM memory_feedback_op WHERE feedback_id = ?",
                    (fid,),
                ).fetchone()["n"]
                if remaining == 0:
                    self._conn.execute("DELETE FROM memory_feedback WHERE id = ?", (fid,))
            cur = self._conn.execute(
                "INSERT INTO memory_feedback (user_id, day, created_at, summary, findings_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, day, now, summary, json.dumps(findings)),
            )
            feedback_id = int(cur.lastrowid or 0)
            for op in ops:
                self._conn.execute(
                    "INSERT INTO memory_feedback_op (feedback_id, op_type, target_json, reason, status) "
                    "VALUES (?, ?, ?, ?, 'pending')",
                    (feedback_id, op["op_type"], json.dumps(op.get("target", {})), op.get("reason", "")),
                )
        return feedback_id

    def list_all(self, *, limit: int = 2000) -> list[MemoryFeedbackRecord]:
        """All users' feedback records, newest day first (operator dashboard).

        The cap is high (not paginated) so the operator view + recap never
        silently hide records — a 60-row default once undercounted a
        multi-month backfill (well over 60 records) as 60, so bulk-apply and the recap
        both worked off a fraction of the real data.
        """
        rows = self._conn.execute(
            "SELECT * FROM memory_feedback ORDER BY day DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_feedback(self, user_id: str, *, limit: int = 30) -> list[MemoryFeedbackRecord]:
        """Feedback records for a user, newest day first."""
        rows = self._conn.execute(
            "SELECT * FROM memory_feedback WHERE user_id = ? ORDER BY day DESC, id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_feedback(self, feedback_id: int) -> MemoryFeedbackRecord | None:
        row = self._conn.execute(
            "SELECT * FROM memory_feedback WHERE id = ?", (feedback_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def list_ops(self, feedback_id: int) -> list[MemoryFeedbackOp]:
        rows = self._conn.execute(
            "SELECT * FROM memory_feedback_op WHERE feedback_id = ? ORDER BY id ASC",
            (feedback_id,),
        ).fetchall()
        return [self._row_to_op(r) for r in rows]

    def get_op(self, op_id: int) -> MemoryFeedbackOp | None:
        row = self._conn.execute(
            "SELECT * FROM memory_feedback_op WHERE id = ?", (op_id,)
        ).fetchone()
        return self._row_to_op(row) if row else None

    def mark_op(self, op_id: int, *, status: str, result: str | None = None) -> None:
        """Transition an op to ``applied``/``dismissed`` with an audit stamp.

        Does NOT mutate memory — the caller applies the mutation against the
        canonical store first, then records the outcome here.

        Raises:
            ValueError: ``status`` is not ``applied`` or ``dismissed``.
        """
        if status not in ("applied", "dismissed"):
            raise ValueError(f"mark_op status must be applied|dismissed, got {status!r}")
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE memory_feedback_op SET status = ?, decided_at = ?, decided_result = ? WHERE id = ?",
                (status, datetime.now(UTC).isoformat(), result, op_id),
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryFeedbackRecord:
        return MemoryFeedbackRecord(
            id=int(row["id"]),
            user_id=row["user_id"],
            day=row["day"],
            created_at=row["created_at"],
            summary=row["summary"],
            findings=json.loads(row["findings_json"]),
        )

    @staticmethod
    def _row_to_op(row: sqlite3.Row) -> MemoryFeedbackOp:
        return MemoryFeedbackOp(
            id=int(row["id"]),
            feedback_id=int(row["feedback_id"]),
            op_type=row["op_type"],
            target=json.loads(row["target_json"]),
            reason=row["reason"],
            status=row["status"],
            decided_at=row["decided_at"],
            decided_result=row["decided_result"],
        )


__all__ = ["MemoryFeedbackStore", "MemoryFeedbackRecord", "MemoryFeedbackOp"]
