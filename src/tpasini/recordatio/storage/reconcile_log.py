"""Durable audit + run state for the memory reconcile.

Kept in its own sidecar DB (``memory_reconcile.db``, like
``memory_feedback.db``) so the canonical memory DB never grows schema for
it — a new table there would change a frozen artifact's bytes on open.

Tables:
- ``reconcile_run``      — one row per (user, run id) once the run first
  mutates: snapshot path, finish time and the finish summary. Dry runs write
  nothing anywhere.
- ``reconcile_verdict``  — the recorded outcome of every submitted verdict,
  keyed by (user, pair token). A replayed token returns this row and never
  has a second effect.
- ``reconcile_deletion`` — one row per physically deleted unit: who kept it,
  why (relation + judge evidence), which run, when. The only surviving
  trace of a deleted unit outside the retained snapshots.
- ``reconcile_cursor``   — per-scope incremental cursors: the highest rowid
  already scanned as "new", and the rolling slice position over older rows.
- ``reconcile_judged``   — verdict cache for pairs that must be kept
  (unrelated / compatible / unsure / rejected), keyed by both units' id +
  content, so a stable store is not re-judged every night.
"""
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeletionRecord:
    run_id: str
    unit_id: str
    agent_id: str
    user_id: str
    kept_unit_id: str | None
    relation: str
    source: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class ScopeCursor:
    new_watermark: int  # rows with rowid > this are "new since last run"
    rolling_after: int  # the rolling slice over older rows resumes after this rowid


@dataclass(frozen=True)
class RunRow:
    run_id: str
    user_id: str
    started_at: str
    finished_at: str | None
    snapshot_path: str | None
    summary: dict[str, Any] | None


class ReconcileLog:
    """SQLite-backed audit/run store for the reconcile service."""

    def __init__(self, db_path: Path) -> None:
        # Same threading model as the other sidecar stores (memory_feedback,
        # cost_log): one connection + a lock around writes, WAL for readers.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS reconcile_run (
                run_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                snapshot_path TEXT,
                summary_json TEXT,
                PRIMARY KEY (user_id, run_id)
            );
            CREATE TABLE IF NOT EXISTS reconcile_verdict (
                user_id TEXT NOT NULL,
                pair_token TEXT NOT NULL,
                run_id TEXT NOT NULL,
                op_id INTEGER,
                outcome TEXT NOT NULL,
                relation TEXT NOT NULL,
                reason TEXT NOT NULL,
                kept_unit_id TEXT,
                dropped_unit_id TEXT,
                decided_at TEXT NOT NULL,
                PRIMARY KEY (user_id, pair_token)
            );
            CREATE INDEX IF NOT EXISTS reconcile_verdict_run ON reconcile_verdict(user_id, run_id);
            CREATE INDEX IF NOT EXISTS reconcile_verdict_op ON reconcile_verdict(op_id);
            CREATE TABLE IF NOT EXISTS reconcile_deletion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                deleted_at TEXT NOT NULL,
                unit_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                kept_unit_id TEXT,
                relation TEXT NOT NULL,
                source TEXT NOT NULL,
                evidence_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS reconcile_deletion_run ON reconcile_deletion(run_id);
            CREATE INDEX IF NOT EXISTS reconcile_deletion_unit ON reconcile_deletion(unit_id);
            CREATE TABLE IF NOT EXISTS reconcile_cursor (
                agent_id TEXT PRIMARY KEY,
                new_watermark INTEGER NOT NULL,
                rolling_after INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reconcile_judged (
                pair_key TEXT PRIMARY KEY,
                relation TEXT NOT NULL,
                judged_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # -- runs ---------------------------------------------------------------

    def get_run(self, user_id: str, run_id: str) -> RunRow | None:
        row = self._conn.execute(
            "SELECT * FROM reconcile_run WHERE user_id = ? AND run_id = ?", (user_id, run_id)
        ).fetchone()
        if row is None:
            return None
        return RunRow(
            run_id=row["run_id"],
            user_id=row["user_id"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            snapshot_path=row["snapshot_path"],
            summary=json.loads(row["summary_json"]) if row["summary_json"] else None,
        )

    def ensure_run(self, user_id: str, run_id: str) -> RunRow:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO reconcile_run (run_id, user_id, started_at) VALUES (?, ?, ?)",
                (run_id, user_id, datetime.now(UTC).isoformat()),
            )
        run = self.get_run(user_id, run_id)
        assert run is not None
        return run

    def set_snapshot(self, user_id: str, run_id: str, path: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE reconcile_run SET snapshot_path = ? WHERE user_id = ? AND run_id = ?", (path, user_id, run_id)
            )

    def finish_run(self, user_id: str, run_id: str, summary: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE reconcile_run SET finished_at = ?, summary_json = ? WHERE user_id = ? AND run_id = ?",
                (datetime.now(UTC).isoformat(), json.dumps(summary, sort_keys=True), user_id, run_id),
            )

    def runs(self) -> list[RunRow]:
        rows = self._conn.execute("SELECT user_id, run_id FROM reconcile_run ORDER BY started_at").fetchall()
        return [r for r in (self.get_run(row["user_id"], row["run_id"]) for row in rows) if r is not None]

    # -- verdicts -----------------------------------------------------------

    def get_verdict(self, user_id: str, pair_token: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM reconcile_verdict WHERE user_id = ? AND pair_token = ?", (user_id, pair_token)
        ).fetchone()
        return dict(row) if row else None

    def record_verdict(
        self,
        *,
        user_id: str,
        pair_token: str,
        run_id: str,
        op_id: int | None,
        outcome: str,
        relation: str,
        reason: str,
        kept_unit_id: str | None,
        dropped_unit_id: str | None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO reconcile_verdict (user_id, pair_token, run_id, op_id, outcome, relation, "
                "reason, kept_unit_id, dropped_unit_id, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, pair_token, run_id, op_id, outcome, relation, reason, kept_unit_id, dropped_unit_id,
                 datetime.now(UTC).isoformat()),
            )

    def run_verdicts(self, user_id: str, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM reconcile_verdict WHERE user_id = ? AND run_id = ? ORDER BY decided_at",
            (user_id, run_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def op_verdicts(self, user_id: str, op_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM reconcile_verdict WHERE user_id = ? AND op_id = ? ORDER BY decided_at", (user_id, op_id)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- deletions ----------------------------------------------------------

    def record_deletion(self, rec: DeletionRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO reconcile_deletion (run_id, deleted_at, unit_id, agent_id, user_id, kept_unit_id, "
                "relation, source, evidence_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rec.run_id,
                    datetime.now(UTC).isoformat(),
                    rec.unit_id,
                    rec.agent_id,
                    rec.user_id,
                    rec.kept_unit_id,
                    rec.relation,
                    rec.source,
                    json.dumps(rec.evidence, sort_keys=True),
                ),
            )

    def deletions(self, run_id: str | None = None, *, unit_id: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if unit_id is not None:
            clauses.append("unit_id = ?")
            params.append(unit_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(f"SELECT * FROM reconcile_deletion{where} ORDER BY id", tuple(params)).fetchall()  # noqa: S608
        return [{**dict(r), "evidence": json.loads(r["evidence_json"])} for r in rows]

    # -- cursors + verdict cache --------------------------------------------

    def get_cursor(self, agent_id: str) -> ScopeCursor:
        row = self._conn.execute(
            "SELECT new_watermark, rolling_after FROM reconcile_cursor WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        if row is None:
            return ScopeCursor(new_watermark=0, rolling_after=0)
        return ScopeCursor(new_watermark=int(row["new_watermark"]), rolling_after=int(row["rolling_after"]))

    def set_cursor(self, agent_id: str, cursor: ScopeCursor) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO reconcile_cursor (agent_id, new_watermark, rolling_after, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(agent_id) DO UPDATE SET "
                "new_watermark = excluded.new_watermark, rolling_after = excluded.rolling_after, "
                "updated_at = excluded.updated_at",
                (agent_id, cursor.new_watermark, cursor.rolling_after, datetime.now(UTC).isoformat()),
            )

    def judged(self, pair_key: str) -> str | None:
        row = self._conn.execute("SELECT relation FROM reconcile_judged WHERE pair_key = ?", (pair_key,)).fetchone()
        return row["relation"] if row else None

    def remember_judged(self, pair_key: str, relation: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO reconcile_judged (pair_key, relation, judged_at) VALUES (?, ?, ?) "
                "ON CONFLICT(pair_key) DO UPDATE SET relation = excluded.relation, judged_at = excluded.judged_at",
                (pair_key, relation, datetime.now(UTC).isoformat()),
            )

    def close(self) -> None:
        self._conn.close()


__all__ = ["DeletionRecord", "ReconcileLog", "RunRow", "ScopeCursor"]
