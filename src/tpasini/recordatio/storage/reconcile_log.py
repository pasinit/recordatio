"""Durable audit + incremental state for the nightly memory reconcile.

Kept in its own sidecar DB (``memory_reconcile.db``, like
``memory_feedback.db``) so the canonical memory DB never grows schema for
it — a new table there would change a frozen artifact's bytes on open.

Tables:
- ``reconcile_run``      — one row per run (applied runs only; a dry run
  writes nothing anywhere) with its JSON summary and snapshot path.
- ``reconcile_deletion`` — one row per physically deleted unit: who kept it,
  why (relation + judge evidence), which run, when. The only surviving
  trace of a deleted unit outside the retained snapshots.
- ``reconcile_cursor``   — per-scope incremental cursors: the highest rowid
  already scanned as "new", and the rolling slice position over older rows.
- ``reconcile_judged``   — verdict cache for pairs the judge said must be
  kept (unrelated / compatible / unsure / unverified), keyed by both units'
  id + content, so a stable store is not re-judged every night.
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


class ReconcileLog:
    """SQLite-backed audit/cursor store for the reconcile job."""

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
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                scope TEXT NOT NULL,
                snapshot_path TEXT,
                summary_json TEXT
            );
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

    def start_run(self, run_id: str, scope: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO reconcile_run (id, started_at, scope) VALUES (?, ?, ?)",
                (run_id, datetime.now(UTC).isoformat(), scope),
            )

    def set_snapshot(self, run_id: str, path: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE reconcile_run SET snapshot_path = ? WHERE id = ?", (path, run_id))

    def finish_run(self, run_id: str, summary: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE reconcile_run SET finished_at = ?, summary_json = ? WHERE id = ?",
                (datetime.now(UTC).isoformat(), json.dumps(summary, sort_keys=True), run_id),
            )

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

    def deletions(self, run_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM reconcile_deletion"
        params: tuple[object, ...] = ()
        if run_id is not None:
            sql += " WHERE run_id = ?"
            params = (run_id,)
        rows = self._conn.execute(sql + " ORDER BY id", params).fetchall()
        return [{**dict(r), "evidence": json.loads(r["evidence_json"])} for r in rows]

    def runs(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM reconcile_run ORDER BY started_at, id").fetchall()
        return [dict(r) for r in rows]

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


__all__ = ["DeletionRecord", "ReconcileLog", "ScopeCursor"]
