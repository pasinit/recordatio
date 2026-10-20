"""Sidecar SQLite store of ingested agent transcripts' idempotency keys.

``POST /v1/agents/{user_id}/{slug}/transcripts`` hands Recordatio an already
finished exchange to extract from. Callers retry, so each submission carries
an ``idempotency_key``; this log records the keys that were accepted (scoped
per ``(user_id, slug)``) so a repeat is answered without re-extracting.

Lives in ``transcripts.db`` next to ``AGENTMEM_DB``. Same threading model as
``briefing_log.BriefingLog``: one connection, a ``threading.Lock`` around
writes, WAL mode.
"""

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_transcripts (
    user_id          TEXT NOT NULL,
    slug             TEXT NOT NULL,
    idempotency_key  TEXT NOT NULL,
    source           TEXT NOT NULL,
    occurred_at      TEXT NOT NULL,
    received_at      TEXT NOT NULL,
    PRIMARY KEY (user_id, slug, idempotency_key)
);
"""


class TranscriptLog:
    """Idempotency keys of accepted agent transcripts."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def claim(self, user_id: str, slug: str, idempotency_key: str, *, source: str, occurred_at: str) -> bool:
        """Record the key; ``True`` if it is new, ``False`` if already accepted."""
        with self._lock:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO agent_transcripts "
                "(user_id, slug, idempotency_key, source, occurred_at, received_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, slug, idempotency_key, source, occurred_at, datetime.now(UTC).isoformat()),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def release(self, user_id: str, slug: str, idempotency_key: str) -> None:
        """Forget a claimed key (its extraction could not be scheduled)."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM agent_transcripts WHERE user_id = ? AND slug = ? AND idempotency_key = ?",
                (user_id, slug, idempotency_key),
            )
            self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM agent_transcripts").fetchone()
        return int(row[0])
