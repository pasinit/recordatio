"""Memory reconcile + compaction driven by an external judge (see ``reconcile.service``)."""

from pathlib import Path

from tpasini.recordatio.reconcile.service import ReconcileError, ReconcileService, RunFinishedError
from tpasini.recordatio.reconcile.settings import ReconcileSettings
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore
from tpasini.recordatio.storage.reconcile_log import ReconcileLog


def reconcile_log_path(db_path: Path) -> Path:
    """The audit/run sidecar lives next to the canonical DB."""
    return db_path.parent / "memory_reconcile.db"


def build_service(
    store: SQLiteStore,
    *,
    settings: ReconcileSettings,
    feedback: MemoryFeedbackStore | None,
    log: ReconcileLog | None = None,
) -> ReconcileService:
    return ReconcileService(
        store,
        log or ReconcileLog(reconcile_log_path(store.db_path)),
        settings=settings,
        feedback=feedback,
    )


__all__ = [
    "ReconcileError",
    "ReconcileService",
    "ReconcileSettings",
    "RunFinishedError",
    "build_service",
    "reconcile_log_path",
]
