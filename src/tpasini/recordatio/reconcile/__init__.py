"""Nightly reconcile + compaction of the memory store (see ``reconcile.runner``)."""

from pathlib import Path

from tpasini.recordatio.proxy.config import resolve_model
from tpasini.recordatio.reconcile.judge import LLMJudge, PairJudge
from tpasini.recordatio.reconcile.runner import Reconciler
from tpasini.recordatio.reconcile.settings import ReconcileSettings
from tpasini.recordatio.storage.base import SQLiteStore
from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore
from tpasini.recordatio.storage.reconcile_log import ReconcileLog


def reconcile_log_path(db_path: Path) -> Path:
    """The audit/cursor sidecar lives next to the canonical DB."""
    return db_path.parent / "memory_reconcile.db"


def make_judge(settings: ReconcileSettings, *, api_key: str | None = None) -> LLMJudge:
    """Build the LLM judge from settings. ``api_key`` is the fallback when no
    reconcile/extraction key is configured (the caller's pass-through key,
    exactly like extraction).

    Raises:
        ValueError: no judge model configured.
    """
    if not settings.model:
        raise ValueError("no judge model: set RECORDATIO_RECONCILE_MODEL or RECORDATIO_EXTRACTION_MODEL")
    bare, backend, base_url = resolve_model(settings.model)
    return LLMJudge(
        model=bare,
        backend=backend,
        api_key=settings.api_key or api_key,
        base_url=base_url,
        max_tokens=settings.judge_max_tokens,
        timeout_s=settings.judge_timeout_s,
    )


def build_reconciler(
    store: SQLiteStore,
    *,
    judge: PairJudge,
    settings: ReconcileSettings,
    feedback: MemoryFeedbackStore | None,
    log: ReconcileLog | None = None,
) -> Reconciler:
    return Reconciler(
        store,
        log or ReconcileLog(reconcile_log_path(store.db_path)),
        judge=judge,
        settings=settings,
        feedback=feedback,
    )


__all__ = [
    "ReconcileSettings",
    "Reconciler",
    "build_reconciler",
    "make_judge",
    "reconcile_log_path",
]
