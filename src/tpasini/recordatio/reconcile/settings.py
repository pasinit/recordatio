"""Settings for the memory reconcile service (env prefix ``RECORDATIO_RECONCILE_``)."""

from pydantic_settings import BaseSettings, SettingsConfigDict

from tpasini.recordatio.reconcile.candidates import PairTuning


class ReconcileSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="recordatio_reconcile_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Per-run budget: pair verdicts one run may submit for one user. A large
    # store converges over several nights — pairs already judged "keep both"
    # are cached, so each run spends its budget on pairs it has not seen.
    max_pairs_per_run: int = 150
    # Upper bound on one candidates batch, whatever the caller asks for.
    max_batch: int = 25
    # Older units re-scanned per scope per run on top of everything new.
    rolling_slice: int = 200
    # Candidate volume (see ``reconcile.candidates.PairTuning``). Measure on a
    # store copy with ``agentmem reconcile --stats`` before changing them.
    cosine_threshold: float = 0.90
    max_neighbours_per_unit: int = 3
    fts_neighbours: int = 3
    # Content-word Jaccard an FTS-only pair needs; above 1 disables FTS-only pairs.
    fts_min_overlap: float = 0.5
    # Neighbours offered against the target of a backlog tombstone op.
    tombstone_neighbours: int = 5
    # Snapshots retained under ``<db dir>/reconcile-snapshots`` (one per
    # mutating run).
    snapshot_keep: int = 7


    def tuning(self) -> PairTuning:
        return PairTuning(
            cosine_threshold=self.cosine_threshold,
            max_per_unit=self.max_neighbours_per_unit,
            fts_neighbours=self.fts_neighbours,
            fts_min_overlap=self.fts_min_overlap,
        )


__all__ = ["ReconcileSettings"]
