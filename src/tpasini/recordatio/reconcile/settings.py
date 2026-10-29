"""Settings for the nightly reconcile/compaction job (env prefix ``RECORDATIO_RECONCILE_``)."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReconcileSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="recordatio_reconcile_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Judge model in ``provider/model`` syntax (same as the extraction model);
    # falls back to RECORDATIO_EXTRACTION_MODEL so a deployment needs no new
    # variable. The judge only emits structured verdicts — never memory text.
    model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("recordatio_reconcile_model", "recordatio_extraction_model"),
    )
    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("recordatio_reconcile_api_key", "recordatio_extraction_api_key"),
    )
    judge_max_tokens: int = 8000
    judge_timeout_s: float = 120.0

    # Per-user, per-run bounds. A large store converges over several nights:
    # pairs already judged "keep both" are cached, so each night spends its
    # budget on pairs it has not seen.
    max_judge_calls_per_user: int = 60
    max_pairs_per_user: int = 150
    # Older units re-scanned per scope per night on top of everything new.
    rolling_slice: int = 200
    # Candidate signals.
    cosine_threshold: float = 0.85
    fts_neighbours: int = 5
    # Neighbours judged against the target of a backlog tombstone op.
    tombstone_neighbours: int = 5

    # Snapshots retained under ``<db dir>/reconcile-snapshots`` (one per
    # mutating run).
    snapshot_keep: int = 7


__all__ = ["ReconcileSettings"]
