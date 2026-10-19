"""Settings for extraction runtime and output contracts."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="recordatio_extraction_", extra="ignore")

    model: str = "claude-haiku-5-5"
    backend: str = "anthropic"
    # Haiku 5.5 thinks by default; thinking tokens count against this cap.
    max_tokens: int = 16000
    window: int = 6


class ExtractionContractSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="recordatio_extraction_contract_", extra="ignore")

    max_relation_endpoint_chars: int = 96
    max_relation_endpoint_words: int = 12
    max_relation_predicate_chars: int = 64
    max_memory_content_chars: int = 768
    max_evidence_quote_chars: int = 512
    max_candidate_id_chars: int = 96


EXTRACTION_SETTINGS = ExtractionSettings()
CONTRACT_SETTINGS = ExtractionContractSettings()
