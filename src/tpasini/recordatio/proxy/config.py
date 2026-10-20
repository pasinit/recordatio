"""Proxy configuration and provider routing."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings for the proxy process."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Extraction LLM - model is required and uses provider/model syntax (e.g. "ollama/qwen2.5:14b").
    # api_key only needed when the extraction provider differs from the upstream.
    recordatio_extraction_model: str
    recordatio_extraction_api_key: str | None = None

    # Additive CLI support: existing deployments keep Claude unless selected.
    recordatio_subscription_provider: Literal["claude", "codex"] = "claude"
    recordatio_agent_api_fallback: bool | None = None

    # MCP stays restricted to loopback and the default Compose service.
    # JSON array override for deployments using another trusted hostname.
    recordatio_mcp_allowed_hosts: list[str] = ["127.0.0.1:*", "localhost:*", "[::1]:*", "recordatio:8787"]

    # Read-only remote access (proxy/network_guard.py). JSON arrays. Callers from
    # these CIDRs must present a recall key; a recall key (stored as its SHA-256
    # hex digest) may only call GET /v1/users/{user_id}/memory-search and
    # GET /v1/users/{user_id}/briefing.
    recordatio_restricted_networks: list[str] = []
    recordatio_recall_key_sha256: list[str] = []

    # Storage
    agentmem_db: str = str(Path.home() / ".agentmem" / "memories.db")

    # Observability. Empty endpoint keeps tracing disabled.
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "recordatio"


# Maps provider prefix -> (pontifex backend, default base_url).
# Clients send model as "provider/model-name"; the prefix selects the backend
# and default endpoint. Unknown prefixes fall back to openai + no base_url.
_PROVIDER_REGISTRY: dict[str, tuple[str, str | None]] = {
    "codex": ("codex", None),
    "anthropic": ("anthropic", None),
    "openai": ("openai", None),
    "ollama": ("openai", "http://localhost:11434/v1"),
    "lmstudio": ("openai", "http://localhost:1234/v1"),
    "gemini": ("openai", "https://generativelanguage.googleapis.com/v1beta/openai/"),
}


def resolve_model(model: str) -> tuple[str, str, str | None]:
    """Parse 'provider/model' and return (bare_model, backend, base_url)."""
    if "/" in model:
        provider, bare = model.split("/", 1)
        backend, base_url = _PROVIDER_REGISTRY.get(provider.lower(), ("openai", None))
        return bare, backend, base_url
    return model, "openai", None
