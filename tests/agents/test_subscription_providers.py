from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tpasini.recordatio.agents.codex_transport import CodexSubscriptionTransport
from tpasini.recordatio.agents.providers import api_fallback_enabled, subscription_transport
from tpasini.recordatio.agents.rate_limit_state import RateLimitStateStore
from tpasini.recordatio.agents.runner import SubscriptionTransport
from tpasini.recordatio.entity_identity import ClaudeCLIBackend, CodexCLIBackend, cli_backend, judge_from_env
from tpasini.recordatio.proxy.config import Settings


def test_service_defaults_remain_claude(monkeypatch):
    monkeypatch.delenv("RECORDATIO_SUBSCRIPTION_PROVIDER", raising=False)
    monkeypatch.delenv("RECORDATIO_AGENT_API_FALLBACK", raising=False)
    settings = Settings(_env_file=None, recordatio_extraction_model="test-model")
    assert settings.recordatio_subscription_provider == "claude"
    assert api_fallback_enabled(settings.recordatio_subscription_provider, settings.recordatio_agent_api_fallback)


def test_provider_setting_reads_dotenv_and_rejects_typos(tmp_path, monkeypatch):
    monkeypatch.delenv("RECORDATIO_SUBSCRIPTION_PROVIDER", raising=False)
    env = tmp_path / ".env"
    env.write_text("RECORDATIO_EXTRACTION_MODEL=test-model\nRECORDATIO_SUBSCRIPTION_PROVIDER=codex\n")
    assert Settings(_env_file=env).recordatio_subscription_provider == "codex"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, recordatio_extraction_model="test-model", recordatio_subscription_provider="typo")


def test_both_subscription_transports_are_selectable(tmp_path):
    state = RateLimitStateStore(tmp_path / "rate.db")
    try:
        assert isinstance(subscription_transport("claude", rate_state=state), SubscriptionTransport)
        assert isinstance(subscription_transport("codex", rate_state=state), CodexSubscriptionTransport)
        with pytest.raises(ValueError, match="Unknown subscription provider"):
            subscription_transport("typo", rate_state=state)
    finally:
        state.close()


@pytest.mark.parametrize("provider,configured,expected", [
    ("claude", None, True), ("codex", None, False),
    ("claude", False, False), ("codex", True, True),
])
def test_provider_fallback_defaults_and_overrides(provider, configured, expected):
    assert api_fallback_enabled(provider, configured) is expected


def test_identity_provider_defaults_and_selection():
    original = cli_backend()
    assert isinstance(original, ClaudeCLIBackend)
    assert original.model == "claude-haiku-4-5-20251001"
    added = cli_backend("codex", model="openai/test-model", timeout=5)
    assert isinstance(added, CodexCLIBackend)
    assert added.model == "openai/test-model" and added.timeout == 5
    with pytest.raises(ValueError, match="Unknown entity judge provider"):
        cli_backend("typo")


def test_original_claude_cli_still_runs_and_isolates_api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token")
    captured = {}

    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0, stdout='{"result":"original works"}', stderr="")

    monkeypatch.setattr("tpasini.recordatio.entity_identity.subprocess.run", run)
    assert ClaudeCLIBackend().complete("prompt") == "original works"
    assert captured["command"][:2] == ["claude", "-p"]
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert "ANTHROPIC_AUTH_TOKEN" not in captured["env"]


def test_identity_env_is_additive(monkeypatch):
    monkeypatch.setenv("RECORDATIO_ENTITY_JUDGE", "1")
    monkeypatch.delenv("RECORDATIO_ENTITY_JUDGE_PROVIDER", raising=False)
    monkeypatch.delenv("RECORDATIO_ENTITY_JUDGE_MODEL", raising=False)
    judge_from_env.cache_clear()
    try:
        assert isinstance(judge_from_env().llm, ClaudeCLIBackend)
        monkeypatch.setenv("RECORDATIO_ENTITY_JUDGE_PROVIDER", "codex")
        judge_from_env.cache_clear()
        assert isinstance(judge_from_env().llm, CodexCLIBackend)
    finally:
        judge_from_env.cache_clear()
