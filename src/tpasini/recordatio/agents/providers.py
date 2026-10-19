"""Explicit subscription-provider selection; existing deployments stay on Claude."""

from tpasini.recordatio.agents.codex_transport import CodexSubscriptionTransport
from tpasini.recordatio.agents.rate_limit_state import RateLimitStateStore
from tpasini.recordatio.agents.runner import SubscriptionTransport
from tpasini.recordatio.agents.warm_pool import WarmPool


def subscription_transport(
    provider: str, *, rate_state: RateLimitStateStore, warm_pool: WarmPool | None = None,
) -> SubscriptionTransport | CodexSubscriptionTransport:
    if provider == "claude":
        return SubscriptionTransport(rate_state=rate_state, warm_pool=warm_pool)
    if provider == "codex":
        return CodexSubscriptionTransport(rate_state=rate_state, warm_pool=warm_pool)
    raise ValueError(f"Unknown subscription provider: {provider!r}; choose claude or codex")


def api_fallback_enabled(provider: str, configured: bool | None) -> bool:
    """Preserve Claude's existing fallback; Codex API billing requires opt-in."""
    return configured if configured is not None else provider == "claude"
