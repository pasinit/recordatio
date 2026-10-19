"""Agent runner module.

The service supports Claude and Codex CLI subscriptions behind the HTTP+SSE
API. Claude remains the default, including its existing API fallback behavior;
Codex with ChatGPT authentication is an explicit provider choice.

Public symbols here are intentionally kept narrow — the runner is composed
inside ``proxy/agent_endpoints.py`` and only the data types leak out for
typing.
"""

from tpasini.recordatio.agents.config_store import AgentConfig, AgentConfigStore, AgentConfigVersionMismatch
from tpasini.recordatio.agents.events import TaskEvent, TaskEventKind
from tpasini.recordatio.agents.rate_limit_state import RateLimitSnapshot, RateLimitStateStore
from tpasini.recordatio.agents.runner import (
    AgentRunner,
    ApiKeyTransport,
    SubscriptionTransport,
    TransportSelection,
)
from tpasini.recordatio.agents.warm_pool import WarmPool

__all__ = [
    "AgentConfig",
    "AgentConfigStore",
    "AgentConfigVersionMismatch",
    "AgentRunner",
    "ApiKeyTransport",
    "RateLimitSnapshot",
    "RateLimitStateStore",
    "SubscriptionTransport",
    "TaskEvent",
    "TaskEventKind",
    "TransportSelection",
    "WarmPool",
]
