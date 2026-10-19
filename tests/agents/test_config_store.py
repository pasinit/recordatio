"""AgentConfig CRUD + monotonic versioning tests.

The PUT 409 path is the regression-prone bit: the improver relies on it to
detect concurrent edits, so it gets explicit coverage. Roundtrip + version
bumping mirror the briefing_log / cost_log tests in shape.
"""

from pathlib import Path

import pytest

from tpasini.recordatio.agents.config_store import AgentConfig, AgentConfigStore, AgentConfigVersionMismatch


@pytest.fixture()
def store(tmp_path: Path) -> AgentConfigStore:
    return AgentConfigStore(tmp_path / "configs.db")


def test_get_returns_none_when_missing(store: AgentConfigStore) -> None:
    assert store.get("alice", "weather") is None


def test_first_put_inserts_at_version_one(store: AgentConfigStore) -> None:
    config = AgentConfig(
        system_prompt="you are helpful",
        model="anthropic/claude-opus-4-7",
        version=42,  # caller's version is ignored on first insert
    )
    stored = store.put("alice", "weather", config)
    assert stored.version == 1
    assert stored.system_prompt == "you are helpful"
    assert stored.model == "anthropic/claude-opus-4-7"

    fetched = store.get("alice", "weather")
    assert fetched is not None
    assert fetched.version == 1


def test_put_bumps_version_on_match(store: AgentConfigStore) -> None:
    store.put("alice", "weather", AgentConfig(system_prompt="v1"))
    # Caller reads back version=1, edits, writes.
    config = AgentConfig(system_prompt="v2", version=1)
    stored = store.put("alice", "weather", config)
    assert stored.version == 2
    assert stored.system_prompt == "v2"


def test_put_409_path_raises_version_mismatch(store: AgentConfigStore) -> None:
    """Two improvers concurrently bump the same row — second must lose.

    This is the spec's acceptance criterion: a PUT carrying a stale version
    is rejected and surfaces enough info that the caller can re-read and
    retry without a second round-trip to discover the actual version.
    """
    store.put("alice", "weather", AgentConfig(system_prompt="v1"))
    # First improver reads v=1, writes — succeeds, now v=2.
    store.put("alice", "weather", AgentConfig(system_prompt="v2", version=1))

    # Second improver also read v=1 originally; its PUT must fail.
    with pytest.raises(AgentConfigVersionMismatch) as exc_info:
        store.put("alice", "weather", AgentConfig(system_prompt="v2-other", version=1))
    assert exc_info.value.expected == 1
    assert exc_info.value.actual == 2

    # And the row stays at the first writer's value.
    fetched = store.get("alice", "weather")
    assert fetched is not None
    assert fetched.system_prompt == "v2"
    assert fetched.version == 2


def test_isolation_across_users_and_slugs(store: AgentConfigStore) -> None:
    store.put("alice", "weather", AgentConfig(system_prompt="alice-weather"))
    store.put("alice", "news", AgentConfig(system_prompt="alice-news"))
    store.put("bob", "weather", AgentConfig(system_prompt="bob-weather"))

    aw = store.get("alice", "weather")
    an = store.get("alice", "news")
    bw = store.get("bob", "weather")
    assert aw is not None and aw.system_prompt == "alice-weather"
    assert an is not None and an.system_prompt == "alice-news"
    assert bw is not None and bw.system_prompt == "bob-weather"


def test_lists_and_allowed_tools_roundtrip(store: AgentConfigStore) -> None:
    config = AgentConfig(
        system_prompt="p",
        model="m",
        mcp_servers=[{"name": "telos", "url": "https://example/mcp", "auth_token": "t"}],
        subagents=[{"name": "spec", "description": "d", "prompt": "p", "tools": ["search"]}],
        allowed_tools=["a", "b"],
    )
    store.put("u", "s", config)
    fetched = store.get("u", "s")
    assert fetched is not None
    assert fetched.mcp_servers == [{"name": "telos", "url": "https://example/mcp", "auth_token": "t"}]
    assert fetched.subagents[0]["name"] == "spec"
    assert fetched.allowed_tools == ["a", "b"]


def test_allowed_tools_none_distinct_from_empty(store: AgentConfigStore) -> None:
    # ``None`` means "use SDK defaults"; ``[]`` means "no tools allowed".
    # The roundtrip must preserve the distinction.
    store.put("u", "none", AgentConfig(allowed_tools=None))
    store.put("u", "empty", AgentConfig(allowed_tools=[]))
    none_cfg = store.get("u", "none")
    empty_cfg = store.get("u", "empty")
    assert none_cfg is not None and none_cfg.allowed_tools is None
    assert empty_cfg is not None and empty_cfg.allowed_tools == []
