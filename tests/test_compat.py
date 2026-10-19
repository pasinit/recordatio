"""Legacy Memento names (env vars, headers, body flags) keep working after the rename."""

from tpasini.recordatio.compat import alias_legacy_env, header, pop_flag


def test_legacy_env_is_aliased_without_overriding_new_names():
    env = {"MEMENTO_AGENT_TIMEOUT_S": "900", "MEMENTO_WARM_POOL_MAX": "2", "RECORDATIO_WARM_POOL_MAX": "8"}
    alias_legacy_env(env)
    assert env["RECORDATIO_AGENT_TIMEOUT_S"] == "900"
    assert env["RECORDATIO_WARM_POOL_MAX"] == "8"


def test_header_prefers_new_name_and_falls_back_to_legacy():
    assert header({"x-recordatio-api-key": "new", "x-memento-api-key": "old"}, "api-key") == "new"
    assert header({"x-memento-api-key": "old"}, "api-key") == "old"
    assert header({}, "api-key") is None
    assert header({}, "source-agent-id", "") == ""


def test_pop_flag_removes_both_keys_and_prefers_new_name():
    body = {"x-recordatio-no-briefing": True, "x-memento-no-briefing": False, "model": "m"}
    assert pop_flag(body, "no-briefing", False) is True
    assert body == {"model": "m"}


def test_pop_flag_accepts_legacy_key_alone():
    body = {"x-memento-extract-from": {"turn": 1}}
    assert pop_flag(body, "extract-from") == {"turn": 1}
    assert body == {}
    assert pop_flag({}, "skip-input-extraction", False) is False
