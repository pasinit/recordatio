"""User-level identity store: helpers, ensure API, reserved slug, isolation.

Implementer gates I-12 (composed-read isolation), I-13 (reserved slug),
I-15 (ensure idempotence + convergence with ensure-on-write) from
IDENTITY_DESIGN.md §9.1.
"""

from datetime import UTC, datetime
from pathlib import Path

from tpasini.recordatio.entity import Entity
from tpasini.recordatio.identity import (
    add_principal_alias,
    ensure_mirror_entity,
    ensure_principal,
    get_principal,
    mirror_entity_id,
    principal_canonical_key,
    user_scope,
    visibility,
)
from tpasini.recordatio.storage.base import SQLiteStore

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


def _store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "identity.db")


# --- scope grammar (S-RULE) ---------------------------------------------------


def test_user_scope_grammar() -> None:
    assert user_scope("tommaso/scout") == "tommaso/_user"
    assert user_scope("tommaso/_maintenance") == "tommaso/_user"  # underscore slug survives
    assert user_scope("tommaso/") == "tommaso/_user"  # empty slug survives
    assert user_scope("alice") == "alice/_user"  # bare scope: whole id is the user segment
    assert user_scope("tommaso/_user") == "tommaso/_user"  # fixed point


def test_visibility_is_scope_plus_user_scope_and_nothing_else() -> None:
    assert visibility("tommaso/scout") == ("tommaso/scout", "tommaso/_user")
    assert visibility("tommaso/_user") == ("tommaso/_user",)
    # I-12: never another user's scope, never another agent's private scope.
    assert "sam/_user" not in visibility("tommaso/scout")
    assert "tommaso/helper" not in visibility("tommaso/scout")
    assert "tommaso/_user" not in visibility("sam/scout")


def test_mirror_id_is_deterministic_and_label_normalized() -> None:
    a = mirror_entity_id("tommaso/_user", "person", "Sam Example")
    b = mirror_entity_id("tommaso/_user", "person", "sam  example")
    assert a == b  # cross-agent dedup by construction
    assert a != mirror_entity_id("tommaso/_user", "organization", "Sam Example")
    assert a != mirror_entity_id("sam/_user", "person", "Sam Example")


# --- ensure principal (I-15) ---------------------------------------------------


def test_ensure_principal_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    principal, created = ensure_principal(store, "tommaso", display_name="Tommaso", now=NOW)
    assert created
    assert principal.type == "person"
    assert principal.canonical_key == principal_canonical_key("tommaso/_user")
    assert principal.canonical_label == "Tommaso"

    again, created_again = ensure_principal(store, "tommaso", display_name="Tommaso", now=NOW)
    assert not created_again
    assert again.id == principal.id
    # display_name rides the alias path on every call, deduplicated.
    aliases = [a.alias for a in store.list_entity_aliases(principal.id, "tommaso/_user")]
    assert aliases == ["Tommaso"]


def test_ensure_api_converges_with_ensure_on_write(tmp_path: Path) -> None:
    """Interleaved login-time ensure and extraction-time ensure produce ONE
    principal — both paths converge on the same canonical key."""
    store = _store(tmp_path)
    first, _ = ensure_principal(store, "tommaso", now=NOW)  # extraction-time minimal ensure
    second, created = ensure_principal(store, "tommaso", display_name="Tommaso", aliases=["T. Pasini"], now=NOW)
    assert not created
    assert second.id == first.id
    rows = store._conn.execute(
        "SELECT COUNT(*) FROM entities WHERE agent_id = ?", ("tommaso/_user",)
    ).fetchone()[0]
    assert rows == 1
    aliases = {a.alias for a in store.list_entity_aliases(first.id, "tommaso/_user")}
    assert aliases == {"Tommaso", "T. Pasini"}


def test_ensure_principal_minimal_body_no_aliases(tmp_path: Path) -> None:
    store = _store(tmp_path)
    principal, created = ensure_principal(store, "sam", now=NOW)
    assert created
    assert principal.label == "User"
    assert principal.canonical_label == "User"
    assert store.list_entity_aliases(principal.id, "sam/_user") == []


# --- standing aliases + collision guard (§2.1) ----------------------------------


def test_alias_collision_guard_blocks_third_party_label(tmp_path: Path) -> None:
    """The sam/coach case: a genuine third-party 'Tommaso' in the
    asserting agent's scope blocks the alias — it must not be swallowed."""
    store = _store(tmp_path)
    principal, _ = ensure_principal(store, "sam", now=NOW)
    store.add_entity(Entity(label="Tommaso", type="person", asserted_at=NOW), "sam/coach")

    added = add_principal_alias(
        store, principal.id, "Tommaso", "sam/_user", agent_scope="sam/coach", now=NOW
    )
    assert not added
    assert store.list_entity_aliases(principal.id, "sam/_user") == []

    # No conflict in a different scope of the same user: the alias lands.
    added = add_principal_alias(
        store, principal.id, "Sam", "sam/_user", agent_scope="sam/scout", now=NOW
    )
    assert added
    assert [a.alias for a in store.list_entity_aliases(principal.id, "sam/_user")] == ["Sam"]


def test_alias_collision_guard_checks_user_scope(tmp_path: Path) -> None:
    store = _store(tmp_path)
    principal, _ = ensure_principal(store, "tommaso", now=NOW)
    # A mirrored third party in the user scope also blocks.
    store.add_entity(Entity(label="Carol", type="person", asserted_at=NOW), "tommaso/_user")
    assert not add_principal_alias(store, principal.id, "Carol", "tommaso/_user", now=NOW)


def test_mirror_entity_reuses_existing_row_on_label_clash(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ensure_principal(store, "tommaso", now=NOW)
    source = Entity(label="Springfield", type="place", description="city", asserted_at=NOW)
    store.add_entity(source, "tommaso/scout")
    mid = ensure_mirror_entity(store, source, "tommaso/_user", source_agent_id="tommaso/scout")
    assert mid == mirror_entity_id("tommaso/_user", "place", "Springfield")
    # Second mirror call from another agent converges on the same row.
    source2 = Entity(label="Springfield", type="place", asserted_at=NOW)
    store.add_entity(source2, "tommaso/ledger")
    assert ensure_mirror_entity(store, source2, "tommaso/_user", source_agent_id="tommaso/ledger") == mid
    rows = store._conn.execute(
        "SELECT COUNT(*) FROM entities WHERE agent_id = ? AND LOWER(label) = 'springfield'", ("tommaso/_user",)
    ).fetchone()[0]
    assert rows == 1


# --- reserved slug (I-13) -------------------------------------------------------


def test_config_api_and_proxy_reject_reserved_slug(tmp_path, monkeypatch) -> None:
    import pytest
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("RECORDATIO_EXTRACTION_MODEL", "anthropic/claude-haiku-4-5-20251001")
    monkeypatch.setenv("AGENTMEM_DB", str(tmp_path / "memories.db"))
    from tpasini.recordatio.proxy import server as server_mod

    client = fastapi_testclient.TestClient(server_mod.app)
    headers = {"x-api-key": "test-key"}

    resp = client.put("/v1/agents/tommaso/_user/config", json={}, headers=headers)
    assert resp.status_code == 400
    assert "reserved" in resp.json()["error"]

    resp = client.post("/v1/agents/tommaso/_user/messages", json={"content": "hi"}, headers=headers)
    assert resp.status_code == 400

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        headers={**headers, "x-agent-id": "tommaso/_user"},
    )
    assert resp.status_code == 400
    assert "reserved" in resp.json()["error"]


def test_principal_endpoint_is_idempotent_ensure(tmp_path, monkeypatch) -> None:
    import pytest
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("RECORDATIO_EXTRACTION_MODEL", "anthropic/claude-haiku-4-5-20251001")
    monkeypatch.setenv("AGENTMEM_DB", str(tmp_path / "memories.db"))
    # TestClient drives the app from a portal thread; rebind the store's
    # connection thread-free (the sidecar stores do the same in production).
    import sqlite3

    from tpasini.recordatio.proxy import server as server_mod

    test_store = SQLiteStore(tmp_path / "principal.db")
    test_store._conn.close()
    conn = sqlite3.connect(str(tmp_path / "principal.db"), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    test_store._conn = conn
    test_store.fts_enabled = False
    monkeypatch.setattr(server_mod, "store", test_store)
    client = fastapi_testclient.TestClient(server_mod.app)
    headers = {"x-api-key": "test-key"}

    # Unauthenticated calls fail at the edge, like the config endpoints.
    assert client.put("/v1/users/tommaso/principal", json={}).status_code == 401

    resp = client.put(
        "/v1/users/tommaso/principal",
        json={"display_name": "Tommaso", "aliases": ["T. Pasini"]},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["created"] is True
    assert body["user_scope"] == "tommaso/_user"
    entity_id = body["entity_id"]

    # PUT with {} is the minimal ensure; repeat calls change nothing else.
    resp = client.put("/v1/users/tommaso/principal", json={}, headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"entity_id": entity_id, "user_scope": "tommaso/_user", "created": False}

    principal = get_principal(server_mod.store, "tommaso/_user")
    assert principal is not None and principal.id == entity_id
    aliases = {a.alias for a in server_mod.store.list_entity_aliases(entity_id, "tommaso/_user")}
    assert aliases == {"Tommaso", "T. Pasini"}

    assert client.put("/v1/users/tommaso/principal", json={"aliases": "nope"}, headers=headers).status_code == 400
