"""Read-only remote access: recall keys and restricted networks (proxy/network_guard.py)."""

import hashlib

import httpx
import pytest
from fastapi import FastAPI

from tpasini.recordatio.proxy import network_guard

RECALL_KEY = "recall-key-for-tests"
RECALL_HASH = hashlib.sha256(RECALL_KEY.encode()).hexdigest()
VPC = ["10.1.0.0/20"]
SEARCH = "/v1/users/sam/memory-search"


def _decide(client: str | None, method: str, path: str, key: str | None,
            networks: tuple[str, ...] = tuple(VPC), hashes: tuple[str, ...] = (RECALL_HASH,)) -> int | None:
    return network_guard.decide(client, method, path, key, network_guard.parse_networks(networks), hashes)


def test_local_callers_are_untouched() -> None:
    assert _decide("172.18.0.5", "POST", "/v1/chat/completions", "any-key") is None
    assert _decide("127.0.0.1", "DELETE", "/v1/users/sam/units/u1", None) is None


def test_restricted_network_needs_a_recall_key() -> None:
    assert _decide("10.1.0.3", "GET", SEARCH, None) == 401
    assert _decide("10.1.0.3", "GET", SEARCH, "some-other-key") == 401
    assert _decide("10.1.0.3", "GET", SEARCH, RECALL_KEY) is None


def test_recall_key_may_only_search() -> None:
    for client in ("10.1.0.3", "172.18.0.5"):
        assert _decide(client, "POST", SEARCH, RECALL_KEY) == 403
        assert _decide(client, "GET", "/v1/users/sam/memory-day", RECALL_KEY) == 403
        assert _decide(client, "GET", "/v1/users/sam/memory-search/extra", RECALL_KEY) == 403
        assert _decide(client, "POST", "/v1/chat/completions", RECALL_KEY) == 403


def test_unparseable_peer_fails_closed_only_when_restrictions_are_on() -> None:
    assert _decide("not-an-ip", "GET", SEARCH, None) == 401
    assert _decide("not-an-ip", "GET", SEARCH, None, networks=(), hashes=()) is None


def test_install_is_a_noop_without_configuration() -> None:
    app = FastAPI()
    network_guard.install(app, [], [])
    assert app.user_middleware == []


def _app() -> FastAPI:
    app = FastAPI()
    network_guard.install(app, VPC, [RECALL_HASH])

    @app.get("/v1/users/{user_id}/memory-search")
    async def search(user_id: str) -> dict[str, str]:
        return {"user_id": user_id}

    @app.post("/v1/users/{user_id}/units")
    async def write(user_id: str) -> dict[str, str]:
        return {"written": user_id}

    return app


async def _status(client_host: str, method: str, path: str, key: str | None) -> int:
    transport = httpx.ASGITransport(app=_app(), client=(client_host, 40000))
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(transport=transport, base_url="http://recordatio") as client:
        response = await client.request(method, path, headers=headers)
    return response.status_code


@pytest.mark.parametrize(
    ("client_host", "method", "path", "key", "expected"),
    [
        ("10.1.0.3", "GET", SEARCH, RECALL_KEY, 200),
        ("10.1.0.3", "GET", SEARCH, None, 401),
        ("10.1.0.3", "POST", "/v1/users/sam/units", RECALL_KEY, 403),
        ("172.18.0.5", "POST", "/v1/users/sam/units", "telos-key", 200),
        ("172.18.0.5", "POST", "/v1/users/sam/units", RECALL_KEY, 403),
    ],
)
async def test_middleware_end_to_end(client_host: str, method: str, path: str, key: str | None, expected: int) -> None:
    assert await _status(client_host, method, path, key) == expected
