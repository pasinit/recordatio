"""Read-only access for callers outside the local deployment.

Recordatio's HTTP API treats any presented API key as a pass-through
credential, which is safe only while every caller is local (loopback or the
Compose network). This guard lets a remote host search memories and nothing
else:

* A **recall key** (configured by SHA-256 hash, never in plain text) may call
  only ``GET /v1/users/{user_id}/memory-search``, from anywhere.
* A request from a **restricted network** (CIDRs, e.g. a private VPC) must
  present a recall key; anything else from there is refused.

Requests that are neither from a restricted network nor carry a recall key
are untouched, so local callers keep their current behaviour.
"""

import hashlib
import hmac
import ipaddress
import re
from collections.abc import Awaitable, Callable, Iterable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

RECALL_PATH = re.compile(r"^/v1/users/[^/]+/memory-search$")

Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def parse_networks(cidrs: Iterable[str]) -> list[Network]:
    """Parse CIDR strings; raises ``ValueError`` on a malformed entry."""
    return [ipaddress.ip_network(cidr.strip(), strict=False) for cidr in cidrs if cidr.strip()]


def is_recall_key(api_key: str | None, key_hashes: Iterable[str]) -> bool:
    """True when ``api_key`` hashes (SHA-256, hex) to one of ``key_hashes``."""
    if not api_key:
        return False
    digest = hashlib.sha256(api_key.encode()).hexdigest()
    return any(hmac.compare_digest(digest, expected.strip().lower()) for expected in key_hashes)


def decide(client_host: str | None, method: str, path: str, api_key: str | None,
           networks: list[Network], key_hashes: Iterable[str]) -> int | None:
    """Return an HTTP status to refuse the request with, or ``None`` to let it through.

    ``401``: a restricted-network caller without a recall key.
    ``403``: a recall key (or restricted caller) asking for anything but memory search.
    """
    recall = is_recall_key(api_key, key_hashes)
    restricted = False
    if client_host and networks:
        try:
            address = ipaddress.ip_address(client_host)
        except ValueError:
            restricted = True  # unparseable peer while restrictions are on: fail closed
        else:
            restricted = any(address in network for network in networks)
    if not (recall or restricted):
        return None
    if restricted and not recall:
        return 401
    if method != "GET" or not RECALL_PATH.match(path):
        return 403
    return None


def _api_key(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth[7:]
    return key or None


def install(app: FastAPI, restricted_networks: Iterable[str], recall_key_hashes: Iterable[str]) -> None:
    """Add the guard as HTTP middleware. A no-op when both lists are empty."""
    networks = parse_networks(restricted_networks)
    hashes = [h for h in recall_key_hashes if h.strip()]
    if not networks and not hashes:
        return

    @app.middleware("http")
    async def network_guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        client = request.client.host if request.client else None
        status = decide(client, request.method, request.url.path, _api_key(request), networks, hashes)
        if status is not None:
            error = "unauthorized" if status == 401 else "forbidden: this key may only search memories"
            return JSONResponse({"error": error}, status_code=status)
        return await call_next(request)
