"""Warm-spare session pool for the subscription transport.

The Claude Agent SDK pays a multi-second cold start on every turn:
subprocess spawn + CLI bootstrap, then the control-protocol ``initialize``
round-trip. ``ClaudeSDKClient.connect()`` performs the spawn and the
initialize upfront and kicks off the MCP server handshakes in the
background immediately after (verified against SDK 0.2.93: ``connect()``
blocks on initialize ~1-1.5s; MCP servers go ``pending`` → ``connected``
in the seconds that follow, with no message sent). A client connected
ahead of time therefore serves its first message with all of that
already paid — including MCP, as long as the spare sat in the pool for
a few seconds.

Why this is NOT a generic pool:

- Session config (system prompt, MCP servers with per-agent Bearer
  credentials, allowed_tools, subagents, model, cwd, env isolation) is
  sealed at ``connect()`` time. A spare is only valid for the exact
  agent + config it was built for, so spares are keyed by
  ``(user_id, slug, config.version)`` — a config update bumps the
  version. A session fingerprint additionally checks inline configuration
  changes that do not bump the version.
- One turn per session: Recordatio's statelessness is the product (the
  bounded context is rebuilt every turn). ``take()`` removes the spare
  from the pool permanently; the caller must ``disconnect()`` it after
  the turn and may NEVER send a second turn through it.
- Predictive warm-up: the transport schedules the next spare for a
  conversation right after serving a turn, so only the first turn after
  idle stays cold (acceptable by design).

Hygiene: a global cap on total spares (``RECORDATIO_WARM_POOL_MAX``,
default 8) with oldest-first eviction, a per-spare TTL
(``RECORDATIO_WARM_POOL_TTL_S``, default 600s) enforced both lazily at
``take()`` time and by a background reaper task that disconnects expired
clients, and a best-effort subprocess liveness check so a crashed CLI
falls back to a cold spawn transparently.
"""

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# (user_id, slug, config_version) — see module docstring for why the
# config version is part of the identity of a spare.
type PoolKey = tuple[str, str, int]

# Builds and connects one SDK client. Injected by the transport so the
# pool stays ignorant of ClaudeAgentOptions construction (and so tests
# can hand it fakes).
type ConnectFn = Callable[[], Awaitable[Any]]


def _client_alive(client: Any) -> bool:
    """Best-effort liveness probe for a pooled SDK client.

    Reaches into the SDK's subprocess transport (``client._transport.
    _process.returncode``) because the SDK exposes no public liveness
    API. Unknown shape ⇒ assume alive — the transport's send-time
    fallback (a failed ``query()`` write → cold spawn) is the authority;
    this probe just avoids handing out *known* corpses.
    """
    if hasattr(client, "alive"):
        return bool(client.alive)
    process = getattr(getattr(client, "_transport", None), "_process", None)
    return process is None or getattr(process, "returncode", None) is None


@dataclass
class _Spare:
    client: Any
    fingerprint: str | None
    expires_at: float  # monotonic-clock deadline
    ready_at: float = 0.0  # earliest monotonic instant the spare may be taken


class WarmPool:
    """Per-(user, slug, config-version) warm SDK sessions, one spare per key.

    All methods must be called from the event loop thread (the pool uses
    plain dict state guarded by the single-threaded asyncio model: no
    awaits between check and mutate).
    """

    def __init__(
        self,
        *,
        max_spares: int | None = None,
        ttl_s: float | None = None,
        min_age_s: float | None = None,
        reap_interval_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # Env defaults are read at construction (not import) time so
        # tests and operators can adjust without reloading the module.
        if max_spares is None:
            max_spares = int(os.environ.get("RECORDATIO_WARM_POOL_MAX", "8"))
        if ttl_s is None:
            ttl_s = float(os.environ.get("RECORDATIO_WARM_POOL_TTL_S", "600"))
        # connect() returns before the CLI's background MCP connect finishes
        # (~2s observed); a spare taken younger than this may serve a turn
        # with its tool servers still pending. Younger spares are left to
        # mature and the turn goes cold instead.
        if min_age_s is None:
            min_age_s = float(os.environ.get("RECORDATIO_WARM_POOL_MIN_AGE_S", "3"))
        self._min_age_s = min_age_s
        self._max = max_spares
        self._ttl_s = ttl_s
        self._reap_interval_s = (
            reap_interval_s if reap_interval_s is not None else min(ttl_s / 2, 60.0)
        )
        self._clock = clock
        self._spares: dict[PoolKey, _Spare] = {}
        self._warming: set[PoolKey] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._reaper: asyncio.Task[None] | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Take / warm
    # ------------------------------------------------------------------

    async def take(
        self, user_id: str, slug: str, config_version: int, *, fingerprint: str | None = None,
    ) -> Any | None:
        """Pop a live spare for the key, or ``None`` (caller goes cold).

        The spare is removed permanently — one turn per session. Spares
        for the same (user, slug) at a *different* config version are
        invalidated on sight: their sealed session no longer matches the
        agent, so keeping them around until TTL would only waste slots.
        """
        stale = [
            key
            for key in self._spares
            if key[0] == user_id and key[1] == slug and key[2] != config_version
        ]
        for key in stale:
            self._discard(key, reason="config version changed")

        key = (user_id, slug, config_version)
        spare = self._spares.get(key)
        if spare is None:
            return None
        if spare.fingerprint != fingerprint:
            self._discard(key, reason="session configuration changed")
            return None
        if self._clock() < spare.ready_at:
            # Too young — its background MCP connect may still be pending.
            # Leave it to mature for the next turn; this one goes cold.
            return None
        self._spares.pop(key, None)
        if self._clock() >= spare.expires_at or not _client_alive(spare.client):
            self._spawn(disconnect_quietly(spare.client))
            return None
        return spare.client

    def schedule_warm(
        self,
        user_id: str,
        slug: str,
        config_version: int,
        connect: ConnectFn,
        *,
        fingerprint: str | None = None,
    ) -> asyncio.Task[None] | None:
        """Spawn a background task that connects one spare for the key.

        No-ops when the pool is closed or disabled (``max_spares <= 0``),
        when a spare (or an in-flight warm-up) already exists for the
        key, or when the pool is at capacity and nothing is evictable.
        At capacity the oldest-expiring spare is evicted — the newest
        warm-up request marks the most recently active conversation.
        """
        if self._closed or self._max <= 0:
            return None
        key: PoolKey = (user_id, slug, config_version)
        spare = self._spares.get(key)
        if spare is not None and spare.fingerprint != fingerprint:
            self._discard(key, reason="session configuration changed")
        if key in self._spares or key in self._warming:
            return None
        # A version bump for the same (user, slug) supersedes old spares
        # outright — don't make them wait for take()/TTL.
        for stale in [
            k for k in self._spares if k[0] == user_id and k[1] == slug and k[2] != config_version
        ]:
            self._discard(stale, reason="config version changed")
        if len(self._spares) + len(self._warming) >= self._max and not self._evict_oldest():
            return None
        self._warming.add(key)
        task = self._spawn(self._warm(key, connect, fingerprint))
        self._ensure_reaper()
        return task

    async def _warm(self, key: PoolKey, connect: ConnectFn, fingerprint: str | None) -> None:
        try:
            client = await connect()
        except Exception:
            logger.exception(
                "warm-spare connect failed for %s/%s@v%d", key[0], key[1], key[2]
            )
            return
        finally:
            self._warming.discard(key)
        if self._closed or key in self._spares or len(self._spares) >= self._max:
            # Lost the race (shutdown / duplicate / capacity) — never
            # leak the subprocess.
            await disconnect_quietly(client)
            return
        self._spares[key] = _Spare(
            client=client,
            fingerprint=fingerprint,
            expires_at=self._clock() + self._ttl_s,
            ready_at=self._clock() + self._min_age_s,
        )
        logger.info("warmed SDK spare for %s/%s@v%d", key[0], key[1], key[2])

    # ------------------------------------------------------------------
    # Hygiene: TTL reaper, eviction, shutdown
    # ------------------------------------------------------------------

    def reap(self) -> int:
        """Disconnect expired spares; returns how many were reaped."""
        now = self._clock()
        expired = [key for key, spare in self._spares.items() if now >= spare.expires_at]
        for key in expired:
            self._discard(key, reason="ttl expired")
        return len(expired)

    def _ensure_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self._reap_loop())

    async def _reap_loop(self) -> None:
        # Runs while there is anything to watch; exits when the pool goes
        # empty (restarted by the next schedule_warm) or on close.
        while not self._closed:
            await asyncio.sleep(self._reap_interval_s)
            self.reap()
            if not self._spares and not self._warming:
                return

    def _evict_oldest(self) -> bool:
        """Drop the oldest-expiring spare to make room. False if none."""
        if not self._spares:
            return False
        oldest = min(self._spares, key=lambda key: self._spares[key].expires_at)
        self._discard(oldest, reason="evicted at capacity")
        return True

    def _discard(self, key: PoolKey, *, reason: str) -> None:
        spare = self._spares.pop(key, None)
        if spare is None:
            return
        logger.info("discarding SDK spare for %s/%s@v%d (%s)", key[0], key[1], key[2], reason)
        self._spawn(disconnect_quietly(spare.client))

    async def close_all(self) -> None:
        """Disconnect every spare and stop background work. Idempotent.

        Wired into server shutdown so no CLI subprocess outlives the
        Recordatio process.
        """
        self._closed = True
        if self._reaper is not None:
            self._reaper.cancel()
            self._reaper = None
        for task in list(self._tasks):
            task.cancel()
        spares = list(self._spares.values())
        self._spares.clear()
        for spare in spares:
            await disconnect_quietly(spare.client)

    # ------------------------------------------------------------------
    # Introspection / test helpers
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._spares)

    def spare_keys(self) -> list[PoolKey]:
        return list(self._spares)

    async def wait_idle(self) -> None:
        """Await all in-flight pool tasks (warm-ups, disconnects)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn(self, coro: Awaitable[None]) -> asyncio.Task[None]:
        # asyncio.create_task matches the codebase idiom (see
        # server._background); the strong reference set keeps tasks from
        # being garbage-collected mid-flight.
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task


async def disconnect_quietly(client: Any) -> None:
    """Terminate a session's subprocess, swallowing (but logging) failures."""
    try:
        await client.disconnect()
    except Exception:
        logger.warning("SDK client disconnect failed", exc_info=True)
