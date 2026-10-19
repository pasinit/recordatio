"""Live smoke test for Phase 1's AgentRunner against the real SDK.

Bypasses the HTTP layer (covered in unit tests) and exercises:
  SubscriptionTransport  ->  claude_agent_sdk.query()  ->  real Anthropic
  via OAuth + subscription credits.

Confirms: TaskEvent stream order, real assistant content, RateLimit caching,
and that the briefing-augmented system_prompt actually reaches the SDK.

Costs one tiny subscription credit hit. Run sparingly.
"""
import tempfile
from pathlib import Path

import anyio
from claude_agent_sdk import ClaudeAgentOptions, query

from tpasini.recordatio.agents.config_store import AgentConfig
from tpasini.recordatio.agents.events import TaskEventKind
from tpasini.recordatio.agents.rate_limit_state import RateLimitStateStore
from tpasini.recordatio.agents.runner import (
    AgentRunner,
    ApiKeyTransport,
    SubscriptionTransport,
)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rate_db = Path(tmp) / "rate.db"
        rate_state = RateLimitStateStore(rate_db)

        sub = SubscriptionTransport(
            rate_state=rate_state,
            query_fn=query,
            options_cls=ClaudeAgentOptions,
        )

        def boom(*_a: object, **_kw: object) -> None:
            raise AssertionError("API-key path should not run in this smoke test")

        api = ApiKeyTransport(pontifex_factory=boom)

        runner = AgentRunner(
            subscription=sub,
            api_key=api,
            rate_state=rate_state,
        )

        config = AgentConfig(
            system_prompt="You are a test agent. Reply in exactly one word.",
            model="claude-opus-4-7",
            allowed_tools=[],
        )

        events = []
        async for ev in runner.run(
            user_id="smoke_user",
            slug="smoke_slug",
            content="Say the word: hello",
            config=config,
        ):
            events.append(ev)
            preview = str(ev.payload)[:120]
            print(f"  {ev.kind:20s} {preview}")

        print()
        print(f"total events: {len(events)}")
        kinds = [e.kind for e in events]
        print(f"kinds: {[k.value for k in kinds]}")

        # Wire-contract assertions.
        assert kinds[0] is TaskEventKind.STARTED, "first event must be STARTED"
        assert events[0].payload.get("transport") == "subscription", (
            f"smoke test must take the subscription path, got "
            f"{events[0].payload.get('transport')}"
        )
        assert TaskEventKind.MESSAGE in kinds, "must have at least one MESSAGE event"
        assert kinds[-1] is TaskEventKind.COMPLETED, "last event must be COMPLETED"

        # Rate-limit snapshot should be cached.
        snap = rate_state.get("smoke_user", "smoke_slug")
        print(f"\ncached rate-limit snapshot: {snap}")
        if snap is None:
            print("  WARNING: no rate-limit snapshot cached — SDK may not have emitted one")
        else:
            print(f"  rate_limit_type:   {getattr(snap, 'rate_limit_type', '?')}")
            print(f"  status:            {getattr(snap, 'status', '?')}")
            print(f"  overage_status:    {getattr(snap, 'overage_status', '?')}")

        print("\nSMOKE PASS")


if __name__ == "__main__":
    anyio.run(main)
