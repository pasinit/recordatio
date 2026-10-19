import asyncio
import tempfile
from pathlib import Path

import uvicorn
from mcp.server.fastmcp import FastMCP

from tpasini.recordatio.agents.codex_transport import CodexSubscriptionTransport
from tpasini.recordatio.agents.config_store import AgentConfig
from tpasini.recordatio.agents.events import TaskEventKind
from tpasini.recordatio.agents.rate_limit_state import RateLimitStateStore


async def main() -> None:
    mcp = FastMCP("Codex migration probe")
    called = []

    @mcp.tool()
    async def echo_probe(value: str) -> str:
        """Echo the supplied probe value."""
        called.append(value)
        return value

    server = uvicorn.Server(uvicorn.Config(mcp.streamable_http_app(), host="127.0.0.1", port=18787, log_level="error"))
    running = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(15):
            while not server.started:
                if running.done():
                    await running
                    raise RuntimeError("MCP server exited before startup")
                await asyncio.sleep(0.05)
        with tempfile.TemporaryDirectory(prefix="codex-mcp-smoke-") as directory:
            state = RateLimitStateStore(Path(directory) / "rate.db")
            transport = CodexSubscriptionTransport(rate_state=state, timeout_s=180)
            config = AgentConfig(model="", allowed_tools=["mcp__probe__echo_probe"], mcp_servers=[
                {"name": "probe", "type": "http", "url": "http://127.0.0.1:18787/mcp"}])
            events = [event async for event in transport.run(user_id="smoke", slug="probe", config=config,
                system_prompt="Call echo_probe exactly once with value CODEX_MCP_OK, then reply with that value.",
                content="Run the MCP smoke check.")]
            errors = [e.payload for e in events if e.kind is TaskEventKind.ERROR]
            if errors:
                raise RuntimeError(str(errors))
            if called != ["CODEX_MCP_OK"] or events[-1].kind is not TaskEventKind.COMPLETED:
                raise RuntimeError("MCP tool or terminal event missing")
            complete = events[-1].payload
            assert complete["billing_mode"] == "subscription" and complete["cost_usd"] == 0
            assert any(e.kind is TaskEventKind.TOOL_CALL for e in events)
            assert any(e.kind is TaskEventKind.TOOL_RESULT for e in events)
            print(f"Codex MCP execution and SSE events verified (model={complete['model']}, billing=subscription).")
            state.close()
    finally:
        server.should_exit = True
        await running


asyncio.run(main())
