"""Recordatio-specific environment for the shared Pontifex Codex client."""
import asyncio  # noqa: F401 -- shared subprocess module, retained for test injection
import os
from pathlib import Path

from pontifex.codex_client import CodexClient as SharedCodexClient
from pontifex.codex_client import CodexRPCError  # noqa: F401
from pontifex.codex_client import codex_environment as shared_environment


def codex_environment() -> dict[str, str]:
    return shared_environment(os.environ.get("RECORDATIO_CODEX_HOME", str(Path.home() / ".codex-memento")))


class CodexClient(SharedCodexClient):
    def __init__(self, *, cwd: str, timeout_s: float = 300, extra_env: dict[str, str] | None = None) -> None:
        super().__init__(cwd=cwd, timeout_s=timeout_s, home=codex_environment()["CODEX_HOME"],
                         binary=os.environ.get("RECORDATIO_CODEX_BIN", "codex"), extra_env=extra_env)
