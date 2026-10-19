"""One small ChatGPT subscription completion using Recordatio's service login."""

import asyncio

from tpasini.recordatio.agents.codex_transport import complete_text


def main() -> None:
    response = asyncio.run(complete_text("Reply with exactly RECORDATIO_CODEX_OK.", timeout_s=90))
    if response.strip() != "RECORDATIO_CODEX_OK":
        raise RuntimeError(f"Unexpected smoke-test response: {response!r}")
    print("ChatGPT subscription smoke test passed.")


if __name__ == "__main__":
    main()
