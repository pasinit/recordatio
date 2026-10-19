"""agentmem CLI entry point."""

import argparse
import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path.home() / ".agentmem" / "logs"
_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agentmem",
        description="Long-term memory proxy for Anthropic API agents.",
    )
    parser.add_argument("--port", type=int, default=8787, help="Port to listen on (default: 8787)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=_LOG_DIR / "agentmem.log",
        when="midnight",
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))

    logging.basicConfig(
        level=args.log_level.upper(),
        format=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
        handlers=[logging.StreamHandler(), file_handler],
    )

    import uvicorn

    from tpasini.recordatio.proxy.server import app

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
