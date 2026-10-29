"""agentmem CLI entry point."""

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path

_LOG_DIR = Path.home() / ".agentmem" / "logs"
_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def reconcile_main(argv: list[str]) -> int:
    """``agentmem reconcile`` — run the memory reconcile against a store file.

    Uses the same settings as the server: ``AGENTMEM_DB`` (or ``--db``) for
    the store, ``RECORDATIO_RECONCILE_MODEL`` / ``RECORDATIO_EXTRACTION_MODEL``
    for the judge. ``--dry-run`` writes nothing anywhere: the FTS index is
    built in memory and no snapshot/audit/cursor/op change is made.
    """
    parser = argparse.ArgumentParser(
        prog="agentmem reconcile",
        description="Reconcile and compact the memory store (delete superseded memories only).",
    )
    default_db = os.environ.get("AGENTMEM_DB") or str(Path.home() / ".agentmem" / "memories.db")
    parser.add_argument("--db", type=Path, default=Path(default_db), help="memory store (default: $AGENTMEM_DB)")
    parser.add_argument("--user", action="append", dest="users", help="user id to reconcile (repeatable; default all)")
    parser.add_argument("--dry-run", action="store_true", help="report what would change; mutate nothing")
    parser.add_argument("--output", type=Path, help="write the JSON report here instead of stdout")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    from tpasini.recordatio.reconcile import ReconcileSettings, build_reconciler, make_judge, reconcile_log_path
    from tpasini.recordatio.storage.base import SQLiteStore
    from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore
    from tpasini.recordatio.storage.reconcile_log import ReconcileLog

    if not args.db.exists():
        parser.error(f"no memory store at {args.db}")
    settings = ReconcileSettings()
    try:
        judge = make_judge(settings)
    except ValueError as exc:
        parser.error(str(exc))
    feedback_path = args.db.parent / "memory_feedback.db"
    log_path = reconcile_log_path(args.db)
    if args.dry_run:
        # Read whatever state exists, create nothing persistent.
        store = SQLiteStore(args.db, ephemeral_fts=True, ephemeral_emb=not Path(f"{args.db}.emb").exists())
        log = ReconcileLog(log_path if log_path.exists() else Path(":memory:"))
        feedback = MemoryFeedbackStore(feedback_path) if feedback_path.exists() else None
    else:
        store = SQLiteStore(args.db)
        log = ReconcileLog(log_path)
        feedback = MemoryFeedbackStore(feedback_path)
    reconciler = build_reconciler(store, judge=judge, settings=settings, feedback=feedback, log=log)
    report = asyncio.run(reconciler.run(args.users, dry_run=args.dry_run, details=True))
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(json.dumps(report["totals"], indent=2, sort_keys=True))
    else:
        print(text)
    return 0


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "reconcile":
        raise SystemExit(reconcile_main(sys.argv[2:]))

    parser = argparse.ArgumentParser(
        prog="agentmem",
        description="Long-term memory proxy for Anthropic API agents.",
        epilog="Subcommand: `agentmem reconcile --help` runs the memory reconcile/compaction job.",
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
