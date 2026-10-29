"""agentmem CLI entry point."""

import argparse
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
    """``agentmem reconcile`` — run the reconcile loop locally with a judge command.

    The nightly reconcile is judged by a Cardea routine; this runs the same
    candidates → verdicts → finish loop against a store file, piping each
    batch to ``--judge-cmd`` (e.g. ``claude -p``) with the routine's judging
    rules from ``--judge-prompt``. Dry run unless ``--apply``: the FTS index is
    built in memory, verdicts are verified but not recorded, and no snapshot,
    audit row, cursor or op change is written.
    """
    parser = argparse.ArgumentParser(
        prog="agentmem reconcile",
        description="Run the memory reconcile loop locally with a judge command (dry run unless --apply).",
    )
    default_db = os.environ.get("AGENTMEM_DB") or str(Path.home() / ".agentmem" / "memories.db")
    parser.add_argument("--db", type=Path, default=Path(default_db), help="memory store (default: $AGENTMEM_DB)")
    parser.add_argument("--user", action="append", dest="users", help="user id to reconcile (repeatable; default all)")
    parser.add_argument("--judge-cmd", required=True, help="command reading the prompt on stdin, e.g. 'claude -p'")
    parser.add_argument("--judge-prompt", type=Path, required=True, help="file with the routine's judging rules")
    parser.add_argument("--batch", type=int, default=10, help="pairs per judge call (default 10)")
    parser.add_argument("--judge-timeout", type=float, default=600.0, help="seconds per judge call (default 600)")
    parser.add_argument("--apply", action="store_true", help="delete for real (default: dry run)")
    parser.add_argument("--output", type=Path, help="write the JSON report here instead of stdout")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    from datetime import UTC, datetime

    from tpasini.recordatio.reconcile import ReconcileSettings, build_service, reconcile_log_path
    from tpasini.recordatio.reconcile.harness import command_judge, run_harness
    from tpasini.recordatio.storage.base import SQLiteStore
    from tpasini.recordatio.storage.memory_feedback import MemoryFeedbackStore
    from tpasini.recordatio.storage.reconcile_log import ReconcileLog

    if not args.db.exists():
        parser.error(f"no memory store at {args.db}")
    rules = args.judge_prompt.read_text(encoding="utf-8")
    feedback_path = args.db.parent / "memory_feedback.db"
    log_path = reconcile_log_path(args.db)
    if args.apply:
        store = SQLiteStore(args.db)
        log = ReconcileLog(log_path)
        feedback = MemoryFeedbackStore(feedback_path)
    else:
        # Read whatever state exists, create nothing persistent.
        store = SQLiteStore(args.db, ephemeral_fts=True, ephemeral_emb=not Path(f"{args.db}.emb").exists())
        log = ReconcileLog(log_path if log_path.exists() else Path(":memory:"))
        feedback = MemoryFeedbackStore(feedback_path) if feedback_path.exists() else None
    service = build_service(store, settings=ReconcileSettings(), feedback=feedback, log=log)
    report = run_harness(
        service,
        args.users or service.users(),
        judge=command_judge(args.judge_cmd, timeout_s=args.judge_timeout),
        rules=rules,
        run_id=f"cli-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        batch=args.batch,
        dry_run=not args.apply,
    )
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(json.dumps({u: r["outcomes"] for u, r in report["users"].items()}, indent=2, sort_keys=True))
    else:
        print(text)
    return 0


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "reconcile":
        raise SystemExit(reconcile_main(sys.argv[2:]))

    parser = argparse.ArgumentParser(
        prog="agentmem",
        description="Long-term memory proxy for Anthropic API agents.",
        epilog="Subcommand: `agentmem reconcile --help` runs the memory reconcile loop locally with a judge command.",
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
