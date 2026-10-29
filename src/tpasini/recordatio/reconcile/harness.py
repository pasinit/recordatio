"""Local judge harness: drive the reconcile service with a judge command.

The production judge is the nightly Cardea routine. This harness runs the
same loop locally — candidates → judge → apply, then finish — with any
command that reads a prompt on stdin and prints the verdicts, e.g.
``claude -p``. It exists to dry-run the routine's judging rules against a
copy of a store before deploy, with the real verification code.

The prompt is the routine's judging rules (a file) followed by the batch of
pairs; the command must answer with a JSON array of
``{"pair_token": ..., "verdict": {...}}`` objects.
"""
import json
import logging
import shlex
import subprocess
from collections import Counter
from collections.abc import Callable
from typing import Any

from tpasini.recordatio.reconcile.judge import extract_json
from tpasini.recordatio.reconcile.service import ReconcileError, ReconcileService

logger = logging.getLogger(__name__)

type JudgeFn = Callable[[str], str]

REPLY_FORMAT = """\
## Reply format

Judge every pair above. Reply with ONLY a JSON array, one object per pair:
[{"pair_token": "<the pair's pair_token, copied exactly>", "verdict": {"relation": "...", "keep": "A"|"B"|null, \
"drop": "A"|"B"|null, "facts": [{"dropped_fact": "...", "kept_span": "...", "status": "same"|"replaced"}], \
"evidence": "one sentence"}}]
"""


def command_judge(command: str, *, timeout_s: float = 600.0) -> JudgeFn:
    """A judge that pipes the prompt to ``command`` and returns its stdout.

    Raises (when called):
        RuntimeError: the command failed or timed out.
    """
    argv = shlex.split(command)

    def judge(prompt: str) -> str:
        try:
            done = subprocess.run(argv, input=prompt, capture_output=True, text=True, timeout=timeout_s, check=False)  # noqa: S603
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"judge command timed out after {timeout_s}s") from exc
        if done.returncode != 0:
            raise RuntimeError(f"judge command exited {done.returncode}: {done.stderr.strip()[:500]}")
        return done.stdout

    return judge


def build_prompt(rules: str, pairs: list[dict[str, Any]]) -> str:
    return f"{rules.strip()}\n\n## Pairs\n\n{json.dumps(pairs, indent=1, ensure_ascii=False)}\n\n{REPLY_FORMAT}"


def run_harness(
    service: ReconcileService,
    users: list[str],
    *,
    judge: JudgeFn,
    rules: str,
    run_id: str,
    batch: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Loop each user until no candidates are left (or nothing progresses),
    then finish. Returns per-user outcomes and the finish summaries."""
    report: dict[str, Any] = {"run_id": run_id, "dry_run": dry_run, "users": {}}
    for user in users:
        outcomes: list[dict[str, Any]] = []
        errors: list[str] = []
        while True:
            batch_out = service.candidates(user, run_id, limit=batch)
            pairs = batch_out["pairs"]
            if not pairs:
                break
            try:
                reply = extract_json(judge(build_prompt(rules, pairs)))
            except (RuntimeError, ValueError) as exc:
                errors.append(f"judge: {exc}")
                break  # keep both for every pair left; the next run retries
            verdicts = {
                str(item.get("pair_token")): item.get("verdict")
                for item in (reply if isinstance(reply, list) else [])
                if isinstance(item, dict)
            }
            progressed = False
            for pair in pairs:
                token = pair["pair_token"]
                if token not in verdicts:
                    errors.append(f"no verdict for {token}")
                    continue
                try:
                    result = service.apply(user, run_id, token, verdicts[token], dry_run=dry_run)
                except ReconcileError as exc:
                    errors.append(f"{token}: {exc}")
                    continue
                progressed = True
                outcomes.append({**result, "a": pair["a"], "b": pair["b"]})
            if not progressed:
                break
        finish = service.finish(user, run_id, dry_run=dry_run)
        report["users"][user] = {
            "outcomes": dict(Counter(o["outcome"] for o in outcomes)),
            "verdicts": outcomes,
            "errors": errors,
            "finish": finish,
        }
    return report


__all__ = ["REPLY_FORMAT", "build_prompt", "command_judge", "run_harness"]
