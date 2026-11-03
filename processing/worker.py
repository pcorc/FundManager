"""Run one JSON ``RunRequest`` in its own process (cohorts, Fund objects and logging are process-global).

    python -m processing.worker request.json [--result result.json]

Each ``ProgressEvent`` is printed to stdout as one JSON line; anything else the
run prints goes to stderr with the logs. The result (exit code, summaries,
paths, files, errors) is written to ``--result``, default
``<request stem>.result.json`` next to the request. Exit code: the run's, or
2 when the request is invalid (including a cohort selection that resolves to no funds,
which is only known once ``run`` has loaded the cohorts).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, TextIO

from processing.api import InvalidRequest, ProgressEvent, RunResult, _logger, request_from_json, run, validate


def result_to_dict(result: RunResult) -> Dict[str, object]:
    """``result`` as plain JSON types (anything else, e.g. a date, becomes its ``str``)."""
    return json.loads(json.dumps({
        "exit_code": result.exit_code,
        "summaries": result.summaries,
        "paths": result.paths,
        "files": [asdict(f) for f in result.files],
        "errors": result.errors,
    }, default=str))


def _printer(stream: TextIO):
    def on_progress(event: ProgressEvent) -> None:
        stream.write(json.dumps(asdict(event)) + "\n")
        stream.flush()
    return on_progress


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one FundManager request from a JSON file.")
    parser.add_argument("request", help="request JSON (processing.api.request_to_json format)")
    parser.add_argument("--result", help="where to write the result JSON (default: <request>.result.json)")
    args = parser.parse_args(argv)
    request_path = Path(args.request)
    result_path = Path(args.result) if args.result else request_path.with_name(f"{request_path.stem}.result.json")

    events, sys.stdout = sys.stdout, sys.stderr  # keep stdout for the JSON-lines progress stream only
    result = RunResult()
    try:
        try:
            request = validate(request_from_json(request_path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            _logger("fund_manager.worker").error("Invalid request: %s", exc)
            result.exit_code = 2
            result.errors.append(f"Invalid request: {exc}")
        else:
            try:
                result = run(request, on_progress=_printer(events))
            except InvalidRequest as exc:
                _logger("fund_manager.worker").error("Invalid request: %s", exc)
                result.exit_code = 2
                result.errors.append(f"Invalid request: {exc}")
            except Exception as exc:
                _logger("fund_manager.worker").exception("Run failed: %s", exc)
                result.exit_code = 1
                result.errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        sys.stdout = events
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result_to_dict(result), indent=1), encoding="utf-8")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
