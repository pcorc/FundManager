"""Run one JSON ``RunRequest`` in its own process (cohorts, Fund objects and logging are process-global).

    python -m processing.worker request.json [--result result.json] [--views views.json]

Each ``ProgressEvent`` is printed to stdout as one JSON line; anything else the
run prints goes to stderr with the logs. The result (exit code, summaries,
paths, files, errors) is written to ``--result``, default
``<request stem>.result.json`` next to the request. Exit code: the run's, or
2 when the request is invalid (including a cohort selection that resolves to no funds,
which is only known once ``run`` has loaded the cohorts).

The per-fund, per-rule display tables (``RunResult.views``) go to a separate file so
result.json stays small: ``--views``, default next to the result with its trailing
``result.json`` replaced by ``views.json`` (``result.json`` -> ``views.json``,
``job.result.json`` -> ``job.views.json``, any other name -> ``<stem>.views.json``).
It is always written, just before result.json, as
``{"schema_version": 1, "days": {"<YYYY-MM-DD>": DayView}}`` (``days`` is empty when
the run failed early or was invalid); the DayView schema is in ``reporting.result_views``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, TextIO

from processing.api import InvalidRequest, ProgressEvent, RunResult, _logger, request_from_json, run, validate
from reporting.result_views import SCHEMA_VERSION, to_jsonable

_RESULT_SUFFIX = "result.json"


def result_to_dict(result: RunResult) -> Dict[str, object]:
    """``result`` as plain JSON types (anything else, e.g. a date, becomes its ``str``)."""
    return json.loads(json.dumps({
        "exit_code": result.exit_code,
        "summaries": result.summaries,
        "paths": result.paths,
        "files": [asdict(f) for f in result.files],
        "errors": result.errors,
    }, default=str))


def views_to_dict(result: RunResult) -> Dict[str, object]:
    """The views.json document for ``result``."""
    return {"schema_version": SCHEMA_VERSION, "days": to_jsonable(result.views)}


def default_views_path(result_path: Path) -> Path:
    """views.json beside ``result_path`` (see the module docstring)."""
    name = result_path.name
    if name.lower().endswith(_RESULT_SUFFIX):
        return result_path.with_name(name[: -len(_RESULT_SUFFIX)] + "views.json")
    return result_path.with_name(f"{result_path.stem}.views.json")


def _printer(stream: TextIO):
    def on_progress(event: ProgressEvent) -> None:
        stream.write(json.dumps(asdict(event)) + "\n")
        stream.flush()
    return on_progress


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one FundManager request from a JSON file.")
    parser.add_argument("request", help="request JSON (processing.api.request_to_json format)")
    parser.add_argument("--result", help="where to write the result JSON (default: <request>.result.json)")
    parser.add_argument("--views", help="where to write the views JSON (default: beside the result, views.json)")
    args = parser.parse_args(argv)
    request_path = Path(args.request)
    result_path = Path(args.result) if args.result else request_path.with_name(f"{request_path.stem}.result.json")
    views_path = Path(args.views) if args.views else default_views_path(result_path)

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
    try:  # before result.json, so a reader that sees the result also finds its views
        views_path.parent.mkdir(parents=True, exist_ok=True)
        views_path.write_text(json.dumps(views_to_dict(result), default=str, separators=(",", ":")),
                              encoding="utf-8")
    except Exception as exc:  # the in-app tables are optional: never fail the run over them
        _logger("fund_manager.worker").warning("Could not write %s: %s", views_path, exc, exc_info=True)
        result.errors.append(f"Result views could not be written: {type(exc).__name__}: {exc}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result_to_dict(result), indent=1), encoding="utf-8")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
