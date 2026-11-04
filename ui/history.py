"""Read past runs back from their run folders: the history table, results, zips and re-run requests.

Everything here tolerates partial or corrupt folders (a run that is still
starting, a UI that died mid-run, a hand-edited JSON file): a folder with
neither ``meta.json`` nor ``request.json`` is not a run and is skipped, and a
missing or unreadable file just leaves its fields empty. Report paths from
``result.json`` are only trusted when they exist inside the run folder.
Error texts are passed through :func:`ui.runner.redact` (they can quote the
DB connection string). The zip of a run is built once, under a per-run lock
(several sessions often render the same just-finished run at once).
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from processing.api import ReportFile, RunRequest, RunResult, bundle, request_from_json
from ui import runner


@dataclass(frozen=True)
class RunRecord:
    """One run folder as the history page shows it."""

    run_dir: Path
    user: str = "?"
    started: Optional[str] = None
    finished: Optional[str] = None
    summary: str = ""
    as_of: str = ""  # the request's date or range, for browsing reports by date
    exit_code: Optional[int] = None
    status: str = "incomplete"  # running, ok, ok with errors, failed, invalid, cancelled, timed out, incomplete
    files: int = 0  # distinct report files
    errors: List[str] = field(default_factory=list)  # redacted
    published: int = 0  # publishes that copied files
    report_days: Tuple[str, ...] = ()  # the distinct ReportFile.day values ("stacked" for a stacked workbook)

    @property
    def name(self) -> str:
        return self.run_dir.name


def list_runs(root: Optional[Path] = None, limit: Optional[int] = None) -> List[RunRecord]:
    """Every run under ``root`` (default: the UI output root), newest first."""
    root = Path(root or runner.output_root())
    try:
        folders = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return []
    active = runner.active_run(root)
    records = []
    for folder in sorted(folders, key=lambda p: p.name, reverse=True):  # names start with YYYYMMDD-HHMMSS
        record = load_record(folder, active=active)
        if record is not None:
            records.append(record)
            if limit and len(records) >= limit:
                break
    return records


def load_record(run_dir: Path, active: Optional[Path] = None) -> Optional[RunRecord]:
    """The :class:`RunRecord` for ``run_dir``, or ``None`` if it is not a run folder."""
    run_dir = Path(run_dir)
    meta = runner.read_json(run_dir / runner.META)
    if meta is None and not (run_dir / runner.REQUEST).is_file():
        return None
    meta = meta or {}
    result = runner.read_json(run_dir / runner.RESULT) or {}
    request = _try_request(run_dir)
    exit_code = meta.get("exit_code", result.get("exit_code"))
    errors = [runner.redact(e) for e in result.get("errors") or [] if e]
    if meta.get("error"):  # e.g. the worker could not be launched, so there is no result.json
        errors.insert(0, runner.redact(str(meta["error"])))
    files = report_files(load_result(run_dir)) if result else []
    if request is not None:
        as_of = request.start_date.isoformat()
        if request.end_date and request.end_date != request.start_date:
            as_of += f"..{request.end_date.isoformat()}"
    else:
        as_of = ""
    return RunRecord(
        run_dir=run_dir,
        user=str(meta.get("user") or "?"),
        started=meta.get("started"),
        finished=meta.get("finished"),
        summary=str(meta.get("summary") or (runner.summarize(request) if request else "")),
        as_of=as_of,
        exit_code=exit_code if isinstance(exit_code, int) else None,
        status=_status(run_dir, meta, exit_code, errors, active),
        files=len(files),
        errors=errors,
        published=len(meta.get("published") or []),
        report_days=tuple(dict.fromkeys(f.day for f in files)),
    )


def _status(run_dir: Path, meta: dict, exit_code: object, errors: List[str], active: Optional[Path]) -> str:
    if not meta.get("finished"):
        return "running" if active is not None and Path(active) == run_dir else "incomplete"
    if meta.get("cancelled"):
        return "cancelled"
    if meta.get("timed_out"):
        return "timed out"
    if exit_code == 0:
        return "ok with errors" if errors else "ok"
    return "invalid" if exit_code == 2 else "failed"


def _try_request(run_dir: Path) -> Optional[RunRequest]:
    try:
        return load_request(run_dir)
    except (OSError, ValueError, TypeError):
        return None


def load_request(run_dir: Path) -> RunRequest:
    """The request a run was started with (for Re-run). Raises ``OSError``/``ValueError`` if unreadable."""
    return request_from_json((Path(run_dir) / runner.REQUEST).read_text(encoding="utf-8"))


def load_result(run_dir: Path) -> RunResult:
    """``result.json`` as a ``RunResult``; ``files`` keeps only existing files inside ``run_dir``."""
    run_dir = Path(run_dir)
    data = runner.read_json(run_dir / runner.RESULT) or {}
    base = os.path.normcase(os.path.abspath(runner.plain_path(str(run_dir)))) + os.sep
    files = []
    for item in data.get("files") or []:
        try:
            report = ReportFile(day=str(item["day"]), label=str(item["label"]), path=str(item["path"]),
                                kind=str(item.get("kind") or "excel"))
            inside = os.path.normcase(os.path.abspath(runner.plain_path(report.path))).startswith(base)
        except (KeyError, TypeError, OSError, ValueError):
            continue
        if inside and os.path.isfile(report.path):  # report.path may be in extended-length form (long paths)
            files.append(report)
    exit_code = data.get("exit_code")
    return RunResult(
        exit_code=exit_code if isinstance(exit_code, int) else 1,
        summaries={str(k): v for k, v in (data.get("summaries") or {}).items() if isinstance(v, dict)},
        paths={str(k): v for k, v in (data.get("paths") or {}).items() if isinstance(v, dict)},
        files=files,
        errors=[str(e) for e in data.get("errors") or [] if e],
    )


_LABEL_WORDS = {"nav": "NAV", "eod": "EOD", "pdf": "PDF", "excel": "Excel"}
_KINDS = ("excel", "pdf")


def readable_label(label: str) -> str:
    """``holdings_reconciliation_excel`` -> ``Holdings reconciliation``; the file kind is shown separately."""
    words = [w for w in str(label).split("_") if w]
    if len(words) > 1 and words[-1] in _KINDS:
        words = words[:-1]
    text = " ".join(_LABEL_WORDS.get(w, w) for w in words) or str(label)
    return text[:1].upper() + text[1:]


def report_files(result: RunResult) -> List[ReportFile]:
    """``result.files`` with each file once (labels of a shared workbook joined) and readable labels."""
    by_path: Dict[str, ReportFile] = {}
    labels: Dict[str, List[str]] = {}
    for report in result.files:
        key = os.path.normcase(runner.plain_path(report.path))
        by_path.setdefault(key, report)
        label = readable_label(report.label)
        if label not in labels.setdefault(key, []):
            labels[key].append(label)
    return [ReportFile(day=r.day, label=" + ".join(labels[k]), path=r.path, kind=r.kind) for k, r in by_path.items()]


_ZIP_LOCKS: Dict[str, threading.Lock] = {}
_ZIP_LOCKS_GUARD = threading.Lock()


def _zip_lock(run_dir: Path) -> threading.Lock:
    with _ZIP_LOCKS_GUARD:
        return _ZIP_LOCKS.setdefault(os.path.normcase(str(run_dir)), threading.Lock())


def zip_reports(run_dir: Path) -> Optional[Path]:
    """``<run_dir>/<run name>.zip`` of every report (built once, rebuilt if result.json is newer); ``None`` if none."""
    run_dir = Path(run_dir)
    result = load_result(run_dir)
    if not result.files:
        return None
    zip_path = Path(runner.long_path(run_dir / f"{run_dir.name}.zip", reserve=40))  # leaves room for the temp name
    with _zip_lock(run_dir):  # sessions of this process build it once; other processes use their own temp file
        try:
            fresh = zip_path.stat().st_mtime >= (run_dir / runner.RESULT).stat().st_mtime
        except OSError:
            fresh = False
        if not fresh:
            tmp = zip_path.with_name(f"{zip_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                bundle(result, tmp)
                runner.retry(os.replace, tmp, zip_path)
            finally:
                if tmp.exists():
                    tmp.unlink()
    return zip_path


def zip_bytes(run_dir: Path) -> bytes:
    """The zip of ``run_dir``'s reports (built on demand), for a deferred download button."""
    zip_path = zip_reports(run_dir)
    if zip_path is None:
        raise FileNotFoundError(f"{Path(run_dir).name} has no reports")
    with _zip_lock(Path(run_dir)):
        return runner.retry(zip_path.read_bytes)


VIEWS_SCHEMA = 1  # reporting.result_views schema_version this UI reads


def load_views(run_dir: Path) -> Optional[dict]:
    """``views.json`` (the per-day display tables, schema in ``reporting.result_views``), or ``None``.

    ``None`` when the file is missing (an older run, or a worker that died early), unreadable or of an
    unknown schema version, so the results view falls back to the downloads.
    """
    data = runner.read_json(Path(run_dir) / runner.VIEWS)
    if not data or data.get("schema_version") != VIEWS_SCHEMA or not isinstance(data.get("days"), dict):
        return None
    days = {str(k): _redact_day(v) for k, v in data["days"].items() if isinstance(v, dict)}
    return {"schema_version": VIEWS_SCHEMA, "days": dict(sorted(days.items()))}


def _redact_day(day: dict) -> dict:
    """The day's free texts that can carry exception messages, redacted in place (``fund_errors``, reasons, notes).

    A fund's error is the text of whatever exception stopped it (a DB error quotes its URL), and a rule's
    ERROR reason is its exception text.
    """
    errors = day.get("fund_errors")
    if isinstance(errors, dict):
        day["fund_errors"] = {fund: [runner.redact(m) for m in messages or [] if m] if isinstance(messages, list)
                              else [runner.redact(messages)] for fund, messages in errors.items()}
    for side in ("compliance", "compliance_ex_ante"):
        for rules in (day.get(side) or {}).values() if isinstance(day.get(side), dict) else ():
            for view in (rules or {}).values() if isinstance(rules, dict) else ():
                if isinstance(view, dict):
                    _redact_view(view)
    return day


def _redact_view(view: dict) -> None:
    if view.get("reason"):
        view["reason"] = runner.redact(view["reason"])
    for name in ("rows", "details"):
        for row in view.get(name) or []:
            if isinstance(row, dict) and row.get("reason"):
                row["reason"] = runner.redact(row["reason"])
    if isinstance(view.get("notes"), list):
        view["notes"] = [runner.redact(n) for n in view["notes"]]


def _seconds_between(start: Optional[str], end: Optional[str]) -> Optional[float]:
    try:
        return (datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))).total_seconds()
    except (TypeError, ValueError):
        return None


def fund_day_seconds(records: List[RunRecord]) -> List[float]:
    """Seconds per fund-day of the successful runs in ``records`` (for the New run time estimate).

    A run's fund-days are the ``total_funds`` of its per-day summaries (a trading day counts its ex-ante
    and ex-post passes); runs without a duration or a fund count are left out.
    """
    rates = []
    for record in records:
        if record.status not in ("ok", "ok with errors"):
            continue
        seconds = _seconds_between(record.started, record.finished)
        summaries = (runner.read_json(record.run_dir / runner.RESULT) or {}).get("summaries") or {}
        fund_days = sum(s.get("total_funds") or s.get("processed_funds") or 0
                        for s in summaries.values() if isinstance(s, dict))
        if seconds and seconds > 0 and isinstance(fund_days, int) and fund_days > 0:
            rates.append(seconds / fund_days)
    return rates
