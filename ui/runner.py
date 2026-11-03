"""Start one ``processing.worker`` per run in its own folder and track it on disk.

Each run gets ``<output root>/<YYYYMMDD-HHMMSS>-<user>-<tag>/`` holding:

- ``request.json``  the request (its ``output_dir`` is always forced to ``reports/`` here)
- ``progress.jsonl`` the worker's stdout: one ``ProgressEvent`` JSON object per line
- ``run.log``       the worker's stderr (logging)
- ``result.json``   written by the worker (or by :func:`_finalize` when it died without one)
- ``meta.json``     who, when, request summary, pid, exit code, cancelled / timed-out flags
- ``reports/``      the report files; ``logs/`` (the worker's cwd-relative log files)

Only one run happens per output root at a time: ``<root>/.run.lock`` holds the
worker's PID and run folder, and a lock whose PID is gone is stale and gets
cleared. Because all state is on disk, any browser session (or a refreshed
one) can reattach to the active run through :func:`active_run`. A watcher
thread in the UI process waits for the worker, enforces the timeout and
writes the final ``meta.json``; :func:`cancel` terminates the worker.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config.run_configurations import FULL_COMPLIANCE_TESTS
from processing.api import RunRequest, request_to_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ui_runs"
WORKER_COMMAND: List[str] = [sys.executable, "-m", "processing.worker"]  # tests swap in a fake worker
LOCK_NAME = ".run.lock"
REQUEST, RESULT, LOG, META, PROGRESS, REPORTS = (
    "request.json", "result.json", "run.log", "meta.json", "progress.jsonl", "reports")
GRACE_SECONDS = 5.0  # between terminate() and kill()
DEFAULT_TIMEOUT_MINUTES = 120.0
_STALE_UNREADABLE_LOCK_SECONDS = 30.0  # a lock file that never became valid JSON

_PROCESSES: Dict[str, subprocess.Popen] = {}  # run folder -> worker started by this UI process
_GUARD = threading.RLock()  # serialises meta/lock updates between sessions and watcher threads


class RunLocked(RuntimeError):
    """Another run holds the output root's lock."""


# ---------------------------------------------------------------------------
# Settings (environment)
# ---------------------------------------------------------------------------

def output_root() -> Path:
    """``FUNDMANAGER_UI_OUTPUT_ROOT``, default ``<repo>/ui_runs`` (never the shared drive)."""
    return Path(os.getenv("FUNDMANAGER_UI_OUTPUT_ROOT") or DEFAULT_OUTPUT_ROOT).expanduser().resolve()


def publish_root() -> Optional[Path]:
    """``FUNDMANAGER_UI_PUBLISH_ROOT`` (e.g. the G: archive), or ``None``: publishing is then off."""
    value = os.getenv("FUNDMANAGER_UI_PUBLISH_ROOT", "").strip()
    return Path(value).expanduser() if value else None


def timeout_seconds() -> Optional[float]:
    """``FUNDMANAGER_UI_TIMEOUT_MINUTES`` (default 120; 0 or less = no timeout)."""
    try:
        minutes = float(os.getenv("FUNDMANAGER_UI_TIMEOUT_MINUTES") or DEFAULT_TIMEOUT_MINUTES)
    except ValueError:
        minutes = DEFAULT_TIMEOUT_MINUTES
    return minutes * 60 if minutes > 0 else None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def safe_name(text: Optional[str], default: str = "run", max_len: int = 40) -> str:
    """``text`` reduced to ``[A-Za-z0-9._-]`` for a folder or file name (``DOMAIN\\user`` -> ``DOMAIN_user``)."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "")).strip("._-")[:max_len].strip("._-")
    return cleaned or default


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Optional[dict]:
    """The JSON object in ``path``, or ``None`` if it is missing, unreadable or not an object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict) -> None:
    """Write atomically (temp file + replace) so readers never see half a file."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def update_meta(run_dir: Path, **fields: object) -> dict:
    """Merge ``fields`` into ``<run_dir>/meta.json`` and return the new contents."""
    with _GUARD:
        meta = read_json(Path(run_dir) / META) or {}
        meta.update(fields)
        _write_json(Path(run_dir) / META, meta)
        return meta


def summarize(request: RunRequest) -> str:
    """One line for history tables: operations | selection | dates | rules."""
    picks = list(request.cohorts) + list(request.funds)
    selection = ", ".join(picks) if picks else "all funds"
    if request.cohorts and request.funds:
        selection = f"{', '.join(request.cohorts)} ∩ {', '.join(request.funds)}"
    if request.exclude:
        selection += f" minus {', '.join(request.exclude)}"
    dates = request.start_date.isoformat()
    if request.end_date and request.end_date != request.start_date:
        dates += f"..{request.end_date.isoformat()} ({request.date_mode})"
    parts = [", ".join(request.operations), selection, dates]
    if "compliance" in request.operations or "trading_compliance" in request.operations:
        tests = set(request.compliance_tests)
        parts.append("all rules" if not tests or tests >= set(FULL_COMPLIANCE_TESTS) else f"{len(tests)} rule(s)")
    return " | ".join(parts)


def pid_alive(pid: Optional[int]) -> bool:
    """Whether a process with ``pid`` exists (Windows: OpenProcess + GetExitCodeProcess; never signals it)."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists, it just isn't ours
        try:
            code = wintypes.DWORD()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # POSIX only: signal 0 checks existence (on Windows os.kill would terminate it)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# Lock file
# ---------------------------------------------------------------------------

def read_lock(root: Path) -> Optional[dict]:
    """The lock's contents, ``{}`` if it exists but cannot be parsed, ``None`` if there is no lock."""
    path = Path(root) / LOCK_NAME
    if not path.exists():
        return None
    return read_json(path) or {}


def _lock_is_stale(root: Path, info: dict) -> bool:
    if not info:  # unreadable: stale once it has had time to be written
        try:
            return time.time() - (Path(root) / LOCK_NAME).stat().st_mtime > _STALE_UNREADABLE_LOCK_SECONDS
        except OSError:
            return True
    if str(info.get("run_dir")) in _PROCESSES:
        return False  # our watcher thread still owns it and will release it
    return not pid_alive(info.get("pid"))


def _acquire_lock(root: Path, run_dir: Path, user: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_NAME
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            info = read_lock(root)
            if info is None:
                continue  # released meanwhile
            if not _lock_is_stale(root, info):
                raise RunLocked(f"A run by {info.get('user', '?')} (started {info.get('started', '?')}) is in "
                                f"progress: {Path(str(info.get('run_dir', '?'))).name}")
            _clear_stale_lock(root, info)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:  # the launching UI process holds it until Popen
            json.dump({"pid": os.getpid(), "run_dir": str(run_dir), "user": user, "started": now_iso()}, handle)
        return
    raise RunLocked("Could not acquire the run lock; try again")


def _release_lock(root: Path, run_dir: Path) -> None:
    """Remove the lock if it still belongs to ``run_dir``."""
    with _GUARD:
        info = read_lock(root)
        if info is not None and str(info.get("run_dir")) == str(run_dir):
            try:
                (Path(root) / LOCK_NAME).unlink()
            except OSError:
                pass


def _clear_stale_lock(root: Path, info: dict) -> None:
    """Drop a dead run's lock and mark its folder finished if nobody did (e.g. the UI restarted mid-run)."""
    with _GUARD:
        run_dir = Path(str(info.get("run_dir"))) if info.get("run_dir") else None
        if run_dir is not None and run_dir.is_dir() and not (read_json(run_dir / META) or {}).get("finished"):
            _finalize(run_dir, None, timed_out=False, orphaned=True)
        try:
            (Path(root) / LOCK_NAME).unlink()
        except OSError:
            pass


def active_run(root: Optional[Path] = None) -> Optional[Path]:
    """The run folder of the run in progress under ``root``, or ``None`` (clearing a stale lock)."""
    root = Path(root or output_root())
    info = read_lock(root)
    if info is None:
        return None
    if not _lock_is_stale(root, info):
        return Path(str(info["run_dir"])) if info.get("run_dir") else None
    _clear_stale_lock(root, info)
    return None


# ---------------------------------------------------------------------------
# Starting, watching, cancelling
# ---------------------------------------------------------------------------

def _new_run_dir(root: Path, user: str, tag: Optional[str], now: Optional[datetime] = None) -> Path:
    stem = f"{(now or datetime.now()):%Y%m%d-%H%M%S}-{safe_name(user, 'user')}-{safe_name(tag, 'run')}"
    candidate, n = root / stem, 1
    while candidate.exists():
        n += 1
        candidate = root / f"{stem}-{n}"
    return candidate


def start_run(request: RunRequest, *, user: str, root: Optional[Path] = None,
              timeout: Optional[float] = -1.0, command: Optional[Sequence[str]] = None) -> Path:
    """Launch the worker for ``request`` in a new run folder and return the folder.

    ``output_dir`` is always replaced with ``<run folder>/reports`` so a UI run can
    never write to the shared drive. ``timeout`` is seconds (``None`` = none;
    default from :func:`timeout_seconds`). Raises :class:`RunLocked` if a run is
    already active under ``root``.
    """
    root = Path(root or output_root()).resolve()
    timeout = timeout_seconds() if timeout == -1.0 else timeout
    run_dir = _new_run_dir(root, user, request.output_tag)
    _acquire_lock(root, run_dir, user)
    try:
        (run_dir / REPORTS).mkdir(parents=True)
        request = replace(request, output_dir=str(run_dir / REPORTS))
        (run_dir / REQUEST).write_text(request_to_json(request), encoding="utf-8")
        update_meta(run_dir, user=user, started=now_iso(), finished=None, summary=summarize(request),
                    tag=request.output_tag, send_email=request.send_email, exit_code=None, cancelled=False,
                    timed_out=False, timeout_seconds=timeout)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
        env["PYTHONUNBUFFERED"] = "1"
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0  # Ctrl+C on the UI spares it
        with open(run_dir / PROGRESS, "wb") as out, open(run_dir / LOG, "wb") as err:
            proc = subprocess.Popen([*(command or WORKER_COMMAND), REQUEST, "--result", RESULT], cwd=run_dir,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env, creationflags=flags)
    except BaseException as exc:
        if run_dir.is_dir():
            update_meta(run_dir, finished=now_iso(), exit_code=1, error=f"Could not start the worker: {exc}")
        _release_lock(root, run_dir)
        raise
    with _GUARD:
        _PROCESSES[str(run_dir)] = proc
        _write_json(root / LOCK_NAME, {**(read_lock(root) or {}), "pid": proc.pid, "run_dir": str(run_dir)})
        update_meta(run_dir, pid=proc.pid)
    threading.Thread(target=_watch, args=(proc, run_dir, root, timeout), name=f"watch-{run_dir.name}",
                     daemon=True).start()
    return run_dir


def _watch(proc: subprocess.Popen, run_dir: Path, root: Path, timeout: Optional[float]) -> None:
    """Wait for the worker (killing it at ``timeout``), then finalize the folder and release the lock."""
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate(proc)
    finally:
        with _GUARD:
            _finalize(run_dir, proc.returncode, timed_out=timed_out)
            _PROCESSES.pop(str(run_dir), None)
            _release_lock(root, run_dir)


def _terminate(proc: subprocess.Popen) -> None:
    """Stop the worker and anything it spawned: taskkill /T on Windows, then terminate(), kill() after a grace."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(GRACE_SECONDS)


def cancel(run_dir: Path, by: Optional[str] = None) -> bool:
    """Ask the run in ``run_dir`` to stop; ``True`` if a live worker was found and terminated."""
    run_dir = Path(run_dir)
    meta = update_meta(run_dir, cancel_requested=True, cancelled_by=by)
    proc = _PROCESSES.get(str(run_dir))
    if proc is not None:
        _terminate(proc)  # the watcher thread finalizes
        return True
    pid = meta.get("pid")
    if not pid_alive(pid):
        return False
    if os.name == "nt":  # started by an earlier UI process: no Popen handle, kill by PID
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        os.kill(int(pid), 15)
    return True


def _finalize(run_dir: Path, returncode: Optional[int], *, timed_out: bool, orphaned: bool = False) -> None:
    """Write the final meta.json (and a result.json if the worker left none)."""
    with _GUARD:
        meta = read_json(run_dir / META) or {}
        cancelled = bool(meta.get("cancel_requested")) and not timed_out
        result = read_json(run_dir / RESULT)
        if result is None or not isinstance(result.get("exit_code"), int):
            if cancelled:
                reason = f"Run cancelled by {meta.get('cancelled_by') or 'a user'}"
            elif timed_out:
                reason = f"Run timed out after {meta.get('timeout_seconds')} s and was stopped"
            elif orphaned:
                reason = "The UI stopped tracking this run before the worker finished (no result written)"
            else:
                reason = f"The worker exited with code {returncode} without writing a result (see run.log)"
            result = {"exit_code": returncode or 1, "summaries": {}, "paths": {}, "files": [], "errors": [reason]}
            _write_json(run_dir / RESULT, result)
        meta.update(finished=now_iso(), exit_code=result["exit_code"], returncode=returncode,
                    cancelled=cancelled, timed_out=timed_out, orphaned=orphaned)
        _write_json(run_dir / META, meta)


def wait(run_dir: Path, timeout: float = 60.0, poll: float = 0.1) -> bool:
    """Block until ``run_dir`` is finalized (for tests and scripts); ``True`` if it finished in time."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (read_json(Path(run_dir) / META) or {}).get("finished") and str(run_dir) not in _PROCESSES:
            return True
        time.sleep(poll)
    return False


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Progress:
    """Where a run is, derived from its progress events."""

    fraction: float  # 0..1 for the progress bar
    stage: str  # last stage seen (day, db, load, process, fund, operation, reports, email, done)
    day: Optional[str]
    text: str  # one human-readable line
    events: int


_STAGE_POINT = {"db": 0.02, "load": 0.05, "process": 0.1, "reports": 0.85, "email": 0.95, "done": 1.0}
_FUNDS_START, _FUNDS_SPAN = 0.1, 0.75  # share of a day spent in the per-fund loop


def read_events(run_dir: Path) -> List[dict]:
    """The progress events so far; a half-written last line or stray text is skipped."""
    try:
        lines = (Path(run_dir) / PROGRESS).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and isinstance(event.get("stage"), str):
            events.append(event)
    return events


def progress(events: Sequence[dict]) -> Progress:
    """Bar fraction: finished days plus the current day's share (stage reached, or fund ``index``/``total``)."""
    day_index, day_total, within, passes = 1, 1, 0.0, []
    stage, day, text = "starting", None, "Starting the worker…"
    for event in events:
        stage, day = event["stage"], event.get("day") or day
        index, total = event.get("index"), event.get("total")
        counted = isinstance(index, int) and isinstance(total, int) and total > 0
        if stage == "day" and counted:
            day_index, day_total, within, passes = index, total, 0.0, []
        elif stage in ("fund", "operation") and counted:
            if stage == "fund" and event.get("message") not in passes:
                passes.append(event.get("message"))  # trading makes two passes over the funds (ex_ante, ex_post)
            span = _FUNDS_SPAN / (2 if {"ex_ante", "ex_post"} & set(passes) else 1)
            start = _FUNDS_START + span * max(len(passes) - 1, 0)
            within = max(within, start + span * (index - (1.0 if stage == "fund" else 0.5)) / total)
        elif stage in _STAGE_POINT:
            within = max(within, _STAGE_POINT[stage])
        text = _describe(event)
    fraction = ((day_index - 1) + within) / day_total
    if day_total > 1 and stage != "done":
        text = f"Day {day_index}/{day_total} · {text}"
    return Progress(min(max(fraction, 0.0), 1.0), stage, day, text, len(events))


def _describe(event: dict) -> str:
    stage, fund, message = event["stage"], event.get("fund"), event.get("message") or ""
    index, total = event.get("index"), event.get("total")
    labels = {"db": "Connecting to the database", "load": "Loading data", "process": "Processing",
              "reports": "Writing reports", "email": "Sending email", "done": "Finished", "day": "Starting day"}
    if stage in ("fund", "operation") and fund:
        count = f" ({index}/{total})" if index and total else ""
        return f"{message or stage}: {fund}{count}"
    text = labels.get(stage, stage)
    if event.get("day"):
        text += f" {event['day']}"
    if message:
        text += f" — {message}"
    return text


def tail(path: Path, lines: int = 40, max_bytes: int = 64_000) -> str:
    """The last ``lines`` lines of a text file ('' if missing)."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - max_bytes, 0))
            data = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-lines:])


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

def publish(run_dir: Path, dest_root: Optional[Path], *, by: str) -> Tuple[List[Path], List[Path]]:
    """Copy ``<run_dir>/reports/**`` into ``dest_root`` (same sub-folders); existing files are skipped.

    Returns (copied, skipped). Raises ``ValueError`` when publishing is not configured or the run is unfinished.
    """
    if dest_root is None:
        raise ValueError("Publishing is disabled: FUNDMANAGER_UI_PUBLISH_ROOT is not set")
    run_dir, dest_root = Path(run_dir), Path(dest_root)
    meta = read_json(run_dir / META) or {}
    if not meta.get("finished"):
        raise ValueError("Only finished runs can be published")
    copied: List[Path] = []
    skipped: List[Path] = []
    reports = run_dir / REPORTS
    for source in sorted(p for p in reports.rglob("*") if p.is_file()):
        target = dest_root / source.relative_to(reports)
        if target.exists():
            skipped.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    history = list(meta.get("published") or [])
    history.append({"to": str(dest_root), "at": now_iso(), "by": by, "copied": len(copied), "skipped": len(skipped)})
    update_meta(run_dir, published=history)
    return copied, skipped
