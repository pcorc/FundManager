"""Start one ``processing.worker`` per run in its own folder and track it on disk.

Each run gets ``<output root>/<YYYYMMDD-HHMMSS>-<user>-<tag>/`` holding:

- ``request.json``  the request (its ``output_dir`` is always forced to ``reports/`` here)
- ``progress.jsonl`` the worker's stdout: one ``ProgressEvent`` JSON object per line
- ``run.log``       the worker's stderr (logging, UTF-8)
- ``result.json``   written by the worker (or by :func:`_finalize` when it died without one)
- ``views.json``    written by the worker just before result.json: per-day display tables
                    (schema in ``reporting.result_views``); absent for a worker that died early
- ``meta.json``     who, when, request summary, pid (+ creation time), exit code, cancelled / timed-out flags
- ``reports/``      the report files; ``logs/`` (the worker's cwd-relative log files)

Only one run happens per output root at a time: ``<root>/.run.lock`` holds the
worker's PID, its process creation time and the run folder. A lock whose
process is gone (or whose PID now belongs to another process) is stale; it is
retired atomically (renamed to a tombstone, so only one session finalizes it).
Because all state is on disk, any browser session (or a refreshed one, or a
restarted UI) can reattach to the active run through :func:`active_run`, and
:func:`refresh` finishes the bookkeeping for a worker no watcher thread owns.
A watcher thread in the UI process that started the worker waits for it,
enforces the timeout and writes the final ``meta.json``; :func:`cancel`
terminates the worker.

On Windows a file another thread is reading cannot be replaced or deleted
(``PermissionError``), so writes, renames and deletes here are retried briefly.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from config.run_configurations import FULL_COMPLIANCE_TESTS
from processing.api import RunRequest, request_to_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ui_runs"
WORKER_COMMAND: List[str] = [sys.executable, "-m", "processing.worker"]  # tests swap in a fake worker
LOCK_NAME = ".run.lock"
REQUEST, RESULT, LOG, META, PROGRESS, REPORTS = (
    "request.json", "result.json", "run.log", "meta.json", "progress.jsonl", "reports")
VIEWS = "views.json"  # the worker writes it beside RESULT (processing.worker.default_views_path)
GRACE_SECONDS = 5.0  # between terminate() and kill()
DEFAULT_TIMEOUT_MINUTES = 120.0
_STALE_UNREADABLE_LOCK_SECONDS = 30.0  # a lock file that never became valid JSON
RETRIES, RETRY_DELAY = 20, 0.025  # for Windows sharing violations (a reader has the file open)
RETRY_SECONDS = 5.0  # how long retry() and read_json() keep trying through sharing violations

_PROCESSES: Dict[str, subprocess.Popen] = {}  # run folder -> worker started by this UI process
_GUARD = threading.RLock()  # serialises meta/lock updates between sessions and watcher threads
_log = logging.getLogger("fundmanager.ui.runner")
T = TypeVar("T")


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


_EXTENDED = "\\\\?\\"  # the Windows extended-length path prefix
_MAX_PATH = 259  # Windows MAX_PATH (260) less the terminating NUL, when long paths are not enabled
REPORT_NAME_RESERVE = 100  # longest "<sub-folder>/<report file>" the backend writes under output_dir
_RUN_FOLDER_RESERVE = 60  # "\<YYYYMMDD-HHMMSS>-<user>-<tag>\reports" for a typical user and tag


def long_path(path: Path, reserve: int = REPORT_NAME_RESERVE) -> str:
    """``path`` as text; on Windows in extended-length form when ``path`` + ``reserve`` chars would pass MAX_PATH.

    A deep output root (or a long user/tag) otherwise makes the backend's report writes fail with
    FileNotFoundError on hosts without LongPathsEnabled. ``path`` must be absolute.
    """
    text = str(path)
    if os.name != "nt" or text.startswith(_EXTENDED) or len(text) + reserve <= _MAX_PATH:
        return text
    return _EXTENDED + "UNC\\" + text[2:] if text.startswith("\\\\") else _EXTENDED + text


def plain_path(text: str) -> str:
    """Undo :func:`long_path` (for comparing and displaying paths)."""
    if text.startswith(_EXTENDED + "UNC\\"):
        return "\\\\" + text[len(_EXTENDED) + 4:]
    return text[len(_EXTENDED):] if text.startswith(_EXTENDED) else text


def root_too_deep(root: Path) -> bool:
    """Whether report paths under ``root`` would typically pass MAX_PATH (plain Windows tools then miss them)."""
    return len(str(root)) + _RUN_FOLDER_RESERVE + REPORT_NAME_RESERVE > _MAX_PATH


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def retry(action: Callable[..., T], *args: object) -> T:
    """``action(*args)``, retried for a moment on ``PermissionError`` (a Windows sharing violation)."""
    deadline = time.monotonic() + RETRY_SECONDS
    attempt = 0
    while time.monotonic() < deadline:
        try:
            return action(*args)
        except PermissionError:  # many short tries: a reader's open window is brief, so a gap comes soon
            time.sleep(RETRY_DELAY * (1 + attempt % 2) / 5)
            attempt += 1
    return action(*args)


@functools.lru_cache(maxsize=None)
def _create_file_w() -> Callable[..., Optional[int]]:
    import ctypes
    create = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create.restype = ctypes.c_void_p
    create.argtypes = (ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                       ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
    return create


def _read_shared(path: Path) -> bytes:
    """``path``'s bytes. On Windows the file is opened with FILE_SHARE_DELETE, so a writer's ``os.replace`` of it
    succeeds while we read (a plain ``open`` blocks the replace, and busy readers could starve the writer)."""
    if os.name != "nt":
        return Path(path).read_bytes()
    import ctypes
    import msvcrt
    generic_read, share_all, open_existing, normal = 0x80000000, 0x7, 3, 0x80
    handle = _create_file_w()(str(path), generic_read, share_all, None, open_existing, normal, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())  # FileNotFoundError / PermissionError as usual
    with open(msvcrt.open_osfhandle(handle, os.O_RDONLY), "rb") as stream:
        return stream.read()


def read_json(path: Path) -> Optional[dict]:
    """The JSON object in ``path``, or ``None`` if it is missing, unreadable or not an object."""
    deadline = time.monotonic() + RETRY_SECONDS
    while True:
        try:
            data = json.loads(_read_shared(path).decode("utf-8"))
        except PermissionError:  # being replaced right now (Windows)
            if time.monotonic() >= deadline:
                return None
            time.sleep(RETRY_DELAY)
            continue
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict) -> None:
    """Write atomically (temp file + replace, retried while a reader holds ``path`` open)."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:12]}.tmp")  # never reused: a scanner may still hold the last one
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    try:
        retry(os.replace, tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _unlink(path: Path) -> None:
    """Delete ``path`` (retried on sharing violations); a missing file is fine."""
    try:
        retry(os.unlink, path)
    except FileNotFoundError:
        pass


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
    if request.date_mode == "stacked" or (request.end_date and request.end_date != request.start_date):
        dates += f"..{(request.end_date or request.start_date).isoformat()} ({request.date_mode})"
    parts = [", ".join(request.operations), selection, dates]
    if "compliance" in request.operations or "trading_compliance" in request.operations:
        tests = set(request.compliance_tests)
        parts.append("all rules" if not tests or tests == set(FULL_COMPLIANCE_TESTS) else f"{len(tests)} rule(s)")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Hiding secrets in text shown to users
# ---------------------------------------------------------------------------

_SECRET_ENV = ("DB_CONNECTION_STRING", "FUNDMANAGER_SMTP_PASSWORD")
_URL_PASSWORD = re.compile(r"(//[^\s/:@'\"]+:)[^\s@'\"]+@")  # scheme://user:password@host (scheme optional)
_DB_USER = re.compile(r"(for user )'[^']*'@'[^']*'", re.IGNORECASE)  # MySQL "Access denied for user 'u'@'h'"
_KEY_VALUE = re.compile(r"\b(password|passwd|pwd)(\s*=\s*)[^\s,;&'\"]+", re.IGNORECASE)


def redact(text: object) -> str:
    """``text`` with credentials masked: the DB connection string / SMTP password, URL passwords, DB users."""
    text = str(text)
    for name in _SECRET_ENV:
        value = (os.getenv(name) or "").strip()
        if len(value) < 4:
            continue
        text = text.replace(value, f"<{name}>")
        match = _URL_PASSWORD.search(value)
        password = match.group(0)[len(match.group(1)):-1] if match else ""
        if len(password) >= 4:
            text = text.replace(password, "***")
    text = _URL_PASSWORD.sub(r"\1***@", text)
    text = _DB_USER.sub(r"\1'***'@'***'", text)
    return _KEY_VALUE.sub(r"\1\2***", text)


# ---------------------------------------------------------------------------
# Process identity (PID + creation time, so a reused PID is never mistaken for the worker)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _kernel32():  # Windows only
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    return kernel32


def _handle_state(handle: int) -> Tuple[bool, Optional[int]]:
    """(still running, creation time in FILETIME ticks) of an open Windows process handle."""
    import ctypes
    from ctypes import wintypes

    kernel32 = _kernel32()
    code = wintypes.DWORD()
    running = bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
    times = [wintypes.FILETIME() for _ in range(4)]
    created = None
    if kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
        created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    return running, created


def _open_state(pid: int) -> Tuple[bool, bool, Optional[int]]:
    """(opened, running, creation time) of ``pid``; ``opened`` is False when it is gone or access is denied."""
    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False, False, None
    try:
        return (True, *_handle_state(handle))
    finally:
        kernel32.CloseHandle(handle)


def process_created(pid: Optional[int]) -> Optional[int]:
    """Creation time of the live process ``pid`` (Windows), or ``None`` if gone, not inspectable, or not Windows."""
    if os.name != "nt" or not isinstance(pid, int) or pid <= 0:
        return None
    _, running, created = _open_state(pid)
    return created if running else None


def _popen_created(proc: subprocess.Popen) -> Optional[int]:
    """Creation time of a process we started (read from its handle, so it works even if it already exited)."""
    if os.name != "nt":
        return None
    return _handle_state(int(proc._handle))[1]  # type: ignore[attr-defined]


def pid_alive(pid: Optional[int], created: Optional[int] = None) -> bool:
    """Whether ``pid`` is running and, when ``created`` is given (Windows), is still the process started then.

    Without ``created`` a process this account may not inspect (access denied) counts as alive; with it,
    such a process cannot be the one we started, so it does not.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        opened, running, actual = _open_state(pid)
        if not opened:
            return created is None and ctypes.get_last_error() == 5  # access denied: it exists, it isn't ours
        return running and (created is None or actual == created)
    try:
        os.kill(pid, 0)  # POSIX only: signal 0 checks existence (on Windows os.kill would terminate it)
    except ProcessLookupError:
        return False
    except PermissionError:
        return created is None
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


def _lock_is_stale(info: dict, path: Path) -> bool:
    """A lock is stale when its process is gone (or its PID was reused); an unreadable one once it is old."""
    if not info:  # unreadable: stale once it has had time to be written
        try:
            return time.time() - path.stat().st_mtime > _STALE_UNREADABLE_LOCK_SECONDS
        except OSError:
            return True
    if str(info.get("run_dir")) in _PROCESSES:
        return False  # our watcher thread still owns it and will release it
    return not pid_alive(info.get("pid"), info.get("pid_created"))


def _locked_message(info: dict) -> str:
    return (f"A run by {info.get('user', '?')} (started {info.get('started', '?')}) is in progress: "
            f"{Path(str(info.get('run_dir', '?'))).name}")


def _retire_stale_lock(root: Path, info: dict) -> bool:
    """Take a stale lock away atomically and finalize its run; ``True`` once that lock is gone.

    The lock is renamed to a unique tombstone, so when several sessions or UI processes find the same
    stale lock only the one whose rename succeeds finalizes the orphan run; the others see the lock gone
    or changed and retry. A lock that changed since ``info`` was read is left alone.
    """
    path = Path(root) / LOCK_NAME
    current = read_lock(root)
    if current is None:
        return True
    if current != info:
        return False  # replaced meanwhile (e.g. another session already took over and started a run)
    tomb = Path(root) / f"{LOCK_NAME}.stale.{uuid.uuid4().hex}"
    try:
        os.rename(path, tomb)
    except FileNotFoundError:
        return True
    except OSError:
        return False  # being read right now (Windows): try again
    taken = read_json(tomb) or {}
    if taken != info and not _lock_is_stale(taken, tomb):  # a live lock slipped in: put it back
        try:
            os.rename(tomb, path)
        except OSError:
            _log.error("Could not restore the run lock from %s", tomb.name)
        return False
    try:
        run_dir = Path(str(taken.get("run_dir"))) if taken.get("run_dir") else None
        if run_dir is not None and run_dir.is_dir():
            with _GUARD:
                if not (read_json(run_dir / META) or {}).get("finished"):
                    _finalize(run_dir, None, timed_out=False, orphaned=True)
    except Exception:
        _log.exception("Finalizing the orphan run of a stale lock failed")
    finally:
        try:
            _unlink(tomb)
        except OSError:
            _log.warning("Could not delete %s", tomb.name)
    return True


def _acquire_lock(root: Path, run_dir: Path, user: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_NAME
    with _GUARD:  # sessions of this UI process take turns; other processes are kept out by O_EXCL + rename
        for _ in range(RETRIES):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                info = read_lock(root)
                if info is None:
                    continue  # released meanwhile
                if not _lock_is_stale(info, path):
                    raise RunLocked(_locked_message(info))
                if not _retire_stale_lock(root, info):
                    time.sleep(RETRY_DELAY)
                continue
            except PermissionError:  # Windows: the old lock is being deleted or renamed right now
                time.sleep(RETRY_DELAY)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:  # the launching UI process holds it until Popen
                json.dump({"pid": os.getpid(), "pid_created": process_created(os.getpid()), "run_dir": str(run_dir),
                           "user": user, "started": now_iso()}, handle)
            return
    raise RunLocked("Could not acquire the run lock; try again")


def _release_lock(root: Path, run_dir: Path) -> None:
    """Remove the lock if it still belongs to ``run_dir``."""
    with _GUARD:
        info = read_lock(root)
        if info is not None and str(info.get("run_dir")) == str(run_dir):
            try:
                _unlink(Path(root) / LOCK_NAME)
            except OSError:
                _log.exception("Could not release the run lock of %s", Path(run_dir).name)


def active_run(root: Optional[Path] = None) -> Optional[Path]:
    """The run folder of the run in progress under ``root``, or ``None`` (retiring a stale lock)."""
    root = Path(root or output_root())
    with _GUARD:
        for _ in range(5):
            info = read_lock(root)
            if info is None:
                return None
            if not _lock_is_stale(info, root / LOCK_NAME):
                return Path(str(info["run_dir"])) if info.get("run_dir") else None
            if _retire_stale_lock(root, info):
                return None
            time.sleep(RETRY_DELAY)
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
    already active under ``root``. Once the worker has started it is always
    watched, even if recording its PID fails.
    """
    root = Path(root or output_root()).resolve()
    timeout = timeout_seconds() if timeout == -1.0 else timeout
    run_dir = _new_run_dir(root, user, request.output_tag)
    _acquire_lock(root, run_dir, user)
    try:
        (run_dir / REPORTS).mkdir(parents=True)
        request = replace(request, output_dir=long_path(run_dir / REPORTS))
        (run_dir / REQUEST).write_text(request_to_json(request), encoding="utf-8")
        update_meta(run_dir, user=user, started=now_iso(), finished=None, summary=summarize(request),
                    tag=request.output_tag, send_email=request.send_email, exit_code=None, cancelled=False,
                    timed_out=False, timeout_seconds=timeout)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"  # run.log is read back as UTF-8 (the stdout JSON is ASCII anyway)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0  # Ctrl+C on the UI spares it
        with open(run_dir / PROGRESS, "wb") as out, open(run_dir / LOG, "wb") as err:
            proc = subprocess.Popen([*(command or WORKER_COMMAND), REQUEST, "--result", RESULT], cwd=run_dir,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env, creationflags=flags)
    except BaseException as exc:
        try:
            if run_dir.is_dir():
                update_meta(run_dir, finished=now_iso(), exit_code=1, error=f"Could not start the worker: {exc}")
        finally:
            _release_lock(root, run_dir)
        raise
    _PROCESSES[str(run_dir)] = proc  # from here on the watcher owns the lock
    try:
        created = _popen_created(proc)
        with _GUARD:
            info = read_lock(root) or {}
            if str(info.get("run_dir")) == str(run_dir):
                _write_json(root / LOCK_NAME, {**info, "pid": proc.pid, "pid_created": created})
            else:
                _log.error("The run lock no longer names %s; not rewriting it", run_dir.name)
            update_meta(run_dir, pid=proc.pid, pid_created=created)
    except Exception:
        _log.exception("Recording the worker of %s failed; it is still watched", run_dir.name)
    finally:
        threading.Thread(target=_watch, args=(proc, run_dir, root, timeout), name=f"watch-{run_dir.name}",
                         daemon=True).start()
    return run_dir


def _watch(proc: subprocess.Popen, run_dir: Path, root: Path, timeout: Optional[float]) -> None:
    """Wait for the worker (killing it at ``timeout``), then finalize the folder and release the lock.

    Each bookkeeping step runs even if the one before failed, so the lock is always released and the run
    untracked; :func:`refresh` finalizes a run whose finalize failed here.
    """
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            _terminate(proc)
        except Exception:
            _log.exception("Stopping the timed-out worker of %s failed", run_dir.name)
    finally:
        try:
            _finalize(run_dir, proc.poll(), timed_out=timed_out)
        except Exception:
            _log.exception("Finalizing %s failed", run_dir.name)
        finally:
            _PROCESSES.pop(str(run_dir), None)
            try:
                _release_lock(root, run_dir)
            except Exception:
                _log.exception("Releasing the lock of %s failed", run_dir.name)


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


def _kill_pid(pid: Optional[int], created: Optional[int]) -> bool:
    """Kill a worker started by an earlier UI process, only while ``pid`` is still that process.

    On Windows the recorded creation time must match, so a PID reused by an unrelated process is never
    touched; without a recorded creation time nothing is killed.
    """
    if os.name == "nt" and created is None:
        return False
    if not pid_alive(pid, created):
        return False
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        os.kill(int(pid), 15)  # type: ignore[arg-type]
    return True


def cancel(run_dir: Path, by: Optional[str] = None) -> bool:
    """Ask the run in ``run_dir`` to stop; ``True`` if a live worker was found and terminated."""
    run_dir = Path(run_dir)
    meta = update_meta(run_dir, cancel_requested=True, cancelled_by=by)
    proc = _PROCESSES.get(str(run_dir))
    if proc is not None:
        _terminate(proc)  # the watcher thread finalizes
        return True
    return _kill_pid(meta.get("pid"), meta.get("pid_created"))  # started by an earlier UI process


def _seconds_since(iso: object) -> float:
    try:
        return (datetime.now() - datetime.fromisoformat(str(iso))).total_seconds()
    except ValueError:
        return 0.0


def refresh(run_dir: Path) -> bool:
    """Bring ``run_dir``'s on-disk state up to date; ``True`` once the run has finished.

    For a worker this UI process watches, that is the watcher's job. For one started by an earlier UI
    process (Streamlit restarted mid-run), this finalizes the run once the worker is gone, releases its
    lock and enforces the run's timeout. The progress panel calls it on every poll.
    """
    run_dir = Path(run_dir)
    meta = read_json(run_dir / META) or {}
    if meta.get("finished"):
        return True
    if str(run_dir) in _PROCESSES:
        return False
    active_run(run_dir.parent)  # retires the lock of a dead worker and finalizes its run
    meta = read_json(run_dir / META) or {}
    if meta.get("finished"):
        return True
    pid, created = meta.get("pid"), meta.get("pid_created")
    if pid is None:
        return False  # still launching (the launching process holds the lock)
    if not pid_alive(pid, created):
        with _GUARD:
            if not (read_json(run_dir / META) or {}).get("finished"):
                _finalize(run_dir, None, timed_out=False, orphaned=True)
        _release_lock(run_dir.parent, run_dir)
        return True
    limit = meta.get("timeout_seconds")
    if isinstance(limit, (int, float)) and limit > 0 and not meta.get("timeout_requested") \
            and _seconds_since(meta.get("started")) > limit:
        update_meta(run_dir, timeout_requested=True)
        _kill_pid(pid, created)
    return False


def _finalize(run_dir: Path, returncode: Optional[int], *, timed_out: bool, orphaned: bool = False) -> None:
    """Write the final meta.json (and a result.json if the worker left none)."""
    with _GUARD:
        meta = read_json(run_dir / META) or {}
        timed_out = timed_out or bool(meta.get("timeout_requested"))
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
    return Progress(min(max(fraction, 0.0), 1.0), stage, day, text)


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

    Returns (copied, skipped); a publish that copied files is recorded in meta.json. Raises ``ValueError``
    when publishing is not configured or the run is unfinished.
    """
    if dest_root is None:
        raise ValueError("Publishing is disabled: FUNDMANAGER_UI_PUBLISH_ROOT is not set")
    run_dir, dest_root = Path(run_dir), Path(dest_root)
    meta = read_json(run_dir / META) or {}
    if not meta.get("finished"):
        raise ValueError("Only finished runs can be published")
    copied: List[Path] = []
    skipped: List[Path] = []
    reports, dest = Path(long_path(run_dir / REPORTS)), Path(long_path(Path(os.path.abspath(dest_root))))
    for source in sorted(p for p in reports.rglob("*") if p.is_file()):
        target = dest / source.relative_to(reports)
        if target.exists():
            skipped.append(Path(plain_path(str(target))))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(Path(plain_path(str(target))))
    if copied:  # a publish that only skipped files changed nothing
        history = list(meta.get("published") or [])
        history.append({"to": str(dest_root), "at": now_iso(), "by": by, "copied": len(copied),
                        "skipped": len(skipped)})
        update_meta(run_dir, published=history)
    return copied, skipped
