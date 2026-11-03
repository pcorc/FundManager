"""ui.runner: run folders, the worker launch, lock file, cancel, timeout and publish, with a fake worker.

    python -m unittest tests.test_ui_runner -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

os.environ["DB_CONNECTION_STRING"] = "sqlite:///:memory:"  # nothing here may reach the real DB
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config.run_configurations import FULL_COMPLIANCE_TESTS  # noqa: E402
from processing.api import RunRequest, request_from_json  # noqa: E402
from ui import runner  # noqa: E402

FAKE_WORKER = [sys.executable, str(Path(__file__).with_name("fake_worker.py"))]
DAY = date(2026, 10, 30)
EXTENDED = "\\" * 2 + "?" + "\\"  # the Windows extended-length path prefix


def req(tag: str = "ok", **kwargs) -> RunRequest:
    kwargs.setdefault("operations", ("compliance",))
    kwargs.setdefault("start_date", DAY)
    kwargs.setdefault("funds", ("KNG", "RDVI"))
    kwargs.setdefault("send_email", False)
    return RunRequest(output_tag=tag, **kwargs)


def dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class RunnerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "runs"
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        for run_dir, proc in list(runner._PROCESSES.items()):
            runner._terminate(proc)
            runner.wait(Path(run_dir), timeout=15)
        shutil.rmtree(runner.long_path(Path(self._tmp.name).resolve(), reserve=1000), ignore_errors=True)  # >260 chars
        self._tmp.cleanup()

    def start(self, request: RunRequest, **kwargs) -> Path:
        kwargs.setdefault("timeout", 60)
        return runner.start_run(request, user="tester", root=self.root, command=FAKE_WORKER, **kwargs)


class SettingsAndNamesTest(RunnerTestCase):
    def test_safe_name(self) -> None:
        self.assertEqual(runner.safe_name("VEST\\j.doe"), "VEST_j.doe")
        self.assertEqual(runner.safe_name("../../etc/passwd"), "etc_passwd")
        self.assertEqual(runner.safe_name("  ", default="user"), "user")
        self.assertEqual(runner.safe_name(None), "run")
        self.assertLessEqual(len(runner.safe_name("x" * 200)), 40)

    def test_run_dir_name_is_unique(self) -> None:
        now = datetime(2026, 11, 3, 9, 30, 0)
        first = runner._new_run_dir(self.root, "VEST\\j.doe", "my tag/1", now)
        self.assertEqual(first.name, "20261103-093000-VEST_j.doe-my_tag_1")
        first.mkdir(parents=True)
        self.assertEqual(runner._new_run_dir(self.root, "VEST\\j.doe", "my tag/1", now).name,
                         "20261103-093000-VEST_j.doe-my_tag_1-2")

    def test_env_settings(self) -> None:
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_OUTPUT_ROOT": str(self.root),
                                                   "FUNDMANAGER_UI_PUBLISH_ROOT": "",
                                                   "FUNDMANAGER_UI_TIMEOUT_MINUTES": "0"}):
            self.assertEqual(runner.output_root(), self.root.resolve())
            self.assertIsNone(runner.publish_root())
            self.assertIsNone(runner.timeout_seconds())
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_TIMEOUT_MINUTES": "1.5"}):
            self.assertEqual(runner.timeout_seconds(), 90)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FUNDMANAGER_UI_OUTPUT_ROOT", None)
            self.assertEqual(runner.output_root(), runner.DEFAULT_OUTPUT_ROOT.resolve())

    def test_summarize(self) -> None:
        text = runner.summarize(req(cohorts=("ETF_FUNDS",), funds=(), exclude=("KNG",),
                                    end_date=date(2026, 11, 2), date_mode="stacked"))
        self.assertEqual(text, "compliance | ETF_FUNDS minus KNG | 2026-10-30..2026-11-02 (stacked) | all rules")
        self.assertIn("1 rule(s)", runner.summarize(req(compliance_tests=("gics_compliance",))))
        superset = ("summary", "summary_metrics", *FULL_COMPLIANCE_TESTS)
        self.assertNotIn("all rules", runner.summarize(req(compliance_tests=superset)))
        self.assertIn("2026-10-30..2026-10-30 (stacked)", runner.summarize(req(end_date=DAY, date_mode="stacked")))

    def test_pid_alive(self) -> None:
        self.assertTrue(runner.pid_alive(os.getpid()))
        self.assertFalse(runner.pid_alive(dead_pid()))
        self.assertFalse(runner.pid_alive(None))


class RunLifecycleTest(RunnerTestCase):
    def test_successful_run(self) -> None:
        run_dir = self.start(req(output_dir=r"G:\Shared drives\nope", create_pdf=True))
        self.assertTrue(runner.wait(run_dir, timeout=30))
        self.assertEqual(run_dir.parent, self.root.resolve())
        self.assertTrue(run_dir.name.endswith("-tester-ok"))
        meta = runner.read_json(run_dir / runner.META)
        self.assertEqual(meta["exit_code"], 0)
        self.assertEqual(meta["user"], "tester")
        self.assertFalse(meta["cancelled"])
        self.assertTrue(meta["finished"])
        self.assertIn("KNG, RDVI", meta["summary"])
        # output_dir is forced into the run folder, never the shared drive
        sent = request_from_json((run_dir / runner.REQUEST).read_text(encoding="utf-8"))
        self.assertEqual(Path(sent.output_dir), run_dir / runner.REPORTS)
        result = runner.read_json(run_dir / runner.RESULT)
        self.assertEqual(len(result["files"]), 2)
        for item in result["files"]:
            self.assertTrue(Path(item["path"]).is_file())
            self.assertTrue(Path(item["path"]).is_relative_to(run_dir))
        # the worker ran in the run folder with the repo on PYTHONPATH, stderr captured to run.log
        log = (run_dir / runner.LOG).read_text(encoding="utf-8")
        self.assertIn(f"cwd={run_dir}", log)
        self.assertIn(str(runner.REPO_ROOT), log)
        self.assertIn("PYTHONUNBUFFERED=1", log)
        self.assertIn("EOD dates \u2014 trade date", runner.tail(run_dir / runner.LOG))  # UTF-8 end to end
        self.assertTrue((run_dir / "logs" / "compliance.log").is_file())
        events = runner.read_events(run_dir)
        self.assertEqual([e["stage"] for e in events][-1], "done")
        self.assertEqual(runner.progress(events).fraction, 1.0)
        self.assertIsNone(runner.read_lock(self.root))
        self.assertIsNone(runner.active_run(self.root))

    def test_lock_blocks_second_run_and_cancel_stops_it(self) -> None:
        run_dir = self.start(req("slow"))
        self.assertEqual(runner.active_run(self.root), run_dir)
        lock = runner.read_lock(self.root)
        self.assertEqual(lock["run_dir"], str(run_dir))
        self.assertEqual(lock["pid"], runner.read_json(run_dir / runner.META)["pid"])
        with self.assertRaises(runner.RunLocked):
            self.start(req("ok"))
        deadline = time.monotonic() + 15
        while not runner.read_events(run_dir) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(runner.read_events(run_dir))  # events stream while the worker runs
        self.assertTrue(runner.cancel(run_dir, by="boss"))
        self.assertTrue(runner.wait(run_dir, timeout=30))
        meta = runner.read_json(run_dir / runner.META)
        self.assertTrue(meta["cancelled"])
        self.assertEqual(meta["cancelled_by"], "boss")
        self.assertNotEqual(meta["exit_code"], 0)
        self.assertFalse(runner.pid_alive(meta["pid"]))
        self.assertIn("cancelled by boss", runner.read_json(run_dir / runner.RESULT)["errors"][0])
        self.assertIsNone(runner.active_run(self.root))
        self.assertTrue(runner.wait(self.start(req("ok")), timeout=30))  # the root is free again

    def test_timeout_kills_the_worker(self) -> None:
        run_dir = self.start(req("slow"), timeout=1.0)
        self.assertTrue(runner.wait(run_dir, timeout=30))
        meta = runner.read_json(run_dir / runner.META)
        self.assertTrue(meta["timed_out"])
        self.assertFalse(meta["cancelled"])
        self.assertIn("timed out", runner.read_json(run_dir / runner.RESULT)["errors"][0])

    def test_crash_without_result_gets_a_result(self) -> None:
        run_dir = self.start(req("crash"))
        self.assertTrue(runner.wait(run_dir, timeout=30))
        result = runner.read_json(run_dir / runner.RESULT)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("without writing a result", result["errors"][0])
        self.assertIn("Traceback: boom", runner.tail(run_dir / runner.LOG))

    def test_invalid_request_exit_code(self) -> None:
        run_dir = self.start(req("invalid"))
        self.assertTrue(runner.wait(run_dir, timeout=30))
        self.assertEqual(runner.read_json(run_dir / runner.META)["exit_code"], 2)

    def test_launch_failure_releases_the_lock(self) -> None:
        with self.assertRaises(OSError):
            runner.start_run(req(), user="tester", root=self.root, command=[str(self.root / "missing.exe")])
        self.assertIsNone(runner.read_lock(self.root))
        meta = runner.read_json(next(self.root.iterdir()) / runner.META)
        self.assertIn("Could not start the worker", meta["error"])


class LongPathTest(RunnerTestCase):
    def test_long_path_helpers(self) -> None:
        short = str(self.root / "x")
        self.assertEqual(runner.long_path(short), short)
        sep = "\\"
        deep = "C:" + sep + "d" * 200
        expected = deep if os.name != "nt" else EXTENDED + deep
        self.assertEqual(runner.long_path(deep), expected)
        self.assertEqual(runner.plain_path(runner.long_path(deep)), deep)
        unc = sep * 2 + "srv" + sep + "share" + sep + "u" * 200
        self.assertEqual(runner.plain_path(runner.long_path(unc)), unc)
        if os.name == "nt":
            self.assertTrue(runner.long_path(unc).startswith(EXTENDED + "UNC" + sep + "srv" + sep))

    @unittest.skipUnless(os.name == "nt", "MAX_PATH is a Windows limit")
    def test_deep_output_root_still_works(self) -> None:
        from ui import history

        self.root = Path(self._tmp.name) / ("deep_" + "r" * 60) / ("deeper_" + "s" * 60)
        run_dir = self.start(req(create_pdf=True))
        self.assertTrue(runner.wait(run_dir, timeout=30))
        sent = request_from_json((run_dir / runner.REQUEST).read_text(encoding="utf-8"))
        self.assertTrue(sent.output_dir.startswith(EXTENDED))
        self.assertEqual(runner.read_json(run_dir / runner.META)["exit_code"], 0)
        result = history.load_result(run_dir)
        self.assertEqual(len(result.files), 2)
        self.assertIsNotNone(history.zip_reports(run_dir))
        copied, _ = runner.publish(run_dir, Path(self._tmp.name) / "pub", by="tester")
        self.assertEqual(len(copied), 2)
        self.assertFalse(str(copied[0]).startswith(EXTENDED))


class StaleLockTest(RunnerTestCase):
    def test_dead_pid_lock_is_cleared_and_run_marked_orphaned(self) -> None:
        run_dir = self.root / "20261103-090000-ghost-run"
        run_dir.mkdir(parents=True)
        runner.update_meta(run_dir, user="ghost", started="2026-11-03T09:00:00", finished=None)
        (self.root / runner.LOCK_NAME).write_text(json.dumps({"pid": dead_pid(), "run_dir": str(run_dir)}))
        self.assertIsNone(runner.active_run(self.root))
        self.assertIsNone(runner.read_lock(self.root))
        meta = runner.read_json(run_dir / runner.META)
        self.assertTrue(meta["orphaned"])
        self.assertTrue(meta["finished"])
        self.assertIn("stopped tracking", runner.read_json(run_dir / runner.RESULT)["errors"][0])
        self.assertTrue(runner.wait(self.start(req()), timeout=30))

    def test_live_pid_lock_blocks(self) -> None:
        self.root.mkdir(parents=True)
        (self.root / runner.LOCK_NAME).write_text(json.dumps({"pid": os.getpid(), "run_dir": "x", "user": "u"}))
        with self.assertRaises(runner.RunLocked):
            self.start(req())

    def test_unreadable_lock_goes_stale_with_age(self) -> None:
        self.root.mkdir(parents=True)
        lock = self.root / runner.LOCK_NAME
        lock.write_text("{half")
        self.assertIsNone(runner.active_run(self.root))
        self.assertTrue(lock.exists())  # too young to be stale: it still blocks a new run
        with self.assertRaises(runner.RunLocked):
            self.start(req())
        old = time.time() - 3600
        os.utime(lock, (old, old))
        self.assertIsNone(runner.active_run(self.root))
        self.assertFalse(lock.exists())


class ProgressTest(unittest.TestCase):
    def test_per_day_fraction(self) -> None:
        events = [{"stage": "day", "day": "d1", "index": 1, "total": 2},
                  {"stage": "fund", "day": "d1", "fund": "A", "index": 1, "total": 2, "message": "eod"},
                  {"stage": "fund", "day": "d1", "fund": "B", "index": 2, "total": 2, "message": "eod"}]
        p = runner.progress(events)
        self.assertAlmostEqual(p.fraction, (0.1 + 0.75 * 0.5) / 2)
        self.assertEqual(p.text, "Day 1/2 · eod: B (2/2)")
        events += [{"stage": "done", "day": "d1", "index": 1, "total": 2},
                   {"stage": "day", "day": "d2", "index": 2, "total": 2}]
        self.assertAlmostEqual(runner.progress(events).fraction, 0.5)
        events.append({"stage": "done", "day": "d2", "index": 2, "total": 2, "message": "completed"})
        done = runner.progress(events)
        self.assertEqual((done.fraction, done.stage, done.day), (1.0, "done", "d2"))

    def test_trading_two_passes(self) -> None:
        events = [{"stage": "fund", "fund": "A", "index": 1, "total": 1, "message": "ex_ante"},
                  {"stage": "fund", "fund": "A", "index": 1, "total": 1, "message": "ex_post"}]
        self.assertAlmostEqual(runner.progress(events).fraction, 0.1 + 0.375)
        self.assertEqual(runner.progress([]).fraction, 0.0)

    def test_read_events_skips_bad_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, runner.PROGRESS).write_text('{"stage": "db"}\nnot json\n[1]\n{"stage": "lo', encoding="utf-8")
            self.assertEqual(runner.read_events(Path(tmp)), [{"stage": "db"}])
            self.assertEqual(runner.read_events(Path(tmp, "missing")), [])

    def test_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "log.txt")
            path.write_text("\n".join(f"line {i}" for i in range(100)), encoding="utf-8")
            self.assertEqual(runner.tail(path, lines=2), "line 98\nline 99")
            self.assertEqual(runner.tail(Path(tmp, "nope")), "")


class PublishTest(RunnerTestCase):
    def test_publish_copies_reports_and_skips_existing(self) -> None:
        run_dir = self.start(req(create_pdf=True))
        self.assertTrue(runner.wait(run_dir, timeout=30))
        dest = Path(self._tmp.name) / "published"
        with self.assertRaises(ValueError):
            runner.publish(run_dir, None, by="tester")
        copied, skipped = runner.publish(run_dir, dest, by="tester")
        self.assertEqual(sorted(p.name for p in copied), ["compliance_report_2026-10-30.pdf",
                                                         "compliance_report_2026-10-30.xlsx"])
        self.assertTrue(all(p.parent == dest / "Compliance" for p in copied))
        self.assertEqual(skipped, [])
        copied, skipped = runner.publish(run_dir, dest, by="tester")
        self.assertEqual((len(copied), len(skipped)), (0, 2))
        published = runner.read_json(run_dir / runner.META)["published"]
        self.assertEqual([p["copied"] for p in published], [2])  # the no-op publish is not recorded

    def test_unfinished_run_cannot_be_published(self) -> None:
        run_dir = self.start(req("slow"))
        with self.assertRaises(ValueError):
            runner.publish(run_dir, Path(self._tmp.name) / "published", by="tester")
        runner.cancel(run_dir)
        self.assertTrue(runner.wait(run_dir, timeout=30))


def sleeper() -> subprocess.Popen:
    """An unrelated live process (stands in for a process that reused a worker's PID)."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


class RobustFilesTest(RunnerTestCase):
    def test_write_json_retries_a_sharing_violation(self) -> None:
        self.root.mkdir(parents=True)
        target, real = self.root / "m.json", os.replace
        calls = []

        def flaky(src, dst):
            calls.append(dst)
            if len(calls) < 3:
                raise PermissionError(13, "Access is denied")
            real(src, dst)

        with mock.patch.object(runner.os, "replace", flaky):
            runner._write_json(target, {"a": 1})
        self.assertEqual((len(calls), runner.read_json(target)), (3, {"a": 1}))
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_concurrent_readers_never_break_meta_updates(self) -> None:
        run_dir = self.root / "r"
        run_dir.mkdir(parents=True)
        runner.update_meta(run_dir, n=0)
        stop, failures = threading.Event(), []

        def reader() -> None:
            while not stop.is_set():
                runner.read_json(run_dir / runner.META)
                runner.read_lock(self.root)

        threads = [threading.Thread(target=reader) for _ in range(2)]
        for thread in threads:
            thread.start()
        try:
            for n in range(300):
                try:
                    runner.update_meta(run_dir, n=n)
                except OSError as exc:
                    failures.append(exc)
        finally:
            stop.set()
            for thread in threads:
                thread.join()
        self.assertEqual(failures, [])
        self.assertEqual(runner.read_json(run_dir / runner.META)["n"], 299)

    def test_failed_finalize_still_releases_the_lock(self) -> None:
        with mock.patch.object(runner, "_finalize", side_effect=PermissionError(13, "denied")):
            run_dir = self.start(req())
            deadline = time.monotonic() + 30
            while str(run_dir) in runner._PROCESSES and time.monotonic() < deadline:
                time.sleep(0.05)
        self.assertNotIn(str(run_dir), runner._PROCESSES)
        self.assertIsNone(runner.read_lock(self.root))
        self.assertTrue(runner.refresh(run_dir))  # finalized later, from disk
        self.assertEqual(runner.read_json(run_dir / runner.META)["exit_code"], 0)

    def test_failure_after_popen_still_watches_the_worker(self) -> None:
        with mock.patch.object(runner, "_popen_created", side_effect=OSError("boom")):
            run_dir = self.start(req())
        self.assertTrue(runner.wait(run_dir, timeout=30))
        self.assertEqual(runner.read_json(run_dir / runner.META)["exit_code"], 0)
        self.assertIsNone(runner.read_lock(self.root))


class StaleLockRaceTest(RunnerTestCase):
    def stale_lock(self, name: str = "20261103-090000-ghost-run") -> dict:
        run_dir = self.root / name
        run_dir.mkdir(parents=True)
        runner.update_meta(run_dir, user="ghost", started="2026-11-03T09:00:00", finished=None)
        info = {"pid": dead_pid(), "run_dir": str(run_dir), "user": "ghost"}
        (self.root / runner.LOCK_NAME).write_text(json.dumps(info))
        return info

    def test_only_one_session_retires_a_stale_lock(self) -> None:
        info = self.stale_lock()
        self.assertTrue(runner._retire_stale_lock(self.root, info))
        live = {"pid": os.getpid(), "run_dir": str(self.root / "new"), "user": "alice"}
        (self.root / runner.LOCK_NAME).write_text(json.dumps(live))  # the winner took the root
        self.assertFalse(runner._retire_stale_lock(self.root, info))  # a late session must not delete it
        self.assertEqual(runner.read_lock(self.root), live)
        self.assertEqual(list(self.root.glob(".run.lock.stale.*")), [])

    def test_two_sessions_racing_for_a_stale_lock_start_one_run(self) -> None:
        self.stale_lock()
        barrier, real = threading.Barrier(2, timeout=2), runner.pid_alive
        started, locked = [], []

        def racing_pid_alive(pid, created=None):  # both sessions read the stale lock before either acts
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            return real(pid, created)

        def session(user: str) -> None:
            try:
                started.append(runner.start_run(req("slow"), user=user, root=self.root, command=FAKE_WORKER))
            except runner.RunLocked:
                locked.append(user)

        with mock.patch.object(runner, "pid_alive", racing_pid_alive):
            threads = [threading.Thread(target=session, args=(u,)) for u in ("alice", "bob")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(30)
        self.assertEqual((len(started), len(locked)), (1, 1))
        self.assertEqual(runner.read_lock(self.root)["run_dir"], str(started[0]))

    def test_reused_pid_is_stale_and_never_killed(self) -> None:
        other = sleeper()
        self.addCleanup(lambda: (other.kill(), other.wait()))
        run_dir = self.root / "20261103-090000-alice-run"
        run_dir.mkdir(parents=True)
        wrong = (runner.process_created(other.pid) or 0) + 1  # the worker's creation time was different
        runner.update_meta(run_dir, user="alice", started="2026-11-03T09:00:00", finished=None, pid=other.pid,
                           pid_created=wrong)
        (self.root / runner.LOCK_NAME).write_text(json.dumps({"pid": other.pid, "pid_created": wrong,
                                                              "run_dir": str(run_dir), "user": "alice"}))
        self.assertFalse(runner.cancel(run_dir, by="boss"))
        if os.name == "nt":
            self.assertIsNone(runner.active_run(self.root))  # stale: the PID belongs to someone else now
            self.assertTrue(runner.read_json(run_dir / runner.META)["orphaned"])
            runner.update_meta(run_dir, pid_created=None)
            self.assertFalse(runner.cancel(run_dir))  # no recorded creation time: never kill by PID
            self.assertIsNone(other.poll())  # the unrelated process survived

    @unittest.skipUnless(os.name == "nt", "process creation times are read on Windows")
    def test_process_identity(self) -> None:
        other = sleeper()
        self.addCleanup(lambda: (other.kill(), other.wait()))
        created = runner.process_created(other.pid)
        self.assertIsNotNone(created)
        self.assertEqual(runner._popen_created(other), created)
        self.assertTrue(runner.pid_alive(other.pid, created))
        self.assertFalse(runner.pid_alive(other.pid, created + 1))
        other.kill()
        other.wait()
        self.assertFalse(runner.pid_alive(other.pid, created))
        self.assertIsNone(runner.process_created(other.pid))


LAUNCHER = """
import os, sys
sys.path.insert(0, sys.argv[1])
from datetime import date
from pathlib import Path
from processing.api import RunRequest
from ui import runner
request = RunRequest(operations=("compliance",), start_date=date(2026, 10, 30), funds=("KNG",), send_email=False,
                     output_tag=sys.argv[3])
run_dir = runner.start_run(request, user="old", root=Path(sys.argv[2]), timeout=float(sys.argv[4]),
                           command=[sys.executable, sys.argv[5]])
print(run_dir, flush=True)
os._exit(0)  # the UI process dies (a Streamlit restart): the worker lives on, its watcher thread does not
"""


class ReattachTest(RunnerTestCase):
    def launch_and_die(self, tag: str, timeout: float) -> Path:
        env = {**os.environ, "DB_CONNECTION_STRING": "sqlite:///:memory:"}
        out = subprocess.run([sys.executable, "-c", LAUNCHER, str(REPO), str(self.root), tag, str(timeout),
                              FAKE_WORKER[1]], capture_output=True, text=True, timeout=60, check=True, env=env)
        return Path(out.stdout.strip().splitlines()[-1])

    def refresh_until_finished(self, run_dir: Path, seconds: float = 60) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if runner.refresh(run_dir):
                return True
            time.sleep(0.2)
        return False

    def test_run_of_a_dead_ui_process_is_finalized_by_refresh(self) -> None:
        run_dir = self.launch_and_die("ok", 60)
        self.assertTrue(self.refresh_until_finished(run_dir))
        meta = runner.read_json(run_dir / runner.META)
        self.assertEqual((meta["exit_code"], meta["timed_out"]), (0, False))
        self.assertIsNone(runner.read_lock(self.root))

    @unittest.skipUnless(os.name == "nt", "killing by PID needs the recorded creation time (Windows)")
    def test_timeout_is_enforced_without_a_watcher(self) -> None:
        run_dir = self.launch_and_die("slow", 1)
        pid = runner.read_json(run_dir / runner.META)["pid"]
        self.assertEqual(runner.active_run(self.root), run_dir)  # still running, owned by nobody here
        self.assertTrue(self.refresh_until_finished(run_dir))
        meta = runner.read_json(run_dir / runner.META)
        self.assertTrue(meta["timed_out"])
        self.assertFalse(runner.pid_alive(pid))
        self.assertIsNone(runner.active_run(self.root))


class RedactTest(unittest.TestCase):
    def test_masks_credentials(self) -> None:
        text = runner.redact("ArgumentError: Could not parse 'mysqlpymysql//fakeuser:FAKESECRET123@nohost.invalid/db'")
        self.assertNotIn("FAKESECRET123", text)
        self.assertIn("//fakeuser:***@nohost", text)
        self.assertNotIn("p4ss", runner.redact("mysql+pymysql://u:p4ss@h:3306/db"))
        self.assertEqual(runner.redact("(1045, \"Access denied for user 'fm'@'10.0.0.5' (using password: YES)\")"),
                         "(1045, \"Access denied for user '***'@'***' (using password: YES)\")")
        self.assertEqual(runner.redact("smtp password=abc123 rest"), "smtp password=*** rest")
        with mock.patch.dict(os.environ, {"DB_CONNECTION_STRING": "mysql+pymysql://me:Zz9!x@dbhost/fm"}):
            self.assertEqual(runner.redact("url mysql+pymysql://me:Zz9!x@dbhost/fm and Zz9!x"),
                             "url <DB_CONNECTION_STRING> and ***")
        self.assertEqual(runner.redact("plain text"), "plain text")


if __name__ == "__main__":
    unittest.main()
