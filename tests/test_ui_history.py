"""ui.history: listing run folders, statuses, results, zips and re-run requests (no DB, no worker).

    python -m unittest discover tests -p "test_ui_history.py" -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path

os.environ["DB_CONNECTION_STRING"] = "sqlite:///:memory:"  # nothing here may reach the real DB
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.api import RunRequest, request_to_json  # noqa: E402
from ui import history, runner  # noqa: E402

DAY = date(2026, 10, 30)


class HistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def make_run(self, name: str, *, meta: object = None, result: object = None, request: RunRequest = None,
                 files: tuple = ()) -> Path:
        run_dir = self.root / name
        (run_dir / runner.REPORTS).mkdir(parents=True)
        if request is not None:
            (run_dir / runner.REQUEST).write_text(request_to_json(request), encoding="utf-8")
        entries = []
        for file_name in files:
            path = run_dir / runner.REPORTS / file_name
            path.write_bytes(b"data " + file_name.encode())
            entries.append({"day": DAY.isoformat(), "label": f"{file_name}_label", "path": str(path),
                            "kind": "pdf" if file_name.endswith(".pdf") else "excel"})
        if result is not None:
            if isinstance(result, dict):
                result = json.dumps({"files": entries, **result})
            (run_dir / runner.RESULT).write_text(result, encoding="utf-8")
        if meta is not None:
            (run_dir / runner.META).write_text(meta if isinstance(meta, str) else json.dumps(meta), encoding="utf-8")
        return run_dir

    def test_list_runs_newest_first_with_statuses(self) -> None:
        request = RunRequest(operations=("compliance",), start_date=DAY, funds=("KNG",), send_email=False)
        self.make_run("20261101-080000-ann-ok", request=request, files=("a.xlsx", "a.pdf"),
                      meta={"user": "ann", "started": "s", "finished": "f", "exit_code": 0, "summary": "S1"},
                      result={"exit_code": 0, "errors": []})
        self.make_run("20261102-080000-bob-fail", request=request,
                      meta={"user": "bob", "finished": "f", "exit_code": 1}, result={"exit_code": 1, "errors": ["x"]})
        self.make_run("20261103-080000-cat-cancel", meta={"user": "cat", "finished": "f", "exit_code": 1,
                                                          "cancelled": True})
        self.make_run("20261103-090000-dan-partial", request=request, meta={"user": "dan", "finished": None})
        self.make_run("20261103-100000-eve-corrupt", request=request, meta="{not json", result="[broken")
        (self.root / "not-a-run").mkdir()
        (self.root / "stray.txt").write_text("x")
        records = history.list_runs(self.root)
        self.assertEqual([r.name for r in records], ["20261103-100000-eve-corrupt", "20261103-090000-dan-partial",
                                                     "20261103-080000-cat-cancel", "20261102-080000-bob-fail",
                                                     "20261101-080000-ann-ok"])
        by_user = {r.user: r for r in records}
        self.assertEqual(by_user["ann"].status, "ok")
        self.assertEqual(by_user["ann"].files, 2)
        self.assertEqual(by_user["ann"].summary, "S1")
        self.assertEqual(by_user["ann"].as_of, "2026-10-30")
        self.assertEqual(by_user["bob"].status, "failed")
        self.assertEqual(by_user["bob"].errors, ["x"])
        self.assertEqual(by_user["cat"].status, "cancelled")
        self.assertEqual(by_user["dan"].status, "incomplete")
        corrupt = records[0]
        self.assertEqual((corrupt.user, corrupt.status, corrupt.files), ("?", "incomplete", 0))
        self.assertIn("KNG", corrupt.summary)  # rebuilt from request.json
        self.assertEqual(len(history.list_runs(self.root, limit=2)), 2)
        self.assertEqual(history.list_runs(self.root / "missing"), [])

    def test_running_status_follows_the_lock(self) -> None:
        run_dir = self.make_run("20261103-110000-me-live", meta={"user": "me", "finished": None})
        (self.root / runner.LOCK_NAME).write_text(json.dumps({"pid": os.getpid(), "run_dir": str(run_dir)}))
        self.assertEqual(history.list_runs(self.root)[0].status, "running")

    def test_load_result_only_trusts_files_inside_the_run(self) -> None:
        outside = self.root / "outside.xlsx"
        outside.write_bytes(b"secret")
        run_dir = self.make_run("20261103-120000-me-x", files=("a.xlsx",), meta={"finished": "f", "exit_code": 0},
                                result={"exit_code": 0, "summaries": {"2026-10-30": {"processed_funds": 1},
                                                                       "bad": 3}})
        data = json.loads((run_dir / runner.RESULT).read_text())
        data["files"] += [{"day": "d", "label": "l", "path": str(outside), "kind": "excel"},
                          {"day": "d", "label": "l", "path": str(run_dir / "gone.pdf"), "kind": "pdf"},
                          {"label": "missing day"}]
        (run_dir / runner.RESULT).write_text(json.dumps(data))
        result = history.load_result(run_dir)
        self.assertEqual([Path(f.path).name for f in result.files], ["a.xlsx"])
        self.assertEqual(result.summaries, {"2026-10-30": {"processed_funds": 1}})
        self.assertEqual(history.load_result(self.root / "nothing").files, [])

    def test_zip_reports(self) -> None:
        run_dir = self.make_run("20261103-130000-me-z", files=("a.xlsx", "a.pdf"), result={"exit_code": 0})
        zip_path = history.zip_reports(run_dir)
        self.assertEqual(zip_path, run_dir / f"{run_dir.name}.zip")
        with zipfile.ZipFile(zip_path) as archive:
            self.assertEqual(sorted(archive.namelist()), ["a.pdf", "a.xlsx"])
        mtime = zip_path.stat().st_mtime
        self.assertEqual(history.zip_reports(run_dir).stat().st_mtime, mtime)  # reused, not rebuilt
        empty = self.make_run("20261103-140000-me-empty", result={"exit_code": 1})
        self.assertIsNone(history.zip_reports(empty))

    def test_load_request_round_trip(self) -> None:
        request = RunRequest(operations=("compliance", "reconciliation"), start_date=DAY, end_date=date(2026, 11, 2),
                             cohorts=("ETF_FUNDS",), exclude=("KNG",), compliance_tests=("gics_compliance",),
                             date_mode="stacked", send_email=False, output_tag="t")
        run_dir = self.make_run("20261103-150000-me-r", request=request)
        self.assertEqual(history.load_request(run_dir), request)
        with self.assertRaises(OSError):
            history.load_request(self.root / "nope")
        (run_dir / runner.REQUEST).write_text('{"operations": ["compliance"]}')
        with self.assertRaises(ValueError):
            history.load_request(run_dir)


if __name__ == "__main__":
    unittest.main()
