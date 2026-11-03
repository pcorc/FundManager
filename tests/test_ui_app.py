"""ui/app.py driven headless with streamlit.testing.v1.AppTest: catalog() faked, a fake worker, no DB.

    python -m unittest discover tests -p "test_ui_app.py" -v

No test here launches a request with send_email=True (the fake worker never emails anyway).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest import mock

os.environ["DB_CONNECTION_STRING"] = "sqlite:///:memory:"  # nothing here may reach the real DB
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from config import fund_definitions  # noqa: E402
from config.run_configurations import FULL_COMPLIANCE_TESTS  # noqa: E402
from processing import api  # noqa: E402
from ui import history, new_run, runner  # noqa: E402

APP = str(REPO / "ui" / "app.py")
FAKE_WORKER = [sys.executable, str(Path(__file__).with_name("fake_worker.py"))]
FUNDS = sorted(fund_definitions.FUND_DEFINITIONS)
CATALOG = {"cohorts": {"ALL_FUNDS": FUNDS, "ETF_FUNDS": ["KNG", "RDVI", "TDVI"], "CLOSED_END_FUNDS": ["FTMIX"],
                       "PRIVATE_FUNDS": [], "VIT_AND_MUTUAL_FUNDS": []},
           "funds": FUNDS, "operations": list(api.OPERATIONS), "compliance_rules": [],
           "rule_catalog": api.rule_catalog(), "date_modes": list(api.DATE_MODES)}
TIMEOUT = 30
FRIDAY, SATURDAY = date(2026, 10, 30), date(2026, 10, 31)


class AppTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.root = self.tmp / "runs"
        env = {"FUNDMANAGER_UI_OUTPUT_ROOT": str(self.root), "FUNDMANAGER_UI_PUBLISH_ROOT": "",
               "FUNDMANAGER_UI_PASSWORD": "", "FUNDMANAGER_UI_USER_HEADER": ""}
        for patcher in (mock.patch.dict(os.environ, env), mock.patch.object(api, "catalog", return_value=CATALOG),
                        mock.patch.object(runner, "WORKER_COMMAND", FAKE_WORKER)):
            patcher.start()
            self.addCleanup(patcher.stop)
        st.cache_data.clear()
        self.addCleanup(self._stop_workers)

    def _stop_workers(self) -> None:
        for run_dir, proc in list(runner._PROCESSES.items()):
            runner._terminate(proc)
            runner.wait(Path(run_dir), timeout=15)

    def ok(self, at: AppTest) -> AppTest:
        """Assert the last script run raised nothing."""
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def app(self) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        return self.ok(at.run())

    def run_once(self, at: AppTest) -> Path:
        """Click Run, wait for the fake worker, rerun to show the results; return the run folder."""
        self.ok(at.button(key="nr_run").click().run())
        run_dir = Path(at.session_state["nr_watch"])
        self.assertTrue(runner.wait(run_dir, timeout=TIMEOUT))
        self.ok(at.run())
        return run_dir

    @staticmethod
    def texts(elements) -> str:
        return "\n".join(str(e.value) for e in elements)


class NewRunPageTest(AppTestCase):
    def test_page_renders_with_defaults(self) -> None:
        at = self.app()
        self.assertEqual(at.title[0].value, "New run")
        self.assertIn(f"{len(FUNDS)} funds selected", self.texts(at.markdown))
        self.assertEqual(at.multiselect(key="nr_cohorts").options,
                         [f"{name} ({len(CATALOG['cohorts'][name])})" for name in sorted(CATALOG["cohorts"])])
        self.assertTrue(at.checkbox(key="nr_op_compliance").value)
        self.assertFalse(at.checkbox(key="nr_email").value)  # email is off by default
        self.assertTrue(at.checkbox(key="nr_pdf").value)
        self.assertEqual(at.date_input(key="nr_as_of").value, new_run.default_as_of())
        self.assertFalse(at.button(key="nr_run").disabled)
        self.assertFalse(at.error)

    def test_groups_narrow_the_funds_and_update_the_count(self) -> None:
        at = self.app()
        self.ok(at.multiselect(key="nr_cohorts").set_value(["ETF_FUNDS"]).run())
        self.assertEqual(at.multiselect(key="nr_funds").options, ["KNG", "RDVI", "TDVI"])
        self.ok(at.multiselect(key="nr_exclude").set_value(["KNG"]).run())
        self.assertIn("2 funds selected", self.texts(at.markdown))

    def test_validation_errors_block_the_run(self) -> None:
        at = self.app()
        self.ok(at.checkbox(key="nr_op_compliance").uncheck().run())
        self.assertIn("Pick at least one operation", self.texts(at.error))
        self.assertTrue(at.button(key="nr_run").disabled)
        self.ok(at.checkbox(key="nr_op_compliance").check().run())
        self.ok(at.multiselect(key="nr_cohorts").set_value(["ETF_FUNDS"]).run())
        self.ok(at.multiselect(key="nr_exclude").set_value(["KNG", "RDVI", "TDVI"]).run())
        self.assertIn("fund selection is empty", self.texts(at.error))
        self.assertTrue(at.button(key="nr_run").disabled)
        self.ok(at.multiselect(key="nr_exclude").set_value([]).run())
        self.ok(at.radio(key="nr_rules_mode").set_value(new_run.CHOOSE_RULES).run())
        self.assertIn("Pick at least one compliance rule", self.texts(at.error))
        self.ok(at.multiselect(key="nr_rules").set_value(["gics_compliance"]).run())
        self.assertFalse(at.error)
        self.ok(at.checkbox(key="nr_email").check().run())
        self.assertIn("confirmation to send notification emails", self.texts(at.error))
        self.assertTrue(at.button(key="nr_run").disabled)
        self.ok(at.checkbox(key="nr_email_ok").check().run())
        self.assertFalse(at.error)
        self.assertFalse(at.button(key="nr_run").disabled)

    def test_range_and_trading_rules(self) -> None:
        at = self.app()
        self.ok(at.radio(key="nr_date_kind").set_value(new_run.RANGE).run())
        self.ok(at.date_input(key="nr_range").set_value((SATURDAY, date(2026, 11, 1))).run())  # a weekend
        self.assertIn("no business days", self.texts(at.error))  # per day: no longer runs Friday silently
        self.ok(at.radio(key="nr_date_mode").set_value(new_run.STACKED).run())
        self.assertIn("no business days", self.texts(at.error))
        self.assertFalse(at.checkbox(key="nr_pdf").value)  # stacked defaults to no PDF, like time_series
        self.ok(at.radio(key="nr_kind").set_value(new_run.KIND_TRADING).run())
        self.assertTrue(at.radio(key="nr_date_mode").disabled)  # stacked is EOD only
        self.assertEqual(at.radio(key="nr_date_mode").value, new_run.PER_DAY)  # and shows what will run

    def test_stacked_recon_without_daily_reports_is_refused(self) -> None:
        at = self.app()
        self.ok(at.checkbox(key="nr_op_reconciliation").check().run())
        self.ok(at.radio(key="nr_date_kind").set_value(new_run.RANGE).run())
        self.ok(at.date_input(key="nr_range").set_value((date(2026, 10, 29), FRIDAY)).run())
        self.ok(at.radio(key="nr_date_mode").set_value(new_run.STACKED).run())
        self.assertIn("recon reports would be lost", self.texts(at.error))
        self.assertTrue(at.button(key="nr_run").disabled)
        self.ok(at.checkbox(key="nr_daily").check().run())
        self.assertFalse(at.error)
        self.assertFalse(at.button(key="nr_run").disabled)

    def test_weekend_as_of_runs_and_says_friday(self) -> None:
        at = self.app()
        self.ok(at.date_input(key="nr_as_of").set_value(SATURDAY).run())
        self.assertIn("Saturday 2026-10-31 is not a business day: the run uses 2026-10-30", self.texts(at.info))
        caption = self.texts(at.caption)
        self.assertIn("| 2026-10-30 |", caption)
        self.assertIn("1 business day(s): 2026-10-30", caption)
        self.assertNotIn("2026-10-31", caption)

    def test_trading_always_needs_the_email_confirmation(self) -> None:
        at = self.app()
        self.ok(at.radio(key="nr_kind").set_value(new_run.KIND_TRADING).run())
        self.assertTrue(at.checkbox(key="nr_email_trading").value)  # shown ticked and locked
        self.assertTrue(at.checkbox(key="nr_email_trading").disabled)
        self.assertIn("always sends the trading-analysis email", self.texts(at.warning))
        self.assertIn("always sends the trading-analysis email", self.texts(at.error))
        self.assertTrue(at.button(key="nr_run").disabled)
        self.ok(at.checkbox(key="nr_trading_ok").check().run())
        self.assertFalse(at.error)
        self.assertFalse(at.button(key="nr_run").disabled)  # not pressed: a trading run would email
        choices = new_run.Choices(kind=new_run.KIND_TRADING, as_of=FRIDAY)
        with self.assertRaises(api.InvalidRequest):
            new_run.build_request(CATALOG, choices)
        choices.email_confirmed = True
        request = new_run.build_request(CATALOG, choices)
        self.assertEqual((request.send_email, request.output_tag), (True, "trading_compliance"))

    def test_email_confirmation_is_cleared_after_a_run(self) -> None:
        at = self.app()
        self.ok(at.multiselect(key="nr_funds").set_value(["KNG"]).run())
        self.ok(at.checkbox(key="nr_email").check().run())
        self.ok(at.checkbox(key="nr_email_ok").check().run())
        self.ok(at.checkbox(key="nr_email").uncheck().run())  # launched with email off (safety rule)
        self.ok(at.checkbox(key="nr_email").check().run())
        self.assertFalse(at.checkbox(key="nr_email_ok").value)  # re-ticking needs a new confirmation
        self.ok(at.checkbox(key="nr_email_ok").check().run())
        real, asked = runner.start_run, []

        def start_without_email(request, **kwargs):  # the form asked for email; the test never sends any
            asked.append(request.send_email)
            return real(replace(request, send_email=False), **kwargs)

        with mock.patch.object(runner, "start_run", side_effect=start_without_email):
            run_dir = self.run_once(at)
        self.assertEqual(asked, [True])
        self.assertFalse(history.load_request(run_dir).send_email)
        self.assertFalse(at.checkbox(key="nr_email").value)  # each emailing run needs its own confirmation
        self.assertNotIn("nr_email_ok", [c.key for c in at.checkbox])

    def test_picks_survive_other_pages_and_hidden_widgets(self) -> None:
        at = self.app()
        self.ok(at.multiselect(key="nr_funds").set_value(["KNG"]).run())
        self.ok(at.checkbox(key="nr_op_reconciliation").check().run())
        self.ok(at.radio(key="nr_date_kind").set_value(new_run.RANGE).run())
        self.ok(at.date_input(key="nr_range").set_value((date(2026, 10, 1), FRIDAY)).run())
        self.ok(at.radio(key="nr_rules_mode").set_value(new_run.CHOOSE_RULES).run())
        self.ok(at.multiselect(key="nr_rules").set_value(["gics_compliance"]).run())
        self.ok(at.sidebar.radio(key="page").set_value("Run history").run())
        self.ok(at.sidebar.radio(key="page").set_value("New run").run())
        self.ok(at.radio(key="nr_kind").set_value(new_run.KIND_TRADING).run())
        self.ok(at.radio(key="nr_kind").set_value(new_run.KIND_EOD).run())
        self.ok(at.radio(key="nr_date_kind").set_value(new_run.AS_OF).run())
        self.ok(at.radio(key="nr_rules_mode").set_value(new_run.ALL_RULES).run())
        self.ok(at.radio(key="nr_date_kind").set_value(new_run.RANGE).run())
        self.ok(at.radio(key="nr_rules_mode").set_value(new_run.CHOOSE_RULES).run())
        self.assertEqual(at.multiselect(key="nr_funds").value, ["KNG"])
        self.assertTrue(at.checkbox(key="nr_op_reconciliation").value)
        self.assertEqual(at.date_input(key="nr_range").value, (date(2026, 10, 1), FRIDAY))
        self.assertEqual(at.multiselect(key="nr_rules").value, ["gics_compliance"])

    def test_run_with_fake_worker_shows_downloads(self) -> None:
        at = self.app()
        self.ok(at.multiselect(key="nr_funds").set_value(["KNG", "RDVI"]).run())
        self.ok(at.text_input(key="nr_tag").input("ui test").run())
        run_dir = self.run_once(at)
        self.assertEqual(run_dir.parent, self.root.resolve())
        self.assertTrue(run_dir.name.endswith("-ui_test"))
        sent = history.load_request(run_dir)
        self.assertEqual((sent.funds, sent.send_email, sent.output_tag), (("KNG", "RDVI"), False, "ui_test"))
        self.assertEqual(Path(sent.output_dir), run_dir / runner.REPORTS)
        self.assertIn("OK", self.texts(at.success))
        labels = [b.proto.label for b in at.download_button]
        self.assertEqual(labels[0], "Download all (zip)")
        self.assertEqual(len(labels), 3)  # zip + xlsx + pdf
        self.assertTrue(any(label.endswith(".pdf") and "Compliance (PDF)" in label for label in labels))
        self.assertFalse((run_dir / f"{run_dir.name}.zip").exists())  # built only when downloaded
        self.assertTrue(history.zip_bytes(run_dir).startswith(b"PK"))
        self.assertIn("| day | processed_funds | funds_with_errors |", self.texts(at.markdown))  # per-day summary
        self.assertIn("Publishing is off", self.texts(at.caption))

    def test_blank_tag_uses_the_cli_name(self) -> None:
        at = self.app()
        self.ok(at.multiselect(key="nr_funds").set_value(["KNG"]).run())
        self.assertIn("tag compliance", self.texts(at.caption))
        run_dir = self.run_once(at)
        self.assertEqual(history.load_request(run_dir).output_tag, "compliance")

    def test_worker_errors_and_log_are_redacted(self) -> None:
        at = self.app()
        self.ok(at.text_input(key="nr_tag").input("leak").run())
        self.run_once(at)
        shown = self.texts(at.error) + self.texts(at.code)
        self.assertIn("Could not parse URL", shown)
        self.assertIn("cannot connect to", shown)
        self.assertNotIn("Hunter2Secret", shown)

    def test_publish_copies_to_the_publish_root(self) -> None:
        publish = self.tmp / "published"
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_PUBLISH_ROOT": str(publish)}):
            at = self.app()
            run_dir = self.run_once(at)
            sure, button = f"nr-publish-sure-{run_dir.name}", f"nr-publish-{run_dir.name}"
            self.assertTrue(at.button(key=button).disabled)  # behind a confirmation
            self.ok(at.checkbox(key=sure).check().run())
            self.ok(at.button(key=button).click().run())
        self.assertIn("Published 2 file(s)", self.texts(at.success))
        self.assertEqual(sorted(p.name for p in publish.rglob("*.*")),
                         ["compliance_report_%s.pdf" % new_run.default_as_of(), "compliance_report_%s.xlsx"
                          % new_run.default_as_of()])

    def test_active_run_disables_the_form_and_shows_progress(self) -> None:
        at = self.app()
        self.ok(at.text_input(key="nr_tag").input("slow").run())
        self.ok(at.button(key="nr_run").click().run())
        run_dir = Path(at.session_state["nr_watch"])
        fresh = self.app()  # a second browser session reattaches through the lock file
        self.assertTrue(fresh.button(key="nr_run").disabled)
        self.assertEqual(fresh.session_state["nr_watch"], str(run_dir))
        self.assertIn("Run in progress", self.texts(fresh.subheader))
        self.ok(fresh.button(key=f"cancel-{run_dir.name}").click().run())
        self.assertTrue(runner.wait(run_dir, timeout=TIMEOUT))
        self.ok(fresh.run())
        self.assertIn("CANCELLED", self.texts(fresh.error))
        self.assertFalse(fresh.button(key="nr_run").disabled)


PANEL_SCRIPT = """
import sys
sys.path.insert(0, {repo!r})
from pathlib import Path
import streamlit as st
from ui import panels
run_dir = Path({run_dir!r})
if panels.is_finished(run_dir):  # what new_run.render does
    st.write("FINISHED")
else:
    panels.progress_panel(str(run_dir), "tester")
"""


class ProgressPanelTest(AppTestCase):
    def test_panel_finalizes_a_run_whose_watcher_is_gone(self) -> None:
        """A worker started by a UI process that has since restarted: the panel finishes it and hands over."""
        run_dir = self.root / "20261103-090000-old-run"
        (run_dir / runner.REPORTS).mkdir(parents=True)
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        runner.update_meta(run_dir, user="old", started=runner.now_iso(), finished=None, pid=dead.pid,
                           pid_created=1, timeout_seconds=600)
        (run_dir / runner.RESULT).write_text(json.dumps({"exit_code": 0, "files": [], "errors": []}))
        (run_dir / runner.PROGRESS).write_text(json.dumps({"stage": "done", "message": "completed"}) + "\n")
        (self.root / runner.LOCK_NAME).write_text(json.dumps({"pid": dead.pid, "pid_created": 1,
                                                              "run_dir": str(run_dir), "user": "old"}))
        at = AppTest.from_string(PANEL_SCRIPT.format(repo=str(REPO), run_dir=str(run_dir)), default_timeout=TIMEOUT)
        self.ok(at.run())
        self.assertIn("FINISHED", self.texts(at.markdown))  # the fragment's st.rerun() swapped the panel out
        meta = runner.read_json(run_dir / runner.META)
        self.assertEqual((meta["exit_code"], meta["orphaned"]), (0, True))
        self.assertIsNone(runner.read_lock(self.root))

    def test_panel_shows_a_live_run(self) -> None:
        run_dir = runner.start_run(new_run.build_request(CATALOG, new_run.Choices(funds=["KNG"], tag="slow")),
                                   user="tester", root=self.root)
        at = AppTest.from_string(PANEL_SCRIPT.format(repo=str(REPO), run_dir=str(run_dir)), default_timeout=TIMEOUT)
        self.ok(at.run())
        self.assertIn("Run in progress", self.texts(at.subheader))
        runner.cancel(run_dir)
        self.assertTrue(runner.wait(run_dir, timeout=TIMEOUT))
        self.ok(at.run())
        self.assertIn("FINISHED", self.texts(at.markdown))


class OtherPagesTest(AppTestCase):
    def test_history_and_rerun(self) -> None:
        at = self.app()
        self.ok(at.multiselect(key="nr_cohorts").set_value(["ETF_FUNDS"]).run())
        self.ok(at.multiselect(key="nr_exclude").set_value(["TDVI"]).run())
        self.ok(at.checkbox(key="nr_op_reconciliation").check().run())
        run_dir = self.run_once(at)
        self.assertEqual(history.load_request(run_dir).output_tag, "eod")  # the CLI preset's name
        self.ok(at.sidebar.radio(key="page").set_value("Run history").run())
        self.assertEqual(at.title[0].value, "Run history")
        self.assertIn(f"| {run_dir.name} |", self.texts(at.markdown))
        self.ok(at.button(key="hist_rerun").click().run())
        self.assertEqual(at.title[0].value, "New run")
        self.assertEqual(at.multiselect(key="nr_cohorts").value, ["ETF_FUNDS"])
        self.assertEqual(at.multiselect(key="nr_exclude").value, ["TDVI"])
        self.assertTrue(at.checkbox(key="nr_op_reconciliation").value)
        self.assertFalse(at.checkbox(key="nr_email").value)
        self.ok(at.sidebar.radio(key="page").set_value("Reports").run())
        self.assertEqual(at.title[0].value, "Reports")
        self.assertEqual(at.selectbox(key="rep_day").options, [str(new_run.default_as_of())])  # the report date
        self.assertGreaterEqual(len(at.download_button), 2)

    def test_publish_confirmation_does_not_carry_over_to_another_run(self) -> None:
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_PUBLISH_ROOT": str(self.tmp / "published")}):
            at = self.app()
            first = self.run_once(at)
            second = self.run_once(at)
            self.ok(at.sidebar.radio(key="page").set_value("Run history").run())
            self.assertEqual(at.selectbox(key="hist_pick").value, second.name)
            self.ok(at.checkbox(key=f"hist-publish-sure-{second.name}").check().run())
            self.assertFalse(at.button(key=f"hist-publish-{second.name}").disabled)
            self.ok(at.selectbox(key="hist_pick").set_value(first.name).run())
            self.assertFalse(at.checkbox(key=f"hist-publish-sure-{first.name}").value)
            self.assertTrue(at.button(key=f"hist-publish-{first.name}").disabled)

    def test_password_gate(self) -> None:
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_PASSWORD": "s3cret"}):
            at = self.app()
            self.assertEqual(len(at.text_input), 1)
            self.assertFalse(at.multiselect)
            at.text_input(key="login_password").input("wrong")
            self.ok(at.button[0].click().run())
            self.assertIn("Wrong password", self.texts(at.error))
            at.text_input(key="login_password").input("s3cret")
            self.ok(at.button[0].click().run())
            self.assertEqual(at.title[0].value, "New run")

    def test_missing_sso_header_blocks(self) -> None:
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_USER_HEADER": "X-Forwarded-User"}):
            at = self.app()
        self.assertIn("Not signed in", self.texts(at.error))
        self.assertFalse(at.multiselect)

    def test_catalog_failure_falls_back_to_tickers_and_is_cached(self) -> None:
        with mock.patch.object(api, "catalog", side_effect=RuntimeError("db down at mysql://secret")) as failing:
            at = self.app()
            errors = self.texts(at.error)
            self.assertIn("Could not load the fund groups", errors)
            self.assertNotIn("secret", errors)
            self.assertEqual(at.multiselect(key="nr_cohorts").options, [])
            self.ok(at.checkbox(key="nr_op_reconciliation").check().run())
            self.ok(at.checkbox(key="nr_op_reconciliation").uncheck().run())
            self.assertEqual(failing.call_count, 1)  # not retried on every click
            self.assertIn("Could not load the fund groups", self.texts(at.error))


class RerunPrefillTest(unittest.TestCase):
    def round_trip(self, **kwargs) -> api.RunRequest:
        request = api.RunRequest(operations=("compliance",), start_date=date(2026, 10, 29), send_email=False,
                                 funds=("KNG",), **kwargs)
        return new_run.build_request(CATALOG, new_run.choices_from_request(CATALOG, request))

    def test_stacked_single_day_stays_stacked(self) -> None:
        self.assertEqual(self.round_trip(end_date=date(2026, 10, 29), date_mode="stacked").date_mode, "stacked")

    def test_superset_of_all_rules_is_kept(self) -> None:
        tests = tuple(r["key"] for r in CATALOG["rule_catalog"])  # every rule plus the report-only steps
        self.assertGreater(set(tests), set(FULL_COMPLIANCE_TESTS))
        rebuilt = self.round_trip(compliance_tests=tests)
        self.assertEqual(set(rebuilt.compliance_tests), set(tests))
        self.assertEqual(tuple(self.round_trip().compliance_tests), tuple(FULL_COMPLIANCE_TESTS))


if __name__ == "__main__":
    unittest.main()
