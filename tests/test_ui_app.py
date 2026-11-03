"""ui/app.py driven headless with streamlit.testing.v1.AppTest: catalog() faked, a fake worker, no DB.

    python -m unittest discover tests -p "test_ui_app.py" -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

os.environ["DB_CONNECTION_STRING"] = "sqlite:///:memory:"  # nothing here may reach the real DB
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from config import fund_definitions  # noqa: E402
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

    def app(self) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def run_once(self, at: AppTest) -> Path:
        """Click Run, wait for the fake worker, rerun to show the results; return the run folder."""
        at.button(key="nr_run").click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        run_dir = Path(at.session_state["nr_watch"])
        self.assertTrue(runner.wait(run_dir, timeout=TIMEOUT))
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
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
        at.multiselect(key="nr_cohorts").set_value(["ETF_FUNDS"]).run()
        self.assertEqual(at.multiselect(key="nr_funds").options, ["KNG", "RDVI", "TDVI"])
        at.multiselect(key="nr_exclude").set_value(["KNG"]).run()
        self.assertIn("2 funds selected", self.texts(at.markdown))

    def test_validation_errors_block_the_run(self) -> None:
        at = self.app()
        at.checkbox(key="nr_op_compliance").uncheck().run()
        self.assertIn("Pick at least one operation", self.texts(at.error))
        self.assertTrue(at.button(key="nr_run").disabled)
        at.checkbox(key="nr_op_compliance").check().run()
        at.multiselect(key="nr_cohorts").set_value(["ETF_FUNDS"]).run()
        at.multiselect(key="nr_exclude").set_value(["KNG", "RDVI", "TDVI"]).run()
        self.assertIn("fund selection is empty", self.texts(at.error))
        self.assertTrue(at.button(key="nr_run").disabled)
        at.multiselect(key="nr_exclude").set_value([]).run()
        at.radio(key="nr_rules_mode").set_value(new_run.CHOOSE_RULES).run()
        self.assertIn("Pick at least one compliance rule", self.texts(at.error))
        at.multiselect(key="nr_rules").set_value(["gics_compliance"]).run()
        self.assertFalse(at.error)
        at.checkbox(key="nr_email").check().run()
        self.assertIn("confirmation to send notification emails", self.texts(at.error))
        self.assertTrue(at.button(key="nr_run").disabled)
        at.checkbox(key="nr_email_ok").check().run()
        self.assertFalse(at.error)
        self.assertFalse(at.button(key="nr_run").disabled)

    def test_range_and_trading_rules(self) -> None:
        at = self.app()
        at.radio(key="nr_date_kind").set_value(new_run.RANGE).run()
        at.date_input(key="nr_range").set_value((date(2026, 10, 31), date(2026, 11, 1))).run()  # a weekend
        at.radio(key="nr_date_mode").set_value(new_run.STACKED).run()
        self.assertIn("no business days", self.texts(at.error))
        at.radio(key="nr_kind").set_value(new_run.KIND_TRADING).run()
        self.assertTrue(at.radio(key="nr_date_mode").disabled)  # stacked is EOD only

    def test_run_with_fake_worker_shows_downloads(self) -> None:
        at = self.app()
        at.multiselect(key="nr_funds").set_value(["KNG", "RDVI"]).run()
        at.text_input(key="nr_tag").input("ui test").run()
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
        self.assertTrue(any(label.endswith(".pdf") for label in labels))
        self.assertTrue((run_dir / f"{run_dir.name}.zip").is_file())
        self.assertIn("| day | processed_funds | funds_with_errors |", self.texts(at.markdown))  # per-day summary
        self.assertIn("Publishing is off", self.texts(at.caption))

    def test_publish_copies_to_the_publish_root(self) -> None:
        publish = self.tmp / "published"
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_PUBLISH_ROOT": str(publish)}):
            at = self.app()
            self.run_once(at)
            self.assertTrue(at.button(key="nr-publish").disabled)  # behind a confirmation
            at.checkbox(key="nr-publish-sure").check().run()
            at.button(key="nr-publish").click().run()
        self.assertIn("Published 2 file(s)", self.texts(at.success))
        self.assertEqual(sorted(p.name for p in publish.rglob("*.*")),
                         ["compliance_report_%s.pdf" % new_run.default_as_of(), "compliance_report_%s.xlsx"
                          % new_run.default_as_of()])

    def test_active_run_disables_the_form_and_shows_progress(self) -> None:
        at = self.app()
        at.text_input(key="nr_tag").input("slow").run()
        at.button(key="nr_run").click().run()
        run_dir = Path(at.session_state["nr_watch"])
        fresh = self.app()  # a second browser session reattaches through the lock file
        self.assertTrue(fresh.button(key="nr_run").disabled)
        self.assertEqual(fresh.session_state["nr_watch"], str(run_dir))
        self.assertIn("Run in progress", self.texts(fresh.subheader))
        fresh.button(key=f"cancel-{run_dir.name}").click().run()
        self.assertTrue(runner.wait(run_dir, timeout=TIMEOUT))
        fresh.run()
        self.assertIn("CANCELLED", self.texts(fresh.error))
        self.assertFalse(fresh.button(key="nr_run").disabled)


class OtherPagesTest(AppTestCase):
    def test_history_and_rerun(self) -> None:
        at = self.app()
        at.multiselect(key="nr_cohorts").set_value(["ETF_FUNDS"]).run()
        at.multiselect(key="nr_exclude").set_value(["TDVI"]).run()
        at.checkbox(key="nr_op_reconciliation").check().run()
        run_dir = self.run_once(at)
        at.sidebar.radio(key="page").set_value("Run history").run()
        self.assertEqual(at.title[0].value, "Run history")
        self.assertIn(f"| {run_dir.name} |", self.texts(at.markdown))
        at.button(key="hist_rerun").click().run()
        self.assertEqual(at.title[0].value, "New run")
        self.assertEqual(at.multiselect(key="nr_cohorts").value, ["ETF_FUNDS"])
        self.assertEqual(at.multiselect(key="nr_exclude").value, ["TDVI"])
        self.assertTrue(at.checkbox(key="nr_op_reconciliation").value)
        self.assertFalse(at.checkbox(key="nr_email").value)
        at.sidebar.radio(key="page").set_value("Reports").run()
        self.assertEqual(at.title[0].value, "Reports")
        self.assertGreaterEqual(len(at.download_button), 2)

    def test_password_gate(self) -> None:
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_PASSWORD": "s3cret"}):
            at = self.app()
            self.assertEqual(len(at.text_input), 1)
            self.assertFalse(at.multiselect)
            at.text_input(key="login_password").input("wrong")
            at.button[0].click().run()
            self.assertIn("Wrong password", self.texts(at.error))
            at.text_input(key="login_password").input("s3cret")
            at.button[0].click().run()
            self.assertEqual(at.title[0].value, "New run")

    def test_missing_sso_header_blocks(self) -> None:
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_USER_HEADER": "X-Forwarded-User"}):
            at = self.app()
        self.assertIn("Not signed in", self.texts(at.error))
        self.assertFalse(at.multiselect)

    def test_catalog_failure_falls_back_to_tickers(self) -> None:
        with mock.patch.object(api, "catalog", side_effect=RuntimeError("db down at mysql://secret")):
            at = self.app()
        errors = self.texts(at.error)
        self.assertIn("Could not load the fund groups", errors)
        self.assertNotIn("secret", errors)
        self.assertEqual(at.multiselect(key="nr_cohorts").options, [])


if __name__ == "__main__":
    unittest.main()
