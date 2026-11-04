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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import ui_views_fixture  # noqa: E402
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
CHECKS = [key for key, _ in new_run.check_rules(CATALOG)]
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
                        mock.patch.object(api, "staged_trades", return_value=self.staged()),
                        mock.patch.object(runner, "WORKER_COMMAND", FAKE_WORKER)):
            patcher.start()
            self.addCleanup(patcher.stop)
        st.cache_data.clear()
        self.addCleanup(self._stop_workers)

    def staged(self) -> dict:
        """What the read-only staged-trades lookup returns in these tests (no DB)."""
        return {}

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

    def pick(self, at: AppTest, *tickers: str) -> AppTest:
        """Tick fund tiles."""
        for ticker in tickers:
            self.ok(at.checkbox(key=f"nrf_{ticker}").check().run())
        return at

    def run_once(self, at: AppTest) -> Path:
        """Click Run, wait for the fake worker, rerun to show the results; return the run folder."""
        self.ok(at.button(key="nr_run").click().run())
        run_dir = Path(at.session_state["nr_watch"])
        self.assertTrue(runner.wait(run_dir, timeout=TIMEOUT))
        self.ok(at.run())
        return run_dir

    def go(self, at: AppTest, page: str) -> AppTest:
        """Switch page with the top navigation."""
        return self.ok(at.button_group(key="page").set_value(page).run())

    @staticmethod
    def run_label(at: AppTest) -> str:
        return [b.label for b in at.button if b.key == "nr_run"][0]

    @staticmethod
    def texts(elements) -> str:
        return "\n".join(str(e.value) for e in elements)

    @staticmethod
    def shown(at: AppTest) -> str:
        """Every text the page shows: HTML pieces, markdown, alerts, captions and code."""
        parts = []
        for kind in ("html", "markdown", "error", "warning", "info", "success", "caption", "code", "title"):
            parts += [str(e.value) for e in at.get(kind)]
        return "\n".join(parts)


class NewRunPageTest(AppTestCase):
    def test_page_renders_with_defaults(self) -> None:
        at = self.app()
        self.assertEqual(at.title[0].value, "New run")
        page = self.shown(at)
        self.assertIn("Configuration Preview", page)
        self.assertIn("Select Fund", page)
        self.assertIn("Pick at least one fund", page)  # nothing is ticked on a fresh form
        self.assertTrue(at.button(key="nr_run").disabled)
        self.assertEqual([b.key for b in at.button if (b.key or "").startswith("nr_chip_") and b.key != "nr_chip__clear"],
                         ["nr_chip_ALL_FUNDS", "nr_chip_CLOSED_END_FUNDS", "nr_chip_ETF_FUNDS"])  # empty groups hidden
        self.assertEqual(len([c for c in at.checkbox if c.key.startswith("nrf_")]), len(FUNDS))
        self.assertTrue(all(at.checkbox(key=f"nrr_{key}").value for key in CHECKS))  # every check ticked
        self.assertTrue(at.checkbox(key="nr_rules_all").value)
        self.assertTrue(at.checkbox(key="nr_op_compliance").value)
        self.assertFalse(at.checkbox(key="nr_email").value)  # email is off by default
        self.assertTrue(at.checkbox(key="nr_pdf").value)
        self.assertEqual(at.date_input(key="nr_as_of").value, new_run.default_as_of())
        self.assertIn("What will be checked", page)  # the check guide before any run
        self.assertIn("RUN COMPLIANCE CHECKS", self.run_label(at))
        self.pick(at, "KNG")
        self.assertFalse(at.button(key="nr_run").disabled)
        self.assertNotIn("Pick at least one fund", self.shown(at))

    def test_group_chip_ticks_its_funds_and_toggles(self) -> None:
        at = self.app()
        self.ok(at.button(key="nr_chip_ETF_FUNDS").click().run())
        self.assertEqual(at.session_state["nr_funds"], ["KNG", "RDVI", "TDVI"])
        self.assertTrue(at.checkbox(key="nrf_KNG").value)
        self.assertFalse(at.checkbox(key="nrf_DOGG").value)
        self.ok(at.checkbox(key="nrf_RDVI").uncheck().run())  # untick one: the chip ticks it again
        self.assertEqual(at.session_state["nr_funds"], ["KNG", "TDVI"])
        self.ok(at.button(key="nr_chip_ETF_FUNDS").click().run())
        self.assertEqual(sorted(at.session_state["nr_funds"]), ["KNG", "RDVI", "TDVI"])
        self.ok(at.button(key="nr_chip_ETF_FUNDS").click().run())  # all ticked: the chip unticks them
        self.assertEqual(at.session_state["nr_funds"], [])
        self.ok(at.button(key="nr_chip_CLOSED_END_FUNDS").click().run())
        self.pick(at, "KNG")
        self.assertIn("2 funds", self.shown(at))
        self.ok(at.button(key="nr_chip__clear").click().run())
        self.assertEqual(at.session_state["nr_funds"], [])

    def test_filter_hides_tiles_but_keeps_the_picks(self) -> None:
        at = self.pick(self.app(), "DOGG")
        self.ok(at.text_input(key="nr_fund_filter").input("kn").run())
        self.assertEqual([c.key for c in at.checkbox if c.key.startswith("nrf_")], ["nrf_KNG", "nrf_KNGIX"])
        self.pick(at, "KNG")
        self.assertEqual(at.session_state["nr_funds"], ["DOGG", "KNG"])  # DOGG is hidden, still picked
        request = new_run.build_request(CATALOG, new_run.Choices(funds=at.session_state["nr_funds"]))
        self.assertEqual(request.funds, ("DOGG", "KNG"))
        self.ok(at.text_input(key="nr_fund_filter").input("zzz").run())
        self.assertIn("No ticker matches", self.shown(at))

    def test_select_all_and_single_checks(self) -> None:
        at = self.pick(self.app(), "KNG")
        self.ok(at.checkbox(key="nr_rules_all").uncheck().run())
        self.assertEqual(at.session_state["nr_rules"], [])
        self.assertFalse(any(at.checkbox(key=f"nrr_{key}").value for key in CHECKS))
        self.assertIn("Pick at least one compliance check", self.shown(at))
        self.assertTrue(at.button(key="nr_run").disabled)
        self.ok(at.checkbox(key="nrr_real_estate_check").check().run())
        self.ok(at.checkbox(key="nrr_gics_compliance").check().run())
        self.assertFalse(at.checkbox(key="nr_rules_all").value)
        self.assertFalse(at.button(key="nr_run").disabled)
        run_dir = self.run_once(at)
        self.assertEqual(history.load_request(run_dir).compliance_tests, ("gics_compliance", "real_estate_check"))
        self.ok(at.checkbox(key="nr_rules_all").check().run())
        self.assertEqual(sorted(at.session_state["nr_rules"]), sorted(CHECKS))
        self.assertTrue(all(at.checkbox(key=f"nrr_{key}").value for key in CHECKS))
        self.assertEqual(new_run.rules_choice(CATALOG, at.session_state["nr_rules"]), (new_run.ALL_RULES, []))

    def test_preview_counts_and_sentence(self) -> None:
        at = self.pick(self.app(), "KNG", "TDVI", "SDVD")
        self.ok(at.button_group(key="nr_date_kind").set_value(new_run.RANGE).run())
        self.ok(at.date_input(key="nr_start").set_value(date(2026, 10, 1)).run())
        self.ok(at.date_input(key="nr_end").set_value(date(2026, 10, 31)).run())  # 22 weekdays (31st is a Saturday)
        self.ok(at.checkbox(key="nr_rules_all").uncheck().run())
        for key in CHECKS[:4]:
            self.ok(at.checkbox(key=f"nrr_{key}").check().run())
        page = self.shown(at)
        self.assertIn('<div class="v">22</div><div class="l">Days Selected</div>', page)
        self.assertIn('<div class="v">3</div><div class="l">Funds Selected</div>', page)
        self.assertIn('<div class="v">4</div><div class="l">Checks Selected</div>', page)
        self.assertIn("3 funds × 22 days × 4 checks", page)
        self.assertIn("264 total operations", page)
        self.assertIn("Estimated completion time", page)
        self.assertIn("rough estimate", page)  # no past runs yet
        self.ok(at.checkbox(key="nr_op_nav_reconciliation").check().run())
        page = self.shown(at)
        self.assertIn("3 funds × 22 days × 5 checks", page)
        self.assertIn("4 compliance rules + NAV recon", page)
        self.assertIn("RUN CHECKS & RECONS", self.run_label(at))

    def test_validation_errors_block_the_run(self) -> None:
        at = self.pick(self.app(), "KNG")
        self.ok(at.checkbox(key="nr_op_compliance").uncheck().run())
        self.assertIn("Pick at least one operation", self.shown(at))
        self.assertTrue(at.button(key="nr_run").disabled)
        self.ok(at.checkbox(key="nr_op_compliance").check().run())
        self.ok(at.checkbox(key="nrf_KNG").uncheck().run())
        self.assertIn("Pick at least one fund", self.shown(at))
        self.assertTrue(at.button(key="nr_run").disabled)
        self.pick(at, "KNG")
        self.ok(at.checkbox(key="nr_email").check().run())
        self.assertIn("confirmation to send notification emails", self.shown(at))
        self.assertTrue(at.button(key="nr_run").disabled)
        self.ok(at.checkbox(key="nr_email_ok").check().run())
        self.assertNotIn("fm-callout error", self.shown(at))
        self.assertFalse(at.button(key="nr_run").disabled)

    def test_range_and_no_trading_on_new_run(self) -> None:
        at = self.pick(self.app(), "KNG")
        self.ok(at.button_group(key="nr_date_kind").set_value(new_run.RANGE).run())
        self.ok(at.date_input(key="nr_start").set_value(SATURDAY).run())
        self.ok(at.date_input(key="nr_end").set_value(date(2026, 11, 1)).run())  # a weekend
        self.assertIn("no business days", self.shown(at))  # per day: no longer runs Friday silently
        self.ok(at.button_group(key="nr_date_mode").set_value(new_run.STACKED).run())
        self.assertIn("no business days", self.shown(at))
        self.assertFalse(at.checkbox(key="nr_pdf").value)  # stacked defaults to no PDF, like time_series
        # New run is end of day only: no run-type switch, a hint to the Trading Compliance page instead
        self.assertNotIn("nr_kind", [b.key for b in at.button_group])
        self.assertTrue([c for c in at.checkbox if c.key.startswith("nr_op_")])
        self.assertIn("has its own page: Trading Compliance", self.shown(at))
        self.assertNotIn("TRADING", self.run_label(at))

    def test_stacked_recon_without_daily_reports_is_refused(self) -> None:
        at = self.pick(self.app(), "KNG")
        self.ok(at.checkbox(key="nr_op_reconciliation").check().run())
        self.ok(at.button_group(key="nr_date_kind").set_value(new_run.RANGE).run())
        self.ok(at.date_input(key="nr_start").set_value(date(2026, 10, 29)).run())
        self.ok(at.date_input(key="nr_end").set_value(FRIDAY).run())
        self.ok(at.button_group(key="nr_date_mode").set_value(new_run.STACKED).run())
        self.assertIn("recon reports would be lost", self.shown(at))
        self.assertTrue(at.button(key="nr_run").disabled)
        self.ok(at.checkbox(key="nr_daily").check().run())
        self.assertNotIn("fm-callout error", self.shown(at))
        self.assertFalse(at.button(key="nr_run").disabled)

    def test_weekend_as_of_runs_and_says_friday(self) -> None:
        at = self.pick(self.app(), "KNG")
        self.ok(at.date_input(key="nr_as_of").set_value(SATURDAY).run())
        page = self.shown(at)
        self.assertIn("Saturday 2026-10-31 is not a business day: the run uses 2026-10-30", page)
        self.assertIn("compliance  ·  KNG  ·  2026-10-30  ·  all rules", page)
        self.assertIn('title="2026-10-30">2026-10-30</div>', page)  # the Days Selected card names the day run

    def test_trading_request_email_needs_the_confirmation(self) -> None:
        choices = new_run.Choices(kind=new_run.KIND_TRADING, as_of=FRIDAY)
        request = new_run.build_request(CATALOG, choices)
        self.assertEqual((request.send_email, request.output_tag), (False, "trading_compliance"))
        choices.send_email = True
        with self.assertRaises(api.InvalidRequest):
            new_run.build_request(CATALOG, choices)
        choices.email_confirmed = True
        self.assertTrue(new_run.build_request(CATALOG, choices).send_email)

    def test_email_confirmation_is_cleared_after_a_run(self) -> None:
        at = self.pick(self.app(), "KNG")
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

    def test_reset_restores_the_defaults(self) -> None:
        at = self.pick(self.app(), "KNG", "TDVI")
        self.ok(at.checkbox(key="nr_op_reconciliation").check().run())
        self.ok(at.checkbox(key="nr_rules_all").uncheck().run())
        self.ok(at.checkbox(key="nr_email").check().run())
        self.ok(at.button(key="nr_reset").click().run())
        self.assertEqual(at.session_state["nr_funds"], [])
        self.assertEqual(sorted(at.session_state["nr_rules"]), sorted(CHECKS))
        self.assertFalse(at.checkbox(key="nr_op_reconciliation").value)
        self.assertFalse(at.checkbox(key="nr_email").value)
        self.assertFalse(at.checkbox(key="nrf_KNG").value)

    def test_picks_survive_other_pages_and_hidden_widgets(self) -> None:
        at = self.pick(self.app(), "KNG")
        self.ok(at.checkbox(key="nr_op_reconciliation").check().run())
        self.ok(at.button_group(key="nr_date_kind").set_value(new_run.RANGE).run())
        self.ok(at.date_input(key="nr_start").set_value(date(2026, 10, 1)).run())
        self.ok(at.date_input(key="nr_end").set_value(FRIDAY).run())
        self.ok(at.checkbox(key="nr_rules_all").uncheck().run())
        self.ok(at.checkbox(key="nrr_gics_compliance").check().run())
        self.go(at, "Run history")
        self.go(at, "New run")
        self.go(at, "Trading Compliance")
        self.go(at, "New run")
        self.ok(at.button_group(key="nr_date_kind").set_value(new_run.AS_OF).run())
        self.ok(at.checkbox(key="nr_op_compliance").uncheck().run())  # hides the checks
        self.ok(at.button_group(key="nr_date_kind").set_value(new_run.RANGE).run())
        self.ok(at.checkbox(key="nr_op_compliance").check().run())
        self.assertEqual(at.session_state["nr_funds"], ["KNG"])
        self.assertTrue(at.checkbox(key="nrf_KNG").value)
        self.assertTrue(at.checkbox(key="nr_op_reconciliation").value)
        self.assertEqual((at.date_input(key="nr_start").value, at.date_input(key="nr_end").value),
                         (date(2026, 10, 1), FRIDAY))
        self.assertEqual(at.session_state["nr_rules"], ["gics_compliance"])
        self.assertTrue(at.checkbox(key="nrr_gics_compliance").value)

    def test_run_with_fake_worker_shows_downloads(self) -> None:
        at = self.pick(self.app(), "KNG", "RDVI")
        self.ok(at.text_input(key="nr_tag").input("ui test").run())
        run_dir = self.run_once(at)
        self.assertEqual(run_dir.parent, self.root.resolve())
        self.assertTrue(run_dir.name.endswith("-ui_test"))
        sent = history.load_request(run_dir)
        self.assertEqual((sent.funds, sent.send_email, sent.output_tag), (("KNG", "RDVI"), False, "ui_test"))
        self.assertEqual(Path(sent.output_dir), run_dir / runner.REPORTS)
        page = self.shown(at)
        self.assertIn(">OK</span>", page)  # the status banner
        labels = [b.proto.label for b in at.download_button]
        self.assertEqual(labels[0], "Download all (zip)")
        self.assertEqual(len(labels), 3)  # zip + xlsx + pdf
        self.assertTrue(any(label.endswith("Compliance (PDF)") for label in labels))
        self.assertTrue(any(b.proto.help.endswith(".pdf") for b in at.download_button[1:]))  # file name as tooltip
        self.assertFalse((run_dir / f"{run_dir.name}.zip").exists())  # built only when downloaded
        self.assertTrue(history.zip_bytes(run_dir).startswith(b"PK"))
        self.assertIn("Summary per day", page)  # no views.json from the fake worker: the summary table
        self.assertIn("Processed Funds", page)  # snake_case summary keys shown as words
        self.assertIn("Publishing is off", self.texts(at.caption))
        self.assertEqual(at.button_group(key="nr_pane").value, new_run.RESULTS)
        self.ok(at.button_group(key="nr_pane").set_value(new_run.GUIDE).run())
        self.assertIn("What will be checked", self.shown(at))
        self.assertFalse(at.download_button)

    def test_results_render_from_views_json(self) -> None:
        at = self.pick(self.app(), "KNG", "TDVI")
        run_dir = self.run_once(at)
        (run_dir / runner.VIEWS).write_text(json.dumps(ui_views_fixture.eod_views()), encoding="utf-8")
        self.ok(at.run())
        page = self.shown(at)
        self.assertIn("fm-matrix", page)  # the overview: funds × checks
        for label in ("Overall", "GICS", "40 Act", "IRS", "NAV", "Holdings"):
            self.assertIn(f">{label}</th>", page)
        self.assertEqual(at.button_group(key=f"nr-fund-{run_dir.name}").value, "KNG")  # the first failing fund
        self.assertIn("IRS Diversification", page)
        self.assertIn('<td class="v-fail nw">FAIL</td>', page)  # FAIL rows in soft red
        self.assertIn("N/A (not evaluated)", page)  # a row's reason under its value
        self.assertIn("&lt;META&gt;", page)  # values are escaped
        self.assertIn("IRS Condition 1: 90% of income qualifying.", page)  # the rule's footnotes
        self.assertIn("No shares outstanding data", page)  # fund notes
        self.assertIn("Industry exposure (12)", page)
        self.assertIn("Showing the first 2 of 12", page)
        self.assertIn("NAV Reconciliation", page)
        self.assertIn("Holdings Reconciliation", page)
        self.assertIn("80 holdings breaks", page)
        self.ok(at.button_group(key=f"nr-fund-{run_dir.name}").set_value("TDVI").run())
        page = self.shown(at)
        self.assertIn("TDVI: price feed was late", page)  # fund errors
        self.assertNotIn("NAV Reconciliation", page)  # no NAV view for TDVI
        self.assertGreaterEqual(len(at.download_button), 2)  # the downloads stay under the results

    def test_trading_results_render_from_views_json(self) -> None:
        at = self.pick(self.app(), "DOGG")
        run_dir = self.run_once(at)
        (run_dir / runner.VIEWS).write_text(json.dumps(ui_views_fixture.trading_views()), encoding="utf-8")
        self.ok(at.run())
        page = self.shown(at)
        self.assertIn("trading day", page)
        self.assertIn("Out Of Compliance", page)
        self.assertIn('class="hl-bad"', page)
        self.assertIn("Checks Before / After Trades", page)
        self.assertIn("Trade Activity", page)
        self.assertIn("7,240,344.52", page)
        side = at.button_group(key=f"nr-side-{run_dir.name}")
        self.assertEqual(side.value, "After trades (ex-post)")
        self.ok(side.set_value("Before trades (ex-ante)").run())
        self.assertIn('<td class="v-pass nw">PASS</td>', self.shown(at))

    def test_blank_tag_uses_the_cli_name(self) -> None:
        at = self.pick(self.app(), "KNG")
        self.assertIn("Output tag “compliance”", self.shown(at))
        run_dir = self.run_once(at)
        self.assertEqual(history.load_request(run_dir).output_tag, "compliance")

    def test_worker_errors_and_log_are_redacted(self) -> None:
        at = self.pick(self.app(), "KNG")
        self.ok(at.text_input(key="nr_tag").input("leak").run())
        self.run_once(at)
        shown = self.shown(at)
        self.assertIn("Could not parse URL", shown)
        self.assertIn("cannot connect to", shown)
        self.assertNotIn("Hunter2Secret", shown)

    def test_publish_copies_to_the_publish_root(self) -> None:
        publish = self.tmp / "published"
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_PUBLISH_ROOT": str(publish)}):
            at = self.pick(self.app(), "KNG")
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
        at = self.pick(self.app(), "KNG")
        self.ok(at.text_input(key="nr_tag").input("slow").run())
        self.ok(at.button(key="nr_run").click().run())
        run_dir = Path(at.session_state["nr_watch"])
        fresh = self.app()  # a second browser session reattaches through the lock file
        self.assertTrue(fresh.button(key="nr_run").disabled)
        self.assertTrue(fresh.checkbox(key="nrf_KNG").disabled)
        self.assertTrue(fresh.button(key="nr_chip_ETF_FUNDS").disabled)
        self.assertTrue(fresh.button(key="nr_reset").disabled)
        self.assertEqual(fresh.session_state["nr_watch"], str(run_dir))
        self.assertIn("Run in progress", self.shown(fresh))
        shown = self.shown(fresh)  # the progress comes first in the right pane, so RUN never leaves it off screen
        self.assertLess(shown.index("Run in progress"), shown.index("Configuration Preview"))
        self.assertNotIn("Pick at least one fund", self.shown(fresh))  # no validation noise while locked
        self.ok(fresh.button(key=f"cancel-{run_dir.name}").click().run())
        self.assertTrue(runner.wait(run_dir, timeout=TIMEOUT))
        self.ok(fresh.run())
        self.assertIn(">CANCELLED</span>", self.shown(fresh))
        self.assertTrue(fresh.button(key="nr_run").disabled)  # unlocked, but this session has no fund picked
        self.pick(fresh, "KNG")
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
        page = self.shown(at)
        self.assertIn("Run in progress", page)
        self.assertIn("fm-steps", page)  # the stage stepper
        self.assertTrue(at.get("progress"))
        runner.cancel(run_dir)
        self.assertTrue(runner.wait(run_dir, timeout=TIMEOUT))
        self.ok(at.run())
        self.assertIn("FINISHED", self.texts(at.markdown))


class OtherPagesTest(AppTestCase):
    def test_history_and_rerun(self) -> None:
        at = self.app()
        self.ok(at.button(key="nr_chip_ETF_FUNDS").click().run())
        self.ok(at.checkbox(key="nrf_TDVI").uncheck().run())
        self.ok(at.checkbox(key="nr_op_reconciliation").check().run())
        run_dir = self.run_once(at)
        sent = history.load_request(run_dir)
        self.assertEqual((sent.output_tag, sent.funds), ("eod", ("KNG", "RDVI")))  # the CLI preset's name
        self.ok(at.button(key="nr_reset").click().run())
        self.go(at, "Run history")
        self.assertEqual(at.title[0].value, "Run history")
        self.assertIn(f'<td class="">{run_dir.name}</td>', self.shown(at))
        self.ok(at.button(key="hist_rerun").click().run())
        self.assertEqual(at.title[0].value, "New run")
        self.assertEqual(at.session_state["nr_funds"], ["KNG", "RDVI"])
        self.assertTrue(at.checkbox(key="nrf_RDVI").value)
        self.assertFalse(at.checkbox(key="nrf_TDVI").value)
        self.assertTrue(at.checkbox(key="nr_op_reconciliation").value)
        self.assertFalse(at.checkbox(key="nr_email").value)
        self.assertIn("Loaded the settings of", self.shown(at))
        self.go(at, "Reports")
        self.assertEqual(at.title[0].value, "Reports")
        self.assertEqual(at.selectbox(key="rep_day").options, [str(new_run.default_as_of())])  # the report date
        self.assertGreaterEqual(len(at.download_button), 2)

    def test_rerun_of_a_group_request_ticks_its_funds(self) -> None:
        run_dir = self.root / "20261101-080000-cli-eod"
        run_dir.mkdir(parents=True)
        request = api.RunRequest(operations=("compliance",), start_date=FRIDAY, cohorts=("ETF_FUNDS",),
                                 exclude=("RDVI",), compliance_tests=("gics_compliance",))
        (run_dir / runner.REQUEST).write_text(api.request_to_json(request), encoding="utf-8")
        at = self.app()
        at.session_state["nr_prefill"] = str(run_dir)  # what the history page's Re-run callback sets
        self.ok(at.run())
        self.assertEqual(at.session_state["nr_funds"], ["KNG", "TDVI"])  # the group less the exclusion
        self.assertEqual(at.session_state["nr_rules"], ["gics_compliance"])
        self.assertFalse(at.checkbox(key="nr_rules_all").value)
        self.assertEqual(at.date_input(key="nr_as_of").value, FRIDAY)

    def test_history_rows_open_the_results_view(self) -> None:
        at = self.pick(self.app(), "KNG", "TDVI")
        run_dir = self.run_once(at)
        (run_dir / runner.VIEWS).write_text(json.dumps(ui_views_fixture.eod_views()), encoding="utf-8")
        self.go(at, "Run history")
        page = self.shown(at)
        self.assertIn("fm-matrix", page)
        self.assertIn("IRS Diversification", page)
        self.assertEqual(at.button_group(key=f"hist-fund-{run_dir.name}").value, "KNG")

    def test_publish_is_not_offered_for_an_unsuccessful_run(self) -> None:
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_PUBLISH_ROOT": str(self.tmp / "published")}):
            at = self.pick(self.app(), "KNG")
            run_dir = self.run_once(at)
            runner.update_meta(run_dir, cancelled=True)
            self.ok(at.run())
            self.assertFalse([b for b in at.button if b.key == f"nr-publish-{run_dir.name}"])
            self.assertIn("Publishing is only offered for runs that finished successfully.", self.texts(at.caption))

    def test_publish_confirmation_does_not_carry_over_to_another_run(self) -> None:
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_PUBLISH_ROOT": str(self.tmp / "published")}):
            at = self.pick(self.app(), "KNG")
            first = self.run_once(at)
            second = self.run_once(at)
            self.go(at, "Run history")
            self.assertEqual(at.selectbox(key="hist_pick").value, second.name)
            self.ok(at.checkbox(key=f"hist-publish-sure-{second.name}").check().run())
            self.assertFalse(at.button(key=f"hist-publish-{second.name}").disabled)
            self.ok(at.selectbox(key="hist_pick").set_value(first.name).run())
            self.assertFalse(at.checkbox(key=f"hist-publish-sure-{first.name}").value)
            self.assertTrue(at.button(key=f"hist-publish-{first.name}").disabled)

    def finished_run(self, name: str, days, fund_errors=None) -> Path:
        """A finished compliance run folder by hand: one Excel per day and a views.json day per day."""
        run_dir = self.root.resolve() / name
        (run_dir / runner.REPORTS).mkdir(parents=True)
        request = api.RunRequest(operations=("compliance",), start_date=date.fromisoformat(days[0]),
                                 end_date=date.fromisoformat(days[-1]), funds=("KNG", "TDVI"))
        (run_dir / runner.REQUEST).write_text(api.request_to_json(request), encoding="utf-8")
        files = []
        for day in days:
            path = run_dir / runner.REPORTS / f"compliance_report_{day}.xlsx"
            path.write_bytes(b"PK test")
            files.append({"day": day, "label": "compliance_excel", "path": str(path), "kind": "excel"})
        (run_dir / runner.RESULT).write_text(json.dumps({
            "exit_code": 0, "summaries": {d: {"total_funds": 2} for d in days}, "paths": {}, "files": files,
            "errors": []}), encoding="utf-8")
        (run_dir / runner.META).write_text(json.dumps({
            "user": "u", "started": "2026-11-03T10:00:00", "finished": "2026-11-03T10:00:20", "exit_code": 0,
            "summary": "compliance | KNG, TDVI"}), encoding="utf-8")
        base = ui_views_fixture.eod_views()["days"][ui_views_fixture.DAY]
        views = {}
        for day in days:
            views[day] = dict(json.loads(json.dumps(base)), date=day, fund_errors=(fund_errors or {}).get(day, {}))
        (run_dir / runner.VIEWS).write_text(json.dumps({"schema_version": 1, "days": views}), encoding="utf-8")
        return run_dir

    def test_reports_tables_follow_the_report_date(self) -> None:
        self.finished_run("20261103-100000-u-compliance", ["2026-10-29", "2026-10-30"])
        at = self.app()
        self.go(at, "Reports")
        heading = lambda: [h for h in self.shown(at).splitlines() if '<div class="fm-h2"><h2>Results' in h][0]  # noqa: E731
        self.assertEqual(at.selectbox(key="rep_day").value, "2026-10-30")
        self.assertIn("2026-10-30 · 2 funds", heading())
        self.ok(at.selectbox(key="rep_day").set_value("2026-10-29").run())
        self.assertIn("2026-10-29 · 2 funds", heading())  # the tables match the date picked ...
        labels = [b.proto.label for b in at.download_button][1:]
        self.assertTrue(labels and all(label.startswith("2026-10-29") for label in labels))  # ... and the downloads
        self.ok(at.selectbox(key="rep_day").set_value("2026-10-30").run())
        self.assertIn("2026-10-30 · 2 funds", heading())
        self.assertFalse([b for b in at.button_group if str(b.key).startswith("rep-day-")])  # one date picker only

    def test_fund_errors_and_reasons_from_views_are_redacted(self) -> None:
        secret = "mysql+pymysql://fmuser:Hunter2Secret@db.internal/fm"
        with mock.patch.dict(os.environ, {"DB_CONNECTION_STRING": secret}):
            run_dir = self.finished_run("20261103-100000-u-compliance", ["2026-10-30"],
                                        {"2026-10-30": {"KNG": [f"KNG: OperationalError connecting to {secret}"]}})
            views = json.loads((run_dir / runner.VIEWS).read_text(encoding="utf-8"))
            irs = views["days"]["2026-10-30"]["compliance"]["KNG"]["diversification_IRS_check"]
            irs.update(status="ERROR", reason=f"ERROR: could not reach {secret}")
            irs["rows"][0]["reason"] = f"password=Hunter2Secret at {secret}"
            (run_dir / runner.VIEWS).write_text(json.dumps(views), encoding="utf-8")
            at = self.app()
            self.go(at, "Run history")
            shown = self.shown(at)
        self.assertIn("OperationalError connecting to", shown)
        self.assertIn("could not reach", shown)
        self.assertNotIn("Hunter2Secret", shown)

    def test_password_gate(self) -> None:
        with mock.patch.dict(os.environ, {"FUNDMANAGER_UI_PASSWORD": "s3cret"}):
            at = self.app()
            self.assertEqual(len(at.text_input), 1)
            self.assertFalse(at.checkbox)
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
        self.assertFalse(at.checkbox)

    def test_catalog_failure_falls_back_to_tickers_and_is_cached(self) -> None:
        with mock.patch.object(api, "catalog", side_effect=RuntimeError("db down at mysql://secret")) as failing:
            at = self.app()
            errors = self.texts(at.error)
            self.assertIn("Could not load the fund groups", errors)
            self.assertNotIn("secret", errors)
            self.assertFalse([b for b in at.button if (b.key or "").startswith("nr_chip_") and b.key != "nr_chip__clear"])
            self.assertEqual(len([c for c in at.checkbox if c.key.startswith("nrf_")]), len(FUNDS))  # tickers
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

    def test_rules_choice(self) -> None:
        self.assertEqual(new_run.rules_choice(CATALOG, CHECKS), (new_run.ALL_RULES, []))
        self.assertEqual(new_run.rules_choice(CATALOG, ["real_estate_check", "gics_compliance"]),
                         (new_run.CHOOSE_RULES, ["gics_compliance", "real_estate_check"]))  # catalog order
        self.assertEqual(new_run.rules_choice(CATALOG, CHECKS + ["summary"])[0], new_run.CHOOSE_RULES)  # exact list


if __name__ == "__main__":
    unittest.main()
