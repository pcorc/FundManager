"""processing.api / run_modes / fund_manager / worker plumbing, with every DB touchpoint faked.

    python -m unittest discover tests -v
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# Anything that slips past the fakes fails fast instead of reaching the real DB (.env never overrides this).
os.environ["DB_CONNECTION_STRING"] = "sqlite:///:memory:"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import fund_definitions  # noqa: E402
from config.run_configurations import DIVERSIFICATION_TESTS, FULL_COMPLIANCE_TESTS, MODES  # noqa: E402
from processing import api, run_modes, worker  # noqa: E402
from processing.api import ReportFile, RunRequest, RunResult  # noqa: E402
from processing.fund_manager import FundManager, ProgressEvent, emit_progress, progress_for_day  # noqa: E402
from services.compliance_checker import RULE_KEYS, RULE_LABELS  # noqa: E402

DAY = date(2026, 10, 30)  # a Friday
FAKE_COHORTS = {"ALL_FUNDS": set(fund_definitions.FUND_DEFINITIONS), "ETF_FUNDS": {"KNG", "RDVI", "TDVI"},
                "CLOSED_END_FUNDS": {"FTMIX", "FTCSH"}, "PRIVATE_FUNDS": {"PD227"}, "VIT_AND_MUTUAL_FUNDS": set()}


def req(**kwargs) -> RunRequest:
    kwargs.setdefault("operations", ("compliance",))
    kwargs.setdefault("start_date", DAY)
    return RunRequest(**kwargs)


class CohortPatch(unittest.TestCase):
    """Cohort sets come from fixed fakes; load_cohorts never touches the DB."""

    def setUp(self):
        for patcher in (mock.patch.dict(api.COHORTS, FAKE_COHORTS), mock.patch.object(api, "load_cohorts")):
            patcher.start()
            self.addCleanup(patcher.stop)


class ValidateTests(CohortPatch):
    def assertInvalid(self, request, *fragments):
        with self.assertRaises(ValueError) as ctx:
            api.validate(request)
        for fragment in fragments:
            self.assertIn(fragment, str(ctx.exception))

    def test_valid_requests_pass_unchanged(self):
        for request in (
            req(),
            req(operations=api.OPERATIONS[:3], funds=("kng", "RDVI,TDVI"), compliance_tests=("all",)),
            req(compliance_tests=("diversification_IRS_check, prospectus_80pct_policy",)),
            req(compliance_tests=("summary_metrics",)),
            req(compliance_tests=("summary", "summary_metrics", "gics_compliance")),
            req(compliance_tests=()),
            req(operations=("trading_compliance",), end_date=date(2026, 11, 3)),
            req(date_mode="stacked", end_date=date(2026, 11, 3), cohorts=("ETF_FUNDS",), exclude=("KNG",)),
            req(funds=("ALL",)),
        ):
            self.assertIs(api.validate(request), request)

    def test_operations(self):
        self.assertInvalid(req(operations=("compliance", "holdings")), "'holdings'", "nav_reconciliation")
        self.assertInvalid(req(operations=()), "At least one operation", "trading_compliance")
        self.assertInvalid(req(operations="compliance"), "operations must be a list")
        self.assertInvalid(req(operations=("trading_compliance", "compliance")), "cannot be combined")

    def test_date_mode_and_dates(self):
        self.assertInvalid(req(date_mode="range"), "'range'", "per_day, stacked")
        self.assertInvalid(req(operations=("trading_compliance",), date_mode="stacked"), "only supports")
        self.assertInvalid(req(end_date=date(2026, 10, 29)), "before start_date")
        self.assertInvalid(req(start_date="2026-10-30"), "start_date must be a date")
        self.assertInvalid(req(previous_date="2026-10-29"), "previous_date must be a date")

    def test_rules(self):
        self.assertInvalid(req(compliance_tests=("gics_compliance, bogus_rule",)), "'bogus_rule'", "twelve_d3_sec_biz")
        self.assertInvalid(req(compliance_tests=("GICS_COMPLIANCE",)), "'GICS_COMPLIANCE'")
        self.assertInvalid(req(compliance_tests="gics_compliance"), "compliance_tests must be a list")

    def test_cohorts_and_funds(self):
        self.assertInvalid(req(cohorts=("ETFS",)), "'ETFS'", "ETF_FUNDS")
        self.assertInvalid(req(funds=("KNG", "XXX")), "'XXX'", "KNG")
        self.assertInvalid(req(exclude=("YYY",)), "'YYY'")
        self.assertInvalid(req(funds=("KNG",), exclude=("KNG",)), "Fund selection is empty")
        self.assertInvalid(req(exclude=tuple(fund_definitions.FUND_DEFINITIONS)), "Fund selection is empty")

    def test_inexact_tickers_with_cohorts_or_exclude(self):
        for request in (req(cohorts=("ETF_FUNDS",), funds=("kng",)), req(cohorts=("ETF_FUNDS",), funds=("KNG,RDVI",)),
                        req(cohorts=("ETF_FUNDS",), funds=("all",)), req(cohorts=("ETF_FUNDS",), exclude=("kng",)),
                        req(funds=("KNG", "RDVI"), exclude=("kng",)), req(funds=("ALL",), exclude=("KNG",)),
                        req(funds=("kng",), exclude=("RDVI",))):
            with self.assertRaises(api.InvalidRequest) as ctx:
                api.validate(request)
            self.assertIn("exact upper-case tickers", str(ctx.exception))
        api.validate(req(funds=("kng", "RDVI,TDVI", "all")))  # without cohorts/exclude these are split and upper-cased

    def test_errors_are_invalid_request(self):
        self.assertTrue(issubclass(api.InvalidRequest, ValueError))
        self.assertRaises(api.InvalidRequest, api.validate, req(operations=("bogus",)))

    def test_cohort_selection_resolving_to_empty_raises_in_run(self):
        request = req(cohorts=("ETF_FUNDS",), funds=("FTMIX",))
        api.validate(request)  # cohort members are only known after load_cohorts()
        with mock.patch.object(api, "initialize_database") as db:
            self.assertRaises(api.InvalidRequest, api.run, request)
            self.assertRaises(api.InvalidRequest, api.run, req(cohorts=("VIT_AND_MUTUAL_FUNDS",)))
        db.assert_not_called()


class SelectionTests(CohortPatch):
    def test_cohort_resolution_matches_build_run(self):
        direct = api._resolve_selection(req(cohorts=("ETF_FUNDS", "CLOSED_END_FUNDS")))
        preset = api.build_run("compliance", cohorts=[api.COHORTS["ETF_FUNDS"], api.COHORTS["CLOSED_END_FUNDS"]],
                               start_date="2026-10-30")
        self.assertEqual(direct.funds, preset.funds)
        self.assertEqual(direct.funds, ("FTCSH", "FTMIX", "KNG", "RDVI", "TDVI"))
        self.assertEqual((direct.cohorts, direct.exclude), ((), ()))

    def test_cohort_with_funds_and_exclude(self):
        resolved = api._resolve_selection(req(cohorts=("ETF_FUNDS",), funds=("TDVI", "KNG"), exclude=("KNG",)))
        self.assertEqual(resolved.funds, ("TDVI",))
        resolved = api._resolve_selection(req(cohorts=("ETF_FUNDS",), exclude=("KNG",)))
        self.assertEqual(resolved.funds, ("RDVI", "TDVI"))

    def test_exclude_without_selection_applies_to_all_funds(self):
        resolved = api._resolve_selection(req(exclude=("KNG",)))
        self.assertEqual(resolved.funds, tuple(sorted(set(fund_definitions.FUND_DEFINITIONS) - {"KNG"})))

    def test_exclude_with_funds(self):
        self.assertEqual(api._resolve_selection(req(funds=("RDVI", "KNG"), exclude=("KNG",))).funds, ("RDVI",))

    def test_plain_funds_untouched(self):
        request = req(funds=("RDVI", "kng"))
        self.assertIs(api._resolve_selection(request), request)

    def test_build_run_empty_selection_raises(self):
        with self.assertRaises(ValueError):
            api.build_run("eod", cohorts=[{"KNG"}], funds=["RDVI"], start_date="2026-10-30")
        with self.assertRaises(ValueError):
            api.build_run("eod", cohorts=[set()], start_date="2026-10-30")

    def test_build_run_cohort_with_inexact_funds_raises(self):
        for funds in (["kng"], ["KNG,RDVI"], ["all"], ["KNG", "rdvi"]):
            with self.assertRaises(api.InvalidRequest) as ctx:
                api.build_run("compliance", cohorts=[api.COHORTS["ETF_FUNDS"]], funds=funds, start_date="2026-10-30")
            self.assertIn("exact upper-case tickers", str(ctx.exception), funds)
        preset = api.build_run("compliance", cohorts=[api.COHORTS["ETF_FUNDS"]], funds=["RDVI", "KNG"],
                               start_date="2026-10-30")
        self.assertEqual(preset.funds, ("KNG", "RDVI"))
        self.assertEqual(api.build_run("compliance", funds=["rdvi", "KNG,TDVI"], start_date="2026-10-30").funds,
                         ("KNG,TDVI", "rdvi"))  # no cohort: kept raw and sorted, as before


class BuildRunTests(CohortPatch):
    def test_every_preset_validates(self):
        for mode in MODES:
            for tests in (None, [], FULL_COMPLIANCE_TESTS, DIVERSIFICATION_TESTS, ["all"]):
                request = api.build_run(mode, funds=["KNG"], start_date="2026-10-26", end_date="2026-10-30",
                                        compliance_tests=tests)
                self.assertIs(api.validate(request), request, (mode, tests))

    def test_preset_values_are_rule_keys(self):
        for mode, preset in MODES.items():
            self.assertTrue(set(preset["compliance_tests"]) <= set(RULE_KEYS), mode)

    def test_presets_match_direct_requests(self):
        common = dict(start_date=DAY, end_date=DAY, funds=("KNG", "RDVI"), output_dir=MODES["eod"]["output_dir"])
        expected = {
            "eod": req(operations=api.EOD_OPERATIONS, generate_daily_reports=True, output_tag="eod", **common),
            "nav": req(operations=("nav_reconciliation",), generate_daily_reports=True, output_tag="nav", **common),
            "trading_compliance": req(operations=("trading_compliance",), generate_daily_reports=True,
                                      output_tag="trading_compliance", **common),
            "time_series": req(operations=("compliance",), date_mode="stacked", create_pdf=False,
                               output_tag="time_series", **common),
        }
        for mode, request in expected.items():
            self.assertEqual(api.build_run(mode, funds=["RDVI", "KNG"], start_date="2026-10-30"), request, mode)

    def test_list_mode_tags_and_errors(self):
        requests = api.build_run(["compliance", "nav"], funds=["KNG"], start_date="2026-10-30", output_tag="pair")
        self.assertEqual([r.output_tag for r in requests], ["pair_compliance", "pair_nav"])
        self.assertRaises(ValueError, api.build_run, "bogus", start_date="2026-10-30")
        self.assertRaises(ValueError, api.build_run, "eod")

    def test_golden_equivalent_requests_resolve_like_their_presets(self):
        from tools.golden.scenarios import EQUIVALENT_REQUESTS, SCENARIOS
        for name, (base, kwargs) in EQUIVALENT_REQUESTS.items():
            kwargs = dict(kwargs, **{k: date.fromisoformat(kwargs[k]) for k in ("start_date", "end_date") if k in kwargs})
            direct = api._resolve_selection(api.validate(RunRequest(**kwargs)))
            spec = dict(SCENARIOS[base])
            spec["cohorts"] = [api.COHORTS[c] for c in spec.get("cohorts", [])] or None
            preset = api.build_run(spec.pop("mode"), **spec)
            self.assertEqual(direct.funds, preset.funds, name)
            self.assertEqual(api._tokens(direct.compliance_tests), api._tokens(preset.compliance_tests), name)


class TokensAndCatalogTests(CohortPatch):
    def test_tokens(self):
        self.assertEqual(api._tokens(["a, b", "c d", "ALL", "a"]), ["a", "b", "c", "d", "a"])
        self.assertEqual(api._tokens(["kng,kng rdvi"], upper=True), ["KNG", "RDVI"])
        self.assertEqual(api._tokens(["all", "All"]), [])
        self.assertEqual(api._tokens([]), [])

    def test_rule_catalog(self):
        catalog = api.rule_catalog()
        self.assertEqual([r["key"] for r in catalog], ["summary", *RULE_KEYS])
        self.assertEqual(catalog[1], {"key": "summary_metrics", "label": RULE_LABELS["summary_metrics"]})
        self.assertTrue(all(r["label"] for r in catalog))

    def test_catalog(self):
        catalog = api.catalog()
        self.assertEqual(catalog["rule_catalog"], api.rule_catalog())
        self.assertEqual(catalog["operations"], list(api.OPERATIONS))
        self.assertEqual(catalog["cohorts"]["ETF_FUNDS"], ["KNG", "RDVI", "TDVI"])
        self.assertEqual(catalog["date_modes"], ["per_day", "stacked"])

    def test_cli_test_choices(self):
        import main
        self.assertEqual(main._parse_cli(["compliance", "--fund", "KNG", "--start", "2026-10-30",
                                          "--test", "gics_compliance", "--test", "all"]).compliance_tests,
                         ("gics_compliance", "all"))
        self.assertEqual(main._parse_cli(["compliance", "--fund", "KNG", "--start", "2026-10-30",
                                          "--test", "summary", "--test", "summary_metrics"]).compliance_tests,
                         ("summary", "summary_metrics"))
        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            main._parse_cli(["compliance", "--start", "2026-10-30", "--test", "bogus"])


class ProgressPrimitivesTests(unittest.TestCase):
    def test_callback_errors_are_swallowed_at_debug(self):
        def boom(event):
            raise RuntimeError("ui went away")
        with self.assertLogs("fund_manager.progress", "DEBUG") as logs:
            emit_progress(boom, "db")
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].levelname, "DEBUG")
        emit_progress(None, "db")  # no callback: no-op

    def test_progress_for_day_stamps_only_missing_days(self):
        seen = []
        stamped = progress_for_day(seen.append, "2026-10-30")
        emit_progress(stamped, "load")
        emit_progress(stamped, "day", day="2026-10-29")
        self.assertEqual([e.day for e in seen], ["2026-10-30", "2026-10-29"])
        self.assertIsNone(progress_for_day(None, "x"))

    def test_event_is_frozen(self):
        event = ProgressEvent("fund", fund="KNG", index=1, total=2)
        with self.assertRaises(Exception):
            event.stage = "x"


class FundManagerProgressTests(unittest.TestCase):
    def test_fund_and_operation_events(self):
        store = SimpleNamespace(loaded_funds=["KNG", "RDVI"], date=DAY, fund_data={})
        registry = {"KNG": SimpleNamespace(name="KNG"), "RDVI": SimpleNamespace(name="RDVI")}
        seen = []
        with mock.patch.object(FundManager, "_populate_fund_data"), \
                mock.patch.object(FundManager, "_run_compliance", return_value={}), \
                mock.patch.object(FundManager, "_run_nav_reconciliation", return_value={}):
            results = FundManager(registry, store, on_progress=seen.append).run_daily_operations(
                ["compliance", "nav_reconciliation"])
        self.assertEqual(results.summary["processed_funds"], 2)
        self.assertEqual([(e.stage, e.fund, e.index, e.total, e.message) for e in seen], [
            ("fund", "KNG", 1, 2, "eod"), ("operation", "KNG", 1, 2, "compliance"),
            ("operation", "KNG", 1, 2, "nav_reconciliation"),
            ("fund", "RDVI", 2, 2, "eod"), ("operation", "RDVI", 2, 2, "compliance"),
            ("operation", "RDVI", 2, 2, "nav_reconciliation"),
        ])

    def test_default_has_no_callback(self):
        manager = FundManager({}, SimpleNamespace(loaded_funds=[]))
        self.assertIsNone(manager.on_progress)
        self.assertEqual(manager.run_daily_operations(["compliance"]).summary["total_funds"], 0)

    def test_prior_business_day(self):
        prior = FundManager._prior_business_day
        self.assertEqual(prior(date(2026, 11, 2)), date(2026, 10, 30))  # Monday -> Friday
        self.assertEqual(prior(date(2026, 11, 1)), date(2026, 10, 30))  # Sunday -> Friday
        self.assertEqual(prior(date(2026, 10, 30)), date(2026, 10, 29))
        self.assertEqual(prior(pd.Timestamp("2026-11-02")), date(2026, 10, 30))
        self.assertEqual(prior(datetime(2026, 11, 2, 9, 30)), date(2026, 10, 30))
        self.assertEqual(prior("2026-11-02"), date(2026, 10, 30))


class RunModesProgressTests(unittest.TestCase):
    def setUp(self):
        self.seen = []
        self.results = SimpleNamespace(fund_results={}, summary={})

    def stages(self):
        return [(e.stage, e.day, e.message) for e in self.seen]

    def test_run_operations_passes_callback(self):
        with mock.patch.object(run_modes, "FundManager") as manager:
            run_modes._run_operations({}, "store", ["compliance"], "eod", [], on_progress=self.seen.append)
        self.assertIs(manager.call_args.kwargs["on_progress"].__self__, self.seen)
        with mock.patch.object(run_modes, "FundManager") as manager:
            run_modes._run_operations({}, "store", ["compliance"], "eod", [])
        self.assertIsNone(manager.call_args.kwargs["on_progress"])

    def test_eod_mode(self):
        params = SimpleNamespace(operations=["compliance"], compliance_tests=[], trade_date=DAY, create_pdf=False,
                                 send_email=True)
        with mock.patch.object(run_modes, "_run_operations", return_value=self.results) as ops, \
                mock.patch.object(run_modes, "_build_eod_artefacts", return_value={}), \
                mock.patch.object(run_modes, "_build_eod_email_payload"), \
                mock.patch.object(run_modes, "EmailManager"):
            run_modes.run_eod_mode({}, SimpleNamespace(), params, Path("."), on_progress=self.seen.append)
        self.assertEqual(self.stages(), [("process", None, "eod"), ("reports", None, ""), ("email", None, "")])
        self.assertIs(ops.call_args.kwargs["on_progress"].__self__, self.seen)

    def test_eod_mode_without_email(self):
        params = SimpleNamespace(operations=["compliance"], compliance_tests=[], trade_date=DAY, create_pdf=False,
                                 send_email=False)
        with mock.patch.object(run_modes, "_run_operations", return_value=self.results), \
                mock.patch.object(run_modes, "_build_eod_artefacts", return_value={}):
            run_modes.run_eod_mode({}, SimpleNamespace(), params, Path("."), on_progress=self.seen.append)
        self.assertEqual([e.stage for e in self.seen], ["process", "reports"])

    def test_email_failures_are_recorded(self):
        params = SimpleNamespace(operations=["compliance"], compliance_tests=[], trade_date=DAY, create_pdf=False,
                                 send_email=True)
        errors = []
        failing = mock.Mock(**{"return_value.send.side_effect": RuntimeError("smtp down")})
        with mock.patch.object(run_modes, "_run_operations", return_value=self.results), \
                mock.patch.object(run_modes, "_build_eod_artefacts", return_value={}), \
                mock.patch.object(run_modes, "_build_eod_email_payload"), \
                mock.patch.object(run_modes, "EmailManager", failing), \
                self.assertLogs("processing.run_modes", "ERROR"):
            run_modes.run_eod_mode({}, SimpleNamespace(), params, Path("."), errors=errors)
            run_modes.run_eod_mode({}, SimpleNamespace(), params, Path("."))  # no list: only logged
        self.assertEqual(errors, ["EOD email notification failed: smtp down"])
        fund = SimpleNamespace(compliance_results={"fund_object": object()})
        results = SimpleNamespace(fund_results={"KNG": fund}, summary={})
        params = SimpleNamespace(compliance_tests=[], ex_post_date=DAY, create_pdf=False)
        with mock.patch.object(run_modes, "_run_operations", return_value=results), \
                mock.patch.object(run_modes, "build_trading_compliance_reports"), \
                mock.patch.object(run_modes, "build_compliance_reports"), \
                mock.patch.object(run_modes, "combine_trading_and_compliance_reports"), \
                mock.patch.object(run_modes, "_build_trading_email_payload"), \
                mock.patch.object(run_modes, "EmailManager", failing), \
                self.assertLogs("processing.run_modes", "ERROR"):
            run_modes.run_trading_mode({}, ex_ante_store=SimpleNamespace(), ex_post_store=SimpleNamespace(),
                                       params=params, output_dir=Path("."), errors=errors)
        self.assertEqual(errors[1:], ["Trading-analysis email notification failed: smtp down"])

    def test_trading_mode(self):
        fund = SimpleNamespace(compliance_results={"fund_object": object()})
        results = SimpleNamespace(fund_results={"KNG": fund}, summary={})
        params = SimpleNamespace(compliance_tests=[], ex_post_date=DAY, create_pdf=False)
        with mock.patch.object(run_modes, "_run_operations", return_value=results), \
                mock.patch.object(run_modes, "build_trading_compliance_reports"), \
                mock.patch.object(run_modes, "build_compliance_reports"), \
                mock.patch.object(run_modes, "combine_trading_and_compliance_reports"), \
                mock.patch.object(run_modes, "_build_trading_email_payload"), \
                mock.patch.object(run_modes, "EmailManager"):
            run_modes.run_trading_mode({}, ex_ante_store=SimpleNamespace(), ex_post_store=SimpleNamespace(),
                                       params=params, output_dir=Path("."), on_progress=self.seen.append)
        self.assertEqual(self.stages(), [("process", None, "ex_ante"), ("process", None, "ex_post"),
                                         ("reports", None, ""), ("email", None, "")])

    def test_range_mode(self):
        with mock.patch.object(run_modes, "BulkDataLoader"), \
                mock.patch.object(run_modes, "_run_operations", return_value=self.results) as ops, \
                mock.patch.object(run_modes, "_build_eod_artefacts", return_value={}), \
                mock.patch.object(run_modes, "build_compliance_reports_for_range"):
            run_modes.run_eod_range_mode(None, {}, start_date=date(2026, 10, 30), end_date=date(2026, 11, 2),
                                         operations=["compliance"], compliance_tests=[], output_dir=Path("."),
                                         on_progress=self.seen.append)
        friday, monday = "2026-10-30", "2026-11-02"
        self.assertEqual(self.stages(), [
            ("day", friday, ""), ("load", friday, "eod"), ("process", friday, "eod"), ("reports", friday, ""),
            ("day", monday, ""), ("load", monday, "eod"), ("process", monday, "eod"), ("reports", monday, ""),
            ("reports", None, "stacked"),
        ])
        self.assertEqual([(e.index, e.total) for e in self.seen if e.stage == "day"], [(1, 2), (2, 2)])
        emit_progress(ops.call_args.kwargs["on_progress"], "fund", fund="KNG")
        self.assertEqual(self.seen[-1].day, monday)


class FakeSession:
    closed = False

    def close(self):
        self.closed = True


class RunTests(CohortPatch):
    """api.run end to end with the DB, loader and run modes replaced by fakes."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.workbook = self.tmp / "reconciliation_summary_2026-10-30.xlsx"
        self.workbook.write_bytes(b"x")
        self.compliance = self.tmp / "compliance_results_2026-10-30.xlsx"
        self.compliance.write_bytes(b"y")
        self.seen = []
        self.email_errors = []
        self.registry = {name: SimpleNamespace(name=name) for name in fund_definitions.FUND_DEFINITIONS}
        for patcher in (
            mock.patch.object(api, "initialize_database", side_effect=lambda: (FakeSession(), None)),
            mock.patch.object(fund_definitions, "load_cohorts_from_db"),
            mock.patch.object(api, "build_fund_registry", side_effect=lambda *a: dict(self.registry)),
            mock.patch.object(api, "BulkDataLoader"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def eod_mode(self, registry, data_store, params, output_dir, *, output_tag=None, on_progress=None, errors=None):
        emit_progress(on_progress, "process", message="eod")
        errors.extend(self.email_errors)
        artefacts = {
            "compliance": SimpleNamespace(excel_path=str(self.compliance), pdf_path=str(self.tmp / "missing.pdf")),
            "reconciliation": SimpleNamespace(holdings=SimpleNamespace(excel_path=str(self.workbook)),
                                              nav=SimpleNamespace(excel_path=str(self.workbook)),
                                              combined_reconciliation_pdf=None),
        }
        return SimpleNamespace(summary={"processed_funds": len(registry)}), artefacts

    def test_per_day_events_files_and_bundle(self):
        request = req(operations=api.EOD_OPERATIONS, cohorts=("ETF_FUNDS",), exclude=("TDVI",),
                      start_date=date(2026, 10, 30), end_date=date(2026, 11, 2), output_dir=str(self.tmp))
        with mock.patch.object(api, "run_eod_mode", side_effect=self.eod_mode) as eod:
            result = api.run(request, on_progress=self.seen.append)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.errors, [])
        self.assertEqual(list(eod.call_args.args[0]), ["KNG", "RDVI"])
        self.assertEqual(eod.call_args.kwargs["params"].previous_trade_date, date(2026, 10, 30))  # Monday's T-1
        self.assertEqual([(e.stage, e.day, e.index, e.total) for e in self.seen][:6], [
            ("day", "2026-10-30", 1, 2), ("db", "2026-10-30", None, None), ("load", "2026-10-30", None, 2),
            ("process", "2026-10-30", None, None), ("done", "2026-10-30", 1, 2), ("day", "2026-11-02", 2, 2),
        ])
        self.assertEqual(self.seen[-1].message, "completed")
        self.assertEqual(set(result.paths), {"2026-10-30", "2026-11-02"})
        self.assertIn("compliance_pdf", result.paths["2026-10-30"])  # paths keep everything reported
        self.assertEqual(result.files[:3], [
            ReportFile("2026-10-30", "compliance_excel", str(self.compliance), "excel"),
            ReportFile("2026-10-30", "holdings_reconciliation_excel", str(self.workbook), "excel"),
            ReportFile("2026-10-30", "nav_reconciliation_excel", str(self.workbook), "excel"),
        ])
        self.assertEqual(len(result.files), 6)
        with zipfile.ZipFile(api.bundle(result, self.tmp / "out" / "bundle.zip")) as archive:
            self.assertEqual(sorted(archive.namelist()), [self.compliance.name, self.workbook.name])

    def test_processing_error_is_recorded(self):
        with mock.patch.object(api, "build_fund_registry", side_effect=RuntimeError("boom")), \
                self.assertLogs("fund_manager.main", "ERROR"):
            result = api.run(req(funds=("KNG",), output_dir=str(self.tmp)), on_progress=self.seen.append)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.errors, ["2026-10-30: Processing failed: boom"])
        self.assertEqual([(e.stage, e.message) for e in self.seen],
                         [("day", ""), ("db", ""), ("done", "failed")])

    def test_trading_day(self):
        def trading_mode(registry, *, ex_ante_store, ex_post_store, params, output_dir, output_tag, on_progress,
                         errors):
            emit_progress(on_progress, "process", message="ex_ante")
            errors.append("Trading-analysis email notification failed: smtp down")
            report = SimpleNamespace(excel_path=str(self.compliance), pdf_path=None)
            return (SimpleNamespace(summary={}), SimpleNamespace(summary={}),
                    SimpleNamespace(report=report, ex_post_compliance=None, combined_report=None))
        with mock.patch.object(api, "run_trading_mode", side_effect=trading_mode):
            result = api.run(req(operations=("trading_compliance",), funds=("KNG",), output_dir=str(self.tmp)),
                             on_progress=self.seen.append)
        self.assertEqual([(e.stage, e.message) for e in self.seen if e.stage in ("load", "process")],
                         [("load", "ex_ante"), ("load", "ex_post"), ("process", "ex_ante")])
        self.assertEqual(set(result.summaries), {"2026-10-30 ex-ante", "2026-10-30 ex-post"})
        self.assertEqual([f.label for f in result.files], ["trading_compliance_excel"])
        self.assertEqual((result.exit_code, result.errors),
                         (0, ["2026-10-30: Trading-analysis email notification failed: smtp down"]))

    def test_email_error_is_recorded_without_failing_the_run(self):
        self.email_errors = ["EOD email notification failed: smtp down"]
        with mock.patch.object(api, "run_eod_mode", side_effect=self.eod_mode):
            result = api.run(req(funds=("KNG",), output_dir=str(self.tmp)))
        self.assertEqual((result.exit_code, result.errors),
                         (0, ["2026-10-30: EOD email notification failed: smtp down"]))

    def test_stacked(self):
        range_results = run_modes.RangeRunResults(
            results_by_date={DAY: SimpleNamespace(summary={"processed_funds": 1})}, daily_artefacts={},
            stacked_compliance=SimpleNamespace(excel_path=str(self.compliance), pdf_path=None))
        with mock.patch.object(api, "run_eod_range_mode", return_value=range_results) as range_mode:
            result = api.run(req(funds=("KNG",), date_mode="stacked", output_dir=str(self.tmp)),
                             on_progress=self.seen.append)
        self.assertEqual(range_mode.call_args.kwargs["operations"], ["compliance"])
        self.assertEqual(result.files, [ReportFile("stacked", "compliance_excel", str(self.compliance), "excel")])
        self.assertEqual([(e.stage, e.day) for e in self.seen], [("db", "stacked"), ("done", "stacked")])

    def test_stacked_error_is_recorded(self):
        with mock.patch.object(api, "run_eod_range_mode", side_effect=RuntimeError("bad day")), \
                self.assertLogs("fund_manager.main.range", "ERROR"):
            result = api.run(req(funds=("KNG",), date_mode="stacked", output_dir=str(self.tmp)))
        self.assertEqual((result.exit_code, result.errors), (1, ["stacked: Processing failed: bad day"]))

    def test_failing_callback_does_not_break_the_run(self):
        def boom(event):
            raise RuntimeError("ui went away")
        with mock.patch.object(api, "run_eod_mode", side_effect=self.eod_mode):
            result = api.run(req(funds=("KNG",), output_dir=str(self.tmp)), on_progress=boom)
        self.assertEqual(result.exit_code, 0)

    def test_invalid_request_never_reaches_the_db(self):
        self.assertRaises(ValueError, api.run, req(funds=("NOPE",)))
        api.initialize_database.assert_not_called()


class BundleTests(unittest.TestCase):
    def test_unique_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "a").mkdir()
            (tmp / "b").mkdir()
            for path in (tmp / "a" / "r.xlsx", tmp / "b" / "r.xlsx", tmp / "a" / "s.pdf"):
                path.write_bytes(b"z")
            result = RunResult(files=[
                ReportFile("d1", "x_excel", str(tmp / "a" / "r.xlsx"), "excel"),
                ReportFile("d1", "y_excel", str(tmp / "a" / "r.xlsx"), "excel"),
                ReportFile("d2", "x_excel", str(tmp / "b" / "r.xlsx"), "excel"),
                ReportFile("d1", "s_pdf", str(tmp / "a" / "s.pdf"), "pdf"),
                ReportFile("d1", "gone_pdf", str(tmp / "a" / "gone.pdf"), "pdf"),
            ])
            with zipfile.ZipFile(api.bundle(result, tmp / "z.zip")) as archive:
                self.assertEqual(archive.namelist(), ["r.xlsx", "d2/r.xlsx", "s.pdf"])


class JsonAndWorkerTests(unittest.TestCase):
    def test_request_round_trip(self):
        request = req(operations=("compliance", "nav_reconciliation"), end_date=date(2026, 11, 2), funds=("KNG",),
                      cohorts=("ETF_FUNDS",), exclude=("RDVI",), compliance_tests=("gics_compliance",),
                      date_mode="stacked", previous_date=date(2026, 10, 29), custodian_date=DAY, output_tag="t")
        text = api.request_to_json(request)
        self.assertEqual(json.loads(text)["start_date"], "2026-10-30")
        self.assertEqual(api.request_from_json(text), request)
        self.assertEqual(api.request_from_json(api.request_to_json(req())), req())

    def test_request_from_json_minimal_and_errors(self):
        parsed = api.request_from_json('{"operations": ["compliance"], "start_date": "2026-10-30", "funds": "KNG",'
                                       ' "exclude": null}')
        self.assertEqual(parsed, req(funds=("KNG",)))
        for text, fragment in (('[]', "JSON object"), ('{"operations": ["compliance"]}', "start_date"),
                               ('{"operations": ["compliance"], "start_date": "2026-10-30", "fund": ["KNG"]}',
                                "Unknown request field")):
            with self.assertRaises(ValueError) as ctx:
                api.request_from_json(text)
            self.assertIn(fragment, str(ctx.exception))
        self.assertRaises(ValueError, api.request_from_json, '{"operations": [], "start_date": "10/30/2026"}')
        for extra in (dict(funds=5), dict(operations=5), dict(compliance_tests=True), dict(funds=[5]),
                      dict(cohorts=[["ETF_FUNDS"]]), dict(create_pdf="no"), dict(output_dir=3), dict(output_tag=1)):
            text = json.dumps(dict({"operations": ["compliance"], "start_date": "2026-10-30"}, **extra))
            with self.assertRaises(ValueError) as ctx:
                api.request_from_json(text)
            self.assertIn(f"{next(iter(extra))} must be", str(ctx.exception))

    def run_worker(self, request_text, fake_run=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        request_path = Path(tmp.name) / "job.json"
        request_path.write_text(request_text, encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(worker, "run", side_effect=fake_run) as run, \
                mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            code = worker.main([str(request_path)])
            self.assertIs(sys.stdout, stdout)
        result = json.loads((Path(tmp.name) / "job.result.json").read_text(encoding="utf-8"))
        return code, stdout.getvalue(), stderr.getvalue(), result, run

    def test_worker_round_trip(self):
        def fake_run(request, on_progress):
            print("stray print from a report builder")
            on_progress(ProgressEvent("day", day="2026-10-30", index=1, total=1))
            on_progress(ProgressEvent("fund", day="2026-10-30", fund="KNG", index=1, total=1, message="eod"))
            return RunResult(exit_code=0, summaries={"2026-10-30": {"processed_funds": 1, "when": DAY}},
                             paths={"2026-10-30": {"compliance_excel": "c.xlsx"}},
                             files=[ReportFile("2026-10-30", "compliance_excel", "c.xlsx", "excel")])
        code, out, err, result, run = self.run_worker(api.request_to_json(req(funds=("KNG",))), fake_run)
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args[0], req(funds=("KNG",)))
        lines = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(lines[1], {"stage": "fund", "day": "2026-10-30", "fund": "KNG", "index": 1, "total": 1,
                                    "message": "eod"})
        self.assertEqual(len(lines), 2)
        self.assertIn("stray print", err)
        self.assertEqual(result, {
            "exit_code": 0, "summaries": {"2026-10-30": {"processed_funds": 1, "when": "2026-10-30"}},
            "paths": {"2026-10-30": {"compliance_excel": "c.xlsx"}},
            "files": [{"day": "2026-10-30", "label": "compliance_excel", "path": "c.xlsx", "kind": "excel"}],
            "errors": [],
        })

    def test_worker_invalid_request(self):
        code, out, _, result, run = self.run_worker('{"operations": ["holdings"], "start_date": "2026-10-30"}')
        self.assertEqual((code, out), (2, ""))
        self.assertIn("'holdings'", result["errors"][0])
        run.assert_not_called()

    def test_worker_bad_field_type_writes_a_result(self):
        code, out, _, result, run = self.run_worker('{"operations": ["compliance"], "start_date": "2026-10-30", '
                                                    '"funds": 5}')
        self.assertEqual((code, out, result["exit_code"]), (2, "", 2))
        self.assertIn("funds must be a list", result["errors"][0])
        run.assert_not_called()

    def test_worker_selection_empty_after_cohort_load(self):
        def fake_run(request, on_progress):
            raise api.InvalidRequest("Fund selection is empty (cohorts=['ETF_FUNDS'], funds=['FTMIX'], exclude=[])")
        request = api.request_to_json(req(cohorts=("ETF_FUNDS",), funds=("FTMIX",)))
        code, _, err, result, _ = self.run_worker(request, fake_run)
        self.assertEqual(code, 2)
        self.assertEqual(result["errors"], ["Invalid request: Fund selection is empty (cohorts=['ETF_FUNDS'], "
                                            "funds=['FTMIX'], exclude=[])"])
        self.assertNotIn("Traceback", err)

    def test_worker_run_exception(self):
        def fake_run(request, on_progress):
            raise RuntimeError("loader exploded")
        code, _, _, result, _ = self.run_worker(api.request_to_json(req()), fake_run)
        self.assertEqual(code, 1)
        self.assertEqual(result["errors"], ["RuntimeError: loader exploded"])


if __name__ == "__main__":
    unittest.main()
