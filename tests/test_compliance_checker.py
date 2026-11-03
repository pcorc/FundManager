"""ComplianceChecker rule logic on small hand-built holdings (no DB).

    python -m unittest discover tests -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("DB_CONNECTION_STRING", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from processing.fund import FundData, FundHoldings, FundSnapshot, build_fund_registry  # noqa: E402
from services.compliance_checker import ComplianceChecker  # noqa: E402


NO_OPTIONS = pd.DataFrame(columns=["optticker", "equity_underlying_ticker", "option_market_value"])


def _fund(name, *, equity, options=NO_OPTIONS, flex=None, ta=1_000.0, tna=1_000.0):
    fund = build_fund_registry()[name]
    holdings = FundHoldings(equity=equity, options=options, flex_options=flex)
    fund.data = FundData(current=FundSnapshot(vest=holdings, ta=ta, tna=tna, fund_name=name))
    return fund


def _equity(**columns):
    base = {"GICS_SECTOR_NAME": "Financials", "GICS_INDUSTRY_GROUP_NAME": "Financial Services",
            "GICS_INDUSTRY_NAME": "Capital Markets", "SECURITY_TYP": "Common Stock"}
    n = len(columns["eqyticker"])
    return pd.DataFrame({**{k: [v] * n for k, v in base.items()}, **columns})


class OptionsPerUnderlyingTests(unittest.TestCase):
    """D5: several options on one underlying must not repeat the issuer in the merged holdings."""

    def setUp(self):
        self.equity = _equity(eqyticker=["AAA", "BBB"], equity_market_value=[100.0, 50.0],
                              nav_shares=[10.0, 5.0], EQY_SH_OUT_million=[1_000.0, 1_000.0])
        self.options = pd.DataFrame({
            "optticker": ["AAA C1", "AAA C2", "AAA C3", "BBB C1"],
            "equity_underlying_ticker": ["AAA", "AAA", "AAA", "BBB"],
            "nav_shares": [-1, -1, -1, -1],
            "option_market_value": [-1.0, -2.0, -3.0, -4.0],
            "option_notional_value": [-10.0, -10.0, -10.0, -10.0],
        })

    def test_irs_holdings_one_row_per_issuer(self):
        fund = _fund("KNG", equity=self.equity, options=self.options)  # listed single-stock options, no flex
        result = ComplianceChecker({fund.name: fund}).diversification_IRS_check(fund)
        holdings = pd.DataFrame(result.calculations["holdings_df"])
        self.assertEqual(sorted(holdings["eqyticker"]), ["AAA", "BBB"])
        by_ticker = holdings.set_index("eqyticker")["net_market_value"]
        self.assertAlmostEqual(by_ticker["AAA"], 100.0 - 6.0)
        self.assertAlmostEqual(by_ticker["BBB"], 50.0 - 4.0)
        self.assertAlmostEqual(result.calculations["qualifying_assets_value"], 150.0 - 10.0)

    def test_irs_single_stock_flex_calls_summed_per_underlying(self):
        flex = pd.DataFrame({
            "optticker": ["2AAA C1", "2AAA C1", "2AAA P1"],
            "equity_underlying_ticker": ["AAA", "AAA", "AAA"],
            "nav_shares": [1, 1, -1],
            "option_market_value": [1.0, 1.0, -1.0],
            "option_notional_value": [20.0, 30.0, -50.0],
        })
        fund = _fund("DOGG", equity=self.equity, options=self.options, flex=flex)  # single-stock FLEX
        result = ComplianceChecker({fund.name: fund}).diversification_IRS_check(fund)
        holdings = pd.DataFrame(result.calculations["holdings_df"])
        self.assertEqual(sorted(holdings["eqyticker"]), ["AAA", "BBB"])
        self.assertAlmostEqual(holdings.set_index("eqyticker")["net_market_value"]["AAA"], 100.0 - 6.0 + 50.0)

    def test_12d3_combined_holdings_one_row_per_issuer(self):
        fund = _fund("KNG", equity=self.equity, options=self.options)
        result = ComplianceChecker({fund.name: fund}).twelve_d3_sec_biz(fund)
        combined = pd.DataFrame(result.calculations["combined_holdings"])
        self.assertEqual(list(combined["eqyticker"]), ["AAA", "BBB"])
        self.assertEqual(list(combined["net_market_value"]), [94.0, 46.0])


class RealOwnershipTests(unittest.TestCase):
    """D6: the ownership tests use the shares held, not zeros; real estate uses its real exposure."""

    def test_irs_10pct_uses_held_shares(self):
        # A is 60% of TNA (top 50%); B and C are the bottom 50%, and the fund owns 20% of C.
        equity = _equity(eqyticker=["A", "B", "C"], equity_market_value=[600.0, 250.0, 150.0],
                         nav_shares=[1.0, 1.0, 200.0], EQY_SH_OUT_million=[1_000.0, 1_000.0, 1_000.0])
        fund = _fund("RDVI", equity=equity)
        result = ComplianceChecker({fund.name: fund}).diversification_IRS_check(fund)
        self.assertFalse(result.details["condition_IRS_2_a_10"])
        self.assertEqual(result.calculations["bottom_50_exceeds_10_percent"], 1)
        self.assertFalse(result.is_compliant)

    def test_ex_post_uses_iiv_shares(self):
        equity = _equity(eqyticker=["A"], equity_market_value=[100.0], nav_shares=[10.0], iiv_shares=[60.0],
                         EQY_SH_OUT_million=[1_000.0], GICS_INDUSTRY_NAME=["Insurance"])
        fund = _fund("RDVI", equity=equity)
        eod = ComplianceChecker({fund.name: fund}).twelve_d2_insurance_cos(fund)
        ex_post = ComplianceChecker({fund.name: fund}, analysis_type="ex_post").twelve_d2_insurance_cos(fund)
        self.assertAlmostEqual(eod.calculations["max_ownership_pct"], 0.01)
        self.assertTrue(eod.is_compliant)
        self.assertAlmostEqual(ex_post.calculations["max_ownership_pct"], 0.06)
        self.assertFalse(ex_post.is_compliant)  # > RULE_12D2_INSURANCE_LIMIT (5%)

    def test_12d3_rule_1_uses_held_shares(self):
        equity = _equity(eqyticker=["GS", "MS"], equity_market_value=[10.0, 10.0], nav_shares=[70.0, 1.0],
                         EQY_SH_OUT_million=[1_000.0, 1_000.0])
        fund = _fund("RDVI", equity=equity)
        result = ComplianceChecker({fund.name: fund}).twelve_d3_sec_biz(fund)
        self.assertAlmostEqual(result.calculations["max_ownership_pct"], 0.07)
        self.assertFalse(result.details["rule_1_pass"])  # > RULE_12D3_EQUITY_LIMIT (5%)

    def test_real_estate_reports_real_exposure(self):
        equity = _equity(eqyticker=["O", "AAPL"], equity_market_value=[25.0, 75.0], nav_shares=[1.0, 1.0],
                         GICS_SECTOR_NAME=["Real Estate", "Information Technology"])
        fund = _fund("RDVI", equity=equity)
        result = ComplianceChecker({fund.name: fund}).real_estate_check(fund)
        self.assertAlmostEqual(result.calculations["real_estate_exposure"], 25.0)
        self.assertAlmostEqual(result.calculations["real_estate_percentage"], 0.25)
        self.assertFalse(result.is_compliant)
        self.assertFalse(result.details["real_estate_check_compliant"])

        fund.data.current.vest.equity["GICS_SECTOR_NAME"] = "Information Technology"
        result = ComplianceChecker({fund.name: fund}).real_estate_check(fund)
        self.assertTrue(result.is_compliant and result.details["real_estate_check_compliant"])


class MissingSharesOutstandingTests(unittest.TestCase):
    """D7: missing shares outstanding is flagged as N/A, never divided by 1."""

    def setUp(self):
        # A is 60% of TNA (top 50%); B (no shares outstanding) and C are the bottom 50% / 75% bucket.
        self.equity = _equity(eqyticker=["A", "B", "C"], equity_market_value=[600.0, 250.0, 40.0],
                              nav_shares=[1.0, 500.0, 1.0], EQY_SH_OUT_million=[1_000.0, 0.0, 1_000.0])

    def test_irs_10pct_missing_is_na_not_fail(self):
        fund = _fund("RDVI", equity=self.equity)
        result = ComplianceChecker({fund.name: fund}).diversification_IRS_check(fund)
        self.assertTrue(result.details["condition_IRS_2_a_10"])
        self.assertEqual(result.details["condition_IRS_2_a_10_status"], "N/A (no shares outstanding data)")
        self.assertEqual(result.calculations["missing_shares_out"], 1)
        self.assertEqual(result.calculations["missing_shares_out_tickers"], ["B"])

    def test_irs_10pct_real_breach_still_fails(self):
        self.equity.loc[2, "nav_shares"] = 200.0  # 20% of C
        fund = _fund("RDVI", equity=self.equity)
        result = ComplianceChecker({fund.name: fund}).diversification_IRS_check(fund)
        self.assertFalse(result.details["condition_IRS_2_a_10"])
        self.assertEqual(result.details["condition_IRS_2_a_10_status"], "FAIL")

    def test_40act_2b_missing_is_na_and_excluded(self):
        fund = _fund("RDVI", equity=self.equity, ta=10_000.0, tna=10_000.0)  # every issuer < 5%
        result = ComplianceChecker({fund.name: fund}).diversification_40act_check(fund)
        self.assertTrue(result.details["condition_40act_2b"])
        self.assertEqual(result.details["condition_40act_2b_status"], "N/A (no shares outstanding data)")
        self.assertEqual(result.calculations["missing_shares_out_tickers"], ["B"])
        self.assertAlmostEqual(result.calculations["max_ownership_float"], 0.001)  # not 500 / 1
        ownership = {e["eqyticker"]: e["vest_ownership_of_float"] for e in result.calculations["remaining_stocks_details"]}
        self.assertTrue(pd.isna(ownership["B"]))

        from reporting.compliance_reporting import _forty_act_row
        payload = {**result.details, "calculations": result.calculations}
        row = _forty_act_row("2026-10-30", fund.name, payload, result.calculations, 2)
        self.assertEqual(row["Condition 40 Act 2b"], "N/A (no shares outstanding data)")
        remaining = next(v for k, v in row.items() if k.startswith("Remaining Securities"))
        self.assertIn("(B, 500, 2.50%, N/A (no shares out))", remaining)

    def test_40act_2b_all_known_keeps_formula(self):
        self.equity.loc[1, "EQY_SH_OUT_million"] = 1_000_000.0
        fund = _fund("RDVI", equity=self.equity, ta=10_000.0, tna=10_000.0)
        result = ComplianceChecker({fund.name: fund}).diversification_40act_check(fund)
        self.assertEqual(result.details["condition_40act_2b_status"], "PASS")
        self.assertEqual(result.calculations["missing_shares_out"], 0)

        from reporting.compliance_reporting import _forty_act_row
        row = _forty_act_row("2026-10-30", fund.name, dict(result.details), result.calculations, 2)
        self.assertTrue(str(row["Condition 40 Act 2b"]).startswith("=IF("))

    def test_shares_outstanding_overrides_apply_to_irs_and_40act(self):
        # TDVI-like: TU, RCI and LOGI have no EQY_SH_OUT_million but a known override; GIB has neither.
        equity = _equity(eqyticker=["A", "TU", "RCI", "LOGI", "GIB"],
                         equity_market_value=[600.0, 150.0, 100.0, 80.0, 70.0],
                         nav_shares=[1.0, 1_374_157.0, 217_999.0, 23_158.0, 11_646.0],
                         EQY_SH_OUT_million=[1_000.0, None, None, None, None])
        fund = _fund("TDVI", equity=equity, ta=10_000.0, tna=1_000.0)
        checker = ComplianceChecker({fund.name: fund})
        irs = checker.diversification_IRS_check(fund)
        forty = checker.diversification_40act_check(fund)
        self.assertEqual(irs.calculations["missing_shares_out_tickers"], ["GIB"])
        self.assertEqual(forty.calculations["missing_shares_out_tickers"], ["GIB"])
        self.assertEqual(irs.details["condition_IRS_2_a_10_status"], "N/A (no shares outstanding data)")

        fund.data.current.vest.equity.loc[4, "EQY_SH_OUT_million"] = 100_000_000.0
        irs = ComplianceChecker({fund.name: fund}).diversification_IRS_check(fund)
        self.assertEqual(irs.details["condition_IRS_2_a_10_status"], "PASS")
        self.assertEqual(irs.calculations["missing_shares_out"], 0)


class TwelveDMissingSharesOutstandingTests(unittest.TestCase):
    """D6 review: 12d1/12d2/12d3 ownership flags missing shares outstanding as N/A instead of 0%."""

    def _holdings(self, industry, **extra):
        # X: 2% owned; Y: shares held, no shares outstanding.
        return _equity(eqyticker=["X", "Y"], equity_market_value=[10.0, 10.0], nav_shares=[20.0, 500.0],
                       EQY_SH_OUT_million=[1_000.0, None], GICS_INDUSTRY_NAME=[industry] * 2, **extra)

    def test_12d2_missing_is_na(self):
        from reporting.compliance_reporting import _12d2_row
        fund = _fund("RDVI", equity=self._holdings("Insurance"))
        result = ComplianceChecker({fund.name: fund}).twelve_d2_insurance_cos(fund)
        self.assertTrue(result.is_compliant)
        self.assertEqual(result.details["test_status"], "N/A (no shares outstanding data)")
        self.assertEqual(result.calculations["missing_shares_out_tickers"], ["Y"])
        self.assertAlmostEqual(result.calculations["max_ownership_pct"], 0.02)
        row = _12d2_row("2026-10-30", fund.name, dict(result.details), result.calculations, 2)
        self.assertEqual(row["Compliant"], "N/A (no shares outstanding data)")
        self.assertEqual(row["Insurance Holdings"], "X (2.00000%), Y (N/A (no shares out))")

        fund.data.current.vest.equity.loc[0, "nav_shares"] = 60.0  # 6% of X: a real breach still fails
        result = ComplianceChecker({fund.name: fund}).twelve_d2_insurance_cos(fund)
        self.assertFalse(result.is_compliant)
        self.assertEqual(result.details["test_status"], "FAIL")

    def test_12d3_rule_1_missing_is_na(self):
        from reporting.compliance_reporting import _12d3_row
        fund = _fund("RDVI", equity=self._holdings("Capital Markets"))
        result = ComplianceChecker({fund.name: fund}).twelve_d3_sec_biz(fund)
        self.assertTrue(result.details["rule_1_pass"])
        self.assertEqual(result.details["rule_1_status"], "N/A (no shares outstanding data)")
        self.assertEqual(result.calculations["missing_shares_out"], 1)
        row = _12d3_row("2026-10-30", fund.name, dict(result.details), result.calculations, 2)
        self.assertEqual(row["Rule 1 (<=5% equities)"], "N/A (no shares outstanding data)")
        self.assertEqual(row["12d3 Sec Biz Compliant"], "PASS" if result.is_compliant else "FAIL")
        self.assertIn("Y (N/A (no shares out))", row["SEC-Related Businesses (Shs Out %)"])

    def test_12d1_uses_held_shares_and_flags_missing(self):
        from reporting.compliance_reporting import _12d1_row
        fund = _fund("RDVI", equity=self._holdings("Capital Markets", REGULATORY_STRUCTURE=["RIC", "RIC"]),
                     ta=10_000.0)
        result = ComplianceChecker({fund.name: fund}).twelve_d1a_other_inv_cos(fund)
        self.assertAlmostEqual(result.calculations["ownership_pct_max"], 0.02)  # held shares, not "quantity"
        self.assertTrue(result.details["test_1_pass"])
        self.assertEqual(result.details["test_1_status"], "N/A (no shares outstanding data)")
        row = _12d1_row("2026-10-30", fund.name, dict(result.details), result.calculations, 2)
        self.assertEqual(row["Test 1 (<=3% Ownership)"], "N/A (no shares outstanding data)")
        self.assertEqual(row["12d1(a) Compliant"], "PASS")

        fund.data.current.vest.equity.loc[0, "nav_shares"] = 40.0  # 4% > 3%
        result = ComplianceChecker({fund.name: fund}).twelve_d1a_other_inv_cos(fund)
        self.assertFalse(result.details["test_1_pass"])
        self.assertEqual(result.details["test_1_status"], "FAIL")


class FortyActRegistrationTests(unittest.TestCase):
    """D8: only a fund registered diversified that is not operating as diversified fails."""

    DIVERSIFIED = dict(eqyticker=[f"T{i}" for i in range(25)], equity_market_value=[40.0] * 25,
                       nav_shares=[1.0] * 25, EQY_SH_OUT_million=[1_000_000.0] * 25)
    CONCENTRATED = dict(eqyticker=["A", "B", "C"], equity_market_value=[400.0, 300.0, 300.0],
                        nav_shares=[1.0] * 3, EQY_SH_OUT_million=[1_000_000.0] * 3)

    def _check(self, registration, holdings):
        fund = _fund("RDVI", equity=_equity(**holdings))
        fund.diversification_status = registration
        return ComplianceChecker({fund.name: fund}).diversification_40act_check(fund)

    def test_verdict_by_registration_and_operation(self):
        cases = [
            ("diversified", self.DIVERSIFIED, "diversified", True),
            ("diversified", self.CONCENTRATED, "non-diversified", False),
            ("non_diversified", self.DIVERSIFIED, "diversified", True),  # FGSI/RDVI/SDVD case
            ("non_diversified", self.CONCENTRATED, "non-diversified", True),
            (None, self.CONCENTRATED, "non-diversified", True),  # unknown registration
        ]
        for registration, holdings, status_today, compliant in cases:
            result = self._check(registration, holdings)
            self.assertEqual(result.calculations["fund_status_today"], status_today, registration)
            self.assertIs(result.is_compliant, compliant, (registration, status_today))
            self.assertEqual(result.details["actual_registration_status"], status_today)

    def test_registration_match_follows_checker(self):
        from reporting.compliance_reporting import ComplianceReportPDF, _forty_act_row
        for registration, holdings in [("diversified", self.DIVERSIFIED), ("diversified", self.CONCENTRATED),
                                       ("non_diversified", self.DIVERSIFIED), (None, self.CONCENTRATED)]:
            result = self._check(registration, holdings)
            payload = {**result.details, "calculations": result.calculations}
            # The PDF helper does not use the instance.
            self.assertIs(ComplianceReportPDF._registrations_match(None, payload), result.is_compliant, registration)
        row = _forty_act_row("2026-10-30", "RDVI", dict(result.details), result.calculations, 7)
        self.assertEqual(row["Registration Match"], '=IF(OR(D7="diversified",C7<>"diversified"),"PASS","FAIL")')



class SummaryMetricsTests(unittest.TestCase):
    """D2: summary metrics always run, and the reports select them under the summary_metrics key."""

    def setUp(self):
        self.fund = _fund("KNG", equity=_equity(eqyticker=["AAA"], equity_market_value=[100.0],
                                                nav_shares=[1.0], EQY_SH_OUT_million=[1_000.0]))
        self.checker = ComplianceChecker({self.fund.name: self.fund})

    def test_filtered_run_still_computes_summary_metrics(self):
        results = self.checker.run_compliance_tests(["real_estate_check"])[self.fund.name]
        self.assertEqual(sorted(results), ["real_estate_check", "summary_metrics"])
        self.assertAlmostEqual(results["summary_metrics"].calculations["equity_market_value"], 100.0)

    def test_summary_sheet_for_default_list_and_alias(self):
        import tempfile
        from config.run_configurations import FULL_COMPLIANCE_TESTS
        from reporting.compliance_reporting import ComplianceReport
        results = self.checker.run_compliance_tests(FULL_COMPLIANCE_TESTS)
        cases = [(FULL_COMPLIANCE_TESTS, True), (["summary", "real_estate_check"], True),
                 (["real_estate_check"], False), (None, True)]
        for tests, has_summary in cases:
            with tempfile.TemporaryDirectory() as tmp:
                report = ComplianceReport(results, "2026-10-30", tmp, test_functions=tests)
            self.assertEqual("Summary" in report.sheet_data, has_summary, tests)
            if has_summary:
                self.assertEqual(report.sheet_data["Summary"]["Equity"].tolist(), [100.0])

    def test_summary_sheet_option_mv_columns(self):
        """D4: the three Option MV columns read regular / flex / total option MV."""
        import tempfile
        from reporting.compliance_reporting import ComplianceReport
        results = self.checker.run_compliance_tests(["summary_metrics"])
        results[self.fund.name]["summary_metrics"].calculations.update(
            option_market_value=-3.0, flex_option_market_value=-2.0, total_option_market_value=-5.0)
        with tempfile.TemporaryDirectory() as tmp:
            sheet = ComplianceReport(results, "2026-10-30", tmp).sheet_data["Summary"]
        self.assertEqual(sheet.loc[0, ["Regular Option MV", "Flex Option MV", "Total Option MV"]].tolist(),
                         [-3.0, -2.0, -5.0])

    def test_pdf_gics_overview_without_summary_section(self):
        from unittest import mock
        from reporting.compliance_reporting import ComplianceReportPDF
        results = self.checker.run_compliance_tests(["gics_compliance"])
        with mock.patch.object(ComplianceReportPDF, "save_pdf"),                 mock.patch.object(ComplianceReportPDF, "_render_gics_compliance_overview") as gics,                 mock.patch.object(ComplianceReportPDF, "_render_metric_section") as section:
            ComplianceReportPDF({"2026-10-30": results}, "unused.pdf", allowed_tests=["gics_compliance"])
        gics.assert_called_once()
        section.assert_not_called()  # Summary Metrics was not selected

if __name__ == "__main__":
    unittest.main()
