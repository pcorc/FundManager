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


if __name__ == "__main__":
    unittest.main()
