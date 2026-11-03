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


def _fund(name, *, equity, options=None, flex=None, ta=1_000.0, tna=1_000.0):
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


if __name__ == "__main__":
    unittest.main()
