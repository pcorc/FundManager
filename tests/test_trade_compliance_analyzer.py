"""Trading compliance analyzer: ex-ante vs ex-post holdings, statuses, one-sided funds.

    python -m unittest discover tests -v
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("DB_CONNECTION_STRING", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from processing import run_modes  # noqa: E402
from processing.fund import Fund, FundData, FundHoldings, FundSnapshot  # noqa: E402
from reporting.trade_compliance_analyzer import TradingComplianceAnalyzer  # noqa: E402

DAY = date(2026, 10, 30)


def _fund_data(equity=None, options=None, treasury=None, tna=1_000_000.0):
    snapshot = FundSnapshot(vest=FundHoldings(equity=equity, options=options, treasury=treasury), tna=tna, ta=tna)
    return FundData(current=snapshot, previous=snapshot)


def _equity(price, nav_shares=100.0, iiv_shares=100.0, trade_rebal=0.0):
    return pd.DataFrame({"eqyticker": ["AAA"], "nav_shares": [nav_shares], "iiv_shares": [iiv_shares],
                         "trade_rebal": [trade_rebal], "price": [price]})


def _options(ticker, quantity, price, trade_rebal):
    return pd.DataFrame({"optticker": [ticker], "nav_shares": [quantity], "iiv_shares": [quantity],
                         "trade_rebal": [trade_rebal], "price": [price]})


def _analyze(post_fund, ante_fund=None, ante=None, post=None):
    info = {"fund_object": post_fund}
    if ante_fund is not None:
        info["ex_ante_fund_object"] = ante_fund
    ante = {"rule": {"is_compliant": True}} if ante is None else ante
    post = {"rule": {"is_compliant": True}} if post is None else post
    analyzer = TradingComplianceAnalyzer({"KNG": ante} if ante else {}, {"KNG": post} if post else {},
                                         DAY, {"KNG": info})
    return analyzer.analyze()["funds"]["KNG"]


class ExAnteHoldingsTests(unittest.TestCase):
    def setUp(self):
        self.post = Fund("KNG", {})
        self.post.data = _fund_data(equity=_equity(price=11.0),
                                    options=_options("NEW 11/06/26 C10", -2.0, 0.5, -2.0))
        self.ante = Fund("KNG", {})
        self.ante.data = _fund_data(equity=_equity(price=10.0),
                                    options=_options("OLD 10/30/26 C10", -3.0, 0.25, None))

    def test_ex_ante_market_value_comes_from_the_ex_ante_fund(self):
        summary = _analyze(self.post, self.ante)["trade_info"]["asset_trade_summary"]["equity"]
        self.assertEqual((summary["ex_ante_market_value"], summary["ex_post_market_value"]), (1000.0, 1100.0))
        self.assertEqual(summary["market_value_delta"], 100.0)
        # No trades: the price move is not trade value.
        self.assertEqual(summary["trade_value"], 0.0)

    def test_closed_ex_ante_option_shows_as_a_closing_trade(self):
        activity = _analyze(self.post, self.ante)["trade_info"]["trade_activity"]["options"]
        self.assertEqual([(t["ticker"], t["quantity"], t["market_value"]) for t in activity["buys"]],
                         [("OLD 10/30/26 C10", 3.0, 75.0)])
        self.assertEqual([t["ticker"] for t in activity["sells"]], ["NEW 11/06/26 C10"])

    def test_without_an_ex_ante_copy_both_sides_read_the_ex_post_fund(self):
        summary = _analyze(self.post)["trade_info"]["asset_trade_summary"]["equity"]
        self.assertEqual(summary["ex_ante_market_value"], summary["ex_post_market_value"])

    def test_share_change_without_trade_rebal_is_valued_at_ex_post_prices(self):
        self.post.data = _fund_data(equity=_equity(price=11.0, iiv_shares=150.0))
        summary = _analyze(self.post, self.ante)["trade_info"]["asset_trade_summary"]["equity"]
        self.assertEqual(summary["trade_value"], 550.0)

    def test_treasury_prices_are_per_100_face(self):
        bills = pd.DataFrame({"cusip": ["912797TM9"], "nav_shares": [0.0], "iiv_shares": [200_000.0],
                              "trade_rebal": [200_000.0], "price": [99.5]})
        self.post.data = _fund_data(treasury=bills)
        trade_info = _analyze(self.post, self.ante)["trade_info"]
        summary = trade_info["asset_trade_summary"]["treasury"]
        self.assertEqual((summary["trade_value"], summary["ex_post_market_value"]), (199_000.0, 199_000.0))
        self.assertEqual(trade_info["trade_activity"]["treasury"]["buys"][0]["market_value"], 199_000.0)


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.fund = Fund("KNG", {})

    def test_numpy_bool_results_resolve_to_pass_fail(self):
        checks = _analyze(self.fund, ante={"rule": SimpleNamespace(is_compliant=np.bool_(True))},
                          post={"rule": {"is_compliant": np.bool_(False)}})["checks"]
        self.assertEqual((checks["rule"]["status_before"], checks["rule"]["status_after"]), ("PASS", "FAIL"))

    def test_fund_on_one_side_only(self):
        result = _analyze(self.fund, ante={}, post={"rule": {"is_compliant": False}})
        self.assertEqual(result["checks"]["rule"]["status_before"], "UNKNOWN")
        self.assertEqual(result["violations_after"], 1)


class TradingModeExAnteSnapshotTests(unittest.TestCase):
    def test_ex_ante_fund_keeps_the_ex_ante_data(self):
        fund = Fund("KNG", {})
        ante_data, post_data = _fund_data(), _fund_data()

        def run_operations(registry, store, operations, analysis_type, tests, *, on_progress=None):
            fund.data = ante_data if analysis_type == "ex_ante" else post_data
            payload = {"rule": {"is_compliant": True}, "fund_object": fund}
            return SimpleNamespace(fund_results={"KNG": SimpleNamespace(compliance_results=payload)}, summary={})

        params = SimpleNamespace(compliance_tests=[], ex_post_date=DAY, create_pdf=False, send_email=False)
        with mock.patch.object(run_modes, "_run_operations", side_effect=run_operations), \
                mock.patch.object(run_modes, "build_trading_compliance_reports") as build, \
                mock.patch.object(run_modes, "build_compliance_reports"), \
                mock.patch.object(run_modes, "combine_trading_and_compliance_reports"), \
                mock.patch.object(run_modes, "EmailManager"):
            run_modes.run_trading_mode({"KNG": fund}, ex_ante_store=SimpleNamespace(),
                                       ex_post_store=SimpleNamespace(), params=params, output_dir=Path("."))
        info = build.call_args.kwargs["traded_funds_info"]["KNG"]
        self.assertIs(info["fund_object"].data, post_data)
        self.assertIs(info["ex_ante_fund_object"].data, ante_data)


if __name__ == "__main__":
    unittest.main()
