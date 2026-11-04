"""reporting.trading_dashboard on small hand-built funds (the real analyzer, no DB, no files).

    python -m unittest discover tests -p "test_trading_dashboard.py" -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("DB_CONNECTION_STRING", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from sqlalchemy import create_engine, event, text  # noqa: E402

from processing.fund import Fund, FundData, FundHoldings, FundSnapshot  # noqa: E402
from processing.staged_trades import staged_trade_counts  # noqa: E402
from reporting import result_views as rv  # noqa: E402
from reporting import trading_dashboard as td  # noqa: E402
from reporting.trade_compliance_analyzer import TradingComplianceAnalyzer  # noqa: E402

DAY = "2026-10-30"
IRS, ACT40, GICS = "diversification_IRS_check", "diversification_40act_check", "gics_compliance"


def _frame(ticker_column, rows):
    """rows: (ticker, nav_shares, iiv_shares, trade_rebal, price[, extra columns])."""
    records = []
    for ticker, nav, iiv, rebal, price, *extra in rows:
        record = {ticker_column: ticker, "nav_shares": nav, "iiv_shares": iiv, "trade_rebal": rebal, "price": price}
        if extra:
            record.update(extra[0])
        records.append(record)
    return pd.DataFrame(records)


def _fund(name, *, equity=None, options=None, treasury=None, flex=None, cash=0.0, tna=0.0, prev_cash=0.0,
          prev_tna=0.0):
    fund = Fund(name, {})
    current = FundSnapshot(vest=FundHoldings(equity=equity, options=options, treasury=treasury, flex_options=flex),
                           cash=cash, tna=tna, ta=tna)
    previous = FundSnapshot(cash=prev_cash, tna=prev_tna, ta=prev_tna)
    fund.data = FundData(current=current, previous=previous)
    return fund


def _rule(status, rows=(), details=()):
    return {"title": "T", "label": "L", "status": status, "reason": None, "rows": list(rows),
            "details": list(details), "tables": [], "notes": []}


def _comparison(funds):
    """Run the real analyzer: funds = {name: (ante Fund | None, post Fund)}."""
    info = {name: ({"fund_object": post, "ex_ante_fund_object": ante} if ante is not None
                   else {"fund_object": post}) for name, (ante, post) in funds.items()}
    results = {name: {"rule": {"is_compliant": True}} for name in funds}
    return TradingComplianceAnalyzer(results, results, DAY, info).analyze()


def _dashboard(fund_objects, ante_rules=None, post_rules=None, **kwargs):
    comparison = _comparison(fund_objects)
    ante_rules = ante_rules if ante_rules is not None else {name: {IRS: _rule("PASS")} for name in fund_objects}
    post_rules = post_rules if post_rules is not None else {name: {IRS: _rule("PASS")} for name in fund_objects}
    kwargs.setdefault("funds", list(fund_objects))
    return td.build_dashboard(comparison, ante_rules, post_rules, day=DAY, **kwargs)


class CashMathsTests(unittest.TestCase):
    """Equity, option premium (buy pays, sell/write receives) and treasury (price per 100 face) cash maths."""

    def setUp(self):
        self.ante = _fund("KNG",
                          equity=_frame("eqyticker", [("AAA", 100.0, 100.0, 0.0, 10.0), ("BBB", 80.0, 80.0, 0.0, 20.0)]),
                          options=_frame("optticker", [("OLD 11/20/26 C10", -3.0, -3.0, None, 0.25,
                                                        {"maturity": date(2026, 11, 20)})]),
                          cash=999.0, tna=2_000_000.0, prev_cash=50_000.0, prev_tna=1_000_000.0)
        self.post = _fund("KNG",
                          equity=_frame("eqyticker", [("AAA", 100.0, 200.0, 100.0, 10.0),
                                                      ("BBB", 80.0, 30.0, -50.0, 20.0)]),
                          options=_frame("optticker", [("NEW 11/06/26 C10", -2.0, -2.0, -2.0, 1.5,
                                                        {"maturity": date(2026, 11, 6)})]),
                          treasury=_frame("cusip", [("912797XX1", 0.0, 10_000.0, 10_000.0, 99.0)]),
                          cash=999.0, tna=2_000_000.0)
        self.dash = _dashboard({"KNG": (self.ante, self.post)})
        self.fund = self.dash["funds"][0]

    def test_current_cash_is_the_ex_ante_t_minus_1_custodian_cash(self):
        cash = self.fund["cash"]
        self.assertEqual((cash["current"], cash["source"], cash["date"]), (50_000.0, "ex_ante_previous", "2026-10-29"))
        self.assertEqual(cash["custodian_t"], 999.0)
        self.assertEqual(self.dash["cash_date"], "2026-10-29")

    def test_post_trade_cash_signs(self):
        assets = self.fund["assets"]
        self.assertEqual((assets["equity"]["buy_value"], assets["equity"]["sell_value"]), (1000.0, 1000.0))
        # Write 2 contracts at 1.50 receives 300; buying back the closed short (3 x 0.25 x 100) pays 75.
        self.assertEqual((assets["options"]["buy_value"], assets["options"]["sell_value"]), (75.0, 300.0))
        self.assertEqual(assets["options"]["cash_effect"], 225.0)
        self.assertEqual(assets["treasury"]["buy_value"], 9_900.0)  # 10,000 face at 99 per 100
        self.assertEqual(assets["treasury"]["cash_effect"], -9_900.0)
        cash = self.fund["cash"]
        self.assertEqual(cash["post"], 50_000.0 - 1000.0 + 1000.0 + 225.0 - 9_900.0)
        self.assertEqual(cash["change"], cash["post"] - cash["current"])
        self.assertAlmostEqual(cash["post_pct_tna"], cash["post"] / 1_000_000.0)
        self.assertFalse(cash["negative"])
        self.assertEqual((cash["excluded"], self.fund["warnings"]), ([], []))

    def test_waterfall_runs_from_current_to_post_cash(self):
        steps = self.fund["waterfall"]
        self.assertEqual([s["key"] for s in steps], ["current_cash", "equity_buys", "equity_sells", "options_net",
                                                     "treasury_net", "post_cash"])
        self.assertEqual([s["value"] for s in steps[1:5]], [-1000.0, 1000.0, 225.0, -9_900.0])
        self.assertEqual(steps[4]["running"], steps[5]["value"])
        self.assertEqual(steps[5]["value"], self.fund["cash"]["post"])

    def test_stats_match_the_trading_report_figures(self):
        info = _comparison({"KNG": (self.ante, self.post)})["funds"]["KNG"]["trade_info"]
        equity = self.fund["assets"]["equity"]
        self.assertEqual(equity["gross_value"], info["asset_trade_summary"]["equity"]["trade_value"])
        self.assertEqual(equity["gross_pct_tna"], info["asset_trade_summary"]["equity"]["pct_of_tna"])
        self.assertEqual(self.fund["tna"], 1_000_000.0)  # the analyzer's (T-1) TNA
        self.assertEqual((equity["buy_quantity"], equity["sell_quantity"], equity["net_value"]), (100.0, 50.0, 0.0))
        self.assertEqual(self.fund["net_value"], self.fund["buy_value"] - self.fund["sell_value"])
        self.assertEqual(self.fund["gross_value"], 1000.0 + 1000.0 + 375.0 + 9_900.0)
        self.assertTrue(all(a["staged"] for a in self.fund["assets"].values()))

    def test_report_trade_value_of_an_unstaged_asset_class_is_flagged(self):
        # DOGG on 2026-11-02: staged treasury rows, equity "trade value" from nav_shares -> iiv_shares only.
        post = _fund("DOGG", equity=pd.DataFrame({"eqyticker": ["AAA"], "nav_shares": [10.0], "iiv_shares": [8.0],
                                                  "price": [5.0]}),
                     treasury=_frame("cusip", [("T1", 0.0, 100.0, 100.0, 100.0)]))
        fund = _dashboard({"DOGG": (_fund("DOGG", prev_cash=1000.0, prev_tna=1.0), post)})["funds"][0]
        equity = fund["assets"]["equity"]
        self.assertEqual((equity["staged"], equity["gross_value"], equity["cash_effect"]), (False, 10.0, 0.0))
        self.assertEqual(fund["cash"]["post"], 900.0)
        self.assertIn("Equity: the trading report shows a trade value of $10", fund["warnings"][0])

    def test_blotter(self):
        groups = {g["asset_class"]: g for g in self.fund["blotter"]}
        self.assertEqual(list(groups), ["equity", "options", "treasury"])
        options = {t["ticker"]: t for t in groups["options"]["trades"]}
        new = options["NEW 11/06/26 C10"]
        self.assertEqual((new["side"], new["quantity"], new["price"], new["multiplier"], new["value"]),
                         ("SELL", 2.0, 1.5, 100.0, 300.0))
        self.assertEqual((new["before_qty"], new["effect"], new["flags"], new["in_cash"]), (0.0, "open", [], True))
        old = options["OLD 11/20/26 C10"]
        self.assertEqual((old["side"], old["before_qty"], old["effect"]), ("BUY", -3.0, "close"))
        aaa = groups["equity"]["trades"][0]
        self.assertEqual((aaa["ticker"], aaa["before_qty"], aaa["after_qty"], aaa["ex_post_qty"], aaa["effect"]),
                         ("AAA", 100.0, 200.0, 200.0, "increase"))
        self.assertAlmostEqual(aaa["pct_tna"], 0.001)
        self.assertEqual(self.fund["trade_count"], 5)

    def test_json_safe(self):
        json.dumps(rv.to_jsonable(self.dash), allow_nan=False)


class ExcludedTradeTests(unittest.TestCase):
    def test_unpriced_unchanged_and_duplicate_trades(self):
        ante = _fund("DOGG", equity=_frame("eqyticker", [("MRK", 100.0, 100.0, 0.0, 10.0)] * 2),
                     treasury=_frame("cusip", [("912797TM9", 1000.0, 1000.0, 1000.0, 99.0)]), prev_cash=100.0,
                     prev_tna=10_000.0)
        post = _fund("DOGG",
                     equity=_frame("eqyticker", [("MRK", 100.0, 60.0, -40.0, 10.0)] * 2),  # a duplicated view row
                     options=_frame("optticker", [("MRK US 11/06/26 C150", -4.0, -4.0, -4.0, 0.0)]),
                     treasury=_frame("cusip", [("912797TM9", 1000.0, 1000.0, 1000.0, 99.0)]))
        fund = _dashboard({"DOGG": (ante, post)})["funds"][0]
        groups = {g["asset_class"]: g["trades"] for g in fund["blotter"]}
        self.assertEqual([t["flags"] for t in groups["equity"]], [[], ["duplicate"]])
        self.assertEqual(groups["options"][0]["flags"], ["no_price"])
        self.assertEqual(groups["options"][0]["value"], 400.0)  # the analyzer's quantity x multiplier fallback
        self.assertEqual(groups["treasury"][0]["flags"], ["restates_position"])
        cash = fund["cash"]
        # Duplicates stay in (as in the report, with a warning); unpriced / unchanged-position trades are left out.
        self.assertEqual(cash["post"], 100.0 + 800.0)
        self.assertEqual(sorted((e["asset_class"], e["reason"]) for e in cash["excluded"]),
                         [("options", "no_price"), ("treasury", "restates_position")])
        self.assertEqual(cash["excluded_value"], 400.0 + 990.0)
        self.assertEqual(fund["assets"]["treasury"]["buy_value"], 990.0)  # still in the trade stats
        self.assertEqual(len(fund["warnings"]), 3)
        self.assertIn("duplicate", fund["warnings"][0])

    def test_a_buy_with_an_unchanged_ex_post_share_column_stays_in(self):
        # Real data: ex-post iiv_shares need not equal nav_shares + trade_rebal (PD227 buys 100 MA, 403 -> 403).
        ante = _fund("PD227", equity=_frame("eqyticker", [("MA", 403.0, 403.0, 0.0, 600.0)]), prev_cash=1.0,
                     prev_tna=1.0)
        post = _fund("PD227", equity=_frame("eqyticker", [("MA", 403.0, 403.0, 100.0, 600.0)]))
        trade = _dashboard({"PD227": (ante, post)})["funds"][0]["blotter"][0]["trades"][0]
        self.assertEqual((trade["flags"], trade["in_cash"], trade["after_qty"], trade["ex_post_qty"]),
                         ([], True, 503.0, 403.0))

    def test_negative_cash_and_flex_warning(self):
        flex = pd.DataFrame({"optticker": ["2 XYZ"], "nav_shares": [1.0], "iiv_shares": [2.0], "trade_rebal": [1.0]})
        ante = _fund("KNG", prev_cash=500.0, prev_tna=10_000.0)
        post = _fund("KNG", equity=_frame("eqyticker", [("AAA", 0.0, 100.0, 100.0, 10.0)]), flex=flex)
        dash = _dashboard({"KNG": (ante, post)})
        fund = dash["funds"][0]
        self.assertEqual((fund["cash"]["post"], fund["cash"]["negative"]), (-500.0, True))
        self.assertTrue(any("FLEX" in w for w in fund["warnings"]))
        self.assertEqual(dash["kpis"]["funds_negative_cash"], 1)
        self.assertEqual(fund["blotter"][0]["trades"][0]["effect"], "open")


class MissingSideTests(unittest.TestCase):
    def test_cash_falls_back_to_the_ex_ante_t_snapshot(self):
        ante = _fund("KNG", cash=700.0, tna=9_000.0)  # no T-1 custodian data
        post = _fund("KNG", equity=_frame("eqyticker", [("AAA", 0.0, 10.0, 10.0, 10.0)]))
        cash = _dashboard({"KNG": (ante, post)})["funds"][0]["cash"]
        self.assertEqual((cash["current"], cash["source"], cash["date"], cash["post"]),
                         (700.0, "ex_ante_current", DAY, 600.0))

    def test_without_the_ex_ante_copy(self):
        post = _fund("KNG", equity=_frame("eqyticker", [("AAA", 5.0, 10.0, 5.0, 10.0)]), cash=300.0, tna=1.0,
                     prev_cash=999.0, prev_tna=1.0)
        fund = _dashboard({"KNG": (None, post)})["funds"][0]
        self.assertEqual((fund["cash"]["current"], fund["cash"]["source"]), (300.0, "ex_post_current"))
        trade = fund["blotter"][0]["trades"][0]
        self.assertEqual((trade["before_qty"], trade["effect"], trade["in_cash"]), (None, None, True))

    def test_no_cash_loaded(self):
        post = _fund("KNG", equity=_frame("eqyticker", [("AAA", 0.0, 10.0, 10.0, 10.0)]))
        dash = _dashboard({"KNG": (_fund("KNG"), post)})
        fund = dash["funds"][0]
        self.assertEqual((fund["cash"]["current"], fund["cash"]["post"], fund["cash"]["negative"]), (None, None, False))
        self.assertIn("No custodian cash", fund["warnings"][0])
        self.assertIsNone(fund["waterfall"][-1]["value"])
        self.assertEqual((dash["totals"]["cash"], dash["totals"]["cash_funds"]), (None, 0))

    def test_no_trade_and_unavailable_funds(self):
        traded = _fund("KNG", equity=_frame("eqyticker", [("AAA", 0.0, 10.0, 10.0, 10.0)]))
        quiet = _fund("FDND", equity=_frame("eqyticker", [("AAA", 10.0, 10.0, 0.0, 10.0)]))
        dash = _dashboard({"KNG": (_fund("KNG"), traded), "FDND": (_fund("FDND"), quiet)},
                          ante_rules={"KNG": {IRS: _rule("PASS")}, "FDND": {IRS: _rule("FAIL")}},
                          post_rules={"KNG": {IRS: _rule("PASS")}, "FDND": {IRS: _rule("FAIL")}},
                          funds=["FDND", "KNG", "TDVI"], fund_errors={"TDVI": ["ex-post: boom"]})
        self.assertEqual([f["fund"] for f in dash["funds"]], ["KNG"])
        self.assertEqual(dash["no_trade_funds"], [{"fund": "FDND", "impact": "STILL_FAILS", "status_after": "FAIL",
                                                   "failed_after": [IRS], "report_trade_value": 0.0}])
        self.assertEqual(dash["unavailable_funds"], [{"fund": "TDVI", "errors": ["ex-post: boom"]}])
        kpis = dash["kpis"]
        self.assertEqual((kpis["funds_selected"], kpis["funds_with_trades"], kpis["funds_without_trades"],
                          kpis["funds_unavailable"]), (3, 1, 1, 1))

    def test_report_trade_value_without_staged_rows_is_not_a_proposed_trade(self):
        # No trade_rebal: the analyzer values the nav_shares -> iiv_shares change (FTMIX on 2026-10-30).
        post = _fund("FTMIX", equity=pd.DataFrame({"eqyticker": ["AAA"], "nav_shares": [10.0], "iiv_shares": [12.0],
                                                   "price": [5.0]}))
        dash = _dashboard({"FTMIX": (_fund("FTMIX"), post)})
        self.assertEqual(dash["funds"], [])
        self.assertEqual(dash["no_trade_funds"][0]["report_trade_value"], 10.0)

    def test_empty_comparison(self):
        dash = td.build_dashboard(None, {}, {}, day=DAY)
        self.assertEqual((dash["funds"], dash["matrix"]["rules"], dash["rules_at_risk"]), ([], [], []))
        self.assertEqual((dash["kpis"]["funds_with_trades"], dash["totals"]["tna"]), (0, 0))


class TransitionTests(unittest.TestCase):
    def test_transition_codes(self):
        cases = {("PASS", "PASS"): "OK", ("PASS", "FAIL"): "BREAKS", ("FAIL", "PASS"): "CURES",
                 ("FAIL", "FAIL"): "STILL_FAILS", (None, "FAIL"): "FAILS", ("N/A", "FAIL"): "FAILS",
                 ("PASS", "ERROR"): "ERROR", ("ERROR", "FAIL"): "ERROR", ("N/A", "N/A"): "N/A",
                 (None, "PASS"): "N/A", ("PASS", None): "N/A"}
        for (before, after), code in cases.items():
            with self.subTest(before=before, after=after):
                self.assertEqual(td.transition(before, after), code)

    def setUp(self):
        trade = _frame("eqyticker", [("AAA", 0.0, 10.0, 10.0, 10.0)])
        funds = {name: (_fund(name), _fund(name, equity=trade)) for name in ("A", "B", "C", "D")}
        irs_before = [{"label": "Condition 2a 50%", "value": "12.10%"}, {"label": "Overall Status", "value": "PASS",
                                                                          "status": "PASS"}]
        irs_after = [{"label": "Condition 2a 50%", "value": "14.30%"}, {"label": "Overall Status", "value": "FAIL",
                                                                         "status": "FAIL"}]
        ante = {"A": {IRS: _rule("PASS", irs_before), ACT40: _rule("PASS"), GICS: _rule("N/A")},
                "B": {IRS: _rule("FAIL"), ACT40: _rule("PASS")},
                "C": {IRS: _rule("FAIL"), ACT40: _rule("FAIL")},
                "D": {IRS: _rule("PASS"), ACT40: _rule("PASS")}}
        post = {"A": {IRS: _rule("FAIL", irs_after), ACT40: _rule("PASS"), GICS: _rule("N/A"),
                      "summary_metrics": _rule(None, [{"label": "Cash", "value": "1"}])},
                "B": {IRS: _rule("PASS"), ACT40: _rule("PASS")},
                "C": {IRS: _rule("FAIL"), ACT40: _rule("PASS")},
                "D": {IRS: _rule("PASS"), ACT40: _rule("PASS")}}
        ante["A"]["summary_metrics"] = _rule(None, [{"label": "Cash", "value": "2"}])
        rules = {GICS: {"title": "GICS Diversification", "label": "GICS Concentration"},
                 ACT40: {"title": "40 Act", "label": "40 Act Diversification"},
                 IRS: {"title": "IRS", "label": "IRS Diversification"}}
        self.dash = _dashboard(funds, ante, post, rules=rules)

    def test_fund_impacts_and_order(self):
        self.assertEqual([(f["fund"], f["impact"]) for f in self.dash["funds"]],
                         [("A", "BREAKS"), ("C", "STILL_FAILS"), ("B", "CURES"), ("D", "COMPLIANT")])
        kpis = self.dash["kpis"]
        self.assertEqual((kpis["funds_breaking"], kpis["funds_still_failing"], kpis["funds_curing"],
                          kpis["funds_compliant"]), (1, 1, 1, 1))
        self.assertEqual((kpis["rule_breaks"], kpis["rule_cures"], kpis["rule_still_fails"]), (1, 2, 1))

    def test_matrix(self):
        matrix = self.dash["matrix"]
        self.assertEqual([(r["key"], r["changes_or_fails"]) for r in matrix["rules"]],
                         [(GICS, False), (ACT40, True), (IRS, True)])  # the DayView's rule order
        self.assertEqual([r["fund"] for r in matrix["rows"]], ["A", "C", "B", "D"])
        self.assertEqual(matrix["rows"][0]["cells"][IRS], {"before": "PASS", "after": "FAIL", "transition": "BREAKS"})
        self.assertNotIn("summary_metrics", matrix["rows"][0]["cells"])

    def test_rules_at_risk(self):
        risk = {r["rule"]: r for r in self.dash["rules_at_risk"]}
        self.assertEqual([r["rule"] for r in self.dash["rules_at_risk"]], [IRS, ACT40])
        self.assertEqual((risk[IRS]["breaks"], risk[IRS]["still_fails"], risk[IRS]["cures"]), (["A"], ["C"], ["B"]))
        self.assertEqual((risk[ACT40]["breaks"], risk[ACT40]["cures"]), ([], ["C"]))
        self.assertEqual(risk[IRS]["label"], "IRS Diversification")

    def test_rule_diffs_mark_changed_values(self):
        fund = self.dash["funds"][0]
        diffs = fund["rules"]
        self.assertEqual([(d["rule"], d["transition"], d["unchanged"]) for d in diffs],
                         [(IRS, "BREAKS", False), (ACT40, "OK", True), (GICS, "N/A", True)])
        self.assertEqual(diffs[0]["rows"][0], {"label": "Condition 2a 50%", "before": "12.10%", "after": "14.30%",
                                               "changed": True})
        self.assertEqual(diffs[0]["rows"][1]["status_before"], "PASS")
        self.assertEqual(diffs[0]["changed_rows"], 2)
        self.assertEqual(fund["metrics"], [{"label": "Cash", "before": "2", "after": "1", "changed": True}])


class TradingViewIntegrationTests(unittest.TestCase):
    def _results(self, **funds):
        return SimpleNamespace(fund_results={name: SimpleNamespace(compliance_results=data, errors=[])
                                             for name, data in funds.items()}, summary={})

    def test_day_view_carries_the_dashboard(self):
        ante, post = _fund("KNG", prev_cash=10.0, prev_tna=100.0), \
            _fund("KNG", equity=_frame("eqyticker", [("AAA", 0.0, 1.0, 1.0, 5.0)]))
        comparison = _comparison({"KNG": (ante, post)})
        view = rv.build_trading_view(self._results(KNG={}), self._results(KNG={}), comparison, day=DAY)
        self.assertEqual(view["dashboard"]["funds"][0]["cash"]["post"], 5.0)
        self.assertNotIn("dashboard_error", view)
        json.dumps(view, allow_nan=False)

    def test_a_failing_dashboard_keeps_the_rest_of_the_view(self):
        with mock.patch.object(rv, "build_dashboard", side_effect=KeyError("trade_info")):
            view = rv.build_trading_view(self._results(), self._results(), {"date": DAY, "funds": {}}, day=DAY)
        self.assertIsNone(view["dashboard"])
        self.assertEqual(view["dashboard_error"], "KeyError: 'trade_info'")
        self.assertEqual(view["kind"], "trading")

    def test_stand_in_fund_objects_are_tolerated(self):
        comparison = {"date": DAY, "funds": {"KNG": {"trade_info": {
            "fund_object": object(), "total_net_assets": 100.0,
            "trade_activity": {"equity": {"buys": [{"ticker": "AAA", "quantity": 1.0, "market_value": 5.0}],
                                          "sells": []}}}}}}
        dash = td.build_dashboard(comparison, {}, {}, day=DAY)
        trade = dash["funds"][0]["blotter"][0]["trades"][0]
        # A report-era payload without a price reads as unpriced.
        self.assertEqual((trade["price"], trade["flags"], dash["funds"][0]["cash"]["current"]), (None, ["no_price"], None))


class AnalyzerPayloadTests(unittest.TestCase):
    def test_trades_carry_price_and_multiplier(self):
        post = _fund("KNG", options=_frame("optticker", [("X 11/06/26 C1", 0.0, 2.0, 2.0, 1.25,
                                                          {"multiplier": 10.0})]),
                     treasury=_frame("cusip", [("T1", 0.0, 1000.0, 1000.0, 98.0)]))
        info = _comparison({"KNG": (_fund("KNG"), post)})["funds"]["KNG"]["trade_info"]["trade_activity"]
        self.assertEqual(info["options"]["buys"][0], {"ticker": "X 11/06/26 C1", "quantity": 2.0, "market_value": 25.0,
                                                      "price": 1.25, "multiplier": 10.0})
        self.assertEqual(info["treasury"]["buys"][0]["market_value"], 980.0)
        self.assertEqual(info["treasury"]["buys"][0]["price"], 98.0)


class StagedTradeCountTests(unittest.TestCase):
    def test_counts_by_fund_and_asset_class(self):
        engine = create_engine("sqlite://")
        event.listen(engine, "connect", lambda conn, _: conn.execute("ATTACH DATABASE ':memory:' AS reconciliation"))
        with engine.begin() as conn:
            for view in ("v_vest_equity", "v_vest_option", "v_vest_treasury"):
                conn.execute(text(f"CREATE TABLE reconciliation.{view} (fund_ticker TEXT, `date` TEXT, "
                                  "analysis_type TEXT, trade_rebal REAL)"))
            rows = [("v_vest_equity", "KNG", DAY, "ex_post", 5.0), ("v_vest_equity", "KNG", DAY, "ex_post", -1.0),
                    ("v_vest_equity", "KNG", DAY, "ex_post", 0.0), ("v_vest_equity", "KNG", DAY, "ex_ante", 3.0),
                    ("v_vest_equity", "FDND", "2026-10-29", "ex_post", 1.0),
                    ("v_vest_option", "DOGG", DAY, " EX_POST ", -2.0), ("v_vest_option", "KNG", DAY, "ex_post", None),
                    ("v_vest_treasury", "DOGG", DAY, "ex_post", 100.0), ("v_vest_treasury", "TDVI", DAY, "ex_post", 0)]
            for view, *values in rows:
                conn.execute(text(f"INSERT INTO reconciliation.{view} VALUES (:f, :d, :a, :t)"),
                             dict(zip("fdat", values)))
        session = SimpleNamespace(bind=engine)
        self.assertEqual(staged_trade_counts(session, date(2026, 10, 30)), {
            "DOGG": {"equity": 0, "options": 1, "treasury": 1, "total": 2},
            "KNG": {"equity": 2, "options": 0, "treasury": 0, "total": 2},
        })
        self.assertEqual(staged_trade_counts(session, date(2026, 11, 2)), {})


if __name__ == "__main__":
    unittest.main()
