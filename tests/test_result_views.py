"""reporting.result_views on small hand-built results (no DB, no files).

    python -m unittest discover tests -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("DB_CONNECTION_STRING", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reporting import combined_reconciliation_report, result_views as rv  # noqa: E402
from reporting.compliance_reporting import _PDF_SECTIONS, ComplianceReportPDF  # noqa: E402
from reporting.footnotes import FOOTNOTES  # noqa: E402
from services.compliance_checker import RULE_KEYS, ComplianceResult  # noqa: E402

DAY = "2026-10-30"
IRS, ACT40, IRC, GICS = "diversification_IRS_check", "diversification_40act_check", "diversification_IRC_check", \
    "gics_compliance"


def _irs(**details):
    return ComplianceResult(True, {
        "rule": "IRS Diversification", "condition_IRS_1": True, "condition_IRS_2_a_50": True,
        "condition_IRS_2_a_5": True, "condition_IRS_2_a_10": True,
        "condition_IRS_2_a_10_status": "N/A (no shares outstanding data)", **details,
    }, {
        "total_assets": 8338122.0, "qualifying_assets_value": 5_000_000.0, "five_pct_gross_assets": 416906.1,
        "sum_large_securities_weights": 0.1056, "large_securities_count": np.int64(1),
        "large_securities": [{"eqyticker": "META", "tna_wgt": 0.1056}], "missing_shares_out_tickers": ["XYZ"],
        "bottom_50_largest": {"eqyticker": "AAPL", "tna_wgt": 0.031},
    })


def _summary():
    return ComplianceResult(True, {"status": "calculated"}, {
        "cash_value": 1250.5, "treasury_market_value": 0.0, "equity_market_value": 8_000_000.0,
        "option_delta_adjusted_notional": -125_000.0, "option_market_value": -2_500.0,
        "flex_option_market_value": 300.0, "total_option_market_value": -2_200.0,
        "total_assets": 8_338_122.0, "total_net_assets": 8_300_000.0, "expenses": 10.0,
    })


def _gics(overall="PASS"):
    return ComplianceResult(overall == "PASS", {
        "rule": "GICS Concentration", "overall_gics_compliance": overall, "industry_exceeds_25": True,
        "industry_group_exceeds_25": False, "compliance_group": "kng_fdnd",
        "exceeding_index_gics": {"GICS_INDUSTRY_NAME": {"Banks": 0.3}, "GICS_INDUSTRY_GROUP_NAME": {}},
    }, {"GICS_INDUSTRY_NAME": {"fund_weights": {"Banks": 0.31, "Software": 0.2}, "index_weights": {"Banks": 0.3},
                               "exceeding_fund": {"Banks": 0.31}, "exceeding_index": {"Banks": 0.3}}})


def _fund_result(compliance=None, nav=None, holdings=None, errors=()):
    return SimpleNamespace(compliance_results=compliance, nav_results=nav, reconciliation_results=holdings,
                           errors=list(errors))


def _results(**funds):
    return SimpleNamespace(fund_results=funds, summary={})


def _pdf_cells(fund_data, key):
    """What ComplianceReportPDF writes in this fund's column of the rule's metric table."""
    pdf = ComplianceReportPDF.__new__(ComplianceReportPDF)
    metrics = next(section[4] for section in _PDF_SECTIONS if section[1] == key)
    return [(label, pdf._cell_text(fund_data, key, fmt, source, field)) for label, fmt, source, field in metrics]


class ComplianceViewTests(unittest.TestCase):
    def setUp(self):
        self.compliance = {
            "summary_metrics": _summary(), IRS: _irs(), GICS: _gics(),
            ACT40: ComplianceResult(True, {"rule": "40 Act Diversification", "skipped": True,
                                           "reason": "Private funds are not RICs"}, {}),
            "twelve_d2_insurance_cos": ComplianceResult(False, {"rule": "12d-2", "status": "error"}, {}, "boom"),
            "real_estate_check": ComplianceResult(False, {"rule": "Real Estate Exposure"},
                                                  {"real_estate_percentage": 0.02, "total_exposure": 1e6}),
            "fund_object": object(),
        }
        self.view = rv.build_day_view(_results(KNG=_fund_result(self.compliance)), day=DAY,
                                      operations=["compliance"], compliance_tests=())
        self.rules = self.view["compliance"]["KNG"]

    def test_rows_are_the_pdf_cells(self):
        from reporting.report_utils import normalize_compliance_results
        fund_data = normalize_compliance_results({"KNG": self.compliance})["KNG"]
        for key in (IRS, ACT40, "summary_metrics", "real_estate_check", "twelve_d2_insurance_cos"):
            with self.subTest(key=key):
                self.assertEqual([(row["label"], row["value"]) for row in self.rules[key]["rows"]],
                                 _pdf_cells(fund_data, key))

    def test_statuses_and_reasons(self):
        statuses = {key: (rule["status"], rule["reason"]) for key, rule in self.rules.items()}
        self.assertEqual(statuses, {
            "summary_metrics": (None, None), GICS: ("PASS", None), ACT40: ("N/A", "Private funds are not RICs"),
            IRS: ("PASS", None), "real_estate_check": ("FAIL", None), "twelve_d2_insurance_cos": ("ERROR", "boom"),
        })
        self.assertEqual(list(self.rules), ["summary_metrics", GICS, ACT40, IRS, "real_estate_check",
                                            "twelve_d2_insurance_cos"])  # PDF order; rules not run are left out
        self.assertEqual(self.view["compliance_summary"]["KNG"], {
            "status": "FAIL", "counts": {"PASS": 2, "FAIL": 1, "N/A": 1, "ERROR": 1}, "failed": ["real_estate_check"]})

    def test_status_rows_carry_verdict_and_na_reason(self):
        rows = {row["label"]: row for row in self.rules[IRS]["rows"]}
        self.assertEqual(rows["Condition 1"]["status"], "PASS")
        self.assertEqual((rows["Condition 2a 10%"]["status"], rows["Condition 2a 10%"]["reason"]),
                         ("N/A", "N/A (no shares outstanding data)"))
        self.assertEqual(rows["Condition 2b"]["reason"], "N/A (not evaluated)")
        self.assertNotIn("status", rows["Large Securities Count"])
        self.assertEqual(rows["Sum Large Securities %"]["value"], "10.56000%")

    def test_details_are_the_workbook_columns(self):
        details = {row["label"]: row["value"] for row in self.rules[IRS]["details"]}
        self.assertEqual(details["Total Assets"], "8,338,122.00")
        self.assertEqual(details["Qualifying Assets Value"], "5,000,000.00")
        self.assertEqual(details["Large Securities"], "(META, 10.56%)")
        self.assertEqual(details["Bottom 50 Largest Holding"], "AAPL (3.10%)")
        self.assertEqual(details["5% Gross Assets Threshold ($)"], "416,906.10")
        self.assertEqual(details["Expenses"], "0.00")
        # Verdicts, formulas and values the PDF rows already show are not repeated.
        self.assertFalse({"Condition IRS 1", "Large Securities Count", "Sum Large Securities Weights"} & set(details))
        self.assertEqual(self.rules[IRS]["notes"],
                         ["No shares outstanding data (left out of the ownership test): XYZ"])
        summary = {row["label"]: row["value"] for row in self.rules["summary_metrics"]["details"]}
        self.assertEqual(summary, {"Regular Option MV": "-2,500.00", "Flex Option MV": "300.00",
                                   "Total Option MV": "-2,200.00"})

    def test_skipped_and_error_rules_have_no_details(self):
        self.assertEqual(self.rules[ACT40]["details"], [])
        self.assertEqual({row["value"] for row in self.rules[ACT40]["rows"] if "status" in row}, {"N/A"})
        self.assertEqual(self.rules["twelve_d2_insurance_cos"]["details"], [])

    def test_gics(self):
        gics = self.rules[GICS]
        self.assertEqual([(r["label"], r["value"]) for r in gics["rows"]], [
            ("Overall GICS Compliance", "PASS"), ("Industry Exceeds 25%", "Yes"), ("Industry Group Exceeds 25%", "No"),
            ("Index Industry Exceeds 25%?", "Yes"), ("Index Industry Group Exceeds 25%?", "No"),
            ("Can Fund exceed 25% if Index does?", "Yes"), ("Exceptions to Conc Policy", "")])
        self.assertEqual(gics["tables"][0]["rows"], [["Banks", "31.00%", "30.00%", "Yes", "Yes"],
                                                     ["Software", "20.00%", "0.00%", "", ""]])
        self.assertIn("Compliance group: kng_fdnd", gics["notes"])

    def test_rules_carry_footnotes_once(self):
        self.assertEqual(list(self.view["rules"]), list(self.rules))
        self.assertEqual(self.view["rules"][IRS]["footnotes"], FOOTNOTES["irs"])
        self.assertNotIn("", self.view["rules"][ACT40]["footnotes"])
        self.assertEqual(self.view["rules"][GICS]["title"], "GICS Diversification")

    def test_selected_rules_only(self):
        view = rv.compliance_views({"KNG": self.compliance}, DAY, ["summary", IRS])
        self.assertEqual(list(view["KNG"]), ["summary_metrics", IRS])

    def test_irc_reads_na_for_non_vit_funds_when_the_day_has_one(self):
        vit = {IRC: ComplianceResult(True, {"condition_IRC_55": True, "condition_IRC_70": True,
                                            "condition_IRC_80": True, "condition_IRC_90": True},
                                     {"top_1": 0.0419, "top_4": 0.15, "total_assets": 10.0})}
        view = rv.compliance_views({"VITF": vit, "KNG": {IRS: _irs()}}, DAY)
        self.assertEqual(view["VITF"][IRC]["status"], "PASS")
        self.assertEqual(view["KNG"][IRC]["status"], "N/A")
        self.assertIn("VIT", view["KNG"][IRC]["reason"])
        self.assertEqual({row["value"] for row in view["KNG"][IRC]["rows"]}, {"N/A"})
        self.assertNotIn(IRC, rv.compliance_views({"KNG": {IRS: _irs()}}, DAY)["KNG"])

    def test_irc_top_exposures_read_as_percentages_like_the_pdf(self):
        # The workbook formats these fractions as #,##0.00 (0.08); the view shows the PDF's .5% like Top 1 / Top 4.
        self.assertEqual(rv._detail_text("Top 2 Exposure", 0.0832), "8.32000%")
        self.assertEqual(rv._detail_text("Top 3 Exposure", np.float64(0.1226)), "12.26000%")
        self.assertEqual(rv._detail_text("Total Assets", 0.0832), "0.08")  # other columns as the workbook
        self.assertFalse(rv.is_percent_header("Top 2 Exposure"))  # the workbook's own check is unchanged

    def test_json_safe(self):
        json.dumps(self.view, allow_nan=False)
        self.assertEqual(rv.to_jsonable({"a": (np.float64("nan"), np.bool_(True), pd.Timestamp(DAY).date())}),
                         {"a": [None, True, DAY]})

    def test_fund_errors_and_missing_results(self):
        view = rv.build_day_view(_results(KNG=_fund_result(errors=["Error processing KNG: boom"])), day=DAY)
        self.assertEqual(view["fund_errors"], {"KNG": ["Error processing KNG: boom"]})
        self.assertEqual((view["compliance"], view["nav"], view["holdings"]), ({}, {}, {}))
        self.assertEqual(rv.build_day_view(SimpleNamespace(summary={}), day=DAY)["funds"], [])


def _nav(**overrides):
    nav = {"Beginning TNA": 100.0, "Equity G/L": 5.0, "Option G/L": -1.0, "Accruals": 0.5, "Distributions": -2.0,
           "Expected TNA": 1_234_567.891, "Custodian TNA": 1_234_560.0, "TNA Diff ($)": 7.891,
           "Expected NAV": 25.12346, "Custodian NAV": 25.1234, "NAV Diff ($)": 0.00005,
           "NAV Good (2 Digit)": True, "NAV Good (4 Digit)": True, "combined_gl_vest": 10.0, "combined_gl_cust": 12.0,
           "listed_gl_vest": 10.0, "listed_gl_cust": 12.0, "exp_nav_vest": 25.1, "exp_nav_cust": 25.2,
           "cust_nav": 25.1234}
    nav.update(overrides)
    return nav


class NavViewTests(unittest.TestCase):
    def test_rows_follow_the_pdf(self):
        view = rv.nav_views({"KNG": _nav()})["KNG"]
        self.assertEqual(view["status"], "PASS")
        self.assertEqual([(c["label"], c["value"]) for c in view["summary"]],
                         [(label, text) for label, text, _ in combined_reconciliation_report.nav_summary_cells(_nav())])
        self.assertEqual(view["summary"][5], {"label": "NAV Diff (2-dec)", "value": "0.00", "status": "PASS"})
        rows = {r["label"]: r["value"] for r in view["rows"]}
        self.assertEqual(rows["Expected TNA"], "$1,234,568")
        self.assertEqual(rows["Expected NAV"], "$25.1235")
        self.assertEqual(rows["Option Pricing Diff"], "$2")
        self.assertEqual(rows["NAV Difference ($ / %)"], "0.0001   (0.0002%)")
        self.assertEqual(view["sections"][1]["rows"], [{"label": "Assignment G/L", "value": "$0.00"},
                                                       {"label": "Accruals", "value": "$-0.50"},
                                                       {"label": "Distributions", "value": "$-2.00"}])
        gl = {r["label"]: r["value"] for r in view["gl"]}
        self.assertEqual((gl["Eqt G/L"], gl["G/L Today"]), ("5", "2"))  # 5 - 1 - 0.5 - |-2|
        self.assertEqual(view["tables"], [])  # not a closed-end fund

    def test_a_failed_flag_fails_the_fund(self):
        view = rv.nav_views({"KNG": _nav(**{"NAV Good (4 Digit)": False})})["KNG"]
        self.assertEqual(view["status"], "FAIL")
        self.assertEqual(view["checks"], [{"label": "NAV Good 2-Dec", "status": "PASS"},
                                          {"label": "NAV Good 4-Dec", "status": "FAIL"}])

    def test_cef_sensitivity_table(self):
        with mock.patch.object(combined_reconciliation_report, "is_cef", return_value=True):
            table = rv.nav_views({"HE3B1": _nav()})["HE3B1"]["tables"][0]
        self.assertEqual(table["rows"][0], ["Listed Option G/L", "$10", "$12", "$2"])
        self.assertEqual(table["rows"][3], ["Expected NAV", "25.1000", "25.2000", "0.1000"])

    def test_dataclass_results_are_converted(self):
        legacy = SimpleNamespace(to_legacy_dict=lambda: _nav())
        self.assertEqual(rv.nav_views({"KNG": legacy})["KNG"]["status"], "PASS")


class HoldingsViewTests(unittest.TestCase):
    def payload(self, equity_breaks=0, price_breaks=0):
        final = pd.DataFrame({"eqyticker": ["AAPL"] * equity_breaks, "discrepancy_type": ["Missing in custodian"]
                              * equity_breaks, "shares_cust": [0.0] * equity_breaks,
                              "shares_vest": [100.0] * equity_breaks})
        price = pd.DataFrame({"eqyticker": ["MSFT"] * price_breaks, "price_vest": [1.0] * price_breaks,
                              "price_cust": [2.0] * price_breaks})
        return {"summary": {"custodian_equity": {"final_recon": equity_breaks, "price_discrepancies_T": price_breaks}},
                "details": {"custodian_equity": {"final_recon": final, "price_discrepancies_T": price}}}

    def test_status_and_counts(self):
        views = rv.holdings_views({"A": self.payload(), "B": self.payload(price_breaks=2),
                                   "C": self.payload(equity_breaks=12, price_breaks=1)})
        self.assertEqual({fund: view["status"] for fund, view in views.items()}, {"A": "PASS", "B": "WARN", "C": "FAIL"})
        self.assertEqual((views["C"]["holdings_breaks"], views["C"]["price_breaks"]), (12, 1))
        self.assertEqual(views["C"]["counts"][0], {"label": "Equity Holdings", "value": 12, "kind": "holdings"})
        self.assertEqual(views["A"]["sections"], [])

    def test_discrepancy_tables_follow_the_pdf(self):
        section = rv.holdings_views({"C": self.payload(equity_breaks=12, price_breaks=1)})["C"]["sections"][0]
        holdings, price = section["tables"]
        self.assertEqual((section["key"], section["title"]), ("custodian_equity", "Custodian Equity"))
        self.assertEqual(holdings["columns"], ["eqy_ticker", "discrepancy_type", "shares_cust", "shares_vest",
                                               "price_vest", "price_cust"])
        self.assertEqual((len(holdings["rows"]), holdings["total"]), (10, 12))  # the PDF's first 10
        self.assertEqual(holdings["rows"][0], ["AAPL", "Missing in custodian", "0.00", "100.00", "", ""])
        self.assertEqual(price["rows"], [["MSFT", "1.00", "2.00"]])

    def test_custodian_sections_use_the_pdf_titles(self):
        payload = self.payload(equity_breaks=2)
        final = payload["details"]["custodian_equity"]["final_recon"]
        regular = pd.DataFrame({"ticker": ["A", "B", "C"]})
        payload["details"] = {"custodian_option": {"final_recon": final, "regular_options": regular},
                              "custodian_treasury": {"final_recon": final}}
        sections = {s["key"]: s for s in rv.holdings_views({"C": payload})["C"]["sections"]}
        self.assertEqual(sections["custodian_option"]["tables"][0]["title"], "Option Discrepancies")
        self.assertEqual(sections["custodian_treasury"]["tables"][0]["title"], "Treasury Discrepancies")
        self.assertEqual(sections["custodian_option"]["counts"], [{"label": "Regular Options", "value": 3}])
        self.assertEqual(sections["custodian_treasury"]["counts"], [])


class TradingViewTests(unittest.TestCase):
    def test_overview_checks_and_activity(self):
        comparison = {
            "date": DAY, "summary": {"total_funds_analyzed": 2, "total_funds_traded": 1},
            "funds": {
                "DOGG": {"status_change": "OUT_OF_COMPLIANCE", "violations_before": 0, "violations_after": 1,
                         "checks": {IRS: {"status_before": "PASS", "status_after": "FAIL", "changed": True}},
                         "trade_info": {"total_net_assets": 1000.0, "total_assets": 1100.0,
                                        "asset_trade_summary": {"equity": {"trade_value": 50.0, "pct_of_tna": 0.05}},
                                        "trade_activity": {"equity": {"buys": [{"ticker": "AAPL", "quantity": 2.0,
                                                                                "market_value": 50.0}], "sells": []}}}},
                "KNG": {"status_change": "UNCHANGED", "violations_before": 0, "violations_after": 0, "checks": {}},
            },
        }
        view = rv.trading_view(comparison)
        self.assertEqual(view["summary"], {"total_funds_analyzed": 2, "total_funds_traded": 1})
        self.assertEqual(view["overview"]["rows"][0], {"fund": "DOGG", "status_change": "OUT_OF_COMPLIANCE",
                                                       "violations_before": 0, "violations_after": 1,
                                                       "net_change": 1, "highlight": "out"})
        self.assertEqual(view["overview"]["totals"], {"violations_before": 0, "violations_after": 1, "net_change": 1})
        self.assertEqual(view["checks"]["DOGG"], [{"key": IRS, "label": "IRS Diversification", "before": "PASS",
                                                   "after": "FAIL", "changed": True}])
        equity = view["activity"]["rows"][0]
        self.assertEqual((equity["fund"], equity["asset_class"]), ("DOGG", "Equity"))
        self.assertEqual(equity["cells"][:3], [{"label": "Total Net Assets", "value": "1,000.00"},
                                               {"label": "Total Assets", "value": "1,100.00"},
                                               {"label": "Trade Value", "value": "50.00"}])
        self.assertEqual(equity["cells"][3], {"label": "% of TNA", "value": "5.00%"})
        self.assertEqual(view["activity_details"]["DOGG"], [{"asset_class": "Equity", "direction": "Buy",
                                                             "ticker": "AAPL", "quantity": "2.00",
                                                             "market_value": "50.00"}])

    def test_trading_day_view(self):
        ante = _results(DOGG=_fund_result({IRS: _irs(), "fund_object": object()}))
        post = _results(DOGG=_fund_result({IRS: _irs(condition_IRS_1=False), "fund_object": object()}),
                        KNG=_fund_result(errors=["Error processing KNG: x"]))
        view = rv.build_trading_view(ante, post, {"date": DAY, "funds": {}}, day=DAY)
        self.assertEqual((view["kind"], view["funds"]), ("trading", ["DOGG", "KNG"]))
        self.assertEqual(view["compliance_ex_ante"]["DOGG"][IRS]["rows"][1]["value"], "PASS")
        self.assertEqual(view["compliance"]["DOGG"][IRS]["rows"][1]["value"], "FAIL")
        self.assertEqual(view["fund_errors"], {"KNG": ["ex-post: Error processing KNG: x"]})
        self.assertEqual(view["trading"]["overview"]["rows"], [])
        json.dumps(view, allow_nan=False)


class RuleDescriptionTests(unittest.TestCase):
    def test_every_rule_is_described_without_the_db(self):
        descriptions = rv.rule_descriptions()
        self.assertEqual(list(descriptions), list(RULE_KEYS))
        for key, entry in descriptions.items():
            with self.subTest(key=key):
                self.assertTrue(entry["title"] and entry["summary"] and entry["applies_to"])
        self.assertIn("Result (DAN) >= 80%", descriptions["prospectus_80pct_policy"]["conditions"])
        self.assertTrue(any("75%" in c for c in descriptions[ACT40]["conditions"]))
        self.assertTrue(any("<= 55%" in c for c in descriptions[IRC]["conditions"]))
        self.assertEqual(descriptions["twelve_d2_insurance_cos"]["footnotes"], FOOTNOTES["12d2"])
        json.dumps(descriptions)


if __name__ == "__main__":
    unittest.main()
