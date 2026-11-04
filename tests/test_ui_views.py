"""The UI's pure pieces: HTML builders, the results view builders, the preview numbers, views.json loading. No DB.

    python -m unittest discover tests -p "test_ui_views.py" -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

os.environ["DB_CONNECTION_STRING"] = "sqlite:///:memory:"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ui_views_fixture  # noqa: E402
from processing import api  # noqa: E402
from ui import components as ui  # noqa: E402
from ui import history, new_run, preview, results, runner, settings_panel  # noqa: E402

CATALOG = {"cohorts": {"ETF_FUNDS": ["KNG", "TDVI"]}, "funds": ["DOGG", "KNG", "TDVI"],
           "rule_catalog": api.rule_catalog()}
DAY = ui_views_fixture.DAY


class ComponentsTest(unittest.TestCase):
    def test_values_are_escaped(self) -> None:
        self.assertEqual(ui.esc("<b>&"), "&lt;b&gt;&amp;")
        self.assertEqual(ui.esc(None), "")
        self.assertIn("&lt;script&gt;", ui.kv_table([{"label": "<script>", "value": 1}]))
        self.assertIn("&lt;x&gt;", ui.data_table(["<x>"], [["1"]]))

    def test_pills(self) -> None:
        self.assertIn("fm-pass", ui.pill("PASS"))
        self.assertIn("fm-fail", ui.pill("fail"))
        self.assertIn("fm-na", ui.pill("N/A"))
        self.assertIn("fm-dash", ui.pill(None))
        self.assertIn('title="why"', ui.pill("N/A", title="why"))

    def test_kv_rows_colour_status_but_not_informational_fails(self) -> None:
        html = ui.kv_table([{"label": "A", "value": "FAIL", "status": "FAIL"},
                            {"label": "B", "value": "FAIL", "status": "FAIL", "informational": True},
                            {"label": "C", "value": "N/A", "status": "N/A", "reason": "N/A (no data)"}])
        self.assertEqual(html.count('class="v-fail nw"'), 1)
        self.assertNotIn("v-fail", html.split("<th>B</th>")[1].split("</tr>")[0])  # informational: plain
        self.assertIn('<span class="why">N/A (no data)</span>', html)
        self.assertIn('class="long"', ui.kv_table([{"label": "L", "value": "x" * 200}]))

    def test_data_table_truncation_note_and_numbers(self) -> None:
        html = ui.data_table(["T", "Qty"], [["XOM", "1,000.00"]], total=80)
        self.assertIn("Showing the first 1 of 80", html)
        self.assertIn('<td class="num">1,000.00</td>', html)
        self.assertNotIn("Showing", ui.data_table(["T"], [["a"]], total=1))

    def test_grid_is_rows_of_two_in_order_then_wide_cards(self) -> None:
        html = ui.grid(["<i>1</i>", "<i>2</i>", "<i>3</i>"], wide=["<i>nav</i>"])
        self.assertNotIn("fm-col", html)  # one grid: cards line up in rows, never two independent columns
        self.assertEqual(html.count('<div class="fm-cell">'), 3)
        self.assertLess(html.index("<i>1</i>"), html.index("<i>2</i>"))
        self.assertLess(html.index("<i>2</i>"), html.index("<i>3</i>"))
        self.assertIn('<div class="fm-cell wide"><i>nav</i></div>', html)
        self.assertGreater(html.index("<i>nav</i>"), html.index("<i>3</i>"))

    def test_cells_numbers_blanks_and_headers(self) -> None:
        kv = ui.kv_table([{"label": "Option MV", "value": "-846,720"}, {"label": "Reg", "value": "Non-Diversified"},
                          {"label": "Exceptions", "value": ""}, {"label": "Cash", "value": "$-19,333,331.89"},
                          {"label": "Distributions", "value": "$-0.00"}])
        self.assertIn('<td class="num nw">−846,720</td>', kv)  # tabular figures and a true minus
        self.assertIn('<td class="nw">Non-Diversified</td>', kv)  # text keeps its hyphen, no tabular figures
        self.assertIn("−$19,333,331.89", kv)  # the sign before the currency symbol
        self.assertIn(">$0.00</td>", kv)  # a value that rounds to zero has no sign
        self.assertIn('<td class="nw"><span class="fm-dash">', kv)  # a blank value is a faint dash
        table = ui.data_table(["eqy_ticker", "shares_cust", "Fund Weight"], [["None", "nan", "-1.50%"]])
        self.assertIn(">Ticker</th>", table)
        self.assertIn(">Shares (Cust)</th>", table)
        self.assertIn(">Fund Weight</th>", table)
        self.assertEqual(table.count("fm-dash"), 2)
        self.assertIn("−1.50%", table)
        self.assertNotIn(">None<", table)
        self.assertNotIn(">nan<", table)
        self.assertEqual(ui.column_label("price_vest"), "Price (Vest)")
        self.assertEqual(ui.column_label("Status Change"), "Status Change")
        from ui import history_page  # noqa: PLC0415
        self.assertEqual(history_page.dates_html("2026-10-29..2026-10-30"),  # wraps at the arrow, never in a date
                         '<span class="nw">2026-10-29</span> → <span class="nw">2026-10-30</span>')

    def test_long_footnotes_fold(self) -> None:
        self.assertIn('<div class="fm-foot">', ui.footnotes("Footnotes", ["short"]))
        self.assertIn('<details class="fm-foot">', ui.footnotes("Footnotes", ["x" * 400]))
        self.assertEqual(ui.footnotes("Footnotes", []), "")


class ResultBuildersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.day = ui_views_fixture.eod_views()["days"][DAY]

    def test_matrix(self) -> None:
        html = results.matrix_html(self.day)
        self.assertEqual(results.rule_keys(self.day),
                         ["gics_compliance", "diversification_40act_check", "diversification_IRS_check"])
        self.assertIn(">KNG</td>", html)
        self.assertIn(">NAV</th>", html)
        self.assertIn(">Holdings</th>", html)
        self.assertIn('title="1 failed"', html)
        self.assertEqual(html.count("fm-dash"), 2)  # TDVI has no NAV and no holdings view
        self.assertNotIn(results.IRC_NOTE, html)
        irc_picked = results.matrix_html(self.day, requested=["gics_compliance", results.IRC_KEY])
        self.assertIn(results.IRC_NOTE, irc_picked)  # a picked IRC check without a column is explained

    def test_fund_cards_start_with_the_summary_and_follow_the_pdf_order(self) -> None:
        cards = results.fund_cards_html(self.day, "KNG")
        titles = [c.split('<span>')[1].split("<")[0] for c in cards]
        self.assertEqual(titles, ["Summary Metrics", "GICS Diversification", "40 Act Diversification",
                                  "IRS Diversification"])
        self.assertIn("IRS Condition 1", cards[-1])
        self.assertIn("More detail (2)", cards[-1])
        self.assertEqual(results.fund_cards_html(self.day, "NOPE"), [])

    def test_nav_and_holdings(self) -> None:
        nav = results.nav_card_html(self.day["nav"]["KNG"])
        self.assertIn("NAV Good 2-Dec", nav)
        self.assertIn("Gain/Loss Components", nav)
        self.assertIn("G/L walk", nav)
        holdings = results.holdings_html(self.day["holdings"]["KNG"])
        self.assertIn('class="fm-stat bad"', holdings)
        self.assertIn('class="fm-stat warnish"', holdings)
        self.assertIn("Custodian Equity · 80 rows", holdings)
        self.assertIn(">Ticker</th>", holdings)  # snake_case headers shown as words
        option = dict(self.day["holdings"]["KNG"], sections=[{"key": "custodian_option", "title": "Custodian Option",
                                                              "counts": [{"label": "Regular Options", "value": 6}],
                                                              "tables": []}])
        self.assertIn("Regular Options<b>6</b>", results.holdings_html(option))
        self.assertIn('class="fm-split"', nav)  # the wide NAV card: summary beside the G/L sections

    def test_trading(self) -> None:
        trading = ui_views_fixture.trading_views()["days"][DAY]["trading"]
        overview = results.trading_overview_html(trading)
        self.assertIn('class="hl-bad"', overview)
        self.assertIn('class="total"', overview)
        self.assertIn("1 of 1 funds traded", overview)
        self.assertIn("hl-breach", results.trading_checks_html(trading["checks"]["DOGG"]))
        activity = results.trading_activity_html(trading, "DOGG")
        self.assertIn("Trades (1)", activity)
        self.assertIn("No trade activity", results.trading_activity_html(trading, "KNG"))


class PreviewTest(unittest.TestCase):
    def test_counts(self) -> None:
        choices = new_run.Choices(funds=["KNG", "TDVI"], date_kind=new_run.RANGE,
                                  date_range=(date(2026, 10, 26), date(2026, 10, 30)),
                                  eod_ops=["compliance", "reconciliation"], rules_mode=new_run.CHOOSE_RULES,
                                  rules=["gics_compliance", "real_estate_check"])
        c = preview.counts(CATALOG, choices)
        self.assertEqual((c.funds, c.days, c.checks, c.recons, c.total), (2, 5, 2, 1, 30))
        trading = preview.counts(CATALOG, new_run.Choices(funds=["KNG"], kind=new_run.KIND_TRADING,
                                                          as_of=date(2026, 10, 30)))
        self.assertEqual((trading.checks, trading.passes, trading.total), (11, 2, 22))
        nothing = preview.counts(CATALOG, new_run.Choices(date_kind=new_run.RANGE, date_range=()))
        self.assertEqual((nothing.funds, nothing.days, nothing.total), (0, None, None))

    def test_sentence(self) -> None:
        c = preview.Counts(funds=3, days=31, checks=4, recons=0, operations=("compliance",), trading=False)
        html = preview.sentence(c, (150.0, True, 6))
        self.assertIn("3 funds × 31 days × 4 checks", html)
        self.assertIn("372 total operations", html)
        self.assertIn("~ 2–4 minutes (estimate from 6 past runs)", html)
        self.assertIn("Pick the dates", preview.sentence(preview.Counts(0, 1, 4, 0, ("compliance",), False), None))
        mixed = preview.Counts(funds=3, days=1, checks=11, recons=2,
                               operations=("compliance", "reconciliation", "nav_reconciliation"), trading=False)
        self.assertIn("3 funds × 1 day × 13 checks (11 rules + 2 recons)", preview.sentence(mixed, None))

    def test_estimate_and_duration(self) -> None:
        self.assertEqual(preview.estimate_seconds(10, 2, [3.0, 5.0, 100.0]), (50.0, True))  # the median rate
        seconds, from_history = preview.estimate_seconds(4, 1)
        self.assertFalse(from_history)
        self.assertEqual(seconds, preview.HEURISTIC_DAY_SECONDS + 4 * preview.HEURISTIC_FUND_DAY_SECONDS)
        self.assertEqual(preview.format_duration(30), "~ 25–40 seconds")
        self.assertEqual(preview.format_duration(150), "~ 2–4 minutes")
        self.assertEqual(preview.format_duration(3 * 3600), "~ 2–4 hours")


class SettingsTest(unittest.TestCase):
    def test_group_labels(self) -> None:
        self.assertEqual(settings_panel.group_label("ETF_FUNDS"), "ETF Funds")
        self.assertEqual(settings_panel.group_label("VIT_AND_MUTUAL_FUNDS"), "VIT and Mutual Funds")
        self.assertEqual(settings_panel.group_label("CLOSED_END_FUNDS"), "Closed End Funds")

    def test_run_labels(self) -> None:
        label = lambda **kw: settings_panel.run_label(new_run.Choices(**kw))  # noqa: E731
        self.assertEqual(label(), "Run compliance checks")
        self.assertEqual(label(eod_ops=["compliance", "nav_reconciliation"]), "Run checks & recons")
        self.assertEqual(label(eod_ops=["reconciliation"]), "Run reconciliation")
        self.assertEqual(label(kind=new_run.KIND_TRADING), "Run trading compliance")
        self.assertEqual(label(eod_ops=[]), "Run")

    def test_empty_pick_is_refused_by_the_page(self) -> None:
        self.assertEqual(new_run.evaluate(CATALOG, new_run.Choices(funds=[]))[0], None)
        request, problem = new_run.evaluate(CATALOG, new_run.Choices(funds=["KNG"], as_of=date(2026, 10, 30)))
        self.assertIsNone(problem)
        self.assertEqual(request.funds, ("KNG",))


class HistoryViewsTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run_dir = Path(tmp.name) / "20261103-100000-u-eod"
        self.run_dir.mkdir()

    def write(self, name: str, data: object) -> None:
        (self.run_dir / name).write_text(json.dumps(data), encoding="utf-8")

    def test_load_views(self) -> None:
        self.assertIsNone(history.load_views(self.run_dir))  # an older run without views.json
        self.write(runner.VIEWS, {"schema_version": 99, "days": {}})
        self.assertIsNone(history.load_views(self.run_dir))  # unknown schema
        self.write(runner.VIEWS, {"schema_version": 1, "days": {"2026-10-30": {"kind": "eod"},
                                                                  "2026-10-29": {"kind": "eod"}, "bad": 3}})
        self.assertEqual(list(history.load_views(self.run_dir)["days"]), ["2026-10-29", "2026-10-30"])
        url = "mysql+pymysql://fmuser:Hunter2Secret@db.internal/fm"
        self.write(runner.VIEWS, {"schema_version": 1, "days": {"2026-10-30": {
            "fund_errors": {"KNG": [f"KNG: cannot connect to {url}"]},
            "compliance": {"KNG": {"r": {"reason": f"ERROR: {url}", "rows": [{"label": "x", "reason": url}],
                                         "notes": [url]}}}}}})
        day = history.load_views(self.run_dir)["days"]["2026-10-30"]
        self.assertNotIn("Hunter2Secret", json.dumps(day))  # exception texts are redacted on load
        self.assertIn("KNG: cannot connect to", day["fund_errors"]["KNG"][0])
        (self.run_dir / runner.VIEWS).write_text("{not json", encoding="utf-8")
        self.assertIsNone(history.load_views(self.run_dir))

    def test_fund_day_seconds(self) -> None:
        self.write(runner.META, {"user": "u", "started": "2026-11-03T10:00:00", "finished": "2026-11-03T10:01:00",
                                 "exit_code": 0})
        self.write(runner.RESULT, {"exit_code": 0, "errors": [], "files": [],
                                   "summaries": {"2026-10-29": {"total_funds": 2}, "2026-10-30": {"total_funds": 4}}})
        record = history.load_record(self.run_dir)
        self.assertEqual(history.fund_day_seconds([record]), [10.0])
        self.write(runner.META, {"user": "u", "started": "2026-11-03T10:00:00", "finished": "2026-11-03T10:01:00",
                                 "exit_code": 1})
        self.assertEqual(history.fund_day_seconds([history.load_record(self.run_dir)]), [])  # failed runs left out


if __name__ == "__main__":
    unittest.main()
