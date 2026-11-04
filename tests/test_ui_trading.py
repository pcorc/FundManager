"""The Trading Compliance page and dashboard: builders (charts, matrix, table, drill-down) and the page via AppTest.

    python -m unittest discover tests -p "test_ui_trading.py" -v

No DB: the catalog and the staged-trades lookup are faked, runs use the fake worker, and no request here
asks for email.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

os.environ["DB_CONNECTION_STRING"] = "sqlite:///:memory:"  # nothing here may reach the real DB
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_trading_dashboard as tdt  # noqa: E402  (hand-built funds through the real data layer)
import ui_views_fixture as fx  # noqa: E402
from processing import api  # noqa: E402
from test_ui_app import CATALOG, AppTestCase  # noqa: E402
from ui import charts, history, runner, trading_page  # noqa: E402
from ui import components as ui  # noqa: E402
from ui import trading_drilldown as dd  # noqa: E402
from ui import trading_results as tr  # noqa: E402

STAGED = {"DOGG": {"equity": 20, "options": 20, "treasury": 1, "total": 41}, "KNG": {"equity": 3, "total": 3},
          "XVMS": {"equity": 5, "total": 5}}  # XVMS: not a selectable fund


def dashboard() -> dict:
    return fx.trading_dashboard()


class NumbersTest(unittest.TestCase):
    def test_money_pct_quantity(self) -> None:
        self.assertEqual(ui.money(301_901_576.37), "$301.9M")
        self.assertEqual(ui.money(7_240_344.52), "$7.24M")
        self.assertEqual(ui.money(853_584.24), "$853.6K")
        self.assertEqual(ui.money(-2_010_750.39), "−$2.01M")
        self.assertEqual(ui.money(90_649_227.7, signed=True), "+$90.65M")
        self.assertEqual(ui.money(1.2e10), "$12.00B")
        self.assertEqual(ui.money(0.2, signed=True), "$0")
        self.assertEqual(ui.money(-7240344.52, compact=False), "−$7,240,345")
        self.assertEqual(ui.money(None), "—")
        self.assertEqual(ui.pct(0.05224956), "5.22%")
        self.assertEqual(ui.pct(-0.0555, signed=True), "−5.55%")
        self.assertEqual(ui.pct(0.00001), "<0.01%")
        self.assertEqual(ui.pct(0), "0.00%")
        self.assertEqual(ui.pct(float("nan")), "—")
        self.assertEqual(ui.quantity(49208.0), "49,208")
        self.assertEqual(ui.quantity(-184.0), "−184")


class ChartsTest(unittest.TestCase):
    def test_nice_domain_covers_zero_and_the_values(self) -> None:
        low, high, ticks = charts.nice_domain([-7.2e6, 61e6])
        self.assertLessEqual(low, -7.2e6)
        self.assertGreaterEqual(high, 61e6)
        self.assertIn(0.0, ticks)
        self.assertEqual(charts.nice_domain([])[:2], (0.0, 1.0))

    def test_svg_builders_escape_text(self) -> None:
        funds = dashboard()["funds"]
        funds[0] = dict(funds[0], fund="<script>x</script>")
        for markup in (charts.net_by_asset_chart(funds), charts.cash_dumbbell_chart(funds),
                       charts.pct_tna_chart(funds)):
            self.assertIn("<svg", markup)
            self.assertNotIn("<script>", markup)
            self.assertIn("&lt;script&gt;x&lt;/script&gt;", markup)
            self.assertIn('role="img"', markup)
            self.assertIn("<title>", markup)  # hover text per mark
        steps = [dict(s, label="Cash <i>&</i>") for s in funds[1]["waterfall"]]
        waterfall = charts.waterfall_chart(steps)
        self.assertNotIn("<i>&</i>", waterfall)
        self.assertIn("Cash &lt;i&gt;&amp;&lt;/i&gt;", waterfall)
        self.assertNotIn("<script", charts.legend([("#000", "<script>")]))

    def test_net_chart_is_diverging_and_split_by_asset_class(self) -> None:
        markup = charts.net_by_asset_chart(dashboard()["funds"])
        self.assertIn(charts.ASSET_COLORS["equity"], markup)
        self.assertIn(charts.ASSET_COLORS["options"], markup)  # DOGG writes options (net selling)
        self.assertIn("KNG · Equity net −$2,000,000", markup)
        self.assertIn("−$2.00M", markup)  # KNG's net, in the value column
        zero = float(re.search(r'<line x1="([\d.]+)%" x2="\1%" y1="16" y2="\d+" stroke="#5A6480"/></svg>', markup).group(1))
        kng = float(re.search(r'<g><rect x="([\d.]+)%" y="47.0"', markup).group(1))  # row 2: 20 + 22 + 5
        self.assertLess(kng, zero)  # net selling sits left of zero

    def test_dumbbell_marks_negative_cash(self) -> None:
        funds = dashboard()["funds"]
        funds.append(dict(funds[0], fund="NOCASH", cash={"current": None, "post": None}))
        markup = charts.cash_dumbbell_chart(funds)
        self.assertIn("Below zero", markup)  # the washed negative region
        self.assertIn('class="neg"', markup)
        self.assertIn("⚠ −$2.00M", markup)  # TDVI
        self.assertIn("Not plotted (no custodian cash or TNA): NOCASH", markup)
        self.assertIn(charts.POST, markup)  # the post-trade dot is a neutral ink, not the equity hue
        self.assertNotIn(charts.ASSET_COLORS["equity"], markup)

    def test_waterfall_runs_from_current_to_post_cash(self) -> None:
        markup = charts.waterfall_chart(dashboard()["funds"][2]["waterfall"])
        self.assertIn("Current cash: $1,000,000", markup)
        self.assertIn("Post-trade cash (est.): −$2,000,000", markup)
        self.assertIn("−$3.00M", markup)  # the equity buys step
        self.assertEqual(charts.waterfall_chart([]), "")


class DashboardBuildersTest(unittest.TestCase):
    def test_kpis(self) -> None:
        html = tr.kpi_html(dashboard())
        self.assertIn("Funds with trades", html)
        self.assertIn("of 4 selected · 1 without trades", html)
        self.assertIn('class="fm-kpi breaks"', html)  # one fund goes out of compliance
        self.assertIn('class="fm-kpi cures"', html)
        self.assertIn('class="fm-kpi neg"', html)  # a fund below zero after the trades: red, as everywhere
        self.assertIn("1 fund(s) below zero", html)
        self.assertIn("Still failing", html)

    def test_matrix_transitions_and_toggle(self) -> None:
        d = dashboard()
        html = tr.matrix_html(d, only_changes=True)
        self.assertIn('class="fm-tx breaks"', html)
        self.assertIn('class="fm-tx cures"', html)
        self.assertIn("DOGG · IRS Diversification: PASS before → FAIL after", html)
        self.assertIn('<tr class="hl-bad"><td class="fund jump" data-fm-jump="DOGG"', html)
        self.assertNotIn(">GICS</th>", html)  # no change, no failure: hidden by the toggle
        self.assertIn(">GICS</th>", tr.matrix_html(d, only_changes=False))
        self.assertIn("No proposed trades:</b> RDVI", html)
        self.assertIn("Fails before the trades, passes after them", html)  # the legend
        for row in d["matrix"]["rows"]:
            for cell in row["cells"].values():
                cell["transition"] = "OK"
        for rule in d["matrix"]["rules"]:
            rule["changes_or_fails"] = False
        self.assertIn("Every rule passes", tr.matrix_html(d, only_changes=True))

    def test_rules_at_risk(self) -> None:
        html = tr.rules_at_risk_html(dashboard())
        self.assertIn("IRS Diversification", html)
        self.assertIn("1 of 3 fail after", html)
        self.assertRegex(html, r'class="grp breaks">.*?<span data-fm-jump="DOGG" data-fm-rule="1"[^>]*>DOGG</span>')
        self.assertRegex(html, r'class="grp cures">.*?data-fm-jump="KNG"[^>]*>KNG</span>')

    def test_summary_table(self) -> None:
        html = tr.summary_table_html(dashboard())
        self.assertIn("Post-trade<br><span>estimate</span>", html)
        self.assertIn('title="−$2,000,000">−$2.00M', html)  # TDVI post cash, exact on hover
        self.assertIn('class="num neg"', html)
        self.assertIn('aria-label="negative cash">⚠</span>', html)
        self.assertIn('<tr class="total"><td class="fund">Total<span class="sub">3 funds</span></td>', html)
        self.assertIn("Options premium", html)  # DOGG wrote options
        self.assertNotIn(">Treasury<", html)  # no treasury trades: no treasury columns
        self.assertIn("Definitions", html)
        self.assertIn("Estimate: current cash - buys + sells.", html)
        self.assertIn('aria-label="2 data notes">2</span>', html)  # DOGG: warnings other than the cash one
        self.assertEqual(tr.summary_table_html({"funds": []}), "")

    def test_drilldown(self) -> None:
        d = dashboard()
        dogg = d["funds"][0]
        head = dd.fund_header_html(dogg)
        self.assertIn("Goes out of compliance (violations 0 → 1)", head)
        self.assertIn("0 → 1", head)
        shown, folded = dd.split_rules(dogg["rules"])
        self.assertEqual([r["rule"] for r in shown], ["diversification_IRS_check"])
        self.assertEqual([r["rule"] for r in folded], ["gics_compliance"])
        card = dd.rule_compare_html(shown[0])
        self.assertIn('class="fm-card fm-cmpcard t-breaks"', card)
        self.assertIn("Before · ex-ante", card)
        self.assertIn('<tr class="chg"><th>Condition 2a 50% weight</th>', card)
        self.assertIn('<i class="up" aria-label="up">▲</i><b>14.3%</b>', card)
        self.assertIn("1 rule unchanged by the trades", dd.folded_rules_html(folded))
        blotter = "".join(dd.blotter_html(dogg))
        self.assertIn("&lt;b&gt;X&lt;/b&gt;", blotter)
        self.assertNotIn("<b>X</b>", blotter)
        self.assertIn(">no price</span>", blotter)
        self.assertIn('<tr class="out" title="Left out of the cash estimate">', blotter)
        self.assertIn("1,000 → 2,000", blotter)
        self.assertIn("Cash waterfall", dd.waterfall_html(dogg))
        self.assertIn("No custodian cash", dd.waterfall_html(dict(dogg, cash={"current": None})))
        options = dd.rule_filter_options(d)
        self.assertEqual(options, [dd.ALL_FUNDS, "diversification_IRS_check", "real_estate_check"])
        self.assertEqual([dd.rule_filter_label(d, o) for o in options],
                         [dd.ALL_FUNDS_LABEL, "IRS Diversification · 1", "Real Estate Exposure · 1"])
        self.assertEqual(dd.funds_for_filter(d, "real_estate_check"), ["KNG"])
        self.assertEqual(dd.funds_for_filter(d, dd.ALL_FUNDS), ["DOGG", "KNG", "TDVI"])

    def test_staged_note_and_dates(self) -> None:
        funds, note = trading_page.staged_note(date(2026, 10, 30), {"funds": STAGED, "error": None}, CATALOG)
        self.assertEqual(funds, ["DOGG", "KNG"])  # XVMS cannot be picked
        self.assertIn("2 funds have staged trades on 2026-10-30", note)
        self.assertEqual(trading_page.staged_note(date(2026, 11, 4), {"funds": {}, "error": None}, CATALOG)[0], [])
        self.assertIn("Could not look up", trading_page.staged_note(date(2026, 11, 4), {"error": "OperationalError"},
                                                                    CATALOG)[1])
        self.assertEqual(trading_page.default_date(date(2026, 10, 31)), date(2026, 10, 30))  # Saturday -> Friday
        self.assertEqual(trading_page.default_date(date(2026, 11, 4)), date(2026, 11, 4))

    def test_load_views_redacts_the_dashboard(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            views = fx.trading_dashboard_views()
            views["days"][fx.DAY]["dashboard_error"] = f"OperationalError: {fx.LEAK}"
            (Path(tmp) / runner.VIEWS).write_text(json.dumps(views), encoding="utf-8")
            text = json.dumps(history.load_views(Path(tmp)))
        self.assertNotIn("Hunter2Secret", text)


def real_dashboard() -> dict:
    """A schema-2 dashboard from :mod:`reporting.trading_dashboard` on hand-built funds.

    BRK: passes before, IRS fails after; WRS: still fails and gets worse, its cash goes below zero; NEG: already
    negative cash; HE3: duplicate rows and report-only option changes; PD2: no custodian TNA or cash.
    """
    irs = tdt.IRS
    key = "Sum Large Securities %"
    trade = tdt._frame("eqyticker", [("AAA", 0.0, 10.0, 10.0, 10.0)])
    options = tdt.pd.DataFrame({"optticker": ["X 12/18/26 P1"], "nav_shares": [-1.0], "iiv_shares": [-3.0],
                                "price": [2.0]})
    funds = {
        "BRK": (tdt._fund("BRK", prev_cash=1000.0, prev_tna=10_000.0), tdt._fund("BRK", equity=trade)),
        "WRS": (tdt._fund("WRS", prev_cash=50.0, prev_tna=10_000.0), tdt._fund("WRS", equity=trade)),
        "NEG": (tdt._fund("NEG", prev_cash=-20.0, prev_tna=10_000.0), tdt._fund("NEG", equity=trade)),
        "HE3": (tdt._fund("HE3", equity=tdt._frame("eqyticker", [("MRK", 100.0, 100.0, 0.0, 10.0)] * 2),
                          prev_cash=500.0, prev_tna=10_000.0),
                tdt._fund("HE3", equity=tdt._frame("eqyticker", [("MRK", 100.0, 60.0, -40.0, 10.0)] * 2),
                          options=options)),
        "PD2": (tdt._fund("PD2"), tdt._fund("PD2", equity=trade)),
    }
    ante = {"BRK": {irs: tdt._rule("PASS")}, "WRS": {irs: tdt._rule("FAIL", [{"label": key, "value": "60.0%"}])},
            "NEG": {irs: tdt._rule("PASS")}, "HE3": {irs: tdt._rule("PASS")}, "PD2": {irs: tdt._rule("FAIL")}}
    post = {"BRK": {irs: tdt._rule("FAIL")}, "WRS": {irs: tdt._rule("FAIL", [{"label": key, "value": "61.5%"}])},
            "NEG": {irs: tdt._rule("PASS")}, "HE3": {irs: tdt._rule("PASS")}, "PD2": {irs: tdt._rule("FAIL")}}
    return tdt._dashboard(funds, ante, post, rules={irs: {"title": "IRS", "label": "IRS Diversification"}})


class RoundOneBuildersTest(unittest.TestCase):
    """The repair-round fixes on a schema-2 dashboard built by the real data layer."""

    def setUp(self) -> None:
        self.d = real_dashboard()
        self.funds = {f["fund"]: f for f in self.d["funds"]}

    def test_kpis_use_the_priced_basis_and_red_for_negative_cash(self) -> None:
        html = tr.kpi_html(self.d)
        k = self.d["kpis"]
        self.assertIn(f'<div class="v">{ui.money(k["priced_sell_value"])}</div>', html)
        self.assertIn("incl. flagged rows", html)  # HE3: 2 duplicate sells, the report counts both
        self.assertIn('class="fm-kpi neg"', html)
        self.assertIn("go below zero", html)
        self.assertIn("1 already negative", html)
        self.assertIn("at the 2026-10-29 close", html)
        self.assertNotIn("today", html)
        self.assertIn('class="fm-kpi worse"', html)  # WRS: still failing and worse
        self.assertEqual(html.count('class="fm-kpi-row"'), 2)

    def test_kpi_notes_guard_missing_figures(self) -> None:
        html = tr.kpi_html({"kpis": {"funds_with_trades": 1}, "totals": {}})
        self.assertNotIn("gross — of TNA", html)
        self.assertNotIn("from —", html)

    def test_data_notes_callout(self) -> None:
        html = tr.data_notes_html(self.d)
        self.assertIn('class="fm-callout warn"', html)
        self.assertIn("1 duplicate trade row(s) worth $400 in 1 fund(s), counted once", html)
        self.assertIn("no custodian TNA or cash at the T-1 close for PD2", html)
        self.assertIn("without staged rows in 1 fund(s)", html)
        self.assertEqual(tr.data_notes_html(fx.trading_dashboard()), "")  # schema 1: no notes block

    def test_matrix_trend_and_jump_targets(self) -> None:
        html = tr.matrix_html(self.d, only_changes=False)
        self.assertIn('class="fm-tx still worse"', html)
        self.assertIn('<i class="tr">· worse</i>', html)
        self.assertIn("Sum Large Securities % 60.0% → 61.5%", html)  # the key figure in the hover
        self.assertIn('data-fm-jump="WRS" data-fm-rule="0"', html)
        self.assertIn('aria-label="missing data">∅</span>', html)  # PD2
        self.assertIn("Breach deepens", html)
        self.assertEqual([r["fund"] for r in self.d["matrix"]["rows"]][:2], ["BRK", "WRS"])
        self.assertIn("the trades make it worse", tr.legend_html(self.d))
        risk = tr.rules_at_risk_html(self.d)
        self.assertRegex(risk, r'class="worse" data-fm-jump="WRS" data-fm-rule="0"')
        self.assertIn("· 1 worse", risk)

    def test_jump_options_and_callback(self) -> None:
        options = tr.jump_options(self.d)
        self.assertIn("WRS", options)
        self.assertIn("WRS #0", options)
        names = tr.keys("tc", "run1")
        state = {names["jump"]: "WRS #0"}
        with mock.patch.object(tr.st, "session_state", state):
            tr.apply_jump(self.d, names)
        self.assertEqual((state[names["fund"]], state[names["focus"]], state[names["rule"]], state[names["tabs"]]),
                         ("WRS", tdt.IRS, tdt.IRS, tr.DRILL_TAB))
        self.assertIsNone(state[names["jump"]])
        state = {names["jump"]: "NEG #0"}  # NEG passes IRS: the filter shows every fund
        with mock.patch.object(tr.st, "session_state", state):
            tr.apply_jump(self.d, names)
        self.assertEqual((state[names["fund"]], state[names["rule"]]), ("NEG", dd.ALL_FUNDS))
        self.assertIn("data-fm-jump", tr.JUMP_SCRIPT)

    def test_summary_table_marks_missing_unpriced_and_report_only(self) -> None:
        html = tr.summary_table_html(self.d)
        self.assertIn('title="No custodian TNA at the T-1 close', html)
        self.assertIn(">no data</td>", html)
        self.assertIn("in report · not staged", html)  # HE3's option changes
        self.assertIn("2026-10-29 close", html)
        self.assertIn("The trades take estimated cash below zero", html)
        self.assertIn("Cash is already below zero before the trades", html)
        unpriced = dict(self.funds["BRK"]["assets"]["options"], unpriced_sell_value=400.0, sell_value=400.0,
                        staged=True)
        self.assertIn(">unpriced</span>", "".join(tr._class_cells(unpriced, "options")))

    def test_empty_state_has_no_kpis(self) -> None:
        d = tdt._dashboard({"FTMIX": (tdt._fund("FTMIX"), tdt._fund("FTMIX"))},
                           {"FTMIX": {tdt.IRS: tdt._rule("FAIL")}}, {"FTMIX": {tdt.IRS: tdt._rule("FAIL")}})
        html = tr.no_trades_html(d)
        self.assertIn("No proposed trades", html)
        self.assertIn("FTMIX", html)
        self.assertNotIn("fm-kpi", html)

    def test_drilldown_labels_cash_bases(self) -> None:
        head = dd.fund_header_html(self.funds["PD2"])
        self.assertIn("TNA<b>no data</b>", head)
        self.assertIn("not estimated (no custodian cash)", head)
        self.assertNotIn("Traded<b>", head)
        self.assertIn("Cash (2026-10-29 close)", dd.fund_header_html(self.funds["WRS"]))
        card = dd.metrics_card_html(dict(self.funds["WRS"], metrics=[{"label": "Cash", "before": "1", "after": "1",
                                                                       "changed": False}]))
        self.assertIn("compliance inputs", card)
        self.assertIn("not adjusted for the trades", card)
        self.assertIn('<div class="fm-reason">No custodian cash', dd.waterfall_html(self.funds["PD2"]))
        self.assertIn("position changes without staged rows", "".join(dd.unstaged_html(self.funds["HE3"])))
        wrs = dd.rule_compare_html(self.funds["WRS"]["rules"][0], focus=True)
        self.assertIn("t-still focus", wrs)
        self.assertIn('class="fm-keyfig worse"', wrs)
        self.assertNotIn("fm-pill", wrs.split("fm-band-tx")[1].split("</div>")[0])  # no FAIL → FAIL pills
        shown, _ = dd.split_rules(self.funds["NEG"]["rules"], focus=tdt.IRS)
        self.assertEqual(shown[0]["rule"], tdt.IRS)

    def test_table_diff_card(self) -> None:
        rule = {"rule": "gics_compliance", "label": "GICS Concentration", "transition": "OK", "changed_rows": 1,
                "rows": [], "details": [], "tables": [{"title": "Industry exposure",
                                                       "columns": ["Category", "Fund Weight", "Index Weight"],
                                                       "total": 3, "rows": [{"key": "Media",
                                                                             "before": ["34.07%", "27.37%"],
                                                                             "after": ["33.83%", "27.37%"]}]}]}
        html = dd.rule_compare_html(rule)
        self.assertIn("Industry exposure: 1 of 3 rows change", html)
        self.assertIn("<th>Media · Fund Weight</th>", html)
        self.assertIn("33.83%", html)

    def test_charts_line_up_and_explain_left_out_trades(self) -> None:
        funds = self.d["funds"]
        pct = charts.pct_tna_chart(funds)
        self.assertIn('<div class="fm-legend" aria-hidden="true">&nbsp;</div>', pct)  # same header height
        self.assertIn("Not plotted", pct)
        self.assertIn("(table:", pct)
        self.assertIn(charts.TOTAL, pct)
        self.assertNotIn(charts.ASSET_COLORS["equity"], pct)
        self.assertIn("Not plotted", charts.net_by_asset_chart(funds))
        # The 2026-11-03 net chart: "−$200K" and "$0" overlapped in a 125px plot; zero gives way, the negative end
        # stays. A wider plot has room for all three.
        money = [-200_000.0, 0.0, 200_000.0, 400_000.0, 600_000.0]
        self.assertEqual(charts.labelled_ticks(money, "money"), [-200_000.0, 600_000.0])
        self.assertEqual(charts.labelled_ticks(money, "money", plot_px=400), [-200_000.0, 0.0, 600_000.0])
        self.assertEqual(charts.labelled_ticks([-0.1, 0.0, 0.1], "pct"), [-0.1, 0.0, 0.1])
        self.assertIn(-0.05, charts.labelled_ticks([-0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], "pct"))
        cases = ((money, "money"), ([-6e6, -4e6, -2e6, 0.0, 2e6], "money"), ([0.0, 0.01, 0.02, 0.03], "pct"))
        for ticks, kind in cases:
            spans = []
            for tick in charts.labelled_ticks(ticks, kind):
                x = (tick - ticks[0]) / (ticks[-1] - ticks[0]) * charts.TICK_PLOT_PX * 0.92
                w = charts.label_px(charts.tick_label(tick, kind))
                spans.append((x, x + w) if tick == ticks[0] else (x - w, x) if tick == ticks[-1]
                             else (x - w / 2, x + w / 2))
            for (_, right), (left, _) in zip(spans, spans[1:]):
                self.assertLessEqual(right, left, (ticks, spans))  # no two labels overlap
        # The zero label that only fits a wide plot is still drawn, shown by the theme's container query.
        grid = charts._grid_lines(charts.Scale(money[0], money[-1]), money, 100, "money")
        self.assertEqual(re.findall(r'<text [^>]*class="tk-wide">([^<]+)</text>', grid), ["$0"])
        self.assertIn("@container (min-width: 240px) { .fm-chart svg text.tk-wide { display: inline; } }",
                      __import__("ui.theme", fromlist=["TRADING_CSS"]).TRADING_CSS)
        dumbbell = charts.cash_dumbbell_chart(funds)
        self.assertIn("% of TNA", dumbbell)
        self.assertIn("the trades take cash below zero", dumbbell)
        waterfall = charts.waterfall_chart(self.funds["BRK"]["waterfall"])
        self.assertEqual(waterfall.count('<line x1="0"'), 1)  # only the zero line

    def test_summary_css_sticks_only_the_first_column(self) -> None:
        from ui import theme

        css = theme.TRADING_CSS
        self.assertIn("table.fm-summary thead tr.top th:first-child, table.fm-summary tbody td:first-child", css)
        self.assertNotIn("table.fm-summary th:first-child, table.fm-summary td:first-child", css)
        self.assertIn("table.fm-summary tr.hl-bad td:first-child { background: linear-gradient", css)
        self.assertIn(".fm-kpi.breaks .s { color: var(--fm-muted); }", css)
        self.assertIn('[class*="st-key-tcjump"] { display: none !important; }', css)


def round_two_dashboard() -> dict:
    """tdt.RoundTwoTests' funds: EMPTY (empty ex-post book) and NOTNA (no trade-date TNA) rest on missing data; LIQ
    and OK break on complete data; OK's T-1 cash (−$5,000) differs in sign from its trade-date cash ($3,000)."""
    case = tdt.RoundTwoTests()
    case.setUp()
    return case.dash


def restated_treasury_dashboard() -> dict:
    """DOGG-like: an equity buy plus a staged treasury quantity that only restates the bill held (flagged)."""
    ante = tdt._fund("TB", equity=tdt._frame("eqyticker", [("AAA", 10.0, 10.0, 0.0, 10.0)]),
                     treasury=tdt._frame("cusip", [("912797XX1", 1000.0, 1000.0, 0.0, 99.0)]),
                     prev_cash=500.0, prev_tna=10_000.0)
    post = tdt._fund("TB", equity=tdt._frame("eqyticker", [("AAA", 10.0, 20.0, 10.0, 10.0)]),
                     treasury=tdt._frame("cusip", [("912797XX1", 1000.0, 1000.0, 1000.0, 99.0)]))
    return tdt._dashboard({"TB": (ante, post)})


class RoundTwoBuildersTest(unittest.TestCase):
    """Missing trade-date data, cash basis notes, summary columns, the badge count and the drill-down header."""

    def setUp(self) -> None:
        self.d = round_two_dashboard()
        self.funds = {f["fund"]: f for f in self.d["funds"]}

    def test_results_on_missing_data_are_not_shown_as_breaches(self) -> None:
        kpis = tr.kpi_html(self.d)
        self.assertIn('<div class="v">2</div><div class="s">2 rule(s) pass before and fail after · 2 more on missing '
                      "data (∅)</div>", kpis)
        self.assertIn('class="fm-kpi miss"', kpis)
        self.assertIn("Rests on missing data", kpis)
        self.assertIn("2 would break; trade-date data missing, not counted above", kpis)
        matrix = tr.matrix_html(self.d, only_changes=False)
        self.assertEqual(matrix.count("∅ Data missing"), 2)
        self.assertEqual(matrix.count('class="fm-tx breaks miss"'), 3)  # EMPTY, NOTNA (dashed), the legend
        self.assertEqual(matrix.count('<tr class="hl-bad">'), 2)  # LIQ and OK only
        self.assertIn("Rests on missing data · EMPTY", matrix)
        self.assertIn("results rest on missing trade-date data, not counted", matrix)  # the legend
        risk = tr.rules_at_risk_html(self.d)
        self.assertIn("4 of 4 fail after · 2 on missing data", risk)
        self.assertIn('<i class="miss" style="width:50.00%"', risk)
        self.assertIn('class="gap" data-fm-jump="EMPTY"', risk)
        notes = tr.data_notes_html(self.d)
        self.assertIn("the rule results of EMPTY, NOTNA rest on missing trade-date data (ex-post holdings (the book "
                      "is empty); trade-date custodian TNA / total assets)", notes)
        self.assertIn("T-1 custodian cash differs sharply from the trade-date figure for 1 fund(s) (OK)", notes)
        icon = tr.missing_icon(self.funds["EMPTY"])
        self.assertIn('class="fm-ico warn"', icon)
        self.assertIn("Rests on missing data (ex-post holdings (the book is empty))", icon)
        head = dd.fund_header_html(self.funds["NOTNA"])
        self.assertIn("∅ Data missing", head)
        self.assertNotIn('class="chg bad"', head)

    def test_cash_tile_shows_the_trade_date_total_when_t_minus_1_differs(self) -> None:
        html = tr.kpi_html(self.d)
        self.assertIn("trade-date cash $3.0K (1 fund(s) differ sharply)", html)

    def test_badge_counts_the_drill_down_notes(self) -> None:
        for fund in self.d["funds"] + real_dashboard()["funds"]:
            count = len(fund["warnings"])
            badge = re.search(r'aria-label="(\d+) data notes"', tr._flags(fund))
            self.assertEqual(int(badge.group(1)) if badge else 0, count, fund["fund"])
            if count:
                self.assertIn(f"Data notes for {fund['fund']} ({count}):", dd.warnings_html(fund))

    def test_summary_keeps_options_and_treasury_sides_and_the_priced_traded_share(self) -> None:
        d = restated_treasury_dashboard()
        html = tr.summary_table_html(d)
        self.assertIn('<th colspan="1" class="c grp">Options</th>', html)  # no option trades: one compact column
        self.assertIn(f"<span>{tr.NONE_STAGED}</span>", html)
        self.assertIn('<th colspan="3" class="c grp">Treasury</th>', html)
        self.assertEqual(html.count('<th class="num grp">Buys</th><th class="num">Sells</th><th class="num">Net</th>'),
                         2)  # equity and treasury
        self.assertIn("flagged · not in est.", html)  # the restated bill: muted, not a trade
        self.assertIn('<td class="num grp">1.00%<span class="sub"', html)  # priced: the $100 equity buy only
        self.assertIn("report 10.90%", html)  # the report's figure incl. the $990 restated bill
        self.assertIn('class="num grp muted flagged" title="The report&#x27;s $990 includes $990 of flagged', html)
        self.assertIn('<s>+$990</s>', html)  # the treasury net: struck through, not a trade

    def test_drill_down_status_reads_as_violation_counts(self) -> None:
        fund = dict(self.funds["OK"], status_change="UNCHANGED", violations_before=1, violations_after=1,
                    impact="STILL_FAILS", worsens=True)
        head = dd.fund_header_html(fund)
        self.assertIn("Breach deepens", head)
        self.assertIn("Violations unchanged (1 → 1)", head)
        self.assertNotIn("Violations<b>", head)
        self.assertEqual(dd.status_change_text(dict(fund, violations_after=3)),
                         "Overall status unchanged · violations up (1 → 3)")
        self.assertIn('class="chg bad"', dd.fund_header_html(dict(fund, violations_after=3)))
        card = dd.metrics_card_html(dict(fund, metrics=[{"label": "Equity", "before": "1", "after": "2",
                                                         "changed": True}]))
        self.assertIn("can differ from the trading report&#x27;s ex-post market value", card)


class TradingPageTest(AppTestCase):
    def staged(self) -> dict:
        return STAGED

    def trading(self):
        at = self.app()
        return self.go(at, trading_page.PAGE)

    def test_nav_puts_trading_second(self) -> None:
        at = self.app()
        self.assertEqual(
                         [o.content for o in at.button_group(key="page").proto.options][:2],
                         ["New run", "Trading Compliance"])
        at = self.go(at, trading_page.PAGE)
        self.assertEqual(at.title[0].value, "Trading Compliance")

    def test_form_chips_select_all_and_validation(self) -> None:
        at = self.trading()
        page = self.shown(at)
        self.assertIn("Trading check preview", page)
        self.assertIn("Ex-ante: the holdings as they stand", page)
        self.assertIn("2 funds have staged trades", page)
        self.assertIn(".st-key-tcf_DOGG label::after, .st-key-tcf_KNG label::after", page)  # the tile dots
        self.assertIn("Pick at least one fund", page)
        self.assertTrue(at.button(key="tc_run").disabled)
        self.assertIn("RUN TRADING CHECK", [b.label for b in at.button if b.key == "tc_run"][0])
        self.assertFalse(at.checkbox(key="tc_email").value)  # email is off by default
        self.assertTrue(all(at.checkbox(key=f"tcr_{k}").value for k in self.checks()))
        chip = at.button(key="tc_chip__extra0")
        self.assertEqual(chip.label, "Staged trades · 2")
        self.ok(chip.click().run())
        self.assertEqual(at.session_state["tc_funds"], ["DOGG", "KNG"])
        self.assertTrue(at.checkbox(key="tcf_DOGG").value)
        self.assertFalse(at.button(key="tc_run").disabled)
        self.assertEqual(at.session_state["nr_funds"], [])  # New run keeps its own picks
        self.ok(at.checkbox(key="tc_rules_all").uncheck().run())
        self.assertEqual(at.session_state["tc_rules"], [])
        self.assertIn("Pick at least one compliance check", self.shown(at))
        self.assertTrue(at.button(key="tc_run").disabled)
        self.ok(at.checkbox(key="tcr_diversification_IRS_check").check().run())
        self.assertFalse(at.button(key="tc_run").disabled)
        self.ok(at.checkbox(key="tcf_TDVI").check().run())
        self.assertIn("have no staged trades on this date", self.shown(at))
        self.ok(at.button(key="tc_chip_ETF_FUNDS").click().run())  # a group chip
        self.assertEqual(set(at.session_state["tc_funds"]), {"DOGG", "KNG", "RDVI", "TDVI"})
        self.ok(at.checkbox(key="tc_email").check().run())
        self.assertTrue(at.button(key="tc_run").disabled)  # emailing needs the second tick
        self.ok(at.checkbox(key="tc_email_ok").check().run())
        self.assertFalse(at.button(key="tc_run").disabled)
        self.ok(at.button(key="tc_reset").click().run())
        self.assertEqual(at.session_state["tc_funds"], [])
        self.assertFalse(at.checkbox(key="tc_email").value)
        self.assertTrue(at.checkbox(key="tc_rules_all").value)

    def checks(self):
        from ui import new_run

        return [k for k, _ in new_run.check_rules(CATALOG)]

    def test_staged_lookup_failure_still_lets_funds_be_picked(self) -> None:
        with mock.patch.object(api, "staged_trades", side_effect=RuntimeError("down")):
            at = self.trading()
        self.assertIn("Could not look up the staged trades (RuntimeError)", self.shown(at))
        self.assertNotIn("tc_chip__extra0", [b.key for b in at.button])
        self.pick_tc(at, "KNG")
        self.assertFalse(at.button(key="tc_run").disabled)

    def pick_tc(self, at, *tickers):
        for ticker in tickers:
            self.ok(at.checkbox(key=f"tcf_{ticker}").check().run())
        return at

    def run_trading(self, at) -> Path:
        self.ok(at.button(key="tc_run").click().run())
        run_dir = Path(at.session_state["tc_watch"])
        self.assertTrue(runner.wait(run_dir, timeout=30))
        self.ok(at.run())
        return run_dir

    def test_run_shows_the_dashboard(self) -> None:
        at = self.pick_tc(self.trading(), "DOGG", "KNG", "RDVI", "TDVI")
        self.ok(at.date_input(key="tc_date").set_value(date(2026, 10, 30)).run())
        run_dir = self.run_trading(at)
        sent = history.load_request(run_dir)
        self.assertEqual((sent.operations, sent.start_date, sent.send_email, sent.output_tag),
                         (("trading_compliance",), date(2026, 10, 30), False, "trading_compliance"))
        self.assertTrue(history.is_trading(run_dir))
        (run_dir / runner.VIEWS).write_text(json.dumps(fx.trading_dashboard_views()), encoding="utf-8")
        self.ok(at.run())
        page = self.shown(at)
        self.assertIn("Next check", page)
        for text in ("Going out of compliance", "Coming into compliance", "Compliance impact", "Rules at risk",
                     "Trading summary", "Trades at a glance", "Fund drill-down", "Cash waterfall",
                     "Rules before → after", "Trade blotter", "No proposed trades:</b> RDVI", "negative cash"):
            self.assertIn(text, page)
        self.assertIn("<svg", self.texts(at.markdown))  # the charts go through Markdown (st.html strips SVG)
        self.assertNotIn("Hunter2Secret", page)  # warnings are redacted
        fund = at.button_group(key=f"tc-tcfund-{run_dir.name}")
        self.assertEqual(fund.value, "DOGG")  # breaks first
        self.assertIn("Data notes for DOGG", page)
        self.ok(fund.set_value("TDVI").run())
        self.assertIn("Estimated post-trade cash is negative.", self.shown(at))
        self.ok(at.selectbox(key=f"tc-tcrule-{run_dir.name}").set_value("real_estate_check").run())
        self.assertEqual(at.button_group(key=f"tc-tcfund-{run_dir.name}").value, "KNG")
        self.ok(at.toggle(key=f"tc-tc-only-{run_dir.name}").set_value(False).run())
        self.assertIn(">GICS</th>", self.shown(at))
        self.assertGreaterEqual(len(at.download_button), 2)  # the downloads under the dashboard

    def test_history_opens_the_dashboard_and_rerun_loads_the_trading_page(self) -> None:
        at = self.pick_tc(self.trading(), "DOGG", "KNG")
        self.ok(at.checkbox(key="tc_pdf").uncheck().run())
        run_dir = self.run_trading(at)
        (run_dir / runner.VIEWS).write_text(json.dumps(fx.trading_dashboard_views()), encoding="utf-8")
        self.ok(at.button(key="tc_reset").click().run())
        self.go(at, "Run history")
        self.assertIn("Compliance impact", self.shown(at))
        self.assertEqual(at.button_group(key=f"hist-tcfund-{run_dir.name}").value, "DOGG")
        self.ok(at.button(key="hist_rerun").click().run())
        self.assertEqual(at.title[0].value, "Trading Compliance")
        self.assertEqual(at.session_state["tc_funds"], ["DOGG", "KNG"])
        self.assertFalse(at.checkbox(key="tc_pdf").value)
        self.assertFalse(at.checkbox(key="tc_email").value)
        self.assertIn("Loaded the settings of", self.shown(at))
        self.go(at, "Reports")
        self.assertIn("Compliance impact", self.shown(at))

    def test_dashboard_error_falls_back_to_the_overview(self) -> None:
        at = self.pick_tc(self.trading(), "DOGG")
        run_dir = self.run_trading(at)
        views = fx.trading_views()
        views["days"][fx.DAY]["dashboard"] = None
        views["days"][fx.DAY]["dashboard_error"] = "KeyError: 'cash'"
        (run_dir / runner.VIEWS).write_text(json.dumps(views), encoding="utf-8")
        self.ok(at.run())
        page = self.shown(at)
        self.assertIn("The trading dashboard could not be built", page)
        self.assertIn("Out Of Compliance", page)  # the report's overview

    def test_jump_opens_the_drilldown_tab(self) -> None:
        at = self.pick_tc(self.trading(), "DOGG", "KNG")
        run_dir = self.run_trading(at)
        (run_dir / runner.VIEWS).write_text(json.dumps(fx.trading_dashboard_views()), encoding="utf-8")
        self.ok(at.run())
        names = tr.keys("tc", run_dir.name)
        self.assertEqual(at.session_state[names["tabs"]], tr.TABS[0])
        self.ok(at.button_group(key=names["jump"]).set_value("KNG #2").run())  # what a click on a cell does
        self.assertEqual(at.session_state[names["tabs"]], tr.DRILL_TAB)
        self.assertEqual(at.button_group(key=names["fund"]).value, "KNG")
        self.assertEqual(at.selectbox(key=names["rule"]).value, "real_estate_check")
        self.assertIn("t-cures focus", self.shown(at))

    def test_refreshed_session_reattaches_a_trading_run_to_its_page(self) -> None:
        at = self.pick_tc(self.trading(), "KNG")
        self.ok(at.text_input(key="tc_tag").input("slow").run())
        self.ok(at.button(key="tc_run").click().run())
        run_dir = Path(at.session_state["tc_watch"])
        fresh = self.app()  # a refresh lands on New run
        self.assertEqual(fresh.session_state["tc_watch"], str(run_dir))
        self.assertNotIn("nr_watch", fresh.session_state)
        runner.cancel(run_dir, by="test")
        self.assertTrue(runner.wait(run_dir, timeout=30))
        self.go(fresh, trading_page.PAGE)
        page = self.shown(fresh)
        self.assertNotIn("Trading check preview", page)  # the finished run, not the empty preview
        self.assertIn("Next check", page)

    def test_rerun_of_a_range_says_which_date_is_loaded(self) -> None:
        at = self.pick_tc(self.trading(), "KNG")
        run_dir = self.run_trading(at)
        request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
        request.update(start_date="2026-10-28", end_date="2026-10-30")
        (run_dir / "request.json").write_text(json.dumps(request), encoding="utf-8")
        self.go(at, "Run history")
        self.ok(at.button(key="hist_rerun").click().run())
        self.assertEqual(at.session_state["tc_date"], date(2026, 10, 28))
        self.assertIn("The original run covered 2026-10-28 to 2026-10-30", self.shown(at))

    def test_new_run_shows_a_trading_run_in_progress_but_keeps_its_form(self) -> None:
        at = self.pick_tc(self.trading(), "KNG")
        self.ok(at.text_input(key="tc_tag").input("slow").run())
        self.ok(at.button(key="tc_run").click().run())
        run_dir = Path(at.session_state["tc_watch"])
        self.go(at, "New run")
        self.assertIn("A trading compliance run is in progress", self.shown(at))
        self.assertTrue(at.button(key="nr_run").disabled)
        self.assertNotIn("nr_watch", at.session_state)
        runner.cancel(run_dir, by="test")
        self.assertTrue(runner.wait(run_dir, timeout=30))


if __name__ == "__main__":
    unittest.main()
