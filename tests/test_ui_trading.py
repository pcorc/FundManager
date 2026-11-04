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
        self.assertIn("No custodian cash for NOCASH", markup)

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
        self.assertIn('class="fm-kpi warn"', html)  # a fund below zero after the trades
        self.assertIn("1 fund(s) below zero", html)
        self.assertIn("Still failing", html)

    def test_matrix_transitions_and_toggle(self) -> None:
        d = dashboard()
        html = tr.matrix_html(d, only_changes=True)
        self.assertIn('class="fm-tx breaks"', html)
        self.assertIn('class="fm-tx cures"', html)
        self.assertIn("DOGG · IRS Diversification: PASS before → FAIL after", html)
        self.assertIn('<tr class="hl-bad"><td class="fund">DOGG</td>', html)
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
        self.assertRegex(html, r'class="grp breaks">.*?<span>DOGG</span>')
        self.assertRegex(html, r'class="grp cures">.*?<span>KNG</span>')

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
        self.assertIn("Out of compliance", head)
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
        self.assertEqual(options, [dd.ALL_FUNDS, "IRS Diversification · 1", "Real Estate · 1"])
        self.assertEqual(dd.funds_for_filter(d, "Real Estate · 1"), ["KNG"])
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
        self.ok(at.selectbox(key=f"tc-tcrule-{run_dir.name}").set_value("Real Estate · 1").run())
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
