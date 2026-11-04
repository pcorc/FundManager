"""Small hand-made ``views.json`` contents (schema of ``reporting.result_views``) for the UI tests; no real data."""
from __future__ import annotations

DAY = "2026-10-30"


def _rule(title, label, status, rows, details=(), tables=(), notes=(), reason=None):
    return {"title": title, "label": label, "status": status, "reason": reason, "rows": list(rows),
            "details": list(details), "tables": list(tables), "notes": list(notes)}


def eod_views() -> dict:
    irs_kng = _rule("IRS Diversification", "IRS Diversification", "FAIL", [
        {"label": "Overall Status", "value": "FAIL", "status": "FAIL"},
        {"label": "Condition 2a 5%", "value": "FAIL", "status": "FAIL"},
        {"label": "Condition 2b", "value": "N/A", "status": "N/A", "reason": "N/A (not evaluated)"},
        {"label": "Sum Large Securities %", "value": "50.37%"},
    ], details=[{"label": "Total Assets", "value": "8,338,122.00"},
                {"label": "Large Securities", "value": "(<META>, 10.56%), (AMZN, 10.40%)"}],
        notes=["No shares outstanding data (left out of the ownership test): GIB"])
    forty = _rule("40 Act Diversification", "40 Act Diversification", "PASS", [
        {"label": "Overall Status", "value": "PASS", "status": "PASS"},
        {"label": "Registered Status", "value": "Non-Diversified"}])
    gics = _rule("GICS Diversification", "GICS Concentration", "PASS", [
        {"label": "Overall GICS Compliance", "value": "PASS", "status": "PASS"}],
        tables=[{"title": "Industry exposure", "columns": ["Category", "Fund Weight"],
                 "rows": [["Software", "29.88%"], ["Banks", "5.00%"]], "total": 12}])
    summary = _rule("Summary Metrics", "summary_metrics", None, [{"label": "Total Assets", "value": "3,596,378,773"}])
    compliance = {
        "KNG": {"summary_metrics": summary, "gics_compliance": gics, "diversification_40act_check": forty,
                "diversification_IRS_check": irs_kng},
        "TDVI": {"summary_metrics": summary, "gics_compliance": gics, "diversification_40act_check": forty,
                 "diversification_IRS_check": dict(irs_kng, status="PASS", rows=[
                     {"label": "Overall Status", "value": "PASS", "status": "PASS"}])},
    }
    return {"schema_version": 1, "days": {DAY: {
        "kind": "eod", "date": DAY, "funds": ["KNG", "TDVI"], "fund_errors": {"TDVI": ["TDVI: price feed was late"]},
        "rules": {"summary_metrics": {"title": "Summary Metrics", "label": "summary_metrics", "footnotes": []},
                  "gics_compliance": {"title": "GICS Diversification", "label": "GICS Concentration", "footnotes": []},
                  "diversification_40act_check": {"title": "40 Act Diversification", "label": "40 Act Diversification",
                                                  "footnotes": ["Condition 40 Act 1: 75% of the fund diversified."]},
                  "diversification_IRS_check": {"title": "IRS Diversification", "label": "IRS Diversification",
                                                "footnotes": ["IRS Condition 1: 90% of income qualifying."]}},
        "compliance": compliance,
        "compliance_summary": {
            "KNG": {"status": "FAIL", "counts": {"PASS": 2, "FAIL": 1, "N/A": 0, "ERROR": 0},
                    "failed": ["diversification_IRS_check"]},
            "TDVI": {"status": "PASS", "counts": {"PASS": 3, "FAIL": 0, "N/A": 0, "ERROR": 0}, "failed": []}},
        "nav": {"KNG": {"status": "FAIL", "summary": [{"label": "NAV Diff (2-dec)", "value": "-0.19", "status": "FAIL"}],
                        "rows": [{"label": "Expected TNA", "value": "$3,417,433,904"}],
                        "sections": [{"title": "Gain/Loss Components",
                                      "rows": [{"label": "Equity G/L", "value": "$14,886,379.72"}]}],
                        "checks": [{"label": "NAV Good 2-Dec", "status": "FAIL"}],
                        "gl": [{"label": "G/L Today", "value": "11,615,167"}], "tables": []}},
        "holdings": {"KNG": {"status": "FAIL", "holdings_breaks": 80, "price_breaks": 1,
                             "counts": [{"label": "Equity Holdings", "value": 80, "kind": "holdings"},
                                        {"label": "Equity Price", "value": 1, "kind": "price"}],
                             "sections": [{"key": "custodian_equity", "title": "Custodian Equity", "tables": [
                                 {"title": "Holdings Discrepancies", "columns": ["eqy_ticker", "discrepancy_type"],
                                  "rows": [["XOM", "Quantity Mismatch"]], "total": 80}]}]}},
    }}}


def trading_views() -> dict:
    irs = _rule("IRS Diversification", "IRS Diversification", "FAIL",
                [{"label": "Overall Status", "value": "FAIL", "status": "FAIL"}])
    before = dict(irs, status="PASS", rows=[{"label": "Overall Status", "value": "PASS", "status": "PASS"}])
    return {"schema_version": 1, "days": {DAY: {
        "kind": "trading", "date": DAY, "funds": ["DOGG"], "fund_errors": {},
        "rules": {"diversification_IRS_check": {"title": "IRS Diversification", "label": "IRS Diversification",
                                                "footnotes": []}},
        "compliance": {"DOGG": {"diversification_IRS_check": irs}},
        "compliance_ex_ante": {"DOGG": {"diversification_IRS_check": before}},
        "compliance_summary": {"DOGG": {"status": "FAIL", "counts": {"PASS": 0, "FAIL": 1, "N/A": 0, "ERROR": 0},
                                        "failed": ["diversification_IRS_check"]}},
        "trading": {
            "date": DAY, "summary": {"total_funds_analyzed": 1, "total_funds_traded": 1},
            "overview": {"columns": ["Fund Name", "Status Change", "Violations Before", "Violations After", "Net Change"],
                         "rows": [{"fund": "DOGG", "status_change": "OUT_OF_COMPLIANCE", "violations_before": 0,
                                   "violations_after": 1, "net_change": 1, "highlight": "out"}],
                         "totals": {"violations_before": 0, "violations_after": 1, "net_change": 1}},
            "checks": {"DOGG": [{"key": "diversification_IRS_check", "label": "IRS Diversification",
                                 "before": "PASS", "after": "FAIL", "changed": True}]},
            "activity": {"columns": ["Fund", "Asset Class", "Trade Value"],
                         "rows": [{"fund": "DOGG", "asset_class": "Equity",
                                   "cells": [{"label": "Trade Value", "value": "7,240,344.52"}]}]},
            "activity_details": {"DOGG": [{"asset_class": "Equity", "direction": "Sell", "ticker": "MRK",
                                           "quantity": "3,440.00", "market_value": "514,348.80"}]}},
    }}}



# ---------------------------------------------------------------------------
# Trading dashboard (reporting.trading_dashboard schema 1): DOGG breaks IRS, KNG cures Real Estate,
# TDVI stays compliant but its estimated post-trade cash is negative, RDVI has no proposed trades.
# ---------------------------------------------------------------------------

RULES = (("gics_compliance", "GICS Concentration", "GICS Diversification"),
         ("diversification_IRS_check", "IRS Diversification", "IRS Diversification"),
         ("real_estate_check", "Real Estate Exposure", "Real Estate"))
LEAK = "mysql+pymysql://fmuser:Hunter2Secret@db.internal/fm"


def _cell(before, after, transition):
    return {"before": before, "after": after, "transition": transition}


def _assets(equity=(0.0, 0.0), options=(0.0, 0.0), treasury=(0.0, 0.0), tna=1.0):
    out = {}
    for name, unit, (buy, sell) in (("equity", "shares", equity), ("options", "contracts", options),
                                    ("treasury", "face", treasury)):
        out[name] = {"label": name.title(), "unit": unit, "staged": bool(buy or sell), "buy_value": buy,
                     "sell_value": sell, "net_value": buy - sell, "gross_value": buy + sell,
                     "buy_quantity": buy / 100, "sell_quantity": sell / 100, "buy_count": int(bool(buy)),
                     "sell_count": int(bool(sell)), "buy_pct_tna": buy / tna, "sell_pct_tna": sell / tna,
                     "net_pct_tna": (buy - sell) / tna, "gross_pct_tna": (buy + sell) / tna, "cash_effect": sell - buy,
                     "ex_ante_market_value": 1000.0, "ex_post_market_value": 1000.0 + buy - sell}
    return out


def _waterfall(current, equity_buy, equity_sell, options_net=0.0, treasury_net=0.0):
    steps, running = [], current
    steps.append({"key": "current_cash", "label": "Current cash", "value": current, "running": current, "kind": "start"})
    for key, label, value in (("equity_buys", "Equity buys", -equity_buy), ("equity_sells", "Equity sells", equity_sell),
                              ("options_net", "Options net premium", options_net),
                              ("treasury_net", "Treasury net", treasury_net)):
        running += value
        steps.append({"key": key, "label": label, "value": value, "running": running, "kind": "delta"})
    steps.append({"key": "post_cash", "label": "Post-trade cash (est.)", "value": running, "running": running,
                  "kind": "end"})
    return steps


def _fund(name, impact, transitions, *, tna, equity, options=(0.0, 0.0), cash, trades, rules, warnings=(),
          status_change="UNCHANGED", violations=(0, 0)):
    buy = equity[0] + options[0]
    sell = equity[1] + options[1]
    post = cash - buy + sell
    return {
        "fund": name, "has_trades": True, "trade_count": len(trades), "impact": impact, "status_change": status_change,
        "violations_before": violations[0], "violations_after": violations[1], "tna": tna, "total_assets": tna * 1.01,
        "assets": _assets(equity, options, tna=tna), "buy_value": buy, "sell_value": sell, "net_value": buy - sell,
        "gross_value": buy + sell, "gross_pct_tna": (buy + sell) / tna,
        "cash": {"current": cash, "source": "ex_ante_previous", "date": "2026-10-29", "custodian_t": cash,
                 "post": post, "change": post - cash, "post_pct_tna": post / tna, "negative": post < 0,
                 "estimate": True, "excluded": [], "excluded_value": 0},
        "waterfall": _waterfall(cash, equity[0], equity[1], options[1] - options[0]),
        "transitions": transitions, "rules": rules,
        "metrics": [{"label": "Cash", "before": f"{cash:,.0f}", "after": f"{cash:,.0f}", "changed": False},
                    {"label": "Equity", "before": "1,000,000", "after": f"{1000000 + equity[0] - equity[1]:,.0f}",
                     "changed": bool(equity[0] or equity[1])}],
        "blotter": [{"asset_class": "equity", "label": "Equity", "unit": "shares", "buy_value": equity[0],
                     "sell_value": equity[1], "trades": trades}],
        "warnings": list(warnings)}


def _trade(ticker, side, quantity, price, flags=()):
    return {"ticker": ticker, "side": side, "quantity": quantity, "price": price, "multiplier": 1.0,
            "value": quantity * price, "pct_tna": 0.01, "before_qty": 1000.0,
            "after_qty": 1000.0 + (quantity if side == "BUY" else -quantity), "ex_post_qty": 1000.0,
            "effect": "increase" if side == "BUY" else "reduce", "flags": list(flags), "in_cash": not flags}


def _diff(rule, title, before, after, transition, rows, unchanged=False):
    return {"rule": rule, "title": title, "label": title, "before": before, "after": after, "transition": transition,
            "changed_rows": sum(1 for r in rows if r.get("changed")), "unchanged": unchanged, "rows": rows,
            "details": []}


def trading_dashboard() -> dict:
    ok = _cell("PASS", "PASS", "OK")
    dogg_rules = [
        _diff("diversification_IRS_check", "IRS Diversification", "PASS", "FAIL", "BREAKS",
              [{"label": "Overall Status", "before": "PASS", "after": "FAIL", "changed": True,
                "status_before": "PASS", "status_after": "FAIL"},
               {"label": "Condition 2a 50% weight", "before": "12.1%", "after": "14.3%", "changed": True}]),
        _diff("gics_compliance", "GICS Diversification", "PASS", "PASS", "OK",
              [{"label": "Overall GICS Compliance", "before": "PASS", "after": "PASS", "changed": False,
                "status_before": "PASS", "status_after": "PASS"}], unchanged=True),
    ]
    kng_rules = [
        _diff("real_estate_check", "Real Estate", "FAIL", "PASS", "CURES",
              [{"label": "Real Estate %", "before": "26.0%", "after": "24.0%", "changed": True}]),
    ]
    funds = [
        _fund("DOGG", "BREAKS", {"gics_compliance": ok, "diversification_IRS_check": _cell("PASS", "FAIL", "BREAKS"),
                                 "real_estate_check": ok},
              tna=100_000_000.0, equity=(5_000_000.0, 1_000_000.0), options=(0.0, 125_400.0), cash=10_000_000.0,
              trades=[_trade("MRK", "BUY", 1000.0, 149.52), _trade("<b>X</b>", "SELL", 10.0, 5.0, ["no_price"])],
              rules=dogg_rules, status_change="OUT_OF_COMPLIANCE", violations=(0, 1),
              warnings=["1 equity trade(s) have no usable price.", f"Lookup failed: {LEAK}"]),
        _fund("KNG", "CURES", {"gics_compliance": ok, "diversification_IRS_check": ok,
                               "real_estate_check": _cell("FAIL", "PASS", "CURES")},
              tna=50_000_000.0, equity=(0.0, 2_000_000.0), cash=500_000.0,
              trades=[_trade("AMT", "SELL", 10000.0, 200.0)], rules=kng_rules, status_change="INTO_COMPLIANCE",
              violations=(1, 0)),
        _fund("TDVI", "COMPLIANT", {"gics_compliance": ok, "diversification_IRS_check": ok, "real_estate_check": ok},
              tna=20_000_000.0, equity=(3_000_000.0, 0.0), cash=1_000_000.0,
              trades=[_trade("MSFT", "BUY", 6000.0, 500.0)], rules=[], warnings=["Estimated post-trade cash is negative."]),
    ]
    matrix_rules = [{"key": k, "label": label, "title": title, "changes_or_fails": k != "gics_compliance"}
                    for k, label, title in RULES]
    totals_assets = {a: {k: sum(f["assets"][a][k] for f in funds) for k in
                         ("buy_value", "sell_value", "net_value", "gross_value", "buy_quantity", "sell_quantity")}
                     for a in ("equity", "options", "treasury")}
    for a in totals_assets:
        totals_assets[a]["gross_pct_tna"] = totals_assets[a]["gross_value"] / 170_000_000.0
    cash = sum(f["cash"]["current"] for f in funds)
    post = sum(f["cash"]["post"] for f in funds)
    buy = sum(f["buy_value"] for f in funds)
    sell = sum(f["sell_value"] for f in funds)
    return {
        "schema": 1, "date": DAY, "cash_date": "2026-10-29",
        "definitions": {"tna": "Custodian TNA at the T-1 close.", "current_cash": "Custodian cash at the T-1 close.",
                        "trade_value": "|quantity| x price x multiplier.", "net_value": "Buys minus sells.",
                        "post_cash": "Estimate: current cash - buys + sells.", "pct_tna": "Fractions of TNA.",
                        "transitions": {"BREAKS": "Passes before the trades, fails after them",
                                        "STILL_FAILS": "Fails before and after the trades",
                                        "FAILS": "Fails after the trades (no ex-ante verdict)",
                                        "ERROR": "The check raised an error on one side",
                                        "CURES": "Fails before the trades, passes after them",
                                        "OK": "Passes before and after the trades",
                                        "N/A": "Not applicable, skipped or not run on one side"},
                        "impacts": {}},
        "kpis": {"funds_selected": 4, "funds_with_trades": 3, "funds_without_trades": 1, "funds_unavailable": 0,
                 "buy_value": buy, "sell_value": sell, "net_value": buy - sell, "gross_value": buy + sell,
                 "equity_buy_value": 8_000_000.0, "equity_sell_value": 3_000_000.0, "options_buy_value": 0.0,
                 "options_sell_value": 125_400.0, "treasury_buy_value": 0.0, "treasury_sell_value": 0.0,
                 "funds_compliant": 1, "funds_breaking": 1, "funds_curing": 1, "funds_still_failing": 0,
                 "funds_error": 0, "funds_unknown": 0, "rule_breaks": 1, "rule_cures": 1, "rule_still_fails": 0,
                 "funds_negative_cash": 1, "funds_with_warnings": 2},
        "funds": funds,
        "no_trade_funds": [{"fund": "RDVI", "impact": "COMPLIANT", "status_after": "PASS", "failed_after": [],
                            "report_trade_value": 0.0}],
        "unavailable_funds": [],
        "matrix": {"rules": matrix_rules,
                   "rows": [{"fund": f["fund"], "impact": f["impact"], "cells": f["transitions"]} for f in funds]},
        "rules_at_risk": [
            {"rule": "diversification_IRS_check", "label": "IRS Diversification", "title": "IRS Diversification",
             "breaks": ["DOGG"], "still_fails": [], "fails": [], "errors": [], "cures": []},
            {"rule": "real_estate_check", "label": "Real Estate Exposure", "title": "Real Estate",
             "breaks": [], "still_fails": [], "fails": [], "errors": [], "cures": ["KNG"]}],
        "totals": {"funds": 3, "tna": 170_000_000.0, "assets": totals_assets, "buy_value": buy, "sell_value": sell,
                   "net_value": buy - sell, "gross_value": buy + sell, "gross_pct_tna": (buy + sell) / 170_000_000.0,
                   "cash": cash, "post_cash": post, "cash_change": post - cash, "post_pct_tna": post / 170_000_000.0,
                   "funds_negative_cash": 1, "cash_funds": 3},
    }


def trading_dashboard_views() -> dict:
    """``trading_views()`` with four funds and the dashboard block."""
    views = trading_views()
    day = views["days"][DAY]
    day["funds"] = ["DOGG", "KNG", "RDVI", "TDVI"]
    day["dashboard"] = trading_dashboard()
    return views
