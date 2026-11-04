"""The Trading Compliance dashboard block of a trading ``DayView`` (``view["dashboard"]``).

Built from the trading analyzer's comparison (the numbers the trading PDF / Excel show) and the
ex-ante / ex-post ``RuleView`` s of :mod:`reporting.result_views`. It only reads; it never writes
files or logs, and nothing here feeds a report. All amounts are dollars, all ``*_pct_tna`` values are
fractions of TNA (0.0769 = 7.69%), quantities are shares / option contracts / treasury face.

Definitions (also in ``dashboard["definitions"]``):

* TNA: the analyzer's ``total_net_assets`` (custodian TNA at the T-1 close, the PDF's TNA column).
* Current cash: custodian cash (``v_custodian_cash``) at the T-1 close, from the ex-ante pass's
  previous snapshot. A trading check runs intraday on T, before the staged trades execute, so the T-1
  close is the opening balance the trades draw on and the latest custodian figure available; it is
  also the snapshot the analyzer takes TNA from. When T-1 has no custodian data the ex-ante T
  snapshot is used, then the ex-post one (``cash.source``).
* Trade value: the analyzer's value of each staged trade (``trade_rebal``): |quantity| x price x
  multiplier, options at premium (price per share x 100 per contract), treasuries at price / 100 x
  face. ``net_value`` = buys - sells (positive = net buying). A fund "has proposed trades" when it has
  staged trade rows; the trading report can also show a trade value for a fund without any (from the
  ex-post share columns), which is kept as ``report_trade_value`` on its ``no_trade_funds`` entry.
* Post-trade cash (an estimate) = current cash - buys + sells over equity, options and treasuries:
  buying pays, selling receives; for options buy-to-open / buy-to-close pays the premium and
  sell-to-open (writing) / sell-to-close receives it. Trades without a usable price (the analyzer then
  values them at quantity x multiplier) and staged quantities that restate the position already held
  (``restates_position``: trade = the ex-ante position and the ex-post position is the same, as the
  DOGG treasury bill carries every day) are left out and listed in ``cash.excluded``. Settlement timing, commissions, accruals, FLEX option trades and other
  cash flows are not included.

Schema::

    Dashboard = {
      "schema": 1, "date": "YYYY-MM-DD", "cash_date": "YYYY-MM-DD" (T-1),
      "definitions": {"tna", "current_cash", "trade_value", "net_value", "post_cash", "pct_tna": str,
                      "transitions": {code: str}, "impacts": {code: str}},
      "kpis": {"funds_selected", "funds_with_trades", "funds_without_trades", "funds_unavailable",
               "buy_value", "sell_value", "net_value", "gross_value",
               "equity_buy_value", "equity_sell_value", "options_buy_value", "options_sell_value",
               "treasury_buy_value", "treasury_sell_value",
               "funds_compliant", "funds_breaking", "funds_curing", "funds_still_failing", "funds_error",
               "funds_unknown", "rule_breaks", "rule_cures", "rule_still_fails", "funds_negative_cash",
               "funds_with_warnings"},        # all over the funds with proposed trades
      "funds": [FundDashboard],            # funds with proposed trades; BREAKS, STILL_FAILS, ERROR, CURES,
                                           # COMPLIANT, UNKNOWN, then by name
      "no_trade_funds": [{"fund", "impact", "status_after", "failed_after": [rule], "report_trade_value"}],
      "unavailable_funds": [{"fund", "errors": [str]}],   # selected but no result on either side
      "matrix": {"rules": [{"key", "label", "title", "changes_or_fails": bool}],
                 "rows": [{"fund", "impact", "cells": {rule: Cell}}]},   # rows in "funds" order
      "rules_at_risk": [{"rule", "label", "title", "breaks", "still_fails", "fails", "errors", "cures": [fund]}],
      "totals": Totals,                    # sums over "funds" (the summary table's totals row)
    }
    Cell = {"before": "PASS"|"FAIL"|"N/A"|"ERROR"|None, "after": ..., "transition": Transition}
    Transition = "OK" (PASS->PASS) | "BREAKS" (PASS->FAIL) | "CURES" (FAIL->PASS) | "STILL_FAILS" (FAIL->FAIL)
               | "FAILS" (FAIL after, no ex-ante verdict) | "ERROR" (either side errored) | "N/A"
               # statuses are the rule cards' (RuleView.status): a skipped rule is N/A here, where the
               # trading report's Compliance Details sheet counts it as PASS
    FundDashboard = {
      "fund", "has_trades": True, "trade_count": int, "impact": Impact, "status_change", "violations_before",
      "violations_after", "tna", "total_assets",
      "assets": {"equity" | "options" | "treasury": AssetStats},   # always all three
      "buy_value", "sell_value", "net_value", "gross_value", "gross_pct_tna",
      "cash": Cash, "waterfall": [Step],
      "transitions": {rule: Cell},
      "rules": [RuleDiff],               # failing / breaking first; "unchanged" ones can be collapsed
      "metrics": [Diff],                 # Summary Metrics rows (cash, equity, options, TA, TNA) before -> after
      "blotter": [{"asset_class", "label", "unit", "buy_value", "sell_value", "trades": [Trade]}],  # classes with trades
      "warnings": [str],
    }
    Impact = "BREAKS" | "STILL_FAILS" | "ERROR" | "CURES" | "COMPLIANT" | "UNKNOWN"
    AssetStats = {"label", "unit", "staged": bool, "buy_value", "sell_value", "net_value", "gross_value",
                  "buy_quantity", "sell_quantity", "buy_count", "sell_count", "buy_pct_tna", "sell_pct_tna",
                  "net_pct_tna", "gross_pct_tna", "cash_effect", "ex_ante_market_value", "ex_post_market_value"}
                  # gross_value / gross_pct_tna = the PDF's Trade Value / % of TNA (without staged rows the
                  # report derives it from the ex-post share columns: staged False, gross_value > 0);
                  # cash_effect = the class's step in the cash estimate (sells - buys of the trades in it)
    Cash = {"current", "source": "ex_ante_previous" | "ex_ante_current" | "ex_post_current" | None,
            "date", "custodian_t" (T custodian cash, for reference), "post", "change", "post_pct_tna",
            "negative": bool, "estimate": True, "excluded": [{"asset_class", "ticker", "side", "value", "reason"}],
            "excluded_value"}         # current / post are None when no custodian cash was loaded
    Step = {"key", "label", "value", "running", "kind": "start" | "delta" | "end"}
           # keys: current_cash, equity_buys, equity_sells, options_net, treasury_net, post_cash
    Trade = {"ticker", "side": "BUY" | "SELL", "quantity", "price" (quoted; None = no usable price),
             "multiplier", "value", "pct_tna",
             "before_qty",            # ex-ante position (nav_shares; 0 = not held; None = no ex-ante data)
             "after_qty",             # before_qty + the signed trade (None without ex-ante data)
             "ex_post_qty",           # the ex-post holdings' iiv_shares for the ticker (as loaded; None if absent)
             "effect": "open" | "close" | "increase" | "reduce" | "flip" | None,
             "flags": ["no_price" | "restates_position" | "duplicate"], "in_cash": bool}
    RuleDiff = {"rule", "title", "label", "before", "after", "transition", "changed_rows": int,
                "unchanged": bool, "rows": [Diff], "details": [Diff]}
    Diff = {"label", "before", "after", "changed": bool, "status_before"?, "status_after"?}  # text as shown
    Totals = {"funds", "tna", "assets": {class: {"buy_value", "sell_value", "net_value", "gross_value",
              "buy_quantity", "sell_quantity", "gross_pct_tna"}}, "buy_value", "sell_value", "net_value",
              "gross_value", "gross_pct_tna", "cash", "post_cash", "cash_change", "post_pct_tna",
              "funds_negative_cash", "cash_funds"}   # cash sums cover the funds with a cash figure
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from reporting.report_utils import normalize_report_date
from reporting.trade_compliance_analyzer import ASSET_ORDER, TradingComplianceAnalyzer
from utilities.numbers import num, to_float
from utilities.workdays import business_day_offset

__all__ = ["DASHBOARD_SCHEMA", "IMPACT_ORDER", "TRANSITIONS", "build_dashboard", "transition"]

DASHBOARD_SCHEMA = 1
SUMMARY = "summary_metrics"
ASSET_LABELS = {"equity": "Equity", "options": "Options", "treasury": "Treasury"}
ASSET_UNITS = {"equity": "shares", "options": "contracts", "treasury": "face"}

TRANSITIONS = {
    "BREAKS": "Passes before the trades, fails after them",
    "STILL_FAILS": "Fails before and after the trades",
    "FAILS": "Fails after the trades (no ex-ante verdict)",
    "ERROR": "The check raised an error on one side",
    "CURES": "Fails before the trades, passes after them",
    "OK": "Passes before and after the trades",
    "N/A": "Not applicable, skipped or not run on one side",
}
IMPACTS = {
    "BREAKS": "A rule that passes before the trades fails after them",
    "STILL_FAILS": "A rule fails after the trades, as it did before (none breaks)",
    "ERROR": "A check raised an error (no rule breaks or fails)",
    "CURES": "The trades bring every failing rule back into compliance",
    "COMPLIANT": "Every rule passes (or is N/A) before and after the trades",
    "UNKNOWN": "No pass/fail verdicts to compare",
}
IMPACT_ORDER = ("BREAKS", "STILL_FAILS", "ERROR", "CURES", "COMPLIANT", "UNKNOWN")
# RuleDiff order: failing / breaking first, then errors, cures, rules whose values move, the rest.
_RULE_RANK = {"BREAKS": 0, "STILL_FAILS": 1, "FAILS": 1, "ERROR": 2, "CURES": 3, "OK": 4, "N/A": 5}

DEFINITIONS = {
    "tna": "Custodian total net assets at the T-1 close (the trading report's TNA column).",
    "current_cash": "Custodian cash at the T-1 close: the opening balance on the trade date, before the "
                    "staged trades execute (the latest custodian figure available intraday).",
    "trade_value": "Each staged trade at |quantity| x price x multiplier: options at premium (100 shares per "
                   "contract), treasuries at price per 100 face. As in the trading report.",
    "net_value": "Buys minus sells (positive = net buying).",
    "post_cash": "Estimate: current cash - buys + sells across equity, options and treasuries (buying, "
                 "including buying options to open or close, pays; selling or writing options receives "
                 "the premium). Trades without a usable price, and staged quantities that only restate "
                 "the position already held, are left out (listed per fund). Settlement timing, commissions, "
                 "accruals, FLEX option trades and other flows are not included.",
    "pct_tna": "Fractions of TNA (0.05 = 5%).",
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_dashboard(
    comparison_data: Optional[Mapping[str, Any]],
    compliance_ex_ante: Mapping[str, Mapping[str, Mapping[str, Any]]],
    compliance_ex_post: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    day: date | datetime | str,
    funds: Iterable[str] = (),
    fund_errors: Optional[Mapping[str, Sequence[str]]] = None,
    rules: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """The dashboard block (see the module docstring); ``rules`` is the DayView's rule title map.

    ``funds`` are the run's funds; those without a comparison or rule results are "unavailable".
    """
    day_str = normalize_report_date(day)
    comparison = comparison_data or {}
    compared = comparison.get("funds", {}) or {}
    rules = rules or {}
    fund_errors = fund_errors or {}
    names = sorted(set(funds) | set(compared) | set(compliance_ex_ante) | set(compliance_ex_post))

    traded: List[Dict[str, Any]] = []
    no_trades: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []
    for name in names:
        fund_data = compared.get(name) or {}
        ante_rules = compliance_ex_ante.get(name) or {}
        post_rules = compliance_ex_post.get(name) or {}
        if not fund_data and not ante_rules and not post_rules:
            unavailable.append({"fund": name, "errors": list(fund_errors.get(name, []))})
            continue
        transitions = _transitions(ante_rules, post_rules)
        impact = _impact(transitions)
        trade_info = fund_data.get("trade_info") or {}
        if not _has_trades(trade_info):
            no_trades.append({"fund": name, "impact": impact, "status_after": _overall(post_rules),
                              "failed_after": [k for k, cell in transitions.items() if cell["after"] == "FAIL"],
                              "report_trade_value": num(trade_info, "total_traded")})
            continue
        entry = _fund_dashboard(name, fund_data, trade_info, ante_rules, post_rules, transitions, impact,
                                rules, day_str)
        traded.append(entry)

    traded.sort(key=lambda f: (IMPACT_ORDER.index(f["impact"]), f["fund"]))
    matrix_rules = _matrix_rules(traded, rules)
    return {
        "schema": DASHBOARD_SCHEMA,
        "date": day_str,
        "cash_date": _previous_business_day(day_str),
        "definitions": {**DEFINITIONS, "transitions": dict(TRANSITIONS), "impacts": dict(IMPACTS)},
        "kpis": _kpis(traded, len(names), len(no_trades), len(unavailable)),
        "funds": traded,
        "no_trade_funds": no_trades,
        "unavailable_funds": unavailable,
        "matrix": {"rules": matrix_rules,
                   "rows": [{"fund": f["fund"], "impact": f["impact"], "cells": f["transitions"]} for f in traded]},
        "rules_at_risk": _rules_at_risk(traded, matrix_rules),
        "totals": _totals(traded),
    }


def _previous_business_day(day_str: str) -> str:
    return business_day_offset(datetime.fromisoformat(day_str).date(), -1).isoformat()


def _has_trades(trade_info: Mapping[str, Any]) -> bool:
    """Staged trade rows (``trade_rebal``) on any asset class."""
    activity = trade_info.get("trade_activity") or {}
    return any((a or {}).get("buys") or (a or {}).get("sells") for a in activity.values())


# ---------------------------------------------------------------------------
# Rule transitions
# ---------------------------------------------------------------------------

def transition(before: Optional[str], after: Optional[str]) -> str:
    """The Transition code of one rule's ex-ante -> ex-post status pair."""
    if "ERROR" in (before, after):
        return "ERROR"
    if before in ("PASS", "FAIL") and after in ("PASS", "FAIL"):
        return {("PASS", "PASS"): "OK", ("PASS", "FAIL"): "BREAKS", ("FAIL", "PASS"): "CURES",
                ("FAIL", "FAIL"): "STILL_FAILS"}[(before, after)]
    return "FAILS" if after == "FAIL" else "N/A"


def _rule_keys(*sides: Mapping[str, Any]) -> List[str]:
    keys: List[str] = []
    for side in sides:
        keys.extend(key for key in side if key != SUMMARY and key not in keys)
    return keys


def _transitions(ante_rules: Mapping[str, Mapping[str, Any]],
                 post_rules: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cells = {}
    for key in _rule_keys(post_rules, ante_rules):
        before = (ante_rules.get(key) or {}).get("status")
        after = (post_rules.get(key) or {}).get("status")
        cells[key] = {"before": before, "after": after, "transition": transition(before, after)}
    return cells


def _impact(transitions: Mapping[str, Mapping[str, Any]]) -> str:
    codes = {cell["transition"] for cell in transitions.values()}
    if "BREAKS" in codes:
        return "BREAKS"
    if codes & {"STILL_FAILS", "FAILS"}:
        return "STILL_FAILS"
    if "ERROR" in codes:
        return "ERROR"
    if "CURES" in codes:
        return "CURES"
    return "COMPLIANT" if "OK" in codes else "UNKNOWN"


def _overall(rules: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    statuses = {(rule or {}).get("status") for key, rule in rules.items() if key != SUMMARY}
    return next((s for s in ("FAIL", "ERROR", "PASS", "N/A") if s in statuses), None)


def _rule_meta(key: str, rules: Mapping[str, Mapping[str, Any]], *views: Mapping[str, Any]) -> Tuple[str, str]:
    meta = rules.get(key) or next((v for v in views if v), {}) or {}
    return str(meta.get("title") or key), str(meta.get("label") or key)


def _diff_rows(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]],
               with_status: bool = True) -> List[Dict[str, Any]]:
    """Rows matched by label (ex-post order, then ex-ante-only rows) with changed values marked."""
    before_by_label = {str(row.get("label")): row for row in before or []}
    after_by_label = {str(row.get("label")): row for row in after or []}
    labels = list(after_by_label) + [label for label in before_by_label if label not in after_by_label]
    rows = []
    for label in labels:
        old, new = before_by_label.get(label) or {}, after_by_label.get(label) or {}
        row: Dict[str, Any] = {"label": label, "before": old.get("value"), "after": new.get("value"),
                               "changed": old.get("value") != new.get("value")}
        if with_status and ("status" in old or "status" in new):
            row.update(status_before=old.get("status"), status_after=new.get("status"))
        rows.append(row)
    return rows


def _rule_diffs(ante_rules, post_rules, transitions, rules) -> List[Dict[str, Any]]:
    diffs = []
    for key, cell in transitions.items():
        before, after = ante_rules.get(key) or {}, post_rules.get(key) or {}
        rows = _diff_rows(before.get("rows") or [], after.get("rows") or [])
        details = _diff_rows(before.get("details") or [], after.get("details") or [], with_status=False)
        changed = sum(row["changed"] for row in rows + details)
        title, label = _rule_meta(key, rules, after, before)
        diffs.append({"rule": key, "title": title, "label": label, "before": cell["before"], "after": cell["after"],
                      "transition": cell["transition"], "changed_rows": changed,
                      "unchanged": cell["transition"] in ("OK", "N/A") and not changed,
                      "rows": rows, "details": details})
    diffs.sort(key=lambda d: (_RULE_RANK[d["transition"]], d["unchanged"]))
    return diffs


def _matrix_rules(traded: Sequence[Mapping[str, Any]], rules: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Matrix columns: every rule of a traded fund, in the DayView's rule order (report order)."""
    keys: List[str] = []
    for fund in traded:
        keys.extend(key for key in fund["transitions"] if key not in keys)
    order = [key for key in rules if key in keys] + [key for key in keys if key not in rules]
    columns = []
    for key in order:
        title, label = _rule_meta(key, rules)
        codes = {f["transitions"][key]["transition"] for f in traded if key in f["transitions"]}
        columns.append({"key": key, "label": label, "title": title,
                        "changes_or_fails": bool(codes - {"OK", "N/A"})})
    return columns


def _rules_at_risk(traded: Sequence[Mapping[str, Any]], columns: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups = (("breaks", "BREAKS"), ("still_fails", "STILL_FAILS"), ("fails", "FAILS"), ("errors", "ERROR"),
              ("cures", "CURES"))
    out = []
    for column in columns:
        key = column["key"]
        entry: Dict[str, Any] = {"rule": key, "label": column["label"], "title": column["title"]}
        for name, code in groups:
            entry[name] = [f["fund"] for f in traded if (f["transitions"].get(key) or {}).get("transition") == code]
        if any(entry[name] for name, _ in groups):
            out.append(entry)
    out.sort(key=lambda e: (-len(e["breaks"]), -(len(e["still_fails"]) + len(e["fails"])), -len(e["errors"]),
                            -len(e["cures"])))
    return out


# ---------------------------------------------------------------------------
# One traded fund
# ---------------------------------------------------------------------------

def _fund_dashboard(name, fund_data, trade_info, ante_rules, post_rules, transitions, impact, rules,
                    day_str) -> Dict[str, Any]:
    tna = num(trade_info, "total_net_assets")
    post_fund = trade_info.get("fund_object") or trade_info.get("fund")
    # The Fund copied after the ex-ante pass; without it there is no ex-ante position or cash data.
    ante_fund = trade_info.get("ex_ante_fund_object")
    warnings: List[str] = []

    blotter = _blotter(trade_info, ante_fund, post_fund, tna, warnings)
    assets = {key: _asset_stats(key, trade_info, tna, blotter) for key in ASSET_ORDER}
    cash = _cash(ante_fund, post_fund, tna, blotter, assets, day_str)
    if cash["current"] is None:
        warnings.append("No custodian cash was loaded for this fund: post-trade cash is not estimated.")
    elif cash["negative"]:
        warnings.append("Estimated post-trade cash is negative.")
    for stats in assets.values():
        if not stats["staged"] and stats["gross_value"] > 0.0:
            warnings.append(f"{stats['label']}: the trading report shows a trade value of "
                            f"{_money(stats['gross_value'])} from the ex-post share columns, without staged trades; "
                            "it is not in the cash estimate.")
    flex = _flex_trade_count(post_fund)
    if flex:
        warnings.append(f"{flex} FLEX option trade row(s) are staged; the trading analysis does not value FLEX "
                        "trades, so they are not in these figures.")

    buy = sum(a["buy_value"] for a in assets.values())
    sell = sum(a["sell_value"] for a in assets.values())
    gross = sum(a["gross_value"] for a in assets.values())
    return {
        "fund": name,
        "has_trades": True,
        "trade_count": sum(len(group["trades"]) for group in blotter),
        "impact": impact,
        "status_change": str(fund_data.get("status_change") or "UNCHANGED"),
        "violations_before": int(fund_data.get("violations_before", 0) or 0),
        "violations_after": int(fund_data.get("violations_after", 0) or 0),
        "tna": tna,
        "total_assets": num(trade_info, "total_assets"),
        "assets": assets,
        "buy_value": buy,
        "sell_value": sell,
        "net_value": buy - sell,
        "gross_value": gross,
        "gross_pct_tna": _pct(gross, tna),
        "cash": cash,
        "waterfall": _waterfall(cash, blotter),
        "transitions": transitions,
        "rules": _rule_diffs(ante_rules, post_rules, transitions, rules),
        "metrics": _diff_rows((ante_rules.get(SUMMARY) or {}).get("rows") or [],
                              (post_rules.get(SUMMARY) or {}).get("rows") or [], with_status=False),
        "blotter": blotter,
        "warnings": warnings,
    }


def _pct(value: Optional[float], tna: float) -> Optional[float]:
    return None if value is None or not tna else value / tna


def _asset_stats(key: str, trade_info: Mapping[str, Any], tna: float,
                 blotter: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """One asset class with the PDF's figures (net_trades / asset_trade_summary) plus its cash effect."""
    net = dict((trade_info.get("net_trades") or {}).get(key) or {})
    summary = dict((trade_info.get("asset_trade_summary") or {}).get(key) or {})
    group = next((g for g in blotter if g["asset_class"] == key), None)
    trades = group["trades"] if group else []
    buy, sell = num(net, "buy_value"), num(net, "sell_value")
    gross = num(summary, "trade_value") or buy + sell
    in_cash = [t for t in trades if t["in_cash"]]
    return {
        "label": ASSET_LABELS[key], "unit": ASSET_UNITS[key], "staged": bool(trades),
        "buy_value": buy, "sell_value": sell, "net_value": buy - sell, "gross_value": gross,
        "buy_quantity": num(net, "buy_quantity"), "sell_quantity": num(net, "sell_quantity"),
        "buy_count": sum(t["side"] == "BUY" for t in trades), "sell_count": sum(t["side"] == "SELL" for t in trades),
        "buy_pct_tna": _pct(buy, tna), "sell_pct_tna": _pct(sell, tna), "net_pct_tna": _pct(buy - sell, tna),
        "gross_pct_tna": num(summary, "pct_of_tna") if "pct_of_tna" in summary else _pct(gross, tna),
        "cash_effect": sum(t["value"] for t in in_cash if t["side"] == "SELL")
        - sum(t["value"] for t in in_cash if t["side"] == "BUY"),
        "ex_ante_market_value": num(summary, "ex_ante_market_value"),
        "ex_post_market_value": num(summary, "ex_post_market_value"),
    }


# ---------------------------------------------------------------------------
# Trade blotter
# ---------------------------------------------------------------------------

def _holdings(fund: Any, key: str) -> pd.DataFrame:
    try:
        return TradingComplianceAnalyzer._get_holdings(fund, key)
    except Exception:  # not a Fund (e.g. a test stand-in): no position data
        return pd.DataFrame()


def _positions(fund: Any, key: str, preferred: str) -> Dict[str, float]:
    """{TICKER: quantity} of the fund's holdings (first row per ticker), as the analyzer names tickers."""
    df = _holdings(fund, key)
    column = TradingComplianceAnalyzer._detect_ticker_column(df, key) if not df.empty else None
    if column is None:
        return {}
    quantities = TradingComplianceAnalyzer._resolve_quantity_series(df, preferred=preferred)
    positions: Dict[str, float] = {}
    for ticker, quantity in zip(df[column], quantities):
        if pd.notna(ticker):
            positions.setdefault(str(ticker).upper().strip(), float(quantity))
    return positions


def _effect(before: Optional[float], signed: float) -> Optional[str]:
    """What the trade does to the ex-ante position (None without ex-ante position data)."""
    if before is None:
        return None
    after = before + signed
    if before == 0.0:
        return "open"
    if after == 0.0:
        return "close"
    if (before > 0) != (after > 0):
        return "flip"
    return "increase" if abs(after) > abs(before) else "reduce"


def _blotter(trade_info, ante_fund, post_fund, tna, warnings) -> List[Dict[str, Any]]:
    activity = trade_info.get("trade_activity") or {}
    groups = []
    for key in ASSET_ORDER:
        lists = activity.get(key) or {}
        if not (lists.get("buys") or lists.get("sells")):
            continue
        ante_known = getattr(ante_fund, "data", None) is not None
        before_positions = _positions(ante_fund, key, "nav_shares")
        after_positions = _positions(post_fund, key, "iiv_shares")
        seen: Counter = Counter()
        trades = []
        for side, entries in (("BUY", lists.get("buys") or []), ("SELL", lists.get("sells") or [])):
            for entry in entries:
                ticker = str(entry.get("ticker", "") or "")
                quantity, value = num(entry, "quantity"), num(entry, "market_value")
                price = entry.get("price")
                price = None if price is None else to_float(price, 0.0) or None
                signed = quantity if side == "BUY" else -quantity
                # Not held ex-ante (with ex-ante holdings loaded) = a position of 0.
                before = before_positions.get(ticker, 0.0) if ante_known else None
                ex_post = after_positions.get(ticker)
                flags = []
                if price is None:
                    flags.append("no_price")
                if before and signed == before and ex_post == before:
                    flags.append("restates_position")
                signature = (side, ticker, quantity, value)
                seen[signature] += 1
                if seen[signature] > 1:
                    flags.append("duplicate")
                trades.append({
                    "ticker": ticker, "side": side, "quantity": quantity, "price": price,
                    "multiplier": to_float(entry.get("multiplier"), 100.0 if key == "options" else 1.0),
                    "value": value, "pct_tna": _pct(value, tna), "before_qty": before,
                    "after_qty": None if before is None else before + signed, "ex_post_qty": ex_post,
                    "effect": _effect(before, signed),
                    "flags": flags, "in_cash": "no_price" not in flags and "restates_position" not in flags,
                })
        trades.sort(key=lambda t: (t["side"], -t["value"], t["ticker"]))
        _blotter_warnings(key, trades, warnings)
        groups.append({"asset_class": key, "label": ASSET_LABELS[key], "unit": ASSET_UNITS[key],
                       "buy_value": sum(t["value"] for t in trades if t["side"] == "BUY"),
                       "sell_value": sum(t["value"] for t in trades if t["side"] == "SELL"),
                       "trades": trades})
    return groups


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _blotter_warnings(key: str, trades: Sequence[Mapping[str, Any]], warnings: List[str]) -> None:
    label = ASSET_LABELS[key].lower()
    for flag, text in (
        ("no_price", "{n} {label} trade(s) have no usable price; the trading report values them at quantity x "
                     "multiplier ({value}) and the cash estimate leaves them out."),
        ("restates_position", "{n} {label} trade(s) only restate the position already held ({value}: the staged "
                              "quantity equals the ex-ante holding, which is unchanged ex-post); likely not a real "
                              "trade, so the cash estimate leaves them out."),
        ("duplicate", "{n} {label} trade row(s) repeat another row exactly (same ticker, quantity and value: "
                      "{value}); the trading report counts them too. Check the holdings for duplicate rows."),
    ):
        flagged = [t for t in trades if flag in t["flags"]]
        if flagged:
            warnings.append(text.format(n=len(flagged), label=label, value=_money(sum(t["value"] for t in flagged))))


def _flex_trade_count(fund: Any) -> int:
    snapshot = getattr(getattr(fund, "data", None), "current", None)
    for source in ("vest", "custodian"):
        df = getattr(getattr(snapshot, source, None), "flex_options", None)
        if isinstance(df, pd.DataFrame) and not df.empty and "trade_rebal" in df.columns:
            return int((pd.to_numeric(df["trade_rebal"], errors="coerce").fillna(0.0) != 0.0).sum())
    return 0


# ---------------------------------------------------------------------------
# Cash
# ---------------------------------------------------------------------------

def _snapshot_has_data(snapshot: Any) -> bool:
    return snapshot is not None and bool(to_float(getattr(snapshot, "cash", 0.0)) or
                                         to_float(getattr(snapshot, "tna", 0.0)))


def _current_cash(ante_fund: Any, post_fund: Any, day_str: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    ante_data = getattr(ante_fund, "data", None)  # None without the ex-ante copy: its T-1 snapshot is unknown
    post_data = getattr(post_fund, "data", None)
    for snapshot, source, when in ((getattr(ante_data, "previous", None), "ex_ante_previous",
                                    _previous_business_day(day_str)),
                                   (getattr(ante_data, "current", None), "ex_ante_current", day_str),
                                   (getattr(post_data, "current", None), "ex_post_current", day_str)):
        if _snapshot_has_data(snapshot):
            return to_float(getattr(snapshot, "cash", 0.0)), source, when
    return None, None, None


def _cash(ante_fund, post_fund, tna, blotter, assets, day_str) -> Dict[str, Any]:
    current, source, when = _current_cash(ante_fund, post_fund, day_str)
    excluded = [{"asset_class": group["asset_class"], "ticker": t["ticker"], "side": t["side"], "value": t["value"],
                 "reason": "no_price" if "no_price" in t["flags"] else "restates_position"}
                for group in blotter for t in group["trades"] if not t["in_cash"]]
    change = sum(a["cash_effect"] for a in assets.values())
    post = None if current is None else current + change
    custodian_t = getattr(getattr(ante_fund, "data", None), "current", None)
    return {
        "current": current, "source": source, "date": when,
        "custodian_t": to_float(getattr(custodian_t, "cash", 0.0)) if _snapshot_has_data(custodian_t) else None,
        "post": post, "change": change, "post_pct_tna": _pct(post, tna),
        "negative": post is not None and post < 0.0, "estimate": True,
        "excluded": excluded, "excluded_value": sum(e["value"] for e in excluded),
    }


def _waterfall(cash: Mapping[str, Any], blotter: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    def in_cash(key: str, side: Optional[str] = None) -> float:
        total = 0.0
        for group in blotter:
            if group["asset_class"] != key:
                continue
            for t in group["trades"]:
                if t["in_cash"] and (side is None or t["side"] == side):
                    total += t["value"] if t["side"] == "SELL" else -t["value"]
        return total

    start = cash["current"]
    steps = [("equity_buys", "Equity buys", in_cash("equity", "BUY")),
             ("equity_sells", "Equity sells", in_cash("equity", "SELL")),
             ("options_net", "Options net premium", in_cash("options")),
             ("treasury_net", "Treasury net", in_cash("treasury"))]
    running = start
    out = [{"key": "current_cash", "label": "Current cash", "value": start, "running": start, "kind": "start"}]
    for key, label, value in steps:
        running = None if running is None else running + value
        out.append({"key": key, "label": label, "value": value, "running": running, "kind": "delta"})
    out.append({"key": "post_cash", "label": "Post-trade cash (est.)", "value": cash["post"], "running": cash["post"],
                "kind": "end"})
    return out


# ---------------------------------------------------------------------------
# KPIs and totals
# ---------------------------------------------------------------------------

def _kpis(traded: Sequence[Mapping[str, Any]], selected: int, no_trades: int, unavailable: int) -> Dict[str, Any]:
    impacts = Counter(f["impact"] for f in traded)
    codes = Counter(cell["transition"] for f in traded for cell in f["transitions"].values())
    kpis: Dict[str, Any] = {
        "funds_selected": selected, "funds_with_trades": len(traded), "funds_without_trades": no_trades,
        "funds_unavailable": unavailable,
        "buy_value": sum(f["buy_value"] for f in traded), "sell_value": sum(f["sell_value"] for f in traded),
        "gross_value": sum(f["gross_value"] for f in traded),
    }
    kpis["net_value"] = kpis["buy_value"] - kpis["sell_value"]
    for key in ASSET_ORDER:
        for side in ("buy", "sell"):
            kpis[f"{key}_{side}_value"] = sum(f["assets"][key][f"{side}_value"] for f in traded)
    kpis.update(
        funds_compliant=impacts["COMPLIANT"], funds_breaking=impacts["BREAKS"], funds_curing=impacts["CURES"],
        funds_still_failing=impacts["STILL_FAILS"], funds_error=impacts["ERROR"], funds_unknown=impacts["UNKNOWN"],
        rule_breaks=codes["BREAKS"], rule_cures=codes["CURES"], rule_still_fails=codes["STILL_FAILS"] + codes["FAILS"],
        funds_negative_cash=sum(bool(f["cash"]["negative"]) for f in traded),
        funds_with_warnings=sum(bool(f["warnings"]) for f in traded),
    )
    return kpis


def _totals(traded: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tna = sum(f["tna"] for f in traded)
    assets = {}
    for key in ASSET_ORDER:
        stats = [f["assets"][key] for f in traded]
        entry = {field: sum(s[field] for s in stats) for field in
                 ("buy_value", "sell_value", "net_value", "gross_value", "buy_quantity", "sell_quantity")}
        entry["gross_pct_tna"] = _pct(entry["gross_value"], tna)
        assets[key] = entry
    with_cash = [f for f in traded if f["cash"]["current"] is not None]
    cash = sum(f["cash"]["current"] for f in with_cash) if with_cash else None
    post = sum(f["cash"]["post"] for f in with_cash) if with_cash else None
    cash_tna = sum(f["tna"] for f in with_cash)
    gross = sum(f["gross_value"] for f in traded)
    buy, sell = sum(f["buy_value"] for f in traded), sum(f["sell_value"] for f in traded)
    return {
        "funds": len(traded), "tna": tna, "assets": assets, "buy_value": buy, "sell_value": sell,
        "net_value": buy - sell, "gross_value": gross, "gross_pct_tna": _pct(gross, tna),
        "cash": cash, "post_cash": post, "cash_change": None if cash is None else post - cash,
        "post_pct_tna": _pct(post, cash_tna), "funds_negative_cash": sum(bool(f["cash"]["negative"]) for f in traded),
        "cash_funds": len(with_cash),
    }
