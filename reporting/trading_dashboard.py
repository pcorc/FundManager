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
  values them at quantity x multiplier), staged quantities that restate the position already held
  (``restates_position``: trade = the ex-ante position and the ex-post position is the same, as the
  DOGG treasury bill carries every day) and rows that repeat another trade row exactly (``duplicate``:
  the DB views return some holdings rows twice; the ex-post shares confirm one trade) are left out and
  listed in ``cash.excluded``. Settlement timing, commissions, accruals, FLEX option trades and other
  cash flows are not included. The same "priced" trades (``in_cash``) give the ``priced`` figures the
  KPI tiles and charts show; the per-class ``buy_value`` / ``sell_value`` stay the report's.
* The compliance checks themselves use the trade-date (T) custodian cash and TNA (``cash.custodian_t``
  and the Summary Metrics rows), unchanged on the ex-post side: they are not trade-adjusted.
* Trend of a rule that fails before and after (``Cell.trend``): ``worse`` / ``better`` when the number
  of failing sub-checks rises / falls, else when the rule's key figure (:data:`KEY_METRICS`) moves away
  from / towards compliance; ``moves`` when other figures change; ``same`` when nothing changes.

Schema::

    Dashboard = {
      "schema": 2, "date": "YYYY-MM-DD", "cash_date": "YYYY-MM-DD" (T-1),
      "definitions": {"tna", "current_cash", "custodian_t", "trade_value", "net_value", "post_cash", "pct_tna",
                      "trend": str, "transitions": {code: str}, "impacts": {code: str}, "trends": {code: str}},
      "kpis": {"funds_selected", "funds_with_trades", "funds_without_trades", "funds_unavailable",
               "buy_value", "sell_value", "net_value", "gross_value",       # the report's (every staged row)
               "equity_buy_value", "equity_sell_value", "options_buy_value", "options_sell_value",
               "treasury_buy_value", "treasury_sell_value",
               "priced_buy_value", "priced_sell_value", "priced_net_value", "priced_gross_value",
               "{class}_priced_buy_value", "{class}_priced_sell_value",   # the cash estimate's trades
               "flagged_value",                                           # report gross - priced gross
               "funds_compliant", "funds_breaking", "funds_curing", "funds_still_failing", "funds_worsening",
               "funds_error", "funds_unknown", "rule_breaks", "rule_cures", "rule_still_fails", "rule_worsens",
               "funds_negative_cash", "funds_negative_before", "funds_going_negative",
               "funds_with_warnings", "funds_data_missing",
               "data_notes": {"duplicate" | "restates_position" | "no_price": {"count", "value", "funds"},
                              "unstaged": {"value", "funds"}, "data_missing": [fund], "flex": [fund]}},
                                           # all over the funds with proposed trades
      "funds": [FundDashboard],            # funds with proposed trades; BREAKS, STILL_FAILS (worsening
                                           # first), ERROR, CURES, COMPLIANT, UNKNOWN, then by name
      "no_trade_funds": [{"fund", "impact", "status_after", "failed_after": [rule], "report_trade_value"}],
      "unavailable_funds": [{"fund", "errors": [str]}],   # selected but no result on either side
      "matrix": {"rules": [{"key", "label", "title", "changes_or_fails": bool}],
                 "rows": [{"fund", "impact", "cells": {rule: Cell}}]},   # rows in "funds" order
      "rules_at_risk": [{"rule", "label", "title", "breaks", "still_fails", "fails", "errors", "cures": [fund],
                         "worsens": [fund], "trends": {fund: trend}}],   # still_fails: worse first
      "totals": Totals,                    # sums over "funds" (the summary table's totals row)
    }
    Cell = {"before": "PASS"|"FAIL"|"N/A"|"ERROR"|None, "after": ..., "transition": Transition,
            "changed"?: int,                 # figures (rows, details, table rows) that differ
            "key"?: {"label", "before", "after"},     # the rule's key figure, when it has one
            "trend"?: "worse" | "better" | "moves" | "same"}   # STILL_FAILS only
    Transition = "OK" (PASS->PASS) | "BREAKS" (PASS->FAIL) | "CURES" (FAIL->PASS) | "STILL_FAILS" (FAIL->FAIL)
               | "FAILS" (FAIL after, no ex-ante verdict) | "ERROR" (either side errored) | "N/A"
               # statuses are the rule cards' (RuleView.status): a skipped rule is N/A here, where the
               # trading report's Compliance Details sheet counts it as PASS
    FundDashboard = {
      "fund", "has_trades": True, "trade_count": int, "impact": Impact, "worsens": bool, "status_change",
      "violations_before", "violations_after", "tna", "total_assets",
      "assets": {"equity" | "options" | "treasury": AssetStats},   # always all three
      "buy_value", "sell_value", "net_value", "gross_value", "gross_pct_tna",
      "priced": {"buy_value", "sell_value", "net_value", "gross_value", "gross_pct_tna"},   # in_cash trades
      "cash": Cash, "waterfall": [Step],
      "transitions": {rule: Cell},
      "rules": [RuleDiff],               # failing / breaking first; "unchanged" ones can be collapsed
      "metrics": [Diff],                 # Summary Metrics rows (cash, equity, options, TA, TNA) before -> after:
                                         # the compliance checks' own trade-date custodian figures
      "blotter": [{"asset_class", "label", "unit", "buy_value", "sell_value", "trades": [Trade]}],  # classes with trades
      "unstaged": [{"asset_class", "label", "unit", "value", "changes": [Change]}],  # report-only classes
      "data_missing": ["tna" | "cash"],  # no custodian TNA / cash: figures and results rest on missing data
      "warnings": [str],
    }
    Change = {"ticker", "before_qty" (ex-post nav_shares), "after_qty" (iiv_shares), "change", "price",
              "multiplier", "value", "rows"}   # an ex-post position change without a staged trade row;
                                               # rows > 1: the holdings repeat it (shown once)
    Impact = "BREAKS" | "STILL_FAILS" | "ERROR" | "CURES" | "COMPLIANT" | "UNKNOWN"
    AssetStats = {"label", "unit", "staged": bool, "buy_value", "sell_value", "net_value", "gross_value",
                  "buy_quantity", "sell_quantity", "buy_count", "sell_count", "buy_pct_tna", "sell_pct_tna",
                  "net_pct_tna", "gross_pct_tna", "cash_effect", "ex_ante_market_value", "ex_post_market_value",
                  "priced_buy_value", "priced_sell_value", "unpriced_value", "unpriced_buy_value",
                  "unpriced_sell_value"}  # unpriced = the report's quantity x multiplier, not dollars
                  # gross_value / gross_pct_tna = the PDF's Trade Value / % of TNA (without staged rows the
                  # report derives it from the ex-post share columns: staged False, gross_value > 0);
                  # cash_effect = the class's step in the cash estimate (sells - buys of the trades in it)
    Cash = {"current", "source": "ex_ante_previous" | "ex_ante_current" | "ex_post_current" | None,
            "date", "custodian_t" (T custodian cash: the figure the compliance checks use), "custodian_t_date",
            "post", "change", "post_pct_tna", "current_pct_tna",
            "negative": bool (post < 0), "negative_before": bool (current < 0),
            "goes_negative": bool (current >= 0 > post: the trades take cash below zero),
            "estimate": True, "excluded": [{"asset_class", "ticker", "side", "value", "reason"}],
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
                "unchanged": bool, "rows": [Diff], "details": [Diff], "tables": [TableDiff],
                "key"?: {"label", "before", "after"}, "trend"?}
    TableDiff = {"title", "columns", "total": int (rows compared), "rows": [{"key" (first cell),
                 "before": [cell] | None, "after": [cell] | None}]}   # only the rows that differ
    Diff = {"label", "before", "after", "changed": bool, "status_before"?, "status_after"?}  # text as shown
    Totals = {"funds", "tna", "assets": {class: {"buy_value", "sell_value", "net_value", "gross_value",
              "buy_quantity", "sell_quantity", "gross_pct_tna", "priced_buy_value", "priced_sell_value",
              "unpriced_value"}}, "buy_value", "sell_value", "net_value", "gross_value", "gross_pct_tna",
              "priced": {...}, "cash", "post_cash", "cash_change", "post_pct_tna", "funds_negative_cash",
              "funds_going_negative", "cash_funds", "tna_funds"}
              # cash sums cover the funds with a cash figure; "tna" the funds with a TNA
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

DASHBOARD_SCHEMA = 2
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
_TREND_RANK = {"worse": 0, "moves": 1, "better": 2, "same": 3}
TRENDS = {
    "worse": "Fails before and after, and the trades make it worse (more failing sub-checks, or the key "
             "figure moves away from the limit)",
    "better": "Fails before and after, but the trades ease the breach",
    "moves": "Fails before and after; its figures change, with no clear direction",
    "same": "Fails before and after; the trades do not change its figures",
}
# The figure that says whether a failing rule gets better or worse: (row label, +1 when a higher value is worse,
# -1 when a lower value is worse). 40 Act has no single figure (its failing condition is per issuer).
KEY_METRICS: Dict[str, Tuple[str, int]] = {
    "diversification_IRS_check": ("Sum Large Securities %", 1),
    "prospectus_80pct_policy": ("Result (DAN)", -1),
    "max_15pct_illiquid_sai": ("Illiquid Exposure %", 1),
    "real_estate_check": ("Real Estate Exposure", 1),
    "commodities_check": ("Commodities Exposure", 1),
    "twelve_d1a_other_inv_cos": ("Ownership % Max", 1),
    "twelve_d2_insurance_cos": ("Max Ownership", 1),
    "twelve_d3_sec_biz": ("Max Ownership %", 1),
}
GICS_KEY = ("gics_compliance", "Largest fund weight", "Fund Weight")  # the top weight over the GICS tables

DEFINITIONS = {
    "tna": "Custodian total net assets at the T-1 close (the trading report's TNA column).",
    "current_cash": "Custodian cash at the T-1 close: the opening balance on the trade date, before the "
                    "staged trades execute (the latest custodian figure available intraday). Custodian cash "
                    "can be carried forward unchanged for days when the custodian has not updated it; compare "
                    "it with the trade-date figure shown beside it.",
    "custodian_t": "Custodian cash on the trade date: the figure the compliance checks (and the Summary Metrics "
                   "card) use, on both sides; it is not adjusted for the trades.",
    "trade_value": "Each staged trade at |quantity| x price x multiplier: options at premium (100 shares per "
                   "contract), treasuries at price per 100 face. As in the trading report.",
    "net_value": "Buys minus sells (positive = net buying).",
    "post_cash": "Estimate: current cash - buys + sells across equity, options and treasuries (buying, "
                 "including buying options to open or close, pays; selling or writing options receives "
                 "the premium). Trades without a usable price, staged quantities that only restate the "
                 "position already held, and rows that repeat another trade row exactly (duplicate holdings "
                 "rows) are left out (listed per fund). Settlement timing, commissions, accruals, FLEX option "
                 "trades and other flows are not included.",
    "pct_tna": "Fractions of TNA (0.05 = 5%).",
    "trend": "For a rule failing before and after: worse when more sub-checks fail after the trades or its key "
             "figure moves away from the limit, better when fewer fail or it moves towards the limit.",
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

    traded.sort(key=lambda f: (IMPACT_ORDER.index(f["impact"]), not f["worsens"], f["fund"]))
    matrix_rules = _matrix_rules(traded, rules)
    return {
        "schema": DASHBOARD_SCHEMA,
        "date": day_str,
        "cash_date": _previous_business_day(day_str),
        "definitions": {**DEFINITIONS, "transitions": dict(TRANSITIONS), "impacts": dict(IMPACTS),
                        "trends": dict(TRENDS)},
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


def _diff_tables(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Tables matched by title, rows by their first cell; only the rows whose other cells differ are kept."""
    before_by_title = {str(t.get("title")): t for t in before or []}
    after_by_title = {str(t.get("title")): t for t in after or []}
    titles = list(after_by_title) + [t for t in before_by_title if t not in after_by_title]
    out = []
    for title in titles:
        old, new = before_by_title.get(title) or {}, after_by_title.get(title) or {}
        old_rows, new_rows = {}, {}
        for rows, target in ((old.get("rows") or [], old_rows), (new.get("rows") or [], new_rows)):
            for row in rows:
                if row:
                    target.setdefault(str(row[0]), list(row[1:]))
        keys = list(new_rows) + [k for k in old_rows if k not in new_rows]
        changed = [{"key": k, "before": old_rows.get(k), "after": new_rows.get(k)} for k in keys
                   if old_rows.get(k) != new_rows.get(k)]
        if changed:
            out.append({"title": title, "columns": list(new.get("columns") or old.get("columns") or []),
                        "total": len(keys), "rows": changed})
    return out


def _number(value: Any) -> Optional[float]:
    """A displayed figure as a number ("12.10%" -> 12.1, "1,234" -> 1234.0); None when it is not one."""
    text = str(value if value is not None else "").strip().replace(",", "").replace("$", "").replace("%", "")
    text = text.replace("−", "-")
    try:
        return float(text) if text else None
    except ValueError:
        return None


def _gics_top(view: Mapping[str, Any]) -> Optional[float]:
    column = GICS_KEY[2]
    best = None
    for table in view.get("tables") or []:
        columns = list(table.get("columns") or [])
        if column not in columns:
            continue
        index = columns.index(column)
        for row in table.get("rows") or []:
            value = _number(row[index]) if index < len(row) else None
            if value is not None and (best is None or value > best):
                best = value
    return best


def _key_figure(key: str, before: Mapping[str, Any], after: Mapping[str, Any]
                ) -> Tuple[Optional[Dict[str, Any]], int]:
    """({"label", "before", "after"} of the rule's key figure | None, its worse direction)."""
    if key == GICS_KEY[0]:
        old, new = _gics_top(before), _gics_top(after)
        if old is None and new is None:
            return None, 0
        shown = (lambda v: None if v is None else f"{v:.2f}%")  # noqa: E731
        return {"label": GICS_KEY[1], "before": shown(old), "after": shown(new), "_b": old, "_a": new}, 1
    label, direction = KEY_METRICS.get(key, (None, 0))
    if label is None:
        return None, 0
    values = []
    for view in (before, after):
        row = next((r for r in (view.get("rows") or []) + (view.get("details") or []) if r.get("label") == label), None)
        values.append(None if row is None else row.get("value"))
    if values == [None, None]:
        return None, 0
    return {"label": label, "before": values[0], "after": values[1], "_b": _number(values[0]),
            "_a": _number(values[1])}, direction


def _failing(rows: Sequence[Mapping[str, Any]], side: str) -> int:
    return sum(1 for row in rows if row.get(side) == "FAIL")


def _trend(rows: Sequence[Mapping[str, Any]], figure: Optional[Mapping[str, Any]], direction: int,
           changed: int) -> str:
    before, after = _failing(rows, "status_before"), _failing(rows, "status_after")
    if after != before:
        return "worse" if after > before else "better"
    if figure and direction and figure.get("_b") is not None and figure.get("_a") is not None \
            and figure["_a"] != figure["_b"]:
        return "worse" if (figure["_a"] - figure["_b"]) * direction > 0 else "better"
    return "moves" if changed else "same"


def _rule_diffs(ante_rules, post_rules, transitions, rules) -> List[Dict[str, Any]]:
    """One RuleDiff per rule; also annotates each transition cell with its change count, key figure and trend."""
    diffs = []
    for key, cell in transitions.items():
        before, after = ante_rules.get(key) or {}, post_rules.get(key) or {}
        rows = _diff_rows(before.get("rows") or [], after.get("rows") or [])
        details = _diff_rows(before.get("details") or [], after.get("details") or [], with_status=False)
        tables = _diff_tables(before.get("tables") or [], after.get("tables") or [])
        changed = sum(row["changed"] for row in rows + details) + sum(len(t["rows"]) for t in tables)
        title, label = _rule_meta(key, rules, after, before)
        figure, direction = _key_figure(key, before, after)
        diff: Dict[str, Any] = {"rule": key, "title": title, "label": label, "before": cell["before"],
                                "after": cell["after"], "transition": cell["transition"], "changed_rows": changed,
                                "unchanged": cell["transition"] in ("OK", "N/A") and not changed,
                                "rows": rows, "details": details, "tables": tables}
        cell["changed"] = changed
        if figure is not None:
            shown = {k: figure[k] for k in ("label", "before", "after")}
            diff["key"] = cell["key"] = shown
        if cell["transition"] == "STILL_FAILS":
            diff["trend"] = cell["trend"] = _trend(rows, figure, direction, changed)
        diffs.append(diff)
    diffs.sort(key=lambda d: (_RULE_RANK[d["transition"]], _TREND_RANK.get(d.get("trend"), 0), d["unchanged"]))
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
        cells = {f["fund"]: f["transitions"].get(key) or {} for f in traded}
        for name, code in groups:
            entry[name] = [fund for fund, cell in cells.items() if cell.get("transition") == code]
        # Still failing: the trades that deepen the breach first.
        entry["still_fails"].sort(key=lambda fund: _TREND_RANK.get(cells[fund].get("trend"), 1))
        entry["trends"] = {fund: cells[fund]["trend"] for fund in entry["still_fails"] if cells[fund].get("trend")}
        entry["worsens"] = [fund for fund, trend in entry["trends"].items() if trend == "worse"]
        if any(entry[name] for name, _ in groups):
            out.append(entry)
    out.sort(key=lambda e: (-len(e["breaks"]), -len(e["worsens"]), -(len(e["still_fails"]) + len(e["fails"])),
                            -len(e["errors"]), -len(e["cures"])))
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
    elif cash["goes_negative"]:
        warnings.append(f"The trades take estimated cash below zero ({_money(cash['current'])} → "
                        f"{_money(cash['post'])}).")
    elif cash["negative"]:
        warnings.append(f"Cash is already below zero at the {cash['date']} close ({_money(cash['current'])}); "
                        f"after the trades it is an estimated {_money(cash['post'])}.")
    if not tna:
        warnings.append("No custodian TNA for the T-1 close: % of TNA figures are not shown, and rule results may "
                        "rest on missing data.")
    unstaged = _unstaged(post_fund, assets)
    for stats in assets.values():
        if not stats["staged"] and stats["gross_value"] > 0.0:
            warnings.append(f"{stats['label']}: the trading report shows a trade value of "
                            f"{_money(stats['gross_value'])} from the ex-post share columns, without staged trades; "
                            "it is not in the cash estimate (see 'Position changes without staged rows').")
    flex = _flex_trade_count(post_fund)
    if flex:
        warnings.append(f"{flex} FLEX option trade row(s) are staged; the trading analysis does not value FLEX "
                        "trades, so they are not in these figures.")

    buy = sum(a["buy_value"] for a in assets.values())
    sell = sum(a["sell_value"] for a in assets.values())
    gross = sum(a["gross_value"] for a in assets.values())
    priced_buy = sum(a["priced_buy_value"] for a in assets.values())
    priced_sell = sum(a["priced_sell_value"] for a in assets.values())
    rule_diffs = _rule_diffs(ante_rules, post_rules, transitions, rules)  # also annotates the transition cells
    return {
        "fund": name,
        "has_trades": True,
        "trade_count": sum(len(group["trades"]) for group in blotter),
        "impact": impact,
        "worsens": any(cell.get("trend") == "worse" for cell in transitions.values()),
        "flex_trades": flex,
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
        "priced": {"buy_value": priced_buy, "sell_value": priced_sell, "net_value": priced_buy - priced_sell,
                   "gross_value": priced_buy + priced_sell, "gross_pct_tna": _pct(priced_buy + priced_sell, tna)},
        "cash": cash,
        "waterfall": _waterfall(cash, blotter),
        "transitions": transitions,
        "rules": rule_diffs,
        "metrics": _diff_rows((ante_rules.get(SUMMARY) or {}).get("rows") or [],
                              (post_rules.get(SUMMARY) or {}).get("rows") or [], with_status=False),
        "blotter": blotter,
        "unstaged": unstaged,
        "data_missing": [what for what, missing in (("tna", not tna), ("cash", cash["current"] is None)) if missing],
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
    priced_buy = sum(t["value"] for t in in_cash if t["side"] == "BUY")
    priced_sell = sum(t["value"] for t in in_cash if t["side"] == "SELL")
    return {
        "priced_buy_value": priced_buy, "priced_sell_value": priced_sell,
        "unpriced_value": sum(t["value"] for t in trades if "no_price" in t["flags"]),
        "unpriced_buy_value": sum(t["value"] for t in trades if "no_price" in t["flags"] and t["side"] == "BUY"),
        "unpriced_sell_value": sum(t["value"] for t in trades if "no_price" in t["flags"] and t["side"] == "SELL"),
        "label": ASSET_LABELS[key], "unit": ASSET_UNITS[key], "staged": bool(trades),
        "buy_value": buy, "sell_value": sell, "net_value": buy - sell, "gross_value": gross,
        "buy_quantity": num(net, "buy_quantity"), "sell_quantity": num(net, "sell_quantity"),
        "buy_count": sum(t["side"] == "BUY" for t in trades), "sell_count": sum(t["side"] == "SELL" for t in trades),
        "buy_pct_tna": _pct(buy, tna), "sell_pct_tna": _pct(sell, tna), "net_pct_tna": _pct(buy - sell, tna),
        "gross_pct_tna": num(summary, "pct_of_tna") if "pct_of_tna" in summary else _pct(gross, tna),
        "cash_effect": priced_sell - priced_buy,
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
                    "flags": flags, "in_cash": not flags,
                })
        trades.sort(key=lambda t: (t["side"], -t["value"], t["ticker"]))
        _blotter_warnings(key, trades, warnings)
        groups.append({"asset_class": key, "label": ASSET_LABELS[key], "unit": ASSET_UNITS[key],
                       "buy_value": sum(t["value"] for t in trades if t["side"] == "BUY"),
                       "sell_value": sum(t["value"] for t in trades if t["side"] == "SELL"),
                       "trades": trades})
    return groups


def _money(value: float) -> str:
    return f"-${-value:,.0f}" if value < 0 else f"${value:,.0f}"


def _blotter_warnings(key: str, trades: Sequence[Mapping[str, Any]], warnings: List[str]) -> None:
    label = ASSET_LABELS[key].lower()
    for flag, text in (
        ("no_price", "{n} {label} trade(s) have no usable price; the trading report values them at quantity x "
                     "multiplier ({value}) and the cash estimate leaves them out."),
        ("restates_position", "{n} {label} trade(s) only restate the position already held ({value}: the staged "
                              "quantity equals the ex-ante holding, which is unchanged ex-post); likely not a real "
                              "trade, so the cash estimate leaves them out."),
        ("duplicate", "{n} {label} trade row(s) repeat another row exactly (same ticker, quantity and value: "
                      "{value}); the trading report counts them too, the cash estimate counts each trade once. "
                      "Check the holdings for duplicate rows."),
    ):
        flagged = [t for t in trades if flag in t["flags"]]
        if flagged:
            warnings.append(text.format(n=len(flagged), label=label, value=_money(sum(t["value"] for t in flagged))))


def _unstaged(post_fund: Any, assets: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per asset class the report values without staged rows: the ex-post holdings' nav_shares -> iiv_shares
    changes (what the report's trade value and the ex-post checks rest on), at ex-post prices."""
    out = []
    for key in ASSET_ORDER:
        stats = assets.get(key) or {}
        if stats.get("staged") or not stats.get("gross_value"):
            continue
        df = _holdings(post_fund, key)
        column = TradingComplianceAnalyzer._detect_ticker_column(df, key) if not df.empty else None
        if column is None:
            continue
        before = TradingComplianceAnalyzer._resolve_quantity_series(df, preferred="nav_shares")
        after = TradingComplianceAnalyzer._resolve_quantity_series(df, preferred="iiv_shares")
        changes: List[Dict[str, Any]] = []
        seen: Dict[Tuple[str, float, float], Dict[str, Any]] = {}
        for index, row in df.iterrows():
            change = float(after.get(index, 0.0)) - float(before.get(index, 0.0))
            if change == 0.0 or pd.isna(row.get(column)):
                continue
            ticker = str(row.get(column)).strip()
            signature = (ticker, float(before.get(index, 0.0)), float(after.get(index, 0.0)))
            if signature in seen:  # a duplicated holdings row (the report counts it again)
                seen[signature]["rows"] += 1
                continue
            price, multiplier = TradingComplianceAnalyzer._trade_price(row, key)
            value = TradingComplianceAnalyzer._estimate_trade_market_value(row, change, key) if price else None
            seen[signature] = {"ticker": ticker, "before_qty": signature[1], "after_qty": signature[2], "change": change,
                               "price": price, "multiplier": multiplier, "value": value, "rows": 1}
            changes.append(seen[signature])
        if changes:
            changes.sort(key=lambda c: (-(c["value"] or 0.0), c["ticker"]))
            out.append({"asset_class": key, "label": ASSET_LABELS[key], "unit": ASSET_UNITS[key],
                        "value": stats["gross_value"], "changes": changes})
    return out


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
                 "reason": t["flags"][0]}
                for group in blotter for t in group["trades"] if not t["in_cash"]]
    change = sum(a["cash_effect"] for a in assets.values())
    post = None if current is None else current + change
    custodian_t = getattr(getattr(ante_fund, "data", None), "current", None)
    negative = post is not None and post < 0.0
    return {
        "current": current, "source": source, "date": when,
        "custodian_t": to_float(getattr(custodian_t, "cash", 0.0)) if _snapshot_has_data(custodian_t) else None,
        "custodian_t_date": day_str,
        "post": post, "change": change, "post_pct_tna": _pct(post, tna), "current_pct_tna": _pct(current, tna),
        "negative": negative, "negative_before": current is not None and current < 0.0,
        "goes_negative": negative and current is not None and current >= 0.0, "estimate": True,
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
    kpis["priced_buy_value"] = sum(f["priced"]["buy_value"] for f in traded)
    kpis["priced_sell_value"] = sum(f["priced"]["sell_value"] for f in traded)
    kpis["priced_net_value"] = kpis["priced_buy_value"] - kpis["priced_sell_value"]
    kpis["priced_gross_value"] = kpis["priced_buy_value"] + kpis["priced_sell_value"]
    kpis["flagged_value"] = kpis["gross_value"] - kpis["priced_gross_value"]
    for key in ASSET_ORDER:
        for side in ("buy", "sell"):
            kpis[f"{key}_{side}_value"] = sum(f["assets"][key][f"{side}_value"] for f in traded)
            kpis[f"{key}_priced_{side}_value"] = sum(f["assets"][key][f"priced_{side}_value"] for f in traded)
    worse = [cell for f in traded for cell in f["transitions"].values() if cell.get("trend") == "worse"]
    kpis.update(
        funds_compliant=impacts["COMPLIANT"], funds_breaking=impacts["BREAKS"], funds_curing=impacts["CURES"],
        funds_still_failing=impacts["STILL_FAILS"], funds_error=impacts["ERROR"], funds_unknown=impacts["UNKNOWN"],
        funds_worsening=sum(1 for f in traded if f["impact"] == "STILL_FAILS" and f["worsens"]),
        rule_breaks=codes["BREAKS"], rule_cures=codes["CURES"], rule_still_fails=codes["STILL_FAILS"] + codes["FAILS"],
        rule_worsens=len(worse),
        funds_negative_cash=sum(bool(f["cash"]["negative"]) for f in traded),
        funds_negative_before=sum(bool(f["cash"].get("negative_before")) for f in traded),
        funds_going_negative=sum(bool(f["cash"].get("goes_negative")) for f in traded),
        funds_with_warnings=sum(bool(f["warnings"]) for f in traded),
        funds_data_missing=sum(bool(f["data_missing"]) for f in traded),
        data_notes=_data_notes(traded),
    )
    return kpis


def _data_notes(traded: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """What the page-level data-quality note sums up: flagged trades by flag, report-only values, missing data."""
    notes: Dict[str, Any] = {flag: {"count": 0, "value": 0.0, "funds": []}
                             for flag in ("duplicate", "restates_position", "no_price")}
    for fund in traded:
        for entry in fund["cash"]["excluded"]:
            note = notes[entry["reason"]]
            note["count"] += 1
            note["value"] += entry["value"]
            if fund["fund"] not in note["funds"]:
                note["funds"].append(fund["fund"])
    unstaged = [f for f in traded if f["unstaged"]]
    notes["unstaged"] = {"value": sum(g["value"] for f in unstaged for g in f["unstaged"]),
                         "funds": [f["fund"] for f in unstaged]}
    notes["data_missing"] = [f["fund"] for f in traded if f["data_missing"]]
    notes["flex"] = [f["fund"] for f in traded if f.get("flex_trades")]
    return notes


def _totals(traded: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    with_tna = [f for f in traded if f["tna"]]
    tna = sum(f["tna"] for f in with_tna)
    assets = {}
    for key in ASSET_ORDER:
        stats = [f["assets"][key] for f in traded]
        entry = {field: sum(s.get(field) or 0.0 for s in stats) for field in
                 ("buy_value", "sell_value", "net_value", "gross_value", "buy_quantity", "sell_quantity",
                  "priced_buy_value", "priced_sell_value", "unpriced_value", "unpriced_buy_value",
                  "unpriced_sell_value")}
        entry["unstaged_value"] = sum(s["gross_value"] for s in stats if not s["staged"])
        entry["gross_pct_tna"] = _pct(sum(f["assets"][key]["gross_value"] for f in with_tna), tna)
        assets[key] = entry
    with_cash = [f for f in traded if f["cash"]["current"] is not None]
    cash = sum(f["cash"]["current"] for f in with_cash) if with_cash else None
    post = sum(f["cash"]["post"] for f in with_cash) if with_cash else None
    cash_tna = sum(f["tna"] for f in with_cash if f["tna"])
    gross = sum(f["gross_value"] for f in traded)
    buy, sell = sum(f["buy_value"] for f in traded), sum(f["sell_value"] for f in traded)
    priced_buy, priced_sell = sum(f["priced"]["buy_value"] for f in traded), sum(f["priced"]["sell_value"] for f in traded)
    return {
        "funds": len(traded), "tna": tna, "tna_funds": len(with_tna), "assets": assets, "buy_value": buy,
        "sell_value": sell, "net_value": buy - sell, "gross_value": gross,
        "gross_pct_tna": _pct(sum(f["gross_value"] for f in with_tna), tna),
        "priced": {"buy_value": priced_buy, "sell_value": priced_sell, "net_value": priced_buy - priced_sell,
                   "gross_value": priced_buy + priced_sell,
                   "gross_pct_tna": _pct(sum(f["priced"]["gross_value"] for f in with_tna), tna)},
        "cash": cash, "post_cash": post, "cash_change": None if cash is None else post - cash,
        "post_pct_tna": _pct(post, cash_tna) if all(f["tna"] for f in with_cash) else None,
        "funds_negative_cash": sum(bool(f["cash"]["negative"]) for f in traded),
        "funds_going_negative": sum(bool(f["cash"].get("goes_negative")) for f in traded),
        "cash_funds": len(with_cash),
    }
