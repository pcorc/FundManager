"""In-app result views: one day's results as JSON-safe display tables, the twin of the PDF reports.

The labels, numbers and PASS/FAIL text come from the PDF builders' own helpers
(``compliance_reporting.MetricCells`` / ``_PDF_SECTIONS`` / ``_RULE_SHEETS``, the
reconciliation PDF's row functions, the trading PDF's record builders), so a
table in the UI reads like the matching PDF table. Building a view only reads
the results; it never writes files or logs.

:func:`build_day_view` (EOD day) and :func:`build_trading_view` (trading day)
return one ``DayView``; ``processing.worker`` writes them to ``views.json``::

    {"schema_version": 1, "days": {"<YYYY-MM-DD>": DayView, ...}}

    DayView = {
      "kind": "eod" | "trading", "date": "YYYY-MM-DD", "funds": [fund, ...],
      "fund_errors": {fund: [message, ...]},            # funds that failed to process
      "rules": {rule_key: {"title", "label", "footnotes": [str]}},   # rules that appear below
      "compliance": {fund: {rule_key: RuleView}},       # ex-post side on a trading day
      "compliance_summary": {fund: {"status", "counts": {"PASS","FAIL","N/A","ERROR"}, "failed": [rule_key]}},
      "nav": {fund: NavView},                           # EOD only, when NAV recon ran
      "holdings": {fund: HoldingsView},                 # EOD only, when holdings recon ran
      # trading day only:
      "compliance_ex_ante": {fund: {rule_key: RuleView}},
      "trading": TradingView,
    }

    RuleView = {
      "title": PDF section title, "label": checker label,
      "status": "PASS" | "FAIL" | "N/A" | "ERROR" | None,   # None = not a test (summary_metrics) / no result
      "reason": str | None,                               # why N/A (skip reason, no shares outstanding) / ERROR
      "rows": [{"label", "value", "status"?, "reason"?, "informational"?}],   # the PDF metric rows
      "details": [{"label", "value"}],                    # the workbook columns not already in rows
      "tables": [Table],                                  # GICS exposures
      "notes": [str],                                     # fund-specific notes (the footnotes are in "rules")
    }
    rows[].status is the verdict of a status row (PASS/FAIL/N/A, None when blank); "informational" marks
    a FAIL the PDF does not highlight (40 Act Registration Match / Condition 2a OCC).

    NavView = {
      "status": "PASS" | "FAIL" | None,                   # FAIL when either NAV Good flag fails
      "summary": [{"label", "value", "status"?}],         # the NAV Reconciliation Summary row
      "rows": [{"label", "value"}],                       # the fund page's NAV table (+ NAV Difference $ / %)
      "sections": [{"title", "rows": [{"label", "value"}]}],   # Gain/Loss Components, Adjustments
      "checks": [{"label", "status"}],                    # NAV Good 2-Dec / 4-Dec
      "gl": [{"label", "value"}],                         # the Gain/Loss Components by Fund row
      "tables": [Table],                                  # CEF option price sensitivity
    }
    HoldingsView = {
      "status": "PASS" | "FAIL" | "WARN",                 # FAIL: holdings/qty breaks; WARN: price/weight only
      "holdings_breaks": int, "price_breaks": int,
      "counts": [{"label", "value": int, "kind": "holdings" | "price"}],   # the Holdings Summary row
      "sections": [{"key", "title", "tables": [Table]}],  # the fund page's discrepancy tables
    }
    TradingView = {
      "date", "summary": {"total_funds_analyzed", "total_funds_traded"},
      "overview": {"columns": [...], "rows": [{"fund", "status_change", "violations_before", "violations_after",
                   "net_change", "highlight": "into" | "out" | "breach" | None}], "totals": {...}},
      "checks": {fund: [{"key", "label", "before", "after", "changed"}]},   # every check (the PDF shows changed)
      "activity": {"columns": [...], "rows": [{"fund", "asset_class", "cells": [{"label", "value"}]}]},
      "activity_details": {fund: [{"asset_class", "direction", "ticker", "quantity", "market_value"}]},
    }
    Table = {"title", "columns": [str], "rows": [[str]], "total": int}   # total = rows before truncation

:func:`rule_descriptions` describes every rule (what it checks, its thresholds
from ``config.constants`` and its footnotes) without the DB, for a configuration preview.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from config import constants as C
from reporting.combined_reconciliation_report import (
    GL_COLS,
    OPTION_SENSITIVITY_ROWS,
    _format_currency,
    _nav_number,
    gl_component_values,
    has_option_sensitivity,
    holdings_summary_counts,
    nav_detail_metrics,
    nav_detail_sections,
    nav_difference_pct,
    nav_good_flags,
    nav_summary_cells,
)
from reporting.compliance_reporting import (
    _PDF_SECTIONS,
    _RULE_SHEETS,
    _SUMMARY_SHEET_FIELDS,
    _12D3_RULE_LABELS,
    GICS_SUMMARY_COLUMNS,
    MetricCells,
    _gics_flags,
    _is_verdict,
    _is_vit_fund,
    _selected_tests,
    is_percent_header,
)
from reporting.footnotes import FOOTNOTES
from reporting.holdings_recon_renderer import HoldingsReconciliationRenderer, discrepancy_table
from reporting.nav_recon_reporting import convert_nav_results_to_dicts
from reporting.report_utils import non_empty, normalize_compliance_results, normalize_report_date
from reporting.trade_compliance_reporting import (
    COMPLIANCE_CHANGE_HEADERS,
    PDF_TRADE_SUMMARY_COLUMNS,
    _compliance_change_rows,
    _format_money,
    _status_class,
    _trade_activity_records,
)
from services.compliance_checker import GICS_CONCENTRATION_THRESHOLD, RULE_KEYS, RULE_LABELS

__all__ = ["SCHEMA_VERSION", "build_day_view", "build_trading_view", "compliance_views", "holdings_views",
           "nav_views", "rule_descriptions", "to_jsonable", "trading_view"]

SCHEMA_VERSION = 1
GICS = "gics_compliance"
IRC = "diversification_IRC_check"
SUMMARY = "summary_metrics"
STATUSES = ("PASS", "FAIL", "N/A", "ERROR")
GICS_TABLE_LIMIT = 15  # exposures listed per GICS class (largest first; every exceeding one is kept)

_CELLS = MetricCells()
_RENDERER = HoldingsReconciliationRenderer()
# rule key -> (title, footnote key, labels the PDF never highlights as FAIL, metrics), in PDF order.
_SECTIONS = {key: (title, footnote, skip, metrics) for title, key, footnote, skip, metrics in _PDF_SECTIONS}
_SHEETS = {spec.key: spec for spec in _RULE_SHEETS}
_GICS_TITLE = "GICS Diversification"
# The PDF order: Summary Metrics, GICS, then the rule sections.
RULE_ORDER = (SUMMARY, GICS) + tuple(key for key in _SECTIONS if key != SUMMARY)
_STATUS_ROW_LABELS = ("Overall Status", "Status")
# Workbook columns that repeat a PDF row under another name.
_DUPLICATE_DETAILS = {"sum_large_securities_weights", "Illiquid Percentage", "Options in Scope for 80%?",
                      "Fund Registration", "Fund Status Today"}


# ---------------------------------------------------------------------------
# Day views
# ---------------------------------------------------------------------------

def build_day_view(
    results: Any,
    *,
    day: date | datetime | str,
    operations: Iterable[str] = ("compliance", "reconciliation", "nav_reconciliation"),
    compliance_tests: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """The EOD ``DayView`` of one day's ``ProcessingResults`` (see the module docstring)."""
    day_str = normalize_report_date(day)
    operations = set(operations)
    fund_results = _fund_results(results)
    view: Dict[str, Any] = {"kind": "eod", "date": day_str, "funds": sorted(fund_results),
                            "fund_errors": _fund_errors(fund_results)}
    if "compliance" in operations:
        compliance = compliance_views(_payload(fund_results, "compliance_results"), day_str, compliance_tests)
        view.update(compliance=compliance, compliance_summary=_compliance_summary(compliance))
    if "nav_reconciliation" in operations:
        view["nav"] = nav_views(_payload(fund_results, "nav_results"))
    if "reconciliation" in operations:
        view["holdings"] = holdings_views(_payload(fund_results, "reconciliation_results"))
    view["rules"] = _used_rules(view.get("compliance", {}))
    return to_jsonable(view)


def build_trading_view(
    results_ex_ante: Any,
    results_ex_post: Any,
    comparison_data: Optional[Mapping[str, Any]],
    *,
    day: date | datetime | str,
    compliance_tests: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """The trading ``DayView``: ex-post (and ex-ante) rules plus the ex-ante vs ex-post comparison."""
    day_str = normalize_report_date(day)
    ante, post = _fund_results(results_ex_ante), _fund_results(results_ex_post)
    compliance = compliance_views(_payload(post, "compliance_results"), day_str, compliance_tests)
    compliance_ex_ante = compliance_views(_payload(ante, "compliance_results"), day_str, compliance_tests)
    errors = _fund_errors(ante, "ex-ante: ")
    for fund, messages in _fund_errors(post, "ex-post: ").items():
        errors.setdefault(fund, []).extend(messages)
    view = {
        "kind": "trading", "date": day_str, "funds": sorted(set(ante) | set(post)), "fund_errors": errors,
        "compliance": compliance, "compliance_summary": _compliance_summary(compliance),
        "compliance_ex_ante": compliance_ex_ante,
        "trading": trading_view(comparison_data or {}),
        "rules": _used_rules({**compliance_ex_ante, **compliance}),
    }
    return to_jsonable(view)


def _fund_results(results: Any) -> Mapping[str, Any]:
    return getattr(results, "fund_results", None) or {}


def _payload(fund_results: Mapping[str, Any], attribute: str) -> Dict[str, Any]:
    return {fund: value for fund, result in fund_results.items() if (value := getattr(result, attribute, None))}


def _fund_errors(fund_results: Mapping[str, Any], prefix: str = "") -> Dict[str, List[str]]:
    return {fund: [f"{prefix}{error}" for error in result.errors]
            for fund, result in sorted(fund_results.items()) if getattr(result, "errors", None)}


def _used_rules(compliance: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    used = {key for rules in compliance.values() for key in rules}
    return {key: {"title": _rule_title(key), "label": RULE_LABELS.get(key, key),
                  "footnotes": _footnotes(key)}
            for key in RULE_ORDER if key in used}


def _rule_title(key: str) -> str:
    return _GICS_TITLE if key == GICS else _SECTIONS[key][0]


def _footnote_key(key: str) -> Optional[str]:
    return "gics" if key == GICS else _SECTIONS[key][1]


def _footnotes(key: str) -> List[str]:
    """The rule's report footnotes (the PDF's blank separator lines dropped)."""
    return [note for note in FOOTNOTES.get(_footnote_key(key) or "", []) if note]


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------

def compliance_views(
    compliance_results: Mapping[str, Mapping[str, Any]],
    day: date | datetime | str,
    compliance_tests: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{fund: {rule_key: RuleView}}`` for ``{fund: fund_result.compliance_results}`` (checker objects or dicts).

    Rules follow the PDF: the requested ones (all when ``compliance_tests`` is empty; summary
    metrics always), and IRC only when the day has a VIT fund (other funds then read N/A).
    """
    normalized = normalize_compliance_results(compliance_results or {})
    report_date = datetime.fromisoformat(normalize_report_date(day)).date()
    selected = _selected_tests(compliance_tests) or None
    has_vit = any(IRC in data or _is_vit_fund(fund) for fund, data in normalized.items())
    keys = [key for key in RULE_ORDER
            if (selected is None or key in selected or key == SUMMARY) and (key != IRC or has_vit)]
    views: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for fund in sorted(normalized):
        fund_data = normalized[fund]
        rules = {}
        for key in keys:
            # A rule the checker did not run for this fund is left out (IRC reads N/A for non-VIT funds).
            if key not in fund_data and key != IRC:
                continue
            rules[key] = _gics_view(fund, fund_data) if key == GICS else _rule_view(key, fund, fund_data, report_date)
        views[fund] = rules
    return views


def _rule_view(key: str, fund: str, fund_data: Mapping[str, Any], report_date: date) -> Dict[str, Any]:
    title, _, skip_labels, metrics = _SECTIONS[key]
    payload = _CELLS._get_test_payload(fund_data, key)
    skip = {label.lower() for label in skip_labels}
    rows = []
    for label, fmt, source, field in metrics:
        row: Dict[str, Any] = {"label": label, "value": _CELLS._cell_text(fund_data, key, fmt, source, field)}
        if fmt == "status" or source == "status_or_skip":
            verdict = row["value"].upper()
            row["status"] = verdict if verdict in ("PASS", "FAIL", "N/A") else None
            reason = _raw_na(payload, source, field)
            if reason:
                row["reason"] = reason
            if verdict == "FAIL" and label.lower() in skip:
                row["informational"] = True
        rows.append(row)

    status, reason = _rule_status(key, fund_data, payload, rows)
    view = {"title": title, "label": RULE_LABELS.get(key, key), "status": status, "reason": reason,
            "rows": rows, "details": [], "tables": [], "notes": []}
    if key in fund_data and status != "ERROR" and not _CELLS._is_test_skipped(fund_data, key):
        view["details"], view["notes"] = _detail_rows(key, fund, fund_data, payload, report_date, rows)
    missing = (payload.get("calculations") or {}).get("missing_shares_out_tickers") or []
    if missing:
        view["notes"].append("No shares outstanding data (left out of the ownership test): " + ", ".join(
            str(ticker) for ticker in missing))
    return view


def _raw_na(payload: Mapping[str, Any], source: str, field: Any) -> Optional[str]:
    """The checker's own N/A text behind a status cell (e.g. "N/A (no shares outstanding data)")."""
    if source == "detail":
        value = _CELLS._get_detail(payload, field)
    elif source == "first_detail":
        value = _CELLS._first_not_none(*(_CELLS._get_detail(payload, key) for key in field))
    elif source == "detail_or_na":
        value = _CELLS._get_detail(payload, field)
        if value is None and payload:
            value = "N/A (not evaluated)"
    else:
        return None
    return value if isinstance(value, str) and value.upper().startswith("N/A") and value.upper() != "N/A" else None


def _rule_status(key: str, fund_data: Mapping[str, Any], payload: Mapping[str, Any],
                 rows: Sequence[Mapping[str, Any]]) -> tuple:
    if key == SUMMARY:
        return None, None
    if key not in fund_data:
        if key == IRC:
            return "N/A", "IRC diversification applies to variable insurance trusts (VIT funds) only"
        return None, "No result"
    if _CELLS._is_test_skipped(fund_data, key):
        details = payload.get("details") if isinstance(payload.get("details"), Mapping) else {}
        return "N/A", payload.get("reason") or details.get("reason") or "Skipped"
    if payload.get("status") == "error":
        return "ERROR", payload.get("error") or "The check raised an error"
    row = next((r for r in rows if r["label"] in _STATUS_ROW_LABELS), None)
    status = row.get("status") if row else _CELLS._status_text(payload.get("is_compliant")) or None
    return status, (row or {}).get("reason") if status == "N/A" else None


def _detail_rows(key, fund, fund_data, payload, report_date, rows) -> tuple:
    """The workbook columns for this rule that the PDF rows do not already show, as display text."""
    shown = {row["label"].lower() for row in rows}
    notes: List[str] = []
    if key == SUMMARY:
        combined = {**dict(fund_data.get("fund_current_totals") or {}), **dict(fund_data.get(SUMMARY) or {})}
        columns = [(label, combined.get(field, 0.0)) for label, field in _SUMMARY_SHEET_FIELDS]
    else:
        spec = _SHEETS.get(key)
        if spec is None or (spec.fund_filter is not None and not spec.fund_filter(fund)):
            return [], notes
        calc = payload.get("calculations", {})
        if not spec.raw_calc:
            calc = calc or {}
        columns = list(spec.row(report_date, fund, payload, calc, 2).items())
    details = []
    for header, value in columns:
        if header in ("Date", "Fund") or header in _DUPLICATE_DETAILS or _is_verdict(value):
            continue
        if header == "Notes":
            if value:
                notes.append(str(value))
            continue
        label = _pretty_label(header)
        if label.lower() in shown:
            continue
        details.append({"label": label, "value": _detail_text(header, value)})
    return details, notes


def _pretty_label(header: str) -> str:
    """Workbook headers as labels: snake_case keys become words ("total_assets" -> "Total Assets")."""
    if " " in header or (header[:1].isupper() and "_" not in header):
        return header
    return " ".join(word.capitalize() if word.islower() else word for word in header.split("_") if word)


def _detail_text(header: str, value: Any) -> str:
    """A workbook value as the workbook displays it: 0.00% for percentage columns, else #,##0.00."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,d}"
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return ""
        return f"{float(value):.2%}" if is_percent_header(header) else f"{float(value):,.2f}"
    return str(value)


def _gics_view(fund: str, fund_data: Mapping[str, Any]) -> Dict[str, Any]:
    gics = fund_data.get(GICS)
    view: Dict[str, Any] = {"title": _GICS_TITLE, "label": RULE_LABELS[GICS], "status": None, "reason": None,
                            "rows": [], "details": [], "tables": [], "notes": []}
    if not isinstance(gics, Mapping):
        view["reason"] = "No result"
        return view
    # Same fallback as the GICS_Compliance sheet / PDF: an early checker fail() carries no overall status.
    overall = str(gics.get("overall_gics_compliance", "FAIL")).upper()
    *flags, can_exceed, exceptions = _gics_flags(str(fund).upper(), gics)
    values = [overall, *("Yes" if flag else "No" for flag in flags), can_exceed, exceptions]
    for label, value in zip(GICS_SUMMARY_COLUMNS[2:], values):
        row: Dict[str, Any] = {"label": label, "value": value}
        if label == GICS_SUMMARY_COLUMNS[2]:
            row["status"] = value if value in ("PASS", "FAIL") else None
        view["rows"].append(row)
    if gics.get("status") == "error":
        view.update(status="ERROR", reason=gics.get("error") or "The check raised an error")
    else:
        view.update(status=overall if overall in ("PASS", "FAIL") else None, reason=gics.get("error"))
    if gics.get("compliance_group"):
        view["notes"].append(f"Compliance group: {gics['compliance_group']}")
    if gics.get("industry_mismatches"):
        view["notes"].append("Industries above 25% where the index is not: " + ", ".join(gics["industry_mismatches"]))
    calculations = gics.get("calculations", {}) or {}
    for gics_class, title in (("GICS_INDUSTRY_NAME", "Industry exposure"),
                              ("GICS_INDUSTRY_GROUP_NAME", "Industry group exposure")):
        table = _gics_table(title, calculations.get(gics_class) or {})
        if table:
            view["tables"].append(table)
    return view


def _gics_table(title: str, calc: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    fund_weights = calc.get("fund_weights", {}) or {}
    if not fund_weights:
        return None
    index_weights = calc.get("index_weights", {}) or {}
    exceeding_fund = calc.get("exceeding_fund", {}) or {}
    exceeding_index = calc.get("exceeding_index", {}) or {}
    ordered = sorted(fund_weights.items(), key=lambda item: _float(item[1]), reverse=True)
    kept = [item for i, item in enumerate(ordered) if i < GICS_TABLE_LIMIT or item[0] in exceeding_fund]
    rows = [[str(category), f"{_float(weight):.2%}", f"{_float(index_weights.get(category)):.2%}",
             "Yes" if category in exceeding_fund else "", "Yes" if category in exceeding_index else ""]
            for category, weight in kept]
    return {"title": title, "columns": ["Category", "Fund Weight", "Index Weight", "Fund > 25%", "Index > 25%"],
            "rows": rows, "total": len(ordered)}


def _float(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number


def _compliance_summary(compliance: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    summary = {}
    for fund, rules in compliance.items():
        counts = dict.fromkeys(STATUSES, 0)
        failed = []
        for key, rule in rules.items():
            status = rule.get("status")
            if status in counts:
                counts[status] += 1
            if status == "FAIL":
                failed.append(key)
        status = next((s for s in ("FAIL", "ERROR", "PASS", "N/A") if counts[s]), None)
        summary[fund] = {"status": status, "counts": counts, "failed": failed}
    return summary


# ---------------------------------------------------------------------------
# NAV and holdings reconciliation
# ---------------------------------------------------------------------------

def nav_views(nav_results: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``{fund: NavView}`` for ``{fund: fund_result.nav_results}`` (NAVReconciliationResults or legacy dicts)."""
    views = {}
    for fund, nav in sorted(convert_nav_results_to_dicts(nav_results or {}).items()):
        if not isinstance(nav, Mapping) or not nav:
            continue
        flags = nav_good_flags(nav)
        checks = [{"label": f"NAV Good {label}", "status": "PASS" if flag else "FAIL"} for label, flag in flags]
        nav_diff, nav_diff_pct = nav_difference_pct(nav)
        rows = [{"label": metric, "value": _format_currency(value, digits=digits)}
                for metric, value, digits in nav_detail_metrics(nav)]
        rows.append({"label": "NAV Difference ($ / %)", "value": f"{nav_diff:.4f}   ({nav_diff_pct:.4f}%)"})
        summary = []
        for label, text, ok in nav_summary_cells(nav):
            cell: Dict[str, Any] = {"label": label, "value": text}
            if ok is not None:
                cell["status"] = "PASS" if ok else "FAIL"
            summary.append(cell)
        gl = gl_component_values(nav)
        views[fund] = {
            "status": ("FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS") if checks else None,
            "summary": summary,
            "rows": rows,
            "sections": [{"title": title, "rows": [{"label": metric.strip(), "value": _format_currency(val, digits=2)}
                                                   for metric, val in items]}
                         for title, items in nav_detail_sections(nav)],
            "checks": checks,
            "gl": [{"label": label, "value": f"{gl[key]:,.0f}"} for key, label, *_ in GL_COLS[1:]],
            "tables": [_sensitivity_table(nav)] if has_option_sensitivity(fund, nav) else [],
        }
    return views


def _sensitivity_table(nav: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for label, vest_key, cust_key in OPTION_SENSITIVITY_ROWS:
        vest, cust = _nav_number(nav, vest_key), _nav_number(nav, cust_key)
        rows.append([label, *(_format_currency(amount, digits=0) for amount in (vest, cust, cust - vest))])
    ev, ec = _nav_number(nav, "exp_nav_vest"), _nav_number(nav, "exp_nav_cust")
    rows.append(["Expected NAV", *(f"{amount:,.4f}" for amount in (ev, ec, ec - ev))])
    rows.append(["Custodian NAV", f"{_nav_number(nav, 'cust_nav'):,.4f}", "", ""])
    return {"title": "Option Price Sensitivity (Vest vs Custodian Pricing)",
            "columns": ["", "Vest Prices", "Custodian Prices", "Impact ($)"], "rows": rows, "total": len(rows)}


def holdings_views(reconciliation_results: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``{fund: HoldingsView}`` for ``{fund: fund_result.reconciliation_results}`` ({"summary", "details"})."""
    views = {}
    for fund, payload in sorted((reconciliation_results or {}).items()):
        if not isinstance(payload, Mapping) or not payload:
            continue
        counts = [{"label": label, "value": count,
                   "kind": "price" if ("Price" in label or "Weight" in label) else "holdings"}
                  for label, count in holdings_summary_counts(payload.get("summary", {}) or {})]
        holdings = sum(c["value"] for c in counts if c["kind"] == "holdings")
        price = sum(c["value"] for c in counts if c["kind"] == "price")
        views[fund] = {
            "status": "FAIL" if holdings else ("WARN" if price else "PASS"),
            "holdings_breaks": holdings,
            "price_breaks": price,
            "counts": counts,
            "sections": _holdings_sections(fund, payload.get("details", {}) or {}),
        }
    return views


def _holdings_sections(fund: str, details: Mapping[str, Any]) -> List[Dict[str, Any]]:
    sections = []
    for recon_type, data in sorted(details.items()):
        if not isinstance(data, Mapping) or not data:
            continue
        tables = []
        final = data.get("final_recon")
        custodian = recon_type in ("custodian_equity", "custodian_option", "custodian_treasury")
        title = "Holdings Discrepancies" if custodian else f"Final Reconciliation - {recon_type}"
        table = _table(title, final)
        if table:
            tables.append(table)
        price_t = data.get("price_discrepancies_T")
        if custodian and non_empty(price_t):
            table = _table("Custodian Price Breaks T", _RENDERER._filter_option_price_breaks(price_t, fund, recon_type))
            if table:
                tables.append(table)
        if tables:
            sections.append({"key": recon_type, "title": recon_type.replace("_", " ").title(), "tables": tables})
    return sections


def _table(title: str, df: Any) -> Optional[Dict[str, Any]]:
    table = discrepancy_table(df, title)
    if table is None:
        return None
    columns, rows, _ = table
    return {"title": title, "columns": columns, "rows": rows, "total": len(df)}


# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------

def trading_view(comparison_data: Mapping[str, Any]) -> Dict[str, Any]:
    """The trading PDF's overview, check comparison and trade activity from the analyzer output."""
    summary = comparison_data.get("summary", {}) or {}
    rows, totals = _compliance_change_rows(comparison_data)
    overview = {
        "columns": list(COMPLIANCE_CHANGE_HEADERS),
        "rows": [{"fund": fund, "status_change": str(change), "violations_before": before,
                  "violations_after": after, "net_change": net,
                  "highlight": _status_class(str(change or "UNCHANGED").upper(), after)}
                 for fund, change, before, after, net in rows],
        "totals": dict(zip(("violations_before", "violations_after", "net_change"), totals)),
    }
    checks = {}
    for fund, fund_data in sorted((comparison_data.get("funds", {}) or {}).items()):
        checks[fund] = [{"key": name, "label": RULE_LABELS.get(name, name),
                         "before": str(check.get("status_before", "")), "after": str(check.get("status_after", "")),
                         "changed": bool(check.get("changed"))}
                        for name, check in sorted(((fund_data or {}).get("checks", {}) or {}).items())
                        if isinstance(check, Mapping)]
    records = _trade_activity_records(comparison_data)
    activity_rows = [
        {"fund": record["fund"], "asset_class": record["label"],
         "cells": [{"label": header, "value": formatter(record[key])}
                   for header, _, key, formatter in PDF_TRADE_SUMMARY_COLUMNS[2:]]}
        for record in sorted(records, key=lambda r: (r["fund"], r["label"])) if record["show_summary"]
    ]
    details: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        if record["show_summary"]:
            for direction, ticker, quantity, value in record["details"]:
                details.setdefault(record["fund"], []).append(
                    {"asset_class": record["label"], "direction": direction, "ticker": ticker,
                     "quantity": _format_money(quantity), "market_value": _format_money(value)})
    return {
        "date": str(comparison_data.get("date", "")),
        "summary": {"total_funds_analyzed": int(summary.get("total_funds_analyzed", 0) or 0),
                    "total_funds_traded": int(summary.get("total_funds_traded", 0) or 0)},
        "overview": overview,
        "checks": checks,
        "activity": {"columns": [header for header, *_ in PDF_TRADE_SUMMARY_COLUMNS], "rows": activity_rows},
        "activity_details": details,
    }


# ---------------------------------------------------------------------------
# Static rule descriptions (no DB)
# ---------------------------------------------------------------------------

def _pct(value: float) -> str:
    return f"{value:.0%}"


def rule_descriptions() -> Dict[str, Dict[str, Any]]:
    """Every rule key with its title, what it checks, its conditions (thresholds from config.constants),
    the funds it applies to and its report footnotes, in report order."""
    t12d3 = _12D3_RULE_LABELS
    specs = {
        SUMMARY: (
            "The fund totals every rule uses: cash, treasury, equity, option delta-adjusted notional (DAN), "
            "option market value, total assets and total net assets. Not a pass/fail test; always calculated.",
            [], "All funds"),
        GICS: (
            "Concentration: the fund's weight in each GICS industry (equity plus long options, as a share of "
            "total assets) against the 25% limit, compared with the fund's index.",
            [f"No GICS industry above {_pct(GICS_CONCENTRATION_THRESHOLD)} of total assets, unless the fund's "
             "index also exceeds it in that industry (KNG, KNGIX, FDND and the default rule)",
             "DOGG: no industry may exceed 25%",
             "TDVI: only Information Technology industries may exceed 25%",
             "Industry groups above 25% are reported but not judged"],
            "All funds"),
        "prospectus_80pct_policy": (
            "Names rule / 80% policy: (equity + T-bills [+ options in scope]) as a share of "
            "(equity + cash + T-bills [+ options in scope]).",
            [f"Result (DAN) >= {_pct(C.PROSPECTUS_MIN_THRESHOLD)}",
             "Options count at delta-adjusted notional for "
             f"{', '.join(sorted(C.PROSPECTUS_OPTIONS_FUNDS))}; closed-end funds count option market value",
             "Result (MV) repeats the ratio with option market value"],
            "All funds except private funds and the P3 / TR closed-end funds (skipped)"),
        "diversification_40act_check": (
            "1940 Act diversification of the fund's total assets, and whether the fund operates as registered.",
            [f"Condition 1: at least {_pct(C.ACT_40_QUALIFYING_ASSETS_MIN)} of total assets in diversified assets",
             f"Condition 2a: no single issuer in the 75% bucket above {_pct(C.ACT_40_ISSUER_LIMIT)} of total assets",
             f"Condition 2b: no holding in the 75% bucket above {_pct(C.ACT_40_OWNERSHIP_LIMIT)} of the issuer's "
             "voting securities (N/A when shares outstanding is missing)",
             "Registration Match: FAIL only when a fund registered as diversified is not operating as diversified"],
            "All funds except private funds (skipped)"),
        "diversification_IRS_check": (
            "IRS (Subchapter M, section 851) diversification for regulated investment companies.",
            ["Condition 1: 90% of income from qualifying sources",
             f"Condition 2a 50%: qualifying assets at least {_pct(C.IRS_QUALIFYING_ASSETS_MIN)} of net assets",
             f"Condition 2a 5%: in the bottom 50% of the portfolio no issuer above {_pct(C.IRS_BOTTOM_50_LIMIT)} "
             "of fund assets",
             f"Condition 2a 10%: in the bottom 50% no holding above {_pct(C.IRS_OWNERSHIP_LIMIT)} of the issuer's "
             "outstanding shares",
             "Condition 2b (no issuer above 25% of assets) is not evaluated"],
            "All funds except private funds (skipped)"),
        IRC: (
            "IRC section 817(h) diversification of a segregated asset account: cumulative exposure of the "
            "largest one to four investments.",
            [f"Top 1 <= {_pct(C.IRC_TOP_1_LIMIT)}", f"Top 2 <= {_pct(C.IRC_TOP_2_LIMIT)}",
             f"Top 3 <= {_pct(C.IRC_TOP_3_LIMIT)}", f"Top 4 <= {_pct(C.IRC_TOP_4_LIMIT)} of total assets"],
            "Variable insurance trusts (VIT funds) only"),
        "max_15pct_illiquid_sai": (
            "Illiquid or restricted securities (Bloomberg tags) as a share of total assets.",
            [f"Illiquid exposure <= {_pct(C.ILLIQUID_MAX_THRESHOLD)} of total assets",
             f"Equity holdings at least {_pct(C.EQUITY_MIN_THRESHOLD)} of total assets is reported"],
            "All funds (the workbook lists ETFs only)"),
        "real_estate_check": (
            "Holdings in the GICS Real Estate sector (Bloomberg tags).",
            ["Real estate exposure must be zero"], "All funds"),
        "commodities_check": (
            "Holdings tagged as commodities (Bloomberg tags).",
            ["Commodities exposure must be zero"], "All funds"),
        "twelve_d1a_other_inv_cos": (
            "Rule 12d1-1: holdings of other investment companies.",
            [f"Test 1: own at most {_pct(C.RULE_12D1A_OWNERSHIP_LIMIT)} of an acquired company's voting stock "
             "(N/A when shares outstanding is missing)",
             f"Test 2: largest holding at most {_pct(C.RULE_12D1A_SINGLE_ASSETS_LIMIT)} of total assets",
             f"Test 3: all investment companies at most {_pct(C.RULE_12D1A_TOTAL_ASSETS_LIMIT)} of total assets"],
            "All funds (skipped without equity holdings or total assets)"),
        "twelve_d2_insurance_cos": (
            "Rule 12d2: ownership of insurance companies (GICS Insurance).",
            [f"Own at most {_pct(C.RULE_12D2_INSURANCE_LIMIT)} of an insurance company's outstanding shares "
             "(N/A when shares outstanding is missing)"],
            "All funds"),
        "twelve_d3_sec_biz": (
            "Rule 12d3-1: securities-related businesses (GICS Capital Markets and Banks).",
            [f"{t12d3[0]}: own at most {_pct(C.RULE_12D3_EQUITY_LIMIT)} of the issuer's equity",
             f"{t12d3[1]}: debt at most {_pct(C.RULE_12D3_DEBT_LIMIT)} of total assets",
             f"{t12d3[2]}: at most {_pct(C.RULE_12D3_ASSET_LIMIT)} of total assets in one issuer"],
            "All funds"),
    }
    return {
        key: {"key": key, "title": _rule_title(key), "label": RULE_LABELS.get(key, key), "summary": summary,
              "conditions": conditions, "applies_to": applies_to,
              "footnotes": _footnotes(key)}
        for key in RULE_KEYS
        for summary, conditions, applies_to in [specs[key]]
    }


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def to_jsonable(value: Any) -> Any:
    """Plain JSON types: numpy scalars unwrapped, NaN/inf -> None, dates -> ISO text, tuples -> lists."""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, str):
        return value
    return str(value)
