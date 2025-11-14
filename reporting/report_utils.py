"""Utility helpers shared by reporting components."""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Any, Dict, Tuple

import pandas as pd

try:  # pragma: no cover - optional import for typing only
    from services.compliance_checker import ComplianceResult
except Exception:  # pragma: no cover - safeguard during cold starts
    ComplianceResult = Any  # type: ignore


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def normalize_report_date(value: date | datetime | str) -> str:
    """Normalise user supplied dates to the ``YYYY-MM-DD`` format."""

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def ensure_dataframe(value: Any) -> pd.DataFrame:
    """Coerce arbitrary objects into a :class:`pandas.DataFrame`."""

    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, Mapping):
        # If mapping already looks like {column: sequence}
        if value and all(isinstance(v, Sequence) and not isinstance(v, (str, bytes)) for v in value.values()):
            return pd.DataFrame(value)
        return pd.DataFrame([value])
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return pd.DataFrame(list(value))
    return pd.DataFrame({"value": [value]})


def format_number(value: Any, digits: int = 0) -> str:
    """Return a string representation with thousand separators."""

    if value is None:
        return ""
    if isinstance(value, numbers.Number):
        format_spec = f"{{:,.{digits}f}}"
        return format_spec.format(value)
    return str(value)


# ---------------------------------------------------------------------------
# Compliance helpers
# ---------------------------------------------------------------------------

def normalize_compliance_results(raw_results: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize the mapping returned by :class:`ComplianceChecker`."""

    normalized: Dict[str, Any] = {}
    for fund_name, tests in raw_results.items():
        fund_normalized: Dict[str, Any] = {}
        for test_name, result in (tests or {}).items():
            if _is_compliance_result(result):
                fund_normalized[test_name] = _expand_compliance_result(test_name, result)
            elif isinstance(result, Mapping):
                fund_normalized[test_name] = dict(result)
            else:
                fund_normalized[test_name] = {"value": result}
        normalized[fund_name] = fund_normalized
    return normalized


def flatten_compliance_results(results_by_date: Mapping[str, Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Flatten a ``{date: {fund: data}}`` mapping for PDF generation."""

    flattened: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for date_str, fund_results in results_by_date.items():
        for fund_name, data in fund_results.items():
            flattened[(fund_name, date_str)] = dict(data)
    return flattened


def summarise_compliance_status(normalized_results: Mapping[str, Any]) -> Dict[str, int]:
    """Return aggregate PASS/FAIL counts from normalized compliance data."""

    summary = {
        "funds": 0,
        "funds_in_breach": 0,
        "total_checks": 0,
        "failed_checks": 0,
    }

    for fund_results in normalized_results.values():
        summary["funds"] += 1
        fund_failed = False
        for check_name, payload in fund_results.items():
            if check_name in {"summary_metrics", "fund_current_totals"}:
                continue
            summary["total_checks"] += 1
            if isinstance(payload, Mapping):
                status = payload.get("is_compliant")
                if status is False:
                    summary["failed_checks"] += 1
                    fund_failed = True
                elif isinstance(status, str) and status.upper() == "FAIL":
                    summary["failed_checks"] += 1
                    fund_failed = True
            elif isinstance(payload, str) and payload.upper() == "FAIL":
                summary["failed_checks"] += 1
                fund_failed = True
        if fund_failed:
            summary["funds_in_breach"] += 1
    return summary


# ---------------------------------------------------------------------------
# Reconciliation helpers
# ---------------------------------------------------------------------------

def normalize_reconciliation_payload(raw_results: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Ensure reconciliation results expose ``summary`` and ``details`` keys."""

    normalized: Dict[str, Dict[str, Any]] = {}
    for fund_name, payload in (raw_results or {}).items():
        if not payload:
            continue
        if "summary" in payload or "details" in payload:
            summary = payload.get("summary", {})
            details = payload.get("details", {})
        else:
            summary = payload
            details = {}
        normalized[fund_name] = {"summary": summary, "details": details}
    return normalized


def summarise_reconciliation_breaks(normalized_results: Mapping[str, Dict[str, Any]]) -> Dict[str, int]:
    """Aggregate break counts across all funds."""

    totals: Dict[str, int] = {}
    for fund_payload in normalized_results.values():
        summary = fund_payload.get("summary", {})
        if not isinstance(summary, Mapping):
            continue
        for recon_type, metrics in summary.items():
            if isinstance(metrics, Mapping):
                values = metrics.values()
            elif isinstance(metrics, Iterable) and not isinstance(metrics, (str, bytes)):
                values = metrics
            else:
                values = [metrics]

            total_breaks = 0
            for value in values:
                if isinstance(value, numbers.Number):
                    total_breaks += int(value)
            totals[recon_type] = totals.get(recon_type, 0) + total_breaks
    return totals


# ---------------------------------------------------------------------------
# NAV reconciliation helpers
# ---------------------------------------------------------------------------

def normalize_nav_payload(raw_results: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Ensure NAV reconciliation payload exposes ``summary`` and ``details``."""

    normalized: Dict[str, Dict[str, Any]] = {}
    for fund_name, payload in (raw_results or {}).items():
        if not payload:
            continue
        summary = payload.get("summary", {}) if isinstance(payload, Mapping) else {}
        details = payload.get("detailed_calculations", {}) if isinstance(payload, Mapping) else {}
        normalized[fund_name] = {"summary": summary, "details": details}
    return normalized


def summarise_nav_differences(normalized_results: Mapping[str, Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate NAV differences across funds."""

    totals = {"funds": 0, "absolute_difference": 0.0}
    for payload in normalized_results.values():
        summary = payload.get("summary", {})
        if not summary:
            continue
        totals["funds"] += 1
        diff = summary.get("difference")
        try:
            totals["absolute_difference"] += abs(float(diff)) if diff is not None else 0.0
        except (TypeError, ValueError):
            continue
    return totals


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_compliance_result(value: Any) -> bool:
    return hasattr(value, "is_compliant") and hasattr(value, "details")


def _expand_compliance_result(test_name: str, result: ComplianceResult) -> Dict[str, Any]:
    calculations = getattr(result, "calculations", {}) or {}
    details = getattr(result, "details", {}) or {}

    if test_name == "summary_metrics":
        summary = dict(calculations)
        if isinstance(details, Mapping):
            summary.setdefault("status", details.get("status", "calculated"))
        return summary

    expanded: Dict[str, Any] = {"is_compliant": getattr(result, "is_compliant", False)}
    if isinstance(details, Mapping):
        expanded.update(details)
    expanded["calculations"] = dict(calculations)
    error = getattr(result, "error", None)
    if error:
        expanded["error"] = error
    return expanded


def build_daily_result_row(
    report_date: date,
    fund_name: str,
    fund_data: Mapping[str, Any],
    analysis_type: str,
) -> Dict[str, Any]:
    """Create a serialisable representation of a fund-day compliance result."""

    summary = fund_data.get("summary_metrics", {})
    return {
        "date": report_date,
        "etf": fund_name,
        "analysis_type": analysis_type,
        "cash_value": summary.get("cash_value"),
        "treasury": summary.get("treasury"),
        "equity_market_value": summary.get("equity_market_value"),
        "option_delta_adjusted_notional": summary.get("option_delta_adjusted_notional"),
        "option_market_value": summary.get("option_market_value"),
    }