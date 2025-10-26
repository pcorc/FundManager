"""Utility helpers shared by compliance reporting components."""

from __future__ import annotations

import numbers
from datetime import date
from typing import Any, Dict, Mapping, Tuple

try:  # pragma: no cover - optional import for typing only
    from services.compliance_checker import ComplianceResult
except Exception:  # pragma: no cover - safeguard during cold starts
    ComplianceResult = Any  # type: ignore


def format_number(value: Any, digits: int = 0) -> str:
    """Return a string representation with thousand separators."""

    if value is None:
        return ""
    if isinstance(value, numbers.Number):
        format_spec = f"{{:,.{digits}f}}"
        return format_spec.format(value)
    return str(value)


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
            flattened[(fund_name, date_str)] = data
    return flattened


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