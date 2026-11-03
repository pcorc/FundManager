"""Utility helpers shared by reporting components."""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from services.compliance_checker import ComplianceResult

_RECON_PAYLOAD_KEYS = ("summary", "details", "custodian_equity", "custodian_option", "index_equity")


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


# ---------------------------------------------------------------------------
# Reconciliation helpers
# ---------------------------------------------------------------------------

def _normalize_reconciliation_payload(raw_results: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Ensure reconciliation results expose ``summary`` and ``details`` keys.
    Also unwraps date-keyed structure if present.

    Handles both structures:
    - {fund_name: payload} (direct)
    - {date_str: {fund_name: payload}} (date-wrapped)

    Returns: {fund_name: {"summary": {...}, "details": {...}}}
    """

    if not raw_results:
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for fund_name, payload in (_unwrap_if_date_keyed(raw_results, _RECON_PAYLOAD_KEYS) or {}).items():
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap_if_date_keyed(payload: Mapping[str, Any], keys: Tuple[str, ...]) -> Mapping[str, Any]:
    """Return the FIRST date's ``{fund: payload}`` if ``payload`` is ``{date: {fund: payload}}``.

    Only the first value is sampled; later dates are dropped.
    """
    first_value = next(iter(payload.values()))
    if isinstance(first_value, dict) and first_value:
        sample = next(iter(first_value.values()))
        if isinstance(sample, dict) and any(key in sample for key in keys):
            return first_value
    return payload


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


