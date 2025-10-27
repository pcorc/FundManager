"""Helpers to create compact compliance summaries for reporting."""
from __future__ import annotations

from typing import Any, Dict, Mapping

from reporting.report_utils import normalize_compliance_results


def extract_compliance_summary(results: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    """Convert raw compliance results into PASS/FAIL/N/A summaries."""

    normalized = normalize_compliance_results(results or {})
    summary: Dict[str, Dict[str, str]] = {}

    for fund_name, tests in normalized.items():
        fund_summary: Dict[str, str] = {}
        for test_name, payload in tests.items():
            status = _coerce_status(payload)
            formatted_name = _format_test_name(test_name)
            fund_summary[formatted_name] = status
        summary[str(fund_name)] = fund_summary

    return summary


def _coerce_status(payload: Any) -> str:
    if isinstance(payload, Mapping):
        if "is_compliant" in payload:
            value = payload.get("is_compliant")
            if isinstance(value, bool):
                return "PASS" if value else "FAIL"
            if isinstance(value, str):
                return _normalise_str_status(value)
        if "status" in payload:
            return _normalise_str_status(payload.get("status"))
        if "value" in payload:
            return _normalise_str_status(payload.get("value"))
    elif isinstance(payload, bool):
        return "PASS" if payload else "FAIL"
    elif isinstance(payload, str):
        return _normalise_str_status(payload)
    return "N/A"


def _normalise_str_status(value: Any) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip().upper()
    if text in {"PASS", "PASSED", "COMPLIANT", "TRUE", "Y"}:
        return "PASS"
    if text in {"FAIL", "FAILED", "BREACH", "FALSE", "N", "NO"}:
        return "FAIL"
    return "N/A"


def _format_test_name(name: Any) -> str:
    if not isinstance(name, str):
        return str(name)
    canonical = name.strip()
    replacements = {
        "illiquid": "Illiquid",
        "summary_metrics": "Summary",
    }
    if canonical.lower() in replacements:
        return replacements[canonical.lower()]
    if canonical.upper() in {"80%", "40 ACT", "12D1", "12D2", "12D3"}:
        return canonical.upper()
    return canonical.replace("_", " ").title()