"""Utilities for summarising compliance checker results."""
from __future__ import annotations

from typing import Any, Dict, Mapping


PASS_VALUES = {True, "PASS", "Pass", "pass"}
FAIL_VALUES = {False, "FAIL", "Fail", "fail"}


def _normalise_status(value: Any) -> str:
    if value in PASS_VALUES:
        return "PASS"
    if value in FAIL_VALUES:
        return "FAIL"
    if isinstance(value, str):
        upper = value.upper()
        if upper in {"PASS", "FAIL"}:
            return upper
    return "N/A"


def extract_compliance_summary(results: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    """Return a light-weight PASS/FAIL summary for each compliance test."""

    summary: Dict[str, Dict[str, str]] = {}
    for fund_name, fund_results in (results or {}).items():
        if not isinstance(fund_results, Mapping):
            continue
        fund_summary: Dict[str, str] = {}
        for test_name, payload in fund_results.items():
            if not isinstance(test_name, str):
                continue
            status = None
            if isinstance(payload, Mapping):
                for key in ("status", "result", "outcome", "is_compliant"):
                    if key in payload:
                        status = payload[key]
                        break
                if status is None and "details" in payload and isinstance(payload["details"], Mapping):
                    status = payload["details"].get("status")
            else:
                status = payload
            fund_summary[test_name] = _normalise_status(status)
        if fund_summary:
            summary[fund_name] = fund_summary
    return summary