"""Run configurations and date calculation utilities for main.py orchestration.

Two concepts:
    MODES       - the three things you can run (eod / trading_compliance / time_series)
    Fund groups - the cohorts you pick from to populate a run

`build_run(mode, cohorts=[...], funds=[...])` combines them into a runnable
config. There is no named-config catalog any more — every run is described
by mode + fund selection at the call site.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from config.fund_definitions import (
    ALL_FUNDS,
    CLOSED_END_FUNDS,
    ETF_FUNDS,
    PRIVATE_FUNDS,
    VIT_AND_MUTUAL_FUNDS,
)


DEFAULT_OUTPUT_DIR = "./outputs"


# ============================================================================
# FUND GROUPS
# ============================================================================

def get_fund_group(group_name: str) -> Set[str]:
    """Retrieve a predefined fund group by name."""
    groups = {
        "ALL_FUNDS":            ALL_FUNDS,
        "ETF_FUNDS":            ETF_FUNDS,
        "CLOSED_END_FUNDS":     CLOSED_END_FUNDS,
        "PRIVATE_FUNDS":        PRIVATE_FUNDS,
        "VIT_AND_MUTUAL_FUNDS": VIT_AND_MUTUAL_FUNDS,
    }
    if group_name not in groups:
        available = ", ".join(sorted(groups))
        raise ValueError(f"Unknown fund group '{group_name}'. Available: {available}")
    return groups[group_name]


def build_fund_list(*items) -> List[str]:
    """Combine fund groups, single tickers, or lists of tickers into a sorted list."""
    combined: Set[str] = set()
    for item in items:
        if isinstance(item, set):
            combined |= item
        elif isinstance(item, list):
            combined |= set(item)
        elif isinstance(item, str):
            combined.add(item)
        else:
            raise TypeError(f"Unsupported type in build_fund_list: {type(item)}")
    return sorted(combined)


def exclude_funds(base, *exclude_items) -> List[str]:
    """Return base with the given groups / lists / tickers removed (sorted)."""
    if isinstance(base, set):
        result = base.copy()
    elif isinstance(base, list):
        result = set(base)
    else:
        raise TypeError(f"base must be a set or list, got {type(base)}")
    for item in exclude_items:
        if isinstance(item, set):
            result -= item
        elif isinstance(item, list):
            result -= set(item)
        elif isinstance(item, str):
            result.discard(item)
        else:
            raise TypeError(f"Unsupported type in exclude_funds: {type(item)}")
    return sorted(result)


# ============================================================================
# COMPLIANCE TEST SUITES
# ============================================================================

FULL_COMPLIANCE_TESTS: List[str] = [
    "gics_compliance",
    "prospectus_80pct_policy",
    "diversification_40act_check",
    "diversification_IRS_check",
    "diversification_IRC_check",
    "max_15pct_illiquid_sai",
    "real_estate_check",
    "commodities_check",
    "twelve_d1a_other_inv_cos",
    "twelve_d2_insurance_cos",
    "twelve_d3_sec_biz",
]

DIVERSIFICATION_TESTS: List[str] = [
    "diversification_40act_check",
    "diversification_IRS_check",
    "diversification_IRC_check",
]


# ============================================================================
# MODES — what you can run
# ============================================================================

MODES: Dict[str, Dict[str, Any]] = {
    "eod": {
        "analysis_type": "eod",
        "date_mode": "single",
        "eod_reports": ["compliance", "reconciliation", "nav"],
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": DEFAULT_OUTPUT_DIR,
    },
    "trading_compliance": {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": DEFAULT_OUTPUT_DIR,
    },
    "time_series": {
        "analysis_type": "eod",
        "date_mode": "range",
        "eod_reports": ["compliance"],
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": False,
        "generate_daily_reports": False,
        "output_dir": DEFAULT_OUTPUT_DIR,
    },
}


def build_run(
    mode: str,
    *,
    cohorts: Optional[List[Set[str]]] = None,
    funds: Optional[List[str]] = None,
    date_range: Optional[Tuple[str, str]] = None,
    output_tag: Optional[str] = None,
    compliance_tests: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Combine a mode with a fund selection into a runnable config.

    Args:
        mode:             "eod" | "trading_compliance" | "time_series"
        cohorts:          List of fund-group sets to include (e.g. [ETF_FUNDS, CLOSED_END_FUNDS])
        funds:            Explicit ticker list to intersect with cohorts (or use alone)
        date_range:       (start, end) — only meaningful for mode="time_series"
        output_tag:       Filename tag; defaults to the mode name
        compliance_tests: Optional override for the test suite (defaults to FULL_COMPLIANCE_TESTS)
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Available: {sorted(MODES)}")

    selected: Set[str] = set()
    if cohorts:
        for c in cohorts:
            selected |= set(c)
    if funds:
        selected = (selected & set(funds)) if selected else set(funds)

    cfg = dict(MODES[mode])
    cfg["funds"] = sorted(selected)
    cfg["output_tag"] = output_tag or mode

    if compliance_tests is not None:
        cfg["compliance_tests"] = compliance_tests

    if mode == "time_series" and date_range:
        cfg["start_date"], cfg["end_date"] = date_range

    return cfg


# ============================================================================
# DATE CALCULATION UTILITIES
# ============================================================================

def calculate_business_day_offset(base_date: date, offset: int) -> date:
    """Business-day offset (positive=forward, negative=back) from base_date."""
    if offset == 0:
        return base_date
    direction = 1 if offset > 0 else -1
    target = abs(offset)
    current = base_date
    moved = 0
    while moved < target:
        current += timedelta(days=direction)
        if current.weekday() < 5:
            moved += 1
    return current


def ensure_business_day(input_date: date) -> date:
    """Return input_date if it's Mon-Fri, else the previous Friday."""
    if input_date.weekday() < 5:
        return input_date
    return input_date - timedelta(days=input_date.weekday() - 4)


def calculate_date_offsets(base_date: date) -> Dict[str, date]:
    """Return {'t', 't1', 't2'} for T, T-1, T-2 business days from base_date."""
    t = ensure_business_day(base_date)
    return {
        "t":  t,
        "t1": calculate_business_day_offset(t, -1),
        "t2": calculate_business_day_offset(t, -2),
    }


def generate_business_date_range(start_date: date, end_date: date) -> List[date]:
    """List of business days between start_date and end_date inclusive."""
    start_date = ensure_business_day(start_date)
    end_date = ensure_business_day(end_date)
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    days: List[date] = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days