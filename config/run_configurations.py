"""Run presets: the fund cohorts, compliance rule suites and MODES a run is built from.

``processing.api.build_run(mode, cohorts=[...], funds=[...])`` turns a mode
plus a fund selection into a ``RunRequest``; ``processing.api.run`` executes it.
"""
from __future__ import annotations

from typing import Any, Dict, List

from config.fund_definitions import (
    ALL_FUNDS,
    CLOSED_END_FUNDS,
    ETF_FUNDS,
    PRIVATE_FUNDS,
    VIT_AND_MUTUAL_FUNDS,
)


DEFAULT_OUTPUT_DIR = r"G:\Shared drives\Portfolio Management\Funds\Archive\Daily_Compliance\2026"

# Cohort name -> live set (filled in place by fund_definitions.load_cohorts_from_db).
COHORTS = {
    "ALL_FUNDS": ALL_FUNDS,
    "ETF_FUNDS": ETF_FUNDS,
    "CLOSED_END_FUNDS": CLOSED_END_FUNDS,
    "PRIVATE_FUNDS": PRIVATE_FUNDS,
    "VIT_AND_MUTUAL_FUNDS": VIT_AND_MUTUAL_FUNDS,
}


def exclude_funds(base, *exclude_items) -> List[str]:
    """Return base with the given groups / lists / tickers removed (sorted)."""
    if not isinstance(base, (set, list)):
        raise TypeError(f"base must be a set or list, got {type(base)}")
    result = set(base)
    for item in exclude_items:
        if isinstance(item, (set, list)):
            result -= set(item)
        elif isinstance(item, str):
            result.discard(item)
        else:
            raise TypeError(f"Unsupported type in exclude_funds: {type(item)}")
    return sorted(result)


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

_BASE: Dict[str, Any] = {
    "analysis_type": "eod",
    "date_mode": "single",
    "compliance_tests": FULL_COMPLIANCE_TESTS,
    "create_pdf": True,
    "output_dir": DEFAULT_OUTPUT_DIR,
}

MODES: Dict[str, Dict[str, Any]] = {
    "eod": {**_BASE, "eod_reports": ["compliance", "reconciliation", "nav"]},
    "compliance": {**_BASE, "eod_reports": ["compliance"]},
    "reconciliation": {**_BASE, "eod_reports": ["reconciliation"]},
    "nav": {**_BASE, "eod_reports": ["nav"]},
    "trading_compliance": {**_BASE, "analysis_type": "trading_compliance"},
    "time_series": {**_BASE, "date_mode": "range", "eod_reports": ["compliance"],
                    "create_pdf": False, "generate_daily_reports": False},
}
