"""Run configurations and date calculation utilities for main.py orchestration."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from config.fund_definitions import (
    CLOSED_END_FUNDS,
    DIVERSIFIED_FUNDS,
    ETF_FUNDS,
    FUND_DEFINITIONS,
    INDEX_FLEX_FUNDS,
    LISTED_INDEX_OPTION_FUNDS,
    LISTED_SINGLE_STOCK_OPTION_FUNDS,
    NON_DIVERSIFIED_FUNDS,
    PRIVATE_FUNDS,
    SINGLE_STOCK_FLEX_FUNDS,
)

# ============================================================================
# FUND GROUP UTILITIES
# ============================================================================

ALL_FUNDS: Set[str] = set(FUND_DEFINITIONS.keys())
ALL_REGISTERED_FUNDS: Set[str] = ETF_FUNDS | CLOSED_END_FUNDS | PRIVATE_FUNDS

# Mutual-fund / VIT cohort + the three loose tickers historically run alongside.
MUTUAL_AND_VIT_FUNDS: Set[str] = {
    ticker
    for ticker, metadata in FUND_DEFINITIONS.items()
    if metadata.get("vehicle_wrapper") in {"mutual_fund", "vit"}
}
VITMF_COHORT: Set[str] = MUTUAL_AND_VIT_FUNDS | {"FTCSH", "FTMIX", "KNGIX"}


def get_fund_group(group_name: str) -> Set[str]:
    """Retrieve a predefined fund group by name."""
    groups = {
        "ALL_FUNDS": ALL_FUNDS,
        "ETF_FUNDS": ETF_FUNDS,
        "CLOSED_END_FUNDS": CLOSED_END_FUNDS,
        "PRIVATE_FUNDS": PRIVATE_FUNDS,
        "DIVERSIFIED_FUNDS": DIVERSIFIED_FUNDS,
        "NON_DIVERSIFIED_FUNDS": NON_DIVERSIFIED_FUNDS,
        "LISTED_INDEX_OPTION_FUNDS": LISTED_INDEX_OPTION_FUNDS,
        "LISTED_SINGLE_STOCK_OPTION_FUNDS": LISTED_SINGLE_STOCK_OPTION_FUNDS,
        "INDEX_FLEX_FUNDS": INDEX_FLEX_FUNDS,
        "SINGLE_STOCK_FLEX_FUNDS": SINGLE_STOCK_FLEX_FUNDS,
        "ALL_REGISTERED_FUNDS": ALL_REGISTERED_FUNDS,
        "MUTUAL_AND_VIT_FUNDS": MUTUAL_AND_VIT_FUNDS,
        "VITMF_COHORT": VITMF_COHORT,
    }
    if group_name not in groups:
        available = ", ".join(sorted(groups.keys()))
        raise ValueError(f"Unknown fund group '{group_name}'. Available groups: {available}")
    return groups[group_name]


def build_fund_list(*items) -> List[str]:
    """Combine fund groups (sets), single tickers (str), or lists of tickers into a sorted list."""
    combined: Set[str] = set()
    for item in items:
        if isinstance(item, set):
            combined.update(item)
        elif isinstance(item, str):
            combined.add(item)
        elif isinstance(item, list):
            combined.update(item)
        else:
            raise TypeError(f"Unsupported type in build_fund_list: {type(item)}")
    return sorted(combined)


def exclude_funds(base_funds, *exclude_items) -> List[str]:
    """Return base_funds with the specified groups/tickers/lists removed (sorted)."""
    if isinstance(base_funds, set):
        result = base_funds.copy()
    elif isinstance(base_funds, list):
        result = set(base_funds)
    else:
        raise TypeError(f"base_funds must be a set or list, got {type(base_funds)}")

    for item in exclude_items:
        if isinstance(item, set):
            result -= item
        elif isinstance(item, str):
            result.discard(item)
        elif isinstance(item, list):
            result -= set(item)
        else:
            raise TypeError(f"Unsupported type in exclude_funds: {type(item)}")
    return sorted(result)


# ============================================================================
# DATE CALCULATION UTILITIES
# ============================================================================

def calculate_business_day_offset(base_date: date, offset: int) -> date:
    """Calculate a business-day offset (positive=forward, negative=back) from base_date."""
    if offset == 0:
        return base_date
    direction = 1 if offset > 0 else -1
    target = abs(offset)
    current = base_date
    days_moved = 0
    while days_moved < target:
        current += timedelta(days=direction)
        if current.weekday() < 5:
            days_moved += 1
    return current


def ensure_business_day(input_date: date) -> date:
    """Return input_date if it's Mon-Fri; otherwise the previous Friday."""
    if input_date.weekday() < 5:
        return input_date
    return input_date - timedelta(days=input_date.weekday() - 4)


def calculate_date_offsets(base_date: date) -> Dict[str, date]:
    """Return {'t', 't1', 't2'} for T, T-1, T-2 business days from base_date."""
    t = ensure_business_day(base_date)
    return {
        "t": t,
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
# CONFIG GENERATION
#
# Every config has the same 8 fields. The only meaningful variation is
# cohort (which funds + filename tag) × operation (compliance / recon / full /
# trading_compliance). Configs are generated programmatically rather than
# repeated as dict literals.
# ============================================================================

DEFAULT_OUTPUT_DIR = "./outputs"


# (cohort_suffix, fund_set, output_tag, descriptive_phrase)
# `cohort_suffix` is appended to config names; `output_tag` goes in filenames.
# They differ historically — e.g. the `closed_end_private` cohort writes files
# tagged `cefs_pfs`.
_COHORTS: Sequence[Tuple[str, Set[str], str, str]] = (
    ("etfs",               ETF_FUNDS,                          "etfs",      "ETF funds"),
    ("closed_end_private", CLOSED_END_FUNDS | PRIVATE_FUNDS,   "cefs_pfs",  "closed-end and private funds"),
    ("vitmf",              VITMF_COHORT,                       "vitmf",     "mutual fund and VIT accounts"),
    ("all_funds",          ALL_FUNDS,                          "all_funds", "all funds"),
    ("custom",             set(),                              "custom",    "custom fund list (must override 'funds')"),
)

# (op_suffix, eod_reports, descriptive_prefix)
_EOD_OPERATIONS: Sequence[Tuple[str, List[str], str]] = (
    ("compliance", ["compliance"],                          "EOD compliance checks for"),
    ("recon",      ["reconciliation", "nav"],               "Holdings and NAV reconciliation for"),
    ("full",       ["compliance", "reconciliation", "nav"], "Compliance + holdings recon + NAV recon for"),
)


def _eod_config(
    funds: Set[str],
    reports: List[str],
    output_tag: str,
    description: str,
    *,
    compliance_tests: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": sorted(funds),
        "eod_reports": reports,
        "compliance_tests": compliance_tests if compliance_tests is not None else FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "output_tag": output_tag,
        "description": description,
    }


def _trading_config(
    funds: Set[str],
    output_tag: str,
    description: str,
    *,
    compliance_tests: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "funds": sorted(funds),
        "compliance_tests": compliance_tests if compliance_tests is not None else FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "output_tag": output_tag,
        "description": description,
    }


def _time_series_config(
    funds: List[str],
    output_tag: str,
    description: str,
    *,
    compliance_tests: List[str],
) -> Dict[str, Any]:
    return {
        "analysis_type": "eod",
        "date_mode": "range",
        "funds": funds,
        "eod_reports": ["compliance"],
        "compliance_tests": compliance_tests,
        "create_pdf": False,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "generate_daily_reports": False,
        "output_tag": output_tag,
        "description": description,
    }


# Cohort × operation product: generates 5 × 3 = 15 EOD configs and 5 trading
# configs. Names follow the pattern `eod_<op>_<cohort>` and `trading_compliance_<cohort>`:
#   eod_compliance_etfs, eod_recon_etfs, eod_full_etfs, trading_compliance_etfs,
#   eod_compliance_closed_end_private, …
RUN_CONFIGS: Dict[str, Dict[str, Any]] = {}

for _cohort_suffix, _funds, _tag, _label in _COHORTS:
    for _op_suffix, _reports, _op_prefix in _EOD_OPERATIONS:
        RUN_CONFIGS[f"eod_{_op_suffix}_{_cohort_suffix}"] = _eod_config(
            _funds, _reports, _tag, f"{_op_prefix} {_label}",
        )
    RUN_CONFIGS[f"trading_compliance_{_cohort_suffix}"] = _trading_config(
        _funds, _tag, f"Trading compliance for {_label}",
    )

# Bespoke time-series configs (range mode, hand-picked funds, no PDF).
RUN_CONFIGS["time_series_diversification"] = _time_series_config(
    funds=[
        "P20127", "P21026", "P2726", "P30128", "P31027", "P3727",
        "R21126", "HE3B1", "HE3B2", "TR2B1", "TR2B2",
    ],
    output_tag="diversification",
    description="Time series diversification testing for private funds",
    compliance_tests=DIVERSIFICATION_TESTS,
)
RUN_CONFIGS["time_series_full_compliance"] = _time_series_config(
    funds=sorted(CLOSED_END_FUNDS),
    output_tag="full_compliance",
    description="Time series full compliance for closed-end funds",
    compliance_tests=FULL_COMPLIANCE_TESTS,
)
RUN_CONFIGS["custom1"] = _time_series_config(
    funds=["FDND", "DOGG"],
    output_tag="jan31_irs_compliance",
    description="Time series IRS diversification for FDND and DOGG",
    compliance_tests=["diversification_IRS_check"],
)


# ============================================================================
# DAILY RUN BATCHES
# ============================================================================

DAILY_EOD_AND_RECON_RUNS: List[str] = [
    "eod_compliance_closed_end_private",
    "eod_recon_closed_end_private",
    "eod_compliance_etfs",
    "eod_recon_etfs",
    "eod_compliance_vitmf",
    "eod_recon_vitmf",
]

DAILY_TRADING_COMPLIANCE_RUNS: List[str] = [
    "trading_compliance_closed_end_private",
    "trading_compliance_etfs",
    "trading_compliance_vitmf",
]


# ============================================================================
# CONFIG HELPERS
# ============================================================================

def get_config(config_name: str) -> Dict[str, Any]:
    """Return a copy of the named run configuration."""
    if config_name not in RUN_CONFIGS:
        available = ", ".join(sorted(RUN_CONFIGS.keys()))
        raise ValueError(f"Unknown configuration '{config_name}'. Available: {available}")
    return RUN_CONFIGS[config_name].copy()


def list_available_configs() -> List[str]:
    """Sorted list of every available configuration name."""
    return sorted(RUN_CONFIGS.keys())


def get_config_description(config_name: str) -> str:
    """Human-readable description for a configuration."""
    return get_config(config_name).get("description", "No description available")


def merge_overrides(config: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a shallow copy of config with overrides applied."""
    if not overrides:
        return config
    merged = config.copy()
    merged.update(overrides)
    return merged
