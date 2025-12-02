"""Run configurations and date calculation utilities for main.py orchestration."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set

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

# Convenience: All funds
ALL_FUNDS = set(FUND_DEFINITIONS.keys())

# Additional custom fund groups
ALL_REGISTERED_FUNDS = ETF_FUNDS | CLOSED_END_FUNDS | PRIVATE_FUNDS

# Wrapper-based groups
MUTUAL_AND_VIT_WRAPPERS = {"mutual_fund", "vit"}
MUTUAL_AND_VIT_FUNDS = {
    ticker
    for ticker, metadata in FUND_DEFINITIONS.items()
    if metadata.get("vehicle_wrapper") in MUTUAL_AND_VIT_WRAPPERS
}

# Convenience groups for daily runs
DAILY_GROUP_1_CEF_PRIVATE = CLOSED_END_FUNDS | PRIVATE_FUNDS
DAILY_GROUP_2_ETF = ETF_FUNDS
DAILY_GROUP_3_VIT_AND_MF = MUTUAL_AND_VIT_FUNDS | {"FTCSH", "FTMIX", "KNGIX"}

# Default batch definitions for common daily runs
DAILY_EOD_AND_RECON_RUNS = [
    "eod_compliance_daily_cef_private",
    "eod_recon_daily_cef_private",
    "eod_compliance_daily_etf",
    "eod_recon_daily_etf",
    "eod_compliance_daily_vitmf",
    "eod_recon_daily_vitmf",
]

DAILY_TRADING_COMPLIANCE_RUNS = [
    "trading_compliance_daily_cef_private",
    "trading_compliance_daily_etf",
    "trading_compliance_daily_vitmf",
]

def get_fund_group(group_name: str) -> Set[str]:
    """
    Retrieve a predefined fund group by name.

    Args:
        group_name: Name of the fund group (e.g., 'ETF_FUNDS', 'ALL_FUNDS')

    Returns:
        Set of fund tickers

    Raises:
        ValueError: If group_name is not recognized
    """
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
    }

    if group_name not in groups:
        available = ", ".join(sorted(groups.keys()))
        raise ValueError(f"Unknown fund group '{group_name}'. Available groups: {available}")

    return groups[group_name]


def build_fund_list(*items) -> List[str]:
    """
    Build a fund list by combining fund groups and individual tickers.

    Args:
        *items: Variable number of arguments that can be:
            - Set[str]: A fund group (e.g., ETF_FUNDS, CLOSED_END_FUNDS)
            - str: An individual fund ticker (e.g., "RDVI", "KNG")
            - List[str]: A list of fund tickers

    Returns:
        Sorted list of unique fund tickers

    Examples:
        # Combine ETFs with a single fund
        build_fund_list(ETF_FUNDS, "P20127")

        # Combine multiple groups
        build_fund_list(ETF_FUNDS, CLOSED_END_FUNDS, PRIVATE_FUNDS)

        # Mix groups and individual tickers
        build_fund_list(ETF_FUNDS, "P20127", "HE3B1")

        # Combine groups with a list of tickers
        build_fund_list(CLOSED_END_FUNDS, ["RDVI", "KNG"])
    """
    combined = set()

    for item in items:
        if isinstance(item, set):
            # It's a fund group
            combined.update(item)
        elif isinstance(item, str):
            # It's a single ticker
            combined.add(item)
        elif isinstance(item, list):
            # It's a list of tickers
            combined.update(item)
        else:
            raise TypeError(f"Unsupported type in build_fund_list: {type(item)}")

    return sorted(combined)


def exclude_funds(base_funds, *exclude_items) -> List[str]:
    """
    Remove specific funds from a fund group or list.

    Args:
        base_funds: Starting fund group (Set) or list (List[str])
        *exclude_items: Funds to exclude - can be:
            - Set[str]: A fund group to exclude
            - str: An individual fund ticker to exclude
            - List[str]: A list of fund tickers to exclude

    Returns:
        Sorted list of fund tickers with exclusions removed

    Examples:
        # All funds except a few specific ones
        exclude_funds(ALL_FUNDS, "RDVI", "KNG", "FTMIX")

        # All funds except private funds
        exclude_funds(ALL_FUNDS, PRIVATE_FUNDS)

        # ETFs except specific ones
        exclude_funds(ETF_FUNDS, ["RDVI", "KNG"])

        # Complex: All funds except private and a few specific ETFs
        exclude_funds(ALL_FUNDS, PRIVATE_FUNDS, "RDVI", "KNG")
    """
    # Convert base to set
    if isinstance(base_funds, set):
        result = base_funds.copy()
    elif isinstance(base_funds, list):
        result = set(base_funds)
    else:
        raise TypeError(f"base_funds must be a set or list, got {type(base_funds)}")

    # Remove excluded items
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
    """
    Calculate a business day offset from base_date.

    Args:
        base_date: Starting date (must be a business day)
        offset: Number of business days to offset (negative for past, positive for future)

    Returns:
        The resulting business day

    Examples:
        base_date = 2025-11-21 (Thursday)
        offset = -1 -> 2025-11-20 (Wednesday)
        offset = -2 -> 2025-11-19 (Tuesday)
    """
    if offset == 0:
        return base_date

    current = base_date
    days_moved = 0
    direction = 1 if offset > 0 else -1
    target_days = abs(offset)

    while days_moved < target_days:
        current += timedelta(days=direction)
        # Count only business days (Monday=0 to Friday=4)
        if current.weekday() < 5:
            days_moved += 1

    return current


def ensure_business_day(input_date: date) -> date:
    """
    Ensure the given date is a business day. If weekend, return previous Friday.

    Args:
        input_date: Any date

    Returns:
        The same date if Mon-Fri, otherwise the previous Friday
    """
    if input_date.weekday() < 5:
        return input_date

    # Saturday (5) -> go back 1 day to Friday
    # Sunday (6) -> go back 2 days to Friday
    days_back = input_date.weekday() - 4
    return input_date - timedelta(days=days_back)


def calculate_date_offsets(base_date: date) -> Dict[str, date]:
    """
    Calculate standard date offsets from a base date.

    Args:
        base_date: The base date (will be adjusted to business day if weekend)

    Returns:
        Dictionary with keys: 't', 't1', 't2' (T, T-1, T-2)
    """
    t = ensure_business_day(base_date)
    t1 = calculate_business_day_offset(t, -1)
    t2 = calculate_business_day_offset(t, -2)

    return {
        "t": t,
        "t1": t1,
        "t2": t2,
    }

def generate_business_date_range(start_date: date, end_date: date) -> List[date]:
    """
    Generate a list of business days (Mon-Fri) between two dates, inclusive.

    Args:
        start_date: Range start (will be adjusted to business day if weekend)
        end_date: Range end (will be adjusted to business day if weekend)

    Returns:
        A list of business-day ``date`` objects between the start and end dates.

    Raises:
        ValueError: If ``start_date`` occurs after ``end_date``
    """

    start_date = ensure_business_day(start_date)
    end_date = ensure_business_day(end_date)

    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    business_days: List[date] = []
    current = start_date

    while current <= end_date:
        if current.weekday() < 5:  # Monday-Friday
            business_days.append(current)
        current += timedelta(days=1)

    return business_days

# ============================================================================
# RUN CONFIGURATION TEMPLATES
# ============================================================================

# Standard compliance test suite
FULL_COMPLIANCE_TESTS = [
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

# Subset for diversification-focused testing
DIVERSIFICATION_TESTS = [
    "diversification_40act_check",
    "diversification_IRS_check",
    "diversification_IRC_check",
]


RUN_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ========================================================================
    # TRADING COMPLIANCE CONFIGURATIONS (Single Date Mode)
    # ========================================================================

    "trading_compliance_etfs": {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "funds": list(ETF_FUNDS),
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "etfs",  # Tag for output file names
        "description": "Trading compliance for all ETF funds",
    },

    "trading_compliance_closed_end_private": {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "funds": list(CLOSED_END_FUNDS | PRIVATE_FUNDS),
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "cefs_pfs",  # Tag for output file names
        "description": "Trading compliance for closed-end and private funds",
    },

    "trading_compliance_all_funds": {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "funds": list(ALL_FUNDS),
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "all_funds",  # Tag for output file names
        "description": "Trading compliance for all funds",
    },

    "trading_compliance_custom": {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "funds": [],  # Override this list when running
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "custom",  # Tag for output file names (override if needed)
        "description": "Trading compliance for custom fund list (must override 'funds')",
    },

    "trading_compliance_all_except": {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "funds": [],  # Override with exclude_funds(ALL_FUNDS, ...)
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "all_except",  # Tag for output file names (override if needed)
        "description": "Trading compliance for all funds except specified ones (must override 'funds')",
    },

    "trading_compliance_daily_cef_private": {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "funds": sorted(DAILY_GROUP_1_CEF_PRIVATE),
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "cef",  # Tag for output file names
        "description": "Daily trading compliance for closed-end and private funds",
    },

    "trading_compliance_daily_etf": {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "funds": sorted(DAILY_GROUP_2_ETF),
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "etf",  # Tag for output file names
        "description": "Daily trading compliance for ETF funds",
    },

    "trading_compliance_daily_vitmf": {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "funds": sorted(DAILY_GROUP_3_VIT_AND_MF),
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "vitmf",  # Tag for output file names
        "description": "Daily trading compliance for mutual fund and VIT accounts",
    },

    # ========================================================================
    # EOD COMPLIANCE CONFIGURATIONS (Single Date Mode)
    # ========================================================================

    "eod_compliance_etfs": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": list(ETF_FUNDS),
        "eod_reports": ["compliance"],
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "etfs",  # Tag for output file names
        "description": "EOD compliance checks for all ETF funds",
    },

    "eod_compliance_closed_end_private": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": list(CLOSED_END_FUNDS | PRIVATE_FUNDS),
        "eod_reports": ["compliance"],
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "cefs_pfs",  # Tag for output file names
        "description": "EOD compliance checks for closed-end and private funds",
    },

    "eod_compliance_all_funds": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": list(ALL_FUNDS),
        "eod_reports": ["compliance"],
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "all_funds",  # Tag for output file names
        "description": "EOD compliance checks for all funds",
    },

    "eod_compliance_custom": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": [],  # Override this list when running
        "eod_reports": ["compliance"],
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "custom",  # Tag for output file names (override if needed)
        "description": "EOD compliance checks for custom fund list (must override 'funds')",
    },

    "eod_compliance_all_except": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": [],  # Override with exclude_funds(ALL_FUNDS, ...)
        "eod_reports": ["compliance"],
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "all_except",  # Tag for output file names (override if needed)
        "description": "EOD compliance checks for all funds except specified ones (must override 'funds')",
    },

    # ========================================================================
    # EOD RECONCILIATION CONFIGURATIONS (Single Date Mode)
    # ========================================================================

    "eod_recon_etfs": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": list(ETF_FUNDS),
        "eod_reports": ["reconciliation", "nav"],
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "etfs",  # Tag for output file names
        "description": "Holdings and NAV reconciliation for all ETF funds",
    },

    "eod_recon_closed_end_private": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": list(CLOSED_END_FUNDS | PRIVATE_FUNDS),
        "eod_reports": ["reconciliation", "nav"],
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "cefs_pfs",  # Tag for output file names
        "description": "Holdings and NAV reconciliation for closed-end and private funds",
    },

    "eod_recon_all_funds": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": list(ALL_FUNDS),
        "eod_reports": ["reconciliation", "nav"],
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "all_funds",  # Tag for output file names
        "description": "Holdings and NAV reconciliation for all funds",
    },

    "eod_recon_custom": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": [],  # Override this list when running
        "eod_reports": ["reconciliation", "nav"],
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "custom",  # Tag for output file names (override if needed)
        "description": "Holdings and NAV reconciliation for custom fund list (must override 'funds')",
    },

    "eod_recon_all_except": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": [],  # Override with exclude_funds(ALL_FUNDS, ...)
        "eod_reports": ["reconciliation", "nav"],
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "all_except",  # Tag for output file names (override if needed)
        "description": "Holdings and NAV reconciliation for all funds except specified ones (must override 'funds')",
    },

    "eod_compliance_daily_cef_private": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": sorted(DAILY_GROUP_1_CEF_PRIVATE),
        "eod_reports": ["compliance"],
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "cef",  # Tag for output file names
        "description": "Daily EOD compliance for closed-end and private funds",
    },

    "eod_compliance_daily_etf": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": sorted(DAILY_GROUP_2_ETF),
        "eod_reports": ["compliance"],
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "etf",  # Tag for output file names
        "description": "Daily EOD compliance for ETF funds",
    },

    "eod_compliance_daily_vitmf": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": sorted(DAILY_GROUP_3_VIT_AND_MF),
        "eod_reports": ["compliance"],
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "vitmf",  # Tag for output file names
        "description": "Daily EOD compliance for mutual fund and VIT accounts",
    },

    "eod_recon_daily_cef_private": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": sorted(DAILY_GROUP_1_CEF_PRIVATE),
        "eod_reports": ["reconciliation", "nav"],
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "cef",  # Tag for output file names
        "description": "Daily holdings and NAV reconciliation for closed-end and private funds",
    },

    "eod_recon_daily_etf": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": sorted(DAILY_GROUP_2_ETF),
        "eod_reports": ["reconciliation", "nav"],
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "etf",  # Tag for output file names
        "description": "Daily holdings and NAV reconciliation for ETF funds",
    },

    "eod_recon_daily_vitmf": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": sorted(DAILY_GROUP_3_VIT_AND_MF),
        "eod_reports": ["reconciliation", "nav"],
        "create_pdf": True,
        "output_dir": "./outputs",
        "output_tag": "vitmf",  # Tag for output file names
        "description": "Daily holdings and NAV reconciliation for mutual fund and VIT accounts",
    },

    # ========================================================================
    # TIME SERIES CONFIGURATIONS (Range Date Mode)
    # ========================================================================

    "time_series_diversification": {
        "analysis_type": "eod",
        "date_mode": "range",
        "funds": [
            "P20127", "P21026", "P2726", "P30128", "P31027", "P3727",
            "R21126", "HE3B1", "HE3B2", "TR2B1", "TR2B2"
        ],
        "eod_reports": ["compliance"],
        "compliance_tests": DIVERSIFICATION_TESTS,
        "create_pdf": False,
        "output_dir": "./outputs",
        "generate_daily_reports": False,
        "output_tag": "diversification",  # Tag for output file names
        "description": "Time series diversification testing for private funds",
    },

    "time_series_full_compliance": {
        "analysis_type": "eod",
        "date_mode": "range",
        "funds": list(CLOSED_END_FUNDS),
        "eod_reports": ["compliance"],
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": False,
        "output_dir": "./outputs",
        "generate_daily_reports": False,
        "output_tag": "full_compliance",  # Tag for output file names
        "description": "Time series full compliance for closed-end funds",
    },
}


# ============================================================================
# CONFIGURATION HELPER FUNCTIONS
# ============================================================================

def get_config(config_name: str) -> Dict[str, Any]:
    """
    Retrieve a run configuration by name.

    Args:
        config_name: Name of the configuration

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If config_name is not found
    """
    if config_name not in RUN_CONFIGS:
        available = ", ".join(sorted(RUN_CONFIGS.keys()))
        raise ValueError(f"Unknown configuration '{config_name}'. Available: {available}")

    return RUN_CONFIGS[config_name].copy()


def list_available_configs() -> List[str]:
    """Return list of all available configuration names."""
    return sorted(RUN_CONFIGS.keys())


def get_config_description(config_name: str) -> str:
    """Get the description for a specific configuration."""
    config = get_config(config_name)
    return config.get("description", "No description available")


def merge_overrides(config: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge user overrides into a configuration.

    Args:
        config: Base configuration
        overrides: Override values (can be None)

    Returns:
        Merged configuration dictionary
    """
    if not overrides:
        return config

    merged = config.copy()
    merged.update(overrides)
    return merged