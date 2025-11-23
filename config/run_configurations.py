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
        "description": "Trading compliance for all ETF funds",
    },

    "trading_compliance_closed_end_private": {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "funds": list(CLOSED_END_FUNDS | PRIVATE_FUNDS),
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "description": "Trading compliance for closed-end and private funds",
    },

    "trading_compliance_all_funds": {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "funds": list(ALL_FUNDS),
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "description": "Trading compliance for all funds",
    },

    "trading_compliance_custom": {
        "analysis_type": "trading_compliance",
        "date_mode": "single",
        "funds": [],  # Override this list when running
        "compliance_tests": FULL_COMPLIANCE_TESTS,
        "create_pdf": True,
        "output_dir": "./outputs",
        "description": "Trading compliance for custom fund list (must override 'funds')",
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
        "description": "EOD compliance checks for custom fund list (must override 'funds')",
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
        "description": "Holdings and NAV reconciliation for all ETF funds",
    },

    "eod_recon_closed_end_private": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": list(CLOSED_END_FUNDS | PRIVATE_FUNDS),
        "eod_reports": ["reconciliation", "nav"],
        "create_pdf": True,
        "output_dir": "./outputs",
        "description": "Holdings and NAV reconciliation for closed-end and private funds",
    },

    "eod_recon_all_funds": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": list(ALL_FUNDS),
        "eod_reports": ["reconciliation", "nav"],
        "create_pdf": True,
        "output_dir": "./outputs",
        "description": "Holdings and NAV reconciliation for all funds",
    },

    "eod_recon_custom": {
        "analysis_type": "eod",
        "date_mode": "single",
        "funds": [],  # Override this list when running
        "eod_reports": ["reconciliation", "nav"],
        "create_pdf": True,
        "output_dir": "./outputs",
        "description": "Holdings and NAV reconciliation for custom fund list (must override 'funds')",
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