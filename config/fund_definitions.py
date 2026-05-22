"""Static fund metadata consumed by :func:`processing.fund.build_fund_registry`."""
from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from sqlalchemy import text


"""Per-fund policy / compliance config.

Anything that names a source table is GONE — those routes now live in the
reconciliation.* views and resolve fund_ticker through
accounts_mapping.vw_tif_account_numbers.

What stays here is the small set of policy flags that drive compliance and
reconciliation logic and have no equivalent column in v_fund_properties.

Fields surfaced by vw_tif_account_numbers (no longer duplicated here):
  expense_ratio        -> Expense_Ratio
  index_ticker_join    -> Benchmark
  vehicle_wrapper      -> EOD_Report_Strategy (CEF / PF / TIF / VIT)
  custodian            -> Custodian / Custody_Service_Provider
  custody_account      -> Custody_Account
  launch_date          -> Launch_Date
  creation_unit_size   -> Creation_Unit_Size
  bbg_ticker           -> BBG_Ticker
"""



FUND_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "DOGG": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.08,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "single_stock",
        "has_otc": False,
        "has_treasury": True,
        "diversification_status": "non-diversified",
    },
    "FDND": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.08,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "index",
        "has_flex_option": False,
        "flex_option_type": None,
        "has_otc": False,
        "has_treasury": False,
        "diversification_status": "non-diversified",
    },
    "FGSI": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.08,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "index",
        "has_flex_option": False,
        "flex_option_type": None,
        "has_otc": False,
        "has_treasury": False,
        "diversification_status": "non-diversified",
    },
    "FTCSH": {
        "option_roll_tenor": "quarterly",
        "overwrite": 0.0,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "index",
        "has_flex_option": False,
        "flex_option_type": None,
        "has_otc": False,
        "has_treasury": False,
        "diversification_status": "non-diversified",
    },
    "FTMIX": {
        "option_roll_tenor": "monthly",
        "overwrite": 0.0,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "index",
        "has_flex_option": False,
        "flex_option_type": None,
        "has_otc": False,
        "has_treasury": False,
    },
    "HE3B1": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.12,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "HE3B2": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.12,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "HE3B3": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.12,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "index",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "KNG": {
        "option_roll_tenor": "monthly",
        "overwrite": 0.08,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": False,
        "flex_option_type": None,
        "has_otc": False,
        "has_treasury": False,
        "diversification_status": "diversified",
    },
    "KNGIX": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.1,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": False,
        "flex_option_type": None,
        "has_otc": False,
        "has_treasury": False,
        "diversification_status": "diversified",
    },
    "P20127": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.15,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "P21026": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.15,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "P2726": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.15,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "P30128": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.15,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "P31027": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.12,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "P3727": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.12,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "PD227": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.12,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": False,
        "flex_option_type": None,
        "has_otc": False,
        "has_treasury": False,
        "diversification_status": "excluded",
    },
    "PF227": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.12,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": False,
        "flex_option_type": None,
        "has_otc": False,
        "has_treasury": False,
        "diversification_status": "excluded",
    },
    "PF27V1": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.16,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": False,
        "flex_option_type": None,
        "has_otc": False,
        "has_treasury": False,
        "diversification_status": "excluded",
    },
    "R21126": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.12,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "index",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "RDVI": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.08,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "index",
        "has_flex_option": False,
        "flex_option_type": None,
        "has_otc": False,
        "has_treasury": False,
        "diversification_status": "non-diversified",
    },
    "SDVD": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.08,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "index",
        "has_flex_option": False,
        "flex_option_type": None,
        "has_otc": False,
        "has_treasury": False,
        "diversification_status": "non-diversified",
    },
    "TDVI": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.08,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "index",
        "has_flex_option": False,
        "flex_option_type": None,
        "has_otc": False,
        "has_treasury": False,
        "diversification_status": "non-diversified",
    },
    "TR2B1": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.15,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "TR2B2": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.15,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "TR2B3": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.15,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
    "TR2B4": {
        "option_roll_tenor": "weekly",
        "overwrite": 0.15,
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": "single_stock",
        "has_flex_option": True,
        "flex_option_type": "index",
        "has_otc": False,
        "has_treasury": False,
        "overlap_benchmark_ticker": "SPY",
        "diversification_status": "diversified",
    },
}


ALL_FUNDS = set(FUND_DEFINITIONS.keys())

# Cohorts populated at startup from reconciliation.v_fund_metadata.
ETF_FUNDS: set[str] = set()
CLOSED_END_FUNDS: set[str] = set()
PRIVATE_FUNDS: set[str] = set()
VIT_AND_MUTUAL_FUNDS: set[str] = set()

_STRATEGY_TO_COHORT = {
    "TIF": ETF_FUNDS,
    "CEF": CLOSED_END_FUNDS,
    "PF":  PRIVATE_FUNDS,
    "VIT": VIT_AND_MUTUAL_FUNDS,
}


def load_cohorts_from_db(session) -> None:
    """Populate cohort sets from reconciliation.v_fund_metadata.

    Call once at startup, after the DB session is open and before any
    run_configuration_batch() invocation.
    """
    df = pd.read_sql(
        text("SELECT fund_ticker, eod_report_strategy "
             "FROM reconciliation.v_fund_metadata"),
        session.bind,
    )
    valid_tickers = set(FUND_DEFINITIONS.keys())

    for target in _STRATEGY_TO_COHORT.values():
        target.clear()
    for _, row in df.iterrows():
        target = _STRATEGY_TO_COHORT.get(row["eod_report_strategy"])
        if target is not None and row["fund_ticker"] in valid_tickers:
            target.add(row["fund_ticker"])



DIVERSIFIED_FUNDS = {
    fund
    for fund, payload in FUND_DEFINITIONS.items()
    if payload.get("diversification_status") == "diversified"
}

NON_DIVERSIFIED_FUNDS = {
    fund
    for fund, payload in FUND_DEFINITIONS.items()
    if payload.get("diversification_status") == "non-diversified"
}

LISTED_INDEX_OPTION_FUNDS = {
    fund
    for fund, payload in FUND_DEFINITIONS.items()
    if payload.get("listed_option_type") == "index"
}

LISTED_SINGLE_STOCK_OPTION_FUNDS = {
    fund
    for fund, payload in FUND_DEFINITIONS.items()
    if payload.get("listed_option_type") == "single_stock"
}

INDEX_FLEX_FUNDS = {
    fund
    for fund, payload in FUND_DEFINITIONS.items()
    if bool(payload.get("has_flex_option"))
    and (payload.get("flex_option_type") or "").lower() == "index"
}

SINGLE_STOCK_FLEX_FUNDS = {
    fund
    for fund, payload in FUND_DEFINITIONS.items()
    if bool(payload.get("has_flex_option"))
    and (payload.get("flex_option_type") or "").lower() == "single_stock"
}