"""Static fund metadata consumed by :func:`processing.fund.build_fund_registry`.

Only policy flags with no equivalent column in the reconciliation.* views live
here; expense ratio, benchmark, vehicle, custodian etc. come from v_fund_metadata.
"""
from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from sqlalchemy import text


_ABSENT = object()  # leave the key out of the fund's definition entirely

_COLUMNS = ("option_roll_tenor", "overwrite", "listed_option_type", "flex_option_type",
            "has_treasury", "diversification_status")
_ROWS = {
    "DOGG":   ("weekly",    0.08, "single_stock", "single_stock", True,  "non-diversified"),
    "FDND":   ("weekly",    0.08, "index",        None,           False, "non-diversified"),
    "FGSI":   ("weekly",    0.08, "index",        None,           False, "non-diversified"),
    "FTCSH":  ("quarterly", 0.0,  "index",        None,           False, "non-diversified"),
    "FTMIX":  ("monthly",   0.0,  "index",        None,           False, _ABSENT),
    "HE3B1":  ("weekly",    0.12, "single_stock", "index",        False, "diversified"),
    "HE3B2":  ("weekly",    0.12, "single_stock", "index",        False, "diversified"),
    "HE3B3":  ("weekly",    0.12, "index",        "index",        False, "diversified"),
    "KNG":    ("monthly",   0.08, "single_stock", None,           False, "diversified"),
    "KNGIX":  ("weekly",    0.1,  "single_stock", None,           False, "diversified"),
    "P20127": ("weekly",    0.15, "single_stock", "index",        False, "diversified"),
    "P21026": ("weekly",    0.15, "single_stock", "index",        False, "diversified"),
    "P2726":  ("weekly",    0.15, "single_stock", "index",        False, "diversified"),
    "P30128": ("weekly",    0.15, "single_stock", "index",        False, "diversified"),
    "P31027": ("weekly",    0.12, "single_stock", "index",        False, "diversified"),
    "P3727":  ("weekly",    0.12, "single_stock", "index",        False, "diversified"),
    "PD227":  ("weekly",    0.12, "single_stock", None,           False, "excluded"),
    "PF227":  ("weekly",    0.12, "single_stock", None,           False, "excluded"),
    "PF27V1": ("weekly",    0.16, "single_stock", None,           False, "excluded"),
    "R21126": ("weekly",    0.12, "index",        "index",        False, "diversified"),
    "RDVI":   ("weekly",    0.08, "index",        None,           False, "non-diversified"),
    "SDVD":   ("weekly",    0.08, "index",        None,           False, "non-diversified"),
    "TDVI":   ("weekly",    0.08, "index",        None,           False, "non-diversified"),
    "TR2A1":  ("weekly",    0.15, "single_stock", "index",        False, "diversified"),
    "TR2B1":  ("weekly",    0.15, "single_stock", "index",        False, "diversified"),
    "TR2B2":  ("weekly",    0.15, "single_stock", "index",        False, "diversified"),
    "TR2B3":  ("weekly",    0.15, "single_stock", "index",        False, "diversified"),
    "TR2B4":  ("weekly",    0.15, "single_stock", "index",        False, "diversified"),
}


def _definition(row: tuple) -> Dict[str, Any]:
    values = dict(zip(_COLUMNS, row))
    flex_type = values["flex_option_type"]
    payload = {
        "option_roll_tenor": values["option_roll_tenor"],
        "overwrite": values["overwrite"],
        "has_equity": True,
        "has_listed_option": True,
        "listed_option_type": values["listed_option_type"],
        "has_flex_option": flex_type is not None,
        "flex_option_type": flex_type,
        "has_otc": False,
        "has_treasury": values["has_treasury"],
    }
    if flex_type == "index":
        payload["overlap_benchmark_ticker"] = "SPY"
    if values["diversification_status"] is not _ABSENT:
        payload["diversification_status"] = values["diversification_status"]
    return payload


FUND_DEFINITIONS: Dict[str, Dict[str, Any]] = {fund: _definition(row) for fund, row in _ROWS.items()}


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


INDEX_FLEX_FUNDS = {
    fund
    for fund, payload in FUND_DEFINITIONS.items()
    if bool(payload.get("has_flex_option"))
    and (payload.get("flex_option_type") or "").lower() == "index"
}
