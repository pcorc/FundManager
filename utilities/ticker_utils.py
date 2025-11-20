import pandas as pd
import re
from typing import Tuple, Dict, Any, Optional

from config.fund_definitions import FUND_DEFINITIONS


def normalize_all_holdings(
        fund_name: str,
        fund_data: Dict[str, Any],
        *,
        fund_definition: Optional[Dict[str, Any]] = None,
        logger=None,
) -> Dict[str, Any]:
    """Optimized version - processes each key directly."""
    definition = fund_definition or FUND_DEFINITIONS.get(fund_name, {})
    result = {}

    asset_processors = []
    if definition.get("has_equity", True):
        asset_processors.extend([
            ("vest_equity", "custodian_equity", _normalize_equity),
            ("vest_equity_t1", "custodian_equity_t1", _normalize_equity),
        ])

    if definition.get("has_listed_option", False) or definition.get("has_flex_option", False):
        asset_processors.extend([
            ("vest_option", "custodian_option", _normalize_options),
            ("vest_option_t1", "custodian_option_t1", _normalize_options),
        ])

    if definition.get("has_treasury", False):
        asset_processors.extend([
            ("vest_treasury", "custodian_treasury", _normalize_treasury),
            ("vest_treasury_t1", "custodian_treasury_t1", _normalize_treasury),
        ])

    # Process all pairs in one loop
    for oms_key, cust_key, processor in asset_processors:
        oms_df = fund_data.get(oms_key, pd.DataFrame())
        cust_df = fund_data.get(cust_key, pd.DataFrame())

        if oms_df.empty and cust_df.empty:
            result[oms_key] = oms_df
            result[cust_key] = cust_df
        else:
            result[oms_key], result[cust_key] = processor(oms_df, cust_df, logger)

    # Copy any unprocessed keys
    for key, value in fund_data.items():
        if key not in result:
            result[key] = value

    return result


def _normalize_equity(
        df_oms: pd.DataFrame, df_cust: pd.DataFrame, logger=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise equity tickers across OMS and custodian datasets."""

    if "sedol" in df_cust.columns and not df_oms.empty and "sedol" in df_oms.columns:
        mapping = df_oms.set_index("sedol")["eqyticker"].dropna().to_dict()
        df_cust = df_cust.assign(eqyticker=df_cust["sedol"].map(mapping))
    elif logger:
        logger.warning("Custodian equity holdings missing recognizable ticker column")

    return df_oms, df_cust


def _normalize_options(
        df_oms: pd.DataFrame, df_cust: pd.DataFrame, logger=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise option identifiers to use ``optticker`` consistently."""

    def _ensure_optticker(df: pd.DataFrame, side: str) -> pd.DataFrame:
        if df.empty or "optticker" in df.columns:
            return df
        if "occ_symbol" in df.columns:
            return df.assign(optticker=df["occ_symbol"])
        elif "ticker" in df.columns:
            return df.assign(optticker=df["ticker"])
        elif logger:
            logger.warning("%s options missing optticker column", side)
        return df

    return (
        _ensure_optticker(df_oms, "OMS"),
        _ensure_optticker(df_cust, "Custodian")
    )


def _normalize_treasury(
        df_oms: pd.DataFrame, df_cust: pd.DataFrame, logger=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure treasury datasets carry a ``cusip`` identifier when possible."""

    def _ensure_cusip(df: pd.DataFrame, side: str) -> pd.DataFrame:
        if df.empty or "cusip" in df.columns:
            return df
        for candidate in ("ticker", "security_id", "isin"):
            if candidate in df.columns:
                return df.assign(cusip=df[candidate])
        if logger:
            logger.warning("%s treasury holdings missing CUSIP-like identifier", side)
        return df

    return (
        _ensure_cusip(df_oms, "OMS"),
        _ensure_cusip(df_cust, "Custodian")
    )

def normalize_option_ticker(ticker: str, logger=None, verbose: bool = False) -> str:
    """Normalise raw option tickers into OCC standard format."""

    raw = ticker.strip().upper()
    raw = raw.replace("BRK/B", "BRKB").replace("BRK.B", "BRKB").replace("BRK B", "BRKB")

    if raw.startswith("SPX0"):
        raw = "SPX" + raw[4:]
        if verbose and logger:
            logger.debug("SPX0 edge case handled: converting to SPX")

    if re.match(r"^[A-Z0-9]{1,6} \d{6}[CP]\d{8}$", raw):
        if verbose and logger:
            logger.debug("Already in OCC format: %s", raw)
        return raw

    if re.match(r"^\d[A-Z0-9]{1,6} \d{6}[CP]\d{8}$", raw):
        result = raw[1:]
        if verbose and logger:
            logger.debug("SG format detected: %s -> %s", raw, result)
        return result

    raw = re.sub(r"\s", " ", raw)

    patterns = [
        r"^([A-Z0-9])\sUS\s(\d{1,2})/(\d{1,2})/(\d{2,4})\s([CP])\s*(\d\.?\d*)$",
        r"^([A-Z0-9])\s(\d{1,2})/(\d{1,2})/(\d{2,4})\s([CP])\s*(\d\.?\d*)$",
        r"^([A-Z0-9])(\d{1,2})/(\d{1,2})/(\d{2,4})\sUS\s([CP])\s*(\d\.?\d*)$",
        r"^([A-Z0-9])(\d{1,2})/(\d{1,2})/(\d{2,4})\s([CP])\s*(\d\.?\d*)$",
        r"^([A-Z0-9])\sUS\s(\d{1,2})/(\d{1,2})/(\d{2,4})\s([CP])\s*(\d\.?\d*)\sINDEX$",
        r"^([A-Z0-9])\s(\d{1,2})/(\d{1,2})/(\d{2,4})\s([CP])\s*(\d\.?\d*)\sINDEX$",
    ]

    for pattern in patterns:
        match = re.match(pattern, raw)
        if match:
            groups = match.groups()
            symbol, month, day, year, cp, strike = groups
            result = _format_occ_symbol(symbol, year, month, day, cp, strike)
            if verbose and logger:
                logger.debug("Matched pattern %s: %s -> %s", pattern, raw, result)
            return result

    if logger:
        logger.warning("Could not normalize option ticker to OCC format: %s", ticker)
    return raw


def _format_occ_symbol(symbol: str, year: str, month: str, day: str, cp: str, strike: str) -> str:
    """Format option components into OCC standard representation."""

    if symbol == "SPXW":
        symbol = "SPX"

    symbol_padded = symbol.ljust(6)

    if len(year) == 4:
        year = year[2:]
    year = year.zfill(2)
    month = month.zfill(2)
    day = day.zfill(2)

    date_str = f"{year}{month}{day}"

    strike_float = float(strike)
    strike_mills = int(round(strike_float * 1000))
    strike_str = f"{strike_mills:08d}"

    return f"{symbol_padded} {date_str}{cp}{strike_str}"

