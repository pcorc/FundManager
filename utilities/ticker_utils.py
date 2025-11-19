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
    """
    Normalize holdings for a fund using its static configuration.

    Only the asset types flagged in :mod:`config.fund_definitions` are
    processed. The function returns a shallow copy of ``fund_data`` with the
    normalised holdings written back to their existing keys (``vest_*`` and
    ``custodian_*``).
    """

    normalized = {**fund_data}
    definition = fund_definition or FUND_DEFINITIONS.get(fund_name, {})

    def _normalize_pair(oms_key: str, cust_key: str, func) -> None:
        oms_df = normalized.get(oms_key, pd.DataFrame()).copy()
        cust_df = normalized.get(cust_key, pd.DataFrame()).copy()

        if oms_df.empty and cust_df.empty:
            normalized[oms_key] = oms_df
            normalized[cust_key] = cust_df
            return

        normalized[oms_key], normalized[cust_key] = func(oms_df, cust_df, logger)

    if definition.get("has_equity", True):
        _normalize_pair("vest_equity", "custodian_equity", normalize_equity_pair)
        _normalize_pair(
            "vest_equity_t1", "custodian_equity_t1", normalize_equity_pair
        )

    if definition.get("has_listed_option", False) or definition.get("has_flex_option", False):
        _normalize_pair("vest_option", "custodian_option", normalize_option_pair)
        _normalize_pair("vest_option_t1", "custodian_option_t1", normalize_option_pair)

    if definition.get("has_treasury", False):
        _normalize_pair("vest_treasury", "custodian_treasury", normalize_treasury_pair)
        _normalize_pair(
            "vest_treasury_t1", "custodian_treasury_t1", normalize_treasury_pair
        )

    if logger:
        logger.info("Holdings normalization complete for fund %s", fund_name)

    return normalized

def normalize_equity_pair(
    df_oms: pd.DataFrame, df_cust: pd.DataFrame, logger=None, debug_mode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise equity tickers across OMS and custodian datasets."""

    df_oms = df_oms.copy()
    df_cust = df_cust.copy()

    if "equity_ticker" not in df_oms.columns:
        if "ticker" in df_oms.columns:
            df_oms["equity_ticker"] = df_oms["ticker"]
        elif logger:
            logger.warning("OMS equity holdings missing equity_ticker column")

    if "equity_ticker" not in df_cust.columns:
        source = None
        for candidate in ("equity_ticker", "ticker", "security_tkr"):
            if candidate in df_cust.columns:
                source = candidate
                break
        if source:
            df_cust["equity_ticker"] = df_cust[source]
        elif "sedol" in df_cust.columns and "sedol" in df_oms.columns:
            mapping = (
                df_oms.dropna(subset=["sedol"])
                .set_index("sedol")
                .get("equity_ticker")
            )
            if mapping is not None:
                df_cust["equity_ticker"] = df_cust["sedol"].map(mapping)
                if logger and debug_mode:
                    mapped = df_cust["equity_ticker"].notna().sum()
                    logger.debug("Mapped %s custodian tickers via SEDOL", mapped)
        elif logger:
            logger.warning("Custodian equity holdings missing recognizable ticker column")

    return df_oms, df_cust


def normalize_option_pair(
    df_oms: pd.DataFrame, df_cust: pd.DataFrame, logger=None, debug_mode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise option identifiers to use ``optticker`` consistently."""

    df_oms = df_oms.copy()
    df_cust = df_cust.copy()

    def ensure_optticker(df: pd.DataFrame, side: str) -> None:
        if "optticker" in df.columns:
            return
        if "occ_symbol" in df.columns:
            df["optticker"] = df["occ_symbol"]
        elif "ticker" in df.columns:
            df["optticker"] = df["ticker"]
        elif logger:
            logger.warning("%s options missing optticker column", side)

    ensure_optticker(df_oms, "OMS")
    ensure_optticker(df_cust, "Custodian")

    return df_oms, df_cust


def normalize_treasury_pair(
    df_oms: pd.DataFrame, df_cust: pd.DataFrame, logger=None, debug_mode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure treasury datasets carry a ``cusip`` identifier when possible."""

    df_oms = df_oms.copy()
    df_cust = df_cust.copy()

    def ensure_cusip(df: pd.DataFrame, side: str) -> None:
        if "cusip" in df.columns:
            return
        for candidate in ("ticker", "security_id", "isin"):
            if candidate in df.columns:
                df["cusip"] = df[candidate]
                return
        if logger:
            logger.warning("%s treasury holdings missing CUSIP-like identifier", side)

    ensure_cusip(df_oms, "OMS")
    ensure_cusip(df_cust, "Custodian")

    return df_oms, df_cust


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

