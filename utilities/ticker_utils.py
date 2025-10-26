import pandas as pd
import re
from typing import Tuple, Dict, Any


def normalize_all_holdings(fund_data: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """
    Normalize all equity, option, and treasury holdings for a fund.

    This entry point harmonises tickers and identifiers for both OMS and
    custodian datasets across T and T-1 snapshots. The function operates on a
    dictionary of holdings DataFrames and returns the same dictionary with
    normalised copies alongside raw backups.
    """

    equity_df = fund_data.get("equity_holdings", pd.DataFrame()).copy()
    options_df = fund_data.get("options_holdings", pd.DataFrame()).copy()
    treasury_df = fund_data.get("treasury_holdings", pd.DataFrame()).copy()

    df_cust_equity = fund_data.get("custodian_equity_holdings", pd.DataFrame()).copy()
    df_cust_option = fund_data.get("custodian_option_holdings", pd.DataFrame()).copy()
    df_cust_treasury = fund_data.get("custodian_treasury_holdings", pd.DataFrame()).copy()

    equity_df_t1 = fund_data.get("t1_equity_holdings", pd.DataFrame()).copy()
    options_df_t1 = fund_data.get("t1_options_holdings", pd.DataFrame()).copy()
    treasury_df_t1 = fund_data.get("t1_treasury_holdings", pd.DataFrame()).copy()

    df_cust_equity_t1 = fund_data.get("t1_custodian_equity_holdings", pd.DataFrame()).copy()
    df_cust_option_t1 = fund_data.get("t1_custodian_option_holdings", pd.DataFrame()).copy()
    df_cust_treasury_t1 = fund_data.get("t1_custodian_treasury_holdings", pd.DataFrame()).copy()

    fund_data["equity_holdings_raw"] = equity_df.copy()
    fund_data["options_holdings_raw"] = options_df.copy()
    fund_data["treasury_holdings_raw"] = treasury_df.copy()

    fund_data["t1_equity_holdings_raw"] = equity_df_t1.copy()
    fund_data["t1_options_holdings_raw"] = options_df_t1.copy()
    fund_data["t1_treasury_holdings_raw"] = treasury_df_t1.copy()

    if not equity_df.empty:
        equity_df, df_cust_equity = normalize_equity_pair(equity_df, df_cust_equity, logger)

    if not options_df.empty or not df_cust_option.empty:
        options_df, df_cust_option = normalize_option_pair(options_df, df_cust_option, logger)
        for col in (
            "option_notional_value",
            "option_delta_adjusted_notional",
            "option_delta_adjusted_market_value",
            "option_market_value",
        ):
            if col not in options_df.columns:
                options_df[col] = 0.0

    if not treasury_df.empty or not df_cust_treasury.empty:
        treasury_df, df_cust_treasury = normalize_treasury_pair(
            treasury_df, df_cust_treasury, logger
        )

    if not equity_df_t1.empty or not df_cust_equity_t1.empty:
        equity_df_t1, df_cust_equity_t1 = normalize_equity_pair(
            equity_df_t1, df_cust_equity_t1, logger
        )

    if not options_df_t1.empty or not df_cust_option_t1.empty:
        if "optticker" not in options_df_t1.columns:
            if "ticker" in options_df_t1.columns:
                options_df_t1["optticker"] = options_df_t1["ticker"]
            elif "occ_symbol" in options_df_t1.columns:
                options_df_t1["optticker"] = options_df_t1["occ_symbol"]

        if not df_cust_option_t1.empty and "optticker" not in df_cust_option_t1.columns:
            if "occ_symbol" in df_cust_option_t1.columns:
                df_cust_option_t1["optticker"] = df_cust_option_t1["occ_symbol"]

        options_df_t1, df_cust_option_t1 = normalize_option_pair(
            options_df_t1, df_cust_option_t1, logger
        )

    if not treasury_df_t1.empty or not df_cust_treasury_t1.empty:
        treasury_df_t1, df_cust_treasury_t1 = normalize_treasury_pair(
            treasury_df_t1, df_cust_treasury_t1, logger
        )

    fund_data["equity_holdings"] = equity_df
    fund_data["options_holdings"] = options_df
    fund_data["treasury_holdings"] = treasury_df

    fund_data["custodian_equity_holdings"] = df_cust_equity
    fund_data["custodian_option_holdings"] = df_cust_option
    fund_data["custodian_treasury_holdings"] = df_cust_treasury

    fund_data["t1_equity_holdings"] = equity_df_t1
    fund_data["t1_options_holdings"] = options_df_t1
    fund_data["t1_treasury_holdings"] = treasury_df_t1

    fund_data["t1_custodian_equity_holdings"] = df_cust_equity_t1
    fund_data["t1_custodian_option_holdings"] = df_cust_option_t1
    fund_data["t1_custodian_treasury_holdings"] = df_cust_treasury_t1

    if logger:
        logger.info("Holdings normalization complete for fund")

    return fund_data


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

