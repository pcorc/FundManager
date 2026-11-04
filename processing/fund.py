"""Fund, FundData, FundSnapshot, FundHoldings — the in-memory model.

Also exposes the fund-registry factory that replaces the old FundRegistry/FundClass
classes: a registry is just a `Dict[str, Fund]`, built once at startup via
:func:`build_fund_registry`.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Sequence, Set

import pandas as pd

from config.constants import VEHICLE_CLOSED_END, VEHICLE_ETF, VEHICLE_PRIVATE
from config.fund_definitions import FUND_DEFINITIONS, vehicle_for_strategy


FundRegistry = Dict[str, "Fund"]


_NUMERIC_COLUMNS = {
    "equity": [
        "equity_market_value", "market_value", "net_market_value",
        "quantity", "nav_shares", "iiv_shares", "shares",
        "price", "EQY_SH_OUT_million",
    ],
    "options": [
        "option_market_value", "market_value", "net_market_value",
        "option_delta_adjusted_notional", "delta_adjusted_notional",
        "quantity", "nav_shares_option", "price", "delta",
    ],
    "flex_options": [
        "flex_option_market_value", "option_market_value",
        "market_value", "net_market_value",
        "flex_option_delta_adjusted_notional",
        "option_delta_adjusted_notional", "delta_adjusted_notional",
        "quantity", "nav_shares_option", "price", "delta",
    ],
    "treasury": [
        "treasury_market_value", "market_value", "net_market_value",
        "quantity", "price", "face_value",
    ],
}

_TICKER_COLUMNS = {"equity": "eqyticker", "options": "optticker", "flex_options": "optticker", "treasury": "ticker"}


def _coerce_numeric_columns(df: Optional[pd.DataFrame], holding_type: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    df = df.copy()
    for col in _NUMERIC_COLUMNS.get(holding_type, []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _filter_frame_by_fund(frame: pd.DataFrame, fund_name: Optional[str]) -> pd.DataFrame:
    if frame.empty or not fund_name or "fund" not in frame.columns:
        return frame
    return frame[frame["fund"] == fund_name]


def _frame_value_sum(
    frame: pd.DataFrame, columns: Sequence[str], fund_name: Optional[str]
) -> Optional[float]:
    filtered = _filter_frame_by_fund(frame, fund_name)
    if filtered.empty:
        return None
    for column in columns:
        if column in filtered.columns:
            series = pd.to_numeric(filtered[column], errors="coerce").dropna()
            if not series.empty:
                return float(series.sum())
    return None


def _price_quantity_sum(frame: pd.DataFrame, fund_name: Optional[str], multiplier: float) -> Optional[float]:
    if frame.empty or not {"price", "quantity"}.issubset(frame.columns):
        return None
    filtered = _filter_frame_by_fund(frame, fund_name)
    if filtered.empty:
        return None
    price = pd.to_numeric(filtered["price"], errors="coerce").fillna(0.0)
    quantity = pd.to_numeric(filtered["quantity"], errors="coerce").fillna(0.0)
    return float((price * quantity * multiplier).sum())


def _asset_class_value(
    vest_frame: pd.DataFrame,
    custodian_frame: pd.DataFrame,
    value_columns: Sequence[str],
    fund_name: Optional[str],
    multiplier: Optional[float] = 1.0,
) -> float:
    """First non-empty value column per frame (Vest first); multiplier=None disables the price*qty fallback."""
    for frame in (vest_frame, custodian_frame):
        value = _frame_value_sum(frame, value_columns, fund_name)
        if value is None and multiplier is not None:
            value = _price_quantity_sum(frame, fund_name, multiplier)
        if value is not None:
            return value
    return 0.0


def _frame_or_empty(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if df is not None else pd.DataFrame()


class FundHoldings:
    """Holdings for a single source (Vest, custodian, or index)."""

    def __init__(
        self,
        *,
        equity: Optional[pd.DataFrame] = None,
        options: Optional[pd.DataFrame] = None,
        flex_options: Optional[pd.DataFrame] = None,
        treasury: Optional[pd.DataFrame] = None,
    ) -> None:
        self.equity = _coerce_numeric_columns(equity, "equity")
        self.options = _coerce_numeric_columns(options, "options")
        self.flex_options = _coerce_numeric_columns(flex_options, "flex_options")
        self.treasury = _coerce_numeric_columns(treasury, "treasury")


class FundSnapshot:
    """All holdings + reported scalars for a single point in time."""

    def __init__(
        self,
        *,
        vest: Optional[FundHoldings] = None,
        custodian: Optional[FundHoldings] = None,
        index: Optional[FundHoldings] = None,
        cash: float = 0.0,
        nav: float = 0.0,
        tna: float = 0.0,
        ta: float = 0.0,
        expenses: float = 0.0,
        shares_outstanding: float = 0.0,
        overlap: Optional[pd.DataFrame] = None,
        fund_name: Optional[str] = None,
        equity_trades: Optional[pd.DataFrame] = None,
        cr_rd_data: Optional[pd.DataFrame] = None,
        option_trades: Optional[pd.DataFrame] = None,
        flex_option_trades: Optional[pd.DataFrame] = None,
    ) -> None:
        self.vest = vest or FundHoldings()
        self.custodian = custodian or FundHoldings()
        self.index = index or FundHoldings()

        self.cash = float(cash)
        self.nav = float(nav)
        self.ta = float(ta)
        self.tna = float(tna)
        self.expenses = float(expenses)
        self.shares_outstanding = float(shares_outstanding)

        self.overlap = _frame_or_empty(overlap)
        self.fund_name = fund_name
        self.equity_trades = _frame_or_empty(equity_trades)
        self.cr_rd_data = _frame_or_empty(cr_rd_data)
        self.option_trades = _frame_or_empty(option_trades)
        self.flex_option_trades = _frame_or_empty(flex_option_trades)

        # Computed totals — prefer Vest, fall back to custodian. Computed once.
        self.total_equity_value = self._total("equity", ["equity_market_value", "market_value", "net_market_value"])
        self.total_option_value = self._total(
            "options", ["option_market_value", "market_value", "net_market_value"], 100.0
        )
        self.total_flex_option_value = self._total(
            "flex_options",
            ["flex_option_market_value", "option_market_value", "market_value", "net_market_value"],
            100.0,
        )
        self.total_treasury_value = self._total(
            "treasury", ["treasury_market_value", "market_value", "net_market_value"]
        )
        self.total_all_options_value = self.total_option_value + self.total_flex_option_value
        self.total_option_delta_adjusted_notional = self._total(
            "options", ["option_delta_adjusted_notional", "delta_adjusted_notional"], None
        ) + self._total(
            "flex_options",
            ["flex_option_delta_adjusted_notional", "option_delta_adjusted_notional", "delta_adjusted_notional"],
            None,
        )

    def _total(self, asset_class: str, columns: Sequence[str], multiplier: Optional[float] = 1.0) -> float:
        return _asset_class_value(
            getattr(self.vest, asset_class), getattr(self.custodian, asset_class),
            columns, self.fund_name, multiplier,
        )


class FundData:
    """Current (T) and previous (T-1) snapshots plus loose loader extras."""

    def __init__(
        self,
        *,
        current: Optional[FundSnapshot] = None,
        previous: Optional[FundSnapshot] = None,
    ) -> None:
        self.current = current or FundSnapshot()
        self.previous = previous or FundSnapshot()
        # Loader-supplied extras consumed by services:
        self.price_breaks: Dict[str, pd.DataFrame] = {}
        self.distributions: Optional[pd.DataFrame] = None
        self.flows: Optional[pd.DataFrame] = None
        self.other: float = 0.0


def _str_or_none(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _third_friday(month_start: pd.Timestamp) -> pd.Timestamp:
    month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
    return pd.date_range(month_start, month_end, freq="WOM-3FRI")[0]


class Fund:
    """Fund entity: configuration + loaded data + typed accessors.

    The FUND_DEFINITIONS policy flags (has_flex_option, …) are computed once in
    __init__. The v_fund_metadata fields (vehicle, index_ticker_join) are
    properties over ``config``, because the loader merges the metadata into
    ``config`` after the Fund is built.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config or {}
        self.data = FundData()

        c = self.config
        self.expense_ratio: float = float(c.get("expense_ratio") or 0.0)

        self.diversification_status: Optional[str] = _str_or_none(c.get("diversification_status"))
        self.has_listed_option: bool = bool(c.get("has_listed_option"))
        self.listed_option_type: Optional[str] = _str_or_none(c.get("listed_option_type"))
        self.has_flex_option: bool = bool(c.get("has_flex_option"))
        flex_type = c.get("flex_option_type")
        self.flex_option_type: Optional[str] = flex_type if isinstance(flex_type, str) and flex_type else None
        single_stock_flex = self.has_flex_option and self.flex_option_type == "single_stock"
        self.flex_option_pattern: str = "^2" if single_stock_flex else "SPX|XSP"
        self.option_roll_tenor: Optional[str] = c.get("option_roll_tenor")
        self.uses_index_flex: bool = (
                self.has_flex_option and (self.flex_option_type or "").lower() == "index"
        )

    # ------------------------------------------------------------------
    # v_fund_metadata-derived attributes (read from config on every access)
    # ------------------------------------------------------------------
    @property
    def index_ticker_join(self) -> Optional[str]:
        return _str_or_none(self.config.get("index_ticker_join"))

    @property
    def vehicle(self) -> Optional[str]:
        """Explicit ``vehicle_wrapper`` if configured, else derived from ``eod_report_strategy``."""
        vehicle = self.config.get("vehicle_wrapper")
        if isinstance(vehicle, str):
            return vehicle
        return vehicle_for_strategy(self.config.get("eod_report_strategy"))

    @property
    def is_private_fund(self) -> bool:
        return (self.vehicle or "").lower() == VEHICLE_PRIVATE

    @property
    def is_closed_end_fund(self) -> bool:
        return (self.vehicle or "").lower() == VEHICLE_CLOSED_END

    @property
    def is_etf(self) -> bool:
        return (self.vehicle or "").lower() == VEHICLE_ETF

    # ------------------------------------------------------------------
    # Holdings accessors
    # ------------------------------------------------------------------
    def _collect_tickers(self, asset_class: str, ticker_col: str, include_prior: bool) -> Set[str]:
        snapshots = [self.data.current, self.data.previous] if include_prior else [self.data.current]
        tickers: Set[str] = set()
        for snap in snapshots:
            for source in (snap.vest, snap.custodian):
                df = getattr(source, asset_class)
                if not df.empty and ticker_col in df.columns:
                    tickers.update(df[ticker_col].dropna().unique())
        return tickers

    def gather_all_tickers(self, asset_class: str, include_prior: bool = True) -> Set[str]:
        ticker_col = _TICKER_COLUMNS.get(asset_class)
        if ticker_col is None:
            return set()
        return self._collect_tickers(asset_class, ticker_col, include_prior)

    def gather_option_tickers(self, include_flex: bool, include_prior: bool = True) -> Set[str]:
        if include_flex and not self.has_flex_option:
            return set()
        return self._collect_tickers("flex_options" if include_flex else "options", "optticker", include_prior)

    def get_price_breaks(self, asset_class: str) -> pd.DataFrame:
        breaks = self.data.price_breaks.get(asset_class)
        return breaks if isinstance(breaks, pd.DataFrame) else pd.DataFrame()

    # ------------------------------------------------------------------
    # Option-roll settlement helpers
    # ------------------------------------------------------------------
    def is_option_settlement_date(self, check_date: str | date) -> bool:
        check_ts = pd.Timestamp(check_date).normalize()
        target_date = pd.bdate_range(end=check_ts, periods=2)[0]
        last_settlement = self._find_last_settlement_date(check_ts)
        if last_settlement is None:
            return False
        return target_date.normalize() == last_settlement.normalize()

    def _find_last_settlement_date(self, from_date: pd.Timestamp) -> Optional[pd.Timestamp]:
        tenor = (self.option_roll_tenor or "").lower()
        from_date = from_date.normalize()

        if tenor == "weekly":
            days_since_friday = (from_date.weekday() - 4) % 7
            return from_date - pd.Timedelta(days=days_since_friday)

        if tenor == "monthly":
            month_start = from_date.replace(day=1)
            third_friday = _third_friday(month_start)
            if third_friday <= from_date:
                return third_friday
            return _third_friday((month_start - pd.Timedelta(days=1)).replace(day=1))

        if tenor == "quarterly":
            search_date = from_date
            for _ in range(12):
                if search_date.month in (3, 6, 9, 12):
                    third_friday = _third_friday(search_date.replace(day=1))
                    if third_friday <= from_date:
                        return third_friday
                search_date = search_date.replace(day=1) - pd.Timedelta(days=1)
            return None

        # Default: most recent third Friday in current or prior two months.
        for months_back in range(3):
            check_month = from_date - pd.offsets.MonthBegin(months_back)
            third_friday = _third_friday(check_month.replace(day=1))
            if third_friday <= from_date:
                return third_friday
        return None

    def get_equity_dividends(self) -> float:
        equity = self.data.current.vest.equity
        if equity.empty or "dividend" not in equity.columns or "quantity" not in equity.columns:
            return 0.0
        return float((equity["dividend"] * equity["quantity"]).sum())


# ----------------------------------------------------------------------
# Registry factory (replaces the old FundRegistry class)
# ----------------------------------------------------------------------
def build_fund_registry(
    session: Any = None,
    base_cls: Any = None,
    *,
    definitions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> FundRegistry:
    """Build a {fund_name: Fund} registry from FUND_DEFINITIONS. ``session`` and
    ``base_cls`` are accepted for existing callers and unused."""
    definitions = definitions if definitions is not None else FUND_DEFINITIONS

    registry: FundRegistry = {}
    for fund_name, payload in definitions.items():
        config = dict(payload)
        config.setdefault("fund", fund_name)
        registry[fund_name] = Fund(name=fund_name, config=config)
    return registry


__all__ = [
    "Fund",
    "FundData",
    "FundSnapshot",
    "FundHoldings",
    "FundRegistry",
    "build_fund_registry",
]
