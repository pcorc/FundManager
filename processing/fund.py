"""Fund, FundData, FundSnapshot, FundHoldings — the in-memory model.

Also exposes the fund-registry factory that replaces the old FundRegistry/FundClass
classes: a registry is just a `Dict[str, Fund]`, built once at startup via
:func:`build_fund_registry`.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd

from config.fund_definitions import FUND_DEFINITIONS


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

# Keys that point at DB tables the loader will need to query for a fund.
_REQUIRED_TABLE_KEYS: tuple[str, ...] = (
    "custodian_equity_holdings",
    "custodian_option_holdings",
    "custodian_treasury_holdings",
    "custodian_navs",
    "cash_table",
    "vest_equity_holdings",
    "vest_options_holdings",
    "vest_treasury_holdings",
    "basket",
    "flows",
    "sg_custodian_holdings",
    "index_holdings",
    "option_custodian_assignment",
    "overlap_table",
)


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
    if frame.empty:
        return None
    filtered = _filter_frame_by_fund(frame, fund_name)
    if filtered.empty:
        return None
    for column in columns:
        if column in filtered.columns:
            series = pd.to_numeric(filtered[column], errors="coerce").dropna()
            if not series.empty:
                return float(series.sum())
    return None


def _price_quantity_sum(
    frame: pd.DataFrame, fund_name: Optional[str], multiplier: float = 1.0
) -> Optional[float]:
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
    multiplier: float = 1.0,
) -> float:
    for frame in (vest_frame, custodian_frame):
        value = _frame_value_sum(frame, value_columns, fund_name)
        if value is not None:
            return value
        fallback = _price_quantity_sum(frame, fund_name, multiplier=multiplier)
        if fallback is not None:
            return fallback
    return 0.0


def _sum_columns(
    vest_frame: pd.DataFrame,
    custodian_frame: pd.DataFrame,
    columns: Sequence[str],
    fund_name: Optional[str],
) -> float:
    for frame in (vest_frame, custodian_frame):
        value = _frame_value_sum(frame, columns, fund_name)
        if value is not None:
            return value
    return 0.0


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

    def copy(self) -> "FundHoldings":
        return FundHoldings(
            equity=self.equity.copy(),
            options=self.options.copy(),
            flex_options=self.flex_options.copy(),
            treasury=self.treasury.copy(),
        )


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
        flows: float = 0.0,
        basket: Optional[pd.DataFrame] = None,
        overlap: Optional[pd.DataFrame] = None,
        fund_name: Optional[str] = None,
        equity_trades: Optional[pd.DataFrame] = None,
        cr_rd_data: Optional[pd.DataFrame] = None,
        assignments: Optional[pd.DataFrame] = None,
        option_trades: Optional[pd.DataFrame] = None,
        flex_option_trades: Optional[pd.DataFrame] = None,
        treasury_trades: Optional[pd.DataFrame] = None,
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
        self.flows = float(flows)

        self.basket = basket if basket is not None else pd.DataFrame()
        self.overlap = overlap if overlap is not None else pd.DataFrame()
        self.fund_name = fund_name
        self.equity_trades = equity_trades if equity_trades is not None else pd.DataFrame()
        self.cr_rd_data = cr_rd_data if cr_rd_data is not None else pd.DataFrame()
        self.assignments = assignments if assignments is not None else pd.DataFrame()
        self.option_trades = option_trades if option_trades is not None else pd.DataFrame()
        self.flex_option_trades = flex_option_trades if flex_option_trades is not None else pd.DataFrame()
        self.treasury_trades = treasury_trades if treasury_trades is not None else pd.DataFrame()

        # Computed totals — prefer Vest, fall back to custodian. Computed once.
        self.total_equity_value = _asset_class_value(
            self.vest.equity, self.custodian.equity,
            ["equity_market_value", "market_value", "net_market_value"],
            fund_name,
        )
        self.total_option_value = _asset_class_value(
            self.vest.options, self.custodian.options,
            ["option_market_value", "market_value", "net_market_value"],
            fund_name, multiplier=100.0,
        )
        self.total_flex_option_value = _asset_class_value(
            self.vest.flex_options, self.custodian.flex_options,
            ["flex_option_market_value", "option_market_value", "market_value", "net_market_value"],
            fund_name, multiplier=100.0,
        )
        self.total_treasury_value = _asset_class_value(
            self.vest.treasury, self.custodian.treasury,
            ["treasury_market_value", "market_value", "net_market_value"],
            fund_name,
        )
        self.total_all_options_value = self.total_option_value + self.total_flex_option_value
        self.total_option_delta_adjusted_notional = _sum_columns(
            self.vest.options, self.custodian.options,
            ["option_delta_adjusted_notional", "delta_adjusted_notional"],
            fund_name,
        ) + _sum_columns(
            self.vest.flex_options, self.custodian.flex_options,
            ["flex_option_delta_adjusted_notional", "option_delta_adjusted_notional", "delta_adjusted_notional"],
            fund_name,
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
        self.assignments: Optional[pd.DataFrame] = None
        self.distributions: Optional[pd.DataFrame] = None
        self.flows: Optional[pd.DataFrame] = None
        self.other: float = 0.0


def _str_or_none(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _custodian_type(config: Dict[str, Any]) -> Optional[str]:
    for key in ("custodian_equity_holdings", "custodian_navs", "custodian_option_holdings"):
        table = config.get(key)
        if not isinstance(table, str) or not table:
            continue
        lowered = table.lower()
        if "bny" in lowered:
            return "bny"
        if "umb" in lowered:
            return "umb"
        if "socgen" in lowered or "sg" in lowered:
            return "socgen"
        if "ccva" in lowered:
            return "ccva"
    return None


def _required_tables(config: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in _REQUIRED_TABLE_KEYS:
        value = config.get(key)
        if isinstance(value, str) and value and value.upper() != "NULL":
            out.append(value)
    return out


def _flex_option_pattern(has_flex: bool, flex_type: Optional[str]) -> str:
    if not has_flex:
        return "SPX|XSP"
    return "^2" if flex_type == "single_stock" else "SPX|XSP"


class Fund:
    """Fund entity: configuration + loaded data + typed accessors.

    All config-derived flags (is_etf, has_flex_option, custodian_type, …) are
    computed once in __init__ and stored as plain attributes — there are no
    @property descriptors, since config is fixed for the lifetime of a Fund.
    """

    def __init__(self, name: str, config: Dict[str, Any], base_cls: Any = None):
        self.name = name
        self.config = config or {}
        self.base_cls = base_cls
        self.data = FundData()

        c = self.config
        self.expense_ratio: float = float(c.get("expense_ratio") or 0.0)
        self.index_identifier: Optional[str] = _str_or_none(c.get("index_identifier"))
        self.index_ticker_join: Optional[str] = c.get("index_ticker_join")

        vehicle = c.get("vehicle_wrapper")
        self.vehicle: Optional[str] = vehicle if isinstance(vehicle, str) else None
        vehicle_lower = (self.vehicle or "").lower()
        self.is_private_fund: bool = vehicle_lower == "private_fund"
        self.is_closed_end_fund: bool = vehicle_lower == "closed_end_fund"
        self.is_etf: bool = vehicle_lower == "etf"

        self.diversification_status: Optional[str] = _str_or_none(c.get("diversification_status"))
        status_lower = (self.diversification_status or "").lower()
        self.is_diversified: bool = status_lower == "diversified"
        self.is_non_diversified: bool = status_lower == "non-diversified"

        self.has_listed_option: bool = bool(c.get("has_listed_option"))
        self.listed_option_type: Optional[str] = _str_or_none(c.get("listed_option_type"))
        self.has_flex_option: bool = bool(c.get("has_flex_option"))
        flex_type = c.get("flex_option_type")
        self.flex_option_type: Optional[str] = flex_type if isinstance(flex_type, str) and flex_type else None
        self.flex_option_pattern: str = _flex_option_pattern(self.has_flex_option, self.flex_option_type)
        self.option_roll_tenor: Optional[str] = c.get("option_roll_tenor")
        self.has_otc: bool = bool(c.get("has_otc"))
        self.has_treasury: bool = bool(c.get("has_treasury"))

        usage = c.get("index_flex_usage")
        self.uses_index_flex: bool = (
            isinstance(usage, str) and usage.lower() in ("true", "yes", "enabled", "1")
        )

        self.custodian_type: Optional[str] = _custodian_type(c)
        self.required_tables: List[str] = _required_tables(c)

    # ------------------------------------------------------------------
    # Holdings accessors
    # ------------------------------------------------------------------
    def gather_all_tickers(self, asset_class: str, include_prior: bool = True) -> Set[str]:
        ticker_col = {
            "equity": "eqyticker",
            "options": "optticker",
            "flex_options": "optticker",
            "treasury": "ticker",
        }.get(asset_class)
        if ticker_col is None:
            return set()

        snapshots = [self.data.current]
        if include_prior:
            snapshots.append(self.data.previous)

        tickers: Set[str] = set()
        for snap in snapshots:
            for source in (snap.vest, snap.custodian):
                df = getattr(source, asset_class)
                if not df.empty and ticker_col in df.columns:
                    tickers.update(df[ticker_col].dropna().unique())
        return tickers

    def gather_option_tickers(self, include_flex: bool, include_prior: bool = True) -> Set[str]:
        if not self.has_flex_option:
            return set() if include_flex else self.gather_all_tickers("options", include_prior)

        snapshots = [self.data.current]
        if include_prior:
            snapshots.append(self.data.previous)

        tickers: Set[str] = set()
        for snap in snapshots:
            for source in (snap.vest, snap.custodian):
                df = source.options
                if df.empty or "optticker" not in df.columns:
                    continue
                mask = df["optticker"].str.contains(self.flex_option_pattern, na=False, regex=True)
                filtered = df[mask if include_flex else ~mask]
                tickers.update(filtered["optticker"].dropna().unique())
        return tickers

    def get_price_breaks(self, asset_class: str) -> pd.DataFrame:
        breaks = self.data.price_breaks.get(asset_class)
        return breaks if isinstance(breaks, pd.DataFrame) else pd.DataFrame()

    def get_assignments(self, filter_date: Optional[date | str] = None) -> pd.DataFrame:
        assignments = self.data.assignments
        if assignments is None or assignments.empty:
            return pd.DataFrame()
        if filter_date is None:
            return assignments.copy()

        date_cols = [c for c in assignments.columns if "date" in str(c).lower()]
        if not date_cols:
            return assignments.copy()

        df = assignments.copy()
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        target = pd.Timestamp(filter_date).normalize()
        return df[df[date_cols[0]].dt.normalize() == target]

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
            third_fridays = pd.date_range(month_start, from_date, freq="WOM-3FRI")
            if len(third_fridays) > 0 and third_fridays[-1] <= from_date:
                return third_fridays[-1]
            prev_month_start = (month_start - pd.Timedelta(days=1)).replace(day=1)
            prev_month_end = month_start - pd.Timedelta(days=1)
            third_fridays = pd.date_range(prev_month_start, prev_month_end, freq="WOM-3FRI")
            return third_fridays[-1] if len(third_fridays) > 0 else None

        if tenor == "quarterly":
            search_date = from_date
            for _ in range(12):
                if search_date.month in (3, 6, 9, 12):
                    month_start = search_date.replace(day=1)
                    month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
                    third_fridays = pd.date_range(month_start, month_end, freq="WOM-3FRI")
                    if len(third_fridays) > 0 and third_fridays[0] <= from_date:
                        return third_fridays[0]
                search_date = search_date.replace(day=1) - pd.Timedelta(days=1)
            return None

        # Default: most recent third Friday in current or prior two months.
        for months_back in range(3):
            check_month = from_date - pd.offsets.MonthBegin(months_back)
            month_start = check_month.replace(day=1)
            month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
            third_fridays = pd.date_range(month_start, month_end, freq="WOM-3FRI")
            if len(third_fridays) > 0 and third_fridays[0] <= from_date:
                return third_fridays[0]
        return None

    def get_equity_dividends(self) -> float:
        equity = self.data.current.vest.equity
        if equity.empty or "dividend" not in equity.columns or "quantity" not in equity.columns:
            return 0.0
        return float((equity["dividend"] * equity["quantity"]).sum())


# ----------------------------------------------------------------------
# Registry factory (replaces the old FundRegistry class)
# ----------------------------------------------------------------------
def _load_account_numbers(session, base_cls) -> Dict[str, Dict[str, Any]]:
    """Load per-fund custodian account numbers from the accounts_mapping schema."""
    account_numbers_tbl = getattr(base_cls.classes, "account_numbers", None)
    if account_numbers_tbl is None:
        return {}

    query = session.query(
        account_numbers_tbl.fund,
        account_numbers_tbl.account_type,
        account_numbers_tbl.service_provider,
        account_numbers_tbl.account_number,
    )
    df = pd.read_sql(query.statement, session.bind)

    mapping: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        fund_key = (row.get("fund") or "").strip() if isinstance(row.get("fund"), str) else ""
        if not fund_key:
            continue
        account_number = row.get("account_number")
        if pd.isna(account_number):
            continue
        account_number = str(account_number).strip()
        if not account_number:
            continue

        account_type = row.get("account_type")
        service_provider = row.get("service_provider")
        account_type_key = account_type.strip().lower() if isinstance(account_type, str) else None
        provider_key = service_provider.strip().lower() if isinstance(service_provider, str) else None

        fund_numbers = mapping.setdefault(fund_key, {})

        if provider_key == "sg" and account_type_key != "collateral":
            accounts = fund_numbers.setdefault("sg", [])
            if account_number not in accounts:
                accounts.append(account_number)
            continue

        key = account_type_key or provider_key or "other"
        existing = fund_numbers.get(key)
        if existing is None:
            fund_numbers[key] = account_number
        elif isinstance(existing, list):
            if account_number not in existing:
                existing.append(account_number)
        elif existing != account_number:
            fund_numbers[key] = [existing, account_number]

    return mapping


def _derive_custodian_account_number(numbers: Dict[str, Any]) -> Optional[str]:
    if not numbers:
        return None
    for key in ("account_number_custodian", "custodian", "primary", "account"):
        value = numbers.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            return value[0]
    sg = numbers.get("sg")
    if isinstance(sg, list) and sg:
        return sg[0]
    for key, value in numbers.items():
        if key == "collateral":
            continue
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            return value[0]
    return None


def build_fund_registry(
    session: Any = None,
    base_cls: Any = None,
    *,
    definitions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> FundRegistry:
    """Build a {fund_name: Fund} registry from FUND_DEFINITIONS, enriched with
    account numbers when a DB session is provided."""
    definitions = definitions if definitions is not None else FUND_DEFINITIONS

    account_numbers: Dict[str, Dict[str, Any]] = {}
    if session is not None and base_cls is not None:
        account_numbers = _load_account_numbers(session, base_cls)

    registry: FundRegistry = {}
    for fund_name, payload in definitions.items():
        config = dict(payload)
        config.setdefault("fund", fund_name)
        numbers = account_numbers.get(fund_name)
        if numbers:
            config["account_numbers"] = numbers
            config.setdefault("account_number_custodian", _derive_custodian_account_number(numbers))
        registry[fund_name] = Fund(name=fund_name, config=config, base_cls=base_cls)
    return registry


__all__ = [
    "Fund",
    "FundData",
    "FundSnapshot",
    "FundHoldings",
    "FundRegistry",
    "build_fund_registry",
]
