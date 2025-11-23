from typing import Dict, Optional, Sequence
import pandas as pd


class FundHoldings:
    """Wrapper for holdings data for a specific source (e.g. Vest or custodian)."""

    def __init__(
        self,
        *,
        equity: Optional[pd.DataFrame] = None,
        options: Optional[pd.DataFrame] = None,
        flex_options: Optional[pd.DataFrame] = None,  # NEW
        treasury: Optional[pd.DataFrame] = None,
        cash: float = 0.0,
        nav: float = 0.0,
        tna: float = 0.0,
        ta: float = 0.0,
        expenses: float = 0.0,
        shares_outstanding: float = 0.0,
    ) -> None:
        self.equity = self._clean_holdings_dataframe(equity, holding_type='equity')
        self.options = self._clean_holdings_dataframe(options, holding_type='options')
        self.flex_options = self._clean_holdings_dataframe(flex_options, holding_type='flex_options')
        self.treasury = self._clean_holdings_dataframe(treasury, holding_type='treasury')

        self.cash = float(cash or 0.0)
        self.nav = float(nav or 0.0)
        self.tna = float(tna or 0.0)
        self.ta = float(ta or 0.0)
        self.expenses = float(expenses or 0.0)
        self.shares_outstanding = float(shares_outstanding or 0.0)

    # Backward-compatible aliases
    @property
    def reported_cash(self) -> float:
        return self.cash

    @property
    def reported_nav(self) -> float:
        return self.nav

    @property
    def reported_tna(self) -> float:
        return self.tna

    @property
    def reported_ta(self) -> float:
        return self.ta

    @property
    def reported_expenses(self) -> float:
        return self.expenses

    @property
    def reported_shares_outstanding(self) -> float:
        return self.shares_outstanding

    def _clean_holdings_dataframe(self, df: Optional[pd.DataFrame], holding_type: str) -> pd.DataFrame:
        """Clean and standardize holdings DataFrame with proper numeric types."""
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()

        if df.empty:
            return df

        # Make a copy to avoid modifying the original
        df = df.copy()

        # Define numeric columns by holding type
        numeric_columns_map = {
            'equity': [
                'equity_market_value', 'market_value', 'net_market_value',
                'quantity', 'nav_shares', 'iiv_shares', 'shares',
                'price', 'EQY_SH_OUT_million'
            ],
            'options': [
                'option_market_value', 'market_value', 'net_market_value',
                'option_delta_adjusted_notional', 'delta_adjusted_notional',
                'quantity', 'nav_shares_option', 'price', 'delta'
            ],
            'flex_options': [
                'flex_option_market_value', 'option_market_value',
                'market_value', 'net_market_value',
                'flex_option_delta_adjusted_notional',
                'option_delta_adjusted_notional', 'delta_adjusted_notional',
                'quantity', 'nav_shares_option', 'price', 'delta'
            ],
            'treasury': [
                'treasury_market_value', 'market_value', 'net_market_value',
                'quantity', 'price', 'face_value'
            ]
        }

        # Get the relevant numeric columns for this holding type
        numeric_columns = numeric_columns_map.get(holding_type, [])

        # Convert relevant columns to numeric
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        return df

    def copy(self) -> "FundHoldings":
        return FundHoldings(
            equity=self.equity.copy() if isinstance(self.equity, pd.DataFrame) else pd.DataFrame(),
            options=self.options.copy() if isinstance(self.options, pd.DataFrame) else pd.DataFrame(),
            flex_options=self.flex_options.copy() if isinstance(self.flex_options, pd.DataFrame) else pd.DataFrame(),  # NEW
            treasury=self.treasury.copy() if isinstance(self.treasury, pd.DataFrame) else pd.DataFrame(),
            cash=self.cash,
            nav=self.nav,
            tna=self.tna,
            ta=self.ta,
            expenses=self.expenses,
            shares_outstanding=self.shares_outstanding,
        )

class FundMetrics:
    """Computed metrics from holdings data."""
    def __init__(self, snapshot: 'FundSnapshot'):
        self.snapshot = snapshot

    @property
    def total_equity_value(self) -> float:
        """Total equity market value computed from holdings."""
        return self._compute_equity_value()

    @property
    def total_option_value(self) -> float:
        """Total regular option market value computed from holdings."""
        return self._compute_option_value()

    @property
    def total_flex_option_value(self) -> float:
        """Total flex option market value computed from holdings."""
        return self._compute_flex_option_value()

    @property
    def total_treasury_value(self) -> float:
        """Total treasury market value computed from holdings."""
        return self._compute_treasury_value()

    @property
    def total_option_delta_adjusted_notional(self) -> float:
        """Total delta-adjusted notional for all options (regular + flex)."""
        regular = self._compute_option_delta_adjusted_notional()
        flex = self._compute_flex_option_delta_adjusted_notional()
        return regular + flex

    @property
    def total_holdings_value(self) -> float:
        """Total value of all holdings (equity + options + flex + treasury)."""
        return (
            self.total_equity_value +
            self.total_option_value +
            self.total_flex_option_value +
            self.total_treasury_value
        )

    def _compute_equity_value(self) -> float:
        """Calculate total equity value - uses Vest if available, otherwise Custodian."""
        # Primary source: Vest (OMS data)
        for frame in (self.snapshot.vest.equity, self.snapshot.custodian.equity):
            value = self.snapshot._frame_value_sum(
                frame, ["equity_market_value", "market_value", "net_market_value"]
            )
            if value is not None:
                return value  # Return first available value (Vest preferred)
            fallback = self.snapshot._price_quantity_sum(frame)
            if fallback is not None:
                return fallback
        return 0.0

    def _compute_option_value(self) -> float:
        """Calculate total regular option value - uses Vest if available, otherwise Custodian."""
        for frame in (self.snapshot.vest.options, self.snapshot.custodian.options):
            value = self.snapshot._frame_value_sum(
                frame, ["option_market_value", "market_value", "net_market_value"]
            )
            if value is not None:
                return value  # Return first available value (Vest preferred)
            fallback = self.snapshot._price_quantity_sum(frame, multiplier=100.0)
            if fallback is not None:
                return fallback
        return 0.0

    def _compute_flex_option_value(self) -> float:
        """Calculate total flex option value - uses Vest if available, otherwise Custodian."""
        for frame in (self.snapshot.vest.flex_options, self.snapshot.custodian.flex_options):
            value = self.snapshot._frame_value_sum(
                frame, ["flex_option_market_value", "option_market_value", "market_value", "net_market_value"]
            )
            if value is not None:
                return value  # Return first available value (Vest preferred)
            fallback = self.snapshot._price_quantity_sum(frame, multiplier=100.0)
            if fallback is not None:
                return fallback
        return 0.0

    def _compute_treasury_value(self) -> float:
        """Calculate total treasury value - uses Vest if available, otherwise Custodian."""
        for frame in (self.snapshot.vest.treasury, self.snapshot.custodian.treasury):
            value = self.snapshot._frame_value_sum(
                frame, ["treasury_market_value", "market_value", "net_market_value"]
            )
            if value is not None:
                return value  # Return first available value (Vest preferred)
            fallback = self.snapshot._price_quantity_sum(frame)
            if fallback is not None:
                return fallback
        return 0.0

    def _compute_option_delta_adjusted_notional(self) -> float:
        """Calculate delta-adjusted notional for regular options - uses Vest if available, otherwise Custodian."""
        for frame in (self.snapshot.vest.options, self.snapshot.custodian.options):
            value = self.snapshot._frame_value_sum(
                frame,
                ["option_delta_adjusted_notional", "delta_adjusted_notional"],
            )
            if value is not None:
                return value  # Return first available value (Vest preferred)
        return 0.0

    def _compute_flex_option_delta_adjusted_notional(self) -> float:
        """Calculate delta-adjusted notional for flex options - uses Vest if available, otherwise Custodian."""
        for frame in (self.snapshot.vest.flex_options, self.snapshot.custodian.flex_options):
            value = self.snapshot._frame_value_sum(
                frame,
                ["flex_option_delta_adjusted_notional", "option_delta_adjusted_notional", "delta_adjusted_notional"],
            )
            if value is not None:
                return value  # Return first available value (Vest preferred)
        return 0.0


class FundSnapshot:
    """Container for all holdings data at a point in time."""

    def __init__(
        self,
        *,
        vest: Optional[FundHoldings] = None,
        custodian: Optional[FundHoldings] = None,
        index: Optional[FundHoldings] = None,
        cash: float = 0.0,
        nav: float = 0.0,  # NAV per share
        tna: float = 0.0,  # Total Net Assets
        ta: float = 0.0,  # Total Assets
        expenses: float = 0.0,  # What custodian reports for expenses
        shares_outstanding: float = 0.0,
        flows: float = 0.0,
        basket: Optional[pd.DataFrame] = None,
        overlap: Optional[pd.DataFrame] = None,
        fund_name: Optional[str] = None,
        equity_trades: Optional[pd.DataFrame] = None,  # FIXED: pd.DataFrame not str
        cr_rd_data: Optional[pd.DataFrame] = None,  # FIXED: pd.DataFrame not str (creation/redemption)
    ) -> None:
        self.vest = vest if isinstance(vest, FundHoldings) else FundHoldings()
        self.custodian = custodian if isinstance(custodian, FundHoldings) else FundHoldings()
        self.index = index if isinstance(index, FundHoldings) else FundHoldings()

        # Store reported values from custodian/admin
        self._cash = float(cash)
        self._nav = float(nav)
        self._ta = float(ta)
        self._tna = float(tna)
        self._expenses = float(expenses)
        self._shares_outstanding = float(shares_outstanding)

        # Other data
        self.flows = float(flows)
        self.basket = basket if isinstance(basket, pd.DataFrame) else pd.DataFrame()
        self.overlap = overlap if isinstance(overlap, pd.DataFrame) else pd.DataFrame()
        self.fund_name = fund_name
        self.equity_trades = equity_trades if isinstance(equity_trades, pd.DataFrame) else pd.DataFrame()
        self.cr_rd_data = cr_rd_data if isinstance(cr_rd_data, pd.DataFrame) else pd.DataFrame()

        # Initialize metrics (computed from holdings)
        self._metrics = None

    @property
    def cash(self) -> float:
        return self._get_value("cash")

    @property
    def nav(self) -> float:
        return self._get_value("nav")

    @property
    def tna(self) -> float:
        return self._get_value("tna")

    @property
    def ta(self) -> float:
        return self._get_value("ta")

    @property
    def expenses(self) -> float:
        return self._get_value("expenses")

    @property
    def shares_outstanding(self) -> float:
        return self._get_value("shares_outstanding")

    @property
    def metrics(self) -> FundMetrics:
        """Lazy-loaded computed metrics from holdings."""
        if self._metrics is None:
            self._metrics = FundMetrics(self)
        return self._metrics

    # Alias properties for backward compatibility
    @property
    def reported_cash(self) -> float:
        return self.cash

    @property
    def reported_nav(self) -> float:
        return self.nav

    @property
    def reported_expenses(self) -> float:
        return self.expenses

    @property
    def reported_ta(self) -> float:
        return self.ta

    @property
    def reported_tna(self) -> float:
        return self.tna

    @property
    def reported_shares_outstanding(self) -> float:
        return self.shares_outstanding


    @property
    def total_assets(self) -> float:
        return self.ta

    @property
    def total_net_assets(self) -> float:
        return self.tna

    # Computed value properties (from metrics)
    @property
    def total_equity_value(self) -> float:
        """Total equity value from metrics."""
        return self.metrics.total_equity_value

    @property
    def total_option_value(self) -> float:
        """Total regular option value from metrics."""
        return self.metrics.total_option_value

    @property
    def total_flex_option_value(self) -> float:
        """Total flex option value from metrics."""
        return self.metrics.total_flex_option_value

    @property
    def total_all_options_value(self) -> float:
        """Total value of all options (regular + flex)."""
        return self.metrics.total_option_value + self.metrics.total_flex_option_value

    @property
    def total_treasury_value(self) -> float:
        """Total treasury value from metrics."""
        return self.metrics.total_treasury_value

    @property
    def total_option_delta_adjusted_notional(self) -> float:
        """Total option delta-adjusted notional from metrics."""
        return self.metrics.total_option_delta_adjusted_notional

    @property
    def calculated_tna(self) -> float:
        """Calculate TNA from holdings + cash."""
        return self.metrics.total_holdings_value

    def _get_reported_value(self, attribute: str) -> float:
        for source in (self.custodian, self.vest):
            value = getattr(source, attribute, None)
            if value is not None:
                return float(value)
        return float(getattr(self, f"_{attribute}", 0.0))

    def _get_value(self, attribute: str) -> float:
        for source in (self.custodian, self.vest):
            value = getattr(source, attribute, None)
            if value is not None:
                return float(value)
        return float(getattr(self, f"_{attribute}", 0.0))

    def _filter_frame_by_fund(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to only include rows for this fund."""
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return frame

        # If no fund_name set, return as-is (backward compatibility)
        if not self.fund_name:
            return frame

        # If fund column exists, filter by it
        if 'fund' in frame.columns:
            return frame[frame['fund'] == self.fund_name].copy()

        # No fund column means data should already be filtered
        return frame

    def _frame_value_sum(self, frame: pd.DataFrame, columns: Sequence[str]) -> Optional[float]:
        """Sum values from specified columns in dataframe."""
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None

        # Filter frame by fund if needed
        filtered_frame = self._filter_frame_by_fund(frame)
        if filtered_frame.empty:
            return None

        for column in columns:
            if column in filtered_frame.columns:
                series = pd.to_numeric(filtered_frame[column], errors="coerce").dropna()
                if not series.empty:
                    return float(series.sum())
        return None

    def _price_quantity_sum(self, frame: pd.DataFrame, multiplier: float = 1.0) -> Optional[float]:
        """Calculate value from price * quantity."""
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None

        # Filter frame by fund if needed
        filtered_frame = self._filter_frame_by_fund(frame)
        if filtered_frame.empty:
            return None

        if {"price", "quantity"}.issubset(filtered_frame.columns):
            price = pd.to_numeric(filtered_frame["price"], errors="coerce").fillna(0.0)
            quantity = pd.to_numeric(filtered_frame["quantity"], errors="coerce").fillna(0.0)
            if not price.empty and not quantity.empty:
                return float((price * quantity * multiplier).sum())
        return None


class FundData:
    """Complete fund data for current day (T) and prior day (T-1)."""

    def __init__(
        self,
        *,
        current: Optional[FundSnapshot] = None,
        previous: Optional[FundSnapshot] = None,
        # index: Optional[FundHoldings] = None,  # ADD - current index holdings
        # previous_index: Optional[FundHoldings] = None,  # ADD - T-1 index holdings
        # equity_trades: Optional[pd.DataFrame] = None,  # ADD - trades data
        # cr_rd_data: Optional[pd.DataFrame] = None,  # ADD - corporate actions
    ) -> None:
        self.current = current if isinstance(current, FundSnapshot) else FundSnapshot()
        self.previous = previous if isinstance(previous, FundSnapshot) else FundSnapshot()
        # self.index = index if isinstance(index, FundHoldings) else FundHoldings()
        # self.previous_index = previous_index if isinstance(previous_index, FundHoldings) else FundHoldings()
        # self.equity_trades = equity_trades if isinstance(equity_trades, pd.DataFrame) else pd.DataFrame()
        # self.cr_rd_data = cr_rd_data if isinstance(cr_rd_data, pd.DataFrame) else pd.DataFrame()


class Fund:
    """Fund entity with configuration and data."""

    def __init__(self, name: str, config: Dict, base_cls=None):
        self.name = name
        self.config = config or {}
        self.base_cls = base_cls
        self._data: Optional[FundData] = None

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------
    @property
    def data(self) -> FundData:
        """Fund processing data; always returns a valid FundData instance."""
        if self._data is None:
            self._data = FundData()
        return self._data

    @data.setter
    def data(self, value: Optional[FundData]) -> None:
        self._data = value if isinstance(value, FundData) else FundData()

    # ------------------------------------------------------------------
    # Configuration properties (from fund.config)
    # ------------------------------------------------------------------
    @property
    def expense_ratio(self) -> float:
        """Get expense ratio from config"""
        return float(self.config.get("expense_ratio", 0.0) or 0.0)

    @property
    def index_identifier(self) -> Optional[str]:
        """Index identifier from config"""
        value = self.config.get("index_identifier")
        if isinstance(value, str):
            value = value.strip()
        return value or None

    @property
    def vehicle(self) -> Optional[str]:
        """Fund vehicle type from config."""
        value = self.config.get("vehicle")
        return value if isinstance(value, str) else None

    @property
    def is_private_fund(self) -> bool:
        """Check if fund is private."""
        vehicle = self.vehicle
        if vehicle is None:
            return False
        return vehicle.lower() == "private_fund"

    @property
    def is_closed_end_fund(self) -> bool:
        """Check if fund is closed-end."""
        vehicle = self.vehicle
        if vehicle is None:
            return False
        return vehicle.lower() == "closed_end_fund"

    @property
    def has_listed_option(self) -> bool:
        """Check if fund has listed options."""
        return bool(self.config.get("has_listed_option", False))

    @property
    def listed_option_type(self) -> Optional[str]:
        """Type of listed options (e.g., 'index', 'single_stock')."""
        value = self.config.get("listed_option_type")
        if isinstance(value, str):
            value = value.strip()
        return value or None

    @property
    def has_flex_option(self) -> bool:
        """Check if fund has flex options."""
        return bool(self.config.get("has_flex_option", False))

    @property
    def flex_option_type(self) -> Optional[str]:
        """Type of flex options."""
        value = self.config.get("flex_option_type")
        return value if isinstance(value, str) and value else None

    @property
    def has_otc(self) -> bool:
        """Check if fund has OTC positions."""
        return bool(self.config.get("has_otc", False))

    @property
    def has_treasury(self) -> bool:
        """Check if fund has treasury holdings."""
        return bool(self.config.get("has_treasury", False))

    @property
    def diversification_status(self) -> Optional[str]:
        """Fund's diversification status from config."""
        value = self.config.get("diversification_status")
        if isinstance(value, str):
            value = value.strip()
        return value or None