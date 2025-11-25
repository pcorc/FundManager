from typing import Dict, Optional, Sequence, Set
import pandas as pd
import re
from typing import Any  # Add to existing typing imports

from datetime import date


class FundHoldings:
    """Wrapper for holdings data for a specific source (e.g. Vest or custodian)."""

    def __init__(
        self,
        *,
        equity: Optional[pd.DataFrame] = None,
        options: Optional[pd.DataFrame] = None,
        flex_options: Optional[pd.DataFrame] = None,  # NEW
        treasury: Optional[pd.DataFrame] = None,

    ) -> None:
        self.equity = self._clean_holdings_dataframe(equity, holding_type='equity')
        self.options = self._clean_holdings_dataframe(options, holding_type='options')
        self.flex_options = self._clean_holdings_dataframe(flex_options, holding_type='flex_options')
        self.treasury = self._clean_holdings_dataframe(treasury, holding_type='treasury')

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
    ) -> None:
        self.current = current if isinstance(current, FundSnapshot) else FundSnapshot()
        self.previous = previous if isinstance(previous, FundSnapshot) else FundSnapshot()


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
    def is_etf(self) -> bool:
        """Check if fund is closed-end."""
        vehicle = self.vehicle
        if vehicle is None:
            return False
        return vehicle.lower() == "etf"

    @property
    def is_diversified(self) -> bool:
        """Check if fund is closed-end."""
        vehicle = self.vehicle
        if vehicle is None:
            return False
        return vehicle.lower() == "diversification_status"

    @property
    def is_non_diversified(self) -> bool:
        """Check if fund is non-diversified."""
        status = self.diversification_status
        if status is None:
            return False
        return status.lower() == "non_diversified"

    @property
    def index_flex_usage(self) -> Optional[str]:
        """Fund index flex usage from config."""
        value = self.config.get("index_flex_usage")
        return value if isinstance(value, str) else None

    @property
    def uses_index_flex(self) -> bool:
        """Check if fund uses index FLEX options."""
        usage = self.index_flex_usage
        if usage is None:
            return False
        return usage.lower() in ("true", "yes", "enabled", "1")

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
    def flex_option_pattern(self) -> str:
        """
        Regex pattern to identify flex options.

        Returns pattern based on flex_option_type:
        - 'index': "SPX|XSP" (index flex options)
        - 'single_stock': "^2" (options starting with "2")
        - default: "SPX|XSP"
        """
        if not self.has_flex_option:
            return "SPX|XSP"

        flex_type = self.flex_option_type

        if flex_type == "index":
            return "SPX|XSP"
        elif flex_type == "single_stock":
            return "^2"
        else:
            return "SPX|XSP"

    @property
    def option_roll_tenor(self) -> Optional[str]:
        """Option roll tenor ('weekly', 'monthly', 'quarterly')."""
        return self.config.get("option_roll_tenor")

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

    def get_flex_options(self, snapshot: str = 'current') -> pd.DataFrame:
        """
        Get flex options matching the fund's flex_option_pattern.

        Args:
            snapshot: 'current' or 'prior' to select which snapshot

        Returns:
            DataFrame with flex options from vest source (preferred)
            Falls back to custodian if vest is empty

        Example:
            flex_opts = fund.get_flex_options('current')
        """
        if not self.config.get("has_flex_option"):
            return pd.DataFrame()

        # Get the snapshot
        snap = self.data.current if snapshot == 'current' else self.data.prior
        if snap is None:
            return pd.DataFrame()

        # Try vest first, then custodian
        for source in (snap.vest, snap.custodian):
            if source is None or source.options.empty:
                continue

            options_df = source.options.copy()

            # Filter by pattern
            if 'optticker' not in options_df.columns:
                continue

            pattern = self.config.get("flex_option_pattern") or "SPX|XSP"
            mask = options_df['optticker'].str.contains(pattern, na=False, regex=True)
            flex_opts = options_df[mask]

            if not flex_opts.empty:
                return flex_opts

        return pd.DataFrame()

    def get_regular_options(self, snapshot: str = 'current') -> pd.DataFrame:
        """
        Get regular (non-flex) options.

        Args:
            snapshot: 'current' or 'prior' to select which snapshot

        Returns:
            DataFrame with non-flex options from vest source (preferred)
            Falls back to custodian if vest is empty

        Example:
            regular_opts = fund.get_regular_options('current')
        """
        # Get the snapshot
        snap = self.data.current if snapshot == 'current' else self.data.prior
        if snap is None:
            return pd.DataFrame()

        # Try vest first, then custodian
        for source in (snap.vest, snap.custodian):
            if source is None or source.options.empty:
                continue

            options_df = source.options.copy()

            if 'optticker' not in options_df.columns:
                continue

            # If no flex options, return all
            if not self.config.get("has_flex_option"):
                return options_df

            # Filter out flex pattern
            pattern = self.config.get("flex_option_pattern") or "SPX|XSP"
            mask = ~options_df['optticker'].str.contains(pattern, na=False, regex=True)
            regular_opts = options_df[mask]

            if not regular_opts.empty:
                return regular_opts

        return pd.DataFrame()

    # ========================================================================
    # TICKER GATHERING METHODS
    # ========================================================================

    def gather_all_tickers(
            self,
            asset_class: str,
            include_prior: bool = True,
    ) -> Set[str]:
        """
        Gather unique tickers from current and optionally prior snapshots.

        Args:
            asset_class: 'equity', 'options', 'flex_options', 'treasury'
            include_prior: Whether to include prior snapshot tickers

        Returns:
            Set of unique ticker symbols from both vest and custodian sources
            across current and (optionally) prior snapshots

        Example:
            all_tickers = fund.gather_all_tickers('equity', include_prior=True)
        """
        all_tickers: Set[str] = set()

        # Map asset class to ticker column name
        ticker_col_map = {
            'equity': 'eqyticker',
            'options': 'optticker',
            'flex_options': 'optticker',
            'treasury': 'ticker',
        }

        ticker_col = ticker_col_map.get(asset_class)
        if not ticker_col:
            return all_tickers

        # Gather from current snapshot
        if self.data.current:
            for source in (self.data.current.vest, self.data.current.custodian):
                if source is None:
                    continue

                df = self._get_asset_class_df(source, asset_class)
                if not df.empty and ticker_col in df.columns:
                    all_tickers.update(df[ticker_col].dropna().unique())

        # Gather from prior snapshot if requested
        if include_prior and self.data.prior:
            for source in (self.data.prior.vest, self.data.prior.custodian):
                if source is None:
                    continue

                df = self._get_asset_class_df(source, asset_class)
                if not df.empty and ticker_col in df.columns:
                    all_tickers.update(df[ticker_col].dropna().unique())

        return all_tickers

    def gather_option_tickers(
            self,
            include_flex: bool,
            include_prior: bool = True,
    ) -> Set[str]:
        """
        Gather option tickers with flex filtering.

        Args:
            include_flex: If True, include only flex options;
                         if False, exclude flex options
            include_prior: Whether to include prior snapshot tickers

        Returns:
            Set of option ticker symbols filtered by flex pattern

        Example:
            # Get only regular options
            regular = fund.gather_option_tickers(include_flex=False)

            # Get only flex options
            flex = fund.gather_option_tickers(include_flex=True)
        """
        all_tickers: Set[str] = set()

        # If no flex options, handle accordingly
        if not self.has_flex_option:  # Changed from self.properties.has_flex_option
            if include_flex:
                return all_tickers  # No flex options exist
            else:
                return self.gather_all_tickers('options', include_prior)

        pattern = self.config.get("flex_option_pattern") or "SPX|XSP"
        snapshots = [self.data.current]
        if include_prior and self.data.prior:
            snapshots.append(self.data.prior)

        for snapshot in snapshots:
            if not snapshot:
                continue

            for source in (snapshot.vest, snapshot.custodian):
                if source is None or source.options.empty:
                    continue

                df = source.options
                if 'optticker' not in df.columns:
                    continue

                # Apply pattern filter
                mask = df['optticker'].str.contains(pattern, na=False, regex=True)

                # Include or exclude based on flag
                filtered = df[mask if include_flex else ~mask]

                all_tickers.update(filtered['optticker'].dropna().unique())

        return all_tickers

    def _get_asset_class_df(self, source, asset_class: str) -> pd.DataFrame:
        """Helper to get DataFrame for an asset class from a source."""
        if asset_class == 'equity':
            return source.equity
        elif asset_class == 'options':
            return source.options
        elif asset_class == 'flex_options':
            return source.flex_options
        elif asset_class == 'treasury':
            return source.treasury
        return pd.DataFrame()

    # ========================================================================
    # PRICE BREAK ACCESS
    # ========================================================================

    def get_price_breaks(self, asset_class: str) -> pd.DataFrame:
        """
        Get price breaks for an asset class.

        Args:
            asset_class: 'equity', 'option', 'treasury'

        Returns:
            DataFrame with price breaks indexed by ticker

        Note: Price breaks should be stored in fund.data.price_breaks
        as a dict with keys matching asset class names.

        Example:
            equity_breaks = fund.get_price_breaks('equity')
            if 'AAPL' in equity_breaks.index:
                cust_price = equity_breaks.loc['AAPL', 'price_cust']
        """
        if not hasattr(self.data, 'price_breaks'):
            return pd.DataFrame()

        price_breaks = self.data.price_breaks
        if not isinstance(price_breaks, dict):
            return pd.DataFrame()

        breaks = price_breaks.get(asset_class, pd.DataFrame())
        if isinstance(breaks, pd.DataFrame):
            return breaks

        return pd.DataFrame()



    # ========================================================================
    # ASSIGNMENT DATA ACCESS
    # ========================================================================

    def get_assignments(
            self,
            filter_date: Optional[date | str] = None,
    ) -> pd.DataFrame:
        """
        Get option assignment data, optionally filtered by date.

        Args:
            filter_date: If provided, filter assignments to this date

        Returns:
            DataFrame with assignment records

        Example:
            assignments = fund.get_assignments(filter_date=analysis_date)
            if not assignments.empty:
                total_gl = assignments['pnl'].sum()
        """
        # Check for assignments in data
        assignments = None
        if hasattr(self.data, 'assignments'):
            assignments = self.data.assignments
        elif hasattr(self.data, 'option_assignments'):
            assignments = self.data.option_assignments

        if not isinstance(assignments, pd.DataFrame) or assignments.empty:
            return pd.DataFrame()

        # If no date filter, return all
        if filter_date is None:
            return assignments.copy()

        # Filter by date
        date_cols = [
            col for col in assignments.columns
            if 'date' in str(col).lower()
        ]

        if not date_cols:
            return assignments.copy()

        try:
            df = assignments.copy()
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            target = pd.Timestamp(filter_date).normalize()
            filtered = df[df[date_cols[0]].dt.normalize() == target]
            return filtered
        except Exception:
            return assignments.copy()



    # ========================================================================
    # SETTLEMENT DATE CHECKING
    # ========================================================================

    def is_option_settlement_date(self, check_date: date | str) -> bool:
        """
        Determine if a given date is an option settlement date.

        Args:
            check_date: Date to check (date object or string)

        Returns:
            True if the date is a settlement date based on:
            - Fund's option_roll_tenor configuration (weekly/monthly/quarterly)
            - Third Friday rules for monthly/quarterly
            - Assignment data if available

        Example:
            if fund.is_option_settlement_date('2024-01-19'):
                # Process assignment G/L
                pass
        """
        if check_date is None:
            return False

        # Convert to pandas Timestamp for easier date math
        timestamp = pd.Timestamp(check_date)

        # Get tenor from properties or config
        tenor = ""
        if hasattr(self.data, 'option_roll_tenor'):
            tenor = str(self.data.option_roll_tenor or "").lower()
        elif hasattr(self.data, 'config') and isinstance(self.data.config, dict):
            tenor = str(self.data.config.get('option_roll_tenor', '')).lower()

        # Check based on tenor
        if tenor == 'weekly':
            # Every Friday
            return timestamp.weekday() == 4

        elif tenor == 'monthly':
            # Third Friday of the month OR last business day
            month_start = timestamp.replace(day=1)
            month_end = timestamp + pd.tseries.offsets.MonthEnd(0)

            # Check for third Friday
            third_fridays = pd.date_range(
                month_start,
                month_end,
                freq='WOM-3FRI',
            )
            if not third_fridays.empty and timestamp.normalize() == third_fridays[0].normalize():
                return True

            # Check for last business day
            last_bday = month_end - pd.tseries.offsets.BDay(0)
            return timestamp.normalize() == last_bday.normalize()

        elif tenor == 'quarterly':
            # Only in March, June, September, December
            if timestamp.month not in (3, 6, 9, 12):
                return False

            # Third Friday of the quarter-end month
            month_start = timestamp.replace(day=1)
            month_end = timestamp + pd.tseries.offsets.MonthEnd(0)

            third_fridays = pd.date_range(
                month_start,
                month_end,
                freq='WOM-3FRI',
            )
            return any(timestamp.normalize() == tf.normalize() for tf in third_fridays)

        # Default: check if it's a third Friday
        month_start = timestamp.replace(day=1)
        month_end = timestamp + pd.tseries.offsets.MonthEnd(0)
        third_fridays = pd.date_range(month_start, month_end, freq='WOM-3FRI')
        if any(timestamp.normalize() == tf.normalize() for tf in third_fridays):
            return True

        # Check assignments data if available
        if hasattr(self.data, 'assignments'):
            assignments = self.data.assignments
            if isinstance(assignments, pd.DataFrame) and not assignments.empty:
                date_cols = [
                    col for col in assignments.columns
                    if 'date' in str(col).lower()
                ]
                if date_cols:
                    try:
                        dates = pd.to_datetime(
                            assignments[date_cols[0]], errors='coerce'
                        ).dt.normalize()
                        return timestamp.normalize() in dates.values
                    except Exception:
                        pass

        return False

    # ========================================================================
    # NAV DATA ACCESS
    # ========================================================================

    def get_nav_metric(
            self,
            metric: str,
            snapshot: str = 'current',
    ) -> float:
        """
        Get NAV-related metrics from fund data.

        Args:
            metric: One of 'total_net_assets', 'shares_outstanding', 'nav', etc.
            snapshot: 'current' or 'prior'

        Returns:
            Float value of the metric, 0.0 if not found

        Looks in:
            - fund.data.{snapshot}.nav DataFrame
            - fund.data.{snapshot}.custodian NAV fields

        Example:
            tna = fund.get_nav_metric('total_net_assets', 'current')
            shares = fund.get_nav_metric('shares_outstanding', 'current')
        """
        snap = self.data.current if snapshot == 'current' else self.data.prior
        if snap is None:
            return 0.0

        # Check if there's a nav DataFrame
        if hasattr(snap, 'nav') and isinstance(snap.nav, pd.DataFrame):
            nav_df = snap.nav
            if not nav_df.empty and metric in nav_df.columns:
                series = pd.to_numeric(nav_df[metric], errors='coerce').dropna()
                if not series.empty:
                    return float(series.iloc[0])

        # Check custodian source
        if snap.custodian and hasattr(snap.custodian, metric):
            value = getattr(snap.custodian, metric)
            if isinstance(value, (int, float)):
                return float(value)

        return 0.0

    # ========================================================================
    # DIVIDEND DATA ACCESS
    # ========================================================================

    def get_equity_dividends(self) -> float:
        """
        Calculate total dividend income from current equity holdings.

        Returns:
            Total dividends from dividend * quantity

        Example:
            div_income = fund.get_equity_dividends()
        """
        if not self.data.current or not self.data.current.vest:
            return 0.0

        equity = self.data.current.vest.equity
        if equity.empty:
            return 0.0

        # Check for dividend columns
        if 'dividend' not in equity.columns or 'quantity' not in equity.columns:
            # Try alternate column names
            div_col = None
            qty_col = None

            for col in equity.columns:
                col_lower = str(col).lower()
                if 'dividend' in col_lower or 'div' in col_lower:
                    div_col = col
                if 'quantity' in col_lower or 'shares' in col_lower:
                    qty_col = col

            if div_col and qty_col:
                return float((equity[div_col] * equity[qty_col]).sum())

            return 0.0

        return float((equity['dividend'] * equity['quantity']).sum())
