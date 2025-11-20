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
    ) -> None:
        self.equity = equity if isinstance(equity, pd.DataFrame) else pd.DataFrame()
        self.options = options if isinstance(options, pd.DataFrame) else pd.DataFrame()
        self.flex_options = flex_options if isinstance(flex_options, pd.DataFrame) else pd.DataFrame()  # NEW
        self.treasury = treasury if isinstance(treasury, pd.DataFrame) else pd.DataFrame()

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
        """Calculate total equity value from vest and custodian holdings."""
        for frame in (self.snapshot.vest.equity, self.snapshot.custodian.equity):
            value = self.snapshot._frame_value_sum(
                frame, ["equity_market_value", "market_value", "net_market_value"]
            )
            if value is not None:
                return value
            fallback = self.snapshot._price_quantity_sum(frame)
            if fallback is not None:
                return fallback
        return 0.0

    def _compute_option_value(self) -> float:
        """Calculate total regular option value from vest and custodian holdings."""
        for frame in (self.snapshot.vest.options, self.snapshot.custodian.options):
            value = self.snapshot._frame_value_sum(
                frame, ["option_market_value", "market_value", "net_market_value"]
            )
            if value is not None:
                return value
            fallback = self.snapshot._price_quantity_sum(frame, multiplier=100.0)
            if fallback is not None:
                return fallback
        return 0.0

    def _compute_flex_option_value(self) -> float:
        """Calculate total flex option value from vest and custodian holdings."""
        for frame in (self.snapshot.vest.flex_options, self.snapshot.custodian.flex_options):
            value = self.snapshot._frame_value_sum(
                frame, ["flex_option_market_value", "option_market_value", "market_value", "net_market_value"]
            )
            if value is not None:
                return value
            fallback = self.snapshot._price_quantity_sum(frame, multiplier=100.0)
            if fallback is not None:
                return fallback
        return 0.0

    def _compute_treasury_value(self) -> float:
        """Calculate total treasury value from vest and custodian holdings."""
        for frame in (self.snapshot.vest.treasury, self.snapshot.custodian.treasury):
            value = self.snapshot._frame_value_sum(
                frame, ["treasury_market_value", "market_value", "net_market_value"]
            )
            if value is not None:
                return value
            fallback = self.snapshot._price_quantity_sum(frame)
            if fallback is not None:
                return fallback
        return 0.0

    def _compute_option_delta_adjusted_notional(self) -> float:
        """Calculate delta-adjusted notional for regular options."""
        for frame in (self.snapshot.vest.options, self.snapshot.custodian.options):
            value = self.snapshot._frame_value_sum(
                frame,
                ["option_delta_adjusted_notional", "delta_adjusted_notional"],
            )
            if value is not None:
                return value
        return 0.0

    def _compute_flex_option_delta_adjusted_notional(self) -> float:
        """Calculate delta-adjusted notional for flex options."""
        for frame in (self.snapshot.vest.flex_options, self.snapshot.custodian.flex_options):
            value = self.snapshot._frame_value_sum(
                frame,
                ["flex_option_delta_adjusted_notional", "option_delta_adjusted_notional", "delta_adjusted_notional"],
            )
            if value is not None:
                return value
        return 0.0

class FundSnapshot:
    """Container for all holdings data at a point in time."""

    def __init__(
        self,
        *,
        vest: Optional[FundHoldings] = None,
        custodian: Optional[FundHoldings] = None,
        index: Optional[FundHoldings] = None,
        reported_cash: float = 0.0,
        reported_nav: float = 0.0,  # NAV per share
        reported_tna: float = 0.0,  # Total Net Assets
        reported_expenses: float = 0.0,  # What custodian reports for expenses
        reported_shares_outstanding: float = 0.0,
        flows: float = 0.0,
        basket: Optional[pd.DataFrame] = None,
        index: Optional[pd.DataFrame] = None,
        overlap: Optional[pd.DataFrame] = None,
        fund_name: Optional[str] = None,
        equity_trades: Optional[str] = None,
        cr_rd_data: Optional[str] = None,
    ) -> None:
        self.vest = vest if isinstance(vest, FundHoldings) else FundHoldings()
        self.custodian = custodian if isinstance(custodian, FundHoldings) else FundHoldings()
        self.index = index if isinstance(index, FundHoldings) else FundHoldings()
        self.cash = float(cash or 0.0)
        self.nav = float(nav or 0.0)
        self.expenses = float(expenses or 0.0)
        self.total_assets = float(total_assets or 0.0)
        self.total_net_assets = float(total_net_assets or 0.0)
        self.fund_name = fund_name

        self.total_equity_value = self._compute_equity_value()
        self.total_option_value = self._compute_option_value()
        self.total_option_delta_adjusted_notional = self._compute_option_delta_adjusted_notional()
        self.total_treasury_value = self._compute_treasury_value()
        self.flows = float(flows or 0.0)
        self.basket = basket if isinstance(basket, pd.DataFrame) else pd.DataFrame()
        self.index = index if isinstance(index, pd.DataFrame) else pd.DataFrame()
        self.overlap = overlap if isinstance(overlap, pd.DataFrame) else pd.DataFrame()

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

    def _compute_equity_value(self) -> float:
        for frame in (self.vest.equity, self.custodian.equity):
            value = self._frame_value_sum(
                frame, ["equity_market_value", "market_value", "net_market_value"]
            )
            if value is not None:
                return value
            fallback = self._price_quantity_sum(frame)
            if fallback is not None:
                return fallback
        return 0.0

    def _compute_option_value(self) -> float:
        for frame in (self.vest.options, self.custodian.options):
            value = self._frame_value_sum(
                frame, ["option_market_value", "market_value", "net_market_value"]
            )
            if value is not None:
                return value
            fallback = self._price_quantity_sum(frame, multiplier=100.0)
            if fallback is not None:
                return fallback
        return 0.0

    def _compute_option_delta_adjusted_notional(self) -> float:
        for frame in (self.vest.options, self.custodian.options):
            value = self._frame_value_sum(
                frame,
                [
                    "option_delta_adjusted_notional",
                    "delta_adjusted_notional",
                ],
            )
            if value is not None:
                return value
        return 0.0

    def _compute_treasury_value(self) -> float:
        """Compute treasury value with proper fund filtering."""
        for frame in (self.vest.treasury, self.custodian.treasury):
            # CRITICAL: Filter frame to ensure we only sum this fund's treasury
            value = self._frame_value_sum(
                frame, ["treasury_market_value", "market_value", "net_market_value"]
            )
            if value is not None:
                return value
            fallback = self._price_quantity_sum(frame)
            if fallback is not None:
                return fallback
        return 0.0

class FundData:
    """Complete fund data for current day (T) and prior day (T-1)."""

    def __init__(
        self,
        *,
        current: Optional[FundSnapshot] = None,
        previous: Optional[FundSnapshot] = None,
        index: Optional[FundHoldings] = None,  # ADD - current index holdings
        previous_index: Optional[FundHoldings] = None,  # ADD - T-1 index holdings
        equity_trades: Optional[pd.DataFrame] = None,  # ADD - trades data
        cr_rd_data: Optional[pd.DataFrame] = None,  # ADD - corporate actions
    ) -> None:
        self.current = current if isinstance(current, FundSnapshot) else FundSnapshot()
        self.previous = previous if isinstance(previous, FundSnapshot) else FundSnapshot()
        self.index = index if isinstance(index, FundHoldings) else FundHoldings()
        self.previous_index = previous_index if isinstance(previous_index, FundHoldings) else FundHoldings()
        self.equity_trades = equity_trades if isinstance(equity_trades, pd.DataFrame) else pd.DataFrame()
        self.cr_rd_data = cr_rd_data if isinstance(cr_rd_data, pd.DataFrame) else pd.DataFrame()

class Fund:
    def __init__(self, name: str, config: Dict, base_cls=None):
        self.name = name
        self.config = config or {}
        self.base_cls = base_cls
        self.data: Optional[FundData] = FundData()  # Keep this!

    @property
    def expense_ratio(self) -> float:
        """Get expense ratio from config"""
        return float(self.config.get("expense_ratio", 0.0) or 0.0)

    @property
    def equity_trades(self) -> pd.DataFrame:
        """Equity trades DataFrame from FundData"""
        return self._copy_dataframe(self.data.equity_trades)

    @property
    def cr_rd_data(self) -> pd.DataFrame:
        """Corporate actions DataFrame from FundData"""
        return self._copy_dataframe(self.data.cr_rd_data)

    @property
    def index_holdings(self) -> pd.DataFrame:
        """Current index holdings"""
        return self._copy_dataframe(self.data.index.equity)

    @property
    def previous_index_holdings(self) -> pd.DataFrame:
        """Previous index holdings"""
        return self._copy_dataframe(self.data.previous_index.equity)

    @staticmethod
    def _extract_numeric_value(source, candidates: Optional[list[str]] = None) -> float:
        """Attempt to coerce ``source`` into a float using optional column hints."""
        if source is None:
            return 0.0

        if isinstance(source, (int, float)):
            return float(source)

        if isinstance(source, pd.Series):
            try:
                return float(pd.to_numeric(source, errors="coerce").fillna(0.0).sum())
            except (ValueError, TypeError):
                return 0.0

        if isinstance(source, pd.DataFrame):
            candidates = candidates or []
            for column in candidates:
                if column in source.columns:
                    try:
                        return float(
                            pd.to_numeric(source[column], errors="coerce").fillna(0.0).sum()
                        )
                    except (ValueError, TypeError):
                        continue
            numeric_cols = source.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                try:
                    return float(source[numeric_cols[0]].fillna(0.0).sum())
                except (ValueError, TypeError):
                    return 0.0
        return 0.0

    @staticmethod
    def _copy_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame):
            return df.copy()
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Data management helpers
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

    # Add these properties that your services use:
    @property
    def cash_value(self) -> float:
        return float(getattr(self.data.current, "cash", 0.0)) or 0.0

    @property
    def total_assets(self) -> float:
        return float(getattr(self.data.current, "total_assets", 0.0)) or 0.0

    @property
    def total_net_assets(self) -> float:
        return float(getattr(self.data.current, "total_net_assets", 0.0)) or 0.0

    @property
    def total_equity_value(self) -> float:
        return float(getattr(self.data.current, "total_equity_value", 0.0) or 0.0)

    @property
    def total_option_value(self) -> float:
        return float(getattr(self.data.current, "total_option_value", 0.0) or 0.0)

    @property
    def total_treasury_value(self) -> float:
        return float(getattr(self.data.current, "total_treasury_value", 0.0) or 0.0)

    @property
    def expenses(self) -> float:
        return float(getattr(self.data.current, "expenses", 0.0) or 0.0)

    @property
    def index_identifier(self) -> Optional[str]:
        value = self.config.get("index_identifier")
        if isinstance(value, str):
            value = value.strip()
        return value or None

    @property
    def is_private_fund(self) -> bool:
        return (self.vehicle or "").lower() == "private_fund"

    @property
    def is_closed_end_fund(self) -> bool:
        return (self.vehicle or "").lower() == "closed_end_fund"

    @property
    def has_listed_option(self) -> bool:
        return bool(self.config.get("has_listed_option", False))

    @property
    def listed_option_type(self) -> Optional[str]:
        value = self.config.get("listed_option_type")
        if isinstance(value, str):
            value = value.strip()
        return value or None

    @property
    def has_flex_option(self) -> bool:
        return bool(self.config.get("has_flex_option", False))

    @property
    def flex_option_type(self) -> Optional[str]:
        value = self.config.get("flex_option_type")
        return value if isinstance(value, str) and value else None

    @property
    def has_otc(self) -> bool:
        return bool(self.config.get("has_otc", False))

    @property
    def has_treasury(self) -> bool:
        return bool(self.config.get("has_treasury", False))

    @property
    def diversification_status(self) -> Optional[str]:
        value = self.config.get("diversification_status")
        if isinstance(value, str):
            value = value.strip()
        return value or None

    @property
    def has_listed_option(self) -> bool:
        return bool(self.config.get("has_listed_option", False))

    @property
    def has_flex_option(self) -> bool:
        return bool(self.config.get("has_flex_option", False))

    @property
    def flex_option_type(self) -> Optional[str]:
        value = self.config.get("flex_option_type")
        return value if isinstance(value, str) and value else None

    @property
    def has_otc(self) -> bool:
        return bool(self.config.get("has_otc", False))

    @property
    def has_treasury(self) -> bool:
        return bool(self.config.get("has_treasury", False))

    @property
    def total_option_delta_adjusted_notional(self) -> float:
        """For compliance checks that need delta-adjusted option values"""
        current = getattr(self.data, "current", None)
        if current is not None and hasattr(current, "total_option_delta_adjusted_notional"):
            value = getattr(current, "total_option_delta_adjusted_notional", 0.0)
            if value not in (None, ""):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

        options = getattr(self.data.current.vest, "options", pd.DataFrame())
        column = None
        for candidate in ("option_delta_adjusted_notional", "delta_adjusted_notional"):
            if isinstance(options, pd.DataFrame) and candidate in options.columns:
                column = candidate
                break

        if column:
            return float(pd.to_numeric(options[column], errors="coerce").fillna(0.0).sum())
        return 0.0

    def get_dividends(self, analysis_date: str) -> float:
        """Return dividend impact for the requested analysis date."""

        dividends = getattr(self.data, "dividends", 0.0)
        return self._extract_numeric_value(dividends, ["dividend", "dividends", "amount", "value"])

    def get_expenses(self, analysis_date: str) -> float:
        """Return expenses for the requested analysis date."""

        current = getattr(self.data, "current", None)
        if current is None:
            return 0.0
        try:
            return float(getattr(current, "expenses", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def get_distributions(self, analysis_date: str) -> float:
        """Return distributions paid on the analysis date."""

        distributions = getattr(self.data, "distributions", 0.0)
        return self._extract_numeric_value(
            distributions,
            ["distribution", "distributions", "amount", "value"],
        )

    def get_flows_adjustment(self, analysis_date: str, prior_date: str) -> float:
        """Return flows adjustment between the two analysis dates."""
        return float(self.data.current.flows or 0.0)


    # ------------------------------------------------------------------
    # DataFrame accessors used by reconciliation services
    # ------------------------------------------------------------------
    @property
    def equity_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(getattr(self.data.current.vest, "equity", pd.DataFrame()))

    @property
    def custodian_equity_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(
            getattr(self.data.current.custodian, "equity", pd.DataFrame())
        )

    @property
    def previous_equity_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(getattr(self.data.previous.vest, "equity", pd.DataFrame()))

    @property
    def previous_custodian_equity_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(
            getattr(self.data.previous.custodian, "equity", pd.DataFrame())
        )

    @property
    def options_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(getattr(self.data.current.vest, "options", pd.DataFrame()))

    @property
    def custodian_option_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(
            getattr(self.data.current.custodian, "options", pd.DataFrame())
        )

    @property
    def previous_options_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(getattr(self.data.previous.vest, "options", pd.DataFrame()))

    @property
    def previous_custodian_option_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(
            getattr(self.data.previous.custodian, "options", pd.DataFrame())
        )

    @property
    def treasury_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(getattr(self.data.current.vest, "treasury", pd.DataFrame()))

    @property
    def custodian_treasury_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(
            getattr(self.data.current.custodian, "treasury", pd.DataFrame())
        )

    @property
    def overlap_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(getattr(self.data.current, "overlap", pd.DataFrame()))

    @property
    def previous_overlap_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(getattr(self.data.previous, "overlap", pd.DataFrame()))

    @property
    def previous_treasury_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(getattr(self.data.previous.vest, "treasury", pd.DataFrame()))

    @property
    def previous_custodian_treasury_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(
            getattr(self.data.previous.custodian, "treasury", pd.DataFrame())
        )

    @property
    def equity_trades(self) -> pd.DataFrame:
        trades = getattr(self.data, "equity_trades", pd.DataFrame())
        return self._copy_dataframe(trades)

    @property
    def cr_rd_data(self) -> pd.DataFrame:
        corporate_actions = getattr(self.data, "cr_rd_data", pd.DataFrame())
        return self._copy_dataframe(corporate_actions)

    @property
    def index_holdings(self) -> pd.DataFrame:
        index_df = getattr(self.data.current, "index", pd.DataFrame())
        return self._copy_dataframe(index_df)

    @property
    def flex_options_holdings(self) -> pd.DataFrame:
        """Current flex options holdings from Vest."""
        return self._copy_dataframe(getattr(self.data.current.vest, "flex_options", pd.DataFrame()))

    @property
    def custodian_flex_options_holdings(self) -> pd.DataFrame:
        """Current flex options holdings from custodian."""
        return self._copy_dataframe(getattr(self.data.current.custodian, "flex_options", pd.DataFrame()))

    @property
    def previous_flex_options_holdings(self) -> pd.DataFrame:
        """Previous flex options holdings from Vest."""
        return self._copy_dataframe(getattr(self.data.previous.vest, "flex_options", pd.DataFrame()))

    @property
    def previous_custodian_flex_options_holdings(self) -> pd.DataFrame:
        """Previous flex options holdings from custodian."""
        return self._copy_dataframe(getattr(self.data.previous.custodian, "flex_options", pd.DataFrame()))

    # Add NAV DataFrame properties
    @property
    def nav_dataframe(self) -> pd.DataFrame:
        """Current NAV DataFrame with all NAV-related columns."""
        return self._copy_dataframe(getattr(self.data.current, "nav_dataframe", pd.DataFrame()))

    @property
    def previous_nav_dataframe(self) -> pd.DataFrame:
        """Previous NAV DataFrame with all NAV-related columns."""
        return self._copy_dataframe(getattr(self.data.previous, "nav_dataframe", pd.DataFrame()))

    # Add convenient metric accessors
    @property
    def current_equity_value(self) -> float:
        """Current total equity market value (calculated from holdings)."""
        return self.data.current.metrics.total_equity_value

    @property
    def current_option_value(self) -> float:
        """Current total regular option market value (calculated from holdings)."""
        return self.data.current.metrics.total_option_value

    @property
    def current_flex_option_value(self) -> float:
        """Current total flex option market value (calculated from holdings)."""
        return self.data.current.metrics.total_flex_option_value

    @property
    def current_treasury_value(self) -> float:
        """Current total treasury market value (calculated from holdings)."""
        return self.data.current.metrics.total_treasury_value

    @property
    def current_tna(self) -> float:
        """Current total net assets (reported by custodian/admin)."""
        return self.data.current.reported_tna

    @property
    def calculated_tna(self) -> float:
        """Current TNA calculated from holdings + cash."""
        return self.data.current.calculated_tna
