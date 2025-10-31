from typing import Dict, Optional, Sequence
import pandas as pd


class GainLossResult:
    """Simple container for gain/loss calculations."""

    def __init__(
        self,
        raw_gl: float = 0.0,
        adjusted_gl: float = 0.0,
        details: Optional[pd.DataFrame] = None,
        price_adjustments: Optional[pd.DataFrame] = None,
    ) -> None:
        self.raw_gl = raw_gl
        self.adjusted_gl = adjusted_gl
        self.details = details if details is not None else pd.DataFrame()
        self.price_adjustments = (
            price_adjustments if price_adjustments is not None else pd.DataFrame()
        )


class FundHoldings:
    """Wrapper for holdings data for a specific source (e.g. Vest or custodian)."""

    def __init__(
        self,
        *,
        equity: Optional[pd.DataFrame] = None,
        options: Optional[pd.DataFrame] = None,
        treasury: Optional[pd.DataFrame] = None,
    ) -> None:
        self.equity = equity if isinstance(equity, pd.DataFrame) else pd.DataFrame()
        self.options = options if isinstance(options, pd.DataFrame) else pd.DataFrame()
        self.treasury = treasury if isinstance(treasury, pd.DataFrame) else pd.DataFrame()

    def copy(self) -> "FundHoldings":
        return FundHoldings(
            equity=self.equity.copy() if isinstance(self.equity, pd.DataFrame) else pd.DataFrame(),
            options=self.options.copy() if isinstance(self.options, pd.DataFrame) else pd.DataFrame(),
            treasury=self.treasury.copy()
            if isinstance(self.treasury, pd.DataFrame)
            else pd.DataFrame(),
        )


class FundSnapshot:
    """Container for all holdings data at a point in time."""

    def __init__(
        self,
        *,
        vest: Optional[FundHoldings] = None,
        custodian: Optional[FundHoldings] = None,
        cash: float = 0.0,
        nav: float = 0.0,
        expenses: float = 0.0,
        total_assets: float = 0.0,
        total_net_assets: float = 0.0,
        flows: float = 0.0,
        basket: Optional[pd.DataFrame] = None,
        index: Optional[pd.DataFrame] = None,
        overlap: Optional[pd.DataFrame] = None,
    ) -> None:
        self.vest = vest if isinstance(vest, FundHoldings) else FundHoldings()
        self.custodian = (
            custodian if isinstance(custodian, FundHoldings) else FundHoldings()
        )
        self.cash = float(cash or 0.0)
        self.nav = float(nav or 0.0)
        self.expenses = float(expenses or 0.0)
        self.total_assets = float(total_assets or 0.0)
        self.total_net_assets = float(total_net_assets or 0.0)
        self.total_equity_value = self._compute_equity_value()
        self.total_option_value = self._compute_option_value()
        self.total_option_delta_adjusted_notional = (
            self._compute_option_delta_adjusted_notional()
        )
        self.total_treasury_value = self._compute_treasury_value()
        self.flows = float(flows or 0.0)
        self.basket = basket if isinstance(basket, pd.DataFrame) else pd.DataFrame()
        self.index = index if isinstance(index, pd.DataFrame) else pd.DataFrame()
        self.overlap = (
            overlap if isinstance(overlap, pd.DataFrame) else pd.DataFrame()
        )

    @staticmethod
    def _frame_value_sum(frame: pd.DataFrame, columns: Sequence[str]) -> Optional[float]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None

        for column in columns:
            if column in frame.columns:
                series = pd.to_numeric(frame[column], errors="coerce").dropna()
                if not series.empty:
                    return float(series.sum())
        return None

    @staticmethod
    def _price_quantity_sum(frame: pd.DataFrame, multiplier: float = 1.0) -> Optional[float]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None

        if {"price", "quantity"}.issubset(frame.columns):
            price = pd.to_numeric(frame["price"], errors="coerce").fillna(0.0)
            quantity = pd.to_numeric(frame["quantity"], errors="coerce").fillna(0.0)
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
        for frame in (self.vest.treasury, self.custodian.treasury):
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
        expense_ratio: float = 0.0,
    ) -> None:
        self.current = current if isinstance(current, FundSnapshot) else FundSnapshot()
        self.previous = (
            previous if isinstance(previous, FundSnapshot) else FundSnapshot()
        )
        self.expense_ratio = float(expense_ratio or 0.0)


class Fund:
    def __init__(self, name: str, config: Dict, base_cls=None):
        self.name = name
        self.config = config or {}
        self.base_cls = base_cls
        self._data: Optional[FundData] = FundData()

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
    def vehicle(self) -> Optional[str]:
        return self.config.get("vehicle_wrapper")

    @property
    def is_private_fund(self) -> bool:
        return (self.vehicle or "").lower() == "private_fund"

    @property
    def is_closed_end_fund(self) -> bool:
        return (self.vehicle or "").lower() == "closed_end_fund"

    @property
    def has_equity(self) -> bool:
        return bool(self.config.get("has_equity", True))

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

    def calculate_gain_loss(self, current_date: str, prior_date: str, asset_class: str) -> GainLossResult:
        """
        Calculate gain/loss for specific asset class.
        Used by NAV reconciliation and performance attribution.
        """
        if asset_class == 'equity':
            return self._calculate_equity_gl(current_date, prior_date)
        elif asset_class == 'options':
            return self._calculate_options_gl(current_date, prior_date)