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


class FundSnapshot:
    """Container for all holdings data at a point in time."""

    def __init__(
        self,
        *,
        vest_equity: Optional[pd.DataFrame] = None,
        vest_options: Optional[pd.DataFrame] = None,
        vest_treasury: Optional[pd.DataFrame] = None,
        custodian_equity: Optional[pd.DataFrame] = None,
        custodian_option: Optional[pd.DataFrame] = None,
        custodian_treasury: Optional[pd.DataFrame] = None,
        cash: float = 0.0,
        nav: float = 0.0,
        total_assets: float = 0.0,
        total_net_assets: float = 0.0,
    ) -> None:
        self.equity = vest_equity if isinstance(vest_equity, pd.DataFrame) else pd.DataFrame()
        self.options = vest_options if isinstance(vest_options, pd.DataFrame) else pd.DataFrame()
        self.treasury = (
            vest_treasury if isinstance(vest_treasury, pd.DataFrame) else pd.DataFrame()
        )
        self.custodian_equity = (
            custodian_equity if isinstance(custodian_equity, pd.DataFrame) else pd.DataFrame()
        )
        self.custodian_option = (
            custodian_option if isinstance(custodian_option, pd.DataFrame) else pd.DataFrame()
        )
        self.custodian_treasury = (
            custodian_treasury if isinstance(custodian_treasury, pd.DataFrame) else pd.DataFrame()
        )
        self.cash = float(cash or 0.0)
        self.nav = float(nav or 0.0)
        self.total_assets = float(total_assets or 0.0)
        self.total_net_assets = float(total_net_assets or 0.0)
        self.total_equity_value = self._compute_equity_value()
        self.total_option_value = self._compute_option_value()
        self.total_treasury_value = self._compute_treasury_value()

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
        for frame in (self.equity, self.custodian_equity):
            value = self._frame_value_sum(frame, ["market_value", "net_market_value"])
            if value is not None:
                return value
            fallback = self._price_quantity_sum(frame)
            if fallback is not None:
                return fallback
        return 0.0

    def _compute_option_value(self) -> float:
        for frame in (self.options, self.custodian_option):
            value = self._frame_value_sum(
                frame, ["option_market_value", "market_value", "net_market_value"]
            )
            if value is not None:
                return value
            fallback = self._price_quantity_sum(frame, multiplier=100.0)
            if fallback is not None:
                return fallback
        return 0.0

    def _compute_treasury_value(self) -> float:
        for frame in (self.treasury, self.custodian_treasury):
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
        flows: float = 0.0,
        expense_ratio: float = 0.0,
        basket: Optional[pd.DataFrame] = None,
        index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.current = current if isinstance(current, FundSnapshot) else FundSnapshot()
        self.previous = (
            previous if isinstance(previous, FundSnapshot) else FundSnapshot()
        )
        self.flows = float(flows or 0.0)
        self.expense_ratio = float(expense_ratio or 0.0)
        self.basket = basket if isinstance(basket, pd.DataFrame) else pd.DataFrame()
        self.index = index if isinstance(index, pd.DataFrame) else pd.DataFrame()


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
    def is_private_fund(self) -> bool:
        return self.config.get('strategy') == 'PF'  # Adjust based on your config

    @property
    def total_option_delta_adjusted_notional(self) -> float:
        """For compliance checks that need delta-adjusted option values"""
        options = getattr(self.data.current, "options", pd.DataFrame())
        if (
            isinstance(options, pd.DataFrame)
            and not options.empty
            and "delta_adjusted_notional" in options.columns
        ):
            return options["delta_adjusted_notional"].sum()
        return 0.0

    def get_dividends(self, analysis_date: str) -> float:
        """Return dividend impact for the requested analysis date."""

        dividends = getattr(self.data, "dividends", 0.0)
        return self._extract_numeric_value(dividends, ["dividend", "dividends", "amount", "value"])

    def get_expenses(self, analysis_date: str) -> float:
        """Return total expenses using stored expense ratio and NAV."""

        expenses = getattr(self.data, "expenses", None)
        if expenses not in (None, ""):
            try:
                return float(expenses)
            except (TypeError, ValueError):
                pass
        try:
            return float(self.expenses)
        except Exception:
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

        flows = getattr(self.data, "flows", 0.0)
        return float(flows or 0.0)


    # Method stubs that your services might call
    def get_irs_holdings_data(self) -> pd.DataFrame:
        """Return data formatted for IRS checks"""
        return self.data.current.equity

    def get_40act_holdings_data(self) -> pd.DataFrame:
        """Return data formatted for 40 Act checks"""
        return self.data.current.equity


    # ------------------------------------------------------------------
    # DataFrame accessors used by reconciliation services
    # ------------------------------------------------------------------
    @property
    def equity_holdings(self) -> pd.DataFrame:
        vest = getattr(self.data, "vest_equity", None)
        if isinstance(vest, pd.DataFrame) and not vest.empty:
            return self._copy_dataframe(vest)
        return self._copy_dataframe(getattr(self.data.current, "equity", pd.DataFrame()))

    @property
    def custodian_equity_holdings(self) -> pd.DataFrame:
        custodian = getattr(self.data, "custodian_equity", None)
        if isinstance(custodian, pd.DataFrame) and not custodian.empty:
            return self._copy_dataframe(custodian)
        return self._copy_dataframe(getattr(self.data.current, "equity", pd.DataFrame()))

    @property
    def previous_equity_holdings(self) -> pd.DataFrame:
        vest_prev = getattr(self.data, "vest_equity_t1", None)
        if isinstance(vest_prev, pd.DataFrame) and not vest_prev.empty:
            return self._copy_dataframe(vest_prev)
        return self._copy_dataframe(getattr(self.data.previous, "equity", pd.DataFrame()))

    @property
    def previous_custodian_equity_holdings(self) -> pd.DataFrame:
        cust_prev = getattr(self.data, "previous_custodian_equity", None)
        if isinstance(cust_prev, pd.DataFrame) and not cust_prev.empty:
            return self._copy_dataframe(cust_prev)
        return pd.DataFrame()

    @property
    def options_holdings(self) -> pd.DataFrame:
        vest = getattr(self.data, "vest_option", None)
        if isinstance(vest, pd.DataFrame) and not vest.empty:
            return self._copy_dataframe(vest)
        return self._copy_dataframe(getattr(self.data.current, "options", pd.DataFrame()))

    @property
    def custodian_option_holdings(self) -> pd.DataFrame:
        custodian = getattr(self.data, "custodian_option", None)
        if isinstance(custodian, pd.DataFrame) and not custodian.empty:
            return self._copy_dataframe(custodian)
        return self._copy_dataframe(getattr(self.data.current, "options", pd.DataFrame()))

    @property
    def previous_options_holdings(self) -> pd.DataFrame:
        vest_prev = getattr(self.data, "vest_option_t1", None)
        if isinstance(vest_prev, pd.DataFrame) and not vest_prev.empty:
            return self._copy_dataframe(vest_prev)
        return self._copy_dataframe(getattr(self.data.previous, "options", pd.DataFrame()))

    @property
    def previous_custodian_option_holdings(self) -> pd.DataFrame:
        cust_prev = getattr(self.data, "previous_custodian_option", None)
        if isinstance(cust_prev, pd.DataFrame) and not cust_prev.empty:
            return self._copy_dataframe(cust_prev)
        return pd.DataFrame()

    @property
    def treasury_holdings(self) -> pd.DataFrame:
        vest = getattr(self.data, "vest_treasury", None)
        if isinstance(vest, pd.DataFrame) and not vest.empty:
            return self._copy_dataframe(vest)
        return self._copy_dataframe(getattr(self.data.current, "treasury", pd.DataFrame()))

    @property
    def custodian_treasury_holdings(self) -> pd.DataFrame:
        custodian = getattr(self.data, "custodian_treasury", None)
        if isinstance(custodian, pd.DataFrame) and not custodian.empty:
            return self._copy_dataframe(custodian)
        return self._copy_dataframe(getattr(self.data.current, "treasury", pd.DataFrame()))

    @property
    def previous_treasury_holdings(self) -> pd.DataFrame:
        vest_prev = getattr(self.data, "vest_treasury_t1", None)
        if isinstance(vest_prev, pd.DataFrame) and not vest_prev.empty:
            return self._copy_dataframe(vest_prev)
        return self._copy_dataframe(getattr(self.data.previous, "treasury", pd.DataFrame()))

    @property
    def previous_custodian_treasury_holdings(self) -> pd.DataFrame:
        cust_prev = getattr(self.data, "previous_custodian_treasury", None)
        if isinstance(cust_prev, pd.DataFrame) and not cust_prev.empty:
            return self._copy_dataframe(cust_prev)
        return pd.DataFrame()

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
        index_df = getattr(self.data, "index", pd.DataFrame())
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