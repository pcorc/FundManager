from typing import Dict, Optional
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
        equity: Optional[pd.DataFrame] = None,
        options: Optional[pd.DataFrame] = None,
        treasury: Optional[pd.DataFrame] = None,
        cash: float = 0.0,
        nav: float = 0.0,
    ) -> None:
        self.equity = equity if isinstance(equity, pd.DataFrame) else pd.DataFrame()
        self.options = options if isinstance(options, pd.DataFrame) else pd.DataFrame()
        self.treasury = (
            treasury if isinstance(treasury, pd.DataFrame) else pd.DataFrame()
        )
        self.cash = float(cash or 0.0)
        self.nav = float(nav or 0.0)


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
        current = self.data.current
        return getattr(current, "cash", 0.0) or 0.0

    @property
    def total_assets(self) -> float:
        current = self.data.current
        nav = getattr(current, "nav", 0.0) or 0.0
        if nav:
            return nav
        return (
            self.total_equity_value
            + self.total_option_value
            + self.total_treasury_value
            + self.cash_value
        )

    @property
    def total_net_assets(self) -> float:
        nav = getattr(self.data.current, "nav", 0.0) or 0.0
        return nav if nav else self.total_assets

    @property
    def total_equity_value(self) -> float:
        equity = getattr(self.data.current, "equity", pd.DataFrame())
        if isinstance(equity, pd.DataFrame) and not equity.empty:
            price_col = "price" if "price" in equity.columns else None
            quantity_col = "quantity" if "quantity" in equity.columns else None
            if price_col and quantity_col:
                return (equity[price_col] * equity[quantity_col]).sum()
            if "market_value" in equity.columns:
                return equity["market_value"].sum()
        return 0.0

    @property
    def total_option_value(self) -> float:
        options = getattr(self.data.current, "options", pd.DataFrame())
        if isinstance(options, pd.DataFrame) and not options.empty:
            if {"price", "quantity"}.issubset(options.columns):
                return (options["price"] * options["quantity"]).sum()
            if "market_value" in options.columns:
                return options["market_value"].sum()
        return 0.0

    @property
    def total_treasury_value(self) -> float:
        treasury = getattr(self.data.current, "treasury", pd.DataFrame())
        if isinstance(treasury, pd.DataFrame) and not treasury.empty:
            if {"price", "quantity"}.issubset(treasury.columns):
                return (treasury["price"] * treasury["quantity"]).sum()
            if "market_value" in treasury.columns:
                return treasury["market_value"].sum()
        return 0.0

    @property
    def expenses(self) -> float:
        nav = getattr(self.data.current, "nav", 0.0) or 0.0
        expense_ratio = getattr(self.data, "expense_ratio", 0.0) or 0.0
        return expense_ratio * nav

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
        return self._copy_dataframe(getattr(self.data.previous, "equity", pd.DataFrame()))

    @property
    def previous_custodian_equity_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(getattr(self.data.previous, "equity", pd.DataFrame()))

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
        return self._copy_dataframe(getattr(self.data.previous, "options", pd.DataFrame()))

    @property
    def previous_custodian_option_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(getattr(self.data.previous, "options", pd.DataFrame()))

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
        return self._copy_dataframe(getattr(self.data.previous, "treasury", pd.DataFrame()))

    @property
    def previous_custodian_treasury_holdings(self) -> pd.DataFrame:
        return self._copy_dataframe(getattr(self.data.previous, "treasury", pd.DataFrame()))

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
        elif asset_class == 'treasury':
            return self._calculate_treasury_gl(current_date, prior_date)

        # Default empty result
        return GainLossResult()

    def _calculate_equity_gl(self, current_date: str, prior_date: str) -> GainLossResult:
        """Calculate equity gain/loss using bulk data"""
        # Get data from self.data (already loaded from bulk store)
        df_current = self.data.current.equity
        df_prior = self.data.previous.equity

        if df_current.empty or df_prior.empty:
            return GainLossResult()

        # Simple merge on ticker
        df_merged = pd.merge(
            df_current, df_prior,
            on='equity_ticker',
            suffixes=('_current', '_prior'),
            how='inner'
        )

        # Calculate gain/loss
        if 'quantity_current' in df_merged.columns and 'price_current' in df_merged.columns:
            df_merged['gl_raw'] = (
                    (df_merged['price_current'] - df_merged.get('price_prior', 0)) *
                    df_merged['quantity_current']
            )

            return GainLossResult(
                raw_gl=df_merged['gl_raw'].sum(),
                adjusted_gl=df_merged['gl_raw'].sum(),  # Add adjustments if needed
                details=df_merged,
                price_adjustments=pd.DataFrame()  # Add if you have price adjustments
            )

        return GainLossResult()

    def _calculate_options_gl(self, current_date: str, prior_date: str) -> GainLossResult:
        """Calculate options gain/loss - similar structure"""
        # Your options-specific logic here
        return GainLossResult()

    def _calculate_treasury_gl(self, current_date: str, prior_date: str) -> GainLossResult:
        """Calculate treasury gain/loss using current vs previous data"""
        df_current = self.data.current.treasury
        df_previous = self.data.previous.treasury

        if df_current.empty or df_previous.empty:
            return GainLossResult()

        # Merge on treasury identifier (CUSIP, ticker, etc.)
        merge_col = 'cusip' if 'cusip' in df_current.columns else 'ticker'

        if merge_col not in df_current.columns or merge_col not in df_previous.columns:
            return GainLossResult()

        df_merged = pd.merge(
            df_current, df_previous,
            on=merge_col,
            suffixes=('_current', '_previous'),
            how='inner'
        )

        # Calculate gain/loss
        if 'quantity_current' in df_merged.columns and 'price_current' in df_merged.columns:
            df_merged['gl_raw'] = (
                    (df_merged['price_current'] - df_merged.get('price_previous', 0)) *
                    df_merged['quantity_current']
            )

            return GainLossResult(
                raw_gl=df_merged['gl_raw'].sum(),
                adjusted_gl=df_merged['gl_raw'].sum(),  # Add accrued interest adjustments if needed
                details=df_merged,
                price_adjustments=pd.DataFrame()  # Add treasury-specific adjustments
            )

        return GainLossResult()