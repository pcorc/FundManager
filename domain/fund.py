from dataclasses import dataclass, field
from typing import Dict, Optional, List
import pandas as pd

@dataclass
class GainLossResult:
    """Simple container for gain/loss calculations"""
    raw_gl: float = 0.0
    adjusted_gl: float = 0.0
    details: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_adjustments: pd.DataFrame = field(default_factory=pd.DataFrame)


# domain/fund.py - STORE BOTH SOURCES

@dataclass
class FundMetrics:
    """All time-specific financial metrics for a single date"""
    # Holdings data
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    options: pd.DataFrame = field(default_factory=pd.DataFrame)
    treasury: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash: float = 0.0

    # CUSTODIAN-PROVIDED VALUES (source of truth #1)
    custodian_total_assets: float = 0.0
    custodian_total_net_assets: float = 0.0
    custodian_nav_per_share: float = 0.0


@dataclass
class FundData:
    current: FundMetrics = field(default_factory=FundMetrics)
    previous: FundMetrics = field(default_factory=FundMetrics)
    expense_ratio: float = 0.0


class Fund:
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.data = FundData()

    # CUSTODIAN-PROVIDED VALUES (for reconciliation)
    @property
    def custodian_total_assets(self) -> float:
        return self.data.current.custodian_total_assets

    @property
    def custodian_total_net_assets(self) -> float:
        return self.data.current.custodian_total_net_assets

    # YOUR CALCULATED VALUES (for reconciliation)
    @property
    def calculated_total_assets(self) -> float:
        """Calculate total assets from components"""
        return (self.total_equity_value +
                self.total_option_value +
                self.total_treasury_value +
                self.cash_value)

    @property
    def calculated_total_net_assets(self) -> float:
        """Calculate net assets (assets - liabilities)"""
        # This might be the same as total_assets for some funds
        # or might subtract expenses, liabilities, etc.
        return self.calculated_total_assets - self.expenses

    @property
    def total_equity_value(self) -> float:
        if not self.data.current.equity.empty:
            return (self.data.current.equity['price'] * self.data.current.equity['quantity']).sum()
        return 0.0

    @property
    def total_option_value(self) -> float:
        if not self.data.current.options.empty:
            return (self.data.current.options['price'] * self.data.current.options['quantity']).sum()
        return 0.0

    @property
    def total_treasury_value(self) -> float:
        if not self.data.current.treasury.empty:
            return (self.data.current.treasury['price'] * self.data.current.treasury['quantity']).sum()
        return 0.0

    @property
    def cash_value(self) -> float:
        return self.data.current.cash

    @property
    def expenses(self) -> float:
        return self.data.expense_ratio * self.custodian_total_net_assets  # Use custodian as base

    @property
    def total_option_delta_adjusted_notional(self) -> float:
        """For compliance checks that need delta-adjusted option values"""
        if not self.data.current.options.empty and 'delta_adjusted_notional' in self.data.current.options.columns:
            return self.data.current.options['delta_adjusted_notional'].sum()
        return 0.0

    # Method stubs that your services might call
    def get_irs_holdings_data(self) -> pd.DataFrame:
        """Return data formatted for IRS checks"""
        return self.data.current.equity

    def get_40act_holdings_data(self) -> pd.DataFrame:
        """Return data formatted for 40 Act checks"""
        return self.data.current.equity

    def get_gics_exposure(self) -> Dict:
        """Return GICS exposure data"""
        # You'll implement this based on your GICS logic
        return {}

    # ADD THIS METHOD - it's useful!
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