from dataclasses import dataclass, field
from typing import Dict, Optional, List
import pandas as pd


@dataclass
class FundHoldings:
    """Container for all holdings data"""
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    options: pd.DataFrame = field(default_factory=pd.DataFrame)
    treasury: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash: float = 0.0
    nav: float = 0.0


@dataclass
class FundData:
    """Complete fund data for T and T-1"""
    current: FundHoldings = field(default_factory=FundHoldings)
    previous: FundHoldings = field(default_factory=FundHoldings)
    # Additional data
    flows: float = 0.0
    expense_ratio: float = 0.0
    basket: pd.DataFrame = field(default_factory=pd.DataFrame)
    index: pd.DataFrame = field(default_factory=pd.DataFrame)


# domain/fund.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
from datetime import date


@dataclass
class Holdings:
    """Value object for holdings data"""
    equities: pd.DataFrame
    options: pd.DataFrame
    treasuries: pd.DataFrame
    cash: float

    @property
    def total_market_value(self) -> float:
        return (self.equities.get('market_value', 0).sum() +
                self.options.get('market_value', 0).sum() +
                self.treasuries.get('market_value', 0).sum() +
                self.cash)



class Fund(ABC):
    """Abstract base class for all funds"""

    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self._data_loader = None

    @abstractmethod
    def get_snapshot(self, date: date) -> FundSnapshot:
        """Get complete fund snapshot for date"""
        pass

    @abstractmethod
    def get_prior_snapshot(self, date: date) -> FundSnapshot:
        """Get previous period snapshot"""
        pass

    def reconcile_holdings(self, date: date) -> 'ReconciliationResult':
        """Run holdings reconciliation"""
        return HoldingsReconciler(self).reconcile(date)

    def check_compliance(self, date: date) -> 'ComplianceResult':
        """Run compliance checks"""
        return ComplianceChecker(self).check(date)

    def reconcile_nav(self, date: date) -> 'NAVReconciliationResult':
        """Run NAV reconciliation"""
        return NAVReconciler(self).reconcile(date)



class Fund:
    def __init__(self, name: str, config: Dict, base_cls=None):
        self.name = name
        self.config = config
        self.base_cls = base_cls
        self.data = FundData()

    def load_data(self, session, date: date_class) -> 'Fund':
        """Load all data for this fund"""
        data_loader = FundDataLoader(self, session, date)
        self.data = data_loader.load_all_data()
        return self

    # Property accessors
    @property
    def nav(self) -> float:
        return self.data.current.nav

    @property
    def equity_holdings(self) -> pd.DataFrame:
        return self.data.current.equity

    @property
    def total_equity_value(self) -> float:
        if not self.equity_holdings.empty:
            return (self.equity_holdings['price'] * self.equity_holdings['quantity']).sum()
        return 0.0

    # Business logic properties
    @property
    def uses_block_trading(self) -> bool:
        return self.config.get('uses_block_trading', False)

    @property
    def custodian_type(self) -> str:
        return self.config.get('custodian')

    def calculate_gain_loss(self, date: str, prior_date: str, asset_class: str) -> GainLossResult:
        """
        Calculate gain/loss for specific asset class.
        This can be used by NAV recon, performance attribution, etc.
        """
        if asset_class == 'equity':
            return self._calculate_equity_gl(date, prior_date)
        elif asset_class == 'options':
            return self._calculate_options_gl(date, prior_date)
        elif asset_class == 'treasury':
            return self._calculate_treasury_gl(date, prior_date)
        elif asset_class == 'flex_options':
            return self._calculate_flex_options_gl(date, prior_date)

        return GainLossResult(0.0, 0.0, pd.DataFrame(), pd.DataFrame())

    def _calculate_equity_gl(self, date: str, prior_date: str) -> GainLossResult:
        """Calculate equity gain/loss - moved from NAV recon"""
        # Get current and prior data
        current_request = DataRequest('holdings', date, self.name, asset_class='equity')
        prior_request = DataRequest('holdings', prior_date, self.name, asset_class='equity')

        df_current = self.data_access.get_data(current_request)
        df_prior = self.data_access.get_data(prior_request)

        if df_current.empty or df_prior.empty:
            return GainLossResult(0.0, 0.0, pd.DataFrame(), pd.DataFrame())

        # Merge and calculate G/L
        df_merged = self._merge_holdings_data(df_current, df_prior, 'equity_ticker')

        # Calculate raw G/L
        quantity_col = self._get_quantity_column(df_merged)
        df_merged['gl_raw'] = (df_merged['price_t'] - df_merged['price_t1']) * df_merged[quantity_col]

        # Apply price adjustments
        df_adjusted = self._apply_price_adjustments(df_merged, 'equity')
        df_adjusted['gl_adj'] = (df_adjusted['price_t_adj'] - df_adjusted['price_t1_adj']) * df_adjusted[quantity_col]

        return GainLossResult(
            raw_gl=df_merged['gl_raw'].sum(),
            adjusted_gl=df_adjusted['gl_adj'].sum(),
            details=df_adjusted,
            price_adjustments=self._get_price_adjustments('equity')
        )

    def _calculate_options_gl(self, date: str, prior_date: str) -> GainLossResult:
        """Calculate options gain/loss - moved from NAV recon"""
        # Similar structure to equity but with option-specific logic
        pass

    def get_holdings_reconciliation_data(self, date: str, reconciliation_type: str) -> pd.DataFrame:
        """
        Get normalized data for reconciliation.
        This replaces the complex data merging in Reconciliator.
        """
        request = DataRequest('holdings', date, self.name, asset_class=reconciliation_type)
        return self.data_access.get_data(request)