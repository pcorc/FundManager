# processing/data_loader.py
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FundData:
    """Container for all fund data needed for compliance"""
    fund_name: str
    date: str
    analysis_type: str
    vest_equity_holdings: pd.DataFrame = None
    vest_option_holdings: pd.DataFrame = None
    vest_treasury_holdings: pd.DataFrame = None
    custodian_equity_holdings: pd.DataFrame = None
    custodian_option_holdings: pd.DataFrame = None
    custodian_treasury_holdings: pd.DataFrame = None
    index_holdings: pd.DataFrame = None
    nav_data: pd.DataFrame = None
    cash_data: pd.DataFrame = None
    weight_analysis: Optional[Dict] = None

    def is_empty(self) -> bool:
        """Check if we have any data"""
        return all([
            self.vest_equity_holdings is None or self.vest_equity_holdings.empty,
            self.custodian_equity_holdings is None or self.custodian_equity_holdings.empty,
            self.nav_data is None or self.nav_data.empty
        ])


class DataLoader:
    """Loads all required data for a fund"""

    def __init__(self, data_access, fund_registry):
        self.data_access = data_access
        self.fund_registry = fund_registry

    def load_fund_data(self, fund_name: str, date: str, analysis_type: str) -> FundData:
        """Load all data needed for compliance analysis"""
        fund_data = FundData(
            fund_name=fund_name,
            date=date,
            analysis_type=analysis_type
        )

        # Load Vest OMS data
        fund_data.vest_equity_holdings = self.data_access.get_holdings(
            fund_name, date, 'vest', 'equity'
        )
        fund_data.vest_option_holdings = self.data_access.get_holdings(
            fund_name, date, 'vest', 'options'
        )
        fund_data.vest_treasury_holdings = self.data_access.get_holdings(
            fund_name, date, 'vest', 'treasury'
        )

        # Load Custodian data
        fund_data.custodian_equity_holdings = self.data_access.get_holdings(
            fund_name, date, 'custodian', 'equity'
        )
        fund_data.custodian_option_holdings = self.data_access.get_holdings(
            fund_name, date, 'custodian', 'options'
        )
        fund_data.custodian_treasury_holdings = self.data_access.get_holdings(
            fund_name, date, 'custodian', 'treasury'
        )

        # Load Index data
        fund_data.index_holdings = self.data_access.get_index_weights(fund_name, date)

        # Load NAV and Cash
        fund_data.nav_data = self.data_access.get_nav(fund_name, date)
        fund_data.cash_data = self.data_access.get_cash(fund_name, date)

        return fund_data