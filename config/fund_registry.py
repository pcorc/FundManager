# config/fund_registry.py
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd


@dataclass
class FundConfig:
    """Complete fund configuration from fund_recon_mappings"""
    name: str
    custodian_equity_table: Optional[str]
    custodian_option_table: Optional[str]
    custodian_treasury_table: Optional[str]
    vest_equity_table: str
    vest_option_table: str
    vest_treasury_table: str
    custodian_nav_table: str
    cash_table: str
    index_table: Optional[str]
    expense_ratio: float
    option_roll_tenor: str
    # ... other fields


class FundRegistry:
    """Manages fund configurations and mappings"""

    def __init__(self, session, base_cls):
        self.session = session
        self.base_cls = base_cls
        self._configs: Dict[str, FundConfig] = {}
        self._load_configs()

    def _load_configs(self):
        """Load configurations from fund_recon_mappings table"""
        fund_recon_mappings = self.base_cls.classes.fund_recon_mappings
        master_accounts = self.base_cls.classes.master_accounts

        query = self.session.query(fund_recon_mappings).join(
            master_accounts, master_accounts.Fund_Ticker == fund_recon_mappings.fund
        ).filter(
            master_accounts.EOD_Report_Strategy.in_(['TIF', 'CEF', 'PF'])
        )

        df = pd.read_sql(query.statement, self.session.bind)

        for _, row in df.iterrows():
            self._configs[row['fund']] = FundConfig(
                name=row['fund'],
                custodian_equity_table=row['custodian_equity_holdings'],
                custodian_option_table=row['custodian_option_holdings'],
                custodian_treasury_table=row['custodian_treasury_holdings'],
                vest_equity_table=row['vest_equity_holdings'],
                vest_option_table=row['vest_options_holdings'],
                vest_treasury_table=row['vest_treasury_holdings'],
                custodian_nav_table=row['custodian_navs'],
                cash_table=row['cash_table'],
                index_table=row['index_holdings'],
                expense_ratio=row['expense_ratio'],
                option_roll_tenor=row['option_roll_tenor']
            )

    def get_config(self, fund_name: str) -> FundConfig:
        """Get configuration for specific fund"""
        return self._configs.get(fund_name)

    def get_all_funds(self) -> Dict[str, FundConfig]:
        """Get all fund configurations"""
        return self._configs.copy()

