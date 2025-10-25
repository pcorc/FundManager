# registry/fund_registry.py
from dataclasses import dataclass
from typing import Dict, Optional, List
import pandas as pd
from config.database import or_


@dataclass
class FundClass:
    """Complete fund configuration and properties"""
    name: str
    # Custodian data sources
    custodian_equity_table: Optional[str]
    custodian_option_table: Optional[str]
    custodian_treasury_table: Optional[str]
    custodian_nav_table: str
    cash_table: str

    # Vest OMS data sources
    vest_equity_table: str
    vest_option_table: str
    vest_treasury_table: str

    # Reference data
    index_table: Optional[str]
    basket_table: Optional[str]
    flows_table: Optional[str]

    # Operational parameters
    expense_ratio: float
    option_roll_tenor: str
    overwrite_rate: float
    sg_holdings_table: str
    option_custodian_assignment: Optional[str]

    # Raw mapping data for reference
    mapping_data: Dict

    def __post_init__(self):
        """Set derived properties"""
        self.custodian_type = self._determine_custodian_type()
        self.fund_strategy = self._determine_strategy()

    def _determine_custodian_type(self) -> str:
        """Determine custodian from table patterns"""
        if self.custodian_equity_table:
            if 'bny' in self.custodian_equity_table.lower():
                return 'bny'
            elif 'umb' in self.custodian_equity_table.lower():
                return 'umb'
            elif 'socgen' in self.custodian_equity_table.lower():
                return 'socgen'
        return 'mixed'

    def _determine_strategy(self) -> str:
        """Determine fund strategy from configuration"""
        if self.custodian_type == 'umb' and not self.basket_table:
            return 'CEF'  # Closed-End Fund
        elif self.custodian_type == 'socgen' and not self.flows_table:
            return 'PF'  # Private Fund
        else:
            return 'TIF'  # Traditional Investment Fund

    def get_required_tables(self) -> List[str]:
        """Get all tables required for this fund"""
        tables = []

        # Custodian tables
        if self.custodian_equity_table and self.custodian_equity_table != 'NULL':
            tables.append(self.custodian_equity_table)
        if self.custodian_option_table and self.custodian_option_table != 'NULL':
            tables.append(self.custodian_option_table)
        if self.custodian_treasury_table and self.custodian_treasury_table != 'NULL':
            tables.append(self.custodian_treasury_table)
        if self.custodian_nav_table and self.custodian_nav_table != 'NULL':
            tables.append(self.custodian_nav_table)
        if self.cash_table and self.cash_table != 'NULL':
            tables.append(self.cash_table)

        # Vest tables
        if self.vest_equity_table and self.vest_equity_table != 'NULL':
            tables.append(self.vest_equity_table)
        if self.vest_option_table and self.vest_option_table != 'NULL':
            tables.append(self.vest_option_table)
        if self.vest_treasury_table and self.vest_treasury_table != 'NULL':
            tables.append(self.vest_treasury_table)

        # Reference tables
        if self.index_table and self.index_table != 'NULL':
            tables.append(self.index_table)
        if self.basket_table and self.basket_table != 'NULL':
            tables.append(self.basket_table)
        if self.flows_table and self.flows_table != 'NULL':
            tables.append(self.flows_table)

        return tables

    @property
    def has_options(self) -> bool:
        """Whether this fund trades options"""
        return bool(self.custodian_option_table and self.custodian_option_table != 'NULL')

    @property
    def is_etf(self) -> bool:
        """Whether this is an ETF"""
        return bool(self.basket_table and self.flows_table)


class FundRegistry:
    """Central registry for all fund configurations"""

    def __init__(self):
        self.funds: Dict[str, FundClass] = {}

    @classmethod
    def from_database(cls, session, base_cls) -> 'FundRegistry':
        """Create registry from database"""
        registry = cls()
        registry._load_from_database(session, base_cls)
        return registry

    def _load_from_database(self, session, base_cls):
        """Load fund configurations from database"""
        fund_recon_mappings = base_cls.classes.fund_recon_mappings
        master_accounts = base_cls.classes.master_accounts

        # Filter for relevant fund strategies
        filter_conditions = or_(
            master_accounts.EOD_Report_Strategy == 'TIF',
            master_accounts.EOD_Report_Strategy == 'CEF',
            master_accounts.EOD_Report_Strategy == 'PF'
        )

        query = session.query(fund_recon_mappings).join(
            master_accounts, master_accounts.Fund_Ticker == fund_recon_mappings.fund
        ).filter(filter_conditions)

        df = pd.read_sql(query.statement, session.bind)

        # Create FundClass instances for each fund
        for _, row in df.iterrows():
            fund = FundClass(
                name=row['fund'],
                custodian_equity_table=row['custodian_equity_holdings'],
                custodian_option_table=row['custodian_option_holdings'],
                custodian_treasury_table=row['custodian_treasury_holdings'],
                custodian_nav_table=row['custodian_navs'],
                cash_table=row['cash_table'],
                vest_equity_table=row['vest_equity_holdings'],
                vest_option_table=row['vest_options_holdings'],
                vest_treasury_table=row['vest_treasury_holdings'],
                index_table=row['index_holdings'],
                basket_table=row.get('basket'),
                flows_table=row.get('flows'),
                expense_ratio=float(row['expense_ratio']),
                option_roll_tenor=row['option_roll_tenor'],
                overwrite_rate=float(row.get('overwrite', 0.0)),
                sg_holdings_table=row['sg_custodian_holdings'],
                option_custodian_assignment=row.get('option_custodian_assignment'),
                mapping_data=row.to_dict()
            )
            self.funds[fund.name] = fund

    def get_fund(self, fund_name: str) -> Optional[FundClass]:
        """Get fund by name"""
        return self.funds.get(fund_name)

    def get_all_funds(self) -> Dict[str, FundClass]:
        """Get all funds"""
        return self.funds.copy()

    def get_funds_by_custodian(self, custodian_type: str) -> List[FundClass]:
        """Get funds by custodian type"""
        return [f for f in self.funds.values() if f.custodian_type == custodian_type]

    def get_funds_by_strategy(self, strategy: str) -> List[FundClass]:
        """Get funds by strategy"""
        return [f for f in self.funds.values() if f.fund_strategy == strategy]

    def get_required_tables(self) -> List[str]:
        """Get all unique tables needed across all funds"""
        all_tables = set()
        for fund in self.funds.values():
            all_tables.update(fund.get_required_tables())
        return list(all_tables)