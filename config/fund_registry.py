# registry/fund_registry.py
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
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
    index_identifier: Optional[str]

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
        account_numbers = self._load_account_numbers(session, base_cls)

        # Create FundClass instances for each fund
        for _, row in df.iterrows():
            mapping_data = self._prepare_mapping_data(row, account_numbers)
            fund_name = mapping_data.get('fund')
            if not fund_name:
                continue
            fund = FundClass(
                name=fund_name,
                custodian_equity_table=mapping_data.get('custodian_equity_holdings'),
                custodian_option_table=mapping_data.get('custodian_option_holdings'),
                custodian_treasury_table=mapping_data.get('custodian_treasury_holdings'),
                custodian_nav_table=mapping_data.get('custodian_navs'),
                cash_table=mapping_data.get('cash_table'),
                vest_equity_table=mapping_data.get('vest_equity_holdings'),
                vest_option_table=mapping_data.get('vest_options_holdings'),
                vest_treasury_table=mapping_data.get('vest_treasury_holdings'),
                index_table=mapping_data.get('index_holdings'),
                basket_table=mapping_data.get('basket'),
                flows_table=mapping_data.get('flows'),
                index_identifier=_get_index_identifier(row),
                expense_ratio=float(mapping_data.get('expense_ratio', 0.0) or 0.0),
                option_roll_tenor=mapping_data.get('option_roll_tenor'),
                overwrite_rate=float(mapping_data.get('overwrite', 0.0) or 0.0),
                sg_holdings_table=mapping_data.get('sg_custodian_holdings'),
                option_custodian_assignment=mapping_data.get('option_custodian_assignment'),
                mapping_data=mapping_data,
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

    def _prepare_mapping_data(self, row: pd.Series, account_numbers: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Normalise raw mapping rows and enrich with custodian account numbers."""

        mapping: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, str):
                trimmed = value.strip()
                mapping[key] = trimmed if trimmed and trimmed.upper() != 'NULL' else None
            elif pd.isna(value):
                mapping[key] = None
            else:
                mapping[key] = value

        fund_name = mapping.get('fund')
        if isinstance(fund_name, str):
            numbers = account_numbers.get(fund_name)
            if numbers:
                mapping.setdefault('account_numbers', numbers)
                custodian_number = mapping.get('account_number_custodian')
                if not custodian_number:
                    custodian_number = self._derive_custodian_account_number(numbers)
                    if custodian_number:
                        mapping['account_number_custodian'] = custodian_number

        return mapping

    def _load_account_numbers(self, session, base_cls) -> Dict[str, Dict[str, Any]]:
        """Collect all account numbers keyed by fund for quick lookup."""

        account_numbers_tbl = getattr(base_cls.classes, 'account_numbers', None)
        if account_numbers_tbl is None:
            return {}

        query = session.query(
            account_numbers_tbl.fund,
            account_numbers_tbl.account_type,
            account_numbers_tbl.service_provider,
            account_numbers_tbl.account_number,
        )

        df_accounts = pd.read_sql(query.statement, session.bind)
        account_mapping: Dict[str, Dict[str, Any]] = {}

        for _, account_row in df_accounts.iterrows():
            fund = account_row.get('fund')
            if not isinstance(fund, str):
                continue
            fund_key = fund.strip()
            if not fund_key:
                continue

            account_number = account_row.get('account_number')
            if pd.isna(account_number):
                continue
            account_number_str = str(account_number).strip()
            if not account_number_str:
                continue

            account_type = account_row.get('account_type')
            service_provider = account_row.get('service_provider')
            account_type_key = str(account_type).strip().lower() if isinstance(account_type, str) else None
            provider_key = str(service_provider).strip().lower() if isinstance(service_provider, str) else None

            fund_numbers = account_mapping.setdefault(fund_key, {})
            if provider_key == 'sg' and account_type_key != 'collateral':
                accounts = fund_numbers.setdefault('sg', [])
                if account_number_str not in accounts:
                    accounts.append(account_number_str)
                continue

            key = account_type_key or provider_key or 'other'
            if key in fund_numbers and isinstance(fund_numbers[key], list):
                if account_number_str not in fund_numbers[key]:
                    fund_numbers[key].append(account_number_str)
            elif key in fund_numbers and fund_numbers[key] != account_number_str:
                # Preserve multiple values by storing them in a list
                existing = fund_numbers[key]
                values = existing if isinstance(existing, list) else [existing]
                if account_number_str not in values:
                    values.append(account_number_str)
                fund_numbers[key] = values
            else:
                fund_numbers[key] = account_number_str

        return account_mapping

    @staticmethod
    def _derive_custodian_account_number(numbers: Dict[str, Any]) -> Optional[str]:
        """Determine the most appropriate custodian account number."""

        if not numbers:
            return None

        priority_keys = [
            'account_number_custodian',
            'custodian',
            'primary',
            'account',
        ]

        for key in priority_keys:
            value = numbers.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, list) and value:
                return value[0]

        sg_accounts = numbers.get('sg')
        if isinstance(sg_accounts, list) and sg_accounts:
            return sg_accounts[0]

        for key, value in numbers.items():
            if key == 'collateral':
                continue
            if isinstance(value, str) and value:
                return value
            if isinstance(value, list) and value:
                return value[0]
        return None

def _get_index_identifier(row: pd.Series) -> Optional[str]:
    """Extract an index identifier from registry metadata when provided."""
    for key in (
        'index_identifier',
        'index_fund_code',
        'index_ticker',
        'index_name',
        'account_number_index',
    ):
        value = row.get(key)
        if isinstance(value, str) and value and value != 'NULL':
            return value
    return None
