# data/unified_data_access.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd


@dataclass
class DataRequest:
    fund_name: str
    table_type: str  # 'custodian_equity', 'vest_equity', 'nav', 'cash', etc.
    date: str
    asset_class: Optional[str] = None  # 'equity', 'options', 'treasury'


class UnifiedDataAccess:
    """Single entry point for all data access - routes to appropriate tables"""

    def __init__(self, session, base_cls, fund_registry: FundRegistry):
        self.session = session
        self.base_cls = base_cls
        self.fund_registry = fund_registry
        self._table_handlers = self._initialize_table_handlers()

    def get_data(self, request: DataRequest) -> pd.DataFrame:
        """Get data for any fund/table combination"""
        config = self.fund_registry.get_config(request.fund_name)
        if not config:
            return pd.DataFrame()

        table_name = self._get_table_name(config, request.table_type)
        if not table_name:
            return pd.DataFrame()

        handler = self._table_handlers.get(request.table_type)
        if handler:
            return handler(table_name, request)

        return self._default_query(table_name, request)

    def _get_table_name(self, config: FundConfig, table_type: str) -> Optional[str]:
        """Map table type to actual table name from config"""
        table_mapping = {
            'custodian_equity': config.custodian_equity_table,
            'custodian_option': config.custodian_option_table,
            'custodian_treasury': config.custodian_treasury_table,
            'vest_equity': config.vest_equity_table,
            'vest_option': config.vest_option_table,
            'vest_treasury': config.vest_treasury_table,
            'nav': config.custodian_nav_table,
            'cash': config.cash_table,
            'index': config.index_table
        }
        return table_mapping.get(table_type)

    def _initialize_table_handlers(self) -> Dict[str, callable]:
        """Initialize specialized handlers for different table types"""
        return {
            'custodian_equity': self._query_custodian_holdings,
            'custodian_option': self._query_custodian_holdings,
            'custodian_treasury': self._query_custodian_holdings,
            'vest_equity': self._query_vest_holdings,
            'vest_option': self._query_vest_holdings,
            'vest_treasury': self._query_vest_holdings,
            'nav': self._query_nav,
            'cash': self._query_cash,
            'index': self._query_index
        }

    def _query_custodian_holdings(self, table_name: str, request: DataRequest) -> pd.DataFrame:
        """Query custodian holdings with appropriate filters"""
        table = getattr(self.base_cls.classes, table_name)

        # Determine asset group filter based on asset_class
        asset_filters = {
            'equity': ~table.asset_group.in_(['O', 'UN', 'ME', 'MM', 'CA', 'TI', 'B']),
            'options': table.asset_group.notin_(['S', 'FS', 'UN', 'ME', 'MM', 'CA', 'B', 'TI']),
            'treasury': table.asset_group == 'TI'
        }

        query = self.session.query(table).filter(
            table.date == request.date,
            table.fund == request.fund_name,
            asset_filters.get(request.asset_class, True)
        )

        return pd.read_sql(query.statement, self.session.bind)

    def _query_vest_holdings(self, table_name: str, request: DataRequest) -> pd.DataFrame:
        """Query Vest OMS holdings"""
        table = getattr(self.base_cls.classes, table_name)

        query = self.session.query(table).filter(
            table.date == request.date,
            table.fund == request.fund_name
        )

        if request.asset_class == 'options':
            # Add option-specific joins if needed
            query = query.join(
                self.base_cls.classes.bbg_options_flds_blotter,
                table.occ_symbol == self.base_cls.classes.bbg_options_flds_blotter.OCC_SYMBOL
            )

        return pd.read_sql(query.statement, self.session.bind)

    def _query_nav(self, table_name: str, request: DataRequest) -> pd.DataFrame:
        """Query NAV data"""
        table = getattr(self.base_cls.classes, table_name)

        # Handle different NAV table structures
        if table_name in ['bny_us_nav_v2', 'bny_vit_nav']:
            query = self.session.query(table).filter(table.date == request.date)
        elif table_name == 'umb_cef_nav':
            query = self.session.query(table).filter(
                table.date == request.date,
                table.fund == request.fund_name
            )
        else:
            query = self.session.query(table).filter(table.date == request.date)

        return pd.read_sql(query.statement, self.session.bind)

    def _default_query(self, table_name: str, request: DataRequest) -> pd.DataFrame:
        """Default query for unspecified table types"""
        table = getattr(self.base_cls.classes, table_name)
        query = self.session.query(table).filter(table.date == request.date)
        return pd.read_sql(query.statement, self.session.bind)

    def get_index_weights(self, fund_name: str, date: str):
        """Get index weights for fund"""
        config = self.fund_registry.get_config(fund_name)
        request = DataRequest(
            fund_name=fund_name,
            data_type='index_weights',
            date=date,
            table_name=config.index_table
        )
        return self.get_data(request)