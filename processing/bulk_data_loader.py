# processing/bulk_data_loader.py
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Set
from datetime import date
import logging


@dataclass
class BulkDataStore:
    """Central storage for ALL fund data - no more SQL calls after initialization"""
    date: str
    # Data organized by [fund_name][data_type] -> DataFrame
    fund_data: Dict[str, Dict[str, pd.DataFrame]] = None
    # Metadata about what we loaded
    loaded_funds: Set[str] = None
    loaded_data_types: Set[str] = None

    def __post_init__(self):
        self.fund_data = {}
        self.loaded_funds = set()
        self.loaded_data_types = set()


class BulkDataLoader:
    """Loads ALL data for ALL funds in minimal SQL calls"""

    def __init__(self, session, base_cls, fund_registry):
        self.session = session
        self.base_cls = base_cls
        self.fund_registry = fund_registry
        self.logger = logging.getLogger(__name__)

    def load_all_data_for_date(self, target_date: str) -> BulkDataStore:
        """ONE-TIME BULK LOAD: Get everything we need for all funds"""
        data_store = BulkDataStore(date=target_date)

        # Get all funds from registry
        all_funds = self.fund_registry.funds

        # Bulk load by table type (not by fund)
        self._bulk_load_custodian_holdings(data_store, all_funds, target_date)
        self._bulk_load_vest_holdings(data_store, all_funds, target_date)
        self._bulk_load_nav_data(data_store, all_funds, target_date)
        self._bulk_load_cash_data(data_store, all_funds, target_date)
        self._bulk_load_index_data(data_store, all_funds, target_date)

        self.logger.info(f"Bulk loaded data for {len(data_store.loaded_funds)} funds")
        return data_store

    def _bulk_load_custodian_holdings(self, data_store: BulkDataStore, all_funds: Dict, target_date: str):
        """Load ALL custodian holdings for ALL funds in one query per table"""
        # Group funds by custodian table (BNY, UMB, etc.)
        table_to_funds = self._group_funds_by_table(all_funds, 'custodian_equity_holdings')

        for table_name, funds in table_to_funds.items():
            if not table_name or table_name == 'NULL':
                continue

            # ONE QUERY: Get all data for this table/date
            try:
                table = getattr(self.base_cls.classes, table_name)
                query = self.session.query(table).filter(table.date == target_date)
                all_data = pd.read_sql(query.statement, self.session.bind)

                # Distribute data to appropriate funds
                for fund_name in funds:
                    fund_data = all_data[all_data['fund'] == fund_name].copy()
                    self._store_fund_data(data_store, fund_name, 'custodian_equity', fund_data)

            except Exception as e:
                self.logger.warning(f"Failed to load {table_name}: {e}")

    def _bulk_load_vest_holdings(self, data_store: BulkDataStore, all_funds: Dict, target_date: str):
        """Load ALL Vest holdings for ALL funds in one query"""
        table_to_funds = self._group_funds_by_table(all_funds, 'vest_equity_holdings')

        for table_name, funds in table_to_funds.items():
            if not table_name or table_name == 'NULL':
                continue

            try:
                table = getattr(self.base_cls.classes, table_name)
                query = self.session.query(table).filter(table.date == target_date)
                all_data = pd.read_sql(query.statement, self.session.bind)

                for fund_name in funds:
                    fund_data = all_data[all_data['fund'] == fund_name].copy()
                    self._store_fund_data(data_store, fund_name, 'vest_equity', fund_data)

            except Exception as e:
                self.logger.warning(f"Failed to load {table_name}: {e}")

    def _bulk_load_nav_data(self, data_store: BulkDataStore, all_funds: Dict, target_date: str):
        """Load ALL NAV data for ALL funds in minimal queries"""
        table_to_funds = self._group_funds_by_table(all_funds, 'custodian_navs')

        for table_name, funds in table_to_funds.items():
            if not table_name or table_name == 'NULL':
                continue

            try:
                table = getattr(self.base_cls.classes, table_name)

                # Handle different NAV table structures
                if table_name in ['bny_us_nav_v2', 'bny_vit_nav']:
                    # These tables don't have fund column - get all and distribute
                    query = self.session.query(table).filter(table.date == target_date)
                    all_nav_data = pd.read_sql(query.statement, self.session.bind)
                    # You'll need mapping logic here based on your data structure

                elif table_name == 'umb_cef_nav':
                    # UMB has fund column - query for all relevant funds at once
                    query = self.session.query(table).filter(
                        table.date == target_date,
                        table.fund.in_(list(funds))
                    )
                    all_nav_data = pd.read_sql(query.statement, self.session.bind)

                    for fund_name in funds:
                        fund_nav = all_nav_data[all_nav_data['fund'] == fund_name].copy()
                        self._store_fund_data(data_store, fund_name, 'nav', fund_nav)

            except Exception as e:
                self.logger.warning(f"Failed to load NAV {table_name}: {e}")

    def _bulk_load_cash_data(self, data_store: BulkDataStore, all_funds: Dict, target_date: str):
        """Load ALL cash data - similar to NAV loading"""
        table_to_funds = self._group_funds_by_table(all_funds, 'cash_table')

        for table_name, funds in table_to_funds.items():
            if not table_name or table_name == 'NULL':
                continue

            try:
                table = getattr(self.base_cls.classes, table_name)
                query = self.session.query(table).filter(table.date == target_date)
                all_cash_data = pd.read_sql(query.statement, self.session.bind)

                for fund_name in funds:
                    fund_cash = all_cash_data[all_cash_data['fund'] == fund_name].copy()
                    self._store_fund_data(data_store, fund_name, 'cash', fund_cash)

            except Exception as e:
                self.logger.warning(f"Failed to load cash {table_name}: {e}")

    def _bulk_load_index_data(self, data_store: BulkDataStore, all_funds: Dict, target_date: str):
        """Load ALL index data"""
        table_to_funds = self._group_funds_by_table(all_funds, 'index_holdings')

        for table_name, funds in table_to_funds.items():
            if not table_name or table_name == 'NULL':
                continue

            try:
                table = getattr(self.base_cls.classes, table_name)
                query = self.session.query(table).filter(table.date == target_date)
                all_index_data = pd.read_sql(query.statement, self.session.bind)

                # Index data might not have fund column - handle accordingly
                for fund_name in funds:
                    # Store same index data for all funds that use this index
                    self._store_fund_data(data_store, fund_name, 'index', all_index_data.copy())

            except Exception as e:
                self.logger.warning(f"Failed to load index {table_name}: {e}")

    def _group_funds_by_table(self, all_funds: Dict, config_key: str) -> Dict[str, List[str]]:
        """Group funds by which table they use for a given data type"""
        table_groups = {}
        for fund_name, fund in all_funds.items():
            table_name = fund.mapping_data.get(config_key)
            if table_name not in table_groups:
                table_groups[table_name] = []
            table_groups[table_name].append(fund_name)
        return table_groups

    def _store_fund_data(self, data_store: BulkDataStore, fund_name: str, data_type: str, data: pd.DataFrame):
        """Store data in the central data store"""
        if fund_name not in data_store.fund_data:
            data_store.fund_data[fund_name] = {}

        data_store.fund_data[fund_name][data_type] = data
        data_store.loaded_funds.add(fund_name)
        data_store.loaded_data_types.add(data_type)