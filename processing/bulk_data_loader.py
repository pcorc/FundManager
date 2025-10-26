# processing/bulk_data_loader.py
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Set
import logging
from sqlalchemy import literal



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
        data_store = BulkDataStore(date=str(target_date))

        # Get all funds from registry
        all_funds = self.fund_registry.funds

        # Bulk load by table type (not by fund)
        self._bulk_load_custodian_holdings(data_store, all_funds, target_date)
        self._bulk_load_vest_holdings(data_store, all_funds, target_date)
        self._bulk_load_nav_data(data_store, all_funds, target_date)
        self._bulk_load_cash_data(data_store, all_funds, target_date)
        self._bulk_load_index_data(data_store, all_funds, target_date)

        # Ensure every fund has an entry even if a data set was empty
        for fund_name in all_funds.keys():
            self._ensure_fund_slot(data_store, fund_name)

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
                date_column = self._get_table_column(table, 'date', 'trade_date', 'business_date')
                query = self.session.query(table)
                if date_column is not None:
                    query = query.filter(date_column == target_date)
                all_data = pd.read_sql(query.statement, self.session.bind)

                # Distribute data to appropriate funds
                for fund in funds:
                    fund_name = fund.name
                    fund_data = self._filter_by_fund(all_data, fund_name, fund.mapping_data)
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
                date_column = self._get_table_column(table, 'date', 'trade_date', 'business_date')
                query = self.session.query(table)
                if date_column is not None:
                    query = query.filter(date_column == target_date)
                all_data = pd.read_sql(query.statement, self.session.bind)

                for fund in funds:
                    fund_name = fund.name
                    fund_data = self._filter_by_fund(all_data, fund_name, fund.mapping_data)
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
                    date_column = self._get_table_column(table, 'date', 'nav_date', 'business_date')
                    query = self.session.query(table)
                    if date_column is not None:
                        query = query.filter(date_column == target_date)
                    all_nav_data = pd.read_sql(query.statement, self.session.bind)
                    for fund in funds:
                        fund_name = fund.name
                        fund_nav = self._filter_by_fund(all_nav_data, fund_name, fund.mapping_data)
                        self._store_fund_data(data_store, fund_name, 'nav', fund_nav)

                elif table_name == 'umb_cef_nav':
                    # UMB has fund column - query for all relevant funds at once
                    date_column = self._get_table_column(table, 'date', 'nav_date', 'business_date')
                    fund_column = self._get_table_column(table, 'fund', 'fund_ticker')
                    query = self.session.query(table)
                    if date_column is not None:
                        query = query.filter(date_column == target_date)
                    if fund_column is not None:
                        query = query.filter(fund_column.in_([fund.name for fund in funds]))
                    all_nav_data = pd.read_sql(query.statement, self.session.bind)

                    for fund in funds:
                        fund_name = fund.name
                        fund_nav = self._filter_by_fund(all_nav_data, fund_name, fund.mapping_data)
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
                date_column = self._get_table_column(table, 'date', 'business_date', 'nav_date')
                query = self.session.query(table)
                if date_column is not None:
                    query = query.filter(date_column == target_date)
                all_cash_data = pd.read_sql(query.statement, self.session.bind)

                for fund in funds:
                    fund_name = fund.name
                    fund_cash = self._filter_by_fund(all_cash_data, fund_name, fund.mapping_data)
                    self._store_fund_data(data_store, fund_name, 'cash', fund_cash)

            except Exception as e:
                self.logger.warning(f"Failed to load cash {table_name}: {e}")

    def _bulk_load_index_data(self, data_store: BulkDataStore, all_funds: Dict, target_date: str):
        """Load ALL index data with provider-specific logic."""
        table_to_funds = self._group_funds_by_table(all_funds, 'index_holdings')

        for table_name, funds in table_to_funds.items():
            if not table_name or table_name == 'NULL':
                continue

            for fund in funds:
                try:
                    fund_index = self._get_index_data(table_name, target_date, fund)
                    self._store_fund_data(data_store, fund.name, 'index', fund_index)
                except Exception as exc:
                    self.logger.warning(
                        "Failed to load index %s for %s: %s", table_name, fund.name, exc
                    )
                    self._store_fund_data(data_store, fund.name, 'index', pd.DataFrame())

    def _get_index_data(self, table_name: str, target_date: str, fund) -> pd.DataFrame:
        """Dispatch to the appropriate index provider query."""
        provider_map = {
            'nasdaq_holdings': self._get_nasdaq_holdings,
            'sp_holdings': self._get_sp_holdings,
            'cboe_holdings': self._get_cboe_holdings,
            'dogg_index': self._get_dogg_index,
        }

        handler = provider_map.get(table_name)
        if handler:
            return handler(target_date, fund)

        # Generic fallback: query table and filter by fund when possible
        table = getattr(self.base_cls.classes, table_name)
        date_column = self._get_table_column(
            table, 'date', 'effective_date', 'business_date', 'trade_date'
        )
        query = self.session.query(table)
        if date_column is not None:
            query = query.filter(date_column == target_date)
        df = pd.read_sql(query.statement, self.session.bind)
        return self._filter_by_fund(df, fund.name, fund.mapping_data)

    def _resolve_index_identifier(self, fund) -> str:
        """Determine which identifier to use when filtering provider data."""
        mapping = fund.mapping_data
        for key in (
            'index_identifier',
            'index_fund_code',
            'index_ticker',
            'index_name',
            'account_number_index',
        ):
            value = mapping.get(key)
            if value and value != 'NULL':
                return value
        return fund.index_identifier or fund.name

    def _get_nasdaq_holdings(self, target_date: str, fund) -> pd.DataFrame:
        nasdaq = getattr(self.base_cls.classes, 'nasdaq_holdings', None)
        if nasdaq is None:
            raise AttributeError('nasdaq_holdings table is not reflected')

        bbg = getattr(self.base_cls.classes, 'bbg_equity_flds_blotter', None)
        columns = [
            nasdaq.date.label('date'),
            nasdaq.fund.label('fund'),
            nasdaq.ticker.label('equity_ticker'),
            nasdaq.index_weight.label('weight_index'),
            nasdaq.price.label('price_index'),
        ]
        if bbg is not None:
            columns.extend(
                [
                    bbg.GICS_SECTOR_NAME,
                    bbg.GICS_INDUSTRY_NAME,
                    bbg.GICS_INDUSTRY_GROUP_NAME,
                ]
            )

        query = self.session.query(*columns)
        if bbg is not None:
            query = query.join(bbg, nasdaq.ticker == bbg.TICKER)
        query = (
            query
            .filter(
                nasdaq.date == target_date,
                nasdaq.time_of_day == 'EOD',
                nasdaq.fund == self._resolve_index_identifier(fund),
            )
            .group_by(nasdaq.ticker)
        )

        return pd.read_sql(query.statement, self.session.bind)

    def _get_sp_holdings(self, target_date: str, fund) -> pd.DataFrame:
        sp = getattr(self.base_cls.classes, 'sp_holdings', None)
        if sp is None:
            raise AttributeError('sp_holdings table is not reflected')

        bbg = getattr(self.base_cls.classes, 'bbg_equity_flds_blotter', None)
        columns = [
            sp.EFFECTIVE_DATE.label('date'),
            sp.INDEX_CODE.label('fund'),
            sp.TICKER.label('equity_ticker'),
            sp.INDEX_WEIGHT.label('weight_index'),
            sp.LOCAL_PRICE.label('price_index'),
        ]
        if bbg is not None:
            columns.extend(
                [
                    bbg.GICS_SECTOR_NAME,
                    bbg.GICS_INDUSTRY_NAME,
                    bbg.GICS_INDUSTRY_GROUP_NAME,
                ]
            )

        query = self.session.query(*columns)
        if bbg is not None:
            query = query.join(bbg, sp.TICKER == bbg.TICKER)
        query = query.filter(
            sp.EFFECTIVE_DATE == target_date,
            sp.INDEX_CODE == self._resolve_index_identifier(fund),
        )

        return pd.read_sql(query.statement, self.session.bind)

    def _get_cboe_holdings(self, target_date: str, fund) -> pd.DataFrame:
        cboe = getattr(self.base_cls.classes, 'cboe_holdings', None)
        if cboe is None:
            raise AttributeError('cboe_holdings table is not reflected')

        bbg = getattr(self.base_cls.classes, 'bbg_equity_flds_blotter', None)
        identifier = self._resolve_index_identifier(fund) or 'SPATI'
        columns = [
            cboe.date.label('date'),
            cboe.index_name.label('fund'),
            cboe.ticker.label('equity_ticker'),
            cboe.stock_weight.label('weight_index'),
            cboe.price.label('price_index'),
        ]
        if bbg is not None:
            columns.extend(
                [
                    bbg.GICS_SECTOR_NAME,
                    bbg.GICS_INDUSTRY_NAME,
                    bbg.GICS_INDUSTRY_GROUP_NAME,
                ]
            )

        query = self.session.query(*columns)
        if bbg is not None:
            query = query.join(bbg, cboe.ticker == bbg.TICKER)
        query = query.filter(
            cboe.date == target_date,
            cboe.index_name == identifier,
        )

        return pd.read_sql(query.statement, self.session.bind)

    def _get_dogg_index(self, target_date: str, fund) -> pd.DataFrame:
        dogg = getattr(self.base_cls.classes, 'dogg_index', None)
        if dogg is None:
            raise AttributeError('dogg_index table is not reflected')

        bbg = getattr(self.base_cls.classes, 'bbg_equity_flds_blotter', None)
        columns = [
            dogg.DATE.label('date'),
            literal(fund.name).label('fund'),
            dogg.TICKER.label('equity_ticker'),
            literal(0.10).label('weight_index'),
        ]
        if bbg is not None:
            columns.extend(
                [
                    bbg.GICS_SECTOR_NAME,
                    bbg.GICS_INDUSTRY_NAME,
                    bbg.GICS_INDUSTRY_GROUP_NAME,
                ]
            )

        query = self.session.query(*columns)
        if bbg is not None:
            query = query.join(bbg, dogg.TICKER == bbg.TICKER)
        query = query.filter(dogg.DATE == target_date)

        return pd.read_sql(query.statement, self.session.bind)

    def _group_funds_by_table(self, all_funds: Dict, config_key: str) -> Dict[str, List]:
        """Group funds by which table they use for a given data type"""
        table_groups = {}
        for fund_name, fund in all_funds.items():
            table_name = fund.mapping_data.get(config_key)
            if table_name not in table_groups:
                table_groups[table_name] = []
            table_groups[table_name].append(fund)
        return table_groups

    def _store_fund_data(self, data_store: BulkDataStore, fund_name: str, data_type: str, data: pd.DataFrame):
        """Store data in the central data store"""
        self._ensure_fund_slot(data_store, fund_name)
        data_store.fund_data[fund_name][data_type] = data
        data_store.loaded_data_types.add(data_type)

    def _ensure_fund_slot(self, data_store: BulkDataStore, fund_name: str):
        if fund_name not in data_store.fund_data:
            data_store.fund_data[fund_name] = {}
        data_store.loaded_funds.add(fund_name)

    def _get_table_column(self, table, *column_names: str):
        for name in column_names:
            if hasattr(table, name):
                return getattr(table, name)
        return None

    def _filter_by_fund(self, df: pd.DataFrame, fund_name: str, mapping_data: Dict) -> pd.DataFrame:
        """Attempt to filter a dataset to the requested fund using best-effort heuristics."""
        if df.empty:
            return df

        lower_columns = {col.lower(): col for col in df.columns}
        for candidate in ('fund', 'fund_ticker', 'ticker', 'portfolio', 'account_name', 'account'):
            if candidate in lower_columns:
                column = lower_columns[candidate]
                value = fund_name
                # Allow overrides from mapping when available
                if candidate in ('portfolio', 'account', 'account_name'):
                    value = mapping_data.get(candidate, fund_name)
                filtered = df[df[column] == value].copy()
                if not filtered.empty:
                    return filtered

        # As a fallback, return the original dataset so downstream logic can decide
        return df.copy()