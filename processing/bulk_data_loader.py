# processing/bulk_data_loader.py
import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from sqlalchemy import literal

from utilities.ticker_utils import normalize_all_holdings



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

    def load_all_data_for_date(
        self,
        target_date: date,
        *,
        previous_date: Optional[date] = None,
    ) -> BulkDataStore:
        """Bulk load data for ``target_date`` (and optionally ``previous_date``)."""

        data_store = BulkDataStore(date=str(target_date))

        # Get all funds from registry
        all_funds = self.fund_registry.funds

        # Bulk load by table type (not by fund)
        self._bulk_load_custodian_holdings(
            data_store, all_funds, target_date, previous_date
        )
        self._bulk_load_vest_holdings(
            data_store, all_funds, target_date, previous_date
        )
        self._bulk_load_nav_data(
            data_store, all_funds, target_date, previous_date
        )
        self._bulk_load_cash_data(
            data_store, all_funds, target_date, previous_date
        )
        self._bulk_load_index_data(
            data_store, all_funds, target_date, previous_date
        )
        self._normalise_loaded_holdings(data_store)


        # Ensure every fund has an entry even if a data set was empty
        for fund_name in all_funds.keys():
            self._ensure_fund_slot(data_store, fund_name)

        self.logger.info(f"Bulk loaded data for {len(data_store.loaded_funds)} funds")
        return data_store

    def _bulk_load_custodian_holdings(
        self,
        data_store: BulkDataStore,
        all_funds: Dict,
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        """Load ALL custodian holdings for ALL funds in one query per table."""
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
                    query = self._apply_date_filter(
                        query, date_column, target_date, previous_date
                    )
                all_data = pd.read_sql(query.statement, self.session.bind)

                # Distribute data to appropriate funds
                for fund in funds:
                    fund_name = fund.name
                    fund_data = self._filter_by_fund(all_data, fund_name, fund.mapping_data)
                    current, previous = self._split_current_previous(
                        fund_data,
                        self._find_date_column(
                            fund_data,
                            'date',
                            'trade_date',
                            'business_date',
                        ),
                        target_date,
                        previous_date,
                    )
                    self._store_fund_data(data_store, fund_name, 'custodian_equity', current)
                    if previous_date is not None:
                        self._store_fund_data(
                            data_store, fund_name, 'custodian_equity_t1', previous
                        )

            except Exception as e:
                self.logger.warning(f"Failed to load {table_name}: {e}")

    def _normalise_loaded_holdings(self, data_store: BulkDataStore) -> None:
        """
        For each fund in data_store, normalise OMS and custodian holdings for T and T-1,
        and write the normalised DataFrames back into fund_payload using your existing
        key conventions (vest_*, custodian_*, *_t1).
        """

        # Map your fund_payload keys -> normalize_all_holdings expected input keys
        payload_to_input = {
            # T (current)
            "vest_equity": "equity_holdings",
            "vest_option": "options_holdings",
            "vest_treasury": "treasury_holdings",
            "custodian_equity": "custodian_equity_holdings",
            "custodian_option": "custodian_option_holdings",
            "custodian_treasury": "custodian_treasury_holdings",

            # T-1
            "vest_equity_t1": "t1_equity_holdings",
            "vest_option_t1": "t1_options_holdings",
            "vest_treasury_t1": "t1_treasury_holdings",
            "custodian_equity_t1": "t1_custodian_equity_holdings",
            "custodian_option_t1": "t1_custodian_option_holdings",
            "custodian_treasury_t1": "t1_custodian_treasury_holdings",
        }

        # Inverse map to push normalised outputs back to your payload keys
        input_to_payload = {v: k for k, v in payload_to_input.items()}

        for fund_name, fund_payload in data_store.fund_data.items():
            if not fund_payload:
                continue

            # Build input dict for the normalisation function with safe defaults
            to_normalize = {}
            for payload_key, input_key in payload_to_input.items():
                to_normalize[input_key] = fund_payload.get(payload_key, pd.DataFrame())

            # Normalise
            normalized = normalize_all_holdings(to_normalize)

            # Write normalised outputs back to the payload using your key names
            for input_key, payload_key in input_to_payload.items():
                fund_payload[payload_key] = normalized.get(input_key, pd.DataFrame())

    def _bulk_load_vest_holdings(
        self,
        data_store: BulkDataStore,
        all_funds: Dict,
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        """Load ALL Vest holdings for ALL funds in one query."""
        table_to_funds = self._group_funds_by_table(all_funds, 'vest_equity_holdings')

        for table_name, funds in table_to_funds.items():
            if not table_name or table_name == 'NULL':
                continue

            try:
                table = getattr(self.base_cls.classes, table_name)
                date_column = self._get_table_column(table, 'date', 'trade_date', 'business_date')
                query = self.session.query(table)
                if date_column is not None:
                    query = self._apply_date_filter(
                        query, date_column, target_date, previous_date
                    )
                all_data = pd.read_sql(query.statement, self.session.bind)

                for fund in funds:
                    fund_name = fund.name
                    fund_data = self._filter_by_fund(all_data, fund_name, fund.mapping_data)
                    current, previous = self._split_current_previous(
                        fund_data,
                        self._find_date_column(
                            fund_data,
                            'date',
                            'trade_date',
                            'business_date',
                        ),
                        target_date,
                        previous_date,
                    )
                    self._store_fund_data(data_store, fund_name, 'vest_equity', current)
                    if previous_date is not None:
                        self._store_fund_data(
                            data_store, fund_name, 'vest_equity_t1', previous
                        )

            except Exception as e:
                self.logger.warning(f"Failed to load {table_name}: {e}")

    def _bulk_load_nav_data(
        self,
        data_store: BulkDataStore,
        all_funds: Dict,
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        """Load ALL NAV data for ALL funds in minimal queries."""
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
                        query = self._apply_date_filter(
                            query, date_column, target_date, previous_date
                        )
                    all_nav_data = pd.read_sql(query.statement, self.session.bind)
                    for fund in funds:
                        fund_name = fund.name
                        fund_nav = self._filter_by_fund(all_nav_data, fund_name, fund.mapping_data)
                        current, previous = self._split_current_previous(
                            fund_nav,
                            self._find_date_column(
                                fund_nav,
                                'date',
                                'nav_date',
                                'business_date',
                                'effective_date',
                            ),
                            target_date,
                            previous_date,
                        )
                        self._store_fund_data(data_store, fund_name, 'nav', current)
                        if previous_date is not None:
                            self._store_fund_data(data_store, fund_name, 'nav_t1', previous)

                elif table_name == 'umb_cef_nav':
                    # UMB has fund column - query for all relevant funds at once
                    date_column = self._get_table_column(table, 'date', 'nav_date', 'business_date')
                    fund_column = self._get_table_column(table, 'fund', 'fund_ticker')
                    query = self.session.query(table)
                    if date_column is not None:
                        query = self._apply_date_filter(
                            query, date_column, target_date, previous_date
                        )
                    if fund_column is not None:
                        query = query.filter(fund_column.in_([fund.name for fund in funds]))
                    all_nav_data = pd.read_sql(query.statement, self.session.bind)

                    for fund in funds:
                        fund_name = fund.name
                        fund_nav = self._filter_by_fund(all_nav_data, fund_name, fund.mapping_data)
                        current, previous = self._split_current_previous(
                            fund_nav,
                            self._find_date_column(
                                fund_nav,
                                'date',
                                'nav_date',
                                'business_date',
                                'effective_date',
                            ),
                            target_date,
                            previous_date,
                        )
                        self._store_fund_data(data_store, fund_name, 'nav', current)
                        if previous_date is not None:
                            self._store_fund_data(data_store, fund_name, 'nav_t1', previous)

            except Exception as e:
                self.logger.warning(f"Failed to load NAV {table_name}: {e}")

    def _bulk_load_cash_data(
        self,
        data_store: BulkDataStore,
        all_funds: Dict,
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        """Load ALL cash data - similar to NAV loading."""
        table_to_funds = self._group_funds_by_table(all_funds, 'cash_table')

        for table_name, funds in table_to_funds.items():
            if not table_name or table_name == 'NULL':
                continue

            try:
                table = getattr(self.base_cls.classes, table_name)
                date_column = self._get_table_column(table, 'date', 'business_date', 'nav_date')
                query = self.session.query(table)
                if date_column is not None:
                    query = self._apply_date_filter(
                        query, date_column, target_date, previous_date
                    )
                all_cash_data = pd.read_sql(query.statement, self.session.bind)

                for fund in funds:
                    fund_name = fund.name
                    fund_cash = self._filter_by_fund(all_cash_data, fund_name, fund.mapping_data)
                    current, previous = self._split_current_previous(
                        fund_cash,
                        self._find_date_column(
                            fund_cash,
                            'date',
                            'business_date',
                            'nav_date',
                        ),
                        target_date,
                        previous_date,
                    )
                    self._store_fund_data(data_store, fund_name, 'cash', current)
                    if previous_date is not None:
                        self._store_fund_data(data_store, fund_name, 'cash_t1', previous)

            except Exception as e:
                self.logger.warning(f"Failed to load cash {table_name}: {e}")

    def _bulk_load_index_data(
        self,
        data_store: BulkDataStore,
        all_funds: Dict,
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        """Load ALL index data with provider-specific logic."""
        table_to_funds = self._group_funds_by_table(all_funds, 'index_holdings')

        for table_name, funds in table_to_funds.items():
            if not table_name or table_name == 'NULL':
                continue

            for fund in funds:
                try:
                    fund_index = self._get_index_data(
                        table_name, target_date, fund, previous_date
                    )
                    if isinstance(fund_index, tuple):
                        current, previous = fund_index
                    else:
                        current, previous = fund_index, pd.DataFrame()

                    self._store_fund_data(data_store, fund.name, 'index', current)
                    if previous_date is not None:
                        self._store_fund_data(data_store, fund.name, 'index_t1', previous)
                except Exception as exc:
                    self.logger.warning(
                        "Failed to load index %s for %s: %s", table_name, fund.name, exc
                    )
                    self._store_fund_data(data_store, fund.name, 'index', pd.DataFrame())

    def _get_index_data(
        self,
        table_name: str,
        target_date: date,
        fund,
        previous_date: Optional[date],
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
        """Dispatch to the appropriate index provider query."""
        provider_map = {
            'nasdaq_holdings': self._get_nasdaq_holdings,
            'sp_holdings': self._get_sp_holdings,
            'cboe_holdings': self._get_cboe_holdings,
            'dogg_index': self._get_dogg_index,
        }

        handler = provider_map.get(table_name)
        if handler:
            return handler(target_date, fund, previous_date)

        # Generic fallback: query table and filter by fund when possible
        table = getattr(self.base_cls.classes, table_name)
        date_column = self._get_table_column(
            table, 'date', 'effective_date', 'business_date', 'trade_date'
        )
        query = self.session.query(table)
        if date_column is not None:
            query = self._apply_date_filter(query, date_column, target_date, previous_date)
        df = pd.read_sql(query.statement, self.session.bind)
        filtered = self._filter_by_fund(df, fund.name, fund.mapping_data)
        current, previous = self._split_current_previous(
            filtered,
            self._find_date_column(
                filtered,
                'date',
                'effective_date',
                'business_date',
                'trade_date',
            ),
            target_date,
            previous_date,
        )
        if previous_date is None:
            return current
        return current, previous

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

    def _get_nasdaq_holdings(
        self,
        target_date: date,
        fund,
        previous_date: Optional[date],
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
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
        filters = [nasdaq.time_of_day == 'EOD', nasdaq.fund == self._resolve_index_identifier(fund)]
        if previous_date is not None:
            query = query.filter(nasdaq.date.in_([target_date, previous_date]), *filters)
        else:
            query = query.filter(nasdaq.date == target_date, *filters)
        query = query.group_by(nasdaq.ticker)

        df = pd.read_sql(query.statement, self.session.bind)
        current, previous = self._split_current_previous(
            df,
            'date',
            target_date,
            previous_date,
        )
        if previous_date is None:
            return current
        return current, previous

    def _get_sp_holdings(
        self,
        target_date: date,
        fund,
        previous_date: Optional[date],
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
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
        query = query.filter(sp.INDEX_CODE == self._resolve_index_identifier(fund))
        if previous_date is not None:
            query = query.filter(sp.EFFECTIVE_DATE.in_([target_date, previous_date]))
        else:
            query = query.filter(sp.EFFECTIVE_DATE == target_date)

        df = pd.read_sql(query.statement, self.session.bind)
        current, previous = self._split_current_previous(
            df,
            'date',
            target_date,
            previous_date,
        )
        if previous_date is None:
            return current
        return current, previous

    def _get_cboe_holdings(
        self,
        target_date: date,
        fund,
        previous_date: Optional[date],
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
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
        query = query.filter(cboe.index_name == identifier)
        if previous_date is not None:
            query = query.filter(cboe.date.in_([target_date, previous_date]))
        else:
            query = query.filter(cboe.date == target_date)

        df = pd.read_sql(query.statement, self.session.bind)
        current, previous = self._split_current_previous(
            df,
            'date',
            target_date,
            previous_date,
        )
        if previous_date is None:
            return current
        return current, previous

    def _get_dogg_index(
        self,
        target_date: date,
        fund,
        previous_date: Optional[date],
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
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
        if previous_date is not None:
            query = query.filter(dogg.DATE.in_([target_date, previous_date]))
        else:
            query = query.filter(dogg.DATE == target_date)

        df = pd.read_sql(query.statement, self.session.bind)
        current, previous = self._split_current_previous(
            df,
            'date',
            target_date,
            previous_date,
        )
        if previous_date is None:
            return current
        return current, previous

    def _apply_date_filter(self, query, column, target_date: date, previous_date: Optional[date]):
        if previous_date is not None:
            return query.filter(column.in_([target_date, previous_date]))
        return query.filter(column == target_date)

    def _find_date_column(self, df: pd.DataFrame, *candidates: str) -> Optional[str]:
        if df.empty:
            return None
        lower_columns = {col.lower(): col for col in df.columns}
        for candidate in candidates:
            column = lower_columns.get(candidate.lower())
            if column:
                return column
        return None

    def _split_current_previous(
        self,
        df: pd.DataFrame,
        date_column: Optional[str],
        target_date: date,
        previous_date: Optional[date],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty or not date_column or date_column not in df.columns:
            return df.copy(), pd.DataFrame()

        normalized = df.copy()
        normalized[date_column] = pd.to_datetime(
            normalized[date_column], errors='coerce'
        ).dt.date

        current = normalized[normalized[date_column] == target_date].copy()
        if previous_date is None:
            return current, pd.DataFrame()

        previous = normalized[normalized[date_column] == previous_date].copy()
        return current, previous

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