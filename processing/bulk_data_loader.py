# processing/bulk_data_loader.py
import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from sqlalchemy import func, literal, and_, or_

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
        analysis_type: Optional[str] = None,

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
            data_store,
            all_funds,
            target_date,
            previous_date,
            analysis_type=analysis_type,
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
        holdings_map = [
            ('custodian_equity_holdings', 'custodian_equity', 'custodian_equity_t1', 'equity'),
            ('custodian_option_holdings', 'custodian_option', 'custodian_option_t1', 'option'),
            (
                'custodian_treasury_holdings',
                'custodian_treasury',
                'custodian_treasury_t1',
                'treasury',
            ),
        ]

        for config_key, payload_key, payload_key_t1, holdings_kind in holdings_map:
            table_to_funds = self._group_funds_by_table(all_funds, config_key)

            for table_name, funds in table_to_funds.items():
                if not table_name or table_name == 'NULL':
                    continue

                try:
                    all_data = self._query_custodian_holdings_table(
                        table_name,
                        holdings_kind,
                        funds,
                        target_date,
                        previous_date,
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Failed to load %s (%s holdings): %s",
                        table_name,
                        holdings_kind,
                        exc,
                    )
                    continue

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
        *,
        analysis_type: Optional[str] = None,
    ) -> None:
        """Load ALL Vest holdings for ALL funds in one query."""
        holdings_map = [
            ('vest_equity_holdings', 'vest_equity', 'vest_equity_t1'),
            ('vest_options_holdings', 'vest_option', 'vest_option_t1'),
            ('vest_treasury_holdings', 'vest_treasury', 'vest_treasury_t1'),
        ]

        for config_key, payload_key, payload_key_t1 in holdings_map:
            table_to_funds = self._group_funds_by_table(all_funds, config_key)

            for table_name, funds in table_to_funds.items():
                if not table_name or table_name == 'NULL':
                    continue

                try:
                    table = getattr(self.base_cls.classes, table_name)
                except AttributeError:
                    self.logger.warning("Table %s not found for Vest holdings", table_name)
                    continue

                try:
                    if config_key == 'vest_equity_holdings':
                        all_data = self._query_vest_equity_holdings(
                            table,
                            funds,
                            target_date,
                            previous_date,
                            analysis_type,
                        )
                    elif config_key == 'vest_options_holdings':
                        all_data = self._query_vest_option_holdings(
                            table,
                            funds,
                            target_date,
                            previous_date,
                            analysis_type,
                        )
                    else:
                        date_column = self._get_table_column(
                            table,
                            'date',
                        )
                        query = self.session.query(table)
                        if date_column is not None:
                            query = self._apply_date_filter(
                                query, date_column, target_date, previous_date
                            )
                        all_data = pd.read_sql(query.statement, self.session.bind)

                        if config_key == 'vest_treasury_holdings':
                            all_data = self._augment_treasury_holdings(all_data)

                        if analysis_type:
                            analysis_column = next(
                                (
                                    column
                                    for column in all_data.columns
                                    if column.lower() == 'analysis_type'
                                ),
                                None,
                            )
                            if analysis_column:
                                mask = (
                                        all_data[analysis_column]
                                        .astype(str)
                                        .str.lower()
                                        == analysis_type.lower()
                                )
                                all_data = all_data.loc[mask].copy()

                    for fund in funds:
                        fund_name = fund.name
                        fund_data = self._filter_by_fund(
                            all_data, fund_name, fund.mapping_data
                        )
                        current, previous = self._split_current_previous(
                            fund_data,
                            self._find_date_column(
                                fund_data,
                                'date',
                            ),
                            target_date,
                            previous_date,
                        )
                        self._store_fund_data(
                            data_store, fund_name, payload_key, current
                        )
                        if previous_date is not None:
                            self._store_fund_data(
                                data_store, fund_name, payload_key_t1, previous
                            )

                except Exception as exc:
                    self.logger.warning(
                        "Failed to load %s for Vest holdings (%s): %s",
                        table_name,
                        payload_key,
                        exc,
                    )

    def _query_vest_equity_holdings(
            self,
            table,
            funds: List,
            target_date: date,
            previous_date: Optional[date],
            analysis_type: Optional[str],
    ) -> pd.DataFrame:
        bbg_equity_table = getattr(
            self.base_cls.classes, 'bbg_equity_flds_blotter', None
        )
        bbg_closes_table = getattr(
            self.base_cls.classes, 'bbg_feed_equity_closes', None
        )

        columns = [table]
        ticker_column = self._get_table_column(
            table,
            'ticker',
            'equity_ticker',
        )

        if bbg_equity_table is not None and ticker_column is not None:
            columns.extend(
                [
                    bbg_equity_table.GICS_SECTOR_NAME,
                    bbg_equity_table.GICS_INDUSTRY_NAME,
                    bbg_equity_table.GICS_INDUSTRY_GROUP_NAME,
                    bbg_equity_table.REGULATORY_STRUCTURE,
                    bbg_equity_table.SECURITY_TYP,
                ]
            )

        closes_subquery = None
        if (
                bbg_closes_table is not None
                and ticker_column is not None
                and hasattr(bbg_closes_table, 'BBG_SEC_ID')
        ):
            closes_subquery = (
                self.session.query(
                    bbg_closes_table.BBG_SEC_ID.label('BBG_SEC_ID'),
                    bbg_closes_table.EQY_SH_OUT.label('EQY_SH_OUT'),
                    func.row_number()
                    .over(
                        partition_by=bbg_closes_table.BBG_SEC_ID,
                        order_by=bbg_closes_table.Date.desc(),
                    )
                    .label('row_num'),
                )
                .filter(bbg_closes_table.PX_MID.is_(None))
                .subquery('bbg_closes_subquery')
            )
            columns.extend(
                [
                    closes_subquery.c.EQY_SH_OUT.label('EQY_SH_OUT'),
                    (closes_subquery.c.EQY_SH_OUT * 1_000_000).label(
                        'EQY_SH_OUT_million'
                    ),
                ]
            )

        query = self.session.query(*columns)

        if bbg_equity_table is not None and ticker_column is not None:
            query = query.outerjoin(
                bbg_equity_table, ticker_column == bbg_equity_table.TICKER
            )

        if closes_subquery is not None and ticker_column is not None:
            query = query.outerjoin(
                closes_subquery,
                and_(
                    func.concat(ticker_column, literal(' US Equity'))
                    == closes_subquery.c.BBG_SEC_ID,
                    closes_subquery.c.row_num == 1,
                ),
            )

        date_column = self._get_table_column(table, 'date')
        if date_column is not None:
            query = self._apply_date_filter(
                query, date_column, target_date, previous_date
            )

        fund_column = self._get_table_column(
            table,
            'fund',
        )
        fund_aliases = self._collect_fund_aliases(funds)
        if fund_column is not None and fund_aliases:
            query = query.filter(fund_column.in_(fund_aliases))

        if analysis_type:
            analysis_column = self._get_table_column(table, 'analysis_type')
            if analysis_column is not None:
                query = query.filter(
                    func.lower(func.trim(analysis_column))
                    == analysis_type.lower()
                )

        df = pd.read_sql(query.statement, self.session.bind)
        df = self._augment_equity_holdings(df)

        if analysis_type and 'analysis_type' in df.columns:
            mask = df['analysis_type'].astype(str).str.lower() == analysis_type.lower()
            df = df.loc[mask].copy()

        return df

    def _query_vest_option_holdings(
            self,
            table,
            funds: List,
            target_date: date,
            previous_date: Optional[date],
            analysis_type: Optional[str],
    ) -> pd.DataFrame:
        bbg_equity_table = getattr(
            self.base_cls.classes, 'bbg_equity_flds_blotter', None
        )

        columns = [table]
        underlying_column = self._get_table_column(
            table,
            'equity_ticker',
            'ticker',
        )

        if bbg_equity_table is not None and underlying_column is not None:
            columns.extend(
                [
                    bbg_equity_table.GICS_SECTOR_NAME,
                    bbg_equity_table.GICS_INDUSTRY_NAME,
                    bbg_equity_table.GICS_INDUSTRY_GROUP_NAME,
                    bbg_equity_table.REGULATORY_STRUCTURE,
                    bbg_equity_table.SECURITY_TYP,
                ]
            )

        query = self.session.query(*columns)

        if bbg_equity_table is not None and underlying_column is not None:
            query = query.outerjoin(
                bbg_equity_table, underlying_column == bbg_equity_table.TICKER
            )

        date_column = self._get_table_column(table, 'date')
        if date_column is not None:
            query = self._apply_date_filter(
                query, date_column, target_date, previous_date
            )

        fund_column = self._get_table_column(
            table,
            'fund',
        )
        fund_aliases = self._collect_fund_aliases(funds)
        if fund_column is not None and fund_aliases:
            query = query.filter(fund_column.in_(fund_aliases))

        if analysis_type:
            analysis_column = self._get_table_column(table, 'analysis_type')
            if analysis_column is not None:
                query = query.filter(
                    func.lower(func.trim(analysis_column))
                    == analysis_type.lower()
                )

        df = pd.read_sql(query.statement, self.session.bind)
        df = self._augment_option_holdings(df)

        if analysis_type and 'analysis_type' in df.columns:
            mask = df['analysis_type'].astype(str).str.lower() == analysis_type.lower()
            df = df.loc[mask].copy()

        return df

    def _augment_equity_holdings(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        result = df.copy()

        quantity, _ = self._get_numeric_series(
            result, 'quantity', 'nav_shares',
        )
        price, _ = self._get_numeric_series(
            result, 'price', 'equity_price'
        )

        result['equity_market_value'] = quantity * price

        result = result.drop(columns=['row_num', 'BBG_SEC_ID'], errors='ignore')

        return result

    def _augment_option_holdings(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        result = df.copy()

        quantity, _ = self._get_numeric_series(
            result, 'quantity', 'nav_shares',
        )
        price, _ = self._get_numeric_series(result, 'price')
        equity_price, _ = self._get_numeric_series(
            result, 'equity_underlying_price',
        )
        delta, _ = self._get_numeric_series(
            result, 'delta',
        )

        result['option_market_value'] = quantity * price * 100
        result['option_notional_value'] = quantity * equity_price * 100
        result['option_delta_adjusted_notional'] = quantity * equity_price * delta * 100
        result['option_delta_adjusted_market_value'] = quantity * price * delta * 100

        return result

    def _get_numeric_series(
        self, df: pd.DataFrame, *candidates: str
    ) -> Tuple[pd.Series, bool]:
        column = self._find_dataframe_column(df, *candidates)
        if column is None:
            return pd.Series(0.0, index=df.index, dtype=float), False
        series = pd.to_numeric(df[column], errors='coerce')
        return series, True

    def _augment_treasury_holdings(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        result = df.copy()

        quantity, _ = self._get_numeric_series(result, 'nav_shares', 'quantity')
        price, _ = self._get_numeric_series(result, 'price')

        result['treasury_market_value'] = quantity * price / 1000

        return result

    @staticmethod
    def _find_dataframe_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
        lower_map = {col.lower(): col for col in df.columns}
        for candidate in candidates:
            column = lower_map.get(candidate.lower())
            if column:
                return column
        return None

    @staticmethod
    def _find_dataframe_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
        lower_map = {col.lower(): col for col in df.columns}
        for candidate in candidates:
            column = lower_map.get(candidate.lower())
            if column:
                return column
        return None

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
                if table_name == 'bny_us_nav_v2':
                    self._load_bny_us_nav(
                        data_store, funds, target_date, previous_date
                    )
                    continue

                if table_name == 'bny_vit_nav':
                    self._load_bny_vit_nav(
                        data_store, funds, target_date, previous_date
                    )
                    continue

                table = getattr(self.base_cls.classes, table_name)

                if table_name == 'umb_cef_nav':
                    # UMB has fund column - query for all relevant funds at once
                    date_column = self._get_table_column(table, 'date')
                    fund_column = self._get_table_column(table, 'fund')
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
                            ),
                            target_date,
                            previous_date,
                        )
                        self._store_fund_data(data_store, fund_name, 'nav', current)
                        if previous_date is not None:
                            self._store_fund_data(data_store, fund_name, 'nav_t1', previous)

                else:
                    # Default handling for NAV tables with standard fund identifiers
                    date_column = self._get_table_column(
                        table, 'date',
                    )
                    query = self.session.query(table)
                    if date_column is not None:
                        query = self._apply_date_filter(
                            query, date_column, target_date, previous_date
                        )
                    all_nav_data = pd.read_sql(query.statement, self.session.bind)

                    for fund in funds:
                        fund_name = fund.name
                        fund_nav = self._filter_by_fund(
                            all_nav_data, fund_name, fund.mapping_data
                        )
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
                if table_name == 'bny_us_cash':
                    self._load_bny_us_cash(data_store, funds, target_date, previous_date)
                    continue
                if table_name == 'bny_vit_cash':
                    self._load_bny_vit_cash(data_store, funds, target_date, previous_date)
                    continue

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

    def _load_bny_us_nav(
        self,
        data_store: BulkDataStore,
        funds: List,
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        for fund in funds:
            account_number = self._get_mapping_value(
                getattr(fund, 'mapping_data', {}) or {},
                'account_number_custodian',
                'account_number',
            )
            if not account_number:
                self.logger.warning(
                    "Skipping BNY US NAV for %s due to missing account number",
                    fund.name,
                )
                self._store_fund_data(data_store, fund.name, 'nav', pd.DataFrame())
                if previous_date is not None:
                    self._store_fund_data(data_store, fund.name, 'nav_t1', pd.DataFrame())
                continue

            frames = []
            current_df = self._get_bny_us_nav(target_date, account_number)
            if not current_df.empty:
                current_df = current_df.assign(fund=fund.name)
                frames.append(current_df)

            if previous_date is not None:
                previous_df = self._get_bny_us_nav(previous_date, account_number)
                if not previous_df.empty:
                    previous_df = previous_df.assign(fund=fund.name)
                    frames.append(previous_df)

            combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            date_column = self._find_date_column(combined, 'date')
            current, previous = self._split_current_previous(
                combined,
                date_column,
                target_date,
                previous_date,
            )
            self._store_fund_data(data_store, fund.name, 'nav', current)
            if previous_date is not None:
                self._store_fund_data(data_store, fund.name, 'nav_t1', previous)

    def _load_bny_vit_nav(
        self,
        data_store: BulkDataStore,
        funds: List,
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        for fund in funds:
            frames = []
            fund_identifier = self._get_mapping_value(
                getattr(fund, 'mapping_data', {}) or {},
                'fund',
                'fund_name',
                'fund_ticker',
                'account_number_custodian',
            )
            current_df = self._get_bny_vit_nav(
                target_date,
                fund_identifier,
            )
            if not current_df.empty:
                current_df = current_df.assign(fund=fund.name)
                frames.append(current_df)

            if previous_date is not None:
                previous_df = self._get_bny_vit_nav(
                    previous_date,
                    fund_identifier,
                )
                if not previous_df.empty:
                    previous_df = previous_df.assign(fund=fund.name)
                    frames.append(previous_df)

            combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            date_column = self._find_date_column(combined, 'date', 'valuation_date')
            current, previous = self._split_current_previous(
                combined,
                date_column,
                target_date,
                previous_date,
            )
            self._store_fund_data(data_store, fund.name, 'nav', current)
            if previous_date is not None:
                self._store_fund_data(data_store, fund.name, 'nav_t1', previous)

    def _load_bny_us_cash(
        self,
        data_store: BulkDataStore,
        funds: List,
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        for fund in funds:
            account_number = self._get_mapping_value(
                getattr(fund, 'mapping_data', {}) or {},
                'account_number_custodian',
                'account_number',
            )
            if not account_number:
                self.logger.warning(
                    "Skipping BNY US cash for %s due to missing account number",
                    fund.name,
                )
                self._store_fund_data(data_store, fund.name, 'cash', pd.DataFrame())
                if previous_date is not None:
                    self._store_fund_data(data_store, fund.name, 'cash_t1', pd.DataFrame())
                continue

            frames = []
            current_df = self._get_bny_us_cash(target_date, account_number)
            if not current_df.empty:
                current_df = current_df.assign(fund=fund.name)
                frames.append(current_df)

            if previous_date is not None:
                previous_df = self._get_bny_us_cash(previous_date, account_number)
                if not previous_df.empty:
                    previous_df = previous_df.assign(fund=fund.name)
                    frames.append(previous_df)

            combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            date_column = self._find_date_column(combined, 'date')
            current, previous = self._split_current_previous(
                combined,
                date_column,
                target_date,
                previous_date,
            )
            self._store_fund_data(data_store, fund.name, 'cash', current)
            if previous_date is not None:
                self._store_fund_data(data_store, fund.name, 'cash_t1', previous)

    def _load_bny_vit_cash(
        self,
        data_store: BulkDataStore,
        funds: List,
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        for fund in funds:
            frames = []
            fund_identifier = self._get_mapping_value(
                getattr(fund, 'mapping_data', {}) or {},
                'fund',
                'fund_name',
                'fund_ticker',
            )
            if not fund_identifier:
                fund_identifier = fund.name
            current_df = self._get_bny_vit_cash(
                target_date,
                fund_identifier,
            )
            if not current_df.empty:
                current_df = current_df.assign(fund=fund.name)
                frames.append(current_df)

            if previous_date is not None:
                previous_df = self._get_bny_vit_cash(
                    previous_date,
                    fund_identifier,
                )
                if not previous_df.empty:
                    previous_df = previous_df.assign(fund=fund.name)
                    frames.append(previous_df)

            combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            date_column = self._find_date_column(combined, 'date')
            current, previous = self._split_current_previous(
                combined,
                date_column,
                target_date,
                previous_date,
            )
            self._store_fund_data(data_store, fund.name, 'cash', current)
            if previous_date is not None:
                self._store_fund_data(data_store, fund.name, 'cash_t1', previous)

    def _get_bny_us_nav(self, nav_date: Optional[date], account_number: Optional[str]) -> pd.DataFrame:
        if nav_date is None or not account_number:
            return pd.DataFrame()

        table = getattr(self.base_cls.classes, 'bny_us_nav_v2', None)
        if table is None:
            raise AttributeError('bny_us_nav_v2 table is not reflected')

        query = (
            self.session.query(
                table.date.label('date'),
                table.account_number.label('account_number'),
                table.nav.label('nav'),
                table.total_net_assets.label('total_net_assets'),
                table.total_assets.label('total_assets'),
                table.income_earned.label('income_earned'),
                table.expenses.label('expenses'),
                table.net_invest_income.label('net_invest_income'),
                table.securities_at_value.label('securities_at_value'),
                table.cash.label('cash'),
                table.accrued_income.label('accrued_income'),
                table.accrued_expenses.label('accrued_expenses'),
                table.shares_outstanding.label('shares_outstanding'),
            )
            .filter(table.account_number == account_number)
            .filter(table.date == nav_date)
        )

        return pd.read_sql(query.statement, self.session.bind)

    def _get_bny_vit_nav(
        self,
        nav_date: Optional[date],
        fund_identifier: Optional[str],
    ) -> pd.DataFrame:
        if nav_date is None:
            return pd.DataFrame()

        table = getattr(self.base_cls.classes, 'bny_vit_nav', None)
        if table is None:
            raise AttributeError('bny_vit_nav table is not reflected')

        query = (
            self.session.query(
                table.valuation_date.label('date'),
                table.fund.label('fund'),
                table.nav_per_share_unrounded.label('nav'),
                table.total_net_assets.label('total_net_assets'),
                table.total_net_assets.label('total_assets'),
                table.market_value_base_cash.label('cash'),
                table.shares_outstanding.label('shares_outstanding'),
                literal(471.29).label('expenses'),
            )
            .filter(table.valuation_date == nav_date)
        )
        if fund_identifier and hasattr(table, 'fund'):
            query = query.filter(table.fund == fund_identifier)
        return pd.read_sql(query.statement, self.session.bind)

    def _get_bny_us_cash(
        self,
        cash_date: Optional[date],
        account_number: Optional[str],
    ) -> pd.DataFrame:
        if cash_date is None or not account_number:
            return pd.DataFrame()

        table = getattr(self.base_cls.classes, 'bny_us_cash', None)
        if table is None:
            raise AttributeError('bny_us_cash table is not reflected')

        latest_update = (
            self.session.query(
                table.account_number.label('account_number'),
                table.date.label('date'),
                func.max(table.update_time).label('max_update_time'),
            )
            .filter(table.account_number == account_number)
            .filter(table.date == cash_date)
            .group_by(table.account_number, table.date)
            .subquery()
        )

        query = (
            self.session.query(
                table.account_number.label('account_number'),
                table.date.label('date'),
                func.sum(table.end_balance).label('cash_value'),
            )
            .join(
                latest_update,
                and_(
                    table.account_number == latest_update.c.account_number,
                    table.date == latest_update.c.date,
                    table.update_time == latest_update.c.max_update_time,
                ),
            )
            .group_by(table.account_number, table.date)
        )
        return pd.read_sql(query.statement, self.session.bind)

    def _get_bny_vit_cash(
        self,
        cash_date: Optional[date],
        fund_identifier: Optional[str],
    ) -> pd.DataFrame:
        if cash_date is None:
            return pd.DataFrame()

        table = getattr(self.base_cls.classes, 'bny_vit_cash', None)
        if table is None:
            raise AttributeError('bny_vit_cash table is not reflected')

        query = (
            self.session.query(
                table.fund.label('fund'),
                table.date.label('date'),
                func.sum(table.cash_balance).label('cash_value'),
            )
            .filter(table.date == cash_date)
        )
        if fund_identifier and hasattr(table, 'fund'):
            query = query.filter(table.fund == fund_identifier)
        query = query.group_by(table.fund, table.date)
        return pd.read_sql(query.statement, self.session.bind)

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

    def _query_custodian_holdings_table(
        self,
        table_name: str,
        holdings_kind: str,
        funds: List,
        target_date: date,
        previous_date: Optional[date],
    ) -> pd.DataFrame:
        """Build a custodian-aware query that labels shared columns consistently."""

        table = getattr(self.base_cls.classes, table_name)
        lower_name = table_name.lower()

        if 'bny' in lower_name:
            query = self._build_bny_holdings_query(
                table, holdings_kind, target_date, previous_date
            )
        elif 'umb' in lower_name:
            query = self._build_umb_holdings_query(
                table, holdings_kind, target_date, previous_date
            )
        else:
            query = self.session.query(table)

        date_column = self._get_table_column(
            table,
            'date',
            'trade_date',
            'process_date',
            'effective_date',
        )
        if date_column is not None:
            query = self._apply_date_filter(query, date_column, target_date, previous_date)

        fund_column = self._get_table_column(
            table,
            'fund',
            'fund_ticker',
            'fund_name',
            'account_number',
        )
        fund_values = self._collect_fund_aliases(funds)
        if fund_column is not None and fund_values:
            query = query.filter(fund_column.in_(fund_values))

        return pd.read_sql(query.statement, self.session.bind)

    def _build_bny_holdings_query(
        self,
        table,
        holdings_kind: str,
        target_date: date,
        previous_date: Optional[date],
    ):
        """Select the canonical columns for BNY custodian datasets."""

        min_date = target_date if previous_date is None else min(target_date, previous_date)

        if holdings_kind == 'equity':
            query = self.session.query(
                table.date.label('date'),
                table.fund.label('fund'),
                table.security_sedol.label('sedol'),
                table.sharespar.label('shares_cust'),
                table.sharespar.label('quantity'),
                table.category_description.label('category_description'),
                table.asset_group.label('asset_group'),
                table.price_base.label('price'),
                table.traded_market_value_base.label('market_value'),
            ).filter(
                    table.asset_group.notin_(['O', 'UN', 'ME', 'MM', 'CA', 'TI', 'B'])
            )
            return query

        if holdings_kind == 'option':
            maturity_column = getattr(table, 'maturity_date', None)
            query = self.session.query(
                table.date.label('date'),
                table.fund.label('fund'),
                table.security_description_long_1.label('optticker'),
                table.sharespar.label('shares_cust'),
                table.sharespar.label('quantity'),
                table.category_description.label('category_description'),
                table.asset_group.label('asset_group'),
                table.price_base.label('price'),
                table.traded_market_value_base.label('market_value'),
                table.security_cins.label('cusip'),
                table.maturity_date.label('maturity_date'),
            ).filter(
                    table.asset_group.notin_(['S', 'FS', 'UN', 'ME', 'MM', 'CA', 'B', 'TI'])
                ).filter(maturity_column >= min_date)

            return query

        if holdings_kind == 'treasury':
            query = self.session.query(
                table.date.label('date'),
                table.fund.label('fund'),
                table.security_cins.label('cusip'),
                table.security_sedol.label('sedol'),
                table.security_description_long_1.label('ticker'),
                table.maturity_date.label('maturity'),
                table.sharespar.label('shares_cust'),
                table.sharespar.label('quantity'),
                table.price_base.label('price'),
                table.traded_market_value_base.label('market_value'),
                table.asset_group.label('asset_group'),
            ).filter(table.asset_group == "TI")
            return query

    def _build_umb_holdings_query(
        self,
        table,
        holdings_kind: str,
        target_date: date,
        previous_date: Optional[date],
    ):
        """Select the canonical columns for UMB custodian datasets."""

        base_columns = [
            self._column_or_literal(
                table, 'date', 'process_date', 'date', 'business_date'
            ),
            self._column_or_literal(table, 'fund', 'fund'),
        ]

        category_column = getattr(table, 'security_catgry', None)

        if holdings_kind == 'equity':
            columns = base_columns + [
                self._label_upper(table, ('security_tkr', 'ticker'), 'equity_ticker'),
                self._column_or_literal(table, 'shares_cust', 'mkt_qty', 'quantity'),
                self._column_or_literal(table, 'quantity', 'mkt_qty', 'quantity'),
                self._column_or_literal(table, 'price', 'eod_close', 'price'),
                self._column_or_literal(table, 'market_value', 'mkt_mktval', 'market_value'),
                self._column_or_literal(
                    table,
                    'category_description',
                    'security_catgry',
                    'category_description',
                ),
            ]
            query = self.session.query(*columns)
            if category_column is not None:
                query = query.filter(category_column.in_(['COMMON', 'REIT']))
            return query

        if holdings_kind == 'option':
            desc_column = getattr(table, 'security_desc', None)
            if desc_column is not None:
                first_space = func.instr(desc_column, ' ')
                optticker_expr = func.concat(
                    func.substr(desc_column, 1, first_space),
                    literal('US'),
                    func.substr(desc_column, first_space),
                )
                optticker_expr = func.upper(func.trim(optticker_expr)).label('optticker')
            else:
                optticker_expr = literal(None).label('optticker')

            columns = base_columns + [
                optticker_expr,
                self._label_upper(table, ('security_tkr', 'ticker'), 'equity_ticker'),
                self._column_or_literal(table, 'shares_cust', 'mkt_qty', 'quantity'),
                self._column_or_literal(table, 'quantity', 'mkt_qty', 'quantity'),
                self._column_or_literal(table, 'price', 'eod_close', 'price'),
                self._column_or_literal(table, 'market_value', 'mkt_mktval', 'market_value'),
                self._column_or_literal(
                    table,
                    'category_description',
                    'security_catgry',
                    'category_description',
                ),
                self._column_or_literal(table, 'occ_symbol', 'occ_id', 'occ_symbol'),
            ]
            query = self.session.query(*columns)
            if category_column is not None:
                query = query.filter(category_column.like('OPT%'))
            return query

        if holdings_kind == 'treasury':
            columns = base_columns + [
                self._label_upper(
                    table,
                    (
                        'security_tkr',
                        'ticker',
                        'legal1',
                    ),
                    'ticker',
                ),
                self._column_or_literal(table, 'shares_cust', 'qty', 'quantity'),
                self._column_or_literal(table, 'quantity', 'qty', 'quantity'),
                self._column_or_literal(table, 'price', 'price', default=None),
                self._column_or_literal(
                    table,
                    'market_value',
                    'mkt_mktval',
                    'market_value',
                    default=None,
                ),
                self._column_or_literal(
                    table,
                    'category_description',
                    'security_catgry',
                    'category_description',
                    default=None,
                ),
                self._column_or_literal(table, 'occ_symbol', 'occ_id', 'occ_symbol', default=None),
            ]
            return self.session.query(*columns)

        return self.session.query(table)

    def _column_or_literal(
        self,
        table,
        alias: str,
        *candidates: str,
        default=None,
    ):
        for name in candidates:
            column = getattr(table, name, None)
            if column is not None:
                return column.label(alias)
        return literal(default).label(alias)

    def _label_upper(
        self,
        table,
        candidates: Tuple[str, ...],
        alias: str,
    ):
        for name in candidates:
            column = getattr(table, name, None)
            if column is not None:
                return func.upper(func.trim(column)).label(alias)
        return literal(None).label(alias)

    def _get_mapping_value(self, mapping: Dict, *keys: str) -> Optional[str]:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed and trimmed.upper() != 'NULL':
                    return trimmed
            elif value:
                return value

        numbers = mapping.get('account_numbers')
        if isinstance(numbers, dict):
            for key in keys:
                candidate = numbers.get(key)
                normalised = self._normalise_mapping_value(candidate)
                if normalised:
                    return normalised
            for candidate in numbers.values():
                normalised = self._normalise_mapping_value(candidate)
                if normalised:
                    return normalised

        return None

    def _collect_fund_aliases(self, funds: List) -> List[str]:
        aliases: Set[str] = set()
        for fund in funds:
            aliases.add(fund.name)
            mapping = getattr(fund, 'mapping_data', {}) or {}
            for key in (
                'fund',
                'fund_name',
                'fund_ticker',
                'portfolio',
                'account',
                'account_number',
                'account_number_custodian',
            ):
                value = mapping.get(key)
                if isinstance(value, str) and value and value != 'NULL':
                    aliases.add(value)
        return sorted(aliases)

    def _apply_date_filter(self, query, column, target_date: date, previous_date: Optional[date]):
        if previous_date is not None:
            return query.filter(column.in_([target_date, previous_date]))
        return query.filter(column == target_date)

    @staticmethod
    def _normalise_mapping_value(value) -> Optional[str]:
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed and trimmed.upper() != 'NULL':
                return trimmed
            return None
        if isinstance(value, (list, tuple, set)):
            for item in value:
                normalised = BulkDataLoader._normalise_mapping_value(item)
                if normalised:
                    return normalised
            return None
        if pd.isna(value):
            return None
        if value is not None:
            trimmed = str(value).strip()
            return trimmed or None
        return None

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