"""Single-pass bulk data loader for all funds in a registry.

`load_all_data_for_date` is the entry point. It walks every table once, fanning
results back out to each fund, and stores the result in a `BulkDataStore`. All
downstream services consume only the data store, not the database.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from pandas.tseries.offsets import BDay
from sqlalchemy import and_, func, literal, text

from config.fund_definitions import CLOSED_END_FUNDS, PRIVATE_FUNDS
from processing.fund import Fund
from utilities.ticker_utils import normalize_all_holdings


@dataclass
class BulkDataStore:
    """Central storage for ALL fund data — no DB calls after load completes."""

    date: str
    session: Optional[Any] = None
    fund_data: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)
    loaded_funds: Set[str] = field(default_factory=set)
    loaded_data_types: Set[str] = field(default_factory=set)
    gics_mapping: pd.DataFrame = field(default_factory=pd.DataFrame)


class BulkDataLoader:
    """Loads all data for all funds with one query per table."""

    def __init__(self, session, base_cls, fund_registry: Dict[str, Fund]):
        self.session = session
        self.base_cls = base_cls
        self.fund_registry = fund_registry
        self.logger = logging.getLogger(__name__)
        # Per-run cache so equity/option/treasury trade queries share one fetch.
        self._trades_cache: Dict[Tuple[str, Optional[date]], pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def load_all_data_for_date(
        self,
        target_date: date,
        *,
        previous_date: Optional[date] = None,
        analysis_type: Optional[str] = None,
    ) -> BulkDataStore:
        self._trades_cache.clear()
        data_store = BulkDataStore(date=str(target_date), session=self.session)
        all_funds = self.fund_registry
        tplus_one = self._business_day_shift(target_date, 1)
        previous_tplus_one = self._business_day_shift(previous_date, 1)

        self._bulk_load_custodian_holdings(data_store, all_funds, target_date, previous_date)
        self._bulk_load_vest_holdings(data_store, all_funds, target_date, analysis_type, previous_date)
        self._bulk_load_nav_data(data_store, all_funds, target_date, previous_date)
        self._bulk_load_cash_data(data_store, all_funds, target_date, previous_date)
        self._bulk_load_distributions(data_store, all_funds, target_date)
        self._bulk_load_assignments(data_store, all_funds, target_date, previous_date)
        self._bulk_load_trades(data_store, all_funds, target_date, previous_date)
        self._bulk_load_index_data(
            data_store, all_funds, target_date, previous_date, tplus_one, previous_tplus_one
        )
        self._bulk_load_overlap_data(data_store, all_funds, target_date, previous_date)
        data_store.gics_mapping = self._load_gics_mapping()

        for fund_name, fund in all_funds.items():
            data_store.fund_data[fund_name] = normalize_all_holdings(
                fund_name,
                data_store.fund_data.get(fund_name, {}),
                fund_definition=fund.config,
                logger=self.logger,
            )

        self.logger.info("Bulk loaded data for %d funds", len(data_store.loaded_funds))
        return data_store

    # ------------------------------------------------------------------
    # GICS mapping
    # ------------------------------------------------------------------
    def _load_gics_mapping(self) -> pd.DataFrame:
        mapper_table = getattr(self.base_cls.classes, "gics_mapper", None)
        if mapper_table is None:
            self.logger.debug("gics_mapper table not reflected; skipping GICS load")
            return pd.DataFrame()

        query = self.session.query(
            mapper_table.GICS_SECTOR_NAME,
            mapper_table.GICS_INDUSTRY_GROUP_NAME,
            mapper_table.GICS_INDUSTRY_NAME,
        ).distinct()
        df = pd.read_sql(query.statement, self.session.bind)

        expected = ["GICS_SECTOR_NAME", "GICS_INDUSTRY_GROUP_NAME", "GICS_INDUSTRY_NAME"]
        missing = [col for col in expected if col not in df.columns]
        if missing:
            self.logger.warning("GICS mapping missing columns: %s", ", ".join(missing))
        return df

    # ------------------------------------------------------------------
    # Vest holdings
    # ------------------------------------------------------------------
    def _bulk_load_vest_holdings(
        self,
        data_store: BulkDataStore,
        all_funds: Dict[str, Fund],
        target_date: date,
        analysis_type: Optional[str],
        previous_date: Optional[date],
    ) -> None:
        query_builders: Dict[str, Callable[..., pd.DataFrame]] = {
            "vest_equity_holdings": self._query_vest_equity_holdings,
            "vest_options_holdings": self._query_vest_option_holdings,
            "vest_treasury_holdings": self._vest_treasury_holdings,
        }
        payload_keys = {
            "vest_equity_holdings": ("vest_equity", "vest_equity_t1"),
            "vest_options_holdings": ("vest_option", "vest_option_t1"),
            "vest_treasury_holdings": ("vest_treasury", "vest_treasury_t1"),
        }

        for config_key, query_builder in query_builders.items():
            payload_key, payload_key_t1 = payload_keys[config_key]
            table_to_funds = self._group_funds_by_table(all_funds, config_key)

            for table_name, funds in table_to_funds.items():
                if not self._is_real_table(table_name):
                    continue
                table = getattr(self.base_cls.classes, table_name, None)
                if table is None:
                    self.logger.warning("Vest table %s not reflected", table_name)
                    continue

                all_data = query_builder(table, funds, target_date, previous_date, analysis_type)

                for fund in funds:
                    fund_data = all_data[all_data["fund"] == fund.name].copy()
                    self._store_split(
                        data_store,
                        fund.name,
                        fund_data,
                        payload_key,
                        payload_key_t1,
                        target_date,
                        previous_date,
                    )

    def _query_vest_equity_holdings(
        self,
        table,
        funds: List[Fund],
        target_date: date,
        previous_date: Optional[date],
        analysis_type: Optional[str],
    ) -> pd.DataFrame:
        bbg_equity = getattr(self.base_cls.classes, "bbg_equity_flds_blotter", None)
        bbg_closes = getattr(self.base_cls.classes, "bbg_feed_equity_closes", None)
        ticker_column = getattr(table, "eqyticker", None)

        columns: List[Any] = [table]
        if bbg_equity is not None and ticker_column is not None:
            columns.extend(
                [
                    bbg_equity.GICS_SECTOR_NAME,
                    bbg_equity.GICS_INDUSTRY_NAME,
                    bbg_equity.GICS_INDUSTRY_GROUP_NAME,
                    bbg_equity.REGULATORY_STRUCTURE,
                    bbg_equity.SECURITY_TYP,
                ]
            )

        closes_subq = None
        if (
            bbg_closes is not None
            and ticker_column is not None
            and hasattr(bbg_closes, "BBG_SEC_ID")
        ):
            closes_subq = (
                self.session.query(
                    bbg_closes.BBG_SEC_ID.label("BBG_SEC_ID"),
                    bbg_closes.EQY_SH_OUT.label("EQY_SH_OUT"),
                    func.row_number()
                    .over(
                        partition_by=bbg_closes.BBG_SEC_ID,
                        order_by=bbg_closes.Date.desc(),
                    )
                    .label("row_num"),
                )
                .filter(bbg_closes.PX_MID.is_(None))
                .subquery("bbg_closes_subquery")
            )
            columns.extend(
                [
                    closes_subq.c.EQY_SH_OUT.label("EQY_SH_OUT"),
                    (closes_subq.c.EQY_SH_OUT * 1_000_000).label("EQY_SH_OUT_million"),
                ]
            )

        query = self.session.query(*columns)
        if bbg_equity is not None and ticker_column is not None:
            query = query.outerjoin(bbg_equity, ticker_column == bbg_equity.TICKER)
        if closes_subq is not None and ticker_column is not None:
            query = query.outerjoin(
                closes_subq,
                and_(
                    func.concat(ticker_column, literal(" US Equity")) == closes_subq.c.BBG_SEC_ID,
                    closes_subq.c.row_num == 1,
                ),
            )

        query = self._apply_date_filter(query, table.date, target_date, previous_date)
        query = query.filter(table.fund.in_([f.name for f in funds]))
        if analysis_type:
            query = query.filter(
                func.lower(func.trim(table.analysis_type)) == analysis_type.lower()
            )

        df = pd.read_sql(query.statement, self.session.bind)
        shares = self._resolve_shares_series(df, analysis_type=analysis_type)
        price = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        df["equity_market_value"] = shares * price
        return df

    def _query_vest_option_holdings(
        self,
        table,
        funds: List[Fund],
        target_date: date,
        previous_date: Optional[date],
        analysis_type: Optional[str],
    ) -> pd.DataFrame:
        bbg_equity = getattr(self.base_cls.classes, "bbg_equity_flds_blotter", None)
        underlying_column = getattr(table, "equity_underlying_ticker", None)

        columns: List[Any] = [table]
        if bbg_equity is not None and underlying_column is not None:
            columns.extend(
                [
                    bbg_equity.GICS_SECTOR_NAME,
                    bbg_equity.GICS_INDUSTRY_NAME,
                    bbg_equity.GICS_INDUSTRY_GROUP_NAME,
                    bbg_equity.REGULATORY_STRUCTURE,
                    bbg_equity.SECURITY_TYP,
                ]
            )

        query = self.session.query(*columns)
        if bbg_equity is not None and underlying_column is not None:
            query = query.outerjoin(bbg_equity, underlying_column == bbg_equity.TICKER)

        query = self._apply_date_filter(query, table.date, target_date, previous_date)
        query = query.filter(table.fund.in_([f.name for f in funds]))
        if analysis_type:
            query = query.filter(
                func.lower(func.trim(table.analysis_type)) == analysis_type.lower()
            )

        df = pd.read_sql(query.statement, self.session.bind)
        shares = self._resolve_shares_series(df, analysis_type=analysis_type)
        price = pd.to_numeric(df.get("price", 0.0), errors="coerce").fillna(0.0)
        delta = pd.to_numeric(df.get("delta", 0.0), errors="coerce").fillna(0.0)
        underlying_price = pd.to_numeric(df["equity_underlying_price"], errors="coerce").fillna(0.0)

        df["option_notional_value"] = underlying_price * shares * 100
        df["option_market_value"] = price * shares * 100
        df["option_delta_adjusted_notional"] = underlying_price * shares * delta * 100
        df["option_delta_adjusted_market_value"] = price * shares * delta * 100
        return df

    def _vest_treasury_holdings(
        self,
        table,
        funds: List[Fund],
        target_date: date,
        previous_date: Optional[date],
        analysis_type: Optional[str],
    ) -> pd.DataFrame:
        query = self.session.query(table)
        query = self._apply_date_filter(query, table.date, target_date, previous_date)
        query = query.filter(table.fund.in_([f.name for f in funds]))
        if analysis_type:
            query = query.filter(
                func.lower(func.trim(table.analysis_type)) == analysis_type.lower()
            )

        df = pd.read_sql(query.statement, self.session.bind)
        shares = self._resolve_shares_series(df, analysis_type=analysis_type)
        price = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        df["treasury_market_value"] = shares * price / 100
        return df

    def _resolve_shares_series(
        self,
        df: pd.DataFrame,
        *,
        analysis_type: Optional[str] = None,
    ) -> pd.Series:
        if df.empty:
            return pd.Series(0.0, index=df.index, dtype=float)

        analysis = (analysis_type or "").strip().lower()
        # Note: README says ex-ante -> iiv_shares, ex-post -> nav_shares. The current
        # logic priorities are reversed; preserved here pending confirmation.
        priority: List[str] = []
        if analysis == "ex_post":
            priority.append("iiv_shares")
        elif analysis == "ex_ante":
            priority.append("nav_shares")
        for fallback in ("quantity", "nav_shares", "iiv_shares", "shares", "units"):
            if fallback not in priority:
                priority.append(fallback)

        result = pd.Series(float("nan"), index=df.index, dtype=float)
        for col in priority:
            if col in df.columns:
                result = result.where(result.notna(), other=pd.to_numeric(df[col], errors="coerce"))
        return result.fillna(0.0)

    # ------------------------------------------------------------------
    # Custodian holdings
    # ------------------------------------------------------------------
    def _bulk_load_custodian_holdings(
        self,
        data_store: BulkDataStore,
        all_funds: Dict[str, Fund],
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        holdings_map = [
            ("custodian_equity_holdings", "custodian_equity", "custodian_equity_t1", "equity"),
            ("custodian_option_holdings", "custodian_option", "custodian_option_t1", "option"),
            ("custodian_treasury_holdings", "custodian_treasury", "custodian_treasury_t1", "treasury"),
        ]

        for config_key, payload_key, payload_key_t1, kind in holdings_map:
            table_to_funds = self._group_funds_by_table(all_funds, config_key)
            for table_name, funds in table_to_funds.items():
                if not self._is_real_table(table_name):
                    continue
                all_data = self._query_custodian_holdings_table(
                    table_name, kind, funds, target_date, previous_date
                )

                for fund in funds:
                    if "fund" in all_data.columns:
                        fund_data = all_data[all_data["fund"] == fund.name].copy()
                    else:
                        fund_data = all_data.copy()
                    self._store_split(
                        data_store,
                        fund.name,
                        fund_data,
                        payload_key,
                        payload_key_t1,
                        target_date,
                        previous_date,
                        date_candidates=("date", "process_date", "effective_date"),
                    )

    def _query_custodian_holdings_table(
        self,
        table_name: str,
        holdings_kind: str,
        funds: List[Fund],
        target_date: date,
        previous_date: Optional[date],
    ) -> pd.DataFrame:
        table = getattr(self.base_cls.classes, table_name, None)
        if table is None:
            self.logger.debug("Custodian holdings table %s not reflected; skipping", table_name)
            return pd.DataFrame()
        lower_name = table_name.lower()

        if "bny" in lower_name:
            query = self._build_bny_holdings_query(table, holdings_kind, target_date, previous_date)
        elif "umb" in lower_name:
            query = self._build_umb_holdings_query(table, holdings_kind, target_date, previous_date)
        elif "ccva_holdings" in lower_name:
            ccva_query = self._build_ccva_holdings_query(
                table, holdings_kind, target_date, previous_date
            )
            if isinstance(ccva_query, pd.DataFrame):
                return ccva_query
            return pd.read_sql(ccva_query.statement, self.session.bind)
        else:
            query = self.session.query(table)
            date_column = self._get_table_column(
                table, "date", "trade_date", "process_date", "effective_date"
            )
            if date_column is not None:
                query = self._apply_date_filter(query, date_column, target_date, previous_date)

        if query is None:
            return pd.DataFrame()

        fund_column = self._get_table_column(table, "fund", "fund_ticker", "account", "fund_id")
        aliases = self._collect_fund_aliases(funds)
        if fund_column is not None and aliases:
            query = query.filter(fund_column.in_(aliases))

        return pd.read_sql(query.statement, self.session.bind)

    def _build_bny_holdings_query(self, table, holdings_kind: str, target_date, previous_date):
        min_date = target_date if previous_date is None else min(target_date, previous_date)

        if holdings_kind == "equity":
            return (
                self.session.query(
                    table.date.label("date"),
                    table.fund.label("fund"),
                    table.security_sedol.label("sedol"),
                    table.sharespar.label("shares_cust"),
                    table.sharespar.label("quantity"),
                    table.category_description.label("category_description"),
                    table.asset_group.label("asset_group"),
                    table.price_base.label("price"),
                    table.traded_market_value_base.label("market_value"),
                )
                .filter(table.asset_group.notin_(["O", "UN", "ME", "MM", "CA", "TI", "B", "CU"]))
                .filter(table.date >= min_date)
            )

        if holdings_kind == "option":
            return (
                self.session.query(
                    table.date.label("date"),
                    table.fund.label("fund"),
                    table.security_description_long_1.label("optticker"),
                    table.sharespar.label("shares_cust"),
                    table.sharespar.label("quantity"),
                    table.category_description.label("category_description"),
                    table.asset_group.label("asset_group"),
                    table.price_base.label("price"),
                    table.traded_market_value_base.label("market_value"),
                    table.security_cins.label("cusip"),
                    table.maturity_date.label("maturity_date"),
                )
                .filter(table.asset_group.notin_(["S", "FS", "UN", "ME", "MM", "CA", "B", "TI", "CU"]))
                .filter(table.maturity_date >= min_date)
            )

        if holdings_kind == "treasury":
            return (
                self.session.query(
                    table.date.label("date"),
                    table.fund.label("fund"),
                    table.security_cins.label("cusip"),
                    table.security_sedol.label("sedol"),
                    table.pricing_number.label("ticker"),
                    table.maturity_date.label("maturity"),
                    table.sharespar.label("shares_cust"),
                    table.sharespar.label("quantity"),
                    table.price_base.label("price"),
                    table.traded_market_value_base.label("market_value"),
                    table.asset_group.label("asset_group"),
                )
                .filter(table.asset_group == "TI")
                .filter(table.date >= min_date)
            )

        return None

    def _build_umb_holdings_query(self, table, holdings_kind: str, target_date, previous_date):
        min_date = target_date if previous_date is None else min(target_date, previous_date)

        if holdings_kind == "equity":
            return (
                self.session.query(
                    table.process_date.label("date"),
                    table.fund.label("fund"),
                    table.security_tkr.label("eqyticker"),
                    table.mkt_qty.label("shares_cust"),
                    table.eod_close.label("price"),
                    table.security_catgry.label("category_description"),
                    table.mkt_mktval.label("market_value"),
                )
                .filter(table.security_catgry.in_(["COMMON", "REIT"]))
                .filter(table.process_date >= min_date)
            )

        if holdings_kind == "option":
            return (
                self.session.query(
                    table.process_date.label("date"),
                    table.fund.label("fund"),
                    func.concat(
                        func.substr(table.security_desc, 1, func.instr(table.security_desc, " ")),
                        "US",
                        func.substr(table.security_desc, func.instr(table.security_desc, " ")),
                    ).label("optticker"),
                    table.mkt_qty.label("shares_cust"),
                    table.eod_close.label("price"),
                    table.security_catgry.label("category_description"),
                    table.mkt_mktval.label("market_value"),
                )
                .filter(table.security_catgry.like("OPT%"))
                .filter(table.process_date >= min_date)
            )

        if holdings_kind == "treasury":
            return (
                self.session.query(
                    table.process_date.label("date"),
                    table.fund.label("fund"),
                    table.security_tkr.label("cusip"),
                    table.mkt_qty.label("shares_cust"),
                    table.eod_close.label("price"),
                    table.security_catgry.label("category_description"),
                    table.mkt_mktval.label("market_value"),
                )
                .filter(table.security_catgry.like("TSY%"))
                .filter(table.process_date >= min_date)
            )

        return self.session.query(table)

    def _build_ccva_holdings_query(self, table, holdings_kind: str, target_date, previous_date):
        min_date = target_date if previous_date is None else min(target_date, previous_date)

        if holdings_kind == "equity":
            return (
                self.session.query(
                    table.date.label("date"),
                    literal("KNGIX").label("fund"),
                    table.ticker.label("eqyticker"),
                    table.quantity.label("shares_cust"),
                    table.quantity.label("quantity"),
                    table.to_price_l.label("price"),
                    table.market_val_b.label("market_value"),
                    table.asset_class.label("category_description"),
                    table.cusip.label("cusip"),
                )
                .filter(
                    table.fund_id == 980,
                    table.asset_class.in_(["Equity", "EQUITY", "EQ"]),
                    table.date >= min_date,
                )
            )

        if holdings_kind == "option":
            return (
                self.session.query(
                    table.date.label("date"),
                    literal("KNGIX").label("fund"),
                    table.ticker.label("optticker"),
                    table.quantity.label("shares_cust"),
                    table.quantity.label("quantity"),
                    table.to_price_l.label("price"),
                    table.market_val_b.label("market_value"),
                    table.asset_class.label("category_description"),
                    table.cusip.label("cusip"),
                    table.maturity.label("maturity_date"),
                )
                .filter(
                    table.fund_id == 980,
                    table.asset_class.in_(["Option", "OPTION", "OPT"]),
                    table.date >= min_date,
                )
            )

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # NAV
    # ------------------------------------------------------------------
    def _bulk_load_nav_data(
        self,
        data_store: BulkDataStore,
        all_funds: Dict[str, Fund],
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        per_fund_fetchers: Dict[str, Callable[[Fund, date], pd.DataFrame]] = {
            "bny_us_nav_v2": self._fetch_bny_us_nav,
            "bny_vit_nav": self._fetch_bny_vit_nav,
            "umb_cef_nav": self._fetch_umb_cef_nav,
            "ccva_nav": self._fetch_ccva_nav,
        }

        table_to_funds = self._group_funds_by_table(all_funds, "custodian_navs")
        for table_name, funds in table_to_funds.items():
            if not self._is_real_table(table_name):
                continue

            fetcher = per_fund_fetchers.get(table_name)
            if fetcher is not None:
                self._iterate_per_fund(
                    data_store,
                    funds,
                    target_date,
                    previous_date,
                    fetch=fetcher,
                    payload_key="nav",
                    payload_key_t1="nav_t1",
                    date_candidates=("date", "valuation_date"),
                )
                continue

            # Generic fallback: query the whole table once and filter per fund.
            table = getattr(self.base_cls.classes, table_name, None)
            if table is None:
                self.logger.debug("NAV table %s not reflected; skipping", table_name)
                continue
            date_column = self._get_table_column(
                table, "date", "process_date", "valuation_date", "effective_date", "business_date"
            )
            query = self.session.query(table)
            if date_column is not None:
                query = self._apply_date_filter(query, date_column, target_date, previous_date)
            all_nav = pd.read_sql(query.statement, self.session.bind)

            for fund in funds:
                fund_nav = self._filter_by_fund(all_nav, fund.name, fund.config)
                self._store_split(
                    data_store,
                    fund.name,
                    fund_nav,
                    "nav",
                    "nav_t1",
                    target_date,
                    previous_date,
                )

    def _fetch_bny_us_nav(self, fund: Fund, nav_date: date) -> pd.DataFrame:
        account = self._account_number(fund)
        if not account:
            self.logger.warning("Skipping BNY US NAV for %s: no account_number", fund.name)
            return pd.DataFrame()

        table = getattr(self.base_cls.classes, "bny_us_nav_v2")
        query = (
            self.session.query(
                table.date.label("date"),
                table.account_number.label("account_number"),
                table.nav.label("nav"),
                table.total_net_assets.label("total_net_assets"),
                table.total_assets.label("total_assets"),
                table.income_earned.label("income_earned"),
                table.expenses.label("expenses"),
                table.net_invest_income.label("net_invest_income"),
                table.securities_at_value.label("securities_at_value"),
                table.cash.label("cash"),
                table.accrued_income.label("accrued_income"),
                table.accrued_expenses.label("accrued_expenses"),
                table.shares_outstanding.label("shares_outstanding"),
            )
            .filter(table.account_number == account)
            .filter(table.date == nav_date)
        )
        return pd.read_sql(query.statement, self.session.bind)

    def _fetch_bny_vit_nav(self, fund: Fund, nav_date: date) -> pd.DataFrame:
        identifier = self._fund_identifier(fund)
        table = getattr(self.base_cls.classes, "bny_vit_nav")
        query = self.session.query(
            table.valuation_date.label("date"),
            table.fund.label("fund"),
            table.nav_per_share_unrounded.label("nav"),
            table.total_net_assets.label("total_net_assets"),
            table.total_net_assets.label("total_assets"),
            table.market_value_base_cash.label("cash"),
            table.shares_outstanding.label("shares_outstanding"),
            literal(471.29).label("expenses"),
        ).filter(table.valuation_date == nav_date)
        if identifier and hasattr(table, "fund"):
            query = query.filter(table.fund == identifier)
        return pd.read_sql(query.statement, self.session.bind)

    def _fetch_umb_cef_nav(self, fund: Fund, nav_date: date) -> pd.DataFrame:
        # Historical quirk: RDATR's UMB rows are keyed under R21126.
        actual_fund_name = "R21126" if fund.name == "RDATR" else fund.name
        table = self.base_cls.classes.umb_cef_nav
        query = self.session.query(
            table.date.label("date"),
            table.fund.label("fund"),
            table.cntnavpershr.label("nav"),
            table.cntassets.label("total_assets"),
            table.cntadjnetasst.label("total_net_assets"),
            table.cntexpense.label("expenses"),
            table.cntshares.label("shares_outstanding"),
        ).filter(table.fund == actual_fund_name, table.date == nav_date)
        return pd.read_sql(query.statement, self.session.bind)

    def _fetch_ccva_nav(self, fund: Fund, nav_date: date) -> pd.DataFrame:
        table = getattr(self.base_cls.classes, "ccva_nav")
        query = self.session.query(
            table.date.label("date"),
            literal(fund.name).label("fund"),
            table.nav.label("nav"),
            table.revised_tna.label("total_net_assets"),
            table.revised_tna.label("total_assets"),
            table.tso.label("shares_outstanding"),
            literal(0.0).label("expenses"),
        ).filter(
            table.fund_number == 980,
            table.share_class == "Fund Total",
            table.date == nav_date,
        )
        return pd.read_sql(query.statement, self.session.bind)

    # ------------------------------------------------------------------
    # Cash
    # ------------------------------------------------------------------
    def _bulk_load_cash_data(
        self,
        data_store: BulkDataStore,
        all_funds: Dict[str, Fund],
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        per_fund_fetchers: Dict[str, Callable[[Fund, date], pd.DataFrame]] = {
            "bny_us_cash": self._fetch_bny_us_cash,
            "bny_vit_cash": self._fetch_bny_vit_cash,
            "umb_cef_cash": self._fetch_umb_cef_cash,
            "ccva_cash": self._fetch_ccva_cash,
        }

        table_to_funds = self._group_funds_by_table(all_funds, "cash_table")
        for table_name, funds in table_to_funds.items():
            if not self._is_real_table(table_name):
                continue

            fetcher = per_fund_fetchers.get(table_name)
            if fetcher is not None:
                self._iterate_per_fund(
                    data_store,
                    funds,
                    target_date,
                    previous_date,
                    fetch=fetcher,
                    payload_key="cash",
                    payload_key_t1="cash_t1",
                )
                continue

            table = getattr(self.base_cls.classes, table_name, None)
            if table is None:
                self.logger.debug("Cash table %s not reflected; skipping", table_name)
                continue
            date_column = self._get_table_column(
                table, "date", "business_date", "nav_date", "process_date"
            )
            query = self.session.query(table)
            if date_column is not None:
                query = self._apply_date_filter(query, date_column, target_date, previous_date)
            all_cash = pd.read_sql(query.statement, self.session.bind)

            for fund in funds:
                fund_cash = self._filter_by_fund(all_cash, fund.name, fund.config)
                self._store_split(
                    data_store,
                    fund.name,
                    fund_cash,
                    "cash",
                    "cash_t1",
                    target_date,
                    previous_date,
                    date_candidates=("date", "business_date", "nav_date"),
                )

    def _fetch_bny_us_cash(self, fund: Fund, cash_date: date) -> pd.DataFrame:
        account = self._account_number(fund)
        if not account:
            self.logger.warning("Skipping BNY US cash for %s: no account_number", fund.name)
            return pd.DataFrame()

        table = getattr(self.base_cls.classes, "bny_us_cash")
        latest = (
            self.session.query(
                table.account_number.label("account_number"),
                table.date.label("date"),
                func.max(table.update_time).label("max_update_time"),
            )
            .filter(table.account_number == account)
            .filter(table.date == cash_date)
            .group_by(table.account_number, table.date)
            .subquery()
        )
        query = (
            self.session.query(
                table.account_number.label("account_number"),
                table.date.label("date"),
                func.sum(table.end_balance).label("cash_value"),
            )
            .join(
                latest,
                and_(
                    table.account_number == latest.c.account_number,
                    table.date == latest.c.date,
                    table.update_time == latest.c.max_update_time,
                ),
            )
            .group_by(table.account_number, table.date)
        )
        return pd.read_sql(query.statement, self.session.bind)

    def _fetch_bny_vit_cash(self, fund: Fund, cash_date: date) -> pd.DataFrame:
        identifier = self._fund_identifier(fund)
        table = getattr(self.base_cls.classes, "bny_vit_cash")
        query = self.session.query(
            table.fund.label("fund"),
            table.date.label("date"),
            func.sum(table.cash_balance).label("cash_value"),
        ).filter(table.date == cash_date)
        if identifier and hasattr(table, "fund"):
            query = query.filter(table.fund == identifier)
        query = query.group_by(table.fund, table.date)
        return pd.read_sql(query.statement, self.session.bind)

    def _fetch_umb_cef_cash(self, fund: Fund, cash_date: date) -> pd.DataFrame:
        table = self.base_cls.classes.umb_cef_cash
        cash_value = (
            self.session.query(func.sum(table.current_balance).label("cash_value"))
            .filter(table.fund == fund.name, table.as_of_date == cash_date)
            .scalar()
        )
        if cash_value is None:
            return pd.DataFrame()
        return pd.DataFrame([{"fund": fund.name, "date": cash_date, "cash_value": cash_value}])

    def _fetch_ccva_cash(self, fund: Fund, cash_date: date) -> pd.DataFrame:
        # CCVA cash lives inside the holdings table, filtered on asset_class.
        table = getattr(self.base_cls.classes, "ccva_holdings")
        query = (
            self.session.query(
                table.date.label("date"),
                literal(fund.name).label("fund"),
                func.sum(table.market_val_b).label("cash_value"),
            )
            .filter(
                table.fund_id == 980,
                table.asset_class == "Cash Management Vehicle",
                table.date == cash_date,
            )
            .group_by(table.date)
        )
        return pd.read_sql(query.statement, self.session.bind)

    # ------------------------------------------------------------------
    # Distributions / Assignments / Overlap / Trades
    # ------------------------------------------------------------------
    def _bulk_load_distributions(
        self,
        data_store: BulkDataStore,
        all_funds: Dict[str, Fund],
        target_date: date,
    ) -> None:
        table_to_funds = self._group_funds_by_table(all_funds, "distributions")
        for table_name, funds in table_to_funds.items():
            if not self._is_real_table(table_name):
                continue
            table = getattr(self.base_cls.classes, table_name, None)
            if table is None:
                self.logger.debug("Distributions table %s not reflected", table_name)
                continue

            query = self.session.query(
                table.fund,
                table.declaration_date,
                table.ex_date,
                table.record_date,
                table.reinvest_date,
                table.payable_date,
                table.distro_amt,
            ).filter(table.ex_date == target_date)
            df = pd.read_sql(query.statement, self.session.bind)

            for col in ("declaration_date", "ex_date", "record_date", "reinvest_date", "payable_date"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            if "distro_amt" in df.columns:
                df["distro_amt"] = pd.to_numeric(df["distro_amt"], errors="coerce")

            for fund in funds:
                fund_df = df[df["fund"] == fund.name].copy() if "fund" in df.columns else df.copy()
                self._store_fund_data(data_store, fund.name, "distributions", fund_df)

    def _bulk_load_assignments(
        self,
        data_store: BulkDataStore,
        all_funds: Dict[str, Fund],
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        table_to_funds = self._group_funds_by_table(all_funds, "option_custodian_assignment")
        for table_name, funds in table_to_funds.items():
            if not self._is_real_table(table_name):
                continue
            table = getattr(self.base_cls.classes, table_name, None)
            if table is None:
                self.logger.debug("Assignment table %s not reflected", table_name)
                continue

            query = self.session.query(table)
            date_column = self._get_table_column(
                table, "date", "process_date", "effective_date", "assignment_date", "trade_date"
            )
            if date_column is not None:
                query = self._apply_date_filter(query, date_column, target_date, previous_date)
            df = pd.read_sql(query.statement, self.session.bind)

            for fund in funds:
                filtered = self._filter_by_fund(df, fund.name, fund.config)
                self._store_split(
                    data_store,
                    fund.name,
                    filtered,
                    "assignments",
                    "assignments_t1",
                    target_date,
                    previous_date,
                    date_candidates=(
                        "date",
                        "process_date",
                        "effective_date",
                        "assignment_date",
                        "trade_date",
                    ),
                )

    def _bulk_load_overlap_data(
        self,
        data_store: BulkDataStore,
        all_funds: Dict[str, Fund],
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        for fund_name, fund in all_funds.items():
            if not fund.is_closed_end_fund:
                continue
            current = self._query_overlap_table(fund, target_date)
            previous = (
                self._query_overlap_table(fund, previous_date)
                if previous_date is not None
                else pd.DataFrame()
            )
            if not current.empty:
                self._store_fund_data(data_store, fund_name, "overlap", current)
            if not previous.empty:
                self._store_fund_data(data_store, fund_name, "overlap_t1", previous)

    def _query_overlap_table(self, fund: Fund, target_date: Optional[date]) -> pd.DataFrame:
        if not target_date:
            return pd.DataFrame()
        table_name = fund.config.get("overlap_table")
        if not table_name:
            return pd.DataFrame()
        table = getattr(self.base_cls.classes, table_name, None)
        if table is None:
            self.logger.debug("Overlap table %s not reflected", table_name)
            return pd.DataFrame()

        query = self.session.query(
            table.security_ticker.label("security_ticker"),
            table.security_weight.label("security_weight"),
        ).filter(table.date == target_date)
        benchmark = fund.config.get("overlap_benchmark_ticker")
        if benchmark:
            query = query.filter(table.etf_ticker == benchmark)

        df = pd.read_sql(query.statement, self.session.bind)
        if not df.empty:
            df["security_weight"] = (
                pd.to_numeric(df["security_weight"], errors="coerce").fillna(0) / 100
            )
        return df

    def _bulk_load_trades(
        self,
        data_store: BulkDataStore,
        all_funds: Dict[str, Fund],
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        for fund in all_funds.values():
            equity_t = self._get_equity_trades(fund, target_date)
            option_t = self._get_option_trades(fund, target_date)
            flex_t = self._split_flex_trades(fund, option_t)
            treasury_t = self._get_treasury_trades(fund, target_date)

            self._store_fund_data(data_store, fund.name, "equity_trades", equity_t)
            self._store_fund_data(data_store, fund.name, "option_trades", option_t)
            self._store_fund_data(data_store, fund.name, "flex_option_trades", flex_t)
            self._store_fund_data(data_store, fund.name, "treasury_trades", treasury_t)

            if previous_date is None:
                continue

            prev_equity = self._get_equity_trades(fund, previous_date)
            prev_option = self._get_option_trades(fund, previous_date)
            prev_flex = self._split_flex_trades(fund, prev_option)
            prev_treasury = self._get_treasury_trades(fund, previous_date)

            self._store_fund_data(data_store, fund.name, "equity_trades_t1", prev_equity)
            self._store_fund_data(data_store, fund.name, "option_trades_t1", prev_option)
            self._store_fund_data(data_store, fund.name, "flex_option_trades_t1", prev_flex)
            self._store_fund_data(data_store, fund.name, "treasury_trades_t1", prev_treasury)

    def _get_trades_data(self, fund: Fund, trade_date: date) -> pd.DataFrame:
        """Cached fetch of EMSX trades for one (fund, date). Re-used by equity/option/treasury."""
        cache_key = (fund.name, trade_date)
        cached = self._trades_cache.get(cache_key)
        if cached is not None:
            return cached

        EmxsOrder = self.base_cls.classes.emsx_equity_order_sub
        EmxsRoute = self.base_cls.classes.emsx_equity_route_sub
        query = (
            self.session.query(
                EmxsRoute.emsx_route_as_of_date.label("date"),
                literal(fund.name).label("fund"),
                EmxsOrder.emsx_asset_class.label("type"),
                EmxsOrder.emsx_ticker.label("ticker"),
                EmxsOrder.emsx_side.label("side"),
                EmxsOrder.emsx_amount.label("quantity"),
                EmxsRoute.emsx_avg_price.label("price"),
            )
            .join(EmxsRoute, EmxsOrder.emsx_sequence == EmxsRoute.emsx_sequence)
            .filter(
                EmxsOrder.emsx_order_as_of_date == trade_date,
                EmxsRoute.emsx_custom_account == fund.name,
                EmxsOrder.emsx_status == "FILLED",
                EmxsRoute.emsx_route_as_of_date == EmxsOrder.emsx_order_as_of_date,
                EmxsRoute.emsx_status == "FILLED",
            )
            .order_by("date", "ticker", EmxsOrder.emsx_side.desc())
        )
        df = pd.read_sql(query.statement, self.session.bind)
        self._trades_cache[cache_key] = df
        return df

    def _get_equity_trades(self, fund: Fund, trade_date: date) -> pd.DataFrame:
        trades = self._get_trades_data(fund, trade_date)
        if trades.empty:
            return pd.DataFrame()
        return trades[trades["type"] == "Equity"].copy()

    def _get_option_trades(self, fund: Fund, trade_date: date) -> pd.DataFrame:
        if fund.name in PRIVATE_FUNDS or fund.name in CLOSED_END_FUNDS:
            return self._get_block_option_trades(fund, trade_date)

        trades = self._get_trades_data(fund, trade_date)
        if trades.empty:
            return pd.DataFrame()
        options = trades[trades["type"] == "Option"].copy()
        if options.empty:
            return pd.DataFrame()

        options["ticker"] = options["ticker"].str.replace(" Equity", "", regex=False)
        options["ticker"] = options["ticker"].str.replace(" Index", "", regex=False)
        options.rename(columns={"ticker": "optticker"}, inplace=True)
        return options

    def _get_block_option_trades(self, fund: Fund, trade_date: date) -> pd.DataFrame:
        sql = text(
            """
            SELECT
                order_as_of_date as date,
                account as fund,
                short_ticker as optticker,
                long_ticker,
                isin,
                emsx_side as side,
                emsx_filled as quantity,
                emsx_avg_price as price,
                emsx_principal as principal
            FROM ftcm.vw_combined_block_opt
            WHERE account = :account_name
              AND order_as_of_date = :trade_date
            ORDER BY order_as_of_date, short_ticker, emsx_side DESC
            """
        )
        result = self.session.execute(
            sql, {"account_name": fund.name, "trade_date": trade_date}
        )
        df = pd.DataFrame(result.fetchall())
        if df.empty:
            return df

        df["optticker"] = df["optticker"].str.replace("BRK/B", "BRKB", regex=False)
        for col in ("quantity", "price", "principal"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _get_treasury_trades(self, fund: Fund, trade_date: date) -> pd.DataFrame:
        trades = self._get_trades_data(fund, trade_date)
        if trades.empty or "type" not in trades.columns:
            return pd.DataFrame()
        mask = trades["type"].str.contains("Treasury", case=False, na=False)
        return trades[mask].copy()

    def _split_flex_trades(self, fund: Fund, option_trades: pd.DataFrame) -> pd.DataFrame:
        if option_trades.empty or not fund.has_flex_option:
            return pd.DataFrame()
        if "optticker" not in option_trades.columns:
            return pd.DataFrame()
        mask = option_trades["optticker"].str.contains(
            fund.flex_option_pattern, na=False, regex=True
        )
        return option_trades[mask].copy()

    # ------------------------------------------------------------------
    # Index data
    # ------------------------------------------------------------------
    def _bulk_load_index_data(
        self,
        data_store: BulkDataStore,
        all_funds: Dict[str, Fund],
        target_date: date,
        previous_date: Optional[date],
        tplus_one: Optional[date],
        previous_tplus_one: Optional[date],
    ) -> None:
        provider_map: Dict[str, Callable[..., Any]] = {
            "nasdaq_pro": self._get_nasdaq_holdings,
            "sp_holdings": self._get_sp_holdings,
            "cboe_holdings": self._get_cboe_holdings,
            "dogg_index": self._get_dogg_index,
        }

        table_to_funds = self._group_funds_by_table(all_funds, "index_holdings")
        for table_name, funds in table_to_funds.items():
            if not self._is_real_table(table_name):
                continue
            handler = provider_map.get(table_name)

            for fund in funds:
                if handler is not None:
                    result = handler(
                        target_date, fund, previous_date, tplus_one, previous_tplus_one
                    )
                else:
                    result = self._get_generic_index_data(
                        table_name, target_date, fund, previous_date
                    )

                if isinstance(result, tuple):
                    current, previous = result
                else:
                    current, previous = result, pd.DataFrame()

                self._store_fund_data(data_store, fund.name, "index", current)
                if previous_date is not None:
                    self._store_fund_data(data_store, fund.name, "index_t1", previous)

    def _get_generic_index_data(
        self,
        table_name: str,
        target_date: date,
        fund: Fund,
        previous_date: Optional[date],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        table = getattr(self.base_cls.classes, table_name, None)
        if table is None:
            self.logger.debug("Generic index table %s not reflected; skipping", table_name)
            return pd.DataFrame(), pd.DataFrame()
        date_column = getattr(table, "date", None) or next(
            (
                col
                for col in (
                    getattr(table, "effective_date", None),
                    getattr(table, "business_date", None),
                    getattr(table, "trade_date", None),
                )
                if col is not None
            ),
            None,
        )
        query = self.session.query(table)
        if date_column is not None:
            query = self._apply_date_filter(query, date_column, target_date, previous_date)
        df = pd.read_sql(query.statement, self.session.bind)
        filtered = self._filter_by_fund(df, fund.name, fund.config)
        return self._split_current_previous(
            filtered,
            self._find_date_column(
                filtered, "date", "effective_date", "business_date", "trade_date"
            ),
            target_date,
            previous_date,
        )

    def _get_nasdaq_holdings(
        self,
        target_date: date,
        fund: Fund,
        previous_date: Optional[date],
        tplus_one: Optional[date],
        previous_tplus_one: Optional[date],
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        nasdaq = getattr(self.base_cls.classes, "nasdaq_pro")
        bbg = getattr(self.base_cls.classes, "bbg_equity_flds_blotter", None)
        columns = [
            nasdaq.file_date.label("date"),
            nasdaq.fund.label("fund"),
            nasdaq.ticker.label("eqyticker"),
            nasdaq.index_weight.label("weight_index"),
            nasdaq.price.label("price_index"),
        ]
        if bbg is not None:
            columns.extend(
                [bbg.GICS_SECTOR_NAME, bbg.GICS_INDUSTRY_NAME, bbg.GICS_INDUSTRY_GROUP_NAME]
            )

        query = self.session.query(*columns)
        if bbg is not None:
            query = query.join(bbg, nasdaq.ticker == bbg.TICKER)
        query = query.filter(nasdaq.fund == fund.index_ticker_join)

        previous_filter_date = None
        if previous_date is not None:
            previous_filter_date = previous_tplus_one or previous_date
            query = query.filter(nasdaq.file_date.in_([tplus_one, previous_filter_date]))
        else:
            query = query.filter(nasdaq.file_date == tplus_one)
        query = query.group_by(nasdaq.ticker)

        df = pd.read_sql(query.statement, self.session.bind)
        current, previous = self._split_current_previous(
            df, "date", tplus_one, previous_filter_date
        )
        return (current, previous) if previous_date is not None else current

    def _get_sp_holdings(
        self,
        target_date: date,
        fund: Fund,
        previous_date: Optional[date],
        tplus_one: Optional[date],
        previous_tplus_one: Optional[date],
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        sp = getattr(self.base_cls.classes, "sp_holdings")
        bbg = getattr(self.base_cls.classes, "bbg_equity_flds_blotter", None)
        columns = [
            sp.EFFECTIVE_DATE.label("date"),
            sp.INDEX_CODE.label("fund"),
            sp.TICKER.label("eqyticker"),
            sp.INDEX_WEIGHT.label("weight_index"),
            sp.LOCAL_PRICE.label("price_index"),
        ]
        if bbg is not None:
            columns.extend(
                [bbg.GICS_SECTOR_NAME, bbg.GICS_INDUSTRY_NAME, bbg.GICS_INDUSTRY_GROUP_NAME]
            )

        query = self.session.query(*columns)
        if bbg is not None:
            query = query.join(bbg, sp.TICKER == bbg.TICKER)
        query = query.filter(sp.INDEX_CODE == fund.index_ticker_join)

        previous_filter_date = None
        if previous_date is not None:
            previous_filter_date = previous_tplus_one or previous_date
            query = query.filter(sp.EFFECTIVE_DATE.in_([tplus_one, previous_filter_date]))
        else:
            query = query.filter(sp.EFFECTIVE_DATE == tplus_one)

        df = pd.read_sql(query.statement, self.session.bind)
        current, previous = self._split_current_previous(
            df, "date", tplus_one, previous_filter_date
        )
        return (current, previous) if previous_date is not None else current

    def _get_cboe_holdings(
        self,
        target_date: date,
        fund: Fund,
        previous_date: Optional[date],
        tplus_one: Optional[date],
        previous_tplus_one: Optional[date],
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        cboe = getattr(self.base_cls.classes, "cboe_holdings")
        bbg = getattr(self.base_cls.classes, "bbg_equity_flds_blotter", None)
        columns = [
            cboe.date.label("date"),
            cboe.index_name.label("fund"),
            cboe.ticker.label("eqyticker"),
            cboe.stock_weight.label("weight_index"),
            cboe.price.label("price_index"),
        ]
        if bbg is not None:
            columns.extend(
                [bbg.GICS_SECTOR_NAME, bbg.GICS_INDUSTRY_NAME, bbg.GICS_INDUSTRY_GROUP_NAME]
            )

        query = self.session.query(*columns)
        if bbg is not None:
            query = query.join(bbg, cboe.ticker == bbg.TICKER)
        query = query.filter(cboe.index_name == fund.index_ticker_join)
        query = self._apply_date_filter(query, cboe.date, target_date, previous_date)

        df = pd.read_sql(query.statement, self.session.bind)
        current, previous = self._split_current_previous(df, "date", target_date, previous_date)
        return (current, previous) if previous_date is not None else current

    def _get_dogg_index(
        self,
        target_date: date,
        fund: Fund,
        previous_date: Optional[date],
        tplus_one: Optional[date],
        previous_tplus_one: Optional[date],
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        dogg = getattr(self.base_cls.classes, "dogg_index")
        bbg = getattr(self.base_cls.classes, "bbg_equity_flds_blotter", None)
        columns = [
            dogg.DATE.label("date"),
            literal(fund.name).label("fund"),
            dogg.TICKER.label("eqyticker"),
            literal(0.10).label("weight_index"),
        ]
        if bbg is not None:
            columns.extend(
                [bbg.GICS_SECTOR_NAME, bbg.GICS_INDUSTRY_NAME, bbg.GICS_INDUSTRY_GROUP_NAME]
            )

        query = self.session.query(*columns)
        if bbg is not None:
            query = query.join(bbg, dogg.TICKER == bbg.TICKER)
        query = self._apply_date_filter(query, dogg.DATE, target_date, previous_date)

        df = pd.read_sql(query.statement, self.session.bind)
        current, previous = self._split_current_previous(df, "date", target_date, previous_date)
        return (current, previous) if previous_date is not None else current

    # ------------------------------------------------------------------
    # Per-fund iterator (used by NAV/cash per-fund fetchers)
    # ------------------------------------------------------------------
    def _iterate_per_fund(
        self,
        data_store: BulkDataStore,
        funds: List[Fund],
        target_date: date,
        previous_date: Optional[date],
        *,
        fetch: Callable[[Fund, date], pd.DataFrame],
        payload_key: str,
        payload_key_t1: str,
        date_candidates: Tuple[str, ...] = ("date",),
    ) -> None:
        for fund in funds:
            frames: List[pd.DataFrame] = []
            current_df = fetch(fund, target_date)
            if not current_df.empty:
                frames.append(current_df.assign(fund=fund.name))

            if previous_date is not None:
                previous_df = fetch(fund, previous_date)
                if not previous_df.empty:
                    frames.append(previous_df.assign(fund=fund.name))

            combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            self._store_split(
                data_store,
                fund.name,
                combined,
                payload_key,
                payload_key_t1,
                target_date,
                previous_date,
                date_candidates=date_candidates,
            )

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------
    def _store_split(
        self,
        data_store: BulkDataStore,
        fund_name: str,
        df: pd.DataFrame,
        payload_key: str,
        payload_key_t1: str,
        target_date: date,
        previous_date: Optional[date],
        date_candidates: Tuple[str, ...] = ("date",),
    ) -> None:
        date_column = self._find_date_column(df, *date_candidates)
        current, previous = self._split_current_previous(
            df, date_column, target_date, previous_date
        )
        self._store_fund_data(data_store, fund_name, payload_key, current)
        if previous_date is not None:
            self._store_fund_data(data_store, fund_name, payload_key_t1, previous)

    def _store_fund_data(
        self,
        data_store: BulkDataStore,
        fund_name: str,
        data_type: str,
        data: pd.DataFrame,
    ) -> None:
        if fund_name not in data_store.fund_data:
            data_store.fund_data[fund_name] = {}
        data_store.fund_data[fund_name][data_type] = data
        data_store.loaded_funds.add(fund_name)
        data_store.loaded_data_types.add(data_type)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_real_table(table_name: Optional[str]) -> bool:
        return bool(table_name) and table_name != "NULL"

    @staticmethod
    def _business_day_shift(anchor: Optional[date], offset: int = 1) -> Optional[date]:
        if anchor is None:
            return None
        if offset == 0:
            return anchor
        return (pd.Timestamp(anchor) + BDay(offset)).date()

    def _apply_date_filter(self, query, column, target_date: date, previous_date: Optional[date]):
        if previous_date is not None:
            return query.filter(column.in_([target_date, previous_date]))
        return query.filter(column == target_date)

    def _group_funds_by_table(
        self, all_funds: Dict[str, Fund], config_key: str
    ) -> Dict[str, List[Fund]]:
        groups: Dict[str, List[Fund]] = {}
        for fund in all_funds.values():
            table_name = fund.config.get(config_key)
            groups.setdefault(table_name, []).append(fund)
        return groups

    def _get_table_column(self, table, *names: str):
        for name in names:
            column = getattr(table, name, None)
            if column is not None:
                return column
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
        normalized[date_column] = pd.to_datetime(normalized[date_column], errors="coerce").dt.date

        current = normalized[normalized[date_column] == target_date].copy()
        if previous_date is None:
            return current, pd.DataFrame()
        previous = normalized[normalized[date_column] == previous_date].copy()
        return current, previous

    def _collect_fund_aliases(self, funds: List[Fund]) -> List[str]:
        aliases: Set[str] = set()
        for fund in funds:
            aliases.add(fund.name)
            for key in ("fund", "fund_ticker", "portfolio", "account", "account_number", "fund_number"):
                value = fund.config.get(key)
                if isinstance(value, str) and value and value != "NULL":
                    aliases.add(value)
        return sorted(aliases)

    def _filter_by_fund(
        self, df: pd.DataFrame, fund_name: str, mapping: Dict[str, Any]
    ) -> pd.DataFrame:
        if df.empty:
            return df
        lower_columns = {col.lower(): col for col in df.columns}
        for candidate in ("fund", "fund_ticker", "account"):
            column = lower_columns.get(candidate)
            if column is None:
                continue
            value = mapping.get(candidate, fund_name) if candidate == "account" else fund_name
            filtered = df[df[column] == value].copy()
            if not filtered.empty:
                return filtered
        return df.copy()

    # ------------------------------------------------------------------
    # Fund-config accessors
    # ------------------------------------------------------------------
    def _account_number(self, fund: Fund) -> Optional[str]:
        return self._first_string_value(
            fund.config, "account_number_custodian", "account_number"
        ) or self._first_account_number_from_block(fund.config)

    def _fund_identifier(self, fund: Fund) -> str:
        identifier = self._first_string_value(
            fund.config, "fund", "fund_name", "fund_ticker", "account_number_custodian"
        )
        return identifier or fund.name

    @staticmethod
    def _first_string_value(mapping: Dict[str, Any], *keys: str) -> Optional[str]:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed and trimmed.upper() != "NULL":
                    return trimmed
            elif value:
                return value
        return None

    @staticmethod
    def _first_account_number_from_block(mapping: Dict[str, Any]) -> Optional[str]:
        block = mapping.get("account_numbers")
        if not isinstance(block, dict):
            return None
        for value in block.values():
            if isinstance(value, str) and value.strip() and value.strip().upper() != "NULL":
                return value.strip()
            if isinstance(value, (list, tuple)) and value:
                first = value[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
        return None
