"""SQL-backed bulk data loader.

Every business-data fetch is a view in `reconciliation.*`. This loader's
only job is: call each view once with the run's (dates, funds) params,
slice the result by fund_ticker, store it. The views own:
  - per-custodian routing
  - GICS / Bloomberg enrichment
  - ticker normalization
  - fund_ticker resolution via accounts_mapping.vw_tif_account_numbers

The loader owns:
  - target_date / previous_date splitting
  - per-source T vs T+1 routing for v_index_holdings (the one data-quirk
    that doesn't belong in SQL)
  - merging v_fund_metadata into each Fund object for downstream services
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from pandas.tseries.offsets import BDay
from sqlalchemy import bindparam, text

from processing.fund import Fund
from utilities.ticker_utils import normalize_all_holdings


VIEW_SCHEMA = "reconciliation"

# Per-source date conventions for v_index_holdings. CBOE/DOGG publish for T,
# NASDAQ/S&P publish for T+1. The view does no transformation; the loader
# routes each (fund, source) slice to the appropriate date pair.
_INDEX_SOURCE_DATES = {
    "cboe": "target", "dogg": "target",
    "nasdaq": "tplus_one", "sp": "tplus_one",
}


@dataclass
class BulkDataStore:
    """All data for a run. No DB calls after load completes."""
    date: str
    fund_data: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)
    fund_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gics_mapping: pd.DataFrame = field(default_factory=pd.DataFrame)
    loaded_funds: Set[str] = field(default_factory=set)
    loaded_data_types: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ViewSpec:
    """One row of the load plan: view -> payload keys."""
    view: str
    key: str
    key_t1: Optional[str] = None
    extra_where: Optional[str] = None
    target_only: bool = False  # ignore previous_date (e.g. distributions)
    analysis_type_filter: bool = False  # apply analysis_type filter (vest views)


VIEWS: Tuple[ViewSpec, ...] = (
    # Custodian holdings + cash + nav
    ViewSpec("v_custodian_equity",            "custodian_equity",   "custodian_equity_t1"),
    ViewSpec("v_custodian_option",            "custodian_option",   "custodian_option_t1"),
    ViewSpec("v_custodian_treasury",          "custodian_treasury", "custodian_treasury_t1"),
    ViewSpec("v_custodian_nav",               "nav",                "nav_t1"),
    ViewSpec("v_custodian_cash",              "cash",               "cash_t1"),
    ViewSpec("v_recon_bbg_equity",            "bbg_unresolved",     "bbg_unresolved_t1",
             extra_where="resolution_status = 'Unresolved'"),

    # Vest (OMS) holdings
    ViewSpec("v_vest_equity",                 "vest_equity",        "vest_equity_t1", analysis_type_filter=True),
    ViewSpec("v_vest_option",                 "vest_option",        "vest_option_t1", analysis_type_filter=True),
    ViewSpec("v_vest_treasury",               "vest_treasury",      "vest_treasury_t1", analysis_type_filter=True),

    # Lifecycle events
    ViewSpec("v_fund_distributions",          "distributions",      None, target_only=True),
    ViewSpec("v_option_custodian_assignment", "assignments",        "assignments_t1"),
    ViewSpec("v_overlap",                     "overlap",            "overlap_t1"),

    # Trades
    ViewSpec("v_emsx_trades",                 "trades",             "trades_t1"),
    ViewSpec("v_block_options",               "block_options",      "block_options_t1"),

    # ETF-specific
    ViewSpec("v_etf_basket",                  "basket",             "basket_t1"),
    ViewSpec("v_etf_flows",                   "flows",              "flows_t1"),

    # SG cross-check
    ViewSpec("v_sg_custodian_holdings",       "sg_holdings",        "sg_holdings_t1"),
)


class BulkDataLoader:
    """Loads all data for all funds via the reconciliation.* views."""

    def __init__(self, session, fund_registry: Dict[str, Fund]):
        self.session = session
        self.fund_registry = fund_registry
        self.logger = logging.getLogger(__name__)

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
        store = BulkDataStore(date=str(target_date))
        funds = list(self.fund_registry.keys())
        dates = [d for d in (target_date, previous_date) if d is not None]

        self._load_metadata(store, funds)

        for spec in VIEWS:
            spec_dates = [target_date] if spec.target_only else dates
            df = self._query_view(spec, dates=spec_dates, funds=funds, analysis_type=analysis_type)
            self._fan_out(store, df, spec, target_date, previous_date)

        store.gics_mapping = self._query_sql(
            f"SELECT * FROM {VIEW_SCHEMA}.v_gics_mapping"
        )
        self._load_index_data(store, target_date, previous_date)
        self._split_trades_into_asset_classes(store, previous_date)

        for name, fund in self.fund_registry.items():
            metadata = store.fund_metadata.get(name)
            if metadata:
                fund.config.update({k: v for k, v in metadata.items() if v is not None})
                fund.expense_ratio = float(fund.config.get("expense_ratio") or 0.0)
            store.fund_data[name] = normalize_all_holdings(
                name,
                store.fund_data.get(name, {}),
                fund_definition=fund.config,
                logger=self.logger,
            )

        self.logger.info("Bulk loaded data for %d funds", len(store.loaded_funds))
        return store

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def _load_metadata(self, store: BulkDataStore, funds: List[str]) -> None:
        if not funds:
            return
        df = self._query_sql(
            f"SELECT * FROM {VIEW_SCHEMA}.v_fund_metadata WHERE fund_ticker IN :funds",
            funds=funds,
        )
        for row in df.to_dict(orient="records"):
            store.fund_metadata[row["fund_ticker"]] = row

    def _query_view(
            self,
            spec: ViewSpec,
            *,
            dates: List[date],
            funds: List[str],
            analysis_type: Optional[str] = None,
    ) -> pd.DataFrame:
        if not dates or not funds:
            return pd.DataFrame()
        where = "`date` IN :dates AND fund_ticker IN :funds"
        params: Dict[str, Any] = {"dates": dates, "funds": funds}
        if spec.extra_where:
            where += f" AND {spec.extra_where}"
        if spec.analysis_type_filter and analysis_type:
            where += " AND LOWER(TRIM(analysis_type)) = :analysis_type"
            params["analysis_type"] = analysis_type.lower()
        sql = f"SELECT * FROM {VIEW_SCHEMA}.{spec.view} WHERE {where}"
        return self._query_sql(sql, **params)


    def _query_sql(self, sql: str, **params: Any) -> pd.DataFrame:
        stmt = text(sql)
        binds = [
            bindparam(k, expanding=True)
            for k, v in params.items()
            if isinstance(v, (list, tuple, set))
        ]
        if binds:
            stmt = stmt.bindparams(*binds)
        return pd.read_sql(stmt, self.session.bind, params=params or None)

    # ------------------------------------------------------------------
    # Per-fund fanout / date splitting
    # ------------------------------------------------------------------
    def _fan_out(
        self,
        store: BulkDataStore,
        df: pd.DataFrame,
        spec: ViewSpec,
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        has_ticker = not df.empty and "fund_ticker" in df.columns
        for name in self.fund_registry:
            fund_df = df[df["fund_ticker"] == name].copy() if has_ticker else pd.DataFrame()
            current, previous = self._split_by_date(fund_df, target_date, previous_date)
            self._stash(store, name, spec.key, current)
            if spec.key_t1 and previous_date is not None:
                self._stash(store, name, spec.key_t1, previous)

    @staticmethod
    def _split_by_date(
        df: pd.DataFrame, target_date: date, previous_date: Optional[date]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty or "date" not in df.columns:
            return df.copy(), pd.DataFrame()
        normalized = df.copy()
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.date
        current = normalized[normalized["date"] == target_date].copy()
        if previous_date is None:
            return current, pd.DataFrame()
        previous = normalized[normalized["date"] == previous_date].copy()
        return current, previous

    @staticmethod
    def _stash(store: BulkDataStore, fund_name: str, key: str, df: pd.DataFrame) -> None:
        store.fund_data.setdefault(fund_name, {})[key] = df
        store.loaded_funds.add(fund_name)
        store.loaded_data_types.add(key)

    # ------------------------------------------------------------------
    # Trades: split combined v_emsx_trades into equity/option/treasury slices
    # so existing services don't need to change their access keys.
    # ------------------------------------------------------------------
    def _split_trades_into_asset_classes(
        self, store: BulkDataStore, previous_date: Optional[date]
    ) -> None:
        slices = (
            ("trades",    "equity_trades",   "option_trades",   "treasury_trades"),
            ("trades_t1", "equity_trades_t1","option_trades_t1","treasury_trades_t1"),
        )
        for source_key, eq_key, opt_key, tsy_key in slices:
            if source_key == "trades_t1" and previous_date is None:
                continue
            for fund in self.fund_registry.values():
                trades = store.fund_data.get(fund.name, {}).get(source_key)
                if trades is None or trades.empty:
                    self._stash(store, fund.name, eq_key, pd.DataFrame())
                    self._stash(store, fund.name, opt_key, pd.DataFrame())
                    self._stash(store, fund.name, tsy_key, pd.DataFrame())
                    continue
                cls = trades["asset_class"].fillna("")
                self._stash(store, fund.name, eq_key,  trades[cls == "Equity"].copy())
                self._stash(store, fund.name, opt_key, self._with_optticker(trades[cls == "Option"]))
                self._stash(store, fund.name, tsy_key, trades[cls.str.contains("Treasury", case=False, na=False)].copy())

    @staticmethod
    def _with_optticker(options: pd.DataFrame) -> pd.DataFrame:
        if options.empty:
            return options
        out = options.copy()
        out.rename(columns={"ticker": "optticker"}, inplace=True)
        return out

    # ------------------------------------------------------------------
    # Index data — kept separate because of per-source T vs T+1 routing
    # ------------------------------------------------------------------
    def _load_index_data(
        self,
        store: BulkDataStore,
        target_date: date,
        previous_date: Optional[date],
    ) -> None:
        index_to_funds: Dict[str, List[Fund]] = {}
        for fund in self.fund_registry.values():
            if fund.index_ticker_join:
                index_to_funds.setdefault(fund.index_ticker_join, []).append(fund)
        if not index_to_funds:
            return

        tplus_one = self._bday_shift(target_date, 1)
        previous_tplus_one = self._bday_shift(previous_date, 1)
        dates = sorted({
            d for d in (target_date, previous_date, tplus_one, previous_tplus_one)
            if d is not None
        })
        df = self._query_sql(
            f"SELECT * FROM {VIEW_SCHEMA}.v_index_holdings "
            "WHERE `date` IN :dates AND fund IN :codes",
            dates=dates, codes=list(index_to_funds.keys()),
        )

        for index_code, funds in index_to_funds.items():
            code_df = df[df["fund"] == index_code] if not df.empty else df
            if code_df.empty:
                for fund in funds:
                    self._stash(store, fund.name, "index", pd.DataFrame())
                    if previous_date is not None:
                        self._stash(store, fund.name, "index_t1", pd.DataFrame())
                continue

            code_df = code_df.copy()
            code_df["date"] = pd.to_datetime(code_df["date"], errors="coerce").dt.date
            source = str(code_df["source"].iloc[0]).lower()
            kind = _INDEX_SOURCE_DATES.get(source, "target")
            if kind == "tplus_one":
                cur_date, prev_date = tplus_one, (previous_tplus_one or previous_date)
            else:
                cur_date, prev_date = target_date, previous_date

            current = code_df[code_df["date"] == cur_date].copy() if cur_date else pd.DataFrame()
            previous = code_df[code_df["date"] == prev_date].copy() if prev_date else pd.DataFrame()
            for fund in funds:
                self._stash(store, fund.name, "index", current.copy())
                if previous_date is not None:
                    self._stash(store, fund.name, "index_t1", previous.copy())

    @staticmethod
    def _bday_shift(anchor: Optional[date], offset: int) -> Optional[date]:
        if anchor is None or offset == 0:
            return anchor
        return (pd.Timestamp(anchor) + BDay(offset)).date()