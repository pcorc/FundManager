import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from processing.fund import Fund

OPTION_QTY_COLS = ['quantity', 'contracts', 'position', 'shares', 'qty']
TREASURY_QTY_COLS = ['quantity', 'par', 'par_value', 'face', 'position', 'amount']
DISCREPANCY_LABELS = ["Missing in OMS", "Missing in Custodian", "Quantity Mismatch"]
YES_NO = {True: "Yes", False: "No"}

SUMMARY_KEYS = {
    "index_equity": ["holdings_discrepancies", "price_discrepancies_T", "price_discrepancies_T1"],
    "custodian_equity": ["final_recon", "price_discrepancies_T", "price_discrepancies_T1"],
    "custodian_equity_t1": ["final_recon"],
    "custodian_option": ["final_recon", "regular_options", "flex_options",
                         "price_discrepancies_T", "price_discrepancies_T1"],
    "custodian_option_t1": ["final_recon", "regular_options", "flex_options"],
    "custodian_flex_option": ["final_recon", "price_discrepancies_T", "price_discrepancies_T1"],
}


@dataclass
class ReconciliationResult:
    """Structured result for reconciliation checks"""
    raw_recon: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_recon: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_discrepancies_T: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_discrepancies_T1: pd.DataFrame = field(default_factory=pd.DataFrame)
    merged_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    regular_options: Optional[pd.DataFrame] = None
    flex_options: Optional[pd.DataFrame] = None


def _outer_merge(left: pd.DataFrame, right: pd.DataFrame, key: str) -> pd.DataFrame:
    """Outer-merge on ``key``, or an empty frame when either side lacks it."""
    if key in left.columns and key in right.columns:
        return pd.merge(left, right, on=key, how='outer', suffixes=('_vest', '_cust'))
    return pd.DataFrame()


def _classify(df: pd.DataFrame, qty_mask, default: str):
    return np.select(
        [~df['in_vest'] & df['in_cust'], df['in_vest'] & ~df['in_cust'], qty_mask],
        DISCREPANCY_LABELS,
        default=default,
    )


def _breakdown(df: pd.DataFrame, qty_text: pd.Series):
    return np.where(
        df['discrepancy_type'] == "Quantity Mismatch",
        qty_text,
        "Present in Vest: " + df['in_vest'].map(YES_NO)
        + " | Present in Cust: " + df['in_cust'].map(YES_NO)
    )


class Reconciliator:
    """Reconciliation engine leveraging Fund object properties."""

    def __init__(self, fund: Fund, analysis_type: str = "eod"):
        self.fund = fund
        self.fund_name = fund.name
        self.analysis_type = analysis_type
        self.results = {}
        self.logger = logging.getLogger(f"Reconciliator_{fund.name}")

    def run_all_reconciliations(self):
        """Run all reconciliations using Fund object data"""
        recon_funcs = [
            ("custodian_equity", self.reconcile_custodian_equity),
            ("custodian_option", self.reconcile_custodian_option),
            ("custodian_flex_option", self.reconcile_custodian_flex_option),
            ("custodian_treasury", self.reconcile_custodian_treasury),
            ("index_equity", self.reconcile_index_equity),
        ]

        for name, func in recon_funcs:
            try:
                func()
            except Exception as e:
                self.logger.error(f"Error in {name} reconciliation: {e}", exc_info=True)

    def _frames(self, attr: str):
        """Copies of (vest T, custodian T, vest T-1, custodian T-1) holdings for ``attr``."""
        current, previous = self.fund.data.current, self.fund.data.previous
        return (getattr(current.vest, attr).copy(), getattr(current.custodian, attr).copy(),
                getattr(previous.vest, attr).copy(), getattr(previous.custodian, attr).copy())

    def reconcile_custodian_equity(self):
        """Reconcile custodian equity using Fund snapshots."""
        df_oms, df_cust, df_oms1, df_cust1 = self._frames('equity')
        current = self.fund.data.current
        df_trades = current.equity_trades.copy()
        df_crrd = current.cr_rd_data.copy()

        # Must have eqyticker
        if 'eqyticker' not in df_oms.columns or 'eqyticker' not in df_cust.columns:
            self.results['custodian_equity'] = ReconciliationResult()
            return

        # Merge base tables (today)
        df = pd.merge(df_oms, df_cust,
                      on='eqyticker', how='outer',
                      suffixes=('_vest', '_cust'), indicator=True)
        df['in_vest'] = df['_merge'] != 'right_only'
        df['in_cust'] = df['_merge'] != 'left_only'
        df.drop(columns=['_merge'], inplace=True)

        # The report's "quantity" column is the vest holding. The vest side has
        # nav_shares (no `quantity` column), so the merge leaves the custodian's
        # `quantity` un-suffixed — make `quantity` explicitly the vest shares.
        df['quantity'] = self._coerce_numeric_series(df.get('nav_shares', pd.Series(dtype=float)))

        # Same for T-1, but only if both frames carry the join key. KNGIX (and any
        # fund with no T-1 holdings) returns empty DataFrames with no columns at all.
        df1 = _outer_merge(df_oms1, df_cust1, 'eqyticker')

        # Trades & corporate actions (only today)
        trade_map = df_trades.set_index('eqyticker')['qty_sign_adj'] \
            if 'eqyticker' in df_trades.columns else pd.Series(dtype=object)
        df['qty_sign_adj'] = df['eqyticker'].map(trade_map).fillna(0)

        # Use fund property to determine if CR/RD applies
        cr_map = (df_crrd.set_index('eqyticker')['cr_rd']
                  if self.fund.is_etf and 'eqyticker' in df_crrd.columns
                  else pd.Series(dtype=object))
        df['cr_rd'] = df['eqyticker'].map(cr_map).fillna(0)

        # Adjusted shares & base discrepancy
        df['adjusted_cust_shares'] = df['shares_cust'].fillna(0) + df['qty_sign_adj']
        df['final_adjusted_shares'] = df['adjusted_cust_shares'] + df['cr_rd']
        df['final_discrepancy'] = df['nav_shares'].fillna(0) - df['final_adjusted_shares']

        mask_missing = df['in_vest'] != df['in_cust']
        mask_qty = df['in_vest'] & df['in_cust'] & df['final_discrepancy'].abs().gt(0.01)
        df['discrepancy_type'] = _classify(df, mask_qty, "")

        df_issues = df.loc[mask_missing | mask_qty].copy()
        df_issues['breakdown'] = _breakdown(
            df_issues,
            "Vest=" + df_issues['nav_shares'].fillna(0).astype(int).astype(str)
            + " | Cust=" + df_issues['shares_cust'].fillna(0).astype(int).astype(str)
            + " | TradesAdj=" + df_issues['qty_sign_adj'].astype(int).astype(str)
            + " | CR/RD=" + df_issues['cr_rd'].astype(int).astype(str),
        )

        # Quantity Mismatch rows first, then the missing-in rows
        if not df_issues.empty:
            qty_rows = df_issues['discrepancy_type'] == "Quantity Mismatch"
            df_issues = pd.concat([df_issues[qty_rows], df_issues[~qty_rows]], ignore_index=True)

        self.results['custodian_equity'] = ReconciliationResult(
            raw_recon=df.loc[mask_qty].reset_index(drop=True),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=self._price_breaks(df, 'eqyticker'),
            price_discrepancies_T1=self._price_breaks(df1, 'eqyticker'),
            merged_data=df.copy()
        )

    def reconcile_custodian_option(self):
        """Reconcile custodian options (regular options only) using Fund snapshots."""
        self._reconcile_option_book('custodian_option', 'options', flex=False)

    def reconcile_custodian_flex_option(self):
        """Reconcile custodian FLEX options separately using Fund snapshots."""
        self._reconcile_option_book('custodian_flex_option', 'flex_options', flex=True)

    def _reconcile_option_book(self, name: str, attr: str, flex: bool):
        if flex and not self.fund.uses_index_flex:
            self.results[name] = ReconciliationResult()
            return

        df_oms, df_cust, df_oms1, df_cust1 = self._frames(attr)

        if (df_oms.empty and df_cust.empty
                or not df_oms.empty and 'optticker' not in df_oms.columns
                or not df_cust.empty and 'optticker' not in df_cust.columns):
            self.results[name] = ReconciliationResult()
            return

        # Vest gets `quantity` before the merge, so a custodian `quantity` column collides
        for oms, cust in ((df_oms, df_cust), (df_oms1, df_cust1)):
            if not oms.empty:
                oms['quantity'] = oms['nav_shares']
                if 'price' in oms.columns:
                    oms['price'] = self._coerce_numeric_series(oms['price']).abs()
            if not cust.empty:
                self._resolve_shares_cust(cust, OPTION_QTY_COLS)

        if not flex and not df_oms.empty and 'price' in df_oms.columns and 'quantity' in df_oms.columns:
            df_oms['market_value'] = df_oms['quantity'].fillna(0) * df_oms['price'].fillna(0) * 100
            total_mv = df_oms['market_value'].sum()
            df_oms['option_weight'] = 0.0
            if total_mv > 0:
                df_oms['option_weight'] = df_oms['market_value'] / total_mv

        if flex:
            # FLEX opttickers don't match between sides — OMS emits
            # "4SPX US 07/31/26 C5000.01" while custodian emits "SPX US 07/31/26 C5000".
            # Both carry the flex contract code instead:
            #   OMS:       equity_underlying_ticker (e.g. SC31761B)
            #   custodian: cusip                    (e.g. SC31761B)
            for frame in (df_oms, df_cust, df_oms1, df_cust1):
                if frame.empty:
                    continue
                if 'equity_underlying_ticker' in frame.columns:
                    frame['flex_code'] = frame['equity_underlying_ticker']
                elif 'cusip' in frame.columns:
                    frame['flex_code'] = frame['cusip']

        key = 'flex_code' if flex else 'optticker'
        df = _outer_merge(df_oms, df_cust, key)
        df1 = _outer_merge(df_oms1, df_cust1, key)

        if flex:
            # Single display optticker: prefer OMS canonical form, fall back to custodian.
            for frame in (df, df1):
                if frame.empty or 'optticker' in frame.columns:
                    continue
                if 'optticker_vest' in frame.columns:
                    frame['optticker'] = frame['optticker_vest'].fillna(frame.get('optticker_cust'))
                elif 'optticker_cust' in frame.columns:
                    frame['optticker'] = frame['optticker_cust']

        # Identify discrepancies in current data
        if 'quantity' not in df.columns and 'quantity_vest' in df.columns:
            df['quantity'] = df['quantity_vest']
        if 'shares_cust' not in df.columns and 'shares_cust_cust' in df.columns:
            df['shares_cust'] = df['shares_cust_cust']
        elif 'shares_cust' not in df.columns:
            df['shares_cust'] = 0.0

        df['quantity'] = self._coerce_numeric_series(df.get('quantity', pd.Series(dtype=float)))
        df['shares_cust'] = self._coerce_numeric_series(df.get('shares_cust', pd.Series(dtype=float)))

        df['in_vest'] = df['quantity'].notnull() & (df['quantity'].abs() > 0.01)
        df['in_cust'] = df['shares_cust'].notnull() & (df['shares_cust'].abs() > 0.01)
        df['trade_discrepancy'] = df['quantity'].fillna(0) - df['shares_cust'].fillna(0)

        hold_disc_mask = df['in_vest'] != df['in_cust']
        qty_disc_mask = df['in_vest'] & df['in_cust'] & (df['trade_discrepancy'].abs() > 0.01)
        df_issues = df[hold_disc_mask | qty_disc_mask].copy()

        # FLEX shows ALL price breaks; listed options get FLEX tagging, weights and limits
        price_fn = self._flex_price_breaks if flex else self._option_price_breaks
        price_T = price_fn(df)
        price_T1 = price_fn(df1)

        if not df_issues.empty:
            df_issues['discrepancy_type'] = _classify(
                df_issues, df_issues['in_vest'] & df_issues['in_cust'], 'Quantity Mismatch'
            )
            df_issues['breakdown'] = _breakdown(
                df_issues,
                "Vest=" + df_issues['quantity'].round(0).astype(int).astype(str)
                + " | Cust=" + df_issues['shares_cust'].round(0).astype(int).astype(str),
            )
            if flex:
                df_issues['is_flex'] = True

        # Listed options carry regular/flex sections; FLEX is reported on its own key
        extra = {} if flex else {
            'regular_options': df_issues.reset_index(drop=True),
            'flex_options': pd.DataFrame(),
        }
        self.results[name] = ReconciliationResult(
            raw_recon=df[hold_disc_mask | qty_disc_mask].reset_index(drop=True) if not df.empty else pd.DataFrame(),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=price_T.reset_index(drop=True),
            price_discrepancies_T1=price_T1.reset_index(drop=True),
            merged_data=df.copy(),
            **extra
        )

    def reconcile_custodian_treasury(self):
        """Reconcile custodian treasury holdings using Fund snapshots."""
        current = self.fund.data.current

        df_oms = current.vest.treasury.copy()
        df_cust = current.custodian.treasury.copy()

        if df_oms.empty and df_cust.empty:
            self.results['custodian_treasury'] = ReconciliationResult()
            return

        # Check for CUSIP column
        for frame, side in ((df_oms, 'OMS'), (df_cust, 'custodian')):
            if 'cusip' not in frame.columns:
                if frame.empty:
                    frame['cusip'] = pd.Series(dtype=object)
                else:
                    self.logger.warning(f"Missing CUSIP column for {side} treasury holdings")
                    self.results['custodian_treasury'] = ReconciliationResult()
                    return

        df_oms['quantity'] = df_oms['nav_shares']
        self._resolve_shares_cust(df_cust, TREASURY_QTY_COLS)

        df = pd.merge(
            df_oms,
            df_cust,
            on='cusip',
            how='outer',
            suffixes=('_vest', '_cust'),
            indicator=True
        )

        quantity_series = df['nav_shares'] if 'nav_shares' in df else pd.Series(0.0, index=df.index)
        shares_series = df['shares_cust'] if 'shares_cust' in df else pd.Series(0.0, index=df.index)
        df['nav_shares'] = self._coerce_numeric_series(quantity_series)
        df['shares_cust'] = self._coerce_numeric_series(shares_series)
        df['in_vest'] = df['_merge'] != 'right_only'
        df['in_cust'] = df['_merge'] != 'left_only'
        df['quantity_diff'] = df['nav_shares'] - df['shares_cust']

        mask_missing = df['in_vest'] != df['in_cust']
        mask_qty = df['in_vest'] & df['in_cust'] & df['quantity_diff'].abs().gt(0)
        df_issues = df.loc[mask_missing | mask_qty].copy()

        if not df_issues.empty:
            df_issues['discrepancy_type'] = _classify(df_issues, mask_qty.loc[df_issues.index], 'Unknown')
            df_issues['breakdown'] = _breakdown(
                df_issues,
                "Vest=" + df_issues['nav_shares'].round(2).astype(str)
                + " | Cust=" + df_issues['shares_cust'].round(2).astype(str),
            )
        else:
            df_issues = pd.DataFrame(columns=[
                'cusip', 'nav_shares', 'shares_cust', 'in_vest', 'in_cust',
                'quantity_diff', 'discrepancy_type', 'breakdown'
            ])

        price_disc_T = self._price_breaks(df, 'cusip')

        self.results['custodian_treasury'] = ReconciliationResult(
            raw_recon=df.loc[mask_qty].reset_index(drop=True),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=price_disc_T,
            price_discrepancies_T1=pd.DataFrame(),
            merged_data=df.copy()
        )

    def reconcile_index_equity(self):
        """Reconcile index equity using Fund snapshots."""
        current = self.fund.data.current
        if self.fund_name == "DOGG" or current.vest.equity.empty or current.index.equity.empty:
            self.results['index_equity'] = {
                'holdings_discrepancies': pd.DataFrame(),
                'significant_diffs': pd.DataFrame()
            }
            return

        df_oms = current.vest.equity.copy()
        df_index = current.index.equity.copy()

        # Merge with outer join to capture all securities
        df = pd.merge(df_oms, df_index, on='eqyticker', how='outer',
                      suffixes=('_vest', '_index'), indicator=True)

        # Add in_vest and in_index flags
        df['in_vest'] = df['nav_wgt_begin'].notnull()
        df['in_index'] = df['weight_index'].notnull()

        # Calculate weight difference for all securities
        df['wgt_diff'] = (df['nav_wgt_begin'].fillna(0) - df['weight_index'].fillna(0)).abs()

        # Holdings discrepancies - now includes weight information
        holdings_disc = df[df['in_vest'] != df['in_index']][
            ['eqyticker', 'in_vest', 'in_index', 'nav_wgt_begin', 'weight_index', 'wgt_diff']
        ].copy()

        # Fill NaN weights with 0 for cleaner display
        holdings_disc['nav_wgt_begin'] = holdings_disc['nav_wgt_begin'].fillna(0)
        holdings_disc['weight_index'] = holdings_disc['weight_index'].fillna(0)

        # Significant weight differences (for securities in both)
        sig = df[df['wgt_diff'].gt(0.001) & df['in_vest'] & df['in_index']][
            ['eqyticker', 'nav_wgt_begin', 'weight_index', 'wgt_diff']
        ].copy()

        self.results['index_equity'] = {
            'holdings_discrepancies': holdings_disc,
            'significant_diffs': sig
        }

    def get_summary(self) -> Dict:
        """Returns simplified counts without resolution tracking"""
        summary = {}
        for recon_type, keys in SUMMARY_KEYS.items():
            result = self.results.get(recon_type)
            summary[recon_type] = {k: self._count(getattr(result, k, None)) for k in keys}
        return summary

    @staticmethod
    def _count(df) -> int:
        return len(df) if hasattr(df, "shape") else 0

    @staticmethod
    def _coerce_numeric_series(series: pd.Series) -> pd.Series:
        """Return a numeric version of ``series`` with NaNs replaced by zero."""
        return pd.to_numeric(series, errors="coerce").fillna(0.0)

    @classmethod
    def _resolve_shares_cust(cls, df: pd.DataFrame, candidates) -> None:
        """Fill ``shares_cust`` from the first candidate column present (else 0), then coerce."""
        if 'shares_cust' not in df.columns:
            df['shares_cust'] = next((df[c] for c in candidates if c in df.columns), 0.0)
        df['shares_cust'] = cls._coerce_numeric_series(df['shares_cust'])

    @staticmethod
    def _price_breaks(df: pd.DataFrame, ticker_col: str, pct: bool = False) -> pd.DataFrame:
        """Rows whose vest/custodian price differ by more than 0.005."""
        if not {'price_vest', 'price_cust'}.issubset(df.columns):
            return pd.DataFrame()
        df_price = df.copy()
        df_price['price_diff'] = (df_price['price_vest'] - df_price['price_cust']).abs()
        cols = [ticker_col, 'price_vest', 'price_cust', 'price_diff']
        if pct:
            df_price['price_pct_diff'] = (
                    (df_price['price_vest'] - df_price['price_cust']) / df_price['price_cust'] * 100
            ).replace([np.inf, -np.inf], 0)
            cols.append('price_pct_diff')
        return df_price.loc[df_price['price_diff'] > 0.005, cols].reset_index(drop=True)

    def _option_price_breaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Option price breaks with FLEX indicator, weights and private/CEF top-5 limit."""
        breaks = self._price_breaks(df, 'optticker', pct=True)
        if breaks.empty:
            return breaks

        if self.fund.uses_index_flex:
            breaks['is_flex'] = breaks['optticker'].str.contains("SPX|XSP", na=False)
        else:
            breaks['is_flex'] = False

        if 'option_weight' in df.columns:
            breaks = breaks.merge(df[['optticker', 'option_weight']], on='optticker', how='left')
            breaks['option_weight'] = breaks['option_weight'].fillna(0)
        else:
            breaks['option_weight'] = 0.0

        # Filter standard options for private/closed-end funds
        if self.fund.vehicle in {"private_fund", "closed_end_fund"}:
            standard_price = breaks[~breaks['is_flex']].copy()
            flex_price = breaks[breaks['is_flex']].copy()
            if not standard_price.empty:
                standard_price = standard_price.nlargest(5, 'option_weight')
            breaks = pd.concat([standard_price, flex_price], ignore_index=True)

        return breaks.reset_index(drop=True)

    def _flex_price_breaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """FLEX option price breaks - ALL breaks shown (no limiting)."""
        breaks = self._price_breaks(df, 'optticker', pct=True)
        if not breaks.empty:
            breaks['is_flex'] = True
        return breaks
