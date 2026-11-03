import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Optional
from processing.fund import Fund


@dataclass
class ReconciliationResult:
    """Structured result for reconciliation checks"""
    raw_recon: pd.DataFrame
    final_recon: pd.DataFrame
    price_discrepancies_T: pd.DataFrame
    price_discrepancies_T1: pd.DataFrame
    merged_data: pd.DataFrame
    regular_options: Optional[pd.DataFrame] = None
    flex_options: Optional[pd.DataFrame] = None


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

    def reconcile_custodian_equity(self):
        """Reconcile custodian equity using Fund snapshots."""
        # Access current and previous snapshots
        current = self.fund.data.current
        previous = self.fund.data.previous

        # Get holdings from vest and custodian
        df_oms = current.vest.equity.copy()
        df_cust = current.custodian.equity.copy()
        df_oms1 = previous.vest.equity.copy()
        df_cust1 = previous.custodian.equity.copy()

        # Get trades and corporate actions from current snapshot
        df_trades = current.equity_trades.copy()
        df_crrd = current.cr_rd_data.copy()

        # Must have eqyticker
        if 'eqyticker' not in df_oms.columns or 'eqyticker' not in df_cust.columns:
            self.results['custodian_equity'] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame()
            )
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
        # fund with no T-1 holdings) returns empty DataFrames with no columns at
        # all, so the merge would raise KeyError on 'eqyticker'.
        if 'eqyticker' in df_oms1.columns and 'eqyticker' in df_cust1.columns:
            df1 = pd.merge(df_oms1, df_cust1,
                           on='eqyticker', how='outer',
                           suffixes=('_vest', '_cust'), indicator=False)
        else:
            df1 = pd.DataFrame()

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

        # Identify mismatches AND create discrepancy_type column FIRST
        mask_missing = df['in_vest'] != df['in_cust']
        mask_qty = df['in_vest'] & df['in_cust'] & df['final_discrepancy'].abs().gt(0.01)

        # Create discrepancy_type for ALL rows first
        conditions = [
            ~df['in_vest'] & df['in_cust'],
            df['in_vest'] & ~df['in_cust'],
            mask_qty
        ]
        choices = ["Missing in OMS", "Missing in Custodian", "Quantity Mismatch"]
        df['discrepancy_type'] = np.select(conditions, choices, default="")

        # Now filter for issues
        df_issues = df.loc[mask_missing | mask_qty].copy()

        # Breakdown text
        df_issues['breakdown'] = np.where(
            df_issues['discrepancy_type'] == "Quantity Mismatch",
            "Vest=" + df_issues['nav_shares'].fillna(0).astype(int).astype(str)
            + " | Cust=" + df_issues['shares_cust'].fillna(0).astype(int).astype(str)
            + " | TradesAdj=" + df_issues['qty_sign_adj'].astype(int).astype(str)
            + " | CR/RD=" + df_issues['cr_rd'].astype(int).astype(str),
            "Present in Vest: " + df_issues['in_vest'].map({True: "Yes", False: "No"})
            + " | Present in Cust: " + df_issues['in_cust'].map({True: "Yes", False: "No"})
        )

        # Only keep rows with actual significant discrepancies
        if not df_issues.empty:
            qty_mask = df_issues['discrepancy_type'] == "Quantity Mismatch"
            sig_qty = df_issues[qty_mask & (df_issues['final_discrepancy'].abs() > 0.01)]
            other_issues = df_issues[~qty_mask]
            df_issues = pd.concat([sig_qty, other_issues], ignore_index=True)

        # Price-difference tables for T and T-1
        price_disc_T = self._calculate_price_discrepancies(df, 'eqyticker')
        price_disc_T1 = self._calculate_price_discrepancies(df1, 'eqyticker')

        # Override small (<1) today's vest price so final_discrepancy reflects it
        if {'price_vest', 'price_cust'}.issubset(df.columns):
            df['price_diff'] = (df['price_vest'] - df['price_cust']).abs()
            small_T = df['price_diff'].between(0.01, 1)
            df.loc[small_T, 'price_vest'] = df.loc[small_T, 'price_cust']
            df['final_discrepancy'] = df['nav_shares'].fillna(0) - df['final_adjusted_shares']

        # Store results
        self.results['custodian_equity'] = ReconciliationResult(
            raw_recon=df.loc[mask_qty & (df['final_discrepancy'].abs() > 0.01)].reset_index(drop=True),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=price_disc_T.reset_index(drop=True),
            price_discrepancies_T1=price_disc_T1.reset_index(drop=True),
            merged_data=df.copy()
        )

    def reconcile_custodian_option(self):
        """Reconcile custodian options (regular options only) using Fund snapshots."""
        empty_result = ReconciliationResult(
            raw_recon=pd.DataFrame(),
            final_recon=pd.DataFrame(),
            price_discrepancies_T=pd.DataFrame(),
            price_discrepancies_T1=pd.DataFrame(),
            merged_data=pd.DataFrame()
        )

        # Access current and previous snapshots - REGULAR OPTIONS ONLY
        current = self.fund.data.current
        previous = self.fund.data.previous

        df_oms = current.vest.options.copy()
        df_cust = current.custodian.options.copy()
        df_oms1 = previous.vest.options.copy()
        df_cust1 = previous.custodian.options.copy()

        # Check if all dataframes are empty
        if df_oms.empty and df_cust.empty:
            self.results['custodian_option'] = empty_result
            return

        # Verify optticker column exists
        if not df_oms.empty and 'optticker' not in df_oms.columns:
            self.results['custodian_option'] = empty_result
            return

        if not df_cust.empty and 'optticker' not in df_cust.columns:
            self.results['custodian_option'] = empty_result
            return

        # Set up quantity columns for current data
        if not df_oms.empty:
            df_oms['quantity'] = df_oms['nav_shares']
            if 'price' in df_oms.columns:
                df_oms['price'] = self._coerce_numeric_series(df_oms['price']).abs()

        if not df_cust.empty:
            if 'shares_cust' not in df_cust.columns:
                for candidate in ['quantity', 'contracts', 'position', 'shares', 'qty']:
                    if candidate in df_cust.columns:
                        df_cust['shares_cust'] = df_cust[candidate]
                        break
                else:
                    df_cust['shares_cust'] = 0.0
            df_cust['shares_cust'] = self._coerce_numeric_series(df_cust['shares_cust'])

        # Set up quantity columns for previous data
        if not df_oms1.empty:
            df_oms1['quantity'] = df_oms1['nav_shares']
            if 'price' in df_oms1.columns:
                df_oms1['price'] = self._coerce_numeric_series(df_oms1['price']).abs()

        if not df_cust1.empty:
            if 'shares_cust' not in df_cust1.columns:
                for candidate in ['quantity', 'contracts', 'position', 'shares', 'qty']:
                    if candidate in df_cust1.columns:
                        df_cust1['shares_cust'] = df_cust1[candidate]
                        break
                else:
                    df_cust1['shares_cust'] = 0.0
            df_cust1['shares_cust'] = self._coerce_numeric_series(df_cust1['shares_cust'])

        # Calculate option weights for regular options
        if not df_oms.empty and 'price' in df_oms.columns and 'quantity' in df_oms.columns:
            df_oms['market_value'] = df_oms['quantity'].fillna(0) * df_oms['price'].fillna(0) * 100
            total_mv = df_oms['market_value'].sum()
            df_oms['option_weight'] = 0.0
            if total_mv > 0:
                df_oms['option_weight'] = df_oms['market_value'] / total_mv

        # Merge current data — guard against empty frames missing the join key.
        if 'optticker' in df_oms.columns and 'optticker' in df_cust.columns:
            df = pd.merge(df_oms, df_cust, on='optticker', how='outer', suffixes=('_vest', '_cust'))
        else:
            df = pd.DataFrame()

        # Same for previous data.
        if 'optticker' in df_oms1.columns and 'optticker' in df_cust1.columns:
            df1 = pd.merge(df_oms1, df_cust1, on='optticker', how='outer', suffixes=('_vest', '_cust'))
        else:
            df1 = pd.DataFrame()

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

        # Filter for actual discrepancies
        hold_disc_mask = df['in_vest'] != df['in_cust']
        qty_disc_mask = df['in_vest'] & df['in_cust'] & (df['trade_discrepancy'].abs() > 0.01)
        df_issues = df[hold_disc_mask | qty_disc_mask].copy()

        # Calculate price discrepancies
        price_T = self._calculate_option_price_discrepancies(df, 'T')
        price_T1 = self._calculate_option_price_discrepancies(df1, 'T-1')

        # Add breakdown descriptions
        if not df_issues.empty:
            df_issues['discrepancy_type'] = np.where(
                ~df_issues['in_vest'] & df_issues['in_cust'],
                'Missing in OMS',
                np.where(
                    df_issues['in_vest'] & ~df_issues['in_cust'],
                    'Missing in Custodian',
                    'Quantity Mismatch'
                )
            )

            df_issues['breakdown'] = np.where(
                df_issues['discrepancy_type'] == 'Quantity Mismatch',
                "Vest=" + df_issues['quantity'].round(0).astype(int).astype(str) +
                " | Cust=" + df_issues['shares_cust'].round(0).astype(int).astype(str),
                "Present in Vest: " + df_issues['in_vest'].map({True: 'Yes', False: 'No'}) +
                " | Present in Cust: " + df_issues['in_cust'].map({True: 'Yes', False: 'No'})
            )

        # Store results - no FLEX separation since FLEX is handled separately
        self.results['custodian_option'] = ReconciliationResult(
            raw_recon=df[hold_disc_mask | qty_disc_mask].reset_index(drop=True) if not df.empty else pd.DataFrame(),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=price_T.reset_index(drop=True),
            price_discrepancies_T1=price_T1.reset_index(drop=True),
            merged_data=df.copy(),
            regular_options=df_issues.reset_index(drop=True),  # Same as final_recon for regular options
            flex_options=pd.DataFrame()  # Always empty - FLEX handled separately
        )

    def reconcile_custodian_flex_option(self):
        """Reconcile custodian FLEX options separately using Fund snapshots."""

        # Only run if fund uses FLEX options
        if not self.fund.uses_index_flex:
            self.results['custodian_flex_option'] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame()
            )
            return

        empty_result = ReconciliationResult(
            raw_recon=pd.DataFrame(),
            final_recon=pd.DataFrame(),
            price_discrepancies_T=pd.DataFrame(),
            price_discrepancies_T1=pd.DataFrame(),
            merged_data=pd.DataFrame()
        )

        # Access current and previous snapshots - FLEX options only
        current = self.fund.data.current
        previous = self.fund.data.previous

        df_oms = current.vest.flex_options.copy()
        df_cust = current.custodian.flex_options.copy()
        df_oms1 = previous.vest.flex_options.copy()
        df_cust1 = previous.custodian.flex_options.copy()

        # Check if all dataframes are empty
        if df_oms.empty and df_cust.empty:
            self.results['custodian_flex_option'] = empty_result
            return

        # Verify optticker column exists
        if not df_oms.empty and 'optticker' not in df_oms.columns:
            self.results['custodian_flex_option'] = empty_result
            return

        if not df_cust.empty and 'optticker' not in df_cust.columns:
            self.results['custodian_flex_option'] = empty_result
            return

        # Set up quantity columns for current data
        if not df_oms.empty:
            df_oms['quantity'] = df_oms['nav_shares']
            if 'price' in df_oms.columns:
                df_oms['price'] = self._coerce_numeric_series(df_oms['price']).abs()

        if not df_cust.empty:
            if 'shares_cust' not in df_cust.columns:
                for candidate in ['quantity', 'contracts', 'position', 'shares', 'qty']:
                    if candidate in df_cust.columns:
                        df_cust['shares_cust'] = df_cust[candidate]
                        break
                else:
                    df_cust['shares_cust'] = 0.0
            df_cust['shares_cust'] = self._coerce_numeric_series(df_cust['shares_cust'])

        # Set up quantity columns for previous data
        if not df_oms1.empty:
            df_oms1['quantity'] = df_oms1['nav_shares']
            if 'price' in df_oms1.columns:
                df_oms1['price'] = self._coerce_numeric_series(df_oms1['price']).abs()

        if not df_cust1.empty:
            if 'shares_cust' not in df_cust1.columns:
                for candidate in ['quantity', 'contracts', 'position', 'shares', 'qty']:
                    if candidate in df_cust1.columns:
                        df_cust1['shares_cust'] = df_cust1[candidate]
                        break
                else:
                    df_cust1['shares_cust'] = 0.0
            df_cust1['shares_cust'] = self._coerce_numeric_series(df_cust1['shares_cust'])

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

        if 'flex_code' in df_oms.columns and 'flex_code' in df_cust.columns:
            df = pd.merge(df_oms, df_cust, on='flex_code', how='outer', suffixes=('_vest', '_cust'))
        else:
            df = pd.DataFrame()

        if 'flex_code' in df_oms1.columns and 'flex_code' in df_cust1.columns:
            df1 = pd.merge(df_oms1, df_cust1, on='flex_code', how='outer', suffixes=('_vest', '_cust'))
        else:
            df1 = pd.DataFrame()

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

        # Filter for actual discrepancies
        hold_disc_mask = df['in_vest'] != df['in_cust']
        qty_disc_mask = df['in_vest'] & df['in_cust'] & (df['trade_discrepancy'].abs() > 0.01)
        df_issues = df[hold_disc_mask | qty_disc_mask].copy()

        # Calculate price discrepancies - ALL FLEX prices are shown
        price_T = self._calculate_flex_price_discrepancies(df, 'T')
        price_T1 = self._calculate_flex_price_discrepancies(df1, 'T-1')

        # Add breakdown descriptions
        if not df_issues.empty:
            df_issues['discrepancy_type'] = np.where(
                ~df_issues['in_vest'] & df_issues['in_cust'],
                'Missing in OMS',
                np.where(
                    df_issues['in_vest'] & ~df_issues['in_cust'],
                    'Missing in Custodian',
                    'Quantity Mismatch'
                )
            )

            df_issues['breakdown'] = np.where(
                df_issues['discrepancy_type'] == 'Quantity Mismatch',
                "Vest=" + df_issues['quantity'].round(0).astype(int).astype(str) +
                " | Cust=" + df_issues['shares_cust'].round(0).astype(int).astype(str),
                "Present in Vest: " + df_issues['in_vest'].map({True: 'Yes', False: 'No'}) +
                " | Present in Cust: " + df_issues['in_cust'].map({True: 'Yes', False: 'No'})
            )

        # Mark as FLEX
        if not df_issues.empty:
            df_issues['is_flex'] = True
        if not price_T.empty:
            price_T['is_flex'] = True
        if not price_T1.empty:
            price_T1['is_flex'] = True

        # Store results
        self.results['custodian_flex_option'] = ReconciliationResult(
            raw_recon=df[hold_disc_mask | qty_disc_mask].reset_index(drop=True) if not df.empty else pd.DataFrame(),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=price_T.reset_index(drop=True),
            price_discrepancies_T1=price_T1.reset_index(drop=True),
            merged_data=df.copy()
        )

    def reconcile_custodian_treasury(self):
        """Reconcile custodian treasury holdings using Fund snapshots."""
        current = self.fund.data.current

        df_oms = current.vest.treasury.copy()
        df_cust = current.custodian.treasury.copy()

        if df_oms.empty and df_cust.empty:
            self.results['custodian_treasury'] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame()
            )
            return

        # Check for CUSIP column
        if 'cusip' not in df_oms.columns:
            if df_oms.empty:
                df_oms['cusip'] = pd.Series(dtype=object)
            else:
                self.logger.warning("Missing CUSIP column for OMS treasury holdings")
                self.results['custodian_treasury'] = ReconciliationResult(
                    raw_recon=pd.DataFrame(),
                    final_recon=pd.DataFrame(),
                    price_discrepancies_T=pd.DataFrame(),
                    price_discrepancies_T1=pd.DataFrame(),
                    merged_data=pd.DataFrame()
                )
                return

        if 'cusip' not in df_cust.columns:
            if df_cust.empty:
                df_cust['cusip'] = pd.Series(dtype=object)
            else:
                self.logger.warning("Missing CUSIP column for custodian treasury holdings")
                self.results['custodian_treasury'] = ReconciliationResult(
                    raw_recon=pd.DataFrame(),
                    final_recon=pd.DataFrame(),
                    price_discrepancies_T=pd.DataFrame(),
                    price_discrepancies_T1=pd.DataFrame(),
                    merged_data=pd.DataFrame()
                )
                return

        df_oms['quantity'] = df_oms['nav_shares']
        if 'shares_cust' not in df_cust.columns:
            for candidate in ['quantity', 'par', 'par_value', 'face', 'position', 'amount']:
                if candidate in df_cust.columns:
                    df_cust['shares_cust'] = df_cust[candidate]
                    break
            else:
                df_cust['shares_cust'] = 0.0
        df_cust['shares_cust'] = self._coerce_numeric_series(df_cust['shares_cust'])

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
            df_issues['discrepancy_type'] = np.select(
                [
                    ~df_issues['in_vest'] & df_issues['in_cust'],
                    df_issues['in_vest'] & ~df_issues['in_cust'],
                    mask_qty.loc[df_issues.index],
                ],
                ['Missing in OMS', 'Missing in Custodian', 'Quantity Mismatch'],
                default='Unknown',
            )
            df_issues['breakdown'] = np.where(
                df_issues['discrepancy_type'] == 'Quantity Mismatch',
                "Vest=" + df_issues['nav_shares'].round(2).astype(str)
                + " | Cust=" + df_issues['shares_cust'].round(2).astype(str),
                "Present in Vest: " + df_issues['in_vest'].map({True: 'Yes', False: 'No'})
                + " | Present in Cust: " + df_issues['in_cust'].map({True: 'Yes', False: 'No'})
            )
        else:
            df_issues = pd.DataFrame(columns=[
                'cusip', 'nav_shares', 'shares_cust', 'in_vest', 'in_cust',
                'quantity_diff', 'discrepancy_type', 'breakdown'
            ])

        price_disc_T = self._calculate_price_discrepancies(df, 'cusip')

        self.results['custodian_treasury'] = ReconciliationResult(
            raw_recon=df.loc[mask_qty].reset_index(drop=True),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=price_disc_T,
            price_discrepancies_T1=pd.DataFrame(),
            merged_data=df.copy()
        )

    def reconcile_index_equity(self):
        """Reconcile index equity using Fund snapshots."""
        # Check for special fund first
        if self.fund_name == "DOGG":
            self.results['index_equity'] = {
                'holdings_discrepancies': pd.DataFrame(),
                'significant_diffs': pd.DataFrame()
            }
            return

        # Get data from fund snapshots
        current = self.fund.data.current
        df_oms = current.vest.equity.copy()
        df_index = current.index.equity.copy()

        # Check if data is available
        if df_oms.empty or df_index.empty:
            self.results['index_equity'] = {
                'holdings_discrepancies': pd.DataFrame(),
                'significant_diffs': pd.DataFrame()
            }
            return

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

        recon_keys = {
            "index_equity": ["holdings_discrepancies", "price_discrepancies_T", "price_discrepancies_T1"],
            "custodian_equity": ["final_recon", "price_discrepancies_T", "price_discrepancies_T1"],
            "custodian_equity_t1": ["final_recon"],
            "custodian_option": ["final_recon", "regular_options", "flex_options",
                                 "price_discrepancies_T", "price_discrepancies_T1"],
            "custodian_option_t1": ["final_recon", "regular_options", "flex_options"],
            "custodian_flex_option": ["final_recon", "price_discrepancies_T", "price_discrepancies_T1"],  # NEW
        }

        for recon_type, keys in recon_keys.items():
            result = self.results.get(recon_type)
            if result:
                counts = {}
                for k in keys:
                    df = getattr(result, k, pd.DataFrame())
                    counts[k] = len(df) if hasattr(df, "shape") else 0
                summary[recon_type] = counts
            else:
                summary[recon_type] = {k: 0 for k in keys}

        return summary

    @staticmethod
    def _coerce_numeric_series(series: pd.Series) -> pd.Series:
        """Return a numeric version of ``series`` with NaNs replaced by zero."""
        return pd.to_numeric(series, errors="coerce").fillna(0.0)

    def _calculate_price_discrepancies(self, df: pd.DataFrame, ticker_col: str) -> pd.DataFrame:
        """Calculate price discrepancies between vest and custodian"""
        if {'price_vest', 'price_cust'}.issubset(df.columns):
            df_price = df.copy()
            # FIX: Remove float() call - just subtract the Series directly
            df_price['price_diff'] = (df_price['price_vest'] - df_price['price_cust']).abs()
            price_discrepancies = df_price.loc[
                df_price['price_diff'] > 0.005,
                [ticker_col, 'price_vest', 'price_cust', 'price_diff']
            ].copy()

            # Apply small price overrides
            small_mask = df_price['price_diff'].lt(1)
            df_price.loc[small_mask, 'price_vest'] = df_price.loc[small_mask, 'price_cust']

            return price_discrepancies.reset_index(drop=True)
        return pd.DataFrame()

    def _calculate_option_price_discrepancies(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Calculate option price discrepancies with FLEX filtering"""
        if {'price_vest', 'price_cust'}.issubset(df.columns):
            df_price = df.copy()
            df_price['price_diff'] = (df_price['price_vest'] - df_price['price_cust']).abs()
            df_price['price_pct_diff'] = (
                    (df_price['price_vest'] - df_price['price_cust']) / df_price['price_cust'] * 100
            ).replace([np.inf, -np.inf], 0)

            price_discrepancies = df_price.loc[
                df_price['price_diff'] > 0.005,
                ['optticker', 'price_vest', 'price_cust', 'price_diff', 'price_pct_diff']
            ].copy()

            # Add FLEX indicator and weights
            if not price_discrepancies.empty:
                if self.fund.uses_index_flex and 'optticker' in price_discrepancies.columns:
                    price_discrepancies['is_flex'] = price_discrepancies['optticker'].str.contains(
                        "SPX|XSP", na=False
                    )
                else:
                    price_discrepancies['is_flex'] = False

                # Add option weights
                if 'option_weight' in df.columns:
                    price_discrepancies = price_discrepancies.merge(
                        df[['optticker', 'option_weight']],
                        on='optticker',
                        how='left'
                    )
                    price_discrepancies['option_weight'] = price_discrepancies['option_weight'].fillna(0)
                else:
                    price_discrepancies['option_weight'] = 0.0

                # Filter standard options for private/closed-end funds
                if self.fund.vehicle in {"private_fund", "closed_end_fund"}:
                    standard_price = price_discrepancies[~price_discrepancies['is_flex']].copy()
                    flex_price = price_discrepancies[price_discrepancies['is_flex']].copy()

                    if not standard_price.empty:
                        standard_price = standard_price.nlargest(5, 'option_weight')

                    price_discrepancies = pd.concat([standard_price, flex_price], ignore_index=True)

            return price_discrepancies.reset_index(drop=True)
        return pd.DataFrame()

    def _calculate_flex_price_discrepancies(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Calculate FLEX option price discrepancies - ALL breaks shown (no limiting)."""
        if {'price_vest', 'price_cust'}.issubset(df.columns):
            df_price = df.copy()
            df_price['price_diff'] = (df_price['price_vest'] - df_price['price_cust']).abs()
            df_price['price_pct_diff'] = (
                    (df_price['price_vest'] - df_price['price_cust']) / df_price['price_cust'] * 100
            ).replace([np.inf, -np.inf], 0)

            price_discrepancies = df_price.loc[
                df_price['price_diff'] > 0.005,
                ['optticker', 'price_vest', 'price_cust', 'price_diff', 'price_pct_diff']
            ].copy()

            # Add FLEX indicator (always True for this method)
            if not price_discrepancies.empty:
                price_discrepancies['is_flex'] = True

            return price_discrepancies.reset_index(drop=True)
        return pd.DataFrame()

