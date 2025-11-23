import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Optional
from processing.fund import Fund
from config.fund_definitions import (
    FUND_DEFINITIONS,
    CLOSED_END_FUNDS,
    PRIVATE_FUNDS,
    ETF_FUNDS,
    DIVERSIFIED_FUNDS,
    NON_DIVERSIFIED_FUNDS,
    INDEX_FLEX_FUNDS,
)


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
        self.summary_rows = []
        self._detailed_calculations: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Access fund properties directly
        self.is_closed_end_fund = fund.is_closed_end_fund
        self.is_private_fund = fund.is_private_fund
        self.is_etf = fund.is_etf
        self.is_diversified = fund.is_diversified
        self.is_non_diversified = fund.is_non_diversified
        self.uses_index_flex = fund.uses_index_flex
        self.vehicle = fund.vehicle

        # Check for SG equity data availability
        #self.has_sg_equity = self._check_sg_equity()

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

        # Build all detail structures
        self._build_equity_details()
        self._build_option_details()  # NEW
        self._build_flex_option_details()  # NEW
        self._build_treasury_details()  # NEW

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

        # Same for T-1
        df1 = pd.merge(df_oms1, df_cust1,
                       on='eqyticker', how='outer',
                       suffixes=('_vest', '_cust'), indicator=False)

        # Trades & corporate actions (only today)
        trade_map = df_trades.set_index('eqyticker')['qty_sign_adj'] \
            if 'eqyticker' in df_trades.columns else pd.Series(dtype=object)
        df['qty_sign_adj'] = df['eqyticker'].map(trade_map).fillna(0)

        # Use fund property to determine if CR/RD applies
        cr_map = (df_crrd.set_index('eqyticker')['cr_rd']
                  if self.is_etf and 'eqyticker' in df_crrd.columns
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

        # Emit summary rows
        self._emit_equity_summary_rows(df_issues, price_disc_T, price_disc_T1)

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

        # Merge current data
        df = pd.merge(df_oms, df_cust, on='optticker', how='outer', suffixes=('_vest', '_cust'))

        # Merge previous data
        df1 = pd.merge(df_oms1, df_cust1, on='optticker', how='outer', suffixes=('_vest', '_cust'))

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

        # Add summary rows
        for _, row in df_issues.iterrows():
            dtype = row.get('discrepancy_type', 'Unknown')
            desc = row.get('breakdown', '')
            value = row.get('trade_discrepancy', 'N/A') if dtype == 'Quantity Mismatch' else 'N/A'
            self.add_summary_row(
                f"Custodian Option: {dtype}",
                row.get('optticker', ''),
                desc,
                value
            )

        for _, row in price_T.iterrows():
            self.add_summary_row(
                "Custodian Option Price (T)",
                row['optticker'],
                f"{row['price_vest']:.2f} vs {row['price_cust']:.2f}",
                row['price_diff']
            )

        for _, row in price_T1.iterrows():
            self.add_summary_row(
                "Custodian Option Price (T-1)",
                row['optticker'],
                f"{row['price_vest']:.2f} vs {row['price_cust']:.2f}",
                row['price_diff']
            )

    def reconcile_custodian_flex_option(self):
        """Reconcile custodian FLEX options separately using Fund snapshots."""

        # Only run if fund uses FLEX options
        if not self.uses_index_flex:
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

        # Merge current data
        df = pd.merge(df_oms, df_cust, on='optticker', how='outer', suffixes=('_vest', '_cust'))

        # Merge previous data
        df1 = pd.merge(df_oms1, df_cust1, on='optticker', how='outer', suffixes=('_vest', '_cust'))

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

        # Add summary rows
        for _, row in df_issues.iterrows():
            dtype = row.get('discrepancy_type', 'Unknown')
            desc = row.get('breakdown', '')
            value = row.get('trade_discrepancy', 'N/A') if dtype == 'Quantity Mismatch' else 'N/A'
            self.add_summary_row(
                f"Custodian FLEX Option: {dtype}",
                row.get('optticker', ''),
                desc,
                value
            )

        for _, row in price_T.iterrows():
            self.add_summary_row(
                "Custodian FLEX Option Price (T)",
                row['optticker'],
                f"{row['price_vest']:.2f} vs {row['price_cust']:.2f}",
                row['price_diff']
            )

        for _, row in price_T1.iterrows():
            self.add_summary_row(
                "Custodian FLEX Option Price (T-1)",
                row['optticker'],
                f"{row['price_vest']:.2f} vs {row['price_cust']:.2f}",
                row['price_diff']
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

        for _, row in df_issues.iterrows():
            dtype = row.get('discrepancy_type', 'Unknown')
            desc = row.get('breakdown', '')
            value = row.get('quantity_diff', 'N/A') if dtype == 'Quantity Mismatch' else 'N/A'
            self.add_summary_row(f"Custodian Treasury: {dtype}", row.get('cusip', ''), desc, value)

        for _, row in price_disc_T.iterrows():
            self.add_summary_row(
                "Custodian Treasury Price (T)",
                row.get('cusip', ''),
                f"{row['price_vest']:.4f} vs {row['price_cust']:.4f}",
                row.get('price_diff', 0.0),
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

        # Add summary rows
        for _, row in holdings_disc.iterrows():
            note = 'Missing in OMS' if not row['in_vest'] else 'Missing in Index'
            weight_info = f"OMS: {row['nav_wgt_begin']:.4f}, Index: {row['weight_index']:.4f}"
            self.add_summary_row('Index Equity', row['eqyticker'],
                                 f"{note} - {weight_info}", row['wgt_diff'])

        for _, row in sig.iterrows():
            weight_info = f"OMS: {row['nav_wgt_begin']:.4f}, Index: {row['weight_index']:.4f}"
            self.add_summary_row('Index Weight Diff', row['eqyticker'],
                                 f"Weight diff - {weight_info}", row['wgt_diff'])

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

        if self.analysis_type != "eod":
            recon_keys["sg_option"] = ["final_recon", "price_discrepancies"]
            if self.has_sg_equity:
                recon_keys["sg_equity"] = ["final_recon", "price_discrepancies"]

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

    def get_detailed_calculations(self):
        """Returns detailed ticker-level calculations for Excel reporting.

        This aggregates all detailed calculations that were built by the
        dedicated builder functions (_build_equity_details, etc.).
        """
        detailed_data = {
            'summary': self.results,
            'equity_details': pd.DataFrame(),
            'option_details': pd.DataFrame(),
            'flex_details': pd.DataFrame(),
            'treasury_details': pd.DataFrame(),
            'price_adjustments': {
                'equity': pd.DataFrame(),
                'option': pd.DataFrame(),
                'flex_option': pd.DataFrame()
            }
        }

        # Get the dataframes from snapshots
        current = self.fund.data.current
        previous = self.fund.data.previous

        eq_today = current.vest.equity
        eq_prior = previous.vest.equity
        opt_today = current.vest.options
        opt_prior = previous.vest.options
        flex_today = current.vest.flex_options
        flex_prior = previous.vest.flex_options

        # Get expense ratio
        expense_rat = self.fund.expense_ratio

        # Process equity details
        if not eq_today.empty and not eq_prior.empty:
            df_eq = eq_today.merge(
                eq_prior[["eqyticker", "price", "quantity"]],
                on="eqyticker", how="inner", suffixes=("_t", "_t1")
            ).dropna(subset=["price_t", "price_t1"])

            qty_col = "quantity_t1" if self.analysis_type == "ex_ante" else "quantity_t"
            df_eq["gl"] = (df_eq["price_t"] - df_eq["price_t1"]) * df_eq[qty_col]
            df_eq = df_eq[df_eq["gl"].abs() > 0.01]

            if not df_eq.empty:
                df_eq_adj = self.apply_small_price_override(
                    df_eq.copy(), kind="equity", key_col="eqyticker"
                )

                price_breaks = self.holdings_price_breaks.get('equity', pd.DataFrame()) if hasattr(self, 'holdings_price_breaks') else pd.DataFrame()
                cust_price_map = {
                    ticker: self._lookup_cust_price(price_breaks, ticker) for ticker in df_eq['eqyticker']
                }
                price_t1_cust = df_eq['eqyticker'].map(cust_price_map).fillna(df_eq['price_t1'])

                equity_detail = pd.DataFrame({
                    'ticker': df_eq['eqyticker'],
                    'quantity_t1': df_eq['quantity_t1'],
                    'quantity_t': df_eq['quantity_t'],
                    'price_t1_raw': df_eq['price_t1'],
                    'price_t_raw': df_eq['price_t'],
                    'price_t1_cust': price_t1_cust,
                    'price_t1_adj': price_t1_cust,
                    'price_t_adj': df_eq_adj.get('price_t', df_eq['price_t']),
                    'quantity_used': df_eq[qty_col],
                    'gl': df_eq["gl"],
                    'gl_adjusted': df_eq_adj.get("gl_adj", df_eq["gl"])
                })
                equity_detail['gl_adjusted'] = (equity_detail['price_t_adj'] - equity_detail['price_t1_cust']) * equity_detail['quantity_used']
                detailed_data['equity_details'] = equity_detail

                if not price_breaks.empty:
                    detailed_data['price_adjustments']['equity'] = price_breaks

        # Process regular option details
        if not opt_today.empty and not opt_prior.empty:
            df_opt = opt_today.merge(
                opt_prior[["optticker", "price", "quantity"]],
                on="optticker", how="inner", suffixes=("_t", "_t1")
            ).dropna(subset=["price_t", "price_t1"])

            qty_col = "quantity_t1" if self.analysis_type == "ex_ante" else "quantity_t"
            df_opt["gl"] = (df_opt["price_t"] - df_opt["price_t1"]) * df_opt[qty_col] * 100
            df_opt = df_opt[df_opt["gl"].abs() > 0.01]

            if not df_opt.empty:
                df_opt_adj = self.apply_small_price_override(
                    df_opt.copy(), kind="option", key_col="optticker"
                )

                price_breaks = self.holdings_price_breaks.get('option', pd.DataFrame()) if hasattr(self, 'holdings_price_breaks') else pd.DataFrame()
                cust_price_map = {
                    ticker: self._lookup_cust_price(price_breaks, ticker) for ticker in df_opt['optticker']
                }
                price_t1_cust = df_opt['optticker'].map(cust_price_map).fillna(df_opt['price_t1'])

                option_detail = pd.DataFrame({
                    'ticker': df_opt['optticker'],
                    'quantity_t1': df_opt['quantity_t1'],
                    'quantity_t': df_opt['quantity_t'],
                    'price_t1_raw': df_opt['price_t1'],
                    'price_t_raw': df_opt['price_t'],
                    'price_t1_cust': price_t1_cust,
                    'price_t1_adj': price_t1_cust,
                    'price_t_adj': df_opt_adj.get('price_t', df_opt['price_t']),
                    'quantity_used': df_opt[qty_col],
                    'gl': df_opt["gl"],
                    'gl_adjusted': df_opt_adj.get("gl_adj", df_opt["gl"])
                })
                option_detail['gl_adjusted'] = (
                                                       option_detail['price_t_adj'] - option_detail['price_t1_cust']
                                               ) * option_detail['quantity_used'] * 100
                detailed_data['option_details'] = option_detail

                if not price_breaks.empty:
                    detailed_data['price_adjustments']['option'] = price_breaks

        # Process FLEX option details
        if not flex_today.empty and not flex_prior.empty:
            df_flex = flex_today.merge(
                flex_prior[["optticker", "price", "quantity"]],
                on="optticker", how="inner", suffixes=("_t", "_t1")
            ).dropna(subset=["price_t", "price_t1"])

            qty_col = "quantity_t1" if self.analysis_type == "ex_ante" else "quantity_t"
            df_flex["gl"] = (df_flex["price_t"] - df_flex["price_t1"]) * df_flex[qty_col] * 100
            df_flex = df_flex[df_flex["gl"].abs() > 0.01]

            if not df_flex.empty:
                df_flex_adj = self.apply_small_price_override(
                    df_flex.copy(), kind="flex_option", key_col="optticker"
                )

                price_breaks = self.holdings_price_breaks.get('flex_option', pd.DataFrame()) if hasattr(self, 'holdings_price_breaks') else pd.DataFrame()
                cust_price_map = {
                    ticker: self._lookup_cust_price(price_breaks, ticker) for ticker in df_flex['optticker']
                }
                price_t1_cust = df_flex['optticker'].map(cust_price_map).fillna(df_flex['price_t1'])

                flex_detail = pd.DataFrame({
                    'ticker': df_flex['optticker'],
                    'quantity_t1': df_flex['quantity_t1'],
                    'quantity_t': df_flex['quantity_t'],
                    'price_t1_raw': df_flex['price_t1'],
                    'price_t_raw': df_flex['price_t'],
                    'price_t1_cust': price_t1_cust,
                    'price_t1_adj': price_t1_cust,
                    'price_t_adj': df_flex_adj.get('price_t', df_flex['price_t']),
                    'quantity_used': df_flex[qty_col],
                    'gl': df_flex["gl"],
                    'gl_adjusted': df_flex_adj.get("gl_adj", df_flex["gl"])
                })
                flex_detail['gl_adjusted'] = (
                                                     flex_detail['price_t_adj'] - flex_detail['price_t1_cust']
                                             ) * flex_detail['quantity_used'] * 100
                detailed_data['flex_details'] = flex_detail

                if not price_breaks.empty:
                    detailed_data['price_adjustments']['flex_option'] = price_breaks

        # Add NAV components
        detailed_data['nav_components'] = {
            'beg_tna': previous.tna,
            'cust_tna': current.tna,
            'cust_nav': current.nav,
            'shares_outstanding': current.shares_outstanding,
            'expense_ratio': expense_rat,
            'analysis_date': getattr(self, 'analysis_date', None),
            'prior_date': getattr(self, 'prior_date', None)
        }

        # Include the structured detailed calculations built by builder functions
        detailed_data['detailed_calculations'] = dict(self._detailed_calculations)

        return detailed_data

    def add_summary_row(self, test_name, ticker, description, value):
        self.logger.info(f"[SUMMARY] {test_name} | {ticker} | {description} | {value}")

    def apply_small_price_override(self, df: pd.DataFrame, kind: str, key_col: str):
        """Apply small price overrides from holdings_price_breaks"""
        if not hasattr(self, 'holdings_price_breaks'):
            return df.copy()

        pb = self.holdings_price_breaks.get(kind, pd.DataFrame())
        if pb.empty:
            return df.copy()

        # only diffs >0.005 and <1
        mask = (pb["price_diff"] > 0.005) & (pb["price_diff"] < 1)
        small = pb.loc[mask, [key_col, "price_cust"]]
        price_map = small.set_index(key_col)["price_cust"].to_dict()
        df2 = df.copy()
        df2.loc[df2[key_col].isin(price_map), "price_t"] = df2[key_col].map(price_map)

        # Recalculate gl_adj if we modified prices
        if 'gl' in df2.columns:
            qty_col = "quantity_t1" if self.analysis_type == "ex_ante" else "quantity_t"
            if qty_col in df2.columns:
                # Use multiplier of 100 for both options and flex_options
                multiplier = 100 if kind in ("option", "flex_option") else 1
                df2["gl_adj"] = (df2["price_t"] - df2.get("price_t1", 0)) * df2[qty_col] * multiplier

        return df2

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

    def _emit_equity_summary_rows(self, df_issues: pd.DataFrame,
                                  price_disc_T: pd.DataFrame,
                                  price_disc_T1: pd.DataFrame):
        """Emit summary rows for equity reconciliation"""
        for _, row in df_issues.iterrows():
            dtype = row['discrepancy_type']
            desc = row.get('breakdown', '')
            val = row['final_discrepancy'] if dtype == 'Quantity Mismatch' else 'N/A'
            self.add_summary_row(f"Custodian Equity: {dtype}",
                                 row['eqyticker'], desc, val)

        for _, row in price_disc_T.iterrows():
            self.add_summary_row("Custodian Equity Price (T)",
                                 row['eqyticker'],
                                 f"{row['price_vest']:.2f} vs {row['price_cust']:.2f}",
                                 row['price_diff'])

        for _, row in price_disc_T1.iterrows():
            self.add_summary_row("Custodian Equity Price (T-1)",
                                 row['eqyticker'],
                                 f"{row['price_vest']:.2f} vs {row['price_cust']:.2f}",
                                 row['price_diff'])

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
                if self.uses_index_flex and 'optticker' in price_discrepancies.columns:
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
                if self.vehicle in {"private_fund", "closed_end_fund"}:
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

    def _lookup_cust_price(self, price_breaks: pd.DataFrame, ticker: str) -> float | None:
        """Extract custodian price for a ticker when available."""

        if price_breaks.empty:
            return None

        if ticker in price_breaks.index:
            entry = price_breaks.loc[ticker]
            if isinstance(entry, pd.Series):
                return entry.get('price_cust')
            if isinstance(entry, pd.DataFrame) and not entry.empty:
                return entry.iloc[0].get('price_cust')

        for col in ('eqyticker', 'optticker', 'cusip'):
            if col in price_breaks.columns:
                match = price_breaks[price_breaks[col] == ticker]
                if not match.empty:
                    return match.iloc[0].get('price_cust')

        return None

    def add_summary_row(self, test_name: str, ticker: str, description: str, value):
        """Capture summary rows without emitting them to the logger."""

        self.summary_rows.append(
            {
                "test": test_name,
                "ticker": ticker,
                "description": description,
                "value": value,
            }
        )

    def _build_equity_details(self):
        """Assemble detailed calculations for downstream reporting."""

        def _safe_result_df(name: str, attr: str) -> pd.DataFrame:
            result = self.results.get(name)
            if not result:
                return pd.DataFrame()
            value = getattr(result, attr, pd.DataFrame())
            return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()

        equity_details = {
            # Current day holdings
            'vest_current': self.fund.data.current.vest.equity.copy(),
            'custodian_current': self.fund.data.current.custodian.equity.copy(),

            # T-1 holdings
            'vest_previous': self.fund.data.previous.vest.equity.copy(),
            'custodian_previous': self.fund.data.previous.custodian.equity.copy(),

            # Other inputs
            'trades': self.fund.data.current.equity_trades.copy(),
            'corporate_actions': self.fund.data.current.cr_rd_data.copy(),
        }

        reconciliation_outputs = {
            'custodian_equity': {
                'final': _safe_result_df('custodian_equity', 'final_recon'),
                'price_T': _safe_result_df('custodian_equity', 'price_discrepancies_T'),
                'price_T1': _safe_result_df('custodian_equity', 'price_discrepancies_T1'),
            },
            'custodian_equity_t1': {
                'final': _safe_result_df('custodian_equity_t1', 'final_recon'),
            },
            'index_equity': {
                'final': self.results.get('index_equity', {}).get('significant_diffs', pd.DataFrame()).copy()
                if isinstance(self.results.get('index_equity'), dict) else pd.DataFrame(),
            },
        }

        self._detailed_calculations['equity'] = {
            'inputs': equity_details,
            'reconciliations': reconciliation_outputs,
        }

    def _build_option_details(self):
        """Assemble detailed calculations for regular options for downstream reporting."""

        def _safe_result_df(name: str, attr: str) -> pd.DataFrame:
            result = self.results.get(name)
            if not result:
                return pd.DataFrame()
            value = getattr(result, attr, pd.DataFrame())
            return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()

        current = self.fund.data.current
        previous = self.fund.data.previous

        option_details = {
            # Current day holdings
            'vest_current': current.vest.options.copy(),
            'custodian_current': current.custodian.options.copy(),

            # T-1 holdings
            'vest_previous': previous.vest.options.copy(),
            'custodian_previous': previous.custodian.options.copy(),

            # Option trades (if available)
            'trades': current.option_trades.copy() if hasattr(current, 'option_trades') else pd.DataFrame(),
        }

        reconciliation_outputs = {
            'custodian_option': {
                'final': _safe_result_df('custodian_option', 'final_recon'),
                'regular_options': _safe_result_df('custodian_option', 'regular_options'),
                'price_T': _safe_result_df('custodian_option', 'price_discrepancies_T'),
                'price_T1': _safe_result_df('custodian_option', 'price_discrepancies_T1'),
            },
            'custodian_option_t1': {
                'final': _safe_result_df('custodian_option_t1', 'final_recon'),
                'regular_options': _safe_result_df('custodian_option_t1', 'regular_options'),
            },
        }

        self._detailed_calculations['option'] = {
            'inputs': option_details,
            'reconciliations': reconciliation_outputs,
        }

    def _build_flex_option_details(self):
        """Assemble detailed calculations for FLEX options for downstream reporting."""

        def _safe_result_df(name: str, attr: str) -> pd.DataFrame:
            result = self.results.get(name)
            if not result:
                return pd.DataFrame()
            value = getattr(result, attr, pd.DataFrame())
            return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()

        # Only build if fund uses FLEX options
        if not self.uses_index_flex:
            self._detailed_calculations['flex_option'] = {
                'inputs': {},
                'reconciliations': {},
            }
            return

        current = self.fund.data.current
        previous = self.fund.data.previous

        flex_option_details = {
            # Current day holdings
            'vest_current': current.vest.flex_options.copy(),
            'custodian_current': current.custodian.flex_options.copy(),

            # T-1 holdings
            'vest_previous': previous.vest.flex_options.copy(),
            'custodian_previous': previous.custodian.flex_options.copy(),

            # FLEX option trades (if available)
            'trades': current.flex_option_trades.copy() if hasattr(current, 'flex_option_trades') else pd.DataFrame(),
        }

        reconciliation_outputs = {
            'custodian_flex_option': {
                'final': _safe_result_df('custodian_flex_option', 'final_recon'),
                'price_T': _safe_result_df('custodian_flex_option', 'price_discrepancies_T'),
                'price_T1': _safe_result_df('custodian_flex_option', 'price_discrepancies_T1'),
            },
            'custodian_flex_option_t1': {
                'final': _safe_result_df('custodian_flex_option_t1', 'final_recon'),
            },
        }

        self._detailed_calculations['flex_option'] = {
            'inputs': flex_option_details,
            'reconciliations': reconciliation_outputs,
        }

    def _build_treasury_details(self):
        """Assemble detailed calculations for treasuries for downstream reporting."""

        def _safe_result_df(name: str, attr: str) -> pd.DataFrame:
            result = self.results.get(name)
            if not result:
                return pd.DataFrame()
            value = getattr(result, attr, pd.DataFrame())
            return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()

        current = self.fund.data.current
        previous = self.fund.data.previous

        treasury_details = {
            # Current day holdings
            'vest_current': current.vest.treasury.copy(),
            'custodian_current': current.custodian.treasury.copy(),

            # T-1 holdings
            'vest_previous': previous.vest.treasury.copy(),
            'custodian_previous': previous.custodian.treasury.copy(),
        }

        reconciliation_outputs = {
            'custodian_treasury': {
                'final': _safe_result_df('custodian_treasury', 'final_recon'),
                'price_T': _safe_result_df('custodian_treasury', 'price_discrepancies_T'),
                'price_T1': _safe_result_df('custodian_treasury', 'price_discrepancies_T1'),
            },
            'custodian_treasury_t1': {
                'final': _safe_result_df('custodian_treasury_t1', 'final_recon'),
            },
        }

        self._detailed_calculations['treasury'] = {
            'inputs': treasury_details,
            'reconciliations': reconciliation_outputs,
        }

    def _check_sg_equity(self) -> bool:
        """Check if this fund has SG equity data"""
        sg_equity = self.fund_data.get('sg_equity', pd.DataFrame()) if isinstance(self.fund_data, dict) else pd.DataFrame()
        return not sg_equity.empty

