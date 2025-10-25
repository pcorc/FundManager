import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Optional
from config.fund_classifications import (
    DIVERSIFIED_FUNDS, NON_DIVERSIFIED_FUNDS, PRIVATE_FUNDS,
    CLOSED_END_FUNDS, FUNDS_WITH_SG_EQUITY
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
    def __init__(self, fund: 'Fund', analysis_type: Optional[str] = None):
        """
        Updated Reconciliator that works with Fund objects.

        Args:
            fund: Fund object containing all data
            analysis_type: Type of analysis ('ex_ante', 'eod', etc.)
        """
        self.fund = fund
        self.fund_name = fund.name
        self.analysis_type = analysis_type
        self.results: Dict[str, ReconciliationResult] = {}
        self.logger = logging.getLogger(f"Reconciliator_{fund.name}")
        self.is_etf = fund.name not in PRIVATE_FUNDS and fund.name not in CLOSED_END_FUNDS

    def run_all_reconciliations(self):
        """Run all reconciliations using Fund object data"""
        recon_funcs = [
            ("custodian_equity", self.reconcile_custodian_equity),
            ("custodian_equity_t1", self.reconcile_custodian_equity_t1),
            ("custodian_option", self.reconcile_custodian_option),
            ("custodian_option_t1", self.reconcile_custodian_option_t1),
            ("index_equity", self.reconcile_index_equity),
        ]

        # Only run SG if not end-of-day
        if self.analysis_type != "eod":
            recon_funcs.append(("sg_option", self.reconcile_sg_option))
            if self.fund_name in FUNDS_WITH_SG_EQUITY:
                recon_funcs.append(("sg_equity", self.reconcile_sg_equity))

        for name, func in recon_funcs:
            try:
                func()
            except Exception as e:
                self.logger.error(f"Error in {name} reconciliation: {e}", exc_info=True)

        self._build_equity_details()

    def _get_quantity_column(self, holdings_df: pd.DataFrame) -> pd.Series:
        """
        Get appropriate quantity column based on analysis type.

        Args:
            holdings_df: DataFrame with holdings data

        Returns:
            Series with correct quantity values
        """
        if holdings_df.empty:
            return pd.Series(dtype=float)

        if self.analysis_type == "ex_ante" and "quantity_tminus1" in holdings_df.columns:
            return holdings_df["quantity_tminus1"].fillna(0)
        else:
            return holdings_df["quantity"].fillna(0)

    def reconcile_custodian_equity(self):
        """Reconcile custodian equity using Fund object data"""
        # Get data from Fund object
        df_oms = self.fund.equity_holdings.copy()
        df_cust = self.fund.custodian_equity_holdings.copy()
        df_oms1 = self.fund.previous_equity_holdings.copy()
        df_cust1 = self.fund.previous_custodian_equity_holdings.copy()

        # Get trades and corporate actions from Fund
        df_trades = self.fund.equity_trades
        df_crrd = self.fund.cr_rd_data

        # Set quantities based on analysis type
        df_oms['quantity'] = self._get_quantity_column(df_oms)
        df_oms1['quantity'] = self._get_quantity_column(df_oms1)

        if 'equity_ticker' not in df_oms.columns or 'equity_ticker' not in df_cust.columns:
            self.results['custodian_equity'] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame()
            )
            return

        # Merge base tables (today)
        df = pd.merge(df_oms, df_cust, on='equity_ticker', how='outer',
                      suffixes=('_vest', '_cust'), indicator=True)
        df['in_vest'] = df['_merge'] != 'right_only'
        df['in_cust'] = df['_merge'] != 'left_only'
        df.drop(columns=['_merge'], inplace=True)

        # Same for T-1
        df1 = pd.merge(df_oms1, df_cust1, on='equity_ticker', how='outer',
                       suffixes=('_vest', '_cust'), indicator=False)

        # Trades & corporate actions (only today)
        if not df_trades.empty and 'equity_ticker' in df_trades.columns:
            trade_map = df_trades.set_index('equity_ticker')['qty_sign_adj']
            df['qty_sign_adj'] = df['equity_ticker'].map(trade_map).fillna(0)
        else:
            df['qty_sign_adj'] = 0

        if self.is_etf and not df_crrd.empty and 'equity_ticker' in df_crrd.columns:
            cr_map = df_crrd.set_index('equity_ticker')['cr_rd']
            df['cr_rd'] = df['equity_ticker'].map(cr_map).fillna(0)
        else:
            df['cr_rd'] = 0

        # Adjusted shares & base discrepancy
        df['adjusted_cust_shares'] = df['shares_cust'].fillna(0) + df['qty_sign_adj']
        df['final_adjusted_shares'] = df['adjusted_cust_shares'] + df['cr_rd']
        df['final_discrepancy'] = df['quantity'].fillna(0) - df['final_adjusted_shares']

        # Identify mismatches
        mask_missing = df['in_vest'] != df['in_cust']
        mask_qty = df['in_vest'] & df['in_cust'] & df['final_discrepancy'].abs().gt(0)
        df_issues = df.loc[mask_missing | mask_qty].copy()

        # Categorize discrepancies
        conditions = [
            ~df_issues['in_vest'] & df_issues['in_cust'],
            df_issues['in_vest'] & ~df_issues['in_cust'],
            mask_qty.loc[df_issues.index]
        ]
        choices = ["Missing in OMS", "Missing in Custodian", "Quantity Mismatch"]
        df_issues['discrepancy_type'] = np.select(conditions, choices, default="Unknown")

        # Breakdown text
        df_issues['breakdown'] = np.where(
            df_issues['discrepancy_type'] == "Quantity Mismatch",
            "Vest=" + df_issues['quantity'].fillna(0).astype(int).astype(str)
            + " | Cust=" + df_issues['shares_cust'].fillna(0).astype(int).astype(str)
            + " | TradesAdj=" + df_issues['qty_sign_adj'].astype(int).astype(str)
            + " | CR/RD=" + df_issues['cr_rd'].astype(int).astype(str),
            "Present in Vest: " + df_issues['in_vest'].map({True: "Yes", False: "No"})
            + " | Present in Cust: " + df_issues['in_cust'].map({True: "Yes", False: "No"})
        )

        # Price-difference tables for T and T-1
        price_disc_T = self._calculate_price_discrepancies(df, 'equity_ticker')
        price_disc_T1 = self._calculate_price_discrepancies(df1, 'equity_ticker')

        # Store results
        self.results['custodian_equity'] = ReconciliationResult(
            raw_recon=df.loc[mask_qty].reset_index(drop=True),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=price_disc_T,
            price_discrepancies_T1=price_disc_T1,
            merged_data=df.copy()
        )

        # Emit summary rows
        self._emit_equity_summary_rows(df_issues, price_disc_T, price_disc_T1)

    def _calculate_price_discrepancies(self, df: pd.DataFrame, ticker_col: str) -> pd.DataFrame:
        """Calculate price discrepancies between vest and custodian"""
        if {'price_vest', 'price_cust'}.issubset(df.columns):
            df_price = df.copy()
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
                                 row['equity_ticker'], desc, val)

        for _, row in price_disc_T.iterrows():
            self.add_summary_row("Custodian Equity Price (T)",
                                 row['equity_ticker'],
                                 f"{row['price_vest']:.2f} vs {row['price_cust']:.2f}",
                                 row['price_diff'])

        for _, row in price_disc_T1.iterrows():
            self.add_summary_row("Custodian Equity Price (T-1)",
                                 row['equity_ticker'],
                                 f"{row['price_vest']:.2f} vs {row['price_cust']:.2f}",
                                 row['price_diff'])

    def reconcile_custodian_option(self):
        """Reconcile custodian options using Fund object data"""
        # Get data from Fund object
        df_oms = self.fund.options_holdings.copy()
        df_cust = self.fund.custodian_option_holdings.copy()
        df_oms1 = self.fund.previous_options_holdings.copy()
        df_cust1 = self.fund.previous_custodian_option_holdings.copy()

        # Take absolute value of option prices
        for d in [df_oms, df_oms1]:
            if not d.empty and 'price' in d.columns:
                d['price'] = d['price'].abs()

        # Calculate option weights for standard options only
        if not df_oms.empty and 'price' in df_oms.columns and 'quantity' in df_oms.columns:
            df_oms['market_value'] = df_oms['quantity'].fillna(0) * df_oms['price'].fillna(0) * 100
            df_oms['is_flex'] = (
                    df_oms['optticker'].str.contains("SPX|XSP", na=False) &
                    (self.fund_name in PRIVATE_FUNDS or self.fund_name in CLOSED_END_FUNDS)
            )

            standard_option_mask = ~df_oms['is_flex']
            total_standard_mv = df_oms.loc[standard_option_mask, 'market_value'].sum()

            df_oms['option_weight'] = 0.0
            if total_standard_mv > 0:
                df_oms.loc[standard_option_mask, 'option_weight'] = (
                        df_oms.loc[standard_option_mask, 'market_value'] / total_standard_mv
                )

        # Merge data
        df = pd.merge(df_oms, df_cust, on='optticker', how='outer',
                      suffixes=('_vest', '_cust'))
        df1 = pd.merge(df_oms1, df_cust1, on='optticker', how='outer',
                       suffixes=('_vest', '_cust'))

        # Holdings & quantity mismatches
        df['in_vest'] = df['quantity'].notnull()
        df['in_cust'] = df['shares_cust'].notnull()
        df['trade_discrepancy'] = df['quantity'].fillna(0) - df['shares_cust'].fillna(0)

        hold_disc = df[df['in_vest'] != df['in_cust']]
        qty_disc = df[df['in_vest'] & df['in_cust'] & df['trade_discrepancy'].abs().gt(0)]
        df_issues = pd.concat([hold_disc, qty_disc], ignore_index=True).drop_duplicates()

        # Add discrepancy type and breakdown
        df_issues['discrepancy_type'] = np.where(
            df_issues['in_vest'] & df_issues['in_cust'],
            'Quantity Mismatch',
            'Holdings Mismatch'
        )

        breakdowns = []
        for _, row in df_issues.iterrows():
            if row['discrepancy_type'] == 'Quantity Mismatch':
                oms_qty = f"{row.get('quantity', 0):.0f}" if not pd.isna(row.get('quantity', 0)) else "0"
                cust_qty = f"{row.get('shares_cust', 0):.0f}" if not pd.isna(row.get('shares_cust', 0)) else "0"
                breakdown = f"OMS: {oms_qty} | Custodian: {cust_qty}"
            else:
                in_vest = "Yes" if row.get('in_vest', False) else "No"
                in_cust = "Yes" if row.get('in_cust', False) else "No"
                breakdown = f"Present in OMS: {in_vest} | Present in Custodian: {in_cust}"
            breakdowns.append(breakdown)

        df_issues['breakdown'] = breakdowns
        df_issues['is_flex'] = (
                df_issues['optticker'].str.contains("SPX|XSP", na=False) &
                (self.fund_name in PRIVATE_FUNDS or self.fund_name in CLOSED_END_FUNDS)
        )

        # Separate FLEX and regular options
        flex_issues = df_issues[df_issues['is_flex']].copy() if not df_issues.empty else pd.DataFrame()
        regular_issues = df_issues[~df_issues['is_flex']].copy() if not df_issues.empty else pd.DataFrame()

        # Price breaks T and T-1
        price_T = self._calculate_option_price_discrepancies(df, 'T')
        price_T1 = self._calculate_option_price_discrepancies(df1, 'T1')

        # Store results
        self.results['custodian_option'] = ReconciliationResult(
            raw_recon=df_issues.copy(),
            final_recon=df_issues.copy(),
            price_discrepancies_T=price_T,
            price_discrepancies_T1=price_T1,
            merged_data=df.copy(),
            regular_options=regular_issues,
            flex_options=flex_issues
        )

        # Emit summaries
        self._emit_option_summary_rows(df_issues, price_T, price_T1)

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
                price_discrepancies['is_flex'] = (
                        price_discrepancies['optticker'].str.contains("SPX|XSP", na=False) &
                        (self.fund_name in PRIVATE_FUNDS or self.fund_name in CLOSED_END_FUNDS)
                )

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
                if self.fund_name in PRIVATE_FUNDS or self.fund_name in CLOSED_END_FUNDS:
                    standard_price = price_discrepancies[~price_discrepancies['is_flex']].copy()
                    flex_price = price_discrepancies[price_discrepancies['is_flex']].copy()

                    if not standard_price.empty:
                        standard_price = standard_price.nlargest(5, 'option_weight')

                    price_discrepancies = pd.concat([standard_price, flex_price], ignore_index=True)

            return price_discrepancies.reset_index(drop=True)
        return pd.DataFrame()

    # ... similar updates for other reconciliation methods

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
        }

        if self.analysis_type != "eod":
            recon_keys["sg_option"] = ["final_recon", "price_discrepancies"]
            if self.fund_name in FUNDS_WITH_SG_EQUITY:
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

    def add_summary_row(self, test_name: str, ticker: str, description: str, value):
        """Add summary row for logging"""
        self.logger.info(f"[SUMMARY] {test_name} | {ticker} | {description} | {value}")

    # Keep your existing _build_equity_details, get_detailed_calculations methods
    # but update them to use self.fund instead of self.fund_data