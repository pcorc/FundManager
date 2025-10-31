import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Optional
from domain.fund import Fund, FundSnapshot, FundHoldings


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
        self._detailed_calculations: Dict[str, Dict] = {}
        self.logger = logging.getLogger(f"Reconciliator_{fund.name}")
        self.is_etf = not fund.is_private_fund and not fund.is_closed_end_fund
        self.has_sg_equity = fund.is_private_fund

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------
    def _current_snapshot(self) -> FundSnapshot:
        snapshot = getattr(self.fund, "data", None)
        if snapshot is None or getattr(snapshot, "current", None) is None:
            return FundSnapshot()
        return snapshot.current

    def _previous_snapshot(self) -> FundSnapshot:
        snapshot = getattr(self.fund, "data", None)
        if snapshot is None or getattr(snapshot, "previous", None) is None:
            return FundSnapshot()
        return snapshot.previous

    def _current_frame(self, attribute: str, *, source: Optional[str] = None) -> pd.DataFrame:
        snapshot = self._current_snapshot()
        target = snapshot
        if source is not None:
            component = getattr(snapshot, source, None)
            if not isinstance(component, FundHoldings):
                return pd.DataFrame()
            target = component
        frame = getattr(target, attribute, pd.DataFrame())
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    def _previous_frame(self, attribute: str, *, source: Optional[str] = None) -> pd.DataFrame:
        snapshot = self._previous_snapshot()
        target = snapshot
        if source is not None:
            component = getattr(snapshot, source, None)
            if not isinstance(component, FundHoldings):
                return pd.DataFrame()
            target = component
        frame = getattr(target, attribute, pd.DataFrame())
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    def run_all_reconciliations(self):
        """Run all reconciliations using Fund object data"""
        recon_funcs = [
            ("custodian_equity", self.reconcile_custodian_equity),
            ("custodian_equity_t1", self.reconcile_custodian_equity_t1),
            ("custodian_option", self.reconcile_custodian_option),
            ("custodian_option_t1", self.reconcile_custodian_option_t1),
            ("custodian_treasury", self.reconcile_custodian_treasury),  # ← ADD THIS
            ("custodian_treasury_t1", self.reconcile_custodian_treasury_t1),  # ← ADD THIS
            ("index_equity", self.reconcile_index_equity),
        ]

        # Only run SG if not end-of-day
        if self.analysis_type != "eod":
            recon_funcs.append(("sg_option", self.reconcile_sg_option))
            if self.has_sg_equity:
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

        df = holdings_df.copy()

        ex_ante_candidates = [
            "quantity_tminus1",
            "shares_tminus1",
            "quantity_t_1",
            "qty_tminus1",
        ]
        base_candidates = [
            "quantity",
            "shares",
            "share_qty",
            "position",
            "vest_quantity",
            "qty",
        ]

        if self.analysis_type == "ex_ante":
            for column in ex_ante_candidates:
                if column in df.columns:
                    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)

        for column in base_candidates:
            if column in df.columns:
                return pd.to_numeric(df[column], errors="coerce").fillna(0.0)

        numeric_columns = df.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            return pd.to_numeric(df[numeric_columns[0]], errors="coerce").fillna(0.0)

        return pd.Series(0.0, index=df.index)

    @staticmethod
    def _coerce_numeric_series(series: pd.Series) -> pd.Series:
        """Return a numeric version of ``series`` with NaNs replaced by zero."""
        return pd.to_numeric(series, errors="coerce").fillna(0.0)

    def reconcile_custodian_equity(self):
        """Reconcile custodian equity using Fund object data"""
        # Get data from Fund object
        df_oms = self._current_frame('equity', source='vest')
        df_cust = self._current_frame('equity', source='custodian')
        df_oms1 = self._previous_frame('equity', source='vest')
        df_cust1 = self._previous_frame('equity', source='custodian')

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


    def reconcile_index_equity(self):
        """Compare fund equity holdings against benchmark weights."""

        df_fund = getattr(self.fund, "equity_holdings", pd.DataFrame()).copy()
        df_index = self._current_frame('index')

        if df_fund.empty or df_index.empty:
            self.results["index_equity"] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame(),
            )
            self.add_summary_row(
                "Index Equity",
                "",
                "Missing fund or benchmark holdings; reconciliation skipped",
                "N/A",
            )
            return

        if "equity_ticker" not in df_fund.columns or "equity_ticker" not in df_index.columns:
            self.results["index_equity"] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame(),
            )
            self.add_summary_row(
                "Index Equity",
                "",
                "equity_ticker column missing on fund or index data",
                "N/A",
            )
            return

        fund_weights = self._get_weight_series(df_fund)
        index_weights = self._get_weight_series(
            df_index,
            preferred_columns=["weight_index", "index_weight", "benchmark_weight"],
        )

        if fund_weights is None or index_weights is None:
            self.results["index_equity"] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame(),
            )
            self.add_summary_row(
                "Index Equity",
                "",
                "Unable to derive weights for comparison",
                "N/A",
            )
            return

        df_fund = df_fund.assign(_fund_weight=fund_weights)
        df_index = df_index.assign(_index_weight=index_weights)

        merged = pd.merge(
            df_fund[["equity_ticker", "_fund_weight"]],
            df_index[["equity_ticker", "_index_weight"]],
            on="equity_ticker",
            how="outer",
            indicator=True,
        )

        merged["_fund_weight"] = merged["_fund_weight"].fillna(0.0)
        merged["_index_weight"] = merged["_index_weight"].fillna(0.0)
        merged["weight_diff"] = merged["_fund_weight"] - merged["_index_weight"]
        merged["abs_weight_diff"] = merged["weight_diff"].abs()

        tolerance = 0.0005
        issues = merged[(merged["_merge"] != "both") | merged["abs_weight_diff"].gt(tolerance)].copy()

        if not issues.empty:
            issues["issue_type"] = np.where(
                issues["_merge"] == "both",
                "Weight Mismatch",
                np.where(issues["_merge"] == "left_only", "Missing in Index", "Missing in Fund"),
            )
        else:
            issues = pd.DataFrame(columns=[
                "equity_ticker",
                "_fund_weight",
                "_index_weight",
                "weight_diff",
                "abs_weight_diff",
                "_merge",
                "issue_type",
            ])

        self.results["index_equity"] = ReconciliationResult(
            raw_recon=issues.copy(),
            final_recon=issues.copy(),
            price_discrepancies_T=pd.DataFrame(),
            price_discrepancies_T1=pd.DataFrame(),
            merged_data=merged.drop(columns=["_merge"]),
        )

        for _, row in issues.iterrows():
            description = (
                f"Fund {row['_fund_weight']:.6f} vs Index {row['_index_weight']:.6f}"
                if row["issue_type"] == "Weight Mismatch"
                else row["issue_type"]
            )
            self.add_summary_row(
                "Index Equity",
                row.get("equity_ticker", ""),
                description,
                row.get("weight_diff", "N/A"),
            )


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
        df_oms = self._current_frame('options', source='vest')
        df_cust = self._current_frame('options', source='custodian')
        df_oms1 = self._previous_frame('options', source='vest')
        df_cust1 = self._previous_frame('options', source='custodian')

        # Take absolute value of option prices
        for d in [df_oms, df_oms1]:
            if not d.empty and 'price' in d.columns:
                d['price'] = d['price'].abs()

        # Calculate option weights for standard options only
        if not df_oms.empty and 'price' in df_oms.columns and 'quantity' in df_oms.columns:
            df_oms['market_value'] = df_oms['quantity'].fillna(0) * df_oms['price'].fillna(0) * 100
            df_oms['is_flex'] = (
                    df_oms['optticker'].str.contains("SPX|XSP", na=False) &
                    (self.fund.is_private_fund or self.fund.is_closed_end_fund)
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
                (self.fund.is_private_fund or self.fund.is_closed_end_fund)
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
                        (self.fund.is_private_fund or self.fund.is_closed_end_fund)
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
                if self.fund.is_private_fund or self.fund.is_closed_end_fund:
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

    def add_summary_row(self, test_name: str, ticker: str, description: str, value):
        """Add summary row for logging"""
        self.logger.info(f"[SUMMARY] {test_name} | {ticker} | {description} | {value}")

    # Keep your existing _build_equity_details, get_detailed_calculations methods
    # but update them to use self.fund instead of self.fund_data

    def reconcile_custodian_equity_t1(self):
        """Reconcile custodian equity for T-1 date (simpler version)"""
        # Get T-1 data from Fund object
        df_oms1 = self._previous_frame('equity', source='vest')
        df_cust1 = self._previous_frame('equity', source='custodian')

        if df_oms1.empty or df_cust1.empty:
            self.results['custodian_equity_t1'] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame()
            )
            return

        # Set quantities for T-1
        df_oms1['quantity'] = self._get_quantity_column(df_oms1)

        # Merge T-1 data
        df1 = pd.merge(df_oms1, df_cust1, on='equity_ticker', how='outer',
                       suffixes=('_vest', '_cust'), indicator=True)
        df1['in_vest'] = df1['_merge'] != 'right_only'
        df1['in_cust'] = df1['_merge'] != 'left_only'
        df1.drop(columns=['_merge'], inplace=True)

        # Calculate T-1 discrepancies (no trades/corporate actions for T-1)
        df1['discrepancy'] = df1['quantity'].fillna(0) - df1.get('shares_cust', 0).fillna(0)

        # Identify T-1 mismatches
        mask_missing_t1 = df1['in_vest'] != df1['in_cust']
        mask_qty_t1 = df1['in_vest'] & df1['in_cust'] & df1['discrepancy'].abs().gt(0)
        df_issues_t1 = df1.loc[mask_missing_t1 | mask_qty_t1].copy()

        # Categorize T-1 discrepancies
        conditions_t1 = [
            ~df_issues_t1['in_vest'] & df_issues_t1['in_cust'],
            df_issues_t1['in_vest'] & ~df_issues_t1['in_cust'],
            mask_qty_t1.loc[df_issues_t1.index]
        ]
        df_issues_t1['discrepancy_type'] = np.select(conditions_t1,
            ["Missing in OMS", "Missing in Custodian", "Quantity Mismatch"],
            default="Unknown")

        # T-1 breakdown
        df_issues_t1['breakdown'] = np.where(
            df_issues_t1['discrepancy_type'] == "Quantity Mismatch",
            "Vest=" + df_issues_t1['quantity'].fillna(0).astype(int).astype(str)
            + " | Cust=" + df_issues_t1.get('shares_cust', 0).fillna(0).astype(int).astype(str),
            "Present in Vest: " + df_issues_t1['in_vest'].map({True: "Yes", False: "No"})
            + " | Present in Cust: " + df_issues_t1['in_cust'].map({True: "Yes", False: "No"})
        )

        # Store T-1 results
        self.results['custodian_equity_t1'] = ReconciliationResult(
            raw_recon=df1.loc[mask_qty_t1].reset_index(drop=True),
            final_recon=df_issues_t1.reset_index(drop=True),
            price_discrepancies_T=pd.DataFrame(),  # No T price discrepancies for T-1 recon
            price_discrepancies_T1=self._calculate_price_discrepancies(df1, 'equity_ticker'),
            merged_data=df1.copy()
        )

        # Emit T-1 summary rows
        for _, row in df_issues_t1.iterrows():
            dtype = row['discrepancy_type']
            desc = row.get('breakdown', '')
            val = row['discrepancy'] if dtype == 'Quantity Mismatch' else 'N/A'
            self.add_summary_row(f"Custodian Equity T-1: {dtype}",
                                 row['equity_ticker'], desc, val)

    def reconcile_custodian_option_t1(self):
        """Reconcile custodian options for T-1 date"""
        # Get T-1 data
        df_oms1 = self.fund.previous_options_holdings.copy()
        df_cust1 = self.fund.previous_custodian_option_holdings.copy()

        if df_oms1.empty and df_cust1.empty:
            self.results['custodian_option_t1'] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame()
            )
            return

        if (
            ('optticker' not in df_oms1.columns and not df_oms1.empty)
            or ('optticker' not in df_cust1.columns and not df_cust1.empty)
        ):
            self.results['custodian_option_t1'] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame()
            )
            return

        df_oms1['quantity'] = self._get_quantity_column(df_oms1)
        if 'shares_cust' not in df_cust1.columns:
            for candidate in ["quantity", "contracts", "par", "position", "shares"]:
                if candidate in df_cust1.columns:
                    df_cust1['shares_cust'] = df_cust1[candidate]
                    break
            else:
                df_cust1['shares_cust'] = 0.0 if not df_cust1.empty else pd.Series(dtype=float)
        df_cust1['shares_cust'] = self._coerce_numeric_series(df_cust1['shares_cust'])

        df1 = pd.merge(
            df_oms1,
            df_cust1,
            on='optticker',
            how='outer',
            suffixes=('_vest', '_cust')
        )

        quantity_series = df1['quantity'] if 'quantity' in df1 else pd.Series(0.0, index=df1.index)
        shares_series = df1['shares_cust'] if 'shares_cust' in df1 else pd.Series(0.0, index=df1.index)
        df1['quantity'] = self._coerce_numeric_series(quantity_series)
        df1['shares_cust'] = self._coerce_numeric_series(shares_series)
        df1['in_vest'] = df1['quantity'].notnull()
        df1['in_cust'] = df1['shares_cust'].notnull()
        df1['discrepancy'] = df1['quantity'] - df1['shares_cust']

        mask_missing = df1['in_vest'] != df1['in_cust']
        mask_qty = df1['in_vest'] & df1['in_cust'] & df1['discrepancy'].abs().gt(0)
        df_issues = df1.loc[mask_missing | mask_qty].copy()

        if not df_issues.empty:
            df_issues['discrepancy_type'] = np.where(
                mask_qty.loc[df_issues.index],
                'Quantity Mismatch',
                'Holdings Mismatch'
            )
            df_issues['breakdown'] = np.where(
                df_issues['discrepancy_type'] == 'Quantity Mismatch',
                "Vest=" + df_issues['quantity'].round(0).astype(int).astype(str)
                + " | Cust=" + df_issues['shares_cust'].round(0).astype(int).astype(str),
                "Present in Vest: " + df_issues['in_vest'].map({True: 'Yes', False: 'No'})
                + " | Present in Cust: " + df_issues['in_cust'].map({True: 'Yes', False: 'No'})
            )
        else:
            df_issues = pd.DataFrame(columns=[
                'optticker',
                'quantity',
                'shares_cust',
                'in_vest',
                'in_cust',
                'discrepancy',
                'discrepancy_type',
                'breakdown'
            ])

        df_issues['is_flex'] = (
            df_issues.get('optticker', pd.Series(dtype=str)).str.contains("SPX|XSP", na=False) &
            (self.fund.is_private_fund or self.fund.is_closed_end_fund)
        )

        flex_issues = df_issues[df_issues['is_flex']].copy() if not df_issues.empty else pd.DataFrame()
        regular_issues = df_issues[~df_issues['is_flex']].copy() if not df_issues.empty else pd.DataFrame()

        price_T1 = self._calculate_option_price_discrepancies(df1, 'T1')

        self.results['custodian_option_t1'] = ReconciliationResult(
            raw_recon=df1.loc[mask_qty].reset_index(drop=True) if 'optticker' in df1 else pd.DataFrame(),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=pd.DataFrame(),
            price_discrepancies_T1=price_T1,
            merged_data=df1.copy(),
            regular_options=regular_issues,
            flex_options=flex_issues
        )

        for _, row in df_issues.iterrows():
            dtype = row.get('discrepancy_type', 'Unknown')
            desc = row.get('breakdown', '')
            value = row.get('discrepancy', 'N/A') if dtype == 'Quantity Mismatch' else 'N/A'
            self.add_summary_row(f"Custodian Option T-1: {dtype}", row.get('optticker', ''), desc, value)

    def reconcile_custodian_treasury(self):
        """Reconcile custodian treasury holdings for current date"""
        df_oms = self.fund.treasury_holdings.copy()
        df_cust = self.fund.custodian_treasury_holdings.copy()

        if df_oms.empty and df_cust.empty:
            self.results['custodian_treasury'] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame()
            )
            return

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

        df_oms['quantity'] = self._get_quantity_column(df_oms)
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

        quantity_series = df['quantity'] if 'quantity' in df else pd.Series(0.0, index=df.index)
        shares_series = df['shares_cust'] if 'shares_cust' in df else pd.Series(0.0, index=df.index)
        df['quantity'] = self._coerce_numeric_series(quantity_series)
        df['shares_cust'] = self._coerce_numeric_series(shares_series)
        df['in_vest'] = df['_merge'] != 'right_only'
        df['in_cust'] = df['_merge'] != 'left_only'
        df['quantity_diff'] = df['quantity'] - df['shares_cust']

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
                "Vest=" + df_issues['quantity'].round(2).astype(str)
                + " | Cust=" + df_issues['shares_cust'].round(2).astype(str),
                "Present in Vest: " + df_issues['in_vest'].map({True: 'Yes', False: 'No'})
                + " | Present in Cust: " + df_issues['in_cust'].map({True: 'Yes', False: 'No'})
            )
        else:
            df_issues = pd.DataFrame(columns=[
                'cusip',
                'quantity',
                'shares_cust',
                'in_vest',
                'in_cust',
                'quantity_diff',
                'discrepancy_type',
                'breakdown'
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

    def reconcile_custodian_treasury_t1(self):
        """Reconcile custodian treasury holdings for T-1 date"""
        # Similar to reconcile_custodian_treasury but using previous data
        df_oms1 = self._previous_frame('treasury', source='vest')
        df_cust1 = self._previous_frame('treasury', source='custodian')
        if df_oms1.empty and df_cust1.empty:
            self.results['custodian_treasury_t1'] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame()
            )
            return

        if 'cusip' not in df_oms1.columns:
            if df_oms1.empty:
                df_oms1['cusip'] = pd.Series(dtype=object)
            else:
                self.logger.warning("Missing CUSIP column for OMS treasury T-1 holdings")
                self.results['custodian_treasury_t1'] = ReconciliationResult(
                    raw_recon=pd.DataFrame(),
                    final_recon=pd.DataFrame(),
                    price_discrepancies_T=pd.DataFrame(),
                    price_discrepancies_T1=pd.DataFrame(),
                    merged_data=pd.DataFrame()
                )
                return

        if 'cusip' not in df_cust1.columns:
            if df_cust1.empty:
                df_cust1['cusip'] = pd.Series(dtype=object)
            else:
                self.logger.warning("Missing CUSIP column for custodian treasury T-1 holdings")
                self.results['custodian_treasury_t1'] = ReconciliationResult(
                    raw_recon=pd.DataFrame(),
                    final_recon=pd.DataFrame(),
                    price_discrepancies_T=pd.DataFrame(),
                    price_discrepancies_T1=pd.DataFrame(),
                    merged_data=pd.DataFrame()
                )
                return

        df_oms1['quantity'] = self._get_quantity_column(df_oms1)
        if 'shares_cust' not in df_cust1.columns:
            for candidate in ['quantity', 'par', 'par_value', 'face', 'position', 'amount']:
                if candidate in df_cust1.columns:
                    df_cust1['shares_cust'] = df_cust1[candidate]
                    break
            else:
                df_cust1['shares_cust'] = 0.0
        df_cust1['shares_cust'] = self._coerce_numeric_series(df_cust1['shares_cust'])

        df1 = pd.merge(
            df_oms1,
            df_cust1,
            on='cusip',
            how='outer',
            suffixes=('_vest', '_cust'),
            indicator=True
        )

        quantity_series = df1['quantity'] if 'quantity' in df1 else pd.Series(0.0, index=df1.index)
        shares_series = df1['shares_cust'] if 'shares_cust' in df1 else pd.Series(0.0, index=df1.index)
        df1['quantity'] = self._coerce_numeric_series(quantity_series)
        df1['shares_cust'] = self._coerce_numeric_series(shares_series)
        df1['in_vest'] = df1['_merge'] != 'right_only'
        df1['in_cust'] = df1['_merge'] != 'left_only'
        df1['quantity_diff'] = df1['quantity'] - df1['shares_cust']

        mask_missing = df1['in_vest'] != df1['in_cust']
        mask_qty = df1['in_vest'] & df1['in_cust'] & df1['quantity_diff'].abs().gt(0)
        df_issues = df1.loc[mask_missing | mask_qty].copy()

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
                "Vest=" + df_issues['quantity'].round(2).astype(str)
                + " | Cust=" + df_issues['shares_cust'].round(2).astype(str),
                "Present in Vest: " + df_issues['in_vest'].map({True: 'Yes', False: 'No'})
                + " | Present in Cust: " + df_issues['in_cust'].map({True: 'Yes', False: 'No'})
            )
        else:
            df_issues = pd.DataFrame(columns=[
                'cusip',
                'quantity',
                'shares_cust',
                'in_vest',
                'in_cust',
                'quantity_diff',
                'discrepancy_type',
                'breakdown'
            ])

        price_disc_T1 = self._calculate_price_discrepancies(df1, 'cusip')

        self.results['custodian_treasury_t1'] = ReconciliationResult(
            raw_recon=df1.loc[mask_qty].reset_index(drop=True),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=pd.DataFrame(),
            price_discrepancies_T1=price_disc_T1,
            merged_data=df1.copy()
        )

        for _, row in df_issues.iterrows():
            dtype = row.get('discrepancy_type', 'Unknown')
            desc = row.get('breakdown', '')
            value = row.get('quantity_diff', 'N/A') if dtype == 'Quantity Mismatch' else 'N/A'
            self.add_summary_row(f"Custodian Treasury T-1: {dtype}", row.get('cusip', ''), desc, value)

        for _, row in price_disc_T1.iterrows():
            self.add_summary_row(
                "Custodian Treasury Price (T-1)",
                row.get('cusip', ''),
                f"{row['price_vest']:.4f} vs {row['price_cust']:.4f}",
                row.get('price_diff', 0.0),
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
            'vest_current': self._current_frame('equity', source='vest'),
            'custodian_current': self._current_frame('equity', source='custodian'),
            'vest_previous': self._previous_frame('equity', source='vest'),
            'custodian_previous': self._previous_frame('equity', source='custodian'),
            'trades': self.fund.equity_trades.copy(),
            'corporate_actions': self.fund.cr_rd_data.copy(),
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
                'final': _safe_result_df('index_equity', 'final_recon'),
            },
        }

        self._detailed_calculations['equity'] = {
            'inputs': equity_details,
            'reconciliations': reconciliation_outputs,
        }

    def get_detailed_calculations(self) -> Dict[str, Dict]:
        """Return cached detailed calculation payloads for reporting."""

        if 'equity' not in self._detailed_calculations:
            self._build_equity_details()
        return self._detailed_calculations

    def _get_weight_series(self, df: pd.DataFrame, preferred_columns: Optional[list[str]] = None) -> Optional[pd.Series]:
        columns = list(preferred_columns or []) + [
            "start_wgt",
            "weight",
            "fund_weight",
            "portfolio_weight",
        ]

        for column in columns:
            if column in df.columns:
                series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
                total = series.sum()
                if total != 0 and total != 1:
                    series = series / total
                return series

        if {"price", "quantity"}.issubset(df.columns):
            prices = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
            quantities = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
            market_values = prices * quantities
            total_mv = market_values.sum()
            if total_mv:
                return market_values / total_mv
        elif "market_value" in df.columns:
            market_values = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)
            total_mv = market_values.sum()
            if total_mv:
                return market_values / total_mv

        return None