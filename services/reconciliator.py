import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Optional
from domain.fund import Fund, FundSnapshot, FundHoldings
from config.fund_definitions import (
    DIVERSIFIED_FUNDS,
    NON_DIVERSIFIED_FUNDS,
    PRIVATE_FUNDS,
    CLOSED_END_FUNDS,
)
from utilities.ticker_utils import normalize_equity_pair, normalize_option_pair


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
    def __init__(self, fund_name, fund_data, analysis_type=None):
        self.fund_name = fund_name
        self.fund_data = fund_data
        self.analysis_type = analysis_type
        self.results = {}
        self.logger = logging.getLogger(f"Reconciliator_{fund_name}")
        self.is_etf = fund_name not in PRIVATE_FUNDS and fund_name not in CLOSED_END_FUNDS
        self.fund = fund_data.get('fund_object') if isinstance(fund_data, dict) else None
        self.holdings_price_breaks = fund_data.get("holdings_price_breaks", {}) if isinstance(fund_data, dict) else {}
        self.summary_rows = []  # Initialize summary_rows list
        self.has_sg_equity = self._check_sg_equity()  # Add property check

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

    def _set_internal_quantity_column(self, df_internal):
        """
        Sets df_internal['quantity'] based on self.analysis_type:
        - 'ex_ante' → quantity_tminus1
        - 'eod'     → quantity (leave as-is)
        """
        if df_internal.empty:
            return df_internal

        if self.analysis_type == "ex_ante" and "quantity_tminus1" in df_internal.columns:
            df_internal["quantity"] = df_internal["quantity_tminus1"]

        return df_internal

    def add_summary_row(self, test_name, ticker, description, value):
        self.logger.info(f"[SUMMARY] {test_name} | {ticker} | {description} | {value}")

    def get_detailed_calculations(self):
        """
        Returns detailed ticker-level calculations for Excel reporting.
        This should be called after run_nav_reconciliation().
        """
        detailed_data = {
            'summary': self.results,
            'equity_details': pd.DataFrame(),
            'option_details': pd.DataFrame(),
            'flex_details': pd.DataFrame(),
            'treasury_details': pd.DataFrame(),
            'price_adjustments': {
                'equity': pd.DataFrame(),
                'option': pd.DataFrame()
            }
        }

        # Get the dataframes
        nav_today = self.fund_data.get("nav", pd.DataFrame())
        nav_prior = self.fund_data.get("t1_nav", pd.DataFrame())
        eq_today = self.fund_data.get("equity_holdings", pd.DataFrame())
        eq_prior = self.fund_data.get("t1_equity_holdings", pd.DataFrame())
        opt_today = self.fund_data.get("options_holdings", pd.DataFrame())
        opt_prior = self.fund_data.get("t1_options_holdings", pd.DataFrame())

        # Get expense ratio
        expense_rat = self.fund_data.get("expense_ratio", 0.0)

        # Process equity details
        if not eq_today.empty and not eq_prior.empty:
            df_eq = eq_today.merge(
                eq_prior[["equity_ticker", "price", "quantity"]],
                on="equity_ticker", how="inner", suffixes=("_t", "_t1")
            ).dropna(subset=["price_t", "price_t1"])

            qty_col = "quantity_t1" if self.analysis_type == "ex_ante" else "quantity_t"

            # Calculate G/L
            df_eq["gl"] = (df_eq["price_t"] - df_eq["price_t1"]) * df_eq[qty_col]

            # Only include equities with non-zero G/L
            df_eq = df_eq[df_eq["gl"].abs() > 0.01]

            if not df_eq.empty:
                # Apply price adjustments and track them
                df_eq_adj = self.apply_small_price_override(
                    df_eq.copy(), kind="equity", key_col="equity_ticker"
                )

                # Create detailed equity dataframe
                equity_detail = pd.DataFrame({
                    'ticker': df_eq['equity_ticker'],
                    'quantity_t1': df_eq['quantity_t1'],
                    'quantity_t': df_eq['quantity_t'],
                    'price_t1_raw': df_eq['price_t1'],
                    'price_t_raw': df_eq['price_t'],
                    'price_t1_adj': df_eq_adj.get('price_t1', df_eq['price_t1']),
                    'price_t_adj': df_eq_adj.get('price_t', df_eq['price_t']),
                    'quantity_used': df_eq[qty_col],
                    'gl': df_eq["gl"],
                    'gl_adjusted': df_eq_adj.get("gl_adj", df_eq["gl"])
                })

                detailed_data['equity_details'] = equity_detail

                # Track price adjustments
                price_breaks = self.holdings_price_breaks.get('equity', pd.DataFrame()) if hasattr(self, 'holdings_price_breaks') else pd.DataFrame()
                if not price_breaks.empty:
                    detailed_data['price_adjustments']['equity'] = price_breaks

        # Process option details
        if not opt_today.empty and not opt_prior.empty:
            df_opt = opt_today.merge(
                opt_prior[["optticker", "price", "quantity"]],
                on="optticker", how="inner", suffixes=("_t", "_t1")
            ).dropna(subset=["price_t", "price_t1"])

            qty_col = "quantity_t1" if self.analysis_type == "ex_ante" else "quantity_t"

            # Separate flex and regular options
            is_flex = (
                    df_opt["optticker"].str.contains("SPX|XSP", na=False) &
                    self.fund_name.startswith(("PF", "PD"))
            )

            # Regular options
            regular_opt = df_opt[~is_flex].copy()
            if not regular_opt.empty:
                regular_opt["gl"] = (regular_opt["price_t"] - regular_opt["price_t1"]) * regular_opt[qty_col] * 100

                # Only include options with non-zero G/L
                regular_opt = regular_opt[regular_opt["gl"].abs() > 0.01]

                if not regular_opt.empty:
                    # Apply price adjustments
                    regular_opt_adj = self.apply_small_price_override(
                        regular_opt.copy(), kind="option", key_col="optticker"
                    )

                    option_detail = pd.DataFrame({
                        'ticker': regular_opt['optticker'],
                        'quantity_t1': regular_opt['quantity_t1'],
                        'quantity_t': regular_opt['quantity_t'],
                        'price_t1_raw': regular_opt['price_t1'],
                        'price_t_raw': regular_opt['price_t'],
                        'price_t1_adj': regular_opt_adj.get('price_t1', regular_opt['price_t1']),
                        'price_t_adj': regular_opt_adj.get('price_t', regular_opt['price_t']),
                        'quantity_used': regular_opt[qty_col],
                        'gl': regular_opt["gl"],
                        'gl_adjusted': regular_opt_adj.get("gl_adj", regular_opt["gl"])
                    })
                    detailed_data['option_details'] = option_detail

            # Flex options
            flex_opt = df_opt[is_flex].copy()
            if not flex_opt.empty:
                flex_opt["gl"] = (flex_opt["price_t"] - flex_opt["price_t1"]) * flex_opt[qty_col] * 100

                # Only include flex options with non-zero G/L
                flex_opt = flex_opt[flex_opt["gl"].abs() > 0.01]

                if not flex_opt.empty:
                    flex_opt_adj = self.apply_small_price_override(
                        flex_opt.copy(), kind="option", key_col="optticker"
                    )

                    flex_detail = pd.DataFrame({
                        'ticker': flex_opt['optticker'],
                        'quantity_t1': flex_opt['quantity_t1'],
                        'quantity_t': flex_opt['quantity_t'],
                        'price_t1_raw': flex_opt['price_t1'],
                        'price_t_raw': flex_opt['price_t'],
                        'price_t1_adj': flex_opt_adj.get('price_t1', flex_opt['price_t1']),
                        'price_t_adj': flex_opt_adj.get('price_t', flex_opt['price_t']),
                        'quantity_used': flex_opt[qty_col],
                        'gl': flex_opt["gl"],
                        'gl_adjusted': flex_opt_adj.get("gl_adj", flex_opt["gl"])
                    })
                    detailed_data['flex_details'] = flex_detail

            # Track price adjustments
            price_breaks = self.holdings_price_breaks.get('option', pd.DataFrame()) if hasattr(self, 'holdings_price_breaks') else pd.DataFrame()
            if not price_breaks.empty:
                detailed_data['price_adjustments']['option'] = price_breaks

        # Add NAV components
        detailed_data['nav_components'] = {
            'beg_tna': nav_prior.get("total_net_assets", pd.Series([0])).iloc[0] if not nav_prior.empty else 0,
            'cust_tna': nav_today.get("total_net_assets", pd.Series([0])).iloc[0] if not nav_today.empty else 0,
            'cust_nav': nav_today.get("nav", pd.Series([0])).iloc[0] if not nav_today.empty else 0,
            'shares_outstanding': nav_today.get("shares_outstanding", pd.Series([0])).iloc[0] if not nav_today.empty else 0,
            'expense_ratio': expense_rat,
            'analysis_date': getattr(self, 'analysis_date', None),
            'prior_date': getattr(self, 'prior_date', None)
        }

        return detailed_data

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
                multiplier = 100 if kind == "option" else 1
                df2["gl_adj"] = (df2["price_t"] - df2.get("price_t1", 0)) * df2[qty_col] * multiplier

        return df2

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
        # 1) Load today & prior data
        df_oms = self._set_internal_quantity_column(
            self.fund_data.get('equity_holdings', pd.DataFrame()))
        df_cust = self.fund_data.get('custodian_equity_holdings', pd.DataFrame())
        df_oms1 = self._set_internal_quantity_column(
            self.fund_data.get('t1_equity_holdings', pd.DataFrame()))
        df_cust1 = self.fund_data.get('t1_custodian_equity_holdings', pd.DataFrame())
        df_trades = self.fund_data.get('trades_data', pd.DataFrame())
        df_crrd = self.fund_data.get('cr_rd_data', pd.DataFrame())

        # 2) Normalize tickers
        df_oms, df_cust = normalize_equity_pair(df_oms, df_cust, logger=self.logger)
        df_oms1, df_cust1 = normalize_equity_pair(df_oms1, df_cust1, logger=self.logger)

        # must have equity_ticker
        if 'equity_ticker' not in df_oms.columns or 'equity_ticker' not in df_cust.columns:
            self.results['custodian_equity'] = {
                'raw_recon': pd.DataFrame(),
                'final_recon': pd.DataFrame(),
                'price_discrepancies': pd.DataFrame(),
                'price_discrepancies_T': pd.DataFrame(),
                'price_discrepancies_T1': pd.DataFrame()
            }
            return

        # 3) Merge base tables (today)
        df = pd.merge(df_oms, df_cust,
                      on='equity_ticker', how='outer',
                      suffixes=('_vest', '_cust'), indicator=True)
        df['in_vest'] = df['_merge'] != 'right_only'
        df['in_cust'] = df['_merge'] != 'left_only'
        df.drop(columns=['_merge'], inplace=True)

        # same for T-1
        df1 = pd.merge(df_oms1, df_cust1,
                       on='equity_ticker', how='outer',
                       suffixes=('_vest', '_cust'), indicator=False)

        # 4) trades & corporate actions (only today)
        trade_map = df_trades.set_index('equity_ticker')['qty_sign_adj'] \
            if 'equity_ticker' in df_trades.columns else pd.Series(dtype=object)
        df['qty_sign_adj'] = df['equity_ticker'].map(trade_map).fillna(0)
        cr_map = (df_crrd.set_index('equity_ticker')['cr_rd']
                  if self.is_etf and 'equity_ticker' in df_crrd.columns else pd.Series(dtype=object))
        df['cr_rd'] = df['equity_ticker'].map(cr_map).fillna(0)

        # 5) adjusted shares & base discrepancy
        df['adjusted_cust_shares'] = df['shares_cust'].fillna(0) + df['qty_sign_adj']
        df['final_adjusted_shares'] = df['adjusted_cust_shares'] + df['cr_rd']
        df['final_discrepancy'] = df['quantity'].fillna(0) - df['final_adjusted_shares']

        # 6) Identify mismatches AND create discrepancy_type column FIRST
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

        # breakdown text
        df_issues['breakdown'] = np.where(
            df_issues['discrepancy_type'] == "Quantity Mismatch",
            "Vest=" + df_issues['quantity'].fillna(0).astype(int).astype(str)
            + " | Cust=" + df_issues['shares_cust'].fillna(0).astype(int).astype(str)
            + " | TradesAdj=" + df_issues['qty_sign_adj'].astype(int).astype(str)
            + " | CR/RD=" + df_issues['cr_rd'].astype(int).astype(str),
            "Present in Vest: " + df_issues['in_vest'].map({True: "Yes", False: "No"})
            + " | Present in Cust: " + df_issues['in_cust'].map({True: "Yes", False: "No"})
        )

        # Only keep rows with actual significant discrepancies
        if not df_issues.empty:
            # For quantity mismatches, only keep if difference is significant
            qty_mask = df_issues['discrepancy_type'] == "Quantity Mismatch"
            sig_qty = df_issues[qty_mask & (df_issues['final_discrepancy'].abs() > 0.01)]
            other_issues = df_issues[~qty_mask]
            df_issues = pd.concat([sig_qty, other_issues], ignore_index=True)

        # 7) Price-difference tables for T and T-1
        price_disc_T = pd.DataFrame()
        price_disc_T1 = pd.DataFrame()

        # T: compare today's vest vs custodian prices
        if {'price_vest', 'price_cust'}.issubset(df.columns):
            df['price_diff'] = (df['price_vest'] - df['price_cust']).abs()
            price_disc_T = df.loc[df['price_diff'] > 0.01,
            ['equity_ticker', 'price_vest', 'price_cust', 'price_diff']].copy()

            # override small (<1) today's vest price so final_discrepancy reflects it
            small_T = df['price_diff'].between(0.01, 1)
            df.loc[small_T, 'price_vest'] = df.loc[small_T, 'price_cust']
            df['final_discrepancy'] = df['quantity'].fillna(0) - df['final_adjusted_shares']

        # T-1: compare prior vest vs custodian prices
        if {'price_vest', 'price_cust'}.issubset(df1.columns):
            df1['price_diff'] = (df1['price_vest'] - df1['price_cust']).abs()
            price_disc_T1 = df1.loc[df1['price_diff'] > 0.01,
            ['equity_ticker', 'price_vest', 'price_cust', 'price_diff']].copy()

        # 8) store everything under the expected keys
        # Combine T and T1 price discrepancies for backward compatibility
        price_combined = pd.concat([price_disc_T, price_disc_T1], ignore_index=True).drop_duplicates()

        self.results['custodian_equity'] = ReconciliationResult(
            raw_recon=df.loc[mask_qty & (df['final_discrepancy'].abs() > 0.01)].reset_index(drop=True),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=price_disc_T.reset_index(drop=True),
            price_discrepancies_T1=price_disc_T1.reset_index(drop=True),
            merged_data=df.copy()
        )

        # 9) emit summary rows
        for _, row in df_issues.iterrows():
            dtype = row['discrepancy_type']
            desc = row['breakdown']
            val = row['final_discrepancy'] if dtype == 'Quantity Mismatch' else 'N/A'
            self.add_summary_row(f"Custodian Equity: {dtype}",
                                 row['equity_ticker'], desc, val)

        for _, row in price_disc_T.iterrows():
            self.add_summary_row("Custodian Equity Price (T)",
                                 row['equity_ticker'],
                                 f"{row['price_vest']} vs {row['price_cust']}",
                                 row['price_diff'])

        for _, row in price_disc_T1.iterrows():
            self.add_summary_row("Custodian Equity Price (T-1)",
                                 row['equity_ticker'],
                                 f"{row['price_vest']} vs {row['price_cust']}",
                                 row['price_diff'])

    def reconcile_index_equity(self):
        # Check for special fund first
        if self.fund_name == "DOGG":
            self.results['index_equity'] = {
                'holdings_discrepancies': pd.DataFrame(),
                'significant_diffs': pd.DataFrame()
            }
            return

        # Get data from fund_data dictionary (not self.fund_data)
        df_oms = self.fund_data.get('equity_holdings', pd.DataFrame())
        df_index = self.fund_data.get('index_holdings', pd.DataFrame())

        # Check if data is available
        if df_oms.empty or df_index.empty:
            self.results['index_equity'] = {
                'holdings_discrepancies': pd.DataFrame(),
                'significant_diffs': pd.DataFrame()
            }
            return

        # Merge with outer join to capture all securities
        df = pd.merge(df_oms, df_index, on='equity_ticker', how='outer', suffixes=('_vest', '_index'), indicator=True)

        # Add in_vest and in_index flags
        df['in_vest'] = df['start_wgt'].notnull()
        df['in_index'] = df['weight_index'].notnull()

        # Calculate weight difference for all securities
        df['wgt_diff'] = (df['start_wgt'].fillna(0) - df['weight_index'].fillna(0)).abs()

        # Holdings discrepancies - now includes weight information
        holdings_disc = df[df['in_vest'] != df['in_index']][
            ['equity_ticker', 'in_vest', 'in_index', 'start_wgt', 'weight_index', 'wgt_diff']
        ].copy()

        # Fill NaN weights with 0 for cleaner display
        holdings_disc['start_wgt'] = holdings_disc['start_wgt'].fillna(0)
        holdings_disc['weight_index'] = holdings_disc['weight_index'].fillna(0)

        # Significant weight differences (for securities in both)
        sig = df[df['wgt_diff'].gt(0.001) & df['in_vest'] & df['in_index']][
            ['equity_ticker', 'start_wgt', 'weight_index', 'wgt_diff']
        ].copy()

        self.results['index_equity'] = {
            'holdings_discrepancies': holdings_disc,
            'significant_diffs': sig
        }

        # Add summary rows
        for _, row in holdings_disc.iterrows():
            note = 'Missing in OMS' if not row['in_vest'] else 'Missing in Index'
            weight_info = f"OMS: {row['start_wgt']:.4f}, Index: {row['weight_index']:.4f}"
            self.add_summary_row('Index Equity', row['equity_ticker'], f"{note} - {weight_info}", row['wgt_diff'])

        for _, row in sig.iterrows():
            weight_info = f"OMS: {row['start_wgt']:.4f}, Index: {row['weight_index']:.4f}"
            self.add_summary_row('Index Weight Diff', row['equity_ticker'], f"Weight diff - {weight_info}", row['wgt_diff'])

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
        from utilities.ticker_utils import normalize_option_pair

        # Initialize empty result early for error cases
        empty_result = ReconciliationResult(
            raw_recon=pd.DataFrame(),
            final_recon=pd.DataFrame(),
            price_discrepancies_T=pd.DataFrame(),
            price_discrepancies_T1=pd.DataFrame(),
            merged_data=pd.DataFrame()
        )

        # Step 1: Get data from Fund object
        df_oms = self._current_frame('options', source='vest')
        df_cust = self._current_frame('options', source='custodian')
        df_oms1 = self._previous_frame('options', source='vest')
        df_cust1 = self._previous_frame('options', source='custodian')

        # Step 2: Check if all dataframes are empty
        if df_oms.empty and df_cust.empty:
            self.results['custodian_option'] = empty_result
            return

        # Step 3: Normalize option tickers BEFORE any processing
        df_oms, df_cust = normalize_option_pair(df_oms, df_cust, logger=self.logger)
        df_oms1, df_cust1 = normalize_option_pair(df_oms1, df_cust1, logger=self.logger)

        # Step 4: Verify optticker column exists after normalization
        if not df_oms.empty and 'optticker' not in df_oms.columns:
            self.logger.warning("OMS options missing optticker column after normalization")
            self.results['custodian_option'] = empty_result
            return

        if not df_cust.empty and 'optticker' not in df_cust.columns:
            self.logger.warning("Custodian options missing optticker column after normalization")
            self.results['custodian_option'] = empty_result
            return

        # Step 5: Set up quantity columns for current data
        if not df_oms.empty:
            df_oms['quantity'] = self._get_quantity_column(df_oms)
            # Take absolute value of option prices
            if 'price' in df_oms.columns:
                df_oms['price'] = self._coerce_numeric_series(df_oms['price']).abs()

        if not df_cust.empty:
            # Find appropriate quantity column for custodian
            if 'shares_cust' not in df_cust.columns:
                for candidate in ['quantity', 'contracts', 'position', 'shares', 'qty']:
                    if candidate in df_cust.columns:
                        df_cust['shares_cust'] = df_cust[candidate]
                        break
                else:
                    df_cust['shares_cust'] = 0.0
            df_cust['shares_cust'] = self._coerce_numeric_series(df_cust['shares_cust'])

        # Step 6: Set up quantity columns for previous data
        if not df_oms1.empty:
            df_oms1['quantity'] = self._get_quantity_column(df_oms1)
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

        # Step 7: Calculate option weights for standard options only
        if not df_oms.empty and 'price' in df_oms.columns and 'quantity' in df_oms.columns:
            df_oms['market_value'] = df_oms['quantity'].fillna(0) * df_oms['price'].fillna(0) * 100

            # Identify flex options
            df_oms['is_flex'] = False
            if 'optticker' in df_oms.columns:
                df_oms['is_flex'] = (
                        df_oms['optticker'].str.contains("SPX|XSP", na=False) &
                        (getattr(self.fund, 'is_private_fund', False) or
                         getattr(self.fund, 'is_closed_end_fund', False))
                )

            standard_option_mask = ~df_oms['is_flex']
            total_standard_mv = df_oms.loc[standard_option_mask, 'market_value'].sum()

            df_oms['option_weight'] = 0.0
            if total_standard_mv > 0:
                df_oms.loc[standard_option_mask, 'option_weight'] = (
                        df_oms.loc[standard_option_mask, 'market_value'] / total_standard_mv
                )

        # Step 8: Merge current data
        df = pd.merge(df_oms, df_cust, on='optticker', how='outer', suffixes=('_vest', '_cust'))

        # Step 9: Merge previous data
        df1 = pd.merge(df_oms1, df_cust1, on='optticker', how='outer', suffixes=('_vest', '_cust'))

        # Step 10: Identify discrepancies in current data
        # Ensure quantity columns exist after merge
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

        # Step 11: Calculate price discrepancies for current data
        price_T = pd.DataFrame()
        if {'price_vest', 'price_cust'}.issubset(df.columns):
            df['price_diff'] = (df['price_vest'] - df['price_cust']).abs()
            price_T = df.loc[df['price_diff'] > 0.01,
            ['optticker', 'price_vest', 'price_cust', 'price_diff']].copy()

        # Step 12: Calculate price discrepancies for previous data
        price_T1 = pd.DataFrame()
        if {'price_vest', 'price_cust'}.issubset(df1.columns):
            df1['price_diff'] = (df1['price_vest'] - df1['price_cust']).abs()
            price_T1 = df1.loc[df1['price_diff'] > 0.01,
            ['optticker', 'price_vest', 'price_cust', 'price_diff']].copy()

        # Step 13: Separate regular and flex options in issues
        regular_issues = pd.DataFrame()
        flex_issues = pd.DataFrame()

        if not df_issues.empty and 'optticker' in df_issues.columns:
            df_issues['is_flex'] = (
                    df_issues['optticker'].str.contains("SPX|XSP", na=False) &
                    (getattr(self.fund, 'is_private_fund', False) or
                     getattr(self.fund, 'is_closed_end_fund', False))
            )
            flex_issues = df_issues[df_issues['is_flex']].copy()
            regular_issues = df_issues[~df_issues['is_flex']].copy()

        # Step 14: Add breakdown descriptions
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

        # Step 15: Store results
        self.results['custodian_option'] = ReconciliationResult(
            raw_recon=df[hold_disc_mask | qty_disc_mask].reset_index(drop=True) if not df.empty else pd.DataFrame(),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=price_T.reset_index(drop=True),
            price_discrepancies_T1=price_T1.reset_index(drop=True),
            merged_data=df.copy(),
            regular_options=regular_issues.reset_index(drop=True),
            flex_options=flex_issues.reset_index(drop=True)
        )

        # Step 16: Add summary rows for reporting
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
        df_oms1['_vest_quantity'] = self._get_quantity_column(df_oms1)

        if 'shares_cust' not in df_cust1.columns:
            for candidate in [
                'quantity',
                'shares',
                'share_qty',
                'position',
                'qty',
            ]:
                if candidate in df_cust1.columns:
                    df_cust1['shares_cust'] = df_cust1[candidate]
                    break
            else:
                df_cust1['shares_cust'] = pd.Series(dtype=float)

        df_cust1['shares_cust'] = self._coerce_numeric_series(df_cust1.get('shares_cust', pd.Series(dtype=float)))

        # Merge T-1 data
        df1 = pd.merge(df_oms1, df_cust1, on='equity_ticker', how='outer',
                       suffixes=('_vest', '_cust'), indicator=True)
        df1['in_vest'] = df1['_merge'] != 'right_only'
        df1['in_cust'] = df1['_merge'] != 'left_only'
        df1.drop(columns=['_merge'], inplace=True)

        # Calculate T-1 discrepancies (no trades/corporate actions for T-1)
        # Calculate T-1 discrepancies (no trades/corporate actions for T-1)
        df1['vest_quantity'] = self._coerce_numeric_series(
            df1.get('_vest_quantity')
            if '_vest_quantity' in df1.columns
            else df1.get('quantity_vest', df1.get('quantity', pd.Series(0.0, index=df1.index)))
        )
        df1['shares_cust'] = self._coerce_numeric_series(df1.get('shares_cust', pd.Series(0.0, index=df1.index, dtype=float)))
        df1['discrepancy'] = df1['vest_quantity'] - df1['shares_cust']
        df1['quantity'] = df1['vest_quantity']

        if '_vest_quantity' in df1.columns:
            df1.drop(columns=['_vest_quantity'], inplace=True)

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
        df_oms1 = self.fund_data.get('t1_options_holdings', pd.DataFrame()).copy()
        df_cust1 = self.fund_data.get('t1_custodian_option_holdings', pd.DataFrame()).copy()

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
        df_oms = self.fund_data.get('treasury_holdings', pd.DataFrame()).copy()
        df_cust = self.fund_data.get('custodian_treasury_holdings', pd.DataFrame()).copy()

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
            'trades': self.fund_data.get('equity_trades', pd.DataFrame()).copy(),
            'corporate_actions': self.fund_data.get('cr_rd_data', pd.DataFrame()).copy(),
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

    def _check_sg_equity(self) -> bool:
        """Check if this fund has SG equity data"""
        sg_equity = self.fund_data.get('sg_equity', pd.DataFrame()) if isinstance(self.fund_data, dict) else pd.DataFrame()
        return not sg_equity.empty


    def reconcile_sg_option(self):
        """Reconcile SocGen options (for ex-ante/ex-post only)"""
        df_oms = self.fund_data.get('vest_option', pd.DataFrame())
        df_sg = self.fund_data.get('sg_option', pd.DataFrame())

        if df_oms.empty or df_sg.empty:
            self.results['sg_option'] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame()
            )
            return

        # Normalize option tickers if needed
        df_oms, df_sg = normalize_option_pair(df_oms, df_sg, logger=self.logger)

        # Merge data
        df = pd.merge(df_oms, df_sg, on='optticker', how='outer',
                      suffixes=('_vest', '_sg'))

        # Identify discrepancies
        df['in_vest'] = df['quantity_vest'].notnull()
        df['in_sg'] = df['quantity_sg'].notnull()
        df['quantity_diff'] = df['quantity_vest'].fillna(0) - df['quantity_sg'].fillna(0)

        # Filter for issues
        mask_missing = df['in_vest'] != df['in_sg']
        mask_qty = df['in_vest'] & df['in_sg'] & (df['quantity_diff'].abs() > 0.01)
        df_issues = df.loc[mask_missing | mask_qty].copy()

        # Calculate price discrepancies
        price_disc = pd.DataFrame()
        if {'price_vest', 'price_sg'}.issubset(df.columns):
            df['price_diff'] = (df['price_vest'] - df['price_sg']).abs()
            price_disc = df.loc[df['price_diff'] > 0.01,
            ['optticker', 'price_vest', 'price_sg', 'price_diff']]

        self.results['sg_option'] = ReconciliationResult(
            raw_recon=df.loc[mask_qty].reset_index(drop=True),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=price_disc.reset_index(drop=True),
            price_discrepancies_T1=pd.DataFrame(),
            merged_data=df.copy()
        )

        # Add summary rows
        for _, row in df_issues.iterrows():
            self.add_summary_row("SG Option Discrepancy",
                                 row['optticker'],
                                 f"Vest: {row.get('quantity_vest', 0)}, SG: {row.get('quantity_sg', 0)}",
                                 row['quantity_diff'])


    def reconcile_sg_equity(self):
        """Reconcile SocGen equity (for ex-ante/ex-post only)"""
        df_oms = self.fund_data.get('vest_equity', pd.DataFrame())
        df_sg = self.fund_data.get('sg_equity', pd.DataFrame())

        if df_oms.empty or df_sg.empty:
            self.results['sg_equity'] = ReconciliationResult(
                raw_recon=pd.DataFrame(),
                final_recon=pd.DataFrame(),
                price_discrepancies_T=pd.DataFrame(),
                price_discrepancies_T1=pd.DataFrame(),
                merged_data=pd.DataFrame()
            )
            return

        # Normalize equity tickers
        df_oms, df_sg = normalize_equity_pair(df_oms, df_sg, logger=self.logger)

        # Merge data
        df = pd.merge(df_oms, df_sg, on='equity_ticker', how='outer',
                      suffixes=('_vest', '_sg'))

        # Identify discrepancies
        df['in_vest'] = df['quantity_vest'].notnull()
        df['in_sg'] = df['quantity_sg'].notnull()
        df['quantity_diff'] = df['quantity_vest'].fillna(0) - df['quantity_sg'].fillna(0)

        # Filter for issues
        mask_missing = df['in_vest'] != df['in_sg']
        mask_qty = df['in_vest'] & df['in_sg'] & (df['quantity_diff'].abs() > 0.01)
        df_issues = df.loc[mask_missing | mask_qty].copy()

        # Calculate price discrepancies
        price_disc = pd.DataFrame()
        if {'price_vest', 'price_sg'}.issubset(df.columns):
            df['price_diff'] = (df['price_vest'] - df['price_sg']).abs()
            price_disc = df.loc[df['price_diff'] > 0.01,
            ['equity_ticker', 'price_vest', 'price_sg', 'price_diff']]

        self.results['sg_equity'] = ReconciliationResult(
            raw_recon=df.loc[mask_qty].reset_index(drop=True),
            final_recon=df_issues.reset_index(drop=True),
            price_discrepancies_T=price_disc.reset_index(drop=True),
            price_discrepancies_T1=pd.DataFrame(),
            merged_data=df.copy()
        )

        # Add summary rows
        for _, row in df_issues.iterrows():
            self.add_summary_row("SG Equity Discrepancy",
                                 row['equity_ticker'],
                                 f"Vest: {row.get('quantity_vest', 0)}, SG: {row.get('quantity_sg', 0)}",
                                 row['quantity_diff'])