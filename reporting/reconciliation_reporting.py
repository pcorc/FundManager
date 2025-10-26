
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from config.fund_classifications import PRIVATE_FUNDS, CLOSED_END_FUNDS


@dataclass
class ReconResult:
    """Standardized result structure for reconciliation processing"""
    fund: str
    date_str: str
    recon_type: str
    holdings_df: Optional[pd.DataFrame] = None
    price_df_t: Optional[pd.DataFrame] = None
    price_df_t1: Optional[pd.DataFrame] = None

    def to_flat_dict(self) -> Dict[Tuple[str, str, str], pd.DataFrame]:
        """Convert to flattened dictionary format"""
        flat = {}

        if self.holdings_df is not None and not self.holdings_df.empty:
            key = (self.fund, self.date_str, self.recon_type)
            flat[key] = self.holdings_df

        if self.price_df_t is not None and not self.price_df_t.empty:
            key = (self.fund, self.date_str, f"{self.recon_type}_price_T")
            flat[key] = self.price_df_t

        if self.price_df_t1 is not None and not self.price_df_t1.empty:
            key = (self.fund, self.date_str, f"{self.recon_type}_price_T-1")
            flat[key] = self.price_df_t1

        return flat

    def has_data(self) -> bool:
        """Check if this result contains any data"""
        return any([
            self.holdings_df is not None and not self.holdings_df.empty,
            self.price_df_t is not None and not self.price_df_t.empty,
            self.price_df_t1 is not None and not self.price_df_t1.empty
        ])


class ReconciliationReport:
    """
    Generates Excel reports for reconciliation results.
    Modular design with separate handlers for each reconciliation type.
    """

    def __init__(self, reconciliation_results, recon_summary, date, file_path_excel):
        self.reconciliation_results = reconciliation_results
        self.recon_summary = recon_summary or []
        self.date = str(date)
        self.output_path = Path(file_path_excel)
        self.flattened_results = self._filter_and_flatten_results()
        self._export_to_excel()

    # ==================== FLATTENING COORDINATOR ====================

    def _filter_and_flatten_results(self) -> Dict[Tuple[str, str, str], pd.DataFrame]:
        """
        Main coordinator - delegates to specific handlers for each reconciliation type.
        Returns flattened dictionary of results.
        """
        flat = {}

        for date_str, fund_data in self.reconciliation_results.items():
            for fund, recon_dict in fund_data.items():
                for recon_type, subresults in recon_dict.items():
                    # Get appropriate handler
                    handler = self._get_handler(recon_type)

                    if handler:
                        result = handler(fund, date_str, recon_type, subresults)
                        if result and result.has_data():
                            flat.update(result.to_flat_dict())

                    # Special handling for custodian_option (T only) FLEX/regular breakdown
                    if recon_type == 'custodian_option':
                        # Add regular options
                        regular_df = subresults.get('regular_options', pd.DataFrame())
                        if not regular_df.empty:
                            regular_prepared = self._prepare_holdings_df(
                                regular_df.copy(), fund, date_str,
                                'custodian_option_regular', ticker_col='optticker'
                            )
                            key = (fund, date_str, 'custodian_option_regular')
                            flat[key] = regular_prepared

                        # Add FLEX options holdings
                        flex_df = subresults.get('flex_options', pd.DataFrame())
                        if not flex_df.empty:
                            flex_prepared = self._prepare_holdings_df(
                                flex_df.copy(), fund, date_str,
                                'custodian_option_flex', ticker_col='optticker'
                            )
                            key = (fund, date_str, 'custodian_option_flex')
                            flat[key] = flex_prepared

                            # Also add to flex_holdings for summary mapping
                            key2 = (fund, date_str, 'custodian_option_flex_holdings')
                            flat[key2] = flex_prepared

                        # Add FLEX price breaks for T
                        price_t_df = subresults.get('price_discrepancies_T', pd.DataFrame())
                        if not price_t_df.empty and 'is_flex' in price_t_df.columns:
                            flex_price_t = price_t_df[price_t_df['is_flex'] == True].copy()
                            if not flex_price_t.empty:
                                flex_price_prepared = self._prepare_price_df(
                                    flex_price_t, fund, date_str, 'custodian_option_flex_price_T',
                                    ticker_col='optticker'
                                )
                                key = (fund, date_str, 'custodian_option_flex_price_T')
                                flat[key] = flex_price_prepared

                        # Add FLEX price breaks for T-1
                        price_t1_df = subresults.get('price_discrepancies_T1', pd.DataFrame())
                        if not price_t1_df.empty and 'is_flex' in price_t1_df.columns:
                            flex_price_t1 = price_t1_df[price_t1_df['is_flex'] == True].copy()
                            if not flex_price_t1.empty:
                                flex_price_prepared = self._prepare_price_df(
                                    flex_price_t1, fund, date_str, 'custodian_option_flex_price_T-1',
                                    ticker_col='optticker'
                                )
                                key = (fund, date_str, 'custodian_option_flex_price_T-1')
                                flat[key] = flex_price_prepared

                    # Handle T-1 FLEX holdings
                    if recon_type == 'custodian_option_t1':
                        flex_df_t1 = subresults.get('flex_options', pd.DataFrame())
                        if not flex_df_t1.empty:
                            flex_prepared_t1 = self._prepare_holdings_df(
                                flex_df_t1.copy(), fund, date_str,
                                'custodian_option_flex_holdings_t1', ticker_col='optticker'
                            )
                            key = (fund, date_str, 'custodian_option_flex_holdings_t1')
                            flat[key] = flex_prepared_t1

        return flat

    def _get_handler(self, recon_type: str):
        """Route to appropriate handler method based on reconciliation type"""
        handlers = {
            'custodian_equity': self._process_custodian_equity,
            'custodian_equity_t1': self._process_custodian_equity_t1,
            'custodian_option': self._process_custodian_option,
            'custodian_option_t1': self._process_custodian_option_t1,
            'index_equity': self._process_index_equity,
            'sg_option': self._process_sg_option,
            'sg_equity': self._process_sg_equity
        }
        return handlers.get(recon_type)

    # ==================== SPECIFIC RECONCILIATION HANDLERS ====================

    def _process_custodian_equity(self, fund: str, date_str: str,
                                  recon_type: str, subresults: dict) -> Optional[ReconResult]:
        """Process custodian equity reconciliation (T)"""
        final_df = subresults.get('final_recon', pd.DataFrame())

        if final_df.empty:
            return None

        # Prepare holdings dataframe
        holdings_df = self._prepare_holdings_df(
            final_df.copy(), fund, date_str, recon_type, ticker_col='equity_ticker'
        )

        # Process price discrepancies
        price_t, price_t1 = self._process_price_discrepancies(
            subresults, fund, date_str, recon_type, ticker_col='equity_ticker'
        )

        return ReconResult(
            fund=fund,
            date_str=date_str,
            recon_type=recon_type,
            holdings_df=holdings_df,
            price_df_t=price_t,
            price_df_t1=price_t1
        )

    def _process_custodian_equity_t1(self, fund: str, date_str: str,
                                     recon_type: str, subresults: dict) -> Optional[ReconResult]:
        """Process custodian equity reconciliation (T-1)"""
        final_df = subresults.get('final_recon', pd.DataFrame())

        if final_df.empty:
            return None

        holdings_df = self._prepare_holdings_df(
            final_df.copy(), fund, date_str, recon_type, ticker_col='equity_ticker'
        )

        return ReconResult(
            fund=fund,
            date_str=date_str,
            recon_type=recon_type,
            holdings_df=holdings_df
        )

    def _process_custodian_option(self, fund: str, date_str: str,
                                  recon_type: str, subresults: dict) -> Optional[ReconResult]:
        """
        Process custodian option reconciliation (T).
        Filters out FLEX options from main price discrepancies to avoid duplication.
        """
        final_df = subresults.get('final_recon', pd.DataFrame())

        if final_df.empty:
            return None

        holdings_df = self._prepare_holdings_df(
            final_df.copy(), fund, date_str, recon_type, ticker_col='optticker'
        )

        # ✅ UPDATED: Get price discrepancies but filter out FLEX to avoid duplication
        price_t_raw = subresults.get('price_discrepancies_T', pd.DataFrame())
        price_t1_raw = subresults.get('price_discrepancies_T1', pd.DataFrame())

        # Filter out FLEX options from main price discrepancies if is_flex column exists
        if not price_t_raw.empty and 'is_flex' in price_t_raw.columns:
            price_t_filtered = price_t_raw[price_t_raw['is_flex'] == False].copy()
        else:
            price_t_filtered = price_t_raw

        if not price_t1_raw.empty and 'is_flex' in price_t1_raw.columns:
            price_t1_filtered = price_t1_raw[price_t1_raw['is_flex'] == False].copy()
        else:
            price_t1_filtered = price_t1_raw

        # Prepare the filtered price DataFrames
        price_t = self._prepare_price_df(
            price_t_filtered, fund, date_str, recon_type, ticker_col='optticker'
        ) if not price_t_filtered.empty else None

        price_t1 = self._prepare_price_df(
            price_t1_filtered, fund, date_str, recon_type, ticker_col='optticker'
        ) if not price_t1_filtered.empty else None

        return ReconResult(
            fund=fund,
            date_str=date_str,
            recon_type=recon_type,
            holdings_df=holdings_df,
            price_df_t=price_t,
            price_df_t1=price_t1
        )

    def _process_custodian_option_t1(self, fund: str, date_str: str,
                                     recon_type: str, subresults: dict) -> Optional[ReconResult]:
        """Process custodian option reconciliation (T-1)"""
        final_df = subresults.get('final_recon', pd.DataFrame())

        if final_df.empty:
            return None

        holdings_df = self._prepare_holdings_df(
            final_df.copy(), fund, date_str, recon_type, ticker_col='optticker'
        )

        return ReconResult(
            fund=fund,
            date_str=date_str,
            recon_type=recon_type,
            holdings_df=holdings_df
        )

    def _process_index_equity(self, fund: str, date_str: str,
                              recon_type: str, subresults: dict) -> Optional[ReconResult]:
        """Process index equity reconciliation"""
        holdings_df = subresults.get('holdings_discrepancies', pd.DataFrame())

        if holdings_df.empty:
            return None

        # Prepare holdings
        holdings_df = holdings_df.copy()
        holdings_df['FUND'] = fund
        holdings_df['RECON_TYPE'] = recon_type
        holdings_df['DATE'] = date_str
        holdings_df['discrepancy_type'] = 'Holdings Mismatch'

        # Select relevant columns
        cols_to_keep = ['FUND', 'RECON_TYPE', 'DATE', 'equity_ticker',
                        'discrepancy_type', 'in_vest', 'in_index']
        cols_to_keep = [col for col in cols_to_keep if col in holdings_df.columns]
        holdings_df = holdings_df[cols_to_keep] if cols_to_keep else holdings_df

        # Process price discrepancies (index uses different column names)
        price_t, price_t1 = self._process_price_discrepancies(
            subresults, fund, date_str, recon_type,
            ticker_col='equity_ticker',
            price_cust_col='price_index'  # Index uses price_index instead of price_cust
        )

        return ReconResult(
            fund=fund,
            date_str=date_str,
            recon_type=recon_type,
            holdings_df=holdings_df,
            price_df_t=price_t,
            price_df_t1=price_t1
        )

    def _process_sg_option(self, fund: str, date_str: str,
                           recon_type: str, subresults: dict) -> Optional[ReconResult]:
        """Process SG option reconciliation"""
        final_df = subresults.get('final_recon', pd.DataFrame())

        if final_df.empty:
            return None

        holdings_df = self._prepare_holdings_df(
            final_df.copy(), fund, date_str, 'sg_option', ticker_col='optticker'
        )

        # SG has single price_discrepancies (not split by T/T-1)
        price_df = subresults.get('price_discrepancies', pd.DataFrame())
        price_t = self._prepare_price_df(price_df, fund, date_str, recon_type,
                                         ticker_col='optticker') if not price_df.empty else None

        return ReconResult(
            fund=fund,
            date_str=date_str,
            recon_type='sg_option',  # Standardize name
            holdings_df=holdings_df,
            price_df_t=price_t
        )

    def _process_sg_equity(self, fund: str, date_str: str,
                           recon_type: str, subresults: dict) -> Optional[ReconResult]:
        """Process SG equity reconciliation"""
        final_df = subresults.get('final_recon', pd.DataFrame())

        if final_df.empty:
            return None

        holdings_df = self._prepare_holdings_df(
            final_df.copy(), fund, date_str, recon_type, ticker_col='equity_ticker'
        )

        # SG has single price_discrepancies (not split by T/T-1)
        price_df = subresults.get('price_discrepancies', pd.DataFrame())
        price_t = self._prepare_price_df(price_df, fund, date_str, recon_type,
                                         ticker_col='equity_ticker') if not price_df.empty else None

        return ReconResult(
            fund=fund,
            date_str=date_str,
            recon_type=recon_type,
            holdings_df=holdings_df,
            price_df_t=price_t
        )

    # ==================== HELPER METHODS ====================

    def _prepare_holdings_df(self, df: pd.DataFrame, fund: str, date_str: str,
                             recon_type: str, ticker_col: str) -> pd.DataFrame:
        """
        Prepare holdings DataFrame with standard columns.

        Args:
            df: Raw holdings DataFrame
            fund: Fund name
            date_str: Date string
            recon_type: Reconciliation type
            ticker_col: Name of ticker column (equity_ticker or optticker)

        Returns:
            Prepared DataFrame with standard columns
        """
        df['FUND'] = fund
        df['RECON_TYPE'] = recon_type
        df['DATE'] = date_str

        # Define standard columns to keep
        base_cols = ['FUND', 'RECON_TYPE', 'DATE', ticker_col, 'discrepancy_type']

        # Add reconciliation-specific columns if they exist
        optional_cols = ['quantity', 'shares_cust', 'final_discrepancy',
                         'trade_discrepancy', 'in_vest', 'in_cust', 'in_index']

        cols_to_keep = base_cols + [col for col in optional_cols if col in df.columns]
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]

        return df[cols_to_keep] if cols_to_keep else df

    def _process_price_discrepancies(self, subresults: dict, fund: str, date_str: str,
                                     recon_type: str, ticker_col: str,
                                     price_cust_col: str = 'price_cust') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Process price discrepancies for both T and T-1.

        Args:
            subresults: Dictionary containing price_discrepancies_T and price_discrepancies_T1
            fund: Fund name
            date_str: Date string
            recon_type: Reconciliation type
            ticker_col: Name of ticker column
            price_cust_col: Name of custodian/index price column (default: 'price_cust')

        Returns:
            Tuple of (price_df_t, price_df_t1)
        """
        price_t = None
        price_t1 = None

        # Process T price discrepancies
        price_disc_t = subresults.get('price_discrepancies_T', pd.DataFrame())
        if not price_disc_t.empty:
            price_t = self._prepare_price_df(
                price_disc_t, fund, date_str, recon_type, ticker_col, price_cust_col
            )

        # Process T-1 price discrepancies
        price_disc_t1 = subresults.get('price_discrepancies_T1', pd.DataFrame())
        if not price_disc_t1.empty:
            price_t1 = self._prepare_price_df(
                price_disc_t1, fund, date_str, recon_type, ticker_col, price_cust_col
            )

        return price_t, price_t1

    def _prepare_price_df(self, df: pd.DataFrame, fund: str, date_str: str,
                          recon_type: str, ticker_col: str,
                          price_cust_col: str = 'price_cust') -> pd.DataFrame:
        """
        Prepare price discrepancy DataFrame with standard columns.

        Args:
            df: Raw price DataFrame
            fund: Fund name
            date_str: Date string
            recon_type: Reconciliation type
            ticker_col: Name of ticker column
            price_cust_col: Name of custodian/index price column

        Returns:
            Prepared DataFrame
        """
        df = df.copy()
        df['FUND'] = fund
        df['RECON_TYPE'] = recon_type
        df['DATE'] = date_str

        # Rename ticker column to TICKER for consistency
        if ticker_col in df.columns:
            df = df.rename(columns={ticker_col: 'TICKER'})

        # Standard columns to keep
        cols = ['FUND', 'RECON_TYPE', 'DATE', 'TICKER', 'price_vest', price_cust_col, 'price_diff']
        cols = [c for c in cols if c in df.columns]

        return df[cols] if cols else df

    # ==================== EXCEL EXPORT METHODS ====================

    def _export_to_excel(self):
        """Main export coordinator - delegates to specific tab creators with enhanced formatting"""
        os.makedirs(self.output_path.parent, exist_ok=True)

        with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
            if not self.flattened_results:
                self._create_no_breaks_message(writer)
            else:
                self._create_summary_tab(writer)
                self._create_detailed_breaks_tab(writer)
                self._create_price_comparison_tab(writer)
                self._create_system_specific_tabs(writer)

                # ✅ NEW: Apply formatting to price comparison tab
                self._format_price_comparison_tab(writer)

            # Ensure first sheet is visible
            if writer.book.sheetnames:
                writer.book[writer.book.sheetnames[0]].sheet_state = 'visible'

    def _create_no_breaks_message(self, writer):
        """Create simple message when no breaks found"""
        pd.DataFrame(["All reconciliations passed - no breaks found"]).to_excel(
            writer, sheet_name="RECONCILIATION BREAKS SUMMARY", index=False)

    def _create_summary_tab(self, writer):
        """
        Create the RECONCILIATION BREAKS SUMMARY tab with transposed structure (funds as columns).
        Includes notes about top 5 weighting for private/closed-end funds.
        """
        funds = sorted(self._get_unique_funds())

        # Build summary data for each fund
        fund_data = {}
        for fund in funds:
            row = self._build_summary_row(fund)
            # Remove Fund and Date from the row data
            row.pop('Fund', None)
            row.pop('Date', None)
            fund_data[fund] = row

        # Create transposed DataFrame
        summary_df = pd.DataFrame(fund_data)

        # Add a row for the date at the top
        date_row = pd.DataFrame({fund: [self.date] for fund in funds}, index=['Date'])
        summary_df = pd.concat([date_row, summary_df])

        # Rename index to "Reconciliation Type"
        summary_df.index.name = 'Reconciliation Type'

        # Write to Excel with index (reconciliation types as first column)
        summary_df.to_excel(writer, sheet_name="RECONCILIATION BREAKS SUMMARY", index=True, startrow=0)

        # ✅ NEW: Add note about option weighting for private/closed-end funds
        worksheet = writer.sheets["RECONCILIATION BREAKS SUMMARY"]

        # Check if any funds are private or closed-end
        from config.fund_classifications import PRIVATE_FUNDS, CLOSED_END_FUNDS
        has_private_or_closed = any(f in PRIVATE_FUNDS or f in CLOSED_END_FUNDS for f in funds)

        if has_private_or_closed:
            # Add note below the table
            note_row = len(summary_df) + 3  # 3 rows below the table
            worksheet.cell(row=note_row, column=1).value = "Note: For private/closed-end funds, standard option price breaks show top 5 by weight. All FLEX option breaks are shown."
            worksheet.cell(row=note_row, column=1).font = worksheet.cell(row=note_row, column=1).font.copy(italic=True, size=9)

        # Autofit first column (Reconciliation Type)
        worksheet.column_dimensions['A'].width = 45

    def _build_summary_row(self, fund: str) -> dict:
        """Build a single summary row for a fund"""
        row = {'Fund': fund, 'Date': self.date}

        # Initialize all columns to 0
        columns = [
            'custodian_equity_holdings_breaks_T',
            'custodian_equity_holdings_breaks_T_1',
            'custodian_equity_price_breaks_T',
            'custodian_equity_price_breaks_T_1',
            'custodian_option_holdings_breaks_T',
            'custodian_option_holdings_breaks_T_1',
            'custodian_option_regular_breaks_T',
            'custodian_option_flex_breaks_T',
            'custodian_option_price_breaks_T',
            'custodian_option_price_breaks_T_1',
            'flex_option_holdings_breaks_T',  # NEW
            'flex_option_holdings_breaks_T_1',  # NEW
            'flex_option_price_breaks_T',  # NEW
            'flex_option_price_breaks_T_1',  # NEW
            'index_equity_holdings_breaks',
            'index_equity_price_breaks_T',
            'index_equity_price_breaks_T_1',
            'sg_option_breaks',
            'sg_equity_breaks'
        ]

        for col in columns:
            row[col] = 0

        # Count breaks by type
        for (f, d, r), df in self.flattened_results.items():
            if f == fund:
                row = self._update_summary_counts(row, r, len(df))

        return row

    def _update_summary_counts(self, row: dict, recon_type: str, count: int) -> dict:
        """Update summary row counts based on reconciliation type"""
        mapping = {
            'custodian_equity': 'custodian_equity_holdings_breaks_T',
            'custodian_equity_t1': 'custodian_equity_holdings_breaks_T_1',
            'custodian_equity_price_T': 'custodian_equity_price_breaks_T',
            'custodian_equity_price_T-1': 'custodian_equity_price_breaks_T_1',
            'custodian_option': 'custodian_option_holdings_breaks_T',
            'custodian_option_t1': 'custodian_option_holdings_breaks_T_1',
            'custodian_option_regular': 'custodian_option_regular_breaks_T',
            'custodian_option_flex': 'custodian_option_flex_breaks_T',
            'custodian_option_price_T': 'custodian_option_price_breaks_T',
            'custodian_option_price_T-1': 'custodian_option_price_breaks_T_1',
            'custodian_option_flex_holdings': 'flex_option_holdings_breaks_T',  # NEW
            'custodian_option_flex_holdings_t1': 'flex_option_holdings_breaks_T_1',  # NEW
            'custodian_option_flex_price_T': 'flex_option_price_breaks_T',  # NEW
            'custodian_option_flex_price_T-1': 'flex_option_price_breaks_T_1',  # NEW
            'index_equity': 'index_equity_holdings_breaks',
            'index_equity_price_T': 'index_equity_price_breaks_T',
            'index_equity_price_T-1': 'index_equity_price_breaks_T_1',
            'sg_option': 'sg_option_breaks',
            'sg_equity': 'sg_equity_breaks'
        }

        col_name = mapping.get(recon_type)
        if col_name:
            row[col_name] = count

        return row

    def _create_detailed_breaks_tab(self, writer):
        """Create DETAILED BREAKS BY SECURITY tab with Asset Type column"""
        all_breaks = []

        for (fund, date_str, recon_type), df in self.flattened_results.items():
            if '_price_' not in recon_type:  # Exclude price-only tabs
                df_copy = df.copy()
                # Add Asset Type based on recon_type
                df_copy['Asset Type'] = self._determine_asset_type(recon_type, df_copy)
                all_breaks.append(df_copy)

        if all_breaks:
            detailed_breaks = pd.concat(all_breaks, ignore_index=True)
            detailed_breaks = self._standardize_ticker_column(detailed_breaks)

            # Reorder columns to put Asset Type in column D (after FUND, RECON_TYPE, DATE, TICKER)
            cols = list(detailed_breaks.columns)
            if 'Asset Type' in cols:
                cols.remove('Asset Type')
                # Insert after TICKER (which should be in position 3 after FUND, RECON_TYPE, DATE)
                ticker_idx = next((i for i, c in enumerate(cols) if c == 'TICKER'), 3)
                cols.insert(ticker_idx + 1, 'Asset Type')
                detailed_breaks = detailed_breaks[cols]

            detailed_breaks.to_excel(writer, sheet_name="DETAILED BREAKS BY SECURITY", index=False)

    def _create_price_comparison_tab(self, writer):
        """Create comprehensive price comparison showing Fund/Custodian/Index prices side-by-side"""
        price_comparison = []

        # Collect all price discrepancies
        for (fund, date_str, recon_type), df in self.flattened_results.items():
            if '_price_' in recon_type:
                df_copy = df.copy()
                df_copy['FUND'] = fund
                df_copy['DATE'] = date_str

                # Extract source and period
                df_copy['SOURCE'] = self._extract_source_from_recon_type(recon_type)
                df_copy['PERIOD'] = self._extract_period_from_recon_type(recon_type)

                price_comparison.append(df_copy)

        if price_comparison:
            all_prices = pd.concat(price_comparison, ignore_index=True)
            comparison_df = self._build_price_comparison_df(all_prices)
            comparison_df.to_excel(writer, sheet_name="COMPREHENSIVE PRICE COMPARISON", index=False)

    def _determine_asset_type(self, recon_type: str, df: pd.DataFrame) -> str:
        """
        Determine asset type based on recon_type and DataFrame characteristics.

        Args:
            recon_type: Reconciliation type string
            df: DataFrame to analyze

        Returns:
            'Equity', 'Option', or 'FLEX'
        """
        # Check for FLEX first
        if 'flex' in recon_type.lower():
            return 'FLEX'

        # Check if is_flex column exists and is True
        if 'is_flex' in df.columns and df['is_flex'].any():
            return 'FLEX'

        # Check recon_type for equity
        if 'equity' in recon_type.lower():
            return 'Equity'

        # Check recon_type for option
        if 'option' in recon_type.lower():
            return 'Option'

        # Check for ticker columns as fallback
        if 'equity_ticker' in df.columns or 'eqyticker' in df.columns:
            return 'Equity'

        if 'optticker' in df.columns or 'occ_symbol' in df.columns:
            # Check if any rows have SPX/XSP (FLEX indicators)
            ticker_col = 'optticker' if 'optticker' in df.columns else 'occ_symbol'
            if df[ticker_col].str.contains('SPX|XSP', na=False).any():
                return 'FLEX'
            return 'Option'

        # Default
        return 'Unknown'

    def _build_price_comparison_df(self, all_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive price comparison DataFrame with Asset Type, Weight, and % Difference.
        For options, shows the option weight and percentage difference.
        Properly identifies FLEX options.
        """
        comparison_rows = []

        # Use TICKER column (already renamed in _prepare_price_df)
        ticker_col = 'TICKER' if 'TICKER' in all_prices.columns else 'equity_ticker'

        for (fund, ticker, period), group in all_prices.groupby(['FUND', ticker_col, 'PERIOD']):
            row = {
                'Fund': fund,
                'Ticker': ticker,
                'Period': period,
                'Fund_Price': None,
                'Custodian_Price': None,
                'Index_Price': None,
                'Price_Diff': None,
                'Asset Type': 'Unknown',
                'Option_Weight': None,
                'Price_Pct_Diff': None
            }

            # ✅ UPDATED: Check for FLEX first by examining the group
            is_flex_option = False
            for _, price_row in group.iterrows():
                # Check if is_flex column exists and is True
                if 'is_flex' in group.columns:
                    if price_row.get('is_flex', False) == True:
                        is_flex_option = True
                        break
                # Also check recon_type for flex indicator
                recon_type = price_row.get('RECON_TYPE', '')
                if 'flex' in recon_type.lower():
                    is_flex_option = True
                    break

            # Set asset type based on FLEX detection
            if is_flex_option:
                row['Asset Type'] = 'FLEX'

            # Process each row in the group
            for _, price_row in group.iterrows():
                source = price_row.get('SOURCE', '')
                recon_type = price_row.get('RECON_TYPE', '')
                fund_price = price_row.get('price_vest', None)
                price_diff = price_row.get('price_diff', None)

                # Capture option weight and % difference
                if 'option_weight' in price_row.index:
                    row['Option_Weight'] = price_row.get('option_weight', None)
                if 'price_pct_diff' in price_row.index:
                    row['Price_Pct_Diff'] = price_row.get('price_pct_diff', None)

                # Determine asset type if not already set to FLEX
                if row['Asset Type'] == 'Unknown':
                    if 'equity' in source or 'equity' in recon_type:
                        row['Asset Type'] = 'Equity'
                    elif 'option' in source or 'option' in recon_type:
                        row['Asset Type'] = 'Option'

                if 'custodian' in source:
                    row['Fund_Price'] = fund_price
                    row['Custodian_Price'] = price_row.get('price_cust', None)
                    row['Price_Diff'] = price_diff
                elif 'index' in source:
                    row['Fund_Price'] = fund_price
                    row['Index_Price'] = price_row.get('price_index', price_row.get('price_cust', None))
                    row['Price_Diff'] = price_diff

            # Calculate max difference if not already set
            if row['Price_Diff'] is None:
                prices = [p for p in [row['Fund_Price'], row['Custodian_Price'], row['Index_Price']] if p is not None]
                row['Price_Diff'] = max(prices) - min(prices) if len(prices) > 1 else 0

            comparison_rows.append(row)

        comparison_df = pd.DataFrame(comparison_rows)

        # Reorder columns to include Weight and % Diff
        cols = ['Fund', 'Asset Type', 'Ticker', 'Period']

        # Add weight column for options only
        if 'Option_Weight' in comparison_df.columns:
            # Format weight as percentage
            comparison_df['Option_Weight_Pct'] = comparison_df['Option_Weight'].apply(
                lambda x: f"{x * 100:.2f}%" if pd.notna(x) and x > 0 else ""
            )
            cols.append('Option_Weight_Pct')

        # Add remaining price columns
        cols.extend(['Fund_Price', 'Custodian_Price', 'Index_Price', 'Price_Diff'])

        # Add % difference for options
        if 'Price_Pct_Diff' in comparison_df.columns:
            comparison_df['Price_Pct_Diff_Formatted'] = comparison_df['Price_Pct_Diff'].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else ""
            )
            cols.append('Price_Pct_Diff_Formatted')

        # Keep only columns that exist
        cols = [c for c in cols if c in comparison_df.columns]
        comparison_df = comparison_df[cols]

        # Rename for clarity
        rename_dict = {
            'Option_Weight_Pct': 'Option Weight',
            'Price_Pct_Diff_Formatted': '% Difference'
        }
        comparison_df = comparison_df.rename(columns=rename_dict)

        # Sort: Fund, Asset Type, Period, then by weight (descending) for options
        sort_cols = ['Fund', 'Asset Type', 'Period']
        sort_ascending = [True, True, True]

        # For options, sort by weight descending if available
        if 'Option Weight' in comparison_df.columns:
            # Convert back to numeric for sorting
            comparison_df['_weight_sort'] = comparison_df['Option Weight'].str.rstrip('%').replace('', '0').astype(float)
            sort_cols.append('_weight_sort')
            sort_ascending.append(False)
            comparison_df = comparison_df.sort_values(sort_cols, ascending=sort_ascending)
            comparison_df = comparison_df.drop(columns=['_weight_sort'])
        else:
            comparison_df = comparison_df.sort_values(sort_cols, ascending=sort_ascending)

        return comparison_df

    def _create_system_specific_tabs(self, writer):
        """Create INDEX_RECONCILIATION and SG_RECONCILIATION tabs"""
        system_tabs = {
            "INDEX_RECONCILIATION": ["index_equity"],
            "SG_RECONCILIATION": ["sg_equity", "sg_option"]
        }

        for tab_name, recon_types_list in system_tabs.items():
            filtered_breaks = []

            for (fund, date_str, recon_type), df in self.flattened_results.items():
                if recon_type in recon_types_list:
                    filtered_breaks.append(df)

            if filtered_breaks:
                rt_breaks = pd.concat(filtered_breaks, ignore_index=True)
                rt_breaks = self._standardize_ticker_column(rt_breaks)
                rt_breaks.to_excel(writer, sheet_name=tab_name, index=False)

    def _get_unique_funds(self) -> set:
        """Extract unique fund names from flattened results"""
        return {fund for (fund, _, _) in self.flattened_results.keys()}

    def _standardize_ticker_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename various ticker columns to standard TICKER"""
        ticker_cols = [c for c in df.columns if c in
                       ['equity_ticker', 'optticker', 'norm_ticker', 'eqyticker', 'occ_symbol']]
        if ticker_cols:
            df = df.rename(columns={ticker_cols[0]: 'TICKER'})
        return df

    def _extract_source_from_recon_type(self, recon_type: str) -> str:
        """Extract source system from reconciliation type"""
        if 'custodian_equity' in recon_type:
            return 'custodian_equity'
        elif 'custodian_option' in recon_type:
            return 'custodian_option'
        elif 'index_equity' in recon_type:
            return 'index_equity'
        return 'unknown'

    def _extract_period_from_recon_type(self, recon_type: str) -> str:
        """Extract time period (T or T-1) from reconciliation type"""
        if 'T-1' in recon_type:
            return 'T-1'
        return 'T'


    def _format_price_comparison_tab(self, writer):
        """
        Apply special formatting to the price comparison tab.
        Highlights columns with weight and % difference information.
        """
        if "COMPREHENSIVE PRICE COMPARISON" not in writer.sheets:
            return

        from openpyxl.styles import Font, PatternFill, Alignment

        worksheet = writer.sheets["COMPREHENSIVE PRICE COMPARISON"]

        # Define header fill (light blue)
        header_fill = PatternFill(start_color="D3E4F7", end_color="D3E4F7", fill_type="solid")

        # Define weight column fill (light green)
        weight_fill = PatternFill(start_color="E2F0D9", end_color="E2F0D9", fill_type="solid")

        # Format header row
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Find weight and % diff columns
        weight_col_idx = None
        pct_diff_col_idx = None

        for idx, cell in enumerate(worksheet[1], start=1):
            if cell.value == "Option Weight":
                weight_col_idx = idx
            elif cell.value == "% Difference":
                pct_diff_col_idx = idx

        # Apply special formatting to weight column
        if weight_col_idx:
            col_letter = worksheet.cell(row=1, column=weight_col_idx).column_letter
            for row in range(2, worksheet.max_row + 1):
                cell = worksheet.cell(row=row, column=weight_col_idx)
                if cell.value:  # Only format cells with values
                    cell.fill = weight_fill
                    cell.alignment = Alignment(horizontal="right")

        # Apply special formatting to % diff column
        if pct_diff_col_idx:
            col_letter = worksheet.cell(row=1, column=pct_diff_col_idx).column_letter
            for row in range(2, worksheet.max_row + 1):
                cell = worksheet.cell(row=row, column=pct_diff_col_idx)
                if cell.value:  # Only format cells with values
                    cell.alignment = Alignment(horizontal="right")

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass

            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width


from reporting.base_report_pdf import BaseReportPDF
from reporting.holdings_recon_renderer import HoldingsReconciliationRenderer
import datetime
import pandas as pd
import logging

# Set up logging
logger = logging.getLogger(__name__)



class ReconciliationReportPDF(BaseReportPDF, HoldingsReconciliationRenderer):
    def __init__(self, reconciliation_results=None, recon_summary=None, date=None, file_path_pdf=None,
                 output_path=None):
        """
        Initialize the ReconciliationReportPDF with data and generate the PDF.
        Enhanced to properly handle reconciliation for all asset types.

        Parameters:
        - reconciliation_results: Dict containing reconciliation results by date and fund
        - recon_summary: List of reconciliation summaries
        - date: Date string for the report
        - file_path_pdf: Path where the PDF will be saved (primary)
        - output_path: Alternative path parameter (for backward compatibility)
        """
        # Handle both parameter styles for the output path
        output_file = file_path_pdf or output_path or f"reconciliation_report_{date or datetime.date.today().strftime('%Y-%m-%d')}.pdf"

        # Initialize the base class
        super().__init__(output_file)

        # Store specific attributes
        self.reconciliation_results = reconciliation_results or {}
        self.recon_summary = recon_summary or []
        self.date = str(date) if date else datetime.date.today().strftime("%Y-%m-%d")

        # Log data structure for debugging
        logger.debug(f"ReconciliationReportPDF initialization with data structure:")
        if self.reconciliation_results:
            logger.debug(f"reconciliation_results keys: {list(self.reconciliation_results.keys())}")
        if self.recon_summary:
            logger.debug(f"recon_summary type: {type(self.recon_summary)}, length: {len(self.recon_summary)}")
            if len(self.recon_summary) > 0:
                logger.debug(f"First item type: {type(self.recon_summary[0])}")

        try:
            # Process the data into a flattened format
            self.flattened_results = self._flatten_reconciliation_results()

            # Generate the PDF
            self._generate_pdf()

            logger.info(f"PDF report successfully generated at {self.output_path}")
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
            # Create a simple error report instead
            self._generate_error_report(str(e))

    def _flatten_reconciliation_results(self):
        """
        Flatten the nested reconciliation results structure for easier processing.
        Returns a dict with (fund_name, date_str) as keys and reconciliation data as values.
        """
        flattened = {}

        try:
            for date_str, fund_data in self.reconciliation_results.items():
                for fund_name, recon_dict in fund_data.items():
                    # Create a flattened key
                    key = (fund_name, date_str)

                    # Store all reconciliation types
                    flattened[key] = recon_dict
        except Exception as e:
            logger.error(f"Error flattening reconciliation results: {str(e)}", exc_info=True)

        return flattened

    def _generate_error_report(self, error_message):
        """Generate a simple error report if the main report fails"""
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 10, "Reconciliation Report - ERROR", ln=True, align="C")
        self.pdf.set_font("Arial", "", 12)
        self.pdf.multi_cell(0, 10, f"An error occurred while generating the report: {error_message}")

        try:
            # Output the PDF
            self.output()
        except Exception as e:
            logger.error(f"Failed to save error report: {str(e)}")

    def _generate_pdf(self):
        """Generate the PDF report and save it to the output path"""
        self.pdf.add_page()
        self._add_header("Reconciliation Report", f"Report Date: {self.date}")
        self._add_consolidated_summary()

        # If no results, show a message
        if not self.flattened_results:
            self.pdf.set_font("Arial", "B", 12)
            self.pdf.cell(0, 10, "No reconciliation results available", ln=True, align="C")
        else:
            # Process each fund's reconciliation data
            for (fund_name, date_str), recon_data in sorted(self.flattened_results.items()):
                # Use the mixin's render method
                self.render_fund_holdings_section(fund_name, date_str, recon_data)

        # Save the PDF
        self.output()

    def _print_recon_section(self, recon_type, recon_data, fund_name=None):  # ✅ UPDATED
        """
        Print the detailed section for a single reconciliation type.
        """

        # ————————— INDEX EQUITY —————————
        if recon_type == "index_equity":
            # ... existing code ...
            return

        # ————————— CUSTODIAN EQUITY —————————
        elif recon_type == "custodian_equity":
            self._print_custodian_equity_section(recon_data)
            return

        # ————————— CUSTODIAN OPTION —————————
        elif recon_type == "custodian_option":
            self._print_custodian_option_section(recon_data, fund_name=fund_name, is_flex=False)  # ✅ UPDATED
            return

        # ————————— DEFAULT (fallback) —————————
        else:
            df = recon_data.get("final_recon", pd.DataFrame())

            if df.empty:
                self._draw_two_column_table([("Status", "No reconciliation data available")])
                self.pdf.ln(4)
                return

            counts = self._count_discrepancies(recon_data)
            rows = [(f"Total {k} Discrepancies", v) for k, v in counts.items()]
            if sum(counts.values()) > 0:
                self._draw_two_column_table(rows)
                self._print_recon_details(recon_data, recon_type)
            else:
                self._draw_two_column_table([("Status", "No discrepancies found")])

            self.pdf.ln(4)

    def _count_discrepancies(self, recon_data):
        """Count discrepancies by type in the reconciliation data"""
        counts = {
            "Total": 0,
            "Holdings": 0,
            "Quantity": 0,
            "Weight": 0,
            "Price": 0
        }

        # Check final_recon first
        final_df = recon_data.get('final_recon')
        if isinstance(final_df, pd.DataFrame) and not final_df.empty:
            counts["Total"] = len(final_df)

            # Count by discrepancy_type if available
            if 'discrepancy_type' in final_df.columns:
                for disc_type in final_df['discrepancy_type']:
                    disc_type_lower = str(disc_type).lower()
                    if 'missing' in disc_type_lower or 'holdings' in disc_type_lower:
                        counts["Holdings"] += 1
                    elif 'quantity' in disc_type_lower or 'mismatch' in disc_type_lower:
                        counts["Quantity"] += 1

        # Check price_discrepancies_T and price_discrepancies_T1
        price_T = recon_data.get('price_discrepancies_T')
        price_T1 = recon_data.get('price_discrepancies_T1')

        if isinstance(price_T, pd.DataFrame):
            counts["Price"] += len(price_T)
            counts["Total"] += len(price_T)

        if isinstance(price_T1, pd.DataFrame):
            counts["Price"] += len(price_T1)
            counts["Total"] += len(price_T1)

        # Check holdings_discrepancies
        holdings_df = recon_data.get('holdings_discrepancies')
        if isinstance(holdings_df, pd.DataFrame) and not holdings_df.empty:
            counts["Holdings"] += len(holdings_df)

        # Check significant_diffs (for index_equity)
        weight_df = recon_data.get('significant_diffs')
        if isinstance(weight_df, pd.DataFrame) and not weight_df.empty:
            counts["Weight"] += len(weight_df)

        return counts

    def _print_recon_details(self, recon_data, recon_type=None):
        """Print detailed discrepancy information"""
        self.pdf.set_font("Arial", "B", 10)
        self.pdf.cell(0, 6, "Discrepancy Details:", ln=True)
        self.pdf.set_font("Arial", "", 9)

        # Add details from final_recon if available
        final_df = recon_data.get('final_recon')
        if isinstance(final_df, pd.DataFrame) and not final_df.empty:
            title = f"Final Reconciliation - {recon_type}" if recon_type else "Final Reconciliation"
            self._print_discrepancy_table(final_df, title)

    def _print_discrepancy_table(self, df, title):
        """Print a formatted table of the first few rows of df."""
        if df.empty:
            return

        # Pick ticker column first
        ticker_col = None
        for col in ["equity_ticker", "optticker", "occ_symbol", "norm_ticker", "ticker"]:
            if col in df.columns:
                ticker_col = col
                break

        if not ticker_col:
            return

        # Build display columns
        display_cols = [ticker_col]
        for col in ["discrepancy_type", "shares_cust", "trade_discrepancy",
                    "final_discrepancy", "price_vest", "price_cust", "price_diff"]:
            if col in df.columns:
                display_cols.append(col)

        # Draw header
        self.pdf.set_font("Arial", "B", 8)
        self.pdf.set_fill_color(200, 200, 200)
        col_w = (self.pdf.w - 2 * self.pdf.l_margin) / len(display_cols)
        for col in display_cols:
            self.pdf.cell(col_w, 6, col, border=1, fill=True, align="C")
        self.pdf.ln()

        # Draw rows
        self.pdf.set_font("Arial", "", 8)
        for _, row in df.head(10).iterrows():
            for col in display_cols:
                val = row[col]
                txt = ""
                if isinstance(val, bool):
                    txt = "YES" if val else "NO"
                elif isinstance(val, (int, float)):
                    txt = f"{val:,.2f}"
                else:
                    txt = str(val)
                self.pdf.cell(col_w, 5, txt, border=1, align="R")
            self.pdf.ln()

        if len(df) > 10:
            self.pdf.set_font("Arial", "I", 8)
            self.pdf.cell(0, 5, f"... and {len(df) - 10} more rows", ln=1)

    def _add_consolidated_summary(self):
        """Add a consolidated summary table showing all funds' reconciliation counts"""
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, "Reconciliation Breaks Summary", ln=True)
        self.pdf.ln(2)

        # Collect summary data for all funds
        all_summaries = {}
        for fund_summary in self.recon_summary:
            if isinstance(fund_summary, dict) and 'fund' in fund_summary:
                all_summaries[fund_summary['fund']] = fund_summary.get('summary', {})

        if not all_summaries:
            return

        # Define columns - Holdings Breaks and Price Breaks for each category
        cols = [
            ("Fund", 30),
            ("Index\nHoldings", 18),
            ("Index\nWgt Diff", 18),
            ("Cust Eq\nHold Brk", 18),
            ("Cust Eq\nPrice T", 18),
            ("Cust Eq\nPrice T-1", 18),
            ("Cust Opt\nHold Brk", 18),
            ("Cust Opt\nPrice T", 18),
            ("Cust Opt\nPrice T-1", 18),
        ]

        # Header row with multi-line support
        self.pdf.set_font("Arial", "B", 6)
        self.pdf.set_fill_color(240, 240, 240)

        max_lines = max(len(header.split('\n')) for header, _ in cols)
        start_y = self.pdf.get_y()
        line_height = 3.5

        # Draw headers
        x_pos = self.pdf.l_margin
        for header, width in cols:
            lines = header.split('\n')
            for i, line in enumerate(lines):
                self.pdf.set_xy(x_pos, start_y + (i * line_height))
                self.pdf.cell(width, line_height, line, border=1, fill=True, align="C")
            x_pos += width

        self.pdf.set_y(start_y + (max_lines * line_height))

        # Data rows
        self.pdf.set_font("Arial", "", 7)
        for fund in sorted(all_summaries.keys()):
            summary = all_summaries[fund]
            y_pos = self.pdf.get_y()

            # Fund name
            self.pdf.set_font("Arial", "B", 7)
            self.pdf.set_xy(self.pdf.l_margin, y_pos)
            self.pdf.cell(cols[0][1], 6, fund, border=1, align="C")

            # Get counts for each column
            index_holdings = summary.get('index_equity', {}).get('holdings_discrepancies', 0)
            index_weight = summary.get('index_equity', {}).get('significant_diffs', 0)

            cust_eq_holdings = summary.get('custodian_equity', {}).get('final_recon', 0)
            cust_eq_price_T = summary.get('custodian_equity', {}).get('price_discrepancies_T', 0)
            cust_eq_price_T1 = summary.get('custodian_equity', {}).get('price_discrepancies_T1', 0)

            cust_opt_holdings = summary.get('custodian_option', {}).get('final_recon', 0)
            cust_opt_price_T = summary.get('custodian_option', {}).get('price_discrepancies_T', 0)
            cust_opt_price_T1 = summary.get('custodian_option', {}).get('price_discrepancies_T1', 0)

            values = [
                index_holdings,
                index_weight,
                cust_eq_holdings,
                cust_eq_price_T,
                cust_eq_price_T1,
                cust_opt_holdings,
                cust_opt_price_T,
                cust_opt_price_T1
            ]

            # Print values
            self.pdf.set_font("Arial", "", 7)
            x_pos = self.pdf.l_margin + cols[0][1]
            for val, (_, width) in zip(values, cols[1:]):
                if val > 0:
                    self.pdf.set_fill_color(255, 200, 200)
                else:
                    self.pdf.set_fill_color(255, 255, 255)

                self.pdf.set_xy(x_pos, y_pos)
                self.pdf.cell(width, 6, str(val), border=1, fill=True, align="C")
                x_pos += width

            self.pdf.set_y(y_pos + 6)

        # Add legend
        self.pdf.ln(4)
        self.pdf.set_font("Arial", "I", 7)
        self.pdf.set_text_color(100, 100, 100)
        self.pdf.cell(0, 4, "Red highlight indicates breaks found | Hold Brk = Holdings Breaks", ln=1)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(6)