import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

logger = logging.getLogger(__name__)


class NAVReconciliationReport:
    """
    Generates Excel reports for NAV reconciliation results.
    The report is automatically generated upon instantiation.
    """

    def __init__(self, reconciliation_results, recon_summary, date, file_path_excel):
        """
        Initialize and automatically generate the NAV Reconciliation Report.

        Args:
            reconciliation_results: Dict with structure {date_str: {fund_name: nav_results}}
            recon_summary: List of summary dictionaries
            date: Date string for the report
            file_path_excel: Path where the Excel file will be saved
        """
        self.reconciliation_results = reconciliation_results or {}
        self.recon_summary = recon_summary or []
        self.date = str(date)
        self.output_path = Path(file_path_excel)

        # Summary statistics
        self.stats = {
            "total_funds": 0,
            "funds_within_tolerance_2d": 0,
            "funds_within_tolerance_4d": 0,
            "total_nav_difference": 0,
            "average_pct_difference": 0
        }

        # Style definitions
        self.header_font = Font(bold=True, size=12)
        self.subheader_font = Font(bold=True, size=10)
        self.title_font = Font(bold=True, size=14)
        self.number_format = '#,##0.00'
        self.number_format_4 = '#,##0.0000'
        self.percent_format = '0.00%'

        # Create workbook
        self._create_workbook()

    def _categorize_component(self, component):
        """Categorize components for grouping"""
        component_lower = component.lower()
        if "equity" in component_lower:
            return "Equity"
        elif "option" in component_lower:
            return "Options"
        elif "treasury" in component_lower:
            return "Fixed Income"
        elif "expense" in component_lower or "accrual" in component_lower:
            return "Expenses"
        elif "dividend" in component_lower:
            return "Income"
        else:
            return "Other"

    def _create_workbook(self):
        """Create the Excel workbook with all sheets."""
        wb = Workbook()

        # Remove default sheet
        wb.remove(wb.active)

        # Create summary sheet first
        self._create_summary_sheet(wb)

        # Create detailed sheet for each fund
        for date_str, fund_results in self.reconciliation_results.items():
            for fund_name, nav_data in fund_results.items():
                self._create_fund_sheet(wb, fund_name, nav_data, date_str)

        # Calculate summary statistics
        self._calculate_summary_stats()

        # Save workbook
        wb.save(self.output_path)
        logger.info(f"NAV reconciliation report generated at {self.output_path}")

    def _create_summary_sheet(self, wb):
        """Create summary sheet with all funds."""
        ws = wb.create_sheet("NAV Summary")

        # Title
        ws.merge_cells('A1:K1')
        ws['A1'] = f"NAV Reconciliation Summary - {self.date}"
        ws['A1'].font = self.title_font
        ws['A1'].alignment = Alignment(horizontal='center')

        # Headers
        headers = [
            'Fund', 'Date', 'Expected TNA', 'Custodian TNA', 'TNA Diff ($)',
            'Expected NAV', 'Custodian NAV', 'NAV Diff ($)', 'Shares Outstanding',
            'NAV Good (2-dec)', 'NAV Good (4-dec)'
        ]

        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=header)
            cell.font = self.header_font
            cell.alignment = Alignment(horizontal='center')

        # Data rows
        row_num = 4
        for date_str, fund_results in sorted(self.reconciliation_results.items()):
            for fund_name, nav_data in sorted(fund_results.items()):
                # Extract results - handle both detailed and simple formats
                if isinstance(nav_data, dict):
                    results = nav_data
                else:
                    continue

                ws.cell(row=row_num, column=1, value=fund_name)
                ws.cell(row=row_num, column=2, value=date_str)
                ws.cell(row=row_num, column=3, value=results.get('Expected TNA', 0)).number_format = self.number_format
                ws.cell(row=row_num, column=4, value=results.get('Custodian TNA', 0)).number_format = self.number_format
                ws.cell(row=row_num, column=5, value=results.get('TNA Diff ($)', 0)).number_format = self.number_format
                ws.cell(row=row_num, column=6, value=results.get('Expected NAV', 0)).number_format = self.number_format_4
                ws.cell(row=row_num, column=7, value=results.get('Custodian NAV', 0)).number_format = self.number_format_4
                ws.cell(row=row_num, column=8, value=results.get('NAV Diff ($)', 0)).number_format = self.number_format_4
                ws.cell(row=row_num, column=9, value=results.get('Shares Outstanding', 0)).number_format = '#,##0'

                # Color code the pass/fail columns
                nav_good_2 = results.get('NAV Good (2 Digit)', False)
                nav_good_4 = results.get('NAV Good (4 Digit)', False)

                cell_2dec = ws.cell(row=row_num, column=10, value='PASS' if nav_good_2 else 'FAIL')
                cell_4dec = ws.cell(row=row_num, column=11, value='PASS' if nav_good_4 else 'FAIL')

                if nav_good_2:
                    cell_2dec.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
                else:
                    cell_2dec.fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")

                if nav_good_4:
                    cell_4dec.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
                else:
                    cell_4dec.fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")

                row_num += 1

        # Auto-fit columns
        for col in range(1, 12):
            ws.column_dimensions[get_column_letter(col)].width = 15

    def _create_fund_sheet(self, wb, fund_name, nav_data, date_str):
        """Create detailed sheet for a fund with formulas."""
        ws = wb.create_sheet(fund_name[:31])  # Excel sheet names limited to 31 chars

        # Title
        ws.merge_cells('A1:C1')
        ws['A1'] = f"{fund_name} - NAV Reconciliation - {date_str}"
        ws['A1'].font = self.title_font
        ws['A1'].alignment = Alignment(horizontal='center')

        # SECTION 1: Summary Calculations with formulas
        row = 3
        ws.cell(row=row, column=1, value="NAV CALCULATION SUMMARY").font = self.header_font
        row += 2

        # Beginning TNA
        ws.cell(row=row, column=1, value="Beginning TNA (T-1)")
        ws.cell(row=row, column=3, value=nav_data.get('Beginning TNA', nav_data.get('Beg TNA', 0))).number_format = self.number_format
        beg_tna_row = row
        row += 1

        # Adjusted Beginning TNA
        ws.cell(row=row, column=1, value="Adjusted Beginning TNA")
        ws.cell(row=row, column=3, value=nav_data.get('Adjusted Beginning TNA', nav_data.get('Beginning TNA', 0))).number_format = self.number_format
        adj_beg_tna_row = row
        row += 2

        # Gain/Loss Components
        ws.cell(row=row, column=1, value="Gain/Loss Components:").font = self.subheader_font
        row += 1

        # First add the details sections to get their locations
        # Store the current row to come back after adding details
        summary_components_row = row

        # Calculate all row positions for the summary section
        equity_gl_row = summary_components_row
        equity_gl_adj_row = summary_components_row + 1
        option_gl_row = summary_components_row + 2
        option_gl_adj_row = summary_components_row + 3
        flex_gl_row = summary_components_row + 4
        flex_gl_adj_row = summary_components_row + 5
        treasury_gl_row = summary_components_row + 6
        assign_gl_row = summary_components_row + 7
        other_row = summary_components_row + 8
        expenses_row = summary_components_row + 10  # Skip a row
        dividends_row = summary_components_row + 11
        expected_tna_row = summary_components_row + 13
        cust_tna_row = summary_components_row + 14
        tna_diff_row = summary_components_row + 15
        shares_row = summary_components_row + 17  # Skip a row
        expected_nav_row = summary_components_row + 19
        cust_nav_row = summary_components_row + 20
        nav_diff_row = summary_components_row + 21
        nav_good_2_row = summary_components_row + 23  # Skip a row
        nav_good_4_row = summary_components_row + 24

        # Jump ahead to add detail sections first
        detail_start_row = nav_good_4_row + 5  # Leave space after summary

        # Add detail sections and capture their locations
        equity_detail_info = self._add_equity_details_with_prices(ws, detail_start_row, 1, nav_data)
        option_detail_info = self._add_option_details_with_prices(ws, equity_detail_info['end_row'] + 3, 1, nav_data)
        flex_detail_info = self._add_flex_details_with_prices(ws, option_detail_info['end_row'] + 3, 1, nav_data)
        treasury_detail_info = self._add_treasury_details_with_prices(ws, flex_detail_info['end_row'] + 3, 1, nav_data)

        # Now go back and fill in the summary with proper cell references
        row = summary_components_row

        # Equity G/L
        ws.cell(row=equity_gl_row, column=1, value="  Equity G/L")
        if equity_detail_info['has_data']:
            ws.cell(row=equity_gl_row, column=3, value=f"={equity_detail_info['gl_total_cell']}")
        else:
            ws.cell(row=equity_gl_row, column=3, value=nav_data.get('Equity G/L', 0))
        ws.cell(row=equity_gl_row, column=3).number_format = self.number_format

        # Equity G/L Adjusted
        ws.cell(row=equity_gl_adj_row, column=1, value="  Equity G/L (Adjusted)")
        if equity_detail_info['has_data']:
            ws.cell(row=equity_gl_adj_row, column=3, value=f"={equity_detail_info['gl_adj_total_cell']}")
        else:
            ws.cell(row=equity_gl_adj_row, column=3, value=nav_data.get('Equity G/L Adj', nav_data.get('Equity G/L', 0)))
        ws.cell(row=equity_gl_adj_row, column=3).number_format = self.number_format

        # Option G/L
        ws.cell(row=option_gl_row, column=1, value="  Option G/L")
        if option_detail_info['has_data']:
            ws.cell(row=option_gl_row, column=3, value=f"={option_detail_info['gl_total_cell']}")
        else:
            ws.cell(row=option_gl_row, column=3, value=nav_data.get('Option G/L', 0))
        ws.cell(row=option_gl_row, column=3).number_format = self.number_format

        # Option G/L Adjusted
        ws.cell(row=option_gl_adj_row, column=1, value="  Option G/L (Adjusted)")
        if option_detail_info['has_data']:
            ws.cell(row=option_gl_adj_row, column=3, value=f"={option_detail_info['gl_adj_total_cell']}")
        else:
            ws.cell(row=option_gl_adj_row, column=3, value=nav_data.get('Option G/L Adj', nav_data.get('Option G/L', 0)))
        ws.cell(row=option_gl_adj_row, column=3).number_format = self.number_format

        # Flex Option G/L
        ws.cell(row=flex_gl_row, column=1, value="  Flex Option G/L")
        if flex_detail_info['has_data']:
            ws.cell(row=flex_gl_row, column=3, value=f"={flex_detail_info['gl_total_cell']}")
        else:
            ws.cell(row=flex_gl_row, column=3, value=nav_data.get('Flex Option G/L', 0))
        ws.cell(row=flex_gl_row, column=3).number_format = self.number_format

        # Flex Option G/L Adjusted
        ws.cell(row=flex_gl_adj_row, column=1, value="  Flex Option G/L (Adjusted)")
        if flex_detail_info['has_data']:
            ws.cell(row=flex_gl_adj_row, column=3, value=f"={flex_detail_info['gl_adj_total_cell']}")
        else:
            ws.cell(row=flex_gl_adj_row, column=3, value=nav_data.get('Flex Option G/L Adj', nav_data.get('Flex Option G/L', 0)))
        ws.cell(row=flex_gl_adj_row, column=3).number_format = self.number_format

        # Treasury G/L
        ws.cell(row=treasury_gl_row, column=1, value="  Treasury G/L")
        if treasury_detail_info['has_data']:
            ws.cell(row=treasury_gl_row, column=3, value=f"={treasury_detail_info['gl_adj_total_cell']}")  # This is correct - K column has the adjusted G/L
        else:
            ws.cell(row=treasury_gl_row, column=3, value=nav_data.get('Treasury G/L', nav_data.get('TSY G/L', 0)))
        ws.cell(row=treasury_gl_row, column=3).number_format = self.number_format

        # Assignment G/L
        ws.cell(row=assign_gl_row, column=1, value="  Assignment G/L")
        ws.cell(row=assign_gl_row, column=3, value=nav_data.get('Assignment G/L', 0)).number_format = self.number_format

        # Other
        ws.cell(row=other_row, column=1, value="  Other")
        ws.cell(row=other_row, column=3, value=nav_data.get('Other', nav_data.get('OTHER', 0))).number_format = self.number_format

        # Expenses (negative value)
        ws.cell(row=expenses_row, column=1, value="Expenses")
        expenses_value = -abs(nav_data.get('Accruals', nav_data.get('ACCRUALS', 0)))  # Ensure negative
        ws.cell(row=expenses_row, column=3, value=expenses_value).number_format = self.number_format

        # Dividends (positive value)
        ws.cell(row=dividends_row, column=1, value="Dividends")
        ws.cell(row=dividends_row, column=3, value=nav_data.get('Dividends', 0)).number_format = self.number_format

        # Expected TNA with formula (now including all components as addition)
        ws.cell(row=expected_tna_row, column=1, value="Expected TNA").font = self.subheader_font
        formula = f"=C{adj_beg_tna_row}+C{equity_gl_adj_row}+C{option_gl_adj_row}+C{flex_gl_adj_row}+C{treasury_gl_row}+C{assign_gl_row}+C{other_row}+C{expenses_row}+C{dividends_row}"
        ws.cell(row=expected_tna_row, column=3, value=formula).number_format = self.number_format

        # Custodian TNA
        ws.cell(row=cust_tna_row, column=1, value="Custodian TNA").font = self.subheader_font
        ws.cell(row=cust_tna_row, column=3, value=nav_data.get('Custodian TNA', 0)).number_format = self.number_format

        # TNA Difference with formula
        ws.cell(row=tna_diff_row, column=1, value="TNA Difference")
        ws.cell(row=tna_diff_row, column=3, value=f"=C{expected_tna_row}-C{cust_tna_row}").number_format = self.number_format

        # Shares Outstanding (no formula)
        ws.cell(row=shares_row, column=1, value="Shares Outstanding")
        ws.cell(row=shares_row, column=3, value=nav_data.get('Shares Outstanding', 0)).number_format = '#,##0'

        # Expected NAV with formula
        ws.cell(row=expected_nav_row, column=1, value="Expected NAV").font = self.subheader_font
        ws.cell(row=expected_nav_row, column=3, value=f"=IF(C{shares_row}<>0,C{expected_tna_row}/C{shares_row},0)").number_format = self.number_format_4

        # Custodian NAV
        ws.cell(row=cust_nav_row, column=1, value="Custodian NAV").font = self.subheader_font
        ws.cell(row=cust_nav_row, column=3, value=nav_data.get('Custodian NAV', 0)).number_format = self.number_format_4

        # NAV Difference with formula
        ws.cell(row=nav_diff_row, column=1, value="NAV Difference")
        ws.cell(row=nav_diff_row, column=3, value=f"=C{expected_nav_row}-C{cust_nav_row}").number_format = self.number_format_4

        # NAV Good formulas
        ws.cell(row=nav_good_2_row, column=1, value="NAV Good (2 decimal)")
        nav_good_2_formula = f'=IF(ABS(C{nav_diff_row}/C{cust_nav_row})<=0.0055,"PASS","FAIL")'
        ws.cell(row=nav_good_2_row, column=3, value=nav_good_2_formula)

        ws.cell(row=nav_good_4_row, column=1, value="NAV Good (4 decimal)")
        nav_good_4_formula = f'=IF(ABS(C{nav_diff_row}/C{cust_nav_row})<=0.0055,"PASS","FAIL")'
        ws.cell(row=nav_good_4_row, column=3, value=nav_good_4_formula)

        # Auto-fit columns
        for col in range(1, 12):
            ws.column_dimensions[get_column_letter(col)].width = 12

        # ws.freeze_panes = 'A33'

    def _add_equity_details_with_prices(self, ws, start_row, start_col, nav_data):
        """Add equity details section with prices used columns."""
        ws.cell(row=start_row, column=start_col, value="EQUITY GAIN/LOSS DETAIL").font = self.header_font

        # Get detailed data from nav_data
        equity_details = self._extract_equity_details(nav_data)

        if not equity_details:
            ws.cell(row=start_row + 2, column=start_col, value="No equity holdings")
            return {
                'has_data': False,
                'gl_total_cell': None,
                'gl_adj_total_cell': None,
                'end_row': start_row + 2
            }

        # Headers with Prices Used columns
        headers = [
            'Ticker', 'Qty T-1', 'Qty T',
            'Price T-1\n(Raw)', 'Price T\n(Raw)',
            'Price T-1\n(Adj)', 'Price T\n(Adj)',
            'Prices Used\nT-1', 'Prices Used\nT',
            'G/L', 'G/L Adj'
        ]

        header_row = start_row + 2
        for i, header in enumerate(headers):
            cell = ws.cell(row=header_row, column=start_col + i, value=header)
            cell.font = self.subheader_font
            cell.alignment = Alignment(horizontal='center', wrap_text=True)

        # Create table
        data_row = header_row + 1
        first_data_row = data_row

        for detail in equity_details:
            col = start_col

            # Ticker
            ws.cell(row=data_row, column=col, value=detail['ticker'])
            col += 1

            # Quantities
            ws.cell(row=data_row, column=col, value=detail.get('quantity_t1', 0)).number_format = '#,##0'
            col += 1
            ws.cell(row=data_row, column=col, value=detail.get('quantity_t', 0)).number_format = '#,##0'
            col += 1

            # Raw prices
            ws.cell(row=data_row, column=col, value=detail.get('price_t1_raw', 0)).number_format = self.number_format
            col += 1
            ws.cell(row=data_row, column=col, value=detail.get('price_t_raw', 0)).number_format = self.number_format
            col += 1

            # Adjusted prices - highlight if different
            price_t1_adj = detail.get('price_t1_adj', detail.get('price_t1_raw', 0))
            price_t_adj = detail.get('price_t_adj', detail.get('price_t_raw', 0))

            adj_t1_cell = ws.cell(row=data_row, column=col, value=price_t1_adj)
            adj_t1_cell.number_format = self.number_format
            if abs(price_t1_adj - detail.get('price_t1_raw', 0)) > 0.001:
                adj_t1_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            adj_t_cell = ws.cell(row=data_row, column=col, value=price_t_adj)
            adj_t_cell.number_format = self.number_format
            if abs(price_t_adj - detail.get('price_t_raw', 0)) > 0.001:
                adj_t_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            # Prices Used columns
            ws.cell(row=data_row, column=col, value=price_t1_adj).number_format = self.number_format
            col += 1
            ws.cell(row=data_row, column=col, value=price_t_adj).number_format = self.number_format
            col += 1

            # G/L calculations - USE FORMULAS instead of values
            # G/L Raw = (Price T Raw - Price T-1 Raw) * Qty T
            gl_formula = f"=(E{data_row}-D{data_row})*C{data_row}"
            ws.cell(row=data_row, column=col, value=gl_formula).number_format = self.number_format
            col += 1

            # G/L Adj = (Price T Adj - Price T-1 Adj) * Qty T
            gl_adj_formula = f"=(G{data_row}-F{data_row})*C{data_row}"
            ws.cell(row=data_row, column=col, value=gl_adj_formula).number_format = self.number_format

            data_row += 1

        # Add totals row if we have data
        if data_row > first_data_row:
            last_data_row = data_row - 1

            # Define the table range for G/L columns
            gl_range = f"J{first_data_row}:J{last_data_row}"
            gl_adj_range = f"K{first_data_row}:K{last_data_row}"

            # Add totals row
            total_row = data_row + 1
            ws.cell(row=total_row, column=start_col, value="TOTAL").font = self.subheader_font
            ws.cell(row=total_row, column=start_col + 9, value=f"=SUM({gl_range})").number_format = self.number_format
            ws.cell(row=total_row, column=start_col + 10, value=f"=SUM({gl_adj_range})").number_format = self.number_format

            return {
                'has_data': True,
                'gl_total_cell': f"J{total_row}",
                'gl_adj_total_cell': f"K{total_row}",
                'end_row': total_row
            }
        else:
            return {
                'has_data': False,
                'gl_total_cell': None,
                'gl_adj_total_cell': None,
                'end_row': data_row
            }

    def _add_option_details_with_prices(self, ws, start_row, start_col, nav_data):
        """Add option details section with prices used columns."""
        ws.cell(row=start_row, column=start_col, value="OPTION GAIN/LOSS DETAIL").font = self.header_font

        # Get detailed data
        option_details = self._extract_option_details(nav_data)

        if not option_details:
            ws.cell(row=start_row + 2, column=start_col, value="No option holdings")
            return {
                'has_data': False,
                'gl_total_cell': None,
                'gl_adj_total_cell': None,
                'end_row': start_row + 2
            }

        # Same structure as equity
        headers = [
            'Ticker', 'Qty T-1', 'Qty T',
            'Price T-1\n(Raw)', 'Price T\n(Raw)',
            'Price T-1\n(Adj)', 'Price T\n(Adj)',
            'Prices Used\nT-1', 'Prices Used\nT',
            'G/L', 'G/L Adj'
        ]

        header_row = start_row + 2
        for i, header in enumerate(headers):
            cell = ws.cell(row=header_row, column=start_col + i, value=header)
            cell.font = self.subheader_font
            cell.alignment = Alignment(horizontal='center', wrap_text=True)

        # Data rows
        data_row = header_row + 1
        first_data_row = data_row

        for detail in option_details:
            col = start_col

            # Similar structure to equity
            ws.cell(row=data_row, column=col, value=detail['ticker'])
            col += 1

            ws.cell(row=data_row, column=col, value=detail.get('quantity_t1', 0)).number_format = '#,##0'
            col += 1
            ws.cell(row=data_row, column=col, value=detail.get('quantity_t', 0)).number_format = '#,##0'
            col += 1

            ws.cell(row=data_row, column=col, value=detail.get('price_t1_raw', 0)).number_format = self.number_format
            col += 1
            ws.cell(row=data_row, column=col, value=detail.get('price_t_raw', 0)).number_format = self.number_format
            col += 1

            price_t1_adj = detail.get('price_t1_adj', detail.get('price_t1_raw', 0))
            price_t_adj = detail.get('price_t_adj', detail.get('price_t_raw', 0))

            adj_t1_cell = ws.cell(row=data_row, column=col, value=price_t1_adj)
            adj_t1_cell.number_format = self.number_format
            if abs(price_t1_adj - detail.get('price_t1_raw', 0)) > 0.001:
                adj_t1_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            adj_t_cell = ws.cell(row=data_row, column=col, value=price_t_adj)
            adj_t_cell.number_format = self.number_format
            if abs(price_t_adj - detail.get('price_t_raw', 0)) > 0.001:
                adj_t_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            # Prices Used
            ws.cell(row=data_row, column=col, value=price_t1_adj).number_format = self.number_format
            col += 1
            ws.cell(row=data_row, column=col, value=price_t_adj).number_format = self.number_format
            col += 1

            # G/L calculations - USE FORMULAS with *100 for options
            # G/L Raw = (Price T Raw - Price T-1 Raw) * Qty T * 100
            gl_formula = f"=(E{data_row}-D{data_row})*C{data_row}*100"
            ws.cell(row=data_row, column=col, value=gl_formula).number_format = self.number_format
            col += 1

            # G/L Adj = (Price T Adj - Price T-1 Adj) * Qty T * 100
            gl_adj_formula = f"=(G{data_row}-F{data_row})*C{data_row}*100"
            ws.cell(row=data_row, column=col, value=gl_adj_formula).number_format = self.number_format

            data_row += 1

        # Create totals if we have data
        if data_row > first_data_row:
            last_data_row = data_row - 1

            gl_range = f"J{first_data_row}:J{last_data_row}"
            gl_adj_range = f"K{first_data_row}:K{last_data_row}"

            total_row = data_row + 1
            ws.cell(row=total_row, column=start_col, value="TOTAL").font = self.subheader_font
            ws.cell(row=total_row, column=start_col + 9, value=f"=SUM({gl_range})").number_format = self.number_format
            ws.cell(row=total_row, column=start_col + 10, value=f"=SUM({gl_adj_range})").number_format = self.number_format

            return {
                'has_data': True,
                'gl_total_cell': f"J{total_row}",
                'gl_adj_total_cell': f"K{total_row}",
                'end_row': total_row
            }
        else:
            return {
                'has_data': False,
                'gl_total_cell': None,
                'gl_adj_total_cell': None,
                'end_row': data_row
            }

    def _add_flex_details_with_prices(self, ws, start_row, start_col, nav_data):
        """Add flex option details section with prices used columns."""
        ws.cell(row=start_row, column=start_col, value="FLEX OPTION GAIN/LOSS DETAIL").font = self.header_font

        # Get detailed data
        flex_details = self._extract_flex_details(nav_data)

        if not flex_details:
            ws.cell(row=start_row + 2, column=start_col, value="No flex option holdings")
            return {
                'has_data': False,
                'gl_total_cell': None,
                'gl_adj_total_cell': None,
                'end_row': start_row + 2
            }

        # Same structure as regular options
        headers = [
            'Ticker', 'Qty T-1', 'Qty T',
            'Price T-1\n(Raw)', 'Price T\n(Raw)',
            'Price T-1\n(Adj)', 'Price T\n(Adj)',
            'Prices Used\nT-1', 'Prices Used\nT',
            'G/L', 'G/L Adj'
        ]

        header_row = start_row + 2
        for i, header in enumerate(headers):
            cell = ws.cell(row=header_row, column=start_col + i, value=header)
            cell.font = self.subheader_font
            cell.alignment = Alignment(horizontal='center', wrap_text=True)

        # Data rows
        data_row = header_row + 1
        first_data_row = data_row

        for detail in flex_details:
            col = start_col

            ws.cell(row=data_row, column=col, value=detail['ticker'])
            col += 1

            ws.cell(row=data_row, column=col, value=detail.get('quantity_t1', 0)).number_format = '#,##0'
            col += 1
            ws.cell(row=data_row, column=col, value=detail.get('quantity_t', 0)).number_format = '#,##0'
            col += 1

            ws.cell(row=data_row, column=col, value=detail.get('price_t1_raw', 0)).number_format = self.number_format
            col += 1
            ws.cell(row=data_row, column=col, value=detail.get('price_t_raw', 0)).number_format = self.number_format
            col += 1

            price_t1_adj = detail.get('price_t1_adj', detail.get('price_t1_raw', 0))
            price_t_adj = detail.get('price_t_adj', detail.get('price_t_raw', 0))

            adj_t1_cell = ws.cell(row=data_row, column=col, value=price_t1_adj)
            adj_t1_cell.number_format = self.number_format
            if abs(price_t1_adj - detail.get('price_t1_raw', 0)) > 0.001:
                adj_t1_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            adj_t_cell = ws.cell(row=data_row, column=col, value=price_t_adj)
            adj_t_cell.number_format = self.number_format
            if abs(price_t_adj - detail.get('price_t_raw', 0)) > 0.001:
                adj_t_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            # Prices Used
            ws.cell(row=data_row, column=col, value=price_t1_adj).number_format = self.number_format
            col += 1
            ws.cell(row=data_row, column=col, value=price_t_adj).number_format = self.number_format
            col += 1

            # G/L calculations - USE FORMULAS with *100 for flex options
            # G/L Raw = (Price T Raw - Price T-1 Raw) * Qty T * 100
            gl_formula = f"=(E{data_row}-D{data_row})*C{data_row}*100"
            ws.cell(row=data_row, column=col, value=gl_formula).number_format = self.number_format
            col += 1

            # G/L Adj = (Price T Adj - Price T-1 Adj) * Qty T * 100
            gl_adj_formula = f"=(G{data_row}-F{data_row})*C{data_row}*100"
            ws.cell(row=data_row, column=col, value=gl_adj_formula).number_format = self.number_format

            data_row += 1

        # Create totals if we have data
        if data_row > first_data_row:
            last_data_row = data_row - 1

            gl_range = f"J{first_data_row}:J{last_data_row}"
            gl_adj_range = f"K{first_data_row}:K{last_data_row}"

            total_row = data_row + 1
            ws.cell(row=total_row, column=start_col, value="TOTAL").font = self.subheader_font
            ws.cell(row=total_row, column=start_col + 9, value=f"=SUM({gl_range})").number_format = self.number_format
            ws.cell(row=total_row, column=start_col + 10, value=f"=SUM({gl_adj_range})").number_format = self.number_format

            return {
                'has_data': True,
                'gl_total_cell': f"J{total_row}",
                'gl_adj_total_cell': f"K{total_row}",
                'end_row': total_row
            }
        else:
            return {
                'has_data': False,
                'gl_total_cell': None,
                'gl_adj_total_cell': None,
                'end_row': data_row
            }

    def _extract_equity_details(self, nav_data):
        """Extract equity details from nav_data with fallback logic."""
        # Try multiple sources for equity data
        details = []

        # First try detailed_calculations (this should now have data)
        if 'detailed_calculations' in nav_data:
            detailed_calcs = nav_data['detailed_calculations']
            if isinstance(detailed_calcs, dict) and 'equity_details' in detailed_calcs:
                equity_df = detailed_calcs['equity_details']
                if isinstance(equity_df, pd.DataFrame) and not equity_df.empty:
                    return equity_df.to_dict('records')
                elif isinstance(equity_df, list):
                    return equity_df

        # Try raw equity data
        if 'raw_equity' in nav_data:
            raw_df = nav_data['raw_equity']
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                # Convert raw data to expected format
                for _, row in raw_df.iterrows():
                    details.append({
                        'ticker': row.get('equity_ticker', row.get('ticker', '')),
                        'quantity_t1': row.get('quantity_t1', 0),
                        'quantity_t': row.get('quantity_t', row.get('quantity', 0)),
                        'price_t1_raw': row.get('price_t1', 0),
                        'price_t_raw': row.get('price_t', 0),
                        'price_t1_adj': row.get('price_t1_adj', row.get('price_t1', 0)),
                        'price_t_adj': row.get('price_t_adj', row.get('price_t', 0))
                    })
                return details

        # Try components data structure
        if 'components' in nav_data and 'equity' in nav_data['components']:
            equity_data = nav_data['components']['equity']
            if 'details' in equity_data:
                return equity_data['details']

        return details

    def _extract_option_details(self, nav_data):
        """Extract option details (non-flex) from nav_data."""
        details = []

        # Similar extraction logic as equity
        if 'detailed_calculations' in nav_data:
            option_df = nav_data['detailed_calculations'].get('option_details', pd.DataFrame())
            if isinstance(option_df, pd.DataFrame) and not option_df.empty:
                return option_df.to_dict('records')

        # Try raw option data
        if 'raw_option' in nav_data:
            raw_df = nav_data['raw_option']
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                # Filter out flex options
                if 'is_flex' in raw_df.columns:
                    non_flex = raw_df[~raw_df['is_flex']]
                else:
                    # Check if ticker contains FLEX
                    if 'optticker' in raw_df.columns:
                        non_flex = raw_df[~raw_df['optticker'].str.contains('FLEX', na=False)]
                    else:
                        non_flex = raw_df  # If no way to identify flex, assume all are non-flex

                for _, row in non_flex.iterrows():
                    details.append({
                        'ticker': row.get('optticker', row.get('ticker', '')),
                        'quantity_t1': row.get('quantity_t1', 0),
                        'quantity_t': row.get('quantity_t', row.get('quantity', 0)),
                        'price_t1_raw': row.get('price_t1', 0),
                        'price_t_raw': row.get('price_t', 0),
                        'price_t1_adj': row.get('price_t1_adj', row.get('price_t1', 0)),
                        'price_t_adj': row.get('price_t_adj', row.get('price_t', 0))
                    })
                return details

        return details

    def _extract_flex_details(self, nav_data):
        """Extract flex option details from nav_data."""
        details = []

        if 'detailed_calculations' in nav_data:
            flex_df = nav_data['detailed_calculations'].get('flex_details', pd.DataFrame())
            if isinstance(flex_df, pd.DataFrame) and not flex_df.empty:
                return flex_df.to_dict('records')

        # Try raw option data filtered for flex
        if 'raw_option' in nav_data:
            raw_df = nav_data['raw_option']
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                # Filter for flex options only
                if 'is_flex' in raw_df.columns:
                    flex_only = raw_df[raw_df['is_flex']]
                else:
                    # Check if ticker contains FLEX
                    if 'optticker' in raw_df.columns:
                        flex_only = raw_df[raw_df['optticker'].str.contains('FLEX', na=False)]
                    else:
                        flex_only = pd.DataFrame()  # If no way to identify flex, return empty

                for _, row in flex_only.iterrows():
                    details.append({
                        'ticker': row.get('optticker', row.get('ticker', '')),
                        'quantity_t1': row.get('quantity_t1', 0),
                        'quantity_t': row.get('quantity_t', row.get('quantity', 0)),
                        'price_t1_raw': row.get('price_t1', 0),
                        'price_t_raw': row.get('price_t', 0),
                        'price_t1_adj': row.get('price_t1_adj', row.get('price_t1', 0)),
                        'price_t_adj': row.get('price_t_adj', row.get('price_t', 0))
                    })
                return details

        return details

    def _calculate_summary_stats(self):
        """Calculate and store summary statistics about the NAV reconciliation"""
        fund_count = 0
        total_pct_diff = 0

        for date_str, fund_results in self.reconciliation_results.items():
            for fund_name, nav_data in fund_results.items():
                if isinstance(nav_data, dict):
                    fund_count += 1
                    self.stats["total_funds"] = fund_count

                    if nav_data.get("NAV Good (2 Digit)", False):
                        self.stats["funds_within_tolerance_2d"] += 1

                    if nav_data.get("NAV Good (4 Digit)", False):
                        self.stats["funds_within_tolerance_4d"] += 1

                    self.stats["total_nav_difference"] += nav_data.get("NAV Diff ($)", 0)
                    total_pct_diff += nav_data.get("Difference (%) - 2 Digit", 0)

        if fund_count > 0:
            self.stats["average_pct_difference"] = total_pct_diff / fund_count

    def _add_option_details_with_prices(self, ws, start_row, start_col, nav_data):
        """Add option details section with prices used columns."""
        ws.cell(row=start_row, column=start_col, value="OPTION GAIN/LOSS DETAIL").font = self.header_font

        # Get detailed data
        option_details = self._extract_option_details(nav_data)

        if not option_details:
            ws.cell(row=start_row + 2, column=start_col, value="No option holdings")
            return {
                'has_data': False,
                'gl_total_cell': None,
                'gl_adj_total_cell': None,
                'end_row': start_row + 2
            }

        # Headers
        headers = [
            'Ticker', 'Qty T-1', 'Qty T',
            'Price T-1\n(Raw/OMS)', 'Price T\n(Raw/OMS)',
            'Price T-1\n(Adj/Cust)', 'Price T\n(Adj/Cust)',
            'Prices Used\nT-1', 'Prices Used\nT',
            'G/L (Raw)', 'G/L (Adj)'
        ]

        header_row = start_row + 2
        for i, header in enumerate(headers):
            cell = ws.cell(row=header_row, column=start_col + i, value=header)
            cell.font = self.subheader_font
            cell.alignment = Alignment(horizontal='center', wrap_text=True)

        # Data rows
        data_row = header_row + 1
        first_data_row = data_row

        for detail in option_details:
            col = start_col

            # Ticker
            ws.cell(row=data_row, column=col, value=detail['ticker'])
            col += 1

            # Quantities
            ws.cell(row=data_row, column=col, value=detail.get('quantity_t1', 0)).number_format = '#,##0'
            col += 1
            ws.cell(row=data_row, column=col, value=detail.get('quantity_t', 0)).number_format = '#,##0'
            col += 1

            # Raw prices (OMS)
            price_t1_raw = detail.get('price_t1_raw', 0)
            price_t_raw = detail.get('price_t_raw', 0)

            ws.cell(row=data_row, column=col, value=price_t1_raw).number_format = self.number_format
            col += 1
            ws.cell(row=data_row, column=col, value=price_t_raw).number_format = self.number_format
            col += 1

            # Adjusted prices (Custodian) - highlight if different from raw
            price_t1_adj = detail.get('price_t1_adj', price_t1_raw)
            price_t_adj = detail.get('price_t_adj', price_t_raw)

            adj_t1_cell = ws.cell(row=data_row, column=col, value=price_t1_adj)
            adj_t1_cell.number_format = self.number_format
            if abs(price_t1_adj - price_t1_raw) > 0.001:
                adj_t1_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            adj_t_cell = ws.cell(row=data_row, column=col, value=price_t_adj)
            adj_t_cell.number_format = self.number_format
            if abs(price_t_adj - price_t_raw) > 0.001:
                adj_t_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            # Prices Used columns (show what went into the NAV calculation)
            ws.cell(row=data_row, column=col, value=price_t1_adj).number_format = self.number_format
            col += 1
            ws.cell(row=data_row, column=col, value=price_t_adj).number_format = self.number_format
            col += 1

            # G/L calculations - USE FORMULAS
            # G/L Raw = (Price T Raw - Price T-1 Raw) * Qty T * 100
            gl_raw_formula = f"=(E{data_row}-D{data_row})*C{data_row}*100"
            ws.cell(row=data_row, column=col, value=gl_raw_formula).number_format = self.number_format
            col += 1

            # G/L Adj = (Price T Adj - Price T-1 Adj) * Qty T * 100
            gl_adj_formula = f"=(G{data_row}-F{data_row})*C{data_row}*100"
            ws.cell(row=data_row, column=col, value=gl_adj_formula).number_format = self.number_format

            data_row += 1

        # Add totals row if we have data
        if data_row > first_data_row:
            last_data_row = data_row - 1

            gl_raw_range = f"J{first_data_row}:J{last_data_row}"
            gl_adj_range = f"K{first_data_row}:K{last_data_row}"

            total_row = data_row + 1
            ws.cell(row=total_row, column=start_col, value="TOTAL").font = self.subheader_font
            ws.cell(row=total_row, column=start_col + 9, value=f"=SUM({gl_raw_range})").number_format = self.number_format
            ws.cell(row=total_row, column=start_col + 10, value=f"=SUM({gl_adj_range})").number_format = self.number_format

            return {
                'has_data': True,
                'gl_total_cell': f"J{total_row}",  # Raw G/L total
                'gl_adj_total_cell': f"K{total_row}",  # Adjusted G/L total
                'end_row': total_row
            }
        else:
            return {
                'has_data': False,
                'gl_total_cell': None,
                'gl_adj_total_cell': None,
                'end_row': data_row
            }

    def _add_detailed_nav_sheet(self):
        """Add detailed NAV reconciliation sheet"""
        ws = self.wb.create_sheet("Detailed NAV Reconciliation")

        # Updated headers with new columns
        headers = [
            "Fund", "Date", "Beginning TNA", "Equity G/L", "Option G/L",
            "Flex Option G/L", "Treasury G/L", "Assignment G/L",
            "Dividends", "Accruals", "Distributions", "Other",
            "Expected TNA", "Custodian TNA", "TNA Diff ($)",
            "Expected NAV", "Custodian NAV", "NAV Diff ($)",
            "Expected NAV (Unadj Options)",  # ✅ NEW
            "NAV Diff vs Unadj ($)",  # ✅ NEW
            "NAV Good (2 Dec)", "NAV Good (4 Dec)"
        ]

        # Write headers
        for col_idx, header in enumerate(headers, start=1):
            cell = ws.cell(row=1, column=col_idx, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="CCE5FF", end_color="CCE5FF", fill_type="solid")

        # Write data
        row_idx = 2
        for date_str, fund_results in sorted(self.reconciliation_results.items()):
            for fund_name, nav_data in sorted(fund_results.items()):
                if not isinstance(nav_data, dict):
                    continue

                data = [
                    fund_name,
                    date_str,
                    nav_data.get("Beginning TNA", 0),
                    nav_data.get("Equity G/L", 0),
                    nav_data.get("Option G/L", 0),
                    nav_data.get("Flex Option G/L", 0),
                    nav_data.get("Treasury G/L", 0),
                    nav_data.get("Assignment G/L", 0),
                    nav_data.get("Dividends", 0),
                    nav_data.get("Accruals", 0),
                    nav_data.get("Distributions", 0),
                    nav_data.get("Other", 0),
                    nav_data.get("Expected TNA", 0),
                    nav_data.get("Custodian TNA", 0),
                    nav_data.get("TNA Diff ($)", 0),
                    nav_data.get("Expected NAV", 0),
                    nav_data.get("Custodian NAV", 0),
                    nav_data.get("NAV Diff ($)", 0),
                    nav_data.get("Expected NAV (Unadj Options)", 0),  # ✅ NEW
                    nav_data.get("NAV Diff vs Unadj ($)", 0),  # ✅ NEW
                    "PASS" if nav_data.get("NAV Good (2 Digit)", False) else "FAIL",
                    "PASS" if nav_data.get("NAV Good (4 Digit)", False) else "FAIL"
                ]

                for col_idx, value in enumerate(data, start=1):
                    cell = ws.cell(row=row_idx, column=col_idx, value=value)

                    # Format numbers
                    if col_idx >= 3 and col_idx <= 18:  # Numeric columns
                        cell.number_format = '#,##0.00'
                    elif col_idx == 19:  # Expected NAV (Unadj Options)
                        cell.number_format = '#,##0.0000'
                    elif col_idx == 20:  # NAV Diff vs Unadj
                        cell.number_format = '#,##0.0000'
                        # Highlight differences
                        if abs(value) > 0.01:
                            cell.fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

                    # Color PASS/FAIL cells
                    if col_idx in [21, 22]:
                        if value == "PASS":
                            cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                        else:
                            cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

                row_idx += 1

        # Auto-adjust column widths
        for col in ws.columns:
            max_length = 0
            col_letter = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 20)
            ws.column_dimensions[col_letter].width = adjusted_width

    def _add_treasury_details_with_prices(self, ws, start_row, start_col, nav_data):
        """Add treasury details section with prices used columns."""
        ws.cell(row=start_row, column=start_col, value="TREASURY GAIN/LOSS DETAIL").font = self.header_font

        # Get detailed data
        treasury_details = self._extract_treasury_details(nav_data)

        if not treasury_details:
            ws.cell(row=start_row + 2, column=start_col, value="No treasury holdings")
            return {
                'has_data': False,
                'gl_total_cell': None,
                'gl_adj_total_cell': None,
                'end_row': start_row + 2
            }

        # Headers
        headers = [
            'Ticker', 'Qty T-1', 'Qty T',
            'Price T-1\n(Raw)', 'Price T\n(Raw)',
            'Price T-1\n(Adj)', 'Price T\n(Adj)',
            'Prices Used\nT-1', 'Prices Used\nT',
            'G/L', 'G/L Adj'
        ]

        header_row = start_row + 2
        for i, header in enumerate(headers):
            cell = ws.cell(row=header_row, column=start_col + i, value=header)
            cell.font = self.subheader_font
            cell.alignment = Alignment(horizontal='center', wrap_text=True)

        # Data rows
        data_row = header_row + 1
        first_data_row = data_row

        for detail in treasury_details:
            col = start_col

            # Ticker
            ws.cell(row=data_row, column=col, value=detail['ticker'])
            col += 1

            # Quantities
            ws.cell(row=data_row, column=col, value=detail.get('quantity_t1', 0)).number_format = '#,##0'
            col += 1
            ws.cell(row=data_row, column=col, value=detail.get('quantity_t', 0)).number_format = '#,##0'
            col += 1

            # Raw prices
            ws.cell(row=data_row, column=col, value=detail.get('price_t1_raw', 0)).number_format = self.number_format
            col += 1
            ws.cell(row=data_row, column=col, value=detail.get('price_t_raw', 0)).number_format = self.number_format
            col += 1

            # Adjusted prices - highlight if different
            price_t1_adj = detail.get('price_t1_adj', detail.get('price_t1_raw', 0))
            price_t_adj = detail.get('price_t_adj', detail.get('price_t_raw', 0))

            adj_t1_cell = ws.cell(row=data_row, column=col, value=price_t1_adj)
            adj_t1_cell.number_format = self.number_format
            if abs(price_t1_adj - detail.get('price_t1_raw', 0)) > 0.001:
                adj_t1_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            adj_t_cell = ws.cell(row=data_row, column=col, value=price_t_adj)
            adj_t_cell.number_format = self.number_format
            if abs(price_t_adj - detail.get('price_t_raw', 0)) > 0.001:
                adj_t_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            # Prices Used columns
            ws.cell(row=data_row, column=col, value=price_t1_adj).number_format = self.number_format
            col += 1
            ws.cell(row=data_row, column=col, value=price_t_adj).number_format = self.number_format
            col += 1

            # G/L calculations - USE FORMULAS
            # G/L Raw = (Price T Raw - Price T-1 Raw) * Qty T
            gl_formula = f"=(E{data_row}-D{data_row})*C{data_row}"
            ws.cell(row=data_row, column=col, value=gl_formula).number_format = self.number_format
            col += 1

            # G/L Adj = (Price T Adj - Price T-1 Adj) * Qty T
            gl_adj_formula = f"=(G{data_row}-F{data_row})*C{data_row}"
            ws.cell(row=data_row, column=col, value=gl_adj_formula).number_format = self.number_format

            data_row += 1

        # Add totals row if we have data
        if data_row > first_data_row:
            last_data_row = data_row - 1

            gl_range = f"J{first_data_row}:J{last_data_row}"
            gl_adj_range = f"K{first_data_row}:K{last_data_row}"

            total_row = data_row + 1
            ws.cell(row=total_row, column=start_col, value="TOTAL").font = self.subheader_font
            ws.cell(row=total_row, column=start_col + 9, value=f"=SUM({gl_range})").number_format = self.number_format
            ws.cell(row=total_row, column=start_col + 10, value=f"=SUM({gl_adj_range})").number_format = self.number_format

            return {
                'has_data': True,
                'gl_total_cell': f"J{total_row}",
                'gl_adj_total_cell': f"K{total_row}",
                'end_row': total_row
            }
        else:
            return {
                'has_data': False,
                'gl_total_cell': None,
                'gl_adj_total_cell': None,
                'end_row': data_row
            }

    def _extract_treasury_details(self, nav_data):
        """Extract treasury details from nav_data."""
        details = []

        # Try detailed_calculations first
        if 'detailed_calculations' in nav_data:
            detailed_calcs = nav_data['detailed_calculations']
            if isinstance(detailed_calcs, dict) and 'treasury_details' in detailed_calcs:
                treasury_df = detailed_calcs['treasury_details']
                if isinstance(treasury_df, pd.DataFrame) and not treasury_df.empty:
                    return treasury_df.to_dict('records')
                elif isinstance(treasury_df, list):
                    return treasury_df

        # Try raw treasury data
        if 'raw_treasury' in nav_data:
            raw_df = nav_data['raw_treasury']
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                for _, row in raw_df.iterrows():
                    details.append({
                        'ticker': row.get('treasury_ticker', row.get('ticker', '')),
                        'quantity_t1': row.get('quantity_t1', 0),
                        'quantity_t': row.get('quantity_t', row.get('quantity', 0)),
                        'price_t1_raw': row.get('price_t1', 0),
                        'price_t_raw': row.get('price_t', 0),
                        'price_t1_adj': row.get('price_t1_adj', row.get('price_t1', 0)),
                        'price_t_adj': row.get('price_t_adj', row.get('price_t', 0))
                    })
                return details

        return details


from fpdf import FPDF
import datetime
import pandas as pd

class NavReconciliationPDF:
    def __init__(self,
                 reconciliation_results: dict,
                 recon_summary: list,
                 date: str,
                 output_path: str):
        """
        reconciliation_results: { date_str: { fund_name: nav_result_dict } }
        recon_summary:        list of summaries (unused here, but could feed a summary page)
        date:                 human‐readable date or date‐range
        output_path:          full path to write the PDF
        """
        self.reconciliation_results = reconciliation_results
        self.recon_summary = recon_summary
        self.date = date
        self.output_path = output_path

        # 1) Setup PDF
        self.pdf = FPDF(orientation='L', unit='mm', format='A4')
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.alias_nb_pages()

        # 2) Build pages
        self._generate_pdf()

        # 3) Save
        self.pdf.output(self.output_path)

    def _generate_pdf(self):
        # PAGE 1: NAV Summary
        self.pdf.add_page()
        self._add_report_title()
        self._add_table_header()
        for _, funds in self.reconciliation_results.items():
            for fund_name, nav in sorted(funds.items()):
                if self.pdf.get_y() > 180:
                    self.pdf.add_page()
                    self._add_report_title()
                    self._add_table_header()
                self._add_fund_nav_row(fund_name, nav)

        # PAGE 2: Gain/Loss Components (no blank “reserved” page, no NAV title here)
        self.pdf.add_page()
        self._add_gl_components_table()

        # PAGE 3+: Individual fund sections (each call adds a page)
        self._add_fund_sections()

    def _add_nav_summary_table(self):
        """Add the NAV summary table"""
        # Add section title
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, "NAV Reconciliation Summary", ln=True, align="C")
        self.pdf.ln(2)

        # Define columns with widths
        cols = [
            ("Fund", 25),
            ("Expected TNA", 35),
            ("Custodian TNA", 35),
            ("TNA Diff ($)", 30),
            ("Expected NAV", 30),
            ("Custodian NAV", 30),
            ("NAV Diff (2-dec)", 30),
            ("NAV Diff (4-dec)", 30)
        ]

        # Header row
        self.pdf.set_font("Arial", "B", 9)
        self.pdf.set_fill_color(240, 240, 240)
        for label, width in cols:
            self.pdf.cell(width, 7, label, border=1, fill=True, align="C")
        self.pdf.ln()

        # Data rows
        for date_str, funds in self.reconciliation_results.items():
            sorted_funds = sorted(funds.items())

            for fund_name, nav in sorted_funds:
                self._add_fund_nav_row(fund_name, nav)

    def _add_consolidated_summary(self):
        """Add a consolidated summary table showing all funds' reconciliation counts"""
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, "Reconciliation Summary - All Funds", ln=True)
        self.pdf.ln(2)

        # Collect all unique funds and build summary data
        summary_data = {}
        all_funds = set()

        # First pass: collect all funds
        for (fund_name, date_str) in self.flattened_results.keys():
            all_funds.add(fund_name)

        # Second pass: count discrepancies by fund
        for fund in sorted(all_funds):
            fund_summary = self._get_fund_summary(fund)
            if fund_summary and "summary" in fund_summary:
                summary_data[fund] = fund_summary["summary"]
            else:
                # Count from flattened results if summary not available
                counts = {}
                for (f, d), recon_data in self.flattened_results.items():
                    if f == fund:
                        for recon_type, df in recon_data.items():
                            if recon_type not in counts:
                                counts[recon_type] = {}
                            # Count different types of discrepancies
                            if isinstance(df, pd.DataFrame):
                                counts[recon_type]['total'] = len(df)
                summary_data[fund] = counts

        # Create the summary table
        if summary_data:
            # Define columns
            col_headers = ["Fund", "Index Equity", "Custodian Equity", "Custodian Option"]
            col_widths = [30, 40, 45, 45]

            # Header row
            self.pdf.set_font("Arial", "B", 9)
            self.pdf.set_fill_color(240, 240, 240)
            for i, header in enumerate(col_headers):
                self.pdf.cell(col_widths[i], 7, header, border=1, fill=True, align="C")
            self.pdf.ln()

            # Data rows
            self.pdf.set_font("Arial", "", 8)
            for fund in sorted(summary_data.keys()):
                fund_data = summary_data[fund]

                # Fund name - bold
                self.pdf.set_font("Arial", "B", 8)
                self.pdf.cell(col_widths[0], 6, fund, border=1, align="C")
                self.pdf.set_font("Arial", "", 8)

                # Index Equity count
                index_count = 0
                if 'index_equity' in fund_data:
                    if isinstance(fund_data['index_equity'], dict):
                        index_count = (fund_data['index_equity'].get('holdings_discrepancies', 0) +
                                       fund_data['index_equity'].get('significant_diffs', 0))
                    else:
                        index_count = fund_data['index_equity']

                # Custodian Equity count
                cust_eq_count = 0
                if 'custodian_equity' in fund_data:
                    if isinstance(fund_data['custodian_equity'], dict):
                        cust_eq_count = fund_data['custodian_equity'].get('final_recon', 0)
                    else:
                        cust_eq_count = fund_data['custodian_equity']

                # Custodian Option count
                cust_opt_count = 0
                if 'custodian_option' in fund_data:
                    if isinstance(fund_data['custodian_option'], dict):
                        cust_opt_count = fund_data['custodian_option'].get('final_recon', 0)
                    else:
                        cust_opt_count = fund_data['custodian_option']

                # Apply red background for non-zero counts
                for i, count in enumerate([index_count, cust_eq_count, cust_opt_count]):
                    if count > 0:
                        self.pdf.set_fill_color(255, 200, 200)  # Light red
                    else:
                        self.pdf.set_fill_color(255, 255, 255)  # White

                    self.pdf.cell(col_widths[i + 1], 6, str(count), border=1, fill=True, align="C")

                self.pdf.ln()

        self.pdf.ln(10)

    def _add_fund_section(self, fund_name, date_str, recon_data):
        """Add a section for a specific fund's reconciliation data"""

        # Fund header with grey background + larger text (without date)
        self.pdf.set_fill_color(230, 230, 230)  # light grey
        self.pdf.set_font("Arial", "B", 14)  # larger, bold
        header_text = f"Fund: {fund_name}"  # Removed date from header
        self.pdf.cell(0, 10, self._sanitize_text(header_text),
                      ln=True, fill=True)
        self.pdf.ln(4)  # small vertical gap

        # Summary section
        try:
            fund_summary = self._get_fund_summary(fund_name)
            if fund_summary:
                self._add_summary_section(fund_summary)
        except Exception as e:
            self.pdf.set_font("Arial", "I", 9)
            self.pdf.cell(0, 6, "Error processing summary information", ln=True)

        self.pdf.ln(2)

        # Each recon type
        orig_margin = self.pdf.l_margin
        for recon_type, result in recon_data.items():
            title = recon_type.replace("_", " ").title()

            # Underline only for the two custodian sections
            if recon_type in ("custodian_equity", "custodian_option"):
                self.pdf.set_font("Arial", "BU", 12)
            else:
                self.pdf.set_font("Arial", "B", 12)
            self.pdf.cell(0, 8, title, ln=True)

            # indent everything that follows
            self.pdf.set_left_margin(orig_margin + 8)
            try:
                self._print_recon_section(recon_type, result)
            except Exception as e:
                self.pdf.set_font("Arial", "I", 9)
                self.pdf.cell(0, 6, f"Error processing {title}", ln=True)
            finally:
                # restore original margin
                self.pdf.set_left_margin(orig_margin)

            self.pdf.ln(4)

        # separator line
        y = self.pdf.get_y()
        self.pdf.set_draw_color(180, 180, 180)
        self.pdf.line(10, y, 200, y)
        self.pdf.ln(6)

    def _add_nav_summary_table(self):
        """Add the NAV summary table"""
        # Add section title
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, "NAV Reconciliation Summary", ln=True, align="C")
        self.pdf.ln(2)

        # Define columns with widths
        cols = [
            ("Fund", 25),
            ("Expected TNA", 35),
            ("Custodian TNA", 35),
            ("TNA Diff ($)", 30),
            ("Expected NAV", 30),
            ("Custodian NAV", 30),
            ("NAV Diff (2-dec)", 30),
            ("NAV Diff (4-dec)", 30)
        ]

        # Header row
        self.pdf.set_font("Arial", "B", 9)
        self.pdf.set_fill_color(240, 240, 240)
        for label, width in cols:
            self.pdf.cell(width, 7, label, border=1, fill=True, align="C")
        self.pdf.ln()

        # Data rows
        for date_str, funds in self.reconciliation_results.items():
            sorted_funds = sorted(funds.items())

            for fund_name, nav in sorted_funds:
                self._add_fund_nav_row(fund_name, nav)

    def _add_report_title(self):
        """Add centered title and date at the top of each page"""
        # Title
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 10, "NAV Reconciliation", ln=True, align="C")

        # Date
        self.pdf.set_font("Arial", "", 12)
        self.pdf.cell(0, 8, f"Date: {self.date}", ln=True, align="C")

        # Add generated on date
        self.pdf.set_font("Arial", "I", 10)
        self.pdf.cell(0, 6, f"Generated on: {datetime.date.today().strftime('%Y-%m-%d')}", ln=True, align="C")

        # Add some space before the table
        self.pdf.ln(5)

    def _add_table_header(self):
        """Add the table header row"""
        # Add section title
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, "NAV Reconciliation Summary", ln=True, align="C")
        self.pdf.cell(0, 6, f"Date: {self.date}", ln=True, align="C")
        self.pdf.ln(2)

        # Define columns with widths
        cols = [
            ("Fund", 25),
            ("Expected TNA", 35),
            ("Custodian TNA", 35),
            ("TNA Diff ($)", 30),
            ("Expected NAV", 30),
            ("Custodian NAV", 30),
            ("NAV Diff (2-dec)", 30),
            ("NAV Diff (4-dec)", 30)
        ]

        # Header row
        self.pdf.set_font("Arial", "B", 9)
        self.pdf.set_fill_color(240, 240, 240)
        for label, width in cols:
            self.pdf.cell(width, 7, label, border=1, fill=True, align="C")
        self.pdf.ln()

    def _draw_table(self, data: dict, columns: list, total_width: float):
        """
        Draws a simple table in the current position.
        data:      dict of key → list_of_values (same length)
        columns:   list of (HeaderLabel, key)
        total_width: total table width in mm
        """
        ncols = len(columns)
        col_w = total_width / ncols

        # Header row
        self.pdf.set_font("Arial", "B", 10)
        self.pdf.set_fill_color(240, 240, 240)
        for label, _ in columns:
            self.pdf.cell(col_w, 6, label, border=1, fill=True, align="C")
        self.pdf.ln()

        # Data rows (we assume one row)
        self.pdf.set_font("Arial", "", 9)
        nrows = max(len(data.get(key, [])) for _, key in columns)
        for i in range(nrows):
            for _, key in columns:
                vals = data.get(key, [])
                val = vals[i] if i < len(vals) else ""
                # Format floats & booleans
                if isinstance(val, float):
                    txt = f"{val:,.6f}" if "Diff" in key else f"{val:,.2f}"
                elif isinstance(val, bool):
                    txt = "YES" if val else "NO"
                else:
                    txt = str(val)
                self.pdf.cell(col_w, 6, txt, border=1, align="R")
            self.pdf.ln()

    def _add_fund_nav_row(self, fund_name: str, nav: dict):
        """Draws one row for a single fund"""
        # Define columns with their data keys and widths
        cols = [
            ("fund", 25),
            ("Expected TNA", 35),
            ("Custodian TNA", 35),
            ("TNA Diff ($)", 30),
            ("Expected NAV", 30),
            ("Custodian NAV", 30),
            ("NAV Diff ($)", 30),  # 2-dec
            ("NAV Diff ($)", 30)  # 4-dec
        ]

        # Get values
        nav_diff = nav.get("NAV Diff ($)", 0.0)
        ok4 = nav.get("NAV Good (4 Digit)", False)
        ok2 = nav.get("NAV Good (2 Digit)", False)

        col_idx = 0
        for key, width in cols:
            # Handle fund name column - make it bold
            if key == "fund":
                self.pdf.set_font("Arial", "B", 8)
                self.pdf.set_fill_color(255, 255, 255)
                self.pdf.cell(width, 7, fund_name, border=1, fill=True, align="C")
                self.pdf.set_font("Arial", "", 8)  # Reset to normal font
                col_idx += 1
                continue

            # Get value
            val = nav.get(key, 0.0)

            # Determine formatting and coloring
            fill = False
            if col_idx == 3:  # TNA Diff ($)
                txt = f"{val:,.0f}"
                self.pdf.set_fill_color(255, 255, 255)
            elif col_idx == 6:  # NAV Diff (2-dec)
                txt = f"{nav_diff:,.2f}"
                # Color based on whether it's zero or not, and if it passes tolerance
                if abs(nav_diff) < 0.001:
                    self.pdf.set_fill_color(255, 255, 255)
                elif ok2:
                    self.pdf.set_fill_color(200, 255, 200)  # Green
                else:
                    self.pdf.set_fill_color(255, 200, 200)  # Red
                fill = True
            elif col_idx == 7:  # NAV Diff (4-dec)
                txt = f"{nav_diff:,.4f}"
                # Color based on whether it's zero or not, and if it passes tolerance
                if abs(nav_diff) < 0.0001:
                    self.pdf.set_fill_color(255, 255, 255)
                elif ok4:
                    self.pdf.set_fill_color(200, 255, 200)  # Green
                else:
                    self.pdf.set_fill_color(255, 200, 200)  # Red
                fill = True
            elif key in ("Expected TNA", "Custodian TNA"):
                txt = f"{val:,.0f}"
                self.pdf.set_fill_color(255, 255, 255)
            else:  # Expected/Custodian NAV
                txt = f"{val:,.2f}"
                self.pdf.set_fill_color(255, 255, 255)

            self.pdf.cell(width, 7, txt, border=1, fill=fill, align="R")
            col_idx += 1

        self.pdf.ln()

    def _add_gl_components_table(self):
        """Add G/L Components table showing breakdown for each fund"""
        # Title for G/L Components section
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, "Gain/Loss Components by Fund", ln=True, align="C")
        self.pdf.ln(2)

        # Define columns with widths
        cols = [
            ("Fund", 25),
            ("Beg. TNA", 32),
            ("Eqt G/L", 25),
            ("Opt G/L", 25),
            ("Flex Opt G/L", 28),
            ("Tsy G/L", 25),
            ("Assign G/L", 25),  # Add Assignment G/L column
            ("Accruals", 25),
            ("Other", 20),
            ("Expected TNA", 32),
            ("G/L Today", 28)
        ]

        # Header row
        self.pdf.set_font("Arial", "B", 8)
        self.pdf.set_fill_color(240, 240, 240)
        for label, width in cols:
            self.pdf.cell(width, 6, label, border=1, fill=True, align="C")
        self.pdf.ln()

        # Data rows
        self.pdf.set_font("Arial", "", 8)

        # Track totals
        total_beg_tna = 0.0
        total_eqt_gl = 0.0
        total_opt_gl = 0.0
        total_flex_gl = 0.0
        total_tsy_gl = 0.0
        total_assign_gl = 0.0
        total_accruals = 0.0
        total_other = 0.0
        total_expected_tna = 0.0
        total_gl_today = 0.0

        for date_str, funds in self.reconciliation_results.items():
            sorted_funds = sorted(funds.items())

            for fund_name, nav in sorted_funds:
                # Get values directly from results - FIXED KEY NAMES
                beg_tna = nav.get("Beginning TNA", 0.0)
                eqt_gl = nav.get("Equity G/L", 0.0)  # Changed from "Equity G/L Adj"
                opt_gl = nav.get("Option G/L", 0.0)  # Changed from "Option G/L Adj"
                flex_gl = nav.get("Flex Option G/L", 0.0)  # Changed from "Flex Option G/L Adj"
                tsy_gl = nav.get("Treasury G/L", 0.0)
                assign_gl = nav.get("Assignment G/L", 0.0)  # Add Assignment G/L
                accruals = nav.get("Accruals", 0.0)
                other = nav.get("Other", 0.0)
                expected_tna = nav.get("Expected TNA", 0.0)

                # Calculate total G/L today
                gl_today = eqt_gl + opt_gl + flex_gl + tsy_gl + assign_gl - accruals + other

                # Update totals
                total_beg_tna += beg_tna
                total_eqt_gl += eqt_gl
                total_opt_gl += opt_gl
                total_flex_gl += flex_gl
                total_tsy_gl += tsy_gl
                total_assign_gl += assign_gl
                total_accruals += accruals
                total_other += other
                total_expected_tna += expected_tna
                total_gl_today += gl_today

                # Fund name
                self.pdf.set_font("Arial", "B", 8)
                self.pdf.cell(cols[0][1], 6, fund_name, border=1, align="C")
                self.pdf.set_font("Arial", "", 8)

                # Format and add remaining values
                values = [
                    f"{beg_tna:,.0f}",
                    f"{eqt_gl:,.0f}",
                    f"{opt_gl:,.0f}",
                    f"{flex_gl:,.0f}",
                    f"{tsy_gl:,.0f}",
                    f"{assign_gl:,.0f}",
                    f"{accruals:,.0f}",
                    f"{other:,.0f}",
                    f"{expected_tna:,.0f}",
                    f"{gl_today:,.0f}"
                ]

                for i, (val, (_, width)) in enumerate(zip(values, cols[1:]), 1):
                    self.pdf.cell(width, 6, val, border=1, align="R")

                self.pdf.ln()

        # Add totals row
        self.pdf.set_font("Arial", "B", 8)
        self.pdf.set_fill_color(230, 230, 230)

        self.pdf.cell(cols[0][1], 6, "TOTAL", border=1, fill=True, align="C")

        totals = [
            f"{total_beg_tna:,.0f}",
            f"{total_eqt_gl:,.0f}",
            f"{total_opt_gl:,.0f}",
            f"{total_flex_gl:,.0f}",
            f"{total_tsy_gl:,.0f}",
            f"{total_assign_gl:,.0f}",
            f"{total_accruals:,.0f}",
            f"{total_other:,.0f}",
            f"{total_expected_tna:,.0f}",
            f"{total_gl_today:,.0f}"
        ]

        for val, (_, width) in zip(totals, cols[1:]):
            self.pdf.cell(width, 6, val, border=1, fill=True, align="R")

        self.pdf.ln()


    def _add_fund_nav_details(self, fund_name: str, nav: dict):
        """Add detailed NAV reconciliation information for a specific fund"""
        # Summary section
        self.pdf.set_font("Arial", "B", 11)
        self.pdf.cell(0, 8, "NAV Reconciliation Details", ln=True)
        self.pdf.ln(2)

        # Create a two-column layout for the details
        self.pdf.set_font("Arial", "", 9)

        details = [
            ("Beginning TNA:", f"${nav.get('Beginning TNA', 0):,.2f}"),
            ("Adjusted Beginning TNA:", f"${nav.get('Adjusted Beginning TNA', 0):,.2f}"),
            ("", ""),
            ("Gain/Loss Components:", ""),
            # The NAVReconciliator stores the adjusted value in "Equity G/L", not "Equity G/L Adj"
            ("  Equity G/L:", f"${nav.get('Equity G/L', 0):,.2f}"),
            ("  Option G/L:", f"${nav.get('Option G/L', 0):,.2f}"),
            ("  Flex Option G/L:", f"${nav.get('Flex Option G/L', 0):,.2f}"),
            ("  Treasury G/L:", f"${nav.get('Treasury G/L', 0):,.2f}"),
            ("  Assignment G/L:", f"${nav.get('Assignment G/L', 0):,.2f}"),
            ("  Accruals:", f"${nav.get('Accruals', 0):,.2f}"),
            ("  Other:", f"${nav.get('Other', 0):,.2f}"),
            ("", ""),
            ("Flows Adjustment:", f"${nav.get('Flows Adjustment', 0):,.2f}"),
            ("", ""),
            ("Expected TNA:", f"${nav.get('Expected TNA', 0):,.2f}"),
            ("Custodian TNA:", f"${nav.get('Custodian TNA', 0):,.2f}"),
            ("TNA Difference:", f"${nav.get('TNA Diff ($)', 0):,.2f}"),
            ("", ""),
            ("Expected NAV:", f"${nav.get('Expected NAV', 0):,.4f}"),
            ("Custodian NAV:", f"${nav.get('Custodian NAV', 0):,.4f}"),
            ("NAV Difference:", f"${nav.get('NAV Diff ($)', 0):,.4f}"),
            ("", ""),
            ("NAV Good (2 Digit):", "PASS" if nav.get('NAV Good (2 Digit)', False) else "FAIL"),
            ("NAV Good (4 Digit):", "PASS" if nav.get('NAV Good (4 Digit)', False) else "FAIL"),
        ]

        for label, value in details:
            if label:
                self.pdf.cell(80, 5, label, ln=False)
                self.pdf.cell(60, 5, value, ln=True)
            else:
                self.pdf.ln(3)

        self.pdf.ln(5)


        self._add_treasury_details_pdf(nav)  # Add treasury details here

    def _add_fund_sections(self):
        """Add individual fund detail sections"""
        for date_str, funds in self.reconciliation_results.items():
            sorted_funds = sorted(funds.items())

            for fund_name, nav in sorted_funds:
                # Add new page for each fund
                self.pdf.add_page()

                # Fund header
                self.pdf.set_fill_color(230, 230, 230)
                self.pdf.set_font("Arial", "B", 14)
                header_text = f"Fund: {fund_name} - {date_str}"
                self.pdf.cell(0, 10, header_text, ln=True, fill=True)
                self.pdf.ln(4)

                # Add detailed NAV reconciliation for this fund
                self._add_fund_nav_details(fund_name, nav)


    def _add_treasury_details_pdf(self, nav_data):
        """Add treasury details section to PDF"""
        self.pdf.set_font("Arial", "B", 11)
        self.pdf.cell(0, 8, "TREASURY GAIN/LOSS DETAIL", ln=True)
        self.pdf.ln(2)

        # Get treasury details
        treasury_details = self._extract_treasury_details_pdf(nav_data)

        if not treasury_details:
            self.pdf.set_font("Arial", "I", 9)
            self.pdf.cell(0, 6, "No treasury holdings", ln=True)
            self.pdf.ln(4)
            return

        # Column headers
        headers = ["Ticker", "Qty T-1", "Qty T", "Price T-1", "Price T", "G/L"]
        col_widths = [30, 25, 25, 25, 25, 30]

        self.pdf.set_font("Arial", "B", 9)
        self.pdf.set_fill_color(240, 240, 240)
        for i, header in enumerate(headers):
            self.pdf.cell(col_widths[i], 6, header, border=1, fill=True, align="C")
        self.pdf.ln()

        # Data rows
        self.pdf.set_font("Arial", "", 8)
        total_gl = 0.0

        for detail in treasury_details:
            ticker = detail.get('ticker', '')
            qty_t1 = detail.get('quantity_t1', 0)
            qty_t = detail.get('quantity_t', 0)
            price_t1 = detail.get('price_t1_adj', detail.get('price_t1_raw', 0))
            price_t = detail.get('price_t_adj', detail.get('price_t_raw', 0))

            # Calculate G/L
            gl = (price_t - price_t1) * qty_t
            total_gl += gl

            # Write row
            self.pdf.cell(col_widths[0], 6, str(ticker), border=1, align="C")
            self.pdf.cell(col_widths[1], 6, f"{qty_t1:,.0f}", border=1, align="R")
            self.pdf.cell(col_widths[2], 6, f"{qty_t:,.0f}", border=1, align="R")
            self.pdf.cell(col_widths[3], 6, f"{price_t1:,.2f}", border=1, align="R")
            self.pdf.cell(col_widths[4], 6, f"{price_t:,.2f}", border=1, align="R")
            self.pdf.cell(col_widths[5], 6, f"{gl:,.2f}", border=1, align="R")
            self.pdf.ln()

        # Totals row
        self.pdf.set_font("Arial", "B", 9)
        self.pdf.set_fill_color(230, 230, 230)
        self.pdf.cell(sum(col_widths[:-1]), 6, "TOTAL", border=1, fill=True, align="R")
        self.pdf.cell(col_widths[-1], 6, f"{total_gl:,.2f}", border=1, fill=True, align="R")
        self.pdf.ln()
        self.pdf.ln(4)


    def _extract_treasury_details_pdf(self, nav_data):
        """Extract treasury details for PDF"""
        details = []

        if 'detailed_calculations' in nav_data:
            treasury_df = nav_data['detailed_calculations'].get('treasury_details', pd.DataFrame())
            if isinstance(treasury_df, pd.DataFrame) and not treasury_df.empty:
                return treasury_df.to_dict('records')

        if 'raw_treasury' in nav_data:
            raw_df = nav_data['raw_treasury']
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                for _, row in raw_df.iterrows():
                    details.append({
                        'ticker': row.get('treasury_ticker', row.get('ticker', '')),
                        'quantity_t1': row.get('quantity_t1', 0),
                        'quantity_t': row.get('quantity_t', row.get('quantity', 0)),
                        'price_t1_raw': row.get('price_t1', 0),
                        'price_t_raw': row.get('price_t', 0),
                        'price_t1_adj': row.get('price_t1_adj', row.get('price_t1', 0)),
                        'price_t_adj': row.get('price_t_adj', row.get('price_t', 0))
                    })

        return details