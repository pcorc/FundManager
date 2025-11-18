from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from config.fund_definitions import INDEX_FLEX_FUNDS
from reporting.base_report_pdf import BaseReportPDF
from reporting.combined_reconciliation_report import build_combined_reconciliation_pdf

from reporting.report_utils import (
    ensure_dataframe,
    format_number,
    normalize_compliance_results,
    normalize_nav_payload,
    normalize_reconciliation_payload,
    normalize_report_date,
    summarise_compliance_status,
    summarise_nav_differences,
)
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
import re


@dataclass
class GeneratedNAVReconciliationReport:
    """File locations for generated NAV reconciliation artefacts."""

    excel_path: Optional[str]
    pdf_path: Optional[str]


class NAVReconciliationExcelReport:
    """Render NAV reconciliation data into an Excel workbook."""
    COMPONENTS = [
        ("equity", "Equity"),
        ("options", "Option"),
        ("flex_options", "Flex Option"),
        ("treasury", "Treasury"),
    ]

    def __init__(
        self,
        results: Mapping[str, Any],
        report_date: str,
        output_path: Path,
    ) -> None:
        self.results = normalize_nav_payload(results)
        self.report_date = report_date
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.title_font = Font(bold=True, size=14)
        self.header_font = Font(bold=True, size=11)
        self.subheader_font = Font(bold=True, size=10)
        self.body_font = Font(size=10)
        self.currency_format = "#,##0.00"
        self.nav_format = "#,##0.0000"
        self.integer_format = "#,##0"
        self.number_format = '#,##0.00'
        self.number_format_4 = '#,##0.0000'
        self.percent_format = '0.00%'
        self._sheet_names: set[str] = set()
        self._sheet_refs: Dict[str, Dict[str, Any]] = {}

        self._export()

    # ------------------------------------------------------------------
    def _export(self) -> None:
        workbook = Workbook()
        # Remove default sheet to control ordering
        if workbook.active:
            workbook.remove(workbook.active)

        for fund_name, payload in sorted(self.results.items()):
            if not payload:
                continue
            # Add self.report_date as the 4th argument
            self._create_fund_sheet(workbook, fund_name, payload, self.report_date)  # <-- FIX: Added self.report_date

        summary_ws = self._create_summary_sheet(workbook)
        if summary_ws is not None:
            # Move the summary sheet to the first position
            workbook._sheets.insert(0, workbook._sheets.pop(workbook._sheets.index(summary_ws)))

        workbook.save(self.output_path)

    # ------------------------------------------------------------------
    # Key changes needed in NAVReconciliationReport class

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
        ws.cell(row=row, column=3, value=nav_data.get('Beginning TNA', 0)).number_format = self.number_format
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

        # Store starting row for components
        summary_components_row = row

        # Define component rows (we'll conditionally add them)
        components_to_add = []

        # Always add equity
        components_to_add.append({
            'name': 'Equity G/L',
            'value': nav_data.get('Equity G/L', 0),
            'adjusted_name': 'Equity G/L (Adjusted)',
            'adjusted_value': nav_data.get('Equity G/L Adj', nav_data.get('Equity G/L', 0))
        })

        # Add option if non-zero
        option_gl = nav_data.get('Option G/L', 0)
        if abs(option_gl) > 0.01:
            components_to_add.append({
                'name': 'Option G/L',
                'value': option_gl,
                'adjusted_name': 'Option G/L (Adjusted)',
                'adjusted_value': nav_data.get('Option G/L Adj', option_gl)
            })

        # Add flex option if non-zero
        flex_gl = nav_data.get('Flex Option G/L', 0)
        if abs(flex_gl) > 0.01:
            components_to_add.append({
                'name': 'Flex Option G/L',
                'value': flex_gl,
                'adjusted_name': 'Flex Option G/L (Adjusted)',
                'adjusted_value': nav_data.get('Flex Option G/L Adj', flex_gl)
            })

        # Add treasury if non-zero
        treasury_gl = nav_data.get('Treasury G/L', 0)
        if abs(treasury_gl) > 0.01:
            components_to_add.append({
                'name': 'Treasury G/L',
                'value': treasury_gl,
                'adjusted_name': None,  # Treasury doesn't have adjusted version
                'adjusted_value': None
            })

        # Add assignment if non-zero
        assignment_gl = nav_data.get('Assignment G/L', 0)
        if abs(assignment_gl) > 0.01:
            components_to_add.append({
                'name': 'Assignment G/L',
                'value': assignment_gl,
                'adjusted_name': None,
                'adjusted_value': None
            })

        # Add other if non-zero
        other = nav_data.get('Other', 0)
        if abs(other) > 0.01:
            components_to_add.append({
                'name': 'Other',
                'value': other,
                'adjusted_name': None,
                'adjusted_value': None
            })

        # Now write the components
        component_rows = {}
        for comp in components_to_add:
            ws.cell(row=row, column=1, value=f"  {comp['name']}")
            ws.cell(row=row, column=3, value=comp['value']).number_format = self.number_format
            component_rows[comp['name']] = row
            row += 1

            if comp.get('adjusted_name'):
                ws.cell(row=row, column=1, value=f"  {comp['adjusted_name']}")
                ws.cell(row=row, column=3, value=comp['adjusted_value']).number_format = self.number_format
                component_rows[comp['adjusted_name']] = row
                row += 1

        row += 1  # Skip a row

        # Expenses (always show)
        ws.cell(row=row, column=1, value="Expenses")
        expenses_value = -abs(nav_data.get('Accruals', 0))  # Ensure negative
        ws.cell(row=row, column=3, value=expenses_value).number_format = self.number_format
        expenses_row = row
        row += 1

        # Dividends (show if non-zero)
        dividends = nav_data.get('Dividends', 0)
        if abs(dividends) > 0.01:
            ws.cell(row=row, column=1, value="Dividends")
            ws.cell(row=row, column=3, value=dividends).number_format = self.number_format
            dividends_row = row
            row += 1

        # Distributions (show if non-zero)
        distributions = nav_data.get('Distributions', 0)
        if abs(distributions) > 0.01:
            ws.cell(row=row, column=1, value="Distributions")
            ws.cell(row=row, column=3, value=-abs(distributions)).number_format = self.number_format
            distributions_row = row
            row += 1

        row += 1  # Skip a row

        # Expected TNA
        ws.cell(row=row, column=1, value="Expected TNA").font = self.subheader_font
        ws.cell(row=row, column=3, value=nav_data.get('Expected TNA', 0)).number_format = self.number_format
        expected_tna_row = row
        row += 1

        # Custodian TNA
        ws.cell(row=row, column=1, value="Custodian TNA").font = self.subheader_font
        ws.cell(row=row, column=3, value=nav_data.get('Custodian TNA', 0)).number_format = self.number_format
        cust_tna_row = row
        row += 1

        # TNA Difference
        ws.cell(row=row, column=1, value="TNA Difference")
        ws.cell(row=row, column=3, value=nav_data.get('TNA Diff ($)', 0)).number_format = self.number_format
        tna_diff_row = row
        row += 2

        # Shares Outstanding
        ws.cell(row=row, column=1, value="Shares Outstanding")
        ws.cell(row=row, column=3, value=nav_data.get('Shares Outstanding', 0)).number_format = '#,##0'
        shares_row = row
        row += 2

        # Expected NAV
        ws.cell(row=row, column=1, value="Expected NAV").font = self.subheader_font
        ws.cell(row=row, column=3, value=nav_data.get('Expected NAV', 0)).number_format = self.number_format_4
        expected_nav_row = row
        row += 1

        # Custodian NAV
        ws.cell(row=row, column=1, value="Custodian NAV").font = self.subheader_font
        ws.cell(row=row, column=3, value=nav_data.get('Custodian NAV', 0)).number_format = self.number_format_4
        cust_nav_row = row
        row += 1

        # NAV Difference
        ws.cell(row=row, column=1, value="NAV Difference")
        ws.cell(row=row, column=3, value=nav_data.get('NAV Diff ($)', 0)).number_format = self.number_format_4
        nav_diff_row = row
        row += 2

        # NAV Good indicators
        ws.cell(row=row, column=1, value="NAV Good (2 decimal)")
        nav_good_2 = nav_data.get('NAV Good (2 Digit)', False)
        ws.cell(row=row, column=3, value='PASS' if nav_good_2 else 'FAIL')
        if nav_good_2:
            ws.cell(row=row, column=3).fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
        else:
            ws.cell(row=row, column=3).fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")
        row += 1

        ws.cell(row=row, column=1, value="NAV Good (4 decimal)")
        nav_good_4 = nav_data.get('NAV Good (4 Digit)', False)
        ws.cell(row=row, column=3, value='PASS' if nav_good_4 else 'FAIL')
        if nav_good_4:
            ws.cell(row=row, column=3).fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
        else:
            ws.cell(row=row, column=3).fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")

        # Add detail sections
        detail_start_row = row + 5

        # Get detailed calculations from nav_data
        detailed_calcs = nav_data.get('detailed_calculations', {})

        # Add equity details if available
        equity_details = detailed_calcs.get('equity_details', pd.DataFrame())
        if not equity_details.empty:
            self._add_equity_details_section(ws, detail_start_row, 1, equity_details)
            detail_start_row = self._get_last_row(ws) + 3

        # Add option details if available and non-zero
        option_details = detailed_calcs.get('option_details', pd.DataFrame())
        if not option_details.empty:
            self._add_option_details_section(ws, detail_start_row, 1, option_details)
            detail_start_row = self._get_last_row(ws) + 3

        # Add flex option details if available and non-zero
        flex_details = detailed_calcs.get('flex_details', pd.DataFrame())
        if not flex_details.empty:
            self._add_flex_details_section(ws, detail_start_row, 1, flex_details)
            detail_start_row = self._get_last_row(ws) + 3

        # Auto-fit columns
        for col in range(1, 12):
            ws.column_dimensions[get_column_letter(col)].width = 12

        ws.freeze_panes = 'A4'

    def _add_equity_details_section(self, ws, start_row, start_col, equity_details):
        """Add equity details section showing only holdings with G/L."""
        ws.cell(row=start_row, column=start_col, value="EQUITY GAIN/LOSS DETAIL").font = self.header_font

        if equity_details.empty:
            ws.cell(row=start_row + 2, column=start_col, value="No equity holdings")
            return

        # Headers
        headers = [
            'Ticker', 'Qty T-1', 'Qty T',
            'Price T-1', 'Price T',
            'G/L'
        ]

        header_row = start_row + 2
        for i, header in enumerate(headers):
            cell = ws.cell(row=header_row, column=start_col + i, value=header)
            cell.font = self.subheader_font
            cell.alignment = Alignment(horizontal='center')

        # Data rows
        data_row = header_row + 1
        first_data_row = data_row

        for _, row in equity_details.iterrows():
            col = start_col

            # Ticker
            ws.cell(row=data_row, column=col, value=row.get('ticker', '')).alignment = Alignment(horizontal='left')
            col += 1

            # Quantities
            ws.cell(row=data_row, column=col, value=row.get('quantity_t1', 0)).number_format = '#,##0'
            col += 1
            ws.cell(row=data_row, column=col, value=row.get('quantity_t', 0)).number_format = '#,##0'
            col += 1

            # Prices (use adjusted if available)
            price_t1 = row.get('price_t1_adj', row.get('price_t1_raw', 0))
            price_t = row.get('price_t_adj', row.get('price_t_raw', 0))

            ws.cell(row=data_row, column=col, value=price_t1).number_format = self.number_format
            col += 1
            ws.cell(row=data_row, column=col, value=price_t).number_format = self.number_format
            col += 1

            # G/L
            gl = row.get('gl_adjusted', row.get('gl', 0))
            cell = ws.cell(row=data_row, column=col, value=gl)
            cell.number_format = self.number_format

            # Highlight negative G/L
            if gl < 0:
                cell.font = Font(color="FF0000")

            data_row += 1

        # Add total row
        if data_row > first_data_row:
            last_data_row = data_row - 1
            total_row = data_row + 1

            ws.cell(row=total_row, column=start_col, value="TOTAL").font = self.subheader_font

            # Sum formula for G/L
            gl_col = start_col + 5
            total_formula = f"=SUM({get_column_letter(gl_col)}{first_data_row}:{get_column_letter(gl_col)}{last_data_row})"
            ws.cell(row=total_row, column=gl_col, value=total_formula).number_format = self.number_format

    def _add_option_details_section(self, ws, start_row, start_col, option_details):
        """Add option details section showing only holdings with G/L."""
        ws.cell(row=start_row, column=start_col, value="OPTION GAIN/LOSS DETAIL").font = self.header_font

        if option_details.empty:
            ws.cell(row=start_row + 2, column=start_col, value="No option holdings")
            return

        # Headers
        headers = [
            'Ticker', 'Qty T-1', 'Qty T',
            'Price T-1', 'Price T',
            'G/L'
        ]

        header_row = start_row + 2
        for i, header in enumerate(headers):
            cell = ws.cell(row=header_row, column=start_col + i, value=header)
            cell.font = self.subheader_font
            cell.alignment = Alignment(horizontal='center')

        # Data rows
        data_row = header_row + 1
        first_data_row = data_row

        for _, row in option_details.iterrows():
            col = start_col

            # Ticker
            ws.cell(row=data_row, column=col, value=row.get('ticker', '')).alignment = Alignment(horizontal='left')
            col += 1

            # Quantities
            ws.cell(row=data_row, column=col, value=row.get('quantity_t1', 0)).number_format = '#,##0'
            col += 1
            ws.cell(row=data_row, column=col, value=row.get('quantity_t', 0)).number_format = '#,##0'
            col += 1

            # Prices
            price_t1 = row.get('price_t1_adj', row.get('price_t1_raw', 0))
            price_t = row.get('price_t_adj', row.get('price_t_raw', 0))

            ws.cell(row=data_row, column=col, value=price_t1).number_format = self.number_format
            col += 1
            ws.cell(row=data_row, column=col, value=price_t).number_format = self.number_format
            col += 1

            # G/L
            gl = row.get('gl_adjusted', row.get('gl', 0))
            cell = ws.cell(row=data_row, column=col, value=gl)
            cell.number_format = self.number_format

            # Highlight negative G/L
            if gl < 0:
                cell.font = Font(color="FF0000")

            data_row += 1

        # Add total row
        if data_row > first_data_row:
            last_data_row = data_row - 1
            total_row = data_row + 1

            ws.cell(row=total_row, column=start_col, value="TOTAL").font = self.subheader_font

            # Sum formula for G/L
            gl_col = start_col + 5
            total_formula = f"=SUM({get_column_letter(gl_col)}{first_data_row}:{get_column_letter(gl_col)}{last_data_row})"
            ws.cell(row=total_row, column=gl_col, value=total_formula).number_format = self.number_format

    def _add_flex_details_section(self, ws, start_row, start_col, flex_details):
        """Add flex option details section."""
        ws.cell(row=start_row, column=start_col, value="FLEX OPTION GAIN/LOSS DETAIL").font = self.header_font

        if flex_details.empty:
            ws.cell(row=start_row + 2, column=start_col, value="No flex option holdings")
            return

        # Same structure as regular options
        self._add_option_details_section(ws, start_row, start_col, flex_details)

    def _get_last_row(self, ws):
        """Get the last row with data in the worksheet."""
        return ws.max_row

    def _extract_equity_details(self, nav_data):
        """Extract equity details from nav_data with fallback logic."""
        # First try detailed_calculations - this is the primary source
        if 'detailed_calculations' in nav_data:
            detailed_calcs = nav_data['detailed_calculations']
            if isinstance(detailed_calcs, dict) and 'equity_details' in detailed_calcs:
                equity_df = detailed_calcs['equity_details']
                if isinstance(equity_df, pd.DataFrame) and not equity_df.empty:
                    return equity_df.to_dict('records')
                elif isinstance(equity_df, list):
                    return equity_df

        # Fallback to raw_equity if detailed_calculations not available
        if 'raw_equity' in nav_data:
            raw_df = nav_data['raw_equity']
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                # Convert raw data to expected format
                details = []
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

        return []

    def _extract_option_details(self, nav_data):
        """Extract option details (non-flex) from nav_data."""
        # First try detailed_calculations
        if 'detailed_calculations' in nav_data:
            detailed_calcs = nav_data['detailed_calculations']
            if isinstance(detailed_calcs, dict) and 'option_details' in detailed_calcs:
                option_df = detailed_calcs['option_details']
                if isinstance(option_df, pd.DataFrame) and not option_df.empty:
                    return option_df.to_dict('records')
                elif isinstance(option_df, list):
                    return option_df

        # Fallback to raw_option if detailed_calculations not available
        if 'raw_option' in nav_data:
            raw_df = nav_data['raw_option']
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                # Filter out flex options
                if 'optticker' in raw_df.columns:
                    non_flex = raw_df[~raw_df['optticker'].str.contains('SPX|XSP', na=False)]
                else:
                    non_flex = raw_df

                details = []
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

        return []

    def _extract_flex_details(self, nav_data, fund_name: Optional[str] = None):
        """Extract flex option details from nav_data."""
        # First try detailed_calculations
        if 'detailed_calculations' in nav_data:
            detailed_calcs = nav_data['detailed_calculations']
            if isinstance(detailed_calcs, dict) and 'flex_details' in detailed_calcs:
                flex_df = detailed_calcs['flex_details']
                if isinstance(flex_df, pd.DataFrame) and not flex_df.empty:
                    return flex_df.to_dict('records')
                elif isinstance(flex_df, list):
                    return flex_df

        # Fallback to raw_option filtered for flex
        inferred_fund = fund_name or nav_data.get('fund') or nav_data.get('fund_name')
        uses_index_flex = self._uses_index_flex(inferred_fund)
        if not uses_index_flex:
            return []

        if 'raw_option' in nav_data:
            raw_df = nav_data['raw_option']
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                # Filter for flex options only
                if 'optticker' in raw_df.columns:
                    flex_only = raw_df[raw_df['optticker'].str.contains('SPX|XSP', na=False)]
                else:
                    flex_only = pd.DataFrame()

                details = []
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

        return []

    # ------------------------------------------------------------------
    def _write_summary_row(
        self,
        worksheet,
        row: int,
        label: str,
        value: float,
        *,
        number_format: str,
        bold: bool = False,
    ) -> None:
        label_cell = worksheet.cell(row=row, column=1, value=label)
        label_cell.font = self.subheader_font if bold else self.body_font
        value_cell = worksheet.cell(row=row, column=3, value=value)
        value_cell.number_format = number_format

    # ------------------------------------------------------------------
    def _add_component_detail_section(
        self,
        worksheet,
        start_row: int,
        title: str,
        dataframe: pd.DataFrame,
    ) -> Dict[str, Any]:
        title_cell = worksheet.cell(row=start_row, column=1, value=f"{title.upper()} DETAIL")
        title_cell.font = self.header_font

        if dataframe.empty:
            worksheet.cell(row=start_row + 2, column=1, value=f"No {title.lower()} data available")
            return {"end_row": start_row + 2, "raw_total": None, "adjusted_total": None}

        column_specs = [
            ("Ticker", ["ticker", "symbol", "security", "identifier"]),
            ("Qty T-1", ["quantity_t1", "qty_t1", "prior_quantity", "quantity_prev"]),
            ("Qty T", ["quantity_t", "qty_t", "quantity", "current_quantity"]),
            ("Price T-1", ["price_t1", "prior_price", "previous_price"]),
            ("Price T", ["price_t", "price", "current_price"]),
            ("Raw G/L", ["raw_gl", "gl_raw", "gain_loss_raw", "raw_gain"]),
            ("Adjusted G/L", ["adjusted_gl", "gl_adj", "gain_loss_adjusted", "adjusted_gain"]),
        ]

        header_row = start_row + 2
        for index, (header, _) in enumerate(column_specs, start=1):
            header_cell = worksheet.cell(row=header_row, column=index, value=header)
            header_cell.font = self.subheader_font
            header_cell.alignment = Alignment(horizontal="center")
            header_cell.fill = PatternFill(start_color="E6F0FF", end_color="E6F0FF", fill_type="solid")

        first_data_row = header_row + 1
        current_row = first_data_row

        records = dataframe.to_dict("records")
        for record in records:
            lowered = {str(key).lower(): record[key] for key in record}
            for column_index, (header, candidates) in enumerate(column_specs, start=1):
                value = self._first_available_value(lowered, candidates)
                cell = worksheet.cell(row=current_row, column=column_index, value=value)
                if header.startswith("Qty"):
                    cell.number_format = self.integer_format
                elif "Price" in header or "G/L" in header:
                    cell.number_format = self.currency_format
                else:
                    cell.number_format = "@"
            current_row += 1

        last_data_row = current_row - 1
        total_row = current_row

        total_label = worksheet.cell(row=total_row, column=1, value="TOTAL")
        total_label.font = self.subheader_font

        raw_total_cell = None
        adjusted_total_cell = None
        for column_index, (header, _) in enumerate(column_specs, start=1):
            if header.endswith("G/L"):
                column_letter = get_column_letter(column_index)
                total_cell = worksheet.cell(
                    row=total_row,
                    column=column_index,
                    value=f"=SUM({column_letter}{first_data_row}:{column_letter}{last_data_row})",
                )
                total_cell.number_format = self.currency_format
                if header.startswith("Raw"):
                    raw_total_cell = f"{column_letter}{total_row}"
                elif header.startswith("Adjusted"):
                    adjusted_total_cell = f"{column_letter}{total_row}"

        return {
            "end_row": total_row,
            "raw_total": raw_total_cell,
            "adjusted_total": adjusted_total_cell,
        }

    # ------------------------------------------------------------------
    def _create_summary_sheet(self, workbook: Workbook):
        worksheet = workbook.create_sheet("NAV Summary")
        worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=13)
        title_cell = worksheet.cell(
            row=1,
            column=1,
            value=f"NAV Reconciliation Summary - {self.report_date}",
        )
        title_cell.font = self.title_font
        title_cell.alignment = Alignment(horizontal="center")

        headers = [
            "Fund",
            "Prior NAV",
            "Equity G/L",
            "Option G/L",
            "Flex Option G/L",
            "Treasury G/L",
            "Net Gain/Loss",
            "Dividends",
            "Expenses",
            "Distributions",
            "Flow Adjustment",
            "Expected NAV",
            "Custodian NAV",
            "Variance",
        ]

        header_row = 3
        for index, header in enumerate(headers, start=1):
            cell = worksheet.cell(row=header_row, column=index, value=header)
            cell.font = self.header_font
            cell.alignment = Alignment(horizontal="center")
            cell.fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")

        current_row = header_row + 1
        for fund_name in sorted(self._sheet_refs):
            refs = self._sheet_refs[fund_name]
            sheet_name = refs.get("sheet_name", "")
            worksheet.cell(row=current_row, column=1, value=fund_name)

            def sheet_ref(cell_ref: str) -> str:
                escaped = sheet_name.replace("'", "''")
                return f"='{escaped}'!{cell_ref}"

            column_map = [
                refs.get("prior_nav"),
                refs.get("components", {}).get("equity"),
                refs.get("components", {}).get("options"),
                refs.get("components", {}).get("flex_options"),
                refs.get("components", {}).get("treasury"),
                refs.get("net_gain"),
                refs.get("dividends"),
                refs.get("expenses"),
                refs.get("distributions"),
                refs.get("flow_adjustment"),
                refs.get("expected_nav"),
                refs.get("cust_nav"),
                refs.get("variance"),
            ]

            for offset, cell_reference in enumerate(column_map, start=2):
                cell = worksheet.cell(row=current_row, column=offset)
                if cell_reference:
                    cell.value = sheet_ref(cell_reference)
                else:
                    cell.value = 0.0
                if offset in {2, 12, 13, 14}:
                    cell.number_format = self.nav_format
                else:
                    cell.number_format = self.currency_format

            current_row += 1

        for column_index in range(1, len(headers) + 1):
            worksheet.column_dimensions[get_column_letter(column_index)].width = 18

        return worksheet

    # ------------------------------------------------------------------
    def _unique_sheet_name(self, fund: str) -> str:
        sanitized = re.sub(r"[\\/*?\[\]:]", " ", fund).strip() or "Fund"
        base = sanitized[:31]
        if base not in self._sheet_names:
            self._sheet_names.add(base)
            return base

        counter = 2
        while True:
            suffix = f"_{counter}"
            candidate = f"{base[: max(0, 31 - len(suffix))]}{suffix}"
            if candidate not in self._sheet_names:
                self._sheet_names.add(candidate)
                return candidate
            counter += 1

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _first_available_value(row: Dict[str, Any], candidates: list[str]) -> Any:
        for candidate in candidates:
            if candidate in row and row[candidate] not in (None, ""):
                return row[candidate]
        return ""

class NAVReconciliationPDF(BaseReportPDF):
    """Generate a PDF summary for NAV reconciliation results."""

    def __init__(
        self,
        output_path: str,
        report_date: str,
        results: Mapping[str, Any],
    ) -> None:
        super().__init__(output_path)
        self.report_date = report_date
        self.results = normalize_nav_payload(results)

    def render(self) -> None:
        self.add_title("NAV Reconciliation Summary")
        self.add_subtitle(f"As of {self.report_date}")

        totals = summarise_nav_differences(self.results)
        if totals["funds"]:
            avg_diff = totals["absolute_difference"] / totals["funds"] if totals["funds"] else 0.0
            self.add_section_heading("Portfolio Overview")
            self.add_key_value_table(
                [
                    ("Funds Analysed", totals["funds"]),
                    ("Total Absolute Variance", format_number(totals["absolute_difference"], 4)),
                    ("Average Absolute Variance", format_number(avg_diff, 4)),
                ],
                header=("Metric", "Value"),
            )

        for fund_name, payload in sorted(self.results.items()):
            summary = payload.get("summary", {})
            if not summary:
                continue
            self.add_section_heading(fund_name)
            rows = [
                ("Prior NAV", format_number(summary.get("prior_nav"), 4)),
                ("Custodian NAV", format_number(summary.get("current_nav"), 4)),
                ("Expected NAV", format_number(summary.get("expected_nav"), 4)),
                ("Variance", format_number(summary.get("difference"), 4)),
                ("Net Gain/Loss", format_number(summary.get("net_gain"), 4)),
                ("Dividends", format_number(summary.get("dividends"), 4)),
                ("Expenses", format_number(summary.get("expenses"), 4)),
                ("Distributions", format_number(summary.get("distributions"), 4)),
                ("Flow Adjustment", format_number(summary.get("flows_adjustment"), 4)),
            ]
            self.add_key_value_table(rows, header=("Metric", "Value"))
            self._add_component_sections(payload.get("details", {}))

        self.output()


    # ------------------------------------------------------------------
    def _add_component_sections(self, details: Mapping[str, Any]) -> None:
        component_titles = [
            ("equity", "Equity Holdings"),
            ("options", "Option Holdings"),
            ("flex_options", "Flex Option Holdings"),
            ("treasury", "Treasury Holdings"),
        ]

        for key, title in component_titles:
            self.add_subtitle(title)
            dataframe = ensure_dataframe(details.get(key))
            if dataframe.empty:
                self.add_paragraph("No data available.", font_size=8)
                continue

            headers, rows = self._build_component_table(dataframe)
            self.add_table(
                headers,
                rows,
                column_widths=[35, 22, 22, 28, 28, 32, 32],
                align=["L", "R", "R", "R", "R", "R", "R"],  # Changed from 'alignments'
            )


    # ------------------------------------------------------------------
    def _build_component_table(self, dataframe: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
        columns = self._resolve_component_columns(dataframe)
        headers = [
            "Ticker",
            "Qty T-1",
            "Qty T",
            "Price T-1",
            "Price T",
            "Raw G/L",
            "Adjusted G/L",
        ]

        rows: list[list[str]] = []
        raw_totals: list[float] = []
        adjusted_totals: list[float] = []

        for _, row in dataframe.iterrows():
            ticker = row.get(columns["ticker"]) if columns["ticker"] else ""
            qty_t1 = self._to_float(row, columns["qty_t1"])
            qty_t = self._to_float(row, columns["qty_t"])
            price_t1 = self._to_float(row, columns["price_t1"])
            price_t = self._to_float(row, columns["price_t"])
            raw_gl = self._to_float(row, columns["raw_gl"])
            adj_gl = self._to_float(row, columns["adjusted_gl"])

            raw_totals.append(raw_gl)
            adjusted_totals.append(adj_gl)

            rows.append(
                [
                    ticker or "",
                    format_number(qty_t1, 0),
                    format_number(qty_t, 0),
                    format_number(price_t1, 4),
                    format_number(price_t, 4),
                    format_number(raw_gl, 2),
                    format_number(adj_gl, 2),
                ]
            )

        totals_row = [
            "TOTAL",
            "",
            "",
            "",
            "",
            format_number(sum(raw_totals), 2),
            format_number(sum(adjusted_totals), 2),
        ]
        rows.append(totals_row)

        return headers, rows

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_component_columns(dataframe: pd.DataFrame) -> Dict[str, Optional[str]]:
        lowered = {str(column).lower(): column for column in dataframe.columns}

        def pick(*candidates: str) -> Optional[str]:
            for candidate in candidates:
                if candidate in lowered:
                    return lowered[candidate]
            return None

        return {
            "ticker": pick("ticker", "symbol", "security", "identifier"),
            "qty_t1": pick("quantity_t1", "qty_t1", "prior_quantity", "quantity_prev"),
            "qty_t": pick("quantity_t", "qty_t", "quantity", "current_quantity"),
            "price_t1": pick("price_t1", "prior_price", "previous_price"),
            "price_t": pick("price_t", "price", "current_price"),
            "raw_gl": pick("raw_gl", "gl_raw", "gain_loss_raw", "raw_gain"),
            "adjusted_gl": pick("adjusted_gl", "gl_adj", "gain_loss_adjusted", "adjusted_gain"),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _to_float(row: pd.Series, column: Optional[str]) -> float:
        if not column:
            return 0.0
        value = row.get(column)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0




class DailyOperationsSummaryPDF(BaseReportPDF):
    """Combined summary across compliance, reconciliation, and NAV."""

    def __init__(
        self,
        output_path: str,
        report_date: str,
        compliance_results: Mapping[str, Any],
        reconciliation_results: Mapping[str, Any],
        nav_results: Mapping[str, Any],
    ) -> None:
        super().__init__(output_path)
        self.report_date = report_date
        self.compliance_results = normalize_compliance_results(compliance_results or {})
        self.reconciliation_results = normalize_reconciliation_payload(reconciliation_results or {})
        self.nav_results = normalize_nav_payload(nav_results or {})

    def render(self) -> None:
        # FIX: Use separate calls for title and subtitle
        self.add_title("Daily Oversight Summary")
        self.add_subtitle(f"As of {self.report_date}")

        if self.compliance_results:
            compliance_summary = summarise_compliance_status(self.compliance_results)
            self.add_section_heading("Compliance Overview")
            rows = [
                ("Funds Processed", compliance_summary["funds"]),
                ("Funds with Breaches", compliance_summary["funds_in_breach"]),
                ("Checks Evaluated", compliance_summary["total_checks"]),
                ("Checks Failed", compliance_summary["failed_checks"]),
            ]
            self.add_key_value_table(rows, header=("Metric", "Value"))

        if self.reconciliation_results:
            totals = summarise_reconciliation_breaks(self.reconciliation_results)
            self.add_section_heading("Holdings Reconciliation")
            rows = [
                (name.replace("_", " ").title(), totals.get(name, 0))
                for name in sorted(totals)
            ]
            if rows:
                # FIX: Use 'align' instead of 'alignments'
                self.add_table([
                    "Reconciliation",
                    "Total Breaks",
                ], rows, column_widths=[100, 40], align=["L", "R"])

            for fund_name, payload in sorted(self.reconciliation_results.items()):
                summary = payload.get("summary", {})
                if not summary:
                    continue
                self.add_section_heading(f"{fund_name} - Breaks")
                fund_rows = []
                for recon_type, metrics in sorted(summary.items()):
                    total_breaks = sum(
                        int(value)
                        for value in (metrics or {}).values()
                        if isinstance(value, (int, float))
                    )
                    fund_rows.append((recon_type.replace("_", " ").title(), total_breaks))
                if fund_rows:
                    self.add_key_value_table(fund_rows, header=("Reconciliation", "Breaks"))

        if self.nav_results:
            self.add_section_heading("NAV Reconciliation")
            totals = summarise_nav_differences(self.nav_results)
            avg_diff = (
                totals["absolute_difference"] / totals["funds"]
                if totals["funds"]
                else 0.0
            )
            self.add_key_value_table(
                [
                    ("Funds Analysed", totals["funds"]),
                    ("Total Absolute Variance", format_number(totals["absolute_difference"], 4)),
                    ("Average Absolute Variance", format_number(avg_diff, 4)),
                ],
                header=("Metric", "Value"),
            )

            detail_rows = []
            for fund_name, payload in sorted(self.nav_results.items()):
                summary = payload.get("summary", {})
                if not summary:
                    continue
                detail_rows.append(
                    (
                        fund_name,
                        format_number(summary.get("expected_nav"), 4),
                        format_number(summary.get("current_nav"), 4),
                        format_number(summary.get("difference"), 4),
                    )
                )
            if detail_rows:
                # FIX: Use 'align' instead of 'alignments'
                self.add_table(
                    ["Fund", "Expected NAV", "Custodian NAV", "Variance"],
                    detail_rows,
                    column_widths=[70, 35, 35, 35],
                    align=["L", "R", "R", "R"],
                )

        self.output()


def generate_nav_reconciliation_reports(reconciliation_results, date_str, excel_path):
    """
    Generate NAV reconciliation reports in Excel format.

    Args:
        reconciliation_results: Dict with structure {date_str: {fund_name: nav_results}}
        date_str: Date string for the report
        excel_path: Path where the Excel file will be saved
    """
    from pathlib import Path

    # Extract the funds data for the specific date
    # The NAVReconciliationExcelReport expects {fund_name: nav_data} structure
    if isinstance(reconciliation_results, dict):
        # Check if it's nested with dates
        if date_str in reconciliation_results:
            # Extract funds for the specific date
            normalized = reconciliation_results[date_str]
        else:
            # Maybe it's already {fund: data} without date nesting
            # Check if the first value is a dict with NAV data
            first_key = next(iter(reconciliation_results.keys())) if reconciliation_results else None
            if first_key and isinstance(reconciliation_results[first_key], dict):
                # Check if it looks like NAV data (has expected keys)
                first_value = reconciliation_results[first_key]
                if any(key in first_value for key in ['NAV Good (2 Digit)', 'NAV Good (4 Digit)', 'NAV Diff ($)', 'Expected NAV', 'Custodian NAV']):
                    # It's already {fund: nav_data}
                    normalized = reconciliation_results
                else:
                    # It's {date: {fund: data}}, take the first/only date's data
                    normalized = reconciliation_results.get(first_key, {})
            else:
                normalized = reconciliation_results
    else:
        normalized = {}

    # Create the Excel report using NAVReconciliationExcelReport
    NAVReconciliationExcelReport(
        results=normalized,  # {fund_name: nav_data}
        report_date=date_str,  # Date string
        output_path=Path(excel_path)  # Path object
    )

    return excel_path

def generate_reconciliation_summary_pdf(
    reconciliation_results: Mapping[str, Any],
    nav_results: Mapping[str, Any],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    file_name: str = "reconciliation_summary.pdf",
) -> Optional[str]:
    """Create the combined holdings + NAV reconciliation PDF."""

    recon_payload = normalize_reconciliation_payload(reconciliation_results)
    nav_payload = normalize_nav_payload(nav_results)
    if not recon_payload and not nav_payload:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path / file_name

    date_str = normalize_report_date(report_date)

    holdings_details: Dict[str, Dict[str, Any]] = {date_str: {}}
    holdings_summary: Dict[str, list[Dict[str, Any]]] = {date_str: []}
    for fund, payload in recon_payload.items():
        holdings_details[date_str][fund] = payload.get("details", {})
        holdings_summary[date_str].append({
            "fund": fund,
            "summary": payload.get("summary", {}),
        })

    nav_details: Dict[str, Dict[str, Any]] = {date_str: {}}
    nav_summary: Dict[str, list[Dict[str, Any]]] = {date_str: []}
    for fund, payload in nav_payload.items():
        nav_details[date_str][fund] = payload.get("summary", {})
        nav_summary[date_str].append({
            "fund": fund,
            "summary": payload.get("summary", {}),
        })

    build_combined_reconciliation_pdf(
        nav_details,
        holdings_details,
        nav_summary,
        holdings_summary,
        date_str,
        pdf_path,
    )


    return str(pdf_path)


def generate_daily_operations_pdf(
    compliance_results: Mapping[str, Any],
    reconciliation_results: Mapping[str, Any],
    nav_results: Mapping[str, Any],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    file_name: str = "daily_operations_summary.pdf",
) -> Optional[str]:
    """Create the combined compliance + reconciliation + NAV PDF summary."""

    if not any([compliance_results, reconciliation_results, nav_results]):
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path / file_name

    pdf = DailyOperationsSummaryPDF(
        str(pdf_path),
        normalize_report_date(report_date),
        compliance_results,
        reconciliation_results,
        nav_results,
    )
    pdf.render()
    return str(pdf_path)