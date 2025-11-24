from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
import re

from reporting.combined_reconciliation_report import build_combined_reconciliation_pdf
from reporting.report_utils import (
    _normalize_nav_payload,
    _normalize_reconciliation_payload,
    normalize_report_date,
)


@dataclass
class GeneratedNAVReconciliationReport:
    """File locations for generated NAV reconciliation artefacts."""

    excel_path: Optional[str]


class NAVReconciliationExcelReport:
    """Render NAV reconciliation data into an Excel workbook with formulas and ticker details."""

    def __init__(
        self,
        results: Mapping[str, Any],
        report_date: str,
        output_path: Path,
        workbook: Workbook | None = None,
    ) -> None:
        self.results = self._normalize_nav_payload(results)
        self.report_date = report_date
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.workbook = workbook
        self._using_existing_workbook = workbook is not None
        # Style definitions
        self.title_font = Font(bold=True, size=14)
        self.header_font = Font(bold=True, size=12)
        self.subheader_font = Font(bold=True, size=10)
        self.body_font = Font(size=10)
        self.currency_format = "#,##0.00"
        self.nav_format = "#,##0.0000"
        self.integer_format = "#,##0"
        self.percent_format = "0.00%"

        self._sheet_names: set[str] = set()
        self._sheet_refs: Dict[str, Dict[str, Any]] = {}

        self._export()

    def _normalize_nav_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Normalize the NAV payload to ensure consistent structure.

        Unwraps date-keyed structure if present and returns fund-level data.
        """
        if not payload:
            return {}

        # Detect and unwrap date-keyed structure if present
        first_key = next(iter(payload.keys()))
        first_value = payload[first_key]

        # Check if this is date-wrapped: {date: {fund: payload}}
        if isinstance(first_value, dict) and first_value:
            sample_nested_key = next(iter(first_value.keys()))
            sample_nested_value = first_value.get(sample_nested_key)

            # If nested value has NAV-specific keys, parent is date wrapper
            if isinstance(sample_nested_value, dict) and any(
                    key in sample_nested_value
                    for key in ["Beginning TNA", "Expected NAV", "Custodian NAV",
                                "Expected TNA", "NAV Diff ($)", "summary", "detailed_calculations"]
            ):
                # This is date-wrapped, unwrap to fund level
                fund_level_data = first_value
            else:
                # Already fund-level
                fund_level_data = payload
        else:
            # Already fund-level
            fund_level_data = payload

        # Return as dict, ensuring we have fund-level data
        normalized: Dict[str, Any] = {}
        for fund_name, fund_data in fund_level_data.items():
            if isinstance(fund_data, dict):
                normalized[fund_name] = fund_data

        return normalized

    def _export(self) -> None:
        """Export the workbook with all sheets."""
        workbook = self.workbook or Workbook()

        # Remove default sheet for new workbooks only
        if not self._using_existing_workbook and workbook.active:
            workbook.remove(workbook.active)

        # Create fund sheets
        for fund_name, payload in sorted(self.results.items()):
            if not payload:
                continue
            self._create_fund_sheet_with_formulas(workbook, fund_name, payload, self.report_date)

        # Create summary sheet
        summary_ws = self._create_summary_sheet(workbook)
        if summary_ws is not None and not self._using_existing_workbook:
            # Move summary to first position only when creating a new workbook
            workbook._sheets.insert(0, workbook._sheets.pop(workbook._sheets.index(summary_ws)))

        workbook.save(self.output_path)

    def _create_fund_sheet_with_formulas(self, wb, fund_name, nav_data, date_str):
        """Create detailed sheet for a fund with formulas matching NAVReconciliationReport."""
        ws = wb.create_sheet(fund_name[:31])  # Excel sheet names limited to 31 chars

        # Store sheet reference
        self._sheet_refs[fund_name] = {
            "sheet_name": fund_name[:31],
            "components": {},
        }

        # Title
        ws.merge_cells("A1:C1")
        ws["A1"] = f"{fund_name} - NAV Reconciliation - {date_str}"
        ws["A1"].font = self.title_font
        ws["A1"].alignment = Alignment(horizontal="center")

        # SECTION 1: Summary Calculations
        row = 3
        ws.cell(row=row, column=1, value="NAV CALCULATION SUMMARY").font = self.header_font
        row += 2

        # Beginning TNA
        ws.cell(row=row, column=1, value="Beginning TNA (T-1)")
        ws.cell(row=row, column=3, value=nav_data.get("Beginning TNA", 0)).number_format = self.currency_format
        beg_tna_row = row
        row += 1

        # Adjusted Beginning TNA
        ws.cell(row=row, column=1, value="Adjusted Beginning TNA")
        ws.cell(row=row, column=3, value=nav_data.get("Adjusted Beginning TNA", nav_data.get("Beginning TNA", 0))).number_format = self.currency_format
        adj_beg_tna_row = row
        row += 2

        # Gain/Loss Components header
        ws.cell(row=row, column=1, value="Gain/Loss Components:").font = self.subheader_font
        row += 1

        component_row_refs: Dict[str, str] = {}

        # Jump ahead to add detail sections first
        detail_start_row = row + 30  # Leave space for summary section

        # Add equity details and get cell references
        equity_detail_info = self._add_equity_details_with_formulas(ws, detail_start_row, nav_data)

        # Add option details
        option_detail_info = self._add_option_details_with_formulas(
            ws, equity_detail_info["end_row"] + 3, nav_data
        )

        # Add flex option details
        flex_detail_info = self._add_flex_option_details_with_formulas(
            ws, option_detail_info["end_row"] + 3, nav_data
        )

        # Now go back and fill in summary section with proper references
        # Equity G/L
        ws.cell(row=row, column=1, value="  Equity G/L")
        if equity_detail_info["has_data"] and equity_detail_info["gl_total_cell"]:
            ws.cell(row=row, column=3, value=f"={equity_detail_info['gl_total_cell']}")
        else:
            ws.cell(row=row, column=3, value=nav_data.get("Equity G/L", 0))
        ws.cell(row=row, column=3).number_format = self.currency_format
        component_row_refs["equity"] = f"C{row}"
        self._sheet_refs[fund_name]["components"]["equity"] = f"C{row}"
        row += 1

        # Equity G/L Adjusted
        ws.cell(row=row, column=1, value="  Equity G/L (Adjusted)")
        if equity_detail_info["has_data"] and equity_detail_info["gl_adj_total_cell"]:
            ws.cell(row=row, column=3, value=f"={equity_detail_info['gl_adj_total_cell']}")
        else:
            ws.cell(row=row, column=3, value=nav_data.get("Equity G/L Adj", nav_data.get("Equity G/L", 0)))
        ws.cell(row=row, column=3).number_format = self.currency_format
        row += 1

        # Option G/L (if non-zero)
        option_gl = nav_data.get("Option G/L", 0)
        if abs(option_gl) > 0.01 or option_detail_info["has_data"]:
            ws.cell(row=row, column=1, value="  Option G/L")
            if option_detail_info["has_data"] and option_detail_info["gl_total_cell"]:
                ws.cell(row=row, column=3, value=f"={option_detail_info['gl_total_cell']}")
            else:
                ws.cell(row=row, column=3, value=option_gl)
            ws.cell(row=row, column=3).number_format = self.currency_format
            component_row_refs["option"] = f"C{row}"
            self._sheet_refs[fund_name]["components"]["options"] = f"C{row}"
            row += 1

            ws.cell(row=row, column=1, value="  Option G/L (Adjusted)")
            if option_detail_info["has_data"] and option_detail_info["gl_adj_total_cell"]:
                ws.cell(row=row, column=3, value=f"={option_detail_info['gl_adj_total_cell']}")
            else:
                ws.cell(row=row, column=3, value=nav_data.get("Option G/L Adj", option_gl))
            ws.cell(row=row, column=3).number_format = self.currency_format
            row += 1

        # Flex Option G/L (if non-zero)
        flex_gl = nav_data.get("Flex Option G/L", 0)
        if abs(flex_gl) > 0.01 or flex_detail_info["has_data"]:
            ws.cell(row=row, column=1, value="  Flex Option G/L")
            if flex_detail_info["has_data"] and flex_detail_info["gl_total_cell"]:
                ws.cell(row=row, column=3, value=f"={flex_detail_info['gl_total_cell']}")
            else:
                ws.cell(row=row, column=3, value=flex_gl)
            ws.cell(row=row, column=3).number_format = self.currency_format
            component_row_refs["flex"] = f"C{row}"
            self._sheet_refs[fund_name]["components"]["flex_options"] = f"C{row}"
            row += 1

            ws.cell(row=row, column=1, value="  Flex Option G/L (Adjusted)")
            if flex_detail_info["has_data"] and flex_detail_info["gl_adj_total_cell"]:
                ws.cell(row=row, column=3, value=f"={flex_detail_info['gl_adj_total_cell']}")
            else:
                ws.cell(row=row, column=3, value=nav_data.get("Flex Option G/L Adj", flex_gl))
            ws.cell(row=row, column=3).number_format = self.currency_format
            row += 1

        # Treasury G/L
        treasury_gl = nav_data.get("Treasury G/L", 0)
        if abs(treasury_gl) > 0.01:
            ws.cell(row=row, column=1, value="  Treasury G/L")
            ws.cell(row=row, column=3, value=treasury_gl).number_format = self.currency_format
            component_row_refs["treasury"] = f"C{row}"
            self._sheet_refs[fund_name]["components"]["treasury"] = f"C{row}"
            row += 1

        # Assignment G/L
        assignment_gl = nav_data.get("Assignment G/L", 0)
        if abs(assignment_gl) > 0.01:
            ws.cell(row=row, column=1, value="  Assignment G/L")
            ws.cell(row=row, column=3, value=assignment_gl).number_format = self.currency_format
            row += 1

        # Other
        other = nav_data.get("Other", 0)
        if abs(other) > 0.01:
            ws.cell(row=row, column=1, value="  Other")
            ws.cell(row=row, column=3, value=other).number_format = self.currency_format
            row += 1

        row += 1  # Skip a row

        # Expenses
        ws.cell(row=row, column=1, value="Expenses")
        expenses_value = -abs(nav_data.get("Accruals", 0))
        ws.cell(row=row, column=3, value=expenses_value).number_format = self.currency_format
        expenses_row = row
        self._sheet_refs[fund_name]["expenses"] = f"C{row}"
        row += 1

        # Dividends
        dividends = nav_data.get("Dividends", 0)
        if abs(dividends) > 0.01:
            ws.cell(row=row, column=1, value="Dividends")
            ws.cell(row=row, column=3, value=dividends).number_format = self.currency_format
            dividends_row = row
            self._sheet_refs[fund_name]["dividends"] = f"C{row}"
            row += 1
        else:
            dividends_row = None

        # Distributions
        distributions = nav_data.get("Distributions", 0)
        if abs(distributions) > 0.01:
            ws.cell(row=row, column=1, value="Distributions")
            ws.cell(row=row, column=3, value=-abs(distributions)).number_format = self.currency_format
            distributions_row = row
            self._sheet_refs[fund_name]["distributions"] = f"C{row}"
            row += 1
        else:
            distributions_row = None

        row += 1  # Skip a row

        # Expected TNA with formula
        ws.cell(row=row, column=1, value="Expected TNA").font = self.subheader_font

        formula_parts = [f"C{adj_beg_tna_row}"]

        # Add adjusted G/L components
        for component_key in (equity_detail_info.get("gl_adj_total_cell"), option_detail_info.get("gl_adj_total_cell"), flex_detail_info.get("gl_adj_total_cell")):
            if component_key:
                formula_parts.append(component_key)

        if "treasury" in component_row_refs:
            formula_parts.append(component_row_refs["treasury"])

        if abs(assignment_gl) > 0.01:
            formula_parts.append(str(nav_data.get("Assignment G/L", 0)))
        if abs(other) > 0.01:
            formula_parts.append(str(other))

        formula_parts.append(f"C{expenses_row}")

        if dividends_row:
            formula_parts.append(f"C{dividends_row}")
        if distributions_row:
            formula_parts.append(f"C{distributions_row}")

        formula = "=" + "+".join(formula_parts)
        ws.cell(row=row, column=3, value=formula).number_format = self.currency_format
        expected_tna_row = row
        self._sheet_refs[fund_name]["expected_tna"] = f"C{row}"
        row += 1

        # Custodian TNA
        ws.cell(row=row, column=1, value="Custodian TNA").font = self.subheader_font
        ws.cell(row=row, column=3, value=nav_data.get("Custodian TNA", 0)).number_format = self.currency_format
        cust_tna_row = row
        self._sheet_refs[fund_name]["cust_tna"] = f"C{row}"
        row += 1

        # TNA Difference with formula
        ws.cell(row=row, column=1, value="TNA Difference")
        ws.cell(row=row, column=3, value=f"=C{cust_tna_row}-C{expected_tna_row}").number_format = self.currency_format
        row += 2

        # Shares Outstanding
        ws.cell(row=row, column=1, value="Shares Outstanding")
        ws.cell(row=row, column=3, value=nav_data.get("Shares Outstanding", 0)).number_format = "#,##0"
        shares_row = row
        row += 2

        # Expected NAV with formula
        ws.cell(row=row, column=1, value="Expected NAV").font = self.subheader_font
        ws.cell(row=row, column=3, value=f"=IF(C{shares_row}<>0,C{expected_tna_row}/C{shares_row},0)").number_format = self.nav_format
        expected_nav_row = row
        self._sheet_refs[fund_name]["expected_nav"] = f"C{row}"
        row += 1

        # Custodian NAV
        ws.cell(row=row, column=1, value="Custodian NAV").font = self.subheader_font
        ws.cell(row=row, column=3, value=nav_data.get("Custodian NAV", 0)).number_format = self.nav_format
        cust_nav_row = row
        self._sheet_refs[fund_name]["cust_nav"] = f"C{row}"
        row += 1

        # NAV Difference with formula
        ws.cell(row=row, column=1, value="NAV Difference")
        ws.cell(row=row, column=3, value=f"=C{cust_nav_row}-C{expected_nav_row}").number_format = self.nav_format
        nav_diff_row = row
        self._sheet_refs[fund_name]["variance"] = f"C{row}"
        row += 2

        # NAV Good indicators with formulas
        ws.cell(row=row, column=1, value="NAV Good (2 decimal)")
        nav_good_2_formula = f"=IF(ABS(ROUND(C{nav_diff_row},2))<=0.01,\"PASS\",\"FAIL\")"
        ws.cell(row=row, column=3, value=nav_good_2_formula)
        row += 1

        ws.cell(row=row, column=1, value="NAV Good (4 decimal)")
        nav_good_4_formula = f"=IF(ABS(ROUND(C{nav_diff_row},4))<=0.0001,\"PASS\",\"FAIL\")"
        ws.cell(row=row, column=3, value=nav_good_4_formula)

        # Store prior NAV reference
        self._sheet_refs[fund_name]["prior_nav"] = f"C{beg_tna_row}"

        # Auto-fit columns
        for col in range(1, 12):
            ws.column_dimensions[get_column_letter(col)].width = 14

        ws.freeze_panes = "A4"

    def _add_equity_details_with_formulas(self, ws, start_row, nav_data):
        """Add equity details section with formulas for G/L calculations."""
        ws.cell(row=start_row, column=1, value="EQUITY GAIN/LOSS DETAIL").font = self.header_font

        detailed_calcs = nav_data.get("detailed_calculations", {})
        equity_details = detailed_calcs.get("equity_details", pd.DataFrame())

        if equity_details.empty:
            ws.cell(row=start_row + 2, column=1, value="No equity holdings with gain/loss")
            return {
                "has_data": False,
                "gl_total_cell": None,
                "gl_adj_total_cell": None,
                "end_row": start_row + 2,
            }

        headers = [
            "Ticker",
            "Qty T-1",
            "Qty T",
            "Price T-1\n(Vest)",
            "Price T\n(Raw)",
            "Price T-1\n(Cust)",
            "Price T\n(Adj)",
            "G/L",
            "G/L Adj",
        ]

        header_row = start_row + 2
        for i, header in enumerate(headers):
            cell = ws.cell(row=header_row, column=i + 1, value=header)
            cell.font = self.subheader_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

        data_row = header_row + 1
        first_data_row = data_row

        for _, row_data in equity_details.iterrows():
            col = 1

            ws.cell(row=data_row, column=col, value=row_data.get("ticker", ""))
            col += 1

            ws.cell(row=data_row, column=col, value=row_data.get("quantity_t1", 0)).number_format = "#,##0"
            col += 1
            ws.cell(row=data_row, column=col, value=row_data.get("quantity_t", 0)).number_format = "#,##0"
            col += 1

            ws.cell(row=data_row, column=col, value=row_data.get("price_t1_raw", 0)).number_format = self.currency_format
            col += 1
            ws.cell(row=data_row, column=col, value=row_data.get("price_t_raw", 0)).number_format = self.currency_format
            col += 1

            price_t1_adj = row_data.get("price_t1_adj", row_data.get("price_t1_raw", 0))
            price_t_adj = row_data.get("price_t_adj", row_data.get("price_t_raw", 0))

            adj_t1_cell = ws.cell(row=data_row, column=col, value=price_t1_adj)
            adj_t1_cell.number_format = self.currency_format
            if abs(price_t1_adj - row_data.get("price_t1_raw", 0)) > 0.001:
                adj_t1_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            adj_t_cell = ws.cell(row=data_row, column=col, value=price_t_adj)
            adj_t_cell.number_format = self.currency_format
            if abs(price_t_adj - row_data.get("price_t_raw", 0)) > 0.001:
                adj_t_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            gl_formula = f"=(E{data_row}-F{data_row})*C{data_row}"
            ws.cell(row=data_row, column=col, value=gl_formula).number_format = self.currency_format
            col += 1

            gl_adj_formula = f"=(G{data_row}-F{data_row})*C{data_row}"
            ws.cell(row=data_row, column=col, value=gl_adj_formula).number_format = self.currency_format

            data_row += 1

        if data_row > first_data_row:
            last_data_row = data_row - 1
            total_row = data_row + 1

            ws.cell(row=total_row, column=1, value="TOTAL").font = self.subheader_font

            gl_range = f"H{first_data_row}:H{last_data_row}"
            gl_adj_range = f"I{first_data_row}:I{last_data_row}"

            ws.cell(row=total_row, column=8, value=f"=SUM({gl_range})").number_format = self.currency_format
            ws.cell(row=total_row, column=9, value=f"=SUM({gl_adj_range})").number_format = self.currency_format

            return {
                "has_data": True,
                "gl_total_cell": f"H{total_row}",
                "gl_adj_total_cell": f"I{total_row}",
                "end_row": total_row,
            }
        return {
            "has_data": False,
            "gl_total_cell": None,
            "gl_adj_total_cell": None,
            "end_row": data_row,
        }

    def _add_option_details_with_formulas(self, ws, start_row, nav_data):
        """Add option details section with formulas for G/L calculations."""
        ws.cell(row=start_row, column=1, value="OPTION GAIN/LOSS DETAIL").font = self.header_font

        detailed_calcs = nav_data.get("detailed_calculations", {})
        option_details = detailed_calcs.get("option_details", pd.DataFrame())

        if option_details.empty:
            ws.cell(row=start_row + 2, column=1, value="No option holdings with gain/loss")
            return {
                "has_data": False,
                "gl_total_cell": None,
                "gl_adj_total_cell": None,
                "end_row": start_row + 2,
            }

        headers = [
            "Ticker",
            "Qty T-1",
            "Qty T",
            "Price T-1\n(Vest)",
            "Price T\n(Raw)",
            "Price T-1\n(Cust)",
            "Price T\n(Adj)",
            "G/L",
            "G/L Adj",
        ]

        header_row = start_row + 2
        for i, header in enumerate(headers):
            cell = ws.cell(row=header_row, column=i + 1, value=header)
            cell.font = self.subheader_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

        data_row = header_row + 1
        first_data_row = data_row

        for _, row_data in option_details.iterrows():
            col = 1

            ws.cell(row=data_row, column=col, value=row_data.get("ticker", ""))
            col += 1

            ws.cell(row=data_row, column=col, value=row_data.get("quantity_t1", 0)).number_format = "#,##0"
            col += 1
            ws.cell(row=data_row, column=col, value=row_data.get("quantity_t", 0)).number_format = "#,##0"
            col += 1

            ws.cell(row=data_row, column=col, value=row_data.get("price_t1_raw", 0)).number_format = self.currency_format
            col += 1
            ws.cell(row=data_row, column=col, value=row_data.get("price_t_raw", 0)).number_format = self.currency_format
            col += 1

            price_t1_adj = row_data.get("price_t1_adj", row_data.get("price_t1_raw", 0))
            price_t_adj = row_data.get("price_t_adj", row_data.get("price_t_raw", 0))

            adj_t1_cell = ws.cell(row=data_row, column=col, value=price_t1_adj)
            adj_t1_cell.number_format = self.currency_format
            if abs(price_t1_adj - row_data.get("price_t1_raw", 0)) > 0.001:
                adj_t1_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            adj_t_cell = ws.cell(row=data_row, column=col, value=price_t_adj)
            adj_t_cell.number_format = self.currency_format
            if abs(price_t_adj - row_data.get("price_t_raw", 0)) > 0.001:
                adj_t_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            gl_formula = f"=(E{data_row}-F{data_row})*C{data_row}*100"
            ws.cell(row=data_row, column=col, value=gl_formula).number_format = self.currency_format
            col += 1

            gl_adj_formula = f"=(G{data_row}-F{data_row})*C{data_row}*100"
            ws.cell(row=data_row, column=col, value=gl_adj_formula).number_format = self.currency_format

            data_row += 1

        if data_row > first_data_row:
            last_data_row = data_row - 1
            total_row = data_row + 1

            ws.cell(row=total_row, column=1, value="TOTAL").font = self.subheader_font

            gl_range = f"H{first_data_row}:H{last_data_row}"
            gl_adj_range = f"I{first_data_row}:I{last_data_row}"

            ws.cell(row=total_row, column=8, value=f"=SUM({gl_range})").number_format = self.currency_format
            ws.cell(row=total_row, column=9, value=f"=SUM({gl_adj_range})").number_format = self.currency_format

            return {
                "has_data": True,
                "gl_total_cell": f"H{total_row}",
                "gl_adj_total_cell": f"I{total_row}",
                "end_row": total_row,
            }
        return {
            "has_data": False,
            "gl_total_cell": None,
            "gl_adj_total_cell": None,
            "end_row": data_row,
        }

    def _add_flex_option_details_with_formulas(self, ws, start_row, nav_data):
        """Add flex option details section with formulas."""
        ws.cell(row=start_row, column=1, value="FLEX OPTION GAIN/LOSS DETAIL").font = self.header_font

        detailed_calcs = nav_data.get("detailed_calculations", {})
        flex_details = detailed_calcs.get("flex_details", pd.DataFrame())

        if flex_details.empty:
            ws.cell(row=start_row + 2, column=1, value="No flex option holdings with gain/loss")
            return {
                "has_data": False,
                "gl_total_cell": None,
                "gl_adj_total_cell": None,
                "end_row": start_row + 2,
            }

        headers = [
            "Ticker",
            "Qty T-1",
            "Qty T",
            "Price T-1\n(Vest)",
            "Price T\n(Raw)",
            "Price T-1\n(Cust)",
            "Price T\n(Adj)",
            "G/L",
            "G/L Adj",
        ]

        header_row = start_row + 2
        for i, header in enumerate(headers):
            cell = ws.cell(row=header_row, column=i + 1, value=header)
            cell.font = self.subheader_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

        data_row = header_row + 1
        first_data_row = data_row

        for _, row_data in flex_details.iterrows():
            col = 1

            ws.cell(row=data_row, column=col, value=row_data.get("ticker", ""))
            col += 1

            ws.cell(row=data_row, column=col, value=row_data.get("quantity_t1", 0)).number_format = "#,##0"
            col += 1
            ws.cell(row=data_row, column=col, value=row_data.get("quantity_t", 0)).number_format = "#,##0"
            col += 1

            ws.cell(row=data_row, column=col, value=row_data.get("price_t1_raw", 0)).number_format = self.currency_format
            col += 1
            ws.cell(row=data_row, column=col, value=row_data.get("price_t_raw", 0)).number_format = self.currency_format
            col += 1

            price_t1_adj = row_data.get("price_t1_adj", row_data.get("price_t1_raw", 0))
            price_t_adj = row_data.get("price_t_adj", row_data.get("price_t_raw", 0))

            adj_t1_cell = ws.cell(row=data_row, column=col, value=price_t1_adj)
            adj_t1_cell.number_format = self.currency_format
            if abs(price_t1_adj - row_data.get("price_t1_raw", 0)) > 0.001:
                adj_t1_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            adj_t_cell = ws.cell(row=data_row, column=col, value=price_t_adj)
            adj_t_cell.number_format = self.currency_format
            if abs(price_t_adj - row_data.get("price_t_raw", 0)) > 0.001:
                adj_t_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")
            col += 1

            gl_formula = f"=(E{data_row}-F{data_row})*C{data_row}*100"
            ws.cell(row=data_row, column=col, value=gl_formula).number_format = self.currency_format
            col += 1

            gl_adj_formula = f"=(G{data_row}-F{data_row})*C{data_row}*100"
            ws.cell(row=data_row, column=col, value=gl_adj_formula).number_format = self.currency_format

            data_row += 1

        if data_row > first_data_row:
            last_data_row = data_row - 1
            total_row = data_row + 1

            ws.cell(row=total_row, column=1, value="TOTAL").font = self.subheader_font

            gl_range = f"H{first_data_row}:H{last_data_row}"
            gl_adj_range = f"I{first_data_row}:I{last_data_row}"

            ws.cell(row=total_row, column=8, value=f"=SUM({gl_range})").number_format = self.currency_format
            ws.cell(row=total_row, column=9, value=f"=SUM({gl_adj_range})").number_format = self.currency_format

            return {
                "has_data": True,
                "gl_total_cell": f"H{total_row}",
                "gl_adj_total_cell": f"I{total_row}",
                "end_row": total_row,
            }
        return {
            "has_data": False,
            "gl_total_cell": None,
            "gl_adj_total_cell": None,
            "end_row": data_row,
        }

    def _create_summary_sheet(self, workbook: Workbook):
        """Create summary sheet with all funds."""
        worksheet = workbook.create_sheet("NAV Summary")

        # Title
        worksheet.merge_cells("A1:K1")
        title_cell = worksheet.cell(
            row=1,
            column=1,
            value=f"NAV Reconciliation Summary - {self.report_date}",
        )
        title_cell.font = self.title_font
        title_cell.alignment = Alignment(horizontal="center")

        headers = [
            "Fund",
            "Date",
            "Expected TNA",
            "Custodian TNA",
            "TNA Diff ($)",
            "Expected NAV",
            "Custodian NAV",
            "NAV Diff ($)",
            "Shares Outstanding",
            "NAV Good (2-dec)",
            "NAV Good (4-dec)",
        ]

        header_row = 3
        for index, header in enumerate(headers, start=1):
            cell = worksheet.cell(row=header_row, column=index, value=header)
            cell.font = self.header_font
            cell.alignment = Alignment(horizontal="center")

        current_row = header_row + 1
        for fund_name in sorted(self.results.keys()):
            nav_data = self.results[fund_name]

            worksheet.cell(row=current_row, column=1, value=fund_name)
            worksheet.cell(row=current_row, column=2, value=self.report_date)
            worksheet.cell(row=current_row, column=3, value=nav_data.get("Expected TNA", 0)).number_format = self.currency_format
            worksheet.cell(row=current_row, column=4, value=nav_data.get("Custodian TNA", 0)).number_format = self.currency_format
            worksheet.cell(row=current_row, column=5, value=nav_data.get("TNA Diff ($)", 0)).number_format = self.currency_format
            worksheet.cell(row=current_row, column=6, value=nav_data.get("Expected NAV", 0)).number_format = self.nav_format
            worksheet.cell(row=current_row, column=7, value=nav_data.get("Custodian NAV", 0)).number_format = self.nav_format
            worksheet.cell(row=current_row, column=8, value=nav_data.get("NAV Diff ($)", 0)).number_format = self.nav_format
            worksheet.cell(row=current_row, column=9, value=nav_data.get("Shares Outstanding", 0)).number_format = "#,##0"

            nav_good_2 = nav_data.get("NAV Good (2 Digit)", False)
            nav_good_4 = nav_data.get("NAV Good (4 Digit)", False)

            cell_2dec = worksheet.cell(row=current_row, column=10, value="PASS" if nav_good_2 else "FAIL")
            cell_4dec = worksheet.cell(row=current_row, column=11, value="PASS" if nav_good_4 else "FAIL")

            if nav_good_2:
                cell_2dec.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
            else:
                cell_2dec.fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")

            if nav_good_4:
                cell_4dec.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
            else:
                cell_4dec.fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")

            current_row += 1

        for column_index in range(1, 12):
            worksheet.column_dimensions[get_column_letter(column_index)].width = 15

        return worksheet


# ------------------------------------------------------------------

def convert_nav_results_to_dicts(nav_data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Convert NAVReconciliationResults dataclasses to dicts.

    Handles nested structures like {date: {fund: NAVReconciliationResults}}
    or flat structures like {fund: NAVReconciliationResults}.

    Args:
        nav_data: Can be {fund: results} or {date: {fund: results}}

    Returns:
        Same structure but with dicts instead of dataclasses
    """
    converted = {}

    for key, value in nav_data.items():
        # Check if value is a NAVReconciliationResults dataclass
        if hasattr(value, 'to_legacy_dict'):
            # It's a dataclass, convert it
            converted[key] = value.to_legacy_dict()
        elif isinstance(value, dict):
            # It's a nested dict, recurse
            converted[key] = convert_nav_results_to_dicts(value)
        else:
            # It's something else, keep as-is
            converted[key] = value

    return converted


def generate_nav_reconciliation_reports(reconciliation_results, date_str, excel_path, *, workbook: Workbook | None = None):
    """Generate NAV reconciliation reports in Excel format."""

    resolved_date = normalize_report_date(date_str)

    # ✅ SINGLE FIX: Convert any dataclasses to dicts first
    converted_results = convert_nav_results_to_dicts(reconciliation_results)

    # Now handle date unwrapping with dicts
    if isinstance(converted_results, dict):
        if date_str in converted_results:
            normalized = converted_results[date_str]
        elif resolved_date in converted_results:
            normalized = converted_results[resolved_date]
        else:
            first_key = next(iter(converted_results.keys())) if converted_results else None
            if first_key and isinstance(converted_results[first_key], dict):
                first_value = converted_results[first_key]
                if any(
                        key in first_value
                        for key in ["NAV Good (2 Digit)", "Expected NAV", "Beginning TNA"]
                ):
                    normalized = converted_results
                else:
                    normalized = converted_results.get(first_key, {})
            else:
                normalized = converted_results
    else:
        normalized = {}

    output_path = Path(excel_path)
    if output_path.suffix.lower() != ".xlsx":
        output_path = output_path / f"nav_reconciliation_{resolved_date}.xlsx"

    if workbook is None and output_path.exists():
        workbook = load_workbook(output_path)

    NAVReconciliationExcelReport(
        results=normalized,
        report_date=resolved_date,
        output_path=output_path,
        workbook=workbook,
    )

    return str(output_path)


def generate_reconciliation_summary_pdf(
        reconciliation_results: Mapping[str, Any],
        nav_results: Mapping[str, Any],
        report_date: date | datetime | str,
        output_dir: str,
        *,
        file_name: str | None = None,
) -> Optional[str]:
    """Create the combined holdings + NAV reconciliation PDF."""

    date_str = normalize_report_date(report_date)

    # ✅ SINGLE FIX: Convert any dataclasses to dicts
    raw_recon = reconciliation_results or {}
    raw_nav = convert_nav_results_to_dicts(nav_results or {})

    # Now unwrap date layer if present (working with dicts now)
    if raw_recon:
        first_key = next(iter(raw_recon.keys()))
        first_value = raw_recon[first_key]
        if isinstance(first_value, dict) and first_value:
            sample_key = next(iter(first_value.keys()))
            sample_value = first_value.get(sample_key)
            if isinstance(sample_value, dict) and any(
                    key in sample_value for key in ["summary", "details", "custodian_equity"]
            ):
                raw_recon = first_value

    if raw_nav:
        first_key = next(iter(raw_nav.keys()))
        first_value = raw_nav[first_key]
        if isinstance(first_value, dict) and first_value:
            sample_key = next(iter(first_value.keys()))
            sample_value = first_value.get(sample_key)
            if isinstance(sample_value, dict) and any(
                    key in sample_value for key in ["Beginning TNA", "Expected NAV", "summary"]
            ):
                raw_nav = first_value

    if not raw_recon and not raw_nav:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_name = file_name or f"reconciliation_summary_{date_str}.pdf"
    pdf_path = output_path / resolved_name

    # Build holdings structure
    holdings_details: Dict[str, Dict[str, Any]] = {date_str: {}}
    holdings_summary: Dict[str, list[Dict[str, Any]]] = {date_str: []}
    for fund, payload in raw_recon.items():
        if not payload:
            continue
        holdings_details[date_str][fund] = payload.get("details", {})
        holdings_summary[date_str].append({
            "fund": fund,
            "summary": payload.get("summary", {}),
        })

    # Build NAV structure (already converted to dicts)
    nav_details: Dict[str, Dict[str, Any]] = {date_str: {}}
    nav_summary: Dict[str, list[Dict[str, Any]]] = {date_str: []}

    for fund, payload in raw_nav.items():
        if not payload:
            continue
        nav_details[date_str][fund] = payload
        nav_summary[date_str].append({
            "fund": fund,
            "summary": payload,
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