from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


@dataclass
class GeneratedNAVReconciliationReport:
    """File locations for generated NAV reconciliation artefacts."""

    excel_path: Optional[str]


class NAVReconciliationExcelReport:
    """Render NAV reconciliation data into an Excel workbook with formulas and ticker details."""

    CURRENCY_FORMAT = "#,##0.00"
    NAV_FORMAT = "#,##0.0000"
    INTEGER_FORMAT = "#,##0"

    def __init__(
        self,
        results: Mapping[str, Any],
        report_date: str,
        output_path: Path,
        workbook: Workbook | None = None,
    ) -> None:
        self.results = {fund: data for fund, data in (results or {}).items() if isinstance(data, dict)}
        self.report_date = report_date
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.workbook = workbook
        self._using_existing_workbook = workbook is not None

        self.title_font = Font(bold=True, size=14)
        self.header_font = Font(bold=True, size=12)
        self.subheader_font = Font(bold=True, size=10)
        self.link_font = Font(color="0000FF", underline="single")

        self._export()

    # ------------------------------------------------------------------
    def _export(self) -> None:
        workbook = self.workbook or Workbook()
        if not self._using_existing_workbook:
            workbook.remove(workbook.active)

        for fund_name, payload in sorted(self.results.items()):
            if payload:
                self._create_fund_sheet_with_formulas(workbook, fund_name, payload, self.report_date)

        summary_ws = self._create_summary_sheet(workbook)
        if summary_ws is not None and not self._using_existing_workbook:
            workbook._sheets.insert(0, workbook._sheets.pop(workbook._sheets.index(summary_ws)))

        workbook.save(self.output_path)

    # ------------------------------------------------------------------
    def _label_value(self, ws, row, label, value, number_format=None, font=None) -> None:
        label_cell = ws.cell(row=row, column=1, value=label)
        if font is not None:
            label_cell.font = font
        value_cell = ws.cell(row=row, column=3, value=value)
        if number_format is not None:
            value_cell.number_format = number_format

    def _create_fund_sheet_with_formulas(self, wb, fund_name, nav_data, date_str):
        """Create detailed sheet for a fund with formulas matching NAVReconciliationReport."""
        ws = wb.create_sheet(fund_name[:31])
        currency, sub = self.CURRENCY_FORMAT, self.subheader_font

        ws.merge_cells("A1:C1")
        ws["A1"] = f"{fund_name} - NAV Reconciliation - {date_str}"
        ws["A1"].font = self.title_font
        ws["A1"].alignment = Alignment(horizontal="center")

        # Back-link to summary
        back = ws.cell(row=1, column=5, value="-> NAV Summary")
        back.hyperlink = "#'NAV Summary'!A1"
        back.font = self.link_font

        ws.cell(row=3, column=1, value="NAV CALCULATION SUMMARY").font = self.header_font
        self._label_value(ws, 5, "Beginning TNA (T-1)", nav_data.get("Beginning TNA", 0), currency)
        adj_beg_tna_row = 6
        self._label_value(
            ws, adj_beg_tna_row, "Adjusted Beginning TNA",
            nav_data.get("Adjusted Beginning TNA", nav_data.get("Beginning TNA", 0)), currency,
        )
        ws.cell(row=8, column=1, value="Gain/Loss Components:").font = sub
        row = 9

        detail_start_row = row + 30
        equity_info = self._add_asset_details_with_formulas(
            ws, detail_start_row, nav_data,
            title="EQUITY GAIN/LOSS DETAIL",
            details_key="equity_details",
            empty_message="No equity holdings with gain/loss",
        )
        option_info = self._add_asset_details_with_formulas(
            ws, equity_info["end_row"] + 3, nav_data,
            title="OPTION GAIN/LOSS DETAIL",
            details_key="option_details",
            empty_message="No option holdings with gain/loss",
            contract_multiplier=100,
        )
        flex_info = self._add_asset_details_with_formulas(
            ws, option_info["end_row"] + 3, nav_data,
            title="FLEX OPTION GAIN/LOSS DETAIL",
            details_key="flex_details",
            empty_message="No flex option holdings with gain/loss",
            contract_multiplier=100,
            always_highlight=True,
        )

        # Raw and adjusted G/L per asset class; option/flex rows only when there is something to show
        for label, info, gl_key, always in (
            ("Equity", equity_info, "Equity G/L", True),
            ("Option", option_info, "Option G/L", False),
            ("Flex Option", flex_info, "Flex Option G/L", False),
        ):
            gl = nav_data.get(gl_key, 0)
            if not (always or abs(gl) > 0.01 or info["has_data"]):
                continue
            self._label_value(
                ws, row, f"  {label} G/L",
                f"={info['gl_total_cell']}" if info["gl_total_cell"] else gl, currency,
            )
            self._label_value(
                ws, row + 1, f"  {label} G/L (Adjusted)",
                f"={info['gl_adj_total_cell']}" if info["gl_adj_total_cell"]
                else nav_data.get(f"{gl_key} Adj", gl),
                currency,
            )
            row += 2

        formula_parts = [f"C{adj_beg_tna_row}"]
        formula_parts += [
            info["gl_adj_total_cell"] for info in (equity_info, option_info, flex_info) if info["gl_adj_total_cell"]
        ]
        # Treasury enters the Expected TNA formula as a cell ref, assignment/other as literals
        for label, key, as_cell in (
            ("  Treasury G/L", "Treasury G/L", True),
            ("  Assignment G/L", "Assignment G/L", False),
            ("  Other", "Other", False),
        ):
            value = nav_data.get(key, 0)
            if abs(value) > 0.01:
                self._label_value(ws, row, label, value, currency)
                formula_parts.append(f"C{row}" if as_cell else str(value))
                row += 1
        row += 1

        self._label_value(ws, row, "Expenses", -abs(nav_data.get("Accruals", 0)), currency)
        formula_parts.append(f"C{row}")
        row += 1

        for label, value in (
            ("Dividends", nav_data.get("Dividends", 0)),
            ("Distributions", nav_data.get("Distributions", 0)),
        ):
            if abs(value) > 0.01:
                self._label_value(ws, row, label, -abs(value) if label == "Distributions" else value, currency)
                formula_parts.append(f"C{row}")
                row += 1
        row += 1

        expected_tna_row, cust_tna_row = row, row + 1
        self._label_value(ws, expected_tna_row, "Expected TNA", "=" + "+".join(formula_parts), currency, sub)
        self._label_value(ws, cust_tna_row, "Custodian TNA", nav_data.get("Custodian TNA", 0), currency, sub)
        self._label_value(ws, row + 2, "TNA Difference", f"=C{cust_tna_row}-C{expected_tna_row}", currency)

        shares_row = row + 4
        self._label_value(ws, shares_row, "Shares Outstanding", nav_data.get("Shares Outstanding", 0), self.INTEGER_FORMAT)

        expected_nav_row, cust_nav_row, nav_diff_row = row + 6, row + 7, row + 8
        self._label_value(
            ws, expected_nav_row, "Expected NAV",
            f"=IF(C{shares_row}<>0,C{expected_tna_row}/C{shares_row},0)", self.NAV_FORMAT, sub,
        )
        self._label_value(ws, cust_nav_row, "Custodian NAV", nav_data.get("Custodian NAV", 0), self.NAV_FORMAT, sub)
        self._label_value(
            ws, nav_diff_row, "NAV Difference", f"=C{cust_nav_row}-C{expected_nav_row}", self.NAV_FORMAT,
        )

        # NAV Good indicators
        self._label_value(
            ws, row + 10, "NAV Good (2 decimal)", f'=IF(ABS(ROUND(C{nav_diff_row},2))<=0.01,"PASS","FAIL")',
        )
        self._label_value(
            ws, row + 11, "NAV Good (4 decimal)",
            f'=IF(ROUND(C{cust_nav_row},2)=ROUND(C{expected_nav_row},2),"PASS","FAIL")',
        )

        for col in range(1, 12):
            ws.column_dimensions[get_column_letter(col)].width = 14
        ws.freeze_panes = "A4"

    # ------------------------------------------------------------------
    def _add_asset_details_with_formulas(
        self,
        ws,
        start_row: int,
        nav_data: Mapping[str, Any],
        *,
        title: str,
        details_key: str,
        empty_message: str,
        contract_multiplier: int = 1,
        always_highlight: bool = False,
    ) -> Dict[str, Any]:
        ws.cell(row=start_row, column=1, value=title).font = self.header_font

        details_df = nav_data.get("detailed_calculations", {}).get(details_key, pd.DataFrame())
        if details_df is None or details_df.empty:
            ws.cell(row=start_row + 2, column=1, value=empty_message)
            return {
                "has_data": False,
                "gl_total_cell": None,
                "gl_adj_total_cell": None,
                "end_row": start_row + 2,
            }

        headers = [
            "Ticker", "Qty T-1", "Qty T",
            "Price T-1\n(Cust)", "Price T\n(Cust)",
            "Price T-1\n(Vest/Adj)", "Price T\n(Vest/Adj)",
            "G/L\n(Cust)", "G/L\n(Vest/Adj)",
        ]
        header_row = start_row + 2
        for i, header in enumerate(headers):
            cell = ws.cell(row=header_row, column=i + 1, value=header)
            cell.font = self.subheader_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

        currency = self.CURRENCY_FORMAT
        first_data_row = header_row + 1
        data_row = first_data_row
        multiplier_suffix = f"*{contract_multiplier}" if contract_multiplier != 1 else ""

        for _, row_data in details_df.iterrows():
            ws.cell(row=data_row, column=1, value=row_data.get("ticker", ""))
            ws.cell(row=data_row, column=2, value=row_data.get("quantity_t1", 0)).number_format = self.INTEGER_FORMAT
            ws.cell(row=data_row, column=3, value=row_data.get("quantity_t", 0)).number_format = self.INTEGER_FORMAT

            price_t1_raw = row_data.get("price_t1_raw", 0)
            price_t_raw = row_data.get("price_t_raw", 0)
            ws.cell(row=data_row, column=4, value=price_t1_raw).number_format = currency
            ws.cell(row=data_row, column=5, value=price_t_raw).number_format = currency

            for column, adj_key, raw in ((6, "price_t1_adj", price_t1_raw), (7, "price_t_adj", price_t_raw)):
                adjusted = row_data.get(adj_key, raw)
                adj_cell = ws.cell(row=data_row, column=column, value=adjusted)
                adj_cell.number_format = currency
                if always_highlight or abs(adjusted - raw) > 0.001:
                    adj_cell.fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")

            # G/L (Cust) and G/L (Vest/Adj) = (Price T - Price T-1) * Qty
            ws.cell(
                row=data_row, column=8,
                value=f"=(E{data_row}-D{data_row})*C{data_row}{multiplier_suffix}",
            ).number_format = currency
            ws.cell(
                row=data_row, column=9,
                value=f"=(G{data_row}-F{data_row})*C{data_row}{multiplier_suffix}",
            ).number_format = currency

            data_row += 1

        last_data_row = data_row - 1
        total_row = data_row + 1

        ws.cell(row=total_row, column=1, value="TOTAL").font = self.subheader_font
        ws.cell(row=total_row, column=8, value=f"=SUM(H{first_data_row}:H{last_data_row})").number_format = currency
        ws.cell(row=total_row, column=9, value=f"=SUM(I{first_data_row}:I{last_data_row})").number_format = currency

        return {
            "has_data": True,
            "gl_total_cell": f"H{total_row}",
            "gl_adj_total_cell": f"I{total_row}",
            "end_row": total_row,
        }

    # ------------------------------------------------------------------
    def _create_summary_sheet(self, workbook: Workbook):
        """Create summary sheet with all funds. Column A links to each fund's tab;
        trailing columns show the vest-vs-custodian option-price sensitivity, with
        the Option Price Impact as a live formula (Cust G/L - Vest G/L)."""
        worksheet = workbook.create_sheet("NAV Summary")

        headers = [
            "Fund", "Date", "Expected TNA", "Custodian TNA", "TNA Diff ($)",
            "Expected NAV", "Custodian NAV", "NAV Diff ($)", "Shares Outstanding",
            "NAV Good (2-dec)", "NAV Good (4-dec)",
            # ---- option-price sensitivity (vest vs custodian) ----
            "Combined Opt G/L (Vest)", "Combined Opt G/L (Cust)", "Option Price Impact ($)",
            "Expected NAV (Cust Opt)", "NAV Good (Cust Opt)",
        ]

        last_col = len(headers)
        worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=last_col)
        title_cell = worksheet.cell(
            row=1, column=1, value=f"NAV Reconciliation Summary - {self.report_date}"
        )
        title_cell.font = self.title_font
        title_cell.alignment = Alignment(horizontal="center")

        header_row = 3
        for index, header in enumerate(headers, start=1):
            cell = worksheet.cell(row=header_row, column=index, value=header)
            cell.font = self.header_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)

        good_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
        bad_fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")
        currency, nav_fmt = self.CURRENCY_FORMAT, self.NAV_FORMAT

        current_row = header_row + 1
        for fund_name in sorted(self.results.keys()):
            nav_data = self.results[fund_name]
            r = current_row

            # Fund cell with internal hyperlink to that fund's tab
            fund_cell = worksheet.cell(row=r, column=1, value=fund_name)
            fund_cell.hyperlink = f"#'{fund_name[:31]}'!A1"
            fund_cell.font = self.link_font

            worksheet.cell(row=r, column=2, value=self.report_date)
            for column, key, number_format in (
                (3, "Expected TNA", currency),
                (4, "Custodian TNA", currency),
                (5, "TNA Diff ($)", currency),
                (6, "Expected NAV", nav_fmt),
                (7, "Custodian NAV", nav_fmt),
                (8, "NAV Diff ($)", nav_fmt),
                (9, "Shares Outstanding", self.INTEGER_FORMAT),
            ):
                worksheet.cell(row=r, column=column, value=nav_data.get(key, 0)).number_format = number_format

            for column, key, digits in ((10, "NAV Good (2 Digit)", 2), (11, "NAV Good (4 Digit)", 4)):
                passed = bool(nav_data.get(key, False))
                worksheet.cell(
                    row=r, column=column,
                    value=f'=IF(ROUND(F{r},{digits})=ROUND(G{r},{digits}),"PASS","FAIL")',
                ).fill = good_fill if passed else bad_fill

            # ---- option-price sensitivity columns; impact = Custodian G/L - Vest G/L (live formula) ----
            worksheet.cell(row=r, column=12, value=nav_data.get("combined_gl_vest", 0.0)).number_format = currency
            worksheet.cell(row=r, column=13, value=nav_data.get("combined_gl_cust", 0.0)).number_format = currency
            worksheet.cell(row=r, column=14, value=f"=M{r}-L{r}").number_format = currency
            worksheet.cell(
                row=r, column=15,
                value=nav_data.get("exp_nav_cust", nav_data.get("Expected NAV", 0)),
            ).number_format = nav_fmt
            worksheet.cell(row=r, column=16, value=f'=IF(ROUND(G{r},2)=ROUND(O{r},2),"PASS","FAIL")')

            current_row += 1

        for column_index in range(1, last_col + 1):
            worksheet.column_dimensions[get_column_letter(column_index)].width = 16

        return worksheet


def convert_nav_results_to_dicts(nav_data: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively convert NAVReconciliationResults dataclasses to plain dicts."""
    converted: Dict[str, Any] = {}
    for key, value in nav_data.items():
        if hasattr(value, "to_legacy_dict"):
            converted[key] = value.to_legacy_dict()
        elif isinstance(value, dict):
            converted[key] = convert_nav_results_to_dicts(value)
        else:
            converted[key] = value
    return converted
