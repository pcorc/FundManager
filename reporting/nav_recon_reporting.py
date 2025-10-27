from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

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
    summarise_reconciliation_breaks,
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
            self._create_fund_sheet(workbook, fund_name, payload)

        summary_ws = self._create_summary_sheet(workbook)
        if summary_ws is not None:
            # Move the summary sheet to the first position
            workbook._sheets.insert(0, workbook._sheets.pop(workbook._sheets.index(summary_ws)))

        workbook.save(self.output_path)

    # ------------------------------------------------------------------
    def _create_fund_sheet(
        self,
        workbook: Workbook,
        fund_name: str,
        payload: Mapping[str, Any],
    ) -> None:
        sheet_name = self._unique_sheet_name(fund_name)
        worksheet = workbook.create_sheet(sheet_name)

        summary = payload.get("summary", {}) or {}
        details = payload.get("details", {}) or {}

        worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=8)
        title_cell = worksheet.cell(row=1, column=1, value=f"{fund_name} - NAV Reconciliation")
        title_cell.font = self.title_font
        title_cell.alignment = Alignment(horizontal="center")

        # Predetermine summary row layout
        prior_nav_row = 3
        dividends_row = prior_nav_row + 1
        expenses_row = prior_nav_row + 2
        section_header_row = prior_nav_row + 3
        component_start_row = section_header_row + 1

        component_rows: Dict[str, int] = {}
        for index, (component_key, _) in enumerate(self.COMPONENTS):
            component_rows[component_key] = component_start_row + index

        net_gain_row = component_start_row + len(self.COMPONENTS)
        distributions_row = net_gain_row + 2
        flow_row = distributions_row + 1
        expected_nav_row = flow_row + 1
        cust_nav_row = expected_nav_row + 1
        variance_row = cust_nav_row + 1
        detail_start_row = variance_row + 3

        # Build detail sections first to capture total cells
        component_totals: Dict[str, Dict[str, Any]] = {}
        next_row = detail_start_row
        for component_key, display_name in self.COMPONENTS:
            section_info = self._add_component_detail_section(
                worksheet,
                next_row,
                display_name,
                ensure_dataframe(details.get(component_key)),
            )
            component_totals[component_key] = section_info
            next_row = section_info["end_row"] + 2

        # Summary labels and formulas
        self._write_summary_row(
            worksheet,
            prior_nav_row,
            "Beginning NAV (T-1)",
            self._coerce_float(summary.get("prior_nav")),
            number_format=self.currency_format,
        )

        self._write_summary_row(
            worksheet,
            dividends_row,
            "Dividends",
            self._coerce_float(summary.get("dividends")),
            number_format=self.currency_format,
        )

        self._write_summary_row(
            worksheet,
            expenses_row,
            "Expenses",
            self._coerce_float(summary.get("expenses")),
            number_format=self.currency_format,
        )

        header_cell = worksheet.cell(row=section_header_row, column=1, value="Gain/Loss by Asset Class")
        header_cell.font = self.subheader_font

        component_cells: list[str] = []
        for component_key, display_name in self.COMPONENTS:
            row_number = component_rows[component_key]
            cell = worksheet.cell(row=row_number, column=1, value=f"  {display_name} G/L")
            cell.font = self.body_font
            total_cell = component_totals.get(component_key, {}).get("adjusted_total")
            value_cell = worksheet.cell(row=row_number, column=3)
            if total_cell:
                value_cell.value = f"={total_cell}"
            else:
                value_cell.value = 0.0
            value_cell.number_format = self.currency_format
            component_cells.append(f"C{row_number}")

        net_gain_cell = worksheet.cell(row=net_gain_row, column=1, value="Net Gain/Loss")
        net_gain_cell.font = self.subheader_font
        net_gain_value_cell = worksheet.cell(row=net_gain_row, column=3)
        if component_cells:
            net_gain_value_cell.value = f"={' + '.join(component_cells)}" if len(component_cells) > 1 else f"={component_cells[0]}"
        else:
            net_gain_value_cell.value = self._coerce_float(summary.get("net_gain"))
        net_gain_value_cell.number_format = self.currency_format

        worksheet.cell(row=net_gain_row + 1, column=1)  # spacer row for readability

        self._write_summary_row(
            worksheet,
            distributions_row,
            "Distributions",
            self._coerce_float(summary.get("distributions")),
            number_format=self.currency_format,
        )

        self._write_summary_row(
            worksheet,
            flow_row,
            "Flow Adjustment",
            self._coerce_float(summary.get("flows_adjustment")),
            number_format=self.currency_format,
        )

        expected_nav_cell = worksheet.cell(row=expected_nav_row, column=1, value="Expected NAV")
        expected_nav_cell.font = self.subheader_font
        expected_nav_value = worksheet.cell(row=expected_nav_row, column=3)
        expected_nav_value.value = (
            f"=C{prior_nav_row}+C{net_gain_row}+C{dividends_row}-C{expenses_row}-C{distributions_row}+C{flow_row}"
        )
        expected_nav_value.number_format = self.nav_format

        self._write_summary_row(
            worksheet,
            cust_nav_row,
            "Custodian NAV",
            self._coerce_float(summary.get("current_nav")),
            number_format=self.nav_format,
            bold=True,
        )

        variance_cell = worksheet.cell(row=variance_row, column=1, value="Variance")
        variance_cell.font = self.subheader_font
        variance_value = worksheet.cell(row=variance_row, column=3)
        variance_value.value = f"=C{cust_nav_row}-C{expected_nav_row}"
        variance_value.number_format = self.nav_format

        # Store references for the summary worksheet
        cell_refs = {
            "sheet_name": sheet_name,
            "prior_nav": f"C{prior_nav_row}",
            "dividends": f"C{dividends_row}",
            "expenses": f"C{expenses_row}",
            "distributions": f"C{distributions_row}",
            "flow_adjustment": f"C{flow_row}",
            "expected_nav": f"C{expected_nav_row}",
            "cust_nav": f"C{cust_nav_row}",
            "variance": f"C{variance_row}",
            "net_gain": f"C{net_gain_row}",
            "components": {
                component_key: f"C{component_rows[component_key]}"
                for component_key, _ in self.COMPONENTS
            },
        }
        self._sheet_refs[fund_name] = cell_refs

        # Formatting tweaks
        for col in range(1, 12):
            worksheet.column_dimensions[get_column_letter(col)].width = 16

        worksheet.freeze_panes = f"A{detail_start_row}"

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
        self.add_title("NAV Reconciliation Summary", f"As of {self.report_date}")

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
                align=["L", "R", "R", "R", "R", "R", "R"],
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
        self.add_title("Daily Oversight Summary", f"As of {self.report_date}")

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
                self.add_table([
                    "Reconciliation",
                    "Total Breaks",
                ], rows, column_widths=[100, 40], alignments=["L", "R"])

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
                self.add_table(
                    ["Fund", "Expected NAV", "Custodian NAV", "Variance"],
                    detail_rows,
                    column_widths=[70, 35, 35, 35],
                    alignments=["L", "R", "R", "R"],
                )

        self.output()


def generate_nav_reconciliation_reports(
    results: Mapping[str, Any],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    file_name_prefix: str = "nav_reconciliation_results",
    create_pdf: bool = True,
) -> GeneratedNAVReconciliationReport:
    """Generate Excel and PDF NAV reconciliation reports."""

    normalized = normalize_nav_payload(results)
    if not normalized:
        return GeneratedNAVReconciliationReport(None, None)

    date_str = normalize_report_date(report_date)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excel_path = output_path / f"{file_name_prefix}_{date_str}.xlsx"
    pdf_path = output_path / f"{file_name_prefix}_{date_str}.pdf"

    NAVReconciliationExcelReport(normalized, date_str, excel_path)

    pdf_result: Optional[str] = None
    if create_pdf:
        pdf = NAVReconciliationPDF(str(pdf_path), date_str, normalized)
        pdf.render()
        pdf_result = str(pdf_path)

    return GeneratedNAVReconciliationReport(str(excel_path), pdf_result)


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