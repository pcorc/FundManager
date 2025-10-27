"""Generate Excel and PDF outputs for trading compliance comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import logging

import pandas as pd
from fpdf import FPDF
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

from reporting.report_utils import normalize_report_date

logger = logging.getLogger(__name__)


@dataclass
class GeneratedTradingComplianceReport:
    """Excel/PDF artefacts for trading compliance."""

    excel_path: Optional[str]
    pdf_path: Optional[str]


def generate_trading_excel_report(comparison_data: Mapping[str, Any], output_path: str) -> None:
    """Create a multi-tab Excel workbook summarising trading compliance changes."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)

    _create_summary_sheet(workbook, comparison_data)
    _create_compliance_changes_sheet(workbook, comparison_data)
    _create_trade_activity_sheet(workbook, comparison_data)
    _create_compliance_details_sheet(workbook, comparison_data)
    _create_detailed_comparison_sheet(workbook, comparison_data)
    _create_individual_fund_sheets(workbook, comparison_data)

    workbook.save(path)
    logger.info("Trading comparison report saved to: %s", path)


def _create_summary_sheet(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Create an executive summary sheet with key metrics."""

    summary = data.get("summary", {})

    sheet = workbook.create_sheet("Executive Summary", 0)
    sheet["A1"] = "Trading Compliance Analysis - Executive Summary"
    sheet["A1"].font = Font(size=14, bold=True)
    sheet["A2"] = f"Date: {data.get('date', '')}"

    metrics = [
        ("Total Funds Analyzed", summary.get("total_funds_analyzed", 0)),
        ("Total Funds Traded", summary.get("total_funds_traded", 0)),
        ("", ""),
        ("Funds Moving OUT of Compliance", summary.get("funds_out_of_compliance", 0)),
        ("Funds Moving INTO Compliance", summary.get("funds_into_compliance", 0)),
        ("Funds with Unchanged Status", summary.get("funds_unchanged", 0)),
        ("Funds with Compliance Changes", summary.get("funds_with_compliance_changes", 0)),
        ("", ""),
        ("Total Violations (Ex-Ante)", summary.get("total_violations_before", 0)),
        ("Total Violations (Ex-Post)", summary.get("total_violations_after", 0)),
        (
            "Net Change in Violations",
            summary.get("total_violations_after", 0)
            - summary.get("total_violations_before", 0),
        ),
        ("", ""),
        ("Total Traded Notional", summary.get("total_traded_notional", 0.0)),
    ]

    row = 4
    for label, value in metrics:
        sheet[f"A{row}"] = label
        sheet[f"A{row}"].font = Font(bold=True)
        sheet[f"B{row}"] = value

        if label == "Funds Moving OUT of Compliance" and value:
            sheet[f"B{row}"].fill = PatternFill(
                start_color="FFCCCC", end_color="FFCCCC", fill_type="solid"
            )
        elif label == "Funds Moving INTO Compliance" and value:
            sheet[f"B{row}"].fill = PatternFill(
                start_color="CCFFCC", end_color="CCFFCC", fill_type="solid"
            )

        row += 1

    sheet.column_dimensions["A"].width = 45
    sheet.column_dimensions["B"].width = 25


def _create_compliance_changes_sheet(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Tab summarising compliance status transitions by fund."""

    sheet = workbook.create_sheet("Compliance Changes")
    headers = [
        "Fund Name",
        "Status Change",
        "Violations Before",
        "Violations After",
        "Net Change",
    ]

    for col, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")

    row = 2
    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        sheet.cell(row=row, column=1, value=fund_name)
        sheet.cell(row=row, column=2, value=fund_data.get("status_change", "UNCHANGED"))
        sheet.cell(row=row, column=3, value=fund_data.get("violations_before", 0))
        sheet.cell(row=row, column=4, value=fund_data.get("violations_after", 0))
        sheet.cell(
            row=row,
            column=5,
            value=fund_data.get("violations_after", 0)
            - fund_data.get("violations_before", 0),
        )

        status = fund_data.get("status_change")
        status_cell = sheet.cell(row=row, column=2)
        if status == "OUT_OF_COMPLIANCE":
            status_cell.fill = PatternFill(
                start_color="FFCCCC", end_color="FFCCCC", fill_type="solid"
            )
        elif status == "INTO_COMPLIANCE":
            status_cell.fill = PatternFill(
                start_color="CCFFCC", end_color="CCFFCC", fill_type="solid"
            )

        row += 1

    for col in range(1, len(headers) + 1):
        sheet.column_dimensions[get_column_letter(col)].width = 20


def _create_trade_activity_sheet(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Tab describing trading activity derived from vest holdings."""

    sheet = workbook.create_sheet("Trade Activity")
    headers = ["Fund", "Total Traded", "Equity", "Treasury", "Options"]

    for col, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")

    row = 2
    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        trade_info = fund_data.get("trade_info", {}) or {}
        if not trade_info:
            continue

        sheet.cell(row=row, column=1, value=fund_name)
        sheet.cell(row=row, column=2, value=float(trade_info.get("total_traded", 0.0)))
        sheet.cell(row=row, column=3, value=float(trade_info.get("equity", 0.0)))
        sheet.cell(row=row, column=4, value=float(trade_info.get("treasury", 0.0)))
        sheet.cell(row=row, column=5, value=float(trade_info.get("options", 0.0)))
        row += 1

    for col in range(1, len(headers) + 1):
        sheet.column_dimensions[get_column_letter(col)].width = 18


def _create_compliance_details_sheet(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Flat table of compliance check changes across all funds."""

    funds_data = data.get("funds", {})
    rows: list[Dict[str, Any]] = []

    for fund_name, fund_info in funds_data.items():
        for check_name, check_info in fund_info.get("checks", {}).items():
            rows.append(
                {
                    "Fund": fund_name,
                    "Compliance Check": check_name,
                    "Status Before": check_info.get("status_before", "UNKNOWN"),
                    "Status After": check_info.get("status_after", "UNKNOWN"),
                    "Violations Before": check_info.get("violations_before", 0),
                    "Violations After": check_info.get("violations_after", 0),
                    "Changed": "Yes" if check_info.get("changed") else "No",
                }
            )

    sheet = workbook.create_sheet("Compliance Details")

    if not rows:
        sheet["A1"] = "No compliance check results available"
        return

    df = pd.DataFrame(rows)
    for row_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), start=1):
        sheet.append(row)
        if row_idx == 1:
            for col in range(1, len(row) + 1):
                cell = sheet.cell(row=row_idx, column=col)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(
                    start_color="DDDDDD", end_color="DDDDDD", fill_type="solid"
                )

    for col in range(1, len(df.columns) + 1):
        sheet.column_dimensions[get_column_letter(col)].width = 22


def _create_detailed_comparison_sheet(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Check-by-check details grouped by fund."""

    sheet = workbook.create_sheet("Detailed Comparison")
    headers = [
        "Fund Name",
        "Compliance Check",
        "Status Before",
        "Status After",
        "Violations Before",
        "Violations After",
        "Changed",
    ]

    for col, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")

    row = 2
    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        for check_name, check_data in sorted(fund_data.get("checks", {}).items()):
            sheet.cell(row=row, column=1, value=fund_name)
            sheet.cell(row=row, column=2, value=check_name)
            sheet.cell(row=row, column=3, value=check_data.get("status_before", "UNKNOWN"))
            sheet.cell(row=row, column=4, value=check_data.get("status_after", "UNKNOWN"))
            sheet.cell(row=row, column=5, value=check_data.get("violations_before", 0))
            sheet.cell(row=row, column=6, value=check_data.get("violations_after", 0))
            changed = bool(check_data.get("changed"))
            sheet.cell(row=row, column=7, value="YES" if changed else "NO")

            if changed:
                for col in range(1, len(headers) + 1):
                    sheet.cell(row=row, column=col).fill = PatternFill(
                        start_color="FFFFCC", end_color="FFFFCC", fill_type="solid"
                    )

            row += 1

    for col in range(1, len(headers) + 1):
        sheet.column_dimensions[get_column_letter(col)].width = 22


def _create_individual_fund_sheets(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Optional per-fund tabs detailing compliance checks."""

    funds = data.get("funds", {})
    for fund_name, fund_info in sorted(funds.items()):
        checks = fund_info.get("checks", {})
        if not checks:
            continue

        sheet_name = fund_name[:31]
        sheet = workbook.create_sheet(sheet_name)
        headers = [
            "Compliance Check",
            "Status Before",
            "Status After",
            "Violations Before",
            "Violations After",
            "Changed",
        ]

        for col, header in enumerate(headers, start=1):
            cell = sheet.cell(row=1, column=col, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(
                start_color="DDDDDD", end_color="DDDDDD", fill_type="solid"
            )

        row = 2
        for check_name, check_info in sorted(checks.items()):
            sheet.cell(row=row, column=1, value=check_name)
            sheet.cell(row=row, column=2, value=check_info.get("status_before", "UNKNOWN"))
            sheet.cell(row=row, column=3, value=check_info.get("status_after", "UNKNOWN"))
            sheet.cell(row=row, column=4, value=check_info.get("violations_before", 0))
            sheet.cell(row=row, column=5, value=check_info.get("violations_after", 0))
            changed = bool(check_info.get("changed"))
            sheet.cell(row=row, column=6, value="YES" if changed else "NO")

            if changed:
                for col in range(1, len(headers) + 1):
                    sheet.cell(row=row, column=col).fill = PatternFill(
                        start_color="FFFFCC", end_color="FFFFCC", fill_type="solid"
                    )

            row += 1

        for col in range(1, len(headers) + 1):
            sheet.column_dimensions[get_column_letter(col)].width = 22


class TradingCompliancePDF(FPDF):
    """Custom PDF class for trading compliance reports."""

    def __init__(self, date: str) -> None:
        super().__init__()
        self.date = date
        self.set_auto_page_break(auto=True, margin=15)

    def header(self) -> None:  # pragma: no cover - provided by FPDF
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Trading Compliance Analysis", 0, 1, "C")
        self.set_font("Arial", "", 10)
        self.cell(0, 8, f"Ex-Ante vs Ex-Post Comparison - {self.date}", 0, 1, "C")
        self.ln(5)

    def footer(self) -> None:  # pragma: no cover - provided by FPDF
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


def generate_trading_pdf_report(comparison_data: Mapping[str, Any], output_path: str) -> None:
    """Create a PDF report for the trading compliance comparison."""

    try:
        pdf = TradingCompliancePDF(date=str(comparison_data.get("date", "")))
        pdf.add_page()

        _add_executive_summary(pdf, comparison_data)

        pdf.add_page()
        _add_compliance_changes(pdf, comparison_data)

        pdf.add_page()
        _add_trade_activity(pdf, comparison_data)

        pdf.add_page()
        _add_detailed_comparison(pdf, comparison_data)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pdf.output(output_path)
        logger.info("Trading PDF report saved to: %s", output_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error generating trading PDF report: %s", exc)
        raise


def _add_executive_summary(pdf: FPDF, data: Mapping[str, Any]) -> None:
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Executive Summary", 0, 1, "L")
    pdf.ln(2)

    summary = data.get("summary", {})

    pdf.set_font("Arial", "B", 11)
    pdf.cell(120, 8, "Metric", 1, 0, "L")
    pdf.cell(60, 8, "Value", 1, 1, "C")

    pdf.set_font("Arial", "", 10)

    metrics = [
        ("Total Funds with Trading Activity", summary.get("total_funds_traded", 0)),
        ("", ""),
        ("Funds Moving OUT of Compliance", summary.get("funds_out_of_compliance", 0)),
        ("Funds Moving INTO Compliance", summary.get("funds_into_compliance", 0)),
        ("Funds with Unchanged Status", summary.get("funds_unchanged", 0)),
        ("", ""),
        ("Total Violations (Ex-Ante)", summary.get("total_violations_before", 0)),
        ("Total Violations (Ex-Post)", summary.get("total_violations_after", 0)),
        (
            "Net Change in Violations",
            summary.get("total_violations_after", 0)
            - summary.get("total_violations_before", 0),
        ),
        ("", ""),
        ("Total Traded Notional", summary.get("total_traded_notional", 0.0)),
    ]

    for label, value in metrics:
        if label == "":
            pdf.cell(120, 6, "", 0, 0)
            pdf.cell(60, 6, "", 0, 1)
            continue

        if label == "Funds Moving OUT of Compliance" and value:
            pdf.set_fill_color(255, 204, 204)
            fill = True
        elif label == "Funds Moving INTO Compliance" and value:
            pdf.set_fill_color(204, 255, 204)
            fill = True
        else:
            fill = False

        pdf.cell(120, 8, label, 1, 0, "L", fill)
        pdf.cell(60, 8, str(value), 1, 1, "C", fill)


def _add_compliance_changes(pdf: FPDF, data: Mapping[str, Any]) -> None:
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Compliance Status Changes by Fund", 0, 1, "L")
    pdf.ln(2)

    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(200, 200, 200)

    col_widths = [50, 40, 25, 25, 25]
    headers = ["Fund", "Status Change", "Viol. Before", "Viol. After", "Net Change"]

    for width, header in zip(col_widths, headers):
        pdf.cell(width, 8, header, 1, 0, "C", True)
    pdf.ln()

    pdf.set_font("Arial", "", 8)

    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        display_name = fund_name if len(fund_name) <= 20 else f"{fund_name[:17]}..."
        status_change = fund_data.get("status_change", "UNCHANGED")
        viol_before = fund_data.get("violations_before", 0)
        viol_after = fund_data.get("violations_after", 0)
        net_change = viol_after - viol_before

        if status_change == "OUT_OF_COMPLIANCE":
            pdf.set_fill_color(255, 204, 204)
            fill = True
        elif status_change == "INTO_COMPLIANCE":
            pdf.set_fill_color(204, 255, 204)
            fill = True
        else:
            fill = False

        pdf.cell(col_widths[0], 7, display_name, 1, 0, "L", fill)
        pdf.cell(col_widths[1], 7, status_change, 1, 0, "C", fill)
        pdf.cell(col_widths[2], 7, str(viol_before), 1, 0, "C", fill)
        pdf.cell(col_widths[3], 7, str(viol_after), 1, 0, "C", fill)
        pdf.cell(col_widths[4], 7, str(net_change), 1, 1, "C", fill)


def _add_trade_activity(pdf: FPDF, data: Mapping[str, Any]) -> None:
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Trade Activity", 0, 1, "L")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(200, 200, 200)

    col_widths = [50, 30, 25, 25, 25]
    headers = ["Fund", "Total Traded", "Equity", "Treasury", "Options"]

    for width, header in zip(col_widths, headers):
        pdf.cell(width, 8, header, 1, 0, "C", True)
    pdf.ln()

    pdf.set_font("Arial", "", 8)

    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        trade_info = fund_data.get("trade_info", {}) or {}
        if not trade_info:
            continue

        display_name = fund_name if len(fund_name) <= 20 else f"{fund_name[:17]}..."
        pdf.cell(col_widths[0], 7, display_name, 1, 0, "L")
        pdf.cell(col_widths[1], 7, f"{float(trade_info.get('total_traded', 0.0)):.2f}", 1, 0, "R")
        pdf.cell(col_widths[2], 7, f"{float(trade_info.get('equity', 0.0)):.2f}", 1, 0, "R")
        pdf.cell(col_widths[3], 7, f"{float(trade_info.get('treasury', 0.0)):.2f}", 1, 0, "R")
        pdf.cell(col_widths[4], 7, f"{float(trade_info.get('options', 0.0)):.2f}", 1, 1, "R")


def _add_detailed_comparison(pdf: FPDF, data: Mapping[str, Any]) -> None:
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Detailed Comparison by Compliance Check", 0, 1, "L")
    pdf.ln(2)

    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        pdf.set_font("Arial", "B", 11)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 8, fund_name, 1, 1, "L", True)

        pdf.set_font("Arial", "B", 8)
        pdf.set_fill_color(200, 200, 200)

        col_widths = [60, 25, 25, 20, 20, 15]
        headers = [
            "Compliance Check",
            "Before",
            "After",
            "Viol. B",
            "Viol. A",
            "Changed",
        ]

        for width, header in zip(col_widths, headers):
            pdf.cell(width, 7, header, 1, 0, "C", True)
        pdf.ln()

        pdf.set_font("Arial", "", 7)

        for check_name, check_data in sorted(fund_data.get("checks", {}).items()):
            display_check = check_name if len(check_name) <= 30 else f"{check_name[:27]}..."
            changed = bool(check_data.get("changed"))

            if changed:
                pdf.set_fill_color(255, 255, 204)
                fill = True
            else:
                fill = False

            pdf.cell(col_widths[0], 6, display_check, 1, 0, "L", fill)
            pdf.cell(
                col_widths[1],
                6,
                check_data.get("status_before", "UNKNOWN"),
                1,
                0,
                "C",
                fill,
            )
            pdf.cell(
                col_widths[2],
                6,
                check_data.get("status_after", "UNKNOWN"),
                1,
                0,
                "C",
                fill,
            )
            pdf.cell(
                col_widths[3],
                6,
                str(check_data.get("violations_before", 0)),
                1,
                0,
                "C",
                fill,
            )
            pdf.cell(
                col_widths[4],
                6,
                str(check_data.get("violations_after", 0)),
                1,
                0,
                "C",
                fill,
            )
            pdf.cell(col_widths[5], 6, "YES" if changed else "NO", 1, 1, "C", fill)

            if pdf.get_y() > 250:
                pdf.add_page()

        pdf.ln(3)


def generate_trading_compliance_reports(
    comparison_data: Mapping[str, Any],
    report_date: Any,
    output_dir: str,
    *,
    file_name_prefix: str = "trading_compliance_results",
    create_pdf: bool = True,
) -> GeneratedTradingComplianceReport:
    """Generate Excel and PDF outputs for trading compliance comparisons."""

    if not comparison_data:
        return GeneratedTradingComplianceReport(None, None)

    date_str = normalize_report_date(report_date)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excel_path = output_path / f"{file_name_prefix}_{date_str}.xlsx"
    pdf_path = output_path / f"{file_name_prefix}_{date_str}.pdf"

    generate_trading_excel_report(comparison_data, str(excel_path))

    pdf_result: Optional[str] = None
    if create_pdf:
        generate_trading_pdf_report(comparison_data, str(pdf_path))
        pdf_result = str(pdf_path)

    return GeneratedTradingComplianceReport(str(excel_path), pdf_result)


__all__ = [
    "GeneratedTradingComplianceReport",
    "generate_trading_compliance_reports",
    "generate_trading_excel_report",
    "generate_trading_pdf_report",
]