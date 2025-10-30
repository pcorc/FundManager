"""Generate Excel and PDF outputs for trading compliance comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import logging

from fpdf import FPDF
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from reporting.report_utils import format_number, normalize_report_date

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
    _create_summary_metrics_sheet(workbook, comparison_data)
    _create_trade_activity_sheet(workbook, comparison_data)
    _create_compliance_details_sheet(workbook, comparison_data)
    _create_detailed_comparison_sheet(workbook, comparison_data)
    _create_individual_fund_sheets(workbook, comparison_data)

    workbook.save(path)
    logger.info("Trading comparison report saved to: %s", path)


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

    aggregate_metrics = summary.get("summary_metrics_totals", {}) or {}
    ex_ante_totals = aggregate_metrics.get("ex_ante", {}) or {}
    ex_post_totals = aggregate_metrics.get("ex_post", {}) or {}

    if ex_ante_totals or ex_post_totals:
        row += 1
        sheet[f"A{row}"] = "Aggregate Summary Metrics"
        sheet[f"A{row}"].font = Font(size=12, bold=True)
        row += 1

        headers = ["Metric", "Ex-Ante", "Ex-Post", "Delta"]
        for col, header in enumerate(headers, start=1):
            cell = sheet.cell(row=row, column=col, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(
                start_color="DDDDDD", end_color="DDDDDD", fill_type="solid"
            )
            cell.alignment = Alignment(horizontal="center")
        row += 1

        metric_labels = {
            "cash_value": "Cash Value",
            "equity_market_value": "Equity Market Value",
            "option_market_value": "Option Market Value",
            "option_delta_adjusted_notional": "Option Delta Adjusted Notional",
            "treasury": "Treasury Market Value",
            "total_assets": "Total Assets",
            "total_net_assets": "Total Net Assets",
        }

        for metric_key, label in metric_labels.items():
            if metric_key not in ex_ante_totals and metric_key not in ex_post_totals:
                continue
            ante_value = float(ex_ante_totals.get(metric_key, 0.0) or 0.0)
            post_value = float(ex_post_totals.get(metric_key, 0.0) or 0.0)
            delta = post_value - ante_value

            sheet.cell(row=row, column=1, value=label)
            sheet.cell(row=row, column=2, value=ante_value)
            sheet.cell(row=row, column=3, value=post_value)
            sheet.cell(row=row, column=4, value=delta)
            row += 1

        sheet.column_dimensions["C"].width = 25
        sheet.column_dimensions["D"].width = 25

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
    """Tab describing detailed trading activity by fund and asset class."""

    sheet = workbook.create_sheet("Trade Activity")
    headers = ["Fund", "Asset Class", "Direction", "Ticker", "Quantity", "Market Value"]

    for col, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")

    asset_labels = {"equity": "Equity", "options": "Options", "treasury": "Treasury"}

    row = 2
    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        trade_info = fund_data.get("trade_info", {}) or {}
        activity = trade_info.get("trade_activity", {}) or {}

        if not activity:
            continue

        for asset_key in ("equity", "options", "treasury"):
            asset_details = activity.get(asset_key)
            if not asset_details:
                continue

            net = asset_details.get("net", {})
            buy_qty = float(net.get("buy_quantity", 0.0) or 0.0)
            sell_qty = float(net.get("sell_quantity", 0.0) or 0.0)
            buy_val = float(net.get("buy_value", 0.0) or 0.0)
            sell_val = float(net.get("sell_value", 0.0) or 0.0)

            if buy_qty or sell_qty or asset_details.get("buys") or asset_details.get("sells"):
                sheet.cell(row=row, column=1, value=fund_name)
                sheet.cell(row=row, column=2, value=asset_labels.get(asset_key, asset_key.title()))
                sheet.cell(row=row, column=3, value="Net Buy")
                buy_qty_cell = sheet.cell(row=row, column=5, value=buy_qty)
                buy_qty_cell.number_format = "#,##0.00"
                buy_val_cell = sheet.cell(row=row, column=6, value=buy_val)
                buy_val_cell.number_format = "#,##0.00"
                row += 1

                sheet.cell(row=row, column=1, value=fund_name)
                sheet.cell(row=row, column=2, value=asset_labels.get(asset_key, asset_key.title()))
                sheet.cell(row=row, column=3, value="Net Sell")
                sell_qty_cell = sheet.cell(row=row, column=5, value=sell_qty)
                sell_qty_cell.number_format = "#,##0.00"
                sell_val_cell = sheet.cell(row=row, column=6, value=sell_val)
                sell_val_cell.number_format = "#,##0.00"
                row += 1

                if asset_details.get("buys"):
                    for trade in asset_details["buys"]:
                        sheet.cell(row=row, column=1, value=fund_name)
                        sheet.cell(row=row, column=2, value=asset_labels.get(asset_key, asset_key.title()))
                        sheet.cell(row=row, column=3, value="Buy")
                        sheet.cell(row=row, column=4, value=trade.get("ticker", ""))
                        qty_cell = sheet.cell(
                            row=row, column=5, value=float(trade.get("quantity", 0.0) or 0.0)
                        )
                        qty_cell.number_format = "#,##0.00"
                        val_cell = sheet.cell(
                            row=row, column=6, value=float(trade.get("market_value", 0.0) or 0.0)
                        )
                        val_cell.number_format = "#,##0.00"
                        row += 1

                if asset_details.get("sells"):
                    for trade in asset_details["sells"]:
                        sheet.cell(row=row, column=1, value=fund_name)
                        sheet.cell(row=row, column=2, value=asset_labels.get(asset_key, asset_key.title()))
                        sheet.cell(row=row, column=3, value="Sell")
                        sheet.cell(row=row, column=4, value=trade.get("ticker", ""))
                        qty_cell = sheet.cell(
                            row=row, column=5, value=float(trade.get("quantity", 0.0) or 0.0)
                        )
                        qty_cell.number_format = "#,##0.00"
                        val_cell = sheet.cell(
                            row=row, column=6, value=float(trade.get("market_value", 0.0) or 0.0)
                        )
                        val_cell.number_format = "#,##0.00"
                        row += 1

                row += 1  # Blank line between asset sections

        row += 1  # Blank line between funds

    for col in range(1, len(headers) + 1):
        sheet.column_dimensions[get_column_letter(col)].width = 18


def _populate_compliance_detail_sheet(sheet, data: Mapping[str, Any]) -> None:
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


def _create_compliance_details_sheet(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Create the compliance details sheet (mirrors detailed comparison)."""

    sheet = workbook.create_sheet("Compliance Details")
    _populate_compliance_detail_sheet(sheet, data)


def _create_detailed_comparison_sheet(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Create the detailed comparison sheet (identical to compliance details)."""

    sheet = workbook.create_sheet("Detailed Comparison")
    _populate_compliance_detail_sheet(sheet, data)


def _create_summary_metrics_sheet(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Display ex-ante vs ex-post summary metrics for each fund."""

    sheet = workbook.create_sheet("Summary Metrics")
    headers = ["Fund", "Metric", "Ex-Ante", "Ex-Post", "Delta"]

    for col, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")

    metric_labels = {
        "cash_value": "Cash Value",
        "equity_market_value": "Equity Market Value",
        "option_market_value": "Option Market Value",
        "option_delta_adjusted_notional": "Option Delta Adjusted Notional",
        "treasury": "Treasury Market Value",
        "total_assets": "Total Assets",
        "total_net_assets": "Total Net Assets",
    }

    row = 2
    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        summary_metrics = fund_data.get("summary_metrics", {}) or {}
        ex_ante = summary_metrics.get("ex_ante", {}) or {}
        ex_post = summary_metrics.get("ex_post", {}) or {}

        if not ex_ante and not ex_post:
            continue

        for metric_key, label in metric_labels.items():
            if metric_key not in ex_ante and metric_key not in ex_post:
                continue
            ante_value = float(ex_ante.get(metric_key, 0.0) or 0.0)
            post_value = float(ex_post.get(metric_key, 0.0) or 0.0)
            delta = post_value - ante_value

            sheet.cell(row=row, column=1, value=fund_name)
            sheet.cell(row=row, column=2, value=label)
            sheet.cell(row=row, column=3, value=ante_value)
            sheet.cell(row=row, column=4, value=post_value)
            sheet.cell(row=row, column=5, value=delta)
            row += 1

        row += 1

    for col in range(1, len(headers) + 1):
        sheet.column_dimensions[get_column_letter(col)].width = 24




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

    pdf.ln(4)
    _add_summary_metrics_section(pdf, data)


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
    pdf.ln(3)

    asset_titles = {"equity": "Equity", "options": "Options", "treasury": "Treasury"}

    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        trade_info = fund_data.get("trade_info", {}) or {}
        activity = trade_info.get("trade_activity", {}) or {}
        if not activity:
            continue

        pdf.set_font("Arial", "B", 11)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 8, fund_name, 1, 1, "L", True)
        pdf.ln(1)

        for asset_key in ("equity", "options", "treasury"):
            asset_details = activity.get(asset_key)
            if not asset_details:
                continue

            net = asset_details.get("net", {})
            buy_qty = float(net.get("buy_quantity", 0.0) or 0.0)
            sell_qty = float(net.get("sell_quantity", 0.0) or 0.0)
            buy_val = float(net.get("buy_value", 0.0) or 0.0)
            sell_val = float(net.get("sell_value", 0.0) or 0.0)

            pdf.set_font("Arial", "B", 9)
            pdf.cell(0, 7, asset_titles.get(asset_key, asset_key.title()), 0, 1, "L")

            pdf.set_font("Arial", "", 8)
            pdf.cell(40, 6, "Net Buy", 1, 0, "L")
            pdf.cell(35, 6, format_number(buy_qty, digits=2), 1, 0, "R")
            pdf.cell(45, 6, format_number(buy_val, digits=2), 1, 1, "R")
            pdf.cell(40, 6, "Net Sell", 1, 0, "L")
            pdf.cell(35, 6, format_number(sell_qty, digits=2), 1, 0, "R")
            pdf.cell(45, 6, format_number(sell_val, digits=2), 1, 1, "R")

            buys = asset_details.get("buys", [])
            sells = asset_details.get("sells", [])

            if buys or sells:
                pdf.ln(1)
                pdf.set_font("Arial", "B", 8)
                pdf.set_fill_color(200, 200, 200)
                col_widths = [20, 40, 35, 45]
                headers = ["Type", "Ticker", "Quantity", "Market Value"]
                for width, header in zip(col_widths, headers):
                    pdf.cell(width, 6, header, 1, 0, "C", True)
                pdf.ln()

                pdf.set_font("Arial", "", 8)
                for trade in buys:
                    if pdf.get_y() > 265:
                        pdf.add_page()
                    pdf.cell(col_widths[0], 6, "Buy", 1, 0, "L")
                    pdf.cell(col_widths[1], 6, trade.get("ticker", ""), 1, 0, "L")
                    pdf.cell(
                        col_widths[2],
                        6,
                        format_number(float(trade.get("quantity", 0.0) or 0.0), digits=2),
                        1,
                        0,
                        "R",
                    )
                    pdf.cell(
                        col_widths[3],
                        6,
                        format_number(float(trade.get("market_value", 0.0) or 0.0), digits=2),
                        1,
                        1,
                        "R",
                    )

                for trade in sells:
                    if pdf.get_y() > 265:
                        pdf.add_page()
                    pdf.cell(col_widths[0], 6, "Sell", 1, 0, "L")
                    pdf.cell(col_widths[1], 6, trade.get("ticker", ""), 1, 0, "L")
                    pdf.cell(
                        col_widths[2],
                        6,
                        format_number(float(trade.get("quantity", 0.0) or 0.0), digits=2),
                        1,
                        0,
                        "R",
                    )
                    pdf.cell(
                        col_widths[3],
                        6,
                        format_number(float(trade.get("market_value", 0.0) or 0.0), digits=2),
                        1,
                        1,
                        "R",
                    )

            pdf.ln(3)

        pdf.ln(4)


def _add_summary_metrics_section(pdf: FPDF, data: Mapping[str, Any]) -> None:
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Fund Summary Metrics", 0, 1, "L")
    pdf.ln(2)

    metric_labels = {
        "cash_value": "Cash Value",
        "equity_market_value": "Equity Market Value",
        "option_market_value": "Option Market Value",
        "option_delta_adjusted_notional": "Option Delta Adjusted Notional",
        "treasury": "Treasury Market Value",
        "total_assets": "Total Assets",
        "total_net_assets": "Total Net Assets",
    }

    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        summary_metrics = fund_data.get("summary_metrics", {}) or {}
        ex_ante = summary_metrics.get("ex_ante", {}) or {}
        ex_post = summary_metrics.get("ex_post", {}) or {}

        if not ex_ante and not ex_post:
            continue

        if pdf.get_y() > 250:
            pdf.add_page()

        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 7, fund_name, 0, 1, "L")
        pdf.set_font("Arial", "B", 8)
        pdf.set_fill_color(200, 200, 200)
        col_widths = [60, 35, 35, 35]
        headers = ["Metric", "Ex-Ante", "Ex-Post", "Delta"]
        for width, header in zip(col_widths, headers):
            pdf.cell(width, 6, header, 1, 0, "C", True)
        pdf.ln()

        pdf.set_font("Arial", "", 8)
        for metric_key, label in metric_labels.items():
            if metric_key not in ex_ante and metric_key not in ex_post:
                continue
            ante_value = float(ex_ante.get(metric_key, 0.0) or 0.0)
            post_value = float(ex_post.get(metric_key, 0.0) or 0.0)
            delta = post_value - ante_value

            if pdf.get_y() > 265:
                pdf.add_page()
                pdf.set_font("Arial", "B", 10)
                pdf.cell(0, 7, fund_name, 0, 1, "L")
                pdf.set_font("Arial", "B", 8)
                pdf.set_fill_color(200, 200, 200)
                for width, header in zip(col_widths, headers):
                    pdf.cell(width, 6, header, 1, 0, "C", True)
                pdf.ln()
                pdf.set_font("Arial", "", 8)

            pdf.cell(col_widths[0], 6, label, 1, 0, "L")
            pdf.cell(col_widths[1], 6, format_number(ante_value, digits=2), 1, 0, "R")
            pdf.cell(col_widths[2], 6, format_number(post_value, digits=2), 1, 0, "R")
            pdf.cell(col_widths[3], 6, format_number(delta, digits=2), 1, 1, "R")

        pdf.ln(3)


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