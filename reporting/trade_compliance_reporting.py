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


def _format_percent(value: float) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "0.00%"

def generate_trading_excel_report(comparison_data: Mapping[str, Any], output_path: str) -> None:
    """Create a multi-tab Excel workbook summarising trading compliance changes."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)

    _create_summary_sheet(workbook, comparison_data)
    _create_trade_activity_sheet(workbook, comparison_data)
    _create_compliance_details_sheet(workbook, comparison_data)

    workbook.save(path)
    logger.info("Trading comparison report saved to: %s", path)


def _create_summary_sheet(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Create a compliance overview sheet with key status changes."""

    summary = data.get("summary", {}) or {}

    sheet = workbook.create_sheet("Compliance Overview", 0)
    sheet["A1"] = "Trading Compliance Analysis - Compliance Overview"
    sheet["A1"].font = Font(size=14, bold=True)
    sheet["A2"] = f"Date: {data.get('date', '')}"

    row = 4
    row = _append_compliance_changes_table(sheet, data, summary, start_row=row)
    _append_fund_summary_metrics_table(sheet, data, start_row=row + 2)

def _append_fund_summary_metrics_table(
    sheet, data: Mapping[str, Any], *, start_row: int
) -> int:
    funds = data.get("funds", {}) or {}

    metric_labels = {
        "cash_value": "Cash Value",
        "equity_market_value": "Equity Market Value",
        "option_market_value": "Option Market Value",
        "option_delta_adjusted_notional": "Option Delta Adjusted Notional",
        "treasury": "Treasury Market Value",
        "total_assets": "Total Assets",
        "total_net_assets": "Total Net Assets",
    }

    populated = False
    row = start_row

    for fund_name, fund_data in sorted(funds.items()):
        summary_metrics = fund_data.get("summary_metrics", {}) or {}
        ex_ante = summary_metrics.get("ex_ante", {}) or {}
        ex_post = summary_metrics.get("ex_post", {}) or {}

        if not ex_ante and not ex_post:
            continue

        if not populated:
            sheet[f"A{row}"] = "Fund Summary Metrics"
            sheet[f"A{row}"].font = Font(size=12, bold=True)
            row += 1

            headers = ["Fund", "Metric", "Ex-Ante", "Ex-Post", "Delta"]
            for col, header in enumerate(headers, start=1):
                cell = sheet.cell(row=row, column=col, value=header)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(
                    start_color="DDDDDD", end_color="DDDDDD", fill_type="solid"
                )
                cell.alignment = Alignment(horizontal="center")
            row += 1
            populated = True

        wrote_row = False
        for metric_key, label in metric_labels.items():
            if metric_key not in ex_ante and metric_key not in ex_post:
                continue

            ante_value = float(ex_ante.get(metric_key, 0.0) or 0.0)
            post_value = float(ex_post.get(metric_key, 0.0) or 0.0)
            delta = post_value - ante_value

            fund_cell = sheet.cell(row=row, column=1, value=fund_name if not wrote_row else "")
            if fund_cell.value:
                fund_cell.font = Font(bold=True)
            sheet.cell(row=row, column=2, value=label)
            ante_cell = sheet.cell(row=row, column=3, value=ante_value)
            ante_cell.number_format = "#,##0.00"
            post_cell = sheet.cell(row=row, column=4, value=post_value)
            post_cell.number_format = "#,##0.00"
            delta_cell = sheet.cell(row=row, column=5, value=delta)
            delta_cell.number_format = "#,##0.00"

            row += 1
            wrote_row = True

        if wrote_row:
            row += 1

    if populated:
        for col_idx in range(1, 6):
            column = get_column_letter(col_idx)
            if sheet.column_dimensions[column].width is None or sheet.column_dimensions[
                column
            ].width < (30 if col_idx == 2 else 20):
                sheet.column_dimensions[column].width = 30 if col_idx == 2 else 20

    return row

def _append_compliance_changes_table(
    sheet, data: Mapping[str, Any], summary: Mapping[str, Any], *, start_row: int
) -> int:
    funds = data.get("funds", {}) or {}

    row = start_row
    sheet[f"A{row}"] = "Compliance Status Changes"
    sheet[f"A{row}"].font = Font(size=12, bold=True)
    row += 1

    sheet[f"A{row}"] = "Total Funds Analyzed"
    sheet[f"A{row}"].font = Font(bold=True)
    sheet[f"B{row}"] = summary.get("total_funds_analyzed", 0)
    sheet[f"B{row}"].font = Font(bold=True)
    sheet[f"B{row}"].alignment = Alignment(horizontal="center")
    row += 1

    sheet[f"A{row}"] = "Total Funds Traded"
    sheet[f"A{row}"].font = Font(bold=True)
    sheet[f"B{row}"] = summary.get("total_funds_traded", 0)
    sheet[f"B{row}"].font = Font(bold=True)
    sheet[f"B{row}"].alignment = Alignment(horizontal="center")
    row += 2

    headers = [
        "Fund Name",
        "Status Change",
        "Violations Before",
        "Violations After",
        "Net Change",
    ]

    for col, header in enumerate(headers, start=1):
        cell = sheet.cell(row=row, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")
    row += 1

    total_before = 0
    total_after = 0
    total_net = 0

    for fund_name, fund_data in sorted(funds.items()):
        violations_before = int(fund_data.get("violations_before", 0) or 0)
        violations_after = int(fund_data.get("violations_after", 0) or 0)
        net_change = violations_after - violations_before

        total_before += violations_before
        total_after += violations_after
        total_net += net_change

        sheet.cell(row=row, column=1, value=fund_name)
        status_cell = sheet.cell(
            row=row, column=2, value=fund_data.get("status_change", "UNCHANGED")
        )
        status_cell.alignment = Alignment(horizontal="center")
        before_cell = sheet.cell(row=row, column=3, value=violations_before)
        before_cell.alignment = Alignment(horizontal="center")
        after_cell = sheet.cell(row=row, column=4, value=violations_after)
        after_cell.alignment = Alignment(horizontal="center")
        net_cell = sheet.cell(row=row, column=5, value=net_change)
        net_cell.alignment = Alignment(horizontal="center")

        status = (fund_data.get("status_change") or "UNCHANGED").upper()
        fill_color: Optional[str]
        if status == "INTO_COMPLIANCE":
            fill_color = "CCFFCC"
        elif status == "OUT_OF_COMPLIANCE":
            fill_color = "FFE0B3"
        elif status == "UNCHANGED" and violations_after > 0:
            fill_color = "FF9999"
        else:
            fill_color = None

        if fill_color:
            fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")
            for col_idx in range(1, len(headers) + 1):
                sheet.cell(row=row, column=col_idx).fill = fill

        row += 1

    totals_fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
    totals_font = Font(bold=True)
    labels = ["Totals", "", total_before, total_after, total_net]
    for col_idx, value in enumerate(labels, start=1):
        cell = sheet.cell(row=row, column=col_idx, value=value)
        cell.font = totals_font
        cell.fill = totals_fill
        if col_idx >= 2:
            cell.alignment = Alignment(horizontal="center")
    row += 1

    for col in range(1, len(headers) + 1):
        column = get_column_letter(col)
        current_width = sheet.column_dimensions[column].width
        min_width = 28 if col == 1 else 22
        if current_width is None or current_width < min_width:
            sheet.column_dimensions[column].width = min_width

    return row

def _create_trade_activity_sheet(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Tab describing detailed trading activity by fund and asset class."""

    sheet = workbook.create_sheet("Trade Activity")
    sheet["A1"] = "Trade Activity Overview"
    sheet["A1"].font = Font(size=12, bold=True)
    sheet.freeze_panes = "A3"

    summary_headers = [
        "Fund",
        "Asset Class",
        "Total Net Assets",
        "Total Assets",
        "Trade Value",
        "% of TNA",
        "% of Total Assets",
        "Ex-Ante Market Value",
        "Ex-Post Market Value",
        "Market Value Delta",
        "Trade vs Ex-Ante %",
        "Ex-Post vs Ex-Ante %",
        "Net Trade Value",
        "Net Buy Qty",
        "Net Sell Qty",
        "Net Buy Value",
        "Net Sell Value",
    ]

    header_row = 3
    for col, header in enumerate(summary_headers, start=1):
        cell = sheet.cell(row=header_row, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")

    asset_labels = {"equity": "Equity", "options": "Options", "treasury": "Treasury"}

    row = header_row + 1
    summary_rows_written = False

    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        trade_info = fund_data.get("trade_info", {}) or {}
        asset_summary = trade_info.get("asset_trade_summary", {}) or {}
        net_trades = trade_info.get("net_trades", {}) or {}
        activity_info = trade_info.get("trade_activity", {}) or {}


        total_net_assets = float(trade_info.get("total_net_assets", 0.0) or 0.0)
        total_assets = float(trade_info.get("total_assets", 0.0) or 0.0)

        for asset_key in ("equity", "options", "treasury"):
            summary = asset_summary.get(asset_key, {}) or {}
            activity_details = activity_info.get(asset_key, {}) or {}

            activity_net: Dict[str, Any]
            if isinstance(activity_details, Mapping):
                activity_net = dict(activity_details.get("net", {}) or {})
            else:
                activity_net = {}

            raw_net = net_trades.get(asset_key, {}) or {}
            if isinstance(raw_net, Mapping):
                net_info = {**activity_net, **raw_net}
            else:
                net_info = activity_net

            trade_value = float(summary.get("trade_value", 0.0) or 0.0)
            market_delta = float(summary.get("market_value_delta", 0.0) or 0.0)
            pct_tna = float(summary.get("pct_of_tna", 0.0) or 0.0)
            pct_assets = float(summary.get("pct_of_total_assets", 0.0) or 0.0)
            ex_ante = float(summary.get("ex_ante_market_value", 0.0) or 0.0)
            ex_post = float(summary.get("ex_post_market_value", 0.0) or 0.0)
            trade_vs_ante = float(summary.get("trade_vs_ex_ante_pct", 0.0) or 0.0)
            post_vs_ante = float(summary.get("ex_post_vs_ex_ante_pct", 0.0) or 0.0)
            buy_qty = float(net_info.get("buy_quantity", net_info.get("buys", 0.0)) or 0.0)
            sell_qty = float(net_info.get("sell_quantity", net_info.get("sells", 0.0)) or 0.0)
            buy_val = float(net_info.get("buy_value", 0.0) or 0.0)
            sell_val = float(net_info.get("sell_value", 0.0) or 0.0)
            raw_net_value = net_info.get("net_value")
            if raw_net_value in (None, ""):
                net_value = buy_val - sell_val
            else:
                try:
                    net_value = float(raw_net_value)
                except (TypeError, ValueError):
                    net_value = buy_val - sell_val

            if not any(
                abs(val) > 0.0
                for val in (
                    trade_value,
                    market_delta,
                    net_value,
                    buy_qty,
                    sell_qty,
                    buy_val,
                    sell_val,
                )
            ):
                continue

            summary_rows_written = True

            sheet.cell(row=row, column=1, value=fund_name)
            sheet.cell(row=row, column=2, value=asset_labels.get(asset_key, asset_key.title()))

            tna_cell = sheet.cell(row=row, column=3, value=total_net_assets)
            tna_cell.number_format = "#,##0.00"

            total_assets_cell = sheet.cell(row=row, column=4, value=total_assets)
            total_assets_cell.number_format = "#,##0.00"

            trade_val_cell = sheet.cell(row=row, column=5, value=trade_value)
            trade_val_cell.number_format = "#,##0.00"

            pct_tna_cell = sheet.cell(row=row, column=6, value=pct_tna)
            pct_tna_cell.number_format = "0.00%"

            pct_assets_cell = sheet.cell(row=row, column=7, value=pct_assets)
            pct_assets_cell.number_format = "0.00%"

            ex_ante_cell = sheet.cell(row=row, column=8, value=ex_ante)
            ex_ante_cell.number_format = "#,##0.00"

            ex_post_cell = sheet.cell(row=row, column=9, value=ex_post)
            ex_post_cell.number_format = "#,##0.00"

            delta_cell = sheet.cell(row=row, column=10, value=market_delta)
            delta_cell.number_format = "#,##0.00"

            trade_vs_ante_cell = sheet.cell(row=row, column=11, value=trade_vs_ante)
            trade_vs_ante_cell.number_format = "0.00%"

            post_vs_ante_cell = sheet.cell(row=row, column=12, value=post_vs_ante)
            post_vs_ante_cell.number_format = "0.00%"

            net_val_cell = sheet.cell(row=row, column=13, value=net_value)
            net_val_cell.number_format = "#,##0.00"

            buy_qty_cell = sheet.cell(row=row, column=14, value=buy_qty)
            buy_qty_cell.number_format = "#,##0.00"

            sell_qty_cell = sheet.cell(row=row, column=15, value=sell_qty)
            sell_qty_cell.number_format = "#,##0.00"

            buy_val_cell = sheet.cell(row=row, column=16, value=buy_val)
            buy_val_cell.number_format = "#,##0.00"

            sell_val_cell = sheet.cell(row=row, column=17, value=sell_val)
            sell_val_cell.number_format = "#,##0.00"

            row += 1

    if not summary_rows_written:
        sheet.cell(row=header_row + 1, column=1, value="No trading activity detected.")

    for col in range(1, len(summary_headers) + 1):
        column = get_column_letter(col)
        width = sheet.column_dimensions[column].width or 0
        desired = 22 if col <= 5 else 18
        if col in (1, 2):
            desired = 24 if col == 1 else 20
        if width < desired:
            sheet.column_dimensions[column].width = desired

    row = max(row + 2, header_row + 3)
    sheet.cell(row=row, column=1, value="Trade Activity Details").font = Font(size=12, bold=True)
    row += 1

    detail_headers = [
        "Fund",
        "Asset Class",
        "Direction",
        "Ticker",
        "Quantity",
        "Market Value",
    ]

    for col, header in enumerate(detail_headers, start=1):
        cell = sheet.cell(row=row, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")

    row += 1
    detail_rows_written = False

    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        trade_info = fund_data.get("trade_info", {}) or {}
        activity = trade_info.get("trade_activity", {}) or {}

        if not activity:
            continue

        fund_rows_written = False

        for asset_key in ("equity", "options", "treasury"):
            asset_details = activity.get(asset_key)
            if not asset_details:
                continue

            asset_rows_written = False
            net = asset_details.get("net", {}) or {}
            buy_qty = float(net.get("buy_quantity", 0.0) or 0.0)
            sell_qty = float(net.get("sell_quantity", 0.0) or 0.0)
            buy_val = float(net.get("buy_value", 0.0) or 0.0)
            sell_val = float(net.get("sell_value", 0.0) or 0.0)

            if any(val != 0.0 for val in (buy_qty, buy_val)):
                detail_rows_written = True
                asset_rows_written = True
                fund_rows_written = True
                sheet.cell(row=row, column=1, value=fund_name)
                sheet.cell(row=row, column=2, value=asset_labels.get(asset_key, asset_key.title()))
                sheet.cell(row=row, column=3, value="Net Buy")
                qty_cell = sheet.cell(row=row, column=5, value=buy_qty)
                qty_cell.number_format = "#,##0.00"
                val_cell = sheet.cell(row=row, column=6, value=buy_val)
                val_cell.number_format = "#,##0.00"
                row += 1

            if any(val != 0.0 for val in (sell_qty, sell_val)):
                detail_rows_written = True
                asset_rows_written = True
                fund_rows_written = True
                sheet.cell(row=row, column=1, value=fund_name)
                sheet.cell(row=row, column=2, value=asset_labels.get(asset_key, asset_key.title()))
                sheet.cell(row=row, column=3, value="Net Sell")
                qty_cell = sheet.cell(row=row, column=5, value=sell_qty)
                qty_cell.number_format = "#,##0.00"
                val_cell = sheet.cell(row=row, column=6, value=sell_val)
                val_cell.number_format = "#,##0.00"
                row += 1

            for trade in asset_details.get("buys", []):
                detail_rows_written = True
                asset_rows_written = True
                fund_rows_written = True
                sheet.cell(row=row, column=1, value=fund_name)
                sheet.cell(row=row, column=2, value=asset_labels.get(asset_key, asset_key.title()))
                sheet.cell(row=row, column=3, value="Buy")
                sheet.cell(row=row, column=4, value=trade.get("ticker", ""))
                qty_cell = sheet.cell(row=row, column=5, value=float(trade.get("quantity", 0.0) or 0.0))
                qty_cell.number_format = "#,##0.00"
                val_cell = sheet.cell(row=row, column=6, value=float(trade.get("market_value", 0.0) or 0.0))
                val_cell.number_format = "#,##0.00"
                row += 1

            for trade in asset_details.get("sells", []):
                detail_rows_written = True
                asset_rows_written = True
                fund_rows_written = True
                sheet.cell(row=row, column=1, value=fund_name)
                sheet.cell(row=row, column=2, value=asset_labels.get(asset_key, asset_key.title()))
                sheet.cell(row=row, column=3, value="Sell")
                sheet.cell(row=row, column=4, value=trade.get("ticker", ""))
                qty_cell = sheet.cell(row=row, column=5, value=float(trade.get("quantity", 0.0) or 0.0))
                qty_cell.number_format = "#,##0.00"
                val_cell = sheet.cell(row=row, column=6, value=float(trade.get("market_value", 0.0) or 0.0))
                val_cell.number_format = "#,##0.00"
                row += 1

            if asset_rows_written:
                row += 1

        if fund_rows_written:
            row += 1

    if not detail_rows_written:
        sheet.cell(row=row, column=1, value="No trade details available.")

    for col in range(1, len(detail_headers) + 1):
        column = get_column_letter(col)
        width = sheet.column_dimensions[column].width or 0
        desired = 18 if col >= 4 else 20
        if width < desired:
            sheet.column_dimensions[column].width = desired

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
            violations_after = int(check_data.get("violations_after", 0) or 0)
            sheet.cell(row=row, column=6, value=violations_after)
            changed = bool(check_data.get("changed"))
            sheet.cell(row=row, column=7, value="YES" if changed else "NO")

            if violations_after > 0:
                fill = PatternFill(
                    start_color="FF9999", end_color="FF9999", fill_type="solid"
                )
                for col in range(1, len(headers) + 1):
                    sheet.cell(row=row, column=col).fill = fill
            elif changed:
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
        pdf.add_page(orientation="L")
        _add_compliance_overview(pdf, comparison_data)

        _add_summary_metrics_section(pdf, comparison_data)

        pdf.add_page(orientation="L")
        _add_detailed_comparison(pdf, comparison_data)

        pdf.add_page(orientation="L")
        _add_trade_activity(pdf, comparison_data)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pdf.output(output_path)
        logger.info("Trading PDF report saved to: %s", output_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error generating trading PDF report: %s", exc)
        raise


def _add_compliance_overview(pdf: FPDF, data: Mapping[str, Any]) -> None:
    summary = data.get("summary", {}) or {}

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Compliance Status Changes", 0, 1, "L")
    pdf.ln(1)

    pdf.set_font("Arial", "", 9)
    pdf.cell(
        0,
        6,
        f"Total Funds Analyzed: {summary.get('total_funds_analyzed', 0)}",
        0,
        1,
        "L",
    )
    pdf.cell(
        0,
        6,
        f"Total Funds Traded: {summary.get('total_funds_traded', 0)}",
        0,
        1,
        "L",
    )
    pdf.ln(2)

    _render_compliance_changes_table(pdf, data)
    pdf.ln(3)


def _render_compliance_changes_table(pdf: FPDF, data: Mapping[str, Any]) -> None:
    col_widths = [55, 40, 25, 25, 25]
    headers = ["Fund", "Status Change", "Viol. Before", "Viol. After", "Net Change"]

    def _draw_header_row(title: Optional[str] = None) -> None:
        if title:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, title, 0, 1, "L")
            pdf.ln(1)
        pdf.set_font("Arial", "B", 8)
        pdf.set_fill_color(200, 200, 200)
        for width, header in zip(col_widths, headers):
            pdf.cell(width, 6, header, 1, 0, "C", True)
        pdf.ln()
        pdf.set_font("Arial", "", 8)

    _draw_header_row()

    total_before = 0
    total_after = 0
    total_net = 0

    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        if pdf.get_y() > 260:
            pdf.add_page()
            _draw_header_row("Compliance Status Changes (cont.)")

        display_name = fund_name if len(fund_name) <= 26 else f"{fund_name[:23]}..."
        status_change = str(fund_data.get("status_change", "UNCHANGED"))
        status_upper = status_change.upper()
        viol_before = int(fund_data.get("violations_before", 0) or 0)
        viol_after = int(fund_data.get("violations_after", 0) or 0)
        net_change = viol_after - viol_before

        total_before += viol_before
        total_after += viol_after
        total_net += net_change

        if status_upper == "INTO_COMPLIANCE":
            pdf.set_fill_color(204, 255, 204)
            fill = True
        elif status_upper == "OUT_OF_COMPLIANCE":
            pdf.set_fill_color(255, 204, 153)
            fill = True
        elif status_upper == "UNCHANGED" and viol_after > 0:
            pdf.set_fill_color(255, 153, 153)
            fill = True
        else:
            fill = False

        pdf.cell(col_widths[0], 6, display_name, 1, 0, "L", fill)
        pdf.cell(col_widths[1], 6, status_change, 1, 0, "C", fill)
        pdf.cell(col_widths[2], 6, str(viol_before), 1, 0, "C", fill)
        pdf.cell(col_widths[3], 6, str(viol_after), 1, 0, "C", fill)
        pdf.cell(col_widths[4], 6, str(net_change), 1, 1, "C", fill)

    if pdf.get_y() > 260:
        pdf.add_page()
        _draw_header_row("Compliance Status Changes (cont.)")

    pdf.set_font("Arial", "B", 8)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(col_widths[0], 6, "Totals", 1, 0, "L", True)
    pdf.cell(col_widths[1], 6, "", 1, 0, "C", True)
    pdf.cell(col_widths[2], 6, str(total_before), 1, 0, "C", True)
    pdf.cell(col_widths[3], 6, str(total_after), 1, 0, "C", True)
    pdf.cell(col_widths[4], 6, str(total_net), 1, 1, "C", True)
    pdf.set_font("Arial", "", 8)


def _add_trade_activity(pdf: FPDF, data: Mapping[str, Any]) -> None:
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Trade Activity", 0, 1, "L")
    pdf.ln(3)

    funds_payload = data.get("funds", {}) or {}
    asset_labels = {"equity": "Equity", "options": "Options", "treasury": "Treasury"}

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[tuple[str, str, str, str, float, float]] = []

    for fund_name, fund_data in sorted(funds_payload.items()):
        trade_info = fund_data.get("trade_info", {}) or {}
        asset_summary = trade_info.get("asset_trade_summary", {}) or {}
        net_trades = trade_info.get("net_trades", {}) or {}
        activity = trade_info.get("trade_activity", {}) or {}

        total_net_assets = float(trade_info.get("total_net_assets", 0.0) or 0.0)
        total_assets = float(trade_info.get("total_assets", 0.0) or 0.0)

        for asset_key in ("equity", "options", "treasury"):
            summary = asset_summary.get(asset_key, {}) or {}
            activity_details = activity.get(asset_key, {}) or {}

            activity_net = {}
            if isinstance(activity_details, Mapping):
                activity_net = dict(activity_details.get("net", {}) or {})
            raw_net = net_trades.get(asset_key, {}) or {}
            if isinstance(raw_net, Mapping):
                net_info = {**activity_net, **raw_net}
            else:
                net_info = activity_net

            trade_value = float(summary.get("trade_value", 0.0) or 0.0)
            market_delta = float(summary.get("market_value_delta", 0.0) or 0.0)
            pct_tna = float(summary.get("pct_of_tna", 0.0) or 0.0)
            pct_assets = float(summary.get("pct_of_total_assets", 0.0) or 0.0)
            ex_ante = float(summary.get("ex_ante_market_value", 0.0) or 0.0)
            ex_post = float(summary.get("ex_post_market_value", 0.0) or 0.0)

            buy_qty = float(net_info.get("buy_quantity", net_info.get("buys", 0.0)) or 0.0)
            sell_qty = float(net_info.get("sell_quantity", net_info.get("sells", 0.0)) or 0.0)
            buy_val = float(net_info.get("buy_value", 0.0) or 0.0)
            sell_val = float(net_info.get("sell_value", 0.0) or 0.0)
            raw_net_value = net_info.get("net_value")
            if raw_net_value in (None, ""):
                net_value = buy_val - sell_val
            else:
                try:
                    net_value = float(raw_net_value)
                except (TypeError, ValueError):
                    net_value = buy_val - sell_val

            if not any(
                abs(val) > 0
                for val in (
                    trade_value,
                    market_delta,
                    net_value,
                    buy_qty,
                    sell_qty,
                    buy_val,
                    sell_val,
                )
            ):
                continue

            summary_rows.append(
                {
                    "fund": fund_name,
                    "asset": asset_labels.get(asset_key, asset_key.title()),
                    "total_net_assets": total_net_assets,
                    "total_assets": total_assets,
                    "trade_value": trade_value,
                    "pct_tna": pct_tna,
                    "pct_assets": pct_assets,
                    "ex_ante": ex_ante,
                    "ex_post": ex_post,
                    "delta": market_delta,
                    "net_value": net_value,
                    "buy_value": buy_val,
                    "sell_value": sell_val,
                }
            )

            if buy_qty or buy_val:
                detail_rows.append(
                    (
                        fund_name,
                        asset_labels.get(asset_key, asset_key.title()),
                        "Net Buy",
                        "",
                        buy_qty,
                        buy_val,
                    )
                )
            if sell_qty or sell_val:
                detail_rows.append(
                    (
                        fund_name,
                        asset_labels.get(asset_key, asset_key.title()),
                        "Net Sell",
                        "",
                        sell_qty,
                        sell_val,
                    )
                )

            for trade in activity_details.get("buys", []):
                quantity = float(trade.get("quantity", 0.0) or 0.0)
                value = float(trade.get("market_value", 0.0) or 0.0)
                if not quantity and not value:
                    continue
                detail_rows.append(
                    (
                        fund_name,
                        asset_labels.get(asset_key, asset_key.title()),
                        "Buy",
                        str(trade.get("ticker", "")),
                        quantity,
                        value,
                    )
                )

            for trade in activity_details.get("sells", []):
                quantity = float(trade.get("quantity", 0.0) or 0.0)
                value = float(trade.get("market_value", 0.0) or 0.0)
                if not quantity and not value:
                    continue
                detail_rows.append(
                    (
                        fund_name,
                        asset_labels.get(asset_key, asset_key.title()),
                        "Sell",
                        str(trade.get("ticker", "")),
                        quantity,
                        value,
                    )
                )

    if not summary_rows and not detail_rows:
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 6, "No trading activity detected.", 0, 1, "L")
        return

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    summary_headers = [
        "Fund",
        "Asset Class",
        "Total Net Assets",
        "Total Assets",
        "Trade Value",
        "% of TNA",
        "% of Assets",
        "Ex-Ante MV",
        "Ex-Post MV",
        "Market Value Delta",
        "Net Trade Value",
        "Net Buy Value",
        "Net Sell Value",
    ]
    summary_rel_widths = [1.1, 1.0, 1.15, 1.1, 1.1, 0.85, 0.85, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0]
    total_weight = sum(summary_rel_widths)
    summary_widths = [usable_width * w / total_weight for w in summary_rel_widths]

    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 6, "Trade Activity Summary", 0, 1, "L")
    pdf.ln(1)

    pdf.set_font("Arial", "B", 6)
    pdf.set_fill_color(200, 200, 200)
    for width, header in zip(summary_widths, summary_headers):
        pdf.cell(width, 5.5, header, 1, 0, "C", True)
    pdf.ln(5.5)

    pdf.set_font("Arial", "", 6)
    page_limit = pdf.h - pdf.b_margin - 10
    row_height = 5.0
    last_fund = None

    for row in sorted(summary_rows, key=lambda item: (item["fund"], item["asset"])):
        if pdf.get_y() + row_height > page_limit:
            pdf.add_page(orientation=getattr(pdf, "cur_orientation", "L"))
            pdf.set_font("Arial", "B", 6)
            pdf.set_fill_color(200, 200, 200)
            for width, header in zip(summary_widths, summary_headers):
                pdf.cell(width, 5.5, header, 1, 0, "C", True)
            pdf.ln(5.5)
            pdf.set_font("Arial", "", 6)

        fund_label = row["fund"] if row["fund"] != last_fund else ""
        if fund_label:
            pdf.set_font("Arial", "B", 6)
        pdf.cell(summary_widths[0], row_height, fund_label, 1, 0, "L")
        if fund_label:
            pdf.set_font("Arial", "", 6)
        pdf.cell(summary_widths[1], row_height, row["asset"], 1, 0, "L")

        pdf.cell(summary_widths[2], row_height, format_number(row["total_net_assets"], digits=2), 1, 0, "R")
        pdf.cell(summary_widths[3], row_height, format_number(row["total_assets"], digits=2), 1, 0, "R")
        pdf.cell(summary_widths[4], row_height, format_number(row["trade_value"], digits=2), 1, 0, "R")
        pdf.cell(summary_widths[5], row_height, _format_percent(row["pct_tna"]), 1, 0, "R")
        pdf.cell(summary_widths[6], row_height, _format_percent(row["pct_assets"]), 1, 0, "R")
        pdf.cell(summary_widths[7], row_height, format_number(row["ex_ante"], digits=2), 1, 0, "R")
        pdf.cell(summary_widths[8], row_height, format_number(row["ex_post"], digits=2), 1, 0, "R")
        pdf.cell(summary_widths[9], row_height, format_number(row["delta"], digits=2), 1, 0, "R")
        pdf.cell(summary_widths[10], row_height, format_number(row["net_value"], digits=2), 1, 0, "R")
        pdf.cell(summary_widths[11], row_height, format_number(row["buy_value"], digits=2), 1, 0, "R")
        pdf.cell(summary_widths[12], row_height, format_number(row["sell_value"], digits=2), 1, 1, "R")

        last_fund = row["fund"]

    pdf.ln(3)

    if detail_rows:
        detail_headers = ["Fund", "Asset Class", "Direction", "Ticker", "Quantity", "Market Value"]
        detail_rel_widths = [1.2, 1.0, 0.9, 1.0, 0.9, 1.1]
        detail_total = sum(detail_rel_widths)
        detail_widths = [usable_width * w / detail_total for w in detail_rel_widths]

        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 6, "Trade Activity Details", 0, 1, "L")
        pdf.ln(1)

        pdf.set_font("Arial", "B", 6)
        pdf.set_fill_color(200, 200, 200)
        for width, header in zip(detail_widths, detail_headers):
            pdf.cell(width, 5.5, header, 1, 0, "C", True)
        pdf.ln(5.5)

        pdf.set_font("Arial", "", 6)
        last_fund = None
        row_height = 5.0

        for fund_name, asset, direction, ticker, quantity, value in detail_rows:
            if pdf.get_y() + row_height > page_limit:
                pdf.add_page(orientation=getattr(pdf, "cur_orientation", "L"))
                pdf.set_font("Arial", "B", 6)
                pdf.set_fill_color(200, 200, 200)
                for width, header in zip(detail_widths, detail_headers):
                    pdf.cell(width, 5.5, header, 1, 0, "C", True)
                pdf.ln(5.5)
                pdf.set_font("Arial", "", 6)

            fund_label = fund_name if fund_name != last_fund else ""
            if fund_label:
                pdf.set_font("Arial", "B", 6)
            pdf.cell(detail_widths[0], row_height, fund_label, 1, 0, "L")
            if fund_label:
                pdf.set_font("Arial", "", 6)

            pdf.cell(detail_widths[1], row_height, asset, 1, 0, "L")
            pdf.cell(detail_widths[2], row_height, direction, 1, 0, "L")
            pdf.cell(detail_widths[3], row_height, ticker, 1, 0, "L")
            pdf.cell(detail_widths[4], row_height, format_number(quantity, digits=2), 1, 0, "R")
            pdf.cell(detail_widths[5], row_height, format_number(value, digits=2), 1, 1, "R")

            last_fund = fund_name

def _add_summary_metrics_section(pdf: FPDF, data: Mapping[str, Any]) -> None:
    metric_labels = {
        "cash_value": "Cash Value",
        "equity_market_value": "Equity Market Value",
        "option_market_value": "Option Market Value",
        "option_delta_adjusted_notional": "Option Delta Adjusted Notional",
        "treasury": "Treasury Market Value",
        "total_assets": "Total Assets",
        "total_net_assets": "Total Net Assets",
    }

    funds = sorted(data.get("funds", {}).keys())
    if not funds:
        return

    rows: list[list[str]] = []
    for metric_key, label in metric_labels.items():
        row: list[str] = [label]
        has_value = False
        for fund_name in funds:
            summary_metrics = (
                data.get("funds", {}).get(fund_name, {}).get("summary_metrics", {}) or {}
            )
            ex_ante = summary_metrics.get("ex_ante", {}) or {}
            ex_post = summary_metrics.get("ex_post", {}) or {}

            if metric_key not in ex_ante and metric_key not in ex_post:
                row.append("")
                continue

            ante_value = float(ex_ante.get(metric_key, 0.0) or 0.0)
            post_value = float(ex_post.get(metric_key, 0.0) or 0.0)
            delta = post_value - ante_value
            combined = " | ".join(
                [
                    f"EA: {format_number(ante_value, digits=2)}",
                    f"EP: {format_number(post_value, digits=2)}",
                    f"Delta: {format_number(delta, digits=2)}",
                ]
            )
            row.append(combined)
            has_value = has_value or any((ante_value, post_value, delta))

        if has_value:
            rows.append(row)

    if not rows:
        return

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    metric_width = min(50.0, usable_width * 0.22)
    remaining_width = usable_width - metric_width
    value_columns = max(len(funds), 1)
    value_width = remaining_width / value_columns if value_columns else remaining_width
    column_widths = [metric_width] + [value_width] * value_columns

    page_limit = pdf.h - pdf.b_margin - 10

    def _draw_header(title: str) -> None:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, title, 0, 1, "L")
        pdf.ln(2)
        pdf.set_font("Arial", "B", 7)
        pdf.set_fill_color(200, 200, 200)
        headers = ["Metric", *funds]
        for width, header in zip(column_widths, headers):
            pdf.cell(width, 6, header, 1, 0, "C", True)
        pdf.ln()
        pdf.set_font("Arial", "", 7)

    if pdf.get_y() > page_limit:
        pdf.add_page(orientation=getattr(pdf, "cur_orientation", "L"))

    _draw_header("Fund Summary Metrics")

    row_height = 5.5
    for idx, row in enumerate(rows):
        if pdf.get_y() + row_height > page_limit:
            pdf.add_page(orientation=getattr(pdf, "cur_orientation", "L"))
            _draw_header("Fund Summary Metrics (cont.)")

        pdf.cell(column_widths[0], row_height, row[0], 1, 0, "L")
        for width, value in zip(column_widths[1:], row[1:]):
            pdf.cell(width, row_height, value, 1, 0, "L")
        pdf.ln(row_height)

        if idx < len(rows) - 1 and pdf.get_y() <= page_limit:
            pdf.ln(0.5)

def _add_detailed_comparison(pdf: FPDF, data: Mapping[str, Any]) -> None:
    column_gap = 10
    column_widths = [26, 58, 17, 17, 15]
    headers = ["Fund", "Compliance Check", "Before", "After", "Changed"]
    table_width = sum(column_widths)
    row_height = 6
    header_height = 6

    def _render_heading() -> None:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Detailed Comparison by Compliance Check", 0, 1, "L")
        pdf.ln(2)

    def _reset_columns() -> tuple[list[float], list[float]]:
        start_y = pdf.get_y()
        positions = [
            pdf.l_margin,
            pdf.l_margin + table_width + column_gap,
        ]
        heights = [start_y, start_y]
        return positions, heights

    def _max_y() -> float:
        return pdf.h - pdf.b_margin

    _render_heading()
    column_positions, column_heights = _reset_columns()
    current_column = 0

    def _ensure_space(required_height: float) -> None:
        nonlocal current_column, column_positions, column_heights

        while column_heights[current_column] + required_height > _max_y():
            current_column += 1
            if current_column >= len(column_positions):
                pdf.add_page(orientation="L")
                _render_heading()
                column_positions, column_heights = _reset_columns()
                current_column = 0
                if column_heights[current_column] + required_height > _max_y():
                    break

    funds = sorted(data.get("funds", {}).items())
    for fund_name, fund_data in funds:
        checks = sorted(fund_data.get("checks", {}).items())
        if not checks:
            continue

        table_height = header_height + len(checks) * row_height + 4
        _ensure_space(table_height)

        x_position = column_positions[current_column]
        pdf.set_xy(x_position, column_heights[current_column])

        pdf.set_font("Arial", "B", 8)
        pdf.set_fill_color(200, 200, 200)
        for width, header in zip(column_widths, headers):
            pdf.cell(width, header_height, header, 1, 0, "C", True)
        pdf.ln()

        pdf.set_x(x_position)
        pdf.set_font("Arial", "", 7)

        for index, (check_name, check_data) in enumerate(checks):
            fund_label = fund_name if index == 0 else ""
            fund_display = fund_label if len(fund_label) <= 22 else f"{fund_label[:19]}..."
            display_check = check_name if len(check_name) <= 30 else f"{check_name[:27]}..."
            before_status = str(check_data.get("status_before", "UNKNOWN"))
            after_status = str(check_data.get("status_after", "UNKNOWN"))
            after_upper = after_status.upper()
            changed_flag = "YES" if bool(check_data.get("changed")) else "NO"

            pdf.set_x(x_position)
            if fund_label:
                pdf.set_font("Arial", "B", 7)
            pdf.cell(column_widths[0], row_height, fund_display, 1, 0, "L")
            if fund_label:
                pdf.set_font("Arial", "", 7)

            if after_upper == "FAIL":
                pdf.set_fill_color(255, 204, 204)
                pdf.cell(column_widths[1], row_height, display_check, 1, 0, "L", True)
                pdf.set_fill_color(255, 255, 255)
            else:
                pdf.cell(column_widths[1], row_height, display_check, 1, 0, "L")

            pdf.cell(column_widths[2], row_height, before_status, 1, 0, "C")
            pdf.cell(column_widths[3], row_height, after_status, 1, 0, "C")
            pdf.cell(column_widths[4], row_height, changed_flag, 1, 1, "C")

        column_heights[current_column] = pdf.get_y() + 4

        if column_heights[current_column] > _max_y():
            _ensure_space(0)

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