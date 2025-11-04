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
    _create_summary_metrics_sheet(workbook, comparison_data)
    _create_trade_activity_sheet(workbook, comparison_data)
    _create_compliance_details_sheet(workbook, comparison_data)
    _create_individual_fund_sheets(workbook, comparison_data)

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
    _append_aggregate_summary_metrics(sheet, summary, start_row=row + 1)

def _append_aggregate_summary_metrics(sheet, summary: Mapping[str, Any], *, start_row: int) -> int:
    aggregate_metrics = summary.get("summary_metrics_totals", {}) or {}
    ex_ante_totals = aggregate_metrics.get("ex_ante", {}) or {}
    ex_post_totals = aggregate_metrics.get("ex_post", {}) or {}

    if not (ex_ante_totals or ex_post_totals):
        return start_row

    row = start_row
    sheet[f"A{row}"] = "Aggregate Summary Metrics"
    sheet[f"A{row}"].font = Font(size=12, bold=True)
    row += 1

    headers = ["Metric", "Ex-Ante", "Ex-Post", "Delta"]
    for col, header in enumerate(headers, start=1):
        cell = sheet.cell(row=row, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
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

    if sheet.column_dimensions["C"].width is None or sheet.column_dimensions["C"].width < 25:
        sheet.column_dimensions["C"].width = 25
    if sheet.column_dimensions["D"].width is None or sheet.column_dimensions["D"].width < 25:
        sheet.column_dimensions["D"].width = 25
    return row + 1


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
    ]

    row = 2
    for col, header in enumerate(summary_headers, start=1):
        cell = sheet.cell(row=row, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")

    asset_labels = {"equity": "Equity", "options": "Options", "treasury": "Treasury"}

    row += 1
    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        trade_info = fund_data.get("trade_info", {}) or {}
        asset_summary = trade_info.get("asset_trade_summary", {}) or {}
        net_trades = trade_info.get("net_trades", {}) or {}

        total_net_assets = float(trade_info.get("total_net_assets", 0.0) or 0.0)
        total_assets = float(trade_info.get("total_assets", 0.0) or 0.0)

        for asset_key in ("equity", "options", "treasury"):
            summary = asset_summary.get(asset_key, {})
            net_info = net_trades.get(asset_key, {})

            trade_value = float(summary.get("trade_value", 0.0) or 0.0)
            market_delta = float(summary.get("market_value_delta", 0.0) or 0.0)
            net_value = float(net_info.get("net_value", 0.0) or 0.0)

            if not any(
                abs(val) > 0.0
                for val in (
                    trade_value,
                    market_delta,
                    net_value,
                )
            ):
                continue

            sheet.cell(row=row, column=1, value=fund_name)
            sheet.cell(row=row, column=2, value=asset_labels.get(asset_key, asset_key.title()))

            tna_cell = sheet.cell(row=row, column=3, value=total_net_assets)
            tna_cell.number_format = "#,##0.00"

            total_assets_cell = sheet.cell(row=row, column=4, value=total_assets)
            total_assets_cell.number_format = "#,##0.00"

            trade_val_cell = sheet.cell(row=row, column=5, value=trade_value)
            trade_val_cell.number_format = "#,##0.00"

            pct_tna_cell = sheet.cell(
                row=row,
                column=6,
                value=float(summary.get("pct_of_tna", 0.0) or 0.0),
            )
            pct_tna_cell.number_format = "0.00%"

            pct_assets_cell = sheet.cell(
                row=row,
                column=7,
                value=float(summary.get("pct_of_total_assets", 0.0) or 0.0),
            )
            pct_assets_cell.number_format = "0.00%"

            ex_ante_cell = sheet.cell(
                row=row,
                column=8,
                value=float(summary.get("ex_ante_market_value", 0.0) or 0.0),
            )
            ex_ante_cell.number_format = "#,##0.00"

            ex_post_cell = sheet.cell(
                row=row,
                column=9,
                value=float(summary.get("ex_post_market_value", 0.0) or 0.0),
            )
            ex_post_cell.number_format = "#,##0.00"

            delta_cell = sheet.cell(row=row, column=10, value=market_delta)
            delta_cell.number_format = "#,##0.00"

            trade_vs_ante_cell = sheet.cell(
                row=row,
                column=11,
                value=float(summary.get("trade_vs_ex_ante_pct", 0.0) or 0.0),
            )
            trade_vs_ante_cell.number_format = "0.00%"

            post_vs_ante_cell = sheet.cell(
                row=row,
                column=12,
                value=float(summary.get("ex_post_vs_ex_ante_pct", 0.0) or 0.0),
            )
            post_vs_ante_cell.number_format = "0.00%"

            net_val_cell = sheet.cell(row=row, column=13, value=net_value)
            net_val_cell.number_format = "#,##0.00"

            row += 1

    for col in range(1, len(summary_headers) + 1):
        column = get_column_letter(col)
        sheet.column_dimensions[column].width = 20

    row += 2
    detail_header_row = row
    detail_headers = [
        "Fund",
        "Asset Class",
        "Direction",
        "Ticker",
        "Quantity",
        "Market Value",
    ]

    for col, header in enumerate(detail_headers, start=1):
        cell = sheet.cell(row=detail_header_row, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")

    row = detail_header_row + 1
    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        trade_info = fund_data.get("trade_info", {}) or {}
        activity = trade_info.get("trade_activity", {}) or {}

        for asset_key in ("equity", "options", "treasury"):
            asset_details = activity.get(asset_key)
            if not asset_details:
                continue

            net = asset_details.get("net", {})
            buy_qty = float(net.get("buy_quantity", 0.0) or 0.0)
            sell_qty = float(net.get("sell_quantity", 0.0) or 0.0)
            buy_val = float(net.get("buy_value", 0.0) or 0.0)
            sell_val = float(net.get("sell_value", 0.0) or 0.0)

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

            for trade in asset_details.get("buys", []):
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

            for trade in asset_details.get("sells", []):
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

            row += 1

    for col in range(1, len(detail_headers) + 1):
        column = get_column_letter(col)
        width = sheet.column_dimensions[column].width
        if width is None or width < 18:
            sheet.column_dimensions[column].width = 18


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
        _add_compliance_overview(pdf, comparison_data)

        pdf.add_page(orientation="L")
        _add_summary_metrics_section(pdf, comparison_data)

        pdf.add_page(orientation="P")
        _add_trade_activity(pdf, comparison_data)

        pdf.add_page(orientation="P")
        _add_detailed_comparison(pdf, comparison_data)

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
    _render_aggregate_summary_metrics(pdf, summary)

def _render_aggregate_summary_metrics(pdf: FPDF, summary: Mapping[str, Any]) -> None:
    aggregate_metrics = summary.get("summary_metrics_totals", {}) or {}
    ex_ante_totals = aggregate_metrics.get("ex_ante", {}) or {}
    ex_post_totals = aggregate_metrics.get("ex_post", {}) or {}

    if not (ex_ante_totals or ex_post_totals):
        return

    if pdf.get_y() > 250:
        pdf.add_page()

    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 7, "Aggregate Summary Metrics", 0, 1, "L")
    pdf.ln(1)

    headers = ["Metric", "Ex-Ante", "Ex-Post", "Delta"]
    col_widths = [60, 35, 35, 35]

    pdf.set_font("Arial", "B", 8)
    pdf.set_fill_color(200, 200, 200)
    for width, header in zip(col_widths, headers):
        pdf.cell(width, 6, header, 1, 0, "C", True)
    pdf.ln()

    pdf.set_font("Arial", "", 8)

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

        pdf.cell(col_widths[0], 6, label, 1, 0, "L")
        pdf.cell(col_widths[1], 6, format_number(ante_value, digits=2), 1, 0, "R")
        pdf.cell(col_widths[2], 6, format_number(post_value, digits=2), 1, 0, "R")
        pdf.cell(col_widths[3], 6, format_number(delta, digits=2), 1, 1, "R")


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

    asset_titles = {"equity": "Equity", "options": "Options", "treasury": "Treasury"}

    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        trade_info = fund_data.get("trade_info", {}) or {}
        activity = trade_info.get("trade_activity", {}) or {}
        asset_summary = trade_info.get("asset_trade_summary", {}) or {}
        net_trades = trade_info.get("net_trades", {}) or {}

        if not activity and not asset_summary:
            continue

        pdf.set_font("Arial", "B", 11)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 8, fund_name, 1, 1, "L", True)
        pdf.ln(1)

        total_net_assets = float(trade_info.get("total_net_assets", 0.0) or 0.0)
        total_assets = float(trade_info.get("total_assets", 0.0) or 0.0)
        pdf.set_font("Arial", "", 8)
        pdf.cell(
            0,
            5,
            f"Total Net Assets: {format_number(total_net_assets, digits=2)}",
            0,
            1,
            "L",
        )
        pdf.cell(
            0,
            5,
            f"Total Assets: {format_number(total_assets, digits=2)}",
            0,
            1,
            "L",
        )

        if asset_summary:
            pdf.ln(1)
            pdf.set_font("Arial", "B", 8)
            pdf.set_fill_color(200, 200, 200)
            col_widths = [28, 28, 20, 20, 28, 28, 28]
            headers = [
                "Asset Class",
                "Trade Value",
                "% TNA",
                "% Assets",
                "Ex-Ante MV",
                "Ex-Post MV",
                "Delta",
            ]
            for width, header in zip(col_widths, headers):
                pdf.cell(width, 6, header, 1, 0, "C", True)
            pdf.ln()

            pdf.set_font("Arial", "", 7)
            asset_labels = {"equity": "Equity", "options": "Options", "treasury": "Treasury"}
            for asset_key in ("equity", "options", "treasury"):
                summary = asset_summary.get(asset_key)
                if not summary:
                    continue

                if pdf.get_y() > 260:
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 8)
                    pdf.set_fill_color(200, 200, 200)
                    for width, header in zip(col_widths, headers):
                        pdf.cell(width, 6, header, 1, 0, "C", True)
                    pdf.ln()
                    pdf.set_font("Arial", "", 7)

                trade_value = float(summary.get("trade_value", 0.0) or 0.0)
                pct_tna = float(summary.get("pct_of_tna", 0.0) or 0.0)
                pct_assets = float(summary.get("pct_of_total_assets", 0.0) or 0.0)
                ex_ante = float(summary.get("ex_ante_market_value", 0.0) or 0.0)
                ex_post = float(summary.get("ex_post_market_value", 0.0) or 0.0)
                delta = float(summary.get("market_value_delta", 0.0) or 0.0)
                trade_vs_ante = float(summary.get("trade_vs_ex_ante_pct", 0.0) or 0.0)
                post_vs_ante = float(summary.get("ex_post_vs_ex_ante_pct", 0.0) or 0.0)
                net_value = float(net_trades.get(asset_key, {}).get("net_value", 0.0) or 0.0)

                pdf.cell(col_widths[0], 6, asset_labels.get(asset_key, asset_key.title()), 1, 0, "L")
                pdf.cell(
                    col_widths[1],
                    6,
                    format_number(trade_value, digits=2),
                    1,
                    0,
                    "R",
                )
                pdf.cell(col_widths[2], 6, _format_percent(pct_tna), 1, 0, "R")
                pdf.cell(col_widths[3], 6, _format_percent(pct_assets), 1, 0, "R")
                pdf.cell(col_widths[4], 6, format_number(ex_ante, digits=2), 1, 0, "R")
                pdf.cell(col_widths[5], 6, format_number(ex_post, digits=2), 1, 0, "R")
                pdf.cell(col_widths[6], 6, format_number(delta, digits=2), 1, 1, "R")

                pdf.set_font("Arial", "I", 6)
                pdf.cell(
                    0,
                    4,
                    "Trade vs Ex-Ante: "
                    f"{_format_percent(trade_vs_ante)} | Ex-Post vs Ex-Ante: "
                    f"{_format_percent(post_vs_ante)} | Net Trade Value: "
                    f"{format_number(net_value, digits=2)}",
                    0,
                    1,
                    "L",
                )
                pdf.set_font("Arial", "", 7)

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
    metric_labels = {
        "cash_value": "Cash Value",
        "equity_market_value": "Equity Market Value",
        "option_market_value": "Option Market Value",
        "option_delta_adjusted_notional": "Option Delta Adjusted Notional",
        "treasury": "Treasury Market Value",
        "total_assets": "Total Assets",
        "total_net_assets": "Total Net Assets",
    }

    column_widths = [55, 85, 45, 45, 45]
    headers = ["Fund", "Metric", "Ex-Ante", "Ex-Post", "Delta"]
    available_height = 190

    def _draw_header(title: str) -> None:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, title, 0, 1, "L")
        pdf.ln(2)
        pdf.set_font("Arial", "B", 8)
        pdf.set_fill_color(200, 200, 200)
        for width, header in zip(column_widths, headers):
            pdf.cell(width, 7, header, 1, 0, "C", True)
        pdf.ln()
        pdf.set_font("Arial", "", 8)

    _draw_header("Fund Summary Metrics")

    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        summary_metrics = fund_data.get("summary_metrics", {}) or {}
        ex_ante = summary_metrics.get("ex_ante", {}) or {}
        ex_post = summary_metrics.get("ex_post", {}) or {}

        if not ex_ante and not ex_post:
            continue

        wrote_metrics = False

        for metric_key, label in metric_labels.items():
            if metric_key not in ex_ante and metric_key not in ex_post:
                continue

            if pdf.get_y() > available_height:
                pdf.add_page(orientation="L")
                _draw_header("Fund Summary Metrics (cont.)")

            ante_value = float(ex_ante.get(metric_key, 0.0) or 0.0)
            post_value = float(ex_post.get(metric_key, 0.0) or 0.0)
            delta = post_value - ante_value

            fund_label = fund_name if not wrote_metrics else ""
            pdf.cell(column_widths[0], 6, fund_label, 1, 0, "L")
            pdf.cell(column_widths[1], 6, label, 1, 0, "L")
            pdf.cell(column_widths[2], 6, format_number(ante_value, digits=2), 1, 0, "R")
            pdf.cell(column_widths[3], 6, format_number(post_value, digits=2), 1, 0, "R")
            pdf.cell(column_widths[4], 6, format_number(delta, digits=2), 1, 1, "R")

            wrote_metrics = True

        if wrote_metrics and pdf.get_y() <= available_height:
            pdf.ln(1)

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