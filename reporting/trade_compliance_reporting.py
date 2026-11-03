"""Generate Excel and PDF outputs for trading compliance comparisons."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import logging

from fpdf import FPDF
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from reporting.report_utils import format_number
from reporting.trade_compliance_analyzer import ASSET_ORDER

logger = logging.getLogger(__name__)

ASSET_LABELS: Dict[str, str] = {"equity": "Equity", "options": "Options", "treasury": "Treasury"}

MONEY_FORMAT = "#,##0.00"
PERCENT_FORMAT = "0.00%"

# (header, record key, number format) for the Excel Trade Activity summary.
TRADE_SUMMARY_COLUMNS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("Fund", "fund", None),
    ("Asset Class", "label", None),
    ("Total Net Assets", "tna", MONEY_FORMAT),
    ("Total Assets", "ta", MONEY_FORMAT),
    ("Trade Value", "trade_value", MONEY_FORMAT),
    ("% of TNA", "pct_tna", PERCENT_FORMAT),
    ("% of Total Assets", "pct_assets", PERCENT_FORMAT),
    ("Ex-Ante Market Value", "ex_ante", MONEY_FORMAT),
    ("Ex-Post Market Value", "ex_post", MONEY_FORMAT),
    ("Market Value Delta", "delta", MONEY_FORMAT),
    ("Trade vs Ex-Ante %", "trade_vs_ante", PERCENT_FORMAT),
    ("Ex-Post vs Ex-Ante %", "post_vs_ante", PERCENT_FORMAT),
    ("Net Trade Value", "net_value", MONEY_FORMAT),
    ("Net Buy Qty", "buy_qty", MONEY_FORMAT),
    ("Net Sell Qty", "sell_qty", MONEY_FORMAT),
    ("Net Buy Value", "buy_val", MONEY_FORMAT),
    ("Net Sell Value", "sell_val", MONEY_FORMAT),
)

COMPLIANCE_CHANGE_HEADERS = ["Fund Name", "Status Change", "Violations Before", "Violations After", "Net Change"]
EXCEL_STATUS_FILLS = {"into": "CCFFCC", "out": "FFE0B3", "breach": "FF9999"}
PDF_STATUS_FILLS = {"into": (204, 255, 204), "out": (255, 204, 153), "breach": (255, 153, 153)}


@dataclass
class GeneratedTradingComplianceReport:
    """Excel/PDF artefacts for trading compliance."""
    excel_path: Optional[str]
    pdf_path: Optional[str]
    comparison_data: Optional[Mapping[str, Any]] = None


def _format_percent(value: float) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


def _format_money(value: float) -> str:
    return format_number(value, digits=2)


# (header, relative width, record key, formatter) for the PDF Trade Activity summary.
PDF_TRADE_SUMMARY_COLUMNS = (
    ("Fund", 1.1, "fund", None),
    ("Asset Class", 1.0, "label", None),
    ("Total Net Assets", 1.15, "tna", _format_money),
    ("Total Assets", 1.1, "ta", _format_money),
    ("Trade Value", 1.1, "trade_value", _format_money),
    ("% of TNA", 0.85, "pct_tna", _format_percent),
    ("% of Assets", 0.85, "pct_assets", _format_percent),
    ("Ex-Ante MV", 1.1, "ex_ante", _format_money),
    ("Ex-Post MV", 1.1, "ex_post", _format_money),
    ("Market Value Delta", 1.1, "delta", _format_money),
    ("Net Trade Value", 1.0, "net_value", _format_money),
    ("Net Buy Value", 1.0, "buy_val", _format_money),
    ("Net Sell Value", 1.0, "sell_val", _format_money),
)


def _num(mapping: Mapping[str, Any], key: str) -> float:
    return float(mapping.get(key, 0.0) or 0.0)


def _as_dict(source: Any, key: str) -> Dict[str, Any]:
    value = source.get(key) if isinstance(source, Mapping) else None
    return dict(value) if isinstance(value, Mapping) else {}


def _float_or(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_top_trades(trades_list: list, n: int = 5) -> list:
    """Get top N trades sorted by market value."""
    if not trades_list:
        return []
    return sorted(trades_list, key=lambda x: abs(float(x.get("market_value", 0))), reverse=True)[:n]


def _trade_activity_records(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One record per (fund, asset class), shared by the Excel and PDF trade activity tables."""
    records: List[Dict[str, Any]] = []
    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        trade_info = fund_data.get("trade_info", {}) or {}
        asset_summary = trade_info.get("asset_trade_summary", {}) or {}
        net_trades = trade_info.get("net_trades", {}) or {}
        activity_info = trade_info.get("trade_activity", {}) or {}
        tna = _num(trade_info, "total_net_assets")
        ta = _num(trade_info, "total_assets")

        for asset_key in ASSET_ORDER:
            summary = _as_dict(asset_summary, asset_key)
            activity = _as_dict(activity_info, asset_key)
            net = {**dict(activity.get("net", {}) or {}), **_as_dict(net_trades, asset_key)}
            buy_val, sell_val = _num(net, "buy_value"), _num(net, "sell_value")
            record: Dict[str, Any] = {
                "fund": fund_name,
                "label": ASSET_LABELS[asset_key],
                "tna": tna,
                "ta": ta,
                "trade_value": _num(summary, "trade_value") or abs(buy_val) + abs(sell_val),
                "pct_tna": _num(summary, "pct_of_tna"),
                "pct_assets": _num(summary, "pct_of_total_assets"),
                "ex_ante": _num(summary, "ex_ante_market_value"),
                "ex_post": _num(summary, "ex_post_market_value"),
                "delta": _num(summary, "market_value_delta"),
                "trade_vs_ante": _num(summary, "trade_vs_ex_ante_pct"),
                "post_vs_ante": _num(summary, "ex_post_vs_ex_ante_pct"),
                "net_value": _float_or(net.get("net_value"), buy_val - sell_val),
                "buy_qty": _num(net, "buy_quantity"),
                "sell_qty": _num(net, "sell_quantity"),
                "buy_val": buy_val,
                "sell_val": sell_val,
            }

            # Top 5 by market value for equity/options; all trades for treasury.
            details: List[Tuple[str, str, float, float]] = []
            for direction, side in (("Buy", "buys"), ("Sell", "sells")):
                trades = activity.get(side, []) or []
                if asset_key in ("equity", "options"):
                    trades = _get_top_trades(trades, n=5)
                for trade in trades:
                    quantity = float(trade.get("quantity", 0.0) or 0.0)
                    value = float(trade.get("market_value", 0.0) or 0.0)
                    if quantity == 0.0 and value == 0.0:
                        continue
                    details.append((direction, str(trade.get("ticker", "")), quantity, value))

            has_trade_lists = bool(activity.get("buys") or activity.get("sells"))
            summary_keys = ("trade_value", "delta", "net_value", "buy_qty", "sell_qty", "buy_val", "sell_val")
            detail_keys = ("buy_qty", "sell_qty", "buy_val", "sell_val")
            record["show_summary"] = has_trade_lists or any(abs(record[k]) > 0.0 for k in summary_keys)
            record["show_detail"] = has_trade_lists or not all(abs(record[k]) == 0.0 for k in detail_keys)
            record["details"] = details
            records.append(record)
    return records


def _compliance_change_rows(
    data: Mapping[str, Any],
) -> Tuple[List[Tuple[str, Any, int, int, int]], Tuple[int, int, int]]:
    rows = []
    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        before = int(fund_data.get("violations_before", 0) or 0)
        after = int(fund_data.get("violations_after", 0) or 0)
        rows.append((fund_name, fund_data.get("status_change", "UNCHANGED"), before, after, after - before))
    totals = (sum(r[2] for r in rows), sum(r[3] for r in rows), sum(r[4] for r in rows))
    return rows, totals


def _status_class(status: str, violations_after: int) -> Optional[str]:
    if status == "INTO_COMPLIANCE":
        return "into"
    if status == "OUT_OF_COMPLIANCE":
        return "out"
    if status == "UNCHANGED" and violations_after > 0:
        return "breach"
    return None


def _fill(color: str) -> PatternFill:
    return PatternFill(start_color=color, end_color=color, fill_type="solid")


def _write_header(sheet, row: int, headers: Sequence[str], *, center: bool = True) -> None:
    for col, header in enumerate(headers, start=1):
        cell = sheet.cell(row=row, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = _fill("DDDDDD")
        if center:
            cell.alignment = Alignment(horizontal="center")


def _min_widths(sheet, widths: Sequence[int]) -> None:
    for col, width in enumerate(widths, start=1):
        dimension = sheet.column_dimensions[get_column_letter(col)]
        if (dimension.width or 0) < width:
            dimension.width = width


def generate_trading_excel_report(comparison_data: Mapping[str, Any], output_path: str) -> None:
    """Create a multi-tab Excel workbook summarising trading compliance changes."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    workbook.remove(workbook.active)

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
    sheet["A4"] = "Compliance Status Changes"
    sheet["A4"].font = Font(size=12, bold=True)

    for row, (label, key) in enumerate(
        (("Total Funds Analyzed", "total_funds_analyzed"), ("Total Funds Traded", "total_funds_traded")), start=5
    ):
        sheet[f"A{row}"] = label
        sheet[f"A{row}"].font = Font(bold=True)
        sheet[f"B{row}"] = summary.get(key, 0)
        sheet[f"B{row}"].font = Font(bold=True)
        sheet[f"B{row}"].alignment = Alignment(horizontal="center")

    _write_header(sheet, 8, COMPLIANCE_CHANGE_HEADERS)
    row = 9
    rows, totals = _compliance_change_rows(data)
    for fund_name, status_change, before, after, net_change in rows:
        for col, value in enumerate((fund_name, status_change, before, after, net_change), start=1):
            cell = sheet.cell(row=row, column=col, value=value)
            if col >= 2:
                cell.alignment = Alignment(horizontal="center")

        status_class = _status_class((status_change or "UNCHANGED").upper(), after)
        if status_class:
            # Persistent breaches highlight the fund name only.
            fill_cols = (1,) if status_class == "breach" else range(1, len(COMPLIANCE_CHANGE_HEADERS) + 1)
            for col in fill_cols:
                sheet.cell(row=row, column=col).fill = _fill(EXCEL_STATUS_FILLS[status_class])
        row += 1

    for col, value in enumerate(("Totals", "", *totals), start=1):
        cell = sheet.cell(row=row, column=col, value=value)
        cell.font = Font(bold=True)
        cell.fill = _fill("DDDDDD")
        if col >= 2:
            cell.alignment = Alignment(horizontal="center")

    _min_widths(sheet, [28, 22, 22, 22, 22])


def _create_trade_activity_sheet(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Tab describing detailed trading activity by fund and asset class."""

    sheet = workbook.create_sheet("Trade Activity")
    sheet["A1"] = "Trade Activity Overview"
    sheet["A1"].font = Font(size=12, bold=True)
    sheet.freeze_panes = "A3"

    header_row = 3
    _write_header(sheet, header_row, [header for header, _, _ in TRADE_SUMMARY_COLUMNS])

    records = _trade_activity_records(data)
    summary_records = [record for record in records if record["show_summary"]]
    row = header_row + 1
    for record in summary_records:
        for col, (_, key, number_format) in enumerate(TRADE_SUMMARY_COLUMNS, start=1):
            cell = sheet.cell(row=row, column=col, value=record[key])
            if number_format:
                cell.number_format = number_format
        row += 1

    if not summary_records:
        sheet.cell(row=header_row + 1, column=1, value="No trading activity detected.")

    _min_widths(sheet, [24, 20, 22, 22, 22] + [18] * 12)

    # Trade Activity Details section
    row = max(row + 2, header_row + 3)
    sheet.cell(row=row, column=1, value="Trade Activity Details (Top 5 by Market Value)").font = Font(size=12, bold=True)
    _write_header(sheet, row + 1, ["Fund", "Asset Class", "Direction", "Ticker", "Quantity", "Market Value"])
    row += 2

    funds_written: set[str] = set()
    for record in records:
        if not record["show_detail"]:
            continue
        fund_name = record["fund"]
        for direction, ticker, quantity, value in record["details"]:
            values = (fund_name if fund_name not in funds_written else "", record["label"], direction, ticker)
            for col, cell_value in enumerate(values, start=1):
                sheet.cell(row=row, column=col, value=cell_value)
            sheet.cell(row=row, column=5, value=quantity).number_format = MONEY_FORMAT
            sheet.cell(row=row, column=6, value=value).number_format = MONEY_FORMAT
            funds_written.add(fund_name)
            row += 1

        # A blank row follows every shown section once the fund has written a trade.
        if fund_name in funds_written:
            row += 1

    if not funds_written:
        sheet.cell(row=row, column=1, value="No trade details available.")


def _create_compliance_details_sheet(workbook: Workbook, data: Mapping[str, Any]) -> None:
    """Create the compliance details sheet (mirrors detailed comparison)."""

    sheet = workbook.create_sheet("Compliance Details")
    headers = [
        "Fund Name", "Compliance Check", "Status Before", "Status After",
        "Violations Before", "Violations After", "Changed",
    ]
    _write_header(sheet, 1, headers, center=False)

    row = 2
    for fund_name, fund_data in sorted(data.get("funds", {}).items()):
        for check_name, check_data in sorted(fund_data.get("checks", {}).items()):
            violations_after = int(check_data.get("violations_after", 0) or 0)
            changed = bool(check_data.get("changed"))
            values = (
                fund_name,
                check_name,
                check_data.get("status_before", "UNKNOWN"),
                check_data.get("status_after", "UNKNOWN"),
                check_data.get("violations_before", 0),
                violations_after,
                "YES" if changed else "NO",
            )
            fill_color = "FF9999" if violations_after > 0 else ("FFFFCC" if changed else None)
            for col, value in enumerate(values, start=1):
                cell = sheet.cell(row=row, column=col, value=value)
                if fill_color:
                    cell.fill = _fill(fill_color)
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


def _pdf_header_row(pdf: FPDF, widths: Sequence[float], headers: Sequence[str], height: float, size: int) -> None:
    pdf.set_font("Arial", "B", size)
    pdf.set_fill_color(200, 200, 200)
    for width, header in zip(widths, headers):
        pdf.cell(width, height, header, 1, 0, "C", True)
    pdf.ln(height)


def _pdf_row(
    pdf: FPDF, widths: Sequence[float], texts: Sequence[str], height: float, aligns: str, fill: bool = False
) -> None:
    """Bordered cells across one row, moving to the next line after the last cell."""
    last = len(widths) - 1
    for idx, (width, text) in enumerate(zip(widths, texts)):
        pdf.cell(width, height, text, 1, 1 if idx == last else 0, aligns[idx], fill)


def generate_trading_pdf_report(comparison_data: Mapping[str, Any], output_path: str) -> None:
    """Create a PDF report for the trading compliance comparison."""

    try:
        pdf = TradingCompliancePDF(date=str(comparison_data.get("date", "")))
        pdf.add_page(orientation="L")
        _add_compliance_overview(pdf, comparison_data)

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
    pdf.cell(0, 6, f"Total Funds Analyzed: {summary.get('total_funds_analyzed', 0)}", 0, 1, "L")
    pdf.cell(0, 6, f"Total Funds Traded: {summary.get('total_funds_traded', 0)}", 0, 1, "L")
    pdf.ln(2)

    _render_compliance_changes_table(pdf, data)
    pdf.ln(3)


def _render_compliance_changes_table(pdf: FPDF, data: Mapping[str, Any]) -> None:
    # Relies on fpdf's auto page break (no header redraw).
    col_widths = [55, 40, 25, 25, 25]
    headers = ["Fund", "Status Change", "Viol. Before", "Viol. After", "Net Change"]
    _pdf_header_row(pdf, col_widths, headers, 6, 8)
    pdf.set_font("Arial", "", 8)

    rows, totals = _compliance_change_rows(data)
    for fund_name, status_change, before, after, net_change in rows:
        display_name = fund_name if len(fund_name) <= 26 else f"{fund_name[:23]}..."
        status_change = str(status_change)
        status_class = _status_class(status_change.upper(), after)
        if status_class:
            pdf.set_fill_color(*PDF_STATUS_FILLS[status_class])
        texts = (display_name, status_change, str(before), str(after), str(net_change))
        _pdf_row(pdf, col_widths, texts, 6, "LCCCC", status_class is not None)

    pdf.set_font("Arial", "B", 8)
    pdf.set_fill_color(220, 220, 220)
    _pdf_row(pdf, col_widths, ("Totals", "", *(str(total) for total in totals)), 6, "LCCCC", True)
    pdf.set_font("Arial", "", 8)


def _add_trade_activity(pdf: FPDF, data: Mapping[str, Any]) -> None:
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Trade Activity", 0, 1, "L")
    pdf.ln(3)

    summary_rows = [record for record in _trade_activity_records(data) if record["show_summary"]]
    by_fund: Dict[str, List[Tuple[str, str, str, float, float]]] = defaultdict(list)
    for record in summary_rows:
        for detail in record["details"]:
            by_fund[record["fund"]].append((record["label"], *detail))

    if not summary_rows and not by_fund:
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 6, "No trading activity detected.", 0, 1, "L")
        return

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    summary_headers = [header for header, _, _, _ in PDF_TRADE_SUMMARY_COLUMNS]
    summary_rel_widths = [rel_width for _, rel_width, _, _ in PDF_TRADE_SUMMARY_COLUMNS]
    total_weight = sum(summary_rel_widths)
    summary_widths = [usable_width * w / total_weight for w in summary_rel_widths]

    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 6, "Trade Activity Summary", 0, 1, "L")
    pdf.ln(1)

    _pdf_header_row(pdf, summary_widths, summary_headers, 5.5, 6)
    pdf.set_font("Arial", "", 6)
    page_limit = pdf.h - pdf.b_margin - 10
    row_height = 5.0
    last_fund = None

    for row in sorted(summary_rows, key=lambda item: (item["fund"], item["label"])):
        if pdf.get_y() + row_height > page_limit:
            pdf.add_page(orientation=getattr(pdf, "cur_orientation", "L"))
            _pdf_header_row(pdf, summary_widths, summary_headers, 5.5, 6)
            pdf.set_font("Arial", "", 6)

        fund_label = row["fund"] if row["fund"] != last_fund else ""
        if fund_label:
            pdf.set_font("Arial", "B", 6)
        pdf.cell(summary_widths[0], row_height, fund_label, 1, 0, "L")
        if fund_label:
            pdf.set_font("Arial", "", 6)
        pdf.cell(summary_widths[1], row_height, row["label"], 1, 0, "L")
        values = [formatter(row[key]) for _, _, key, formatter in PDF_TRADE_SUMMARY_COLUMNS[2:]]
        _pdf_row(pdf, summary_widths[2:], values, row_height, "R" * len(values))
        last_fund = row["fund"]

    pdf.ln(3)

    if by_fund:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 6, "Trade Activity Details", 0, 1, "L")
        pdf.ln(1)

        # 5-col mini-table (no Fund column; fund is the header bar above it).
        sub_headers = ["Asset Class", "Direction", "Ticker", "Quantity", "Market Value"]
        sub_rel_widths = [1.1, 0.9, 1.1, 1.0, 1.2]
        sub_total = sum(sub_rel_widths)
        sub_widths = [usable_width * w / sub_total for w in sub_rel_widths]
        header_height = 5.5
        fund_bar_height = 6.0

        def _fund_bar(text: str) -> None:
            pdf.set_font("Arial", "B", 8)
            pdf.set_fill_color(220, 230, 241)  # light blue
            pdf.cell(usable_width, fund_bar_height, text, 1, 1, "L", True)
            _pdf_header_row(pdf, sub_widths, sub_headers, header_height, 6)
            pdf.set_font("Arial", "", 6)

        for fund_name in sorted(by_fund.keys()):
            rows = by_fund[fund_name]

            # If the fund's whole block won't fit on the current page, page-break.
            block_needed = fund_bar_height + header_height + row_height * len(rows) + 2
            if pdf.get_y() + block_needed > page_limit:
                pdf.add_page(orientation=getattr(pdf, "cur_orientation", "L"))
            _fund_bar(fund_name)

            for asset, direction, ticker, quantity, value in rows:
                # Mid-fund page break: redraw the fund bar + header on the new page.
                if pdf.get_y() + row_height > page_limit:
                    pdf.add_page(orientation=getattr(pdf, "cur_orientation", "L"))
                    _fund_bar(f"{fund_name} (continued)")
                texts = (asset, direction, ticker, _format_money(quantity), _format_money(value))
                _pdf_row(pdf, sub_widths, texts, row_height, "LLLRR")

            pdf.ln(2)


def _comparison_cell(check_data: Mapping[str, Any]) -> Tuple[str, str, str]:
    if not check_data:
        return "", "", ""
    changed = "YES" if bool(check_data.get("changed")) else "NO"
    return str(check_data.get("status_before", "")), str(check_data.get("status_after", "")), changed


def _add_detailed_comparison(pdf: FPDF, data: Mapping[str, Any]) -> None:
    funds = []
    for fund_name, fund_payload in sorted(data.get("funds", {}).items()):
        checks = fund_payload.get("checks", {}) or {}
        if checks:
            funds.append((fund_name, checks))

    table_rows: list[tuple[str, list[tuple[str, str, str]]]] = []
    for check_name in sorted({check_name for _, checks in funds for check_name in checks}):
        values = [_comparison_cell(checks.get(check_name) or {}) for _, checks in funds]
        if any(changed for _, _, changed in values):
            table_rows.append((check_name, values))

    if not table_rows:
        return

    header_height = 6
    row_height = 6
    page_limit = pdf.h - pdf.b_margin - 5

    def _render_header(title: str, fund_subset: list[tuple[str, Any]], per_value_width: float, metric_width: float) -> None:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, 0, 1, "L")
        pdf.ln(1)

        start_x = pdf.l_margin
        start_y = pdf.get_y()

        pdf.set_font("Arial", "B", 7)
        pdf.set_fill_color(200, 200, 200)

        metric_height = header_height * 2
        pdf.cell(metric_width, metric_height, "Compliance Check", 1, 0, "C", True)
        for fund_name, _ in fund_subset:
            display = fund_name if len(fund_name) <= 24 else f"{fund_name[:21]}..."
            pdf.cell(per_value_width * 3, header_height, display, 1, 0, "C", True)

        pdf.ln(header_height)
        pdf.set_xy(start_x + metric_width, start_y + header_height)
        for _ in fund_subset:
            for sub in ("Before", "After", "Changed"):
                pdf.cell(per_value_width, header_height, sub, 1, 0, "C", True)

        pdf.ln(header_height)
        pdf.set_y(start_y + metric_height)
        pdf.set_font("Arial", "", 7)

    def _status_cell(width: float, text: str, highlight: Optional[Tuple[int, int, int]]) -> None:
        if highlight:
            pdf.set_fill_color(*highlight)
            pdf.cell(width, row_height, text, 1, 0, "C", True)
            pdf.set_fill_color(255, 255, 255)
        else:
            pdf.cell(width, row_height, text, 1, 0, "C")

    for chunk_start in range(0, len(funds), 5):
        fund_subset = funds[chunk_start:chunk_start + 5]
        usable_width = pdf.w - pdf.l_margin - pdf.r_margin
        metric_width = min(70.0, max(45.0, usable_width * 0.25))
        remaining_width = max(usable_width - metric_width, 60.0)
        per_value_width = remaining_width / (len(fund_subset) * 3)

        title = "Detailed Comparison by Compliance Check"
        if chunk_start:
            title = f"{title} (Group {chunk_start // 5 + 1})"

        _render_header(title, fund_subset, per_value_width, metric_width)

        for check_name, values in table_rows:
            if pdf.get_y() + row_height > page_limit:
                pdf.add_page(orientation=getattr(pdf, "cur_orientation", "L"))
                _render_header(f"{title} (cont.)", fund_subset, per_value_width, metric_width)

            pdf.set_x(pdf.l_margin)
            display_check = check_name if len(check_name) <= 40 else f"{check_name[:37]}..."
            pdf.set_font("Arial", "B", 7)
            pdf.cell(metric_width, row_height, display_check, 1, 0, "L")
            pdf.set_font("Arial", "", 7)

            for before_status, after_status, changed_flag in values[chunk_start:chunk_start + len(fund_subset)]:
                pdf.cell(per_value_width, row_height, before_status, 1, 0, "C")
                _status_cell(per_value_width, after_status, (255, 204, 204) if after_status.upper() == "FAIL" else None)
                _status_cell(per_value_width, changed_flag, (255, 255, 204) if changed_flag.upper() == "YES" else None)

            pdf.ln(row_height)

        pdf.ln(3)


__all__ = [
    "GeneratedTradingComplianceReport",
    "generate_trading_excel_report",
    "generate_trading_pdf_report",
]
