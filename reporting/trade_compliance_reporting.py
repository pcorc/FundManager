"""Generate Excel and PDF outputs for trading compliance comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from reporting.base_report_pdf import BaseReportPDF
from reporting.report_utils import format_number, normalize_report_date


@dataclass
class GeneratedTradingComplianceReport:
    """Excel/PDF artefacts for trading compliance."""

    excel_path: Optional[str]
    pdf_path: Optional[str]


class TradingComplianceExcelReport:
    """Export trading compliance comparisons to Excel."""

    def __init__(self, comparison_data: Mapping[str, Any], output_path: Path) -> None:
        self.data = comparison_data
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._export()

    def _export(self) -> None:
        with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
            self._write_summary_sheet(writer)
            self._write_fund_sheet(writer)
            self._write_check_sheet(writer)

    def _write_summary_sheet(self, writer: pd.ExcelWriter) -> None:
        summary = self.data.get("summary", {})
        rows = [(self._format_metric_name(key), value) for key, value in summary.items()]
        df = pd.DataFrame(rows, columns=["Metric", "Value"])
        df.to_excel(writer, sheet_name="Summary", index=False)

    def _write_fund_sheet(self, writer: pd.ExcelWriter) -> None:
        funds = self.data.get("funds", {})
        rows: list[Dict[str, Any]] = []
        for fund_name, info in funds.items():
            trade_info = info.get("trade_info", {})
            rows.append(
                {
                    "Fund": fund_name,
                    "Total Traded": trade_info.get("total_traded", 0.0),
                    "Equity": trade_info.get("equity", 0.0),
                    "Options": trade_info.get("options", 0.0),
                    "Treasury": trade_info.get("treasury", 0.0),
                    "Status Before": info.get("overall_before"),
                    "Status After": info.get("overall_after"),
                    "Status Change": info.get("status_change"),
                    "Violations Before": info.get("violations_before"),
                    "Violations After": info.get("violations_after"),
                    "Checks Changed": info.get("num_changes"),
                }
            )
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Funds", index=False)

    def _write_check_sheet(self, writer: pd.ExcelWriter) -> None:
        rows: list[Dict[str, Any]] = []
        for fund_name, info in self.data.get("funds", {}).items():
            for check_name, check in info.get("checks", {}).items():
                rows.append(
                    {
                        "Fund": fund_name,
                        "Check": check_name,
                        "Status Before": check.get("status_before"),
                        "Status After": check.get("status_after"),
                        "Changed": check.get("changed"),
                    }
                )
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Check Changes", index=False)

    @staticmethod
    def _format_metric_name(name: str) -> str:
        return name.replace("_", " ").title()


class TradingCompliancePDF(BaseReportPDF):
    """PDF rendering for trading compliance comparisons."""

    def __init__(self, output_path: str, comparison_data: Mapping[str, Any]) -> None:
        super().__init__(output_path)
        self.data = comparison_data

    def render(self) -> None:
        date_str = self.data.get("date", "")
        subtitle = f"Ex-Ante vs Ex-Post Comparison - {date_str}" if date_str else None
        self.add_title("Trading Compliance Analysis", subtitle)

        summary_rows = [
            (self._format_metric_name(key), value)
            for key, value in self.data.get("summary", {}).items()
        ]
        if summary_rows:
            self.add_section_heading("Executive Summary")
            self.add_key_value_table(summary_rows, header=("Metric", "Value"))

        for fund_name, info in sorted(self.data.get("funds", {}).items()):
            self.add_section_heading(fund_name)
            trade_info = info.get("trade_info", {})
            trade_rows = [
                ("Total Traded", format_number(trade_info.get("total_traded", 0.0), 2)),
                ("Equity", format_number(trade_info.get("equity", 0.0), 2)),
                ("Options", format_number(trade_info.get("options", 0.0), 2)),
                ("Treasury", format_number(trade_info.get("treasury", 0.0), 2)),
                ("Status Before", info.get("overall_before")),
                ("Status After", info.get("overall_after")),
                ("Status Change", info.get("status_change")),
                ("Violations Before", info.get("violations_before")),
                ("Violations After", info.get("violations_after")),
                ("Checks Changed", info.get("num_changes")),
            ]
            self.add_key_value_table(trade_rows, header=("Metric", "Value"))

            check_rows = [
                (
                    check_name,
                    check.get("status_before"),
                    check.get("status_after"),
                    "Yes" if check.get("changed") else "No",
                )
                for check_name, check in sorted(info.get("checks", {}).items())
            ]
            if check_rows:
                self.add_table(
                    ["Check", "Before", "After", "Changed"],
                    check_rows,
                    column_widths=[70, 35, 35, 35],
                    alignments=["L", "C", "C", "C"],
                )

        self.output()

    @staticmethod
    def _format_metric_name(name: str) -> str:
        return name.replace("_", " ").title()


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

    TradingComplianceExcelReport(comparison_data, excel_path)

    pdf_result: Optional[str] = None
    if create_pdf:
        pdf = TradingCompliancePDF(str(pdf_path), comparison_data)
        pdf.render()
        pdf_result = str(pdf_path)

    return GeneratedTradingComplianceReport(str(excel_path), pdf_result)