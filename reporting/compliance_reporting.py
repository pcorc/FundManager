"""Generate Excel and PDF summaries for compliance results."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd

from reporting.base_report_pdf import BaseReportPDF
from reporting.footnotes import FOOTNOTES
from reporting.report_utils import (
    flatten_compliance_results,
    format_number,
    normalize_compliance_results,
)
from utilities.logger import setup_logger

logger = setup_logger("compliance_report", "logs/compliance.log")


@dataclass
class GeneratedComplianceReport:
    excel_path: Optional[str]
    pdf_path: Optional[str]


class ComplianceReport:
    """Prepare an Excel workbook summarising compliance results."""

    def __init__(
        self,
        results: Mapping[str, Mapping[str, object]],
        report_date: date | datetime | str,
        output_dir: str,
        *,
        file_name_prefix: str = "compliance_results",
        test_functions: Optional[Iterable[str]] = None,
        gics_mapping: Optional[pd.DataFrame] = None,
    ) -> None:
        self.report_date = self._normalise_date(report_date)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.output_dir / f"{file_name_prefix}_{self.report_date}.xlsx"

        self.gics_mapping = gics_mapping
        self.test_functions = set(test_functions) if test_functions else None
        self.sheet_data: Dict[str, pd.DataFrame] = {}

        self.results = {self.report_date: normalize_compliance_results(results)}

        self.process_all_checks()
        self.export_to_excel()

    # ------------------------------------------------------------------
    def process_all_checks(self) -> None:
        test_map = {
            "summary": self.process_summary,
            "gics_compliance": self.process_gics_compliance,
            "prospectus_80pct_policy": self.process_prospectus_80pct_policy,
            "diversification_40act_check": self.process_40_act_diversification,
            "diversification_IRS_check": self.process_irs_diversification,
            "diversification_IRC_check": self.process_irc_diversification,
            "max_15pct_illiquid_sai": self.process_max_15pct_illiquid,
            "real_estate_check": self.process_real_estate_check,
            "commodities_check": self.process_commodities_check,
            "twelve_d1a_other_inv_cos": self.process_12d1_other_inv_cos,
            "twelve_d2_insurance_cos": self.process_12d2_insurance_cos,
            "twelve_d3_sec_biz": self.process_12d3_sec_biz,
        }

        for test_name, processor in test_map.items():
            if self.test_functions and test_name not in self.test_functions:
                continue
            processor()

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_date(value: date | datetime | str) -> str:
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

    # ------------------------------------------------------------------
    def process_summary(self) -> None:
        rows = []
        for date_str, funds in self.results.items():
            for fund_name, data in funds.items():
                summary = data.get("summary_metrics", {})
                if not summary:
                    continue
                rows.append(
                    {
                        "Date": date_str,
                        "Fund": fund_name,
                        "Cash": summary.get("cash_value", 0.0),
                        "Treasury": summary.get("treasury", 0.0),
                        "Equity": summary.get("equity_market_value", 0.0),
                        "Option DAN": summary.get("option_delta_adjusted_notional", 0.0),
                        "Option MV": summary.get("option_market_value", 0.0),
                        "Total Assets": summary.get("total_assets", 0.0),
                        "Total Net Assets": summary.get("total_net_assets", 0.0),
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            self.sheet_data["Summary"] = df

    # ------------------------------------------------------------------
    def process_gics_compliance(self) -> None:
        summary_rows = []
        detail_rows = []

        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                gics = fund_data.get("gics_compliance")
                if not gics:
                    continue

                calculations = gics.get("calculations", {})
                exposures = calculations.get("sector_exposure", {})
                breaches = gics.get("breaches", {})
                limits = gics.get("limits", {})

                summary_rows.append(
                    {
                        "Date": report_date,
                        "Fund": fund_name,
                        "Overall Compliance": "PASS" if gics.get("is_compliant") else "FAIL",
                        "Breached Sectors": ", ".join(
                            f"{sector} ({exposures.get(sector, 0):.2%})" for sector in breaches
                        )
                        or "None",
                        "Max Sector Exposure": max(exposures.values()) if exposures else 0.0,
                        "Weight Source": gics.get("weight_source", ""),
                    }
                )

                for sector, weight in exposures.items():
                    detail_rows.append(
                        {
                            "Date": report_date,
                            "Fund": fund_name,
                            "Sector": sector,
                            "Weight": weight,
                            "Limit": limits.get(sector, limits.get("_default")),
                            "Breached": sector in breaches,
                        }
                    )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.sort_values(["Fund", "Date"], inplace=True)
            fn_df = pd.DataFrame({"Note": [f"* {note}" for note in FOOTNOTES.get("gics", [])]})
            self.sheet_data["GICS_Compliance"] = pd.concat([summary_df, fn_df], ignore_index=True)

        if detail_rows:
            details_df = pd.DataFrame(detail_rows)
            details_df.sort_values(["Fund", "Weight"], ascending=[True, False], inplace=True)
            self.sheet_data["GICS_Details"] = details_df

    # ------------------------------------------------------------------
    def process_prospectus_80pct_policy(self) -> None:
        rows = []
        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                prospectus = fund_data.get("prospectus_80pct_policy")
                if not prospectus:
                    continue
                calculations = prospectus.get("calculations", {})
                threshold = calculations.get("threshold", 0.8)
                names_test = calculations.get("names_test") or 0.0
                names_test_mv = calculations.get("names_test_mv") or 0.0

                rows.append(
                    {
                        "Date": report_date,
                        "Fund": fund_name,
                        "Compliance (DAN)": "PASS" if names_test >= threshold else "FAIL",
                        "Names Test (DAN)": names_test,
                        "Names Test (Market Value)": names_test_mv,
                        "Threshold": threshold,
                        "Options In Scope": "Yes" if prospectus.get("options_in_scope") else "No",
                        "Total Equity MV": calculations.get("total_equity_market_value", 0.0),
                        "Total Option DAN": abs(calculations.get("total_opt_delta_notional_value", 0.0)),
                        "Total Option MV": calculations.get("total_opt_market_value", 0.0),
                        "Total T-Bill Value": calculations.get("total_tbill_value", 0.0),
                        "Total Cash Value": calculations.get("total_cash_value", 0.0),
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            fn_df = pd.DataFrame({"Note": [f"* {note}" for note in FOOTNOTES.get("prospectus_80pct", [])]})
            self.sheet_data["Prospectus_80pct"] = pd.concat([df, fn_df], ignore_index=True)

    # ------------------------------------------------------------------
    def process_40_act_diversification(self) -> None:
        rows = []
        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                forty_act = fund_data.get("diversification_40act_check")
                if not forty_act:
                    continue
                calc = forty_act.get("calculations", {})
                rows.append(
                    {
                        "Date": report_date,
                        "Fund": fund_name,
                        "Condition 1": "PASS" if forty_act.get("condition_40act_1") else "FAIL",
                        "Condition 2a": "PASS" if forty_act.get("condition_40act_2a") else "FAIL",
                        "Condition 2b": "PASS" if forty_act.get("condition_40act_2b") else "FAIL",
                        "Condition 2a OCC": "PASS" if forty_act.get("condition_40act_2a_occ") else "FAIL",
                        "Fund Registration": calc.get("fund_registration"),
                        "Total Assets": calc.get("total_assets", 0.0),
                        "Net Assets": calc.get("net_assets", 0.0),
                        "Issuer Limited Sum": calc.get("issuer_limited_sum", 0.0),
                        "OCC Market Value": calc.get("occ_market_value", 0.0),
                        "OCC Weight": calc.get("occ_weight", 0.0),
                        "Cumulative Weight Excluded": calc.get("cumulative_weight_excluded", 0.0),
                        "Cumulative Weight Remaining": calc.get("cumulative_weight_remaining", 0.0),
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            fn_df = pd.DataFrame({"Note": [f"* {note}" for note in FOOTNOTES.get("40act", [])]})
            self.sheet_data["40Act_Diversification"] = pd.concat([df, fn_df], ignore_index=True)

    # ------------------------------------------------------------------
    def process_irs_diversification(self) -> None:
        rows = []
        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                irs = fund_data.get("diversification_IRS_check")
                if not irs:
                    continue
                calc = irs.get("calculations", {})
                rows.append(
                    {
                        "Date": report_date,
                        "Fund": fund_name,
                        "Condition 1": "PASS" if irs.get("condition_IRS_1") else "FAIL",
                        "Condition 2a 50%": "PASS" if irs.get("condition_IRS_2_a_50") else "FAIL",
                        "Condition 2a 5%": "PASS" if irs.get("condition_IRS_2_a_5") else "FAIL",
                        "Condition 2a 10%": "PASS" if irs.get("condition_IRS_2_a_10") else "FAIL",
                        "Total Assets": calc.get("total_assets", 0.0),
                        "Qualifying Assets Value": calc.get("qualifying_assets_value", 0.0),
                        "Five % Gross Assets": calc.get("five_pct_gross_assets", 0.0),
                        "Sum Large Securities Wt": calc.get("sum_large_securities_weights", 0.0),
                        "Large Securities Count": calc.get("large_securities_count", 0),
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            fn_df = pd.DataFrame({"Note": [f"* {note}" for note in FOOTNOTES.get("irs", [])]})
            self.sheet_data["IRS_Diversification"] = pd.concat([df, fn_df], ignore_index=True)

    # ------------------------------------------------------------------
    def process_irc_diversification(self) -> None:
        rows = []
        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                irc = fund_data.get("diversification_IRC_check")
                if not irc:
                    continue
                calc = irc.get("calculations", {})
                rows.append(
                    {
                        "Date": report_date,
                        "Fund": fund_name,
                        "Condition 55": "PASS" if irc.get("condition_IRC_55") else "FAIL",
                        "Condition 70": "PASS" if irc.get("condition_IRC_70") else "FAIL",
                        "Condition 80": "PASS" if irc.get("condition_IRC_80") else "FAIL",
                        "Condition 90": "PASS" if irc.get("condition_IRC_90") else "FAIL",
                        "Top 1 Exposure": calc.get("top_1", 0.0),
                        "Top 2 Exposure": calc.get("top_2", 0.0),
                        "Top 3 Exposure": calc.get("top_3", 0.0),
                        "Top 4 Exposure": calc.get("top_4", 0.0),
                        "Total Assets": calc.get("total_assets", 0.0),
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            fn_df = pd.DataFrame({"Note": [f"* {note}" for note in FOOTNOTES.get("irc", [])]})
            self.sheet_data["IRC_Diversification"] = pd.concat([df, fn_df], ignore_index=True)

    # ------------------------------------------------------------------
    def process_real_estate_check(self) -> None:
        rows = []
        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                real_estate = fund_data.get("real_estate_check")
                if not real_estate:
                    continue
                calc = real_estate.get("calculations", {})
                rows.append(
                    {
                        "Date": report_date,
                        "Fund": fund_name,
                        "Real Estate Exposure": calc.get("real_estate_percentage", 0.0),
                        "Total Exposure": calc.get("total_exposure", 0.0),
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            fn_df = pd.DataFrame({"Note": [f"* {note}" for note in FOOTNOTES.get("real_estate", [])]})
            self.sheet_data["Real_Estate"] = pd.concat([df, fn_df], ignore_index=True)

    # ------------------------------------------------------------------
    def process_commodities_check(self) -> None:
        rows = []
        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                commodities = fund_data.get("commodities_check")
                if not commodities:
                    continue
                calc = commodities.get("calculations", {})
                exposure = calc.get("commodities_percentage")
                if exposure is None:
                    exposure = calc.get("commodities_exposure", 0.0)
                rows.append(
                    {
                        "Date": report_date,
                        "Fund": fund_name,
                        "Commodities Exposure": exposure,
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            fn_df = pd.DataFrame({"Note": [f"* {note}" for note in FOOTNOTES.get("commodities", [])]})
            self.sheet_data["Commodities"] = pd.concat([df, fn_df], ignore_index=True)

    # ------------------------------------------------------------------
    def process_12d1_other_inv_cos(self) -> None:
        rows = []
        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                rule = fund_data.get("twelve_d1a_other_inv_cos")
                if not rule:
                    continue
                calc = rule.get("calculations", {})
                holdings = calc.get("investment_companies", [])
                holdings_str = ", ".join(
                    f"{h.get('equity_ticker') or h.get('ticker')} ({(h.get('ownership_pct') or 0)*100:.2f}%)" for h in holdings
                ) or "None"
                rows.append(
                    {
                        "Date": report_date,
                        "Fund": fund_name,
                        "Test 1 (<=3%)": "PASS" if rule.get("test_1_pass") else "FAIL",
                        "Test 2 (<=5% assets)": "PASS" if rule.get("test_2_pass") else "FAIL",
                        "Test 3 (<=10% assets)": "PASS" if rule.get("test_3_pass") else "FAIL",
                        "Compliant": "PASS" if rule.get("twelve_d1a_other_inv_cos_compliant") else "FAIL",
                        "Total Assets": calc.get("total_assets", 0.0),
                        "Equity MV Sum": calc.get("equity_market_value_sum", 0.0),
                        "Ownership % Max": calc.get("ownership_pct_max", 0.0),
                        "Investment Companies": holdings_str,
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            fn_df = pd.DataFrame({"Note": [f"* {note}" for note in FOOTNOTES.get("12d1", [])]})
            self.sheet_data["12d1_Other_Investment_Companies"] = pd.concat([df, fn_df], ignore_index=True)

    # ------------------------------------------------------------------
    def process_12d2_insurance_cos(self) -> None:
        rows = []
        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                rule = fund_data.get("twelve_d2_insurance_cos")
                if not rule:
                    continue
                calc = rule.get("calculations", {})
                holdings = calc.get("insurance_holdings", [])
                holdings_str = ", ".join(
                    f"{h.get('equity_ticker') or h.get('ticker')} ({(h.get('ownership_pct') or 0)*100:.5f}%)" for h in holdings
                ) or "None"
                rows.append(
                    {
                        "Date": report_date,
                        "Fund": fund_name,
                        "Compliant": "PASS" if rule.get("test_pass") else "FAIL",
                        "Total Assets": calc.get("total_assets", 0.0),
                        "Insurance Holdings": holdings_str,
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            fn_df = pd.DataFrame({"Note": [f"* {note}" for note in FOOTNOTES.get("12d2", [])]})
            self.sheet_data["12d2_Insurance_Companies"] = pd.concat([df, fn_df], ignore_index=True)

    # ------------------------------------------------------------------
    def process_12d3_sec_biz(self) -> None:
        rows = []
        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                rule = fund_data.get("twelve_d3_sec_biz")
                if not rule:
                    continue
                calc = rule.get("calculations", {})
                combined = calc.get("combined_holdings", [])
                combined_str = ", ".join(
                    f"{h.get('equity_ticker') or h.get('ticker')} ({(h.get('vest_weight') or 0)*100:.2f}%)"
                    for h in combined
                ) or "None"
                rows.append(
                    {
                        "Date": report_date,
                        "Fund": fund_name,
                        "Rule 1 (<=5% equities)": "PASS" if rule.get("rule_1_pass") else "FAIL",
                        "Rule 2 (<=10% debt)": "PASS" if rule.get("rule_2_pass") else "FAIL",
                        "Rule 3 (<=5% assets)": "PASS" if rule.get("rule_3_pass") else "FAIL",
                        "Compliant": "PASS" if rule.get("twelve_d3_sec_biz_compliant") else "FAIL",
                        "Total Assets": calc.get("total_assets", 0.0),
                        "OCC Market Value": calc.get("occ_weight_mkt_val", 0.0),
                        "OCC Weight": calc.get("occ_weight", 0.0),
                        "Combined Holdings": combined_str,
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            fn_df = pd.DataFrame({"Note": [f"* {note}" for note in FOOTNOTES.get("12d3", [])]})
            self.sheet_data["12d3_Securities_Business"] = pd.concat([df, fn_df], ignore_index=True)

    # ------------------------------------------------------------------
    def process_max_15pct_illiquid(self) -> None:
        rows = []
        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                illiquid = fund_data.get("max_15pct_illiquid_sai")
                if not illiquid:
                    continue
                calc = illiquid.get("calculations", {})
                rows.append(
                    {
                        "Date": report_date,
                        "Fund": fund_name,
                        "Illiquid Compliance": "PASS" if illiquid.get("max_15pct_illiquid_sai") else "FAIL",
                        "Equity 85% Compliance": "PASS" if illiquid.get("equity_holdings_85pct_compliant") else "FAIL",
                        "Total Assets": calc.get("total_assets", 0.0),
                        "Total Illiquid Value": calc.get("total_illiquid_value", 0.0),
                        "Illiquid Percentage": calc.get("illiquid_percentage", 0.0),
                        "Equity Holdings Percentage": calc.get("equity_holdings_percentage", 0.0),
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            fn_df = pd.DataFrame({"Note": [f"* {note}" for note in FOOTNOTES.get("illiquid", [])]})
            self.sheet_data["Illiquid"] = pd.concat([df, fn_df], ignore_index=True)

    # ------------------------------------------------------------------
    def export_to_excel(self) -> None:
        if not self.sheet_data:
            return

        with pd.ExcelWriter(self.file_path, engine="openpyxl") as writer:
            for sheet_name, df in self.sheet_data.items():
                output_df = df.copy()
                if not output_df.empty:
                    for column in output_df.columns:
                        if output_df[column].dtype.kind in {"f", "i"}:
                            output_df[column] = output_df[column].apply(lambda x: format_number(x, 2))
                safe_sheet = sheet_name[:31]
                output_df.to_excel(writer, sheet_name=safe_sheet, index=False)

            workbook = writer.book
            for sheet in workbook.sheetnames:
                ws = workbook[sheet]
                for column_cells in ws.columns:
                    max_length = 0
                    column_letter = column_cells[0].column_letter
                    for cell in column_cells:
                        try:
                            if cell.value is not None:
                                max_length = max(max_length, len(str(cell.value)))
                        except Exception:  # pragma: no cover - defensive
                            continue
                    ws.column_dimensions[column_letter].width = min(max_length + 2, 50)


class ComplianceReportPDF(BaseReportPDF):
    """Render compliance results to a PDF document."""

    def __init__(self, results: Mapping[str, Mapping[str, object]], output_path: str) -> None:
        self.results = flatten_compliance_results(results)
        super().__init__(output_path)
        self.generate_pdf()

    # ------------------------------------------------------------------
    def generate_pdf(self) -> None:
        self._add_header("Compliance Report")
        for (fund_name, date_str), fund_data in sorted(self.results.items(), key=lambda item: (item[0][1], item[0][0])):
            report_date = self._parse_date(date_str)
            self._add_fund_section(fund_name, report_date, fund_data)
        self.output()

    @staticmethod
    def _parse_date(value: str) -> str:
        try:
            return datetime.fromisoformat(value).strftime("%Y-%m-%d")
        except Exception:  # pragma: no cover - fallback
            return value

    # ------------------------------------------------------------------
    def _add_fund_section(self, fund_name: str, report_date: str, fund_data: Mapping[str, object]) -> None:
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, self._sanitize_text(f"Fund: {fund_name} | Date: {report_date}"), ln=True)
        self.pdf.ln(2)

        self._print_summary_table(fund_data)

        sections = [
            ("Prospectus 80% Policy", fund_data.get("prospectus_80pct_policy"), self._print_prospectus_section),
            ("40 Act Diversification", fund_data.get("diversification_40act_check"), self._print_40_act_section),
            ("IRS Diversification", fund_data.get("diversification_IRS_check"), self._print_irs_section),
            ("IRC Diversification", fund_data.get("diversification_IRC_check"), self._print_irc_section),
            ("Illiquid Holdings", fund_data.get("max_15pct_illiquid_sai"), self._print_illiquid_section),
            ("Real Estate", fund_data.get("real_estate_check"), self._print_real_estate_section),
            ("Commodities", fund_data.get("commodities_check"), self._print_commodities_section),
            ("Rule 12d-1", fund_data.get("twelve_d1a_other_inv_cos"), self._print_12d1_section),
            ("Rule 12d-2", fund_data.get("twelve_d2_insurance_cos"), self._print_12d2_section),
            ("Rule 12d-3", fund_data.get("twelve_d3_sec_biz"), self._print_12d3_section),
        ]

        for title, data, renderer in sections:
            if data:
                self._print_section_header(title)
                renderer(data)
                self.pdf.ln(3)

        self.pdf.ln(4)
        self.pdf.set_draw_color(180, 180, 180)
        y = self.pdf.get_y()
        self.pdf.line(10, y, 200, y)
        self.pdf.ln(6)

    # ------------------------------------------------------------------
    def _print_section_header(self, title: str) -> None:
        self.pdf.set_font("Arial", "B", 10)
        self.pdf.cell(0, 6, self._sanitize_text(title), ln=True)
        self.pdf.set_font("Arial", size=9)

    def _print_summary_table(self, fund_data: Mapping[str, object]) -> None:
        summary = fund_data.get("summary_metrics", {})
        rows = [
            ("Cash", format_number(summary.get("cash_value", 0.0))),
            ("Treasury", format_number(summary.get("treasury", 0.0))),
            ("Equity", format_number(summary.get("equity_market_value", 0.0))),
            ("Option DAN", format_number(summary.get("option_delta_adjusted_notional", 0.0))),
            ("Option MV", format_number(summary.get("option_market_value", 0.0))),
        ]
        self._draw_two_column_table(rows)

    def _print_prospectus_section(self, data: Mapping[str, object]) -> None:
        calc = data.get("calculations", {})
        rows = [
            ("Compliance (DAN)", "PASS" if data.get("is_compliant") else "FAIL"),
            ("Names Test (DAN)", f"{calc.get('names_test', 0.0):.2%}"),
            ("Names Test (Market Value)", f"{calc.get('names_test_mv', 0.0):.2%}"),
            ("Threshold", f"{calc.get('threshold', 0.8):.0%}"),
            ("Options In Scope", "Yes" if data.get("options_in_scope") else "No"),
        ]
        self._draw_two_column_table(rows)

    def _print_40_act_section(self, data: Mapping[str, object]) -> None:
        rows = [
            ("Condition 1", "PASS" if data.get("condition_40act_1") else "FAIL"),
            ("Condition 2a", "PASS" if data.get("condition_40act_2a") else "FAIL"),
            ("Condition 2b", "PASS" if data.get("condition_40act_2b") else "FAIL"),
            ("Condition 2a OCC", "PASS" if data.get("condition_40act_2a_occ") else "FAIL"),
        ]
        self._draw_two_column_table(rows)

    def _print_irs_section(self, data: Mapping[str, object]) -> None:
        rows = [
            ("Condition 1", "PASS" if data.get("condition_IRS_1") else "FAIL"),
            ("Condition 2a 50%", "PASS" if data.get("condition_IRS_2_a_50") else "FAIL"),
            ("Condition 2a 5%", "PASS" if data.get("condition_IRS_2_a_5") else "FAIL"),
            ("Condition 2a 10%", "PASS" if data.get("condition_IRS_2_a_10") else "FAIL"),
        ]
        self._draw_two_column_table(rows)

    def _print_irc_section(self, data: Mapping[str, object]) -> None:
        rows = [
            ("Condition 55", "PASS" if data.get("condition_IRC_55") else "FAIL"),
            ("Condition 70", "PASS" if data.get("condition_IRC_70") else "FAIL"),
            ("Condition 80", "PASS" if data.get("condition_IRC_80") else "FAIL"),
            ("Condition 90", "PASS" if data.get("condition_IRC_90") else "FAIL"),
        ]
        self._draw_two_column_table(rows)

    def _print_illiquid_section(self, data: Mapping[str, object]) -> None:
        rows = [
            ("Illiquid Compliance", "PASS" if data.get("max_15pct_illiquid_sai") else "FAIL"),
            ("Equity >=85%", "PASS" if data.get("equity_holdings_85pct_compliant") else "FAIL"),
        ]
        self._draw_two_column_table(rows)

    def _print_real_estate_section(self, data: Mapping[str, object]) -> None:
        calc = data.get("calculations", {})
        rows = [
            ("Real Estate %", f"{calc.get('real_estate_percentage', 0.0):.2%}"),
        ]
        self._draw_two_column_table(rows)

    def _print_commodities_section(self, data: Mapping[str, object]) -> None:
        calc = data.get("calculations", {})
        exposure = calc.get("commodities_percentage")
        if exposure is None:
            exposure = calc.get("commodities_exposure", 0.0)
        rows = [("Commodities %", f"{exposure:.2%}")]
        self._draw_two_column_table(rows)

    def _print_12d1_section(self, data: Mapping[str, object]) -> None:
        rows = [
            ("Compliant", "PASS" if data.get("twelve_d1a_other_inv_cos_compliant") else "FAIL"),
            ("Test 1", "PASS" if data.get("test_1_pass") else "FAIL"),
            ("Test 2", "PASS" if data.get("test_2_pass") else "FAIL"),
            ("Test 3", "PASS" if data.get("test_3_pass") else "FAIL"),
        ]
        self._draw_two_column_table(rows)

    def _print_12d2_section(self, data: Mapping[str, object]) -> None:
        rows = [("Compliant", "PASS" if data.get("test_pass") else "FAIL")]
        self._draw_two_column_table(rows)

    def _print_12d3_section(self, data: Mapping[str, object]) -> None:
        rows = [
            ("Rule 1", "PASS" if data.get("rule_1_pass") else "FAIL"),
            ("Rule 2", "PASS" if data.get("rule_2_pass") else "FAIL"),
            ("Rule 3", "PASS" if data.get("rule_3_pass") else "FAIL"),
            ("Compliant", "PASS" if data.get("twelve_d3_sec_biz_compliant") else "FAIL"),
        ]
        self._draw_two_column_table(rows)


def generate_compliance_reports(
    results: Mapping[str, Mapping[str, object]],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    file_name_prefix: str = "compliance_results",
    test_functions: Optional[Iterable[str]] = None,
    gics_mapping: Optional[pd.DataFrame] = None,
    create_pdf: bool = True,
) -> GeneratedComplianceReport:
    """Create Excel and PDF compliance reports."""

    report = ComplianceReport(
        results=results,
        report_date=report_date,
        output_dir=output_dir,
        file_name_prefix=file_name_prefix,
        test_functions=test_functions,
        gics_mapping=gics_mapping,
    )

    pdf_path = None
    if create_pdf:
        try:
            ComplianceReportPDF(report.results, str(Path(output_dir) / f"{file_name_prefix}_{report.report_date}.pdf"))
            pdf_path = os.path.join(output_dir, f"{file_name_prefix}_{report.report_date}.pdf")
        except RuntimeError as exc:  # pragma: no cover - optional dependency missing
            logger.warning("PDF generation skipped: %s", exc)

    return GeneratedComplianceReport(excel_path=str(report.file_path), pdf_path=pdf_path)