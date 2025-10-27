"""Generate Excel and PDF summaries for compliance results."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

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
    def _append_footnotes(self, df: pd.DataFrame, footnote_key: str) -> pd.DataFrame:
        notes = FOOTNOTES.get(footnote_key, [])
        if not notes:
            return df

        columns = list(df.columns) if not df.empty else ["Note"]
        target_column = "Date" if "Date" in columns else columns[0]

        footnote_rows = []
        for note in notes:
            row = {column: "" for column in columns}
            row[target_column] = f"* {note}"
            footnote_rows.append(row)

        footnotes_df = pd.DataFrame(footnote_rows)
        if df.empty:
            return footnotes_df
        return pd.concat([df, footnotes_df], ignore_index=True)

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

        columns = [
            "Date",
            "Fund",
            "Cash",
            "Treasury",
            "Equity",
            "Option DAN",
            "Option MV",
            "Total Assets",
            "Total Net Assets",
        ]

        if not rows:
            self.sheet_data["Summary"] = pd.DataFrame(columns=columns)
            return

        df = pd.DataFrame(rows, columns=columns)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        numeric_columns = [column for column in columns if column not in {"Date", "Fund"}]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

        df.sort_values(by=["Date", "Fund"], inplace=True)
        df.reset_index(drop=True, inplace=True)
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
    # ------------------------------------------------------------------
    def process_prospectus_80pct_policy(self) -> None:
        formatted_rows: list[OrderedDict[str, object]] = []
        excel_row = 2

        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                prospectus = fund_data.get("prospectus_80pct_policy")
                if not prospectus:
                    logger.warning("No prospectus 80%% policy data for %s", fund_name)
                    continue

                calculations = prospectus.get("calculations", {}) or {}

                total_equity_market_value = calculations.get("total_equity_market_value", 0.0) or 0.0
                total_option_dan = abs(calculations.get("total_opt_delta_notional_value", 0.0) or 0.0)
                total_opt_market_value = calculations.get("total_opt_market_value", 0.0) or 0.0
                total_tbill_value = calculations.get("total_tbill_value", 0.0) or 0.0
                total_cash_value = calculations.get("total_cash_value", 0.0) or 0.0

                options_in_scope = bool(prospectus.get("options_in_scope"))

                numerator = (
                    total_equity_market_value
                    + (total_option_dan if options_in_scope else 0.0)
                    + total_tbill_value
                )
                denominator = (
                    total_equity_market_value
                    + total_option_dan
                    + total_cash_value
                    + total_tbill_value
                )

                numerator_mv = (
                    total_equity_market_value
                    + (total_opt_market_value if options_in_scope else 0.0)
                    + total_tbill_value
                )
                denominator_mv = (
                    total_equity_market_value
                    + total_opt_market_value
                    + total_cash_value
                    + total_tbill_value
                )

                max_ccet_dan = max((total_cash_value + total_tbill_value) - total_option_dan, 0.0)

                row_data = OrderedDict(
                    [
                        ("Date", report_date),
                        ("Fund", fund_name),
                        (
                            "Prospectus 80% Compliance",
                            f'=IF(K{excel_row}/J{excel_row}>=0.8,"PASS","FAIL")',
                        ),
                        ("Total Equity Market Value", total_equity_market_value),
                        ("Total Option Delta Notional Value (DAN)", total_option_dan),
                        ("Total Option Market Value", total_opt_market_value),
                        ("Total T-Bill Value", total_tbill_value),
                        ("Total Cash Value", total_cash_value),
                        ("Max(CCET - DAN, 0)", max_ccet_dan),
                        ("Denominator", denominator),
                        ("Numerator", numerator),
                        ("Formula", f"=K{excel_row}/J{excel_row}"),
                        ("Denominator (Market Value)", denominator_mv),
                        ("Numerator (Market Value)", numerator_mv),
                        ("Formula (MV)", f"=N{excel_row}/M{excel_row}"),
                        ("Options in Scope for 80%?", "Yes" if options_in_scope else "No"),
                    ]
                )

                formatted_rows.append(row_data)
                excel_row += 1

        if formatted_rows:
            df = pd.DataFrame(formatted_rows)
            df = self._append_footnotes(df, "prospectus_80pct")
            self.sheet_data["Prospectus_80pct"] = df

    # ------------------------------------------------------------------
    def process_40_act_diversification(self) -> None:
        formatted_rows: list[OrderedDict[str, object]] = []
        excel_row = 2

        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                forty_act = fund_data.get("diversification_40act_check")
                if not forty_act:
                    continue

                calculations = forty_act.get("calculations", {}) or {}

                issuer_limited = calculations.get("issuer_limited_securities", []) or []
                if issuer_limited:
                    issuer_limited_str = ", ".join(
                        f"({sec.get('equity_ticker')}, {sec.get('quantity')}, {sec.get('net_market_value', 0):,.2f})"
                        for sec in issuer_limited
                    )
                else:
                    issuer_limited_str = "None"

                excluded = calculations.get("excluded_securities", []) or []
                excluded_str = ", ".join(excluded) if excluded else "None"

                remaining_details = calculations.get("remaining_stocks_details", []) or []
                if remaining_details:
                    remaining_str = ", ".join(
                        "({ticker}, {shares:,.0f}, {vest_weight:.2%}, {ownership:.2%})".format(
                            ticker=entry.get('equity_ticker'),
                            shares=entry.get('quantity', 0) or 0,
                            vest_weight=float(entry.get('vest_weight', 0.0) or 0.0),
                            ownership=float(entry.get('vest_ownership_of_float', 0.0) or 0.0),
                        )
                        for entry in remaining_details
                    )
                else:
                    remaining_str = "None"

                max_ownership_float = float(calculations.get("max_ownership_float", 0.0) or 0.0)
                occ_market_value = float(calculations.get("occ_market_value", 0.0) or 0.0)

                notes = ""
                if fund_name in {"FDND", "TDVI"} and forty_act.get("condition_40act_2a"):
                    if calculations.get("issuer_limited_sum", 0):
                        notes = (
                            "NOTE: Jeremy indicated this fund should FAIL condition 2a "
                            "(has securities >5% in 75% bucket)."
                        )

                row_data = OrderedDict(
                    [
                        ("Date", report_date),
                        ("Fund", fund_name),
                        ("Fund Registration", calculations.get("fund_registration")),
                        (
                            "Condition 40 Act 1",
                            f'=IF(J{excel_row}/H{excel_row}>=0.75,"PASS","FAIL")',
                        ),
                        (
                            "Condition 40 Act 2a",
                            f'=IF(L{excel_row}=0,"PASS","FAIL")',
                        ),
                        (
                            "Condition 40 Act 2b",
                            f'=IF(R{excel_row}="","",IF(R{excel_row}<=0.1,"PASS","FAIL"))',
                        ),
                        (
                            "Condition 2a OCC",
                            f'=IF(S{excel_row}=0,"PASS",IF(S{excel_row}<=0.05*J{excel_row},"PASS","FAIL"))',
                        ),
                        ("Total Assets", float(calculations.get("total_assets", 0.0) or 0.0)),
                        (
                            "Non-Qualifying Assets Weight",
                            float(calculations.get("non_qualifying_assets_weight", 0.0) or 0.0),
                        ),
                        ("Net Assets", float(calculations.get("net_assets", 0.0) or 0.0)),
                        ("Expenses", abs(float(calculations.get("expenses", 0.0) or 0.0))),
                        (
                            "Issuer Limited Assets (Condition 2a) Sum",
                            float(calculations.get("issuer_limited_sum", 0.0) or 0.0),
                        ),
                        (
                            "Issuer Limited Assets (Condition 2a) (Ticker, Shares, Mkt Value)",
                            issuer_limited_str,
                        ),
                        ("Excluded Securities (25% Bucket)", excluded_str),
                        (
                            "Cumulative Weight of Excluded Securities",
                            float(calculations.get("cumulative_weight_excluded", 0.0) or 0.0),
                        ),
                        (
                            "Cumulative Weight of Remaining Securities",
                            float(calculations.get("cumulative_weight_remaining", 0.0) or 0.0),
                        ),
                        (
                            "Remaining Securities (Ticker, Shares Held, Vest Weight, Vest Ownership of Float)",
                            remaining_str,
                        ),
                        ("Max Ownership % of Float in 75% bucket", max_ownership_float),
                        ("OCC Market Value", occ_market_value),
                        ("OCC Weight", float(calculations.get("occ_weight", 0.0) or 0.0)),
                        ("Notes", notes),
                    ]
                )

                formatted_rows.append(row_data)
                excel_row += 1

        if formatted_rows:
            df = pd.DataFrame(formatted_rows)
            df = self._append_footnotes(df, "40act")
            self.sheet_data["40Act_Diversification"] = df

    # ------------------------------------------------------------------
    def process_irs_diversification(self) -> None:
        formatted_rows: list[OrderedDict[str, object]] = []
        excel_row = 2

        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                irs = fund_data.get("diversification_IRS_check")
                if not irs:
                    continue

                calculations = irs.get("calculations", {}) or {}

                large_securities = calculations.get("large_securities", []) or []
                if large_securities:
                    large_securities_str = ", ".join(
                        f"({sec.get('equity_ticker', '')}, {float(sec.get('tna_wgt', 0) or 0):.2%})"
                        for sec in large_securities
                    )
                else:
                    large_securities_str = "None"

                largest = calculations.get("bottom_50_largest", {}) or {}
                second = calculations.get("bottom_50_second_largest", {}) or {}

                row_data = OrderedDict(
                    [
                        ("Date", report_date),
                        ("Fund", fund_name),
                        ("condition_IRS_1", "PASS" if irs.get("condition_IRS_1") else "FAIL"),
                        (
                            "condition_IRS_2_a_50",
                            f'=IF(I{excel_row}/H{excel_row}>=0.5,"PASS","FAIL")',
                        ),
                        (
                            "condition_IRS_2_a_5",
                            f'=IF(L{excel_row}<=0.5,"PASS","FAIL")',
                        ),
                        ("condition_IRS_2_a_10", "PASS" if irs.get("condition_IRS_2_a_10") else "FAIL"),
                        ("condition_IRS_2_b", "PASS" if irs.get("condition_IRS_2_b", True) else "FAIL"),
                        ("total_assets", float(calculations.get("total_assets", 0.0) or 0.0)),
                        (
                            "qualifying_assets_value",
                            float(calculations.get("qualifying_assets_value", 0.0) or 0.0),
                        ),
                        ("expenses", abs(float(calculations.get("expenses", 0.0) or 0.0))),
                        (
                            "five_pct_gross_assets",
                            float(calculations.get("five_pct_gross_assets", 0.0) or 0.0),
                        ),
                        (
                            "sum_large_securities_weights",
                            float(calculations.get("sum_large_securities_weights", 0.0) or 0.0),
                        ),
                        (
                            "large_securities_count",
                            int(calculations.get("large_securities_count", 0) or 0),
                        ),
                        ("large_securities", large_securities_str),
                        (
                            "bottom_50_largest_holding",
                            f"{largest.get('equity_ticker', 'N/A')} ({float(largest.get('tna_wgt', 0) or 0):.2%})",
                        ),
                        (
                            "bottom_50_second_largest_holding",
                            f"{second.get('equity_ticker', 'N/A')} ({float(second.get('tna_wgt', 0) or 0):.2%})",
                        ),
                    ]
                )

                formatted_rows.append(row_data)
                excel_row += 1

        if formatted_rows:
            df = pd.DataFrame(formatted_rows)
            df = self._append_footnotes(df, "irs")
            self.sheet_data["IRS_Diversification"] = df

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
            df = self._append_footnotes(df, "irc")
            self.sheet_data["IRC_Diversification"] = df

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
            df = self._append_footnotes(df, "real_estate")
            self.sheet_data["Real_Estate"] = df

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
            df = self._append_footnotes(df, "commodities")
            self.sheet_data["Commodities"] = df

    # ------------------------------------------------------------------
    def process_12d1_other_inv_cos(self) -> None:
        formatted_rows: list[OrderedDict[str, object]] = []
        excel_row = 2

        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                rule = fund_data.get("twelve_d1a_other_inv_cos")
                if not rule:
                    continue

                calculations = rule.get("calculations", {}) or {}
                holdings = calculations.get("investment_companies", []) or []
                if holdings:
                    holdings_str = ", ".join(
                        f"{holding.get('equity_ticker') or holding.get('ticker')} ({(holding.get('ownership_pct') or 0)*100:.2f}%)"
                        for holding in holdings
                    )
                else:
                    holdings_str = "None"

                row_data = OrderedDict(
                    [
                        ("Date", report_date),
                        ("Fund", fund_name),
                        ("Total Assets", float(calculations.get("total_assets", 0.0) or 0.0)),
                        ("Investment Companies", holdings_str),
                        (
                            "Ownership % Max",
                            float(calculations.get("ownership_pct_max", 0.0) or 0.0),
                        ),
                        (
                            "Equity Market Value Sum",
                            float(calculations.get("equity_market_value_sum", 0.0) or 0.0),
                        ),
                        (
                            "Test 1 (<=3% Ownership)",
                            f'=IF(E{excel_row}<=0.03,"PASS","FAIL")',
                        ),
                        (
                            "Test 2 (<=5% Total Assets)",
                            f'=IF(F{excel_row}/C{excel_row}<=0.05,"PASS","FAIL")',
                        ),
                        (
                            "Test 3 (<=10% Total Assets)",
                            f'=IF(F{excel_row}/C{excel_row}<=0.10,"PASS","FAIL")',
                        ),
                        (
                            "12d1(a) Compliant",
                            f'=IF(AND(G{excel_row}="PASS",H{excel_row}="PASS",I{excel_row}="PASS"),"PASS","FAIL")',
                        ),
                    ]
                )

                formatted_rows.append(row_data)
                excel_row += 1

        if formatted_rows:
            df = pd.DataFrame(formatted_rows)
            df = self._append_footnotes(df, "12d1")
            self.sheet_data["12d1_Other_Investment_Companies"] = df

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
            df = self._append_footnotes(df, "12d2")
            self.sheet_data["12d2_Insurance_Companies"] = df

    # ------------------------------------------------------------------
    def process_12d3_sec_biz(self) -> None:
        formatted_rows: list[OrderedDict[str, object]] = []
        excel_row = 2

        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                rule = fund_data.get("twelve_d3_sec_biz")
                if not rule:
                    continue

                calculations = rule.get("calculations", {}) or {}
                sec_related = calculations.get("sec_related_businesses", []) or []
                if sec_related:
                    sec_related_str = ", ".join(
                        f"{biz.get('equity_ticker') or biz.get('ticker')} ({(biz.get('ownership_pct') or 0)*100:.5f}%)"
                        for biz in sec_related
                    )
                else:
                    sec_related_str = "None"

                combined = calculations.get("combined_holdings", []) or []
                if combined:
                    combined_str = ", ".join(
                        f"{holding.get('equity_ticker') or holding.get('ticker')} ({(holding.get('vest_weight') or 0)*100:.2f}%)"
                        for holding in combined
                    )
                else:
                    combined_str = "None"

                row_data = OrderedDict(
                    [
                        ("Date", report_date),
                        ("Fund", fund_name),
                        (
                            "Rule 1 (<=5% equities)",
                            f'=IF(K{excel_row}<=0.05,"PASS","FAIL")',
                        ),
                        (
                            "Rule 2 (<=10% debt)",
                            "PASS" if rule.get("rule_2_pass") else "FAIL",
                        ),
                        (
                            "Rule 3 (<=5% total assets)",
                            f'=IF(L{excel_row}<=0.05,"PASS","FAIL")',
                        ),
                        (
                            "Rule 3 OCC (<=5% TNA)",
                            f'=IF(N{excel_row}<=0.05,"PASS","FAIL")',
                        ),
                        (
                            "12d3 Sec Biz Compliant",
                            f'=IF(AND(C{excel_row}="PASS",D{excel_row}="PASS",E{excel_row}="PASS",F{excel_row}="PASS"),"PASS","FAIL")',
                        ),
                        ("SEC-Related Businesses (Shs Out %)", sec_related_str),
                        ("Combined Holdings (Portfolio %)", combined_str),
                        ("Total Assets", float(calculations.get("total_assets", 0.0) or 0.0)),
                        ("Max Ownership %", float(calculations.get("max_ownership_pct", 0.0) or 0.0)),
                        ("Max Weight", float(calculations.get("max_weight", 0.0) or 0.0)),
                        (
                            "OCC Market Value",
                            float(calculations.get("occ_weight_mkt_val", 0.0) or 0.0),
                        ),
                        ("OCC Weight", float(calculations.get("occ_weight", 0.0) or 0.0)),
                    ]
                )

                formatted_rows.append(row_data)
                excel_row += 1

        if formatted_rows:
            df = pd.DataFrame(formatted_rows)
            df = self._append_footnotes(df, "12d3")
            self.sheet_data["12d3_Securities_Business"] = df

    # ------------------------------------------------------------------
    def process_max_15pct_illiquid(self) -> None:
        formatted_rows: list[OrderedDict[str, object]] = []
        excel_row = 2

        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                illiquid = fund_data.get("max_15pct_illiquid_sai")
                if not illiquid:
                    continue

                calculations = illiquid.get("calculations", {}) or {}

                row_data = OrderedDict(
                    [
                        ("Date", report_date),
                        ("Fund", fund_name),
                        ("Total Assets", float(calculations.get("total_assets", 0.0) or 0.0)),
                        ("Total Illiquid Value", float(calculations.get("total_illiquid_value", 0.0) or 0.0)),
                        ("Illiquid Percentage", float(calculations.get("illiquid_percentage", 0.0) or 0.0)),
                        (
                            "Equity Holdings Percentage",
                            float(calculations.get("equity_holdings_percentage", 0.0) or 0.0),
                        ),
                        (
                            "Max 15% Illiquid Compliance",
                            f'=IF(E{excel_row}<=0.15,"PASS","FAIL")',
                        ),
                        (
                            "Equity Holdings 85% Compliance",
                            f'=IF(F{excel_row}>=0.85,"PASS","FAIL")',
                        ),
                    ]
                )

                formatted_rows.append(row_data)
                excel_row += 1

        if formatted_rows:
            df = pd.DataFrame(formatted_rows)
            df = self._append_footnotes(df, "illiquid")
            self.sheet_data["Illiquid"] = df

    # ------------------------------------------------------------------
    def export_to_excel(self) -> None:
        if not self.sheet_data:
            return

        with pd.ExcelWriter(self.file_path, engine="openpyxl") as writer:
            for sheet_name, df in self.sheet_data.items():
                output_df = df.copy()
                if not output_df.empty:
                    output_df = output_df.where(pd.notnull(output_df), None)
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
                            value = cell.value
                            if value is not None:
                                max_length = max(max_length, len(str(value)))
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
        self.add_title("Compliance Report")
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

    def _draw_two_column_table(self, rows: Iterable[Tuple[object, object]]) -> None:
        if not rows:
            return

        formatted_rows = [(str(label), str(value)) for label, value in rows]
        self.add_table(["Metric", "Value"], formatted_rows, align=["L", "R"], header_fill=False)

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