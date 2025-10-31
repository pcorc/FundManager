"""Generate Excel and PDF summaries for compliance results."""

from __future__ import annotations
from numbers import Real

import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numbers
import re
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
_PERCENT_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?%$")


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

        self.results = self._prepare_results(results)

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
            if self.test_functions is None or test_name in self.test_functions:
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
    def _prepare_results(
        self, raw_results: Mapping[str, Mapping[str, object]]
    ) -> Dict[str, Dict[str, Any]]:
        if not raw_results:
            return {}

        if self._looks_like_date_mapping(raw_results):
            prepared: Dict[str, Dict[str, Any]] = {}
            for date_key, payload in raw_results.items():
                date_iso = self._normalise_date(date_key)
                prepared[date_iso] = normalize_compliance_results(payload or {})
            return prepared

        return {self.report_date: normalize_compliance_results(raw_results)}

    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_date_mapping(payload: Mapping[object, Any]) -> bool:
        if not payload:
            return False

        if not any(ComplianceReport._is_date_key(key) for key in payload.keys()):
            return False

        first_value = next(iter(payload.values()))
        return isinstance(first_value, Mapping)

    # ------------------------------------------------------------------
    @staticmethod
    def _is_date_key(value: object) -> bool:
        if isinstance(value, (date, datetime)):
            return True
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value)
            except ValueError:
                return False
            return True
        return False

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

                overlap_records = calculations.get("overlap", []) or []
                if overlap_records:
                    overlap_summary = ", ".join(
                        f"{item.get('security_ticker', '')}: {float(item.get('security_weight', 0) or 0):.2%}"
                        for item in overlap_records
                    )
                else:
                    overlap_summary = "None"

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
                            "overlap_weight",
                            float(calculations.get("overlap_weight_sum", 0.0) or 0.0),
                        ),
                        ("overlap_constituents", overlap_summary),
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
                        f"{holding.get('equity_ticker') or holding.get('ticker')} ({(holding.get('ownership_pct') or 0):.2%})"
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
                header_row = next(ws.iter_rows(min_row=1, max_row=1), ())
                headers = {
                    cell.column_letter: (cell.value or "")
                    for cell in header_row
                }
                for column_cells in ws.columns:
                    max_length = 0
                    column_letter = column_cells[0].column_letter
                    header_text = str(headers.get(column_letter, "")).lower()
                    is_percent = any(
                        keyword in header_text
                        for keyword in ("percent", "pct", "weight", "ownership")
                    )
                    for cell in column_cells:
                        try:
                            value = cell.value
                            if value is not None:
                                max_length = max(max_length, len(str(value)))
                                if cell.row > 1 and cell.data_type in {"n", "f"}:
                                    cell.number_format = "0.00%" if is_percent else "#,##0.00"
                        except Exception:  # pragma: no cover - defensive
                            continue
                    ws.column_dimensions[column_letter].width = min(max_length + 2, 50)




class ComplianceReportPDF(BaseReportPDF):
    """Render compliance results to a PDF document."""

    def __init__(self, results: Mapping[str, Mapping[str, object]], output_path: str) -> None:
        super().__init__(output_path)
        self.results = flatten_compliance_results(results)
        self.generate_pdf()

    # ------------------------------------------------------------------
    def generate_pdf(self) -> None:
        summary_data: list[list[object]] = []
        sorted_results = sorted(
            self.results.items(),
            key=lambda item: (item[0][1], item[0][0]),
        )

        for (fund_name, date_str), fund_data in sorted_results:
            cash = self._extract_numeric(fund_data.get("cash_data"))
            treasury = (
                fund_data.get("prospectus_80pct_policy", {})
                .get("calculations", {})
                .get("total_tbill_value", 0)
            )
            equity = (
                fund_data.get("prospectus_80pct_policy", {})
                .get("calculations", {})
                .get("total_equity_market_value", 0)
            )
            std_option_dan = (
                fund_data.get("prospectus_80pct_policy", {})
                .get("calculations", {})
                .get("total_opt_delta_notional_value", 0)
            )
            std_option_mv = (
                fund_data.get("prospectus_80pct_policy", {})
                .get("calculations", {})
                .get("total_opt_market_value", 0)
            )

            summary_data.append(
                [
                    fund_name,
                    f"{cash:,.0f}",
                    f"{treasury:,.0f}",
                    f"{equity:,.0f}",
                    f"{std_option_dan:,.0f}",
                    f"{std_option_mv:,.0f}",
                ]
            )

        self.pdf.add_page()
        self._add_header("Compliance Summary")
        self._draw_table(
            [
                "Fund",
                "Cash",
                "Treasury",
                "Equity",
                "Std Option (DAN)",
                "Std Option (MV)",
            ],
            summary_data,
        )
        self.pdf.ln(6)

        self._add_header("Detailed Compliance Report")
        for (fund_name, date_str), fund_data in sorted_results:
            report_date = self._parse_date(date_str)
            self._add_fund_section(fund_name, report_date, fund_data)

        self.output()

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_date(value: str) -> object:
        try:
            return datetime.fromisoformat(value).date()
        except Exception:  # pragma: no cover - fallback
            return value

    @staticmethod
    def _extract_numeric(value: object) -> float:
        if isinstance(value, Mapping):
            for key in ("cash_value", "value", "amount", "total", "total_cash_value"):
                if key in value:
                    return float(value.get(key) or 0)
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    @staticmethod
    def _format_numeric(value: object, *, is_percent: bool = False, prefix: str = "") -> str:
        try:
            number = float(value or 0)
        except (TypeError, ValueError):
            return str(value)

        if is_percent:
            return f"{number:.2%}"

        formatted = f"{number:,.0f}"
        return f"{prefix}{formatted}" if prefix else formatted

    def _format_table_value(self, value: object) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            stripped = value.strip()
            if _PERCENT_RE.match(stripped):
                try:
                    return f"{float(stripped.rstrip('%')):.2f}%"
                except ValueError:
                    return value
            return value

        if isinstance(value, numbers.Number) and not isinstance(value, bool):
            if pd.isna(value):
                return ""
            return f"{float(value):,.0f}"

        return str(value)

    # ------------------------------------------------------------------
    def _add_header(self, title: str) -> None:
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, self._sanitize_text(title), ln=True, align="C")
        self.pdf.ln(4)

    def _add_fund_section(self, fund_name: str, report_date: object, fund_data: Mapping[str, object]) -> None:
        exclude_funds = {"PF227", "PD227", "R21126", "FTMIX"}
        exclude_prefixes = ("TR",)

        self.pdf.set_font("Arial", "B", 12)
        header_text = f"Fund: {fund_name}  |  Date: {report_date}"
        self.pdf.cell(0, 8, self._sanitize_text(header_text), ln=True)
        self.pdf.ln(2)

        self.print_summary_table(fund_data)

        tests = [
            ("gics_compliance", self.print_gics_compliance),
            ("prospectus_80pct_policy", self.print_prospectus_80pct_policy),
            ("diversification_40act_check", self.print_40act_diversification),
            ("diversification_IRS_check", self.print_irs_diversification),
            ("diversification_IRC_check", self.print_irc_diversification),
            ("max_15pct_illiquid_sai", self.print_max_15pct_illiquid),
            ("real_estate_check", self.print_real_estate_check),
            ("commodities_check", self.print_commodities_check),
            ("twelve_d1a_other_inv_cos", self.print_12d1_other_inv_cos),
            ("twelve_d2_insurance_cos", self.print_12d2_insurance_cos),
            ("twelve_d3_sec_biz", self.print_12d3_sec_biz),
        ]

        for key, renderer in tests:
            if key == "prospectus_80pct_policy" and (
                fund_name.startswith(exclude_prefixes) or fund_name in exclude_funds
            ):
                continue

            test_data = fund_data.get(key)
            if test_data:
                self._print_test_header(key.replace("_", " ").title())
                renderer(test_data)
                self.pdf.ln(4)

        y = self.pdf.get_y()
        self.pdf.set_draw_color(150, 150, 150)
        self.pdf.line(10, y, 200, y)
        self.pdf.ln(6)

    def _print_test_header(self, title: str) -> None:
        self.pdf.set_font("Arial", "B", 10)
        self.pdf.cell(0, 6, self._sanitize_text(title), ln=True)
        self.pdf.set_font("Arial", "", 9)

    # ------------------------------------------------------------------
    def _draw_table(
        self,
        headers: Iterable[object],
        rows: Iterable[Iterable[object]],
        col_widths: Optional[Iterable[float]] = None,
        row_height: int = 8,
    ) -> None:
        headers = list(headers)
        rows = [list(row) for row in rows]
        if not headers or not rows:
            return

        num_cols = len(headers)
        widths = list(col_widths) if col_widths else []
        if len(widths) != num_cols:
            total_width = 190
            widths = [total_width / num_cols] * num_cols

        self.pdf.set_font("Arial", "B", 9)
        self.pdf.set_fill_color(230, 230, 230)
        for width, header in zip(widths, headers):
            self.pdf.cell(width, row_height, self._sanitize_text(str(header)), border=1, align="C", fill=True)
        self.pdf.ln(row_height)

        self.pdf.set_font("Arial", "", 8)
        for row in rows:
            for idx in range(num_cols):
                text = self._sanitize_text(str(row[idx]) if idx < len(row) else "")
                if text.upper() == "FAIL":
                    self.pdf.set_fill_color(255, 200, 200)
                    self.pdf.set_text_color(139, 0, 0)
                    self.pdf.cell(widths[idx], row_height, text, border=1, align="C", fill=True)
                    self.pdf.set_text_color(0, 0, 0)
                else:
                    self.pdf.set_fill_color(255, 255, 255)
                    self.pdf.cell(widths[idx], row_height, text, border=1, align="C")
            self.pdf.ln(row_height)

    def _draw_footnotes_section(self, key: str) -> None:
        notes = FOOTNOTES.get(key, [])
        if not notes:
            return

        self.pdf.set_font("Arial", "I", 8)
        self.pdf.ln(1)
        self.pdf.multi_cell(0, 5, self._sanitize_text("Footnotes:"))
        for note in notes:
            self.pdf.multi_cell(0, 4.5, self._sanitize_text(f"* {note}"))
        self.pdf.ln(2)

    def _draw_two_column_table(self, rows: Iterable[Tuple[object, object]]) -> None:
        if not rows:
            return

        usable_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin
        label_width = min(usable_width * 0.4, 80)
        value_width = usable_width - label_width
        line_height = 5

        header_x = self.pdf.get_x()
        header_y = self.pdf.get_y()
        self.pdf.set_font("Arial", "B", 8)
        self.pdf.set_fill_color(230, 230, 230)
        self.pdf.multi_cell(
            label_width,
            line_height,
            self._sanitize_text("Metric"),
            border=1,
            fill=True,
        )
        header_label_end = self.pdf.get_y()

        self.pdf.set_xy(header_x + label_width, header_y)
        self.pdf.multi_cell(
            value_width,
            line_height,
            self._sanitize_text("Value"),
            border=1,
            fill=True,
            align="R",
        )
        header_value_end = self.pdf.get_y()
        header_end_y = max(header_label_end, header_value_end)
        self.pdf.set_xy(header_x, header_end_y)

        self.pdf.set_fill_color(255, 255, 255)
        self.pdf.set_font("Arial", "", 8)
        for label, value in rows:
            sanitized_label = self._sanitize_text(label)
            display_value = value
            if isinstance(display_value, Real) and not isinstance(display_value, bool):
                numeric_value = float(display_value)
                if pd.isna(numeric_value):
                    numeric_value = 0.0
                display_value = f"{numeric_value:,.0f}"
            sanitized_value = self._sanitize_text(display_value)

            x_start = self.pdf.get_x()
            y_start = self.pdf.get_y()

            self.pdf.multi_cell(label_width, line_height, sanitized_label, border=1)
            label_end_y = self.pdf.get_y()

            self.pdf.set_xy(x_start + label_width, y_start)
            value_align = "R"
            if "\n" in sanitized_value or any(ch.isalpha() for ch in sanitized_value):
                value_align = "L"
            self.pdf.multi_cell(
                value_width,
                line_height,
                sanitized_value,
                border=1,
                align=value_align,
            )
            value_end_y = self.pdf.get_y()

            row_end_y = max(label_end_y, value_end_y)
            self.pdf.set_xy(x_start, row_end_y)

        self.pdf.ln(2)

    # ------------------------------------------------------------------
    def print_summary_table(self, fund_data: Mapping[str, object]) -> None:
        cash_value = self._extract_numeric(fund_data.get("cash_data"))
        prospectus_calc = (
            fund_data.get("prospectus_80pct_policy", {})
            .get("calculations", {})
        )
        treasury = prospectus_calc.get("total_tbill_value", 0)
        equity = prospectus_calc.get("total_equity_market_value", 0)
        std_option_dan = prospectus_calc.get("total_opt_delta_notional_value", 0)
        std_option_mv = prospectus_calc.get("total_opt_market_value", 0)

        def format_num(value: object) -> str:
            try:
                return f"${int(round(float(value or 0))):,}"
            except Exception:  # pragma: no cover - fallback formatting
                return str(value)

        rows = [
            ("Cash", format_num(cash_value)),
            ("Treasury", format_num(treasury)),
            ("Equity", format_num(equity)),
            ("Std Options (DAN)", format_num(std_option_dan)),
            ("Std Options (MV)", format_num(std_option_mv)),
        ]

        self._draw_two_column_table(rows)
        self.pdf.ln(4)

    def print_gics_compliance(self, data: Mapping[str, object]) -> None:
        rows = [
            ("Overall Gics Compliance", data.get("overall_gics_compliance", "N/A")),
            ("Industry Exceeds 25%", "YES" if data.get("industry_exceeds_25") else "NO"),
            (
                "Industry Group Exceeds 25%",
                "YES" if data.get("industry_group_exceeds_25") else "NO",
            ),
        ]
        self._draw_two_column_table(rows)
        self._draw_footnotes_section("gics")

    def print_prospectus_80pct_policy(self, data: Mapping[str, object]) -> None:
        calculations = data.get("calculations", {})
        total_eqy_mv = calculations.get("total_equity_market_value", 0)
        total_opt_dan = calculations.get("total_opt_delta_notional_value", 0)
        total_opt_mv = calculations.get("total_opt_market_value", 0)
        total_tbill = calculations.get("total_tbill_value", 0)
        total_cash = calculations.get("total_cash_value", 0)

        denom = calculations.get("denominator") or 0
        numer = calculations.get("numerator") or 0
        names_test = calculations.get("names_test", numer / denom if denom else 0)

        denom_mv = calculations.get("denominator_mv") or 0
        numer_mv = calculations.get("numerator_mv") or 0
        names_test_mv = calculations.get(
            "names_test_mv", numer_mv / denom_mv if denom_mv else 0
        )

        options_in_scope = calculations.get("options_in_scope", "No")

        rows = [
            ("Prospectus 80% Compliance (DAN)", "PASS" if names_test >= 0.80 else "FAIL"),
            ("Total Equity Market Value", f"{total_eqy_mv:,.0f}"),
            ("Total Option Delta Notional Value (DAN)", f"{total_opt_dan:,.0f}"),
            ("Total T-Bill Value", f"{total_tbill:,.0f}"),
            ("Total Cash Value", f"{total_cash:,.0f}"),
            ("Denominator (DAN)", f"{denom:,.0f}"),
            ("Numerator (DAN)", f"{numer:,.0f}"),
            ("Formula Result (DAN)", f"{names_test:.2%}"),
            ("Options in Scope for 80%?", options_in_scope),
            ("---", "---"),
            (
                "Prospectus 80% Compliance (Market Value)",
                "PASS" if names_test_mv >= 0.80 else "FAIL",
            ),
            ("Total Option Market Value", f"{total_opt_mv:,.0f}"),
            ("Denominator (MV)", f"{denom_mv:,.0f}"),
            ("Numerator (MV)", f"{numer_mv:,.0f}"),
            ("Formula Result (MV)", f"{names_test_mv:.2%}"),
        ]
        self._draw_two_column_table(rows)
        self._draw_footnotes_section("prospectus_80pct")

    def print_40act_diversification(self, data: Mapping[str, object]) -> None:
        calculations = data.get("calculations", {})

        total_assets = calculations.get("total_assets", 0)
        non_qual_weight = calculations.get("non_qualifying_assets_1_wgt", 0)
        issuer_limited_sum = calculations.get("issuer_limited_sum", 0)
        cumulative_excluded = calculations.get("cumulative_weight_excluded", 0)
        cumulative_remaining = calculations.get("cumulative_weight_remaining", 0)
        occ_mkt_val = calculations.get("occ_market_value", 0)

        issuer_limited_assets_list = calculations.get("issuer_limited_securities", [])
        if isinstance(issuer_limited_assets_list, list) and issuer_limited_assets_list:
            issuer_limited_assets_str = ", ".join(
                f"({item.get('equity_ticker')}, {abs(item.get('net_market_value', 0)) / total_assets:.2%})"
                for item in issuer_limited_assets_list
                if total_assets
            )
        else:
            issuer_limited_assets_str = "None"

        rows = [
            ("Fund Registration", calculations.get("fund_registration")),
            ("Condition 40 Act 1", "PASS" if data.get("condition_40act_1") else "FAIL"),
            ("Condition 40 Act 2a", "PASS" if data.get("condition_40act_2a") else "FAIL"),
            ("Condition 40 Act 2b", "PASS" if data.get("condition_40act_2b") else "FAIL"),
            ("Total Assets", f"{total_assets:,.0f}"),
            ("Non-Qualifying Assets Weight", f"{non_qual_weight:.2%}"),
            ("Issuer Limited Assets (Sum)", f"{issuer_limited_sum:,.0f}"),
            ("Issuer Limited Assets (Detail)", issuer_limited_assets_str),
            ("", ""),
            ("", ""),
            ("Cumulative Weight Excluded", f"{cumulative_excluded:.2%}"),
            ("Cumulative Weight Remaining", f"{cumulative_remaining:.2%}"),
            ("OCC Market Value", f"{occ_mkt_val:,.0f}"),
        ]

        self._draw_two_column_table(rows)
        self._draw_footnotes_section("40act")

    def print_irs_diversification(self, data: Mapping[str, object]) -> None:
        calculations = data.get("calculations", {})
        largest_holding = calculations.get("largest_holding", {})

        large_securities = calculations.get("large_securities", []) or []
        if large_securities:
            large_lines: list[str] = []
            for sec in large_securities:
                ticker = str(sec.get("equity_ticker", ""))
                market_value = float(sec.get("net_market_value", 0) or 0)
                if pd.isna(market_value):
                    market_value = 0.0
                weight = float(sec.get("tna_wgt", 0) or 0)
                if pd.isna(weight):
                    weight = 0.0
                large_lines.append(
                    " | ".join(
                        [
                            ticker,
                            f"${market_value:,.0f}",
                            f"{weight:.2%}",
                        ]
                    )
                )
            large_securities_str = "\n".join(large_lines)
        else:
            large_securities_str = "None"

        large_securities_count = calculations.get("large_securities_count", 0)
        if isinstance(large_securities_count, (int, float)) and not pd.isna(
            large_securities_count
        ):
            large_count_display = f"{int(float(large_securities_count))}"
        else:
            large_count_display = str(large_securities_count or 0)

        rows = [
            ("Condition IRS 1", "PASS" if data.get("condition_IRS_1") else "FAIL"),
            ("Condition IRS 2a_50%", "PASS" if data.get("condition_IRS_2_a_50") else "FAIL"),
            ("2a_50% Weight", f"{calculations.get('weight_2a_50', 0):.2%}"),
            ("Condition IRS 2a_5%", "PASS" if data.get("condition_IRS_2_a_5") else "FAIL"),
            ("Condition IRS 2a_10%", "PASS" if data.get("condition_IRS_2_a_10") else "FAIL"),
            ("Total Assets", f"{calculations.get('total_assets', 0):,.0f}"),
            ("Expenses", f"{calculations.get('expenses', 0):,.0f}"),
            ("Qualifying Assets Value", f"{calculations.get('qualifying_assets_value', 0):,.0f}"),
            ("Largest Holding $", f"{largest_holding.get('net_market_value', 0):,.0f}"),
            ("Largest Holding %", f"{largest_holding.get('tna_wgt', 0):.2%}"),
            ("Condition IRS 2b", "PASS" if data.get("condition_IRS_2_b") else "FAIL"),
            ("5% Gross Assets", f"{calculations.get('five_pct_gross_assets', 0):,.0f}"),
            ("Sum Large Securities %", f"{calculations.get('sum_large_securities_weights', 0):.2%}"),
            ("Large Securities Count", large_count_display),
            (
                "Large Securities",
                large_securities_str,
            ),
        ]

        self._draw_two_column_table(rows)
        self.pdf.ln(6)

        if self.pdf.get_y() > 250:
            self.pdf.add_page()

        self._draw_footnotes_section("irs")

    def print_irc_diversification(self, data: Mapping[str, object]) -> None:
        calculations = data.get("calculations", {})

        rows = [
            ("Condition IRC 55", "PASS" if data.get("condition_IRC_55") else "FAIL"),
            ("Condition IRC 70", "PASS" if data.get("condition_IRC_70") else "FAIL"),
            ("Condition IRC 80", "PASS" if data.get("condition_IRC_80") else "FAIL"),
            ("Condition IRC 90", "PASS" if data.get("condition_IRC_90") else "FAIL"),
            ("Top 1 Exposure", f"{calculations.get('top_1', 0):.2%}"),
            ("Top 2 Exposure", f"{calculations.get('top_2', 0):.2%}"),
            ("Top 3 Exposure", f"{calculations.get('top_3', 0):.2%}"),
            ("Top 4 Exposure", f"{calculations.get('top_4', 0):.2%}"),
            ("Total Assets", f"{calculations.get('total_assets', 0):,.0f}"),
        ]

        self._draw_two_column_table(rows)
        self._draw_footnotes_section("irc")

    def print_real_estate_check(self, data: Mapping[str, object]) -> None:
        real_estate_pct = data.get("real_estate_percentage", 0)
        exposure = "None"
        rows = [("Real Estate Exposure", exposure)]

        self._draw_two_column_table(rows)
        self._draw_footnotes_section("real_estate")

    def print_commodities_check(self, data: Mapping[str, object]) -> None:
        rows = [("Commodities Exposure", data.get("Commodities Exposure", "None"))]
        self._draw_two_column_table(rows)
        self._draw_footnotes_section("commodities")

    def print_12d1_other_inv_cos(self, data: Mapping[str, object]) -> None:
        calculations = data.get("calculations", {})
        inv_companies = calculations.get("investment_companies", [])

        holdings_str = ", ".join(
            f"{c.get('equity_ticker', '')} ({float(c.get('ownership_pct', 0) or 0):.2%})"
            for c in inv_companies
        ) or "None"

        rows = [
            ("Total Assets", f"{calculations.get('total_assets', 0):,.0f}"),
            ("Investment Companies", holdings_str),
            ("Ownership % Max", f"{(calculations.get('ownership_pct_max') or 0):.2%}"),
            ("Equity Market Value Sum", f"{calculations.get('equity_market_value_sum', 0):,.0f}"),
            ("Test 1 (<=3% Ownership)", "PASS" if data.get("test_1_pass") else "FAIL"),
            ("Test 2 (<=5% Total Assets)", "PASS" if data.get("test_2_pass") else "FAIL"),
            ("Test 3 (<=10% Total Assets)", "PASS" if data.get("test_3_pass") else "FAIL"),
            ("12d1(a) Compliant", "PASS" if data.get("twelve_d1a_other_inv_cos_compliant") else "FAIL"),
        ]

        self._draw_two_column_table(rows)
        self._draw_footnotes_section("12d1")


    def print_12d2_insurance_cos(self, data: Mapping[str, object]) -> None:
        calculations = data.get("calculations", {})
        holdings = calculations.get("insurance_holdings", [])

        holdings_str = ", ".join(
            f"{h.get('equity_ticker', '')} ({float(h.get('ownership_pct', 0) or 0):.2%})"
            for h in holdings
        ) or "None"

        rows = [
            ("Total Assets", f"{calculations.get('total_assets', 0):,.0f}"),
            ("Insurance Holdings", holdings_str),
        ]

        self._draw_two_column_table(rows)
        self._draw_footnotes_section("12d2")

    def print_12d3_sec_biz(self, data: Mapping[str, object]) -> None:
        calculations = data.get("calculations", {})
        combined = calculations.get("combined_holdings", [])

        if self.pdf.get_y() > 230:
            self.pdf.add_page()

        summary_rows = [
            ("Rule 1 (<=5% equities)", "PASS" if data.get("rule_1_pass") else "FAIL"),
            ("Rule 2 (<=10% debt)", "PASS" if data.get("rule_2_pass") else "FAIL"),
            ("Rule 3 (<=5% total assets)", "PASS" if data.get("rule_3_pass") else "FAIL"),
            ("12d3 Sec Biz Compliant", "PASS" if data.get("twelve_d3_sec_biz_compliant") else "FAIL"),
        ]
        self._draw_two_column_table(summary_rows)

        self.pdf.set_font("Arial", "B", 9)
        self.pdf.cell(0, 6, self._sanitize_text("Investment Holdings"), ln=True)

        headers = ["Ticker", "Vest Weight", "Ownership %"]
        col_widths = [40, 40, 40]
        self.pdf.set_font("Arial", "B", 9)
        for i, header in enumerate(headers):
            self.pdf.set_fill_color(230, 230, 230)
            self.pdf.cell(col_widths[i], 6, self._sanitize_text(header), border=1, fill=True)
        self.pdf.ln()

        self.pdf.set_font("Arial", "", 9)
        for holding in combined:
            ticker = holding.get("ticker") or holding.get("equity_ticker") or "N/A"
            for holding in combined:
                ticker = holding.get("ticker") or holding.get("equity_ticker") or "N/A"
                vest_weight = float(holding.get("vest_weight", 0) or 0)
                ownership_pct = float(holding.get("ownership_pct", 0) or 0)

                self.pdf.set_fill_color(255, 255, 255)
                self.pdf.cell(col_widths[0], 6, self._sanitize_text(str(ticker)), border=1)
                self.pdf.cell(col_widths[1], 6, f"{vest_weight:.2%}", border=1)
                self.pdf.cell(col_widths[2], 6, f"{ownership_pct:.2%}", border=1)
            self.pdf.ln()
        self.pdf.ln(2)

        rows_bottom = [
            ("Total Assets", f"{calculations.get('total_assets', 0):,.0f}"),
            ("OCC Market Value", f"{calculations.get('occ_weight_mkt_val', 0):,.0f}"),
            ("OCC Weight", f"{calculations.get('occ_weight', 0):.2%}"),
        ]
        self._draw_two_column_table(rows_bottom)
        self._draw_footnotes_section("12d3")

    def print_max_15pct_illiquid(self, data: Mapping[str, object]) -> None:
        calculations = data.get("calculations", {})

        total_assets = calculations.get("total_assets", 0)
        illiquid_value = calculations.get("total_illiquid_value", 0)
        illiquid_pct = calculations.get("illiquid_percentage", 0)
        equity_pct = calculations.get("equity_holdings_percentage", 0)

        rows = [
            ("Total Assets", f"{total_assets:,.0f}"),
            ("Total Illiquid Value", f"{illiquid_value:,.0f}"),
            ("Illiquid Percentage", f"{illiquid_pct:.2%}"),
            ("Equity Holdings Percentage", f"{equity_pct:.2%}"),
            ("Max 15% Illiquid Compliance", "PASS" if data.get("max_15pct_illiquid_sai") else "FAIL"),
            (
                "Equity Holdings 85% Compliance",
                "PASS" if data.get("equity_holdings_85pct_compliant") else "FAIL",
            ),
        ]

        self._draw_two_column_table(rows)
        self._draw_footnotes_section("illiquid")


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