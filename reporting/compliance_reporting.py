"""Generate Excel and PDF summaries for compliance results."""

from __future__ import annotations
from functools import lru_cache

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd

from config.fund_definitions import FUND_DEFINITIONS
from reporting.base_report_pdf import BaseReportPDF
from reporting.footnotes import FOOTNOTES
from reporting.report_utils import (
    flatten_compliance_results,
    normalize_compliance_results,
)
from utilities.logger import setup_logger

logger = setup_logger("compliance_report", "logs/compliance.log")
@lru_cache(maxsize=None)
def _vehicle_wrapper(fund_name: str) -> str:
    """Return the lowercase vehicle_wrapper for a fund. Cached for the process."""
    return str(FUND_DEFINITIONS.get(fund_name, {}).get("vehicle_wrapper", "") or "").lower()


def _is_vit_fund(fund_name: str) -> bool:
    return _vehicle_wrapper(fund_name) == "vit"


def _is_etf_fund(fund_name: str) -> bool:
    return _vehicle_wrapper(fund_name) == "etf"


def _is_closed_end_fund(fund_name: str) -> bool:
    return _vehicle_wrapper(fund_name) == "closed_end_fund"


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

        self.gics_mapping = gics_mapping.copy() if isinstance(gics_mapping, pd.DataFrame) else pd.DataFrame()
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

    def _is_vit_fund(self, fund_name: str) -> bool:
        return _is_vit_fund(fund_name)

    def _is_etf_fund(self, fund_name: str) -> bool:
        return _is_etf_fund(fund_name)

    def _is_closed_end_fund(self, fund_name: str) -> bool:
        return _is_closed_end_fund(fund_name)

    # ------------------------------------------------------------------
    @staticmethod
    def _format_share_label(column: Optional[str]) -> str:
        if not column:
            return "Shares"
        label = str(column).strip()
        if not label:
            return "Shares"
        title = label.replace("_", " ").title()
        return (
            title.replace("Nav", "NAV")
            .replace("Iiv", "IIV")
            .replace("Dan", "DAN")
        )

    @staticmethod
    def _extract_share_value(entry: Mapping[str, Any]) -> float:
        if not isinstance(entry, Mapping):
            return 0.0
        for key in ("nav_shares", "iiv_shares", "shares", "quantity", "units"):
            if key in entry:
                try:
                    value = entry.get(key)
                    return float(value or 0.0)
                except (TypeError, ValueError):
                    continue
        return 0.0

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
                summary = dict(data.get("summary_metrics") or {})
                fund_totals = dict(data.get("fund_current_totals") or {})

                if not summary and not fund_totals:
                    continue

                combined = {**fund_totals, **summary}

                rows.append(
                    {
                        "Date": date_str,
                        "Fund": fund_name,
                        "Cash": combined.get("cash_value", 0.0),
                        "Treasury": combined.get("treasury_market_value", 0.0),  # CHANGED
                        "Equity": combined.get("equity_market_value", 0.0),
                        "Option DAN": combined.get("option_delta_adjusted_notional", 0.0),
                        "Option MV": combined.get("option_market_value", 0.0),
                        "Total Assets": combined.get("total_assets", 0.0),
                        "Total Net Assets": combined.get("total_net_assets", 0.0),
                    }
                )

        columns = [
            "Date",
            "Fund",
            "Cash",
            "Treasury",
            "Equity",
            "Option DAN",
            "Regular Option MV",
            "Flex Option MV",
            "Total Option MV",
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
        gics_results: list[Dict[str, object]] = []
        gics_calculations: list[Dict[str, object]] = []
        mapping_df = (
            self.gics_mapping.copy()
            if isinstance(self.gics_mapping, pd.DataFrame)
            else pd.DataFrame()
        )
        if not mapping_df.empty:
            mapping_df = mapping_df.map(
                lambda value: str(value).strip() if value is not None else ""
            )

        # Precompute industry-name -> sector-name lookup once, instead of
        # filtering mapping_df per (fund, gics_class, category) inside the loop.
        sector_lookup: Dict[str, Dict[str, str]] = {}
        if not mapping_df.empty and "GICS_SECTOR_NAME" in mapping_df.columns:
            for gics_class in ("GICS_INDUSTRY_GROUP_NAME", "GICS_INDUSTRY_NAME"):
                if gics_class not in mapping_df.columns:
                    continue
                table: Dict[str, str] = {}
                for category, sector in zip(mapping_df[gics_class], mapping_df["GICS_SECTOR_NAME"]):
                    key = str(category).strip().upper()
                    if not key:
                        continue
                    table.setdefault(key, str(sector).strip() or "N/A")
                sector_lookup[gics_class] = table

        summary_columns = [
            "Date",
            "Fund",
            "Overall GICS Compliance",
            "Industry Exceeds 25%",
            "Industry Group Exceeds 25%",
            "Index Industry Exceeds 25%?",
            "Index Industry Group Exceeds 25%?",
            "Can Fund exceed 25% if Index does?",
            "Exceptions to Conc Policy",
        ]
        detail_value_columns = [
            "Date",
            "Fund",
            "GICS_Class",
            "GICS_Class Value",
            "GICS_SECTOR_NAME",
            "Fund_Weight_Value",
            "Index_Weight_Value",
            "Exceeding_Fund_25%?",
            "Exceeding_Index_25%?",
        ]

        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()

            for fund_name, compliance_dict in funds.items():
                gics_data = compliance_dict.get("gics_compliance", {})

                if not isinstance(gics_data, Mapping):
                    continue

                can_exceed = "Yes" if fund_name in {"KNG", "FDND", "KNGIX"} else "No"
                exceptions = "Information Technology Sector" if fund_name == "TDVI" else ""

                exceeding_index = gics_data.get("exceeding_index_gics", {}) or {}
                gics_results.append(
                    {
                        "Date": report_date,
                        "Fund": fund_name,
                        "Overall GICS Compliance": gics_data.get(
                            "overall_gics_compliance", "FAIL"
                        ),
                        "Industry Exceeds 25%": not bool(
                            gics_data.get("industry_exceeds_25", True)
                        ),
                        "Industry Group Exceeds 25%": not bool(
                            gics_data.get("industry_group_exceeds_25", True)
                        ),
                        "Index Industry Exceeds 25%?": len(
                            (exceeding_index.get("GICS_INDUSTRY_NAME") or {})
                        )
                        > 0,
                        "Index Industry Group Exceeds 25%?": len(
                            (exceeding_index.get("GICS_INDUSTRY_GROUP_NAME") or {})
                        )
                        > 0,
                        "Can Fund exceed 25% if Index does?": can_exceed,
                        "Exceptions to Conc Policy": exceptions,
                    }
                )

                calculations = gics_data.get("calculations", {}) or {}
                for gics_class, calc in calculations.items():
                    if gics_class not in {"GICS_INDUSTRY_GROUP_NAME", "GICS_INDUSTRY_NAME"}:
                        continue
                    calc = calc or {}
                    fund_weights = calc.get("fund_weights", {}) or {}
                    index_weights = calc.get("index_weights", {}) or {}
                    exceeding_fund = calc.get("exceeding_fund", {}) or {}
                    exceeding_index_vals = calc.get("exceeding_index", {}) or {}

                    for category, fund_wt in fund_weights.items():
                        try:
                            fund_weight_value = float(fund_wt)
                        except (TypeError, ValueError):
                            fund_weight_value = 0.0
                        try:
                            index_weight_value = float(
                                index_weights.get(category, 0.0) or 0.0
                            )
                        except (TypeError, ValueError):
                            index_weight_value = 0.0

                        sector_name = sector_lookup.get(gics_class, {}).get(
                            str(category).strip().upper(), "N/A"
                        )

                        gics_calculations.append(
                            {
                                "Date": report_date,
                                "Fund": fund_name,
                                "GICS_Class": gics_class,
                                "GICS_Class Value": category,
                                "GICS_SECTOR_NAME": sector_name,
                                "Fund_Weight_Value": fund_weight_value,
                                "Index_Weight_Value": index_weight_value,
                                "Exceeding_Fund_25%?": True if category in exceeding_fund else None,
                                "Exceeding_Index_25%?": True if category in exceeding_index_vals else None,
                            }
                        )

        df_summary = pd.DataFrame(gics_results, columns=summary_columns)
        if not df_summary.empty:
            df_summary.sort_values(by=["Fund"], inplace=True)
            footnotes = [
                {"Date": f"* {note}"}
                for note in FOOTNOTES.get("gics", [])
            ]
            if footnotes:
                fn_df = pd.DataFrame(footnotes)
                df_summary = pd.concat([df_summary, fn_df], ignore_index=True)
        else:
            df_summary = pd.DataFrame(columns=summary_columns)

        df_details = pd.DataFrame(gics_calculations, columns=detail_value_columns)
        if not df_details.empty:
            df_details.sort_values(
                by=["Fund", "Fund_Weight_Value"], ascending=[True, False], inplace=True
            )
            df_details["Fund_Weight"] = df_details["Fund_Weight_Value"].apply(
                lambda value: f"{value:.2%}"
            )
            df_details["Index_Weight"] = df_details["Index_Weight_Value"].apply(
                lambda value: f"{value:.2%}"
            )
            df_details.drop(
                columns=["Fund_Weight_Value", "Index_Weight_Value"], inplace=True
            )
        else:
            df_details = pd.DataFrame(
                columns=[
                    "Date",
                    "Fund",
                    "GICS_Class",
                    "GICS_Class Value",
                    "GICS_SECTOR_NAME",
                    "Fund_Weight",
                    "Index_Weight",
                    "Exceeding_Fund_25%?",
                    "Exceeding_Index_25%?",
                ]
            )

        self.sheet_data["GICS_Compliance"] = df_summary
        include_details = self.test_functions is None or len(self.test_functions) > 1
        if include_details and (not df_details.empty):
            self.sheet_data["GICS_Calculations"] = df_details


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

                if self._is_closed_end_fund(fund_name) and fund_name.startswith("TR"):
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

                share_column = calculations.get("share_column_used")
                share_label = self._format_share_label(share_column)

                issuer_limited = calculations.get("issuer_limited_securities", []) or []
                if issuer_limited:
                    issuer_limited_str = ", ".join(
                        "({ticker}, {shares:,.0f}, {value:,.2f})".format(
                            ticker=sec.get("eqyticker"),
                            shares=self._extract_share_value(sec),
                            value=float(sec.get("net_market_value", 0.0) or 0.0),
                        )
                        for sec in issuer_limited
                    )
                else:
                    issuer_limited_str = "None"

                excluded = calculations.get("excluded_securities", []) or []
                excluded_str = ", ".join(excluded) if excluded else "None"

                remaining_details = calculations.get("remaining_stocks_details", []) or []
                if remaining_details:
                    remaining_str = ", ".join(
                        "({ticker}, {shares:,.0f}, {vest_weight:.2%}, {ownership:.7%})".format(
                            ticker=entry.get('eqyticker'),
                            shares=self._extract_share_value(entry),
                            vest_weight=float(entry.get('vest_weight', 0.0) or 0.0),
                            ownership=float(entry.get('vest_ownership_of_float', 0.0) or 0.0),
                        )
                        for entry in remaining_details
                    )
                else:
                    remaining_str = "None"

                remaining_label = (
                    f"Remaining Securities (Ticker, {share_label}, Vest Weight, Vest Ownership of Float)"
                )
                issuer_label = (
                    f"Issuer Limited Assets (Condition 2a) (Ticker, {share_label}, Mkt Value)"
                )

                max_ownership_raw = calculations.get("max_ownership_float", 0.0)
                try:
                    max_ownership_float = float(max_ownership_raw if max_ownership_raw is not None else 0.0)
                except (TypeError, ValueError):
                    max_ownership_float = 0.0
                if pd.isna(max_ownership_float):
                    max_ownership_float = 0.0
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
                        ("Fund Status Today", calculations.get("fund_status_today")),
                        ("Registration Match", f'=IF(C{excel_row}=D{excel_row},"PASS","FAIL")',),
                        ("Condition 40 Act 1", f'=IF(L{excel_row}>=0.75,"PASS","FAIL")'),
                        ("Condition 40 Act 2a", f'=IF(O{excel_row}=0,"PASS","FAIL")'),
                        ("Condition 40 Act 2b", f'=IF(VALUE(U{excel_row})="","",IF(VALUE(U{excel_row})<=0.1,"PASS","FAIL"))'),
                        ("Condition 2a OCC", f'=IF(V{excel_row}=0,"PASS",IF(V{excel_row}<=0.05*M{excel_row},"PASS","FAIL"))'),
                        ("Total Assets", float(calculations.get("total_assets", 0.0) or 0.0)),
                        ("Non-Qualifying Assets Weight", float(calculations.get("non_qualifying_assets_1_wgt", 0.0) or 0.0)),
                        ("Qualifying Assets Pct", float(calculations.get("qualifying_assets_pct", 0.0) or 0.0)),
                        ("Net Assets", float(calculations.get("net_assets", 0.0) or 0.0)),
                        ("Expenses", abs(float(calculations.get("expenses", 0.0) or 0.0))),
                        ("Issuer Limited Assets (Condition 2a) Sum", float(calculations.get("issuer_limited_sum", 0.0) or 0.0),),
                        (issuer_label, issuer_limited_str,),
                        ("Excluded Securities (25% Bucket)", excluded_str),
                        ("Cumulative Weight of Excluded Securities", float(calculations.get("cumulative_weight_excluded", 0.0) or 0.0),),
                        ("Cumulative Weight of Remaining Securities", float(calculations.get("cumulative_weight_remaining", 0.0) or 0.0),),
                        (remaining_label, remaining_str,),
                        ("Max Ownership % of Float in 75% bucket", f"{max_ownership_float:.5%}"),
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
                        f"({sec.get('eqyticker', '')}, {float(sec.get('tna_wgt', 0) or 0):.2%})"
                        for sec in large_securities
                    )
                else:
                    large_securities_str = "None"

                overlap_records = calculations.get("overlap", []) or []
                overlap_weight_total = float(calculations.get("overlap_weight_sum", 0.0) or 0.0)

                if overlap_records:
                    overlap_summary = ", ".join(
                        f"{item.get('security_ticker', '')}: {float(item.get('security_weight', 0) or 0):.2%}"
                        for item in overlap_records
                    )
                    if not overlap_weight_total:
                        overlap_weight_total = sum(
                            float(item.get("security_weight", 0.0) or 0.0)
                            for item in overlap_records
                        )
                else:
                    overlap_summary = "None"
                if pd.isna(overlap_weight_total):
                    overlap_weight_total = 0.0

                largest = calculations.get("bottom_50_largest", {}) or {}
                second = calculations.get("bottom_50_second_largest", {}) or {}

                row_data = OrderedDict(
                    [
                        ("Date", report_date),
                        ("Fund", fund_name),
                        ("condition_IRS_1", "PASS" if irs.get("condition_IRS_1") else "FAIL"),
                        (
                            "condition_IRS_2_a_50",
                            f'=IF((I{excel_row}+S{excel_row})/H{excel_row}>=0.5,"PASS","FAIL")',
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
                            "5% Gross Assets Threshold ($)",
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
                        ("overlap_weight", float(overlap_weight_total)),
                        ("overlap_constituents", overlap_summary),
                        (
                            "bottom_50_largest_holding",
                            f"{largest.get('eqyticker', 'N/A')} ({float(largest.get('tna_wgt', 0) or 0):.2%})",
                        ),
                        (
                            "bottom_50_second_largest_holding",
                            f"{second.get('eqyticker', 'N/A')} ({float(second.get('tna_wgt', 0) or 0):.2%})",
                        ),
                        ("total_tbill_value", float(calculations.get("total_tbill_value", 0.0) or 0.0)),
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
                if not self._is_vit_fund(fund_name):
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
                        f"{holding.get('ticker')} ({(holding.get('ownership_pct') or 0):.2%})"
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
                    f"{h.get('eqyticker') or h.get('ticker')} ({(h.get('ownership_pct') or 0)*100:.5f}%)" for h in holdings
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
                        f"{biz.get('eqyticker')} ({(biz.get('ownership_pct') or 0) * 100:.5f}%)"
                        for biz in sec_related
                    )
                else:
                    sec_related_str = "None"

                combined = calculations.get("combined_holdings", []) or []
                if combined:
                    combined_str = ", ".join(
                        f"{holding.get('eqyticker')} ({(holding.get('vest_weight') or 0) * 100:.2f}%)"
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
                        # (
                        #     "Rule 3 OCC (<=5% TNA)",
                        #     f'=IF(N{excel_row}<=0.05,"PASS","FAIL")',
                        # ),
                        (
                            "12d3 Sec Biz Compliant",
                            f'=IF(AND(C{excel_row}="PASS",D{excel_row}="PASS",E{excel_row}="PASS"),"PASS","FAIL")',
                        ),
                        ("SEC-Related Businesses (Shs Out %)", sec_related_str),
                        ("Combined Holdings (Portfolio %)", combined_str),
                        ("Total Assets", float(calculations.get("total_assets", 0.0) or 0.0)),
                        ("Max Ownership %", f"{float(calculations.get('max_ownership_pct', 0.0) or 0.0):.8%}"),
                        ("Max Weight", float(calculations.get("max_weight", 0.0) or 0.0)),
                        # (
                        #     "OCC Market Value",
                        #     float(calculations.get("occ_weight_mkt_val", 0.0) or 0.0),
                        # ),
                        # ("OCC Weight", float(calculations.get("occ_weight", 0.0) or 0.0)),
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

                if not self._is_etf_fund(fund_name):
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
                        # (
                        #     "Equity Holdings 85% Compliance",
                        #     f'=IF(F{excel_row}>=0.85,"PASS","FAIL")',
                        # ),
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
                    is_formula_percent = header_text in {"formula", "formula (mv)"}
                    for cell in column_cells:
                        try:
                            value = cell.value
                            if value is not None:
                                max_length = max(max_length, len(str(value)))
                                if cell.row > 1 and cell.data_type in {"n", "f"}:
                                    if is_percent or is_formula_percent:
                                        cell.number_format = "0.00%"
                                    else:
                                        cell.number_format = "#,##0.00"
                        except Exception:  # pragma: no cover - defensive
                            continue
                    ws.column_dimensions[column_letter].width = min(max_length + 2, 50)


class ComplianceReportPDF(BaseReportPDF):
    """Render compliance results to a PDF document."""

    def __init__(
        self,
        results: Mapping[str, Mapping[str, object]],
        output_path: str,
        *,
        allowed_tests: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(output_path, orientation="L")
        self.results = flatten_compliance_results(results)
        self.allowed_tests = (
            {test for test in allowed_tests if test}
            if allowed_tests is not None
            else None
        )
        self.generate_pdf()
        self.save_pdf()

    @staticmethod
    def _get_vehicle_wrapper(fund_name: str) -> str:
        return _vehicle_wrapper(fund_name)

    def save_pdf(self) -> None:
        """Save the PDF to the specified output path."""
        try:
            # Ensure the output directory exists
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save the PDF file
            self.pdf.output(self.output_path)
        except Exception as e:
            logger.error(f"Failed to save PDF to {self.output_path}: {e}")
            raise

    # ------------------------------------------------------------------
    def generate_pdf(self) -> None:
        """Generate the PDF with all compliance metrics - CORRECTED VERSION"""
        grouped_results = self._group_results_by_fund()
        if not grouped_results:
            if self.pdf.page_no() == 0:
                self.pdf.add_page()
            else:
                self.pdf.set_y(self.pdf.t_margin)

            # Even with no grouped results, check if there's GICS data in self.results
            if self._is_test_selected("gics_compliance") and self.results:
                self._render_gics_compliance_overview()

            self._add_header("Compliance Overview")
            self.pdf.set_font("Arial", "", 9)
            self.pdf.cell(0, 6, "No compliance results available.", ln=True)
            self.output()
            return

        self.grouped_results = grouped_results
        self.fund_order = sorted(grouped_results)

        if self.pdf.page_no() == 0:
            self.pdf.add_page()
        else:
            self.pdf.set_y(self.pdf.t_margin)

        summary_metrics = [
            ("Cash", lambda data: self._format_currency(self._get_summary_value(data, "cash_value"))),
            ("Treasury", lambda data: self._format_currency(self._get_summary_value(data, "treasury_market_value"))),  # CHANGED
            ("Equity", lambda data: self._format_currency(self._get_summary_value(data, "equity_market_value"))),
            ("Option DAN", lambda data: self._format_currency(self._get_summary_value(data, "option_delta_adjusted_notional"))),
            ("Option MV", lambda data: self._format_currency(self._get_summary_value(data, "option_market_value"))),
            ("Total Assets", lambda data: self._format_currency(self._get_summary_value(data, "total_assets"))),
            ("Total Net Assets", lambda data: self._format_currency(self._get_summary_value(data, "total_net_assets"))),
        ]

        prospectus_metrics = [
            ("Status", lambda data: self._status_text(self._get_test_payload(data, "prospectus_80pct_policy").get("is_compliant"))),
            (
                "Result (DAN)",
                lambda data: self._format_percent(
                    self._get_calculation(self._get_test_payload(data, "prospectus_80pct_policy"), "names_test")
                ),
            ),
            (
                "Result (MV)",
                lambda data: self._format_percent(
                    self._get_calculation(self._get_test_payload(data, "prospectus_80pct_policy"), "names_test_mv")
                ),
            ),
            (
                "Options In Scope",
                lambda data: self._yes_no(
                    self._get_calculation(self._get_test_payload(data, "prospectus_80pct_policy"), "options_in_scope")
                ),
            ),
            (
                "Option Contribution",
                lambda data: self._format_currency(
                    self._get_calculation(self._get_test_payload(data, "prospectus_80pct_policy"), "option_contribution")
                ),
            ),
        ]

        forty_act_metrics = [
            (
                "Overall Status",
                lambda data: self._status_or_skip(data, "diversification_40act_check"),
            ),
            (
                "Registered Status",
                lambda data: self._title_case(
                    self._get_calculation(
                        self._get_test_payload(data, "diversification_40act_check"), "fund_registration"
                    )
                ),
            ),
            (
                "Status Today",
                lambda data: self._title_case(
                    self._get_calculation(
                        self._get_test_payload(data, "diversification_40act_check"), "fund_status_today"
                    )
                ),
            ),
            (
                "Registration Match",
                lambda data: self._status_text(
                    self._registrations_match(
                        self._get_test_payload(data, "diversification_40act_check")
                    )
                ),
            ),
            (
                "Condition 1",
                lambda data: self._status_text(
                    self._get_detail(
                        self._get_test_payload(data, "diversification_40act_check"),
                        "condition_40act_1",
                    )
                ),
            ),
            (
                "Condition 2a",
                lambda data: self._status_text(
                    self._get_detail(
                        self._get_test_payload(data, "diversification_40act_check"),
                        "condition_40act_2a",
                    )
                ),
            ),
            (
                "Condition 2b",
                lambda data: self._status_text(
                    self._get_detail(
                        self._get_test_payload(data, "diversification_40act_check"),
                        "condition_40act_2b",
                    )
                ),
            ),
            (
                "Condition 2a OCC",
                lambda data: self._status_text(
                    self._get_detail(
                        self._get_test_payload(data, "diversification_40act_check"),
                        "condition_40act_2a_occ",
                    )
                ),
            ),
            (
                "Max Ownership %",
                lambda data: self._format_percent(
                    self._get_calculation(
                        self._get_test_payload(data, "diversification_40act_check"),
                        "max_ownership_float",
                    )
                ),
            ),
        ]

        irs_metrics = [
            (
                "Overall Status",
                lambda data: self._status_or_skip(data, "diversification_IRS_check"),
            ),
            (
                "Condition 1",
                lambda data: self._status_text(
                    self._get_detail(self._get_test_payload(data, "diversification_IRS_check"), "condition_IRS_1")
                ),
            ),
            (
                "Condition 2a 50%",
                lambda data: self._status_text(
                    self._get_detail(self._get_test_payload(data, "diversification_IRS_check"), "condition_IRS_2_a_50")
                ),
            ),
            (
                "Condition 2a 5%",
                lambda data: self._status_text(
                    self._get_detail(self._get_test_payload(data, "diversification_IRS_check"), "condition_IRS_2_a_5")
                ),
            ),
            (
                "Condition 2a 10%",
                lambda data: self._status_text(
                    self._get_detail(self._get_test_payload(data, "diversification_IRS_check"), "condition_IRS_2_a_10")
                ),
            ),
            (
                "Condition 2b",
                lambda data: self._status_text(
                    self._get_detail(self._get_test_payload(data, "diversification_IRS_check"), "condition_IRS_2_b")
                ),
            ),
            (
                "Sum Large Securities %",
                lambda data: self._format_percent(
                    self._get_calculation(
                        self._get_test_payload(data, "diversification_IRS_check"),
                        "sum_large_securities_weights",
                    )
                ),
            ),
            (
                "Large Securities Count",
                lambda data: self._format_integer(
                    self._get_calculation(
                        self._get_test_payload(data, "diversification_IRS_check"),
                        "large_securities_count",
                    )
                ),
            ),
        ]

        # FIXED: IRC metrics now use helper methods to handle N/A for non-VIT funds
        irc_metrics = [
            (
                "Overall Status",
                lambda data: self._check_irc_status(data)
            ),
            (
                "Condition 55",
                lambda data: self._check_irc_condition(data, "condition_IRC_55")
            ),
            (
                "Condition 70",
                lambda data: self._check_irc_condition(data, "condition_IRC_70")
            ),
            (
                "Condition 80",
                lambda data: self._check_irc_condition(data, "condition_IRC_80")
            ),
            (
                "Condition 90",
                lambda data: self._check_irc_condition(data, "condition_IRC_90")
            ),
            (
                "Top 1 Exposure",
                lambda data: self._format_irc_exposure(data, "top_1")
            ),
            (
                "Top 4 Exposure",
                lambda data: self._format_irc_exposure(data, "top_4")
            ),
        ]

        illiquid_metrics = [
            (
                "Status",
                lambda data: self._status_text(
                    self._get_test_payload(data, "max_15pct_illiquid_sai").get("is_compliant")
                ),
            ),
            (
                "Illiquid Exposure %",
                lambda data: self._format_percent(
                    self._get_calculation(self._get_test_payload(data, "max_15pct_illiquid_sai"), "illiquid_percentage")
                ),
            ),
            (
                "Total Illiquid Value",
                lambda data: self._format_currency(
                    self._get_calculation(self._get_test_payload(data, "max_15pct_illiquid_sai"), "total_illiquid_value")
                ),
            ),
        ]

        # FIXED: Real estate metrics now include status
        real_estate_metrics = [
            (
                "Status",
                lambda data: self._status_text(
                    self._get_test_payload(data, "real_estate_check").get("is_compliant")
                ),
            ),
            (
                "Real Estate Exposure",
                lambda data: self._format_percent(
                    self._get_calculation(self._get_test_payload(data, "real_estate_check"), "real_estate_percentage")
                ),
            ),
        ]

        # FIXED: Commodities metrics now include status
        commodities_metrics = [
            (
                "Status",
                lambda data: self._status_text(
                    self._get_test_payload(data, "commodities_check").get("is_compliant")
                ),
            ),
            (
                "Commodities Exposure",
                lambda data: self._format_percent(
                    self._first_not_none(
                        self._get_calculation(
                            self._get_test_payload(data, "commodities_check"),
                            "commodities_percentage",
                        ),
                        self._get_calculation(
                            self._get_test_payload(data, "commodities_check"),
                            "commodities_exposure",
                        ),
                    )
                ),
            ),
        ]

        rule_12d1_metrics = [
            (
                "Status",
                lambda data: self._status_text(
                    self._get_test_payload(data, "twelve_d1a_other_inv_cos").get("is_compliant")
                ),
            ),
            (
                "Test 1",
                lambda data: self._status_text(
                    self._get_detail(self._get_test_payload(data, "twelve_d1a_other_inv_cos"), "test_1_pass")
                ),
            ),
            (
                "Test 2",
                lambda data: self._status_text(
                    self._get_detail(self._get_test_payload(data, "twelve_d1a_other_inv_cos"), "test_2_pass")
                ),
            ),
            (
                "Test 3",
                lambda data: self._status_text(
                    self._get_detail(self._get_test_payload(data, "twelve_d1a_other_inv_cos"), "test_3_pass")
                ),
            ),
        ]

        rule_12d2_metrics = [
            (
                "Status",
                lambda data: self._status_text(
                    self._get_test_payload(data, "twelve_d2_insurance_cos").get("is_compliant")
                ),
            ),
            (
                "Max Ownership",
                lambda data: self._format_percent(
                    self._get_calculation(self._get_test_payload(data, "twelve_d2_insurance_cos"), "max_ownership_pct")
                ),
            ),
        ]

        # FIXED: Rule 12d3 metrics now show all three rules plus overall status
        rule_12d3_metrics = [
            (
                "Overall Status",
                lambda data: self._status_text(
                    self._get_test_payload(data, "twelve_d3_sec_biz").get("is_compliant")
                ),
            ),
            (
                "Rule 1 (<=5%)",
                lambda data: self._status_text(
                    self._get_detail(self._get_test_payload(data, "twelve_d3_sec_biz"), "rule_1_pass")
                ),
            ),
            (
                "Rule 2 (<=10%)",
                lambda data: self._status_text(
                    self._get_detail(self._get_test_payload(data, "twelve_d3_sec_biz"), "rule_2_pass")
                ),
            ),
            (
                "Rule 3 (<=5%)",
                lambda data: self._status_text(
                    self._get_detail(self._get_test_payload(data, "twelve_d3_sec_biz"), "rule_3_pass")
                ),
            ),
        ]

        sections = [
            ("Summary Metrics", summary_metrics, None, None, ("summary_metrics",)),
            (
                "Prospectus 80% Policy",
                prospectus_metrics,
                "prospectus_80pct",
                None,
                ("prospectus_80pct_policy",),
            ),
            (
                "40 Act Diversification",
                forty_act_metrics,
                "40act",
                {"Registration Match", "Condition 2a OCC"},
                ("diversification_40act_check",),
            ),
            (
                "IRS Diversification",
                irs_metrics,
                "irs",
                None,
                ("diversification_IRS_check",),
            ),
            (
                "IRC Diversification",
                irc_metrics,
                "irc",
                None,
                ("diversification_IRC_check",),
            ),
            (
                "15% Illiquid Assets",
                illiquid_metrics,
                "illiquid",
                None,
                ("max_15pct_illiquid_sai",),
            ),
            (
                "Real Estate",
                real_estate_metrics,
                "real_estate",
                None,
                ("real_estate_check",),
            ),
            (
                "Commodities",
                commodities_metrics,
                "commodities",
                None,
                ("commodities_check",),
            ),
            (
                "Rule 12d1-1",
                rule_12d1_metrics,
                "12d1",
                None,
                ("twelve_d1a_other_inv_cos",),
            ),
            (
                "Rule 12d2",
                rule_12d2_metrics,
                "12d2",
                None,
                ("twelve_d2_insurance_cos",),
            ),
            (
                "Rule 12d3",
                rule_12d3_metrics,
                "12d3",
                None,
                ("twelve_d3_sec_biz",),
            ),
        ]

        for idx, (title, metrics, footnote, skip_labels, keys) in enumerate(sections):
            if not any(self._is_test_selected(key) for key in keys):
                continue

            self._render_metric_section(
                title,
                metrics,
                footnote_key=footnote,
                skip_fail_highlight_labels=skip_labels,
            )

            # Inject GICS overview immediately AFTER Summary Metrics
            if idx == 0 and self._is_test_selected("gics_compliance"):
                has_gics_data = any(
                    isinstance(payload, dict) and "gics_compliance" in payload
                    for (fund, date), payload in self.results.items()
                )
                if has_gics_data:
                    # Space before GICS block
                    self.pdf.ln(4)
                    self._print_test_header("GICS Diversification")
                    self._render_gics_compliance_overview()
                    # Space after GICS block
                    self.pdf.ln(8)

    # ------------------------------------------------------------------
    def _group_results_by_fund(self) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for (fund_name, date_str), payload in sorted(
            self.results.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            grouped[fund_name] = payload
        return grouped

    def _render_metric_section(
        self,
        title: str,
        metrics: Iterable[Tuple[str, Any]],
        *,
        footnote_key: Optional[str] = None,
        skip_fail_highlight_labels: Optional[Iterable[str]] = None,
    ) -> None:
        metrics = list(metrics)
        if not metrics:
            return

        rows: list[list[str]] = []
        for label, extractor in metrics:
            row: list[str] = [self._sanitize_text(label)]
            for fund in self.fund_order:
                fund_payload = self.grouped_results.get(fund, {})
                try:
                    value = extractor(fund_payload)
                except Exception:
                    value = ""
                row.append(self._sanitize_text(str(value or "")))
            rows.append(row)

        estimated_height = self._estimate_table_height(len(rows))
        if self.pdf.get_y() + estimated_height > self.pdf.h - self.pdf.b_margin:
            self.pdf.add_page()

        self._print_test_header(title)
        headers = ["Metric", *self.fund_order]
        self._draw_metric_table(
            headers,
            rows,
            skip_fail_labels={
                self._sanitize_text(label)
                for label in (skip_fail_highlight_labels or [])
                if label
            },
        )
        if footnote_key:
            self._draw_footnotes_section(footnote_key)
        self.pdf.ln(2)

    def _estimate_table_height(self, row_count: int) -> float:
        header_height = 5.0
        row_height = 4.2
        return header_height + (row_height * max(row_count, 0)) + 6


    # ------------------------------------------------------------------
    def _add_header(self, title: str) -> None:
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, self._sanitize_text(title), ln=True, align="C")
        self.pdf.ln(4)

    def _print_test_header(self, title: str) -> None:
        self.pdf.set_font("Arial", "B", 10)
        self.pdf.cell(0, 6, self._sanitize_text(title), ln=True)
        self.pdf.set_font("Arial", "", 9)


    def _draw_metric_table(
        self,
        headers: Iterable[str],
        rows: Iterable[Iterable[str]],
        *,
        skip_fail_labels: Optional[Iterable[str]] = None,
    ) -> None:
        headers = list(headers)
        rows = [list(row) for row in rows]
        if not headers or not rows:
            return

        skip_fail = {
            self._sanitize_text(label).lower()
            for label in (skip_fail_labels or [])
            if label
        }

        usable_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin
        metric_width = min(45.0, usable_width * 0.3)
        value_cols = max(len(headers) - 1, 1)
        value_width = (usable_width - metric_width) / value_cols if value_cols else usable_width
        widths = [metric_width] + [value_width] * (len(headers) - 1)

        header_height = 5.0
        row_height = 4.2

        self.pdf.set_font("Arial", "B", 7)
        self.pdf.set_fill_color(230, 230, 230)
        for width, header in zip(widths, headers):
            self.pdf.cell(width, header_height, self._sanitize_text(header), border=1, align="C", fill=True)
        self.pdf.ln(header_height)

        self.pdf.set_font("Arial", "", 6)
        for row in rows:
            row_label = self._sanitize_text(row[0]).lower() if row else ""
            for idx, value in enumerate(row):
                text = self._sanitize_text(value).replace("\n", " /")
                width = widths[idx]
                if text.upper() == "FAIL" and row_label not in skip_fail:
                    self.pdf.set_fill_color(255, 200, 200)
                    self.pdf.set_text_color(139, 0, 0)
                    self.pdf.cell(width, row_height, text, border=1, align="C", fill=True)
                    self.pdf.set_text_color(0, 0, 0)
                else:
                    self.pdf.set_fill_color(255, 255, 255)
                    align = "L" if idx == 0 else ("R" if self._looks_like_number(text) else "L")
                    self.pdf.cell(width, row_height, text, border=1, align=align)
            self.pdf.ln(row_height)

    @staticmethod
    def _looks_like_number(value: str) -> bool:
        stripped = value.replace(",", "").replace("%", "").strip()
        if not stripped:
            return False
        try:
            float(stripped)
            return True
        except ValueError:
            return False

    def _get_summary_value(self, fund_data: Mapping[str, Any], key: str) -> object:
        """
        Retrieve summary metric value from summary_metrics test results.

        Returns the value for the given key, or 0.0 if not found.
        """
        if not isinstance(fund_data, Mapping):
            return 0.0

        # Get the summary_metrics result (it's a flat dict)
        summary_result = fund_data.get("summary_metrics")

        if not isinstance(summary_result, Mapping):
            return 0.0

        # Get the value directly from the flat dict
        return summary_result.get(key, 0.0)

    def _is_test_selected(self, test_name: str) -> bool:
        if test_name == "diversification_IRC_check":
            has_vit_wrapper = any(
                self._get_vehicle_wrapper(fund_name) == "vit"
                for funds in self.results.values()
                for fund_name in funds.keys()
            )
            if not has_vit_wrapper:
                return False
        if not self.allowed_tests:
            return True
        return test_name in self.allowed_tests

    def _is_test_skipped(self, fund_data: Mapping[str, Any], key: str) -> bool:
        """True if the test was deliberately skipped (e.g. private fund + 40 Act / IRS)."""
        payload = self._get_test_payload(fund_data, key)
        details = payload.get("details") if isinstance(payload, Mapping) else None
        if not isinstance(details, Mapping):
            return False
        return bool(details.get("skipped") or details.get("skip"))

    def _status_or_skip(self, fund_data: Mapping[str, Any], key: str) -> str:
        """Render N/A for skipped tests, otherwise PASS/FAIL from is_compliant."""
        if self._is_test_skipped(fund_data, key):
            return "N/A"
        payload = self._get_test_payload(fund_data, key)
        return self._status_text(payload.get("is_compliant"))

    def _get_test_payload(self, fund_data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        if not isinstance(fund_data, Mapping):
            return {}
        payload = fund_data.get(key, {})
        return payload if isinstance(payload, Mapping) else {}

    def _get_calculation(self, payload: Mapping[str, Any], key: str) -> object:
        return self._get_nested(payload, "calculations", key)

    def _get_detail(self, payload: Mapping[str, Any], key: str) -> object:
        value = self._get_nested(payload, "details", key)
        if value is None and isinstance(payload, Mapping):
            return payload.get(key)
        return value

    @staticmethod
    def _get_nested(payload: Mapping[str, Any], *keys: str) -> object:
        current: object = payload
        for key in keys:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
        return current

    @staticmethod
    def _title_case(value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return text.replace("_", " ").title()

    def _registrations_match(self, payload: Mapping[str, Any]) -> Optional[bool]:
        if not isinstance(payload, Mapping):
            return None
        calculations = payload.get("calculations", {})
        if not isinstance(calculations, Mapping):
            return None
        expected = calculations.get("fund_registration")
        actual = calculations.get("fund_status_today")
        if expected is None or actual is None:
            return None
        return str(expected).strip().lower() == str(actual).strip().lower()

    @staticmethod
    def _status_text(value: object) -> str:
        if isinstance(value, bool):
            return "PASS" if value else "FAIL"
        if isinstance(value, str):
            upper = value.strip().upper()
            if upper in {"PASS", "FAIL"}:
                return upper
        # Unknown / missing test result: surface it rather than silently passing.
        return ""

    @staticmethod
    def _yes_no(value: object) -> str:
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return ""

    def _format_currency(self, value: object) -> str:
        try:
            number = float(value or 0.0)
        except (TypeError, ValueError):
            return ""
        if pd.isna(number):
            return ""
        return f"{number:,.0f}"

    def _format_percent(self, value: object) -> str:
        try:
            number = float(value or 0.0)
        except (TypeError, ValueError):
            return ""
        if pd.isna(number):
            return ""
        return f"{number:.5%}"

    def _format_integer(self, value: object) -> str:
        try:
            number = int(float(value or 0.0))
        except (TypeError, ValueError):
            return ""
        return f"{number:d}"

    @staticmethod
    def _first_not_none(*values: object) -> object:
        for value in values:
            if value is not None:
                return value
        return None


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
        # Reset cursor X to the left margin before multi_cell(width=0). Previous
        # cell rendering can leave the X near the right margin, which makes
        # "available width" effectively zero and triggers fpdf2's
        # "Not enough horizontal space to render a single character".
        self.pdf.set_x(self.pdf.l_margin)
        self.pdf.multi_cell(0, 5, self._sanitize_text("Footnotes:"))
        for note in notes:
            self.pdf.set_x(self.pdf.l_margin)
            self.pdf.multi_cell(0, 4.5, self._sanitize_text(f"* {note}"))
        self.pdf.ln(2)

    # ------------------------------------------------------------------
    def _render_gics_compliance_overview(self) -> None:
        """Render the Excel-like GICS_Compliance summary as a wrapped PDF table (no calculations)."""
        headers = [
            "Date",
            "Fund",
            "Overall",  # was: Overall GICS Compliance
            "Ind >25%",  # was: Industry Exceeds 25%
            "IndGrp >25%",  # was: Industry Group Exceeds 25%
            "Idx Ind >25%",  # was: Index Industry Exceeds 25%?
            "Idx IndGrp >25%",  # was: Index Industry Group Exceeds 25%?
            "Fund>25% if Idx?",  # was: Can Fund exceed 25% if Index does?
            "Exceptions",  # was: Exceptions to Conc Policy
        ]

        rows: list[list[str]] = []
        for (fund_name, date_str), payload in sorted(
                self.results.items(), key=lambda kv: (kv[0][0], kv[0][1])
        ):
            gics = (payload or {}).get("gics_compliance")
            if not isinstance(gics, dict):
                continue

            industry_exceeds = not bool(gics.get("industry_exceeds_25", True))
            industry_group_exceeds = not bool(gics.get("industry_group_exceeds_25", True))

            exceeding_index = gics.get("exceeding_index_gics", {}) or {}
            index_industry_exceeds = len(exceeding_index.get("GICS_INDUSTRY_NAME", {}) or {}) > 0
            index_industry_group_exceeds = len(exceeding_index.get("GICS_INDUSTRY_GROUP_NAME", {}) or {}) > 0

            fund_upper = str(fund_name).upper()
            can_exceed = "Yes" if fund_upper in {"KNG", "FDND", "KNGIX"} else "No"
            exceptions = "Information Technology Sector" if fund_upper == "TDVI" else ""

            try:
                date_disp = datetime.fromisoformat(date_str).date()
            except Exception:
                date_disp = date_str

            rows.append([
                str(date_disp),
                str(fund_name),
                str(gics.get("overall_gics_compliance", "")),
                "Yes" if industry_exceeds else "No",
                "Yes" if industry_group_exceeds else "No",
                "Yes" if index_industry_exceeds else "No",
                "Yes" if index_industry_group_exceeds else "No",
                can_exceed,
                exceptions,
            ])

        if not rows:
            return

        # Measure text widths for headers and rows, then compute comfortable column widths.
        usable = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin

        # Measure with same fonts you'll use to draw the table
        self.pdf.set_font("Arial", "B", 7)
        hdr_w = [self.pdf.get_string_width(h) for h in headers]
        self.pdf.set_font("Arial", "", 7)
        row_w = [0.0] * len(headers)
        for r in rows:
            for i, val in enumerate(r):
                row_w[i] = max(row_w[i], self.pdf.get_string_width(str(val) if val is not None else ""))

        # Base width = max(header, row) + padding
        PADDING = 8.0
        MIN_W = 20.0
        col_widths = [max(MIN_W, hw, rw) + PADDING for hw, rw in zip(hdr_w, row_w)]

        total = sum(col_widths)
        if total > usable:
            # Two-pass shrink: reserve MIN_W per column first, then distribute the
            # remaining width proportionally to each column's *desired excess* over
            # MIN_W. A naive proportional scale would shrink narrow Yes/No columns
            # below the minimum glyph width and trigger fpdf2's
            # "Not enough horizontal space" error.
            n = len(col_widths)
            total_floor = MIN_W * n
            if total_floor >= usable:
                col_widths = [usable / n] * n
            else:
                extras = [max(w - MIN_W, 0.0) for w in col_widths]
                extras_sum = sum(extras)
                leftover = usable - total_floor
                if extras_sum > 0:
                    col_widths = [MIN_W + (e / extras_sum) * leftover for e in extras]
                else:
                    col_widths = [usable / n] * n
        else:
            # Extra room -> give remainder to the last column (Exceptions)
            col_widths[-1] += (usable - total)

        # Now draw as usual (no overlap)
        self._draw_table(headers, rows, col_widths=col_widths, row_height=6)
        self._draw_footnotes_section("gics")
        self.pdf.ln(4)

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

    pdf_file_path = None
    if create_pdf:
        try:
            pdf_file_path = str(Path(output_dir) / f"{file_name_prefix}_{report.report_date}.pdf")
            pdf_report = ComplianceReportPDF(
                report.results,
                pdf_file_path,
                allowed_tests=test_functions,
            )

        except RuntimeError as exc:  # pragma: no cover - optional dependency missing
            logger.warning("PDF generation skipped: %s", exc)
        except Exception as exc:
            # exc_info=True so the actual fpdf2 stack trace surfaces in the log;
            # otherwise "PDF generation failed: <msg>" hides where the layout broke.
            logger.error("PDF generation failed: %s", exc, exc_info=True)

    return GeneratedComplianceReport(excel_path=str(report.file_path), pdf_path=pdf_file_path)