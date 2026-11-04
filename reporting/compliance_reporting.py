"""Generate Excel and PDF summaries for compliance results."""

from __future__ import annotations
from functools import partial

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, NamedTuple, Optional

import numpy as np
import pandas as pd

from config.constants import (
    PROSPECTUS_MIN_THRESHOLD,
    RULE_12D1A_SINGLE_ASSETS_LIMIT,
    RULE_12D1A_TOTAL_ASSETS_LIMIT,
    RULE_12D3_ASSET_LIMIT,
    RULE_12D3_DEBT_LIMIT,
    RULE_12D3_EQUITY_LIMIT,
)
from config.fund_definitions import vehicle_wrapper as _vehicle_wrapper
from reporting.base_report_pdf import BaseReportPDF, draw_header_row
from reporting.footnotes import FOOTNOTES
from reporting.report_utils import (
    ensure_dir,
    flatten_compliance_results,
    normalize_compliance_results,
    normalize_report_date,
)
from utilities.logger import setup_logger
from utilities.numbers import float_or, num, resolve_number, to_float

logger = setup_logger("compliance_report", "logs/compliance.log")


def _is_vit_fund(fund_name: str) -> bool:
    return _vehicle_wrapper(fund_name) == "vit"


def _is_etf_fund(fund_name: str) -> bool:
    return _vehicle_wrapper(fund_name) == "etf"


def _is_closed_end_fund(fund_name: str) -> bool:
    return _vehicle_wrapper(fund_name) == "closed_end_fund"


_SUMMARY_FIELDS = (
    ("Cash", "cash_value"),
    ("Treasury", "treasury_market_value"),
    ("Equity", "equity_market_value"),
    ("Option DAN", "option_delta_adjusted_notional"),
    ("Option MV", "option_market_value"),
    ("Total Assets", "total_assets"),
    ("Total Net Assets", "total_net_assets"),
)
# The Summary sheet splits the PDF's (regular) "Option MV" into regular / flex / total.
_SUMMARY_SHEET_FIELDS = (
    *_SUMMARY_FIELDS[:4],
    ("Regular Option MV", "option_market_value"),
    ("Flex Option MV", "flex_option_market_value"),
    ("Total Option MV", "total_option_market_value"),
    *_SUMMARY_FIELDS[5:],
)
_SUMMARY_COLUMNS = ["Date", "Fund", *(column for column, _ in _SUMMARY_SHEET_FIELDS)]

_SUMMARY_ALIAS = "summary"  # processing.api accepts it for the summary_metrics step


def _selected_tests(test_functions: Optional[Iterable[str]]) -> Optional[set]:
    """Requested report steps with the "summary" alias mapped to summary_metrics; None = everything."""
    if test_functions is None:
        return None
    return {"summary_metrics" if test == _SUMMARY_ALIAS else test for test in test_functions if test}


_GICS_CAN_EXCEED = {"KNG", "FDND", "KNGIX"}
_GICS_EXCEPTIONS = {"TDVI": "Information Technology Sector"}


def _gics_flags(fund_key: str, gics: Mapping[str, Any]) -> tuple:
    """(industry, group, index industry, index group) flags, can-exceed and exceptions text."""
    exceeding_index = gics.get("exceeding_index_gics", {}) or {}
    return (
        bool(gics.get("industry_exceeds_25")),
        bool(gics.get("industry_group_exceeds_25")),
        len(exceeding_index.get("GICS_INDUSTRY_NAME") or {}) > 0,
        len(exceeding_index.get("GICS_INDUSTRY_GROUP_NAME") or {}) > 0,
        "Yes" if fund_key in _GICS_CAN_EXCEED else "No",
        _GICS_EXCEPTIONS.get(fund_key, ""),
    )


# ----------------------------------------------------------------------
# Excel per-rule rows. Formulas hard-code column letters: keep key order.
# ----------------------------------------------------------------------
def _pf(value: object) -> str:
    return "PASS" if value else "FAIL"


def _ownership_text(value: Any, spec: str) -> str:
    # The checker leaves ownership NaN where shares are held but shares outstanding is missing.
    if value is not None and pd.isna(value):
        return "N/A (no shares out)"
    return format(float(value or 0.0), spec)


def _na_status(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.startswith("N/A") else None


def _join(items: Any, fmt: Callable[[Any], str]) -> str:
    return ", ".join(fmt(item) for item in items) if items else "None"


def _format_share_label(column: Optional[str]) -> str:
    label = str(column).strip() if column else ""
    if not label:
        return "Shares"
    return label.replace("_", " ").title().replace("Nav", "NAV").replace("Iiv", "IIV").replace("Dan", "DAN")


def _extract_share_value(entry: Mapping[str, Any]) -> float:
    shares_keys = ("nav_shares", "iiv_shares", "shares", "quantity", "units")
    return resolve_number(entry, shares_keys) if isinstance(entry, Mapping) else 0.0


def _prospectus_row(report_date, fund_name, payload, calc, r):
    equity = calc.get("total_equity_market_value", 0.0) or 0.0
    option_dan = abs(calc.get("total_opt_delta_notional_value", 0.0) or 0.0)
    option_mv = calc.get("total_opt_market_value", 0.0) or 0.0
    tbill = calc.get("total_tbill_value", 0.0) or 0.0
    cash = calc.get("total_cash_value", 0.0) or 0.0
    in_scope = bool(payload.get("options_in_scope"))
    # The checker's ratios: options count on both sides only when in scope (closed-end funds use the
    # option MV as the primary amount). A skipped fund has none, so its ratio leaves options out.
    numerator = calc.get("numerator", equity + tbill)
    denominator = calc.get("denominator", equity + cash + tbill)
    numerator_mv = calc.get("numerator_mv", equity + tbill)
    denominator_mv = calc.get("denominator_mv", equity + cash + tbill)
    return OrderedDict(
        [
            ("Date", report_date),
            ("Fund", fund_name),
            ("Prospectus 80% Compliance", f'=IF(K{r}/J{r}>={PROSPECTUS_MIN_THRESHOLD:g},"PASS","FAIL")'),
            ("Total Equity Market Value", equity),
            ("Total Option Delta Notional Value (DAN)", option_dan),
            ("Total Option Market Value", option_mv),
            ("Total T-Bill Value", tbill),
            ("Total Cash Value", cash),
            ("Max(CCET - DAN, 0)", max((cash + tbill) - option_dan, 0.0)),
            ("Denominator", denominator),
            ("Numerator", numerator),
            ("Formula", f"=K{r}/J{r}"),
            ("Denominator (Market Value)", denominator_mv),
            ("Numerator (Market Value)", numerator_mv),
            ("Formula (MV)", f"=N{r}/M{r}"),
            ("Options in Scope for 80%?", "Yes" if in_scope else "No"),
        ]
    )


def _forty_act_row(report_date, fund_name, payload, calc, r):
    share_label = _format_share_label(calc.get("share_column_used"))
    issuer_limited_str = _join(
        calc.get("issuer_limited_securities", []) or [],
        lambda sec: f"({sec.get('eqyticker')}, {_extract_share_value(sec):,.0f}, {num(sec, 'net_market_value'):,.2f})",
    )
    excluded = calc.get("excluded_securities", []) or []
    excluded_str = ", ".join(excluded) if excluded else "None"
    remaining_str = _join(
        calc.get("remaining_stocks_details", []) or [],
        lambda e: (
            f"({e.get('eqyticker')}, {_extract_share_value(e):,.0f}, "
            f"{num(e, 'vest_weight'):.2%}, {_ownership_text(e.get('vest_ownership_of_float'), '.7%')})"
        ),
    )

    max_ownership = to_float(calc.get("max_ownership_float", 0.0))
    occ_market_value = num(calc, "occ_market_value")

    notes = ""
    if fund_name in {"FDND", "TDVI"} and payload.get("condition_40act_2a"):
        if calc.get("issuer_limited_sum", 0):
            notes = (
                "NOTE: Jeremy indicated this fund should FAIL condition 2a "
                "(has securities >5% in 75% bucket)."
            )

    return OrderedDict(
        [
            ("Date", report_date),
            ("Fund", fund_name),
            ("Fund Registration", calc.get("fund_registration")),
            ("Fund Status Today", calc.get("fund_status_today")),
            # The checker's rule: only a fund registered as diversified that is not operating as diversified fails.
            ("Registration Match", f'=IF(OR(D{r}="diversified",C{r}<>"diversified"),"PASS","FAIL")'),
            ("Condition 40 Act 1", f'=IF(L{r}>=0.75,"PASS","FAIL")'),
            ("Condition 40 Act 2a", f'=IF(O{r}=0,"PASS","FAIL")'),
            ("Condition 40 Act 2b", _na_status(payload.get("condition_40act_2b_status"))
             or f'=IF(VALUE(U{r})="","",IF(VALUE(U{r})<=0.1,"PASS","FAIL"))'),
            ("Condition 2a OCC", f'=IF(V{r}=0,"PASS",IF(V{r}<=0.05*M{r},"PASS","FAIL"))'),
            ("Total Assets", num(calc, "total_assets")),
            ("Non-Qualifying Assets Weight", num(calc, "non_qualifying_assets_1_wgt")),
            ("Qualifying Assets Pct", num(calc, "qualifying_assets_pct")),
            ("Net Assets", num(calc, "net_assets")),
            ("Expenses", abs(num(calc, "expenses"))),
            ("Issuer Limited Assets (Condition 2a) Sum", num(calc, "issuer_limited_sum")),
            (f"Issuer Limited Assets (Condition 2a) (Ticker, {share_label}, Mkt Value)", issuer_limited_str),
            ("Excluded Securities (25% Bucket)", excluded_str),
            ("Cumulative Weight of Excluded Securities", num(calc, "cumulative_weight_excluded")),
            ("Cumulative Weight of Remaining Securities", num(calc, "cumulative_weight_remaining")),
            (f"Remaining Securities (Ticker, {share_label}, Vest Weight, Vest Ownership of Float)", remaining_str),
            ("Max Ownership % of Float in 75% bucket", f"{max_ownership:.5%}"),
            ("OCC Market Value", occ_market_value),
            ("OCC Weight", num(calc, "occ_weight")),
            ("Notes", notes),
        ]
    )


def _irs_row(report_date, fund_name, payload, calc, r):
    large_securities_str = _join(
        calc.get("large_securities", []) or [],
        lambda sec: f"({sec.get('eqyticker', '')}, {num(sec, 'tna_wgt'):.2%})",
    )
    overlap_records = calc.get("overlap", []) or []
    overlap_weight_total = num(calc, "overlap_weight_sum")
    overlap_summary = _join(
        overlap_records,
        lambda item: f"{item.get('security_ticker', '')}: {num(item, 'security_weight'):.2%}",
    )
    if overlap_records and not overlap_weight_total:
        overlap_weight_total = sum(num(item, "security_weight") for item in overlap_records)
    if pd.isna(overlap_weight_total):
        overlap_weight_total = 0.0

    def holding(key):
        held = calc.get(key, {}) or {}
        return f"{held.get('eqyticker', 'N/A')} ({num(held, 'tna_wgt'):.2%})"

    return OrderedDict(
        [
            ("Date", report_date),
            ("Fund", fund_name),
            ("condition_IRS_1", _pf(payload.get("condition_IRS_1"))),
            # The checker's test: qualifying assets >= 50% of net assets (net assets is not on the sheet).
            ("condition_IRS_2_a_50", _pf(payload.get("condition_IRS_2_a_50"))),
            ("condition_IRS_2_a_5", f'=IF(L{r}<=0.5,"PASS","FAIL")'),
            ("condition_IRS_2_a_10", _na_status(payload.get("condition_IRS_2_a_10_status"))
             or _pf(payload.get("condition_IRS_2_a_10"))),
            ("condition_IRS_2_b", _pf(payload["condition_IRS_2_b"]) if "condition_IRS_2_b" in payload
             else "N/A (not evaluated)"),  # the checker has no 2b test
            ("total_assets", num(calc, "total_assets")),
            ("qualifying_assets_value", num(calc, "qualifying_assets_value")),
            ("expenses", abs(num(calc, "expenses"))),
            ("5% Gross Assets Threshold ($)", num(calc, "five_pct_gross_assets")),
            ("sum_large_securities_weights", num(calc, "sum_large_securities_weights")),
            ("large_securities_count", int(calc.get("large_securities_count", 0) or 0)),
            ("large_securities", large_securities_str),
            ("overlap_weight", float(overlap_weight_total)),
            ("overlap_constituents", overlap_summary),
            ("bottom_50_largest_holding", holding("bottom_50_largest")),
            ("bottom_50_second_largest_holding", holding("bottom_50_second_largest")),
            ("total_tbill_value", num(calc, "total_tbill_value")),
        ]
    )


def _irc_row(report_date, fund_name, payload, calc, r):
    row = {"Date": report_date, "Fund": fund_name}
    row.update((f"Condition {n}", _pf(payload.get(f"condition_IRC_{n}"))) for n in (55, 70, 80, 90))
    row.update((f"Top {n} Exposure", calc.get(f"top_{n}", 0.0)) for n in (1, 2, 3, 4))
    row["Total Assets"] = calc.get("total_assets", 0.0)
    return row


def _illiquid_row(report_date, fund_name, payload, calc, r):
    return OrderedDict(
        [
            ("Date", report_date),
            ("Fund", fund_name),
            ("Total Assets", num(calc, "total_assets")),
            ("Total Illiquid Value", num(calc, "total_illiquid_value")),
            ("Illiquid Percentage", num(calc, "illiquid_percentage")),
            ("Equity Holdings Percentage", num(calc, "equity_holdings_percentage")),
            ("Max 15% Illiquid Compliance", f'=IF(E{r}<=0.15,"PASS","FAIL")'),
        ]
    )


def _real_estate_row(report_date, fund_name, payload, calc, r):
    return {
        "Date": report_date,
        "Fund": fund_name,
        "Real Estate Exposure": calc.get("real_estate_percentage", 0.0),
        "Total Exposure": calc.get("total_exposure", 0.0),
    }


def _commodities_row(report_date, fund_name, payload, calc, r):
    exposure = calc.get("commodities_percentage")
    if exposure is None:
        exposure = calc.get("commodities_exposure", 0.0)
    return {"Date": report_date, "Fund": fund_name, "Commodities Exposure": exposure}


def _12d1_row(report_date, fund_name, payload, calc, r):
    holdings_str = _join(
        calc.get("investment_companies", []) or [],
        lambda h: f"{h.get('ticker')} ({_ownership_text(h.get('ownership_pct'), '.2%')})",
    )
    # With an N/A ownership test the AND formula would read FAIL, so write the checker's verdict.
    test_1_na = _na_status(payload.get("test_1_status"))
    return OrderedDict(
        [
            ("Date", report_date),
            ("Fund", fund_name),
            ("Total Assets", num(calc, "total_assets")),
            ("Investment Companies", holdings_str),
            ("Ownership % Max", num(calc, "ownership_pct_max")),
            ("Equity Market Value Sum", num(calc, "equity_market_value_sum")),
            ("Test 1 (<=3% Ownership)", test_1_na or f'=IF(E{r}<=0.03,"PASS","FAIL")'),
            # Test 2 is the largest single holding (K), Test 3 the sum (F), as in the checker; with no
            # total assets the formula falls back to the checker's verdict instead of #DIV/0!.
            ("Test 2 (<=5% Total Assets)", f'=IFERROR(IF(K{r}/C{r}<={RULE_12D1A_SINGLE_ASSETS_LIMIT:g},'
             f'"PASS","FAIL"),"{_pf(payload.get("test_2_pass"))}")'),
            ("Test 3 (<=10% Total Assets)", f'=IFERROR(IF(F{r}/C{r}<={RULE_12D1A_TOTAL_ASSETS_LIMIT:g},'
             f'"PASS","FAIL"),"{_pf(payload.get("test_3_pass"))}")'),
            ("12d1(a) Compliant", _pf(payload.get("twelve_d1a_other_inv_cos_compliant")) if test_1_na
             else f'=IF(AND(G{r}="PASS",H{r}="PASS",I{r}="PASS"),"PASS","FAIL")'),
            ("Equity Market Value Max", num(calc, "equity_market_value_max")),
        ]
    )


def _12d2_row(report_date, fund_name, payload, calc, r):
    holdings = calc.get("insurance_holdings", [])
    holdings_str = ", ".join(
        f"{h.get('eqyticker') or h.get('ticker')} ({_ownership_text(h.get('ownership_pct'), '.5%')})"
        for h in holdings
    ) or "None"
    return {
        "Date": report_date,
        "Fund": fund_name,
        "Compliant": _na_status(payload.get("test_status")) or _pf(payload.get("test_pass")),
        "Total Assets": calc.get("total_assets", 0.0),
        "Insurance Holdings": holdings_str,
    }


# The checker's 12d3 limits; the Excel columns and the PDF rows share these labels.
_12D3_RULE_LABELS = (
    f"Rule 1 (<={RULE_12D3_EQUITY_LIMIT:.0%} equities)",
    f"Rule 2 (<={RULE_12D3_DEBT_LIMIT:.0%} debt)",
    f"Rule 3 (<={RULE_12D3_ASSET_LIMIT:.0%} total assets)",
)


def _12d3_row(report_date, fund_name, payload, calc, r):
    sec_related_str = _join(
        calc.get("sec_related_businesses", []) or [],
        lambda biz: f"{biz.get('eqyticker')} ({_ownership_text(biz.get('ownership_pct'), '.5%')})",
    )
    combined_str = _join(
        calc.get("combined_holdings", []) or [],
        lambda h: f"{h.get('eqyticker')} ({(h.get('vest_weight') or 0) * 100:.2f}%)",
    )
    # With an N/A ownership test the AND formula would read FAIL, so write the checker's verdict.
    rule_1_na = _na_status(payload.get("rule_1_status"))
    # Rule 1 is the max ownership of shares outstanding (J is text), so it is the checker's status;
    # Rule 3 is the max portfolio weight (K) against the checker's limit.
    return OrderedDict(
        [
            ("Date", report_date),
            ("Fund", fund_name),
            (_12D3_RULE_LABELS[0], rule_1_na or _pf(payload.get("rule_1_pass"))),
            (_12D3_RULE_LABELS[1], _pf(payload.get("rule_2_pass"))),
            (_12D3_RULE_LABELS[2], f'=IF(K{r}<={RULE_12D3_ASSET_LIMIT:g},"PASS","FAIL")'),
            ("12d3 Sec Biz Compliant", _pf(payload.get("twelve_d3_sec_biz_compliant")) if rule_1_na
             else f'=IF(AND(C{r}="PASS",D{r}="PASS",E{r}="PASS"),"PASS","FAIL")'),
            ("SEC-Related Businesses (Shs Out %)", sec_related_str),
            ("Combined Holdings (Portfolio %)", combined_str),
            ("Total Assets", num(calc, "total_assets")),
            ("Max Ownership %", f"{num(calc, 'max_ownership_pct'):.8%}"),
            ("Max Weight", num(calc, "max_weight")),
        ]
    )


def _is_verdict(value: object) -> bool:
    return isinstance(value, str) and (value.startswith(("=", "N/A")) or value in {"PASS", "FAIL"})


def _skipped_row(row: Mapping[str, object], reason: object) -> Dict[str, object]:
    """A rule the checker skipped: its verdict / formula cells say why, the other values stay blank."""
    status = f"N/A (skipped: {reason})" if reason else "N/A (skipped)"
    return {
        column: value if column in ("Date", "Fund") else (status if _is_verdict(value) else None)
        for column, value in row.items()
    }


class _RuleSheet(NamedTuple):
    key: str
    sheet: str
    footnote: str
    row: Callable[..., Mapping[str, object]]
    fund_filter: Optional[Callable[[str], bool]] = None
    raw_calc: bool = False  # calculations used without the `or {}` fallback
    warn_missing: bool = False


# Order = sheet order (after Summary, GICS_Compliance and GICS_Calculations).
_RULE_SHEETS = (
    _RuleSheet(
        "prospectus_80pct_policy", "Prospectus_80pct", "prospectus_80pct", _prospectus_row,
        # The checker skips the P3 and TR closed-end funds (prospectus_80pct_policy).
        fund_filter=lambda fund: not (_is_closed_end_fund(fund) and fund.startswith(("P3", "TR"))),
        warn_missing=True,
    ),
    _RuleSheet("diversification_40act_check", "40Act_Diversification", "40act", _forty_act_row),
    _RuleSheet("diversification_IRS_check", "IRS_Diversification", "irs", _irs_row),
    _RuleSheet("diversification_IRC_check", "IRC_Diversification", "irc", _irc_row, _is_vit_fund, raw_calc=True),
    _RuleSheet("max_15pct_illiquid_sai", "Illiquid", "illiquid", _illiquid_row, _is_etf_fund),
    _RuleSheet("real_estate_check", "Real_Estate", "real_estate", _real_estate_row, raw_calc=True),
    _RuleSheet("commodities_check", "Commodities", "commodities", _commodities_row, raw_calc=True),
    _RuleSheet("twelve_d1a_other_inv_cos", "12d1_Other_Investment_Companies", "12d1", _12d1_row),
    _RuleSheet("twelve_d2_insurance_cos", "12d2_Insurance_Companies", "12d2", _12d2_row, raw_calc=True),
    _RuleSheet("twelve_d3_sec_biz", "12d3_Securities_Business", "12d3", _12d3_row),
)


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
        file_stem: Optional[str] = None,
        test_functions: Optional[Iterable[str]] = None,
        gics_mapping: Optional[pd.DataFrame] = None,
    ) -> None:
        self.report_date = normalize_report_date(report_date)
        self.output_dir = ensure_dir(output_dir)
        self.file_stem = file_stem or f"{file_name_prefix}_{self.report_date}"
        self.file_path = self.output_dir / f"{self.file_stem}.xlsx"

        self.gics_mapping = gics_mapping.copy() if isinstance(gics_mapping, pd.DataFrame) else pd.DataFrame()
        self.test_functions = _selected_tests(test_functions) or None
        self.sheet_data: Dict[str, pd.DataFrame] = {}

        self.results = self._prepare_results(results)

        self.process_all_checks()
        self.export_to_excel()

    # ------------------------------------------------------------------
    def process_all_checks(self) -> None:
        steps = [("summary_metrics", self.process_summary), ("gics_compliance", self.process_gics_compliance)]
        steps += [(spec.key, partial(self._build_rule_sheet, spec)) for spec in _RULE_SHEETS]
        for test_name, processor in steps:
            if self.test_functions is None or test_name in self.test_functions:
                processor()

    # ------------------------------------------------------------------
    def _prepare_results(
        self, raw_results: Mapping[str, Mapping[str, object]]
    ) -> Dict[str, Dict[str, Any]]:
        if not raw_results:
            return {}
        if self._looks_like_date_mapping(raw_results):
            return {
                normalize_report_date(date_key): normalize_compliance_results(payload or {})
                for date_key, payload in raw_results.items()
            }
        return {self.report_date: normalize_compliance_results(raw_results)}

    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_date_mapping(payload: Mapping[object, Any]) -> bool:
        if not payload:
            return False
        if not any(ComplianceReport._is_date_key(key) for key in payload.keys()):
            return False
        return isinstance(next(iter(payload.values())), Mapping)

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
    def _build_rule_sheet(self, spec: _RuleSheet) -> None:
        rows: list[Mapping[str, object]] = []
        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()
            for fund_name, fund_data in funds.items():
                payload = fund_data.get(spec.key)
                if not payload:
                    if spec.warn_missing:
                        logger.warning("No prospectus 80%% policy data for %s", fund_name)
                    continue
                if spec.fund_filter is not None and not spec.fund_filter(fund_name):
                    continue
                calc = payload.get("calculations", {})
                if not spec.raw_calc:
                    calc = calc or {}
                # Excel row of this record: header is row 1, data starts at row 2.
                row = spec.row(report_date, fund_name, payload, calc, len(rows) + 2)
                rows.append(_skipped_row(row, payload.get("reason")) if payload.get("skipped") else row)

        if rows:
            self.sheet_data[spec.sheet] = self._append_footnotes(pd.DataFrame(rows), spec.footnote)

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
                row = {"Date": date_str, "Fund": fund_name}
                row.update((column, combined.get(key, 0.0)) for column, key in _SUMMARY_SHEET_FIELDS)
                rows.append(row)

        if not rows:
            self.sheet_data["Summary"] = pd.DataFrame(columns=_SUMMARY_COLUMNS)
            return

        df = pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        for column in _SUMMARY_COLUMNS[2:]:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

        df.sort_values(by=["Date", "Fund"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.sheet_data["Summary"] = df

    # ------------------------------------------------------------------
    def process_gics_compliance(self) -> None:
        gics_results: list[Dict[str, object]] = []
        gics_calculations: list[Dict[str, object]] = []
        mapping_df = self.gics_mapping
        if not mapping_df.empty:
            mapping_df = mapping_df.map(
                lambda value: str(value).strip() if value is not None else ""
            )

        # Precompute industry-name -> sector-name lookup once.
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
            "Date", "Fund", "GICS_Class", "GICS_Class Value", "GICS_SECTOR_NAME",
            "Fund_Weight_Value", "Index_Weight_Value", "Exceeding_Fund_25%?", "Exceeding_Index_25%?",
        ]

        for date_str, funds in self.results.items():
            report_date = pd.to_datetime(date_str).date()

            for fund_name, compliance_dict in funds.items():
                gics_data = compliance_dict.get("gics_compliance", {})
                if not isinstance(gics_data, Mapping):
                    continue

                gics_results.append(
                    dict(
                        zip(
                            summary_columns,
                            (
                                report_date,
                                fund_name,
                                gics_data.get("overall_gics_compliance", "FAIL"),
                                *_gics_flags(fund_name, gics_data),
                            ),
                        )
                    )
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
                        # Weights are floats (Series.to_dict()); `or 0.0` keeps -0.0 -> 0.0.
                        fund_weight_value = float_or(fund_wt, 0.0)
                        index_weight_value = float_or(index_weights.get(category, 0.0) or 0.0, 0.0)

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
            # GICS footnote rows only carry Date; the other cells stay NaN (written as None).
            footnotes = [{"Date": f"* {note}"} for note in FOOTNOTES.get("gics", [])]
            if footnotes:
                df_summary = pd.concat([df_summary, pd.DataFrame(footnotes)], ignore_index=True)
        else:
            df_summary = pd.DataFrame(columns=summary_columns)

        df_details = pd.DataFrame(gics_calculations, columns=detail_value_columns)
        if not df_details.empty:
            df_details.sort_values(
                by=["Fund", "Fund_Weight_Value"], ascending=[True, False], inplace=True
            )
            for column in ("Fund_Weight", "Index_Weight"):
                df_details[column] = df_details[f"{column}_Value"].apply(lambda value: f"{value:.2%}")
            df_details.drop(columns=["Fund_Weight_Value", "Index_Weight_Value"], inplace=True)
        else:
            df_details = pd.DataFrame(
                columns=[
                    "Date", "Fund", "GICS_Class", "GICS_Class Value", "GICS_SECTOR_NAME",
                    "Fund_Weight", "Index_Weight", "Exceeding_Fund_25%?", "Exceeding_Index_25%?",
                ]
            )

        self.sheet_data["GICS_Compliance"] = df_summary
        include_details = self.test_functions is None or len(self.test_functions) > 1
        if include_details and (not df_details.empty):
            self.sheet_data["GICS_Calculations"] = df_details

    # ------------------------------------------------------------------
    def export_to_excel(self) -> None:
        if not self.sheet_data:
            return

        with pd.ExcelWriter(self.file_path, engine="openpyxl") as writer:
            for sheet_name, df in self.sheet_data.items():
                output_df = df.copy()
                if not output_df.empty:
                    output_df = output_df.where(pd.notnull(output_df), None)
                output_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

            workbook = writer.book
            for sheet in workbook.sheetnames:
                ws = workbook[sheet]
                header_row = next(ws.iter_rows(min_row=1, max_row=1), ())
                headers = {cell.column_letter: (cell.value or "") for cell in header_row}
                for column_cells in ws.columns:
                    max_length = 0
                    column_letter = column_cells[0].column_letter
                    header_text = str(headers.get(column_letter, "")).lower()
                    is_percent = any(
                        keyword in header_text
                        for keyword in ("percent", "pct", "weight", "ownership")
                    ) or header_text in {"formula", "formula (mv)"}
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


# ----------------------------------------------------------------------
# PDF metric tables: (title, test_key, footnote_key, labels exempt from FAIL
# highlighting, metrics). Each metric is (label, formatter, source, field).
# ----------------------------------------------------------------------
_PDF_SECTIONS = (
    (
        "Summary Metrics", "summary_metrics", None, (),
        tuple((label, "cur", "summary", key) for label, key in _SUMMARY_FIELDS),
    ),
    (
        "Prospectus 80% Policy", "prospectus_80pct_policy", "prospectus_80pct", (),
        (
            ("Status", "status", "top", "is_compliant"),
            ("Result (DAN)", "pct", "calc", "names_test"),
            ("Result (MV)", "pct", "calc", "names_test_mv"),
            ("Options In Scope", "yesno", "calc", "options_in_scope"),
            ("Option Contribution", "cur", "calc", "option_contribution"),
        ),
    ),
    (
        "40 Act Diversification", "diversification_40act_check", "40act",
        ("Registration Match", "Condition 2a OCC"),
        (
            ("Overall Status", None, "status_or_skip", None),
            ("Registered Status", "title", "calc", "fund_registration"),
            ("Status Today", "title", "calc", "fund_status_today"),
            ("Registration Match", "status", "registrations_match", None),
            ("Condition 1", "status", "detail", "condition_40act_1"),
            ("Condition 2a", "status", "detail", "condition_40act_2a"),
            ("Condition 2b", "status", "detail", "condition_40act_2b_status"),
            ("Condition 2a OCC", "status", "detail", "condition_40act_2a_occ"),
            ("Max Ownership %", "pct", "calc", "max_ownership_float"),
        ),
    ),
    (
        "IRS Diversification", "diversification_IRS_check", "irs", (),
        (
            ("Overall Status", None, "status_or_skip", None),
            ("Condition 1", "status", "detail", "condition_IRS_1"),
            ("Condition 2a 50%", "status", "detail", "condition_IRS_2_a_50"),
            ("Condition 2a 5%", "status", "detail", "condition_IRS_2_a_5"),
            ("Condition 2a 10%", "status", "detail", "condition_IRS_2_a_10_status"),
            ("Condition 2b", "status", "detail_or_na", "condition_IRS_2_b"),  # the checker has no 2b test
            ("Sum Large Securities %", "pct", "calc", "sum_large_securities_weights"),
            ("Large Securities Count", "int", "calc", "large_securities_count"),
        ),
    ),
    (
        "IRC Diversification", "diversification_IRC_check", "irc", (),
        (
            ("Overall Status", "status", "top", "is_compliant"),
            *((f"Condition {n}", "status", "detail", f"condition_IRC_{n}") for n in (55, 70, 80, 90)),
            ("Top 1 Exposure", "pct", "calc", "top_1"),
            ("Top 4 Exposure", "pct", "calc", "top_4"),
        ),
    ),
    (
        "15% Illiquid Assets", "max_15pct_illiquid_sai", "illiquid", (),
        (
            ("Status", "status", "top", "is_compliant"),
            ("Illiquid Exposure %", "pct", "calc", "illiquid_percentage"),
            ("Total Illiquid Value", "cur", "calc", "total_illiquid_value"),
        ),
    ),
    (
        "Real Estate", "real_estate_check", "real_estate", (),
        (
            ("Status", "status", "top", "is_compliant"),
            ("Real Estate Exposure", "pct", "calc", "real_estate_percentage"),
        ),
    ),
    (
        "Commodities", "commodities_check", "commodities", (),
        (
            ("Status", "status", "top", "is_compliant"),
            ("Commodities Exposure", "pct", "first_calc", ("commodities_percentage", "commodities_exposure")),
        ),
    ),
    (
        "Rule 12d1-1", "twelve_d1a_other_inv_cos", "12d1", (),
        (
            ("Status", "status", "top", "is_compliant"),
            ("Test 1", "status", "first_detail", ("test_1_status", "test_1_pass")),
            ("Test 2", "status", "detail", "test_2_pass"),
            ("Test 3", "status", "detail", "test_3_pass"),
        ),
    ),
    (
        "Rule 12d2", "twelve_d2_insurance_cos", "12d2", (),
        (
            # A single test: its status (N/A when shares outstanding is missing), else the verdict.
            ("Status", "status", "first_detail", ("test_status", "is_compliant")),
            ("Max Ownership", "pct", "calc", "max_ownership_pct"),
        ),
    ),
    (
        "Rule 12d3", "twelve_d3_sec_biz", "12d3", (),
        (
            ("Overall Status", "status", "top", "is_compliant"),
            (_12D3_RULE_LABELS[0], "status", "first_detail", ("rule_1_status", "rule_1_pass")),
            (_12D3_RULE_LABELS[1], "status", "detail", "rule_2_pass"),
            (_12D3_RULE_LABELS[2], "status", "detail", "rule_3_pass"),
        ),
    ),
)


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
        self.allowed_tests = _selected_tests(allowed_tests)
        self.generate_pdf()
        self.save_pdf()

    def save_pdf(self) -> None:
        try:
            self.output()
        except Exception as e:
            logger.error(f"Failed to save PDF to {self.output_path}: {e}")
            raise

    # ------------------------------------------------------------------
    def generate_pdf(self) -> None:
        # BaseReportPDF.__init__ has already added the first page.
        self.pdf.set_y(self.pdf.t_margin)
        grouped_results = self._group_results_by_fund()
        if not grouped_results:
            self._add_header("Compliance Overview")
            self.pdf.set_font("Arial", "", 9)
            self.pdf.cell(0, 6, "No compliance results available.", ln=True)
            self.output()  # written again by save_pdf (kept as-is)
            return

        self.grouped_results = grouped_results
        self.fund_order = sorted(grouped_results)

        for idx, (title, test_key, footnote, skip_labels, metrics) in enumerate(_PDF_SECTIONS):
            if self._is_test_selected(test_key):
                self._render_metric_section(title, test_key, metrics, footnote, skip_labels)

            # GICS overview goes right after the Summary Metrics slot, whether or not Summary is selected.
            if idx == 0 and self._is_test_selected("gics_compliance"):
                has_gics_data = any(
                    isinstance(payload, dict) and "gics_compliance" in payload
                    for payload in self.results.values()
                )
                if has_gics_data:
                    self.pdf.ln(4)
                    self._print_test_header("GICS Diversification")
                    self._render_gics_compliance_overview()
                    self.pdf.ln(8)

    # ------------------------------------------------------------------
    def _group_results_by_fund(self) -> Dict[str, Dict[str, Any]]:
        """Payload of each fund's last date."""
        grouped: Dict[str, Dict[str, Any]] = {}
        for (fund_name, date_str), payload in sorted(
            self.results.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            grouped[fund_name] = payload
        return grouped

    def _render_metric_section(self, title, test_key, metrics, footnote_key, skip_fail_labels) -> None:
        rows: list[list[str]] = []
        for label, fmt, source, field in metrics:
            row: list[str] = [self._sanitize_text(label)]
            for fund in self.fund_order:
                fund_payload = self.grouped_results.get(fund, {})
                row.append(self._sanitize_text(self._cell_text(fund_payload, test_key, fmt, source, field)))
            rows.append(row)

        estimated_height = 5.0 + (4.2 * len(rows)) + 6
        if self.pdf.get_y() + estimated_height > self.pdf.h - self.pdf.b_margin:
            self.pdf.add_page()

        self._print_test_header(title)
        self._draw_metric_table(["Metric", *self.fund_order], rows, skip_fail_labels)
        self._draw_footnotes_section(footnote_key)
        self.pdf.ln(2)

    def _cell_text(self, fund_data, test_key, fmt, source, field) -> str:
        # The checker runs IRC only for VIT funds; another fund in the run has no IRC result,
        # so its column reads N/A rather than blank statuses and a measured-looking 0% exposure.
        if test_key == "diversification_IRC_check" and test_key not in (fund_data or {}):
            return "N/A"
        try:
            value = self._metric_value(fund_data, test_key, fmt, source, field)
        except Exception:
            value = ""
        return str(value or "")

    def _metric_value(self, fund_data, test_key, fmt, source, field) -> object:
        if source == "status_or_skip" or (fmt == "status" and self._is_test_skipped(fund_data, test_key)):
            return self._status_or_skip(fund_data, test_key)
        payload = self._get_test_payload(fund_data, test_key)
        if source == "summary":
            value = self._get_summary_value(fund_data, field)
        elif source == "top":
            value = payload.get(field)
        elif source == "calc":
            value = self._get_calculation(payload, field)
        elif source == "first_calc":
            value = self._first_not_none(*(self._get_calculation(payload, key) for key in field))
        elif source == "detail":
            value = self._get_detail(payload, field)
        elif source == "detail_or_na":
            # A condition the checker does not evaluate reads N/A, as in the workbook, instead of blank.
            value = self._get_detail(payload, field)
            if value is None and payload:
                value = "N/A (not evaluated)"
        elif source == "first_detail":
            value = self._first_not_none(*(self._get_detail(payload, key) for key in field))
        elif source == "registrations_match":
            value = self._registrations_match(payload)
        else:
            raise NotImplementedError(source)
        formatters = {
            "status": self._status_text,
            "pct": self._format_percent,
            "cur": self._format_currency,
            "int": self._format_integer,
            "yesno": self._yes_no,
            "title": self._title_case,
        }
        return formatters[fmt](value)

    # ------------------------------------------------------------------
    def _add_header(self, title: str) -> None:
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, self._sanitize_text(title), ln=True, align="C")
        self.pdf.ln(4)

    def _print_test_header(self, title: str) -> None:
        self.pdf.set_font("Arial", "B", 10)
        self.pdf.cell(0, 6, self._sanitize_text(title), ln=True)
        self.pdf.set_font("Arial", "", 9)

    def _header_row(self, widths, headers, height: float, font_size: int) -> None:
        headers = [self._sanitize_text(header) for header in headers]
        draw_header_row(self.pdf, widths, headers, height, font_size, (230, 230, 230), height)

    def _value_cell(self, width: float, height: float, text: str, align: str, highlight_fail: bool) -> None:
        if highlight_fail:
            self.pdf.set_fill_color(255, 200, 200)
            self.pdf.set_text_color(139, 0, 0)
            self.pdf.cell(width, height, text, border=1, align="C", fill=True)
            self.pdf.set_text_color(0, 0, 0)
        else:
            self.pdf.set_fill_color(255, 255, 255)
            self.pdf.cell(width, height, text, border=1, align=align)

    def _draw_metric_table(self, headers: list[str], rows: list[list[str]], skip_fail_labels) -> None:
        skip_fail = {self._sanitize_text(label).lower() for label in skip_fail_labels}

        usable_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin
        metric_width = min(45.0, usable_width * 0.3)
        value_width = (usable_width - metric_width) / max(len(headers) - 1, 1)
        widths = [metric_width] + [value_width] * (len(headers) - 1)

        self._header_row(widths, headers, 5.0, 7)
        self.pdf.set_font("Arial", "", 6)
        for row in rows:
            row_label = self._sanitize_text(row[0]).lower()
            for idx, value in enumerate(row):
                text = self._sanitize_text(value).replace("\n", " /")
                align = "R" if idx != 0 and self._looks_like_number(text) else "L"
                fail = text.upper() == "FAIL" and row_label not in skip_fail
                self._value_cell(widths[idx], 4.2, text, align, fail)
            self.pdf.ln(4.2)

    def _draw_table(self, headers: list[str], rows: list[list[str]], col_widths: list[float], row_height: int) -> None:
        self._header_row(col_widths, headers, row_height, 9)
        self.pdf.set_font("Arial", "", 8)
        for row in rows:
            for width, value in zip(col_widths, row):
                text = self._sanitize_text(value)
                self._value_cell(width, row_height, text, "C", text.upper() == "FAIL")
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

    # ------------------------------------------------------------------
    def _get_summary_value(self, fund_data: Mapping[str, Any], key: str) -> object:
        if not isinstance(fund_data, Mapping):
            return 0.0
        summary_result = fund_data.get("summary_metrics")
        if not isinstance(summary_result, Mapping):
            return 0.0
        return summary_result.get(key, 0.0)

    def _is_test_selected(self, test_name: str) -> bool:
        # The IRC section only appears when the run has a VIT fund (self.results is keyed by (fund, date)).
        if test_name == "diversification_IRC_check":
            has_vit_wrapper = any(_vehicle_wrapper(fund_name) == "vit" for fund_name, _ in self.results)
            if not has_vit_wrapper:
                return False
        if not self.allowed_tests:
            return True
        return test_name in self.allowed_tests

    def _is_test_skipped(self, fund_data: Mapping[str, Any], key: str) -> bool:
        # The checker's details are flattened into the payload (report_utils); a nested
        # "details" mapping is still honoured.
        payload = self._get_test_payload(fund_data, key)
        details = payload.get("details")
        details = details if isinstance(details, Mapping) else {}
        return bool(payload.get("skipped") or details.get("skipped") or details.get("skip"))

    def _status_or_skip(self, fund_data: Mapping[str, Any], key: str) -> str:
        if self._is_test_skipped(fund_data, key):
            return "N/A"
        return self._status_text(self._get_test_payload(fund_data, key).get("is_compliant"))

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
        return "" if value is None else str(value).strip().replace("_", " ").title()

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
        # Same rule as the checker: a non-diversified registration permits operating either way.
        return str(actual).strip().lower() == "diversified" or str(expected).strip().lower() != "diversified"

    @staticmethod
    def _status_text(value: object) -> str:
        if isinstance(value, (bool, np.bool_)):  # numpy.bool_ from checks on numpy values
            return "PASS" if value else "FAIL"
        if isinstance(value, str):
            upper = value.strip().upper()
            if upper in {"PASS", "FAIL"}:
                return upper
            if upper.startswith("N/A"):  # e.g. no shares outstanding data for an ownership test
                return "N/A"
        # Unknown / missing test result: surface it rather than silently passing.
        return ""

    @staticmethod
    def _yes_no(value: object) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if value else "No"
        return ""

    @staticmethod
    def _format_number(value: object, spec: str) -> str:
        try:
            number = float(value or 0.0)
        except (TypeError, ValueError):
            return ""
        return "" if pd.isna(number) else format(number, spec)

    def _format_currency(self, value: object) -> str:
        return self._format_number(value, ",.0f")

    def _format_percent(self, value: object) -> str:
        return self._format_number(value, ".5%")

    def _format_integer(self, value: object) -> str:
        try:
            number = int(float(value or 0.0))
        except (TypeError, ValueError):
            return ""
        return f"{number:d}"

    @staticmethod
    def _first_not_none(*values: object) -> object:
        return next((value for value in values if value is not None), None)

    # ------------------------------------------------------------------
    def _draw_footnotes_section(self, key: Optional[str]) -> None:
        notes = FOOTNOTES.get(key, [])
        if not notes:
            return

        self.pdf.set_font("Arial", "I", 8)
        self.pdf.ln(1)
        # Reset X to the left margin before multi_cell(width=0); a cursor near the
        # right margin triggers fpdf2's "Not enough horizontal space" error.
        self.pdf.set_x(self.pdf.l_margin)
        self.pdf.multi_cell(0, 5, self._sanitize_text("Footnotes:"))
        for note in notes:
            self.pdf.set_x(self.pdf.l_margin)
            self.pdf.multi_cell(0, 4.5, self._sanitize_text(f"* {note}"))
        self.pdf.ln(2)

    # ------------------------------------------------------------------
    def _render_gics_compliance_overview(self) -> None:
        """Render the GICS_Compliance summary (all dates) as a PDF table."""
        headers = [
            "Date", "Fund", "Overall", "Ind >25%", "IndGrp >25%", "Idx Ind >25%",
            "Idx IndGrp >25%", "Fund>25% if Idx?", "Exceptions",
        ]

        rows: list[list[str]] = []
        for (fund_name, date_str), payload in sorted(
                self.results.items(), key=lambda kv: (kv[0][0], kv[0][1])
        ):
            gics = (payload or {}).get("gics_compliance")
            if not isinstance(gics, dict):
                continue

            *flags, can_exceed, exceptions = _gics_flags(str(fund_name).upper(), gics)
            try:
                date_disp = datetime.fromisoformat(date_str).date()
            except Exception:
                date_disp = date_str

            rows.append([
                str(date_disp),
                str(fund_name),
                # Same fallback as the GICS_Compliance sheet: an early checker fail() carries no overall status.
                str(gics.get("overall_gics_compliance", "FAIL")),
                *("Yes" if flag else "No" for flag in flags),
                can_exceed,
                exceptions,
            ])

        if not rows:
            return

        usable = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin

        # Measure with the fonts used to draw the table.
        self.pdf.set_font("Arial", "B", 7)
        hdr_w = [self.pdf.get_string_width(h) for h in headers]
        self.pdf.set_font("Arial", "", 7)
        row_w = [0.0] * len(headers)
        for r in rows:
            for i, val in enumerate(r):
                row_w[i] = max(row_w[i], self.pdf.get_string_width(val))

        PADDING = 8.0
        MIN_W = 20.0
        col_widths = [max(MIN_W, hw, rw) + PADDING for hw, rw in zip(hdr_w, row_w)]

        total = sum(col_widths)
        if total > usable:
            # Two-pass shrink: reserve MIN_W per column, then share the rest in
            # proportion to each column's excess over MIN_W (a naive scale would
            # squeeze Yes/No columns below fpdf2's minimum glyph width).
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
            # Extra room goes to the last column (Exceptions).
            col_widths[-1] += (usable - total)

        self._draw_table(headers, rows, col_widths, row_height=6)
        self._draw_footnotes_section("gics")
        self.pdf.ln(4)

def generate_compliance_reports(
    results: Mapping[str, Mapping[str, object]],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    file_name_prefix: str = "compliance_results",
    file_stem: Optional[str] = None,
    test_functions: Optional[Iterable[str]] = None,
    gics_mapping: Optional[pd.DataFrame] = None,
    create_pdf: bool = True,
) -> GeneratedComplianceReport:
    """Create Excel and PDF compliance reports named ``<file_stem>`` (default ``<prefix>_<report_date>``)."""

    report = ComplianceReport(
        results=results,
        report_date=report_date,
        output_dir=output_dir,
        file_name_prefix=file_name_prefix,
        file_stem=file_stem,
        test_functions=test_functions,
        gics_mapping=gics_mapping,
    )

    # Report only the files this call wrote: export_to_excel writes nothing without sheet data,
    # and a failed PDF leaves no file behind.
    excel_file_path = str(report.file_path) if report.sheet_data else None
    pdf_file_path = None
    if create_pdf:
        try:
            pdf_target = str(Path(output_dir) / f"{report.file_stem}.pdf")
            ComplianceReportPDF(report.results, pdf_target, allowed_tests=test_functions)
            pdf_file_path = pdf_target
        except RuntimeError as exc:  # pragma: no cover - optional dependency missing
            logger.warning("PDF generation skipped: %s", exc)
        except Exception as exc:
            # exc_info=True so the actual fpdf2 stack trace surfaces in the log.
            logger.error("PDF generation failed: %s", exc, exc_info=True)

    return GeneratedComplianceReport(excel_path=excel_file_path, pdf_path=pdf_file_path)