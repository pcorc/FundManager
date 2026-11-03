"""Holdings reconciliation reporting utilities."""
from __future__ import annotations
from openpyxl.styles import Alignment, Font
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from config.fund_definitions import FUND_DEFINITIONS
from reporting.report_utils import (
    _normalize_reconciliation_payload,
    ensure_dir,
    non_empty,
    normalize_report_date,
    solid_fill,
)

_PRICE_KEYS: Tuple[str, str] = ("price_discrepancies_T", "price_discrepancies_T1")
_PRICE_SUFFIXES: Tuple[str, str] = ("_price_T", "_price_T-1")
_ASSET_TYPE_TOKENS: Tuple[Tuple[str, str], ...] = (
    ("flex", "FLEX"),
    ("treasury", "Treasury"),
    ("equity", "Equity"),
    ("option", "Option"),
)

FlatResults = Dict[Tuple[str, str, str], pd.DataFrame]


@dataclass
class GeneratedReconciliationReport:
    """Container for holdings reconciliation artefact locations."""
    excel_path: Optional[str]

@dataclass(frozen=True)
class ReconDescriptor:
    """Declarative definition of how to flatten reconciliation payloads."""

    ticker: str
    price_keys: Tuple[str | None, str | None] = _PRICE_KEYS
    select_standard_prices: bool = False
    callbacks: Tuple[str, ...] = ("_append_holdings_breakdowns",)


RECON_DESCRIPTOR_REGISTRY: Dict[str, ReconDescriptor] = {
    "custodian_equity": ReconDescriptor("eqyticker"),
    "custodian_option": ReconDescriptor(
        "optticker",
        select_standard_prices=True,
        callbacks=("_append_option_breakdowns", "_append_holdings_breakdowns"),
    ),
    "custodian_flex_option": ReconDescriptor("optticker"),  # Don't filter FLEX price breaks
    "custodian_treasury": ReconDescriptor("cusip", price_keys=("price_discrepancies_T", None)),
}


def vehicle_wrapper(fund: str) -> str:
    """Lower-cased FUND_DEFINITIONS vehicle_wrapper ('' when unset)."""
    return (FUND_DEFINITIONS.get(fund, {}).get("vehicle_wrapper") or "").lower()


def split_flex_price_frames(df: Any, flag_column: str = "is_flex") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(FLEX rows, non-FLEX rows) copies of ``df``; all rows are non-FLEX without the flag column."""
    if not non_empty(df):
        return pd.DataFrame(), pd.DataFrame()

    flag_series = df.get(flag_column)
    if flag_series is None:
        return pd.DataFrame(), df.copy()

    mask = flag_series.fillna(False).astype(bool)
    return df.loc[mask].copy(), df.loc[~mask].copy()


def top_option_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the five largest option price breaks by weight (else by price diff)."""
    df = df.copy()
    if "option_weight" in df.columns:
        df = df.sort_values("option_weight", ascending=False)
    elif "price_diff" in df.columns:
        df = df.sort_values("price_diff", ascending=False)
    return df.head(5)


class ReconciliationReport:
    """Generate the enhanced Excel holdings reconciliation report."""
    SUMMARY_MAPPING: Dict[str, str] = {
        "custodian_equity": "custodian_equity_holdings_breaks_T",
        "custodian_equity_price_T": "custodian_equity_price_breaks_T",
        "custodian_option": "custodian_option_holdings_breaks_T",
        "custodian_option_price_T": "custodian_option_price_breaks_T",
        "custodian_flex_option": "custodian_flex_option_holdings_breaks_T",
        "custodian_flex_option_price_T": "custodian_flex_option_price_breaks_T",
        "custodian_treasury": "custodian_treasury_holdings_breaks_T",
        "custodian_treasury_price_T": "custodian_treasury_price_breaks_T",
        "index_equity": "index_equity_holdings_breaks",
        "index_equity_price_T": "index_equity_price_breaks_T",
    }

    def __init__(
        self,
        reconciliation_results: Mapping[str, Mapping[str, Any]],
        date: str,
        file_path_excel: str | Path,
    ) -> None:
        self.reconciliation_results = reconciliation_results
        self.date = str(date)
        self.output_path = Path(file_path_excel)
        self.flattened_results = self._filter_and_flatten_results()
        self._export_to_excel()

    # ------------------------------------------------------------------
    def _filter_and_flatten_results(self) -> FlatResults:
        flat: FlatResults = {}
        for date_str, fund_data in self.reconciliation_results.items():
            for fund, recon_dict in (fund_data or {}).items():
                for recon_type, subresults in (recon_dict or {}).items():
                    descriptor = RECON_DESCRIPTOR_REGISTRY.get(recon_type)
                    if not descriptor:
                        continue
                    subresults = subresults or {}
                    holdings = subresults.get("final_recon")
                    if non_empty(holdings):
                        flat[(fund, date_str, recon_type)] = self._prepare_holdings_df(
                            holdings, fund, date_str, recon_type, descriptor.ticker
                        )
                    for suffix, price_key in zip(_PRICE_SUFFIXES, descriptor.price_keys):
                        price = subresults.get(price_key) if price_key else None
                        if descriptor.select_standard_prices:
                            price = split_flex_price_frames(price)[1]
                        if non_empty(price):
                            flat[(fund, date_str, recon_type + suffix)] = self._prepare_price_df(
                                price, fund, date_str, recon_type, descriptor.ticker
                            )
                    for callback in descriptor.callbacks:
                        getattr(self, callback)(flat, fund, date_str, subresults, recon_type)
        return flat

    def _append_option_breakdowns(
        self, flat: FlatResults, fund: str, date_str: str, subresults: Mapping[str, Any], recon_type: str
    ) -> None:
        regular_df = subresults.get("regular_options")
        if non_empty(regular_df):
            flat[(fund, date_str, "custodian_option_regular")] = self._prepare_holdings_df(
                regular_df, fund, date_str, "custodian_option_regular", "optticker"
            )

        for suffix, price_key in zip(_PRICE_SUFFIXES, _PRICE_KEYS):
            flex_price, _ = split_flex_price_frames(subresults.get(price_key))
            if not flex_price.empty:
                key = f"custodian_option_flex{suffix}"
                flat[(fund, date_str, key)] = self._prepare_price_df(flex_price, fund, date_str, key, "optticker")

    def _append_holdings_breakdowns(
        self, flat: FlatResults, fund: str, date_str: str, subresults: Mapping[str, Any], recon_type: str
    ) -> None:
        final_df = subresults.get("final_recon")
        if not non_empty(final_df):
            return

        if "breakdown" not in final_df.columns or "discrepancy_type" not in final_df.columns:
            return

        breakdowns = (
            final_df[["discrepancy_type", "breakdown"]]
            .copy()
            .assign(count=1)
            .groupby(["discrepancy_type", "breakdown"], dropna=False)["count"]
            .sum()
            .reset_index()
            .rename(columns={"count": "occurrences"})
        )
        flat[(fund, date_str, f"{recon_type}_breakdowns")] = breakdowns

    # ------------------------------------------------------------------
    def _prepare_holdings_df(
        self,
        df: pd.DataFrame,
        fund: str,
        date_str: str,
        recon_type: str,
        ticker_col: str,
    ) -> pd.DataFrame:
        df = df.copy()
        df["FUND"] = fund
        df["RECON_TYPE"] = recon_type
        df["DATE"] = date_str

        base_cols = ["FUND", "RECON_TYPE", "DATE", ticker_col, "discrepancy_type"]
        optional_cols = [
            "quantity",
            "shares_cust",
            "final_discrepancy",
            "trade_discrepancy",
            "in_vest",
            "in_cust",
            "in_index",
            "quantity_diff",
            "breakdown",
        ]
        cols_to_keep = [col for col in base_cols + optional_cols if col in df.columns]
        return df[cols_to_keep] if cols_to_keep else df

    def _limit_option_prices(self, df: pd.DataFrame, fund: str, recon_type: str) -> pd.DataFrame:
        vehicle = vehicle_wrapper(fund)
        recon_lower = recon_type.lower()
        if (
            df.empty
            or "option" not in recon_lower
            or "flex" in recon_lower
            or vehicle not in {"private_fund", "closed_end_fund"}
        ):
            return df
        return top_option_prices(df)

    def _prepare_price_df(
        self,
        df: pd.DataFrame,
        fund: str,
        date_str: str,
        recon_type: str,
        ticker_col: str,
    ) -> pd.DataFrame:
        df = df.copy()
        df = self._limit_option_prices(df, fund, recon_type)
        df["FUND"] = fund
        df["RECON_TYPE"] = recon_type
        df["DATE"] = date_str
        if ticker_col in df.columns:
            df = df.rename(columns={ticker_col: "TICKER"})
        cols = [
            "FUND",
            "RECON_TYPE",
            "DATE",
            "TICKER",
            "price_vest",
            "price_cust",
            "price_diff",
            "price_pct_diff",
            "option_weight",
            "discrepancy_type",
        ]
        cols = [c for c in cols if c in df.columns]
        return df[cols] if cols else df

    # ------------------------------------------------------------------
    def _export_to_excel(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
            if not self.flattened_results:
                pd.DataFrame(["All reconciliations passed - no breaks found"]).to_excel(
                    writer,
                    sheet_name="RECONCILIATION BREAKS SUMMARY",
                    index=False,
                )
            else:
                self._create_summary_tab(writer)
                self._create_detailed_breaks_tab(writer)
                self._create_price_comparison_tab(writer)
                self._create_flex_option_tab(writer)
                self._format_price_comparison_tab(writer)
            self._apply_header_styles(writer)
            if writer.book.sheetnames:
                writer.book[writer.book.sheetnames[0]].sheet_state = "visible"

    def _create_summary_tab(self, writer: pd.ExcelWriter) -> None:
        funds = sorted({fund for (fund, _date, _recon) in self.flattened_results})

        # Rows are "Date" + one row per recon-break column; columns are the funds.
        rows: Dict[str, Dict[str, Any]] = {"Date": {fund: self.date for fund in funds}}
        for col in self.SUMMARY_MAPPING.values():
            rows[col] = {fund: 0 for fund in funds}

        for (fund_name, _date, recon_key), df in self.flattened_results.items():
            column = self.SUMMARY_MAPPING.get(recon_key)
            if column and fund_name in rows[column]:
                rows[column][fund_name] += len(df)

        summary_df = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=funds)
        summary_df.index.name = "Reconciliation Type"
        summary_df.to_excel(
            writer,
            sheet_name="RECONCILIATION BREAKS SUMMARY",
            index=True,
            startrow=0,
        )

        worksheet = writer.sheets["RECONCILIATION BREAKS SUMMARY"]
        red_fill, orange_fill = solid_fill("FFC8C8"), solid_fill("FFC896")

        # Row 1 = header (fund names); row 2+ = data rows in the order of summary_df.index.
        for r_offset, recon_label in enumerate(summary_df.index, start=2):
            label_lower = str(recon_label).lower()
            if "holdings_breaks" in label_lower:
                fill = red_fill
            elif "price_breaks" in label_lower:
                fill = orange_fill
            else:
                continue
            for c in range(2, len(funds) + 2):
                cell = worksheet.cell(row=r_offset, column=c)
                try:
                    if cell.value is not None and float(cell.value) > 0:
                        cell.fill = fill
                except (TypeError, ValueError):
                    pass

        has_private_or_closed = any(
            (FUND_DEFINITIONS.get(f, {}).get("vehicle_wrapper") in {"private_fund", "closed_end_fund"})
            for f in funds
        )
        if has_private_or_closed:
            cell = worksheet.cell(row=len(summary_df) + 3, column=1)
            cell.value = (
                "Note: For private/closed-end funds, standard option price breaks show top 5 by weight. "
                "All FLEX option breaks are shown."
            )
            cell.font = cell.font.copy(italic=True, size=9)
        worksheet.column_dimensions["A"].width = 45

    def _standardize_ticker_column(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["eqyticker", "optticker", "occ_symbol"]:
            if col in df.columns:
                return df.rename(columns={col: "TICKER"})
        return df

    @staticmethod
    def _determine_asset_type(recon_type: str) -> str:
        recon_lower = recon_type.lower()
        return next((label for token, label in _ASSET_TYPE_TOKENS if token in recon_lower), "Unknown")

    def _create_detailed_breaks_tab(self, writer: pd.ExcelWriter) -> None:
        all_breaks = [
            df.assign(**{"Asset Type": self._determine_asset_type(recon_type)})
            for (_fund, _date, recon_type), df in self.flattened_results.items()
            if "_price_" not in recon_type
        ]
        if not all_breaks:
            return
        detailed_breaks = pd.concat(all_breaks, ignore_index=True)
        detailed_breaks = self._standardize_ticker_column(detailed_breaks)

        cols = list(detailed_breaks.columns)
        cols.remove("Asset Type")
        ticker_idx = next((i for i, c in enumerate(cols) if c == "TICKER"), 3)
        cols.insert(min(ticker_idx + 1, len(cols)), "Asset Type")
        detailed_breaks[cols].to_excel(writer, sheet_name="DETAILED BREAKS BY SECURITY", index=False)

    def _create_price_comparison_tab(self, writer: pd.ExcelWriter) -> None:
        # Price frames already carry FUND/DATE; T-1 entries are not shown, so PERIOD is always "T".
        price_comparison = [
            df.assign(PERIOD="T")
            for (_fund, _date, recon_type), df in self.flattened_results.items()
            if "_price_" in recon_type and "t-1" not in recon_type.lower()
        ]
        if not price_comparison:
            return
        all_prices = pd.concat(price_comparison, ignore_index=True)
        all_prices = self._limit_price_comparison_options(all_prices)
        comparison_df = self._build_price_comparison_df(all_prices)
        comparison_df.to_excel(writer, sheet_name="COMPREHENSIVE PRICE COMPARISON", index=False)

    def _limit_price_comparison_options(self, all_prices: pd.DataFrame) -> pd.DataFrame:
        if all_prices.empty or "RECON_TYPE" not in all_prices.columns:
            return all_prices

        limited_frames: list[pd.DataFrame] = []
        for (_fund, recon_type), subset in all_prices.groupby(["FUND", "RECON_TYPE"]):
            recon_lower = str(recon_type).lower()
            if "flex" in recon_lower or "option" not in recon_lower:
                limited_frames.append(subset)
            else:
                limited_frames.append(top_option_prices(subset))
        return pd.concat(limited_frames, ignore_index=True)

    def _build_price_comparison_df(self, all_prices: pd.DataFrame) -> pd.DataFrame:
        ticker_col = "TICKER" if "TICKER" in all_prices.columns else "equity_ticker"
        rows = [
            self._build_price_row(fund, ticker, period, group)
            for (fund, ticker, period), group in all_prices.groupby(["FUND", ticker_col, "PERIOD"])
        ]
        return pd.DataFrame(rows).sort_values(["Fund", "Asset Type", "Period"])

    def _build_price_row(self, fund: str, ticker: str, period: str, group: pd.DataFrame) -> Dict[str, Any]:
        """Build a single price-comparison row (every price source is a custodian recon)."""
        row: Dict[str, Any] = {
            "Fund": fund,
            "Ticker": ticker,
            "Period": period,
            "Fund_Price": np.nan,
            "Custodian_Price": np.nan,
            "Index_Price": np.nan,
            "Price_Diff": np.nan,
            "Asset Type": self._determine_price_asset_type(group),
            "Option Weight": "",
            "% Difference": "",
        }

        for _, price_row in group.iterrows():
            row["Fund_Price"] = price_row.get("price_vest")
            row["Custodian_Price"] = price_row.get("price_cust")
            row["Price_Diff"] = price_row.get("price_diff")

            weight = price_row.get("option_weight")
            if isinstance(weight, (int, float)) and pd.notna(weight) and weight:
                row["Option Weight"] = f"{weight * 100:.2f}%"

            pct = price_row.get("price_pct_diff")
            if isinstance(pct, (int, float)):
                row["% Difference"] = f"{pct:.1f}%"

        prices = [p for p in (row["Fund_Price"], row["Custodian_Price"], row["Index_Price"]) if pd.notna(p)]
        if prices:
            row["Price_Diff"] = max(prices) - min(prices)
        return row

    @staticmethod
    def _determine_price_asset_type(group: pd.DataFrame) -> str:
        recon_types = group["RECON_TYPE"].str.lower().fillna("")
        for token, label in _ASSET_TYPE_TOKENS[1:]:
            if recon_types.str.contains(token).any():
                return label
        return "Unknown"

    def _create_flex_option_tab(self, writer: pd.ExcelWriter) -> None:
        filtered = [
            df for (_fund, _date, recon), df in self.flattened_results.items() if recon == "custodian_flex_option"
        ]
        if filtered:
            tab_df = self._standardize_ticker_column(pd.concat(filtered, ignore_index=True))
            tab_df.to_excel(writer, sheet_name="FLEX_OPTION_RECONCILIATION", index=False)

    def _format_price_comparison_tab(self, writer: pd.ExcelWriter) -> None:
        if "COMPREHENSIVE PRICE COMPARISON" not in writer.sheets:
            return

        # Header styles are applied for every sheet by _apply_header_styles;
        # this method only handles the price-specific column tinting + alignment.
        worksheet = writer.sheets["COMPREHENSIVE PRICE COMPARISON"]
        weight_fill = solid_fill("E2F0D9")
        col_index = {cell.value: idx for idx, cell in enumerate(worksheet[1], start=1)}

        for header, fill in (("Option Weight", weight_fill), ("% Difference", None)):
            idx = col_index.get(header)
            if not idx:
                continue
            for row in range(2, worksheet.max_row + 1):
                cell = worksheet.cell(row=row, column=idx)
                if cell.value:
                    if fill:
                        cell.fill = fill
                    cell.alignment = Alignment(horizontal="right")

        for column in worksheet.columns:
            max_length = 0
            for cell in column:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            worksheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)

    @staticmethod
    def _apply_header_styles(writer: pd.ExcelWriter) -> None:
        header_fill = solid_fill("D3E4F7")
        header_font = Font(bold=True)
        for worksheet in writer.sheets.values():
            for cell in worksheet[1]:
                if cell.value is None:
                    continue
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center", vertical="center")


def generate_reconciliation_reports(
    results: Mapping[str, Any],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    file_name_prefix: str = "reconciliation_results",
) -> GeneratedReconciliationReport:
    normalized = _normalize_reconciliation_payload(results)
    if not normalized:
        return GeneratedReconciliationReport(None)

    date_str = normalize_report_date(report_date)
    output_path = ensure_dir(output_dir)

    excel_path = output_path / f"{file_name_prefix}_{date_str}.xlsx"
    structured_details = {date_str: {fund: payload.get("details", {}) for fund, payload in normalized.items()}}
    ReconciliationReport(structured_details, date_str, excel_path)

    return GeneratedReconciliationReport(str(excel_path))
