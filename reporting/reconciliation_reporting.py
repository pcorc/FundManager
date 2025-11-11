"""Holdings reconciliation reporting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.fund_definitions import FUND_DEFINITIONS
from reporting.base_report_pdf import BaseReportPDF
from reporting.holdings_recon_renderer import HoldingsReconciliationRenderer
from reporting.report_utils import normalize_reconciliation_payload, normalize_report_date
from utilities.reconciliation_utils import split_flex_price_frames


@dataclass
class GeneratedReconciliationReport:
    """Container for holdings reconciliation artefact locations."""

    excel_path: Optional[str]
    pdf_path: Optional[str]

@dataclass(frozen=True)
class ReconDescriptor:
    """Declarative definition of how to flatten reconciliation payloads."""

    holdings_key: str | None = None
    holdings_ticker: str | None = None
    holdings_builder: str | None = None
    require_holdings: bool = True
    price_keys: Tuple[str | None, str | None] = ("price_discrepancies_T", "price_discrepancies_T1")
    price_ticker: str | None = None
    price_cust_col: str = "price_cust"
    price_transform: str | None = None
    extra_callbacks: Sequence[str] = ()


RECON_DESCRIPTOR_REGISTRY: Dict[str, ReconDescriptor] = {
    "custodian_equity": ReconDescriptor(
        holdings_key="final_recon",
        holdings_ticker="equity_ticker",
        price_ticker="equity_ticker",
    ),
    "custodian_equity_t1": ReconDescriptor(
        holdings_key="final_recon",
        holdings_ticker="equity_ticker",
        price_keys=(None, None),
    ),
    "custodian_option": ReconDescriptor(
        holdings_key="final_recon",
        holdings_ticker="optticker",
        price_ticker="optticker",
        price_transform="_select_standard_option_prices",
        extra_callbacks=("_append_option_breakdowns",),
    ),
    "custodian_option_t1": ReconDescriptor(
        holdings_key="final_recon",
        holdings_ticker="optticker",
        price_keys=(None, None),
        extra_callbacks=("_append_option_t1_breakdowns",),
    ),
    "index_equity": ReconDescriptor(
        holdings_key="holdings_discrepancies",
        holdings_ticker="equity_ticker",
        holdings_builder="_prepare_index_holdings_df",
        price_ticker="equity_ticker",
        price_cust_col="price_index",
    ),
    "sg_option": ReconDescriptor(
        holdings_key="final_recon",
        holdings_ticker="optticker",
        price_keys=("price_discrepancies", None),
        price_ticker="optticker",
    ),
    "sg_equity": ReconDescriptor(
        holdings_key="final_recon",
        holdings_ticker="equity_ticker",
        price_keys=("price_discrepancies", None),
        price_ticker="equity_ticker",
    ),
    "custodian_treasury": ReconDescriptor(
        holdings_key="final_recon",
        holdings_ticker="cusip",
        price_keys=("price_discrepancies_T", None),
        price_ticker="cusip",
        require_holdings=False,
    ),
    "custodian_treasury_t1": ReconDescriptor(
        holdings_key="final_recon",
        holdings_ticker="cusip",
        price_keys=(None, "price_discrepancies_T1"),
        price_ticker="cusip",
        require_holdings=False,
    ),
}


class ReconResult:
    """Standardized container used when flattening reconciliation payloads."""

    def __init__(
        self,
        fund: str,
        date_str: str,
        recon_type: str,
        holdings_df: pd.DataFrame | None = None,
        price_df_t: pd.DataFrame | None = None,
        price_df_t1: pd.DataFrame | None = None,
    ) -> None:
        self.fund = fund
        self.date_str = date_str
        self.recon_type = recon_type
        self.holdings_df = holdings_df
        self.price_df_t = price_df_t
        self.price_df_t1 = price_df_t1

    def to_flat_dict(self) -> Dict[Tuple[str, str, str], pd.DataFrame]:
        flat: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        if self.holdings_df is not None and not self.holdings_df.empty:
            flat[(self.fund, self.date_str, self.recon_type)] = self.holdings_df
        if self.price_df_t is not None and not self.price_df_t.empty:
            flat[(self.fund, self.date_str, f"{self.recon_type}_price_T")] = self.price_df_t
        if self.price_df_t1 is not None and not self.price_df_t1.empty:
            flat[(self.fund, self.date_str, f"{self.recon_type}_price_T-1")] = self.price_df_t1
        return flat

    def has_data(self) -> bool:
        return any(
            [
                self.holdings_df is not None and not self.holdings_df.empty,
                self.price_df_t is not None and not self.price_df_t.empty,
                self.price_df_t1 is not None and not self.price_df_t1.empty,
            ]
        )


class ReconciliationReport:
    """Generate the enhanced Excel holdings reconciliation report."""

    def __init__(
        self,
        reconciliation_results: Mapping[str, Mapping[str, Any]],
        recon_summary: Iterable[Dict[str, Any]] | None,
        date: str,
        file_path_excel: str | Path,
    ) -> None:
        self.reconciliation_results = reconciliation_results
        self.recon_summary = list(recon_summary or [])
        self.date = str(date)
        self.output_path = Path(file_path_excel)
        self.flattened_results = self._filter_and_flatten_results()
        self._export_to_excel()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _filter_and_flatten_results(self) -> Dict[Tuple[str, str, str], pd.DataFrame]:
        flat: Dict[Tuple[str, str, str], pd.DataFrame] = {}

        for date_str, fund_data in self.reconciliation_results.items():
            for fund, recon_dict in (fund_data or {}).items():
                for recon_type, subresults in (recon_dict or {}).items():
                    descriptor = RECON_DESCRIPTOR_REGISTRY.get(recon_type)
                    if not descriptor:
                        continue

                    result = self._process_from_descriptor(
                        descriptor,
                        fund,
                        date_str,
                        recon_type,
                        subresults or {},
                    )
                    if result and result.has_data():
                        flat.update(result.to_flat_dict())

                    for callback_name in descriptor.extra_callbacks:
                        callback = getattr(self, callback_name, None)
                        if callback:
                            callback(flat, fund, date_str, subresults or {})
        return flat

    def _process_from_descriptor(
        self,
        descriptor: ReconDescriptor,
        fund: str,
        date_str: str,
        recon_type: str,
        subresults: Mapping[str, Any],
    ) -> ReconResult | None:
        """Build a :class:`ReconResult` from the provided descriptor."""

        holdings_df: pd.DataFrame | None = None
        if descriptor.holdings_key:
            holdings_raw = self._ensure_dataframe(subresults.get(descriptor.holdings_key))
            if not holdings_raw.empty:
                holdings_df = self._build_holdings_from_descriptor(
                    descriptor,
                    holdings_raw,
                    fund,
                    date_str,
                    recon_type,
                )
        elif descriptor.require_holdings:
            # If holdings are required but no key is defined, treat as no data.
            return None

        price_t, price_t1 = self._build_price_dfs(
            descriptor,
            subresults,
            fund,
            date_str,
            recon_type,
        )

        if not any([holdings_df is not None and not holdings_df.empty, price_t is not None, price_t1 is not None]):
            if descriptor.require_holdings:
                return None

        return ReconResult(
            fund,
            date_str,
            recon_type,
            holdings_df,
            price_t,
            price_t1,
        )

    def _append_option_breakdowns(
        self,
        flat: Dict[Tuple[str, str, str], pd.DataFrame],
        fund: str,
        date_str: str,
        subresults: Mapping[str, Any],
    ) -> None:
        regular_df = self._ensure_dataframe(subresults.get("regular_options"))
        if not regular_df.empty:
            key = (fund, date_str, "custodian_option_regular")
            flat[key] = self._prepare_holdings_df(regular_df, fund, date_str, "custodian_option_regular", "optticker")

        flex_df = self._ensure_dataframe(subresults.get("flex_options"))
        if not flex_df.empty:
            prepared = self._prepare_holdings_df(flex_df, fund, date_str, "custodian_option_flex", "optticker")
            flat[(fund, date_str, "custodian_option_flex")] = prepared
            flat[(fund, date_str, "custodian_option_flex_holdings")] = prepared

        price_t_df = self._ensure_dataframe(subresults.get("price_discrepancies_T"))
        flex_price_t, _ = split_flex_price_frames(price_t_df)
        if not flex_price_t.empty:
            prepared = self._prepare_price_df(
                flex_price_t,
                fund,
                date_str,
                "custodian_option_flex_price_T",
                "optticker",
            )
            flat[(fund, date_str, "custodian_option_flex_price_T")] = prepared

        price_t1_df = self._ensure_dataframe(subresults.get("price_discrepancies_T1"))
        flex_price_t1, _ = split_flex_price_frames(price_t1_df)
        if not flex_price_t1.empty:
            prepared = self._prepare_price_df(
                flex_price_t1,
                fund,
                date_str,
                "custodian_option_flex_price_T-1",
                "optticker",
            )
            flat[(fund, date_str, "custodian_option_flex_price_T-1")] = prepared

    def _append_option_t1_breakdowns(
        self,
        flat: Dict[Tuple[str, str, str], pd.DataFrame],
        fund: str,
        date_str: str,
        subresults: Mapping[str, Any],
    ) -> None:
        flex_df = self._ensure_dataframe(subresults.get("flex_options"))
        if not flex_df.empty:
            prepared = self._prepare_holdings_df(
                flex_df,
                fund,
                date_str,
                "custodian_option_flex_holdings_t1",
                "optticker",
            )
            flat[(fund, date_str, "custodian_option_flex_holdings_t1")] = prepared

    def _build_holdings_from_descriptor(
        self,
        descriptor: ReconDescriptor,
        df: pd.DataFrame,
        fund: str,
        date_str: str,
        recon_type: str,
    ) -> pd.DataFrame | None:
        builder_name = descriptor.holdings_builder or "_prepare_holdings_df"
        builder: Callable[..., pd.DataFrame | None] | None = getattr(self, builder_name, None)
        if builder is None:
            return None

        ticker_col = descriptor.holdings_ticker or descriptor.price_ticker
        if not ticker_col:
            return builder(df, fund, date_str, recon_type, "TICKER")
        return builder(df, fund, date_str, recon_type, ticker_col)

    def _build_price_dfs(
        self,
        descriptor: ReconDescriptor,
        subresults: Mapping[str, Any],
        fund: str,
        date_str: str,
        recon_type: str,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        ticker_col = descriptor.price_ticker or descriptor.holdings_ticker
        if not ticker_col:
            return None, None

        price_t: pd.DataFrame | None = None
        price_t1: pd.DataFrame | None = None
        price_t_key, price_t1_key = descriptor.price_keys

        if price_t_key:
            price_t_raw = self._ensure_dataframe(subresults.get(price_t_key))
            price_t_raw = self._apply_price_transform(price_t_raw, descriptor)
            if not price_t_raw.empty:
                price_t = self._prepare_price_df(
                    price_t_raw,
                    fund,
                    date_str,
                    recon_type,
                    ticker_col,
                    descriptor.price_cust_col,
                )

        if price_t1_key:
            price_t1_raw = self._ensure_dataframe(subresults.get(price_t1_key))
            price_t1_raw = self._apply_price_transform(price_t1_raw, descriptor)
            if not price_t1_raw.empty:
                price_t1 = self._prepare_price_df(
                    price_t1_raw,
                    fund,
                    date_str,
                    recon_type,
                    ticker_col,
                    descriptor.price_cust_col,
                )

        return price_t, price_t1

    def _apply_price_transform(
        self, df: pd.DataFrame, descriptor: ReconDescriptor
    ) -> pd.DataFrame:
        if df.empty or not descriptor.price_transform:
            return df

        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = getattr(
            self, descriptor.price_transform, None
        )
        if transform is None:
            return df
        transformed = transform(df)
        return transformed if isinstance(transformed, pd.DataFrame) else df

    def _select_standard_option_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        _flex, standard = split_flex_price_frames(df)
        return standard

    def _prepare_index_holdings_df(
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
        if "discrepancy_type" not in df.columns:
            df["discrepancy_type"] = "Holdings Mismatch"

        cols_to_keep = [
            col
            for col in [
                "FUND",
                "RECON_TYPE",
                "DATE",
                ticker_col,
                "discrepancy_type",
                "in_vest",
                "in_index",
            ]
            if col in df.columns
        ]
        return df[cols_to_keep] if cols_to_keep else df

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



    def _prepare_price_df(
        self,
        df: pd.DataFrame,
        fund: str,
        date_str: str,
        recon_type: str,
        ticker_col: str,
        price_cust_col: str = "price_cust",
    ) -> pd.DataFrame:
        df = df.copy()
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
            price_cust_col,
            "price_diff",
        ]
        cols = [c for c in cols if c in df.columns]
        return df[cols] if cols else df

    def _ensure_dataframe(self, value: Any) -> pd.DataFrame:
        if value is None:
            return pd.DataFrame()
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, Mapping):
            return pd.DataFrame(value)
        return pd.DataFrame(value)

    # ------------------------------------------------------------------
    def _export_to_excel(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
            if not self.flattened_results:
                self._create_no_breaks_message(writer)
            else:
                self._create_summary_tab(writer)
                self._create_detailed_breaks_tab(writer)
                self._create_price_comparison_tab(writer)
                self._create_system_specific_tabs(writer)
                self._format_price_comparison_tab(writer)
            if writer.book.sheetnames:
                writer.book[writer.book.sheetnames[0]].sheet_state = "visible"

    def _create_no_breaks_message(self, writer: pd.ExcelWriter) -> None:
        pd.DataFrame(["All reconciliations passed - no breaks found"]).to_excel(
            writer,
            sheet_name="RECONCILIATION BREAKS SUMMARY",
            index=False,
        )

    def _create_summary_tab(self, writer: pd.ExcelWriter) -> None:
        funds = sorted(self._get_unique_funds())
        fund_data: Dict[str, Dict[str, Any]] = {}
        for fund in funds:
            row = self._build_summary_row(fund)
            row.pop("Fund", None)
            row.pop("Date", None)
            fund_data[fund] = row

        summary_df = pd.DataFrame(fund_data)
        date_row = pd.DataFrame({fund: [self.date] for fund in funds}, index=["Date"])
        summary_df = pd.concat([date_row, summary_df])
        summary_df.index.name = "Reconciliation Type"
        summary_df.to_excel(
            writer,
            sheet_name="RECONCILIATION BREAKS SUMMARY",
            index=True,
            startrow=0,
        )

        worksheet = writer.sheets["RECONCILIATION BREAKS SUMMARY"]
        has_private_or_closed = any(
            (FUND_DEFINITIONS.get(f, {}).get("vehicle_wrapper") in {"private_fund", "closed_end_fund"})
            for f in funds
        )
        if has_private_or_closed:
            note_row = len(summary_df) + 3
            cell = worksheet.cell(row=note_row, column=1)
            cell.value = (
                "Note: For private/closed-end funds, standard option price breaks show top 5 by weight. "
                "All FLEX option breaks are shown."
            )
            font = cell.font.copy(italic=True, size=9)
            cell.font = font
        worksheet.column_dimensions["A"].width = 45

    def _build_summary_row(self, fund: str) -> Dict[str, Any]:
        row = {"Fund": fund, "Date": self.date}
        columns = [
            "custodian_equity_holdings_breaks_T",
            "custodian_equity_holdings_breaks_T_1",
            "custodian_equity_price_breaks_T",
            "custodian_equity_price_breaks_T_1",
            "custodian_option_holdings_breaks_T",
            "custodian_option_holdings_breaks_T_1",
            "custodian_option_regular_breaks_T",
            "custodian_option_flex_breaks_T",
            "custodian_option_price_breaks_T",
            "custodian_option_price_breaks_T_1",
            "flex_option_holdings_breaks_T",
            "flex_option_holdings_breaks_T_1",
            "flex_option_price_breaks_T",
            "flex_option_price_breaks_T_1",
            "custodian_treasury_holdings_breaks_T",
            "custodian_treasury_holdings_breaks_T_1",
            "custodian_treasury_price_breaks_T",
            "custodian_treasury_price_breaks_T_1",
            "index_equity_holdings_breaks",
            "index_equity_price_breaks_T",
            "index_equity_price_breaks_T_1",
            "sg_option_breaks",
            "sg_equity_breaks",
        ]
        for col in columns:
            row[col] = 0

        for (fund_name, _date, recon_key), df in self.flattened_results.items():
            if fund_name != fund:
                continue
            column_name = self._map_recon_key_to_summary(recon_key)
            if column_name:
                row[column_name] += len(df)
        return row

    def _map_recon_key_to_summary(self, recon_type: str) -> str | None:
        mapping = {
            "custodian_equity": "custodian_equity_holdings_breaks_T",
            "custodian_equity_t1": "custodian_equity_holdings_breaks_T_1",
            "custodian_equity_price_T": "custodian_equity_price_breaks_T",
            "custodian_equity_price_T-1": "custodian_equity_price_breaks_T_1",
            "custodian_option": "custodian_option_holdings_breaks_T",
            "custodian_option_t1": "custodian_option_holdings_breaks_T_1",
            "custodian_option_regular": "custodian_option_regular_breaks_T",
            "custodian_option_flex": "custodian_option_flex_breaks_T",
            "custodian_option_price_T": "custodian_option_price_breaks_T",
            "custodian_option_price_T-1": "custodian_option_price_breaks_T_1",
            "custodian_option_flex_holdings": "flex_option_holdings_breaks_T",
            "custodian_option_flex_holdings_t1": "flex_option_holdings_breaks_T_1",
            "custodian_option_flex_price_T": "flex_option_price_breaks_T",
            "custodian_option_flex_price_T-1": "flex_option_price_breaks_T_1",
            "custodian_treasury": "custodian_treasury_holdings_breaks_T",
            "custodian_treasury_t1": "custodian_treasury_holdings_breaks_T_1",
            "custodian_treasury_price_T": "custodian_treasury_price_breaks_T",
            "custodian_treasury_price_T-1": "custodian_treasury_price_breaks_T_1",
            "index_equity": "index_equity_holdings_breaks",
            "index_equity_price_T": "index_equity_price_breaks_T",
            "index_equity_price_T-1": "index_equity_price_breaks_T_1",
            "sg_option": "sg_option_breaks",
            "sg_equity": "sg_equity_breaks",
        }
        return mapping.get(recon_type)

    def _create_detailed_breaks_tab(self, writer: pd.ExcelWriter) -> None:
        all_breaks: list[pd.DataFrame] = []
        for (fund, date_str, recon_type), df in self.flattened_results.items():
            if "_price_" in recon_type:
                continue
            df_copy = df.copy()
            df_copy["Asset Type"] = self._determine_asset_type(recon_type, df_copy)
            all_breaks.append(df_copy)
        if not all_breaks:
            return
        detailed_breaks = pd.concat(all_breaks, ignore_index=True)
        detailed_breaks = self._standardize_ticker_column(detailed_breaks)

        cols = list(detailed_breaks.columns)
        if "Asset Type" in cols:
            cols.remove("Asset Type")
            ticker_idx = next((i for i, c in enumerate(cols) if c == "TICKER"), 3)
            cols.insert(min(ticker_idx + 1, len(cols)), "Asset Type")
            detailed_breaks = detailed_breaks[cols]
        detailed_breaks.to_excel(writer, sheet_name="DETAILED BREAKS BY SECURITY", index=False)

    def _create_price_comparison_tab(self, writer: pd.ExcelWriter) -> None:
        price_comparison: list[pd.DataFrame] = []
        for (fund, date_str, recon_type), df in self.flattened_results.items():
            if "_price_" not in recon_type:
                continue
            df_copy = df.copy()
            df_copy["FUND"] = fund
            df_copy["DATE"] = date_str
            df_copy["SOURCE"] = self._extract_source_from_recon_type(recon_type)
            df_copy["PERIOD"] = self._extract_period_from_recon_type(recon_type)
            price_comparison.append(df_copy)
        if not price_comparison:
            return
        all_prices = pd.concat(price_comparison, ignore_index=True)
        comparison_df = self._build_price_comparison_df(all_prices)
        comparison_df.to_excel(writer, sheet_name="COMPREHENSIVE PRICE COMPARISON", index=False)

    def _determine_asset_type(self, recon_type: str, df: pd.DataFrame) -> str:
        recon_lower = recon_type.lower()
        if "flex" in recon_lower:
            return "FLEX"
        if "treasury" in recon_lower:
            return "Treasury"
        if "equity" in recon_lower:
            return "Equity"
        if "option" in recon_lower:
            return "Option"
        if "cusip" in df.columns:
            return "Treasury"
        if "equity_ticker" in df.columns or "eqyticker" in df.columns:
            return "Equity"
        if "optticker" in df.columns or "occ_symbol" in df.columns:
            tickers = df.get("optticker") or df.get("occ_symbol")
            if isinstance(tickers, pd.Series) and tickers.str.contains("SPX|XSP", na=False).any():
                return "FLEX"
            return "Option"
        return "Unknown"

    def _build_price_comparison_df(self, all_prices: pd.DataFrame) -> pd.DataFrame:
        ticker_col = "TICKER" if "TICKER" in all_prices.columns else "equity_ticker"
        rows: list[Dict[str, Any]] = []
        for (fund, ticker, period), group in all_prices.groupby(["FUND", ticker_col, "PERIOD"]):
            row: Dict[str, Any] = {
                "Fund": fund,
                "Ticker": ticker,
                "Period": period,
                "Fund_Price": np.nan,
                "Custodian_Price": np.nan,
                "Index_Price": np.nan,
                "Price_Diff": np.nan,
                "Asset Type": "Unknown",
                "Option Weight": "",
                "% Difference": "",
            }

            is_flex = False
            if "is_flex" in group.columns and group["is_flex"].any():
                is_flex = True
            if group["RECON_TYPE"].str.contains("flex", case=False, na=False).any():
                is_flex = True
            if is_flex:
                row["Asset Type"] = "FLEX"

            for _, price_row in group.iterrows():
                source = price_row.get("SOURCE", "")
                recon_name = price_row.get("RECON_TYPE", "")
                fund_price = price_row.get("price_vest")
                price_diff = price_row.get("price_diff")
                lower_source = str(source).lower()
                lower_recon = str(recon_name).lower()
                if row["Asset Type"] == "Unknown":
                    if "treasury" in lower_source or "treasury" in lower_recon:
                        row["Asset Type"] = "Treasury"
                    elif "equity" in lower_source or "equity" in lower_recon:
                        row["Asset Type"] = "Equity"
                    elif "option" in lower_source or "option" in lower_recon:
                        row["Asset Type"] = "Option"
                if "custodian" in str(source):
                    row["Fund_Price"] = fund_price
                    row["Custodian_Price"] = price_row.get("price_cust")
                    row["Price_Diff"] = price_diff
                elif "index" in str(source):
                    row["Fund_Price"] = fund_price
                    row["Index_Price"] = price_row.get("price_index", price_row.get("price_cust"))
                    row["Price_Diff"] = price_diff
                if "option_weight" in price_row:
                    weight = price_row.get("option_weight")
                    if isinstance(weight, (int, float)) and weight:
                        row["Option Weight"] = f"{weight * 100:.2f}%"
                if "price_pct_diff" in price_row:
                    pct = price_row.get("price_pct_diff")
                    if isinstance(pct, (int, float)):
                        row["% Difference"] = f"{pct:.1f}%"

            prices = [p for p in [row["Fund_Price"], row["Custodian_Price"], row["Index_Price"]] if pd.notna(p)]
            if prices:
                row["Price_Diff"] = max(prices) - min(prices)
            rows.append(row)
        comparison_df = pd.DataFrame(rows)
        sort_cols = ["Fund", "Asset Type", "Period"]
        comparison_df = comparison_df.sort_values(sort_cols)
        return comparison_df

    def _create_system_specific_tabs(self, writer: pd.ExcelWriter) -> None:
        system_tabs = {
            "INDEX_RECONCILIATION": ["index_equity"],
            "SG_RECONCILIATION": ["sg_equity", "sg_option"],
        }
        for tab_name, recon_types in system_tabs.items():
            filtered = [df for (fund, date, recon), df in self.flattened_results.items() if recon in recon_types]
            if not filtered:
                continue
            tab_df = pd.concat(filtered, ignore_index=True)
            tab_df = self._standardize_ticker_column(tab_df)
            tab_df.to_excel(writer, sheet_name=tab_name, index=False)

    def _get_unique_funds(self) -> set[str]:
        return {fund for (fund, _date, _recon) in self.flattened_results.keys()}

    def _standardize_ticker_column(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["equity_ticker", "optticker", "norm_ticker", "eqyticker", "occ_symbol"]:
            if col in df.columns:
                return df.rename(columns={col: "TICKER"})
        return df

    def _extract_source_from_recon_type(self, recon_type: str) -> str:
        recon_lower = recon_type.lower()
        if "custodian_equity" in recon_lower:
            return "custodian_equity"
        if "custodian_option" in recon_lower:
            return "custodian_option"
        if "custodian_treasury" in recon_lower:
            return "custodian_treasury"
        if "index_equity" in recon_lower:
            return "index_equity"
        return "unknown"

    def _extract_period_from_recon_type(self, recon_type: str) -> str:
        return "T-1" if "T-1" in recon_type else "T"

    def _format_price_comparison_tab(self, writer: pd.ExcelWriter) -> None:
        if "COMPREHENSIVE PRICE COMPARISON" not in writer.sheets:
            return
        from openpyxl.styles import Alignment, Font, PatternFill

        worksheet = writer.sheets["COMPREHENSIVE PRICE COMPARISON"]
        header_fill = PatternFill(start_color="D3E4F7", end_color="D3E4F7", fill_type="solid")
        weight_fill = PatternFill(start_color="E2F0D9", end_color="E2F0D9", fill_type="solid")

        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        weight_idx = None
        pct_idx = None
        for idx, cell in enumerate(worksheet[1], start=1):
            if cell.value == "Option Weight":
                weight_idx = idx
            if cell.value == "% Difference":
                pct_idx = idx

        if weight_idx:
            col_letter = worksheet.cell(row=1, column=weight_idx).column_letter
            for row in range(2, worksheet.max_row + 1):
                cell = worksheet.cell(row=row, column=weight_idx)
                if cell.value:
                    cell.fill = weight_fill
                    cell.alignment = Alignment(horizontal="right")
        if pct_idx:
            for row in range(2, worksheet.max_row + 1):
                cell = worksheet.cell(row=row, column=pct_idx)
                if cell.value:
                    cell.alignment = Alignment(horizontal="right")

        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)


class ReconciliationReportPDF(BaseReportPDF, HoldingsReconciliationRenderer):
    """Generate the enhanced holdings reconciliation PDF report."""

    def __init__(
        self,
        reconciliation_results: Mapping[str, Mapping[str, Any]] | None,
        recon_summary: Iterable[Dict[str, Any]] | None,
        date: str | None,
        file_path_pdf: str | Path | None = None,
        *,
        output_path: str | Path | None = None,
    ) -> None:
        output_file = file_path_pdf or output_path or f"reconciliation_report_{date}.pdf"
        super().__init__(str(output_file))
        self.reconciliation_results = reconciliation_results or {}
        self.recon_summary = list(recon_summary or [])
        self.date = str(date) if date else ""

        try:
            self.flattened_results = self._flatten_reconciliation_results()
            self._generate_pdf()
        except Exception as exc:  # pragma: no cover - defensive
            self._generate_error_report(str(exc))

    def _flatten_reconciliation_results(self) -> Dict[Tuple[str, str], Mapping[str, Any]]:
        flattened: Dict[Tuple[str, str], Mapping[str, Any]] = {}
        for date_str, fund_data in self.reconciliation_results.items():
            for fund_name, recon_dict in (fund_data or {}).items():
                flattened[(fund_name, date_str)] = recon_dict
        return flattened

    def _generate_error_report(self, error_message: str) -> None:
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 10, "Reconciliation Report - ERROR", ln=True, align="C")
        self.pdf.set_font("Arial", size=12)
        self.pdf.multi_cell(0, 10, f"An error occurred while generating the report: {error_message}")
        self.output()

    def _generate_pdf(self) -> None:
        self.pdf.add_page()
        self._add_header("Reconciliation Report", f"Report Date: {self.date}")
        self._add_consolidated_summary()
        if not self.flattened_results:
            self.pdf.set_font("Arial", "B", 12)
            self.pdf.cell(0, 10, "No reconciliation results available", ln=True, align="C")
        else:
            for (fund_name, date_str), recon_data in sorted(self.flattened_results.items()):
                self.render_fund_holdings_section(fund_name, date_str, recon_data)
        self.output()

    def _add_header(self, title: str, subtitle: str) -> None:
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 10, title, ln=True, align="C")
        self.pdf.set_font("Arial", "", 11)
        self.pdf.cell(0, 8, subtitle, ln=True, align="C")
        self.pdf.ln(6)

    def _add_consolidated_summary(self) -> None:
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 8, "Reconciliation Breaks Summary", ln=True)
        self.pdf.ln(2)

        all_summaries: Dict[str, Mapping[str, Any]] = {}
        for summary in self.recon_summary:
            if isinstance(summary, Mapping) and "fund" in summary:
                all_summaries[summary["fund"]] = summary.get("summary", {})

        if not all_summaries:
            return

        cols = [
            ("Fund", 30),
            ("Index\nHoldings", 18),
            ("Index\nWgt Diff", 18),
            ("Cust Eq\nHold Brk", 18),
            ("Cust Eq\nPrice T", 18),
            ("Cust Eq\nPrice T-1", 18),
            ("Cust Opt\nHold Brk", 18),
            ("Cust Opt\nPrice T", 18),
            ("Cust Opt\nPrice T-1", 18),
            ("Cust Tsy\nHold Brk", 18),
            ("Cust Tsy\nPrice T", 18),
            ("Cust Tsy\nPrice T-1", 18),
        ]

        self.pdf.set_font("Arial", "B", 6)
        self.pdf.set_fill_color(240, 240, 240)
        max_lines = max(len(header.split("\n")) for header, _ in cols)
        start_y = self.pdf.get_y()
        line_height = 3.5

        x_pos = self.pdf.l_margin
        for header, width in cols:
            lines = header.split("\n")
            for i, line in enumerate(lines):
                self.pdf.set_xy(x_pos, start_y + (i * line_height))
                self.pdf.cell(width, line_height, line, border=1, fill=True, align="C")
            x_pos += width
        self.pdf.set_y(start_y + (max_lines * line_height))

        self.pdf.set_font("Arial", size=7)
        for fund in sorted(all_summaries):
            summary = all_summaries[fund]
            y_pos = self.pdf.get_y()
            self.pdf.set_font("Arial", "B", 7)
            self.pdf.set_xy(self.pdf.l_margin, y_pos)
            self.pdf.cell(cols[0][1], 6, fund, border=1, align="C")

            values = [
                summary.get("index_equity", {}).get("holdings_discrepancies", 0),
                summary.get("index_equity", {}).get("significant_diffs", 0),
                summary.get("custodian_equity", {}).get("final_recon", 0),
                summary.get("custodian_equity", {}).get("price_discrepancies_T", 0),
                summary.get("custodian_equity", {}).get("price_discrepancies_T1", 0),
                summary.get("custodian_option", {}).get("final_recon", 0),
                summary.get("custodian_option", {}).get("price_discrepancies_T", 0),
                summary.get("custodian_option", {}).get("price_discrepancies_T1", 0),
                summary.get("custodian_treasury", {}).get("final_recon", 0),
                summary.get("custodian_treasury", {}).get("price_discrepancies_T", 0),
                summary.get("custodian_treasury", {}).get("price_discrepancies_T1", 0),
            ]

            self.pdf.set_font("Arial", size=7)
            x_pos = self.pdf.l_margin + cols[0][1]
            for val, (_, width) in zip(values, cols[1:]):
                if isinstance(val, (int, float)) and val > 0:
                    self.pdf.set_fill_color(255, 200, 200)
                else:
                    self.pdf.set_fill_color(255, 255, 255)
                self.pdf.set_xy(x_pos, y_pos)
                self.pdf.cell(width, 6, str(int(val)), border=1, fill=True, align="C")
                x_pos += width
            self.pdf.set_y(y_pos + 6)

        self.pdf.ln(4)
        self.pdf.set_font("Arial", "I", 7)
        self.pdf.set_text_color(100, 100, 100)
        self.pdf.cell(0, 4, "Red highlight indicates breaks found | Hold Brk = Holdings Breaks", ln=1)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(6)


def _structure_reconciliation_details(
    normalized_results: Mapping[str, Dict[str, Any]],
    date_str: str,
) -> Dict[str, Dict[str, Any]]:
    details_by_date: Dict[str, Dict[str, Any]] = {date_str: {}}
    for fund, payload in normalized_results.items():
        details_by_date[date_str][fund] = payload.get("details", {})
    return details_by_date


def _build_summary_records(
    normalized_results: Mapping[str, Dict[str, Any]],
    date_str: str,
) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    for fund, payload in normalized_results.items():
        records.append({"fund": fund, "date": date_str, "summary": payload.get("summary", {})})
    return records


def generate_reconciliation_reports(
    results: Mapping[str, Any],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    file_name_prefix: str = "reconciliation_results",
    create_pdf: bool = True,
) -> GeneratedReconciliationReport:
    normalized = normalize_reconciliation_payload(results)
    if not normalized:
        return GeneratedReconciliationReport(None, None)

    date_str = normalize_report_date(report_date)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    excel_path = output_path / f"{file_name_prefix}_{date_str}.xlsx"
    pdf_path = output_path / f"{file_name_prefix}_{date_str}.pdf"

    structured_details = _structure_reconciliation_details(normalized, date_str)
    summary_records = _build_summary_records(normalized, date_str)

    ReconciliationReport(structured_details, summary_records, date_str, excel_path)

    pdf_result: Optional[str] = None
    if create_pdf:
        ReconciliationReportPDF(structured_details, summary_records, date_str, file_path_pdf=str(pdf_path))
        pdf_result = str(pdf_path)

    return GeneratedReconciliationReport(str(excel_path), pdf_result)