"""Reusable PDF rendering helpers for holdings reconciliation reports."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence
from config.fund_definitions import vehicle_wrapper
from reporting.base_report_pdf import draw_header_row
from reporting.reconciliation_reporting import top_option_prices
from reporting.report_utils import non_empty

import numpy as np
import pandas as pd


class HoldingsReconciliationRenderer:
    """Mixin that provides reusable holdings reconciliation PDF sections."""

    pdf: Any  # populated by subclasses (either BaseReportPDF or raw FPDF)

    # ------------------------------------------------------------------
    def render_fund_holdings_section(
        self,
        fund_name: str,
        date_str: str,
        recon_data: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Render all reconciliation sections for a given fund."""
        if not self._fund_has_holdings_data(recon_data):
            return

        self._ensure_page_space(20)
        self._draw_section_header(f"Fund: {fund_name}", subtitle=f"Date: {date_str}")

        ordered = sorted(recon_data.items(), key=lambda item: item[0])
        for recon_type, payload in ordered:
            if not isinstance(payload, Mapping) or not payload:
                continue
            self._ensure_page_space(15)
            self._draw_subsection_header(recon_type)
            self._print_recon_section(recon_type, payload, fund_name=fund_name)
            if recon_type in {"custodian_equity", "custodian_option", "custodian_flex_option"}:
                self.pdf.ln(3)
            else:
                self.pdf.ln(1)

    # ------------------------------------------------------------------
    def _draw_section_header(
        self, title: str, *, subtitle: str | None = None, size: int = 14, height: float = 10, gap: float = 4
    ) -> None:
        self.pdf.set_fill_color(230, 230, 230)
        self.pdf.set_font("Arial", "B", size)
        self.pdf.cell(0, height, title, ln=True, fill=True)
        if subtitle:
            self.pdf.set_font("Arial", "", 10)
            self.pdf.cell(0, 6, subtitle, ln=True)
        self.pdf.ln(gap)

    def _draw_subsection_header(self, recon_type: str) -> None:
        label = recon_type.replace("_", " ").title()
        self.pdf.set_fill_color(240, 240, 240)
        self.pdf.set_font("Arial", "B", 11)
        self.pdf.cell(0, 8, label, ln=True, fill=True)
        self.pdf.ln(2)

    def _ensure_page_space(self, min_space: float) -> None:
        if self.pdf.get_y() + min_space >= (self.pdf.h - self.pdf.b_margin):
            self.pdf.add_page()

    # ------------------------------------------------------------------
    def _print_recon_section(
        self,
        recon_type: str,
        recon_data: Mapping[str, Any],
        *,
        fund_name: str,
    ) -> None:
        if recon_type == "custodian_equity":
            self._print_custodian_equity_section(recon_data)
            return
        if recon_type == "custodian_option":
            self._print_custodian_option_section(recon_data, fund_name=fund_name)
            return
        if recon_type == "custodian_treasury":
            if self._has_recon_activity(recon_data):
                self._print_custodian_treasury_section(recon_data)
            return

        df = recon_data.get("final_recon", pd.DataFrame())
        if not non_empty(df):
            # Distinguish "ran cleanly, no breaks" from "genuinely no data".
            merged = recon_data.get("merged_data")
            price_t = recon_data.get("price_discrepancies_T")
            ran = non_empty(merged) or non_empty(price_t)
            msg = "All reconciliations successful - no breaks found" if ran else "No reconciliation data available"
            self._draw_two_column_table([("Status", msg)])
            self.pdf.ln(4)
            return

        counts = self._count_discrepancies(recon_data)
        rows = [(f"Total {k} Discrepancies", v) for k, v in counts.items()]
        if sum(counts.values()) > 0:
            self._draw_two_column_table(rows)
            self._print_recon_details(recon_data, recon_type=recon_type)
        else:
            self._draw_two_column_table([("Status", "No discrepancies found")])
        self.pdf.ln(4)

    # ------------------------------------------------------------------
    def _count_discrepancies(self, recon_data: Mapping[str, Any]) -> dict[str, int]:
        counts = {"Total": 0, "Holdings": 0, "Quantity": 0, "Weight": 0, "Price": 0}

        final_df = recon_data.get("final_recon")
        if non_empty(final_df):
            counts["Total"] = len(final_df)
            if "discrepancy_type" in final_df.columns:
                for disc_type in final_df["discrepancy_type"]:
                    disc_lower = str(disc_type).lower()
                    if "missing" in disc_lower or "holdings" in disc_lower:
                        counts["Holdings"] += 1
                    elif "quantity" in disc_lower or "mismatch" in disc_lower:
                        counts["Quantity"] += 1

        price_t = recon_data.get("price_discrepancies_T")
        if isinstance(price_t, pd.DataFrame):
            counts["Price"] += len(price_t)
            counts["Total"] += len(price_t)
        return counts

    # ------------------------------------------------------------------
    def _draw_two_column_table(self, rows: Iterable[Sequence[Any]]) -> None:
        column_widths = [60, 30]
        self.pdf.set_font("Arial", "", 9)
        for label, value in rows:
            self.pdf.cell(column_widths[0], 6, str(label), border=1)
            self.pdf.cell(column_widths[1], 6, str(value), border=1, align="R")
            self.pdf.ln()

    # ------------------------------------------------------------------
    def _print_recon_details(
        self,
        recon_data: Mapping[str, Any],
        *,
        recon_type: str | None = None,
    ) -> None:
        self.pdf.set_font("Arial", "B", 10)
        self.pdf.cell(0, 6, "Discrepancy Details:", ln=True)
        self.pdf.set_font("Arial", size=9)
        final_df = recon_data.get("final_recon")
        if non_empty(final_df):
            title = f"Final Reconciliation - {recon_type}" if recon_type else "Final Reconciliation"
            self._print_discrepancy_table(final_df, title)

    def _print_discrepancy_table(self, df: pd.DataFrame, title: str) -> None:
        if df.empty:
            return

        ticker_col = next(
            (
                col
                for col in ["equity_ticker", "eqyticker", "optticker", "occ_symbol", "norm_ticker", "ticker", "cusip"]
                if col in df.columns
            ),
            None,
        )

        # Price-break tables (caller passes a title containing "Price") only need
        # ticker + the two price columns. Holdings-discrepancy tables keep the
        # full column set.
        is_price_table = "Price" in title

        if is_price_table:
            column_sources = {
                "eqy_ticker": [ticker_col] if ticker_col else [],
                "price_vest": ["price_vest", "fund_price", "vest_price"],
                "price_cust": ["price_cust", "cust_price", "price_custodian"],
            }
            right_aligned = {"price_vest", "price_cust"}
        else:
            column_sources = {
                "eqy_ticker": [ticker_col] if ticker_col else [],
                "discrepancy_type": ["discrepancy_type", "type"],
                "shares_cust": ["shares_cust", "cust_shares", "custodian_shares"],
                "shares_vest": ["shares_vest", "nav_shares", "quantity", "shares_oms", "vest_shares"],
                "price_vest": ["price_vest", "fund_price", "vest_price"],
                "price_cust": ["price_cust", "cust_price", "price_custodian"],
            }
            right_aligned = {"shares_cust", "shares_vest", "price_vest", "price_cust"}

        resolved_columns = {}
        for display_name, candidates in column_sources.items():
            resolved_columns[display_name] = next((col for col in candidates if col in df.columns), None)

        if not resolved_columns.get("eqy_ticker"):
            return

        display_cols = list(column_sources.keys())
        usable_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin
        col_width = usable_width / len(display_cols)

        draw_header_row(self.pdf, [col_width] * len(display_cols), display_cols, 6, 8, (200, 200, 200))

        self.pdf.set_font("Arial", size=8)
        for _, row in df.head(10).iterrows():
            for col in display_cols:
                source = resolved_columns.get(col)
                val = row.get(source) if source else ""
                if isinstance(val, bool):
                    text = "YES" if val else "NO"
                elif isinstance(val, (int, float, np.number)):
                    text = f"{val:,.2f}"
                else:
                    text = str(val)
                align = "R" if col in right_aligned else "L"
                self.pdf.cell(col_width, 5, text[:20], border=1, align=align)
            self.pdf.ln()

        if len(df) > 10:
            self.pdf.set_font("Arial", "I", 8)
            self.pdf.cell(0, 5, f"... and {len(df) - 10} more rows", ln=1)

    # ------------------------------------------------------------------
    def _print_holdings_block(
        self, recon_data: Mapping[str, Any], title: str, extra_rows: Sequence[tuple[str, int]] = ()
    ) -> bool:
        final_df = recon_data.get("final_recon")
        has_holdings = non_empty(final_df)
        self._draw_two_column_table([("Holdings Breaks", len(final_df) if has_holdings else 0), *extra_rows])
        if has_holdings:
            self._print_discrepancy_table(final_df, title)
        return has_holdings

    def _print_custodian_equity_section(self, recon_data: Mapping[str, Any]) -> None:
        price_t = recon_data.get("price_discrepancies_T")
        has_holdings = self._print_holdings_block(recon_data, "Holdings Discrepancies")
        self._print_price_discrepancies(price_t, price_label="Custodian")
        if not has_holdings and not non_empty(price_t):
            self._draw_two_column_table([("Status", "All reconciliations successful - no breaks found")])

    def _print_custodian_option_section(self, recon_data: Mapping[str, Any], *, fund_name: str) -> None:
        regular_df = recon_data.get("regular_options")
        extra_rows = []
        if non_empty(regular_df):
            extra_rows.append(("Regular Options", len(regular_df)))
        self._print_holdings_block(recon_data, "Option Discrepancies", extra_rows)
        self._print_price_discrepancies(
            recon_data.get("price_discrepancies_T"),
            price_label="Custodian",
            fund_name=fund_name,
            recon_type="custodian_option",
        )

    def _print_custodian_treasury_section(self, recon_data: Mapping[str, Any]) -> None:
        self._print_holdings_block(recon_data, "Treasury Discrepancies")
        self._print_price_discrepancies(recon_data.get("price_discrepancies_T"), price_label="Custodian")

    # ------------------------------------------------------------------
    def _print_price_discrepancies(
        self,
        price_t: Any,
        *,
        price_label: str,
        fund_name: str | None = None,
        recon_type: str | None = None,
    ) -> None:
        if non_empty(price_t):
            filtered = self._filter_option_price_breaks(price_t, fund_name, recon_type)
            self._draw_two_column_table([(f"{price_label} Price Breaks T", len(filtered))])
            self._print_discrepancy_table(filtered, f"{price_label} Price Breaks T")
        else:
            self._draw_two_column_table([(f"{price_label} Price Breaks", 0)])

    _OPTION_PRICE_BREAK_MIN_ABS = 1.00

    def _filter_option_price_breaks(
            self,
            price_df: pd.DataFrame,
            fund_name: str | None,
            recon_type: str | None,
    ) -> pd.DataFrame:
        if not non_empty(price_df):
            return pd.DataFrame()

        if not fund_name or not recon_type or "option" not in recon_type.lower():
            return price_df

        working = price_df.copy()

        # Compute |price_vest - price_cust|. Prefer the pre-computed price_diff
        # column if present; otherwise derive it from the two price columns.
        if "price_diff" in working.columns:
            diff = pd.to_numeric(working["price_diff"], errors="coerce").abs()
        elif {"price_vest", "price_cust"}.issubset(working.columns):
            pv = pd.to_numeric(working["price_vest"], errors="coerce")
            pc = pd.to_numeric(working["price_cust"], errors="coerce")
            diff = (pv - pc).abs()
        else:
            diff = None

        if diff is not None:
            working = working[diff >= self._OPTION_PRICE_BREAK_MIN_ABS]

        if working.empty:
            return working

        if vehicle_wrapper(fund_name) not in {"private_fund", "closed_end_fund"}:
            return working
        return top_option_prices(working)

    def _fund_has_holdings_data(self, recon_data: Mapping[str, Mapping[str, Any]]) -> bool:
        for payload in recon_data.values():
            if not isinstance(payload, Mapping):
                continue
            if self._payload_has_content(payload):
                return True
        return False

    def _payload_has_content(self, payload: Mapping[str, Any]) -> bool:
        for value in payload.values():
            if non_empty(value):
                return True
            if isinstance(value, Mapping) and value:
                if self._payload_has_content(value):
                    return True
        return False

    def _has_recon_activity(self, recon_data: Mapping[str, Any]) -> bool:
        return any(non_empty(recon_data.get(key)) for key in ("final_recon", "price_discrepancies_T"))