"""Reusable PDF rendering helpers for holdings reconciliation reports."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

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

        self._ensure_page_space(20)
        self._draw_section_header(f"Fund: {fund_name}", subtitle=f"Date: {date_str}")

        ordered = sorted(recon_data.items(), key=lambda item: item[0])
        for recon_type, payload in ordered:
            if not isinstance(payload, Mapping) or not payload:
                continue
            self._ensure_page_space(15)
            self._draw_subsection_header(recon_type)
            self._print_recon_section(recon_type, payload, fund_name=fund_name)

    # ------------------------------------------------------------------
    def _draw_section_header(self, title: str, *, subtitle: str | None = None) -> None:
        self.pdf.set_fill_color(230, 230, 230)
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, title, ln=True, fill=True)
        if subtitle:
            self.pdf.set_font("Arial", "", 10)
            self.pdf.cell(0, 6, subtitle, ln=True)
        self.pdf.ln(4)

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
        if recon_type == "index_equity":
            self._print_index_equity_section(recon_data)
            return
        if recon_type == "custodian_equity":
            self._print_custodian_equity_section(recon_data)
            return
        if recon_type == "custodian_option":
            self._print_custodian_option_section(
                recon_data,
                fund_name=fund_name,
                is_flex=False,
            )
            return
        if recon_type == "custodian_option_t1":
            self._print_custodian_option_section(
                recon_data,
                fund_name=fund_name,
                is_flex=False,
                is_t1=True,
            )
            return

        df = recon_data.get("final_recon", pd.DataFrame())
        if not isinstance(df, pd.DataFrame) or df.empty:
            self._draw_two_column_table([("Status", "No reconciliation data available")])
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
        if isinstance(final_df, pd.DataFrame) and not final_df.empty:
            counts["Total"] = len(final_df)
            if "discrepancy_type" in final_df.columns:
                for disc_type in final_df["discrepancy_type"]:
                    disc_lower = str(disc_type).lower()
                    if "missing" in disc_lower or "holdings" in disc_lower:
                        counts["Holdings"] += 1
                    elif "quantity" in disc_lower or "mismatch" in disc_lower:
                        counts["Quantity"] += 1

        price_t = recon_data.get("price_discrepancies_T")
        price_t1 = recon_data.get("price_discrepancies_T1")
        if isinstance(price_t, pd.DataFrame):
            counts["Price"] += len(price_t)
            counts["Total"] += len(price_t)
        if isinstance(price_t1, pd.DataFrame):
            counts["Price"] += len(price_t1)
            counts["Total"] += len(price_t1)

        holdings_df = recon_data.get("holdings_discrepancies")
        if isinstance(holdings_df, pd.DataFrame) and not holdings_df.empty:
            counts["Holdings"] += len(holdings_df)

        weight_df = recon_data.get("significant_diffs")
        if isinstance(weight_df, pd.DataFrame) and not weight_df.empty:
            counts["Weight"] += len(weight_df)

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
        if isinstance(final_df, pd.DataFrame) and not final_df.empty:
            title = f"Final Reconciliation - {recon_type}" if recon_type else "Final Reconciliation"
            self._print_discrepancy_table(final_df, title)

    def _print_discrepancy_table(self, df: pd.DataFrame, title: str) -> None:
        if df.empty:
            return

        ticker_col = next(
            (
                col
                for col in ["equity_ticker", "optticker", "occ_symbol", "norm_ticker", "ticker"]
                if col in df.columns
            ),
            None,
        )
        if not ticker_col:
            return

        display_cols = [ticker_col]
        for col in [
            "discrepancy_type",
            "shares_cust",
            "trade_discrepancy",
            "final_discrepancy",
            "price_vest",
            "price_cust",
            "price_diff",
        ]:
            if col in df.columns:
                display_cols.append(col)

        col_width = (self.pdf.w - 2 * self.pdf.l_margin) / len(display_cols)
        self.pdf.set_font("Arial", "B", 8)
        self.pdf.set_fill_color(200, 200, 200)
        for col in display_cols:
            self.pdf.cell(col_width, 6, str(col), border=1, fill=True, align="C")
        self.pdf.ln()

        self.pdf.set_font("Arial", size=8)
        for _, row in df.head(10).iterrows():
            for col in display_cols:
                val = row[col]
                if isinstance(val, bool):
                    text = "YES" if val else "NO"
                elif isinstance(val, (int, float, np.number)):
                    text = f"{val:,.2f}"
                else:
                    text = str(val)
                self.pdf.cell(col_width, 5, text, border=1, align="R")
            self.pdf.ln()

        if len(df) > 10:
            self.pdf.set_font("Arial", "I", 8)
            self.pdf.cell(0, 5, f"... and {len(df) - 10} more rows", ln=1)

    # ------------------------------------------------------------------
    def _print_index_equity_section(self, recon_data: Mapping[str, Any]) -> None:
        holdings_df = recon_data.get("holdings_discrepancies")
        if isinstance(holdings_df, pd.DataFrame) and not holdings_df.empty:
            self._draw_two_column_table([("Holdings Breaks", len(holdings_df))])
            self._print_discrepancy_table(holdings_df, "Holdings Discrepancies")
        else:
            self._draw_two_column_table([("Holdings Breaks", 0)])

        price_t = recon_data.get("price_discrepancies_T")
        price_t1 = recon_data.get("price_discrepancies_T1")
        self._print_price_discrepancies(price_t, price_t1, price_label="Index")

    def _print_custodian_equity_section(self, recon_data: Mapping[str, Any]) -> None:
        final_df = recon_data.get("final_recon")
        if isinstance(final_df, pd.DataFrame) and not final_df.empty:
            self._draw_two_column_table([("Holdings Breaks", len(final_df))])
            self._print_discrepancy_table(final_df, "Holdings Discrepancies")
        else:
            self._draw_two_column_table([("Holdings Breaks", 0)])

        price_t = recon_data.get("price_discrepancies_T")
        price_t1 = recon_data.get("price_discrepancies_T1")
        self._print_price_discrepancies(price_t, price_t1, price_label="Custodian")

    def _print_custodian_option_section(
        self,
        recon_data: Mapping[str, Any],
        *,
        fund_name: str,
        is_flex: bool,
        is_t1: bool = False,
    ) -> None:
        final_df = recon_data.get("final_recon")
        if isinstance(final_df, pd.DataFrame) and not final_df.empty:
            rows = [("Holdings Breaks", len(final_df))]
        else:
            rows = [("Holdings Breaks", 0)]
        regular_df = recon_data.get("regular_options")
        flex_df = recon_data.get("flex_options")
        if isinstance(regular_df, pd.DataFrame) and not regular_df.empty:
            rows.append(("Regular Options", len(regular_df)))
        if isinstance(flex_df, pd.DataFrame) and not flex_df.empty:
            rows.append(("FLEX Options", len(flex_df)))
        self._draw_two_column_table(rows)

        if isinstance(final_df, pd.DataFrame) and not final_df.empty:
            self._print_discrepancy_table(final_df, "Option Discrepancies")

        price_t = recon_data.get("price_discrepancies_T")
        price_t1 = recon_data.get("price_discrepancies_T1")
        self._print_price_discrepancies(price_t, price_t1, price_label="Custodian")

    # ------------------------------------------------------------------
    def _print_price_discrepancies(
        self,
        price_t: Any,
        price_t1: Any,
        *,
        price_label: str,
    ) -> None:
        rows = []
        if isinstance(price_t, pd.DataFrame) and not price_t.empty:
            rows.append((f"{price_label} Price Breaks T", len(price_t)))
        if isinstance(price_t1, pd.DataFrame) and not price_t1.empty:
            rows.append((f"{price_label} Price Breaks T-1", len(price_t1)))
        if rows:
            self._draw_two_column_table(rows)
            if isinstance(price_t, pd.DataFrame) and not price_t.empty:
                self._print_discrepancy_table(price_t, f"{price_label} Price Breaks T")
            if isinstance(price_t1, pd.DataFrame) and not price_t1.empty:
                self._print_discrepancy_table(price_t1, f"{price_label} Price Breaks T-1")
        else:
            self._draw_two_column_table([(f"{price_label} Price Breaks", 0)])