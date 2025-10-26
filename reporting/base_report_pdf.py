"""Shared PDF rendering helpers for reporting modules."""

from __future__ import annotations

import os
from typing import Iterable, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from fpdf import FPDF
except Exception as exc:  # pragma: no cover
    FPDF = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover - set to None when import succeeds
    _IMPORT_ERROR = None


class BaseReportPDF:
    """Light-weight wrapper around :class:`fpdf.FPDF` with common helpers."""

    DEFAULT_FONT = ("Arial", "", 10)

    def __init__(self, output_path: str) -> None:
        if FPDF is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "The optional dependency 'fpdf2' is required to generate PDFs. "
                "Install it with `pip install fpdf2`."
            ) from _IMPORT_ERROR

        self.output_path = output_path
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.set_margins(12, 15, 12)
        self.pdf.add_page()
        self._set_font(*self.DEFAULT_FONT)

    # ------------------------------------------------------------------
    # General helpers
    # ------------------------------------------------------------------
    def _set_font(self, family: str, style: str = "", size: int = 10) -> None:
        self.pdf.set_font(family, style=style, size=size)

    def _sanitize_text(self, value: object) -> str:
        text = "" if value is None else str(value)
        return text.replace("\u2013", "-")

    def add_title(self, title: str, subtitle: str | None = None) -> None:
        self._set_font("Arial", "B", 16)
        self.pdf.cell(0, 12, self._sanitize_text(title), ln=True)
        if subtitle:
            self._set_font("Arial", "", 11)
            self.pdf.cell(0, 8, self._sanitize_text(subtitle), ln=True)
        self.pdf.ln(4)
        self._set_font(*self.DEFAULT_FONT)

    def add_section_heading(self, heading: str) -> None:
        self._set_font("Arial", "B", 12)
        self.pdf.cell(0, 9, self._sanitize_text(heading), ln=True)
        self.pdf.ln(1)
        self._set_font(*self.DEFAULT_FONT)

    def add_paragraph(self, text: str) -> None:
        sanitized = self._sanitize_text(text)
        self.pdf.multi_cell(0, 6, sanitized)
        self.pdf.ln(1)

    def add_bullet_list(self, items: Sequence[str]) -> None:
        if not items:
            return
        self._set_font("Arial", size=9)
        for item in items:
            self.pdf.cell(5, 6, "\u2022")
            self.pdf.multi_cell(0, 6, self._sanitize_text(item))
        self.pdf.ln(1)
        self._set_font(*self.DEFAULT_FONT)

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------
    def add_key_value_table(
        self,
        rows: Iterable[Tuple[object, object]],
        *,
        label_width: float = 80,
        value_width: float = 60,
        row_height: float = 6,
        header: Tuple[str, str] | None = None,
    ) -> None:
        """Render a two-column table with optional header row."""

        self._set_font("Arial", size=9)
        if header:
            self._set_font("Arial", "B", 9)
            self.pdf.cell(label_width, row_height, self._sanitize_text(header[0]), border=1)
            self.pdf.cell(value_width, row_height, self._sanitize_text(header[1]), border=1)
            self.pdf.ln(row_height)
            self._set_font("Arial", size=9)

        for label, value in rows:
            self.pdf.cell(label_width, row_height, self._sanitize_text(label), border=1)
            self.pdf.cell(value_width, row_height, self._sanitize_text(value), border=1, align="R")
            self.pdf.ln(row_height)
        self.pdf.ln(2)
        self._set_font(*self.DEFAULT_FONT)

    def add_table(
        self,
        headers: Sequence[str],
        rows: Iterable[Sequence[object]],
        *,
        column_widths: Sequence[float] | None = None,
        row_height: float = 6,
        header_style: Tuple[str, str, int] = ("Arial", "B", 9),
        cell_style: Tuple[str, str, int] = ("Arial", "", 9),
        alignments: Sequence[str] | None = None,
    ) -> None:
        """Render a table with arbitrary number of columns."""

        if column_widths is None:
            width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin
            column_widths = [width / len(headers)] * len(headers)

        self._set_font(*header_style)
        for header, width in zip(headers, column_widths):
            self.pdf.cell(width, row_height, self._sanitize_text(header), border=1, align="C")
        self.pdf.ln(row_height)

        self._set_font(*cell_style)
        for row in rows:
            for idx, (value, width) in enumerate(zip(row, column_widths)):
                align = "L"
                if alignments and idx < len(alignments):
                    align = alignments[idx]
                self.pdf.cell(width, row_height, self._sanitize_text(value), border=1, align=align)
            self.pdf.ln(row_height)
        self.pdf.ln(2)
        self._set_font(*self.DEFAULT_FONT)

    # ------------------------------------------------------------------
    def add_spacer(self, height: float = 4.0) -> None:
        self.pdf.ln(height)

    def output(self) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.pdf.output(self.output_path)