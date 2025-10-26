"""Shared PDF rendering helpers for the reporting package."""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from fpdf import FPDF
except Exception as exc:  # pragma: no cover
    FPDF = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover - set to None when import succeeds
    _IMPORT_ERROR = None


class BaseReportPDF:
    """Thin wrapper around :class:`fpdf.FPDF` with higher level helpers."""

    def __init__(
        self,
        output_path: str,
        *,
        orientation: str = "P",
        unit: str = "mm",
        format: str | Tuple[float, float] = "Letter",
    ) -> None:
        if FPDF is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "The optional dependency 'fpdf2' is required to generate PDFs. "
                "Install it with `pip install fpdf2`."
            ) from _IMPORT_ERROR

        self.output_path = output_path
        self.pdf = FPDF(orientation=orientation, unit=unit, format=format)
        self.pdf.set_auto_page_break(auto=True, margin=12)
        self.pdf.set_margins(10, 12, 10)
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=10)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def add_page(self, *, title: str | None = None, subtitle: str | None = None) -> None:
        """Create a new page optionally including a title/subtitle."""

        self.pdf.add_page()
        if title:
            self.add_title(title)
        if subtitle:
            self.add_subtitle(subtitle)

    def add_title(self, text: str, *, font_size: int = 14) -> None:
        self.pdf.set_font("Arial", "B", font_size)
        self.pdf.cell(0, 10, self._sanitize_text(text), ln=True)
        self.add_spacer(2)
        self.pdf.set_font("Arial", size=10)

    def add_subtitle(self, text: str, *, font_size: int = 11) -> None:
        self.pdf.set_font("Arial", "B", font_size)
        self.pdf.cell(0, 8, self._sanitize_text(text), ln=True)
        self.add_spacer(1)
        self.pdf.set_font("Arial", size=9)

    def add_paragraph(
        self,
        text: str,
        *,
        font_size: int = 9,
        bold: bool = False,
        ln: bool = True,
    ) -> None:
        style = "B" if bold else ""
        self.pdf.set_font("Arial", style, font_size)
        self.pdf.multi_cell(0, 5, self._sanitize_text(text))
        if ln:
            self.add_spacer(2)
        self.pdf.set_font("Arial", size=9)

    def add_spacer(self, height: float = 4) -> None:
        self.pdf.ln(height)

    def add_table(
        self,
        headers: Sequence[str],
        rows: Iterable[Sequence[object]],
        *,
        column_widths: Sequence[float] | None = None,
        header_fill: bool = True,
        align: Sequence[str] | None = None,
        font_size: int = 8,
    ) -> None:
        """Render a table with optional header row."""

        widths = self._resolve_column_widths(len(headers), column_widths)
        alignments = self._resolve_alignments(len(headers), align)

        self.pdf.set_font("Arial", "B", font_size)
        if header_fill:
            self.pdf.set_fill_color(230, 230, 230)
        for header, width, cell_align in zip(headers, widths, alignments):
            self.pdf.cell(width, 6, self._sanitize_text(header), border=1, align=cell_align, fill=header_fill)
        self.pdf.ln(6)

        self.pdf.set_font("Arial", size=font_size)
        self.pdf.set_fill_color(255, 255, 255)
        for row in rows:
            for value, width, cell_align in zip(row, widths, alignments):
                self.pdf.cell(width, 6, self._sanitize_text(value), border=1, align=cell_align)
            self.pdf.ln(6)
        self.add_spacer(3)

    # ------------------------------------------------------------------
    def _sanitize_text(self, value: object) -> str:
        text = "" if value is None else str(value)
        return text.replace("\u2013", "-")

    def _resolve_column_widths(
        self,
        column_count: int,
        column_widths: Sequence[float] | None,
    ) -> List[float]:
        if column_widths and len(column_widths) == column_count:
            return list(column_widths)

        usable_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin
        default_width = usable_width / max(column_count, 1)
        return [default_width] * column_count

    def _resolve_alignments(
        self,
        column_count: int,
        align: Sequence[str] | None,
    ) -> List[str]:
        if align and len(align) == column_count:
            return [a.upper()[:1] or "L" for a in align]
        return ["L"] * column_count

    # ------------------------------------------------------------------
    def output(self) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        self.pdf.output(self.output_path)