"""Shared PDF rendering helpers for compliance style reports."""

from __future__ import annotations

import os
from typing import Iterable, Tuple

try:  # pragma: no cover - optional dependency
    from fpdf import FPDF
except Exception as exc:  # pragma: no cover
    FPDF = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover - set to None when import succeeds
    _IMPORT_ERROR = None


class BaseReportPDF:
    """Light-weight wrapper around :class:`fpdf.FPDF`."""

    def __init__(self, output_path: str) -> None:
        if FPDF is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "The optional dependency 'fpdf2' is required to generate PDFs. "
                "Install it with `pip install fpdf2`."
            ) from _IMPORT_ERROR

        self.output_path = output_path
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=12)
        self.pdf.set_margins(10, 12, 10)
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=10)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _sanitize_text(self, value: object) -> str:
        text = "" if value is None else str(value)
        return text.replace("\u2013", "-")

    def _add_header(self, title: str) -> None:
        self.pdf.set_font("Arial", "B", 14)
        self.pdf.cell(0, 10, self._sanitize_text(title), ln=True)
        self.pdf.ln(4)
        self.pdf.set_font("Arial", size=10)

    def _draw_two_column_table(
        self,
        rows: Iterable[Tuple[object, object]],
        label_width: float = 80,
        value_width: float = 90,
        row_height: float = 6,
    ) -> None:
        self.pdf.set_font("Arial", size=9)
        for label, value in rows:
            self.pdf.cell(label_width, row_height, self._sanitize_text(label), border=0)
            self.pdf.cell(value_width, row_height, self._sanitize_text(value), border=0, align="R")
            self.pdf.ln(row_height)
        self.pdf.ln(2)

    # ------------------------------------------------------------------
    def output(self) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.pdf.output(self.output_path)