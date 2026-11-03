"""Shared PDF rendering helpers for the reporting package."""

from __future__ import annotations

import os
from typing import Tuple
import unicodedata

from fpdf import FPDF


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
        self.output_path = output_path
        self.pdf = FPDF(orientation=orientation, unit=unit, format=format)
        self.pdf.set_auto_page_break(auto=True, margin=12)
        self.pdf.set_margins(10, 12, 10)
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=10)

    # ------------------------------------------------------------------
    def _sanitize_text(self, value: object) -> str:
        text = "" if value is None else str(value)
        text = text.translate(_SANITIZE_MAP)
        try:
            text.encode("latin-1")
        except UnicodeEncodeError:
            text = unicodedata.normalize("NFKD", text)
            text = text.encode("latin-1", "replace").decode("latin-1")
        return text

    # ------------------------------------------------------------------
    def output(self) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        self.pdf.output(self.output_path)

_SANITIZE_MAP = {
    ord("\u00a0"): " ",  # non-breaking space
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
    ord("\u2015"): "-",  # horizontal bar
    ord("\u2212"): "-",  # minus sign
    ord("\u2018"): "'",  # left single quotation mark
    ord("\u2019"): "'",  # right single quotation mark
    ord("\u201a"): "'",  # single low-9 quotation mark
    ord("\u2032"): "'",  # prime
    ord("\u201c"): '"',  # left double quotation mark
    ord("\u201d"): '"',  # right double quotation mark
    ord("\u201e"): '"',  # double low-9 quotation mark
    ord("\u2033"): '"',  # double prime
    ord("\u2026"): "...",  # ellipsis
}