"""Small HTML builders for the pages (styled by :mod:`ui.theme`); every value is escaped here.

They return strings, so they can be unit-tested and combined into one
``st.html`` call (one element per card grid keeps the page light). Tables are
HTML rather than ``st.dataframe``: Streamlit's Arrow serialisation needs a
pyarrow build that matches the project's NumPy 1.26 pin, and the installed
pyarrow 26 needs NumPy 2.
"""
from __future__ import annotations

import html
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import streamlit as st

STATUS_CLASS = {"PASS": "pass", "OK": "pass", "FAIL": "fail", "FAILED": "fail", "WARN": "warn", "N/A": "na",
                "ERROR": "err", "RUNNING": "run"}


def esc(value: object) -> str:
    """``value`` as escaped text (``None`` -> '')."""
    return html.escape("" if value is None else str(value))


def summary_text(summary: object) -> str:
    """A run summary (``ui.runner.summarize``: fields joined by `` | ``) for display, joined by middle dots."""
    return str(summary or "").replace(" | ", "  ·  ")


def show(markup: str) -> None:
    """Render a built piece of HTML."""
    st.html(markup)


def status_class(status: Optional[str]) -> str:
    return STATUS_CLASS.get(str(status or "").upper(), "none")


def pill(status: Optional[str], text: Optional[str] = None, title: Optional[str] = None,
         small: bool = False) -> str:
    """A rounded status pill: PASS soft green, FAIL soft red, WARN amber, N/A grey; '—' when there is no status."""
    if not status and text is None:
        return '<span class="fm-dash">—</span>'
    tip = f' title="{esc(title)}"' if title else ""
    size = " sm" if small else ""
    return f'<span class="fm-pill{size} fm-{status_class(status)}"{tip}>{esc(text if text is not None else status)}</span>'


def section(title: str, count: Optional[int] = None, aside: str = "") -> str:
    """A settings-panel heading such as ``Select Fund (3)``; ``aside`` is a short right-hand note."""
    badge = f'<span class="fm-count">{count}</span>' if count is not None else ""
    note = f"<small>{esc(aside)}</small>" if aside else ""
    return f'<div class="fm-sec"><h3>{esc(title)}{badge}</h3>{note}</div>'


def divider() -> str:
    return '<div class="fm-rule"></div>'


def heading(title: str, aside: str = "") -> str:
    """A content heading in the right pane / results (``aside`` in small muted text)."""
    return f'<div class="fm-h2"><h2>{esc(title)}</h2><span>{esc(aside)}</span></div>'


def page_head(title: str, subtitle: str = "") -> str:
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    return f'<div class="fm-page-head"><h1>{esc(title)}</h1>{sub}</div>'


def metric_cards(cards: Iterable[Tuple[object, str, str]], accent: Sequence[int] = ()) -> str:
    """A row of metric cards ``(value, label, small note)``; the indices in ``accent`` get the accent colour."""
    items = []
    for i, (value, label, note) in enumerate(cards):
        cls = "fm-metric accent" if i in accent else "fm-metric"
        sub = f'<div class="s" title="{esc(note)}">{esc(note)}</div>' if note else ""
        items.append(f'<div class="{cls}"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div>{sub}</div>')
    return f'<div class="fm-metrics">{"".join(items)}</div>'


_CALLOUT_ICONS = {"error": "!", "warn": "!", "info": "i", "ok": "✓"}


def callout(kind: str, message: str, items: Sequence[str] = ()) -> str:
    """An inline message box (``kind``: error, warn, info, ok) with an optional bullet list."""
    bullets = "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>" if items else ""
    return (f'<div class="fm-callout {esc(kind)}"><span class="ic">{_CALLOUT_ICONS.get(kind, "i")}</span>'
            f"<div>{esc(message)}{bullets}</div></div>")


def empty(title: str, text: str) -> str:
    return f'<div class="fm-empty"><b>{esc(title)}</b>{esc(text)}</div>'


def card(title: str, body: str, status: Optional[str] = None, subtitle: str = "", plain: bool = False,
         footer: str = "") -> str:
    """A card with a header band (title, optional subtitle and status pill), a body and an optional footer.

    ``body`` and ``footer`` are already-built HTML.
    """
    sub = f' <span class="sub">{esc(subtitle)}</span>' if subtitle else ""
    right = pill(status) if status else ""
    band = "fm-band plain" if plain else "fm-band"
    return (f'<div class="fm-card"><div class="{band}"><span>{esc(title)}{sub}</span>{right}</div>'
            f'<div class="fm-body">{body}</div>{footer}</div>')


def grid(cards: Iterable[str], wide: Iterable[str] = ()) -> str:
    """Cards in rows of two, read in pairs (1 | 2, 3 | 4 ...) like the PDF's side-by-side tables, then the
    ``wide`` cards across both columns; one column, in order, when narrow.

    Each row's cards line up at the top, so a long card never leaves the other column running on alone.
    """
    cells = [f'<div class="fm-cell">{c}</div>' for c in cards]
    cells += [f'<div class="fm-cell wide">{c}</div>' for c in wide]
    return f'<div class="fm-grid">{"".join(cells)}</div>'


def split(left: str, right: str) -> str:
    """Two halves side by side inside a wide card (stacked when narrow); both are built HTML."""
    return f'<div class="fm-split"><div>{left}</div><div>{right}</div></div>'


_SHORT_FOOTNOTES = 320  # characters; longer footnotes start folded


def footnotes(title: str, notes: Sequence[str]) -> str:
    """The rule's footnotes under its card (folded when long)."""
    if not notes:
        return ""
    text = "".join(f"<p>{esc(n)}</p>" for n in notes)
    if sum(len(str(n)) for n in notes) <= _SHORT_FOOTNOTES:
        return f'<div class="fm-foot"><div class="t">{esc(title)}</div>{text}</div>'
    return f'<details class="fm-foot"><summary class="t">{esc(title)}</summary>{text}</details>'


_LONG_VALUE = 90  # characters; longer values (security lists) scroll inside their cell
_SHORT_VALUE = 24  # characters; shorter values never wrap (the label wraps instead)
_BLANKS = ("", "nan", "none", "nat")
MINUS = "−"
DASH = '<span class="fm-dash">—</span>'


def is_blank(value: object) -> bool:
    """A missing value as the reports' data carry it: ``None``, '' or a pandas 'nan' / 'None' turned to text."""
    return value is None or str(value).strip().lower() in _BLANKS


def cell_text(value: object, numeric: bool = False) -> str:
    """``value`` escaped for a table cell: a faint dash when blank; a true minus sign in a negative number.

    The minus is display-only: with tabular figures a hyphen is figure-wide and reads as '- 846,720'.
    """
    if is_blank(value):
        return DASH
    text = str(value)
    if numeric and "-" in text[:2]:
        text = text.replace("-", MINUS, 1)
    return esc(text)


def kv_table(rows: Iterable[Mapping[str, object]]) -> str:
    """Label / value rows like the PDF's metric tables.

    A row's ``status`` colours its value (an ``informational`` FAIL stays plain, as in the PDF) and its
    ``reason`` is shown under the value. Numbers get tabular figures, short values never wrap (the label
    does) and blank values show as a faint dash.
    """
    out = []
    for row in rows:
        value, status = row.get("value"), row.get("status")
        classes = []
        if status and not row.get("informational"):
            classes.append(f"v-{status_class(status)}")
        numeric = not is_blank(value) and _looks_numeric(str(value))
        if numeric:
            classes.append("num")
        reason = row.get("reason")
        has_reason = bool(reason) and str(reason) != str(value)
        if not has_reason and len(str(value or "")) <= _SHORT_VALUE:
            classes.append("nw")
        text = cell_text(value, numeric)
        if value is not None and len(str(value)) > _LONG_VALUE:
            text = f'<span class="long">{text}</span>'
        if has_reason:
            text += f'<span class="why">{esc(reason)}</span>'
        cls = f' class="{" ".join(classes)}"' if classes else ""
        out.append(f"<tr><th>{esc(row.get('label'))}</th><td{cls}>{text}</td></tr>")
    return f'<table class="fm-kv">{"".join(out)}</table>' if out else ""


def subhead(text: str) -> str:
    return f'<div class="fm-sub">{esc(text)}</div>'


def details(summary: str, body: str, open_: bool = False) -> str:
    """A native disclosure (``<details>``) for secondary content."""
    return f'<details{" open" if open_ else ""}><summary>{esc(summary)}</summary>{body}</details>'


# Display names for the raw snake_case columns of the holdings discrepancy tables (the PDF prints them raw).
_COLUMN_LABELS = {"eqy_ticker": "Ticker", "eqyticker": "Ticker", "discrepancy_type": "Discrepancy"}
_COLUMN_SIDES = {"cust": "Cust", "vest": "Vest", "index": "Index", "oms": "OMS"}


def column_label(name: object) -> str:
    """A column header for display: ``eqy_ticker`` -> ``Ticker``, ``shares_cust`` -> ``Shares (Cust)``.

    Only lower-case snake_case names change; headers that are already words are kept as they are.
    """
    text = "" if name is None else str(name)
    if text in _COLUMN_LABELS:
        return _COLUMN_LABELS[text]
    if "_" not in text or text != text.lower():
        return text
    words = [w for w in text.split("_") if w]
    side = _COLUMN_SIDES.get(words[-1]) if len(words) > 1 else None
    if side:
        words = words[:-1]
    label = " ".join(w.upper() if w in ("nav", "mv", "oms", "cusip", "isin") else w.capitalize() for w in words)
    return f"{label} ({side})" if side else label


def _looks_numeric(text: str) -> bool:
    stripped = text.replace(",", "").replace("$", "").replace("%", "").replace("-", "").replace(".", "").strip()
    return bool(stripped) and stripped.isdigit()


def data_table(columns: Sequence[str], rows: Iterable[Sequence[object]], *, total: Optional[int] = None,
               title: str = "", caption: str = "", row_classes: Sequence[str] = (), raw: Sequence[int] = (),
               css: str = "") -> str:
    """A plain table; numeric-looking cells are right-aligned. ``total`` > rows shown adds a 'first n of total' note.

    Cells in the ``raw`` column indices are already-built HTML (pills); every other cell is escaped, a blank
    one ('', 'nan', 'None') shown as a faint dash. snake_case headers are shown as words (:func:`column_label`).
    """
    rows = [list(r) for r in rows]
    numeric = [bool(rows) and all(_looks_numeric(str(r[i])) or is_blank(r[i]) or str(r[i]) == "—"
                                  for r in rows if i < len(r)) and i not in raw for i in range(len(columns))]
    head = "".join(f'<th class="{"num" if numeric[i] else "c" if i in raw else ""}">{esc(column_label(c))}</th>'
                   for i, c in enumerate(columns))
    body = []
    for n, row in enumerate(rows):
        cells = []
        for i, value in enumerate(row):
            cls = "num" if i < len(numeric) and numeric[i] else "c" if i in raw else ""
            text = value if i in raw else cell_text(value, i < len(numeric) and numeric[i])
            cells.append(f'<td class="{cls}">{text}</td>')
        rcls = f' class="{row_classes[n]}"' if n < len(row_classes) and row_classes[n] else ""
        body.append(f"<tr{rcls}>{''.join(cells)}</tr>")
    note = caption
    if total is not None and total > len(rows):
        note = (f"Showing the first {len(rows)} of {total}; the full list is in the report. " + caption).strip()
    top = f'<div class="fm-tabletitle">{esc(title)}</div>' if title else ""
    cap = f'<div class="fm-tablecap">{esc(note)}</div>' if note else ""
    return (f'{top}<div class="fm-tablewrap"><table class="fm-table {esc(css)}"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>{cap}')


def panel(body: str) -> str:
    """A bordered surface for a full-width table."""
    return f'<div class="fm-panel">{body}</div>'


def stats(items: Iterable[Tuple[str, object, str]]) -> str:
    """Small ``label value`` chips; the third item is an extra class (``bad``, ``warnish`` or '')."""
    chips = "".join(f'<span class="fm-stat {esc(cls)}">{esc(label)}<b>{esc(value)}</b></span>' for label, value, cls in items)
    return f'<div class="fm-stats">{chips}</div>'
