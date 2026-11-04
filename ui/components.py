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


def grid(cards: Iterable[str]) -> str:
    """Cards in two columns, read in pairs (1 | 2, 3 | 4 …) like the PDF; one column, in order, when narrow.

    The cards alternate between the columns, so each column packs its cards without gaps; ``order`` keeps
    the sequence when the columns collapse into one.
    """
    cards = list(cards)
    left = "".join(f'<div class="fm-cell" style="order:{i}">{c}</div>' for i, c in enumerate(cards) if i % 2 == 0)
    right = "".join(f'<div class="fm-cell" style="order:{i}">{c}</div>' for i, c in enumerate(cards) if i % 2 == 1)
    return f'<div class="fm-grid"><div class="fm-col">{left}</div><div class="fm-col">{right}</div></div>'


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


def kv_table(rows: Iterable[Mapping[str, object]]) -> str:
    """Label / value rows like the PDF's metric tables.

    A row's ``status`` colours its value (an ``informational`` FAIL stays plain, as in the PDF) and its
    ``reason`` is shown under the value.
    """
    out = []
    for row in rows:
        value, status = row.get("value"), row.get("status")
        cls = ""
        if status and not row.get("informational"):
            cls = f' class="v-{status_class(status)}"'
        text = esc(value)
        if value is not None and len(str(value)) > _LONG_VALUE:
            text = f'<span class="long">{text}</span>'
        reason = row.get("reason")
        if reason and str(reason) != str(value):
            text += f'<span class="why">{esc(reason)}</span>'
        out.append(f"<tr><th>{esc(row.get('label'))}</th><td{cls}>{text}</td></tr>")
    return f'<table class="fm-kv">{"".join(out)}</table>' if out else ""


def subhead(text: str) -> str:
    return f'<div class="fm-sub">{esc(text)}</div>'


def details(summary: str, body: str, open_: bool = False) -> str:
    """A native disclosure (``<details>``) for secondary content."""
    return f'<details{" open" if open_ else ""}><summary>{esc(summary)}</summary>{body}</details>'


def _looks_numeric(text: str) -> bool:
    stripped = text.replace(",", "").replace("$", "").replace("%", "").replace("-", "").replace(".", "").strip()
    return bool(stripped) and stripped.isdigit()


def data_table(columns: Sequence[str], rows: Iterable[Sequence[object]], *, total: Optional[int] = None,
               title: str = "", caption: str = "", row_classes: Sequence[str] = (), raw: Sequence[int] = (),
               css: str = "") -> str:
    """A plain table; numeric-looking cells are right-aligned. ``total`` > rows shown adds a 'first n of total' note.

    Cells in the ``raw`` column indices are already-built HTML (pills); every other cell is escaped.
    """
    rows = [list(r) for r in rows]
    numeric = [bool(rows) and all(_looks_numeric(str(r[i])) or str(r[i]) in ("", "nan", "—")
                                  for r in rows if i < len(r)) and i not in raw for i in range(len(columns))]
    head = "".join(f'<th class="{"num" if numeric[i] else "c" if i in raw else ""}">{esc(c)}</th>'
                   for i, c in enumerate(columns))
    body = []
    for n, row in enumerate(rows):
        cells = []
        for i, value in enumerate(row):
            cls = "num" if i < len(numeric) and numeric[i] else "c" if i in raw else ""
            cells.append(f'<td class="{cls}">{value if i in raw else esc(value)}</td>')
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
