"""Hand-built inline SVG charts for the Trading Compliance dashboard (no chart library: pyarrow 26 needs NumPy 2).

Each builder returns one HTML string: a ``<figure>`` with a title, a legend and,
for the per-fund charts, three aligned columns: the fund names (HTML), the
plot (an SVG whose x positions are percentages, so it stretches with its card
while text and marks keep their pixel size) and the value the row is about
(HTML). Rows are a fixed :data:`ROW` pixels, so the three per-fund charts line
up as small multiples when they share the same fund order.

Render them with :func:`ui.components.show_svg` (``st.html`` strips SVG).
Every text is escaped here; each mark carries a ``<title>`` (hover) and the
SVG an ``aria-label``; the trading summary table holds the same numbers (the
table view).

Colours (validated with the dataviz skill's ``validate_palette.js`` against
the card surface ``#262D3B``, dark mode, all pairs: lightness band, chroma,
CVD and normal-vision separation, 3:1 contrast): asset classes are
categorical slots blue / orange / aqua, and those hues mean an asset class
only. Totals (current and post-trade cash, all-asset traded %) use
non-categorical inks: a neutral slate (:data:`TOTAL`, 4.6:1 on the card) and
the text ink for the filled post-trade dot (:data:`POST`). Text always uses
the theme's text tokens.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from ui import components as ui

ASSET_COLORS = {"equity": "#6C8BE6", "options": "#D2774A", "treasury": "#31A27F"}
ASSET_LABELS = {"equity": "Equity", "options": "Options", "treasury": "Treasury"}
ASSET_UNITS = {"equity": "shares", "options": "contracts", "treasury": "face"}
ASSETS = ("equity", "options", "treasury")
TOTAL = "#8C96AA"  # start / end bars of the waterfall, all-asset traded % bars (not an asset-class hue)
POST = "#E3E7EF"  # the filled "post-trade cash" dot (theme TEXT ink: not an asset-class hue)
BEFORE = "#A3ACBD"  # the hollow "current cash" dot
SURFACE = "#262D3B"  # the card surface: gaps and rings
GRID = "#353E51"
AXIS = "#5A6480"
TICK = "#929CB0"  # theme FAINT
INK = "#B0B9CA"  # theme MUTED
NEG_WASH = "rgba(238,142,142,0.07)"

ROW = 22  # px per fund row
BAR = 12  # px bar thickness (<= 24)
AXIS_BAND = 20  # px for the tick labels above the plot
PLOT_LEFT, PLOT_RIGHT = 4.0, 96.0  # % of the SVG width the data may use
LEFT_OUT_SHOWN = 5  # funds named in a chart's "not plotted" note (largest first)
# Tick labels are kept apart for the narrowest plot a per-fund chart gets (about 125px, three charts across at 1600px).
TICK_PLOT_PX = 120.0
TICK_GAP_PX = 4.0
WIDE_PLOT_PX = 240.0  # labels that only fit a plot this wide are shown by a container query (ui.theme)
_CHAR_PX = {"−": 6.0, "-": 6.0, "%": 8.6, ".": 2.7, ",": 2.7, "M": 8.6, "K": 6.6, "B": 6.6, "<": 6.0}


def esc(value: object) -> str:
    return ui.esc(value)


def _num(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------

def nice_step(span: float, count: int = 4) -> float:
    """A 1 / 2 / 2.5 / 5 × 10^n step giving about ``count`` intervals over ``span``."""
    if span <= 0:
        return 1.0
    raw = span / max(count, 1)
    power = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * power:
            return factor * power
    return 10 * power


def nice_domain(values: Iterable[float], count: int = 4) -> Tuple[float, float, List[float]]:
    """(low, high, ticks) covering ``values`` and zero, rounded out to clean ticks."""
    numbers = [v for v in values if v is not None]
    low, high = min(numbers + [0.0]), max(numbers + [0.0])
    if low == high:
        high = low + 1.0
    step = nice_step(high - low, count)
    low, high = math.floor(low / step) * step, math.ceil(high / step) * step
    ticks, value = [], low
    while value <= high + step / 1e6:
        ticks.append(round(value, 10))
        value += step
    return low, high, ticks


class Scale:
    """Value -> percent of the SVG width, inside [PLOT_LEFT, PLOT_RIGHT]."""

    def __init__(self, low: float, high: float):
        self.low, self.high = low, high

    def __call__(self, value: float) -> float:
        share = (value - self.low) / (self.high - self.low) if self.high != self.low else 0.0
        return round(PLOT_LEFT + share * (PLOT_RIGHT - PLOT_LEFT), 3)


def tick_label(value: float, kind: str) -> str:
    """Axis text: clean and short (``$20M``, ``−$2.5M``, ``$500K``, ``10%``)."""
    if kind == "pct":
        return ui.pct(value, digits=0 if abs(value) >= 0.01 or value == 0 else 1)
    size = abs(value)
    for unit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if size >= unit:
            body = f"{size / unit:.2f}".rstrip("0").rstrip(".") + suffix
            break
    else:
        body = f"{size:,.0f}"
    return ("−" if value < 0 else "") + "$" + body


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

def legend(items: Sequence[Tuple[str, ...]], shape: str = "square") -> str:
    """``(colour, label[, shape])`` swatches; ``shape``: square (bars), dot, ring (hollow dot)."""
    keys = []
    for item in items:
        color, label = item[0], item[1]
        kind = item[2] if len(item) > 2 else shape
        if kind == "ring":
            mark = f'<i class="ring" style="border-color:{esc(color)}"></i>'
        else:
            mark = f'<i class="{esc(kind)}" style="background:{esc(color)}"></i>'
        keys.append(f"<span>{mark}{esc(label)}</span>")
    return f'<div class="fm-legend">{"".join(keys)}</div>'


def _figure(title: str, subtitle: str, body: str, *, legend_html: str = "", note: str = "",
            label: str = "") -> str:
    sub = f"<span>{esc(subtitle)}</span>" if subtitle else ""
    foot = f'<div class="fm-chart-note">{esc(note)}</div>' if note else ""
    # Every per-fund chart keeps a legend row (an empty one when it needs none), so their plots start at the same
    # height and the fund rows line up across the small multiples.
    legend_html = legend_html or '<div class="fm-legend" aria-hidden="true">&nbsp;</div>'
    return (f'<figure class="fm-chart" aria-label="{esc(label or title)}"><figcaption><b>{esc(title)}</b>{sub}'
            f"</figcaption>{legend_html}{body}{foot}</figure>")


def label_px(text: str) -> float:
    """The approximate width in px of a 10.5px tick label (digits and $ ~5.9px; %, M wider; . narrower)."""
    return sum(_CHAR_PX.get(char, 5.9) for char in text)


def labelled_ticks(ticks: Sequence[float], kind: str = "money", plot_px: float = TICK_PLOT_PX) -> List[float]:
    """The ticks that get a label: the two ends and zero, as long as their text does not collide; every tick keeps
    its gridline. The widths are measured against ``plot_px``, the narrowest the plot gets (the SVG is in percent,
    so the real width is unknown here). A negative low end is always labelled, so a below-zero zone is never
    unlabelled; then zero, then the high end, each only when it clears the labels already placed (the low end is
    drawn from its left edge, the high end up to its right edge, the rest centred)."""
    if len(ticks) <= 2:
        return list(ticks)
    low, high = ticks[0], ticks[-1]
    span = (high - low) or 1.0
    width = plot_px * (PLOT_RIGHT - PLOT_LEFT) / 100.0

    def extent(tick: float) -> Tuple[float, float]:
        x, w = (tick - low) / span * width, label_px(tick_label(tick, kind))
        if tick == low:
            return x, x + w
        if tick == high:
            return x - w, x
        return x - w / 2, x + w / 2

    placed: List[Tuple[float, float]] = []
    keep: List[float] = []
    for tick in dict.fromkeys(t for t in ([low] if low < 0 else []) + [0.0, high] if low <= t <= high):
        left, right = extent(tick)
        if all(right + TICK_GAP_PX <= a or left >= b + TICK_GAP_PX for a, b in placed):
            placed.append((left, right))
            keep.append(tick)
    return sorted(keep)


def _grid_lines(scale: Scale, ticks: Sequence[float], height: int, kind: str) -> str:
    """Gridlines with their labels: the ones that fit the narrowest plot always; the ones that only fit a plot of
    WIDE_PLOT_PX get class ``tk-wide``, which the theme shows only when the plot is that wide (container query)."""
    out = []
    shown = labelled_ticks(ticks, kind)
    wide = labelled_ticks(ticks, kind, plot_px=WIDE_PLOT_PX)
    for tick in ticks:
        x = scale(tick)
        anchor = "start" if x <= PLOT_LEFT + 0.01 else "end" if x >= PLOT_RIGHT - 0.01 else "middle"
        color = AXIS if tick == 0 else GRID
        css = "" if tick in shown else ' class="tk-wide"'
        out.append(f'<line x1="{x}%" x2="{x}%" y1="{AXIS_BAND - 4}" y2="{height}" stroke="{color}" stroke-width="1"/>'
                   + (f'<text x="{x}%" y="{AXIS_BAND - 8}" text-anchor="{anchor}" fill="{TICK}" font-size="10.5"{css}>'
                      f"{esc(tick_label(tick, kind))}</text>" if tick in shown or tick in wide else ""))
    return "".join(out)


def _bar(x0: float, x1: float, y: float, color: str, title: str, *, round_end: bool = True,
         height: int = BAR) -> str:
    """A horizontal bar from x0% (the baseline side) to x1%: 4px rounded data end, square at the baseline.

    The rounded rect is covered on its baseline half by a square one (percent widths cannot take a px
    offset), so any bar length works.
    """
    left, right = min(x0, x1), max(x0, x1)
    width = round(right - left, 3)
    if width <= 0:
        return ""
    tip = f"<title>{esc(title)}</title>"
    if not round_end or width < 0.6:
        return f'<rect x="{left}%" y="{y}" width="{width}%" height="{height}" fill="{color}">{tip}</rect>'
    half = round(width / 2, 3)
    square_x = left if x1 >= x0 else round(left + half, 3)
    return (f'<g><rect x="{left}%" y="{y}" width="{width}%" height="{height}" rx="4" fill="{color}"/>'
            f'<rect x="{square_x}%" y="{y}" width="{half}%" height="{height}" fill="{color}"/>{tip}</g>')


def _gap(x: float, y: float, height: int = BAR) -> str:
    """A 2px surface gap across a boundary between two touching segments."""
    return f'<rect x="{x}%" y="{y - 1}" width="2" height="{height + 2}" fill="{SURFACE}" transform="translate(-1 0)"/>'


def _rows_frame(names: Sequence[str], values: Sequence[str], svg: str, value_classes: Sequence[str] = ()) -> str:
    """The fund-name column, the plot and the value column, row-aligned."""
    labels = "".join(f'<div style="height:{ROW}px">{esc(n)}</div>' for n in names)
    cells = "".join(f'<div class="{esc(value_classes[i]) if i < len(value_classes) else ""}" style="height:{ROW}px">'
                    f"{v}</div>" for i, v in enumerate(values))
    pad = f'<div style="height:{AXIS_BAND}px"></div>'
    return (f'<div class="fm-chart-rows"><div class="names">{pad}{labels}</div><div class="plot">{svg}</div>'
            f'<div class="vals">{pad}{cells}</div></div>')


def _svg(height: int, label: str, inner: str) -> str:
    return (f'<svg width="100%" height="{height}" role="img" aria-label="{esc(label)}" '
            f'xmlns="http://www.w3.org/2000/svg" style="display:block;overflow:visible">{inner}</svg>')


def _row_y(index: int) -> float:
    return AXIS_BAND + index * ROW + (ROW - BAR) / 2


# ---------------------------------------------------------------------------
# Per-fund charts (one row per fund, same order in all three)
# ---------------------------------------------------------------------------

def priced_net(asset: Mapping) -> float:
    """Buys − sells of the class's trades that are in the cash estimate (``-cash_effect``), else its net value.

    Trades without a usable price (valued at quantity × multiplier) and staged quantities that only restate
    the position held are left out, as in the post-trade cash, so the bars are dollars actually traded.
    """
    effect = _num(asset.get("cash_effect"))
    return -effect if effect is not None else (_num(asset.get("net_value")) or 0.0)


def left_out_note(funds: Sequence[Mapping], *, table_pct: bool = False) -> str:
    """The trades the priced charts leave out (as the cash estimate does), per fund; '' when none.

    ``table_pct`` adds each fund's all-trades % of TNA, as the summary table shows it.
    """
    left_out = sorted(((_num((fund.get("cash") or {}).get("excluded_value")) or 0.0, fund) for fund in funds),
                      key=lambda item: -item[0])
    left_out = [(value, fund) for value, fund in left_out if value]
    if not left_out:
        return ""
    parts = []
    for value, fund in left_out[:LEFT_OUT_SHOWN]:
        text = f"{fund.get('fund')} {ui.money(value)}"
        if table_pct and _num(fund.get("gross_pct_tna")) is not None:
            text += f" (table: {ui.pct(fund.get('gross_pct_tna'))})"
        parts.append(text)
    more = len(left_out) - LEFT_OUT_SHOWN
    tail = f" and {more} more (each fund's drill-down lists them)" if more > 0 else ""
    return ("Not plotted: trades without a usable price, that restate the position held, or that repeat another "
            "row: " + ", ".join(parts) + tail + ".")


def net_by_asset_chart(funds: Sequence[Mapping]) -> str:
    """Net trade value (buys − sells) per fund, split by asset class: net buying right of zero, selling left."""
    rows = []
    for fund in funds:
        nets = [(a, priced_net(((fund.get("assets") or {}).get(a) or {}))) for a in ASSETS]
        rows.append((str(fund.get("fund") or ""), nets))
    if not rows:
        return ""
    low, high, ticks = nice_domain([sum(v for _, v in nets if v < 0) for _, nets in rows]
                                   + [sum(v for _, v in nets if v > 0) for _, nets in rows], count=3)
    scale = Scale(low, high)
    height = AXIS_BAND + ROW * len(rows)
    marks = [_grid_lines(scale, ticks, height, "money")]
    values, classes = [], []
    for i, (name, nets) in enumerate(rows):
        y = _row_y(i)
        for sign in (1, -1):
            parts = [(a, v) for a, v in nets if v * sign > 0]
            start = 0.0
            for n, (asset, value) in enumerate(parts):
                end = start + value
                title = f"{name} · {ASSET_LABELS[asset]} net {ui.money(value, signed=True, compact=False)}"
                marks.append(_bar(scale(start), scale(end), y, ASSET_COLORS[asset], title,
                                  round_end=n == len(parts) - 1))
                if n:
                    marks.append(_gap(scale(start), y))
                start = end
        total = sum(v for _, v in nets)
        values.append(esc(ui.money(total, signed=True)) if any(v for _, v in nets) else ui.DASH)
        classes.append("")
    marks.append(f'<line x1="{scale(0)}%" x2="{scale(0)}%" y1="{AXIS_BAND - 4}" y2="{height}" stroke="{AXIS}"/>')
    svg = _svg(height, "Net trade value per fund by asset class", "".join(marks))
    used = [a for a in ASSETS if any(dict(nets)[a] for _, nets in rows)] or ["equity"]
    note = left_out_note(funds)
    return _figure("Net trade value by asset class", "priced buys − sells; buying right of zero",
                   _rows_frame([r[0] for r in rows], values, svg, classes),
                   legend_html=legend([(ASSET_COLORS[a], ASSET_LABELS[a]) for a in used]), note=note,
                   label="Net trade value per fund by asset class")


def _cash_pct(cash: Mapping, side: str, tna: Optional[float]) -> Optional[float]:
    value = _num(cash.get(f"{side}_pct_tna"))
    if value is None and tna:
        amount = _num(cash.get(side))
        value = None if amount is None else amount / tna
    return value


def cash_dumbbell_chart(funds: Sequence[Mapping]) -> str:
    """Current cash (hollow dot) → estimated post-trade cash (filled dot) per fund, as % of TNA so small and large
    funds compare; below zero is washed red. The value column gives the post-trade dollars."""
    rows = []
    for fund in funds:
        cash = fund.get("cash") or {}
        tna = _num(fund.get("tna")) or None
        rows.append((str(fund.get("fund") or ""), _cash_pct(cash, "current", tna), _cash_pct(cash, "post", tna),
                     cash))
    if not rows:
        return ""
    known = [v for _, a, b, _ in rows for v in (a, b) if v is not None]
    if not known:
        return ""
    low, high, ticks = nice_domain(known, count=3)
    scale = Scale(low, high)
    height = AXIS_BAND + ROW * len(rows)
    marks = []
    if low < 0:
        marks.append(f'<rect x="{scale(low)}%" y="{AXIS_BAND - 4}" width="{round(scale(0) - scale(low), 3)}%" '
                     f'height="{height - AXIS_BAND + 4}" fill="{NEG_WASH}"><title>Below zero cash</title></rect>')
    marks.append(_grid_lines(scale, ticks, height, "pct"))
    values, classes, missing = [], [], []
    for i, (name, current, post, cash) in enumerate(rows):
        cy = AXIS_BAND + i * ROW + ROW / 2
        negative = bool(cash.get("negative"))
        if current is None or post is None:
            values.append('<span class="none">no cash</span>' if cash.get("current") is None
                          else '<span class="none">no TNA</span>')
            classes.append("")
            missing.append(name)
            continue
        why = (" · the trades take cash below zero" if cash.get("goes_negative")
               else " · already below zero before the trades" if cash.get("negative_before") else "")
        title = (f"{name} · current cash {ui.money(cash.get('current'), compact=False)} ({ui.pct(current)} of TNA) → "
                 f"post-trade {ui.money(cash.get('post'), compact=False)} ({ui.pct(post)}) (est.){why}")
        marks.append(f'<g><title>{esc(title)}</title>'
                     f'<line x1="{scale(current)}%" x2="{scale(post)}%" y1="{cy}" y2="{cy}" stroke="{INK}" '
                     f'stroke-width="2" stroke-opacity=".55"/>'
                     f'<circle cx="{scale(current)}%" cy="{cy}" r="4.5" fill="{SURFACE}" stroke="{BEFORE}" stroke-width="2"/>'
                     f'<circle cx="{scale(post)}%" cy="{cy}" r="5" fill="{POST}" stroke="{SURFACE}" stroke-width="2"/>'
                     f'<rect x="{min(scale(current), scale(post))}%" y="{cy - ROW / 2}" height="{ROW}" '
                     f'width="{max(abs(scale(post) - scale(current)), 1.5)}%" fill="transparent"/></g>')
        values.append(("⚠ " if negative else "") + esc(ui.money(cash.get("post"))))
        classes.append("neg" if negative else "")
    svg = _svg(height, "Cash as a share of TNA before and after the proposed trades per fund", "".join(marks))
    note = f"Not plotted (no custodian cash or TNA): {', '.join(missing)}." if missing else ""
    return _figure("Cash before → after", "% of TNA: T-1 custodian cash → estimated post-trade; $ post-trade at right",
                   _rows_frame([r[0] for r in rows], values, svg, classes),
                   legend_html=legend([(BEFORE, "Current cash", "ring"), (POST, "Post-trade (est.)", "dot")]),
                   note=note, label="Cash as a share of TNA before and after the proposed trades per fund")


def priced_gross_pct(fund: Mapping) -> Optional[float]:
    """Buys + sells as a share of TNA, less the trades left out of the cash estimate (see :func:`priced_net`);
    None without a TNA."""
    tna = _num(fund.get("tna"))
    if not tna:
        return None
    priced = _num((fund.get("priced") or {}).get("gross_value"))
    if priced is not None:
        return priced / tna
    gross = _num(fund.get("gross_value"))
    if gross is None:
        return _num(fund.get("gross_pct_tna"))
    excluded = _num((fund.get("cash") or {}).get("excluded_value")) or 0.0
    return max(gross - excluded, 0.0) / tna


def pct_tna_chart(funds: Sequence[Mapping]) -> str:
    """Gross priced traded value (buys + sells) as a share of TNA, per fund."""
    rows = [(str(f.get("fund") or ""), priced_gross_pct(f)) for f in funds]
    if not rows:
        return ""
    low, high, ticks = nice_domain([v for _, v in rows if v is not None] or [0.01], count=3)
    scale = Scale(0.0, high)
    height = AXIS_BAND + ROW * len(rows)
    marks = [_grid_lines(scale, [t for t in ticks if t >= 0], height, "pct")]
    values = []
    for i, (name, value) in enumerate(rows):
        if value is None:
            values.append('<span class="none">no TNA</span>')
            continue
        marks.append(_bar(scale(0), scale(value), _row_y(i), TOTAL, f"{name} · traded {ui.pct(value)} of TNA"))
        values.append(esc(ui.pct(value)))
    svg = _svg(height, "Share of TNA traded per fund", "".join(marks))
    return _figure("Traded % of TNA", "priced buys + sells over net assets, all asset classes",
                   _rows_frame([r[0] for r in rows], values, svg),
                   note=left_out_note(funds, table_pct=True),
                   label="Share of TNA traded per fund")


# ---------------------------------------------------------------------------
# Cash waterfall (one fund)
# ---------------------------------------------------------------------------

_STEP_LINES = {"current_cash": ("Current", "cash"), "equity_buys": ("Equity", "buys"),
               "equity_sells": ("Equity", "sells"), "options_net": ("Options", "net"),
               "treasury_net": ("Treasury", "net"), "post_cash": ("Post-trade", "(est.)")}
_STEP_COLORS = {"equity_buys": ASSET_COLORS["equity"], "equity_sells": ASSET_COLORS["equity"],
                "options_net": ASSET_COLORS["options"], "treasury_net": ASSET_COLORS["treasury"]}
WATERFALL_PLOT = 170  # px


def _wrap(label: str, width: int = 13) -> List[str]:
    words, lines = str(label).split(), [""]
    for word in words:
        if lines[-1] and len(lines[-1]) + 1 + len(word) > width and len(lines) < 2:
            lines.append(word)
        else:
            lines[-1] = f"{lines[-1]} {word}".strip()
    return lines


def waterfall_chart(steps: Sequence[Mapping], *, title: str = "Cash waterfall", note: str = "") -> str:
    """Current cash, each asset class's cash effect, then the estimated post-trade cash, as columns."""
    steps = [s for s in steps if _num(s.get("running")) is not None]
    if not steps:
        return ""
    levels = [0.0]
    previous = 0.0
    for step in steps:
        running = _num(step.get("running")) or 0.0
        levels += [running, previous]
        previous = running
    low, high, ticks = nice_domain(levels, count=4)
    top, plot = 22, WATERFALL_PLOT  # headroom for the value label above the tallest column
    y = lambda v: round(top + (high - v) / (high - low) * plot, 2)  # noqa: E731
    count = len(steps)
    slot = 100.0 / count
    marks = []
    # Only the zero line: every column is direct-labelled, so unlabelled gridlines would carry nothing.
    marks.append(f'<line x1="0" x2="100%" y1="{y(0.0)}" y2="{y(0.0)}" stroke="{AXIS}" stroke-width="1"/>')
    previous = 0.0
    for i, step in enumerate(steps):
        kind = step.get("kind")
        value = _num(step.get("value")) or 0.0
        running = _num(step.get("running")) or 0.0
        center = round(slot * (i + 0.5), 3)
        start, end = (0.0, running) if kind in ("start", "end") else (previous, running)
        color = TOTAL if kind in ("start", "end") else _STEP_COLORS.get(str(step.get("key")), TOTAL)
        y0, y1 = y(start), y(end)
        top_y, bottom_y = min(y0, y1), max(y0, y1)
        label = str(step.get("label") or "")
        shown = ui.money(value, signed=kind == "delta")
        tip = f"<title>{esc(label)}: {esc(ui.money(value, signed=kind == 'delta', compact=False))}</title>"
        if bottom_y - top_y >= 1:
            height = round(bottom_y - top_y, 2)
            marks.append(f'<g transform="translate(-12 0)"><rect x="{center}%" y="{top_y}" width="24" '
                         f'height="{height}" rx="4" fill="{color}"/>')
            if kind in ("start", "end"):  # anchored at zero: square there
                square_y = top_y + height / 2 if end >= 0 else top_y
                marks.append(f'<rect x="{center}%" y="{round(square_y, 2)}" width="24" height="{round(height / 2, 2)}" '
                             f'fill="{color}"/>')
            marks.append(f"{tip}</g>")
        else:
            marks.append(f'<g transform="translate(-12 0)"><rect x="{center}%" y="{round(y1 - 1, 2)}" width="24" '
                         f'height="2" fill="{GRID}"/>{tip}</g>')
        if i < count - 1:  # connector at the running level to the next column
            nxt = round(slot * (i + 1.5), 3)
            marks.append(f'<line x1="{center}%" x2="{nxt}%" y1="{y(running)}" y2="{y(running)}" stroke="{AXIS}" '
                         f'stroke-width="1" transform="translate(12 0)" stroke-opacity=".8"/>')
        label_y = top_y - 6 if (end >= start or kind in ("start", "end")) and end >= 0 else bottom_y + 13
        fill = "#E3E7EF" if kind in ("start", "end") else INK
        weight = ' font-weight="650"' if kind == "end" else ""
        marks.append(f'<text x="{center}%" y="{round(label_y, 2)}" text-anchor="middle" fill="{fill}" '
                     f'font-size="11"{weight}>{esc(shown if value or kind != "delta" else "—")}</text>')
        for n, line in enumerate(_STEP_LINES.get(str(step.get("key")), None) or _wrap(label)):
            marks.append(f'<text x="{center}%" y="{top + plot + 18 + n * 12}" text-anchor="middle" fill="{TICK}" '
                         f'font-size="10">{esc(line)}</text>')
        previous = running
    svg = _svg(top + plot + 46, "Cash waterfall from current to estimated post-trade cash", "".join(marks))
    keys = [(TOTAL, "Cash")] + [(ASSET_COLORS[a], ASSET_LABELS[a]) for a in ASSETS
                                if any(str(s.get("key", "")).startswith(a) and _num(s.get("value")) for s in steps)]
    return _figure(title, "sells and premiums received add, buys and premiums paid subtract",
                   f'<div class="fm-chart-plot">{svg}</div>', legend_html=legend(keys), note=note,
                   label="Cash waterfall")
