"""Markdown review of golden differences (reference run roots vs current), for human sign-off.

Whether a scenario/file counts as changed is decided by differ.py exactly as
``golden.py check`` does; this module only explains *what* changed.
"""
from __future__ import annotations

import difflib
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from tools.golden.differ import (
    META_TEXT, _STYLE_PARTS, _annots, _norm_text, _page_stream, _style,
    cell_pairs, cell_value, diff_file, diff_meta, meta_lines, out_files, sheet_attrs,
)

MAX_CELLS = 40           # changed-value rows listed per sheet
MAX_STYLE_EXAMPLES = 5   # formatting examples listed per sheet
MAX_DIFF_LINES = 80      # unified-diff lines shown per PDF page / log / email
MAX_PAGES = 10           # PDF pages with a text diff shown per file
MAX_TEXT = 160           # characters shown per value / diff line
STYLE_NAMES = ("number format",) + tuple(_STYLE_PARTS)
ATTR_NAMES = {"dimensions": "Used range", "merged": "Merged cells", "freeze_panes": "Frozen panes",
              "col widths": "Column widths", "row heights": "Row heights"}
_GENERATED = re.compile(r"Generated: \d{4}-\d{2}-\d{2}")
_BLOCK_TAG = re.compile(r"(?i)(?=<(?:tr|h\d|p|br|li|div|table)\b)")
COUNTS = ("files_added", "files_removed", "files_changed", "sheets_added", "sheets_removed", "cells", "styles",
          "pages", "log", "email", "status")


@dataclass
class Review:
    name: str
    changed: bool = False
    counts: dict = field(default_factory=lambda: dict.fromkeys(COUNTS, 0))
    lines: List[str] = field(default_factory=list)


def _flat(v) -> str:
    return str(v).replace("\r", "").replace("\n", "\\n")


def _clip(s: str, limit: int, start: int = 0) -> str:
    head = "..." if start else ""
    body = s[start:start + limit - len(head)]
    return head + body if start + len(body) >= len(s) else head + body[:-3] + "..."


def _short(v, limit: int = MAX_TEXT) -> str:
    s = _flat(v)
    return s if len(s) <= limit else _clip(s, limit)


def _pair(a, b, limit: int = MAX_TEXT):
    """Both values cut to limit, keeping the first differing character in view on both sides."""
    sa, sb = _flat(a), _flat(b)
    p = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), min(len(sa), len(sb)))
    start = 0 if p < limit - 40 else p - 30
    return tuple(s if len(s) <= limit else _clip(s, limit, start) for s in (sa, sb))


def _shown(ca, cb):
    va, vb = ca.value, cb.value
    if va is not None and type(va) is type(vb) and va != vb and _flat(va) == _flat(vb):
        va, vb = repr(va), repr(vb)  # differ only in characters flattened for display (e.g. \r)
    sa, sb = ("(empty)" if v is None else s.replace("|", "\\|") for v, s in zip((ca.value, cb.value), _pair(va, vb)))
    if sa == sb:  # same text, different type (e.g. 1 vs "1", or an empty cell's data type)
        ka, kb = (type(ca.value).__name__, type(cb.value).__name__)
        ka, kb = (ka, kb) if ka != kb else (f"cell type {ca.data_type}", f"cell type {cb.data_type}")
        sa, sb = f"{sa} ({ka})", f"{sb} ({kb})"
    return sa, sb


def _udiff(a: List[str], b: List[str], context: int = 2) -> List[str]:
    raw = list(difflib.unified_diff(a, b, "before", "after", n=context, lineterm=""))
    lines, i = raw[:2], 2
    while i < len(raw):  # pair each run of -lines with the +lines after it and cut both around their first difference
        j = i
        while j < len(raw) and raw[j][:1] == "-":
            j += 1
        k = j
        while k < len(raw) and raw[k][:1] == "+":
            k += 1
        if k == i:
            lines.append(_short(raw[i], 300))
            i += 1
            continue
        minus, plus = raw[i:j], raw[j:k]
        for n in range(min(len(minus), len(plus))):
            ma, pb = _pair(minus[n][1:], plus[n][1:], 299)
            minus[n], plus[n] = "-" + ma, "+" + pb
        lines += [x[0] + _short(x[1:], 299) for x in minus + plus]
        i = k
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + [f"... ({len(lines) - MAX_DIFF_LINES} more diff lines not shown)"]
    return ["```diff", *lines, "```"] if lines else []


def _attr_change(va, vb) -> str:
    if isinstance(va, dict) and isinstance(vb, dict):
        keys = sorted({k for k in va.keys() | vb.keys() if va.get(k) != vb.get(k)}, key=str)
        parts = [f"{k}: {va.get(k, '(none)')} -> {vb.get(k, '(none)')}" for k in keys]
        return _short("; ".join(parts), 400)
    return " -> ".join(_pair(va, vb, 200))


def _sheet(name: str, sa, sb, counts: dict) -> List[str]:
    attrs = [(label, va, vb) for label, va, vb in sheet_attrs(sa, sb) if va != vb]
    values, styles, kinds = [], [], {}
    for ca, cb in cell_pairs(sa, sb):
        if cell_value(ca) != cell_value(cb):
            values.append((ca, cb))
        st_a, st_b = _style(ca), _style(cb)
        if st_a != st_b:
            parts = [i for i, (x, y) in enumerate(zip(st_a, st_b)) if x != y]
            for i in parts:
                kinds[STYLE_NAMES[i]] = kinds.get(STYLE_NAMES[i], 0) + 1
            styles.append((ca.coordinate, [(STYLE_NAMES[i], st_a[i], st_b[i]) for i in parts]))
    if not (attrs or values or styles):
        return []
    counts["cells"] += len(values)
    counts["styles"] += len(styles)
    out = [f"#### Sheet `{name}`: {len(values)} cell value(s) changed, {len(styles)} cell(s) with formatting changes", ""]
    out += [f"- {ATTR_NAMES.get(label, label)}: {_attr_change(va, vb)}" for label, va, vb in attrs]
    if attrs:
        out.append("")
    if values:
        out += ["| Cell | Before | After |", "|---|---|---|"]
        for ca, cb in values[:MAX_CELLS]:
            out.append(f"| {name}!{ca.coordinate} | {' | '.join(_shown(ca, cb))} |")
        if len(values) > MAX_CELLS:
            out.append(f"| ... | {len(values) - MAX_CELLS} more changed cell(s) in this sheet | |")
        out.append("")
    if styles:
        summary = ", ".join(f"{k}: {n}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]))
        cols = sorted({c.rstrip("0123456789") for c, _ in styles}, key=lambda c: (len(c), c))
        out.append(f"Formatting changed in {len(styles)} cell(s) ({summary}) in column(s) {', '.join(cols)}. "
                   "First examples:")
        for coord, parts in styles[:MAX_STYLE_EXAMPLES]:
            desc = "; ".join("{} `{}` -> `{}`".format(k, *_pair(x, y, 120)) for k, x, y in parts)
            out.append(f"- {name}!{coord}: {desc}")
        out.append("")
    return out


def _xlsx(pa: Path, pb: Path, counts: dict) -> List[str]:
    import openpyxl

    wa, wb = openpyxl.load_workbook(pa), openpyxl.load_workbook(pb)
    added = [s for s in wb.sheetnames if s not in wa.sheetnames]
    removed = [s for s in wa.sheetnames if s not in wb.sheetnames]
    counts["sheets_added"] += len(added)
    counts["sheets_removed"] += len(removed)
    out = [f"- Sheet added: `{s}`" for s in added] + [f"- Sheet removed: `{s}`" for s in removed]
    common = [s for s in wa.sheetnames if s in wb.sheetnames]
    if [s for s in wb.sheetnames if s in common] != common:
        out.append(f"- Sheet order changed: {', '.join(common)} -> {', '.join(s for s in wb.sheetnames if s in common)}")
    if out:
        out.append("")
    for name in common:
        out += _sheet(name, wa[name], wb[name], counts)
    return out


def _page_text(page) -> List[str]:
    return _GENERATED.sub("Generated: <DATE>", page.extract_text() or "").splitlines()


def _pdf(pa: Path, pb: Path, counts: dict) -> List[str]:
    from pypdf import PdfReader

    ra, rb = PdfReader(str(pa)), PdfReader(str(pb))
    na, nb = len(ra.pages), len(rb.pages)
    out = [f"- Page count: {na} -> {nb}", ""] if na != nb else []
    shown = 0
    for i in range(max(na, nb)):
        pga, pgb = (ra.pages[i] if i < na else None), (rb.pages[i] if i < nb else None)
        notes = []
        if pga is None or pgb is None:
            notes.append("page added" if pga is None else "page removed")
        else:
            if list(pga.mediabox) != list(pgb.mediabox):
                notes.append(f"page size {list(pga.mediabox)} -> {list(pgb.mediabox)}")
            if _annots(pga) != _annots(pgb):
                notes.append("links changed")
            if _page_stream(pga) == _page_stream(pgb) and not notes:
                continue
        diff = _udiff(_page_text(pga) if pga else [], _page_text(pgb) if pgb else [], context=1)
        if not diff and not notes:
            notes.append("layout/formatting only (text identical)")
        counts["pages"] += 1
        shown += 1
        if shown <= MAX_PAGES:
            out += [f"Page {i + 1}: {'; '.join(notes) if notes else 'text changed'}", "", *diff, ""]
    if shown > MAX_PAGES:
        out += [f"... {shown - MAX_PAGES} more changed page(s) not shown", ""]
    return out


def _other(pa: Path, pb: Path) -> List[str]:
    try:
        a, b = pa.read_text(encoding="utf-8"), pb.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ["- File contents differ (binary file)", ""]
    return [*_udiff(a.splitlines(), b.splitlines()), ""]


def _html_lines(body: str) -> List[str]:
    body = re.sub(r"(?is)<(style|head)\b[^>]*>.*?</\1>", "", body)
    body = re.sub(r"(?i)</t[dh]>", " | ", _BLOCK_TAG.sub("\n", body))
    text = html.unescape(re.sub(r"<[^>]+>", "", body))
    return [ln.strip().rstrip("|").strip() for ln in text.splitlines() if ln.strip()]


def _email_lines(text: str, as_text: bool) -> List[str]:
    out = []
    for i, mail in enumerate(json.loads(text), start=1):
        body = mail.get("body_html") or ""
        out += [f"== Email {i}: {mail.get('subject')}", f"To: {', '.join(mail.get('recipients') or [])}"]
        out += [f"Attachment: {a}" for a in mail.get("attachments") or []]
        out += _html_lines(body) if as_text else _BLOCK_TAG.sub("\n", body).splitlines()
    return out


def _emails(ta: List[str], tb: List[str]) -> List[str]:
    try:
        diff = _udiff(_email_lines("\n".join(ta), True), _email_lines("\n".join(tb), True))
        raw = _udiff(_email_lines("\n".join(ta), False), _email_lines("\n".join(tb), False), context=0)
    except (ValueError, AttributeError, TypeError):
        return _udiff(ta, tb)
    if not diff:
        return ["Email text is identical; only its HTML formatting (e.g. PASS/FAIL cell colours) changed:", "", *raw]
    removed = lambda d: sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
    out = ["Email content (text as the reader sees it):", "", *diff]
    if removed(raw) > removed(diff):  # some rows changed formatting only
        out += ["", "Some rows also changed HTML formatting (e.g. cell colours):", "", *raw]
    return out


def _status(root_a: Path, root_b: Path, counts: dict) -> List[str]:
    sa = json.loads((root_a / "meta" / "status.json").read_text(encoding="utf-8"))
    sb = json.loads((root_b / "meta" / "status.json").read_text(encoding="utf-8"))
    out = []
    if sa.get("exit_code") != sb.get("exit_code"):
        out.append(f"- Exit code: {sa.get('exit_code')} -> {sb.get('exit_code')}")
    ea, eb = (_norm_text(s["error"], r) if s.get("error") else None for s, r in ((sa, root_a), (sb, root_b)))
    if ea != eb:
        out.append("- Error: {} -> {}".format(*_pair(ea or "(none)", eb or "(none)", 400)))
        if sb.get("traceback"):
            out += ["", "Current traceback (last lines):", "", "```",
                    *_norm_text(sb["traceback"], root_b).splitlines()[-15:], "```"]
    if out:
        counts["status"] += 1
        out.append("")
    return out


def review_scenario(name: str, ref_root: Path, cur_root: Path) -> Review:
    rv = Review(name)
    if not (cur_root / "meta" / "status.json").is_file():
        rv.changed = True
        rv.counts["status"] += 1
        rv.lines = [f"- No current output found at `{cur_root}` (scenario not run?)", ""]
        return rv
    c, lines = rv.counts, rv.lines
    files_a, files_b = out_files(ref_root), out_files(cur_root)
    for rel in sorted(files_b - files_a):
        c["files_added"] += 1
        lines += [f"- New file (not in reference): `{rel.as_posix()}`"]
    for rel in sorted(files_a - files_b):
        c["files_removed"] += 1
        lines += [f"- File no longer produced: `{rel.as_posix()}`"]
    if files_a ^ files_b:
        lines.append("")
    for rel in sorted(files_a & files_b):
        pa, pb = ref_root / "out" / rel, cur_root / "out" / rel
        if not diff_file(pa, pb):
            continue
        c["files_changed"] += 1
        lines += [f"### `{rel.as_posix()}`", ""]
        try:
            render = {".xlsx": _xlsx, ".pdf": _pdf}.get(rel.suffix)
            lines += render(pa, pb, c) if render else _other(pa, pb)
        except Exception as exc:  # explain what we can; check still flags the file
            lines += [f"- Could not describe the change: {type(exc).__name__}: {exc}", ""]
    meta_problems = diff_meta(ref_root, cur_root)
    if meta_problems:
        status = _status(ref_root, cur_root, c)
        if status:
            lines += ["### Run status", "", *status]
        for fname in META_TEXT:
            ta, tb = meta_lines(ref_root, fname), meta_lines(cur_root, fname)
            if ta == tb:
                continue
            key, title = ("email", "Emails") if fname == "emails.json" else ("log", "Run log")
            c[key] += 1
            lines += [f"### {title}", "", *(_emails(ta, tb) if key == "email" else _udiff(ta, tb)), ""]
    rv.changed = bool(files_a ^ files_b or c["files_changed"] or meta_problems)
    if rv.changed and not lines:  # flagged by the differ but not explained above
        lines += [f"- {_short(p, 400)}" for p in meta_problems] + [""]
    return rv


def render(reviews: List[Review], reference: str) -> str:
    total = {k: sum(r.counts[k] for r in reviews) for k in COUNTS}
    changed = [r for r in reviews if r.changed]
    per = lambda key: sum(1 for r in reviews if r.counts[key])
    out = [
        "# Golden output review", "",
        f"Reference: **{reference}**. Compared against the current outputs of this checkout.", "",
        "## Summary", "",
        f"- Scenarios compared: {len(reviews)} ({len(reviews) - len(changed)} unchanged, **{len(changed)} changed**)",
        f"- Report files: {total['files_added']} new, {total['files_removed']} no longer produced, "
        f"{total['files_changed']} changed",
        f"- Spreadsheet sheets: {total['sheets_added']} new, {total['sheets_removed']} removed",
        f"- Spreadsheet cells: {total['cells']} value(s) changed, {total['styles']} with formatting changes",
        f"- PDF pages changed: {total['pages']}",
        f"- Run log changed in {per('log')} scenario(s); emails in {per('email')}; run status in {per('status')}",
        "",
    ]
    if changed:
        out += ["| Scenario | Files new / gone / changed | Sheets new / gone | Cell values | Formatting | PDF pages "
                "| Log | Email | Status |", "|---|---|---|---|---|---|---|---|---|"]
        yes = lambda n: "changed" if n else ""
        for r in changed:
            c = r.counts
            out.append(f"| {r.name} | {c['files_added']} / {c['files_removed']} / {c['files_changed']} "
                       f"| {c['sheets_added']} / {c['sheets_removed']} | {c['cells']} "
                       f"| {c['styles']} | {c['pages']} | {yes(c['log'])} | {yes(c['email'])} | {yes(c['status'])} |")
        out.append("")
    unchanged = [r.name for r in reviews if not r.changed]
    if unchanged:
        out += [f"Unchanged: {', '.join(unchanged)}", ""]
    for r in changed:
        out += [f"## {r.name}", "", *r.lines]
    return "\n".join(out).rstrip() + "\n"
