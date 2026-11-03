"""Strict comparison of two golden run roots (``<root>/out`` files + ``<root>/meta``).

xlsx: sheet order, values, formulas, number formats, fonts, fills, borders,
      alignment, merges, freeze panes, column widths, row heights.
pdf:  page count, page boxes, link annotations, and each page's decompressed
      content stream (masked only for the run-date "Generated:" stamp).
meta: emails.json / run.log / status (exit code + error text) with run-root
      paths and timestamps normalised.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

MAX_REPORTED = 25
_GENERATED = re.compile(rb"Generated: \d{4}-\d{2}-\d{2}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}")


def _norm_text(text: str, root: Path) -> str:
    for form in {str(root), str(root).replace("\\", "/"), str(root).replace("/", "\\")}:
        text = text.replace(form, "<ROOT>").replace(form.replace("\\", "\\\\"), "<ROOT>")
    return _TIMESTAMP.sub("<TS>", text)


META_TEXT = ("emails.json", "run.log")


def meta_lines(root: Path, fname: str) -> List[str]:
    return _norm_text((root / "meta" / fname).read_text(encoding="utf-8"), root).splitlines()


def _color(c):
    return None if c is None else (c.type, c.rgb if c.type == "rgb" else c.value, c.tint)


_STYLE_PARTS = {
    "font": lambda f: (f.name, f.sz, f.b, f.i, f.u, _color(f.color)),
    "fill": lambda fl: (fl.fill_type, _color(getattr(fl, "fgColor", None)), _color(getattr(fl, "bgColor", None))),
    "border": lambda b: tuple((s.style, _color(s.color)) for s in (b.left, b.right, b.top, b.bottom)),
    "alignment": lambda a: (a.horizontal, a.vertical, a.wrap_text, a.indent),
}


def _style(cell) -> tuple:
    parts = [cell.number_format]
    for name, describe in _STYLE_PARTS.items():
        try:
            parts.append(describe(getattr(cell, name)))
        except IndexError:  # dangling style index (sheets copied between workbooks): compare the raw ids
            parts.append(f"<dangling {name} {cell._style!r}>")
    return tuple(parts)


def sheet_attrs(sa, sb) -> list:
    """(label, a, b) for each sheet-level property the differ compares."""
    return [
        ("dimensions", sa.dimensions, sb.dimensions),
        ("merged", sorted(map(str, sa.merged_cells.ranges)), sorted(map(str, sb.merged_cells.ranges))),
        ("freeze_panes", sa.freeze_panes, sb.freeze_panes),
        ("col widths", {k: (d.width, d.hidden) for k, d in sa.column_dimensions.items()},
         {k: (d.width, d.hidden) for k, d in sb.column_dimensions.items()}),
        ("row heights", {k: d.height for k, d in sa.row_dimensions.items() if d.height},
         {k: d.height for k, d in sb.row_dimensions.items() if d.height}),
    ]


def cell_pairs(sa, sb):
    """Every (a, b) cell pair over the union of both sheets' used ranges, row-major."""
    max_r, max_c = max(sa.max_row, sb.max_row), max(sa.max_column, sb.max_column)
    for r in range(1, max_r + 1):
        for c in range(1, max_c + 1):
            yield sa.cell(r, c), sb.cell(r, c)


def cell_value(cell) -> tuple:
    return cell.value, cell.data_type


def diff_xlsx(a: Path, b: Path) -> List[str]:
    import openpyxl

    wa, wb = openpyxl.load_workbook(a), openpyxl.load_workbook(b)
    out: List[str] = []
    if wa.sheetnames != wb.sheetnames:
        return [f"sheet order/names: {wa.sheetnames} != {wb.sheetnames}"]
    for name in wa.sheetnames:
        sa, sb = wa[name], wb[name]
        for label, va, vb in sheet_attrs(sa, sb):
            if va != vb:
                out.append(f"[{name}] {label}: {va!r} != {vb!r}")
        for ca, cb in cell_pairs(sa, sb):
            if cell_value(ca) != cell_value(cb):
                out.append(f"[{name}] {ca.coordinate} value: {ca.value!r} != {cb.value!r}")
            elif _style(ca) != _style(cb):
                out.append(f"[{name}] {ca.coordinate} style: {_style(ca)} != {_style(cb)}")
            if len(out) >= MAX_REPORTED:
                return out
    return out


def _page_stream(page) -> bytes:
    contents = page.get_contents()
    data = contents.get_data() if contents is not None else b""
    return _GENERATED.sub(b"Generated: <DATE>", data)


def _annots(page) -> list:
    result = []
    for ref in page.get("/Annots") or []:
        obj = ref.get_object()
        action = obj.get("/A")
        target = action.get_object().get("/URI") if action is not None else None
        rect = [round(float(x), 3) for x in obj.get("/Rect", [])]
        dest = [getattr(d, "idnum", None) or str(d) for d in (obj.get("/Dest") or [])]
        result.append((str(obj.get("/Subtype")), rect, str(target), dest))
    return result


def diff_pdf(a: Path, b: Path) -> List[str]:
    from pypdf import PdfReader

    ra, rb = PdfReader(str(a)), PdfReader(str(b))
    if len(ra.pages) != len(rb.pages):
        return [f"page count: {len(ra.pages)} != {len(rb.pages)}"]
    out: List[str] = []
    for i, (pa, pb) in enumerate(zip(ra.pages, rb.pages), start=1):
        if list(pa.mediabox) != list(pb.mediabox):
            out.append(f"p{i} mediabox: {list(pa.mediabox)} != {list(pb.mediabox)}")
        if _annots(pa) != _annots(pb):
            out.append(f"p{i} annotations differ")
        if _page_stream(pa) != _page_stream(pb):
            ta, tb = pa.extract_text().splitlines(), pb.extract_text().splitlines()
            first = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            ctx = lambda t: t[first] if first < len(t) else "<end>"
            out.append(f"p{i} content stream differs; first text diff line {first}: {ctx(ta)!r} != {ctx(tb)!r}")
        if len(out) >= MAX_REPORTED:
            break
    return out


def diff_meta(root_a: Path, root_b: Path) -> List[str]:
    out: List[str] = []
    sa = json.loads((root_a / "meta" / "status.json").read_text(encoding="utf-8"))
    sb = json.loads((root_b / "meta" / "status.json").read_text(encoding="utf-8"))
    for k in ("exit_code", "error"):
        va, vb = sa.get(k), sb.get(k)
        if k == "error" and va and vb:
            va, vb = _norm_text(va, root_a), _norm_text(vb, root_b)
        if va != vb:
            out.append(f"status.{k}: {va!r} != {vb!r}")
    for fname in META_TEXT:
        ta, tb = meta_lines(root_a, fname), meta_lines(root_b, fname)
        if ta != tb:
            first = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            ctx = lambda t: t[first][:300] if first < len(t) else "<end>"
            out.append(f"{fname} line {first + 1}: {ctx(ta)!r} != {ctx(tb)!r}")
    return out


def out_files(root: Path) -> set:
    return {p.relative_to(root / "out") for p in (root / "out").rglob("*") if p.is_file()}


def diff_file(pa: Path, pb: Path) -> List[str]:
    if pa.suffix == ".xlsx":
        return diff_xlsx(pa, pb)
    if pa.suffix == ".pdf":
        return diff_pdf(pa, pb)
    return [] if pa.read_bytes() == pb.read_bytes() else ["bytes differ"]


def diff_roots(root_a: Path, root_b: Path) -> List[str]:
    files_a, files_b = out_files(root_a), out_files(root_b)
    out = [f"only in baseline: {p}" for p in sorted(files_a - files_b)]
    out += [f"only in current: {p}" for p in sorted(files_b - files_a)]
    for rel in sorted(files_a & files_b):
        out += [f"{rel}: {p}" for p in diff_file(root_a / "out" / rel, root_b / "out" / rel)]
    return out + diff_meta(root_a, root_b)
