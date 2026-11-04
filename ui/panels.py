"""Streamlit panels shared by the pages: live progress of a run, and a finished run's results.

The progress panel is a fragment that re-reads the run folder every
``POLL_SECONDS`` (so it keeps the page responsive, survives reruns and lets
the user cancel). Each poll calls :func:`ui.runner.refresh`, which also
finalizes a worker started before a UI restart; once the run has finished it
reruns the whole app, which then shows :func:`results_panel` instead.

The results panel shows the run's status banner and errors, its result tables
from ``views.json`` (:mod:`ui.results`; older runs without one show the
summary per day), one download per report, the zip, Publish and the log.

Error messages and log tails go through :func:`ui.runner.redact`: a worker's
DB error can quote the connection string. Widget keys that confirm an action
(Publish) include the run folder, so a confirmation never carries over to
another run.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import streamlit as st

from ui import components as ui
from ui import history, results, runner

POLL_SECONDS = 1.0
STAGES = ("db", "load", "process", "reports", "email", "done")
_STAGE_LABELS = {"db": "Database", "load": "Load data", "process": "Process funds", "reports": "Reports",
                 "email": "Email", "done": "Done"}
STATUS_PILL = {"ok": "PASS", "ok with errors": "WARN", "running": "RUNNING", "incomplete": "WARN",
               "cancelled": "N/A", "timed out": "FAIL", "failed": "FAIL", "invalid": "FAIL"}
_MIME = {"pdf": "application/pdf", "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}


def is_finished(run_dir: Path) -> bool:
    """Whether the run's meta.json carries a finish time."""
    return bool((runner.read_json(Path(run_dir) / runner.META) or {}).get("finished"))


def status_pill(status: str) -> str:
    return ui.pill(STATUS_PILL.get(status, "FAIL"), status.upper())


@st.fragment(run_every=POLL_SECONDS)
def progress_panel(run_dir: str, user: str) -> None:
    """Progress bar (fund index/total), stage stepper, current step, Cancel and the log tail of an active run."""
    path = Path(run_dir)
    if runner.refresh(path):
        st.rerun()  # whole app: swap this panel for the results
    meta = runner.read_json(path / runner.META) or {}
    state = runner.progress(runner.read_events(path))
    with st.container(key="nr_progress"):
        ui.show(f'<div class="fm-live"><span class="t">Run in progress</span>'
                f'<span class="m">{ui.esc(path.name)} · started by {ui.esc(meta.get("user", "?"))} at '
                f'{ui.esc(clock(meta.get("started")))}</span></div>'
                f'<p class="fm-hint">{ui.esc(ui.summary_text(meta.get("summary")))}</p>')
        st.progress(state.fraction, text=f"{state.fraction:.0%}")
        ui.show(stepper_html(state.stage) + f'<div class="fm-now">Now: <b>{ui.esc(state.text)}</b></div>')
        if meta.get("cancel_requested"):
            ui.show(ui.callout("warn", f"Cancelling (requested by {meta.get('cancelled_by') or 'a user'})…"))
        elif st.button("Cancel run", key=f"cancel-{path.name}", type="secondary", icon=":material/stop_circle:"):
            runner.cancel(path, by=user)
            st.rerun()
        with st.expander("Run log (tail)"):
            st.code(runner.redact(runner.tail(path / runner.LOG, lines=30)) or "(empty)", language="log")


def stepper_html(stage: str) -> str:
    """Database ✓ — Load data ✓ — (Process funds) — Reports — … for the stage reached."""
    current = STAGES.index(stage) if stage in STAGES else (2 if stage in ("fund", "operation") else -1)
    parts = []
    for i, name in enumerate(STAGES):
        done = i < current or stage == "done"
        cls = "done" if done else "now" if i == current else ""
        mark = "✓" if done else str(i + 1)
        if parts:
            parts.append('<span class="fm-step-line"></span>')
        parts.append(f'<span class="fm-step {cls}"><span class="dot">{mark}</span>{ui.esc(_STAGE_LABELS[name])}</span>')
    return f'<div class="fm-steps">{"".join(parts)}</div>'


def clock(iso: Optional[str]) -> str:
    """``2026-11-03T09:15:02`` -> ``2026-11-03 09:15``."""
    try:
        return datetime.fromisoformat(str(iso)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(iso or "?")


def duration_text(started: Optional[str], finished: Optional[str]) -> str:
    try:
        seconds = (datetime.fromisoformat(str(finished)) - datetime.fromisoformat(str(started))).total_seconds()
    except (TypeError, ValueError):
        return ""
    minutes, secs = divmod(int(max(seconds, 0)), 60)
    return f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


def banner_html(record: history.RunRecord) -> str:
    took = duration_text(record.started, record.finished)
    meta = " · ".join(x for x in (record.user, f"started {clock(record.started)}", f"took {took}" if took else "",
                                  f"exit code {record.exit_code}" if record.exit_code is not None else "") if x)
    return (f'<div class="fm-banner">{status_pill(record.status)}<div><div class="t">{ui.esc(ui.summary_text(record.summary))}</div>'
            f'<div class="m">{ui.esc(record.name)} · {ui.esc(meta)}</div></div></div>')


def results_panel(run_dir: Path, user: str, key: str, day: Optional[str] = None) -> None:
    """Status, errors, the result tables, one download per report, zip-all, Publish and the log of a finished run.

    ``day`` limits the per-file downloads to that report date and opens the tables on it (the Reports page);
    the zip has every file.
    """
    run_dir = Path(run_dir)
    record = history.load_record(run_dir)
    if record is None:
        ui.show(ui.callout("warn", f"{run_dir.name} is not a run folder."))
        return
    result = history.load_result(run_dir)
    ui.show(banner_html(record))
    if record.errors:  # redacted
        ui.show(ui.callout("error", f"{len(record.errors)} error{'s' if len(record.errors) != 1 else ''} in this run:",
                           record.errors))
    views = history.load_views(run_dir)
    requested = record_tests(run_dir)
    shown = bool(views) and results.render_views(views, key=key, run=run_dir.name,
                                                 day=None if day == "stacked" else day, requested=requested)
    if not shown and result.summaries:
        rows = summary_rows(result.summaries)
        columns = list(dict.fromkeys(k for row in rows for k in row))
        ui.show(ui.heading("Summary per day") + ui.panel(ui.data_table(columns, [[r.get(c, "") for c in columns]
                                                                                  for r in rows])))
    _downloads(run_dir, record, result, user, key, day)
    with st.expander("Run log (tail)"):
        st.code(runner.redact(runner.tail(run_dir / runner.LOG, lines=60)) or "(empty)", language="log")


def record_tests(run_dir: Path) -> List[str]:
    """The compliance tests the run asked for ([] when its request is unreadable)."""
    try:
        return list(history.load_request(run_dir).compliance_tests)
    except (OSError, ValueError, TypeError, KeyError):
        return []


def _downloads(run_dir: Path, record: history.RunRecord, result, user: str, key: str, day: Optional[str]) -> None:
    files = history.report_files(result)
    if not files:
        if record.status not in ("running", "incomplete"):
            ui.show(ui.callout("info", "This run produced no report files."))
        return
    ui.show(ui.heading("Reports", f"{len(files)} file{'s' if len(files) != 1 else ''}"))
    st.download_button("Download all (zip)", data=_zip_reader(run_dir), file_name=f"{run_dir.name}.zip",
                       mime="application/zip", key=f"{key}-zip", type="primary", on_click="ignore",
                       icon=":material/folder_zip:")
    shown = [(i, r) for i, r in enumerate(files) if day is None or r.day == day]
    cols = st.columns(2, gap="small")
    for n, (i, report) in enumerate(shown):
        path = Path(report.path)
        kind = "PDF" if report.kind == "pdf" else "Excel" if report.kind == "excel" else report.kind
        cols[n % 2].download_button(
            f"{report.day} · {report.label} ({kind})", data=_reader(path), file_name=path.name, help=path.name,
            mime=_MIME.get(report.kind, "application/octet-stream"), key=f"{key}-file-{i}", on_click="ignore",
            icon=":material/picture_as_pdf:" if report.kind == "pdf" else ":material/table_view:", width="stretch")
    if record.status in ("ok", "ok with errors"):  # never publish partial reports
        _publish_controls(run_dir, user, key)
    else:
        st.caption("Publishing is only offered for runs that finished successfully.")


def _reader(path: Path) -> Callable[[], bytes]:
    """Deferred download data: the file is read only when the user clicks."""
    return lambda: runner.retry(path.read_bytes)


def _zip_reader(run_dir: Path) -> Callable[[], bytes]:
    """Deferred zip: built (once) and read only when the user clicks."""
    return lambda: history.zip_bytes(run_dir)


def summary_rows(summaries: Dict[str, dict]) -> List[dict]:
    """One row per day; list values (operations, errors) shown as text, the errors redacted."""
    rows: List[dict] = []
    for day, summary in summaries.items():
        row = {"day": day}
        for name, value in summary.items():
            if isinstance(value, (list, tuple)):
                value = "; ".join(map(str, value))
            row[name] = runner.redact(value) if name == "errors" else value
        rows.append(row)
    return rows


def _publish_controls(run_dir: Path, user: str, key: str) -> None:
    """Copy the reports to FUNDMANAGER_UI_PUBLISH_ROOT, behind a confirmation; hidden when that is unset."""
    dest = runner.publish_root()
    if dest is None:
        st.caption("Publishing is off (FUNDMANAGER_UI_PUBLISH_ROOT is not set).")
        return
    with st.expander(f"Publish to {dest}", icon=":material/publish:"):
        st.write("Copies this run's reports into the shared folder (same sub-folders). Existing files are "
                 "left untouched and listed as skipped.")
        sure = st.checkbox("I want to publish these reports", key=f"{key}-publish-sure-{run_dir.name}")
        if st.button("Publish", key=f"{key}-publish-{run_dir.name}", disabled=not sure, type="primary"):
            try:
                copied, skipped = runner.publish(run_dir, dest, by=user)
            except (OSError, ValueError) as exc:
                st.error(f"Publish failed: {exc}")
            else:
                st.success(f"Published {len(copied)} file(s) to {dest}." if copied else
                           "Nothing new to publish: every file is already there.")
                if skipped:
                    st.warning("Skipped (already there): " + ", ".join(p.name for p in skipped))
