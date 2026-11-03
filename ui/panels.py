"""Streamlit panels shared by the pages: live progress of a run, and a finished run's results.

Tables are rendered as Markdown, not ``st.dataframe``: Streamlit's Arrow
serialisation needs a pyarrow build that matches the project's pinned NumPy
1.26, and the one installed with Streamlit 1.66 (pyarrow 26) requires NumPy 2.

The progress panel is a fragment that re-reads the run folder every
``POLL_SECONDS`` (so it keeps the page responsive, survives reruns and lets
the user cancel). Each poll calls :func:`ui.runner.refresh`, which also
finalizes a worker started before a UI restart; once the run has finished it
reruns the whole app, which then shows :func:`results_panel` instead.

Error messages and log tails go through :func:`ui.runner.redact`: a worker's
DB error can quote the connection string. Widget keys that confirm an action
(Publish) include the run folder, so a confirmation never carries over to
another run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import streamlit as st

from ui import history, runner

POLL_SECONDS = 1.0
STAGES = ("db", "load", "process", "reports", "email", "done")
_STAGE_LABELS = {"db": "Database", "load": "Load data", "process": "Process funds", "reports": "Reports",
                 "email": "Email", "done": "Done"}
_STATUS_STYLE = {"ok": st.success, "ok with errors": st.warning, "running": st.info, "incomplete": st.warning}
_MIME = {"pdf": "application/pdf", "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}


def is_finished(run_dir: Path) -> bool:
    """Whether the run's meta.json carries a finish time."""
    return bool((runner.read_json(Path(run_dir) / runner.META) or {}).get("finished"))


@st.fragment(run_every=POLL_SECONDS)
def progress_panel(run_dir: str, user: str) -> None:
    """Progress bar (fund index/total), stage indicator, log tail and Cancel for an active run."""
    path = Path(run_dir)
    if runner.refresh(path):
        st.rerun()  # whole app: swap this panel for the results
    meta = runner.read_json(path / runner.META) or {}
    state = runner.progress(runner.read_events(path))
    with st.container(border=True):
        st.subheader("Run in progress")
        st.caption(f"{path.name} · started by {meta.get('user', '?')} at {meta.get('started', '?')}")
        st.caption(meta.get("summary", ""))
        st.progress(state.fraction, text=state.text)
        st.markdown(_stage_line(state.stage))
        if meta.get("cancel_requested"):
            st.warning(f"Cancelling (requested by {meta.get('cancelled_by') or 'a user'})…")
        elif st.button("Cancel run", key=f"cancel-{path.name}", type="secondary"):
            runner.cancel(path, by=user)
            st.rerun()
        with st.expander("Run log (tail)"):
            st.code(runner.redact(runner.tail(path / runner.LOG, lines=30)) or "(empty)", language="log")


def _stage_line(stage: str) -> str:
    """``Database ✓ → Load data ✓ → **Process funds** → Reports → …`` for the stage reached."""
    current = STAGES.index(stage) if stage in STAGES else (2 if stage in ("fund", "operation") else -1)
    parts = []
    for i, name in enumerate(STAGES):
        label = _STAGE_LABELS[name]
        parts.append(f"{label} ✓" if i < current or stage == "done" else f"**{label}**" if i == current else label)
    return " → ".join(parts)


def results_panel(run_dir: Path, user: str, key: str, day: Optional[str] = None) -> None:
    """Status, errors, per-day summary, one download per report, zip-all, Publish and the log of a finished run.

    ``day`` limits the per-file downloads to that report date (the Reports page); the zip has every file.
    """
    run_dir = Path(run_dir)
    record = history.load_record(run_dir)
    if record is None:
        st.warning(f"{run_dir.name} is not a run folder")
        return
    result = history.load_result(run_dir)
    show = _STATUS_STYLE.get(record.status, st.error)
    show(f"**{record.status.upper()}** · exit code {record.exit_code} · {record.name} · {record.user} · "
         f"{record.started or '?'} → {record.finished or '…'}")
    st.caption(record.summary)
    for error in record.errors:  # redacted
        st.error(error)
    if result.summaries:
        st.markdown("**Summary per day**")
        st.markdown(markdown_table(summary_rows(result.summaries)))
    files = history.report_files(result)
    if files:
        st.markdown(f"**Reports ({len(files)})**")
        st.download_button("Download all (zip)", data=_zip_reader(run_dir), file_name=f"{run_dir.name}.zip",
                           mime="application/zip", key=f"{key}-zip", type="primary", on_click="ignore")
        for i, report in enumerate(files):
            if day is not None and report.day != day:
                continue
            path = Path(report.path)
            kind = "PDF" if report.kind == "pdf" else "Excel" if report.kind == "excel" else report.kind
            st.download_button(f"{report.day} · {report.label} ({kind}) · {path.name}", data=_reader(path),
                               file_name=path.name, mime=_MIME.get(report.kind, "application/octet-stream"),
                               key=f"{key}-file-{i}", on_click="ignore")
        _publish_controls(run_dir, user, key)
    elif record.status not in ("running", "incomplete"):
        st.info("This run produced no report files.")
    with st.expander("Run log (tail)"):
        st.code(runner.redact(runner.tail(run_dir / runner.LOG, lines=60)) or "(empty)", language="log")


def _reader(path: Path) -> Callable[[], bytes]:
    """Deferred download data: the file is read only when the user clicks."""
    return lambda: runner.retry(path.read_bytes)


def _zip_reader(run_dir: Path) -> Callable[[], bytes]:
    """Deferred zip: built (once) and read only when the user clicks."""
    return lambda: history.zip_bytes(run_dir)


def summary_rows(summaries: Dict[str, dict]) -> List[dict]:
    """One row per day; list values (operations, errors) shown as text."""
    rows: List[dict] = []
    for day, summary in summaries.items():
        row = {"day": day}
        for name, value in summary.items():
            row[name] = "; ".join(map(str, value)) if isinstance(value, (list, tuple)) else value
        rows.append(row)
    return rows


def markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str] = ()) -> str:
    """A GitHub-style Markdown table (columns default to every key, in first-seen order)."""
    columns = list(columns) or list(dict.fromkeys(key for row in rows for key in row))

    def cell(value: object) -> str:
        text = "" if value is None else str(value)
        return text.replace("\\", "\\\\").replace("|", "\\|").replace("$", "\\$").replace("\n", " ") or " "

    lines = ["| " + " | ".join(cell(c) for c in columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines)


def _publish_controls(run_dir: Path, user: str, key: str) -> None:
    """Copy the reports to FUNDMANAGER_UI_PUBLISH_ROOT, behind a confirmation; hidden when that is unset."""
    dest = runner.publish_root()
    if dest is None:
        st.caption("Publishing is off (FUNDMANAGER_UI_PUBLISH_ROOT is not set).")
        return
    with st.expander(f"Publish to {dest}"):
        st.write("Copies this run's reports into the shared folder (same sub-folders). Existing files are "
                 "left untouched and listed as skipped.")
        sure = st.checkbox("I want to publish these reports", key=f"{key}-publish-sure-{run_dir.name}")
        if st.button("Publish", key=f"{key}-publish-{run_dir.name}", disabled=not sure):
            try:
                copied, skipped = runner.publish(run_dir, dest, by=user)
            except (OSError, ValueError) as exc:
                st.error(f"Publish failed: {exc}")
            else:
                st.success(f"Published {len(copied)} file(s) to {dest}." if copied else
                           "Nothing new to publish: every file is already there.")
                if skipped:
                    st.warning("Skipped (already there): " + ", ".join(p.name for p in skipped))
