"""The Run history and Reports pages, both read from the run folders under the UI output root."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import streamlit as st

from ui import history, new_run, panels
from ui.history import RunRecord

HISTORY_LIMIT = 300


def render_history(user: str) -> None:
    """A table of past runs (who, when, what, status, files); pick one to see its results or Re-run it."""
    st.title("Run history")
    records = history.list_runs(limit=HISTORY_LIMIT)
    if not records:
        st.info("No runs yet. Start one on the New run page.")
        return
    st.markdown(panels.markdown_table([{
        "started": r.started, "user": r.user, "status": r.status, "dates": r.as_of, "request": r.summary,
        "files": r.files, "exit": r.exit_code, "published": r.published, "folder": r.name,
    } for r in records]))
    by_name = {r.name: r for r in records}
    name = st.selectbox("Run", list(by_name), key="hist_pick",
                        format_func=lambda n: f"{n} · {by_name[n].status} · {by_name[n].summary}")
    record = by_name[name]
    st.button("Re-run with these settings", key="hist_rerun", on_click=new_run.request_rerun,
              args=(record.run_dir,), help="Loads this request into the New run form (email off).")
    if record.status == "running":
        st.info("This run is still in progress; follow it on the New run page.")
    else:
        panels.results_panel(record.run_dir, user, key="hist")


def render_reports(user: str) -> None:
    """Browse the reports of finished runs by report date (each day a run wrote reports for)."""
    st.title("Reports")
    by_day: Dict[str, List[Tuple[RunRecord, str]]] = defaultdict(list)  # shown date -> (run, ReportFile.day)
    for record in history.list_runs(limit=HISTORY_LIMIT):
        for report_day in record.report_days:
            shown = f"{record.as_of} (stacked)" if report_day == "stacked" else report_day
            by_day[shown].append((record, report_day))
    if not by_day:
        st.info("No reports yet.")
        return
    day = st.selectbox("Report date", sorted(by_day, reverse=True), key="rep_day")
    runs = {r.name: (r, d) for r, d in by_day[day]}
    name = st.selectbox("Run", list(runs), key="rep_run",
                        format_func=lambda n: f"{n} · {runs[n][0].user} · {runs[n][0].status} · {runs[n][0].summary}")
    record, report_day = runs[name]
    panels.results_panel(record.run_dir, user, key="rep", day=report_day)
