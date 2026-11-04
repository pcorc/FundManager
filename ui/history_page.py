"""The Run history and Reports pages, both read from the run folders under the UI output root."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import streamlit as st

from ui import components as ui
from ui import history, new_run, panels
from ui.history import RunRecord

HISTORY_LIMIT = 300
TABLE_ROWS = 25  # newest runs in the history table; every run stays pickable below it


def history_table_html(records: List[RunRecord]) -> str:
    rows = [[panels.clock(r.started), r.user, panels.status_pill(r.status), r.as_of.replace("..", " → "),
             ui.summary_text(r.summary), r.files, r.published or "", r.name] for r in records]
    return ui.panel(ui.data_table(["Started", "User", "Status", "Dates", "Request", "Files", "Published", "Folder"],
                                  rows, raw=(2,), css="fm-history"))


def _stats(records: List[RunRecord]) -> str:
    ok = sum(r.status == "ok" for r in records)
    warn = sum(r.status == "ok with errors" for r in records)
    bad = sum(r.status in ("failed", "invalid", "timed out") for r in records)
    return ui.metric_cards([(len(records), "Runs", ""), (ok, "Succeeded", ""), (warn, "With errors", ""),
                            (bad, "Failed", ""), (len({r.user for r in records}), "Users", "")])


def render_history(user: str) -> None:
    """Past runs (who, when, what, status, files); pick one to see its results or Re-run it."""
    st.title("Run history")
    records = history.list_runs(limit=HISTORY_LIMIT)
    if not records:
        ui.show(ui.empty("No runs yet", "Start one on the New run page."))
        return
    ui.show(_stats(records) + ui.heading("Recent runs", f"newest {min(len(records), TABLE_ROWS)} of {len(records)}")
            + history_table_html(records[:TABLE_ROWS]))
    by_name = {r.name: r for r in records}
    ui.show(ui.heading("Open a run", "its results, downloads and Re-run"))
    pick, rerun = st.columns([4, 1.2], vertical_alignment="bottom")
    name = pick.selectbox("Run", list(by_name), key="hist_pick",
                          format_func=lambda n: f"{n}  ·  {by_name[n].status}  ·  {ui.summary_text(by_name[n].summary)}")
    record = by_name[name]
    rerun.button("Re-run with these settings", key="hist_rerun", on_click=new_run.request_rerun,
                 args=(record.run_dir,), help="Loads this request into the New run form (email off).",
                 icon=":material/replay:", width="stretch")
    if record.status == "running":
        ui.show(ui.callout("info", "This run is still in progress; follow it on the New run page."))
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
        ui.show(ui.empty("No reports yet", "Reports of finished runs appear here by report date."))
        return
    day_col, run_col = st.columns([1, 3])
    day = day_col.selectbox("Report date", sorted(by_day, reverse=True), key="rep_day")
    runs = {r.name: (r, d) for r, d in by_day[day]}
    name = run_col.selectbox("Run", list(runs), key="rep_run",
                             format_func=lambda n: (f"{n}  ·  {runs[n][0].user}  ·  {runs[n][0].status}  ·  "
                                                    f"{ui.summary_text(runs[n][0].summary)}"))
    record, report_day = runs[name]
    panels.results_panel(record.run_dir, user, key="rep", day=report_day)
