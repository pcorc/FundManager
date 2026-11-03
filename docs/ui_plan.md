# FundManager UI plan (Streamlit)

Status: built (v1). Launch with `python -m streamlit run ui/app.py`; usage, settings and safety notes are in
the README's "Web UI" section. This file records the design and what was built.

## Goal
Give ops and compliance a web page to:
- pick funds, by fund group (cohort) or individually
- pick what to run: compliance, trading compliance, holdings recon, NAV recon, or one or more individual compliance rules
- pick an as-of date or a date range
- download the reports

The UI holds no business logic. Everything goes through `processing.api`; `ui/` only assembles `RunRequest`s.

## Architecture
```
Browser ──> Streamlit app (ui/app.py + pages)
              │  catalog() (cached), rule_catalog(), select_funds(), validate()
              │
              └─ ui/runner.py: subprocess  python -m processing.worker request.json --result result.json
                    │  cwd = run folder, PYTHONPATH = repo, PYTHONUNBUFFERED=1
                    │  stdout -> progress.jsonl (one JSON ProgressEvent per line), stderr -> run.log
                    └─ writes reports to <output_root>/<run_id>/reports/ and result.json
```
- **One run per worker process.** Cohort sets, Fund objects and logging are process-global, so a run never shares a process with the UI or with another run. The worker's cwd is its run folder, so `logs/compliance.log` lands there too.
- **Inputs are validated before launch.** The app calls `validate(request)` in-process (plus a few UI checks: no operation, empty fund selection, incomplete range, no business days, no rule picked, email not confirmed), so mistakes show up next to the form and **Run** stays disabled.
- **Progress is read from disk.** The worker's stdout goes straight to `progress.jsonl`; a fragment (`st.fragment(run_every=1s)`) re-reads it, so the page stays responsive, survives reruns and refreshes, and allows Cancel. The bar is driven by `index`/`total` of the `fund` events (two passes for trading), across days by the `day` events; a step indicator shows `stage` (db → load → process → reports → email → done). A log pane tails `run.log`.
- **A watcher thread** in the UI process waits for the worker, kills it at the timeout, writes the final `meta.json` (and a `result.json` if the worker died without one) and releases the lock.
- **Results come from `result.json`.** The app reads `files` (only files that exist inside the run folder), `errors` and `summaries`. It shows one download button per file, a "Download all (zip)" button (`bundle()`), and a summary table per day.

## Screens
Pages are picked in the sidebar (a radio, so Re-run can switch page from a button callback).
1. **New run**
   - **Fund groups:** multiselect over `catalog()["cohorts"]` (shown with their sizes). A fund-count badge updates as you pick (same rule as `select_funds`).
   - **Funds:** multiselect, narrowed by the chosen groups. With no groups chosen it lists every fund.
   - **Exclude:** multiselect.
   - **Operations:** a Run type radio (End of day / Trading compliance); for End of day, checkboxes for Compliance, Holdings recon and NAV recon. Trading can't be mixed with the others; `validate` enforces this too.
   - **Compliance rules:** "All rules" (default, sends `FULL_COMPLIANCE_TESTS`), or a multiselect over `rule_catalog()` labels, so you can run any single rule.
   - **Dates:** "As of" with one date picker (default: previous business day), or "Range" with a start/end picker and a choice of *One report per day* or *One stacked workbook* (EOD only; optional *Also write daily reports*). The caption shows how many business days the run covers.
   - **Options:** create PDF (on), send email (**off** by default, and a second "Yes, send the emails" tick is required), output tag.
   - **Run** button. The widgets are plain (not an `st.form`) so the badge and validation update live. While any run is active the form is disabled and the progress panel shows; a refreshed or second browser reattaches through the lock file.
2. **Run history**
   - A table of past runs from each run folder (`meta.json` + `result.json`): who, when, status, dates, request summary, files, exit code, publish count.
   - Pick a run to see its results; **Re-run** reloads that request into the form (email always off).
3. **Reports browser:** finished runs grouped by report date, with their downloads.

Results (on New run, History and Reports): status, errors, summary per day, downloads, zip, run-log tail, and **Publish**.

## Files
- `ui/app.py`: entry point: page config, password gate, username, cached catalog (with a DB-free fallback), sidebar navigation.
- `ui/new_run.py`: the New run page, `Choices` → `build_request()` → `validate()`, and Re-run pre-fill.
- `ui/panels.py`: the progress fragment and the results panel (downloads, zip, publish, Markdown tables).
- `ui/history_page.py`: the Run history and Reports pages.
- `ui/runner.py`: run folders, the worker launch, the lock file, timeout, cancel, progress parsing and publish.
- `ui/history.py`: reads run folders back (`RunRecord`, `load_result`, `load_request`, `zip_reports`), tolerating partial or corrupt folders.
- `tests/test_ui_runner.py`, `tests/test_ui_history.py`, `tests/test_ui_app.py` (`AppTest`) with `tests/fake_worker.py`; no DB.
- `requirements.txt`: adds `streamlit==1.66.0` to the pins.

## Run folders
`<FUNDMANAGER_UI_OUTPUT_ROOT>/<YYYYMMDD-HHMMSS>-<user>-<tag>/` (user and tag reduced to `[A-Za-z0-9._-]`):
`request.json` (with `output_dir` forced to `reports/`), `progress.jsonl`, `run.log`, `result.json`,
`meta.json` (user, started, finished, summary, pid, exit code, cancelled / timed-out / orphaned flags, publish
history), `logs/`, `reports/` and, once the results are shown, `<run_id>.zip`. `<root>/.run.lock` holds the
active run's worker PID and folder; a lock whose process is gone is stale and is cleared (its run is marked
orphaned).

## Configuration and deployment
- **Secrets:** the DB connection string and SMTP settings come from `.env` or the host environment (see `.env.example`). Nothing secret goes in the UI code or `st.secrets`, apart from the optional UI password. DB errors are shown and logged by type only.
- **Output:** `FUNDMANAGER_UI_OUTPUT_ROOT`, default `<repo>/ui_runs` (gitignored). A UI run never writes to the shared `G:` drive. **Publish** copies a finished run's `reports/` to `FUNDMANAGER_UI_PUBLISH_ROOT` (same sub-folders, existing files skipped) after a confirmation tick; with that variable unset the button is not shown.
- **Timeout / cancel:** `FUNDMANAGER_UI_TIMEOUT_MINUTES` (default 120, 0 = none). Cancel and timeout stop the worker's process tree (`taskkill /T` on Windows, then `terminate()`/`kill()`).
- **Hosting:** one internal Windows host or VM running `streamlit run ui/app.py --server.address 0.0.0.0`, reachable only on the internal network or VPN. Access control is one of:
  - (a) behind the company SSO reverse proxy (preferred): set `FUNDMANAGER_UI_USER_HEADER` to the header carrying the username; requests without it are refused, or
  - (b) a simple shared password: `FUNDMANAGER_UI_PASSWORD` in the environment or `.streamlit/secrets.toml` (compared with `hmac.compare_digest`).

  The UI writes the username into each run's `meta.json` (and the folder name) for the audit trail. Without a header the username is the OS account running Streamlit.
- **Scheduling:** the existing scheduled CLI runs (`python main.py …`) are unchanged. The UI is for ad-hoc and back-dated runs.
- **Known environment issue:** Streamlit 1.66 installed pyarrow 26, which needs NumPy 2, while the project pins NumPy 1.26. `st.dataframe`/`st.table` therefore fail, so the UI renders its tables as Markdown.

## Build order (done)
1. `ui/runner.py`, plus a test that runs a fake worker.
2. `ui/history.py`: run history and re-run.
3. The New run screen with validation and progress; results, downloads and the zip bundle; history and reports pages; lock file, cancel, the email confirmation and the Publish button; password gate.
4. `AppTest` coverage.
5. Deployment notes and a README section.

## Out of scope for v1
Multi-user concurrent runs (one run per output root at a time), report previews in the browser (beyond downloads), and editing fund definitions.
