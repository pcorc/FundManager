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
- **Inputs are validated before launch.** The app calls `validate(request)` in-process (plus a few UI checks: no operation, empty fund selection, incomplete range, no business days, no rule picked, email not confirmed, a stacked run with recon operations but no daily reports), so mistakes show up next to the form and **Run** stays disabled.
- **Progress is read from disk.** The worker's stdout goes straight to `progress.jsonl`; a fragment (`st.fragment(run_every=1s)`) re-reads it, so the page stays responsive, survives reruns and refreshes, and allows Cancel. The bar is driven by `index`/`total` of the `fund` events (two passes for trading), across days by the `day` events; a step indicator shows `stage` (db → load → process → reports → email → done). A log pane tails `run.log`.
- **A watcher thread** in the UI process waits for the worker, kills it at the timeout, writes the final `meta.json` (and a `result.json` if the worker died without one) and releases the lock. Each of those steps runs even if the one before failed. A worker outlives a Streamlit restart (it is in its own process group); the progress fragment then calls `runner.refresh()` on every poll, which finalizes the run once the worker is gone and enforces its timeout.
- **Windows file sharing.** A file another session is reading cannot be replaced or deleted, so JSON writes (temp file + `os.replace`), lock removal and zip writes retry briefly on `PermissionError`; zips are built under a per-run lock with per-thread temp names.
- **Results come from `result.json`.** The app reads `files` (only files that exist inside the run folder), `errors` and `summaries`. It shows one download button per file, a "Download all (zip)" button (`bundle()`), and a summary table per day.

## Screens
Pages are picked in the sidebar (a radio, so Re-run can switch page from a button callback).
1. **New run**
   - **Fund groups:** multiselect over `catalog()["cohorts"]` (shown with their sizes). A fund-count badge updates as you pick (same rule as `select_funds`).
   - **Funds:** multiselect, narrowed by the chosen groups. With no groups chosen it lists every fund.
   - **Exclude:** multiselect.
   - **Operations:** a Run type radio (End of day / Trading compliance); for End of day, checkboxes for Compliance, Holdings recon and NAV recon. Trading can't be mixed with the others; `validate` enforces this too.
   - **Compliance rules:** "All rules" (default, sends `FULL_COMPLIANCE_TESTS`), or a multiselect over `rule_catalog()` labels, so you can run any single rule.
   - **Dates:** "As of" with one date picker (default: previous business day), or "Range" with a start/end picker and a choice of *One report per day* or *One stacked workbook* (EOD only, so it is reset to per day and greyed out for Trading; choosing stacked turns *Create PDF* off like the CLI's `time_series` preset; optional *Also write daily reports*). Weekend dates are snapped in the UI (as-of: back to Friday; range: first to last business day), so the request, history and Reports name the days that run; a note says so, and the caption lists the days. There is no holiday calendar.
   - **Options:** create PDF (on), send email (**off** by default, and a second "Yes, send the emails" tick is required; both are cleared after each run start), output tag (blank = the CLI preset's tag: `eod`, `compliance`, `reconciliation`, `nav`, `time_series` or `trading_compliance`, so file names match CLI runs).
   - **Trading compliance and email:** the same *Send email* box and confirmation apply (the trading path honours `send_email` since fix D16).
   - **Run** button. The widgets are plain (not an `st.form`) so the badge and validation update live. Picks survive page switches and hidden widgets (`new_run.keep_form_state()` re-assigns their keys each run; confirmation ticks are deliberately not kept). While any run is active the form is disabled and the progress panel shows; a refreshed or second browser reattaches through the lock file.
2. **Run history**
   - A table of past runs from each run folder (`meta.json` + `result.json`): who, when, status, dates, request summary, files, exit code, publish count.
   - Pick a run to see its results; **Re-run** reloads that request into the form (email and confirmations always off; a stacked single day stays stacked, an exact rule list stays exact).
3. **Reports browser:** finished runs grouped by report date (the `ReportFile.day` of their files; a stacked workbook under its range), showing that date's downloads.

Results (on New run, History and Reports): status, errors, summary per day, downloads (one per distinct file, labels joined when two results share a workbook), a zip built when it is first downloaded, run-log tail, and **Publish**. Errors and log tails pass through `runner.redact()` (URL passwords, the `DB_CONNECTION_STRING` / `FUNDMANAGER_SMTP_PASSWORD` values, MySQL user names). The Publish confirmation is keyed per run folder.

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
history), `logs/`, `reports/` and, once downloaded, `<run_id>.zip`. `<root>/.run.lock` holds the
active run's worker PID, its process creation time and folder; a lock whose process is gone, or whose PID was
reused by another process (creation time differs), is stale. A stale lock is retired atomically: renamed to a
unique `.run.lock.stale.<id>` tombstone, so only the session whose rename wins finalizes the orphan run (marked
orphaned) and the others retry. Cancel of a worker started by an earlier UI process kills it by PID only when the
creation time matches. The worker's stderr is UTF-8 (`PYTHONIOENCODING=utf-8`). When `<run folder>/reports` is so deep that report paths would pass Windows MAX_PATH (260) on a host
without long paths enabled, the runner passes `output_dir` in extended-length (`\\?\`) form; history, zip and
publish handle both forms.

## Configuration and deployment
- **Secrets:** the DB connection string and SMTP settings come from `.env` or the host environment (see `.env.example`). Nothing secret goes in the UI code or `st.secrets`, apart from the optional UI password. A failed catalog load is shown and logged by type only (and cached for 60 s, so a down DB does not cost a connect timeout per click); worker errors and log tails are shown redacted.
- **Output:** `FUNDMANAGER_UI_OUTPUT_ROOT`, default `<repo>/ui_runs` (gitignored). A UI run never writes to the shared `G:` drive. **Publish** copies a finished run's `reports/` to `FUNDMANAGER_UI_PUBLISH_ROOT` (same sub-folders, existing files skipped) after a confirmation tick; with that variable unset the button is not shown.
- **Timeout / cancel:** `FUNDMANAGER_UI_TIMEOUT_MINUTES` (default 120, 0 = none). Cancel and timeout stop the worker's process tree (`taskkill /T` on Windows, then `terminate()`/`kill()`).
- **Hosting:** one internal Windows host or VM running `python -m streamlit run ui/app.py`, reachable only on the internal network or VPN. Listen on `127.0.0.1` behind the SSO proxy (a), or on `0.0.0.0` in shared-password mode (b); see the README *Web UI* section. Access control is one of:
  - (a) behind the company SSO reverse proxy (preferred): set `FUNDMANAGER_UI_USER_HEADER` to the header carrying the username; requests without it are refused. The header is trusted as is, so Streamlit must listen on `127.0.0.1` (or be firewalled) so only the proxy reaches it, and the proxy must overwrite any client-supplied copy of the header, or
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

## API gaps (worked around in `ui/`)
- **Weekend handling differs by mode** (`business_date_range` snaps a weekend start back to Friday, `iter_business_days` skips it); the UI snaps dates itself so both modes run the days shown.

## Out of scope for v1
Multi-user concurrent runs (one run per output root at a time), report previews in the browser (beyond downloads), and editing fund definitions.
