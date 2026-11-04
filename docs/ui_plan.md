# FundManager UI plan (Streamlit)

Status: built (v2: the soft dark redesign with results on the page). Launch with `python -m streamlit run ui/app.py`; usage, settings and safety notes are in
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
- **Results come from `result.json` and `views.json`.** The app reads `files` (only files that exist inside the run folder), `errors` and `summaries` from `result.json`, and the per-day display tables from `views.json` (schema in `reporting.result_views`; the same labels, numbers and PASS/FAIL text as the PDFs). A run without a readable `views.json` of schema version 1 (older runs, a worker that died early) shows the summary table per day instead. Downloads: one button per file and a "Download all (zip)" button (`bundle()`).

## Screens
A top bar holds the brand, the page navigation (a segmented control on the `page` key, so Re-run can switch page from a button callback) and a popover under the user's name with the output and publish folders and *Reload fund groups*.

1. **New run**: two panes, modelled on the reference design the users supplied.
   - **Settings panel** (left, `ui/settings_panel.py`), in order:
     - **Date Range:** *Single day* (one *As of* date, default the previous business day) or *Date range* (*Start Date* / *End Date* side by side, then *One report per day* or *One stacked workbook*; stacked is EOD only, so it is reset to per day and greyed out for Trading; choosing stacked turns *Create PDF* off like the CLI's `time_series` preset; optional *Also write daily reports*). Weekend dates are snapped in the UI (as-of: back to Friday; range: first to last business day), so the request, history and Reports name the days that run; a note says so. There is no holiday calendar.
     - **Select Fund (N):** one chip per non-empty fund group (cohort, with its size) that ticks its funds, or unticks them when they are all ticked already; *Clear*; a ticker filter; and a three-column grid of fund tiles (checkboxes `nrf_<ticker>`, highlighted when ticked). The picks are the list `nr_funds`; the tiles are synced from it every run, so the filter can hide tiles without losing picks. The request carries the ticked tickers (no cohorts, so unticking a tile is how a fund is excluded). Re-run of a request with groups and exclusions ticks the funds they resolve to (same rule as `select_funds`). Nothing is ticked on a fresh form, and Run stays disabled until a fund is.
     - **Operations:** *End of day* or *Trading compliance* (a segmented control; trading can't be mixed with the others, `validate` enforces this too); for End of day, tiles for Compliance checks, Holdings reconciliation and NAV reconciliation.
     - **Compliance Checks (N)** (when compliance or trading runs): *Select All* and one tile per rule of `FULL_COMPLIANCE_TESTS` (labels from `rule_catalog()`; the summary sheet always runs). Every tile ticked sends "All rules" (`FULL_COMPLIANCE_TESTS`); otherwise the exact ticked list, in catalog order.
     - **Options:** create PDF (on), send email (**off** by default, and a second "Yes, send the emails" tick is required; both are cleared after each run start, for Trading too since fix D16), output tag (blank = the CLI preset's tag: `eod`, `compliance`, `reconciliation`, `nav`, `time_series` or `trading_compliance`, so file names match CLI runs).
     - **RUN ...** (primary; the label follows the operations: *Run compliance checks*, *Run checks & recons*, *Run reconciliation*, *Run trading compliance*) and **RESET** (every pick back to its default, email off), pinned at the foot of the panel (`nr_actions`).
   - The settings column is sticky with its own scroll, so RUN and the top of the right pane stay in view together. Below 1200px wide the settings and the preview stack (Streamlit itself only stacks columns below 640px).
   - **Configuration Preview** (right, `ui/preview.py`): metric cards (Days Selected = business days, Funds Selected, Checks Selected, Operations) and the sentence *Your checks will run for: 3 funds × 1 day × 13 checks (11 rules + 2 recons) = 39 total operations* (checks = compliance rules + recon operations, broken down when both run; trading counts its two passes), with an estimated completion time: the median seconds per fund-day of the last 40 successful runs in the output root (`history.fund_day_seconds`, cached for two minutes), else a heuristic (10 s per day + 7 s per fund-day, measured on the live DB), always labelled as an estimate. Validation problems (`validate` plus the UI checks: no fund, no operation, incomplete or empty range, no check picked, email not confirmed, a stacked run with recon operations but no daily reports) show here inline and keep Run disabled.
   - Under the preview: **before a run**, cards describing each picked check (what it checks, its conditions and thresholds, which funds it applies to, the report footnotes; from `reporting.result_views.rule_descriptions()`, no DB) and each recon / trading operation; **during a run**, the progress panel, shown first in the right pane above the preview (bar, stage stepper db → load → process → reports → email → done, the current step and fund, Cancel, log tail); **after a run**, its results (below), with a *Results / Check guide* switch.
   - Widgets are plain (not an `st.form`) so the preview updates live. Picks survive page switches and hidden widgets (`new_run.keep_form_state()` re-assigns their keys each run; confirmation ticks are deliberately not kept). While any run is active the form is disabled and the progress panel shows; a refreshed or second browser reattaches through the lock file.
2. **Run history:** summary cards (runs, succeeded, with errors, failed, users), a table of the newest 25 runs (started, user, status pill, dates, request, files, publishes, folder), then *Open a run*: pick any run to see the same results view; **Re-run** reloads that request into the form (email and confirmations always off; a stacked single day stays stacked, an exact rule list stays exact).
3. **Reports:** finished runs grouped by report date (the `ReportFile.day` of their files; a stacked workbook under its range), showing that run's results and that date's downloads. The Report date picks the results' day too (no second date picker), so the tables always match the downloads.

**Results** (`ui/results.py`, on New run, History and Reports): a status banner and the errors; then per day (a picker for ranges) an overview matrix of funds × checks with PASS / FAIL / N/A pills (overall first; NAV and holdings recon when they ran; the Fund column stays put when the matrix scrolls sideways; a note when IRC was picked but no fund is a VIT; on a trading day the trading overview first, coloured like the PDF); then, for a chosen fund (the first failing one by default), the rule tables as cards in rows of two, read in pairs and aligned at the top of each row (a soft blue header band with the rule title and status, the PDF's label / value rows with PASS in soft green and FAIL in soft red, *More detail*, the GICS exposure tables, fund notes and the rule's footnotes, folded when long; numbers in tabular figures with a true minus sign, blank values as a faint dash), then across both columns the trade activity (trading day) and the NAV reconciliation card (summary and checks beside the G/L sections), then the holdings reconciliation (break counts, the discrepancy tables with the PDF's section titles and readable column names, the first 10 rows as in the PDF); on a trading day also the checks before / after the trades and an ex-post / ex-ante switch. Then the downloads (one per distinct file, labels joined when two results share a workbook; the file name is the tooltip), a zip built when it is first downloaded, **Publish** and the run-log tail. Errors, log tails and the exception texts in `views.json` (per-fund errors, rule reasons and notes, redacted by `history.load_views`) pass through `runner.redact()` (URL passwords, the `DB_CONNECTION_STRING` / `FUNDMANAGER_SMTP_PASSWORD` values, MySQL user names). The Publish confirmation is keyed per run folder.

## Look
A soft dark theme: `.streamlit/config.toml` sets Streamlit's palette (background `#1F2430`, surfaces `#2C3446`, text `#E3E7EF`, secondary text `#B0B9CA` and hints `#929CB0` (both at least 4.5:1 on every surface), periwinkle accent `#7C9CF5`, borders `#363F52`, the Inter font) and `ui/theme.py` holds the matching CSS (pills: PASS `#7FCB9B`, FAIL `#EE8E8E`, WARN `#E6C07B`, N/A `#A3ACBD`; cards, tables, tiles, stepper), injected once per script run. Widget groups are styled by key prefix (`nrf_`, `nrr_`, `nr_op_`, `nr_settings`, `fm_nav`). All custom HTML is built in `ui/components.py` and `ui/results.py`, with every value escaped, and rendered with `st.html`.

## Files
- `ui/app.py`: entry point: page config, theme, password gate, username, cached catalog (with a DB-free fallback), top bar navigation.
- `ui/new_run.py`: the New run page layout and state, `Choices` → `build_request()` → `validate()`, and Re-run pre-fill.
- `ui/settings_panel.py`: the New run settings panel (dates, fund chips and tiles, operations, checks, options, RUN / RESET).
- `ui/preview.py`: the Configuration Preview (counts, sentence, time estimate) and the check guide.
- `ui/results.py`: the results view from `views.json` (matrix, rule cards, NAV, holdings, trading).
- `ui/panels.py`: the progress fragment and the results panel (banner, errors, results, downloads, zip, publish, log).
- `ui/theme.py`, `ui/components.py`, `.streamlit/config.toml`: the palette, the CSS and the escaped HTML builders.
- `ui/history_page.py`: the Run history and Reports pages.
- `ui/runner.py`: run folders, the worker launch, the lock file, timeout, cancel, progress parsing and publish.
- `ui/history.py`: reads run folders back (`RunRecord`, `load_result`, `load_request`, `load_views`, `zip_reports`, `fund_day_seconds`), tolerating partial or corrupt folders.
- `tests/test_ui_runner.py`, `tests/test_ui_history.py`, `tests/test_ui_app.py` (`AppTest`), `tests/test_ui_views.py` (the HTML and preview builders) with `tests/fake_worker.py` and `tests/ui_views_fixture.py` (hand-made views); no DB.
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
- **Known environment issue:** Streamlit 1.66 installed pyarrow 26, which needs NumPy 2, while the project pins NumPy 1.26. `st.dataframe`/`st.table` therefore fail, so the UI renders its tables as escaped HTML (`st.html`).

## Build order (done)
1. `ui/runner.py`, plus a test that runs a fake worker.
2. `ui/history.py`: run history and re-run.
3. The New run screen with validation and progress; results, downloads and the zip bundle; history and reports pages; lock file, cancel, the email confirmation and the Publish button; password gate.
4. `AppTest` coverage.
5. Deployment notes and a README section.

## API gaps (worked around in `ui/`)
- **Weekend handling differs by mode** (`business_date_range` snaps a weekend start back to Friday, `iter_business_days` skips it); the UI snaps dates itself so both modes run the days shown.

## Out of scope
Multi-user concurrent runs (one run per output root at a time) and editing fund definitions. The results view shows the PDF's tables; the full lists (discrepancy rows past the first 10, every GICS industry) stay in the Excel reports.
