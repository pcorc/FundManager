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
     - **Date Range:** *Single day* (one *As of* date, default the previous business day) or *Date range* (*Start Date* / *End Date* side by side, then *One report per day* or *One stacked workbook*; choosing stacked turns *Create PDF* off like the CLI's `time_series` preset; optional *Also write daily reports*). Weekend dates are snapped in the UI (as-of: back to Friday; range: first to last business day), so the request, history and Reports name the days that run; a note says so. There is no holiday calendar.
     - **Select Fund (N):** one chip per non-empty fund group (cohort, with its size) that ticks its funds, or unticks them when they are all ticked already; *Clear*; a ticker filter; and a three-column grid of fund tiles (checkboxes `nrf_<ticker>`, highlighted when ticked). The picks are the list `nr_funds`; the tiles are synced from it every run, so the filter can hide tiles without losing picks. The request carries the ticked tickers (no cohorts, so unticking a tile is how a fund is excluded). Re-run of a request with groups and exclusions ticks the funds they resolve to (same rule as `select_funds`). Nothing is ticked on a fresh form, and Run stays disabled until a fund is.
     - **Operations** (end of day): tiles for Compliance checks, Holdings reconciliation and NAV reconciliation, and a hint that trading compliance has its own page (Trading Compliance).
     - **Compliance Checks (N)** (when compliance runs): *Select All* and one tile per rule of `FULL_COMPLIANCE_TESTS` (labels from `rule_catalog()`; the summary sheet always runs). Every tile ticked sends "All rules" (`FULL_COMPLIANCE_TESTS`); otherwise the exact ticked list, in catalog order.
     - **Options:** create PDF (on), send email (**off** by default, and a second "Yes, send the emails" tick is required; both are cleared after each run start), output tag (blank = the CLI preset's tag: `eod`, `compliance`, `reconciliation`, `nav`, `time_series` or `trading_compliance`, so file names match CLI runs).
     - **RUN ...** (primary; the label follows the operations: *Run compliance checks*, *Run checks & recons*, *Run reconciliation*) and **RESET** (every pick back to its default, email off), pinned at the foot of the panel (`nr_actions`).
   - The settings column is sticky with its own scroll, so RUN and the top of the right pane stay in view together. Below 1200px wide the settings and the preview stack (Streamlit itself only stacks columns below 640px).
   - **Configuration Preview** (right, `ui/preview.py`): metric cards (Days Selected = business days, Funds Selected, Checks Selected, Operations) and the sentence *Your checks will run for: 3 funds × 1 day × 13 checks (11 rules + 2 recons) = 39 total operations* (checks = compliance rules + recon operations, broken down when both run; trading counts its two passes), with an estimated completion time: the median seconds per fund-day of the last 40 successful runs in the output root (`history.fund_day_seconds`, cached for two minutes), else a heuristic (10 s per day + 7 s per fund-day, measured on the live DB), always labelled as an estimate. Validation problems (`validate` plus the UI checks: no fund, no operation, incomplete or empty range, no check picked, email not confirmed, a stacked run with recon operations but no daily reports) show here inline and keep Run disabled.
   - Under the preview: **before a run**, cards describing each picked check (what it checks, its conditions and thresholds, which funds it applies to, the report footnotes; from `reporting.result_views.rule_descriptions()`, no DB) and each recon / trading operation; **during a run**, the progress panel, shown first in the right pane above the preview (bar, stage stepper db → load → process → reports → email → done, the current step and fund, Cancel, log tail); **after a run**, its results (below), with a *Results / Check guide* switch.
   - Widgets are plain (not an `st.form`) so the preview updates live. Picks survive page switches and hidden widgets (`new_run.keep_form_state()` re-assigns their keys each run; confirmation ticks are deliberately not kept). While any run is active the form is disabled and the progress panel shows; a refreshed or second browser reattaches through the lock file.
2. **Trading Compliance** (`ui/trading_page.py`; second in the top bar): checks the funds' *proposed* trades (staged in Vest, not yet executed) before they execute, comparing **ex-ante** (the holdings as they stand, with custodian cash and TNA at the previous business day's close) with **ex-post** (the same holdings assuming the staged trades execute). One business day per run (`trading_compliance` operation, built with `new_run.build_request(kind=KIND_TRADING)` and validated by `processing.api.validate`, under the same one-run lock).
   - **Settings panel** (left, same look as New run, keys `tc_*`, tiles `tcf_` / `tcr_`): *Trade Date* (default today; a weekend snaps to the Friday before; the hint names the T-1 close the cash comes from); **Select Fund** with a *Staged trades · N* quick-pick chip and a dot on the tiles of the funds that have staged trades on the date (`processing.api.staged_trades`, read-only, cached five minutes per date; on failure the page says so and funds are picked by hand), the group chips, the search and the tiles; **Compliance Checks** (*Select All* + one tile per rule); **Options** (PDF, email off by default with the second confirmation, cleared once a run starts; output tag, blank = `trading_compliance`); **RUN TRADING CHECK** / **RESET**. The settings sections are the New run ones (`settings_panel.funds_section` / `rules_section` / `options_section` with a `FormKeys`).
   - **Right pane:** before a run, the *Trading check preview* (cards: trade date, funds (how many with staged trades), checks, 2 passes; one card each for ex-ante and ex-post; the operations sentence and time estimate; problems inline; a note for picked funds without staged trades, reported as "no proposed trades") and the check guide; during a run, the shared progress panel; after it, a one-line *Next check* summary and the **dashboard** (`ui/trading_results.py`, `ui/trading_drilldown.py`, `ui/charts.py`), read from `views.json` `days[day]["dashboard"]` (built by `reporting/trading_dashboard.py`, whose docstring has the schema and definitions):
     1. *KPI strip* in two rows: proposed trades (funds with trades of those selected, buys, sells, net, estimated post-trade cash with the funds below zero) and the compliance outcome (staying compliant, going out of compliance (soft red), coming into compliance (soft green), still failing (amber); an icon and label on each, not colour alone).
     2. *Compliance impact* matrix: funds with proposed trades (breaks first) × rules, each cell the before → after transition (PASS → PASS a faint tick; BREAKS a solid red chip; CURES green; STILL FAILS amber; N/A grey), a hover with both statuses, a legend, and an *Only rules that change or fail* toggle (on by default). Funds without staged trades and funds without a result are listed under it.
     3. *Rules at risk*: per rule, a count bar and the funds that break, still fail, error or are cured.
     4. *Trading summary* table (one row per fund + a totals row; the Fund column stays put when it scrolls): TNA; Equity / Options premium / Treasury buys, sells, net (with contracts or % TNA beneath; treasury only when traded); traded % TNA; current cash (T-1 close), post-trade cash (estimate) and post % TNA, red when below zero; notes (negative cash, no cash data, other data notes on hover). A *Definitions* footnote gives TNA, current cash, trade value, net and the post-trade cash estimate (current − buys + sells over equity, options at premium and treasuries at price per 100 face; trades without a usable price or that only restate the position are left out; no settlement timing, fees or flows).
     5. *Trades at a glance*: three small-multiple charts in the same fund order (net trade value by asset class as diverging bars; cash before → after as a dumbbell with the region below zero washed red; traded % of TNA). Inline SVG with percentage x positions (they stretch with the card, text keeps its size), a `<title>` hover on each mark, an `aria-label` on each chart; the summary table is their table view. Asset classes are blue / orange / aqua (validated with the dataviz palette script against the card surface, dark mode, all pairs). `st.html` strips SVG (DOMPurify HTML profile), so charts render with `st.markdown(unsafe_allow_html=True)` through `components.show_svg`; every text is escaped.
     6. *Fund drill-down*: a *Funds failing a rule* filter and the fund pills (breaks first); the fund header (impact, status change, TNA, trades, buys, sells, % TNA, violations and cash before → after); its data notes; the cash waterfall (current cash → equity buys, equity sells, options net premium, treasury net → post-trade cash) beside the Summary Metrics before → after; one card per rule with before (ex-ante) and after (ex-post) side by side, changed rows highlighted with an up / down mark (failing, breaking, curing and changing rules first; the unchanged ones folded); the trade blotter per asset class (ticker, side, quantity, price, est. value, % TNA, position before → after, flags such as *no price*, *restates position*, *duplicate row*; trades left out of the cash estimate are greyed); and, folded, the full rule cards as in the PDF with an ex-post / ex-ante switch.
     7. Then the downloads, zip and Publish as for every run.
   - New run is end of day only; its Operations section says trading has its own page. A run started on one page locks both: the other page shows its progress and keeps its own form. Run history / Reports render the same dashboard for trading runs, and **Re-run** of a trading run loads it into this page (email off).
3. **Run history:** summary cards (runs, succeeded, with errors, failed, users), a table of the newest 25 runs (started, user, status pill, dates, request, files, publishes, folder), then *Open a run*: pick any run to see the same results view; **Re-run** reloads that request into its form (New run, or Trading Compliance for a trading run; email and confirmations always off; a stacked single day stays stacked, an exact rule list stays exact).
4. **Reports:** finished runs grouped by report date (the `ReportFile.day` of their files; a stacked workbook under its range), showing that run's results and that date's downloads. The Report date picks the results' day too (no second date picker), so the tables always match the downloads.

**Results** (`ui/results.py`, on New run, History and Reports): a status banner and the errors; then per day (a picker for ranges) an overview matrix of funds × checks with PASS / FAIL / N/A pills (overall first; NAV and holdings recon when they ran; the Fund column stays put when the matrix scrolls sideways; a note when IRC was picked but no fund is a VIT; a trading day shows the Trading Compliance dashboard instead, or the trading overview, coloured like the PDF, when `views.json` has no dashboard); then, for a chosen fund (the first failing one by default), the rule tables as cards in rows of two, read in pairs and aligned at the top of each row (a soft blue header band with the rule title and status, the PDF's label / value rows with PASS in soft green and FAIL in soft red, *More detail*, the GICS exposure tables, fund notes and the rule's footnotes, folded when long; numbers in tabular figures with a true minus sign, blank values as a faint dash), then across both columns the trade activity (trading day) and the NAV reconciliation card (summary and checks beside the G/L sections), then the holdings reconciliation (break counts, the discrepancy tables with the PDF's section titles and readable column names, the first 10 rows as in the PDF); on a trading day also the checks before / after the trades and an ex-post / ex-ante switch. Then the downloads (one per distinct file, labels joined when two results share a workbook; the file name is the tooltip), a zip built when it is first downloaded, **Publish** and the run-log tail. Errors, log tails and the exception texts in `views.json` (per-fund errors, rule reasons and notes, redacted by `history.load_views`) pass through `runner.redact()` (URL passwords, the `DB_CONNECTION_STRING` / `FUNDMANAGER_SMTP_PASSWORD` values, MySQL user names). The Publish confirmation is keyed per run folder.

## Look
A soft dark theme: `.streamlit/config.toml` sets Streamlit's palette (background `#1F2430`, surfaces `#2C3446`, text `#E3E7EF`, secondary text `#B0B9CA` and hints `#929CB0` (both at least 4.5:1 on every surface), periwinkle accent `#7C9CF5`, borders `#363F52`, the Inter font) and `ui/theme.py` holds the matching CSS (pills: PASS `#7FCB9B`, FAIL `#EE8E8E`, WARN `#E6C07B`, N/A `#A3ACBD`; cards, tables, tiles, stepper), injected once per script run. Widget groups are styled by key prefix (`nrf_`, `nrr_`, `nr_op_`, `nr_settings`, `fm_nav`; on the Trading page `tcf_`, `tcr_`, `tc_settings`); `TRADING_CSS` styles the dashboard. All custom HTML is built in `ui/components.py` and `ui/results.py`, with every value escaped, and rendered with `st.html`.

## Files
- `ui/app.py`: entry point: page config, theme, password gate, username, cached catalog (with a DB-free fallback), top bar navigation.
- `ui/new_run.py`: the New run page layout and state, `Choices` → `build_request()` → `validate()`, and Re-run pre-fill.
- `ui/settings_panel.py`: the New run settings panel (dates, fund chips and tiles, operations, checks, options, RUN / RESET).
- `ui/trading_page.py`: the Trading Compliance page (trade date, staged-trades quick pick, checks, options, preview, run).
- `ui/trading_results.py`, `ui/trading_drilldown.py`: the trading dashboard (KPIs, impact matrix, rules at risk, summary table, charts) and its fund drill-down.
- `ui/charts.py`: the inline SVG charts (per-fund small multiples, cash waterfall).
- `ui/preview.py`: the Configuration Preview (counts, sentence, time estimate) and the check guide.
- `ui/results.py`: the results view from `views.json` (matrix, rule cards, NAV, holdings, trading).
- `ui/panels.py`: the progress fragment and the results panel (banner, errors, results, downloads, zip, publish, log).
- `ui/theme.py`, `ui/components.py`, `.streamlit/config.toml`: the palette, the CSS and the escaped HTML builders.
- `ui/history_page.py`: the Run history and Reports pages.
- `ui/runner.py`: run folders, the worker launch, the lock file, timeout, cancel, progress parsing and publish.
- `ui/history.py`: reads run folders back (`RunRecord`, `load_result`, `load_request`, `load_views`, `zip_reports`, `fund_day_seconds`), tolerating partial or corrupt folders.
- `tests/test_ui_runner.py`, `tests/test_ui_history.py`, `tests/test_ui_app.py` (`AppTest`), `tests/test_ui_views.py` (the HTML and preview builders), `tests/test_ui_trading.py` (the trading page via `AppTest`, the dashboard builders and the SVG charts) with `tests/fake_worker.py` and `tests/ui_views_fixture.py` (hand-made views); no DB.
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
