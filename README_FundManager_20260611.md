# VestFundManager

Python platform that runs Vest's daily fund operations: compliance checks,
holdings reconciliation, NAV reconciliation, and intraday trading compliance.
One pass against the database loads everything; downstream services read from
in-memory snapshots.

---

## What runs

Five operational **modes**, all built on the same loader + fund-object
foundation. Each mode is a preset in `config/run_configurations.py`, turned
into a `RunRequest` and executed by `processing/api.py`.

| Mode | What it does | When it runs |
|---|---|---|
| `eod` | Compliance + Holdings Recon + NAV Recon, packaged into one combined PDF + Excel set | After US market close (~6:00 PM ET) |
| `compliance` | Standalone compliance check (40 Act, IRS/IRC, prospectus 80%, GICS, etc.) | Morning portfolio review (~8:00 AM ET) and ad-hoc |
| `reconciliation` | Holdings break detection between OMS and custodian | Bundled inside `eod`; rarely standalone |
| `nav` (alias for `nav_reconciliation`) | NAV gain/loss attribution; expected vs custodian NAV; option-price sensitivity matrix | Bundled inside `eod` |
| `trading_compliance` | Pre-trade (ex-ante) vs post-trade (ex-post) compliance comparison + trade activity summary | Intraday: ex-ante ~10:00 AM ET; ex-post shortly after trades execute |
| `time_series` | Compliance run across a date range, stacked into a single workbook | Ad-hoc, for audit / trend review |

Two **run shapes**:

- **Daily** — single business date (the common case).
- **Time series** — date range; iterates business days and produces a stacked
  compliance workbook in addition to per-day artefacts.

---

## Fund cohorts

Funds are grouped into four cohorts so each mode can target the right subset.
Cohorts are loaded at startup from `reconciliation.v_fund_metadata` via
`config.fund_definitions.load_cohorts_from_db(session)`:

- `ETF_FUNDS` — ETFs (custodied at BNY / CCVA)
- `CLOSED_END_FUNDS` — CEFs (custodied at UMB)
- `PRIVATE_FUNDS` — private funds
- `VIT_AND_MUTUAL_FUNDS` — VIT trust + mutual funds (custodied at BNY VIT)

Pass cohorts (or explicit fund tickers) to `build_run(...)` to scope a run.

---

## Architecture

```
processing.api
        │ build_run(mode, cohorts/funds, dates, …) → RunRequest   (or build one directly, e.g. from a UI)
        │ run(request) → RunResult  (per-day loop, or one stacked range run)
        ▼
processing.run_modes  (run_eod_mode / run_trading_mode / run_eod_range_mode)
        ▼
processing.bulk_data_loader.BulkDataLoader
        │ one pass against reconciliation.* SQL views per date
        ▼
processing.fund.Fund  ←── FundSnapshot (current, previous)
                              └── FundHoldings (vest, custodian, index)
        ▼
processing.fund_manager.FundManager
        ├── services.compliance_checker.CompliancePolicyChecker
        ├── services.reconciliator.Reconciliator
        └── services.nav_reconciliator.NAVReconciliator
        ▼
reporting.*  (Excel + PDF generation per asset class + combined PDF)
        ▼
notifications.email_manager.EmailManager (Outlook draft / SMTP send)
```

### Core code worth knowing

- **`processing/fund.py`** — domain layer. `Fund` (per-fund policy + helpers),
  `FundSnapshot` (a fund's full state on a given date), and `FundHoldings`
  (equity/options/flex options/treasury frames per data source: vest,
  custodian, index). Every downstream service operates on these objects.
- **`processing/bulk_data_loader.py`** — data layer. Pulls everything needed
  for a given target / previous date pair in one batched query pass, then
  fans the results out onto each `Fund`'s snapshots. No other code touches
  the database.
- **`processing/fund_manager.py`** — executes the requested operations
  (compliance / reconciliation / nav_reconciliation) per fund and packs the
  outputs into `ProcessingResults`.
- **`processing/api.py`** — `RunRequest` / `RunResult`, `build_run(...)`
  presets, `validate(...)`, `run(...)`, `bundle(...)`, and `catalog()` /
  `rule_catalog()` (cohorts, funds, operations and compliance rules a UI can
  offer). **This is the single entry point** for kicking off any run
  (script, scheduler, or UI).
- **`processing/worker.py`** — runs one JSON request in its own process and
  streams progress as JSON lines (what a UI launches).
- **`config/run_configurations.py`** — `MODES` presets, `COHORTS`, rule suites.
- **`processing/run_modes.py`** — orchestration. `run_eod_mode`,
  `run_trading_mode`, `run_eod_range_mode` — each loads data, runs
  operations, builds reports, and fires email notifications.

### SQL surface

All data lives in the `reconciliation.*` schema (a set of views). Custodian
fund-ticker resolution flows through `accounts_mapping.vw_tif_account_numbers`.
The Python code never queries operational base tables directly; if a new
field is needed, it's added to the appropriate view.

---

## Outputs

Reports land on the shared drive, partitioned by mode:

```
G:\Shared drives\Portfolio Management\Funds\Archive\Daily_Compliance\2026\
    ├── Compliance\
    │     compliance_results_<tag>_<date>.{xlsx,pdf}
    ├── Holdings and NAV Recon\
    │     reconciliation_summary_<tag>_<date>.{xlsx,pdf}
    └── Trading Compliance\
          trading_analysis_<tag>_<date>.{xlsx,pdf}
          trading_compliance_results_combined_<tag>_<date>.{xlsx,pdf}
```

The base directory is configured in
`config/run_configurations.DEFAULT_OUTPUT_DIR`; mode subfolders are created on
first use.

---

## Email notifications

`notifications/email_manager.py` produces an email per pipeline run:

- **EOD** → NAV recon summary, holdings recon summary, and G/L components
  rendered as HTML tables in the body; PDF + Excel attached.
- **Trading analysis** → compliance status changes and trade activity summary
  in the body (mirrors the PDF); PDF + Excel attached.
- **Compliance** → attachments only, no body content.

Two delivery modes, toggled by env var `FUNDMANAGER_EMAIL_MODE`:

- `display` (default) — opens the message as an Outlook draft for review
  before sending. Requires `pywin32` on Windows.
- `send` — sends straight through SMTP. Credentials read from env vars
  `FUNDMANAGER_SMTP_HOST/PORT/USER/PASSWORD/FROM_ADDRESS`.

Suppress per-run with `build_run(..., send_email=False)`.

---

## Configuring a run

Everything goes through `build_run(...)`:

```python
from processing.api import build_run, execute_run
from config.fund_definitions import ETF_FUNDS

# EOD for the ETF cohort
cfg = build_run(
    "eod",
    cohorts=[ETF_FUNDS],
    start_date="2026-05-22",
    output_tag="etfs",
)

# Trading analysis on two funds. Note: send_email is currently ignored for trading;
# the trading-analysis email is always sent (the trading path has no switch yet).
cfg = build_run(
    "trading_compliance",
    funds=["P2726", "RDVI"],
    start_date="2026-05-22",
    output_tag="may22",
    send_email=False,
)

# Time-series stacked compliance across a date range
cfg = build_run(
    "time_series",
    cohorts=[ETF_FUNDS],
    start_date="2026-05-01",
    end_date="2026-05-15",
    output_tag="may_etfs",
)
```

`cfg` is then handed to `execute_run(cfg)`. From the command line:
`python main.py eod --cohort ETF_FUNDS --start 2026-05-22 --tag etfs`. A UI can
build `processing.api.RunRequest(...)` directly — e.g. any subset of
`operations`, `compliance_tests=("diversification_40act_check",)` for a single
rule, `cohorts=("ETF_FUNDS",)`, and `date_mode="per_day"` or `"stacked"` over
`start_date`..`end_date` — and call `run(request)`. See
`docs/build_run_examples.txt` for a fuller catalogue of invocation patterns.

### Validation, progress and report files

- `run()` first calls `validate(request)`, which raises `InvalidRequest` (a
  `ValueError`) listing the valid choices for an unknown operation,
  `date_mode`, compliance test (`api.COMPLIANCE_TEST_KEYS`: the checker's
  `RULE_KEYS` plus the report-only `"summary"` sheet; `"all"` means every
  rule), cohort or fund ticker; for `end_date < start_date`; for trading
  mixed with EOD operations or run stacked; and for a fund selection that
  resolves to nothing. Tickers combined with cohorts or `exclude` must be
  exact upper-case tickers, one per entry. An `exclude` with no
  cohorts/funds is applied to all funds. Cohort selections are checked once
  `run()` has loaded the cohorts from the DB. `python main.py --test`
  accepts only those keys (or `all`).
- `run(request, on_progress=callback)` calls `callback(ProgressEvent(stage,
  day, fund, index, total, message))` per day, per phase (`db`, `load`,
  `process`, `reports`, `email`, `done`), per fund (`index`/`total`) and per
  operation. A failing callback is logged at debug level and never breaks
  the run.
- `RunResult.files` lists each written report as `ReportFile(day, label,
  path, kind)` (only files that exist; `paths` keeps the raw labels), and
  `RunResult.errors` collects the failures that are otherwise only logged
  (including a failed notification email, which leaves the exit code 0).
  `bundle(result, "reports.zip")` zips each distinct file once.
- `python -m processing.worker request.json [--result result.json]` runs one
  request (`request_to_json` / `request_from_json` format) in a fresh
  process, prints each `ProgressEvent` as a JSON line on stdout (logs go to
  stderr) and writes the exit code, summaries, paths, files and errors to
  `result.json` (default `<request>.result.json`). Exit code 2 means the
  request was invalid (`InvalidRequest`).
- Tests: `python -m unittest discover tests -v` (no DB access).

---

## Web UI

A Streamlit page for ad-hoc and back-dated runs (the scheduled `python main.py …`
runs are unchanged). Design notes: `docs/ui_plan.md`.

```bash
python -m streamlit run ui/app.py                                   # http://localhost:8501
# behind the SSO proxy (FUNDMANAGER_UI_USER_HEADER set): listen on loopback, the proxy forwards to it
python -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
# shared-password mode only (no user header): reachable on the internal network
python -m streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

- **New run:** fund groups (cohorts from the DB, cached; *Reload fund groups* in the
  sidebar re-reads them), funds and exclusions with a live fund count; End of day
  (Compliance / Holdings recon / NAV recon) or Trading compliance; all rules or any
  subset; one as-of date or a range, per day or one stacked compliance workbook;
  PDF, email, output tag (blank = the CLI's tag for that run type, e.g. `eod`,
  `compliance`, `time_series`, so the report file names match the CLI's). Weekend
  dates snap to business days (an as-of Saturday runs Friday, a range runs its
  first to last business day) and the caption lists the days that will run; there
  is no holiday calendar. The request is checked with `processing.api.validate`
  before **Run** is enabled. A stacked run with recon operations needs *Also write
  daily reports* (the stacked workbook holds compliance only). Picks are kept while
  you visit other pages. While a run is active the form is locked and a panel
  shows the progress bar (fund *index/total*), the stage, the log tail and **Cancel**;
  a refreshed or second browser reattaches to it. Afterwards: one download per
  report, **Download all (zip)**, the summary per day and any errors.
- **Run history:** every past run (who, when, what, status, files) with its results
  and **Re-run** (loads the request back into the form, email off).
- **Reports:** finished runs' reports by report date (each day a run wrote reports
  for; a stacked workbook is listed under its range).

Each run is one `python -m processing.worker` process (cwd = the run folder,
`PYTHONPATH` = the repo) in `<FUNDMANAGER_UI_OUTPUT_ROOT>/<YYYYMMDD-HHMMSS>-<user>-<tag>/`,
which holds `request.json`, `progress.jsonl` (the worker's JSON-lines stdout),
`run.log` (its stderr), `result.json`, `meta.json` (user, start/finish, request
summary, exit code, cancelled / timed-out flags, publish history), `logs/` and
`reports/`. Only one run happens per output root at a time (`.run.lock` holds the
worker PID and its process creation time; a lock whose process is gone, or whose PID
now belongs to another process, is cleared, and Cancel never kills a process whose
creation time does not match). A run keeps going if Streamlit restarts; the page
reattaches, enforces the timeout and records the result when the worker ends.
Keep the output root short (e.g. `C:\FundManager\ui_runs`; the sidebar warns when it
is too long): report paths over 260 characters are handled with the `\\?\` path
form, but Explorer and Excel may not open them from the run folder.

| Var | Default | Purpose |
|---|---|---|
| `FUNDMANAGER_UI_OUTPUT_ROOT` | `<repo>/ui_runs` (gitignored) | where run folders go |
| `FUNDMANAGER_UI_PUBLISH_ROOT` | unset | target of the **Publish** button (e.g. the G: archive); unset = no publishing |
| `FUNDMANAGER_UI_TIMEOUT_MINUTES` | `120` | a run is killed after this; `0` = never |
| `FUNDMANAGER_UI_USER_HEADER` | unset | request header with the username set by the SSO proxy; when set, requests without it are refused |
| `FUNDMANAGER_UI_PASSWORD` | unset | optional shared password (also read from `.streamlit/secrets.toml`) |
| `FUNDMANAGER_UI_CATALOG_TTL` | `600` | seconds the fund groups stay cached |

They can live in `.env` with the DB settings (see `.env.example`). Without
`FUNDMANAGER_UI_USER_HEADER` the username is the OS account running Streamlit.

Safety:
- A UI run never writes to the shared drive: the runner always replaces the
  request's `output_dir` with the run folder. **Publish** copies a finished run's
  `reports/` into `FUNDMANAGER_UI_PUBLISH_ROOT` (same sub-folders, existing files
  skipped, recorded in `meta.json`) only after a confirmation tick.
- **Send email** is off by default for End-of-day runs (also on Re-run) and needs a
  second confirmation tick; both are cleared once a run starts, so every emailing
  run is confirmed again. With `FUNDMANAGER_EMAIL_MODE=display` (the default)
  emails open as Outlook drafts on the host running the UI.
- **Trading compliance always sends** (or drafts) the trading-analysis email: the
  backend's trading path has no `send_email` switch yet (`processing/run_modes.py`
  `run_trading_mode`). The UI therefore shows *Send email* ticked and locked for
  trading, warns, and requires the confirmation tick before **Run**.
- Host it on the internal network only, behind the SSO proxy or with the shared
  password. With `FUNDMANAGER_UI_USER_HEADER` set the app trusts that header as
  the user's identity, so Streamlit must only be reachable through the proxy
  (`--server.address 127.0.0.1`, or a firewall rule) and the proxy must overwrite
  any copy of the header sent by the client; otherwise anyone can claim any name.
- The DB connection string stays in `.env`. A catalog failure is shown by
  exception type only; worker errors and run-log tails are shown with
  credentials masked (URL passwords, the `DB_CONNECTION_STRING` and
  `FUNDMANAGER_SMTP_PASSWORD` values, MySQL user names) but are otherwise
  verbatim, so they can still name the DB host.
- Streamlit installs pyarrow, and recent pyarrow requires NumPy 2 while the
  project pins NumPy 1.26, so the UI avoids `st.dataframe` and renders tables as
  Markdown.

Tests: `python -m unittest discover tests -v` also runs `tests/test_ui_runner.py`,
`tests/test_ui_history.py` and `tests/test_ui_app.py` (Streamlit `AppTest`, a fake
worker, no DB).

---

## Reconciliation outputs at a glance

- **Compliance** — per-fund pass/fail across the rule set + per-rule detail
  tables. Rule logic lives in
  `services/compliance_checker.CompliancePolicyChecker`; reports in
  `reporting/compliance_reporting.py`.
- **Holdings recon** — break counts by asset class (equity, options, flex
  options, treasury, index), broken into "holdings breaks" (red cells) and
  "price breaks" (orange cells); per-fund discrepancy tables.
- **NAV recon** — beginning TNA → expected TNA → custodian TNA chain;
  per-asset-class G/L; NAV Good (2-dec / 4-dec / Cust Opt) PASS/FAIL strip;
  **Option Price Sensitivity matrix** (Vest vs Custodian option pricing —
  listed, flex, combined; expected NAV under each).
- **Trading analysis** — Compliance Status Changes (changes between ex-ante
  and ex-post + funds still out of compliance); Trade Activity Summary
  (per fund per asset class: trade value, % of TNA, % of total assets, MV
  delta); per-fund Trade Activity Details sub-tables.

---

## Setup

```bash
git clone https://github.com/pcorc/FundManager.git
cd FundManager

python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # macOS / Linux

pip install -r requirements.txt
pip install pywin32              # Windows only, for Outlook draft mode
```

Environment variables that affect runtime behaviour:

| Var | Default | Purpose |
|---|---|---|
| `FUNDMANAGER_EMAIL_MODE` | `display` | `display` opens an Outlook draft; `send` uses SMTP |
| `FUNDMANAGER_SMTP_HOST` | `smtp.office365.com` | SMTP server |
| `FUNDMANAGER_SMTP_PORT` | `587` | SMTP port |
| `FUNDMANAGER_SMTP_USER` | — | SMTP login |
| `FUNDMANAGER_SMTP_PASSWORD` | — | SMTP password |
| `FUNDMANAGER_FROM_ADDRESS` | SMTP user | Email From address |

DB connection settings are configured in `config/db_config.py` — point at the
`reconciliation` schema.
