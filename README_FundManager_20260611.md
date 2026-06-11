# VestFundManager

Python platform that runs Vest's daily fund operations: compliance checks,
holdings reconciliation, NAV reconciliation, and intraday trading compliance.
One pass against the database loads everything; downstream services read from
in-memory snapshots.

---

## What runs

Five operational **modes**, all built on the same loader + fund-object
foundation. Each mode is parameterised through `config/run_configurations.py`
and dispatched by `processing/run_modes.py`.

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
config.run_configurations
        │ build_run(mode, cohorts/funds, dates, …) → cfg
        ▼
main.py
        │ cfg → params (via utilities/cli_options.apply_overrides + resolve_*)
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
- **`config/run_configurations.py`** — `MODES` dict + `build_run(...)`
  factory. **This is the single entry point** for kicking off any run
  (script, scheduler, or UI).
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
from config.run_configurations import build_run
from config.fund_definitions import ETF_FUNDS

# EOD for the ETF cohort
cfg = build_run(
    "eod",
    cohorts=[ETF_FUNDS],
    start_date="2026-05-22",
    output_tag="etfs",
)

# Trading analysis on two funds, email suppressed
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

`cfg` is then handed to `main.execute_run(cfg)`. See
`docs/build_run_examples.txt` for a fuller catalogue of invocation patterns.

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
