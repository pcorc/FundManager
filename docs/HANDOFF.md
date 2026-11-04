# FundManager handoff: context and tasks for your AI assistant

Paste everything below into your AI coding assistant once you have cloned the repository.

---

You are taking over **FundManager** (https://github.com/pcorc/FundManager). It is Vest Financial's Python 3.11 system for daily fund compliance and reconciliation. Read this whole brief before you change anything. Then read `README_FundManager_20260611.md`, `docs/ui_plan.md` and `docs/build_run_examples.txt`.

## 1. What the system does

FundManager loads holdings, trades, NAV and index data from the `reconciliation` MySQL schema (`reconciliation.v_*` views). It builds one Fund object per fund and runs four operations:

| Operation | What it produces |
|---|---|
| **Compliance** | 40 Act diversification, IRS (2a 50%, 2a 10%, 2b) / IRC (VIT funds), prospectus 80%, GICS concentration, 12d1/12d2/12d3, real estate, illiquid, summary metrics. Excel and PDF. |
| **Holdings recon** | OMS vs custodian holdings breaks (equity, option, treasury). Excel and PDF. |
| **NAV recon** | Expected vs custodian NAV/TNA, gain/loss attribution and option-price sensitivity. Excel, plus a combined EOD PDF. |
| **Trading compliance** | Ex-ante (pre-trade) vs ex-post (post-trade) compliance, plus a trade activity summary. Excel, PDF and email. |

**Modes** are set in `config/run_configurations.py`: `eod` (compliance + holdings recon + NAV recon), `compliance`, `reconciliation`, `nav`, `trading_compliance`, `time_series` (a stacked compliance workbook over a date range).

**Fund groups (cohorts)** are loaded from `reconciliation.v_fund_metadata` at startup: `ALL_FUNDS`, `ETF_FUNDS`, `CLOSED_END_FUNDS`, `PRIVATE_FUNDS`, `VIT_AND_MUTUAL_FUNDS`.

### Code map

| Area | Files |
|---|---|
| Entry points | `main.py` (CLI); `processing/api.py`: `build_run`, `validate`, `run(request, on_progress)`, `catalog()`, `rule_catalog()`, `bundle()`; `processing/worker.py`: one run per process, JSON-lines progress on stdout, writes `result.json` and `views.json` |
| Data and processing | `processing/bulk_data_loader.py`, `processing/fund.py`, `processing/fund_manager.py`, `processing/run_modes.py` |
| Rules and recon | `services/compliance_checker.py` (the rule logic; **the Python checker is the source of truth** and Excel formulas follow it), `services/reconciliator.py`, `services/nav_reconciliator.py` |
| Reports | `reporting/*` (Excel and PDF); `notifications/email_manager.py` (Outlook draft or SMTP, set by env) |
| Config | `config/fund_definitions.py` (static policy flags per fund), `config/constants.py` (thresholds), `config/run_configurations.py` (presets, default output folder on the G: drive), `config/database.py` (reads `DB_CONNECTION_STRING` from `.env` and connects lazily) |
| Web UI (Streamlit) | `ui/` (`python -m streamlit run ui/app.py`). Pages: **New run** (end of day; settings on the left: date range, fund groups/funds, checks with Select All, Run/Reset; configuration preview and on-page results on the right), **Trading Compliance** (`ui/trading_page.py`: the proposed trades of a trade date, ex-ante vs ex-post; a staged-trades quick pick; afterwards a dashboard from `views.json` `dashboard` (`reporting/trading_dashboard.py`): KPIs, a funds × rules before → after matrix, rules at risk, a summary table with current and estimated post-trade cash, SVG charts (`ui/charts.py`, rendered through `st.markdown` because `st.html` strips SVG) and a fund drill-down), Run history and Reports. Also downloads, a zip bundle, cancel, and a lock so only one run happens at a time. Each run runs `processing.worker` in `<FUNDMANAGER_UI_OUTPUT_ROOT>/<YYYYMMDD-HHMMSS>-<user>-<tag>/`. |
| Tests | `tests/` (stdlib `unittest`, about 200 tests): `python -m unittest discover tests` |
| Golden-master harness | `tools/golden/golden.py` with `check`, `capture`, `snapshot`, `rebaseline`, `report`. It replays recorded DB data through 19 scenarios and diffs every Excel cell, PDF text and log line against the baselines. The recorded data and baselines live in the **gitignored** `tools/golden/_work/`, so you do not have them. Run `golden.py capture` (read-only against the DB) to create your own before making changes. |

### Recent history

All of this is on branch `refactor/condense`, which is merging to `main`.

- About 20.7k lines condensed to about 10.3k with no change to output. This was proven with golden.
- API hardening: validation, credentials moved to `.env`, a progress hook, report file links and `RunResult.errors`.
- 21 output-bug fixes, D1–D21. Each is one commit with a before/after review. Examples: fund-type detection, NAV flows double-count, IRS option duplication, real ownership data for 12d2/12d3/IRS 10%, 40 Act non-diversified-registered funds now PASS, the IRC PDF section, the trading ex-ante snapshot, treasury notional ÷100, and trading honouring `send_email`.
- Streamlit UI, then a full UI redesign: soft dark theme in `.streamlit/config.toml` plus `ui/theme.py`, a configuration preview, and results on the page.

## 2. Setup

1. `git clone https://github.com/pcorc/FundManager.git`, then `git checkout main` (or `refactor/condense` if it is not merged yet).
2. Use **Python 3.11** with a venv and `pip install -r requirements.txt`. On Windows also install `pywin32` for Outlook drafts. **Do not use Anaconda's Python.** Its packages differ: fpdf imports fail, and with pyarrow 26 and numpy 1.26 `st.dataframe` crashes. That crash is why the UI renders its tables as HTML/Markdown.
3. Copy `.env.example` to `.env` and fill in `DB_CONNECTION_STRING`. Get the value from Patrick through a secure channel. `.env` is gitignored. Never commit, print or log it.
4. Smoke test: `python main.py eod --cohort ETF_FUNDS --start <recent business day> --no-email --tag smoke`. Point the output at a local temp folder, not the G: drive. The default output folder is in `config/run_configurations.py`; override it or set `FUNDMANAGER_UI_OUTPUT_ROOT` for UI runs.

## 3. Guardrails (always)

- No credentials in code, logs, commits or chat. That includes the DB string, SMTP, and Slack/Zapier/xMatters webhooks. Everything goes in `.env` or the scheduler's secret store.
- **While testing, send no emails and write nothing to the G: drive.** Use `--no-email` or `send_email=False`, and a local output folder.
- Work on feature branches and open PRs. Never force-push `main`.
- Before and after any change that could affect output, run `python tools/golden/golden.py check` and `python -m unittest discover tests`. If a change is meant to alter output, run `golden.py report` to write the before/after diff for review, then `golden.py rebaseline <scenario>` after approval.
- Keep CRLF line endings in `config/database.py`, `config/operation_config.py`, `services/nav_reconciliator.py` and `services/reconciliator.py`. Keep logger names unchanged.
- Recorded holdings data stays out of git, in `tools/golden/_work/`.

## 4. Known open items (decide or confirm with compliance and ops)

1. **Real-estate limit.** It is now zero-tolerance on real ownership data, and about 20 funds FAIL at 0.24–3.5% exposure. Confirm the intended limit.
2. **12d3 thresholds.** The code uses 25%/10%; the statute says 10%/5%. Confirm.
3. **NAV residual gaps** after the D3 flows fix. Some funds still show residual expected-vs-custodian differences.
4. **Ownership spot-check.** Compare shares-outstanding and ownership % against an independent source for a few funds.
5. **FTMIX.** Check its registration and closed-end classification.
6. **FDND GICS PASS** relies on duplicate rows in the DJINET index data. That is a data-quality issue upstream.
7. **CEF top-5 option breaks.** Confirm they are real.
8. **No holiday calendar.** Weekends snap to business days, but market holidays do not.
9. **The DB password is still in git history.** It must be rotated. Scrubbing history would need a force-push; that is the owner's decision.
10. The README says DB settings are in `config/db_config.py`. That is out of date: they are in `.env` / `config/database.py`. Fix it.

## 5. Your tasks, in order

### Task 1: Test every fund group × every operation

For each cohort (`ETF_FUNDS`, `CLOSED_END_FUNDS`, `PRIVATE_FUNDS`, `VIT_AND_MUTUAL_FUNDS`, and `ALL_FUNDS` once as a whole), run each of these on one recent business day:

| Operation | Command |
|---|---|
| Holdings recon | `python main.py reconciliation --cohort <C> --start <D> --no-email --tag t1` |
| NAV recon | `python main.py nav --cohort <C> --start <D> --no-email --tag t1` |
| Compliance | `python main.py compliance --cohort <C> --start <D> --no-email --tag t1`, plus one run per single rule with `--test <rule>` |
| Trading compliance | `python main.py trading_compliance --cohort <C> --start <D> --no-email --tag t1`. Trading cannot be combined with EOD operations and cannot run stacked; it is per day only. |
| EOD combined | `python main.py eod --cohort <C> --start <D> --no-email` |
| Date range | `python main.py time_series --cohort <C> --start <D1> --end <D2> --no-email` |

Run `python main.py --help` for the exact options. Repeat a sample of these through the Streamlit UI and check that the on-page results match the Excel and PDF files.

For every run, record in `docs/test_matrix.md` (and open a PR):

- the exit code
- the files written
- `RunResult.errors`
- the PASS/FAIL counts
- anything odd, such as empty sheets, N/A where data should exist, absurd percentages, or missing funds

Expected behaviour:

- Private funds show "N/A (skipped: …)" for 40 Act/IRS.
- VIT funds get the IRC section.
- CEFs use the CEF prospectus variant.

Open an issue for each real defect. Fix defects in small PRs, with golden and the tests run each time.

### Task 2: Implement the single-stock product suite

There is no spec in the repo yet. **Ask your user for the requirements before you write code.** You need:

- the fund tickers and their vehicle (ETF, mutual fund, etc.)
- whether they are leveraged, inverse or covered-call single-stock products
- the custodian and data feeds, and whether `v_fund_metadata` and holdings already include them
- which compliance rules apply. For example: 40 Act non-diversified, IRS diversification for a fund concentrated in one issuer, swap/derivative exposure limits, Rule 18f-4 VaR, and prospectus policy.
- recon specifics, such as swaps or total-return swaps with counterparties
- reporting and email recipients

Note: `config/fund_definitions.py` already uses `"single_stock"` as a `listed_option_type`/`flex_option_type` value, meaning options on a single stock rather than an index. That is not a product suite, so don't confuse the two.

Expected work:

1. Add a cohort, e.g. `SINGLE_STOCK_FUNDS`, in `config/fund_definitions.py` and `config/run_configurations.py`, sourced from `v_fund_metadata`.
2. Add per-fund policy flags.
3. Add any new instrument handling (e.g. swaps) in the loader, Fund, reconciliator and NAV recon.
4. Add new rules to `services/compliance_checker.py` and its rule catalog, with their Excel/PDF sections.
5. Make the UI pick it all up via `catalog()`/`rule_catalog()`.
6. Add unit tests and a golden scenario.

### Task 3: Scheduled nightly run, a Slack NAV check, and email links to the UI

**a. Schedule.** Run `eod` for all cohorts nightly after the close (about 6:00 PM ET) on the always-on internal host that also serves the UI. Use Windows Task Scheduler or the company's scheduler.

- Use the worker (`python -m processing.worker request.json --result result.json`), or a small wrapper script, so each run writes the same run-folder layout the UI uses: `request.json`, `result.json`, `views.json`, `run.log`, `meta.json`, `reports/` under `FUNDMANAGER_UI_OUTPUT_ROOT`. Scheduled runs then appear in the UI's Run history.
- Retry once on failure.
- Alert on non-zero exit codes or on `errors` in `result.json`.

**b. High-level Slack NAV check.** After the run, build a short summary from `result.json`/`views.json`:

- run date and status
- number of funds checked
- NAV breaks (count, and the worst funds by NAV diff in cents/bps)
- holdings break count
- compliance FAIL count
- the link to the run in the UI

POST it as JSON to a **Zapier Catch Hook** or an **xMatters** inbound integration (HTTP trigger). Have that tool post to the Slack channel. xMatters can also page on-call when NAV breaks exceed a threshold.

- Keep the webhook URL in an env var such as `FUNDMANAGER_NAV_WEBHOOK_URL`. Never put it in git.
- Add a `--no-notify` flag for testing.
- Write unit tests with a mocked HTTP call.
- Do not post to the real channel during development. Use a test channel or a test Zap.

**c. Hyperlink in the nightly email that opens the breaks in the UI.** The UI has **no deep-link route yet**. Add one:

- Read `st.query_params` in `ui/app.py`, e.g. `?run=<run_folder_id>&view=breaks&fund=<ticker>`. Open that run's results page filtered to breaks only: holdings breaks, NAV breaks and compliance FAILs. Use the existing `views.json` and `ui/results.py`.
- Validate the `run` id strictly against existing run folders, to prevent path traversal.
- Respect the auth gate: SSO header or shared password.

Then add `FUNDMANAGER_UI_BASE_URL` (the internal host URL) and have `notifications/email_manager.py` put a "View breaks in FundManager" link (`<base>/?run=<id>&view=breaks`) in the EOD email body, plus a per-fund link on each break row. Use the same link in the Slack message.

Test with `FUNDMANAGER_EMAIL_MODE=display`, which opens an Outlook draft and sends nothing.

### Task 4: Keep it healthy

- After each change: run golden and the tests, and update the README and `docs/ui_plan.md`.
- Consider a GitHub Actions workflow that runs the unit tests on every PR. Golden needs DB recordings, so keep it local.
