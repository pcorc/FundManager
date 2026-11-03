# FundManager UI plan (Streamlit)

## Goal
Give ops and compliance a web page to:
- pick funds, by fund group (cohort) or individually
- pick what to run: compliance, trading compliance, holdings recon, NAV recon, or one or more individual compliance rules
- pick an as-of date or a date range
- download the reports

The UI holds no business logic. Everything goes through `processing.api`.

## Architecture
```
Browser ──> Streamlit app (ui/app.py)
              │  catalog(), rule_catalog(), validate(), request_to_json()
              │
              └─ subprocess: python -m processing.worker run.json --result result.json
                    │  stdout: one JSON ProgressEvent per line
                    └─ writes reports to <output_root>/<run_id>/ and result.json
```
- **One run per worker process.** Cohort sets, Fund objects and logging are process-global, so a run must never share a process with the UI or with another run.
- **Inputs are validated before launch.** The app calls `validate(request)` in-process, so typos and empty selections show up next to the form instead of as a failed run.
- **The app streams progress.** It reads the worker's stdout line by line. A progress bar is driven by `index`/`total` of the `fund` events, and a step indicator by `stage` (db → load → process → reports → email → done). A log pane tails the run log.
- **Results come from `result.json`.** The app reads `files` (only files that exist), `errors` and `summaries`. It shows one download button per file, a "Download all (zip)" button (`bundle()`), and a summary table per day.

## Screens
1. **New run**
   - **Fund groups:** multiselect over `catalog()["cohorts"]`. A fund-count badge updates as you pick.
   - **Funds:** multiselect, narrowed by the chosen groups. With no groups chosen it lists every fund.
   - **Exclude:** multiselect.
   - **Operations:** checkboxes for Compliance, Holdings recon and NAV recon, or a radio for Trading compliance. Trading can't be mixed with the others; `validate` enforces this too.
   - **Compliance rules:** "All rules" (default), or a multiselect over `rule_catalog()` labels, so you can run any single rule.
   - **Dates:** "As of" with one date picker, or "Range" with start/end pickers and a choice of *One report per day* or *One stacked workbook*. Stacked applies to compliance only.
   - **Options:** create PDF (on), send email (**off** by default, with a confirmation prompt when switched on), output tag.
   - **Run** button. While a run is active, the form is disabled and the progress panel shows.
2. **Run history**
   - A table of past runs read from each `<run_id>/result.json`: who, when, request summary, exit code and the files.
   - **Re-run:** reloads that request into the form.
3. **Reports browser** (optional): browse `<output_root>` by date.

## Files
- `ui/app.py`: the page layout and wiring, about 250 lines.
- `ui/runner.py`: starts the worker, streams the JSON lines, holds a lock file per output root so only one run happens at a time, and handles timeouts and cancel (terminating the worker).
- `ui/history.py`: reads and writes `result.json`.
- `requirements.txt`: adds `streamlit` (pinned) to the current pins: SQLAlchemy 1.4.46, PyMySQL 1.1.0, pandas 2.1.4, openpyxl 3.1.2, fpdf2 2.8.7, python-dotenv 1.0.1.

## Configuration and deployment
- **Secrets:** the DB connection string and SMTP settings come from `.env` or the host environment (see `.env.example`). Nothing secret goes in the UI code or `st.secrets`, apart from the optional UI password.
- **Output:** `FUNDMANAGER_UI_OUTPUT_ROOT`, defaulting to a local folder. Each run writes to `<root>/<YYYYMMDD-HHMMSS>-<user>-<tag>/`. Publishing to the shared `G:` drive is a separate "Publish" button that copies the files, so test runs never land there.
- **Hosting:** one internal Windows host or VM running `streamlit run ui/app.py --server.address 0.0.0.0`, reachable only on the internal network or VPN. Access control is one of:
  - (a) behind the company SSO reverse proxy (preferred), or
  - (b) a simple shared password via `st.secrets`.

  The UI logs the username into each run's `result.json` for the audit trail.
- **Scheduling:** the existing scheduled CLI runs (`python main.py …`) are unchanged. The UI is for ad-hoc and back-dated runs.

## Build order
1. `ui/runner.py`, plus a test that runs a fake worker.
2. The New run screen with validation and progress.
3. Results, downloads and the zip bundle.
4. Run history and re-run.
5. Lock file, cancel, the email confirmation prompt, and the Publish button.
6. Deployment notes and a README section.

## Out of scope for v1
Multi-user concurrent runs, report previews in the browser (beyond PDF links), and editing fund definitions.
