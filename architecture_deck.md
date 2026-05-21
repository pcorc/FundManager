---
marp: true
theme: default
paginate: true
size: 16:9
---

# VestFundManager
## Daily Compliance, Holdings & NAV Reconciliation Architecture

A SQL-first, view-driven pipeline for ~24 funds across BNY US, BNY VIT,
UMB, CCVA, and SocGen custody.

For intranet integration: configure per-cohort, per-custodian, per-timing run
plans and let the views handle the rest.

---

# What this system does

Every business day, for each fund in scope:

1. **Compliance** — 40 Act diversification, IRS 851/817(h), GICS overlap,
   12d-1/2/3 sub-rules, prospectus 80% policy, illiquidity caps
2. **Holdings reconciliation** — OMS positions vs. each custodian's holdings
3. **NAV reconciliation** — Expected TNA from G/L roll-forward vs. custodian TNA
4. **Trade compliance** — ex-ante (intent) vs. ex-post (filled) policy checks
5. **Creation / redemption tracking** — ETF baskets and AP flows

All driven from one bulk data load and one fund registry.

---

# Architecture in one diagram

```
       Custodian / OMS / EMSX / Bloomberg / Index source tables
                              |
                              v
       +--------------------------------------------------+
       |  accounts_mapping.vw_tif_account_numbers          |
       |  (fund_ticker <-> custody/clearing accounts)      |
       +--------------------------------------------------+
                              |
                              v
       +--------------------------------------------------+
       |  reconciliation.v_*  (Phase 4 / 5 SQL views)      |
       |  Standardized columns, fund_ticker on every row   |
       +--------------------------------------------------+
                              |
                              v
       BulkDataLoader  ->  one query per view, one fanout
                              |
                              v
       Fund objects + BulkDataStore (in-memory, per-run)
                              |
                              v
       services/* (compliance / recon / NAV / trade)
                              |
                              v
       reporting/* (PDF + Excel deliverables)
```

---

# Why SQL views (not Python)

**Before**: 940-line `bulk_data_loader.py` with one query method per custodian
× per asset class × per analysis type. Hundreds of column-name fallbacks
(`shares_cust` / `cust_shares` / `custodian_shares` / ...).

**After**: ~200-line loader that:
- iterates a `VIEWS` tuple of `(view_name, payload_key, key_t1, ...)`
- runs one query per view with `WHERE date IN :dates AND fund_ticker IN :funds`
- slices client-side by `fund_ticker`

**Wins**:
- Adding a new custodian = one view edit, zero Python changes
- Column normalization happens once, in SQL
- DBAs can validate data with the same queries the loader runs
- Per-fund slicing is `df[df.fund_ticker == name]`, never per-table key guessing

---

# Canonical key: `fund_ticker`

Every view exposes `fund_ticker` as the canonical fund identifier.

`accounts_mapping.vw_tif_account_numbers` is the resolver. It pivots the
flat `account_numbers` table and exposes:

| Column | Meaning |
| --- | --- |
| `Fund` | canonical fund_ticker (e.g. `DOGG`, `KNG`, `HE3B1`) |
| `Custody_Account` | numeric custodian account (BNY) or fund ticker (UMB/VIT/CCVA) |
| `Custody_Service_Provider` | BNY / UMB / CCVA / etc. |
| `Clearing_Account` | SG account number for assignments / SG holdings |
| `Block_Fund_Identifier` | groups CEFs into block accounts (BK608, BK659) |
| `BBG_Ticker` | exchange-listed ticker for Bloomberg joins |
| `Benchmark`, `Expense_Ratio`, `Launch_Date`, etc. | metadata |

Filter: `WHERE eod_report_strategy IN ('CEF', 'PF', 'TIF', 'VIT')`
(four strategies = the 24-fund TIF universe).

---

# View catalog — Phase 4 (custodian + index + recon)

| View | What it does |
| --- | --- |
| `v_custodian_equity` | BNY/UMB/CCVA equity holdings, GICS-enriched via SEDOL or TICKER |
| `v_custodian_option` | BNY/UMB/CCVA options |
| `v_custodian_treasury` | BNY/UMB treasury holdings |
| `v_custodian_nav` | BNY US + BNY VIT + UMB + CCVA NAV components |
| `v_custodian_cash` | Latest cash balance per custodian |
| `v_index_holdings` | CBOE / NASDAQ / S&P / DOGG index constituents |
| `v_recon_custodian_equity` | OMS vs. custodian holdings recon (FULL-OUTER emulated) |
| `v_recon_custodian_option` | same shape, option asset class |
| `v_recon_custodian_treasury` | same shape, treasury |
| `v_recon_index_equity` | OMS vs. index holdings recon |
| `v_recon_bbg_equity` | Per-row Bloomberg resolution status (Resolved/Unresolved) |

---

# View catalog — Phase 5 (vest OMS + trade + lifecycle)

| View | What it does |
| --- | --- |
| `v_fund_metadata` | Strategy / vehicle_wrapper / expense_ratio / benchmark per fund |
| `v_vest_equity` | OMS equity + GICS join + EQY_SH_OUT + computed market value |
| `v_vest_option` | OMS options + GICS on underlying + notional / market / delta-adj values |
| `v_vest_treasury` | OMS treasury + computed market value |
| `v_emsx_trades` | All FILLED EMSX trades (equity/option/treasury), ticker normalized |
| `v_block_options` | Block option trades (CEF/PF), BRK/B normalized |
| `v_etf_basket` | ETF basket (forward) data, per-fund per-date |
| `v_etf_flows` | Creation / redemption activity per ETF |
| `v_fund_distributions` | Ex-date / payable-date / distro amount per fund |
| `v_option_custodian_assignment` | SG assignment statement, joined via `Clearing_Account` |
| `v_sg_custodian_holdings` | SG cross-check holdings |
| `v_overlap` | Per-CEF benchmark overlap constituents |
| `v_gics_mapping` | Distinct GICS triples for compliance |

---

# Standardization rules (locked into the views)

Every view that surfaces these columns uses **exactly these names**:

| Canonical column | Used by |
| --- | --- |
| `fund_ticker` | every view |
| `bbg_ticker` | every view (BBG join key) |
| `date` | every view (filter key) |
| `eqyticker` | equity holdings + equity recon + index |
| `optticker` | option holdings + option recon |
| `cusip` | treasury holdings + treasury recon |
| `nav_shares` / `iiv_shares` | OMS shares (no `quantity` / `shares` aliases) |
| `shares_vest` / `shares_cust` | recon views |
| `price_vest` / `price_cust` / `price_diff` | recon views |
| `GICS_SECTOR_NAME` / `GICS_INDUSTRY_NAME` / `GICS_INDUSTRY_GROUP_NAME` | uppercase (matches Bloomberg source) |
| `REGULATORY_STRUCTURE` / `SECURITY_TYP` | uppercase |
| `discrepancy_type` | recon views — `Match` / `Quantity Mismatch` / `Missing in OMS` / etc. |

**No fallback chains anywhere in Python.** Add a column to the view, not a
fallback to the code.

---

# Bloomberg resolution — `v_recon_bbg_equity`

BNY US holdings don't carry the equity ticker; they carry SEDOL.
UMB/CCVA carry tickers directly.

Every equity row goes through:

```sql
LEFT JOIN bloomberg_emsx.bbg_equity_flds_blotter
    ON <SEDOL or TICKER, depending on source>
```

`v_recon_bbg_equity` exposes each row with `resolution_status =
'Resolved' | 'Unresolved'`. Unresolved rows have NULL GICS sector, which
breaks 12d-1/2/3 and industry-overlap rules.

Operational query — list unresolved holdings by market value:

```sql
SELECT source, fund_ticker, join_key, security_description,
       quantity, ROUND(market_value, 2) AS mv
FROM reconciliation.v_recon_bbg_equity
WHERE resolution_status = 'Unresolved' AND `date` = :target_date
ORDER BY market_value DESC;
```

Action: add the missing SEDOL/TICKER to `bbg_equity_flds_blotter`.

---

# Analysis type

OMS holdings carry an `analysis_type` column. The loader filters the three
vest views to one type per run:

| analysis_type | Used by | Source |
| --- | --- | --- |
| `eod` | Daily compliance + NAV + holdings recon | End-of-day OMS snapshot |
| `ex_ante` | Trading compliance (intent-time) | Pre-trade OMS |
| `ex_post` | Trading compliance (filled) | Post-trade OMS |

Loader applies this only to `v_vest_equity`, `v_vest_option`,
`v_vest_treasury` via the `analysis_type_filter=True` flag on the `ViewSpec`.
All other views ignore it.

---

# Python: BulkDataLoader

```python
VIEWS = (
    ViewSpec("v_custodian_equity",   "custodian_equity",   "custodian_equity_t1"),
    ViewSpec("v_custodian_option",   "custodian_option",   "custodian_option_t1"),
    ViewSpec("v_custodian_treasury", "custodian_treasury", "custodian_treasury_t1"),
    ViewSpec("v_custodian_nav",      "nav",                "nav_t1"),
    ViewSpec("v_custodian_cash",     "cash",               "cash_t1"),
    ViewSpec("v_recon_bbg_equity",   "bbg_unresolved",     "bbg_unresolved_t1",
             extra_where="resolution_status = 'Unresolved'"),
    ViewSpec("v_vest_equity",        "vest_equity",        "vest_equity_t1",
             analysis_type_filter=True),
    # ... and so on for option, treasury, distributions, assignments, overlap,
    #     trades, block_options, basket, flows, sg_holdings
)
```

Adding a new dataset = one tuple entry. No new fetcher class, no new ORM
join, no SQLAlchemy reflection.

---

# T and T-1 in one query

Custodian and recon views are date-filterable:

```sql
WHERE `date` IN (:target_date, :previous_date)
  AND fund_ticker IN (:funds)
```

The loader runs one query per view, then splits client-side:

```python
current  = df[df.date == target_date]
previous = df[df.date == previous_date]
```

Stored under `key` and `key_t1` (e.g. `custodian_equity` and
`custodian_equity_t1`). Services consume both without re-querying.

Result: ~13 views × 1 query per run, instead of 13 × 2 (T and T-1 separately).

---

# Per-source date conventions (index data)

Index data is the one place the loader still does date routing. Different
providers publish on different cadences:

| Source | T | T-1 |
| --- | --- | --- |
| CBOE | T | T-1 |
| DOGG | T | T-1 |
| NASDAQ | T+1 | T (= "previous T+1") |
| S&P | T+1 | T |

Loader maps:

```python
INDEX_SOURCE_DATES = {
    "cboe": "target", "dogg": "target",
    "nasdaq": "tplus_one", "sp": "tplus_one",
}
```

Pull all four candidate dates in one query, slice per source.

---

# Equity reconciliation

Inputs: `v_vest_equity` (OMS) and `v_custodian_equity` (custodian).

`v_recon_custodian_equity` joins them and emits one row per
`(fund, date, eqyticker)` with:

- `shares_vest`, `shares_cust`, `shares_diff`
- `price_vest`, `price_cust`, `price_diff`
- `discrepancy_type` — one of:
  - `Match`
  - `Quantity Mismatch`
  - `Missing in OMS`
  - `OMS Quantity Zero`
  - `Missing in Custodian`
  - `Custodian Quantity Zero`

FULL OUTER JOIN emulated with `LEFT JOIN ... UNION ALL ... LEFT JOIN` so
ticker rows present only on one side still surface.

---

# Option reconciliation

Same shape as equity recon but keyed on `optticker`.

OMS uses Bloomberg-style optticker (`ABT US 05/15/26 C97.5`); custodian
descriptions match this format directly.

Block option trades for CEF / PF complexes flow through `v_block_options`,
which surfaces `Fund_Ticker` (already resolved per-fund) and normalizes
`BRK/B` → `BRKB`.

Flex options for HE3B*/TR2B* are split client-side by regex against
`fund.flex_option_pattern` once the trades are loaded.

---

# Treasury reconciliation

Inputs: `v_vest_treasury` (OMS) and `v_custodian_treasury` (BNY US + UMB).

Joined on `cusip`. Same `discrepancy_type` taxonomy as equity/option.

Notes:
- BNY US uses `pricing_number` for treasury "ticker" display
- OMS treasury `price` is varchar; views cast to `DECIMAL(18,6)`
- Treasury market value = `nav_shares * price / 100` (par notation)

Renderer suppresses the entire Treasury section when both holdings and
price-break tables are empty (most ETFs / VIT funds).

---

# Index reconciliation

`v_recon_index_equity` matches OMS equity holdings against the
benchmark index constituents:

- `oms_fund` — fund ticker in OMS
- `index_fund` — the index code (SPX, CBOE/SPATI, etc.) the fund tracks
- `weight_index` / `price_index` — index-side values
- `shares_vest` / `price_vest` — fund-side values
- `discrepancy_type` — `Match` / `Missing in OMS` / `Missing in Index`

Per-source date routing happens in the loader (see Index slide).

---

# NAV reconciliation — Expected TNA

For each fund:

```
Expected_TNA(T) = TNA(T-1)
               + sum(equity G/L)
               + sum(option G/L)
               + sum(treasury G/L)
               + cash flow
               + dividends accrued
               - expenses
               + creation / redemption $
```

Compared against `v_custodian_nav.total_net_assets` for the day.

Tolerances are configured per fund. Discrepancies > tolerance get flagged
on the NAV-recon PDF with line-by-line G/L breakdown.

---

# Gain / Loss formula (equity)

For each position present on T or T-1:

```python
qty_t           = nav_shares on T   (0 if absent)
qty_t1          = nav_shares on T-1
price_t_vest    = OMS price on T
price_t1_vest   = OMS price on T-1
price_t_custodian, price_t1_custodian = same with break adjustments

gl_raw      = (price_t_vest      - price_t1_custodian) * qty_t
gl_adjusted = (price_t_custodian - price_t1_custodian) * qty_t
```

`gl_adjusted` is the one used in Expected_TNA — custodian-side prices on
both legs so the NAV math matches the books-and-records side.

Option G/L is the same with a `multiplier` (typically 100).

---

# Trading activity & creation/redemption

EMSX (`v_emsx_trades`) returns all FILLED trades with `asset_class` so the
loader can slice equity / option / treasury in one pass. Ticker suffixes
(` Equity`, ` Index`) are stripped in the view.

Block options (`v_block_options`) feed CEF / PF trade compliance using
`Fund_Ticker` resolution (already per-fund in the source view).

ETF creation/redemption flows (`v_etf_flows`):
- `Side` = `BUY` (creation) or `SELL` (redemption)
- `Units` × creation_unit_size = shares created/redeemed
- Used in NAV roll-forward and AP exposure reporting

ETF baskets (`v_etf_basket`): `tif_frwd_data` per-fund per-date — the
expected basket for next-day creations.

---

# Compliance pipeline

Inputs (per fund, per date):
- `v_vest_equity` (with GICS) + `v_vest_option` + `v_vest_treasury`
- `v_fund_metadata` (vehicle_wrapper, diversification_status, expense_ratio)
- `FUND_DEFINITIONS[fund]` (policy flags: `has_listed_option`,
  `flex_option_type`, `option_roll_tenor`, `overwrite`)
- `v_recon_bbg_equity` (to flag unresolved tickers that would break GICS rules)

Tests run:
- `summary_metrics`
- `prospectus_80pct_policy`
- `diversification_40act_check` / `IRS` / `IRC_817h`
- `gics_compliance`
- `max_15pct_illiquid_sai`
- `real_estate_check` / `commodities_check`
- `twelve_d1a_other_inv_cos` / `twelve_d2_insurance_cos` / `twelve_d3_sec_biz`

Private funds skip the 40 Act + IRS checks (excluded by
`diversification_status = 'excluded'`).

---

# Fund cohorts

Configured in `config/run_configurations.py`:

| Cohort | Funds | Custodian |
| --- | --- | --- |
| ETFs (BNY US) | DOGG, FDND, FGSI, KNG, RDVI, SDVD, TDVI | BNY |
| Mutual fund (CCVA) | KNGIX | CCVA |
| VIT | FTCSH | BNY VIT |
| Mutual fund (UMB) | FTMIX | UMB |
| CEFs (UMB) | HE3B1/2/3, P-series, R21126, TR2A1, TR2B1-4 | UMB |
| Private funds (SG) | PD227, PF227, PF27V1 | SG (NAV via SG statement) |

Each cohort × operation (eod / trading_compliance_ante /
trading_compliance_post) is one run configuration. Generated as
`{cohort} × {operation}` dict comprehension, not 22 hand-typed dicts.

---

# Per-fund policy config (`FUND_DEFINITIONS`)

After SQL migration: per-fund dict shrank from ~25 keys to ~10. Everything
data-routing-related now lives in SQL views via `vw_tif_account_numbers`.

```python
"KNG": {
    "option_roll_tenor": "monthly",
    "overwrite": 0.08,
    "has_equity": True,
    "has_listed_option": True,
    "listed_option_type": "single_stock",
    "has_flex_option": False,
    "flex_option_type": None,
    "has_otc": False,
    "has_treasury": False,
    "diversification_status": "diversified",
},
```

What's still here: policy flags (`has_*`), tenor/overwrite policy,
diversification class. Compliance services read these; data layer never
touches them.

---

# Run configuration model

A run configuration is `(cohort, operation, date_window, output_prefix)`.

```python
{
    "name": "eod_compliance_etfs",
    "operation": "eod_compliance",
    "funds": ETF_FUNDS,
    "target_date": today,
    "previous_date": prior_business_day(today),
    "output_prefix": "compliance_results_etfs",
}
```

`fetch_data_stores(session, registry, requests)` runs ONE bulk load per
unique `(target_date, previous_date, analysis_type)` triple and serves all
configurations that share that cache key.

Result: 5 cohorts × 3 EOD ops = 15 configs, but typically 1 SQL load per
date because the keys collapse.

---

# Where to wire the intranet

Recommended entry points:

1. **Trigger UI** — POST `/runs` with `{cohort, operation, target_date}`.
   Backend builds a `DataLoadRequest` and calls `fetch_data_stores`.

2. **Run inspector** — GET `/runs/:id/diagnostics` should call:
   ```sql
   SELECT * FROM reconciliation.v_recon_bbg_equity
   WHERE fund_ticker = :fund AND `date` = :date AND resolution_status = 'Unresolved';
   ```
   Surface the unresolved-Bloomberg-join list so ops can correct data
   before re-running.

3. **Report viewer** — outputs land in
   `G:\Shared drives\Portfolio Management\Funds\Archive\Daily_Compliance\`
   (compliance_results_*), `\NAV_Recon\`, `\Holdings_Recon\`. Web layer
   should index by `(date, cohort, fund)`.

4. **Config editor** — `FUND_DEFINITIONS` should be DB-backed for
   self-service edits to policy flags. Treat `v_fund_metadata` as the
   read model.

---

# Adding a new fund

1. Insert into `accounts_mapping.fund_properties` and `account_numbers`.
2. Verify `accounts_mapping.vw_tif_account_numbers` returns the row
   (strategy must be CEF / PF / TIF / VIT).
3. Add policy dict to `FUND_DEFINITIONS`.
4. Assign to a cohort in `run_configurations.py`.
5. Run validation queries from `docs/architecture_deck.md` slide
   "Validation queries" (next slide).

No view changes required. No loader changes required.

---

# Validation queries (run after install / changes)

```sql
-- 1. Every fund resolves to one metadata row
SELECT COUNT(*) FROM reconciliation.v_fund_metadata;

-- 2. Bloomberg coverage by source
SELECT source, resolution_status, COUNT(*) FROM reconciliation.v_recon_bbg_equity
WHERE `date` = :T GROUP BY source, resolution_status;

-- 3. Per-fund holdings/recon coverage
SELECT v.fund_ticker,
       EXISTS(SELECT 1 FROM reconciliation.v_vest_equity ve WHERE ve.fund_ticker=v.fund_ticker AND ve.`date`=:T) AS has_equity,
       EXISTS(SELECT 1 FROM reconciliation.v_custodian_nav cn WHERE cn.fund_ticker=v.fund_ticker AND cn.`date`=:T) AS has_nav
FROM reconciliation.v_fund_metadata v ORDER BY v.fund_ticker;

-- 4. Spot-check EMSX trades
SELECT fund_ticker, asset_class, COUNT(*) FROM reconciliation.v_emsx_trades
WHERE `date` = :T GROUP BY fund_ticker, asset_class;
```

---

# Operations playbook

| Symptom | First query | Likely fix |
| --- | --- | --- |
| Empty equity recon | `v_custodian_equity` for the fund | Check custody account in `vw_tif_account_numbers` |
| GICS rule mass-fail | `v_recon_bbg_equity` unresolved list | Add SEDOL/ticker to `bbg_equity_flds_blotter` |
| NAV diff = NaN | `v_vest_option` price column for the fund | Check varchar→decimal cast / expired contracts |
| Trades missing for CEF | `v_emsx_trades` and `v_block_options` | CEF/PF use block_options; check `emsx_custom_account` |
| Treasury section showing empty | n/a | Already auto-hidden when no holdings + no price breaks |
| `analysis_type` returns 0 rows | OMS tables for the date | Confirm `'eod'` is what's in `tif_oms_*.analysis_type` |

---

# Implementation order for a new environment

1. Install `accounts_mapping.vw_tif_account_numbers` (Phase 4 base)
2. Install Phase 4 views: custodian + index + recon
3. Install Phase 5 views: vest OMS + trades + lifecycle + metadata
4. Verify all 24 funds resolve in `v_fund_metadata`
5. Run validation queries for `:T = today's business date`
6. Deploy Python (`main.py` + `run_modes.py` + `bulk_data_loader.py`)
7. Run one fund (`--funds DOGG`) for `eod_compliance`
8. Expand to one cohort, then all
9. Wire intranet endpoints to `fetch_data_stores` + report archive

---

# What lives where

| Concern | Owner |
| --- | --- |
| Data routing (which table holds X for custodian Y) | SQL views |
| Column name normalization | SQL views |
| Ticker normalization (` Equity` suffix, `BRK/B`) | SQL views |
| Fund_ticker resolution from account numbers | `vw_tif_account_numbers` |
| GICS / Bloomberg enrichment | SQL views (LEFT JOIN bbg) |
| T vs. T+1 source-date routing | Python (`_load_index_data`) |
| Policy flags (has_flex, diversification_status) | Python (`FUND_DEFINITIONS`) |
| G/L math, compliance rules | Python services |
| Report rendering | Python `reporting/*` |
| Run scheduling, cohort grouping | Python `run_configurations.py` |

If a question fits "what data does X need?" → answer is a view.
If it fits "what should we do with the data?" → answer is Python.

---

# Reference — module map

```
config/
  fund_definitions.py     # 10-key per-fund policy dicts
  run_configurations.py   # cohort × operation matrix
  database.py             # SQLAlchemy session

processing/
  bulk_data_loader.py     # ~200 lines, view-driven
  fund.py                 # Fund dataclass + build_fund_registry
  fund_manager.py         # Orchestrator
  run_modes.py            # eod / trading_compliance entry points

services/
  compliance_checker.py   # 40 Act, IRS, GICS, 12d rules
  reconciliator.py        # Holdings recon (uses v_recon_* shapes)
  nav_reconciliator.py    # Expected TNA + G/L roll-forward

reporting/
  compliance_reporting.py        # right-justified PDF
  reconciliation_reporting.py    # holdings recon PDF + Excel
  holdings_recon_renderer.py     # canonical column tables
  nav_recon_reporting.py         # NAV recon PDF
  trade_compliance_reporting.py  # trading compliance Excel

main.py                   # CLI + batch runner
```

---

# Open enhancements (worth doing post-deploy)

- **Populate `block_fund_identifier`** in `v_fund_properties` — currently
  empty. Required for block-level cost rollups.
- **Move `overlap_benchmark_ticker` to v_fund_properties** so `v_overlap`
  doesn't hardcode `'SPY'`.
- **Vehicle_Wrapper column** on `v_fund_properties` — currently derived
  in `v_fund_metadata` via CASE on EOD_Report_Strategy + 2 hardcoded
  exceptions (KNGIX, FTMIX).
- **Replace bulk_data_loader caching with materialized views** if/when
  intranet load grows past ~50 cohort-runs/day.
- **Parallel view fetches** with one connection per worker — currently
  sequential. Local MySQL is fast enough that this isn't on the critical
  path yet.

---

# End

Questions to direct at:
- SQL views / source data → `accounts_mapping`, `reconciliation` schema owners
- Python loader / services / reporting → engineering
- Fund metadata / policy → fund admin / compliance
- Run scheduling → ops

This deck lives at `docs/architecture_deck.md`. Render to slides with
`marp` (the front-matter is already configured), or read as-is.
