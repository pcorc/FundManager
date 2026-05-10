# VestFundManager — Architecture Reference

A top-down description of what this codebase does, the three external perspectives it reconciles against, the Fund data model that holds them, and the compliance, NAV-reconciliation, and holdings-reconciliation work that runs against them.

---

## 1. What the codebase is

VestFundManager is the Python platform that handles the daily operational and regulatory lifecycle for Vest's fund family. On any given run it loads point-in-time holdings, NAV, cash, trade, corporate-action and index data for one or more funds across one or more dates, then performs three categories of work against that data:

1. **Compliance checking** — does the fund satisfy 40 Act, IRS, IRC, GICS, prospectus 80%, illiquid-asset, sector-cap and 12d1/12d2/12d3 rules?
2. **NAV reconciliation** — does the change in TNA from T-1 to T explain itself as the sum of realized P&L on what we hold plus accruals, distributions and corporate actions?
3. **Holdings reconciliation** — do positions and prices on each side (Vest vs custodian, Vest vs index, Vest vs SocGen) actually agree?

Output is layered Excel workbooks plus matching PDF reports — one of each per asset-class section, plus a combined reconciliation PDF that bundles the NAV summary and per-fund holdings sections into one document.

### Fund universe

The fund universe spans four `vehicle_wrapper` types declared in `config/fund_definitions.py`, each with their own custodial and reporting peculiarities:

| `vehicle_wrapper` | Constant set | Examples | Custodian |
|---|---|---|---|
| `etf` | `ETF_FUNDS` | DOGG, FDND, TDVI, KNG, FGSI | BNY (`bny_us_*`) |
| `etf` (special) | included in `ETF_FUNDS` | KNGIX | CCVA (`ccva_*`) |
| `closed_end_fund` | `CLOSED_END_FUNDS` | HE3B1–4, P20127, P21026, P2726, P30128, P31027, P3727, R21126, TR2B1–4 | UMB (`umb_cef_*`) |
| `private_fund` | `PRIVATE_FUNDS` | (private fund tickers) | — |
| `vit` / `mutual_fund` | `MUTUAL_AND_VIT_FUNDS` | FTCSH, FTMIX, plus VIT trust funds | BNY VIT (`bny_vit_*`) |

These sets are derived directly from each fund definition's `vehicle_wrapper` and `diversification_status` fields and combined or filtered via `build_fund_list()` / `exclude_funds()` for run configurations defined in `config/run_configurations.py`. Additional derived groups exist for option / flex types: `LISTED_INDEX_OPTION_FUNDS`, `LISTED_SINGLE_STOCK_OPTION_FUNDS`, `INDEX_FLEX_FUNDS`, `SINGLE_STOCK_FLEX_FUNDS`, and `DIVERSIFIED_FUNDS` / `NON_DIVERSIFIED_FUNDS`.

For daily batch runs, `run_configurations.py` also pre-bundles three fund cohorts:

- `DAILY_GROUP_1_CEF_PRIVATE` = `CLOSED_END_FUNDS | PRIVATE_FUNDS`
- `DAILY_GROUP_2_ETF` = `ETF_FUNDS`
- `DAILY_GROUP_3_VIT_AND_MF` = `MUTUAL_AND_VIT_FUNDS | {FTCSH, FTMIX, KNGIX}`

and matching named batches of configs to drive them (`DAILY_EOD_AND_RECON_RUNS`, `DAILY_TRADING_COMPLIANCE_RUNS`).

---

## 2. The three stakeholder perspectives

Every fund, on every date, is observed through three independent lenses, and the reconciliation work exists precisely because they don't always agree.

### Vest (internal / OMS / book of record)

This is the authoritative position file — what the portfolio manager believes the fund holds. It feeds from the OMS / pricing system into the `tif_oms_*` tables in the `reconciliation` schema and is the source for:

- `nav_shares` — the NAV-rebalanced share count, used in the ex-post / EOD compliance world
- `iiv_shares` — the intraday-indicative share count, used in the ex-ante / trade-compliance world
- `equity_trades` — trades executed during the day (from `bloomberg_emsx`)
- `cr_rd_data` — corporate actions (splits, dividends, mergers) from the `calendar` schema's `distributions` table

Trades and corporate actions are Vest-side artifacts and live on the snapshot.

### Custodian (third-party administrator / pricing agent)

The fund's official admin record — quantities, prices, and NAV as they will be struck at end of day. Several custodians exist in the universe, each with their own SQLAlchemy-reflected table and column conventions. A lot of the data-loader complexity is fanning out to the right one per fund:

| Custodian | Schema | Holdings / NAV / Cash tables | Funds served |
|---|---|---|---|
| **BNY (US)** | `first_trust_usa` | `bny_us_holdings_v2`, `bny_us_nav_v2`, `bny_us_cash` | ETFs (DOGG, FDND, TDVI, KNG, FGSI…) |
| **BNY (VIT)** | `first_trust_vit` | `bny_vit_holdings`, `bny_vit_nav`, `bny_vit_cash` | VIT trust funds |
| **UMB** | `ftcm` | `umb_cef_px`, `umb_cef_nav`, `umb_cef_cash`, plus `umb_ftmix_*` for FTMIX | Closed-end funds, FTMIX |
| **SocGen** | `first_trust_usa` | `socgen_holdings_statement`, `socgen_equity_statement`, `sg_assignment_statement_v2` | Parallel book for trade-compliance ex-ante/ex-post |
| **CCVA** | `mf_fund_data` | `ccva_holdings`, `ccva_nav`, `ccva_cash` | KNGIX (`fund_id = 980`, `share_class = 'Fund Total'`) |

A handful of custodian-specific quirks are worth remembering:

- **UMB** uses `process_date`, not `date`, as the date column on `umb_cef_px`. Referencing `table.date` on it throws `AttributeError`, and the silent `continue` in the loader exception handler causes all CEF funds to proceed with empty holdings data. The loader resolves this via `_find_date_column(..., 'date', 'process_date', 'effective_date')`.
- **CCVA / KNGIX** cash is sourced from `ccva_holdings` rows where `asset_class == 'Cash Management Vehicle'`, **not** from `ccva_cash` directly. The fund is keyed by `fund_id = 980` and `share_class = 'Fund Total'` across all CCVA tables. NAV uses the `revised_tna` column.
- **Custodian type sniffing**: `FundClass.custodian_type` (and the parallel logic in `bulk_data_loader._query_custodian_holdings_table`) decides between BNY / UMB / SocGen / CCVA by sniffing the table-name substrings of `custodian_equity_holdings`, `custodian_navs`, or `custodian_option_holdings`. The type isn't declared explicitly — it's inferred from configuration.

### Index provider (benchmark)

For the index-tracking funds, this is the third-party constructed target portfolio. The actual table varies by index family:

| Index | Schema | Table | Used by | Date column behavior |
|---|---|---|---|---|
| **CBOE SPATI** | `pricing_data` | `cboe_holdings` | KNG, KNGIX | Loaded daily from CSV (`SPATI_Daily_Inputs_YYYYMMDD.csv`) on the shared drive; expiration date computed as third Friday of the file's month per CBOE convention. `_get_cboe_holdings` filters on `target_date`. |
| **NASDAQ** | `pricing_data` | `nasdaq_holdings`, `nasdaq_pro` | NASDAQ-tracking ETFs (e.g., FDND uses `DJINET` join) | `_get_nasdaq_holdings` filters on `tplus_one` |
| **S&P** | `pricing_data` | `sp_holdings`, `sp_cls` | TDVI and most CEFs (`SPXFIOUP` join) | `_get_sp_holdings` filters on `tplus_one` |
| **DOGG internal** | `pricing_data` | `dogg_index` | DOGG | Equal-weighted at `0.10` per holding; joined with `bbg_equity_flds_blotter` for GICS sector/industry data. Filters on `target_date`. |

Worth verifying: `_get_cboe_holdings` filters on `target_date` while the others use `tplus_one`. This needs confirmation that CBOE SPATI files are dated for T (not T+1) — see Open Items.

These three perspectives are loaded into a single per-fund snapshot at the start of every run, then compared in different combinations depending on the task.

---

## 3. The data model

The Fund framework is the spine of everything. It replaced an earlier dictionary-based pattern (`fund_data["vest_equity"]`, `fund_data["custodian_option_t1"]`, etc.) that scattered defensive `.get()` calls across services.

### Shape

```
Fund
 ├── name, config (from FUND_DEFINITIONS)
 ├── properties: is_etf, is_closed_end_fund, is_private_fund,
 │              is_diversified, is_non_diversified,
 │              has_listed_option, has_flex_option, has_treasury,
 │              listed_option_type, flex_option_type,
 │              uses_index_flex, custodian_type,
 │              flex_option_pattern, option_roll_tenor,
 │              diversification_status, expense_ratio, overwrite,
 │              index_identifier, index_ticker_join,
 │              equity_benchmark_index_ticker
 └── data : FundData
            ├── current  : FundSnapshot   (T)
            └── previous : FundSnapshot   (T-1)
```

The `FundMetrics` class that previously sat alongside `FundSnapshot` has been collapsed into it — all `_compute_*` methods now live directly on `FundSnapshot`, and the lazy-init indirection is gone.

### FundSnapshot

Each `FundSnapshot` carries:

- **Reported scalars** — `cash`, `nav`, `ta`, `tna`, `expenses`, `shares_outstanding`, `flows`
- **Three FundHoldings blocks** — `vest`, `custodian`, `index` (with SocGen blocks accessed alongside where applicable)
- **Computed totals** — `total_equity_value`, `total_option_value`, `total_flex_option_value`, `total_treasury_value`, `cash_value`, `total_assets`, `total_net_assets`. These use prefer-Vest-then-fall-back-to-custodian semantics: Vest is the authoritative source; custodian fills in if Vest is absent.
- **Trade and corporate-action data** — `equity_trades`, `cr_rd_data`, `assignments`, `option_trades`, `flex_option_trades`, `treasury_trades` for the snapshot's date

T+1 data for index funds (whose feeds are dated one day forward) is handled inside the loader using business-day shifting (`pandas.tseries.offsets.BDay`) and is stored against the appropriate `current` / `previous` snapshot — there is no separate `index_tplus_one` field on `FundData`.

### FundHoldings

Each `FundHoldings` block holds four DataFrames: `equity`, `options`, `flex_options`, `treasury`. Flex options are split off from regular options **at load time** based on the fund's `flex_option_pattern` (index flex by `SPX|XSP` regex, single-stock flex by leading `^2` regex), so downstream code doesn't have to re-detect.

`_clean_holdings_dataframe(holding_type)` runs at construction time to coerce all numeric columns to floats per asset class — e.g., on `equity` it cleans `equity_market_value`, `quantity`, `nav_shares`, `iiv_shares`, `price`, `EQY_SH_OUT_million`; on `options` and `flex_options` it cleans the delta-adjusted notional fields. NaNs are filled with `0.0`.

### BulkDataLoader

The `BulkDataLoader` does one pass per run — for every fund, loads custodian holdings, vest holdings, NAV, cash, trades, distributions, assignments, index data, and overlap data for both T and T-1 (plus T+1 for index funds whose feed is dated one day forward). The pass goes by table, not by fund: `_group_funds_by_table` collects every fund pointing at the same custodian table, then queries that table once and fans the results back out.

After all loads, `normalize_all_holdings()` from `utilities/ticker_utils.py` canonicalizes equity and option tickers across all loaded frames before anything else touches them — this enforces the "data transformation once at the source" principle.

### Known gotchas

- **Treasury cross-contamination** — anything pulled via the `_price_quantity_sum` fallback on treasury must be filtered on the `fund` column first, or one fund's treasury position leaks into every other fund. (This produced the DOGG $4B-vs-$75M-of-total-assets bug.) `_filter_frame_by_fund` is the gate.
- **UMB `process_date`** — the column on `umb_cef_px` is `process_date`, not `date`; references to `table.date` throw `AttributeError` and the silent `continue` in the loader exception handler causes downstream compliance to run on empty holdings.
- **`Fund.vehicle`** — currently reads `self.config.get("vehicle")` instead of `self.config.get("vehicle_wrapper")`, silently causing `is_etf`, `is_closed_end_fund`, `is_private_fund` to always return `False`. The constant sets in `fund_definitions.py` use the correct key (`vehicle_wrapper`), so `ETF_FUNDS` etc. are populated correctly — the bug only affects the per-instance `Fund.is_*` properties. Anywhere `fund.is_etf` is consulted is silently going to the false branch.
- **`Fund.is_diversified`** — compares `self.vehicle.lower() == "diversification_status"`, which is comparing the vehicle field to a config *key name*. This can never be true; the property always returns `False`. Should read `self.diversification_status.lower() == "diversified"` for symmetry with `is_non_diversified`.
- **`Fund.is_non_diversified`** — compares `status.lower() == "non_diversified"` (underscore) but the values in `fund_definitions.py` are `"non-diversified"` (hyphen). `compliance_checker.py` already defends against this with `.replace("_", "-")` when normalizing `diversification_status`, but `is_non_diversified` itself returns `False` for the funds that *are* non-diversified.
- **`numpy.bool_` vs Python `bool`** — pandas boolean reductions like `(df["x"] <= limit).all()` return `numpy.bool_`, which serializes to blank in PDF output. Wrap with `bool(...)` consistently.

---

## 4. Per-fund configuration schema

Each entry in `FUND_DEFINITIONS` is a dictionary carrying the fund's full operational profile. The keys are consumed across the loader, the Fund property layer, the compliance checker, and the reconciliators. The fields fall into five groups:

**Data source pointers** (which DB table to read for each role):
- `custodian_equity_holdings`, `custodian_option_holdings`, `custodian_treasury_holdings`, `custodian_navs`, `cash_table`
- `vest_equity_holdings`, `vest_options_holdings`, `vest_treasury_holdings` (always `tif_oms_*`)
- `sg_custodian_holdings` (typically `socgen_holdings_statement`)
- `index_holdings`, `index_ticker_join` (e.g., `SPATI`, `SPXFIOUP`, `DJINET`)
- `option_custodian_assignment` (typically `sg_assignment_statement_v2`)
- `basket`, `flows`, `distributions`, `overlap_table`

**Vehicle and registration**:
- `vehicle_wrapper`: `"etf"` | `"closed_end_fund"` | `"private_fund"` | `"vit"` | `"mutual_fund"`
- `diversification_status`: `"diversified"` | `"non-diversified"`

**Asset class capabilities** (drive which loader paths run and which compliance tests apply):
- `has_equity`, `has_listed_option`, `has_flex_option`, `has_otc`, `has_treasury` (booleans)
- `listed_option_type`: `"index"` | `"single_stock"` | `None`
- `flex_option_type`: `"index"` | `"single_stock"` | `None`

**Strategy parameters**:
- `option_roll_tenor`: `"weekly"` | `"monthly"` | `"quarterly"` — drives `_find_last_settlement_date` in NAV recon
- `overwrite` — the overlay's option overwrite ratio (e.g., 0.08 for KNG, 0.15 for many CEFs)
- `expense_ratio` — annual expense ratio used in NAV reconciliation
- `equity_benchmark_index_ticker`, `overlap_benchmark_ticker` (e.g., `SPY`)

Examples of how this shapes execution: a fund with `has_flex_option=True, flex_option_type="single_stock"` gets the `^2` flex pattern and routes through single-stock flex code paths; a fund with `vehicle_wrapper="closed_end_fund"` triggers `_bulk_load_overlap_data`; KNG with `option_roll_tenor="monthly"` is checked against the third-Friday-of-the-month settlement window, while DOGG/KNGIX with `"weekly"` settle on every Friday.

---

## 5. Compliance checks

These live in `services/compliance_checker.py`, render to Excel via `reporting/compliance_reporting.py`, and to PDF via `ComplianceReportPDF` in the same module. Results flow as a dict keyed by `(fund_name, date_str)` and are flattened to rows at the reporting layer. Footnotes for each test are centralized in `reporting/footnotes.py`.

### Summary Metrics

Always runs. Pulls cash, treasury, equity market value, regular option market value, flex option market value, and total assets off `fund.data.current` for the top-of-report summary block.

### Prospectus 80% Policy

At least 80% of net assets must be in the strategy described in the fund name.

```
Numerator   = total_equity_market_value + Max(CCET − DAN, 0)
Denominator = total_net_assets
CCET        = total_cash_value + total_tbill_value
DAN         = abs(total_option_delta_adjusted_notional)
```

Cash/treasury count toward the 80% only to the extent they aren't already collateralizing the option overlay. Whether options are "in scope" for the 80% test is configured per fund.

### 40 Act Diversification (75/25 rule)

The regulated-fund issuer-concentration test, with explicit early-return for private funds (which are not registered under the 40 Act). Portfolio splits into a 75% "diversified" bucket and a 25% "excluded securities" bucket:

- **Condition 1** — at least 75% of total assets is "qualifying" (non-qualifying ≤ 25%).
- **Condition 2a (issuer test)** — in the 75% bucket, no individual issuer may exceed 5%. Sequential-allocation logic: only issuers with 5%+ individual exposure are eligible for the 25% excluded bucket; they're placed largest-first until the next one would breach the 25% cap; any 5%+ issuer that didn't fit becomes a violation and must reduce below 5% to qualify for the 75% bucket.
- **Condition 2a OCC** — parallel test on the OCC (cleared options) market value/weight: `occ_weight_mkt_val ≤ 0.05 × total_net_assets`.
- **Condition 2b** — in the 75% bucket, no more than 10% of any single issuer's outstanding voting shares (`EQY_SH_OUT_million`) may be held by the fund.

The checker also persists a `diversification_failure_duration_40act` counter — it pulls the prior day's value from `compliance_daily_results` and increments if the fund is failing again today, resetting to 0 on a passing day.

Reporting subtleties:

- Funds with `"Non Diversified"` registered status show `"N/A"` (not red `"FAIL"`) for conditions they're permitted to fail.
- Long flex options must be added to issuer exposure in 2a (they increase concentration on the underlier).
- Short listed options must **not** be netted out — you can't shrink an issuer's weight by writing a call on it.
- For index-option-only funds (`has_listed_option=True, listed_option_type="index"`), `option_market_value` is zeroed in the consolidation since the options aren't single-name exposure.

### IRS Diversification

IRS §851 test for regulated investment companies. Five conditions surfaced:

- **Condition 1** — 90% of income from qualifying sources
- **Condition 2a (50%)** — at least 50% of assets in qualifying securities
- **Condition 2a 5%** — for that 50%, no issuer > 5% of fund assets
- **Condition 2a 10%** — for that 50%, fund doesn't hold > 10% of any issuer's outstanding float
- **Condition 2b** — no single issuer > 25% of assets

### IRC Diversification

§817(h) test, applied **only to VIT funds** (`vehicle_wrapper == "vit"`). PDF rendering checks "is any fund in this run a VIT?" before processing; non-VIT funds show N/A when the section is still rendered.

### GICS Compliance

Industry concentration check at the GICS_INDUSTRY_NAME level (simplified down from the original sector / industry-group / industry triple-level test to industry-only at 25%). The compliance group is resolved via `_resolve_gics_compliance_group`:

| Group | Funds | Rule |
|---|---|---|
| `tdvi` | TDVI | Fails if any GICS industry > 25%, **with an exception for Information Technology** which is allowed to exceed |
| `kng_fdnd` | KNG, FDND | Can exceed 25% in any industry *if the index also exceeds 25% there* — requires index DataFrame from `fund.data.current.index.equity` |
| `dogg` | DOGG | Standalone; evaluated purely on its own industry concentrations, no index comparison |
| default | everything else | Strict 25% on industry only |

The industry-to-sector mapping (used by the `tdvi` group) is built from the equity holdings DataFrame's GICS columns via `_build_industry_to_sector_map`, with graceful handling of nulls and trimming of whitespace.

DOGG is currently triple-excluded from GICS:
1. `skip_tests_by_ticker` set in `compliance_checker.py`
2. Explicit `if fund_name == "DOGG": continue` in `compliance_reporting.py`
3. `_resolve_gics_compliance_group` routing

Open question: whether DOGG exclusions are temporary placeholders pending finalized `"dogg"` group logic.

The GICS overview renders as a standalone PDF table titled "GICS Diversification", injected right after the Summary Metrics section.

### 15% Illiquid Assets (SAI)

Checks `is_compliant` against the SAI's illiquid-asset cap.

### Real Estate / Commodities

Per-asset-class checks with explicit PASS/FAIL status display.

### Rule 12d1-1 (Other Investment Companies)

Investments in other ICs are capped.

### Rule 12d2 (Insurance Companies)

Investments in insurance companies are limited.

### Rule 12d3 (Securities-Related Businesses)

Two sub-rules and a combined overall status, all displayed:

- **Rule 1** — limit on equity of any sec-related business (`RULE_12D3_EQUITY_LIMIT`)
- **Rule 2** — limit on debt of any sec-related business

Dataframes use `eqyticker` (not `ticker`). The `numpy.bool_` issue lives here: `(sec_related["ownership_pct"] <= RULE_12D3_EQUITY_LIMIT).all()` returns `numpy.bool_`, which serializes to blank in PDF output unless wrapped in `bool(...)`.

### General pattern fix across multiple checks

`_consolidate_holdings_by_issuer` previously overwrote `nav_shares` / `iiv_shares` to zero post-consolidation. The fix preserves existing share columns and only initializes to zero when the column doesn't exist or contains NaN.

---

## 6. NAV reconciliation

Lives in `services/nav_reconciliator.py` (`NAVReconciliator`) with reporting in `reporting/nav_recon_reporting.py` and `reporting/combined_reconciliation_report.py`. Dataclass shapes (`AssetClassGainLoss`, `TickerGainLoss`, `NAVComponents`, `NAVSummary`, `NAVReconciliationResults`) live in `services/nav_recon_dataclasses.py`.

**Question being answered:** the change in TNA from T-1 to T — can we explain that change as the sum of realized gains/losses on each asset class, plus accruals, distributions, expenses, and corporate actions?

### Orchestrator (`run_nav_reconciliation`)

1. **Settlement-day detection** — `fund.is_option_settlement_date(analysis_date)` decides whether today is an assignment day. Branches the rest of the calculation.
2. **Equity G/L** — both **raw** (vest price T-1 vs vest price T) and **adjusted** (after trade and corporate action adjustments), per ticker. `_calculate_equity_gl` returns an `AssetClassGainLoss` with ticker-level detail; the Excel sheet renders individual quantities and prices with **formulas** (not static values) so the user can re-derive totals in cell.
3. **Option G/L** — on a non-assignment day, mark-to-mark between vest prices at T-1 and T. On an assignment day, `_process_assignments` handles expired options' P&L (this is the consolidated version — the old wrapper layer is gone) and `_calculate_rolled_option_gl` calculates the contribution of new rolled options using executed prices from the trade dataframe rather than mark prices. `_find_last_settlement_date` walks back to the most recent settlement based on the fund's `option_roll_tenor`:
   - Weekly = last Friday
   - Monthly = last third Friday
   - Quarterly = last third Friday of Mar/Jun/Sep/Dec
4. **Flex Option G/L** — same shape as options, with flex-specific pattern detection.
5. **Treasury G/L** — straight price × quantity comparison.
6. **Assignment G/L** — only non-zero on assignment days.
7. **Dividends, expenses, flows, distributions, other** — pulled from the snapshot and the NAV table; each returns a scalar that lands in `NAVComponents`.

### Detail columns

The `TickerGainLoss` dataclass and `DETAIL_COLUMNS` on `NAVReconciliator` drive the per-ticker Excel detail rows:

```
ticker, quantity_t1, quantity_t,
price_t1_raw, price_t_raw,
price_t1_adj, price_t_adj,
gl_raw, gl_adjusted
```

`equity_details`, `option_details`, `flex_details`, `treasury_details` are accumulated as DataFrames during the run.

### Dataclass-through-pipeline pattern

`NAVReconciliationResults` dataclasses flow through the pipeline and are only converted to dicts via `convert_nav_results_to_dicts()` (using `to_legacy_dict()`) at the reporting layer. The typed structure is preserved until output time.

### Combined Reconciliation Report

`CombinedReconciliationPDF` (in `combined_reconciliation_report.py`, mixed in with `HoldingsReconciliationRenderer`) is the all-in-one PDF. The "Gain/Loss Components by Fund" summary page contains:

`Beginning TNA | Equity G/L | Option G/L | Flex G/L | Treasury G/L | Accruals | Expenses | Distributions | Ending TNA | Custodian NAV | NAV Diff`

Rules for this page:
- All-zero columns (often Distributions) are hidden
- Must fit one landscape page
- Any fund whose section dataframe is empty (often treasury) is not rendered at all — `_has_holdings_data` gates this
- `_count_price_breaks` counts only T price breaks, not T-1

The per-fund NAV worksheet uses Excel cell formulas for the Expected TNA derivation — `=C{adj_beg_tna} + C{equity_gl_adj} + C{option_gl_adj} + ... + C{expenses} + C{dividends} + C{distributions}` — so any cell-level edit reflows the recon.

---

## 7. Holdings reconciliation — custodian side

Position-by-position recon. `Reconciliator` (`services/reconciliator.py`) produces `ReconciliationResult` dataclasses:

```
ReconciliationResult
 ├── raw_recon
 ├── final_recon
 ├── price_discrepancies_T
 ├── price_discrepancies_T1
 ├── merged_data
 ├── regular_options   (option recon only)
 └── flex_options      (option recon only)
```

The Excel reporter (`reporting/reconciliation_reporting.py`) writes a tab per asset class via `ReconDescriptor` declarations. The PDF reporter (`combined_reconciliation_pdf.py` via the renderer mixin) inlines them into the combined report.

### Asset-class methods

Each compares Vest against custodian on a CUSIP/ticker key:

- **`reconcile_custodian_equity`** — joins Vest equity on `eqyticker` to custodian equity (UMB: `security_tkr` → `eqyticker`, `mkt_qty` → `shares_cust`, `eod_close` → `price`, `mkt_mktval` → `market_value`, filtered to `security_catgry IN ('COMMON','REIT')`). Vest quantity is adjusted for trades and corporate actions on the day. Computes quantity discrepancies and price discrepancies T and T-1. Small price differences (< $1) are auto-overridden to the custodian value so they don't flag as breaks.

- **`reconcile_custodian_option`** — same shape on `optticker`. UMB option tickers are built at query time: take `security_desc`, splice `'US '` in after the first space (so `AAPL 240119C00200000` becomes `AAPL US 240119C00200000`). Result is split into `regular_options` and `flex_options` based on the fund's flex pattern, but the FLEX side is always returned empty here since flex is handled separately.
  - Quantity mismatches show `"Vest=X | Cust=Y"`
  - Presence mismatches show `"Present in Vest: Yes/No | Present in Cust: Yes/No"`

- **`reconcile_custodian_treasury`** — CUSIP-joined. Skipped entirely for funds where treasury frames are empty.

- **`reconcile_custodian_flex_option`** — dedicated method (built during the property-driven refactor) that handles flex end-to-end, separately from regular options.

- **`reconcile_sg_equity` / `reconcile_sg_option`** — SocGen comparison; used for the ex-ante/ex-post side of the trade compliance flow rather than the EOD flow. Activated when `analysis_type != "eod"`.

Ticker normalization (`normalize_equity_pair`, `normalize_option_pair` from `utilities/ticker_utils.py`) runs per pair before merging.

The combined PDF holdings section excludes T-1 sections explicitly — no separate "Custodian Equity T1" / "Custodian Option T1" sections. T-1 price breaks roll into the T section as additional context. Empty sections are skipped.

---

## 8. Index reconciliation — the third lens

Specific to index-tracking funds (KNG / KNGIX, FDND, TDVI, and the CEFs tracking S&P), and distinct from custodian recon.

**`reconcile_index_equity`** joins `fund.data.current.vest.equity` to `fund.data.current.index.equity` on `eqyticker` via outer join with indicator, and produces:

- **`holdings_discrepancies`** — names in Vest but not Index, or in Index but not Vest. Records carry `in_vest`, `in_index`, both weights, and absolute weight diff.
- **`significant_diffs`** — names in both, where `abs(nav_wgt_begin − weight_index) > 0.001`.

Summary rows label as `"Index Equity — Missing in OMS"` / `"Missing in Index"` or `"Index Weight Diff"` with the actual OMS vs Index weights printed.

### Index sources by fund

- **KNG / KNGIX** → `cboe_holdings` in `pricing_data` schema. Loaded from CBOE SPATI daily input CSVs at `G:\Shared drives\Portfolio Management\Funds\Archive\Index Data\KNG` via `load_kng_index_files.py` (review-stage export) and `import_spati_backfill.py` (DB import). `expiration_date` computed as third Friday of file's month. `ENABLE_DELETE = False` safety toggle gates the delete block.
- **FDND** → NASDAQ feed (`nasdaq_holdings`), `index_ticker_join = "DJINET"`
- **TDVI / others** → S&P feed (`sp_holdings`), various `index_ticker_join` values
- **CEFs** → S&P feed (`sp_holdings`, `index_ticker_join = "SPXFIOUP"`)
- **DOGG** → internal `dogg_index` table (equal-weighted, 0.10 per name); joined with `bbg_equity_flds_blotter` for GICS sector/industry data at query time

`_get_cboe_holdings` filters on `target_date` directly while `_get_nasdaq_holdings` / `_get_sp_holdings` filter on `tplus_one`. CBOE on T vs T+1 needs confirmation given how SPATI files are dated — see Open Items.

DOGG is currently bypassed in `reconcile_index_equity` with an early return — same exclusion question as GICS.

---

## 9. Trade compliance — ex-ante vs ex-post

Lives in `reporting/trade_compliance_analyzer.py` (the comparison engine) and `reporting/trade_compliance_reporting.py` / `reporting/trade_compliance_reporter.py` (Excel/PDF output and orchestration).

The flow is two parallel compliance runs against the same fund universe:

- **Ex-ante run** — uses `iiv_shares` (intraday-indicative count), reflecting positions *before* trades settle
- **Ex-post run** — uses `nav_shares` (NAV-rebalanced count), reflecting positions *after* trades settle

Both runs go through the same `ComplianceChecker` machinery; the difference is the share series resolved by `_resolve_shares_series` in `bulk_data_loader.py`, which keys off `analysis_type`.

`TradingComplianceAnalyzer.analyze()` then compares the two result sets per fund, producing a `FundComplianceComparison` per fund with:
- `overall_before` / `overall_after` — overall PASS/FAIL status
- `status_change` — `UNCHANGED` | `INTO_COMPLIANCE` | `OUT_OF_COMPLIANCE` | `STILL_COMPLIANT_WITH_CHANGES` | etc.
- `violations_before` / `violations_after` — count of failing checks
- `num_changes` — count of checks whose status flipped
- `checks` — per-check before/after dict

`build_trading_compliance_reports` produces the trading compliance Excel/PDF. `combine_trading_and_compliance_reports` merges the trading output with the ex-post compliance output into a combined Excel and PDF — useful for delivering the "what trades did we do and what did they change" story alongside the EOD compliance picture.

`run_modes.run_trading_mode` is the entry point; it bulk-loads two data stores (one for each analysis_type) and runs both flows in sequence.

---

## 10. How internal definition reconciles against reality

For every fund in a run configuration, for every date in the requested range:

1. `BulkDataLoader` produces a single populated `Fund` object whose `data.current` and `data.previous` snapshots carry — side by side — Vest's book, the custodian's book, and (where applicable) the index. All three are filtered to that fund and date during loading. `_filter_frame_by_fund` keeps one fund's positions from contaminating another's totals.
2. The `Fund` object is then handed independently to three services:

   | Service | Question |
   |---|---|
   | `ComplianceChecker` | Given what Vest holds, does the fund satisfy each rule? (Vest authoritative, custodian fallback for totals) |
   | `NAVReconciliator` | Given Vest holdings at T-1, trades and corporate actions during the day, and Vest prices at T — does the calculated TNA equal the custodian's reported NAV at T? |
   | `Reconciliator` | (a) Vest vs Custodian on positions and prices at T and T-1, per asset class. (b) Vest vs Index on weights, for index-tracking funds. (c) Vest vs SocGen for funds with a parallel SG book (trade compliance only) |

3. The discrepancies surfaced — failed compliance tests, NAV gaps, position/price/weight breaks — are what the PM investigates.

Everything else in the codebase (run-configuration system, time-series stacking, report renderers, fund-group constants) is plumbing to do that for any subset of funds across any range of dates without rewriting the orchestration each time.

---

## 11. Run configurations and orchestration

`main.py` calls `run_configuration_batch()` with configurations defined in `config/run_configurations.py`. Each configuration is a template with parameters that get computed at runtime:

- **Dynamic dates** — `base_date` (the run date) drives T-1 and T-2 business-day calculations
- **Fund groups** — passed via `build_fund_list()` combining group constants, or `exclude_funds()` to remove specific tickers
- **Output tagging** — each configuration includes an `output_tag` (e.g., `"etfs"`, `"cefs_pfs"`, `"vitmf"`) that gets folded into output filenames so concurrent runs don't collide
- **Date mode** — `date_mode: "single"` for one-shot, `date_mode: "range"` for time series; `generate_daily_reports: False` produces a single stacked file across multiple dates

The pre-built templates cover trading compliance, EOD compliance checks, and EOD reconciliation across `ETF_FUNDS`, `CLOSED_END_FUNDS`, `PRIVATE_FUNDS`, `MUTUAL_AND_VIT_FUNDS`, and custom selections. For stacked time-series compliance output, use `time_series_full_compliance` with `start_date` / `end_date` in the override dict and `generate_daily_reports: False`.

Two pre-bundled batch lists ready for daily orchestration:

```python
DAILY_EOD_AND_RECON_RUNS = [
    "eod_compliance_daily_cef_private",
    "eod_recon_daily_cef_private",
    "eod_compliance_daily_etf",
    "eod_recon_daily_etf",
    "eod_compliance_daily_vitmf",
    "eod_recon_daily_vitmf",
]

DAILY_TRADING_COMPLIANCE_RUNS = [
    "trading_compliance_daily_cef_private",
    "trading_compliance_daily_etf",
    "trading_compliance_daily_vitmf",
]
```

For time-series compliance, `time_series_diversification` (CEFs + select private funds, DIVERSIFICATION_TESTS only) and `time_series_full_compliance` (closed-end funds, full compliance suite) are the standard templates.

---

## 12. Database schema map

`config/database.py` reflects ten schemas with SQLAlchemy's automap. The role of each:

| Schema | Tables | Role |
|---|---|---|
| `accounts_mapping` | `account_numbers`, `funds`, `fund_service_providers`, view `v_fund_properties` | Fund / account identity. `fund_recon_mappings` is reflected but unused downstream (candidate for removal) |
| `first_trust_vit` | `bny_vit_holdings`, `bny_vit_nav`, `bny_vit_cash` | BNY custodian for VIT funds |
| `first_trust_usa` | `bny_us_holdings_v2`, `bny_us_nav_v2`, `bny_us_cash`, `socgen_holdings_statement`, `socgen_equity_statement`, `tif_frwd_data`, `master_accounts`, `etf_flows` | BNY custodian for US ETFs + SocGen statements |
| `ftcm` | `umb_cef_px`, `umb_cef_nav`, `umb_cef_cash`, `umb_ftmix_px`, `umb_ftmix_nav`, `overlap_test` | UMB custodian for CEFs (and FTMIX) |
| `mf_fund_data` | `ccva_holdings`, `ccva_nav`, `ccva_cash` | CCVA custodian for KNGIX |
| `bloomberg_emsx` | `bbg_equity_flds_blotter`, `bbg_options_flds_blotter`, `emsx_equity_route_sub`, `emsx_equity_order_sub`, `bbg_feed_equity_closes`, `daily_bbg_flex_pricing` | Trades, equity reference data, flex pricing |
| `compliance` | `compliance_daily_results`, `gics_mapper` | Persisted compliance results (used by 40 Act `diversification_failure_duration_40act` lookback) + GICS mapping reference |
| `pricing_data` | `cboe_holdings`, `sp_cls`, `sp_holdings`, `nasdaq_holdings`, `nasdaq_pro`, `dogg_index`, `tif_index_mappings`, `tif_iiv_index_def` | Index data (CBOE SPATI, S&P, NASDAQ, DOGG internal) and IIV definitions |
| `reconciliation` | `tif_oms_option_holdings`, `tif_oms_equity_holdings`, `tif_oms_treasury_holdings` | Vest's OMS / book of record |
| `calendar` | `settlement_holidays`, `distributions` | Settlement calendars and corporate actions |

The default `DB_CONNECTION_STRING` lives in `config/database.py` — overridable via env var. Reflection happens once at module import via `initialize_database()`.

---

## 13. Key files

```
main.py                                  Entry point; calls run_configuration_batch
utilities/main_helpers.py                filter_registry, log_processing_summary, date coercion
config/run_configurations.py             Configuration templates + date calculations
config/fund_definitions.py               FUND_DEFINITIONS, fund group constants
config/fund_registry.py                  FundRegistry / FundClass (pending elimination)
config/database.py                       SQLAlchemy engine, table reflection
processing/fund.py                       Fund, FundData, FundSnapshot, FundHoldings
processing/bulk_data_loader.py           Single-pass loader, custodian/vest/index/cash/nav queries
processing/fund_manager.py               Per-fund orchestration
processing/run_modes.py                  run_eod_mode, run_eod_range_mode, run_trading_mode
services/compliance_checker.py           All compliance test methods
services/reconciliator.py                Holdings recon (custodian + index + SG)
services/nav_reconciliator.py            NAV reconciliation engine
services/nav_recon_dataclasses.py        AssetClassGainLoss, NAVComponents, NAVSummary, NAVReconciliationResults, TickerGainLoss
reporting/compliance_reporting.py        Excel + PDF compliance reports
reporting/reconciliation_reporting.py    Excel holdings recon reports (ReconDescriptor map)
reporting/nav_recon_reporting.py         Excel + PDF NAV recon reports
reporting/combined_reconciliation_report.py   Combined NAV + holdings PDF
reporting/holdings_recon_renderer.py     Mixin for holdings sections in combined PDF
reporting/trade_compliance_analyzer.py   Ex-ante vs ex-post compliance comparison
reporting/trade_compliance_reporting.py  Trade compliance Excel + PDF generation
reporting/trade_compliance_reporter.py   Orchestrator (build_trading_compliance_reports, combine_*)
reporting/footnotes.py                   Footnotes for each compliance test
utilities/ticker_utils.py                normalize_all_holdings, normalize_equity_pair, normalize_option_pair
```

---

## 14. Open / pending items

- **`fund_registry.py` elimination** — five-step plan to fold capabilities into `fund.py`; four call sites to update (`main.py`, `main_helpers.py`, `run_modes.py`, `bulk_data_loader.py`). The required-tables, account-number-loading, and custodian-type-sniffing logic all need to move first.
- **`Fund.vehicle` bug** — reads `vehicle` instead of `vehicle_wrapper`, breaking `is_etf` / `is_closed_end_fund` / `is_private_fund` checks silently. Fund-group constant sets in `fund_definitions.py` are NOT affected (they use the correct key), but every `fund.is_etf` consult in service code is silently routing through the `False` branch.
- **`Fund.is_diversified` bug** — compares vehicle to `"diversification_status"` (a config key name); always returns `False`. Should compare `self.diversification_status.lower() == "diversified"`. Also the docstring is wrong ("Check if fund is closed-end").
- **`Fund.is_non_diversified` bug** — compares to `"non_diversified"` (underscore) but real values are `"non-diversified"` (hyphen). Always returns `False` for non-diversified funds.
- **UMB CEF loader** — `process_date` vs `date` column mismatch; the silent `continue` in the loader exception handler should be replaced with explicit error logging so empty-holdings cascades surface.
- **DOGG GICS exclusions** — three-point exclusion may be a temporary placeholder; reporting-layer exclusion may need to be lifted alongside the checker-level skip, once finalized DOGG group logic is in place.
- **`fund_recon_mappings`** — appears only in the SQLAlchemy reflection list comment, not queried downstream; safe removal candidate.
- **`_get_cboe_holdings` date filter** — uses `target_date` while NASDAQ / S&P use `tplus_one`; confirm whether CBOE SPATI files are dated for T (consistent with current behavior) or T+1 (would need change).
- **`numpy.bool_` wrapping** — apply `bool(...)` consistently across all compliance check methods to prevent PDF serialization blanks (notably in 12d3, but the pattern recurs).
- **Custodian type explicit declaration** — currently `custodian_type` is sniffed from table-name substrings (`"bny"`, `"umb"`, `"socgen"`, `"ccva"`). Adding an explicit `custodian_type` field to each `FUND_DEFINITIONS` entry would remove a brittle substring match and make the BNY-US-vs-BNY-VIT distinction explicit.
