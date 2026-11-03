"""Golden-master scenarios: each entry is the kwargs for ``build_run``.

``cohorts`` holds cohort *names*; the runner resolves them to the live sets
after the cohort load, exactly as ``main.py``'s ``__main__`` block does.
"""

SCENARIOS = {
    "eod_etf": dict(mode="eod", cohorts=["ETF_FUNDS"], start_date="2026-10-30", output_tag="etfs"),
    "eod_cef": dict(mode="eod", cohorts=["CLOSED_END_FUNDS"], start_date="2026-10-30", output_tag="cefs"),
    "eod_vit": dict(mode="eod", cohorts=["VIT_AND_MUTUAL_FUNDS"], start_date="2026-10-30", output_tag="vit"),
    "eod_private": dict(mode="eod", cohorts=["PRIVATE_FUNDS"], start_date="2026-10-30", output_tag="private"),
    "compliance_etf": dict(mode="compliance", cohorts=["ETF_FUNDS"], start_date="2026-10-30"),
    "compliance_cef_40act": dict(
        mode="compliance", cohorts=["CLOSED_END_FUNDS"], start_date="2026-10-30",
        compliance_tests=["diversification_40act_check"],
    ),
    "compliance_etf_two_rules": dict(
        mode="compliance", cohorts=["ETF_FUNDS"], start_date="2026-10-30",
        compliance_tests=["diversification_IRS_check", "prospectus_80pct_policy"],
    ),
    "compliance_cef_all_empty": dict(
        mode="compliance", cohorts=["CLOSED_END_FUNDS"], start_date="2026-10-30", compliance_tests=[],
    ),
    "nav_etf": dict(mode="nav", cohorts=["ETF_FUNDS"], start_date="2026-10-30"),
    "recon_etf": dict(mode="reconciliation", cohorts=["ETF_FUNDS"], start_date="2026-10-30"),
    "trading_etf": dict(mode="trading_compliance", cohorts=["ETF_FUNDS"], start_date="2026-10-30", output_tag="etfs"),
    "time_series_cef": dict(
        mode="time_series", cohorts=["CLOSED_END_FUNDS"], start_date="2026-10-26", end_date="2026-10-30",
        compliance_tests=["diversification_40act_check", "diversification_IRS_check"], output_tag="cefs_ts",
    ),
    "per_day_weekend_range": dict(
        mode="compliance", cohorts=["ETF_FUNDS"], start_date="2026-10-24", end_date="2026-10-27",
    ),
    "multi_mode_funds": dict(
        mode=["compliance", "nav"], funds=["KNG", "RDVI"], start_date="2026-10-30", output_tag="pair",
    ),
    # A VIT fund next to an ETF: the IRC PDF section has a column for a fund the IRC test does not apply to.
    "compliance_vit_mix": dict(
        mode="compliance", funds=["FTCSH", "KNG"], start_date="2026-10-30", output_tag="vitmix",
    ),
}

# Direct RunRequest kwargs (the UI path: cohort names resolved inside run()) that
# must reproduce an existing scenario exactly: name -> (base scenario, kwargs).
EQUIVALENT_REQUESTS = {
    "request_compliance_cef_40act": ("compliance_cef_40act", dict(
        operations=("compliance",), cohorts=("CLOSED_END_FUNDS",), start_date="2026-10-30",
        compliance_tests=("diversification_40act_check",), output_tag="compliance",
    )),
    "request_trading_etf": ("trading_etf", dict(
        operations=("trading_compliance",), cohorts=("ETF_FUNDS",), start_date="2026-10-30", output_tag="etfs",
    )),
    "request_time_series_cef": ("time_series_cef", dict(
        operations=("compliance",), cohorts=("CLOSED_END_FUNDS",), start_date="2026-10-26", end_date="2026-10-30",
        compliance_tests=("diversification_40act_check", "diversification_IRS_check"), date_mode="stacked",
        create_pdf=False, output_tag="cefs_ts",
    )),
}
