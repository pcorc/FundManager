"""Project-wide constants used by compliance and reporting services."""
from __future__ import annotations

# Prospectus / Names rule configuration
PROSPECTUS_MIN_THRESHOLD: float = 0.80
PROSPECTUS_OPTIONS_FUNDS = {"KNG", "KNGIX", "DOGG", "FTCSH", "FGSI"}

# IRS diversification thresholds
IRS_QUALIFYING_ASSETS_MIN: float = 0.50
IRS_BOTTOM_50_LIMIT: float = 0.05
IRS_OWNERSHIP_LIMIT: float = 0.10

# 40 Act diversification
ACT_40_QUALIFYING_ASSETS_MIN: float = 0.75
ACT_40_ISSUER_LIMIT: float = 0.05
ACT_40_OWNERSHIP_LIMIT: float = 0.10

# IRC thresholds
IRC_TOP_1_LIMIT: float = 0.55
IRC_TOP_2_LIMIT: float = 0.70
IRC_TOP_3_LIMIT: float = 0.80
IRC_TOP_4_LIMIT: float = 0.90

# Illiquidity and equity floors
ILLIQUID_MAX_THRESHOLD: float = 0.15
EQUITY_MIN_THRESHOLD: float = 0.85

# Rule 12d limits
RULE_12D1A_OWNERSHIP_LIMIT: float = 0.03
RULE_12D1A_SINGLE_ASSETS_LIMIT: float = 0.05
RULE_12D1A_TOTAL_ASSETS_LIMIT: float = 0.10
RULE_12D2_INSURANCE_LIMIT: float = 0.05
RULE_12D3_EQUITY_LIMIT: float = 0.05
RULE_12D3_DEBT_LIMIT: float = 0.25
RULE_12D3_ASSET_LIMIT: float = 0.10

# Vehicle helper tokens
VEHICLE_CLOSED_END = "closed_end_fund"
VEHICLE_PRIVATE = "private_fund"
VEHICLE_ETF = "etf"
VEHICLE_VIT = "vit"

# Flex option helpers
FLEX_BENCHMARK_TICKER = "SPX"