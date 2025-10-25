# config/fund_classifications.py
"""
Fund classifications for 40 Act compliance testing.
Based on regulatory requirements and fund structure types.
"""

# Fund classifications based on compliance ruleset
DIVERSIFIED_FUNDS = {
    'KNGIX', 'KNG',
    # CEF funds (Closed-End Funds) - all are diversified
    'HE3B1', 'TR2B1', 'HE3B2', 'TR2B2',
    'P20127', 'P21026', 'P2726', 'P30128', 'P31027', 'P3727',  #
    'R21126'
}

NON_DIVERSIFIED_FUNDS = {
    # ETF funds - all are non-diversified
    'DOGG', 'FDND', 'FGSI', 'FTCSH', 'FTMIX',
    'RDVI', 'SDVD', 'TDVI'
}

PRIVATE_FUNDS = {
    # Private funds - not subject to 40 Act (should be excluded from test)
    'PD227', 'PF227', 'PF27V1'
}

CLOSED_END_FUNDS = {
    'P20127', 'P21026', 'P2726', "P30128", 'P31027', 'P3727', 'R21126', 'HE3B1', 'HE3B2', 'TR2B1', 'TR2B2'
}

FUNDS_WITH_SG_EQUITY = {"PF227", "PD227", "PF27V1"}

