# utilities/footnotes.py

FOOTNOTES = {
    "gics": [
        "Certain funds may not have more than 25% of total assets in any industry or group of industries. "
        "However, other funds may have more than 25% of total assets in an industry or group of industries if "
        "(1) the fund’s Underlying ETF invests more than 25% of its assets in that industry or group of industries or "
        "(2) the fund is specifically permitted to concentrate in a specified industry or group of industries in its "
        "fundamental investment policy (such as information technology). Therefore, some funds have additional info listed "
        "about their underlying ETF (represented by an index) or have a note that they can concentrate in a certain industry "
        "or group of industries."
    ],

    "prospectus_80pct": [
        "Note: Some funds' 80% policies include the options positions while others only include equity exposure.",
        "CCET = Cash, Cash Equivalents, and T-bills (less than 1-year maturity)."
    ],

    # "40act": [
    #     "Condition 40 Act 1: 75% of the fund must be invested in diversified assets.",
    #     "Condition 40 Act 2a: No single issuer in the 75% can exceed 5% of total assets.",
    #     "Condition 40 Act 2b: No single issuer in the 75% can exceed 10% of the issuer's voting securities."
    # ],

    "40act": [
        "To be classified as a diversified company under the 1940 Act, at least 75% of the value of the fund's total assets must be invested in:",
        "1) cash and cash items (including receivables),",
        "2) government securities,",
        "3) securities of other investment companies, and",
        "4) other securities",
        "",  # Empty line for separation
        "Compliance Conditions:",
        "Condition 40 Act 1: 75% of the fund must be invested in diversified assets.",
        "Condition 40 Act 2a: No single issuer in the 75% can exceed 5% of total assets.",
        "Condition 40 Act 2b: No single issuer in the 75% can exceed 10% of the issuer's voting securities."
    ],

    "irs": [
        "IRS Condition 1: 90% of income must come from qualifying sources.",
        "IRS Condition 2a: At least 50% of assets must be allocated to qualifying securities.",
        "IRS Condition 2a5: for 50% of portfolio, no issuer is more than 5% of fund assets.",
        "IRS Condition 2a10: for 50% of portfolio, fund doesn't hold more than 10% of any issuer's outstanding float.",
        "IRS Condition 2b: No single issuer may constitute more than 25% of assets."
    ],

    "irc": [
        "This test only applies to variable insurance trusts. Under applicable regulations, the investments of a "
        "segregated asset account generally will be deemed adequately diversified only if: "
        "(i) no more than 55% of the value of the total assets of the account is represented by any one investment; "
        "(ii) no more than 70% of such value is represented by any two investments; "
        "(iii) no more than 80% of such value is represented by any three investments; and "
        "(iv) no more than 90% of such value is represented by any four investments."
    ],

    "12d1": [
        "The Fund will not purchase or otherwise acquire: "
        "(i) more than 3% of the total outstanding voting stock of the acquired company; "
        "(ii) securities issued by the acquired company having an aggregate value in excess of 5% of the value of the "
        "total assets of the acquiring company; or "
        "(iii) securities issued by the acquired company and all other investment companies (other than treasury stock of "
        "the acquiring company) having an aggregate value in excess of 10% of the value of the total assets of the acquiring company."
    ],

    "12d2": [
        "The Fund may not own more than 10% of the total outstanding voting stock of an insurance company."
    ],

    "12d3": [
        "Notwithstanding section 12(d)(3) of the Act, an acquiring company may acquire any security issued by a person "
        "that, in its most recent fiscal year, derived more than 15 percent of its gross revenues from securities related activities, "
        "provided that: "
        "(1) Immediately after the acquisition of any equity security, the acquiring company owns not more than 5% of the outstanding securities "
        "of that class of the issuer's equity securities; "
        "(2) Immediately after the acquisition of any debt security, the acquiring company owns not more than 10% of the outstanding principal amount "
        "of the issuer's debt securities; and "
        "(3) Immediately after any such acquisition, the acquiring company has invested not more than 5% of the value of its total assets in the securities of the issuer."
    ],

    "real_estate": [
        "We search for tags in Bloomberg for any fund holdings categorized as real estate.",
        "If none, it reports 'None'."
    ],

    "commodities": [
        "We search for tags in Bloomberg for any fund holdings categorized as commodities.",
        "If none, it reports 'None'."
    ],

    "illiquid": [
        "We search for tags in Bloomberg for any illiquid or restricted securities.",
        "If none, it reports 'None'."
    ]
}
