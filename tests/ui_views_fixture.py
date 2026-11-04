"""Small hand-made ``views.json`` contents (schema of ``reporting.result_views``) for the UI tests; no real data."""
from __future__ import annotations

DAY = "2026-10-30"


def _rule(title, label, status, rows, details=(), tables=(), notes=(), reason=None):
    return {"title": title, "label": label, "status": status, "reason": reason, "rows": list(rows),
            "details": list(details), "tables": list(tables), "notes": list(notes)}


def eod_views() -> dict:
    irs_kng = _rule("IRS Diversification", "IRS Diversification", "FAIL", [
        {"label": "Overall Status", "value": "FAIL", "status": "FAIL"},
        {"label": "Condition 2a 5%", "value": "FAIL", "status": "FAIL"},
        {"label": "Condition 2b", "value": "N/A", "status": "N/A", "reason": "N/A (not evaluated)"},
        {"label": "Sum Large Securities %", "value": "50.37%"},
    ], details=[{"label": "Total Assets", "value": "8,338,122.00"},
                {"label": "Large Securities", "value": "(<META>, 10.56%), (AMZN, 10.40%)"}],
        notes=["No shares outstanding data (left out of the ownership test): GIB"])
    forty = _rule("40 Act Diversification", "40 Act Diversification", "PASS", [
        {"label": "Overall Status", "value": "PASS", "status": "PASS"},
        {"label": "Registered Status", "value": "Non-Diversified"}])
    gics = _rule("GICS Diversification", "GICS Concentration", "PASS", [
        {"label": "Overall GICS Compliance", "value": "PASS", "status": "PASS"}],
        tables=[{"title": "Industry exposure", "columns": ["Category", "Fund Weight"],
                 "rows": [["Software", "29.88%"], ["Banks", "5.00%"]], "total": 12}])
    summary = _rule("Summary Metrics", "summary_metrics", None, [{"label": "Total Assets", "value": "3,596,378,773"}])
    compliance = {
        "KNG": {"summary_metrics": summary, "gics_compliance": gics, "diversification_40act_check": forty,
                "diversification_IRS_check": irs_kng},
        "TDVI": {"summary_metrics": summary, "gics_compliance": gics, "diversification_40act_check": forty,
                 "diversification_IRS_check": dict(irs_kng, status="PASS", rows=[
                     {"label": "Overall Status", "value": "PASS", "status": "PASS"}])},
    }
    return {"schema_version": 1, "days": {DAY: {
        "kind": "eod", "date": DAY, "funds": ["KNG", "TDVI"], "fund_errors": {"TDVI": ["TDVI: price feed was late"]},
        "rules": {"summary_metrics": {"title": "Summary Metrics", "label": "summary_metrics", "footnotes": []},
                  "gics_compliance": {"title": "GICS Diversification", "label": "GICS Concentration", "footnotes": []},
                  "diversification_40act_check": {"title": "40 Act Diversification", "label": "40 Act Diversification",
                                                  "footnotes": ["Condition 40 Act 1: 75% of the fund diversified."]},
                  "diversification_IRS_check": {"title": "IRS Diversification", "label": "IRS Diversification",
                                                "footnotes": ["IRS Condition 1: 90% of income qualifying."]}},
        "compliance": compliance,
        "compliance_summary": {
            "KNG": {"status": "FAIL", "counts": {"PASS": 2, "FAIL": 1, "N/A": 0, "ERROR": 0},
                    "failed": ["diversification_IRS_check"]},
            "TDVI": {"status": "PASS", "counts": {"PASS": 3, "FAIL": 0, "N/A": 0, "ERROR": 0}, "failed": []}},
        "nav": {"KNG": {"status": "FAIL", "summary": [{"label": "NAV Diff (2-dec)", "value": "-0.19", "status": "FAIL"}],
                        "rows": [{"label": "Expected TNA", "value": "$3,417,433,904"}],
                        "sections": [{"title": "Gain/Loss Components",
                                      "rows": [{"label": "Equity G/L", "value": "$14,886,379.72"}]}],
                        "checks": [{"label": "NAV Good 2-Dec", "status": "FAIL"}],
                        "gl": [{"label": "G/L Today", "value": "11,615,167"}], "tables": []}},
        "holdings": {"KNG": {"status": "FAIL", "holdings_breaks": 80, "price_breaks": 1,
                             "counts": [{"label": "Equity Holdings", "value": 80, "kind": "holdings"},
                                        {"label": "Equity Price", "value": 1, "kind": "price"}],
                             "sections": [{"key": "custodian_equity", "title": "Custodian Equity", "tables": [
                                 {"title": "Holdings Discrepancies", "columns": ["eqy_ticker", "discrepancy_type"],
                                  "rows": [["XOM", "Quantity Mismatch"]], "total": 80}]}]}},
    }}}


def trading_views() -> dict:
    irs = _rule("IRS Diversification", "IRS Diversification", "FAIL",
                [{"label": "Overall Status", "value": "FAIL", "status": "FAIL"}])
    before = dict(irs, status="PASS", rows=[{"label": "Overall Status", "value": "PASS", "status": "PASS"}])
    return {"schema_version": 1, "days": {DAY: {
        "kind": "trading", "date": DAY, "funds": ["DOGG"], "fund_errors": {},
        "rules": {"diversification_IRS_check": {"title": "IRS Diversification", "label": "IRS Diversification",
                                                "footnotes": []}},
        "compliance": {"DOGG": {"diversification_IRS_check": irs}},
        "compliance_ex_ante": {"DOGG": {"diversification_IRS_check": before}},
        "compliance_summary": {"DOGG": {"status": "FAIL", "counts": {"PASS": 0, "FAIL": 1, "N/A": 0, "ERROR": 0},
                                        "failed": ["diversification_IRS_check"]}},
        "trading": {
            "date": DAY, "summary": {"total_funds_analyzed": 1, "total_funds_traded": 1},
            "overview": {"columns": ["Fund Name", "Status Change", "Violations Before", "Violations After", "Net Change"],
                         "rows": [{"fund": "DOGG", "status_change": "OUT_OF_COMPLIANCE", "violations_before": 0,
                                   "violations_after": 1, "net_change": 1, "highlight": "out"}],
                         "totals": {"violations_before": 0, "violations_after": 1, "net_change": 1}},
            "checks": {"DOGG": [{"key": "diversification_IRS_check", "label": "IRS Diversification",
                                 "before": "PASS", "after": "FAIL", "changed": True}]},
            "activity": {"columns": ["Fund", "Asset Class", "Trade Value"],
                         "rows": [{"fund": "DOGG", "asset_class": "Equity",
                                   "cells": [{"label": "Trade Value", "value": "7,240,344.52"}]}]},
            "activity_details": {"DOGG": [{"asset_class": "Equity", "direction": "Sell", "ticker": "MRK",
                                           "quantity": "3,440.00", "market_value": "514,348.80"}]}},
    }}}
