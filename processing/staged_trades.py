"""Which funds have staged (proposed, not yet executed) trades on a date: a light, read-only DB query.

A staged trade is a Vest holdings row of the ex-post analysis with a non-zero ``trade_rebal`` -- the
rows the trading analysis values. One grouped query per Vest view (equity, option, treasury), no
holdings are loaded::

    staged_trade_counts(session, date(2026, 10, 30))
    -> {"DOGG": {"equity": 20, "options": 20, "treasury": 1, "total": 41}, ...}

Counts are rows as the views return them (duplicate view rows count twice) and the option count
includes FLEX option rows, which the trading analysis does not value.
"""
from __future__ import annotations

from datetime import date
from typing import Dict

import pandas as pd
from sqlalchemy import text

__all__ = ["STAGED_TRADE_VIEWS", "staged_trade_counts"]

# asset class -> Vest view (the bulk loader's trading sources)
STAGED_TRADE_VIEWS = (("equity", "v_vest_equity"), ("options", "v_vest_option"), ("treasury", "v_vest_treasury"))


def staged_trade_counts(session, day: date, *, analysis_type: str = "ex_post") -> Dict[str, Dict[str, int]]:
    """``{fund_ticker: {"equity", "options", "treasury", "total": rows}}`` for funds with staged trades on ``day``."""
    counts: Dict[str, Dict[str, int]] = {}
    for asset_class, view in STAGED_TRADE_VIEWS:
        sql = text(
            "SELECT fund_ticker, SUM(CASE WHEN trade_rebal <> 0 THEN 1 ELSE 0 END) AS staged "
            f"FROM reconciliation.{view} "
            "WHERE `date` = :day AND LOWER(TRIM(analysis_type)) = :analysis_type GROUP BY fund_ticker"
        )
        frame = pd.read_sql(sql, session.bind, params={"day": day.isoformat(), "analysis_type": analysis_type.lower()})
        for fund, staged in zip(frame["fund_ticker"], pd.to_numeric(frame["staged"], errors="coerce").fillna(0)):
            if staged > 0 and fund:
                entry = counts.setdefault(str(fund), {"equity": 0, "options": 0, "treasury": 0, "total": 0})
                entry[asset_class] += int(staged)
                entry["total"] += int(staged)
    return dict(sorted(counts.items()))
