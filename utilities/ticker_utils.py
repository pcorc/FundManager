"""Holdings pass-through.

All identifier normalization now happens in the reconciliation.* SQL views:
  * equity   -> eqyticker resolved per custodian (BNY via bbg SEDOL->TICKER,
               UMB via security_tkr, CCVA via ticker)
  * option   -> optticker emitted canonically by every custodian branch
  * treasury -> cusip emitted by every custodian branch

There is therefore nothing left to normalize in Python. This module keeps a
single pass-through so existing call sites don't need to change.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def normalize_all_holdings(
    fund_name: str,
    fund_data: Dict[str, Any],
    *,
    fund_definition: Optional[Dict[str, Any]] = None,
    logger=None,
) -> Dict[str, Any]:
    """Return fund_data unchanged. Normalization is done in the SQL views."""
    return fund_data