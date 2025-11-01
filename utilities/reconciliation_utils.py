"""Utilities for reconciliation reporting pipelines."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def split_flex_price_frames(
    df: pd.DataFrame, flag_column: str = "is_flex"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return FLEX and non-FLEX price rows from ``df``.

    Parameters
    ----------
    df:
        Source frame containing price discrepancies.
    flag_column:
        Column indicating whether a row represents a FLEX option. Defaults to
        ``"is_flex"`` which is what the reconciliation payloads emit today.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Two copies of ``df`` filtered to FLEX and non-FLEX rows respectively.
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    flag_series = df.get(flag_column)
    if flag_series is None:
        return pd.DataFrame(), df.copy()

    mask = flag_series.fillna(False).astype(bool)
    flex = df.loc[mask].copy()
    standard = df.loc[~mask].copy()
    return flex, standard