"""Gain/loss utilities shared across NAV and reconciliation workflows.

This module operates purely on pandas :class:`~pandas.DataFrame`
structures so that both the NAV reconciliation service and any ad-hoc
analytics can share the same calculation logic. The functions lean on
column-name heuristics because our upstream custodians provide slightly
different schemas. Wherever possible we normalise numeric data and
surface the intermediate details for downstream reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import pandas as pd

from domain.fund import GainLossResult, Fund


@dataclass(frozen=True)
class _ColumnHeuristics:
    identifiers: Tuple[str, ...]
    quantity: Tuple[str, ...]
    price: Tuple[str, ...]
    value: Tuple[str, ...]


class GainLossCalculator:
    """Static helpers that consolidate gain/loss calculations."""

    _EQUITY_HINTS = _ColumnHeuristics(
        identifiers=("equity_ticker", "ticker", "cusip", "symbol"),
        quantity=("quantity", "shares", "position", "share_quantity"),
        price=("price", "close", "unit_price", "market_price"),
        value=("market_value", "marketvalue", "value", "mv"),
    )
    _OPTION_HINTS = _ColumnHeuristics(
        identifiers=("option_symbol", "option_ticker", "ticker", "id"),
        quantity=("quantity", "contracts", "position"),
        price=("price", "mark", "unit_price", "option_price"),
        value=("market_value", "marketvalue", "value", "delta_adjusted_notional"),
    )
    _TREASURY_HINTS = _ColumnHeuristics(
        identifiers=("cusip", "ticker", "security_id"),
        quantity=("quantity", "par", "face", "position"),
        price=("price", "clean_price", "unit_price"),
        value=("market_value", "marketvalue", "value", "accrued_interest"),
    )

    _ASSET_TO_HINTS = {
        "equity": _EQUITY_HINTS,
        "equities": _EQUITY_HINTS,
        "options": _OPTION_HINTS,
        "flex_options": _OPTION_HINTS,
        "treasury": _TREASURY_HINTS,
        "treasuries": _TREASURY_HINTS,
    }

    @classmethod
    def calculate_for_asset(cls, fund: Fund, asset_class: str) -> GainLossResult:
        """Return gain/loss for the requested asset class."""

        asset_key = asset_class.lower()
        hints = cls._ASSET_TO_HINTS.get(asset_key)
        if hints is None:
            return GainLossResult()

        current_snapshot = getattr(fund.data.current, cls._snapshot_attribute(asset_key), pd.DataFrame())
        previous_snapshot = getattr(fund.data.previous, cls._snapshot_attribute(asset_key), pd.DataFrame())

        return cls._calculate_from_frames(current_snapshot, previous_snapshot, hints)

    @staticmethod
    def _snapshot_attribute(asset_key: str) -> str:
        if asset_key in {"equity", "equities"}:
            return "equity"
        if asset_key in {"treasury", "treasuries"}:
            return "treasury"
        return "options"

    @classmethod
    def _calculate_from_frames(
        cls,
        current: Optional[pd.DataFrame],
        previous: Optional[pd.DataFrame],
        hints: _ColumnHeuristics,
    ) -> GainLossResult:
        if current is None or current.empty:
            return GainLossResult()

        current_df = current.copy()
        previous_df = previous.copy() if isinstance(previous, pd.DataFrame) else pd.DataFrame()

        identifier = cls._find_shared_column(current_df, previous_df, hints.identifiers)
        if identifier is None:
            # Without a shared identifier we cannot compare security level data
            return GainLossResult()

        if previous_df.empty or identifier not in previous_df.columns:
            merged = current_df.add_suffix("_current")
            merged[identifier] = current_df[identifier].values
        else:
            merged = pd.merge(
                current_df,
                previous_df,
                on=identifier,
                how="left",
                suffixes=("_current", "_prior"),
            )

        current_value_col, prior_value_col = cls._resolve_value_columns(merged, hints)
        if current_value_col is None:
            return GainLossResult()

        merged["value_current"] = cls._coerce_numeric(merged[current_value_col])
        merged["value_prior"] = cls._coerce_numeric(merged[prior_value_col]) if prior_value_col else 0.0
        merged["gain_loss"] = merged["value_current"] - merged["value_prior"]

        result = GainLossResult(
            raw_gl=float(merged["gain_loss"].sum()),
            adjusted_gl=float(merged["gain_loss"].sum()),
            details=merged[[identifier, "value_current", "value_prior", "gain_loss"]],
        )
        return result

    @staticmethod
    def _find_shared_column(
        current: pd.DataFrame,
        previous: pd.DataFrame,
        candidates: Iterable[str],
    ) -> Optional[str]:
        current_lower = {col.lower(): col for col in current.columns}
        previous_lower = {col.lower(): col for col in previous.columns}

        for candidate in candidates:
            col_current = current_lower.get(candidate)
            if col_current is None:
                continue
            if previous.empty:
                return col_current
            if candidate in previous_lower:
                return col_current
        return None

    @classmethod
    def _resolve_value_columns(
        cls,
        merged: pd.DataFrame,
        hints: _ColumnHeuristics,
    ) -> Tuple[Optional[str], Optional[str]]:
        value_current = cls._find_column(merged.columns, hints.value, suffix="_current")
        value_prior = cls._find_column(merged.columns, hints.value, suffix="_prior")

        if value_current:
            return value_current, value_prior

        # Fall back to price * quantity when explicit market value is missing
        price_current = cls._find_column(merged.columns, hints.price, suffix="_current")
        price_prior = cls._find_column(merged.columns, hints.price, suffix="_prior")
        quantity_current = cls._find_column(merged.columns, hints.quantity, suffix="_current")
        quantity_prior = cls._find_column(merged.columns, hints.quantity, suffix="_prior")

        if price_current and quantity_current:
            merged["derived_value_current"] = (
                cls._coerce_numeric(merged[price_current]) * cls._coerce_numeric(merged[quantity_current])
            )
            derived_current = "derived_value_current"
        else:
            derived_current = None

        if price_prior and quantity_prior:
            merged["derived_value_prior"] = (
                cls._coerce_numeric(merged[price_prior]) * cls._coerce_numeric(merged[quantity_prior])
            )
            derived_prior = "derived_value_prior"
        else:
            derived_prior = None

        return derived_current, derived_prior

    @staticmethod
    def _find_column(columns: Iterable[str], candidates: Iterable[str], suffix: str) -> Optional[str]:
        columns_list = list(columns)
        lowered = [col.lower() for col in columns_list]
        suffix_lower = suffix.lower()

        for candidate in candidates:
            candidate_lower = candidate.lower()
            target = f"{candidate_lower}{suffix_lower}" if suffix else candidate_lower

            # Exact match first (common when pandas appends suffixes)
            for idx, col_lower in enumerate(lowered):
                if col_lower == target:
                    return columns_list[idx]

            # Next try "contains" match so columns like market_value_current_x work
            if suffix_lower:
                for idx, col_lower in enumerate(lowered):
                    if candidate_lower in col_lower and suffix_lower in col_lower:
                        return columns_list[idx]

            # Finally allow unsuffixed columns when prior data was missing
            for idx, col_lower in enumerate(lowered):
                if col_lower == candidate_lower:
                    return columns_list[idx]

        return None

    @staticmethod
    def _coerce_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").fillna(0.0)