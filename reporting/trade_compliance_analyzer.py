"""Analytics for comparing ex-ante and ex-post compliance results."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple
import numpy as np
import pandas as pd
from processing.fund import Fund
from reporting.report_utils import non_empty
from utilities.numbers import num

ASSET_ORDER: Tuple[str, ...] = ("equity", "options", "treasury")

_TICKER_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "equity": ("ticker", "equity_ticker", "eqyticker", "symbol", "underlying_symbol"),
    "options": ("ticker", "optticker", "occ_symbol", "option_ticker"),
    "treasury": ("cusip", "ticker", "security_id", "bond_ticker"),
}


def _bool_status(value: Any) -> Optional[str]:
    # numpy.bool_ too: checks comparing numpy floats return it.
    if isinstance(value, (bool, np.bool_)):
        return "PASS" if value else "FAIL"
    return None


def _str_status(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.upper() in {"PASS", "FAIL"}:
        return value.upper()
    return None


class TradingComplianceAnalyzer:
    """Compare ex-ante and ex-post compliance outcomes for traded funds."""

    def __init__(
        self,
        results_ex_ante: Mapping[str, Any],
        results_ex_post: Mapping[str, Any],
        date: Any,
        traded_funds_info: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.results_ex_ante = results_ex_ante or {}
        self.results_ex_post = results_ex_post or {}
        self.date = str(date)
        self.traded_funds_info = {
            fund: dict(info) for fund, info in (traded_funds_info or {}).items()
        }

    # ------------------------------------------------------------------
    def analyze(self) -> Dict[str, Any]:
        """Return a serialisable comparison of compliance changes."""

        fund_names = sorted(
            set(self.results_ex_ante) | set(self.results_ex_post) | set(self.traded_funds_info)
        )

        fund_details: Dict[str, Dict[str, Any]] = {}
        total_funds_traded = 0
        for fund_name in fund_names:
            comparison = self._compare_fund(fund_name)
            fund_details[fund_name] = comparison
            if num(comparison["trade_info"], "total_traded"):
                total_funds_traded += 1

        return {
            "date": self.date,
            "summary": {
                "total_funds_analyzed": len(fund_names),
                "total_funds_traded": total_funds_traded,
            },
            "funds": fund_details,
        }

    # ------------------------------------------------------------------
    def _compare_fund(self, fund_name: str) -> Dict[str, Any]:
        # A fund can be on one side only (e.g. it failed to process in the other pass).
        ante_results = self.results_ex_ante.get(fund_name) or {}
        post_results = self.results_ex_post.get(fund_name) or {}
        trade_info = self._build_trade_info(fund_name, ante_results, post_results)

        checks = {}
        violations_before = 0
        violations_after = 0

        # summary_metrics (always computed) and fund_object are not checks, on either side.
        check_names = sorted(
            (set(ante_results.keys()) | set(post_results.keys())) - {"summary_metrics", "fund_object"}
        )

        for check in check_names:
            ante_status = self._extract_status(ante_results.get(check))
            post_status = self._extract_status(post_results.get(check))
            if ante_status == post_status == "UNKNOWN":
                continue

            if ante_status == "FAIL":
                violations_before += 1
            if post_status == "FAIL":
                violations_after += 1

            checks[check] = {
                "status_before": ante_status,
                "status_after": post_status,
                "violations_before": 1 if ante_status == "FAIL" else 0,
                "violations_after": 1 if post_status == "FAIL" else 0,
                "changed": ante_status != post_status,
            }

        overall_before = self._overall_status(ante_results)
        overall_after = self._overall_status(post_results)

        status_change = "UNCHANGED"
        if overall_before != overall_after:
            if overall_after == "PASS":
                status_change = "INTO_COMPLIANCE"
            elif overall_after == "FAIL":
                status_change = "OUT_OF_COMPLIANCE"

        return {
            "fund_name": fund_name,
            "trade_info": trade_info,
            "status_change": status_change,
            "violations_before": violations_before,
            "violations_after": violations_after,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    def _build_trade_info(
        self,
        fund_name: str,
        ante_results: Mapping[str, Any],
        post_results: Mapping[str, Any],
    ) -> Dict[str, Any]:
        base_info = dict(self.traded_funds_info.get(fund_name, {}))

        # base_info carries the ex-post Fund and a copy of the Fund taken after the ex-ante pass
        # (both passes populate fund.data on the same Fund object).
        fund = self._resolve_fund_object(base_info, post_results, ante_results)
        ante_fund = base_info.get("ex_ante_fund_object")
        if not isinstance(ante_fund, Fund):
            ante_fund = fund  # no ex-ante copy given: both sides read the ex-post holdings

        for key, value in self._compute_trade_summary(fund, ante_fund).items():
            if key not in base_info or base_info[key] in (None, "", {}, [], ()):  # type: ignore[arg-type]
                base_info[key] = value

        return base_info

    def _compute_trade_summary(self, fund: Optional[Fund], ante_fund: Optional[Fund] = None) -> Dict[str, Any]:
        # Trading analysis runs intraday — the custodian hasn't reported T's TNA/TA yet,
        # so fund.data.current.{tna,ta} would be zero. Use the previous (T-1) snapshot
        # the bulk loader populated with the prior-day custodian numbers, falling back
        # to current only if T-1 is missing (e.g., a Monday after a holiday).
        # The ex-ante pass loads previous = T-1; the ex-post pass loads previous = the custodian
        # previous date (T-2), so its snapshot is only a fallback.
        data = getattr(fund, "data", None)
        previous_snapshot = getattr(data, "previous", None)
        current_snapshot = getattr(data, "current", None)
        ante_previous_snapshot = getattr(getattr(ante_fund, "data", None), "previous", None)

        def _from_snapshot(snapshot: Any, attr: str) -> float:
            try:
                return float(getattr(snapshot, attr, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        total_net_assets, total_assets = (
            _from_snapshot(ante_previous_snapshot, attr)
            or _from_snapshot(previous_snapshot, attr)
            or _from_snapshot(current_snapshot, attr)
            for attr in ("tna", "ta")
        )

        net_trades: Dict[str, Dict[str, float]] = {}
        trade_activity: Dict[str, Dict[str, Any]] = {}
        asset_trade_summary: Dict[str, Dict[str, float]] = {}
        total_traded = 0.0

        for key in ASSET_ORDER:
            post_df = self._get_holdings(fund, key)
            ante_df = self._get_holdings(ante_fund if ante_fund is not None else fund, key)

            post_quantities = self._resolve_quantity_series(post_df, preferred="iiv_shares")
            ante_quantities = self._resolve_quantity_series(ante_df, preferred="nav_shares")

            # Splice closed positions back in so sell-to-close shows up (in trades only, not in market values).
            traded_df = post_df
            if key == "options" and non_empty(ante_df) and "optticker" in ante_df.columns:
                post_tickers = post_df["optticker"] if "optticker" in post_df.columns else []
                ante_only = ante_df[~ante_df["optticker"].isin(post_tickers)].copy()
                # Options expiring on or before the analysis date drop out by expiry, not by a trade.
                ante_only = ante_only.loc[~self._expires_by(ante_only, self.date)]
                if not ante_only.empty:
                    ante_only["iiv_shares"] = 0.0
                    ante_only["trade_rebal"] = -pd.to_numeric(
                        ante_only.get("nav_shares", 0.0), errors="coerce"
                    ).fillna(0.0)
                    traded_df = pd.concat([post_df, ante_only], ignore_index=True)

            trade_rebal_series = self._extract_trade_series(traded_df)
            quantity_delta = float(trade_rebal_series.sum())
            if quantity_delta == 0.0:
                quantity_delta = float(post_quantities.sum()) - float(ante_quantities.sum())

            buy_quantity = float(trade_rebal_series[trade_rebal_series > 0.0].sum())
            sell_quantity = abs(float(trade_rebal_series[trade_rebal_series < 0.0].sum()))
            if buy_quantity == 0.0 and sell_quantity == 0.0:
                if quantity_delta >= 0.0:
                    buy_quantity = quantity_delta
                else:
                    sell_quantity = abs(quantity_delta)

            post_notional = self._calculate_notional(post_df, post_quantities, asset_type=key)
            ante_notional = self._calculate_notional(ante_df, ante_quantities, asset_type=key)
            # Trade-value fallback without trade_rebal values: the ex-post holdings' share change
            # (nav_shares -> iiv_shares) at ex-post prices, so ex-ante -> ex-post price moves are not trades.
            post_start_notional = self._calculate_notional(
                post_df, self._resolve_quantity_series(post_df, preferred="nav_shares"), asset_type=key
            )

            activity_details = self._calculate_trade_activity(traded_df, asset_type=key)
            if activity_details["buys"] or activity_details["sells"]:
                trade_activity[key] = activity_details

            buy_value = num(activity_details["net"], "buy_value")
            sell_value = num(activity_details["net"], "sell_value")
            net_trades[key] = {
                "buy_quantity": buy_quantity,
                "sell_quantity": sell_quantity,
                "net_quantity": quantity_delta,
                "buy_value": buy_value,
                "sell_value": sell_value,
                "net_value": buy_value - sell_value,
            }

            trade_total = buy_value + sell_value
            if trade_total <= 0.0:
                trade_total = abs(post_notional - post_start_notional)
            total_traded += trade_total

            asset_trade_summary[key] = {
                "trade_value": trade_total,
                "pct_of_tna": self._safe_percent(trade_total, total_net_assets),
                "pct_of_total_assets": self._safe_percent(trade_total, total_assets),
                "ex_ante_market_value": ante_notional,
                "ex_post_market_value": post_notional,
                "market_value_delta": post_notional - ante_notional,
                "trade_vs_ex_ante_pct": self._safe_percent(trade_total, ante_notional),
                "ex_post_vs_ex_ante_pct": self._safe_percent(post_notional, ante_notional),
            }

        return {
            "net_trades": net_trades,
            "total_traded": total_traded,
            "trade_activity": trade_activity,
            "asset_trade_summary": asset_trade_summary,
            "total_net_assets": total_net_assets,
            "total_assets": total_assets,
        }

    @staticmethod
    def _get_holdings(fund: Optional[Fund], attribute: str) -> pd.DataFrame:
        snapshot = getattr(getattr(fund, "data", None), "current", None)
        for source in ("vest", "custodian"):
            holdings = getattr(getattr(snapshot, source, None), attribute, None)
            if non_empty(holdings):
                return holdings.copy()
        return pd.DataFrame()

    @staticmethod
    def _resolve_fund_object(*sources: Mapping[str, Any]) -> Optional[Fund]:
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            for key in ("fund", "fund_object"):
                candidate = source.get(key)
                if isinstance(candidate, Fund):
                    return candidate
        return None

    @staticmethod
    def _expires_by(df: pd.DataFrame, as_of: Any) -> pd.Series:
        """True for rows whose maturity (vest ``maturity`` / custodian ``maturity_date``) is on or before ``as_of``;
        rows without a parsable maturity are False."""
        cutoff = pd.to_datetime(as_of, errors="coerce")
        for column in ("maturity", "maturity_date"):
            if column in df.columns and not pd.isna(cutoff):
                return (pd.to_datetime(df[column], errors="coerce") <= cutoff).fillna(False)
        return pd.Series(False, index=df.index, dtype=bool)

    @staticmethod
    def _extract_trade_series(df: pd.DataFrame) -> pd.Series:
        if not non_empty(df):
            return pd.Series(dtype=float)
        if "trade_rebal" not in df.columns:
            return pd.Series(0.0, index=df.index, dtype=float)
        return pd.to_numeric(df["trade_rebal"], errors="coerce").fillna(0.0)

    @staticmethod
    def _calculate_notional(df: pd.DataFrame, quantities: pd.Series, *, asset_type: str) -> float:
        if not non_empty(df) or "price" not in df.columns or quantities.empty:
            return 0.0
        prices = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        if asset_type == "treasury":
            prices = prices / 100.0  # quoted per 100 face, as in the holdings' treasury_market_value

        # First existing multiplier column (not first positive value, unlike per-trade estimates).
        multiplier_series = pd.Series(1.0, index=df.index, dtype=float)
        if asset_type == "options":
            multiplier_series = pd.Series(100.0, index=df.index, dtype=float)
            for column in ("contract_multiplier", "multiplier"):
                if column in df.columns:
                    multiplier_series = pd.to_numeric(df[column], errors="coerce").fillna(100.0)
                    break

        try:
            return float((quantities * prices * multiplier_series).sum())
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _resolve_quantity_series(df: pd.DataFrame, *, preferred: Optional[str] = None) -> pd.Series:
        if not non_empty(df):
            return pd.Series(dtype=float)

        column = preferred or "nav_shares"
        fallbacks = ("nav_shares", "iiv_shares", "shares", "quantity", "units")
        for candidate in [column] + [name for name in fallbacks if name != column]:
            if candidate in df.columns:
                return pd.to_numeric(df[candidate], errors="coerce").fillna(0.0)

        return pd.Series(0.0, index=df.index, dtype=float)

    def _calculate_trade_activity(self, holdings: pd.DataFrame, *, asset_type: str) -> Dict[str, Any]:
        empty_result = {
            "buys": [],
            "sells": [],
            "net": {"buy_quantity": 0.0, "sell_quantity": 0.0, "buy_value": 0.0, "sell_value": 0.0},
        }

        if not non_empty(holdings):
            return empty_result

        trade_series = self._extract_trade_series(holdings)
        if trade_series.empty or (trade_series == 0.0).all():
            return empty_result

        df = holdings.copy()
        df["trade_rebal"] = trade_series
        df = df.loc[df["trade_rebal"] != 0.0]

        if df.empty:
            return empty_result

        buys = []
        sells = []
        total_buy_qty = 0.0
        total_sell_qty = 0.0
        total_buy_value = 0.0
        total_sell_value = 0.0

        ticker_column = self._detect_ticker_column(df, asset_type) or "ticker"

        for _, row in df.iterrows():
            quantity = float(row.get("trade_rebal", 0.0))
            if quantity == 0.0:
                continue

            ticker = ""
            if ticker_column in row:
                ticker_value = row.get(ticker_column)
                if pd.notna(ticker_value):
                    ticker = str(ticker_value).upper().strip()

            market_value = self._estimate_trade_market_value(row, quantity, asset_type)
            price, multiplier = self._trade_price(row, asset_type)
            # price / multiplier feed the in-app trade blotter; the reports read ticker, quantity, market_value.
            trade_payload = {
                "ticker": ticker,
                "quantity": abs(quantity),
                "market_value": abs(market_value),
                "price": price,
                "multiplier": multiplier,
            }

            if quantity > 0:
                total_buy_qty += abs(quantity)
                total_buy_value += abs(market_value)
                buys.append(trade_payload)
            else:
                total_sell_qty += abs(quantity)
                total_sell_value += abs(market_value)
                sells.append(trade_payload)

        return {
            "buys": buys,
            "sells": sells,
            "net": {
                "buy_quantity": total_buy_qty,
                "sell_quantity": total_sell_qty,
                "buy_value": total_buy_value,
                "sell_value": total_sell_value,
            },
        }

    @staticmethod
    def _trade_price(row: pd.Series, asset_type: str) -> Tuple[Optional[float], float]:
        """(quoted price, multiplier) of one trade row; the price is None unless it is positive.

        Options default to a multiplier of 100 (the first positive multiplier column wins); treasury
        prices are quoted per 100 face.
        """
        multiplier = 1.0
        if asset_type == "options":
            multiplier = 100.0
            for column in ("contract_multiplier", "multiplier"):
                value = row.get(column)
                if pd.notna(value):
                    try:
                        parsed = float(value)
                    except (TypeError, ValueError):
                        continue
                    if parsed > 0:
                        multiplier = parsed
                        break

        price_value = 0.0
        if "price" in row and pd.notna(row.get("price")):
            try:
                price_value = float(row.get("price"))
            except (TypeError, ValueError):
                price_value = 0.0
        return (price_value if price_value > 0 else None), multiplier

    @classmethod
    def _estimate_trade_market_value(cls, row: pd.Series, quantity: float, asset_type: str) -> float:
        """Market value of one trade: price x quantity x multiplier (treasury prices per 100 face), or
        abs(qty x multiplier) without a price."""
        price_value, multiplier = cls._trade_price(row, asset_type)
        if price_value is not None:
            if asset_type == "treasury":
                price_value /= 100.0  # quoted per 100 face
            return abs(price_value * quantity * multiplier)
        return abs(quantity * multiplier)

    @staticmethod
    def _detect_ticker_column(df: pd.DataFrame, asset_type: str) -> Optional[str]:
        for column in _TICKER_COLUMNS.get(asset_type, ("ticker",)):
            if column in df.columns and df[column].notna().any():
                return column
        if "ticker" in df.columns and df["ticker"].notna().any():
            return "ticker"
        return None

    # ------------------------------------------------------------------
    def _overall_status(self, payload: Mapping[str, Any]) -> str:
        failures = sum(
            1 for name, result in payload.items()
            if name != "summary_metrics" and self._extract_status(result) == "FAIL"
        )
        if failures:
            return "FAIL"
        return "PASS" if payload else "UNKNOWN"

    @staticmethod
    def _extract_status(result: Any) -> str:
        mapping: Optional[Mapping[str, Any]] = result if isinstance(result, Mapping) else None
        if mapping is None:
            # Objects: is_compliant must be an exact bool, status a PASS/FAIL string.
            status = _bool_status(getattr(result, "is_compliant", None)) or _str_status(
                getattr(result, "status", None)
            )
            if status:
                return status
            details = getattr(result, "details", None)
            if not isinstance(details, Mapping):
                return _str_status(result) or "UNKNOWN"
            mapping = details

        is_compliant = mapping.get("is_compliant")
        return (
            _bool_status(is_compliant)
            or _str_status(is_compliant)
            or _str_status(mapping.get("status"))
            or "UNKNOWN"
        )

    @staticmethod
    def _safe_percent(value: float, denominator: float) -> float:
        try:
            if denominator in (0, 0.0, None):
                return 0.0
            return float(value) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
