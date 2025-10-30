"""Analytics for comparing ex-ante and ex-post compliance results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd


@dataclass
class FundComplianceComparison:
    """Summary of compliance changes for a single fund."""

    fund_name: str
    trade_info: Dict[str, Any]
    overall_before: str
    overall_after: str
    status_change: str
    violations_before: int
    violations_after: int
    num_changes: int
    checks: Dict[str, Dict[str, Any]]


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

        summary = {
            "total_funds_analyzed": len(fund_names),
            "total_funds_traded": 0,
            "funds_with_compliance_changes": 0,
            "funds_into_compliance": 0,
            "funds_out_of_compliance": 0,
            "funds_unchanged": 0,
            "total_checks_changed": 0,
            "total_violations_before": 0,
            "total_violations_after": 0,
            "total_traded_notional": 0.0,
        }

        fund_details: Dict[str, Dict[str, Any]] = {}

        for fund_name in fund_names:
            comparison = self._compare_fund(fund_name)
            fund_details[fund_name] = comparison

            if comparison["num_changes"]:
                summary["funds_with_compliance_changes"] += 1
            if comparison["status_change"] == "INTO_COMPLIANCE":
                summary["funds_into_compliance"] += 1
            elif comparison["status_change"] == "OUT_OF_COMPLIANCE":
                summary["funds_out_of_compliance"] += 1
            else:
                summary["funds_unchanged"] += 1

            summary["total_checks_changed"] += comparison["num_changes"]
            summary["total_violations_before"] += comparison["violations_before"]
            summary["total_violations_after"] += comparison["violations_after"]

            trade_total = float(comparison["trade_info"].get("total_traded", 0.0) or 0.0)
            if trade_total:
                summary["total_funds_traded"] += 1
            summary["total_traded_notional"] += trade_total

        return {
            "date": self.date,
            "summary": summary,
            "funds": fund_details,
        }

    # ------------------------------------------------------------------
    def _compare_fund(self, fund_name: str) -> Dict[str, Any]:
        ante_results = self._as_mapping(self.results_ex_ante.get(fund_name))
        post_results = self._as_mapping(self.results_ex_post.get(fund_name))
        trade_info = self._build_trade_info(fund_name, ante_results, post_results)

        checks = {}
        violations_before = 0
        violations_after = 0
        num_changes = 0

        check_names = sorted(
            set(ante_results.keys()) | set(post_results.keys()) - {"summary_metrics", "fund_object"}
        )

        for check in check_names:
            ante_status = self._extract_status(ante_results.get(check))
            post_status = self._extract_status(post_results.get(check))
            changed = ante_status != post_status

            if ante_status == "FAIL":
                violations_before += 1
            if post_status == "FAIL":
                violations_after += 1
            if changed:
                num_changes += 1

            checks[check] = {
                "status_before": ante_status,
                "status_after": post_status,
                "violations_before": 1 if ante_status == "FAIL" else 0,
                "violations_after": 1 if post_status == "FAIL" else 0,
                "changed": changed,
            }

        overall_before = self._overall_status(ante_results)
        overall_after = self._overall_status(post_results)

        status_change = "UNCHANGED"
        if overall_before != overall_after:
            if overall_after == "PASS":
                status_change = "INTO_COMPLIANCE"
            elif overall_after == "FAIL":
                status_change = "OUT_OF_COMPLIANCE"

        summary_metrics = {
            "ex_ante": self._extract_summary_metrics_payload(
                ante_results.get("summary_metrics")
            ),
            "ex_post": self._extract_summary_metrics_payload(
                post_results.get("summary_metrics")
            ),
        }
        return {
            "fund_name": fund_name,
            "trade_info": trade_info,
            "overall_before": overall_before,
            "overall_after": overall_after,
            "status_change": status_change,
            "violations_before": violations_before,
            "violations_after": violations_after,
            "num_changes": num_changes,
            "checks": checks,
            "summary_metrics": summary_metrics,
        }

    # ------------------------------------------------------------------
    def _build_trade_info(
        self,
        fund_name: str,
        ante_results: Mapping[str, Any],
        post_results: Mapping[str, Any],
    ) -> Dict[str, Any]:
        base_info = dict(self.traded_funds_info.get(fund_name, {}))

        computed_info = self._compute_trade_summary(ante_results, post_results)

        for key, value in computed_info.items():
            if key not in base_info or base_info[key] in (None, ""):
                base_info[key] = value

        return base_info

    def _compute_trade_summary(
        self,
        ante_results: Mapping[str, Any],
        post_results: Mapping[str, Any],
    ) -> Dict[str, Any]:
        post_fund = self._extract_fund_object(post_results)
        ante_fund = self._extract_fund_object(ante_results)

        if post_fund is None and ante_fund is None:
            return {}

        asset_mappings: Tuple[Tuple[str, str], ...] = (
            ("equity", "equity"),
            ("options", "options"),
            ("treasury", "treasury"),
        )

        final_shares: Dict[str, float] = {}
        net_trades: Dict[str, Dict[str, float]] = {}
        notional_changes: Dict[str, float] = {}
        trade_activity: Dict[str, Dict[str, Any]] = {}
        asset_trade_totals: Dict[str, float] = {}
        total_traded = 0.0

        for key, attr in asset_mappings:
            post_df = self._extract_holdings(post_fund, attr)
            ante_df = self._extract_holdings(ante_fund, attr)

            final_quantity = self._sum_quantity(post_df)
            initial_quantity = self._sum_quantity(ante_df)
            quantity_delta = final_quantity - initial_quantity

            final_shares[key] = final_quantity
            net_trades[key] = {
                "buys": max(quantity_delta, 0.0),
                "sells": abs(min(quantity_delta, 0.0)),
                "net": quantity_delta,
            }

            post_notional = self._sum_notional(post_df)
            ante_notional = self._sum_notional(ante_df)
            notional_delta = post_notional - ante_notional
            notional_changes[key] = notional_delta

            activity_details = self._calculate_trade_activity(post_df, asset_type=key)
            if activity_details["buys"] or activity_details["sells"]:
                trade_activity[key] = activity_details

            buy_value = activity_details["net"].get("buy_value", 0.0)
            sell_value = activity_details["net"].get("sell_value", 0.0)
            net_trades[key].update(
                {
                    "buy_value": buy_value,
                    "sell_value": sell_value,
                    "net_value": buy_value - sell_value,
                }
            )

            trade_total = buy_value + sell_value
            if trade_total <= 0.0:
                trade_total = abs(notional_delta)
            asset_trade_totals[key] = trade_total
            total_traded += trade_total

        # Provide convenience aliases for downstream reporting
        trade_summary: Dict[str, Any] = {
            "final_shares": final_shares,
            "net_trades": net_trades,
            "total_traded": total_traded,
            "trade_activity": trade_activity,
            "notional_changes": notional_changes,
        }

        trade_summary.update(asset_trade_totals)

        return trade_summary

    def _extract_fund_object(self, result: Mapping[str, Any]) -> Optional[Any]:
        fund_object = result.get("fund_object") if isinstance(result, Mapping) else None
        return fund_object if fund_object is not None else None

    def _extract_holdings(self, fund_object: Any, attribute: str) -> pd.DataFrame:
        if fund_object is None:
            return pd.DataFrame()

        data = getattr(fund_object, "data", None)
        current = getattr(data, "current", None)
        holdings = getattr(current, attribute, None)
        if isinstance(holdings, pd.DataFrame):
            return holdings
        return pd.DataFrame()

    @staticmethod
    def _sum_quantity(df: pd.DataFrame, candidates: Iterable[str] = ("quantity", "shares", "units")) -> float:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return 0.0
        for column in candidates:
            if column in df.columns:
                series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
                return float(series.sum())
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            series = pd.to_numeric(df[numeric_cols[0]], errors="coerce").fillna(0.0)
            return float(series.sum())
        return 0.0

    @staticmethod
    def _sum_notional(
        df: pd.DataFrame,
        *,
        quantity_cols: Iterable[str] = ("quantity", "shares", "units"),
        price_cols: Iterable[str] = ("price", "px_last", "close_price"),
    ) -> float:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return 0.0

        if "market_value" in df.columns:
            series = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)
            return float(series.sum())

        quantity_series: Optional[pd.Series] = None
        price_series: Optional[pd.Series] = None

        for column in quantity_cols:
            if column in df.columns:
                quantity_series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
                break

        for column in price_cols:
            if column in df.columns:
                price_series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
                break

        if quantity_series is not None and price_series is not None:
            return float((quantity_series * price_series).sum())

        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            series = pd.to_numeric(df[numeric_cols[0]], errors="coerce").fillna(0.0)
            return float(series.sum())
        return 0.0

    # ------------------------------------------------------------------
    def _extract_summary_metrics_payload(self, result: Any) -> Dict[str, float]:
        calculations: Mapping[str, Any]
        if isinstance(result, Mapping):
            calculations = result
        else:
            calculations = getattr(result, "calculations", {}) or {}

        metrics: Dict[str, float] = {}
        for key, value in calculations.items():
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        return metrics

    def _calculate_trade_activity(
        self,
        holdings: pd.DataFrame,
        *,
        asset_type: str,
    ) -> Dict[str, Any]:
        if not isinstance(holdings, pd.DataFrame) or holdings.empty:
            return {
                "buys": [],
                "sells": [],
                "net": {
                    "buy_quantity": 0.0,
                    "sell_quantity": 0.0,
                    "buy_value": 0.0,
                    "sell_value": 0.0,
                },
            }

        if "trade_rebal" not in holdings.columns:
            return {
                "buys": [],
                "sells": [],
                "net": {
                    "buy_quantity": 0.0,
                    "sell_quantity": 0.0,
                    "buy_value": 0.0,
                    "sell_value": 0.0,
                },
            }

        df = holdings.copy()
        df["trade_rebal"] = pd.to_numeric(df["trade_rebal"], errors="coerce").fillna(0.0)
        df = df.loc[df["trade_rebal"] != 0.0]

        if df.empty:
            return {
                "buys": [],
                "sells": [],
                "net": {
                    "buy_quantity": 0.0,
                    "sell_quantity": 0.0,
                    "buy_value": 0.0,
                    "sell_value": 0.0,
                },
            }

        ticker_column = self._detect_ticker_column(df, asset_type)
        price_column = self._detect_price_column(df)

        buys = []
        sells = []
        total_buy_qty = 0.0
        total_sell_qty = 0.0
        total_buy_value = 0.0
        total_sell_value = 0.0

        for _, row in df.iterrows():
            quantity = float(row.get("trade_rebal", 0.0))
            if quantity == 0.0:
                continue

            ticker = str(row.get(ticker_column, "")) if ticker_column else ""
            ticker = ticker.upper().strip()

            market_value = self._estimate_trade_market_value(row, quantity, price_column, asset_type)

            trade_payload = {
                "ticker": ticker,
                "quantity": abs(quantity),
                "market_value": abs(market_value),
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
    def _detect_ticker_column(df: pd.DataFrame, asset_type: str) -> Optional[str]:
        candidates = {
            "equity": (
                "equity_ticker",
                "ticker",
                "symbol",
                "underlying_symbol",
            ),
            "options": ("optticker", "occ_symbol", "ticker"),
            "treasury": (
                "cusip",
                "ticker",
                "security_id",
            ),
        }

        for column in candidates.get(asset_type, ("ticker",)):
            if column in df.columns:
                return column
        return None

    @staticmethod
    def _detect_price_column(df: pd.DataFrame) -> Optional[str]:
        for column in ("trade_price", "price", "px_last", "close_price"):
            if column in df.columns:
                return column
        return None

    def _estimate_trade_market_value(
        self,
        row: pd.Series,
        quantity: float,
        price_column: Optional[str],
        asset_type: str,
    ) -> float:
        preferred_columns = (
            "trade_rebal_market_value",
            "trade_rebal_mv",
            "trade_rebal_value",
            "trade_rebal_notional",
        )
        for column in preferred_columns:
            if column in row and pd.notna(row[column]):
                try:
                    return float(row[column])
                except (TypeError, ValueError):
                    continue

        if price_column and price_column in row and pd.notna(row[price_column]):
            try:
                price_value = float(row[price_column])
                multiplier = 100.0 if asset_type == "options" else 1.0
                return price_value * quantity * multiplier
            except (TypeError, ValueError):
                pass

        if "market_value" in row and pd.notna(row["market_value"]):
            try:
                base_quantity = float(row.get("quantity", 0.0))
                market_value = float(row["market_value"])
                if base_quantity:
                    proportion = quantity / base_quantity
                    return market_value * proportion
                return market_value
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        return float(quantity)

    # ------------------------------------------------------------------
    def _overall_status(self, payload: Mapping[str, Any]) -> str:
        failures = 0
        for name, result in payload.items():
            if name == "summary_metrics":
                continue
            status = self._extract_status(result)
            if status == "FAIL":
                failures += 1
        if failures:
            return "FAIL"
        return "PASS" if payload else "UNKNOWN"

    # ------------------------------------------------------------------
    def _overall_status(self, payload: Mapping[str, Any]) -> str:
        failures = 0
        for name, result in payload.items():
            if name == "summary_metrics":
                continue
            status = self._extract_status(result)
            if status == "FAIL":
                failures += 1
        if failures:
            return "FAIL"
        return "PASS" if payload else "UNKNOWN"

    def _extract_status(self, result: Any) -> str:
        if not isinstance(result, Mapping):
            if isinstance(result, str):
                value = result.upper()
                if value in {"PASS", "FAIL"}:
                    return value
            return "UNKNOWN"

        status = result.get("is_compliant")
        if isinstance(status, bool):
            return "PASS" if status else "FAIL"
        if isinstance(status, str):
            upper = status.upper()
            if upper in {"PASS", "FAIL"}:
                return upper
        if "status" in result and isinstance(result["status"], str):
            upper = result["status"].upper()
            if upper in {"PASS", "FAIL"}:
                return upper
        return "UNKNOWN"

    @staticmethod
    def _as_mapping(value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}
