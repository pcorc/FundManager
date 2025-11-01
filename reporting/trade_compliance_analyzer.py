"""Analytics for comparing ex-ante and ex-post compliance results."""

from __future__ import annotations
from collections import defaultdict

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

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
        summary_metric_totals = {
            "ex_ante": defaultdict(float),
            "ex_post": defaultdict(float),
        }

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

            fund_summary_metrics = comparison.get("summary_metrics", {}) or {}
            for phase in ("ex_ante", "ex_post"):
                metrics = fund_summary_metrics.get(phase, {}) or {}
                aggregates = summary_metric_totals[phase]
                for key, value in metrics.items():
                    try:
                        aggregates[key] += float(value)
                    except (TypeError, ValueError):
                        continue

            trade_total = float(comparison["trade_info"].get("total_traded", 0.0) or 0.0)
            if trade_total:
                summary["total_funds_traded"] += 1
            summary["total_traded_notional"] += trade_total

        return {
            "date": self.date,
            "summary": {
                **summary,
                "summary_metrics_totals": {
                    "ex_ante": dict(summary_metric_totals["ex_ante"]),
                    "ex_post": dict(summary_metric_totals["ex_post"]),
                },
            },
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

        asset_mappings: Tuple[Tuple[str, str], ...] = (
            ("equity", "equity"),
            ("options", "options"),
            ("treasury", "treasury"),
        )

        final_shares: Dict[str, float] = {}
        initial_shares: Dict[str, float] = {}
        net_trades: Dict[str, Dict[str, float]] = {}
        notional_changes: Dict[str, float] = {}
        ex_ante_market_values: Dict[str, float] = {}
        ex_post_market_values: Dict[str, float] = {}
        trade_activity: Dict[str, Dict[str, Any]] = {}
        asset_trade_totals: Dict[str, float] = {}
        total_traded = 0.0

        for key, attr in asset_mappings:
            post_df = self._get_holdings(post_results, attr)
            ante_df = self._get_holdings(ante_results, attr)

            final_quantity = post_df["iiv_shares"].sum()
            initial_quantity = ante_df["nav_shares"].sum()

            quantity_delta = post_df["trade_rebal"].sum()
            if quantity_delta == 0.0:
                quantity_delta = final_quantity - initial_quantity

            final_shares[key] = final_quantity
            initial_shares[key] = initial_quantity

            trade_rebal_series = post_df["trade_rebal"]
            buy_quantity = float(trade_rebal_series[trade_rebal_series > 0.0].sum())
            sell_quantity = abs(
                float(trade_rebal_series[trade_rebal_series < 0.0].sum())
            )
            if buy_quantity == 0.0 and sell_quantity == 0.0:
                if quantity_delta >= 0.0:
                    buy_quantity = quantity_delta
                else:
                    sell_quantity = abs(quantity_delta)

            net_trades[key] = {
                "buys": buy_quantity,
                "sells": sell_quantity,
                "net": quantity_delta,
            }

            post_notional = self._calculate_notional(post_df, "iiv_shares", asset_type=key)
            ante_notional = self._calculate_notional(ante_df, "nav_shares", asset_type=key)
            notional_delta = post_notional - ante_notional
            notional_changes[key] = notional_delta

            ex_ante_market_values[key] = ante_notional
            ex_post_market_values[key] = post_notional

            activity_details = self._calculate_trade_activity(post_df, asset_type=key)
            if activity_details["buys"] or activity_details["sells"]:
                trade_activity[key] = activity_details
            else:
                net_payload = activity_details.get("net", {}) if activity_details else {}
                net_values = [
                    float(net_payload.get(field, 0.0) or 0.0)
                    for field in ("buy_quantity", "sell_quantity", "buy_value", "sell_value")
                ]
                if any(net_values):
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
        ante_metrics = self._extract_summary_metrics_payload(
            ante_results.get("summary_metrics")
        )
        post_metrics = self._extract_summary_metrics_payload(
            post_results.get("summary_metrics")
        )

        total_net_assets = float(
            post_metrics.get("total_net_assets")
            or ante_metrics.get("total_net_assets")
            or 0.0
        )
        total_assets = float(
            post_metrics.get("total_assets") or ante_metrics.get("total_assets") or 0.0
        )

        asset_trade_summary: Dict[str, Dict[str, float]] = {}

        for asset_key, trade_total in asset_trade_totals.items():
            ante_notional = ex_ante_market_values.get(asset_key, 0.0)
            post_notional = ex_post_market_values.get(asset_key, 0.0)

            asset_trade_summary[asset_key] = {
                "trade_value": trade_total,
                "pct_of_tna": self._safe_percent(trade_total, total_net_assets),
                "pct_of_total_assets": self._safe_percent(trade_total, total_assets),
                "ex_ante_market_value": ante_notional,
                "ex_post_market_value": post_notional,
                "market_value_delta": post_notional - ante_notional,
                "trade_vs_ex_ante_pct": self._safe_percent(trade_total, ante_notional),
                "ex_post_vs_ex_ante_pct": self._safe_percent(post_notional, ante_notional),
            }

        # Provide convenience aliases for downstream reporting
        trade_summary: Dict[str, Any] = {
            "final_shares": final_shares,
            "initial_shares": initial_shares,
            "net_trades": net_trades,
            "total_traded": total_traded,
            "trade_activity": trade_activity,
            "notional_changes": notional_changes,
            "asset_trade_totals": asset_trade_totals,
            "asset_trade_summary": asset_trade_summary,
            "ex_ante_market_values": ex_ante_market_values,
            "ex_post_market_values": ex_post_market_values,
            "total_net_assets": total_net_assets,
            "total_assets": total_assets,
        }

        return trade_summary

    def _get_holdings(self, results: Mapping[str, Any], attribute: str) -> pd.DataFrame:
        if not isinstance(results, Mapping):
            return pd.DataFrame()
        candidate = results.get(attribute)
        if isinstance(candidate, pd.DataFrame):
            return candidate
        return pd.DataFrame()

    def _calculate_notional(
        self,
        df: pd.DataFrame,
        quantity_column: str,
        *,
        asset_type: str,
    ) -> float:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return 0.0
        if quantity_column not in df.columns or "price" not in df.columns:
            return 0.0
        quantities = df[quantity_column]
        prices = df["price"]
        multiplier = 100.0 if asset_type == "options" else 1.0
        return float((quantities * prices * multiplier).sum())

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

            ticker = str(row.get("ticker", ""))
            ticker = ticker.upper().strip()

            market_value = self._estimate_trade_market_value(row, quantity, asset_type)

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

    def _estimate_trade_market_value(
        self,
        row: pd.Series,
        quantity: float,
        asset_type: str,
    ) -> float:
        multiplier = 100.0 if asset_type == "options" else 1.0
        if "price" in row and pd.notna(row["price"]):
            try:
                price_value = float(row["price"])
                return price_value * quantity * multiplier
            except (TypeError, ValueError):
                pass
        return float(quantity * multiplier)


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
        mapping: Optional[Mapping[str, Any]] = None
        if isinstance(result, Mapping):
            mapping = result
        else:
            attr_status = getattr(result, "is_compliant", None)
            if isinstance(attr_status, bool):
                return "PASS" if attr_status else "FAIL"
            # Some result objects expose a ``status`` attribute or nested dict
            attr_status_str = getattr(result, "status", None)
            if isinstance(attr_status_str, str):
                upper = attr_status_str.upper()
                if upper in {"PASS", "FAIL"}:
                    return upper
            details = getattr(result, "details", None)
            if isinstance(details, Mapping):
                mapping = details
        if mapping is None:
            if isinstance(result, str):
                value = result.upper()
                if value in {"PASS", "FAIL"}:
                    return value
            return "UNKNOWN"

        status = mapping.get("is_compliant")
        if isinstance(status, bool):
            return "PASS" if status else "FAIL"
        if isinstance(status, str):
            upper = status.upper()
            if upper in {"PASS", "FAIL"}:
                return upper
        status_value = mapping.get("status")
        if isinstance(status_value, str):
            upper = status_value.upper()
            if upper in {"PASS", "FAIL"}:
                return upper
        return "UNKNOWN"

    @staticmethod
    def _as_mapping(value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    @staticmethod
    def _safe_percent(value: float, denominator: float) -> float:
        try:
            if denominator in (0, 0.0, None):
                return 0.0
            return float(value) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0