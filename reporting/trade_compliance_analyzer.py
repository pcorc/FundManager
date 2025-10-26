"""Analytics for comparing ex-ante and ex-post compliance results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass
class FundComplianceComparison:
    """Summary of compliance changes for a single fund."""

    fund_name: str
    trade_info: Dict[str, float]
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
        traded_funds_info: Mapping[str, Mapping[str, float]],
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
            "total_funds_traded": len(self.traded_funds_info),
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

            trade_total = comparison["trade_info"].get("total_traded", 0.0)
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
        trade_info = dict(self.traded_funds_info.get(fund_name, {}))

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
        }

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
