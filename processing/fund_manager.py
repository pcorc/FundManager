"""Per-fund orchestration of compliance, recon, and NAV-recon services."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from services.compliance_checker import ComplianceChecker
from services.nav_reconciliator import NAVReconciliator
from services.reconciliator import Reconciliator

from processing.fund import Fund, FundData, FundSnapshot, FundHoldings, FundRegistry
from processing.bulk_data_loader import BulkDataStore


@dataclass
class FundResult:
    fund_name: str
    compliance_results: Dict[str, Any] = None
    reconciliation_results: Dict[str, Any] = None
    nav_results: Dict[str, Any] = None
    errors: List[str] = field(default_factory=list)
    fund: Optional[Fund] = None


@dataclass
class ProcessingResults:
    fund_results: Dict[str, FundResult]
    summary: Dict[str, Any]


def _split_options(
    options_df: pd.DataFrame, pattern: Optional[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pattern is None or options_df.empty or "optticker" not in options_df.columns:
        return options_df, pd.DataFrame()
    mask = options_df["optticker"].str.contains(pattern, na=False, regex=True)
    return options_df[~mask].copy(), options_df[mask].copy()


def _first_numeric(df: pd.DataFrame, columns: Sequence[str]) -> float:
    if df.empty:
        return 0.0
    for col in columns:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[0])
    return 0.0


def _sum_numeric(df: pd.DataFrame, columns: Sequence[str]) -> float:
    if df.empty:
        return 0.0
    for col in columns:
        if col in df.columns:
            return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
    return 0.0


class FundManager:
    """Runs the daily operations across every fund in a registry."""

    def __init__(self, fund_registry: FundRegistry, data_store: BulkDataStore, analysis_type: str = "eod"):
        self.fund_registry = fund_registry
        self.data_store = data_store
        self.analysis_type = analysis_type
        self.logger = logging.getLogger(__name__)
        self.available_funds = list(data_store.loaded_funds)

    def run_daily_operations(
        self, operations: List[str], *, compliance_tests: Optional[Sequence[str]] = None
    ) -> ProcessingResults:
        results: Dict[str, FundResult] = {}
        summary: Dict[str, Any] = {
            "requested_operations": operations,
            "processed_funds": 0,
            "funds_with_errors": 0,
            "errors": [],
        }
        tests = [t for t in (compliance_tests or []) if t]

        self.logger.info(
            "Starting %s processing for %d funds", self.analysis_type, len(self.available_funds)
        )

        for fund_name in self.available_funds:
            self.logger.info("[%s] Processing %s...", self.analysis_type, fund_name)
            fund_result = FundResult(fund_name=fund_name)

            try:
                fund = self.fund_registry.get(fund_name)
                if fund is None:
                    fund_result.errors.append("Fund not found in registry")
                    results[fund_name] = fund_result
                    continue

                self._populate_fund_data(fund)
                fund_result.fund = fund

                if "compliance" in operations:
                    fund_result.compliance_results = self._run_compliance(fund, tests)
                if "reconciliation" in operations:
                    fund_result.reconciliation_results = self._run_reconciliation(fund)
                if "nav_reconciliation" in operations:
                    fund_result.nav_results = self._run_nav_reconciliation(fund)

                summary["processed_funds"] += 1

            except Exception as exc:
                error_msg = f"Error processing {fund_name}: {exc}"
                fund_result.errors.append(error_msg)
                self.logger.exception(error_msg)
                summary["errors"].append(error_msg)
                summary["funds_with_errors"] += 1

            results[fund_name] = fund_result

        summary["total_funds"] = len(results)
        return ProcessingResults(fund_results=results, summary=summary)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def _run_compliance(self, fund: Fund, tests: List[str]) -> Dict[str, Any]:
        checker = ComplianceChecker(
            getattr(self.data_store, "session", None),
            funds={fund.name: fund},
            date=getattr(self.data_store, "date", None),
            base_cls=getattr(self.data_store, "base_cls", None),
            analysis_type=self.analysis_type,
        )
        checker_results = checker.run_compliance_tests(test_functions=tests or None)
        fund_results = dict(checker_results.get(fund.name, {}))
        fund_results["fund_object"] = fund
        return fund_results

    def _run_nav_reconciliation(self, fund: Fund) -> Dict[str, Any]:
        prior_date = self._prior_business_day(self.data_store.date)
        return NAVReconciliator(
            fund=fund,
            analysis_date=self.data_store.date,
            prior_date=prior_date,
        ).run_nav_reconciliation()

    def _run_reconciliation(self, fund: Fund) -> Dict[str, Any]:
        reconciliator = Reconciliator(fund, self.analysis_type)
        reconciliator.run_all_reconciliations()

        detail_payload: Dict[str, Dict[str, Any]] = {}
        for name, result in reconciliator.results.items():
            sections: Dict[str, Any] = {}
            for attr in (
                "raw_recon",
                "final_recon",
                "price_discrepancies_T",
                "price_discrepancies_T1",
                "merged_data",
                "regular_options",
                "flex_options",
            ):
                value = getattr(result, attr, None)
                if value is not None:
                    sections[attr] = value
            detail_payload[name] = sections

        return {"summary": reconciliator.get_summary(), "details": detail_payload}

    # ------------------------------------------------------------------
    # Populate Fund.data from BulkDataStore
    # ------------------------------------------------------------------
    def _populate_fund_data(self, fund: Fund) -> None:
        fund_data_dict = self.data_store.fund_data.get(fund.name, {})

        flex_pattern: Optional[str] = None
        if fund.has_flex_option:
            flex_pattern = fund.flex_option_pattern

        vest_opts, vest_flex = _split_options(
            fund_data_dict.get("vest_option", pd.DataFrame()), flex_pattern
        )
        vest_opts_t1, vest_flex_t1 = _split_options(
            fund_data_dict.get("vest_option_t1", pd.DataFrame()), flex_pattern
        )
        cust_opts, cust_flex = _split_options(
            fund_data_dict.get("custodian_option", pd.DataFrame()), flex_pattern
        )
        cust_opts_t1, cust_flex_t1 = _split_options(
            fund_data_dict.get("custodian_option_t1", pd.DataFrame()), flex_pattern
        )

        nav_t = fund_data_dict.get("nav", pd.DataFrame())
        nav_t1 = fund_data_dict.get("nav_t1", pd.DataFrame())

        fund.data = FundData(
            current=FundSnapshot(
                vest=FundHoldings(
                    equity=fund_data_dict.get("vest_equity", pd.DataFrame()),
                    options=vest_opts,
                    flex_options=vest_flex,
                    treasury=fund_data_dict.get("vest_treasury", pd.DataFrame()),
                ),
                custodian=FundHoldings(
                    equity=fund_data_dict.get("custodian_equity", pd.DataFrame()),
                    options=cust_opts,
                    flex_options=cust_flex,
                    treasury=fund_data_dict.get("custodian_treasury", pd.DataFrame()),
                ),
                index=FundHoldings(equity=fund_data_dict.get("index", pd.DataFrame())),
                cash=_sum_numeric(fund_data_dict.get("cash", pd.DataFrame()), ["cash_value"]),
                nav=_first_numeric(nav_t, ["nav_per_share", "nav", "net_asset_value"]),
                ta=_first_numeric(nav_t, ["total_assets", "gross_assets", "gross_value"]),
                tna=_first_numeric(nav_t, ["total_net_assets", "tna", "net_assets", "nav_total"]),
                expenses=_first_numeric(nav_t, ["expense_amount", "expenses"]),
                shares_outstanding=_first_numeric(nav_t, ["shares_outstanding", "shares", "total_shares"]),
                flows=_sum_numeric(
                    fund_data_dict.get("flows", pd.DataFrame()),
                    ["net_flows", "flows", "amount", "value"],
                ),
                equity_trades=fund_data_dict.get("equity_trades", pd.DataFrame()),
                option_trades=fund_data_dict.get("option_trades", pd.DataFrame()),
                flex_option_trades=fund_data_dict.get("flex_option_trades", pd.DataFrame()),
                treasury_trades=fund_data_dict.get("treasury_trades", pd.DataFrame()),
                cr_rd_data=fund_data_dict.get("cr_rd", pd.DataFrame()),
                assignments=fund_data_dict.get("assignments", pd.DataFrame()),
                fund_name=fund.name,
            ),
            previous=FundSnapshot(
                vest=FundHoldings(
                    equity=fund_data_dict.get("vest_equity_t1", pd.DataFrame()),
                    options=vest_opts_t1,
                    flex_options=vest_flex_t1,
                    treasury=fund_data_dict.get("vest_treasury_t1", pd.DataFrame()),
                ),
                custodian=FundHoldings(
                    equity=fund_data_dict.get("custodian_equity_t1", pd.DataFrame()),
                    options=cust_opts_t1,
                    flex_options=cust_flex_t1,
                    treasury=fund_data_dict.get("custodian_treasury_t1", pd.DataFrame()),
                ),
                index=FundHoldings(equity=fund_data_dict.get("index", pd.DataFrame())),
                cash=_sum_numeric(fund_data_dict.get("cash_t1", pd.DataFrame()), ["cash_value"]),
                nav=_first_numeric(nav_t1, ["nav_per_share", "nav", "net_asset_value"]),
                ta=_first_numeric(nav_t1, ["total_assets", "gross_assets", "gross_value"]),
                tna=_first_numeric(nav_t1, ["total_net_assets", "tna", "net_assets", "nav_total"]),
                expenses=_first_numeric(nav_t1, ["expense_amount", "expenses"]),
                shares_outstanding=_first_numeric(nav_t1, ["shares_outstanding", "shares", "total_shares"]),
                flows=_sum_numeric(
                    fund_data_dict.get("flows_t1", pd.DataFrame()),
                    ["net_flows", "flows", "amount", "value"],
                ),
                equity_trades=fund_data_dict.get("equity_trades_t1", pd.DataFrame()),
                option_trades=fund_data_dict.get("option_trades_t1", pd.DataFrame()),
                flex_option_trades=fund_data_dict.get("flex_option_trades_t1", pd.DataFrame()),
                treasury_trades=fund_data_dict.get("treasury_trades_t1", pd.DataFrame()),
                assignments=fund_data_dict.get("assignments_t1", pd.DataFrame()),
                fund_name=fund.name,
            ),
        )

        fund.data.price_breaks = fund_data_dict.get("holdings_price_breaks", {}) or {}
        fund.data.assignments = (
            fund_data_dict.get("option_assignments")
            or fund_data_dict.get("assignments")
        )
        fund.data.distributions = fund_data_dict.get("distributions")
        fund.data.flows = fund_data_dict.get("t1_flows") or fund_data_dict.get("flows_t1")
        fund.data.other = fund_data_dict.get("other", 0.0)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    @staticmethod
    def _prior_business_day(current_date) -> date:
        if isinstance(current_date, pd.Timestamp):
            current = current_date.to_pydatetime().date()
        elif isinstance(current_date, datetime):
            current = current_date.date()
        elif isinstance(current_date, date):
            current = current_date
        else:
            current = pd.Timestamp(current_date).date()
        prior = current - timedelta(days=1)
        while prior.weekday() >= 5:
            prior -= timedelta(days=1)
        return prior
