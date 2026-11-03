"""Per-fund orchestration of compliance, recon, and NAV-recon services."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from services.compliance_checker import ComplianceChecker
from services.nav_reconciliator import NAVReconciliator
from services.reconciliator import Reconciliator

from processing.fund import Fund, FundData, FundSnapshot, FundHoldings, FundRegistry
from processing.bulk_data_loader import BulkDataStore
from utilities.workdays import business_day_offset

_progress_logger = logging.getLogger("fund_manager.progress")


@dataclass(frozen=True)
class ProgressEvent:
    """One progress step. ``stage``: day, db, load, process, fund, operation, reports, email or done."""

    stage: str
    day: Optional[str] = None
    fund: Optional[str] = None
    index: Optional[int] = None
    total: Optional[int] = None
    message: str = ""


ProgressCallback = Callable[[ProgressEvent], None]


def emit_progress(on_progress: Optional[ProgressCallback], stage: str, **fields: Any) -> None:
    """Send one event; a failing callback is logged at debug level and never breaks the run."""
    if on_progress is None:
        return
    event = ProgressEvent(stage, **fields)
    try:
        on_progress(event)
    except Exception:
        _progress_logger.debug("Progress callback failed for %s", event, exc_info=True)


def progress_for_day(on_progress: Optional[ProgressCallback], day: str) -> Optional[ProgressCallback]:
    """Wrap ``on_progress`` so events that carry no day are stamped with ``day``."""
    if on_progress is None:
        return None
    return lambda event: on_progress(event if event.day else replace(event, day=day))


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


_NAV_FIELDS = {
    "nav": ["nav_per_share", "nav", "net_asset_value"],
    "ta": ["total_assets", "gross_assets", "gross_value"],
    "tna": ["total_net_assets", "tna", "net_assets", "nav_total"],
    "expenses": ["expense_amount", "expenses"],
    "shares_outstanding": ["shares_outstanding", "shares", "total_shares"],
}


class FundManager:
    """Runs the daily operations across every fund in a registry."""

    def __init__(self, fund_registry: FundRegistry, data_store: BulkDataStore, analysis_type: str = "eod",
                 *, on_progress: Optional[ProgressCallback] = None):
        self.fund_registry = fund_registry
        self.on_progress = on_progress
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

        total = len(self.available_funds)
        for index, fund_name in enumerate(self.available_funds, 1):
            self.logger.info("[%s] Processing %s...", self.analysis_type, fund_name)
            progress = dict(fund=fund_name, index=index, total=total)
            emit_progress(self.on_progress, "fund", message=self.analysis_type, **progress)
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
                    emit_progress(self.on_progress, "operation", message="compliance", **progress)
                    fund_result.compliance_results = self._run_compliance(fund, tests)
                if "reconciliation" in operations:
                    emit_progress(self.on_progress, "operation", message="reconciliation", **progress)
                    fund_result.reconciliation_results = self._run_reconciliation(fund)
                if "nav_reconciliation" in operations:
                    emit_progress(self.on_progress, "operation", message="nav_reconciliation", **progress)
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
        checker = ComplianceChecker(funds={fund.name: fund}, analysis_type=self.analysis_type)
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
        d = self.data_store.fund_data.get(fund.name, {})
        flex_pattern = fund.flex_option_pattern if fund.has_flex_option else None
        options = {
            key: _split_options(d.get(key, pd.DataFrame()), flex_pattern)
            for key in ("vest_option", "vest_option_t1", "custodian_option", "custodian_option_t1")
        }

        def snapshot(sfx: str) -> FundSnapshot:
            holdings = {}
            for src in ("vest", "custodian"):
                opts, flex = options[f"{src}_option{sfx}"]
                holdings[src] = FundHoldings(
                    equity=d.get(f"{src}_equity{sfx}", pd.DataFrame()),
                    options=opts,
                    flex_options=flex,
                    treasury=d.get(f"{src}_treasury{sfx}", pd.DataFrame()),
                )
            nav = d.get(f"nav{sfx}", pd.DataFrame())
            return FundSnapshot(
                **holdings,
                index=FundHoldings(equity=d.get(f"index{sfx}", pd.DataFrame())),
                overlap=d.get(f"overlap{sfx}", pd.DataFrame()),
                cash=_sum_numeric(d.get(f"cash{sfx}", pd.DataFrame()), ["cash_value"]),
                **{name: _first_numeric(nav, columns) for name, columns in _NAV_FIELDS.items()},
                equity_trades=d.get(f"equity_trades{sfx}", pd.DataFrame()),
                option_trades=d.get(f"option_trades{sfx}", pd.DataFrame()),
                flex_option_trades=d.get(f"flex_option_trades{sfx}", pd.DataFrame()),
                cr_rd_data=None if sfx else d.get("cr_rd", pd.DataFrame()),
                fund_name=fund.name,
            )

        fund.data = FundData(current=snapshot(""), previous=snapshot("_t1"))
        fund.data.price_breaks = d.get("holdings_price_breaks", {}) or {}
        fund.data.distributions = d.get("distributions")
        fund.data.flows = d.get("flows_t1")
        fund.data.other = d.get("other", 0.0)

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
        return business_day_offset(current, -1)
