"""Execution helpers for FundManager run modes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

from config.fund_registry import FundRegistry
from processing.bulk_data_loader import BulkDataLoader
from processing.cli import EODRunParameters, TradingComplianceParameters
from processing.fund_manager import FundManager, ProcessingResults
from reporting.compliance_reporter import build_compliance_reports
from reporting.nav_recon_reporter import build_reconciliation_reports
from reporting.trade_compliance_reporter import build_trading_compliance_reports


@dataclass
class EODOutcome:
    results: ProcessingResults
    artefacts: Mapping[str, object]
    generated_paths: Dict[str, str]


@dataclass
class TradingOutcome:
    results_ex_ante: ProcessingResults
    results_ex_post: ProcessingResults
    artefacts: object
    generated_paths: Dict[str, str]


class RunExecutor:
    """Orchestrates FundManager EOD and trading-compliance runs."""

    def __init__(
        self,
        session,
        base_cls,
        registry: FundRegistry,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.session = session
        self.base_cls = base_cls
        self.registry = registry
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        self._loader = BulkDataLoader(session, base_cls, registry)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_eod(self, params: EODRunParameters) -> EODOutcome:
        results = self._process_funds(
            target_date=params.trade_date,
            operations=params.operations,
            analysis_type="eod",
        )

        compliance_payload = self._extract_payload(results, "compliance_results")
        reconciliation_payload = self._extract_payload(results, "reconciliation_results")
        nav_payload = self._extract_payload(results, "nav_results")

        artefacts: MutableMapping[str, object] = {}

        if "compliance" in params.operations and compliance_payload:
            artefacts["compliance"] = build_compliance_reports(
                results,
                report_date=params.trade_date,
                output_dir=str(self.output_dir),
                create_pdf=params.create_pdf,
            )

        if any(name in params.operations for name in ("reconciliation", "nav_reconciliation")):
            artefacts["reconciliation"] = build_reconciliation_reports(
                holdings_results=reconciliation_payload if "reconciliation" in params.operations else None,
                nav_results=nav_payload if "nav_reconciliation" in params.operations else None,
                report_date=params.trade_date,
                output_dir=str(self.output_dir),
                create_pdf=params.create_pdf,
                compliance_results=compliance_payload if "compliance" in params.operations else None,
            )

        generated_paths = _flatten_eod_paths(artefacts)
        return EODOutcome(results=results, artefacts=artefacts, generated_paths=generated_paths)

    def run_trading_compliance(self, params: TradingComplianceParameters) -> TradingOutcome:
        ex_ante_results = self._process_funds(
            target_date=params.ex_ante_date,
            operations=["compliance"],
            analysis_type="ex_ante",
        )
        ex_post_results = self._process_funds(
            target_date=params.ex_post_date,
            operations=["compliance"],
            analysis_type="ex_post",
        )

        ante_payload = self._extract_payload(ex_ante_results, "compliance_results")
        post_payload = self._extract_payload(ex_post_results, "compliance_results")

        if not ante_payload and not post_payload:
            raise RuntimeError("Trading compliance run produced no compliance results to compare")

        artefacts = build_trading_compliance_reports(
            results_ex_ante=ante_payload,
            results_ex_post=post_payload,
            report_date=params.ex_post_date,
            output_dir=str(self.output_dir),
            create_pdf=params.create_pdf,
        )

        generated_paths = _flatten_trading_paths(artefacts)
        return TradingOutcome(
            results_ex_ante=ex_ante_results,
            results_ex_post=ex_post_results,
            artefacts=artefacts,
            generated_paths=generated_paths,
        )

    def log_processing_summary(self, label: str, summary: Mapping[str, object]) -> None:
        pretty = ", ".join(f"{key}={value}" for key, value in summary.items())
        self.logger.info("Processing summary (%s): %s", label, pretty)

    def log_generated_paths(self, paths: Mapping[str, str]) -> None:
        if not paths:
            self.logger.warning("No report artefacts were produced")
            return
        for label, path in sorted(paths.items()):
            self.logger.info("Generated %s -> %s", label, path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _process_funds(
        self,
        *,
        target_date,
        operations: Sequence[str],
        analysis_type: str,
    ) -> ProcessingResults:
        data_store = self._loader.load_all_data_for_date(target_date)
        manager = FundManager(self.registry, data_store, analysis_type=analysis_type)
        return manager.run_daily_operations(list(operations))

    @staticmethod
    def _extract_payload(results: ProcessingResults, attribute: str) -> Dict[str, Mapping[str, object]]:
        payload: Dict[str, Mapping[str, object]] = {}
        for fund_name, fund_result in results.fund_results.items():
            value = getattr(fund_result, attribute, None)
            if value:
                payload[fund_name] = value
        return payload


def filter_registry(registry: FundRegistry, funds: Sequence[str]) -> FundRegistry:
    """Return a registry containing only the requested funds (if any)."""

    if not funds:
        return registry

    missing = [fund for fund in funds if fund not in registry.funds]
    if missing:
        raise ValueError(f"Requested funds not in registry: {', '.join(sorted(missing))}")

    filtered = FundRegistry()
    filtered.funds = {fund: registry.funds[fund] for fund in funds}
    return filtered


def _flatten_eod_paths(artefacts: Mapping[str, object]) -> Dict[str, str]:
    paths: Dict[str, str] = {}

    compliance = artefacts.get("compliance")
    if compliance is not None:
        excel_path = getattr(compliance, "excel_path", None)
        pdf_path = getattr(compliance, "pdf_path", None)
        if excel_path:
            paths["compliance_excel"] = excel_path
        if pdf_path:
            paths["compliance_pdf"] = pdf_path

    reconciliation = artefacts.get("reconciliation")
    if reconciliation is not None:
        holdings = getattr(reconciliation, "holdings", None)
        if holdings is not None:
            if getattr(holdings, "excel_path", None):
                paths["holdings_reconciliation_excel"] = holdings.excel_path  # type: ignore[attr-defined]
            if getattr(holdings, "pdf_path", None):
                paths["holdings_reconciliation_pdf"] = holdings.pdf_path  # type: ignore[attr-defined]

        nav = getattr(reconciliation, "nav", None)
        if nav is not None:
            if getattr(nav, "excel_path", None):
                paths["nav_reconciliation_excel"] = nav.excel_path  # type: ignore[attr-defined]
            if getattr(nav, "pdf_path", None):
                paths["nav_reconciliation_pdf"] = nav.pdf_path  # type: ignore[attr-defined]

        combined = getattr(reconciliation, "combined_reconciliation_pdf", None)
        if combined:
            paths["reconciliation_summary_pdf"] = combined

        full = getattr(reconciliation, "full_summary_pdf", None)
        if full:
            paths["oversight_summary_pdf"] = full

    return paths


def _flatten_trading_paths(artefacts) -> Dict[str, str]:  # type: ignore[no-untyped-def]
    paths: Dict[str, str] = {}
    if artefacts is None:
        return paths

    report = getattr(artefacts, "report", None)
    if report is not None:
        if getattr(report, "excel_path", None):
            paths["trading_compliance_excel"] = report.excel_path  # type: ignore[attr-defined]
        if getattr(report, "pdf_path", None):
            paths["trading_compliance_pdf"] = report.pdf_path  # type: ignore[attr-defined]

    return paths