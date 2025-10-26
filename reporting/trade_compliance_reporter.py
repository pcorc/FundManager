"""Orchestration utilities for trading compliance reporting."""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Optional

from reporting.trade_compliance_analyzer import TradingComplianceAnalyzer
from reporting.trade_compliance_reporting import (
    GeneratedTradingComplianceReport,
    generate_trading_compliance_reports,
)


def build_trading_compliance_reports(
    results_ex_ante: Mapping[str, Any],
    results_ex_post: Mapping[str, Any],
    report_date: date | datetime | str,
    output_dir: str,
    *,
    traded_funds_info: Mapping[str, Mapping[str, float]] | None = None,
    create_pdf: bool = True,
) -> Optional[GeneratedTradingComplianceReport]:
    """Generate trading compliance comparison reports."""

    if not (results_ex_ante or results_ex_post):
        return None

    analyzer = TradingComplianceAnalyzer(
        results_ex_ante=results_ex_ante,
        results_ex_post=results_ex_post,
        date=report_date,
        traded_funds_info=traded_funds_info or {},
    )
    comparison_data = analyzer.analyze()

    return generate_trading_compliance_reports(
        comparison_data,
        report_date,
        output_dir,
        create_pdf=create_pdf,
    )
