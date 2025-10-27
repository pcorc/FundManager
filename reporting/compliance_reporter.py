
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional

import pandas as pd

from processing.fund_manager import ProcessingResults
from reporting.compliance_reporting import (
    GeneratedComplianceReport,
    generate_compliance_reports,
)


def build_compliance_reports(
    results: ProcessingResults,
    report_date: date | datetime | str,
    output_dir: str,
    *,
    test_functions: Optional[Iterable[str]] = None,
    gics_mapping: Optional[pd.DataFrame] = None,
    create_pdf: bool = True,
) -> Optional[GeneratedComplianceReport]:
    """Generate compliance Excel/PDF artefacts from :class:`ProcessingResults`."""

    compliance_payload = {
        fund_name: fund_result.compliance_results
        for fund_name, fund_result in results.fund_results.items()
        if fund_result.compliance_results
    }

    if not compliance_payload:
        print("*************    NO COMPLIANCE RESULTS   *************")
        return None

    return generate_compliance_reports(
        results=compliance_payload,
        report_date=report_date,
        output_dir=output_dir,
        test_functions=test_functions,
        gics_mapping=gics_mapping,
        create_pdf=create_pdf,
    )