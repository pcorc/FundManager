from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from processing.fund_manager import ProcessingResults
from reporting.compliance_reporting import (
    GeneratedComplianceReport,
    generate_compliance_reports,
)
from reporting.report_utils import normalize_compliance_results, normalize_report_date


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


def build_compliance_reports_for_range(
    dated_results: Mapping[date | datetime | str, ProcessingResults]
    | Sequence[Tuple[date | datetime | str, ProcessingResults]],
    output_dir: str,
    *,
    test_functions: Optional[Iterable[str]] = None,
    gics_mapping: Optional[pd.DataFrame] = None,
    create_pdf: bool = True,
) -> Optional[GeneratedComplianceReport]:
    """Generate a single stacked compliance workbook covering multiple dates."""

    if isinstance(dated_results, Mapping):
        items = dated_results.items()
    else:
        items = list(dated_results)

    normalized_runs: list[Tuple[str, Dict[str, Any]]] = []
    for run_date, results in sorted(
        items,
        key=lambda item: normalize_report_date(item[0]),
    ):
        compliance_payload = {
            fund_name: fund_result.compliance_results
            for fund_name, fund_result in results.fund_results.items()
            if fund_result.compliance_results
        }

        if not compliance_payload:
            continue

        date_iso = normalize_report_date(run_date)
        normalized_runs.append(
            (date_iso, normalize_compliance_results(compliance_payload))
        )

    if not normalized_runs:
        print("*************    NO COMPLIANCE RESULTS   *************")
        return None

    aggregated_results = OrderedDict(normalized_runs)
    first_date = next(iter(aggregated_results))
    last_date = next(reversed(aggregated_results))
    if first_date == last_date:
        prefix = f"compliance_results_{first_date}"
    else:
        prefix = f"compliance_results_{first_date}_to_{last_date}"

    return generate_compliance_reports(
        results=aggregated_results,
        report_date=last_date,
        output_dir=output_dir,
        file_name_prefix=prefix,
        test_functions=test_functions,
        gics_mapping=gics_mapping,
        create_pdf=create_pdf,
    )

def build_compliance_reports_for_range(
    dated_results: Mapping[date | datetime | str, ProcessingResults]
    | Sequence[Tuple[date | datetime | str, ProcessingResults]],
    output_dir: str,
    *,
    test_functions: Optional[Iterable[str]] = None,
    gics_mapping: Optional[pd.DataFrame] = None,
    create_pdf: bool = True,
) -> Optional[GeneratedComplianceReport]:
    """Generate a single stacked compliance workbook covering multiple dates."""

    if isinstance(dated_results, Mapping):
        items = dated_results.items()
    else:
        items = list(dated_results)

    normalized_runs: list[Tuple[str, Dict[str, Any]]] = []
    for run_date, results in sorted(
        items,
        key=lambda item: normalize_report_date(item[0]),
    ):
        compliance_payload = {
            fund_name: fund_result.compliance_results
            for fund_name, fund_result in results.fund_results.items()
            if fund_result.compliance_results
        }

        if not compliance_payload:
            continue

        date_iso = normalize_report_date(run_date)
        normalized_runs.append(
            (date_iso, normalize_compliance_results(compliance_payload))
        )

    if not normalized_runs:
        print("*************    NO COMPLIANCE RESULTS   *************")
        return None

    aggregated_results = OrderedDict(normalized_runs)
    first_date = next(iter(aggregated_results))
    last_date = next(reversed(aggregated_results))
    if first_date == last_date:
        prefix = f"compliance_results_{first_date}"
    else:
        prefix = f"compliance_results_{first_date}_to_{last_date}"

    return generate_compliance_reports(
        results=aggregated_results,
        report_date=last_date,
        output_dir=output_dir,
        file_name_prefix=prefix,
        test_functions=test_functions,
        gics_mapping=gics_mapping,
        create_pdf=create_pdf,
    )