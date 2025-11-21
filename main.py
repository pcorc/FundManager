from __future__ import annotations


import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from collections.abc import Iterable, Mapping as MappingABC, Sequence as SequenceABC
from typing import Mapping, Optional, Sequence

from config.database import initialize_database
from config.fund_registry import FundRegistry
from processing.run_modes import (
    fetch_data_stores,
    flatten_eod_paths,
    flatten_trading_paths,
    plan_eod_requests,
    plan_trading_requests,
    run_eod_mode,
    run_eod_range_mode,
    run_trading_mode,
)
from utilities.cli_options import (
    apply_overrides,
    parse_arguments,
    resolve_eod_parameters,
    resolve_trading_parameters,
)

# Optional runtime overrides for quick local tweaking without CLI arguments.
RUNTIME_OVERRIDES: Optional[Mapping[str, object]] = None


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    overrides: Optional[Mapping[str, object]] = None,
) -> int:
    """Program entry point used by ``python -m`` or direct execution."""

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("fund_manager.main")

    raw_overrides = overrides if overrides is not None else RUNTIME_OVERRIDES
    override_payload = dict(raw_overrides or {})
    multi_day_dates = _extract_override_dates(override_payload)
    if multi_day_dates:
        exit_code = 0
        for run_date in multi_day_dates:
            iter_overrides = dict(override_payload)
            iter_overrides["as_of_date"] = run_date
            result = main(argv=argv, overrides=iter_overrides)
            if result != 0:
                return result
            exit_code = result
        return exit_code

    options = parse_arguments(argv)
    options = apply_overrides(options, override_payload or None)
    output_dir = Path(options.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting %s run for %s (funds: %s)",
        options.analysis_type,
        options.as_of_date.isoformat(),
        ",".join(options.funds) if options.funds else "ALL",
    )

    params = None
    data_stores: Mapping[str, object] = {}

    session, base_cls = initialize_database()
    try:
        registry = FundRegistry.from_database(session, base_cls)
        registry = _filter_registry(registry, options.funds)
        if not registry.funds:
            logger.error("No funds available after applying filters")
            return 1

        if options.compliance_tests:
            logger.info(
                "Limiting compliance checks to: %s",
                ", ".join(options.compliance_tests),
            )
        else:
            logger.info("Compliance checks: all available tests will run")

        if options.analysis_type == "trading_compliance":
            params = resolve_trading_parameters(options)
            requests = plan_trading_requests(params)
        else:
            params = resolve_eod_parameters(options)
            requests = plan_eod_requests(params)

        data_stores = fetch_data_stores(session, base_cls, registry, requests)

    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Processing failed: %s", exc)
        return 1
    finally:
        session.close()

    if options.analysis_type == "trading_compliance":
        assert params is not None
        trading_params = params
        logger.info(
            "Trading compliance dates — ex-ante: %s, ex-post: %s, vest T-1: %s, custodian: %s/%s",
            trading_params.ex_ante_date,
            trading_params.ex_post_date,
            trading_params.vest_previous_date,
            trading_params.custodian_date,
            trading_params.custodian_previous_date,
        )
        results_ex_ante, results_ex_post, artefacts = run_trading_mode(
            registry,
            ex_ante_store=data_stores["ex_ante"],
            ex_post_store=data_stores["ex_post"],
            params=trading_params,
            output_dir=output_dir,
        )

        _log_processing_summary(logger, "ex-ante", results_ex_ante.summary)
        _log_processing_summary(logger, "ex-post", results_ex_post.summary)
        _log_generated_paths(logger, flatten_trading_paths(artefacts))
    else:
        assert params is not None
        eod_params = params
        logger.info(
            "EOD dates — trade date: %s, previous business date: %s",
            eod_params.trade_date,
            eod_params.previous_trade_date,
        )
        processing_results, artefacts = run_eod_mode(
            registry,
            data_store=data_stores["eod"],
            params=eod_params,
            output_dir=output_dir,
        )

        _log_processing_summary(logger, "eod", processing_results.summary)
        _log_generated_paths(logger, flatten_eod_paths(artefacts))

    logger.info("Run completed successfully")
    return 0


def run_time_series(
    *, overrides: Optional[Mapping[str, object]] = None
) -> int:
    """Execute EOD processing for each business day in a range."""

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("fund_manager.main.range")

    payload = dict(overrides or {})
    if not payload:
        raise ValueError("Date range overrides must be provided for run_time_series")

    start_raw = payload.pop("start_date", None)
    end_raw = payload.pop("end_date", None)
    if start_raw is None or end_raw is None:
        raise ValueError("Both 'start_date' and 'end_date' overrides are required")

    start_date = _coerce_date(start_raw, "start_date")
    end_date = _coerce_date(end_raw, "end_date")
    generate_daily_reports = bool(payload.pop("generate_daily_reports", True))

    options = parse_arguments([])
    options = apply_overrides(options, payload)

    if options.analysis_type != "eod":
        raise ValueError("run_time_series only supports analysis_type='eod'")

    output_dir = Path(options.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting time-series EOD run for %s funds between %s and %s",
        ",".join(options.funds) if options.funds else "ALL",
        start_date.isoformat(),
        end_date.isoformat(),
    )

    params = resolve_eod_parameters(options)

    session, base_cls = initialize_database()
    try:
        registry = FundRegistry.from_database(session, base_cls)
        registry = _filter_registry(registry, options.funds)
        if not registry.funds:
            logger.error("No funds available after applying filters")
            return 1

        range_results = run_eod_range_mode(
            session,
            base_cls,
            registry,
            start_date=start_date,
            end_date=end_date,
            operations=params.operations,
            compliance_tests=params.compliance_tests,
            output_dir=output_dir,
            create_pdf=params.create_pdf,
            generate_daily_reports=generate_daily_reports,
        )

    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Processing failed: %s", exc)
        return 1
    finally:
        session.close()

    if not range_results.results_by_date:
        logger.warning("No business days found within the provided range")
        return 0

    for trade_date, results in range_results.results_by_date.items():
        _log_processing_summary(
            logger,
            trade_date.isoformat(),
            results.summary,
        )

    for date_str, artefacts in range_results.daily_artefacts.items():
        _log_generated_paths(logger, flatten_eod_paths(artefacts))

    if range_results.stacked_compliance:
        _log_generated_paths(
            logger,
            flatten_eod_paths({"compliance": range_results.stacked_compliance}),
        )

    logger.info("Time-series run completed successfully")
    return 0


def _filter_registry(registry: FundRegistry, funds: Sequence[str]) -> FundRegistry:
    """Return a registry containing only the requested funds (if any)."""

    if not funds:
        return registry

    missing = [fund for fund in funds if fund not in registry.funds]
    if missing:
        raise ValueError(f"Requested funds not in registry: {', '.join(sorted(missing))}")

    filtered = FundRegistry()
    filtered.funds = {fund: registry.funds[fund] for fund in funds}
    return filtered


def _log_processing_summary(logger: logging.Logger, label: str, summary: Mapping[str, object]) -> None:
    pretty = ", ".join(f"{key}={value}" for key, value in summary.items())
    logger.info("Processing summary (%s): %s", label, pretty)


def _log_generated_paths(logger: logging.Logger, paths: Mapping[str, str]) -> None:
    if not paths:
        logger.warning("No report artefacts were produced")
        return
    for label, path in sorted(paths.items()):
        logger.info("Generated %s -> %s", label, path)


def _extract_override_dates(payload: dict[str, object]) -> list[date]:
    if not payload:
        return []

    dates: list[date] = []

    if "as_of_dates" in payload:
        dates.extend(_coerce_date_sequence(payload.pop("as_of_dates"), "as_of_dates"))

    if "date_range" in payload:
        dates.extend(_expand_date_range(payload.pop("date_range")))

    ordered: list[date] = []
    seen: set[date] = set()
    for run_date in dates:
        if run_date not in seen:
            seen.add(run_date)
            ordered.append(run_date)
    return ordered


def _coerce_date_sequence(values: object, field_name: str) -> list[date]:
    if isinstance(values, MappingABC):
        raise TypeError(f"{field_name} override must be a sequence of dates, not a mapping")
    if isinstance(values, (str, bytes)):
        text = values.decode() if isinstance(values, bytes) else values
        tokens = [token.strip() for token in text.replace(";", ",").split(",") if token.strip()]
        if len(tokens) > 1:
            return [_coerce_date(token, field_name) for token in tokens]
        return [_coerce_date(text.strip(), field_name)]
    if isinstance(values, (date, datetime)):
        return [_coerce_date(values, field_name)]
    if isinstance(values, Iterable):
        return [_coerce_date(item, field_name) for item in values]
    raise TypeError(f"Unsupported {field_name} override payload: {type(values)!r}")


def _expand_date_range(raw_range: object) -> list[date]:
    if isinstance(raw_range, MappingABC):
        start_value = (
            raw_range.get("start")
            or raw_range.get("start_date")
            or raw_range.get("from")
        )
        end_value = raw_range.get("end") or raw_range.get("end_date") or raw_range.get("to")
    elif isinstance(raw_range, SequenceABC) and len(raw_range) == 2:
        start_value, end_value = raw_range
    else:
        raise TypeError("date_range override must be a mapping with 'start'/'end' or a 2-tuple")

    if start_value is None or end_value is None:
        raise ValueError("date_range override requires both start and end values")

    start_date = _coerce_date(start_value, "date_range.start")
    end_date = _coerce_date(end_value, "date_range.end")

    if start_date > end_date:
        raise ValueError("date_range start must be on or before end")

    return list(_iter_business_days_inclusive(start_date, end_date))


def _iter_business_days_inclusive(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def _coerce_date(value: object, field_name: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError as exc:
            raise ValueError(f"Invalid ISO date for {field_name}: {value!r}") from exc
    raise TypeError(f"Unsupported type for {field_name}: {type(value)!r}")


if __name__ == "__main__":

    # 'KNG', 'TDVI', 'RDVI', 'SDVD', 'FDND', 'FGSI', 'DOGG',  'FTCSH'
    # 'P20127', 'P21026', 'P2726', "P30128", 'P31027', 'P3727', 'R21126', 'HE3B1', 'HE3B2', 'TR2B1', 'TR2B2',
    #  'FTCSH'
    # ['KNG', 'TDVI', 'RDVI', 'SDVD', 'FDND', 'FGSI', 'DOGG', 'P21026', 'P2726', "P30128", 'P31027', 'P3727', 'R21126', 'HE3B1', 'HE3B2', 'TR2B1', 'TR2B2'], #

    use_dater_t = "2025-11-19"
    use_dater_t1 = "2025-11-18"
    use_dater_t2 = "2025-11-17"

    RUNTIME_OVERRIDES = {
        "analysis_type": "trading_compliance",
        "as_of_date": use_dater_t,
        "funds": ['KNG', 'TDVI', 'RDVI', 'SDVD', 'FDND', 'FGSI', "DOGG"],
        "compliance_tests": [
            "gics_compliance",
            "prospectus_80pct_policy",
            "diversification_40act_check",
            "diversification_IRS_check",
            "diversification_IRC_check",
            "max_15pct_illiquid_sai",
            "real_estate_check",
            "commodities_check",
            "twelve_d1a_other_inv_cos",
            "twelve_d2_insurance_cos",
            "twelve_d3_sec_biz",
        ],
        "ex_ante_date": use_dater_t,
        "ex_post_date": use_dater_t,
        "custodian_date": use_dater_t1,
        "custodian_previous_date": use_dater_t2,
        "create_pdf": True,
        "output_dir": "./outputs",
    }
    raise SystemExit(main(overrides=RUNTIME_OVERRIDES))

    # RUNTIME_OVERRIDES = {
    #     "analysis_type": "eod",
    #     "as_of_date": "2025-11-18",
    #     #"funds": ['KNG', 'TDVI', 'RDVI', 'SDVD', 'FDND', 'FGSI', 'DOGG', 'P21026', 'P2726', "P30128", 'P31027', 'P3727', 'R21126', 'HE3B1', 'HE3B2', 'TR2B1', 'TR2B2'],
    #     "funds": ["KNG"],
    #     "previous_date": "2025-11-17",
    #     "eod_reports": ["compliance"],
    #     "compliance_tests": [
    #         "gics_compliance",
    #         "prospectus_80pct_policy",
    #         "diversification_40act_check",
    #         "diversification_IRS_check",
    #         "diversification_IRC_check",
    #         "max_15pct_illiquid_sai",
    #         "real_estate_check",
    #         "commodities_check",
    #         "twelve_d1a_other_inv_cos",
    #         "twelve_d2_insurance_cos",
    #         "twelve_d3_sec_biz",
    #     ],
    #     "create_pdf": True,
    #     "output_dir": "./outputs",
    # }
    # raise SystemExit(main(overrides=RUNTIME_OVERRIDES))

    # RUNTIME_OVERRIDES = {
    #     "analysis_type": "eod",
    #     "as_of_date": "2025-11-06",
    #     "previous_date": "2025-11-05",  # optional; defaults to prior business day if omitted
    #     "funds": ["TDVI", "P3727"],  # "DOGG",
    #     "eod_reports": ["reconciliation", "nav"],
    #     "create_pdf": True,
    #     "output_dir": "./outputs",
    # }
    # raise SystemExit(main(overrides=RUNTIME_OVERRIDES))

    # Example: run the same configuration for a list of explicit business dates.
    # MULTI_DAY_OVERRIDES = {
    #     **RUNTIME_OVERRIDES,
    #     "as_of_dates": [
    #         "2025-07-01",
    #         "2025-07-02",
    #         "2025-07-03",
    #     ],
    # }
    # raise SystemExit(main(overrides=MULTI_DAY_OVERRIDES))

    # Example: automatically iterate over every business day in a range.
    # RANGE_OVERRIDES = {
    #     **RUNTIME_OVERRIDES,
    #     "date_range": {"start": "2025-07-01", "end": "2025-07-10"},
    # }
    # raise SystemExit(main(overrides=RANGE_OVERRIDES))

    # TIME_SERIES_OVERRIDES = {
    #     "analysis_type": "eod",
    #     "funds": ['P20127', 'P21026', 'P2726', "P30128", 'P31027', 'P3727', 'R21126', 'HE3B1', 'HE3B2', 'TR2B1', 'TR2B2'],
    #     "eod_reports": ["compliance"],
    #     "compliance_tests": [
    #         "diversification_40act_check",
    #         "diversification_IRS_check",
    #     ],
    #     "start_date": "2025-06-30",
    #     "end_date": "2025-09-30",
    #     "create_pdf": False,
    #     "output_dir": "./outputs",
    #     "generate_daily_reports": False,
    # }
    # raise SystemExit(run_time_series(overrides=TIME_SERIES_OVERRIDES))