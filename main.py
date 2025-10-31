from __future__ import annotations


import logging
import os
from datetime import date, datetime
from pathlib import Path
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

    options = parse_arguments(argv)
    options = apply_overrides(options, overrides or RUNTIME_OVERRIDES)
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
    # RUNTIME_OVERRIDES = {
    #     "analysis_type": "trading_compliance",
    #     "as_of_date": "2025-10-24",
    #     "funds": ["DOGG"],
    #     "compliance_tests": [
    #         # "gics_compliance",
    #         "prospectus_80pct_policy",
    #         "diversification_40act_check",
    #         "diversification_IRS_check",
    #         # "diversification_IRC_check",
    #         # "max_15pct_illiquid_sai",
    #         # "real_estate_check",
    #         # "commodities_check",
    #         # "twelve_d1a_other_inv_cos",
    #         # "twelve_d2_insurance_cos",
    #         # "twelve_d3_sec_biz",
    #     ],
    #     "ex_ante_date": "2025-10-24",
    #     "ex_post_date": "2025-10-24",
    #     "custodian_date": "2025-10-23",
    #     "custodian_previous_date": "2025-10-22",
    #     "create_pdf": True,
    #     "output_dir": "./outputs",
    # }
    # raise SystemExit(main(overrides=RUNTIME_OVERRIDES))

    # 'KNG', 'TDVI', 'RDVI', 'SDVD', 'FDND', 'FGSI', 'DOGG',
    # 'P20127', 'P21026', 'P2726', "P30128", 'P31027', 'P3727',
    # 'R21126', 'HE3B1', 'HE3B2', 'TR2B1', 'TR2B2', 'FTCSH'

    RUNTIME_OVERRIDES = {
        "analysis_type": "eod",
        "as_of_date": "2025-10-24",
        "funds": ['P20127',], # 'P21026', 'P2726', "P30128", 'P31027', 'P3727', 'R21126', 'HE3B1', 'HE3B2', 'TR2B1', 'TR2B2'
        "previous_date": "2025-10-23",
        "eod_reports": ["compliance"],
        "compliance_tests": [
            # "gics_compliance",
            # "prospectus_80pct_policy",
            # "diversification_40act_check",
            "diversification_IRS_check",
            # "diversification_IRC_check",
            # "max_15pct_illiquid_sai",
            # "real_estate_check",
            # "commodities_check",
            # "twelve_d1a_other_inv_cos",
            # "twelve_d2_insurance_cos",
            # "twelve_d3_sec_biz",
        ],
        "create_pdf": True,
        "output_dir": "./outputs",
    }
    raise SystemExit(main(overrides=RUNTIME_OVERRIDES))

    # TIME_SERIES_OVERRIDES = {
    #     "analysis_type": "eod",
    #     "funds": [    'P20127', 'P21026', 'P2726', "P30128", 'P31027', 'P3727',
    # 'R21126', 'HE3B1', 'HE3B2', 'TR2B1', 'TR2B2'],
    #     "eod_reports": ["compliance",],
    #     "compliance_tests": [
    #         "diversification_40act_check",
    #         "diversification_IRS_check",
    #     ],
    #     "start_date": "2025-06-30",
    #     "end_date": "2025-09-19",
    #     "create_pdf": True,
    #     "output_dir": "./outputs",
    #     "generate_daily_reports": False,
    # }
    # raise SystemExit(run_time_series(overrides=TIME_SERIES_OVERRIDES))