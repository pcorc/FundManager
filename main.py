"""
Refactored main entry point for fund compliance processing.

This module provides orchestration for running compliance and reconciliation
operations across multiple funds and date ranges.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import (Dict, List, Mapping, Optional, Sequence)
from processing.fund import build_fund_registry
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
from utilities.main_helpers import (
    coerce_date as helper_coerce_date,
    extract_override_dates,
    filter_registry,
    log_generated_paths,
    log_processing_summary,
)

from config.run_configurations import (
    build_run,
    exclude_funds,
    generate_business_date_range
)
from config.database import initialize_database
from config.fund_definitions import (
    load_cohorts_from_db,
    ETF_FUNDS, CLOSED_END_FUNDS, PRIVATE_FUNDS, VIT_AND_MUTUAL_FUNDS,
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
    multi_day_dates = extract_override_dates(override_payload)
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

        load_cohorts_from_db(session)

        registry = build_fund_registry(session, base_cls)
        registry = filter_registry(registry, options.funds)
        if not registry:
            logger.error("No funds available after applying filters")
            return 1

        if options.analysis_type == "trading_compliance":
            params = resolve_trading_parameters(options)
            requests = plan_trading_requests(params)
        else:
            params = resolve_eod_parameters(options)
            requests = plan_eod_requests(params)

        data_stores = fetch_data_stores(session, registry, requests)

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
            output_tag=options.output_tag,
        )

        log_processing_summary(logger, "ex-ante", results_ex_ante.summary)
        log_processing_summary(logger, "ex-post", results_ex_post.summary)
        log_generated_paths(logger, flatten_trading_paths(artefacts))
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
            output_tag=options.output_tag,
        )

        log_processing_summary(logger, "eod", processing_results.summary)
        log_generated_paths(logger, flatten_eod_paths(artefacts))

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

    start_date = helper_coerce_date(start_raw, "start_date")
    end_date = helper_coerce_date(end_raw, "end_date")
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
        registry = build_fund_registry(session, base_cls)
        registry = filter_registry(registry, options.funds)
        if not registry:
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
            output_tag=params.output_tag,
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
        log_processing_summary(
            logger,
            trade_date.isoformat(),
            results.summary,
        )

    for date_str, artefacts in range_results.daily_artefacts.items():
        log_generated_paths(logger, flatten_eod_paths(artefacts))

    if range_results.stacked_compliance:
        log_generated_paths(
            logger,
            flatten_eod_paths({"compliance": range_results.stacked_compliance}),
        )

    logger.info("Time-series run completed successfully")
    return 0


def execute_run(cfg: Mapping[str, object] | List[Mapping[str, object]]) -> int:
    """Execute a config (dict) or batch of configs (list of dicts)."""
    if isinstance(cfg, list):
        for sub in cfg:
            result = execute_run(sub)
            if result != 0:
                return result
        return 0

    # ... existing single-config body unchanged ...

    start = date.fromisoformat(str(cfg["start_date"]))
    end   = date.fromisoformat(str(cfg["end_date"]))

    if cfg.get("date_mode") == "range":
        return run_time_series(overrides=cfg)

    exit_code = 0
    for trade_date in generate_business_date_range(start, end):
        iter_cfg = dict(cfg)
        iter_cfg["as_of_date"] = trade_date.isoformat()
        result = main(overrides=iter_cfg)
        if result != 0:
            return result
    return exit_code

# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================

if __name__ == "__main__":

    _boot_session, _ = initialize_database()
    try:
        load_cohorts_from_db(_boot_session)
    finally:
        _boot_session.close()

    # 2) Build the run

    # # EOD all reports, single day, single cohort
    # cfg = build_run("eod", cohorts=[ETF_FUNDS], start_date="2026-05-14")
    #
    # # Just NAV recon, single day, one fund
    # cfg = build_run("nav", funds=["KNG"], start_date="2026-05-14")
    #
    # # Trading compliance over a week, multiple cohorts
    # cfg = build_run(
    #     "trading_compliance",
    #     cohorts=[ETF_FUNDS, CLOSED_END_FUNDS],
    #     start_date="2026-05-14",
    #     end_date="2026-05-21",
    # )
    #
    # # Time-series stacked compliance over a month
    # cfg = build_run(
    #     "time_series",
    #     cohorts=[CLOSED_END_FUNDS],
    #     start_date="2026-04-21",
    #     end_date="2026-05-21",
    # )
    #
    # # All four EOD report flavors on the same day — chain them
    # cfg = build_run(
    #     ["compliance", "reconciliation", "nav"],
    #     cohorts=[ETF_FUNDS],
    #     start_date="2026-05-14",
    #     output_tag="etfs",
    # )
    #
    # cfg = build_run(
    #     mode="compliance",
    #     #funds=exclude_funds(CLOSED_END_FUNDS, "FTMIX"),
    #     cohorts=[ETF_FUNDS],
    #     start_date="2026-05-14",
    #     end_date="2026-05-22",
    #     output_tag="etfs",
    # )

    # Just NAV recon, single day, one fund
    # cfg = build_run(
    #                 mode="eod",
    #                 funds=exclude_funds(CLOSED_END_FUNDS, "FTMIX"),
    #                 start_date="2026-05-26",
    #                 end_date="2026-05-27",
    #                 output_tag="cefs"
    #                 )

    cfg = build_run(
        "trading_compliance",
        funds=["p2726", "RDVI"],
        start_date="2026-05-22",
        #end_date="2026-05-21",
        output_tag="may22"
    )

    # 3) Execute
    exit_code = execute_run(cfg)
    raise SystemExit(exit_code)
