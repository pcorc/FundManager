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
from typing import Dict, List, Mapping, Optional, Sequence

from config.database import initialize_database
from config.fund_registry import FundRegistry
from config.run_configurations import (
    calculate_date_offsets,
    get_config,
    list_available_configs,
    merge_overrides,
)
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

# Optional runtime overrides for quick local tweaking without CLI arguments.
RUNTIME_OVERRIDES: Optional[Mapping[str, object]] = None

# Import fund groups and helper functions
from config.fund_definitions import (
    ETF_FUNDS,
    CLOSED_END_FUNDS,
    PRIVATE_FUNDS,
    ALL_FUNDS,
)
from config.run_configurations import build_fund_list, exclude_funds

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
        registry = FundRegistry.from_database(session, base_cls)
        registry = filter_registry(registry, options.funds)
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
        registry = FundRegistry.from_database(session, base_cls)
        registry = filter_registry(registry, options.funds)
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


# ============================================================================
# NEW: BATCH CONFIGURATION ORCHESTRATOR
# ============================================================================

def run_configuration_batch(
        config_names: List[str],
        base_date: str | date,
        overrides: Optional[Dict[str, Dict[str, object]]] = None,
) -> int:
    """
    Execute multiple run configurations in sequence.

    Args:
        config_names: List of configuration names to execute
        base_date: Base date for all runs (T date, used to calculate T-1, T-2)
        overrides: Optional per-config overrides dict {config_name: {param: value}}

    Returns:
        0 if all runs succeeded, 1 if any failed

    Example:
        run_configuration_batch(
            config_names=["trading_compliance_daily", "eod_compliance_all_funds"],
            base_date="2025-11-21",
            overrides={
                "trading_compliance_daily": {"funds": ["RDVI"]},
            }
        )
    """
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("fund_manager.batch")

    # Validate and convert base_date
    if isinstance(base_date, str):
        base_date = helper_coerce_date(base_date, "base_date")

    # Calculate date offsets
    date_offsets = calculate_date_offsets(base_date)
    logger.info(
        "Base date configuration: T=%s, T-1=%s, T-2=%s",
        date_offsets["t"],
        date_offsets["t1"],
        date_offsets["t2"],
    )

    # Validate all configs exist before starting
    logger.info("Validating %d configurations...", len(config_names))
    for config_name in config_names:
        try:
            get_config(config_name)
        except ValueError as e:
            logger.error("Configuration validation failed: %s", e)
            return 1

    # Execute each configuration
    results = []
    logger.info("Starting batch execution of %d configurations", len(config_names))

    for idx, config_name in enumerate(config_names, 1):
        logger.info("=" * 80)
        logger.info("Executing configuration %d/%d: %s", idx, len(config_names), config_name)
        logger.info("=" * 80)

        try:
            # Load base config
            config = get_config(config_name)

            # Apply any user overrides for this specific config
            if overrides and config_name in overrides:
                config = merge_overrides(config, overrides[config_name])

            # Execute based on date_mode
            date_mode = config.get("date_mode", "single")

            if date_mode == "single":
                exit_code = _execute_single_date_config(config, date_offsets, logger)
            elif date_mode == "range":
                exit_code = _execute_range_date_config(config, date_offsets, logger)
            else:
                raise ValueError(f"Unknown date_mode: {date_mode}")

            results.append({
                "config": config_name,
                "status": "success" if exit_code == 0 else "failed",
                "exit_code": exit_code,
            })

            if exit_code != 0:
                logger.error("Configuration '%s' failed with exit code %d", config_name, exit_code)
            else:
                logger.info("Configuration '%s' completed successfully", config_name)

        except Exception as exc:
            logger.exception("Configuration '%s' raised exception: %s", config_name, exc)
            results.append({
                "config": config_name,
                "status": "error",
                "error": str(exc),
            })

    # Report summary
    logger.info("=" * 80)
    logger.info("BATCH EXECUTION SUMMARY")
    logger.info("=" * 80)

    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] in ("failed", "error")]

    logger.info("Total configurations: %d", len(results))
    logger.info("Successful: %d", len(successes))
    logger.info("Failed: %d", len(failures))

    if failures:
        logger.error("Failed configurations:")
        for result in failures:
            error_detail = result.get("error", f"exit code {result.get('exit_code', 'unknown')}")
            logger.error("  - %s: %s", result["config"], error_detail)
        return 1

    logger.info("All configurations completed successfully!")
    return 0


def _execute_single_date_config(
        config: Dict[str, object],
        date_offsets: Dict[str, date],
        logger: logging.Logger,
) -> int:
    """Execute a single-date configuration (trading_compliance or eod)."""

    analysis_type = config["analysis_type"]

    # Build overrides for main()
    override_payload = {
        "analysis_type": analysis_type,
        "as_of_date": date_offsets["t"],
        "funds": config.get("funds", []),
        "create_pdf": config.get("create_pdf", True),
        "output_dir": config.get("output_dir", "./outputs"),
        "output_tag": config.get("output_tag"),
    }

    if analysis_type == "trading_compliance":
        override_payload.update({
            "ex_ante_date": date_offsets["t"],
            "ex_post_date": date_offsets["t"],
            "custodian_date": date_offsets["t1"],
            "custodian_previous_date": date_offsets["t2"],
            "compliance_tests": config.get("compliance_tests", []),
        })
    elif analysis_type == "eod":
        override_payload.update({
            "previous_date": date_offsets["t1"],
            "eod_reports": config.get("eod_reports", []),
            "compliance_tests": config.get("compliance_tests", []),
        })

    return main(overrides=override_payload)


def _execute_range_date_config(
        config: Dict[str, object],
        date_offsets: Dict[str, date],
        logger: logging.Logger,
) -> int:
    """Execute a range-date configuration (time series)."""

    # For range configs, user must provide start_date/end_date in overrides
    # or we use a default range around the base date
    if "start_date" not in config or "end_date" not in config:
        logger.warning(
            "Range config missing start_date/end_date; using default 30-day range from base date"
        )
        from datetime import timedelta
        start_date = date_offsets["t"] - timedelta(days=30)
        end_date = date_offsets["t"]
    else:
        start_date = helper_coerce_date(config["start_date"], "start_date")
        end_date = helper_coerce_date(config["end_date"], "end_date")

    override_payload = {
        "analysis_type": config["analysis_type"],
        "start_date": start_date,
        "end_date": end_date,
        "funds": config.get("funds", []),
        "eod_reports": config.get("eod_reports", []),
        "compliance_tests": config.get("compliance_tests", []),
        "create_pdf": config.get("create_pdf", False),
        "output_dir": config.get("output_dir", "./outputs"),
        "generate_daily_reports": config.get("generate_daily_reports", False),
    }

    return run_time_series(overrides=override_payload)

# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================

if __name__ == "__main__":

    # Base date: All date offsets (T, T-1, T-2) are calculated from this
    BASE_DATE = "2025-11-24"

    # ------------------------------------------------------------------------
    # Example 1: Run predefined configurations
    # ------------------------------------------------------------------------
    # ACTIVE_RUNS = [
    #     # "trading_compliance_etfs",
    #     # "eod_compliance_etfs",
    #     "eod_recon_etfs",
    # ]
    #
    # exit_code = run_configuration_batch(
    #     config_names=ACTIVE_RUNS,
    #     base_date=BASE_DATE,
    # )
    # raise SystemExit(exit_code)


    # ------------------------------------------------------------------------
    # Example 1: Run predefined configurations
    # ------------------------------------------------------------------------
    # ACTIVE_RUNS = [
    #     "trading_compliance_closed_end_private",
    #     # "eod_compliance_closed_end_private",
    #     # "eod_recon_closed_end_private",
    # ]
    #
    # RUN_OVERRIDES = {
    #     "trading_compliance_closed_end_private": {
    #         "funds": build_fund_list(
    #             exclude_funds(CLOSED_END_FUNDS, PRIVATE_FUNDS),
    #             "P3727"
    #         ),
    #         "output_tag": "p3727",  # Custom tag for file names
    #     },
    #     "eod_compliance_custom": {
    #         "funds": build_fund_list(
    #             exclude_funds(CLOSED_END_FUNDS, PRIVATE_FUNDS),
    #             "P3727"
    #         ),
    #         "output_tag": "p3727",  # Custom tag for file names
    #         "compliance_tests": [
    #                     "gics_compliance",
    #                     "prospectus_80pct_policy",
    #                     "diversification_40act_check",
    #                     "diversification_IRS_check",
    #                     "diversification_IRC_check",
    #                     "max_15pct_illiquid_sai",
    #                     "real_estate_check",
    #                     "commodities_check",
    #                     "twelve_d1a_other_inv_cos",
    #                     "twelve_d2_insurance_cos",
    #                     "twelve_d3_sec_biz"
    #                 ],
    #     },
    #     "eod_recon_custom": {
    #         "funds": build_fund_list(
    #             exclude_funds(CLOSED_END_FUNDS, PRIVATE_FUNDS),
    #             "P3727"
    #         ),
    #         "output_tag": "p3727",  # Custom tag for file names
    #     },
    # }
    #
    # exit_code = run_configuration_batch(
    #     config_names=ACTIVE_RUNS,
    #     base_date=BASE_DATE,
    #     overrides=RUN_OVERRIDES,
    #
    # )
    # raise SystemExit(exit_code)

    # ------------------------------------------------------------------------
    # Example 2: Combine multiple fund groups + individual tickers
    # ------------------------------------------------------------------------
    # ETF_FUNDS,
    # CLOSED_END_FUNDS,
    # PRIVATE_FUNDS,
    # ALL_FUNDS,

    ACTIVE_RUNS = [
        # "trading_compliance_custom",
        # "eod_compliance_custom",
        "eod_recon_custom",
    ]

    RUN_OVERRIDES = {
        # "trading_compliance_custom": {
        #     # ETFs + one specific closed-end fund
        #     "funds": build_fund_list( ETF_FUNDS),
        #     "output_tag": "custom_cef",  # Custom tag for file names
        # },
        # "eod_compliance_custom": {
        #     # All three fund groups combined
        #     "funds": build_fund_list(
        #         "KNG"
        #     ),
        #     "output_tag": "cef",  # Custom tag for file names
        #     "compliance_tests": [
        #                 "summary_metrics",
        #                 "gics_compliance",
        #                 "prospectus_80pct_policy",
        #                 "diversification_40act_check",
        #                 "diversification_IRS_check",
        #                 "diversification_IRC_check",
        #                 "max_15pct_illiquid_sai",
        #                 "real_estate_check",
        #                 "commodities_check",
        #                 "twelve_d1a_other_inv_cos",
        #                 "twelve_d2_insurance_cos",
        #                 "twelve_d3_sec_biz"
        #             ],
        # },
        "eod_recon_custom": {
            # Closed-end funds + ETFs + two specific funds
            "funds": build_fund_list(
                "KNG"
            ),
            "output_tag": "cef",  # Custom tag for file names
        },
    }

    exit_code = run_configuration_batch(
        config_names=ACTIVE_RUNS,
        base_date=BASE_DATE,
        overrides=RUN_OVERRIDES,
    )
    raise SystemExit(exit_code)

    # ------------------------------------------------------------------------
    # Example 3: Run ALL funds EXCEPT specific ones (exclusion pattern)
    # ------------------------------------------------------------------------
    # ACTIVE_RUNS = [
    #     "trading_compliance_all_except",
    #     "eod_compliance_all_except",
    #     "eod_recon_all_except",
    # ]
    #
    # RUN_OVERRIDES = {
    #     "trading_compliance_all_except": {
    #         # All funds except these three
    #         "funds": exclude_funds(ALL_FUNDS, "RDVI", "KNG", "FTMIX"),
    #     },
    #     "eod_compliance_all_except": {
    #         # All funds except private funds
    #         "funds": exclude_funds(ALL_FUNDS, PRIVATE_FUNDS),
    #     },
    #     "eod_recon_all_except": {
    #         # All funds except private funds and a few specific ETFs
    #         "funds": exclude_funds(ALL_FUNDS, PRIVATE_FUNDS, "RDVI", "KNG"),
    #     },
    # }
    #
    # exit_code = run_configuration_batch(
    #     config_names=ACTIVE_RUNS,
    #     base_date=BASE_DATE,
    #     overrides=RUN_OVERRIDES,
    # )
    # raise SystemExit(exit_code)

    # ------------------------------------------------------------------------
    # Example 4: Complex combinations
    # ------------------------------------------------------------------------
    # ACTIVE_RUNS = [
    #     # "trading_compliance_custom",
    #     # "eod_compliance_custom",
    #     "eod_recon_custom"
    # ]
    #
    # RUN_OVERRIDES = {
    #     # "trading_compliance_custom": {
    #     #     # ETFs except RDVI and KNG, plus one closed-end fund
    #     #     "funds": build_fund_list(
    #     #         # exclude_funds(ETF_FUNDS, "RDVI", "KNG"),
    #     #         "RDVI"
    #     #     ),
    #     # },
    #     # "eod_compliance_custom": {
    #     #     # All funds except private funds and specific ETFs, plus add back one private fund
    #     #     "funds": build_fund_list(
    #     #         # exclude_funds(ALL_FUNDS, PRIVATE_FUNDS, "RDVI", "KNG"),
    #     #         "RDVI"  # Add back one private fund
    #     #     ),
    #     # },
    #     "eod_recon_all_custom": {
    #         # All funds except private funds and specific ETFs, plus add back one private fund
    #         "funds":
    #             build_fund_list("RDVI"),
    #     },
    # }

    # exit_code = run_configuration_batch(
    #     config_names=ACTIVE_RUNS,
    #     base_date=BASE_DATE,
    #     overrides=RUN_OVERRIDES,
    # )
    # raise SystemExit(exit_code)

    # ------------------------------------------------------------------------
    # Example 5: Run configurations with overrides (simple)
    # ------------------------------------------------------------------------
    # ACTIVE_RUNS = [
    #     "trading_compliance_etfs",
    #     "eod_compliance_closed_end_private",
    # ]
    #
    # RUN_OVERRIDES = {
    #     "trading_compliance_etfs": {
    #         "funds": ["RDVI", "KNG", "TDVI"],  # Just these three instead of all ETFs
    #     },
    # }
    #
    # exit_code = run_configuration_batch(
    #     config_names=ACTIVE_RUNS,
    #     base_date=BASE_DATE,
    #     overrides=RUN_OVERRIDES,
    # )
    # raise SystemExit(exit_code)

    # ------------------------------------------------------------------------
    # Example 6: Run all funds across all modes
    # ------------------------------------------------------------------------
    # ACTIVE_RUNS = [
    #     "trading_compliance_all_funds",
    #     "eod_compliance_all_funds",
    #     "eod_recon_all_funds",
    # ]
    #
    # exit_code = run_configuration_batch(
    #     config_names=ACTIVE_RUNS,
    #     base_date=BASE_DATE,
    # )
    # raise SystemExit(exit_code)