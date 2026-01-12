# ------------------------------------------------------------------------
# Example 1: Run predefined configurations
# ------------------------------------------------------------------------
# ACTIVE_RUNS = [
#     # "trading_compliance_closed_end_private",
#     "eod_compliance_closed_end_private",
#     # "eod_recon_closed_end_private",
# ]
#
# RUN_OVERRIDES = {
#     # "trading_compliance_closed_end_private": {
#     #     "funds": build_fund_list(CLOSED_END_FUNDS, PRIVATE_FUNDS),
#     #     "output_tag": "cef",  # Custom tag for file names
#     # },
#     "eod_compliance_custom": {
#         "funds": build_fund_list(CLOSED_END_FUNDS, PRIVATE_FUNDS),
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
#     # "eod_recon_custom": {
#     #     "funds": build_fund_list(
#     #         exclude_funds(CLOSED_END_FUNDS, PRIVATE_FUNDS),
#     #         "P3727"
#     #     ),
#     #     "output_tag": "p3727",  # Custom tag for file names
#     # },
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

# ACTIVE_RUNS = [
#     # "trading_compliance_custom",
#     # "eod_compliance_custom",
#     "eod_recon_custom",
# ]
#
# RUN_OVERRIDES = {
#     # "trading_compliance_custom": {
#     #     # ETFs + one specific closed-end fund
#     #     "funds": build_fund_list( ETF_FUNDS),
#     #     "output_tag": "custom_cef",  # Custom tag for file names
#     # },
#     # "eod_compliance_custom": {
#     #     # All three fund groups combined
#     #     "funds": build_fund_list(
#     #         "KNG"
#     #     ),
#     #     "output_tag": "cef",  # Custom tag for file names
#     #     "compliance_tests": [
#     #                 "summary_metrics",
#     #                 "gics_compliance",
#     #                 "prospectus_80pct_policy",
#     #                 "diversification_40act_check",
#     #                 "diversification_IRS_check",
#     #                 "diversification_IRC_check",
#     #                 "max_15pct_illiquid_sai",
#     #                 "real_estate_check",
#     #                 "commodities_check",
#     #                 "twelve_d1a_other_inv_cos",
#     #                 "twelve_d2_insurance_cos",
#     #                 "twelve_d3_sec_biz"
#     #             ],
#     # },
#     "eod_recon_custom": {
#         # Closed-end funds + ETFs + two specific funds
#         "funds": build_fund_list(
#             "KNG"
#         ),
#         "output_tag": "cef",  # Custom tag for file names
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