# reporting/report_orchestrator.py
class ReportOrchestrator:
    """Efficiently generates all required reports"""

    def __init__(self, output_dir: str, session, base_cls, gics_dict: Dict):
        self.output_dir = output_dir
        self.session = session
        self.base_cls = base_cls
        self.gics_dict = gics_dict

    def generate_reports(self, result: ProcessingResult, config: OperationConfig) -> Dict[str, str]:
        """Generate reports based on processing results"""
        date_str = result.date
        file_paths = {}

        if config.mode == OperationMode.TRADING_COMPLIANCE:
            if result.traded_funds_info and result.compliance_results:
                file_paths.update(self._generate_trading_compliance_reports(result, date_str))
        else:
            if config.run_compliance and result.compliance_results:
                file_paths.update(self._generate_compliance_reports(result, date_str, config))

            if config.run_reconciliation and result.reconciliation_results:
                file_paths.update(self._generate_reconciliation_reports(result, date_str, config))

        return file_paths

    def _generate_trading_compliance_reports(self, result: ProcessingResult, date_str: str) -> Dict[str, str]:
        """Generate trading compliance comparison reports"""
        file_paths = {}

        trading_excel = os.path.join(self.output_dir, f"trading_compliance_{date_str}.xlsx")
        trading_pdf = os.path.join(self.output_dir, f"trading_compliance_{date_str}.pdf")

        # Generate comparison report
        generate_trading_excel_report(result.compliance_results['comparison'], trading_excel)
        generate_trading_pdf_report(result.compliance_results['comparison'], trading_pdf)

        # Generate detailed ex_post reports for traded funds only
        if result.compliance_results.get('ex_post'):
            compliance_excel = os.path.join(self.output_dir, f"compliance_ex_post_{date_str}.xlsx")
            compliance_pdf = os.path.join(self.output_dir, f"compliance_ex_post_{date_str}.pdf")

            ComplianceReport(
                results=result.compliance_results['ex_post'],
                file_path=compliance_excel,
                test_functions=config.test_functions,
                gics_dict=self.gics_dict,
                session=self.session,
                analysis_type='ex_post',
                base_cls=self.base_cls
            )

            ComplianceReportPDF(
                results=result.compliance_results['ex_post'],
                output_path=compliance_pdf
            )

            file_paths.update({
                'trading_excel': trading_excel,
                'trading_pdf': trading_pdf,
                'compliance_excel': compliance_excel,
                'compliance_pdf': compliance_pdf
            })

        return file_paths

    def _generate_compliance_reports(self, result: ProcessingResult, date_str: str, config: OperationConfig) -> Dict[str, str]:
        """Generate standard compliance reports"""
        master_excel = os.path.join(self.output_dir, f"compliance_results_{date_str}.xlsx")
        master_pdf = os.path.join(self.output_dir, f"compliance_results_{date_str}.pdf")

        ComplianceReport(
            results={date_str: result.compliance_results},
            file_path=master_excel,
            test_functions=config.test_functions,
            gics_dict=self.gics_dict,
            session=self.session,
            analysis_type=config.analysis_type,
            base_cls=self.base_cls
        )

        ComplianceReportPDF(
            results={date_str: result.compliance_results},
            output_path=master_pdf
        )

        return {
            'compliance_excel': master_excel,
            'compliance_pdf': master_pdf
        }

    def _generate_reconciliation_reports(self, result: ProcessingResult, date_str: str, config: OperationConfig) -> Dict[str, str]:
        """Generate reconciliation reports"""
        file_paths = {}

        # Holdings reconciliation
        recon_excel = os.path.join(self.output_dir, f"reconciliation_results_{date_str}.xlsx")
        recon_pdf = os.path.join(self.output_dir, f"reconciliation_results_{date_str}.pdf")

        ReconciliationReport(
            reconciliation_results={date_str: result.reconciliation_results},
            recon_summary=[],  # You might need to adjust this
            date=date_str,
            file_path_excel=recon_excel
        )

        file_paths.update({
            'recon_excel': recon_excel,
            'recon_pdf': recon_pdf
        })

        # NAV reconciliation
        if config.run_nav_reconciliation and result.nav_reconciliation_results:
            nav_recon_excel = os.path.join(self.output_dir, f"nav_reconciliation_results_{date_str}.xlsx")

            NAVReconciliationReport(
                reconciliation_results={date_str: result.nav_reconciliation_results},
                recon_summary={date_str: []},  # Adjust as needed
                date=date_str,
                file_path_excel=nav_recon_excel
            )

            file_paths['nav_recon_excel'] = nav_recon_excel

            # Combined PDF
            combined_pdf = os.path.join(self.output_dir, f"combined_reconciliation_results_{date_str}.pdf")
            create_combined_report(
                nav_reconciliation_results={date_str: result.nav_reconciliation_results},
                holdings_reconciliation_results={date_str: result.reconciliation_results},
                nav_recon_summary={date_str: []},
                holdings_recon_summary={date_str: []},
                compliance_results={date_str: result.compliance_results} if result.compliance_results else {},
                date=date_str,
                output_path=combined_pdf
            )

            file_paths['combined_pdf'] = combined_pdf

        return file_paths