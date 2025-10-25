# notifications/email_manager.py
class EmailManager:
    """Manages email notifications efficiently"""

    def __init__(self, poppler_path: str):
        self.poppler_path = poppler_path
        self.recipient_map = {
            OperationMode.EOD: [
                "pcorcoran@vestfin.com",
                "sshiang@vestfin.com",
                "dbugbee@vestfin.com",
                "mrodriguez@vestfin.com"
            ],
            OperationMode.TRADING_COMPLIANCE: [
                "pcorcoran@vestfin.com"
            ]
        }

    def send_notifications(self, date: str, file_paths: Dict[str, str], config: OperationConfig):
        """Send appropriate email notifications"""
        recipients = self.recipient_map.get(config.mode, ["pcorcoran@vestfin.com"])

        if config.mode == OperationMode.TRADING_COMPLIANCE:
            self._send_trading_compliance_email(date, file_paths, recipients)
        else:
            self._send_regular_email(date, file_paths, recipients, config)

    def _send_trading_compliance_email(self, date: str, file_paths: Dict[str, str], recipients: List[str]):
        """Send trading compliance email"""
        if file_paths.get('trading_pdf'):
            extra_attachments = []
            if file_paths.get('trading_excel'):
                extra_attachments.append(file_paths['trading_excel'])
            if file_paths.get('compliance_excel'):
                extra_attachments.append(file_paths['compliance_excel'])

            send_combined_recon_with_inline_pages(
                pdf_path=file_paths['trading_pdf'],
                recipients=recipients,
                date_range=f"Trading Compliance - {date}",
                poppler_path=self.poppler_path,
                extra_attachments=extra_attachments,
                delete_temp_images=True
            )

    def _send_regular_email(self, date: str, file_paths: Dict[str, str], recipients: List[str], config: OperationConfig):
        """Send regular EOD email"""
        if file_paths.get('combined_pdf'):
            extra_attachments = []

            # Add Excel reports
            if file_paths.get('recon_excel'):
                extra_attachments.append(file_paths['recon_excel'])
            if file_paths.get('nav_recon_excel'):
                extra_attachments.append(file_paths['nav_recon_excel'])

            # Add compliance reports if run
            if config.run_compliance:
                if file_paths.get('compliance_excel'):
                    extra_attachments.append(file_paths['compliance_excel'])
                if file_paths.get('compliance_pdf'):
                    extra_attachments.append(file_paths['compliance_pdf'])

            send_combined_recon_with_inline_pages(
                pdf_path=file_paths['combined_pdf'],
                recipients=recipients,
                date_range=date,
                poppler_path=self.poppler_path,
                extra_attachments=extra_attachments,
                delete_temp_images=True
            )