from fpdf import FPDF
import logging

logger = logging.getLogger(__name__)


class TradingCompliancePDF(FPDF):
    """Custom PDF class for trading compliance reports."""

    def __init__(self, date):
        super().__init__()
        self.date = date
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        """Page header."""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Trading Compliance Analysis', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 8, f'Ex-Ante vs Ex-Post Comparison - {self.date}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        """Page footer."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def generate_trading_pdf_report(comparison_data: dict, output_path: str):
    """
    Generates PDF report for trading compliance analysis using FPDF.
    """
    try:
        pdf = TradingCompliancePDF(date=comparison_data['date'])
        pdf.add_page()

        # Executive Summary
        _add_executive_summary(pdf, comparison_data)

        # Compliance Changes
        pdf.add_page()
        _add_compliance_changes(pdf, comparison_data)

        # Trade Activity
        pdf.add_page()
        _add_trade_activity(pdf, comparison_data)

        # Detailed Comparison
        pdf.add_page()
        _add_detailed_comparison(pdf, comparison_data)

        # Save PDF
        pdf.output(output_path)
        logger.info(f"Trading PDF report saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error generating trading PDF report: {str(e)}")
        raise


def _add_executive_summary(pdf: FPDF, data: dict):
    """Adds executive summary section."""
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Executive Summary', 0, 1, 'L')
    pdf.ln(2)

    summary = data['summary']

    # Summary metrics
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(120, 8, 'Metric', 1, 0, 'L')
    pdf.cell(60, 8, 'Value', 1, 1, 'C')

    pdf.set_font('Arial', '', 10)

    metrics = [
        ('Total Funds with Trading Activity', summary.get('total_funds_traded', 0)),
        ('', ''),
        ('Funds Moving OUT of Compliance', summary['funds_out_of_compliance']),
        ('Funds Moving INTO Compliance', summary['funds_into_compliance']),
        ('Funds with Unchanged Status', summary['funds_unchanged']),
        ('', ''),
        ('Total Violations (Ex-Ante)', summary['total_violations_before']),
        ('Total Violations (Ex-Post)', summary['total_violations_after']),
        ('Net Change in Violations', summary['total_violations_after'] - summary['total_violations_before'])
    ]

    for label, value in metrics:
        if label == '':
            pdf.cell(120, 6, '', 0, 0)
            pdf.cell(60, 6, '', 0, 1)
        else:
            # Color coding for critical metrics
            if label == 'Funds Moving OUT of Compliance' and value > 0:
                pdf.set_fill_color(255, 204, 204)  # Light red
                pdf.cell(120, 8, label, 1, 0, 'L', True)
                pdf.cell(60, 8, str(value), 1, 1, 'C', True)
            elif label == 'Funds Moving INTO Compliance' and value > 0:
                pdf.set_fill_color(204, 255, 204)  # Light green
                pdf.cell(120, 8, label, 1, 0, 'L', True)
                pdf.cell(60, 8, str(value), 1, 1, 'C', True)
            else:
                pdf.cell(120, 8, label, 1, 0, 'L')
                pdf.cell(60, 8, str(value), 1, 1, 'C')


def _add_compliance_changes(pdf: FPDF, data: dict):
    """Adds compliance changes detail section."""
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Compliance Status Changes by Fund', 0, 1, 'L')
    pdf.ln(2)

    # Table headers
    pdf.set_font('Arial', 'B', 9)
    pdf.set_fill_color(200, 200, 200)

    col_widths = [50, 40, 25, 25, 25]
    headers = ['Fund', 'Status Change', 'Viol. Before', 'Viol. After', 'Net Change']

    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, 1, 0, 'C', True)
    pdf.ln()

    # Data rows
    pdf.set_font('Arial', '', 8)

    for fund_name, fund_data in data['funds'].items():
        # Truncate fund name if too long
        display_name = fund_name if len(fund_name) <= 20 else fund_name[:17] + '...'

        status_change = fund_data['status_change']
        viol_before = fund_data['violations_before']
        viol_after = fund_data['violations_after']
        net_change = viol_after - viol_before

        # Color coding for status changes
        if status_change == 'OUT_OF_COMPLIANCE':
            pdf.set_fill_color(255, 204, 204)  # Light red
            fill = True
        elif status_change == 'INTO_COMPLIANCE':
            pdf.set_fill_color(204, 255, 204)  # Light green
            fill = True
        else:
            fill = False

        pdf.cell(col_widths[0], 7, display_name, 1, 0, 'L', fill)
        pdf.cell(col_widths[1], 7, status_change, 1, 0, 'C', fill)
        pdf.cell(col_widths[2], 7, str(viol_before), 1, 0, 'C', fill)
        pdf.cell(col_widths[3], 7, str(viol_after), 1, 0, 'C', fill)
        pdf.cell(col_widths[4], 7, str(net_change), 1, 1, 'C', fill)


def _add_trade_activity(pdf, comparison_data):
    """Add trade activity section showing changes between ex_ante and ex_post"""
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Trade Activity", ln=True)
    pdf.ln(5)

    # Table headers
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(200, 200, 200)

    col_widths = [50, 30, 25, 25, 25]
    headers = ['Fund', 'Total Traded', 'Equity', 'Treasury', 'Options']

    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, 1, 0, 'C', True)
    pdf.ln()

    # Data rows
    pdf.set_font("Arial", "", 8)

    funds_dict = comparison_data.get('funds', {})

    for fund_name, fund_data in funds_dict.items():
        # Get trade info if it exists
        trade_info = fund_data.get('trade_info', {})

        if not trade_info:
            continue

        # Extract values safely
        total_traded = trade_info.get('total_traded', 0)
        equity = trade_info.get('equity', 0)
        treasury = trade_info.get('treasury', 0)
        options = trade_info.get('options', 0)

        # Truncate fund name if needed
        display_name = fund_name if len(fund_name) <= 20 else fund_name[:17] + '...'

        pdf.cell(col_widths[0], 7, display_name, 1, 0, 'L')
        pdf.cell(col_widths[1], 7, f"{total_traded:.2f}", 1, 0, 'R')
        pdf.cell(col_widths[2], 7, f"{equity:.2f}", 1, 0, 'R')
        pdf.cell(col_widths[3], 7, f"{treasury:.2f}", 1, 0, 'R')
        pdf.cell(col_widths[4], 7, f"{options:.2f}", 1, 1, 'R')


def _add_detailed_comparison(pdf: FPDF, data: dict):
    """Adds detailed check-by-check comparison."""
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Detailed Comparison by Compliance Check', 0, 1, 'L')
    pdf.ln(2)

    for fund_name, fund_data in data['funds'].items():
        # Fund header
        pdf.set_font('Arial', 'B', 11)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 8, fund_name, 1, 1, 'L', True)

        # Check headers
        pdf.set_font('Arial', 'B', 8)
        pdf.set_fill_color(200, 200, 200)

        col_widths = [60, 25, 25, 20, 20, 15]
        headers = ['Compliance Check', 'Before', 'After', 'Viol. B', 'Viol. A', 'Changed']

        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7, header, 1, 0, 'C', True)
        pdf.ln()

        # Check data
        pdf.set_font('Arial', '', 7)

        for check_name, check_data in fund_data['checks'].items():
            # Truncate check name if too long
            display_check = check_name if len(check_name) <= 30 else check_name[:27] + '...'

            # Highlight changed rows
            if check_data['changed']:
                pdf.set_fill_color(255, 255, 204)  # Light yellow
                fill = True
            else:
                fill = False

            pdf.cell(col_widths[0], 6, display_check, 1, 0, 'L', fill)
            pdf.cell(col_widths[1], 6, check_data['status_before'], 1, 0, 'C', fill)
            pdf.cell(col_widths[2], 6, check_data['status_after'], 1, 0, 'C', fill)
            pdf.cell(col_widths[3], 6, str(check_data['violations_before']), 1, 0, 'C', fill)
            pdf.cell(col_widths[4], 6, str(check_data['violations_after']), 1, 0, 'C', fill)
            pdf.cell(col_widths[5], 6, 'YES' if check_data['changed'] else 'NO', 1, 1, 'C', fill)

        pdf.ln(3)

        # Check if we need a new page
        if pdf.get_y() > 250:
            pdf.add_page()