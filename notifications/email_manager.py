"""Email notification manager for FundManager pipeline outputs."""
from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from config.operation_config import OperationConfig, OperationMode

logger = logging.getLogger(__name__)


# SMTP config — env-driven so credentials never live in the repo.
SMTP_HOST = os.environ.get("FUNDMANAGER_SMTP_HOST", "smtp.office365.com")
SMTP_PORT = int(os.environ.get("FUNDMANAGER_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("FUNDMANAGER_SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("FUNDMANAGER_SMTP_PASSWORD", "")
FROM_ADDRESS = os.environ.get("FUNDMANAGER_FROM_ADDRESS", SMTP_USER or "noreply@vestfin.com")


@dataclass
class EmailPayload:
    """Bundle of artefacts + structured summary data for one pipeline run."""
    date: str
    attachments: List[Path] = field(default_factory=list)
    # EOD body sections (each is {fund: {key: value}})
    nav_summary: Optional[Mapping[str, Mapping[str, Any]]] = None
    holdings_summary: Optional[Mapping[str, Mapping[str, Any]]] = None
    gl_summary: Optional[Mapping[str, Mapping[str, Any]]] = None
    # Trading-analysis body sections
    compliance_changes: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None
    trade_activity_summary: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None


class EmailManager:
    """Sends per-run notification emails with attachments and inline summaries."""

    def __init__(self) -> None:
        self.recipient_map = {
            OperationMode.EOD: [
                "pcorcoran@vestfin.com",
                "sshiang@vestfin.com",
                "dbugbee@vestfin.com",
                "mrodriguez@vestfin.com",
            ],
            OperationMode.TRADING_COMPLIANCE: [
                "pcorcoran@vestfin.com",
            ],
            OperationMode.COMPLIANCE_ONLY: [
                "pcorcoran@vestfin.com",
            ],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send(self, payload: EmailPayload, config: OperationConfig) -> None:
        recipients = self.recipient_map.get(config.mode, ["pcorcoran@vestfin.com"])

        if config.mode == OperationMode.EOD:
            subject = f"EOD Reconciliation — {payload.date}"
            body_html = self._build_eod_body(payload)
        elif config.mode == OperationMode.TRADING_COMPLIANCE:
            subject = f"Trading Analysis — {payload.date}"
            body_html = self._build_trading_body(payload)
        elif config.mode == OperationMode.COMPLIANCE_ONLY:
            subject = f"Compliance Report — {payload.date}"
            body_html = self._build_compliance_body(payload)
        else:
            subject = f"FundManager — {payload.date}"
            body_html = self._html_doc(f"<p>FundManager output for {payload.date} attached.</p>")

        self._send_email(recipients, subject, body_html, payload.attachments)

    # ------------------------------------------------------------------
    # Body builders
    # ------------------------------------------------------------------
    def _build_eod_body(self, payload: EmailPayload) -> str:
        sections = [f"<h2>EOD Reconciliation — {payload.date}</h2>"]
        if payload.nav_summary:
            sections.append(self._render_nav_summary(payload.nav_summary))
        if payload.holdings_summary:
            sections.append(self._render_holdings_summary(payload.holdings_summary))
        if payload.gl_summary:
            sections.append(self._render_gl_summary(payload.gl_summary))
        sections.append("<p>Full PDF and Excel attachments included.</p>")
        return self._html_doc("".join(sections))

    def _build_trading_body(self, payload: EmailPayload) -> str:
        sections = [f"<h2>Trading Analysis — {payload.date}</h2>"]
        sections.append(self._render_compliance_changes(payload.compliance_changes or {}))
        sections.append(self._render_trade_activity(payload.trade_activity_summary or {}))
        sections.append("<p>Full PDF and Excel attachments included.</p>")
        return self._html_doc("".join(sections))

    def _build_compliance_body(self, payload: EmailPayload) -> str:
        # Per spec: attachments only.
        return self._html_doc(f"<p>Compliance report for {payload.date} attached.</p>")

    # ------------------------------------------------------------------
    # Summary renderers
    # ------------------------------------------------------------------
    def _render_nav_summary(self, summary: Mapping[str, Mapping[str, Any]]) -> str:
        rows = []
        for fund, data in sorted(summary.items()):
            data = data or {}
            ok2 = bool(data.get("NAV Good (2 Digit)", data.get("nav_good_two")))
            ok4 = bool(data.get("NAV Good (4 Digit)", data.get("nav_good_four")))
            rows.append(
                "<tr>"
                f"<td>{fund}</td>"
                f"<td class='r'>{self._num(data.get('Expected NAV', data.get('expected_nav')), 4)}</td>"
                f"<td class='r'>{self._num(data.get('Custodian NAV', data.get('current_nav')), 4)}</td>"
                f"<td class='r'>{self._num(data.get('NAV Diff ($)', data.get('difference')), 4)}</td>"
                f"<td class='{'pass' if ok2 else 'fail'}'>{'PASS' if ok2 else 'FAIL'}</td>"
                f"<td class='{'pass' if ok4 else 'fail'}'>{'PASS' if ok4 else 'FAIL'}</td>"
                "</tr>"
            )
        return (
            "<h3>NAV Reconciliation Summary</h3>"
            "<table><tr><th>Fund</th><th>Expected NAV</th><th>Custodian NAV</th>"
            "<th>NAV Diff</th><th>NAV Good (2-dec)</th><th>NAV Good (4-dec)</th></tr>"
            + "".join(rows) + "</table>"
        )

    def _render_holdings_summary(self, summary: Mapping[str, Mapping[str, Any]]) -> str:
        rows = []
        for fund, data in sorted(summary.items()):
            data = data or {}
            cells = [f"<td>{fund}</td>"]
            for key, kind in (
                ("equity_holdings", "holdings"),
                ("equity_price", "price"),
                ("option_holdings", "holdings"),
                ("option_price", "price"),
                ("flex_option_holdings", "holdings"),
                ("flex_option_price", "price"),
                ("treasury_holdings", "holdings"),
                ("treasury_price", "price"),
                ("index_holdings", "holdings"),
            ):
                cells.append(self._break_cell(data.get(key, 0), kind=kind))
            rows.append("<tr>" + "".join(cells) + "</tr>")
        return (
            "<h3>Holdings Reconciliation Summary</h3>"
            "<table><tr><th>Fund</th>"
            "<th>Eq Holdings</th><th>Eq Price</th>"
            "<th>Opt Holdings</th><th>Opt Price</th>"
            "<th>Flex Holdings</th><th>Flex Price</th>"
            "<th>Treasury</th><th>Treasury Price</th>"
            "<th>Index</th></tr>"
            + "".join(rows) + "</table>"
        )

    def _render_gl_summary(self, summary: Mapping[str, Mapping[str, Any]]) -> str:
        rows = []
        for fund, data in sorted(summary.items()):
            data = data or {}
            rows.append(
                "<tr>"
                f"<td>{fund}</td>"
                f"<td class='r'>{self._num(data.get('Equity G/L'), 2)}</td>"
                f"<td class='r'>{self._num(data.get('Option G/L'), 2)}</td>"
                f"<td class='r'>{self._num(data.get('Flex Option G/L'), 2)}</td>"
                f"<td class='r'>{self._num(data.get('Treasury G/L'), 2)}</td>"
                "</tr>"
            )
        return (
            "<h3>Gain/Loss Components</h3>"
            "<table><tr><th>Fund</th><th>Equity G/L</th><th>Option G/L</th>"
            "<th>Flex Option G/L</th><th>Treasury G/L</th></tr>"
            + "".join(rows) + "</table>"
        )

    def _render_compliance_changes(
        self, changes: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> str:
        rows = []
        for fund, fund_changes in sorted(changes.items()):
            for change in (fund_changes or []):
                test = change.get("test", "")
                ante = str(change.get("ante_status", "") or "")
                post = str(change.get("post_status", "") or "")
                rows.append(
                    "<tr>"
                    f"<td>{fund}</td><td>{test}</td>"
                    f"<td class='{self._status_class(ante)}'>{ante}</td>"
                    f"<td class='{self._status_class(post)}'>{post}</td>"
                    "</tr>"
                )
        if not rows:
            return "<h3>Compliance Status Changes</h3><p>No status changes between ex-ante and ex-post.</p>"
        return (
            "<h3>Compliance Status Changes</h3>"
            "<table><tr><th>Fund</th><th>Test</th><th>Ex-Ante</th><th>Ex-Post</th></tr>"
            + "".join(rows) + "</table>"
        )

    def _render_trade_activity(
        self, summary: Mapping[str, Mapping[str, Mapping[str, Any]]]
    ) -> str:
        rows = []
        for fund, asset_data in sorted(summary.items()):
            for asset, metrics in sorted((asset_data or {}).items()):
                metrics = metrics or {}
                rows.append(
                    "<tr>"
                    f"<td>{fund}</td><td>{str(asset).title()}</td>"
                    f"<td class='r'>{self._num(metrics.get('trade_value'), 2)}</td>"
                    f"<td class='r'>{self._pct(metrics.get('pct_of_tna'))}</td>"
                    f"<td class='r'>{self._pct(metrics.get('pct_of_total_assets'))}</td>"
                    f"<td class='r'>{self._num(metrics.get('ex_ante_market_value'), 2)}</td>"
                    f"<td class='r'>{self._num(metrics.get('ex_post_market_value'), 2)}</td>"
                    f"<td class='r'>{self._num(metrics.get('market_value_delta'), 2)}</td>"
                    "</tr>"
                )
        if not rows:
            return "<h3>Trade Activity Summary</h3><p>No trading activity detected.</p>"
        return (
            "<h3>Trade Activity Summary</h3>"
            "<table><tr><th>Fund</th><th>Asset Class</th><th>Trade Value</th>"
            "<th>% of TNA</th><th>% of Assets</th><th>Ex-Ante MV</th><th>Ex-Post MV</th>"
            "<th>MV Delta</th></tr>"
            + "".join(rows) + "</table>"
        )

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _html_doc(body: str) -> str:
        return f"""<html>
<head><style>
body {{ font-family: Arial, sans-serif; font-size: 12px; color: #222; }}
h2 {{ color: #1f3864; margin: 10px 0 6px; }}
h3 {{ color: #1f3864; margin: 14px 0 4px; font-size: 13px; }}
table {{ border-collapse: collapse; margin-bottom: 12px; font-size: 11px; }}
th, td {{ border: 1px solid #999; padding: 3px 6px; }}
th {{ background-color: #d9d9d9; }}
td.r {{ text-align: right; }}
td.pass {{ background-color: #c8e6c9; text-align: center; font-weight: bold; }}
td.fail {{ background-color: #f5b8b8; text-align: center; font-weight: bold; }}
</style></head>
<body>{body}</body>
</html>"""

    @staticmethod
    def _num(value: Any, digits: int = 2) -> str:
        try:
            return f"{float(value):,.{digits}f}"
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _pct(value: Any) -> str:
        try:
            return f"{float(value):.2%}"
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _break_cell(value: Any, *, kind: str = "holdings") -> str:
        try:
            count = int(float(value))
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            color = "#ffc896" if kind == "price" else "#ffc8c8"   # orange / red
            return f"<td class='r' style='background-color:{color};'>{count}</td>"
        return f"<td class='r'>{count}</td>"

    @staticmethod
    def _status_class(status: str) -> str:
        s = status.strip().lower()
        if s in ("pass", "compliant", "ok", "yes", "true"):
            return "pass"
        if s in ("fail", "non-compliant", "violation", "no", "false"):
            return "fail"
        return "r"

    # ------------------------------------------------------------------
    # SMTP send
    # ------------------------------------------------------------------
    def _send_email(
        self,
        recipients: Sequence[str],
        subject: str,
        body_html: str,
        attachments: Iterable[Path],
    ) -> None:
        if not recipients:
            logger.warning("No recipients configured; skipping email '%s'", subject)
            return

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = FROM_ADDRESS
        msg["To"] = ", ".join(recipients)
        msg.set_content("This email is HTML-only; please view in an HTML-capable client.")
        msg.add_alternative(body_html, subtype="html")

        for path in attachments:
            p = Path(path)
            if not p.exists():
                logger.warning("Skipping missing attachment %s", p)
                continue
            maintype, subtype = self._mime_type(p)
            msg.add_attachment(p.read_bytes(), maintype=maintype, subtype=subtype, filename=p.name)

        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
                smtp.ehlo()
                smtp.starttls()
                if SMTP_USER and SMTP_PASSWORD:
                    smtp.login(SMTP_USER, SMTP_PASSWORD)
                smtp.send_message(msg)
            logger.info("Sent email '%s' to %s with %d attachment(s)",
                        subject, list(recipients), len(list(attachments)))
        except Exception:
            logger.exception("Failed to send email '%s'", subject)
            raise

    @staticmethod
    def _mime_type(path: Path) -> Tuple[str, str]:
        suffix = path.suffix.lower()
        return {
            ".pdf":  ("application", "pdf"),
            ".xlsx": ("application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ".xls":  ("application", "vnd.ms-excel"),
            ".csv":  ("text", "csv"),
            ".txt":  ("text", "plain"),
        }.get(suffix, ("application", "octet-stream"))