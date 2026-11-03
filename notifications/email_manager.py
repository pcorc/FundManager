"""Email notification manager for FundManager pipeline outputs."""
from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from dotenv import load_dotenv

from config.operation_config import OperationConfig, OperationMode
from utilities.numbers import fmt_num, fmt_pct

logger = logging.getLogger(__name__)


# SMTP config — env-driven so credentials never live in the repo.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
SMTP_HOST = os.environ.get("FUNDMANAGER_SMTP_HOST", "smtp.office365.com")
SMTP_PORT = int(os.environ.get("FUNDMANAGER_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("FUNDMANAGER_SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("FUNDMANAGER_SMTP_PASSWORD", "")
FROM_ADDRESS = os.environ.get("FUNDMANAGER_FROM_ADDRESS", SMTP_USER or "noreply@vestfin.com")
EMAIL_MODE = os.environ.get("FUNDMANAGER_EMAIL_MODE", "display")

GL_COLUMNS = ("Equity G/L", "Option G/L", "Flex Option G/L", "Treasury G/L")
_NAV_VALUE_COLUMNS = (("Expected NAV", "expected_nav"), ("Custodian NAV", "current_nav"), ("NAV Diff ($)", "difference"))
_NAV_GOOD_COLUMNS = (("NAV Good (2 Digit)", "nav_good_two"), ("NAV Good (4 Digit)", "nav_good_four"))
# (summary key, break kind): equity/option/flex/treasury holdings+price, then index holdings.
_HOLDINGS_COLUMNS = tuple(
    (f"{asset}_{kind}", kind) for asset in ("equity", "option", "flex_option", "treasury") for kind in ("holdings", "price")
) + (("index_holdings", "holdings"),)
# (metric key, rendered as percent)
_TRADE_COLUMNS = (("trade_value", False), ("pct_of_tna", True), ("pct_of_total_assets", True),
                  ("ex_ante_market_value", False), ("ex_post_market_value", False), ("market_value_delta", False))
_DEFAULT_RECIPIENTS = ["pcorcoran@vestfin.com"]


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send(self, payload: EmailPayload, config: OperationConfig) -> None:
        title, recipients, sections, attached = self._MODES.get(
            config.mode, ("FundManager", _DEFAULT_RECIPIENTS, None, "FundManager output")
        )
        subject = f"{title} — {payload.date}"
        if sections is None:
            body = f"<p>{attached} for {payload.date} attached.</p>"
        else:
            body = (f"<h2>{subject}</h2>" + "".join(sections(self, payload))
                    + "<p>Full PDF and Excel attachments included.</p>")
        self._send_email(recipients, subject, self._html_doc(body), payload.attachments)

    # ------------------------------------------------------------------
    # Body builders
    # ------------------------------------------------------------------
    def _eod_sections(self, payload: EmailPayload) -> List[str]:
        return [
            render(summary)
            for summary, render in (
                (payload.nav_summary, self._render_nav_summary),
                (payload.holdings_summary, self._render_holdings_summary),
                (payload.gl_summary, self._render_gl_summary),
            )
            if summary
        ]

    def _trading_sections(self, payload: EmailPayload) -> List[str]:
        return [
            self._render_compliance_changes(payload.compliance_changes or {}),
            self._render_trade_activity(payload.trade_activity_summary or {}),
        ]

    # mode -> (title, recipients, section builder or None for attachments-only, attachments-only text)
    _MODES = {
        OperationMode.EOD: (
            "EOD Reconciliation",
            ["pcorcoran@vestfin.com", "sshiang@vestfin.com", "dbugbee@vestfin.com", "mrodriguez@vestfin.com"],
            _eod_sections,
            None,
        ),
        OperationMode.TRADING_COMPLIANCE: ("Trading Analysis", ["pcorcoran@vestfin.com"], _trading_sections, None),
        OperationMode.COMPLIANCE_ONLY: ("Compliance Report", ["pcorcoran@vestfin.com"], None, "Compliance report"),
    }

    # ------------------------------------------------------------------
    # Summary renderers
    # ------------------------------------------------------------------
    def _render_nav_summary(self, summary: Mapping[str, Mapping[str, Any]]) -> str:
        rows = []
        for fund, data in sorted(summary.items()):
            data = data or {}
            cells = [f"<td>{fund}</td>"]
            cells += [self._td_r(fmt_num(data.get(key, data.get(alias)), 4)) for key, alias in _NAV_VALUE_COLUMNS]
            for key, alias in _NAV_GOOD_COLUMNS:
                ok = bool(data.get(key, data.get(alias)))
                cells.append(f"<td class='{'pass' if ok else 'fail'}'>{'PASS' if ok else 'FAIL'}</td>")
            rows.append(cells)
        return self._table(
            "NAV Reconciliation Summary",
            ("Fund", "Expected NAV", "Custodian NAV", "NAV Diff", "NAV Good (2-dec)", "NAV Good (4-dec)"),
            rows,
        )

    def _render_holdings_summary(self, summary: Mapping[str, Mapping[str, Any]]) -> str:
        rows = []
        for fund, data in sorted(summary.items()):
            data = data or {}
            rows.append([f"<td>{fund}</td>"]
                        + [self._break_cell(data.get(key, 0), kind=kind) for key, kind in _HOLDINGS_COLUMNS])
        return self._table(
            "Holdings Reconciliation Summary",
            ("Fund", "Eq Holdings", "Eq Price", "Opt Holdings", "Opt Price", "Flex Holdings", "Flex Price",
             "Treasury", "Treasury Price", "Index"),
            rows,
        )

    def _render_gl_summary(self, summary: Mapping[str, Mapping[str, Any]]) -> str:
        rows = []
        for fund, data in sorted(summary.items()):
            data = data or {}
            rows.append([f"<td>{fund}</td>"] + [self._td_r(fmt_num(data.get(key), 2)) for key in GL_COLUMNS])
        return self._table("Gain/Loss Components", ("Fund",) + GL_COLUMNS, rows)

    def _render_compliance_changes(
        self, changes: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> str:
        rows = []
        for fund, fund_changes in sorted(changes.items()):
            for change in (fund_changes or []):
                test = change.get("test", "")
                ante = str(change.get("ante_status", "") or "")
                post = str(change.get("post_status", "") or "")
                rows.append([
                    f"<td>{fund}</td><td>{test}</td>",
                    f"<td class='{self._status_class(ante)}'>{ante}</td>",
                    f"<td class='{self._status_class(post)}'>{post}</td>",
                ])
        return self._table("Compliance Status Changes", ("Fund", "Test", "Ex-Ante", "Ex-Post"), rows,
                           empty_msg="No status changes between ex-ante and ex-post.")

    def _render_trade_activity(
            self, summary: Mapping[str, Mapping[str, Mapping[str, Any]]]
    ) -> str:
        rows = []
        for fund, asset_data in sorted(summary.items()):
            for asset, metrics in sorted((asset_data or {}).items()):
                metrics = metrics or {}
                # Skip rows where every numeric column is zero / blank (e.g. funds
                # with no treasury activity).
                if all(self._is_zero(metrics.get(key)) for key, _ in _TRADE_COLUMNS):
                    continue
                rows.append([f"<td>{fund}</td><td>{str(asset).title()}</td>"] + [
                    self._td_r(fmt_pct(metrics.get(key)) if is_pct else fmt_num(metrics.get(key), 2))
                    for key, is_pct in _TRADE_COLUMNS
                ])
        return self._table(
            "Trade Activity Summary",
            ("Fund", "Asset Class", "Trade Value", "% of TNA", "% of Assets", "Ex-Ante MV", "Ex-Post MV", "MV Delta"),
            rows,
            empty_msg="No trading activity detected.",
        )

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _table(title: str, headers: Sequence[str], rows: Sequence[Sequence[str]],
               empty_msg: Optional[str] = None) -> str:
        if not rows and empty_msg:
            return f"<h3>{title}</h3><p>{empty_msg}</p>"
        return (
            f"<h3>{title}</h3><table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
            + "".join("<tr>" + "".join(cells) + "</tr>" for cells in rows) + "</table>"
        )

    @staticmethod
    def _td_r(text: str) -> str:
        return f"<td class='r'>{text}</td>"

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
    def _is_zero(value: Any) -> bool:
        try:
            return float(value or 0) == 0.0
        except (TypeError, ValueError):
            return True

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
    @staticmethod
    def _present(attachments: Iterable[Path]) -> Iterator[Path]:
        for path in attachments:
            p = Path(path)
            if p.exists():
                yield p
            else:
                logger.warning("Skipping missing attachment %s", p)

    def _display_via_outlook(
            self,
            recipients: Sequence[str],
            subject: str,
            body_html: str,
            attachments: Iterable[Path],
    ) -> None:
        """Pop the message up in Outlook as a draft; user reviews and clicks Send."""
        try:
            import win32com.client  # pywin32
        except ImportError:
            logger.error(
                "pywin32 is required to display Outlook drafts. "
                "Install with: pip install pywin32"
            )
            raise

        outlook = win32com.client.Dispatch("Outlook.Application")
        mail = outlook.CreateItem(0)  # 0 = olMailItem
        mail.To = "; ".join(recipients)
        mail.Subject = subject
        mail.HTMLBody = body_html

        for p in self._present(attachments):
            # Outlook requires an absolute path string.
            mail.Attachments.Add(str(p.resolve()))

        mail.Display(True)  # opens the draft window in the foreground
        # The count deliberately includes skipped (missing) attachments.
        logger.info("Opened Outlook draft '%s' for %s with %d attachment(s)",
                    subject, list(recipients), sum(1 for _ in attachments))

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

        if EMAIL_MODE == "display":
            self._display_via_outlook(recipients, subject, body_html, attachments)
            return

        # ----- original SMTP path -----
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = FROM_ADDRESS
        msg["To"] = ", ".join(recipients)
        msg.set_content("This email is HTML-only; please view in an HTML-capable client.")
        msg.add_alternative(body_html, subtype="html")

        for p in self._present(attachments):
            maintype, subtype = self._mime_type(p)
            msg.add_attachment(p.read_bytes(), maintype=maintype, subtype=subtype, filename=p.name)

        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
                smtp.ehlo()
                smtp.starttls()
                if SMTP_USER and SMTP_PASSWORD:
                    smtp.login(SMTP_USER, SMTP_PASSWORD)
                smtp.send_message(msg)
            logger.info("Sent email '%s' to %s", subject, list(recipients))
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
