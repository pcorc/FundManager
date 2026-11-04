"""FundManager web UI (Streamlit): New run, Run history and Reports.

    python -m streamlit run ui/app.py [--server.address 0.0.0.0 --server.port 8501]

The app holds no business logic: it builds a ``processing.api.RunRequest``,
validates it in-process and hands it to ``ui.runner``, which runs
``python -m processing.worker`` in its own process and run folder under
``FUNDMANAGER_UI_OUTPUT_ROOT``. Runs never write to the shared drive;
"Publish" copies a finished run's reports to ``FUNDMANAGER_UI_PUBLISH_ROOT``.

Settings (environment or ``.env``): ``FUNDMANAGER_UI_OUTPUT_ROOT``,
``FUNDMANAGER_UI_PUBLISH_ROOT``, ``FUNDMANAGER_UI_TIMEOUT_MINUTES``,
``FUNDMANAGER_UI_USER_HEADER`` (SSO proxy header with the username),
``FUNDMANAGER_UI_PASSWORD`` (optional shared password; also read from
``st.secrets``) and ``FUNDMANAGER_UI_CATALOG_TTL`` (seconds, default 600).

With ``FUNDMANAGER_UI_USER_HEADER`` set, the app trusts that header as the
user's identity, so Streamlit must only be reachable through the SSO proxy
(listen on 127.0.0.1 or firewall the port) and the proxy must overwrite any
copy of the header the client sends.
"""
from __future__ import annotations

import getpass
import hmac
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # `streamlit run ui/app.py` only puts ui/ on the path
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from config import fund_definitions  # noqa: E402
from processing import api  # noqa: E402
from ui import components as ui  # noqa: E402
from ui import history_page, new_run, runner, theme  # noqa: E402

_log = logging.getLogger("fundmanager.ui")
CATALOG_RETRY_SECONDS = 60  # how long a failed catalog load is cached before the DB is tried again


def _catalog_ttl() -> int:
    try:
        return max(int(os.getenv("FUNDMANAGER_UI_CATALOG_TTL") or 600), 0)
    except ValueError:
        return 600


@st.cache_data(ttl=_catalog_ttl(), show_spinner="Loading fund groups from the database…")
def load_catalog() -> dict:
    """``processing.api.catalog()`` (reads the cohorts from the DB), cached for the TTL.

    A failure is cached too, as a DB-free fallback (no fund groups) marked with ``_error``, so a down DB
    costs one connect timeout per :data:`CATALOG_RETRY_SECONDS`, not one per click.
    """
    try:
        return api.catalog()
    except Exception as exc:  # the DB is down or misconfigured: keep the app usable with explicit tickers
        _log.error("Loading the catalog failed: %s", type(exc).__name__)  # no message: it may name the DB
        return {"cohorts": {}, "funds": sorted(fund_definitions.FUND_DEFINITIONS),
                "operations": list(api.OPERATIONS), "rule_catalog": api.rule_catalog(),
                "date_modes": list(api.DATE_MODES), "_error": type(exc).__name__, "_at": time.time()}


def get_catalog() -> dict:
    """The cached catalog (or its fallback, retried after :data:`CATALOG_RETRY_SECONDS`), with a notice on failure."""
    catalog = load_catalog()
    if "_error" in catalog and time.time() - catalog.get("_at", 0) > CATALOG_RETRY_SECONDS:
        load_catalog.clear()
        catalog = load_catalog()
    if "_error" in catalog:
        st.error(f"Could not load the fund groups from the database ({catalog['_error']}); see the server log. "
                 "You can still pick funds by ticker. Use 'Reload fund groups' (under your name, top right) to retry.")
    return catalog


PAGES: Dict[str, Callable[[str], None]] = {
    "New run": lambda user: new_run.render(get_catalog(), user),
    "Run history": history_page.render_history,
    "Reports": history_page.render_reports,
}


def _configured_password() -> Optional[str]:
    value = os.getenv("FUNDMANAGER_UI_PASSWORD")
    if not value:
        try:
            value = st.secrets.get("FUNDMANAGER_UI_PASSWORD")
        except Exception:  # no secrets.toml
            value = None
    return str(value) if value else None


def password_gate() -> bool:
    """``True`` when no password is configured or this session entered it (``hmac.compare_digest``)."""
    expected = _configured_password()
    if expected is None or st.session_state.get("authenticated"):
        return True
    st.title("FundManager")
    with st.form("login"):
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if hmac.compare_digest(password.encode("utf-8"), expected.encode("utf-8")):
            st.session_state["authenticated"] = True
            st.session_state.pop("login_password", None)
            st.rerun()
        st.error("Wrong password.")
    return False


def current_user() -> Optional[str]:
    """The SSO proxy's username header when ``FUNDMANAGER_UI_USER_HEADER`` is set, else the OS user."""
    header = os.getenv("FUNDMANAGER_UI_USER_HEADER", "").strip()
    if header:
        return st.context.headers.get(header) or None
    return getpass.getuser()


def top_bar(user: str) -> str:
    """Brand, page navigation (a segmented control on ``page``, so Re-run can switch page from a callback) and
    a popover with the user, the output / publish folders and *Reload fund groups*; returns the page."""
    brand, nav, account = st.columns([1.3, 2.4, 1.3], vertical_alignment="center")
    brand.html('<div class="fm-brand"><span class="fm-brand-mark">FM</span><span class="fm-brand-name">FundManager'
               '</span><span class="fm-brand-sub">compliance &amp; reconciliation</span></div>')
    with nav.container(key="fm_nav", horizontal=True, horizontal_alignment="center"):
        page = st.segmented_control("Page", list(PAGES), key="page", required=True, label_visibility="collapsed")
    with account.container(horizontal=True, horizontal_alignment="right"):
        with st.popover(user, icon=":material/account_circle:"):
            st.caption(f"Signed in as **{user}**")
            st.caption(f"Output: `{runner.output_root()}`")
            publish = runner.publish_root()
            st.caption(f"Publish: `{publish}`" if publish else "Publish: off")
            if st.button("Reload fund groups", key="fm_reload", help="Re-read the fund groups (cohorts) from the database",
                         icon=":material/refresh:"):
                load_catalog.clear()
    if runner.root_too_deep(runner.output_root()):
        ui.show(ui.callout("warn", "The output folder path is long: some report paths will pass 260 characters, "
                                   "which Explorer and Excel may not open. Set FUNDMANAGER_UI_OUTPUT_ROOT to a short "
                                   "folder."))
    return page or "New run"


def main() -> None:
    st.set_page_config(page_title="FundManager", page_icon=":material/fact_check:", layout="wide")
    theme.inject()
    if not password_gate():
        return
    user = current_user()
    if user is None:
        st.error("Not signed in: the sign-in header is missing. Open the app through the company SSO link.")
        return
    st.session_state.setdefault("page", "New run")
    new_run.keep_form_state()  # the New run picks survive visits to the other pages
    page = top_bar(user)
    PAGES[page](user)

main()
