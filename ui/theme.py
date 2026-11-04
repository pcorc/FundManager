"""The UI's look: one soft dark palette and the CSS that styles Streamlit's widgets and the app's HTML.

``.streamlit/config.toml`` sets the same palette as Streamlit's theme (so its
own widgets, focus rings and the progress bar match); :func:`inject` adds the
CSS for the custom HTML pieces built in :mod:`ui.components` and for the
widget groups the pages key by prefix:

- ``nrf_<ticker>`` fund checkboxes, ``nrr_<rule>`` compliance-check checkboxes and
  ``nr_op_<operation>`` operation checkboxes render as selectable tiles that light
  up when ticked;
- ``nr_settings`` is the New run settings panel; ``fm_nav`` the top navigation;
- the Trading Compliance page uses ``tc_settings``, ``tc_chips``, ``tc_actions`` and ``tcf_`` / ``tcr_`` tiles the
  same way; :data:`TRADING_CSS` styles its dashboard (KPI tiles, transition chips, the summary table, charts).

Call :func:`inject` once at the top of every script run (it emits a style-only
``st.html``, which takes no space on the page).
"""
from __future__ import annotations

import streamlit as st

# Soft dark palette: no pure black, no neon. Keep in step with .streamlit/config.toml.
BG = "#1F2430"
SURFACE = "#262D3B"
SURFACE_2 = "#2C3446"
SURFACE_3 = "#333C50"
BORDER = "#363F52"
TEXT = "#E3E7EF"
# Text tiers: TEXT for values, MUTED for secondary text people read (labels, footnotes, reasons), FAINT for
# hints and meta lines. Each clears the 4.5:1 body-text minimum on BG, SURFACE and SURFACE_2.
MUTED = "#B0B9CA"
FAINT = "#929CB0"
ACCENT = "#7C9CF5"
ACCENT_TEXT = "#A9BDF8"
BAND = "#3A4A75"  # rule-card header band (a soft blue)
PASS = "#7FCB9B"
FAIL = "#EE8E8E"
WARN = "#E6C07B"
NA = "#A3ACBD"

FONT_STACK = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"

CSS = f"""
:root {{
  --fm-bg: {BG}; --fm-surface: {SURFACE}; --fm-surface-2: {SURFACE_2}; --fm-surface-3: {SURFACE_3};
  --fm-border: {BORDER}; --fm-text: {TEXT}; --fm-muted: {MUTED}; --fm-faint: {FAINT};
  --fm-accent: {ACCENT}; --fm-accent-text: {ACCENT_TEXT}; --fm-band: {BAND};
  --fm-accent-soft: rgba(124, 156, 245, 0.13); --fm-accent-line: rgba(124, 156, 245, 0.55);
  --fm-pass: {PASS}; --fm-fail: {FAIL}; --fm-warn: {WARN}; --fm-na: {NA};
  --fm-pass-soft: rgba(127, 203, 155, 0.13); --fm-fail-soft: rgba(238, 142, 142, 0.13);
  --fm-warn-soft: rgba(230, 192, 123, 0.14); --fm-na-soft: rgba(163, 172, 189, 0.12);
  --fm-radius: 12px; --fm-shadow: 0 1px 2px rgba(10, 12, 20, 0.25), 0 8px 24px rgba(10, 12, 20, 0.18);
}}

/* ---- page frame -------------------------------------------------------------------------------------- */
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(124,156,245,0.07), transparent 60%), var(--fm-bg); }}
header[data-testid="stHeader"] {{ background: transparent; height: 2.5rem; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1680px; }}
html, body, .stApp, .stMarkdown, button, input, textarea, select {{ font-family: {FONT_STACK}; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
[data-testid="stHeading"] h1 {{ font-size: 1.55rem; font-weight: 650; padding: 0.2rem 0 0; }}
[data-testid="stCaptionContainer"] {{ color: var(--fm-muted); }}
hr {{ border-color: var(--fm-border) !important; opacity: .7; }}

/* ---- top bar ----------------------------------------------------------------------------------------- */
.fm-brand {{ display: flex; align-items: center; gap: .65rem; height: 2.5rem; }}
.fm-brand-mark {{ width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
  background: linear-gradient(135deg, #8EA9F7, #6C86D9); color: #1F2430; font-weight: 800; font-size: .95rem; }}
.fm-brand-name {{ font-weight: 650; font-size: 1.08rem; color: var(--fm-text); }}
.fm-brand-sub {{ color: var(--fm-faint); font-size: .8rem; margin-left: .15rem; }}
.st-key-fm_nav [data-testid="stButtonGroup"] button {{ min-width: 7.5rem; }}
@media (max-width: 1440px) {{ .fm-brand-sub {{ display: none; }} .st-key-fm_nav [data-testid="stButtonGroup"] button {{ min-width: 6rem; }} }}
@media (max-width: 1100px) {{ .st-key-fm_nav [data-testid="stButtonGroup"] button {{ min-width: 0; padding: 0 .55rem; }} }}
.fm-page-head {{ margin: .4rem 0 1.1rem; }}
.fm-page-head h1 {{ font-size: 1.5rem; font-weight: 650; margin: 0; color: var(--fm-text); }}
.fm-page-head p {{ margin: .2rem 0 0; color: var(--fm-muted); font-size: .92rem; }}

/* ---- Streamlit widgets: softer surfaces -------------------------------------------------------------- */
[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stVerticalBlock"][class*="st-key-"][style*="border"] {{
  border-radius: var(--fm-radius); }}
.st-key-nr_settings, .st-key-tc_settings {{ background: var(--fm-surface); border: 1px solid var(--fm-border) !important;
  border-radius: 16px; padding: 1.1rem 1.15rem 0; box-shadow: var(--fm-shadow); }}
/* The settings column stays in view with its own scroll, and RUN / RESET stay pinned at its foot, so the progress
   and results that appear at the top of the right pane are on screen when RUN is pressed. */
[data-testid="stColumn"]:has(.st-key-nr_settings, .st-key-tc_settings) {{ position: sticky; top: 3rem; align-self: flex-start; }}
.st-key-nr_settings, .st-key-tc_settings {{ max-height: calc(100vh - 4rem); overflow-y: auto; overscroll-behavior: contain;
  scrollbar-width: thin; scrollbar-color: var(--fm-surface-3) transparent; }}
[data-testid="stLayoutWrapper"]:has(> .st-key-nr_actions, > .st-key-tc_actions) {{ position: sticky; bottom: 0; z-index: 3;
  background: var(--fm-surface); padding: 0 0 1rem; }}
.st-key-nr_actions .fm-rule, .st-key-tc_actions .fm-rule {{ margin-top: 0; }}
/* Below 1200px the settings and the preview stack (Streamlit itself stacks columns only below 640px). */
@media (max-width: 1200px) {{
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] :is(.st-key-nr_settings, .st-key-tc_settings)) {{
    flex-direction: column; }}
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] :is(.st-key-nr_settings, .st-key-tc_settings))
    > [data-testid="stColumn"] {{ width: 100% !important; flex: 1 1 100% !important; }}
  [data-testid="stColumn"]:has(.st-key-nr_settings, .st-key-tc_settings) {{ position: static; align-self: stretch; }}
  .st-key-nr_settings, .st-key-tc_settings {{ max-height: none; overflow: visible; }}
}}
.st-key-nr_preview {{ padding: 0 .2rem; }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {{
  background: var(--fm-surface-2) !important; border-color: var(--fm-border) !important; border-radius: 9px !important; }}
[data-testid="stDateInput"] input, [data-testid="stTextInput"] input {{ font-variant-numeric: tabular-nums; }}
[data-testid="stWidgetLabel"] p {{ color: var(--fm-muted); font-size: .8rem; font-weight: 500; letter-spacing: .01em; }}
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button {{ border-radius: 9px; font-weight: 550;
  border-color: var(--fm-border); transition: all .15s ease; }}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {{ color: #1C2130; letter-spacing: .04em;
  background: var(--fm-accent) !important; border-color: var(--fm-accent) !important; box-shadow: 0 4px 14px rgba(124,156,245,.22); }}
.stButton button[kind="primary"]:hover:not(:disabled), .stDownloadButton button[kind="primary"]:hover {{
  background: #8DAAF7 !important; color: #1C2130 !important; }}
.stButton button[kind="primary"]:focus, .stButton button[kind="primary"]:active,
.stDownloadButton button[kind="primary"]:focus, .stDownloadButton button[kind="primary"]:active {{ color: #1C2130 !important; }}
.stButton button[kind="primary"]:disabled {{ background: var(--fm-surface-2) !important; border-color: var(--fm-border) !important;
  color: var(--fm-faint) !important; box-shadow: none; }}
.st-key-nr_chips button[kind="primary"], .st-key-tc_chips button[kind="primary"] {{ background: var(--fm-accent-soft) !important; color: var(--fm-accent-text) !important;
  border-color: var(--fm-accent-line) !important; box-shadow: none; letter-spacing: 0; }}
.stButton button[kind="secondary"]:hover:not(:disabled) {{ border-color: var(--fm-accent-line); color: var(--fm-accent-text); }}
[data-testid="stButtonGroup"] button {{ border-color: var(--fm-border); }}
/* Segmented controls and pills (Streamlit 1.66 marks them data-variant / aria-checked): unselected segments sit on
   SURFACE_2 like the tiles and inputs beside them, the selected one is tinted like a ticked tile. */
:is(.st-key-nr_settings, .st-key-tc_settings) [data-testid="stButtonGroup"] button[data-variant="segmented_control"] {{ background: var(--fm-surface-2) !important; }}
[data-testid="stButtonGroup"] button[data-variant][aria-checked="true"] {{ background: var(--fm-accent-soft) !important;
  color: var(--fm-accent-text) !important; border-color: var(--fm-accent-line) !important; }}
:is(.st-key-nr_settings, .st-key-tc_settings) [data-testid="stButtonGroup"] > div {{ flex-wrap: nowrap; }}
:is(.st-key-nr_settings, .st-key-tc_settings) [data-testid="stButtonGroup"] button p {{ white-space: nowrap; }}
.stDownloadButton button {{ height: auto; min-height: 2.5rem; }}
.stDownloadButton button p {{ white-space: normal; text-align: left; line-height: 1.3; }}
[data-testid="stExpander"] details {{ border-color: var(--fm-border); border-radius: var(--fm-radius); background: var(--fm-surface); }}
[data-testid="stExpander"] summary {{ color: var(--fm-muted); }}
[data-testid="stAlert"] {{ border-radius: var(--fm-radius); }}
[data-testid="stCode"] pre {{ background: #232938 !important; border-radius: 10px; }}

/* ---- selectable tiles (fund, check and operation checkboxes) ----------------------------------------- */
[class*="st-key-nrf_"] label, [class*="st-key-nrr_"] label, [class*="st-key-nr_op_"] label, [class*="st-key-tcf_"] label, [class*="st-key-tcr_"] label {{
  width: 100%; margin: 0; padding: .34rem .6rem; border: 1px solid var(--fm-border); border-radius: 8px;
  background: var(--fm-surface-2); transition: background .12s ease, border-color .12s ease; }}
[class*="st-key-nrf_"] label:hover, [class*="st-key-nrr_"] label:hover, [class*="st-key-nr_op_"] label:hover,
[class*="st-key-tcf_"] label:hover, [class*="st-key-tcr_"] label:hover {{
  border-color: var(--fm-accent-line); }}
[class*="st-key-nrf_"]:has(input:checked) label, [class*="st-key-nrr_"]:has(input:checked) label,
[class*="st-key-nr_op_"]:has(input:checked) label,
[class*="st-key-tcf_"]:has(input:checked) label, [class*="st-key-tcr_"]:has(input:checked) label {{ background: var(--fm-accent-soft); border-color: var(--fm-accent-line); }}
[class*="st-key-nrf_"]:has(input:checked) label p, [class*="st-key-nrr_"]:has(input:checked) label p,
[class*="st-key-nr_op_"]:has(input:checked) label p,
[class*="st-key-tcf_"]:has(input:checked) label p, [class*="st-key-tcr_"]:has(input:checked) label p {{ color: var(--fm-accent-text); }}
[class*="st-key-nrr_"] label p, [class*="st-key-nr_op_"] label p, [class*="st-key-tcr_"] label p {{ font-size: .86rem; }}
[class*="st-key-nrf_"] label p, [class*="st-key-tcf_"] label p {{ font-size: .86rem; font-weight: 550; letter-spacing: .03em; font-variant-numeric: tabular-nums; }}
[class*="st-key-nrf_"], [class*="st-key-nrr_"], [class*="st-key-nr_op_"], [class*="st-key-tcf_"],
[class*="st-key-tcr_"] {{ margin-bottom: -0.35rem; }}
:is(.st-key-nr_chips, .st-key-tc_chips) button {{ padding: .1rem .7rem; min-height: 1.9rem; font-size: .8rem; border-radius: 999px; }}

/* ---- section headings inside the settings panel ------------------------------------------------------ */
.fm-sec {{ display: flex; flex-wrap: wrap; gap: .2rem .6rem; align-items: baseline; justify-content: space-between;
  margin: .35rem 0 .15rem; }}
.fm-sec h3 {{ font-size: 1.02rem; font-weight: 640; margin: 0; padding: 0; color: var(--fm-text); white-space: nowrap; }}
.fm-sec .fm-count {{ display: inline-block; margin-left: .45rem; padding: 0 .5rem; border-radius: 999px; font-size: .75rem;
  background: var(--fm-accent-soft); color: var(--fm-accent-text); font-weight: 600; vertical-align: 2px; }}
.fm-sec small {{ color: var(--fm-faint); font-size: .76rem; }}
.fm-rule {{ height: 1px; background: var(--fm-border); margin: .9rem -1.15rem .7rem; opacity: .8; }}
.fm-hint {{ color: var(--fm-faint); font-size: .78rem; margin: .1rem 0 .2rem; }}

/* ---- pills ------------------------------------------------------------------------------------------- */
.fm-pill {{ display: inline-block; min-width: 2.9rem; text-align: center; padding: .08rem .55rem; border-radius: 999px;
  font-size: .72rem; font-weight: 650; letter-spacing: .05em; line-height: 1.45rem; border: 1px solid transparent;
  white-space: nowrap; }}
.fm-pass {{ color: var(--fm-pass); background: var(--fm-pass-soft); border-color: rgba(127,203,155,.28); }}
.fm-fail {{ color: var(--fm-fail); background: var(--fm-fail-soft); border-color: rgba(238,142,142,.3); }}
.fm-warn {{ color: var(--fm-warn); background: var(--fm-warn-soft); border-color: rgba(230,192,123,.3); }}
.fm-err {{ color: #F2A7C0; background: rgba(242,167,192,.13); border-color: rgba(242,167,192,.3); }}
.fm-na, .fm-none {{ color: var(--fm-na); background: var(--fm-na-soft); border-color: rgba(163,172,189,.2); }}
.fm-run {{ color: var(--fm-accent-text); background: var(--fm-accent-soft); border-color: var(--fm-accent-line); }}
.fm-pill.sm {{ min-width: 2.5rem; padding: 0 .4rem; font-size: .7rem; line-height: 1.3rem; }}
.fm-dash {{ color: var(--fm-faint); }}

/* ---- metric cards and the preview sentence ----------------------------------------------------------- */
.fm-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: .9rem; margin: .2rem 0 1rem; }}
.fm-metric {{ background: var(--fm-surface); border: 1px solid var(--fm-border); border-radius: 14px; padding: .95rem 1rem .85rem;
  text-align: center; box-shadow: var(--fm-shadow); }}
.fm-metric .v {{ font-size: 1.85rem; font-weight: 650; color: var(--fm-text); line-height: 1.1; font-variant-numeric: tabular-nums; }}
.fm-metric .l {{ color: var(--fm-muted); font-size: .8rem; margin-top: .3rem; letter-spacing: .02em; }}
.fm-metric .s {{ color: var(--fm-faint); font-size: .74rem; margin-top: .15rem; line-height: 1.35; overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }}
.fm-metric.accent .v {{ color: var(--fm-accent-text); }}
.fm-sentence {{ text-align: center; color: var(--fm-muted); font-size: .92rem; line-height: 1.6; margin: .2rem 0 1.2rem; }}
.fm-sentence b {{ color: var(--fm-text); font-weight: 650; }}
.fm-sentence .eq {{ color: var(--fm-accent-text); }}
.fm-sentence .est {{ color: var(--fm-faint); font-size: .82rem; }}

/* ---- callouts ---------------------------------------------------------------------------------------- */
.fm-callout {{ border-radius: 12px; padding: .75rem 1rem; margin: .2rem 0 1rem; border: 1px solid var(--fm-border);
  background: var(--fm-surface); font-size: .9rem; display: flex; gap: .7rem; align-items: flex-start; }}
.fm-callout .ic {{ font-weight: 800; width: 1.4rem; height: 1.4rem; border-radius: 50%; display: grid; place-items: center;
  font-size: .8rem; flex: none; margin-top: .05rem; }}
.fm-callout.error {{ border-color: rgba(238,142,142,.35); background: linear-gradient(0deg, var(--fm-fail-soft), var(--fm-fail-soft)), var(--fm-surface); }}
.fm-callout.error .ic {{ background: var(--fm-fail); color: #2A2025; }}
.fm-callout.info {{ border-color: var(--fm-accent-line); background: linear-gradient(0deg, var(--fm-accent-soft), var(--fm-accent-soft)), var(--fm-surface); }}
.fm-callout.info .ic {{ background: var(--fm-accent); color: #1C2130; }}
.fm-callout.warn {{ border-color: rgba(230,192,123,.35); background: linear-gradient(0deg, var(--fm-warn-soft), var(--fm-warn-soft)), var(--fm-surface); }}
.fm-callout.warn .ic {{ background: var(--fm-warn); color: #2A2517; }}
.fm-callout.ok {{ border-color: rgba(127,203,155,.35); background: linear-gradient(0deg, var(--fm-pass-soft), var(--fm-pass-soft)), var(--fm-surface); }}
.fm-callout.ok .ic {{ background: var(--fm-pass); color: #17261D; }}
.fm-callout ul {{ margin: .25rem 0 0 1rem; padding: 0; }}

/* ---- cards: rule guide, rule results, NAV / holdings / trading --------------------------------------- */
.fm-h2 {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: .15rem .6rem; margin: 1.4rem 0 .7rem; }}
.fm-h2 h2 {{ font-size: 1.12rem; font-weight: 640; margin: 0; padding: 0; color: var(--fm-text); white-space: nowrap; }}
.fm-h2 span {{ color: var(--fm-faint); font-size: .82rem; }}
.fm-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.1rem; align-items: start; }}
.fm-cell {{ min-width: 0; }}
.fm-cell.wide {{ grid-column: 1 / -1; }}
@media (max-width: 1100px) {{ .fm-grid {{ grid-template-columns: minmax(0, 1fr); }} }}
.fm-split {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: start; }}
.fm-split > div + div {{ border-left: 1px solid var(--fm-border); }}
@media (max-width: 900px) {{ .fm-split {{ grid-template-columns: minmax(0, 1fr); }} .fm-split > div + div {{ border-left: none; }} }}
.fm-card {{ background: var(--fm-surface); border: 1px solid var(--fm-border); border-radius: 14px; overflow: hidden;
  box-shadow: var(--fm-shadow); }}
.fm-band {{ background: linear-gradient(180deg, #3E4F7D, var(--fm-band)); color: #EEF2FB; padding: .5rem 1rem;
  display: flex; align-items: center; justify-content: space-between; gap: .6rem; font-weight: 620; font-size: .93rem;
  min-height: 2.85rem; box-sizing: border-box; }}
.fm-band .fm-pill {{ background: var(--fm-surface); font-size: .7rem; line-height: 1.25rem; flex: none; }}
.fm-band .sub {{ color: #C3CEEA; font-weight: 450; font-size: .78rem; }}
.fm-band.plain {{ background: var(--fm-surface-2); color: var(--fm-text); border-bottom: 1px solid var(--fm-border); }}
.fm-body {{ padding: .3rem 0 .2rem; }}
.fm-card p.fm-desc {{ color: var(--fm-muted); font-size: .86rem; margin: .7rem 1rem .4rem; line-height: 1.5; }}
.fm-card ul.fm-conds {{ margin: .2rem 1rem .6rem 1.1rem; padding: 0 0 0 .8rem; color: var(--fm-text); font-size: .84rem; line-height: 1.55; }}
.fm-card ul.fm-conds li::marker {{ color: var(--fm-accent); }}
.fm-card .fm-applies {{ margin: 0 1rem .85rem; color: var(--fm-muted); font-size: .8rem; }}
table.fm-kv {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
table.fm-kv th {{ text-align: left; font-weight: 560; color: var(--fm-accent-text); padding: .45rem 1rem; width: auto;
  border-bottom: 1px solid rgba(54,63,82,.7); vertical-align: top; }}
table.fm-kv td {{ text-align: right; padding: .45rem 1rem; color: var(--fm-text); border-bottom: 1px solid rgba(54,63,82,.7);
  overflow-wrap: anywhere; }}
table.fm-kv td.nw {{ white-space: nowrap; overflow-wrap: normal; }}
table.fm-kv th {{ min-width: 7rem; }}
table.fm-kv td:has(> .long) {{ width: 58%; }}
table.fm-kv td.num, table.fm-table td.num {{ font-variant-numeric: tabular-nums; }}
table.fm-kv tr:last-child th, table.fm-kv tr:last-child td {{ border-bottom: none; }}
table.fm-kv tr:hover td, table.fm-kv tr:hover th {{ background: rgba(124,156,245,.04); }}
table.fm-kv td.v-pass {{ color: var(--fm-pass); font-weight: 600; }}
table.fm-kv td.v-fail {{ color: var(--fm-fail); font-weight: 600; }}
table.fm-kv td.v-warn {{ color: var(--fm-warn); font-weight: 600; }}
table.fm-kv td.v-na {{ color: var(--fm-na); }}
table.fm-kv td .why {{ display: block; color: var(--fm-muted); font-size: .76rem; font-weight: 400; }}
table.fm-kv td .long {{ display: block; max-height: 7.5em; overflow: auto; text-align: left; font-size: .78rem; color: var(--fm-muted); }}
.fm-sub {{ padding: .55rem 1rem .25rem; color: var(--fm-muted); font-size: .74rem; font-weight: 650; letter-spacing: .07em;
  text-transform: uppercase; border-top: 1px solid var(--fm-border); }}
.fm-sub.first {{ border-top: none; }}
.fm-card details {{ border-top: 1px solid var(--fm-border); }}
.fm-card details > summary {{ cursor: pointer; padding: .5rem 1rem; color: var(--fm-muted); font-size: .8rem; list-style: none; }}
.fm-card details > summary::-webkit-details-marker {{ display: none; }}
.fm-card details > summary::before {{ content: "\\25B8"; display: inline-block; margin-right: .45rem; transition: transform .15s; color: var(--fm-accent); }}
.fm-card details[open] > summary::before {{ transform: rotate(90deg); }}
.fm-notes {{ margin: .1rem 1rem .6rem; padding: .45rem .65rem; border-left: 2px solid var(--fm-accent-line); background: var(--fm-accent-soft);
  border-radius: 0 8px 8px 0; color: var(--fm-muted); font-size: .78rem; }}
.fm-foot {{ padding: .6rem 1rem .8rem; border-top: 1px solid var(--fm-border); background: rgba(31,36,48,.35); }}
.fm-foot .t {{ color: var(--fm-muted); font-size: .76rem; font-weight: 600; margin-bottom: .2rem; }}
.fm-card details.fm-foot {{ padding: 0 1rem .1rem; }}
.fm-card details.fm-foot > summary.t {{ padding: .55rem 0; margin: 0; font-size: .76rem; }}
.fm-card details.fm-foot[open] {{ padding-bottom: .8rem; }}
.fm-foot p {{ margin: .15rem 0; color: var(--fm-muted); font-size: .78rem; line-height: 1.5; }}
.fm-reason {{ margin: .7rem 1rem; color: var(--fm-muted); font-size: .84rem; }}
.fm-stats {{ display: flex; flex-wrap: wrap; gap: .45rem; padding: .7rem 1rem .4rem; }}
.fm-stat {{ background: var(--fm-surface-2); border: 1px solid var(--fm-border); border-radius: 9px; padding: .3rem .6rem; font-size: .78rem; color: var(--fm-muted); }}
.fm-stat b {{ color: var(--fm-text); margin-left: .3rem; font-variant-numeric: tabular-nums; }}
.fm-stat.bad b {{ color: var(--fm-fail); }} .fm-stat.warnish b {{ color: var(--fm-warn); }}

/* ---- data tables (matrix, history, GICS, discrepancies, trading) ------------------------------------- */
/* A wide table scrolls sideways; a soft shadow at the right edge says there is more (it is covered once the
   table is scrolled to its end: the cover scrolls with the content, the shadow stays with the box). */
.fm-tablewrap {{ overflow-x: auto;
  background: linear-gradient(to left, var(--fm-surface) 40%, rgba(38,45,59,0)) right center / 2.5rem 100% no-repeat local,
              radial-gradient(farthest-side at 100% 50%, rgba(8,10,16,.55), rgba(8,10,16,0)) right center / .9rem 100% no-repeat scroll; }}
table.fm-table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: .82rem; }}
table.fm-table th {{ text-align: left; color: var(--fm-muted); font-weight: 600; font-size: .74rem; letter-spacing: .04em;
  padding: .5rem .75rem; background: var(--fm-surface-2); border-bottom: 1px solid var(--fm-border); white-space: nowrap; }}
table.fm-table td {{ padding: .42rem .75rem; border-bottom: 1px solid rgba(54,63,82,.6); color: var(--fm-text);
  vertical-align: middle; }}
.fm-card table.fm-table th:first-child, .fm-card table.fm-table td:first-child {{ padding-left: 1rem; }}
.fm-card table.fm-table th:last-child, .fm-card table.fm-table td:last-child {{ padding-right: 1rem; }}
table.fm-table tr:last-child td {{ border-bottom: none; }}
table.fm-table tbody tr:hover td {{ background: rgba(124,156,245,.05); }}
table.fm-table td.num, table.fm-table th.num {{ text-align: right; }}
table.fm-table td.c, table.fm-table th.c {{ text-align: center; }}
table.fm-matrix th {{ white-space: normal; text-align: center; padding: .5rem .2rem; line-height: 1.2; font-size: .7rem; }}
table.fm-matrix th:first-child {{ text-align: left; padding-left: .9rem; }}
table.fm-matrix td {{ padding: .38rem .15rem; }}
table.fm-matrix th:first-child, table.fm-matrix td:first-child {{ position: sticky; left: 0; z-index: 1;
  background: var(--fm-surface); box-shadow: inset -1px 0 0 var(--fm-border); }}
table.fm-matrix th:first-child {{ background: var(--fm-surface-2); }}
table.fm-history td {{ font-size: .8rem; }}
table.fm-history td:nth-child(1), table.fm-history td:nth-child(3) {{ white-space: nowrap; }}
table.fm-history td:nth-child(4), table.fm-history th:nth-child(4) {{ text-align: left; }}
table.fm-history .nw {{ white-space: nowrap; }}
table.fm-history td:nth-child(8) {{ color: var(--fm-faint); font-size: .76rem; overflow-wrap: anywhere; }}
@media (max-width: 1100px) {{ table.fm-history th:nth-child(8), table.fm-history td:nth-child(8) {{ display: none; }}
  table.fm-history th, table.fm-history td {{ padding-left: .5rem; padding-right: .5rem; }}
  table.fm-history td:nth-child(5) {{ min-width: 11rem; }} }}
table.fm-matrix td:first-child {{ font-weight: 650; letter-spacing: .03em; padding-left: .9rem; }}
table.fm-table td.fund {{ font-weight: 650; letter-spacing: .03em; }}
table.fm-table tr.hl-bad td {{ background: var(--fm-fail-soft); }}
table.fm-table tr.hl-good td {{ background: var(--fm-pass-soft); }}
table.fm-table tr.hl-breach td {{ background: var(--fm-warn-soft); }}
table.fm-table tr.total td {{ font-weight: 650; background: var(--fm-surface-2); }}
table.fm-table td.muted {{ color: var(--fm-muted); }}
table.fm-table td .sub {{ display: block; color: var(--fm-faint); font-size: .72rem; }}
.fm-card .fm-table th {{ background: rgba(44,52,70,.6); }}
.fm-tablecap {{ padding: .45rem 1rem .6rem; color: var(--fm-faint); font-size: .74rem; }}
.fm-tabletitle {{ padding: .6rem 1rem .35rem; color: var(--fm-muted); font-size: .8rem; font-weight: 600; }}
.fm-panel {{ background: var(--fm-surface); border: 1px solid var(--fm-border); border-radius: 14px; overflow: hidden; box-shadow: var(--fm-shadow); margin-bottom: 1rem; }}

/* ---- run banner, stepper, progress ------------------------------------------------------------------- */
.fm-banner {{ display: flex; align-items: center; gap: 1rem; padding: .9rem 1.1rem; border-radius: 14px; margin: .2rem 0 1rem;
  background: var(--fm-surface); border: 1px solid var(--fm-border); box-shadow: var(--fm-shadow); }}
.fm-banner > .fm-pill {{ font-size: .8rem; padding: .15rem .8rem; }}
.fm-banner .t {{ font-weight: 620; color: var(--fm-text); }}
.fm-banner .m {{ color: var(--fm-faint); font-size: .8rem; margin-top: .1rem; }}
.fm-steps {{ display: flex; align-items: center; gap: .35rem; flex-wrap: wrap; margin: .6rem 0 .4rem; }}
.fm-step {{ display: flex; align-items: center; gap: .4rem; color: var(--fm-faint); font-size: .8rem; }}
.fm-step .dot {{ width: 1.35rem; height: 1.35rem; border-radius: 50%; display: grid; place-items: center; font-size: .7rem;
  border: 1px solid var(--fm-border); background: var(--fm-surface-2); color: var(--fm-faint); }}
.fm-step.done {{ color: var(--fm-muted); }}
.fm-step.done .dot {{ background: var(--fm-pass-soft); border-color: rgba(127,203,155,.4); color: var(--fm-pass); }}
.fm-step.now {{ color: var(--fm-text); font-weight: 600; }}
.fm-step.now .dot {{ background: var(--fm-accent-soft); border-color: var(--fm-accent); color: var(--fm-accent-text);
  box-shadow: 0 0 0 4px rgba(124,156,245,.12); }}
.fm-step-line {{ width: 1.4rem; height: 1px; background: var(--fm-border); }}
.fm-live {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }}
.fm-live .t {{ font-size: 1.05rem; font-weight: 640; }}
.fm-live .m {{ color: var(--fm-faint); font-size: .8rem; }}
.fm-now {{ color: var(--fm-muted); font-size: .86rem; margin: .2rem 0 .1rem; }}
.fm-now b {{ color: var(--fm-accent-text); }}
.st-key-nr_progress {{ background: var(--fm-surface); border: 1px solid var(--fm-accent-line) !important; border-radius: 16px;
  padding: 1rem 1.15rem; box-shadow: 0 0 0 4px rgba(124,156,245,.06), var(--fm-shadow); }}
.fm-empty {{ text-align: center; color: var(--fm-faint); padding: 2.2rem 1rem; border: 1px dashed var(--fm-border); border-radius: 14px; }}
.fm-empty b {{ display: block; color: var(--fm-muted); font-size: 1rem; margin-bottom: .3rem; }}
.fm-files {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: .5rem; }}
"""


# The Trading Compliance dashboard (ui.trading_results, ui.trading_drilldown, ui.charts). Plain CSS (no f-string).
TRADING_CSS = """
/* ---- trading page: preview ---------------------------------------------------------------------------- */
.fm-passes { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); gap: .8rem; align-items: stretch;
  margin: 0 0 1rem; }
.fm-passes > div:not(.arrow) { background: var(--fm-surface); border: 1px solid var(--fm-border); border-radius: 12px;
  padding: .7rem .9rem; font-size: .84rem; color: var(--fm-muted); line-height: 1.5; }
.fm-passes b { display: block; color: var(--fm-text); font-weight: 640; margin-bottom: .15rem; }
.fm-passes .arrow { align-self: center; color: var(--fm-accent-text); font-size: 1.2rem; }
@media (max-width: 900px) { .fm-passes { grid-template-columns: minmax(0, 1fr); } .fm-passes .arrow { display: none; } }
.fm-nextrun { display: flex; flex-wrap: wrap; gap: .3rem .8rem; align-items: baseline; padding: .55rem .9rem;
  border: 1px solid var(--fm-border); border-radius: 12px; background: var(--fm-surface); margin: .2rem 0 1rem;
  font-size: .84rem; color: var(--fm-muted); }
.fm-nextrun > span:first-child { text-transform: uppercase; letter-spacing: .07em; font-size: .7rem; font-weight: 650;
  color: var(--fm-faint); }
.fm-nextrun b { color: var(--fm-text); font-weight: 600; }
.fm-nextrun .ok { color: var(--fm-pass); } .fm-nextrun .bad { color: var(--fm-fail); }

/* ---- KPI strip --------------------------------------------------------------------------------------- */
.fm-kpis { margin: .1rem 0 .6rem; }
.fm-kpi-group { color: var(--fm-muted); font-size: .7rem; font-weight: 650; letter-spacing: .08em; text-transform: uppercase;
  margin: .4rem 0 .45rem; }
.fm-kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(122px, 1fr)); gap: .7rem; margin-bottom: .7rem; }
/* Both rows share one five-column grid when wide, so the tile edges line up (an error tile wraps to a new row). */
@media (min-width: 1200px) { .fm-kpi-row { grid-template-columns: repeat(5, minmax(0, 1fr)); } }
.fm-kpi { background: var(--fm-surface); border: 1px solid var(--fm-border); border-radius: 14px; padding: .75rem .9rem .7rem;
  box-shadow: var(--fm-shadow); border-top: 3px solid var(--fm-border); min-width: 0; }
.fm-kpi .l { color: var(--fm-muted); font-size: .78rem; font-weight: 550; display: flex; align-items: center; gap: .4rem; }
.fm-kpi .ic { width: 1.15rem; height: 1.15rem; border-radius: 50%; display: inline-grid; place-items: center; font-size: .68rem;
  font-weight: 800; background: var(--fm-surface-3); color: var(--fm-muted); flex: none; }
.fm-kpi .v { font-size: 1.6rem; font-weight: 650; color: var(--fm-text); line-height: 1.2; margin-top: .3rem; white-space: nowrap; }
.fm-kpi .s { color: var(--fm-faint); font-size: .74rem; margin-top: .2rem; line-height: 1.35; }
.fm-kpi.ok { border-top-color: var(--fm-pass); } .fm-kpi.ok .ic { background: var(--fm-pass); color: #17261D; }
.fm-kpi.cures { border-top-color: var(--fm-pass); } .fm-kpi.cures .ic { background: var(--fm-pass); color: #17261D; }
.fm-kpi.breaks { border-top-color: var(--fm-fail); background: linear-gradient(0deg, var(--fm-fail-soft), var(--fm-fail-soft)), var(--fm-surface); }
.fm-kpi.breaks .ic { background: var(--fm-fail); color: #2A1E22; }
.fm-kpi.breaks .s { color: var(--fm-muted); }  /* FAINT on the red tint is under 4.5:1 */
.fm-kpi.worse { border-top-color: var(--fm-fail); } .fm-kpi.worse .ic { background: var(--fm-warn); color: #2A2517; }
/* Negative cash is red everywhere (table, charts, header); amber stays for STILL FAILS. */
.fm-kpi.neg { border-top-color: var(--fm-fail); } .fm-kpi.neg .ic { background: var(--fm-fail-soft); color: var(--fm-fail); }
.fm-dq .fm-callout, .fm-dq .fm-callout ul, .fm-dq .fm-callout li { font-size: .84rem !important; line-height: 1.45; }
.fm-dq .fm-callout li { margin: .1rem 0; }
.fm-beforeafter { margin: -.45rem 0 .7rem; color: var(--fm-muted); font-size: .8rem; }
.fm-kpi.still { border-top-color: var(--fm-warn); } .fm-kpi.still .ic { background: var(--fm-warn); color: #2A2517; }
.fm-kpi.error { border-top-color: #F2A7C0; } .fm-kpi.error .ic { background: #F2A7C0; color: #2A1E25; }
/* Results that rest on missing trade-date data: counted apart, dashed, never a loud verdict. */
.fm-kpi.miss { border-top: 3px dashed var(--fm-warn); }
.fm-kpi.miss .ic { background: var(--fm-warn-soft); color: var(--fm-warn); }
@media (max-width: 1500px) { .fm-kpi .l { min-height: 2.5em; align-items: flex-start; } .fm-kpi .v { font-size: 1.25rem; }
  .fm-kpi { padding: .7rem .75rem .65rem; } }

/* ---- transition chips (matrix cells, legend, rule cards) --------------------------------------------- */
.fm-tx { display: inline-flex; align-items: center; gap: .25rem; padding: .05rem .5rem; border-radius: 999px; font-size: .68rem;
  font-weight: 700; letter-spacing: .04em; line-height: 1.35rem; white-space: nowrap; border: 1px solid transparent; }
.fm-tx b { font-size: .7rem; }
.fm-tx.breaks { background: var(--fm-fail); color: #2A1E22; box-shadow: 0 0 0 3px rgba(238,142,142,.16); }
.fm-tx.fails { color: var(--fm-fail); background: var(--fm-fail-soft); border-color: rgba(238,142,142,.32); }
.fm-tx.cures { color: var(--fm-pass); background: var(--fm-pass-soft); border-color: rgba(127,203,155,.45); }
.fm-tx.still { color: var(--fm-warn); background: var(--fm-warn-soft); border-color: rgba(230,192,123,.32); }
.fm-tx .tr { font-style: normal; font-weight: 650; letter-spacing: .02em; margin-left: .1rem; text-transform: none; }
.fm-tx.still.worse { border-color: rgba(238,142,142,.75); box-shadow: inset 3px 0 0 var(--fm-fail); }
.fm-tx.still.worse .tr { color: var(--fm-fail); }
.fm-tx.still.better .tr { color: var(--fm-pass); }
.fm-tx.still.moves .tr { color: var(--fm-muted); }
.fm-tx.still.same { opacity: .82; }
.fm-tx.error { color: #F2A7C0; background: rgba(242,167,192,.13); border-color: rgba(242,167,192,.3); }
.fm-tx.ok { color: var(--fm-pass); background: var(--fm-pass-soft); border-color: rgba(127,203,155,.22); }
.fm-tx.na { color: var(--fm-na); background: var(--fm-na-soft); border-color: rgba(163,172,189,.2); }
.fm-tx.quiet { background: none; border: none; box-shadow: none; padding: 0; font-weight: 600; }
.fm-tx.ok.quiet { color: rgba(127,203,155,.75); font-size: .85rem; }
.fm-tx.na.quiet { color: var(--fm-faint); font-weight: 500; font-size: .7rem; letter-spacing: .02em; }
.fm-tx.impact { min-width: 5.6rem; justify-content: center; }
.fm-tx.impact.ok { background: none; border-color: rgba(127,203,155,.35); }
.fm-tx.miss { background: none !important; box-shadow: none !important; border-style: dashed; opacity: .78; }
.fm-tx.miss.breaks, .fm-tx.miss.fails { color: var(--fm-fail); border-color: rgba(238,142,142,.6); }
.fm-tx.miss.quiet { border: none; opacity: .5; }
.fm-tx.impact.miss { color: var(--fm-warn); border-color: rgba(230,192,123,.55); opacity: 1; }
.fm-tx .ms { font-style: normal; font-weight: 700; color: var(--fm-warn); margin-left: .15rem; }
table.fm-impact td { padding-top: .32rem; padding-bottom: .32rem; }
table.fm-impact th:nth-child(2), table.fm-impact td:nth-child(2) { padding-right: .6rem; border-right: 1px solid var(--fm-border); }
.fm-txlegend { display: flex; flex-wrap: wrap; gap: .45rem 1.1rem; padding: .7rem 1rem .5rem; border-top: 1px solid var(--fm-border); }
.fm-txlegend > span { display: inline-flex; align-items: center; gap: .45rem; }
.fm-txlegend em { font-style: normal; color: var(--fm-muted); font-size: .76rem; }
.fm-txlegend .hint { flex-basis: 100%; color: var(--fm-muted); font-size: .74rem; }
/* Clickable cells and chips open the fund (and rule) in the drill-down tab. */
table.fm-impact td.jump { cursor: pointer; }
table.fm-impact tbody td.jump:hover { background: rgba(124,156,245,.12) !important; }
table.fm-impact td.jump:focus-visible, .fm-risk .funds span:focus-visible { outline: 2px solid var(--fm-accent); outline-offset: -2px; }
table.fm-impact td.fund.jump:hover { color: var(--fm-accent-text); text-decoration: underline; text-underline-offset: 3px; }
[class*="st-key-tcjump"] { display: none !important; }

/* ---- rules at risk ------------------------------------------------------------------------------------ */
.fm-risks { display: grid; grid-template-columns: minmax(0, 1fr); }
.fm-risk { display: grid; grid-template-columns: minmax(11rem, 15rem) minmax(0, 1fr); gap: .4rem 1.1rem; padding: .7rem 1rem;
  border-bottom: 1px solid rgba(54,63,82,.6); }
.fm-risk:last-child { border-bottom: none; }
.fm-risk .head b { display: block; color: var(--fm-text); font-weight: 620; font-size: .88rem; }
.fm-risk .head > span { color: var(--fm-faint); font-size: .74rem; }
.fm-risk .bar { display: flex; gap: 2px; height: 6px; border-radius: 999px; background: var(--fm-surface-3); margin-top: .4rem;
  overflow: hidden; }
.fm-risk .bar i { display: block; height: 100%; }
.fm-risk .bar i.breaks, .fm-risk .bar i.fails { background: var(--fm-fail); }
.fm-risk .bar i.still { background: var(--fm-warn); } .fm-risk .bar i.cures { background: var(--fm-pass); }
.fm-risk .bar i.error { background: #F2A7C0; }
.fm-risk .bar i.worse { background: repeating-linear-gradient(135deg, var(--fm-warn) 0 3px, var(--fm-fail) 3px 6px); }
.fm-risk .bar i.miss { background: repeating-linear-gradient(90deg, var(--fm-na) 0 3px, transparent 3px 5px);
  opacity: .7; }
.fm-risk .body { display: grid; gap: .35rem; align-content: center; }
.fm-risk .grp { display: flex; gap: .55rem; align-items: flex-start; }
.fm-risk .grp em { font-style: normal; flex: none; min-width: 6.4rem; }
.fm-risk .funds { display: flex; flex-wrap: wrap; gap: .3rem; }
.fm-risk .funds span { font-size: .74rem; font-weight: 600; letter-spacing: .03em; color: var(--fm-text); padding: .05rem .45rem;
  border-radius: 6px; background: var(--fm-surface-2); border: 1px solid var(--fm-border); cursor: pointer;
  transition: border-color .12s ease, background .12s ease; }
.fm-risk .funds span:hover { border-color: var(--fm-accent-line); background: var(--fm-accent-soft); }
.fm-risk .funds span b { color: var(--fm-fail); font-size: .62rem; margin-right: .25rem; }
.fm-risk .funds span.worse { border-color: rgba(238,142,142,.55); }
.fm-risk .funds span .miss { font-style: normal; color: var(--fm-warn); margin-left: .2rem; }
.fm-risk .funds span.gap { border-style: dashed !important; border-color: rgba(163,172,189,.5) !important;
  color: var(--fm-muted); }
.fm-risk .grp.breaks .funds span { border-color: rgba(238,142,142,.5); }
.fm-risk .grp.cures .funds span { border-color: rgba(127,203,155,.45); }
@media (max-width: 900px) { .fm-risk { grid-template-columns: minmax(0, 1fr); } }

/* ---- trading summary table ------------------------------------------------------------------------------ */
table.fm-summary { font-size: .78rem; }
table.fm-summary th { font-size: .7rem; padding: .4rem .5rem; text-align: right; line-height: 1.25; }
table.fm-summary th span { color: var(--fm-faint); font-weight: 500; letter-spacing: 0; }
table.fm-summary thead tr.top th { border-bottom: none; padding-bottom: .1rem; }
/* Only the real first column is sticky (not the first cell of the second header row). */
table.fm-summary thead tr.top th:first-child, table.fm-summary tbody td:first-child { position: sticky; left: 0; z-index: 1;
  text-align: left; background: var(--fm-surface); box-shadow: inset -1px 0 0 var(--fm-border); }
table.fm-summary thead tr.top th:first-child { background: var(--fm-surface-2); z-index: 4; }
table.fm-summary thead tr:not(.top) th { text-align: right; }
/* The BREAKS tint over an opaque surface, so scrolled cells never show through the sticky fund name. */
table.fm-matrix tbody tr:hover td:first-child, table.fm-summary tbody tr:hover td:first-child {
  background: linear-gradient(rgba(124,156,245,.05), rgba(124,156,245,.05)), var(--fm-surface); }
table.fm-matrix tr.hl-bad td:first-child,
table.fm-summary tr.hl-bad td:first-child { background: linear-gradient(var(--fm-fail-soft), var(--fm-fail-soft)), var(--fm-surface); }
/* The table scrolls inside a capped box (header pinned), so its sideways scrollbar is on screen. */
.fm-sumwrap { max-height: 34rem; overflow: auto; }
.fm-sumwrap thead { position: sticky; top: 0; z-index: 3; }
.fm-sumwrap thead th { background: var(--fm-surface-2) !important; }
table.fm-summary th.c { text-align: center; }
table.fm-summary td { padding: .38rem .5rem; white-space: nowrap; font-variant-numeric: tabular-nums; }
table.fm-summary td.num { text-align: right; }
table.fm-summary td .sub { display: block; color: var(--fm-faint); font-size: .68rem; }
table.fm-table tr.hl-bad td .sub, table.fm-cmp tr.chg .sub { color: var(--fm-muted); }  /* FAINT on tints < 4.5:1 */
table.fm-summary td.muted .sub { color: var(--fm-muted); }
table.fm-summary td.flagged s { color: var(--fm-faint); text-decoration-color: rgba(146,156,176,.7); }
table.fm-summary .grp { border-left: 1px solid rgba(54,63,82,.85); }
table.fm-summary td.neg { color: var(--fm-fail); font-weight: 600; }
table.fm-summary tr.total td { border-top: 1px solid var(--fm-border); }
table.fm-summary tr.total td:first-child { background: var(--fm-surface-2); }
.fm-ico { display: inline-grid; place-items: center; min-width: 1.05rem; height: 1.05rem; padding: 0 .2rem; margin-left: .35rem;
  border-radius: 999px; font-size: .64rem; font-weight: 750; vertical-align: 1px; cursor: default; letter-spacing: 0; }
.fm-ico.bad { color: var(--fm-fail); background: var(--fm-fail-soft); border: 1px solid rgba(238,142,142,.35); }
.fm-ico.warn { color: var(--fm-warn); background: var(--fm-warn-soft); border: 1px solid rgba(230,192,123,.35); }
.fm-ico.na { color: var(--fm-muted); background: var(--fm-na-soft); border: 1px solid rgba(163,172,189,.3); }
.fm-flag { display: inline-block; font-size: .68rem; font-weight: 600; padding: .02rem .45rem; border-radius: 999px; margin-right: .25rem;
  border: 1px solid transparent; white-space: nowrap; cursor: default; }
.fm-flag.bad { color: var(--fm-fail); background: var(--fm-fail-soft); border-color: rgba(238,142,142,.3); }
.fm-flag.warn { color: var(--fm-warn); background: var(--fm-warn-soft); border-color: rgba(230,192,123,.3); }
.fm-flag.na { color: var(--fm-na); background: var(--fm-na-soft); border-color: rgba(163,172,189,.2); }
.fm-defs p b { color: var(--fm-text); font-weight: 600; }

/* ---- charts (inline SVG, rendered through Markdown) ---------------------------------------------------- */
.fm-charts-wrap { container-type: inline-size; margin: 0 0 1rem; }
.fm-charts { display: grid; grid-template-columns: minmax(0, 1fr); gap: 1rem; }
@container (min-width: 620px) { .fm-charts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .fm-charts > .fm-chart:nth-child(3) { grid-column: 1 / -1; } }
@container (min-width: 1100px) { .fm-charts { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .fm-charts > .fm-chart:nth-child(3) { grid-column: auto; } }
.fm-chart { margin: 0; background: var(--fm-surface); border: 1px solid var(--fm-border); border-radius: 14px;
  padding: .8rem .8rem .7rem; box-shadow: var(--fm-shadow); min-width: 0; }
.fm-chart figcaption { margin-bottom: .35rem; }
.fm-chart figcaption b { display: block; color: var(--fm-text); font-size: .9rem; font-weight: 640; }
.fm-chart figcaption span { color: var(--fm-faint); font-size: .74rem; }
.fm-legend { display: flex; flex-wrap: wrap; gap: .25rem .9rem; margin: .1rem 0 .4rem; font-size: .74rem; color: var(--fm-muted); }
.fm-legend span { display: inline-flex; align-items: center; gap: .35rem; }
.fm-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; }
.fm-legend i.dot { border-radius: 50%; } .fm-legend i.ring { border-radius: 50%; border: 2px solid; box-sizing: border-box; }
.fm-chart-rows { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; column-gap: .45rem; }
.fm-chart-rows .plot { container-type: inline-size; }
/* A tick label that collides in a narrow plot (charts.WIDE_PLOT_PX) shows only when the plot is wide. */
.fm-chart svg text.tk-wide { display: none; }
@container (min-width: 240px) { .fm-chart svg text.tk-wide { display: inline; } }
.fm-chart-rows .names div, .fm-chart-rows .vals div { font-size: .72rem; line-height: 22px; white-space: nowrap; }
.fm-chart-rows .names div { color: var(--fm-muted); font-weight: 600; letter-spacing: .03em; }
.fm-chart-rows .vals div { color: var(--fm-text); text-align: right; font-variant-numeric: tabular-nums; }
.fm-chart-rows .vals div.neg { color: var(--fm-fail); font-weight: 600; }
.fm-chart-rows .vals .none { color: var(--fm-faint); font-size: .68rem; }
.fm-chart-plot { overflow: visible; }
.fm-chart-note { color: var(--fm-faint); font-size: .72rem; margin-top: .4rem; line-height: 1.4; }
.fm-chart svg text { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; font-variant-numeric: tabular-nums; }

/* ---- fund drill-down --------------------------------------------------------------------------------- */
.fm-fundhead { background: var(--fm-surface); border: 1px solid var(--fm-border); border-radius: 14px; box-shadow: var(--fm-shadow);
  padding: .75rem .4rem .35rem; margin-bottom: 1rem; }
.fm-fundhead .who { display: flex; align-items: center; gap: .7rem; padding: 0 .6rem; flex-wrap: wrap; }
.fm-fundhead .who .t { font-size: 1.25rem; font-weight: 700; letter-spacing: .03em; color: var(--fm-text); }
.fm-fundhead .who .chg { color: var(--fm-muted); font-size: .82rem; }
.fm-fundhead .who .chg.bad { color: var(--fm-fail); }
.fm-fundhead .fm-stats { padding-top: .55rem; }
.fm-wf { padding: .2rem 0 0; }
.fm-wf .fm-desc { margin: .6rem .9rem 0; font-size: .78rem; color: var(--fm-faint); }
.fm-wf .fm-chart { border: none; box-shadow: none; background: none; }
table.fm-cmp { width: 100%; border-collapse: collapse; font-size: .82rem; }
table.fm-cmp thead th { color: var(--fm-faint); font-size: .72rem; font-weight: 600; letter-spacing: .02em; white-space: nowrap;
  padding: .45rem 1rem .3rem; border-bottom: 1px solid var(--fm-border); }
table.fm-cmp tbody th { text-align: left; font-weight: 520; color: var(--fm-accent-text); padding: .4rem .8rem .4rem 1rem; vertical-align: top; }
table.fm-cmp td { text-align: right; padding: .4rem .8rem; color: var(--fm-text); font-variant-numeric: tabular-nums;
  width: 30%; white-space: nowrap; }
table.fm-cmp td:last-child { padding-right: 1rem; }
table.fm-cmp tbody tr { border-bottom: 1px solid rgba(54,63,82,.55); }
table.fm-cmp tbody tr:last-child { border-bottom: none; }
table.fm-cmp tr.chg th, table.fm-cmp tr.chg td { background: var(--fm-accent-soft); }
table.fm-cmp tr.chg th { box-shadow: inset 3px 0 0 var(--fm-accent); }
table.fm-cmp tr.chg td b { font-weight: 680; }
table.fm-cmp i { font-style: normal; font-size: .62rem; margin-right: .35rem; color: var(--fm-accent-text); }
.fm-card .fm-band .fm-band-tx { display: inline-flex; align-items: center; gap: .35rem; flex: none; }
.fm-card .fm-band .fm-band-tx .arrow { color: #C3CEEA; }
.fm-card .fm-band .fm-tx { margin-left: .3rem; }
.fm-card .fm-band .fm-tx.ok, .fm-card .fm-band .fm-tx.na, .fm-card .fm-band .fm-tx.still, .fm-card .fm-band .fm-tx.cures,
.fm-card .fm-band .fm-tx.fails, .fm-card .fm-band .fm-tx.error { background-color: var(--fm-surface); }
.fm-cmpcard.t-breaks { border-color: rgba(238,142,142,.6); box-shadow: 0 0 0 3px rgba(238,142,142,.1), var(--fm-shadow); }
.fm-cmpcard.t-cures { border-color: rgba(127,203,155,.5); }
.fm-cmpcard.t-still { border-color: rgba(230,192,123,.4); }
.fm-cmpcard.focus { border-color: var(--fm-accent); box-shadow: 0 0 0 3px rgba(124,156,245,.18), var(--fm-shadow); }
.fm-cmpcard .fm-band > span:first-child { min-width: 0; }
.fm-cmpcard .fm-band .sub { white-space: nowrap; }
.fm-keyfig { display: flex; flex-wrap: wrap; align-items: baseline; gap: .2rem .6rem; margin: .5rem 1rem .35rem;
  padding: .4rem .65rem; border-radius: 8px; background: var(--fm-surface-2); font-size: .8rem; color: var(--fm-muted); }
.fm-keyfig b { color: var(--fm-text); font-variant-numeric: tabular-nums; font-weight: 650; }
.fm-keyfig em { font-style: normal; color: var(--fm-muted); }
.fm-keyfig.worse { box-shadow: inset 3px 0 0 var(--fm-fail); } .fm-keyfig.worse em { color: var(--fm-fail); }
.fm-keyfig.better { box-shadow: inset 3px 0 0 var(--fm-pass); } .fm-keyfig.better em { color: var(--fm-pass); }
.fm-folded { margin-top: 1.1rem; }
.fm-folded details { border-top: none !important; }
.fm-blot { margin-bottom: 1rem; }
.fm-scroll { max-height: 26rem; overflow-y: auto; }
.fm-scroll thead th { position: sticky; top: 0; z-index: 2; background: var(--fm-surface-2) !important; }
table.fm-blotter td, table.fm-blotter th { white-space: nowrap; font-variant-numeric: tabular-nums; }
table.fm-blotter td .sub { display: inline; margin-left: .4rem; }
table.fm-blotter tr.out td { color: var(--fm-faint); }
.fm-side { font-size: .66rem; font-weight: 700; letter-spacing: .06em; padding: .05rem .45rem; border-radius: 6px; }
.fm-side.buy { color: var(--fm-accent-text); background: var(--fm-accent-soft); }
.fm-side.sell { color: var(--fm-muted); background: var(--fm-surface-3); }
"""


def inject() -> None:
    """Add the app's CSS to the page (style-only ``st.html`` takes no layout space)."""
    st.html(f"<style>{CSS}{TRADING_CSS}</style>")
