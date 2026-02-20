"""
CEO Appointment Scanner — v2
Scans SEC EDGAR 8-K Item 5.02 filings for a user-supplied ticker list,
extracts and separates annual/ongoing compensation from one-time hire awards.
"""

import streamlit as st
import requests
import json
import time
import re
import csv
import io
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from typing import Optional
import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CEO Comp Scanner",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Stylesheet — refined data-terminal aesthetic, dark with amber accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,300&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,300&display=swap');

:root {
  --bg:       #0e0f0f;
  --surface:  #161818;
  --surface2: #1e2020;
  --border:   #2a2d2d;
  --border2:  #383c3c;
  --amber:    #e8a829;
  --amber-dim:#a87820;
  --green:    #3dba7e;
  --red:      #e05252;
  --blue:     #5b9bd6;
  --text:     #e8e4dc;
  --muted:    #7a7d7d;
  --muted2:   #4a4d4d;
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Mono', monospace;
  background: var(--bg) !important;
  color: var(--text);
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] label {
  font-size: 0.65rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] .stSelectbox > div > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] textarea { min-height: 180px !important; }

.stSlider > div > div > div > div { background: var(--amber) !important; }
.stSlider > div > div > div { background: var(--border2) !important; }

.main .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

.app-header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 1.8rem 2.5rem 1.5rem;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
}
.app-title {
  font-family: 'Fraunces', serif;
  font-size: 2.2rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.02em;
  line-height: 1;
}
.app-title span { color: var(--amber); }
.app-subtitle {
  font-size: 0.62rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 0.35rem;
}
.header-badge {
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 0.3rem 0.7rem;
  border: 1px solid var(--amber-dim);
  color: var(--amber);
}

.metric-strip {
  display: flex;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}
.metric-block {
  flex: 1;
  padding: 1rem 1.5rem;
  border-right: 1px solid var(--border);
}
.metric-block:last-child { border-right: none; }
.metric-label {
  font-size: 0.58rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.3rem;
}
.metric-value {
  font-family: 'Fraunces', serif;
  font-size: 1.9rem;
  font-weight: 600;
  color: var(--amber);
  line-height: 1;
}
.metric-sub {
  font-size: 0.62rem;
  color: var(--muted);
  margin-top: 0.2rem;
}

.content-area { padding: 1.5rem 2rem 3rem; }

.section-head {
  font-size: 0.6rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--muted);
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.filing-card {
  background: var(--surface);
  border: 1px solid var(--border);
  margin-bottom: 1px;
}
.filing-card:hover { border-color: var(--amber-dim); }

.card-top {
  padding: 1rem 1.4rem 0.8rem;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  border-bottom: 1px solid var(--border);
}
.card-ticker {
  font-family: 'Fraunces', serif;
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--amber);
  line-height: 1;
}
.card-company { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; }
.card-exec { font-size: 0.9rem; color: var(--text); margin-top: 0.4rem; font-weight: 500; }
.card-meta { text-align: right; font-size: 0.65rem; color: var(--muted); line-height: 2; }

.appt-badge {
  display: inline-block;
  font-size: 0.58rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 0.15rem 0.5rem;
  border: 1px solid;
  margin-left: 0.4rem;
}
.badge-new      { color: var(--green); border-color: var(--green); }
.badge-interim  { color: var(--amber); border-color: var(--amber); }
.badge-promoted { color: var(--blue);  border-color: var(--blue);  }

.comp-tables {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  background: var(--border);
}
.comp-section { background: var(--surface); padding: 0.9rem 1.4rem 1rem; }
.comp-section-title {
  font-size: 0.58rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 0.7rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border2);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.comp-section-title.annual  { color: var(--green); }
.comp-section-title.onetime { color: var(--amber); }
.dot { width: 5px; height: 5px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dot-green { background: var(--green); }
.dot-amber { background: var(--amber); }

.comp-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0.28rem 0;
  border-bottom: 1px solid var(--border2);
  gap: 1rem;
}
.comp-row:last-child { border-bottom: none; }
.comp-key { font-size: 0.68rem; color: var(--muted); flex-shrink: 0; }
.comp-val { font-size: 0.75rem; color: var(--text); text-align: right; font-weight: 500; }
.comp-val.na    { color: var(--muted2); font-style: italic; font-weight: 400; font-size: 0.68rem; }
.comp-val.found { color: var(--text); }
.comp-val.large { color: var(--amber); }
.comp-note {
  font-size: 0.62rem; color: var(--muted); margin-top: 0.5rem;
  padding-top: 0.5rem; border-top: 1px solid var(--border2);
  line-height: 1.6; font-style: italic;
}
.summary-text {
  padding: 0.8rem 1.4rem; font-size: 0.73rem; color: var(--muted);
  line-height: 1.7; border-top: 1px solid var(--border); background: var(--surface2);
}

.ticker-pills {
  display: flex; flex-wrap: wrap; gap: 0.4rem;
  padding: 0.8rem 1.4rem; background: var(--surface2);
  border-bottom: 1px solid var(--border);
}
.ticker-pill { font-size: 0.65rem; padding: 0.18rem 0.5rem; letter-spacing: 0.08em; border: 1px solid var(--border2); color: var(--muted); }
.ticker-pill.resolved { border-color: var(--amber-dim); color: var(--amber); }
.ticker-pill.failed   { border-color: var(--red); color: var(--red); opacity: 0.7; }

.info-bar {
  background: var(--surface2); border-left: 3px solid var(--amber);
  padding: 0.7rem 1rem; font-size: 0.72rem; color: var(--muted);
  margin-bottom: 1rem; line-height: 1.6;
}
.error-bar {
  background: #1e0e0e; border-left: 3px solid var(--red);
  padding: 0.7rem 1rem; font-size: 0.72rem; color: var(--red); margin-bottom: 0.5rem;
}

.stButton > button {
  background: transparent !important; border: 1px solid var(--amber) !important;
  color: var(--amber) !important; border-radius: 0 !important;
  font-family: 'IBM Plex Mono', monospace !important; font-size: 0.7rem !important;
  letter-spacing: 0.15em !important; text-transform: uppercase !important;
  padding: 0.55rem 1.4rem !important; width: 100%;
}
.stButton > button:hover { background: var(--amber) !important; color: var(--bg) !important; }

.stDownloadButton > button {
  background: transparent !important; border: 1px solid var(--green) !important;
  color: var(--green) !important; border-radius: 0 !important;
  font-family: 'IBM Plex Mono', monospace !important; font-size: 0.68rem !important;
  letter-spacing: 0.12em !important; text-transform: uppercase !important;
  padding: 0.45rem 1rem !important;
}
.stDownloadButton > button:hover { background: var(--green) !important; color: var(--bg) !important; }

div[data-testid="stExpander"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 0 !important; margin-top: 1px;
}
.streamlit-expanderHeader {
  font-family: 'IBM Plex Mono', monospace !important; font-size: 0.65rem !important;
  letter-spacing: 0.12em !important; text-transform: uppercase !important;
  color: var(--muted) !important; background: var(--surface) !important;
}
div[data-testid="stExpanderDetails"] {
  background: var(--surface2) !important; border-top: 1px solid var(--border) !important;
}

.stProgress > div > div { background: var(--amber) !important; }
.stProgress > div { background: var(--border) !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }

.landing-card {
  background: var(--surface); border: 1px solid var(--border);
  border-top: 2px solid var(--amber); padding: 1.5rem;
}
.landing-card h3 {
  font-family: 'Fraunces', serif; font-size: 1rem; font-weight: 600;
  color: var(--text); margin: 0 0 0.5rem;
}
.landing-card p { font-size: 0.72rem; color: var(--muted); line-height: 1.7; margin: 0; }

.comp-table-wrap { overflow-x: auto; margin-top: 1rem; }
table.comp-table { width: 100%; border-collapse: collapse; font-size: 0.7rem; }
table.comp-table th {
  text-align: left; padding: 0.5rem 0.8rem; font-size: 0.58rem;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted);
  border-bottom: 1px solid var(--border2); white-space: nowrap; background: var(--surface2);
}
table.comp-table td {
  padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--border);
  color: var(--text); vertical-align: top;
}
table.comp-table tr:hover td { background: var(--surface2); }
table.comp-table td.na { color: var(--muted2); font-style: italic; }
table.comp-table td.tkr { color: var(--amber); font-weight: 600; }
table.comp-table td.grn { color: var(--green); }
table.comp-table td.amb { color: var(--amber); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EDGAR_BASE      = "https://www.sec.gov"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {
    "User-Agent": "CEO Compensation Scanner research@example.com",
    "Accept-Encoding": "gzip, deflate",
}
CEO_TITLES = [
    "chief executive officer", "president and chief executive",
    "principal executive officer",
]
APPT_WORDS = [
    "appoint", "hire", "named", "elected", "promoted",
    "effective", "will serve", "joined", "succeed",
]

# ─────────────────────────────────────────────────────────────────────────────
# EDGAR helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_ticker_map() -> dict:
    """Download and invert EDGAR's ticker->CIK map. Cached 1 hour."""
    try:
        resp = requests.get(TICKERS_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
        mapping = {}
        for entry in raw.values():
            ticker = entry.get("ticker", "").upper().strip()
            if ticker:
                cik_str = str(entry.get("cik_str", entry.get("cik", "0")))
                mapping[ticker] = {
                    "cik":     str(int(cik_str)) if cik_str.isdigit() else cik_str.lstrip("0") or "0",
                    "cik_pad": cik_str.zfill(10),
                    "title":   entry.get("title", ""),
                }
        return mapping
    except Exception as e:
        return {}


def resolve_tickers(tickers: list, ticker_map: dict):
    resolved, failed = {}, []
    for t in tickers:
        t = t.upper().strip()
        if not t:
            continue
        if t in ticker_map:
            resolved[t] = ticker_map[t]
        else:
            failed.append(t)
    return resolved, failed


def get_company_8ks(cik_pad: str, start_date: date, end_date: date) -> list:
    """Fetch 8-K filings from submissions API for a CIK within date range."""
    url = SUBMISSIONS_URL.format(cik=cik_pad)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        f = data.get("filings", {}).get("recent", {})
        forms  = f.get("form", [])
        dates  = f.get("filingDate", [])
        accs   = f.get("accessionNumber", [])
        pdocs  = f.get("primaryDocument", [])

        results = []
        for form, fdate, acc, pdoc in zip(forms, dates, accs, pdocs):
            if form not in ("8-K", "8-K/A"):
                continue
            try:
                fd = datetime.strptime(fdate, "%Y-%m-%d").date()
            except ValueError:
                continue
            if start_date <= fd <= end_date:
                results.append({
                    "form": form, "file_date": fdate,
                    "accession_no": acc, "primary_doc": pdoc,
                })
        return results
    except Exception:
        return []


def fetch_filing_text(cik: str, accession_no: str, primary_doc: str) -> str:
    """Fetch and clean the primary 8-K document."""
    acc_clean = accession_no.replace("-", "")
    if primary_doc:
        url = f"{EDGAR_BASE}/Archives/edgar/data/{cik}/{acc_clean}/{primary_doc}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code == 200 and len(resp.text) > 300:
                return _clean(resp.text)[:16000]
        except Exception:
            pass
    # Fall back to index listing
    idx_url = f"{EDGAR_BASE}/Archives/edgar/data/{cik}/{acc_clean}/"
    try:
        resp = requests.get(idx_url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return ""
        docs = re.findall(r'href="([^"]+\.(?:htm|txt))"', resp.text, re.IGNORECASE)
        primary = [d for d in docs if not any(x in d.lower() for x in
                   ["ex-","ex1","ex2","ex3","ex4","ex9","exhibit","xbrl",".xml","r1.","r2."])] or docs[:3]
        for doc in primary[:2]:
            doc_url = (f"{EDGAR_BASE}{doc}" if doc.startswith("/")
                       else f"{EDGAR_BASE}/Archives/edgar/data/{cik}/{acc_clean}/{doc.split('/')[-1]}")
            r2 = requests.get(doc_url, headers=HEADERS, timeout=12)
            if r2.status_code == 200 and len(r2.text) > 300:
                return _clean(r2.text)[:16000]
    except Exception:
        pass
    return ""


def _clean(raw: str) -> str:
    t = re.sub(r'<[^>]+>', ' ', raw)
    t = re.sub(r'&nbsp;', ' ', t)
    t = re.sub(r'&amp;', '&', t)
    t = re.sub(r'&#\d+;', ' ', t)
    t = re.sub(r'\s{3,}', '\n\n', t)
    return t.strip()


def is_ceo_appointment(text: str) -> bool:
    t = text.lower()
    return any(c in t for c in CEO_TITLES) and any(w in t for w in APPT_WORDS)


# ─────────────────────────────────────────────────────────────────────────────
# Claude extraction
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior executive compensation analyst specializing in SEC 8-K Item 5.02 filings.
Critically distinguish between:
1. ANNUAL/ONGOING compensation (recurring each year)
2. ONE-TIME/NEW HIRE awards (paid once to induce joining or replace forfeited awards)

Classification rules:
- Annual/Ongoing: base salary, annual target bonus, annual LTI equity program targets (state per year)
- One-Time/Hire: cash signing bonuses, make-whole awards explicitly replacing forfeited equity/bonuses from prior employer, inducement grants described as "initial" or "upon commencement" without recurring language, relocation allowances
- If a grant is described as both initial AND representative of the annual program, classify as annual/ongoing
Return ONLY valid JSON, no markdown."""

EXTRACTION_PROMPT = """Analyze this 8-K filing. Extract CEO appointment and compensation data.

Return ONLY this JSON structure:

{{
  "is_ceo_appointment": true/false,
  "executive_name": "string or null",
  "title": "exact title or null",
  "appointment_type": "new_hire|promotion|interim|successor|unknown",
  "effective_date": "YYYY-MM-DD or descriptive or null",
  "annual": {{
    "base_salary_usd": number or null,
    "target_bonus_pct_of_salary": number or null,
    "target_bonus_usd": number or null,
    "max_bonus_pct_of_salary": number or null,
    "max_bonus_usd": number or null,
    "annual_lti_target_usd": number or null,
    "annual_rsu_shares": number or null,
    "annual_option_shares": number or null,
    "annual_equity_notes": "string or null",
    "other_annual_benefits": "string or null"
  }},
  "onetime": {{
    "signing_bonus_cash_usd": number or null,
    "signing_bonus_conditions": "repayment/vesting conditions or null",
    "makewhole_cash_usd": number or null,
    "makewhole_cash_notes": "what forfeited awards replaced or null",
    "makewhole_equity_value_usd": number or null,
    "makewhole_equity_notes": "description of replaced equity or null",
    "onetime_rsu_shares": number or null,
    "onetime_rsu_value_usd": number or null,
    "onetime_rsu_vesting": "vesting schedule or null",
    "onetime_option_shares": number or null,
    "onetime_option_value_usd": number or null,
    "onetime_option_vesting": "vesting schedule or null",
    "onetime_option_strike_usd": number or null,
    "relocation_usd": number or null,
    "other_onetime_usd": number or null,
    "other_onetime_desc": "description or null"
  }},
  "totals": {{
    "estimated_annual_comp_usd": number or null,
    "estimated_total_onetime_usd": number or null
  }},
  "summary": "2-3 sentences distinguishing annual pay from one-time awards"
}}

Rules: all USD as plain numbers; null if not disclosed; compute totals where possible.

Filing text:
{text}"""


def extract_with_claude(text: str) -> Optional[dict]:
    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text[:13000])}]
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        return json.loads(raw)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_usd(v, na="—") -> str:
    if v is None:
        return na
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:,.0f}K"
    return f"${v:,.0f}"


def fmt_pct(v, na="—") -> str:
    return f"{v:.0f}%" if v is not None else na


def val_cls(v) -> str:
    if v is None:
        return "na"
    if isinstance(v, (int, float)) and v >= 1_000_000:
        return "large"
    return "found"


def comp_row_html(label: str, value, fmt=None, suffix="") -> str:
    if fmt is None:
        fmt = fmt_usd
    display = (fmt(value) + suffix) if value is not None else "—"
    cls = val_cls(value)
    return (f'<div class="comp-row">'
            f'<span class="comp-key">{label}</span>'
            f'<span class="comp-val {cls}">{display}</span>'
            f'</div>')


def appt_badge_html(appt_type: str) -> str:
    m = {"new_hire": ("badge-new", "New Hire"), "promotion": ("badge-promoted", "Promoted"),
         "interim": ("badge-interim", "Interim"), "successor": ("badge-new", "Successor")}
    cls, lbl = m.get(appt_type, ("badge-new", "Appointed"))
    return f'<span class="appt-badge {cls}">{lbl}</span>'


def render_card(ticker, company, file_date, cik, accession_no, data) -> str:
    ann  = data.get("annual", {}) or {}
    one  = data.get("onetime", {}) or {}
    tots = data.get("totals", {}) or {}
    appt = data.get("appointment_type", "unknown")

    edgar_url = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                 f"&CIK={cik}&type=8-K&dateb=&owner=include&count=5")

    sh_fmt = lambda v: f"{v:,.0f} sh" if v is not None else "—"

    annual_rows = "".join([
        comp_row_html("Base Salary", ann.get("base_salary_usd")),
        comp_row_html("Target Bonus (% salary)", ann.get("target_bonus_pct_of_salary"),
                      fmt_pct, " of base"),
        comp_row_html("Target Bonus ($)", ann.get("target_bonus_usd")),
        comp_row_html("Max Bonus ($)", ann.get("max_bonus_usd")),
        comp_row_html("Annual LTI Target", ann.get("annual_lti_target_usd")),
        comp_row_html("Annual RSU (shares)", ann.get("annual_rsu_shares"), sh_fmt),
        comp_row_html("Annual Options (shares)", ann.get("annual_option_shares"), sh_fmt),
    ])
    annual_note = ann.get("annual_equity_notes") or ann.get("other_annual_benefits") or ""

    onetime_rows = "".join([
        comp_row_html("Cash Signing Bonus", one.get("signing_bonus_cash_usd")),
        comp_row_html("Make-Whole Cash", one.get("makewhole_cash_usd")),
        comp_row_html("Make-Whole Equity", one.get("makewhole_equity_value_usd")),
        comp_row_html("One-Time RSU Value", one.get("onetime_rsu_value_usd")),
        comp_row_html("One-Time RSU Shares", one.get("onetime_rsu_shares"), sh_fmt),
        comp_row_html("One-Time Options Shares", one.get("onetime_option_shares"), sh_fmt),
        comp_row_html("Relocation", one.get("relocation_usd")),
        comp_row_html("Other One-Time", one.get("other_onetime_usd")),
    ])
    ot_note_parts = []
    if one.get("signing_bonus_conditions"):
        ot_note_parts.append(f"Signing conditions: {one['signing_bonus_conditions']}")
    if one.get("makewhole_equity_notes"):
        ot_note_parts.append(f"Make-whole: {one['makewhole_equity_notes']}")
    if one.get("onetime_rsu_vesting"):
        ot_note_parts.append(f"RSU vesting: {one['onetime_rsu_vesting']}")
    if one.get("other_onetime_desc"):
        ot_note_parts.append(one["other_onetime_desc"])
    onetime_note = " · ".join(ot_note_parts)

    est_annual  = tots.get("estimated_annual_comp_usd")
    est_onetime = tots.get("estimated_total_onetime_usd")
    total_block = ""
    if est_annual or est_onetime:
        a_str = fmt_usd(est_annual) if est_annual else "—"
        o_str = fmt_usd(est_onetime) if est_onetime else "—"
        total_block = f"""
        <div style="display:flex;border-top:1px solid var(--border);background:var(--surface2);">
          <div style="flex:1;padding:0.6rem 1.4rem;border-right:1px solid var(--border);">
            <div class="comp-key" style="margin-bottom:0.2rem;">Est. Annual Total Comp</div>
            <div style="font-family:'Fraunces',serif;font-size:1.1rem;color:var(--green);">{a_str}</div>
          </div>
          <div style="flex:1;padding:0.6rem 1.4rem;">
            <div class="comp-key" style="margin-bottom:0.2rem;">Est. Total One-Time Awards</div>
            <div style="font-family:'Fraunces',serif;font-size:1.1rem;color:var(--amber);">{o_str}</div>
          </div>
        </div>"""

    summary = data.get("summary", "")

    return f"""
<div class="filing-card">
  <div class="card-top">
    <div>
      <div><span class="card-ticker">{ticker}</span>{appt_badge_html(appt)}</div>
      <div class="card-company">{company}</div>
      <div class="card-exec">{data.get('executive_name') or 'Executive'} · {data.get('title') or 'CEO'}</div>
    </div>
    <div class="card-meta">
      <div>Filed {file_date}</div>
      <div>Effective: {data.get('effective_date') or 'not specified'}</div>
      <a href="{edgar_url}" target="_blank"
         style="color:var(--amber-dim);text-decoration:none;font-size:0.62rem;">EDGAR ↗</a>
    </div>
  </div>
  <div class="comp-tables">
    <div class="comp-section">
      <div class="comp-section-title annual">
        <span class="dot dot-green"></span> Annual / Ongoing Compensation
      </div>
      {annual_rows}
      {f'<div class="comp-note">{annual_note}</div>' if annual_note else ''}
    </div>
    <div class="comp-section">
      <div class="comp-section-title onetime">
        <span class="dot dot-amber"></span> One-Time / New Hire Awards
      </div>
      {onetime_rows}
      {f'<div class="comp-note">{onetime_note}</div>' if onetime_note else ''}
    </div>
  </div>
  {total_block}
  {f'<div class="summary-text">◈ {summary}</div>' if summary else ''}
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar inputs
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ◈ Scanner")
    st.markdown("---")

    ticker_input = st.text_area(
        "Tickers (comma, space, or newline separated)",
        placeholder="AAPL, MSFT, GOOGL\nAMZN TSLA META\nNVDA JPM GS",
        height=180,
        help="Paste up to 100 tickers. Resolved against EDGAR's live CIK database."
    )

    st.markdown("---")

    search_back_from = st.date_input(
        "Search back from",
        value=date.today(),
        max_value=date.today(),
    )

    months_back = st.slider(
        "Months to look back",
        min_value=1, max_value=36, value=12, step=1,
        format="%d mo"
    )

    start_date = search_back_from - relativedelta(months=months_back)
    end_date   = search_back_from

    st.markdown(f"""
    <div style="font-size:0.62rem;color:var(--muted);margin-top:0.3rem;line-height:1.8;">
      {start_date.strftime('%b %d, %Y')} → {end_date.strftime('%b %d, %Y')}
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Filters**")
    exclude_interim   = st.checkbox("Exclude interim appointments", value=False)
    only_appointments = st.checkbox("CEO hires only (exclude COO/CFO)", value=True)

    st.markdown("---")
    run_scan = st.button("⟳  Run Scan")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.6rem;color:var(--muted2);line-height:2;">
    Source · SEC EDGAR<br>Forms  · 8-K, 8-K/A<br>
    Item   · 5.02 (Officers)<br>AI     · Claude Sonnet
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <div>
    <div class="app-title">CEO <span>Comp</span> Scanner</div>
    <div class="app-subtitle">SEC EDGAR · 8-K Item 5.02 · Annual vs One-Time Award Breakdown</div>
  </div>
  <div class="header-badge">◈ Real-Time EDGAR</div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Landing state
# ─────────────────────────────────────────────────────────────────────────────

if not run_scan:
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="landing-card">
          <h3>Ticker-Driven Search</h3>
          <p>Paste up to 100 tickers. Each resolves to an EDGAR CIK for precise company-level filing lookup — no approximate text matching across all filers.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="landing-card">
          <h3>Annual vs One-Time</h3>
          <p>Claude separates recurring pay (base salary, annual bonus target, LTI) from hire-specific awards (signing bonuses, make-wholes replacing forfeited equity, inducement grants, relocation).</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="landing-card">
          <h3>1–36 Month Window</h3>
          <p>Dial the lookback from 1 month to 3 years, anchored to any end date. Useful for peer group analysis, proxy season prep, or tracking a cohort over time.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-bar" style="margin-top:1.5rem;">
      <strong style="color:var(--amber);">On compensation classification.</strong>
      Annual/ongoing = base salary + annual bonus (target and max) + ongoing annual LTI program target.
      One-time = cash signing bonuses (repayment conditions noted), make-whole cash and equity explicitly
      replacing forfeited prior-employer awards, inducement equity grants, and relocation payments.
      Estimated totals are computed where all components are disclosed.
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Parse tickers
# ─────────────────────────────────────────────────────────────────────────────

raw_tickers = re.split(r'[\s,;|]+', ticker_input.strip().upper())
tickers = list(dict.fromkeys([t for t in raw_tickers if t]))

if not tickers:
    st.error("Please enter at least one ticker in the sidebar.")
    st.stop()

if len(tickers) > 100:
    st.warning(f"Truncating to first 100 tickers ({len(tickers)} supplied).")
    tickers = tickers[:100]

# Resolve CIKs
with st.spinner("Loading EDGAR ticker map…"):
    ticker_map = load_ticker_map()

resolved, failed = resolve_tickers(tickers, ticker_map)

pills_html = (
    "".join(f'<span class="ticker-pill resolved">{t}</span>' for t in resolved) +
    "".join(f'<span class="ticker-pill failed">{t} ✗</span>' for t in failed)
)
st.markdown(f'<div class="ticker-pills">{pills_html}</div>', unsafe_allow_html=True)

if failed:
    st.markdown(
        f'<div class="error-bar">Could not resolve to EDGAR CIK: {", ".join(failed)} — these will be skipped.</div>',
        unsafe_allow_html=True)

if not resolved:
    st.error("No tickers resolved. Check your input.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Scan loop
# ─────────────────────────────────────────────────────────────────────────────

results, skipped, no_filings, api_errors, total_8ks = [], 0, 0, 0, 0
progress_bar = st.progress(0)
status_line  = st.empty()
n_tickers    = len(resolved)

for i, (ticker, info) in enumerate(resolved.items()):
    cik     = info["cik"]
    cik_pad = info["cik_pad"]
    company = info["title"]

    status_line.markdown(
        f'<div style="font-size:0.68rem;color:var(--muted);font-family:\'IBM Plex Mono\',monospace;padding:0.3rem 0;">'
        f'[{i+1}/{n_tickers}] {ticker} · {company}</div>',
        unsafe_allow_html=True)
    progress_bar.progress((i + 1) / n_tickers)

    time.sleep(0.12)
    filings = get_company_8ks(cik_pad, start_date, end_date)

    if not filings:
        no_filings += 1
        continue

    for filing in filings:
        total_8ks += 1
        acc   = filing["accession_no"]
        pdoc  = filing.get("primary_doc", "")
        fdate = filing["file_date"]

        time.sleep(0.12)
        text = fetch_filing_text(cik, acc, pdoc)

        if not text or len(text) < 200:
            skipped += 1
            continue

        if only_appointments and not is_ceo_appointment(text):
            skipped += 1
            continue

        data = extract_with_claude(text)
        if not data:
            api_errors += 1
            continue

        if not data.get("is_ceo_appointment"):
            skipped += 1
            continue

        if exclude_interim and data.get("appointment_type") == "interim":
            skipped += 1
            continue

        results.append({
            "ticker": ticker, "company": company, "cik": cik,
            "accession_no": acc, "file_date": fdate,
            "data": data, "raw_text": text,
        })

progress_bar.empty()
status_line.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics strip
# ─────────────────────────────────────────────────────────────────────────────

n_results   = len(results)
n_w_salary  = sum(1 for r in results if (r["data"].get("annual") or {}).get("base_salary_usd"))
n_w_onetime = sum(1 for r in results if any(
    (r["data"].get("onetime") or {}).get(k)
    for k in ["signing_bonus_cash_usd","makewhole_cash_usd",
              "makewhole_equity_value_usd","onetime_rsu_value_usd"]))

st.markdown(f"""
<div class="metric-strip">
  <div class="metric-block">
    <div class="metric-label">CEO Appointments</div>
    <div class="metric-value">{n_results}</div>
    <div class="metric-sub">across {len(resolved)} tickers</div>
  </div>
  <div class="metric-block">
    <div class="metric-label">Salary Disclosed</div>
    <div class="metric-value">{n_w_salary}</div>
    <div class="metric-sub">of {n_results} filings</div>
  </div>
  <div class="metric-block">
    <div class="metric-label">One-Time Awards Found</div>
    <div class="metric-value">{n_w_onetime}</div>
    <div class="metric-sub">signing · make-whole · inducement</div>
  </div>
  <div class="metric-block">
    <div class="metric-label">8-Ks Scanned</div>
    <div class="metric-value">{total_8ks}</div>
    <div class="metric-sub">{skipped} filtered · {api_errors} errors</div>
  </div>
  <div class="metric-block">
    <div class="metric-label">Lookback Window</div>
    <div class="metric-value" style="font-size:1.1rem;">{months_back}mo</div>
    <div class="metric-sub">{start_date.strftime('%b %Y')} → {end_date.strftime('%b %Y')}</div>
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="content-area">', unsafe_allow_html=True)

if not results:
    st.markdown("""
    <div class="info-bar" style="border-left-color:var(--muted2);">
      No confirmed CEO appointments found for these tickers in the selected window.
      Try extending the month range, adding more tickers, or disabling filters.
    </div>""", unsafe_allow_html=True)
else:
    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-head">
      <span>Compensation Summary</span>
      <span style="color:var(--muted2);font-size:0.55rem;">
        <span style="color:var(--green);">■</span> Annual &nbsp;
        <span style="color:var(--amber);">■</span> One-Time
      </span>
    </div>""", unsafe_allow_html=True)

    def td(v, cls="grn", formatter=fmt_usd):
        if v is None:
            return '<td class="na">—</td>'
        return f'<td class="{cls}">{formatter(v)}</td>'

    rows_html = ""
    for r in results:
        ann  = r["data"].get("annual", {}) or {}
        one  = r["data"].get("onetime", {}) or {}
        tots = r["data"].get("totals", {}) or {}

        ot_cash = sum(filter(None, [
            one.get("signing_bonus_cash_usd"), one.get("makewhole_cash_usd"),
            one.get("relocation_usd"), one.get("other_onetime_usd"),
        ])) or None
        ot_eq = sum(filter(None, [
            one.get("makewhole_equity_value_usd"), one.get("onetime_rsu_value_usd"),
            one.get("onetime_option_value_usd"),
        ])) or None

        appt_lbl = {"new_hire":"New Hire","promotion":"Promoted",
                    "interim":"Interim","successor":"Successor"}.get(
            r["data"].get("appointment_type","unknown"), "Appointed")

        bonus_pct = ann.get("target_bonus_pct_of_salary")
        bonus_td = f'<td>{fmt_pct(bonus_pct)}</td>' if bonus_pct is not None else '<td class="na">—</td>'

        rows_html += f"""<tr>
          <td class="tkr">{r['ticker']}</td>
          <td style="color:var(--muted);font-size:0.65rem;">{r['data'].get('executive_name') or '—'}</td>
          <td style="color:var(--muted);font-size:0.65rem;">{appt_lbl}</td>
          <td style="color:var(--muted);">{r['file_date']}</td>
          {td(ann.get('base_salary_usd'))}
          {bonus_td}
          {td(ann.get('annual_lti_target_usd'))}
          {td(tots.get('estimated_annual_comp_usd'))}
          {td(ot_cash, 'amb')}
          {td(ot_eq, 'amb')}
          {td(tots.get('estimated_total_onetime_usd'), 'amb')}
        </tr>"""

    st.markdown(f"""
    <div class="comp-table-wrap">
    <table class="comp-table">
      <thead><tr>
        <th>Ticker</th><th>Executive</th><th>Type</th><th>Filed</th>
        <th style="color:var(--green);">Base Salary</th>
        <th style="color:var(--green);">Bonus Target</th>
        <th style="color:var(--green);">Annual LTI</th>
        <th style="color:var(--green);">Est. Annual Total</th>
        <th style="color:var(--amber);">One-Time Cash</th>
        <th style="color:var(--amber);">One-Time Equity</th>
        <th style="color:var(--amber);">Est. One-Time Total</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    # ── Detailed cards ────────────────────────────────────────────────────────
    st.markdown("""<div style="margin-top:2.5rem;">
    <div class="section-head"><span>Detailed Filing Cards</span></div>
    </div>""", unsafe_allow_html=True)

    for r in results:
        card = render_card(
            ticker=r["ticker"], company=r["company"],
            file_date=r["file_date"], cik=r["cik"],
            accession_no=r["accession_no"], data=r["data"],
        )
        st.markdown(card, unsafe_allow_html=True)
        with st.expander(f"▸ {r['ticker']} — raw JSON + filing text"):
            col_j, col_t = st.columns(2)
            with col_j:
                st.json(r["data"])
            with col_t:
                st.code(r["raw_text"][:4000], language=None)

    # ── CSV Export ────────────────────────────────────────────────────────────
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        # Identity
        "Ticker","Company","Executive Name","Title","Appointment Type",
        "Effective Date","Filing Date","EDGAR Accession",
        # Annual
        "Base Salary (USD)","Target Bonus % of Salary","Target Bonus (USD)",
        "Max Bonus % of Salary","Max Bonus (USD)",
        "Annual LTI Target (USD)","Annual RSU Shares","Annual Option Shares",
        "Annual Equity Notes","Other Annual Benefits","Est. Annual Total Comp (USD)",
        # One-Time
        "Signing Bonus Cash (USD)","Signing Bonus Conditions",
        "Make-Whole Cash (USD)","Make-Whole Cash Notes",
        "Make-Whole Equity Value (USD)","Make-Whole Equity Notes",
        "One-Time RSU Shares","One-Time RSU Value (USD)","One-Time RSU Vesting",
        "One-Time Option Shares","One-Time Option Value (USD)",
        "One-Time Option Vesting","One-Time Option Strike (USD)",
        "Relocation (USD)","Other One-Time (USD)","Other One-Time Description",
        "Est. Total One-Time Awards (USD)",
        "Summary",
    ])
    for r in results:
        d   = r["data"]
        ann = d.get("annual", {}) or {}
        one = d.get("onetime", {}) or {}
        tot = d.get("totals", {}) or {}
        w.writerow([
            r["ticker"], r["company"], d.get("executive_name",""),
            d.get("title",""), d.get("appointment_type",""),
            d.get("effective_date",""), r["file_date"], r["accession_no"],
            ann.get("base_salary_usd",""), ann.get("target_bonus_pct_of_salary",""),
            ann.get("target_bonus_usd",""), ann.get("max_bonus_pct_of_salary",""),
            ann.get("max_bonus_usd",""), ann.get("annual_lti_target_usd",""),
            ann.get("annual_rsu_shares",""), ann.get("annual_option_shares",""),
            ann.get("annual_equity_notes",""), ann.get("other_annual_benefits",""),
            tot.get("estimated_annual_comp_usd",""),
            one.get("signing_bonus_cash_usd",""), one.get("signing_bonus_conditions",""),
            one.get("makewhole_cash_usd",""), one.get("makewhole_cash_notes",""),
            one.get("makewhole_equity_value_usd",""), one.get("makewhole_equity_notes",""),
            one.get("onetime_rsu_shares",""), one.get("onetime_rsu_value_usd",""),
            one.get("onetime_rsu_vesting",""), one.get("onetime_option_shares",""),
            one.get("onetime_option_value_usd",""), one.get("onetime_option_vesting",""),
            one.get("onetime_option_strike_usd",""), one.get("relocation_usd",""),
            one.get("other_onetime_usd",""), one.get("other_onetime_desc",""),
            tot.get("estimated_total_onetime_usd",""),
            d.get("summary",""),
        ])

    st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
    st.download_button(
        label=f"↓  Export {n_results} result{'s' if n_results != 1 else ''} as CSV",
        data=buf.getvalue(),
        file_name=f"ceo_comp_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
