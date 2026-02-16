import calendar
import datetime as dt
from zoneinfo import ZoneInfo
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# ----------
# App config
# ----------
st.set_page_config(page_title="8-K Executive Appointment Screener", layout="wide")

RUN_LOCK = threading.Lock()
DB_PATH = Path("exec_8k_scanner.sqlite3")
CACHE_DIR = Path(".cache_edgar")
SCANNER = Path("exec_8k_scanner.py")


# ----------
# Helpers
# ----------
def today_chicago() -> dt.date:
    try:
        return dt.datetime.now(ZoneInfo("America/Chicago")).date()
    except Exception:
        return dt.date.today()


def add_months(d: dt.date, months: int) -> dt.date:
    """Add (or subtract) months to a date, clamping the day to end-of-month as needed."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return dt.date(y, m, day)


def parse_tickers(text: str) -> List[str]:
    """Parse tickers from a free-form blob (newlines/commas/spaces; supports # comments)."""
    tickers: List[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0]
        for tok in [t for t in line.replace(",", " ").split() if t.strip()]:
            tickers.append(tok.upper())
    # de-dupe while preserving order
    seen = set()
    out = []
    for t in tickers:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def get_user_agent() -> str:
    # Prefer Streamlit secrets, fall back to environment variable.
    ua = ""
    try:
        ua = st.secrets.get("SEC_USER_AGENT", "")
    except Exception:
        ua = ""
    return (ua or os.environ.get("SEC_USER_AGENT", "")).strip()


def run_scan(
    tickers: List[str],
    position: str,
    lookback_months: int,
    as_of_date: dt.date,
    max_rps: int,
    force: bool,
    mode: str = "submissions",
) -> Tuple[int, str, str]:
    """Run the scanner as a subprocess. Returns (exit_code, stdout, stderr)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tickers_file = td_path / "tickers.txt"
        tickers_file.write_text("\n".join(tickers) + "\n", encoding="utf-8")

        # Outputs are written, but the app reads results from SQLite.
        out_csv = td_path / "events.csv"
        out_jsonl = td_path / "events.jsonl"
        out_md = td_path / "events.md"

        ua = get_user_agent()

        cmd = [
            sys.executable,
            str(SCANNER),
            "--user-agent",
            ua,
            "--db",
            str(DB_PATH),
            "--cache-dir",
            str(CACHE_DIR),
            "--max-rps",
            str(max_rps),
            "scan",
            "--tickers-file",
            str(tickers_file),
            "--position",
            position,
            "--lookback-months",
            str(int(lookback_months)),
            "--as-of",
            as_of_date.isoformat(),
            "--mode",
            mode,
            "--out-csv",
            str(out_csv),
            "--out-jsonl",
            str(out_jsonl),
            "--out-md",
            str(out_md),
        ]
        if force:
            cmd.append("--force")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr


def load_events_from_db(tickers: List[str], position: str, lookback_months: int, as_of_date: dt.date) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    end = as_of_date
    start = add_months(end, -int(lookback_months))

    placeholders = ",".join(["?"] * len(tickers)) if tickers else ""

    sql = f"""
        SELECT
            ticker,
            company_name,
            filing_date,
            person,
            matched_title,
            effective_date,
            event_type,
            confidence,
            raw_json
        FROM exec_events
        WHERE position_query = ?
          AND filing_date >= ?
          AND filing_date <= ?
          {"AND ticker IN (" + placeholders + ")" if tickers else ""}
        ORDER BY filing_date DESC, confidence DESC
    """

    params: List[str] = [position, start.isoformat(), end.isoformat()]
    if tickers:
        params.extend(tickers)

    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    finally:
        con.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)

    # Expand curated compensation fields from raw_json
    def comp_fields(raw: str) -> Dict[str, str]:
        try:
            obj = json.loads(raw)
        except Exception:
            return {}

        comp = (obj.get("compensation") or {})
        filing = (obj.get("filing") or {})
        other = comp.get("other") or []

        # Build URLs from filing fields (dataclasses.asdict doesn't include computed properties)
        source_8k_url = ""
        primary_doc_url = ""
        try:
            cik = str(filing.get("cik") or "")
            accession = str(filing.get("accession") or "")
            primary_doc = str(filing.get("primary_doc") or "")
            if cik.isdigit() and accession:
                cik_int = int(cik)
                acc_no = accession.replace("-", "")
                source_8k_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no}/{accession}-index.html"
                if primary_doc:
                    primary_doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no}/{primary_doc}"
        except Exception:
            pass

        eq_target = comp.get("equity_target_annual_values") or []
        eq_one_time = comp.get("equity_one_time_values") or []
        eq_labels = comp.get("equity_one_time_labels") or []

        # A short human-readable summary
        parts = []
        if comp.get("base_salary"):
            parts.append(f"Base {comp['base_salary']}")
        if comp.get("target_bonus"):
            parts.append(f"Bonus {comp['target_bonus']}")
        if comp.get("sign_on_bonus"):
            parts.append(f"Sign-on {comp['sign_on_bonus']}")
        if eq_target:
            parts.append("Equity target/annual " + ", ".join(eq_target))
        if eq_one_time:
            lab = f" ({', '.join(eq_labels)})" if eq_labels else ""
            parts.append("One-time equity " + ", ".join(eq_one_time) + lab)

        return {
            "base_salary": comp.get("base_salary", ""),
            "target_bonus": comp.get("target_bonus", ""),
            "sign_on_bonus": comp.get("sign_on_bonus", ""),
            "equity_target_annual_values": "; ".join(eq_target),
            "equity_one_time_values": "; ".join(eq_one_time),
            "equity_one_time_labels": ", ".join(eq_labels),
            "other_keywords": ", ".join(other),
            "compensation_summary": "; ".join(parts) if parts else "No comp terms detected in scanned docs/exhibits.",
            "source_8k_url": source_8k_url,
            "primary_doc_url": primary_doc_url,
        }

    expanded = df["raw_json"].apply(comp_fields).apply(pd.Series)
    df = pd.concat([df.drop(columns=["raw_json"]), expanded], axis=1)

    df = df.rename(columns={"person": "new_executive", "matched_title": "position"})

    ordered = [
        "ticker",
        "company_name",
        "filing_date",
        "source_8k_url",
        "new_executive",
        "position",
        "effective_date",
        "event_type",
        "base_salary",
        "target_bonus",
        "sign_on_bonus",
        "equity_target_annual_values",
        "equity_one_time_values",
        "equity_one_time_labels",
        "other_keywords",
        "compensation_summary",
        "primary_doc_url",
        "confidence",
    ]
    df = df[[c for c in ordered if c in df.columns]]
    return df


# ----------
# UI
# ----------
st.title("8-K Executive Appointment Screener")
st.caption(
    "Scans SEC Form 8-K / 8-K/A filings (Item 5.02) for executive appointments/promotions for a chosen position "
    "and summarizes compensation terms (heuristic). Always review the underlying filing for full context."
)

ua = get_user_agent()
if not ua or "@" not in ua:
    st.warning(
        "SEC_USER_AGENT is not configured. Set it in Streamlit secrets as SEC_USER_AGENT = \"YourOrg your.email@domain.com\". "
        "The SEC requests a declared User-Agent for automated access."
    )

st.sidebar.header("Scan settings")

common_positions = ["CEO", "CFO", "COO", "CTO", "CIO", "CMO", "CHRO", "CAO", "CLO", "Custom"]
sel = st.sidebar.selectbox("Position", common_positions, index=1)
if sel == "Custom":
    position = st.sidebar.text_input("Custom position title", value="Chief Risk Officer")
else:
    position = sel

as_of_date = st.sidebar.date_input(
    "Search back from (end date)",
    value=today_chicago(),
    max_value=today_chicago(),
)

lookback_months = st.sidebar.slider("Look-back months", min_value=1, max_value=36, value=6)
max_rps = st.sidebar.slider("Max requests/sec (keep ≤ 10)", min_value=1, max_value=10, value=6)

with st.sidebar.expander("Advanced"):
    mode = st.selectbox("Ingestion mode", ["submissions", "daily-index"], index=0,
                        help="'submissions' is best for small watchlists; 'daily-index' scans the market-wide daily index across dates.")
    force = st.checkbox("Refresh existing filings (force re-scan)", value=False,
                        help="Turn on once after updating the extractor to refresh older saved results in the SQLite DB.")

st.sidebar.caption(
    f"Window: {add_months(as_of_date, -int(lookback_months)).isoformat()} → {as_of_date.isoformat()}  "
    f"({lookback_months} month{'s' if lookback_months != 1 else ''})"
)

st.subheader("Watchlist")
colA, colB = st.columns([1, 1])

with colA:
    uploaded = st.file_uploader("Upload tickers file (.txt)", type=["txt"])
with colB:
    pasted = st.text_area(
        "Or paste tickers (comma/space/newline separated)",
        placeholder="MPC\nAAPL\nMSFT\n...",
        height=140,
    )

tickers: List[str] = []
if uploaded is not None:
    try:
        tickers.extend(parse_tickers(uploaded.getvalue().decode("utf-8", errors="replace")))
    except Exception:
        st.error("Could not read uploaded file as UTF-8 text.")
if pasted.strip():
    tickers.extend(parse_tickers(pasted))

# de-dupe while preserving order
seen = set()
watchlist = [t for t in tickers if not (t in seen or seen.add(t))]

st.write(f"**Tickers loaded:** {len(watchlist)}")
if len(watchlist) > 0:
    st.write(", ".join(watchlist[:50]) + (" …" if len(watchlist) > 50 else ""))

run_clicked = st.button("Run scan", type="primary", disabled=(len(watchlist) == 0 or not ua or "@" not in ua))

if run_clicked:
    if not RUN_LOCK.acquire(blocking=False):
        st.warning("A scan is already running. Please try again in a few minutes.")
    else:
        try:
            with st.spinner("Scanning 8-K filings…"):
                code, out, err = run_scan(
                    watchlist,
                    position=position,
                    lookback_months=lookback_months,
                    as_of_date=as_of_date,
                    max_rps=max_rps,
                    force=force,
                    mode=mode,
                )

            if code != 0:
                st.error("Scan failed.")
                if out:
                    st.code(out)
                if err:
                    st.code(err)
            else:
                if out.strip():
                    st.code(out)
                if err.strip():
                    st.warning("Scanner warnings:")
                    st.code(err)

                st.success("Scan completed.")

        finally:
            RUN_LOCK.release()

st.divider()

st.subheader("Results")
results = load_events_from_db(watchlist, position=position, lookback_months=lookback_months, as_of_date=as_of_date)

if results.empty:
    st.info("No matching executive appointment events found in the selected window (or none scanned yet).")
else:
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "source_8k_url": st.column_config.LinkColumn("8‑K", display_text="Open 8‑K"),
            "primary_doc_url": st.column_config.LinkColumn("Primary doc", display_text="Open"),
        },
    )

    # Download CSV
    csv_buf = io.StringIO()
    results.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="events.csv",
        mime="text/csv",
    )

st.caption(
    "Note: This app throttles requests, but avoid running multiple instances in parallel to stay within SEC fair-access limits."
)
