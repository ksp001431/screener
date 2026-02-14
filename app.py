import datetime as dt
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
st.set_page_config(page_title="8-K Executive Appointment Scanner", layout="wide")

RUN_LOCK = threading.Lock()
DB_PATH = Path("exec_8k_scanner.sqlite3")
CACHE_DIR = Path(".cache_edgar")
SCANNER = Path("exec_8k_scanner.py")


# ----------
# Helpers
# ----------

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


def run_scan(tickers: List[str], position: str, lookback_days: int, max_rps: int) -> Tuple[int, str, str]:
    """Run the scanner as a subprocess. Returns (exit_code, stdout, stderr)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tickers_file = td_path / "tickers.txt"
        tickers_file.write_text("\n".join(tickers) + "\n", encoding="utf-8")

        # Write outputs to a temp directory (we read results from the SQLite DB).
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
            "--lookback-days",
            str(int(lookback_days)),
            "--out-csv",
            str(out_csv),
            "--out-jsonl",
            str(out_jsonl),
            "--out-md",
            str(out_md),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr


def load_events_from_db(tickers: List[str], position: str, lookback_days: int) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    today = dt.date.today()
    start = today - dt.timedelta(days=int(lookback_days) - 1)

    # SQLite has a default limit of 999 variables; 500 tickers is safe.
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

    params: List[str] = [position, start.isoformat(), today.isoformat()]
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

    # Expand compensation details from raw_json
    def comp_fields(raw: str) -> Dict[str, str]:
        try:
            obj = json.loads(raw)
        except Exception:
            return {}
        comp = (obj.get("compensation") or {})
        filing = (obj.get("filing") or {})
        equity = comp.get("equity_awards") or []
        sever = comp.get("severance") or []
        other = comp.get("other") or []

        # Build primary doc URL from filing fields (dataclasses.asdict doesn't include computed properties)
        primary_url = ""
        try:
            cik = str(filing.get("cik") or "")
            accession = str(filing.get("accession") or "")
            primary_doc = str(filing.get("primary_doc") or "")
            if cik.isdigit() and accession and primary_doc:
                cik_int = int(cik)
                acc_no = accession.replace("-", "")
                primary_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no}/{primary_doc}"
        except Exception:
            primary_url = ""

        # A short summary suitable for spreadsheet / UI
        parts = []
        if comp.get("base_salary"):
            parts.append(f"Base {comp['base_salary']}")
        if comp.get("target_bonus"):
            parts.append(f"Bonus {comp['target_bonus']}")
        if comp.get("sign_on_bonus"):
            parts.append(f"Sign-on {comp['sign_on_bonus']}")
        if equity:
            parts.append(f"Equity mentions {len(equity)}")
        if sever:
            parts.append(f"Severance/CIC mentions {len(sever)}")
        if other:
            parts.append("Other: " + ", ".join(other))

        return {
            "base_salary": comp.get("base_salary", ""),
            "target_bonus": comp.get("target_bonus", ""),
            "sign_on_bonus": comp.get("sign_on_bonus", ""),
            "equity_mentions_count": str(len(equity)),
            "severance_mentions_count": str(len(sever)),
            "other_keywords": ", ".join(other),
            "compensation_details": "; ".join(parts) if parts else "No comp terms detected in scanned docs/exhibits.",
            "primary_doc_url": primary_url,
        }

    expanded = df["raw_json"].apply(comp_fields).apply(pd.Series)
    df = pd.concat([df.drop(columns=["raw_json"]), expanded], axis=1)

    # Rename to match your requested output labels
    df = df.rename(
        columns={
            "person": "new_executive",
            "matched_title": "position",
        }
    )

    # Order columns
    ordered = [
        "ticker",
        "company_name",
        "filing_date",
        "new_executive",
        "position",
        "effective_date",
        "event_type",
        "base_salary",
        "target_bonus",
        "sign_on_bonus",
        "equity_mentions_count",
        "severance_mentions_count",
        "other_keywords",
        "compensation_details",
        "primary_doc_url",
        "confidence",
    ]
    df = df[[c for c in ordered if c in df.columns]]
    return df


# ----------
# UI
# ----------

st.title("8-K Executive Appointment Scanner")
st.caption(
    "Scans SEC Form 8-K / 8-K/A filings (Item 5.02) for executive appointments for a chosen position "
    "and summarizes compensation terms (heuristic). Always review the filing for full context."
)

ua = get_user_agent()
if not ua or "@" not in ua:
    st.warning(
        "SEC_USER_AGENT is not configured. Set it in Streamlit secrets as SEC_USER_AGENT = \"YourOrg your.email@domain.com\". "
        "The SEC requests a declared User-Agent for automated access."
    )

st.sidebar.header("Scan settings")
common_positions = ["CEO", "CFO", "COO", "CTO", "CIO", "CMO", "CHRO", "CAO", "CLO", "Custom"]
sel = st.sidebar.selectbox("Position", common_positions, index=0)
if sel == "Custom":
    position = st.sidebar.text_input("Custom position title", value="Chief Risk Officer")
else:
    position = sel

lookback_days = st.sidebar.slider("Look-back days", min_value=1, max_value=120, value=14)
max_rps = st.sidebar.slider("Max requests/sec (keep ≤ 10)", min_value=1, max_value=10, value=8)

st.subheader("Watchlist")
colA, colB = st.columns([1, 1])

with colA:
    uploaded = st.file_uploader("Upload tickers file (.txt)", type=["txt"])
with colB:
    pasted = st.text_area(
        "Or paste tickers (comma/space/newline separated)",
        placeholder="AAPL\nMSFT\nNVDA\n...",
        height=140,
    )

repo_watchlist_path = st.text_input("Optional: watchlist file path in repo", value="")

tickers: List[str] = []
if uploaded is not None:
    try:
        tickers.extend(parse_tickers(uploaded.getvalue().decode("utf-8", errors="replace")))
    except Exception:
        st.error("Could not read uploaded file as UTF-8 text.")
if pasted.strip():
    tickers.extend(parse_tickers(pasted))

if repo_watchlist_path.strip():
    pth = Path(repo_watchlist_path.strip())
    if pth.exists() and pth.is_file():
        tickers.extend(parse_tickers(pth.read_text(encoding="utf-8", errors="replace")))
    else:
        st.info("Repo watchlist path not found (ignored).")

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
            with st.spinner("Scanning 8-K filings (this can take a few minutes depending on lookback and watchlist)…"):
                code, out, err = run_scan(watchlist, position=position, lookback_days=lookback_days, max_rps=max_rps)

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
results = load_events_from_db(watchlist, position=position, lookback_days=lookback_days)

if results.empty:
    st.info("No matching executive appointment events found in the selected lookback window (or none scanned yet).")
else:
    st.dataframe(results, use_container_width=True)

    # Download CSV
    csv_buf = io.StringIO()
    results.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="events.csv",
        mime="text/csv",
    )

    # Show detailed evidence snippets from raw JSON (optional)
    with st.expander("Show evidence snippets (from filings)"):
        con = sqlite3.connect(DB_PATH)
        try:
            today = dt.date.today()
            start = today - dt.timedelta(days=int(lookback_days) - 1)
            placeholders = ",".join(["?"] * len(watchlist))
            sql = f"""
                SELECT ticker, company_name, filing_date, person, matched_title, raw_json
                FROM exec_events
                WHERE position_query = ?
                  AND filing_date >= ?
                  AND filing_date <= ?
                  AND ticker IN ({placeholders})
                ORDER BY filing_date DESC
            """
            params = [position, start.isoformat(), today.isoformat()] + watchlist
            rows = con.execute(sql, params).fetchall()
        finally:
            con.close()

        for (ticker, company, fdate, person, title, raw) in rows[:25]:
            try:
                obj = json.loads(raw)
                snippets = ((obj.get("compensation") or {}).get("evidence_snippets") or [])
                filing = (obj.get("filing") or {})
                # Rebuild primary document URL from filing fields.
                primary_url = ""
                cik = str(filing.get("cik") or "")
                accession = str(filing.get("accession") or "")
                primary_doc = str(filing.get("primary_doc") or "")
                if cik.isdigit() and accession and primary_doc:
                    cik_int = int(cik)
                    acc_no = accession.replace("-", "")
                    primary_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no}/{primary_doc}"
            except Exception:
                snippets = []
                primary_url = ""

            st.markdown(f"**{ticker} — {company}**  \nFiled: {fdate}  \nExecutive: {person}  \nPosition: {title}")
            if primary_url:
                st.write(primary_url)
            if snippets:
                for sn in snippets[:6]:
                    st.write("- " + sn)
            else:
                st.write("(No evidence snippets captured.)")
            st.write("—")

st.caption(
    "Note: This app throttles requests, but avoid running multiple instances in parallel to stay within SEC fair-access limits."
)
