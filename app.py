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
    """Load scanned events from SQLite and expand curated comp fields from raw_json.

    IMPORTANT: Always return a DataFrame (possibly empty).
    """
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

    try:
        con = sqlite3.connect(DB_PATH)
        try:
            cur = con.execute(sql, params)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        finally:
            con.close()
    except Exception as e:
        st.error(f"Failed to load results from SQLite DB: {e}")
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)

    def comp_fields(raw: str) -> Dict[str, object]:
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
        accession = str(filing.get("accession") or "")
        try:
            cik = str(filing.get("cik") or "")
            primary_doc = str(filing.get("primary_doc") or "")
            if cik.isdigit() and accession:
                cik_int = int(cik)
                acc_no = accession.replace("-", "")
                source_8k_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no}/{accession}-index.html"
                if primary_doc:
                    primary_doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no}/{primary_doc}"
        except Exception:
            pass

        # Numeric core comp fields (USD assumed)
        base_salary_usd = comp.get("base_salary_usd")
        target_bonus_pct = comp.get("target_bonus_pct")
        target_bonus_usd = comp.get("target_bonus_usd")

        # Convert % bonus target to $ if possible
        if (target_bonus_usd in (None, "")) and (target_bonus_pct not in (None, "")) and (base_salary_usd not in (None, "")):
            try:
                target_bonus_usd = int(round(float(base_salary_usd) * (float(target_bonus_pct) / 100.0)))
            except Exception:
                pass

        equity_annual_usd = comp.get("equity_target_annual_usd_total")
        equity_annual_advance_usd = comp.get("equity_annual_advance_usd_total")
        equity_annual_pregrant_usd = comp.get("equity_annual_pregrant_usd_total")


        # One-time cash: prefer curated field; fall back to older sign_on_bonus_usd
        one_time_cash_usd_total = comp.get("one_time_cash_usd_total")
        if one_time_cash_usd_total in (None, ""):
            one_time_cash_usd_total = comp.get("sign_on_bonus_usd")

        one_time_cash_values = comp.get("one_time_cash_values") or []
        if (not one_time_cash_values) and comp.get("sign_on_bonus"):
            one_time_cash_values = [comp.get("sign_on_bonus")]

        # One-time equity (new hire / signing / inducement / make-whole)
        equity_one_time_usd_total = comp.get("equity_one_time_usd_total")
        equity_one_time_values = comp.get("equity_one_time_values") or []
        equity_one_time_labels = comp.get("equity_one_time_labels") or []

        # Annual/target equity/LTI values (ongoing)
        equity_target_values = comp.get("equity_target_annual_values") or []

        # Evidence snippets for traceability
        evidence_snips = comp.get("evidence_snippets") or []
        # Keep it readable in CSV/UI
        evidence_text = "\n".join([str(s) for s in evidence_snips if s])

        # Target total comp (salary + target bonus $ + annual/ongoing equity/LTI target $)
        target_total_comp_usd = None
        try:
            if base_salary_usd not in (None, "") or target_bonus_usd not in (None, "") or equity_annual_usd not in (None, ""):
                target_total_comp_usd = int((base_salary_usd or 0) + (target_bonus_usd or 0) + (equity_annual_usd or 0))
        except Exception:
            pass

        # A short human-readable summary (kept for quick scanning)
        parts = []
        try:
            if base_salary_usd not in (None, ""):
                parts.append(f"Salary ${int(base_salary_usd):,}")
            if target_bonus_usd not in (None, ""):
                if target_bonus_pct not in (None, ""):
                    parts.append(f"Target bonus ${int(target_bonus_usd):,} ({float(target_bonus_pct):g}%)")
                else:
                    parts.append(f"Target bonus ${int(target_bonus_usd):,}")
            if equity_annual_usd not in (None, ""):
                parts.append(f"Target/annual equity ${int(equity_annual_usd):,}")
            if equity_annual_advance_usd not in (None, ""):
                parts.append(f"Annual equity advance ${int(equity_annual_advance_usd):,}")
            if equity_annual_pregrant_usd not in (None, ""):
                parts.append(f"Annual equity (pregrant) ${int(equity_annual_pregrant_usd):,}")
            if one_time_cash_usd_total not in (None, ""):
                parts.append(f"One-time cash ${int(one_time_cash_usd_total):,}")
            if equity_one_time_usd_total not in (None, ""):
                lab = f" ({', '.join(equity_one_time_labels)})" if equity_one_time_labels else ""
                parts.append(f"One-time equity ${int(equity_one_time_usd_total):,}" + lab)
        except Exception:
            pass

        return {
            # IDs/links
            "accession": accession,
            "source_8k_url": source_8k_url,
            "primary_doc_url": primary_doc_url,

            # Curated numeric fields
            "base_salary_usd": base_salary_usd,
            "target_bonus_pct": target_bonus_pct,
            "target_bonus_usd": target_bonus_usd,
            "equity_target_annual_usd_total": equity_annual_usd,
            "equity_annual_advance_usd_total": equity_annual_advance_usd,
            "equity_annual_pregrant_usd_total": equity_annual_pregrant_usd,
            "target_total_comp_usd": target_total_comp_usd,
            "one_time_cash_usd_total": one_time_cash_usd_total,
            "equity_one_time_usd_total": equity_one_time_usd_total,

            # Audit/detail fields (hidden from the default table, but downloadable and visible in expanders)
            "one_time_cash_values": "; ".join([str(x) for x in one_time_cash_values if x]),
            "equity_one_time_values": "; ".join([str(x) for x in equity_one_time_values if x]),
            "equity_one_time_labels": ", ".join([str(x) for x in equity_one_time_labels if x]),
            "equity_target_annual_values": "; ".join([str(x) for x in equity_target_values if x]),
            "equity_annual_advance_values": "; ".join([str(x) for x in (comp.get("equity_annual_advance_values") or []) if x]),
            "equity_annual_pregrant_values": "; ".join([str(x) for x in (comp.get("equity_annual_pregrant_values") or []) if x]),
            "other_keywords": ", ".join(other),
            "compensation_summary": "; ".join(parts) if parts else "No comp terms detected in scanned docs/exhibits.",
            "evidence_snippets": evidence_text,
        }

    expanded = df["raw_json"].apply(comp_fields).apply(pd.Series)
    df = pd.concat([df.drop(columns=["raw_json"]), expanded], axis=1)

    df = df.rename(columns={"person": "new_executive", "matched_title": "position"})

    # Keep a stable ordering (full results)
    ordered = [
        "ticker",
        "company_name",
        "filing_date",
        "source_8k_url",
        "new_executive",
        "position",
        "effective_date",
        "event_type",

        # Curated numeric fields
        "base_salary_usd",
        "target_bonus_pct",
        "target_bonus_usd",
        "equity_target_annual_usd_total",
        "target_total_comp_usd",
        "one_time_cash_usd_total",
        "equity_one_time_usd_total",

        # Detail fields
        "compensation_summary",
        "one_time_cash_values",
        "equity_target_annual_values",
        "equity_one_time_values",
        "equity_one_time_labels",
        "other_keywords",
        "primary_doc_url",
        "accession",
        "confidence",
        "evidence_snippets",
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
include_interim = st.sidebar.checkbox(
    "Include interim appointments",
    value=False,
    help=(
        "If unchecked, interim appointments are excluded from the Results table. "
        "Companies where only an interim appointment was detected will be listed separately."
    ),
)
max_rps = st.sidebar.slider("Max requests/sec (keep ≤ 10)", min_value=1, max_value=10, value=6)

with st.sidebar.expander("Advanced"):
    mode = st.selectbox(
        "Ingestion mode",
        ["submissions", "daily-index"],
        index=0,
        help="'submissions' is best for small watchlists; 'daily-index' scans the market-wide daily index across dates.",
    )
    force = st.checkbox(
        "Refresh existing filings (force re-scan)",
        value=False,
        help="Turn on once after updating the extractor to refresh older saved results in the SQLite DB.",
    )
    show_audit_cols = st.checkbox(
        "Show audit columns in Results table",
        value=False,
        help="By default, the Results table is curated/compact. Turn this on to show all detail columns.",
    )

st.sidebar.caption(
    f"Window: {add_months(as_of_date, -int(lookback_months)).isoformat()} → {as_of_date.isoformat()}  "
    f"({lookback_months} month{'s' if lookback_months != 1 else ''})"
)

st.subheader("Watchlist")

pasted = st.text_area(
    "Paste ticker symbols (comma/space/newline separated)",
    placeholder="MPC\nFIS\nAAPL\n...",
    height=160,
    help="Paste tickers here. Lines starting with # are ignored.",
)

tickers: List[str] = []
if pasted.strip():
    tickers.extend(parse_tickers(pasted))

seen = set()
watchlist = [t for t in tickers if not (t in seen or seen.add(t))]

st.write(f"**Tickers loaded:** {len(watchlist)}")
if len(watchlist) > 0:
    st.write(", ".join(watchlist[:50]) + (" …" if len(watchlist) > 50 else ""))

run_clicked = st.button(
    "Run scan",
    type="primary",
    disabled=(len(watchlist) == 0 or not ua or "@" not in ua),
)

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
raw_results = load_events_from_db(
    watchlist,
    position=position,
    lookback_months=lookback_months,
    as_of_date=as_of_date,
)

results = raw_results if isinstance(raw_results, pd.DataFrame) else pd.DataFrame()

excluded_interim_lines: List[str] = []

if (not include_interim) and (not results.empty):
    et = results.get("event_type")
    pos_col = results.get("position")
    interim_mask = pd.Series([False] * len(results), index=results.index)
    if et is not None:
        interim_mask = interim_mask | et.fillna("").astype(str).str.lower().eq("interim")
    if pos_col is not None:
        interim_mask = interim_mask | pos_col.fillna("").astype(str).str.contains(r"\binterim\b", case=False, regex=True)

    interim_rows = results[interim_mask].copy()
    non_interim_rows = results[~interim_mask].copy()

    interim_only = set(interim_rows.get("ticker", [])) - set(non_interim_rows.get("ticker", []))
    interim_only_ordered = [t for t in watchlist if t in interim_only]

    if interim_only_ordered and (not interim_rows.empty):
        try:
            interim_rows = interim_rows.sort_values(["filing_date", "confidence"], ascending=[False, False])
        except Exception:
            pass

        for t in interim_only_ordered:
            r = interim_rows[interim_rows["ticker"] == t]
            if r.empty:
                continue
            top = r.iloc[0]
            company = str(top.get("company_name", "") or "").strip()
            exec_name = str(top.get("new_executive", "") or "").strip()
            filed = str(top.get("filing_date", "") or "").strip()
            title = str(top.get("position", "") or "").strip()

            label = f"{t}"
            if company:
                label += f" — {company}"
            details = []
            if exec_name:
                details.append(f"Interim: {exec_name}")
            if filed:
                details.append(f"Filed {filed}")
            if title:
                details.append(title)
            if details:
                label += " (" + "; ".join(details) + ")"
            excluded_interim_lines.append(label)

    results = non_interim_rows

if results.empty:
    if (not include_interim) and excluded_interim_lines:
        st.info(
            "No non-interim appointment events found in the selected window. "
            "Interim-only matches were detected but excluded (see below)."
        )
    else:
        st.info("No matching executive appointment events found in the selected window (or none scanned yet).")
else:
    # ----
    # Clean display view (default)
    # ----
    clean_cols = [
        "ticker",
        "company_name",
        "filing_date",
        "source_8k_url",
        "primary_doc_url",
        "new_executive",
        "position",
        "effective_date",
        "event_type",
        "base_salary_usd",
        "target_bonus_pct",
        "target_bonus_usd",
        "equity_target_annual_usd_total",
        "target_total_comp_usd",
        "one_time_cash_usd_total",
        "equity_one_time_usd_total",
        "compensation_summary",
    ]

    display_df = results if show_audit_cols else results[[c for c in clean_cols if c in results.columns]]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
                       "source_8k_url": st.column_config.LinkColumn("8‑K index", display_text="Open"),
            "primary_doc_url": st.column_config.LinkColumn("Primary 8‑K", display_text="Open"),
            # Currency formatting (USD assumed)
            "base_salary_usd": st.column_config.NumberColumn("Salary", format="dollar", step=1),
            "target_bonus_usd": st.column_config.NumberColumn("Bonus target (USD)", format="dollar", step=1),
            "equity_target_annual_usd_total": st.column_config.NumberColumn("Annual/target equity (USD)", format="dollar", step=1),
            "target_total_comp_usd": st.column_config.NumberColumn("Target total comp (USD)", format="dollar", step=1),
            "one_time_cash_usd_total": st.column_config.NumberColumn("One-time cash (USD)", format="dollar", step=1),
            "equity_one_time_usd_total": st.column_config.NumberColumn("One-time equity (USD)", format="dollar", step=1),

            # Percent/score formatting
            "target_bonus_pct": st.column_config.NumberColumn("Bonus target", format="%.0f%%", step=1),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
        },
    )

    # Downloads: curated + full
    col1, col2 = st.columns([1, 1])
    with col1:
        csv_buf = io.StringIO()
        display_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download curated CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="events_curated.csv",
            mime="text/csv",
        )

    with col2:
        csv_buf2 = io.StringIO()
        results.to_csv(csv_buf2, index=False)
        st.download_button(
            label="Download full CSV (audit fields)",
            data=csv_buf2.getvalue().encode("utf-8"),
            file_name="events_full.csv",
            mime="text/csv",
        )

    # ----
    # Traceability section (keeps the main table clean)
    # ----
    with st.expander("Audit & evidence (traceability)", expanded=False):
        st.caption(
            "This section contains the fields hidden from the default Results table (e.g., extracted value strings, labels, keywords, "
            "and evidence snippets). Use this to quickly verify the extraction against the source filing."
        )

        # Limit expansion noise if a scan returns a lot of rows
        max_rows = min(len(results), 50)
        if len(results) > max_rows:
            st.info(f"Showing audit expanders for the first {max_rows} rows.")

        for _, r in results.head(max_rows).iterrows():
            t = str(r.get("ticker", "") or "").strip()
            filed = str(r.get("filing_date", "") or "").strip()
            name = str(r.get("new_executive", "") or "").strip()
            title = str(r.get("position", "") or "").strip()
            company = str(r.get("company_name", "") or "").strip()

            label = f"{t} — {filed}"
            if name:
                label += f" — {name}"
            if title:
                label += f" ({title})"
            if company:
                label += f" — {company}"

            with st.expander(label, expanded=False):
                # Source links
                src = str(r.get("source_8k_url", "") or "").strip()
                prim = str(r.get("primary_doc_url", "") or "").strip()
                acc = str(r.get("accession", "") or "").strip()

                links = []
                if src:
                    links.append(f"[Open 8‑K index]({src})")
                if prim:
                    links.append(f"[Open primary document]({prim})")
                if links:
                    st.markdown(" | ".join(links))
                if acc:
                    st.caption(f"Accession: {acc}")

                # Curated summary
                summ = str(r.get("compensation_summary", "") or "").strip()
                if summ:
                    st.markdown(f"**Comp summary:** {summ}")

                # Detail fields (strings)
                details = {
                    "One-time cash values": r.get("one_time_cash_values"),
                    "Annual/target equity values": r.get("equity_target_annual_values"),
                    "One-time equity values": r.get("equity_one_time_values"),
                    "One-time equity labels": r.get("equity_one_time_labels"),
                    "Other keywords": r.get("other_keywords"),
                }
                for k, v in details.items():
                    v = "" if v is None else str(v)
                    if v and v.strip() and v.strip().lower() != "nan":
                        st.markdown(f"**{k}:** {v}")

                # Evidence snippets
                ev = str(r.get("evidence_snippets", "") or "").strip()
                if ev:
                    with st.expander("Evidence snippets", expanded=False):
                        st.text(ev)

                # Confidence and metadata
                conf = r.get("confidence")
                evt = str(r.get("event_type", "") or "").strip()
                eff = str(r.get("effective_date", "") or "").strip()
                meta_parts = []
                if evt:
                    meta_parts.append(f"Event type: {evt}")
                if eff:
                    meta_parts.append(f"Effective: {eff}")
                if conf is not None and str(conf) != "nan":
                    try:
                        meta_parts.append(f"Confidence: {float(conf):.2f}")
                    except Exception:
                        pass
                if meta_parts:
                    st.caption(" • ".join(meta_parts))

if (not include_interim) and excluded_interim_lines:
    st.markdown("### Interim appointments excluded")
    st.text_area(
        "Companies where only an interim appointment was detected (excluded from Results)",
        value="\n".join(excluded_interim_lines),
        height=min(260, 80 + 18 * len(excluded_interim_lines)),
        disabled=True,
    )
