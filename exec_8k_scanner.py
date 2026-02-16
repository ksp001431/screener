#!/usr/bin/env python3
"""
exec_8k_scanner.py

Scan SEC Form 8-K / 8-K/A filings for *executive appointment* events (Item 5.02)
for a user-specified POSITION (e.g., CEO or CFO) and summarize compensation terms.

Inputs:
  - A list of tickers (or a tickers file)
  - The position to detect (e.g., CEO, CFO, or a full title like "Chief Financial Officer")
  - A look-back period in months (up to 36)

Outputs:
  - JSONL (structured)
  - CSV (flattened, spreadsheet-friendly)
  - Markdown (human review)
  - SQLite DB (dedupe + saved events)

Notes:
  - Extraction is heuristic; always review the underlying filing and exhibits.
  - Respect SEC fair-access guidance: set a declared User-Agent and keep request rate <= 10/sec.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
# -----------------------------
# SEC client + rate limiting
# -----------------------------

class SecEdgarClient:
    def __init__(
        self,
        user_agent: str,
        max_req_per_sec: float = 8.0,
        timeout: float = 30.0,
        cache_dir: Optional[Path] = None,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "Please set a declared User-Agent like 'YourOrg your.email@domain.com' "
                "(set env var SEC_USER_AGENT or pass --user-agent)."
            )
        if max_req_per_sec <= 0:
            raise ValueError("--max-rps must be > 0")
        self.user_agent = user_agent
        self.min_interval = 1.0 / max_req_per_sec
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "*/*",
            }
        )
        self._last_request_ts = 0.0
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _throttle(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_ts = time.time()

    def _cache_path(self, url: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.cache"

    def get_bytes(self, url: str) -> bytes:
        cache_path = self._cache_path(url)
        if cache_path and cache_path.exists():
            return cache_path.read_bytes()

        self._throttle()
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code == 429:
            # simple backoff
            time.sleep(2.0)
            self._throttle()
            resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.content
        if cache_path:
            cache_path.write_bytes(data)
        return data

    def get_text(self, url: str, encoding: Optional[str] = None) -> str:
        b = self.get_bytes(url)
        if encoding:
            return b.decode(encoding, errors="replace")
        try:
            return b.decode("utf-8", errors="replace")
        except Exception:
            return b.decode("latin-1", errors="replace")

    def get_json(self, url: str) -> Any:
        return json.loads(self.get_text(url))


# -----------------------------
# Data models
# -----------------------------

@dataclass(frozen=True)
class FilingRef:
    cik: str                  # zero-padded 10-digit string
    accession: str            # with dashes
    filing_date: str          # YYYY-MM-DD (best effort)
    form: str                 # 8-K or 8-K/A
    primary_doc: str          # e.g. d12345d8k.htm (filled from index)
    company_name: Optional[str] = None
    ticker: Optional[str] = None

    @property
    def cik_int(self) -> int:
        return int(self.cik)

    @property
    def accession_nodashes(self) -> str:
        return self.accession.replace("-", "")

    def base_dir_url(self) -> str:
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik_int}/{self.accession_nodashes}/"

    def primary_url(self) -> str:
        return self.base_dir_url() + self.primary_doc

    def index_html_url(self) -> str:
        return self.base_dir_url() + f"{self.accession}-index.html"


@dataclass
class ExtractedComp:
    base_salary: Optional[str] = None
    target_bonus: Optional[str] = None
    sign_on_bonus: Optional[str] = None
    # Numeric versions (USD / percent) for spreadsheets
    base_salary_usd: Optional[int] = None
    target_bonus_pct: Optional[float] = None
    target_bonus_usd: Optional[int] = None
    sign_on_bonus_usd: Optional[int] = None


    # One-time cash (signing / inducement / make-whole / one-time payments)
    one_time_cash_values: List[str] = dataclasses.field(default_factory=list)
    one_time_cash_values_usd: List[int] = dataclasses.field(default_factory=list)
    one_time_cash_usd_total: Optional[int] = None
    one_time_cash_details: List[str] = dataclasses.field(default_factory=list)


    equity_target_annual_values_usd: List[int] = dataclasses.field(default_factory=list)
    equity_target_annual_usd_total: Optional[int] = None

    equity_one_time_values_usd: List[int] = dataclasses.field(default_factory=list)
    equity_one_time_usd_total: Optional[int] = None

    equity_total_usd: Optional[int] = None


    # Curated equity fields (prefer $ values)
    equity_target_annual_values: List[str] = dataclasses.field(default_factory=list)
    equity_target_annual_details: List[str] = dataclasses.field(default_factory=list)

    equity_one_time_values: List[str] = dataclasses.field(default_factory=list)
    equity_one_time_labels: List[str] = dataclasses.field(default_factory=list)
    equity_one_time_details: List[str] = dataclasses.field(default_factory=list)

    # Raw equity mentions (snippets) kept for auditability
    equity_awards: List[str] = dataclasses.field(default_factory=list)

    severance: List[str] = dataclasses.field(default_factory=list)
    other: List[str] = dataclasses.field(default_factory=list)
    evidence_snippets: List[str] = dataclasses.field(default_factory=list)


@dataclass
class ExecEvent:
    event_id: str
    filing: FilingRef
    position_query: str
    detected: bool
    confidence: float
    person: Optional[str] = None
    matched_title: Optional[str] = None
    effective_date: Optional[str] = None
    event_type: Optional[str] = None  # hire | promotion | interim | unknown
    compensation: Optional[ExtractedComp] = None
    summary: Optional[str] = None


# -----------------------------
# Persistence (dedupe/state)
# -----------------------------

class Store:
    """
    Dedupe is per (accession, position_query), so running a CEO scan won't block a later CFO scan.
    Each detected executive match becomes a separate row (event_id primary key).
    """
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_accessions (
              accession TEXT NOT NULL,
              position_query TEXT NOT NULL,
              processed_at TEXT NOT NULL,
              PRIMARY KEY(accession, position_query)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exec_events (
              event_id TEXT PRIMARY KEY,
              accession TEXT NOT NULL,
              cik TEXT NOT NULL,
              filing_date TEXT,
              form TEXT,
              ticker TEXT,
              company_name TEXT,
              person TEXT,
              position_query TEXT,
              matched_title TEXT,
              effective_date TEXT,
              event_type TEXT,
              confidence REAL,
              summary TEXT,
              raw_json TEXT
            )
            """
        )
        self.conn.commit()

    def already_processed(self, accession: str, position_query: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM processed_accessions WHERE accession = ? AND position_query = ? LIMIT 1",
            (accession, position_query),
        )
        return cur.fetchone() is not None

    def mark_processed(self, filing: FilingRef, position_query: str) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO processed_accessions(accession, position_query, processed_at)
            VALUES (?, ?, ?)
            """,
            (
                filing.accession,
                position_query,
                dt.datetime.now(getattr(dt, "UTC", dt.timezone.utc)).isoformat(timespec="seconds").replace("+00:00","Z"),
            ),
        )
        self.conn.commit()

    def save_event(self, event: ExecEvent) -> None:
        raw = json.dumps(dataclasses.asdict(event), ensure_ascii=False)
        f = event.filing
        self.conn.execute(
            """
            INSERT OR REPLACE INTO exec_events(
              event_id, accession, cik, filing_date, form, ticker, company_name, person,
              position_query, matched_title, effective_date, event_type, confidence, summary, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                f.accession,
                f.cik,
                f.filing_date,
                f.form,
                f.ticker,
                f.company_name,
                event.person,
                event.position_query,
                event.matched_title,
                event.effective_date,
                event.event_type,
                event.confidence,
                event.summary,
                raw,
            ),
        )
        self.conn.commit()


# -----------------------------
# Utilities
# -----------------------------

WS_RE = re.compile(r"\s+")
ITEM_HDR_RE = re.compile(r"\bItem\s+(\d+)\.(\d+)\b", re.IGNORECASE)

def norm_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


def sentence_snippet(text: str, start: int, end: int, max_len: int = 420) -> str:
    """Extract a sentence-ish snippet around a match to avoid mixing adjacent comp terms."""
    if not text:
        return ""
    n = len(text)
    start = max(0, min(start, n))
    end = max(0, min(end, n))

    left_bound = max(0, start - max_len)
    left_candidates = [
        text.rfind(".", left_bound, start),
        text.rfind(";", left_bound, start),
        text.rfind("\n", left_bound, start),
    ]
    left = max(left_candidates)
    left = (left + 1) if left != -1 else left_bound

    right_bound = min(n, end + max_len)
    right_candidates = [
        text.find(".", end, right_bound),
        text.find(";", end, right_bound),
        text.find("\n", end, right_bound),
    ]
    right_hits = [i for i in right_candidates if i != -1]
    right = min(right_hits) if right_hits else right_bound

    return norm_ws(text[left:right])[:max_len]

def parse_date(s: str) -> Optional[str]:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return dt.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None

def money_norm(s: str) -> str:
    return norm_ws(s).replace("\u00a0", " ")

MONEY_PARSE_RE = re.compile(
    r"(?:US\$|\$)\s*"
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*(?P<scale>million|billion|thousand|m|bn|b|k)?",
    re.IGNORECASE,
)

PCT_PARSE_RE = re.compile(r"(?P<pct>\d{1,3}(?:\.\d+)?)\s*%")

def money_to_usd(value: str) -> Optional[int]:
    """Parse strings like '$800,000' or '$2.4 million' to an integer USD amount."""
    if not value:
        return None
    s = value.strip()
    m = MONEY_PARSE_RE.search(s)
    if not m:
        return None
    num = float(m.group("num").replace(",", ""))
    scale = (m.group("scale") or "").lower()
    if scale in ("million", "m"):
        num *= 1_000_000
    elif scale in ("billion", "bn", "b"):
        num *= 1_000_000_000
    elif scale in ("thousand", "k"):
        num *= 1_000
    return int(round(num))

def percent_to_float(value: str) -> Optional[float]:
    """Parse '120%' -> 120.0"""
    if not value:
        return None
    m = PCT_PARSE_RE.search(value)
    if not m:
        return None
    try:
        return float(m.group("pct"))
    except Exception:
        return None


def safe_event_id(*parts: str) -> str:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:24]

def slice_item_502(text: str) -> str:
    """
    Attempt to slice the text to the Item 5.02 section, to reduce false positives.
    If Item 5.02 isn't found, returns the full text.
    """
    m = re.search(r"\bItem\s+5\.02\b", text, re.IGNORECASE)
    if not m:
        return text
    start = m.start()
    tail = text[start:]
    m2 = ITEM_HDR_RE.search(tail, 1)
    # Find the next Item header after the first one (Item 5.02 itself)
    # We'll search for the second header occurrence.
    iters = list(ITEM_HDR_RE.finditer(tail))
    if len(iters) >= 2:
        end = iters[1].start()
        return tail[:end]
    return tail

def load_tickers_from_file(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # allow commas, spaces, newlines; allow comments with '#'
    toks: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0]
        toks.extend(re.split(r"[,\s]+", line.strip()))
    return [t.upper() for t in toks if t.strip()]

def daterange_inclusive(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    d = start
    while d <= end:
        yield d
        d = d + dt.timedelta(days=1)



def add_months(d: dt.date, months: int) -> dt.date:
    """
    Add (or subtract) whole months to a date, clamping the day to the end of month if needed.
    Example: add_months(date(2026, 3, 31), -1) -> date(2026, 2, 28)
    """
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return dt.date(y, m, day)

# -----------------------------
# Ticker/CIK mapping
# -----------------------------

@dataclass(frozen=True)
class TickerInfo:
    ticker: str
    cik: str
    title: str

def load_ticker_info(client: SecEdgarClient) -> Dict[str, TickerInfo]:
    """
    Loads the SEC's company_tickers.json mapping (ticker -> CIK + company title).
    Returns dict { 'AAPL': TickerInfo(ticker='AAPL', cik='0000320193', title='APPLE INC'), ... }
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    data = client.get_json(url)
    out: Dict[str, TickerInfo] = {}
    for _, row in data.items():
        t = (row.get("ticker") or "").strip()
        cik = row.get("cik_str")
        title = (row.get("title") or "").strip()
        if t and cik is not None:
            out[t.upper()] = TickerInfo(ticker=t.upper(), cik=str(cik).zfill(10), title=title)
    return out




# -----------------------------
# Ingestion via SEC Submissions API (fast for small watchlists)
# -----------------------------

def _filings_from_submissions_block(
    block: Dict[str, Any],
    cik10: str,
    ticker: Optional[str],
    company_name: Optional[str],
    start: dt.date,
    end: dt.date,
) -> List[FilingRef]:
    filings = (block.get("filings") or {})
    recent = (filings.get("recent") or {})
    forms = recent.get("form") or []
    accession_numbers = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    primary_docs = recent.get("primaryDocument") or []

    out: List[FilingRef] = []
    for form, acc, fdate, pdoc in zip(forms, accession_numbers, filing_dates, primary_docs):
        if form not in ("8-K", "8-K/A"):
            continue
        d_str = parse_date(str(fdate)) or str(fdate)
        try:
            d_obj = dt.date.fromisoformat(d_str)
        except Exception:
            continue
        if d_obj < start or d_obj > end:
            continue

        pd = (str(pdoc) if pdoc is not None else "").strip().split()[0]
        out.append(
            FilingRef(
                cik=cik10,
                accession=str(acc),
                filing_date=d_str,
                form=str(form),
                primary_doc=pd,  # may be blank; we'll still validate via index.html
                company_name=company_name,
                ticker=ticker,
            )
        )
    return out


def filings_from_submissions_range(
    client: SecEdgarClient,
    cik10: str,
    start: dt.date,
    end: dt.date,
    ticker: Optional[str] = None,
    company_title_fallback: Optional[str] = None,
) -> List[FilingRef]:
    """
    Pull 8-K / 8-K/A filings for a single company over [start, end] (inclusive)
    using the SEC Submissions API (data.sec.gov). Handles pagination via filings.files.

    This is typically far more efficient than scanning daily master indexes when the watchlist is small.
    """
    main_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    data = client.get_json(main_url)

    company_name = (data.get("name") or company_title_fallback or "").strip() or company_title_fallback

    out: List[FilingRef] = []
    out.extend(_filings_from_submissions_block(data, cik10, ticker, company_name, start, end))

    files = (((data.get("filings") or {}).get("files")) or [])
    for f in files:
        name = (f.get("name") or "").strip()
        if not name:
            continue

        # Only fetch file segments that overlap our date range
        try:
            frm = dt.date.fromisoformat(str(f.get("filingFrom")))
            to = dt.date.fromisoformat(str(f.get("filingTo")))
        except Exception:
            frm = None
            to = None

        if frm and to:
            if to < start or frm > end:
                continue

        seg_url = "https://data.sec.gov/submissions/" + name
        try:
            seg = client.get_json(seg_url)
        except requests.HTTPError:
            continue
        out.extend(_filings_from_submissions_block(seg, cik10, ticker, company_name, start, end))

    # Dedupe by accession
    dedup: Dict[str, FilingRef] = {}
    for fr in out:
        dedup[fr.accession] = fr

    # Sort newest-first
    res = list(dedup.values())
    res.sort(key=lambda x: x.filing_date, reverse=True)
    return res

# -----------------------------
# Ingestion via daily master index (scales to big watchlists)
# -----------------------------

def filings_from_daily_master_index_filtered(
    client: SecEdgarClient,
    day: dt.date,
    allowed_ciks: Set[str],
    cik_to_ticker: Dict[str, str],
    cik_to_title: Dict[str, str],
) -> List[FilingRef]:
    """
    Parses the SEC daily master index for a given date and returns 8-K/8-K/A filings
    for CIKs in allowed_ciks.

    Format: CIK|Company Name|Form Type|Date Filed|Filename
    """
    year = day.year
    qtr = (day.month - 1) // 3 + 1
    yyyymmdd = day.strftime("%Y%m%d")

    # SEC daily master index files are often not published on weekends (and some holidays).
    # Skipping weekends avoids unnecessary errors and reduces load.
    if day.weekday() >= 5:  # 5=Sat, 6=Sun
        return []

    url = f"https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{qtr}/master.{yyyymmdd}.idx"
    dir_url = f"https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{qtr}/"

    try:
        txt = client.get_text(url, encoding="latin-1")
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)

        # Many non-business days will 404 (no index published)
        if status == 404:
            return []

        # Some non-published days have been observed to return 403 instead of 404.
        # Differentiate "missing file" vs "blocked" by checking the quarter directory listing.
        if status == 403:
            try:
                listing_html = client.get_text(dir_url)
                if f"master.{yyyymmdd}.idx" not in listing_html:
                    return []
            except Exception:
                # If we can't confirm it's missing, don't silently ignore a potential real block.
                pass

            raise RuntimeError(
                f"SEC returned 403 for daily index: {url}. "
                "If this file exists in the quarter directory listing, you may be blocked "
                "(check declared User-Agent / reduce max-rps / shared egress IP)."
            ) from e

        raise

    lines = txt.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("----"):
            start = i + 1
            break

    out: List[FilingRef] = []
    for line in lines[start:]:
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik_raw, company, form, filed, filename = parts
        form = form.strip()
        if form not in ("8-K", "8-K/A"):
            continue

        try:
            cik10 = str(int(cik_raw)).zfill(10)
        except Exception:
            continue
        if cik10 not in allowed_ciks:
            continue

        filed_date = parse_date(filed) or filed.strip()
        acc_match = re.search(r"/(\d{10}-\d{2}-\d{6})\.txt$", filename)
        if not acc_match:
            continue
        accession = acc_match.group(1)

        comp_name = (company or "").strip() or cik_to_title.get(cik10)

        out.append(
            FilingRef(
                cik=cik10,
                accession=accession,
                filing_date=filed_date,
                form=form,
                primary_doc="",  # filled later from index.html
                company_name=comp_name,
                ticker=cik_to_ticker.get(cik10),
            )
        )
    return out


# -----------------------------
# Filing document retrieval
# -----------------------------

@dataclass
class FilingDocument:
    filename: str
    description: str
    doc_type: str
    url: str

def parse_filing_index_html(client: SecEdgarClient, filing: FilingRef) -> List[FilingDocument]:
    """
    Parse the SEC filing "-index.html" page and return a list of documents.

    IMPORTANT: The "Document" column can include Inline XBRL viewer links like:
        ix?doc=/Archives/edgar/data/.../file.htm
    and/or a separate "iXBRL" link/label. We must avoid mistakenly treating those
    as the primary document filename (e.g., "ix" or "file.htm iXBRL").
    """
    html = client.get_text(filing.index_html_url())
    soup = BeautifulSoup(html, "lxml")

    docs: List[FilingDocument] = []
    table = soup.find("table", class_="tableFile")
    if not table:
        return docs

    rows = table.find_all("tr")

    # Used to derive a safe "filename" from hrefs
    base = filing.base_dir_url()
    base_noscheme = base.replace("https://www.sec.gov", "")

    # Local import to avoid adding a hard dependency elsewhere
    from urllib.parse import urlparse, parse_qs, unquote

    def _unwrap_ix_href(href: str) -> str:
        """
        If href is an Inline XBRL viewer link like 'ix?doc=/Archives/.../file.htm',
        return the underlying doc path; otherwise return href unchanged.
        """
        h = href.strip()
        low = h.lower()
        if low.startswith("ix?") or low.startswith("/ix?"):
            abs_ix = "https://www.sec.gov" + h if h.startswith("/") else "https://www.sec.gov/" + h
            q = parse_qs(urlparse(abs_ix).query)
            doc = q.get("doc", [None])[0]
            if doc:
                return unquote(doc)
        return h

    def _looks_like_file(s: str) -> bool:
        return bool(re.search(r"\.(?:htm|html|txt|xml|pdf)$", s or "", flags=re.IGNORECASE))

    for r in rows[1:]:
        cols = r.find_all("td")
        if len(cols) < 4:
            continue

        description = norm_ws(cols[1].get_text(" ", strip=True))
        doc_type = norm_ws(cols[3].get_text(" ", strip=True))

        links = cols[2].find_all("a")
        if not links:
            continue

        # Choose the best candidate link for the underlying document.
        # Prefer a link whose visible text OR underlying href/doc path looks like a real file,
        # and avoid selecting the iXBRL label link.
        best_a = None
        best_score = -1
        best_href_resolved = None

        for a in links:
            t = (a.get_text(" ", strip=True) or "").strip()
            if t.lower() == "ixbrl":
                continue  # label link, not the doc

            href = (a.get("href") or "").strip()
            if not href:
                continue

            href_resolved = _unwrap_ix_href(href)

            # Score candidates
            score = 0
            if _looks_like_file(t):
                score += 3
            if _looks_like_file(href_resolved):
                score += 3

            # Penalize bare ix/ixviewer links if not unwrapped
            href_noq = href.split("?", 1)[0].split("#", 1)[0].strip().lower()
            if href_noq in ("ix", "/ix") or "ixviewer" in href_noq:
                score -= 2

            if score > best_score:
                best_score = score
                best_a = a
                best_href_resolved = href_resolved

        # Fallback if nothing scored well (rare)
        if best_a is None:
            best_a = links[0]
            best_href_resolved = _unwrap_ix_href((best_a.get("href") or "").strip())

        href = (best_href_resolved or "").strip()
        if not href:
            continue

        # Build absolute fetch URL
        if href.startswith("http"):
            url = href
            href_clean = href
        elif href.startswith("/"):
            url = "https://www.sec.gov" + href
            href_clean = href
        else:
            # Some hrefs are relative to the filing directory (e.g., "file.htm")
            url = filing.base_dir_url() + href
            href_clean = href

        # Derive a clean filename/relative path from the href (strip query/fragment)
        href_clean = href_clean.split("?", 1)[0].split("#", 1)[0]

        if href_clean.startswith(base):
            filename = href_clean[len(base):]
        elif href_clean.startswith(base_noscheme):
            filename = href_clean[len(base_noscheme):]
        elif href_clean.startswith("http") or href_clean.startswith("/"):
            filename = href_clean.rsplit("/", 1)[-1]
        else:
            filename = href_clean  # preserve relative subpaths if any

        filename = filename.strip().split()[0]  # safety net vs "file.htm iXBRL"

        docs.append(FilingDocument(filename, description, doc_type, url))

    return docs
def pick_primary_doc(filing: FilingRef, docs: List[FilingDocument]) -> FilingRef:
    if filing.primary_doc:
        return filing

    for d in docs:
        if d.doc_type in ("8-K", "8-K/A"):
            return dataclasses.replace(filing, primary_doc=d.filename.split()[0])

    for d in docs:
        if re.search(r"8k(\.htm|\.html|\.txt)$", d.filename, flags=re.I):
            return dataclasses.replace(filing, primary_doc=d.filename.split()[0])

    for d in docs:
        if d.filename.lower().endswith((".htm", ".html")):
            return dataclasses.replace(filing, primary_doc=d.filename.split()[0])

    # last resort: first doc
    if docs:
        return dataclasses.replace(filing, primary_doc=docs[0].filename.split()[0])

    return filing

def html_to_text(html: str) -> str:
    """
    Convert (X)HTML to plain text.

    Some SEC documents (notably iXBRL / XHTML) start with an XML prolog (<?xml ...?>).
    BeautifulSoup will warn if you parse XML as HTML; this function picks an XML parser
    for those cases and also suppresses the warning as a safety net.
    """
    s = html.lstrip()
    # iXBRL is typically XHTML (XML prolog + <html xmlns=...>), so use an XML parser there.
    parser = "lxml-xml" if s.startswith("<?xml") else "lxml"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, parser)

    # For HTML, remove script/style noise. For XML/XHTML, these tags may still exist and it's safe to remove.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return norm_ws(soup.get_text(" ", strip=True))

def fetch_text_for_doc(client: SecEdgarClient, doc_url: str) -> str:
    # Avoid PDFs (no OCR here); still downloadable but not parseable as text reliably.
    if doc_url.lower().endswith(".pdf"):
        return ""
    raw = client.get_text(doc_url)
    if "<html" in raw.lower() or "<!doctype" in raw.lower():
        return html_to_text(raw)
    return norm_ws(raw)


# -----------------------------
# Position matching + executive event detection
# -----------------------------

POSITION_SYNONYMS: Dict[str, List[str]] = {
    "CEO": [
        "Chief Executive Officer",
        "CEO",
        "Principal Executive Officer",
        "President and Chief Executive Officer",
        "President & Chief Executive Officer",
    ],
    "CFO": [
        "Chief Financial Officer",
        "CFO",
        "Principal Financial Officer",
    ],
    "COO": [
        "Chief Operating Officer",
        "COO",
        "Principal Operating Officer",
    ],
    "CTO": ["Chief Technology Officer", "CTO"],
    "CIO": ["Chief Information Officer", "CIO"],
    "CMO": ["Chief Marketing Officer", "CMO"],
    "CHRO": ["Chief Human Resources Officer", "CHRO"],
    "CAO": ["Chief Accounting Officer", "CAO", "Principal Accounting Officer"],
    "CLO": ["Chief Legal Officer", "CLO"],
}

APPOINT_VERBS = [
    r"appoint(?:ed|s)?",
    r"name(?:d|s)?",
    r"elect(?:ed|s)?",
    r"promot(?:ed|es|ion)",
    r"select(?:ed|s)?",
    r"designat(?:ed|es)?",
    r"succ(?:eed|ession)",
    r"hire(?:d|s)?",
]

NAME_RE = r"(?:Mr\.|Ms\.|Mrs\.|Dr\.)?\s*(?:(?-i:[A-Z][A-Za-z\.\-’\']+))(?:\s+(?:(?-i:[A-Z][A-Za-z\.\-’\']+))){1,4}"

EFFECTIVE_DATE_RE = re.compile(
    r"(?:effective|with effect|as of)\s+(?:on\s+)?(?P<date>(?:\w+\s+\d{1,2},\s+\d{4})|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})",
    re.IGNORECASE,
)

def build_position_patterns(position_query: str) -> Tuple[str, List[str]]:
    pos = position_query.strip()
    if not pos:
        raise ValueError("position_query cannot be empty")
    up = pos.upper()

    if up in POSITION_SYNONYMS:
        return up, POSITION_SYNONYMS[up]

    # If they pass full title ("Chief Financial Officer"), also match a derived acronym ("CFO")
    words = re.findall(r"[A-Za-z]+", pos)
    stop = {"and", "of", "the", "for", "to", "a", "an"}
    letters = [w[0].upper() for w in words if w.lower() not in stop]
    acronym = "".join(letters) if 2 <= len(letters) <= 6 else ""

    pats = [pos]
    if acronym and acronym != up:
        pats.append(acronym)

    return pos, pats

@dataclass
class ExecMatch:
    name: str
    title: str
    context: str
    is_interim: bool
    event_type: str
    effective_date: Optional[str]

def _compile_exec_regexes(position_titles: List[str]) -> List[re.Pattern]:
    """
    Build regexes that try to catch common appointment language in Item 5.02.

    Important nuance: filings often embed the target role inside a longer title, e.g.
      "Executive Vice President and Chief Financial Officer"
    so we match titles that *contain* one of the requested position tokens.
    """
    # escape titles to treat them as literals inside regex
    title_alt = "|".join(re.escape(t) for t in position_titles if t.strip())

    # Title phrase that CONTAINS one of the key titles, bounded by sentence-ish punctuation.
    # Example match: "Executive Vice President and Chief Financial Officer"
    title_re = rf"(?P<title>[^.;\n]{{0,100}}?\b(?:{title_alt})\b[^.;\n]{{0,100}}?)"
    verbs_alt = "|".join(APPOINT_VERBS)

    patterns = [
        # "appointed John Doe as Chief Financial Officer"
        rf"\b(?P<lemma>{verbs_alt})\b\s+(?P<name>{NAME_RE})\s+(?:as|to serve as|to be)\s+(?:the\s+)?(?:its\s+)?{title_re}",
        # "John Doe was appointed as Chief Financial Officer"
        rf"(?P<name>{NAME_RE})\s+(?:has been|was|is)\s+\b(?P<lemma>{verbs_alt})\b\s+(?:as|to serve as|to be)\s+(?:the\s+)?(?:its\s+)?{title_re}",
        # "the appointment of John Doe to serve as Chief Financial Officer"
        rf"\bappointment\s+of\s+(?P<name>{NAME_RE})\s+(?:as|to serve as|to be)\s+(?:the\s+)?(?:its\s+)?{title_re}",
        # "John Doe will serve as Chief Financial Officer"
        rf"(?P<name>{NAME_RE})\s+will\s+(?:serve|act)\s+as\s+(?:the\s+)?(?:its\s+)?{title_re}",
        # "John Doe will assume the role of Chief Financial Officer"
        rf"(?P<name>{NAME_RE})\s+will\s+assum(?:e|ing)\s+(?:the\s+)?(?:role\s+of\s+)?{title_re}",
    ]
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]
def detect_exec_matches(text: str, position_query: str) -> Tuple[bool, Dict[str, Any], List[ExecMatch]]:
    """
    Detect matches for the requested position in Item 5.02 context.
    Returns:
      detected_bool, details_dict, list_of_matches
    """
    canonical_pos, pos_titles = build_position_patterns(position_query)

    full_text = text
    item_text = slice_item_502(full_text)
    has_item_502 = bool(re.search(r"\bItem\s+5\.02\b", full_text, re.IGNORECASE))

    regexes = _compile_exec_regexes(pos_titles)
    matches: List[ExecMatch] = []

    for rx in regexes:
        for m in rx.finditer(item_text):
            name = norm_ws(m.group("name"))
            title = norm_ws(m.group("title"))

            ctx = norm_ws(item_text[max(0, m.start()-260): m.end()+260])
            is_interim = bool(re.search(r"\binterim\b", ctx, re.IGNORECASE))

            # effective date: first one in nearby context (prefer local)
            eff = None
            eff_m = EFFECTIVE_DATE_RE.search(ctx)
            if eff_m:
                eff = parse_date(eff_m.group("date"))

            et = "unknown"
            if is_interim:
                et = "interim"
            elif re.search(r"\bpromot(?:ed|ion)\b", ctx, re.IGNORECASE):
                et = "promotion"
            elif re.search(r"\b(hire(?:d)?|join(?:ed|ing))\b", ctx, re.IGNORECASE):
                et = "hire"

            matches.append(
                ExecMatch(
                    name=name,
                    title=title,
                    context=ctx,
                    is_interim=is_interim,
                    event_type=et,
                    effective_date=eff,
                )
            )

    # Dedupe by (name,title)
    seen = set()
    deduped: List[ExecMatch] = []
    for mm in matches:
        k = (mm.name.lower(), mm.title.lower())
        if k in seen:
            continue
        seen.add(k)
        deduped.append(mm)

    detected = bool(deduped)

    details: Dict[str, Any] = {
        "canonical_position": canonical_pos,
        "has_item_502": has_item_502,
        "match_count": len(deduped),
    }
    if deduped:
        details["examples"] = [dataclasses.asdict(deduped[0])]

    return detected, details, deduped


# -----------------------------
# Compensation extraction (heuristic)
# -----------------------------

DOLLAR_RE = r"(?:US\$|[$€£])\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|thousand|m|bn|b|k))?"
COMP_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("base_salary", re.compile(rf"\b(annual base salary|base salary)\b[^.:\n]{{0,160}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("base_salary", re.compile(rf"\b(annual salary|salary)\b[^.:\n]{{0,160}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("target_bonus", re.compile(r"\b(target (?:annual )?bonus|annual bonus target|target cash bonus|bonus opportunity)\b[^.:\n]{0,220}?(\d{1,3}\s?%)", re.IGNORECASE)),
    ("target_bonus", re.compile(rf"\b(target (?:annual )?bonus|annual bonus target|target cash bonus|bonus opportunity)\b[^.:\n]{{0,220}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("sign_on_bonus", re.compile(rf"\b(sign(?:ing)?-?on bonus|one-time (?:cash )?(?:signing|sign-on) bonus)\b[^.:\n]{{0,220}}?({DOLLAR_RE})", re.IGNORECASE)),

    ("one_time_cash", re.compile(rf"\b(sign(?:ing)?(?:-?on)?\s+bonus|signing bonus|sign-?on bonus|inducement|make-?whole|make whole)\b[^.:\n]{{0,240}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("one_time_cash", re.compile(rf"({DOLLAR_RE})[^.:\n]{{0,80}}?\b(sign(?:ing)?(?:-?on)?\s+bonus|signing bonus|sign-?on bonus|inducement|make-?whole|make whole)\b", re.IGNORECASE)),  # money-before-label

    ("one_time_cash", re.compile(rf"\b(one-?time|one time)\b[^.:\n]{{0,80}}?\b(?:cash\s+)?(?:bonus|payment)\b[^.:\n]{{0,240}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("one_time_cash", re.compile(rf"\blump sum payment\b[^.:\n]{{0,240}}?({DOLLAR_RE})", re.IGNORECASE)),

    # Equity / LTI - shares/units or $ value
    ("equity_award", re.compile(r"\b(restricted stock units|RSUs)\b[^.:\n]{0,260}?\b(\d{1,3}(?:,\d{3})*)\b\s+(?:shares|units)", re.IGNORECASE)),
    ("equity_award", re.compile(r"\b(stock option|options?)\b[^.:\n]{0,280}?\b(\d{1,3}(?:,\d{3})*)\b\s+(?:shares)", re.IGNORECASE)),
    ("equity_award", re.compile(rf"\b(restricted stock units|RSUs|stock options?|option award)\b[^.:\n]{{0,260}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("equity_award", re.compile(rf"\b(long-?term incentive|LTI|equity incentive|target award value|grant date value)\b[^.:\n]{{0,260}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("equity_award", re.compile(rf"\b(annual|target)\b[^.:\n]{{0,240}}?\b(long-?term incentive|LTI|equity)\b[^.:\n]{{0,240}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("equity_award", re.compile(rf"\b(equity\s+grant\s+opportunity|annual\s+equity\s+grant\s+opportunity|target\s+annual\s+equity\s+grant\s+opportunity|equity\s+grant)\b[^.:\n]{{0,240}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("equity_award", re.compile(rf"\b(inducement|make-?whole|make whole|sign(?:ing)?|new hire|initial|one-?time)\b[^.:\n]{{0,240}}?\b(award|grant|RSUs?|restricted stock|equity|stock options?)\b[^.:\n]{{0,240}}?({DOLLAR_RE})", re.IGNORECASE)),

    # Severance / CIC
    ("severance", re.compile(r"\bseverance\b[^.:\n]{0,320}?\b(\d{1,2})\s+(?:months?|month)\b", re.IGNORECASE)),
    ("severance", re.compile(r"\bchange in control\b[^.:\n]{0,340}?\b(\d{1,2})\s+(?:months?|month)\b", re.IGNORECASE)),
    ("severance", re.compile(r"\bseverance\b[^.:\n]{0,340}?\b(\d(?:\.\d)?)\s?x\b", re.IGNORECASE)),
]

def _dedupe(seq: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


MONEY_FIND_RE = re.compile(DOLLAR_RE)

EQUITY_CONTEXT_KWS = (
    "equity",
    "long-term incentive",
    "long term incentive",
    "lti",
    "restricted stock",
    "rsu",
    "stock option",
    "options",
    "performance share",
    "psu",
    "grant",
    "award",
)

ONE_TIME_KWS = (
    "one-time",
    "one time",
    "signing",
    "sign-on",
    "sign on",
    "inducement",
    "make-whole",
    "make whole",
    "new hire",
    "initial",
    "commencement",
    "makewhole",
    "make whole",
    "special",
)

ANNUAL_TARGET_KWS = (
    "annual",
    "target",
    "target award value",
    "lti target",
    "long-term incentive target",
    "long term incentive target",
    "annual long-term incentive",
    "annual equity",
    "equity incentive target",
)

def _extract_money_strings(s: str) -> List[str]:
    return [money_norm(x) for x in MONEY_FIND_RE.findall(s or "")]

def _equity_labels(s_low: str) -> List[str]:
    labels: List[str] = []
    if "inducement" in s_low:
        labels.append("inducement")
    if "make-whole" in s_low or "make whole" in s_low:
        labels.append("make-whole")
    if "signing" in s_low or "sign-on" in s_low or "sign on" in s_low:
        labels.append("signing")
    if "new hire" in s_low or "initial" in s_low or "commencement" in s_low:
        labels.append("new-hire")
    if "one-time" in s_low or "one time" in s_low or "special" in s_low:
        labels.append("one-time")
    return _dedupe(labels)

def _classify_equity_snippet(s: str) -> str:
    s_low = (s or "").lower()
    if not any(k in s_low for k in EQUITY_CONTEXT_KWS):
        return "other"

    one_time_score = 0
    annual_score = 0

    if any(k in s_low for k in ONE_TIME_KWS):
        one_time_score += 3
    if "grant date value" in s_low:
        one_time_score += 2

    if any(k in s_low for k in ANNUAL_TARGET_KWS):
        annual_score += 3
    # light signals
    if "target" in s_low:
        annual_score += 1
    if re.search(r"\b20\d{2}\b", s_low):
        annual_score += 1

    if one_time_score > annual_score and one_time_score >= 2:
        return "one_time"
    if annual_score >= 2:
        return "annual_target"

    # Tie-breakers
    if any(k in s_low for k in ("one-time", "one time", "inducement", "make-whole", "make whole", "signing", "new hire", "initial")):
        return "one_time"
    if "annual" in s_low or "target" in s_low:
        return "annual_target"

    return "other"


def extract_compensation(text: str) -> ExtractedComp:
    comp = ExtractedComp()

    for field, pat in COMP_PATTERNS:
        for m in pat.finditer(text):
            snippet = norm_ws(text[max(0, m.start()-180): m.end()+180])
            comp.evidence_snippets.append(snippet)

            if field == "base_salary" and comp.base_salary is None:
                comp.base_salary = money_norm(m.group(2))
                comp.base_salary_usd = money_to_usd(comp.base_salary)
            elif field == "target_bonus" and comp.target_bonus is None:
                comp.target_bonus = money_norm(m.group(2))
                comp.target_bonus_pct = percent_to_float(comp.target_bonus)
                if comp.target_bonus_pct is None:
                    comp.target_bonus_usd = money_to_usd(comp.target_bonus)
            elif field == "sign_on_bonus" and comp.sign_on_bonus is None:
                comp.sign_on_bonus = money_norm(m.group(2))
                comp.sign_on_bonus_usd = money_to_usd(comp.sign_on_bonus)
            elif field == "one_time_cash":
                cash_snip = sentence_snippet(text, m.start(), m.end())
                comp.one_time_cash_details.append(cash_snip)
                comp.one_time_cash_values.extend(_extract_money_strings(cash_snip))

            elif field == "equity_award":
                comp.equity_awards.append(sentence_snippet(text, m.start(), m.end()))
            elif field == "severance":
                comp.severance.append(snippet[:360])

    # Curate equity into annual/target vs one-time buckets (prefer $ values)
    for snip in comp.equity_awards:
        monies = _extract_money_strings(snip)
        if not monies:
            continue

        cat = _classify_equity_snippet(snip)
        if cat == "annual_target":
            comp.equity_target_annual_details.append(snip[:360])
            comp.equity_target_annual_values.extend(monies)
        elif cat == "one_time":
            comp.equity_one_time_details.append(snip[:360])
            comp.equity_one_time_values.extend(monies)
            comp.equity_one_time_labels.extend(_equity_labels(snip.lower()))

    for kw in (
        "relocation",
        "car allowance",
        "housing",
        "tax gross-up",
        "gross-up",
        "clawback",
        "vesting",
        "performance-based",
        "change in control",
        "CIC",
    ):
        if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
            comp.other.append(kw)

    # Dedup lists
    comp.equity_awards = _dedupe(comp.equity_awards)
    comp.equity_target_annual_values = _dedupe(comp.equity_target_annual_values)
    comp.equity_target_annual_details = _dedupe(comp.equity_target_annual_details)
    comp.equity_one_time_values = _dedupe(comp.equity_one_time_values)
    comp.equity_one_time_labels = _dedupe(comp.equity_one_time_labels)
    comp.one_time_cash_values = _dedupe(comp.one_time_cash_values)
    comp.one_time_cash_details = _dedupe(comp.one_time_cash_details)
    # Numeric equity (USD) + totals for spreadsheets
    def _dedupe_int(seq: List[Optional[int]]) -> List[int]:
        seen_i: Set[int] = set()
        out_i: List[int] = []
        for v in seq:
            if v is None:
                continue
            if v in seen_i:
                continue
            seen_i.add(v)
            out_i.append(v)
        return out_i

    # Numeric one-time cash (USD) + total
    comp.one_time_cash_values_usd = _dedupe_int([money_to_usd(x) for x in comp.one_time_cash_values])
    comp.one_time_cash_usd_total = int(sum(comp.one_time_cash_values_usd)) if comp.one_time_cash_values_usd else None

    comp.equity_target_annual_values_usd = _dedupe_int([money_to_usd(x) for x in comp.equity_target_annual_values])
    comp.equity_one_time_values_usd = _dedupe_int([money_to_usd(x) for x in comp.equity_one_time_values])

    comp.equity_target_annual_usd_total = sum(comp.equity_target_annual_values_usd) if comp.equity_target_annual_values_usd else None
    comp.equity_one_time_usd_total = sum(comp.equity_one_time_values_usd) if comp.equity_one_time_values_usd else None

    if comp.equity_target_annual_usd_total is not None or comp.equity_one_time_usd_total is not None:
        comp.equity_total_usd = int((comp.equity_target_annual_usd_total or 0) + (comp.equity_one_time_usd_total or 0))

    comp.equity_one_time_details = _dedupe(comp.equity_one_time_details)

    comp.severance = _dedupe(comp.severance)
    comp.other = _dedupe(comp.other)
    comp.evidence_snippets = _dedupe(comp.evidence_snippets)
    # Compute target_bonus_usd from pct if needed (common in filings)
    if comp.target_bonus_usd is None and comp.target_bonus_pct is not None and comp.base_salary_usd is not None:
        try:
            comp.target_bonus_usd = int(round(comp.base_salary_usd * (float(comp.target_bonus_pct) / 100.0)))
        except Exception:
            pass


    return comp

def compensation_brief(comp: ExtractedComp) -> str:
    bits: List[str] = []
    if comp.base_salary:
        bits.append(f"Base salary {comp.base_salary}")
    if comp.target_bonus:
        bits.append(f"Target bonus {comp.target_bonus}")
    if comp.one_time_cash_usd_total is not None:
        bits.append(f"One-time cash ${comp.one_time_cash_usd_total:,}")

    if comp.equity_target_annual_values:
        bits.append("Equity target/annual " + ", ".join(comp.equity_target_annual_values))
    if comp.equity_one_time_values:
        lab = f" ({', '.join(comp.equity_one_time_labels)})" if comp.equity_one_time_labels else ""
        bits.append("One-time equity " + ", ".join(comp.equity_one_time_values) + lab)

    # Fallback if we saw equity snippets but couldn't extract $ values
    if not comp.equity_target_annual_values and not comp.equity_one_time_values and comp.equity_awards:
        bits.append(f"Equity mentions {len(comp.equity_awards)}")

    if comp.severance:
        bits.append(f"Severance/CIC mentions {len(comp.severance)}")
    if comp.other:
        bits.append("Other: " + ", ".join(comp.other))
    return "; ".join(bits) if bits else "No comp terms detected in scanned docs/exhibits."

# -----------------------------
# Summarization
# -----------------------------

def build_summary(event: ExecEvent) -> str:
    f = event.filing
    comp = event.compensation

    hdr = f"{f.company_name or f.cik} — {f.form} filed {f.filing_date} (accession {f.accession})"
    if f.ticker:
        hdr = f"{f.ticker} / " + hdr

    lines = [hdr]
    if event.person or event.matched_title:
        who = " • ".join([p for p in [event.person, event.matched_title] if p])
        lines.append(f"New executive detected: {who} ({event.event_type or 'unknown'})")
    if event.effective_date:
        lines.append(f"Effective date (best effort): {event.effective_date}")
    if comp:
        lines.append("Compensation (heuristic): " + compensation_brief(comp))
    lines.append(f"Confidence: {event.confidence:.2f}")
    lines.append(f"Primary doc: {f.primary_url()}")
    return "\n".join(lines)


# -----------------------------
# Orchestration
# -----------------------------

def process_filing(
    client: SecEdgarClient,
    store: Store,
    filing: FilingRef,
    position_query: str,
    max_exhibits: int = 8,
    force: bool = False,
) -> List[ExecEvent]:
    if (not force) and store.already_processed(filing.accession, position_query):
        return []

    docs = parse_filing_index_html(client, filing)
    filing2 = pick_primary_doc(filing, docs)

    primary_text = fetch_text_for_doc(client, filing2.primary_url())
    detected, _, matches = detect_exec_matches(primary_text, position_query=position_query)

    # Candidate exhibits likely to contain comp terms
    comp_text_blobs: List[str] = [primary_text]
    likely_docs: List[FilingDocument] = []
    for d in docs:
        fn = (d.filename or "").lower()
        if fn.endswith(".pdf"):
            continue
        t = (d.doc_type or "").upper()
        desc = (d.description or "").lower()
        if t.startswith("EX-10") or any(
            k in desc for k in ("employment", "offer", "compens", "severance", "change in control", "indemnif", "award")
        ):
            likely_docs.append(d)
        elif t.startswith("EX-99") and ("press" in desc or "release" in desc):
            likely_docs.append(d)

    for d in likely_docs[:max_exhibits]:
        try:
            comp_text_blobs.append(fetch_text_for_doc(client, d.url))
        except requests.HTTPError:
            continue

    combined_text = "\n\n".join([t for t in comp_text_blobs if t])

    # If no match from the primary doc, try combined (primary + exhibits)
    if not matches and combined_text:
        detected2, _, matches2 = detect_exec_matches(combined_text, position_query=position_query)
        if detected2:
            matches = matches2
            detected = True

    if not detected or not matches:
        store.mark_processed(filing2, position_query)
        return []

    # Confidence (simple, interpretable)
    confidence = 0.55
    if re.search(r"\bItem\s+5\.02\b", combined_text, re.IGNORECASE):
        confidence += 0.15
    confidence += min(0.20, 0.05 * len(matches))
    confidence = min(1.0, confidence)

    comp = extract_compensation(combined_text) if combined_text else ExtractedComp()

    events: List[ExecEvent] = []
    for mm in matches[:3]:  # cap for sanity
        event_id = safe_event_id(filing2.accession, position_query, mm.name, mm.title)
        evt = ExecEvent(
            event_id=event_id,
            filing=filing2,
            position_query=position_query,
            detected=True,
            confidence=confidence,
            person=mm.name,
            matched_title=mm.title,
            effective_date=mm.effective_date,
            event_type=mm.event_type,
            compensation=comp,
        )
        evt.summary = build_summary(evt)
        store.save_event(evt)
        events.append(evt)

    store.mark_processed(filing2, position_query)
    return events


def write_outputs(
    events: List[ExecEvent],
    out_jsonl: Optional[Path],
    out_csv: Optional[Path],
    out_md: Optional[Path],
) -> None:
    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("a", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(dataclasses.asdict(e), ensure_ascii=False) + "\n")

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "ticker",
            "company_name",
            "filing_date",
            "source_8k_url",
            "accession",
            "form",
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
            "one_time_cash_values",
            "equity_one_time_usd_total",
            "equity_one_time_values",
            "equity_one_time_labels",
            "equity_target_annual_values",
            "other_keywords",
            "compensation_summary",
            "primary_doc_url",
            "confidence",
        ]
        exists = out_csv.exists()
        with out_csv.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                w.writeheader()
            for e in events:
                comp = e.compensation or ExtractedComp()

                # Derived numeric values: salary + target bonus ($) + annual/ongoing equity/LTI target ($)
                base = comp.base_salary_usd
                bonus_usd = comp.target_bonus_usd
                if bonus_usd is None and comp.target_bonus_pct is not None and base is not None:
                    try:
                        bonus_usd = int(round(base * (float(comp.target_bonus_pct) / 100.0)))
                    except Exception:
                        bonus_usd = None
                equity_annual = comp.equity_target_annual_usd_total
                target_total = None
                if base is not None or bonus_usd is not None or equity_annual is not None:
                    target_total = int((base or 0) + (bonus_usd or 0) + (equity_annual or 0))

                w.writerow(
                    {
                        "ticker": e.filing.ticker or "",
                        "company_name": e.filing.company_name or "",
                        "filing_date": e.filing.filing_date,
                        "source_8k_url": e.filing.index_html_url(),
                        "accession": e.filing.accession,
                        "form": e.filing.form,
                        "new_executive": e.person or "",
                        "position": e.matched_title or "",
                        "effective_date": e.effective_date or "",
                        "event_type": e.event_type or "",
                        "base_salary_usd": "" if base is None else base,
                        "target_bonus_pct": "" if comp.target_bonus_pct is None else comp.target_bonus_pct,
                        "target_bonus_usd": "" if bonus_usd is None else bonus_usd,
                        "equity_target_annual_usd_total": "" if equity_annual is None else equity_annual,
                        "target_total_comp_usd": "" if target_total is None else target_total,
                        "one_time_cash_usd_total": "" if comp.one_time_cash_usd_total is None else comp.one_time_cash_usd_total,
                        "one_time_cash_values": "; ".join(comp.one_time_cash_values),
                        "equity_one_time_usd_total": "" if comp.equity_one_time_usd_total is None else comp.equity_one_time_usd_total,
                        "equity_one_time_values": "; ".join(comp.equity_one_time_values),
                        "equity_one_time_labels": ", ".join(comp.equity_one_time_labels),
                        "equity_target_annual_values": "; ".join(comp.equity_target_annual_values),
                        "other_keywords": ", ".join(comp.other),
                        "compensation_summary": compensation_brief(comp),
                        "primary_doc_url": e.filing.primary_url(),
                        "confidence": e.confidence,
                    }
                )

    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        with out_md.open("a", encoding="utf-8") as f:
            for e in events:
                f.write("\n\n---\n\n")
                f.write(e.summary or "(no summary)\n")
                if e.compensation and e.compensation.evidence_snippets:
                    f.write("\n\nEvidence snippets (review filing for full context):\n")
                    for snip in e.compensation.evidence_snippets[:6]:
                        f.write(f"- {snip}\n")

# -----------------------------
# CLI
# -----------------------------

def _parse_positive_int(s: str) -> int:
    try:
        v = int(s)
    except Exception:
        raise argparse.ArgumentTypeError(f"Not an integer: {s}")
    if v <= 0:
        raise argparse.ArgumentTypeError("Value must be >= 1")
    return v

def _parse_iso_date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        raise argparse.ArgumentTypeError("Date must be in YYYY-MM-DD format")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Scan 8-K filings for executive appointments (Item 5.02) for a target position (e.g., CEO/CFO) and summarize compensation."
    )
    p.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT", ""), help="Declared User-Agent (e.g., 'YourOrg your.email@domain.com').")
    p.add_argument("--db", default="exec_8k_scanner.sqlite3", help="SQLite DB path for dedupe/results.")
    p.add_argument("--cache-dir", default=".cache_edgar", help="Cache directory for HTTP responses.")
    p.add_argument("--max-rps", type=float, default=8.0, help="Max requests per second (keep <= 10).")

    sub = p.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Scan a watchlist of tickers for executive appointment 8-Ks (Item 5.02) over a look-back window (months).")
    scan.add_argument("--tickers", nargs="*", default=[], help="Ticker symbols (e.g., AAPL MSFT). For large lists, prefer --tickers-file.")
    scan.add_argument("--tickers-file", default="", help="Path to a file with tickers (one per line, or comma/space separated).")
    scan.add_argument("--position", required=True, help="Position to detect (e.g., CEO, CFO, or 'Chief Financial Officer').")
    scan.add_argument("--lookback-months", type=_parse_positive_int, required=True, help="Look-back period in months (1-36). Includes the as-of date.")
    scan.add_argument("--as-of", type=_parse_iso_date, default=None, help="End date to look back from (YYYY-MM-DD). Defaults to today in America/Chicago.")
    scan.add_argument("--mode", choices=["submissions", "daily-index"], default="submissions",
                      help="Ingestion mode. 'submissions' is efficient for small watchlists; 'daily-index' scans the daily master index across dates.")
    scan.add_argument("--force", action="store_true", help="Re-scan filings even if already processed (refresh extraction).")
    scan.add_argument("--out-jsonl", default="events.jsonl")
    scan.add_argument("--out-csv", default="events.csv")
    scan.add_argument("--out-md", default="events.md")



    args = p.parse_args(argv)

    client = SecEdgarClient(
        user_agent=args.user_agent,
        max_req_per_sec=args.max_rps,
        cache_dir=Path(args.cache_dir),
    )
    store = Store(Path(args.db))

    tickers: List[str] = [t.upper() for t in (args.tickers or [])]
    if args.tickers_file:
        tickers.extend(load_tickers_from_file(Path(args.tickers_file)))
    tickers = [t for t in tickers if t.strip()]
    # Dedupe while preserving order
    seen_t = set()
    tickers = [t for t in tickers if not (t in seen_t or seen_t.add(t))]

    if not tickers:
        print("No tickers provided. Use --tickers and/or --tickers-file.", file=sys.stderr)
        return 2

    ticker_info = load_ticker_info(client)
    allowed_ciks: Set[str] = set()
    cik_to_ticker: Dict[str, str] = {}
    cik_to_title: Dict[str, str] = {}
    missing: List[str] = []
    for t in tickers:
        info = ticker_info.get(t)
        if not info:
            missing.append(t)
            continue
        allowed_ciks.add(info.cik)
        cik_to_ticker.setdefault(info.cik, info.ticker)
        if info.title:
            cik_to_title.setdefault(info.cik, info.title)

    if missing:
        print(f"Warning: {len(missing)} ticker(s) not found in SEC mapping (skipped): {', '.join(missing[:25])}{'...' if len(missing) > 25 else ''}", file=sys.stderr)

    if not allowed_ciks:
        print("None of the provided tickers mapped to a CIK. Nothing to do.", file=sys.stderr)
        return 2

    try:
        today = dt.datetime.now(ZoneInfo("America/Chicago")).date()
    except Exception:
        today = dt.date.today()
    as_of = args.as_of or today
    if as_of > today:
        print("--as-of cannot be in the future.", file=sys.stderr)
        return 2

    if args.lookback_months > 36:
        print("--lookback-months cannot exceed 36.", file=sys.stderr)
        return 2

    start = add_months(as_of, -int(args.lookback_months))

    all_events: List[ExecEvent] = []

    if args.mode == "daily-index":
        # Daily master index mode (scales better for huge watchlists, but is slower for long lookbacks)
        for day in daterange_inclusive(start, as_of):
            filings = filings_from_daily_master_index_filtered(
                client=client,
                day=day,
                allowed_ciks=allowed_ciks,
                cik_to_ticker=cik_to_ticker,
                cik_to_title=cik_to_title,
            )
            for filing in filings:
                try:
                    events = process_filing(
                        client=client,
                        store=store,
                        filing=filing,
                        position_query=args.position,
                        force=bool(args.force),
                    )
                    all_events.extend(events)
                except requests.HTTPError as e:
                    # Skip problematic filings but keep scanning
                    print(f"HTTP error processing {filing.accession}: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing {filing.accession}: {e}", file=sys.stderr)
    else:
        # Submissions API mode (recommended for small watchlists; fast even up to 36 months)
        filings_to_scan: List[FilingRef] = []
        for t in tickers:
            info = ticker_info.get(t)
            if not info:
                continue
            try:
                filings_to_scan.extend(
                    filings_from_submissions_range(
                        client=client,
                        cik10=info.cik,
                        start=start,
                        end=as_of,
                        ticker=info.ticker,
                        company_title_fallback=info.title,
                    )
                )
            except requests.HTTPError as e:
                print(f"HTTP error fetching submissions for {t}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error fetching submissions for {t}: {e}", file=sys.stderr)

        # Dedupe by accession
        seen_acc: Set[str] = set()
        unique_filings: List[FilingRef] = []
        for fr in filings_to_scan:
            if fr.accession in seen_acc:
                continue
            seen_acc.add(fr.accession)
            unique_filings.append(fr)

        # Sort newest-first
        unique_filings.sort(key=lambda x: x.filing_date, reverse=True)

        for filing in unique_filings:
            try:
                events = process_filing(
                    client=client,
                    store=store,
                    filing=filing,
                    position_query=args.position,
                    force=bool(args.force),
                )
                all_events.extend(events)
            except requests.HTTPError as e:
                print(f"HTTP error processing {filing.accession}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error processing {filing.accession}: {e}", file=sys.stderr)

    write_outputs(
        all_events,
        Path(args.out_jsonl) if args.out_jsonl else None,
        Path(args.out_csv) if args.out_csv else None,
        Path(args.out_md) if args.out_md else None,
    )

    print(f"Found {len(all_events)} event(s) for position '{args.position}'. Results stored in {args.db}.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
