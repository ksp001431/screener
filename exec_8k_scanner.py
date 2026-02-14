#!/usr/bin/env python3
"""
exec_8k_scanner.py

Scan SEC Form 8-K / 8-K/A filings for *executive appointment* events (Item 5.02)
for a user-specified POSITION (e.g., CEO or CFO) and summarize compensation terms.

Inputs:
  - A list of tickers (or a tickers file)
  - The position to detect (e.g., CEO, CFO, or a full title like "Chief Financial Officer")
  - A look-back period in days

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup


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
                dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
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
    html = client.get_text(filing.index_html_url())
    soup = BeautifulSoup(html, "lxml")

    docs: List[FilingDocument] = []
    table = soup.find("table", class_="tableFile")
    if not table:
        return docs

    rows = table.find_all("tr")
    for r in rows[1:]:
        cols = r.find_all("td")
        if len(cols) < 4:
            continue
        description = norm_ws(cols[1].get_text(" ", strip=True))
        document = norm_ws(cols[2].get_text(" ", strip=True))
        doc_type = norm_ws(cols[3].get_text(" ", strip=True))

        href_tag = cols[2].find("a")
        if not href_tag or not href_tag.get("href"):
            continue
        href = href_tag["href"]
        if href.startswith("/"):
            url = "https://www.sec.gov" + href
        elif href.startswith("http"):
            url = href
        else:
            url = filing.base_dir_url() + href

        docs.append(FilingDocument(document, description, doc_type, url))
    return docs

def pick_primary_doc(filing: FilingRef, docs: List[FilingDocument]) -> FilingRef:
    if filing.primary_doc:
        return filing

    for d in docs:
        if d.doc_type in ("8-K", "8-K/A"):
            return dataclasses.replace(filing, primary_doc=d.filename)

    for d in docs:
        if re.search(r"8k(\.htm|\.html|\.txt)$", d.filename, flags=re.I):
            return dataclasses.replace(filing, primary_doc=d.filename)

    for d in docs:
        if d.filename.lower().endswith((".htm", ".html")):
            return dataclasses.replace(filing, primary_doc=d.filename)

    # last resort: first doc
    if docs:
        return dataclasses.replace(filing, primary_doc=docs[0].filename)

    return filing

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
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

NAME_RE = r"(?:Mr\.|Ms\.|Mrs\.|Dr\.)?\s*(?:[A-Z][A-Za-z\.\-’']+)(?:\s+[A-Z][A-Za-z\.\-’']+){1,4}"

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
    # escape titles to treat them as literals inside regex
    title_alt = "|".join(re.escape(t) for t in position_titles if t.strip())
    # Allow a small suffix like "of the Company"
    title_re = rf"(?P<title>(?:{title_alt}))(?:\s+(?:of|for)\s+(?:the\s+)?Company)?"

    verbs_alt = "|".join(APPOINT_VERBS)

    patterns = [
        # "appointed John Doe as Chief Financial Officer"
        rf"\b(?P<lemma>{verbs_alt})\b\s+(?P<name>{NAME_RE})\s+(?:as|to serve as|to be)\s+(?:the\s+)?{title_re}",
        # "John Doe was appointed as Chief Financial Officer"
        rf"(?P<name>{NAME_RE})\s+(?:has been|was|is)\s+\b(?P<lemma>{verbs_alt})\b\s+(?:as|to serve as|to be)\s+(?:the\s+)?{title_re}",
        # "John Doe will serve as Chief Financial Officer"
        rf"(?P<name>{NAME_RE})\s+will\s+(?:serve|act)\s+as\s+(?:the\s+)?{title_re}",
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

DOLLAR_RE = r"[$€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?"
COMP_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("base_salary", re.compile(rf"\b(annual base salary|base salary)\b[^.:\n]{{0,160}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("base_salary", re.compile(rf"\b(annual salary|salary)\b[^.:\n]{{0,160}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("target_bonus", re.compile(r"\b(target (?:annual )?bonus|annual bonus target|target cash bonus|bonus opportunity)\b[^.:\n]{0,220}?(\d{1,3}\s?%)", re.IGNORECASE)),
    ("target_bonus", re.compile(rf"\b(target (?:annual )?bonus|annual bonus target|target cash bonus|bonus opportunity)\b[^.:\n]{{0,220}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("sign_on_bonus", re.compile(rf"\b(sign(?:ing)?-?on bonus|one-time (?:cash )?(?:signing|sign-on) bonus)\b[^.:\n]{{0,220}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("equity_award", re.compile(r"\b(restricted stock units|RSUs)\b[^.:\n]{0,260}?\b(\d{1,3}(?:,\d{3})*)\b\s+(?:shares|units)", re.IGNORECASE)),
    ("equity_award", re.compile(r"\b(stock option|options?)\b[^.:\n]{0,280}?\b(\d{1,3}(?:,\d{3})*)\b\s+(?:shares)", re.IGNORECASE)),
    ("equity_award", re.compile(rf"\b(restricted stock units|RSUs|stock options?|option award)\b[^.:\n]{{0,260}}?({DOLLAR_RE})", re.IGNORECASE)),
    ("severance", re.compile(r"\bseverance\b[^.:\n]{0,320}?\b(\d{1,2})\s+(?:months?|month)\b", re.IGNORECASE)),
    ("severance", re.compile(r"\bchange in control\b[^.:\n]{0,340}?\b(\d{1,2})\s+(?:months?|month)\b", re.IGNORECASE)),
    ("severance", re.compile(r"\bseverance\b[^.:\n]{0,340}?\b(\d(?:\.\d)?)\s?x\b", re.IGNORECASE)),
]

def extract_compensation(text: str) -> ExtractedComp:
    comp = ExtractedComp()

    for field, pat in COMP_PATTERNS:
        for m in pat.finditer(text):
            snippet = norm_ws(text[max(0, m.start()-180): m.end()+180])
            comp.evidence_snippets.append(snippet)

            if field == "base_salary" and comp.base_salary is None:
                comp.base_salary = money_norm(m.group(2))
            elif field == "target_bonus" and comp.target_bonus is None:
                comp.target_bonus = money_norm(m.group(2))
            elif field == "sign_on_bonus" and comp.sign_on_bonus is None:
                comp.sign_on_bonus = money_norm(m.group(2))
            elif field == "equity_award":
                comp.equity_awards.append(snippet[:320])
            elif field == "severance":
                comp.severance.append(snippet[:320])

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

    def dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    comp.equity_awards = dedupe(comp.equity_awards)
    comp.severance = dedupe(comp.severance)
    comp.other = dedupe(comp.other)
    comp.evidence_snippets = dedupe(comp.evidence_snippets)
    return comp

def compensation_brief(comp: ExtractedComp) -> str:
    bits = []
    if comp.base_salary:
        bits.append(f"Base salary {comp.base_salary}")
    if comp.target_bonus:
        bits.append(f"Target bonus {comp.target_bonus}")
    if comp.sign_on_bonus:
        bits.append(f"Sign-on {comp.sign_on_bonus}")
    if comp.equity_awards:
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
) -> List[ExecEvent]:
    if store.already_processed(filing.accession, position_query):
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
            "company_name",
            "ticker",
            "cik",
            "filing_date",
            "form",
            "accession",
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
            "compensation_brief",
            "primary_doc_url",
        ]
        exists = out_csv.exists()
        with out_csv.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                w.writeheader()
            for e in events:
                comp = e.compensation or ExtractedComp()
                w.writerow(
                    {
                        "company_name": e.filing.company_name or "",
                        "ticker": e.filing.ticker or "",
                        "cik": e.filing.cik,
                        "filing_date": e.filing.filing_date,
                        "form": e.filing.form,
                        "accession": e.filing.accession,
                        "new_executive": e.person or "",
                        "position": e.matched_title or "",
                        "effective_date": e.effective_date or "",
                        "event_type": e.event_type or "",
                        "base_salary": comp.base_salary or "",
                        "target_bonus": comp.target_bonus or "",
                        "sign_on_bonus": comp.sign_on_bonus or "",
                        "equity_mentions_count": str(len(comp.equity_awards)),
                        "severance_mentions_count": str(len(comp.severance)),
                        "other_keywords": ", ".join(comp.other),
                        "compensation_brief": compensation_brief(comp),
                        "primary_doc_url": e.filing.primary_url(),
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

    scan = sub.add_parser("scan", help="Scan a watchlist of tickers over a look-back period using daily master indexes (scales to large lists).")
    scan.add_argument("--tickers", nargs="*", default=[], help="Ticker symbols (e.g., AAPL MSFT). For large lists, prefer --tickers-file.")
    scan.add_argument("--tickers-file", default="", help="Path to a file with tickers (one per line, or comma/space separated).")
    scan.add_argument("--position", required=True, help="Position to detect (e.g., CEO, CFO, or 'Chief Financial Officer').")
    scan.add_argument("--lookback-days", type=_parse_positive_int, required=True, help="Look-back period in days (>= 1). Includes the as-of date.")
    scan.add_argument("--as-of", type=_parse_iso_date, default=None, help="End date to look back from (YYYY-MM-DD). Defaults to today in America/Chicago.")
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

    start = as_of - dt.timedelta(days=args.lookback_days - 1)

    all_events: List[ExecEvent] = []

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
                )
                all_events.extend(events)
            except requests.HTTPError as e:
                # Skip problematic filings but keep scanning
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
