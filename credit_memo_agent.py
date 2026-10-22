
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional, TypedDict

import requests
from bs4 import BeautifulSoup
from langgraph.graph import END, StateGraph
from rank_bm25 import BM25Okapi

# Check Google's model page for the current ID; override with the GEMINI_MODEL env var.
DEFAULT_MODEL = "gemini-3.1-flash-lite"
MAX_REPAIRS = 1
MAX_EXCERPTS = 14
EXCERPT_CHARS = 1400
DISCLAIMER = (
    "*Automated, preliminary draft built only from the company's latest Form 10-K. "
    "It is not investment advice or a credit rating. Verify every figure against the filing before use.*"
)
USAGE = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}


# --------------------------------------------------------------------------- #
# HTTP (SEC EDGAR) with polite rate limiting and a 24h on-disk cache
# --------------------------------------------------------------------------- #
def http_get(url: str, as_json: bool = False) -> Any:
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        raise SystemExit("Set SEC_USER_AGENT to e.g. 'Your Name you@example.com' (SEC requires a contact).")
    cache_dir = Path(os.getenv("CMA_CACHE_DIR", ".sec_cache"))
    cache_dir.mkdir(exist_ok=True)
    path = cache_dir / hashlib.sha256(url.encode()).hexdigest()[:32]
    if path.exists() and time.time() - path.stat().st_mtime < 86400:
        body = path.read_text(encoding="utf-8")
    else:
        body = None
        for attempt in range(4):
            time.sleep(0.2)  # stay well below SEC's 10 requests/second limit
            r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=60)
            if r.status_code == 200:
                body = r.content.decode("utf-8", errors="replace")
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        if body is None:
            raise RuntimeError(f"Failed to fetch {url}")
        path.write_text(body, encoding="utf-8")
    return json.loads(body) if as_json else body


# --------------------------------------------------------------------------- #
# Financial facts (XBRL "companyfacts") and ratios, all computed in Python
# --------------------------------------------------------------------------- #
@dataclass
class Fact:
    id: str
    label: str
    value: float
    display: str
    kind: str  # "reported" or "computed"
    source: str


def fmt_money(v: float) -> str:
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e9:
        return f"{sign}${a / 1e9:.1f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.1f}M"
    return f"{sign}${a:,.0f}"


def fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def fmt_x(v: float) -> str:
    return f"{v:.2f}x"


DURATION_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "op_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense"],
    "d_and_a": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                "DepreciationAmortizationAndAccretionNet"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}
INSTANT_TAGS = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
}
LABELS = {
    "revenue": "Revenue", "gross_profit": "Gross profit", "op_income": "Operating income",
    "net_income": "Net income", "interest_expense": "Interest expense", "d_and_a": "Depreciation and amortization",
    "cfo": "Operating cash flow", "capex": "Capital expenditures", "total_assets": "Total assets",
    "total_liabilities": "Total liabilities", "equity": "Shareholders' equity",
    "current_assets": "Current assets", "current_liabilities": "Current liabilities", "cash": "Cash and equivalents",
}


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(a) - date.fromisoformat(b)).days


def _series(gaap: dict, tag: str, instant: bool) -> dict:
    """Annual (10-K, FY) USD values for a tag, keyed by period end; the latest filing wins."""
    out: dict[str, dict] = {}
    for unit, rows in gaap.get(tag, {}).get("units", {}).items():
        if unit != "USD":
            continue
        for r in rows:
            if r.get("form") not in ("10-K", "10-K/A") or r.get("fp") != "FY":
                continue
            if not instant:
                if "start" not in r or not 330 <= _days(r["end"], r["start"]) <= 400:
                    continue
            prev = out.get(r["end"])
            if prev is None or r["filed"] > prev["filed"]:
                out[r["end"]] = r
    return out


def build_facts(companyfacts: dict) -> tuple[list[Fact], list[str], str]:
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    assets = _series(gaap, "Assets", True)
    if not assets:
        raise SystemExit("No US-GAAP annual (10-K) XBRL data found. Only US 10-K filers are supported.")
    ends = sorted(assets)
    cur = ends[-1]
    prior = next((e for e in reversed(ends[:-1]) if 330 <= _days(cur, e) <= 400), None)

    facts: list[Fact] = []
    K: dict[str, Fact] = {}
    missing: list[str] = []

    def add(key: str, label: str, value: float, kind: str, source: str, fmt) -> Fact:
        f = Fact(f"F{len(facts) + 1}", label, value, fmt(value), kind, source)
        facts.append(f)
        K[key] = f
        return f

    def pick(tags: list[str], instant: bool, end: str) -> Optional[tuple[float, str]]:
        for t in tags:
            row = _series(gaap, t, instant).get(end)
            if row is not None:
                return float(row["val"]), t
        return None

    for key, tags in {**DURATION_TAGS, **INSTANT_TAGS}.items():
        instant = key in INSTANT_TAGS
        got = pick(tags, instant, cur)
        if got:
            add(key, f"{LABELS[key]}, FYE {cur}", got[0], "reported", f"us-gaap:{got[1]}", fmt_money)
        else:
            missing.append(LABELS[key])
    if prior:
        for key in ("revenue", "net_income"):
            got = pick(DURATION_TAGS[key], False, prior)
            if got:
                add(f"{key}_prior", f"{LABELS[key]}, FYE {prior} (prior year)", got[0], "reported",
                    f"us-gaap:{got[1]}", fmt_money)

    # Debt: sum of the standard components that exist (approximation; excludes leases).
    parts = []
    for tags in (["LongTermDebtNoncurrent"], ["LongTermDebtCurrent", "DebtCurrent"], ["ShortTermBorrowings", "CommercialPaper"]):
        got = pick(tags, True, cur)
        if got:
            parts.append(got)
    if not parts:
        got = pick(["LongTermDebt"], True, cur)
        if got:
            parts.append(got)
    if parts:
        add("total_debt", f"Total debt (approx.), FYE {cur}", sum(p[0] for p in parts), "computed",
            "sum of " + ", ".join(f"us-gaap:{p[1]}" for p in parts), fmt_money)
    else:
        missing.append("Total debt")

    def has(*keys: str) -> bool:
        return all(k in K for k in keys)

    def ids(*keys: str) -> str:
        return ", ".join(K[k].id for k in keys)

    if has("revenue", "revenue_prior") and K["revenue_prior"].value:
        add("rev_growth", "Revenue growth, year over year", K["revenue"].value / K["revenue_prior"].value - 1,
            "computed", f"{ids('revenue')} / {ids('revenue_prior')} - 1", fmt_pct)
    for key, num, label in (("gross_margin", "gross_profit", "Gross margin"), ("op_margin", "op_income", "Operating margin"),
                            ("net_margin", "net_income", "Net margin")):
        if has(num, "revenue") and K["revenue"].value:
            add(key, label, K[num].value / K["revenue"].value, "computed", f"{ids(num)} / {ids('revenue')}", fmt_pct)
    if has("op_income", "d_and_a"):
        add("ebitda", "EBITDA (operating income + D&A)", K["op_income"].value + K["d_and_a"].value, "computed",
            f"{ids('op_income')} + {ids('d_and_a')}", fmt_money)
    if has("op_income", "interest_expense") and K["interest_expense"].value > 0:
        add("interest_cov", "Interest coverage (operating income / interest expense)",
            K["op_income"].value / K["interest_expense"].value, "computed",
            f"{ids('op_income')} / {ids('interest_expense')}", fmt_x)
    else:
        missing.append("Interest coverage")
    if has("cfo", "capex"):
        add("fcf", "Free cash flow (operating cash flow - capex)", K["cfo"].value - K["capex"].value, "computed",
            f"{ids('cfo')} - {ids('capex')}", fmt_money)
        if has("revenue") and K["revenue"].value:
            add("fcf_margin", "Free cash flow margin", K["fcf"].value / K["revenue"].value, "computed",
                f"{ids('fcf')} / {ids('revenue')}", fmt_pct)
    if has("total_debt", "cash"):
        add("net_debt", "Net debt (total debt - cash)", K["total_debt"].value - K["cash"].value, "computed",
            f"{ids('total_debt')} - {ids('cash')}", fmt_money)
    if has("total_debt", "equity") and K["equity"].value > 0:
        add("debt_equity", "Debt to equity", K["total_debt"].value / K["equity"].value, "computed",
            f"{ids('total_debt')} / {ids('equity')}", fmt_x)
    if has("total_debt", "ebitda") and K["ebitda"].value > 0:
        add("debt_ebitda", "Total debt to EBITDA", K["total_debt"].value / K["ebitda"].value, "computed",
            f"{ids('total_debt')} / {ids('ebitda')}", fmt_x)
    if has("current_assets", "current_liabilities") and K["current_liabilities"].value:
        add("current_ratio", "Current ratio", K["current_assets"].value / K["current_liabilities"].value,
            "computed", f"{ids('current_assets')} / {ids('current_liabilities')}", fmt_x)
    if has("total_liabilities", "total_assets") and K["total_assets"].value:
        add("liab_assets", "Liabilities to assets", K["total_liabilities"].value / K["total_assets"].value,
            "computed", f"{ids('total_liabilities')} / {ids('total_assets')}", fmt_pct)
    return facts, missing, cur


# --------------------------------------------------------------------------- #
# Filing text: download, extract sections, chunk, retrieve with BM25
# --------------------------------------------------------------------------- #
def html_to_text(html: str) -> str:
    try:
        import lxml  # noqa: F401
        parser = "lxml"
    except ImportError:
        parser = "html.parser"
    soup = BeautifulSoup(html, parser)
    for t in soup(["script", "style"]):
        t.decompose()
    for t in soup.find_all("ix:header"):  # hidden inline-XBRL metadata block
        t.decompose()
    text = soup.get_text("\n").replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def extract_section(text: str, start_re: str, end_re: str, min_len: int = 4000, cap: int = 250_000) -> str:
    """Longest span from a heading match to the next end-heading (skips the table of contents)."""
    best = ""
    for m in re.finditer(start_re, text, re.I | re.M):
        e = re.search(end_re, text[m.end():], re.I | re.M)
        seg = text[m.start(): m.end() + e.start()] if e else text[m.start(): m.start() + cap]
        if len(seg) > len(best):
            best = seg[:cap]
    return best if len(best) >= min_len else ""


def chunk_text(text: str, size: int = 1200) -> list[str]:
    pieces: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) <= size:
            pieces.append(line)
        else:  # split very long paragraphs at sentence boundaries
            cur = ""
            for s in re.split(r"(?<=[.!?])\s+", line):
                if len(cur) + len(s) > size and cur:
                    pieces.append(cur)
                    cur = ""
                cur = (cur + " " + s).strip()
            if cur:
                pieces.append(cur)
    chunks, cur = [], ""
    for p in pieces:
        if not p:
            continue
        if len(cur) + len(p) + 1 > size and cur:
            chunks.append(cur)
            cur = ""
        cur = (cur + " " + p).strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) > 200]


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


RETRIEVAL_PLAN = [  # (section, query, top_k)
    ("Business", "company business overview segments products services customers markets", 2),
    ("Risk Factors", "competition competitors pricing pressure market share", 1),
    ("Risk Factors", "indebtedness debt covenants refinancing credit facility liquidity", 2),
    ("Risk Factors", "legal proceedings litigation regulation compliance investigations fines", 1),
    ("Risk Factors", "supplier concentration customer dependence supply chain disruption", 1),
    ("Risk Factors", "macroeconomic conditions inflation interest rates currency exchange", 1),
    ("MD&A", "liquidity and capital resources cash flows debt repayment dividends repurchases", 2),
    ("MD&A", "results of operations revenue increased decreased primarily due to", 2),
    ("MD&A", "going concern substantial doubt impairment restructuring material weakness", 1),
]


def retrieve(sections: dict[str, str]) -> list[dict]:
    pool = {name: chunk_text(txt) for name, txt in sections.items() if txt}
    picked: list[tuple[str, int]] = []
    for sec, query, k in RETRIEVAL_PLAN:
        chunks = pool.get(sec) or pool.get("Full filing")
        if not chunks:
            continue
        key = sec if sec in pool else "Full filing"
        scores = BM25Okapi([tokenize(c) for c in chunks]).get_scores(tokenize(query))
        for i in sorted(range(len(chunks)), key=lambda j: -scores[j])[:k]:
            if (key, i) not in picked:
                picked.append((key, i))
    order = {n: i for i, n in enumerate(["Business", "Risk Factors", "MD&A", "Full filing"])}
    picked = sorted(picked, key=lambda p: (order[p[0]], p[1]))[:MAX_EXCERPTS]
    return [{"id": f"S{n}", "section": sec, "text": pool[sec][i][:EXCERPT_CHARS]}
            for n, (sec, i) in enumerate(picked, 1)]


# --------------------------------------------------------------------------- #
# LLM (Gemini through LangChain)
# --------------------------------------------------------------------------- #
def get_llm():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY (create one in Google AI Studio).")
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {"model": os.getenv("GEMINI_MODEL", DEFAULT_MODEL), "google_api_key": key,
                              "max_output_tokens": 4096}
    if os.getenv("GEMINI_TEMPERATURE"):  # left unset by default: Gemini 3 models prefer their default
        kwargs["temperature"] = float(os.environ["GEMINI_TEMPERATURE"])
    return ChatGoogleGenerativeAI(**kwargs)


def text_of(msg: Any) -> str:
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    return "".join(p if isinstance(p, str) else p.get("text", "") for p in c
                   if isinstance(p, str) or (isinstance(p, dict) and p.get("type") == "text"))


def llm_call(messages: list) -> str:
    llm = get_llm()
    for attempt in range(4):  # free-tier rate limits (429) are common, so back off and retry
        try:
            resp = llm.invoke(messages)
            um = getattr(resp, "usage_metadata", None) or {}
            USAGE["llm_calls"] += 1
            USAGE["input_tokens"] += int(um.get("input_tokens", 0))
            USAGE["output_tokens"] += int(um.get("output_tokens", 0))
            return text_of(resp).strip()
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                raise
            wait = 5 * 2**attempt
            print(f"  LLM error ({type(exc).__name__}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


SYSTEM_PROMPT = """You are a credit analyst assistant. You draft a preliminary credit memo using ONLY the data provided.

Rules:
1. Every sentence that states a fact must end with one or more citations like [F3] or [S2]. Use [F#] for the facts table and [S#] for filing excerpts. Never cite an ID that is not provided.
2. Copy numbers EXACTLY as written in the facts table (for example $391.0B, 44.1%, 1.07x). Do not round, convert, or recompute anything. Do not do arithmetic.
3. Do not use outside knowledge. If something is not in the data, write 'Not available in the provided data.'
4. Write about the company's credit profile: ability to service debt, liquidity, earnings quality, and key risks. Do not give an investment recommendation or a formal rating.
5. Be concise and specific. No filler. About 450 to 650 words."""

MEMO_TEMPLATE = """Write the memo in Markdown with exactly these sections:
## 1. Business overview
## 2. Financial summary
## 3. Credit strengths
## 4. Key risks
## 5. Red flags and watch items
## 6. Preliminary credit view
Section 6 gives a qualitative view (Stronger / Adequate / Weaker) in 2 to 3 cited sentences, labelled preliminary."""


def context_block(state: dict) -> str:
    facts = "\n".join(f"{f.id} | {f.label} | {f.display} | {f.kind}: {f.source}" for f in state["facts"])
    miss = ", ".join(state.get("missing", [])) or "none"
    ex = "\n\n".join(f"[{c['id']}] ({c['section']}) {c['text']}" for c in state["chunks"])
    fl = state["filing"]
    return (f"COMPANY: {state['company']} ({state['ticker']})\n"
            f"FILING: Form 10-K filed {fl['filingDate']} (accession {fl['accessionNumber']})\n\n"
            f"FACTS TABLE (ID | metric | value | source):\n{facts}\n\n"
            f"METRICS NOT AVAILABLE: {miss}\n\nFILING EXCERPTS:\n{ex}\n")


# --------------------------------------------------------------------------- #
# Verifier: citations must exist, and numbers must come from the cited sources
# --------------------------------------------------------------------------- #
CITE_RE = re.compile(r"\[([FS]\d+(?:\s*[,;]\s*[FS]\d+)*)\]")
NUM_RE = re.compile(r"(?<![\w.])(-?\$?-?\d[\d,]*(?:\.\d+)?)(\s?(?:%|x\b|billion\b|million\b|thousand\b|[BMK]\b))?", re.I)


def numeric_tokens(s: str) -> list[str]:
    """Numbers carrying a unit ($, %, x, B/M/K, billion/million). Bare integers and years are ignored."""
    return [m.group(0) for m in NUM_RE.finditer(s) if "$" in m.group(0) or m.group(2)]


def norm_num(tok: str) -> str:
    t = tok.lower().replace(" ", "").replace(",", "")
    t = t.replace("billion", "b").replace("million", "m").replace("thousand", "k")
    neg = "-" in t
    return ("-" if neg else "") + t.replace("-", "").replace("$", "")


def verify_memo(memo: str, facts: list[Fact], chunks: list[dict]) -> list[str]:
    fmap = {f.id: f for f in facts}
    cmap = {c["id"]: c for c in chunks}
    issues: list[str] = []
    for raw in memo.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]\s+|\d+\.\s+)", "", raw.strip())
        if not line or line.startswith("#"):
            continue
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            cited = [i.strip() for m in CITE_RE.finditer(sent) for i in re.split(r"[,;]", m.group(1))]
            bad = [i for i in cited if i not in fmap and i not in cmap]
            if bad:
                issues.append(f"Unknown citation(s) {bad} in: {sent[:100]}")
            body = CITE_RE.sub("", sent)
            if len(body.split()) >= 6 and not cited:
                issues.append(f"Uncited sentence: {sent[:100]}")
                continue
            allowed = {norm_num(fmap[i].display) for i in cited if i in fmap}
            for i in cited:
                if i in cmap:
                    allowed |= {norm_num(t) for t in numeric_tokens(cmap[i]["text"])}
            for tok in numeric_tokens(body):
                if norm_num(tok) not in allowed:
                    issues.append(f"Number '{tok.strip()}' is not in the cited sources: {sent[:100]}")
    return issues


# --------------------------------------------------------------------------- #
# LangGraph nodes
# --------------------------------------------------------------------------- #
class State(TypedDict, total=False):
    ticker: str
    facts_only: bool
    company: str
    cik: int
    filing: dict
    facts: list[Fact]
    missing: list[str]
    fy_end: str
    chunks: list[dict]
    memo: str
    issues: list[str]
    attempts: int
    output: str


def log(msg: str) -> None:
    print(f"[agent] {msg}", file=sys.stderr)


def n_resolve(s: State) -> dict:
    log("Resolving ticker and finding the latest 10-K")
    want = s["ticker"].upper().replace(".", "-")
    rows = http_get("https://www.sec.gov/files/company_tickers.json", as_json=True)
    hit = next((r for r in rows.values() if r["ticker"].upper() == want), None)
    if not hit:
        raise SystemExit(f"Ticker {s['ticker']} not found in SEC's company list (US-listed companies only).")
    cik = int(hit["cik_str"])
    sub = http_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", as_json=True)
    rec = sub["filings"]["recent"]
    for i, form in enumerate(rec["form"]):
        if form == "10-K":
            acc = rec["accessionNumber"][i]
            doc = rec["primaryDocument"][i]
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
            filing = {"accessionNumber": acc, "primaryDocument": doc, "filingDate": rec["filingDate"][i], "url": url}
            return {"company": hit["title"], "cik": cik, "filing": filing}
    raise SystemExit(f"No recent 10-K found for {s['ticker']}. Foreign filers (20-F/40-F) are not supported.")


def n_financials(s: State) -> dict:
    log("Fetching XBRL financial data and computing ratios")
    cf = http_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{s['cik']:010d}.json", as_json=True)
    facts, missing, fy_end = build_facts(cf)
    return {"facts": facts, "missing": missing, "fy_end": fy_end}


def n_filing_text(s: State) -> dict:
    log("Downloading and parsing the 10-K text")
    text = html_to_text(http_get(s["filing"]["url"]))
    sections = {
        "Business": extract_section(text, r"^\s*item\s*1\.?\s*business", r"^\s*item\s*1a\b"),
        "Risk Factors": extract_section(text, r"^\s*item\s*1a\.?\s*risk\s*factors", r"^\s*item\s*(?:1b|1c|2)\b"),
        "MD&A": extract_section(text, r"^\s*item\s*7\.?\s*management", r"^\s*item\s*(?:7a|8)\b"),
    }
    if not any(sections.values()):
        log("Could not locate Items 1/1A/7; falling back to the start of the filing")
        sections = {"Full filing": text[:300_000]}
    return {"chunks": retrieve(sections)}


def n_draft(s: State) -> dict:
    log("Drafting the memo with Gemini")
    from langchain_core.messages import HumanMessage, SystemMessage

    memo = llm_call([SystemMessage(content=SYSTEM_PROMPT),
                     HumanMessage(content=context_block(s) + "\n" + MEMO_TEMPLATE)])
    return {"memo": memo, "attempts": 0}


def n_verify(s: State) -> dict:
    issues = verify_memo(s["memo"], s["facts"], s["chunks"])
    log(f"Verification found {len(issues)} issue(s)")
    return {"issues": issues}


def n_repair(s: State) -> dict:
    log("Repairing the memo")
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    problems = "\n".join(f"- {i}" for i in s["issues"][:12])
    memo = llm_call([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context_block(s) + "\n" + MEMO_TEMPLATE),
        AIMessage(content=s["memo"]),
        HumanMessage(content="Automated checks found these problems:\n" + problems +
                     "\nRewrite the full memo. Cite every factual sentence, copy numbers exactly from the cited "
                     "fact or excerpt, and delete any claim you cannot support."),
    ])
    return {"memo": memo, "attempts": s.get("attempts", 0) + 1}


def n_finalize(s: State) -> dict:
    fl = s["filing"]
    out = [f"# Preliminary Credit Memo: {s['company']} ({s['ticker'].upper()})",
           f"*Source: Form 10-K for the fiscal year ended {s.get('fy_end', '?')}, filed {fl['filingDate']}. "
           f"Generated {date.today().isoformat()}.*", ""]
    if s.get("memo"):
        out += [s["memo"], "", "---", "## Verification report"]
        issues = s.get("issues", [])
        out.append("All automated checks passed (citations exist; every unit-bearing number appears in its cited source)."
                   if not issues else
                   f"{len(issues)} unresolved issue(s). Treat these statements with extra caution:\n"
                   + "\n".join(f"- {i}" for i in issues))
    else:
        out += ["*Facts-only mode: no memo was drafted.*"]
    out += ["", "## Financial data (computed from SEC XBRL)", "| ID | Metric | Value | Source |", "|---|---|---|---|"]
    out += [f"| {f.id} | {f.label} | {f.display} | {f.kind}: {f.source} |" for f in s["facts"]]
    if s.get("missing"):
        out.append(f"\nNot available from standard tags: {', '.join(s['missing'])}.")
    out += ["", "## Source excerpts", f"Filing: {fl['url']}", ""]
    out += [f"**[{c['id']}] {c['section']}:** {c['text'][:500]}{'...' if len(c['text']) > 500 else ''}\n" for c in s["chunks"]]
    if USAGE["llm_calls"]:
        line = f"LLM usage: {USAGE['llm_calls']} call(s), {USAGE['input_tokens']} input / {USAGE['output_tokens']} output tokens"
        pin, pout = os.getenv("GEMINI_INPUT_PRICE_PER_M"), os.getenv("GEMINI_OUTPUT_PRICE_PER_M")
        if pin and pout:
            cost = USAGE["input_tokens"] / 1e6 * float(pin) + USAGE["output_tokens"] / 1e6 * float(pout)
            line += f", estimated ${cost:.4f}"
        out.append(f"\n*{line}.*")
    out += ["", DISCLAIMER]
    return {"output": "\n".join(out)}


def build_graph():
    g = StateGraph(State)
    for name, fn in [("resolve", n_resolve), ("financials", n_financials), ("filing_text", n_filing_text),
                     ("draft", n_draft), ("verify", n_verify), ("repair", n_repair), ("finalize", n_finalize)]:
        g.add_node(name, fn)
    g.set_entry_point("resolve")
    g.add_edge("resolve", "financials")
    g.add_edge("financials", "filing_text")
    g.add_conditional_edges("filing_text", lambda s: "finalize" if s.get("facts_only") else "draft",
                            {"draft": "draft", "finalize": "finalize"})
    g.add_edge("draft", "verify")
    g.add_conditional_edges(
        "verify",
        lambda s: "repair" if s["issues"] and s.get("attempts", 0) < MAX_REPAIRS else "finalize",
        {"repair": "repair", "finalize": "finalize"})
    g.add_edge("repair", "verify")
    g.add_edge("finalize", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# Offline self-test: fake SEC data + fake LLM (no network, no API key needed)
# --------------------------------------------------------------------------- #
def selftest() -> None:
    global http_get, llm_call
    real_http, real_llm = http_get, llm_call

    def row(val, end, start=None, filed="2025-11-01"):
        r = {"val": val, "end": end, "fy": 2025, "fp": "FY", "form": "10-K", "filed": filed, "accn": "0001-25-000001"}
        if start:
            r["start"] = start
        return r

    def tag(*rows):
        return {"units": {"USD": list(rows)}}

    cur_e, pri_e, cur_s, pri_s = "2025-09-27", "2024-09-28", "2024-09-29", "2023-10-01"
    cf = {"facts": {"us-gaap": {
        "Revenues": tag(row(100e9, cur_e, cur_s), row(90e9, pri_e, pri_s)),
        "GrossProfit": tag(row(40e9, cur_e, cur_s)),
        "OperatingIncomeLoss": tag(row(30e9, cur_e, cur_s)),
        "NetIncomeLoss": tag(row(24e9, cur_e, cur_s), row(20e9, pri_e, pri_s)),
        "InterestExpense": tag(row(2e9, cur_e, cur_s)),
        "DepreciationDepletionAndAmortization": tag(row(5e9, cur_e, cur_s)),
        "NetCashProvidedByUsedInOperatingActivities": tag(row(35e9, cur_e, cur_s)),
        "PaymentsToAcquirePropertyPlantAndEquipment": tag(row(10e9, cur_e, cur_s)),
        "Assets": tag(row(300e9, cur_e), row(280e9, pri_e)),
        "Liabilities": tag(row(200e9, cur_e)),
        "StockholdersEquity": tag(row(100e9, cur_e)),
        "AssetsCurrent": tag(row(120e9, cur_e)),
        "LiabilitiesCurrent": tag(row(80e9, cur_e)),
        "CashAndCashEquivalentsAtCarryingValue": tag(row(30e9, cur_e)),
        "LongTermDebtNoncurrent": tag(row(50e9, cur_e)),
        "LongTermDebtCurrent": tag(row(10e9, cur_e)),
    }}}
    facts, missing, fy = build_facts(cf)
    by: dict[str, Fact] = {}
    for f in facts:  # first occurrence wins (current year before prior year)
        by.setdefault(f.label.split(",")[0], f)
    assert fy == cur_e
    assert by["Revenue"].display == "$100.0B"
    assert by["Revenue growth"].display == "11.1%", by["Revenue growth"].display
    assert by["Operating margin"].display == "30.0%"
    assert by["EBITDA (operating income + D&A)"].display == "$35.0B"
    assert by["Interest coverage (operating income / interest expense)"].display == "15.00x"
    assert by["Free cash flow (operating cash flow - capex)"].display == "$25.0B"
    assert by["Net debt (total debt - cash)"].display == "$30.0B"
    assert by["Current ratio"].display == "1.50x"
    print("ok: facts and ratios")

    def sec(n: int) -> str:
        return " ".join(f"Sentence {i} about the company's operations and its markets, customers and risks." for i in range(n))

    html = ("<html><body><div>Table of contents</div><div>Item 1. Business</div><div>Item 1A. Risk Factors</div>"
            "<div>Item 7. Management's Discussion</div><div>Item 8. Financial Statements</div>"
            f"<div>Item 1. Business</div><p>{sec(80)}</p><p>Net sales were $100.0 billion.</p>"
            f"<div>Item 1A. Risk Factors</div><p>{sec(60)} Indebtedness could hurt liquidity and refinancing.</p>"
            f"<div>Item 1B. Unresolved Staff Comments</div><div>Item 7. Management's Discussion and Analysis</div>"
            f"<p>{sec(70)} Liquidity and capital resources remain strong.</p><div>Item 7A. Quantitative</div></body></html>")
    secs = {
        "Business": extract_section(html_to_text(html), r"^\s*item\s*1\.?\s*business", r"^\s*item\s*1a\b"),
        "Risk Factors": extract_section(html_to_text(html), r"^\s*item\s*1a\.?\s*risk\s*factors", r"^\s*item\s*(?:1b|1c|2)\b"),
        "MD&A": extract_section(html_to_text(html), r"^\s*item\s*7\.?\s*management", r"^\s*item\s*(?:7a|8)\b"),
    }
    assert all(len(v) > 4000 for v in secs.values()), {k: len(v) for k, v in secs.items()}
    chunks = retrieve(secs)
    assert chunks and chunks[0]["id"] == "S1"
    print(f"ok: section extraction and BM25 retrieval ({len(chunks)} excerpts)")

    rev = by["Revenue"].id
    good = f"Revenue was $100.0B in the latest fiscal year [{rev}]. The company has solid liquidity and earnings power [{rev}]."
    assert verify_memo(good, facts, chunks) == []
    bad = (f"Revenue was $110.0B in the latest fiscal year [{rev}]. The company faces many competitive pressures "
           f"across all of its markets. See [F999].")
    kinds = " | ".join(verify_memo(bad, facts, chunks))
    assert "not in the cited sources" in kinds and "Uncited" in kinds and "Unknown citation" in kinds, kinds
    print("ok: verifier catches wrong numbers, uncited sentences and unknown citations")

    calls = {"n": 0}

    def fake_http(url: str, as_json: bool = False):
        if url.endswith("company_tickers.json"):
            return {"0": {"cik_str": 123, "ticker": "TEST", "title": "Test Corp"}}
        if "submissions" in url:
            return {"filings": {"recent": {"form": ["8-K", "10-K"], "accessionNumber": ["a", "0001-25-000001"],
                                           "primaryDocument": ["x.htm", "test10k.htm"],
                                           "filingDate": ["2025-12-01", "2025-11-01"]}}}
        if "companyfacts" in url:
            return cf
        return html

    def fake_llm(messages: list) -> str:
        calls["n"] += 1
        if calls["n"] == 1:  # first draft contains a wrong number and an uncited sentence
            return (f"## 1. Business overview\nThe company has broad operations across many different markets and customers.\n"
                    f"## 2. Financial summary\nRevenue was $120.0B [{rev}].")
        return f"## 2. Financial summary\nRevenue was $100.0B [{rev}]."

    http_get, llm_call = fake_http, fake_llm
    try:
        out = build_graph().invoke({"ticker": "TEST", "facts_only": False})["output"]
    finally:
        http_get, llm_call = real_http, real_llm
    assert calls["n"] == 2 and "All automated checks passed" in out, out[-900:]
    print("ok: full graph with draft -> verify -> repair -> verify -> finalize (offline)")
    print("\nAll self-tests passed.")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a cited preliminary credit memo from SEC 10-K data.")
    ap.add_argument("ticker", nargs="?", help="US stock ticker, e.g. AAPL")
    ap.add_argument("--out", help="Output .md path (default: memo_<TICKER>.md)")
    ap.add_argument("--facts-only", action="store_true", help="Skip the LLM; output data and excerpts only")
    ap.add_argument("--selftest", action="store_true", help="Run offline tests and exit")
    ap.add_argument("--max-repairs", type=int, default=1, help="Repair attempts if verification fails")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.ticker:
        ap.error("ticker is required (or use --selftest)")
    global MAX_REPAIRS
    MAX_REPAIRS = args.max_repairs
    state = build_graph().invoke({"ticker": args.ticker, "facts_only": args.facts_only})
    path = Path(args.out or f"memo_{args.ticker.upper()}.md")
    path.write_text(state["output"], encoding="utf-8")
    print(state["output"])
    log(f"Saved to {path}")


if __name__ == "__main__":
    main()
