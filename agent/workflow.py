from __future__ import annotations

import os
import sys
from datetime import date

from langgraph.graph import END, StateGraph

from .config import DISCLAIMER, MAX_REPAIRS, MEMO_TEMPLATE, SYSTEM_PROMPT, USAGE
from .filing import extract_section, html_to_text, retrieve
from .financials import build_facts
from .llm import llm_call
from .memo import context_block, verify_memo
from .models import State
from .sec_client import http_get


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

def build_graph(max_repairs: int = MAX_REPAIRS):
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
        lambda s: "repair" if s["issues"] and s.get("attempts", 0) < max_repairs else "finalize",
        {"repair": "repair", "finalize": "finalize"})
    g.add_edge("repair", "verify")
    g.add_edge("finalize", END)
    return g.compile()
