from __future__ import annotations

from . import workflow
from .financials import build_facts
from .filing import extract_section, html_to_text, retrieve
from .memo import verify_memo
from .models import Fact


def selftest() -> None:
    real_http, real_llm = workflow.http_get, workflow.llm_call

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

    workflow.http_get, workflow.llm_call = fake_http, fake_llm
    try:
        out = workflow.build_graph().invoke({"ticker": "TEST", "facts_only": False})["output"]
        facts_only = workflow.build_graph().invoke({"ticker": "TEST", "facts_only": True})["output"]
    finally:
        workflow.http_get, workflow.llm_call = real_http, real_llm
    assert calls["n"] == 2 and "All automated checks passed" in out, out[-900:]
    print("ok: full graph with draft -> verify -> repair -> verify -> finalize (offline)")
    assert calls["n"] == 2 and "*Facts-only mode: no memo was drafted.*" in facts_only
    print("ok: facts-only workflow skips memo drafting")
    print("\nAll self-tests passed.")
