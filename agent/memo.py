from __future__ import annotations

import re

from .config import CITE_RE, NUM_RE
from .models import Fact


def context_block(state: dict) -> str:
    facts = "\n".join(f"{f.id} | {f.label} | {f.display} | {f.kind}: {f.source}" for f in state["facts"])
    miss = ", ".join(state.get("missing", [])) or "none"
    ex = "\n\n".join(f"[{c['id']}] ({c['section']}) {c['text']}" for c in state["chunks"])
    fl = state["filing"]
    return (f"COMPANY: {state['company']} ({state['ticker']})\n"
            f"FILING: Form 10-K filed {fl['filingDate']} (accession {fl['accessionNumber']})\n\n"
            f"FACTS TABLE (ID | metric | value | source):\n{facts}\n\n"
            f"METRICS NOT AVAILABLE: {miss}\n\nFILING EXCERPTS:\n{ex}\n")

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
