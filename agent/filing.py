from __future__ import annotations

import re

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

from .config import EXCERPT_CHARS, MAX_EXCERPTS, RETRIEVAL_PLAN


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
