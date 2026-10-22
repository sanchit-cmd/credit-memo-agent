from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


@dataclass
class Fact:
    id: str
    label: str
    value: float
    display: str
    kind: str  # "reported" or "computed"
    source: str

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
