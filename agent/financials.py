from __future__ import annotations

from datetime import date
from typing import Optional

from .config import DURATION_TAGS, INSTANT_TAGS, LABELS
from .models import Fact


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
