from __future__ import annotations

import argparse
from pathlib import Path

from .selftest import selftest
from .workflow import build_graph, log


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
    state = build_graph(max_repairs=args.max_repairs).invoke({"ticker": args.ticker, "facts_only": args.facts_only})
    path = Path(args.out or f"memo_{args.ticker.upper()}.md")
    path.write_text(state["output"], encoding="utf-8")
    print(state["output"])
    log(f"Saved to {path}")
