from __future__ import annotations

import os
import sys
import time
from typing import Any

from .config import DEFAULT_MODEL, USAGE


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
