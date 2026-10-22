from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


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
