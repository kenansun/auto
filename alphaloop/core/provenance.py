"""Data provenance and SHA-256 fingerprinting utilities (anti-mock enforcement)."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

PROVENANCE_REAL = "REAL_EXCHANGE_API"
PROVENANCE_PAPER = "PAPER_LIVE_FEED"
PROVENANCE_SYNTHETIC = "SYNTHETIC_TEST_ONLY"


def canonical_json(obj: Any) -> str:
    """Deterministic compact JSON serialization for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_obj(obj: Any) -> str:
    """Return SHA-256 hex digest of any JSON-serializable object."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    """Return SHA-256 hex digest of a file streamed in 64KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def utc_now_ms() -> int:
    """Current wall-clock UTC timestamp in milliseconds."""
    return int(time.time() * 1000)


def utc_iso(ts_ms: int | None = None) -> str:
    """Format UTC millisecond timestamp as ISO-8601 Z string."""
    sec = (ts_ms if ts_ms is not None else utc_now_ms()) / 1000.0
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(sec))
