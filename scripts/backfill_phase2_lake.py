#!/usr/bin/env python3
"""Expand real multi-exchange SQLite market data lake to 180d-365d (Phase 2).

Fetches strictly REAL exchange API data (zero synthetic fallback):
  - 25-symbol universe: PAXG (Gold Perp) + 24 liquid crypto perps.
  - Hyperliquid: up to 210d 1h funding + ~208d 1h candles.
  - Lighter: up to 190d 1h funding.
  - Gate.io: up to 179d funding + 365d 1h candles (core majors + PAXG) / 185d (alts).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.data.collectors import GateCollector, HyperliquidCollector, LighterCollector
from alphaloop.data.store import MarketDataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase2-backfill")

CORE_CTA_SYMBOLS = [
    "PAXG", "BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "AAVE", "UNI", "SUI", "HYPE", "LTC",
]
ALT_CARRY_SYMBOLS = [
    "ADA", "APT", "ARB", "ENA", "INJ", "NEAR", "OP", "PENDLE", "TIA", "VVV", "WLD", "ZEC",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill 180d-365d real multi-exchange history")
    parser.add_argument("--db", default=str(ROOT / "data" / "market_data.sqlite"))
    parser.add_argument("--funding-days", type=int, default=190)
    parser.add_argument("--cta-candle-days", type=int, default=365)
    parser.add_argument("--alt-candle-days", type=int, default=185)
    args = parser.parse_args()

    store = MarketDataStore(args.db)
    hl = HyperliquidCollector()
    lt = LighterCollector()
    gate = GateCollector()

    all_syms = CORE_CTA_SYMBOLS + ALT_CARRY_SYMBOLS
    t0 = time.monotonic()
    stats = {"hl_fr": 0, "hl_1h": 0, "lt_fr": 0, "gate_fr": 0, "gate_1h": 0, "errors": 0}

    try:
        for idx, sym in enumerate(all_syms, 1):
            c_days = args.cta_candle_days if sym in CORE_CTA_SYMBOLS else args.alt_candle_days
            hl_days = max(210, args.funding_days)

            for tag, fn, upsert in (
                ("hl_fr", lambda s=sym: hl.fetch_funding_history(s, days=hl_days), store.upsert_funding),
                ("hl_1h", lambda s=sym: hl.fetch_candles_1h(s, days=210), store.upsert_candles),
                ("lt_fr", lambda s=sym: lt.fetch_funding_history(s, days=args.funding_days), store.upsert_funding),
                ("gate_fr", lambda s=sym: gate.fetch_funding_history(s, days=179), store.upsert_funding),
                ("gate_1h", lambda s=sym, d=c_days: gate.fetch_candles_1h(s, days=d), store.upsert_candles),
            ):
                try:
                    rows = fn()
                    if rows:
                        upsert(rows)
                        stats[tag] += len(rows)
                except Exception as exc:
                    stats["errors"] += 1
                    logger.warning("[%d/%d] %s %s failed: %s", idx, len(all_syms), sym, tag, exc)

            logger.info(
                "[%d/%d] %s done (elapsed=%.1fs | totals=%s)",
                idx, len(all_syms), sym, time.monotonic() - t0, stats,
            )
    finally:
        store.close()

    logger.info("Phase 2 backfill complete in %.1fs: %s", time.monotonic() - t0, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
