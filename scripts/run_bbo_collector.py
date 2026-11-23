#!/usr/bin/env python3
"""Background live 4-exchange BBO micro-probe collector.

Continuously captures synchronized Hyperliquid, Lighter, OKX, and Gate top-of-book
bid/ask quotes into SQLite WAL (`bbo_ticks`) so development and empirical BBO
validation run in parallel without blocking each other.
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

from alphaloop.core.models import FundingRateBar
from alphaloop.data.bbo_collector import DEFAULT_BBO_SYMBOLS, MultiVenueBBOCollector
from alphaloop.data.collectors import HyperliquidCollector, LighterCollector
from alphaloop.data.store import MarketDataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bbo-collector")


def _sync_recent_realized_funding(symbols: list[str], hl: HyperliquidCollector, lt: LighterCollector) -> list[FundingRateBar]:
    """Sync strictly realized hourly funding settlements (last 1 day) without polluting cadence alignment."""
    bars: list[FundingRateBar] = []
    for sym in symbols:
        try:
            bars.extend(hl.fetch_funding_history(sym, days=1))
        except Exception as exc:
            logger.warning("Recent funding HL skip %s: %s", sym, exc)
        try:
            bars.extend(lt.fetch_funding_history(sym, days=1))
        except Exception as exc:
            logger.warning("Recent funding Lighter skip %s: %s", sym, exc)
    return bars


def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaLoop Live 4-Exchange BBO (+Funding) Collector")
    parser.add_argument("--db", default=str(ROOT / "data" / "market_data.sqlite"))
    parser.add_argument("--symbols", default=",".join(DEFAULT_BBO_SYMBOLS))
    parser.add_argument("--interval-sec", type=float, default=20.0)
    parser.add_argument("--duration-minutes", type=float, default=45.0)
    parser.add_argument("--with-live-funding", action="store_true", help="Also sync realized hourly funding every 15 rounds")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    store = MarketDataStore(args.db)
    collector = MultiVenueBBOCollector(symbols=symbols)
    hl = HyperliquidCollector() if args.with_live_funding else None
    lt = LighterCollector() if args.with_live_funding else None

    deadline = time.monotonic() + max(0.2, args.duration_minutes) * 60.0
    rounds = 0
    total_saved = 0
    total_funding_saved = 0
    try:
        while True:
            t0 = time.monotonic()
            ticks, errs = collector.capture_round()
            saved = store.upsert_bbo_ticks(ticks)
            rounds += 1
            total_saved += saved
            if args.with_live_funding and hl is not None and lt is not None and (rounds == 1 or rounds % 15 == 0):
                fbars = _sync_recent_realized_funding(symbols, hl, lt)
                if fbars:
                    total_funding_saved += store.upsert_funding(fbars)
            if rounds == 1 or rounds % 5 == 0:
                logger.info(
                    "Round #%d saved=%d total=%d funding_saved=%d errors=%s",
                    rounds, saved, total_saved, total_funding_saved, errs or "none",
                )
            if time.monotonic() >= deadline:
                break
            sleep_s = max(1.0, args.interval_sec - (time.monotonic() - t0))
            if time.monotonic() + sleep_s > deadline:
                break
            time.sleep(sleep_s)
    finally:
        store.close()
    logger.info("BBO collection complete: rounds=%d total_ticks_saved=%d funding_saved=%d", rounds, total_saved, total_funding_saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
