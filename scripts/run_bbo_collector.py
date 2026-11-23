#!/usr/bin/env python3
"""Background live 4-exchange BBO (+Funding +1h Candles) collector (Red-Team BLOCKER #5 Hardened).

Continuously captures synchronized Hyperliquid, Lighter, OKX, and Gate top-of-book
bid/ask quotes into SQLite WAL (`bbo_ticks`), plus periodic realized funding (HL/Lighter/Gate)
and recent 1h candles (HL/Gate) so `ShadowPortfolioDaemon` and `ShadowParityAuditor` stay live.

Use `--duration-minutes 0` for always-on systemd execution with graceful SIGINT/SIGTERM stop.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.core.models import FundingRateBar, PriceBar
from alphaloop.data.bbo_collector import DEFAULT_BBO_SYMBOLS, MultiVenueBBOCollector
from alphaloop.data.collectors import GateCollector, HyperliquidCollector, LighterCollector
from alphaloop.data.store import MarketDataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bbo-collector")


def _sync_recent_realized_funding(
    symbols: list[str],
    hl: HyperliquidCollector,
    lt: LighterCollector,
    gate: GateCollector,
) -> list[FundingRateBar]:
    """Sync strictly realized funding settlements (last 2 days) across HL, Lighter, and Gate."""
    bars: list[FundingRateBar] = []
    for sym in symbols:
        try:
            bars.extend(hl.fetch_funding_history(sym, days=2))
        except Exception as exc:
            logger.warning("Recent funding HL skip %s: %s", sym, exc)
        try:
            bars.extend(lt.fetch_funding_history(sym, days=2))
        except Exception as exc:
            logger.warning("Recent funding Lighter skip %s: %s", sym, exc)
        try:
            bars.extend(gate.fetch_funding_history(sym, days=2))
        except Exception as exc:
            logger.warning("Recent funding Gate skip %s: %s", sym, exc)
    return bars


def _sync_recent_candles_1h(
    symbols: list[str],
    hl: HyperliquidCollector,
    gate: GateCollector,
    days: int = 2,
) -> list[PriceBar]:
    """Sync recent 1h OHLCV candles across HL & Gate so ShadowParityAuditor has continuous bars."""
    candles: list[PriceBar] = []
    for sym in symbols:
        try:
            candles.extend(hl.fetch_candles_1h(sym, days=days))
        except Exception as exc:
            logger.warning("Recent 1h candle HL skip %s: %s", sym, exc)
        try:
            candles.extend(gate.fetch_candles_1h(sym, days=days))
        except Exception as exc:
            logger.warning("Recent 1h candle Gate skip %s: %s", sym, exc)
    return candles


def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaLoop Live 4-Exchange BBO (+Funding +Candles) Collector")
    parser.add_argument("--db", default=str(ROOT / "data" / "market_data.sqlite"))
    parser.add_argument("--symbols", default=",".join(DEFAULT_BBO_SYMBOLS))
    parser.add_argument("--interval-sec", type=float, default=20.0)
    parser.add_argument("--duration-minutes", type=float, default=45.0, help="0 = run forever until SIGINT/SIGTERM")
    parser.add_argument("--with-live-funding", action="store_true", help="Sync realized HL/Lighter/Gate funding every 30 rounds")
    parser.add_argument("--with-live-candles", action="store_true", help="Sync recent 1h HL/Gate candles every 60 rounds")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    store = MarketDataStore(args.db)
    collector = MultiVenueBBOCollector(symbols=symbols)
    need_collectors = args.with_live_funding or args.with_live_candles
    hl = HyperliquidCollector() if need_collectors else None
    lt = LighterCollector() if args.with_live_funding else None
    gate = GateCollector() if need_collectors else None

    stop_flag = [False]

    def _on_sig(signum: int, _frame: Any) -> None:
        logger.info("BBO collector received signal %s; shutting down cleanly...", signum)
        stop_flag[0] = True

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    infinite = float(args.duration_minutes) <= 0.0
    deadline = None if infinite else (time.monotonic() + max(0.2, args.duration_minutes) * 60.0)
    rounds = 0
    total_saved = 0
    total_funding_saved = 0
    total_candles_saved = 0
    try:
        while not stop_flag[0]:
            t0 = time.monotonic()
            ticks, errs = collector.capture_round()
            saved = store.upsert_bbo_ticks(ticks)
            rounds += 1
            total_saved += saved

            if args.with_live_funding and hl is not None and lt is not None and gate is not None and (rounds == 1 or rounds % 30 == 0):
                fbars = _sync_recent_realized_funding(symbols, hl, lt, gate)
                if fbars:
                    total_funding_saved += store.upsert_funding(fbars)

            if args.with_live_candles and hl is not None and gate is not None and (rounds == 1 or rounds % 60 == 0):
                cbars = _sync_recent_candles_1h(symbols, hl, gate, days=25 if rounds == 1 else 2)
                if cbars:
                    total_candles_saved += store.upsert_candles(cbars)

            if rounds == 1 or rounds % 5 == 0:
                logger.info(
                    "Round #%d saved=%d total=%d funding_saved=%d candles_saved=%d errors=%s",
                    rounds, saved, total_saved, total_funding_saved, total_candles_saved, errs or "none",
                )
            if deadline is not None and time.monotonic() >= deadline:
                break
            sleep_s = max(1.0, args.interval_sec - (time.monotonic() - t0))
            if deadline is not None and time.monotonic() + sleep_s > deadline:
                break
            # Sleep in 1s slices for prompt SIGTERM response
            slept = 0.0
            while slept < sleep_s and not stop_flag[0]:
                step = min(1.0, sleep_s - slept)
                time.sleep(step)
                slept += step
    finally:
        store.close()
    logger.info(
        "BBO collection complete: rounds=%d total_ticks=%d funding=%d candles=%d",
        rounds, total_saved, total_funding_saved, total_candles_saved,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
