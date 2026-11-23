#!/usr/bin/env python3
"""AlphaLoop v2.2 Always-On Shadow Portfolio Daemon Runner (Stage 1 S1-03).

Strictly SHADOW_DRY_RUN (`LIVE_EXECUTION_ENABLED = False`).
Zero live order placement, zero private keys, loopback/local-watchdog only.

Examples:
  # Bootstrap verified Balanced 50/50 SHADOW_LIVE configs and run 1 cycle
  python3 scripts/run_shadow_daemon.py --data-dir data --bootstrap-balanced --once

  # Run continuously under systemd collecting live BBO every 30s
  python3 scripts/run_shadow_daemon.py --data-dir /opt/alphaloop/data --collect-live --interval-sec 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.core.config import HotConfigStore
from alphaloop.daemon.shadow_runner import LIVE_EXECUTION_ENABLED, ShadowPortfolioDaemon
from alphaloop.plugins import FundingCarryPlugin, TrendCTAPlugin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("alphaloop-shadow-daemon")

CTA_UNIVERSE = [
    "PAXG", "BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "AAVE", "UNI", "SUI", "HYPE", "LTC",
]
CARRY_UNIVERSE = CTA_UNIVERSE + [
    "ADA", "APT", "ARB", "ENA", "NEAR", "OP", "PENDLE", "TIA", "VVV", "WLD", "ZEC",
]


def bootstrap_balanced_shadow_configs(data_dir: Path) -> list[str]:
    """Seed verified Balanced 50/50 SHADOW_LIVE configs if not already present (idempotent)."""
    store = HotConfigStore(data_dir / "state")
    seeded: list[str] = []
    try:
        if store.read_active("hl_trend_cta_prod") is None:
            cta_cfg = TrendCTAPlugin().default_config(
                symbols=CTA_UNIVERSE,
                strategy_id="hl_trend_cta_prod",
                version="v2.2-shadow",
                status="SHADOW_LIVE",
                venue_a="hyperliquid",
                venue_b="hyperliquid",
                capital_usd=1500.0,
                max_positions=5,
                leverage=2.0,
                execution_mode="TAKER_TAKER",
                params={
                    "timeframe_hours": 12,
                    "entry_n": 40,
                    "exit_n": 16,
                    "atr_n": 14,
                    "atr_mult": 3.5,
                    "risk_per_trade_pct": 0.8,
                    "max_symbol_leverage": 1.5,
                    "ref_venue": "gate",
                },
            )
            store.register_version(cta_cfg, actor="shadow-bootstrap", reason="Balanced 50/50 Pillar 2 SHADOW_LIVE seed")
            seeded.append("hl_trend_cta_prod")

        if store.read_active("hl_gate_safe_carry_prod") is None:
            carry_cfg = FundingCarryPlugin().default_config(
                symbols=CARRY_UNIVERSE,
                strategy_id="hl_gate_safe_carry_prod",
                version="v2.2-shadow",
                status="SHADOW_LIVE",
                venue_a="hyperliquid",
                venue_b="gate",
                capital_usd=1500.0,
                max_positions=4,
                leverage=3.5,
                slippage_per_leg_bps=0.75,
                params={
                    "vol_tier_low_ann_pct": 65.0,
                    "vol_tier_high_ann_pct": 130.0,
                    "min_symbol_leverage": 1.5,
                    "protective_margin_exit_pct": 0.50,
                },
            )
            store.register_version(carry_cfg, actor="shadow-bootstrap", reason="Balanced 50/50 Pillar 1 SHADOW_LIVE seed")
            seeded.append("hl_gate_safe_carry_prod")
    finally:
        store.close()
    return seeded


def main(argv: list[str] | None = None) -> int:
    if LIVE_EXECUTION_ENABLED:
        logger.error("Refusing to run: LIVE_EXECUTION_ENABLED must remain False")
        return 2

    parser = argparse.ArgumentParser(description="AlphaLoop v2.2 Always-On Shadow Portfolio Daemon")
    parser.add_argument("--data-dir", default="data", help="Runtime data directory (SQLite + active configs)")
    parser.add_argument("--interval-sec", type=float, default=30.0, help="Loop interval seconds (min 2.0)")
    parser.add_argument("--max-bbo-stale-sec", type=float, default=180.0, help="Stale BBO halt threshold")
    parser.add_argument("--max-spread-ratio", type=float, default=1.5, help="Wide spread ratio threshold")
    parser.add_argument("--wide-spread-kill-sec", type=float, default=7200.0, help="Sustained wide spread kill window")
    parser.add_argument("--max-drift-bps", type=float, default=150.0, help="Shadow vs backtest drift kill threshold")
    parser.add_argument("--collect-live", action="store_true", help="Poll live exchange BBO inside each cycle")
    parser.add_argument("--bootstrap-balanced", action="store_true", help="Seed Balanced 50/50 SHADOW_LIVE configs if missing")
    parser.add_argument("--once", action="store_true", help="Execute exactly one shadow cycle and print JSON")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if args.bootstrap_balanced:
        seeded = bootstrap_balanced_shadow_configs(data_dir)
        if seeded:
            logger.info("Bootstrapped SHADOW_LIVE strategies: %s", seeded)

    daemon = ShadowPortfolioDaemon(
        data_dir=data_dir,
        max_bbo_stale_sec=args.max_bbo_stale_sec,
        max_spread_ratio=args.max_spread_ratio,
        wide_spread_kill_sec=args.wide_spread_kill_sec,
        max_drift_bps=args.max_drift_bps,
    )
    if args.once:
        try:
            summary = daemon.run_once(collect_live=args.collect_live)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        finally:
            daemon.close()

    daemon.run_forever(poll_interval_sec=args.interval_sec, collect_live=args.collect_live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
