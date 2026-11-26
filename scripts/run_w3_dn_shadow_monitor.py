#!/usr/bin/env python3
"""CLI Entrypoint: W3-DN Dual-Leg Spot+Perp Read-Only Shadow Blind-Run Monitor (>=48h Gate).

Strictly read-only (`SHADOW_LIVE_EXECUTION_ENABLED = False`). Never places orders,
never touches `scripts/run_shadow_daemon.py` or live execution credentials.

Examples:
  # 3-tick live public read-only smoke verification (zero orders)
  python3 scripts/run_w3_dn_shadow_monitor.py --state-dir data/w3_dn_shadow --ticks 3 --interval-sec 5

  # Production/Tokyo P8 MD-Gateway BBO file reuse mode (STD-32)
  python3 scripts/run_w3_dn_shadow_monitor.py --state-dir data/w3_dn_shadow --p8-feed /opt/dex_cex_research/scripts/events.csv --ticks 2880 --interval-sec 60
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.platform.shadow.w3_dn_shadow_monitor import (
    DEFAULT_W3_DN_SHADOW_COINS,
    SHADOW_LIVE_EXECUTION_ENABLED,
    P8MDGatewayFileQuoteProvider,
    PublicRestWaterfallQuoteProvider,
    ShadowMonitorConfig,
    W3DNShadowMonitor,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W3-DN Read-Only Shadow Blind-Run Monitor (Zero Live Execution)")
    p.add_argument("--state-dir", type=Path, default=ROOT / "data" / "w3_dn_shadow", help="SQLite/log/heartbeat directory")
    p.add_argument("--coins", type=str, default=",".join(DEFAULT_W3_DN_SHADOW_COINS), help="Comma-separated primary coins")
    p.add_argument(
        "--observe-only-coins",
        type=str,
        default="",
        help="Optional comma-separated diagnostic-only coins (L2/basis telemetry only, zero virtual positions)",
    )
    p.add_argument("--ticks", type=int, default=1, help="Number of polling ticks (0 = run until SIGINT/SIGTERM)")
    p.add_argument("--interval-sec", type=float, default=60.0, help="Polling interval in seconds")
    p.add_argument("--capital-usd", type=float, default=10_000.0, help="Virtual shadow capital USD")
    p.add_argument("--p8-feed", type=Path, default=None, help="Optional STD-32 P8 MD-Gateway BBO JSONL/CSV path")
    p.add_argument("--p16-heartbeat-url", type=str, default="", help="Optional local P16 watchdog heartbeat URL")
    p.add_argument("--p16-heartbeat-path", type=Path, default=None, help="Optional explicit P16 heartbeat JSON path")
    p.add_argument(
        "--live-readiness-path",
        type=Path,
        default=None,
        help="Optional atomic 0644 readiness JSON updated after every tick for continuous monitoring",
    )
    p.add_argument(
        "--prewarm",
        action="store_true",
        help="Fetch read-only public historical 12h bars to pre-warm kernel before live ticks",
    )
    p.add_argument(
        "--prewarm-bars",
        type=int,
        default=12,
        help="Number of historical 12h bars for --prewarm (default: 12)",
    )
    p.add_argument("--report-out", type=Path, default=None, help="Optional JSON output path for final readiness report")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if SHADOW_LIVE_EXECUTION_ENABLED:
        print("FATAL: SHADOW_LIVE_EXECUTION_ENABLED must be False", file=sys.stderr)
        return 2

    args = parse_args(argv)
    coins = tuple(c.strip().upper() for c in str(args.coins).split(",") if c.strip())
    obs_coins = tuple(c.strip().upper() for c in str(args.observe_only_coins).split(",") if c.strip())
    prewarm_bars = max(0, int(args.prewarm_bars))
    cfg = ShadowMonitorConfig(
        state_dir=args.state_dir,
        coins=coins or DEFAULT_W3_DN_SHADOW_COINS,
        observe_only_coins=obs_coins,
        capital_usd=float(args.capital_usd),
        enable_prewarm=bool(args.prewarm),
        prewarm_bars=prewarm_bars,
        p16_heartbeat_url=str(args.p16_heartbeat_url or ""),
        p16_heartbeat_path=args.p16_heartbeat_path,
        readiness_report_path=args.live_readiness_path,
    )
    provider = (
        P8MDGatewayFileQuoteProvider(args.p8_feed)
        if args.p8_feed is not None
        else PublicRestWaterfallQuoteProvider()
    )

    stop_requested = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        _ = signum
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    tick_results: list[dict[str, object]] = []
    with W3DNShadowMonitor(cfg, provider=provider) as mon:
        if (
            bool(args.prewarm)
            and prewarm_bars > 0
            and not mon._has_live_execution_started()
            and mon._prewarmed_bars == 0
            and hasattr(provider, "fetch_historical_prewarm_bars")
        ):
            try:
                hist_frames = provider.fetch_historical_prewarm_bars(
                    cfg.coins,
                    bars=prewarm_bars,
                    bucket_sec=int(cfg.kernel_bar_interval_sec),
                )
                pw_res = mon.prewarm_from_bars(hist_frames, max_bars=prewarm_bars)
                print(
                    json.dumps({"event": "W3DN_CLI_PREWARM", **pw_res}, sort_keys=True, allow_nan=False),
                    flush=True,
                )
            except Exception as exc:
                print(f"WARN: W3DN_CLI_PREWARM failed (continuing live ticks): {exc}", file=sys.stderr, flush=True)
        n_ticks = int(args.ticks)
        idx = 0
        while not stop_requested:
            res = mon.run_once()
            tick_results.append(res)
            idx += 1
            if n_ticks > 0 and idx >= n_ticks:
                break
            sleep_s = max(0.2, float(args.interval_sec))
            deadline = time.monotonic() + sleep_s
            while not stop_requested and time.monotonic() < deadline:
                time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        readiness = mon.evaluate_48h_readiness().to_dict()

    summary = {
        "live_execution_enabled": SHADOW_LIVE_EXECUTION_ENABLED,
        "ticks_executed": len(tick_results),
        "last_tick": tick_results[-1] if tick_results else None,
        "readiness": readiness,
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if readiness["verdict"] != "FAIL_CLOSED_BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
