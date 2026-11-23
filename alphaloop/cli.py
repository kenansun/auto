"""Unified CLI entrypoint for AlphaLoop v2 Strategy R&D Platform."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from alphaloop.data.collectors import GateCollector, HyperliquidCollector, LighterCollector, OKXCollector
from alphaloop.data.quality import DataQualityAuditor
from alphaloop.data.store import MarketDataStore
from alphaloop.fleet.ssh_auditor import ReadOnlyFleetAuditor
from alphaloop.plugins import default_registry
from alphaloop.web.server import ConsoleAppContext, create_server

DEFAULT_UNIVERSE = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "AAVE",
    "UNI", "SUI", "APT", "ARB", "OP", "NEAR", "ADA", "LTC",
    "HYPE", "ZEC", "PENDLE", "ENA", "WLD", "TIA", "INJ", "VVV",
]


def cmd_fetch(args: argparse.Namespace) -> int:
    store = MarketDataStore(Path(args.data_dir) / "market_data.sqlite")
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else DEFAULT_UNIVERSE
    hl = HyperliquidCollector()
    lt = LighterCollector()
    okx = OKXCollector()
    gate = GateCollector()

    print(f"Fetching {args.days}d REAL exchange history for {len(symbols)} symbols -> {store.db_path}")
    for sym in symbols:
        counts: dict[str, int] = {}
        try:
            bars = hl.fetch_funding_history(sym, days=args.days)
            counts["hl_fr"] = store.upsert_funding(bars)
            if args.candles:
                cbars = hl.fetch_candles_1h(sym, days=args.days)
                counts["hl_1h"] = store.upsert_candles(cbars)
        except Exception as exc:
            logging.warning("HL fetch failed for %s: %s", sym, exc)

        try:
            bars = lt.fetch_funding_history(sym, days=args.days)
            counts["lt_fr"] = store.upsert_funding(bars)
        except Exception as exc:
            logging.warning("Lighter fetch failed for %s: %s", sym, exc)

        try:
            bars = okx.fetch_funding_history(sym, days=args.days)
            counts["okx_fr"] = store.upsert_funding(bars)
            if args.candles and bars:
                cbars = okx.fetch_candles_1h(sym, days=args.days, spot=False)
                counts["okx_1h"] = store.upsert_candles(cbars)
        except Exception as exc:
            logging.warning("OKX fetch failed for %s: %s", sym, exc)

        try:
            bars = gate.fetch_funding_history(sym, days=args.days)
            counts["gate_fr"] = store.upsert_funding(bars)
            if args.candles and bars:
                cbars = gate.fetch_candles_1h(sym, days=args.days)
                counts["gate_1h"] = store.upsert_candles(cbars)
        except Exception as exc:
            logging.warning("Gate fetch failed for %s: %s", sym, exc)

        print(f"  {sym:8s} -> {counts}")

    print("Summary:", json.dumps(store.summary_stats(), indent=2))
    store.close()
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    hl = HyperliquidCollector()
    lt = LighterCollector()
    hl_live = hl.fetch_live_universe()
    lt_all = lt.fetch_live_funding_all()
    lt_live = lt_all.get("lighter", {})
    bn_live = lt_all.get("binance", {})

    rows = []
    for sym, h in hl_live.items():
        if h["oi_usd"] < args.min_oi_m * 1e6 or h["vol_24h_usd"] < args.min_vol_m * 1e6:
            continue
        if sym in lt_live:
            diff_lt = h["apr_pct"] - lt_live[sym]
            rows.append((abs(diff_lt), sym, "HL vs Lighter", h["apr_pct"], lt_live[sym], diff_lt, h["oi_usd"] / 1e6))
        if sym in bn_live:
            diff_bn = h["apr_pct"] - bn_live[sym]
            rows.append((abs(diff_bn), sym, "HL vs Binance", h["apr_pct"], bn_live[sym], diff_bn, h["oi_usd"] / 1e6))

    rows.sort(reverse=True)
    print(f"Top {args.top} Live Cross-Venue Funding Spreads (OI >= ${args.min_oi_m}M):")
    for abs_d, sym, pair, a, b, d, oi in rows[: args.top]:
        side = "Short A / Long B" if d > 0 else "Long A / Short B"
        print(f"  {sym:8s} | {pair:14s} | A={a:+7.2f}% | B={b:+7.2f}% | Spread={abs_d:6.2f}% | {side} | OI=${oi:.1f}M")
    return 0


def cmd_dq_audit(args: argparse.Namespace) -> int:
    store = MarketDataStore(Path(args.data_dir) / "market_data.sqlite")
    auditor = DataQualityAuditor(store)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else DEFAULT_UNIVERSE[:10]
    if args.csv:
        pcols = [c.strip() for c in args.price_cols.split(",") if c.strip()] if args.price_cols else []
        rep = auditor.audit_csv_sample(args.csv, dataset_name=Path(args.csv).name, ts_col=args.ts_col, price_cols=pcols)
        print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
    elif args.bbo:
        rep, summary = auditor.audit_bbo_ticks(args.venue_a, args.venue_b, symbols, max_stale_sec=None)
        print(json.dumps({"report": rep.to_dict(), "bbo_summary": summary}, indent=2, ensure_ascii=False))
    else:
        rep = auditor.audit_funding_pair(args.venue_a, args.venue_b, symbols)
        print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False))
    store.close()
    return 0


def cmd_plugins(_args: argparse.Namespace) -> int:
    reg = default_registry()
    print(json.dumps(reg.list_plugins(), indent=2, ensure_ascii=False))
    return 0


def cmd_micro_run(args: argparse.Namespace) -> int:
    ctx = ConsoleAppContext(args.data_dir)
    cfg = ctx.config_store.read_active(args.strategy_id)
    if cfg is None:
        print(f"Strategy not found: {args.strategy_id}", file=sys.stderr)
        ctx.close()
        return 1
    rec = ctx.micro_session.evaluate_bbo_window(cfg, target_hours=args.target_hours)
    print(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False))
    ctx.close()
    return 0


def cmd_fleet_audit(args: argparse.Namespace) -> int:
    auditor = ReadOnlyFleetAuditor(args.roles, args.ssh_key)
    save_path = Path(args.data_dir) / "fleet_audit" / "fleet_snapshot.json"
    result = auditor.audit_all(save_path=save_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    srv, ctx = create_server(args.data_dir, host=args.host, port=args.port)
    print(f"AlphaLoop v2 Console serving on http://{args.host}:{args.port} (data_dir={args.data_dir})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down AlphaLoop Console.")
    finally:
        srv.server_close()
        ctx.close()
    return 0


def cmd_agent_cycle(args: argparse.Namespace) -> int:
    ctx = ConsoleAppContext(args.data_dir)
    cfg = ctx.config_store.read_active(args.strategy_id)
    if cfg is None:
        print(f"Strategy not found: {args.strategy_id}", file=sys.stderr)
        ctx.close()
        return 1
    if args.rounds > 1:
        sprint = ctx.orchestrator.run_multi_round_sprint(cfg, max_rounds=args.rounds, version_prefix=args.next_version)
        print(json.dumps({
            "strategy_id": sprint["strategy_id"],
            "rounds_executed": sprint["rounds_executed"],
            "stop_reason": sprint["stop_reason"],
            "final_version": sprint["final_version"],
            "final_promotion_stage": sprint["final_promotion_stage"],
        }, indent=2))
    else:
        rec = ctx.orchestrator.run_cycle(cfg, next_version=args.next_version)
        print(json.dumps({
            "cycle_id": rec["cycle_id"],
            "plugin_id": rec["plugin_id"],
            "anomaly": rec["feedback"]["anomaly_type"],
            "proposer": rec["proposal"]["model_used"],
            "baseline_apr": rec["baseline_metrics"]["annualized_apr_pct"],
            "candidate_apr": rec["candidate_metrics"]["annualized_apr_pct"],
            "dq_tier": rec["gate_verdict"]["dq_tier"],
            "gates": f"{rec['gate_verdict']['passed_count']}/{rec['gate_verdict']['total_count']}",
            "red_team": rec["red_team_verdict"]["decision"],
            "promotion_stage": rec["promotion_stage"],
        }, indent=2))
    ctx.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="alphaloop", description="AlphaLoop v2 Strategy Factory & R&D Platform")
    p.add_argument("--data-dir", default="data", help="Directory for SQLite databases and active configs")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="Fetch real multi-venue historical funding & candles")
    pf.add_argument("--days", type=int, default=90)
    pf.add_argument("--symbols", default="", help="Comma-separated symbols (default 24-coin liquid universe)")
    pf.add_argument("--candles", action="store_true", help="Also fetch 1h OHLCV candles for real basis PnL")
    pf.set_defaults(func=cmd_fetch)

    ps = sub.add_parser("scan", help="Live cross-venue funding spread scanner")
    ps.add_argument("--min-oi-m", type=float, default=5.0)
    ps.add_argument("--min-vol-m", type=float, default=5.0)
    ps.add_argument("--top", type=int, default=20)
    ps.set_defaults(func=cmd_scan)

    pdq = sub.add_parser("dq-audit", help="Run DQ-01..DQ-08 certification on funding/BBO/CSV datasets")
    pdq.add_argument("--venue-a", default="hyperliquid")
    pdq.add_argument("--venue-b", default="okx")
    pdq.add_argument("--symbols", default="")
    pdq.add_argument("--bbo", action="store_true", help="Audit live captured BBO ticks")
    pdq.add_argument("--csv", default="", help="Audit a local/server-captured CSV file")
    pdq.add_argument("--ts-col", default="ts_ms")
    pdq.add_argument("--price-cols", default="")
    pdq.set_defaults(func=cmd_dq_audit)

    ppl = sub.add_parser("plugins", help="List registered Strategy Factory v2 plugins")
    ppl.set_defaults(func=cmd_plugins)

    pmr = sub.add_parser("micro-run", help="Evaluate captured BBO window via Fast Micro-Run state machine")
    pmr.add_argument("--strategy-id", required=True)
    pmr.add_argument("--target-hours", type=float, default=24.0)
    pmr.set_defaults(func=cmd_micro_run)

    pfl = sub.add_parser("fleet-audit", help="Run strictly read-only allowlisted SSH audit across configured servers")
    pfl.add_argument("--roles", default="config/fleet_roles.json")
    pfl.add_argument("--ssh-key", default=os.environ.get("ALPHALOOP_SSH_KEY") or str(Path.home() / ".ssh" / "id_ed25519"))
    pfl.set_defaults(func=cmd_fleet_audit)

    pv = sub.add_parser("serve", help="Run local Strategy R&D Web Console v2")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8796)
    pv.set_defaults(func=cmd_serve)

    pa = sub.add_parser("agent-cycle", help="Run closed-loop Agent evolution cycle or multi-round sprint")
    pa.add_argument("--strategy-id", required=True)
    pa.add_argument("--next-version", required=True)
    pa.add_argument("--rounds", type=int, default=1)
    pa.set_defaults(func=cmd_agent_cycle)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
