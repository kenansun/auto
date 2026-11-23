#!/usr/bin/env python3
"""AlphaLoop v2.2 Unattended Nightly Strategy Factory Worker (Stage 1 S1-03).

Designed for scheduled daily execution (e.g., systemd timer on research worker node):
  1. Incremental real-exchange backfill (recent `backfill_days` funding + 1h candles).
  2. Hard DQ-01..DQ-08 re-certification on HL single-perp, Gate single-perp, and HL<->Gate carry.
  3. 6-Gate regression evaluation across active strategies.
  4. Optional single-round closed-loop AgentEvolutionOrchestrator check (never auto-promotes LIVE).
  5. Atomic JSON report export (`<data_dir>/nightly_factory_report.json`) + Watchdog heartbeat.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.agent_loop.model_router import MultiModelRouter
from alphaloop.agent_loop.orchestrator import AgentEvolutionOrchestrator
from alphaloop.core.config import HotConfigStore
from alphaloop.core.provenance import utc_now_ms
from alphaloop.daemon.shadow_runner import _atomic_write_json
from alphaloop.daemon.watchdog_hook import WatchdogHook
from alphaloop.data.collectors import GateCollector, HyperliquidCollector, LighterCollector
from alphaloop.data.quality import DataQualityAuditor
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier, TrialLedger
from alphaloop.harness.micro_runner import MicroRunStore
from alphaloop.plugins import default_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("alphaloop-nightly-factory")

CTA_UNIVERSE = [
    "PAXG", "BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "AAVE", "UNI", "SUI", "HYPE", "LTC",
]
CARRY_UNIVERSE = CTA_UNIVERSE + [
    "ADA", "APT", "ARB", "ENA", "NEAR", "OP", "PENDLE", "TIA", "VVV", "WLD", "ZEC",
]


def run_incremental_backfill(store: MarketDataStore, days: int = 3) -> dict[str, Any]:
    """Incrementally refresh recent funding and 1h candles across HL/Gate/Lighter."""
    hl = HyperliquidCollector()
    gate = GateCollector()
    lt = LighterCollector()
    stats: dict[str, Any] = {"days": int(days), "hl_fr": 0, "hl_1h": 0, "gate_fr": 0, "gate_1h": 0, "lt_fr": 0, "errors": []}

    for sym in CARRY_UNIVERSE:
        try:
            fr = hl.fetch_funding_history(sym, days=days)
            stats["hl_fr"] += store.upsert_funding(fr)
            cd = hl.fetch_candles_1h(sym, days=days)
            stats["hl_1h"] += store.upsert_candles(cd)
        except Exception as exc:
            stats["errors"].append(f"hl:{sym}:{exc}")

        try:
            gfr = gate.fetch_funding_history(sym, days=days)
            stats["gate_fr"] += store.upsert_funding(gfr)
            gcd = gate.fetch_candles_1h(sym, days=days)
            stats["gate_1h"] += store.upsert_candles(gcd)
        except Exception as exc:
            stats["errors"].append(f"gate:{sym}:{exc}")

        try:
            lfr = lt.fetch_funding_history(sym, days=days)
            stats["lt_fr"] += store.upsert_funding(lfr)
        except Exception as exc:
            stats["errors"].append(f"lt:{sym}:{exc}")
    return stats


def run_nightly_factory(
    data_dir: str | Path,
    *,
    skip_fetch: bool = False,
    backfill_days: int = 3,
    run_evolution: bool = False,
) -> dict[str, Any]:
    root_dir = Path(data_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    state_dir = root_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    started_ms = utc_now_ms()
    store = MarketDataStore(root_dir / "market_data.sqlite")
    ledger = TrialLedger(root_dir / "trial_ledger.sqlite")
    cfg_store = HotConfigStore(state_dir)
    micro_store = MicroRunStore(state_dir / "micro_runs.sqlite")
    registry = default_registry()
    backtester = FundingCarryBacktester(store)
    verifier = SixGateVerifier(backtester, ledger, registry=registry)
    router = MultiModelRouter(budget_db_path=str(state_dir / "llm_budget.sqlite"))
    orchestrator = AgentEvolutionOrchestrator(
        state_dir=state_dir,
        backtester=backtester,
        verifier=verifier,
        config_store=cfg_store,
        router=router,
        registry=registry,
        micro_store=micro_store,
    )
    auditor = DataQualityAuditor(store)
    watchdog = WatchdogHook(async_mode=False)

    try:
        backfill = {"skipped": True} if skip_fetch else run_incremental_backfill(store, days=backfill_days)

        dq_summary: dict[str, Any] = {}
        try:
            rep_hl = auditor.audit_single_venue_perp("hyperliquid", CTA_UNIVERSE, ref_venue="gate", dataset_name="nightly_hl_perp")
            store.save_dq_certificate(rep_hl)
            dq_summary["hl_perp"] = {"tier": rep_hl.tier, "passed": f"{rep_hl.passed_count}/{rep_hl.total_count}", "rows": rep_hl.row_count}
        except Exception as exc:
            dq_summary["hl_perp"] = {"error": str(exc)}

        try:
            rep_pair = auditor.audit_funding_pair("hyperliquid", "gate", CARRY_UNIVERSE, dataset_name="nightly_hl_gate")
            store.save_dq_certificate(rep_pair)
            dq_summary["hl_gate_carry"] = {"tier": rep_pair.tier, "passed": f"{rep_pair.passed_count}/{rep_pair.total_count}", "rows": rep_pair.row_count}
        except Exception as exc:
            dq_summary["hl_gate_carry"] = {"error": str(exc)}

        regressions: list[dict[str, Any]] = []
        evolution_cycles: list[dict[str, Any]] = []
        for item in cfg_store.list_strategies():
            sid = item["strategy_id"]
            cfg = cfg_store.read_active(sid)
            if cfg is None:
                continue
            try:
                verdict, _res, _extras = verifier.verify(cfg)
                regressions.append({
                    "strategy_id": sid,
                    "version": cfg.version,
                    "plugin_id": cfg.plugin_id,
                    "overall_pass": verdict.overall_pass,
                    "gates": f"{verdict.passed_count}/{verdict.total_count}",
                    "dq_tier": verdict.dq_tier,
                    "apr_pct": _res.metrics.annualized_apr_pct,
                    "sharpe": _res.metrics.sharpe_ratio,
                    "max_dd_pct": _res.metrics.max_drawdown_pct,
                })
                if run_evolution and (not verdict.overall_pass):
                    cyc = orchestrator.run_cycle(cfg, next_version=f"{cfg.version}-n{started_ms % 100000}")
                    evolution_cycles.append({
                        "strategy_id": sid,
                        "cycle_id": cyc["cycle_id"],
                        "anomaly": cyc["feedback"]["anomaly_type"],
                        "promotion_stage": cyc["promotion_stage"],
                    })
            except Exception as exc:
                logger.warning("Nightly verification failed for %s: %s", sid, exc)
                regressions.append({"strategy_id": sid, "error": str(exc)})

        finished_ms = utc_now_ms()
        # Only ping watchdog with fresh real data timestamp (never fake wall clock on empty DB)
        summary_stats = store.summary_stats()
        freshest_ts = 0
        for bucket in ("funding", "candles_1h", "bbo_ticks"):
            for row in summary_stats.get(bucket, []):
                freshest_ts = max(freshest_ts, int(row.get("max_ts") or 0))
        if freshest_ts > 0 and (finished_ms - freshest_ts) <= 86_400_000:
            watchdog.max_stale_ms = 86_400_000
            watchdog.heartbeat("alphaloop_nightly_factory", ts_event_ms=freshest_ts, seq=int(finished_ms // 1000))

        report = {
            "started_ts_ms": started_ms,
            "finished_ts_ms": finished_ms,
            "duration_sec": round((finished_ms - started_ms) / 1000.0, 2),
            "backfill": backfill,
            "dq_summary": dq_summary,
            "regressions": regressions,
            "evolution_cycles": evolution_cycles,
            "watchdog": watchdog.stats(),
        }
        _atomic_write_json(root_dir / "nightly_factory_report.json", report)
        return report
    finally:
        watchdog.close()
        orchestrator.close()
        router.close()
        micro_store.close()
        cfg_store.close()
        ledger.close()
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AlphaLoop v2.2 Nightly Backfill & 6-Gate Factory Worker")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--days", type=int, default=3, help="Incremental backfill lookback days")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip live exchange API fetch (offline regression)")
    parser.add_argument("--evolve", action="store_true", help="Run 1 closed-loop Agent cycle on failing active strategies")
    args = parser.parse_args(argv)

    rep = run_nightly_factory(
        args.data_dir,
        skip_fetch=args.skip_fetch,
        backfill_days=args.days,
        run_evolution=args.evolve,
    )
    logger.info(
        "Nightly factory done in %.1fs | regressions=%d | evolutions=%d",
        rep["duration_sec"],
        len(rep["regressions"]),
        len(rep["evolution_cycles"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
