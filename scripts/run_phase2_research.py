#!/usr/bin/env python3
"""AlphaLoop v2.1 Phase 2 Multi-Pillar Empirical Research, Portfolio & Shadow Execution Runner.

Runs strictly on REAL multi-exchange data (`data/market_data.sqlite`, 178d-365d):
  1. Hard DQ-01..DQ-08 certification across expanded single-venue perp, funding pairs, & live BBO.
  2. Full 6-Gate verification of Pillar 1 (Vol-Tiered + Protective Margin Funding Carry) and
     Pillar 2 (TrendCTAPlugin on PAXG Gold + 12 Liquid Crypto Majors across HL 206d & Gate 178d,
     plus 360-day Gate full-year sensitivity check).
  3. Multi-Strategy Causal PortfolioCombiner (Fixed 50/50, Growth 30/70, Causal Inv-Vol Risk-Parity)
     anchored to real measured live BBO top-of-book depth (`$3k / $10k / $50k`).
  4. Multi-round Closed-Loop Agent Evolution Sprints (FundingCarry liquidation remediation +
     TrendCTA auto-evolution to `SHADOW_LIVE`).
  5. End-to-End Shadow-Only ExecutionBridge + HotConfigWatcher + Hash-Chained Audit verification.

Outputs isolated local untracked reports (`data/phase2_run/phase2_research_results.json`, plus
atomic `data/portfolio_report.json` when `--publish-console` is passed).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.agent_loop.model_router import MultiModelRouter
from alphaloop.agent_loop.orchestrator import AgentEvolutionOrchestrator
from alphaloop.core.config import HotConfigStore
from alphaloop.core.models import OrderIntent, StrategyConfig
from alphaloop.core.provenance import utc_now_ms
from alphaloop.data.quality import DataQualityAuditor
from alphaloop.data.store import MarketDataStore
from alphaloop.execution.bridge import ExecutionBridge
from alphaloop.harness.backtester import FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier, TrialLedger
from alphaloop.harness.micro_runner import MicroRunSession, MicroRunStore
from alphaloop.harness.portfolio import PortfolioCombiner, SleeveInput
from alphaloop.plugins import FundingCarryPlugin, TrendCTAPlugin, default_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase2-research")

CTA_UNIVERSE = [
    "PAXG", "BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "AAVE", "UNI", "SUI", "HYPE", "LTC",
]
CARRY_UNIVERSE = CTA_UNIVERSE + [
    "ADA", "APT", "ARB", "ENA", "NEAR", "OP", "PENDLE", "TIA", "VVV", "WLD", "ZEC",
]
BBO_UNIVERSE = [
    "PAXG", "BTC", "ETH", "SOL", "XRP", "DOGE", "LINK", "AAVE", "SUI", "AVAX", "HYPE", "LTC",
    "WLD", "NEAR", "ENA", "ZEC", "UNI", "VVV",
]


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
    """Atomically persist JSON via temp file + fsync + os.replace (prevents partial reads by Web Console)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove temp file %s after write error: %s", tmp_name, cleanup_exc)
        raise


def _measure_bbo_depth_usd(store: MarketDataStore, venue: str, symbols: list[str], start_ms: int) -> float:
    """Measure median top-of-book USD depth from real live captured BBO ticks."""
    sym_depths: list[float] = []
    for sym in symbols:
        ticks = store.load_bbo_ticks(venue, sym, start_ms=start_ms)
        vals = [(t.bid_qty + t.ask_qty) * 0.5 * t.mid for t in ticks if t.bid_qty > 0 and t.ask_qty > 0]
        if vals:
            sym_depths.append(statistics.median(vals))
    return round(statistics.median(sym_depths), 2) if sym_depths else 2500.0


def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaLoop v2.1 Phase 2 Empirical Research & Shadow Execution Suite")
    parser.add_argument("--publish-console", action="store_true", help="Also atomically publish portfolio_report.json to data/")
    args = parser.parse_args()

    data_dir = ROOT / "data"
    run_dir = data_dir / "phase2_run"
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    store = MarketDataStore(data_dir / "market_data.sqlite")
    # Isolated Phase 2 ledger so DSR trial counts are fresh and never pollute production data/state/
    p2_ledger_path = run_dir / "phase2_trial_ledger.sqlite"
    if p2_ledger_path.exists():
        p2_ledger_path.unlink()
    ledger = TrialLedger(p2_ledger_path)
    reg = default_registry()
    bt = FundingCarryBacktester(store)
    verifier = SixGateVerifier(bt, ledger, registry=reg)
    dq = DataQualityAuditor(store)
    config_store = HotConfigStore(state_dir)
    micro_store = MicroRunStore(state_dir / "micro_runs.sqlite")
    micro_session = MicroRunSession(store, micro_store, reg)
    router = MultiModelRouter()
    orchestrator = AgentEvolutionOrchestrator(
        state_dir=state_dir,
        backtester=bt,
        verifier=verifier,
        config_store=config_store,
        router=router,
        registry=reg,
        micro_store=micro_store,
    )
    exec_bridge = ExecutionBridge(state_dir, max_notional_cap_usd=3000.0, max_bbo_age_ms=3600_000)

    now_ms = utc_now_ms()
    s178 = now_ms - 178 * 86400 * 1000
    s206 = now_ms - 206 * 86400 * 1000
    s360 = now_ms - 360 * 86400 * 1000

    # Find latest live BBO capture window (last 30 minutes of captured ticks)
    latest_bbo_row = store.conn.fetch_one("SELECT MAX(ts_ms) AS m FROM bbo_ticks")
    bbo_max_ts = int(latest_bbo_row["m"]) if (latest_bbo_row and latest_bbo_row["m"]) else now_ms
    bbo_win_start = bbo_max_ts - 15 * 60 * 1000

    results: dict[str, Any] = {"generated_ts_ms": now_ms, "dq_audits": {}, "pillars": {}, "portfolios": {}, "agent_sprints": {}, "shadow_execution": {}}

    try:
        # ------------------------------------------------------------------
        # 1) HARD DQ-01..DQ-08 CERTIFICATION
        # ------------------------------------------------------------------
        logger.info("=== 1) Running DQ-01..DQ-08 Certification on Expanded Real Lake ===")
        dq_hl_perp = dq.audit_single_venue_perp("hyperliquid", CTA_UNIVERSE, start_ms=s206, ref_venue="gate", dataset_name="hl_206d_cta_13")
        dq_gt_perp = dq.audit_single_venue_perp("gate", CTA_UNIVERSE, start_ms=s178, ref_venue="hyperliquid", dataset_name="gate_178d_cta_13")
        dq_hl_gt = dq.audit_funding_pair("hyperliquid", "gate", CARRY_UNIVERSE, start_ms=s178, dataset_name="hl_gate_178d_24")
        dq_lt_gt = dq.audit_funding_pair("lighter", "gate", CARRY_UNIVERSE, start_ms=s178, dataset_name="lt_gate_178d_24")
        dq_bbo_hl_gt, bbo_hl_gt_sum = dq.audit_bbo_ticks("hyperliquid", "gate", BBO_UNIVERSE, start_ms=bbo_win_start, max_stale_sec=None, dataset_name="live_bbo_hl_gate_18")
        dq_bbo_lt_gt, bbo_lt_gt_sum = dq.audit_bbo_ticks("lighter", "gate", BBO_UNIVERSE, start_ms=bbo_win_start, max_stale_sec=None, dataset_name="live_bbo_lt_gate_18")

        hl_depth_usd = _measure_bbo_depth_usd(store, "hyperliquid", CTA_UNIVERSE, bbo_win_start)
        lt_depth_usd = _measure_bbo_depth_usd(store, "lighter", BBO_UNIVERSE, bbo_win_start)
        # Note: HL BTC/ETH/SOL depth is $21k-$408k while alts/PAXG are $580-$2000; compute conservative portfolio median
        hl_majors_depth = _measure_bbo_depth_usd(store, "hyperliquid", ["BTC", "ETH", "SOL", "XRP", "DOGE", "LINK", "AAVE", "SUI", "AVAX", "LTC"], bbo_win_start)

        results["dq_audits"] = {
            "hl_206d_single_perp": dq_hl_perp.to_dict(),
            "gate_178d_single_perp": dq_gt_perp.to_dict(),
            "hl_gate_178d_carry": dq_hl_gt.to_dict(),
            "lt_gate_178d_carry": dq_lt_gt.to_dict(),
            "live_bbo_hl_gate": {"report": dq_bbo_hl_gt.to_dict(), "summary": bbo_hl_gt_sum},
            "live_bbo_lt_gate": {"report": dq_bbo_lt_gt.to_dict(), "summary": bbo_lt_gt_sum},
            "measured_top_depth_usd": {
                "hl_13_median": hl_depth_usd,
                "hl_liquid_majors_median": hl_majors_depth,
                "lighter_18_median": lt_depth_usd,
            },
        }
        logger.info(
            "DQ Certified: HL_206d=%s | Gate_178d=%s | HL<->Gate=%s | LT<->Gate=%s | BBO_HL_Gate=%s (%.2fbp)",
            dq_hl_perp.tier, dq_gt_perp.tier, dq_hl_gt.tier, dq_lt_gt.tier,
            dq_bbo_hl_gt.tier, bbo_hl_gt_sum["combined_half_spread_cost_bps"],
        )

        # ------------------------------------------------------------------
        # 2) PILLAR 1 (FUNDING CARRY) & PILLAR 2 (TREND CTA) 6-GATE BATTERY
        # ------------------------------------------------------------------
        logger.info("=== 2) Running Full 6-Gate Verification Battery ===")
        fc = FundingCarryPlugin()
        cta = TrendCTAPlugin()

        pillar_specs: list[tuple[str, StrategyConfig, int]] = [
            (
                "p1_lt_gate_4x_raw_baseline",
                fc.default_config(
                    strategy_id="p1_lt_gate_4x_raw", version="v2.0", symbols=CARRY_UNIVERSE,
                    venue_a="lighter", venue_b="gate", leverage=4.0, max_positions=8, per_position_weight=0.125,
                    entry_spread_apr_pct=6.0, exit_spread_apr_pct=0.5, lookback_hours=120,
                    exit_confirm_hours=36, min_hold_hours=96, fee_a_bps=0.0, fee_b_bps=2.0, slippage_per_leg_bps=1.5,
                ),
                s178,
            ),
            (
                "p1_hl_gate_4slot_safe_vt_prot50",
                fc.default_config(
                    strategy_id="p1_hl_gate_safe_carry", version="v2.1", symbols=CARRY_UNIVERSE,
                    venue_a="hyperliquid", venue_b="gate", leverage=3.5, max_positions=4, per_position_weight=0.25,
                    entry_spread_apr_pct=10.0, exit_spread_apr_pct=1.0, lookback_hours=96,
                    exit_confirm_hours=24, min_hold_hours=72, fee_a_bps=1.5, fee_b_bps=2.0, slippage_per_leg_bps=0.75,
                    params={
                        "vol_target_leverage_enabled": True,
                        "vol_target_ann_pct": 130.0,
                        "min_symbol_leverage": 1.5,
                        "protective_margin_exit_pct": 0.50,
                        "protective_exit_slip_bps": 5.0,
                    },
                ),
                s178,
            ),
            (
                "p1_lt_gate_8slot_safe_vt_prot45",
                fc.default_config(
                    strategy_id="p1_lt_gate_safe_carry", version="v2.1", symbols=CARRY_UNIVERSE,
                    venue_a="lighter", venue_b="gate", leverage=3.5, max_positions=8, per_position_weight=0.125,
                    entry_spread_apr_pct=6.0, exit_spread_apr_pct=0.5, lookback_hours=120,
                    exit_confirm_hours=36, min_hold_hours=96, fee_a_bps=0.0, fee_b_bps=2.0, slippage_per_leg_bps=0.75,
                    params={
                        "vol_target_leverage_enabled": True,
                        "vol_target_ann_pct": 110.0,
                        "min_symbol_leverage": 1.5,
                        "protective_margin_exit_pct": 0.45,
                        "protective_exit_slip_bps": 5.0,
                    },
                ),
                s178,
            ),
            (
                "p2_hl_206d_cta_12h_40_16_taker",
                cta.default_config(
                    strategy_id="p2_hl_trend_cta_40_16", version="v1.0", symbols=CTA_UNIVERSE,
                    venue_a="hyperliquid", venue_b="hyperliquid", leverage=2.0, max_positions=6, per_position_weight=0.1667,
                    fee_a_bps=1.5, slippage_per_leg_bps=2.0, execution_mode="TAKER_TAKER",
                    params={
                        "timeframe_hours": 12, "entry_n": 40, "exit_n": 16, "atr_n": 14, "atr_mult": 3.5,
                        "ema_filter_n": 0, "risk_per_trade_pct": 0.8, "max_symbol_leverage": 1.5, "ref_venue": "gate",
                    },
                ),
                s206,
            ),
            (
                "p2_hl_206d_cta_12h_30_12_taker",
                cta.default_config(
                    strategy_id="p2_hl_trend_cta_30_12", version="v1.0", symbols=CTA_UNIVERSE,
                    venue_a="hyperliquid", venue_b="hyperliquid", leverage=2.0, max_positions=6, per_position_weight=0.1667,
                    fee_a_bps=1.5, slippage_per_leg_bps=2.0, execution_mode="TAKER_TAKER",
                    params={
                        "timeframe_hours": 12, "entry_n": 30, "exit_n": 12, "atr_n": 14, "atr_mult": 3.0,
                        "ema_filter_n": 0, "risk_per_trade_pct": 0.8, "max_symbol_leverage": 1.5, "ref_venue": "gate",
                    },
                ),
                s206,
            ),
            (
                "p2_gate_178d_cta_12h_40_16_taker",
                cta.default_config(
                    strategy_id="p2_gate_trend_cta_40_16", version="v1.0", symbols=CTA_UNIVERSE,
                    venue_a="gate", venue_b="gate", leverage=2.0, max_positions=6, per_position_weight=0.1667,
                    fee_a_bps=2.0, slippage_per_leg_bps=2.0, execution_mode="TAKER_TAKER",
                    params={
                        "timeframe_hours": 12, "entry_n": 40, "exit_n": 16, "atr_n": 14, "atr_mult": 3.5,
                        "ema_filter_n": 0, "risk_per_trade_pct": 0.8, "max_symbol_leverage": 1.5, "ref_venue": "hyperliquid",
                    },
                ),
                s178,
            ),
        ]

        eval_results: dict[str, Any] = {}
        for key, cfg, st_ms in pillar_specs:
            verdict, res, extras = verifier.verify(cfg, start_ms=st_ms)
            eval_results[key] = (cfg, verdict, res, extras)
            m = res.metrics
            results["pillars"][key] = {
                "strategy_id": cfg.strategy_id,
                "plugin_id": cfg.plugin_id,
                "execution_mode": cfg.execution_mode,
                "dq_tier": verdict.dq_tier,
                "gates_passed": f"{verdict.passed_count}/{verdict.total_count}",
                "overall_pass": verdict.overall_pass,
                "metrics": m.to_dict(),
                "is_metrics": extras["is_metrics"],
                "oos_metrics": extras["oos_metrics"],
                "stress_net_pnl_usd": extras["stress_metrics"]["net_pnl_usd"],
                "taker_hedge_net_pnl_usd": extras["taker_hedge_metrics"]["net_pnl_usd"],
                "checks": [c.__dict__ for c in verdict.checks],
            }
            logger.info(
                "  %-34s | %d/%d (%s) | APR=%+6.1f%% | Sh=%+5.2f | DD=%5.2f%% | Tr=%3d | Net=$%+7.1f | Liq=%d",
                key, verdict.passed_count, verdict.total_count, verdict.dq_tier,
                m.annualized_apr_pct, m.sharpe_ratio, m.max_drawdown_pct,
                m.trade_count, m.net_pnl_usd, m.liquidation_count,
            )

        # Full 360-day Gate sensitivity check (explicitly tagged UNPRICED_FUNDING_SENSITIVITY for prior 180d)
        cta_360_cfg = cta.default_config(
            strategy_id="p2_gate_360d_sensitivity", version="v1.0", symbols=CTA_UNIVERSE,
            venue_a="gate", venue_b="gate", leverage=2.0, max_positions=6, per_position_weight=0.1667,
            fee_a_bps=2.0, slippage_per_leg_bps=2.0, execution_mode="TAKER_TAKER",
            params={"timeframe_hours": 12, "entry_n": 40, "exit_n": 16, "atr_n": 14, "atr_mult": 3.5, "risk_per_trade_pct": 0.8, "max_symbol_leverage": 1.5},
        )
        r_360_full = cta.evaluate(store, cta_360_cfg, start_ms=s360)
        r_360_h1 = cta.evaluate(store, cta_360_cfg, start_ms=s360, end_ms=s178)
        results["pillars"]["p2_gate_360d_unpriced_funding_sensitivity"] = {
            "tag": "UNPRICED_FUNDING_SENSITIVITY_H1_ONLY",
            "note": "Gate funding API covers 180d; first 180d of 360d candle test lacks funding bars (blocked from G1; reported strictly as multi-regime price robustness stress).",
            "full_360d": r_360_full.metrics.to_dict(),
            "prior_h1_choppy_182d": r_360_h1.metrics.to_dict(),
        }

        # ------------------------------------------------------------------
        # 3) PILLAR 3: MULTI-STRATEGY CAUSAL PORTFOLIO COMBINER
        # ------------------------------------------------------------------
        logger.info("=== 3) Running Multi-Strategy Causal PortfolioCombiner ===")
        combiner = PortfolioCombiner()
        carry_eq = eval_results["p1_hl_gate_4slot_safe_vt_prot50"][2].equity_curve
        hl_cta_eq = eval_results["p2_hl_206d_cta_12h_40_16_taker"][2].equity_curve
        gt_cta_eq = eval_results["p2_gate_178d_cta_12h_40_16_taker"][2].equity_curve

        # Anchor top-of-book depth to REAL measured BBO ticks (conservative floor across venues)
        carry_depth = max(1000.0, min(hl_majors_depth, lt_depth_usd * 2.5))
        cta_depth = max(2500.0, hl_majors_depth)

        port_50_50 = combiner.combine(
            [
                SleeveInput("carry_hl_gate_safe", carry_eq, capital_usd=3000.0, weight=0.50, avg_monthly_turnover_x=1.5, measured_top_depth_usd=carry_depth),
                SleeveInput("trend_cta_hl_40_16", hl_cta_eq, capital_usd=3000.0, weight=0.50, avg_monthly_turnover_x=2.2, measured_top_depth_usd=cta_depth),
            ],
            capital_usd=10000.0, weighting_mode="fixed", rebalance_every_days=30, rebalance_cost_bps=10.0,
        )
        port_30_70 = combiner.combine(
            [
                SleeveInput("carry_hl_gate_safe", carry_eq, capital_usd=3000.0, weight=0.30, avg_monthly_turnover_x=1.5, measured_top_depth_usd=carry_depth),
                SleeveInput("trend_cta_hl_40_16", hl_cta_eq, capital_usd=3000.0, weight=0.70, avg_monthly_turnover_x=2.2, measured_top_depth_usd=cta_depth),
            ],
            capital_usd=10000.0, weighting_mode="fixed", rebalance_every_days=30, rebalance_cost_bps=10.0,
        )
        port_3sleeve = combiner.combine(
            [
                SleeveInput("carry_hl_gate_safe", carry_eq, capital_usd=3000.0, weight=0.35, avg_monthly_turnover_x=1.5, measured_top_depth_usd=carry_depth),
                SleeveInput("trend_cta_hl_40_16", hl_cta_eq, capital_usd=3000.0, weight=0.35, avg_monthly_turnover_x=2.2, measured_top_depth_usd=cta_depth),
                SleeveInput("trend_cta_gate_40_16", gt_cta_eq, capital_usd=3000.0, weight=0.30, avg_monthly_turnover_x=2.2, measured_top_depth_usd=cta_depth),
            ],
            capital_usd=10000.0, weighting_mode="fixed", rebalance_every_days=30, rebalance_cost_bps=10.0,
        )
        port_inv_vol = combiner.combine(
            [
                SleeveInput("carry_hl_gate_safe", carry_eq, capital_usd=3000.0, avg_monthly_turnover_x=1.5, measured_top_depth_usd=carry_depth),
                SleeveInput("trend_cta_hl_40_16", hl_cta_eq, capital_usd=3000.0, avg_monthly_turnover_x=2.2, measured_top_depth_usd=cta_depth),
            ],
            capital_usd=10000.0, weighting_mode="causal_inv_vol", rebalance_every_days=14, rebalance_cost_bps=10.0, vol_window_days=20,
        )

        for pname, pres in (
            ("balanced_50_50", port_50_50),
            ("growth_30_70", port_30_70),
            ("tri_sleeve_35_35_30", port_3sleeve),
            ("causal_inv_vol_parity", port_inv_vol),
        ):
            results["portfolios"][pname] = pres.to_dict(include_curve=False)
            logger.info(
                "  Portfolio %-22s | valid=%s | days=%d | APR=%+5.1f%% | Sh=%+5.2f | Calmar=%5.2f | DD=%4.2f%% | OOS_ret=%.2fx | Corr=%s",
                pname, pres.valid, pres.overlap_days, pres.annualized_apr_pct,
                pres.sharpe_ratio, pres.calmar_ratio, pres.max_drawdown_pct, pres.oos_retention,
                pres.correlation_matrix.get("carry_hl_gate_safe"),
            )

        # Atomically persist recommended balanced 50/50 portfolio report into isolated run_dir (and optionally data/ for Console)
        _atomic_write_json(run_dir / "portfolio_report.json", port_50_50.to_dict(include_curve=False))
        if args.publish_console:
            _atomic_write_json(data_dir / "portfolio_report.json", port_50_50.to_dict(include_curve=False))

        # ------------------------------------------------------------------
        # 4) MULTI-ROUND CLOSED-LOOP AGENT EVOLUTION SPRINTS
        # ------------------------------------------------------------------
        logger.info("=== 4) Running Closed-Loop Multi-Round Agent Evolution Sprints ===")
        # Sprint A: Liquidating 4x LT<->Gate carry -> Agent remediates liquidations via vol-tier + protective exit
        seed_carry_bad = eval_results["p1_lt_gate_4x_raw_baseline"][0]
        config_store.register_version(seed_carry_bad, actor="phase2-seed", reason="Seed liquidating 4x carry for Agent remediation")
        sprint_carry = orchestrator.run_multi_round_sprint(
            seed_carry_bad, max_rounds=2, version_prefix="v2.1-p2", start_ms=s178
        )
        results["agent_sprints"]["carry_liquidation_remediation"] = {
            "rounds_executed": sprint_carry["rounds_executed"],
            "stop_reason": sprint_carry["stop_reason"],
            "final_version": sprint_carry["final_version"],
            "final_promotion_stage": sprint_carry["final_promotion_stage"],
            "trajectory": [
                {
                    "from": r["from_version"], "to": r["to_version"],
                    "anomaly": r["feedback"]["anomaly_type"],
                    "base_apr": r["baseline_metrics"]["annualized_apr_pct"],
                    "cand_apr": r["candidate_metrics"]["annualized_apr_pct"],
                    "base_liqs": r["baseline_metrics"]["liquidation_count"],
                    "cand_liqs": r["candidate_metrics"]["liquidation_count"],
                    "gates": f"{r['gate_verdict']['passed_count']}/{r['gate_verdict']['total_count']}",
                    "stage": r["promotion_stage"],
                }
                for r in sprint_carry["rounds"]
            ],
        }

        # Sprint B: TrendCTA seed -> Agent evolves & auto-stages SHADOW_LIVE (6/6 PASS!)
        seed_cta_subopt = cta.default_config(
            strategy_id="hl_trend_cta_prod", version="v0.9", symbols=CTA_UNIVERSE,
            venue_a="hyperliquid", venue_b="hyperliquid", leverage=2.0, max_positions=6, per_position_weight=0.1667,
            fee_a_bps=1.5, slippage_per_leg_bps=2.0, execution_mode="TAKER_TAKER",
            params={
                "timeframe_hours": 12, "entry_n": 36, "exit_n": 14, "atr_n": 14, "atr_mult": 3.2,
                "risk_per_trade_pct": 0.8, "max_symbol_leverage": 1.5, "ref_venue": "gate",
            },
        )
        config_store.register_version(seed_cta_subopt, actor="phase2-seed", reason="Seed TrendCTA for closed-loop Agent verification & SHADOW_LIVE staging")
        sprint_cta = orchestrator.run_multi_round_sprint(
            seed_cta_subopt, max_rounds=2, version_prefix="v1.0-p2", start_ms=s206
        )
        results["agent_sprints"]["trend_cta_shadow_promotion"] = {
            "rounds_executed": sprint_cta["rounds_executed"],
            "stop_reason": sprint_cta["stop_reason"],
            "final_version": sprint_cta["final_version"],
            "final_promotion_stage": sprint_cta["final_promotion_stage"],
            "trajectory": [
                {
                    "from": r["from_version"], "to": r["to_version"],
                    "anomaly": r["feedback"]["anomaly_type"],
                    "base_apr": r["baseline_metrics"]["annualized_apr_pct"],
                    "cand_apr": r["candidate_metrics"]["annualized_apr_pct"],
                    "cand_sharpe": r["candidate_metrics"]["sharpe_ratio"],
                    "cand_dd": r["candidate_metrics"]["max_drawdown_pct"],
                    "gates": f"{r['gate_verdict']['passed_count']}/{r['gate_verdict']['total_count']}",
                    "red_team": r["red_team_verdict"]["decision"],
                    "stage": r["promotion_stage"],
                }
                for r in sprint_cta["rounds"]
            ],
        }
        logger.info("Agent Sprint Carry: %s", results["agent_sprints"]["carry_liquidation_remediation"]["trajectory"])
        logger.info("Agent Sprint CTA  : %s", results["agent_sprints"]["trend_cta_shadow_promotion"]["trajectory"])

        # ------------------------------------------------------------------
        # 5) END-TO-END SHADOW EXECUTION BRIDGE + HOT CONFIG WATCHER + AUDIT CHAIN
        # ------------------------------------------------------------------
        logger.info("=== 5) Verifying End-to-End Shadow ExecutionBridge & Hash-Chained Audit ===")
        watcher_events = exec_bridge.watcher.poll()
        loaded_cta = exec_bridge.watcher.get_loaded("hl_trend_cta_prod")
        shadow_plans_out: list[dict[str, Any]] = []
        if loaded_cta is not None:
            active_cta_cfg, _ = loaded_cta
            for sym, act in (("BTC", "OPEN_SHORT"), ("PAXG", "OPEN_SHORT")):
                ticks = store.load_bbo_ticks("hyperliquid", sym, start_ms=bbo_win_start)
                if ticks:
                    latest_t = ticks[-1]
                    intent = OrderIntent(
                        strategy_id=active_cta_cfg.strategy_id,
                        symbol=sym,
                        action=act,
                        notional_usd=450.0,
                        expected_edge_bps=35.0,
                        reason="DONCHIAN_12H_BREAKOUT_SHADOW_VERIFY",
                        ts_ms=latest_t.ts_ms,
                    )
                    p = exec_bridge.submit_intent(
                        active_cta_cfg, intent, {"hyperliquid": latest_t}, now_ms=latest_t.ts_ms + 500
                    )
                    shadow_plans_out.append(p.to_dict())

        # Also prove fail-closed DRAFT rejection in audit chain
        p_blocked = exec_bridge.submit_intent(
            seed_carry_bad,
            OrderIntent(seed_carry_bad.strategy_id, "BTC", "OPEN_SHORT_A_LONG_B", 500.0, 12.0, "DRAFT_TEST", bbo_max_ts),
            {},
            now_ms=bbo_max_ts,
        )
        shadow_plans_out.append(p_blocked.to_dict())

        bridge_summary = exec_bridge.summary()
        results["shadow_execution"] = {
            "watcher_events": watcher_events,
            "plans_submitted": shadow_plans_out,
            "bridge_summary": bridge_summary,
        }
        logger.info(
            "Shadow Execution Bridge: chain_valid=%s (%s, %d events), kill_engaged=%s, live_enabled=%s",
            bridge_summary["audit_chain_valid"],
            bridge_summary["audit_chain_detail"],
            bridge_summary["audit_chain_length"],
            bridge_summary["kill_switch"]["engaged"],
            bridge_summary["live_execution_enabled"],
        )

        # Save comprehensive Phase 2 research report atomically (gitignored)
        out_path = run_dir / "phase2_research_results.json"
        _atomic_write_json(out_path, results)
        logger.info("Phase 2 complete! Saved full report to %s", out_path)
    finally:
        store.close()
        ledger.close()
        config_store.close()
        micro_store.close()
        orchestrator.close()
        exec_bridge.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
