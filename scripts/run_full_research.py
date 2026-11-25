#!/usr/bin/env python3
"""End-to-end 90-day REAL market data ingestion + multi-venue funding arbitrage research + Agent evolution demo."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.agent_loop.model_router import MultiModelRouter
from alphaloop.agent_loop.orchestrator import AgentEvolutionOrchestrator
from alphaloop.core.config import HotConfigStore
from alphaloop.core.models import StrategyConfig
from alphaloop.core.provenance import utc_iso, utc_now_ms
from alphaloop.data.collectors import GateCollector, HyperliquidCollector, LighterCollector, OKXCollector
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier, TrialLedger
from alphaloop.harness.micro_runner import MicroRunSession, MicroRunStore
from alphaloop.plugins import BasisReversionPlugin, StructuralArbPlugin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("research")

# 24 liquid tokens spanning majors, DeFi blue chips, L1/L2s, and high-OI perp venues
UNIVERSE = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "AAVE",
    "UNI", "SUI", "APT", "ARB", "OP", "NEAR", "ADA", "LTC",
    "HYPE", "ZEC", "PENDLE", "ENA", "WLD", "TIA", "INJ", "VVV",
]


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=".research.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def fetch_dataset(store: MarketDataStore, days: int = 90) -> dict[str, dict[str, int]]:
    hl = HyperliquidCollector()
    lt = LighterCollector()
    okx = OKXCollector()
    gate = GateCollector()

    cached_syms = {
        r["symbol"]
        for r in store.conn.execute(
            "SELECT symbol, COUNT(*) AS n FROM funding_rates WHERE venue='hyperliquid' GROUP BY symbol HAVING n >= 1000"
        ).fetchall()
    }
    missing = [s for s in UNIVERSE if s not in cached_syms]
    if not missing:
        logger.info("All %d symbols cached in %s; skipping fetch.", len(UNIVERSE), store.db_path)
        return {"cached_symbols": {"count": len(cached_syms)}}

    logger.info("Resuming fetch: %d cached, %d remaining (%s)", len(cached_syms), len(missing), ",".join(missing))

    try:
        hl_live = hl.fetch_live_universe()
        store.save_live_snapshot("hyperliquid", hl_live)
    except Exception as exc:
        logger.warning("Failed live HL snapshot: %s", exc)
    try:
        lt_live = lt.fetch_live_funding_all()
        store.save_live_snapshot("lighter_multi", lt_live)
    except Exception as exc:
        logger.warning("Failed live Lighter snapshot: %s", exc)

    report: dict[str, dict[str, int]] = {}
    for sym in missing:
        counts: dict[str, int] = {}
        try:
            counts["hl_fr"] = store.upsert_funding(hl.fetch_funding_history(sym, days=days))
            counts["hl_1h"] = store.upsert_candles(hl.fetch_candles_1h(sym, days=days))
        except Exception as exc:
            logger.warning("HL %s error: %s", sym, exc)

        try:
            counts["lt_fr"] = store.upsert_funding(lt.fetch_funding_history(sym, days=days))
        except Exception as exc:
            logger.warning("Lighter %s error: %s", sym, exc)

        try:
            okx_fr = okx.fetch_funding_history(sym, days=days)
            counts["okx_fr"] = store.upsert_funding(okx_fr)
            if okx_fr:
                counts["okx_swap_1h"] = store.upsert_candles(okx.fetch_candles_1h(sym, days=days, spot=False))
                counts["okx_spot_1h"] = store.upsert_candles(okx.fetch_candles_1h(sym, days=days, spot=True))
        except Exception as exc:
            logger.warning("OKX %s error: %s", sym, exc)

        try:
            gate_fr = gate.fetch_funding_history(sym, days=days)
            counts["gate_fr"] = store.upsert_funding(gate_fr)
            if gate_fr:
                counts["gate_1h"] = store.upsert_candles(gate.fetch_candles_1h(sym, days=days))
        except Exception as exc:
            logger.warning("Gate %s error: %s", sym, exc)

        report[sym] = counts
        logger.info("Fetched %-6s -> %s", sym, counts)
    return report


def run_experiments(data_dir: Path) -> dict:
    store = MarketDataStore(data_dir / "market_data.sqlite")
    fetch_report = fetch_dataset(store, days=90)
    db_summary = store.summary_stats()

    # Isolate batch research ledger/state under data/research_run/ (NEVER wipe production trial_ledger.sqlite!)
    run_dir = data_dir / "research_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    ledger = TrialLedger(run_dir / "trial_ledger.sqlite")
    config_store = HotConfigStore(state_dir)
    micro_store = MicroRunStore(state_dir / "micro_runs.sqlite")
    backtester = FundingCarryBacktester(store)
    verifier = SixGateVerifier(backtester, ledger)
    router = MultiModelRouter()
    orchestrator = AgentEvolutionOrchestrator(
        state_dir, backtester, verifier, config_store, router, micro_store=micro_store
    )

    hl_btc = store.load_funding("hyperliquid", "BTC")
    lt_btc = store.load_funding("lighter", "BTC")
    okx_btc = store.load_funding("okx", "BTC")
    gate_btc = store.load_funding("gate", "BTC")

    common_start = max(
        hl_btc[0].ts_ms if hl_btc else 0,
        lt_btc[0].ts_ms if lt_btc else 0,
        okx_btc[0].ts_ms if okx_btc else 0,
        gate_btc[0].ts_ms if gate_btc else 0,
    )
    common_end = min(
        hl_btc[-1].ts_ms if hl_btc else utc_now_ms(),
        lt_btc[-1].ts_ms if lt_btc else utc_now_ms(),
        okx_btc[-1].ts_ms if okx_btc else utc_now_ms(),
        gate_btc[-1].ts_ms if gate_btc else utc_now_ms(),
    )
    logger.info(
        "Common 4-venue overlap window: %s -> %s (%.1f days)",
        utc_iso(common_start),
        utc_iso(common_end),
        (common_end - common_start) / (86400 * 1000),
    )

    def valid_syms(va: str, vb: str | None) -> list[str]:
        out = []
        for s in UNIVERSE:
            fa = store.load_funding(va, s, common_start, common_end)
            if len(fa) < 100:
                continue
            if vb is not None:
                fb = store.load_funding(vb, s, common_start, common_end)
                if len(fb) < 100:
                    continue
            out.append(s)
        return out

    syms_hl_lt = valid_syms("hyperliquid", "lighter")
    syms_lt_gate = valid_syms("lighter", "gate")
    syms_lt_okx = valid_syms("lighter", "okx")
    syms_hl_gate = valid_syms("hyperliquid", "gate")
    syms_hl_okx = valid_syms("hyperliquid", "okx")
    syms_okx = valid_syms("okx", None)
    logger.info(
        "Valid overlapping universes: LT-Gate=%d, LT-OKX=%d, HL-Gate=%d, HL-OKX=%d, HL-LT=%d, OKX-SpotPerp=%d",
        len(syms_lt_gate), len(syms_lt_okx), len(syms_hl_gate), len(syms_hl_okx), len(syms_hl_lt), len(syms_okx),
    )

    configs: list[StrategyConfig] = [
        # EXP 1: Forensic Reproduction of ATB H6 Buggy Spot-Perp Rules
        StrategyConfig(
            strategy_id="exp1_atb_h6_buggy_repro",
            version="v0.1-buggy",
            family="spot_perp_forensic",
            status="DRAFT",
            venue_a="okx",
            venue_b="okx_spot",
            mode="spot_perp",
            symbols=syms_okx,
            capital_usd=3000.0,
            leverage=1.0,
            max_positions=3,
            per_position_weight=0.33,
            entry_spread_apr_pct=10.95,
            exit_spread_apr_pct=54.75,
            lookback_hours=72,
            fee_a_bps=5.0,
            fee_b_bps=10.0,
            slippage_per_leg_bps=1.0,
            rebalance_drag_bps_per_month=0.0,
            allow_bidirectional=False,
            use_atb_buggy_rules=True,
            notes="Forensic replay of ATB H6 inverted entry/exit hysteresis bug",
        ),
        # EXP 2: Fixed Single-Venue Spot-Perp Carry (OKX real Spot + Swap dual candles)
        StrategyConfig(
            strategy_id="exp2_okx_spot_perp_fixed",
            version="v1.0",
            family="spot_perp_carry",
            status="DRAFT",
            venue_a="okx",
            venue_b="okx_spot",
            mode="spot_perp",
            symbols=syms_okx,
            capital_usd=3000.0,
            leverage=1.0,
            max_positions=3,
            per_position_weight=0.33,
            entry_spread_apr_pct=18.0,
            exit_spread_apr_pct=4.0,
            lookback_hours=48,
            exit_confirm_hours=24,
            min_hold_hours=48,
            fee_a_bps=2.0,
            fee_b_bps=8.0,
            slippage_per_leg_bps=1.5,
            rebalance_drag_bps_per_month=0.0,
            allow_bidirectional=False,
            notes="Single-venue OKX Spot-Perp with fixed hysteresis (long-only positive funding)",
        ),
        # EXP 3: DEX-vs-DEX Negative Control (Hyperliquid <-> Lighter: baselines cancel out!)
        StrategyConfig(
            strategy_id="exp3_hl_lighter_baseline_cancel",
            version="v1.0",
            family="dex_dex_control",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="lighter",
            mode="perp_perp",
            symbols=syms_hl_lt,
            capital_usd=3000.0,
            leverage=2.0,
            max_positions=8,
            per_position_weight=0.125,
            entry_spread_apr_pct=8.0,
            exit_spread_apr_pct=0.5,
            lookback_hours=120,
            exit_confirm_hours=36,
            min_hold_hours=96,
            fee_a_bps=1.5,
            fee_b_bps=0.0,
            slippage_per_leg_bps=1.5,
            rebalance_drag_bps_per_month=3.0,
            allow_bidirectional=True,
            notes="Negative control: both HL & Lighter carry +10.5-10.95% baseline so spread cancels out",
        ),
        # EXP 4D (SAFE 2.0x/leg): Lighter (0% fee) <-> Gate (2bp maker) @ 2.0x per-leg leverage
        StrategyConfig(
            strategy_id="exp4d_lt_gate_safe_2x",
            version="v2.3",
            family="dex_cex_lt_gate_safe",
            status="DRAFT",
            venue_a="lighter",
            venue_b="gate",
            mode="perp_perp",
            symbols=syms_lt_gate,
            capital_usd=3000.0,
            leverage=2.0,
            max_positions=8,
            per_position_weight=0.125,
            entry_spread_apr_pct=6.0,
            exit_spread_apr_pct=0.5,
            lookback_hours=120,
            exit_confirm_hours=36,
            min_hold_hours=96,
            fee_a_bps=0.0,
            fee_b_bps=2.0,
            slippage_per_leg_bps=1.5,
            rebalance_drag_bps_per_month=3.0,
            allow_bidirectional=True,
            notes="Safe 2.0x per-leg isolated leverage (45% adverse buffer): Lighter 0% + Gate 2bp maker",
        ),
        # EXP 4E (MODERATE 3.0x/leg): Lighter <-> Gate @ 3.0x per-leg leverage
        StrategyConfig(
            strategy_id="exp4e_lt_gate_safe_3x",
            version="v2.3",
            family="dex_cex_lt_gate_3x",
            status="DRAFT",
            venue_a="lighter",
            venue_b="gate",
            mode="perp_perp",
            symbols=syms_lt_gate,
            capital_usd=3000.0,
            leverage=3.0,
            max_positions=8,
            per_position_weight=0.125,
            entry_spread_apr_pct=6.0,
            exit_spread_apr_pct=0.5,
            lookback_hours=120,
            exit_confirm_hours=36,
            min_hold_hours=96,
            fee_a_bps=0.0,
            fee_b_bps=2.0,
            slippage_per_leg_bps=1.5,
            rebalance_drag_bps_per_month=3.0,
            allow_bidirectional=True,
            notes="3.0x per-leg isolated leverage (30% adverse buffer): Lighter 0% + Gate 2bp maker",
        ),
        # EXP 4A (HIGH-LEVERAGE STRESS 4.0x/leg): Exposes real altcoin volatility margin liquidation risk
        StrategyConfig(
            strategy_id="exp4a_lt_gate_maker_4x",
            version="v2.3",
            family="dex_cex_lt_gate",
            status="DRAFT",
            venue_a="lighter",
            venue_b="gate",
            mode="perp_perp",
            symbols=syms_lt_gate,
            capital_usd=3000.0,
            leverage=4.0,
            max_positions=8,
            per_position_weight=0.125,
            entry_spread_apr_pct=6.0,
            exit_spread_apr_pct=0.5,
            lookback_hours=120,
            exit_confirm_hours=36,
            min_hold_hours=96,
            fee_a_bps=0.0,
            fee_b_bps=2.0,
            slippage_per_leg_bps=1.5,
            rebalance_drag_bps_per_month=3.0,
            allow_bidirectional=True,
            notes="4.0x per-leg isolated leverage stress test (22.5% liquidation trigger; tests real altcoin moves)",
        ),
        # EXP 4B (HIGH-LEVERAGE STRESS 5.0x/leg)
        StrategyConfig(
            strategy_id="exp4b_lt_gate_maker_5x",
            version="v2.3",
            family="dex_cex_lt_gate_5x",
            status="DRAFT",
            venue_a="lighter",
            venue_b="gate",
            mode="perp_perp",
            symbols=syms_lt_gate,
            capital_usd=3000.0,
            leverage=5.0,
            max_positions=8,
            per_position_weight=0.125,
            entry_spread_apr_pct=6.0,
            exit_spread_apr_pct=0.5,
            lookback_hours=120,
            exit_confirm_hours=36,
            min_hold_hours=96,
            fee_a_bps=0.0,
            fee_b_bps=2.0,
            slippage_per_leg_bps=1.5,
            rebalance_drag_bps_per_month=3.0,
            allow_bidirectional=True,
            notes="5.0x per-leg isolated leverage stress test (18% liquidation trigger)",
        ),
        # EXP 5B (SAFE 2.0x/leg): Lighter (0% fee) <-> OKX (2bp maker) @ 2.0x per-leg leverage
        StrategyConfig(
            strategy_id="exp5b_lt_okx_safe_2x",
            version="v2.3",
            family="dex_cex_lt_okx_safe",
            status="DRAFT",
            venue_a="lighter",
            venue_b="okx",
            mode="perp_perp",
            symbols=syms_lt_okx,
            capital_usd=3000.0,
            leverage=2.0,
            max_positions=8,
            per_position_weight=0.125,
            entry_spread_apr_pct=6.0,
            exit_spread_apr_pct=0.5,
            lookback_hours=120,
            exit_confirm_hours=36,
            min_hold_hours=96,
            fee_a_bps=0.0,
            fee_b_bps=2.0,
            slippage_per_leg_bps=1.5,
            rebalance_drag_bps_per_month=3.0,
            allow_bidirectional=True,
            notes="Safe 2.0x per-leg isolated leverage: Lighter 0% + OKX 2bp maker",
        ),
        # EXP 6A (SAFE 2.0x/leg): Dual-Candle Real Basis Verification — Hyperliquid <-> Gate
        StrategyConfig(
            strategy_id="exp6a_hl_gate_safe_2x_realbasis",
            version="v2.3",
            family="dex_cex_hl_gate",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="gate",
            mode="perp_perp",
            symbols=syms_hl_gate,
            capital_usd=3000.0,
            leverage=2.0,
            max_positions=8,
            per_position_weight=0.125,
            entry_spread_apr_pct=6.0,
            exit_spread_apr_pct=0.5,
            lookback_hours=120,
            exit_confirm_hours=36,
            min_hold_hours=96,
            fee_a_bps=1.5,
            fee_b_bps=2.0,
            slippage_per_leg_bps=1.5,
            rebalance_drag_bps_per_month=3.0,
            allow_bidirectional=True,
            notes="100% real dual-venue 1h basis accounting verification at safe 2.0x/leg: HL <-> Gate",
        ),
        # EXP 6B (SAFE 2.0x/leg): Dual-Candle Real Basis Verification — Hyperliquid <-> OKX
        StrategyConfig(
            strategy_id="exp6b_hl_okx_safe_2x_realbasis",
            version="v2.3",
            family="dex_cex_hl_okx",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="okx",
            mode="perp_perp",
            symbols=syms_hl_okx,
            capital_usd=3000.0,
            leverage=2.0,
            max_positions=8,
            per_position_weight=0.125,
            entry_spread_apr_pct=6.0,
            exit_spread_apr_pct=0.5,
            lookback_hours=120,
            exit_confirm_hours=36,
            min_hold_hours=96,
            fee_a_bps=1.5,
            fee_b_bps=2.0,
            slippage_per_leg_bps=1.5,
            rebalance_drag_bps_per_month=3.0,
            allow_bidirectional=True,
            notes="100% real dual-venue 1h basis accounting verification at safe 2.0x/leg: HL <-> OKX",
        ),
    ]

    def daily_series(eq_curve: list[tuple[int, float]]) -> list[dict]:
        by_day: dict[str, float] = {}
        for ts, eq in eq_curve:
            day = utc_iso(ts)[:10]
            by_day[day] = round(eq, 2)
        return [{"date": d, "equity": by_day[d]} for d in sorted(by_day)]

    experiment_outputs: dict[str, dict] = {}
    curves_by_id: dict[str, list[tuple[int, float]]] = {}
    for cfg in configs:
        verdict, res, extras = verifier.verify(cfg, start_ms=common_start, end_ms=common_end)
        m = res.metrics
        curves_by_id[cfg.strategy_id] = res.equity_curve
        if verdict.overall_pass:
            cfg.status = "BACKTESTED"
            config_store.register_version(cfg, actor="research-harness", reason=f"6-Gate PASS ({verdict.passed_count}/6)")
        experiment_outputs[cfg.strategy_id] = {
            "config": cfg.to_dict(),
            "metrics": m.to_dict(),
            "basis_coverage_pct": res.basis_coverage_pct,
            "warnings": res.warnings,
            "verdict": verdict.to_dict(),
            "dq_report": extras["dq_report"],
            "stress_metrics": extras["stress_metrics"],
            "taker_hedge_metrics": extras["taker_hedge_metrics"],
            "is_metrics": extras["is_metrics"],
            "oos_metrics": extras["oos_metrics"],
            "plateau_grid": extras["plateau_grid"],
            "daily_equity": daily_series(res.equity_curve),
            "sample_trades": [t.to_dict() for t in res.trades[:12]],
        }
        logger.info(
            "%-32s | trades=%3d | gross=$%+7.2f | basis=$%+6.2f | cost=$%6.2f | NET=$%+7.2f | APR=%+6.2f%% | Sharpe=%5.2f | DD=%4.2f%% | Liq=%d | 6-Gate=%d/6 %s",
            cfg.strategy_id,
            m.trade_count,
            m.gross_funding_usd,
            m.basis_pnl_usd,
            m.total_fees_usd + m.total_slippage_usd + m.total_rebalance_usd,
            m.net_pnl_usd,
            m.annualized_apr_pct,
            m.sharpe_ratio,
            m.max_drawdown_pct,
            m.liquidation_count,
            verdict.passed_count,
            "PASS" if verdict.overall_pass else "FAIL",
        )

    # EXP 7: Closed-Loop Agent Evolution Demo on REAL 90-day Lighter <-> Gate data (safe 2.0x/leg)
    buggy_seed = StrategyConfig(
        strategy_id="lt_gate_prod_candidate",
        version="v0.1-churn",
        family="prod_lt_gate_evolution",
        status="DRAFT",
        venue_a="lighter",
        venue_b="gate",
        mode="perp_perp",
        symbols=syms_lt_gate,
        capital_usd=3000.0,
        leverage=2.0,
        max_positions=8,
        per_position_weight=0.125,
        entry_spread_apr_pct=10.0,
        exit_spread_apr_pct=9.0,
        lookback_hours=12,
        exit_confirm_hours=2,
        min_hold_hours=4,
        fee_a_bps=0.0,
        fee_b_bps=2.0,
        slippage_per_leg_bps=1.5,
        rebalance_drag_bps_per_month=3.0,
        allow_bidirectional=True,
        use_atb_buggy_rules=True,
        notes="Buggy ATB-style churn seed before closed-loop Agent Evolution",
    )
    config_store.register_version(buggy_seed, actor="seed", reason="Initial buggy seed for live closed-loop Agent demo")
    agent_cycle_1 = orchestrator.run_cycle(buggy_seed, next_version="v0.2-agent", start_ms=common_start, end_ms=common_end)
    logger.info(
        "AGENT CYCLE 1 (REAL 90D): anomaly=%s | APR %+.1f%% -> %+.2f%% | gates=%d/%d | red_team=%s | stage=%s",
        agent_cycle_1["feedback"]["anomaly_type"],
        agent_cycle_1["baseline_metrics"]["annualized_apr_pct"],
        agent_cycle_1["candidate_metrics"]["annualized_apr_pct"],
        agent_cycle_1["gate_verdict"]["passed_count"],
        agent_cycle_1["gate_verdict"]["total_count"],
        agent_cycle_1["red_team_verdict"]["decision"],
        agent_cycle_1["promotion_stage"],
    )

    # Blended Custody-Diversified 3-Pair Safe 2.0x/leg Portfolio (50% LT-Gate 2x + 25% LT-OKX 2x + 25% HL-OKX 2x)
    c_ltg = dict(curves_by_id["exp4d_lt_gate_safe_2x"])
    c_lto = dict(curves_by_id["exp5b_lt_okx_safe_2x"])
    c_hlo = dict(curves_by_id["exp6b_hl_okx_safe_2x_realbasis"])
    common_ts_blend = sorted(set(c_ltg) & set(c_lto) & set(c_hlo))
    blend_curve = [(ts, 0.50 * c_ltg[ts] + 0.25 * c_lto[ts] + 0.25 * c_hlo[ts]) for ts in common_ts_blend]
    b_start, b_end = blend_curve[0][1], blend_curve[-1][1]
    b_days = (blend_curve[-1][0] - blend_curve[0][0]) / (86400 * 1000)
    b_ret_pct = (b_end / b_start - 1.0) * 100.0
    b_apr = b_ret_pct * (365.0 / b_days)
    pk, b_mdd = b_start, 0.0
    for _, eq in blend_curve:
        if eq > pk:
            pk = eq
        dd = (pk - eq) / pk * 100.0
        if dd > b_mdd:
            b_mdd = dd
    sp = int(len(blend_curve) * 0.70)
    b_is_apr = ((blend_curve[sp - 1][1] / blend_curve[0][1]) - 1.0) * 100.0 * (365.0 / ((blend_curve[sp - 1][0] - blend_curve[0][0]) / (86400 * 1000)))
    b_oos_apr = ((blend_curve[-1][1] / blend_curve[sp - 1][1]) - 1.0) * 100.0 * (365.0 / ((blend_curve[-1][0] - blend_curve[sp - 1][0]) / (86400 * 1000)))
    blended_portfolio = {
        "weights": {"exp4d_lt_gate_safe_2x": 0.50, "exp5b_lt_okx_safe_2x": 0.25, "exp6b_hl_okx_safe_2x_realbasis": 0.25},
        "net_pnl_usd_on_3k": round(b_end - b_start, 2),
        "annualized_apr_pct": round(b_apr, 2),
        "is_apr_pct": round(b_is_apr, 2),
        "oos_apr_pct": round(b_oos_apr, 2),
        "max_dd_pct": round(b_mdd, 3),
        "daily_equity": daily_series(blend_curve),
    }
    logger.info(
        "BLENDED 3-PAIR SAFE-2X PORTFOLIO: APR=%+.2f%% (IS %+.1f%% / OOS %+.1f%%) | MaxDD=%.3f%% | Net on $3k=$%+.2f",
        b_apr, b_is_apr, b_oos_apr, b_mdd, b_end - b_start,
    )

    # EXP 8 & 9: Evaluate Strategy Factory v2 Heterogeneous Plugins with Red-Team-hardened T+1 execution lag!
    br_plug = BasisReversionPlugin()
    sa_plug = StructuralArbPlugin()
    v2_plugin_cfgs = [
        br_plug.default_config(
            strategy_id="exp8a_basis_reversion_hl_okx",
            version="v2.0-t1lag",
            symbols=syms_hl_okx,
            venue_a="hyperliquid",
            venue_b="okx",
            params={"entry_zscore": 2.4, "exit_zscore": 0.25, "min_basis_bps": 14.0, "z_lookback_hours": 48, "max_hold_hours": 36, "stop_basis_bps": 120.0, "execution_lag_bars": 1},
        ),
        br_plug.default_config(
            strategy_id="exp8b_basis_reversion_hl_gate",
            version="v2.0-t1lag",
            symbols=syms_hl_gate,
            venue_a="hyperliquid",
            venue_b="gate",
            params={"entry_zscore": 2.4, "exit_zscore": 0.25, "min_basis_bps": 14.0, "z_lookback_hours": 48, "max_hold_hours": 36, "stop_basis_bps": 120.0, "execution_lag_bars": 1},
        ),
        sa_plug.default_config(
            strategy_id="exp9_structural_arb_hl_okx",
            version="v2.0-causal",
            symbols=syms_hl_okx,
            venue_a="hyperliquid",
            venue_b="okx",
            params={"min_edge_bps": 22.0, "confirm_decay_bps": 4.0, "rebalance_cost_bps": 6.0, "gas_usd_per_tx": 0.40, "revert_rate_pct": 8.0, "notional_per_trade_usd": 500.0},
        ),
    ]
    for cfg in v2_plugin_cfgs:
        verdict, res, extras = verifier.verify(cfg, start_ms=common_start, end_ms=common_end)
        m = res.metrics
        experiment_outputs[cfg.strategy_id] = {
            "config": cfg.to_dict(),
            "metrics": m.to_dict(),
            "basis_coverage_pct": res.basis_coverage_pct,
            "warnings": res.warnings,
            "verdict": verdict.to_dict(),
            "dq_report": extras["dq_report"],
            "stress_metrics": extras["stress_metrics"],
            "taker_hedge_metrics": extras["taker_hedge_metrics"],
            "is_metrics": extras["is_metrics"],
            "oos_metrics": extras["oos_metrics"],
            "plateau_grid": extras["plateau_grid"],
            "daily_equity": daily_series(res.equity_curve),
            "sample_trades": [t.to_dict() for t in res.trades[:12]],
        }
        logger.info(
            "%-32s | plugin=%-15s | trades=%3d | gross=$%+7.2f | basis=$%+6.2f | cost=$%6.2f | NET=$%+7.2f | APR=%+6.2f%% | Liq=%d | 6-Gate=%d/6 %s",
            cfg.strategy_id, cfg.plugin_id, m.trade_count, m.gross_funding_usd, m.basis_pnl_usd,
            m.total_fees_usd + m.total_slippage_usd + m.total_rebalance_usd,
            m.net_pnl_usd, m.annualized_apr_pct, m.liquidation_count,
            verdict.passed_count, "PASS" if verdict.overall_pass else "FAIL",
        )

    # Fast Micro-Run Replay against real captured live BBO ticks!
    micro_session = MicroRunSession(store, micro_store)
    micro_run_outputs: dict[str, dict] = {}
    for sid in ("exp4d_lt_gate_safe_2x", "exp5b_lt_okx_safe_2x", "exp6b_hl_okx_safe_2x_realbasis"):
        c_mr = StrategyConfig.from_dict(experiment_outputs[sid]["config"])
        rec_mr = micro_session.evaluate_bbo_window(c_mr, target_hours=24.0, max_stale_sec=None)
        micro_run_outputs[sid] = rec_mr.to_dict()
        logger.info(
            "MICRO-RUN %-30s | status=%-12s | ticks=%4d | liveHalf=%.2fbp vs assumed=%.2fbp (%.2fx) | dq=%s",
            sid, rec_mr.status, rec_mr.ticks_observed, rec_mr.realized_spread_bps_median,
            rec_mr.assumed_slippage_bps, rec_mr.slippage_ratio, rec_mr.dq_tier,
        )

    # Capital scaling simulation on safe 2.0x/leg candidate (exp4d_lt_gate_safe_2x)
    best_id = "exp4d_lt_gate_safe_2x"
    best_cfg = StrategyConfig.from_dict(experiment_outputs[best_id]["config"])
    capital_scaling = []
    for cap in (1000.0, 3000.0, 5000.0, 10000.0, 20000.0, 50000.0):
        size_slip = best_cfg.slippage_per_leg_bps + (0.0 if cap <= 5000 else (0.5 if cap <= 20000 else 1.5))
        c_cfg = StrategyConfig.from_dict({**best_cfg.to_dict(), "capital_usd": cap, "slippage_per_leg_bps": size_slip})
        r = backtester.run(c_cfg, start_ms=common_start, end_ms=common_end)
        capital_scaling.append({
            "capital_usd": cap,
            "slippage_per_leg_bps": size_slip,
            "net_pnl_window_usd": r.metrics.net_pnl_usd,
            "monthly_usd_est": r.metrics.monthly_usd_est,
            "annualized_apr_pct": r.metrics.annualized_apr_pct,
            "max_dd_pct": r.metrics.max_drawdown_pct,
            "sharpe": r.metrics.sharpe_ratio,
            "liquidation_count": r.metrics.liquidation_count,
            "forced_rebalance_count": r.metrics.forced_rebalance_count,
        })

    output = {
        "generated_utc": utc_iso(),
        "engine_version": "alphaloop-v2.0-redteam-hardened",
        "window_start_utc": utc_iso(common_start),
        "window_end_utc": utc_iso(common_end),
        "window_days": round((common_end - common_start) / (86400 * 1000), 2),
        "db_summary": db_summary,
        "universes": {
            "lt_gate": syms_lt_gate,
            "lt_okx": syms_lt_okx,
            "hl_gate": syms_hl_gate,
            "hl_okx": syms_hl_okx,
            "hl_lighter": syms_hl_lt,
            "okx_spot_perp": syms_okx,
        },
        "experiments": experiment_outputs,
        "micro_runs": micro_run_outputs,
        "blended_portfolio": blended_portfolio,
        "agent_evolution_demo": agent_cycle_1,
        "best_strategy_id": best_id,
        "capital_scaling": capital_scaling,
        "fetch_report_keys": len(fetch_report),
    }

    out_path = data_dir / "research_results.json"
    _atomic_write_json(out_path, output)
    logger.info("Saved full empirical research dossier -> %s", out_path)

    store.close()
    ledger.close()
    config_store.close()
    micro_store.close()
    orchestrator.close()
    return output


if __name__ == "__main__":
    run_experiments(ROOT / "data")
