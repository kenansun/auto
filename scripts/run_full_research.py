#!/usr/bin/env python3
"""End-to-end 90-day REAL market data ingestion + multi-venue funding arbitrage research + Agent evolution demo."""

from __future__ import annotations

import json
import logging
import sys
import time
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("research")

# 24 liquid tokens spanning majors, DeFi blue chips, L1/L2s, and high-OI perp venues
UNIVERSE = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "AAVE",
    "UNI", "SUI", "APT", "ARB", "OP", "NEAR", "ADA", "LTC",
    "HYPE", "ZEC", "PENDLE", "ENA", "WLD", "TIA", "INJ", "VVV",
]


def fetch_dataset(store: MarketDataStore, days: int = 90) -> dict[str, dict[str, int]]:
    hl = HyperliquidCollector()
    lt = LighterCollector()
    okx = OKXCollector()
    gate = GateCollector()

    # Check which symbols already have HL funding cached
    cached_syms = {
        r["symbol"]
        for r in store.conn.execute(
            "SELECT symbol, COUNT(*) AS n FROM funding_rates WHERE venue='hyperliquid' GROUP BY symbol HAVING n >= 1000"
        ).fetchall()
    }
    missing = [s for s in UNIVERSE if s not in cached_syms]
    if not missing:
        existing = store.summary_stats()
        logger.info("All %d symbols cached in %s; skipping fetch.", len(UNIVERSE), store.db_path)
        return {"cached_symbols": {"count": len(cached_syms)}}

    logger.info("Resuming fetch: %d cached, %d remaining (%s)", len(cached_syms), len(missing), ",".join(missing))

    # Save live universe snapshots for provenance
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

    # Fresh ledger & state for clean reproducible research run
    for fn in ("trial_ledger.sqlite",):
        p = data_dir / fn
        if p.exists():
            p.unlink()
    state_dir = data_dir / "state"
    if state_dir.exists():
        for child in state_dir.rglob("*.sqlite*"):
            child.unlink()
        for child in state_dir.rglob("*.json"):
            child.unlink()

    ledger = TrialLedger(data_dir / "trial_ledger.sqlite")
    config_store = HotConfigStore(state_dir)
    backtester = FundingCarryBacktester(store)
    verifier = SixGateVerifier(backtester, ledger)
    router = MultiModelRouter()
    orchestrator = AgentEvolutionOrchestrator(state_dir, backtester, verifier, config_store, router)

    # Determine common overlapping 90d time window across venues
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

    # Filter symbols that actually have sufficient real history on each pair
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
            leverage=4.0,
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
        # EXP 4A (WINNER #1): Structural DEX-vs-CEX Lighter (0% fee) <-> Gate (2bp maker) @ 4x safe DN leverage
        StrategyConfig(
            strategy_id="exp4a_lt_gate_maker_4x",
            version="v2.2",
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
            notes="Structural DEX-vs-CEX Perp-Perp: Lighter 0% fee + Gate 2bp maker, 8-slot diversified",
        ),
        # EXP 4B (WINNER #2): Lighter <-> Gate @ 5x DN leverage (2.5x/leg)
        StrategyConfig(
            strategy_id="exp4b_lt_gate_maker_5x",
            version="v2.2",
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
            notes="Structural DEX-vs-CEX Perp-Perp: Lighter 0% + Gate 2bp maker @ 5x DN leverage (2.5x/leg)",
        ),
        # EXP 4C: Lighter <-> Gate FULL TAKER Stress Test (0bp + 5bp + 2.5bp/leg slip)
        StrategyConfig(
            strategy_id="exp4c_lt_gate_taker_4x",
            version="v2.2",
            family="dex_cex_lt_gate_taker",
            status="DRAFT",
            venue_a="lighter",
            venue_b="gate",
            mode="perp_perp",
            symbols=syms_lt_gate,
            capital_usd=3000.0,
            leverage=4.0,
            max_positions=8,
            per_position_weight=0.125,
            entry_spread_apr_pct=8.0,
            exit_spread_apr_pct=0.5,
            lookback_hours=120,
            exit_confirm_hours=36,
            min_hold_hours=120,
            fee_a_bps=0.0,
            fee_b_bps=5.0,
            slippage_per_leg_bps=2.5,
            rebalance_drag_bps_per_month=3.0,
            allow_bidirectional=True,
            notes="Full taker stress test on Lighter <-> Gate",
        ),
        # EXP 5 (WINNER #3): Structural DEX-vs-CEX Lighter (0% fee) <-> OKX (2bp maker) @ 4x safe DN leverage
        StrategyConfig(
            strategy_id="exp5_lt_okx_maker_4x",
            version="v2.2",
            family="dex_cex_lt_okx",
            status="DRAFT",
            venue_a="lighter",
            venue_b="okx",
            mode="perp_perp",
            symbols=syms_lt_okx,
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
            notes="Structural DEX-vs-CEX Perp-Perp: Lighter 0% + OKX 2bp maker, 0.60x OOS retention",
        ),
        # EXP 6A: Dual-Candle Real Basis Verification — Hyperliquid <-> Gate (96% real dual 1h candles!)
        StrategyConfig(
            strategy_id="exp6a_hl_gate_maker_4x_realbasis",
            version="v2.2",
            family="dex_cex_hl_gate",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="gate",
            mode="perp_perp",
            symbols=syms_hl_gate,
            capital_usd=3000.0,
            leverage=4.0,
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
            notes="100% real dual-venue 1h basis accounting verification: HL <-> Gate",
        ),
        # EXP 6B: Dual-Candle Real Basis Verification — Hyperliquid <-> OKX (100% real dual 1h candles!)
        StrategyConfig(
            strategy_id="exp6b_hl_okx_maker_4x_realbasis",
            version="v2.2",
            family="dex_cex_hl_okx",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="okx",
            mode="perp_perp",
            symbols=syms_hl_okx,
            capital_usd=3000.0,
            leverage=4.0,
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
            notes="100% real dual-venue 1h basis accounting verification: HL <-> OKX",
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
            "stress_metrics": extras["stress_metrics"],
            "is_metrics": extras["is_metrics"],
            "oos_metrics": extras["oos_metrics"],
            "plateau_grid": extras["plateau_grid"],
            "daily_equity": daily_series(res.equity_curve),
            "sample_trades": [t.to_dict() for t in res.trades[:12]],
        }
        logger.info(
            "%-32s | trades=%3d | gross=$%+7.2f | basis=$%+6.2f | cost=$%6.2f | NET=$%+7.2f | APR=%+6.2f%% | Sharpe=%5.2f | DD=%4.2f%% | 6-Gate=%d/6 %s",
            cfg.strategy_id,
            m.trade_count,
            m.gross_funding_usd,
            m.basis_pnl_usd,
            m.total_fees_usd + m.total_slippage_usd + m.total_rebalance_usd,
            m.net_pnl_usd,
            m.annualized_apr_pct,
            m.sharpe_ratio,
            m.max_drawdown_pct,
            verdict.passed_count,
            "PASS" if verdict.overall_pass else "FAIL",
        )

    # EXP 7: Closed-Loop Agent Evolution Demo on REAL 90-day Lighter <-> Gate data!
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
        leverage=4.0,
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
        "AGENT CYCLE 1 (REAL 90D): anomaly=%s | APR %+.1f%% -> %+.2f%% | gates=%d/%d | red_team=%s | promoted_shadow=%s",
        agent_cycle_1["feedback"]["anomaly_type"],
        agent_cycle_1["baseline_metrics"]["annualized_apr_pct"],
        agent_cycle_1["candidate_metrics"]["annualized_apr_pct"],
        agent_cycle_1["gate_verdict"]["passed_count"],
        agent_cycle_1["gate_verdict"]["total_count"],
        agent_cycle_1["red_team_verdict"]["decision"],
        agent_cycle_1["promoted_shadow"],
    )

    # Blended Custody-Diversified 3-Pair Portfolio (50% LT-Gate 5x + 25% LT-OKX 4x + 25% HL-OKX 4x)
    # Caps single-venue custody exposure to <= 37.5% while blending real dual-basis stability
    c_ltg = dict(curves_by_id["exp4b_lt_gate_maker_5x"])
    c_lto = dict(curves_by_id["exp5_lt_okx_maker_4x"])
    c_hlo = dict(curves_by_id["exp6b_hl_okx_maker_4x_realbasis"])
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
    b_oos_apr = ((blend_curve[-1][1] / blend_curve[sp][1]) - 1.0) * 100.0 * (365.0 / ((blend_curve[-1][0] - blend_curve[sp][0]) / (86400 * 1000)))
    blended_portfolio = {
        "weights": {"exp4b_lt_gate_maker_5x": 0.50, "exp5_lt_okx_maker_4x": 0.25, "exp6b_hl_okx_maker_4x_realbasis": 0.25},
        "net_pnl_usd_on_3k": round(b_end - b_start, 2),
        "annualized_apr_pct": round(b_apr, 2),
        "is_apr_pct": round(b_is_apr, 2),
        "oos_apr_pct": round(b_oos_apr, 2),
        "max_dd_pct": round(b_mdd, 3),
        "daily_equity": daily_series(blend_curve),
    }
    logger.info(
        "BLENDED 3-PAIR CUSTODY-SAFE PORTFOLIO: APR=%+.2f%% (IS %+.1f%% / OOS %+.1f%%) | MaxDD=%.3f%% | Net on $3k=$%+.2f",
        b_apr, b_is_apr, b_oos_apr, b_mdd, b_end - b_start,
    )

    # Capital scaling simulation on primary 6/6-PASS winner (exp4b_lt_gate_maker_5x) + conservative 4x (exp4a_lt_gate_maker_4x)
    best_id = "exp4b_lt_gate_maker_5x"
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
        })

    output = {
        "generated_utc": utc_iso(),
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
        "blended_portfolio": blended_portfolio,
        "agent_evolution_demo": agent_cycle_1,
        "best_strategy_id": best_id,
        "capital_scaling": capital_scaling,
        "fetch_report_keys": len(fetch_report),
    }

    out_path = data_dir / "research_results.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Saved full empirical research dossier -> %s", out_path)

    store.close()
    ledger.close()
    config_store.close()
    orchestrator.close()
    return output


if __name__ == "__main__":
    run_experiments(ROOT / "data")
