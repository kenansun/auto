#!/usr/bin/env python3
"""Stage 3 S3-02: Strictly Zero-Lookahead Multi-Horizon CTA Chop-Defense Empirical Research.

Tests whether combining multi-horizon Donchian TrendCTA sleeves (`12h 40/16` + `24h 20/10`)
or slower breakout horizons reduces the choppy Gate 360d H1 drawdown (`10.72%`) WITHOUT
sacrificing `6/6` Gate passage on certified HL 206d and Gate 178d datasets (full TAKER_TAKER).

Scientific Honesty Mandate (STD-06 / REA-001):
  - Reports every metric verbatim (including failures and unpriced funding disclaimer on Gate H1).
  - Recommends adopting a variant ONLY if BOTH HL 206d and Gate 178d pass 6/6 AND 360d H1 MaxDD
    drops >= 2.0 percentage points without OOS collapse; otherwise keeps incumbent `12h 40/16`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.core.provenance import utc_now_ms
from alphaloop.daemon.shadow_runner import _atomic_write_json
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier, TrialLedger
from alphaloop.harness.portfolio import PortfolioCombiner, SleeveInput
from alphaloop.plugins import TrendCTAPlugin, default_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cta-chop-research")

CTA_UNIVERSE = [
    "PAXG", "BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "AAVE", "UNI", "SUI", "HYPE", "LTC",
]


def _make_cta_cfg(
    sid: str,
    venue: str,
    tf_h: int,
    entry_bars: int,
    exit_bars: int,
    atr_mult: float,
    risk_pct: float = 0.8,
) -> Any:
    return TrendCTAPlugin().default_config(
        symbols=CTA_UNIVERSE,
        strategy_id=sid,
        version="v3-chop",
        venue_a=venue,
        venue_b=venue,
        capital_usd=3000.0,
        max_positions=6,
        leverage=2.0,
        execution_mode="TAKER_TAKER",
        params={
            "timeframe_hours": tf_h,
            "entry_n": entry_bars,
            "exit_n": exit_bars,
            "atr_n": 14,
            "atr_mult": atr_mult,
            "risk_per_trade_pct": risk_pct,
            "max_symbol_leverage": 1.5,
            "ref_venue": "gate" if venue == "hyperliquid" else "hyperliquid",
        },
    )


def run_study(data_dir: Path, ledger_path: Path | None = None) -> dict[str, Any]:
    out_dir = data_dir / "phase3_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    store = MarketDataStore(data_dir / "market_data.sqlite")
    ledger = TrialLedger(ledger_path or (data_dir / "trial_ledger.sqlite"))
    reg = default_registry()
    plugin = TrendCTAPlugin()
    bt = FundingCarryBacktester(store)
    verifier = SixGateVerifier(bt, ledger, registry=reg)
    combiner = PortfolioCombiner()

    try:
        # Determine exact 360d Gate candle span and certified 178d Gate funding overlap start
        btc_gate = store.load_candles("gate", "BTC")
        btc_gate_fr = store.load_funding("gate", "BTC")
        if not btc_gate or not btc_gate_fr:
            raise RuntimeError("Missing Gate BTC 1h candles or funding in market_data.sqlite")
        g_start = btc_gate[0].ts_ms
        g_end = btc_gate[-1].ts_ms
        g_fr_start = btc_gate_fr[0].ts_ms
        g_mid = g_fr_start

        variants = [
            ("incumbent_12h_40_16_atr35", 12, 40, 16, 3.5, 0.8),
            ("slow_24h_20_10_atr35", 24, 20, 10, 3.5, 0.8),
            ("slow_24h_30_12_atr30", 24, 30, 12, 3.0, 0.8),
            ("tight_12h_40_16_atr28_r06", 12, 40, 16, 2.8, 0.6),
        ]

        results: dict[str, Any] = {}
        hl_curves: dict[str, list[tuple[int, float]]] = {}
        gate_360_curves: dict[str, list[tuple[int, float]]] = {}
        gate_h1_curves: dict[str, list[tuple[int, float]]] = {}

        for name, tf_h, ent, ext, atr, rpct in variants:
            hl_cfg = _make_cta_cfg(f"hl_{name}", "hyperliquid", tf_h, ent, ext, atr, rpct)
            gate_cfg = _make_cta_cfg(f"gate_{name}", "gate", tf_h, ent, ext, atr, rpct)

            v_hl, res_hl, _ = verifier.verify(hl_cfg, plugin=plugin)
            v_gt, res_gt, _ = verifier.verify(gate_cfg, start_ms=g_fr_start, end_ms=g_end, plugin=plugin)

            # 360d Gate full-year & H1 choppy stress (honestly tagged unpriced funding on H1)
            res_360 = plugin.evaluate(store, gate_cfg, start_ms=g_start, end_ms=g_end)
            res_h1 = plugin.evaluate(store, gate_cfg, start_ms=g_start, end_ms=g_mid)

            hl_curves[name] = res_hl.equity_curve
            gate_360_curves[name] = res_360.equity_curve
            gate_h1_curves[name] = res_h1.equity_curve

            results[name] = {
                "hl_206d": {
                    "gates": f"{v_hl.passed_count}/{v_hl.total_count}",
                    "overall_pass": v_hl.overall_pass,
                    "dq_tier": v_hl.dq_tier,
                    "apr_pct": res_hl.metrics.annualized_apr_pct,
                    "sharpe": res_hl.metrics.sharpe_ratio,
                    "max_dd_pct": res_hl.metrics.max_drawdown_pct,
                    "net_usd": round(res_hl.metrics.net_pnl_usd, 2),
                },
                "gate_178d": {
                    "gates": f"{v_gt.passed_count}/{v_gt.total_count}",
                    "overall_pass": v_gt.overall_pass,
                    "dq_tier": v_gt.dq_tier,
                    "apr_pct": res_gt.metrics.annualized_apr_pct,
                    "sharpe": res_gt.metrics.sharpe_ratio,
                    "max_dd_pct": res_gt.metrics.max_drawdown_pct,
                    "net_usd": round(res_gt.metrics.net_pnl_usd, 2),
                },
                "gate_360d_unpriced_funding": {
                    "full_apr_pct": res_360.metrics.annualized_apr_pct,
                    "full_sharpe": res_360.metrics.sharpe_ratio,
                    "full_max_dd_pct": res_360.metrics.max_drawdown_pct,
                    "h1_chop_apr_pct": res_h1.metrics.annualized_apr_pct,
                    "h1_chop_max_dd_pct": res_h1.metrics.max_drawdown_pct,
                },
            }
            logger.info(
                "%-28s | HL=%s (%+.1f%%, DD %.2f%%) | Gate178=%s (%+.1f%%) | Gate360=%+.1f%% (H1 %+.1f%%, H1 DD %.2f%%)",
                name,
                results[name]["hl_206d"]["gates"],
                results[name]["hl_206d"]["apr_pct"],
                results[name]["hl_206d"]["max_dd_pct"],
                results[name]["gate_178d"]["gates"],
                results[name]["gate_178d"]["apr_pct"],
                results[name]["gate_360d_unpriced_funding"]["full_apr_pct"],
                results[name]["gate_360d_unpriced_funding"]["h1_chop_apr_pct"],
                results[name]["gate_360d_unpriced_funding"]["h1_chop_max_dd_pct"],
            )

        # Evaluate causal 50/50 Multi-Horizon Ensemble (12h 40/16 + 24h 20/10)
        ens_hl = combiner.combine(
            [
                SleeveInput("cta_12h_40_16", hl_curves["incumbent_12h_40_16_atr35"], weight=0.50),
                SleeveInput("cta_24h_20_10", hl_curves["slow_24h_20_10_atr35"], weight=0.50),
            ],
            weighting_mode="fixed",
        )
        ens_360 = combiner.combine(
            [
                SleeveInput("cta_12h_40_16", gate_360_curves["incumbent_12h_40_16_atr35"], weight=0.50),
                SleeveInput("cta_24h_20_10", gate_360_curves["slow_24h_20_10_atr35"], weight=0.50),
            ],
            weighting_mode="fixed",
        )
        ens_h1 = combiner.combine(
            [
                SleeveInput("cta_12h_40_16", gate_h1_curves["incumbent_12h_40_16_atr35"], weight=0.50),
                SleeveInput("cta_24h_20_10", gate_h1_curves["slow_24h_20_10_atr35"], weight=0.50),
            ],
            weighting_mode="fixed",
        )

        inc_h1_dd = results["incumbent_12h_40_16_atr35"]["gate_360d_unpriced_funding"]["h1_chop_max_dd_pct"]
        ens_h1_dd = ens_h1.max_drawdown_pct if ens_h1.valid else 999.0
        dd_improvement_pct = round(inc_h1_dd - ens_h1_dd, 2)
        both_pass = (
            results["incumbent_12h_40_16_atr35"]["hl_206d"]["overall_pass"]
            and results["incumbent_12h_40_16_atr35"]["gate_178d"]["overall_pass"]
            and results["slow_24h_20_10_atr35"]["hl_206d"]["overall_pass"]
            and results["slow_24h_20_10_atr35"]["gate_178d"]["overall_pass"]
        )
        verdict = (
            "ADOPT_MULTI_HORIZON_ENSEMBLE"
            if (both_pass and dd_improvement_pct >= 2.0 and ens_hl.valid and ens_hl.oos_retention >= 0.5)
            else "KEEP_INCUMBENT_12H_40_16_RECORD_FINDINGS"
        )

        report = {
            "generated_ts_ms": utc_now_ms(),
            "variants": results,
            "ensemble_12h_24h_50_50": {
                "hl_206d": {
                    "valid": ens_hl.valid,
                    "apr_pct": ens_hl.annualized_apr_pct,
                    "sharpe": ens_hl.sharpe_ratio,
                    "max_dd_pct": ens_hl.max_drawdown_pct,
                    "oos_retention": ens_hl.oos_retention,
                    "corr": ens_hl.correlation_matrix,
                },
                "gate_360d_full": {
                    "valid": ens_360.valid,
                    "apr_pct": ens_360.annualized_apr_pct,
                    "sharpe": ens_360.sharpe_ratio,
                    "max_dd_pct": ens_360.max_drawdown_pct,
                },
                "gate_h1_chop": {
                    "valid": ens_h1.valid,
                    "apr_pct": ens_h1.annualized_apr_pct,
                    "max_dd_pct": ens_h1.max_drawdown_pct,
                    "dd_reduction_vs_incumbent_pct": dd_improvement_pct,
                },
            },
            "verdict": verdict,
        }
        _atomic_write_json(out_dir / "s302_cta_chop_defense.json", report)
        logger.info("S3-02 Verdict: %s (H1 DD reduction = %+.2f pct)", verdict, dd_improvement_pct)
        return report
    finally:
        ledger.close()
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 S3-02 CTA Chop-Defense Research")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--ledger", default="", help="Optional override path for TrialLedger SQLite")
    args = parser.parse_args()
    run_study(Path(args.data_dir), ledger_path=Path(args.ledger) if args.ledger else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
