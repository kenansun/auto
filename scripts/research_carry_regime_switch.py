#!/usr/bin/env python3
"""Stage 3 S3-03: Causal Funding-Carry Entry-Hurdle & Bear-Regime Filter Empirical Study.

Tests whether raising the causal entry APR hurdle (`min_entry_apr_pct` / `min_hold_hours`)
eliminates unprofitable churn in low-funding bear regimes across 178d real HL<->Gate and
LT<->Gate data, and measures net impact on the Balanced 50/50 multi-sleeve portfolio.
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

from alphaloop.core.provenance import utc_now_ms
from alphaloop.daemon.shadow_runner import _atomic_write_json
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier, TrialLedger
from alphaloop.harness.portfolio import PortfolioCombiner, SleeveInput
from alphaloop.plugins import FundingCarryPlugin, TrendCTAPlugin, default_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("carry-regime-research")

CTA_UNIVERSE = [
    "PAXG", "BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "AAVE", "UNI", "SUI", "HYPE", "LTC",
]
CARRY_UNIVERSE = CTA_UNIVERSE + [
    "ADA", "APT", "ARB", "ENA", "NEAR", "OP", "PENDLE", "TIA", "VVV", "WLD", "ZEC",
]


def run_study(data_dir: Path, ledger_path: Path | None = None) -> dict[str, Any]:
    out_dir = data_dir / "phase3_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    store = MarketDataStore(data_dir / "market_data.sqlite")
    ledger = TrialLedger(ledger_path or (data_dir / "trial_ledger.sqlite"))
    reg = default_registry()
    carry_plugin = FundingCarryPlugin()
    cta_plugin = TrendCTAPlugin()
    bt = FundingCarryBacktester(store)
    verifier = SixGateVerifier(bt, ledger, registry=reg)
    combiner = PortfolioCombiner()

    try:
        # Pin deterministic Phase-2 aligned evaluation windows anchored to dataset max timestamp (zero wall-clock drift)
        max_row = store.conn.fetch_one(
            "SELECT MAX(ts_ms) AS m FROM funding_rates WHERE venue IN ('hyperliquid', 'gate')"
        )
        ref_end_ms = int(max_row["m"]) if (max_row and max_row["m"]) else utc_now_ms()
        s178 = ref_end_ms - 178 * 86_400 * 1000
        s206 = ref_end_ms - 206 * 86_400 * 1000

        # Reference Pillar 2 HL TrendCTA curve for Balanced 50/50 portfolio delta check
        cta_cfg = cta_plugin.default_config(
            symbols=CTA_UNIVERSE,
            strategy_id="s303_ref_cta",
            version="v3.1",
            venue_a="hyperliquid",
            venue_b="hyperliquid",
            capital_usd=3000.0,
            max_positions=6,
            per_position_weight=0.1667,
            leverage=2.0,
            fee_a_bps=1.5,
            slippage_per_leg_bps=2.0,
            execution_mode="TAKER_TAKER",
            params={
                "timeframe_hours": 12,
                "entry_n": 40,
                "exit_n": 16,
                "atr_n": 14,
                "atr_mult": 3.5,
                "ema_filter_n": 0,
                "risk_per_trade_pct": 0.8,
                "max_symbol_leverage": 1.5,
                "cooldown_hours": 24,
                "ref_venue": "gate",
            },
        )
        cta_res = cta_plugin.evaluate(store, cta_cfg, start_ms=s206, end_ms=ref_end_ms)

        # (name, va, vb, slots, lev, entry_apr, exit_apr, lb_h, exit_conf_h, hold_h, vol_tgt, prot)
        grid = [
            ("hl_gate_p2_certified_10_h72", "hyperliquid", "gate", 4, 3.5, 10.0, 1.0, 96, 24, 72, 130.0, 0.50),
            ("hl_gate_regime_hurdle18_h72", "hyperliquid", "gate", 4, 3.5, 18.0, 1.0, 96, 24, 72, 130.0, 0.50),
            ("hl_gate_regime_hurdle30_h48", "hyperliquid", "gate", 4, 3.5, 30.0, 1.0, 96, 24, 48, 130.0, 0.50),
            ("hl_gate_regime_hurdle40_h72", "hyperliquid", "gate", 4, 3.5, 40.0, 1.0, 96, 24, 72, 130.0, 0.50),
            ("lt_gate_p2_certified_6_h96", "lighter", "gate", 8, 3.5, 6.0, 0.5, 120, 36, 96, 110.0, 0.45),
            ("lt_gate_regime_hurdle18_h48", "lighter", "gate", 8, 3.5, 18.0, 0.5, 120, 36, 48, 110.0, 0.45),
            ("lt_gate_regime_hurdle30_h48", "lighter", "gate", 8, 3.5, 30.0, 0.5, 120, 36, 48, 110.0, 0.45),
        ]

        rows: dict[str, Any] = {}
        for name, va, vb, slots, lev, entry_apr, exit_apr, lb_h, exit_conf_h, hold_h, vol_tgt, prot in grid:
            cfg = carry_plugin.default_config(
                symbols=CARRY_UNIVERSE,
                strategy_id=f"s303_{name}",
                version="v3.1",
                venue_a=va,
                venue_b=vb,
                capital_usd=3000.0,
                max_positions=slots,
                per_position_weight=round(1.0 / slots, 4),
                leverage=lev,
                entry_spread_apr_pct=entry_apr,
                exit_spread_apr_pct=exit_apr,
                lookback_hours=lb_h,
                exit_confirm_hours=exit_conf_h,
                min_hold_hours=hold_h,
                fee_a_bps=1.5 if va == "hyperliquid" else 0.0,
                fee_b_bps=2.0,
                slippage_per_leg_bps=0.75,
                params={
                    "vol_target_leverage_enabled": True,
                    "vol_lookback_hours": 168,
                    "vol_target_ann_pct": vol_tgt,
                    "min_symbol_leverage": 1.5,
                    "protective_margin_exit_pct": prot,
                    "protective_exit_slip_bps": 5.0,
                },
            )
            verdict, res, _ = verifier.verify(cfg, start_ms=s178, end_ms=ref_end_ms, plugin=carry_plugin)
            port = combiner.combine(
                [
                    SleeveInput(name, res.equity_curve, weight=0.50),
                    SleeveInput("cta_hl_40_16", cta_res.equity_curve, weight=0.50),
                ],
                capital_usd=10000.0,
                weighting_mode="fixed",
                rebalance_every_days=30,
                rebalance_cost_bps=10.0,
            )
            rows[name] = {
                "gates": f"{verdict.passed_count}/{verdict.total_count}",
                "dq_tier": verdict.dq_tier,
                "apr_pct": res.metrics.annualized_apr_pct,
                "sharpe": res.metrics.sharpe_ratio,
                "max_dd_pct": res.metrics.max_drawdown_pct,
                "trades": res.metrics.trade_count,
                "net_usd": round(res.metrics.net_pnl_usd, 2),
                "liqs": res.metrics.liquidation_count,
                "fee_ratio": res.metrics.fee_to_gross_ratio,
                "balanced_50_50": {
                    "valid": port.valid,
                    "apr_pct": port.annualized_apr_pct,
                    "sharpe": port.sharpe_ratio,
                    "max_dd_pct": port.max_drawdown_pct,
                    "oos_retention": port.oos_retention,
                },
            }
            logger.info(
                "%-30s | %s | APR=%+5.2f%% Sh=%6.2f DD=%.2f%% Tr=%3d Liq=%d | 50/50 APR=%+.2f%% Sh=%.2f DD=%.2f%%",
                name,
                rows[name]["gates"],
                rows[name]["apr_pct"],
                rows[name]["sharpe"],
                rows[name]["max_dd_pct"],
                rows[name]["trades"],
                rows[name]["liqs"],
                port.annualized_apr_pct,
                port.sharpe_ratio,
                port.max_drawdown_pct,
            )

        report = {
            "generated_ts_ms": utc_now_ms(),
            "window_start_ms": s178,
            "window_end_ms": ref_end_ms,
            "rebalance_cost_bps": 10.0,
            "results": rows,
        }
        _atomic_write_json(out_dir / "s303_carry_regime_switch.json", report)
        return report
    finally:
        ledger.close()
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 S3-03 Carry Regime-Switch Research")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--ledger", default="", help="Optional override path for TrialLedger SQLite")
    args = parser.parse_args()
    run_study(Path(args.data_dir), ledger_path=Path(args.ledger) if args.ledger else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
