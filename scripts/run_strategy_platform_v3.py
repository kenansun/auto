#!/usr/bin/env python3
"""End-to-End Real-Data Runner for AlphaLoop Strategy R&D Platform v3 (Web3 + Gold).

Ingests REAL public market data (Hyperliquid + Gate.io 1h crypto/PAXG bars & funding +
Yahoo Finance 5y daily Gold & Macro series: GC=F, GLD, IEF, DX-Y.NYB, ^TNX, TIP, ^VIX, SI=F),
runs multi-asset DQ certification, cross-source PAXG<->GC=F arbitration, cross-asset G2 factor
transferability audit, physics-based automatic parameter inference, plateau calibration,
CPCV/PBO/exact-DSR verification, and G7 bit-identical single-kernel parity checks.

Zero synthetic data fallback. Zero live execution.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.platform.assets import (
    GOLD_MACRO_SYMBOLS,
    WEB3_LIQUID_UNIVERSE,
    default_gold_specs,
    default_macro_specs,
    default_web3_specs,
)
from alphaloop.platform.data_lake import (
    MultiAssetDataLake,
    audit_cross_source_divergence,
    ingest_real_web3_and_gold_lake,
    resample_canonical_bars,
)
from alphaloop.platform.factors import MacroRegimeTimeline, StreamingFactorEngine
from alphaloop.platform.regime_inference import MarketRegimeProfiler
from alphaloop.platform.strategy_customizer import StrategyCustomizer, simulate_universe

logger = logging.getLogger("platform_v3_runner")


def run_platform_v3_research(
    data_dir: Path,
    refresh_data: bool = True,
    crypto_days: int = 200,
    gold_macro_range: str = "5y",
    capital_usd: float = 3000.0,
) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    lake = MultiAssetDataLake(data_dir / "platform_v3_lake.sqlite")
    try:
        web3_specs = default_web3_specs(WEB3_LIQUID_UNIVERSE, venue="hyperliquid")
        gold_specs = default_gold_specs()
        macro_specs = default_macro_specs()
        for sp in list(web3_specs.values()) + list(gold_specs.values()) + list(macro_specs.values()):
            lake.register_asset(sp)

        ingest_summary: dict[str, Any] = {"skipped": True}
        if refresh_data:
            ingest_summary = ingest_real_web3_and_gold_lake(
                lake,
                web3_symbols=list(WEB3_LIQUID_UNIVERSE),
                crypto_days=crypto_days,
                gold_macro_range=gold_macro_range,
                include_gate_ref=True,
            )

        # Load Macro series (5y daily)
        macro_bars: dict[str, list[Any]] = {}
        for msym in GOLD_MACRO_SYMBOLS:
            mb = lake.load_bars("yahoo", msym, 24.0)
            if mb:
                macro_bars[msym] = mb

        customizer = StrategyCustomizer(target_annual_vol_pct=18.0)

        # Track 1: Web3 12-Major Multi-Asset Portfolio (Hyperliquid 12h resampled)
        web3_bars_12h: dict[str, list[Any]] = {}
        for sym in WEB3_LIQUID_UNIVERSE:
            b1h = lake.load_bars("hyperliquid", sym, 1.0)
            if len(b1h) >= 600:
                web3_bars_12h[sym] = resample_canonical_bars(b1h, 12.0)

        res_web3 = customizer.build_and_validate(
            track_name="web3_hl_12h_majors",
            specs=web3_specs,
            bars_by_symbol=web3_bars_12h,
            macro_bars_by_symbol=macro_bars,
            capital_usd=capital_usd,
            execution_mode="TAKER_TAKER",
            factor_horizon_bars=8,  # 4 days forward
            lake=lake,
        )

        # Track 1B: Web3 + PAXG Combined Cross-Asset Portfolio (Hyperliquid 12h)
        paxg_hl_1h = lake.load_bars("hyperliquid", "PAXG", 1.0)
        paxg_gate_1h = lake.load_bars("gate", "PAXG", 1.0)
        combined_specs = {**web3_specs, "PAXG": gold_specs["PAXG"]}
        combined_bars_12h = dict(web3_bars_12h)
        if len(paxg_hl_1h) >= 600:
            combined_bars_12h["PAXG"] = resample_canonical_bars(paxg_hl_1h, 12.0)
        res_web3_plus_gold = customizer.build_and_validate(
            track_name="web3_plus_paxg_12h",
            specs=combined_specs,
            bars_by_symbol=combined_bars_12h,
            macro_bars_by_symbol=macro_bars,
            capital_usd=capital_usd,
            execution_mode="TAKER_TAKER",
            factor_horizon_bars=8,
            lake=lake,
        )

        # Track 2A: Gold PAXG 24/7 Perp (Hyperliquid 12h + Macro overlay)
        paxg_12h = {"PAXG": resample_canonical_bars(paxg_hl_1h, 12.0)} if len(paxg_hl_1h) >= 600 else {}
        res_gold_paxg = None
        if paxg_12h:
            res_gold_paxg = customizer.build_and_validate(
                track_name="gold_paxg_hl_12h",
                specs={"PAXG": gold_specs["PAXG"]},
                bars_by_symbol=paxg_12h,
                macro_bars_by_symbol=macro_bars,
                capital_usd=capital_usd,
                execution_mode="TAKER_TAKER",
                factor_horizon_bars=8,
                lake=lake,
            )

        # Track 2B: Gold TradFi 5-Year Daily (GC=F & GLD with QuantPedia Gold-Treasury/DXY/TIP Macro Overlay)
        gold_tradfi_bars: dict[str, list[Any]] = {}
        for gsym in ("GC=F", "GLD"):
            gb = lake.load_bars("yahoo", gsym, 24.0)
            if len(gb) >= 200:
                gold_tradfi_bars[gsym] = gb

        res_gold_tradfi = None
        gold_macro_ablation: dict[str, Any] = {}
        if gold_tradfi_bars:
            res_gold_tradfi = customizer.build_and_validate(
                track_name="gold_tradfi_5y_daily",
                specs={k: gold_specs[k] for k in gold_tradfi_bars},
                bars_by_symbol=gold_tradfi_bars,
                macro_bars_by_symbol=macro_bars,
                capital_usd=capital_usd,
                execution_mode="TAKER_TAKER",
                factor_horizon_bars=10,  # ~2 trading weeks forward
                lake=lake,
            )

            # Honest A/B Macro Overlay Ablation on identical 5y Gold bars & plateau config
            snaps_g: dict[str, list[Any]] = {}
            regs_g: dict[str, list[Any]] = {}
            prof_tool = MarketRegimeProfiler()
            for sym, bars in gold_tradfi_bars.items():
                eng = StreamingFactorEngine(gold_specs[sym].calendar, 24.0)
                tl = MacroRegimeTimeline(macro_bars)
                snaps = eng.compute_series(bars, macro_timeline=tl)
                _, rpts = prof_tool.profile_asset(gold_specs[sym], bars, snaps)
                snaps_g[sym] = snaps
                regs_g[sym] = rpts
            cfg_with = res_gold_tradfi.final_kernel_config
            cfg_without = dataclasses.replace(cfg_with, use_macro_overlay=False)
            m_with, _ = simulate_universe(cfg_with, {k: gold_specs[k] for k in gold_tradfi_bars}, gold_tradfi_bars, snaps_g, regs_g)
            m_without, _ = simulate_universe(cfg_without, {k: gold_specs[k] for k in gold_tradfi_bars}, gold_tradfi_bars, snaps_g, regs_g)
            gold_macro_ablation = {
                "with_macro_overlay": m_with.summary_dict(),
                "without_macro_overlay": m_without.summary_dict(),
                "delta_sharpe": round(m_with.sharpe_ratio - m_without.sharpe_ratio, 4),
                "delta_max_dd_pct": round(m_with.max_drawdown_pct - m_without.max_drawdown_pct, 2),
                "delta_apr_pct": round(m_with.apr_pct - m_without.apr_pct, 2),
            }

        # Cross-Source Gold Arbitration (PAXG vs GC=F)
        cross_source: dict[str, Any] = {}
        gc_daily = lake.load_bars("yahoo", "GC=F", 24.0)
        if paxg_hl_1h and gc_daily:
            cross_source["hl_paxg_vs_gc"] = audit_cross_source_divergence(paxg_hl_1h, gc_daily, "HL:PAXG", "COMEX:GC=F").to_dict()
        if paxg_gate_1h and gc_daily:
            cross_source["gate_paxg_vs_gc"] = audit_cross_source_divergence(paxg_gate_1h, gc_daily, "Gate:PAXG", "COMEX:GC=F").to_dict()

        # Cross-Asset Factor Transferability Matrix (Web3 vs Gold PAXG vs Gold TradFi 5y)
        def _ic_map(audit_summary: Any) -> dict[str, dict[str, Any]]:
            if audit_summary is None:
                return {}
            return {
                r.factor_name: {
                    "ic": r.spearman_ic,
                    "mono": r.monotonicity_rho,
                    "passed_g2": r.passed_g2,
                    "selected": (r.factor_name in audit_summary.selected_non_redundant_factors),
                }
                for r in audit_summary.results
            }

        w3_ic = _ic_map(res_web3.factor_audit)
        gp_ic = _ic_map(res_gold_paxg.factor_audit if res_gold_paxg else None)
        gt_ic = _ic_map(res_gold_tradfi.factor_audit if res_gold_tradfi else None)
        transfer_matrix: dict[str, Any] = {}
        for fname in sorted(set(w3_ic) | set(gp_ic) | set(gt_ic)):
            transfer_matrix[fname] = {
                "web3_12h": w3_ic.get(fname),
                "gold_paxg_12h": gp_ic.get(fname),
                "gold_tradfi_5y_1d": gt_ic.get(fname),
                "passes_both_web3_and_gold_tradfi": bool(
                    (w3_ic.get(fname) or {}).get("passed_g2") and (gt_ic.get(fname) or {}).get("passed_g2")
                ),
            }

        report: dict[str, Any] = {
            "lake_summary": lake.summary_stats(),
            "ingest_summary": ingest_summary,
            "cross_source_gold_arbitration": cross_source,
            "cross_asset_factor_transfer_matrix": transfer_matrix,
            "gold_macro_ablation_5y": gold_macro_ablation,
            "tracks": {
                "web3_hl_12h_majors": res_web3.to_dict(),
                "web3_plus_paxg_12h": res_web3_plus_gold.to_dict(),
                "gold_paxg_hl_12h": res_gold_paxg.to_dict() if res_gold_paxg else None,
                "gold_tradfi_5y_daily": res_gold_tradfi.to_dict() if res_gold_tradfi else None,
            },
        }
        out_json = data_dir / "platform_v3_report.json"
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report
    finally:
        lake.close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Run AlphaLoop Strategy R&D Platform v3 on REAL Web3 + Gold data")
    p.add_argument("--data-dir", default="data", help="Directory for SQLite lake and report JSON")
    p.add_argument("--no-fetch", action="store_true", help="Reuse existing local SQLite lake without network fetch")
    p.add_argument("--crypto-days", type=int, default=200)
    p.add_argument("--gold-range", default="5y")
    p.add_argument("--capital", type=float, default=3000.0)
    args = p.parse_args(argv)

    rep = run_platform_v3_research(
        data_dir=Path(args.data_dir),
        refresh_data=not args.no_fetch,
        crypto_days=args.crypto_days,
        gold_macro_range=args.gold_range,
        capital_usd=args.capital,
    )
    print("\n=== ALPHALOOP STRATEGY R&D PLATFORM v3 — REAL-DATA SUMMARY ===")
    for name, tr in rep["tracks"].items():
        if not tr:
            continue
        m = tr["baseline_metrics"]
        print(
            f"  [{name:22s}] verdict={tr['verdict']:24s} ({tr['passed_gates']}/{tr['total_gates']} gates) | "
            f"APR={m['apr_pct']:+6.2f}% | SR={m['sharpe_ratio']:5.2f} | DD={m['max_drawdown_pct']:5.2f}% | "
            f"CPCV_pos={tr['cpcv_report']['pct_positive_splits']:4.0f}% | DSR={tr['dsr_report']['dsr']:.3f} | "
            f"PBO={tr['pbo_report']['pbo']:.3f}"
        )
    print(f"Report written to {Path(args.data_dir) / 'platform_v3_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
