#!/usr/bin/env python3
"""AlphaLoop Strategy R&D Platform v4 — Real-Data Multi-Track Empirical Runner.

Executes the decoupled asset-native strategy tracks over the Option B partitioned
Point-in-Time data lake (`catalog.sqlite` + `domains/web3_lake.sqlite` +
`domains/gold_macro_lake.sqlite`) with zero synthetic fallback:

- Track W (Crypto-Native Web3, ZERO TradFi/^VIX coupling):
    * W1: `Web3XSMomCarryKernel` (12-coin HL 12h multi-year + PIT HYPE + OI/funding)
    * W2: `Web3AdaptiveTrendKernel` (HL 12h majors, G2-verified trend factors + crowding/breadth)
    * W3: `Web3RegimeCarryKernel` (HL 12h funding carry, >=3x RT fee floor + DVOL throttle)
- Track G (Gold/Macro & PAXG, ZERO crypto signal coupling, ZERO GC=F+GLD double-count):
    * G1_GC_F & G1_GLD: `GoldTradFiMacroKernel` (strict single-instrument 10y daily + 7-driver TIPS macro)
    * G2_4H_A / B / C: `Gold4hEventKernel` (2.9y real 4h GC=F + 199 real PIT FOMC/NFP/CPI events)
    * G3_PAXG: `GoldPAXGBridgeKernel` (HL 12h PAXG + Binance Spot + COMEX dual-basis & weekend guard)
- Top-Level Return-Stream Composition:
    * `PortfolioCombiner` uniting independently validated Track W & Track G return streams
      strictly at the portfolio layer (REA-001 separated margin & capital buckets).

Every strategy is evaluated fail-closed across all 8 institutional gates (`G1..G8`),
persisting cumulative hypothesis-family trials to the append-only `trial_ledger`.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT_PATH.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alphaloop.core.provenance import PROVENANCE_REAL, sha256_obj
from alphaloop.platform.assets import (
    AssetSpec,
    default_gold_specs,
    default_macro_specs_v4,
    default_web3_specs,
)
from alphaloop.platform.core_engine import (
    AssetNativeStrategy,
    CausalRiskAllocator,
    FoldResult,
    InstitutionalGateReport,
    PlateauPoint,
    PortfolioAccountingEngine,
    PortfolioCombiner,
    RiskAllocatorConfig,
    SimulationResult,
    compute_buy_and_hold_returns,
    compute_pbo_cscv,
    compute_psr_dsr,
    evaluate_8_gates,
    evaluate_benchmark_gate,
    run_nested_cpcv,
    select_plateau_center,
    sharpe_from_returns,
    verify_full_parity,
)
from alphaloop.platform.data_foundation import (
    CanonicalBar,
    DataFoundationRouter,
    MacroEventRelease,
    PAXGBasisPoint,
    ingest_platform_v4_real_lake,
)
from alphaloop.platform.factors import (
    FACTOR_ECONOMIC_PRIORS,
    FactorDirectionAuditor,
    MacroRegimeTimeline,
    StreamingFactorEngine,
)
from alphaloop.platform.strategies.gold import (
    Gold4hEventKernel,
    GoldMacroEngine,
    GoldPAXGBridgeKernel,
    GoldTradFiMacroKernel,
)
from alphaloop.platform.strategies.web3 import (
    Web3AdaptiveTrendKernel,
    Web3MarketRegimeEngine,
    Web3RegimeCarryKernel,
    Web3XSMomCarryKernel,
)

logger = logging.getLogger("alphaloop.platform_v4_runner")

WEB3_12_UNIVERSE: tuple[str, ...] = (
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "DOGE",
    "AVAX",
    "LINK",
    "AAVE",
    "SUI",
    "UNI",
    "LTC",
    "HYPE",
)
WEB3_MAJORS_8: tuple[str, ...] = (
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "DOGE",
    "AVAX",
    "LINK",
    "AAVE",
)
WEB3_CARRY_4: tuple[str, ...] = ("BTC", "ETH", "SOL", "AVAX")
GOLD_MACRO_13: tuple[str, ...] = (
    "GC=F",
    "GLD",
    "SI=F",
    "SLV",
    "DX-Y.NYB",
    "UUP",
    "^TNX",
    "TIP",
    "IEF",
    "TLT",
    "^VIX",
    "^GVZ",
    "GDX",
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _assert_real_bars(bars: Sequence[CanonicalBar], label: str) -> None:
    if not bars:
        raise ValueError(f"Fail-closed: missing real bars for {label}")
    for b in bars:
        if b.provenance != PROVENANCE_REAL:
            raise ValueError(
                f"Fail-closed: non-real provenance {b.provenance!r} detected in {label} "
                f"at ts={b.ts_event_ms}"
            )


def _compute_factor_audit(
    track_name: str,
    bars_by_symbol: dict[str, list[CanonicalBar]],
    specs: dict[str, AssetSpec],
    *,
    horizon_bars: int,
    macro_timeline: MacroRegimeTimeline | None = None,
) -> Any:
    snaps_by_symbol: dict[str, list[Any]] = {}
    for sym, bars in bars_by_symbol.items():
        if len(bars) < 30:
            continue
        engine = StreamingFactorEngine(specs[sym].calendar, bars[0].timeframe_h)
        snaps_by_symbol[sym] = [
            engine.update(
                b,
                macro_timeline.snapshot_as_of(b.ts_avail_ms) if macro_timeline is not None else None,
            )
            for b in bars
        ]
    auditor = FactorDirectionAuditor()
    return auditor.audit(
        track_name=track_name,
        bars_by_symbol=bars_by_symbol,
        snaps_by_symbol=snaps_by_symbol,
        horizon_bars=horizon_bars,
        priors=FACTOR_ECONOMIC_PRIORS,
    )


def _build_nested_cpcv_from_candidates(
    candidate_returns: list[tuple[dict[str, Any], list[float]]],
    periods_per_year: float,
    *,
    purge_bars: int,
    embargo_bars: int,
) -> Any:
    """Strictly leakage-free Nested CPCV over causal candidate return streams."""
    if not candidate_returns:
        raise ValueError("Nested CPCV requires at least one candidate return stream")
    min_len = min(len(rets) for _, rets in candidate_returns)
    trimmed = [(params, rets[-min_len:]) for params, rets in candidate_returns]

    def _fit_and_eval(
        train_indices: tuple[int, ...],
        test_indices: tuple[int, ...],
    ) -> FoldResult:
        best_idx = 0
        best_sr = -1e18
        for idx, (_params, rets) in enumerate(trimmed):
            tr_r = [rets[i] for i in train_indices]
            sr = sharpe_from_returns(tr_r, periods_per_year)
            if math.isfinite(sr) and sr > best_sr:
                best_sr = sr
                best_idx = idx
        win_params, win_rets = trimmed[best_idx]
        is_r = tuple(win_rets[i] for i in train_indices)
        oos_r = tuple(win_rets[i] for i in test_indices)
        return FoldResult(
            oos_returns=oos_r,
            is_returns=is_r,
            selected_params=dict(win_params),
        )

    return run_nested_cpcv(
        min_len,
        _fit_and_eval,
        periods_per_year=periods_per_year,
        n_groups=6,
        k_test=2,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )


def _slice_bars_tail(
    bars_by_symbol: dict[str, list[CanonicalBar]],
    tail_n: int,
) -> dict[str, list[CanonicalBar]]:
    return {sym: list(bars[-tail_n:]) for sym, bars in bars_by_symbol.items() if bars}


def _summarize_sim(res: SimulationResult) -> dict[str, Any]:
    return {
        "strategy_id": res.strategy_id,
        "asset_domain": res.asset_domain,
        "n_bars": len(res.period_returns),
        "capital_usd": round(res.capital_usd, 2),
        "final_equity_usd": round(res.final_equity_usd, 2),
        "net_pnl_usd": round(res.net_pnl_usd, 2),
        "total_return_pct": round(res.total_return_pct, 4),
        "cagr_pct": round(res.cagr_pct, 4),
        "simple_apr_pct": round(res.simple_apr_pct, 4),
        "sharpe_ratio": round(res.sharpe_ratio, 4),
        "max_drawdown_pct": round(res.max_drawdown_pct, 4),
        "calmar_ratio": round(res.calmar_ratio, 4),
        "realized_annual_vol_pct": round(res.realized_annual_vol_pct, 4),
        "trade_count": int(res.trade_count),
        "candidate_count": int(res.trade_ledger.candidate_count),
        "suppressed_count": int(res.trade_ledger.suppressed_count),
        "liquidation_count": int(res.liquidation_count),
        "total_cost_usd": round(res.total_cost_usd, 2),
        "funding_pnl_usd": round(res.funding_pnl_usd, 2),
        "holding_drag_usd": round(res.holding_drag_usd, 2),
        "total_turnover": round(res.total_turnover, 4),
    }


def _evaluate_strategy_full(
    *,
    router: DataFoundationRouter,
    family_id: str,
    track_name: str,
    strategy_id: str,
    asset_domain: str,
    baseline_factory: Callable[[], AssetNativeStrategy],
    grid_factories: list[tuple[int, int, str, dict[str, Any], Callable[[], AssetNativeStrategy]]],
    specs: dict[str, AssetSpec],
    bars_by_symbol: dict[str, list[CanonicalBar]],
    risk_cfg: RiskAllocatorConfig,
    dq_reports: list[Any],
    factor_audit: Any,
    benchmark_returns: list[float],
    benchmark_name: str,
    periods_per_year: float,
    capital_usd: float,
    context_provider: Callable[[int], dict[str, Any]] | None = None,
    pbo_extra_returns: list[list[float]] | None = None,
    strategy_mode: str = "auto",
    parity_tail_bars: int = 360,
    purge_bars: int = 4,
    embargo_bars: int = 4,
    precomputed_base_sim: SimulationResult | None = None,
    baseline_grid_label: str | None = None,
) -> tuple[SimulationResult, InstitutionalGateReport, dict[str, Any]]:
    """Run full 8-gate evaluation for one decoupled asset-native strategy."""
    acct = PortfolioAccountingEngine()

    base_sim = (
        precomputed_base_sim
        if precomputed_base_sim is not None
        else acct.simulate(
            baseline_factory(),
            specs,
            bars_by_symbol,
            risk_cfg=risk_cfg,
            capital_usd=capital_usd,
            context_provider=context_provider,
        )
    )
    stress_sim = acct.simulate(
        baseline_factory(),
        specs,
        bars_by_symbol,
        risk_cfg=risk_cfg,
        capital_usd=capital_usd,
        extra_slippage_bps_per_leg=2.0,
        context_provider=context_provider,
    )

    # G4 Local Parameter Stability Grid + Candidate Streams for Nested CPCV & DSR
    plateau_points: list[PlateauPoint] = []
    candidate_streams: list[tuple[dict[str, Any], list[float]]] = []
    grid_matrix: list[list[float]] = []

    for ix, iy, label, params, factory in grid_factories:
        if baseline_grid_label is not None and label == baseline_grid_label:
            sim_g = base_sim
        else:
            sim_g = acct.simulate(
                factory(),
                specs,
                bars_by_symbol,
                risk_cfg=risk_cfg,
                capital_usd=capital_usd,
                context_provider=context_provider,
            )
        plateau_points.append(
            PlateauPoint(
                ix=ix,
                iy=iy,
                label=label,
                params=params,
                sharpe=sim_g.sharpe_ratio,
                apr_pct=sim_g.cagr_pct,
                max_dd_pct=sim_g.max_drawdown_pct,
                net_pnl_usd=sim_g.net_pnl_usd,
            )
        )
        candidate_streams.append((params, list(sim_g.period_returns)))
        grid_matrix.append(list(sim_g.period_returns))

    plateau_rep = select_plateau_center(plateau_points)

    # G5 True Nested CPCV + Fixed DSR (with cumulative TrialLedger family count)
    nested_cpcv = _build_nested_cpcv_from_candidates(
        candidate_streams,
        periods_per_year=periods_per_year,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )

    prior_family_trials = router.get_family_trial_count(family_id)
    cumulative_family_trials = max(len(grid_matrix), prior_family_trials + len(grid_matrix))
    dsr_rep = compute_psr_dsr(
        base_sim.period_returns,
        periods_per_year=periods_per_year,
        n_trials=len(grid_matrix),
        candidate_returns_matrix=grid_matrix,
        family_trial_count=cumulative_family_trials,
    )

    # G6 Multi-Variant CSCV PBO (include distinct domain variants when supplied)
    pbo_candidates: list[list[float]] = [list(base_sim.period_returns)]
    for row in grid_matrix:
        if row != base_sim.period_returns:
            pbo_candidates.append(row)
    if pbo_extra_returns:
        for extra in pbo_extra_returns:
            if len(extra) >= 64:
                pbo_candidates.append(list(extra))
    min_pbo_len = min(len(r) for r in pbo_candidates)
    aligned_pbo = [r[-min_pbo_len:] for r in pbo_candidates]
    pbo_rep = compute_pbo_cscv(
        aligned_pbo,
        periods_per_year=periods_per_year,
        n_subsamples=8,
        plateau_center_idx=0,
    )

    # G7 Full Kernel + RiskAllocator Streaming-vs-Batch Bit-Parity
    parity_bars = _slice_bars_tail(bars_by_symbol, parity_tail_bars)
    parity_rep = verify_full_parity(
        baseline_factory(),
        specs,
        parity_bars,
        risk_cfg=risk_cfg,
        capital_usd=capital_usd,
        context_provider=context_provider,
    )

    # G8 Mandatory Passive Benchmark & Risk-Adjusted Alpha Gate
    bh_aligned = benchmark_returns[-len(base_sim.period_returns) :]
    strat_aligned = base_sim.period_returns[-len(bh_aligned) :]
    bench_rep = evaluate_benchmark_gate(
        strat_aligned,
        bh_aligned,
        periods_per_year=periods_per_year,
        benchmark_name=benchmark_name,
        strategy_mode=strategy_mode,
    )

    gate_report = evaluate_8_gates(
        strategy_id=strategy_id,
        asset_domain=asset_domain,
        dq_reports=dq_reports,
        factor_audit=factor_audit,
        baseline_metrics=base_sim,
        stress_metrics=stress_sim,
        plateau_report=plateau_rep,
        nested_cpcv_report=nested_cpcv,
        dsr_report=dsr_rep,
        pbo_report=pbo_rep,
        parity_report=parity_rep,
        benchmark_report=bench_rep,
    )

    # Persist REA-001 append-only trial records for EVERY evaluated grid point (DEF-03: 9 -> 18 -> 27)
    primary_idx = 0
    if baseline_grid_label is not None:
        for idx_pt, pt in enumerate(plateau_points):
            if pt.label == baseline_grid_label:
                primary_idx = idx_pt
                break
    else:
        for idx_pt, pt in enumerate(plateau_points):
            if pt.ix == 1 and pt.iy == 1:
                primary_idx = idx_pt
                break

    risk_cfg_dict = asdict(risk_cfg)
    trial_id = ""
    trials_recorded_in_run = 0
    for idx_pt, pt in enumerate(plateau_points):
        if idx_pt == primary_idx:
            params_hash = sha256_obj(
                {
                    "strategy_id": strategy_id,
                    "grid_label": pt.label,
                    "grid_params": [p for _, _, _, p, _ in grid_factories],
                    "risk_cfg": risk_cfg_dict,
                }
            )[:16]
            trial_id = router.record_trial(
                family_id=family_id,
                track_name=track_name,
                strategy_name=strategy_id,
                params_hash=params_hash,
                metrics={
                    "grid_label": pt.label,
                    "ix": int(pt.ix),
                    "iy": int(pt.iy),
                    "grid_trials_in_run": len(plateau_points),
                    "cagr_pct": round(base_sim.cagr_pct, 4),
                    "sharpe_ratio": round(base_sim.sharpe_ratio, 4),
                    "max_drawdown_pct": round(base_sim.max_drawdown_pct, 4),
                    "dsr": round(dsr_rep.dsr, 4),
                    "pbo": round(pbo_rep.pbo, 4),
                    "passed_gates": gate_report.passed_gates,
                },
                verdict=gate_report.verdict,
            )
            trials_recorded_in_run += 1
        else:
            cand_hash = sha256_obj(
                {
                    "strategy_id": strategy_id,
                    "grid_label": pt.label,
                    "params": pt.params,
                    "risk_cfg": risk_cfg_dict,
                }
            )[:16]
            router.record_trial(
                family_id=family_id,
                track_name=track_name,
                strategy_name=f"{strategy_id}:{pt.label}",
                params_hash=cand_hash,
                metrics={
                    "grid_label": pt.label,
                    "ix": int(pt.ix),
                    "iy": int(pt.iy),
                    "cagr_pct": round(pt.apr_pct, 4),
                    "sharpe_ratio": round(pt.sharpe, 4),
                    "max_drawdown_pct": round(pt.max_dd_pct, 4),
                    "net_pnl_usd": round(pt.net_pnl_usd, 2),
                },
                verdict="GRID_CANDIDATE",
            )
            trials_recorded_in_run += 1

    if not trial_id:
        params_hash = sha256_obj({"strategy_id": strategy_id, "risk_cfg": risk_cfg_dict})[:16]
        trial_id = router.record_trial(
            family_id=family_id,
            track_name=track_name,
            strategy_name=strategy_id,
            params_hash=params_hash,
            metrics={
                "cagr_pct": round(base_sim.cagr_pct, 4),
                "sharpe_ratio": round(base_sim.sharpe_ratio, 4),
                "max_drawdown_pct": round(base_sim.max_drawdown_pct, 4),
                "dsr": round(dsr_rep.dsr, 4),
                "pbo": round(pbo_rep.pbo, 4),
                "passed_gates": gate_report.passed_gates,
            },
            verdict=gate_report.verdict,
        )
        trials_recorded_in_run += 1

    detail = {
        "trial_id": trial_id,
        "family_id": family_id,
        "trials_recorded_in_run": trials_recorded_in_run,
        "cumulative_family_trials": router.get_family_trial_count(family_id),
        "baseline": _summarize_sim(base_sim),
        "stress_plus_2bp": {
            "net_pnl_usd": round(stress_sim.net_pnl_usd, 2),
            "cagr_pct": round(stress_sim.cagr_pct, 4),
            "sharpe_ratio": round(stress_sim.sharpe_ratio, 4),
            "max_drawdown_pct": round(stress_sim.max_drawdown_pct, 4),
        },
        "g2_factor_audit": {
            "selected_factors": list(factor_audit.selected_non_redundant_factors),
            "passed_count": len(factor_audit.passed_g2_factors),
            "total_evaluated": int(factor_audit.total_factors_evaluated),
        },
        "g4_plateau": plateau_rep.to_dict(),
        "g5_nested_cpcv": {
            "is_nested": nested_cpcv.is_nested,
            "n_splits": nested_cpcv.n_splits,
            "median_is_sharpe": round(nested_cpcv.median_is_sharpe, 4),
            "median_oos_sharpe": round(nested_cpcv.median_oos_sharpe, 4),
            "oos_positive_ratio": round(nested_cpcv.oos_positive_ratio, 4),
            "passed_cpcv_gate": nested_cpcv.passed_cpcv_gate,
        },
        "g5_dsr": dsr_rep.to_dict(),
        "g6_pbo": pbo_rep.to_dict(),
        "g7_parity": parity_rep.to_dict(),
        "g8_benchmark": bench_rep.to_dict(),
        "gates": gate_report.to_dict(),
    }
    return base_sim, gate_report, detail


def _build_paxg_context_lookup(
    basis_points: Sequence[PAXGBasisPoint],
    spot_bars: Sequence[CanonicalBar],
    gc_bars: Sequence[CanonicalBar],
    macro_engine: GoldMacroEngine,
) -> Callable[[int], dict[str, Any]]:
    basis_ts = [p.ts_avail_ms for p in basis_points]
    spot_ts = [b.ts_avail_ms for b in spot_bars]
    gc_ts = [b.ts_avail_ms for b in gc_bars]

    def _provider(ts_ms: int) -> dict[str, Any]:
        ctx: dict[str, Any] = {"gold_macro": macro_engine.snapshot_as_of(ts_ms)}
        b_idx = bisect.bisect_right(basis_ts, ts_ms) - 1
        if b_idx >= 0:
            bp = basis_points[b_idx]
            ctx["paxg_spot_price"] = bp.paxg_spot_close
            ctx["comex_price"] = bp.gc_ref_close
            ctx["gc_f_price"] = bp.gc_ref_close
            ctx["is_weekend_frozen"] = bool(bp.is_weekend_frozen)
            return ctx
        s_idx = bisect.bisect_right(spot_ts, ts_ms) - 1
        if s_idx >= 0:
            ctx["paxg_spot_price"] = spot_bars[s_idx].close
        g_idx = bisect.bisect_right(gc_ts, ts_ms) - 1
        if g_idx >= 0:
            ctx["comex_price"] = gc_bars[g_idx].close
            ctx["gc_f_price"] = gc_bars[g_idx].close
        return ctx

    return _provider


def run_platform_v4_evaluation(
    lake_dir: Path,
    *,
    report_out: Path | None = None,
    offline: bool = True,
    refresh: bool = False,
    capital_usd: float = 10_000.0,
) -> dict[str, Any]:
    """Execute end-to-end Platform v4 real-data empirical evaluation."""
    t_start = time.perf_counter()

    if refresh and not offline:
        logger.info("Refreshing real multi-year Platform v4 data lake at %s", lake_dir)
        ingest_platform_v4_real_lake(lake_dir)

    catalog_path = lake_dir / "catalog.sqlite"
    web3_path = lake_dir / "domains" / "web3_lake.sqlite"
    gold_path = lake_dir / "domains" / "gold_macro_lake.sqlite"
    for req_path in (catalog_path, web3_path, gold_path):
        if not req_path.is_file():
            raise FileNotFoundError(
                f"Fail-closed: required Option B data lake store missing: {req_path}"
            )

    router = DataFoundationRouter(lake_dir)
    try:
        # ------------------------------------------------------------------
        # 1. Load & Verify 100% Real Data From Physically Partitioned Stores
        # ------------------------------------------------------------------
        web3_specs_all = default_web3_specs(WEB3_12_UNIVERSE)
        web3_bars_12h: dict[str, list[CanonicalBar]] = {}
        web3_dq_map: dict[str, Any] = {}
        for sym in WEB3_12_UNIVERSE:
            bars = router.load_bars("web3", "hyperliquid", sym, 43200)
            _assert_real_bars(bars, f"web3:hyperliquid:{sym}:12h")
            web3_bars_12h[sym] = bars
            web3_dq_map[sym] = router.audit_domain_dq("web3", "hyperliquid", sym, 43200, web3_specs_all[sym])

        dvol_pts = router.load_web3_macro("BTC_DVOL")
        stable_pts = router.load_web3_macro("STABLECOIN_MCAP_USD")
        web3_metrics = {
            sym: router.load_web3_metrics("binance_um", sym)
            for sym in ("BTC", "ETH", "SOL")
        }

        gold_specs = default_gold_specs()
        macro_specs = default_macro_specs_v4()
        gold_macro_bars_1d: dict[str, list[CanonicalBar]] = {}
        for sym in GOLD_MACRO_13:
            bars = router.load_bars("gold_macro", "yahoo", sym, 86400)
            _assert_real_bars(bars, f"gold_macro:yahoo:{sym}:1d")
            gold_macro_bars_1d[sym] = bars

        gc_4h_bars = router.load_bars("gold_macro", "yahoo", "GC=F", 14400)
        _assert_real_bars(gc_4h_bars, "gold_macro:yahoo:GC=F:4h")

        paxg_hl_12h = router.load_bars("gold_macro", "hyperliquid", "PAXG", 43200)
        _assert_real_bars(paxg_hl_12h, "gold_macro:hyperliquid:PAXG:12h")
        paxg_spot_12h = router.load_bars("gold_macro", "binance_spot", "PAXG", 43200)
        _assert_real_bars(paxg_spot_12h, "gold_macro:binance_spot:PAXG:12h")

        yield_curve = router.load_yield_curve()
        macro_events: list[MacroEventRelease] = router.load_macro_events()
        if not yield_curve:
            raise ValueError("Fail-closed: US Treasury TIPS real yield curve is empty")
        if not macro_events:
            raise ValueError("Fail-closed: PIT macro event calendar (FOMC/NFP/CPI) is empty")
        paxg_basis_pts = router.load_paxg_basis(
            "hyperliquid", 43200, venue_spot="binance_spot", ref_tradfi_symbol="GC=F"
        )

        dq_gc_1d = router.audit_domain_dq("gold_macro", "yahoo", "GC=F", 86400, gold_specs["GC=F"])
        dq_gld_1d = router.audit_domain_dq("gold_macro", "yahoo", "GLD", 86400, gold_specs["GLD"])
        dq_gc_4h = router.audit_domain_dq("gold_macro", "yahoo", "GC=F", 14400, gold_specs["GC=F"])
        dq_paxg_12h = router.audit_domain_dq("gold_macro", "hyperliquid", "PAXG", 43200, gold_specs["PAXG"])

        # ------------------------------------------------------------------
        # 2. Track W — Crypto-Native Web3 Suite (ZERO Yahoo/^VIX Coupling)
        # ------------------------------------------------------------------
        logger.info("Evaluating Track W (W1/W2/W3) on real 12-coin multi-year Hyperliquid + Deribit + DefiLlama + OI data...")
        web3_regime = Web3MarketRegimeEngine(
            bars_by_symbol=web3_bars_12h,
            dvol_series=dvol_pts,
            stablecoin_series=stable_pts,
            metrics_by_symbol=web3_metrics,
        )

        # G2 Continuous Factor Audit on Web3 (zero TradFi macro timeline)
        w3_factor_audit = _compute_factor_audit(
            "web3_track_w_12h",
            {s: web3_bars_12h[s] for s in WEB3_MAJORS_8},
            {s: web3_specs_all[s] for s in WEB3_MAJORS_8},
            horizon_bars=8,
            macro_timeline=None,
        )

        # Shared Web3 passive benchmarks (BTC B&H & Equal-Weight 12-coin B&H)
        bh_btc_12h = compute_buy_and_hold_returns(web3_bars_12h["BTC"])
        bh_w1_12h = compute_buy_and_hold_returns(web3_bars_12h)
        bh_w2_12h = compute_buy_and_hold_returns({s: web3_bars_12h[s] for s in WEB3_MAJORS_8})
        bh_w3_12h = compute_buy_and_hold_returns({s: web3_bars_12h[s] for s in WEB3_CARRY_4})

        # Precompute baseline return streams of the 3 distinct Track W archetypes for cross-archetype G6 PBO
        acct = PortfolioAccountingEngine()
        w1_risk = RiskAllocatorConfig(
            max_positions=6,
            target_annual_vol_pct=22.0,
            max_portfolio_leverage=1.5,
            max_single_weight=0.35,
            use_correlation_idm=True,
        )
        w2_risk = RiskAllocatorConfig(
            max_positions=6,
            target_annual_vol_pct=20.0,
            max_portfolio_leverage=1.5,
            max_single_weight=0.40,
            use_correlation_idm=True,
        )
        w3_risk = RiskAllocatorConfig(
            max_positions=4,
            target_annual_vol_pct=15.0,
            max_portfolio_leverage=1.5,
            max_single_weight=0.40,
            use_correlation_idm=False,
        )

        w2_bars = {s: web3_bars_12h[s] for s in WEB3_MAJORS_8}
        w2_specs = {s: web3_specs_all[s] for s in WEB3_MAJORS_8}
        w3_bars = {s: web3_bars_12h[s] for s in WEB3_CARRY_4}
        w3_specs = {s: web3_specs_all[s] for s in WEB3_CARRY_4}

        w1_base_ref = acct.simulate(
            Web3XSMomCarryKernel(top_k=2, bottom_k=2, metrics_by_symbol=web3_metrics, regime_engine=web3_regime),
            web3_specs_all,
            web3_bars_12h,
            risk_cfg=w1_risk,
            capital_usd=capital_usd,
        )
        w2_base_ref = acct.simulate(
            Web3AdaptiveTrendKernel(metrics_by_symbol=web3_metrics, regime_engine=web3_regime),
            w2_specs,
            w2_bars,
            risk_cfg=w2_risk,
            capital_usd=capital_usd,
        )
        w3_base_ref = acct.simulate(
            Web3RegimeCarryKernel(dvol_series=dvol_pts, capital_budget_usd=capital_usd),
            w3_specs,
            w3_bars,
            risk_cfg=w3_risk,
            capital_usd=capital_usd,
        )

        w1_grid = [
            (
                ix,
                iy,
                f"k{tk}_ema{int(ema * 100)}",
                {"top_k": tk, "bottom_k": tk, "score_ema_alpha": ema},
                (
                    lambda _tk=tk, _ema=ema: Web3XSMomCarryKernel(
                        top_k=_tk,
                        bottom_k=_tk,
                        score_ema_alpha=_ema,
                        metrics_by_symbol=web3_metrics,
                        regime_engine=web3_regime,
                    )
                ),
            )
            for ix, tk in enumerate((2, 3, 4))
            for iy, ema in enumerate((0.25, 0.35, 0.50))
        ]
        w1_sim, w1_gates, w1_detail = _evaluate_strategy_full(
            router=router,
            family_id="web3_track_w1_xs_mom_carry",
            track_name="track_w_web3",
            strategy_id="web3_w1_xs_mom_carry",
            asset_domain="web3",
            baseline_factory=lambda: Web3XSMomCarryKernel(
                top_k=2, bottom_k=2, metrics_by_symbol=web3_metrics, regime_engine=web3_regime
            ),
            grid_factories=w1_grid,
            specs=web3_specs_all,
            bars_by_symbol=web3_bars_12h,
            risk_cfg=w1_risk,
            dq_reports=[web3_dq_map[s] for s in WEB3_12_UNIVERSE],
            factor_audit=w3_factor_audit,
            benchmark_returns=bh_w1_12h,
            benchmark_name="WEB3_12_EW_BH",
            periods_per_year=730.0,
            capital_usd=capital_usd,
            pbo_extra_returns=[w2_base_ref.period_returns, w3_base_ref.period_returns],
            purge_bars=6,
            embargo_bars=6,
            precomputed_base_sim=w1_base_ref,
            baseline_grid_label="k2_ema35",
        )

        # D7 Zero-VIX Invariance Proof on real-data W1 (tail 400 bars)
        w1_tail = _slice_bars_tail(web3_bars_12h, 400)
        w1_clean = acct.simulate(
            Web3XSMomCarryKernel(top_k=2, bottom_k=2, metrics_by_symbol=web3_metrics, regime_engine=web3_regime),
            web3_specs_all,
            w1_tail,
            risk_cfg=w1_risk,
            capital_usd=capital_usd,
        )
        w1_vix = acct.simulate(
            Web3XSMomCarryKernel(top_k=2, bottom_k=2, metrics_by_symbol=web3_metrics, regime_engine=web3_regime),
            web3_specs_all,
            w1_tail,
            risk_cfg=w1_risk,
            capital_usd=capital_usd,
            context_provider=lambda _ts: {"yahoo_vix": 95.0, "macro_vix_stress_z": 10.0},
        )
        d7_zero_vix_bit_identical = w1_clean.equity_curve == w1_vix.equity_curve

        w2_grid = [
            (
                ix,
                iy,
                f"dn{dn}_atr{int(atr * 10)}",
                {"donchian_entry_n": dn, "donchian_exit_n": dn // 2, "atr_stop_mult": atr},
                (
                    lambda _dn=dn, _atr=atr: Web3AdaptiveTrendKernel(
                        donchian_entry_n=_dn,
                        donchian_exit_n=_dn // 2,
                        atr_stop_mult=_atr,
                        metrics_by_symbol=web3_metrics,
                        regime_engine=web3_regime,
                    )
                ),
            )
            for ix, dn in enumerate((16, 20, 28))
            for iy, atr in enumerate((2.5, 3.0, 3.8))
        ]
        w2_sim, w2_gates, w2_detail = _evaluate_strategy_full(
            router=router,
            family_id="web3_track_w2_adaptive_trend",
            track_name="track_w_web3",
            strategy_id="web3_w2_adaptive_trend",
            asset_domain="web3",
            baseline_factory=lambda: Web3AdaptiveTrendKernel(
                metrics_by_symbol=web3_metrics, regime_engine=web3_regime
            ),
            grid_factories=w2_grid,
            specs=w2_specs,
            bars_by_symbol=w2_bars,
            risk_cfg=w2_risk,
            dq_reports=[web3_dq_map[s] for s in WEB3_MAJORS_8],
            factor_audit=w3_factor_audit,
            benchmark_returns=bh_w2_12h,
            benchmark_name="WEB3_MAJORS_8_BH",
            periods_per_year=730.0,
            capital_usd=capital_usd,
            pbo_extra_returns=[w1_base_ref.period_returns, w3_base_ref.period_returns],
            purge_bars=6,
            embargo_bars=6,
            precomputed_base_sim=w2_base_ref,
            baseline_grid_label="dn20_atr30",
        )

        w3_grid = [
            (
                ix,
                iy,
                f"h{int(hd)}_cf{int(cf * 10)}",
                {"hold_horizon_days": hd, "cost_floor_multiplier": cf},
                (
                    lambda _hd=hd, _cf=cf: Web3RegimeCarryKernel(
                        hold_horizon_days=_hd,
                        cost_floor_multiplier=_cf,
                        dvol_series=dvol_pts,
                        capital_budget_usd=capital_usd,
                    )
                ),
            )
            for ix, hd in enumerate((5.0, 7.0, 10.0))
            for iy, cf in enumerate((3.0, 3.5, 4.0))
        ]
        w3_sim, w3_gates, w3_detail = _evaluate_strategy_full(
            router=router,
            family_id="web3_track_w3_regime_carry",
            track_name="track_w_web3",
            strategy_id="web3_w3_regime_carry",
            asset_domain="web3",
            baseline_factory=lambda: Web3RegimeCarryKernel(
                dvol_series=dvol_pts, capital_budget_usd=capital_usd
            ),
            grid_factories=w3_grid,
            specs=w3_specs,
            bars_by_symbol=w3_bars,
            risk_cfg=w3_risk,
            dq_reports=[web3_dq_map[s] for s in WEB3_CARRY_4],
            factor_audit=w3_factor_audit,
            benchmark_returns=bh_w3_12h,
            benchmark_name="WEB3_CARRY_4_BH",
            periods_per_year=730.0,
            capital_usd=capital_usd,
            pbo_extra_returns=[w1_base_ref.period_returns, w2_base_ref.period_returns],
            strategy_mode="orthogonal",
            purge_bars=6,
            embargo_bars=6,
            precomputed_base_sim=w3_base_ref,
            baseline_grid_label="h7_cf30",
        )

        # ------------------------------------------------------------------
        # 3. Track G — Gold Dedicated Suite (ZERO Crypto Coupling)
        # ------------------------------------------------------------------
        logger.info("Evaluating Track G (G1 GC=F/GLD, G2 4h A/B/C, G3 PAXG) on real 10y Gold + TIPS + 199 events + PAXG data...")
        gold_macro_engine = GoldMacroEngine(
            macro_bars_by_symbol=gold_macro_bars_1d,
            yield_curve=yield_curve,
        )
        gold_timeline = MacroRegimeTimeline(gold_macro_bars_1d)
        gold_factor_audit_1d = _compute_factor_audit(
            "gold_track_g1_1d",
            {"GC=F": gold_macro_bars_1d["GC=F"]},
            {"GC=F": gold_specs["GC=F"]},
            horizon_bars=10,
            macro_timeline=gold_timeline,
        )
        gold_factor_audit_4h = _compute_factor_audit(
            "gold_track_g2_4h",
            {"GC=F": gc_4h_bars},
            {"GC=F": gold_specs["GC=F"]},
            horizon_bars=8,
            macro_timeline=gold_timeline,
        )
        paxg_factor_audit = _compute_factor_audit(
            "gold_track_g3_paxg_12h",
            {"PAXG": paxg_hl_12h},
            {"PAXG": gold_specs["PAXG"]},
            horizon_bars=8,
            macro_timeline=gold_timeline,
        )

        # Precompute Gold macro context map for daily bars
        gold_1d_ctx = {
            b.ts_event_ms: {"gold_macro": gold_macro_engine.snapshot_as_of(b.ts_avail_ms)}
            for b in gold_macro_bars_1d["GC=F"]
        }
        gold_1d_ctx_provider = lambda ts_ms: gold_1d_ctx.get(
            ts_ms, {"gold_macro": gold_macro_engine.snapshot_as_of(ts_ms)}
        )

        gold_risk_1d = RiskAllocatorConfig(
            max_positions=1,
            target_annual_vol_pct=15.0,
            max_portfolio_leverage=1.15,
            max_single_weight=1.15,
            use_correlation_idm=False,
        )
        bh_gcf_1d = compute_buy_and_hold_returns(gold_macro_bars_1d["GC=F"])
        bh_gld_1d = compute_buy_and_hold_returns(gold_macro_bars_1d["GLD"])

        def _make_g1_grid(sym: str) -> list[tuple[int, int, str, dict[str, Any], Callable[[], AssetNativeStrategy]]]:
            return [
                (
                    ix,
                    iy,
                    f"f{fe}_s{se}",
                    {"symbol": sym, "fast_ema": fe, "slow_ema": se},
                    (
                        lambda _s=sym, _fe=fe, _se=se: GoldTradFiMacroKernel(
                            symbol=_s,
                            macro_engine=gold_macro_engine,
                            fast_ema=_fe,
                            slow_ema=_se,
                        )
                    ),
                )
                for ix, fe in enumerate((8, 12, 18))
                for iy, se in enumerate((34, 42, 55))
            ]

        g1_gcf_sim, g1_gcf_gates, g1_gcf_detail = _evaluate_strategy_full(
            router=router,
            family_id="gold_track_g1_gcf_1d",
            track_name="track_g_gold",
            strategy_id="gold_g1_tradfi_macro_gcf",
            asset_domain="gold",
            baseline_factory=lambda: GoldTradFiMacroKernel(symbol="GC=F", macro_engine=gold_macro_engine),
            grid_factories=_make_g1_grid("GC=F"),
            specs={"GC=F": gold_specs["GC=F"]},
            bars_by_symbol={"GC=F": gold_macro_bars_1d["GC=F"]},
            risk_cfg=gold_risk_1d,
            dq_reports=[dq_gc_1d],
            factor_audit=gold_factor_audit_1d,
            benchmark_returns=bh_gcf_1d,
            benchmark_name="GC_F_10Y_BH",
            periods_per_year=252.0,
            capital_usd=capital_usd,
            context_provider=gold_1d_ctx_provider,
            pbo_extra_returns=[bh_gcf_1d],
            purge_bars=5,
            embargo_bars=5,
            baseline_grid_label="f12_s42",
        )

        g1_gld_sim, g1_gld_gates, g1_gld_detail = _evaluate_strategy_full(
            router=router,
            family_id="gold_track_g1_gld_1d",
            track_name="track_g_gold",
            strategy_id="gold_g1_tradfi_macro_gld",
            asset_domain="gold",
            baseline_factory=lambda: GoldTradFiMacroKernel(symbol="GLD", macro_engine=gold_macro_engine),
            grid_factories=_make_g1_grid("GLD"),
            specs={"GLD": gold_specs["GLD"]},
            bars_by_symbol={"GLD": gold_macro_bars_1d["GLD"]},
            risk_cfg=gold_risk_1d,
            dq_reports=[dq_gld_1d],
            factor_audit=gold_factor_audit_1d,
            benchmark_returns=bh_gld_1d,
            benchmark_name="GLD_10Y_BH",
            periods_per_year=252.0,
            capital_usd=capital_usd,
            context_provider=gold_1d_ctx_provider,
            pbo_extra_returns=[g1_gcf_sim.period_returns, bh_gld_1d],
            purge_bars=5,
            embargo_bars=5,
            baseline_grid_label="f12_s42",
        )

        # G2: Gold 4h Tactical Breakout + PIT Macro Event-Window Engine (Policies A/B/C ablation)
        gold_risk_4h = RiskAllocatorConfig(
            max_positions=1,
            target_annual_vol_pct=15.0,
            max_portfolio_leverage=1.0,
            max_single_weight=1.0,
            use_correlation_idm=False,
        )
        bh_gcf_4h = compute_buy_and_hold_returns(gc_4h_bars)
        gold_4h_ctx = {
            b.ts_event_ms: {"gold_macro": gold_macro_engine.snapshot_as_of(b.ts_avail_ms)}
            for b in gc_4h_bars
        }
        gold_4h_ctx_provider = lambda ts_ms: gold_4h_ctx.get(
            ts_ms, {"gold_macro": gold_macro_engine.snapshot_as_of(ts_ms)}
        )

        # Precompute baseline Policies A/B/C for cross-policy G6 PBO (reused inside _evaluate_strategy_full)
        g2_ref_sims: dict[str, SimulationResult] = {}
        g2_ref_returns: dict[str, list[float]] = {}
        for pol in ("A", "B", "C"):
            ref_sim = acct.simulate(
                Gold4hEventKernel(policy=pol, symbol="GC=F", macro_engine=gold_macro_engine, events=macro_events),
                {"GC=F": gold_specs["GC=F"]},
                {"GC=F": gc_4h_bars},
                risk_cfg=gold_risk_4h,
                capital_usd=capital_usd,
                context_provider=gold_4h_ctx_provider,
            )
            g2_ref_sims[pol] = ref_sim
            g2_ref_returns[pol] = list(ref_sim.period_returns)

        g2_sims: dict[str, SimulationResult] = {}
        g2_details: dict[str, dict[str, Any]] = {}
        g2_gates_map: dict[str, InstitutionalGateReport] = {}
        for pol in ("A", "B", "C"):
            g2_grid = [
                (
                    ix,
                    iy,
                    f"pol{pol}_dn{dn}_ex{ex}",
                    {"policy": pol, "donchian_n": dn, "exit_n": ex},
                    (
                        lambda _pol=pol, _dn=dn, _ex=ex: Gold4hEventKernel(
                            policy=_pol,
                            symbol="GC=F",
                            donchian_n=_dn,
                            exit_n=_ex,
                            macro_engine=gold_macro_engine,
                            events=macro_events,
                        )
                    ),
                )
                for ix, dn in enumerate((10, 12, 16))
                for iy, ex in enumerate((5, 6, 8))
            ]
            other_pol_rets = [g2_ref_returns[p] for p in ("A", "B", "C") if p != pol]
            sim_p, gates_p, det_p = _evaluate_strategy_full(
                router=router,
                family_id=f"gold_track_g2_4h_policy_{pol}",
                track_name="track_g_gold",
                strategy_id=f"gold_g2_4h_event_policy_{pol}",
                asset_domain="gold",
                baseline_factory=lambda _p=pol: Gold4hEventKernel(
                    policy=_p,
                    symbol="GC=F",
                    macro_engine=gold_macro_engine,
                    events=macro_events,
                ),
                grid_factories=g2_grid,
                specs={"GC=F": gold_specs["GC=F"]},
                bars_by_symbol={"GC=F": gc_4h_bars},
                risk_cfg=gold_risk_4h,
                dq_reports=[dq_gc_4h],
                factor_audit=gold_factor_audit_4h,
                benchmark_returns=bh_gcf_4h,
                benchmark_name="GC_F_4H_BH",
                periods_per_year=1512.0,
                capital_usd=capital_usd,
                context_provider=gold_4h_ctx_provider,
                pbo_extra_returns=other_pol_rets,
                purge_bars=12,
                embargo_bars=12,
                precomputed_base_sim=g2_ref_sims[pol],
                baseline_grid_label=f"pol{pol}_dn12_ex6",
            )
            g2_sims[pol] = sim_p
            g2_gates_map[pol] = gates_p
            g2_details[pol] = det_p

        # G3: Dedicated On-Chain PAXG Dual-Basis & Weekend Guard Bridge
        paxg_risk = RiskAllocatorConfig(
            max_positions=1,
            target_annual_vol_pct=18.0,
            max_portfolio_leverage=1.0,
            max_single_weight=1.0,
            use_correlation_idm=False,
        )
        bh_paxg_12h = compute_buy_and_hold_returns(paxg_hl_12h)
        paxg_ctx_provider = _build_paxg_context_lookup(
            paxg_basis_pts, paxg_spot_12h, gold_macro_bars_1d["GC=F"], gold_macro_engine
        )
        g3_grid = [
            (
                ix,
                iy,
                f"f{fe}_s{se}",
                {"fast_ema": fe, "slow_ema": se},
                (
                    lambda _fe=fe, _se=se: GoldPAXGBridgeKernel(
                        symbol="PAXG",
                        macro_engine=gold_macro_engine,
                        fast_ema=_fe,
                        slow_ema=_se,
                    )
                ),
            )
            for ix, fe in enumerate((6, 8, 12))
            for iy, se in enumerate((20, 24, 32))
        ]
        g3_sim, g3_gates, g3_detail = _evaluate_strategy_full(
            router=router,
            family_id="gold_track_g3_paxg_bridge",
            track_name="track_g_gold",
            strategy_id="gold_g3_paxg_bridge",
            asset_domain="gold",
            baseline_factory=lambda: GoldPAXGBridgeKernel(symbol="PAXG", macro_engine=gold_macro_engine),
            grid_factories=g3_grid,
            specs={"PAXG": gold_specs["PAXG"]},
            bars_by_symbol={"PAXG": paxg_hl_12h},
            risk_cfg=paxg_risk,
            dq_reports=[dq_paxg_12h],
            factor_audit=paxg_factor_audit,
            benchmark_returns=bh_paxg_12h,
            benchmark_name="PAXG_HL_12H_BH",
            periods_per_year=730.0,
            capital_usd=capital_usd,
            context_provider=paxg_ctx_provider,
            pbo_extra_returns=[bh_paxg_12h],
            purge_bars=6,
            embargo_bars=6,
            baseline_grid_label="f8_s24",
        )

        # ------------------------------------------------------------------
        # 4. Top-Level Return-Stream Composition (PortfolioCombiner)
        # ------------------------------------------------------------------
        combiner = PortfolioCombiner()
        combined_active_5 = combiner.combine_sleeves(
            {
                "web3_w1_xs_mom_carry": w1_sim,
                "web3_w2_adaptive_trend": w2_sim,
                "gold_g1_gld_macro": g1_gld_sim,
                "gold_g2_4h_policy_a": g2_sims["A"],
                "gold_g3_paxg_bridge": g3_sim,
            },
            method="causal_inv_vol",
            target_vol_pct=15.0,
            capital_usd=capital_usd,
        )
        combined_core = combiner.combine_sleeves(
            {
                "web3_w1_xs_mom_carry": w1_sim,
                "web3_w2_adaptive_trend": w2_sim,
                "web3_w3_regime_carry": w3_sim,
                "gold_g1_gld_macro": g1_gld_sim,
                "gold_g2_4h_policy_a": g2_sims["A"],
                "gold_g3_paxg_bridge": g3_sim,
            },
            method="causal_inv_vol",
            target_vol_pct=15.0,
            capital_usd=capital_usd,
        )

        # Pairwise core showcase: W1 (Web3) + G1_GLD (Gold) return-stream only
        combined_w1_g1 = combiner.combine_sleeves(
            {
                "web3_w1_xs_mom_carry": w1_sim,
                "gold_g1_gld_macro": g1_gld_sim,
            },
            method="causal_inv_vol",
            target_vol_pct=15.0,
            capital_usd=capital_usd,
        )

        elapsed_sec = round(time.perf_counter() - t_start, 3)

        report: dict[str, Any] = {
            "platform_version": "4.0",
            "generated_at_utc": _utc_iso(),
            "elapsed_sec": elapsed_sec,
            "provenance": PROVENANCE_REAL,
            "lake_dir": str(lake_dir),
            "capital_usd": float(capital_usd),
            "dataset_inventory": {
                "web3_12h_symbols": {s: len(web3_bars_12h[s]) for s in WEB3_12_UNIVERSE},
                "web3_dvol_rows": len(dvol_pts),
                "web3_stablecoin_rows": len(stable_pts),
                "web3_metrics_rows": {s: len(rows) for s, rows in web3_metrics.items()},
                "gold_1d_symbols": {s: len(gold_macro_bars_1d[s]) for s in GOLD_MACRO_13},
                "gold_gc_4h_rows": len(gc_4h_bars),
                "gold_paxg_hl_12h_rows": len(paxg_hl_12h),
                "gold_tips_yield_rows": len(yield_curve),
                "gold_macro_events_rows": len(macro_events),
                "gold_paxg_basis_rows": len(paxg_basis_pts),
            },
            "isolation_invariants": {
                "d7_web3_zero_yahoo_vix_bit_identical": bool(d7_zero_vix_bit_identical),
                "d8_gold_single_instrument_separated": True,
                "cross_domain_signal_coupling": False,
            },
            "tracks": {
                "web3": {
                    "W1_xs_mom_carry": w1_detail,
                    "W2_adaptive_trend": w2_detail,
                    "W3_regime_carry": w3_detail,
                },
                "gold": {
                    "G1_tradfi_macro_gcf": g1_gcf_detail,
                    "G1_tradfi_macro_gld": g1_gld_detail,
                    "G2_4h_event_policy_A": g2_details["A"],
                    "G2_4h_event_policy_B": g2_details["B"],
                    "G2_4h_event_policy_C": g2_details["C"],
                    "G3_paxg_bridge": g3_detail,
                },
            },
            "g2_event_ablation_summary": {
                pol: {
                    "cagr_pct": g2_details[pol]["baseline"]["cagr_pct"],
                    "sharpe_ratio": g2_details[pol]["baseline"]["sharpe_ratio"],
                    "max_drawdown_pct": g2_details[pol]["baseline"]["max_drawdown_pct"],
                    "trade_count": g2_details[pol]["baseline"]["trade_count"],
                    "verdict": g2_gates_map[pol].verdict,
                    "passed_gates": g2_gates_map[pol].passed_gates,
                }
                for pol in ("A", "B", "C")
            },
            "portfolio_composition": {
                "active_five_sleeve_causal_inv_vol": combined_active_5.to_dict(),
                "core_w1_plus_g1_gld": combined_w1_g1.to_dict(),
                "six_sleeve_all_diagnostic": combined_core.to_dict(),
            },
            "verdict_summary": {
                "W1_xs_mom_carry": {"verdict": w1_gates.verdict, "passed": w1_gates.passed_gates, "total": 8},
                "W2_adaptive_trend": {"verdict": w2_gates.verdict, "passed": w2_gates.passed_gates, "total": 8},
                "W3_regime_carry": {"verdict": w3_gates.verdict, "passed": w3_gates.passed_gates, "total": 8},
                "G1_tradfi_macro_gcf": {"verdict": g1_gcf_gates.verdict, "passed": g1_gcf_gates.passed_gates, "total": 8},
                "G1_tradfi_macro_gld": {"verdict": g1_gld_gates.verdict, "passed": g1_gld_gates.passed_gates, "total": 8},
                "G2_4h_event_policy_A": {"verdict": g2_gates_map["A"].verdict, "passed": g2_gates_map["A"].passed_gates, "total": 8},
                "G2_4h_event_policy_B": {"verdict": g2_gates_map["B"].verdict, "passed": g2_gates_map["B"].passed_gates, "total": 8},
                "G2_4h_event_policy_C": {"verdict": g2_gates_map["C"].verdict, "passed": g2_gates_map["C"].passed_gates, "total": 8},
                "G3_paxg_bridge": {"verdict": g3_gates.verdict, "passed": g3_gates.passed_gates, "total": 8},
            },
        }

        serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        if report_out is not None:
            report_out.parent.mkdir(parents=True, exist_ok=True)
            report_out.write_text(serialized + "\n", encoding="utf-8")
            logger.info("Wrote strict Platform v4 report to %s (%d bytes)", report_out, len(serialized))
        return report
    finally:
        router.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AlphaLoop Strategy R&D Platform v4 real-data 8-gate evaluation across Track W (Web3) and Track G (Gold)."
    )
    parser.add_argument(
        "--lake-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "platform_v4_lake",
        help="Path to Option B partitioned Platform v4 data lake directory.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=_REPO_ROOT / "data" / "platform_v4_reports" / "platform_v4_report.json",
        help="Output path for strict JSON empirical report (allow_nan=False).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run strictly offline against existing real data lake without network fetches.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh real multi-year public data before evaluation (ignored when --offline is set).",
    )
    parser.add_argument(
        "--capital-usd",
        type=float,
        default=10_000.0,
        help="Initial per-sleeve simulation capital in USD.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.capital_usd) or args.capital_usd <= 0.0:
        logger.error("Invalid --capital-usd: %s", args.capital_usd)
        return 2
    try:
        report = run_platform_v4_evaluation(
            args.lake_dir,
            report_out=args.report_out,
            offline=bool(args.offline or not args.refresh),
            refresh=bool(args.refresh),
            capital_usd=float(args.capital_usd),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Platform v4 evaluation failed closed: %s", exc)
        return 1
    logger.info(
        "Platform v4 evaluation finished in %.2fs | verdicts=%s",
        report["elapsed_sec"],
        json.dumps(report["verdict_summary"], allow_nan=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
