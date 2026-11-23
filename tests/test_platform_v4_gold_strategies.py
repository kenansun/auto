"""Comprehensive Unit, Parity, Isolation & Real-Lake Smoke Tests for Track G (Gold) & Track A (A-Share)."""

from __future__ import annotations

import ast
import bisect
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.platform.assets import (
    ASSET_CLASS_CN_A_SHARE,
    ASSET_CLASS_GOLD_PERP,
    ASSET_CLASS_GOLD_TRADFI,
    ASSET_CLASS_MACRO_INDEX,
    COST_CN_A_SHARE,
    default_ashare_template_spec,
    default_gold_specs,
)
from alphaloop.platform.core_engine.accounting import PortfolioAccountingEngine
from alphaloop.platform.core_engine.interfaces import AssetNativeStrategy, BarDecision
from alphaloop.platform.core_engine.risk_allocator import RiskAllocatorConfig
from alphaloop.platform.core_engine.validation import (
    compute_buy_and_hold_returns,
    evaluate_benchmark_gate,
    verify_full_parity,
)
from alphaloop.platform.data_foundation.models import (
    CanonicalBar,
    MacroEventRelease,
    MacroYieldPoint,
)
from alphaloop.platform.data_foundation.router import DataFoundationRouter
from alphaloop.platform.strategies.ashare_contract import (
    AShareStrategyInterface,
    default_board_price_limit_pct,
)
from alphaloop.platform.strategies.gold import (
    Gold4hEventKernel,
    GoldMacroEngine,
    GoldPAXGBridgeKernel,
    GoldTradFiMacroKernel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_V4_LAKE_DIR = REPO_ROOT / "data" / "platform_v4_lake"
HOUR_MS = 3_600_000
DAY_MS = 86_400_000


def _utc_ms(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> int:
    return int(datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp() * 1000)


def _make_bar(
    *,
    symbol: str,
    venue: str,
    asset_class: str,
    timeframe_h: float,
    ts_event_ms: int,
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    volume_usd: float = 50_000_000.0,
    carry_rate_period: float = 0.0,
    carry_interval_h: float = 1.0,
    contract_multiplier: float = 1.0,
    session_tag: str = "ALL_24_7",
) -> CanonicalBar:
    tf_ms = int(round(timeframe_h * HOUR_MS))
    return CanonicalBar(
        asset_class=asset_class,
        venue=venue,
        symbol=symbol,
        timeframe_h=float(timeframe_h),
        ts_event_ms=int(ts_event_ms),
        ts_avail_ms=int(ts_event_ms + tf_ms),
        open=float(open_px),
        high=float(high_px),
        low=float(low_px),
        close=float(close_px),
        volume_usd=float(volume_usd),
        carry_rate_period=float(carry_rate_period),
        carry_interval_h=float(carry_interval_h),
        spread_bps=1.0,
        provenance=PROVENANCE_REAL,
        source_fingerprint="unit_test_gold",
        session_tag=session_tag,
        contract_multiplier=float(contract_multiplier),
    )


def _build_series(
    symbol: str,
    closes: list[float],
    *,
    venue: str = "yahoo",
    asset_class: str = ASSET_CLASS_GOLD_TRADFI,
    timeframe_h: float = 24.0,
    start_ms: int | None = None,
    carry_rates: list[float] | None = None,
    carry_interval_h: float = 24.0,
    contract_multiplier: float = 1.0,
) -> list[CanonicalBar]:
    base = start_ms if start_ms is not None else _utc_ms(2025, 1, 2, 0, 0)
    tf_ms = int(round(timeframe_h * HOUR_MS))
    out: list[CanonicalBar] = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        hi = max(o, c) * 1.003
        lo = min(o, c) * 0.997
        cr = carry_rates[i] if carry_rates is not None else 0.0
        out.append(
            _make_bar(
                symbol=symbol,
                venue=venue,
                asset_class=asset_class,
                timeframe_h=timeframe_h,
                ts_event_ms=base + i * tf_ms,
                open_px=o,
                high_px=hi,
                low_px=lo,
                close_px=c,
                carry_rate_period=cr,
                carry_interval_h=carry_interval_h,
                contract_multiplier=contract_multiplier,
            )
        )
    return out


def _trend_closes(n: int, start: float, step_ret: float, amp: float = 0.002) -> list[float]:
    vals = [float(start)]
    cur = float(start)
    for i in range(1, n):
        cur = max(1e-3, cur * (1.0 + step_ret + amp * math.sin(i / 4.0)))
        vals.append(cur)
    return vals


def _build_seven_driver_fixture(
    *,
    n_days: int = 150,
    start_ms: int | None = None,
    real_10y_start: float = 2.40,
    real_10y_end: float = 1.35,
    dxy_step: float = -0.0008,
    ief_step: float = 0.0009,
    tip_step: float = 0.0010,
    si_step: float = 0.0022,
    gdx_step: float = 0.0024,
    gvz_level: float = 16.0,
    vix_level: float = 16.0,
    gold_step: float = 0.0018,
) -> tuple[dict[str, list[CanonicalBar]], list[MacroYieldPoint]]:
    t0 = start_ms if start_ms is not None else _utc_ms(2025, 1, 2, 0, 0)
    bars: dict[str, list[CanonicalBar]] = {
        "GC=F": _build_series("GC=F", _trend_closes(n_days, 2100.0, gold_step), start_ms=t0, contract_multiplier=100.0),
        "GLD": _build_series("GLD", _trend_closes(n_days, 195.0, gold_step), start_ms=t0, contract_multiplier=1.0),
        "SI=F": _build_series("SI=F", _trend_closes(n_days, 25.0, si_step), asset_class=ASSET_CLASS_MACRO_INDEX, start_ms=t0),
        "DX-Y.NYB": _build_series("DX-Y.NYB", _trend_closes(n_days, 104.0, dxy_step), asset_class=ASSET_CLASS_MACRO_INDEX, start_ms=t0),
        "IEF": _build_series("IEF", _trend_closes(n_days, 94.0, ief_step), asset_class=ASSET_CLASS_MACRO_INDEX, start_ms=t0),
        "TIP": _build_series("TIP", _trend_closes(n_days, 106.0, tip_step), asset_class=ASSET_CLASS_MACRO_INDEX, start_ms=t0),
        "GDX": _build_series("GDX", _trend_closes(n_days, 31.0, gdx_step), asset_class=ASSET_CLASS_MACRO_INDEX, start_ms=t0),
        "^GVZ": _build_series(
            "^GVZ",
            [gvz_level * (1.0 + 0.02 * math.sin(i / 6.0)) for i in range(n_days)],
            asset_class=ASSET_CLASS_MACRO_INDEX,
            start_ms=t0,
        ),
        "^VIX": _build_series(
            "^VIX",
            [vix_level * (1.0 + 0.02 * math.cos(i / 5.0)) for i in range(n_days)],
            asset_class=ASSET_CLASS_MACRO_INDEX,
            start_ms=t0,
        ),
    }
    yields: list[MacroYieldPoint] = []
    for i in range(n_days):
        frac = i / max(1, n_days - 1)
        r10 = real_10y_start + frac * (real_10y_end - real_10y_start)
        ts_ev = t0 + i * DAY_MS
        yields.append(
            MacroYieldPoint(
                source="us_treasury",
                ts_event_ms=ts_ev,
                ts_avail_ms=ts_ev + DAY_MS,
                real_5y=r10 - 0.15,
                real_7y=r10 - 0.07,
                real_10y=r10,
                real_20y=r10 + 0.12,
                real_30y=r10 + 0.20,
                provenance=PROVENANCE_REAL,
                source_fingerprint="tips_unit",
            )
        )
    return bars, yields


class TestGoldMacroEngineF16(unittest.TestCase):
    """F-16: Stateless 7-Driver PIT GoldMacroEngine tests."""

    def test_stateless_out_of_order_and_future_tail_pit_bit_identity(self) -> None:
        bars, yields = _build_seven_driver_fixture(n_days=160)
        ts_list = [b.ts_avail_ms for b in bars["GC=F"]]
        probes = (15, 60, 110, 159)

        refs = {idx: GoldMacroEngine(macro_bars_by_symbol=bars, yield_curve=yields).snapshot_as_of(ts_list[idx]) for idx in probes}
        reused = GoldMacroEngine(bars_by_symbol=bars, yield_points=yields)

        for idx in (159, 15, 110, 60, 15, 159, 60):
            got = reused.snapshot_as_of(ts_list[idx])
            self.assertEqual(got, refs[idx], f"Out-of-order query drifted at idx={idx}")
        self.assertNotEqual(refs[15], refs[159])

        # Future tail mutation after ts_list[80] must not alter snapshot at ts_list[80]
        ts_q = ts_list[80]
        base_snap = reused.snapshot_as_of(ts_q)
        mut_bars = {sym: list(seq) for sym, seq in bars.items()}
        mut_bars["GC=F"] = [
            _make_bar(
                symbol=b.symbol, venue=b.venue, asset_class=b.asset_class, timeframe_h=24.0,
                ts_event_ms=b.ts_event_ms, open_px=b.open * 3.0, high_px=b.high * 3.0,
                low_px=b.low * 3.0, close_px=b.close * 3.0, contract_multiplier=100.0,
            )
            if b.ts_avail_ms > ts_q
            else b
            for b in bars["GC=F"]
        ]
        mut_yields = [
            MacroYieldPoint(
                y.source, y.ts_event_ms, y.ts_avail_ms, 8.0, 8.0, 8.0, 8.0, 8.0, y.provenance, y.source_fingerprint
            )
            if y.ts_avail_ms > ts_q
            else y
            for y in yields
        ]
        mut_snap = GoldMacroEngine(mut_bars, yield_curve=mut_yields).snapshot_as_of(ts_q)
        self.assertEqual(base_snap, mut_snap)
        # Strict JSON serializable with zero NaN/Inf
        encoded = json.dumps(base_snap, allow_nan=False)
        self.assertIn("macro_gold_composite", encoded)

    def test_individual_sensitivity_all_seven_drivers_and_safe_haven_vix(self) -> None:
        def _snap(**kw: Any) -> dict[str, Any]:
            b_map, y_pts = _build_seven_driver_fixture(n_days=150, **kw)
            eng = GoldMacroEngine(macro_bars_by_symbol=b_map, yield_curve=y_pts)
            return eng.snapshot_as_of(b_map["GC=F"][-1].ts_avail_ms)

        # 1. US Treasury 10Y Real Yield
        s_ry_fall = _snap(real_10y_start=2.6, real_10y_end=1.2)
        s_ry_rise = _snap(real_10y_start=1.2, real_10y_end=2.8)
        self.assertGreater(s_ry_fall["macro_gold_composite"], s_ry_rise["macro_gold_composite"])
        self.assertGreater(s_ry_fall["real_yield_score"], s_ry_rise["real_yield_score"])

        # 2. QuantPedia Gold x IEF & TIP
        s_bond_up = _snap(ief_step=0.0025, tip_step=0.0025)
        s_bond_dn = _snap(ief_step=-0.0025, tip_step=-0.0025)
        self.assertGreater(s_bond_up["macro_gold_composite"], s_bond_dn["macro_gold_composite"])
        self.assertGreater(s_bond_up["bond_inflation_score"], s_bond_dn["bond_inflation_score"])

        # 3. Dollar trend (DX-Y.NYB)
        s_dxy_fall = _snap(dxy_step=-0.0025)
        s_dxy_rise = _snap(dxy_step=0.0025)
        self.assertGreater(s_dxy_fall["macro_gold_composite"], s_dxy_rise["macro_gold_composite"])
        self.assertGreater(s_dxy_fall["dxy_score"], s_dxy_rise["dxy_score"])

        # 4. Gold/Silver ratio (SI=F)
        s_si_up = _snap(si_step=0.0040)
        s_si_dn = _snap(si_step=-0.0030)
        self.assertGreater(s_si_up["gsr_score"], s_si_dn["gsr_score"])
        self.assertNotEqual(s_si_up["macro_gold_composite"], s_si_dn["macro_gold_composite"])

        # 5. Gold Miners relative strength (GDX/GLD)
        s_gdx_up = _snap(gdx_step=0.0040)
        s_gdx_dn = _snap(gdx_step=-0.0030)
        self.assertGreater(s_gdx_up["miners_rs_score"], s_gdx_dn["miners_rs_score"])
        self.assertNotEqual(s_gdx_up["macro_gold_composite"], s_gdx_dn["macro_gold_composite"])

        # 6. ^GVZ Gold vol level & exhaustion
        s_gvz_calm = _snap(gvz_level=14.0)
        s_gvz_ext = _snap(gvz_level=42.0)
        self.assertGreater(s_gvz_calm["gvz_score"], s_gvz_ext["gvz_score"])
        self.assertGreater(s_gvz_calm["exposure_scalar"], s_gvz_ext["exposure_scalar"])
        self.assertNotEqual(s_gvz_calm["macro_gold_composite"], s_gvz_ext["macro_gold_composite"])

        # 7. Safe-haven asymmetric ^VIX response (safe-haven bid vs liquidation dash-for-cash)
        b_safe, y_safe = _build_seven_driver_fixture(
            n_days=150, real_10y_start=2.5, real_10y_end=1.3, dxy_step=-0.0015, gold_step=0.0025, vix_level=15.0
        )
        b_liq, y_liq = _build_seven_driver_fixture(
            n_days=150, real_10y_start=1.3, real_10y_end=2.8, dxy_step=0.0015, gold_step=-0.0025, vix_level=15.0
        )
        for bm in (b_safe, b_liq):
            bm["^VIX"] = [
                _make_bar(
                    symbol="^VIX", venue="yahoo", asset_class=ASSET_CLASS_MACRO_INDEX, timeframe_h=24.0,
                    ts_event_ms=b.ts_event_ms, open_px=42.0, high_px=43.0, low_px=41.0, close_px=42.0,
                )
                if i >= 140
                else b
                for i, b in enumerate(bm["^VIX"])
            ]
        snap_safe = GoldMacroEngine(b_safe, yield_curve=y_safe).snapshot_as_of(b_safe["GC=F"][-1].ts_avail_ms)
        snap_liq = GoldMacroEngine(b_liq, yield_curve=y_liq).snapshot_as_of(b_liq["GC=F"][-1].ts_avail_ms)
        self.assertGreater(snap_safe["vix_safe_haven_score"], 0.0)
        self.assertLess(snap_liq["vix_safe_haven_score"], 0.0)
        self.assertGreater(snap_safe["macro_gold_composite"], 0.0)
        self.assertGreater(snap_safe["macro_gold_composite"], snap_liq["macro_gold_composite"])


class TestGoldG1TradFiMacroCoreF17(unittest.TestCase):
    """F-17: G1 Single-Instrument TradFi Gold Macro-Regime Core tests."""

    def test_single_leg_gc_f_and_gld_plus_d8_double_count_guard(self) -> None:
        bars, yields = _build_seven_driver_fixture(n_days=120, gold_step=0.002)
        me = GoldMacroEngine(bars, yield_curve=yields)

        # Reject composite symbol string at init
        with self.assertRaises(ValueError):
            GoldTradFiMacroKernel(symbol="GC=F+GLD")

        # Separate GC=F and GLD execution
        for sym in ("GC=F", "GLD"):
            g1 = GoldTradFiMacroKernel(symbol=sym, macro_engine=me)
            self.assertIsInstance(g1, AssetNativeStrategy)
            self.assertEqual(g1.asset_domain, "gold")
            g1.reset()
            last_w = 0.0
            for b in bars[sym]:
                dec = g1.on_bar_step(b.ts_event_ms, {sym: b}, {"gold_macro": me.snapshot_as_of(b.ts_avail_ms)})
                last_w = dec[sym].target_weight
                g1.sync_executed_weights({sym: last_w})
            self.assertGreater(last_w, 0.0)

        # Simultaneous GC=F + GLD input -> strictly <= 1 active gold leg (exactly 1 active)
        g1_cap = GoldTradFiMacroKernel(symbol="GC=F", macro_engine=me)
        g1_cap.reset()
        max_active = 0
        for i in range(120):
            dec = g1_cap.on_bar_step(
                bars["GC=F"][i].ts_event_ms,
                {"GC=F": bars["GC=F"][i], "GLD": bars["GLD"][i]},
                {"gold_macro": {"macro_gold_composite": 0.9}},
            )
            active = sum(1 for s in ("GC=F", "GLD") if abs(dec[s].target_weight) > 1e-9)
            max_active = max(max_active, active)
            self.assertEqual(dec["GLD"].target_weight, 0.0)
            g1_cap.sync_executed_weights({s: d.target_weight for s, d in dec.items()})
        self.assertEqual(max_active, 1)

        # Strict rejection mode raises ValueError
        g1_strict = GoldTradFiMacroKernel(symbol="GC=F", reject_multi_gold_input=True)
        with self.assertRaises(ValueError):
            g1_strict.on_bar_step(
                bars["GC=F"][0].ts_event_ms,
                {"GC=F": bars["GC=F"][0], "GLD": bars["GLD"][0]},
            )

    def test_asymmetric_scaling_short_suppression_g7_parity_and_g8_crash_avoidance(self) -> None:
        # Asymmetric bull vs bear scaling
        bull_b, bull_y = _build_seven_driver_fixture(
            n_days=130, real_10y_start=2.6, real_10y_end=1.3, dxy_step=-0.0015, ief_step=0.0015, gold_step=0.002
        )
        bear_b, bear_y = _build_seven_driver_fixture(
            n_days=130, real_10y_start=1.3, real_10y_end=2.8, dxy_step=0.0015, ief_step=-0.0015, gold_step=0.002
        )

        def _last_w(bm: dict[str, list[CanonicalBar]], yp: list[MacroYieldPoint]) -> float:
            me = GoldMacroEngine(bm, yield_curve=yp)
            g1 = GoldTradFiMacroKernel(symbol="GC=F", macro_engine=me)
            g1.reset()
            w = 0.0
            for b in bm["GC=F"]:
                d = g1.on_bar_step(b.ts_event_ms, {"GC=F": b}, {"gold_macro": me.snapshot_as_of(b.ts_avail_ms)})
                w = d["GC=F"].target_weight
                g1.sync_executed_weights({"GC=F": w})
            return w

        w_bull = _last_w(bull_b, bull_y)
        w_bear = _last_w(bear_b, bear_y)
        self.assertGreater(w_bull, w_bear)

        # Counter-macro short suppression on pullback during falling-real-yield bull
        closes = [2000.0 + 3.0 * i for i in range(80)] + [2240.0 - 15.0 * k for k in range(1, 18)]
        dip_bars = _build_series("GC=F", closes, contract_multiplier=100.0)

        def _min_dip_w(score: float) -> float:
            g1 = GoldTradFiMacroKernel(symbol="GC=F")
            g1.reset()
            mn = 1.0
            for idx, b in enumerate(dip_bars):
                d = g1.on_bar_step(b.ts_event_ms, {"GC=F": b}, {"gold_macro": {"macro_gold_composite": score}})
                w = d["GC=F"].target_weight
                if idx >= 80:
                    mn = min(mn, w)
                g1.sync_executed_weights({"GC=F": w})
            return mn

        min_bull = _min_dip_w(+1.1)
        min_bear = _min_dip_w(-1.1)
        self.assertGreaterEqual(min_bull, 0.0)
        self.assertGreater(min_bull, min_bear)

        # G7 full parity + G8 crash avoidance
        crash_closes = [2000.0]
        for i in range(1, 260):
            r = 0.002 if i < 160 else (-0.0075 if i < 220 else 0.001)
            crash_closes.append(crash_closes[-1] * (1.0 + r))
        crash_bars = _build_series("GC=F", crash_closes, contract_multiplier=100.0)
        ts_cut_1 = crash_bars[160].ts_event_ms
        ts_cut_2 = crash_bars[220].ts_event_ms

        def _ctx_crash(ts: int) -> dict[str, Any]:
            if ts < ts_cut_1:
                return {"gold_macro": {"macro_gold_composite": 1.1}}
            if ts < ts_cut_2:
                return {"gold_macro": {"macro_gold_composite": -1.3}}
            return {"gold_macro": {"macro_gold_composite": 0.0}}

        cfg = RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=15.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
        specs = {"GC=F": default_gold_specs()["GC=F"]}
        g1_sim = GoldTradFiMacroKernel(symbol="GC=F")
        res = PortfolioAccountingEngine().simulate(g1_sim, specs, {"GC=F": crash_bars}, risk_cfg=cfg, context_provider=_ctx_crash)
        self.assertGreater(res.trade_count, 0)

        bh_rets = compute_buy_and_hold_returns(crash_bars, timestamps_ms=res.period_timestamps_ms)
        g8 = evaluate_benchmark_gate(res.period_returns, bh_rets, periods_per_year=252.0)
        self.assertTrue(g8.passed_benchmark_gate, f"G1 crash avoidance must beat Gold B&H: {g8.detail}")

        parity = verify_full_parity(g1_sim, specs, {"GC=F": crash_bars}, risk_cfg=cfg, batch_result=res, context_provider=_ctx_crash)
        self.assertTrue(parity.passed, parity.detail)
        self.assertLessEqual(parity.max_abs_weight_delta, 1e-10)
        self.assertLessEqual(parity.max_abs_equity_delta, 1e-8)


class TestGoldG2SessionEvent4hF18(unittest.TestCase):
    """F-18: G2 4h Tactical Breakout + PIT Event-Window Policies A/B/C tests."""

    def test_policies_abc_ablation_and_pre_event_whipsaw_shield(self) -> None:
        t_ev = _utc_ms(2026, 12, 9, 19, 0)
        start_ms = t_ev - 40 * 4 * HOUR_MS
        closes = [2700.0 + 0.5 * math.sin(i / 2.0) for i in range(39)]
        # Bar 39 (T-4h): fakeout breakout; Bar 40 (T): plunge; Bars 41..47: post-event rally
        closes.extend([2724.0, 2670.0, 2732.0, 2758.0, 2782.0, 2805.0, 2825.0, 2840.0])
        bars = _build_series("GC=F", closes, timeframe_h=4.0, start_ms=start_ms, contract_multiplier=100.0)
        ev = MacroEventRelease("FOMC", "FOMC-2026-12-09", "2026-12", t_ev, t_ev, True, 4.25, 4.50, -1.2, "fed", PROVENANCE_REAL)

        spec = {"GC=F": default_gold_specs()["GC=F"]}
        cfg = RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=18.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
        ctx_fn = lambda ts: {"macro_events": [ev], "gold_macro": {"macro_gold_composite": 0.9}, "gvz_z": 1.4}

        engine = PortfolioAccountingEngine()
        results: dict[str, Any] = {}
        for pol in ("A", "B", "C"):
            strat = Gold4hEventKernel(policy=pol)
            self.assertIsInstance(strat, AssetNativeStrategy)
            self.assertEqual(strat.asset_domain, "gold")
            self.assertEqual(strat.strategy_id, f"gold_g2_4h_event_policy_{pol}")
            res = engine.simulate(strat, spec, {"GC=F": bars}, risk_cfg=cfg, context_provider=ctx_fn)
            results[pol] = res
            self.assertGreater(res.trade_count, 0)
            parity = verify_full_parity(strat, spec, {"GC=F": bars}, risk_cfg=cfg, batch_result=res, context_provider=ctx_fn)
            self.assertTrue(parity.passed, f"Policy {pol} parity failed: {parity.detail}")

        # Policy A enters on T-4h fakeout (executed at Bar 40 open); Policies B & C stay flat into T!
        self.assertGreater(abs(results["A"].executed_weights_history[40]["GC=F"]), 0.0)
        self.assertEqual(results["B"].executed_weights_history[40]["GC=F"], 0.0)
        self.assertEqual(results["C"].executed_weights_history[40]["GC=F"], 0.0)
        self.assertGreater(results["B"].net_pnl_usd, results["A"].net_pnl_usd)
        self.assertGreater(results["C"].net_pnl_usd, results["A"].net_pnl_usd)

        # Policy C activates post-event expansion inside [T, T+8h] with strict-JSON metadata
        post_c = results["C"].raw_decisions_history[41]["GC=F"]
        self.assertGreater(abs(post_c.target_weight), 0.0)
        self.assertEqual(post_c.metadata["event_window_state"], "POST_EVENT_EXPANSION")
        json.dumps(post_c.metadata, allow_nan=False)

    def test_zero_pre_release_actual_or_surprise_leakage_and_session_tags(self) -> None:
        t_ev = _utc_ms(2026, 12, 10, 13, 30)
        start_ms = t_ev - 35 * 4 * HOUR_MS
        closes = [2680.0 + 0.5 * math.sin(i / 2.0) for i in range(25)] + [2685.0 + 3.0 * k for k in range(15)]
        bars = _build_series("GC=F", closes, timeframe_h=4.0, start_ms=start_ms, contract_multiplier=100.0)

        ev_pos = MacroEventRelease("CPI", "CPI-DEC", "2026-11", t_ev, t_ev, False, 5.5, 2.5, +8.0, "bls", PROVENANCE_REAL)
        ev_neg = MacroEventRelease("CPI", "CPI-DEC", "2026-11", t_ev, t_ev, False, 0.5, 2.5, -8.0, "bls", PROVENANCE_REAL)

        c1 = Gold4hEventKernel(policy="C")
        c2 = Gold4hEventKernel(policy="C")
        c1.reset()
        c2.reset()
        post_active = 0.0
        for b in bars:
            d1 = c1.on_bar_step(b.ts_event_ms, {"GC=F": b}, {"macro_events": [ev_pos], "gold_macro": {"macro_gold_composite": 0.6}})
            d2 = c2.on_bar_step(b.ts_event_ms, {"GC=F": b}, {"macro_events": [ev_neg], "gold_macro": {"macro_gold_composite": 0.6}})
            c1.sync_executed_weights({"GC=F": d1["GC=F"].target_weight})
            c2.sync_executed_weights({"GC=F": d2["GC=F"].target_weight})
            if b.ts_event_ms < t_ev:
                self.assertEqual(d1["GC=F"].target_weight, d2["GC=F"].target_weight)
                self.assertEqual(d1["GC=F"].raw_signal, d2["GC=F"].raw_signal)
            else:
                post_active = max(post_active, abs(d1["GC=F"].target_weight))
        self.assertGreater(post_active, 0.0)

        # Distinct session tags + zero flat-bar fade (anti-tombstone)
        start_0000 = _utc_ms(2026, 11, 10, 0, 0)
        flat_then_break = [2700.0] * 30 + [2720.0, 2735.0, 2750.0]
        s_bars = _build_series("GC=F", flat_then_break, timeframe_h=4.0, start_ms=start_0000, contract_multiplier=100.0)
        g2 = Gold4hEventKernel(policy="A")
        g2.reset()
        tags_by_hour: dict[int, set[str]] = {}
        for idx, b in enumerate(s_bars):
            dec = g2.on_bar_step(b.ts_event_ms, {"GC=F": b}, {"gold_macro": {"macro_gold_composite": 0.0}})
            hr = (b.ts_event_ms // HOUR_MS) % 24
            tags_by_hour.setdefault(hr, set()).add(dec["GC=F"].metadata["session_tag"])
            if idx < 30:
                self.assertEqual(dec["GC=F"].target_weight, 0.0)
            g2.sync_executed_weights({"GC=F": dec["GC=F"].target_weight})
        self.assertEqual(tags_by_hour[12], {"LONDON_NY"})
        self.assertEqual(tags_by_hour[0], {"ASIAN"})
        self.assertEqual(tags_by_hour[20], {"OFF_HOURS"})
        self.assertGreater(dec["GC=F"].target_weight, 0.0)


class TestGoldG3PAXGBridgeF19(unittest.TestCase):
    """F-19: G3 PAXG Dual-Basis, Own-Bar Stop, Weekend Guard & Funding Trim tests."""

    def test_dual_basis_exact_bps_and_own_bar_stop_independence(self) -> None:
        g3 = GoldPAXGBridgeKernel()
        self.assertIsInstance(g3, AssetNativeStrategy)
        self.assertEqual(g3.asset_domain, "gold")
        g3.reset()

        # Dual basis exact bps across two distinct tuples
        t0 = _utc_ms(2026, 11, 16, 0, 0)  # Monday
        b1 = _make_bar(
            symbol="PAXG", venue="hyperliquid", asset_class=ASSET_CLASS_GOLD_PERP, timeframe_h=12.0,
            ts_event_ms=t0, open_px=2690.0, high_px=2695.0, low_px=2685.0, close_px=2690.0,
            carry_rate_period=0.00002, carry_interval_h=12.0,
        )
        d1 = g3.on_bar_step(t0, {"PAXG": b1}, {"paxg_spot_price": 2685.0, "comex_price": 2700.0})
        self.assertAlmostEqual(d1["PAXG"].metadata["perp_spot_basis_bps"], (2690.0 / 2685.0 - 1.0) * 10_000.0, delta=0.05)
        self.assertAlmostEqual(d1["PAXG"].metadata["spot_comex_basis_bps"], (2685.0 / 2700.0 - 1.0) * 10_000.0, delta=0.05)

        b2 = _make_bar(
            symbol="PAXG", venue="hyperliquid", asset_class=ASSET_CLASS_GOLD_PERP, timeframe_h=12.0,
            ts_event_ms=t0 + 12 * HOUR_MS, open_px=2670.0, high_px=2675.0, low_px=2665.0, close_px=2670.0,
            carry_rate_period=0.00002, carry_interval_h=12.0,
        )
        d2 = g3.on_bar_step(b2.ts_event_ms, {"PAXG": b2}, {"paxg_spot_price": 2680.0, "comex_price": 2660.0})
        self.assertAlmostEqual(d2["PAXG"].metadata["perp_spot_basis_bps"], (2670.0 / 2680.0 - 1.0) * 10_000.0, delta=0.05)
        self.assertAlmostEqual(d2["PAXG"].metadata["spot_comex_basis_bps"], (2680.0 / 2660.0 - 1.0) * 10_000.0, delta=0.05)

        # Own-bar stop independence from GC=F
        ramp = _build_series(
            "PAXG", [2600.0 + 2.5 * i for i in range(41)], venue="hyperliquid",
            asset_class=ASSET_CLASS_GOLD_PERP, timeframe_h=12.0, start_ms=_utc_ms(2026, 10, 1, 0, 0),
            carry_rates=[0.00002] * 41, carry_interval_h=12.0,
        )

        def _prepped() -> GoldPAXGBridgeKernel:
            k = GoldPAXGBridgeKernel()
            k.reset()
            for b in ramp:
                dd = k.on_bar_step(b.ts_event_ms, {"PAXG": b}, {"paxg_spot_price": b.close, "comex_price": b.close, "gold_macro": {"macro_gold_composite": 0.8}})
                k.sync_executed_weights({"PAXG": dd["PAXG"].target_weight})
            return k

        nxt = ramp[-1].ts_event_ms + 12 * HOUR_MS
        # GC=F crashes to 2400 while PAXG holds 2700 -> MUST NOT stop out!
        k_hold = _prepped()
        bar_hold = _make_bar(
            symbol="PAXG", venue="hyperliquid", asset_class=ASSET_CLASS_GOLD_PERP, timeframe_h=12.0,
            ts_event_ms=nxt, open_px=2700.0, high_px=2705.0, low_px=2695.0, close_px=2700.0,
            carry_rate_period=0.00002, carry_interval_h=12.0,
        )
        d_hold = k_hold.on_bar_step(nxt, {"PAXG": bar_hold}, {"paxg_spot_price": 2700.0, "comex_price": 2400.0, "gold_macro": {"macro_gold_composite": 0.8}})
        self.assertGreater(d_hold["PAXG"].target_weight, 0.0)

        # PAXG itself drops to 2450 while GC=F holds 2700 -> MUST stop out!
        k_stop = _prepped()
        bar_drop = _make_bar(
            symbol="PAXG", venue="hyperliquid", asset_class=ASSET_CLASS_GOLD_PERP, timeframe_h=12.0,
            ts_event_ms=nxt, open_px=2690.0, high_px=2690.0, low_px=2450.0, close_px=2460.0,
            carry_rate_period=0.00002, carry_interval_h=12.0,
        )
        d_stop = k_stop.on_bar_step(nxt, {"PAXG": bar_drop}, {"paxg_spot_price": 2460.0, "comex_price": 2700.0, "gold_macro": {"macro_gold_composite": 0.8}})
        self.assertEqual(d_stop["PAXG"].target_weight, 0.0)
        self.assertTrue(d_stop["PAXG"].metadata["own_bar_stop_triggered"])

    def test_weekend_guard_frozen_comex_plus50pct_inertia_dislocation_and_funding_trim(self) -> None:
        fri_18 = _utc_ms(2026, 11, 20, 18, 0)
        sat_14 = _utc_ms(2026, 11, 21, 14, 0)
        wed_12 = _utc_ms(2026, 11, 18, 12, 0)
        sat_12 = _utc_ms(2026, 11, 21, 12, 0)

        # 1. Weekend >2.5 sigma dislocation blocks fresh entry from flat; own-bar stop still works on existing long
        warm = _build_series(
            "PAXG", [2680.0] * 40, venue="hyperliquid", asset_class=ASSET_CLASS_GOLD_PERP,
            timeframe_h=12.0, start_ms=fri_18 - 39 * 12 * HOUR_MS,
            carry_rates=[0.00002] * 40, carry_interval_h=12.0,
        )
        g3 = GoldPAXGBridgeKernel()
        g3.reset()
        for b in warm:
            d = g3.on_bar_step(b.ts_event_ms, {"PAXG": b}, {"paxg_spot_price": 2680.0, "comex_price": 2680.0, "basis_z": 0.1})
            g3.sync_executed_weights({"PAXG": d["PAXG"].target_weight})

        sat_break = _make_bar(
            symbol="PAXG", venue="hyperliquid", asset_class=ASSET_CLASS_GOLD_PERP, timeframe_h=12.0,
            ts_event_ms=sat_14, open_px=2740.0, high_px=2760.0, low_px=2735.0, close_px=2755.0,
            carry_rate_period=0.00002, carry_interval_h=12.0,
        )
        d_sat_block = g3.on_bar_step(sat_14, {"PAXG": sat_break}, {"paxg_spot_price": 2755.0, "comex_price": 2680.0, "basis_z": 3.5, "gold_macro": {"macro_gold_composite": 0.9}})
        self.assertEqual(d_sat_block["PAXG"].target_weight, 0.0)
        self.assertTrue(d_sat_block["PAXG"].metadata["is_weekend_guard"])
        self.assertTrue(d_sat_block["PAXG"].metadata["weekend_dislocation_blocked"])

        # 2. +50% wider weekend inertia buffer verified via adaptive perturbation scan (Wed vs Sat)
        def _max_held_frac(end_ts: int) -> float:
            ramp = _build_series(
                "PAXG", [2600.0 + 2.5 * i for i in range(40)], venue="hyperliquid",
                asset_class=ASSET_CLASS_GOLD_PERP, timeframe_h=12.0, start_ms=end_ts - 39 * 12 * HOUR_MS,
                carry_rates=[0.00002] * 40, carry_interval_h=12.0,
            )
            nxt_ts = end_ts + 12 * HOUR_MS
            last_px = ramp[-1].close
            flat_nxt = _make_bar(
                symbol="PAXG", venue="hyperliquid", asset_class=ASSET_CLASS_GOLD_PERP, timeframe_h=12.0,
                ts_event_ms=nxt_ts, open_px=last_px, high_px=last_px * 1.001, low_px=last_px * 0.999, close_px=last_px,
                carry_rate_period=0.00002, carry_interval_h=12.0,
            )
            max_held = 0.0
            for k_pct in range(1, 35):
                frac = k_pct / 100.0
                kern = GoldPAXGBridgeKernel()
                kern.reset()
                w_des = 0.0
                for b in ramp:
                    dec = kern.on_bar_step(b.ts_event_ms, {"PAXG": b}, {"paxg_spot_price": b.close, "comex_price": b.close, "basis_z": 0.1, "gold_macro": {"macro_gold_composite": 0.8}})
                    w_des = dec["PAXG"].target_weight
                    kern.sync_executed_weights({"PAXG": w_des})
                self.assertGreater(w_des, 0.05)
                synced_w = w_des * (1.0 - frac)
                kern.sync_executed_weights({"PAXG": synced_w})
                dec_nxt = kern.on_bar_step(nxt_ts, {"PAXG": flat_nxt}, {"paxg_spot_price": last_px, "comex_price": last_px, "basis_z": 0.1, "gold_macro": {"macro_gold_composite": 0.8}})
                if abs(dec_nxt["PAXG"].target_weight - synced_w) <= 1e-6:
                    max_held = frac
            return max_held

        held_wed = _max_held_frac(wed_12)
        held_sat = _max_held_frac(sat_12)
        self.assertGreater(held_wed, 0.0)
        self.assertGreater(held_sat, held_wed)
        self.assertLess(held_sat, 0.40)

        # 3. High funding >25% APR trims long perp exposure (with proper carry_interval_h=12.0)
        ramp_px = [2600.0 + 2.5 * i for i in range(45)]
        high_r = (0.60 / 8760.0) * 12.0

        def _final_w(rate: float) -> float:
            bs = _build_series(
                "PAXG", ramp_px, venue="hyperliquid", asset_class=ASSET_CLASS_GOLD_PERP,
                timeframe_h=12.0, start_ms=_utc_ms(2026, 10, 1, 0, 0),
                carry_rates=[rate] * len(ramp_px), carry_interval_h=12.0,
            )
            k = GoldPAXGBridgeKernel()
            k.reset()
            w = 0.0
            for b in bs:
                d = k.on_bar_step(b.ts_event_ms, {"PAXG": b}, {"paxg_spot_price": b.close, "comex_price": b.close, "gold_macro": {"macro_gold_composite": 0.8}})
                w = d["PAXG"].target_weight
                k.sync_executed_weights({"PAXG": w})
            return w

        w_zero = _final_w(0.0)
        w_high = _final_w(high_r)
        self.assertGreater(w_zero, 0.0)
        self.assertLess(w_high, w_zero)

        # 4. G7 parity check
        paxg_bars = _build_series(
            "PAXG", _trend_closes(60, 2650.0, 0.002), venue="hyperliquid",
            asset_class=ASSET_CLASS_GOLD_PERP, timeframe_h=12.0,
            start_ms=_utc_ms(2026, 9, 1, 0, 0), carry_rates=[0.00005] * 60, carry_interval_h=12.0,
        )
        by_ts_px = {b.ts_event_ms: b.close for b in paxg_bars}
        specs = {"PAXG": default_gold_specs()["PAXG"]}
        cfg = RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=18.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
        parity = verify_full_parity(
            GoldPAXGBridgeKernel(),
            specs,
            {"PAXG": paxg_bars},
            risk_cfg=cfg,
            context_provider=lambda ts: {"paxg_spot_price": by_ts_px[ts], "comex_price": by_ts_px[ts], "gold_macro": {"macro_gold_composite": 0.7}},
        )
        self.assertTrue(parity.passed, parity.detail)


class TestAShareContractF20AndDomainIsolation(unittest.TestCase):
    """F-20: Track A AShareStrategyInterface & Strict AST Domain Isolation tests."""

    def test_ashare_long_only_t_plus_1_limits_stamp_duty_hooks_and_parity(self) -> None:
        inst = AShareStrategyInterface(daily_price_limit_pct=10.0)
        self.assertIsInstance(inst, AssetNativeStrategy)
        self.assertEqual(inst.asset_domain, "ashare")
        self.assertIsNot(inst.clone_fresh(), inst)

        # Board prefix limit inference
        self.assertEqual(default_board_price_limit_pct("600519.SH"), 10.0)
        self.assertEqual(default_board_price_limit_pct("300750.SZ"), 20.0)
        self.assertEqual(default_board_price_limit_pct("688981.SH"), 20.0)

        # Sell-only stamp duty (5bp sell, 0bp buy)
        self.assertEqual(inst.estimate_stamp_duty_usd(100_000.0, is_sell=False), 0.0)
        self.assertAlmostEqual(inst.estimate_stamp_duty_usd(100_000.0, is_sell=True), 50.0, places=6)
        self.assertAlmostEqual(inst.estimate_one_way_cost_bps(is_sell=False), COST_CN_A_SHARE.one_way_cost_bps("TAKER_TAKER", is_sell=False), places=6)
        self.assertAlmostEqual(inst.estimate_one_way_cost_bps(is_sell=True), COST_CN_A_SHARE.one_way_cost_bps("TAKER_TAKER", is_sell=True), places=6)

        # Build 2 days x 4 intraday 1h CST bars (01:30, 02:30, 05:00, 06:00 UTC)
        bars: list[CanonicalBar] = []
        for d_off in range(2):
            for hh, mm in ((1, 30), (2, 30), (5, 0), (6, 0)):
                ts = _utc_ms(2026, 11, 23 + d_off, hh, mm)
                bars.append(
                    _make_bar(
                        symbol="600519.SH", venue="sse", asset_class=ASSET_CLASS_CN_A_SHARE,
                        timeframe_h=1.0, ts_event_ms=ts, open_px=100.0, high_px=100.5, low_px=99.5, close_px=100.0,
                    )
                )

        # Negative signal -> strictly clamped to 0.0
        inst.reset()
        d_neg = inst.on_bar_step(bars[0].ts_event_ms, {"600519.SH": bars[0]}, {"raw_signal": -5.0, "desired_weight": -0.7})
        self.assertEqual(d_neg["600519.SH"].target_weight, 0.0)

        # T+1 same-day sell lock -> unlocked next CST date
        inst.reset()
        d0 = inst.on_bar_step(bars[0].ts_event_ms, {"600519.SH": bars[0]}, {"raw_signal": 5.0, "desired_weight": 0.6})
        self.assertGreater(d0["600519.SH"].target_weight, 0.0)
        inst.sync_executed_weights({"600519.SH": d0["600519.SH"].target_weight})

        for b in bars[1:4]:
            d_same = inst.on_bar_step(b.ts_event_ms, {"600519.SH": b}, {"raw_signal": -5.0, "desired_weight": 0.0})
            self.assertAlmostEqual(d_same["600519.SH"].target_weight, 0.6, places=6)
            self.assertTrue(d_same["600519.SH"].metadata["t_plus_1_locked"])
            inst.sync_executed_weights({"600519.SH": d_same["600519.SH"].target_weight})

        d_next = inst.on_bar_step(bars[4].ts_event_ms, {"600519.SH": bars[4]}, {"raw_signal": -5.0, "desired_weight": 0.0})
        self.assertEqual(d_next["600519.SH"].target_weight, 0.0)
        self.assertFalse(d_next["600519.SH"].metadata["t_plus_1_locked"])

        # Mainboard ±10% limit-up blocks buy AND limit-down blocks sell; ChiNext ±20% allows +12%, blocks +20%
        b_d1 = _make_bar(symbol="600519.SH", venue="sse", asset_class=ASSET_CLASS_CN_A_SHARE, timeframe_h=24.0, ts_event_ms=_utc_ms(2026, 11, 23, 6, 0), open_px=100.0, high_px=100.0, low_px=100.0, close_px=100.0)
        b_up = _make_bar(symbol="600519.SH", venue="sse", asset_class=ASSET_CLASS_CN_A_SHARE, timeframe_h=24.0, ts_event_ms=_utc_ms(2026, 11, 24, 6, 0), open_px=110.0, high_px=110.0, low_px=110.0, close_px=110.0)
        b_dn = _make_bar(symbol="600519.SH", venue="sse", asset_class=ASSET_CLASS_CN_A_SHARE, timeframe_h=24.0, ts_event_ms=_utc_ms(2026, 11, 24, 6, 0), open_px=90.0, high_px=90.0, low_px=90.0, close_px=90.0)

        inst.reset()
        inst.on_bar_step(b_d1.ts_event_ms, {"600519.SH": b_d1}, {"desired_weight": 0.0, "prev_close": 100.0})
        inst.sync_executed_weights({"600519.SH": 0.0})
        du = inst.on_bar_step(b_up.ts_event_ms, {"600519.SH": b_up}, {"desired_weight": 0.8, "prev_close": 100.0})
        self.assertEqual(du["600519.SH"].target_weight, 0.0)
        self.assertEqual(du["600519.SH"].metadata["price_limit_state"], "LIMIT_UP_LOCK")

        inst.reset()
        inst.on_bar_step(b_d1.ts_event_ms, {"600519.SH": b_d1}, {"desired_weight": 0.8, "prev_close": 100.0})
        inst.sync_executed_weights({"600519.SH": 0.8})
        dd = inst.on_bar_step(b_dn.ts_event_ms, {"600519.SH": b_dn}, {"desired_weight": 0.0, "prev_close": 100.0})
        self.assertAlmostEqual(dd["600519.SH"].target_weight, 0.8, places=6)
        self.assertEqual(dd["600519.SH"].metadata["price_limit_state"], "LIMIT_DOWN_LOCK")

        # ChiNext ±20% auto-inferred via symbol prefix
        inst_cn = AShareStrategyInterface()
        inst_cn.reset()
        b_sz0 = _make_bar(symbol="300750.SZ", venue="szse", asset_class=ASSET_CLASS_CN_A_SHARE, timeframe_h=24.0, ts_event_ms=_utc_ms(2026, 11, 23, 6, 0), open_px=100.0, high_px=100.0, low_px=100.0, close_px=100.0)
        b_sz12 = _make_bar(symbol="300750.SZ", venue="szse", asset_class=ASSET_CLASS_CN_A_SHARE, timeframe_h=24.0, ts_event_ms=_utc_ms(2026, 11, 24, 6, 0), open_px=112.0, high_px=112.0, low_px=112.0, close_px=112.0)
        b_sz20 = _make_bar(symbol="300750.SZ", venue="szse", asset_class=ASSET_CLASS_CN_A_SHARE, timeframe_h=24.0, ts_event_ms=_utc_ms(2026, 11, 25, 6, 0), open_px=120.0, high_px=120.0, low_px=120.0, close_px=120.0)
        inst_cn.on_bar_step(b_sz0.ts_event_ms, {"300750.SZ": b_sz0}, {"desired_weight": 0.0, "prev_close": 100.0})
        inst_cn.sync_executed_weights({"300750.SZ": 0.0})
        d12 = inst_cn.on_bar_step(b_sz12.ts_event_ms, {"300750.SZ": b_sz12}, {"desired_weight": 0.8, "prev_close": 100.0})
        self.assertGreater(d12["300750.SZ"].target_weight, 0.0)
        inst_cn.sync_executed_weights({"300750.SZ": 0.0})
        d20 = inst_cn.on_bar_step(b_sz20.ts_event_ms, {"300750.SZ": b_sz20}, {"desired_weight": 0.8, "prev_close": 100.0})
        self.assertEqual(d20["300750.SZ"].target_weight, 0.0)

        # Hook slots combine auction + OFI + theme rotation cleanly
        inst_hooks = AShareStrategyInterface()
        inst_hooks.reset()
        dh = inst_hooks.on_bar_step(
            b_d1.ts_event_ms,
            {"600519.SH": b_d1},
            {"call_auction_score": 2.0, "ofi_score": 1.5, "theme_rotation_score": 1.0},
        )
        self.assertGreater(dh["600519.SH"].target_weight, 0.0)
        self.assertAlmostEqual(dh["600519.SH"].metadata["call_auction_score"], 2.0, places=6)
        self.assertAlmostEqual(dh["600519.SH"].metadata["ofi_score"], 1.5, places=6)
        self.assertAlmostEqual(dh["600519.SH"].metadata["theme_rotation_score"], 1.0, places=6)

        # Full G7 parity through PortfolioAccountingEngine with CN A-Share spec
        cn_spec = {"600519.SH": default_ashare_template_spec("600519.SH")}
        cfg = RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False)
        day0_ts = bars[0].ts_event_ms
        parity = verify_full_parity(
            AShareStrategyInterface(daily_price_limit_pct=10.0),
            cn_spec,
            {"600519.SH": bars},
            risk_cfg=cfg,
            context_provider=lambda ts: {"desired_weight": 0.7} if ts == day0_ts else {"desired_weight": 0.0},
        )
        self.assertTrue(parity.passed, parity.detail)

    def test_ast_strict_domain_isolation_zero_tombstones_and_stdlib_only(self) -> None:
        gold_dir = REPO_ROOT / "alphaloop" / "platform" / "strategies" / "gold"
        ashare_file = REPO_ROOT / "alphaloop" / "platform" / "strategies" / "ashare_contract.py"
        forbidden_dnt = (
            "alphaloop.execution",
            "alphaloop.daemon",
            "alphaloop.plugins",
            "run_shadow_daemon",
            "research_carry_regime_switch",
        )
        tombstone_re = re.compile(r"\b(GoldS1|SessionMeanReversion|OFLOW|HOF3|H_OF3|gold_s1_mean_reversion)\b", re.IGNORECASE)
        stale_768_re = re.compile(r"(?i)768\s*mb")

        for path in sorted(gold_dir.rglob("*.py")):
            txt = path.read_text(encoding="utf-8")
            self.assertIsNone(tombstone_re.search(txt), f"Tombstone token in {path}")
            self.assertIsNone(stale_768_re.search(txt), f"Stale 768MB reference in {path}")
            tree = ast.parse(txt, filename=str(path))
            for node in ast.walk(tree):
                mods = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                )
                for m in mods:
                    low = m.lower()
                    self.assertNotIn("web3", low)
                    self.assertNotIn("ashare", low)
                    for dnt in forbidden_dnt:
                        self.assertNotIn(dnt, m)

        a_txt = ashare_file.read_text(encoding="utf-8")
        self.assertIsNone(stale_768_re.search(a_txt))
        a_tree = ast.parse(a_txt, filename=str(ashare_file))
        for node in ast.walk(a_tree):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            )
            for m in mods:
                low = m.lower()
                self.assertNotIn("web3", low)
                self.assertNotIn("gold", low)
                for dnt in forbidden_dnt:
                    self.assertNotIn(dnt, m)

        # Verify pure stdlib import cleanly (blocking numpy/scipy/pyarrow/duckdb/pandas)
        probe = (
            "import sys\n"
            "BLOCKED = {'numpy','scipy','pyarrow','duckdb','polars','pandas','pytest'}\n"
            "class Blocker:\n"
            "  def find_spec(self, fullname, path, target=None):\n"
            "    if fullname.split('.')[0] in BLOCKED:\n"
            "      raise ImportError(fullname)\n"
            "    return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "import alphaloop.platform.strategies.gold\n"
            "import alphaloop.platform.strategies.ashare_contract\n"
            "print('GOLD_ASHARE_STDLIB_OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("GOLD_ASHARE_STDLIB_OK", proc.stdout)


class TestRealGoldMacroLakeSmoke(unittest.TestCase):
    """Real multi-year data smoke verification against data/platform_v4_lake/."""

    def test_real_lake_g1_g2_abc_and_g3_paxg_accounting_and_g7_parity(self) -> None:
        self.assertTrue(
            (REAL_V4_LAKE_DIR / "domains" / "gold_macro_lake.sqlite").is_file(),
            f"Real v4 gold_macro_lake.sqlite missing at {REAL_V4_LAKE_DIR}",
        )
        with tempfile.TemporaryDirectory(prefix="gold_real_smoke_") as tmpdir:
            copy_dir = Path(tmpdir) / "lake"
            (copy_dir / "domains").mkdir(parents=True)
            shutil.copy2(REAL_V4_LAKE_DIR / "catalog.sqlite", copy_dir / "catalog.sqlite")
            shutil.copy2(
                REAL_V4_LAKE_DIR / "domains" / "gold_macro_lake.sqlite",
                copy_dir / "domains" / "gold_macro_lake.sqlite",
            )
            router = DataFoundationRouter(copy_dir)
            self.addCleanup(router.close)

            macro_syms = ("GC=F", "GLD", "SI=F", "DX-Y.NYB", "IEF", "TIP", "GDX", "^GVZ", "^VIX")
            macro_bars = {s: router.load_bars("gold_macro", "yahoo", s, 86400) for s in macro_syms}
            yields = router.load_yield_curve()
            events = router.load_macro_events()
            gc_4h = router.load_bars("gold_macro", "yahoo", "GC=F", 14400)
            paxg_12h = router.load_bars("gold_macro", "hyperliquid", "PAXG", 43200)
            paxg_spot_12h = router.load_bars("gold_macro", "binance_spot", "PAXG", 43200)

            self.assertGreater(len(macro_bars["GC=F"]), 1500)
            self.assertGreater(len(yields), 1000)
            self.assertGreater(len(events), 150)
            self.assertGreater(len(gc_4h), 3000)
            self.assertGreater(len(paxg_12h), 1000)

            me = GoldMacroEngine(macro_bars_by_symbol=macro_bars, yield_curve=yields)
            specs = default_gold_specs()
            cfg = RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=15.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
            engine = PortfolioAccountingEngine()

            # G1 on real GC=F & separately real GLD
            for sym in ("GC=F", "GLD"):
                sl = macro_bars[sym][-500:]
                g1 = GoldTradFiMacroKernel(symbol=sym, macro_engine=me)
                res = engine.simulate(g1, {sym: specs[sym]}, {sym: sl}, risk_cfg=cfg, context_provider=lambda ts: {"gold_macro": me.snapshot_as_of(ts)})
                self.assertGreater(res.trade_count, 0)
                self.assertTrue(math.isfinite(res.cagr_pct) and math.isfinite(res.sharpe_ratio))
                bh = compute_buy_and_hold_returns(sl, timestamps_ms=res.period_timestamps_ms)
                g8 = evaluate_benchmark_gate(res.period_returns, bh, periods_per_year=252.0)
                self.assertTrue(math.isfinite(g8.strategy_sharpe) and math.isfinite(g8.benchmark_sharpe))
                p1 = verify_full_parity(g1, {sym: specs[sym]}, {sym: sl}, risk_cfg=cfg, batch_result=res, context_provider=lambda ts: {"gold_macro": me.snapshot_as_of(ts)})
                self.assertTrue(p1.passed, p1.detail)

            # G2 Policies A/B/C on real 4h GC=F + real PIT macro calendar
            sl_4h = gc_4h[-600:]
            g2_weights: dict[str, list[float]] = {}
            for pol in ("A", "B", "C"):
                g2 = Gold4hEventKernel(policy=pol, macro_engine=me, events=events)
                res_p = engine.simulate(
                    g2, {"GC=F": specs["GC=F"]}, {"GC=F": sl_4h}, risk_cfg=cfg,
                    context_provider=lambda ts: {"macro_events": events, "gold_macro": me.snapshot_as_of(ts)},
                )
                self.assertGreater(res_p.trade_count, 0)
                self.assertTrue(math.isfinite(res_p.sharpe_ratio))
                g2_weights[pol] = [round(w.get("GC=F", 0.0), 6) for w in res_p.executed_weights_history]
                p2 = verify_full_parity(
                    g2, {"GC=F": specs["GC=F"]}, {"GC=F": sl_4h}, risk_cfg=cfg, batch_result=res_p,
                    context_provider=lambda ts: {"macro_events": events, "gold_macro": me.snapshot_as_of(ts)},
                )
                self.assertTrue(p2.passed, p2.detail)
            # Non-degenerate ablation across Policies A/B/C on real 4h events
            self.assertFalse(g2_weights["A"] == g2_weights["B"] == g2_weights["C"])

            # G3 on real Hyperliquid PAXG 12h + real Binance Spot PAXG + real COMEX GC=F alignment
            spot_ts = [b.ts_avail_ms for b in paxg_spot_12h]
            spot_px = [b.close for b in paxg_spot_12h]

            def _paxg_ctx(ts: int) -> dict[str, Any]:
                idx_s = bisect.bisect_right(spot_ts, ts) - 1
                sp = spot_px[idx_s] if idx_s >= 0 else None
                gc = me.latest_close_as_of("GC=F", ts)
                ctx: dict[str, Any] = {"gold_macro": me.snapshot_as_of(ts)}
                if sp is not None:
                    ctx["paxg_spot_price"] = sp
                if gc is not None:
                    ctx["comex_price"] = gc
                return ctx

            sl_paxg = paxg_12h[-400:]
            g3 = GoldPAXGBridgeKernel(macro_engine=me)
            res_g3 = engine.simulate(g3, {"PAXG": specs["PAXG"]}, {"PAXG": sl_paxg}, risk_cfg=cfg, context_provider=_paxg_ctx)
            self.assertGreater(res_g3.trade_count, 0)
            self.assertTrue(math.isfinite(res_g3.sharpe_ratio) and math.isfinite(res_g3.cagr_pct))
            p3 = verify_full_parity(g3, {"PAXG": specs["PAXG"]}, {"PAXG": sl_paxg}, risk_cfg=cfg, batch_result=res_g3, context_provider=_paxg_ctx)
            self.assertTrue(p3.passed, p3.detail)


if __name__ == "__main__":
    unittest.main()
