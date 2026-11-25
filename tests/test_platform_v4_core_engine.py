"""Comprehensive Unit & Adversarial Regression Suite for Platform v4 Core Engine (M2).

Covers ALL 7 mandatory adversarial regression tests (D1, D2/V-04, D3/D4, D6, D7, D8, D9)
plus D5, D11, D12, D13/D15, G7 full kernel+RiskAllocator bit-parity, F-11 PortfolioCombiner,
fast-float accuracy/speed, and V-01 768MB docstring scrub.

100% deterministic (`random.Random(fixed_seed)`), zero network calls, pure Python 3.12 stdlib
compatible (passes identically with or without optional NumPy installed).
"""

from __future__ import annotations

import copy
import dataclasses
import math
import random
import statistics
import time
import unittest
from pathlib import Path
from typing import Any

from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.platform.assets import (
    ASSET_CLASS_CN_A_SHARE,
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_GOLD_TRADFI,
    CALENDAR_CN_A_SHARE,
    CALENDAR_CRYPTO_24_7,
    CALENDAR_GOLD_TRADFI_23_5,
    CALENDAR_US_DAILY,
    COST_GOLD_FUTURES_ETF,
    COST_HYPERLIQUID_PERP,
    AssetSpec,
    CostModel,
    default_ashare_template_spec,
    default_web3_specs,
)
from alphaloop.platform.core_engine.accounting import (
    PortfolioAccountingEngine,
    SimulationResult,
    exchange_trading_date,
)
from alphaloop.platform.core_engine.interfaces import AssetNativeStrategy, BarDecision
from alphaloop.platform.core_engine.portfolio_combiner import PortfolioCombiner
from alphaloop.platform.core_engine.risk_allocator import (
    CausalRiskAllocator,
    RiskAllocatorConfig,
)
from alphaloop.platform.core_engine.validation import (
    CPCVReport,
    FoldResult,
    GateVerdict,
    PlateauPoint,
    compute_buy_and_hold_returns,
    compute_effective_independent_trials,
    compute_hac_factor_ic,
    compute_pbo_cscv,
    compute_psr_dsr,
    compute_sparse_event_expectancy,
    evaluate_8_gates,
    evaluate_benchmark_gate,
    fast_moments,
    fast_var_std,
    newey_west_mean_stats,
    run_nested_cpcv,
    select_plateau_center,
    sharpe_from_returns,
    verify_full_parity,
)
from alphaloop.platform.data_lake import CanonicalBar
from alphaloop.platform.factors import (
    FactorDirectionAuditor,
    FactorSnapshot,
    MacroRegimeTimeline,
    StreamingFactorEngine,
)
from alphaloop.platform.regime_inference import (
    REGIME_HIGH_VOL_STRESS,
    REGIME_TREND_CALM,
    CausalRegimeTracker,
    MarketRegimeProfiler,
    RegimePoint,
    classify_raw_regime,
)
from alphaloop.platform.strategy_customizer import (
    AdaptiveMultiFactorKernel,
    KernelConfig,
    StrategyCustomizer,
    simulate_universe,
)

BASE_TS_MS = 1_735_689_600_000  # 2025-01-01T00:00:00Z
HOUR_MS = 3_600_000
DAY_MS = 86_400_000


def _bar(
    sym: str,
    ts_event_ms: int,
    o: float,
    h: float,
    l: float,
    c: float,
    *,
    tf_h: float = 24.0,
    asset_class: str = ASSET_CLASS_CRYPTO_PERP,
    venue: str = "test",
    volume_usd: float = 1_000_000.0,
    carry_rate_period: float = 0.0,
    carry_interval_h: float | None = None,
) -> CanonicalBar:
    step_ms = int(round(tf_h * HOUR_MS))
    ci_h = tf_h if carry_interval_h is None else float(carry_interval_h)
    return CanonicalBar(
        asset_class=asset_class,
        venue=venue,
        symbol=sym,
        timeframe_h=tf_h,
        ts_event_ms=ts_event_ms,
        ts_avail_ms=ts_event_ms + step_ms,
        open=o,
        high=max(h, o, c),
        low=min(l, o, c),
        close=c,
        volume_usd=volume_usd,
        carry_rate_period=carry_rate_period,
        carry_interval_h=ci_h,
        provenance=PROVENANCE_REAL,
    )


def _trend_bars(
    sym: str,
    n: int,
    *,
    tf_h: float = 12.0,
    asset_class: str = ASSET_CLASS_CRYPTO_PERP,
    venue: str = "hyperliquid",
    seed: int = 11,
    vol: float = 0.010,
) -> list[CanonicalBar]:
    rng = random.Random(seed)
    px = 100.0
    step_ms = int(round(tf_h * HOUR_MS))
    out: list[CanonicalBar] = []
    for i in range(n):
        phase = (i // 45) % 3
        drift = 0.008 if phase in (0, 2) else -0.007
        ret = drift + rng.gauss(0.0, vol)
        o = px
        c = max(2.0, o * math.exp(ret))
        h = max(o, c) * (1.0 + abs(rng.gauss(0.0, vol * 0.4)))
        l = min(o, c) * max(0.5, 1.0 - abs(rng.gauss(0.0, vol * 0.4)))
        out.append(
            _bar(
                sym,
                BASE_TS_MS + i * step_ms,
                o,
                h,
                l,
                c,
                tf_h=tf_h,
                asset_class=asset_class,
                venue=venue,
                volume_usd=2_000_000.0 * (1.0 + abs(ret) * 15.0),
                carry_rate_period=0.00005,
                carry_interval_h=tf_h,
            )
        )
        px = c
    return out


class _ScriptedStrategy:
    """Minimal AssetNativeStrategy emitting pre-scripted weights by bar index for exact unit proofs."""

    def __init__(
        self,
        schedule: dict[int, dict[str, tuple[float, float, float | None]]],
        *,
        strategy_id: str = "scripted",
        asset_domain: str = "web3",
        warmup_bars: int = 0,
    ) -> None:
        self.strategy_id = strategy_id
        self.asset_domain = asset_domain
        self.warmup_bars = warmup_bars
        self._schedule = schedule
        self._step = 0
        self.last_synced: dict[str, float] = {}

    def reset(self) -> None:
        self._step = 0
        self.last_synced = {}

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        _ = context
        idx = self._step
        self._step += 1
        step_map = self._schedule.get(idx, {})
        out: dict[str, BarDecision] = {}
        for sym in sorted(bars_at_ts):
            w, sig, stop_px = step_map.get(sym, (self.last_synced.get(sym, 0.0), 0.0, None))
            out[sym] = BarDecision(
                symbol=sym,
                ts_event_ms=ts_ms,
                target_weight=w,
                raw_signal=sig,
                regime_label="SCRIPTED",
                stop_price=stop_px,
            )
        return out

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        self.last_synced = dict(executed_weights)

    def clone_fresh(self) -> _ScriptedStrategy:
        return _ScriptedStrategy(
            self._schedule,
            strategy_id=self.strategy_id,
            asset_domain=self.asset_domain,
            warmup_bars=self.warmup_bars,
        )


class _StatefulInertiaStrategy:
    """Stateful AssetNativeStrategy with Carver inertia buffer relying on `sync_executed_weights()`."""

    def __init__(
        self,
        symbols: list[str],
        *,
        warmup_bars: int = 15,
        buffer_pct: float = 0.15,
        inject_nondeterminism: bool = False,
    ) -> None:
        self.strategy_id = "stateful_inertia_mock"
        self.asset_domain = "web3"
        self.warmup_bars = warmup_bars
        self._symbols = sorted(symbols)
        self._buffer_pct = buffer_pct
        self._inject_nondeterminism = inject_nondeterminism
        self._bar_count = 0
        self._ema_fast: dict[str, float] = {}
        self._ema_slow: dict[str, float] = {}
        self._held: dict[str, float] = {}
        self._clone_nonce = 0

    def reset(self) -> None:
        self._bar_count = 0
        self._ema_fast = {}
        self._ema_slow = {}
        self._held = {s: 0.0 for s in self._symbols}

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        _ = context
        self._bar_count += 1
        out: dict[str, BarDecision] = {}
        for idx, sym in enumerate(sorted(bars_at_ts)):
            c = float(bars_at_ts[sym].close)
            if sym not in self._ema_fast:
                self._ema_fast[sym] = c
                self._ema_slow[sym] = c
            else:
                self._ema_fast[sym] = 0.25 * c + 0.75 * self._ema_fast[sym]
                self._ema_slow[sym] = 0.06 * c + 0.94 * self._ema_slow[sym]
            if self._bar_count <= self.warmup_bars:
                out[sym] = BarDecision(sym, ts_ms, 0.0, 0.0, "WARMUP")
                continue
            raw_sig = ((self._ema_fast[sym] - self._ema_slow[sym]) / max(c, 1e-6)) * 100.0 + (0.05 * (idx + 1))
            desired_w = max(-0.65, min(0.65, raw_sig * 0.25))
            if self._inject_nondeterminism and self._bar_count == self.warmup_bars + 5:
                desired_w += 0.05 * self._clone_nonce
            cur_w = self._held.get(sym, 0.0)
            if abs(desired_w - cur_w) < self._buffer_pct * max(abs(cur_w), 0.1):
                w = cur_w
            else:
                w = desired_w
            regime = "TREND_UP" if raw_sig >= 0.0 else "TREND_DN"
            out[sym] = BarDecision(sym, ts_ms, w, raw_sig, regime)
        return out

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        for sym, w in executed_weights.items():
            self._held[sym] = float(w)

    def clone_fresh(self) -> _StatefulInertiaStrategy:
        self._clone_nonce += 1
        cloned = _StatefulInertiaStrategy(
            list(self._symbols),
            warmup_bars=self.warmup_bars,
            buffer_pct=self._buffer_pct,
            inject_nondeterminism=self._inject_nondeterminism,
        )
        cloned._clone_nonce = self._clone_nonce
        return cloned


class TestPlatformV4CoreEngine(unittest.TestCase):
    """Full M2 verification & adversarial regression suite (13 tests)."""

    def test_01_exit_gap_and_holiday_gap_conservation_d1(self) -> None:
        """D1: Exit-bar overnight gap AND per-symbol holiday gap MUST be conserved identically."""
        zero_cost = CostModel(maker_fee_bps=0.0, taker_fee_bps=0.0, slippage_bps_per_leg=0.0)
        spec_btc = AssetSpec("BTC", ASSET_CLASS_CRYPTO_PERP, "test", CALENDAR_CRYPTO_24_7, zero_cost, max_leverage_cap=2.0)

        # Part A: Exit-bar +10% overnight gap (bar 0 signals +1.0, bar 1 fills@100 & signals 0.0, bar 2 opens 110)
        bars_a = [
            _bar("BTC", BASE_TS_MS + 0 * DAY_MS, 100.0, 100.0, 100.0, 100.0),
            _bar("BTC", BASE_TS_MS + 1 * DAY_MS, 100.0, 100.0, 100.0, 100.0),
            _bar("BTC", BASE_TS_MS + 2 * DAY_MS, 110.0, 110.0, 110.0, 110.0),
            _bar("BTC", BASE_TS_MS + 3 * DAY_MS, 110.0, 110.0, 110.0, 110.0),
        ]
        sched_a = {
            0: {"BTC": (1.0, 10.0, None)},
            1: {"BTC": (0.0, 0.0, None)},
            2: {"BTC": (0.0, 0.0, None)},
            3: {"BTC": (0.0, 0.0, None)},
        }
        strat_a = _ScriptedStrategy(sched_a)
        self.assertIsInstance(strat_a, AssetNativeStrategy)
        engine = PortfolioAccountingEngine()
        res_a = engine.simulate(
            strat_a,
            {"BTC": spec_btc},
            {"BTC": bars_a},
            risk_cfg=RiskAllocatorConfig(
                max_positions=1,
                target_annual_vol_pct=500.0,
                max_portfolio_leverage=2.0,
                max_single_weight=1.0,
                use_correlation_idm=False,
            ),
            capital_usd=10_000.0,
        )
        self.assertAlmostEqual(res_a.gross_price_pnl_usd, 1000.0, places=6)
        self.assertAlmostEqual(res_a.gap_pnl_usd, 1000.0, places=6)
        self.assertAlmostEqual(res_a.net_pnl_usd, 1000.0, places=6)
        self.assertAlmostEqual(res_a.final_equity_usd, 11_000.0, places=6)

        # Also verify v3 simulate_universe preserves exit-bar gap conservation!
        # Construct snapshots triggering entry at bar 0 and exit at bar 1 with zero warmup & zero cost
        def _mk_snap(b: CanonicalBar, f_val: float) -> FactorSnapshot:
            return FactorSnapshot(
                symbol=b.symbol,
                venue=b.venue,
                timeframe_h=b.timeframe_h,
                ts_event_ms=b.ts_event_ms,
                ts_avail_ms=b.ts_avail_ms,
                close=b.close,
                atr_14=1.0,
                prior_high_40=101.0,
                prior_low_40=99.0,
                prior_high_20=101.0,
                prior_low_20=99.0,
                values={
                    "ewmac_8_32": f_val,
                    "ewmac_16_64": f_val,
                    "ewmac_32_128": f_val,
                    "donchian_pos_40": 0.0,
                    "realized_vol_ann_20": 20.0,
                    "rel_volume_20": 1.0,
                },
            )

        def _mk_reg(b: CanonicalBar) -> RegimePoint:
            return RegimePoint(b.ts_event_ms, b.ts_avail_ms, REGIME_TREND_CALM, REGIME_TREND_CALM, 25.0, 0.4, 0.5, 1.0, 0.5, 0.0)

        snaps_a = [_mk_snap(bars_a[0], 12.0), _mk_snap(bars_a[1], 0.0), _mk_snap(bars_a[2], 0.0), _mk_snap(bars_a[3], 0.0)]
        regs_a = [_mk_reg(b) for b in bars_a]
        cfg_a = KernelConfig(
            strategy_name="d1_exit_gap",
            asset_class=ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=24.0,
            capital_usd=10_000.0,
            max_positions=1,
            target_annual_vol_pct=200.0,
            max_symbol_leverage=1.0,
            entry_forecast_threshold=4.0,
            exit_forecast_threshold=1.0,
            atr_stop_mult=5.0,
            min_hold_bars=1,
            donchian_weight=0.0,
            warmup_bars=0,
            regime_exposure_map={REGIME_TREND_CALM: 1.0},
            risk_per_trade_pct=50.0,
        )
        v3_m, v3_decs = simulate_universe(cfg_a, {"BTC": spec_btc}, {"BTC": bars_a}, {"BTC": snaps_a}, {"BTC": regs_a})
        entry_w = v3_decs["BTC"][0].target_weight
        self.assertGreater(entry_w, 0.5)
        self.assertAlmostEqual(v3_decs["BTC"][1].target_weight, 0.0, places=6)
        expected_v3_gap = round(entry_w * 10_000.0 * 0.10, 2)
        self.assertAlmostEqual(v3_m.gross_price_pnl_usd, expected_v3_gap, places=2)
        self.assertAlmostEqual(v3_m.net_pnl_usd, expected_v3_gap, places=2)

        # Part B: Per-symbol holiday gap (GC=F prints on day 2 while GLD has holiday; GLD opens +5% on day 3)
        spec_gc = AssetSpec("GC=F", ASSET_CLASS_GOLD_TRADFI, "yahoo", CALENDAR_GOLD_TRADFI_23_5, zero_cost, max_leverage_cap=2.0)
        spec_gld = AssetSpec("GLD", ASSET_CLASS_GOLD_TRADFI, "yahoo", CALENDAR_US_DAILY, zero_cost, max_leverage_cap=1.5)
        gc_bars = [
            _bar("GC=F", BASE_TS_MS + d * DAY_MS, 2000.0, 2000.0, 2000.0, 2000.0, asset_class=ASSET_CLASS_GOLD_TRADFI)
            for d in range(4)
        ]
        # GLD misses day 2 (holiday), reopens day 3 at +5% (105.0)
        gld_bars = [
            _bar("GLD", BASE_TS_MS + 0 * DAY_MS, 100.0, 100.0, 100.0, 100.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
            _bar("GLD", BASE_TS_MS + 1 * DAY_MS, 100.0, 100.0, 100.0, 100.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
            _bar("GLD", BASE_TS_MS + 3 * DAY_MS, 105.0, 105.0, 105.0, 105.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
        ]
        sched_b = {
            0: {"GC=F": (0.0, 0.0, None), "GLD": (0.5, 10.0, None)},
            1: {"GC=F": (0.0, 0.0, None), "GLD": (0.5, 10.0, None)},
            2: {"GC=F": (0.0, 0.0, None)},  # GLD holiday
            3: {"GC=F": (0.0, 0.0, None), "GLD": (0.0, 0.0, None)},
        }
        res_b = engine.simulate(
            _ScriptedStrategy(sched_b, asset_domain="gold"),
            {"GC=F": spec_gc, "GLD": spec_gld},
            {"GC=F": gc_bars, "GLD": gld_bars},
            risk_cfg=RiskAllocatorConfig(
                max_positions=2,
                target_annual_vol_pct=500.0,
                max_portfolio_leverage=2.0,
                max_single_weight=1.0,
                use_correlation_idm=False,
            ),
            capital_usd=10_000.0,
        )
        # Held 0.5 * 10,000 = $5,000 across holiday -> +5% gap = +$250.00!
        self.assertAlmostEqual(res_b.gap_pnl_usd, 250.0, places=6)
        self.assertAlmostEqual(res_b.net_pnl_usd, 250.0, places=6)
        self.assertAlmostEqual(res_b.by_symbol_net_usd["GLD"], 250.0, places=6)

    def test_02_macro_regime_timeline_non_monotonic_reuse_d6(self) -> None:
        """D6: Stateless bisect_right guarantees bit-identical non-monotonic reuse + SI=F/^TNX support."""
        rng = random.Random(202)
        n = 160
        macro_data: dict[str, list[CanonicalBar]] = {
            s: [] for s in ("GC=F", "IEF", "DX-Y.NYB", "TIP", "^VIX", "SI=F", "^TNX")
        }
        px = {"GC=F": 2000.0, "IEF": 100.0, "DX-Y.NYB": 102.0, "TIP": 108.0, "^VIX": 16.0, "SI=F": 24.0, "^TNX": 3.8}
        for i in range(n):
            ts = BASE_TS_MS + i * DAY_MS
            # Sharp macro regime flip after bar 80
            flip = 1.0 if i < 80 else -1.0
            moves = {
                "GC=F": 0.004 * flip + rng.gauss(0.0, 0.004),
                "IEF": 0.003 * flip + rng.gauss(0.0, 0.003),
                "DX-Y.NYB": -0.002 * flip + rng.gauss(0.0, 0.003),
                "TIP": 0.0025 * flip + rng.gauss(0.0, 0.003),
                "^VIX": -0.005 * flip + rng.gauss(0.0, 0.04),
                "SI=F": 0.0015 * flip + rng.gauss(0.0, 0.006),
            }
            for sym, ret in moves.items():
                o = px[sym]
                c = max(1.0, o * math.exp(ret))
                macro_data[sym].append(_bar(sym, ts, o, max(o, c), min(o, c), c, tf_h=24.0, asset_class=ASSET_CLASS_GOLD_TRADFI, venue="yahoo"))
                px[sym] = c
            # ^TNX is yield in percentage points
            tnx_o = px["^TNX"]
            tnx_c = max(0.5, tnx_o + (-0.015 * flip + rng.gauss(0.0, 0.03)))
            macro_data["^TNX"].append(_bar("^TNX", ts, tnx_o, max(tnx_o, tnx_c), min(tnx_o, tnx_c), tnx_c, tf_h=24.0, asset_class=ASSET_CLASS_GOLD_TRADFI, venue="yahoo"))
            px["^TNX"] = tnx_c

        reused_tl = MacroRegimeTimeline(macro_data)
        ts_list = [b.ts_avail_ms for b in macro_data["GC=F"]]

        # Trigger exact D6 non-monotonic sequence: bar 10 -> bar 159 -> bar 9 -> bar 10 -> 40 random queries
        query_indices = [10, 159, 9, 10] + [rng.randrange(0, n) for _ in range(40)]
        for qi in query_indices:
            ts = ts_list[qi]
            got = reused_tl.snapshot_as_of(ts)
            fresh = MacroRegimeTimeline(macro_data).snapshot_as_of(ts)
            self.assertEqual(got, fresh, f"Non-monotonic reuse drift at index {qi}")

        # Verify SI=F (GSR) and ^TNX yield signals are active when present and 0.0 when omitted
        late_with = reused_tl.snapshot_as_of(ts_list[140])
        self.assertNotEqual(late_with["macro_gsr_trend"], 0.0)
        self.assertNotEqual(late_with["macro_gsr_z"], 0.0)
        self.assertNotEqual(late_with["macro_tnx_mom"], 0.0)
        self.assertNotEqual(late_with["macro_tnx_z"], 0.0)

        reduced_data = {k: v for k, v in macro_data.items() if k not in ("SI=F", "^TNX")}
        late_without = MacroRegimeTimeline(reduced_data).snapshot_as_of(ts_list[140])
        self.assertEqual(late_without["macro_gsr_trend"], 0.0)
        self.assertEqual(late_without["macro_gsr_z"], 0.0)
        self.assertEqual(late_without["macro_tnx_mom"], 0.0)
        self.assertEqual(late_without["macro_tnx_z"], 0.0)

    def test_03_dead_parameter_guard_d2_and_v04_squeeze_gating(self) -> None:
        """D2 & V-04: Dynamic Donchian & risk_per_trade_pct affect output; squeeze gated behind use_squeeze_boost."""
        bars = _trend_bars("BTC", 180, tf_h=12.0, seed=31, vol=0.014)
        spec = default_web3_specs(["BTC"])["BTC"]

        # Part A1: Verify StreamingFactorEngine.donchian_pos(lookback) matches exact strictly-prior manual formula
        eng = StreamingFactorEngine(spec.calendar, 12.0, donchian_entry_n=25, donchian_exit_n=10)
        snaps = eng.compute_series(bars)
        for lb in (10, 20, 40, 55):
            prior_h = max(b.high for b in bars[-1 - lb : -1])
            prior_l = min(b.low for b in bars[-1 - lb : -1])
            c_last = bars[-1].close
            manual = max(-1.5, min(1.5, 2.0 * ((c_last - prior_l) / (prior_h - prior_l)) - 1.0))
            self.assertAlmostEqual(eng.donchian_pos(lb), manual, places=12)

        regs = CausalRegimeTracker(confirm_bars=2, asset_class=spec.asset_class).classify_series(snaps)
        base_cfg = KernelConfig(
            strategy_name="dead_param_guard",
            asset_class=ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0,
            max_positions=1,
            target_annual_vol_pct=25.0,
            max_symbol_leverage=2.0,
            donchian_weight=0.45,
            atr_stop_mult=3.0,
            min_hold_bars=2,
            warmup_bars=45,
            regime_exposure_map={REGIME_TREND_CALM: 1.0},
            donchian_entry_n=20,
            donchian_exit_n=12,
            risk_per_trade_pct=0.75,
            use_squeeze_boost=False,
        )

        def _eval_decs(cfg: KernelConfig) -> list[tuple[float, str]]:
            k = AdaptiveMultiFactorKernel(cfg, spec)
            return [(d.target_weight, d.reason) for d in (k.decide_bar(b, s, r) for b, s, r in zip(bars, snaps, regs))]

        # Part A2: Each parameter MUST produce measurable differences
        for field_name, val_a, val_b in (
            ("donchian_entry_n", 12, 50),
            ("donchian_exit_n", 6, 25),
            ("risk_per_trade_pct", 0.25, 1.50),
            ("atr_stop_mult", 2.0, 4.8),
        ):
            out_a = _eval_decs(dataclasses.replace(base_cfg, **{field_name: val_a}))
            out_b = _eval_decs(dataclasses.replace(base_cfg, **{field_name: val_b}))
            diffs = sum(1 for a, b in zip(out_a, out_b) if abs(a[0] - b[0]) > 1e-6 or a[1] != b[1])
            self.assertGreater(diffs, 0, f"Parameter {field_name} ({val_a} vs {val_b}) had ZERO effect (dead param!)")

        # Part B: V-04 Squeeze Gating (default False -> zero contribution; True -> positive boost)
        b0 = bars[60]
        sq_snap = FactorSnapshot(
            symbol="BTC",
            venue="hyperliquid",
            timeframe_h=12.0,
            ts_event_ms=b0.ts_event_ms,
            ts_avail_ms=b0.ts_avail_ms,
            close=b0.close,
            atr_14=1.0,
            prior_high_40=b0.close,
            prior_low_40=b0.close,
            prior_high_20=b0.close,
            prior_low_20=b0.close,
            values={
                "ewmac_8_32": 0.0,
                "ewmac_16_64": 0.0,
                "ewmac_32_128": 0.0,
                "donchian_pos_40": 0.0,
                "squeeze_breakout_dir": 1.0,
                "realized_vol_ann_20": 20.0,
                "rel_volume_20": 1.0,
            },
        )
        reg_pt = RegimePoint(b0.ts_event_ms, b0.ts_avail_ms, REGIME_TREND_CALM, REGIME_TREND_CALM, 20.0, 0.3, 0.5, 0.8, 0.5, 0.0)
        cfg_sq_off = dataclasses.replace(base_cfg, warmup_bars=0, use_squeeze_boost=False)
        cfg_sq_on = dataclasses.replace(base_cfg, warmup_bars=0, use_squeeze_boost=True)
        dec_off = AdaptiveMultiFactorKernel(cfg_sq_off, spec).decide_bar(b0, sq_snap, reg_pt)
        dec_on = AdaptiveMultiFactorKernel(cfg_sq_on, spec).decide_bar(b0, sq_snap, reg_pt)
        self.assertAlmostEqual(dec_off.raw_forecast, 0.0, places=6)
        self.assertGreater(dec_on.raw_forecast, 3.0)

    def test_04_nested_cpcv_and_dsr_null_variance_floor_d3_d4(self) -> None:
        """D3 & D4: DSR null variance floor prevents SR* collapse; Nested CPCV rejects random-walk IS overfit."""
        rng = random.Random(404)

        # Part A: 9 correlated twins (rho > 0.99) + tiny trial_std=1e-5 MUST keep SR* > 0.30
        base_rw = [rng.gauss(0.0, 0.015) for _ in range(400)]
        twins = [[r + rng.gauss(0.0, 0.0002) for r in base_rw] for _ in range(9)]
        n_eff_9, avg_corr_9 = compute_effective_independent_trials(twins)
        self.assertLess(n_eff_9, 1.2)
        self.assertGreater(avg_corr_9, 0.99)

        # Dead zero rows MUST NOT inflate N_eff
        twins_plus_dead = twins + [[0.0] * 400 for _ in range(10)]
        n_eff_dead, _ = compute_effective_independent_trials(twins_plus_dead)
        self.assertAlmostEqual(n_eff_dead, n_eff_9, places=8)

        dsr_12h = compute_psr_dsr(
            base_rw,
            periods_per_year=730.0,
            n_trials=9,
            trial_sharpe_std_per_period=1e-5,
            candidate_returns_matrix=twins,
        )
        self.assertGreater(dsr_12h.sr_star_annualized, 0.30)
        self.assertGreaterEqual(dsr_12h.sig_trials_per_period, (1.0 / math.sqrt(400)) - 1e-6)

        # Cumulative TrialLedger family_trial_count=50 strictly raises SR* and lowers DSR
        dsr_rw_9 = compute_psr_dsr(base_rw[:250], periods_per_year=252.0, n_trials=9, trial_sharpe_std_per_period=1e-5, family_trial_count=9)
        self.assertGreater(dsr_rw_9.sr_star_annualized, 0.30)
        pos_series = [0.0014 + rng.gauss(0.0, 0.011) for _ in range(250)]
        dsr_daily_9 = compute_psr_dsr(pos_series, periods_per_year=252.0, n_trials=9, trial_sharpe_std_per_period=1e-5, family_trial_count=9)
        dsr_daily_50 = compute_psr_dsr(pos_series, periods_per_year=252.0, n_trials=9, trial_sharpe_std_per_period=1e-5, family_trial_count=50)
        self.assertGreater(dsr_daily_9.sr_star_annualized, 0.30)
        self.assertGreater(dsr_daily_50.sr_star_annualized, dsr_daily_9.sr_star_annualized)
        self.assertLess(dsr_daily_50.dsr, dsr_daily_9.dsr)

        # Part B: True Nested CPCV on pure random walk with aggressive IS argmax overfit (25 random strategies)
        n_obs = 360
        cand_streams = [[rng.gauss(0.0, 0.012) for _ in range(n_obs)] for _ in range(25)]
        # Exact zero-drift center each candidate to simulate pure noise
        cand_streams = [[x - (sum(row) / n_obs) for x in row] for row in cand_streams]
        checked_splits = 0

        def _overfit_rw_fold(train_idx: tuple[int, ...], test_idx: tuple[int, ...]) -> FoldResult:
            nonlocal checked_splits
            checked_splits += 1
            tr_set = set(train_idx)
            te_set = set(test_idx)
            self.assertTrue(tr_set.isdisjoint(te_set))
            # Pick best candidate strictly on train_idx
            best_c = 0
            best_sr = -1e9
            for c_i, row in enumerate(cand_streams):
                tr_sr = sharpe_from_returns([row[i] for i in train_idx], 252.0)
                if tr_sr > best_sr:
                    best_sr = tr_sr
                    best_c = c_i
            tr_rets = tuple(cand_streams[best_c][i] for i in train_idx)
            te_rets = tuple(cand_streams[best_c][i] for i in test_idx)
            return FoldResult(oos_returns=te_rets, is_returns=tr_rets, selected_params={"cand": best_c})

        nested_rw = run_nested_cpcv(n_obs, _overfit_rw_fold, 252.0, n_groups=6, k_test=2, purge_bars=4, embargo_bars=4)
        self.assertTrue(nested_rw.is_nested)
        self.assertEqual(checked_splits, 15)
        self.assertEqual(nested_rw.stitched_oos_paths_count, 5)
        self.assertGreater(nested_rw.median_is_sharpe, 0.5)
        self.assertFalse(nested_rw.passed_cpcv_gate)

        # Verify genuine persistent drift stream passes Nested CPCV
        drift_stream = [0.005 + rng.gauss(0.0, 0.008) for _ in range(240)]
        nested_good = run_nested_cpcv(
            240,
            lambda tr, te: FoldResult(
                oos_returns=tuple(drift_stream[i] for i in te),
                is_returns=tuple(drift_stream[i] for i in tr),
            ),
            252.0,
            n_groups=6,
            k_test=2,
            purge_bars=3,
            embargo_bars=3,
        )
        self.assertTrue(nested_good.passed_cpcv_gate)
        self.assertEqual(nested_good.stitched_oos_paths_count, 5)

    def test_05_web3_zero_tradfi_vix_coupling_d7(self) -> None:
        """D7: Yahoo ^VIX NEVER leaks into Web3 regime classification, profiling, or customization."""
        specs = default_web3_specs(["BTC", "ETH", "SOL"], venue="hyperliquid")
        bars_by_sym = {
            "BTC": _trend_bars("BTC", 180, tf_h=12.0, seed=51),
            "ETH": _trend_bars("ETH", 180, tf_h=12.0, seed=52),
            "SOL": _trend_bars("SOL", 180, tf_h=12.0, seed=53),
        }

        # Build hostile Yahoo macro bars with violent ^VIX spikes (vix_z > 3.5 on ~40% of bars)
        vix_bars: list[CanonicalBar] = []
        gc_bars: list[CanonicalBar] = []
        ief_bars: list[CanonicalBar] = []
        for i in range(120):
            ts = BASE_TS_MS + i * DAY_MS
            vix_c = 15.0 + (0.05 * (i % 3)) if i < 65 else (55.0 if i % 2 == 0 else 16.0)
            vix_bars.append(_bar("^VIX", ts, vix_c, vix_c, vix_c, vix_c, tf_h=24.0, asset_class=ASSET_CLASS_GOLD_TRADFI, venue="yahoo"))
            gc_c = 2000.0 + i * 2.0
            gc_bars.append(_bar("GC=F", ts, gc_c, gc_c, gc_c, gc_c, tf_h=24.0, asset_class=ASSET_CLASS_GOLD_TRADFI, venue="yahoo"))
            ief_c = 100.0 + i * 0.1
            ief_bars.append(_bar("IEF", ts, ief_c, ief_c, ief_c, ief_c, tf_h=24.0, asset_class=ASSET_CLASS_GOLD_TRADFI, venue="yahoo"))
        hostile_yahoo = {"^VIX": vix_bars, "GC=F": gc_bars, "IEF": ief_bars}

        # 1. Direct classify_raw_regime hard guard
        vix_poison_snap = FactorSnapshot(
            symbol="BTC",
            venue="hyperliquid",
            timeframe_h=12.0,
            ts_event_ms=BASE_TS_MS,
            ts_avail_ms=BASE_TS_MS + 12 * HOUR_MS,
            close=100.0,
            atr_14=1.0,
            prior_high_40=102.0,
            prior_low_40=98.0,
            prior_high_20=101.0,
            prior_low_20=99.0,
            values={
                "vol_pctile_60": 0.40,
                "macro_vix_stress_z": 4.2,
                "adx_14": 22.0,
                "kaufman_er_20": 0.30,
                "squeeze_ratio_20": 1.1,
            },
        )
        self.assertNotEqual(
            classify_raw_regime(vix_poison_snap, use_macro_overlay=False, asset_class=ASSET_CLASS_CRYPTO_PERP),
            REGIME_HIGH_VOL_STRESS,
        )
        self.assertNotEqual(
            classify_raw_regime(vix_poison_snap, use_macro_overlay=True, asset_class=ASSET_CLASS_CRYPTO_PERP),
            REGIME_HIGH_VOL_STRESS,
        )
        self.assertEqual(
            classify_raw_regime(vix_poison_snap, use_macro_overlay=True, asset_class=ASSET_CLASS_GOLD_TRADFI),
            REGIME_HIGH_VOL_STRESS,
        )

        # 2. MarketRegimeProfiler on Web3 with vs without Yahoo macro -> 0 diffs
        profiler = MarketRegimeProfiler()
        snaps_no_macro = StreamingFactorEngine(specs["BTC"].calendar, 12.0).compute_series(bars_by_sym["BTC"], macro_timeline=None)
        snaps_with_macro = StreamingFactorEngine(specs["BTC"].calendar, 12.0).compute_series(
            bars_by_sym["BTC"], macro_timeline=MacroRegimeTimeline(hostile_yahoo)
        )
        prof_a, regs_a = profiler.profile_asset(specs["BTC"], bars_by_sym["BTC"], snaps_no_macro)
        prof_b, regs_b = profiler.profile_asset(specs["BTC"], bars_by_sym["BTC"], snaps_with_macro)
        self.assertEqual(prof_a.to_dict(), prof_b.to_dict())
        self.assertEqual(regs_a, regs_b)

        # 3. Full StrategyCustomizer on Web3 with hostile_yahoo vs None -> 0 diffs & G7 PASS!
        cust = StrategyCustomizer(target_annual_vol_pct=18.0)
        res_clean = cust.build_and_validate(
            "web3_d7_clean",
            specs,
            bars_by_sym,
            macro_bars_by_symbol=None,
            factor_horizon_bars=6,
            allow_synthetic_in_test=True,
        )
        res_hostile = cust.build_and_validate(
            "web3_d7_clean",
            specs,
            bars_by_sym,
            macro_bars_by_symbol=hostile_yahoo,
            factor_horizon_bars=6,
            allow_synthetic_in_test=True,
        )
        self.assertEqual(res_clean.baseline_metrics, res_hostile.baseline_metrics)
        self.assertEqual(res_clean.final_kernel_config.to_dict(), res_hostile.final_kernel_config.to_dict())
        self.assertFalse(res_hostile.final_kernel_config.use_macro_overlay)
        gates_h = {g.gate_id: g for g in res_hostile.gates}
        self.assertTrue(gates_h["G7"].passed, gates_h["G7"].detail)

    def test_06_hard_max_positions_and_corr_idm_vol_calibration_d9(self) -> None:
        """D9: Hard max_positions cap + shrunk-correlation IDM calibrates vol on correlated majors."""
        rng = random.Random(606)
        syms = [f"COIN{i:02d}" for i in range(12)]
        zero_cost = CostModel(maker_fee_bps=0.0, taker_fee_bps=0.0, slippage_bps_per_leg=0.0)
        specs = {
            s: AssetSpec(s, ASSET_CLASS_CRYPTO_PERP, "hyperliquid", CALENDAR_CRYPTO_24_7, zero_cost, max_leverage_cap=2.0)
            for s in syms
        }

        # 12 correlated majors (rho ~ 0.75, ~60% annualized vol on 12h bars -> bar_vol ~ 0.60 / sqrt(730) = 0.0222)
        n_bars = 300
        bar_vol = 0.60 / math.sqrt(730.0)
        beta_m = math.sqrt(0.75)
        beta_i = math.sqrt(0.25)
        mkt_shocks = [rng.gauss(0.0, bar_vol) for _ in range(n_bars)]
        bars_by_sym: dict[str, list[CanonicalBar]] = {}
        for s in syms:
            px = 100.0
            s_bars: list[CanonicalBar] = []
            for t in range(n_bars):
                ret = beta_m * mkt_shocks[t] + beta_i * rng.gauss(0.0, bar_vol)
                o = px
                c = max(1.0, o * math.exp(ret))
                s_bars.append(_bar(s, BASE_TS_MS + t * 12 * HOUR_MS, o, max(o, c), min(o, c), c, tf_h=12.0))
                px = c
            bars_by_sym[s] = s_bars

        class _AllLongRankedStrategy:
            strategy_id = "all_long_12"
            asset_domain = "web3"
            warmup_bars = 65

            def reset(self) -> None:
                self._synced: dict[str, float] = {}

            def on_bar_step(self, ts_ms: int, bars_at_ts: dict[str, CanonicalBar], context: dict[str, Any] | None = None) -> dict[str, BarDecision]:
                _ = context
                decs: dict[str, BarDecision] = {}
                for idx, s in enumerate(sorted(bars_at_ts)):
                    # COIN00..COIN03 have highest signal strength -> MUST be top-4 kept!
                    sig = 12.0 - 0.25 * idx
                    decs[s] = BarDecision(s, ts_ms, target_weight=0.50, raw_signal=sig, regime_label="TREND")
                return decs

            def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
                self._synced = dict(executed_weights)

            def clone_fresh(self) -> _AllLongRankedStrategy:
                return _AllLongRankedStrategy()

        engine = PortfolioAccountingEngine()
        cfg_idm = RiskAllocatorConfig(
            max_positions=4,
            target_annual_vol_pct=18.0,
            max_portfolio_leverage=1.5,
            max_single_weight=0.5,
            use_correlation_idm=True,
            corr_lookback_bars=60,
            corr_shrinkage=0.25,
        )
        res_idm = engine.simulate(_AllLongRankedStrategy(), specs, bars_by_sym, risk_cfg=cfg_idm, capital_usd=10_000.0)

        # 1. Hard max_positions <= 4 on EVERY bar & top-4 ranked by abs(raw_signal)
        self.assertLessEqual(res_idm.max_active_positions, 4)
        for _, alloc_w in res_idm.allocated_weights_history:
            active_syms = {s for s, w in alloc_w.items() if abs(w) > 1e-9}
            self.assertLessEqual(len(active_syms), 4)
            self.assertTrue(active_syms.issubset({"COIN00", "COIN01", "COIN02", "COIN03"}))
            self.assertLessEqual(sum(abs(w) for w in alloc_w.values()), 1.5 + 1e-9)

        # 2. Hard gross leverage <= 1.5
        self.assertLessEqual(res_idm.max_gross_leverage, 1.5 + 1e-9)

        # 3. Realized annualized portfolio vol calibrated within [0.6x, 1.35x] of 18% target
        self.assertGreaterEqual(res_idm.realized_annual_vol_pct, 0.6 * 18.0)
        self.assertLessEqual(res_idm.realized_annual_vol_pct, 1.35 * 18.0)

        # 4. Correlation-aware IDM has strictly lower vol overshoot than naive uncorrelated scaling
        cfg_naive = dataclasses.replace(cfg_idm, use_correlation_idm=False)
        res_naive = engine.simulate(_AllLongRankedStrategy(), specs, bars_by_sym, risk_cfg=cfg_naive, capital_usd=10_000.0)
        self.assertLess(
            abs(res_idm.realized_annual_vol_pct - 18.0),
            abs(res_naive.realized_annual_vol_pct - 18.0),
        )

    def test_07_benchmark_gate_g8_rejects_underperforming_bh_d8(self) -> None:
        """D8: G8 rejects v3-style Gold underperforming B&H; accepts outperformance & orthogonal alpha."""
        rng = random.Random(707)
        n = 1260  # 5 years daily
        # Passive GC=F B&H: ~14.5% CAGR, ~0.80 Sharpe
        bench_rets = [0.00060 + rng.gauss(0.0, 0.0110) for _ in range(n)]
        # Underperforming correlated directional strategy (~0.65x beta + drag -> lower CAGR, Sharpe, Calmar)
        weak_strat_rets = [0.65 * b - 0.00004 + rng.gauss(0.0, 0.0040) for b in bench_rets]

        rep_fail = evaluate_benchmark_gate(
            weak_strat_rets, bench_rets, 252.0, benchmark_name="GC=F_BH", strategy_mode="auto"
        )
        self.assertFalse(rep_fail.passed_benchmark_gate)
        self.assertEqual(rep_fail.pass_pathway, "FAILED")
        self.assertLess(rep_fail.strategy_sharpe, rep_fail.benchmark_sharpe)

        # Even if G1..G7 ALL pass, failing G8 MUST NOT promote to PROMOTE_SHADOW_CANDIDATE!
        dummy_dq = [type("DQ", (), {"passed": True, "provenance": PROVENANCE_REAL})()]
        dummy_fa = type("FA", (), {"selected_non_redundant_factors": ["ewmac_16_64"]})()
        good_plateau = select_plateau_center(
            [PlateauPoint(ix, iy, f"p{ix}{iy}", {}, 1.5, 20.0, 5.0, 500.0) for ix in range(3) for iy in range(3)]
        )
        good_nested = run_nested_cpcv(
            180,
            lambda tr, te: FoldResult(oos_returns=tuple(0.006 + rng.gauss(0.0, 0.006) for _ in te)),
            252.0,
        )
        good_dsr = compute_psr_dsr([0.005 + rng.gauss(0.0, 0.007) for _ in range(300)], 252.0, n_trials=5)
        good_pbo = compute_pbo_cscv(
            [[0.008 + rng.gauss(0.0, 0.007) for _ in range(160)]]
            + [[-0.002 + rng.gauss(0.0, 0.01) for _ in range(160)] for _ in range(5)],
            252.0,
            8,
        )
        good_parity = type(
            "PR",
            (),
            {
                "passed": True,
                "max_abs_weight_delta": 0.0,
                "max_abs_equity_delta": 0.0,
                "prefix_max_weight_delta": 0.0,
                "prefix_max_equity_delta": 0.0,
                "regime_mismatch_count": 0,
                "detail": "ok",
            },
        )()
        good_metrics = {"net_pnl_usd": 1500.0, "sharpe_ratio": 1.2, "liquidation_count": 0, "trade_count": 20}

        gate_rep_7_of_8 = evaluate_8_gates(
            "gold_v3_legacy",
            "gold",
            dq_reports=dummy_dq,
            factor_audit=dummy_fa,
            baseline_metrics=good_metrics,
            stress_metrics=good_metrics,
            plateau_report=good_plateau,
            nested_cpcv_report=good_nested,
            dsr_report=good_dsr,
            pbo_report=good_pbo,
            parity_report=good_parity,
            benchmark_report=rep_fail,
        )
        self.assertEqual(gate_rep_7_of_8.passed_gates, 7)
        self.assertFalse(gate_rep_7_of_8.gate_map["G8"])
        self.assertNotEqual(gate_rep_7_of_8.verdict, GateVerdict.PROMOTE_SHADOW_CANDIDATE.value)
        self.assertEqual(gate_rep_7_of_8.verdict, GateVerdict.CONDITIONAL_RESEARCH.value)

        # Legitimate PASS #1: Directional strategy beating B&H on Sharpe/Calmar -> 8/8 PROMOTE_SHADOW_CANDIDATE
        strong_strat_rets = [0.85 * b + 0.00045 + rng.gauss(0.0, 0.003) for b in bench_rets]
        rep_dir = evaluate_benchmark_gate(strong_strat_rets, bench_rets, 252.0, benchmark_name="GC=F_BH")
        self.assertTrue(rep_dir.passed_benchmark_gate)
        self.assertEqual(rep_dir.pass_pathway, "DIRECTIONAL_OUTPERFORMANCE")

        gate_rep_8_of_8 = evaluate_8_gates(
            "gold_v4_strong",
            "gold",
            dq_reports=dummy_dq,
            factor_audit=dummy_fa,
            baseline_metrics=good_metrics,
            stress_metrics=good_metrics,
            plateau_report=good_plateau,
            nested_cpcv_report=good_nested,
            dsr_report=good_dsr,
            pbo_report=good_pbo,
            parity_report=good_parity,
            benchmark_report=rep_dir,
        )
        self.assertEqual(gate_rep_8_of_8.passed_gates, 8)
        self.assertEqual(gate_rep_8_of_8.verdict, GateVerdict.PROMOTE_SHADOW_CANDIDATE.value)

        # Legitimate PASS #2: Orthogonal carry/crisis alpha (|corr| <= 0.30, positive SR below strong B&H)
        strong_bench = [0.0012 + rng.gauss(0.0, 0.010) for _ in range(n)]
        orth_rets = [0.00020 + rng.gauss(0.0, 0.0065) for _ in range(n)]
        rep_orth = evaluate_benchmark_gate(orth_rets, strong_bench, 252.0, benchmark_name="STRONG_BH", strategy_mode="auto")
        self.assertTrue(rep_orth.passed_benchmark_gate)
        self.assertEqual(rep_orth.pass_pathway, "ORTHOGONAL_ALPHA")
        self.assertLessEqual(abs(rep_orth.corr_to_benchmark), 0.30)
        self.assertGreater(rep_orth.strategy_sharpe, 0.0)

        # Non-nested CPCVReport MUST fail G5; missing G7 MUST produce REJECTED!
        non_nested = CPCVReport(6, 2, 15, 2, 2, [1.5] * 15, 1.5, 1.0, 100.0, 0.9, is_nested=False)
        rep_non_nested = evaluate_8_gates(
            "non_nested_test",
            "web3",
            dq_reports=dummy_dq,
            factor_audit=dummy_fa,
            baseline_metrics=good_metrics,
            stress_metrics=good_metrics,
            plateau_report=good_plateau,
            nested_cpcv_report=non_nested,
            dsr_report=good_dsr,
            pbo_report=good_pbo,
            parity_report=None,
            benchmark_report=rep_dir,
        )
        self.assertFalse(rep_non_nested.gate_map["G5"])
        self.assertFalse(rep_non_nested.gate_map["G7"])
        self.assertEqual(rep_non_nested.verdict, GateVerdict.REJECTED.value)

        # Verify passive B&H helper on single and multi-symbol bars
        bh_bars = [_bar("GC=F", BASE_TS_MS + i * DAY_MS, 100.0 + i, 100.0 + i, 100.0 + i, 100.0 + i) for i in range(10)]
        bh_rets = compute_buy_and_hold_returns(bh_bars)
        self.assertEqual(len(bh_rets), 9)
        self.assertAlmostEqual(bh_rets[0], 101.0 / 100.0 - 1.0, places=12)

    def test_08_cagr_t_plus_1_exchange_date_carry_and_margin_d13_d15(self) -> None:
        """D13 & D15: True CAGR, intraday A-share T+1 date lock, per-symbol carry, leveraged/unleveraged guard, contract_multiplier."""
        engine = PortfolioAccountingEngine()
        zero_cost = CostModel(maker_fee_bps=0.0, taker_fee_bps=0.0, slippage_bps_per_leg=0.0)

        # Part A (D13): 5y compounding equity 10,000 -> 15,864 over 1,825 days -> CAGR ~ 9.67%, simple APR ~ 11.73%
        spec_etf = AssetSpec("GLD", ASSET_CLASS_GOLD_TRADFI, "yahoo", CALENDAR_US_DAILY, zero_cost, max_leverage_cap=1.0)
        bars_5y = [
            _bar("GLD", BASE_TS_MS, 100.0, 100.0, 100.0, 100.0, tf_h=24.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
            _bar("GLD", BASE_TS_MS + 1 * DAY_MS, 100.0, 100.0, 100.0, 100.0, tf_h=24.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
            _bar("GLD", BASE_TS_MS + 1825 * DAY_MS, 158.64, 158.64, 158.64, 158.64, tf_h=24.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
        ]
        res_5y = engine.simulate(
            _ScriptedStrategy({0: {"GLD": (1.0, 5.0, None)}, 1: {"GLD": (1.0, 5.0, None)}, 2: {"GLD": (1.0, 5.0, None)}}, asset_domain="gold"),
            {"GLD": spec_etf},
            {"GLD": bars_5y},
            risk_cfg=RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        self.assertAlmostEqual(res_5y.cagr_pct, 9.67, delta=0.05)
        self.assertAlmostEqual(res_5y.simple_apr_pct, 11.73, delta=0.05)

        # Part B (D15a): Intraday 1h CN A-Share T+1 lock (Asia/Shanghai UTC+8)
        ash_spec = default_ashare_template_spec("600519.SH")
        # Day 1 CST 09:30, 10:30, 13:30, 14:30 -> UTC 01:30, 02:30, 05:30, 06:30 on 2025-01-02
        # Day 2 CST 09:30 -> UTC 01:30 on 2025-01-03
        d1_base = BASE_TS_MS + DAY_MS + int(1.5 * HOUR_MS)  # 2025-01-02 01:30 UTC == 09:30 CST
        d2_base = d1_base + DAY_MS                          # 2025-01-03 01:30 UTC == 09:30 CST
        self.assertEqual(exchange_trading_date(d1_base, CALENDAR_CN_A_SHARE).isoformat(), "2025-01-02")
        self.assertEqual(exchange_trading_date(d1_base + 5 * HOUR_MS, CALENDAR_CN_A_SHARE).isoformat(), "2025-01-02")
        self.assertEqual(exchange_trading_date(d2_base, CALENDAR_CN_A_SHARE).isoformat(), "2025-01-03")

        ash_bars = [
            _bar("600519.SH", d1_base + 0 * HOUR_MS, 100.0, 101.0, 99.5, 100.5, tf_h=1.0, asset_class=ASSET_CLASS_CN_A_SHARE),
            _bar("600519.SH", d1_base + 1 * HOUR_MS, 100.5, 101.5, 100.0, 101.0, tf_h=1.0, asset_class=ASSET_CLASS_CN_A_SHARE),
            _bar("600519.SH", d1_base + 4 * HOUR_MS, 101.0, 101.5, 100.5, 101.0, tf_h=1.0, asset_class=ASSET_CLASS_CN_A_SHARE),
            _bar("600519.SH", d1_base + 5 * HOUR_MS, 101.0, 101.5, 100.5, 101.0, tf_h=1.0, asset_class=ASSET_CLASS_CN_A_SHARE),
            _bar("600519.SH", d2_base + 0 * HOUR_MS, 102.0, 102.5, 101.5, 102.0, tf_h=1.0, asset_class=ASSET_CLASS_CN_A_SHARE),
        ]
        # Bar 0 signals buy +0.8 (fills on Day 1 bar 1 open); Bar 1, 2, 3 all try to exit -> MUST be locked until Day 2 bar 4!
        ash_sched = {
            0: {"600519.SH": (0.8, 10.0, None)},
            1: {"600519.SH": (0.0, -5.0, None)},
            2: {"600519.SH": (0.0, -5.0, None)},
            3: {"600519.SH": (0.0, -5.0, None)},
            4: {"600519.SH": (0.0, -5.0, None)},
        }
        res_ash = engine.simulate(
            _ScriptedStrategy(ash_sched, asset_domain="ashare"),
            {"600519.SH": ash_spec},
            {"600519.SH": ash_bars},
            risk_cfg=RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        # Day 1 bars 2 & 3 MUST remain held at 0.8 (blocked by T_PLUS_1_LOCK); Day 2 bar 4 MUST exit to 0.0!
        self.assertAlmostEqual(res_ash.executed_weights_history[1]["600519.SH"], 0.8, places=6)
        self.assertAlmostEqual(res_ash.executed_weights_history[2]["600519.SH"], 0.8, places=6)
        self.assertAlmostEqual(res_ash.executed_weights_history[3]["600519.SH"], 0.8, places=6)
        self.assertAlmostEqual(res_ash.executed_weights_history[4]["600519.SH"], 0.0, places=6)
        self.assertGreaterEqual(res_ash.trade_ledger.suppression_counts().get("T_PLUS_1_LOCK", 0), 2)
        self.assertGreater(res_ash.trade_ledger.total_stamp_duty_usd, 0.0)

        # Also verify AdaptiveMultiFactorKernel enforces exchange-date T+1 on intraday 1h A-share bars!
        ash_k_cfg = KernelConfig(
            strategy_name="ash_intraday_t1",
            asset_class=ASSET_CLASS_CN_A_SHARE,
            timeframe_h=1.0,
            max_positions=1,
            allow_short=False,
            min_hold_bars=1,
            warmup_bars=0,
            donchian_weight=0.0,
            regime_exposure_map={REGIME_TREND_CALM: 1.0},
            risk_per_trade_pct=50.0,
        )
        ash_k = AdaptiveMultiFactorKernel(ash_k_cfg, ash_spec)
        k_decs = []
        for bi, b in enumerate(ash_bars):
            f_val = 12.0 if bi == 0 else -12.0
            sp = FactorSnapshot(
                symbol=b.symbol, venue=b.venue, timeframe_h=1.0,
                ts_event_ms=b.ts_event_ms, ts_avail_ms=b.ts_avail_ms,
                close=b.close, atr_14=1.0,
                prior_high_40=b.high, prior_low_40=b.low,
                prior_high_20=b.high, prior_low_20=b.low,
                values={"ewmac_8_32": f_val, "ewmac_16_64": f_val, "ewmac_32_128": f_val, "realized_vol_ann_20": 20.0, "rel_volume_20": 1.0},
            )
            rp = RegimePoint(b.ts_event_ms, b.ts_avail_ms, REGIME_TREND_CALM, REGIME_TREND_CALM, 25.0, 0.4, 0.5, 1.0, 0.5, 0.0)
            k_decs.append(ash_k.decide_bar(b, sp, rp))
        self.assertGreater(k_decs[0].target_weight, 0.0)
        for same_day_i in (1, 2, 3):
            self.assertEqual(k_decs[same_day_i].reason, "T_PLUS_1_LOCK")
            self.assertAlmostEqual(k_decs[same_day_i].target_weight, k_decs[0].target_weight, places=6)
        self.assertAlmostEqual(k_decs[4].target_weight, 0.0, places=6)

        # Part C (D15b): Per-symbol calendar carry drag in mixed CRYPTO_24_7 + US_EQUITY_DAILY portfolio
        drag_cost_c = CostModel(0.0, 0.0, 0.0, annual_carry_drag_bps=365.0)  # 3.65% / 365 = 0.0001 per bar
        drag_cost_u = CostModel(0.0, 0.0, 0.0, annual_carry_drag_bps=252.0)  # 2.52% / 252 = 0.0001 per bar
        spec_c = AssetSpec("BTC", ASSET_CLASS_CRYPTO_PERP, "hl", CALENDAR_CRYPTO_24_7, drag_cost_c, max_leverage_cap=1.0)
        spec_u = AssetSpec("GLD", ASSET_CLASS_GOLD_TRADFI, "yahoo", CALENDAR_US_DAILY, drag_cost_u, max_leverage_cap=1.0)
        mix_bars = {
            "BTC": [_bar("BTC", BASE_TS_MS + i * DAY_MS, 100.0, 100.0, 100.0, 100.0, tf_h=24.0) for i in range(3)],
            "GLD": [_bar("GLD", BASE_TS_MS + i * DAY_MS, 100.0, 100.0, 100.0, 100.0, tf_h=24.0, asset_class=ASSET_CLASS_GOLD_TRADFI) for i in range(3)],
        }
        mix_sched = {
            0: {"BTC": (0.5, 5.0, None), "GLD": (0.5, 5.0, None)},
            1: {"BTC": (0.5, 5.0, None), "GLD": (0.5, 5.0, None)},
            2: {"BTC": (0.5, 5.0, None), "GLD": (0.5, 5.0, None)},
        }
        res_mix = engine.simulate(
            _ScriptedStrategy(mix_sched),
            {"BTC": spec_c, "GLD": spec_u},
            mix_bars,
            risk_cfg=RiskAllocatorConfig(max_positions=2, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        # Bar 1 & Bar 2: each symbol holds $5,000 * 0.0001 = -$0.50/bar -> equal -$1.00 drag per symbol!
        self.assertAlmostEqual(res_mix.by_symbol_net_usd["BTC"], res_mix.by_symbol_net_usd["GLD"], places=3)
        self.assertAlmostEqual(res_mix.holding_drag_usd, -2.0, delta=0.02)

        # Part D (D15c): Unleveraged ETF dropping -76% is NOT wiped to $0; 2x leveraged perp intrabar -55% IS liquidated
        etf_crash_bars = [
            _bar("GLD", BASE_TS_MS + 0 * DAY_MS, 100.0, 100.0, 100.0, 100.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
            _bar("GLD", BASE_TS_MS + 1 * DAY_MS, 100.0, 100.0, 24.0, 24.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
            _bar("GLD", BASE_TS_MS + 2 * DAY_MS, 24.0, 24.0, 24.0, 24.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
        ]
        res_etf_crash = engine.simulate(
            _ScriptedStrategy({0: {"GLD": (1.0, 5.0, None)}, 1: {"GLD": (1.0, 5.0, None)}, 2: {"GLD": (1.0, 5.0, None)}}, asset_domain="gold"),
            {"GLD": spec_etf},
            {"GLD": etf_crash_bars},
            risk_cfg=RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        self.assertEqual(res_etf_crash.liquidation_count, 0)
        self.assertAlmostEqual(res_etf_crash.final_equity_usd, 2400.0, places=4)

        spec_perp = AssetSpec("BTC", ASSET_CLASS_CRYPTO_PERP, "hl", CALENDAR_CRYPTO_24_7, zero_cost, max_leverage_cap=3.0)
        perp_crash_bars = [
            _bar("BTC", BASE_TS_MS + 0 * DAY_MS, 100.0, 100.0, 100.0, 100.0),
            # Intrabar low crashes -55% even though close rebounds to 90 -> 2x long MUST be intrabar liquidated!
            _bar("BTC", BASE_TS_MS + 1 * DAY_MS, 100.0, 100.0, 45.0, 90.0),
            _bar("BTC", BASE_TS_MS + 2 * DAY_MS, 90.0, 90.0, 90.0, 90.0),
        ]
        res_perp_crash = engine.simulate(
            _ScriptedStrategy({0: {"BTC": (2.0, 10.0, None)}, 1: {"BTC": (2.0, 10.0, None)}, 2: {"BTC": (0.0, 0.0, None)}}),
            {"BTC": spec_perp},
            {"BTC": perp_crash_bars},
            risk_cfg=RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=2.5, max_single_weight=2.0, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        self.assertGreaterEqual(res_perp_crash.liquidation_count, 1)
        self.assertTrue(any(t.reason == "INTRABAR_LIQUIDATION" for t in res_perp_crash.trade_ledger.executed))

        # Part E (D15d): contract_multiplier & exact PnL reconciliation + 3-tier REA-001 ledger
        class _ComexSpecProxy:
            def __init__(self, base: AssetSpec, mult: float) -> None:
                self._base = base
                self.contract_multiplier = mult

            def __getattr__(self, item: str) -> Any:
                return getattr(self._base, item)

        comex_spec = _ComexSpecProxy(
            AssetSpec("GC=F", ASSET_CLASS_GOLD_TRADFI, "yahoo", CALENDAR_GOLD_TRADFI_23_5, COST_GOLD_FUTURES_ETF, max_leverage_cap=2.0),
            100.0,
        )
        comex_bars = [
            _bar("GC=F", BASE_TS_MS + 0 * DAY_MS, 2000.0, 2010.0, 1995.0, 2000.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
            _bar("GC=F", BASE_TS_MS + 1 * DAY_MS, 2000.0, 2050.0, 1990.0, 2040.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
            _bar("GC=F", BASE_TS_MS + 2 * DAY_MS, 2050.0, 2060.0, 2040.0, 2050.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
        ]
        res_comex = engine.simulate(
            _ScriptedStrategy({0: {"GC=F": (1.5, 8.0, None)}, 1: {"GC=F": (0.0, 0.0, None)}, 2: {"GC=F": (0.0, 0.0, None)}}, asset_domain="gold"),
            {"GC=F": comex_spec},  # type: ignore[dict-item]
            {"GC=F": comex_bars},
            risk_cfg=RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False),
            capital_usd=100_000.0,
        )
        self.assertGreater(res_comex.trade_ledger.candidate_count, 0)
        self.assertGreater(res_comex.trade_ledger.executed_count, 0)
        self.assertGreater(res_comex.trade_ledger.suppressed_count, 0)  # 1.5 capped to 1.0!
        first_exec = res_comex.trade_ledger.executed[0]
        self.assertEqual(first_exec.contract_multiplier, 100.0)
        self.assertAlmostEqual(first_exec.contracts, first_exec.notional_usd / (first_exec.fill_price * 100.0), places=8)
        # Exact PnL identity: net_pnl == gross_price + funding_carry - total_cost
        reconciled = res_comex.gross_price_pnl_usd + res_comex.funding_carry_pnl_usd - res_comex.total_cost_usd
        self.assertAlmostEqual(res_comex.net_pnl_usd, reconciled, places=6)

    def test_09_gate_8h_carry_apr_normalization_d11(self) -> None:
        """D11: carry_apr_24h normalizes properly by carry_interval_h across Gate sparse/dense & HL 1h."""
        # A. Real Gate sparse 8h on 1h bars: r=0.0001 every 8th bar, 0.0 otherwise -> exact 10.95% APR, 0 z-score!
        eng_gate_sparse = StreamingFactorEngine(CALENDAR_CRYPTO_24_7, 1.0)
        snaps_gs = []
        for i in range(48):
            r = 0.0001 if (i % 8 == 0) else 0.0
            b = _bar("BTC", BASE_TS_MS + i * HOUR_MS, 100.0, 100.5, 99.5, 100.0, tf_h=1.0, venue="gate", carry_rate_period=r, carry_interval_h=8.0)
            snaps_gs.append(eng_gate_sparse.update(b))
        for idx in (0, 8, 23, 47):
            self.assertAlmostEqual(snaps_gs[idx].get("carry_apr_24h"), 10.95, delta=0.05)
            self.assertLess(abs(snaps_gs[idx].get("carry_zscore_60")), 0.01)

        # B. Hyperliquid continuous 1h: r=1.25e-5 every bar, ci=1.0 -> exact 10.95% APR
        eng_hl = StreamingFactorEngine(CALENDAR_CRYPTO_24_7, 1.0)
        snaps_hl = [
            eng_hl.update(_bar("BTC", BASE_TS_MS + i * HOUR_MS, 100.0, 100.5, 99.5, 100.0, tf_h=1.0, venue="hyperliquid", carry_rate_period=1.25e-5, carry_interval_h=1.0))
            for i in range(30)
        ]
        self.assertAlmostEqual(snaps_hl[-1].get("carry_apr_24h"), 10.95, delta=0.05)

        # C. Dense synthetic Gate (ci=8.0 on every bar) -> still exact 10.95% APR (no 8x inflation!)
        eng_dense = StreamingFactorEngine(CALENDAR_CRYPTO_24_7, 1.0)
        snaps_dense = [
            eng_dense.update(_bar("BTC", BASE_TS_MS + i * HOUR_MS, 100.0, 100.5, 99.5, 100.0, tf_h=1.0, venue="gate", carry_rate_period=0.0001, carry_interval_h=8.0))
            for i in range(30)
        ]
        self.assertAlmostEqual(snaps_dense[-1].get("carry_apr_24h"), 10.95, delta=0.05)

        # D. Negative funding (r=-0.0001, ci=8.0) -> exact -10.95% APR
        eng_neg = StreamingFactorEngine(CALENDAR_CRYPTO_24_7, 1.0)
        snaps_neg = [
            eng_neg.update(_bar("BTC", BASE_TS_MS + i * HOUR_MS, 100.0, 100.5, 99.5, 100.0, tf_h=1.0, venue="gate", carry_rate_period=(-0.0001 if i % 8 == 0 else 0.0), carry_interval_h=8.0))
            for i in range(32)
        ]
        self.assertAlmostEqual(snaps_neg[-1].get("carry_apr_24h"), -10.95, delta=0.05)

    def test_10_d12_hac_phase_invariant_ic_and_sparse_event_audit(self) -> None:
        """D12: Phase-averaged IC eliminates warmup phase lottery, NW HAC deflates overlap, sparse events audited."""
        bars = _trend_bars("PAXG", 210, tf_h=12.0, seed=101, vol=0.011)
        snaps = StreamingFactorEngine(CALENDAR_CRYPTO_24_7, 12.0).compute_series(bars)

        # Part A1: Phase invariance across warmup_bars in (45, 46, 47, 48) with H=8
        ics_ewmac: list[float] = []
        for wb in (45, 46, 47, 48):
            aud = FactorDirectionAuditor(warmup_bars=wb)
            summ = aud.audit("paxg_d12", {"PAXG": bars}, {"PAXG": snaps}, horizon_bars=8, candidate_factors=["ewmac_16_64"])
            ics_ewmac.append(summ.results[0].spearman_ic)
        self.assertLess(max(ics_ewmac) - min(ics_ewmac), 0.04)

        # Part A2: On persistent overlapping H=8 factor, |hac_t_stat| < |naive_t_stat|
        xs = [snaps[i].get("ewmac_16_64") for i in range(45, len(bars) - 9)]
        ys = [(bars[i + 9].close / bars[i + 1].open - 1.0) * 10_000.0 for i in range(45, len(bars) - 9)]
        hac_rep = compute_hac_factor_ic(xs, ys, horizon_bars=8, factor_name="ewmac_16_64")
        self.assertLess(abs(hac_rep.hac_t_stat), abs(hac_rep.naive_t_stat))
        self.assertEqual(len(hac_rep.phase_ics), 8)

        # Segment boundary isolation in newey_west_mean_stats
        s1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        s2 = [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0]
        nw_iso = newey_west_mean_stats(s1 + s2, max_lag=2, segment_lengths=[8, 8])
        nw_raw = newey_west_mean_stats(s1 + s2, max_lag=2)
        self.assertNotAlmostEqual(nw_iso.hac_variance, nw_raw.hac_variance, places=4)

        # Part B: Sparse event expectancy (positive trigger PASS vs sparse noise FAIL)
        rng = random.Random(102)
        n_ev_bars = 320
        ev_bars = []
        px = 100.0
        pos_triggers = set(range(50, 290, 12))  # 20 strictly non-overlapping events
        for i in range(n_ev_bars):
            o = px
            # If bar i-1 triggered positive event, inject +220bp move over next 4 bars
            boost = 0.006 if any((i - k) in pos_triggers for k in (1, 2, 3, 4)) else 0.0
            c = max(2.0, o * math.exp(boost + rng.gauss(0.0, 0.004)))
            ev_bars.append(_bar("BTC", BASE_TS_MS + i * 12 * HOUR_MS, o, max(o, c), min(o, c), c, tf_h=12.0))
            px = c

        ev_snaps = []
        for i, b in enumerate(ev_bars):
            noise_trig = rng.choice((-1.0, 1.0)) if (i >= 45 and rng.random() < 0.08) else 0.0
            ev_snaps.append(
                FactorSnapshot(
                    symbol="BTC",
                    venue="test",
                    timeframe_h=12.0,
                    ts_event_ms=b.ts_event_ms,
                    ts_avail_ms=b.ts_avail_ms,
                    close=b.close,
                    atr_14=1.0,
                    prior_high_40=b.high,
                    prior_low_40=b.low,
                    prior_high_20=b.high,
                    prior_low_20=b.low,
                    values={
                        "sparse_alpha": 1.0 if i in pos_triggers else 0.0,
                        "sparse_noise": noise_trig,
                    },
                )
            )
        aud_ev = FactorDirectionAuditor(warmup_bars=45, min_event_count=12, min_t_stat=1.28)
        summ_ev = aud_ev.audit(
            "sparse_test",
            {"BTC": ev_bars},
            {"BTC": ev_snaps},
            horizon_bars=4,
            candidate_factors=["sparse_alpha", "sparse_noise"],
            priors={"sparse_alpha": +1, "sparse_noise": +1},
        )
        by_f = {r.factor_name: r for r in summ_ev.results}
        self.assertEqual(by_f["sparse_alpha"].audit_mode, "sparse_event")
        self.assertGreaterEqual(by_f["sparse_alpha"].event_count, 15)
        self.assertGreater(by_f["sparse_alpha"].event_mean_return_bps, 100.0)
        self.assertTrue(by_f["sparse_alpha"].passed_g2)
        self.assertEqual(by_f["sparse_noise"].audit_mode, "sparse_event")
        self.assertFalse(by_f["sparse_noise"].passed_g2)

        # Direct compute_sparse_event_expectancy check
        direct_ev = compute_sparse_event_expectancy(
            [1.0 if i % 10 == 0 else 0.0 for i in range(200)],
            [0.025 if i % 10 == 0 else rng.gauss(0.0, 0.01) for i in range(200)],
            horizon_bars=4,
            min_events=10,
        )
        self.assertTrue(direct_ev.passed_event_gate)
        self.assertEqual(direct_ev.n_events, 20)
        self.assertAlmostEqual(direct_ev.mean_fwd_ret_bps, 250.0, places=2)

    def test_11_d5_cscv_oldest_residue_slicing_and_diagnostics(self) -> None:
        """D5: CSCV drops oldest warmup residue (NEVER newest live bars), fast stdlib speed, micro-grid diagnostics."""
        rng = random.Random(111)
        t = 164  # 164 % 8 = 4 residue bars
        base_c0 = [0.005 + rng.gauss(0.0, 0.003) for _ in range(t)]
        others = [[0.001 + rng.gauss(0.0, 0.006) for _ in range(t)] for _ in range(7)]

        # Poison ONLY oldest 4 bars [0:4] -> MUST have ZERO effect on CSCV output!
        c0_old_poison = [-0.90] * 4 + base_c0[4:]
        rep_clean = compute_pbo_cscv([base_c0] + others, 252.0, n_subsamples=8)
        rep_old = compute_pbo_cscv([c0_old_poison] + others, 252.0, n_subsamples=8)
        self.assertEqual(rep_clean.dropped_oldest_bars, 4)
        self.assertEqual(rep_clean.to_dict(), rep_old.to_dict())

        # Poison ONLY newest 4 bars [160:164] -> MUST materially impact Candidate 0's IS/OOS Sharpe!
        c0_new_poison = base_c0[:-4] + [-0.25] * 4
        rep_new = compute_pbo_cscv([c0_new_poison] + others, 252.0, n_subsamples=8)
        self.assertLess(rep_new.median_oos_winner_sharpe, rep_clean.median_oos_winner_sharpe - 0.5)

        # Micro-grid flat twins flag is_flat_micro_grid=True and effective_spread_over_se < 0.50
        micro_base = [0.003 + rng.gauss(0.0, 0.01) for _ in range(200)]
        micro_twins = [[x + rng.gauss(0.0, 0.0001) for x in micro_base] for _ in range(9)]
        rep_micro = compute_pbo_cscv(micro_twins, 252.0, n_subsamples=8)
        self.assertTrue(rep_micro.is_flat_micro_grid)
        self.assertLess(rep_micro.effective_spread_over_se, 0.50)
        self.assertGreater(rep_micro.avg_pairwise_corr, 0.95)

        # Plateau-center OOS degradation discriminates consistent vs single-block regime-concentrated fragile curve
        consistent_mat = [[0.006 + rng.gauss(0.0, 0.004) for _ in range(160)] for _ in range(5)]
        fragile_row = [-0.002 + rng.gauss(0.0, 0.005) for _ in range(140)] + [0.040 + rng.gauss(0.0, 0.005) for _ in range(20)]
        fragile_mat = [[x + rng.gauss(0.0, 0.001) for x in fragile_row] for _ in range(5)]
        rep_cons = compute_pbo_cscv(consistent_mat, 252.0, n_subsamples=8)
        rep_frag = compute_pbo_cscv(fragile_mat, 252.0, n_subsamples=8)
        self.assertLess(rep_cons.plateau_center_oos_degradation, 0.40)
        self.assertGreater(rep_frag.plateau_center_oos_degradation, 0.70)

        # Pure-stdlib speed check: 16 cands x 1,000 bars x S=8 MUST finish in < 0.5s!
        big_mat = [[rng.gauss(0.0, 0.01) for _ in range(1000)] for _ in range(16)]
        t0 = time.perf_counter()
        rep_big = compute_pbo_cscv(big_mat, 252.0, n_subsamples=8)
        elapsed = time.perf_counter() - t0
        self.assertEqual(rep_big.n_combinations, 70)
        self.assertLess(elapsed, 0.5, f"CSCV 16x1000xS8 took {elapsed:.3f}s (>0.5s regression!)")

    def test_12_g7_full_kernel_plus_risk_allocator_bit_parity(self) -> None:
        """G7: Full kernel + CausalRiskAllocator streaming-vs-batch bit-parity AND negative perturbation trap."""
        syms = ["BTC", "ETH", "SOL", "AVAX", "LINK"]
        specs = default_web3_specs(syms, venue="hyperliquid")
        bars_by_sym = {s: _trend_bars(s, 120, tf_h=12.0, seed=120 + idx) for idx, s in enumerate(syms)}
        risk_cfg = RiskAllocatorConfig(
            max_positions=3,
            target_annual_vol_pct=18.0,
            max_portfolio_leverage=1.4,
            max_single_weight=0.5,
            use_correlation_idm=True,
            corr_lookback_bars=40,
            corr_shrinkage=0.25,
        )

        strat = _StatefulInertiaStrategy(syms, warmup_bars=15, buffer_pct=0.15)
        engine = PortfolioAccountingEngine()
        batch_res = engine.simulate(strat.clone_fresh(), specs, bars_by_sym, risk_cfg=risk_cfg, capital_usd=10_000.0)
        self.assertLessEqual(batch_res.max_active_positions, 3)

        # Positive test: bit-identical parity + 60% prefix truncation causality check
        parity_ok = verify_full_parity(strat, specs, bars_by_sym, risk_cfg=risk_cfg, batch_result=batch_res, capital_usd=10_000.0)
        self.assertTrue(parity_ok.passed, parity_ok.detail)
        self.assertLessEqual(parity_ok.max_abs_weight_delta, 1e-10)
        self.assertLessEqual(parity_ok.max_abs_equity_delta, 1e-8)
        self.assertLessEqual(parity_ok.prefix_max_weight_delta, 1e-10)
        self.assertLessEqual(parity_ok.prefix_max_equity_delta, 1e-8)
        self.assertEqual(parity_ok.regime_mismatch_count, 0)

        # Negative test 1: Perturb a single allocated weight in batch_result by 1e-4 -> MUST FAIL!
        tampered = copy.deepcopy(batch_res)
        ts_t, w_t = tampered.allocated_weights_history[40]
        w_t = dict(w_t)
        first_sym = sorted(w_t)[0]
        w_t[first_sym] = w_t[first_sym] + 1e-4
        tampered.allocated_weights_history[40] = (ts_t, w_t)
        parity_tampered = verify_full_parity(strat, specs, bars_by_sym, risk_cfg=risk_cfg, batch_result=tampered, capital_usd=10_000.0)
        self.assertFalse(parity_tampered.passed)
        self.assertGreaterEqual(parity_tampered.max_abs_weight_delta, 9e-5)

        # Negative test 2: Non-deterministic strategy -> MUST FAIL!
        bad_strat = _StatefulInertiaStrategy(syms, warmup_bars=15, buffer_pct=0.15, inject_nondeterminism=True)
        parity_bad = verify_full_parity(bad_strat, specs, bars_by_sym, risk_cfg=risk_cfg, capital_usd=10_000.0)
        self.assertFalse(parity_bad.passed)

    def test_13_portfolio_combiner_return_stream_level_f11_and_v01_scrub(self) -> None:
        """F-11 & V-01: Cross-calendar causal PortfolioCombiner, REA-001 margin separation, V-01 scrub, float accuracy."""
        rng = random.Random(1313)

        def _make_sleeve_result(
            sid: str,
            domain: str,
            n_days: int,
            daily_vol: float,
            daily_drift: float,
            *,
            skip_weekends: bool,
            capital: float,
            max_lev: float,
        ) -> SimulationResult:
            eq = capital
            curve: list[tuple[int, float]] = []
            ts_list: list[int] = []
            rets: list[float] = []
            for d in range(n_days):
                if skip_weekends and (d % 7 in (5, 6)):
                    continue
                r = daily_drift + rng.gauss(0.0, daily_vol)
                if curve:
                    eq *= max(0.01, 1.0 + r)
                    rets.append(r)
                ts = BASE_TS_MS + d * DAY_MS
                curve.append((ts, eq))
                ts_list.append(ts)
            return SimulationResult(
                strategy_id=sid,
                asset_domain=domain,
                capital_usd=capital,
                final_equity_usd=eq,
                days=float(n_days),
                bars_simulated=len(rets),
                bars_per_year=252.0 if skip_weekends else 365.0,
                trade_count=30,
                total_turnover=10.0,
                gross_price_pnl_usd=eq - capital,
                gap_pnl_usd=0.0,
                intraday_pnl_usd=eq - capital,
                funding_pnl_usd=0.0,
                holding_drag_usd=0.0,
                funding_carry_pnl_usd=0.0,
                total_fee_usd=0.0,
                total_slippage_usd=0.0,
                total_stamp_duty_usd=0.0,
                total_cost_usd=0.0,
                net_pnl_usd=eq - capital,
                total_return_pct=((eq / capital) - 1.0) * 100.0,
                cagr_pct=(((eq / capital) ** (365.0 / n_days)) - 1.0) * 100.0,
                simple_apr_pct=((eq / capital) - 1.0) * (365.0 / n_days) * 100.0,
                sharpe_ratio=sharpe_from_returns(rets, 252.0 if skip_weekends else 365.0),
                calmar_ratio=1.0,
                realized_annual_vol_pct=fast_var_std(rets, ddof=1)[2] * math.sqrt(252.0 if skip_weekends else 365.0) * 100.0,
                max_drawdown_pct=5.0,
                max_active_positions=3,
                mean_active_positions=2.0,
                max_gross_leverage=max_lev,
                liquidation_count=0,
                equity_curve=curve,
                timestamps_ms=ts_list,
                period_returns=rets,
                period_timestamps_ms=ts_list[1:],
            )

        # 24/7 Web3 sleeve (~24% vol) + 252d TradFi Gold sleeve (~10% vol) across 260 calendar days
        s_web3 = _make_sleeve_result("w2_trend", "web3", 260, daily_vol=0.24 / math.sqrt(365.0), daily_drift=0.0005, skip_weekends=False, capital=5000.0, max_lev=1.5)
        s_gold = _make_sleeve_result("g1_macro", "gold", 260, daily_vol=0.10 / math.sqrt(252.0), daily_drift=0.0004, skip_weekends=True, capital=5000.0, max_lev=1.2)
        snap_w3_before = copy.deepcopy(s_web3.to_dict(include_curves=True))
        snap_g_before = copy.deepcopy(s_gold.to_dict(include_curves=True))

        combiner = PortfolioCombiner()
        comb_full = combiner.combine_sleeves(
            {"web3": s_web3, "gold": s_gold},
            method="causal_inv_vol",
            target_vol_pct=15.0,
            vol_lookback_days=25,
            rebalance_every_days=5,
        )
        self.assertTrue(comb_full.valid, comb_full.error)
        # Input SimulationResult objects MUST remain completely untouched!
        self.assertEqual(s_web3.to_dict(include_curves=True), snap_w3_before)
        self.assertEqual(s_gold.to_dict(include_curves=True), snap_g_before)

        # 1. Lower-vol Gold sleeve MUST receive strictly higher InvVol weight than higher-vol Web3 sleeve
        self.assertGreater(comb_full.sleeve_weights_final["gold"], comb_full.sleeve_weights_final["web3"])
        # 2. Diversification ratio > 1.05 for uncorrelated sleeves
        self.assertGreater(comb_full.diversification_ratio, 1.05)
        self.assertLess(abs(comb_full.pairwise_corr_matrix["web3"]["gold"]), 0.25)

        # 3. REA-001 margin bucket separation & exact capital conservation
        alloc_w3 = comb_full.sleeve_capital_allocations["web3"]
        alloc_g = comb_full.sleeve_capital_allocations["gold"]
        self.assertEqual(alloc_w3.margin_bucket, "web3_perp_margin")
        self.assertEqual(alloc_g.margin_bucket, "gold_tradfi_margin")
        conserved = (
            alloc_w3.allocated_capital_usd
            + alloc_g.allocated_capital_usd
            + comb_full.total_capital_usd * comb_full.cash_reserve_weight
        )
        self.assertAlmostEqual(conserved, comb_full.total_capital_usd, places=6)

        # 4. Causal prefix truncation: weights on days 0..k MUST be bit-identical (zero lookahead!)
        cut_ts = comb_full.equity_curve[90][0]
        s_web3_pre = dataclasses.replace(s_web3, equity_curve=[pt for pt in s_web3.equity_curve if pt[0] <= cut_ts])
        s_gold_pre = dataclasses.replace(s_gold, equity_curve=[pt for pt in s_gold.equity_curve if pt[0] <= cut_ts])
        comb_pre = combiner.combine_sleeves(
            {"web3": s_web3_pre, "gold": s_gold_pre},
            method="causal_inv_vol",
            target_vol_pct=15.0,
            vol_lookback_days=25,
            rebalance_every_days=5,
        )
        self.assertTrue(comb_pre.valid)
        self.assertGreaterEqual(len(comb_pre.sleeve_weights_history), 80)
        for i, (ts_p, w_p) in enumerate(comb_pre.sleeve_weights_history):
            ts_f, w_f = comb_full.sleeve_weights_history[i]
            self.assertEqual(ts_p, ts_f)
            for sid in ("web3", "gold"):
                self.assertAlmostEqual(w_p[sid], w_f[sid], places=12)

        # Also exercise causal_corr_aware cleanly
        comb_ca = combiner.combine_sleeves({"web3": s_web3, "gold": s_gold}, method="causal_corr_aware", target_vol_pct=15.0)
        self.assertTrue(comb_ca.valid)
        self.assertGreater(comb_ca.diversification_ratio, 1.05)

        # 5. V-01 Scrub: "768MB" MUST NOT appear in validation.py or core_engine/validation.py
        platform_dir = Path(__file__).resolve().parent.parent / "alphaloop" / "platform"
        for rel in ("validation.py", "core_engine/validation.py"):
            txt = (platform_dir / rel).read_text(encoding="utf-8")
            self.assertNotIn("768MB", txt, f"Stale 768MB constraint text still present in {rel}")

        # 6. Fast-float accuracy vs Python statistics.stdev (including large-offset catastrophic-cancellation test!)
        large_offset = [1_000_000.0 + rng.gauss(0.0, 0.01) for _ in range(200)]
        _, _, fast_sd = fast_var_std(large_offset, ddof=1)
        exact_sd = statistics.stdev(large_offset)
        self.assertAlmostEqual(fast_sd, exact_sd, places=10)
        norm_rets = [0.001 + rng.gauss(0.0, 0.015) for _ in range(200)]
        self.assertAlmostEqual(
            sharpe_from_returns(norm_rets, 252.0),
            (statistics.mean(norm_rets) / statistics.stdev(norm_rets)) * math.sqrt(252.0),
            places=12,
        )
        n_m, mu_m, sd_m, _, _, _ = fast_moments(norm_rets)
        self.assertEqual(n_m, 200)
        self.assertAlmostEqual(mu_m, statistics.mean(norm_rets), places=12)
        self.assertAlmostEqual(sd_m, statistics.stdev(norm_rets), places=12)

    def test_14_fix1_holiday_and_t_plus_1_lock_caps_and_merged_sync(self) -> None:
        """FIX-1: Non-printing holidays AND T+1 locks NEVER breach max_positions/leverage/vol or drop sync_executed_weights."""
        engine = PortfolioAccountingEngine()
        zero_cost = CostModel(maker_fee_bps=0.0, taker_fee_bps=0.0, slippage_bps_per_leg=0.0)

        # Part A: Cross-calendar holiday (A absent on days 2-3 while held at 0.5)
        specs_hol = {
            "A": AssetSpec("A", ASSET_CLASS_GOLD_TRADFI, "yahoo", CALENDAR_US_DAILY, zero_cost, max_leverage_cap=1.0),
            "B": AssetSpec("B", ASSET_CLASS_CRYPTO_PERP, "hl", CALENDAR_CRYPTO_24_7, zero_cost, max_leverage_cap=1.0),
            "C": AssetSpec("C", ASSET_CLASS_CRYPTO_PERP, "hl", CALENDAR_CRYPTO_24_7, zero_cost, max_leverage_cap=1.0),
        }
        bars_hol = {
            "A": [_bar("A", BASE_TS_MS + d * DAY_MS, 100.0, 100.0, 100.0, 100.0, asset_class=ASSET_CLASS_GOLD_TRADFI) for d in (0, 1, 4)],
            "B": [_bar("B", BASE_TS_MS + d * DAY_MS, 100.0, 100.0, 100.0, 100.0) for d in range(5)],
            "C": [_bar("C", BASE_TS_MS + d * DAY_MS, 100.0, 100.0, 100.0, 100.0) for d in range(5)],
        }
        sched_hol = {
            d: {"A": (0.5, 10.0, None), "B": (0.5, 8.0, None), "C": (0.5, 6.0, None)}
            for d in range(5)
        }
        sync_log: list[dict[str, float]] = []

        class _SyncRecordingStrat(_ScriptedStrategy):
            def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
                super().sync_executed_weights(executed_weights)
                sync_log.append(dict(executed_weights))

            def clone_fresh(self) -> _SyncRecordingStrat:
                return _SyncRecordingStrat(self._schedule, strategy_id=self.strategy_id, asset_domain=self.asset_domain)

        risk_hol = RiskAllocatorConfig(
            max_positions=2,
            target_annual_vol_pct=500.0,
            max_portfolio_leverage=1.0,
            max_single_weight=0.6,
            use_correlation_idm=False,
        )
        strat_hol = _SyncRecordingStrat(sched_hol)
        res_hol = engine.simulate(strat_hol, specs_hol, bars_hol, risk_cfg=risk_hol, capital_usd=10_000.0)
        self.assertLessEqual(res_hol.max_active_positions, 2)
        self.assertLessEqual(res_hol.max_gross_leverage, 1.0 + 1e-9)
        for ew in res_hol.executed_weights_history:
            self.assertLessEqual(sum(1 for w in ew.values() if abs(w) > 1e-9), 2)
            self.assertLessEqual(sum(abs(w) for w in ew.values()), 1.0 + 1e-9)
        # On holiday days 2 & 3, synced weights MUST still retain held non-printing 'A' = 0.5!
        self.assertAlmostEqual(sync_log[2].get("A", 0.0), 0.5, places=6)
        self.assertAlmostEqual(sync_log[3].get("A", 0.0), 0.5, places=6)
        # G7 parity MUST pass on holiday universe with locked weights!
        par_hol = verify_full_parity(strat_hol, specs_hol, bars_hol, risk_cfg=risk_hol, batch_result=res_hol, capital_usd=10_000.0)
        self.assertTrue(par_hol.passed, par_hol.detail)

        # Part B: Exact quadratic vol scaling with non-printing locked holdings
        rng = random.Random(1401)
        base_r = [rng.gauss(0.0, 0.015) for _ in range(60)]
        rets_q = {
            "LOCK": list(base_r),
            "ADJ": [0.8 * x + rng.gauss(0.0, 0.006) for x in base_r],
        }
        specs_q = {
            "LOCK": AssetSpec("LOCK", ASSET_CLASS_CRYPTO_PERP, "hl", CALENDAR_CRYPTO_24_7, zero_cost, max_leverage_cap=2.0),
            "ADJ": AssetSpec("ADJ", ASSET_CLASS_CRYPTO_PERP, "hl", CALENDAR_CRYPTO_24_7, zero_cost, max_leverage_cap=2.0),
        }
        alloc_q = CausalRiskAllocator(
            RiskAllocatorConfig(max_positions=2, target_annual_vol_pct=20.0, max_portfolio_leverage=2.0, max_single_weight=1.0, use_correlation_idm=True, corr_shrinkage=0.25)
        )
        # B1: v_lock < target_var -> quadratic solves exact target_vol = 0.20
        dec_pos = {"ADJ": BarDecision("ADJ", BASE_TS_MS, 0.8, 5.0, "R")}
        w_q1 = alloc_q.allocate(BASE_TS_MS, dec_pos, rets_q, specs_q, locked_weights={"LOCK": 0.3}, timeframe_h_by_symbol={"LOCK": 24.0, "ADJ": 24.0})
        sig_after, _, _ = alloc_q.compute_portfolio_vol_and_idm(w_q1, dec_pos, rets_q, specs_q, locked_weights={"LOCK": 0.3}, timeframe_h_by_symbol={"LOCK": 24.0, "ADJ": 24.0})
        self.assertAlmostEqual(sig_after, 0.20, places=5)
        # B2: v_lock >= target_var with positive cov (same sign) -> ADJ scaled to 0.0
        w_q2 = alloc_q.allocate(BASE_TS_MS, dec_pos, rets_q, specs_q, locked_weights={"LOCK": 1.0}, timeframe_h_by_symbol={"LOCK": 24.0, "ADJ": 24.0})
        self.assertAlmostEqual(w_q2["ADJ"], 0.0, places=9)
        # B3: v_lock >= target_var with negative cov (hedging short ADJ) -> keeps variance-minimizing non-zero hedge!
        dec_hedge = {"ADJ": BarDecision("ADJ", BASE_TS_MS, -0.8, 5.0, "R")}
        w_q3 = alloc_q.allocate(BASE_TS_MS, dec_hedge, rets_q, specs_q, locked_weights={"LOCK": 1.0}, timeframe_h_by_symbol={"LOCK": 24.0, "ADJ": 24.0})
        self.assertLess(w_q3["ADJ"], -0.05)

        # Part C: T+1 Lock Post-Lock Guard (both max_positions=2 AND max_positions=3 / max_leverage=1.0)
        d1_base = BASE_TS_MS + DAY_MS + int(1.5 * HOUR_MS)  # 2025-01-02 09:30 CST
        specs_t1 = {s: default_ashare_template_spec(s) for s in ("A", "B", "C")}
        bars_t1 = {
            s: [
                _bar(s, d1_base + h_off * HOUR_MS, 100.0, 100.5, 99.5, 100.0, tf_h=1.0, asset_class=ASSET_CLASS_CN_A_SHARE)
                for h_off in (0, 1, 4, 5)
            ]
            for s in ("A", "B", "C")
        }
        sched_t1 = {
            0: {"A": (0.5, 10.0, None), "B": (0.5, 8.0, None), "C": (0.0, 0.0, None)},
            1: {"A": (0.0, 2.0, None), "B": (0.5, 8.0, None), "C": (0.5, 12.0, None)},
            2: {"A": (0.0, 2.0, None), "B": (0.5, 8.0, None), "C": (0.5, 12.0, None)},
            3: {"A": (0.0, 2.0, None), "B": (0.5, 8.0, None), "C": (0.5, 12.0, None)},
        }
        res_t1_pos = engine.simulate(
            _ScriptedStrategy(sched_t1, asset_domain="ashare"),
            specs_t1,
            bars_t1,
            risk_cfg=RiskAllocatorConfig(max_positions=2, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=0.5, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        self.assertLessEqual(res_t1_pos.max_active_positions, 2)
        self.assertLessEqual(res_t1_pos.max_gross_leverage, 1.0 + 1e-9)
        self.assertGreaterEqual(res_t1_pos.trade_ledger.suppression_counts().get("POST_LOCK_MAX_POSITIONS_CAP", 0), 1)

        res_t1_lev = engine.simulate(
            _ScriptedStrategy(sched_t1, asset_domain="ashare"),
            specs_t1,
            bars_t1,
            risk_cfg=RiskAllocatorConfig(max_positions=3, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=0.5, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        self.assertLessEqual(res_t1_lev.max_gross_leverage, 1.0 + 1e-9)
        self.assertGreaterEqual(res_t1_lev.trade_ledger.suppression_counts().get("POST_LOCK_LEVERAGE_CAP", 0), 1)

    def test_15_fix2_mixed_timeframe_alphabetical_renaming_invariance(self) -> None:
        """FIX-2: Mixed-timeframe vol scaling MUST be 100% invariant to alphabetical symbol renaming."""
        zero_cost = CostModel(0.0, 0.0, 0.0)
        rng = random.Random(1502)
        px_24 = 100.0
        px_12 = 100.0
        raw_24: list[tuple[int, float, float, float, float]] = []
        raw_12: list[tuple[int, float, float, float, float]] = []
        for d in range(40):
            r24 = rng.gauss(0.0005, 0.012)
            o24 = px_24
            c24 = o24 * math.exp(r24)
            raw_24.append((BASE_TS_MS + d * DAY_MS, o24, max(o24, c24), min(o24, c24), c24))
            px_24 = c24
            for h_i in (0, 12):
                r12 = rng.gauss(0.0003, 0.018)
                o12 = px_12
                c12 = o12 * math.exp(r12)
                raw_12.append((BASE_TS_MS + d * DAY_MS + h_i * HOUR_MS, o12, max(o12, c12), min(o12, c12), c12))
                px_12 = c12

        def _run_pair(name_24: str, name_12: str) -> tuple[SimulationResult, Any]:
            specs = {
                name_24: AssetSpec(name_24, ASSET_CLASS_GOLD_TRADFI, "yahoo", CALENDAR_US_DAILY, zero_cost, max_leverage_cap=1.5),
                name_12: AssetSpec(name_12, ASSET_CLASS_CRYPTO_PERP, "hl", CALENDAR_CRYPTO_24_7, zero_cost, max_leverage_cap=1.5),
            }
            bars = {
                name_24: [_bar(name_24, ts, o, h, l, c, tf_h=24.0, asset_class=ASSET_CLASS_GOLD_TRADFI) for ts, o, h, l, c in raw_24],
                name_12: [_bar(name_12, ts, o, h, l, c, tf_h=12.0, asset_class=ASSET_CLASS_CRYPTO_PERP) for ts, o, h, l, c in raw_12],
            }
            sched = {i: {name_24: (0.8, 6.0, None), name_12: (0.8, 5.0, None)} for i in range(100)}
            risk_cfg = RiskAllocatorConfig(max_positions=2, target_annual_vol_pct=15.0, max_portfolio_leverage=1.5, max_single_weight=1.0, use_correlation_idm=True, corr_lookback_bars=30)
            strat = _ScriptedStrategy(sched)
            eng = PortfolioAccountingEngine()
            res = eng.simulate(strat.clone_fresh(), specs, bars, risk_cfg=risk_cfg, capital_usd=10_000.0)
            par = verify_full_parity(strat, specs, bars, risk_cfg=risk_cfg, batch_result=res, capital_usd=10_000.0)
            return res, par

        res_az, par_az = _run_pair("AA_24H", "ZZ_12H")
        res_za, par_za = _run_pair("ZZ_24H", "AA_12H")
        self.assertTrue(par_az.passed, par_az.detail)
        self.assertTrue(par_za.passed, par_za.detail)
        self.assertAlmostEqual(res_az.final_equity_usd, res_za.final_equity_usd, places=9)

    def test_16_fix3_v3_intraday_last_bar_t_plus_1_and_spec_leverage_cap(self) -> None:
        """FIX-3: v3 AdaptiveMultiFactorKernel & simulate_universe block same-day T+0 sell after last-bar buy & respect spec.max_leverage_cap."""
        ash_spec = default_ashare_template_spec("600519.SH")  # t_plus_days=1, max_leverage_cap=1.0
        # Bar 0: 2025-01-02 14:30 CST (06:30 UTC) — last bar of Day 1 -> fills at Bar 1 open (Jan 3 09:30 CST)!
        # Bars 1..3: 2025-01-03 09:30, 10:30, 13:30 CST — SAME CST trading day as fill!
        # Bar 4: 2025-01-04 09:30 CST — Day 3 (T+1 unlocked!)
        d1_1430 = BASE_TS_MS + DAY_MS + int(6.5 * HOUR_MS)
        d2_0930 = BASE_TS_MS + 2 * DAY_MS + int(1.5 * HOUR_MS)
        d3_0930 = BASE_TS_MS + 3 * DAY_MS + int(1.5 * HOUR_MS)
        bars_ov = [
            _bar("600519.SH", d1_1430, 100.0, 101.0, 99.5, 100.5, tf_h=1.0, asset_class=ASSET_CLASS_CN_A_SHARE),
            _bar("600519.SH", d2_0930, 100.5, 100.5, 90.5, 91.0, tf_h=1.0, asset_class=ASSET_CLASS_CN_A_SHARE),
            _bar("600519.SH", d2_0930 + 1 * HOUR_MS, 91.0, 91.5, 90.0, 90.5, tf_h=1.0, asset_class=ASSET_CLASS_CN_A_SHARE),
            _bar("600519.SH", d2_0930 + 4 * HOUR_MS, 90.5, 91.0, 89.5, 90.0, tf_h=1.0, asset_class=ASSET_CLASS_CN_A_SHARE),
            _bar("600519.SH", d3_0930, 90.0, 91.0, 89.5, 90.0, tf_h=1.0, asset_class=ASSET_CLASS_CN_A_SHARE),
        ]
        snaps_ov = []
        regs_ov = []
        for bi, b in enumerate(bars_ov):
            f_val = 15.0 if bi == 0 else -15.0
            snaps_ov.append(
                FactorSnapshot(
                    symbol=b.symbol, venue=b.venue, timeframe_h=1.0,
                    ts_event_ms=b.ts_event_ms, ts_avail_ms=b.ts_avail_ms,
                    close=b.close, atr_14=1.0,
                    prior_high_40=b.high, prior_low_40=b.low,
                    prior_high_20=b.high, prior_low_20=b.low,
                    values={"ewmac_8_32": f_val, "ewmac_16_64": f_val, "ewmac_32_128": f_val, "realized_vol_ann_20": 15.0, "rel_volume_20": 1.0},
                )
            )
            regs_ov.append(RegimePoint(b.ts_event_ms, b.ts_avail_ms, REGIME_TREND_CALM, REGIME_TREND_CALM, 25.0, 0.4, 0.5, 1.0, 0.5, 0.0))

        cfg_ov = KernelConfig(
            strategy_name="ash_last_bar_t1",
            asset_class=ASSET_CLASS_CN_A_SHARE,
            timeframe_h=1.0,
            max_positions=1,
            max_symbol_leverage=1.5,  # Intentionally > spec.max_leverage_cap (1.0)!
            allow_short=False,
            min_hold_bars=1,
            warmup_bars=0,
            donchian_weight=0.0,
            atr_stop_mult=2.0,
            regime_exposure_map={REGIME_TREND_CALM: 1.0},
            risk_per_trade_pct=50.0,
        )
        metrics_ov, decs_ov_map = simulate_universe(
            cfg_ov, {"600519.SH": ash_spec}, {"600519.SH": bars_ov}, {"600519.SH": snaps_ov}, {"600519.SH": regs_ov}
        )
        decs_ov = decs_ov_map["600519.SH"]
        # Spec max_leverage_cap=1.0 MUST clamp entry target_weight <= 1.0!
        self.assertGreater(decs_ov[0].target_weight, 0.0)
        self.assertLessEqual(decs_ov[0].target_weight, 1.0 + 1e-9)
        # All Day 2 intraday bars (1, 2, 3) MUST stay T_PLUS_1_LOCK (NO same-day stop or reversal sell!)
        for bi in (1, 2, 3):
            self.assertEqual(decs_ov[bi].reason, "T_PLUS_1_LOCK")
            self.assertAlmostEqual(decs_ov[bi].target_weight, decs_ov[0].target_weight, places=6)
        # Day 3 bar 4 unlocks exit!
        self.assertAlmostEqual(decs_ov[4].target_weight, 0.0, places=6)
        # Only 1 fill occurred (buy at bar 1; exit signal at bar 4 has no bar 5) -> no illegal Day 2 sell!
        self.assertEqual(metrics_ov.trade_count, 1)

    def test_17_fix4_wipeout_conservation_and_short_etf_margin_liquidation(self) -> None:
        """FIX-4: -55% overnight gap conserves PnL in v4 & v3; 1x short ETF +150% rally triggers margin liquidation & conserves PnL."""
        engine = PortfolioAccountingEngine()
        zero_cost = CostModel(0.0, 0.0, 0.0)

        # Part A: -55% overnight gap on 2x long BTC perp -> exact conservation + zero ghost weights
        spec_btc = AssetSpec("BTC", ASSET_CLASS_CRYPTO_PERP, "hl", CALENDAR_CRYPTO_24_7, zero_cost, max_leverage_cap=3.0)
        gap_wipe_bars = [
            _bar("BTC", BASE_TS_MS + 0 * DAY_MS, 100.0, 100.0, 100.0, 100.0),
            _bar("BTC", BASE_TS_MS + 1 * DAY_MS, 100.0, 100.0, 100.0, 100.0),
            _bar("BTC", BASE_TS_MS + 2 * DAY_MS, 45.0, 45.0, 45.0, 45.0),
            _bar("BTC", BASE_TS_MS + 3 * DAY_MS, 50.0, 50.0, 50.0, 50.0),
        ]
        res_v4_gap = engine.simulate(
            _ScriptedStrategy({0: {"BTC": (2.0, 10.0, None)}, 1: {"BTC": (2.0, 10.0, None)}, 2: {"BTC": (0.0, 0.0, None)}, 3: {"BTC": (0.0, 0.0, None)}}),
            {"BTC": spec_btc},
            {"BTC": gap_wipe_bars},
            risk_cfg=RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=2.5, max_single_weight=2.0, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        self.assertAlmostEqual(res_v4_gap.final_equity_usd, 0.0, places=6)
        self.assertGreaterEqual(res_v4_gap.liquidation_count, 1)
        rhs_gap = res_v4_gap.gross_price_pnl_usd + res_v4_gap.funding_carry_pnl_usd - res_v4_gap.total_cost_usd
        self.assertAlmostEqual(res_v4_gap.net_pnl_usd, rhs_gap, places=6)
        self.assertAlmostEqual(sum(res_v4_gap.by_symbol_net_usd.values()), res_v4_gap.net_pnl_usd, places=6)
        # Zero ghost weights on wipeout bar & subsequent halted bars
        self.assertAlmostEqual(res_v4_gap.executed_weights_history[2]["BTC"], 0.0, places=9)
        self.assertAlmostEqual(res_v4_gap.executed_weights_history[3]["BTC"], 0.0, places=9)

        # Same -55% overnight gap in v3 simulate_universe
        snaps_gap = [
            FactorSnapshot(
                symbol="BTC", venue="hl", timeframe_h=24.0, ts_event_ms=b.ts_event_ms, ts_avail_ms=b.ts_avail_ms,
                close=b.close, atr_14=2.0, prior_high_40=b.high, prior_low_40=b.low, prior_high_20=b.high, prior_low_20=b.low,
                values={"ewmac_8_32": 20.0, "ewmac_16_64": 20.0, "ewmac_32_128": 20.0, "realized_vol_ann_20": 10.0, "rel_volume_20": 1.0},
            )
            for b in gap_wipe_bars
        ]
        regs_gap = [RegimePoint(b.ts_event_ms, b.ts_avail_ms, REGIME_TREND_CALM, REGIME_TREND_CALM, 25.0, 0.4, 0.5, 1.0, 0.5, 0.0) for b in gap_wipe_bars]
        cfg_v3_gap = KernelConfig(
            strategy_name="v3_gap_wipe", asset_class=ASSET_CLASS_CRYPTO_PERP, timeframe_h=24.0, capital_usd=10_000.0,
            max_positions=1, target_annual_vol_pct=200.0, max_symbol_leverage=2.0, warmup_bars=0, donchian_weight=0.0,
            regime_exposure_map={REGIME_TREND_CALM: 1.0}, risk_per_trade_pct=50.0,
        )
        m_v3_gap, _ = simulate_universe(cfg_v3_gap, {"BTC": spec_btc}, {"BTC": gap_wipe_bars}, {"BTC": snaps_gap}, {"BTC": regs_gap})
        self.assertAlmostEqual(m_v3_gap.net_pnl_usd, -10_000.0, places=2)
        self.assertAlmostEqual(m_v3_gap.net_pnl_usd, m_v3_gap.gross_price_pnl_usd + m_v3_gap.funding_carry_pnl_usd - m_v3_gap.total_cost_usd, places=2)
        self.assertGreaterEqual(m_v3_gap.liquidation_count, 1)

        # Part B: 1x short ETF rallying +150% intraday MUST trigger margin liquidation AND conserve PnL!
        spec_etf_short = AssetSpec("GLD", ASSET_CLASS_GOLD_TRADFI, "yahoo", CALENDAR_US_DAILY, zero_cost, max_leverage_cap=1.0)
        short_squeeze_bars = [
            _bar("GLD", BASE_TS_MS + 0 * DAY_MS, 100.0, 100.0, 100.0, 100.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
            _bar("GLD", BASE_TS_MS + 1 * DAY_MS, 100.0, 250.0, 100.0, 250.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
            _bar("GLD", BASE_TS_MS + 2 * DAY_MS, 250.0, 250.0, 250.0, 250.0, asset_class=ASSET_CLASS_GOLD_TRADFI),
        ]
        res_short_etf = engine.simulate(
            _ScriptedStrategy({0: {"GLD": (-1.0, -10.0, None)}, 1: {"GLD": (-1.0, -10.0, None)}, 2: {"GLD": (0.0, 0.0, None)}}, asset_domain="gold"),
            {"GLD": spec_etf_short},
            {"GLD": short_squeeze_bars},
            risk_cfg=RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        self.assertGreaterEqual(res_short_etf.liquidation_count, 1)
        self.assertAlmostEqual(res_short_etf.final_equity_usd, 0.0, places=6)
        rhs_short = res_short_etf.gross_price_pnl_usd + res_short_etf.funding_carry_pnl_usd - res_short_etf.total_cost_usd
        self.assertAlmostEqual(res_short_etf.net_pnl_usd, rhs_short, places=6)
        self.assertAlmostEqual(sum(res_short_etf.by_symbol_net_usd.values()), res_short_etf.net_pnl_usd, places=6)

        # Same 1x short ETF +150% intraday rally in v3 simulate_universe -> liquidation + conservation
        snaps_se = [
            FactorSnapshot(
                symbol="GLD", venue="yahoo", timeframe_h=24.0, ts_event_ms=b.ts_event_ms, ts_avail_ms=b.ts_avail_ms,
                close=b.close, atr_14=2.0, prior_high_40=b.high, prior_low_40=b.low, prior_high_20=b.high, prior_low_20=b.low,
                values={"ewmac_8_32": -20.0, "ewmac_16_64": -20.0, "ewmac_32_128": -20.0, "realized_vol_ann_20": 10.0, "rel_volume_20": 1.0},
            )
            for b in short_squeeze_bars
        ]
        regs_se = [RegimePoint(b.ts_event_ms, b.ts_avail_ms, REGIME_TREND_CALM, REGIME_TREND_CALM, 25.0, 0.4, 0.5, 1.0, 0.5, 0.0) for b in short_squeeze_bars]
        cfg_v3_se = KernelConfig(
            strategy_name="v3_short_etf_liq", asset_class=ASSET_CLASS_GOLD_TRADFI, timeframe_h=24.0, capital_usd=10_000.0,
            max_positions=1, target_annual_vol_pct=200.0, max_symbol_leverage=1.0, allow_short=True, warmup_bars=0, donchian_weight=0.0,
            regime_exposure_map={REGIME_TREND_CALM: 1.0}, risk_per_trade_pct=50.0,
        )
        m_v3_se, _ = simulate_universe(cfg_v3_se, {"GLD": spec_etf_short}, {"GLD": short_squeeze_bars}, {"GLD": snaps_se}, {"GLD": regs_se})
        self.assertGreaterEqual(m_v3_se.liquidation_count, 1)
        self.assertAlmostEqual(m_v3_se.net_pnl_usd, -10_000.0, places=2)
        self.assertAlmostEqual(m_v3_se.net_pnl_usd, m_v3_se.gross_price_pnl_usd + m_v3_se.funding_carry_pnl_usd - m_v3_se.total_cost_usd, places=2)

    def test_18_fix5_fix6_fix7_stop_limit_lock_zombie_pending_and_liq_wick_notional(self) -> None:
        """FIX-5/6/7: INTRABAR_STOP respects PRICE_LIMIT_DOWN_LOCK; zero zombie re-entry; liquidation wick notional & contracts exact."""
        engine = PortfolioAccountingEngine(maint_margin_rate=0.05, liquidation_fee_bps=0.0)
        zero_cost = CostModel(0.0, 0.0, 0.0)

        # FIX-5: INTRABAR_STOP blocked by CN A-share -10% 1-tick PRICE_LIMIT_DOWN_LOCK
        ash_spec = default_ashare_template_spec("600519.SH")
        lim_bars = [
            _bar("600519.SH", BASE_TS_MS + 0 * DAY_MS, 100.0, 100.0, 100.0, 100.0, asset_class=ASSET_CLASS_CN_A_SHARE),
            _bar("600519.SH", BASE_TS_MS + 1 * DAY_MS, 100.0, 100.0, 100.0, 100.0, asset_class=ASSET_CLASS_CN_A_SHARE),
            _bar("600519.SH", BASE_TS_MS + 2 * DAY_MS, 90.0, 90.0, 90.0, 90.0, asset_class=ASSET_CLASS_CN_A_SHARE),
        ]
        res_lim = engine.simulate(
            _ScriptedStrategy({0: {"600519.SH": (0.8, 10.0, 95.0)}, 1: {"600519.SH": (0.0, -10.0, 95.0)}, 2: {"600519.SH": (0.0, -10.0, 95.0)}}, asset_domain="ashare"),
            {"600519.SH": ash_spec},
            {"600519.SH": lim_bars},
            risk_cfg=RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        self.assertFalse(any(t.reason == "INTRABAR_STOP" for t in res_lim.trade_ledger.executed))
        self.assertGreaterEqual(res_lim.trade_ledger.suppression_counts().get("PRICE_LIMIT_DOWN_STOP_LOCK", 0), 1)
        self.assertAlmostEqual(res_lim.executed_weights_history[2]["600519.SH"], 0.8, places=6)

        # FIX-6: Zero zombie _pending_weights re-entry when strategy omits symbol after INTRABAR_STOP or INTRABAR_LIQUIDATION
        class _OneShotSparseStrategy:
            def __init__(self, w0: float, stop0: float | None) -> None:
                self.strategy_id = "oneshot_sparse"
                self.asset_domain = "web3"
                self.warmup_bars = 0
                self._w0 = w0
                self._stop0 = stop0
                self._step = 0

            def reset(self) -> None:
                self._step = 0

            def on_bar_step(self, ts_ms: int, bars_at_ts: dict[str, CanonicalBar], context: dict[str, Any] | None = None) -> dict[str, BarDecision]:
                _ = (bars_at_ts, context)
                idx = self._step
                self._step += 1
                if idx == 0:
                    return {"BTC": BarDecision("BTC", ts_ms, self._w0, 10.0, "R", stop_price=self._stop0)}
                return {}

            def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
                _ = executed_weights

            def clone_fresh(self) -> _OneShotSparseStrategy:
                return _OneShotSparseStrategy(self._w0, self._stop0)

        spec_btc = AssetSpec("BTC", ASSET_CLASS_CRYPTO_PERP, "hl", CALENDAR_CRYPTO_24_7, zero_cost, max_leverage_cap=3.0)
        stop_z_bars = [
            _bar("BTC", BASE_TS_MS + 0 * DAY_MS, 100.0, 100.0, 100.0, 100.0),
            _bar("BTC", BASE_TS_MS + 1 * DAY_MS, 100.0, 100.0, 94.0, 98.0),
            _bar("BTC", BASE_TS_MS + 2 * DAY_MS, 98.0, 99.0, 97.0, 98.0),
        ]
        res_z_stop = engine.simulate(
            _OneShotSparseStrategy(1.0, 95.0),
            {"BTC": spec_btc},
            {"BTC": stop_z_bars},
            risk_cfg=RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=2.5, max_single_weight=2.0, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        self.assertTrue(any(t.reason == "INTRABAR_STOP" for t in res_z_stop.trade_ledger.executed))
        self.assertAlmostEqual(res_z_stop.executed_weights_history[2]["BTC"], 0.0, places=9)

        liq_z_bars = [
            _bar("BTC", BASE_TS_MS + 0 * DAY_MS, 100.0, 100.0, 100.0, 100.0),
            _bar("BTC", BASE_TS_MS + 1 * DAY_MS, 100.0, 100.0, 52.0, 90.0),
            _bar("BTC", BASE_TS_MS + 2 * DAY_MS, 90.0, 92.0, 89.0, 91.0),
        ]
        res_z_liq = engine.simulate(
            _OneShotSparseStrategy(2.0, None),
            {"BTC": spec_btc},
            {"BTC": liq_z_bars},
            risk_cfg=RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=2.5, max_single_weight=2.0, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        self.assertEqual(res_z_liq.liquidation_count, 1)
        self.assertAlmostEqual(res_z_liq.executed_weights_history[2]["BTC"], 0.0, places=9)

        # FIX-7: Exact contract conservation (200.0 == 200.0) & wick notional on long/short liquidation + 2x short high=144 MM breach!
        liq_trades_long = [t for t in res_z_liq.trade_ledger.executed if t.reason == "INTRABAR_LIQUIDATION"]
        self.assertEqual(len(liq_trades_long), 1)
        self.assertAlmostEqual(liq_trades_long[0].contracts, 200.0, places=6)
        self.assertAlmostEqual(liq_trades_long[0].notional_usd, 10_400.0, places=4)

        short_148_bars = [
            _bar("BTC", BASE_TS_MS + 0 * DAY_MS, 100.0, 100.0, 100.0, 100.0),
            _bar("BTC", BASE_TS_MS + 1 * DAY_MS, 100.0, 148.0, 100.0, 95.0),
            _bar("BTC", BASE_TS_MS + 2 * DAY_MS, 95.0, 95.0, 95.0, 95.0),
        ]
        res_short_148 = engine.simulate(
            _OneShotSparseStrategy(-2.0, None),
            {"BTC": spec_btc},
            {"BTC": short_148_bars},
            risk_cfg=RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=2.5, max_single_weight=2.0, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        liq_trades_short = [t for t in res_short_148.trade_ledger.executed if t.reason == "INTRABAR_LIQUIDATION"]
        self.assertEqual(len(liq_trades_short), 1)
        self.assertAlmostEqual(liq_trades_short[0].contracts, 200.0, places=6)
        self.assertAlmostEqual(liq_trades_short[0].notional_usd, 29_600.0, places=4)

        # 2x short BTC at high=144: eq=$1,200, wick notional=$28,800, 5% MM=$1,440 > $1,200 -> MUST liquidate!
        short_144_bars = [
            _bar("BTC", BASE_TS_MS + 0 * DAY_MS, 100.0, 100.0, 100.0, 100.0),
            _bar("BTC", BASE_TS_MS + 1 * DAY_MS, 100.0, 144.0, 95.0, 95.0),
            _bar("BTC", BASE_TS_MS + 2 * DAY_MS, 95.0, 95.0, 95.0, 95.0),
        ]
        res_short_144 = engine.simulate(
            _OneShotSparseStrategy(-2.0, None),
            {"BTC": spec_btc},
            {"BTC": short_144_bars},
            risk_cfg=RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=2.5, max_single_weight=2.0, use_correlation_idm=False),
            capital_usd=10_000.0,
        )
        self.assertEqual(res_short_144.liquidation_count, 1)
        self.assertTrue(any(t.reason == "INTRABAR_LIQUIDATION" for t in res_short_144.trade_ledger.executed))

    def test_19_fix8_cscv_chan_pebay_flat_half_and_constant_blocks(self) -> None:
        """FIX-8: Chan/Pebay block-variance merge eliminates IEEE-754 cancellation on flat/constant halves in BOTH NumPy & stdlib paths."""
        import alphaloop.platform.core_engine.validation as val_mod

        # Reproduce Challenger 2's exact seed=4 (cl=23) and seed=10 (cl=19) flat-half matrices
        rng = random.Random(20261123)
        mat_seed4: list[list[float]] = []
        mat_seed10: list[list[float]] = []
        for seed in range(11):
            cl = rng.choice([7, 11, 13, 17, 19, 23])
            nh = cl * 4
            mat = [
                [0.0] * nh + [rng.gauss(-0.002, 0.01) for _ in range(nh)],
                [0.0] * nh + [rng.gauss(-0.001, 0.01) for _ in range(nh)],
                [0.0] * nh + [rng.gauss(0.0005, 0.01) for _ in range(nh)],
                [0.0] * nh + [rng.gauss(-0.003, 0.01) for _ in range(nh)],
            ]
            if seed == 4:
                mat_seed4 = mat
            elif seed == 10:
                mat_seed10 = mat

        # Piecewise-constant blocks [0.002]*76 + [0.007]*76 MUST have 0.0 Sharpe on constant halves (NEVER +1.18e9!)
        pw_const = [[0.002] * 76 + [0.007] * 76, [0.001] * 76 + [0.003] * 76]

        orig_np = val_mod._np
        try:
            for force_stdlib in (False, True):
                if force_stdlib:
                    val_mod._np = None
                rep10 = compute_pbo_cscv(mat_seed10, 252.0, n_subsamples=8)
                self.assertAlmostEqual(rep10.pbo, 0.3571, places=4)
                self.assertGreater(rep10.pbo, 0.35)  # Must FAIL G6 <= 0.35!

                rep4 = compute_pbo_cscv(mat_seed4, 252.0, n_subsamples=8)
                self.assertAlmostEqual(rep4.pbo, 0.2000, places=4)
                self.assertAlmostEqual(rep4.median_oos_winner_sharpe, 0.3005, places=3)

                rep_pw = compute_pbo_cscv(pw_const, 252.0, n_subsamples=8)
                self.assertLess(abs(rep_pw.median_is_winner_sharpe), 100.0)
                self.assertLess(abs(rep_pw.median_oos_winner_sharpe), 100.0)
        finally:
            val_mod._np = orig_np

    def test_20_fix9_macro_timeline_dedup_and_empty_key_fallback(self) -> None:
        """FIX-9: MacroRegimeTimeline deduplicates identical ts_avail_ms bars and falls back cleanly when GC=F/SI=F list is empty."""
        n = 300
        macro_clean = {
            "GC=F": [_bar("GC=F", BASE_TS_MS + i * DAY_MS, 2000 + i, 2005 + i, 1995 + i, 2000 + i * 1.1, asset_class=ASSET_CLASS_GOLD_TRADFI) for i in range(n)],
            "GLD": [_bar("GLD", BASE_TS_MS + i * DAY_MS, 200 + i * 0.1, 201 + i * 0.1, 199 + i * 0.1, 200 + i * 0.11, asset_class=ASSET_CLASS_GOLD_TRADFI) for i in range(n)],
            "SI=F": [_bar("SI=F", BASE_TS_MS + i * DAY_MS, 25 + i * 0.01, 25.5 + i * 0.01, 24.5 + i * 0.01, 25 + i * 0.012, asset_class=ASSET_CLASS_GOLD_TRADFI) for i in range(n)],
            "SLV": [_bar("SLV", BASE_TS_MS + i * DAY_MS, 22 + i * 0.01, 22.5 + i * 0.01, 21.5 + i * 0.01, 22 + i * 0.011, asset_class=ASSET_CLASS_GOLD_TRADFI) for i in range(n)],
            "IEF": [_bar("IEF", BASE_TS_MS + i * DAY_MS, 95 + i * 0.02, 95.5 + i * 0.02, 94.5 + i * 0.02, 95 + i * 0.02, asset_class=ASSET_CLASS_GOLD_TRADFI) for i in range(n)],
            "DX-Y.NYB": [_bar("DX-Y.NYB", BASE_TS_MS + i * DAY_MS, 105 - i * 0.01, 105.5 - i * 0.01, 104.5 - i * 0.01, 105 - i * 0.01, asset_class=ASSET_CLASS_GOLD_TRADFI) for i in range(n)],
            "^VIX": [_bar("^VIX", BASE_TS_MS + i * DAY_MS, 15 + (i % 9) * 0.7, 16 + (i % 9) * 0.7, 14 + (i % 9) * 0.7, 15 + (i % 9) * 0.7, asset_class=ASSET_CLASS_GOLD_TRADFI) for i in range(n)],
            "^TNX": [_bar("^TNX", BASE_TS_MS + i * DAY_MS, 4.2 - i * 0.001, 4.3 - i * 0.001, 4.1 - i * 0.001, 4.2 - i * 0.001, asset_class=ASSET_CLASS_GOLD_TRADFI) for i in range(n)],
        }
        q_ts = BASE_TS_MS + 280 * DAY_MS
        tl_clean = MacroRegimeTimeline(macro_clean)
        tl_dup2x = MacroRegimeTimeline({k: v + v for k, v in macro_clean.items()})
        self.assertEqual(tl_clean.snapshot_as_of(q_ts), tl_dup2x.snapshot_as_of(q_ts))

        # Empty "GC=F": [] and "SI=F": [] MUST fall back cleanly to "GLD" and "SLV"!
        macro_gld_slv = {k: v for k, v in macro_clean.items() if k not in ("GC=F", "SI=F")}
        macro_empty_keys = {**macro_gld_slv, "GC=F": [], "SI=F": []}
        self.assertEqual(
            MacroRegimeTimeline(macro_empty_keys).snapshot_as_of(q_ts),
            MacroRegimeTimeline(macro_gld_slv).snapshot_as_of(q_ts),
        )

    def test_21_fix10_fix11_d12_zero_se_sparse_cluster_cooldown_and_g1_fail_closed(self) -> None:
        """FIX-10 & FIX-11: Oracle IC=1.0 PASS, constant overlapping sparse PASS, single cluster REJECT, cooldown disjoint, G1 fail-closed."""
        # FIX-10a: newey_west_mean_stats on constant positive series
        nw_const = newey_west_mean_stats([0.05] * 50, max_lag=2)
        self.assertAlmostEqual(nw_const.t_stat, 50.0, places=6)
        self.assertLess(nw_const.p_value_two_sided, 1e-6)

        # FIX-10b: compute_hac_factor_ic & FactorDirectionAuditor on exact oracle IC=1.0 factor
        hac_oracle = compute_hac_factor_ic([float(i) for i in range(100)], [2.0 * i + 1.0 for i in range(100)], horizon_bars=4)
        self.assertAlmostEqual(hac_oracle.overlapping_ic, 1.0, places=6)
        self.assertGreaterEqual(hac_oracle.hac_t_stat, 50.0)
        self.assertLess(hac_oracle.p_value_two_sided, 1e-6)

        oracle_bars = _trend_bars("BTC", 120, tf_h=12.0, seed=2101)
        oracle_snaps = []
        for i, b in enumerate(oracle_bars):
            fwd_val = (oracle_bars[i + 5].close / oracle_bars[i + 1].open - 1.0) * 10_000.0 if i + 5 < len(oracle_bars) else 0.0
            oracle_snaps.append(
                FactorSnapshot(
                    symbol="BTC", venue="test", timeframe_h=12.0, ts_event_ms=b.ts_event_ms, ts_avail_ms=b.ts_avail_ms,
                    close=b.close, atr_14=1.0, prior_high_40=b.high, prior_low_40=b.low, prior_high_20=b.high, prior_low_20=b.low,
                    values={"oracle_alpha": fwd_val},
                )
            )
        aud_oracle = FactorDirectionAuditor(warmup_bars=20)
        summ_oracle = aud_oracle.audit("oracle_test", {"BTC": oracle_bars}, {"BTC": oracle_snaps}, horizon_bars=4, candidate_factors=["oracle_alpha"], priors={"oracle_alpha": +1})
        self.assertTrue(summ_oracle.results[0].passed_g2, summ_oracle.results[0].detail)
        self.assertGreaterEqual(summ_oracle.results[0].hac_t_stat, 50.0)

        # FIX-10c: Constant overlapping events with >= 10 effective events PASS!
        ev_const_ov = compute_sparse_event_expectancy(
            [1.0 if (i % 2 == 0 and i < 60) else 0.0 for i in range(100)],
            [0.02] * 100,
            horizon_bars=4,
            min_events=10,
        )
        self.assertEqual(ev_const_ov.effective_events, 15)
        self.assertAlmostEqual(ev_const_ov.t_stat, 10.0, places=4)
        self.assertTrue(ev_const_ov.passed_event_gate)

        # FIX-10d: Single 12-bar consecutive cluster [50..61] with horizon_bars=8 MUST be rejected by min_events=10!
        cluster_sigs = [1.0 if 50 <= i < 62 else 0.0 for i in range(120)]
        cluster_rets = [0.015 + (i % 3) * 0.0005 for i in range(120)]
        ev_single_cluster = compute_sparse_event_expectancy(cluster_sigs, cluster_rets, horizon_bars=8, min_events=10)
        self.assertEqual(ev_single_cluster.n_events, 12)
        self.assertEqual(ev_single_cluster.effective_events, 2)
        self.assertFalse(ev_single_cluster.passed_event_gate)

        # FIX-10d2: Clustered overlapping events use 2x Bartlett bandwidth (nw_lag=14 for 8-bar repeats, H=8)
        rng_cl = random.Random(2103)
        base_12_r = [0.008 + rng_cl.gauss(0.0, 0.014) for _ in range(12)]
        cl_sigs = [0.0] * 300
        cl_rets = [0.0] * 300
        for ci, rv in enumerate(base_12_r):
            p0 = 10 + ci * 20
            for off in range(8):
                cl_sigs[p0 + off] = 1.0
                cl_rets[p0 + off] = rv
        ev_cl2x = compute_sparse_event_expectancy(cl_sigs, cl_rets, horizon_bars=8, min_events=10)
        self.assertEqual(ev_cl2x.n_events, 96)
        self.assertEqual(ev_cl2x.effective_events, 12)
        self.assertIn("nw_lag=14", ev_cl2x.detail)

        # FIX-10e: 1-bar cooldown fencepost in FactorDirectionAuditor (bar_i + h_bars + 1)
        # Triggers at bars 25, 29 (spaced by h_bars=4 -> overlapping on bar 30 -> deduplicated to 1) and 30 (25+4+1 -> kept!)
        cd_bars = _trend_bars("BTC", 80, tf_h=12.0, seed=2102)
        trig_set = {25, 29, 30}
        cd_snaps = [
            FactorSnapshot(
                symbol="BTC", venue="test", timeframe_h=12.0, ts_event_ms=b.ts_event_ms, ts_avail_ms=b.ts_avail_ms,
                close=b.close, atr_14=1.0, prior_high_40=b.high, prior_low_40=b.low, prior_high_20=b.high, prior_low_20=b.low,
                values={"cd_sig": 1.0 if i in trig_set else 0.0},
            )
            for i, b in enumerate(cd_bars)
        ]
        summ_cd = FactorDirectionAuditor(warmup_bars=20, min_event_count=2).audit(
            "cd_test", {"BTC": cd_bars}, {"BTC": cd_snaps}, horizon_bars=4, candidate_factors=["cd_sig"], priors={"cd_sig": +1}
        )
        # Trigger 29 is blocked (< 25 + 4 + 1 = 30), while trigger 30 IS allowed -> exact 2 events!
        self.assertEqual(summ_cd.results[0].event_count, 2)

        # FIX-11: G1 fail-closed on missing .provenance attribute AND synthetic provenance
        fake_dq_missing_prov = type("FakeDQNoProv", (), {"passed": True})()
        rep_no_prov = evaluate_8_gates("g1_missing_prov", "web3", dq_reports=[fake_dq_missing_prov], allow_synthetic_in_test=False)
        self.assertFalse(rep_no_prov.gate_map["G1"])
        fake_dq_synth = type("FakeDQSynth", (), {"passed": True, "provenance": "SYNTHETIC"})()
        rep_synth = evaluate_8_gates("g1_synth_prov", "web3", dq_reports=[fake_dq_synth], allow_synthetic_in_test=False)
        self.assertFalse(rep_synth.gate_map["G1"])

    def test_22_fix12_verify_full_parity_fee_halted_and_prefix_regime_flip(self) -> None:
        """FIX-12: verify_full_parity PASSES on honest fee-halted unleveraged account AND catches prefix-only regime_label flip."""
        # Part A: Honest alternating strategy draining unleveraged account to $0 via 8000bp/leg fee
        huge_fee = CostModel(maker_fee_bps=8000.0, taker_fee_bps=8000.0, slippage_bps_per_leg=0.0)
        spec_fee = AssetSpec("GLD", ASSET_CLASS_GOLD_TRADFI, "yahoo", CALENDAR_US_DAILY, huge_fee, max_leverage_cap=1.0)
        fee_bars = [_bar("GLD", BASE_TS_MS + i * DAY_MS, 100.0, 100.0, 100.0, 100.0, asset_class=ASSET_CLASS_GOLD_TRADFI) for i in range(25)]
        fee_sched = {i: {"GLD": (1.0 if i % 2 == 0 else 0.0, 5.0 if i % 2 == 0 else 0.0, None)} for i in range(25)}
        risk_fee = RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False)
        strat_fee = _ScriptedStrategy(fee_sched, asset_domain="gold")
        eng = PortfolioAccountingEngine()
        res_fee = eng.simulate(strat_fee.clone_fresh(), {"GLD": spec_fee}, {"GLD": fee_bars}, risk_cfg=risk_fee, capital_usd=10_000.0)
        self.assertLessEqual(res_fee.final_equity_usd, 1e-6)
        # Exact conservation even under 8000bp fee wipeout
        rhs_fee = res_fee.gross_price_pnl_usd + res_fee.funding_carry_pnl_usd - res_fee.total_cost_usd
        self.assertAlmostEqual(res_fee.net_pnl_usd, rhs_fee, places=6)
        par_fee = verify_full_parity(strat_fee, {"GLD": spec_fee}, {"GLD": fee_bars}, risk_cfg=risk_fee, batch_result=res_fee, capital_usd=10_000.0)
        self.assertTrue(par_fee.passed, par_fee.detail)

        # Part B: Prefix-only regime_label flip (identical weights & equity!) MUST be caught by Check 3!
        class _PrefixRegimeFlipStrategy:
            _clone_counter = 0

            def __init__(self, clone_id: int = 0) -> None:
                self.strategy_id = "prefix_regime_flip"
                self.asset_domain = "web3"
                self.warmup_bars = 0
                self._clone_id = clone_id

            def reset(self) -> None:
                pass

            def on_bar_step(self, ts_ms: int, bars_at_ts: dict[str, CanonicalBar], context: dict[str, Any] | None = None) -> dict[str, BarDecision]:
                _ = context
                # Clone #3 is created for Check 3 (60% prefix truncation probe) when batch_result is pre-supplied!
                lbl = "PREFIX_FLIPPED" if self._clone_id >= 3 else "CLEAN_REGIME"
                return {s: BarDecision(s, ts_ms, 0.5, 5.0, lbl) for s in sorted(bars_at_ts)}

            def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
                _ = executed_weights

            def clone_fresh(self) -> _PrefixRegimeFlipStrategy:
                _PrefixRegimeFlipStrategy._clone_counter += 1
                return _PrefixRegimeFlipStrategy(_PrefixRegimeFlipStrategy._clone_counter)

        _PrefixRegimeFlipStrategy._clone_counter = 0
        spec_btc = AssetSpec("BTC", ASSET_CLASS_CRYPTO_PERP, "hl", CALENDAR_CRYPTO_24_7, CostModel(0.0, 0.0, 0.0), max_leverage_cap=1.0)
        bars_rf = [_bar("BTC", BASE_TS_MS + i * DAY_MS, 100.0, 101.0, 99.0, 100.0) for i in range(15)]
        risk_rf = RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False)
        strat_rf = _PrefixRegimeFlipStrategy(0)
        res_rf = eng.simulate(strat_rf, {"BTC": spec_btc}, {"BTC": bars_rf}, risk_cfg=risk_rf, capital_usd=10_000.0)
        par_rf = verify_full_parity(strat_rf, {"BTC": spec_btc}, {"BTC": bars_rf}, risk_cfg=risk_rf, batch_result=res_rf, capital_usd=10_000.0)
        self.assertFalse(par_rf.passed)
        self.assertGreater(par_rf.regime_mismatch_count, 0)


if __name__ == "__main__":
    unittest.main()
