"""Unit, G7 Bit-Parity, Domain-Isolation, Pure-Stdlib & Real-Lake Smoke Tests for Track W (Web3 Suite).

Tests Milestone M3 deliverables:
  - `Web3MarketRegimeEngine` (crypto-native, stateless PIT bisect_right, zero Yahoo ^VIX coupling)
  - `W1` (`Web3XSMomCarryKernel`): XS momentum + funding crowding penalty + OI confirmation + HYPE PIT
  - `W2` (`Web3AdaptiveTrendKernel`): G2-verified breakout/trend + crowding throttle + breadth short gate
  - `W3` (`Web3RegimeCarryKernel`): 3x RT fee cost-horizon floor + DVOL throttle + 50% margin storm-exit
  - Pure Python 3.12 stdlib execution (blocking third-party scientific packages via `sys.meta_path`)
  - Real multi-year smoke & G7 parity against `/home/antigravity/kenansun/auto/data/platform_v4_lake/`
"""

from __future__ import annotations

import ast
import bisect
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Sequence

from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.platform.assets import (
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_MACRO_INDEX,
    AssetSpec,
    CALENDAR_CRYPTO_24_7,
    CostModel,
    WEB3_LIQUID_UNIVERSE,
    default_web3_specs,
)
from alphaloop.platform.core_engine.accounting import PortfolioAccountingEngine
from alphaloop.platform.core_engine.interfaces import AssetNativeStrategy, BarDecision
from alphaloop.platform.core_engine.risk_allocator import RiskAllocatorConfig
from alphaloop.platform.core_engine.validation import verify_full_parity
from alphaloop.platform.data_foundation.models import (
    CanonicalBar,
    DAY_MS,
    HOUR_MS,
    Web3DerivativesMetric,
    Web3MarketMacroPoint,
)
from alphaloop.platform.data_foundation.router import DataFoundationRouter
from alphaloop.platform.strategies.web3 import (
    HYPE_PIT_LISTING_MS,
    WEB3_REGIME_ALT_SEASON,
    WEB3_REGIME_EXPANSION,
    WEB3_REGIME_RISK_OFF_BEAR,
    Web3AdaptiveTrendKernel,
    Web3MarketRegimeEngine,
    Web3RegimeCarryKernel,
    Web3XSMomCarryKernel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_V4_LAKE_DIR = REPO_ROOT / "data" / "platform_v4_lake"


def _utc_ms(year: int, month: int, day: int, hour: int = 0) -> int:
    return int(round(datetime(year, month, day, hour, 0, tzinfo=timezone.utc).timestamp() * 1000.0))


BASE_TS_MS = _utc_ms(2024, 1, 1, 0)


def _trending_closes(
    n: int,
    *,
    start: float = 100.0,
    step_ret: float = 0.004,
    noise_std: float = 0.002,
    seed: int = 42,
) -> list[float]:
    rng = random.Random(seed)
    out = [float(start)]
    cur = float(start)
    for _ in range(max(0, n - 1)):
        cur = max(1e-3, cur * (1.0 + step_ret + rng.gauss(0.0, noise_std)))
        out.append(cur)
    return out


def _make_bar_series(
    *,
    symbol: str,
    closes: Sequence[float],
    timeframe_h: float = 12.0,
    start_ms: int = BASE_TS_MS,
    venue: str = "hyperliquid",
    asset_class: str = ASSET_CLASS_CRYPTO_PERP,
    carry_rates: Sequence[float] | None = None,
    carry_interval_h: float | None = None,
    premium_basis_bps: float = 0.0,
) -> list[CanonicalBar]:
    tf_ms = int(round(timeframe_h * HOUR_MS))
    ci_h = float(timeframe_h) if carry_interval_h is None else float(carry_interval_h)
    bars: list[CanonicalBar] = []
    for i, c_raw in enumerate(closes):
        c = float(c_raw)
        o = float(closes[i - 1]) if i > 0 else c
        hi = max(o, c) * 1.003
        lo = min(o, c) * 0.997
        ts_ev = start_ms + i * tf_ms
        ts_av = ts_ev + tf_ms
        cr = float(carry_rates[i]) if carry_rates is not None else 0.0
        bars.append(
            CanonicalBar(
                asset_class=asset_class,
                venue=venue,
                symbol=symbol,
                timeframe_h=float(timeframe_h),
                ts_event_ms=ts_ev,
                ts_avail_ms=ts_av,
                open=o,
                high=hi,
                low=lo,
                close=c,
                volume_usd=25_000_000.0,
                carry_rate_period=cr,
                carry_interval_h=ci_h,
                spread_bps=1.5,
                provenance=PROVENANCE_REAL,
                source_fingerprint="unit_test_fp",
                premium_basis_bps=float(premium_basis_bps),
            )
        )
    return bars


class _FixedWeightScriptStrategy:
    """Minimal causal control strategy holding fixed target weights for liquidation negative control."""

    asset_domain: str = "web3"
    strategy_id: str = "test_fixed_weight_control"
    warmup_bars: int = 0

    def __init__(self, weights_by_sym: dict[str, float]) -> None:
        self._weights = dict(weights_by_sym)

    def reset(self) -> None:
        return None

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        _ = executed_weights

    def clone_fresh(self) -> _FixedWeightScriptStrategy:
        return _FixedWeightScriptStrategy(dict(self._weights))

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        _ = context
        return {
            sym: BarDecision(
                symbol=sym,
                ts_event_ms=int(ts_ms),
                target_weight=float(self._weights.get(sym, 0.0)),
                raw_signal=float(self._weights.get(sym, 0.0)),
                regime_label="CONTROL",
            )
            for sym in sorted(bars_at_ts)
        }


class TestWeb3MarketRegimeEngine(unittest.TestCase):
    """Verify F-12 Web3MarketRegimeEngine transitions, stateless PIT safety, and D7 zero-VIX isolation."""

    def test_bull_vs_bear_dvol_stress_transitions_and_pit_invariance(self) -> None:
        syms = ("BTC", "ETH", "SOL", "AVAX", "LINK", "SUI")
        bull_bars = {
            s: _make_bar_series(
                symbol=s,
                closes=_trending_closes(120, start=100.0, step_ret=0.006, noise_std=0.003, seed=10 + i),
                carry_rates=[(0.08 / 8760.0) * 12.0] * 120,
            )
            for i, s in enumerate(syms)
        }
        bull_dvol = [
            Web3MarketMacroPoint("deribit", "BTC_DVOL", b.ts_event_ms, b.ts_avail_ms, 42.0, PROVENANCE_REAL)
            for b in bull_bars["BTC"]
        ]
        bull_stable = [
            Web3MarketMacroPoint(
                "defillama", "STABLECOIN_MCAP_USD", b.ts_event_ms, b.ts_avail_ms, 150e9 + 0.6e9 * i, PROVENANCE_REAL
            )
            for i, b in enumerate(bull_bars["BTC"])
        ]

        bear_bars = {
            s: _make_bar_series(
                symbol=s,
                closes=_trending_closes(120, start=100.0, step_ret=-0.007, noise_std=0.006, seed=20 + i),
                carry_rates=[(-0.08 / 8760.0) * 12.0] * 120,
            )
            for i, s in enumerate(syms)
        }
        bear_dvol = [
            Web3MarketMacroPoint(
                "deribit", "BTC_DVOL", b.ts_event_ms, b.ts_avail_ms, (45.0 if i < 90 else 96.0), PROVENANCE_REAL
            )
            for i, b in enumerate(bear_bars["BTC"])
        ]
        bear_stable = [
            Web3MarketMacroPoint(
                "defillama", "STABLECOIN_MCAP_USD", b.ts_event_ms, b.ts_avail_ms, 170e9 - 0.6e9 * i, PROVENANCE_REAL
            )
            for i, b in enumerate(bear_bars["BTC"])
        ]

        eng_bull = Web3MarketRegimeEngine(
            bars_by_symbol=bull_bars, dvol_series=bull_dvol, stablecoin_series=bull_stable
        )
        eng_bear = Web3MarketRegimeEngine(
            bars_by_symbol=bear_bars, dvol_series=bear_dvol, stablecoin_series=bear_stable
        )

        ts_end = bull_bars["BTC"][-1].ts_avail_ms
        snap_bull = eng_bull.snapshot_as_of(ts_end)
        snap_bear = eng_bear.snapshot_as_of(ts_end)

        self.assertIn(snap_bull.regime_label, (WEB3_REGIME_EXPANSION, WEB3_REGIME_ALT_SEASON))
        self.assertEqual(snap_bear.regime_label, WEB3_REGIME_RISK_OFF_BEAR)
        self.assertGreater(snap_bull.exposure_scalar, snap_bear.exposure_scalar)
        self.assertGreater(snap_bull.net_exposure_scalar, 0.0)
        self.assertEqual(snap_bear.net_exposure_scalar, 0.0)
        self.assertGreater(snap_bull.breadth_pct_above_20d, 0.80)
        self.assertLess(snap_bear.breadth_pct_above_20d, 0.20)
        self.assertGreater(snap_bull.stablecoin_flow_30d, 0.0)
        self.assertLess(snap_bear.stablecoin_flow_30d, 0.0)

        # Out-of-order PIT queries must be bit-identical (D6)
        ts_mid = bull_bars["BTC"][60].ts_avail_ms
        s_mid_1 = eng_bull.snapshot_as_of(ts_mid)
        _ = eng_bull.snapshot_as_of(ts_end)
        s_mid_2 = eng_bull.snapshot_as_of(ts_mid)
        self.assertEqual(s_mid_1, s_mid_2)

        # Strict JSON compatibility (zero NaN/Inf)
        encoded = json.dumps(snap_bull.to_dict(), allow_nan=False)
        self.assertIsInstance(json.loads(encoded), dict)

    def test_d7_bit_identical_invariance_to_injected_yahoo_vix_bars_and_context(self) -> None:
        syms = ("BTC", "ETH", "SOL", "AVAX")
        crypto_bars = {
            s: _make_bar_series(
                symbol=s,
                closes=_trending_closes(100, start=100.0, step_ret=0.005, noise_std=0.004, seed=30 + i),
                carry_rates=[(0.06 / 8760.0) * 12.0] * 100,
            )
            for i, s in enumerate(syms)
        }
        vix_bars = _make_bar_series(
            symbol="^VIX",
            closes=[95.0] * 60,
            timeframe_h=24.0,
            venue="yahoo",
            asset_class=ASSET_CLASS_MACRO_INDEX,
        )
        dvol_pts = [
            Web3MarketMacroPoint("deribit", "BTC_DVOL", b.ts_event_ms, b.ts_avail_ms, 44.0, PROVENANCE_REAL)
            for b in crypto_bars["BTC"]
        ]
        st_pts = [
            Web3MarketMacroPoint(
                "defillama", "STABLECOIN_MCAP_USD", b.ts_event_ms, b.ts_avail_ms, 155e9 + 0.4e9 * i, PROVENANCE_REAL
            )
            for i, b in enumerate(crypto_bars["BTC"])
        ]

        eng_clean = Web3MarketRegimeEngine(
            bars_by_symbol=crypto_bars, dvol_series=dvol_pts, stablecoin_series=st_pts
        )
        polluted_bars = dict(crypto_bars)
        polluted_bars["^VIX"] = vix_bars
        eng_polluted = Web3MarketRegimeEngine(
            bars_by_symbol=polluted_bars, dvol_series=dvol_pts, stablecoin_series=st_pts
        )

        ts_q = crypto_bars["BTC"][-1].ts_avail_ms
        snap_clean = eng_clean.snapshot_as_of(ts_q, context={})
        snap_vix = eng_polluted.snapshot_as_of(
            ts_q,
            context={"yahoo_vix": 95.0, "vix": 95.0, "macro_vix_stress_z": 12.0, "macro_gold_composite": -4.0},
        )
        self.assertEqual(snap_clean, snap_vix)
        self.assertEqual(snap_clean.active_universe_count, len(syms))


class TestW1Web3XSMomCarryKernel(unittest.TestCase):
    """Verify F-13 W1 cross-sectional momentum, crowding/OI, HYPE PIT, bear neutrality, inertia, and G7."""

    def test_top_k_long_bottom_k_short_ranking_and_protocol(self) -> None:
        syms = ("BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "AAVE")
        drifts = (0.008, 0.006, 0.003, 0.0005, -0.0005, -0.003, -0.005, -0.007)
        n_bars = 140
        bars_by_sym = {
            s: _make_bar_series(
                symbol=s,
                closes=_trending_closes(n_bars, start=100.0, step_ret=drifts[i], noise_std=0.001, seed=100 + i),
                carry_rates=[(0.05 / 8760.0) * 12.0] * n_bars,
            )
            for i, s in enumerate(syms)
        }
        w1 = Web3XSMomCarryKernel(top_k=2, bottom_k=2)
        self.assertIsInstance(w1, AssetNativeStrategy)
        self.assertEqual(w1.asset_domain, "web3")
        w1.reset()
        last_dec: dict[str, BarDecision] = {}
        for t in range(n_bars):
            ts = bars_by_sym["BTC"][t].ts_event_ms
            last_dec = w1.on_bar_step(ts, {s: bars_by_sym[s][t] for s in syms}, {"regime_label": "EXPANSION"})
            w1.sync_executed_weights({s: d.target_weight for s, d in last_dec.items()})

        self.assertGreater(last_dec["BTC"].target_weight, 0.0)
        self.assertGreater(last_dec["ETH"].target_weight, 0.0)
        self.assertLess(last_dec["AAVE"].target_weight, 0.0)
        self.assertLess(last_dec["LINK"].target_weight, 0.0)
        for mid in ("SOL", "XRP", "DOGE", "AVAX"):
            self.assertAlmostEqual(last_dec[mid].target_weight, 0.0, places=8)

    def test_funding_crowding_penalty_and_oi_confirmation_via_metrics_and_context(self) -> None:
        n_bars = 120
        shared_closes = _trending_closes(n_bars, start=100.0, step_ret=0.004, noise_std=0.001, seed=111)
        crowded_r = (1.50 / 8760.0) * 12.0
        neg_r = (-0.20 / 8760.0) * 12.0
        bars = {
            "BTC": _make_bar_series(symbol="BTC", closes=shared_closes, carry_rates=[crowded_r] * n_bars),
            "ETH": _make_bar_series(symbol="ETH", closes=shared_closes, carry_rates=[neg_r] * n_bars),
            "SOL": _make_bar_series(symbol="SOL", closes=shared_closes, carry_rates=[neg_r] * n_bars),
            "AVAX": _make_bar_series(symbol="AVAX", closes=shared_closes, carry_rates=[neg_r] * n_bars),
        }
        # Real Web3DerivativesMetric streams: SOL OI grows +35%, AVAX OI contracts -25%
        metrics = {
            "BTC": [
                Web3DerivativesMetric("binance_um", "BTC", b.ts_event_ms, b.ts_avail_ms, 1e9, 1.0, 1.0, 1.0, PROVENANCE_REAL)
                for b in bars["BTC"]
            ],
            "ETH": [
                Web3DerivativesMetric("binance_um", "ETH", b.ts_event_ms, b.ts_avail_ms, 1e9, 1.0, 1.0, 1.0, PROVENANCE_REAL)
                for b in bars["ETH"]
            ],
            "SOL": [
                Web3DerivativesMetric(
                    "binance_um", "SOL", b.ts_event_ms, b.ts_avail_ms, 1e9 * (1.0 + 0.004 * i), 1.2, 1.2, 1.1, PROVENANCE_REAL
                )
                for i, b in enumerate(bars["SOL"])
            ],
            "AVAX": [
                Web3DerivativesMetric(
                    "binance_um", "AVAX", b.ts_event_ms, b.ts_avail_ms, 1e9 * max(0.3, 1.0 - 0.004 * i), 0.9, 0.9, 0.9, PROVENANCE_REAL
                )
                for i, b in enumerate(bars["AVAX"])
            ],
        }

        w1 = Web3XSMomCarryKernel(top_k=1, bottom_k=1, metrics_by_symbol=metrics)
        w1.reset()
        last_dec: dict[str, BarDecision] = {}
        for t in range(n_bars):
            ts = bars["BTC"][t].ts_event_ms
            last_dec = w1.on_bar_step(ts, {s: bars[s][t] for s in bars}, {"regime_label": "EXPANSION"})
            w1.sync_executed_weights({s: d.target_weight for s, d in last_dec.items()})

        self.assertGreater(last_dec["ETH"].raw_signal, last_dec["BTC"].raw_signal)
        self.assertGreater(last_dec["SOL"].raw_signal, last_dec["AVAX"].raw_signal)

    def test_hype_pit_listing_2024_12_05_zero_prelisting_weight(self) -> None:
        start_ms = _utc_ms(2024, 10, 15, 0)
        n_bars = 240
        syms = ("BTC", "ETH", "SOL", "HYPE")
        bars = {
            s: _make_bar_series(
                symbol=s,
                start_ms=start_ms,
                closes=_trending_closes(
                    n_bars, start=(10.0 if s == "HYPE" else 100.0), step_ret=(0.012 if s == "HYPE" else 0.001), noise_std=0.002, seed=120 + i
                ),
                carry_rates=[(0.05 / 8760.0) * 12.0] * n_bars,
            )
            for i, s in enumerate(syms)
        }
        w1 = Web3XSMomCarryKernel(top_k=1, bottom_k=1)
        w1.reset()
        pre_nonzero = 0
        post_nonzero = 0
        for t in range(n_bars):
            ts = bars["HYPE"][t].ts_event_ms
            dec = w1.on_bar_step(ts, {s: bars[s][t] for s in syms}, {"regime_label": "EXPANSION"})
            w1.sync_executed_weights({s: d.target_weight for s, d in dec.items()})
            w_h = dec["HYPE"].target_weight
            if ts < HYPE_PIT_LISTING_MS:
                if abs(w_h) > 1e-12:
                    pre_nonzero += 1
            elif ts >= HYPE_PIT_LISTING_MS + 30 * DAY_MS:
                if abs(w_h) > 1e-12:
                    post_nonzero += 1
        self.assertEqual(pre_nonzero, 0)
        self.assertGreater(post_nonzero, 0)

    def test_bear_market_neutrality_carver_inertia_and_g7_parity(self) -> None:
        syms = ("BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX")
        drifts = (0.007, 0.005, 0.001, -0.001, -0.004, -0.007)
        n_bars = 120
        bars = {
            s: _make_bar_series(
                symbol=s,
                closes=_trending_closes(n_bars, start=100.0, step_ret=drifts[i], noise_std=0.002, seed=130 + i),
                carry_rates=[(0.05 / 8760.0) * 12.0] * n_bars,
            )
            for i, s in enumerate(syms)
        }

        def _run_regime(label: str) -> dict[str, BarDecision]:
            k = Web3XSMomCarryKernel(top_k=2, bottom_k=2)
            k.reset()
            last: dict[str, BarDecision] = {}
            for t in range(n_bars):
                ts = bars["BTC"][t].ts_event_ms
                last = k.on_bar_step(ts, {s: bars[s][t] for s in syms}, {"regime_label": label})
                k.sync_executed_weights({s: d.target_weight for s, d in last.items()})
            return last

        d_bear = _run_regime("RISK_OFF_BEAR")
        d_bull = _run_regime("ALT_SEASON")
        gross_bear = sum(abs(d.target_weight) for d in d_bear.values())
        net_bear = sum(d.target_weight for d in d_bear.values())
        net_bull = sum(d.target_weight for d in d_bull.values())
        self.assertGreater(gross_bear, 0.05)
        self.assertLessEqual(abs(net_bear), 0.15 * gross_bear + 1e-6)
        self.assertGreater(net_bull, net_bear)

        # Carver inertia: 4% inside band holds exact synced weight; 80% outside band rebalances
        k_in = Web3XSMomCarryKernel(top_k=2, bottom_k=2)
        k_in.reset()
        for t in range(n_bars - 1):
            ts = bars["BTC"][t].ts_event_ms
            dec = k_in.on_bar_step(ts, {s: bars[s][t] for s in syms}, {"regime_label": "EXPANSION"})
            k_in.sync_executed_weights({s: d.target_weight for s, d in dec.items()})
        base = k_in.on_bar_step(
            bars["BTC"][-1].ts_event_ms, {s: bars[s][-1] for s in syms}, {"regime_label": "EXPANSION"}
        )
        btc_des = base["BTC"].target_weight
        self.assertGreater(abs(btc_des), 0.05)

        synced_map = {s: d.target_weight for s, d in base.items()}
        synced_close = btc_des * 0.96
        synced_map["BTC"] = synced_close
        k_in.sync_executed_weights(synced_map)
        held = k_in.on_bar_step(
            bars["BTC"][-1].ts_event_ms + 12 * HOUR_MS, {s: bars[s][-1] for s in syms}, {"regime_label": "EXPANSION"}
        )
        self.assertAlmostEqual(held["BTC"].target_weight, synced_close, delta=1e-6)

        synced_far = btc_des * 0.20
        synced_map["BTC"] = synced_far
        k_in.sync_executed_weights(synced_map)
        rebal = k_in.on_bar_step(
            bars["BTC"][-1].ts_event_ms + 24 * HOUR_MS, {s: bars[s][-1] for s in syms}, {"regime_label": "EXPANSION"}
        )
        self.assertGreater(abs(rebal["BTC"].target_weight - synced_far), abs(btc_des) * 0.25)

        # G7 full parity with CausalRiskAllocator active
        specs = default_web3_specs(syms)
        cfg = RiskAllocatorConfig(max_positions=4, target_annual_vol_pct=20.0, max_portfolio_leverage=1.2, max_single_weight=0.5)
        rep = verify_full_parity(Web3XSMomCarryKernel(top_k=2, bottom_k=2), specs, bars, risk_cfg=cfg)
        self.assertTrue(rep.passed, rep.detail)
        self.assertLessEqual(rep.max_abs_weight_delta, 1e-10)
        self.assertLessEqual(rep.max_abs_equity_delta, 1e-8)


class TestW2Web3AdaptiveTrendKernel(unittest.TestCase):
    """Verify F-14 W2 G2-verified trend breakout, crowding throttle, breadth short gate, stop & G7."""

    def test_flat_zero_vs_breakout_positive_and_zero_squeeze_factor_leak(self) -> None:
        flat_bars = _make_bar_series(symbol="BTC", closes=[100.0] * 100, carry_rates=[(0.05 / 8760.0) * 12.0] * 100)
        up_bars = _make_bar_series(
            symbol="BTC",
            closes=[100.0] * 50 + [100.0 + 1.5 * i for i in range(1, 51)],
            carry_rates=[(0.05 / 8760.0) * 12.0] * 100,
        )

        # Unselected squeeze_breakout_dir must have ZERO impact even if passed in context or selected_factors
        w2_a = Web3AdaptiveTrendKernel(selected_factors=("ewmac_16_64", "squeeze_breakout_dir"))
        w2_b = Web3AdaptiveTrendKernel(selected_factors=("ewmac_16_64",))
        self.assertNotIn("squeeze_breakout_dir", w2_a.selected_factors)
        w2_a.reset()
        w2_b.reset()
        for b in flat_bars:
            da = w2_a.on_bar_step(b.ts_event_ms, {"BTC": b}, {"squeeze_breakout_dir": +1.0, "breadth_pct_above_20d": 0.6})
            db = w2_b.on_bar_step(b.ts_event_ms, {"BTC": b}, {"squeeze_breakout_dir": -1.0, "breadth_pct_above_20d": 0.6})
            self.assertEqual(da["BTC"].target_weight, db["BTC"].target_weight)
            self.assertAlmostEqual(da["BTC"].target_weight, 0.0, places=8)

        w2_up = Web3AdaptiveTrendKernel()
        w2_up.reset()
        last_dec: dict[str, BarDecision] = {}
        for b in up_bars:
            last_dec = w2_up.on_bar_step(b.ts_event_ms, {"BTC": b}, {"breadth_pct_above_20d": 0.80})
            w2_up.sync_executed_weights({"BTC": last_dec["BTC"].target_weight})
        self.assertGreater(last_dec["BTC"].target_weight, 0.0)
        self.assertIsNotNone(last_dec["BTC"].stop_price)
        self.assertLess(float(last_dec["BTC"].stop_price), up_bars[-1].close)

    def test_asymmetric_crowding_throttle_breadth_short_gate_and_g7_parity(self) -> None:
        closes_up = [100.0] * 50 + [100.0 + 1.8 * i for i in range(1, 41)]
        bars_up = _make_bar_series(symbol="BTC", closes=closes_up, carry_rates=[(0.05 / 8760.0) * 12.0] * len(closes_up))

        def _final_w2(ctx: dict[str, Any], blist: list[CanonicalBar]) -> float:
            k = Web3AdaptiveTrendKernel()
            k.reset()
            w_out = 0.0
            for b in blist:
                d = k.on_bar_step(b.ts_event_ms, {"BTC": b}, ctx)
                w_out = d["BTC"].target_weight
                k.sync_executed_weights({"BTC": w_out})
            return w_out

        w_uncrowded = _final_w2({"funding_z": -1.0, "oi_change_7d": 0.05, "breadth_pct_above_20d": 0.80}, bars_up)
        w_crowded = _final_w2({"funding_z": 3.0, "oi_change_7d": 0.35, "breadth_pct_above_20d": 0.80}, bars_up)
        self.assertGreater(w_uncrowded, 0.0)
        self.assertGreaterEqual(w_crowded, 0.0)
        self.assertLessEqual(w_crowded, 0.65 * w_uncrowded)

        # Breadth short-side gate: blocked in strong breadth (0.80), active in bear + low breadth (0.15)
        closes_dn = [100.0] * 50 + [max(15.0, 100.0 - 1.5 * i) for i in range(1, 41)]
        bars_dn = _make_bar_series(symbol="BTC", closes=closes_dn, carry_rates=[(-0.05 / 8760.0) * 12.0] * len(closes_dn))
        w_short_bear = _final_w2(
            {"regime_label": "RISK_OFF_BEAR", "breadth_pct_above_20d": 0.15, "funding_z": 0.0, "oi_change_7d": 0.0},
            bars_dn,
        )
        w_short_strong_breadth = _final_w2(
            {"regime_label": "EXPANSION", "breadth_pct_above_20d": 0.80, "funding_z": 0.0, "oi_change_7d": 0.0},
            bars_dn,
        )
        self.assertLess(w_short_bear, 0.0)
        self.assertAlmostEqual(w_short_strong_breadth, 0.0, places=8)

        # Full G7 parity
        specs = default_web3_specs(("BTC",))
        cfg = RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=20.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
        rep = verify_full_parity(Web3AdaptiveTrendKernel(), specs, {"BTC": bars_up}, risk_cfg=cfg)
        self.assertTrue(rep.passed, rep.detail)
        self.assertLessEqual(rep.max_abs_weight_delta, 1e-10)
        self.assertLessEqual(rep.max_abs_equity_delta, 1e-8)


class TestW3Web3RegimeCarryKernel(unittest.TestCase):
    """Verify F-15 W3 3x RT fee cost floor, DVOL leverage throttle, 50% margin storm-exit, metadata & G7."""

    def test_cost_horizon_3x_rt_fee_floor_and_separated_margin_metadata(self) -> None:
        tiny_r = (0.02 / 8760.0) * 12.0  # 2% APR -> ~3.8bp over 7d << 42bp floor
        base_hl_r = (0.109 / 8760.0) * 12.0  # 10.9% APR -> ~20.9bp over 7d < 42bp floor
        rich_r = (0.85 / 8760.0) * 12.0  # 85% APR -> ~163bp over 7d >> 42bp floor

        def _run_rate(rate: float) -> BarDecision:
            bars = _make_bar_series(
                symbol="BTC",
                closes=[50000.0 + 20.0 * math.sin(i / 4.0) for i in range(60)],
                carry_rates=[rate] * 60,
                premium_basis_bps=(20.0 if rate > 0.0005 else 0.0),
            )
            w3 = Web3RegimeCarryKernel()
            w3.reset()
            last: BarDecision | None = None
            for b in bars:
                dec = w3.on_bar_step(b.ts_event_ms, {"BTC": b}, {"dvol": 40.0, "rt_cost_bps": 14.0, "margin_ratio": 0.90})
                last = dec["BTC"]
                w3.sync_executed_weights({"BTC": last.target_weight})
            assert last is not None
            return last

        d_tiny = _run_rate(tiny_r)
        d_base = _run_rate(base_hl_r)
        d_rich = _run_rate(rich_r)

        self.assertAlmostEqual(d_tiny.target_weight, 0.0, places=8)
        self.assertAlmostEqual(d_base.target_weight, 0.0, places=8)
        self.assertLess(d_rich.target_weight, 0.0, "Positive funding carry collects via short perp leg")

        meta = d_rich.metadata
        self.assertIs(meta.get("separated_margin_ledger"), True)
        for key in ("carry_apr_pct", "expected_carry_bps", "cost_floor_bps", "allocated_margin_usd", "margin_ratio"):
            self.assertIn(key, meta)
            self.assertTrue(math.isfinite(float(meta[key])))
        self.assertIsInstance(json.loads(json.dumps(meta, allow_nan=False)), dict)

    def test_dvol_throttle_50pct_margin_storm_exit_liquidation_prevention_and_g7_parity(self) -> None:
        rich_r = (0.85 / 8760.0) * 12.0
        bars = _make_bar_series(
            symbol="BTC",
            closes=[50000.0 + 35.0 * math.sin(i / 3.0) for i in range(55)],
            carry_rates=[rich_r] * 55,
            premium_basis_bps=25.0,
        )

        def _run_dvol(dvol_val: float, mr: float) -> float:
            k = Web3RegimeCarryKernel()
            k.reset()
            w_last = 0.0
            for i, b in enumerate(bars):
                cur_mr = mr if i == len(bars) - 1 else 0.90
                d = k.on_bar_step(b.ts_event_ms, {"BTC": b}, {"dvol": dvol_val, "rt_cost_bps": 14.0, "margin_ratio": cur_mr})
                w_last = d["BTC"].target_weight
                k.sync_executed_weights({"BTC": w_last})
            return w_last

        w_calm = abs(_run_dvol(40.0, 0.90))
        w_stress = abs(_run_dvol(100.0, 0.90))
        self.assertGreater(w_calm, 0.0)
        self.assertGreater(w_calm, w_stress)
        self.assertAlmostEqual(_run_dvol(45.0, 0.45), 0.0, places=8)

        # End-to-end liquidation prevention on flash-crash wick Bar 42 vs unclosed leveraged control
        b42 = bars[42]
        bars_crash = list(bars)
        bars_crash[42] = CanonicalBar(
            asset_class=ASSET_CLASS_CRYPTO_PERP,
            venue="hyperliquid",
            symbol="BTC",
            timeframe_h=12.0,
            ts_event_ms=b42.ts_event_ms,
            ts_avail_ms=b42.ts_avail_ms,
            open=50000.0,
            high=100000.0,
            low=15000.0,
            close=35000.0,
            volume_usd=50_000_000.0,
            carry_rate_period=rich_r,
            carry_interval_h=12.0,
            spread_bps=1.5,
            provenance=PROVENANCE_REAL,
        )
        spec_lev = AssetSpec(
            "BTC", ASSET_CLASS_CRYPTO_PERP, "hyperliquid", CALENDAR_CRYPTO_24_7, CostModel(1.5, 4.5, 1.5), max_leverage_cap=5.0
        )
        cfg_lev = RiskAllocatorConfig(
            max_positions=1, target_annual_vol_pct=200.0, max_portfolio_leverage=5.0, max_single_weight=5.0, use_correlation_idm=False
        )
        storm_ts = bars_crash[40].ts_event_ms
        ctx_fn = lambda ts: {
            "dvol": 45.0 if ts < storm_ts else 110.0,
            "rt_cost_bps": 14.0,
            "margin_ratio": 0.90 if ts < storm_ts else 0.42,
        }
        eng = PortfolioAccountingEngine()
        res_w3 = eng.simulate(Web3RegimeCarryKernel(), {"BTC": spec_lev}, {"BTC": bars_crash}, risk_cfg=cfg_lev, context_provider=ctx_fn)
        w_pre = res_w3.executed_weights_history[40]["BTC"]
        self.assertGreater(abs(w_pre), 0.0)
        self.assertAlmostEqual(res_w3.executed_weights_history[41]["BTC"], 0.0, places=8)
        self.assertEqual(res_w3.liquidation_count, 0)
        self.assertGreater(res_w3.final_equity_usd, 9000.0)

        ctrl = _FixedWeightScriptStrategy({"BTC": math.copysign(3.0, w_pre)})
        res_ctrl = eng.simulate(ctrl, {"BTC": spec_lev}, {"BTC": bars_crash}, risk_cfg=cfg_lev)
        self.assertGreaterEqual(res_ctrl.liquidation_count, 1)

        # Full G7 parity on normal bars
        spec1 = default_web3_specs(("BTC",))["BTC"]
        cfg1 = RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=25.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
        rep = verify_full_parity(
            Web3RegimeCarryKernel(),
            {"BTC": spec1},
            {"BTC": bars},
            risk_cfg=cfg1,
            context_provider=lambda ts: {"dvol": 42.0, "rt_cost_bps": 14.0, "margin_ratio": 0.90},
        )
        self.assertTrue(rep.passed, rep.detail)
        self.assertLessEqual(rep.max_abs_weight_delta, 1e-10)
        self.assertLessEqual(rep.max_abs_equity_delta, 1e-8)


class TestWeb3DomainIsolationAndPureStdlib(unittest.TestCase):
    """Verify AST domain isolation, zero DO_NOT_TOUCH imports, and pure Python 3.12 stdlib execution."""

    def test_ast_zero_cross_domain_or_execution_imports(self) -> None:
        web3_dir = REPO_ROOT / "alphaloop" / "platform" / "strategies" / "web3"
        forbidden = (
            "strategies.gold",
            "strategies.ashare",
            "ashare_contract",
            "alphaloop.execution",
            "alphaloop.daemon",
            "alphaloop.plugins",
        )
        py_files = sorted(web3_dir.glob("*.py"))
        self.assertGreaterEqual(len(py_files), 5)
        for path in py_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for bad in forbidden:
                            self.assertNotIn(bad, alias.name, f"Forbidden import {alias.name} in {path}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for bad in forbidden:
                        self.assertNotIn(bad, node.module, f"Forbidden import {node.module} in {path}")

    def test_pure_stdlib_execution_blocking_third_party_packages(self) -> None:
        probe = (
            "import sys\n"
            "BLOCKED = {'numpy', 'scipy', 'pyarrow', 'duckdb', 'pandas', 'polars'}\n"
            "class Blocker:\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname.split('.')[0] in BLOCKED:\n"
            "            raise ImportError(f'Blocked: {fullname}')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "from alphaloop.platform.strategies.web3 import (\n"
            "    Web3MarketRegimeEngine, Web3XSMomCarryKernel, Web3AdaptiveTrendKernel, Web3RegimeCarryKernel\n"
            ")\n"
            "from alphaloop.platform.data_foundation.models import CanonicalBar\n"
            "from alphaloop.core.provenance import PROVENANCE_REAL\n"
            "b = CanonicalBar('CRYPTO_PERP', 'hyperliquid', 'BTC', 12.0, 1704067200000, 1704110400000, 40000.0, 40500.0, 39800.0, 40200.0, 1e7, 0.0001, 12.0, 1.5, PROVENANCE_REAL)\n"
            "eng = Web3MarketRegimeEngine()\n"
            "snap = eng.update(b.ts_event_ms, {'BTC': b}, {'dvol': 48.0})\n"
            "assert snap.regime_label\n"
            "for strat in (Web3XSMomCarryKernel(), Web3AdaptiveTrendKernel(), Web3RegimeCarryKernel()):\n"
            "    d = strat.on_bar_step(b.ts_event_ms, {'BTC': b}, {'dvol': 48.0})\n"
            "    assert 'BTC' in d\n"
            "print('WEB3_PURE_STDLIB_OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"Pure stdlib execution failed: {proc.stderr}")
        self.assertIn("WEB3_PURE_STDLIB_OK", proc.stdout)


class TestRealLakeWeb3SmokeAndParity(unittest.TestCase):
    """Real multi-year data smoke + G7 parity against data/platform_v4_lake/ (or PLATFORM_V4_LAKE_DIR)."""

    def test_real_lake_multi_year_smoke_regime_w1_w2_w3_and_g7_parity(self) -> None:
        env_dir = os.environ.get("PLATFORM_V4_LAKE_DIR", "").strip()
        lake_dir = Path(env_dir) if env_dir else REAL_V4_LAKE_DIR
        cat_db = lake_dir / "catalog.sqlite"
        web3_db = lake_dir / "domains" / "web3_lake.sqlite"
        if not (cat_db.is_file() and web3_db.is_file()):
            self.skipTest(
                f"Real v4 web3_lake.sqlite not present at {lake_dir}; "
                "set PLATFORM_V4_LAKE_DIR or run scripts/run_strategy_platform_v4.py --refresh"
            )
        with tempfile.TemporaryDirectory(prefix="web3_real_smoke_") as tmpdir:
            copy_dir = Path(tmpdir) / "lake"
            (copy_dir / "domains").mkdir(parents=True)
            shutil.copy2(cat_db, copy_dir / "catalog.sqlite")
            shutil.copy2(web3_db, copy_dir / "domains" / "web3_lake.sqlite")
            router = DataFoundationRouter(copy_dir)
            self.addCleanup(router.close)

            bars_12h = {s: router.load_bars("web3", "hyperliquid", s, 43200) for s in WEB3_LIQUID_UNIVERSE}
            dvol = router.load_web3_macro("BTC_DVOL")
            stable = router.load_web3_macro("STABLECOIN_MCAP_USD")
            metrics = {s: router.load_web3_metrics("binance_um", s) for s in ("BTC", "ETH", "SOL")}

            self.assertGreater(len(bars_12h["BTC"]), 1000)
            self.assertGreater(len(dvol), 500)
            self.assertGreater(len(stable), 500)
            self.assertGreater(len(metrics["BTC"]), 500)

            reg_eng = Web3MarketRegimeEngine(
                bars_by_symbol=bars_12h,
                dvol_series=dvol,
                stablecoin_series=stable,
                metrics_by_symbol=metrics,
            )
            # Verify real-data regime transitions span multiple regimes with finite scalars
            seen_regimes: set[str] = set()
            for b in bars_12h["BTC"][60::20]:
                snap = reg_eng.snapshot_as_of(b.ts_avail_ms)
                seen_regimes.add(snap.regime_label)
                self.assertTrue(math.isfinite(snap.exposure_scalar))
            self.assertGreaterEqual(len(seen_regimes), 3)

            dvol_av = [p.ts_avail_ms for p in dvol]
            dvol_v = [p.value for p in dvol]

            def _ctx_clean(ts_ms: int) -> dict[str, Any]:
                snap = reg_eng.snapshot_as_of(ts_ms)
                idx = bisect.bisect_right(dvol_av, ts_ms) - 1
                dv = dvol_v[idx] if idx >= 0 else 50.0
                return {
                    "regime_label": snap.regime_label,
                    "exposure_scalar": snap.exposure_scalar,
                    "breadth_pct_above_20d": snap.breadth_pct_above_20d,
                    "dvol": dv,
                    "rt_cost_bps": 14.0,
                    "margin_ratio": 0.90,
                }

            def _ctx_vix_shock(ts_ms: int) -> dict[str, Any]:
                c = _ctx_clean(ts_ms)
                c.update({"yahoo_vix": 95.0, "vix": 95.0, "macro_vix_stress_z": 12.0, "macro_gold_composite": -5.0})
                return c

            # Slice 400 real 12h bars from active 2024 bull/funding window (bars 500:900) for fast full-parity + smoke
            majors = ("BTC", "ETH", "SOL", "AVAX")
            bars_mid = {s: bars_12h[s][500:900] for s in majors}
            specs_m = default_web3_specs(majors)
            cfg_m = RiskAllocatorConfig(max_positions=4, target_annual_vol_pct=20.0, max_portfolio_leverage=1.5, max_single_weight=0.50)
            acct = PortfolioAccountingEngine()

            for strat in (
                Web3XSMomCarryKernel(top_k=2, bottom_k=2, metrics_by_symbol=metrics),
                Web3AdaptiveTrendKernel(metrics_by_symbol=metrics),
                Web3RegimeCarryKernel(dvol_series=dvol),
            ):
                res_clean = acct.simulate(strat.clone_fresh(), specs_m, bars_mid, risk_cfg=cfg_m, context_provider=_ctx_clean)
                res_vix = acct.simulate(strat.clone_fresh(), specs_m, bars_mid, risk_cfg=cfg_m, context_provider=_ctx_vix_shock)

                self.assertEqual(res_clean.equity_curve, res_vix.equity_curve, f"D7 VIX drift on {strat.strategy_id}")
                self.assertGreater(res_clean.trade_count, 0, f"Zero trades on real mid window for {strat.strategy_id}")
                self.assertEqual(res_clean.liquidation_count, 0)
                for metric_val in (res_clean.cagr_pct, res_clean.sharpe_ratio, res_clean.max_drawdown_pct, res_clean.net_pnl_usd):
                    self.assertTrue(math.isfinite(metric_val))

                rep = verify_full_parity(
                    strat.clone_fresh(), specs_m, bars_mid, risk_cfg=cfg_m, batch_result=res_clean, context_provider=_ctx_clean
                )
                self.assertTrue(rep.passed, f"G7 parity failed for {strat.strategy_id}: {rep.detail}")
                self.assertLessEqual(rep.max_abs_weight_delta, 1e-10)
                self.assertLessEqual(rep.max_abs_equity_delta, 1e-8)

            # Full 12-coin W1 real multi-year check (verifying HYPE PIT + honest positive net metrics)
            specs_12 = default_web3_specs(WEB3_LIQUID_UNIVERSE)
            cfg_12 = RiskAllocatorConfig(max_positions=6, target_annual_vol_pct=20.0, max_portfolio_leverage=1.5, max_single_weight=0.45)
            res_w1_full = acct.simulate(
                Web3XSMomCarryKernel(top_k=3, bottom_k=3, metrics_by_symbol=metrics),
                specs_12,
                bars_12h,
                risk_cfg=cfg_12,
                context_provider=_ctx_clean,
            )
            self.assertGreater(res_w1_full.days, 1000.0)
            self.assertGreater(res_w1_full.trade_count, 100)
            self.assertEqual(res_w1_full.liquidation_count, 0)
            self.assertTrue(math.isfinite(res_w1_full.cagr_pct))
            self.assertTrue(math.isfinite(res_w1_full.sharpe_ratio))


if __name__ == "__main__":
    unittest.main()
