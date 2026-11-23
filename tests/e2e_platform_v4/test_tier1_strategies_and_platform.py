"""Tier 1 Feature Coverage E2E Tests: FA-06..FA-11 (Strategies, Platform Runner & Governance).

Strictly opaque-box tests derived from ORIGINAL_REQUEST.md and PROJECT.md Interface Contracts.
All Platform v4 imports are lazy (inside test methods) so `--list` discovers cleanly before
parallel implementation milestones complete. Zero skips, zero fake passes, pure Python 3.12 stdlib.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_AUTO_ROOT = _THIS_DIR.parents[1]
if str(_AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

if __package__:
    from . import fixtures
else:
    import fixtures


class TestTier1FA06Web3Strategies(unittest.TestCase):
    """FA-06: Crypto-Native Web3 Regime & Decoupled Strategy Suite W1/W2/W3 (F-12..F-15)."""

    def test_t1_fa06_web3_market_regime_engine_crypto_native_transitions(self) -> None:
        reg_mod = fixtures.import_web3_regime_mod()
        assets = fixtures.import_assets()
        models = fixtures.import_df_models()
        prov_real, _, _ = fixtures.import_provenance()

        # Construct constructive bull vs severe DVOL/bear stress (12h bars with exact carry_interval_h=12.0)
        bull_bars = {
            s: fixtures.build_bar_series(
                symbol=s,
                venue="hyperliquid",
                asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0,
                start_ms=fixtures.BASE_TS_MS,
                closes=fixtures.build_trending_closes(120, start_price=100.0, step_ret=0.006, noise_std=0.004, seed=60 + i),
                carry_rates=[0.00008] * 120,
                carry_interval_h=12.0,
            )
            for i, s in enumerate(("BTC", "ETH", "SOL", "AVAX", "LINK", "SUI"))
        }
        bull_dvol = [
            models.Web3MarketMacroPoint("deribit", "BTC_DVOL", b.ts_event_ms, b.ts_avail_ms, 42.0, prov_real)
            for b in bull_bars["BTC"]
        ]
        bull_stable = [
            models.Web3MarketMacroPoint("defillama", "STABLECOIN_MCAP_USD", b.ts_event_ms, b.ts_avail_ms, 150e9 + 0.5e9 * i, prov_real)
            for i, b in enumerate(bull_bars["BTC"])
        ]

        bear_bars = {
            s: fixtures.build_bar_series(
                symbol=s,
                venue="hyperliquid",
                asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0,
                start_ms=fixtures.BASE_TS_MS,
                closes=fixtures.build_trending_closes(120, start_price=100.0, step_ret=-0.007, noise_std=0.008, seed=70 + i),
                carry_rates=[-0.0001] * 120,
                carry_interval_h=12.0,
            )
            for i, s in enumerate(("BTC", "ETH", "SOL", "AVAX", "LINK", "SUI"))
        }
        bear_dvol = [
            models.Web3MarketMacroPoint(
                "deribit", "BTC_DVOL", b.ts_event_ms, b.ts_avail_ms, (45.0 if i < 95 else 96.0), prov_real
            )
            for i, b in enumerate(bear_bars["BTC"])
        ]
        bear_stable = [
            models.Web3MarketMacroPoint("defillama", "STABLECOIN_MCAP_USD", b.ts_event_ms, b.ts_avail_ms, 170e9 - 0.6e9 * i, prov_real)
            for i, b in enumerate(bear_bars["BTC"])
        ]

        ts_end = bull_bars["BTC"][-1].ts_event_ms
        bull_res = fixtures.invoke_web3_regime_engine(
            reg_mod.Web3MarketRegimeEngine, bull_bars, bull_dvol, bull_stable, ts_end
        )
        bear_res = fixtures.invoke_web3_regime_engine(
            reg_mod.Web3MarketRegimeEngine, bear_bars, bear_dvol, bear_stable, ts_end
        )

        bull_scalar = float(fixtures.get_metric(bull_res, "exposure_scalar", "net_exposure_scalar", "risk_scalar", "beta_scalar"))
        bear_scalar = float(fixtures.get_metric(bear_res, "exposure_scalar", "net_exposure_scalar", "risk_scalar", "beta_scalar"))
        self.assertGreater(bull_scalar, 0.0, "Bull/expansion regime must produce strictly positive exposure scalar")
        self.assertGreaterEqual(bear_scalar, 0.0, "Exposure scalar must remain non-negative")
        self.assertGreater(bull_scalar, bear_scalar, "Bull/expansion regime must allow strictly higher exposure than bear/DVOL stress")

    def test_t1_fa06_w1_xs_mom_carry_top_k_long_bottom_k_short_ranking(self) -> None:
        w1_mod = fixtures.import_w1_mod()
        assets = fixtures.import_assets()

        syms = ("BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX", "LINK", "AAVE")
        # Monotonic 60d+ trends: BTC strongest (+0.008/bar) down to AAVE weakest (-0.007/bar)
        drifts = [0.008, 0.006, 0.003, 0.0005, -0.0005, -0.003, -0.005, -0.007]
        n_bars = 140
        bars_by_sym = {
            s: fixtures.build_bar_series(
                symbol=s,
                venue="hyperliquid",
                asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0,
                start_ms=fixtures.BASE_TS_MS,
                closes=fixtures.build_trending_closes(n_bars, start_price=100.0, step_ret=drifts[i], noise_std=0.001, seed=80 + i),
                carry_rates=[0.00005] * n_bars,
                carry_interval_h=12.0,
            )
            for i, s in enumerate(syms)
        }
        w1 = fixtures.build_strategy_instance(w1_mod.Web3XSMomCarryKernel, top_k=2, bottom_k=2)
        w1.reset()
        last_dec: dict[str, Any] = {}
        for i in range(n_bars):
            ts_i = bars_by_sym["BTC"][i].ts_event_ms
            step_bars = {s: bars_by_sym[s][i] for s in syms}
            last_dec = w1.on_bar_step(ts_i, step_bars, {"regime_label": "EXPANSION", "exposure_scalar": 1.0})
            w1.sync_executed_weights({s: d.target_weight for s, d in last_dec.items()})

        self.assertGreater(last_dec["BTC"].target_weight, 0.0, "Top momentum coin must be long")
        self.assertGreater(last_dec["ETH"].target_weight, 0.0, "2nd momentum coin must be long")
        self.assertLess(last_dec["AAVE"].target_weight, 0.0, "Worst momentum coin must be short")
        self.assertLess(last_dec["LINK"].target_weight, 0.0, "2nd worst momentum coin must be short")
        self.assertAlmostEqual(last_dec["XRP"].target_weight, 0.0, places=6, msg="Middle coin XRP must be flat")
        self.assertAlmostEqual(last_dec["DOGE"].target_weight, 0.0, places=6, msg="Middle coin DOGE must be flat")

    def test_t1_fa06_w1_funding_crowding_penalty_and_oi_confirmation(self) -> None:
        w1_mod = fixtures.import_w1_mod()
        assets = fixtures.import_assets()
        models = fixtures.import_df_models()
        prov_real, _, _ = fixtures.import_provenance()

        n_bars = 130
        shared_closes = fixtures.build_trending_closes(n_bars, start_price=100.0, step_ret=0.004, noise_std=0.001, seed=88)
        # Identical momentum on BTC (crowded funding +150% APR) vs ETH (negative funding -20% APR)
        crowded_rate = 1.50 / 8760.0
        neg_rate = -0.20 / 8760.0
        bars_by_sym = {
            "BTC": fixtures.build_bar_series(
                symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=shared_closes,
                carry_rates=[crowded_rate * 12.0] * n_bars, carry_interval_h=12.0,
            ),
            "ETH": fixtures.build_bar_series(
                symbol="ETH", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=shared_closes,
                carry_rates=[neg_rate * 12.0] * n_bars, carry_interval_h=12.0,
            ),
            "SOL": fixtures.build_bar_series(
                symbol="SOL", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=shared_closes,
                carry_rates=[neg_rate * 12.0] * n_bars, carry_interval_h=12.0,
            ),
            "AVAX": fixtures.build_bar_series(
                symbol="AVAX", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=shared_closes,
                carry_rates=[neg_rate * 12.0] * n_bars, carry_interval_h=12.0,
            ),
        }
        # Build real Web3DerivativesMetric OI trajectories matching +5% (BTC/ETH), +30% (SOL), -20% (AVAX) 7d change
        oi_slopes = {"BTC": 0.05 / 14.0, "ETH": 0.05 / 14.0, "SOL": 0.30 / 14.0, "AVAX": -0.20 / 14.0}
        metrics_by_step: list[dict[str, Any]] = []
        for i in range(n_bars):
            step_m: dict[str, Any] = {}
            for sym in bars_by_sym:
                b = bars_by_sym[sym][i]
                oi_val = 100_000_000.0 * max(0.2, 1.0 + oi_slopes[sym] * max(0, i - (n_bars - 20)))
                step_m[sym] = models.Web3DerivativesMetric(
                    venue="binance_um",
                    symbol=sym,
                    ts_event_ms=b.ts_event_ms,
                    ts_avail_ms=b.ts_avail_ms,
                    open_interest_usd=oi_val,
                    top_trader_lsr=1.1,
                    account_lsr=1.0,
                    taker_ls_vol_ratio=1.05,
                    provenance=prov_real,
                )
            metrics_by_step.append(step_m)

        w1 = fixtures.build_strategy_instance(w1_mod.Web3XSMomCarryKernel)
        w1.reset()
        last_dec: dict[str, Any] = {}
        for i in range(n_bars):
            ts_i = bars_by_sym["BTC"][i].ts_event_ms
            ctx = {
                "oi_change_7d": {"BTC": 0.05, "ETH": 0.05, "SOL": 0.30, "AVAX": -0.20},
                "web3_metrics": metrics_by_step[i],
                "derivatives_metrics": metrics_by_step[i],
                "regime_label": "EXPANSION",
                "exposure_scalar": 1.0,
            }
            last_dec = w1.on_bar_step(ts_i, {s: bars_by_sym[s][i] for s in bars_by_sym}, ctx)
            w1.sync_executed_weights({s: d.target_weight for s, d in last_dec.items()})

        self.assertGreater(
            last_dec["ETH"].raw_signal,
            last_dec["BTC"].raw_signal,
            "Negative-funding coin must outscore extreme crowded-long funding coin with identical momentum",
        )
        self.assertGreater(
            last_dec["SOL"].raw_signal,
            last_dec["AVAX"].raw_signal,
            "Positive +30% OI confirmation must outscore -20% OI contraction with identical momentum/funding",
        )

    def test_t1_fa06_w2_adaptive_trend_breakout_directional_and_g7_parity(self) -> None:
        w2_mod = fixtures.import_w2_mod()
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()

        spec = assets.default_web3_specs(("BTC",))["BTC"]
        closes = [100.0] * 60 + [100.0 + 1.5 * i for i in range(1, 61)]
        bars = fixtures.build_bar_series(
            symbol="BTC",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=closes,
            carry_rates=[0.00005] * len(closes),
            carry_interval_h=12.0,
        )
        w2 = fixtures.build_strategy_instance(w2_mod.Web3AdaptiveTrendKernel)
        self.assertEqual(w2.asset_domain, "web3")
        w2.reset()
        w_bar59_flat = 0.0
        last_dec = None
        for idx, b in enumerate(bars):
            last_dec = w2.on_bar_step(b.ts_event_ms, {"BTC": b}, {"breadth_pct_above_20d": 0.80})
            w_cur = float(last_dec["BTC"].target_weight)
            if idx == 59:
                w_bar59_flat = w_cur
            w2.sync_executed_weights({"BTC": w_cur})
        self.assertIsNotNone(last_dec)
        w_breakout = float(last_dec["BTC"].target_weight)
        self.assertAlmostEqual(
            w_bar59_flat, 0.0, delta=1e-6,
            msg="Anti-stub (C-02): W2 must remain flat on pure range-bound prefix before breakout",
        )
        self.assertGreater(w_breakout, 0.0, "Strong upward breakout must yield positive long weight")
        self.assertGreater(w_breakout, abs(w_bar59_flat) + 0.05, "Breakout weight must materially exceed flat prefix")

        cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=20.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
        max_w_delta, max_eq_delta = fixtures.run_streaming_vs_batch_check(
            w2, {"BTC": spec}, {"BTC": bars}, risk_cfg=cfg, require_active=True
        )
        self.assertLessEqual(max_w_delta, 1e-10)
        self.assertLessEqual(max_eq_delta, 1e-8)

    def test_t1_fa06_w3_regime_carry_enters_above_cost_floor_with_separated_margin(self) -> None:
        w3_mod = fixtures.import_w3_mod()
        assets = fixtures.import_assets()

        n_bars = 80
        # High positive funding spread (~75% APR -> >> 3x RT fee floor over horizon; exact carry_interval_h=12.0)
        high_funding_12h = (0.75 / 8760.0) * 12.0
        rich_bars = fixtures.build_bar_series(
            symbol="BTC",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=[50000.0 + 10.0 * math.sin(i / 4.0) for i in range(n_bars)],
            carry_rates=[high_funding_12h] * n_bars,
            carry_interval_h=12.0,
            premium_basis_bps=25.0,
        )
        zero_bars = fixtures.build_bar_series(
            symbol="BTC",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=[50000.0 + 10.0 * math.sin(i / 4.0) for i in range(n_bars)],
            carry_rates=[0.0] * n_bars,
            carry_interval_h=12.0,
            premium_basis_bps=0.0,
        )

        def _run_w3(series: list[Any]) -> Any:
            w3 = fixtures.build_strategy_instance(w3_mod.Web3RegimeCarryKernel)
            self.assertEqual(w3.asset_domain, "web3")
            w3.reset()
            dec = None
            for b in series:
                dec = w3.on_bar_step(
                    b.ts_event_ms,
                    {"BTC": b},
                    {"dvol": 42.0, "rt_cost_bps": 14.0, "margin_ratio": 0.90},
                )
                w3.sync_executed_weights({"BTC": dec["BTC"].target_weight})
            return dec

        dec_rich = _run_w3(rich_bars)
        dec_zero = _run_w3(zero_bars)
        self.assertIsNotNone(dec_rich)
        self.assertIsNotNone(dec_zero)
        self.assertGreater(abs(float(dec_rich["BTC"].target_weight)), 0.0, "Profitable spread >= 3x RT fee must enter")
        self.assertAlmostEqual(
            float(dec_zero["BTC"].target_weight), 0.0, places=8,
            msg="Anti-stub (C-02): zero-carry zero-basis regime must NOT enter W3 carry position",
        )
        meta = dec_rich["BTC"].metadata
        carry_margin_vals = [
            v for k, v in meta.items() if ("margin" in str(k).lower() or "carry" in str(k).lower())
        ]
        self.assertGreaterEqual(
            len(carry_margin_vals), 1,
            f"W3 metadata must report separated carry/margin tracking, got {meta}",
        )
        num_meta = [float(v) for v in carry_margin_vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        self.assertTrue(all(math.isfinite(x) for x in num_meta))

    def test_t1_fa06_web3_suite_protocol_compliance_and_domain_identity(self) -> None:
        w1 = fixtures.build_strategy_instance(fixtures.import_w1_mod().Web3XSMomCarryKernel)
        w2 = fixtures.build_strategy_instance(fixtures.import_w2_mod().Web3AdaptiveTrendKernel)
        w3 = fixtures.build_strategy_instance(fixtures.import_w3_mod().Web3RegimeCarryKernel)
        for strat in (w1, w2, w3):
            self.assertEqual(strat.asset_domain, "web3")
            self.assertIsInstance(strat.strategy_id, str)
            self.assertTrue(len(strat.strategy_id.strip()) > 0)
            self.assertGreaterEqual(int(strat.warmup_bars), 0)
            for method in ("reset", "on_bar_step", "sync_executed_weights", "clone_fresh"):
                self.assertTrue(callable(getattr(strat, method, None)), f"{strat} missing {method}")
            clone = strat.clone_fresh()
            self.assertIsNot(clone, strat, f"{strat} clone_fresh() returned self")
            self.assertEqual(clone.strategy_id, strat.strategy_id)
            strat.reset()


class TestTier1FA07GoldMacroAndG1(unittest.TestCase):
    """FA-07: Gold Macro Regime Engine & Single-Instrument TradFi Core G1 (F-16, F-17)."""

    def test_t1_fa07_gold_macro_engine_computes_pit_regime_from_seven_drivers(self) -> None:
        gm_mod = fixtures.import_gold_macro_mod()
        bars_map, yields = fixtures.build_gold_seven_driver_fixture(
            n_days=150, real_10y_start=2.6, real_10y_end=1.3, dxy_step=-0.0015, ief_step=0.0015, gold_step=0.002
        )
        engine = fixtures.build_strategy_instance(
            gm_mod.GoldMacroEngine,
            macro_bars_by_symbol=bars_map,
            bars_by_symbol=bars_map,
            yield_curve=yields,
            yield_points=yields,
        )
        ts_q = bars_map["GC=F"][120].ts_avail_ms
        snap_dict = fixtures.normalize_snapshot(engine.snapshot_as_of(ts_q))
        self.assertGreater(len(snap_dict), 0)
        numeric_vals = [
            float(v) for v in snap_dict.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        self.assertGreaterEqual(len(numeric_vals), 3)
        self.assertTrue(all(math.isfinite(v) for v in numeric_vals))

        # PIT tail-mutation invariance: 5x shock strictly AFTER index 120 must NOT alter snapshot_as_of(ts_q)
        mut_bars = {}
        for sym, blist in bars_map.items():
            mut_list = []
            for idx, b in enumerate(blist):
                if idx > 120:
                    mut_list.append(
                        fixtures.make_bar(
                            venue=b.venue, symbol=b.symbol, asset_class=b.asset_class, timeframe_h=b.timeframe_h,
                            ts_event_ms=b.ts_event_ms, ts_avail_ms=b.ts_avail_ms,
                            open=b.open * 5.0, high=b.high * 5.0, low=b.low * 5.0, close=b.close * 5.0,
                            volume_usd=b.volume_usd, contract_multiplier=b.contract_multiplier,
                        )
                    )
                else:
                    mut_list.append(b)
            mut_bars[sym] = mut_list
        eng_mut = fixtures.build_strategy_instance(
            gm_mod.GoldMacroEngine,
            macro_bars_by_symbol=mut_bars,
            bars_by_symbol=mut_bars,
            yield_curve=yields,
            yield_points=yields,
        )
        self.assertEqual(
            fixtures.normalize_snapshot(eng_mut.snapshot_as_of(ts_q)),
            snap_dict,
            "GoldMacroEngine leaked future tail bars into PIT snapshot_as_of(ts_q)",
        )

        # Anti-stub (Stub R): bearish rising-real-yield / rising-DXY macro must score strictly lower than bull
        bear_bars, bear_y = fixtures.build_gold_seven_driver_fixture(
            n_days=150, real_10y_start=1.3, real_10y_end=2.8, dxy_step=0.002, ief_step=-0.002, gold_step=-0.002
        )
        eng_bear = fixtures.build_strategy_instance(
            gm_mod.GoldMacroEngine,
            macro_bars_by_symbol=bear_bars,
            bars_by_symbol=bear_bars,
            yield_curve=bear_y,
            yield_points=bear_y,
        )
        bear_dict = fixtures.normalize_snapshot(eng_bear.snapshot_as_of(bear_bars["GC=F"][120].ts_avail_ms))
        score_bull = float(fixtures.get_metric(snap_dict, "macro_gold_composite", "composite_score", "gold_macro_score", "score"))
        score_bear = float(fixtures.get_metric(bear_dict, "macro_gold_composite", "composite_score", "gold_macro_score", "score"))
        self.assertGreater(score_bull, score_bear, "Anti-stub: bullish macro must outscore bearish macro")

    def test_t1_fa07_g1_single_instrument_gc_f_execution(self) -> None:
        g1_mod = fixtures.import_g1_mod()
        assets = fixtures.import_assets()
        gm_mod = fixtures.import_gold_macro_mod()

        def _run_gc(bars_m: dict[str, list[Any]], y_pts: list[Any]) -> float:
            macro_eng = fixtures.build_strategy_instance(
                gm_mod.GoldMacroEngine,
                macro_bars_by_symbol=bars_m,
                bars_by_symbol=bars_m,
                yield_curve=y_pts,
                yield_points=y_pts,
            )
            g1 = fixtures.build_strategy_instance(g1_mod.GoldTradFiMacroKernel, symbol="GC=F", macro_engine=macro_eng)
            self.assertEqual(g1.asset_domain, "gold")
            g1.reset()
            last_dec = None
            for b in bars_m["GC=F"]:
                ctx = {"gold_macro": macro_eng.snapshot_as_of(b.ts_avail_ms)}
                last_dec = g1.on_bar_step(b.ts_event_ms, {"GC=F": b}, ctx)
                g1.sync_executed_weights({"GC=F": last_dec["GC=F"].target_weight})
            self.assertIsNotNone(last_dec)
            self.assertIn("GC=F", last_dec)
            return float(last_dec["GC=F"].target_weight)

        bull_bars, bull_y = fixtures.build_gold_seven_driver_fixture(
            n_days=140, real_10y_start=2.6, real_10y_end=1.3, dxy_step=-0.0015, ief_step=0.0015, gold_step=0.0025
        )
        bear_bars, bear_y = fixtures.build_gold_seven_driver_fixture(
            n_days=140, real_10y_start=1.2, real_10y_end=3.0, dxy_step=0.002, ief_step=-0.002, gold_step=-0.004
        )
        w_bull = _run_gc(bull_bars, bull_y)
        w_bear = _run_gc(bear_bars, bear_y)
        self.assertGreater(w_bull, 0.0, "G1 GC=F must hold positive weight in gold bull")
        self.assertGreater(w_bull, w_bear, "Anti-stub (C-02): G1 GC=F bull weight must strictly exceed bear weight")
        self.assertEqual(assets.default_gold_specs()["GC=F"].contract_multiplier, 100.0)

    def test_t1_fa07_g1_single_instrument_gld_execution(self) -> None:
        g1_mod = fixtures.import_g1_mod()
        assets = fixtures.import_assets()
        gm_mod = fixtures.import_gold_macro_mod()

        def _run_gld(bars_m: dict[str, list[Any]], y_pts: list[Any]) -> float:
            macro_eng = fixtures.build_strategy_instance(
                gm_mod.GoldMacroEngine,
                macro_bars_by_symbol=bars_m,
                bars_by_symbol=bars_m,
                yield_curve=y_pts,
                yield_points=y_pts,
            )
            g1 = fixtures.build_strategy_instance(g1_mod.GoldTradFiMacroKernel, symbol="GLD", macro_engine=macro_eng)
            g1.reset()
            last_dec = None
            for b in bars_m["GLD"]:
                ctx = {"gold_macro": macro_eng.snapshot_as_of(b.ts_avail_ms)}
                last_dec = g1.on_bar_step(b.ts_event_ms, {"GLD": b}, ctx)
                g1.sync_executed_weights({"GLD": last_dec["GLD"].target_weight})
            self.assertIsNotNone(last_dec)
            return float(last_dec["GLD"].target_weight)

        bull_bars, bull_y = fixtures.build_gold_seven_driver_fixture(
            n_days=140, real_10y_start=2.6, real_10y_end=1.3, dxy_step=-0.0015, ief_step=0.0015, gold_step=0.0025
        )
        bear_bars, bear_y = fixtures.build_gold_seven_driver_fixture(
            n_days=140, real_10y_start=1.2, real_10y_end=3.0, dxy_step=0.002, ief_step=-0.002, gold_step=-0.004
        )
        w_gld_bull = _run_gld(bull_bars, bull_y)
        w_gld_bear = _run_gld(bear_bars, bear_y)
        self.assertGreater(w_gld_bull, 0.0, "G1 GLD must hold positive weight in gold bull")
        self.assertGreater(w_gld_bull, w_gld_bear, "Anti-stub (C-02): G1 GLD bull weight must strictly exceed bear weight")
        self.assertEqual(assets.default_gold_specs()["GLD"].contract_multiplier, 1.0)

    def test_t1_fa07_g1_asymmetric_macro_exposure_scaling_bull_vs_bear(self) -> None:
        g1_mod = fixtures.import_g1_mod()
        gm_mod = fixtures.import_gold_macro_mod()

        bull_bars, bull_y = fixtures.build_gold_seven_driver_fixture(
            n_days=140, real_10y_start=2.6, real_10y_end=1.3, dxy_step=-0.0015, ief_step=0.0015, gold_step=0.002
        )
        bear_bars, bear_y = fixtures.build_gold_seven_driver_fixture(
            n_days=140, real_10y_start=1.3, real_10y_end=2.8, dxy_step=0.0015, ief_step=-0.0015, gold_step=0.002
        )

        def _final_weight(bars_m: dict[str, list[Any]], y_pts: list[Any]) -> float:
            me = fixtures.build_strategy_instance(
                gm_mod.GoldMacroEngine, macro_bars_by_symbol=bars_m, bars_by_symbol=bars_m, yield_curve=y_pts, yield_points=y_pts
            )
            g1 = fixtures.build_strategy_instance(g1_mod.GoldTradFiMacroKernel, symbol="GC=F", macro_engine=me)
            g1.reset()
            w_last = 0.0
            for b in bars_m["GC=F"]:
                dec = g1.on_bar_step(b.ts_event_ms, {"GC=F": b}, {"gold_macro": me.snapshot_as_of(b.ts_avail_ms)})
                w_last = float(dec["GC=F"].target_weight)
                g1.sync_executed_weights({"GC=F": w_last})
            return w_last

        w_bull = _final_weight(bull_bars, bull_y)
        w_bear = _final_weight(bear_bars, bear_y)
        self.assertGreater(w_bull, 0.0)
        self.assertLessEqual(w_bull, 1.5)
        self.assertGreater(w_bull, w_bear, "Falling-real-yield Gold macro tailwind must scale long exposure above rising-real-yield headwind")

    def test_t1_fa07_g1_streaming_vs_batch_g7_bit_parity(self) -> None:
        g1_mod = fixtures.import_g1_mod()
        gm_mod = fixtures.import_gold_macro_mod()
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        bars_map, yields = fixtures.build_gold_seven_driver_fixture(n_days=120)
        me = fixtures.build_strategy_instance(
            gm_mod.GoldMacroEngine, macro_bars_by_symbol=bars_map, bars_by_symbol=bars_map, yield_curve=yields, yield_points=yields
        )
        g1 = fixtures.build_strategy_instance(g1_mod.GoldTradFiMacroKernel, symbol="GC=F", macro_engine=me)
        cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=15.0, max_portfolio_leverage=1.2, max_single_weight=1.2)
        ctx_fn = lambda ts: {"gold_macro": me.snapshot_as_of(ts)}
        res_sim = acct_mod.PortfolioAccountingEngine().simulate(
            g1.clone_fresh(), {"GC=F": assets.default_gold_specs()["GC=F"]}, {"GC=F": bars_map["GC=F"]},
            risk_cfg=cfg, context_provider=ctx_fn,
        )
        self.assertGreaterEqual(
            fixtures.count_distinct_raw_decisions(res_sim), 2,
            "Anti-stub (C-02): G1 must produce non-constant raw decisions across warmup/trend",
        )
        max_w_delta, max_eq_delta = fixtures.run_streaming_vs_batch_check(
            g1,
            {"GC=F": assets.default_gold_specs()["GC=F"]},
            {"GC=F": bars_map["GC=F"]},
            risk_cfg=cfg,
            context_provider=ctx_fn,
            require_active=True,
        )
        self.assertLessEqual(max_w_delta, 1e-10)
        self.assertLessEqual(max_eq_delta, 1e-8)


class TestTier1FA08Gold4hEventAndPAXG(unittest.TestCase):
    """FA-08: Gold 4h Tactical Event Engine G2 & PAXG Bridge G3 (F-18, F-19)."""

    def _build_4h_event_series(self) -> tuple[list[Any], Any]:
        assets = fixtures.import_assets()
        models = fixtures.import_df_models()
        prov_real, _, _ = fixtures.import_provenance()

        # 55 4h bars: Bar 48 == T-8h, Bar 49 == T-4h (inside [T-4h, T)), Bar 50 == T (DEC_2026_FOMC_MS)
        t_event = fixtures.DEC_2026_FOMC_MS
        start_ms = t_event - 50 * 4 * fixtures.HOUR_MS
        # Bars 0..48 strictly range-bound [2699, 2701] through T-8h inclusive (ZERO pre-window breakout!)
        closes = [2700.0 + 1.0 * math.sin(i / 3.0) for i in range(49)]
        # Bar 49 (T-4h): FIRST breakout inside [T-4h, T); Bars 50..54: post-event expansion [T, T+16h]
        closes.extend([2725.0, 2738.0, 2752.0, 2765.0, 2775.0, 2782.0])
        bars = fixtures.build_bar_series(
            symbol="GC=F",
            venue="yahoo",
            asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
            timeframe_h=4.0,
            start_ms=start_ms,
            closes=closes,
            contract_multiplier=100.0,
            session_tag="LONDON_NY",
        )
        event = models.MacroEventRelease(
            event_type="FOMC",
            event_id="FOMC_2026_12_09",
            ref_period="2026-12",
            ts_release_ms=t_event,
            ts_avail_ms=t_event,
            is_sep_meeting=True,
            actual=4.25,
            prior=4.50,
            surprise_z=-1.2,
            source="fed_html",
            provenance=prov_real,
        )
        return bars, event

    def test_t1_fa08_g2_policy_a_baseline_4h_donchian_breakout(self) -> None:
        g2_mod = fixtures.import_g2_mod()
        bars, event = self._build_4h_event_series()
        g2_a = fixtures.build_strategy_instance(g2_mod.Gold4hEventKernel, policy="A")
        g2_a.reset()
        max_w = 0.0
        w_bar48_flat = 0.0
        for idx, b in enumerate(bars):
            dec = g2_a.on_bar_step(
                b.ts_event_ms,
                {"GC=F": b},
                {"macro_events": [event], "gold_macro": {"macro_gold_composite": 0.8}, "gvz_z": 1.1},
            )
            w_cur = float(dec["GC=F"].target_weight)
            if idx == 48:
                w_bar48_flat = w_cur
            max_w = max(max_w, abs(w_cur))
            g2_a.sync_executed_weights({"GC=F": w_cur})
        self.assertAlmostEqual(
            w_bar48_flat, 0.0, delta=1e-6,
            msg="Anti-stub (C-02): Policy A must stay flat on strictly range-bound Bar 48 (T-8h)",
        )
        self.assertGreater(max_w, 0.0, "Policy A baseline 4h Donchian breakout must enter on strong breakout")

    def test_t1_fa08_g2_policy_b_event_whipsaw_shield_active(self) -> None:
        g2_mod = fixtures.import_g2_mod()
        bars, event = self._build_4h_event_series()
        g2_a = fixtures.build_strategy_instance(g2_mod.Gold4hEventKernel, policy="A")
        g2_b = fixtures.build_strategy_instance(g2_mod.Gold4hEventKernel, policy="B")
        g2_a.reset()
        g2_b.reset()
        w_a_pre = None
        w_b_pre = None
        max_post_b = 0.0
        for b in bars:
            ctx = {"macro_events": [event], "gold_macro": {"macro_gold_composite": 0.8}, "gvz_z": 1.1}
            da = g2_a.on_bar_step(b.ts_event_ms, {"GC=F": b}, ctx)
            db = g2_b.on_bar_step(b.ts_event_ms, {"GC=F": b}, ctx)
            wa = float(da["GC=F"].target_weight)
            wb = float(db["GC=F"].target_weight)
            if (event.ts_release_ms - 4 * fixtures.HOUR_MS) <= b.ts_event_ms < event.ts_release_ms:
                w_a_pre = wa
                w_b_pre = wb
            if b.ts_event_ms >= event.ts_release_ms:
                max_post_b = max(max_post_b, abs(wb))
            g2_a.sync_executed_weights({"GC=F": wa})
            g2_b.sync_executed_weights({"GC=F": wb})
        self.assertIsNotNone(w_a_pre)
        self.assertIsNotNone(w_b_pre)
        self.assertGreater(abs(float(w_a_pre)), 0.0, "Policy A control must enter breakout inside [T-4h, T)")
        self.assertAlmostEqual(float(w_b_pre), 0.0, places=8, msg="Policy B must block fresh entries inside [T-4h, T)")
        self.assertGreater(max_post_b, 0.0, "Anti-stub (C-02): Policy B must resume active post-event participation")

    def test_t1_fa08_g2_policy_c_post_event_expansion_and_json_signal_output(self) -> None:
        g2_mod = fixtures.import_g2_mod()
        bars, event = self._build_4h_event_series()
        g2_c = fixtures.build_strategy_instance(g2_mod.Gold4hEventKernel, policy="C")
        g2_c.reset()
        w_bar48_flat = 0.0
        max_post_c = 0.0
        post_dec = None
        for idx, b in enumerate(bars):
            dec = g2_c.on_bar_step(
                b.ts_event_ms,
                {"GC=F": b},
                {"macro_events": [event], "gold_macro": {"macro_gold_composite": 0.9}, "gvz_z": 1.6},
            )
            w_cur = float(dec["GC=F"].target_weight)
            if idx == 48:
                w_bar48_flat = w_cur
            if event.ts_release_ms <= b.ts_event_ms <= (event.ts_release_ms + 8 * fixtures.HOUR_MS):
                if abs(w_cur) >= max_post_c:
                    max_post_c = abs(w_cur)
                    post_dec = dec["GC=F"]
            g2_c.sync_executed_weights({"GC=F": w_cur})
        self.assertAlmostEqual(w_bar48_flat, 0.0, delta=1e-6, msg="Anti-stub (C-02): Policy C must stay flat on range Bar 48")
        self.assertIsNotNone(post_dec)
        self.assertGreater(max_post_c, 0.0, "Policy C must allow aligned post-event [T, T+8h] expansion")
        self.assertGreaterEqual(len(post_dec.metadata), 1, "Policy C post-event decision must expose structured metadata")
        payload_str = fixtures.strict_json_dumps(post_dec.metadata)
        self.assertIsInstance(fixtures.strict_json_loads(payload_str), dict)

    def test_t1_fa08_g3_paxg_dual_basis_decomposition_metadata(self) -> None:
        g3_mod = fixtures.import_g3_mod()
        assets = fixtures.import_assets()
        bars = fixtures.build_bar_series(
            symbol="PAXG",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_GOLD_PERP,
            timeframe_h=12.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=[2690.0] * 40,
            carry_rates=[0.00005] * 40,
            carry_interval_h=12.0,
        )
        g3 = fixtures.build_strategy_instance(g3_mod.GoldPAXGBridgeKernel)
        g3.reset()
        last_dec = None
        for b in bars:
            ctx = {"paxg_spot_price": 2685.0, "comex_price": 2700.0, "gc_f_price": 2700.0}
            last_dec = g3.on_bar_step(b.ts_event_ms, {"PAXG": b}, ctx)
            g3.sync_executed_weights({"PAXG": last_dec["PAXG"].target_weight})
        self.assertIsNotNone(last_dec)
        meta = last_dec["PAXG"].metadata
        perp_spot = float(fixtures.get_metric(meta, "perp_spot_basis_bps", "paxg_perp_spot_basis_bps"))
        spot_comex = float(fixtures.get_metric(meta, "spot_comex_basis_bps", "paxg_spot_comex_basis_bps"))
        self.assertAlmostEqual(perp_spot, (2690.0 / 2685.0 - 1.0) * 10_000.0, delta=1.5)
        self.assertAlmostEqual(spot_comex, (2685.0 / 2700.0 - 1.0) * 10_000.0, delta=1.5)

        # Second distinct price tuple prevents hardcoded constant basis metadata
        next_ts = bars[-1].ts_event_ms + 12 * fixtures.HOUR_MS
        b2 = fixtures.make_bar(
            venue="hyperliquid", symbol="PAXG", asset_class=assets.ASSET_CLASS_GOLD_PERP, timeframe_h=12.0,
            ts_event_ms=next_ts, ts_avail_ms=next_ts + 12 * fixtures.HOUR_MS,
            open=2670.0, high=2675.0, low=2665.0, close=2670.0,
            volume_usd=1e7, carry_rate_period=0.00005, carry_interval_h=12.0,
        )
        dec2 = g3.on_bar_step(next_ts, {"PAXG": b2}, {"paxg_spot_price": 2680.0, "comex_price": 2660.0, "gc_f_price": 2660.0})
        meta2 = dec2["PAXG"].metadata
        perp_spot_2 = float(fixtures.get_metric(meta2, "perp_spot_basis_bps", "paxg_perp_spot_basis_bps"))
        spot_comex_2 = float(fixtures.get_metric(meta2, "spot_comex_basis_bps", "paxg_spot_comex_basis_bps"))
        self.assertAlmostEqual(perp_spot_2, (2670.0 / 2680.0 - 1.0) * 10_000.0, delta=1.5)
        self.assertAlmostEqual(spot_comex_2, (2680.0 / 2660.0 - 1.0) * 10_000.0, delta=1.5)

    def test_t1_fa08_g2_and_g3_streaming_vs_batch_g7_parity(self) -> None:
        g2_mod = fixtures.import_g2_mod()
        g3_mod = fixtures.import_g3_mod()
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        bars_4h, event = self._build_4h_event_series()
        cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=15.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
        g2 = fixtures.build_strategy_instance(g2_mod.Gold4hEventKernel, policy="B")
        g2_ctx = lambda ts: {"macro_events": [event], "gold_macro": {"macro_gold_composite": 0.6}}
        res_g2 = acct_mod.PortfolioAccountingEngine().simulate(
            g2.clone_fresh(), {"GC=F": assets.default_gold_specs()["GC=F"]}, {"GC=F": bars_4h},
            risk_cfg=cfg, context_provider=g2_ctx,
        )
        self.assertGreaterEqual(fixtures.count_distinct_raw_decisions(res_g2), 2, "Anti-stub: G2 decisions must vary")
        w_d2, eq_d2 = fixtures.run_streaming_vs_batch_check(
            g2,
            {"GC=F": assets.default_gold_specs()["GC=F"]},
            {"GC=F": bars_4h},
            risk_cfg=cfg,
            context_provider=g2_ctx,
            require_active=True,
        )
        self.assertLessEqual(w_d2, 1e-10)
        self.assertLessEqual(eq_d2, 1e-8)

        paxg_bars = fixtures.build_bar_series(
            symbol="PAXG",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_GOLD_PERP,
            timeframe_h=12.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=fixtures.build_trending_closes(60, start_price=2650.0, step_ret=0.002, seed=91),
            carry_rates=[0.00005] * 60,
            carry_interval_h=12.0,
        )
        close_by_ts = {int(b.ts_event_ms): float(b.close) for b in paxg_bars}
        g3_ctx = lambda ts: {
            "paxg_spot_price": close_by_ts.get(int(ts), 2650.0),
            "comex_price": close_by_ts.get(int(ts), 2650.0),
            "gold_macro": {"macro_gold_composite": 0.8},
        }
        g3 = fixtures.build_strategy_instance(g3_mod.GoldPAXGBridgeKernel)
        res_g3 = acct_mod.PortfolioAccountingEngine().simulate(
            g3.clone_fresh(), {"PAXG": assets.default_gold_specs()["PAXG"]}, {"PAXG": paxg_bars},
            risk_cfg=cfg, context_provider=g3_ctx,
        )
        self.assertGreaterEqual(fixtures.count_distinct_raw_decisions(res_g3), 2, "Anti-stub: G3 decisions must vary")
        w_d3, eq_d3 = fixtures.run_streaming_vs_batch_check(
            g3,
            {"PAXG": assets.default_gold_specs()["PAXG"]},
            {"PAXG": paxg_bars},
            risk_cfg=cfg,
            context_provider=g3_ctx,
            require_active=True,
        )
        self.assertLessEqual(w_d3, 1e-10)
        self.assertLessEqual(eq_d3, 1e-8)

    def test_t1_fa08_g2_and_g3_protocol_compliance_gold_domain(self) -> None:
        g2 = fixtures.build_strategy_instance(fixtures.import_g2_mod().Gold4hEventKernel, policy="A")
        g3 = fixtures.build_strategy_instance(fixtures.import_g3_mod().GoldPAXGBridgeKernel)
        for strat in (g2, g3):
            self.assertEqual(strat.asset_domain, "gold")
            self.assertIsInstance(strat.strategy_id, str)
            self.assertTrue(len(strat.strategy_id.strip()) > 0)
            self.assertGreaterEqual(int(strat.warmup_bars), 0)
            for method in ("reset", "on_bar_step", "sync_executed_weights", "clone_fresh"):
                self.assertTrue(callable(getattr(strat, method, None)))
            clone = strat.clone_fresh()
            self.assertIsNot(clone, strat)
            self.assertEqual(clone.strategy_id, strat.strategy_id)


class TestTier1FA09AShareAndCombiner(unittest.TestCase):
    """FA-09: Isolated A-Share Contract & Return-Stream PortfolioCombiner (F-11, F-20)."""

    def _make_sleeve_result(self, name: str, returns: list[float], capital_usd: float = 10_000.0) -> Any:
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        closes = fixtures.closes_from_returns(100.0, returns)
        opens = [closes[0]] + [closes[i - 1] for i in range(1, len(closes))]
        spec = assets.AssetSpec(
            symbol=name,
            asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
            venue="yahoo",
            calendar=assets.CALENDAR_US_DAILY,
            cost_model=assets.CostModel(0.0, 0.0, 0.0),
            max_leverage_cap=1.0,
        )
        bars = fixtures.build_bar_series(
            symbol=name,
            venue="yahoo",
            asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
            timeframe_h=24.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=closes,
            opens=opens,
            highs=[max(o, c) for o, c in zip(opens, closes)],
            lows=[min(o, c) for o, c in zip(opens, closes)],
        )
        strat = fixtures.build_scripted_strategy([{name: 1.0} for _ in closes], strategy_id=f"sleeve_{name}", asset_domain="gold")
        cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False)
        return acct_mod.PortfolioAccountingEngine().simulate(strat, {name: spec}, {name: bars}, risk_cfg=cfg, capital_usd=capital_usd)

    def test_t1_fa09_ashare_interface_long_only_protocol_and_hook_slots(self) -> None:
        mod = fixtures.import_ashare_contract_mod()
        cls = mod.AShareStrategyInterface
        self.assertTrue(hasattr(cls, "on_bar_step"))
        self.assertTrue(hasattr(cls, "reset"))
        self.assertTrue(hasattr(cls, "sync_executed_weights"))
        self.assertTrue(hasattr(cls, "clone_fresh"))

        inst = fixtures.instantiate_ashare_interface(cls, default_signal=1.0)
        self.assertEqual(inst.asset_domain, "ashare")
        clone = inst.clone_fresh()
        self.assertIsNot(clone, inst)
        # Verify negative signal clamps long-only target_weight >= 0.0
        bars = fixtures.build_ashare_multiday_1h_bars(symbol="600519.SH", n_days=1, base_price=100.0)
        inst.reset()
        d_neg = inst.on_bar_step(
            bars[0].ts_event_ms,
            {"600519.SH": bars[0]},
            {"raw_signal": -5.0, "desired_weight": -0.8, "signal": -5.0},
        )
        self.assertGreaterEqual(float(d_neg["600519.SH"].target_weight), 0.0, "AShareStrategyInterface must clamp long-only >= 0")

        # Verify auction / OFI / theme extension hook slots exist
        member_names = " ".join(dir(inst)).lower()
        self.assertTrue(
            any(k in member_names for k in ("auction", "ofi", "theme")),
            f"AShareStrategyInterface must expose auction/OFI/theme hook slots; got {dir(inst)}",
        )

    def test_t1_fa09_portfolio_combiner_causal_inv_vol_assigns_higher_weight_to_lower_vol(self) -> None:
        comb_mod = fixtures.import_portfolio_combiner_mod()
        r_low = fixtures.generate_correlated_return_matrix(1, 180, 0.0, annual_vol=0.10, periods_per_year=252.0, drift_annual=0.12, seed=101)[0]
        r_high = fixtures.generate_correlated_return_matrix(1, 180, 0.0, annual_vol=0.30, periods_per_year=252.0, drift_annual=0.12, seed=102)[0]
        s_low = self._make_sleeve_result("LOWVOL", r_low)
        s_high = self._make_sleeve_result("HIGHVOL", r_high)

        combiner = comb_mod.PortfolioCombiner()
        combined = combiner.combine_sleeves({"low": s_low, "high": s_high}, method="causal_inv_vol", target_vol_pct=15.0)
        weights = fixtures.get_metric(combined, "sleeve_weights_final", "sleeve_weights", "final_weights", "sleeve_capital_allocations", "capital_allocation")
        self.assertGreater(float(weights["low"]), float(weights["high"]))

    def test_t1_fa09_portfolio_combiner_correlation_matrix_and_diversification_ratio(self) -> None:
        comb_mod = fixtures.import_portfolio_combiner_mod()
        mat = fixtures.generate_correlated_return_matrix(2, 200, 0.0, annual_vol=0.18, periods_per_year=252.0, drift_annual=0.15, seed=103)
        s1 = self._make_sleeve_result("SLEEVE_W", mat[0])
        s2 = self._make_sleeve_result("SLEEVE_G", mat[1])

        combined = comb_mod.PortfolioCombiner().combine_sleeves({"web3": s1, "gold": s2}, method="causal_inv_vol", target_vol_pct=15.0)
        corr = fixtures.get_metric(combined, "pairwise_corr_matrix", "correlation_matrix", "pairwise_correlation")
        dr = float(fixtures.get_metric(combined, "diversification_ratio"))
        self.assertGreater(dr, 1.05, "Uncorrelated sleeves must achieve diversification_ratio > 1.0")
        self.assertAlmostEqual(float(corr["web3"]["web3"]), 1.0, places=6)
        self.assertAlmostEqual(float(corr["gold"]["gold"]), 1.0, places=6)
        self.assertAlmostEqual(float(corr["web3"]["gold"]), float(corr["gold"]["web3"]), places=6)
        self.assertLessEqual(abs(float(corr["web3"]["gold"])), 1.0 + 1e-9)

    def test_t1_fa09_portfolio_combiner_explicit_rea001_capital_and_margin_split(self) -> None:
        comb_mod = fixtures.import_portfolio_combiner_mod()
        mat = fixtures.generate_correlated_return_matrix(2, 160, 0.05, annual_vol=0.15, periods_per_year=252.0, drift_annual=0.10, seed=104)
        s1 = self._make_sleeve_result("W_TREND", mat[0], capital_usd=10_000.0)
        s2 = self._make_sleeve_result("G_MACRO", mat[1], capital_usd=10_000.0)
        combined = comb_mod.PortfolioCombiner().combine_sleeves({"w_trend": s1, "g_macro": s2})
        cagr = float(fixtures.get_metric(combined, "cagr_pct"))
        sr = float(fixtures.get_metric(combined, "sharpe_ratio"))
        dd = float(fixtures.get_metric(combined, "max_drawdown_pct"))
        alloc = fixtures.get_metric(combined, "sleeve_capital_allocations", "capital_allocation", "sleeve_capital_allocation", "margin_allocation", "sleeve_weights_final", "sleeve_weights")
        self.assertTrue(math.isfinite(cagr) and math.isfinite(sr) and math.isfinite(dd))
        self.assertIn("w_trend", alloc)
        self.assertIn("g_macro", alloc)
        total_alloc = sum(
            float(v)
            if isinstance(v, (int, float))
            else float(fixtures.get_metric(v, "allocated_capital_usd", "capital_usd", "allocated_weight", "weight", "margin_usd"))
            for v in alloc.values()
        )
        self.assertGreater(total_alloc, 0.0)

    def test_t1_fa09_portfolio_combiner_does_not_mutate_input_sleeve_results(self) -> None:
        comb_mod = fixtures.import_portfolio_combiner_mod()
        mat = fixtures.generate_correlated_return_matrix(2, 120, 0.1, annual_vol=0.15, periods_per_year=252.0, seed=105)
        s1 = self._make_sleeve_result("A", mat[0])
        s2 = self._make_sleeve_result("B", mat[1])
        before_1 = list(fixtures.get_metric(s1, "period_returns"))
        before_2 = list(fixtures.get_metric(s2, "period_returns"))
        comb_mod.PortfolioCombiner().combine_sleeves({"a": s1, "b": s2})
        self.assertEqual(list(fixtures.get_metric(s1, "period_returns")), before_1)
        self.assertEqual(list(fixtures.get_metric(s2, "period_returns")), before_2)


class TestTier1FA10RunnerAndResearchStack(unittest.TestCase):
    """FA-10: Integrated Platform v4 CLI Runner, Report & Research Stack (F-22)."""

    def test_t1_fa10_run_strategy_platform_v4_cli_exists_and_exposes_help(self) -> None:
        self.assertTrue(fixtures.V4_RUNNER_PATH.is_file(), f"Missing v4 runner: {fixtures.V4_RUNNER_PATH}")
        proc = subprocess.run(
            [sys.executable, str(fixtures.V4_RUNNER_PATH), "--help"],
            cwd=str(fixtures.AUTO_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"v4 runner --help failed: {proc.stderr}")
        self.assertIn("usage", proc.stdout.lower())

    def test_t1_fa10_requirements_research_txt_exists_with_pinned_accelerators(self) -> None:
        req = fixtures.REQUIREMENTS_RESEARCH_PATH
        self.assertTrue(req.is_file(), f"Missing {req}")
        lines = [
            ln.strip()
            for ln in req.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        self.assertGreaterEqual(len(lines), 4, "requirements-research.txt must list pinned Research Plane packages")
        joined = " ".join(lines).lower()
        for expected_pkg in ("numpy", "pyarrow", "duckdb"):
            self.assertIn(expected_pkg, joined, f"Expected MIT/Apache accelerator {expected_pkg} in requirements-research.txt")

    def test_t1_fa10_platform_package_exports_v3_compat_and_v4_namespaces(self) -> None:
        import importlib
        pkg = importlib.import_module("alphaloop.platform")
        for sym in ("AssetSpec", "CanonicalBar", "MultiAssetDataLake", "DataFoundationRouter", "PortfolioAccountingEngine", "PortfolioCombiner"):
            self.assertTrue(hasattr(pkg, sym), f"alphaloop.platform missing export {sym}")

    def test_t1_fa10_core_v4_modules_import_and_execute_on_pure_python_stdlib(self) -> None:
        probe = (
            "import sys\n"
            "BLOCKED = {'numpy','scipy','pyarrow','duckdb','polars','pandas','pytest'}\n"
            "class Blocker:\n"
            "  def find_spec(self, fullname, path, target=None):\n"
            "    if fullname.split('.')[0] in BLOCKED:\n"
            "      raise ImportError(f'Blocked optional third-party: {fullname}')\n"
            "    return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "import alphaloop.platform\n"
            "import alphaloop.platform.data_foundation.models\n"
            "import alphaloop.platform.data_foundation.router\n"
            "import alphaloop.platform.core_engine.accounting\n"
            "import alphaloop.platform.core_engine.risk_allocator\n"
            "import alphaloop.platform.core_engine.validation\n"
            "import alphaloop.platform.core_engine.portfolio_combiner\n"
            "print('STDLIB_OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(fixtures.AUTO_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"Pure stdlib import failed: {proc.stderr}")
        self.assertIn("STDLIB_OK", proc.stdout)

    def test_t1_fa10_v4_runner_report_serialization_is_strict_json_compatible(self) -> None:
        self.assertTrue(fixtures.V4_RUNNER_PATH.is_file())
        src = fixtures.V4_RUNNER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(fixtures.V4_RUNNER_PATH))
        fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertIn("main", fn_names, "run_strategy_platform_v4.py must define main()")
        has_allow_nan_false = False
        has_strict_json_helper = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "allow_nan" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                        has_allow_nan_false = True
                callee = ""
                if isinstance(node.func, ast.Name):
                    callee = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    callee = node.func.attr
                if "strict_json" in callee.lower():
                    has_strict_json_helper = True
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "strict_json" in node.name.lower():
                has_strict_json_helper = True
        self.assertTrue(
            has_allow_nan_false or has_strict_json_helper,
            "v4 runner JSON output must enforce allow_nan=False (AST keyword) or a strict_json helper",
        )


class TestTier1FA11ComplianceAndBoundaries(unittest.TestCase):
    """FA-11: STD-34 Compliance, Governance Deliverables & Boundary Integrity (F-21, F-23)."""

    def test_t1_fa11_audit_service_readiness_script_reachable_and_strict_capable(self) -> None:
        self.assertTrue(fixtures.AUDIT_SCRIPT_PATH.is_file())
        rc, payload, stderr = fixtures.run_audit_script(
            fixtures.E2E_SUITE_DIR, strict=True, include_tests=True
        )
        self.assertEqual(rc, 0, f"STD-34 audit on E2E suite failed: {payload} {stderr}")
        self.assertGreaterEqual(int(payload.get("files_scanned", 0)), 9)
        self.assertEqual(payload.get("findings"), [])

    def test_t1_fa11_execution_safety_locks_remain_hard_false(self) -> None:
        import importlib
        bridge = importlib.import_module("alphaloop.execution.bridge")
        adapter = importlib.import_module("alphaloop.execution.arb_executor_adapter")
        daemon = importlib.import_module("alphaloop.daemon.shadow_runner")

        self.assertIs(bridge.LIVE_EXECUTION_ENABLED, False)
        self.assertIs(adapter.ADAPTER_LIVE_EXECUTION_ENABLED, False)
        self.assertIs(daemon.DAEMON_LIVE_EXECUTION_ENABLED, False)
        self.assertEqual(bridge.ALLOWED_SHADOW_STATUSES, frozenset({"SHADOW_LIVE", "PAPER_LIVE"}))
        for mod in (bridge, adapter, daemon):
            self.assertIs(getattr(mod, "DRY_RUN_ONLY", True), True)

    def test_t1_fa11_platform_zero_imports_from_do_not_touch_execution_or_plugins(self) -> None:
        forbidden_targets = (
            "alphaloop.execution",
            "alphaloop.daemon",
            "alphaloop.plugins.funding_carry_plugin",
            "alphaloop.plugins.trend_cta_plugin",
            "run_shadow_daemon",
            "research_carry_regime_switch",
        )
        platform_dir = fixtures.AUTO_REPO_ROOT / "alphaloop" / "platform"
        py_files = sorted(platform_dir.rglob("*.py"))
        if fixtures.V4_RUNNER_PATH.is_file():
            py_files.append(fixtures.V4_RUNNER_PATH)
        self.assertGreater(len(py_files), 0)
        violations: list[str] = []
        for path in py_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(alias.name.startswith(f) or f in alias.name for f in forbidden_targets):
                            violations.append(f"{path}:{node.lineno} imports {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if any(node.module.startswith(f) or f in node.module for f in forbidden_targets):
                        violations.append(f"{path}:{node.lineno} imports {node.module}")
                    for alias in node.names:
                        full = f"{node.module}.{alias.name}"
                        if any(full.startswith(f) or f in full for f in forbidden_targets):
                            violations.append(f"{path}:{node.lineno} imports {full}")
        self.assertEqual(violations, [], f"Forbidden DO_NOT_TOUCH imports found: {violations}")

    def test_t1_fa11_strategy_domains_strictly_decoupled_at_ast_level(self) -> None:
        strat_root = fixtures.AUTO_REPO_ROOT / "alphaloop" / "platform" / "strategies"
        self.assertTrue(strat_root.is_dir(), f"Missing decoupled strategies package: {strat_root}")

        def _check_file_imports(path: Path, forbidden_segments: frozenset[str]) -> list[str]:
            hits: list[str] = []
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if set(alias.name.split(".")) & forbidden_segments:
                            hits.append(f"{path}:{node.lineno} -> import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod_segs = node.module.split(".") if node.module else []
                    if set(mod_segs) & forbidden_segments:
                        hits.append(f"{path}:{node.lineno} -> from {'.' * node.level}{node.module or ''}")
                        continue
                    is_parent_or_strategies = (
                        node.level >= 2
                        or (node.level >= 1 and not mod_segs)
                        or (bool(mod_segs) and mod_segs[-1] == "strategies")
                    )
                    if is_parent_or_strategies:
                        for alias in node.names:
                            if alias.name in forbidden_segments:
                                hits.append(
                                    f"{path}:{node.lineno} -> from {'.' * node.level}{node.module or ''} import {alias.name}"
                                )
            return hits

        violations: list[str] = []
        for dom_name, forbidden_segs in (
            ("web3", frozenset({"gold", "ashare_contract"})),
            ("gold", frozenset({"web3", "ashare_contract"})),
        ):
            dom_dir = strat_root / dom_name
            self.assertTrue(dom_dir.is_dir(), f"Missing domain strategy directory: {dom_dir}")
            for path in sorted(dom_dir.rglob("*.py")):
                violations.extend(_check_file_imports(path, forbidden_segs))

        ashare_file = strat_root / "ashare_contract.py"
        self.assertTrue(ashare_file.is_file(), f"Missing {ashare_file}")
        violations.extend(_check_file_imports(ashare_file, frozenset({"web3", "gold"})))
        self.assertEqual(violations, [], f"Cross-domain strategy coupling detected (violates V-02): {violations}")

    def test_t1_fa11_zero_agents_teamwork_metadata_inside_active_git_repos(self) -> None:
        for repo in (fixtures.AUTO_REPO_ROOT, fixtures.SCHQ_REPO_ROOT):
            self.assertFalse((repo / ".agents" / "teamwork").exists(), f".agents/teamwork leaked in {repo}")
            proc = subprocess.run(
                ["git", "-C", str(repo), "ls-files", ".agents/teamwork"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout.strip(), "", f"Tracked .agents/teamwork in {repo}")

    def test_t1_fa11_schq_v4_deliverable_parent_dirs_and_ticket_present(self) -> None:
        for key in ("ticket", "architecture", "empirical_report"):
            path = fixtures.SCHQ_DELIVERABLES[key]
            self.assertTrue(path.is_file(), f"Missing SCHQ deliverable ({key}): {path}")
            self.assertGreater(path.stat().st_size, 200, f"SCHQ deliverable ({key}) unexpectedly empty: {path}")


if __name__ == "__main__":
    unittest.main()
