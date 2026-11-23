"""Comprehensive Unit & Adversarial Tests for Phase 2 Modules (A, B, C, D, E).

Tests strictly verify:
  - Module A: Causal 7d vol-tiered leverage, fail-safe thin history, protective margin exit precedence
              (>=0.90 STILL liquidates; [0.55, 0.90) storm-priced TAKER_TAKER + >=5bp/leg + 24h cooldown).
  - Module B: TrendCTAPlugin incomplete-bucket drop, T+1 open causality, prior-bar Chandelier stop,
              gap-through-stop fill, real perp funding carry/drag, and DQ-04 >=98% funding overlap gate.
  - Module C: PortfolioCombiner UTC daily alignment, gap fail-closed, weight conservation, causal_inv_vol,
              rebalance turnover cost, correlation diversification, and depth-anchored capacity drag.
  - Module D: ExecutionBridge SHADOW_DRY_RUN enforcement, fail-closed BBO/notional/status/identity blocks,
              deterministic idempotency, hash-chain tamper detection, KillSwitch HUMAN_RESET token, and
              HotConfigWatcher auto-trip on corrupt JSON.
  - Module E: Web Console v2.1 endpoints (/api/portfolio, /api/execution/plans, 4 plugins, zero onclick).
"""

from __future__ import annotations

import json
import math
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from alphaloop.core.config import HotConfigStore
from alphaloop.core.models import BBOTick, FundingRateBar, OrderIntent, PriceBar, StrategyConfig
from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.data.quality import DataQualityAuditor
from alphaloop.data.store import MarketDataStore
from alphaloop.execution.bridge import (
    ExecutionBridge,
    KillTrigger,
    LIVE_EXECUTION_ENABLED,
)
from alphaloop.harness.backtester import FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier, TrialLedger
from alphaloop.harness.portfolio import PortfolioCombiner, SleeveInput
from alphaloop.plugins import TrendCTAPlugin, default_registry
from alphaloop.plugins.trend_cta_plugin import resample_ohlcv_1h
from alphaloop.web.server import create_server

HOUR_MS = 3600 * 1000
DAY_MS = 24 * HOUR_MS
BASE_TS = 1_760_000_000_000  # 2025-10-09 UTC (aligned cleanly > 2020)
BASE_TS = (BASE_TS // DAY_MS) * DAY_MS


class TestPhase2ProfitabilityAndBridge(unittest.TestCase):
    # =========================================================================
    # MODULE A: Vol-Tiered & Active Margin Defense in FundingCarryBacktester
    # =========================================================================

    def test_module_a_protective_exit_vs_single_bar_liquidation_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = MarketDataStore(Path(td) / "mkt.sqlite")
            # Build 120 hours of positive LT->Gate funding spread on ALT
            # Case 1 (gradual rally): +18% over several hours at 4x lev -> 0.72 margin consumed -> PROTECTIVE EXIT (0 liqs!)
            # Case 2 (instant flash): +30% in 1 bar at 4x lev -> 1.20 margin consumed -> STILL LIQUIDATES (1 liq)!
            f_lt, f_gt, c_gt = [], [], []
            px = 100.0
            for h in range(120):
                ts = BASE_TS + h * HOUR_MS
                f_lt.append(FundingRateBar("lighter", "GRAD", ts, 0.0005, 1.0, PROVENANCE_REAL))
                f_gt.append(FundingRateBar("gate", "GRAD", ts, 0.0, 1.0, PROVENANCE_REAL))
                # After entry around h=24, gradually climb +17% by h=36 (4x * 17% = 68% in [55%, 90%))
                if 26 <= h <= 36:
                    px_open = 100.0 * (1.0 + 0.016 * (h - 25))
                    px_high = px_open * 1.005
                else:
                    px_open = px if h < 26 else 117.6
                    px_high = px_open * 1.002
                c_gt.append(PriceBar("gate", "GRAD", ts, px_open, px_high, px_open * 0.998, px_open, 1e6, PROVENANCE_REAL))

                # Flash spike coin: +30% in single bar h=27 (4x * 30% = 120% >= 90%)
                f_lt.append(FundingRateBar("lighter", "FLASH", ts, 0.0005, 1.0, PROVENANCE_REAL))
                f_gt.append(FundingRateBar("gate", "FLASH", ts, 0.0, 1.0, PROVENANCE_REAL))
                fl_open = 100.0
                fl_high = 132.0 if h == 27 else 100.2
                c_gt.append(PriceBar("gate", "FLASH", ts, fl_open, fl_high, 99.8, fl_open, 1e6, PROVENANCE_REAL))

            store.upsert_funding(f_lt + f_gt)
            store.upsert_candles(c_gt)
            bt = FundingCarryBacktester(store)

            # Test GRAD with protective_margin_exit_pct=0.55 -> 0 liquidations, MARGIN_PROTECT_EXIT fired + storm fee!
            cfg_grad = StrategyConfig(
                strategy_id="test_prot_grad", version="v1", family="fc", status="DRAFT",
                venue_a="lighter", venue_b="gate", mode="perp_perp", symbols=["GRAD"],
                capital_usd=3000.0, leverage=4.0, max_positions=1, per_position_weight=1.0,
                entry_spread_apr_pct=10.0, exit_spread_apr_pct=1.0, lookback_hours=12,
                exit_confirm_hours=6, min_hold_hours=48, fee_a_bps=0.0, fee_b_bps=2.0,
                slippage_per_leg_bps=1.5, params={"protective_margin_exit_pct": 0.55, "protective_exit_slip_bps": 1.0},
            )
            res_grad = bt.run(cfg_grad)
            self.assertEqual(res_grad.metrics.liquidation_count, 0)
            prot_trades = [t for t in res_grad.trades if t.exit_reason == "MARGIN_PROTECT_EXIT"]
            self.assertGreaterEqual(len(prot_trades), 1, "Expected MARGIN_PROTECT_EXIT on gradual adverse rally")
            # Physical floor >=5bp/leg storm slip + taker exit surcharge MUST appear in rebalance_usd > $5
            self.assertGreater(prot_trades[0].rebalance_usd, 5.0)

            # Test FLASH with protective_margin_exit_pct=0.55 -> single-bar >=0.90 jump MUST STILL LIQUIDATE!
            cfg_flash = StrategyConfig.from_dict({**cfg_grad.to_dict(), "strategy_id": "test_prot_flash", "symbols": ["FLASH"]})
            res_flash = bt.run(cfg_flash)
            self.assertGreaterEqual(res_flash.metrics.liquidation_count, 1)
            self.assertTrue(any(t.exit_reason == "MARGIN_LIQUIDATION" for t in res_flash.trades))
            store.close()

    def test_module_a_causal_vol_tiered_leverage_and_nan_safety(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = MarketDataStore(Path(td) / "mkt.sqlite")
            # CALM coin: ~0.05% hourly moves (~5% ann vol) -> gets full 4.0x leverage
            # WILD coin: ~4.0% hourly moves (~370% ann vol) -> capped down to min_symbol_leverage (1.5x)
            bars_f, bars_c = [], []
            calm_px, wild_px = 100.0, 100.0
            for h in range(120):
                ts = BASE_TS + h * HOUR_MS
                for sym in ("CALM", "WILD"):
                    bars_f.append(FundingRateBar("lighter", sym, ts, 0.0004, 1.0, PROVENANCE_REAL))
                    bars_f.append(FundingRateBar("gate", sym, ts, 0.0, 1.0, PROVENANCE_REAL))
                c_next = calm_px * (1.0005 if h % 2 == 0 else 0.9995)
                w_next = wild_px * (1.04 if h % 2 == 0 else 0.9615)
                bars_c.append(PriceBar("gate", "CALM", ts, calm_px, max(calm_px, c_next), min(calm_px, c_next), c_next, 1e6, PROVENANCE_REAL))
                bars_c.append(PriceBar("gate", "WILD", ts, wild_px, max(wild_px, w_next), min(wild_px, w_next), w_next, 1e6, PROVENANCE_REAL))
                calm_px, wild_px = c_next, w_next

            store.upsert_funding(bars_f)
            store.upsert_candles(bars_c)
            bt = FundingCarryBacktester(store)
            cfg = StrategyConfig(
                strategy_id="vol_tier_test", version="v1", family="fc", status="DRAFT",
                venue_a="lighter", venue_b="gate", mode="perp_perp", symbols=["CALM", "WILD"],
                capital_usd=2000.0, leverage=4.0, max_positions=2, per_position_weight=0.5,
                entry_spread_apr_pct=10.0, exit_spread_apr_pct=1.0, lookback_hours=48,
                exit_confirm_hours=12, min_hold_hours=24,
                params={
                    "vol_target_leverage_enabled": True,
                    "vol_lookback_hours": 48,
                    "vol_target_ann_pct": 120.0,
                    "min_symbol_leverage": 1.5,
                    "protective_margin_exit_pct": float("nan"),  # Adversarial NaN ignored safely
                },
            )
            res = bt.run(cfg)
            by_sym_notional = {t.symbol: t.notional_per_leg_usd for t in res.trades}
            # slot_capital=$1000, slot_margin_per_leg=$500 -> CALM at 4.0x = $2000, WILD at 1.5x = $750
            self.assertAlmostEqual(by_sym_notional["CALM"], 2000.0, delta=5.0)
            self.assertAlmostEqual(by_sym_notional["WILD"], 750.0, delta=5.0)
            store.close()

    # =========================================================================
    # MODULE B: TrendCTAPlugin & Single-Venue Perp DQ-04 Overlap Gate
    # =========================================================================

    def test_module_b_resample_drops_incomplete_and_t1_causality_and_funding_drag(self) -> None:
        # 1) Resampling drops incomplete trailing bucket (<12h)
        raw_1h = [
            PriceBar("gate", "BTC", BASE_TS + h * HOUR_MS, 100.0 + h, 101.0 + h, 99.0 + h, 100.5 + h, 1e5, PROVENANCE_REAL)
            for h in range(30)  # 2 complete 12h buckets (0..11, 12..23) + 1 incomplete 6h bucket (24..29)
        ]
        resamp = resample_ohlcv_1h(raw_1h, 12)
        self.assertEqual(len(resamp), 2)
        self.assertEqual(resamp[0].open, 100.0)
        self.assertEqual(resamp[0].close, 111.5)

        # 2) TrendCTA captures sustained breakout with T+1 open fill & real perp funding drag!
        with tempfile.TemporaryDirectory() as td:
            store = MarketDataStore(Path(td) / "cta.sqlite")
            candles, fundings = [], []
            # 25 days = 600 hourly bars (50 complete 12h bars):
            # Bars 0..19 (h=0..239): sideways range 100..102
            # Bars 20..49 (h=240..599): clean uptrend from 102 -> 160
            for h in range(600):
                ts = BASE_TS + h * HOUR_MS
                if h < 240:
                    px = 100.0 + (h % 12) * 0.15
                else:
                    px = 102.0 + (h - 240) * 0.16
                candles.append(PriceBar("gate", "PAXG", ts, px, px * 1.003, px * 0.997, px * 1.001, 5e5, PROVENANCE_REAL))
                # Positive 1h funding (+0.0002/h) -> LONG must PAY negative gross_funding_usd!
                fundings.append(FundingRateBar("gate", "PAXG", ts, 0.0002, 1.0, PROVENANCE_REAL))
            store.upsert_candles(candles)
            store.upsert_funding(fundings)

            plugin = TrendCTAPlugin()
            cfg = plugin.default_config(
                symbols=["PAXG"],
                venue_a="gate",
                venue_b="gate",
                capital_usd=3000.0,
                max_positions=1,
                per_position_weight=1.0,
                params={"timeframe_hours": 12, "entry_n": 12, "exit_n": 6, "atr_n": 10, "atr_mult": 3.0, "max_symbol_leverage": 1.0},
            )
            res = plugin.evaluate(store, cfg)
            self.assertGreaterEqual(res.metrics.trade_count, 1)
            self.assertEqual(res.trades[0].direction, "LONG")
            self.assertGreater(res.metrics.basis_pnl_usd, 0.0, "Expected positive directional trend PnL")
            self.assertLess(res.metrics.gross_funding_usd, 0.0, "LONG perp MUST pay positive funding rate!")
            self.assertEqual(res.metrics.liquidation_count, 0)

            # Immutable friction test: Agent trying to zero out fee/slippage MUST be blocked
            clamped = plugin.validate_and_clamp(cfg, "v1.1", {"fee_a_bps": 0.0, "slippage_per_leg_bps": 0.0, "entry_n": 25})
            self.assertGreaterEqual(clamped.fee_a_bps, 2.0)
            self.assertGreaterEqual(clamped.slippage_per_leg_bps, 2.0)
            self.assertEqual(clamped.params["entry_n"], 25)
            store.close()

    def test_module_b_single_venue_dq_rejects_missing_funding_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = MarketDataStore(Path(td) / "dq_single.sqlite")
            # 100 hours of candles, ONLY 50 hours of funding (50% overlap < 98% -> MUST BE DATA_REJECTED!)
            candles = [
                PriceBar("gate", "BTC", BASE_TS + h * HOUR_MS, 50000.0 + h * 10, 50050.0 + h * 10, 49950.0 + h * 10, 50010.0 + h * 10, 1e6, PROVENANCE_REAL)
                for h in range(100)
            ]
            fund_half = [
                FundingRateBar("gate", "BTC", BASE_TS + h * HOUR_MS, 0.0001, 1.0, PROVENANCE_REAL)
                for h in range(50)
            ]
            store.upsert_candles(candles)
            store.upsert_funding(fund_half)
            aud = DataQualityAuditor(store)
            rep_bad = aud.audit_single_venue_perp("gate", ["BTC"], persist=False)
            self.assertEqual(rep_bad.tier, "DATA_REJECTED")
            dq4 = next(c for c in rep_bad.checks if c.gate_id == "DQ-04")
            self.assertFalse(dq4.passed)

            # Fill remaining 50 hours of funding -> 100% overlap -> DATA_QUALIFIED_FULL!
            fund_rest = [
                FundingRateBar("gate", "BTC", BASE_TS + h * HOUR_MS, 0.0001, 1.0, PROVENANCE_REAL)
                for h in range(50, 100)
            ]
            store.upsert_funding(fund_rest)
            rep_ok = aud.audit_single_venue_perp("gate", ["BTC"], persist=False)
            self.assertEqual(rep_ok.tier, "DATA_QUALIFIED_FULL")
            store.close()

    def test_module_b_chandelier_stop_causality_and_gap_fill(self) -> None:
        """Verify Chandelier stop uses ONLY k-1 peak/ATR (no same-bar favorable peak update) & gap fills at min(open, stop)."""
        with tempfile.TemporaryDirectory() as td:
            store = MarketDataStore(Path(td) / "gap.sqlite")
            # 35 complete 12h bars (420 hours):
            # Bars 0..14: flat 100..101 (ATR ~ 1.0)
            # Bar 15: breakout close=105 -> schedules OPEN_LONG at Bar 16 open
            # Bar 16: open=105, high=106, low=104.5, close=105.5 (holds)
            # Bar 17: gap down open=95 (below active_stop ~ 102), low=94, high=96, close=95
            #   -> MUST stop out at min(open=95, stop~102) = 95.0 (NOT optimistic stop=102!) with CHANDELIER_STOP & taker fee!
            candles, fundings = [], []
            for b in range(30):
                if b < 15:
                    o, h, l, c = 100.0, 101.0, 99.5, 100.2
                elif b == 15:
                    o, h, l, c = 100.5, 105.5, 100.5, 105.0
                elif b == 16:
                    o, h, l, c = 105.0, 106.0, 104.5, 105.5
                elif b == 17:
                    o, h, l, c = 95.0, 96.0, 94.0, 95.0
                else:
                    o, h, l, c = 95.0, 95.5, 94.5, 95.0
                for sub_h in range(12):
                    ts = BASE_TS + (b * 12 + sub_h) * HOUR_MS
                    bo = o if sub_h == 0 else c
                    bh = h if sub_h == 6 else max(bo, c)
                    bl = l if sub_h == 6 else min(bo, c)
                    candles.append(PriceBar("gate", "ETH", ts, bo, bh, bl, c, 1e6, PROVENANCE_REAL))
                    fundings.append(FundingRateBar("gate", "ETH", ts, 0.0, 1.0, PROVENANCE_REAL))
            store.upsert_candles(candles)
            store.upsert_funding(fundings)

            plugin = TrendCTAPlugin()
            cfg = plugin.default_config(
                symbols=["ETH"], venue_a="gate", venue_b="gate", capital_usd=3000.0,
                max_positions=1, per_position_weight=1.0, fee_a_bps=2.0, slippage_per_leg_bps=2.0,
                params={"timeframe_hours": 12, "entry_n": 10, "exit_n": 5, "atr_n": 10, "atr_mult": 2.0, "max_symbol_leverage": 1.0},
            )
            res = plugin.evaluate(store, cfg)
            self.assertGreaterEqual(len(res.trades), 1)
            tr = res.trades[0]
            self.assertEqual(tr.exit_reason, "CHANDELIER_STOP")
            # Gap fill at 95 vs entry 105 is -9.52% loss (if filled optimistically at stop ~102 would only be -2.8%)
            loss_pct = (tr.basis_pnl_usd / tr.notional_per_leg_usd) * 100.0
            self.assertLess(loss_pct, -8.0, f"Expected conservative gap fill near 95 (-9.5%), got {loss_pct:.2f}%")
            store.close()

    # =========================================================================
    # MODULE C: PortfolioCombiner Alignment, Conservation, Correlation & Scaling
    # =========================================================================

    def test_module_c_portfolio_combiner_correlation_conservation_and_scaling(self) -> None:
        combiner = PortfolioCombiner()
        # Construct 90 overlapping days of two negatively correlated profitable sleeves
        eq_a, eq_b = [], []
        va, vb = 3000.0, 3000.0
        for d in range(90):
            ts = BASE_TS + d * DAY_MS
            ra = 0.0055 if d % 2 == 0 else -0.0020
            rb = -0.0025 if d % 2 == 0 else 0.0040
            va *= (1.0 + ra)
            vb *= (1.0 + rb)
            eq_a.append((ts, va))
            eq_b.append((ts, vb))

        s_a = SleeveInput("carry", eq_a, capital_usd=3000.0, weight=0.5, avg_monthly_turnover_x=2.0, measured_top_depth_usd=20000.0)
        s_b = SleeveInput("trend", eq_b, capital_usd=3000.0, weight=0.5, avg_monthly_turnover_x=3.0, measured_top_depth_usd=30000.0)

        res = combiner.combine([s_a, s_b], capital_usd=10000.0, weighting_mode="fixed", rebalance_every_days=30, rebalance_cost_bps=10.0)
        self.assertTrue(res.valid)
        self.assertEqual(res.overlap_days, 90)
        self.assertLess(res.correlation_matrix["carry"]["trend"], -0.90)
        # Diversification MUST dramatically lift portfolio Sharpe vs each individual sleeve
        self.assertGreater(res.sharpe_ratio, res.sleeve_summaries["carry"]["sharpe"] * 2.0)
        self.assertGreater(res.rebalance_cost_total_pct, 0.0)
        # Capacity scaling: $50k drag MUST exceed $3k drag
        drag_3k = res.scaling_projections[0]["capacity_impact_drag_apr_pct"]
        drag_50k = res.scaling_projections[2]["capacity_impact_drag_apr_pct"]
        self.assertGreater(drag_50k, drag_3k * 3.0)

        # Causal inverse-vol mode must conserve weights == 1.0
        res_iv = combiner.combine([s_a, s_b], capital_usd=10000.0, weighting_mode="causal_inv_vol", rebalance_every_days=15)
        self.assertTrue(res_iv.valid)
        self.assertAlmostEqual(sum(res_iv.sleeve_weights_final.values()), 1.0, places=3)

        # Fail-closed checks: weights not summing to 1.0 raises ValueError; <60 overlap days returns valid=False
        with self.assertRaises(ValueError):
            combiner.combine([SleeveInput("x", eq_a, weight=0.6), SleeveInput("y", eq_b, weight=0.6)])
        res_short = combiner.combine([SleeveInput("x", eq_a[:30], weight=0.5), SleeveInput("y", eq_b[:30], weight=0.5)])
        self.assertFalse(res_short.valid)
        self.assertIn("INSUFFICIENT_OVERLAP", res_short.error)

    # =========================================================================
    # MODULE D: ExecutionBridge (Shadow-Only, Hash Chain, Kill Switch, Hot Watch)
    # =========================================================================

    def test_module_d_execution_bridge_shadow_gates_idempotency_tamper_and_kill_switch(self) -> None:
        self.assertFalse(LIVE_EXECUTION_ENABLED)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                ExecutionBridge(td, live_mode=True)

            bridge = ExecutionBridge(td, max_notional_cap_usd=2000.0, max_bbo_age_ms=10_000)
            cfg_shadow = StrategyConfig(
                strategy_id="shadow_carry", version="v1.0", family="fc", status="SHADOW_LIVE",
                venue_a="lighter", venue_b="gate", mode="perp_perp", symbols=["BTC", "ETH"],
                capital_usd=3000.0, leverage=2.0, max_positions=3, per_position_weight=0.33,
                slippage_per_leg_bps=2.0,
            )
            now = BASE_TS + 50 * HOUR_MS
            good_bbo = {
                "lighter": BBOTick("lighter", "BTC", now - 1000, 90000.0, 90008.0, 5.0, 5.0, 40.0, PROVENANCE_REAL),
                "gate": BBOTick("gate", "BTC", now - 1200, 89998.0, 90006.0, 5.0, 5.0, 45.0, PROVENANCE_REAL),
            }
            intent = OrderIntent("shadow_carry", "BTC", "OPEN_SHORT_A_LONG_B", 800.0, 22.5, "SPREAD_ENTRY", now)

            # 1) Valid SHADOW_LIVE plan succeeds & dedups deterministically on replay
            p1 = bridge.submit_intent(cfg_shadow, intent, good_bbo, now_ms=now)
            self.assertEqual(p1.status, "PLANNED_SHADOW")
            self.assertEqual(len(p1.legs), 2)
            self.assertFalse(p1.duplicate)

            p1_dup = bridge.submit_intent(cfg_shadow, intent, good_bbo, now_ms=now + 5000)
            self.assertTrue(p1_dup.duplicate)
            self.assertEqual(p1_dup.plan_id, p1.plan_id)

            # 2) Fail-closed blocks: DRAFT status, stale BBO, wrong symbol BBO, NaN notional
            cfg_draft = StrategyConfig.from_dict({**cfg_shadow.to_dict(), "status": "DRAFT"})
            p_draft = bridge.submit_intent(cfg_draft, OrderIntent("shadow_carry", "BTC", "OPEN_SHORT_A_LONG_B", 500.0, 10.0, "R", now + 120_000), good_bbo, now_ms=now + 120_000)
            self.assertEqual(p_draft.status, "BLOCKED")
            self.assertIn("STATUS_NOT_SHADOW_ELIGIBLE", p_draft.block_reason)

            stale_bbo = {
                "lighter": BBOTick("lighter", "BTC", now - 60_000, 90000.0, 90008.0, 5.0, 5.0, 40.0, PROVENANCE_REAL),
                "gate": good_bbo["gate"],
            }
            p_stale = bridge.submit_intent(cfg_shadow, OrderIntent("shadow_carry", "BTC", "OPEN_SHORT_A_LONG_B", 500.0, 10.0, "R", now + 240_000), stale_bbo, now_ms=now)
            self.assertEqual(p_stale.status, "BLOCKED")
            self.assertIn("STALE_BBO", p_stale.block_reason)

            wrong_sym_bbo = {
                "lighter": BBOTick("lighter", "ETH", now - 1000, 3000.0, 3001.0, 5.0, 5.0, 40.0, PROVENANCE_REAL),
                "gate": good_bbo["gate"],
            }
            p_mis = bridge.submit_intent(cfg_shadow, OrderIntent("shadow_carry", "BTC", "OPEN_SHORT_A_LONG_B", 500.0, 10.0, "R", now + 360_000), wrong_sym_bbo, now_ms=now)
            self.assertEqual(p_mis.status, "BLOCKED")
            self.assertIn("BBO_IDENTITY_MISMATCH", p_mis.block_reason)

            p_nan = bridge.submit_intent(cfg_shadow, OrderIntent("shadow_carry", "BTC", "OPEN_SHORT_A_LONG_B", float("nan"), 10.0, "R", now + 480_000), good_bbo, now_ms=now)
            self.assertEqual(p_nan.status, "BLOCKED")
            self.assertIn("INVALID_NOTIONAL", p_nan.block_reason)

            # 3) KillSwitch engage blocks valid intent; clear requires HUMAN_RESET:<id>
            bridge.kill_switch.engage(KillTrigger.OPERATOR, actor="owner", detail="emergency halt test")
            p_killed = bridge.submit_intent(cfg_shadow, OrderIntent("shadow_carry", "BTC", "OPEN_SHORT_A_LONG_B", 600.0, 15.0, "R", now + 600_000), good_bbo, now_ms=now)
            self.assertEqual(p_killed.status, "BLOCKED")
            self.assertEqual(p_killed.block_reason, "KILL_SWITCH_ENGAGED")

            with self.assertRaises(PermissionError):
                bridge.kill_switch.clear("owner", "BAD_TOKEN", "try bypass")
            bridge.kill_switch.clear("owner", "HUMAN_RESET:ticket-42", "Verified safe")
            self.assertFalse(bridge.kill_switch.is_engaged())

            # 4) HotConfigWatcher reloads atomic active_configs AND auto-trips KillSwitch on corrupt JSON
            cstore = HotConfigStore(td)
            cstore.register_version(cfg_shadow, actor="test", reason="publish shadow")
            ev_ok = bridge.watcher.poll()
            self.assertEqual(ev_ok[0]["status"], "RELOADED")
            self.assertFalse(bridge.kill_switch.is_engaged())

            (Path(td) / "active_configs" / "corrupt.json").write_text("{broken json!!!", encoding="utf-8")
            ev_bad = bridge.watcher.poll()
            self.assertTrue(any(e["status"] == "CORRUPT_KILL_ENGAGED" for e in ev_bad))
            self.assertTrue(bridge.kill_switch.is_engaged())
            cstore.close()

            # 5) Hash chain integrity passes BEFORE tampering, FAILS immediately AFTER tampering
            ok, detail, n_events = bridge.audit.verify_chain()
            self.assertTrue(ok, detail)
            self.assertGreaterEqual(n_events, 7)

            bridge.conn.execute("UPDATE execution_audit_chain SET payload_json='{\"tampered\":true}' WHERE seq=2")
            bridge.conn.commit()
            ok_after, detail_after, _ = bridge.audit.verify_chain()
            self.assertFalse(ok_after)
            self.assertIn("TAMPERED_RECORD_HASH_AT_SEQ_2", detail_after)
            bridge.close()

    # =========================================================================
    # MODULE E: Web Console v2.1 New Endpoints & Zero Inline Handler Hygiene
    # =========================================================================

    def test_module_e_web_console_v21_portfolio_and_execution_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            srv, ctx = create_server(td, host="127.0.0.1", port=0)
            port = srv.server_address[1]
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            try:
                base = f"http://127.0.0.1:{port}"
                html = urllib.request.urlopen(f"{base}/").read().decode("utf-8")
                self.assertIn("PortfolioCombiner", html)
                self.assertIn("Shadow Execution Bridge", html)
                self.assertNotIn("onclick=", html.lower())

                pl = json.loads(urllib.request.urlopen(f"{base}/api/plugins").read())
                self.assertEqual(len(pl["plugins"]), 4)
                self.assertIn("trend_cta", {p["plugin_id"] for p in pl["plugins"]})

                pf = json.loads(urllib.request.urlopen(f"{base}/api/portfolio").read())
                self.assertFalse(pf["valid"])

                ex = json.loads(urllib.request.urlopen(f"{base}/api/execution/plans").read())
                self.assertFalse(ex["live_execution_enabled"])
                self.assertTrue(ex["audit_chain_valid"])
                self.assertEqual(ex["run_mode"], "SHADOW_DRY_RUN")
            finally:
                srv.shutdown()
                srv.server_close()
                ctx.close()

    # =========================================================================
    # RED-TEAM PHASE 2 REGRESSION TESTS (BLOCKER #1/#2/#3, HIGH #4/#6, MED #7, LOW #10)
    # =========================================================================

    def test_redteam_p2_blocker1_nan_cfg_spread_strategy_spoof_and_blocker2_ts_idem_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bridge = ExecutionBridge(td, max_notional_cap_usd=2000.0, max_bbo_age_ms=10_000)
            now = BASE_TS + 80 * HOUR_MS
            cfg = StrategyConfig(
                strategy_id="shadow_ok", version="v1.0", family="fc", status="SHADOW_LIVE",
                venue_a="hyperliquid", venue_b="hyperliquid", mode="single_venue_perp",
                symbols=["BTC"], capital_usd=3000.0, leverage=2.0, max_positions=2,
                per_position_weight=0.5, slippage_per_leg_bps=2.0,
            )
            bbo = {"hyperliquid": BBOTick("hyperliquid", "BTC", now - 500, 90000.0, 90005.0, 5.0, 5.0, 20.0, PROVENANCE_REAL)}

            # BLOCKER #1a: NaN cfg.capital_usd MUST NOT bypass max_notional_cap_usd!
            cfg_nan_cap = StrategyConfig.from_dict({**cfg.to_dict(), "capital_usd": float("nan")})
            p_cap = bridge.submit_intent(cfg_nan_cap, OrderIntent("shadow_ok", "BTC", "OPEN_LONG", 5_000_000.0, 10.0, "R", now), bbo, now_ms=now)
            self.assertEqual(p_cap.status, "BLOCKED")
            self.assertIn("INVALID_CFG_CAPITAL", p_cap.block_reason)

            # BLOCKER #1b: NaN max_spread_bps MUST NOT bypass spread ceiling!
            p_sp = bridge.submit_intent(cfg, OrderIntent("shadow_ok", "BTC", "OPEN_LONG", 500.0, 10.0, "R", now + 60_000), bbo, now_ms=now, max_spread_bps=float("nan"))
            self.assertEqual(p_sp.status, "BLOCKED")
            self.assertIn("INVALID_MAX_SPREAD_BPS", p_sp.block_reason)

            # BLOCKER #1c: intent.strategy_id != cfg.strategy_id MUST block!
            p_spoof = bridge.submit_intent(cfg, OrderIntent("unapproved_draft", "BTC", "OPEN_LONG", 500.0, 10.0, "R", now + 120_000), bbo, now_ms=now)
            self.assertEqual(p_spoof.status, "BLOCKED")
            self.assertIn("STRATEGY_ID_MISMATCH", p_spoof.block_reason)

            # BLOCKER #1d: NaN intent.ts_ms MUST fail-closed (never crash!)
            p_ts = bridge.submit_intent(cfg, OrderIntent("shadow_ok", "BTC", "OPEN_LONG", 500.0, 10.0, "R", float("nan")), bbo, now_ms=now)  # type: ignore[arg-type]
            self.assertEqual(p_ts.status, "BLOCKED")
            self.assertIn("INVALID_INTENT_TS", p_ts.block_reason)

            # LOW #10: trailing newline in HUMAN_RESET token MUST be rejected by fullmatch!
            bridge.kill_switch.engage(KillTrigger.OPERATOR, actor="owner", detail="test")
            with self.assertRaises(PermissionError):
                bridge.kill_switch.clear("owner", "HUMAN_RESET:ticket-99\n", "trailing newline bypass")
            bridge.kill_switch.clear("owner", "HUMAN_RESET:ticket-99", "valid reset")

            # BLOCKER #2: mutating ts_ms OR idempotency_key in execution_audit_chain MUST fail verify_chain()!
            ok_before, _, _ = bridge.audit.verify_chain()
            self.assertTrue(ok_before)
            bridge.conn.execute("UPDATE execution_audit_chain SET ts_ms=0 WHERE seq=1")
            bridge.conn.commit()
            ok_ts, detail_ts, _ = bridge.audit.verify_chain()
            self.assertFalse(ok_ts)
            self.assertIn("TAMPERED_RECORD_HASH_AT_SEQ_1", detail_ts)

            # Restore ts_ms and tamper idempotency_key on seq=2
            orig_ts = json.loads(bridge.conn.fetch_one("SELECT payload_json FROM execution_audit_chain WHERE seq=1")["payload_json"])["ts_ms"]
            bridge.conn.execute("UPDATE execution_audit_chain SET ts_ms=? WHERE seq=1", (orig_ts,))
            bridge.conn.execute("UPDATE execution_audit_chain SET idempotency_key='forged_key' WHERE seq=2")
            bridge.conn.commit()
            ok_ik, detail_ik, _ = bridge.audit.verify_chain()
            self.assertFalse(ok_ik)
            self.assertIn("TAMPERED_RECORD_HASH_AT_SEQ_2", detail_ik)
            bridge.close()

    def test_redteam_p2_blocker3_microrun_pre_bbo_funding_and_trend_cta_execution(self) -> None:
        from alphaloop.core.models import MarketEvent
        from alphaloop.harness.micro_runner import MicroRunSession
        from alphaloop.plugins import FundingCarryPlugin

        # BLOCKER #3A: FundingCarry MUST NOT open on pre-BBO funding before live BBO arrives!
        fc = FundingCarryPlugin()
        cfg_fc = fc.default_config(symbols=["BTC"], venue_a="lighter", venue_b="gate", entry_spread_apr_pct=10.0)
        events_pre = [
            MarketEvent("FUNDING", BASE_TS, "lighter", "BTC", {"funding_rate": 0.0005, "apr_pct": 438.0}),
            MarketEvent("FUNDING", BASE_TS, "gate", "BTC", {"funding_rate": 0.0, "apr_pct": 0.0}),
        ]
        exec_intents, pnl, dd = MicroRunSession._simulate_paper_intents(events_pre, fc, cfg_fc)
        self.assertEqual(len(exec_intents), 0, "Pre-BBO funding MUST NOT produce executed paper trades!")
        self.assertEqual(pnl, 0.0)

        # BLOCKER #3B & MEDIUM #7: TrendCTA on_event buckets BBO ticks by timeframe_hours and executes single-venue paper fills
        cta = TrendCTAPlugin()
        cfg_cta = cta.default_config(
            symbols=["BTC"], venue_a="hyperliquid", venue_b="hyperliquid",
            max_positions=1, per_position_weight=1.0, fee_a_bps=4.5,
            params={"timeframe_hours": 4, "entry_n": 5, "exit_n": 3, "atr_n": 5, "atr_mult": 2.0},
        )
        # Rapid 20-second ticks inside ONE 4h bucket MUST emit 0 intents
        rapid_events = [
            MarketEvent("BBO", BASE_TS + i * 20_000, "hyperliquid", "BTC", {"bid": 100.0 + i, "ask": 100.1 + i})
            for i in range(15)
        ]
        rapid_exec, _, _ = MicroRunSession._simulate_paper_intents(rapid_events, cta, cfg_cta)
        self.assertEqual(len(rapid_exec), 0, "Raw 20s ticks inside one bucket must NOT trigger TrendCTA breakout!")

        # Crossing 8 sequential 4h buckets with a breakout on bucket 7 -> executes OPEN_LONG & realizes non-zero MTM PnL!
        tf_events = []
        for b in range(9):
            ts = BASE_TS + b * 4 * HOUR_MS
            px = 100.0 if b < 6 else (112.0 if b == 6 else 120.0)
            tf_events.append(MarketEvent("BBO", ts, "hyperliquid", "BTC", {"bid": px - 0.05, "ask": px + 0.05}))
        cta_exec, cta_pnl, _ = MicroRunSession._simulate_paper_intents(tf_events, cta, cfg_cta)
        self.assertGreaterEqual(len(cta_exec), 1)
        self.assertEqual(cta_exec[0].action, "OPEN_LONG")
        self.assertNotEqual(cta_pnl, 0.0, "Executed TrendCTA paper position must reflect real MTM PnL/fees!")

    def test_redteam_p2_high4_watcher_sha_symlink_evict_and_high6_portfolio_bankruptcy(self) -> None:
        # HIGH #4: HotConfigWatcher rejects SHA mismatch, filename mismatch, symlinks, and evicts deleted configs
        with tempfile.TemporaryDirectory() as td:
            bridge = ExecutionBridge(td)
            cstore = HotConfigStore(td)
            cfg = StrategyConfig(
                strategy_id="good_strat", version="v1.0", family="fc", status="SHADOW_LIVE",
                venue_a="hyperliquid", venue_b="gate", mode="perp_perp", symbols=["BTC"],
                capital_usd=3000.0, leverage=2.0, max_positions=2, per_position_weight=0.5,
            )
            cstore.register_version(cfg, actor="test", reason="valid")
            self.assertEqual(bridge.watcher.poll()[0]["status"], "RELOADED")
            self.assertIsNotNone(bridge.watcher.get_loaded("good_strat"))

            # Tamper config body while keeping old config_sha256 -> MUST reject & evict & trip kill switch!
            active_f = Path(td) / "active_configs" / "good_strat.json"
            doc = json.loads(active_f.read_text(encoding="utf-8"))
            doc["config"]["leverage"] = 99.0  # Tampered without updating config_sha256
            active_f.write_text(json.dumps(doc), encoding="utf-8")
            ev_sha = bridge.watcher.poll()
            self.assertTrue(any(e["status"] == "CORRUPT_KILL_ENGAGED" for e in ev_sha))
            self.assertIsNone(bridge.watcher.get_loaded("good_strat"))
            self.assertTrue(bridge.kill_switch.is_engaged())
            bridge.kill_switch.clear("owner", "HUMAN_RESET:rt-4", "reset after sha test")

            # Remove tampered file, re-publish valid, then delete file -> MUST evict from memory!
            active_f.unlink()
            cstore.register_version(cfg, actor="test", reason="republish")
            bridge.watcher.poll()
            self.assertIsNotNone(bridge.watcher.get_loaded("good_strat"))
            active_f.unlink()
            ev_del = bridge.watcher.poll()
            self.assertTrue(any(e["status"] == "REMOVED" for e in ev_del))
            self.assertIsNone(bridge.watcher.get_loaded("good_strat"))

            # Symlink attack -> MUST reject & trip kill switch!
            outside = Path(td) / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            (Path(td) / "active_configs" / "evil_link.json").symlink_to(outside)
            ev_sym = bridge.watcher.poll()
            self.assertTrue(any(e["status"] == "CORRUPT_KILL_ENGAGED" for e in ev_sym))
            self.assertTrue(bridge.kill_switch.is_engaged())
            cstore.close()
            bridge.close()

        # HIGH #6: PortfolioCombiner MUST fail-closed on a day-66 blowup (eq <= 0), never silently truncate to days 1..65!
        combiner = PortfolioCombiner()
        eq_ok = [(BASE_TS + d * DAY_MS, 3000.0 * (1.0 + 0.001 * d)) for d in range(80)]
        eq_blowup = [(BASE_TS + d * DAY_MS, 3000.0 * (1.0 + 0.002 * d) if d < 66 else -50.0) for d in range(80)]
        res = combiner.combine([SleeveInput("ok", eq_ok, weight=0.5), SleeveInput("blown", eq_blowup, weight=0.5)])
        self.assertFalse(res.valid)
        self.assertIn("INVALID_OR_BANKRUPT_EQUITY:blown", res.error)


if __name__ == "__main__":
    unittest.main()
