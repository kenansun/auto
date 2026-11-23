"""Single-Kernel Parity, Fail-Closed Param Schema, and A1/A4 Regression Tests (CC-20261123-001)."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from alphaloop.core.config import HotConfigStore
from alphaloop.core.models import BBOTick, FundingRateBar, MarketEvent, PriceBar
from alphaloop.core.provenance import PROVENANCE_REAL, utc_now_ms
from alphaloop.daemon.shadow_runner import ShadowParityAuditor, ShadowPortfolioDaemon
from alphaloop.daemon.watchdog_hook import WatchdogHook
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import HOUR_MS, compute_causal_symbol_leverage
from alphaloop.plugins import FundingCarryPlugin, TrendCTAPlugin
from alphaloop.plugins.trend_cta_plugin import resample_ohlcv_1h
from scripts.run_nightly_factory import run_nightly_factory
from scripts.run_shadow_daemon import bootstrap_balanced_shadow_configs


class TestSingleKernelParityAndSchema(unittest.TestCase):
    def test_dead_and_unknown_params_rejected_fail_closed(self) -> None:
        carry = FundingCarryPlugin()
        with self.assertRaises(ValueError):
            carry.default_config(
                symbols=["BTC"],
                params={"vol_tier_low_ann_pct": 65.0, "vol_tier_high_ann_pct": 130.0},
            )

        # Direct mutated config must also fail in BOTH evaluate() and on_event()
        bad_carry = carry.default_config(symbols=["BTC"])
        bad_carry.params["vol_tier_high_ann_pct"] = 130.0
        with tempfile.TemporaryDirectory() as td:
            store = MarketDataStore(Path(td) / "md.sqlite")
            try:
                with self.assertRaises(ValueError):
                    carry.evaluate(store, bad_carry)
            finally:
                store.close()
        with self.assertRaises(ValueError):
            carry.on_event(
                MarketEvent("BBO", 1_795_000_000_000, "lighter", "BTC", {"bid": 100.0, "ask": 100.1}),
                {},
                bad_carry,
            )

        cta = TrendCTAPlugin()
        with self.assertRaises(ValueError):
            cta.default_config(symbols=["BTC"], params={"typo_param": 123})

        bad_cta = cta.default_config(symbols=["BTC"])
        bad_cta.params["dead_cta_key"] = True
        with tempfile.TemporaryDirectory() as td:
            store = MarketDataStore(Path(td) / "md.sqlite")
            try:
                with self.assertRaises(ValueError):
                    cta.evaluate(store, bad_cta)
            finally:
                store.close()
        with self.assertRaises(ValueError):
            cta.on_event(
                MarketEvent("BBO", 1_795_000_000_000, "gate", "BTC", {"bid": 100.0, "ask": 100.1}),
                {},
                bad_cta,
            )

    def test_bootstrap_balanced_configs_match_phase2_certified_spec(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            seeded = bootstrap_balanced_shadow_configs(data_dir)
            self.assertEqual(set(seeded), {"hl_trend_cta_prod", "hl_gate_safe_carry_prod"})
            # Idempotent without force_reseed
            self.assertEqual(bootstrap_balanced_shadow_configs(data_dir), [])
            # force_reseed updates both
            self.assertEqual(
                set(bootstrap_balanced_shadow_configs(data_dir, force_reseed=True)),
                {"hl_trend_cta_prod", "hl_gate_safe_carry_prod"},
            )

            cstore = HotConfigStore(data_dir / "state")
            try:
                carry = cstore.read_active("hl_gate_safe_carry_prod")
                cta = cstore.read_active("hl_trend_cta_prod")
            finally:
                cstore.close()

            self.assertIsNotNone(carry)
            self.assertIsNotNone(cta)
            assert carry is not None and cta is not None

            # Pillar 1 exact Phase 2 certified spec
            self.assertEqual(carry.venue_a, "hyperliquid")
            self.assertEqual(carry.venue_b, "gate")
            self.assertEqual(carry.max_positions, 4)
            self.assertAlmostEqual(carry.per_position_weight, 0.25)
            self.assertAlmostEqual(carry.leverage, 3.5)
            self.assertAlmostEqual(carry.entry_spread_apr_pct, 10.0)
            self.assertAlmostEqual(carry.exit_spread_apr_pct, 1.0)
            self.assertEqual(carry.lookback_hours, 96)
            self.assertEqual(carry.exit_confirm_hours, 24)
            self.assertEqual(carry.min_hold_hours, 72)
            self.assertAlmostEqual(carry.fee_a_bps, 1.5)
            self.assertAlmostEqual(carry.fee_b_bps, 2.0)
            self.assertTrue(carry.params["vol_target_leverage_enabled"])
            self.assertEqual(carry.params["vol_lookback_hours"], 168)
            self.assertAlmostEqual(carry.params["vol_target_ann_pct"], 130.0)
            self.assertAlmostEqual(carry.params["min_symbol_leverage"], 1.5)
            self.assertAlmostEqual(carry.params["protective_margin_exit_pct"], 0.50)
            self.assertAlmostEqual(carry.params["protective_exit_slip_bps"], 5.0)
            self.assertNotIn("vol_tier_low_ann_pct", carry.params)
            self.assertNotIn("vol_tier_high_ann_pct", carry.params)

            # Pillar 2 exact Phase 2 certified spec
            self.assertEqual(cta.venue_a, "hyperliquid")
            self.assertEqual(cta.max_positions, 6)
            self.assertAlmostEqual(cta.per_position_weight, 0.1667)
            self.assertAlmostEqual(cta.fee_a_bps, 1.5)
            self.assertEqual(cta.execution_mode, "TAKER_TAKER")
            self.assertEqual(cta.params["entry_n"], 40)
            self.assertEqual(cta.params["exit_n"], 16)
            self.assertAlmostEqual(cta.params["atr_mult"], 3.5)
            self.assertEqual(cta.params["cooldown_hours"], 24)

    def test_funding_carry_bar_by_bar_parity_evaluate_vs_on_event(self) -> None:
        """Exact bar-by-bar parity: rolling APR, vol-tiered sizing, protective margin exit, cooldown & hysteresis."""
        carry = FundingCarryPlugin()
        cfg = carry.default_config(
            symbols=["SOL"],
            strategy_id="parity_carry",
            version="v1",
            venue_a="hyperliquid",
            venue_b="gate",
            capital_usd=3000.0,
            max_positions=4,
            per_position_weight=0.25,
            leverage=3.5,
            entry_spread_apr_pct=15.0,
            exit_spread_apr_pct=2.0,
            lookback_hours=24,
            exit_confirm_hours=6,
            min_hold_hours=12,
            fee_a_bps=1.5,
            fee_b_bps=2.0,
            slippage_per_leg_bps=0.75,
            params={
                "vol_target_leverage_enabled": True,
                "vol_lookback_hours": 48,
                "vol_target_ann_pct": 120.0,
                "min_symbol_leverage": 1.5,
                "protective_margin_exit_pct": 0.50,
                "protective_exit_slip_bps": 5.0,
            },
        )

        t0 = 1_790_000_000_000 - (1_790_000_000_000 % HOUR_MS)
        n_hours = 140
        candles_a: list[PriceBar] = []
        candles_b: list[PriceBar] = []
        fund_a: list[FundingRateBar] = []
        fund_b: list[FundingRateBar] = []

        px = 100.0
        for h in range(n_hours):
            ts = t0 + h * HOUR_MS
            # Deterministic oscillating hourly vol so vol-tiered leverage is strictly between min_sym_lev & max lev
            step_ret = 0.008 if (h % 2 == 0) else -0.0075
            op = px
            cl = op * math.exp(step_ret)
            hi = max(op, cl) * 1.002
            lo = min(op, cl) * 0.998
            # Inject an adverse upward spike on bar 45 (prev_bar at h=46) to trigger MARGIN_PROTECT_EXIT on SHORT_A!
            if h == 45:
                hi = op * 1.38
            px = cl

            candles_a.append(PriceBar("hyperliquid", "SOL", ts, op, hi, lo, cl, 1_000_000.0, PROVENANCE_REAL))
            candles_b.append(PriceBar("gate", "SOL", ts, op, hi, lo, cl, 1_000_000.0, PROVENANCE_REAL))

            # High positive funding spread h=0..90, collapsed h=91..139
            ra = 0.00008 if h <= 90 else 0.000001
            rb = 0.00001 if h <= 90 else 0.000001
            fund_a.append(FundingRateBar("hyperliquid", "SOL", ts, ra, 1.0, PROVENANCE_REAL))
            fund_b.append(FundingRateBar("gate", "SOL", ts, rb, 1.0, PROVENANCE_REAL))

        with tempfile.TemporaryDirectory() as td:
            store = MarketDataStore(Path(td) / "md.sqlite")
            try:
                store.upsert_candles(candles_a + candles_b)
                store.upsert_funding(fund_a + fund_b)
                bt_res = carry.evaluate(store, cfg)
            finally:
                store.close()

        # Stream the exact same hourly timeline through on_event()
        state: dict = {}
        stream_opens: list[tuple[int, str, float]] = []
        stream_closes: list[tuple[int, str]] = []
        for h in range(n_hours):
            ts = t0 + h * HOUR_MS
            # Backtester checks prev bar (h-1) high/low and cur bar open at decision time ts
            prev_ca = candles_a[h - 1] if h > 0 else candles_a[0]
            prev_cb = candles_b[h - 1] if h > 0 else candles_b[0]
            cur_ca = candles_a[h]
            cur_cb = candles_b[h]

            events_h: list[MarketEvent] = []
            if h > 0:
                # Closed bar h-1 candle provides high/low for margin guard & hourly log return under prev_ca.ts_ms
                # with decision timestamp end_ts_ms=ts (ZERO manual state mutation!)
                events_h.append(MarketEvent("CANDLE", prev_ca.ts_ms, "hyperliquid", "SOL", {
                    "open": prev_ca.open, "high": prev_ca.high, "low": prev_ca.low, "close": prev_ca.close, "end_ts_ms": ts,
                }))
                events_h.append(MarketEvent("CANDLE", prev_cb.ts_ms, "gate", "SOL", {
                    "open": prev_cb.open, "high": prev_cb.high, "low": prev_cb.low, "close": prev_cb.close, "end_ts_ms": ts,
                }))
            events_h.append(MarketEvent("BBO", ts, "hyperliquid", "SOL", {"bid": cur_ca.open, "ask": cur_ca.open}))
            events_h.append(MarketEvent("BBO", ts, "gate", "SOL", {"bid": cur_cb.open, "ask": cur_cb.open}))
            for ev in events_h:
                for intent in carry.on_event(ev, state, cfg):
                    if intent.action.startswith("OPEN_"):
                        stream_opens.append((intent.ts_ms, intent.action, intent.notional_usd))
                    elif intent.action == "CLOSE":
                        stream_closes.append((intent.ts_ms, intent.reason))

            # Deliver paired funding rates at hour ts (first venue defers decision until second venue arrives)
            for fa_ev in (
                MarketEvent("FUNDING", ts, "hyperliquid", "SOL", {"funding_rate": fund_a[h].funding_rate, "defer_decision": True}),
                MarketEvent("FUNDING", ts, "gate", "SOL", {"funding_rate": fund_b[h].funding_rate, "defer_decision": False}),
            ):
                for intent in carry.on_event(fa_ev, state, cfg):
                    if intent.action.startswith("OPEN_"):
                        stream_opens.append((intent.ts_ms, intent.action, intent.notional_usd))
                    elif intent.action == "CLOSE":
                        stream_closes.append((intent.ts_ms, intent.reason))

        # Backtester trades: Trade 1 enters at warmup_end (h=24), exits via MARGIN_PROTECT_EXIT at h=46;
        # cooldown blocks 24h -> Trade 2 re-enters at h=70, exits via HYSTERESIS_EXIT after funding collapses!
        self.assertGreaterEqual(len(bt_res.trades), 2)
        self.assertEqual(bt_res.trades[0].exit_reason, "MARGIN_PROTECT_EXIT")
        self.assertEqual(bt_res.trades[1].exit_reason, "HYSTERESIS_EXIT")
        self.assertEqual(len(stream_opens), len(bt_res.trades))
        self.assertEqual(len(stream_closes), len(bt_res.trades))

        for tr, (s_open_ts, _s_act, s_notional), (s_close_ts, s_reason) in zip(
            bt_res.trades, stream_opens, stream_closes
        ):
            self.assertEqual(s_open_ts, tr.entry_ts_ms)
            self.assertEqual(s_close_ts, tr.exit_ts_ms)
            self.assertEqual(s_reason, tr.exit_reason)
            self.assertAlmostEqual(s_notional, tr.notional_per_leg_usd, places=2)
            # Vol-tiered leverage MUST have scaled notional below raw 3.5x (=1312.5 USD)
            self.assertLess(s_notional, 3000.0 * 0.25 * 0.5 * 3.5)

    def test_trend_cta_bar_by_bar_parity_evaluate_vs_on_event(self) -> None:
        """Exact bar-by-bar parity: gap True Range, EMA filter, inverse-vol sizing, Chandelier stop & cooldown."""
        cta = TrendCTAPlugin()
        cfg = cta.default_config(
            symbols=["BTC"],
            strategy_id="parity_cta",
            version="v1",
            venue_a="hyperliquid",
            venue_b="hyperliquid",
            capital_usd=3000.0,
            max_positions=4,
            per_position_weight=0.25,
            leverage=2.0,
            fee_a_bps=1.5,
            slippage_per_leg_bps=2.0,
            execution_mode="TAKER_TAKER",
            params={
                "timeframe_hours": 12,
                "entry_n": 10,
                "exit_n": 5,
                "atr_n": 7,
                "atr_mult": 2.5,
                "ema_filter_n": 8,
                "risk_per_trade_pct": 1.0,
                "max_symbol_leverage": 1.5,
                "cooldown_hours": 24,
                "ref_venue": "gate",
            },
        )

        tf_ms = 12 * HOUR_MS
        t0 = 1_790_000_000_000 - (1_790_000_000_000 % tf_ms)
        # Construct 45 12h bars with gaps, trend breakout, Chandelier stop hit, cooldown, and reversal!
        bar_specs: list[tuple[float, float, float, float]] = []
        for b in range(45):
            if b < 14:
                # Tight flat consolidation around 100
                bar_specs.append((100.0, 101.0, 99.0, 100.2))
            elif b == 14:
                # Gap-up breakout above 101 AND above EMA
                bar_specs.append((103.0, 107.0, 102.5, 106.5))
            elif 15 <= b <= 19:
                # Uptrend continuation
                base = 107.0 + (b - 15) * 2.5
                bar_specs.append((base, base + 2.5, base - 0.5, base + 2.0))
            elif b == 20:
                # Sharp reversal wick hitting Chandelier stop!
                bar_specs.append((118.0, 118.5, 104.0, 106.0))
            elif 21 <= b <= 24:
                # Sideways during & after 24h (2-bar) cooldown
                bar_specs.append((106.0, 108.0, 105.0, 106.5))
            elif b == 25:
                # Downward breakdown below prior 10-bar low and below EMA
                bar_specs.append((105.0, 105.5, 94.0, 95.0))
            elif 26 <= b <= 32:
                # Downtrend continuation
                base = 94.0 - (b - 26) * 2.0
                bar_specs.append((base, base + 0.5, base - 2.0, base - 1.5))
            elif b == 33:
                # Donchian exit bounce above exit_n high
                bar_specs.append((83.0, 96.0, 82.5, 95.5))
            else:
                bar_specs.append((95.0, 96.0, 94.0, 95.0))

        candles_1h: list[PriceBar] = []
        for b_idx, (op, hi, lo, cl) in enumerate(bar_specs):
            bts = t0 + b_idx * tf_ms
            for h in range(12):
                h_op = op if h == 0 else cl
                h_hi = hi if h == 3 else max(h_op, cl)
                h_lo = lo if h == 6 else min(h_op, cl)
                candles_1h.append(
                    PriceBar("hyperliquid", "BTC", bts + h * HOUR_MS, h_op, h_hi, h_lo, cl, 500_000.0, PROVENANCE_REAL)
                )

        with tempfile.TemporaryDirectory() as td:
            store = MarketDataStore(Path(td) / "md.sqlite")
            try:
                store.upsert_candles(candles_1h)
                bt_res = cta.evaluate(store, cfg)
            finally:
                store.close()

        rbars = resample_ohlcv_1h(candles_1h, 12)
        state: dict = {}
        stream_opens: list[tuple[int, str, float]] = []
        stream_closes: list[tuple[int, str]] = []
        for rb in rbars:
            ev = MarketEvent(
                "CANDLE",
                rb.ts_ms,
                "hyperliquid",
                "BTC",
                {"open": rb.open, "high": rb.high, "low": rb.low, "close": rb.close, "end_ts_ms": rb.end_ts_ms},
            )
            for intent in cta.on_event(ev, state, cfg):
                if intent.action in ("OPEN_LONG", "OPEN_SHORT"):
                    stream_opens.append((intent.ts_ms, "LONG" if intent.action == "OPEN_LONG" else "SHORT", intent.notional_usd))
                elif intent.action == "CLOSE":
                    stream_closes.append((intent.ts_ms, intent.reason))

        non_eow_trades = [t for t in bt_res.trades if t.exit_reason != "END_OF_WINDOW"]
        self.assertGreaterEqual(len(non_eow_trades), 2)
        self.assertEqual(non_eow_trades[0].exit_reason, "CHANDELIER_STOP")
        self.assertEqual(len(stream_opens), len(bt_res.trades))
        self.assertEqual(len(stream_closes), len(non_eow_trades))

        for tr, (s_ts, s_dir, s_notional) in zip(bt_res.trades, stream_opens):
            self.assertEqual(s_ts, tr.entry_ts_ms)
            self.assertEqual(s_dir, tr.direction)
            self.assertAlmostEqual(s_notional, tr.notional_per_leg_usd, places=2)

        for tr, (c_ts, c_reason) in zip(non_eow_trades, stream_closes):
            self.assertEqual(c_ts, tr.exit_ts_ms)
            self.assertEqual(c_reason, tr.exit_reason)

    def test_shadow_auditor_int_map_roundtrip_and_funding_carry_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            store = MarketDataStore(data_dir / "market_data.sqlite")
            auditor = ShadowParityAuditor(data_dir / "parity.sqlite", store)
            try:
                sample_state = {
                    "funding_hist": {("hyperliquid", "BTC"): {1_795_000_000_000: 0.0001}},
                    "hourly_log_rets": {"BTC": {1_795_000_000_000: 0.005, 1_795_003_600_000: -0.004}},
                    "open_pos": {},
                }
                auditor.save_runtime_state(
                    "s1",
                    started_ts_ms=1_795_000_000_000,
                    realized_pnl_usd=0.0,
                    plans_created=0,
                    plans_blocked=0,
                    open_positions={},
                    updated_ts_ms=1_795_003_600_000,
                    plugin_state=sample_state,
                )
                loaded = auditor.load_all_runtime_states()["s1"]["plugin_state"]
                f_keys = list(loaded["funding_hist"][("hyperliquid", "BTC")].keys())
                r_keys = list(loaded["hourly_log_rets"]["BTC"].keys())
                self.assertTrue(all(isinstance(k, int) for k in f_keys))
                self.assertTrue(all(isinstance(k, int) for k in r_keys))
            finally:
                auditor.close()
                store.close()

            # Verify FundingCarry warmup excludes current incomplete hour, populates vol returns,
            # AND automatically refreshes hourly_log_rets when clock advances to next hour!
            t_now = 1_795_400_000_000 - (1_795_400_000_000 % HOUR_MS) + 120_000
            hour_floor = (t_now // HOUR_MS) * HOUR_MS
            store2 = MarketDataStore(data_dir / "market_data.sqlite")
            try:
                candles = [
                    PriceBar("hyperliquid", "BTC", hour_floor - k * HOUR_MS, 100.0, 101.0, 99.0, 100.0 * math.exp(0.01 if k % 2 else -0.01), 100.0, PROVENANCE_REAL)
                    for k in range(1, 40)
                ]
                # Current incomplete hour bar (ts == hour_floor) MUST be excluded until next hour!
                candles.append(PriceBar("hyperliquid", "BTC", hour_floor, 100.0, 102.0, 98.0, 101.5, 100.0, PROVENANCE_REAL))
                store2.upsert_candles(candles)
            finally:
                store2.close()

            bootstrap_balanced_shadow_configs(data_dir)
            daemon = ShadowPortfolioDaemon(data_dir, max_bbo_stale_sec=300.0)
            try:
                ticks = [
                    BBOTick("hyperliquid", "BTC", t_now - 1_000, 100.0, 100.02, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", t_now - 1_000, 100.0, 100.02, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                ]
                daemon.run_once(now_ms=t_now, live_ticks=ticks)
                pstate = daemon._plugin_states["hl_gate_safe_carry_prod"]
                rmap = pstate.get("hourly_log_rets", {}).get("BTC", {})
                self.assertNotIn(hour_floor, rmap, "Current open hour bar MUST be excluded from vol warmup!")
                self.assertGreaterEqual(len(rmap), 30)
                loaded_pair = daemon.watcher.get_loaded("hl_gate_safe_carry_prod")
                self.assertIsNotNone(loaded_pair)
                assert loaded_pair is not None
                carry_cfg = loaded_pair[0]
                eff_lev = compute_causal_symbol_leverage(carry_cfg, rmap, t_now)
                self.assertGreaterEqual(eff_lev, 1.5)
                self.assertLess(eff_lev, 3.5)

                # Advance clock by 1 hour: previous hour_floor candle is now closed and MUST refresh automatically!
                t_next = t_now + HOUR_MS
                ticks2 = [
                    BBOTick("hyperliquid", "BTC", t_next - 1_000, 101.5, 101.52, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", t_next - 1_000, 101.5, 101.52, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                ]
                daemon.run_once(now_ms=t_next, live_ticks=ticks2)
                self.assertIn(hour_floor, daemon._plugin_states["hl_gate_safe_carry_prod"]["hourly_log_rets"]["BTC"])
            finally:
                daemon.close()

    def test_nightly_factory_seed_balanced_runs_regressions_without_shadow_live(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            now_ms = utc_now_ms()
            store = MarketDataStore(data_dir / "market_data.sqlite")
            store.upsert_candles([
                PriceBar("hyperliquid", "BTC", now_ms - 60_000, 60000.0, 60100.0, 59900.0, 60050.0, 10.0, PROVENANCE_REAL)
            ])
            store.close()

            with mock.patch.dict(os.environ, {"ALPHALOOP_WATCHDOG_URL": "http://127.0.0.1:8794/api/v1/heartbeat"}):
                with mock.patch.object(WatchdogHook, "_default_http_post", return_value=(True, "OK")):
                    rep1 = run_nightly_factory(
                        data_dir,
                        backfill_days=1,
                        skip_fetch=True,
                        run_evolution=False,
                        seed_balanced=True,
                    )
                    rep2 = run_nightly_factory(
                        data_dir,
                        backfill_days=1,
                        skip_fetch=True,
                        run_evolution=False,
                        seed_balanced=True,
                    )

            self.assertTrue(rep1["ok"])
            self.assertEqual(set(rep1["seeded_strategies"]), {"hl_trend_cta_prod", "hl_gate_safe_carry_prod"})
            self.assertEqual(len(rep1["regressions"]), 2)
            self.assertEqual(rep2["seeded_strategies"], [])
            self.assertEqual(len(rep2["regressions"]), 2)
            # Confirm seeded status on research worker is BACKTESTED, NOT SHADOW_LIVE, with distinct version tag
            cstore = HotConfigStore(data_dir / "state")
            try:
                for sid in ("hl_trend_cta_prod", "hl_gate_safe_carry_prod"):
                    c = cstore.read_active(sid)
                    self.assertIsNotNone(c)
                    assert c is not None
                    self.assertEqual(c.status, "BACKTESTED")
                    self.assertEqual(c.version, "v2.2.1-nightly")
            finally:
                cstore.close()

            # Verify Shadow bootstrap can subsequently force_reseed SHADOW_LIVE on the SAME data_dir without collision!
            reseeded = bootstrap_balanced_shadow_configs(data_dir, status="SHADOW_LIVE", force_reseed=True)
            self.assertEqual(set(reseeded), {"hl_trend_cta_prod", "hl_gate_safe_carry_prod"})

    def test_strict_param_schema_across_registry_watcher_and_overrides(self) -> None:
        import json
        from alphaloop.execution.bridge import ExecutionBridge

        carry = FundingCarryPlugin()
        cta = TrendCTAPlugin()
        # Reject unknown top-level StrategyConfig overrides
        with self.assertRaises(ValueError):
            carry.default_config(symbols=["BTC"], bogus_top_field=123)
        with self.assertRaises(ValueError):
            cta.default_config(symbols=["BTC"], unknown_override="bad")
        # Reject non-dict params ([], "", 0) and stringified bool "false"
        for bad_params in ([], "", 0, {"vol_target_leverage_enabled": "false"}, {"vol_lookback_hours": True}):
            with self.assertRaises(ValueError):
                carry.default_config(symbols=["BTC"], params=bad_params)  # type: ignore[arg-type]
        for bad_cta_params in ([], "", {"entry_n": True}, {"ref_venue": 123}):
            with self.assertRaises(ValueError):
                cta.default_config(symbols=["BTC"], params=bad_cta_params)  # type: ignore[arg-type]

        # HotConfigStore rejects dead param keys fail-closed; HotConfigWatcher trips CONFIG_CORRUPT on tampered file!
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            cstore = HotConfigStore(state_dir)
            bridge = ExecutionBridge(state_dir)
            try:
                bad_cfg = carry.default_config(symbols=["BTC"], strategy_id="bad_carry", status="SHADOW_LIVE")
                bad_cfg.params["vol_tier_high_ann_pct"] = 130.0
                with self.assertRaises(ValueError):
                    cstore.register_version(bad_cfg, actor="test", reason="must fail")

                # Simulate pre-A1 legacy JSON file sitting on disk in active_configs/
                legacy_path = state_dir / "active_configs" / "bad_carry.json"
                legacy_path.write_text(json.dumps({"config": bad_cfg.to_dict()}), encoding="utf-8")
                evs = bridge.watcher.poll()
                self.assertTrue(any(e["status"] == "CORRUPT_KILL_ENGAGED" for e in evs))
                self.assertTrue(bridge.kill_switch.is_engaged())
                self.assertIsNone(bridge.watcher.get_loaded("bad_carry"))
            finally:
                bridge.close()
                cstore.close()

    def test_rolling_apr_subhour_gate8h_and_mixed_int_str_dedup(self) -> None:
        from alphaloop.harness.backtester import compute_rolling_funding_apr

        h0 = 1_795_000_000_000 - (1_795_000_000_000 % HOUR_MS)
        # Gate 8h settlement right at boundary h0 - 95h MUST NOT be truncated by sub-hour BBO tick h0 + 30s!
        boundary_ts = h0 - 95 * HOUR_MS
        rates = {
            boundary_ts: 0.0008,
            str(boundary_ts): 0.0008,  # Duplicate string key must NOT double-count!
            h0: 0.0004,
        }
        apr_exact = compute_rolling_funding_apr(rates, h0, 96)
        apr_subhour = compute_rolling_funding_apr(rates, h0 + 30_000, 96)
        expected = ((0.0008 + 0.0004) / 96.0) * 24.0 * 365.0 * 100.0
        self.assertIsNotNone(apr_exact)
        self.assertIsNotNone(apr_subhour)
        assert apr_exact is not None and apr_subhour is not None
        self.assertAlmostEqual(apr_exact, expected, places=6)
        self.assertAlmostEqual(apr_subhour, expected, places=6)

    def test_shadow_portfolio_cross_symbol_edge_ranking_and_prestart_no_replay(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            cstore = HotConfigStore(data_dir / "state")
            carry = FundingCarryPlugin()
            cfg = carry.default_config(
                symbols=["ADA", "SOL"],
                strategy_id="rank_carry",
                version="v1",
                status="SHADOW_LIVE",
                venue_a="hyperliquid",
                venue_b="gate",
                capital_usd=3000.0,
                max_positions=1,
                per_position_weight=1.0,
                leverage=2.0,
                entry_spread_apr_pct=15.0,
                exit_spread_apr_pct=2.0,
                lookback_hours=4,
                exit_confirm_hours=2,
                min_hold_hours=0,
                slippage_per_leg_bps=2.0,
            )
            cstore.register_version(cfg, actor="test", reason="rank test")
            cstore.close()

            h_now = 1_795_400_000_000 - (1_795_400_000_000 % HOUR_MS)
            store = MarketDataStore(data_dir / "market_data.sqlite")
            fbars: list[FundingRateBar] = []
            for k in range(6, 0, -1):
                ts_k = h_now - k * HOUR_MS
                # Historical bars: ADA has moderate spread (50% APR), SOL has much higher spread (200% APR)
                fbars.append(FundingRateBar("hyperliquid", "ADA", ts_k, 0.00006, 1.0, PROVENANCE_REAL))
                fbars.append(FundingRateBar("gate", "ADA", ts_k, 0.0, 1.0, PROVENANCE_REAL))
                fbars.append(FundingRateBar("hyperliquid", "SOL", ts_k, 0.00024, 1.0, PROVENANCE_REAL))
                fbars.append(FundingRateBar("gate", "SOL", ts_k, 0.0, 1.0, PROVENANCE_REAL))
            store.upsert_funding(fbars)
            store.close()

            daemon = ShadowPortfolioDaemon(data_dir, max_bbo_stale_sec=300.0)
            try:
                t_tick = h_now + 10_000
                ticks = [
                    BBOTick("hyperliquid", "ADA", t_tick, 0.50, 0.5001, 100.0, 100.0, 5.0, PROVENANCE_REAL),
                    BBOTick("gate", "ADA", t_tick, 0.50, 0.5001, 100.0, 100.0, 5.0, PROVENANCE_REAL),
                    BBOTick("hyperliquid", "SOL", t_tick, 150.0, 150.03, 10.0, 10.0, 5.0, PROVENANCE_REAL),
                    BBOTick("gate", "SOL", t_tick, 150.0, 150.03, 10.0, 10.0, 5.0, PROVENANCE_REAL),
                ]
                rep = daemon.run_once(now_ms=t_tick + 1_000, live_ticks=ticks)
                self.assertEqual(rep["status"], "PARITY_OK")
                # Only 1 position opened (max_positions=1) and SOL MUST beat alphabetical ADA!
                self.assertEqual(rep["strategies"][0]["plans_created"], 1)
                self.assertEqual(set(daemon._open_positions["rank_carry"].keys()), {"SOL"})
            finally:
                daemon.close()

    def test_prewindow_warmed_parity_auditor_and_emergency_close_spread_exemption(self) -> None:
        from alphaloop.core.models import OrderIntent
        from alphaloop.execution.bridge import ExecutionBridge

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            store = MarketDataStore(data_dir / "market_data.sqlite")
            auditor = ShadowParityAuditor(data_dir / "parity.sqlite", store)
            bridge = ExecutionBridge(data_dir / "state")
            try:
                cta = TrendCTAPlugin()
                cfg = cta.default_config(
                    symbols=["BTC"],
                    strategy_id="prewarm_cta",
                    version="v1",
                    status="SHADOW_LIVE",
                    venue_a="hyperliquid",
                    venue_b="hyperliquid",
                    params={"timeframe_hours": 12, "entry_n": 10, "exit_n": 5, "atr_n": 7, "atr_mult": 2.5},
                )
                tf_ms = 12 * HOUR_MS
                win_start = 1_795_400_000_000 - (1_795_400_000_000 % tf_ms)
                win_end = win_start + 3 * tf_ms

                # Without pre-window history, auditor MUST mark backtest_comparable=False (zero false PARITY_DRIFT_KILL!)
                rec_cold = auditor.evaluate_strategy(
                    cfg,
                    window_start_ms=win_start,
                    window_end_ms=win_end,
                    shadow_equity_usd=3100.0,
                    live_half_spreads_bps=[1.0],
                    plans_created=1,
                    plans_blocked=0,
                    open_positions=0,
                    bbo_freshness_sec=5.0,
                    audit_chain_valid=True,
                    kill_switch_engaged=False,
                    drift_check_min_hours=1.0,
                )
                self.assertFalse(rec_cold.backtest_comparable)
                self.assertEqual(rec_cold.status, "PARITY_OK")
                self.assertIn("INSUFFICIENT_PREWINDOW_WARMUP", rec_cold.detail)

                # Seed 20 completed pre-window 12h bars + 3 in-window 12h bars -> backtest_comparable=True!
                candles = []
                for b in range(-20, 3):
                    bts = win_start + b * tf_ms
                    for h in range(12):
                        candles.append(
                            PriceBar("hyperliquid", "BTC", bts + h * HOUR_MS, 100.0, 101.0, 99.0, 100.0, 1e5, PROVENANCE_REAL)
                        )
                store.upsert_candles(candles)
                rec_warm = auditor.evaluate_strategy(
                    cfg,
                    window_start_ms=win_start,
                    window_end_ms=win_end,
                    shadow_equity_usd=3000.0,
                    live_half_spreads_bps=[1.0],
                    plans_created=0,
                    plans_blocked=0,
                    open_positions=0,
                    bbo_freshness_sec=5.0,
                    audit_chain_valid=True,
                    kill_switch_engaged=False,
                    drift_check_min_hours=1.0,
                )
                self.assertTrue(rec_warm.backtest_comparable)
                self.assertEqual(rec_warm.status, "PARITY_OK")

                # Emergency CHANDELIER_STOP CLOSE must pass through wide 150bp crash spread while OPEN_LONG is blocked!
                wide_tick = BBOTick("hyperliquid", "BTC", win_end, 100.0, 101.5, 1.0, 1.0, 5.0, PROVENANCE_REAL)
                plan_open = bridge.submit_intent(
                    cfg,
                    OrderIntent("prewarm_cta", "BTC", "OPEN_LONG", 500.0, 1.0, "DONCHIAN_LONG_BREAKOUT", win_end),
                    {"hyperliquid": wide_tick},
                    now_ms=win_end,
                )
                self.assertEqual(plan_open.status, "BLOCKED")
                self.assertIn("SPREAD_CEILING_EXCEEDED", plan_open.block_reason)

                plan_stop = bridge.submit_intent(
                    cfg,
                    OrderIntent("prewarm_cta", "BTC", "CLOSE", 500.0, 0.0, "CHANDELIER_STOP", win_end),
                    {"hyperliquid": wide_tick},
                    now_ms=win_end,
                )
                self.assertEqual(plan_stop.status, "PLANNED_SHADOW")
            finally:
                bridge.close()
                auditor.close()
                store.close()

    def test_web_console_readonly_mode_rejects_post_with_405_and_allow_header(self) -> None:
        import http.client
        import threading
        from alphaloop.web.server import create_server

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {"ALPHALOOP_CONSOLE_READONLY": "1"}):
                server, ctx = create_server(data_dir=Path(td), host="127.0.0.1", port=0)
                port = int(server.server_address[1])
                th = threading.Thread(target=server.serve_forever, daemon=True)
                th.start()
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    conn.request("GET", "/api/health")
                    r_get = conn.getresponse()
                    r_get.read()
                    self.assertEqual(r_get.status, 200)

                    conn.request("POST", "/api/agent/run-cycle", body="{}", headers={"Content-Type": "application/json"})
                    r_post = conn.getresponse()
                    r_post.read()
                    self.assertEqual(r_post.status, 405)
                    self.assertEqual(r_post.getheader("Allow"), "GET")
                    conn.close()
                finally:
                    server.shutdown()
                    server.server_close()
                    ctx.close()

    def test_nightly_factory_flock_blocks_concurrent_and_excludes_bbo_ticks_from_freshness(self) -> None:
        import fcntl

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            now_ms = utc_now_ms()
            # Only fresh BBO tick (NO funding or candles_1h) -> fresh_ok MUST be False and ok MUST be False!
            store = MarketDataStore(data_dir / "market_data.sqlite")
            store.upsert_bbo_ticks([BBOTick("hyperliquid", "BTC", now_ms - 1_000, 60000.0, 60001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL)])
            store.close()

            rep = run_nightly_factory(data_dir, skip_fetch=True, seed_balanced=False)
            self.assertFalse(rep["fresh_ok"])
            self.assertFalse(rep["ok"])

            # Verify flock rejects concurrent run fail-fast
            lock_p = data_dir / "state" / "nightly_factory.lock"
            lock_p.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_p, "a+", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    with self.assertRaises(RuntimeError):
                        run_nightly_factory(data_dir, skip_fetch=True, lock_path=lock_p)
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def test_cta_restart_evicts_stale_active_bucket_without_double_commit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            cstore = HotConfigStore(data_dir / "state")
            cta = TrendCTAPlugin()
            cfg = cta.default_config(
                symbols=["BTC"],
                strategy_id="cta_evict_test",
                version="v1",
                status="SHADOW_LIVE",
                venue_a="hyperliquid",
                venue_b="hyperliquid",
                params={"timeframe_hours": 12, "entry_n": 10, "exit_n": 5, "atr_n": 7, "atr_mult": 2.5},
            )
            cstore.register_version(cfg, actor="test", reason="evict test")
            cstore.close()

            tf_ms = 12 * HOUR_MS
            b0 = 1_795_400_000_000 - (1_795_400_000_000 % tf_ms)
            store = MarketDataStore(data_dir / "market_data.sqlite")
            candles = []
            # 15 completed buckets before b0 + 2 closed 1h bars inside b0
            for b in range(-15, 0):
                bts = b0 + b * tf_ms
                px = 100.0 + b * 0.5
                for h in range(12):
                    candles.append(PriceBar("hyperliquid", "BTC", bts + h * HOUR_MS, px, px + 1.0, px - 1.0, px, 1e5, PROVENANCE_REAL))
            for h in range(2):
                candles.append(PriceBar("hyperliquid", "BTC", b0 + h * HOUR_MS, 102.0, 103.0, 101.0, 102.0, 1e5, PROVENANCE_REAL))
            store.upsert_candles(candles)
            store.close()

            # Daemon 1 runs at b0 + 2h + 5m: persists active_bucket with bts=b0
            t1 = b0 + 2 * HOUR_MS + 300_000
            d1 = ShadowPortfolioDaemon(data_dir, max_bbo_stale_sec=300.0)
            try:
                d1.run_once(
                    now_ms=t1,
                    live_ticks=[BBOTick("hyperliquid", "BTC", t1 - 1_000, 102.0, 102.1, 1.0, 1.0, 5.0, PROVENANCE_REAL)],
                )
                st1 = d1._plugin_states["cta_evict_test"]
                self.assertEqual(st1["bars_seen"]["BTC"], 15)
                self.assertEqual(int(st1["active_bucket"]["BTC"]["bts"]), b0)
            finally:
                d1.close()

            # 24h downtime: b0 (all 12h) and b0+12h (all 12h) complete; NO closed 1h candle in b0+24h yet!
            store = MarketDataStore(data_dir / "market_data.sqlite")
            more = []
            for h in range(12):
                more.append(PriceBar("hyperliquid", "BTC", b0 + h * HOUR_MS, 102.0, 104.0, 101.0, 103.0, 1e5, PROVENANCE_REAL))
            for h in range(12):
                more.append(PriceBar("hyperliquid", "BTC", b0 + tf_ms + h * HOUR_MS, 103.0, 106.0, 102.5, 105.0, 1e5, PROVENANCE_REAL))
            store.upsert_candles(more)
            store.close()

            # Daemon 2 restarts at b0 + 24h + 5m (cur_open_closed_1h is EMPTY!)
            b_cur = b0 + 2 * tf_ms
            t2 = b_cur + 300_000
            d2 = ShadowPortfolioDaemon(data_dir, max_bbo_stale_sec=300.0)
            try:
                d2.run_once(
                    now_ms=t2,
                    live_ticks=[BBOTick("hyperliquid", "BTC", t2 - 1_000, 105.5, 105.6, 1.0, 1.0, 5.0, PROVENANCE_REAL)],
                )
                st2 = d2._plugin_states["cta_evict_test"]
                # Exactly 15 + 2 = 17 completed bars (stale b0 was evicted, NEVER double-committed!)
                self.assertEqual(st2["bars_seen"]["BTC"], 17)
                self.assertAlmostEqual(st2["ohlc_hist"]["BTC"][-1][2], 105.0, places=6)
                self.assertEqual(int(st2["active_bucket"]["BTC"]["bts"]), b_cur)
            finally:
                d2.close()

    def test_carry_pass2_defers_decision_until_both_bbo_legs_sync_blocking_adverse_basis(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            cstore = HotConfigStore(data_dir / "state")
            carry = FundingCarryPlugin()
            cfg = carry.default_config(
                symbols=["BTC"],
                strategy_id="basis_sync_carry",
                version="v1",
                status="SHADOW_LIVE",
                venue_a="hyperliquid",
                venue_b="gate",
                capital_usd=3000.0,
                max_positions=1,
                per_position_weight=1.0,
                leverage=2.0,
                entry_spread_apr_pct=15.0,
                exit_spread_apr_pct=2.0,
                max_basis_entry_bps=40.0,
                stop_loss_basis_bps=100.0,
                lookback_hours=4,
                exit_confirm_hours=2,
                min_hold_hours=0,
                slippage_per_leg_bps=2.0,
            )
            cstore.register_version(cfg, actor="test", reason="basis sync test")
            cstore.close()

            h_now = 1_795_400_000_000 - (1_795_400_000_000 % HOUR_MS)
            store = MarketDataStore(data_dir / "market_data.sqlite")
            fbars = []
            # Initial flat funding (0 spread) so Cycle 1 caches synchronized BBOs HL=100.0, Gate=100.0
            for k in range(6, 0, -1):
                ts_k = h_now - k * HOUR_MS
                fbars.append(FundingRateBar("hyperliquid", "BTC", ts_k, 0.0, 1.0, PROVENANCE_REAL))
                fbars.append(FundingRateBar("gate", "BTC", ts_k, 0.0, 1.0, PROVENANCE_REAL))
            store.upsert_funding(fbars)
            store.close()

            daemon = ShadowPortfolioDaemon(data_dir, max_bbo_stale_sec=300.0)
            try:
                t1 = h_now + 10_000
                rep1 = daemon.run_once(
                    now_ms=t1,
                    live_ticks=[
                        BBOTick("hyperliquid", "BTC", t1 - 1_000, 99.99, 100.01, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                        BBOTick("gate", "BTC", t1 - 1_000, 99.99, 100.01, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    ],
                )
                self.assertEqual(rep1["strategies"][0]["plans_created"], 0)

                # Hour +1: massive +200% APR spread, BUT live BBO(gate)=100.8 creates 79.4bp adverse basis (>40bp limit!)
                h_next = h_now + HOUR_MS
                store2 = MarketDataStore(data_dir / "market_data.sqlite")
                store2.upsert_funding([
                    FundingRateBar("hyperliquid", "BTC", h_next, 0.0012, 1.0, PROVENANCE_REAL),
                    FundingRateBar("gate", "BTC", h_next, 0.0, 1.0, PROVENANCE_REAL),
                ])
                store2.close()

                t2 = h_next + 10_000
                rep2 = daemon.run_once(
                    now_ms=t2,
                    live_ticks=[
                        BBOTick("hyperliquid", "BTC", t2 - 1_000, 99.99, 100.01, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                        BBOTick("gate", "BTC", t2 - 1_000, 100.79, 100.81, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    ],
                )
                # Pass 2 MUST defer until BOTH BBOs update -> blocks adverse 79.4bp entry cleanly!
                self.assertEqual(rep2["strategies"][0]["plans_created"], 0)
                self.assertEqual(daemon._open_positions.get("basis_sync_carry", {}), {})

                # Cycle 3: basis tightens to 10bp (<=40bp) -> enters cleanly
                t3 = h_next + 20_000
                rep3 = daemon.run_once(
                    now_ms=t3,
                    live_ticks=[
                        BBOTick("hyperliquid", "BTC", t3 - 1_000, 99.99, 100.01, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                        BBOTick("gate", "BTC", t3 - 1_000, 100.09, 100.11, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    ],
                )
                self.assertEqual(rep3["strategies"][0]["plans_created"], 1)
                self.assertIn("BTC", daemon._open_positions["basis_sync_carry"])

                # Cycle 4: BOTH venues rally +3% together (103.0 / 103.1) -> Pass-1 BBO defer_decision prevents false BASIS_STOP!
                t4 = h_next + 30_000
                rep4 = daemon.run_once(
                    now_ms=t4,
                    live_ticks=[
                        BBOTick("hyperliquid", "BTC", t4 - 1_000, 102.99, 103.01, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                        BBOTick("gate", "BTC", t4 - 1_000, 103.09, 103.11, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    ],
                )
                self.assertEqual(rep4["strategies"][0]["plans_created"], 1)
                self.assertIn("BTC", daemon._open_positions["basis_sync_carry"])
            finally:
                daemon.close()

    def test_carry_cold_start_empty_funding_recovers_warmup_without_96h_lockout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            cstore = HotConfigStore(data_dir / "state")
            carry = FundingCarryPlugin()
            cfg = carry.default_config(
                symbols=["BTC"],
                strategy_id="cold_start_carry",
                version="v1",
                status="SHADOW_LIVE",
                venue_a="hyperliquid",
                venue_b="gate",
                capital_usd=3000.0,
                max_positions=1,
                per_position_weight=1.0,
                leverage=2.0,
                entry_spread_apr_pct=15.0,
                exit_spread_apr_pct=2.0,
                lookback_hours=96,
                exit_confirm_hours=2,
                min_hold_hours=0,
                slippage_per_leg_bps=2.0,
            )
            cstore.register_version(cfg, actor="test", reason="cold start test")
            cstore.close()

            h_now = 1_795_400_000_000 - (1_795_400_000_000 % HOUR_MS)
            daemon = ShadowPortfolioDaemon(data_dir, max_bbo_stale_sec=300.0)
            try:
                # Cycle 1: cold start with ZERO funding rows in SQLite!
                t1 = h_now + 10_000
                rep1 = daemon.run_once(
                    now_ms=t1,
                    live_ticks=[
                        BBOTick("hyperliquid", "BTC", t1 - 1_000, 99.99, 100.01, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                        BBOTick("gate", "BTC", t1 - 1_000, 99.99, 100.01, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    ],
                )
                self.assertEqual(rep1["strategies"][0]["plans_created"], 0)
                self.assertNotIn(("cold_start_carry", "hyperliquid", "BTC"), daemon._last_delivered_funding_ts)
                self.assertNotIn(("cold_start_carry", "gate", "BTC"), daemon._last_delivered_funding_ts)

                # Backfill completes before Cycle 2: 100h of real historical funding up to h_now
                store = MarketDataStore(data_dir / "market_data.sqlite")
                fbars = []
                for k in range(99, -1, -1):
                    ts_k = h_now - k * HOUR_MS
                    fbars.append(FundingRateBar("hyperliquid", "BTC", ts_k, 0.0002, 1.0, PROVENANCE_REAL))
                    fbars.append(FundingRateBar("gate", "BTC", ts_k, 0.0, 1.0, PROVENANCE_REAL))
                store.upsert_funding(fbars)
                store.close()

                # Cycle 2: automatically recovers silent warmup & immediately enters live (zero 96h lockout, zero phantom replay!)
                t2 = h_now + 40_000
                rep2 = daemon.run_once(
                    now_ms=t2,
                    live_ticks=[
                        BBOTick("hyperliquid", "BTC", t2 - 1_000, 99.99, 100.01, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                        BBOTick("gate", "BTC", t2 - 1_000, 99.99, 100.01, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    ],
                )
                self.assertEqual(daemon._last_delivered_funding_ts[("cold_start_carry", "hyperliquid", "BTC")], h_now)
                self.assertEqual(daemon._last_delivered_funding_ts[("cold_start_carry", "gate", "BTC")], h_now)
                self.assertEqual(rep2["strategies"][0]["plans_created"], 1)
                self.assertIn("BTC", daemon._open_positions["cold_start_carry"])
            finally:
                daemon.close()


if __name__ == "__main__":
    unittest.main()
