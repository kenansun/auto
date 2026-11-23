"""Comprehensive Unit & Adversarial Tests for Stage 1, Stage 2 (S2-02) & Stage 3 (S3-01, S3-04)."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
import unittest
import urllib.request
from decimal import Decimal
from pathlib import Path
from unittest import mock

from alphaloop.agent_loop.model_router import MultiModelRouter
from alphaloop.agent_loop.orchestrator import AgentEvolutionOrchestrator
from alphaloop.core.config import HotConfigStore
from alphaloop.core.models import BBOTick, FundingRateBar, PriceBar
from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.daemon.shadow_runner import (
    DAEMON_LIVE_EXECUTION_ENABLED,
    LIVE_EXECUTION_ENABLED,
    ShadowPortfolioDaemon,
)
from alphaloop.daemon.watchdog_hook import AlertNotifier, WatchdogHook
from alphaloop.data.store import MarketDataStore
from alphaloop.execution.arb_executor_adapter import (
    ADAPTER_LIVE_EXECUTION_ENABLED,
    to_arb_audit_record_envelope,
    to_arb_plan_risk_envelope,
)
from alphaloop.execution.bridge import ExecutionPlan, LegSpec
from alphaloop.harness.backtester import FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier, TrialLedger
from alphaloop.plugins import FundingCarryPlugin, TrendCTAPlugin, default_registry
from alphaloop.web.server import create_server
from scripts.ingest_usdc_dip_audit import evaluate_usdc_dip_csv
from scripts.run_shadow_daemon import bootstrap_balanced_shadow_configs


class TestPhase3ShadowDaemonAdapterAndLLM(unittest.TestCase):
    def test_watchdog_hook_rejects_stale_and_future_and_bounds_queue(self) -> None:
        now = [1_795_000_000_000]
        sent: list[dict] = []

        def transport(_url: str, payload: dict, _timeout: float) -> tuple[bool, str]:
            sent.append(payload)
            return True, "OK"

        hook = WatchdogHook(
            endpoint_url="http://127.0.0.1:8794",
            max_stale_ms=60_000,
            max_future_ms=10_000,
            async_mode=False,
            clock_ms_fn=lambda: now[0],
            transport=transport,
        )
        try:
            ok, reason = hook.heartbeat("feed_a", now[0] - 5_000, seq=1, meta={"ok": True, "nan": float("nan")})
            self.assertTrue(ok)
            self.assertEqual(reason, "OK")
            self.assertEqual(len(sent), 1)
            self.assertIn("metrics", sent[0])
            self.assertNotIn("nan", sent[0]["metrics"])

            # Stale payload rejected fail-closed
            ok_stale, r_stale = hook.heartbeat("feed_a", now[0] - 120_000, seq=2)
            self.assertFalse(ok_stale)
            self.assertIn("STALE_EVENT_TS", r_stale)

            # Future payload rejected fail-closed
            ok_fut, r_fut = hook.heartbeat("feed_a", now[0] + 60_000, seq=3)
            self.assertFalse(ok_fut)
            self.assertIn("FUTURE_EVENT_TS", r_fut)

            # Invalid scheme rejected
            with self.assertRaises(ValueError):
                WatchdogHook(endpoint_url="file:///etc/passwd", async_mode=False)
        finally:
            hook.close()

        # AlertNotifier cooldown dedup prevents alert storms & does not burn cooldown on sink failure
        emitted: list[dict] = []
        fail_next = [True]

        def flaky_sink(rec: dict) -> None:
            if fail_next[0]:
                raise RuntimeError("sink down")
            emitted.append(rec)

        notifier = AlertNotifier(cooldown_sec=1800.0, clock_ms_fn=lambda: now[0], sink=flaky_sink)
        self.assertEqual(notifier.emit("P0", "SPREAD_KILL", "t0", "d0")[0], False)
        fail_next[0] = False
        self.assertEqual(notifier.emit("P0", "SPREAD_KILL", "t1", "d1")[0], True)
        self.assertEqual(notifier.emit("P0", "SPREAD_KILL", "t2", "d2"), (False, "COOLDOWN_SUPPRESSED"))
        now[0] += 1_801_000
        self.assertEqual(notifier.emit("P0", "SPREAD_KILL", "t3", "d3")[0], True)
        self.assertEqual(len(emitted), 2)

    def test_shadow_daemon_stale_halt_wide_spread_kill_and_corrupt_config_kill(self) -> None:
        self.assertFalse(LIVE_EXECUTION_ENABLED)
        self.assertFalse(DAEMON_LIVE_EXECUTION_ENABLED)
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            seeded = bootstrap_balanced_shadow_configs(data_dir)
            self.assertEqual(set(seeded), {"hl_trend_cta_prod", "hl_gate_safe_carry_prod"})

            t0 = 1_795_400_000_000
            heartbeats: list[dict] = []
            hook = WatchdogHook(
                endpoint_url="http://127.0.0.1:8794",
                async_mode=False,
                clock_ms_fn=lambda: t0 + 10_000,
                transport=lambda _u, p, _t: (heartbeats.append(p) is None, "OK"),
            )
            daemon = ShadowPortfolioDaemon(
                data_dir,
                watchdog=hook,
                max_bbo_stale_sec=60.0,
                max_spread_ratio=1.5,
                wide_spread_kill_sec=60.0,
            )
            try:
                # 1) Missing BBO -> DEGRADED_HALT and ZERO watchdog heartbeats
                r0 = daemon.run_once(now_ms=t0)
                self.assertEqual(r0["status"], "DEGRADED_HALT")
                self.assertEqual(len(heartbeats), 0)

                # 2) Fresh tight BBO -> PARITY_OK + valid heartbeat + atomic parity report written
                fresh_ticks = [
                    BBOTick("hyperliquid", "BTC", t0 + 5_000, 60000.0, 60003.0, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", t0 + 5_000, 60000.0, 60004.0, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                ]
                r1 = daemon.run_once(now_ms=t0 + 10_000, live_ticks=fresh_ticks)
                self.assertEqual(r1["status"], "PARITY_OK")
                self.assertFalse(r1["live_execution_enabled"])
                self.assertTrue(r1["audit_chain_valid"])
                self.assertEqual(len(heartbeats), 1)
                self.assertTrue((data_dir / "shadow_parity_report.json").exists())

                # 3) Stale BBO (>60s old) -> DEGRADED_HALT
                r_stale = daemon.run_once(now_ms=t0 + 120_000)
                self.assertEqual(r_stale["status"], "DEGRADED_HALT")

                # 4) Sustained wide spread (>1.5x assumed 0.75bp) for >= 60s -> KillSwitch SLIPPAGE_EXCEEDED
                wide_t1 = t0 + 130_000
                wide_ticks_1 = [
                    BBOTick("hyperliquid", "BTC", wide_t1, 59900.0, 60100.0, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", wide_t1, 59900.0, 60100.0, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                ]
                r_w1 = daemon.run_once(now_ms=wide_t1 + 1_000, live_ticks=wide_ticks_1)
                self.assertEqual(r_w1["status"], "WARN_WIDE_SPREAD")
                self.assertFalse(r_w1["kill_switch"]["engaged"])

                wide_t2 = wide_t1 + 65_000
                wide_ticks_2 = [
                    BBOTick("hyperliquid", "BTC", wide_t2, 59900.0, 60100.0, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", wide_t2, 59900.0, 60100.0, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                ]
                r_w2 = daemon.run_once(now_ms=wide_t2 + 1_000, live_ticks=wide_ticks_2)
                self.assertEqual(r_w2["status"], "KILL_TRIPPED")
                self.assertTrue(r_w2["kill_switch"]["engaged"])
                self.assertEqual(r_w2["kill_switch"]["trigger"], "SLIPPAGE_EXCEEDED")

                # 5) Corrupt active config -> CONFIG_CORRUPT kill engaged & deduped on repeated polls
                corrupt_path = data_dir / "state" / "active_configs" / "evil_bad.json"
                corrupt_path.write_text("{corrupt-json", encoding="utf-8")
                ev1 = daemon.bridge.watcher.poll()
                ev2 = daemon.bridge.watcher.poll()
                self.assertTrue(any(e["status"] == "CORRUPT_KILL_ENGAGED" for e in ev1))
                self.assertEqual(ev2, [])
            finally:
                daemon.close()
                hook.close()

    def test_shadow_daemon_real_carry_open_funding_close_and_restart_hydration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            cfg_store = HotConfigStore(data_dir / "state")
            carry = FundingCarryPlugin()
            cfg = carry.default_config(
                symbols=["BTC"],
                strategy_id="test_live_carry",
                version="v1",
                venue_a="hyperliquid",
                venue_b="gate",
                capital_usd=3000.0,
                max_positions=2,
                leverage=2.0,
                entry_spread_apr_pct=15.0,
                exit_spread_apr_pct=2.0,
                lookback_hours=1,
                exit_confirm_hours=2,
                min_hold_hours=0,
                slippage_per_leg_bps=0.75,
            )
            cfg = cfg.__class__(**{**cfg.to_dict(), "status": "SHADOW_LIVE"})
            cfg_store.register_version(cfg, actor="test", reason="shadow e2e")
            cfg_store.close()

            store = MarketDataStore(data_dir / "market_data.sqlite")
            t0 = 1_795_400_000_000
            # Seed current-hour wide positive spread funding (HL +0.0005/1h = 438% APR vs Gate +0.0001/8h = 10.95% APR)
            store.upsert_funding([
                FundingRateBar("hyperliquid", "BTC", t0, 0.0005, 1.0, PROVENANCE_REAL),
                FundingRateBar("gate", "BTC", t0, 0.0001, 8.0, PROVENANCE_REAL),
            ])
            store.close()

            daemon = ShadowPortfolioDaemon(data_dir, max_bbo_stale_sec=300.0)
            try:
                ticks1 = [
                    BBOTick("hyperliquid", "BTC", t0, 60000.0, 60003.0, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", t0, 60000.0, 60003.0, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                ]
                r1 = daemon.run_once(now_ms=t0 + 1_000, live_ticks=ticks1)
                self.assertEqual(r1["status"], "PARITY_OK")
                strat1 = r1["strategies"][0]
                self.assertGreaterEqual(strat1["plans_created"], 1)
                self.assertEqual(strat1["open_positions"], 1)
            finally:
                daemon.close()

            # Restart daemon -> hydrate open position, accrue funding, and trigger CLOSE after 3 collapsed hours
            daemon2 = ShadowPortfolioDaemon(data_dir, max_bbo_stale_sec=300.0)
            try:
                r2: dict = {}
                for step in range(1, 5):
                    ts_step = t0 + step * 3_600_000
                    store_step = MarketDataStore(data_dir / "market_data.sqlite")
                    store_step.upsert_funding([
                        FundingRateBar("hyperliquid", "BTC", ts_step, 0.000001, 1.0, PROVENANCE_REAL),
                        FundingRateBar("gate", "BTC", ts_step, 0.000001, 8.0, PROVENANCE_REAL),
                    ])
                    store_step.close()
                    ticks_step = [
                        BBOTick("hyperliquid", "BTC", ts_step, 60010.0, 60013.0, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                        BBOTick("gate", "BTC", ts_step, 60010.0, 60013.0, 1.0, 1.0, 10.0, PROVENANCE_REAL),
                    ]
                    r2 = daemon2.run_once(now_ms=ts_step + 1_000, live_ticks=ticks_step)
                strat2 = r2["strategies"][0]
                self.assertEqual(strat2["open_positions"], 0)
                self.assertGreaterEqual(strat2["plans_created"], 2)
                self.assertTrue(r2["audit_chain_valid"])
            finally:
                daemon2.close()

    def test_shadow_daemon_cta_candle_warmup_populates_ohlc_hist_and_triggers_breakout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            cfg_store = HotConfigStore(data_dir / "state")
            cta = TrendCTAPlugin()
            cfg = cta.default_config(
                symbols=["BTC"],
                strategy_id="test_cta_warmup",
                version="v1",
                venue_a="hyperliquid",
                venue_b="hyperliquid",
                capital_usd=3000.0,
                max_positions=2,
                leverage=2.0,
                execution_mode="TAKER_TAKER",
                params={
                    "timeframe_hours": 12,
                    "entry_n": 10,
                    "exit_n": 5,
                    "atr_n": 6,
                    "atr_mult": 3.0,
                    "risk_per_trade_pct": 0.8,
                    "max_symbol_leverage": 1.5,
                },
            )
            cfg = cfg.__class__(**{**cfg.to_dict(), "status": "SHADOW_LIVE"})
            cfg_store.register_version(cfg, actor="test", reason="cta warmup")
            cfg_store.close()

            # Seed 15 completed 12h buckets (=180 hourly bars) in a tight 100..102 range + 1 incomplete current bucket
            tf_ms = 12 * 3_600_000
            bucket_now = (1_795_400_000_000 // tf_ms) * tf_ms
            candles = []
            for b_idx in range(15, 0, -1):
                b_start = bucket_now - b_idx * tf_ms
                for h in range(12):
                    candles.append(
                        PriceBar("hyperliquid", "BTC", b_start + h * 3_600_000, 100.0, 102.0, 99.0, 100.5, 100.0, PROVENANCE_REAL)
                    )
            # Incomplete current bucket bar that MUST be excluded from warmup!
            candles.append(PriceBar("hyperliquid", "BTC", bucket_now, 100.0, 999.0, 99.0, 999.0, 100.0, PROVENANCE_REAL))
            store = MarketDataStore(data_dir / "market_data.sqlite")
            store.upsert_candles(candles)
            store.close()

            daemon = ShadowPortfolioDaemon(data_dir, max_bbo_stale_sec=300.0)
            try:
                # Tick 1 inside bucket_now starts the live bar; Tick 2 in bucket_now + tf_ms closes bucket_now at breakout 120.0!
                t_break = bucket_now + 60_000
                daemon.run_once(
                    now_ms=t_break + 1_000,
                    live_ticks=[BBOTick("hyperliquid", "BTC", t_break, 120.0, 120.02, 1.0, 1.0, 10.0, PROVENANCE_REAL)],
                )
                hist = daemon._plugin_states["test_cta_warmup"].get("ohlc_hist", {}).get("BTC", [])
                self.assertEqual(len(hist), 15)
                self.assertTrue(all(isinstance(x, tuple) and len(x) == 3 and x[0] <= 102.0 for x in hist))

                # Advance to next 12h bucket so breakout bar closes and triggers OPEN_LONG immediately!
                t_next = bucket_now + tf_ms + 60_000
                r_brk = daemon.run_once(
                    now_ms=t_next + 1_000,
                    live_ticks=[BBOTick("hyperliquid", "BTC", t_next, 121.0, 121.02, 1.0, 1.0, 10.0, PROVENANCE_REAL)],
                )
                self.assertGreaterEqual(r_brk["strategies"][0]["plans_created"], 1)
                self.assertEqual(r_brk["strategies"][0]["open_positions"], 1)
            finally:
                daemon.close()

    def test_nightly_factory_skip_fetch_runs_cleanly(self) -> None:
        from alphaloop.core.provenance import utc_now_ms
        from scripts.run_nightly_factory import run_nightly_factory

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            cfg_store = HotConfigStore(data_dir / "state")
            cta = TrendCTAPlugin()
            cfg = cta.default_config(symbols=["BTC"], strategy_id="nf_cta", version="v1", venue_a="hyperliquid", venue_b="hyperliquid")
            cfg = cfg.__class__(**{**cfg.to_dict(), "status": "SHADOW_LIVE"})
            cfg_store.register_version(cfg, actor="test", reason="nightly test")
            cfg_store.close()

            now_ms = utc_now_ms()
            store = MarketDataStore(data_dir / "market_data.sqlite")
            store.upsert_candles([PriceBar("hyperliquid", "BTC", now_ms - 60_000, 60000.0, 60100.0, 59900.0, 60050.0, 10.0, PROVENANCE_REAL)])
            store.close()

            with mock.patch.dict(os.environ, {"ALPHALOOP_WATCHDOG_URL": "http://127.0.0.1:8794/api/v1/heartbeat"}):
                with mock.patch.object(WatchdogHook, "_default_http_post", return_value=(True, "OK")):
                    rep = run_nightly_factory(data_dir, backfill_days=1, skip_fetch=True, run_evolution=False)
            self.assertIn("dq_summary", rep)
            self.assertEqual(len(rep["regressions"]), 1)
            self.assertNotIn("error", rep["regressions"][0])
            self.assertEqual(rep["watchdog"]["accepted"], 1)
            self.assertTrue((data_dir / "nightly_factory_report.json").exists())

    def test_web_console_v22_shadow_parity_endpoint_and_zero_onclick(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            bootstrap_balanced_shadow_configs(data_dir)
            t0 = 1_795_400_000_000
            daemon = ShadowPortfolioDaemon(data_dir)
            try:
                daemon.run_once(
                    now_ms=t0 + 1_000,
                    live_ticks=[
                        BBOTick("hyperliquid", "PAXG", t0, 3600.0, 3600.5, 2.0, 2.0, 10.0, PROVENANCE_REAL),
                        BBOTick("gate", "PAXG", t0, 3600.0, 3600.5, 2.0, 2.0, 10.0, PROVENANCE_REAL),
                    ],
                )
            finally:
                daemon.close()

            srv, ctx = create_server(data_dir, host="127.0.0.1", port=0)
            port = srv.server_address[1]
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            try:
                html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode("utf-8")
                self.assertNotIn("onclick=", html.lower())
                self.assertIn("shadowParityBox", html)

                health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5).read())
                self.assertEqual(health["service"], "alphaloop-v2.2")

                parity = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/shadow/parity", timeout=5).read())
                self.assertTrue(parity["valid"])
                self.assertFalse(parity["live_execution_enabled"])
                self.assertEqual(parity["status"], "PARITY_OK")

                # DNS rebinding Host header rejection (403)
                req_bad_host = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/health",
                    headers={"Host": "evil.attacker.example"},
                )
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(req_bad_host, timeout=5)
                self.assertEqual(cm.exception.code, 403)
            finally:
                srv.shutdown()
                srv.server_close()
                ctx.close()

    def test_arb_executor_adapter_strict_decimal_and_adversarial_rejections(self) -> None:
        self.assertFalse(ADAPTER_LIVE_EXECUTION_ENABLED)
        t0 = 1_795_400_000_000
        good_plan = ExecutionPlan(
            plan_id="SHD-0123456789abcdef",
            idempotency_key="a" * 64,
            run_mode="SHADOW_DRY_RUN",
            status="PLANNED_SHADOW",
            block_reason="",
            ts_ms=t0,
            strategy_id="hl_trend_cta_prod",
            version="v2.2",
            config_sha256="b" * 64,
            symbol="BTC",
            action="OPEN_LONG",
            notional_usd=450.25,
            expected_edge_bps=35.0,
            legs=[
                LegSpec(
                    leg_id="LEG_A",
                    venue="hyperliquid",
                    symbol="BTC",
                    side="BUY",
                    notional_usd=450.25,
                    ref_price=61234.5,
                    limit_price=61240.0,
                    spread_bps=0.5,
                    bbo_ts_ms=t0 - 1000,
                )
            ],
        )
        env = to_arb_plan_risk_envelope(good_plan, now_ms=t0, max_quote_age_ms=10_000)
        self.assertFalse(env["live_execution_enabled"])
        self.assertEqual(Decimal(env["notional_quote"]), Decimal("450.25"))
        self.assertEqual(env["quote_prices"], ("61234.5",))
        self.assertEqual(env["created_utc_ns"], t0 * 1_000_000)

        # Reject BLOCKED plan, stale quote, NaN price, mismatched symbol, extreme exponent, non-shadow audit
        blocked = ExecutionPlan(**{**good_plan.__dict__, "status": "BLOCKED", "block_reason": "KILL_ENGAGED"})
        with self.assertRaises(ValueError):
            to_arb_plan_risk_envelope(blocked, now_ms=t0)
        with self.assertRaises(ValueError):
            to_arb_plan_risk_envelope(good_plan, now_ms=t0 + 60_000, max_quote_age_ms=10_000)
        nan_leg = LegSpec("LEG_A", "hyperliquid", "BTC", "BUY", 450.0, float("nan"), 61000.0, 0.5, t0)
        with self.assertRaises(ValueError):
            to_arb_plan_risk_envelope(ExecutionPlan(**{**good_plan.__dict__, "legs": [nan_leg]}), now_ms=t0)
        bad_sym_leg = LegSpec("LEG_A", "hyperliquid", "ETH", "BUY", 450.0, 3000.0, 3001.0, 0.5, t0)
        with self.assertRaises(ValueError):
            to_arb_plan_risk_envelope(ExecutionPlan(**{**good_plan.__dict__, "legs": [bad_sym_leg]}), now_ms=t0)
        huge_leg = LegSpec("LEG_A", "hyperliquid", "BTC", "BUY", Decimal("1e50"), 60000.0, 60001.0, 0.5, t0)
        with self.assertRaises(ValueError):
            to_arb_plan_risk_envelope(ExecutionPlan(**{**good_plan.__dict__, "legs": [huge_leg]}), now_ms=t0)

        # Audit record masks normalized secret keys & rejects non-shadow mode
        with self.assertRaises(ValueError):
            to_arb_audit_record_envelope({"ts_ms": t0, "event_type": "X", "run_mode": "LIVE_WIRE"})

        audit_env = to_arb_audit_record_envelope({
            "ts_ms": t0,
            "event_type": "PLAN_CREATED",
            "plan_id": good_plan.plan_id,
            "idempotency_key": good_plan.idempotency_key,
            "record_hash": "c" * 64,
            "payload": {
                "symbol": "BTC",
                "X-Api-Key": "super_secret_123",
                "Authorization": "Bearer xyz",
                "nested": {"private_key": "0xabc"},
            },
        })
        self.assertEqual(audit_env["utc_ns"], t0 * 1_000_000)
        self.assertEqual(audit_env["payload"]["X-Api-Key"], "***REDACTED***")
        self.assertEqual(audit_env["payload"]["Authorization"], "***REDACTED***")
        self.assertEqual(audit_env["payload"]["nested"]["private_key"], "***REDACTED***")

    def test_pillar3_usdc_dip_gate_waits_under_30_and_evaluates_at_35(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            small_csv = td_path / "small.csv"
            with small_csv.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["ts_iso", "event_id", "track", "symbol", "chain", "pool", "side", "t_offset_s", "raw_net_bp", "extra_cost_bp", "realized_net_bp", "verdict", "detail"])
                for i in range(5):
                    w.writerow([f"2026-11-22T0{i}:00:00Z", f"E{i}", "USDC_DIP_INSTANT", "USDC", "base", "p", "BUY", 0, 22.0, 2.0, 20.0, "OK", ""])
            # Without notional column or explicit default_notional_usd -> strictly skips missing notional (zero hidden $500!)
            rep_no_notional = evaluate_usdc_dip_csv(small_csv, data_dir=td_path, min_samples=30)
            self.assertEqual(rep_no_notional["valid_samples"], 0)
            self.assertEqual(rep_no_notional["skipped_missing_notional"], 5)
            self.assertEqual(rep_no_notional["provenance"], "UNVERIFIED_LOCAL_CSV")

            # With explicit notional assumption -> parses all 5 ts_iso rows and flags notional_assumed=True
            rep_small = evaluate_usdc_dip_csv(small_csv, data_dir=td_path, min_samples=30, default_notional_usd=1000.0)
            self.assertFalse(rep_small["gate_eligible"])
            self.assertIn("WAITING_FOR_SAMPLES (5/30)", rep_small["status"])
            self.assertTrue(rep_small["notional_assumed"])

            big_csv = td_path / "big.csv"
            with big_csv.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["ts_ms", "strategy", "symbol", "notional_usd", "edge_bps", "net_pnl_usd"])
                for i in range(35):
                    w.writerow([1_795_000_000_000 + i * 60_000, "USDC_DIP_INSTANT", "USDC", 1000, 18.0, 1.0])
            rep_big = evaluate_usdc_dip_csv(big_csv, data_dir=td_path, min_samples=30, provenance="SERVER_COLLECTED_LIVE")
            self.assertTrue(rep_big["gate_eligible"])
            self.assertTrue(rep_big["passed_event_gates"])
            self.assertGreater(rep_big["ex_top5_net_pnl_usd"], 0.0)
            self.assertGreater(rep_big["oos_30pct_net_usd"], 0.0)
            self.assertGreaterEqual(rep_big["trial_id"], 1)

    def test_model_router_persistent_daily_budget_and_orchestrator_llm_fault_sanitization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "llm_budget.sqlite"
            router = MultiModelRouter(
                max_cycle_budget_usd=5.0,
                max_daily_budget_usd=0.02,
                budget_db_path=str(db_path),
            )
            try:
                # 0) Invalid ALPHALOOP_AI_GATEWAY_URL rejected fail-closed
                with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-dummy-key", "ALPHALOOP_AI_GATEWAY_URL": "file:///etc/passwd"}):
                    out_url, tag_url = router.invoke_json("QUANT_HYPOTHESIS", "sys", {"a": 1})
                    self.assertIsNone(out_url)
                    self.assertIn("invalid_gateway_url", tag_url)

                # 1) Simulate successful call that spends $0.025 (exceeding $0.02 daily cap for next call)
                with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-dummy-key"}, clear=False):
                    os.environ.pop("ALPHALOOP_AI_GATEWAY_URL", None)
                    with mock.patch.object(
                        router,
                        "_call_openai_compat",
                        return_value=('{"param_changes":{"leverage":2.5}}', 20_000, 10_000),
                    ):
                        out1, tag1 = router.invoke_json("QUANT_HYPOTHESIS", "sys", {"a": 1})
                        self.assertIsInstance(out1, dict)
                        self.assertIn("deepseek/", tag1)

                    # Re-open a fresh router pointing to same SQLite DB -> daily cap MUST persist across restarts!
                    router2 = MultiModelRouter(
                        max_cycle_budget_usd=5.0,
                        max_daily_budget_usd=0.02,
                        budget_db_path=str(db_path),
                    )
                    try:
                        out2, tag2 = router2.invoke_json("QUANT_HYPOTHESIS", "sys", {"a": 2})
                        self.assertIsNone(out2)
                        self.assertIn("daily_budget_cap_reached", tag2)
                    finally:
                        router2.close()

                    # 2) Malformed JSON -> deterministic fallback; zero usage still charges conservative byte fallback
                    router_bad = MultiModelRouter(max_cycle_budget_usd=5.0, max_daily_budget_usd=5.0)
                    with mock.patch.object(router_bad, "_call_openai_compat", return_value=("not-json{{{", 0, 0)):
                        out_bad, tag_bad = router_bad.invoke_json("QUANT_HYPOTHESIS", "sys", {"a": 1})
                        self.assertIsNone(out_bad)
                        self.assertIn("api_error", tag_bad)
                        self.assertGreater(router_bad.daily_spent_usd(), 0.0)

                    # 3) Out-of-bounds / unknown LLM param_changes -> rejected fail-closed in orchestrator!
                    store = MarketDataStore(Path(td) / "mkt.sqlite")
                    ledger = TrialLedger(Path(td) / "ledger.sqlite")
                    cfg_store = HotConfigStore(Path(td) / "state")
                    reg = default_registry()
                    bt = FundingCarryBacktester(store)
                    ver = SixGateVerifier(bt, ledger, registry=reg)
                    orch = AgentEvolutionOrchestrator(Path(td) / "state", bt, ver, cfg_store, router_bad, registry=reg)
                    try:
                        plugin = TrendCTAPlugin()
                        cfg = plugin.default_config(symbols=["BTC"])
                        fb = plugin.diagnose_telemetry(cfg, plugin.evaluate(store, cfg).metrics)
                        with mock.patch.object(
                            router_bad,
                            "invoke_json",
                            return_value=({"param_changes": {"leverage": 999.0, "evil_key": 1}}, "deepseek/mock"),
                        ):
                            prop = orch._propose_mutation(plugin, cfg, fb, plugin.evaluate(store, cfg).metrics)
                            self.assertIn("sanitized_reject", prop.model_used)
                            self.assertNotIn("evil_key", prop.param_changes)
                    finally:
                        orch.close()
                        cfg_store.close()
                        ledger.close()
                        store.close()
            finally:
                router.close()


if __name__ == "__main__":
    unittest.main()
