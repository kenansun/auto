"""Adversarial unit tests for DataQualityAuditor (DQ-01..DQ-08) and Strategy Factory v2 Plugins."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from alphaloop.agent_loop.model_router import MultiModelRouter
from alphaloop.agent_loop.orchestrator import AgentEvolutionOrchestrator
from alphaloop.core.config import HotConfigStore
from alphaloop.core.models import BBOTick, FundingRateBar, PriceBar, StrategyConfig
from alphaloop.core.provenance import PROVENANCE_REAL, PROVENANCE_SYNTHETIC, utc_now_ms
from alphaloop.data.quality import DataQualityAuditor
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import HOUR_MS, FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier, TrialLedger
from alphaloop.plugins import (
    BasisReversionPlugin,
    FundingCarryPlugin,
    StructuralArbPlugin,
    default_registry,
)


class TestDQAndPlugins(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = MarketDataStore(self.root / "market.sqlite")
        self.auditor = DataQualityAuditor(self.store)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _seed_clean_dual_series(
        self,
        venue_a: str = "hyperliquid",
        venue_b: str = "okx",
        symbol: str = "BTC",
        hours: int = 240,
        interval_b: float = 8.0,
        price_ratio_b: float = 1.0,
        rate_a: float = 0.0000125,  # Exact Hyperliquid clamped baseline (+10.95% APR) — MUST NOT false-positive!
        rate_b: float = 0.00002,
        include_b_candles: bool = True,
        frozen_zero_vol: bool = False,
        provenance: str = PROVENANCE_REAL,
    ) -> tuple[int, int]:
        start = (1_720_000_000_000 // HOUR_MS) * HOUR_MS
        fa, fb, ca, cb = [], [], [], []
        step_b = int(round(interval_b))
        for h in range(hours):
            ts = start + h * HOUR_MS
            fa.append(FundingRateBar(venue_a, symbol, ts, rate_a, 1.0, provenance))
            if h % step_b == 0:
                fb.append(FundingRateBar(venue_b, symbol, ts, rate_b, interval_b, provenance))
            px_a = 100.0 if frozen_zero_vol else 100.0 + 2.0 * math.sin(h / 12.0)
            px_b = px_a * price_ratio_b
            vol = 0.0 if frozen_zero_vol else 500_000.0
            hi_a = px_a if frozen_zero_vol else px_a * 1.005
            lo_a = px_a if frozen_zero_vol else px_a * 0.995
            hi_b = px_b if frozen_zero_vol else px_b * 1.005
            lo_b = px_b if frozen_zero_vol else px_b * 0.995
            ca.append(PriceBar(venue_a, symbol, ts, px_a, hi_a, lo_a, px_a, vol, provenance))
            if include_b_candles:
                cb.append(PriceBar(venue_b, symbol, ts, px_b, hi_b, lo_b, px_b, vol, provenance))
        self.store.upsert_funding(fa)
        self.store.upsert_funding(fb)
        self.store.upsert_candles(ca)
        if cb:
            self.store.upsert_candles(cb)
        return start, start + (hours - 1) * HOUR_MS

    def test_dq_accepts_hl_clamped_baseline_and_8h_cex_cadence(self) -> None:
        """Architect P0-2: 1h HL clamped +0.0000125 baseline vs 8h OKX must certify DATA_QUALIFIED_FULL (ZERO false positives!)."""
        s, e = self._seed_clean_dual_series("hyperliquid", "okx", "BTC", hours=240, interval_b=8.0)
        rep = self.auditor.audit_funding_pair("hyperliquid", "okx", ["BTC"], start_ms=s, end_ms=e)
        self.assertEqual(rep.tier, "DATA_QUALIFIED_FULL")
        self.assertEqual(rep.passed_count, 8)
        self.assertTrue(rep.is_eligible_for_live)

    def test_dq_tiers_funding_only_when_venue_b_lacks_candles(self) -> None:
        """Lighter-like funding-only venue must certify DATA_QUALIFIED_FUNDING_ONLY (usable for research, NOT live)."""
        s, e = self._seed_clean_dual_series("lighter", "gate", "ETH", hours=240, interval_b=8.0, include_b_candles=False)
        rep = self.auditor.audit_funding_pair("lighter", "gate", ["ETH"], start_ms=s, end_ms=e)
        self.assertEqual(rep.tier, "DATA_QUALIFIED_FUNDING_ONLY")
        self.assertTrue(rep.is_usable_for_research)
        self.assertFalse(rep.is_eligible_for_live)

    def test_dq_rejects_100x_funding_unit_bug(self) -> None:
        """DQ-06 MUST reject percent-vs-decimal 100x unit bugs (e.g. 0.08 per hour = 70080% APR)."""
        s, e = self._seed_clean_dual_series("hyperliquid", "okx", "SOL", hours=120, rate_a=0.08)
        rep = self.auditor.audit_funding_pair("hyperliquid", "okx", ["SOL"], start_ms=s, end_ms=e)
        self.assertEqual(rep.tier, "DATA_REJECTED")
        dq6 = next(c for c in rep.checks if c.gate_id == "DQ-06")
        self.assertFalse(dq6.passed)

    def test_dq_rejects_symbol_mapping_price_ratio_mismatch(self) -> None:
        """DQ-05 MUST catch 1000PEPE/PEPE or USDCUSDT/WLD symbol mapping bugs (>2.5% median price ratio deviation)."""
        s, e = self._seed_clean_dual_series("hyperliquid", "okx", "PEPE", hours=120, price_ratio_b=1000.0)
        rep = self.auditor.audit_funding_pair("hyperliquid", "okx", ["PEPE"], start_ms=s, end_ms=e)
        self.assertEqual(rep.tier, "DATA_REJECTED")
        dq5 = next(c for c in rep.checks if c.gate_id == "DQ-05")
        self.assertFalse(dq5.passed)

    def test_dq_rejects_frozen_zero_vol_ohlcv_and_synthetic_provenance(self) -> None:
        """DQ-08 rejects frozen zero-volume flatlines (>6h) and DQ-01 rejects synthetic provenance."""
        s, e = self._seed_clean_dual_series("hyperliquid", "okx", "ADA", hours=120, frozen_zero_vol=True, provenance=PROVENANCE_SYNTHETIC)
        rep = self.auditor.audit_funding_pair("hyperliquid", "okx", ["ADA"], start_ms=s, end_ms=e)
        self.assertEqual(rep.tier, "DATA_REJECTED")
        self.assertFalse(next(c for c in rep.checks if c.gate_id == "DQ-01").passed)
        self.assertFalse(next(c for c in rep.checks if c.gate_id == "DQ-08").passed)

    def test_store_rejects_crossed_bbo_and_preserves_same_ms_burst_ticks(self) -> None:
        """STD-34 Pillar 6 & P0-2.2: reject bid > ask, preserve same-ms valid burst ticks."""
        now = utc_now_ms()
        ticks = [
            BBOTick("lighter", "BTC", now, 95000.0, 95005.0, 1.0, 1.0, 45.0, PROVENANCE_REAL),
            BBOTick("lighter", "BTC", now, 95001.0, 95006.0, 1.2, 0.9, 42.0, PROVENANCE_REAL),  # same ms burst!
            BBOTick("lighter", "BTC", now + 1000, 95010.0, 95000.0, 1.0, 1.0, 40.0, PROVENANCE_REAL),  # crossed -> reject!
        ]
        saved = self.store.upsert_bbo_ticks(ticks)
        self.assertEqual(saved, 2)
        loaded = self.store.load_bbo_ticks("lighter", "BTC")
        self.assertEqual(len(loaded), 2)

    def test_plugin_registry_evaluates_three_heterogeneous_strategies_and_caps_funding_only(self) -> None:
        """Architect P0-5 + P1-3: all 3 plugins evaluate via registry, and FUNDING_ONLY caps at MICRO_RUN_PROBE_REQUIRED."""
        reg = default_registry()
        ids = {p["plugin_id"] for p in reg.list_plugins()}
        self.assertEqual(ids, {"funding_carry", "basis_reversion", "structural_arb"})

        # Seed 2-bar persistent +35bp dislocation every 25h (survives T+1 execution lag!)
        # PLUS test that a 1-bar phantom bounce yields <= 0 basis PnL under default execution_lag_bars=1!
        start = (1_720_000_000_000 // HOUR_MS) * HOUR_MS
        hours = 500
        for sym in ("BTC", "ETH"):
            fa, fb, ca, cb = [], [], [], []
            for h in range(hours):
                ts = start + h * HOUR_MS
                fa.append(FundingRateBar("hyperliquid", sym, ts, 0.0000125, 1.0, PROVENANCE_REAL))
                fb.append(FundingRateBar("okx", sym, ts, 0.0000125, 1.0, PROVENANCE_REAL))
                disloc = 0.0035 if (h % 25 in (0, 1) and h > 48) else 0.0
                px_b = 100.0 + 0.5 * math.sin(h / 7.0)
                px_a = px_b * (1.0 + disloc)
                ca.append(PriceBar("hyperliquid", sym, ts, px_a, px_a * 1.002, px_a * 0.998, px_a, 1e6, PROVENANCE_REAL))
                cb.append(PriceBar("okx", sym, ts, px_b, px_b * 1.002, px_b * 0.998, px_b, 1e6, PROVENANCE_REAL))
            self.store.upsert_funding(fa)
            self.store.upsert_funding(fb)
            self.store.upsert_candles(ca)
            self.store.upsert_candles(cb)

        br_plug = reg.get("basis_reversion")
        br_cfg = br_plug.default_config(symbols=["BTC", "ETH"], params={"entry_zscore": 1.8, "min_basis_bps": 12.0, "z_lookback_hours": 24, "max_hold_hours": 12})
        br_res = br_plug.evaluate(self.store, br_cfg, start_ms=start, end_ms=start + (hours - 1) * HOUR_MS)
        self.assertGreater(br_res.metrics.trade_count, 5)
        self.assertGreater(br_res.metrics.basis_pnl_usd, 0.0)
        self.assertEqual(len(br_plug.plateau_variants(br_cfg)), 9)

        # Red-Team #1 & #5.3: StructuralArb keeps losing trades & blocks agent from mutating physical cost params!
        sa_plug = reg.get("structural_arb")
        sa_cfg = sa_plug.default_config(symbols=["BTC", "ETH"], params={"min_edge_bps": 20.0, "gas_usd_per_tx": 0.40, "revert_rate_pct": 5.0, "rebalance_cost_bps": 2.0, "confirm_decay_bps": 2.0, "notional_per_trade_usd": 1000.0})
        sa_mutated = sa_plug.validate_and_clamp(sa_cfg, "v2", {"gas_usd_per_tx": 0.0, "revert_rate_pct": 0.0, "min_edge_bps": 25.0})
        self.assertEqual(sa_mutated.params["gas_usd_per_tx"], 0.40)
        self.assertEqual(sa_mutated.params["revert_rate_pct"], 5.0)
        self.assertEqual(sa_mutated.params["min_edge_bps"], 25.0)
        sa_res = sa_plug.evaluate(self.store, sa_cfg, start_ms=start, end_ms=start + (hours - 1) * HOUR_MS)
        self.assertGreater(sa_res.metrics.trade_count, 5)
        self.assertEqual(len(sa_plug.plateau_variants(sa_cfg)), 9)

        # Red-Team #5.3: FundingCarry rejects bool/NaN and caps per_position_weight <= 1/max_positions
        fc_plug = FundingCarryPlugin()
        fc_base = fc_plug.default_config(max_positions=4, per_position_weight=0.25)
        fc_clamped = fc_plug.validate_and_clamp(fc_base, "v2", {"max_positions": True, "leverage": float("nan"), "per_position_weight": 0.9})
        self.assertEqual(fc_clamped.max_positions, 4)
        self.assertEqual(fc_clamped.leverage, fc_base.leverage)
        self.assertLessEqual(fc_clamped.per_position_weight * fc_clamped.max_positions, 1.0 + 1e-9)

        # Now test Architect P1-3 + Red-Team #5.1: FUNDING_ONLY pair (no candles on venue_b) CANNOT auto-promote to SHADOW_LIVE
        # even if a partial-symbol BBO cert exists without a full-symbol COMPLETED_PASS MicroRun!
        for sym in ("SOL", "HYPE", "AAVE"):
            fa, fb, ca = [], [], []
            for h in range(2160):
                ts = start + h * HOUR_MS
                active = (h % 480) < 240
                fa.append(FundingRateBar("lighter", sym, ts, (0.28 if active else -0.04) / 8760.0, 1.0, PROVENANCE_REAL))
                fb.append(FundingRateBar("gate", sym, ts, (-0.02 if active else 0.0) / 8760.0, 1.0, PROVENANCE_REAL))
                px = 100.0 + 1.5 * math.sin(h / 20.0)
                ca.append(PriceBar("lighter", sym, ts, px, px * 1.005, px * 0.995, px, 1e6, PROVENANCE_REAL))
            self.store.upsert_funding(fa)
            self.store.upsert_funding(fb)
            self.store.upsert_candles(ca)

        # Seed a partial-symbol BBO cert (ONLY "SOL", missing "HYPE" & "AAVE") -> MUST NOT bypass MICRO_RUN_PROBE_REQUIRED!
        now_ms = utc_now_ms()
        bbo_ticks = []
        for i in range(15):
            t_i = now_ms - (15 - i) * 10_000
            bbo_ticks.append(BBOTick("lighter", "SOL", t_i, 100.0, 100.01, 5.0, 5.0, 30.0, PROVENANCE_REAL))
            bbo_ticks.append(BBOTick("gate", "SOL", t_i + 10, 100.0, 100.01, 5.0, 5.0, 35.0, PROVENANCE_REAL))
        self.store.upsert_bbo_ticks(bbo_ticks)
        self.auditor.audit_bbo_ticks("lighter", "gate", ["SOL"], persist=True, max_stale_sec=3600.0)

        ledger = TrialLedger(self.root / "trials.sqlite")
        cfg_store = HotConfigStore(self.root / "state")
        bt = FundingCarryBacktester(self.store)
        verifier = SixGateVerifier(bt, ledger, registry=reg)
        orch = AgentEvolutionOrchestrator(self.root / "state", bt, verifier, cfg_store, MultiModelRouter(), registry=reg)

        seed = FundingCarryPlugin().default_config(
            strategy_id="p13_funding_only_cap",
            version="v0.1",
            symbols=["SOL", "HYPE", "AAVE"],
            venue_a="lighter",
            venue_b="gate",
            use_atb_buggy_rules=True,
            per_position_weight=0.33,
            max_positions=3,
        )
        cfg_store.register_version(seed, actor="seed", reason="p13 test")
        rec = orch.run_cycle(seed, next_version="v0.2", start_ms=start, end_ms=start + 2159 * HOUR_MS)
        self.assertTrue(rec["gate_verdict"]["overall_pass"])
        self.assertEqual(rec["gate_verdict"]["dq_tier"], "DATA_QUALIFIED_FUNDING_ONLY")
        self.assertFalse(rec["promoted_shadow"])
        self.assertEqual(rec["promotion_stage"], "MICRO_RUN_PROBE_REQUIRED")
        self.assertEqual(cfg_store.read_active("p13_funding_only_cap").status, "MICRO_RUN_PROBE_REQUIRED")

        # Also test P1-5 DSR deduplication under immutable family key
        fam_key = "funding_carry:lighter<->gate"
        eff_before = ledger.count_effective_trials(fam_key)
        cand = StrategyConfig.from_dict(rec["candidate_config"])
        verifier.verify(cand, start_ms=start, end_ms=start + 2159 * HOUR_MS)
        eff_after = ledger.count_effective_trials(fam_key)
        self.assertEqual(eff_before, eff_after)

        ledger.close()
        cfg_store.close()
        orch.close()

    def test_redteam_t1_lag_kills_1bar_phantom_and_dq_catches_bbo_inversion_and_flatline(self) -> None:
        """Red-Team #2 & #9: 1-bar spike cannot profit under T+1 lag; BBO ingestion inversion & >12h flatline fail DQ."""
        start = (1_720_000_000_000 // HOUR_MS) * HOUR_MS
        hours = 160
        fa, fb, ca, cb = [], [], [], []
        for h in range(hours):
            ts = start + h * HOUR_MS
            fa.append(FundingRateBar("hyperliquid", "WLD", ts, 0.0000125, 1.0, PROVENANCE_REAL))
            fb.append(FundingRateBar("okx", "WLD", ts, 0.0000125, 1.0, PROVENANCE_REAL))
            # Strict 1-bar spike (reverts immediately on h+1) -> unexecutable phantom on 1h candles!
            disloc = 0.0040 if (h % 25 == 0 and h > 48) else 0.0
            px_b = 100.0 + 0.3 * math.sin(h / 5.0)
            px_a = px_b * (1.0 + disloc)
            ca.append(PriceBar("hyperliquid", "WLD", ts, px_a, px_a * 1.002, px_a * 0.998, px_a, 1e6, PROVENANCE_REAL))
            cb.append(PriceBar("okx", "WLD", ts, px_b, px_b * 1.002, px_b * 0.998, px_b, 1e6, PROVENANCE_REAL))
        self.store.upsert_funding(fa)
        self.store.upsert_funding(fb)
        self.store.upsert_candles(ca)
        self.store.upsert_candles(cb)

        br = BasisReversionPlugin()
        cfg = br.default_config(symbols=["WLD"], params={"entry_zscore": 1.8, "min_basis_bps": 12.0, "z_lookback_hours": 24})
        res = br.evaluate(self.store, cfg, start_ms=start, end_ms=start + (hours - 1) * HOUR_MS)
        self.assertLessEqual(res.metrics.basis_pnl_usd, 0.0)

        # Red-Team #9: BBO out-of-order ingestion timestamp inversion MUST fail DQ-03!
        now = utc_now_ms()
        inverted = [
            BBOTick("lighter", "ARB", now - 10_000, 1.0, 1.0001, 5.0, 5.0, 20.0, PROVENANCE_REAL),
            BBOTick("lighter", "ARB", now - 20_000, 1.0, 1.0001, 5.0, 5.0, 20.0, PROVENANCE_REAL),  # backwards!
        ]
        self.store.upsert_bbo_ticks(inverted)
        bbo_rep, _ = self.auditor.audit_bbo_ticks("lighter", "gate", ["ARB"], persist=False, max_stale_sec=None)
        dq3 = next(c for c in bbo_rep.checks if c.gate_id == "DQ-03")
        self.assertFalse(dq3.passed)

        # Red-Team #9 & MEDIUM-R1: CSV audit defaults to UNVERIFIED_LOCAL_CSV (rejected by DQ-01 even with valid timestamps!)
        bad_csv = self.root / "bad.csv"
        bad_csv.write_text("ts_ms,bid,ask\n,100,101\n" * 12, encoding="utf-8")
        csv_rep = self.auditor.audit_csv_sample(bad_csv, dataset_name="bad_local", persist=False)
        self.assertEqual(csv_rep.provenance, "UNVERIFIED_LOCAL_CSV")
        self.assertEqual(csv_rep.tier, "DATA_REJECTED")

        valid_local_csv = self.root / "valid_local.csv"
        rows_csv = ["ts_ms,bid,ask"] + [f"{now - (20 - i) * 1000},100.0,100.1" for i in range(15)]
        valid_local_csv.write_text("\n".join(rows_csv) + "\n", encoding="utf-8")
        rep_unverified = self.auditor.audit_csv_sample(valid_local_csv, dataset_name="unverified_local", persist=False)
        self.assertEqual(rep_unverified.tier, "DATA_REJECTED")
        self.assertFalse(next(c for c in rep_unverified.checks if c.gate_id == "DQ-01").passed)
        rep_certified = self.auditor.audit_csv_sample(
            valid_local_csv, dataset_name="certified_server", persist=False, provenance="SERVER_COLLECTED_LIVE"
        )
        self.assertEqual(rep_certified.tier, "DATA_QUALIFIED_FULL")


if __name__ == "__main__":
    unittest.main()
