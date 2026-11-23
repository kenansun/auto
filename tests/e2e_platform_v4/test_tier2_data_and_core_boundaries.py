"""Tier 2 Boundary & Corner Case E2E Tests: FA-01..FA-05 (Defects D1..D6, D8..D16 & Edge Cases).

Every defect oracle deterministically fails if the corresponding v3 bug regresses.
All Platform v4 imports are lazy (inside test methods). Zero skips, zero fake passes,
pure Python 3.12 stdlib, 100% STD-34 compliant.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import random
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


class TestTier2FA01DataFoundationBoundaries(unittest.TestCase):
    """FA-01 Boundary & Corner Cases (Fail-Closed Validation, D14 Index/Idempotency)."""

    def test_t2_fa01_validate_canonical_bar_fail_closed_matrix(self) -> None:
        assets = fixtures.import_assets()
        models = fixtures.import_df_models()
        prov_real, _, prov_synth = fixtures.import_provenance()
        validate = models.validate_canonical_bar

        base_kw: dict[str, Any] = {
            "venue": "hyperliquid",
            "symbol": "BTC",
            "asset_class": assets.ASSET_CLASS_CRYPTO_PERP,
            "timeframe_h": 12.0,
            "ts_event_ms": fixtures.BASE_TS_MS,
            "ts_avail_ms": fixtures.BASE_TS_MS + 12 * fixtures.HOUR_MS,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume_usd": 1_000_000.0,
            "spread_bps": 1.5,
            "provenance": prov_real,
        }
        # Clean bar passes
        validate(fixtures.make_bar(**base_kw), allow_synthetic_in_test=False)

        invalid_mutations: list[tuple[str, dict[str, Any]]] = [
            ("lookahead_ts_avail", {"ts_avail_ms": fixtures.BASE_TS_MS - 1}),
            ("nan_close", {"close": float("nan")}),
            ("inf_high", {"high": float("inf")}),
            ("zero_price", {"open": 0.0, "low": 0.0}),
            ("negative_price", {"low": -1.0}),
            ("high_below_close", {"high": 101.0, "close": 102.0}),
            ("low_above_open", {"low": 101.0, "open": 100.0}),
            ("negative_volume", {"volume_usd": -10.0}),
            ("negative_spread", {"spread_bps": -0.5}),
            ("synthetic_blocked", {"provenance": prov_synth}),
        ]
        for label, patch in invalid_mutations:
            kw = dict(base_kw)
            kw.update(patch)
            with self.subTest(case=label):
                with self.assertRaises(ValueError, msg=f"validate_canonical_bar must reject {label}"):
                    bar = fixtures.make_bar(**kw)
                    validate(bar, allow_synthetic_in_test=False)

        # Synthetic allowed ONLY when explicitly opted-in for tests
        synth_kw = dict(base_kw)
        synth_kw["provenance"] = prov_synth
        validate(fixtures.make_bar(**synth_kw), allow_synthetic_in_test=True)

    def test_t2_fa01_d14_explain_query_plan_uses_exact_timeframe_sec_index(self) -> None:
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t2_d14_plan_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            router.upsert_bars(
                web3_dom,
                fixtures.build_bar_series(
                    symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                    timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=[100.0 + i for i in range(40)],
                    carry_rates=[0.0001] * 40,
                ),
            )
            router.upsert_bars(
                gold_dom,
                fixtures.build_bar_series(
                    symbol="GC=F", venue="yahoo", asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
                    timeframe_h=24.0, start_ms=fixtures.BASE_TS_MS, closes=[2000.0 + i for i in range(40)],
                ),
            )

            for db_path, venue, sym, tf_sec in (
                (tmp / "domains" / "web3_lake.sqlite", "hyperliquid", "BTC", 43200),
                (tmp / "domains" / "gold_macro_lake.sqlite", "yahoo", "GC=F", 86400),
            ):
                bar_tbl = fixtures.discover_bar_table(db_path)
                plan = fixtures.query_explain_plan(
                    db_path,
                    f'SELECT * FROM "{bar_tbl}" WHERE venue=? AND symbol=? AND timeframe_sec=? AND ts_avail_ms<=? ORDER BY ts_event_ms ASC',
                    (venue, sym, tf_sec, fixtures.BASE_TS_MS + 100 * fixtures.DAY_MS),
                )
                self.assertTrue(
                    ("USING" in plan.upper()) and ("INDEX" in plan.upper() or "PRIMARY KEY" in plan.upper()),
                    f"D14 violation: query plan on {db_path.name}.{bar_tbl} must use integer timeframe_sec index; got:\n{plan}",
                )
                self.assertNotIn(f"SCAN {bar_tbl}", plan, f"D14 violation: full table scan detected:\n{plan}")

    def test_t2_fa01_d14_same_ms_rapid_record_trial_and_audit_upserts_no_integrity_error(self) -> None:
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t2_d14_idemp_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            # 10 rapid back-to-back record_trial calls inside <1ms must never raise IntegrityError
            trial_ids: list[str] = []
            for i in range(10):
                tid = router.record_trial(
                    family_id="rapid_family",
                    track_name="track_w",
                    strategy_name="W2",
                    params_hash=f"p_{i}",
                    metrics={"sharpe": 0.5 + 0.01 * i},
                    verdict="REJECTED",
                )
                trial_ids.append(tid)
            self.assertEqual(len(set(trial_ids)), 10)
            self.assertEqual(router.get_family_trial_count("rapid_family"), 10)

            # Re-upserting identical bars must be idempotent without IntegrityError or duplicate rows
            btc_spec = assets.default_web3_specs(("BTC",))["BTC"]
            bars = fixtures.build_bar_series(
                symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=[100.0 + 0.5 * i for i in range(45)],
                carry_rates=[0.0001] * 45, carry_interval_h=12.0,
            )
            router.upsert_bars(web3_dom, bars)
            router.upsert_bars(web3_dom, bars)
            loaded = router.load_bars(web3_dom, "hyperliquid", "BTC", 43200)
            self.assertEqual(len(loaded), 45)

            # 10 rapid back-to-back same-ms audit_domain_dq writes must never collide on dq_certificates PK
            for _ in range(10):
                dq_rep = router.audit_domain_dq(web3_dom, "hyperliquid", "BTC", 43200, btc_spec)
                self.assertTrue(dq_rep.passed, f"Expected clean BTC bars to pass DQ: {dq_rep}")
            cat_conn = fixtures.open_sqlite_ro(tmp / "catalog.sqlite")
            try:
                dq_cnt = int(cat_conn.execute("SELECT COUNT(*) FROM dq_certificates").fetchone()[0])
            finally:
                cat_conn.close()
            self.assertEqual(dq_cnt, 10, f"Expected 10 distinct persisted DQ certificates, got {dq_cnt}")

    def test_t2_fa01_unknown_domain_fails_closed(self) -> None:
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        with fixtures.temp_sandbox(prefix="e2e_t2_fa01_dom_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            bars = fixtures.build_bar_series(
                symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=[100.0],
            )
            with self.assertRaises((ValueError, KeyError)):
                router.upsert_bars("nonexistent_domain_xyz", bars)
            with self.assertRaises((ValueError, KeyError)):
                router.load_bars("nonexistent_domain_xyz", "hyperliquid", "BTC", 43200)

    def test_t2_fa01_migrate_from_v3_sqlite_double_run_is_idempotent(self) -> None:
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t2_fa01_mig2_") as tmp:
            v3_db = fixtures.create_synthetic_v3_lake_db(tmp / "v3.sqlite", n_web3_bars=20, n_gold_bars=20)
            router = router_mod.DataFoundationRouter(tmp / "v4")
            self.addCleanup(fixtures.close_router, router)

            router.migrate_from_v3_sqlite(v3_db)
            w1 = len(router.load_bars(web3_dom, "hyperliquid", "BTC", 43200))
            g1 = len(router.load_bars(gold_dom, "yahoo", "GC=F", 86400))

            router.migrate_from_v3_sqlite(v3_db)
            w2 = len(router.load_bars(web3_dom, "hyperliquid", "BTC", 43200))
            g2 = len(router.load_bars(gold_dom, "yahoo", "GC=F", 86400))
            self.assertEqual((w1, g1), (20, 20))
            self.assertEqual((w2, g2), (w1, g1), "Second migration must be 100% idempotent")

    def test_t2_fa01_empty_range_and_future_cutoff_return_empty_without_crash(self) -> None:
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t2_fa01_empty_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            bars = fixtures.build_bar_series(
                symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=[100.0, 101.0],
            )
            router.upsert_bars(web3_dom, bars)
            self.assertEqual(router.load_bars(web3_dom, "hyperliquid", "BTC", 43200, as_of_ms=fixtures.BASE_TS_MS - 1), [])
            self.assertEqual(
                router.load_bars(
                    web3_dom, "hyperliquid", "BTC", 43200,
                    start_ms=fixtures.BASE_TS_MS + 10 * fixtures.DAY_MS,
                    end_ms=fixtures.BASE_TS_MS,
                ),
                [],
            )
            self.assertEqual(router.load_bars(web3_dom, "hyperliquid", "NO_SUCH_COIN", 43200), [])


class TestTier2FA02ResamplingDQAndCollectorBoundaries(unittest.TestCase):
    """FA-02 Boundary & Corner Cases (D10 Misaligned Resample, D11 DQ-MA-07/Gap, D16 PIT)."""

    def test_t2_fa02_d10_misaligned_4800_1h_bars_resample_to_exact_399_12h_never_401(self) -> None:
        assets = fixtures.import_assets()
        resample_bars = fixtures.import_resample_bars()

        # Exact Hyperliquid misalignment pattern: starts at 11:00 UTC -> 1h leading partial + 399 complete + 11h trailing partial
        start_1100 = fixtures.BASE_TS_MS + 11 * fixtures.HOUR_MS
        bars_1h = fixtures.build_bar_series(
            symbol="BTC",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=1.0,
            start_ms=start_1100,
            closes=[100.0 + 0.005 * i for i in range(4800)],
            carry_rates=[0.00004] * 4800,
            carry_interval_h=1.0,
        )
        bars_12h = resample_bars(bars_1h, 12.0, require_complete=True)
        self.assertEqual(
            len(bars_12h),
            399,
            f"D10 violation: 4,800 misaligned 1h bars starting 11:00 UTC must resample to exact 399 12h bars, got {len(bars_12h)}",
        )
        # First complete 12h bar starts at 12:00 UTC (bars_1h[1])
        self.assertEqual(bars_12h[0].ts_event_ms, bars_1h[1].ts_event_ms)
        self.assertEqual(bars_12h[0].ts_avail_ms, max(b.ts_avail_ms for b in bars_1h[1:13]))

    def test_t2_fa02_d10_tradfi_1h_to_4h_drops_partial_buckets_and_preserves_causal_ts_avail(self) -> None:
        assets = fixtures.import_assets()
        resample_bars = fixtures.import_resample_bars()

        # 1 stray leading 1h auction bar at 03:00 UTC + 12 complete 1h bars (3 x 4h buckets) + 1 stray trailing 1h bar
        start_0300 = fixtures.BASE_TS_MS + 3 * fixtures.HOUR_MS
        bars_1h = fixtures.build_bar_series(
            symbol="GC=F",
            venue="yahoo",
            asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
            timeframe_h=1.0,
            start_ms=start_0300,
            closes=[2500.0 + i for i in range(14)],
            contract_multiplier=100.0,
        )
        bars_4h = resample_bars(bars_1h, 4.0, require_complete=True)
        self.assertEqual(len(bars_4h), 3, f"Stray 1h leading/trailing TradFi bars must be dropped; got {len(bars_4h)}")
        self.assertEqual(bars_4h[0].ts_avail_ms, max(b.ts_avail_ms for b in bars_1h[1:5]))
        self.assertTrue(all(b.ts_avail_ms >= b.ts_event_ms for b in bars_4h))

    def test_t2_fa02_d11_dq_ma_07_rejects_zero_58pct_and_unit_spike_funding_exempts_tradfi(self) -> None:
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        btc_spec = assets.default_web3_specs(("BTC",))["BTC"]
        paxg_spec = assets.default_gold_specs()["PAXG"]
        gc_spec = assets.default_gold_specs()["GC=F"]

        with fixtures.temp_sandbox(prefix="e2e_t2_d11_dq07_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            # Case 1: 0% funding coverage on CRYPTO_PERP -> FAIL DQ-MA-07
            zero_bars = fixtures.build_bar_series(
                symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=[40000.0 + i for i in range(100)],
                carry_rates=[0.0] * 100,
            )
            router.upsert_bars(web3_dom, zero_bars)
            rep_zero = router.audit_domain_dq(web3_dom, "hyperliquid", "BTC", 43200, btc_spec)
            dq07_zero = [c for c in rep_zero.checks if c.gate_id == "DQ-MA-07"][0]
            self.assertFalse(dq07_zero.passed, "0% funding coverage on perp must fail DQ-MA-07")
            self.assertFalse(rep_zero.passed)

            # Case 2: 58% Gate-PAXG partial coverage -> FAIL DQ-MA-07
            partial_rates = [0.0001 if i < 58 else 0.0 for i in range(100)]
            partial_bars = fixtures.build_bar_series(
                symbol="PAXG", venue="gate", asset_class=assets.ASSET_CLASS_GOLD_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=[2600.0 + i * 0.5 for i in range(100)],
                carry_rates=partial_rates,
            )
            router.upsert_bars(web3_dom, partial_bars)
            rep_part = router.audit_domain_dq(web3_dom, "gate", "PAXG", 43200, paxg_spec)
            dq07_part = [c for c in rep_part.checks if c.gate_id == "DQ-MA-07"][0]
            self.assertFalse(dq07_part.passed, "58% funding coverage (<98%) must fail DQ-MA-07")
            self.assertFalse(rep_part.passed)

            # Case 3: Implausible unit spike (0.50 = 50% per period!) -> FAIL DQ-MA-07
            spike_rates = [0.0001] * 99 + [0.50]
            spike_bars = fixtures.build_bar_series(
                symbol="ETH", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=[2500.0 + i for i in range(100)],
                carry_rates=spike_rates,
            )
            router.upsert_bars(web3_dom, spike_bars)
            rep_spike = router.audit_domain_dq(web3_dom, "hyperliquid", "ETH", 43200, assets.default_web3_specs(("ETH",))["ETH"])
            dq07_spike = [c for c in rep_spike.checks if c.gate_id == "DQ-MA-07"][0]
            self.assertFalse(dq07_spike.passed, "Implausible 0.50/period funding unit spike must fail DQ-MA-07")

            # Case 4: GOLD_TRADFI with 0.0 carry -> EXEMPT / PASS DQ-MA-07
            gc_bars = fixtures.build_bar_series(
                symbol="GC=F", venue="yahoo", asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
                timeframe_h=24.0, start_ms=fixtures.BASE_TS_MS, closes=[2500.0 + i for i in range(100)],
                carry_rates=[0.0] * 100,
            )
            router.upsert_bars(gold_dom, gc_bars)
            rep_gc = router.audit_domain_dq(gold_dom, "yahoo", "GC=F", 86400, gc_spec)
            dq07_gc = [c for c in rep_gc.checks if c.gate_id == "DQ-MA-07"]
            if dq07_gc:
                self.assertTrue(dq07_gc[0].passed, "GOLD_TRADFI with 0.0 carry must be exempt/pass DQ-MA-07")
            self.assertTrue(rep_gc.passed)

    def test_t2_fa02_d11_tightened_24_7_gap_budget_rejects_48h_gap_on_12h_crypto(self) -> None:
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        btc_spec = assets.default_web3_specs(("BTC",))["BTC"]

        with fixtures.temp_sandbox(prefix="e2e_t2_d11_gap_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            bars_a = fixtures.build_bar_series(
                symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=[40000.0 + i for i in range(50)],
                carry_rates=[0.0001] * 50,
            )
            # Inject 48h gap (> 3 * 12h = 36h tightened 24/7 budget!)
            gap_start = bars_a[-1].ts_event_ms + 48 * fixtures.HOUR_MS
            bars_b = fixtures.build_bar_series(
                symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=gap_start, closes=[40060.0 + i for i in range(50)],
                carry_rates=[0.0001] * 50,
            )
            router.upsert_bars(web3_dom, bars_a + bars_b)
            rep = router.audit_domain_dq(web3_dom, "hyperliquid", "BTC", 43200, btc_spec)
            dq04 = [c for c in rep.checks if c.gate_id == "DQ-MA-04"][0]
            self.assertFalse(dq04.passed, "D11 violation: 48h gap on 12h 24/7 crypto must fail DQ-MA-04")
            self.assertFalse(rep.passed)

    def test_t2_fa02_d16_macro_event_and_tips_pit_conceals_unreleased_at_t_minus_1ms(self) -> None:
        models = fixtures.import_df_models()
        router_mod = fixtures.import_df_router()
        prov_real, _, _ = fixtures.import_provenance()

        with fixtures.temp_sandbox(prefix="e2e_t2_d16_pit_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            ts_tips = fixtures.utc_ms(2026, 11, 20, 21, 0)
            router.upsert_yield_curve(
                [
                    models.MacroYieldPoint(
                        "us_treasury", ts_tips - fixtures.HOUR_MS, ts_tips, 2.46, 2.61, 2.77, 2.92, 2.99, prov_real
                    )
                ]
            )
            self.assertEqual(router.load_yield_curve(as_of_ms=ts_tips - 1), [])
            got_y = router.load_yield_curve(as_of_ms=ts_tips)
            self.assertEqual(len(got_y), 1)
            self.assertAlmostEqual(float(got_y[0].real_10y), 2.77, places=6)

            dec_events = [
                models.MacroEventRelease(
                    "NFP", "NFP_2026_12_04", "2026-11", fixtures.DEC_2026_NFP_MS, fixtures.DEC_2026_NFP_MS,
                    False, 215.0, 180.0, 1.4, "bls", prov_real,
                ),
                models.MacroEventRelease(
                    "FOMC", "FOMC_2026_12_09", "2026-12", fixtures.DEC_2026_FOMC_MS, fixtures.DEC_2026_FOMC_MS,
                    True, 4.25, 4.50, -0.8, "fed", prov_real,
                ),
                models.MacroEventRelease(
                    "CPI", "CPI_2026_12_10", "2026-11", fixtures.DEC_2026_CPI_MS, fixtures.DEC_2026_CPI_MS,
                    False, 2.7, 2.6, 0.5, "bls", prov_real,
                ),
            ]
            router.upsert_macro_events(dec_events)
            # 1ms before FOMC release: NFP visible, FOMC & CPI MUST NOT leak!
            before_fomc = router.load_macro_events(as_of_ms=fixtures.DEC_2026_FOMC_MS - 1)
            self.assertEqual([e.event_id for e in before_fomc], ["NFP_2026_12_04"])
            # Exact FOMC release ms: NFP + FOMC visible, CPI still concealed
            at_fomc = router.load_macro_events(as_of_ms=fixtures.DEC_2026_FOMC_MS)
            self.assertEqual({e.event_id for e in at_fomc}, {"NFP_2026_12_04", "FOMC_2026_12_09"})

    def test_t2_fa02_resample_empty_and_single_bar_edge_cases(self) -> None:
        assets = fixtures.import_assets()
        resample_bars = fixtures.import_resample_bars()
        self.assertEqual(resample_bars([], 12.0, require_complete=True), [])
        one = fixtures.build_bar_series(
            symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=1.0, start_ms=fixtures.BASE_TS_MS, closes=[100.0],
        )
        self.assertEqual(resample_bars(one, 12.0, require_complete=True), [])


class TestTier2FA03AccountingAndCompatBoundaries(unittest.TestCase):
    """FA-03 Boundary & Corner Cases (D1 Exit/Holiday Gap, D2/V-04, D6, D11 Carry, D15)."""

    def test_t2_fa03_d1_exit_bar_overnight_gap_pnl_exact_conservation(self) -> None:
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        zero_cost = assets.CostModel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        spec_zero = assets.AssetSpec(
            symbol="GAP_SYM", asset_class=assets.ASSET_CLASS_GOLD_TRADFI, venue="yahoo",
            calendar=assets.CALENDAR_US_DAILY, cost_model=zero_cost, max_leverage_cap=1.0,
        )
        # Bar 0: O=C=100 (enter w=+1.0 at Bar 1 open=100); Bar 1: O=C=100 (held); Bar 2: O=C=110 (EXIT w=0.0, +10% gap!)
        bars = [
            fixtures.make_bar(
                venue="yahoo", symbol="GAP_SYM", asset_class=assets.ASSET_CLASS_GOLD_TRADFI, timeframe_h=24.0,
                ts_event_ms=fixtures.BASE_TS_MS, ts_avail_ms=fixtures.BASE_TS_MS + fixtures.DAY_MS,
                open=100.0, high=100.0, low=100.0, close=100.0, volume_usd=1e7,
            ),
            fixtures.make_bar(
                venue="yahoo", symbol="GAP_SYM", asset_class=assets.ASSET_CLASS_GOLD_TRADFI, timeframe_h=24.0,
                ts_event_ms=fixtures.BASE_TS_MS + fixtures.DAY_MS, ts_avail_ms=fixtures.BASE_TS_MS + 2 * fixtures.DAY_MS,
                open=100.0, high=100.0, low=100.0, close=100.0, volume_usd=1e7,
            ),
            fixtures.make_bar(
                venue="yahoo", symbol="GAP_SYM", asset_class=assets.ASSET_CLASS_GOLD_TRADFI, timeframe_h=24.0,
                ts_event_ms=fixtures.BASE_TS_MS + 2 * fixtures.DAY_MS, ts_avail_ms=fixtures.BASE_TS_MS + 3 * fixtures.DAY_MS,
                open=110.0, high=110.0, low=110.0, close=110.0, volume_usd=1e7,
            ),
        ]
        # Bar 0 close (100): emit +1.0 -> Bar 1 open (100) enters OPEN_ENTRY w=1.0
        # Bar 1 close (100): emit 0.0  -> Bar 2 open (110, +10% gap) captures gap PnL THEN executes OPEN_EXIT w=0.0!
        steps = [{"GAP_SYM": 1.0}, {"GAP_SYM": 0.0}, {"GAP_SYM": 0.0}]
        strat = fixtures.build_scripted_strategy(steps, strategy_id="d1_exit_gap", asset_domain="gold")
        risk_cfg = risk_mod.RiskAllocatorConfig(
            max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0,
            max_single_weight=1.0, use_correlation_idm=False,
        )
        res = acct_mod.PortfolioAccountingEngine().simulate(
            strat, {"GAP_SYM": spec_zero}, {"GAP_SYM": bars}, risk_cfg=risk_cfg, capital_usd=10_000.0
        )
        exec_w = [float(w.get("GAP_SYM", 0.0)) for w in res.executed_weights_history]
        self.assertAlmostEqual(exec_w[0], 0.0, places=8)
        self.assertGreater(exec_w[1], 0.5, "Position must be held on Bar 1 entering the Bar 2 exit gap")
        self.assertAlmostEqual(exec_w[2], 0.0, places=8, msg="Position must exit to 0.0 at Bar 2 open")
        exit_trades = [
            t for t in res.trade_ledger.executed
            if t.symbol == "GAP_SYM" and t.reason == "OPEN_EXIT" and abs(float(t.fill_price) - 110.0) < 1e-6
        ]
        self.assertEqual(len(exit_trades), 1, f"Must execute OPEN_EXIT at 110.0, got {res.trade_ledger.executed}")
        w_held = exec_w[1]
        expected_gap = w_held * 10_000.0 * (110.0 / 100.0 - 1.0)
        gap_pnl = float(fixtures.get_metric(res, "gap_pnl_usd"))
        gross_pnl = float(fixtures.get_metric(res, "gross_price_pnl_usd", "net_pnl_usd"))
        net_pnl = float(fixtures.get_metric(res, "net_pnl_usd"))
        self.assertAlmostEqual(gap_pnl, expected_gap, delta=1e-3, msg="D1 violation: exit-bar +10% overnight gap PnL dropped!")
        self.assertAlmostEqual(gross_pnl, expected_gap, delta=1e-3)
        self.assertAlmostEqual(net_pnl, expected_gap, delta=1e-3)

    def test_t2_fa03_d1_per_symbol_holiday_skip_gap_pnl_uses_last_bar_close(self) -> None:
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        zero_cost = assets.CostModel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        specs = {
            "A": assets.AssetSpec("A", assets.ASSET_CLASS_GOLD_TRADFI, "yahoo", assets.CALENDAR_US_DAILY, zero_cost, 1.0),
            "B": assets.AssetSpec("B", assets.ASSET_CLASS_GOLD_TRADFI, "yahoo", assets.CALENDAR_US_DAILY, zero_cost, 1.0),
        }
        t0 = fixtures.BASE_TS_MS
        t1 = t0 + fixtures.DAY_MS
        t2 = t0 + 2 * fixtures.DAY_MS
        t3 = t0 + 3 * fixtures.DAY_MS

        # Symbol A trades Days 0..3 flat at 100 (weight 0).
        # Symbol B trades Day 0 (enter), Day 1 (C=100, emit exit 0.0), SKIPS Day 2 (holiday), exits Day 3 at O=C=106 (+6% gap!)
        bars_a = [
            fixtures.make_bar(
                venue="yahoo", symbol="A", asset_class=assets.ASSET_CLASS_GOLD_TRADFI, timeframe_h=24.0,
                ts_event_ms=ts, ts_avail_ms=ts + fixtures.DAY_MS, open=100.0, high=100.0, low=100.0, close=100.0, volume_usd=1e7,
            )
            for ts in (t0, t1, t2, t3)
        ]
        bars_b = [
            fixtures.make_bar(
                venue="yahoo", symbol="B", asset_class=assets.ASSET_CLASS_GOLD_TRADFI, timeframe_h=24.0,
                ts_event_ms=t0, ts_avail_ms=t0 + fixtures.DAY_MS, open=100.0, high=100.0, low=100.0, close=100.0, volume_usd=1e7,
            ),
            fixtures.make_bar(
                venue="yahoo", symbol="B", asset_class=assets.ASSET_CLASS_GOLD_TRADFI, timeframe_h=24.0,
                ts_event_ms=t1, ts_avail_ms=t1 + fixtures.DAY_MS, open=100.0, high=100.0, low=100.0, close=100.0, volume_usd=1e7,
            ),
            # Day 2 SKIPPED (holiday for B)
            fixtures.make_bar(
                venue="yahoo", symbol="B", asset_class=assets.ASSET_CLASS_GOLD_TRADFI, timeframe_h=24.0,
                ts_event_ms=t3, ts_avail_ms=t3 + fixtures.DAY_MS, open=106.0, high=106.0, low=106.0, close=106.0, volume_usd=1e7,
            ),
        ]
        steps = [{"A": 0.0, "B": 0.5}, {"A": 0.0, "B": 0.0}, {"A": 0.0}, {"A": 0.0, "B": 0.0}]
        strat = fixtures.build_scripted_strategy(steps, strategy_id="d1_holiday_gap", asset_domain="gold")
        risk_cfg = risk_mod.RiskAllocatorConfig(
            max_positions=2, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0,
            max_single_weight=1.0, use_correlation_idm=False,
        )
        res = acct_mod.PortfolioAccountingEngine().simulate(
            strat, specs, {"A": bars_a, "B": bars_b}, risk_cfg=risk_cfg, capital_usd=10_000.0
        )
        exec_b = [float(w.get("B", 0.0)) for w in res.executed_weights_history]
        self.assertAlmostEqual(exec_b[1], 0.5, places=6)
        self.assertAlmostEqual(exec_b[2], 0.5, places=6)
        self.assertAlmostEqual(exec_b[3], 0.0, places=6)
        b_exits = [
            t for t in res.trade_ledger.executed
            if t.symbol == "B" and t.reason == "OPEN_EXIT" and abs(float(t.fill_price) - 106.0) < 1e-6
        ]
        self.assertEqual(len(b_exits), 1, f"B must execute OPEN_EXIT at Day 3 open 106.0, got {res.trade_ledger.executed}")
        expected_gap = exec_b[1] * 10_000.0 * 0.06
        self.assertAlmostEqual(float(fixtures.get_metric(res, "gap_pnl_usd")), expected_gap, delta=1e-3)
        self.assertAlmostEqual(float(fixtures.get_metric(res, "net_pnl_usd")), expected_gap, delta=1e-3)

    def test_t2_fa03_d2_and_v04_donchian_lookback_live_and_zero_hardcoded_squeeze(self) -> None:
        assets = fixtures.import_assets()
        factors_mod = fixtures.import_factors_mod()
        cust_mod = fixtures.import_strategy_customizer_mod()
        reg_mod = fixtures.import_regime_inference_mod()

        spec = assets.default_web3_specs(("BTC",))["BTC"]
        # Flat at 100 for bars 0..49, spike to 120 at bar 10, then breakout to 105 at bar 50
        opens = [100.0] * 51
        closes = [100.0] * 50 + [105.0]
        highs = [100.5] * 50 + [105.5]
        lows = [99.5] * 50 + [99.8]
        closes[10] = 120.0
        highs[10] = 120.0
        bars = fixtures.build_bar_series(
            symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, opens=opens, closes=closes, highs=highs, lows=lows,
            carry_rates=[0.0001] * len(closes), carry_interval_h=12.0, use_v3_model=True,
        )
        eng = factors_mod.StreamingFactorEngine(spec.calendar, 12.0)
        snaps = eng.compute_series(bars)

        def _run_with(entry_n: int, risk_pct: float = 2.0) -> Any:
            cfg = cust_mod.KernelConfig(
                strategy_name=f"d2_e{entry_n}",
                asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0,
                donchian_weight=0.80,
                ewmac_weight_fast=0.0,
                ewmac_weight_mid=0.0,
                ewmac_weight_slow=0.0,
                donchian_entry_n=entry_n,
                donchian_exit_n=max(5, entry_n // 2),
                risk_per_trade_pct=risk_pct,
                warmup_bars=20,
                position_buffer_pct=0.0,
                use_squeeze_boost=False,
            )
            kernel = cust_mod.AdaptiveMultiFactorKernel(cfg, spec)
            last_d = None
            for b, sn in zip(bars, snaps):
                rpt = reg_mod.RegimePoint(
                    ts_event_ms=b.ts_event_ms,
                    ts_avail_ms=b.ts_avail_ms,
                    raw_regime=reg_mod.REGIME_TREND_EXPANSION,
                    confirmed_regime=reg_mod.REGIME_TREND_EXPANSION,
                    adx_14=32.0,
                    kaufman_er_20=0.55,
                    vol_pctile_60=0.50,
                    squeeze_ratio_20=1.0,
                    hurst_proxy_40=0.60,
                    macro_gold_composite=0.0,
                )
                last_d = kernel.decide_bar(b, sn, rpt)
            assert last_d is not None
            return last_d

        dec_10 = _run_with(10, risk_pct=2.0)
        dec_46 = _run_with(46, risk_pct=2.0)
        self.assertGreater(
            float(dec_10.raw_forecast),
            float(dec_46.raw_forecast),
            f"D2 violation: donchian_entry_n=10 ({dec_10.raw_forecast}) must exceed donchian_entry_n=46 ({dec_46.raw_forecast})",
        )
        # Verify risk_per_trade_pct actively scales position sizing
        dec_high_risk = _run_with(10, risk_pct=3.0)
        dec_low_risk = _run_with(10, risk_pct=0.25)
        self.assertGreater(
            float(dec_high_risk.target_weight),
            float(dec_low_risk.target_weight),
            f"D2 violation: risk_per_trade_pct=3.0 ({dec_high_risk.target_weight}) must exceed 0.25 ({dec_low_risk.target_weight})",
        )

        # V-04 check: squeeze_breakout_dir=+1.0 with neutral selected factors & use_squeeze_boost=False must NOT inject +4.0!
        cfg_no_sq = cust_mod.KernelConfig(
            strategy_name="v04_no_sq",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0,
            donchian_weight=0.0,
            warmup_bars=0,
            position_buffer_pct=0.0,
            use_squeeze_boost=False,
            active_g2_factors=("ewmac_16_64",),
        )
        kernel_sq = cust_mod.AdaptiveMultiFactorKernel(cfg_no_sq, spec)
        neutral_snap = factors_mod.FactorSnapshot(
            symbol="BTC", venue="hyperliquid", timeframe_h=12.0,
            ts_event_ms=bars[-1].ts_event_ms, ts_avail_ms=bars[-1].ts_avail_ms,
            close=100.0, atr_14=2.0, prior_high_40=110.0, prior_low_40=90.0,
            prior_high_20=105.0, prior_low_20=95.0,
            values={
                "ewmac_8_32": 0.0,
                "ewmac_16_64": 0.0,
                "ewmac_32_128": 0.0,
                "squeeze_breakout_dir": 1.0,
                "realized_vol_ann_20": 30.0,
            },
        )
        rpt_sq = reg_mod.RegimePoint(
            ts_event_ms=bars[-1].ts_event_ms,
            ts_avail_ms=bars[-1].ts_avail_ms,
            raw_regime=reg_mod.REGIME_SQUEEZE_COIL,
            confirmed_regime=reg_mod.REGIME_SQUEEZE_COIL,
            adx_14=15.0,
            kaufman_er_20=0.25,
            vol_pctile_60=0.20,
            squeeze_ratio_20=0.70,
            hurst_proxy_40=0.50,
            macro_gold_composite=0.0,
        )
        dec_sq = kernel_sq.decide_bar(bars[-1], neutral_snap, rpt_sq)
        self.assertLess(
            abs(float(dec_sq.raw_forecast)),
            1e-6,
            f"V-04 violation: unselected squeeze_breakout_dir injected hardcoded forecast {dec_sq.raw_forecast}",
        )
        # Confirm explicit opt-in toggle use_squeeze_boost=True is wired (not hardcoded zero)
        cfg_with_sq = cust_mod.KernelConfig(
            strategy_name="v04_with_sq",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0,
            donchian_weight=0.0,
            warmup_bars=0,
            position_buffer_pct=0.0,
            use_squeeze_boost=True,
            active_g2_factors=("ewmac_16_64", "squeeze_breakout_dir"),
        )
        dec_sq_on = cust_mod.AdaptiveMultiFactorKernel(cfg_with_sq, spec).decide_bar(bars[-1], neutral_snap, rpt_sq)
        self.assertGreater(float(dec_sq_on.raw_forecast), 3.0)

    def test_t2_fa03_d6_macro_regime_timeline_non_monotonic_out_of_order_bit_identical(self) -> None:
        factors_mod = fixtures.import_factors_mod()
        bars_map, _ = fixtures.build_gold_seven_driver_fixture(n_days=200)
        ts_list = [b.ts_avail_ms for b in bars_map["GC=F"]]

        probe_indices = (10, 50, 100, 150, 199)
        ref: dict[int, dict[str, float]] = {}
        for idx in probe_indices:
            fresh = factors_mod.MacroRegimeTimeline(bars_map)
            ref[idx] = fresh.snapshot_as_of(ts_list[idx])

        reused = factors_mod.MacroRegimeTimeline(bars_map)
        adversarial_order = (10, 199, 9, 10, 150, 50, 100, 10)
        for idx in adversarial_order:
            got = reused.snapshot_as_of(ts_list[idx])
            if idx in ref:
                for k, v_ref in ref[idx].items():
                    self.assertEqual(
                        got[k],
                        v_ref,
                        f"D6 lookahead violation on MacroRegimeTimeline idx={idx} key={k}: got {got[k]} != ref {v_ref}",
                    )

    def test_t2_fa03_d11_carry_interval_h_8h_vs_1h_normalization_eliminates_8x_bug(self) -> None:
        assets = fixtures.import_assets()
        factors_mod = fixtures.import_factors_mod()
        spec = assets.default_web3_specs(("BTC",))["BTC"]

        bars_8h = fixtures.build_bar_series(
            symbol="BTC", venue="gate", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=1.0, start_ms=fixtures.BASE_TS_MS, closes=[40000.0] * 60,
            carry_rates=[0.0001] * 60, carry_interval_h=8.0, use_v3_model=True,
        )
        bars_1h = fixtures.build_bar_series(
            symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=1.0, start_ms=fixtures.BASE_TS_MS, closes=[40000.0] * 60,
            carry_rates=[0.0001] * 60, carry_interval_h=1.0, use_v3_model=True,
        )
        snap_8h = factors_mod.StreamingFactorEngine(spec.calendar, 1.0).compute_series(bars_8h)[-1]
        snap_1h = factors_mod.StreamingFactorEngine(spec.calendar, 1.0).compute_series(bars_1h)[-1]

        apr_8h = float(snap_8h.values["carry_apr_24h"])
        apr_1h = float(snap_1h.values["carry_apr_24h"])
        # Expected: 0.0001 * (8760 / 8) * 100 = 10.95% (or decimal 0.1095); ratio MUST be 1:8!
        ratio = apr_1h / apr_8h if abs(apr_8h) > 1e-12 else 0.0
        self.assertAlmostEqual(ratio, 8.0, delta=0.25, msg=f"D11 violation: 1h vs 8h carry_interval_h ratio expected 8.0, got {ratio} (apr_1h={apr_1h}, apr_8h={apr_8h})")
        # Also check absolute magnitude (accepting % units ~10.95 or decimal ~0.1095)
        apr_8h_pct = apr_8h * 100.0 if abs(apr_8h) < 1.0 else apr_8h
        self.assertAlmostEqual(apr_8h_pct, 10.95, delta=0.6)

    def test_t2_fa03_d15_intraday_t_plus_1_mixed_calendar_order_and_perp_liq_vs_etf_survival(self) -> None:
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        # --- D15-A: Exchange-date T+1 lock on intraday 1h A-Share bars ---
        ashare_spec = assets.default_ashare_template_spec("600519.SH")
        a_bars = fixtures.build_ashare_multiday_1h_bars(symbol="600519.SH", n_days=2, base_price=100.0)
        # 4 bars Day 1 (enter bar 0 -> executes at bar 1 open, attempt sell on bars 1, 2, 3), 4 bars Day 2 (sell bar 4 open)
        a_steps = [{"600519.SH": 0.8}] + [{"600519.SH": 0.0}] * (len(a_bars) - 1)
        a_strat = fixtures.build_scripted_strategy(a_steps, strategy_id="d15_t1", asset_domain="ashare")
        cfg_a = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False)
        res_a = acct_mod.PortfolioAccountingEngine().simulate(a_strat, {"600519.SH": ashare_spec}, {"600519.SH": a_bars}, risk_cfg=cfg_a)
        exec_w = [float(w.get("600519.SH", 0.0)) for w in res_a.executed_weights_history]
        self.assertAlmostEqual(exec_w[0], 0.0, places=6)
        self.assertGreater(exec_w[1], 0.0, "D15-A: Day 1 bar 1 must hold bought position")
        self.assertGreater(exec_w[2], 0.0, "D15-A: T+1 must lock same-day bar 2")
        self.assertGreater(exec_w[3], 0.0, "D15-A: T+1 must lock same-day bar 3")
        self.assertAlmostEqual(exec_w[4], 0.0, places=6, msg="D15-A: T+1 must unlock next exchange date")
        t1_locks = [s for s in res_a.trade_ledger.suppressed if s.reason == "T_PLUS_1_LOCK"]
        self.assertGreaterEqual(len(t1_locks), 2, f"Expected >=2 T_PLUS_1_LOCK suppressions, got {res_a.trade_ledger.suppressed}")
        buys = [e for e in res_a.trade_ledger.executed if e.side == "BUY"]
        sells = [e for e in res_a.trade_ledger.executed if e.side == "SELL"]
        self.assertEqual(len(buys), 1)
        self.assertEqual(len(sells), 1)
        self.assertAlmostEqual(float(buys[0].stamp_duty_usd), 0.0, places=8)
        self.assertGreater(float(sells[0].stamp_duty_usd), 0.0, "CN A-Share SELL must incur 5bp sell-only stamp duty")

        # --- D15-B: Mixed-calendar carry drag invariant to dict insertion order ---
        btc_spec = assets.AssetSpec(
            "BTC", assets.ASSET_CLASS_CRYPTO_PERP, "hyperliquid", assets.CALENDAR_CRYPTO_24_7,
            assets.CostModel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 2.0,
        )
        gld_spec = assets.AssetSpec(
            "GLD", assets.ASSET_CLASS_GOLD_TRADFI, "yahoo", assets.CALENDAR_US_DAILY,
            assets.CostModel(0.0, 0.0, 0.0, 0.0, 40.0, 0.0), 1.0,
        )
        btc_bars = fixtures.build_bar_series(
            symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=24.0, start_ms=fixtures.BASE_TS_MS, closes=[100.0] * 30,
        )
        gld_bars = fixtures.build_bar_series(
            symbol="GLD", venue="yahoo", asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
            timeframe_h=24.0, start_ms=fixtures.BASE_TS_MS, closes=[100.0] * 30,
        )
        mix_steps = [{"BTC": 0.4, "GLD": 0.4} for _ in range(30)]
        cfg_mix = risk_mod.RiskAllocatorConfig(max_positions=2, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=0.5, use_correlation_idm=False)
        res_bg = acct_mod.PortfolioAccountingEngine().simulate(
            fixtures.build_scripted_strategy(mix_steps), {"BTC": btc_spec, "GLD": gld_spec}, {"BTC": btc_bars, "GLD": gld_bars}, risk_cfg=cfg_mix
        )
        res_gb = acct_mod.PortfolioAccountingEngine().simulate(
            fixtures.build_scripted_strategy(mix_steps), {"GLD": gld_spec, "BTC": btc_spec}, {"GLD": gld_bars, "BTC": btc_bars}, risk_cfg=cfg_mix
        )
        net_bg = float(fixtures.get_metric(res_bg, "net_pnl_usd"))
        net_gb = float(fixtures.get_metric(res_gb, "net_pnl_usd"))
        self.assertLess(net_bg, 0.0, f"D15-B: 40bp GLD holding drag must produce strictly negative net PnL on flat bars, got {net_bg}")
        self.assertAlmostEqual(
            net_bg,
            net_gb,
            places=6,
            msg="D15-B: Per-symbol calendar carry must be invariant to dict insertion order",
        )

        # --- D15-D: Leveraged 5x perp intrabar liquidation vs 1x unleveraged ETF survival (-76% drop) ---
        perp_spec = assets.AssetSpec(
            "PERP5X", assets.ASSET_CLASS_CRYPTO_PERP, "hyperliquid", assets.CALENDAR_CRYPTO_24_7,
            assets.CostModel(0.0, 0.0, 0.0), max_leverage_cap=5.0,
        )
        perp_bars = [
            fixtures.make_bar(
                venue="hyperliquid", symbol="PERP5X", asset_class=assets.ASSET_CLASS_CRYPTO_PERP, timeframe_h=12.0,
                ts_event_ms=fixtures.BASE_TS_MS, ts_avail_ms=fixtures.BASE_TS_MS + 12 * fixtures.HOUR_MS,
                open=100.0, high=100.0, low=100.0, close=100.0, volume_usd=1e7,
            ),
            # Intrabar wick drops -25% to low=75 (exceeding 20% margin on 5x!), then closes back at 100
            fixtures.make_bar(
                venue="hyperliquid", symbol="PERP5X", asset_class=assets.ASSET_CLASS_CRYPTO_PERP, timeframe_h=12.0,
                ts_event_ms=fixtures.BASE_TS_MS + 12 * fixtures.HOUR_MS, ts_avail_ms=fixtures.BASE_TS_MS + 24 * fixtures.HOUR_MS,
                open=100.0, high=100.0, low=75.0, close=100.0, volume_usd=1e7,
            ),
        ]
        perp_cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=5000.0, max_portfolio_leverage=5.0, max_single_weight=5.0, use_correlation_idm=False)
        res_perp = acct_mod.PortfolioAccountingEngine().simulate(
            fixtures.build_scripted_strategy([{"PERP5X": 5.0}, {"PERP5X": 5.0}]),
            {"PERP5X": perp_spec}, {"PERP5X": perp_bars}, risk_cfg=perp_cfg, capital_usd=10_000.0,
        )
        liq_perp = int(fixtures.get_metric(res_perp, "liquidation_count", default=0))
        net_perp = float(fixtures.get_metric(res_perp, "net_pnl_usd"))
        self.assertTrue(
            liq_perp >= 1 or net_perp < -5_000.0,
            f"D15-D: 5x leveraged perp with -25% intrabar wick must trigger liquidation/loss (liq={liq_perp}, net={net_perp})",
        )

        # Unleveraged 1x ETF drops -76% (100 -> 24) MUST survive with >$2,000 equity and 0 liquidations
        etf_spec = assets.AssetSpec(
            "ETF1X", assets.ASSET_CLASS_GOLD_TRADFI, "yahoo", assets.CALENDAR_US_DAILY,
            assets.CostModel(0.0, 0.0, 0.0), max_leverage_cap=1.0,
        )
        etf_bars = fixtures.build_bar_series(
            symbol="ETF1X", venue="yahoo", asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
            timeframe_h=24.0, start_ms=fixtures.BASE_TS_MS, closes=[100.0, 70.0, 40.0, 24.0],
            opens=[100.0, 100.0, 70.0, 40.0], highs=[100.0, 100.0, 70.0, 40.0], lows=[100.0, 70.0, 40.0, 24.0],
        )
        etf_cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=5000.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False)
        res_etf = acct_mod.PortfolioAccountingEngine().simulate(
            fixtures.build_scripted_strategy([{"ETF1X": 1.0}] * 4),
            {"ETF1X": etf_spec}, {"ETF1X": etf_bars}, risk_cfg=etf_cfg, capital_usd=10_000.0,
        )
        liq_etf = int(fixtures.get_metric(res_etf, "liquidation_count", default=0))
        final_eq_etf = 10_000.0 + float(fixtures.get_metric(res_etf, "net_pnl_usd"))
        self.assertEqual(liq_etf, 0, "D15-D: Unleveraged 1x ETF must NEVER liquidate on -76% drawdown")
        self.assertGreater(final_eq_etf, 2_000.0, f"D15-D: Unleveraged 1x ETF must retain ~24% capital, got {final_eq_etf}")


class TestTier2FA04RiskAllocatorBoundaries(unittest.TestCase):
    """FA-04 Boundary & Corner Cases (D9 Top-K Cap, Correlated Vol Calibration, Degenerate Inputs)."""

    def test_t2_fa04_d9_max_positions_keeps_exact_top_4_of_12_by_abs_signal(self) -> None:
        assets = fixtures.import_assets()
        iface = fixtures.import_core_interfaces()
        risk_mod = fixtures.import_risk_allocator_mod()

        syms = tuple(assets.WEB3_LIQUID_UNIVERSE)
        self.assertEqual(len(syms), 12)
        specs = assets.default_web3_specs(syms)
        cfg = risk_mod.RiskAllocatorConfig(
            max_positions=4, target_annual_vol_pct=20.0, max_portfolio_leverage=2.0, max_single_weight=0.5,
        )
        alloc = fixtures.make_risk_allocator(cfg)
        mat = fixtures.generate_correlated_return_matrix(12, 80, 0.25, annual_vol=0.35, seed=201)
        recent = {s: mat[i] for i, s in enumerate(syms)}
        # Signals 1..12 with alternating signs -> exact top 4 by |raw_signal| are indices 8,9,10,11
        decisions = {
            s: iface.BarDecision(s, fixtures.BASE_TS_MS, 0.3 * (1 if i % 2 == 0 else -1), float((i + 1) * (1 if i % 2 == 0 else -1)), "TREND")
            for i, s in enumerate(syms)
        }
        weights = fixtures.call_risk_allocate(alloc, cfg, fixtures.BASE_TS_MS, decisions, recent, specs)
        active = {s for s, w in weights.items() if abs(w) > 1e-12}
        self.assertEqual(len(active), 4)
        self.assertEqual(active, set(syms[8:12]))

    def test_t2_fa04_d9_correlated_10_asset_basket_idm_prevents_vol_overshoot(self) -> None:
        assets = fixtures.import_assets()
        iface = fixtures.import_core_interfaces()
        risk_mod = fixtures.import_risk_allocator_mod()

        syms = tuple(assets.WEB3_LIQUID_UNIVERSE[:10])
        specs = assets.default_web3_specs(syms)
        mat = fixtures.generate_correlated_return_matrix(10, 260, 0.90, annual_vol=0.30, periods_per_year=365.0, seed=202)
        recent = {s: mat[i] for i, s in enumerate(syms)}
        decisions = {s: iface.BarDecision(s, fixtures.BASE_TS_MS, 0.5, 2.0, "TREND") for s in syms}

        cfg_corr = risk_mod.RiskAllocatorConfig(
            max_positions=10, target_annual_vol_pct=15.0, max_portfolio_leverage=3.0,
            max_single_weight=1.0, use_correlation_idm=True, corr_lookback_bars=120, corr_shrinkage=0.25,
        )
        cfg_uncorr = risk_mod.RiskAllocatorConfig(
            max_positions=10, target_annual_vol_pct=15.0, max_portfolio_leverage=3.0,
            max_single_weight=1.0, use_correlation_idm=False,
        )
        w_corr = fixtures.call_risk_allocate(fixtures.make_risk_allocator(cfg_corr), cfg_corr, fixtures.BASE_TS_MS, decisions, recent, specs)
        w_uncorr = fixtures.call_risk_allocate(fixtures.make_risk_allocator(cfg_uncorr), cfg_uncorr, fixtures.BASE_TS_MS, decisions, recent, specs)

        gross_corr = sum(abs(v) for v in w_corr.values())
        gross_uncorr = sum(abs(v) for v in w_uncorr.values())
        self.assertLess(
            gross_corr,
            gross_uncorr,
            f"D9 violation: correlation-aware IDM gross ({gross_corr:.4f}) must be < naive uncorrelated ({gross_uncorr:.4f}) on rho=0.90 basket",
        )
        # Realized portfolio vol of w_corr across the 10-asset return matrix must be calibrated near 15% (not 2.03x = 30.5%!)
        port_rets = [sum(w_corr[s] * recent[s][t] for s in syms) for t in range(len(mat[0]))]
        realized_ann_vol_pct = fixtures.annualized_vol(port_rets, 365.0) * 100.0
        self.assertGreaterEqual(realized_ann_vol_pct, 8.25, f"Over-damped vol {realized_ann_vol_pct:.2f}%")
        self.assertLessEqual(realized_ann_vol_pct, 20.25, f"D9 vol overshoot {realized_ann_vol_pct:.2f}% > 20.25%")

    def test_t2_fa04_degenerate_zero_vol_and_all_zero_signals_safe_no_nan(self) -> None:
        assets = fixtures.import_assets()
        iface = fixtures.import_core_interfaces()
        risk_mod = fixtures.import_risk_allocator_mod()

        specs = assets.default_web3_specs(("BTC", "ETH"))
        cfg = risk_mod.RiskAllocatorConfig(max_positions=2, target_annual_vol_pct=15.0, max_portfolio_leverage=1.0, max_single_weight=0.5)
        alloc = fixtures.make_risk_allocator(cfg)

        # Empty decisions
        self.assertEqual(fixtures.call_risk_allocate(alloc, cfg, fixtures.BASE_TS_MS, {}, {}, specs), {})

        # Zero-signal + zero-vol returns
        decs_zero = {
            "BTC": iface.BarDecision("BTC", fixtures.BASE_TS_MS, 0.0, 0.0, "FLAT"),
            "ETH": iface.BarDecision("ETH", fixtures.BASE_TS_MS, 0.0, 0.0, "FLAT"),
        }
        w_zero = fixtures.call_risk_allocate(alloc, cfg, fixtures.BASE_TS_MS, decs_zero, {"BTC": [0.0] * 50, "ETH": [0.0] * 50}, specs)
        for v in w_zero.values():
            self.assertTrue(math.isfinite(v))
            self.assertAlmostEqual(v, 0.0, places=9)

    def test_t2_fa04_allow_short_false_clamps_negative_decisions_to_zero(self) -> None:
        assets = fixtures.import_assets()
        iface = fixtures.import_core_interfaces()
        risk_mod = fixtures.import_risk_allocator_mod()

        cn_spec = assets.default_ashare_template_spec("600519.SH")
        cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=15.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
        alloc = fixtures.make_risk_allocator(cfg)
        decs = {"600519.SH": iface.BarDecision("600519.SH", fixtures.BASE_TS_MS, -0.8, -5.0, "BEAR")}
        rets = {"600519.SH": fixtures.generate_correlated_return_matrix(1, 60, 0.0, annual_vol=0.25, seed=205)[0]}
        w = fixtures.call_risk_allocate(alloc, cfg, fixtures.BASE_TS_MS, decs, rets, {"600519.SH": cn_spec})
        self.assertGreaterEqual(float(w.get("600519.SH", 0.0)), 0.0, "allow_short=False must clamp negative weight to 0.0")

    def test_t2_fa04_d9_binding_leverage_cap_g7_parity_with_weight_feedback(self) -> None:
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()

        syms = ("BTC", "ETH", "SOL", "LINK")
        specs = assets.default_web3_specs(syms)
        n_bars = 50
        bars_by_sym = {
            s: fixtures.build_bar_series(
                symbol=s, venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS,
                closes=fixtures.build_trending_closes(n_bars, start_price=100.0, step_ret=0.002, noise_std=0.004, seed=210 + i),
                carry_rates=[0.0001] * n_bars,
            )
            for i, s in enumerate(syms)
        }
        # Raw weights sum to 2.4 (> binding max_portfolio_leverage=0.75) with inertia band active
        steps = [{s: (0.60 if t % 3 != 0 else 0.55) for s in syms} for t in range(n_bars)]
        strat = fixtures.build_scripted_strategy(steps, inertia_band=0.08)
        cfg = risk_mod.RiskAllocatorConfig(
            max_positions=4, target_annual_vol_pct=50.0, max_portfolio_leverage=0.75,
            max_single_weight=0.40, use_correlation_idm=True,
        )
        max_w_delta, max_eq_delta = fixtures.run_streaming_vs_batch_check(strat, specs, bars_by_sym, risk_cfg=cfg)
        self.assertLessEqual(max_w_delta, 1e-10)
        self.assertLessEqual(max_eq_delta, 1e-8)


class TestTier2FA05ValidationBoundaries(unittest.TestCase):
    """FA-05 Boundary & Corner Cases (D3 Nested CPCV, D4 DSR Floor, D5 Oldest-Slice PBO, D8 G8, D12 HAC, D13)."""

    def test_t2_fa05_d3_nested_cpcv_purge_embargo_and_zero_test_fold_leakage(self) -> None:
        val_mod = fixtures.import_core_validation_mod()
        split_fn = fixtures.resolve_validation_callable(val_mod, "generate_cpcv_splits")
        splits = split_fn(n_obs=300, n_groups=6, k_test=2, purge_bars=5, embargo_bars=5)
        self.assertGreater(len(splits), 0)
        for sp in splits:
            train_set = set(sp.train_indices)
            test_sorted = sorted(sp.test_indices)
            self.assertTrue(train_set.isdisjoint(set(test_sorted)))
            # Identify contiguous test blocks [a, b]
            blocks: list[tuple[int, int]] = []
            start = test_sorted[0]
            prev = test_sorted[0]
            for idx in test_sorted[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    blocks.append((start, prev))
                    start = idx
                    prev = idx
            blocks.append((start, prev))
            for a, b in blocks:
                for p_idx in range(max(0, a - 5), a):
                    self.assertNotIn(p_idx, train_set, f"D3 purge violation at index {p_idx}")
                for e_idx in range(b + 1, min(300, b + 6)):
                    self.assertNotIn(e_idx, train_set, f"D3 embargo violation at index {e_idx}")

        # Nested refit zero test-fold leakage proof
        nested_fn = fixtures.resolve_validation_callable(
            val_mod,
            "run_nested_cpcv",
            "evaluate_nested_cpcv",
            "nested_cpcv_validate",
            "nested_walk_forward_validate",
        )
        rng = random.Random(303)
        base_series = [0.001 + rng.gauss(0.0, 0.01) for _ in range(240)]
        first_test_set = set(split_fn(n_obs=240, n_groups=6, k_test=2, purge_bars=4, embargo_bars=4)[0].test_indices)
        shocked_series = [v + (5.0 if i in first_test_set else 0.0) for i, v in enumerate(base_series)]

        fitted_base: list[float] = []
        fitted_shock: list[float] = []
        fold_result_cls = getattr(val_mod, "FoldResult", None)

        def _make_fit_eval_cb(series: list[float], bucket: list[float]) -> Any:
            def _fit_and_eval(train_idx: Any, test_idx: Any) -> Any:
                train_vals = [series[i] for i in train_idx]
                mu = sum(train_vals) / max(len(train_vals), 1)
                bucket.append(mu)
                sgn = 1.0 if mu >= 0.0 else -1.0
                oos = [sgn * series[i] for i in test_idx]
                is_r = [sgn * series[i] for i in train_idx]
                if fold_result_cls is not None:
                    return fold_result_cls(
                        oos_returns=oos,
                        is_returns=is_r,
                        selected_params={"sign": sgn, "train_mean": mu},
                    )
                return oos
            return _fit_and_eval

        def _make_refitter(bucket: list[float]) -> Any:
            def _refit(train_idx: list[int], data: Any) -> float:
                train_vals = [data[i] for i in train_idx]
                mean_v = sum(train_vals) / max(len(train_vals), 1)
                bucket.append(mean_v)
                return mean_v
            return _refit

        def _eval_fold(model_param: float, test_idx: list[int], data: Any) -> list[float]:
            sign = 1.0 if model_param >= 0.0 else -1.0
            return [sign * data[i] for i in test_idx]

        sig = inspect.signature(nested_fn)
        if "fit_and_eval_fn" in sig.parameters or "n_obs" in sig.parameters:
            rep_base = nested_fn(
                len(base_series), _make_fit_eval_cb(base_series, fitted_base), 252.0,
                n_groups=6, k_test=2, purge_bars=4, embargo_bars=4,
            )
            _ = nested_fn(
                len(shocked_series), _make_fit_eval_cb(shocked_series, fitted_shock), 252.0,
                n_groups=6, k_test=2, purge_bars=4, embargo_bars=4,
            )
            self.assertTrue(bool(fixtures.get_metric(rep_base, "is_nested", default=True)))
            self.assertEqual(int(fixtures.get_metric(rep_base, "n_splits")), 15)
            self.assertEqual(len(fitted_base), 15)

            # D3-C: Pure zero-mean Gaussian random walk must FAIL Nested CPCV gate
            rw_series = [random.Random(3040).gauss(0.0, 0.01) for _ in range(240)]
            rep_rw = nested_fn(
                len(rw_series), _make_fit_eval_cb(rw_series, []), 252.0,
                n_groups=6, k_test=2, purge_bars=4, embargo_bars=4,
            )
            self.assertTrue(bool(fixtures.get_metric(rep_rw, "is_nested", default=True)))
            self.assertFalse(
                bool(fixtures.get_metric(rep_rw, "passed_cpcv_gate", "passed_nested_cpcv", "passed")),
                "D3-C: Zero-mean Gaussian random walk must fail Nested CPCV gate",
            )
            self.assertLessEqual(float(fixtures.get_metric(rep_rw, "median_oos_sharpe")), 0.0)
        elif "fit_fn" in sig.parameters:
            nested_fn(base_series, fit_fn=_make_refitter(fitted_base), eval_fn=_eval_fold, periods_per_year=252.0)
            nested_fn(shocked_series, fit_fn=_make_refitter(fitted_shock), eval_fn=_eval_fold, periods_per_year=252.0)
        else:
            nested_fn(base_series, _make_refitter(fitted_base), _eval_fold, periods_per_year=252.0)
            nested_fn(shocked_series, _make_refitter(fitted_shock), _eval_fold, periods_per_year=252.0)
        self.assertGreater(len(fitted_base), 0)
        self.assertEqual(
            fitted_base[0],
            fitted_shock[0],
            "D3 leakage: mutating ONLY test_indices of fold 0 altered training-fold fitted parameter!",
        )

    def test_t2_fa05_d4_dsr_null_variance_floor_random_walk_reject_and_trial_penalty(self) -> None:
        val_mod = fixtures.import_core_validation_mod()
        dsr_fn = fixtures.resolve_validation_callable(val_mod, "compute_psr_dsr")

        rng = random.Random(304)
        rets = [0.001 + rng.gauss(0.0, 0.01) for _ in range(400)]
        rep_tiny_std = dsr_fn(rets, periods_per_year=252.0, n_trials=9, trial_sharpe_std_per_period=1e-6)
        self.assertGreater(
            rep_tiny_std.sr_star_annualized,
            0.30,
            f"D4 violation: correlated grid trial_std=1e-6 collapsed sr_star_annualized to {rep_tiny_std.sr_star_annualized:.4f} (v3 bug was 0.047)",
        )

        # Random walk DSR ceiling (< 0.95)
        rw = [rng.gauss(0.0, 0.01) for _ in range(500)]
        rep_rw = dsr_fn(rw, periods_per_year=252.0, n_trials=20)
        self.assertLess(rep_rw.dsr, 0.95, f"D4 violation: random walk passed DSR with {rep_rw.dsr}")

        # Higher cumulative TrialLedger N_trials strictly lowers DSR and raises SR*
        mod_rets = [0.0015 + rng.gauss(0.0, 0.01) for _ in range(400)]
        rep_n5 = dsr_fn(mod_rets, periods_per_year=252.0, n_trials=5, trial_sharpe_std_per_period=0.05)
        rep_n50 = dsr_fn(mod_rets, periods_per_year=252.0, n_trials=50, trial_sharpe_std_per_period=0.05)
        self.assertGreater(rep_n50.sr_star_annualized, rep_n5.sr_star_annualized)
        self.assertLess(rep_n50.dsr, rep_n5.dsr)

    def test_t2_fa05_d5_pbo_slices_oldest_warmup_bars_never_newest_live_bars(self) -> None:
        val_mod = fixtures.import_core_validation_mod()
        pbo_fn = fixtures.resolve_validation_callable(val_mod, "compute_pbo_cscv")

        rng = random.Random(305)
        # M=6 candidates, T=404, S=8 -> chunk_len=50, remainder=4 oldest warmup bars [0:4]
        base = [[rng.gauss(0.0005 * (m == 0), 0.01) for _ in range(404)] for m in range(6)]
        mut_oldest = [list(row) for row in base]
        mut_newest = [list(row) for row in base]
        for m in range(6):
            for k in range(4):
                mut_oldest[m][k] = 0.90 if (m + k) % 2 == 0 else -0.90
                mut_newest[m][400 + k] = 0.90 if (m + k) % 2 == 0 else -0.90

        rep_base = pbo_fn(base, periods_per_year=252.0, n_subsamples=8)
        rep_old = pbo_fn(mut_oldest, periods_per_year=252.0, n_subsamples=8)
        rep_new = pbo_fn(mut_newest, periods_per_year=252.0, n_subsamples=8)

        self.assertEqual(
            abs(rep_base.pbo - rep_old.pbo),
            0.0,
            f"D5 violation: mutating oldest warmup bars [0:4] changed PBO ({rep_base.pbo} vs {rep_old.pbo})",
        )
        self.assertEqual(abs(rep_base.median_is_winner_sharpe - rep_old.median_is_winner_sharpe), 0.0)
        newest_changed = (
            abs(rep_base.pbo - rep_new.pbo) > 1e-12
            or abs(rep_base.median_is_winner_sharpe - rep_new.median_is_winner_sharpe) > 1e-12
            or abs(rep_base.median_oos_winner_sharpe - rep_new.median_oos_winner_sharpe) > 1e-12
        )
        self.assertTrue(newest_changed, "D5 violation: newest live bars [400:404] were dropped instead of oldest warmup bars!")

        # 10 pure independent zero-mean noise variants MUST fail PBO > 0.35
        pure_noise = [[rng.gauss(0.0, 0.01) for _ in range(320)] for _ in range(10)]
        rep_noise = pbo_fn(pure_noise, periods_per_year=252.0, n_subsamples=8)
        self.assertGreater(rep_noise.pbo, 0.35)
        self.assertGreaterEqual(float(rep_noise.prob_oos_loss), 0.0)
        self.assertLessEqual(float(rep_noise.prob_oos_loss), 1.0)
        deg = float(fixtures.get_metric(rep_noise, "plateau_center_oos_degradation"))
        self.assertTrue(math.isfinite(deg) and deg >= 0.0, f"D5-C: Invalid plateau_center_oos_degradation={deg}")

    def test_t2_fa05_d8_g8_benchmark_gate_blocks_underperforming_gold_allows_orthogonal_alpha(self) -> None:
        val_mod = fixtures.import_core_validation_mod()

        def _call(sr: float, cal: float, b_sr: float, b_cal: float, corr: float) -> bool:
            res = fixtures.evaluate_g8_contract(
                val_mod,
                strategy_sharpe=sr,
                strategy_calmar=cal,
                benchmark_sharpe=b_sr,
                benchmark_calmar=b_cal,
                corr_to_benchmark=corr,
            )
            return bool(fixtures.get_metric(res, "passed_benchmark_gate", "passed"))

        # v3 Gold flaw: directional strategy loses to passive Gold B&H -> MUST FAIL G8
        self.assertFalse(_call(0.70, 0.38, 0.80, 0.55, 0.85), "D8: Underperforming directional strategy must fail G8")
        # Beating B&H on Sharpe or Calmar -> PASS
        self.assertTrue(_call(1.15, 0.50, 0.80, 0.55, 0.85), "Beating B&H on Sharpe must pass G8")
        self.assertTrue(_call(0.75, 0.90, 0.80, 0.55, 0.85), "Beating B&H on Calmar must pass G8")
        # Orthogonal carry/crisis alpha (|corr| <= 0.30, Sharpe > 0) -> PASS even if SR < B&H
        self.assertTrue(_call(0.65, 0.40, 0.80, 0.55, 0.10), "Orthogonal positive-Sharpe carry alpha must pass G8")
        # Orthogonal with negative Sharpe -> FAIL
        self.assertFalse(_call(-0.20, -0.10, 0.80, 0.55, 0.10), "Orthogonal negative-Sharpe sleeve must fail G8")

    def test_t2_fa05_d12_hac_phase_stability_t_stat_attenuation_and_sparse_event_expectancy(self) -> None:
        assets = fixtures.import_assets()
        factors_mod = fixtures.import_factors_mod()

        spec = assets.default_web3_specs(("BTC",))["BTC"]
        rng = random.Random(312)
        closes = [100.0]
        for i in range(1, 260):
            drift = 0.006 if (i // 50) % 2 == 0 else -0.004
            closes.append(max(closes[-1] * (1.0 + drift + rng.gauss(0.0, 0.004)), 1.0))
        bars = fixtures.build_bar_series(
            symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=closes,
            carry_rates=[0.0001] * len(closes), use_v3_model=True,
        )
        snaps = factors_mod.StreamingFactorEngine(spec.calendar, 12.0).compute_series(bars)

        # Phase-shift stability across warmup_bars in 40..47
        ics: list[float] = []
        for wb in range(40, 48):
            aud = factors_mod.FactorDirectionAuditor(warmup_bars=wb)
            summ = aud.audit("phase", {"BTC": bars}, {"BTC": snaps}, horizon_bars=8, candidate_factors=["ewmac_16_64"])
            ics.append(float(fixtures.get_metric(summ.results[0], "spearman_ic", "ic_mean")))
        self.assertLess(max(ics) - min(ics), 0.05, f"D12-A: Phase-shift IC dispersion too wide: {ics}")

        # Newey-West HAC t-stat must be smaller than naive unadjusted OLS t-stat on overlapping H=8 persistent series
        aud_hac = factors_mod.FactorDirectionAuditor(warmup_bars=45)
        summ_hac = aud_hac.audit("hac", {"BTC": bars}, {"BTC": snaps}, horizon_bars=8, candidate_factors=["ewmac_16_64"])
        r_hac = summ_hac.results[0]
        # Compute naive overlapping Pearson t-stat directly for comparison
        xs: list[float] = []
        ys: list[float] = []
        for idx in range(45, len(bars) - 8):
            xs.append(float(snaps[idx].values["ewmac_16_64"]))
            ys.append(bars[idx + 8].close / bars[idx].close - 1.0)
        corr_naive = fixtures.sample_pearson_corr(xs, ys)
        n_eff_naive = len(xs)
        t_naive = abs(corr_naive) * math.sqrt(max(n_eff_naive - 2, 1) / max(1.0 - corr_naive * corr_naive, 1e-12))
        self.assertGreater(abs(t_naive), 1.5, f"D12-B setup requires non-trivial naive t-stat (>1.5), got {t_naive:.3f}")
        self.assertLess(
            abs(r_hac.t_stat),
            abs(t_naive),
            f"D12-B: Newey-West HAC t-stat ({r_hac.t_stat:.3f}) must attenuate naive overlapping t-stat ({t_naive:.3f})",
        )

        # D12-C & D12-D: Predictive sparse event (>=12 events) + predictive redundant duplicate pair (|corr| > 0.99)
        rng_d12 = random.Random(3120)
        n_d12 = 280
        event_indices = set(range(50, 260, 14))  # 15 non-overlapping events spaced 14 >> horizon=2
        z_sig = [rng_d12.gauss(0.0, 1.0) for _ in range(n_d12)]
        closes_d12 = [100.0]
        for k in range(1, n_d12):
            ev_boost = 0.025 if (k - 1) in event_indices else 0.0
            r_k = 0.006 * z_sig[k - 1] + ev_boost + rng_d12.gauss(0.0, 0.0035)
            closes_d12.append(max(closes_d12[-1] * (1.0 + r_k), 1.0))
        s_bars = fixtures.build_bar_series(
            symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=closes_d12,
            carry_rates=[0.00005] * n_d12, carry_interval_h=12.0, use_v3_model=True,
        )
        s_snaps = []
        for idx, b in enumerate(s_bars):
            sig_v = 1.0 if idx in event_indices else 0.0
            f1 = z_sig[idx]
            f2 = 2.5 * f1 + 0.05 * rng_d12.gauss(0.0, 1.0)
            s_snaps.append(
                factors_mod.FactorSnapshot(
                    symbol="BTC", venue="hyperliquid", timeframe_h=12.0,
                    ts_event_ms=b.ts_event_ms, ts_avail_ms=b.ts_avail_ms,
                    close=b.close, atr_14=1.0, prior_high_40=b.close, prior_low_40=b.close,
                    prior_high_20=b.close, prior_low_20=b.close,
                    values={"sparse_ev": sig_v, "f_base": f1, "f_dup": f2},
                )
            )
        summ_sp = factors_mod.FactorDirectionAuditor(warmup_bars=40).audit(
            "sparse", {"BTC": s_bars}, {"BTC": s_snaps}, horizon_bars=2,
            candidate_factors=["sparse_ev", "f_base", "f_dup"],
            priors={"sparse_ev": 1, "f_base": 1, "f_dup": 1},
        )
        res_by_name = {r.factor_name: r for r in summ_sp.results}

        # Non-vacuous D12-D: BOTH f_base and f_dup genuinely pass G2, then exactly 1 is pruned as redundant!
        self.assertIn("f_base", summ_sp.passed_g2_factors)
        self.assertIn("f_dup", summ_sp.passed_g2_factors)
        self.assertGreaterEqual(len(summ_sp.passed_g2_factors), 2)
        self.assertGreaterEqual(len(summ_sp.pruned_redundant_pairs), 1, "D12-D: Expected >=1 pruned redundant pair")
        sel = set(summ_sp.selected_non_redundant_factors)
        self.assertTrue(
            ("f_base" in sel) ^ ("f_dup" in sel),
            f"D12-D: Redundant |corr|>0.99 pair must select strictly ONE factor (XOR); selected={sel}",
        )
        self.assertTrue(
            bool(res_by_name["f_base"].pruned_as_redundant) ^ bool(res_by_name["f_dup"].pruned_as_redundant),
            "D12-D: Exactly one of f_base / f_dup must be marked pruned_as_redundant",
        )

        # Non-vacuous D12-C: Sparse event conditional expectancy
        sp_res = res_by_name["sparse_ev"]
        self.assertEqual(str(fixtures.get_metric(sp_res, "audit_mode")), "sparse_event")
        self.assertTrue(bool(sp_res.passed_g2), f"D12-C: Strong sparse event signal must pass G2: {sp_res}")
        self.assertGreaterEqual(int(fixtures.get_metric(sp_res, "event_count")), 12)
        self.assertGreater(
            float(fixtures.get_metric(sp_res, "event_mean_return_bps", "event_expectancy_bps")),
            100.0,
            "D12-C: Sparse event conditional expectancy must exceed +100 bps",
        )

    def test_t2_fa05_d13_short_high_return_window_cagr_exceeds_simple_apr(self) -> None:
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        zero_cost = assets.CostModel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        spec = assets.AssetSpec("FAST", assets.ASSET_CLASS_CRYPTO_PERP, "hyperliquid", assets.CALENDAR_CRYPTO_24_7, zero_cost, 1.0)
        # +45.6% total return in 200 days -> CAGR = ((1.456)^(365/200)-1)*100 = 98.54% > simple APR = 83.22%
        n_days = 201
        g = 1.456 ** (1.0 / (n_days - 1))
        closes = [100.0 * (g ** i) for i in range(n_days)]
        opens = [closes[0]] + [closes[i - 1] for i in range(1, n_days)]
        bars = fixtures.build_bar_series(
            symbol="FAST", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=24.0, start_ms=fixtures.BASE_TS_MS, closes=closes, opens=opens,
            highs=[max(o, c) for o, c in zip(opens, closes)], lows=[min(o, c) for o, c in zip(opens, closes)],
        )
        cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=5000.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False)
        res = acct_mod.PortfolioAccountingEngine().simulate(
            fixtures.build_scripted_strategy([{"FAST": 1.0}] * n_days),
            {"FAST": spec}, {"FAST": bars}, risk_cfg=cfg, capital_usd=10_000.0,
        )
        cagr = float(fixtures.get_metric(res, "cagr_pct"))
        simple = float(fixtures.get_metric(res, "simple_apr_pct"))
        self.assertGreater(cagr, simple, f"D13: Short 200d high-return CAGR ({cagr:.2f}%) must exceed simple APR ({simple:.2f}%)")
        self.assertAlmostEqual(cagr, 98.54, delta=3.0)
        self.assertAlmostEqual(simple, 83.22, delta=3.0)


if __name__ == "__main__":
    unittest.main()
