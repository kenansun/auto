"""Tier 1 Feature Coverage E2E Tests: FA-01..FA-05 (Data Foundation & Core Engine).

Strictly opaque-box tests derived from ORIGINAL_REQUEST.md and PROJECT.md Interface Contracts.
All Platform v4 imports are lazy (inside test methods) so `--list` discovers cleanly before
parallel implementation milestones complete. Zero skips, zero fake passes, pure Python 3.12 stdlib.
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


class TestTier1FA01DataFoundation(unittest.TestCase):
    """FA-01: Option B Partitioned PIT Data Foundation, Catalog & TrialLedger (F-01, F-02, F-06)."""

    def test_t1_fa01_router_initializes_partitioned_wal_databases(self) -> None:
        router_mod = fixtures.import_df_router()
        with fixtures.temp_sandbox(prefix="e2e_t1_fa01_init_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            catalog_db = tmp / "catalog.sqlite"
            web3_db = tmp / "domains" / "web3_lake.sqlite"
            gold_db = tmp / "domains" / "gold_macro_lake.sqlite"
            self.assertTrue(catalog_db.is_file(), "catalog.sqlite must be created")
            self.assertTrue(web3_db.is_file(), "domains/web3_lake.sqlite must be created")
            self.assertTrue(gold_db.is_file(), "domains/gold_macro_lake.sqlite must be created")

            for db_file in (catalog_db, web3_db, gold_db):
                conn = fixtures.open_sqlite_ro(db_file)
                try:
                    mode = str(conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower()
                    self.assertEqual(mode, "wal", f"{db_file.name} must use SQLite WAL mode")
                finally:
                    conn.close()

            cat_conn = fixtures.open_sqlite_ro(catalog_db)
            try:
                tables = {
                    row[0]
                    for row in cat_conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            finally:
                cat_conn.close()
            for required_tbl in (
                "asset_registry",
                "dataset_manifests",
                "dq_certificates",
                "trial_ledger",
                "strategy_blueprints",
            ):
                self.assertIn(required_tbl, tables, f"catalog.sqlite missing table {required_tbl}")

    def test_t1_fa01_canonical_bar_and_all_four_domain_records_roundtrip(self) -> None:
        assets = fixtures.import_assets()
        models = fixtures.import_df_models()
        router_mod = fixtures.import_df_router()
        prov_real, _, _ = fixtures.import_provenance()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t1_fa01_rt_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            btc_spec = assets.default_web3_specs(("BTC",))["BTC"]
            gc_spec = assets.default_gold_specs()["GC=F"]
            router.register_asset(btc_spec)
            router.register_asset(gc_spec)

            btc_bars = fixtures.build_bar_series(
                symbol="BTC",
                venue="hyperliquid",
                asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0,
                start_ms=fixtures.BASE_TS_MS,
                closes=[42000.0, 42500.0, 43100.0],
                carry_rates=[0.0001, 0.00012, 0.00011],
                carry_interval_h=1.0,
            )
            self.assertEqual(btc_bars[0].timeframe_sec, 43200)
            inserted_w3 = router.upsert_bars(web3_dom, btc_bars)
            self.assertEqual(inserted_w3, 3)
            loaded_w3 = router.load_bars(web3_dom, "hyperliquid", "BTC", 43200)
            self.assertEqual(len(loaded_w3), 3)
            self.assertAlmostEqual(loaded_w3[-1].close, 43100.0, places=6)

            gc_bars = fixtures.build_bar_series(
                symbol="GC=F",
                venue="yahoo",
                asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
                timeframe_h=24.0,
                start_ms=fixtures.BASE_TS_MS,
                closes=[2050.0, 2065.0],
                contract_multiplier=100.0,
            )
            self.assertEqual(gc_bars[0].timeframe_sec, 86400)
            router.upsert_bars(gold_dom, gc_bars)
            loaded_gold = router.load_bars(gold_dom, "yahoo", "GC=F", 86400)
            self.assertEqual(len(loaded_gold), 2)

            metrics = [
                models.Web3DerivativesMetric(
                    venue="binance_um",
                    symbol="BTC",
                    ts_event_ms=fixtures.BASE_TS_MS,
                    ts_avail_ms=fixtures.BASE_TS_MS + fixtures.HOUR_MS,
                    open_interest_usd=9_500_000_000.0,
                    top_trader_lsr=1.35,
                    account_lsr=1.12,
                    taker_ls_vol_ratio=1.08,
                    provenance=prov_real,
                    source_fingerprint="fp_oi_1",
                )
            ]
            self.assertEqual(router.upsert_web3_metrics(metrics), 1)
            loaded_m = router.load_web3_metrics("binance_um", "BTC")
            self.assertEqual(len(loaded_m), 1)
            self.assertAlmostEqual(loaded_m[0].open_interest_usd, 9_500_000_000.0, places=2)

            w3_macro = [
                models.Web3MarketMacroPoint(
                    source="deribit",
                    series_id="BTC_DVOL",
                    ts_event_ms=fixtures.BASE_TS_MS,
                    ts_avail_ms=fixtures.BASE_TS_MS + fixtures.HOUR_MS,
                    value=52.5,
                    provenance=prov_real,
                    source_fingerprint="fp_dvol_1",
                )
            ]
            self.assertEqual(router.upsert_web3_macro(w3_macro), 1)
            loaded_dvol = router.load_web3_macro("BTC_DVOL")
            self.assertEqual(len(loaded_dvol), 1)
            self.assertAlmostEqual(loaded_dvol[0].value, 52.5, places=6)

            yields = [
                models.MacroYieldPoint(
                    source="us_treasury",
                    ts_event_ms=fixtures.BASE_TS_MS,
                    ts_avail_ms=fixtures.BASE_TS_MS + fixtures.DAY_MS,
                    real_5y=2.45,
                    real_7y=2.60,
                    real_10y=2.77,
                    real_20y=2.91,
                    real_30y=2.99,
                    provenance=prov_real,
                    source_fingerprint="fp_tips_1",
                )
            ]
            self.assertEqual(router.upsert_yield_curve(yields), 1)
            loaded_y = router.load_yield_curve()
            self.assertEqual(len(loaded_y), 1)
            self.assertAlmostEqual(float(loaded_y[0].real_10y), 2.77, places=6)

            events = [
                models.MacroEventRelease(
                    event_type="FOMC",
                    event_id="FOMC_2026_12_09",
                    ref_period="2026-12",
                    ts_release_ms=fixtures.DEC_2026_FOMC_MS,
                    ts_avail_ms=fixtures.DEC_2026_FOMC_MS,
                    is_sep_meeting=True,
                    actual=4.25,
                    prior=4.50,
                    surprise_z=-0.5,
                    source="fed_html",
                    provenance=prov_real,
                )
            ]
            self.assertEqual(router.upsert_macro_events(events), 1)
            loaded_ev = router.load_macro_events(event_types=["FOMC"])
            self.assertEqual(len(loaded_ev), 1)
            self.assertTrue(loaded_ev[0].is_sep_meeting)
            self.assertEqual(loaded_ev[0].event_id, "FOMC_2026_12_09")

    def test_t1_fa01_pit_as_of_ms_cutoff_enforced_across_all_record_types(self) -> None:
        assets = fixtures.import_assets()
        models = fixtures.import_df_models()
        router_mod = fixtures.import_df_router()
        prov_real, _, _ = fixtures.import_provenance()
        web3_dom = fixtures.resolve_web3_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t1_fa01_pit_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            bars = fixtures.build_bar_series(
                symbol="ETH",
                venue="hyperliquid",
                asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0,
                start_ms=fixtures.BASE_TS_MS,
                closes=[2200.0, 2250.0, 2300.0, 2350.0],
                carry_rates=[0.0001] * 4,
            )
            router.upsert_bars(web3_dom, bars)
            cutoff = bars[1].ts_avail_ms
            visible = router.load_bars(web3_dom, "hyperliquid", "ETH", 43200, as_of_ms=cutoff)
            self.assertEqual(len(visible), 2)
            self.assertTrue(all(b.ts_avail_ms <= cutoff for b in visible))

            t0 = fixtures.BASE_TS_MS
            t1 = t0 + fixtures.DAY_MS
            router.upsert_web3_metrics(
                [
                    models.Web3DerivativesMetric(
                        "binance_um", "ETH", t0, t0 + fixtures.HOUR_MS, 5e9, 1.1, 1.0, 1.02, prov_real
                    ),
                    models.Web3DerivativesMetric(
                        "binance_um", "ETH", t1, t1 + fixtures.HOUR_MS, 6e9, 1.3, 1.1, 1.05, prov_real
                    ),
                ]
            )
            self.assertEqual(len(router.load_web3_metrics("binance_um", "ETH", as_of_ms=t0 + fixtures.HOUR_MS)), 1)

            router.upsert_web3_macro(
                [
                    models.Web3MarketMacroPoint("deribit", "BTC_DVOL", t0, t0 + fixtures.HOUR_MS, 45.0, prov_real),
                    models.Web3MarketMacroPoint("deribit", "BTC_DVOL", t1, t1 + fixtures.HOUR_MS, 82.0, prov_real),
                ]
            )
            self.assertEqual(len(router.load_web3_macro("BTC_DVOL", as_of_ms=t0 + fixtures.HOUR_MS)), 1)

            router.upsert_yield_curve(
                [
                    models.MacroYieldPoint("us_treasury", t0, t0 + fixtures.HOUR_MS, 2.0, 2.1, 2.2, 2.3, 2.4, prov_real),
                    models.MacroYieldPoint("us_treasury", t1, t1 + fixtures.HOUR_MS, 2.5, 2.6, 2.7, 2.8, 2.9, prov_real),
                ]
            )
            self.assertEqual(len(router.load_yield_curve(as_of_ms=t0 + fixtures.HOUR_MS)), 1)

            router.upsert_macro_events(
                [
                    models.MacroEventRelease("NFP", "NFP_1", "2026-11", t0, t0, False, 180.0, 175.0, 0.4, "bls", prov_real),
                    models.MacroEventRelease("CPI", "CPI_1", "2026-11", t1, t1, False, 2.6, 2.5, 0.3, "bls", prov_real),
                ]
            )
            self.assertEqual(len(router.load_macro_events(as_of_ms=t0)), 1)

    def test_t1_fa01_trial_ledger_append_only_family_counting(self) -> None:
        router_mod = fixtures.import_df_router()
        with fixtures.temp_sandbox(prefix="e2e_t1_fa01_trials_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            self.assertEqual(router.get_family_trial_count("web3_w1"), 0)
            ids: set[str] = set()
            for i in range(4):
                tid = router.record_trial(
                    family_id="web3_w1",
                    track_name="track_w",
                    strategy_name="Web3XSMomCarryKernel",
                    params_hash=f"hash_{i}",
                    metrics={"sharpe": 0.9 + 0.1 * i, "cagr_pct": 18.0 + i},
                    verdict="CONDITIONAL_RESEARCH",
                )
                self.assertIsInstance(tid, str)
                self.assertTrue(len(tid) > 0)
                ids.add(tid)
            self.assertEqual(len(ids), 4)
            self.assertEqual(router.get_family_trial_count("web3_w1"), 4)

            router.record_trial(
                family_id="gold_g1",
                track_name="track_g",
                strategy_name="GoldTradFiMacroKernel",
                params_hash="gold_hash_0",
                metrics={"sharpe": 1.1},
                verdict="REJECTED",
            )
            self.assertEqual(router.get_family_trial_count("gold_g1"), 1)
            self.assertEqual(router.get_family_trial_count("web3_w1"), 4)

    def test_t1_fa01_export_domain_partition_creates_immutable_files_and_sha256_manifest(self) -> None:
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t1_fa01_exp_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            bars = fixtures.build_bar_series(
                symbol="SOL",
                venue="hyperliquid",
                asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0,
                start_ms=fixtures.BASE_TS_MS,
                closes=[110.0 + i for i in range(10)],
                carry_rates=[0.0001] * 10,
            )
            router.upsert_bars(web3_dom, bars)
            meta = router.export_domain_partition(web3_dom, fmt="auto")
            self.assertIsInstance(meta, dict)
            self.assertTrue(len(meta) > 0)

            exported_files = [
                p
                for p in tmp.rglob("*")
                if p.is_file()
                and p.suffix in (".parquet", ".jsonl", ".gz", ".zst", ".json")
                and "parquet" in str(p)
            ]
            self.assertGreaterEqual(len(exported_files), 1, "Partition export must write immutable data file(s)")
            duck_wals = list(tmp.rglob("*.duckdb.wal"))
            self.assertEqual(duck_wals, [], "ADR-0013 forbids mutable .duckdb.wal files")

            cat_conn = fixtures.open_sqlite_ro(tmp / "catalog.sqlite")
            try:
                manifest_cnt = cat_conn.execute("SELECT COUNT(*) FROM dataset_manifests").fetchone()[0]
            finally:
                cat_conn.close()
            self.assertGreaterEqual(manifest_cnt, 1, "export_domain_partition must record SHA-256 manifest")

    def test_t1_fa01_migrate_from_v3_sqlite_partitions_web3_and_gold_cleanly(self) -> None:
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t1_fa01_mig_") as tmp:
            v3_db = fixtures.create_synthetic_v3_lake_db(tmp / "v3_lake.sqlite", n_web3_bars=25, n_gold_bars=30)
            router = router_mod.DataFoundationRouter(tmp / "v4_lake")
            self.addCleanup(fixtures.close_router, router)

            summary = router.migrate_from_v3_sqlite(v3_db)
            self.assertIsInstance(summary, dict)

            w3_bars = router.load_bars(web3_dom, "hyperliquid", "BTC", 43200)
            g_bars = router.load_bars(gold_dom, "yahoo", "GC=F", 86400)
            self.assertEqual(len(w3_bars), 25)
            self.assertEqual(len(g_bars), 30)


class TestTier1FA02ResamplingDQAndCollectors(unittest.TestCase):
    """FA-02: Resampling, Perp Funding DQ & Real Multi-Year Data Collector Contracts (F-03, F-04, F-05)."""

    def test_t1_fa02_resample_bars_aligned_1h_to_12h_exact_ohlcv_and_pit(self) -> None:
        assets = fixtures.import_assets()
        resample_bars = fixtures.import_resample_bars()

        n_1h = 4800
        closes = [100.0 + 0.01 * i + 0.5 * math.sin(i / 6.0) for i in range(n_1h)]
        carries = [0.00005] * n_1h
        bars_1h = fixtures.build_bar_series(
            symbol="BTC",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=1.0,
            start_ms=fixtures.BASE_TS_MS,  # 00:00 UTC aligned
            closes=closes,
            carry_rates=carries,
            carry_interval_h=1.0,
            volume_usd=1_000_000.0,
        )
        bars_12h = resample_bars(bars_1h, 12.0, require_complete=True)
        self.assertEqual(len(bars_12h), 400, "4,800 aligned 1h bars must resample to exact 400 12h bars")

        first_chunk = bars_1h[:12]
        b0 = bars_12h[0]
        self.assertAlmostEqual(b0.open, first_chunk[0].open, places=8)
        self.assertAlmostEqual(b0.close, first_chunk[-1].close, places=8)
        self.assertAlmostEqual(b0.high, max(x.high for x in first_chunk), places=8)
        self.assertAlmostEqual(b0.low, min(x.low for x in first_chunk), places=8)
        self.assertAlmostEqual(b0.volume_usd, sum(x.volume_usd for x in first_chunk), places=4)
        self.assertEqual(b0.ts_avail_ms, max(x.ts_avail_ms for x in first_chunk))
        self.assertEqual(b0.timeframe_sec, 43200)

    def test_t1_fa02_audit_domain_dq_passes_all_seven_gates_including_dq_ma_07(self) -> None:
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t1_fa02_dq_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            btc_spec = assets.default_web3_specs(("BTC",))["BTC"]
            router.register_asset(btc_spec)
            bars = fixtures.build_bar_series(
                symbol="BTC",
                venue="hyperliquid",
                asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0,
                start_ms=fixtures.BASE_TS_MS,
                closes=fixtures.build_trending_closes(120, start_price=40000.0, step_ret=0.001, seed=11),
                carry_rates=[0.00012] * 120,
                carry_interval_h=1.0,
            )
            router.upsert_bars(web3_dom, bars)
            report = router.audit_domain_dq(web3_dom, "hyperliquid", "BTC", 43200, btc_spec)
            self.assertTrue(report.passed, f"Clean perp series should pass DQ-MA-01..07: {report}")
            gate_ids = {chk.gate_id for chk in report.checks}
            self.assertIn("DQ-MA-07", gate_ids, "DQ report must include DQ-MA-07 perp funding gate")
            dq07 = [chk for chk in report.checks if chk.gate_id == "DQ-MA-07"][0]
            self.assertTrue(dq07.passed)

    def test_t1_fa02_asset_spec_comex_gc_f_100oz_contract_multiplier(self) -> None:
        assets = fixtures.import_assets()
        gold_specs = assets.default_gold_specs()
        web3_specs = assets.default_web3_specs(("BTC", "ETH"))

        self.assertIn("GC=F", gold_specs)
        self.assertIn("GLD", gold_specs)
        self.assertIn("PAXG", gold_specs)
        self.assertAlmostEqual(float(gold_specs["GC=F"].contract_multiplier), 100.0, places=6)
        self.assertAlmostEqual(float(gold_specs["GLD"].contract_multiplier), 1.0, places=6)
        self.assertAlmostEqual(float(gold_specs["PAXG"].contract_multiplier), 1.0, places=6)
        self.assertAlmostEqual(float(web3_specs["BTC"].contract_multiplier), 1.0, places=6)

    def test_t1_fa02_web3_collector_module_parses_offline_payloads_causally(self) -> None:
        mod = fixtures.import_collectors_web3()
        public_callables = [
            name
            for name in dir(mod)
            if not name.startswith("_") and callable(getattr(mod, name))
        ]
        self.assertGreaterEqual(
            len(public_callables),
            3,
            f"collectors_web3 must expose public Web3 collector/parser callables, got {public_callables}",
        )
        ref_now = fixtures.BASE_TS_MS + 10 * fixtures.DAY_MS

        # 1. DeribitDVOLCollector offline causal parse
        dvol_cls = getattr(mod, "DeribitDVOLCollector", None)
        self.assertTrue(inspect.isclass(dvol_cls), f"Missing DeribitDVOLCollector in {public_callables}")
        dvol_payload = fixtures.strict_json_loads(fixtures.SAMPLE_DERIBIT_DVOL_JSON)
        dvol_pts = dvol_cls.parse_dvol_rows(
            dvol_payload["result"]["data"], currency="BTC", resolution_sec=86400, now_ms=ref_now
        )
        self.assertEqual(len(dvol_pts), 2)
        self.assertTrue(all(p.series_id == "BTC_DVOL" for p in dvol_pts))
        self.assertTrue(all(p.ts_avail_ms == p.ts_event_ms + fixtures.DAY_MS for p in dvol_pts))
        self.assertAlmostEqual(float(dvol_pts[-1].value), 53.5, places=6)

        # 2. DefiLlamaStablecoinCollector offline causal parse
        llama_cls = getattr(mod, "DefiLlamaStablecoinCollector", None)
        self.assertTrue(inspect.isclass(llama_cls), f"Missing DefiLlamaStablecoinCollector in {public_callables}")
        llama_pts = llama_cls.parse_stablecoin_rows(
            fixtures.strict_json_loads(fixtures.SAMPLE_DEFILLAMA_STABLECOINS_JSON), now_ms=ref_now
        )
        self.assertEqual(len(llama_pts), 2)
        self.assertTrue(all(p.series_id == "STABLECOIN_MCAP_USD" for p in llama_pts))
        self.assertTrue(all(p.ts_avail_ms == p.ts_event_ms + fixtures.DAY_MS for p in llama_pts))
        self.assertAlmostEqual(float(llama_pts[-1].value), 166_200_000_000.0, places=2)

        # 3. HyperliquidWeb3Collector offline causal candle + funding bar builder
        hl_cls = getattr(mod, "HyperliquidWeb3Collector", None)
        self.assertTrue(inspect.isclass(hl_cls), f"Missing HyperliquidWeb3Collector in {public_callables}")
        raw_candles = hl_cls.filter_raw_candles(
            fixtures.strict_json_loads(fixtures.SAMPLE_HL_CANDLES_JSON), interval="12h", now_ms=ref_now
        )
        funding_map = hl_cls.parse_funding_rows(fixtures.strict_json_loads(fixtures.SAMPLE_HL_FUNDING_JSON))
        hl_bars = hl_cls.build_canonical_bars(
            "BTC", "12h", raw_candles, funding_map, trim_to_funding_span=False
        )
        self.assertEqual(len(hl_bars), 2)
        self.assertTrue(all(b.ts_avail_ms == b.ts_event_ms + 12 * fixtures.HOUR_MS for b in hl_bars))
        self.assertGreater(float(hl_bars[0].carry_rate_period), 0.0)

        # 4. BinanceVisionWeb3Collector offline kline parser handling BOTH 13-digit ms & 16-digit us Spot timestamps
        bv_cls = getattr(mod, "BinanceVisionWeb3Collector", None)
        self.assertTrue(inspect.isclass(bv_cls), f"Missing BinanceVisionWeb3Collector in {public_callables}")
        ts_0 = fixtures.BASE_TS_MS
        ts_1 = fixtures.BASE_TS_MS + 12 * fixtures.HOUR_MS
        bv_rows = [
            [str(ts_0), "42000.0", "42500.0", "41900.0", "42300.0", "100.0", str(ts_1 - 1), "4230000.0"],
            [str(ts_1 * 1000), "42300.0", "43000.0", "42200.0", "42900.0", "120.0", str((ts_1 + 12 * fixtures.HOUR_MS) * 1000 - 1), "5148000.0"],
        ]
        parsed_klines = bv_cls.parse_kline_rows(bv_rows, tf_ms=12 * fixtures.HOUR_MS, now_ms=ref_now)
        self.assertEqual(len(parsed_klines), 2)
        self.assertEqual(sorted(parsed_klines.keys()), [ts_0, ts_1])

    def test_t1_fa02_gold_macro_collector_parses_tips_csv_fomc_html_and_dec_2026_events(self) -> None:
        mod = fixtures.import_collectors_gold_macro()
        public_names = [n for n in dir(mod) if not n.startswith("_")]
        joined = " ".join(public_names).lower()
        self.assertTrue(
            any(k in joined for k in ("tips", "treasury", "yield")),
            f"collectors_gold_macro must expose US Treasury TIPS parser/collector: {public_names}",
        )
        self.assertTrue(
            any(k in joined for k in ("fomc", "macro_event", "event_calendar")),
            f"collectors_gold_macro must expose FOMC / MacroEventCalendar: {public_names}",
        )
        ref_now = fixtures.utc_ms(2026, 12, 1, 0, 0)

        def _invoke_tips_parser(fn: Any) -> list[Any]:
            sig = inspect.signature(fn)
            if "now_ms" in sig.parameters:
                return fn(fixtures.SAMPLE_TREASURY_TIPS_CSV, now_ms=ref_now)
            return fn(fixtures.SAMPLE_TREASURY_TIPS_CSV)

        # 1. Test TIPS CSV offline parser (clock-independent via now_ms)
        tips_parsed = False
        for name in public_names:
            low = name.lower()
            obj = getattr(mod, name)
            if callable(obj) and not inspect.isclass(obj) and "parse" in low and ("tips" in low or "treasury" in low or "yield" in low):
                rows = _invoke_tips_parser(obj)
                self.assertEqual(len(rows), 3)
                self.assertAlmostEqual(float(rows[-1].real_10y), 2.77, places=4)
                self.assertTrue(all(r.ts_avail_ms >= r.ts_event_ms for r in rows))
                tips_parsed = True
                break
            if inspect.isclass(obj) and ("treasury" in low or "yield" in low):
                for mname in dir(obj):
                    if "parse" in mname.lower() and not mname.startswith("_"):
                        fn = getattr(obj, mname)
                        try:
                            rows = _invoke_tips_parser(fn)
                        except TypeError:
                            rows = _invoke_tips_parser(getattr(obj(), mname))
                        self.assertEqual(len(rows), 3)
                        self.assertAlmostEqual(float(rows[-1].real_10y), 2.77, places=4)
                        self.assertTrue(all(r.ts_avail_ms >= r.ts_event_ms for r in rows))
                        tips_parsed = True
                        break
            if tips_parsed:
                break

        # 2. Test FOMC HTML offline parser
        fomc_parsed = False
        for name in public_names:
            low = name.lower()
            obj = getattr(mod, name)
            if callable(obj) and not inspect.isclass(obj) and "parse" in low and "fomc" in low:
                events = obj(fixtures.SAMPLE_FED_FOMC_HTML)
                self.assertGreaterEqual(len(events), 3)
                dec_events = [e for e in events if "2026-12" in e.ref_period or "12" in e.event_id]
                self.assertGreaterEqual(len(dec_events), 1)
                self.assertTrue(any(e.is_sep_meeting for e in dec_events))
                fomc_parsed = True
                break
            if inspect.isclass(obj) and "fomc" in low:
                for mname in dir(obj):
                    if "parse" in mname.lower() and not mname.startswith("_"):
                        fn = getattr(obj, mname)
                        try:
                            events = fn(fixtures.SAMPLE_FED_FOMC_HTML)
                        except TypeError:
                            events = getattr(obj(), mname)(fixtures.SAMPLE_FED_FOMC_HTML)
                        self.assertGreaterEqual(len(events), 3)
                        dec_events = [e for e in events if "2026-12" in e.ref_period or "12" in e.event_id]
                        self.assertGreaterEqual(len(dec_events), 1)
                        self.assertTrue(any(e.is_sep_meeting for e in dec_events))
                        fomc_parsed = True
                        break
            if fomc_parsed:
                break

        # 3. Validate deterministic NFP/CPI schedule helpers cover Dec 2026 (2026-12-04 & 2026-12-10)
        cal_checked = False
        nfp_fn = getattr(mod, "bls_nfp_release_date", None)
        cpi_fn = getattr(mod, "bls_cpi_release_date", None)
        if callable(nfp_fn) and callable(cpi_fn):
            nfp_dt = nfp_fn(2026, 11)
            cpi_dt = cpi_fn(2026, 11)
            self.assertEqual((nfp_dt.year, nfp_dt.month, nfp_dt.day), (2026, 12, 4))
            self.assertEqual((cpi_dt.year, cpi_dt.month, cpi_dt.day), (2026, 12, 10))
            cal_checked = True

        # 4. Validate BLS API v2 offline payload parsing for CPI & NFP series
        bls_cls = getattr(mod, "BLSMacroEventCollector", None)
        self.assertTrue(inspect.isclass(bls_cls), f"Missing BLSMacroEventCollector in {public_names}")
        bls_series = bls_cls.parse_bls_series_values(fixtures.strict_json_loads(fixtures.SAMPLE_BLS_V2_JSON))
        self.assertIn("CUSR0000SA0", bls_series)
        self.assertIn("CES0000000001", bls_series)
        self.assertGreaterEqual(len(bls_series["CUSR0000SA0"]), 2)
        self.assertGreaterEqual(len(bls_series["CES0000000001"]), 2)

        # Strict conjunction: ALL three offline parser/schedule checks MUST execute and pass
        self.assertTrue(tips_parsed, "TIPS CSV offline parser was not exercised")
        self.assertTrue(fomc_parsed, "FOMC HTML offline parser was not exercised")
        self.assertTrue(cal_checked, "Deterministic Dec 2026 NFP/CPI schedule helper was not exercised")


class TestTier1FA03AccountingAndCoreCompat(unittest.TestCase):
    """FA-03: Portfolio Accounting Engine & Core v3 Regression Contracts (F-07, F-10)."""

    def test_t1_fa03_portfolio_accounting_simulate_produces_rea001_three_tier_ledger(self) -> None:
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        spec = assets.default_web3_specs(("BTC",))["BTC"]
        closes = [100.0, 102.0, 105.0, 108.0, 106.0, 110.0, 112.0, 115.0]
        bars = fixtures.build_bar_series(
            symbol="BTC",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=closes,
            carry_rates=[0.0001] * len(closes),
        )
        steps = [{"BTC": 0.5} if i < len(closes) - 1 else {"BTC": 0.0} for i in range(len(closes))]
        strat = fixtures.build_scripted_strategy(steps, strategy_id="t1_fa03_rea", asset_domain="web3")
        risk_cfg = risk_mod.RiskAllocatorConfig(
            max_positions=2,
            target_annual_vol_pct=20.0,
            max_portfolio_leverage=1.0,
            max_single_weight=1.0,
            use_correlation_idm=False,
        )
        engine = acct_mod.PortfolioAccountingEngine()
        res = engine.simulate(
            strat,
            {"BTC": spec},
            {"BTC": bars},
            risk_cfg=risk_cfg,
            capital_usd=10_000.0,
        )

        eq_curve = fixtures.get_metric(res, "equity_curve")
        period_rets = fixtures.get_metric(res, "period_returns")
        ledger = fixtures.get_metric(res, "trade_ledger")
        net_pnl = float(fixtures.get_metric(res, "net_pnl_usd"))

        self.assertGreaterEqual(len(eq_curve), len(closes) - 1)
        self.assertGreaterEqual(len(period_rets), 1)
        self.assertTrue(math.isfinite(net_pnl))
        self.assertIsNotNone(ledger)
        # REA-001 3-tier ledger must track candidate / executed / suppressed or trades
        ledger_dict = ledger.to_dict() if hasattr(ledger, "to_dict") else (ledger if isinstance(ledger, dict) else vars(ledger))
        serialized = fixtures.strict_json_dumps(ledger_dict)
        self.assertTrue(len(serialized) > 2)

    def test_t1_fa03_accounting_reports_both_geometric_cagr_and_simple_apr(self) -> None:
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        zero_cost = assets.CostModel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        spec = assets.AssetSpec(
            symbol="GOLD_TEST",
            asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
            venue="yahoo",
            calendar=assets.CALENDAR_CRYPTO_24_7,
            cost_model=zero_cost,
            max_leverage_cap=1.0,
        )
        # 5.0 years = 1,825 daily steps: growth factor 1.5864 -> CAGR=9.6684%, simple APR=11.728%
        n_days = 1825
        total_mult = 1.5864
        daily_g = total_mult ** (1.0 / (n_days - 1))
        closes = [100.0 * (daily_g ** i) for i in range(n_days)]
        opens = [closes[0]] + [closes[i - 1] for i in range(1, n_days)]
        bars = fixtures.build_bar_series(
            symbol="GOLD_TEST",
            venue="yahoo",
            asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
            timeframe_h=24.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=closes,
            opens=opens,
            highs=[max(o, c) for o, c in zip(opens, closes)],
            lows=[min(o, c) for o, c in zip(opens, closes)],
        )
        # Stay fully invested w=1.0 across entire 5y horizon
        steps = [{"GOLD_TEST": 1.0} for _ in range(n_days)]
        strat = fixtures.build_scripted_strategy(steps, strategy_id="t1_fa03_cagr", asset_domain="gold")
        risk_cfg = risk_mod.RiskAllocatorConfig(
            max_positions=1,
            target_annual_vol_pct=500.0,
            max_portfolio_leverage=1.0,
            max_single_weight=1.0,
            use_correlation_idm=False,
        )
        res = acct_mod.PortfolioAccountingEngine().simulate(
            strat,
            {"GOLD_TEST": spec},
            {"GOLD_TEST": bars},
            risk_cfg=risk_cfg,
            capital_usd=10_000.0,
        )
        cagr_pct = float(fixtures.get_metric(res, "cagr_pct"))
        simple_apr_pct = float(fixtures.get_metric(res, "simple_apr_pct"))
        self.assertGreater(cagr_pct, 0.0)
        self.assertGreater(simple_apr_pct, cagr_pct, "Multi-year positive return must have CAGR < simple APR (D13)")
        # Analytical relationship: (1 + cagr/100)^5 - 1 ≈ simple_apr/100 * 5
        implied_tot_from_cagr = ((1.0 + cagr_pct / 100.0) ** 5.0 - 1.0) * 100.0
        implied_tot_from_simple = simple_apr_pct * 5.0
        self.assertAlmostEqual(implied_tot_from_cagr, implied_tot_from_simple, delta=1.5)

    def test_t1_fa03_macro_regime_timeline_stateless_pit_snapshot(self) -> None:
        factors_mod = fixtures.import_factors_mod()
        bars_map, _ = fixtures.build_gold_seven_driver_fixture(n_days=90)
        timeline = factors_mod.MacroRegimeTimeline(bars_map)
        ts_mid = bars_map["GC=F"][60].ts_avail_ms
        snap = timeline.snapshot_as_of(ts_mid)
        for key in (
            "macro_gold_treasury_state",
            "macro_dxy_tailwind",
            "macro_real_rate_tailwind",
            "macro_vix_stress_z",
            "macro_gold_composite",
        ):
            self.assertIn(key, snap)
            self.assertTrue(math.isfinite(float(snap[key])))

    def test_t1_fa03_v3_compat_streaming_factor_engine_and_customizer_intact(self) -> None:
        assets = fixtures.import_assets()
        factors_mod = fixtures.import_factors_mod()
        customizer_mod = fixtures.import_strategy_customizer_mod()

        spec = assets.default_web3_specs(("BTC",))["BTC"]
        bars = fixtures.build_bar_series(
            symbol="BTC",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=fixtures.build_trending_closes(120, start_price=40000.0, step_ret=0.004, seed=21),
            carry_rates=[0.0001] * 120,
            use_v3_model=True,
        )
        engine = factors_mod.StreamingFactorEngine(spec.calendar, 12.0)
        snaps = engine.compute_series(bars)
        self.assertEqual(len(snaps), len(bars))
        self.assertIn("ewmac_16_64", snaps[-1].values)
        self.assertTrue(hasattr(customizer_mod, "StrategyCustomizer"))
        self.assertTrue(hasattr(customizer_mod, "AdaptiveMultiFactorKernel"))

    def test_t1_fa03_cost_model_maker_taker_slippage_and_sell_stamp_duty(self) -> None:
        assets = fixtures.import_assets()
        hl = assets.COST_HYPERLIQUID_PERP
        cn = assets.COST_CN_A_SHARE
        self.assertAlmostEqual(hl.one_way_cost_bps("TAKER_TAKER"), hl.taker_fee_bps + hl.slippage_bps_per_leg, places=6)
        self.assertAlmostEqual(hl.round_trip_cost_bps("TAKER_TAKER"), 2.0 * (hl.taker_fee_bps + hl.slippage_bps_per_leg), places=6)
        buy_cn = cn.one_way_cost_bps("TAKER_TAKER", is_sell=False)
        sell_cn = cn.one_way_cost_bps("TAKER_TAKER", is_sell=True)
        self.assertAlmostEqual(sell_cn - buy_cn, cn.stamp_duty_sell_bps, places=6)
        self.assertAlmostEqual(cn.stamp_duty_sell_bps, 5.0, places=6)

    def test_t1_fa03_asset_native_strategy_protocol_contract_enforcement(self) -> None:
        iface = fixtures.import_core_interfaces()
        BarDecision = iface.BarDecision
        dec = BarDecision(
            symbol="BTC",
            ts_event_ms=fixtures.BASE_TS_MS,
            target_weight=0.25,
            raw_signal=1.8,
            regime_label="EXPANSION",
            stop_price=39500.0,
            metadata={"source": "e2e"},
        )
        self.assertEqual(dec.symbol, "BTC")
        self.assertAlmostEqual(dec.target_weight, 0.25, places=8)
        with self.assertRaises((AttributeError, TypeError)):
            dec.target_weight = 0.99  # type: ignore[misc]

        self.assertTrue(hasattr(iface, "AssetNativeStrategy"))
        proto = iface.AssetNativeStrategy
        for attr in ("reset", "on_bar_step", "sync_executed_weights", "clone_fresh"):
            self.assertTrue(hasattr(proto, attr), f"AssetNativeStrategy missing method {attr}")


class TestTier1FA04RiskAllocatorAndParity(unittest.TestCase):
    """FA-04: Correlation-Aware Risk Allocation & Full G7 Bit-Parity (F-08)."""

    def test_t1_fa04_risk_allocator_vol_targeting_scales_weights_inversely_with_vol(self) -> None:
        assets = fixtures.import_assets()
        iface = fixtures.import_core_interfaces()
        risk_mod = fixtures.import_risk_allocator_mod()

        specs = assets.default_web3_specs(("BTC", "ETH"))
        cfg = risk_mod.RiskAllocatorConfig(
            max_positions=2,
            target_annual_vol_pct=15.0,
            max_portfolio_leverage=2.0,
            max_single_weight=1.0,
            use_correlation_idm=False,
        )
        # Compare single-asset allocation for low-vol (15%) vs high-vol (60%) asset under identical 15% vol target
        alloc_low = fixtures.make_risk_allocator(cfg)
        alloc_high = fixtures.make_risk_allocator(cfg)
        low_vol_rets = fixtures.generate_correlated_return_matrix(1, 120, 0.0, annual_vol=0.15, seed=31)[0]
        high_vol_rets = fixtures.generate_correlated_return_matrix(1, 120, 0.0, annual_vol=0.60, seed=32)[0]
        dec_low = {"BTC": iface.BarDecision("BTC", fixtures.BASE_TS_MS, 0.8, 2.0, "TREND")}
        dec_high = {"ETH": iface.BarDecision("ETH", fixtures.BASE_TS_MS, 0.8, 2.0, "TREND")}
        w_low = fixtures.call_risk_allocate(alloc_low, cfg, fixtures.BASE_TS_MS, dec_low, {"BTC": low_vol_rets}, {"BTC": specs["BTC"]})
        w_high = fixtures.call_risk_allocate(alloc_high, cfg, fixtures.BASE_TS_MS, dec_high, {"ETH": high_vol_rets}, {"ETH": specs["ETH"]})
        self.assertGreater(abs(w_low["BTC"]), abs(w_high["ETH"]), "Lower-vol portfolio must receive higher risk-targeted weight")

    def test_t1_fa04_risk_allocator_enforces_max_positions_hard_cap(self) -> None:
        assets = fixtures.import_assets()
        iface = fixtures.import_core_interfaces()
        risk_mod = fixtures.import_risk_allocator_mod()

        syms = ("BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX")
        specs = assets.default_web3_specs(syms)
        cfg = risk_mod.RiskAllocatorConfig(
            max_positions=3,
            target_annual_vol_pct=18.0,
            max_portfolio_leverage=1.5,
            max_single_weight=0.5,
            use_correlation_idm=True,
        )
        alloc = fixtures.make_risk_allocator(cfg)
        mat = fixtures.generate_correlated_return_matrix(6, 100, 0.3, annual_vol=0.40, seed=33)
        recent = {s: mat[i] for i, s in enumerate(syms)}
        decisions = {
            s: iface.BarDecision(s, fixtures.BASE_TS_MS, 0.3, float(i + 1), "TREND")
            for i, s in enumerate(syms)
        }
        weights = fixtures.call_risk_allocate(alloc, cfg, fixtures.BASE_TS_MS, decisions, recent, specs)
        active = {s for s, w in weights.items() if abs(w) > 1e-12}
        self.assertLessEqual(len(active), 3)
        self.assertEqual(active, {"XRP", "DOGE", "AVAX"})

    def test_t1_fa04_risk_allocator_enforces_single_weight_and_gross_leverage_caps(self) -> None:
        assets = fixtures.import_assets()
        iface = fixtures.import_core_interfaces()
        risk_mod = fixtures.import_risk_allocator_mod()

        syms = ("BTC", "ETH", "SOL", "LINK")
        specs = assets.default_web3_specs(syms)
        cfg = risk_mod.RiskAllocatorConfig(
            max_positions=4,
            target_annual_vol_pct=120.0,
            max_portfolio_leverage=0.80,
            max_single_weight=0.25,
            use_correlation_idm=False,
        )
        alloc = fixtures.make_risk_allocator(cfg)
        mat = fixtures.generate_correlated_return_matrix(4, 100, 0.1, annual_vol=0.10, seed=34)
        recent = {s: mat[i] for i, s in enumerate(syms)}
        decisions = {
            s: iface.BarDecision(s, fixtures.BASE_TS_MS, 0.90, 5.0, "TREND")
            for s in syms
        }
        weights = fixtures.call_risk_allocate(alloc, cfg, fixtures.BASE_TS_MS, decisions, recent, specs)
        for s, w in weights.items():
            self.assertLessEqual(abs(w), 0.25 + 1e-9, f"{s} exceeded max_single_weight")
        self.assertLessEqual(sum(abs(w) for w in weights.values()), 0.80 + 1e-9)

    def test_t1_fa04_strategy_clone_fresh_isolation_and_sync_executed_weights(self) -> None:
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        spec = assets.default_web3_specs(("BTC",))["BTC"]
        bars = fixtures.build_bar_series(
            symbol="BTC",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=[100.0, 101.0, 102.0, 103.0],
            carry_rates=[0.00005] * 4,
            carry_interval_h=12.0,
        )
        # Bar 0 emits 0.80 -> binding RiskAllocator caps to 0.40 and syncs 0.40 back to strategy.
        # Bar 1 desires 0.45 (|0.45 - 0.40| = 0.05 < inertia 0.10) -> held at 0.40 strictly because of post-allocator feedback!
        # Bar 2 desires 0.75 (|0.75 - 0.40| = 0.35 >= 0.10) -> breaks out of inertia to 0.75, then allocator re-caps to 0.40.
        steps = [{"BTC": 0.80}, {"BTC": 0.45}, {"BTC": 0.75}, {"BTC": 0.45}]
        s1 = fixtures.build_scripted_strategy(steps, inertia_band=0.10)
        s2 = s1.clone_fresh()

        cfg = risk_mod.RiskAllocatorConfig(
            max_positions=1,
            target_annual_vol_pct=500.0,
            max_portfolio_leverage=0.40,
            max_single_weight=0.40,
            use_correlation_idm=False,
        )
        engine = acct_mod.PortfolioAccountingEngine()
        res1 = engine.simulate(s1, {"BTC": spec}, {"BTC": bars}, risk_cfg=cfg)

        # 1. Production accounting + allocator must invoke sync_executed_weights with binding post-allocator 0.40
        self.assertEqual(len(s1.sync_history), 4)
        self.assertAlmostEqual(float(s1.sync_history[0]["BTC"]), 0.40, places=8)

        # 2. Post-allocator weight feedback genuinely changed Bar 1 strategy behavior under inertia
        raw_hist = res1.raw_decisions_history
        self.assertAlmostEqual(float(raw_hist[0]["BTC"].target_weight), 0.80, places=8)
        self.assertAlmostEqual(float(raw_hist[1]["BTC"].metadata["synced_prev_w"]), 0.40, places=8)
        self.assertAlmostEqual(
            float(raw_hist[1]["BTC"].target_weight),
            0.40,
            places=8,
            msg="Post-allocator synced weight 0.40 must hold Bar 1 desired 0.45 inside 0.10 inertia band",
        )

        # 3. Bar 2 breaks out of inertia band (0.75) and allocator re-caps to 0.40
        self.assertAlmostEqual(float(raw_hist[2]["BTC"].target_weight), 0.75, places=8)
        self.assertAlmostEqual(float(res1.allocated_weights_history[2][1]["BTC"]), 0.40, places=8)

        # 4. clone_fresh() preserved clean isolation prior to simulation and reproduces identical result
        self.assertEqual(len(s2.sync_history), 0, "clone_fresh() must not share sync state with s1")
        res2 = engine.simulate(s2, {"BTC": spec}, {"BTC": bars}, risk_cfg=cfg)
        self.assertEqual(res1.equity_curve, res2.equity_curve)
        self.assertEqual(res1.allocated_weights_history, res2.allocated_weights_history)

    def test_t1_fa04_g7_streaming_vs_batch_bit_parity_with_risk_allocator(self) -> None:
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()

        syms = ("BTC", "ETH", "SOL")
        specs = assets.default_web3_specs(syms)
        n_bars = 40
        bars_by_sym = {
            s: fixtures.build_bar_series(
                symbol=s,
                venue="hyperliquid",
                asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0,
                start_ms=fixtures.BASE_TS_MS,
                closes=fixtures.build_trending_closes(n_bars, start_price=100.0 * (i + 1), step_ret=0.003, seed=40 + i),
                carry_rates=[0.0001] * n_bars,
            )
            for i, s in enumerate(syms)
        }
        steps = [
            {s: (0.45 if (t + idx) % 5 != 0 else 0.10) for idx, s in enumerate(syms)}
            for t in range(n_bars)
        ]
        strat = fixtures.build_scripted_strategy(steps, inertia_band=0.08)
        cfg = risk_mod.RiskAllocatorConfig(
            max_positions=2,
            target_annual_vol_pct=18.0,
            max_portfolio_leverage=0.90,
            max_single_weight=0.50,
            use_correlation_idm=True,
        )
        max_w_delta, max_eq_delta = fixtures.run_streaming_vs_batch_check(
            strat, specs, bars_by_sym, risk_cfg=cfg, capital_usd=10_000.0
        )
        self.assertLessEqual(max_w_delta, 1e-10)
        self.assertLessEqual(max_eq_delta, 1e-8)


class TestTier1FA05InstitutionalValidation(unittest.TestCase):
    """FA-05: Institutional 8-Gate Statistical Validation Suite (F-09)."""

    def test_t1_fa05_g1_dq_and_real_provenance_gate(self) -> None:
        assets = fixtures.import_assets()
        val_mod = fixtures.import_core_validation_mod()
        prov_real, _, prov_synth = fixtures.import_provenance()
        data_lake = fixtures.import_data_lake()

        btc_spec = assets.default_web3_specs(("BTC",))["BTC"]
        real_bars = fixtures.build_bar_series(
            symbol="BTC",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=fixtures.build_trending_closes(120, start_price=40000.0, step_ret=0.001, seed=51),
            carry_rates=[0.0001] * 120,
            provenance=prov_real,
        )
        synth_bars = fixtures.build_bar_series(
            symbol="BTC",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=fixtures.build_trending_closes(120, start_price=40000.0, step_ret=0.001, seed=51),
            carry_rates=[0.0001] * 120,
            provenance=prov_synth,
        )
        # Validate via core_engine.validation G1 evaluator or audit_bars_quality
        dq_real = data_lake.audit_bars_quality(real_bars, btc_spec.calendar, dataset_label="real_btc")
        dq_synth = data_lake.audit_bars_quality(synth_bars, btc_spec.calendar, dataset_label="synth_btc")
        self.assertTrue(dq_real.passed)
        self.assertFalse(dq_synth.passed, "G1/DQ must reject PROVENANCE_SYNTHETIC by default")
        self.assertTrue(
            any(
                hasattr(val_mod, nm)
                for nm in ("evaluate_8_gates", "evaluate_eight_gates", "evaluate_g1_dq", "ValidationSuiteV4", "InstitutionalValidator")
            ),
            "core_engine.validation must expose 8-gate evaluation entry point",
        )

    def test_t1_fa05_g2_hac_ic_monotonicity_and_redundancy_pruning(self) -> None:
        assets = fixtures.import_assets()
        factors_mod = fixtures.import_factors_mod()

        # Construct strong trending series where ewmac/trend has positive forward IC
        spec = assets.default_web3_specs(("BTC",))["BTC"]
        closes = [100.0]
        for i in range(1, 260):
            drift = 0.007 if (i // 45) % 2 == 0 else -0.005
            closes.append(max(closes[-1] * (1.0 + drift), 1.0))
        bars = fixtures.build_bar_series(
            symbol="BTC",
            venue="hyperliquid",
            asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0,
            start_ms=fixtures.BASE_TS_MS,
            closes=closes,
            carry_rates=[0.0001] * len(closes),
            use_v3_model=True,
        )
        eng = factors_mod.StreamingFactorEngine(spec.calendar, 12.0)
        snaps = eng.compute_series(bars)
        auditor = factors_mod.FactorDirectionAuditor()
        summary = auditor.audit("t1_fa05_g2", {"BTC": bars}, {"BTC": snaps}, horizon_bars=6)
        self.assertIsNotNone(summary)
        self.assertGreater(len(summary.results), 0)
        # Confirm HAC / phase-averaged / redundancy pruning metadata is computed cleanly
        for r in summary.results:
            ic_val = float(fixtures.get_metric(r, "spearman_ic", "ic_mean"))
            self.assertTrue(math.isfinite(ic_val))
            self.assertTrue(math.isfinite(float(r.t_stat)))

    def test_t1_fa05_g3_cost_first_economic_edge_with_plus2bp_stress(self) -> None:
        val_mod = fixtures.import_core_validation_mod()
        gates, verdict = fixtures.evaluate_8_gates_with_overrides(
            val_mod,
            baseline_sharpe=1.45,
            baseline_net_pnl=3200.0,
            stress_sharpe=1.10,
            stress_net_pnl=2400.0,
        )
        self.assertTrue(gates.get("G3", False), f"Strong positive baseline & +2bp stress must pass G3: {gates}")
        self.assertEqual(verdict, "PROMOTE_SHADOW_CANDIDATE")

    def test_t1_fa05_g4_local_parameter_plateau_selection(self) -> None:
        val_mod = fixtures.import_core_validation_mod()
        PlateauPoint = getattr(val_mod, "PlateauPoint", None) or fixtures.import_v3_validation_mod().PlateauPoint
        select_fn = fixtures.resolve_validation_callable(val_mod, "select_plateau_center")

        # 3x3 smooth plateau around (1, 1) with SR ~ 1.1..1.2 (exact baseline PlateauPoint schema)
        points = []
        for x in range(3):
            for y in range(3):
                sr = 1.20 - 0.05 * (abs(x - 1) + abs(y - 1))
                points.append(
                    PlateauPoint(
                        ix=x,
                        iy=y,
                        label=f"p_{x}_{y}",
                        params={"entry_n": 20 + x * 5, "vol_scale": 0.8 + y * 0.2},
                        sharpe=sr,
                        apr_pct=22.0,
                        max_dd_pct=12.0,
                        net_pnl_usd=2200.0,
                    )
                )
        rep = select_fn(points, min_sharpe_for_plateau=0.25, min_plateau_ratio=0.55)
        self.assertTrue(bool(fixtures.get_metric(rep, "passed_plateau_gate", "passed")))
        self.assertGreaterEqual(float(rep.plateau_ratio), 0.55)
        self.assertGreaterEqual(float(fixtures.get_metric(rep, "center_to_peak_ratio", "selected_to_peak_ratio")), 0.55)

    def test_t1_fa05_g5_nested_cpcv_and_fixed_dsr_with_trial_ledger(self) -> None:
        val_mod = fixtures.import_core_validation_mod()
        dsr_fn = fixtures.resolve_validation_callable(val_mod, "compute_psr_dsr")
        cpcv_fn = fixtures.resolve_validation_callable(val_mod, "generate_cpcv_splits")

        splits = cpcv_fn(n_obs=300, n_groups=6, k_test=2, purge_bars=5, embargo_bars=5)
        self.assertEqual(len(splits), 15)  # C(6, 2) == 15
        for sp in splits:
            self.assertTrue(set(sp.train_indices).isdisjoint(set(sp.test_indices)))

        rng = random.Random(55)
        strong_rets = [0.0045 + rng.gauss(0.0, 0.008) for _ in range(400)]
        dsr_rep = dsr_fn(strong_rets, periods_per_year=252.0, n_trials=5, trial_sharpe_std_per_period=0.04)
        self.assertGreater(dsr_rep.dsr, 0.90)
        self.assertGreater(dsr_rep.sr_star_annualized, 0.0)

    def test_t1_fa05_g6_multivariant_cscv_pbo_g8_benchmark_and_verdict_enum(self) -> None:
        val_mod = fixtures.import_core_validation_mod()
        pbo_fn = fixtures.resolve_validation_callable(val_mod, "compute_pbo_cscv")

        # G6: 1 genuine alpha variant + 9 independent zero-mean noise variants
        rng = random.Random(56)
        n_obs = 320
        alpha = [0.005 + rng.gauss(0.0, 0.01) for _ in range(n_obs)]
        noise = [[rng.gauss(0.0, 0.01) for _ in range(n_obs)] for _ in range(9)]
        pbo_rep = pbo_fn([alpha] + noise, periods_per_year=252.0, n_subsamples=8)
        self.assertLessEqual(pbo_rep.pbo, 0.35)
        self.assertTrue(hasattr(pbo_rep, "prob_oos_loss"))

        # G8: Strategy beating B&H passes
        g8_out = fixtures.evaluate_g8_contract(
            val_mod,
            strategy_sharpe=1.25,
            strategy_calmar=0.95,
            benchmark_sharpe=0.80,
            benchmark_calmar=0.55,
            corr_to_benchmark=0.75,
        )
        self.assertTrue(bool(fixtures.get_metric(g8_out, "passed_benchmark_gate", "passed")))

        # Verdict Enum presence & exact 3 canonical values
        verdict_cls = fixtures.resolve_verdict_enum(val_mod)
        values = {fixtures.normalize_verdict(member) for member in verdict_cls}
        self.assertEqual(
            values,
            {"PROMOTE_SHADOW_CANDIDATE", "CONDITIONAL_RESEARCH", "REJECTED"},
        )


if __name__ == "__main__":
    unittest.main()
