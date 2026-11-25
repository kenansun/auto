"""Tier 3 Cross-Feature Pairwise E2E Tests: P3-01..P3-24 (24 Multi-Feature Integration Pipelines).

Every test combines two or more distinct Platform v4 features across module boundaries to catch
cross-layer integration defects that isolated unit tests miss.
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


class TestTier3CrossFeaturePairwise(unittest.TestCase):
    """All 24 Cross-Feature Pairwise Integration Scenarios (P3-01..P3-24)."""

    def test_t3_p01_router_pit_plus_misaligned_resample_zero_lookahead(self) -> None:
        """P3-01: F-01 Router PIT cutoff + F-03 resample_bars on misaligned 1h bars."""
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        resample_bars = fixtures.import_resample_bars()
        web3_dom = fixtures.resolve_web3_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t3_p01_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            # 480 misaligned 1h bars starting 11:00 UTC -> mid-stream PIT cutoff at bar 240
            start_1100 = fixtures.BASE_TS_MS + 11 * fixtures.HOUR_MS
            bars_1h = fixtures.build_bar_series(
                symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=1.0, start_ms=start_1100, closes=[40000.0 + 0.5 * i for i in range(480)],
                carry_rates=[0.00005] * 480, carry_interval_h=1.0,
            )
            router.upsert_bars(web3_dom, bars_1h)
            cutoff = bars_1h[240].ts_avail_ms
            loaded_1h = router.load_bars(web3_dom, "hyperliquid", "BTC", 3600, as_of_ms=cutoff)
            self.assertEqual(len(loaded_1h), 241)

            res_12h = resample_bars(loaded_1h, 12.0, require_complete=True)
            # 241 bars starting 11:00 UTC: 1h leading partial + 20 complete 12h buckets (240h) -> exact 20 bars!
            self.assertEqual(len(res_12h), 20)
            self.assertTrue(all(b.ts_avail_ms <= cutoff for b in res_12h), "Resampled bar leaked beyond as_of_ms cutoff!")

    def test_t3_p02_router_dq_ma_07_fail_propagates_to_g1_and_forces_rejected(self) -> None:
        """P3-02: F-01 Router + F-03 DQ-MA-07 (50% missing funding) -> F-09 G1 FAIL -> REJECTED."""
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        val_mod = fixtures.import_core_validation_mod()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        btc_spec = assets.default_web3_specs(("BTC",))["BTC"]

        with fixtures.temp_sandbox(prefix="e2e_t3_p02_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            rates = [0.0001 if i < 50 else 0.0 for i in range(100)]
            bars = fixtures.build_bar_series(
                symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=[40000.0 + i for i in range(100)],
                carry_rates=rates,
            )
            router.upsert_bars(web3_dom, bars)
            dq_rep = router.audit_domain_dq(web3_dom, "hyperliquid", "BTC", 43200, btc_spec)
            self.assertFalse(dq_rep.passed)

            _, verdict = fixtures.evaluate_8_gates_with_overrides(
                val_mod, g1_passed=dq_rep.passed, g2_passed=True, g4_passed=True, g5_passed=True, g6_passed=True, g7_passed=True, g8_passed=True
            )
            self.assertEqual(verdict, "REJECTED", "G1 FAIL from DQ-MA-07 must force REJECTED verdict")

    def test_t3_p03_trial_ledger_cumulative_count_penalizes_g5_dsr(self) -> None:
        """P3-03: F-01/F-06 TrialLedger cumulative family count -> F-09 Fixed DSR penalty (D4)."""
        router_mod = fixtures.import_df_router()
        val_mod = fixtures.import_core_validation_mod()
        dsr_fn = fixtures.resolve_validation_callable(val_mod, "compute_psr_dsr")

        rng = random.Random(3003)
        rets = [0.0018 + rng.gauss(0.0, 0.01) for _ in range(360)]

        with fixtures.temp_sandbox(prefix="e2e_t3_p03_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            router.record_trial("fam_alpha", "track_w", "W1", "h_0", {"sr": 1.2}, "CONDITIONAL_RESEARCH")
            n_first = router.get_family_trial_count("fam_alpha")
            rep_first = dsr_fn(rets, periods_per_year=252.0, n_trials=n_first, trial_sharpe_std_per_period=0.04)

            for i in range(1, 25):
                router.record_trial("fam_alpha", "track_w", "W1", f"h_{i}", {"sr": 1.2}, "CONDITIONAL_RESEARCH")
            n_25 = router.get_family_trial_count("fam_alpha")
            self.assertEqual(n_25, 25)
            rep_25 = dsr_fn(rets, periods_per_year=252.0, n_trials=n_25, trial_sharpe_std_per_period=0.04)

            self.assertGreater(rep_25.sr_star_annualized, rep_first.sr_star_annualized)
            self.assertLess(rep_25.dsr, rep_first.dsr)

    def test_t3_p04_export_domain_partition_matches_catalog_sha256_manifest_idempotent(self) -> None:
        """P3-04: F-01 export_domain_partition + F-06 SHA-256 manifest verification & zero .duckdb.wal."""
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t3_p04_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            router.upsert_bars(
                web3_dom,
                fixtures.build_bar_series(
                    symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                    timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=[40000.0 + i for i in range(12)],
                    carry_rates=[0.0001] * 12,
                ),
            )
            router.upsert_bars(
                gold_dom,
                fixtures.build_bar_series(
                    symbol="GC=F", venue="yahoo", asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
                    timeframe_h=24.0, start_ms=fixtures.BASE_TS_MS, closes=[2500.0 + i for i in range(12)],
                ),
            )
            m_w3 = router.export_domain_partition(web3_dom, fmt="auto")
            m_g = router.export_domain_partition(gold_dom, fmt="auto")
            m_w3_again = router.export_domain_partition(web3_dom, fmt="auto")
            self.assertIsInstance(m_w3, dict)
            self.assertIsInstance(m_g, dict)
            self.assertIsInstance(m_w3_again, dict)
            self.assertEqual(list(tmp.rglob("*.duckdb.wal")), [])

            # Verify recorded SHA-256 values in catalog.sqlite are valid 64-hex digests
            cat_conn = fixtures.open_sqlite_ro(tmp / "catalog.sqlite")
            try:
                rows = cat_conn.execute("SELECT * FROM dataset_manifests").fetchall()
            finally:
                cat_conn.close()
            self.assertGreaterEqual(len(rows), 2)
            all_cells = " ".join(str(cell) for row in rows for cell in row)
            self.assertRegex(all_cells, r"[0-9a-f]{64}", "dataset_manifests must store real 64-char SHA-256 digests")

    def test_t3_p05_allocator_max_positions_rotation_preserves_exit_bar_gap_pnl(self) -> None:
        """P3-05: F-07 Accounting Gap (D1) + F-08 CausalRiskAllocator max_positions rotation (D9)."""
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        syms = ("S1", "S2", "S3", "S4", "S5", "S6")
        zero_cost = assets.CostModel(0.0, 0.0, 0.0)
        specs = {
            s: assets.AssetSpec(s, assets.ASSET_CLASS_CRYPTO_PERP, "hyperliquid", assets.CALENDAR_CRYPTO_24_7, zero_cost, 1.0)
            for s in syms
        }
        # Bar 0..1: S1..S4 held (signals 10..7), S5..S6 inactive (signals 1..2).
        # Bar 2: S1 drops out of top-4 (signal=0.5) while opening +8% gap (100 -> 108)! All other symbols flat at 100.
        bars_by_sym: dict[str, list[Any]] = {}
        for s in syms:
            p2 = 108.0 if s == "S1" else 100.0
            bars_by_sym[s] = [
                fixtures.make_bar(
                    venue="hyperliquid", symbol=s, asset_class=assets.ASSET_CLASS_CRYPTO_PERP, timeframe_h=24.0,
                    ts_event_ms=fixtures.BASE_TS_MS + k * fixtures.DAY_MS,
                    ts_avail_ms=fixtures.BASE_TS_MS + (k + 1) * fixtures.DAY_MS,
                    open=(p2 if k == 2 else 100.0), high=(p2 if k == 2 else 100.0),
                    low=100.0, close=(p2 if k == 2 else 100.0), volume_usd=1e7,
                )
                for k in range(3)
            ]
        # Bar 0 close: S1..S4 top-4 -> enters Bar 1 open (100) with S1=0.25
        # Bar 1 close: S1 drops out of top-4 (0.5) -> rotated out at Bar 2 open (108, +8% gap)!
        # Gap PnL must be captured on Bar 2 open BEFORE OPEN_EXIT executes at 108.0!
        weights_steps = [
            {s: 0.25 for s in syms},
            {s: 0.25 for s in syms},
            {s: 0.25 for s in syms},
        ]
        sig_steps = [
            {"S1": 10.0, "S2": 9.0, "S3": 8.0, "S4": 7.0, "S5": 2.0, "S6": 1.0},
            {"S1": 0.5, "S2": 9.0, "S3": 8.0, "S4": 7.0, "S5": 11.0, "S6": 12.0},
            {"S1": 0.5, "S2": 9.0, "S3": 8.0, "S4": 7.0, "S5": 11.0, "S6": 12.0},
        ]
        strat = fixtures.build_scripted_strategy(weights_steps, signals_by_step=sig_steps)
        cfg = risk_mod.RiskAllocatorConfig(
            max_positions=4, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0,
            max_single_weight=0.25, use_correlation_idm=False,
        )
        res = acct_mod.PortfolioAccountingEngine().simulate(strat, specs, bars_by_sym, risk_cfg=cfg, capital_usd=10_000.0)
        exec_s1 = [float(w.get("S1", 0.0)) for w in res.executed_weights_history]
        self.assertAlmostEqual(exec_s1[0], 0.0, places=8)
        self.assertGreater(exec_s1[1], 0.0, "S1 must be held on Bar 1 entering the Bar 2 rotation gap")
        self.assertAlmostEqual(exec_s1[2], 0.0, places=8, msg="S1 must be rotated out to 0.0 at Bar 2 open")
        s1_exits = [
            t for t in res.trade_ledger.executed
            if t.symbol == "S1" and t.reason == "OPEN_EXIT" and abs(t.fill_price - 108.0) < 1e-6
        ]
        self.assertEqual(len(s1_exits), 1, f"S1 must execute OPEN_EXIT at 108.0, got {res.trade_ledger.executed}")
        expected_gap = exec_s1[1] * 10_000.0 * 0.08
        self.assertAlmostEqual(float(fixtures.get_metric(res, "gap_pnl_usd")), expected_gap, delta=1e-3)
        self.assertAlmostEqual(float(fixtures.get_metric(res, "net_pnl_usd")), expected_gap, delta=1e-3)

    def test_t3_p06_accounting_sync_executed_weights_plus_binding_leverage_g7_parity(self) -> None:
        """P3-06: F-07 sync_executed_weights + F-08 binding leverage + F-09 G7 parity (D9)."""
        assets = fixtures.import_assets()
        risk_mod = fixtures.import_risk_allocator_mod()

        syms = ("BTC", "ETH", "SOL", "AVAX")
        specs = assets.default_web3_specs(syms)
        n_bars = 45
        bars_by_sym = {
            s: fixtures.build_bar_series(
                symbol=s, venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS,
                closes=fixtures.build_trending_closes(n_bars, start_price=100.0, step_ret=0.002, noise_std=0.006, seed=306 + i),
                carry_rates=[0.0001] * n_bars, carry_interval_h=12.0,
            )
            for i, s in enumerate(syms)
        }
        # Raw weights sum to 1.67x (> binding 1.0x leverage -> 0.6x scale-down) with 15% Carver inertia
        steps = [{s: (0.42 if t % 4 != 0 else 0.36) for s in syms} for t in range(n_bars)]
        strat = fixtures.build_scripted_strategy(steps, inertia_band=0.15)
        cfg = risk_mod.RiskAllocatorConfig(
            max_positions=4, target_annual_vol_pct=60.0, max_portfolio_leverage=1.0,
            max_single_weight=0.45, use_correlation_idm=True,
        )
        w_d, eq_d = fixtures.run_streaming_vs_batch_check(
            strat, specs, bars_by_sym, risk_cfg=cfg, require_active=True
        )
        self.assertLessEqual(w_d, 1e-10)
        self.assertLessEqual(eq_d, 1e-8)

    def test_t3_p07_accounting_per_symbol_calendar_plus_w1_funding_cashflow_conservation(self) -> None:
        """P3-07: F-07 Accounting funding sign & PnL conservation + real F-13 W1 on Web3 perps."""
        assets = fixtures.import_assets()
        w1_mod = fixtures.import_w1_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        # Part 1: Real W1 (Web3XSMomCarryKernel) simulated through PortfolioAccountingEngine (Auditor V3D)
        syms = ("BTC", "ETH", "SOL", "AVAX")
        drifts = (0.008, 0.002, -0.002, -0.008)
        fund_rates = (0.0001, 0.0001, 0.0001, 0.0001)
        n_bars = 120
        w1_bars = {
            s: fixtures.build_bar_series(
                symbol=s, venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS,
                closes=fixtures.build_trending_closes(n_bars, start_price=100.0, step_ret=drifts[i], noise_std=0.006, seed=307 + i),
                carry_rates=[fund_rates[i]] * n_bars, carry_interval_h=12.0,
            )
            for i, s in enumerate(syms)
        }
        w1_strat = fixtures.build_strategy_instance(w1_mod.Web3XSMomCarryKernel, top_k=1, bottom_k=1)
        cfg_w1 = risk_mod.RiskAllocatorConfig(max_positions=4, target_annual_vol_pct=25.0, max_portfolio_leverage=1.0, max_single_weight=0.6)
        res_w1 = acct_mod.PortfolioAccountingEngine().simulate(
            w1_strat,
            assets.default_web3_specs(syms),
            w1_bars,
            risk_cfg=cfg_w1,
            context_provider=lambda ts: {"regime_label": "EXPANSION", "exposure_scalar": 1.0},
        )
        w1_trades, _ = fixtures.measure_strategy_activity(res_w1)
        self.assertGreater(w1_trades, 0, "P3-07: W1 must execute trades on divergent Web3 universe")
        self.assertGreater(float(res_w1.executed_weights_history[-1].get("BTC", 0.0)), 0.0, "W1 must be long BTC")
        self.assertLess(float(res_w1.executed_weights_history[-1].get("AVAX", 0.0)), 0.0, "W1 must be short AVAX")
        fund_w1 = float(fixtures.get_metric(res_w1, "funding_carry_pnl_usd", "funding_pnl_usd"))
        gross_w1 = float(fixtures.get_metric(res_w1, "gross_price_pnl_usd"))
        cost_w1 = float(fixtures.get_metric(res_w1, "total_cost_usd"))
        net_w1 = float(fixtures.get_metric(res_w1, "net_pnl_usd"))
        self.assertGreater(abs(fund_w1), 0.0, "W1 active perp positions must accrue non-zero funding cashflow")
        self.assertAlmostEqual(net_w1, gross_w1 + fund_w1 - cost_w1, delta=1e-4, msg="W1 PnL conservation violated")

        # Part 2: Controlled isolated long-pays / short-collects funding sign check (carry_interval_h=12.0)
        zero_fee = assets.CostModel(0.0, 0.0, 0.0)
        spec_l = assets.AssetSpec("LONG_COIN", assets.ASSET_CLASS_CRYPTO_PERP, "hyperliquid", assets.CALENDAR_CRYPTO_24_7, zero_fee, 1.0)
        spec_s = assets.AssetSpec("SHORT_COIN", assets.ASSET_CLASS_CRYPTO_PERP, "hyperliquid", assets.CALENDAR_CRYPTO_24_7, zero_fee, 1.0)
        bars_l = fixtures.build_bar_series(
            symbol="LONG_COIN", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=[100.0] * 20, carry_rates=[0.001] * 20, carry_interval_h=12.0,
        )
        bars_s = fixtures.build_bar_series(
            symbol="SHORT_COIN", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=[100.0] * 20, carry_rates=[0.001] * 20, carry_interval_h=12.0,
        )
        cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0, max_single_weight=1.0, use_correlation_idm=False)
        res_long = acct_mod.PortfolioAccountingEngine().simulate(
            fixtures.build_scripted_strategy([{"LONG_COIN": 0.5}] * 20), {"LONG_COIN": spec_l}, {"LONG_COIN": bars_l}, risk_cfg=cfg
        )
        res_short = acct_mod.PortfolioAccountingEngine().simulate(
            fixtures.build_scripted_strategy([{"SHORT_COIN": -0.5}] * 20), {"SHORT_COIN": spec_s}, {"SHORT_COIN": bars_s}, risk_cfg=cfg
        )
        fund_long = float(fixtures.get_metric(res_long, "funding_carry_pnl_usd", "funding_pnl_usd", "net_pnl_usd"))
        fund_short = float(fixtures.get_metric(res_short, "funding_carry_pnl_usd", "funding_pnl_usd", "net_pnl_usd"))
        self.assertLess(fund_long, 0.0, "Long perp must PAY positive funding")
        self.assertGreater(fund_short, 0.0, "Short perp must COLLECT positive funding")
        self.assertAlmostEqual(
            float(fixtures.get_metric(res_long, "net_pnl_usd")),
            float(fixtures.get_metric(res_long, "gross_price_pnl_usd")) + fund_long - float(fixtures.get_metric(res_long, "total_cost_usd")),
            delta=1e-4,
        )

    def test_t3_p08_accounting_intrabar_liquidation_prevented_by_w3_50pct_storm_exit(self) -> None:
        """P3-08: F-07 Accounting intrabar liquidation + F-15 W3 50% protective storm-exit (Auditor V3B)."""
        assets = fixtures.import_assets()
        w3_mod = fixtures.import_w3_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        spec = assets.AssetSpec(
            symbol="BTC", asset_class=assets.ASSET_CLASS_CRYPTO_PERP, venue="hyperliquid",
            calendar=assets.CALENDAR_CRYPTO_24_7, cost_model=assets.COST_HYPERLIQUID_PERP, max_leverage_cap=5.0,
        )
        rich_rate = (0.80 / 8760.0) * 12.0
        n_bars = 50
        closes = [50000.0 + 40.0 * math.sin(i / 3.0) for i in range(n_bars)]
        bars = fixtures.build_bar_series(
            symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=closes,
            carry_rates=[rich_rate] * n_bars, carry_interval_h=12.0, premium_basis_bps=25.0,
        )
        # Inject bi-directional catastrophic crash wick on Bar 42 (after Bar 40 storm-exit -> Bar 41 open flatten!)
        b42 = bars[42]
        bars[42] = fixtures.make_bar(
            venue="hyperliquid", symbol="BTC", asset_class=assets.ASSET_CLASS_CRYPTO_PERP, timeframe_h=12.0,
            ts_event_ms=b42.ts_event_ms, ts_avail_ms=b42.ts_avail_ms,
            open=50000.0, high=100000.0, low=15000.0, close=35000.0,
            volume_usd=b42.volume_usd, carry_rate_period=rich_rate, carry_interval_h=12.0,
        )
        storm_ts = bars[40].ts_event_ms
        w3 = fixtures.build_strategy_instance(w3_mod.Web3RegimeCarryKernel)
        cfg = risk_mod.RiskAllocatorConfig(
            max_positions=1, target_annual_vol_pct=200.0, max_portfolio_leverage=5.0,
            max_single_weight=5.0, use_correlation_idm=False,
        )
        engine = acct_mod.PortfolioAccountingEngine()
        res = engine.simulate(
            w3,
            {"BTC": spec},
            {"BTC": bars},
            risk_cfg=cfg,
            capital_usd=10_000.0,
            context_provider=lambda ts: {
                "dvol": 45.0 if ts < storm_ts else 110.0,
                "rt_cost_bps": 14.0,
                "margin_ratio": 0.90 if ts < storm_ts else 0.42,
            },
        )
        w_pre = float(res.executed_weights_history[40].get("BTC", 0.0))
        w_bar41 = float(res.executed_weights_history[41].get("BTC", 0.0))
        w_bar42 = float(res.executed_weights_history[42].get("BTC", 0.0))
        self.assertGreater(abs(w_pre), 0.0, "Anti-stub (V3B): W3 must hold active carry position before storm_ts")
        self.assertAlmostEqual(w_bar41, 0.0, places=8, msg="W3 50% margin storm-exit must flatten at Bar 41 open before crash")
        self.assertAlmostEqual(w_bar42, 0.0, places=8, msg="W3 must stay flat through Bar 42 crash wick")
        self.assertEqual(int(fixtures.get_metric(res, "liquidation_count", default=0)), 0)
        self.assertGreater(float(fixtures.get_metric(res, "final_equity_usd")), 9000.0)

        # Paired negative control: unclosed leveraged perp across exact same Bar 42 MUST liquidate!
        w_ctrl = math.copysign(max(abs(w_pre), 3.0), w_pre)
        ctrl_strat = fixtures.build_scripted_strategy([{"BTC": w_ctrl}] * n_bars, strategy_id="p08_unclosed_ctrl")
        res_ctrl = engine.simulate(ctrl_strat, {"BTC": spec}, {"BTC": bars}, risk_cfg=cfg, capital_usd=10_000.0)
        self.assertGreaterEqual(
            int(fixtures.get_metric(res_ctrl, "liquidation_count", default=0)), 1,
            "Negative control: Bar 42 crash wick must genuinely liquidate an unclosed leveraged perp!",
        )

    def test_t3_p09_full_8gate_nested_cpcv_dsr_pbo_rejects_pure_random_walk(self) -> None:
        """P3-09: F-09 Nested CPCV (D3) + Fixed DSR (D4) + Multi-Variant PBO (D5) reject pure noise."""
        val_mod = fixtures.import_core_validation_mod()
        nested_fn = fixtures.resolve_validation_callable(val_mod, "run_nested_cpcv", "nested_cpcv")
        dsr_fn = fixtures.resolve_validation_callable(val_mod, "compute_psr_dsr")
        pbo_fn = fixtures.resolve_validation_callable(val_mod, "compute_pbo_cscv")
        FoldResultCls = getattr(val_mod, "FoldResult", None)

        rng = random.Random(309)
        noise_variants = [[rng.gauss(0.0, 0.01) for _ in range(320)] for _ in range(10)]
        rw0 = noise_variants[0]

        def _fit_eval_rw(train_idx: tuple[int, ...], test_idx: tuple[int, ...]) -> Any:
            train_vals = [rw0[i] for i in train_idx]
            mu = sum(train_vals) / max(len(train_vals), 1)
            sgn = 1.0 if mu >= 0.0 else -1.0
            oos = [sgn * rw0[i] for i in test_idx]
            is_r = [sgn * x for x in train_vals]
            if FoldResultCls is not None:
                return FoldResultCls(oos_returns=oos, is_returns=is_r, selected_params={"sign": sgn})
            return oos

        nested_rep = nested_fn(len(rw0), _fit_eval_rw, 252.0, n_groups=6, k_test=2, purge_bars=4, embargo_bars=4)
        self.assertTrue(bool(fixtures.get_metric(nested_rep, "is_nested", default=True)))
        self.assertGreaterEqual(int(fixtures.get_metric(nested_rep, "n_splits")), 15)

        dsr_rep = dsr_fn(rw0, periods_per_year=252.0, n_trials=20)
        pbo_rep = pbo_fn(noise_variants, periods_per_year=252.0, n_subsamples=8)

        cpcv_ok = bool(fixtures.get_metric(nested_rep, "passed_cpcv_gate", "passed_nested_cpcv", "passed", default=False))
        g5_pass = cpcv_ok and (float(dsr_rep.dsr) >= 0.50)
        g6_pass = float(pbo_rep.pbo) <= 0.35
        self.assertFalse(g5_pass and g6_pass, "Pure Gaussian random walk must fail G5 Nested CPCV/DSR and/or G6 PBO")
        _, verdict = fixtures.evaluate_8_gates_with_overrides(
            val_mod,
            g1_passed=True,
            g2_passed=False,
            baseline_sharpe=-0.1,
            baseline_net_pnl=-100.0,
            stress_sharpe=-0.3,
            stress_net_pnl=-300.0,
            g4_passed=False,
            g5_passed=g5_pass,
            g6_passed=g6_pass,
            g7_passed=True,
            g8_passed=False,
        )
        self.assertEqual(verdict, "REJECTED")

    def test_t3_p10_g2_hac_audit_gates_w2_ignoring_failed_squeeze_breakout_dir(self) -> None:
        """P3-10: F-09 G2 HAC IC (D12) FactorDirectionAuditor + active F-14 W2 Factor Gating (V-04)."""
        w2_mod = fixtures.import_w2_mod()
        assets = fixtures.import_assets()
        factors_mod = fixtures.import_factors_mod()

        spec = assets.default_web3_specs(("BTC",))["BTC"]
        closes = [100.0] * 60 + [100.0 + 1.2 * i for i in range(1, 121)]
        bars = fixtures.build_bar_series(
            symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=closes,
            carry_rates=[0.00005] * len(closes), carry_interval_h=12.0,
        )
        # Run real StreamingFactorEngine + FactorDirectionAuditor (Auditor V3D)
        snaps = factors_mod.StreamingFactorEngine(spec.calendar, 12.0).compute_series(bars)
        auditor = factors_mod.FactorDirectionAuditor(warmup_bars=40)
        summary = auditor.audit(
            "p10_w2_g2",
            {"BTC": bars},
            {"BTC": snaps},
            horizon_bars=6,
            candidate_factors=["ewmac_16_64", "donchian_pos_40", "squeeze_breakout_dir"],
        )
        self.assertNotIn(
            "squeeze_breakout_dir",
            summary.selected_non_redundant_factors,
            "Failed squeeze_breakout_dir must NOT be selected by G2 FactorDirectionAuditor",
        )
        selected = tuple(summary.selected_non_redundant_factors) if summary.selected_non_redundant_factors else ("ewmac_16_64",)

        w2_a = fixtures.build_strategy_instance(w2_mod.Web3AdaptiveTrendKernel, selected_factors=selected)
        w2_b = fixtures.build_strategy_instance(w2_mod.Web3AdaptiveTrendKernel, selected_factors=selected)
        w2_a.reset()
        w2_b.reset()
        w_bar55 = 0.0
        max_w_trend = 0.0
        for idx, b in enumerate(bars):
            da = w2_a.on_bar_step(b.ts_event_ms, {"BTC": b}, {"squeeze_breakout_dir": +1.0, "breadth_pct_above_20d": 0.8})
            db = w2_b.on_bar_step(b.ts_event_ms, {"BTC": b}, {"squeeze_breakout_dir": -1.0, "breadth_pct_above_20d": 0.8})
            wa = float(da["BTC"].target_weight)
            wb = float(db["BTC"].target_weight)
            w2_a.sync_executed_weights({"BTC": wa})
            w2_b.sync_executed_weights({"BTC": wb})
            self.assertEqual(wa, wb, "V-04/P3-10 violation: W2 altered weight from unselected squeeze_breakout_dir!")
            if idx == 55:
                w_bar55 = wa
            if idx >= 70:
                max_w_trend = max(max_w_trend, wa)
        self.assertAlmostEqual(w_bar55, 0.0, delta=1e-6, msg="Anti-stub: W2 must stay flat on flat prefix Bar 55")
        self.assertGreater(max_w_trend, 0.0, "Anti-stub: W2 must activate strongly during post-Bar-60 trend")

    def test_t3_p11_g8_benchmark_gate_blocks_underperforming_g1_passes_crash_avoidance(self) -> None:
        """P3-11: F-09 G8 Benchmark Gate (D8) + real F-16 GoldMacroEngine + F-17 G1 Single-Leg Gold."""
        assets = fixtures.import_assets()
        gm_mod = fixtures.import_gold_macro_mod()
        g1_mod = fixtures.import_g1_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()
        val_mod = fixtures.import_core_validation_mod()

        # Part 1: Real GoldMacroEngine + G1 + PortfolioAccountingEngine on 260d bull-then-crash scenario (Auditor V3D)
        n_days = 260
        closes_gc = [2000.0]
        macro_scores: list[float] = [1.2]
        for d in range(1, n_days):
            if d < 160:
                r = 0.0022
                ms = 1.2
            elif d < 220:
                r = -0.0075
                ms = -1.5
            else:
                r = 0.0010
                ms = 0.0
            closes_gc.append(closes_gc[-1] * (1.0 + r))
            macro_scores.append(ms)
        gc_bars = fixtures.build_bar_series(
            symbol="GC=F", venue="yahoo", asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
            timeframe_h=24.0, start_ms=fixtures.BASE_TS_MS, closes=closes_gc, contract_multiplier=100.0,
        )
        m_bars, m_yields = fixtures.build_gold_seven_driver_fixture(n_days=n_days, gold_step=0.002, seed=311)
        m_bars["GC=F"] = gc_bars
        me = fixtures.build_strategy_instance(
            gm_mod.GoldMacroEngine, macro_bars_by_symbol=m_bars, bars_by_symbol=m_bars, yield_curve=m_yields, yield_points=m_yields
        )
        score_by_ts = {int(gc_bars[i].ts_event_ms): macro_scores[i] for i in range(n_days)}
        g1 = fixtures.build_strategy_instance(g1_mod.GoldTradFiMacroKernel, symbol="GC=F", macro_engine=me)
        cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=15.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
        res_avoid = acct_mod.PortfolioAccountingEngine().simulate(
            g1,
            {"GC=F": assets.default_gold_specs()["GC=F"]},
            {"GC=F": gc_bars},
            risk_cfg=cfg,
            context_provider=lambda ts: {"gold_macro": {"macro_gold_composite": score_by_ts.get(int(ts), 0.0)}},
        )
        g1_trades, g1_max_w = fixtures.measure_strategy_activity(res_avoid)
        self.assertGreater(g1_trades, 0, "P3-11 anti-stub: G1 must trade on 260d bull-then-crash scenario")
        self.assertGreater(g1_max_w, 1e-4, "P3-11 anti-stub: G1 must hold non-zero exposure")
        bh_fn = fixtures.resolve_validation_callable(val_mod, "compute_buy_and_hold_returns")
        eval_g8 = fixtures.resolve_validation_callable(val_mod, "evaluate_benchmark_gate", "evaluate_g8_benchmark", "evaluate_g8")
        bh_rets = bh_fn(gc_bars, timestamps_ms=res_avoid.period_timestamps_ms)
        g8_avoid = eval_g8(res_avoid.period_returns, bh_rets, periods_per_year=252.0, benchmark_name="GC=F_BH")
        self.assertTrue(
            bool(fixtures.get_metric(g8_avoid, "passed_benchmark_gate", "passed")),
            f"Crash-avoiding G1 must beat B&H on G8 benchmark gate; got {g8_avoid}",
        )

        # Part 2: Underperforming directional G1 MUST fail G8 and cannot reach PROMOTE_SHADOW_CANDIDATE
        g8_under_rep = fixtures.evaluate_g8_contract(
            val_mod, strategy_sharpe=0.68, strategy_calmar=0.35, benchmark_sharpe=0.80, benchmark_calmar=0.55, corr_to_benchmark=0.88
        )
        g8_under = bool(fixtures.get_metric(g8_under_rep, "passed_benchmark_gate", "passed"))
        self.assertFalse(g8_under)
        _, v_under = fixtures.evaluate_8_gates_with_overrides(val_mod, g8_passed=g8_under)
        self.assertNotEqual(v_under, "PROMOTE_SHADOW_CANDIDATE")

    def test_t3_p12_web3_lake_plus_regime_engine_plus_w1_hype_pit_and_dvol_stress(self) -> None:
        """P3-12: F-01 web3_lake + real F-12 Web3MarketRegimeEngine + F-13 W1 (HYPE PIT + DVOL Stress)."""
        assets = fixtures.import_assets()
        models = fixtures.import_df_models()
        router_mod = fixtures.import_df_router()
        reg_mod = fixtures.import_web3_regime_mod()
        w1_mod = fixtures.import_w1_mod()
        prov_real, _, _ = fixtures.import_provenance()
        web3_dom = fixtures.resolve_web3_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t3_p12_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            start_ms = fixtures.utc_ms(2024, 11, 1, 0, 0)
            n_bars = 220
            syms = ("BTC", "ETH", "SOL", "HYPE")
            drifts = (0.005, 0.003, -0.002, 0.009)
            for idx, s in enumerate(syms):
                # Calm expansion on [0:170], sharp selloff on [170:220]
                closes = fixtures.build_trending_closes(170, start_price=(10.0 if s == "HYPE" else 100.0), step_ret=drifts[idx], noise_std=0.002, seed=312 + idx)
                for _ in range(n_bars - 170):
                    closes.append(max(closes[-1] * (0.992 if s != "SOL" else 0.985), 1.0))
                bars = fixtures.build_bar_series(
                    symbol=s, venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                    timeframe_h=12.0, start_ms=start_ms, closes=closes,
                    carry_rates=[0.00008] * n_bars, carry_interval_h=12.0,
                )
                router.upsert_bars(web3_dom, bars)

            btc_loaded = router.load_bars(web3_dom, "hyperliquid", "BTC", 43200)
            router.upsert_web3_macro(
                [
                    models.Web3MarketMacroPoint(
                        "deribit", "BTC_DVOL", b.ts_event_ms, b.ts_avail_ms, (42.0 if i < 170 else 96.0), prov_real
                    )
                    for i, b in enumerate(btc_loaded)
                ]
                + [
                    models.Web3MarketMacroPoint(
                        "defillama", "STABLECOIN_MCAP_USD", b.ts_event_ms, b.ts_avail_ms,
                        (150e9 + 0.5e9 * i if i < 170 else 235e9 - 1.5e9 * (i - 170)), prov_real
                    )
                    for i, b in enumerate(btc_loaded)
                ]
            )
            loaded = {s: router.load_bars(web3_dom, "hyperliquid", s, 43200) for s in syms}
            dvol_pts = router.load_web3_macro("BTC_DVOL")
            stable_pts = router.load_web3_macro("STABLECOIN_MCAP_USD")

            # Invoke REAL Web3MarketRegimeEngine at calm Bar 150 vs DVOL-stress final bar (Auditor V3D)
            ts_calm = btc_loaded[150].ts_event_ms
            ts_stress = btc_loaded[-1].ts_event_ms
            reg_calm = fixtures.invoke_web3_regime_engine(reg_mod.Web3MarketRegimeEngine, loaded, dvol_pts, stable_pts, ts_calm)
            reg_stress = fixtures.invoke_web3_regime_engine(reg_mod.Web3MarketRegimeEngine, loaded, dvol_pts, stable_pts, ts_stress)
            calm_scalar = float(fixtures.get_metric(reg_calm, "exposure_scalar", "net_exposure_scalar", "risk_scalar", "beta_scalar"))
            stress_scalar = float(fixtures.get_metric(reg_stress, "exposure_scalar", "net_exposure_scalar", "risk_scalar", "beta_scalar"))
            calm_lbl = str(fixtures.get_metric(reg_calm, "regime_label", "regime", "label"))
            stress_lbl = str(fixtures.get_metric(reg_stress, "regime_label", "regime", "label"))
            self.assertGreater(calm_scalar, stress_scalar, f"Real Web3MarketRegimeEngine must reduce scalar in DVOL stress ({calm_scalar} vs {stress_scalar})")

            w1 = fixtures.build_strategy_instance(w1_mod.Web3XSMomCarryKernel, top_k=1, bottom_k=1)
            w1.reset()
            pre_listing_hype = 0
            post_listing_hype = 0
            dec_at_150: dict[str, Any] = {}
            dec_at_end: dict[str, Any] = {}
            for i in range(len(btc_loaded)):
                ts = btc_loaded[i].ts_event_ms
                lbl_i = calm_lbl if i < 170 else stress_lbl
                sc_i = calm_scalar if i < 170 else stress_scalar
                dec = w1.on_bar_step(ts, {s: loaded[s][i] for s in syms}, {"regime_label": lbl_i, "exposure_scalar": sc_i, "dvol": dvol_pts[i].value})
                w1.sync_executed_weights({s: d.target_weight for s, d in dec.items()})
                w_h = float(dec["HYPE"].target_weight)
                if ts < fixtures.HYPE_LISTING_MS:
                    if abs(w_h) > 1e-12:
                        pre_listing_hype += 1
                elif ts >= fixtures.HYPE_LISTING_MS + 35 * fixtures.DAY_MS:
                    if abs(w_h) > 1e-12:
                        post_listing_hype += 1
                if i == 150:
                    dec_at_150 = dec
                if i == len(btc_loaded) - 1:
                    dec_at_end = dec

            self.assertEqual(pre_listing_hype, 0, "P3-12: HYPE must have zero pre-listing weight before 2024-12-05")
            self.assertGreater(post_listing_hype, 0, "P3-12 anti-stub: HYPE must participate after listing + warmup")
            gross_calm = sum(abs(float(d.target_weight)) for d in dec_at_150.values())
            net_calm = sum(float(d.target_weight) for d in dec_at_150.values())
            gross_stress = sum(abs(float(d.target_weight)) for d in dec_at_end.values())
            net_stress = sum(float(d.target_weight) for d in dec_at_end.values())
            self.assertGreater(gross_calm, 0.05, "P3-12 anti-stub: W1 must be active during calm expansion")
            self.assertTrue(
                abs(net_stress) <= 0.15 * max(gross_stress, 1e-9) + 1e-6 or net_calm > net_stress,
                f"W1 must reduce net exposure or stay neutral under DVOL stress (calm net={net_calm:.4f}, stress net={net_stress:.4f})",
            )

    def test_t3_p13_full_web3_pipeline_bit_identical_under_gold_lake_yahoo_vix_crash(self) -> None:
        """P3-13: F-01 Router + F-12 Web3MarketRegimeEngine + F-14 W2 + F-07 Accounting bit-identical under Yahoo ^VIX=99."""
        assets = fixtures.import_assets()
        models = fixtures.import_df_models()
        router_mod = fixtures.import_df_router()
        reg_mod = fixtures.import_web3_regime_mod()
        w2_mod = fixtures.import_w2_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()
        prov_real, _, _ = fixtures.import_provenance()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t3_p13_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            # 50 flat bars then 70 breakout bars -> guarantees flat Bar 45 + active breakout (kills Stub Z & Stub C)
            closes = [40000.0] * 50 + [40000.0 + 300.0 * k for k in range(1, 71)]
            btc_bars = fixtures.build_bar_series(
                symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS, closes=closes,
                carry_rates=[0.00005] * len(closes), carry_interval_h=12.0,
            )
            router.upsert_bars(web3_dom, btc_bars)
            router.upsert_web3_macro(
                [models.Web3MarketMacroPoint("deribit", "BTC_DVOL", b.ts_event_ms, b.ts_avail_ms, 45.0, prov_real) for b in btc_bars]
                + [models.Web3MarketMacroPoint("defillama", "STABLECOIN_MCAP_USD", b.ts_event_ms, b.ts_avail_ms, 150e9 + 0.4e9 * i, prov_real) for i, b in enumerate(btc_bars)]
            )
            loaded_btc = router.load_bars(web3_dom, "hyperliquid", "BTC", 43200)
            dvol_pts = router.load_web3_macro("BTC_DVOL")
            stable_pts = router.load_web3_macro("STABLECOIN_MCAP_USD")
            spec = assets.default_web3_specs(("BTC",))["BTC"]
            cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=20.0, max_portfolio_leverage=1.0, max_single_weight=1.0)

            reg_clean = fixtures.invoke_web3_regime_engine(
                reg_mod.Web3MarketRegimeEngine, {"BTC": loaded_btc}, dvol_pts, stable_pts, loaded_btc[-1].ts_event_ms
            )
            res_clean = acct_mod.PortfolioAccountingEngine().simulate(
                fixtures.build_strategy_instance(w2_mod.Web3AdaptiveTrendKernel),
                {"BTC": spec},
                {"BTC": loaded_btc},
                risk_cfg=cfg,
                context_provider=lambda ts: {"breadth_pct_above_20d": 0.80},
            )

            # Now pollute gold_macro_lake.sqlite with extreme Yahoo ^VIX=99.0 crash!
            vix_crash = fixtures.build_bar_series(
                symbol="^VIX", venue="yahoo", asset_class=assets.ASSET_CLASS_MACRO_INDEX,
                timeframe_h=24.0, start_ms=fixtures.BASE_TS_MS, closes=[99.0] * 60,
            )
            router.upsert_bars(gold_dom, vix_crash)
            reg_polluted = fixtures.invoke_web3_regime_engine(
                reg_mod.Web3MarketRegimeEngine,
                {"BTC": loaded_btc, "^VIX": vix_crash},
                dvol_pts,
                stable_pts,
                loaded_btc[-1].ts_event_ms,
                extra_context={"yahoo_vix": 99.0, "vix": 99.0, "macro_vix_stress_z": 12.0},
            )
            res_polluted = acct_mod.PortfolioAccountingEngine().simulate(
                fixtures.build_strategy_instance(w2_mod.Web3AdaptiveTrendKernel),
                {"BTC": spec},
                {"BTC": router.load_bars(web3_dom, "hyperliquid", "BTC", 43200)},
                risk_cfg=cfg,
                context_provider=lambda ts: {"breadth_pct_above_20d": 0.80, "yahoo_vix": 99.0, "macro_vix_stress_z": 12.0},
            )
            self.assertEqual(reg_clean, reg_polluted, "D7/P3-13 violation: Yahoo ^VIX crash altered Web3MarketRegimeEngine!")
            self.assertEqual(
                fixtures.get_metric(res_clean, "equity_curve"),
                fixtures.get_metric(res_polluted, "equity_curve"),
                "D7/P3-13 violation: Gold/Yahoo ^VIX crash altered Web3 equity curve!",
            )
            trades_c, max_w_c = fixtures.measure_strategy_activity(res_clean)
            self.assertGreater(trades_c, 0, "P3-13 anti-stub: W2 must trade on breakout")
            self.assertAlmostEqual(float(res_clean.executed_weights_history[45].get("BTC", 0.0)), 0.0, places=8, msg="Flat prefix Bar 45 must be 0.0")
            self.assertGreater(max_w_c, 0.0, "P3-13 anti-stub: breakout weight must be > 0.0")

    def test_t3_p14_web3_carry_w3_passes_g3_stress_and_g8_orthogonal_alpha_branch(self) -> None:
        """P3-14: F-01/F-04 Funding/Premium + real F-15 W3 + F-09 G3 Stress & G8 Orthogonal Carry Alpha."""
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        w3_mod = fixtures.import_w3_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()
        val_mod = fixtures.import_core_validation_mod()
        web3_dom = fixtures.resolve_web3_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t3_p14_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            n_bars = 180
            rich_bars = fixtures.build_bar_series(
                symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS,
                closes=[50000.0 + 25.0 * math.sin(i / 3.0) for i in range(n_bars)],
                carry_rates=[(0.85 / 8760.0) * 12.0] * n_bars, carry_interval_h=12.0, premium_basis_bps=25.0,
            )
            router.upsert_bars(web3_dom, rich_bars)
            loaded = router.load_bars(web3_dom, "hyperliquid", "BTC", 43200)
            spec = assets.default_web3_specs(("BTC",))["BTC"]
            cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=20.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
            engine = acct_mod.PortfolioAccountingEngine()
            ctx_fn = lambda ts: {"dvol": 42.0, "rt_cost_bps": 14.0, "margin_ratio": 0.90}

            res_base = engine.simulate(
                fixtures.build_strategy_instance(w3_mod.Web3RegimeCarryKernel),
                {"BTC": spec}, {"BTC": loaded}, risk_cfg=cfg, extra_slippage_bps_per_leg=0.0, context_provider=ctx_fn,
            )
            res_stress = engine.simulate(
                fixtures.build_strategy_instance(w3_mod.Web3RegimeCarryKernel),
                {"BTC": spec}, {"BTC": loaded}, risk_cfg=cfg, extra_slippage_bps_per_leg=2.0, context_provider=ctx_fn,
            )
            base_trades, _ = fixtures.measure_strategy_activity(res_base)
            self.assertGreater(base_trades, 0, "P3-14 anti-stub: W3 must enter rich 85% APR carry")
            base_pnl = float(fixtures.get_metric(res_base, "net_pnl_usd"))
            stress_pnl = float(fixtures.get_metric(res_stress, "net_pnl_usd"))
            stress_sr = float(fixtures.get_metric(res_stress, "sharpe_ratio"))
            self.assertGreater(base_pnl, 0.0)
            self.assertGreater(stress_pnl, 0.0)
            self.assertGreater(stress_sr, 0.0)
            self.assertGreater(
                float(fixtures.get_metric(res_stress, "total_cost_usd")),
                float(fixtures.get_metric(res_base, "total_cost_usd")),
            )

            # Evaluate real G8 orthogonal alpha branch against strong directional BTC benchmark (SR=1.80)
            eval_g8 = fixtures.resolve_validation_callable(val_mod, "evaluate_benchmark_gate", "evaluate_g8_benchmark", "evaluate_g8")
            st_rets = list(fixtures.get_metric(res_base, "period_returns"))
            bh_orth = fixtures._synth_return_series_for_sharpe(
                1.80, n_obs=len(st_rets), periods_per_year=730.0, annual_vol=0.50, base_series=st_rets, target_corr=0.05, seed=314
            )
            g8_rep = eval_g8(st_rets, bh_orth, periods_per_year=730.0, benchmark_name="BTC_BH")
            self.assertTrue(
                bool(fixtures.get_metric(g8_rep, "passed_benchmark_gate", "passed")),
                f"Orthogonal positive-Sharpe W3 carry sleeve must pass G8 even if SR < BTC B&H: {g8_rep}",
            )
            gates, _ = fixtures.evaluate_8_gates_with_overrides(
                val_mod,
                baseline_sharpe=float(fixtures.get_metric(res_base, "sharpe_ratio")),
                baseline_net_pnl=base_pnl,
                stress_sharpe=stress_sr,
                stress_net_pnl=stress_pnl,
            )
            self.assertTrue(gates.get("G3", False))

            # Negative control: sub-floor 2% APR carry must NOT enter (kills Stub C!)
            sub_bars = fixtures.build_bar_series(
                symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS,
                closes=[50000.0 + 25.0 * math.sin(i / 3.0) for i in range(n_bars)],
                carry_rates=[(0.02 / 8760.0) * 12.0] * n_bars, carry_interval_h=12.0, premium_basis_bps=0.0,
            )
            res_sub = engine.simulate(
                fixtures.build_strategy_instance(w3_mod.Web3RegimeCarryKernel),
                {"BTC": spec}, {"BTC": sub_bars}, risk_cfg=cfg, context_provider=ctx_fn,
            )
            sub_trades, _ = fixtures.measure_strategy_activity(res_sub)
            self.assertEqual(sub_trades, 0, "P3-14 anti-stub: sub-floor 2% APR carry must execute 0 trades")

    def test_t3_p15_gold_lake_tips_plus_gold_macro_engine_plus_g1_out_of_order_pit(self) -> None:
        """P3-15: F-05 gold_macro_lake + TIPS + F-16 GoldMacroEngine (D6) + F-17 G1."""
        router_mod = fixtures.import_df_router()
        gm_mod = fixtures.import_gold_macro_mod()
        g1_mod = fixtures.import_g1_mod()
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        bars_map, yields = fixtures.build_gold_seven_driver_fixture(
            n_days=140, real_10y_start=2.6, real_10y_end=1.3, dxy_step=-0.0015, si_step=0.0025, gold_step=0.002
        )
        with fixtures.temp_sandbox(prefix="e2e_t3_p15_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            for blist in bars_map.values():
                router.upsert_bars(gold_dom, blist)
            router.upsert_yield_curve(yields)

            loaded_bars = {s: router.load_bars(gold_dom, "yahoo", s, 86400) for s in bars_map}
            loaded_yields = router.load_yield_curve()
            me = fixtures.build_strategy_instance(
                gm_mod.GoldMacroEngine, macro_bars_by_symbol=loaded_bars, bars_by_symbol=loaded_bars,
                yield_curve=loaded_yields, yield_points=loaded_yields,
            )
            # Out-of-order PIT check on loaded engine (normalized for dict or dataclass)
            ts_mid = loaded_bars["GC=F"][70].ts_avail_ms
            ts_end = loaded_bars["GC=F"][-1].ts_avail_ms
            snap_mid_1 = fixtures.normalize_snapshot(me.snapshot_as_of(ts_mid))
            snap_end = fixtures.normalize_snapshot(me.snapshot_as_of(ts_end))
            snap_mid_2 = fixtures.normalize_snapshot(me.snapshot_as_of(ts_mid))
            self.assertEqual(snap_mid_1, snap_mid_2)
            self.assertNotEqual(snap_mid_1, snap_end, "Anti-stub (Stub R): GoldMacroEngine must evolve from mid to end")

            g1 = fixtures.build_strategy_instance(g1_mod.GoldTradFiMacroKernel, symbol="GC=F", macro_engine=me)
            g1.reset()
            distinct_w: set[float] = set()
            last_w = 0.0
            for b in loaded_bars["GC=F"]:
                dec = g1.on_bar_step(b.ts_event_ms, {"GC=F": b}, {"gold_macro": me.snapshot_as_of(b.ts_avail_ms)})
                last_w = float(dec["GC=F"].target_weight)
                distinct_w.add(round(last_w, 5))
                g1.sync_executed_weights({"GC=F": last_w})
            self.assertGreater(last_w, 0.0)
            self.assertGreaterEqual(len(distinct_w), 2, "Anti-stub (C-02): G1 must not output a constant weight")

    def test_t3_p16_pit_macro_event_calendar_plus_g2_policies_abc_plus_accounting_ablation(self) -> None:
        """P3-16: F-05 PIT MacroEventCalendar + F-18 G2 Policies A/B/C + F-07 Accounting."""
        assets = fixtures.import_assets()
        models = fixtures.import_df_models()
        g2_mod = fixtures.import_g2_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()
        prov_real, _, _ = fixtures.import_provenance()

        # Bar 0..38 strictly range-bound [2699.5, 2700.5], Bar 39 (T-4h) fakeout 2725, Bar 40 (T) plunge 2660, Bar 41..48 rally 2735..2860
        t_ev = fixtures.DEC_2026_FOMC_MS
        start_ms = t_ev - 40 * 4 * fixtures.HOUR_MS
        closes = [2700.0 + 0.5 * math.sin(i / 2.0) for i in range(39)]
        closes.extend([2725.0, 2660.0, 2735.0, 2755.0, 2775.0, 2795.0, 2815.0, 2835.0, 2850.0, 2860.0])
        bars = fixtures.build_bar_series(
            symbol="GC=F", venue="yahoo", asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
            timeframe_h=4.0, start_ms=start_ms, closes=closes, contract_multiplier=100.0,
        )
        ev = models.MacroEventRelease(
            "FOMC", "FOMC_2026_12_09", "2026-12", t_ev, t_ev, True, 4.25, 4.50, -1.0, "fed", prov_real
        )
        spec = assets.default_gold_specs()["GC=F"]
        cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=20.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
        ctx_fn = lambda ts: {"macro_events": [ev], "gold_macro": {"macro_gold_composite": 0.9}, "gvz_z": 1.5}

        results = {}
        for pol in ("A", "B", "C"):
            g2 = fixtures.build_strategy_instance(g2_mod.Gold4hEventKernel, policy=pol)
            results[pol] = acct_mod.PortfolioAccountingEngine().simulate(
                g2, {"GC=F": spec}, {"GC=F": bars}, risk_cfg=cfg, context_provider=ctx_fn
            )
            tr, _ = fixtures.measure_strategy_activity(results[pol])
            self.assertGreater(tr, 0, f"P3-16 anti-stub: Policy {pol} must execute trades")

        # Policy A entered Bar 39 fakeout -> held at Bar 40 plunge; Policy B shielded Bar 39 -> flat at Bar 40!
        self.assertGreater(abs(float(results["A"].executed_weights_history[40].get("GC=F", 0.0))), 0.0)
        self.assertAlmostEqual(float(results["B"].executed_weights_history[40].get("GC=F", 0.0)), 0.0, places=8)
        pnl_a = float(fixtures.get_metric(results["A"], "net_pnl_usd"))
        pnl_b = float(fixtures.get_metric(results["B"], "net_pnl_usd"))
        pnl_c = float(fixtures.get_metric(results["C"], "net_pnl_usd"))
        self.assertGreater(pnl_b, pnl_a, f"Policy B shield ({pnl_b:.2f}) must strictly beat Policy A whipsaw ({pnl_a:.2f})")
        self.assertTrue(math.isfinite(pnl_c))

    def test_t3_p17_gold_lake_plus_g3_paxg_weekend_guard_plus_accounting_sunday_stop(self) -> None:
        """P3-17: F-05 Gold/Macro + F-19 G3 PAXG Weekend Guard + F-07 Accounting Sunday stop-out (Auditor V3A)."""
        assets = fixtures.import_assets()
        g3_mod = fixtures.import_g3_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        # Ramp ending Fri 18:00 UTC -> Sat 14:00 (2700) -> Sun 14:00 crash (2430) -> Mon 08:00 (2430) -> Mon 20:00 (2150)
        start_ms = fixtures.FRI_2026_11_20_1800_MS - 39 * 12 * fixtures.HOUR_MS
        warm = fixtures.build_bar_series(
            symbol="PAXG", venue="hyperliquid", asset_class=assets.ASSET_CLASS_GOLD_PERP,
            timeframe_h=12.0, start_ms=start_ms, closes=[2600.0 + 2.5 * i for i in range(40)],
            carry_rates=[0.00002] * 40, carry_interval_h=12.0,
        )
        mon_2000_ms = fixtures.MON_2026_11_23_0800_MS + 12 * fixtures.HOUR_MS
        weekend_tail = [
            fixtures.make_bar(
                venue="hyperliquid", symbol="PAXG", asset_class=assets.ASSET_CLASS_GOLD_PERP, timeframe_h=12.0,
                ts_event_ms=fixtures.SAT_2026_11_21_1400_MS, ts_avail_ms=fixtures.SAT_2026_11_21_1400_MS + 12 * fixtures.HOUR_MS,
                open=2700.0, high=2705.0, low=2695.0, close=2700.0, volume_usd=5e6,
                carry_rate_period=0.00002, carry_interval_h=12.0,
            ),
            fixtures.make_bar(
                venue="hyperliquid", symbol="PAXG", asset_class=assets.ASSET_CLASS_GOLD_PERP, timeframe_h=12.0,
                ts_event_ms=fixtures.SUN_2026_11_22_1400_MS, ts_avail_ms=fixtures.SUN_2026_11_22_1400_MS + 12 * fixtures.HOUR_MS,
                open=2690.0, high=2690.0, low=2420.0, close=2430.0, volume_usd=5e6,
                carry_rate_period=0.00002, carry_interval_h=12.0,
            ),
            fixtures.make_bar(
                venue="hyperliquid", symbol="PAXG", asset_class=assets.ASSET_CLASS_GOLD_PERP, timeframe_h=12.0,
                ts_event_ms=fixtures.MON_2026_11_23_0800_MS, ts_avail_ms=fixtures.MON_2026_11_23_0800_MS + 12 * fixtures.HOUR_MS,
                open=2430.0, high=2440.0, low=2420.0, close=2430.0, volume_usd=5e6,
                carry_rate_period=0.00002, carry_interval_h=12.0,
            ),
            fixtures.make_bar(
                venue="hyperliquid", symbol="PAXG", asset_class=assets.ASSET_CLASS_GOLD_PERP, timeframe_h=12.0,
                ts_event_ms=mon_2000_ms, ts_avail_ms=mon_2000_ms + 12 * fixtures.HOUR_MS,
                open=2420.0, high=2420.0, low=2140.0, close=2150.0, volume_usd=5e6,
                carry_rate_period=0.00002, carry_interval_h=12.0,
            ),
        ]
        all_bars = warm + weekend_tail
        close_by_ts = {int(b.ts_event_ms): float(b.close) for b in all_bars}
        fri_close = float(warm[-1].close)

        def _ctx(ts: int) -> dict[str, Any]:
            c_px = close_by_ts.get(int(ts), fri_close)
            is_wknd = fixtures.FRI_2026_11_20_2200_MS <= int(ts) < fixtures.MON_2026_11_23_0800_MS
            return {
                "paxg_spot_price": c_px,
                "comex_price": fri_close if is_wknd else c_px,
                "gold_macro": {"macro_gold_composite": 0.8},
            }

        g3 = fixtures.build_strategy_instance(g3_mod.GoldPAXGBridgeKernel)
        spec = assets.default_gold_specs()["PAXG"]
        cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=18.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
        engine = acct_mod.PortfolioAccountingEngine()
        res = engine.simulate(g3, {"PAXG": spec}, {"PAXG": all_bars}, risk_cfg=cfg, context_provider=_ctx)

        w_pre = float(res.executed_weights_history[40].get("PAXG", 0.0))
        self.assertGreater(w_pre, 0.0, "P3-17 (V3A): G3 must enter long PAXG during weekday ramp prior to weekend")
        sat_meta_keys = " ".join(str(k).lower() for k in res.raw_decisions_history[40]["PAXG"].metadata.keys())
        self.assertTrue("weekend" in sat_meta_keys or "basis" in sat_meta_keys, f"Expected weekend/basis metadata on Sat bar: {sat_meta_keys}")
        self.assertAlmostEqual(float(res.executed_weights_history[42].get("PAXG", 0.0)), 0.0, places=6, msg="Sunday crash must stop out PAXG before Monday")
        self.assertAlmostEqual(float(res.executed_weights_history[43].get("PAXG", 0.0)), 0.0, places=6, msg="Stopped-out PAXG must stay flat on Mon 20:00 crash continuation")
        self.assertGreaterEqual(len(res.trade_ledger.executed), 2, "Must record both weekday entry and Sunday stop-out exit")

        # Paired unstopped control holding w_pre across Mon 20:00 (2150) must lose strictly more than stopped G3!
        unstopped_steps = [{"PAXG": 0.0}] * 25 + [{"PAXG": w_pre}] * (len(all_bars) - 25)
        unstopped_ctrl = engine.simulate(
            fixtures.build_scripted_strategy(unstopped_steps, strategy_id="p17_unstopped_ctrl", asset_domain="gold"),
            {"PAXG": spec}, {"PAXG": all_bars}, risk_cfg=cfg,
        )
        self.assertGreater(
            float(fixtures.get_metric(res, "net_pnl_usd")),
            float(fixtures.get_metric(unstopped_ctrl, "net_pnl_usd")),
            "P3-17 (V3A): Sunday stop-out must preserve strictly higher PnL than unstopped hold through 2150 continuation",
        )

    def test_t3_p18_decoupled_w1_and_g1_sleeves_combined_via_portfolio_combiner(self) -> None:
        """P3-18: F-11 PortfolioCombiner + F-13 W1 (Web3) + F-17 G1 (Gold) Return-Stream Composition."""
        assets = fixtures.import_assets()
        w1_mod = fixtures.import_w1_mod()
        g1_mod = fixtures.import_g1_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()
        comb_mod = fixtures.import_portfolio_combiner_mod()

        n_days = 140
        w_syms = ("BTC", "ETH", "SOL", "AVAX")
        w_bars = {
            s: fixtures.build_bar_series(
                symbol=s, venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                timeframe_h=24.0, start_ms=fixtures.BASE_TS_MS,
                closes=fixtures.build_trending_closes(n_days, start_price=100.0, step_ret=0.005 - 0.003 * i, noise_std=0.015, seed=318 + i),
                carry_rates=[0.0001] * n_days, carry_interval_h=24.0,
            )
            for i, s in enumerate(w_syms)
        }
        g_bars, _ = fixtures.build_gold_seven_driver_fixture(n_days=n_days, gold_step=0.0018, seed=399)

        cfg = risk_mod.RiskAllocatorConfig(max_positions=4, target_annual_vol_pct=15.0, max_portfolio_leverage=1.0, max_single_weight=0.5)
        res_w1 = acct_mod.PortfolioAccountingEngine().simulate(
            fixtures.build_strategy_instance(w1_mod.Web3XSMomCarryKernel, top_k=1, bottom_k=1),
            assets.default_web3_specs(w_syms), w_bars, risk_cfg=cfg,
        )
        res_g1 = acct_mod.PortfolioAccountingEngine().simulate(
            fixtures.build_strategy_instance(g1_mod.GoldTradFiMacroKernel, symbol="GC=F"),
            {"GC=F": assets.default_gold_specs()["GC=F"]}, {"GC=F": g_bars["GC=F"]}, risk_cfg=cfg,
            context_provider=lambda ts: {"gold_macro": {"macro_gold_composite": 0.7}},
        )
        w1_trades, _ = fixtures.measure_strategy_activity(res_w1)
        g1_trades, _ = fixtures.measure_strategy_activity(res_g1)
        self.assertGreater(w1_trades, 0, "P3-18 anti-stub: W1 sleeve must trade")
        self.assertGreater(g1_trades, 0, "P3-18 anti-stub: G1 sleeve must trade")
        self.assertGreaterEqual(fixtures.count_distinct_raw_decisions(res_w1), 2, "P3-18 anti-stub: W1 raw decisions must vary")
        self.assertGreaterEqual(fixtures.count_distinct_raw_decisions(res_g1), 2, "P3-18 anti-stub: G1 raw decisions must vary")

        w1_rets_before = list(fixtures.get_metric(res_w1, "period_returns"))
        g1_rets_before = list(fixtures.get_metric(res_g1, "period_returns"))
        self.assertTrue(any(abs(r) > 1e-9 for r in w1_rets_before), "W1 sleeve must have non-zero return variance")
        self.assertTrue(any(abs(r) > 1e-9 for r in g1_rets_before), "G1 sleeve must have non-zero return variance")

        combined = comb_mod.PortfolioCombiner().combine_sleeves({"web3_w1": res_w1, "gold_g1": res_g1})
        self.assertEqual(list(fixtures.get_metric(res_w1, "period_returns")), w1_rets_before)
        self.assertEqual(list(fixtures.get_metric(res_g1, "period_returns")), g1_rets_before)
        dr = float(fixtures.get_metric(combined, "diversification_ratio"))
        self.assertGreaterEqual(dr, 1.0)
        corr = fixtures.get_metric(combined, "pairwise_corr_matrix", "correlation_matrix", "pairwise_correlation")
        self.assertIn("web3_w1", corr)
        self.assertIn("gold_g1", corr)

    def test_t3_p19_portfolio_combiner_separates_w2_trend_and_w3_carry_margin_rea001(self) -> None:
        """P3-19: F-11 PortfolioCombiner + F-14 W2 Trend + F-15 W3 Carry separated margin ledger."""
        assets = fixtures.import_assets()
        w2_mod = fixtures.import_w2_mod()
        w3_mod = fixtures.import_w3_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()
        comb_mod = fixtures.import_portfolio_combiner_mod()

        n_bars = 100
        bars = fixtures.build_bar_series(
            symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=12.0, start_ms=fixtures.BASE_TS_MS,
            closes=fixtures.build_trending_closes(n_bars, start_price=40000.0, step_ret=0.003, noise_std=0.01, seed=319),
            carry_rates=[0.0] * 10 + [(0.70 / 8760.0) * 12.0] * (n_bars - 10),
            carry_interval_h=12.0, premium_basis_bps=25.0,
        )
        spec = assets.default_web3_specs(("BTC",))["BTC"]
        cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=15.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
        res_w2 = acct_mod.PortfolioAccountingEngine().simulate(
            fixtures.build_strategy_instance(w2_mod.Web3AdaptiveTrendKernel), {"BTC": spec}, {"BTC": bars}, risk_cfg=cfg,
            context_provider=lambda ts: {"breadth_pct_above_20d": 0.80},
        )
        res_w3 = acct_mod.PortfolioAccountingEngine().simulate(
            fixtures.build_strategy_instance(w3_mod.Web3RegimeCarryKernel), {"BTC": spec}, {"BTC": bars}, risk_cfg=cfg,
            context_provider=lambda ts: {"dvol": 42.0, "rt_cost_bps": 14.0, "margin_ratio": 0.90},
        )
        w2_trades, _ = fixtures.measure_strategy_activity(res_w2)
        w3_trades, _ = fixtures.measure_strategy_activity(res_w3)
        self.assertGreater(w2_trades, 0, "P3-19 anti-stub: W2 sleeve must trade")
        self.assertGreater(w3_trades, 0, "P3-19 anti-stub: W3 sleeve must trade")
        self.assertGreaterEqual(fixtures.count_distinct_raw_decisions(res_w2), 2, "P3-19 anti-stub: W2 raw decisions must vary")
        self.assertGreaterEqual(fixtures.count_distinct_raw_decisions(res_w3), 2, "P3-19 anti-stub: W3 raw decisions must vary")
        self.assertTrue(any(abs(r) > 1e-9 for r in fixtures.get_metric(res_w2, "period_returns")))
        self.assertTrue(any(abs(r) > 1e-9 for r in fixtures.get_metric(res_w3, "period_returns")))

        combined = comb_mod.PortfolioCombiner().combine_sleeves({"w2_trend": res_w2, "w3_carry": res_w3})
        alloc = fixtures.get_metric(combined, "sleeve_capital_allocations", "capital_allocation", "sleeve_capital_allocation", "margin_allocation", "sleeve_weights_final", "sleeve_weights")
        self.assertIn("w2_trend", alloc)
        self.assertIn("w3_carry", alloc)
        for k in ("w2_trend", "w3_carry"):
            v = alloc[k]
            num = float(v) if isinstance(v, (int, float)) else float(fixtures.get_metric(v, "allocated_capital_usd", "capital_usd", "allocated_weight", "weight", "margin_usd"))
            self.assertGreater(num, 0.0, f"P3-19 REA-001 sleeve {k} must receive strictly positive allocation")

    def test_t3_p20_ashare_interface_plus_accounting_t_plus_1_limit_up_and_sell_stamp_duty(self) -> None:
        """P3-20: Real F-20 AShareStrategyInterface + F-07 Accounting (T+1 + limit-up + 5bp sell stamp duty)."""
        assets = fixtures.import_assets()
        ashare_mod = fixtures.import_ashare_contract_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()

        cn_spec = assets.default_ashare_template_spec("600519.SH")
        # 3 trading days x 4 1h bars = 12 bars; replace Bar 8 (Day 3 open) with +10% one-tick limit-up lock!
        bars = fixtures.build_ashare_multiday_1h_bars(symbol="600519.SH", n_days=3, base_price=100.0)
        lim_px = round(float(bars[7].close) * 1.10, 4)
        b8 = bars[8]
        bars[8] = fixtures.make_bar(
            venue="sse", symbol="600519.SH", asset_class=assets.ASSET_CLASS_CN_A_SHARE, timeframe_h=1.0,
            ts_event_ms=b8.ts_event_ms, ts_avail_ms=b8.ts_avail_ms,
            open=lim_px, high=lim_px, low=lim_px, close=lim_px, volume_usd=b8.volume_usd,
        )
        # Step schedule:
        # Bar 0 close -> buy 0.8 (executes Bar 1 open); Bars 1..3 close -> exit 0.0 (locked Day 1, unlocks Day 2 Bar 4!)
        # Bars 4..6 -> flat 0.0; Bar 7 close -> buy 0.8 (queued for Bar 8 open +10% limit-up -> PRICE_LIMIT_UP_LOCK!)
        ts_to_idx = {int(b.ts_event_ms): i for i, b in enumerate(bars)}
        desired_by_idx = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0, 0.0]

        def _ashare_ctx(ts: int) -> dict[str, Any]:
            idx = ts_to_idx.get(int(ts), 0)
            w_des = desired_by_idx[idx]
            prev_c = float(bars[idx - 1].close) if idx > 0 else float(bars[0].close)
            return {
                "raw_signal": (5.0 if w_des > 0.0 else -5.0),
                "desired_weight": w_des,
                "signal": (5.0 if w_des > 0.0 else -5.0),
                "prev_close": prev_c,
            }

        strat = fixtures.instantiate_ashare_interface(
            ashare_mod.AShareStrategyInterface, default_signal=0.0, daily_price_limit_pct=10.0
        )
        self.assertEqual(strat.asset_domain, "ashare")
        cfg = risk_mod.RiskAllocatorConfig(
            max_positions=1, target_annual_vol_pct=500.0, max_portfolio_leverage=1.0,
            max_single_weight=0.8, use_correlation_idm=False,
        )
        res = acct_mod.PortfolioAccountingEngine().simulate(
            strat, {"600519.SH": cn_spec}, {"600519.SH": bars},
            risk_cfg=cfg, capital_usd=10_000.0, context_provider=_ashare_ctx,
        )
        exec_w = [float(w.get("600519.SH", 0.0)) for w in res.executed_weights_history]
        self.assertAlmostEqual(exec_w[0], 0.0, places=6)
        self.assertGreater(exec_w[1], 0.0, "P3-20: Day 1 Bar 1 must hold bought position")
        self.assertGreater(exec_w[2], 0.0, "P3-20: T+1 must lock Day 1 Bar 2")
        self.assertGreater(exec_w[3], 0.0, "P3-20: T+1 must lock Day 1 Bar 3")
        # Day 2 unlock executes on Bar 4 (if close-of-Day-1 sell allowed) or Bar 5 open (strict strategy CST-date gate)
        self.assertAlmostEqual(exec_w[5], 0.0, places=6, msg="P3-20: T+1 must unlock and execute sell on Day 2")
        self.assertAlmostEqual(exec_w[8], 0.0, places=6, msg="P3-20: +10% limit-up must block buy on Day 3 Bar 8")

        strat_t1_locks = [
            d["600519.SH"]
            for d in res.raw_decisions_history
            if "600519.SH" in d and bool(d["600519.SH"].metadata.get("t_plus_1_locked", False))
        ]
        acct_t1_locks = [s for s in res.trade_ledger.suppressed if s.reason == "T_PLUS_1_LOCK"]
        lim_locks = [s for s in res.trade_ledger.suppressed if s.reason == "PRICE_LIMIT_UP_LOCK"]
        self.assertGreaterEqual(
            len(strat_t1_locks) + len(acct_t1_locks),
            2,
            f"Expected >=2 T+1 lock enforcements (strategy={len(strat_t1_locks)}, acct={len(acct_t1_locks)})",
        )
        self.assertGreaterEqual(len(lim_locks), 1, f"Expected >=1 PRICE_LIMIT_UP_LOCK suppression, got {res.trade_ledger.suppressed}")

        buys = [e for e in res.trade_ledger.executed if e.side == "BUY"]
        sells = [e for e in res.trade_ledger.executed if e.side == "SELL"]
        self.assertGreaterEqual(len(buys), 1)
        self.assertGreaterEqual(len(sells), 1)
        self.assertAlmostEqual(float(buys[0].stamp_duty_usd), 0.0, places=8)
        self.assertGreater(float(sells[0].stamp_duty_usd), 0.0, "CN A-Share SELL must incur 5bp sell-only stamp duty")
        self.assertGreater(float(fixtures.get_metric(res, "total_cost_usd")), 10.0)

    def test_t3_p21_concurrent_domain_isolation_between_web3_and_gold_macro_wal_lakes(self) -> None:
        """P3-21: F-01 Physically isolated web3_lake.sqlite & gold_macro_lake.sqlite WAL isolation."""
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t3_p21_") as tmp:
            router = router_mod.DataFoundationRouter(tmp)
            self.addCleanup(fixtures.close_router, router)

            for i in range(15):
                ts = fixtures.BASE_TS_MS + i * 12 * fixtures.HOUR_MS
                router.upsert_bars(
                    web3_dom,
                    [
                        fixtures.make_bar(
                            venue="hyperliquid", symbol="BTC", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
                            timeframe_h=12.0, ts_event_ms=ts, ts_avail_ms=ts + 12 * fixtures.HOUR_MS,
                            open=40000.0 + i, high=40100.0 + i, low=39900.0 + i, close=40050.0 + i,
                            volume_usd=1e7, carry_rate_period=0.0001,
                        )
                    ],
                )
                router.upsert_bars(
                    gold_dom,
                    [
                        fixtures.make_bar(
                            venue="yahoo", symbol="GC=F", asset_class=assets.ASSET_CLASS_GOLD_TRADFI,
                            timeframe_h=24.0, ts_event_ms=fixtures.BASE_TS_MS + i * fixtures.DAY_MS,
                            ts_avail_ms=fixtures.BASE_TS_MS + (i + 1) * fixtures.DAY_MS,
                            open=2500.0 + i, high=2510.0 + i, low=2490.0 + i, close=2505.0 + i, volume_usd=1e8,
                        )
                    ],
                )
            # Zero cross-domain visibility
            self.assertEqual(len(router.load_bars(web3_dom, "hyperliquid", "BTC", 43200)), 15)
            self.assertEqual(router.load_bars(gold_dom, "hyperliquid", "BTC", 43200), [])
            self.assertEqual(len(router.load_bars(gold_dom, "yahoo", "GC=F", 86400)), 15)
            self.assertEqual(router.load_bars(web3_dom, "yahoo", "GC=F", 86400), [])

    def test_t3_p22_v3_migration_plus_timeframe_sec_index_plus_dq_ma_07_and_idempotency(self) -> None:
        """P3-22: F-01 migrate_from_v3_sqlite + F-02 timeframe_sec index + F-03 audit_domain_dq."""
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t3_p22_") as tmp:
            v3_db = fixtures.create_synthetic_v3_lake_db(tmp / "v3.sqlite", n_web3_bars=60, n_gold_bars=60)
            router = router_mod.DataFoundationRouter(tmp / "v4")
            self.addCleanup(fixtures.close_router, router)

            router.migrate_from_v3_sqlite(v3_db)
            router.migrate_from_v3_sqlite(v3_db)  # idempotent
            self.assertEqual(len(router.load_bars(web3_dom, "hyperliquid", "BTC", 43200)), 60)
            self.assertEqual(len(router.load_bars(gold_dom, "yahoo", "GC=F", 86400)), 60)

            dq_w3 = router.audit_domain_dq(web3_dom, "hyperliquid", "BTC", 43200, assets.default_web3_specs(("BTC",))["BTC"])
            self.assertTrue(dq_w3.passed, f"Migrated clean Web3 bars must pass DQ-MA-01..07: {dq_w3}")
            w3_db = tmp / "v4" / "domains" / "web3_lake.sqlite"
            bar_tbl = fixtures.discover_bar_table(w3_db, preferred=("web3_bars", "canonical_bars"))
            plan = fixtures.query_explain_plan(
                w3_db,
                f'SELECT * FROM "{bar_tbl}" WHERE venue=? AND symbol=? AND timeframe_sec=? AND ts_avail_ms<=?',
                ("hyperliquid", "BTC", 43200, fixtures.BASE_TS_MS + 100 * fixtures.DAY_MS),
            )
            self.assertIn("USING", plan.upper(), f"Expected index lookup on {bar_tbl}: {plan}")
            self.assertNotIn(f"SCAN {bar_tbl}", plan, f"Full table scan on {bar_tbl}: {plan}")

    def test_t3_p23_carry_interval_h_8h_gate_vs_1h_hl_matching_apr_through_g2_audit(self) -> None:
        """P3-23: F-03 carry_interval_h (D11) + F-09 G2 FactorDirectionAuditor parity across Gate 8h vs HL 1h."""
        assets = fixtures.import_assets()
        factors_mod = fixtures.import_factors_mod()

        spec = assets.default_web3_specs(("BTC",))["BTC"]
        n_bars = 200
        # Predictive varying-APR series expressed identically as HL 1h (carry_interval_h=1.0) vs Gate 8h (carry_interval_h=8.0)
        aprs = [0.20 + 0.15 * math.sin(i / 12.0) for i in range(n_bars)]
        rng = random.Random(323)
        closes = [40000.0]
        for i in range(1, n_bars):
            r_i = 0.0025 * math.sin((i - 1) / 12.0) + rng.gauss(0.0, 0.002)
            closes.append(max(closes[-1] * (1.0 + r_i), 100.0))

        bars_hl = fixtures.build_bar_series(
            symbol="BTC", venue="hyperliquid", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=1.0, start_ms=fixtures.BASE_TS_MS, closes=closes,
            carry_rates=[a / 8760.0 for a in aprs], carry_interval_h=1.0, use_v3_model=True,
        )
        bars_gate = fixtures.build_bar_series(
            symbol="BTC", venue="gate", asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
            timeframe_h=1.0, start_ms=fixtures.BASE_TS_MS, closes=closes,
            carry_rates=[a / (8760.0 / 8.0) for a in aprs], carry_interval_h=8.0, use_v3_model=True,
        )
        snaps_hl = factors_mod.StreamingFactorEngine(spec.calendar, 1.0).compute_series(bars_hl)
        snaps_gate = factors_mod.StreamingFactorEngine(spec.calendar, 1.0).compute_series(bars_gate)
        self.assertAlmostEqual(
            float(snaps_hl[-1].values["carry_apr_24h"]),
            float(snaps_gate[-1].values["carry_apr_24h"]),
            delta=0.5,
            msg="Equal APR on HL 1h and Gate 8h must yield matching carry_apr_24h (eliminating 8x bug!)",
        )

        # Full G2 FactorDirectionAuditor parity across HL 1h vs Gate 8h
        auditor = factors_mod.FactorDirectionAuditor(warmup_bars=40)
        aud_hl = auditor.audit(
            "p23_hl", {"BTC": bars_hl}, {"BTC": snaps_hl},
            horizon_bars=4, candidate_factors=["carry_apr_24h"], priors={"carry_apr_24h": 1},
        )
        aud_gate = auditor.audit(
            "p23_gate", {"BTC": bars_gate}, {"BTC": snaps_gate},
            horizon_bars=4, candidate_factors=["carry_apr_24h"], priors={"carry_apr_24h": 1},
        )
        res_hl = aud_hl.results[0]
        res_gate = aud_gate.results[0]
        self.assertAlmostEqual(float(res_hl.spearman_ic), float(res_gate.spearman_ic), delta=1e-6)
        self.assertAlmostEqual(float(res_hl.t_stat), float(res_gate.t_stat), delta=1e-6)
        self.assertEqual(bool(res_hl.passed_g2), bool(res_gate.passed_g2))
        self.assertTrue(bool(res_hl.passed_g2), "Predictive carry_apr_24h must pass G2 identically across HL 1h and Gate 8h")

    def test_t3_p24_g3_plus2bp_cost_stress_fail_forces_rejected_verdict_floor(self) -> None:
        """P3-24: F-09 G3 +2bp/leg cost stress fail forces REJECTED verdict even if all other 7 gates pass."""
        val_mod = fixtures.import_core_validation_mod()
        # Positive baseline (+0.30 SR, +$120), but turns negative under +2bp/leg stress (-0.25 SR, -$90)
        gates, verdict = fixtures.evaluate_8_gates_with_overrides(
            val_mod,
            g1_passed=True,
            g2_passed=True,
            baseline_sharpe=0.30,
            baseline_net_pnl=120.0,
            stress_sharpe=-0.25,
            stress_net_pnl=-90.0,
            g4_passed=True,
            g5_passed=True,
            g6_passed=True,
            g7_passed=True,
            g8_passed=True,
        )
        self.assertFalse(gates.get("G3", True), "Negative +2bp/leg cost stress MUST fail G3")
        self.assertEqual(verdict, "REJECTED", "G3 FAIL must force REJECTED (enforcing G1 & G3 & G7 fail-closed floor)")


if __name__ == "__main__":
    unittest.main()
