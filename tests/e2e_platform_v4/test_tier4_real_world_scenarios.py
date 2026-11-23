"""Tier 4 Real-World Application Scenarios E2E Tests (T4-01..T4-10).

Exercises the real local Platform v4 workspace (real multi-year data lake, full CLI runner
into isolated temp output, Track W & Track G real-data workflows, PortfolioCombiner, v3->v4
migration & v3 test regression, pure-stdlib research stack compliance, platform-wide STD-34
strict audit, @S-1123-SCHQ-09 DO_NOT_TOUCH & safety locks, and all 9 SCHQ deliverables).

Strictly read-only on production lakes and git trees (all mutable outputs go to temp_sandbox).
All Platform v4 imports are lazy (inside test methods). Zero skips, zero fake passes,
pure Python 3.12 stdlib, 100% STD-34 compliant.
"""

from __future__ import annotations

import ast
import bisect
import importlib
import math
import os
from pathlib import Path
import re
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


def _copy_real_v4_lake_to_sandbox(sandbox_dir: Path) -> Path:
    """Create a safe read-only snapshot copy of auto/data/platform_v4_lake inside sandbox_dir."""
    src_lake = fixtures.V4_LAKE_DIR
    if not src_lake.is_dir():
        raise AssertionError(f"Real Platform v4 data lake missing: {src_lake}")
    dst_lake = sandbox_dir / "platform_v4_lake_copy"
    (dst_lake / "domains").mkdir(parents=True, exist_ok=True)
    for rel in ("catalog.sqlite", "domains/web3_lake.sqlite", "domains/gold_macro_lake.sqlite"):
        src_db = src_lake / rel
        if not src_db.is_file():
            raise AssertionError(f"Missing real lake SQLite DB: {src_db}")
        fixtures.copy_sqlite_readonly_snapshot(src_db, dst_lake / rel)
    return dst_lake


class TestTier4RealWorldScenarios(unittest.TestCase):
    """10 Real-World Application Scenarios (T4-01..T4-10)."""

    def test_t4_01_real_multi_year_partitioned_data_lake_audit(self) -> None:
        """T4-01: Real multi-year partitioned data lake integrity, provenance, spans, TIPS/events & indexes."""
        lake_dir = fixtures.V4_LAKE_DIR
        self.assertTrue(lake_dir.is_dir(), f"Real Platform v4 data lake missing: {lake_dir}")

        catalog_db = lake_dir / "catalog.sqlite"
        web3_db = lake_dir / "domains" / "web3_lake.sqlite"
        gold_db = lake_dir / "domains" / "gold_macro_lake.sqlite"
        for db in (catalog_db, web3_db, gold_db):
            self.assertTrue(db.is_file(), f"Missing real lake DB: {db}")
            conn = fixtures.open_sqlite_ro(db)
            try:
                mode = str(conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower()
                self.assertEqual(mode, "wal", f"{db.name} must be in WAL mode")
            finally:
                conn.close()

        # Dynamically resolve partitioned bar table names (web3_bars / tradfi_bars / canonical_bars)
        w3_bar_tbl = fixtures.discover_bar_table(web3_db, preferred=("web3_bars", "canonical_bars"))
        gold_bar_tbl = fixtures.discover_bar_table(gold_db, preferred=("tradfi_bars", "canonical_bars"))

        # 100% PROVENANCE_REAL + multi-year spans in web3_lake.sqlite
        w3_conn = fixtures.open_sqlite_ro(web3_db)
        try:
            w3_total = w3_conn.execute(f'SELECT COUNT(*) FROM "{w3_bar_tbl}"').fetchone()[0]
            self.assertGreater(w3_total, 0, f"Empty {w3_bar_tbl} in web3_lake.sqlite")
            non_real_w3 = w3_conn.execute(
                f'SELECT COUNT(*) FROM "{w3_bar_tbl}" WHERE provenance != \'REAL_EXCHANGE_API\''
            ).fetchone()[0]
            self.assertEqual(non_real_w3, 0, f"Web3 lake contains {non_real_w3} non-REAL_EXCHANGE_API rows!")

            for aux_tbl in ("web3_derivatives_metrics", "web3_market_macro"):
                aux_cnt = w3_conn.execute(f'SELECT COUNT(*) FROM "{aux_tbl}"').fetchone()[0]
                self.assertGreater(aux_cnt, 0, f"Web3 auxiliary table {aux_tbl} is empty")
                aux_non_real = w3_conn.execute(
                    f'SELECT COUNT(*) FROM "{aux_tbl}" WHERE provenance != \'REAL_EXCHANGE_API\''
                ).fetchone()[0]
                self.assertEqual(aux_non_real, 0, f"{aux_tbl} contains {aux_non_real} non-REAL_EXCHANGE_API rows!")

            for major in ("BTC", "ETH", "SOL"):
                row = w3_conn.execute(
                    f'SELECT MIN(ts_event_ms), MAX(ts_event_ms), COUNT(*) FROM "{w3_bar_tbl}" WHERE symbol=?',
                    (major,),
                ).fetchone()
                self.assertIsNotNone(row[0], f"Missing {major} in web3_lake.sqlite.{w3_bar_tbl}")
                span_days = (int(row[1]) - int(row[0])) / float(fixtures.DAY_MS)
                self.assertGreaterEqual(span_days, 600.0, f"Web3 {major} span {span_days:.1f}d < 600d")

            w3_paxg_row = w3_conn.execute(
                f'SELECT MIN(ts_event_ms), MAX(ts_event_ms), COUNT(*) FROM "{w3_bar_tbl}" WHERE symbol=\'PAXG\''
            ).fetchone()
        finally:
            w3_conn.close()

        g_conn = fixtures.open_sqlite_ro(gold_db)
        try:
            g_total = g_conn.execute(f'SELECT COUNT(*) FROM "{gold_bar_tbl}"').fetchone()[0]
            self.assertGreater(g_total, 0, f"Empty {gold_bar_tbl} in gold_macro_lake.sqlite")
            non_real_g = g_conn.execute(
                f'SELECT COUNT(*) FROM "{gold_bar_tbl}" WHERE provenance != \'REAL_EXCHANGE_API\''
            ).fetchone()[0]
            self.assertEqual(non_real_g, 0, f"Gold lake contains {non_real_g} non-REAL_EXCHANGE_API rows!")

            gc_1d = g_conn.execute(
                f'SELECT MIN(ts_event_ms), MAX(ts_event_ms) FROM "{gold_bar_tbl}" WHERE symbol=\'GC=F\' AND timeframe_sec=86400'
            ).fetchone()
            self.assertIsNotNone(gc_1d[0], "Missing GC=F 1d (86400s) in gold_macro_lake.sqlite")
            self.assertGreaterEqual((int(gc_1d[1]) - int(gc_1d[0])) / float(fixtures.DAY_MS), 1800.0, "GC=F 1d span < 1800d")

            gc_4h = g_conn.execute(
                f'SELECT MIN(ts_event_ms), MAX(ts_event_ms) FROM "{gold_bar_tbl}" WHERE symbol=\'GC=F\' AND timeframe_sec=14400'
            ).fetchone()
            self.assertIsNotNone(gc_4h[0], "Missing GC=F 4h (14400s) in gold_macro_lake.sqlite")
            self.assertGreaterEqual((int(gc_4h[1]) - int(gc_4h[0])) / float(fixtures.DAY_MS), 700.0, "GC=F 4h span < 700d")

            g_paxg_row = g_conn.execute(
                f'SELECT MIN(ts_event_ms), MAX(ts_event_ms), COUNT(*) FROM "{gold_bar_tbl}" WHERE symbol=\'PAXG\''
            ).fetchone()

            tips_cnt = g_conn.execute(
                "SELECT COUNT(*) FROM macro_yield_curve WHERE real_10y IS NOT NULL"
            ).fetchone()[0]
            self.assertGreater(tips_cnt, 0, "US Treasury TIPS real_10y must be populated in gold_macro_lake.sqlite")
            tips_non_real = g_conn.execute(
                "SELECT COUNT(*) FROM macro_yield_curve WHERE provenance != 'REAL_EXCHANGE_API'"
            ).fetchone()[0]
            self.assertEqual(tips_non_real, 0, "macro_yield_curve contains non-REAL_EXCHANGE_API rows!")

            ev_rows = g_conn.execute(
                "SELECT event_type, ts_release_ms, is_sep_meeting FROM macro_events"
            ).fetchall()
            self.assertGreater(len(ev_rows), 0, "PIT macro_events table must be populated")
            ev_non_real = g_conn.execute(
                "SELECT COUNT(*) FROM macro_events WHERE provenance != 'REAL_EXCHANGE_API'"
            ).fetchone()[0]
            self.assertEqual(ev_non_real, 0, "macro_events contains non-REAL_EXCHANGE_API rows!")
            ev_types = {r[0] for r in ev_rows}
            self.assertTrue({"FOMC", "NFP", "CPI"}.issubset(ev_types))
            # December 2026 event week coverage (2026-12-04 NFP, 2026-12-09 FOMC SEP, 2026-12-10 CPI)
            dec_start = fixtures.utc_ms(2026, 12, 1, 0, 0)
            dec_end = fixtures.utc_ms(2026, 12, 15, 0, 0)
            dec_rows = [r for r in ev_rows if dec_start <= int(r[1]) <= dec_end]
            dec_types = {r[0] for r in dec_rows}
            self.assertTrue({"FOMC", "NFP", "CPI"}.issubset(dec_types), f"Missing Dec 2026 event week types: {dec_types}")
            self.assertTrue(
                any(r[0] == "FOMC" and int(r[2]) == 1 for r in dec_rows),
                "Dec 2026 FOMC must carry is_sep_meeting=1",
            )
        finally:
            g_conn.close()

        # Cross-domain PAXG resolution: gold_macro_lake.sqlite FIRST, fallback web3_lake.sqlite
        paxg_row = g_paxg_row if (g_paxg_row is not None and int(g_paxg_row[2]) > 0) else w3_paxg_row
        self.assertIsNotNone(paxg_row[0], "Missing PAXG in both gold_macro_lake.sqlite and web3_lake.sqlite")
        self.assertGreater(int(paxg_row[2]), 0, "Zero PAXG bars in domain lakes")
        paxg_days = (int(paxg_row[1]) - int(paxg_row[0])) / float(fixtures.DAY_MS)
        self.assertGreaterEqual(paxg_days, 550.0, f"PAXG span {paxg_days:.1f}d < 550d")

        # Exact timeframe_sec index query plan check on discovered bar tables (zero full scans)
        for db, bar_tbl, v, s, tf in (
            (web3_db, w3_bar_tbl, "hyperliquid", "BTC", 43200),
            (gold_db, gold_bar_tbl, "yahoo", "GC=F", 86400),
        ):
            plan = fixtures.query_explain_plan(
                db,
                f'SELECT * FROM "{bar_tbl}" WHERE venue=? AND symbol=? AND timeframe_sec=? AND ts_avail_ms<=?',
                (v, s, tf, fixtures.utc_ms(2026, 12, 1, 0, 0)),
            )
            self.assertIn("USING", plan.upper(), f"Missing index usage on {db.name}.{bar_tbl}: {plan}")
            self.assertTrue(
                "INDEX" in plan.upper() or "PRIMARY KEY" in plan.upper(),
                f"Expected index lookup on {db.name}.{bar_tbl}: {plan}",
            )
            self.assertNotIn(f"SCAN {bar_tbl}", plan, f"Full table scan detected on {db.name}.{bar_tbl}: {plan}")

        # Immutable exported partitions + SHA-256 catalog manifest byte-for-byte verification
        cat_conn = fixtures.open_sqlite_ro(catalog_db)
        try:
            manifest_rows = cat_conn.execute(
                "SELECT rel_path, sha256, provenance FROM dataset_manifests"
            ).fetchall()
        finally:
            cat_conn.close()
        self.assertGreaterEqual(len(manifest_rows), 1, "catalog.sqlite dataset_manifests is empty")
        for rel_path, expected_sha, prov in manifest_rows:
            self.assertEqual(str(prov), "REAL_EXCHANGE_API", f"Non-real manifest provenance for {rel_path}: {prov}")
            export_file = lake_dir / str(rel_path)
            self.assertTrue(export_file.is_file(), f"Manifest points to missing export file: {export_file}")
            self.assertEqual(
                fixtures.sha256_file(export_file),
                str(expected_sha),
                f"SHA-256 mismatch on exported partition {rel_path}",
            )

    def test_t4_02_full_cli_runner_end_to_end_and_report_schema_verification(self) -> None:
        """T4-02: Execute scripts/run_strategy_platform_v4.py into isolated temp dir & validate strict JSON report."""
        self.assertTrue(fixtures.V4_RUNNER_PATH.is_file(), f"Missing runner: {fixtures.V4_RUNNER_PATH}")
        help_proc = subprocess.run(
            [sys.executable, str(fixtures.V4_RUNNER_PATH), "--help"],
            cwd=str(fixtures.AUTO_REPO_ROOT), capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(help_proc.returncode, 0, f"Runner --help failed: {help_proc.stderr}")
        help_txt = help_proc.stdout

        expected_gates = frozenset({"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"})
        canonical_verdicts = frozenset({"PROMOTE_SHADOW_CANDIDATE", "CONDITIONAL_RESEARCH", "REJECTED"})

        def _extract_gate_blocks(obj: Any) -> list[dict[str, bool]]:
            found: list[dict[str, bool]] = []
            if isinstance(obj, dict):
                if expected_gates.issubset(obj.keys()):
                    block: dict[str, bool] = {}
                    valid = True
                    for gid in expected_gates:
                        val = obj[gid]
                        if isinstance(val, bool):
                            block[gid] = val
                        elif isinstance(val, dict) and isinstance(val.get("passed"), bool):
                            block[gid] = val["passed"]
                        else:
                            valid = False
                            break
                    if valid:
                        found.append(block)
                for v in obj.values():
                    found.extend(_extract_gate_blocks(v))
            elif isinstance(obj, list):
                if len(obj) >= 8 and all(isinstance(item, dict) for item in obj):
                    by_id = {
                        str(item.get("gate_id", item.get("id", ""))): item.get("passed")
                        for item in obj
                    }
                    if expected_gates.issubset(by_id.keys()) and all(
                        isinstance(by_id[g], bool) for g in expected_gates
                    ):
                        found.append({g: bool(by_id[g]) for g in expected_gates})
                for item in obj:
                    found.extend(_extract_gate_blocks(item))
            return found

        def _collect_field_values(obj: Any, field_name: str) -> list[Any]:
            vals: list[Any] = []
            if isinstance(obj, dict):
                if field_name in obj:
                    vals.append(obj[field_name])
                for v in obj.values():
                    vals.extend(_collect_field_values(v, field_name))
            elif isinstance(obj, list):
                for item in obj:
                    vals.extend(_collect_field_values(item, field_name))
            return vals

        with fixtures.temp_sandbox(prefix="e2e_t4_02_runner_") as tmp:
            lake_copy = _copy_real_v4_lake_to_sandbox(tmp)
            out_report = tmp / "platform_v4_report.json"

            cmd = [sys.executable, str(fixtures.V4_RUNNER_PATH)]
            for flag in ("--lake-dir", "--data-dir", "--db-dir"):
                if flag in help_txt:
                    cmd.extend([flag, str(lake_copy)])
                    break
            for flag in ("--report-out", "--output", "--out", "--report-path"):
                if flag in help_txt:
                    cmd.extend([flag, str(out_report)])
                    break
            for flag in ("--offline", "--no-fetch", "--skip-fetch"):
                if flag in help_txt:
                    cmd.append(flag)
                    break

            proc = subprocess.run(
                cmd,
                cwd=str(fixtures.AUTO_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"v4 runner failed (RC={proc.returncode}):\nSTDOUT:\n{proc.stdout[-1500:]}\nSTDERR:\n{proc.stderr[-1500:]}")
            self.assertTrue(out_report.is_file(), f"Runner did not write report to {out_report}")

            payload = fixtures.strict_json_loads(out_report.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, dict)
            raw_txt = fixtures.strict_json_dumps(payload)
            self.assertIn("REAL_EXCHANGE_API", raw_txt)
            prov_vals = [str(p).upper() for p in _collect_field_values(payload, "provenance")]
            for pv in prov_vals:
                self.assertNotIn("SYNTHETIC", pv, f"Synthetic provenance leaked into runner report: {pv}")
                self.assertNotIn("SIMULATED_FALLBACK", pv, f"Simulated fallback provenance leaked: {pv}")

            # Structured 8-gate block verification (immune to G1/G2/G3 Gold strategy name collisions!)
            gate_blocks = _extract_gate_blocks(payload)
            self.assertGreaterEqual(len(gate_blocks), 2, "Report must contain structured G1..G8 boolean blocks for >=2 evaluated tracks")

            verdicts = [str(v) for v in _collect_field_values(payload, "verdict") if isinstance(v, str)]
            self.assertGreaterEqual(len(verdicts), 2, f"Report missing structured verdicts for >=2 tracks: {verdicts}")
            for v in verdicts:
                self.assertIn(v, canonical_verdicts, f"Non-canonical Verdict string in runner report: {v!r}")

            for metric_key in ("cagr_pct", "sharpe_ratio", "max_drawdown_pct"):
                mvals = _collect_field_values(payload, metric_key)
                finite_nums = [
                    float(v) for v in mvals
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
                ]
                self.assertGreaterEqual(len(finite_nums), 1, f"Report missing finite numeric metric {metric_key}")

            # Confirm TrialLedger recorded trials in the isolated lake copy
            cat_conn = fixtures.open_sqlite_ro(lake_copy / "catalog.sqlite")
            try:
                t_cnt = cat_conn.execute("SELECT COUNT(*) FROM trial_ledger").fetchone()[0]
            finally:
                cat_conn.close()
            self.assertGreater(t_cnt, 0, "v4 runner must persist cumulative trials in catalog.sqlite trial_ledger")

    def test_t4_03_real_data_web3_track_w_end_to_end_w1_w2_w3_and_zero_vix_parity(self) -> None:
        """T4-03: Real-data Web3 Track W (Web3MarketRegimeEngine + W1/W2/W3), D7 zero-VIX proof, anti-stub, and G7 parity."""
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        reg_mod = fixtures.import_web3_regime_mod()
        w1_mod = fixtures.import_w1_mod()
        w2_mod = fixtures.import_w2_mod()
        w3_mod = fixtures.import_w3_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t4_03_web3_") as tmp:
            lake_copy = _copy_real_v4_lake_to_sandbox(tmp)
            router = router_mod.DataFoundationRouter(lake_copy)
            self.addCleanup(fixtures.close_router, router)

            dvol_pts = router.load_web3_macro("BTC_DVOL")
            stable_pts = router.load_web3_macro("STABLECOIN_MCAP_USD")
            btc_metrics = router.load_web3_metrics("binance_um", "BTC")
            vix_bars_real = router.load_bars(gold_dom, "yahoo", "^VIX", 86400)
            self.assertGreater(len(dvol_pts), 0, "Missing real BTC_DVOL in web3_lake.sqlite")
            self.assertGreater(len(stable_pts), 0, "Missing real STABLECOIN_MCAP_USD in web3_lake.sqlite")
            self.assertGreater(len(btc_metrics), 0, "Missing real BTC derivatives metrics in web3_lake.sqlite")

            syms = ("BTC", "ETH", "SOL", "AVAX")
            bars_by_sym = {}
            for s in syms:
                loaded = router.load_bars(web3_dom, "hyperliquid", s, 43200)
                if not loaded:
                    loaded = router.load_bars(web3_dom, "binance_um", s, 43200)
                self.assertGreater(len(loaded), 200, f"Insufficient real 12h Web3 bars for {s}")
                # Select real 200d high-funding bull window [500:900] when available so W1, W2, and W3 all trade genuinely
                bars_by_sym[s] = loaded[500:900] if len(loaded) >= 1000 else loaded[-400:]

            # Real Web3MarketRegimeEngine D7 invariance against real Yahoo ^VIX bars & VIX=95 shock
            ts_probe = bars_by_sym["BTC"][-1].ts_event_ms
            reg_clean = fixtures.invoke_web3_regime_engine(
                reg_mod.Web3MarketRegimeEngine, bars_by_sym, dvol_pts, stable_pts, ts_probe
            )
            bars_plus_vix = dict(bars_by_sym)
            if vix_bars_real:
                bars_plus_vix["^VIX"] = vix_bars_real
            reg_vix = fixtures.invoke_web3_regime_engine(
                reg_mod.Web3MarketRegimeEngine,
                bars_plus_vix,
                dvol_pts,
                stable_pts,
                ts_probe,
                extra_context={"yahoo_vix": 95.0, "vix": 95.0, "macro_vix_stress_z": 10.0},
            )
            self.assertEqual(reg_clean, reg_vix, "T4-03 D7 violation: Yahoo ^VIX altered real-data Web3MarketRegimeEngine!")

            specs = assets.default_web3_specs(syms)
            cfg = risk_mod.RiskAllocatorConfig(max_positions=4, target_annual_vol_pct=18.0, max_portfolio_leverage=1.0, max_single_weight=0.5)
            engine = acct_mod.PortfolioAccountingEngine()

            for name, strat in (
                ("W1", fixtures.build_strategy_instance(w1_mod.Web3XSMomCarryKernel)),
                ("W2", fixtures.build_strategy_instance(w2_mod.Web3AdaptiveTrendKernel)),
                ("W3", fixtures.build_strategy_instance(w3_mod.Web3RegimeCarryKernel)),
            ):
                res_clean = engine.simulate(strat.clone_fresh(), specs, bars_by_sym, risk_cfg=cfg)
                res_vix = engine.simulate(
                    strat.clone_fresh(), specs, bars_by_sym, risk_cfg=cfg,
                    context_provider=lambda ts: {"yahoo_vix": 95.0, "macro_vix_stress_z": 10.0},
                )
                self.assertTrue(math.isfinite(float(fixtures.get_metric(res_clean, "cagr_pct"))))
                self.assertIsNotNone(fixtures.get_metric(res_clean, "trade_ledger"))

                # Anti-stub non-trivial activity & weight-variation guard (C-02 #8)
                exec_trades, max_abs_w = fixtures.measure_strategy_activity(res_clean)
                self.assertGreater(exec_trades, 0, f"T4-03 anti-stub: {name} executed 0 trades on real Web3 data")
                self.assertGreater(max_abs_w, 1e-4, f"T4-03 anti-stub: {name} produced zero executed weights")
                self.assertGreaterEqual(
                    fixtures.count_distinct_raw_decisions(res_clean), 2,
                    f"T4-03 anti-stub: {name} produced constant raw decisions",
                )

                self.assertEqual(
                    fixtures.get_metric(res_clean, "equity_curve"),
                    fixtures.get_metric(res_vix, "equity_curve"),
                    f"T4-03 D7 violation: Yahoo ^VIX altered real-data {name} equity curve!",
                )
                w_d, eq_d = fixtures.run_streaming_vs_batch_check(
                    strat.clone_fresh(), specs, bars_by_sym, risk_cfg=cfg, require_active=True
                )
                self.assertLessEqual(w_d, 1e-10, f"{name} G7 weight parity exceeded 1e-10")
                self.assertLessEqual(eq_d, 1e-8, f"{name} G7 equity parity exceeded 1e-8")
                if name == "W3":
                    self.assertEqual(int(fixtures.get_metric(res_clean, "liquidation_count", default=0)), 0)

    def test_t4_04_real_data_gold_track_g_end_to_end_g1_g2_abc_and_g3_paxg(self) -> None:
        """T4-04: Real-data Gold Track G (G1 GC=F & GLD + G8, G2 4h A/B/C, G3 PAXG in gold_macro + G7 parity)."""
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        gm_mod = fixtures.import_gold_macro_mod()
        g1_mod = fixtures.import_g1_mod()
        g2_mod = fixtures.import_g2_mod()
        g3_mod = fixtures.import_g3_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()
        val_mod = fixtures.import_core_validation_mod()
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t4_04_gold_") as tmp:
            lake_copy = _copy_real_v4_lake_to_sandbox(tmp)
            router = router_mod.DataFoundationRouter(lake_copy)
            self.addCleanup(fixtures.close_router, router)

            macro_syms = ("GC=F", "GLD", "SI=F", "DX-Y.NYB", "IEF", "TIP", "GDX", "^GVZ", "^VIX")
            macro_bars = {s: router.load_bars(gold_dom, "yahoo", s, 86400) for s in macro_syms}
            self.assertGreater(len(macro_bars["GC=F"]), 500, "Need multi-year real GC=F 1d bars")
            self.assertGreater(len(macro_bars["GLD"]), 500, "Need multi-year real GLD 1d bars")
            yields = router.load_yield_curve()
            events = router.load_macro_events()
            self.assertGreater(len(yields), 0)
            self.assertGreater(len(events), 0)

            me = fixtures.build_strategy_instance(
                gm_mod.GoldMacroEngine, macro_bars_by_symbol=macro_bars, bars_by_symbol=macro_bars,
                yield_curve=yields, yield_points=yields,
            )
            gold_specs = assets.default_gold_specs()
            cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=15.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
            engine = acct_mod.PortfolioAccountingEngine()
            eval_g8 = fixtures.resolve_validation_callable(val_mod, "evaluate_benchmark_gate", "evaluate_g8_benchmark", "evaluate_g8")
            bh_fn = fixtures.resolve_validation_callable(val_mod, "compute_buy_and_hold_returns")

            # G1 single-instrument GC=F and separately GLD + real G8 passive benchmark evaluation + G7 parity
            for sym in ("GC=F", "GLD"):
                g1 = fixtures.build_strategy_instance(g1_mod.GoldTradFiMacroKernel, symbol=sym, macro_engine=me)
                sl = macro_bars[sym][-500:]
                ctx_g1 = lambda ts: {"gold_macro": me.snapshot_as_of(ts)}
                res = engine.simulate(g1, {sym: gold_specs[sym]}, {sym: sl}, risk_cfg=cfg, context_provider=ctx_g1)
                self.assertTrue(math.isfinite(float(fixtures.get_metric(res, "cagr_pct"))))

                tr_g1, max_w_g1 = fixtures.measure_strategy_activity(res)
                self.assertGreater(tr_g1, 0, f"T4-04 anti-stub: G1({sym}) executed 0 trades")
                self.assertGreater(max_w_g1, 1e-4, f"T4-04 anti-stub: G1({sym}) produced zero executed weights")
                self.assertGreaterEqual(fixtures.count_distinct_raw_decisions(res), 2, f"T4-04 anti-stub: G1({sym}) raw decisions constant")

                bh_rets = bh_fn(sl, timestamps_ms=res.period_timestamps_ms)
                g8_rep = eval_g8(res.period_returns, bh_rets, periods_per_year=252.0, benchmark_name=f"{sym}_BH")
                self.assertGreaterEqual(int(fixtures.get_metric(g8_rep, "n_aligned_obs")), 200)
                for mk in ("strategy_sharpe", "benchmark_sharpe", "strategy_calmar", "benchmark_calmar"):
                    self.assertTrue(math.isfinite(float(fixtures.get_metric(g8_rep, mk))))
                self.assertIsInstance(fixtures.get_metric(g8_rep, "passed_benchmark_gate", "passed"), bool)
                self.assertIn(
                    str(fixtures.get_metric(g8_rep, "pass_pathway")),
                    {"DIRECTIONAL_OUTPERFORMANCE", "ORTHOGONAL_ALPHA", "FAILED"},
                )

                w_d, eq_d = fixtures.run_streaming_vs_batch_check(
                    g1.clone_fresh(), {sym: gold_specs[sym]}, {sym: sl}, risk_cfg=cfg,
                    context_provider=ctx_g1, require_active=True,
                )
                self.assertLessEqual(w_d, 1e-10)
                self.assertLessEqual(eq_d, 1e-8)

            # G2 4h Policies A/B/C on real 4h GC=F (800 bars covering ~16 real macro events)
            gc_4h = router.load_bars(gold_dom, "yahoo", "GC=F", 14400)
            self.assertGreater(len(gc_4h), 300, "Need real 4h GC=F bars")
            sl_4h = gc_4h[-800:] if len(gc_4h) >= 800 else gc_4h[-400:]
            res_by_pol: dict[str, Any] = {}
            meta_dump_by_pol: dict[str, str] = {}
            for pol in ("A", "B", "C"):
                g2 = fixtures.build_strategy_instance(g2_mod.Gold4hEventKernel, policy=pol)
                res_p = engine.simulate(
                    g2, {"GC=F": gold_specs["GC=F"]}, {"GC=F": sl_4h}, risk_cfg=cfg,
                    context_provider=lambda ts: {"macro_events": events, "gold_macro": me.snapshot_as_of(ts)},
                )
                res_by_pol[pol] = res_p
                self.assertTrue(math.isfinite(float(fixtures.get_metric(res_p, "sharpe_ratio"))))
                self.assertTrue(math.isfinite(float(fixtures.get_metric(res_p, "cagr_pct"))))
                tr_p, max_w_p = fixtures.measure_strategy_activity(res_p)
                self.assertGreater(tr_p, 0, f"T4-04 anti-stub: G2 Policy {pol} executed 0 trades")
                self.assertGreater(max_w_p, 1e-4, f"T4-04 anti-stub: G2 Policy {pol} zero executed weights")
                # Strict JSON round-trip on all BarDecision metadata
                meta_list = [
                    d["GC=F"].metadata for d in res_p.raw_decisions_history if "GC=F" in d
                ]
                encoded_meta = fixtures.strict_json_dumps(meta_list)
                self.assertIsInstance(fixtures.strict_json_loads(encoded_meta), list)
                meta_dump_by_pol[pol] = encoded_meta

            # Policy B vs Policy A must exhibit non-degenerate distinction across real macro events
            ab_distinct = (
                res_by_pol["A"].allocated_weights_history != res_by_pol["B"].allocated_weights_history
                or res_by_pol["A"].equity_curve != res_by_pol["B"].equity_curve
                or meta_dump_by_pol["A"] != meta_dump_by_pol["B"]
            )
            self.assertTrue(ab_distinct, "T4-04: G2 Policy A and Policy B produced identical weights, equity, and metadata!")

            # G3 real PAXG bars (gold_macro domain FIRST, fallback web3) + causal bisect_right spot/COMEX context + G7 parity
            paxg_bars = fixtures.resolve_paxg_bars(
                router, router_mod=router_mod, timeframe_sec=43200, venues=("hyperliquid", "binance_um"), min_bars=200
            )
            self.assertGreater(len(paxg_bars), 200, "Need real PAXG 12h bars in gold_macro or web3 domain")
            paxg_spot_bars = router.load_bars(gold_dom, "binance_spot", "PAXG", 43200)
            spot_ts = [int(b.ts_avail_ms) for b in paxg_spot_bars]
            spot_px = [float(b.close) for b in paxg_spot_bars]
            gc_ts = [int(b.ts_avail_ms) for b in macro_bars["GC=F"]]
            gc_px = [float(b.close) for b in macro_bars["GC=F"]]

            def _causal_lookup(ts_arr: list[int], px_arr: list[float], ts: int, fallback: float) -> float:
                pos = bisect.bisect_right(ts_arr, int(ts)) - 1
                return px_arr[pos] if pos >= 0 else fallback

            paxg_slice = paxg_bars[-300:]
            paxg_close_map = {int(b.ts_event_ms): float(b.close) for b in paxg_slice}

            def _g3_ctx(ts: int) -> dict[str, Any]:
                fb = paxg_close_map.get(int(ts), 2650.0)
                return {
                    "gold_macro": me.snapshot_as_of(ts),
                    "paxg_spot_price": _causal_lookup(spot_ts, spot_px, ts, fb),
                    "comex_price": _causal_lookup(gc_ts, gc_px, ts, fb),
                }

            g3 = fixtures.build_strategy_instance(g3_mod.GoldPAXGBridgeKernel)
            res_g3 = engine.simulate(
                g3.clone_fresh(), {"PAXG": gold_specs["PAXG"]}, {"PAXG": paxg_slice},
                risk_cfg=cfg, context_provider=_g3_ctx,
            )
            tr_g3, max_w_g3 = fixtures.measure_strategy_activity(res_g3)
            self.assertGreater(tr_g3, 0, "T4-04 anti-stub: G3 executed 0 trades on real PAXG")
            self.assertGreater(max_w_g3, 1e-4, "T4-04 anti-stub: G3 zero executed weights")
            self.assertGreaterEqual(fixtures.count_distinct_raw_decisions(res_g3), 2, "T4-04 anti-stub: G3 raw decisions constant")
            w_d3, eq_d3 = fixtures.run_streaming_vs_batch_check(
                g3, {"PAXG": gold_specs["PAXG"]}, {"PAXG": paxg_slice}, risk_cfg=cfg,
                context_provider=_g3_ctx, require_active=True,
            )
            self.assertLessEqual(w_d3, 1e-10)
            self.assertLessEqual(eq_d3, 1e-8)

    def test_t4_05_decoupled_return_stream_multi_asset_portfolio_composition(self) -> None:
        """T4-05: Combine real-data Web3 & Gold SimulationResult sleeves via PortfolioCombiner."""
        assets = fixtures.import_assets()
        router_mod = fixtures.import_df_router()
        gm_mod = fixtures.import_gold_macro_mod()
        w2_mod = fixtures.import_w2_mod()
        g1_mod = fixtures.import_g1_mod()
        risk_mod = fixtures.import_risk_allocator_mod()
        acct_mod = fixtures.import_accounting_mod()
        comb_mod = fixtures.import_portfolio_combiner_mod()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        with fixtures.temp_sandbox(prefix="e2e_t4_05_comb_") as tmp:
            lake_copy = _copy_real_v4_lake_to_sandbox(tmp)
            router = router_mod.DataFoundationRouter(lake_copy)
            self.addCleanup(fixtures.close_router, router)

            btc_bars = router.load_bars(web3_dom, "hyperliquid", "BTC", 43200)[-400:]
            macro_syms = ("GC=F", "GLD", "SI=F", "DX-Y.NYB", "IEF", "TIP", "GDX", "^GVZ", "^VIX")
            macro_bars = {s: router.load_bars(gold_dom, "yahoo", s, 86400) for s in macro_syms}
            yields = router.load_yield_curve()
            gc_bars = macro_bars["GC=F"][-400:]
            self.assertGreater(len(btc_bars), 100)
            self.assertGreater(len(gc_bars), 100)

            me = fixtures.build_strategy_instance(
                gm_mod.GoldMacroEngine, macro_bars_by_symbol=macro_bars, bars_by_symbol=macro_bars,
                yield_curve=yields, yield_points=yields,
            )
            cfg = risk_mod.RiskAllocatorConfig(max_positions=1, target_annual_vol_pct=15.0, max_portfolio_leverage=1.0, max_single_weight=1.0)
            engine = acct_mod.PortfolioAccountingEngine()
            res_w = engine.simulate(
                fixtures.build_strategy_instance(w2_mod.Web3AdaptiveTrendKernel),
                assets.default_web3_specs(("BTC",)), {"BTC": btc_bars}, risk_cfg=cfg,
            )
            res_g = engine.simulate(
                fixtures.build_strategy_instance(g1_mod.GoldTradFiMacroKernel, symbol="GC=F", macro_engine=me),
                {"GC=F": assets.default_gold_specs()["GC=F"]}, {"GC=F": gc_bars}, risk_cfg=cfg,
                context_provider=lambda ts: {"gold_macro": me.snapshot_as_of(ts)},
            )
            w_trades, w_max = fixtures.measure_strategy_activity(res_w)
            g_trades, g_max = fixtures.measure_strategy_activity(res_g)
            self.assertGreater(w_trades, 0, "T4-05 anti-stub: Web3 sleeve executed 0 trades")
            self.assertGreater(w_max, 1e-4, "T4-05 anti-stub: Web3 sleeve produced zero weights")
            self.assertGreaterEqual(fixtures.count_distinct_raw_decisions(res_w), 2, "T4-05 anti-stub: Web3 sleeve raw decisions constant")
            self.assertGreater(g_trades, 0, "T4-05 anti-stub: Gold sleeve executed 0 trades")
            self.assertGreater(g_max, 1e-4, "T4-05 anti-stub: Gold sleeve produced zero weights")
            self.assertGreaterEqual(fixtures.count_distinct_raw_decisions(res_g), 2, "T4-05 anti-stub: Gold sleeve raw decisions constant")

            before_w = list(fixtures.get_metric(res_w, "period_returns"))
            before_g = list(fixtures.get_metric(res_g, "period_returns"))
            self.assertTrue(any(abs(r) > 1e-9 for r in before_w), "T4-05 anti-stub: Web3 sleeve has zero return variance")
            self.assertTrue(any(abs(r) > 1e-9 for r in before_g), "T4-05 anti-stub: Gold sleeve has zero return variance")

            combined = comb_mod.PortfolioCombiner().combine_sleeves({"web3": res_w, "gold": res_g})
            self.assertEqual(list(fixtures.get_metric(res_w, "period_returns")), before_w)
            self.assertEqual(list(fixtures.get_metric(res_g, "period_returns")), before_g)

            corr = fixtures.get_metric(combined, "pairwise_corr_matrix", "correlation_matrix", "pairwise_correlation")
            self.assertAlmostEqual(float(corr["web3"]["web3"]), 1.0, places=6)
            self.assertAlmostEqual(float(corr["gold"]["gold"]), 1.0, places=6)
            self.assertAlmostEqual(float(corr["web3"]["gold"]), float(corr["gold"]["web3"]), places=6)
            self.assertGreaterEqual(float(fixtures.get_metric(combined, "diversification_ratio")), 1.0)
            self.assertTrue(math.isfinite(float(fixtures.get_metric(combined, "cagr_pct"))))

            alloc = fixtures.get_metric(
                combined, "sleeve_capital_allocations", "capital_allocation", "sleeve_weights_final", "sleeve_weights"
            )
            for k in ("web3", "gold"):
                self.assertIn(k, alloc)
                val = alloc[k]
                num = float(val) if isinstance(val, (int, float)) else float(fixtures.get_metric(val, "allocated_capital_usd", "capital_usd", "allocated_weight", "weight", "margin_usd"))
                self.assertGreater(num, 0.0, f"T4-05 REA-001 sleeve {k} must have positive allocation")

    def test_t4_06_backwards_compatibility_v3_migration_and_v3_unit_suite(self) -> None:
        """T4-06: Idempotent real v3->v4 SQLite migration (read-only snapshot) + 100% pass on test_platform_v3_*.py."""
        router_mod = fixtures.import_df_router()
        web3_dom = fixtures.resolve_web3_domain(router_mod)
        gold_dom = fixtures.resolve_gold_domain(router_mod)

        # 1) Migrate read-only snapshot of real platform_v3_lake.sqlite twice (idempotency)
        self.assertTrue(fixtures.V3_LAKE_PATH.is_file(), f"Expected v3 lake at {fixtures.V3_LAKE_PATH}")
        with fixtures.temp_sandbox(prefix="e2e_t4_06_mig_") as tmp:
            v3_snap = fixtures.copy_sqlite_readonly_snapshot(fixtures.V3_LAKE_PATH, tmp / "v3_ro_copy.sqlite")
            router = router_mod.DataFoundationRouter(tmp / "v4_migrated")
            self.addCleanup(fixtures.close_router, router)

            router.migrate_from_v3_sqlite(v3_snap)
            btc_1 = len(router.load_bars(web3_dom, "hyperliquid", "BTC", 3600))
            gc_1 = len(router.load_bars(gold_dom, "yahoo", "GC=F", 86400))
            self.assertGreater(btc_1, 1000)
            self.assertGreater(gc_1, 500)

            router.migrate_from_v3_sqlite(v3_snap)
            self.assertEqual(len(router.load_bars(web3_dom, "hyperliquid", "BTC", 3600)), btc_1)
            self.assertEqual(len(router.load_bars(gold_dom, "yahoo", "GC=F", 86400)), gc_1)

        # 2) Run existing v3 unit test suite with PYTHONPATH=AUTO_REPO_ROOT (NO -t flag!)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(fixtures.AUTO_REPO_ROOT)
        proc = subprocess.run(
            [
                sys.executable, "-m", "unittest", "discover",
                "-s", str(fixtures.AUTO_REPO_ROOT / "tests"),
                "-p", "test_platform_v3_*.py",
                "-v",
            ],
            cwd=str(fixtures.AUTO_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"v3 unit test suite regressed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("OK", proc.stderr)

    def test_t4_07_research_dependency_stack_and_thin_plane_stdlib_compliance(self) -> None:
        """T4-07: requirements-research.txt zero GPL/LGPL, pure-stdlib runtime check, scrubbed 768MB claims."""
        req = fixtures.REQUIREMENTS_RESEARCH_PATH
        self.assertTrue(req.is_file(), f"Missing {req}")
        lines = [ln.strip() for ln in req.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
        self.assertGreaterEqual(len(lines), 4)
        for ln in lines:
            low = ln.split("#", 1)[0].strip().lower()
            if not low:
                continue
            for banned in ("nautilus", "pysystemtrade", "spectre", "ta-lib", "talib", "psycopg2", "mysqlclient", "pyodbc", "gpl"):
                self.assertNotIn(banned, low, f"Forbidden copyleft dependency {banned!r} in {ln!r}")
            self.assertRegex(low, r"(==|>=|~=)", f"Unpinned package in requirements-research.txt: {ln!r}")

        # Pure Python 3.12 stdlib execution with third-party imports blocked
        probe = (
            "import sys\n"
            "BLOCKED = {'numpy','scipy','pyarrow','duckdb','polars','pandas','pytest'}\n"
            "class B:\n"
            "  def find_spec(self, name, path, target=None):\n"
            "    if name.split('.')[0] in BLOCKED:\n"
            "      raise ImportError(name)\n"
            "    return None\n"
            "sys.meta_path.insert(0, B())\n"
            "import alphaloop.platform\n"
            "from alphaloop.platform.strategies.web3 import regime_engine, xs_mom_carry, adaptive_trend, regime_carry\n"
            "from alphaloop.platform.strategies.gold import macro_engine, tradfi_macro_core, session_event_4h, paxg_bridge\n"
            "from alphaloop.platform.strategies import ashare_contract\n"
            "print('THIN_PLANE_STDLIB_PASS')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(fixtures.AUTO_REPO_ROOT), capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(proc.returncode, 0, f"Thin-plane pure stdlib import failed:\n{proc.stderr}")
        self.assertIn("THIN_PLANE_STDLIB_PASS", proc.stdout)

        # Scrubbed stale 768MB constraint claims across platform code
        stale_re = re.compile(r"(?i)(?:768\s*mb\s+cloud\s+server|fit\w*\s+768\s*mb|single\s+768\s*mb)")
        for path in sorted((fixtures.AUTO_REPO_ROOT / "alphaloop" / "platform").rglob("*.py")):
            txt = path.read_text(encoding="utf-8")
            self.assertIsNone(stale_re.search(txt), f"Stale 768MB constraint wording found in {path}")

    def test_t4_08_platform_wide_std34_strict_service_readiness_audit(self) -> None:
        """T4-08: STD-34 audit_service_readiness.py --strict EXIT 0 on platform, runner, and E2E suite."""
        targets = (
            (fixtures.AUTO_REPO_ROOT / "alphaloop" / "platform", False, 15),
            (fixtures.V4_RUNNER_PATH, False, 1),
            (fixtures.E2E_SUITE_DIR, True, 10),
        )
        for target_path, inc_tests, min_files in targets:
            self.assertTrue(target_path.exists(), f"Missing audit target: {target_path}")
            rc, payload, stderr = fixtures.run_audit_script(target_path, strict=True, include_tests=inc_tests)
            self.assertEqual(rc, 0, f"STD-34 audit failed on {target_path}: {payload} {stderr}")
            self.assertGreaterEqual(
                int(payload.get("files_scanned", 0)),
                min_files,
                f"STD-34 scanned too few files on {target_path}: {payload}",
            )
            self.assertEqual(int(payload.get("high_severity_count", 0)), 0)
            self.assertEqual(int(payload.get("medium_severity_count", 0)), 0)
            self.assertEqual(payload.get("findings"), [], f"Unexpected findings on {target_path}: {payload.get('findings')}")

    def test_t4_09_parallel_session_do_not_touch_safety_locks_and_clean_git_boundaries(self) -> None:
        """T4-09: Safety locks False, zero @S-1123-SCHQ-09 coupling, zero .agents/teamwork in active repos."""
        bridge = importlib.import_module("alphaloop.execution.bridge")
        adapter = importlib.import_module("alphaloop.execution.arb_executor_adapter")
        daemon = importlib.import_module("alphaloop.daemon.shadow_runner")

        self.assertIs(bridge.LIVE_EXECUTION_ENABLED, False)
        self.assertIs(adapter.ADAPTER_LIVE_EXECUTION_ENABLED, False)
        self.assertIs(daemon.DAEMON_LIVE_EXECUTION_ENABLED, False)
        self.assertEqual(bridge.ALLOWED_SHADOW_STATUSES, frozenset({"SHADOW_LIVE", "PAPER_LIVE"}))
        for mod in (bridge, adapter, daemon):
            self.assertIs(getattr(mod, "DRY_RUN_ONLY", True), True)

        # Zero AST imports from @S-1123-SCHQ-09 exclusive modules inside platform or v4 runner
        forbidden = (
            "alphaloop.execution",
            "alphaloop.daemon",
            "funding_carry_plugin",
            "trend_cta_plugin",
            "run_shadow_daemon",
            "research_carry_regime_switch",
        )
        scan_files = list((fixtures.AUTO_REPO_ROOT / "alphaloop" / "platform").rglob("*.py"))
        if fixtures.V4_RUNNER_PATH.is_file():
            scan_files.append(fixtures.V4_RUNNER_PATH)
        for path in scan_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        for bad in forbidden:
                            self.assertNotIn(bad, a.name, f"DO_NOT_TOUCH import {a.name} in {path}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for bad in forbidden:
                        self.assertNotIn(bad, node.module, f"DO_NOT_TOUCH import {node.module} in {path}")
                    for a in node.names:
                        full = f"{node.module}.{a.name}"
                        for bad in forbidden:
                            self.assertNotIn(bad, full, f"DO_NOT_TOUCH import {full} in {path}")

        # Zero .agents/teamwork leakage across our 2 target repos + 5 active sibling repos (excluding frozen cypo_trade_help)
        for repo in fixtures.ACTIVE_WORKSPACE_REPOS:
            if not repo.is_dir():
                continue
            self.assertFalse((repo / ".agents" / "teamwork").exists(), f".agents/teamwork exists on disk in {repo}")
            ls_proc = subprocess.run(
                ["git", "-C", str(repo), "ls-files", ".agents/teamwork"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            self.assertEqual(ls_proc.stdout.strip(), "", f"Tracked .agents/teamwork in {repo}")
            st_proc = subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain", ".agents/teamwork"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            self.assertEqual(st_proc.stdout.strip(), "", f"Untracked .agents/teamwork in {repo}")

        # Zero tracked sqlite/parquet lake binaries under data/platform_v4_lake
        lake_ls = subprocess.run(
            ["git", "-C", str(fixtures.AUTO_REPO_ROOT), "ls-files", "data/platform_v4_lake"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        self.assertEqual(lake_ls.stdout.strip(), "", f"Tracked binaries in data/platform_v4_lake: {lake_ls.stdout}")

    def test_t4_10_schq_governance_roadmap_and_empirical_audit_deliverables(self) -> None:
        """T4-10: Verify all 9 canonical Unicode SCHQ F-23 deliverables & semantic content markers."""
        for key, path in fixtures.SCHQ_DELIVERABLES.items():
            self.assertTrue(path.is_file(), f"Missing SCHQ deliverable {key!r}: {path}")
            self.assertGreater(path.stat().st_size, 100, f"Empty SCHQ deliverable {key!r}: {path}")

        # 1. Ticket CC-20261123-004 (first 40 lines patrol_3check compatibility; CC-20261123-003 claimed by @S-1123-SCHQ-10)
        ticket_txt = fixtures.SCHQ_DELIVERABLES["ticket"].read_text(encoding="utf-8")
        head40 = ticket_txt.splitlines()[:40]
        status_re = re.compile(
            r"^(?:[>|\s*-]|\d+\.)*\*{0,2}STATUS\*{0,2}\s*[|:：]\*{0,2}\s*[`*_✅🟢🔴🟡⚪⚫⏸\s]*([A-Za-z_]+|[\u4e00-\u9fff]+)",
            re.IGNORECASE,
        )
        claim_re = re.compile(r"@S-(\d{4})-([A-Za-z]+)-([A-Za-z0-9_]+)")
        status_val = None
        has_claim = False
        for ln in head40:
            m_st = status_re.match(ln)
            if m_st:
                status_val = m_st.group(1).upper()
            if "REVIEWER" not in ln.upper() and claim_re.search(ln):
                has_claim = True
        self.assertIn(status_val, {"SUBMITTED", "CLOSED", "DONE"}, f"Ticket STATUS must be SUBMITTED/terminal, got {status_val!r}")
        self.assertTrue(has_claim, "Ticket missing valid @S-MMDD-REPO-NN claim ID in first 40 lines")
        for req_f in ("PROJECT_ID", "EXIT_CRITERIA", "DEPENDENCIES", "DO_NOT_TOUCH", "DELIVERABLES", "EVIDENCE_LINK", "CC-20261123-001"):
            self.assertIn(req_f, ticket_txt)

        # 2. Architecture & Multi-Server Roadmap
        arch_txt = fixtures.SCHQ_DELIVERABLES["architecture"].read_text(encoding="utf-8")
        self.assertGreaterEqual(len(arch_txt), 1500)
        self.assertIn("CC-20261123-004", arch_txt)
        self.assertTrue("STD-32" in arch_txt or "复用核查" in arch_txt)
        self.assertTrue("STD-34" in arch_txt or "服务架构六问" in arch_txt)
        self.assertTrue("768M" in arch_txt or "768MB" in arch_txt)
        self.assertTrue(any(tok in arch_txt for tok in ("Node #6", "6号", "新开", "Node 6", "elastic")))
        for tok in ("catalog.sqlite", "web3_lake.sqlite", "gold_macro_lake.sqlite", "PortfolioCombiner"):
            self.assertIn(tok, arch_txt, f"Architecture doc missing {tok}")
        for tok in ("W1", "W2", "W3", "G1", "G2", "G3", "2027"):
            self.assertIn(tok, arch_txt, f"Architecture doc missing {tok}")

        # 3. Empirical Audit & A/B Report (V-01..V-20 + D1..D16 + v3 Gold revocation + SELF_REPORTED)
        emp_txt = fixtures.SCHQ_DELIVERABLES["empirical_report"].read_text(encoding="utf-8")
        self.assertGreaterEqual(len(emp_txt), 1500)
        self.assertIn("SELF_REPORTED", emp_txt)
        v_ids = set(re.findall(r"(?<![A-Za-z0-9])V-(?:0[1-9]|1[0-9]|20)(?![A-Za-z0-9])", emp_txt))
        d_ids = {
            f"D{int(num)}"
            for num in re.findall(r"(?<![A-Za-z0-9])D(0?[1-9]|1[0-6])(?![A-Za-z0-9])", emp_txt)
        }
        self.assertEqual(len(v_ids), 20, f"Empirical report must disposition all 20 V-01..V-20 opinions; found {sorted(v_ids)}")
        self.assertEqual(len(d_ids), 16, f"Empirical report must disposition all 16 D1..D16 defects; found {sorted(d_ids)}")
        self.assertIn("PROMOTE_SHADOW_CANDIDATE", emp_txt)
        self.assertTrue(any(tok in emp_txt.lower() for tok in ("revok", "撤销", "撤回", "downgrad")), "Must document honest v3 Gold revocation")
        for g_id in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"):
            self.assertIn(g_id, emp_txt)

        # 4. OWN_REPOS_REGISTRY registers kenansun/auto
        own_txt = fixtures.SCHQ_DELIVERABLES["own_repos"].read_text(encoding="utf-8")
        self.assertTrue(
            any(ln.lstrip().startswith("|") and "kenansun/auto" in ln for ln in own_txt.splitlines()),
            "OWN_REPOS_REGISTRY must register kenansun/auto in a markdown table row",
        )

        # 5. SHARED_SERVICES_CATALOG adds v4 collectors & preserves Tombstones
        cat_txt = fixtures.SCHQ_DELIVERABLES["shared_services"].read_text(encoding="utf-8")
        self.assertTrue(any(tok in cat_txt for tok in ("data_foundation", "collectors_web3", "collectors_gold_macro", "platform_v4")))
        for tomb in ("Gold S1", "H-OF3", "S7", "S8", "S10"):
            self.assertIn(tomb, cat_txt, f"SHARED_SERVICES_CATALOG must preserve tombstone {tomb}")

        # 6. OPEN_SOURCE_RESEARCH_LEDGER append-only rows 23..27 preserved + 768MB update appended
        oss_txt = fixtures.SCHQ_DELIVERABLES["oss_ledger"].read_text(encoding="utf-8")
        for orig in ("nautilus_trader", "RD-Agent", "QuantConnect/Lean", "pysystemtrade", "duckdb/duckdb"):
            self.assertIn(orig, oss_txt, f"OPEN_SOURCE_RESEARCH_LEDGER broke append-only row {orig}")
        self.assertIn("CC-20261123-004", oss_txt)

        # 7. CHANGE_NOTIFICATION_LEDGER preserves N-134..N-137 (N-137 claimed by @S-1123-SCHQ-09) + adds N-138+ for CC-20261123-004
        chg_txt = fixtures.SCHQ_DELIVERABLES["change_ledger"].read_text(encoding="utf-8")
        for prev_n in ("N-134", "N-135", "N-136", "N-137"):
            self.assertIn(prev_n, chg_txt)
        n_rows = [
            ln for ln in chg_txt.splitlines()
            if re.match(r"^\|\s*\*{0,2}`?N-(?:13[89]|1[4-9]\d|[2-9]\d{2})`?\*{0,2}\s*\|", ln)
            and "CC-20261123-004" in ln
        ]
        self.assertGreaterEqual(len(n_rows), 1, "CHANGE_NOTIFICATION_LEDGER missing N-138+ row for CC-20261123-004")
        self.assertIn("✅", n_rows[0])
        self.assertTrue("N-135" in n_rows[0] or "N-135" in emp_txt, "Must note N-135 resolved A5 item 3")

        # 8. Kanban CC-20261113-003 preserves CC-20261123-002 + appends CC-20261123-004 quote line
        kb_txt = fixtures.SCHQ_DELIVERABLES["kanban"].read_text(encoding="utf-8")
        self.assertIn("CC-20261123-002", kb_txt)
        kb_lines = [ln for ln in kb_txt.splitlines() if ln.lstrip().startswith(">") and "CC-20261123-004" in ln]
        self.assertGreaterEqual(len(kb_lines), 1, "Kanban CC-20261113-003 missing appended > [...] delivery line for CC-20261123-004")

        # 9. MAINTENANCE.md retains mandatory sections
        maint_txt = fixtures.SCHQ_DELIVERABLES["maintenance"].read_text(encoding="utf-8")
        for sec in ("定位与状态", "写入规则", "已知债务与待办", "红线", "repo_lease.py"):
            self.assertIn(sec, maint_txt)


if __name__ == "__main__":
    unittest.main()
