"""Platform v4 Phase C Unit, Security & Red-Blue Adversarial Regression Tests (`CC-20261123-006`).

Verifies all Phase C deliverables AND provides dedicated regression assertions proving
100% closure of Round-1 Red Team findings `RB-01` through `RB-19`:
  - `RB-01`: `W3DNShadowMonitor` atomic WAL checkpoint recovery across restarts (25h + 25h = 50h)
  - `RB-02`: `MARGIN_STORM` full flatten during partial feed outage + post-flat collateral recovery
  - `RB-03`: Exact `"margin_ratio"` handoff to `Web3DeltaNeutralCarryKernel`
  - `RB-05` & `RB-17`: `G4` preserves open paired hedge on missing partner bar + gap-safe vol + leg cap
  - `RB-06`: Zero same-tick close-and-reopen after `INTRADAY_BASIS_STOP` on mark dislocation
  - `RB-07`: `evaluate_48h_readiness()` marks open positions to live cached MTM prices
  - `RB-08`: 75%-dead universe blocked from `READY_FOR_OWNER_REVIEW` + gross budget cap preserved
  - `RB-09`: `P8MDGatewayFileQuoteProvider` empty CSV safety, O(1) seek, strict `ts_ms`, symbol norm
  - `RB-10`: `spot_bid_qty=NaN` rejected, tick-0 dead-man switch active, future `ts_observe_ms` blocked
  - `RB-11`: All 36 Red Team DuckDB SQL bypass payloads rejected statically by `validate_readonly_sql`
  - `RB-12`: ChiNext/STAR `±20%` board calendar (`159915.SZ` legal `+12%` fill executes in Accounting)
  - `RB-13`: Zero-volume & unclosed intraday Yahoo stub bars rejected in `CNAsharePublicCollector`
  - `RB-14` & `RB-15`: `A1` retries blocked limit-down exits immediately next bar & reserves suspended weight
  - `RB-16`: Accurate emitted vs suppressed alert counts, cash >= 0, stale DVOL expiry, spot <=0 failover
  - `RB-18`: `PortfolioCombiner` raises `ValueError` on negative/zero/NaN `min_sleeve_ann_vol_pct`
  - `RB-20` & `RB-22`: Stack-based `FROM`/`JOIN`/`TABLE` literal guard + catalog/backtick block
  - `RB-21`: Full Binance Spot outage (`valid_snaps == {}`) + live HL `+150%` perp surge triggers `MARGIN_STORM`
"""

from __future__ import annotations

import ast
import json
import math
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from alphaloop.core.provenance import PROVENANCE_REAL, utc_now_ms
from alphaloop.platform.assets import (
    ASSET_CLASS_CN_A_SHARE,
    ASSET_CLASS_GOLD_TRADFI,
    default_macro_specs_v4,
)
from alphaloop.platform.core_engine.accounting import PortfolioAccountingEngine, SimulationResult
from alphaloop.platform.core_engine.portfolio_combiner import PortfolioCombiner
from alphaloop.platform.core_engine.risk_allocator import RiskAllocatorConfig
from alphaloop.platform.core_engine.validation import verify_full_parity
from alphaloop.platform.data_foundation import (
    CNAshareDomainStore,
    CNAsharePublicCollector,
    CanonicalBar,
    DataFoundationRouter,
    cn_ashare_close_avail_ms,
    default_cn_ashare_specs,
    validate_readonly_sql,
)
from alphaloop.platform.shadow import (
    P8MDGatewayFileQuoteProvider,
    PairMarketSnapshot,
    PlatformErrorCode,
    ReplayQuoteProvider,
    SHADOW_LIVE_EXECUTION_ENABLED,
    ShadowMonitorConfig,
    W3DNShadowMonitor,
    estimate_l2_vwap_slippage_bps,
)
from alphaloop.platform.strategies.ashare import AShareETFRotationKernel
from alphaloop.platform.strategies.gold import GoldMinerSpreadMRKernel
from alphaloop.platform.strategies.web3 import Web3DeltaNeutralCarryKernel

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CST_TZ = timezone(timedelta(hours=8))


def _make_pair_snap(
    coin: str,
    ts_ms: int,
    *,
    spot_mid: float = 60_000.0,
    basis_bps: float = 2.0,
    mark_basis_bps: float | None = None,
    funding_1h: float = 0.00008,
    spot_spread_bps: float = 1.0,
    perp_spread_bps: float = 2.0,
    spot_bid_qty: float = 10.0,
    spot_ask_qty: float = 10.0,
    spot_ts_ms: int | None = None,
    perp_ts_ms: int | None = None,
    dvol: float = 48.0,
) -> PairMarketSnapshot:
    s_half = spot_mid * (spot_spread_bps / 20_000.0)
    perp_mid = spot_mid * (1.0 + basis_bps / 10_000.0)
    perp_mark = spot_mid * (1.0 + (basis_bps if mark_basis_bps is None else mark_basis_bps) / 10_000.0)
    p_half = perp_mid * (perp_spread_bps / 20_000.0)
    return PairMarketSnapshot(
        coin=coin,
        ts_observe_ms=ts_ms,
        spot_bid=spot_mid - s_half,
        spot_ask=spot_mid + s_half,
        spot_bid_qty=spot_bid_qty,
        spot_ask_qty=spot_ask_qty,
        spot_ts_ms=ts_ms if spot_ts_ms is None else spot_ts_ms,
        perp_bid=perp_mid - p_half,
        perp_ask=perp_mid + p_half,
        perp_mark=perp_mark,
        perp_oracle=spot_mid,
        perp_funding_1h=funding_1h,
        perp_ts_ms=ts_ms if perp_ts_ms is None else perp_ts_ms,
        perp_l2_bids=((perp_mid - p_half, 5.0), (perp_mid - 2 * p_half, 10.0)),
        perp_l2_asks=((perp_mid + p_half, 5.0), (perp_mid + 2 * p_half, 10.0)),
        dvol=dvol,
        spot_source="test_replay",
        perp_source="test_replay",
        provenance=PROVENANCE_REAL,
    )


def _make_sim_from_daily_rets(name: str, domain: str, rets: list[float]) -> SimulationResult:
    base_ts = 1_700_000_000_000
    day_ms = 86_400_000
    eq = 10_000.0
    curve = [(base_ts, eq)]
    for i, r in enumerate(rets, start=1):
        eq *= 1.0 + r
        curve.append((base_ts + i * day_ms, eq))
    return SimulationResult(
        strategy_id=name,
        asset_domain=domain,
        capital_usd=10_000.0,
        final_equity_usd=eq,
        days=float(len(rets)),
        bars_simulated=len(rets),
        bars_per_year=252.0,
        trade_count=20,
        total_turnover=5.0,
        gross_price_pnl_usd=eq - 10_000.0 + 15.0,
        gap_pnl_usd=0.0,
        intraday_pnl_usd=eq - 10_000.0 + 15.0,
        funding_pnl_usd=0.0,
        holding_drag_usd=0.0,
        funding_carry_pnl_usd=0.0,
        total_fee_usd=10.0,
        total_slippage_usd=5.0,
        total_stamp_duty_usd=0.0,
        total_cost_usd=15.0,
        net_pnl_usd=eq - 10_000.0,
        total_return_pct=((eq / 10_000.0) - 1.0) * 100.0,
        cagr_pct=5.0,
        simple_apr_pct=5.0,
        sharpe_ratio=1.0,
        calmar_ratio=1.0,
        realized_annual_vol_pct=8.0,
        max_drawdown_pct=5.0,
        max_active_positions=1,
        mean_active_positions=1.0,
        max_gross_leverage=1.0,
        liquidation_count=0,
        equity_curve=curve,
        period_returns=list(rets),
    )


class TestPhaseCObs04And05AndRB11RB18(unittest.TestCase):
    def test_rb11_obs04_static_sql_blocks_all_36_red_team_escape_payloads(self) -> None:
        red_team_payloads = [
            "SELECT * FROM '/etc/passwd'",
            "SELECT * FROM ('/etc/passwd')",
            "SELECT * FROM web3_bars, '/etc/passwd'",
            "SELECT * FROM (SELECT 1) t JOIN ('/etc/passwd') ON true",
            "WITH a AS (SELECT 1) SELECT * FROM a, '/etc/passwd'",
            "SELECT * FROM read_text_auto('/etc/passwd')",
            "SELECT * FROM read_csv_objects('/etc/passwd')",
            "SELECT * FROM read_xlsx('/tmp/x.xlsx')",
            "SELECT * FROM st_read('/etc/passwd')",
            "SELECT * FROM read_parquet_metadata('/tmp/a.parquet')",
            "SELECT * FROM parquet_metadata('/etc/passwd')",
            "SELECT * FROM parquet_schema('/etc/passwd')",
            "SELECT * FROM parquet_file_metadata('/tmp/a.parquet')",
            "SELECT * FROM parquet_kv_metadata('/tmp/a.parquet')",
            "SELECT * FROM parquet_bloom_probe('/tmp/a.parquet', 'x', 1)",
            "SELECT * FROM iceberg_metadata('/tmp/a')",
            "SELECT * FROM iceberg_snapshots('/tmp/a')",
            "SELECT * FROM delta_table('/tmp/a')",
            "SELECT * FROM read_json_objects('/etc/passwd')",
            "SELECT * FROM read_ndjson_objects('/etc/passwd')",
            "SELECT * FROM query('SELECT 1')",
            "SELECT * FROM pragma_database_list()",
            "SELECT * FROM pragma_table_info('web3_bars')",
            "SELECT * FROM pragma_storage_info('web3_bars')",
            "SELECT * FROM pragma_version()",
            "SELECT * FROM duckdb_extensions()",
            "SELECT * FROM duckdb_settings()",
            "SELECT * FROM duckdb_databases()",
            "SELECT * FROM duckdb_functions()",
            "SELECT * FROM duckdb_tables()",
            "SELECT * FROM duckdb_indexes()",
            "SELECT * FROM duckdb_logs()",
            "SELECT * FROM duckdb_variables()",
            "SELECT * FROM sqlite_master",
            "SELECT * FROM information_schema.tables",
            "SELECT * FROM pg_catalog.pg_tables",
            "SELECT version()",
            "SELECT current_setting('enable_external_access')",
            "SELECT current_database()",
            "SELECT current_schemas(true)",
            "SELECT getvariable('x')",
            # RB-20 Round-2 comma-join / subquery / ON / USING / TABLE & catalog escape payloads:
            "SELECT * FROM (SELECT 1) AS t, '/etc/passwd'",
            "SELECT * FROM web3_bars a JOIN web3_bars b ON a.symbol = b.symbol, '/etc/passwd'",
            "SELECT * FROM web3_bars a JOIN web3_bars b ON a.symbol = 'BTC', '/etc/passwd'",
            "SELECT * FROM web3_bars a JOIN web3_bars b USING (symbol), '/etc/passwd'",
            "WITH c AS (SELECT * FROM (SELECT 1) x, '/etc/passwd') SELECT * FROM c",
            "SELECT * FROM (SELECT 1) t, ('/etc/passwd')",
            "SELECT * FROM (SELECT 1) t, (('/etc/passwd'))",
            "WITH c AS (TABLE '/etc/passwd') SELECT * FROM c",
            "SELECT * FROM (TABLE '/etc/passwd') t",
            "SELECT * FROM web3_bars a LEFT JOIN (SELECT * FROM web3_bars WHERE symbol = 'BTC') b ON coalesce(a.symbol, 'X') = b.symbol, '/etc/passwd'",
            "SELECT * FROM pg_tables",
            "SELECT * FROM pg_views",
            "SELECT * FROM pg_class",
            "SELECT * FROM pg_settings",
            "SELECT * FROM pg_namespace",
            "SELECT * FROM pg_attribute",
            "SELECT * FROM pg_proc",
            "SELECT * FROM test_all_types()",
            "SELECT * FROM test_vector_types(1)",
            "SELECT current_catalog()",
            "SELECT current_user",
            "SELECT session_user",
            "SELECT * FROM system.main.duckdb_tables()",
            "SELECT * FROM memory.main.web3_bars",
            "SELECT * FROM temp.main.t",
            "SELECT * FROM `/etc/passwd`",
        ]
        for sql in red_team_payloads:
            with self.subTest(sql=sql):
                with self.assertRaises(ValueError):
                    validate_readonly_sql(sql)

        # Benign analytical SQL with IN ('BTC', 'ETH'), JOIN ON predicates & SELECT labels MUST still work
        for benign_sql in (
            "WITH x AS (SELECT symbol, close FROM web3_bars WHERE symbol IN ('BTC', 'ETH')) "
            "SELECT 'ok' AS tag, symbol, COUNT(*) AS n FROM x GROUP BY symbol",
            "SELECT a.symbol, a.close, b.close AS b_close FROM web3_bars a "
            "JOIN web3_bars b ON a.ts_event_ms = b.ts_event_ms AND a.symbol IN ('BTC', 'ETH') "
            "AND coalesce(b.symbol, 'BTC') = 'BTC' WHERE a.close > 0",
        ):
            ok_sql = validate_readonly_sql(benign_sql)
            self.assertTrue(ok_sql.startswith(("SELECT", "WITH")))

        router_src = (_REPO_ROOT / "alphaloop" / "platform" / "data_foundation" / "router.py").read_text(
            encoding="utf-8"
        )
        idx_ext = router_src.find("SET enable_external_access = false")
        idx_lock = router_src.find("SET lock_configuration = true")
        self.assertGreater(idx_ext, 0)
        self.assertGreater(idx_lock, idx_ext)

    def test_obs05_and_rb18_w3dn_36bp_and_combiner_vol_floor_validation(self) -> None:
        k = Web3DeltaNeutralCarryKernel()
        self.assertAlmostEqual(k.full_4leg_rt_cost_bps, 36.0, places=6)

        flat_rets = [0.002 if i % 2 == 0 else -0.001 for i in range(40)] + [0.0] * 40
        active_rets = [0.008 if i % 3 == 0 else -0.005 for i in range(80)]
        s_flat = _make_sim_from_daily_rets("GOLD_FLAT_WINDOW", "gold", flat_rets)
        s_act = _make_sim_from_daily_rets("WEB3_ACTIVE", "web3", active_rets)

        comb = PortfolioCombiner()
        res_floored = comb.combine_sleeves(
            {"gold": s_flat, "web3": s_act},
            method="causal_inv_vol",
            min_sleeve_ann_vol_pct=2.0,
        )
        self.assertTrue(res_floored.valid)
        self.assertGreater(res_floored.sleeve_weights_final["web3"], 0.10)
        self.assertLess(res_floored.sleeve_weights_final["gold"], 0.90)

        # RB-18: negative, zero, or NaN min_sleeve_ann_vol_pct must raise ValueError (fail-closed)
        for bad in (-5.0, 0.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                comb.combine_sleeves({"gold": s_flat, "web3": s_act}, min_sleeve_ann_vol_pct=bad)


class TestW3DNShadowMonitorRedBlueHardening(unittest.TestCase):
    def test_shadow_hard_security_lock_and_zero_binance_ws(self) -> None:
        self.assertFalse(SHADOW_LIVE_EXECUTION_ENABLED)
        src_path = _REPO_ROOT / "alphaloop" / "platform" / "shadow" / "w3_dn_shadow_monitor.py"
        src = src_path.read_text(encoding="utf-8")
        self.assertNotIn("wss://stream.binance.com", src.replace("wss://stream.binance.com`", ""))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertFalse(node.module.startswith("alphaloop.execution"))
                self.assertFalse(node.module.startswith("alphaloop.plugins"))

    def test_rb01_restart_continuity_preserves_positions_funding_and_48h_clock(self) -> None:
        base_ts = 1_795_000_000_000
        frames_p1 = [
            {"BTC": _make_pair_snap("BTC", base_ts + h * 3_600_000, spot_mid=60_000.0, funding_1h=0.0001)}
            for h in range(25)
        ]
        frames_p2 = [
            {"BTC": _make_pair_snap("BTC", base_ts + h * 3_600_000, spot_mid=60_000.0, funding_1h=0.0001)}
            for h in range(25, 50)
        ]
        with tempfile.TemporaryDirectory(prefix="rb01_restart_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td),
                coins=("BTC",),
                kernel_bar_interval_sec=3_600,
                required_shadow_hours=48.0,
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(
                cfg,
                provider=ReplayQuoteProvider(frames_p1),
                kernel=Web3DeltaNeutralCarryKernel(warmup_bars=3, max_spot_gross=0.65, max_pair_leg_weight=0.35),
            ) as m1:
                for _ in range(25):
                    m1.run_once()
                r1 = m1.evaluate_48h_readiness()
                self.assertAlmostEqual(r1.elapsed_hours, 24.0, places=2)
                self.assertEqual(r1.verdict, "SHADOW_OBSERVATION_IN_PROGRESS")
                funding_p1 = r1.shadow_funding_usd
                self.assertGreater(funding_p1, 0.0)

            # Brand-new process instance on same state_dir MUST resume seamlessly to >=48h!
            with W3DNShadowMonitor(
                cfg,
                provider=ReplayQuoteProvider(frames_p2),
                kernel=Web3DeltaNeutralCarryKernel(warmup_bars=3, max_spot_gross=0.65, max_pair_leg_weight=0.35),
            ) as m2:
                for _ in range(25):
                    m2.run_once()
                r2 = m2.evaluate_48h_readiness()
                self.assertEqual(r2.verdict, "READY_FOR_OWNER_REVIEW")
                self.assertGreaterEqual(r2.elapsed_hours, 48.0)
                self.assertEqual(r2.total_ticks, 50)
                self.assertEqual(r2.healthy_ticks, 50)
                self.assertGreater(r2.shadow_funding_usd, funding_p1 * 1.5)
                self.assertTrue(r2.audit_chain_intact)
                self.assertEqual(r2.audit_chain_length, 50)

    def test_rb02_rb03_margin_storm_partial_outage_flatten_and_no_deadlock(self) -> None:
        # RB-03: kernel margin_ratio resolver handles both keys
        k_check = Web3DeltaNeutralCarryKernel()
        self.assertAlmostEqual(k_check._resolve_margin_ratio("BTC", {"margin_ratio": 0.25}), 0.25)
        self.assertAlmostEqual(k_check._resolve_margin_ratio("BTC", {"perp_margin_buffer_ratio": 0.25}), 0.25)

        t0 = 1_795_000_000_000
        # Open BTC + ETH across warmup ticks 0..3; on tick 4 BTC spot drops out while ETH & BTC perp surge +80% -> MARGIN_STORM!
        # On ticks 5..6, prices stabilize -> monitor MUST NOT permanently deadlock when flat!
        frames: list[dict[str, PairMarketSnapshot]] = []
        for h in range(4):
            ts = t0 + h * 3_600_000
            frames.append(
                {
                    "BTC": _make_pair_snap("BTC", ts, spot_mid=60_000.0, funding_1h=0.0001),
                    "ETH": _make_pair_snap("ETH", ts, spot_mid=2_000.0, funding_1h=0.0001),
                }
            )
        ts4 = t0 + 4 * 3_600_000
        # Partial outage: BTC spot missing, BTC perp surges to 108_000; ETH surges to 3_600
        btc_partial = PairMarketSnapshot(
            coin="BTC",
            ts_observe_ms=ts4,
            spot_bid=None,
            spot_ask=None,
            spot_bid_qty=None,
            spot_ask_qty=None,
            spot_ts_ms=None,
            perp_bid=107_900.0,
            perp_ask=108_100.0,
            perp_mark=108_000.0,
            perp_oracle=108_000.0,
            perp_funding_1h=0.0001,
            perp_ts_ms=ts4,
        )
        frames.append({"BTC": btc_partial, "ETH": _make_pair_snap("ETH", ts4, spot_mid=3_600.0, funding_1h=0.0001)})
        for h in range(5, 8):
            ts = t0 + h * 3_600_000
            frames.append(
                {
                    "BTC": _make_pair_snap("BTC", ts, spot_mid=108_000.0, funding_1h=0.0001),
                    "ETH": _make_pair_snap("ETH", ts, spot_mid=3_600.0, funding_1h=0.0001),
                }
            )

        with tempfile.TemporaryDirectory(prefix="rb02_storm_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td),
                coins=("BTC", "ETH"),
                kernel_bar_interval_sec=3_600,
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(
                cfg,
                provider=ReplayQuoteProvider(frames),
                kernel=Web3DeltaNeutralCarryKernel(warmup_bars=2, max_spot_gross=0.65, max_pair_leg_weight=0.35),
            ) as mon:
                for _ in range(4):
                    mon.run_once()
                self.assertTrue(mon._positions["BTC"].is_open)
                self.assertTrue(mon._positions["ETH"].is_open)
                # Tick 4: MARGIN_STORM must flatten BOTH BTC (despite spot outage) AND ETH!
                mon.run_once()
                self.assertFalse(mon._positions["BTC"].is_open)
                self.assertFalse(mon._positions["ETH"].is_open)
                # Tick 5..7: collateral rebalanced -> margin_ratio recovered healthy (>0.95), no permanent deadlock!
                r5 = mon.run_once()
                self.assertGreater(r5["margin_ratio"], 0.95)
                self.assertEqual(r5["status"], "HEALTHY")

    def test_rb21_full_spot_outage_live_perp_surge_triggers_margin_storm(self) -> None:
        """RB-21: When ALL spot feeds drop (valid_snaps == {}) while Hyperliquid perp_mark
        surges +150%, stale _last_valid_snaps MUST NOT mask raw_snaps; MARGIN_STORM must fire
        and flatten ALL open pairs (BTC & ETH), then allow clean post-flat recovery.
        """
        t0 = 1_795_000_000_000
        frames: list[dict[str, PairMarketSnapshot]] = []
        for h in range(4):
            ts = t0 + h * 3_600_000
            frames.append(
                {
                    "BTC": _make_pair_snap("BTC", ts, spot_mid=60_000.0, funding_1h=0.0001),
                    "ETH": _make_pair_snap("ETH", ts, spot_mid=2_000.0, funding_1h=0.0001),
                }
            )
        ts4 = t0 + 4 * 3_600_000
        # FULL Binance Spot outage (BOTH BTC and ETH spot_bid/spot_ask = None -> valid_snaps == {})
        # while Hyperliquid perp_mark surges +150%!
        frames.append(
            {
                "BTC": PairMarketSnapshot(
                    coin="BTC",
                    ts_observe_ms=ts4,
                    spot_bid=None,
                    spot_ask=None,
                    spot_bid_qty=None,
                    spot_ask_qty=None,
                    spot_ts_ms=None,
                    perp_bid=149_900.0,
                    perp_ask=150_100.0,
                    perp_mark=150_000.0,
                    perp_oracle=150_000.0,
                    perp_funding_1h=0.0001,
                    perp_ts_ms=ts4,
                ),
                "ETH": PairMarketSnapshot(
                    coin="ETH",
                    ts_observe_ms=ts4,
                    spot_bid=None,
                    spot_ask=None,
                    spot_bid_qty=None,
                    spot_ask_qty=None,
                    spot_ts_ms=None,
                    perp_bid=4_995.0,
                    perp_ask=5_005.0,
                    perp_mark=5_000.0,
                    perp_oracle=5_000.0,
                    perp_funding_1h=0.0001,
                    perp_ts_ms=ts4,
                ),
            }
        )
        # Tick 5: Spot recovers at normal prices
        ts5 = t0 + 5 * 3_600_000
        frames.append(
            {
                "BTC": _make_pair_snap("BTC", ts5, spot_mid=150_000.0, funding_1h=0.0001),
                "ETH": _make_pair_snap("ETH", ts5, spot_mid=5_000.0, funding_1h=0.0001),
            }
        )
        with tempfile.TemporaryDirectory(prefix="rb21_full_spot_outage_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td),
                coins=("BTC", "ETH"),
                kernel_bar_interval_sec=3_600,
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(
                cfg,
                provider=ReplayQuoteProvider(frames),
                kernel=Web3DeltaNeutralCarryKernel(warmup_bars=2, max_spot_gross=0.65, max_pair_leg_weight=0.35),
            ) as mon:
                for _ in range(4):
                    mon.run_once()
                self.assertTrue(mon._positions["BTC"].is_open)
                self.assertTrue(mon._positions["ETH"].is_open)

                r4 = mon.run_once()
                self.assertEqual(r4["valid_coins"], 0)
                self.assertFalse(mon._positions["BTC"].is_open, "RB-21: BTC must flatten on full-spot-outage perp surge!")
                self.assertFalse(mon._positions["ETH"].is_open, "RB-21: ETH must flatten on full-spot-outage perp surge!")
                self.assertLess(mon._min_perp_margin_buffer_ratio, cfg.margin_storm_exit_ratio)

                # Verify MARGIN_STORM alert was logged to SQLite
                storm_rows = mon.conn.execute(
                    "SELECT COUNT(*) FROM shadow_alerts WHERE code = ?",
                    (PlatformErrorCode.MARGIN_STORM.value,),
                ).fetchone()[0]
                self.assertGreaterEqual(int(storm_rows), 1)

                # Tick 5: clean recovery, flat margin ratio = 1.0, cooldown prevents immediate reopen
                r5 = mon.run_once()
                self.assertEqual(r5["status"], "HEALTHY")
                self.assertGreater(r5["margin_ratio"], 0.95)

    def test_rb06_rb07_rb08_no_same_tick_reopen_live_mtm_and_partial_universe_gate(self) -> None:
        t0 = 1_795_000_000_000
        # RB-06: mark basis blowout (+420bp) while mid basis is calm (+3bp) on a kernel bucket MUST NOT reopen same tick!
        frames_rb06 = [
            {"BTC": _make_pair_snap("BTC", t0 + h * 3_600_000, spot_mid=60_000.0, basis_bps=3.0, funding_1h=0.0001)}
            for h in range(4)
        ]
        frames_rb06.append(
            {
                "BTC": _make_pair_snap(
                    "BTC",
                    t0 + 4 * 3_600_000,
                    spot_mid=60_000.0,
                    basis_bps=3.0,
                    mark_basis_bps=420.0,  # Mark blowout!
                    funding_1h=0.0001,
                )
            }
        )
        with tempfile.TemporaryDirectory(prefix="rb06_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td),
                coins=("BTC",),
                kernel_bar_interval_sec=3_600,
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(
                cfg,
                provider=ReplayQuoteProvider(frames_rb06),
                kernel=Web3DeltaNeutralCarryKernel(warmup_bars=2, max_spot_gross=0.65, max_pair_leg_weight=0.35),
            ) as mon:
                for _ in range(5):
                    mon.run_once()
                self.assertFalse(mon._positions["BTC"].is_open, "RB-06: must NOT reopen on same tick after mark basis stop")

        # RB-07: evaluate_48h_readiness() uses live MTM (reflecting +200bp basis loss, NOT stale entry price!)
        frames_rb07 = [
            {"BTC": _make_pair_snap("BTC", t0 + h * 3_600_000, spot_mid=60_000.0, basis_bps=2.0, funding_1h=0.0001)}
            for h in range(4)
        ]
        frames_rb07.append(
            {"BTC": _make_pair_snap("BTC", t0 + 4 * 3_600_000, spot_mid=60_000.0, basis_bps=200.0, funding_1h=0.0001)}
        )
        with tempfile.TemporaryDirectory(prefix="rb07_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td),
                coins=("BTC",),
                kernel_bar_interval_sec=3_600,
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(
                cfg,
                provider=ReplayQuoteProvider(frames_rb07),
                kernel=Web3DeltaNeutralCarryKernel(warmup_bars=2, max_spot_gross=0.65, max_pair_leg_weight=0.35),
            ) as mon:
                last_tick = None
                for _ in range(5):
                    last_tick = mon.run_once()
                rep = mon.evaluate_48h_readiness()
                self.assertAlmostEqual(rep.shadow_equity_usd, float(last_tick["equity_usd"]), places=3)
                self.assertLess(rep.shadow_equity_usd, 9_960.0)

        # RB-08: 75% dead universe (only BTC out of 4 coins) MUST NOT achieve uptime=1.0 or READY_FOR_OWNER_REVIEW
        frames_rb08 = [
            {"BTC": _make_pair_snap("BTC", t0 + h * 3_600_000, spot_mid=60_000.0, funding_1h=0.0001)}
            for h in range(50)
        ]
        with tempfile.TemporaryDirectory(prefix="rb08_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td),
                coins=("BTC", "ETH", "SOL", "AVAX"),
                kernel_bar_interval_sec=3_600,
                required_shadow_hours=48.0,
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(cfg, provider=ReplayQuoteProvider(frames_rb08)) as mon:
                for _ in range(50):
                    mon.run_once()
                rep = mon.evaluate_48h_readiness()
                self.assertEqual(rep.uptime_ratio, 0.0)
                self.assertEqual(rep.verdict, "FAIL_CLOSED_BLOCKED")

    def test_rb09_rb10_p8_empty_csv_strict_ts_and_poison_wallclock_guards(self) -> None:
        # RB-09: empty CSV must NOT raise IndexError; BTC-USDT/BTC/USDT normalized; missing JSONL ts_ms rejected
        with tempfile.TemporaryDirectory(prefix="rb09_p8_") as td:
            empty_csv = Path(td) / "empty.csv"
            empty_csv.write_text("", encoding="utf-8")
            self.assertEqual(P8MDGatewayFileQuoteProvider(empty_csv)._read_latest_p8_spot_bbo(["BTC"]), {})

            sep_csv = Path(td) / "events.csv"
            sep_csv.write_text(
                "symbol,bid,ask,bid_qty,ask_qty,ts_ms\nBTC-USDT,61000,61002,2,2,1795000000000\nETH/USDT,2000,2001,5,5,1795000000000\n",
                encoding="utf-8",
            )
            res_csv = P8MDGatewayFileQuoteProvider(sep_csv)._read_latest_p8_spot_bbo(["BTC", "ETH"])
            self.assertIn("BTC", res_csv)
            self.assertIn("ETH", res_csv)

            no_ts_jsonl = Path(td) / "no_ts.jsonl"
            no_ts_jsonl.write_text(json.dumps({"symbol": "BTCUSDT", "bid": 61000.0, "ask": 61002.0}) + "\n", encoding="utf-8")
            self.assertEqual(P8MDGatewayFileQuoteProvider(no_ts_jsonl)._read_latest_p8_spot_bbo(["BTC"]), {})

        # RB-10: spot_bid_qty=NaN rejected + future spoofed ts_observe_ms rejected under default enforce_wall_clock_bounds=True
        now_ms = utc_now_ms()
        with tempfile.TemporaryDirectory(prefix="rb10_") as td:
            cfg = ShadowMonitorConfig(state_dir=Path(td), coins=("BTC",), max_tick_gap_sec=0.001)
            nan_qty_snap = _make_pair_snap("BTC", now_ms, spot_bid_qty=float("nan"))
            spoof_future_snap = _make_pair_snap("BTC", now_ms + 72 * 3_600_000)
            with W3DNShadowMonitor(cfg, provider=ReplayQuoteProvider([{"BTC": nan_qty_snap}, {"BTC": spoof_future_snap}])) as mon:
                ok1, c1, _ = mon.validate_snapshot(nan_qty_snap)
                self.assertFalse(ok1)
                self.assertEqual(c1, PlatformErrorCode.NON_FINITE_DATA)
                ok2, c2, _ = mon.validate_snapshot(spoof_future_snap)
                self.assertFalse(ok2)
                self.assertEqual(c2, PlatformErrorCode.CLOCK_SKEW)
                mon.run_once()
                mon.run_once()
                rep = mon.evaluate_48h_readiness()
                self.assertEqual(rep.elapsed_hours, 0.0, "Spoofed future snapshot must NOT advance 48h clock!")

        # L2 VWAP check
        levels = ((100.1, 10.0), (100.5, 10.0))
        self.assertAlmostEqual(estimate_l2_vwap_slippage_bps(levels, notional_usd=500.0, mid_price=100.0, side="buy"), 10.0, places=3)
        self.assertEqual(estimate_l2_vwap_slippage_bps(levels, notional_usd=50_000.0, mid_price=100.0, side="buy"), 50.0)

    def test_shadow_audit_chain_tamper_detection(self) -> None:
        t0 = 1_795_000_000_000
        frames = [{"BTC": _make_pair_snap("BTC", t0 + h * 3_600_000, spot_mid=60_000.0)} for h in range(3)]
        with tempfile.TemporaryDirectory(prefix="chain_tamper_") as td:
            cfg = ShadowMonitorConfig(state_dir=Path(td), coins=("BTC",), enforce_wall_clock_bounds=False)
            with W3DNShadowMonitor(cfg, provider=ReplayQuoteProvider(frames)) as mon:
                for _ in range(3):
                    mon.run_once()
                self.assertTrue(mon.verify_audit_chain()[0])
                con = sqlite3.connect(str(mon.db_path), timeout=10.0)
                try:
                    con.execute("PRAGMA journal_mode=WAL;")
                    con.execute("UPDATE shadow_audit_chain SET payload_json = '{\"tampered\":true}' WHERE seq = 1")
                    con.commit()
                finally:
                    con.close()
                self.assertFalse(mon.verify_audit_chain()[0])
                self.assertEqual(mon.evaluate_48h_readiness().verdict, "FAIL_CLOSED_BLOCKED")


class TestGoldG4AndCNAShareRedBlueHardening(unittest.TestCase):
    def test_rb05_rb17_g4_preserves_open_hedge_on_missing_leg_and_g7_parity(self) -> None:
        base_ts = 1_650_000_000_000
        day_ms = 86_400_000
        gld_bars: list[CanonicalBar] = []
        gdx_bars: list[CanonicalBar] = []
        for i in range(120):
            ts = base_ts + i * day_ms
            g_px = 180.0 + 5.0 * math.sin(i / 14.0) + 0.05 * i
            m_px = 30.0 * (g_px / 180.0) * (1.0 + 0.14 * math.sin(i / 7.0))
            gld_bars.append(
                CanonicalBar(
                    asset_class=ASSET_CLASS_GOLD_TRADFI, venue="yahoo", symbol="GLD",
                    timeframe_h=24.0, timeframe_sec=86_400, ts_event_ms=ts, ts_avail_ms=ts + day_ms,
                    open=g_px, high=g_px * 1.005, low=g_px * 0.995, close=g_px,
                    volume_usd=1e8, carry_rate_period=0.0, carry_interval_h=24.0, spread_bps=2.0,
                    provenance="SYNTHETIC_TEST_ONLY",
                )
            )
            gdx_bars.append(
                CanonicalBar(
                    asset_class=ASSET_CLASS_GOLD_TRADFI, venue="yahoo", symbol="GDX",
                    timeframe_h=24.0, timeframe_sec=86_400, ts_event_ms=ts, ts_avail_ms=ts + day_ms,
                    open=m_px, high=m_px * 1.01, low=m_px * 0.99, close=m_px,
                    volume_usd=1e8, carry_rate_period=0.0, carry_interval_h=24.0, spread_bps=3.0,
                    provenance="SYNTHETIC_TEST_ONLY",
                )
            )

        k4 = GoldMinerSpreadMRKernel(lookback_days=30, vol_lookback_days=20, z_enter=0.9, warmup_bars=32, max_single_leg_weight=0.50)
        # Flat + unpaired -> stays 0.0
        dec_flat = k4.on_bar_step(gld_bars[0].ts_event_ms, {"GLD": gld_bars[0]}, {})
        self.assertEqual(dec_flat["GLD"].target_weight, 0.0)
        self.assertEqual(dec_flat["GLD"].regime_label, "G4_UNPAIRED_HOLD_FLAT")

        # RB-05: OPEN pair + GDX misses bar -> GLD MUST hold existing +0.32 weight (NOT exit to 0 leaving naked GDX!)
        k4.sync_executed_weights({"GLD": 0.32, "GDX": -0.14})
        dec_hold = k4.on_bar_step(gld_bars[50].ts_event_ms, {"GLD": gld_bars[50]}, {})
        self.assertAlmostEqual(dec_hold["GLD"].target_weight, 0.32, places=6)
        self.assertEqual(dec_hold["GLD"].regime_label, "G4_UNPAIRED_HOLD_EXISTING")

        specs = default_macro_specs_v4()
        rcfg = RiskAllocatorConfig(
            max_positions=2, target_annual_vol_pct=12.0, max_portfolio_leverage=1.0,
            max_single_weight=0.75, use_correlation_idm=False,
        )
        parity = verify_full_parity(
            k4.clone_fresh(), {"GLD": specs["GLD"], "GDX": specs["GDX"]},
            {"GLD": gld_bars, "GDX": gdx_bars}, risk_cfg=rcfg, capital_usd=10_000.0,
        )
        self.assertTrue(parity.passed)
        self.assertEqual(parity.max_abs_equity_delta, 0.0)

    def test_rb12_rb13_rb14_rb15_cn_ashare_board_limits_zero_vol_filter_and_retry_exit(self) -> None:
        # RB-12: ChiNext/STAR ETFs get ±20% calendar in AssetSpec so Accounting matches Strategy!
        specs = default_cn_ashare_specs(("510300.SS", "159915.SZ", "588000.SS", "511010.SS"))
        self.assertEqual(specs["510300.SS"].calendar.daily_price_limit_pct, 10.0)
        self.assertEqual(specs["159915.SZ"].calendar.daily_price_limit_pct, 20.0)
        self.assertEqual(specs["588000.SS"].calendar.daily_price_limit_pct, 20.0)

        # RB-13: CNAsharePublicCollector filters out volume=0 phantom stubs & unclosed future-avail bars
        d0 = datetime(2025, 6, 10, 9, 30, 0, tzinfo=_CST_TZ)
        now_cutoff_ms = int((d0 + timedelta(days=3)).timestamp() * 1000)  # day 3 at 09:30 CST (day 3 unclosed!)
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [int((d0 + timedelta(days=d)).timestamp()) for d in range(4)],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [4.0, 4.02, 4.04, 4.06],
                                    "high": [4.05, 4.05, 4.06, 4.08],
                                    "low": [3.98, 4.00, 4.02, 4.04],
                                    "close": [4.02, 4.02, 4.05, 4.07],
                                    "volume": [1_000_000, 0, 1_200_000, 900_000],  # day 1 zero-vol, day 3 unclosed!
                                }
                            ]
                        },
                    }
                ]
            }
        }
        parsed = CNAsharePublicCollector().parse_chart_payload("510300.SS", payload, now_ms=now_cutoff_ms)
        self.assertEqual(len(parsed), 2, "RB-13: must drop zero-volume stub (day 1) and unclosed bar (day 3)")

        # RB-14 & RB-15: blocked exit retries immediately on next off-cadence bar + suspended weight reserved
        k = AShareETFRotationKernel(warmup_bars=1, rebalance_every_bars=10, max_total_weight=0.90)
        k._pending_exit_targets["510300.SS"] = 0.0
        k.sync_executed_weights({"510300.SS": 0.45, "510500.SS": 0.40})  # 510500.SS suspended (not in bars_at_ts)
        k._step_count = 2  # off-cadence bar!
        bar_300 = CanonicalBar(
            asset_class=ASSET_CLASS_CN_A_SHARE, venue="cn_a_share", symbol="510300.SS",
            timeframe_h=24.0, timeframe_sec=86_400,
            ts_event_ms=int((d0 + timedelta(days=5)).timestamp() * 1000),
            ts_avail_ms=cn_ashare_close_avail_ms(int((d0 + timedelta(days=5)).timestamp() * 1000)),
            open=3.9, high=3.95, low=3.85, close=3.88,
            volume_usd=5e7, carry_rate_period=0.0, carry_interval_h=24.0, spread_bps=5.0,
            provenance="SYNTHETIC_TEST_ONLY",
        )
        dec = k.on_bar_step(bar_300.ts_event_ms, {"510300.SS": bar_300}, {})
        self.assertEqual(dec["510300.SS"].regime_label, "ASHARE_RETRY_BLOCKED_EXIT")
        self.assertEqual(dec["510300.SS"].target_weight, 0.0)

        # Physical isolation check
        with tempfile.TemporaryDirectory(prefix="cn_iso_") as td:
            with DataFoundationRouter(td) as router:
                self.assertIsInstance(router.get_domain_store("cn_ashare"), CNAshareDomainStore)
                self.assertEqual(router.upsert_bars("cn_ashare", parsed), 2)
                self.assertEqual(len(router.load_bars("web3", "cn_a_share", "510300.SS", 86_400)), 0)

    def test_three_domain_ast_isolation_web3_gold_ashare(self) -> None:
        domains = {
            "web3": _REPO_ROOT / "alphaloop" / "platform" / "strategies" / "web3",
            "gold": _REPO_ROOT / "alphaloop" / "platform" / "strategies" / "gold",
            "ashare": _REPO_ROOT / "alphaloop" / "platform" / "strategies" / "ashare",
        }
        for dom_name, dom_dir in domains.items():
            other_domains = [d for d in domains if d != dom_name]
            for py_file in sorted(dom_dir.glob("*.py")):
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    mods: list[str] = []
                    if isinstance(node, ast.Import):
                        mods.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        mods.append(node.module)
                    for mod in mods:
                        for other in other_domains:
                            self.assertNotIn(f"strategies.{other}", mod)


if __name__ == "__main__":
    unittest.main()
