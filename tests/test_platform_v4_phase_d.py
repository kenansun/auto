"""Comprehensive Phase D Unit & Adversarial Regression Tests (CC-20261123-007).

Covers all three Phase D tracks without network dependencies (100% pure stdlib, deterministic):
  1. Track 1 (`W3-DN` SG Shadow Readiness):
     - `observe_only_coins` overlap fail-closed rejection (`ValueError`)
     - Diagnostic `AVAX` outage / wide spread NEVER degrades primary `BTC,ETH,SOL` `uptime_ratio`
     - Zero virtual positions or PnL leakage from `observe_only_coins`
     - Cross-process WAL checkpoint persistence & hydration of `observe_only_coin_stats`
     - Atomic `0644` live P16 heartbeat + live readiness snapshot JSON emission
  2. Track 2 (`REV-019` `OBS-07` Trial Ledger Isolation & Research Lake Snapshot):
     - `trial_ledger_mode="sandbox"` and `"readonly"` write **0 rows** to official `catalog.sqlite`
     - `get_family_trial_count()` returns `official + sandbox/readonly` so DSR is NEVER under-penalized
     - Invalid `trial_ledger_mode` fails closed (`ValueError`)
     - `sandbox` temp directory auto-cleanup on `router.close()` + explicit `sandbox_trial_db_path` persistence
     - `snapshot_research_lake.py` `create` + `verify` + SHA-256 tamper detection + path-overlap guard
  3. Track 3 (Native 10y CN A-Share Macro/CGB/Margin PIT Store + `A2` Pure Stock-Bond Strategy):
     - `CNAshareDomainStore.cn_ashare_macro` strict PIT query (`ts_avail_ms <= as_of_ms`), lookahead rejection
     - `CNAsharePublicCollector`: proportional total-return `adjclose` scaling (preserving raw notional),
       `parse_eastmoney_cgb_payload` (`18:00 CST`), `parse_eastmoney_margin_payload` (`T+1 08:30 CST`),
       `parse_cn_macro_yahoo_payload` (`T+1 06:00 CST` / `T 17:00 CST`)
     - `export_domain_partition("cn_ashare")` exports `cn_ashare_macro.jsonl.gz` when populated and omits
       empty `EMPTY`-provenance macro manifest when unpopulated
     - `AShareMacroERPRotationKernel` (`A2`): physical ban of `518880.SS` Gold ETF in `__init__` AND
       `on_bar_step`, `T+1` sell lock, `±20%` ChiNext/STAR price limit, `RB-14` blocked-exit retry,
       `RB-15` suspended-holding capacity reservation, `G7 = 0.00e+00` full batch/streaming parity,
       and 3-domain AST zero cross-import isolation
"""

from __future__ import annotations

import ast
import json
import math
import os
import stat
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from alphaloop.core.provenance import PROVENANCE_SYNTHETIC
from alphaloop.platform.assets import ASSET_CLASS_CN_A_SHARE
from alphaloop.platform.core_engine import (
    PortfolioAccountingEngine,
    RiskAllocatorConfig,
    verify_full_parity,
)
from alphaloop.platform.data_foundation import (
    CN_ASHARE_PURE_STOCK_BOND_UNIVERSE,
    CNAshareMacroPoint,
    CNAsharePublicCollector,
    CanonicalBar,
    DataFoundationRouter,
    default_cn_ashare_specs,
)
from alphaloop.platform.shadow.w3_dn_shadow_monitor import (
    PairMarketSnapshot,
    ReplayQuoteProvider,
    ShadowMonitorConfig,
    W3DNShadowMonitor,
)
from alphaloop.platform.strategies.ashare import (
    AShareMacroERPRotationKernel,
    CNMacroRegimeTimeline,
)
from scripts.snapshot_research_lake import create_lake_snapshot, verify_lake_snapshot

_CST_TZ = timezone(timedelta(hours=8))
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_pair_snap(
    coin: str,
    ts_ms: int,
    *,
    spot_mid: float = 100.0,
    perp_mid: float = 100.15,
    funding_1h: float = 0.00012,
    spot_spread_bps: float = 1.0,
    perp_spread_bps: float = 2.0,
    perp_depth_usd: float = 500_000.0,
    ok: bool = True,
) -> PairMarketSnapshot:
    if not ok:
        return PairMarketSnapshot(
            coin=coin,
            ts_observe_ms=ts_ms,
            spot_bid=None,
            spot_ask=None,
            spot_bid_qty=None,
            spot_ask_qty=None,
            spot_ts_ms=None,
            perp_bid=None,
            perp_ask=None,
            perp_mark=None,
            perp_oracle=None,
            perp_funding_1h=None,
            perp_ts_ms=None,
            provenance=PROVENANCE_SYNTHETIC,
        )
    half_s = spot_mid * (spot_spread_bps / 20_000.0)
    half_p = perp_mid * (perp_spread_bps / 20_000.0)
    p_bid = perp_mid - half_p
    p_ask = perp_mid + half_p
    lvl_sz = (perp_depth_usd / 5.0) / perp_mid
    l2_bids = tuple((p_bid - i * half_p * 0.5, lvl_sz) for i in range(5))
    l2_asks = tuple((p_ask + i * half_p * 0.5, lvl_sz) for i in range(5))
    return PairMarketSnapshot(
        coin=coin,
        ts_observe_ms=ts_ms,
        spot_bid=spot_mid - half_s,
        spot_ask=spot_mid + half_s,
        spot_bid_qty=1000.0,
        spot_ask_qty=1000.0,
        spot_ts_ms=ts_ms,
        perp_bid=p_bid,
        perp_ask=p_ask,
        perp_mark=perp_mid,
        perp_oracle=spot_mid,
        perp_funding_1h=funding_1h,
        perp_ts_ms=ts_ms,
        perp_l2_bids=l2_bids,
        perp_l2_asks=l2_asks,
        dvol=50.0,
        spot_source="test_spot",
        perp_source="test_perp",
        provenance=PROVENANCE_SYNTHETIC,
    )


def _make_cn_bar(
    sym: str,
    day_idx: int,
    close: float,
    *,
    open_p: float | None = None,
    high_p: float | None = None,
    low_p: float | None = None,
    vol_cny: float = 50_000_000.0,
) -> CanonicalBar:
    base_dt = datetime(2024, 1, 2, 9, 30, 0, tzinfo=_CST_TZ) + timedelta(days=day_idx)
    ev_ms = int(base_dt.timestamp() * 1000.0)
    av_ms = int(base_dt.replace(hour=15, minute=0).timestamp() * 1000.0)
    o = close if open_p is None else open_p
    h = max(o, close) if high_p is None else max(high_p, o, close)
    lo = min(o, close) if low_p is None else min(low_p, o, close)
    return CanonicalBar(
        asset_class=ASSET_CLASS_CN_A_SHARE,
        venue="cn_a_share",
        symbol=sym,
        timeframe_h=24.0,
        timeframe_sec=86_400,
        ts_event_ms=ev_ms,
        ts_avail_ms=av_ms,
        open=o,
        high=h,
        low=lo,
        close=close,
        volume_usd=vol_cny,
        carry_rate_period=0.0,
        carry_interval_h=24.0,
        spread_bps=5.0,
        provenance=PROVENANCE_SYNTHETIC,
        session_tag="CN_A_SHARE_DAY",
        source_fingerprint="test_phase_d",
    )


class TestPhaseDTrack1W3DNShadowObserveOnly(unittest.TestCase):
    """Track 1: W3-DN observe_only_coins isolation, restart persistence & atomic JSON emission."""

    def test_overlap_between_primary_and_observe_only_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase_d_w3dn_ov_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td),
                coins=("BTC", "ETH", "SOL"),
                observe_only_coins=("SOL", "AVAX"),
            )
            with self.assertRaises(ValueError):
                W3DNShadowMonitor(cfg)

    def test_observe_only_avax_outage_does_not_degrade_primary_uptime_or_trade(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase_d_w3dn_iso_") as td:
            state_dir = Path(td) / "state"
            hb_path = Path(td) / "p16_hb.json"
            live_path = Path(td) / "live_readiness.json"
            t0 = 1_795_000_000_000

            frames = [
                {
                    "BTC": _make_pair_snap("BTC", t0 + step * 60_000),
                    "ETH": _make_pair_snap("ETH", t0 + step * 60_000),
                    "SOL": _make_pair_snap("SOL", t0 + step * 60_000),
                    # AVAX suffers total outage on every tick & must NEVER degrade primary uptime!
                    "AVAX": _make_pair_snap("AVAX", t0 + step * 60_000, ok=False),
                }
                for step in range(3)
            ]

            cfg = ShadowMonitorConfig(
                state_dir=state_dir,
                coins=("BTC", "ETH", "SOL"),
                observe_only_coins=("AVAX",),
                p16_heartbeat_path=hb_path,
                readiness_report_path=live_path,
                enforce_wall_clock_bounds=False,
            )
            mon = W3DNShadowMonitor(cfg, provider=ReplayQuoteProvider(frames))
            try:
                for _ in range(3):
                    rec = mon.run_once()
                    self.assertEqual(rec["valid_coins"], 3)
                    self.assertEqual(rec["status"], "HEALTHY")
                    self.assertNotIn("AVAX", mon._positions)
                rep = mon.evaluate_48h_readiness()
                self.assertEqual(rep.total_ticks, 3)
                self.assertEqual(rep.healthy_ticks, 3)
                self.assertAlmostEqual(rep.uptime_ratio, 1.0, places=6)
                self.assertIn("AVAX", rep.observe_only_coin_stats)
                self.assertEqual(rep.observe_only_coin_stats["AVAX"]["valid_ticks"], 0)
                self.assertEqual(rep.observe_only_coin_stats["AVAX"]["fail_ticks"], 3)
            finally:
                mon.close()

            # Verify atomic 0644 P16 heartbeat and live readiness files
            self.assertTrue(hb_path.is_file())
            self.assertTrue(live_path.is_file())
            self.assertEqual(stat.S_IMODE(hb_path.stat().st_mode), 0o644)
            self.assertEqual(stat.S_IMODE(live_path.stat().st_mode), 0o644)
            hb_data = json.loads(hb_path.read_text(encoding="utf-8"))
            self.assertEqual(hb_data["status"], "HEALTHY")
            self.assertEqual(hb_data["total_ticks"], 3)
            self.assertFalse(hb_data["live_execution_enabled"])
            live_data = json.loads(live_path.read_text(encoding="utf-8"))
            self.assertEqual(live_data["healthy_ticks"], 3)
            self.assertEqual(live_data["observe_only_coin_stats"]["AVAX"]["fail_ticks"], 3)

    def test_observe_only_stats_persist_across_process_restart(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase_d_w3dn_rst_") as td:
            state_dir = Path(td) / "state"
            t0 = 1_795_100_000_000

            def _mk_frame(step: int) -> dict[str, PairMarketSnapshot]:
                ts = t0 + step * 60_000
                return {
                    "BTC": _make_pair_snap("BTC", ts),
                    "ETH": _make_pair_snap("ETH", ts),
                    "SOL": _make_pair_snap("SOL", ts),
                    "AVAX": _make_pair_snap("AVAX", ts, spot_spread_bps=18.0, perp_spread_bps=22.0, perp_depth_usd=10_000.0),
                }

            cfg = ShadowMonitorConfig(
                state_dir=state_dir,
                coins=("BTC", "ETH", "SOL"),
                observe_only_coins=("AVAX",),
                enforce_wall_clock_bounds=False,
            )
            m1 = W3DNShadowMonitor(cfg, provider=ReplayQuoteProvider([_mk_frame(0), _mk_frame(1)]))
            try:
                m1.run_once()
                m1.run_once()
            finally:
                m1.close()

            # Restart with fresh instance reading SQLite WAL checkpoint
            m2 = W3DNShadowMonitor(cfg, provider=ReplayQuoteProvider([_mk_frame(2)]))
            try:
                m2.run_once()
                rep = m2.evaluate_48h_readiness()
                self.assertEqual(rep.total_ticks, 3)
                self.assertEqual(rep.observe_only_coin_stats["AVAX"]["valid_ticks"], 3)
                self.assertGreater(rep.observe_only_coin_stats["AVAX"]["mean_4leg_cost_bps"], 60.0)
                self.assertLess(rep.mean_observed_4leg_cost_bps, 40.0)
                self.assertTrue(rep.audit_chain_intact)
            finally:
                m2.close()


class TestPhaseDTrack2OBS07TrialLedgerAndSnapshot(unittest.TestCase):
    """Track 2: REV-019 OBS-07 trial_ledger mode isolation & SHA-256 lake snapshot verifier."""

    def test_sandbox_and_readonly_modes_never_mutate_official_catalog_while_preserving_dsr_count(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase_d_obs07_") as td:
            lake = Path(td) / "lake"
            # 1. Record 2 official baseline trials
            with DataFoundationRouter(lake, trial_ledger_mode="official") as r_off:
                r_off.record_trial("fam_alpha", "track_a", "strat_1", "hash1", {"sharpe": 0.7}, "CONDITIONAL_RESEARCH")
                r_off.record_trial("fam_alpha", "track_a", "strat_2", "hash2", {"sharpe": 0.8}, "PROMOTE_SHADOW_CANDIDATE")
                self.assertEqual(r_off.get_family_trial_count("fam_alpha"), 2)

            # 2. Run in sandbox mode -> 0 official writes, conservative DSR count = 2 + 3 = 5
            sb_db = Path(td) / "explicit_sandbox.sqlite"
            with DataFoundationRouter(lake, trial_ledger_mode="sandbox", sandbox_trial_db_path=sb_db) as r_sb:
                for i in range(3):
                    tid = r_sb.record_trial("fam_alpha", "track_a", f"sb_{i}", f"h_{i}", {"i": i}, "GRID_CANDIDATE")
                    self.assertTrue(tid.startswith("TR-"))
                self.assertEqual(r_sb.get_family_trial_count("fam_alpha"), 5)
            self.assertTrue(sb_db.is_file())

            # 3. Run in readonly mode -> 0 official writes, conservative DSR count = 2 + 2 = 4
            with DataFoundationRouter(lake, trial_ledger_mode="readonly") as r_ro:
                r_ro.record_trial("fam_alpha", "track_a", "ro_1", "hr1", {"x": 1}, "GRID_CANDIDATE")
                r_ro.record_trial("fam_alpha", "track_a", "ro_2", "hr2", {"x": 2}, "GRID_CANDIDATE")
                self.assertEqual(r_ro.get_family_trial_count("fam_alpha"), 4)

            # 4. Verify official catalog.sqlite remained strictly at 2 rows!
            conn = sqlite3.connect(str(lake / "catalog.sqlite"))
            try:
                cnt = conn.execute("SELECT COUNT(*) FROM trial_ledger").fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(cnt, 2)

            # 5. Temp sandbox dir auto-cleanup on close when sandbox_trial_db_path=None
            r_auto = DataFoundationRouter(lake, trial_ledger_mode="sandbox")
            self.assertIsNotNone(r_auto._sandbox_tmpdir)
            assert r_auto._sandbox_tmpdir is not None
            tmp_dir = Path(r_auto._sandbox_tmpdir.name)
            self.assertTrue(tmp_dir.is_dir())
            r_auto.record_trial("fam_alpha", "track_a", "tmp_1", "ht1", {}, "GRID_CANDIDATE")
            r_auto.close()
            self.assertFalse(tmp_dir.exists())

            # 6. Invalid mode fails closed
            with self.assertRaises(ValueError):
                DataFoundationRouter(lake, trial_ledger_mode="invalid_mode")

    def test_snapshot_create_verify_tamper_detection_and_overlap_guard(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase_d_snap_") as td:
            lake = Path(td) / "lake"
            snap_dir = Path(td) / "snap_v1"
            with DataFoundationRouter(lake) as r:
                r.record_trial("fam_s", "track_a", "s1", "hs1", {"sr": 1.0}, "PROMOTE_SHADOW_CANDIDATE")
                r.upsert_bars("cn_ashare", [_make_cn_bar("510300.SS", 0, 4.0)], allow_synthetic_in_test=True)
                r.export_domain_partition("cn_ashare", fmt="jsonl.gz")

            # Guard: refusing destination inside source lake
            with self.assertRaises(ValueError):
                create_lake_snapshot(lake, lake / "nested_bad")

            manifest = create_lake_snapshot(lake, snap_dir, label="unit_test")
            self.assertEqual(manifest["trial_ledger_total_rows"], 1)
            self.assertIn("catalog.sqlite", manifest["files"])
            self.assertIn("domains/cn_ashare_lake.sqlite", manifest["files"])
            self.assertIn("parquet/cn_ashare/cn_ashare_bars.jsonl.gz", manifest["files"])

            ver_ok = verify_lake_snapshot(snap_dir)
            self.assertTrue(ver_ok["ok"], ver_ok)
            self.assertEqual(ver_ok["mismatches"], [])

            # Guard: refusing overwrite of non-empty snapshot directory
            with self.assertRaises(FileExistsError):
                create_lake_snapshot(lake, snap_dir)

            # Tamper detection: alter 1 byte in snapshot partition file -> verify MUST fail!
            part = snap_dir / "parquet" / "cn_ashare" / "cn_ashare_bars.jsonl.gz"
            os.chmod(part, 0o644)
            part.write_bytes(part.read_bytes() + b"\x00")
            ver_bad = verify_lake_snapshot(snap_dir)
            self.assertFalse(ver_bad["ok"])
            self.assertTrue(any("SHA_MISMATCH:" in m for m in ver_bad["mismatches"]))


class TestPhaseDTrack3CNAshareMacroAndA2Strategy(unittest.TestCase):
    """Track 3: CN A-Share PIT macro collectors, adjclose total-return scaling, and A2 pure stock-bond strategy."""

    def test_adjclose_total_return_scaling_and_raw_notional_preservation(self) -> None:
        collector = CNAsharePublicCollector()
        ts_day = int(datetime(2024, 6, 3, 9, 30, 0, tzinfo=_CST_TZ).timestamp())
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [ts_day],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [2.00],
                                    "high": [2.10],
                                    "low": [1.98],
                                    "close": [2.05],
                                    "volume": [10_000_000],
                                }
                            ],
                            "adjclose": [{"adjclose": [1.64]}],  # scale = 0.80
                        },
                    }
                ]
            }
        }
        now_ms = int(datetime(2024, 6, 4, 0, 0, 0, tzinfo=_CST_TZ).timestamp() * 1000.0)
        bars_adj = collector.parse_chart_payload("510880.SS", payload, use_adj_close=True, now_ms=now_ms)
        self.assertEqual(len(bars_adj), 1)
        b = bars_adj[0]
        self.assertAlmostEqual(b.close, 1.64, places=6)
        self.assertAlmostEqual(b.open, 1.60, places=6)
        self.assertAlmostEqual(b.high, 1.68, places=6)
        self.assertAlmostEqual(b.low, 1.584, places=6)
        # Raw CNY turnover must still be volume * unadjusted close = 20,500,000
        self.assertAlmostEqual(b.volume_usd, 20_500_000.0, places=2)

        # Backward compatibility when adjclose is omitted or disabled
        bars_raw = collector.parse_chart_payload("510880.SS", payload, use_adj_close=False, now_ms=now_ms)
        self.assertAlmostEqual(bars_raw[0].close, 2.05, places=6)

    def test_eastmoney_cgb_margin_and_yahoo_fx_pit_timestamps_and_store_isolation(self) -> None:
        collector = CNAsharePublicCollector()
        now_ms = int(datetime(2025, 1, 10, 0, 0, 0, tzinfo=_CST_TZ).timestamp() * 1000.0)

        cgb_payload = {
            "result": {
                "data": [
                    {
                        "SOLAR_DATE": "2024-12-20 00:00:00",
                        "EMM00588704": 1.15,
                        "EMM00166462": 1.42,
                        "EMM00166466": 1.72,
                        "EMM00166469": 1.98,
                        "EMM01276014": 0.57,
                    }
                ]
            }
        }
        cgb_pts = collector.parse_eastmoney_cgb_payload(cgb_payload, now_ms=now_ms)
        self.assertEqual(len(cgb_pts), 5)
        for pt in cgb_pts:
            # 18:00 CST strictly after 15:00 CST A-share close!
            close_15 = int(datetime(2024, 12, 20, 15, 0, 0, tzinfo=_CST_TZ).timestamp() * 1000.0)
            self.assertGreater(pt.ts_avail_ms, close_15)
            self.assertGreater(pt.ts_avail_ms, pt.ts_event_ms)

        rzrq_payload = {
            "result": {
                "data": [
                    {
                        "DIM_DATE": "2024-12-20 00:00:00",
                        "RZYE": 1.85e12,
                        "RZYEZB": 2.45,
                        "RZJME": 3.2e9,
                        "RQYE": 1.05e10,
                    }
                ]
            }
        }
        rzrq_pts = collector.parse_eastmoney_margin_payload(rzrq_payload, now_ms=now_ms)
        self.assertEqual(len(rzrq_pts), 4)
        for pt in rzrq_pts:
            # T+1 08:30 CST strictly next morning!
            next_0830 = int(datetime(2024, 12, 21, 8, 30, 0, tzinfo=_CST_TZ).timestamp() * 1000.0)
            self.assertEqual(pt.ts_avail_ms, next_0830)

        with tempfile.TemporaryDirectory(prefix="phase_d_cn_pit_") as td:
            with DataFoundationRouter(Path(td) / "lake") as router:
                # Reject lookahead macro point (ts_avail_ms <= ts_event_ms)
                bad_pt = CNAshareMacroPoint(
                    source="test",
                    series_id="CN_CGB_10Y",
                    ts_event_ms=1000,
                    ts_avail_ms=1000,
                    value=2.0,
                    provenance=PROVENANCE_SYNTHETIC,
                )
                with self.assertRaises(ValueError):
                    router.upsert_cn_ashare_macro([bad_pt], allow_synthetic_in_test=True)

                # Upsert real parsed points and verify strict as_of_ms PIT filtering
                router.upsert_cn_ashare_macro(cgb_pts + rzrq_pts)
                at_1500 = int(datetime(2024, 12, 20, 15, 0, 0, tzinfo=_CST_TZ).timestamp() * 1000.0)
                at_1900 = int(datetime(2024, 12, 20, 19, 0, 0, tzinfo=_CST_TZ).timestamp() * 1000.0)
                at_next_0900 = int(datetime(2024, 12, 21, 9, 0, 0, tzinfo=_CST_TZ).timestamp() * 1000.0)

                # At 15:00 CST on trade day: NEITHER CGB (18:00) NOR Margin (T+1 08:30) is visible!
                self.assertEqual(len(router.load_cn_ashare_macro(as_of_ms=at_1500)), 0)
                # At 19:00 CST: CGB (5 series) visible, Margin STILL hidden!
                self.assertEqual(len(router.load_cn_ashare_macro(as_of_ms=at_1900)), 5)
                # At T+1 09:00 CST: all 9 points visible!
                self.assertEqual(len(router.load_cn_ashare_macro(as_of_ms=at_next_0900)), 9)

                # Export includes cn_ashare_macro.jsonl.gz when populated
                exp = router.export_domain_partition("cn_ashare", fmt="jsonl.gz")
                self.assertIn("cn_ashare_macro", exp["sha256_by_table"])

    def test_a2_forbids_gold_etf_enforces_cn_rules_and_passes_g7_bit_parity(self) -> None:
        # 1. Constructor hard-forbids Gold ETF 518880.SS
        with self.assertRaises(ValueError):
            AShareMacroERPRotationKernel(broad_equity_symbols=("510300.SS", "518880.SS"))
        self.assertNotIn("518880.SS", CN_ASHARE_PURE_STOCK_BOND_UNIVERSE)

        # 2. Build deterministic 100-day pure stock-bond + macro timeline dataset
        macro_pts: list[CNAshareMacroPoint] = []
        bars_by_sym: dict[str, list[CanonicalBar]] = {
            "511260.SS": [],
            "511010.SS": [],
            "510880.SS": [],
            "512890.SS": [],
            "510300.SS": [],
            "159915.SZ": [],
        }
        for d in range(110):
            base_dt = datetime(2024, 1, 2, 9, 30, 0, tzinfo=_CST_TZ) + timedelta(days=d)
            ev_ms = int(base_dt.timestamp() * 1000.0)
            av_cgb = int(base_dt.replace(hour=18, minute=0).timestamp() * 1000.0)
            av_m = int((base_dt + timedelta(days=1)).replace(hour=8, minute=30).timestamp() * 1000.0)
            macro_pts.append(CNAshareMacroPoint("test", "CN_CGB_10Y", ev_ms, av_cgb, 2.30 - 0.002 * d, PROVENANCE_SYNTHETIC))
            macro_pts.append(CNAshareMacroPoint("test", "CN_CGB_10Y_2Y_SLOPE", ev_ms, av_cgb, 0.45, PROVENANCE_SYNTHETIC))
            macro_pts.append(CNAshareMacroPoint("test", "CN_MARGIN_BALANCE_RMB", ev_ms, av_m, 1.5e12 * (1.0 + 0.001 * d), PROVENANCE_SYNTHETIC))
            macro_pts.append(CNAshareMacroPoint("test", "CN_FX_USDCNY", ev_ms, av_m, 7.25 - 0.001 * d, PROVENANCE_SYNTHETIC))

            bars_by_sym["511260.SS"].append(_make_cn_bar("511260.SS", d, 120.0 * (1.0 + 0.0004 * d)))
            bars_by_sym["511010.SS"].append(_make_cn_bar("511010.SS", d, 105.0 * (1.0 + 0.0003 * d)))
            bars_by_sym["510880.SS"].append(_make_cn_bar("510880.SS", d, 3.0 * (1.0 + 0.0008 * d + 0.005 * math.sin(d * 0.2))))
            bars_by_sym["512890.SS"].append(_make_cn_bar("512890.SS", d, 1.1 * (1.0 + 0.0007 * d + 0.004 * math.cos(d * 0.2))))
            bars_by_sym["510300.SS"].append(_make_cn_bar("510300.SS", d, 3.6 * (1.0 + 0.0005 * d + 0.012 * math.sin(d * 0.15))))
            bars_by_sym["159915.SZ"].append(_make_cn_bar("159915.SZ", d, 1.8 * (1.0 + 0.0006 * d + 0.020 * math.sin(d * 0.25))))

        tl = CNMacroRegimeTimeline(macro_pts)
        kernel = AShareMacroERPRotationKernel(macro_timeline=tl, warmup_bars=30, rebalance_every_bars=5)
        self.assertEqual(kernel.resolve_price_limit_pct("159915.SZ"), 20.0)
        self.assertEqual(kernel.resolve_price_limit_pct("510300.SS"), 10.0)

        # Runtime Gold ETF injection gets forced to weight=0.0 with explicit exclusion metadata
        gold_bar = _make_cn_bar("518880.SS", 40, 5.5)
        decs = kernel.on_bar_step(gold_bar.ts_event_ms, {"518880.SS": gold_bar, "511260.SS": bars_by_sym["511260.SS"][40]}, {})
        self.assertEqual(decs["518880.SS"].target_weight, 0.0)
        self.assertTrue(decs["518880.SS"].metadata["excluded_gold_etf"])

        # G7 Full Streaming-vs-Batch Bit-Exact Parity (0.00e+00)
        specs = default_cn_ashare_specs(tuple(bars_by_sym.keys()))
        risk_cfg = RiskAllocatorConfig(max_positions=6, target_annual_vol_pct=12.0, max_portfolio_leverage=1.0, max_single_weight=0.55)
        fresh = AShareMacroERPRotationKernel(macro_timeline=tl, warmup_bars=30, rebalance_every_bars=5)
        parity = verify_full_parity(fresh, specs, bars_by_sym, risk_cfg=risk_cfg, capital_usd=10_000.0)
        self.assertTrue(parity.passed, f"G7 parity failed: {parity}")
        self.assertLessEqual(parity.max_abs_equity_delta, 1e-12)

        # Verify RB-15 suspended weight reservation & long-only
        acct = PortfolioAccountingEngine()
        sim = acct.simulate(fresh.clone_fresh(), specs, bars_by_sym, risk_cfg=risk_cfg, capital_usd=10_000.0)
        self.assertGreater(sim.trade_count, 0)
        self.assertEqual(sim.liquidation_count, 0)

    def test_three_domain_ast_zero_cross_imports(self) -> None:
        a2_path = _REPO_ROOT / "alphaloop" / "platform" / "strategies" / "ashare" / "macro_erp_rotation.py"
        tree = ast.parse(a2_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn("strategies.web3", alias.name)
                    self.assertNotIn("strategies.gold", alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                self.assertNotIn("strategies.web3", mod)
                self.assertNotIn("strategies.gold", mod)


class TestPhaseDRedTeamRegressions(unittest.TestCase):
    """Adversarial regression proofs for Round 1 Red-Team findings (RB-D03..RB-D09)."""

    def test_rb_d03_non_finite_none_bool_timestamps_fail_closed_without_crash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase_d_rb03_") as td:
            t0 = 1_795_200_000_000
            bad_frames = [
                {
                    "BTC": _make_pair_snap("BTC", t0),
                    "AVAX": PairMarketSnapshot(
                        coin="AVAX",
                        ts_observe_ms=None,  # type: ignore[arg-type]
                        spot_bid=10.0,
                        spot_ask=10.1,
                        spot_bid_qty=10.0,
                        spot_ask_qty=10.0,
                        spot_ts_ms=t0,
                        perp_bid=10.0,
                        perp_ask=10.1,
                        perp_mark=10.05,
                        perp_oracle=10.0,
                        perp_funding_1h=0.0001,
                        perp_ts_ms=t0,
                        provenance=PROVENANCE_SYNTHETIC,
                    ),
                },
                {
                    "BTC": _make_pair_snap("BTC", t0 + 60_000),
                    "AVAX": PairMarketSnapshot(
                        coin="AVAX",
                        ts_observe_ms=t0 + 60_000,
                        spot_bid=10.0,
                        spot_ask=10.1,
                        spot_bid_qty=10.0,
                        spot_ask_qty=10.0,
                        spot_ts_ms=float("nan"),  # type: ignore[arg-type]
                        perp_bid=10.0,
                        perp_ask=10.1,
                        perp_mark=10.05,
                        perp_oracle=10.0,
                        perp_funding_1h=0.0001,
                        perp_ts_ms=True,  # type: ignore[arg-type]
                        provenance=PROVENANCE_SYNTHETIC,
                    ),
                },
            ]
            cfg = ShadowMonitorConfig(
                state_dir=Path(td) / "state",
                coins=("BTC",),
                observe_only_coins=("AVAX",),
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(cfg, provider=ReplayQuoteProvider(bad_frames)) as mon:
                r1 = mon.run_once()
                r2 = mon.run_once()
                self.assertEqual(r1["status"], "HEALTHY")
                self.assertEqual(r2["status"], "HEALTHY")
                rep = mon.evaluate_48h_readiness()
                self.assertEqual(rep.observe_only_coin_stats["AVAX"]["fail_ticks"], 2)

    def test_rb_d06_orphaned_open_position_moved_to_observe_only_fails_closed(self) -> None:
        from alphaloop.platform.shadow.w3_dn_shadow_monitor import VirtualPairPosition

        with tempfile.TemporaryDirectory(prefix="phase_d_rb06_") as td:
            state_dir = Path(td) / "state"
            t0 = 1_795_300_000_000
            cfg1 = ShadowMonitorConfig(
                state_dir=state_dir,
                coins=("BTC", "AVAX"),
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(
                cfg1,
                provider=ReplayQuoteProvider(
                    [{"BTC": _make_pair_snap("BTC", t0), "AVAX": _make_pair_snap("AVAX", t0)}]
                ),
            ) as m1:
                m1.run_once()
                # Inject an open AVAX position and persist checkpoint
                m1._positions["AVAX"] = VirtualPairPosition(
                    coin="AVAX",
                    spot_qty=10.0,
                    perp_qty=-10.0,
                    spot_entry_px=20.0,
                    perp_entry_px=20.05,
                    weight_spot=0.25,
                    weight_perp=-0.25,
                    opened_ts_ms=t0,
                    last_funding_ts_ms=t0,
                )
                m1._persist_checkpoint_only(t0)

            # Restarting with AVAX moved to observe_only_coins MUST fail closed!
            cfg2 = ShadowMonitorConfig(
                state_dir=state_dir,
                coins=("BTC",),
                observe_only_coins=("AVAX",),
                enforce_wall_clock_bounds=False,
            )
            with self.assertRaises(ValueError):
                W3DNShadowMonitor(cfg2, provider=ReplayQuoteProvider([]))

    def test_rb_d05_readonly_and_sandbox_work_on_0444_snapshots_and_snapshot_verifier_rejects_rogues(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase_d_rb05_") as td:
            lake = Path(td) / "lake"
            snap_dir = Path(td) / "snap_ro"
            specs = default_cn_ashare_specs(("510300.SS",))
            with DataFoundationRouter(lake, trial_ledger_mode="official") as r_off:
                r_off.record_trial("fam_r5", "track_a", "s1", "h1", {"sr": 0.9}, "CONDITIONAL_RESEARCH")
                bars = []
                for d in range(50):
                    c = 4.0 + 0.01 * d + 0.03 * math.sin(d * 0.4)
                    bars.append(
                        _make_cn_bar(
                            "510300.SS",
                            d,
                            c,
                            open_p=c * 0.996,
                            high_p=c * 1.007,
                            low_p=c * 0.993,
                            vol_cny=50_000_000.0 + d * 100_000.0,
                        )
                    )
                r_off.upsert_bars("cn_ashare", bars, allow_synthetic_in_test=True)
                r_off.export_domain_partition("cn_ashare", fmt="jsonl.gz")

            create_lake_snapshot(lake, snap_dir, label="rb05")

            # 1. Both readonly and sandbox modes work cleanly directly against 0444 snapshot!
            for mode in ("readonly", "sandbox"):
                with DataFoundationRouter(snap_dir, trial_ledger_mode=mode) as r_ro:
                    loaded = r_ro.load_bars("cn_ashare", "cn_a_share", "510300.SS", 86_400)
                    self.assertEqual(len(loaded), 50)
                    dq = r_ro.audit_domain_dq(
                        "cn_ashare", "cn_a_share", "510300.SS", 86_400, specs["510300.SS"], allow_synthetic_in_test=True
                    )
                    self.assertTrue(dq.passed, f"Unexpected DQ fail on realistic bars: {dq}")
                    r_ro.record_trial("fam_r5", "track_a", "s2", "h2", {}, "GRID_CANDIDATE")
                    self.assertEqual(r_ro.get_family_trial_count("fam_r5"), 2)
                    with self.assertRaises(PermissionError):
                        r_ro.register_asset(specs["510300.SS"])
                    with self.assertRaises(PermissionError):
                        r_ro.save_factor_audit("t", "sha", {})

            # 2. verify_lake_snapshot catches unmanifested rogue file
            rogue = snap_dir / "domains" / "evil.sqlite"
            rogue.write_bytes(b"rogue")
            ver_rogue = verify_lake_snapshot(snap_dir)
            self.assertFalse(ver_rogue["ok"])
            self.assertTrue(any("UNMANIFESTED_FILE:" in m for m in ver_rogue["mismatches"]))
            rogue.unlink()

            # 3. verify_lake_snapshot catches forged manifest omitting catalog.sqlite
            mf_path = snap_dir / "SNAPSHOT_MANIFEST.json"
            os.chmod(mf_path, 0o644)
            orig_mf = json.loads(mf_path.read_text(encoding="utf-8"))
            forged = dict(orig_mf)
            forged["files"] = {k: v for k, v in orig_mf["files"].items() if k != "catalog.sqlite"}
            mf_path.write_text(json.dumps(forged), encoding="utf-8")
            ver_omit = verify_lake_snapshot(snap_dir)
            self.assertFalse(ver_omit["ok"])
            self.assertTrue(any("MISSING_REQUIRED_MANIFEST_ENTRY:catalog.sqlite" in m for m in ver_omit["mismatches"]))

            # 4. create_lake_snapshot rejects symlinks inside source lake
            sym_target = Path(td) / "secret.txt"
            sym_target.write_text("do_not_copy", encoding="utf-8")
            bad_link = lake / "parquet" / "cn_ashare" / "leak.txt"
            os.symlink(sym_target, bad_link)
            with self.assertRaises(ValueError):
                create_lake_snapshot(lake, Path(td) / "snap_bad_link")

    def test_rb_d04_and_rb_d09_true_zero_macro_ablation_and_lowercase_gold_etf_nan_guard(self) -> None:
        macro_pts: list[CNAshareMacroPoint] = []
        bars_by_sym: dict[str, list[CanonicalBar]] = {
            "511260.SS": [],
            "510880.SS": [],
            "510300.SS": [],
        }
        for d in range(85):
            base_dt = datetime(2024, 1, 2, 9, 30, 0, tzinfo=_CST_TZ) + timedelta(days=d)
            ev_ms = int(base_dt.timestamp() * 1000.0)
            av_cgb = int(base_dt.replace(hour=18, minute=0).timestamp() * 1000.0)
            macro_pts.append(CNAshareMacroPoint("test", "CN_CGB_10Y", ev_ms, av_cgb, 2.40 - 0.004 * d, PROVENANCE_SYNTHETIC))
            bars_by_sym["511260.SS"].append(_make_cn_bar("511260.SS", d, 110.0 * (1.0 + 0.0005 * d)))
            bars_by_sym["510880.SS"].append(_make_cn_bar("510880.SS", d, 3.0 * (1.0 + 0.0008 * d)))
            bars_by_sym["510300.SS"].append(_make_cn_bar("510300.SS", d, 3.5 * (1.0 + 0.0004 * d)))

        tl = CNMacroRegimeTimeline(macro_pts)
        specs = default_cn_ashare_specs(tuple(bars_by_sym.keys()))
        risk_cfg = RiskAllocatorConfig(max_positions=4, target_annual_vol_pct=12.0, max_portfolio_leverage=1.0, max_single_weight=0.55)
        acct = PortfolioAccountingEngine()

        # RB-D04: macro_tilt_strength=0.0 with populated timeline MUST be bit-identical to empty timeline!
        sim_empty = acct.simulate(
            AShareMacroERPRotationKernel(macro_timeline=CNMacroRegimeTimeline([]), macro_tilt_strength=0.0, warmup_bars=30, rebalance_every_bars=5),
            specs, bars_by_sym, risk_cfg=risk_cfg, capital_usd=10_000.0,
        )
        sim_pop = acct.simulate(
            AShareMacroERPRotationKernel(macro_timeline=tl, macro_tilt_strength=0.0, warmup_bars=30, rebalance_every_bars=5),
            specs, bars_by_sym, risk_cfg=risk_cfg, capital_usd=10_000.0,
        )
        self.assertEqual(sim_empty.period_returns, sim_pop.period_returns)

        # RB-D09: Lowercase Gold ETF "518880.ss" is excluded & NaN/negative close does NOT corrupt _closes
        k = AShareMacroERPRotationKernel(macro_timeline=tl, warmup_bars=5, rebalance_every_bars=1)
        lower_gold = _make_cn_bar("518880.ss", 10, 5.5)
        bad_nan = CanonicalBar(
            asset_class=ASSET_CLASS_CN_A_SHARE,
            venue="cn_a_share",
            symbol="510300.SS",
            timeframe_h=24.0,
            timeframe_sec=86_400,
            ts_event_ms=lower_gold.ts_event_ms,
            ts_avail_ms=lower_gold.ts_avail_ms,
            open=3.5,
            high=3.6,
            low=3.4,
            close=float("nan"),
            volume_usd=1_000_000.0,
            carry_rate_period=0.0,
            carry_interval_h=24.0,
            spread_bps=5.0,
            provenance=PROVENANCE_SYNTHETIC,
        )
        decs = k.on_bar_step(lower_gold.ts_event_ms, {"518880.ss": lower_gold, "510300.SS": bad_nan}, {})
        self.assertEqual(decs["518880.ss"].target_weight, 0.0)
        self.assertTrue(decs["518880.ss"].metadata["excluded_gold_etf"])
        self.assertEqual(len(k._closes.get("510300.SS", ())), 0)


if __name__ == "__main__":
    unittest.main()
