"""Comprehensive Offline + Real-Lake Integrity Unit Tests for Platform v4 Data Foundation (M1).

Covers:
- D10: Complete bucket resampling (4,800 aligned -> 400; 4,800 11:00Z-offset -> 399; TradFi 1h->4h min_subbars=2; :30 causal ts_avail_ms)
- D11: DQ-MA-07 (zero-funding fail, 58% funding fail, 100x unit bug fail, >=98% pass, non-perp N/A), tightened 24/7 gap budget (40h on 12h fails), carry_interval_h preservation
- D14: EXPLAIN QUERY PLAN verifying exact integer timeframe_sec index on BOTH DataFoundationRouter AND MultiAssetDataLake (including legacy v3 schema auto-upgrade) + zero same-ms IntegrityError on audit/snapshot/blueprint/trial upserts
- D15d: AssetSpec contract_multiplier (GC=F=100.0, SI=F=5000.0) and 13-symbol GOLD_MACRO_SYMBOLS_V4
- PIT lookahead & fail-closed validation on CanonicalBar and domain records + macro event lookahead redaction
- Physical cross-domain isolation (web3_lake.sqlite vs gold_macro_lake.sqlite) + DomainStoreProtocol mock A-Share plugin
- REA-001 TrialLedger append-only enforcement via SQLite BEFORE UPDATE / BEFORE DELETE triggers
- Deterministic partition export (mtime=0 jsonl.gz bit-identical SHA-256) + optional RO DuckDB SQL guard
- Idempotent read-only v3 migration leaving source SQLite SHA-256 unchanged
- Offline wire-fixture tests for ALL Web3 & Gold/Macro collectors
- Read-only integrity verification of ingested data/platform_v4_lake (runs automatically when lake exists)
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import sqlite3
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence
from unittest.mock import patch

from alphaloop.core.provenance import PROVENANCE_REAL, PROVENANCE_SYNTHETIC, sha256_file
from alphaloop.platform.assets import (
    ASSET_CLASS_CN_A_SHARE,
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_CRYPTO_SPOT,
    ASSET_CLASS_GOLD_PERP,
    ASSET_CLASS_GOLD_TRADFI,
    ASSET_CLASS_MACRO_INDEX,
    CALENDAR_CRYPTO_24_7,
    CALENDAR_GOLD_TRADFI_23_5,
    COST_BINANCE_PERP,
    COST_BINANCE_SPOT,
    COST_OKX_PERP,
    GOLD_MACRO_SYMBOLS,
    GOLD_MACRO_SYMBOLS_V4,
    AssetSpec,
    default_ashare_template_spec,
    default_gold_specs,
    default_macro_specs,
    default_macro_specs_v4,
)
from alphaloop.platform.data_foundation import (
    BLSMacroEventCollector,
    BinanceVisionWeb3Collector,
    CanonicalBar,
    DAY_MS,
    DataFoundationRouter,
    DefiLlamaStablecoinCollector,
    DeribitDVOLCollector,
    DomainStoreProtocol,
    FedFOMCCalendarCollector,
    GateWeb3Collector,
    HOUR_MS,
    HyperliquidWeb3Collector,
    MacroEventRelease,
    MacroYieldPoint,
    OKXWeb3Collector,
    PAXGBasisPoint,
    RateLimitedHTTPClient,
    USTreasuryRealYieldCollector,
    Web3DerivativesMetric,
    Web3MarketMacroPoint,
    YahooGoldMacroCollector,
    audit_bars_quality,
    bls_cpi_release_date,
    bls_nfp_release_date,
    compute_paxg_basis_points,
    is_comex_weekend_utc,
    normalize_binance_ts_ms,
    ny_local_to_utc_ms,
    parse_zip_csv_rows,
    resample_bars,
    snap_to_nearest_hour_ms,
    validate_canonical_bar,
    validate_macro_event_release,
    validate_macro_yield_point,
    validate_paxg_basis_point,
    validate_readonly_sql,
    validate_web3_derivatives_metric,
    validate_web3_market_macro_point,
)
from alphaloop.platform.data_foundation import router as _router_mod
from alphaloop.platform.data_lake import MultiAssetDataLake, resample_canonical_bars

BASE_2025_MS = 1_735_689_600_000  # 2025-01-01T00:00:00Z
REAL_V4_LAKE_DIR = Path(__file__).resolve().parent.parent / "data" / "platform_v4_lake"


def _make_hourly_bars(
    count: int,
    *,
    start_ms: int = BASE_2025_MS,
    venue: str = "hyperliquid",
    symbol: str = "BTC",
    asset_class: str = ASSET_CLASS_CRYPTO_PERP,
    carry_rate: float = 0.0000125,
    carry_interval_h: float = 1.0,
    tf_h: float = 1.0,
    provenance: str = PROVENANCE_REAL,
) -> list[CanonicalBar]:
    step_ms = int(round(tf_h * HOUR_MS))
    out: list[CanonicalBar] = []
    px = 100.0
    for i in range(count):
        ts = start_ms + i * step_ms
        c = px * (1.0 + 0.0002 * math.sin(i / 5.0))
        o = px
        h = max(o, c) * 1.001
        l = min(o, c) * 0.999
        out.append(
            CanonicalBar(
                asset_class=asset_class,
                venue=venue,
                symbol=symbol,
                timeframe_h=tf_h,
                ts_event_ms=ts,
                ts_avail_ms=ts + step_ms,
                open=o,
                high=h,
                low=l,
                close=c,
                volume_usd=500_000.0,
                carry_rate_period=carry_rate,
                carry_interval_h=carry_interval_h,
                spread_bps=1.2,
                provenance=provenance,
                premium_basis_bps=2.5,
            )
        )
        px = c
    return out


def _zip_csv_bytes(filename: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, text.encode("utf-8"))
    return buf.getvalue()


class _MockAShareStore:
    """In-memory mock DomainStoreProtocol plugin verifying clean CN A-share pluggability."""

    def __init__(self, db_path: Path) -> None:
        self.domain_name = "cn_ashare"
        self.db_path = db_path
        self._bars: dict[tuple[str, str, int, int], CanonicalBar] = {}

    def upsert_bars(self, bars: Sequence[CanonicalBar], *, allow_synthetic_in_test: bool = False) -> int:
        for b in bars:
            validate_canonical_bar(b, allow_synthetic_in_test=allow_synthetic_in_test)
            self._bars[(b.venue, b.symbol, b.timeframe_sec, b.ts_event_ms)] = b
        return len(bars)

    def load_bars(
        self,
        venue: str,
        symbol: str,
        timeframe_sec: int,
        *,
        as_of_ms: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[CanonicalBar]:
        out = [
            b
            for b in self._bars.values()
            if b.venue == venue
            and b.symbol == symbol
            and b.timeframe_sec == timeframe_sec
            and (as_of_ms is None or b.ts_avail_ms <= as_of_ms)
            and (start_ms is None or b.ts_event_ms >= start_ms)
            and (end_ms is None or b.ts_event_ms <= end_ms)
        ]
        return sorted(out, key=lambda x: x.ts_event_ms)

    def list_series(self) -> list[dict[str, Any]]:
        return [{"domain": self.domain_name, "count": len(self._bars)}]

    def iter_export_rows(self) -> Iterator[dict[str, Any]]:
        for b in sorted(self._bars.values(), key=lambda x: x.ts_event_ms):
            yield b.to_dict()

    def close(self) -> None:
        return None


class TestPlatformV4DataFoundation(unittest.TestCase):
    # --- D15d: AssetSpec contract multipliers & V4 macro universe ---
    def test_d15d_asset_specs_and_contract_multipliers(self) -> None:
        self.assertEqual(len(GOLD_MACRO_SYMBOLS), 8)
        self.assertEqual(len(GOLD_MACRO_SYMBOLS_V4), 13)
        for req in ("^GVZ", "GDX", "SLV", "UUP", "TLT"):
            self.assertIn(req, GOLD_MACRO_SYMBOLS_V4)

        gold_specs = default_gold_specs()
        self.assertEqual(gold_specs["GC=F"].contract_multiplier, 100.0)
        self.assertEqual(gold_specs["GLD"].contract_multiplier, 1.0)
        self.assertEqual(gold_specs["PAXG"].contract_multiplier, 1.0)

        m_v3 = default_macro_specs()
        self.assertEqual(len(m_v3), 8)
        self.assertEqual(m_v3["GC=F"].contract_multiplier, 100.0)
        self.assertEqual(m_v3["SI=F"].contract_multiplier, 5000.0)

        m_v4 = default_macro_specs_v4()
        self.assertEqual(len(m_v4), 13)
        self.assertEqual(m_v4["GC=F"].contract_multiplier, 100.0)
        self.assertEqual(m_v4["SI=F"].contract_multiplier, 5000.0)
        self.assertEqual(m_v4["^GVZ"].contract_multiplier, 1.0)

        self.assertGreater(COST_BINANCE_PERP.round_trip_cost_bps("TAKER_TAKER"), 0.0)
        self.assertGreater(COST_BINANCE_SPOT.round_trip_cost_bps("TAKER_TAKER"), 0.0)
        self.assertGreater(COST_OKX_PERP.round_trip_cost_bps("TAKER_TAKER"), 0.0)

        with self.assertRaises(ValueError):
            AssetSpec(
                "GC=F",
                ASSET_CLASS_GOLD_TRADFI,
                "yahoo",
                CALENDAR_GOLD_TRADFI_23_5,
                COST_BINANCE_PERP,
                contract_multiplier=-10.0,
            )

    # --- D10 & D11: Resampling & DQ-MA-07 & Gap Budget ---
    def test_d10_complete_resampling_aligned_offset_and_tradfi_causal(self) -> None:
        # 4,800 aligned 1h bars -> exact 400 12h bars
        aligned_1h = _make_hourly_bars(4_800, start_ms=BASE_2025_MS)
        res_aligned = resample_bars(aligned_1h, 12.0, require_complete=True)
        self.assertEqual(len(res_aligned), 400)
        self.assertEqual(res_aligned[0].timeframe_sec, 43_200)
        self.assertAlmostEqual(res_aligned[0].carry_interval_h, 12.0, places=6)

        # 4,800 offset 11:00Z-start 1h bars -> exact 399 complete 12h bars (drops 1h lead & 11h trail!)
        offset_1h = _make_hourly_bars(4_800, start_ms=BASE_2025_MS + 11 * HOUR_MS)
        res_offset = resample_canonical_bars(offset_1h, 12.0)
        self.assertEqual(len(res_offset), 399)
        self.assertEqual(res_offset[0].ts_event_ms, BASE_2025_MS + 12 * HOUR_MS)

        # D11 carry_interval_h preservation: Gate 1h (carry_interval_h=8.0) -> 4h MUST keep 8.0!
        gate_1h = _make_hourly_bars(48, venue="gate", carry_interval_h=8.0)
        gate_4h = resample_bars(gate_1h, 4.0, require_complete=True)
        self.assertEqual(len(gate_4h), 12)
        self.assertAlmostEqual(gate_4h[0].carry_interval_h, 8.0, places=6)

        # TradFi 1h -> 4h with :30 NYSE bars: min_subbars=2 drops 1-bar tail and keeps causal ts_avail_ms!
        # Bucket 12:00..16:00 has 13:30 (avail 14:30), 14:30 (avail 15:30), 15:30 (avail 16:30!)
        # Bucket 20:00..24:00 has only 20:30 (1 bar -> dropped by min_subbars=2!)
        nyse_1h = [
            CanonicalBar(
                asset_class=ASSET_CLASS_GOLD_TRADFI,
                venue="yahoo",
                symbol="GLD",
                timeframe_h=1.0,
                ts_event_ms=BASE_2025_MS + 13 * HOUR_MS + 30 * 60_000 + i * HOUR_MS,
                ts_avail_ms=BASE_2025_MS + 14 * HOUR_MS + 30 * 60_000 + i * HOUR_MS,
                open=250.0,
                high=251.0,
                low=249.0,
                close=250.5,
                volume_usd=1_000_000.0,
                carry_interval_h=1.0,
            )
            for i in range(8)  # 13:30..20:30 UTC (3 in 12:00, 4 in 16:00, 1 in 20:00)
        ]
        gld_4h = resample_bars(nyse_1h, 4.0, require_complete=True, min_subbars=2)
        self.assertEqual(len(gld_4h), 2)
        # Critical causal check: 12:00 bucket contains 15:30 bar (avail 16:30Z), so ts_avail_ms MUST be 16:30Z!
        self.assertEqual(gld_4h[0].ts_event_ms, BASE_2025_MS + 12 * HOUR_MS)
        self.assertEqual(gld_4h[0].ts_avail_ms, BASE_2025_MS + 16 * HOUR_MS + 30 * 60_000)
        self.assertEqual(gld_4h[0].session_tag, "US_EQUITY_RTH")

    def test_d11_dq_ma_07_and_tightened_24_7_gap_budget(self) -> None:
        # 1. Zero-funding perp MUST fail DQ-MA-07
        zero_f = _make_hourly_bars(100, tf_h=12.0, carry_rate=0.0, carry_interval_h=12.0)
        rep_zero = audit_bars_quality(zero_f, CALENDAR_CRYPTO_24_7)
        self.assertFalse(rep_zero.passed)
        self.assertEqual(rep_zero.checks[0].gate_id, "DQ-MA-01")
        self.assertEqual(rep_zero.checks[-1].gate_id, "DQ-MA-07")
        self.assertFalse(rep_zero.checks[-1].passed)

        # 2. 58% funding coverage perp MUST fail DQ-MA-07
        partial = _make_hourly_bars(100, tf_h=12.0, carry_rate=0.0001, carry_interval_h=12.0)
        partial_58 = [
            CanonicalBar(**{**b.to_dict(), "carry_rate_period": 0.0}) if i >= 58 else b
            for i, b in enumerate(partial)
        ]
        rep_58 = audit_bars_quality(partial_58, CALENDAR_CRYPTO_24_7)
        self.assertFalse(rep_58.checks[-1].passed)

        # 3. 100x funding unit bug (e.g. 0.08 per 8h -> >150% median APR) MUST fail DQ-MA-07
        unit_bug = _make_hourly_bars(100, tf_h=12.0, carry_rate=0.08, carry_interval_h=12.0)
        rep_bug = audit_bars_quality(unit_bug, CALENDAR_CRYPTO_24_7)
        self.assertFalse(rep_bug.checks[-1].passed)

        # 4. Healthy >=98% perp PASSES DQ-MA-01..07
        healthy = _make_hourly_bars(100, tf_h=12.0, carry_rate=0.00012, carry_interval_h=12.0)
        rep_ok = audit_bars_quality(healthy, CALENDAR_CRYPTO_24_7)
        self.assertTrue(rep_ok.passed, msg=[c for c in rep_ok.checks if not c.passed])

        # 5. Non-perp passes DQ-MA-07 as N/A
        spot = _make_hourly_bars(100, asset_class=ASSET_CLASS_CRYPTO_SPOT, tf_h=12.0, carry_rate=0.0, carry_interval_h=12.0)
        rep_spot = audit_bars_quality(spot, CALENDAR_CRYPTO_24_7)
        self.assertTrue(rep_spot.passed)
        self.assertIn("N/A", rep_spot.checks[-1].value)

        # 6. Tightened 24/7 gap budget: 48h gap (missing 3 consecutive 12h bars -> gap=48h > 36h) MUST fail DQ-MA-04
        gapped = healthy[:50] + [
            CanonicalBar(
                **{
                    **b.to_dict(),
                    "ts_event_ms": b.ts_event_ms + 36 * HOUR_MS,
                    "ts_avail_ms": b.ts_avail_ms + 36 * HOUR_MS,
                }
            )
            for b in healthy[50:]
        ]
        rep_gap = audit_bars_quality(gapped, CALENDAR_CRYPTO_24_7)
        dq4 = next(c for c in rep_gap.checks if c.gate_id == "DQ-MA-04")
        self.assertFalse(dq4.passed)

    # --- Fail-Closed Validation ---
    def test_validate_canonical_bar_fail_closed(self) -> None:
        good = _make_hourly_bars(1)[0]
        validate_canonical_bar(good)

        # Lookahead ts_avail_ms < ts_event_ms
        with self.assertRaises(ValueError):
            validate_canonical_bar(CanonicalBar(**{**good.to_dict(), "ts_avail_ms": good.ts_event_ms - 1}))
        # NaN close
        with self.assertRaises(ValueError):
            validate_canonical_bar(CanonicalBar(**{**good.to_dict(), "close": float("nan")}))
        # Non-positive price
        with self.assertRaises(ValueError):
            validate_canonical_bar(CanonicalBar(**{**good.to_dict(), "open": -1.0}))
        # OHLC inversion
        with self.assertRaises(ValueError):
            validate_canonical_bar(CanonicalBar(**{**good.to_dict(), "high": good.open * 0.5}))
        # Negative volume
        with self.assertRaises(ValueError):
            validate_canonical_bar(CanonicalBar(**{**good.to_dict(), "volume_usd": -5.0}))
        # Synthetic provenance rejected by default, allowed only with allow_synthetic_in_test=True
        syn = CanonicalBar(**{**good.to_dict(), "provenance": PROVENANCE_SYNTHETIC})
        with self.assertRaises(ValueError):
            validate_canonical_bar(syn, allow_synthetic_in_test=False)
        validate_canonical_bar(syn, allow_synthetic_in_test=True)
        # 16-digit microsecond timestamp rejected
        with self.assertRaises(ValueError):
            validate_canonical_bar(
                CanonicalBar(
                    **{
                        **good.to_dict(),
                        "ts_event_ms": good.ts_event_ms * 1000,
                        "ts_avail_ms": good.ts_avail_ms * 1000,
                    }
                )
            )

    # --- D14: EXPLAIN QUERY PLAN, Legacy v3 Auto-Upgrade, Same-ms Upserts, TrialLedger Triggers ---
    def test_d14_explain_query_plan_and_legacy_upgrade_and_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # 1. Test DataFoundationRouter exact timeframe_sec index & TrialLedger triggers
            with DataFoundationRouter(base / "v4_lake") as router:
                w3_bars = _make_hourly_bars(60, tf_h=4.0, carry_interval_h=4.0)
                g_bars = _make_hourly_bars(
                    60,
                    venue="yahoo",
                    symbol="GC=F",
                    asset_class=ASSET_CLASS_GOLD_TRADFI,
                    tf_h=4.0,
                    carry_rate=0.0,
                    carry_interval_h=4.0,
                )
                self.assertEqual(router.upsert_bars("web3", w3_bars), 60)
                self.assertEqual(router.upsert_bars("gold_macro", g_bars), 60)

                # Verify physically separate SQLite files
                self.assertTrue((base / "v4_lake" / "domains" / "web3_lake.sqlite").exists())
                self.assertTrue((base / "v4_lake" / "domains" / "gold_macro_lake.sqlite").exists())
                self.assertNotEqual(router.web3_store.db_path, router.gold_macro_store.db_path)

                # EXPLAIN QUERY PLAN on web3_bars (with as_of_ms)
                plan_w3 = router.web3_store.conn.fetch_all(
                    "EXPLAIN QUERY PLAN SELECT * FROM web3_bars "
                    "WHERE venue=? AND symbol=? AND timeframe_sec=? AND ts_avail_ms<=? ORDER BY ts_event_ms ASC",
                    ("hyperliquid", "BTC", 14_400, BASE_2025_MS + 100 * HOUR_MS),
                )
                detail_w3 = " | ".join(str(r["detail"]) for r in plan_w3)
                self.assertIn("idx_web3_bars_pit_sec", detail_w3)
                self.assertIn("timeframe_sec=?", detail_w3)
                self.assertNotIn("SCAN web3_bars", detail_w3)

                # EXPLAIN QUERY PLAN on tradfi_bars
                plan_g = router.gold_macro_store.conn.fetch_all(
                    "EXPLAIN QUERY PLAN SELECT * FROM tradfi_bars "
                    "WHERE venue=? AND symbol=? AND timeframe_sec=? AND ts_avail_ms<=? ORDER BY ts_event_ms ASC",
                    ("yahoo", "GC=F", 14_400, BASE_2025_MS + 100 * HOUR_MS),
                )
                detail_g = " | ".join(str(r["detail"]) for r in plan_g)
                self.assertIn("idx_tradfi_bars_pit_sec", detail_g)
                self.assertIn("timeframe_sec=?", detail_g)

                # Rapid same-ms calls to audit/snapshot/blueprint/trial raise ZERO IntegrityError
                aids = {router.save_factor_audit("t", "sha", {"k": i}) for i in range(10)}
                sids = {router.save_regime_snapshot("hyperliquid", "BTC", 12.0, "TREND", {"k": i}) for i in range(10)}
                bids = {router.save_strategy_blueprint("t", "W1", "PASS", "sha", {"k": i}) for i in range(10)}
                tids = {router.record_trial("fam_w1", "W1", "xs_mom", f"p{i}", {"sr": 1.2}, "PASS") for i in range(10)}
                self.assertEqual(len(aids), 10)
                self.assertEqual(len(sids), 10)
                self.assertEqual(len(bids), 10)
                self.assertEqual(len(tids), 10)
                self.assertEqual(router.get_family_trial_count("fam_w1"), 10)

                # REA-001 append-only enforcement: UPDATE and DELETE MUST raise sqlite3.IntegrityError!
                some_tid = next(iter(tids))
                with self.assertRaises(sqlite3.IntegrityError):
                    router.catalog_conn.execute(
                        "UPDATE trial_ledger SET verdict='FAKE' WHERE trial_id=?", (some_tid,)
                    )
                with self.assertRaises(sqlite3.IntegrityError):
                    router.catalog_conn.execute(
                        "DELETE FROM trial_ledger WHERE trial_id=?", (some_tid,)
                    )
                self.assertEqual(router.get_family_trial_count("fam_w1"), 10)

                # Deterministic partition export + bit-identical re-export SHA-256
                exp1 = router.export_domain_partition("web3", fmt="auto")
                exp2 = router.export_domain_partition("web3", fmt="auto")
                self.assertEqual(exp1["sha256_by_table"], exp2["sha256_by_table"])
                self.assertGreaterEqual(len(router.list_dataset_manifests("web3")), 3)

                # Read-only DuckDB SQL guard rejects mutating SQL regardless of duckdb presence
                with self.assertRaises(ValueError):
                    validate_readonly_sql("DROP TABLE web3_bars")
                with self.assertRaises(ValueError):
                    validate_readonly_sql("SELECT 1; DELETE FROM web3_bars")
                with self.assertRaises(ValueError):
                    router.query_domain_readonly_duckdb("web3", "ATTACH 'evil.db' AS x")

                # Pluggable DomainStoreProtocol mock A-share plugin
                ash_store = _MockAShareStore(base / "cn_ashare.mock")
                self.assertIsInstance(ash_store, DomainStoreProtocol)
                router.register_domain_store(ash_store)
                ash_spec = default_ashare_template_spec()
                router.register_asset(ash_spec, domain="cn_ashare")
                ash_bars = _make_hourly_bars(
                    5,
                    venue="cn_a_share",
                    symbol="000300.SH",
                    asset_class=ASSET_CLASS_CN_A_SHARE,
                    tf_h=24.0,
                    carry_rate=0.0,
                    carry_interval_h=24.0,
                )
                self.assertEqual(router.upsert_bars("cn_ashare", ash_bars), 5)
                self.assertEqual(len(router.load_bars("cn_ashare", "cn_a_share", "000300.SH", 86_400)), 5)

            # 2. Test legacy v3 MultiAssetDataLake auto-upgrade & EXPLAIN QUERY PLAN
            legacy_db = base / "legacy_v3.sqlite"
            raw_con = sqlite3.connect(str(legacy_db), timeout=30.0)
            raw_con.execute("PRAGMA journal_mode=WAL;")
            raw_con.execute(
                """
                CREATE TABLE canonical_bars (
                    asset_class TEXT NOT NULL, venue TEXT NOT NULL, symbol TEXT NOT NULL,
                    timeframe_h REAL NOT NULL, ts_event_ms INTEGER NOT NULL, ts_avail_ms INTEGER NOT NULL,
                    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
                    volume_usd REAL NOT NULL, carry_rate_period REAL NOT NULL, carry_interval_h REAL NOT NULL,
                    spread_bps REAL NOT NULL, provenance TEXT NOT NULL,
                    PRIMARY KEY (venue, symbol, timeframe_h, ts_event_ms)
                );
                """
            )
            raw_con.execute(
                "INSERT INTO canonical_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ASSET_CLASS_CRYPTO_PERP,
                    "hyperliquid",
                    "BTC",
                    1.0,
                    BASE_2025_MS,
                    BASE_2025_MS + HOUR_MS,
                    100.0,
                    101.0,
                    99.0,
                    100.5,
                    1e6,
                    0.00001,
                    1.0,
                    0.0,
                    PROVENANCE_REAL,
                ),
            )
            raw_con.commit()
            raw_con.close()

            # Open legacy DB with upgraded MultiAssetDataLake -> must auto-add timeframe_sec & index!
            lake = MultiAssetDataLake(legacy_db)
            try:
                loaded = lake.load_bars("hyperliquid", "BTC", 1.0)
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0].timeframe_sec, 3_600)

                plan_leg = lake.conn.fetch_all(
                    "EXPLAIN QUERY PLAN SELECT * FROM canonical_bars "
                    "WHERE venue=? AND symbol=? AND timeframe_sec=? AND ts_avail_ms<=? ORDER BY ts_event_ms ASC",
                    ("hyperliquid", "BTC", 3_600, BASE_2025_MS + 2 * HOUR_MS),
                )
                detail_leg = " | ".join(str(r["detail"]) for r in plan_leg)
                self.assertIn("idx_canonical_pit_sec", detail_leg)
                self.assertIn("timeframe_sec=?", detail_leg)

                # Same-ms rapid calls on MultiAssetDataLake raise zero IntegrityError
                for i in range(10):
                    lake.save_factor_audit("tk", "sha", {"i": i})
                    lake.save_regime_snapshot("hyperliquid", "BTC", 12.0, "TREND", {"i": i})
                    lake.save_strategy_blueprint("tk", "A", "PASS", "sha", {"i": i})
            finally:
                lake.close()

            # 3. Test idempotent read-only migrate_from_v3_sqlite leaves source SHA-256 unchanged!
            sha_before = sha256_file(str(legacy_db))
            with DataFoundationRouter(base / "migrated_v4") as router_mig:
                res1 = router_mig.migrate_from_v3_sqlite(legacy_db)
                res2 = router_mig.migrate_from_v3_sqlite(legacy_db)
                self.assertEqual(res1["bars_web3"], 1)
                self.assertEqual(res2["bars_web3"], 1)
                self.assertEqual(len(router_mig.load_bars("web3", "hyperliquid", "BTC", 3_600)), 1)
            sha_after = sha256_file(str(legacy_db))
            self.assertEqual(sha_before, sha_after)

    # --- PIT Macro Events Redaction & PAXG Weekend Freeze Basis ---
    def test_macro_event_pit_redaction_and_paxg_weekend_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with DataFoundationRouter(Path(td) / "v4") as router:
                rel_ts = ny_local_to_utc_ms(2026, 12, 10, 8, 30)  # 2026-12-10T13:30:00Z
                ev = MacroEventRelease(
                    event_type="CPI",
                    event_id="CPI-2026-12-10",
                    ref_period="2026-11",
                    ts_release_ms=rel_ts,
                    ts_avail_ms=rel_ts,
                    actual=0.25,
                    prior=0.20,
                    surprise_z=1.15,
                    source="bls_v2_latest_vintage",
                )
                router.upsert_macro_events([ev])

                # Before release without lookahead_schedule_ms -> NOT visible
                self.assertEqual(router.load_macro_events(["CPI"], as_of_ms=rel_ts - 2 * HOUR_MS), [])
                # Before release WITH 4h lookahead_schedule_ms -> visible with actual/surprise_z REDACTED!
                upcoming = router.load_macro_events(
                    ["CPI"], as_of_ms=rel_ts - 2 * HOUR_MS, lookahead_schedule_ms=4 * HOUR_MS
                )
                self.assertEqual(len(upcoming), 1)
                self.assertEqual(upcoming[0].ts_release_ms, rel_ts)
                self.assertIsNone(upcoming[0].actual)
                self.assertIsNone(upcoming[0].surprise_z)
                # At release -> full outcome visible!
                realized = router.load_macro_events(["CPI"], as_of_ms=rel_ts)
                self.assertEqual(len(realized), 1)
                self.assertAlmostEqual(realized[0].actual or 0.0, 0.25, places=6)
                self.assertAlmostEqual(realized[0].surprise_z or 0.0, 1.15, places=6)

                # PAXG weekend freeze basis test (Fri 2026-11-20 16:00Z -> Sat 2026-11-21 12:00Z)
                fri_16 = int(datetime(2026, 11, 20, 16, 0, tzinfo=timezone.utc).timestamp() * 1000)
                sat_12 = int(datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
                sun_22 = int(datetime(2026, 11, 22, 22, 0, tzinfo=timezone.utc).timestamp() * 1000)
                self.assertFalse(is_comex_weekend_utc(fri_16))
                self.assertTrue(is_comex_weekend_utc(sat_12))
                self.assertFalse(is_comex_weekend_utc(sun_22))

                gc_bars = [
                    CanonicalBar(
                        asset_class=ASSET_CLASS_GOLD_TRADFI,
                        venue="yahoo",
                        symbol="GC=F",
                        timeframe_h=4.0,
                        ts_event_ms=fri_16,
                        ts_avail_ms=fri_16 + 4 * HOUR_MS,
                        open=3600.0,
                        high=3610.0,
                        low=3595.0,
                        close=3600.0,
                        volume_usd=1e9,
                    ),
                    # Future Sunday bar MUST NOT leak into Saturday!
                    CanonicalBar(
                        asset_class=ASSET_CLASS_GOLD_TRADFI,
                        venue="yahoo",
                        symbol="GC=F",
                        timeframe_h=4.0,
                        ts_event_ms=sun_22,
                        ts_avail_ms=sun_22 + 4 * HOUR_MS,
                        open=3700.0,
                        high=3710.0,
                        low=3695.0,
                        close=3700.0,
                        volume_usd=1e9,
                    ),
                ]
                perp_bars = [
                    CanonicalBar(
                        asset_class=ASSET_CLASS_GOLD_PERP,
                        venue="hyperliquid",
                        symbol="PAXG",
                        timeframe_h=4.0,
                        ts_event_ms=sat_12,
                        ts_avail_ms=sat_12 + 4 * HOUR_MS,
                        open=3618.0,
                        high=3620.0,
                        low=3615.0,
                        close=3618.0,
                        volume_usd=1e6,
                    )
                ]
                spot_bars = [
                    CanonicalBar(
                        asset_class=ASSET_CLASS_CRYPTO_SPOT,
                        venue="binance_spot",
                        symbol="PAXG",
                        timeframe_h=4.0,
                        ts_event_ms=sat_12,
                        ts_avail_ms=sat_12 + 4 * HOUR_MS,
                        open=3609.0,
                        high=3610.0,
                        low=3605.0,
                        close=3609.0,
                        volume_usd=1e6,
                    )
                ]
                pts = compute_paxg_basis_points(perp_bars, spot_bars, gc_bars)
                self.assertEqual(len(pts), 1)
                self.assertTrue(pts[0].is_weekend_frozen)
                self.assertEqual(pts[0].gc_ref_close, 3600.0)  # Friday close frozen, ZERO Sunday 3700 leak!
                self.assertAlmostEqual(pts[0].paxg_perp_gc_bps or 0.0, 50.0, places=2)
                self.assertEqual(router.upsert_paxg_basis(pts), 1)
                loaded_bp = router.load_paxg_basis("hyperliquid", 14_400, as_of_ms=sat_12 + 4 * HOUR_MS)
                self.assertEqual(len(loaded_bp), 1)

    # --- Offline Parser Fixture Tests: ALL Web3 Collectors ---
    def test_web3_collectors_offline_wire_fixtures(self) -> None:
        now_ms = BASE_2025_MS + 48 * HOUR_MS
        # 1. Hyperliquid: v=0 pre-launch filter, T+1 > now_ms unclosed drop, +48ms jitter snap, premium_basis_bps
        raw_hl_candles = [
            {"t": BASE_2025_MS - 12 * HOUR_MS, "T": BASE_2025_MS - 1, "o": "100", "h": "100", "l": "100", "c": "100", "v": "0.0", "n": 0},
            {"t": BASE_2025_MS, "T": BASE_2025_MS + 12 * HOUR_MS - 1, "o": "100", "h": "105", "l": "99", "c": "104", "v": "50.0", "n": 120},
            {"t": now_ms - 4 * HOUR_MS, "T": now_ms + 8 * HOUR_MS - 1, "o": "104", "h": "106", "l": "103", "c": "105", "v": "10.0", "n": 25},
        ]
        filtered = HyperliquidWeb3Collector.filter_raw_candles(raw_hl_candles, interval="12h", now_ms=now_ms)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["t"], BASE_2025_MS)

        raw_hl_funding = [
            {"coin": "PAXG", "fundingRate": "0.00001", "premium": "-0.0005", "time": BASE_2025_MS + h * HOUR_MS + 48}
            for h in range(12)
        ]
        fmap = HyperliquidWeb3Collector.parse_funding_rows(raw_hl_funding)
        self.assertEqual(len(fmap), 12)
        self.assertIn(BASE_2025_MS, fmap)
        self.assertEqual(snap_to_nearest_hour_ms(BASE_2025_MS + 48), BASE_2025_MS)
        hl_bars = HyperliquidWeb3Collector.build_canonical_bars("PAXG", "12h", filtered, fmap)
        self.assertEqual(len(hl_bars), 1)
        self.assertAlmostEqual(hl_bars[0].carry_rate_period, 0.00012, places=8)
        self.assertAlmostEqual(hl_bars[0].carry_interval_h, 12.0, places=6)
        self.assertAlmostEqual(hl_bars[0].premium_basis_bps, -5.0, places=4)

        # 2. Binance Vision: 16-digit µs Spot switchover, UM header vs Spot headerless, shuffled metrics, PAXG 8h->4h & BTC 12h alternating carry_interval_h
        self.assertEqual(normalize_binance_ts_ms(1_735_689_600_000_000), 1_735_689_600_000)
        self.assertEqual(normalize_binance_ts_ms(1_733_011_200_000), 1_733_011_200_000)
        spot_zip = _zip_csv_bytes(
            "PAXGUSDT-4h-2025-01.csv",
            f"{BASE_2025_MS * 1000},2600,2610,2595,2605,10,{(BASE_2025_MS + 4*HOUR_MS - 1)*1000},26050,50,5,13025,0\n",
        )
        um_zip = _zip_csv_bytes(
            "BTCUSDT-12h-2025-01.csv",
            "open_time,open,high,low,close,volume,close_time,quote_volume,count,tbv,tbqv,ignore\n"
            f"{BASE_2025_MS},90000,91000,89500,90500,100,{BASE_2025_MS + 12*HOUR_MS - 1},9050000,1000,50,4500000,0\n"
            f"{BASE_2025_MS + 12*HOUR_MS},90500,92000,90000,91500,100,{BASE_2025_MS + 24*HOUR_MS - 1},9150000,1000,50,4500000,0\n",
        )
        spot_rows = parse_zip_csv_rows(spot_zip)
        um_rows = parse_zip_csv_rows(um_zip)
        self.assertEqual(len(spot_rows), 1)
        self.assertEqual(len(um_rows), 2)
        parsed_spot = BinanceVisionWeb3Collector.parse_kline_rows(spot_rows, tf_ms=4 * HOUR_MS, now_ms=now_ms)
        self.assertIn(BASE_2025_MS, parsed_spot)

        parsed_um_k = BinanceVisionWeb3Collector.parse_kline_rows(um_rows, tf_ms=12 * HOUR_MS, now_ms=now_ms)
        # 3 x 8h settlements across 24h: 00:00 + 08:00 inside bar 0 (16h!), 16:00 inside bar 1 (8h!)
        um_fund_rows = [
            [str(BASE_2025_MS + 5), "8", "0.0001"],
            [str(BASE_2025_MS + 8 * HOUR_MS + 3), "8", "0.0002"],
            [str(BASE_2025_MS + 16 * HOUR_MS + 2), "8", "0.0003"],
        ]
        um_fmap = BinanceVisionWeb3Collector.parse_um_funding_rows(um_fund_rows)
        btc_12h = BinanceVisionWeb3Collector.assemble_um_bars(
            canonical_sym="BTC",
            interval="12h",
            klines=parsed_um_k,
            funding_map=um_fmap,
            premium_map={BASE_2025_MS: -4.5, BASE_2025_MS + 12 * HOUR_MS: 2.0},
        )
        self.assertEqual(len(btc_12h), 2)
        self.assertAlmostEqual(btc_12h[0].carry_interval_h, 16.0, places=6)
        self.assertAlmostEqual(btc_12h[1].carry_interval_h, 8.0, places=6)
        self.assertAlmostEqual(btc_12h[0].premium_basis_bps, -4.5, places=4)

        # Unsorted Spark metrics CSV MUST sort chronologically before causal 4h downsampling
        shuffled_metrics = [
            ["2025-01-01 03:55:00", "BTCUSDT", "100", "9000000", "1.1", "1.4", "1.2", "1.3"],
            ["2025-01-01 00:05:00", "BTCUSDT", "90", "8100000", "1.0", "1.1", "1.0", "0.7"],
            ["2025-01-01 01:30:00", "BTCUSDT", "95", "8500000", "1.05", "1.2", "1.1", "1.0"],
        ]
        m_4h = BinanceVisionWeb3Collector.parse_um_metrics_rows(
            shuffled_metrics, symbol="BTCUSDT", downsample_tf_h=4.0, now_ms=now_ms
        )
        self.assertEqual(len(m_4h), 1)
        self.assertEqual(m_4h[0].open_interest_usd, 9_000_000.0)  # Last chronological row (03:55), NOT 01:30!
        self.assertAlmostEqual(m_4h[0].taker_ls_vol_ratio, 1.0, places=4)  # mean(1.3, 0.7, 1.0)

        # 3. Gate non-Chrome UA & `sum` USD volume; OKX `12Hutc`/`4H` & `confirm=="1"`
        gate_c = GateWeb3Collector()
        self.assertNotIn("Chrome", gate_c.client.user_agent)
        gate_bars = GateWeb3Collector.parse_candles_and_funding(
            canonical_sym="BTC",
            interval="12h",
            raw_candles=[
                {"t": BASE_2025_MS // 1000, "o": "90000", "h": "91000", "l": "89000", "c": "90500", "v": 100, "sum": "90500000.0"},
                {"t": (now_ms - HOUR_MS) // 1000, "o": "90500", "h": "91000", "l": "90000", "c": "90800", "v": 10, "sum": "900000.0"},
            ],
            raw_funding=[{"t": BASE_2025_MS // 1000, "r": "0.0001"}],
            now_ms=now_ms,
        )
        self.assertEqual(len(gate_bars), 1)
        self.assertAlmostEqual(gate_bars[0].volume_usd, 90_500_000.0, places=2)

        self.assertEqual(OKXWeb3Collector.resolve_okx_bar("4h"), "4H")
        self.assertEqual(OKXWeb3Collector.resolve_okx_bar("12h"), "12Hutc")
        okx_bars = OKXWeb3Collector.parse_candles_and_funding(
            canonical_sym="BTC",
            interval="12h",
            raw_candles=[
                [str(BASE_2025_MS), "90000", "91000", "89000", "90500", "100", "10", "905000", "1"],
                [str(BASE_2025_MS + 12 * HOUR_MS), "90500", "91000", "90000", "90800", "10", "1", "90800", "0"],  # unclosed!
            ],
            raw_funding=[{"fundingTime": str(BASE_2025_MS), "fundingRate": "0.0001", "realizedRate": "0.00012"}],
            now_ms=now_ms,
        )
        self.assertEqual(len(okx_bars), 1)
        self.assertAlmostEqual(okx_bars[0].carry_rate_period, 0.00012, places=8)

        # 4. Deribit backward continuation & unclosed filter; DefiLlama totalCirculatingUSD ONLY
        dvol_pts = DeribitDVOLCollector.parse_dvol_rows(
            [
                [BASE_2025_MS, 55.0, 58.0, 54.0, 56.5],
                [now_ms - 2 * HOUR_MS, 56.5, 57.0, 56.0, 56.8],  # unclosed today!
            ],
            now_ms=now_ms,
        )
        self.assertEqual(len(dvol_pts), 1)
        self.assertEqual(dvol_pts[0].series_id, "BTC_DVOL")
        self.assertAlmostEqual(dvol_pts[0].value, 56.5, places=4)

        llama_pts = DefiLlamaStablecoinCollector.parse_stablecoin_rows(
            [
                {
                    "date": str(BASE_2025_MS // 1000),
                    "totalCirculating": {"peggedUSD": 200e9, "peggedJPY": 999e12},  # MUST ignore!
                    "totalCirculatingUSD": {"peggedUSD": 200e9, "peggedEUR": 1e9},
                },
                {
                    "date": str((now_ms - HOUR_MS) // 1000),  # unclosed today!
                    "totalCirculatingUSD": {"peggedUSD": 201e9},
                },
            ],
            now_ms=now_ms,
        )
        self.assertEqual(len(llama_pts), 1)
        self.assertAlmostEqual(llama_pts[0].value, 201e9, places=1)

    # --- Offline Parser Fixture Tests: ALL Gold & Macro Collectors ---
    def test_gold_macro_collectors_offline_wire_fixtures(self) -> None:
        now_ms = int(datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)

        # 1. Yahoo 1h: drop live tick (ts%60 != 0), drop ETF :00 V=0 1800s print, preserve :30 & COMEX 100x
        ts_1430 = int(datetime(2026, 11, 20, 14, 30, tzinfo=timezone.utc).timestamp())
        ts_1530 = ts_1430 + 3600
        ts_1600_print = ts_1530 + 1800  # ETF :00 V=0 closing print -> drop!
        ts_live_tick = ts_1530 + 2513   # non-minute live tick -> drop!
        gld_payload = {
            "chart": {
                "result": [
                    {
                        "meta": {"symbol": "GLD", "instrumentType": "ETF"},
                        "timestamp": [ts_1430, ts_1530, ts_1600_print, ts_live_tick],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [250.0, 251.0, 252.0, 252.0],
                                    "high": [251.5, 252.5, 252.0, 252.0],
                                    "low": [249.5, 250.5, 252.0, 252.0],
                                    "close": [251.0, 252.0, 252.0, 252.0],
                                    "volume": [100000, 120000, 0, 0],
                                }
                            ]
                        },
                    }
                ]
            }
        }
        gld_1h = YahooGoldMacroCollector.parse_chart_payload(
            gld_payload, symbol="GLD", interval="1h", range_str="730d", now_ms=now_ms
        )
        self.assertEqual(len(gld_1h), 2)
        self.assertEqual(gld_1h[0].ts_event_ms, ts_1430 * 1000)
        self.assertEqual(gld_1h[0].ts_avail_ms, (ts_1430 + 3600) * 1000)

        # Yahoo 1d adjclose scaling on IEF + COMEX 100x on GC=F
        ts_day = int(datetime(2026, 11, 19, 14, 30, tzinfo=timezone.utc).timestamp())
        ief_payload = {
            "chart": {
                "result": [
                    {
                        "meta": {"symbol": "IEF", "instrumentType": "ETF"},
                        "timestamp": [ts_day],
                        "indicators": {
                            "quote": [{"open": [100.0], "high": [102.0], "low": [99.0], "close": [100.0], "volume": [1000]}],
                            "adjclose": [{"adjclose": [90.0]}],
                        },
                    }
                ]
            }
        }
        ief_1d = YahooGoldMacroCollector.parse_chart_payload(
            ief_payload, symbol="IEF", interval="1d", range_str="5y", now_ms=now_ms
        )
        self.assertEqual(len(ief_1d), 1)
        self.assertAlmostEqual(ief_1d[0].close, 90.0, places=6)
        self.assertAlmostEqual(ief_1d[0].high, 91.8, places=6)

        gc_payload = {
            "chart": {
                "result": [
                    {
                        "meta": {"symbol": "GC=F", "instrumentType": "FUTURE"},
                        "timestamp": [ts_day],
                        "indicators": {
                            "quote": [{"open": [3600.0], "high": [3620.0], "low": [3590.0], "close": [3610.0], "volume": [1000]}],
                        },
                    }
                ]
            }
        }
        gc_1d = YahooGoldMacroCollector.parse_chart_payload(
            gc_payload, symbol="GC=F", interval="1d", range_str="5y", now_ms=now_ms
        )
        self.assertEqual(gc_1d[0].contract_multiplier, 100.0)
        self.assertAlmostEqual(gc_1d[0].volume_usd, 1000 * 3610.0 * 100.0, places=2)

        # 2. US Treasury descending CSV with BOM & N/A
        csv_text = "\ufeffDate,5 YR,7 YR,10 YR,20 YR,30 YR\n11/20/2026,2.10,N/A,2.77,2.85,2.90\n11/19/2026,2.05,2.30,2.70,2.80,2.85\n"
        y_pts = USTreasuryRealYieldCollector.parse_csv_text(csv_text, now_ms=now_ms)
        self.assertEqual(len(y_pts), 2)
        self.assertLess(y_pts[0].ts_event_ms, y_pts[1].ts_event_ms)  # Ascending!
        self.assertIsNone(y_pts[1].real_7y)
        self.assertAlmostEqual(y_pts[1].real_10y or 0.0, 2.77, places=4)
        self.assertEqual(y_pts[1].ts_avail_ms - y_pts[1].ts_event_ms, 22 * HOUR_MS)

        # 3. Fed FOMC HTML zebra-striping trap & notation-vote skip & 2026-12-09 19:00Z SEP*
        fomc_html = """
        <div class="panel-heading">2025 FOMC Meetings</div>
        <div class="row fomc-meeting">
          <div class="fomc-meeting--shaded fomc-meeting__month"><strong>October</strong></div>
          <div class="fomc-meeting__date">28-29</div>
        </div>
        <div class="row fomc-meeting">
          <div class="fomc-meeting__month"><strong>August</strong></div>
          <div class="fomc-meeting__date">22 (notation vote)</div>
        </div>
        <div class="panel-heading">2026 FOMC Meetings</div>
        <div class="row fomc-meeting" ">
          <div class="fomc-meeting__month"><strong>December</strong></div>
          <div class="fomc-meeting__date">8-9*</div>
        </div>
        """
        fomc_evs = FedFOMCCalendarCollector.parse_fomc_html(fomc_html)
        self.assertEqual(len(fomc_evs), 2)
        oct25 = next(e for e in fomc_evs if e.event_id == "FOMC-2025-10-29")
        dec26 = next(e for e in fomc_evs if e.event_id == "FOMC-2026-12-09")
        self.assertFalse(oct25.is_sep_meeting)  # Shaded row WITHOUT '*' is NOT SEP!
        self.assertTrue(dec26.is_sep_meeting)   # '*' IS SEP!
        self.assertEqual(dec26.ts_release_ms, 1_796_842_800_000)  # 2026-12-09T19:00:00Z

        # 4. BLS 2025-M10 value="-" + statutory 2026-12-04 NFP + official 2026-12-10 CPI
        self.assertEqual(bls_nfp_release_date(2026, 11).isoformat(), "2026-12-04")
        self.assertEqual(ny_local_to_utc_ms(2026, 12, 4, 8, 30), 1_796_391_000_000)
        cpi_26_11 = bls_cpi_release_date(2026, 11)
        self.assertIsNotNone(cpi_26_11)
        self.assertEqual(cpi_26_11.isoformat(), "2026-12-10")  # type: ignore[union-attr]
        self.assertEqual(ny_local_to_utc_ms(2026, 12, 10, 8, 30), 1_796_909_400_000)
        self.assertIsNone(bls_cpi_release_date(2025, 10))

        bls_fixture = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [
                    {
                        "seriesID": "CUSR0000SA0",
                        "data": [
                            {"year": "2025", "period": "M09", "value": "320.0", "footnotes": [{}]},
                            {"year": "2025", "period": "M10", "value": "-", "footnotes": [{"code": "X", "text": "lapse"}]},
                            {"year": "2025", "period": "M11", "value": "321.5", "footnotes": [{}]},
                            {"year": "2026", "period": "M09", "value": "336.0", "footnotes": [{}]},
                            {"year": "2026", "period": "M10", "value": "337.476", "footnotes": [{}]},
                        ],
                    },
                    {
                        "seriesID": "CES0000000001",
                        "data": [
                            {"year": "2026", "period": "M08", "value": "159068", "footnotes": [{}]},
                            {"year": "2026", "period": "M09", "value": "159128", "footnotes": [{"code": "P"}]},
                            {"year": "2026", "period": "M10", "value": "159293", "footnotes": [{"code": "P"}]},
                        ],
                    },
                ]
            },
        }
        bls_evs = BLSMacroEventCollector.build_events_from_payload(
            bls_fixture, schedule_start_year=2025, schedule_end_year=2026, now_ms=now_ms
        )
        nfp_dec26 = next(e for e in bls_evs if e.event_id == "NFP-2026-12-04")
        cpi_dec26 = next(e for e in bls_evs if e.event_id == "CPI-2026-12-10")
        nfp_nov26 = next(e for e in bls_evs if e.ref_period == "2026-10" and e.event_type == "NFP")
        self.assertEqual(nfp_dec26.ts_release_ms, 1_796_391_000_000)
        self.assertIsNone(nfp_dec26.actual)  # Future event -> actual MUST be None!
        self.assertEqual(cpi_dec26.ts_release_ms, 1_796_909_400_000)
        self.assertIsNone(cpi_dec26.actual)
        self.assertAlmostEqual(nfp_nov26.actual or 0.0, 165.0, places=2)
        self.assertFalse(any(e.event_type == "CPI" and e.ref_period == "2025-10" for e in bls_evs))

    def test_hl_v3_seed_skipping_paxg_8h_to_4h_deribit_continuation_and_dq_cert(self) -> None:
        # 1. HL fetch_funding_history skips already-seeded v3 range cleanly
        hl = HyperliquidWeb3Collector()
        seed_start = BASE_2025_MS + 10 * HOUR_MS
        seed_end = BASE_2025_MS + 20 * HOUR_MS
        seed_map = {seed_start + h * HOUR_MS: (0.00001, 1.0) for h in range(11)}
        called_windows: list[tuple[int, int]] = []

        def _fake_hl_post(_url: str, *, json_payload: dict[str, Any] | None = None, **_kw: Any) -> list[dict[str, Any]]:
            assert json_payload is not None
            s_t = int(json_payload["startTime"])
            e_t = int(json_payload["endTime"])
            called_windows.append((s_t, e_t))
            return [
                {"coin": "XRP", "fundingRate": "0.00002", "premium": "0.0001", "time": s_t + 12}
            ]

        with patch.object(hl.client, "fetch_json", side_effect=_fake_hl_post):
            fmap = hl.fetch_funding_history(
                "XRP",
                start_ms=BASE_2025_MS,
                end_ms=BASE_2025_MS + 25 * HOUR_MS,
                seed_hourly_map=seed_map,
                pace_sec=0.0,
            )
        # Must call ONLY pre-seed [0..seed_start-1] and post-seed [seed_end+1h..end]
        self.assertEqual(len(called_windows), 2)
        self.assertEqual(called_windows[0], (BASE_2025_MS, seed_start - 1))
        self.assertEqual(called_windows[1], (seed_end + HOUR_MS, BASE_2025_MS + 25 * HOUR_MS))
        self.assertIn(BASE_2025_MS, fmap)
        self.assertIn(seed_start, fmap)
        self.assertIn(seed_end + HOUR_MS, fmap)

        # 2. Binance UM PAXG mid-month 8h -> 4h funding interval switch
        paxg_k = {
            BASE_2025_MS: (2600.0, 2610.0, 2595.0, 2605.0, 1e6),
            BASE_2025_MS + 12 * HOUR_MS: (2605.0, 2615.0, 2600.0, 2610.0, 1e6),
        }
        paxg_f_rows = [
            [str(BASE_2025_MS), "8", "0.0001"],
            [str(BASE_2025_MS + 8 * HOUR_MS), "8", "0.0001"],
            # Switched to 4h settlement on second 12h bar (12:00, 16:00, 20:00 -> 3 x 4h = 12.0h!)
            [str(BASE_2025_MS + 12 * HOUR_MS), "4", "0.00005"],
            [str(BASE_2025_MS + 16 * HOUR_MS), "4", "0.00005"],
            [str(BASE_2025_MS + 20 * HOUR_MS), "4", "0.00005"],
        ]
        paxg_fmap = BinanceVisionWeb3Collector.parse_um_funding_rows(paxg_f_rows)
        paxg_12h = BinanceVisionWeb3Collector.assemble_um_bars(
            canonical_sym="PAXG",
            interval="12h",
            klines=paxg_k,
            funding_map=paxg_fmap,
            premium_map={},
        )
        self.assertEqual(len(paxg_12h), 2)
        self.assertAlmostEqual(paxg_12h[0].carry_interval_h, 16.0, places=6)
        self.assertAlmostEqual(paxg_12h[1].carry_interval_h, 12.0, places=6)

        # 3. Deribit backward continuation pagination across 2 pages
        deribit = DeribitDVOLCollector()
        day0 = BASE_2025_MS
        day1 = BASE_2025_MS + DAY_MS
        day2 = BASE_2025_MS + 2 * DAY_MS
        pages = [
            {"result": {"data": [[day1, 50, 52, 49, 51.0], [day2, 51, 53, 50, 52.0]], "continuation": day0}},
            {"result": {"data": [[day0, 48, 51, 47, 50.0]], "continuation": None}},
        ]
        with patch.object(deribit.client, "fetch_json", side_effect=pages):
            pts = deribit.fetch_dvol(start_ms=day0 - DAY_MS, end_ms=day2 + DAY_MS, now_ms=day2 + 2 * DAY_MS)
        self.assertEqual([p.ts_event_ms for p in pts], [day0, day1, day2])

        # 4. DataFoundationRouter.audit_domain_dq persists DQ certificate to catalog.sqlite
        with tempfile.TemporaryDirectory() as td:
            with DataFoundationRouter(Path(td) / "v4") as router:
                bars = _make_hourly_bars(50, tf_h=12.0, carry_rate=0.0001, carry_interval_h=12.0)
                router.upsert_bars("web3", bars)
                spec = default_gold_specs()["PAXG"]
                rep = router.audit_domain_dq("web3", "hyperliquid", "BTC", 43_200, spec)
                self.assertTrue(rep.passed)
                cert_cnt = router.catalog_conn.fetch_one("SELECT COUNT(*) AS c FROM dq_certificates")
                self.assertIsNotNone(cert_cnt)
                self.assertEqual(int(cert_cnt["c"]), 1)

    # --- Real Ingested data/platform_v4_lake Read-Only Integrity Check ---
    def test_real_platform_v4_lake_integrity_if_present(self) -> None:
        env_dir = os.environ.get("PLATFORM_V4_LAKE_DIR", "").strip()
        lake_dir = Path(env_dir) if env_dir else REAL_V4_LAKE_DIR
        cat_path = lake_dir / "catalog.sqlite"
        w3_path = lake_dir / "domains" / "web3_lake.sqlite"
        gm_path = lake_dir / "domains" / "gold_macro_lake.sqlite"
        if not (cat_path.exists() and w3_path.exists() and gm_path.exists()):
            self.skipTest(f"data/platform_v4_lake not yet populated at {lake_dir}")

        # Open strictly read-only (?mode=ro) so test never mutates the real lake
        c_cat = sqlite3.connect(f"file:{cat_path.resolve()}?mode=ro", uri=True, timeout=30.0)
        c_w3 = sqlite3.connect(f"file:{w3_path.resolve()}?mode=ro", uri=True, timeout=30.0)
        c_gm = sqlite3.connect(f"file:{gm_path.resolve()}?mode=ro", uri=True, timeout=30.0)
        try:
            for c in (c_cat, c_w3, c_gm):
                c.row_factory = sqlite3.Row
                c.execute("PRAGMA query_only=ON;")

            # 100% REAL provenance & zero lookahead across all bar tables
            bad_w3 = c_w3.execute(
                "SELECT COUNT(*) AS n FROM web3_bars WHERE provenance != ? OR ts_avail_ms < ts_event_ms",
                (PROVENANCE_REAL,),
            ).fetchone()["n"]
            bad_gm = c_gm.execute(
                "SELECT COUNT(*) AS n FROM tradfi_bars WHERE provenance != ? OR ts_avail_ms < ts_event_ms",
                (PROVENANCE_REAL,),
            ).fetchone()["n"]
            self.assertEqual(bad_w3, 0)
            self.assertEqual(bad_gm, 0)

            # Multi-year Web3 HL depth
            btc_12h = c_w3.execute(
                "SELECT COUNT(*) AS n FROM web3_bars WHERE venue='hyperliquid' AND symbol='BTC' AND timeframe_sec=43200"
            ).fetchone()["n"]
            btc_4h = c_w3.execute(
                "SELECT COUNT(*) AS n FROM web3_bars WHERE venue='hyperliquid' AND symbol='BTC' AND timeframe_sec=14400"
            ).fetchone()["n"]
            self.assertGreaterEqual(btc_12h, 2_000)
            self.assertGreaterEqual(btc_4h, 4_800)

            # Deribit DVOL + DefiLlama Stablecoins + Binance metrics
            dvol_n = c_w3.execute(
                "SELECT COUNT(*) AS n FROM web3_market_macro WHERE series_id='BTC_DVOL'"
            ).fetchone()["n"]
            llama_n = c_w3.execute(
                "SELECT COUNT(*) AS n FROM web3_market_macro WHERE series_id='STABLECOIN_MCAP_USD'"
            ).fetchone()["n"]
            metrics_n = c_w3.execute("SELECT COUNT(*) AS n FROM web3_derivatives_metrics").fetchone()["n"]
            self.assertGreaterEqual(dvol_n, 1_800)
            self.assertGreaterEqual(llama_n, 2_500)
            self.assertGreaterEqual(metrics_n, 5_000)

            # Gold/Macro: GC=F 1d & 4h, US Treasury TIPS yields, FOMC/NFP/CPI Dec 2026 events, PAXG basis
            gc_1d = c_gm.execute(
                "SELECT COUNT(*) AS n FROM tradfi_bars WHERE venue='yahoo' AND symbol='GC=F' AND timeframe_sec=86400"
            ).fetchone()["n"]
            gc_4h = c_gm.execute(
                "SELECT COUNT(*) AS n FROM tradfi_bars WHERE venue='yahoo' AND symbol='GC=F' AND timeframe_sec=14400"
            ).fetchone()["n"]
            yields_n = c_gm.execute("SELECT COUNT(*) AS n FROM macro_yield_curve").fetchone()["n"]
            basis_n = c_gm.execute("SELECT COUNT(*) AS n FROM paxg_basis_arbitration").fetchone()["n"]
            self.assertGreaterEqual(gc_1d, 1_200)
            self.assertGreaterEqual(gc_4h, 3_000)
            self.assertGreaterEqual(yields_n, 1_200)
            self.assertGreaterEqual(basis_n, 1_000)

            for eid, exp_ts in (
                ("FOMC-2026-12-09", 1_796_842_800_000),
                ("NFP-2026-12-04", 1_796_391_000_000),
                ("CPI-2026-12-10", 1_796_909_400_000),
            ):
                row = c_gm.execute(
                    "SELECT ts_release_ms FROM macro_events WHERE event_id=?", (eid,)
                ).fetchone()
                self.assertIsNotNone(row, msg=f"Missing critical event {eid}")
                self.assertEqual(int(row["ts_release_ms"]), exp_ts)

            # SHA-256 manifests match exported partition files on disk
            manifests = c_cat.execute(
                "SELECT rel_path, sha256 FROM dataset_manifests ORDER BY created_ts_ms DESC"
            ).fetchall()
            self.assertGreaterEqual(len(manifests), 7)
            checked_paths: set[str] = set()
            for m in manifests:
                rp = str(m["rel_path"])
                if rp in checked_paths:
                    continue
                checked_paths.add(rp)
                fpath = lake_dir / rp
                self.assertTrue(fpath.exists(), msg=f"Missing exported partition {fpath}")
                self.assertEqual(sha256_file(str(fpath)), str(m["sha256"]))

            # FIX-1: 0/39+ rows in asset_registry missing contract_multiplier; GC=F & SI=F exact
            ar_rows = c_cat.execute("SELECT venue, symbol, spec_json FROM asset_registry").fetchall()
            self.assertGreaterEqual(len(ar_rows), 39)
            for ar in ar_rows:
                spec_d = json.loads(str(ar["spec_json"]))
                self.assertIn("contract_multiplier", spec_d, msg=f"Missing multiplier on {ar['venue']}:{ar['symbol']}")
                if ar["venue"] == "yahoo" and ar["symbol"] == "GC=F":
                    self.assertEqual(float(spec_d["contract_multiplier"]), 100.0)
                    self.assertEqual(float(spec_d["max_leverage_cap"]), 2.0)
                if ar["venue"] == "yahoo" and ar["symbol"] == "SI=F":
                    self.assertEqual(float(spec_d["contract_multiplier"]), 5000.0)

            # FIX-3: All 3 append-only triggers exist on trial_ledger
            trigs = {
                str(r["name"])
                for r in c_cat.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
            }
            self.assertTrue(
                {
                    "trg_trial_ledger_no_update",
                    "trg_trial_ledger_no_delete",
                    "trg_trial_ledger_no_replace",
                }.issubset(trigs)
            )

            # FIX-2: ZERO weekends with gc_ref_close drift inside contiguous is_weekend_frozen=1 runs
            for v_perp in ("hyperliquid", "binance_um"):
                for tf_sec in (14_400, 43_200):
                    b_rows = c_gm.execute(
                        "SELECT ts_event_ms, gc_ref_close, is_weekend_frozen "
                        "FROM paxg_basis_arbitration "
                        "WHERE venue_perp=? AND ref_tradfi_symbol='GC=F' AND timeframe_sec=? "
                        "ORDER BY ts_event_ms ASC",
                        (v_perp, tf_sec),
                    ).fetchall()
                    self.assertGreater(len(b_rows), 0)
                    drift_runs = 0
                    in_frozen = False
                    run_gcs: set[float] = set()
                    for br in b_rows:
                        if int(br["is_weekend_frozen"]) == 1:
                            if not in_frozen:
                                in_frozen = True
                                run_gcs = set()
                            if br["gc_ref_close"] is not None:
                                run_gcs.add(round(float(br["gc_ref_close"]), 6))
                        else:
                            if in_frozen and len(run_gcs) > 1:
                                drift_runs += 1
                            in_frozen = False
                            run_gcs = set()
                    if in_frozen and len(run_gcs) > 1:
                        drift_runs += 1
                    self.assertEqual(drift_runs, 0, msg=f"Weekend gc_ref_close drift on {v_perp}:{tf_sec}s")

            # FIX-7: EXPLAIN QUERY PLAN for default load_paxg_basis has ZERO USE TEMP B-TREE FOR ORDER BY
            plan_bp = c_gm.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM paxg_basis_arbitration "
                "WHERE venue_perp=? AND timeframe_sec=? AND ref_tradfi_symbol=? "
                "ORDER BY ts_event_ms ASC",
                ("hyperliquid", 14_400, "GC=F"),
            ).fetchall()
            detail_bp = " | ".join(str(r["detail"]) for r in plan_bp)
            self.assertIn("idx_paxg_basis_ev", detail_bp)
            self.assertNotIn("USE TEMP B-TREE FOR ORDER BY", detail_bp)
        finally:
            c_cat.close()
            c_w3.close()
            c_gm.close()

    # --- Iteration 2 Surgical Remediation Regression Tests (FIX-1 .. FIX-10) ---
    def test_fix1_asset_registry_survives_v3_migration_and_backfills_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            v3_db = base / "legacy_v3.sqlite"
            con = sqlite3.connect(str(v3_db), timeout=30.0)
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute(
                """
                CREATE TABLE asset_registry (
                    venue TEXT NOT NULL, symbol TEXT NOT NULL, asset_class TEXT NOT NULL,
                    spec_json TEXT NOT NULL, updated_ts_ms INTEGER NOT NULL,
                    PRIMARY KEY (venue, symbol)
                );
                """
            )
            stale_gc = json.dumps(
                {"symbol": "GC=F", "venue": "yahoo", "asset_class": ASSET_CLASS_GOLD_TRADFI, "max_leverage_cap": 1.0}
            )
            stale_si = json.dumps(
                {"symbol": "SI=F", "venue": "yahoo", "asset_class": ASSET_CLASS_MACRO_INDEX, "max_leverage_cap": 1.0}
            )
            con.execute(
                "INSERT INTO asset_registry VALUES (?,?,?,?,?)",
                ("yahoo", "GC=F", ASSET_CLASS_GOLD_TRADFI, stale_gc, BASE_2025_MS),
            )
            con.execute(
                "INSERT INTO asset_registry VALUES (?,?,?,?,?)",
                ("yahoo", "SI=F", ASSET_CLASS_MACRO_INDEX, stale_si, BASE_2025_MS),
            )
            con.commit()
            con.close()

            with DataFoundationRouter(base / "v4_lake") as router:
                # Register authoritative v4 GC=F first; migrate_from_v3_sqlite MUST NOT overwrite it!
                router.register_asset(default_macro_specs_v4()["GC=F"], domain="gold_macro")
                router.migrate_from_v3_sqlite(v3_db)

                row_gc = router.catalog_conn.fetch_one(
                    "SELECT spec_json FROM asset_registry WHERE venue='yahoo' AND symbol='GC=F'"
                )
                row_si = router.catalog_conn.fetch_one(
                    "SELECT spec_json FROM asset_registry WHERE venue='yahoo' AND symbol='SI=F'"
                )
                assert row_gc is not None and row_si is not None
                gc_d = json.loads(str(row_gc["spec_json"]))
                si_d = json.loads(str(row_si["spec_json"]))
                self.assertEqual(float(gc_d["contract_multiplier"]), 100.0)
                self.assertEqual(float(gc_d["max_leverage_cap"]), 2.0)
                # SI=F migrated from v3 without prior v4 row MUST get backfilled 5000.0 multiplier!
                self.assertEqual(float(si_d["contract_multiplier"]), 5000.0)

    def test_fix2_paxg_basis_weekend_freeze_phase_alignment_4h_and_12h(self) -> None:
        # Fri 2025-04-11 16:00Z GC=F close=3300.0 (avail Fri 20:00Z); Sun 2025-04-13 20:00Z GC=F reopen=3800.0 (avail Mon 00:00Z)
        fri_16 = int(datetime(2025, 4, 11, 16, 0, tzinfo=timezone.utc).timestamp() * 1000)
        sun_20 = int(datetime(2025, 4, 13, 20, 0, tzinfo=timezone.utc).timestamp() * 1000)
        gc_bars = [
            CanonicalBar(
                asset_class=ASSET_CLASS_GOLD_TRADFI, venue="yahoo", symbol="GC=F", timeframe_h=4.0,
                ts_event_ms=fri_16, ts_avail_ms=fri_16 + 4 * HOUR_MS,
                open=3300.0, high=3305.0, low=3295.0, close=3300.0, volume_usd=1e9,
            ),
            CanonicalBar(
                asset_class=ASSET_CLASS_GOLD_TRADFI, venue="yahoo", symbol="GC=F", timeframe_h=4.0,
                ts_event_ms=sun_20, ts_avail_ms=sun_20 + 4 * HOUR_MS,
                open=3790.0, high=3810.0, low=3785.0, close=3800.0, volume_usd=1e9,
            ),
        ]
        # 4h PAXG bars from Fri 16:00Z through Sun 20:00Z (14 bars)
        paxg_4h = [
            CanonicalBar(
                asset_class=ASSET_CLASS_GOLD_PERP, venue="hyperliquid", symbol="PAXG", timeframe_h=4.0,
                ts_event_ms=fri_16 + i * 4 * HOUR_MS, ts_avail_ms=fri_16 + (i + 1) * 4 * HOUR_MS,
                open=3310.0, high=3315.0, low=3305.0, close=3310.0, volume_usd=1e6,
            )
            for i in range(14)
        ]
        pts_4h = compute_paxg_basis_points(paxg_4h, [], gc_bars)
        self.assertFalse(pts_4h[0].is_weekend_frozen)   # Fri 16:00Z (closes Fri 20:00Z)
        self.assertTrue(pts_4h[1].is_weekend_frozen)    # Fri 20:00Z (closes Sat 00:00Z) -> frozen!
        self.assertTrue(pts_4h[-2].is_weekend_frozen)   # Sun 16:00Z (closes Sun 20:00Z) -> frozen!
        self.assertFalse(pts_4h[-1].is_weekend_frozen)  # Sun 20:00Z (closes Mon 00:00Z) -> unfrozen!
        self.assertEqual(pts_4h[-1].gc_ref_close, 3800.0)
        for p in pts_4h:
            if p.is_weekend_frozen:
                self.assertEqual(p.gc_ref_close, 3300.0)

        # 12h PAXG bars: Fri 12:00Z, Sat 00:00Z, Sat 12:00Z, Sun 00:00Z, Sun 12:00Z
        fri_12 = int(datetime(2025, 4, 11, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
        paxg_12h = [
            CanonicalBar(
                asset_class=ASSET_CLASS_GOLD_PERP, venue="hyperliquid", symbol="PAXG", timeframe_h=12.0,
                ts_event_ms=fri_12 + i * 12 * HOUR_MS, ts_avail_ms=fri_12 + (i + 1) * 12 * HOUR_MS,
                open=3310.0, high=3315.0, low=3305.0, close=3310.0, volume_usd=1e6,
            )
            for i in range(5)
        ]
        pts_12h = compute_paxg_basis_points(paxg_12h, [], gc_bars)
        self.assertEqual([p.is_weekend_frozen for p in pts_12h], [True, True, True, True, False])
        for p in pts_12h:
            if p.is_weekend_frozen:
                self.assertEqual(p.gc_ref_close, 3300.0)
        self.assertEqual(pts_12h[-1].gc_ref_close, 3800.0)

    def test_fix3_trial_ledger_blocks_insert_or_replace(self) -> None:
        with tempfile.TemporaryDirectory() as td, DataFoundationRouter(Path(td) / "v4") as router:
            tid = router.record_trial("fam_sec", "W1", "xs_mom", "orig_hash", {"sr": 0.5}, "REJECTED")
            with self.assertRaises(sqlite3.IntegrityError):
                router.catalog_conn.execute(
                    "INSERT OR REPLACE INTO trial_ledger "
                    "(trial_id, created_ts_ms, family_id, track_name, strategy_name, params_hash, "
                    " dataset_sha256, metrics_json, verdict, schema_version) "
                    "VALUES (?, 1, 'fam_sec', 'W1', 'xs_mom', 'hacked', '', '{}', 'PROMOTE_SHADOW_CANDIDATE', 'rea-001-v1')",
                    (tid,),
                )
            row = router.catalog_conn.fetch_one(
                "SELECT verdict, params_hash FROM trial_ledger WHERE trial_id=?", (tid,)
            )
            assert row is not None
            self.assertEqual(str(row["verdict"]), "REJECTED")
            self.assertEqual(str(row["params_hash"]), "orig_hash")

    def test_fix4_dq_ma_07_hardening_and_binance_um_causal_interval(self) -> None:
        # 1. Unbounded carry_interval_h=10000.0 bypass MUST FAIL
        huge_ci = [
            CanonicalBar(
                asset_class=ASSET_CLASS_CRYPTO_PERP, venue="hyperliquid", symbol="BTC",
                timeframe_h=12.0, ts_event_ms=BASE_2025_MS + i * 12 * HOUR_MS,
                ts_avail_ms=BASE_2025_MS + (i + 1) * 12 * HOUR_MS,
                open=100.0, high=101.0, low=99.0, close=100.5, volume_usd=1e6,
                carry_rate_period=(0.08 if i == 0 else 0.0), carry_interval_h=10000.0,
            )
            for i in range(100)
        ]
        dq7_ci = next(c for c in audit_bars_quality(huge_ci, CALENDAR_CRYPTO_24_7).checks if c.gate_id == "DQ-MA-07")
        self.assertFalse(dq7_ci.passed)

        # 2. Mixed asset_class (bar[0]=SPOT, 99 zero-funding PERPs) MUST FAIL
        mixed = _make_hourly_bars(1, asset_class=ASSET_CLASS_CRYPTO_SPOT, tf_h=12.0, carry_rate=0.0, carry_interval_h=12.0) + _make_hourly_bars(
            99, start_ms=BASE_2025_MS + 12 * HOUR_MS, asset_class=ASSET_CLASS_CRYPTO_PERP, tf_h=12.0, carry_rate=0.0, carry_interval_h=12.0
        )
        dq7_mix = next(c for c in audit_bars_quality(mixed, CALENDAR_CRYPTO_24_7).checks if c.gate_id == "DQ-MA-07")
        self.assertFalse(dq7_mix.passed)
        self.assertIn("mixed_perp_and_non_perp", dq7_mix.detail)

        # 3. Odd-length 101-bar Binance UM 12h alternating (16.0, 8.0) with 25% zero funding MUST FAIL
        odd_um = [
            CanonicalBar(
                asset_class=ASSET_CLASS_CRYPTO_PERP, venue="binance_um", symbol="BTC",
                timeframe_h=12.0, ts_event_ms=BASE_2025_MS + i * 12 * HOUR_MS,
                ts_avail_ms=BASE_2025_MS + (i + 1) * 12 * HOUR_MS,
                open=100.0, high=101.0, low=99.0, close=100.5, volume_usd=1e6,
                carry_rate_period=(0.0 if i % 4 == 3 else 0.0001),
                carry_interval_h=(16.0 if i % 2 == 0 else 8.0),
            )
            for i in range(101)
        ]
        dq7_odd = next(c for c in audit_bars_quality(odd_um, CALENDAR_CRYPTO_24_7).checks if c.gate_id == "DQ-MA-07")
        self.assertFalse(dq7_odd.passed)

        # 4. 4%-subset 0.01/1h (8760% APR) & single-bar 0.04/1h (35040% APR) MUST FAIL (no 8x under-scaling)
        for bad_rate, bad_cnt in ((0.01, 4), (0.04, 1)):
            bars_1h = _make_hourly_bars(100, tf_h=1.0, carry_rate=0.0000125, carry_interval_h=1.0)
            poisoned = [
                CanonicalBar(**{**b.to_dict(), "carry_rate_period": bad_rate}) if i < bad_cnt else b
                for i, b in enumerate(bars_1h)
            ]
            dq7_u = next(c for c in audit_bars_quality(poisoned, CALENDAR_CRYPTO_24_7).checks if c.gate_id == "DQ-MA-07")
            self.assertFalse(dq7_u.passed, msg=f"Should fail for rate={bad_rate}, cnt={bad_cnt}")

        # 5. Clustered duplicate non-zero bars in same 8h slot MUST FAIL (50/100 slots covered)
        clustered_4h = [
            CanonicalBar(
                asset_class=ASSET_CLASS_CRYPTO_PERP, venue="binance_um", symbol="BTC",
                timeframe_h=4.0, ts_event_ms=BASE_2025_MS + i * 4 * HOUR_MS,
                ts_avail_ms=BASE_2025_MS + (i + 1) * 4 * HOUR_MS,
                open=100.0, high=101.0, low=99.0, close=100.5, volume_usd=1e6,
                carry_rate_period=(0.0001 if i < 100 else 0.0),
                carry_interval_h=8.0,
            )
            for i in range(200)
        ]
        dq7_cl = next(c for c in audit_bars_quality(clustered_4h, CALENDAR_CRYPTO_24_7).checks if c.gate_id == "DQ-MA-07")
        self.assertFalse(dq7_cl.passed)

        # 6. Exact 97.9% FAIL vs 98.0% PASS boundary
        b1000 = _make_hourly_bars(1_000, tf_h=12.0, carry_rate=0.0001, carry_interval_h=12.0)
        b979 = [CanonicalBar(**{**b.to_dict(), "carry_rate_period": 0.0}) if i < 21 else b for i, b in enumerate(b1000)]
        b980 = [CanonicalBar(**{**b.to_dict(), "carry_rate_period": 0.0}) if i < 20 else b for i, b in enumerate(b1000)]
        self.assertFalse(next(c for c in audit_bars_quality(b979, CALENDAR_CRYPTO_24_7).checks if c.gate_id == "DQ-MA-07").passed)
        self.assertTrue(next(c for c in audit_bars_quality(b980, CALENDAR_CRYPTO_24_7).checks if c.gate_id == "DQ-MA-07").passed)

        # 7. Binance UM PAXG 4h 8h->4h mid-history switch: causal current_iv (NO future lookahead) & DQ-MA-07 PASS!
        paxg_kl4 = {
            BASE_2025_MS + i * 4 * HOUR_MS: (2600.0, 2610.0, 2595.0, 2605.0, 1e6)
            for i in range(60)
        }
        paxg_f: dict[int, tuple[float, float]] = {}
        # First 5 days (30 x 4h bars): 8h funding at 00:00, 08:00, 16:00 (15 settlements)
        for s in range(15):
            paxg_f[BASE_2025_MS + s * 8 * HOUR_MS] = (0.0001, 8.0)
        # Next 5 days (30 x 4h bars): 4h funding (30 settlements)
        for s in range(30):
            paxg_f[BASE_2025_MS + 5 * DAY_MS + s * 4 * HOUR_MS] = (0.00005, 4.0)
        paxg_4h = BinanceVisionWeb3Collector.assemble_um_bars(
            canonical_sym="PAXG", interval="4h", klines=paxg_kl4, funding_map=paxg_f, premium_map={}
        )
        self.assertEqual(len(paxg_4h), 60)
        # Unmatched bars in first 5 days MUST remain carry_interval_h=8.0 (zero future peeking to 4.0!)
        for b in paxg_4h[:30]:
            self.assertAlmostEqual(b.carry_interval_h, 8.0, places=6)
        for b in paxg_4h[30:]:
            self.assertAlmostEqual(b.carry_interval_h, 4.0, places=6)
        rep_sw = audit_bars_quality(paxg_4h, CALENDAR_CRYPTO_24_7)
        dq7_sw = next(c for c in rep_sw.checks if c.gate_id == "DQ-MA-07")
        self.assertTrue(dq7_sw.passed, msg=dq7_sw)

    def test_fix5_strict_type_and_consistency_validation(self) -> None:
        good = _make_hourly_bars(1)[0]
        # Explicit negative timeframe_sec=-1 MUST NOT be overwritten and MUST fail validation
        neg_tf = CanonicalBar(**{**good.to_dict(), "timeframe_sec": -1})
        self.assertEqual(neg_tf.timeframe_sec, -1)
        with self.assertRaises(ValueError):
            validate_canonical_bar(neg_tf)
        # Mismatched timeframe_h=1.0 vs timeframe_sec=14400 MUST fail
        with self.assertRaises(ValueError):
            validate_canonical_bar(CanonicalBar(**{**good.to_dict(), "timeframe_h": 1.0, "timeframe_sec": 14_400}))
        # NaN / bool timestamps MUST fail
        with self.assertRaises(ValueError):
            validate_canonical_bar(CanonicalBar(**{**good.to_dict(), "ts_event_ms": float("nan")}))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            validate_canonical_bar(CanonicalBar(**{**good.to_dict(), "ts_avail_ms": True}))  # type: ignore[arg-type]
        # bool smuggled as OHLC price MUST fail
        with self.assertRaises(ValueError):
            validate_canonical_bar(CanonicalBar(**{**good.to_dict(), "open": True, "high": True, "low": True, "close": True}))  # type: ignore[arg-type]
        # Whitespace-only venue/symbol/asset_class MUST fail
        for field_k in ("venue", "symbol", "asset_class"):
            with self.assertRaises(ValueError):
                validate_canonical_bar(CanonicalBar(**{**good.to_dict(), field_k: "   "}))

        # All 5 domain validators reject NaN timestamps, bool numerics, and whitespace identifiers
        with self.assertRaises(ValueError):
            validate_web3_derivatives_metric(
                Web3DerivativesMetric("binance_um", "BTC", float("nan"), BASE_2025_MS, 1.0, 1.0, 1.0, 1.0)  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            validate_web3_derivatives_metric(
                Web3DerivativesMetric("  ", "BTC", BASE_2025_MS, BASE_2025_MS, 1.0, 1.0, 1.0, 1.0)
            )
        with self.assertRaises(ValueError):
            validate_web3_market_macro_point(
                Web3MarketMacroPoint("deribit", "BTC_DVOL", BASE_2025_MS, BASE_2025_MS, True)  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            validate_macro_yield_point(
                MacroYieldPoint("us_treasury", BASE_2025_MS, BASE_2025_MS, True, None, 2.0, None, None)  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            validate_macro_event_release(
                MacroEventRelease("CPI", "   ", "2026-11", BASE_2025_MS, BASE_2025_MS, actual=0.2)
            )
        with self.assertRaises(ValueError):
            validate_paxg_basis_point(
                PAXGBasisPoint("hyperliquid", "binance_spot", "GC=F", True, BASE_2025_MS, BASE_2025_MS, 3000.0)  # type: ignore[arg-type]
            )

    def test_fix6_resample_bars_us_equity_rth_24h_and_mixed_tf_rejection(self) -> None:
        # 30 trading days of GLD 1h (7 RTH bars/day) -> 24h MUST return all 30 days
        gld_1h: list[CanonicalBar] = []
        gld_4h: list[CanonicalBar] = []
        for d in range(30):
            day_base = BASE_2025_MS + d * DAY_MS
            for h in range(7):
                t0 = day_base + 13 * HOUR_MS + 30 * 60_000 + h * HOUR_MS
                gld_1h.append(
                    CanonicalBar(
                        asset_class=ASSET_CLASS_GOLD_TRADFI, venue="yahoo", symbol="GLD",
                        timeframe_h=1.0, ts_event_ms=t0, ts_avail_ms=t0 + HOUR_MS,
                        open=250.0, high=251.0, low=249.0, close=250.5, volume_usd=1e6,
                        session_tag="US_EQUITY_RTH",
                    )
                )
            for slot in (12, 16):
                t0 = day_base + slot * HOUR_MS
                gld_4h.append(
                    CanonicalBar(
                        asset_class=ASSET_CLASS_GOLD_TRADFI, venue="yahoo", symbol="GLD",
                        timeframe_h=4.0, ts_event_ms=t0, ts_avail_ms=t0 + 4 * HOUR_MS + 30 * 60_000,
                        open=250.0, high=251.0, low=249.0, close=250.5, volume_usd=3e6,
                        session_tag="US_EQUITY_RTH",
                    )
                )
        res_1h_24h = resample_bars(gld_1h, 24.0, require_complete=True)
        res_4h_24h = resample_bars(gld_4h, 24.0, require_complete=True)
        self.assertEqual(len(res_1h_24h), 30)
        self.assertEqual(len(res_4h_24h), 30)

        # Mixed timeframe_h input MUST raise ValueError
        mixed_tf = _make_hourly_bars(11, tf_h=1.0) + _make_hourly_bars(1, start_ms=BASE_2025_MS + 12 * HOUR_MS, tf_h=12.0)
        with self.assertRaises(ValueError):
            resample_bars(mixed_tf, 12.0)

    def test_fix7_paxg_basis_index_and_duckdb_external_io_sandbox(self) -> None:
        # 1. Default load_paxg_basis query plan has ZERO USE TEMP B-TREE FOR ORDER BY
        with tempfile.TemporaryDirectory() as td, DataFoundationRouter(Path(td) / "v4") as router:
            pt = PAXGBasisPoint(
                venue_perp="hyperliquid", venue_spot="binance_spot", ref_tradfi_symbol="GC=F",
                timeframe_sec=14_400, ts_event_ms=BASE_2025_MS, ts_avail_ms=BASE_2025_MS + 4 * HOUR_MS,
                paxg_perp_close=3000.0, paxg_spot_close=2998.0, gc_ref_close=2995.0,
            )
            router.upsert_paxg_basis([pt])
            plan = router.gold_macro_store.conn.fetch_all(
                "EXPLAIN QUERY PLAN SELECT * FROM paxg_basis_arbitration "
                "WHERE venue_perp=? AND timeframe_sec=? AND ref_tradfi_symbol=? "
                "ORDER BY ts_event_ms ASC",
                ("hyperliquid", 14_400, "GC=F"),
            )
            detail = " | ".join(str(r["detail"]) for r in plan)
            self.assertIn("idx_paxg_basis_ev", detail)
            self.assertNotIn("USE TEMP B-TREE FOR ORDER BY", detail)

            # 2. DuckDB external I/O & system-introspection functions blocked by validate_readonly_sql
            for attack_sql in (
                "SELECT * FROM read_text('/etc/passwd')",
                "SELECT * FROM read_csv_auto('/etc/passwd')",
                "SELECT * FROM read_json_auto('/etc/passwd')",
                "SELECT * FROM read_parquet('/tmp/x.parquet')",
                "SELECT * FROM glob('/etc/*')",
                "SELECT * FROM duckdb_secrets()",
                "SELECT getenv('HOME')",
                "SELECT * FROM sqlite_scan('/tmp/a.db', 't')",
                "SELECT * FROM query_table('web3_bars')",
            ):
                with self.assertRaises(ValueError, msg=f"Attack SQL must be blocked: {attack_sql}"):
                    validate_readonly_sql(attack_sql)

            # 3. Verify query_domain_readonly_duckdb uses literal escaped view paths + SET enable_external_access = false
            router.export_domain_partition("gold_macro", fmt="auto")
            executed_sqls: list[str] = []

            class _FakeCur:
                description = [("c",)]

                @staticmethod
                def fetchall() -> list[tuple[int]]:
                    return [(1,)]

            class _FakeDuckCon:
                def execute(self, sql_text: str, *args: Any) -> _FakeCur:
                    if args:
                        raise AssertionError("DuckDB DDL views must not use ? parameters")
                    executed_sqls.append(sql_text)
                    return _FakeCur()

                def close(self) -> None:
                    return None

            class _FakeDuckMod:
                @staticmethod
                def connect(target: str) -> _FakeDuckCon:
                    assert target == ":memory:"
                    return _FakeDuckCon()

            with patch.object(_router_mod, "_HAS_DUCKDB", True), patch.object(_router_mod, "duckdb", _FakeDuckMod):
                res = router.query_domain_readonly_duckdb("gold_macro", "SELECT COUNT(*) AS c FROM paxg_basis_arbitration")
            self.assertEqual(res, [{"c": 1}])
            self.assertIn("SET enable_external_access = false", executed_sqls)
            lock_idx = executed_sqls.index("SET enable_external_access = false")
            self.assertGreater(lock_idx, 0)
            self.assertEqual(executed_sqls[lock_idx + 1], "SELECT COUNT(*) AS c FROM paxg_basis_arbitration")

    def test_fix8_fix9_fix10_hl_partial_trim_empty_funding_defillama_fomc_bls(self) -> None:
        now_ms = BASE_2025_MS + 96 * HOUR_MS
        # FIX-8A: HL partial trailing 12h bar trimmed when funding ends at t0+24h
        raw_hl = [
            {"t": BASE_2025_MS + i * 12 * HOUR_MS, "T": BASE_2025_MS + (i + 1) * 12 * HOUR_MS - 1,
             "o": "100", "h": "105", "l": "99", "c": "104", "v": "50.0", "n": 100}
            for i in range(3)
        ]
        fmap_25 = {BASE_2025_MS + h * HOUR_MS: (0.0001, 1.0, 0.0002) for h in range(25)}
        hl_trimmed = HyperliquidWeb3Collector.build_canonical_bars(
            "BTC", "12h", raw_hl, fmap_25, trim_to_funding_span=True
        )
        self.assertEqual(len(hl_trimmed), 2)
        self.assertAlmostEqual(hl_trimmed[0].carry_interval_h, 12.0, places=6)
        self.assertAlmostEqual(hl_trimmed[1].carry_interval_h, 12.0, places=6)

        # FIX-8B: Empty funding_map with trim_to_funding_span=True is strictly fail-closed across ALL 4 perp collectors
        self.assertEqual(HyperliquidWeb3Collector.build_canonical_bars("BTC", "12h", raw_hl, {}, trim_to_funding_span=True), [])
        self.assertEqual(len(HyperliquidWeb3Collector.build_canonical_bars("BTC", "12h", raw_hl, {}, trim_to_funding_span=False)), 3)
        self.assertEqual(
            BinanceVisionWeb3Collector.assemble_um_bars(
                canonical_sym="BTC", interval="12h",
                klines={BASE_2025_MS: (100.0, 105.0, 99.0, 104.0, 1e6)},
                funding_map={}, premium_map={}, trim_to_funding_span=True,
            ),
            [],
        )
        self.assertEqual(
            GateWeb3Collector.parse_candles_and_funding(
                canonical_sym="BTC", interval="12h",
                raw_candles=[{"t": BASE_2025_MS // 1000, "o": "100", "h": "105", "l": "99", "c": "104", "v": 10, "sum": "10000"}],
                raw_funding=[], trim_to_funding_span=True, now_ms=now_ms,
            ),
            [],
        )
        self.assertEqual(
            OKXWeb3Collector.parse_candles_and_funding(
                canonical_sym="BTC", interval="12h",
                raw_candles=[[str(BASE_2025_MS), "100", "105", "99", "104", "10", "1", "10000", "1"]],
                raw_funding=[], trim_to_funding_span=True, now_ms=now_ms,
            ),
            [],
        )

        # FIX-9: DefiLlama non-numeric peg entries ("N/A", {}, True) do NOT crash parser
        llama_pts = DefiLlamaStablecoinCollector.parse_stablecoin_rows(
            [
                {
                    "date": str(BASE_2025_MS // 1000),
                    "totalCirculatingUSD": {"peggedUSD": 200e9, "peggedBAD": "N/A", "peggedOBJ": {}, "peggedBOOL": True},
                }
            ],
            now_ms=now_ms,
        )
        self.assertEqual(len(llama_pts), 1)
        self.assertAlmostEqual(llama_pts[0].value, 200e9, places=1)

        # FIX-10A: Fed FOMC <sup>1</sup> footnote stripped before day parsing
        fomc_sup_html = """
        <div class="panel-heading">2026 FOMC Meetings</div>
        <div class="row fomc-meeting">
          <div class="fomc-meeting__month"><strong>June</strong></div>
          <div class="fomc-meeting__date">14-15<sup>1</sup>*</div>
        </div>
        """
        evs_sup = FedFOMCCalendarCollector.parse_fomc_html(fomc_sup_html)
        self.assertEqual(len(evs_sup), 1)
        self.assertEqual(evs_sup[0].event_id, "FOMC-2026-06-15")
        self.assertTrue(evs_sup[0].is_sep_meeting)

        # FIX-10B: BLS raw_prior tightened to 20 * DAY_MS historical replay guard
        rel_dec_nfp = ny_local_to_utc_ms(2026, 12, 4, 8, 30)
        bls_min = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [
                    {
                        "seriesID": "CES0000000001",
                        "data": [
                            {"year": "2026", "period": "M09", "value": "159128", "footnotes": [{}]},
                            {"year": "2026", "period": "M10", "value": "159293", "footnotes": [{}]},
                        ],
                    }
                ]
            },
        }
        evs_25d_early = BLSMacroEventCollector.build_events_from_payload(
            bls_min, schedule_start_year=2026, schedule_end_year=2026, now_ms=rel_dec_nfp - 25 * DAY_MS
        )
        dec_early = next(e for e in evs_25d_early if e.event_id == "NFP-2026-12-04")
        self.assertIsNone(dec_early.prior)
        evs_15d_early = BLSMacroEventCollector.build_events_from_payload(
            bls_min, schedule_start_year=2026, schedule_end_year=2026, now_ms=rel_dec_nfp - 15 * DAY_MS
        )
        dec_near = next(e for e in evs_15d_early if e.event_id == "NFP-2026-12-04")
        self.assertAlmostEqual(dec_near.prior or 0.0, 165.0, places=2)

    def test_def03_runner_cumulative_grid_trials_advance_9_18_27_and_penalize_dsr(self) -> None:
        """DEF-03 regression: 3 consecutive 9-point grid runs strictly advance trial_ledger 9 -> 18 -> 27."""
        import importlib.util

        from alphaloop.platform.assets import default_web3_specs
        from alphaloop.platform.core_engine.risk_allocator import RiskAllocatorConfig
        from alphaloop.platform.core_engine.validation import compute_buy_and_hold_returns
        from alphaloop.platform.strategies.web3 import Web3AdaptiveTrendKernel

        runner_path = Path(__file__).resolve().parent.parent / "scripts" / "run_strategy_platform_v4.py"
        spec = importlib.util.spec_from_file_location("_test_v4_runner_def03", runner_path)
        assert spec is not None and spec.loader is not None
        runner_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner_mod)

        bars = _make_hourly_bars(
            260,
            tf_h=12.0,
            symbol="BTC",
            carry_rate=0.0002,
            carry_interval_h=12.0,
            provenance=PROVENANCE_SYNTHETIC,
        )
        specs = default_web3_specs(("BTC",))
        bars_by_sym = {"BTC": bars}
        risk_cfg = RiskAllocatorConfig(
            max_positions=1,
            target_annual_vol_pct=18.0,
            max_portfolio_leverage=1.0,
            max_single_weight=1.0,
        )
        fa = runner_mod._compute_factor_audit("def03_track", bars_by_sym, specs, horizon_bars=2)
        bh_rets = compute_buy_and_hold_returns(bars)
        grid_factories: list[tuple[int, int, str, dict[str, Any], Any]] = []
        for ix, dn in enumerate((14, 20, 28)):
            for iy, atr in enumerate((2.5, 3.0, 3.8)):
                grid_factories.append(
                    (
                        ix,
                        iy,
                        f"dn{dn}_atr{int(atr * 10)}",
                        {"donchian_entry_n": dn, "donchian_exit_n": dn // 2, "atr_stop_mult": atr},
                        lambda _dn=dn, _atr=atr: Web3AdaptiveTrendKernel(
                            donchian_entry_n=_dn, donchian_exit_n=_dn // 2, atr_stop_mult=_atr
                        ),
                    )
                )

        with tempfile.TemporaryDirectory(prefix="def03_trials_") as tmpdir:
            router = DataFoundationRouter(Path(tmpdir) / "lake")
            try:
                fam = "def03_regression_family"
                self.assertEqual(router.get_family_trial_count(fam), 0)
                sr_stars: list[float] = []
                counts: list[int] = []
                for run_idx in range(3):
                    _sim, _gate, detail = runner_mod._evaluate_strategy_full(
                        router=router,
                        strategy_id="W2_def03_test",
                        family_id=fam,
                        track_name="web3_def03",
                        asset_domain="web3",
                        specs=specs,
                        bars_by_symbol=bars_by_sym,
                        dq_reports=[],
                        factor_audit=fa,
                        baseline_factory=lambda: Web3AdaptiveTrendKernel(),
                        grid_factories=grid_factories,
                        risk_cfg=risk_cfg,
                        periods_per_year=730.0,
                        benchmark_returns=bh_rets,
                        benchmark_name="BTC_BH",
                        capital_usd=10_000.0,
                        parity_tail_bars=120,
                        baseline_grid_label="dn20_atr30",
                    )
                    self.assertEqual(detail["trials_recorded_in_run"], 9)
                    counts.append(router.get_family_trial_count(fam))
                    self.assertEqual(detail["cumulative_family_trials"], counts[-1])
                    self.assertEqual(int(detail["g5_dsr"]["family_trial_count"]), 9 * (run_idx + 1))
                    sr_stars.append(float(detail["g5_dsr"]["sr_star_annualized"]))
                    self.assertEqual(counts[-1], 9 * (run_idx + 1))

                self.assertEqual(counts, [9, 18, 27])
                self.assertLess(sr_stars[0], sr_stars[1])
                self.assertLess(sr_stars[1], sr_stars[2])

                # SQLite immutability triggers still block UPDATE/DELETE across all 27 rows
                trials = router.list_family_trials(fam)
                self.assertEqual(len(trials), 27)
                some_tid = trials[0]["trial_id"]
                with self.assertRaises(sqlite3.IntegrityError):
                    router.catalog_conn.execute("UPDATE trial_ledger SET verdict='FAKE' WHERE trial_id=?", (some_tid,))
                with self.assertRaises(sqlite3.IntegrityError):
                    router.catalog_conn.execute("DELETE FROM trial_ledger WHERE trial_id=?", (some_tid,))
            finally:
                router.close()

    def test_phase_b1_duckdb_temp_table_materialization_and_obs02_path_literal_guard(self) -> None:
        """Phase B1 (REV-017 + OBS-02): TEMP TABLE materialization before sandbox lock & FROM/JOIN path block."""
        # 1. OBS-02 static guard blocks quoted file-path literals after FROM / JOIN
        for attack_sql in (
            "SELECT * FROM '/etc/passwd'",
            "SELECT * FROM \"/etc/passwd\"",
            "SELECT * FROM web3_bars JOIN '/tmp/leak.parquet' ON 1=1",
            "SELECT * FROM web3_bars LEFT JOIN '/etc/hosts' AS h ON 1=1",
            "WITH x AS (SELECT * FROM '/var/secret.csv') SELECT * FROM x",
        ):
            with self.assertRaises(ValueError, msg=f"OBS-02 must reject path literal SQL: {attack_sql}"):
                validate_readonly_sql(attack_sql)

        # Legitimate string predicates in WHERE remain permitted
        validate_readonly_sql("SELECT COUNT(*) AS n FROM web3_bars WHERE symbol = 'BTC' AND venue = 'hyperliquid'")
        validate_readonly_sql("WITH cte AS (SELECT symbol, close FROM web3_bars WHERE symbol = 'ETH') SELECT COUNT(*) FROM cte")

        # 2. Verify CREATE TEMP TABLE materialization runs strictly BEFORE SET enable_external_access = false
        with tempfile.TemporaryDirectory(prefix="phase_b1_duckdb_") as td, DataFoundationRouter(Path(td) / "v4") as router:
            pt = PAXGBasisPoint(
                venue_perp="hyperliquid", venue_spot="binance_spot", ref_tradfi_symbol="GC=F",
                timeframe_sec=14_400, ts_event_ms=BASE_2025_MS, ts_avail_ms=BASE_2025_MS + 4 * HOUR_MS,
                paxg_perp_close=3000.0, paxg_spot_close=2998.0, gc_ref_close=2995.0,
            )
            router.upsert_paxg_basis([pt])
            router.export_domain_partition("gold_macro", fmt="jsonl.gz")

            # Empty jsonl.gz files must be skipped gracefully rather than crashing DuckDB schema inference
            executed_sqls: list[str] = []

            class _FakeCur:
                description = [("c",)]

                @staticmethod
                def fetchall() -> list[tuple[int]]:
                    return [(1,)]

            class _FakeDuckCon:
                def execute(self, sql_text: str, *args: Any) -> _FakeCur:
                    if args:
                        raise AssertionError("DuckDB DDL must not use positional ? params")
                    executed_sqls.append(sql_text)
                    return _FakeCur()

                def close(self) -> None:
                    return None

            class _FakeDuckMod:
                @staticmethod
                def connect(target: str) -> _FakeDuckCon:
                    assert target == ":memory:"
                    return _FakeDuckCon()

            with patch.object(_router_mod, "_HAS_DUCKDB", True), patch.object(_router_mod, "duckdb", _FakeDuckMod):
                res = router.query_domain_readonly_duckdb(
                    "gold_macro",
                    "SELECT COUNT(*) AS c FROM paxg_basis_arbitration",
                )
            self.assertEqual(res, [{"c": 1}])
            self.assertIn("SET enable_external_access = false", executed_sqls)
            lock_idx = executed_sqls.index("SET enable_external_access = false")
            pre_lock_sqls = executed_sqls[:lock_idx]
            self.assertGreater(len(pre_lock_sqls), 0)
            for ddl in pre_lock_sqls:
                self.assertIn("CREATE TEMP TABLE ", ddl)
                self.assertNotIn("CREATE OR REPLACE VIEW", ddl)
            self.assertTrue(any("paxg_basis_arbitration" in ddl for ddl in pre_lock_sqls))
            # Empty tables (zero rows -> 0-byte uncompressed jsonl.gz) were cleanly skipped
            self.assertFalse(any("macro_events" in ddl for ddl in pre_lock_sqls))
            self.assertEqual(executed_sqls[lock_idx + 1], "SELECT COUNT(*) AS c FROM paxg_basis_arbitration")

            # 3. Incremental re-export replaces superseded dataset_manifests rows 1:1 with on-disk SHA-256
            pt2 = PAXGBasisPoint(
                venue_perp="hyperliquid", venue_spot="binance_spot", ref_tradfi_symbol="GC=F",
                timeframe_sec=14_400, ts_event_ms=BASE_2025_MS + 4 * HOUR_MS, ts_avail_ms=BASE_2025_MS + 8 * HOUR_MS,
                paxg_perp_close=3005.0, paxg_spot_close=3002.0, gc_ref_close=3000.0,
            )
            router.upsert_paxg_basis([pt2])
            router.export_domain_partition("gold_macro", fmt="jsonl.gz")
            manifests = router.list_dataset_manifests("gold_macro")
            rel_paths = [m["rel_path"] for m in manifests]
            self.assertEqual(len(rel_paths), len(set(rel_paths)), "Stale duplicate rel_path in dataset_manifests")
            for m in manifests:
                self.assertEqual(sha256_file(router.base_dir / m["rel_path"]), m["sha256"])


if __name__ == "__main__":
    unittest.main()
