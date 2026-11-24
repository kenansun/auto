"""Unit & integration tests for M3 STD-46 hardening across BBO, W3-DN Shadow, Shadow Runner & Core Audit Layer.

Strictly 100% local & offline (zero network calls, zero live execution, zero remote mutations).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from alphaloop.core.models import BBOTick, FundingRateBar, StrategyConfig
from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.daemon.shadow_runner import (
    DAEMON_LIVE_EXECUTION_ENABLED,
    PARITY_DEPTH_ACTIVE,
    PARITY_DEPTH_VACUOUS,
    SHADOW_LIVE_EXECUTION_ENABLED as RUNNER_SHADOW_LIVE_ENABLED,
    ShadowParityAuditor,
    ShadowPortfolioDaemon,
    validate_prewindow_funding_warmup,
)
from alphaloop.data.bbo_collector import (
    TS_SOURCE_RECV_LOCAL,
    TS_SOURCE_VENUE,
    MultiVenueBBOCollector,
    TimestampedBBOTick,
    _http_json,
)
from alphaloop.data.store import MarketDataStore
from alphaloop.execution.bridge import LIVE_EXECUTION_ENABLED
from alphaloop.platform.assets import ASSET_CLASS_CRYPTO_PERP, ASSET_CLASS_CRYPTO_SPOT
from alphaloop.platform.core_engine import (
    audit_composite_portfolio,
    build_std46_audit_report,
    compute_dual_sharpe,
    evaluate_static_ablation,
)
from alphaloop.platform.data_foundation.models import CanonicalBar
from alphaloop.platform.shadow.w3_dn_shadow_monitor import (
    SHADOW_LIVE_EXECUTION_ENABLED,
    SHADOW_STATUS_ACTIVE,
    SHADOW_STATUS_VACUOUS,
    PairMarketSnapshot,
    PublicRestWaterfallQuoteProvider,
    ReplayQuoteProvider,
    ShadowMonitorConfig,
    W3DNShadowMonitor,
)
from alphaloop.platform.strategies.web3.delta_neutral_carry import (
    Web3DeltaNeutralCarryKernel,
    perp_leg_symbol,
    spot_leg_symbol,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCHQ_GATE_SCRIPT = (
    _REPO_ROOT.parent / "Strategic-Command-Headquarters" / "scripts" / "verify_data_and_strategy_gate.py"
)
_VALIDATION_PY = _REPO_ROOT / "alphaloop" / "platform" / "core_engine" / "validation.py"
_VALIDATION_EXPECTED_SIZE = 92878
_VALIDATION_EXPECTED_SHA256 = "98b89c001769f1c0feb09247e90d12c40e3a097d79e31c2e026c76817571c231"


def _make_pair_snap(
    coin: str,
    ts_ms: int,
    *,
    spot_mid: float = 60_000.0,
    basis_bps: float = 2.0,
    funding_1h: float = 0.0001,
    spot_spread_bps: float = 1.0,
    perp_spread_bps: float = 1.0,
    perp_l2_slip_bps: float | None = None,
    dvol: float = 55.0,
) -> PairMarketSnapshot:
    s_half = spot_mid * (spot_spread_bps / 20_000.0)
    perp_mid = spot_mid * (1.0 + basis_bps / 10_000.0)
    p_half = perp_mid * (perp_spread_bps / 20_000.0)

    if perp_l2_slip_bps is not None:
        # Single deep L2 level at exact slip distance from perp_mid
        ask_px = perp_mid * (1.0 + perp_l2_slip_bps / 10_000.0)
        bid_px = perp_mid * (1.0 - perp_l2_slip_bps / 10_000.0)
        qty = 1_000_000.0 / perp_mid
        l2_asks = ((ask_px, qty),)
        l2_bids = ((bid_px, qty),)
    else:
        qty = 1_000_000.0 / perp_mid
        l2_asks = ((perp_mid + p_half, qty),)
        l2_bids = ((perp_mid - p_half, qty),)

    return PairMarketSnapshot(
        coin=coin,
        ts_observe_ms=ts_ms,
        spot_bid=spot_mid - s_half,
        spot_ask=spot_mid + s_half,
        spot_bid_qty=10.0,
        spot_ask_qty=10.0,
        spot_ts_ms=ts_ms,
        perp_bid=perp_mid - p_half,
        perp_ask=perp_mid + p_half,
        perp_mark=perp_mid,
        perp_oracle=perp_mid,
        perp_funding_1h=funding_1h,
        perp_ts_ms=ts_ms,
        perp_l2_bids=l2_bids,
        perp_l2_asks=l2_asks,
        dvol=dvol,
        spot_source="test_spot",
        perp_source="test_perp",
        provenance=PROVENANCE_REAL,
    )


class TestBBOCollectorSTD46Hardening(unittest.TestCase):
    """F5.2: finiteness guards across all 4 venues, Lighter level safety, allow_nan=False & dual timestamps."""

    def test_http_json_rejects_nan_and_inf_payload(self) -> None:
        with self.assertRaises(ValueError):
            _http_json("https://example.invalid/info", payload={"x": float("nan")}, max_retries=1)
        with self.assertRaises(ValueError):
            _http_json("https://example.invalid/info", payload={"x": float("inf")}, max_retries=1)

    def test_okx_finiteness_and_dual_timestamps(self) -> None:
        collector = MultiVenueBBOCollector(["BTC", "ETH", "SOL", "DOGE", "LINK", "AVAX", "UNI", "NEAR"])
        exch_ts = 1_795_000_123_456
        payload = {
            "data": [
                # Valid with explicit exchange timestamp
                {"instId": "BTC-USDT-SWAP", "bidPx": "60000.0", "askPx": "60001.5", "bidSz": "5", "askSz": "6", "ts": str(exch_ts)},
                # Valid with missing exchange timestamp -> RECV_LOCAL_ONLY
                {"instId": "ETH-USDT-SWAP", "bidPx": "2500.0", "askPx": "2500.5", "bidSz": "", "askSz": None, "ts": ""},
                # Poisoned cases: NaN, +inf, crossed, bool, negative qty, overflow 1e308
                {"instId": "SOL-USDT-SWAP", "bidPx": "NaN", "askPx": "150.0", "bidSz": "1", "askSz": "1", "ts": str(exch_ts)},
                {"instId": "DOGE-USDT-SWAP", "bidPx": "0.2", "askPx": "Infinity", "bidSz": "1", "askSz": "1", "ts": str(exch_ts)},
                {"instId": "LINK-USDT-SWAP", "bidPx": "20.0", "askPx": "19.5", "bidSz": "1", "askSz": "1", "ts": str(exch_ts)},
                {"instId": "AVAX-USDT-SWAP", "bidPx": True, "askPx": 2.0, "bidSz": "1", "askSz": "1", "ts": str(exch_ts)},
                {"instId": "UNI-USDT-SWAP", "bidPx": "10.0", "askPx": "10.1", "bidSz": "-1", "askSz": "1", "ts": str(exch_ts)},
                {"instId": "NEAR-USDT-SWAP", "bidPx": "1e308", "askPx": "1e308", "bidSz": "1", "askSz": "1", "ts": str(exch_ts)},
            ]
        }
        with patch("alphaloop.data.bbo_collector._http_json", return_value=(payload, 12.5)):
            ticks = collector.fetch_okx_swap_bbo()

        self.assertEqual([t.symbol for t in ticks], ["BTC", "ETH"])
        btc, eth = ticks[0], ticks[1]
        self.assertIsInstance(btc, BBOTick)
        self.assertIsInstance(btc, TimestampedBBOTick)
        self.assertEqual(btc.venue_ts_ms, exch_ts)
        self.assertGreater(btc.recv_ts_ms, 0)
        self.assertEqual(btc.ts_source, TS_SOURCE_VENUE)
        self.assertEqual(btc.ts_ms, exch_ts)
        btc_dict = btc.to_dict()
        self.assertEqual(btc_dict["venue_ts_ms"], exch_ts)
        self.assertEqual(btc_dict["ts_source"], TS_SOURCE_VENUE)

        self.assertIsNone(eth.venue_ts_ms)
        self.assertGreater(eth.recv_ts_ms, 0)
        self.assertEqual(eth.ts_source, TS_SOURCE_RECV_LOCAL)
        self.assertEqual(eth.ts_ms, eth.recv_ts_ms)

        # Store round-trip compatibility
        with tempfile.TemporaryDirectory(prefix="bbo_store_") as td:
            store = MarketDataStore(Path(td) / "market.sqlite")
            try:
                inserted = store.upsert_bbo_ticks(ticks)
                self.assertEqual(inserted, 2)
                loaded = store.load_latest_bbo_tick("okx", "BTC")
                self.assertIsNotNone(loaded)
                self.assertAlmostEqual(loaded.bid, 60000.0)
                self.assertEqual(loaded.ts_ms, exch_ts)
            finally:
                store.close()

    def test_gate_and_hyperliquid_finiteness_and_timestamps(self) -> None:
        collector = MultiVenueBBOCollector(["BTC", "ETH", "SOL"])
        gate_payload = [
            {"contract": "BTC_USDT", "highest_bid": "60000", "lowest_ask": "60002", "highest_size": "3", "lowest_size": "4"},
            {"contract": "ETH_USDT", "highest_bid": "inf", "lowest_ask": "inf", "highest_size": "3", "lowest_size": "4"},
            {"contract": "SOL_USDT", "highest_bid": "100", "lowest_ask": "99", "highest_size": "3", "lowest_size": "4"},
        ]
        with patch("alphaloop.data.bbo_collector._http_json", return_value=(gate_payload, 8.0)):
            g_ticks = collector.fetch_gate_futures_bbo()
        self.assertEqual(len(g_ticks), 1)
        self.assertEqual(g_ticks[0].symbol, "BTC")
        self.assertIsNone(g_ticks[0].venue_ts_ms)
        self.assertEqual(g_ticks[0].ts_source, TS_SOURCE_RECV_LOCAL)
        self.assertEqual(g_ticks[0].ts_ms, g_ticks[0].recv_ts_ms)

        hl_ts = 1_795_000_555_000
        hl_responses = {
            "BTC": ({"levels": [[{"px": "60010", "sz": "2.0"}], [{"px": "60012", "sz": "1.5"}]], "time": hl_ts}, 9.2),
            "ETH": ({"levels": [[{"px": "2500", "sz": "NaN"}], [{"px": "2501", "sz": "1.0"}]], "time": hl_ts}, 9.2),
            "SOL": ({"levels": [[{"px": "150", "sz": "5.0"}], [{"px": "150.1", "sz": "4.0"}]], "time": None}, 9.2),
        }

        def _fake_hl(_url: str, payload: dict[str, Any] | None = None, **_kw: Any) -> tuple[Any, float]:
            assert payload is not None
            return hl_responses[payload["coin"]]

        with patch("alphaloop.data.bbo_collector._http_json", side_effect=_fake_hl):
            hl_ticks = collector.fetch_hyperliquid_bbo()
        self.assertEqual([t.symbol for t in hl_ticks], ["BTC", "SOL"])
        self.assertEqual(hl_ticks[0].venue_ts_ms, hl_ts)
        self.assertEqual(hl_ticks[0].ts_source, TS_SOURCE_VENUE)
        self.assertIsNone(hl_ticks[1].venue_ts_ms)
        self.assertEqual(hl_ticks[1].ts_source, TS_SOURCE_RECV_LOCAL)

    def test_lighter_rejects_multilevel_nan_and_inf_poisoning(self) -> None:
        collector = MultiVenueBBOCollector(["BTC", "ETH", "SOL", "AVAX"])
        collector._lighter_markets = {"BTC": 1, "ETH": 2, "SOL": 3, "AVAX": 4}
        books = {
            1: (
                {
                    "bids": [{"price": "59990", "remaining_base_amount": "1"}, {"price": "60000", "remaining_base_amount": "2"}],
                    "asks": [{"price": "60005", "remaining_base_amount": "1"}, {"price": "60001", "remaining_base_amount": "3"}],
                },
                7.5,
            ),
            # NaN in second bid level MUST poison-reject (cannot silently pick 100 via Python max!)
            2: (
                {
                    "bids": [{"price": "2500", "remaining_base_amount": "1"}, {"price": "NaN", "remaining_base_amount": "1"}],
                    "asks": [{"price": "2501", "remaining_base_amount": "1"}],
                },
                7.5,
            ),
            # +inf in bid level rejected
            3: (
                {
                    "bids": [{"price": "150", "remaining_base_amount": "1"}, {"price": "inf", "remaining_base_amount": "1"}],
                    "asks": [{"price": "151", "remaining_base_amount": "1"}],
                },
                7.5,
            ),
            # NaN in ask level rejected
            4: (
                {
                    "bids": [{"price": "25", "remaining_base_amount": "1"}],
                    "asks": [{"price": "NaN", "remaining_base_amount": "1"}, {"price": "26", "remaining_base_amount": "1"}],
                },
                7.5,
            ),
        }

        def _fake_lighter(url: str, **_kw: Any) -> tuple[Any, float]:
            for mid, resp in books.items():
                if f"market_id={mid}&" in url:
                    return resp
            raise AssertionError(f"unexpected url {url}")

        with patch("alphaloop.data.bbo_collector._http_json", side_effect=_fake_lighter):
            l_ticks = collector.fetch_lighter_bbo()
        self.assertEqual([t.symbol for t in l_ticks], ["BTC"])
        self.assertAlmostEqual(l_ticks[0].bid, 60000.0)
        self.assertAlmostEqual(l_ticks[0].ask, 60001.0)
        self.assertIsNone(l_ticks[0].venue_ts_ms)
        self.assertEqual(l_ticks[0].ts_source, TS_SOURCE_RECV_LOCAL)

    def test_exported_bbo_dataset_passes_schq_gate_if_present(self) -> None:
        if not _SCHQ_GATE_SCRIPT.is_file():
            self.fail(f"Expected SCHQ gate script missing at {_SCHQ_GATE_SCRIPT}")
        collector = MultiVenueBBOCollector(["BTC"])
        exch_ts = 1_795_000_123_456
        payload = {
            "data": [
                {"instId": "BTC-USDT-SWAP", "bidPx": "60000.0", "askPx": "60001.0", "bidSz": "5", "askSz": "6", "ts": str(exch_ts)},
            ]
        }
        with patch("alphaloop.data.bbo_collector._http_json", return_value=(payload, 10.0)):
            ticks = collector.fetch_okx_swap_bbo()
        self.assertEqual(len(ticks), 1)
        row = ticks[0].to_dict()
        row["ts_avail_ms"] = max(int(row["ts_ms"]), int(row["recv_ts_ms"]))
        with tempfile.TemporaryDirectory(prefix="bbo_gate_") as td:
            p = Path(td) / "bbo.jsonl"
            p.write_text(json.dumps(row, allow_nan=False) + "\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-B", str(_SCHQ_GATE_SCRIPT), "--mode", "dataset", str(p), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"STD-46 dataset gate failed: {proc.stdout}\n{proc.stderr}")


class TestW3DNShadowMonitorSTD46Hardening(unittest.TestCase):
    """F5.3: finite guard, 144h-in-48h pre-warm closure, SHADOW_VACUOUS_ZERO_TRADES & tail-cost telemetry."""

    def test_config_prewarm_knobs_strictly_bool_or_int_and_live_locked(self) -> None:
        self.assertFalse(SHADOW_LIVE_EXECUTION_ENABLED)
        prewarm_fields = [f for f in dataclasses.fields(ShadowMonitorConfig) if "prewarm" in f.name.lower()]
        self.assertTrue(prewarm_fields, "Expected explicit prewarm config knobs on ShadowMonitorConfig")
        with tempfile.TemporaryDirectory(prefix="w3dn_cfg_") as td:
            cfg = ShadowMonitorConfig(state_dir=Path(td))
            for f in prewarm_fields:
                val = getattr(cfg, f.name)
                self.assertIsInstance(val, (bool, int), f"Prewarm knob {f.name} must be bool/int, got {type(val)}")
                self.assertNotIsInstance(val, float)
                self.assertNotIsInstance(val, Path)

    def test_binance_spot_bbo_rejects_nan_inf_bool_and_fails_over(self) -> None:
        calls: list[str] = []

        class _FakeClient:
            def fetch_json(self, url: str, **_kw: Any) -> dict[str, Any]:
                calls.append(url)
                if "host1" in url:
                    return {"bidPrice": "NaN", "askPrice": "60001", "bidQty": "1", "askQty": "1"}
                if "host2" in url:
                    return {"bidPrice": "60000", "askPrice": "Infinity", "bidQty": "1", "askQty": "1"}
                if "host3" in url:
                    return {"bidPrice": True, "askPrice": "60001", "bidQty": "1", "askQty": "1"}
                return {"bidPrice": "60000.0", "askPrice": "60001.0", "bidQty": "2.0", "askQty": "3.0"}

        prov = PublicRestWaterfallQuoteProvider(
            client=_FakeClient(),  # type: ignore[arg-type]
            spot_bases=("https://host1.invalid", "https://host2.invalid", "https://host3.invalid", "https://host4.invalid"),
        )
        res = prov._fetch_binance_spot_bbo("BTC")
        self.assertIsNotNone(res)
        assert res is not None
        self.assertAlmostEqual(res[0], 60000.0)
        self.assertAlmostEqual(res[1], 60001.0)
        self.assertEqual(res[5], "https://host4.invalid")
        self.assertEqual(len(calls), 4)

    def test_vacuous_48h_run_without_prewarm_blocks_ready_for_owner_review(self) -> None:
        # Default 12h bars x 5 bars = 48h (< 12 warmup_bars = 144h) -> 0 trades -> SHADOW_VACUOUS_ZERO_TRADES
        t0 = 1_795_000_000_000
        bar_ms = 12 * 3_600_000
        frames = [{"BTC": _make_pair_snap("BTC", t0 + i * bar_ms, funding_1h=0.0001)} for i in range(5)]
        with tempfile.TemporaryDirectory(prefix="w3dn_vacuous_") as td:
            hb_path = Path(td) / "hb.json"
            cfg = ShadowMonitorConfig(
                state_dir=Path(td),
                coins=("BTC",),
                kernel_bar_interval_sec=43_200,
                required_shadow_hours=48.0,
                enforce_wall_clock_bounds=False,
                p16_heartbeat_path=hb_path,
            )
            with W3DNShadowMonitor(cfg, provider=ReplayQuoteProvider(frames)) as mon:
                for _ in range(5):
                    rec = mon.run_once()
                    self.assertEqual(rec["status"], "HEALTHY")
                rep = mon.evaluate_48h_readiness()
                self.assertGreaterEqual(rep.elapsed_hours, 48.0)
                self.assertEqual(rep.executed_trades, 0)
                self.assertFalse(rep.non_vacuous)
                self.assertFalse(rep.non_vacuous_parity)
                self.assertEqual(rep.shadow_status, SHADOW_STATUS_VACUOUS)
                self.assertEqual(rep.parity_depth, SHADOW_STATUS_VACUOUS)
                self.assertEqual(rep.verdict, SHADOW_STATUS_VACUOUS)
                self.assertNotEqual(rep.verdict, "READY_FOR_OWNER_REVIEW")
                self.assertTrue(any("SHADOW_VACUOUS_ZERO_TRADES" in b for b in rep.blocking_reasons))

                hb = json.loads(hb_path.read_text(encoding="utf-8"))
                self.assertEqual(hb["status"], "HEALTHY")
                self.assertEqual(hb["executed_trades"], 0)
                self.assertFalse(hb["non_vacuous_parity"])
                self.assertEqual(hb["shadow_status"], SHADOW_STATUS_VACUOUS)

    def test_prewarm_closes_144h_in_48h_gap_idempotent_and_restart_safe(self) -> None:
        t0 = 1_795_000_000_000
        bar_ms = 12 * 3_600_000
        hist_frames = [
            {"BTC": _make_pair_snap("BTC", t0 - (12 - i) * bar_ms, funding_1h=0.0001)}
            for i in range(12)
        ]
        live_frames = [
            {"BTC": _make_pair_snap("BTC", t0 + i * bar_ms, funding_1h=0.0001)}
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory(prefix="w3dn_prewarm_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td),
                coins=("BTC",),
                kernel_bar_interval_sec=43_200,
                required_shadow_hours=48.0,
                enforce_wall_clock_bounds=False,
                enable_prewarm=True,
                prewarm_bars=12,
            )
            with W3DNShadowMonitor(cfg, provider=ReplayQuoteProvider([])) as mon1:
                pw1 = mon1.prewarm_from_snapshots(hist_frames)
                self.assertEqual(pw1["prewarmed_bars_added"], 12)
                self.assertEqual(pw1["prewarmed_bars_total"], 12)
                self.assertTrue(pw1["kernel_warm_ready"])
                self.assertEqual(pw1["executed_trades"], 0)
                self.assertEqual(pw1["total_ticks"], 0)
                self.assertEqual(mon1._total_ticks, 0)
                self.assertIsNone(mon1._start_ts_ms)
                self.assertIsNone(mon1._last_kernel_bucket)
                self.assertAlmostEqual(mon1._cash_usd, 10_000.0)
                self.assertEqual(mon1.verify_audit_chain(), (True, 0))

                # Calling again with identical historical frames is strictly idempotent
                pw2 = mon1.prewarm_from_snapshots(hist_frames)
                self.assertEqual(pw2["prewarmed_bars_added"], 0)
                self.assertEqual(pw2["prewarmed_bars_total"], 12)

            # Restart on same state_dir preserves prewarmed kernel state -> live bar #1 immediately trades!
            with W3DNShadowMonitor(cfg, provider=ReplayQuoteProvider(live_frames)) as mon2:
                self.assertEqual(mon2._prewarmed_bars, 12)
                mon2.run_once()
                self.assertTrue(mon2._positions["BTC"].is_open, "Prewarmed W3-DN must open virtual trade on live bar #1")
                self.assertEqual(mon2._executed_trades, 1)
                for _ in range(4):
                    mon2.run_once()
                rep = mon2.evaluate_48h_readiness()
                self.assertEqual(rep.verdict, "READY_FOR_OWNER_REVIEW")
                self.assertTrue(rep.non_vacuous)
                self.assertTrue(rep.non_vacuous_parity)
                self.assertEqual(rep.shadow_status, SHADOW_STATUS_ACTIVE)
                self.assertEqual(rep.parity_depth, SHADOW_STATUS_ACTIVE)
                self.assertGreaterEqual(rep.executed_trades, 1)
                self.assertEqual(rep.prewarmed_bars, 12)
                self.assertEqual(rep.total_ticks, 5)
                self.assertTrue(rep.audit_chain_intact)
                self.assertEqual(rep.audit_chain_length, 5)

    def test_prewarm_input_shapes_init_kwarg_and_anti_poisoning(self) -> None:
        t0 = 1_795_000_000_000
        bar_ms = 12 * 3_600_000
        # Shape B (dict[str, dict]) + poisoned frames mixed in
        shape_b = [
            {
                "BTC": {
                    "coin": "BTC",
                    "ts_observe_ms": t0 - (4 - i) * bar_ms,
                    "spot_mid": 60_000.0,
                    "perp_mid": 60_012.0,
                    "perp_funding_1h": 0.0001,
                }
            }
            for i in range(3)
        ]
        # Poison: NaN funding, crossed book, unpaired leg
        shape_b.append(
            {
                "BTC": {
                    "coin": "BTC",
                    "ts_observe_ms": t0 - bar_ms,
                    "spot_bid": 60_010.0,
                    "spot_ask": 60_000.0,  # crossed!
                    "perp_bid": 60_000.0,
                    "perp_ask": 60_005.0,
                    "perp_funding_1h": 0.0001,
                }
            }
        )
        shape_b.append(
            {
                "BTC": {
                    "coin": "BTC",
                    "ts_observe_ms": t0 - bar_ms + 1000,
                    "spot_bid": float("nan"),
                    "spot_ask": 60_005.0,
                    "perp_bid": 60_000.0,
                    "perp_ask": 60_005.0,
                    "perp_funding_1h": 0.0001,
                }
            }
        )

        with tempfile.TemporaryDirectory(prefix="w3dn_shapes_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td) / "s1",
                coins=("BTC",),
                kernel_bar_interval_sec=43_200,
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(
                cfg,
                provider=ReplayQuoteProvider([]),
                kernel=Web3DeltaNeutralCarryKernel(warmup_bars=3),
                prewarm_frames=shape_b,
            ) as m_init:
                self.assertEqual(m_init._prewarmed_bars, 3)
                self.assertTrue(m_init._is_kernel_warm_ready())
                self.assertEqual(m_init._total_ticks, 0)

            # Shape C (flat list of PairMarketSnapshot) + Shape D (CanonicalBar dicts)
            cfg2 = ShadowMonitorConfig(
                state_dir=Path(td) / "s2",
                coins=("BTC",),
                kernel_bar_interval_sec=43_200,
                enforce_wall_clock_bounds=False,
            )
            flat_snaps = [_make_pair_snap("BTC", t0 - (4 - i) * bar_ms) for i in range(2)]
            bar_ts = t0 - bar_ms
            bar_frames = [
                {
                    spot_leg_symbol("BTC"): CanonicalBar(
                        asset_class=ASSET_CLASS_CRYPTO_SPOT,
                        venue="binance_spot",
                        symbol=spot_leg_symbol("BTC"),
                        timeframe_h=12.0,
                        timeframe_sec=43_200,
                        ts_event_ms=bar_ts,
                        ts_avail_ms=bar_ts,
                        open=60_000.0,
                        high=60_005.0,
                        low=59_995.0,
                        close=60_000.0,
                        volume_usd=1_000_000.0,
                        carry_rate_period=0.0,
                        carry_interval_h=12.0,
                        spread_bps=1.0,
                        provenance=PROVENANCE_REAL,
                    ),
                    perp_leg_symbol("BTC"): CanonicalBar(
                        asset_class=ASSET_CLASS_CRYPTO_PERP,
                        venue="hyperliquid",
                        symbol=perp_leg_symbol("BTC"),
                        timeframe_h=12.0,
                        timeframe_sec=43_200,
                        ts_event_ms=bar_ts,
                        ts_avail_ms=bar_ts,
                        open=60_010.0,
                        high=60_015.0,
                        low=60_005.0,
                        close=60_010.0,
                        volume_usd=1_000_000.0,
                        carry_rate_period=0.0012,
                        carry_interval_h=12.0,
                        spread_bps=1.0,
                        provenance=PROVENANCE_REAL,
                    ),
                }
            ]
            with W3DNShadowMonitor(
                cfg2,
                provider=ReplayQuoteProvider([]),
                kernel=Web3DeltaNeutralCarryKernel(warmup_bars=3),
            ) as m2:
                r_flat = m2.prewarm_from_snapshots(flat_snaps)
                self.assertEqual(r_flat["prewarmed_bars_added"], 2)
                r_bars = m2.prewarm_from_snapshots(bar_frames)
                self.assertEqual(r_bars["prewarmed_bars_added"], 1)
                self.assertEqual(r_bars["prewarmed_bars_total"], 3)
                self.assertTrue(r_bars["kernel_warm_ready"])

    def test_tail_cost_spike_telemetry_and_legacy_checkpoint_fallback(self) -> None:
        # Reproduce SG W3-DN telemetry: mean < 36bp while BTC (80.17bp), SOL (82.76bp), AVAX (80.06bp) spike!
        # Formula: est_4leg = 2*(spot_fee=10 + spot_half) + 2*(perp_fee=4.5) + slip_buy + slip_sell = 29 + spot_spread + 2*l2_slip
        t0 = 1_795_000_000_000
        frames: list[dict[str, PairMarketSnapshot]] = []
        for i in range(20):
            ts = t0 + i * 60_000
            btc_slip = 25.085 if i == 18 else 0.5   # 29 + 1 + 2*25.085 = 80.17bp spike!
            sol_slip = 26.38 if i == 19 else 0.5    # 29 + 1 + 2*26.38  = 82.76bp spike!
            avax_slip = 25.03                       # 29 + 1 + 2*25.03  = 80.06bp observe-only!
            frames.append(
                {
                    "BTC": _make_pair_snap("BTC", ts, spot_mid=60_000.0, perp_l2_slip_bps=btc_slip),
                    "ETH": _make_pair_snap("ETH", ts, spot_mid=2_500.0, perp_l2_slip_bps=0.5),
                    "SOL": _make_pair_snap("SOL", ts, spot_mid=150.0, perp_l2_slip_bps=sol_slip),
                    "AVAX": _make_pair_snap("AVAX", ts, spot_mid=25.0, perp_l2_slip_bps=avax_slip),
                }
            )

        with tempfile.TemporaryDirectory(prefix="w3dn_spikes_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td),
                coins=("BTC", "ETH", "SOL"),
                observe_only_coins=("AVAX",),
                kernel_bar_interval_sec=60,
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(
                cfg,
                provider=ReplayQuoteProvider(frames),
                kernel=Web3DeltaNeutralCarryKernel(warmup_bars=2),
            ) as mon:
                for _ in range(20):
                    mon.run_once()
                rep = mon.evaluate_48h_readiness()
                self.assertLess(rep.mean_observed_4leg_cost_bps, 36.0)
                self.assertGreaterEqual(rep.p99_observed_4leg_cost_bps, 80.0)
                self.assertGreaterEqual(rep.max_observed_4leg_cost_bps, 82.7)
                self.assertEqual(rep.tail_cost_spike_threshold_bps, 36.0)
                self.assertEqual(rep.tail_cost_spike_count, 2)
                self.assertGreater(rep.tail_cost_spike_rate, 0.0)
                self.assertAlmostEqual(rep.primary_coin_cost_stats["BTC"]["max_4leg_cost_bps"], 80.17, places=1)
                self.assertAlmostEqual(rep.primary_coin_cost_stats["SOL"]["max_4leg_cost_bps"], 82.76, places=1)
                self.assertAlmostEqual(rep.observe_only_coin_stats["AVAX"]["max_4leg_cost_bps"], 80.06, places=1)
                self.assertEqual(rep.observe_only_coin_stats["AVAX"]["spike_count"], 20)
                # Strict JSON allow_nan=False round-trip
                serialized = json.dumps(rep.to_dict(), allow_nan=False)
                self.assertIn("p99_observed_4leg_cost_bps", serialized)
                expected_trades = rep.executed_trades
                self.assertGreater(expected_trades, 0)
                db_path = mon.db_path

            # Strip new keys from checkpoint JSON to verify legacy upgrade fallback reads SQLite shadow_events
            con = sqlite3.connect(str(db_path), timeout=5.0)
            try:
                con.execute("PRAGMA journal_mode=WAL;")
                row = con.execute("SELECT state_json FROM shadow_state_checkpoint WHERE id = 1").fetchone()
                st = json.loads(row[0])
                st.pop("executed_trades", None)
                st.pop("prewarmed_bars", None)
                st.pop("last_prewarm_bucket", None)
                con.execute(
                    "UPDATE shadow_state_checkpoint SET state_json = ? WHERE id = 1",
                    (json.dumps(st, allow_nan=False),),
                )
                con.commit()
            finally:
                con.close()

            with W3DNShadowMonitor(
                cfg,
                provider=ReplayQuoteProvider([]),
                kernel=Web3DeltaNeutralCarryKernel(warmup_bars=2),
            ) as mon_restarted:
                self.assertEqual(mon_restarted._executed_trades, expected_trades)
                rep2 = mon_restarted.evaluate_48h_readiness()
                self.assertEqual(rep2.executed_trades, expected_trades)
                self.assertGreaterEqual(rep2.max_observed_4leg_cost_bps, 82.7)
                self.assertEqual(rep2.tail_cost_spike_count, 2)


class TestShadowRunnerSTD46Hardening(unittest.TestCase):
    """F5.4: non-vacuous parity_depth, dual-venue pre-window funding warmup validator & finite BBO guard."""

    def _make_carry_cfg(self, *, lookback_hours: int = 96, dual: bool = True) -> StrategyConfig:
        return StrategyConfig(
            strategy_id="hl_gate_safe_carry_prod",
            version="v2.2.1-shadow",
            family="cross_venue_funding_carry",
            status="SHADOW_LIVE",
            plugin_id="funding_carry",
            venue_a="hyperliquid",
            venue_b="gate" if dual else "",
            mode="perp_perp",
            symbols=["BTC", "ETH"],
            capital_usd=10_000.0,
            lookback_hours=lookback_hours,
            slippage_per_leg_bps=3.0,
        )

    def test_live_execution_locks_and_parity_depth_vacuous_vs_active(self) -> None:
        self.assertFalse(LIVE_EXECUTION_ENABLED)
        self.assertFalse(DAEMON_LIVE_EXECUTION_ENABLED)
        self.assertFalse(RUNNER_SHADOW_LIVE_ENABLED)

        cfg = self._make_carry_cfg(lookback_hours=24)
        t0 = 1_795_000_000_000
        with tempfile.TemporaryDirectory(prefix="runner_parity_") as td:
            store = MarketDataStore(Path(td) / "market.sqlite")
            auditor = ShadowParityAuditor(Path(td) / "parity.sqlite", store)
            try:
                rec_vac = auditor.evaluate_strategy(
                    cfg,
                    window_start_ms=t0,
                    window_end_ms=t0 + 2 * 3_600_000,
                    shadow_equity_usd=10_000.0,
                    live_half_spreads_bps=[1.2, 1.4],
                    plans_created=0,
                    plans_blocked=0,
                    open_positions=0,
                    bbo_freshness_sec=5.0,
                    audit_chain_valid=True,
                    kill_switch_engaged=False,
                )
                self.assertEqual(rec_vac.status, "PARITY_OK")
                self.assertEqual(rec_vac.parity_depth, PARITY_DEPTH_VACUOUS)
                self.assertFalse(rec_vac.non_vacuous_parity)
                self.assertIn(PARITY_DEPTH_VACUOUS, rec_vac.detail)

                rec_act = auditor.evaluate_strategy(
                    cfg,
                    window_start_ms=t0,
                    window_end_ms=t0 + 2 * 3_600_000,
                    shadow_equity_usd=10_010.0,
                    live_half_spreads_bps=[1.2, 1.4],
                    plans_created=3,
                    plans_blocked=0,
                    open_positions=1,
                    bbo_freshness_sec=5.0,
                    audit_chain_valid=True,
                    kill_switch_engaged=False,
                )
                self.assertEqual(rec_act.parity_depth, PARITY_DEPTH_ACTIVE)
                self.assertTrue(rec_act.non_vacuous_parity)
                self.assertNotIn(PARITY_DEPTH_VACUOUS, rec_act.detail)
            finally:
                auditor.close()
                store.close()

    def test_validate_prewindow_funding_warmup_dual_venue_asymmetry_and_coverage(self) -> None:
        cfg = self._make_carry_cfg(lookback_hours=96, dual=True)
        win_start = 1_795_000_000_000
        hour_ms = 3_600_000

        with tempfile.TemporaryDirectory(prefix="runner_warmup_") as td:
            store = MarketDataStore(Path(td) / "market.sqlite")
            auditor = ShadowParityAuditor(Path(td) / "parity.sqlite", store)
            try:
                # 1. Cold DB (0h) -> insufficient
                r_cold = validate_prewindow_funding_warmup(store, cfg, win_start)
                self.assertFalse(r_cold["sufficient"])
                self.assertFalse(r_cold["warm_ok"])
                self.assertEqual(r_cold["reason"], "INSUFFICIENT_PREWINDOW_WARMUP")
                self.assertEqual(r_cold["required_hours"], 96)
                self.assertEqual(r_cold["observed_hours"], 0.0)
                self.assertEqual(r_cold["coverage_ratio"], 0.0)
                self.assertEqual(r_cold["required_venues"], ["hyperliquid", "gate"])

                # 2. Seed 96h on hyperliquid ONLY (gate has 0h) -> bottleneck MUST stay insufficient!
                hl_bars = [
                    FundingRateBar("hyperliquid", "BTC", win_start - h * hour_ms, 0.0001, 1.0, PROVENANCE_REAL)
                    for h in range(1, 100)
                ]
                store.upsert_funding(hl_bars)
                r_asym = validate_prewindow_funding_warmup(store, cfg, win_start)
                self.assertFalse(r_asym["sufficient"])
                self.assertEqual(r_asym["observed_hours"], 0.0)
                self.assertEqual(r_asym["warmed_symbols"], [])
                self.assertIn("BTC", r_asym["unwarmed_symbols"])

                # 3. Seed 48h on gate -> partial coverage (~0.5) -> still insufficient
                gate_48h = [
                    FundingRateBar("gate", "BTC", win_start - h * hour_ms, 0.00002, 1.0, PROVENANCE_REAL)
                    for h in range(1, 49)
                ]
                store.upsert_funding(gate_48h)
                r_part = auditor.validate_prewindow_warmup(cfg, win_start)
                self.assertFalse(r_part["sufficient"])
                self.assertAlmostEqual(r_part["observed_hours"], 48.0, places=1)
                self.assertAlmostEqual(r_part["coverage_ratio"], 0.5, places=2)

                # 4. Complete >=96h on gate -> warmed!
                gate_full = [
                    FundingRateBar("gate", "BTC", win_start - h * hour_ms, 0.00002, 1.0, PROVENANCE_REAL)
                    for h in range(49, 100)
                ]
                store.upsert_funding(gate_full)
                r_warm = validate_prewindow_funding_warmup(store, cfg, win_start)
                self.assertTrue(r_warm["sufficient"])
                self.assertTrue(r_warm["warmup_ok"])
                self.assertEqual(r_warm["reason"], "")
                self.assertEqual(r_warm["coverage_ratio"], 1.0)
                self.assertEqual(r_warm["warmed_symbols"], ["BTC"])
            finally:
                auditor.close()
                store.close()

    def test_daemon_finalize_report_fields_and_nan_bbo_rejection(self) -> None:
        now_ms = 1_795_400_000_000
        with tempfile.TemporaryDirectory(prefix="runner_daemon_") as td:
            daemon = ShadowPortfolioDaemon(Path(td), bbo_symbols=["BTC"])
            try:
                poison_ticks = [
                    BBOTick("hyperliquid", "BTC", now_ms, float("nan"), 60_001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    BBOTick("hyperliquid", "BTC", now_ms, 60_000.0, float("inf"), 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", now_ms, True, 60_001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL),  # type: ignore[arg-type]
                ]
                rep = daemon.run_once(now_ms=now_ms, live_ticks=poison_ticks, collect_live=False)
                self.assertEqual(rep["status"], "DEGRADED_HALT")
                self.assertEqual(rep["plans_created"], 0)
                self.assertEqual(rep["parity_depth"], PARITY_DEPTH_VACUOUS)
                self.assertFalse(rep["non_vacuous_parity"])
                self.assertFalse(rep["all_strategies_non_vacuous"])
                self.assertEqual(daemon._latest_bbo_cache, {})
                disk_report = json.loads((Path(td) / "shadow_parity_report.json").read_text(encoding="utf-8"))
                self.assertEqual(disk_report["parity_depth"], PARITY_DEPTH_VACUOUS)
                self.assertFalse(disk_report["non_vacuous_parity"])
                self.assertEqual(disk_report["plans_created"], 0)
            finally:
                daemon.close()


class TestSTD46AuditLayerAndImmutability(unittest.TestCase):
    """F5.5: Dual Sharpe, Static Ablation, Composite Eligibility, validation.py anchor & SCHQ report gate."""

    def test_validation_py_zero_byte_untouched_anchor(self) -> None:
        self.assertTrue(_VALIDATION_PY.is_file())
        raw = _VALIDATION_PY.read_bytes()
        self.assertEqual(len(raw), _VALIDATION_EXPECTED_SIZE)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), _VALIDATION_EXPECTED_SHA256)

    def test_ver02_dual_ratio_platform_v4_fixtures_and_boundaries(self) -> None:
        # 1. Platform v4 5-sleeve 3.17 composite @ USD Rf=4.5% -> ~1.20 excess Sharpe
        comp = compute_dual_sharpe(cagr_pct=7.231, ann_vol_pct=2.28, rf_pct=4.5, raw_sharpe=3.1721)
        self.assertAlmostEqual(comp["raw_sharpe"], 3.1721, places=6)
        self.assertAlmostEqual(comp["excess_sharpe"], 3.1721 - 4.5 / 2.28, places=6)
        self.assertAlmostEqual(comp["excess_sharpe"], 1.198416, places=4)
        self.assertAlmostEqual(comp["excess_cagr_pct"], 2.731, places=6)
        self.assertEqual(comp["rf_pct"], 4.5)

        # 2. A2 CN A-share @ RMB Rf=1.5%
        a2 = compute_dual_sharpe(cagr_pct=4.3701, ann_vol_pct=5.325, rf_pct=1.5, raw_sharpe=0.8517)
        self.assertAlmostEqual(a2["excess_sharpe"], 0.8517 - 1.5 / 5.325, places=6)
        self.assertLess(a2["excess_sharpe"], a2["raw_sharpe"])

        # 3. W3-DN @ USD Rf=4.5%
        w3 = compute_dual_sharpe(cagr_pct=6.95, ann_vol_pct=0.74, rf_pct=4.5, raw_sharpe=9.148)
        self.assertAlmostEqual(w3["excess_sharpe"], 9.148 - 4.5 / 0.74, places=6)

        # 4. Derived raw_sharpe=None
        derived = compute_dual_sharpe(cagr_pct=9.0, ann_vol_pct=3.0, rf_pct=4.5)
        self.assertAlmostEqual(derived["raw_sharpe"], 3.0, places=9)
        self.assertAlmostEqual(derived["excess_sharpe"], 1.5, places=9)

        # 5. Fail-closed boundary validation
        for bad_kwargs in (
            {"cagr_pct": 5.0, "ann_vol_pct": 0.0, "rf_pct": 4.5},
            {"cagr_pct": 5.0, "ann_vol_pct": -1.0, "rf_pct": 4.5},
            {"cagr_pct": 5.0, "ann_vol_pct": 2.0, "rf_pct": -0.1},
            {"cagr_pct": float("nan"), "ann_vol_pct": 2.0, "rf_pct": 4.5},
            {"cagr_pct": 5.0, "ann_vol_pct": float("inf"), "rf_pct": 4.5},
            {"cagr_pct": True, "ann_vol_pct": 2.0, "rf_pct": 4.5},
            {"cagr_pct": 5.0, "ann_vol_pct": 2.0, "rf_pct": 4.5, "raw_sharpe": float("nan")},
        ):
            with self.assertRaises(ValueError):
                compute_dual_sharpe(**bad_kwargs)  # type: ignore[arg-type]

    def test_evaluate_static_ablation_a2_dominance_aliases_and_sign_invariance(self) -> None:
        a2_pos = {"cagr_pct": 4.3701, "sharpe": 0.8517, "max_dd_pct": 6.8795, "calmar": 0.6352}
        static_pos = {"cagr_pct": 4.6007, "sharpe": 0.8804, "max_dd_pct": 6.3405, "calmar": 0.7256}
        res_pos = evaluate_static_ablation(a2_pos, static_pos)
        self.assertTrue(res_pos["dominated_by_static"])
        self.assertEqual(res_pos["verdict_cap"], "STATIC_ALLOCATION_DOMINATED")
        self.assertAlmostEqual(res_pos["deltas"]["cagr"], -0.2306, places=4)
        self.assertAlmostEqual(res_pos["deltas"]["sharpe"], -0.0287, places=4)
        self.assertAlmostEqual(res_pos["deltas"]["max_dd"], -0.5390, places=4)
        self.assertAlmostEqual(res_pos["deltas"]["calmar"], -0.0904, places=4)

        # Negative MaxDD sign convention + alternate alias keys yield identical verdict & deltas
        a2_neg = {"cagr": 4.3701, "raw_sharpe": 0.8517, "maxdd": -6.8795, "calmar_ratio": 0.6352}
        static_neg = {"cagr": 4.6007, "raw_sharpe": 0.8804, "max_drawdown": -6.3405, "calmar_ratio": 0.7256}
        res_neg = evaluate_static_ablation(a2_neg, static_neg)
        self.assertTrue(res_neg["dominated_by_static"])
        self.assertEqual(res_neg["verdict_cap"], "STATIC_ALLOCATION_DOMINATED")
        self.assertAlmostEqual(res_neg["deltas"]["max_dd"], -0.5390, places=4)

        # Exact tie on all 4 metrics -> dominated
        res_tie = evaluate_static_ablation(static_pos, static_pos)
        self.assertTrue(res_tie["dominated_by_static"])
        self.assertEqual(res_tie["verdict_cap"], "STATIC_ALLOCATION_DOMINATED")

        # Better on even 1 metric (smaller MaxDD) -> PASS
        better_dd = dict(a2_pos, max_dd_pct=5.0)
        res_pass = evaluate_static_ablation(better_dd, static_pos)
        self.assertFalse(res_pass["dominated_by_static"])
        self.assertEqual(res_pass["verdict_cap"], "PASS")
        self.assertGreater(res_pass["deltas"]["max_dd"], 0.0)

        # Apples-to-apples common key preference
        strat_both = {"cagr_pct": 5.0, "excess_sharpe": 0.4, "raw_sharpe": 1.1, "max_dd_pct": 5.0, "calmar": 1.0}
        static_raw = {"cagr_pct": 5.0, "raw_sharpe": 1.0, "max_dd_pct": 5.0, "calmar": 1.0}
        res_common = evaluate_static_ablation(strat_both, static_raw)
        self.assertFalse(res_common["dominated_by_static"])
        self.assertAlmostEqual(res_common["deltas"]["sharpe"], 0.1, places=6)

        # Fail-closed missing / non-finite / bool
        with self.assertRaises(ValueError):
            evaluate_static_ablation({"cagr_pct": 5.0}, static_pos)
        with self.assertRaises(ValueError):
            evaluate_static_ablation(dict(a2_pos, calmar=float("nan")), static_pos)
        with self.assertRaises(ValueError):
            evaluate_static_ablation(dict(a2_pos, cagr_pct=True), static_pos)

    def test_audit_composite_portfolio_platform_v4_317_and_edge_cases(self) -> None:
        sleeves_317 = [
            {"sleeve_id": "W1", "gates_passed": 6, "gates_total": 8, "currency": "USD"},
            {"sleeve_id": "W3-DN", "gates_passed": 8, "gates_total": 8, "currency": "USD"},
            {"sleeve_id": "G1_GLD", "gates_passed": 7, "gates_total": 8, "currency": "USD"},
            {"sleeve_id": "G4", "gates_passed": 7, "gates_total": 8, "currency": "USD"},
            {"sleeve_id": "A2", "gates_passed": 8, "gates_total": 8, "currency": "RMB"},
        ]
        res = audit_composite_portfolio(sleeves_317, {"USD", "RMB"}, fx_hedged=False)
        self.assertFalse(res["eligible_for_production"])
        self.assertEqual(res["classification"], "DIAGNOSTIC_HYPOTHETICAL_ONLY")
        self.assertEqual(res["failing_sleeves"], ["W1", "G1_GLD", "G4"])
        self.assertIsNotNone(res["fx_warning"])
        self.assertEqual(res["currencies"], ["RMB", "USD"])

        # Disqualifying status on an 8/8 sleeve (e.g. A2 STATIC_ALLOCATION_DOMINATED) also fails
        sleeves_dom = [dict(s) for s in sleeves_317]
        sleeves_dom[-1]["status"] = "STATIC_ALLOCATION_DOMINATED"
        res_dom = audit_composite_portfolio(sleeves_dom, {"USD", "RMB"}, fx_hedged=False)
        self.assertEqual(res_dom["failing_sleeves"], ["W1", "G1_GLD", "G4", "A2"])

        # All 8/8 single-currency -> PRODUCTION_ELIGIBLE
        clean_usd = [
            {"sleeve_id": "W3-DN", "passed_gates": 8, "total_gates": 8, "currency": "USD", "weight": 0.5},
            {"sleeve_id": "G2", "gates": "8/8", "currency": "USD", "weight": 0.5},
        ]
        res_ok = audit_composite_portfolio(clean_usd, {"USD"}, fx_hedged=False)
        self.assertTrue(res_ok["eligible_for_production"])
        self.assertEqual(res_ok["classification"], "PRODUCTION_ELIGIBLE")
        self.assertEqual(res_ok["failing_sleeves"], [])
        self.assertIsNone(res_ok["fx_warning"])

        # Multi-currency CNY->RMB normalization: unhedged fails, hedged passes
        multi = [
            {"sleeve_id": "S1", "passed_8_of_8": True, "currency": "USD"},
            {"sleeve_id": "S2", "gates_passed": 8, "gates_total": 8, "currency": "CNY"},
        ]
        res_unhedged = audit_composite_portfolio(multi, {"USD", "CNY"}, fx_hedged=False)
        self.assertFalse(res_unhedged["eligible_for_production"])
        self.assertEqual(res_unhedged["classification"], "DIAGNOSTIC_HYPOTHETICAL_ONLY")
        self.assertIsNotNone(res_unhedged["fx_warning"])
        self.assertEqual(res_unhedged["currencies"], ["RMB", "USD"])

        res_hedged = audit_composite_portfolio(multi, {"USD", "CNY"}, fx_hedged=True)
        self.assertTrue(res_hedged["eligible_for_production"])
        self.assertEqual(res_hedged["classification"], "PRODUCTION_ELIGIBLE")
        self.assertIsNone(res_hedged["fx_warning"])

        # CNY + RMB alone is single currency
        cny_rmb = [
            {"sleeve_id": "A1", "gates_passed": 8, "gates_total": 8, "currency": "CNY"},
            {"sleeve_id": "A2", "gates_passed": 8, "gates_total": 8, "currency": "RMB"},
        ]
        res_same = audit_composite_portfolio(cny_rmb, {"CNY", "RMB"}, fx_hedged=False)
        self.assertTrue(res_same["eligible_for_production"])
        self.assertEqual(res_same["currencies"], ["RMB"])

        # Empty sleeves -> fail-closed diagnostic
        res_empty = audit_composite_portfolio([], {"USD"}, fx_hedged=False)
        self.assertFalse(res_empty["eligible_for_production"])
        self.assertEqual(res_empty["classification"], "DIAGNOSTIC_HYPOTHETICAL_ONLY")

        # Non-bool fx_hedged -> ValueError
        with self.assertRaises(ValueError):
            audit_composite_portfolio(clean_usd, {"USD"}, fx_hedged="yes")  # type: ignore[arg-type]

    def test_schq_verify_data_and_strategy_gate_mode_report_roundtrip(self) -> None:
        self.assertTrue(_SCHQ_GATE_SCRIPT.is_file(), f"Missing SCHQ verifier CLI: {_SCHQ_GATE_SCRIPT}")
        a2_ablation = evaluate_static_ablation(
            {"cagr_pct": 4.3701, "sharpe": 0.8517, "max_dd_pct": 6.8795, "calmar": 0.6352},
            {"cagr_pct": 4.6007, "sharpe": 0.8804, "max_dd_pct": 6.3405, "calmar": 0.7256},
        )
        a2_dual = compute_dual_sharpe(cagr_pct=4.3701, ann_vol_pct=5.325, rf_pct=1.5, raw_sharpe=0.8517)
        comp_dual = compute_dual_sharpe(cagr_pct=7.231, ann_vol_pct=2.28, rf_pct=4.5, raw_sharpe=3.1721)
        sleeves_317 = [
            {"sleeve_id": "W1", "gates_passed": 6, "gates_total": 8, "currency": "USD"},
            {"sleeve_id": "W3-DN", "gates_passed": 8, "gates_total": 8, "currency": "USD"},
            {"sleeve_id": "G1_GLD", "gates_passed": 7, "gates_total": 8, "currency": "USD"},
            {"sleeve_id": "G4", "gates_passed": 7, "gates_total": 8, "currency": "USD"},
            {"sleeve_id": "A2", "gates_passed": 8, "gates_total": 8, "currency": "RMB"},
        ]

        compliant = build_std46_audit_report(
            report_name="platform_v4_std46_m3_audit",
            strategies=[
                {
                    "strategy_id": "A2_cn_ashare_macro_erp_rotation",
                    "currency": "RMB",
                    **a2_dual,
                    "static_ablation": a2_ablation,
                    "status": "PROMOTE_SHADOW_CANDIDATE",  # MUST be auto-capped by build_std46_audit_report!
                },
            ],
            composites=[
                {
                    "portfolio_id": "phase_d_three_domain_w1_w3dn_g1_g4_a2",
                    "currencies": ["USD", "RMB"],
                    "fx_hedged": False,
                    "sleeves": sleeves_317,
                    **comp_dual,
                },
            ],
            shadows=[
                {"shadow_id": "w3_dn_sg_vacuous", "executed_trades": 0, "status": "READY_FOR_OWNER_REVIEW"},
                {"shadow_id": "w3_dn_prewarmed", "executed_trades": 2, "plans_created": 2, "non_vacuous_parity": True, "shadow_status": "ACTIVE_EXECUTION"},
            ],
        )
        self.assertEqual(compliant["strategies"][0]["status"], "STATIC_ALLOCATION_DOMINATED")
        self.assertEqual(compliant["composites"][0]["classification"], "DIAGNOSTIC_HYPOTHETICAL_ONLY")
        self.assertEqual(compliant["shadows"][0]["status"], "SHADOW_VACUOUS_ZERO_TRADES")

        with tempfile.TemporaryDirectory(prefix="std46_rep_") as td:
            rep_ok_path = Path(td) / "compliant_report.json"
            rep_ok_path.write_text(json.dumps(compliant, indent=2, allow_nan=False), encoding="utf-8")
            proc_ok = subprocess.run(
                [sys.executable, "-B", str(_SCHQ_GATE_SCRIPT), "--mode", "report", str(rep_ok_path), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_ok.returncode, 0, f"Compliant STD-46 report failed:\n{proc_ok.stdout}\n{proc_ok.stderr}")
            parsed_ok = json.loads(proc_ok.stdout)
            self.assertEqual(parsed_ok["status"], "PASS")

            # Negative control 1: A2 claiming PROMOTE_SHADOW_CANDIDATE -> EXIT 1 (VER-03)
            bad_a2 = json.loads(json.dumps(compliant, allow_nan=False))
            bad_a2["strategies"][0]["status"] = "PROMOTE_SHADOW_CANDIDATE"
            bad_a2["strategies"][0]["verdict"] = "PROMOTE_SHADOW_CANDIDATE"
            bad_a2_path = Path(td) / "bad_a2.json"
            bad_a2_path.write_text(json.dumps(bad_a2, allow_nan=False), encoding="utf-8")
            proc_a2 = subprocess.run(
                [sys.executable, "-B", str(_SCHQ_GATE_SCRIPT), "--mode", "report", str(bad_a2_path), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_a2.returncode, 1)

            # Negative control 2: 5-sleeve composite claiming PRODUCTION -> EXIT 1 (VER-04)
            bad_comp = json.loads(json.dumps(compliant, allow_nan=False))
            bad_comp["composites"][0]["status"] = "PRODUCTION"
            bad_comp["composites"][0]["classification"] = "PRODUCTION_ELIGIBLE"
            bad_comp["composites"][0]["eligible_for_production"] = True
            bad_comp_path = Path(td) / "bad_comp.json"
            bad_comp_path.write_text(json.dumps(bad_comp, allow_nan=False), encoding="utf-8")
            proc_comp = subprocess.run(
                [sys.executable, "-B", str(_SCHQ_GATE_SCRIPT), "--mode", "report", str(bad_comp_path), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_comp.returncode, 1)

            # Negative control 3: 0-trade shadow claiming SHADOW_READY -> EXIT 1 (VER-07)
            bad_sh = json.loads(json.dumps(compliant, allow_nan=False))
            bad_sh["shadows"][0]["status"] = "SHADOW_READY"
            bad_sh["shadows"][0]["shadow_status"] = "SHADOW_READY"
            bad_sh_path = Path(td) / "bad_shadow.json"
            bad_sh_path.write_text(json.dumps(bad_sh, allow_nan=False), encoding="utf-8")
            proc_sh = subprocess.run(
                [sys.executable, "-B", str(_SCHQ_GATE_SCRIPT), "--mode", "report", str(bad_sh_path), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_sh.returncode, 1)


if __name__ == "__main__":
    unittest.main()
