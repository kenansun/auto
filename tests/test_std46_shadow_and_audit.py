"""Unit & integration tests for M3 STD-46 hardening across BBO, W3-DN Shadow, Shadow Runner & Core Audit Layer.

Strictly 100% local & offline (zero network calls, zero live execution, zero remote mutations).
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from alphaloop.core.models import BBOTick, FundingRateBar, PriceBar, StrategyConfig
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
    MAX_VALID_EPOCH_MS,
    MIN_VALID_EPOCH_MS,
    TS_SOURCE_RECV_LOCAL,
    TS_SOURCE_VENUE,
    MultiVenueBBOCollector,
    TimestampedBBOTick,
    _http_json,
    _parse_optional_nonneg_qty,
    _parse_venue_ts_ms,
)
from alphaloop.data.store import MarketDataStore
from alphaloop.execution import bridge as bridge_module
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

    def test_parse_venue_ts_ms_epoch_bounds_and_signed_zero_qty(self) -> None:
        # Valid boundary values
        self.assertEqual(_parse_venue_ts_ms(MIN_VALID_EPOCH_MS), MIN_VALID_EPOCH_MS)
        self.assertEqual(_parse_venue_ts_ms(MAX_VALID_EPOCH_MS), MAX_VALID_EPOCH_MS)
        self.assertEqual(_parse_venue_ts_ms("1795000123456"), 1_795_000_123_456)
        self.assertEqual(_parse_venue_ts_ms(1_795_000_123_456.7), 1_795_000_123_456)

        # Fractional < MIN, monotonic, seconds, microseconds, > MAX, and 1e308 overflow MUST return None
        for bad_ts in (
            0.5,
            "0.5",
            1e-9,
            0,
            -1,
            12345,
            1_795_000_123,
            1_795_000_123_456_000,
            MIN_VALID_EPOCH_MS - 1,
            MAX_VALID_EPOCH_MS + 1,
            1e308,
            "1e308",
            float("nan"),
            float("inf"),
            True,
            False,
            None,
            "",
        ):
            self.assertIsNone(_parse_venue_ts_ms(bad_ts), f"Expected None for out-of-bounds ts={bad_ts!r}")

        # Out-of-bounds exchange timestamps fall back cleanly to RECV_LOCAL_ONLY and store without OverflowError
        collector = MultiVenueBBOCollector(["BTC", "ETH"])
        payload = {
            "data": [
                {"instId": "BTC-USDT-SWAP", "bidPx": "60000", "askPx": "60001", "bidSz": "-0.0", "askSz": "1", "ts": 0.5},
                {"instId": "ETH-USDT-SWAP", "bidPx": "2500", "askPx": "2501", "bidSz": "1", "askSz": "1", "ts": 1e308},
            ]
        }
        with patch("alphaloop.data.bbo_collector._http_json", return_value=(payload, 5.0)):
            ticks = collector.fetch_okx_swap_bbo()
        self.assertEqual(len(ticks), 2)
        for tk in ticks:
            self.assertIsNone(tk.venue_ts_ms)
            self.assertEqual(tk.ts_source, TS_SOURCE_RECV_LOCAL)
            self.assertEqual(tk.ts_ms, tk.recv_ts_ms)
            self.assertGreaterEqual(tk.ts_ms, MIN_VALID_EPOCH_MS)
        self.assertEqual(math.copysign(1.0, ticks[0].bid_qty), 1.0)
        q_zero = _parse_optional_nonneg_qty(-0.0)
        self.assertIsNotNone(q_zero)
        assert q_zero is not None
        self.assertEqual(math.copysign(1.0, q_zero), 1.0)
        with tempfile.TemporaryDirectory(prefix="bbo_bounds_") as td:
            store = MarketDataStore(Path(td) / "market.sqlite")
            try:
                self.assertEqual(store.upsert_bbo_ticks(ticks), 2)
            finally:
                store.close()

    def test_http_json_nonstandard_constants_malformed_envelopes_and_dual_key_ts(self) -> None:
        # 1. Non-standard bare NaN / Infinity in HTTP JSON body -> RuntimeError isolated by capture_round
        class _FakeResp:
            def __init__(self, body: bytes) -> None:
                self._buf = io.BytesIO(body)

            def read(self) -> bytes:
                return self._buf.read()

            def __enter__(self) -> _FakeResp:
                return self

            def __exit__(self, *_a: Any) -> None:
                return None

        for raw_body in (b'{"bid": NaN}', b'{"bid": Infinity}', b'{"bid": -Infinity}'):
            with patch("urllib.request.urlopen", return_value=_FakeResp(raw_body)):
                with self.assertRaises(RuntimeError):
                    _http_json("https://example.invalid/ticker", max_retries=1)

        # 2. Malformed envelopes on OKX/Gate/Lighter do not crash fetch_* or abort capture_round()
        collector = MultiVenueBBOCollector(["BTC"])
        exch_ts = 1_795_000_222_333

        with patch("alphaloop.data.bbo_collector._http_json", return_value=({"code": "50000", "data": None}, 4.0)):
            self.assertEqual(collector.fetch_okx_swap_bbo(), [])
        with patch("alphaloop.data.bbo_collector._http_json", return_value=({"data": [{"instId": ["BTC-USDT-SWAP"], "bidPx": "60000", "askPx": "60001"}]}, 4.0)):
            self.assertEqual(collector.fetch_okx_swap_bbo(), [])

        # Gate malformed row + dual-key TS fallback (ts_ms=None or ts_ms=0 -> time_ms)
        gate_rows = [
            {"contract": ["BTC_USDT"], "highest_bid": "60000", "lowest_ask": "60001"},
            {"contract": "BTC_USDT", "highest_bid": "60000", "lowest_ask": "60001", "ts_ms": None, "time_ms": exch_ts},
            {"contract": "BTC_USDT", "highest_bid": "60002", "lowest_ask": "60003", "ts_ms": 0, "time_ms": exch_ts + 100},
        ]
        with patch("alphaloop.data.bbo_collector._http_json", return_value=(gate_rows, 4.0)):
            g_ticks = collector.fetch_gate_futures_bbo()
        self.assertEqual(len(g_ticks), 2)
        self.assertEqual(g_ticks[0].venue_ts_ms, exch_ts)
        self.assertEqual(g_ticks[0].ts_source, TS_SOURCE_VENUE)
        self.assertEqual(g_ticks[1].venue_ts_ms, exch_ts + 100)
        self.assertEqual(g_ticks[1].ts_source, TS_SOURCE_VENUE)

        # Lighter market discovery malformed rows + dual-key TS fallback (ts_ms=None/0 -> timestamp_ms)
        bad_markets = {
            "order_books": [
                {"status": "active", "market_type": "perp", "symbol": "BAD1"},
                {"status": "active", "market_type": "perp", "symbol": "BAD2", "market_id": "not-int"},
                {"status": "active", "market_type": "perp", "symbol": "BAD3", "market_id": True},
                {"status": "active", "market_type": "perp", "symbol": 123, "market_id": 2},
                {"status": "active", "market_type": "perp", "symbol": "BTC", "market_id": 0},
            ]
        }
        c_none = MultiVenueBBOCollector(["BTC"])
        with patch("alphaloop.data.bbo_collector._http_json", return_value=({"order_books": None}, 3.0)):
            self.assertEqual(c_none._ensure_lighter_markets(), {})
        with patch("alphaloop.data.bbo_collector._http_json", return_value=(bad_markets, 3.0)):
            self.assertEqual(c_none._ensure_lighter_markets(), {"BTC": 0})

        lighter_book = {
            "bids": [{"price": "60000", "remaining_base_amount": "1"}],
            "asks": [{"price": "60001", "remaining_base_amount": "1"}],
            "ts_ms": 0,
            "timestamp_ms": exch_ts,
        }
        with patch("alphaloop.data.bbo_collector._http_json", return_value=(lighter_book, 3.0)):
            lt = c_none.fetch_lighter_bbo()
        self.assertEqual(len(lt), 1)
        self.assertEqual(lt[0].venue_ts_ms, exch_ts)
        self.assertEqual(lt[0].ts_source, TS_SOURCE_VENUE)

        # capture_round isolates RuntimeError on one venue without aborting remaining venues
        def _route_round(url: str, **_kw: Any) -> tuple[Any, float]:
            if "okx.com" in url:
                raise RuntimeError("okx bad json NaN")
            if "gateio.ws" in url:
                return gate_rows[1:2], 5.0
            if "hyperliquid.xyz" in url:
                return {"levels": [[{"px": "60000", "sz": "1"}], [{"px": "60001", "sz": "1"}]], "time": exch_ts}, 5.0
            return {"order_books": None}, 5.0

        c_round = MultiVenueBBOCollector(["BTC"])
        with patch("alphaloop.data.bbo_collector._http_json", side_effect=_route_round):
            all_ticks, errs = c_round.capture_round()
        self.assertIn("okx", errs)
        self.assertEqual(sorted(t.venue for t in all_ticks), ["gate", "hyperliquid"])


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

    def test_prewarm_hardening_leg_ts_post_live_noop_canonical_pairing_and_edge_cases(self) -> None:
        t0 = 1_795_000_000_000
        bar_ms = 12 * 3_600_000

        # 1. Poisoned spot_ts_ms / perp_ts_ms in dict prewarm inputs MUST fail closed (never default to obs_ms)
        poisoned_ts_frames: list[dict[str, Any]] = []
        for idx, (s_ts, p_ts) in enumerate((
            (float("nan"), t0 - bar_ms),
            (float("inf"), t0 - bar_ms),
            (-999, t0 - bar_ms),
            (0, t0 - bar_ms),
            (True, t0 - bar_ms),
            (12345, t0 - bar_ms),
            (t0 - bar_ms, "garbage"),
        )):
            poisoned_ts_frames.append(
                {
                    "coin": "BTC",
                    "ts_observe_ms": t0 - (10 - idx) * bar_ms,
                    "spot_bid": 60_000.0,
                    "spot_ask": 60_001.0,
                    "spot_ts_ms": s_ts,
                    "perp_bid": 60_010.0,
                    "perp_ask": 60_011.0,
                    "perp_ts_ms": p_ts,
                    "perp_funding_1h": 0.0001,
                }
            )
        # Flat dict with nested metadata dict + absent leg timestamps (defaults to obs_ms cleanly)
        flat_with_meta = [
            {
                "coin": "BTC",
                "ts_observe_ms": t0 - bar_ms,
                "spot_bid": 60_000.0,
                "spot_ask": 60_001.0,
                "perp_bid": 60_010.0,
                "perp_ask": 60_011.0,
                "perp_funding_1h": 0.0001,
                "meta": {"source": "archive", "batch": 1},
            }
        ]
        with tempfile.TemporaryDirectory(prefix="w3dn_hardening_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td) / "m1",
                coins=("BTC",),
                kernel_bar_interval_sec=43_200,
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(cfg, provider=ReplayQuoteProvider([]), kernel=Web3DeltaNeutralCarryKernel(warmup_bars=3)) as mon:
                r_bad = mon.prewarm_from_snapshots(poisoned_ts_frames)
                self.assertEqual(r_bad["prewarmed_bars_added"], 0)
                self.assertEqual(r_bad["rejected_frames"], len(poisoned_ts_frames))
                self.assertFalse(r_bad["kernel_warm_ready"])

                r_meta = mon.prewarm_from_snapshots(flat_with_meta)
                self.assertEqual(r_meta["prewarmed_bars_added"], 1)
                self.assertEqual(r_meta["rejected_frames"], 0)

            # 2. enable_prewarm=True, prewarm_bars=0 MUST pre-warm 0 bars (not unlimited); prewarm_bars=-1 raises ValueError
            hist_15 = [{"BTC": _make_pair_snap("BTC", t0 - (15 - i) * bar_ms)} for i in range(15)]
            cfg_zero = ShadowMonitorConfig(
                state_dir=Path(td) / "m_zero",
                coins=("BTC",),
                kernel_bar_interval_sec=43_200,
                enforce_wall_clock_bounds=False,
                enable_prewarm=True,
                prewarm_bars=0,
            )
            with W3DNShadowMonitor(cfg_zero, provider=ReplayQuoteProvider([]), prewarm_frames=hist_15) as mon_z:
                self.assertEqual(mon_z._prewarmed_bars, 0)
                self.assertFalse(mon_z._is_kernel_warm_ready())

            cfg_neg = ShadowMonitorConfig(
                state_dir=Path(td) / "m_neg",
                coins=("BTC",),
                enable_prewarm=True,
                prewarm_bars=-1,
            )
            with self.assertRaises(ValueError):
                W3DNShadowMonitor(cfg_neg, provider=ReplayQuoteProvider([]))

            # 3. Post-live prewarm MUST be a strict no-op (both direct call and restart with prewarm_frames)
            live_dir = Path(td) / "m_live"
            cfg_live = ShadowMonitorConfig(
                state_dir=live_dir,
                coins=("BTC",),
                kernel_bar_interval_sec=43_200,
                enforce_wall_clock_bounds=False,
            )
            live_snap = [{"BTC": _make_pair_snap("BTC", t0, funding_1h=-0.00001)}]
            with W3DNShadowMonitor(cfg_live, provider=ReplayQuoteProvider(live_snap)) as mon_l:
                mon_l.run_once()
                st_before = mon_l.kernel.export_state()
                ckpt_ts_before = mon_l.conn.fetch_one("SELECT updated_ts_ms FROM shadow_state_checkpoint WHERE id = 1")["updated_ts_ms"]
                r_noop = mon_l.prewarm_from_snapshots(hist_15)
                self.assertEqual(r_noop["prewarmed_bars_added"], 0)
                self.assertEqual(mon_l.kernel.export_state(), st_before)
                ckpt_ts_after = mon_l.conn.fetch_one("SELECT updated_ts_ms FROM shadow_state_checkpoint WHERE id = 1")["updated_ts_ms"]
                self.assertEqual(ckpt_ts_after, ckpt_ts_before)

            cfg_restart = ShadowMonitorConfig(
                state_dir=live_dir,
                coins=("BTC",),
                kernel_bar_interval_sec=43_200,
                enforce_wall_clock_bounds=False,
                enable_prewarm=True,
                prewarm_bars=12,
            )
            with W3DNShadowMonitor(cfg_restart, provider=ReplayQuoteProvider([]), prewarm_frames=hist_15) as mon_r:
                self.assertEqual(mon_r._prewarmed_bars, 0)
                self.assertEqual(mon_r.kernel.export_state(), st_before)
                ckpt_ts_restart = mon_r.conn.fetch_one("SELECT updated_ts_ms FROM shadow_state_checkpoint WHERE id = 1")["updated_ts_ms"]
                self.assertEqual(ckpt_ts_restart, ckpt_ts_before)
                db_null_test = mon_r.db_path

            # 4. Null prewarmed_bars in checkpoint hydrates cleanly to 0 without TypeError
            con = sqlite3.connect(str(db_null_test), timeout=5.0)
            try:
                con.execute("PRAGMA journal_mode=WAL;")
                row = con.execute("SELECT state_json FROM shadow_state_checkpoint WHERE id = 1").fetchone()
                st = json.loads(row[0])
                st["prewarmed_bars"] = None
                con.execute(
                    "UPDATE shadow_state_checkpoint SET state_json = ? WHERE id = 1",
                    (json.dumps(st, allow_nan=False),),
                )
                con.commit()
            finally:
                con.close()
            with W3DNShadowMonitor(cfg_live, provider=ReplayQuoteProvider([])) as mon_null:
                self.assertEqual(mon_null._prewarmed_bars, 0)

            # 5. CanonicalBar unpaired legs, NaN ts_event_ms, and leading {} delegation
            def _mk_bar(sym: str, ts_val: Any, open_px: float = 60_000.0, is_perp: bool = False) -> CanonicalBar:
                return CanonicalBar(
                    asset_class=ASSET_CLASS_CRYPTO_PERP if is_perp else ASSET_CLASS_CRYPTO_SPOT,
                    venue="hyperliquid" if is_perp else "binance_spot",
                    symbol=sym,
                    timeframe_h=12.0,
                    timeframe_sec=43_200,
                    ts_event_ms=ts_val,
                    ts_avail_ms=t0 - bar_ms,
                    open=open_px,
                    high=60_005.0,
                    low=59_995.0,
                    close=60_000.0,
                    volume_usd=1_000_000.0,
                    carry_rate_period=0.0012 if is_perp else 0.0,
                    carry_interval_h=12.0,
                    spread_bps=1.0,
                    provenance=PROVENANCE_REAL,
                )

            unpaired_frames = [
                {
                    spot_leg_symbol("BTC"): _mk_bar(spot_leg_symbol("BTC"), t0 - (12 - i) * bar_ms),
                    perp_leg_symbol("BTC"): _mk_bar(perp_leg_symbol("BTC"), t0 - (12 - i) * bar_ms, open_px=float("nan"), is_perp=True),
                }
                for i in range(12)
            ]
            nan_ts_frame = [
                {
                    spot_leg_symbol("BTC"): _mk_bar(spot_leg_symbol("BTC"), float("nan")),
                    perp_leg_symbol("BTC"): _mk_bar(perp_leg_symbol("BTC"), t0 - bar_ms, is_perp=True),
                }
            ]
            valid_12_bars = [
                {
                    spot_leg_symbol("BTC"): _mk_bar(spot_leg_symbol("BTC"), t0 - (12 - i) * bar_ms),
                    perp_leg_symbol("BTC"): _mk_bar(perp_leg_symbol("BTC"), t0 - (12 - i) * bar_ms, is_perp=True),
                }
                for i in range(12)
            ]

            cfg_bars = ShadowMonitorConfig(
                state_dir=Path(td) / "m_bars",
                coins=("BTC",),
                kernel_bar_interval_sec=43_200,
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(cfg_bars, provider=ReplayQuoteProvider([]), kernel=Web3DeltaNeutralCarryKernel(warmup_bars=12)) as mon_b:
                r_unp = mon_b.prewarm_from_bars(unpaired_frames)
                self.assertEqual(r_unp["prewarmed_bars_added"], 0)
                self.assertEqual(r_unp["rejected_frames"], 12)
                self.assertFalse(r_unp["kernel_warm_ready"])
                self.assertEqual(mon_b.kernel.export_state()["states"], {})

                r_nants = mon_b.prewarm_from_bars(nan_ts_frame)
                self.assertEqual(r_nants["prewarmed_bars_added"], 0)
                self.assertEqual(r_nants["rejected_frames"], 1)

                r_lead_empty = mon_b.prewarm_from_snapshots([{}] + valid_12_bars)
                self.assertEqual(r_lead_empty["prewarmed_bars_added"], 12)
                self.assertEqual(r_lead_empty["rejected_frames"], 1)
                self.assertTrue(r_lead_empty["kernel_warm_ready"])

            # 6. Staggered multi-coin bars (6 BTC + 6 ETH with warmup_bars=12) MUST NOT claim kernel_warm_ready
            cfg_stag = ShadowMonitorConfig(
                state_dir=Path(td) / "m_stag",
                coins=("BTC", "ETH"),
                kernel_bar_interval_sec=43_200,
                enforce_wall_clock_bounds=False,
            )
            staggered_snaps = [
                {"BTC": _make_pair_snap("BTC", t0 - (12 - i) * bar_ms)}
                if i < 6
                else {"ETH": _make_pair_snap("ETH", t0 - (12 - i) * bar_ms, spot_mid=2_500.0)}
                for i in range(12)
            ]
            with W3DNShadowMonitor(cfg_stag, provider=ReplayQuoteProvider([]), kernel=Web3DeltaNeutralCarryKernel(warmup_bars=12)) as mon_s:
                r_st = mon_s.prewarm_from_snapshots(staggered_snaps)
                self.assertEqual(r_st["prewarmed_bars_added"], 12)
                self.assertFalse(r_st["kernel_warm_ready"])

            # 7. Overflow-safe midpoint on 1e308 quotes & fail-closed on inf basis ratio
            extreme_ok = PairMarketSnapshot(
                coin="BTC",
                ts_observe_ms=t0,
                spot_bid=1e308,
                spot_ask=1.5e308,
                spot_bid_qty=10.0,
                spot_ask_qty=10.0,
                spot_ts_ms=t0,
                perp_bid=1e308,
                perp_ask=1.5e308,
                perp_mark=1.25e308,
                perp_oracle=1.25e308,
                perp_funding_1h=0.0001,
                perp_ts_ms=t0,
                dvol=55.0,
                provenance=PROVENANCE_REAL,
            )
            cfg_ext = ShadowMonitorConfig(
                state_dir=Path(td) / "m_ext",
                coins=("BTC",),
                kernel_bar_interval_sec=43_200,
                enforce_wall_clock_bounds=False,
            )
            with W3DNShadowMonitor(cfg_ext, provider=ReplayQuoteProvider([{"BTC": extreme_ok}])) as mon_ext:
                res_ext = mon_ext.run_once()
                self.assertEqual(res_ext["status"], "HEALTHY")
                serialized_ext = json.dumps(res_ext, allow_nan=False)
                self.assertIn("BTC", serialized_ext)

    def test_waterfall_provider_handles_json_null_without_type_error(self) -> None:
        class _NullClient:
            def fetch_json(self, url: str, method: str = "GET", json_payload: Any = None, **_kw: Any) -> Any:
                if "host1" in url:
                    return {"bidPrice": None, "askPrice": "60001", "bidQty": "1", "askQty": "1"}
                if "host2" in url:
                    return {"bidPrice": "60000", "askPrice": "60001", "bidQty": None, "askQty": "1"}
                if "host3" in url:
                    return {"bidPrice": "60000", "askPrice": "60001", "bidQty": "2", "askQty": "3"}
                if json_payload and json_payload.get("type") == "metaAndAssetCtxs":
                    return [{"universe": [{"name": "BTC"}]}, [{"markPx": "60005", "oraclePx": "60005", "funding": "0.0001"}]]
                if json_payload and json_payload.get("type") == "l2Book":
                    return {"levels": [[{"px": None, "sz": "1.0"}], [{"px": "60010", "sz": None}]], "time": None}
                return {"result": {"data": []}}

        prov = PublicRestWaterfallQuoteProvider(
            client=_NullClient(),  # type: ignore[arg-type]
            spot_bases=("https://host1.invalid", "https://host2.invalid", "https://host3.invalid"),
        )
        spot = prov._fetch_binance_spot_bbo("BTC")
        self.assertIsNotNone(spot)
        assert spot is not None
        self.assertEqual(spot[5], "https://host3.invalid")

        snaps = prov.fetch_snapshots(("BTC",), trace_id="test-null-failover")
        self.assertIn("BTC", snaps)
        self.assertAlmostEqual(snaps["BTC"].spot_bid, 60_000.0)
        self.assertIsNone(snaps["BTC"].perp_bid)
        self.assertIsNone(snaps["BTC"].perp_ask)


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
            daemon = ShadowPortfolioDaemon(Path(td) / "d1", bbo_symbols=["BTC"])
            try:
                poison_ticks = [
                    BBOTick("hyperliquid", "BTC", now_ms, float("nan"), 60_001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    BBOTick("hyperliquid", "BTC", now_ms, 60_000.0, float("inf"), 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", now_ms, True, 60_001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL),  # type: ignore[arg-type]
                    BBOTick("gate", "BTC", now_ms, None, 60_001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL),  # type: ignore[arg-type]
                    BBOTick("gate", "BTC", now_ms, 1e308, 1e308, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", now_ms, 60_000.0, 60_001.0, float("nan"), 1.0, 5.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", now_ms, 60_000.0, 60_001.0, 1.0, 1.0, float("inf"), PROVENANCE_REAL),
                    BBOTick("gate", "BTC", True, 60_000.0, 60_001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL),  # type: ignore[arg-type]
                ]
                rep = daemon.run_once(now_ms=now_ms, live_ticks=poison_ticks, collect_live=False)
                self.assertEqual(rep["status"], "DEGRADED_HALT")
                self.assertEqual(rep["detail"], "NO_BBO_TICKS_AVAILABLE")
                self.assertEqual(rep["plans_created"], 0)
                self.assertEqual(rep["parity_depth"], PARITY_DEPTH_VACUOUS)
                self.assertFalse(rep["non_vacuous_parity"])
                self.assertFalse(rep["all_strategies_non_vacuous"])
                self.assertEqual(daemon._latest_bbo_cache, {})
                self.assertIsNone(daemon.store.load_latest_bbo_tick("gate", "BTC"))
                self.assertIsNone(daemon.store.load_latest_bbo_tick("hyperliquid", "BTC"))
                disk_report = json.loads((Path(td) / "d1" / "shadow_parity_report.json").read_text(encoding="utf-8"))
                self.assertEqual(disk_report["parity_depth"], PARITY_DEPTH_VACUOUS)
                self.assertFalse(disk_report["non_vacuous_parity"])
                self.assertEqual(disk_report["plans_created"], 0)

                # Valid hyperliquid + poisoned bool gate MUST still fail closed (never persist bid=1.0!)
                mixed_ticks = [
                    BBOTick("hyperliquid", "BTC", now_ms, 60_000.0, 60_001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", now_ms, True, 60_001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL),  # type: ignore[arg-type]
                ]
                rep_mix = daemon.run_once(now_ms=now_ms, live_ticks=mixed_ticks, collect_live=False)
                self.assertEqual(rep_mix["status"], "DEGRADED_HALT")
                self.assertEqual(rep_mix["detail"], "MISSING_VENUE_BBO(gate)")
                self.assertIsNone(daemon.store.load_latest_bbo_tick("gate", "BTC"))

                # collect_live=True filters collected ticks before upsert and does NOT double-upsert
                collected_batch = [
                    BBOTick("hyperliquid", "BTC", now_ms + 1_000, 60_010.0, 60_011.0, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", now_ms + 1_000, 60_010.0, 60_011.0, 1.0, 1.0, 5.0, PROVENANCE_REAL),
                    BBOTick("gate", "BTC", now_ms + 1_000, True, 60_011.0, 1.0, 1.0, 5.0, PROVENANCE_REAL),  # type: ignore[arg-type]
                ]
                with (
                    patch.object(daemon._collector, "capture_round", return_value=(collected_batch, {})),
                    patch.object(daemon.store, "upsert_bbo_ticks", wraps=daemon.store.upsert_bbo_ticks) as spy_upsert,
                ):
                    rep_live = daemon.run_once(now_ms=now_ms + 1_000, collect_live=True)
                self.assertEqual(rep_live["status"], "PARITY_OK")
                self.assertEqual(spy_upsert.call_count, 1)
                self.assertEqual(len(spy_upsert.call_args[0][0]), 2)
            finally:
                daemon.close()

            # Dynamic bridge.LIVE_EXECUTION_ENABLED monkeypatch after init MUST raise RuntimeError on run_once()
            daemon2 = ShadowPortfolioDaemon(Path(td) / "d2", bbo_symbols=["BTC"])
            try:
                with patch.object(bridge_module, "LIVE_EXECUTION_ENABLED", True):
                    with self.assertRaises(RuntimeError):
                        daemon2.run_once(now_ms=now_ms, live_ticks=[], collect_live=False)
            finally:
                daemon2.close()

    def test_shadow_auditor_window_validation_cta_ohlc_filter_and_bool_counter_rejection(self) -> None:
        cfg_carry = self._make_carry_cfg(lookback_hours=24)
        cfg_cta = StrategyConfig(
            strategy_id="trend_cta_test",
            version="v1",
            family="trend_cta",
            status="SHADOW_LIVE",
            plugin_id="trend_cta",
            venue_a="hyperliquid",
            venue_b="",
            mode="perp",
            symbols=["BTC"],
            capital_usd=10_000.0,
            lookback_hours=24,
            slippage_per_leg_bps=3.0,
            params={"timeframe_hours": 4},
        )
        win_start = 1_795_000_000_000
        win_start_bucket = (win_start // (4 * 3_600_000)) * (4 * 3_600_000)
        hour_ms = 3_600_000

        with tempfile.TemporaryDirectory(prefix="runner_cta_hard_") as td:
            store = MarketDataStore(Path(td) / "market.sqlite")
            auditor = ShadowParityAuditor(Path(td) / "parity.sqlite", store)
            try:
                # 1. Invalid window_start_ms rejected for BOTH funding_carry and trend_cta (no bool/float/str coercion)
                for bad_win in (True, False, -1, 1.5, "1795000000000"):
                    with self.assertRaises(ValueError):
                        auditor.validate_prewindow_warmup(cfg_carry, bad_win)  # type: ignore[arg-type]
                    with self.assertRaises(ValueError):
                        auditor.validate_prewindow_warmup(cfg_cta, bad_win)  # type: ignore[arg-type]

                # 2. trend_cta warmup rejects NaN/Inf/inverted OHLC candles inside 31 x 4h buckets
                poison_candles: list[PriceBar] = []
                for b in range(1, 32):
                    bts = win_start_bucket - b * 4 * hour_ms
                    for h in range(3):
                        poison_candles.append(
                            PriceBar("hyperliquid", "BTC", bts + h * hour_ms, 60_000.0, 60_100.0, 59_900.0, 60_050.0, 100.0, PROVENANCE_REAL)
                        )
                    # 4th candle in every bucket is poisoned (Inf / 'NaN' stored in SQLite, or inverted OHLC)
                    if b % 2 == 1:
                        poison_candles.append(
                            PriceBar("hyperliquid", "BTC", bts + 3 * hour_ms, float("inf"), float("inf"), float("-inf"), "NaN", 100.0, PROVENANCE_REAL)  # type: ignore[arg-type]
                        )
                    else:
                        poison_candles.append(
                            PriceBar("hyperliquid", "BTC", bts + 3 * hour_ms, 60_000.0, 59_000.0, 61_000.0, 60_000.0, 100.0, PROVENANCE_REAL)
                        )
                store.upsert_candles(poison_candles)
                res_poison = auditor.validate_prewindow_warmup(cfg_cta, win_start)
                self.assertFalse(res_poison["sufficient"])
                self.assertEqual(res_poison["observed_bars"], 0)

                # Replace 4th candle in each bucket with finite valid OHLC -> warmed (31/31)!
                valid_4th = [
                    PriceBar("hyperliquid", "BTC", (win_start_bucket - b * 4 * hour_ms) + 3 * hour_ms, 60_000.0, 60_100.0, 59_900.0, 60_050.0, 100.0, PROVENANCE_REAL)
                    for b in range(1, 32)
                ]
                store.upsert_candles(valid_4th)
                res_valid = auditor.validate_prewindow_warmup(cfg_cta, win_start)
                self.assertTrue(res_valid["sufficient"])
                self.assertEqual(res_valid["observed_bars"], 31)

                # 3. evaluate_strategy rejects bool plans_created / plans_blocked / open_positions
                for bad_kw in (
                    {"plans_created": True, "plans_blocked": 0, "open_positions": 0},
                    {"plans_created": 0, "plans_blocked": False, "open_positions": 0},
                    {"plans_created": 0, "plans_blocked": 0, "open_positions": True},
                ):
                    with self.assertRaises(ValueError):
                        auditor.evaluate_strategy(
                            cfg_carry,
                            window_start_ms=win_start,
                            window_end_ms=win_start + 2 * hour_ms,
                            shadow_equity_usd=10_000.0,
                            live_half_spreads_bps=[1.0],
                            bbo_freshness_sec=5.0,
                            audit_chain_valid=True,
                            kill_switch_engaged=False,
                            **bad_kw,  # type: ignore[arg-type]
                        )
            finally:
                auditor.close()
                store.close()


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

    def test_std46_audit_contradictory_sleeves_half_vacuous_fx_bool_and_apples_to_apples_ablation(self) -> None:
        self.assertTrue(_SCHQ_GATE_SCRIPT.is_file(), f"Missing SCHQ verifier CLI: {_SCHQ_GATE_SCRIPT}")

        # 1. Contradictory sleeve indicators MUST fail closed
        contradictory_sleeves = [
            {"sleeve_id": "CONTRA_1", "gates_passed": 8, "passed_8_of_8": False, "currency": "USD"},
            {"sleeve_id": "CONTRA_2", "gates_passed": 8, "gates": "7/8", "currency": "USD"},
            {"sleeve_id": "CONTRA_3", "gates_passed": 8, "classification": "STATIC_ALLOCATION_DOMINATED", "currency": "USD"},
            {"sleeve_id": "CONTRA_4", "gates_passed": 9, "gates_total": 8, "currency": "USD"},
            {"sleeve_id": "CONTRA_5", "gates_passed": 8, "eligible_for_production": False, "currency": "USD"},
            {"sleeve_id": "CONTRA_6", "gates_passed": 8, "dominated_by_static": True, "currency": "USD"},
            {"sleeve_id": "CONTRA_7", "gates_passed": 8, "weight": 0.5, "weight_final": -0.9, "currency": "USD"},
        ]
        res_contra = audit_composite_portfolio(contradictory_sleeves, {"USD"}, fx_hedged=False)
        self.assertFalse(res_contra["eligible_for_production"])
        self.assertEqual(res_contra["classification"], "DIAGNOSTIC_HYPOTHETICAL_ONLY")
        self.assertEqual(
            res_contra["failing_sleeves"],
            ["CONTRA_1", "CONTRA_2", "CONTRA_3", "CONTRA_4", "CONTRA_5", "CONTRA_6", "CONTRA_7"],
        )

        # Dual consistent weight aliases (weight == weight_final == 0.5) must NOT double-count above 1.0
        dual_w_ok = [
            {"sleeve_id": "W_A", "gates_passed": 8, "gates_total": 8, "weight": 0.5, "weight_final": 0.5, "currency": "USD"},
            {"sleeve_id": "W_B", "gates_passed": 8, "gates_total": 8, "weight": 0.5, "weight_final": 0.5, "currency": "USD"},
        ]
        res_dw = audit_composite_portfolio(dual_w_ok, {"USD"}, fx_hedged=False)
        self.assertTrue(res_dw["eligible_for_production"])
        self.assertEqual(res_dw["classification"], "PRODUCTION_ELIGIBLE")

        # 2. build_std46_audit_report rejects string fx_hedged="false" and malformed non-list sleeves
        with self.assertRaises(ValueError):
            build_std46_audit_report(
                report_name="bad_fx_str",
                composites=[
                    {
                        "portfolio_id": "c_bad_fx",
                        "currencies": ["USD", "RMB"],
                        "fx_hedged": "false",
                        "cagr_pct": 8.0,
                        "ann_vol_pct": 2.0,
                        "sleeves": [{"sleeve_id": "S1", "gates_passed": 8, "gates_total": 8, "currency": "USD"}],
                    }
                ],
            )
        with self.assertRaises(ValueError):
            build_std46_audit_report(
                report_name="bad_sleeves_str",
                composites=[
                    {
                        "portfolio_id": "c_bad_sleeves",
                        "currencies": ["USD"],
                        "fx_hedged": False,
                        "sleeves": "bad",
                        "status": "PRODUCTION",
                    }
                ],
            )

        # 3. Half-vacuous shadows, alias-gate production composite & apples-to-apples ablation pass SCHQ --mode report
        rep = build_std46_audit_report(
            report_name="std46_iter2_parity_report",
            strategies=[
                {
                    "strategy_id": "RAW_BEAT_STATIC",
                    "currency": "USD",
                    "cagr_pct": 5.0,
                    "ann_vol_pct": 5.5556,
                    "raw_sharpe": 0.90,
                    "max_dd_pct": 5.0,
                    "calmar": 1.0,
                    "static_metrics": {"cagr_pct": 5.0, "sharpe": 0.85, "max_dd_pct": 5.0, "calmar": 1.0},
                }
            ],
            composites=[
                {
                    "portfolio_id": "prod_alias_gates",
                    "currencies": ["USD"],
                    "fx_hedged": False,
                    "cagr_pct": 10.0,
                    "ann_vol_pct": 3.0,
                    "sleeves": [
                        {"sleeve_id": "S1", "passed_gates": 8, "total_gates": 8, "currency": "USD"},
                        {"sleeve_id": "S2", "passed_8_of_8": True, "currency": "USD"},
                    ],
                },
                {
                    "portfolio_id": "diag_contra_sleeve",
                    "currencies": ["USD"],
                    "fx_hedged": False,
                    "cagr_pct": 9.0,
                    "ann_vol_pct": 3.0,
                    "sleeves": [
                        {"sleeve_id": "BAD_CAP", "gates_passed": 8, "gates_total": 8, "verdict_cap": "STATIC_ALLOCATION_DOMINATED", "currency": "USD"},
                    ],
                },
            ],
            shadows=[
                {"shadow_id": "sh_plans_gt0_trades_0", "plans_created": 5, "executed_trades": 0, "status": "SHADOW_READY"},
                {"shadow_id": "sh_plans_0_trades_gt0", "plans_created": 0, "executed_trades": 3, "status": "PARITY_OK"},
                {"shadow_id": "sh_alias_vacuous", "orders_planned": 2, "fills": 0, "status": "SHADOW_READY"},
                {"shadow_id": "sh_bool_counter", "plans_created": True, "executed_trades": 2, "status": "SHADOW_READY"},
                {"shadow_id": "sh_missing_counters", "status": "SHADOW_READY"},
            ],
        )

        # Apples-to-apples ablation check: raw_sharpe=0.90 > static sharpe=0.85 -> NOT dominated
        self.assertFalse(rep["strategies"][0]["static_ablation"]["dominated_by_static"])
        self.assertAlmostEqual(rep["strategies"][0]["static_ablation"]["deltas"]["sharpe"], 0.05, places=6)

        # Canonicalized sleeve gates on production composite
        prod_c = rep["composites"][0]
        self.assertTrue(prod_c["eligible_for_production"])
        self.assertEqual(prod_c["sleeves"][0]["gates_passed"], 8)
        self.assertEqual(prod_c["sleeves"][0]["gates_total"], 8)
        self.assertEqual(prod_c["sleeves"][1]["gates_passed"], 8)
        self.assertEqual(prod_c["sleeves"][1]["gates_total"], 8)

        # Disqualifying verdict_cap synced into sleeve verdict/status
        diag_c = rep["composites"][1]
        self.assertFalse(diag_c["eligible_for_production"])
        self.assertEqual(diag_c["sleeves"][0]["verdict"], "STATIC_ALLOCATION_DOMINATED")

        # All 5 vacuous/half-vacuous shadows capped properly
        self.assertEqual(rep["shadows"][0]["status"], "SHADOW_VACUOUS_ZERO_TRADES")
        self.assertFalse(rep["shadows"][0]["non_vacuous_parity"])
        self.assertEqual(rep["shadows"][1]["status"], "VACUOUS_ZERO_PLANS")
        self.assertFalse(rep["shadows"][1]["non_vacuous_parity"])
        self.assertEqual(rep["shadows"][2]["status"], "SHADOW_VACUOUS_ZERO_TRADES")
        self.assertEqual(rep["shadows"][3]["status"], "VACUOUS_ZERO_PLANS")
        self.assertEqual(rep["shadows"][4]["status"], "VACUOUS_ZERO_PLANS")

        with tempfile.TemporaryDirectory(prefix="std46_iter2_gate_") as td:
            rp = Path(td) / "iter2_report.json"
            rp.write_text(json.dumps(rep, indent=2, allow_nan=False), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-B", str(_SCHQ_GATE_SCRIPT), "--mode", "report", str(rp), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"SCHQ --mode report failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertEqual(json.loads(proc.stdout)["status"], "PASS")


class TestTier5AutoAdversarialHardening(unittest.TestCase):
    """Tier-5 adversarial regression tests for GAP-C2-04..08 (zero touch on validation.py)."""

    def test_gap_c2_04_w3_dn_multithread_wal_audit_chain_integrity(self) -> None:
        base_ts = 1_795_000_000_000
        frames = [
            {"BTC": _make_pair_snap("BTC", base_ts + i * 60_000, spot_mid=90_000.0, basis_bps=1.0)}
            for i in range(65)
        ]
        with tempfile.TemporaryDirectory(prefix="w3dn_mt_chain_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td),
                coins=("BTC",),
                observe_only_coins=(),
                enforce_wall_clock_bounds=False,
                enable_prewarm=False,
            )
            mon = W3DNShadowMonitor(config=cfg, provider=ReplayQuoteProvider(frames))
            try:
                errors: list[str] = []
                barrier = threading.Barrier(4)

                def _worker() -> None:
                    barrier.wait()
                    for _ in range(15):
                        try:
                            mon.run_once()
                            mon.evaluate_48h_readiness()
                        except Exception as exc:  # noqa: BLE001
                            errors.append(f"{type(exc).__name__}: {exc}")

                threads = [threading.Thread(target=_worker) for _ in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=30.0)

                self.assertEqual(errors, [])
                chain_ok, chain_len = mon.verify_audit_chain()
                self.assertTrue(chain_ok)
                self.assertEqual(chain_len, 60)
                readiness = mon.evaluate_48h_readiness()
                self.assertTrue(readiness.audit_chain_intact)
                self.assertEqual(readiness.audit_chain_length, 60)
                self.assertEqual(readiness.total_ticks, 60)
            finally:
                mon.close()

    def test_gap_c2_05_310_digit_int_overflow_inf_candles_and_bbo_tick_bounds(self) -> None:
        from alphaloop.daemon import shadow_runner as sr
        from alphaloop.data import bbo_collector as bc
        from alphaloop.platform.shadow import w3_dn_shadow_monitor as w3

        huge_int = 10**310 - 1
        self.assertIsNone(bc._parse_finite_float(huge_int))
        self.assertFalse(sr._is_finite(huge_int))
        self.assertFalse(w3._is_pos_finite(huge_int))
        self.assertFalse(w3._is_finite(huge_int))
        with self.assertRaises(ValueError):
            compute_dual_sharpe(cagr_pct=huge_int, ann_vol_pct=10.0)

        # bbo_collector capture_round survives 310-digit RFC-8259 JSON integer from upstream without crashing
        collector = MultiVenueBBOCollector(symbols=["BTC"])
        def _fake_http(url: str, *args: Any, **kwargs: Any) -> Any:
            if "okx.com" in url:
                return ({"code": "0", "data": [{"instId": "BTC-USDT-SWAP", "bidPx": huge_int, "askPx": 90001.0, "bidSz": 1.0, "askSz": 1.0, "ts": 1_795_000_000_000}]}, 5.0)
            if "gateio.ws" in url:
                return ([{"contract": "BTC_USDT", "highest_bid": "90000.0", "lowest_ask": "90001.0", "highest_size": "1.0", "lowest_size": "1.0"}], 5.0)
            if "hyperliquid.xyz" in url:
                return ([{"universe": [{"name": "BTC"}]}, [{"impactPxs": ["90000.0", "90001.0"]}]], 5.0)
            if "lighter" in url:
                return ({"order_books": []}, 5.0)
            return ({}, 5.0)

        with patch.object(bc, "_http_json", side_effect=_fake_http):
            ticks, _stats = collector.capture_round()
        self.assertGreaterEqual(len(ticks), 1)
        self.assertTrue(all(math.isfinite(t.bid) and math.isfinite(t.ask) for t in ticks))

        # _is_valid_bbo_tick enforces _MIN_VALID_EPOCH_MS, catches OverflowError, and rejects synthetic provenance
        now_ms = 1_795_000_000_000
        self.assertTrue(sr._is_valid_bbo_tick(BBOTick("hyperliquid", "BTC", now_ms, 90000.0, 90001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL)))
        self.assertFalse(sr._is_valid_bbo_tick(BBOTick("hyperliquid", "BTC", now_ms, huge_int, 90001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL)))  # type: ignore[arg-type]
        self.assertFalse(sr._is_valid_bbo_tick(BBOTick("hyperliquid", "BTC", 1, 90000.0, 90001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL)))
        self.assertFalse(sr._is_valid_bbo_tick(BBOTick("hyperliquid", "BTC", 1_795_000_000, 90000.0, 90001.0, 1.0, 1.0, 5.0, PROVENANCE_REAL)))
        for bad_prov in ("", "   ", None, 123, "L4_SYNTHETIC_OR_MOCK", "MOCK", "synthetic"):
            self.assertFalse(
                sr._is_valid_bbo_tick(BBOTick("hyperliquid", "BTC", now_ms, 90000.0, 90001.0, 1.0, 1.0, 5.0, bad_prov))  # type: ignore[arg-type]
            )

        # Unfiltered inf candles must NEVER poison CTA warmup or carry hourly_log_rets (save_runtime_state allow_nan=False)
        with tempfile.TemporaryDirectory(prefix="sr_inf_candle_") as td:
            data_dir = Path(td)
            store = MarketDataStore(data_dir / "market_data.sqlite")
            t_hour = (now_ms // 3_600_000) * 3_600_000
            good_and_inf = [
                PriceBar("hyperliquid", "BTC", t_hour - 5 * 3_600_000, float("inf"), float("inf"), float("inf"), float("inf"), 1.0, PROVENANCE_REAL),
                PriceBar("hyperliquid", "BTC", t_hour - 4 * 3_600_000, 100.0, 101.0, 99.0, 100.5, 10.0, PROVENANCE_REAL),
                PriceBar("hyperliquid", "BTC", t_hour - 3 * 3_600_000, 100.5, 102.0, 100.0, 101.0, 10.0, PROVENANCE_REAL),
                PriceBar("hyperliquid", "BTC", t_hour - 2 * 3_600_000, 101.0, 102.5, 100.5, 101.5, 10.0, PROVENANCE_REAL),
                PriceBar("hyperliquid", "BTC", t_hour - 1 * 3_600_000, 101.5, 103.0, 101.0, 102.0, 10.0, PROVENANCE_REAL),
            ]
            store.upsert_candles(good_and_inf)
            store.close()
            cfg_carry = StrategyConfig(
                strategy_id="carry_inf_test",
                version="v1",
                family="funding_carry",
                status="active",
                mode="shadow",
                plugin_id="funding_carry",
                venue_a="hyperliquid",
                venue_b="gate",
                symbols=("BTC",),
                capital_usd=1000.0,
                params={"vol_target_leverage_enabled": True, "vol_lookback_hours": 24},
            )
            cfg_cta = StrategyConfig(
                strategy_id="cta_inf_test",
                version="v1",
                family="trend_cta",
                status="active",
                mode="shadow",
                plugin_id="trend_cta",
                venue_a="hyperliquid",
                venue_b="",
                symbols=("BTC",),
                capital_usd=1000.0,
                params={"timeframe_hours": 4, "entry_lookback": 1, "atr_period": 2, "ema_period": 2},
            )
            daemon = ShadowPortfolioDaemon(data_dir=data_dir)
            try:
                carry_state: dict[str, Any] = {}
                daemon._refresh_carry_hourly_log_rets(cfg_carry, carry_state, t_hour + 60_000)
                rmap = carry_state.get("hourly_log_rets", {}).get("BTC", {})
                self.assertNotIn(t_hour - 5 * 3_600_000, rmap)
                self.assertTrue(all(math.isfinite(v) for v in rmap.values()))
                json.dumps(carry_state, allow_nan=False)

                cta_state: dict[str, Any] = {}
                daemon._warmup_plugin_state_from_candles(cfg_cta, "sha_cta_test", cta_state, t_hour + 60_000)
                json.dumps(cta_state, allow_nan=False)
                daemon.parity.save_runtime_state(
                    cfg_carry.strategy_id,
                    started_ts_ms=t_hour,
                    realized_pnl_usd=0.0,
                    plans_created=0,
                    plans_blocked=0,
                    open_positions={},
                    updated_ts_ms=t_hour + 60_000,
                    plugin_state=carry_state,
                )
                daemon.parity.save_runtime_state(
                    cfg_cta.strategy_id,
                    started_ts_ms=t_hour,
                    realized_pnl_usd=0.0,
                    plans_created=0,
                    plans_blocked=0,
                    open_positions={},
                    updated_ts_ms=t_hour + 60_000,
                    plugin_state=cta_state,
                )
            finally:
                daemon.close()

    def test_gap_c2_06_std46_audit_mixed_family_gates_and_none_sentinels(self) -> None:
        bad_sleeves = [
            {"sleeve_id": "S_MIX1", "gates_passed": 8, "total_gates": 10, "currency": "USD"},
            {"sleeve_id": "S_MIX2", "passed_gates": 8, "gates_total": 10, "currency": "USD"},
            {"sleeve_id": "S_CONTRA_TOT", "gates_passed": 8, "gates_total": 8, "total_gates": 10, "currency": "USD"},
            {"sleeve_id": "S_CONTRA_RATIO", "gates_passed": 8, "gates_total": 8, "gates": "10/10", "currency": "USD"},
        ]
        for sl in bad_sleeves:
            res = audit_composite_portfolio([sl], {"USD"}, fx_hedged=False)
            self.assertFalse(res["eligible_for_production"], msg=f"Must reject {sl}")
            self.assertEqual(res["classification"], "DIAGNOSTIC_HYPOTHETICAL_ONLY")

        rep = build_std46_audit_report(
            report_name="none_sentinel_test",
            strategies=[
                {"strategy_id": "S1", "cagr_pct": 12.0, "ann_vol_pct": 10.0, "currency": "USD", "excess_sharpe": None, "raw_sharpe": None},
            ],
            composites=[
                {
                    "portfolio_id": "C1",
                    "currencies": ["USD"],
                    "fx_hedged": False,
                    "cagr_pct": 12.0,
                    "ann_vol_pct": 10.0,
                    "raw_sharpe": None,
                    "excess_sharpe": None,
                    "sleeves": [{"sleeve_id": "S1", "gates_passed": 8, "gates_total": 8, "currency": "USD"}],
                },
            ],
        )
        self.assertIsNotNone(rep["strategies"][0]["excess_sharpe"])
        self.assertIsNotNone(rep["strategies"][0]["raw_sharpe"])
        self.assertIsNotNone(rep["composites"][0]["excess_sharpe"])
        self.assertIsNotNone(rep["composites"][0]["raw_sharpe"])

        with tempfile.TemporaryDirectory(prefix="std46_c2_06_") as td:
            rp = Path(td) / "none_sentinel_report.json"
            rp.write_text(json.dumps(rep, indent=2, allow_nan=False), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-B", str(_SCHQ_GATE_SCRIPT), "--mode", "report", str(rp), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"SCHQ --mode report failed: {proc.stdout}\n{proc.stderr}")

    def test_gap_c2_07_deribit_null_result_bool_dvol_and_p8_bool_bid(self) -> None:
        from alphaloop.platform.shadow.w3_dn_shadow_monitor import P8MDGatewayFileQuoteProvider

        class _StubClient:
            def __init__(self, payloads: list[Any]) -> None:
                self._payloads = list(payloads)
                self._idx = 0

            def fetch_json(self, *_a: Any, **_kw: Any) -> Any:
                val = self._payloads[min(self._idx, len(self._payloads) - 1)]
                self._idx += 1
                return val

        # Deribit {"result": null} must not raise AttributeError
        prov_null = PublicRestWaterfallQuoteProvider(client=_StubClient([{"result": None}]))  # type: ignore[arg-type]
        self.assertIsNone(prov_null._fetch_dvol_cached())

        # Deribit bool DVOL True must be rejected (never coerced to 1.0)
        prov_bool = PublicRestWaterfallQuoteProvider(client=_StubClient([{"result": {"data": [[1, 2, 3, 4, True]]}}]))  # type: ignore[arg-type]
        self.assertIsNone(prov_bool._fetch_dvol_cached())

        # P8 JSONL bool bid/ask/ts must be rejected
        with tempfile.TemporaryDirectory(prefix="p8_bool_") as td:
            feed = Path(td) / "p8.jsonl"
            feed.write_text(
                "\n".join([
                    json.dumps({"symbol": "BTC", "bid": True, "ask": 2.0, "bid_qty": 1.0, "ask_qty": 1.0, "ts_ms": 1_795_000_000_000}, allow_nan=False),
                    json.dumps({"symbol": "ETH", "bid": 3000.0, "ask": 3001.0, "bid_qty": True, "ask_qty": 1.0, "ts_ms": 1_795_000_000_000}, allow_nan=False),
                    json.dumps({"symbol": "SOL", "bid": 150.0, "ask": 151.0, "bid_qty": 1.0, "ask_qty": 1.0, "ts_ms": True}, allow_nan=False),
                ]) + "\n",
                encoding="utf-8",
            )
            p8 = P8MDGatewayFileQuoteProvider(p8_feed_path=feed, perp_provider=prov_null)
            parsed = p8._read_latest_p8_spot_bbo(["BTC", "ETH", "SOL"])
            self.assertEqual(parsed, {})

    def test_gap_c2_08_prewarm_max_bars_strict_int_validation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="w3dn_max_bars_") as td:
            cfg = ShadowMonitorConfig(
                state_dir=Path(td),
                coins=("BTC",),
                observe_only_coins=(),
                enforce_wall_clock_bounds=False,
                enable_prewarm=False,
            )
            mon = W3DNShadowMonitor(config=cfg, provider=ReplayQuoteProvider([]))
            try:
                for bad_mb in (1.9, "5", float("inf"), float("nan"), {}, True, -1):
                    with self.assertRaises(ValueError, msg=f"prewarm_from_snapshots must reject {bad_mb!r}"):
                        mon.prewarm_from_snapshots([], max_bars=bad_mb)  # type: ignore[arg-type]
                    with self.assertRaises(ValueError, msg=f"prewarm_from_bars must reject {bad_mb!r}"):
                        mon.prewarm_from_bars([], max_bars=bad_mb)  # type: ignore[arg-type]
            finally:
                mon.close()

    def test_gap_c2r2_01_static_ablation_forbids_incompatible_units_and_sharpe_families(self) -> None:
        # 1. raw_sharpe vs excess_sharpe MUST raise ValueError (no apples-to-oranges Sharpe bypass)
        with self.assertRaises(ValueError):
            evaluate_static_ablation(
                {"cagr_pct": 4.37, "raw_sharpe": 0.85, "maxdd_pct": 6.88, "calmar": 0.63},
                {"cagr_pct": 4.60, "excess_sharpe": 0.60, "maxdd_pct": 6.34, "calmar": 0.72},
            )
        # 2. cagr_pct vs decimal cagr MUST raise ValueError
        with self.assertRaises(ValueError):
            evaluate_static_ablation(
                {"cagr_pct": 4.37, "excess_sharpe": 0.55, "maxdd_pct": 6.88, "calmar": 0.63},
                {"cagr": 0.046, "excess_sharpe": 0.60, "maxdd_pct": 6.34, "calmar": 0.72},
            )
        # 3. maxdd_pct vs decimal max_dd MUST raise ValueError
        with self.assertRaises(ValueError):
            evaluate_static_ablation(
                {"cagr_pct": 4.37, "excess_sharpe": 0.55, "maxdd_pct": 6.88, "calmar": 0.63},
                {"cagr_pct": 4.60, "excess_sharpe": 0.60, "max_dd": 0.0634, "calmar": 0.72},
            )
        # 4. build_std46_audit_report must NOT launder cagr_pct vs cagr past SCHQ --mode report
        with self.assertRaises(ValueError):
            build_std46_audit_report(
                report_name="v3_launder_attempt",
                strategies=[
                    {
                        "strategy_id": "S_MIX",
                        "status": "PROMOTE",
                        "cagr_pct": 4.37,
                        "ann_vol_pct": 5.14,
                        "rf_pct": 4.5,
                        "maxdd_pct": 6.88,
                        "calmar": 0.63,
                        "static_metrics": {"cagr": 0.046, "excess_sharpe": 0.60, "maxdd_pct": 6.34, "calmar": 0.72},
                    }
                ],
            )
        # 5. Valid same-family aliases still work cleanly (max_dd_pct vs maxdd_pct, sharpe vs sharpe_ratio, calmar vs calmar_ratio)
        res_same = evaluate_static_ablation(
            {"cagr_pct": 4.37, "sharpe": 0.85, "max_dd_pct": 6.88, "calmar": 0.63},
            {"cagr_pct": 4.60, "sharpe_ratio": 0.88, "maxdd_pct": 6.34, "calmar_ratio": 0.72},
        )
        self.assertTrue(res_same["dominated_by_static"])
        self.assertEqual(res_same["verdict_cap"], "STATIC_ALLOCATION_DOMINATED")

        # 6. Decimal cagr/max_dd family preserved without misleading *_pct re-labeling
        res_dec = evaluate_static_ablation(
            {"cagr": 0.0437, "excess_sharpe": 0.55, "max_dd": 0.0688, "calmar": 0.63},
            {"cagr": 0.0460, "excess_sharpe": 0.60, "maxdd": 0.0634, "calmar": 0.72},
        )
        self.assertTrue(res_dec["dominated_by_static"])
        self.assertIn("cagr", res_dec["strategy_metrics"])
        self.assertNotIn("cagr_pct", res_dec["strategy_metrics"])
        self.assertIn("excess_sharpe", res_dec["strategy_metrics"])

    def test_gap_c2r2_02_passed_8_of_8_contradicts_explicit_total_and_schq_parity(self) -> None:
        # Contradictory passed_8_of_8=True vs total_gates/gates_total != 8 MUST fail closed
        for bad_sleeve in (
            {"sleeve_id": "S_TOT10", "passed_8_of_8": True, "total_gates": 10, "currency": "USD"},
            {"sleeve_id": "S_GT10", "passed_8_of_8": True, "gates_total": 10, "currency": "USD"},
            {"sleeve_id": "S_TOT6", "passed_8_of_8": True, "total_gates": 6, "currency": "USD"},
        ):
            res = audit_composite_portfolio([bad_sleeve], {"USD"}, fx_hedged=False)
            self.assertFalse(res["eligible_for_production"], msg=f"Must reject {bad_sleeve}")
            self.assertEqual(res["classification"], "DIAGNOSTIC_HYPOTHETICAL_ONLY")
            self.assertEqual(res["failing_sleeves"], [bad_sleeve["sleeve_id"]])

        # Valid 10/10 with passed_8_of_8=True and valid 8/8 with passed_8_of_8=True MUST still pass
        for good_sleeve in (
            {"sleeve_id": "S_10_10", "gates_passed": 10, "gates_total": 10, "passed_8_of_8": True, "currency": "USD"},
            {"sleeve_id": "S_8_8", "passed_8_of_8": True, "gates_total": 8, "currency": "USD"},
        ):
            res_ok = audit_composite_portfolio([good_sleeve], {"USD"}, fx_hedged=False)
            self.assertTrue(res_ok["eligible_for_production"], msg=f"Must accept {good_sleeve}")
            self.assertEqual(res_ok["classification"], "PRODUCTION_ELIGIBLE")

        # Cross-repo parity: auto build_std46_audit_report output for contradictory sleeve passes SCHQ --mode report
        # cleanly as DIAGNOSTIC_HYPOTHETICAL_ONLY (no fake gates_passed=8/gates_total=8 contradiction)
        rep = build_std46_audit_report(
            report_name="gap_c2r2_02_parity",
            composites=[
                {
                    "portfolio_id": "C_CONTRA",
                    "currencies": ["USD"],
                    "fx_hedged": False,
                    "cagr_pct": 12.0,
                    "ann_vol_pct": 10.0,
                    "sleeves": [{"sleeve_id": "S1", "passed_8_of_8": True, "total_gates": 10, "currency": "USD"}],
                },
                {
                    "portfolio_id": "C_10_OF_10",
                    "currencies": ["USD"],
                    "fx_hedged": False,
                    "cagr_pct": 12.0,
                    "ann_vol_pct": 10.0,
                    "sleeves": [
                        {"sleeve_id": "S2", "gates_passed": 10, "gates_total": 10, "passed_8_of_8": True, "currency": "USD"}
                    ],
                },
            ],
        )
        self.assertEqual(rep["composites"][0]["classification"], "DIAGNOSTIC_HYPOTHETICAL_ONLY")
        self.assertEqual(rep["composites"][0]["status"], "DIAGNOSTIC_HYPOTHETICAL_ONLY")
        self.assertNotIn("gates_passed", rep["composites"][0]["sleeves"][0])
        self.assertEqual(rep["composites"][1]["classification"], "PRODUCTION_ELIGIBLE")

        if _SCHQ_GATE_SCRIPT.exists():
            with tempfile.TemporaryDirectory(prefix="std46_c2r2_02_") as td:
                rp = Path(td) / "parity_c2r2_02.json"
                rp.write_text(json.dumps(rep, indent=2, allow_nan=False), encoding="utf-8")
                proc = subprocess.run(
                    [sys.executable, "-B", str(_SCHQ_GATE_SCRIPT), "--mode", "report", str(rp), "--json"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=f"SCHQ --mode report failed: {proc.stdout}\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main()
