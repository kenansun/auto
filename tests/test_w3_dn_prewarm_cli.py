"""Offline unit & CLI tests for W3-DN historical --prewarm backfill."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from alphaloop.core.provenance import PROVENANCE_REAL, utc_now_ms
from alphaloop.platform.shadow.w3_dn_shadow_monitor import (
    SHADOW_LIVE_EXECUTION_ENABLED,
    PairMarketSnapshot,
    PublicRestWaterfallQuoteProvider,
    ReplayQuoteProvider,
    ShadowMonitorConfig,
    W3DNShadowMonitor,
)
from alphaloop.platform.strategies.web3.delta_neutral_carry import (
    perp_leg_symbol,
    spot_leg_symbol,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import run_w3_dn_shadow_monitor as cli_mod  # noqa: E402


class _MockHistoricalClient:
    """Deterministic offline mock for Binance Spot klines + HL candleSnapshot/fundingHistory."""

    def __init__(self, *, now_ms: int, coins: tuple[str, ...] = ("BTC", "ETH")) -> None:
        self.now_ms = int(now_ms)
        self.coins = coins
        self.calls: list[tuple[str, str]] = []

    def fetch_json(
        self,
        url: str,
        *,
        method: str = "GET",
        json_payload: dict[str, Any] | None = None,
        timeout_sec: float = 10.0,
        max_retries: int = 2,
        min_interval_sec: float = 0.0,
    ) -> Any:
        _ = (timeout_sec, max_retries, min_interval_sec)
        self.calls.append((method, url))
        bucket_ms = 43_200 * 1000
        # Anchor to 14 closed 12h buckets ending before now_ms, plus 1 unclosed future bucket
        latest_closed_idx = (self.now_ms // bucket_ms) - 1
        start_idx = latest_closed_idx - 13

        if "api/v3/klines" in url:
            rows: list[list[Any]] = []
            for idx in range(start_idx, latest_closed_idx + 2):
                open_ms = idx * bucket_ms
                close_ms = open_ms + bucket_ms - 1
                if idx == start_idx:
                    # Inject NaN row to verify fail-closed filtering
                    rows.append([open_ms, "nan", "60100", "59900", "60000", "10.0", close_ms])
                else:
                    rows.append([open_ms, "60000.0", "60200.0", "59800.0", "60100.0", "12.5", close_ms])
            return rows

        if isinstance(json_payload, dict) and json_payload.get("type") == "candleSnapshot":
            req = json_payload.get("req", {})
            coin = str(req.get("coin", "BTC"))
            candles: list[dict[str, Any]] = []
            for idx in range(start_idx, latest_closed_idx + 2):
                open_ms = idx * bucket_ms
                close_ms = open_ms + bucket_ms - 1
                candles.append({
                    "t": open_ms,
                    "T": close_ms,
                    "s": coin,
                    "i": "12h",
                    "o": "60010.0",
                    "h": "60210.0",
                    "l": "59810.0",
                    "c": "60110.0",
                    "v": "8.0",
                })
            return candles

        if isinstance(json_payload, dict) and json_payload.get("type") == "fundingHistory":
            coin = str(json_payload.get("coin", "BTC"))
            funding: list[dict[str, Any]] = []
            for idx in range(start_idx + 1, latest_closed_idx + 1):
                open_ms = idx * bucket_ms
                for hr in range(1, 13):
                    funding.append({
                        "coin": coin,
                        "fundingRate": "0.00002",  # 12 * 0.00002 = 0.00024 per 12h bar
                        "time": open_ms + hr * 3_600_000,
                    })
            return funding

        raise AssertionError(f"Unexpected mock URL/payload: {method} {url} {json_payload}")


class TestW3DNPrewarmCLI(unittest.TestCase):
    def test_safety_lock_remains_false(self) -> None:
        self.assertFalse(SHADOW_LIVE_EXECUTION_ENABLED)

    def test_fetch_historical_prewarm_bars_filters_unclosed_and_bad_rows(self) -> None:
        now_ms = utc_now_ms()
        client = _MockHistoricalClient(now_ms=now_ms, coins=("BTC", "ETH"))
        provider = PublicRestWaterfallQuoteProvider(client=client)
        frames = provider.fetch_historical_prewarm_bars(("BTC", "ETH"), bars=12, bucket_sec=43_200)
        self.assertEqual(len(frames), 12)
        for ts_ms, bmap in frames:
            self.assertLessEqual(ts_ms, now_ms)
            for coin in ("BTC", "ETH"):
                s_bar = bmap[spot_leg_symbol(coin)]
                p_bar = bmap[perp_leg_symbol(coin)]
                self.assertEqual(s_bar.provenance, PROVENANCE_REAL)
                self.assertEqual(p_bar.provenance, PROVENANCE_REAL)
                self.assertEqual(s_bar.carry_rate_period, 0.0)
                self.assertAlmostEqual(p_bar.carry_rate_period, 0.00024, places=9)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ShadowMonitorConfig(
                state_dir=Path(tmp),
                coins=("BTC", "ETH"),
                enable_prewarm=True,
                prewarm_bars=12,
            )
            with W3DNShadowMonitor(cfg, provider=provider) as mon:
                res = mon.prewarm_from_bars(frames, max_bars=12)
                self.assertEqual(res["prewarmed_bars_added"], 12)
                self.assertTrue(res["kernel_warm_ready"])
                self.assertEqual(res["executed_trades"], 0)
                self.assertEqual(res["total_ticks"], 0)
                readiness = mon.evaluate_48h_readiness()
                self.assertEqual(readiness.prewarmed_bars, 12)
                self.assertEqual(readiness.audit_chain_length, 0)

    def test_cli_prewarm_runs_before_live_tick_and_noop_after_live(self) -> None:
        now_ms = utc_now_ms()
        mock_client = _MockHistoricalClient(now_ms=now_ms, coins=("BTC",))
        hist_provider = PublicRestWaterfallQuoteProvider(client=mock_client)
        live_snap = {
            "BTC": PairMarketSnapshot(
                coin="BTC",
                ts_observe_ms=now_ms,
                spot_bid=60000.0,
                spot_ask=60010.0,
                spot_bid_qty=2.0,
                spot_ask_qty=2.0,
                spot_ts_ms=now_ms,
                perp_bid=60015.0,
                perp_ask=60025.0,
                perp_mark=60020.0,
                perp_oracle=60020.0,
                perp_funding_1h=0.00001,
                perp_ts_ms=now_ms,
                perp_l2_bids=((60015.0, 2.0),),
                perp_l2_asks=((60025.0, 2.0),),
                dvol=55.0,
                spot_source="mock",
                perp_source="mock",
                provenance=PROVENANCE_REAL,
            )
        }

        class _HybridMockProvider(ReplayQuoteProvider):
            def __init__(self) -> None:
                super().__init__([live_snap])

            def fetch_historical_prewarm_bars(
                self, coins: Any, *, bars: int = 12, bucket_sec: int = 43_200
            ) -> list[tuple[int, dict[str, Any]]]:
                return hist_provider.fetch_historical_prewarm_bars(coins, bars=bars, bucket_sec=bucket_sec)

        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "w3dn"
            report_out = Path(tmp) / "report.json"
            with patch.object(cli_mod, "PublicRestWaterfallQuoteProvider", return_value=_HybridMockProvider()):
                rc = cli_mod.main([
                    "--state-dir", str(state_dir),
                    "--coins", "BTC",
                    "--ticks", "1",
                    "--interval-sec", "0.2",
                    "--prewarm",
                    "--prewarm-bars", "12",
                    "--report-out", str(report_out),
                ])
            self.assertEqual(rc, 0)
            summary = json.loads(report_out.read_text(encoding="utf-8"))
            self.assertEqual(summary["readiness"]["prewarmed_bars"], 12)
            self.assertEqual(summary["ticks_executed"], 1)

            # Negative case: restarting with --prewarm after live ticks started must be strict no-op
            with patch.object(cli_mod, "PublicRestWaterfallQuoteProvider", return_value=_HybridMockProvider()):
                rc2 = cli_mod.main([
                    "--state-dir", str(state_dir),
                    "--coins", "BTC",
                    "--ticks", "1",
                    "--interval-sec", "0.2",
                    "--prewarm",
                    "--prewarm-bars", "12",
                    "--report-out", str(report_out),
                ])
            self.assertEqual(rc2, 0)
            summary2 = json.loads(report_out.read_text(encoding="utf-8"))
            self.assertEqual(summary2["readiness"]["prewarmed_bars"], 12)
            self.assertEqual(summary2["ticks_executed"], 1)


if __name__ == "__main__":
    unittest.main()
