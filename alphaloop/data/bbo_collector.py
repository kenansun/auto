"""Live 4-Exchange Public BBO Orderbook Collector (Hyperliquid / Lighter / OKX / Gate).

Collects synchronized top-of-book bid/ask/qty snapshots with real HTTP latency tracking
and crossed-book/NaN validation (STD-34 Pillars 2 & 6). Zero API keys required;
zero synthetic fallback on upstream failure.
"""

from __future__ import annotations

import http.client
import json
import logging
import random
import time
import urllib.error
import urllib.request
from typing import Any

from alphaloop.core.models import BBOTick
from alphaloop.core.provenance import PROVENANCE_REAL, utc_now_ms

logger = logging.getLogger(__name__)

DEFAULT_BBO_SYMBOLS = [
    "BTC", "ETH", "SOL", "VVV", "WLD", "NEAR", "ENA", "ZEC", "UNI", "HYPE",
]


def _http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
    max_retries: int = 3,
) -> tuple[Any, float]:
    """Perform HTTP GET/POST with exponential backoff + jitter and return (json, latency_ms)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"User-Agent": "AlphaLoop-BBO/2.0", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return json.loads(raw), round(latency_ms, 2)
        except (urllib.error.URLError, urllib.error.HTTPError, http.client.HTTPException, OSError, TimeoutError, ValueError) as exc:
            last_exc = exc
            if attempt + 1 < max_retries:
                delay = (0.4 * (2 ** attempt)) + random.uniform(0.05, 0.25)
                time.sleep(delay)
    raise RuntimeError(f"BBO fetch failed for {url}: {last_exc}")


class MultiVenueBBOCollector:
    """Fetches synchronized real top-of-book BBO ticks across 4 perp venues."""

    def __init__(self, symbols: list[str] | None = None) -> None:
        self.symbols = [s.upper() for s in (symbols or DEFAULT_BBO_SYMBOLS)]
        self._lighter_markets: dict[str, int] | None = None

    def _ensure_lighter_markets(self) -> dict[str, int]:
        if self._lighter_markets is None:
            data, _ = _http_json("https://mainnet.zklighter.elliot.ai/api/v1/orderBooks")
            mapping: dict[str, int] = {}
            for item in data.get("order_books", []):
                if item.get("status") == "active" and item.get("market_type") == "perp":
                    mapping[str(item.get("symbol", "")).upper()] = int(item["market_id"])
            self._lighter_markets = mapping
        return self._lighter_markets

    def fetch_okx_swap_bbo(self) -> list[BBOTick]:
        data, lat_ms = _http_json("https://www.okx.com/api/v5/market/tickers?instType=SWAP")
        want = {f"{s}-USDT-SWAP": s for s in self.symbols}
        ticks: list[BBOTick] = []
        now = utc_now_ms()
        for row in data.get("data", []):
            inst = row.get("instId", "")
            if inst not in want:
                continue
            try:
                bid = float(row["bidPx"])
                ask = float(row["askPx"])
                bid_q = float(row.get("bidSz", 0.0) or 0.0)
                ask_q = float(row.get("askSz", 0.0) or 0.0)
                ts_ms = int(row.get("ts") or now)
            except (TypeError, ValueError, KeyError):
                continue
            if bid > 0 and ask >= bid:
                ticks.append(BBOTick("okx", want[inst], ts_ms, bid, ask, bid_q, ask_q, lat_ms, PROVENANCE_REAL))
        return ticks

    def fetch_gate_futures_bbo(self) -> list[BBOTick]:
        data, lat_ms = _http_json("https://api.gateio.ws/api/v4/futures/usdt/tickers")
        want = {f"{s}_USDT": s for s in self.symbols}
        ticks: list[BBOTick] = []
        ts_ms = utc_now_ms()
        for row in data:
            contract = row.get("contract", "")
            if contract not in want:
                continue
            try:
                bid = float(row["highest_bid"])
                ask = float(row["lowest_ask"])
                bid_q = float(row.get("highest_size", 0.0) or 0.0)
                ask_q = float(row.get("lowest_size", 0.0) or 0.0)
            except (TypeError, ValueError, KeyError):
                continue
            if bid > 0 and ask >= bid:
                ticks.append(BBOTick("gate", want[contract], ts_ms, bid, ask, bid_q, ask_q, lat_ms, PROVENANCE_REAL))
        return ticks

    def fetch_hyperliquid_bbo(self) -> list[BBOTick]:
        ticks: list[BBOTick] = []
        for sym in self.symbols:
            try:
                data, lat_ms = _http_json("https://api.hyperliquid.xyz/info", payload={"type": "l2Book", "coin": sym})
                levels = data.get("levels") or []
                if len(levels) < 2 or not levels[0] or not levels[1]:
                    continue
                bid = float(levels[0][0]["px"])
                bid_q = float(levels[0][0]["sz"])
                ask = float(levels[1][0]["px"])
                ask_q = float(levels[1][0]["sz"])
                ts_ms = int(data.get("time") or utc_now_ms())
                if bid > 0 and ask >= bid:
                    ticks.append(BBOTick("hyperliquid", sym, ts_ms, bid, ask, bid_q, ask_q, lat_ms, PROVENANCE_REAL))
            except (RuntimeError, KeyError, ValueError, TypeError) as exc:
                logger.warning("HL BBO skip %s: %s", sym, exc)
        return ticks

    def fetch_lighter_bbo(self) -> list[BBOTick]:
        markets = self._ensure_lighter_markets()
        ticks: list[BBOTick] = []
        for sym in self.symbols:
            mid_id = markets.get(sym)
            if mid_id is None:
                continue
            try:
                url = f"https://mainnet.zklighter.elliot.ai/api/v1/orderBookOrders?market_id={mid_id}&limit=3"
                data, lat_ms = _http_json(url)
                asks = data.get("asks") or []
                bids = data.get("bids") or []
                if not asks or not bids:
                    continue
                # Pick strictly best bid (max) and best ask (min) across returned top levels
                best_bid_row = max(bids, key=lambda x: float(x["price"]))
                best_ask_row = min(asks, key=lambda x: float(x["price"]))
                bid = float(best_bid_row["price"])
                ask = float(best_ask_row["price"])
                bid_q = float(best_bid_row.get("remaining_base_amount", 0.0) or 0.0)
                ask_q = float(best_ask_row.get("remaining_base_amount", 0.0) or 0.0)
                ts_ms = utc_now_ms()
                if bid > 0 and ask >= bid:
                    ticks.append(BBOTick("lighter", sym, ts_ms, bid, ask, bid_q, ask_q, lat_ms, PROVENANCE_REAL))
            except (RuntimeError, KeyError, ValueError, TypeError) as exc:
                logger.warning("Lighter BBO skip %s: %s", sym, exc)
        return ticks

    def capture_round(self) -> tuple[list[BBOTick], dict[str, str]]:
        """Capture one synchronized cross-venue snapshot round across all 4 venues."""
        all_ticks: list[BBOTick] = []
        errors: dict[str, str] = {}
        for name, fn in (
            ("okx", self.fetch_okx_swap_bbo),
            ("gate", self.fetch_gate_futures_bbo),
            ("hyperliquid", self.fetch_hyperliquid_bbo),
            ("lighter", self.fetch_lighter_bbo),
        ):
            try:
                all_ticks.extend(fn())
            except RuntimeError as exc:
                errors[name] = str(exc)
                logger.warning("BBO venue %s round error: %s", name, exc)
        return all_ticks, errors
