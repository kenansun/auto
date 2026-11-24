"""Live 4-Exchange Public BBO Orderbook Collector (Hyperliquid / Lighter / OKX / Gate).

Collects synchronized top-of-book bid/ask/qty snapshots with real HTTP latency tracking
and crossed-book/NaN validation (STD-34 Pillars 2 & 6). Zero API keys required;
zero synthetic fallback on upstream failure.
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from alphaloop.core.models import BBOTick
from alphaloop.core.provenance import PROVENANCE_REAL, utc_now_ms

logger = logging.getLogger(__name__)

DEFAULT_BBO_SYMBOLS = [
    "BTC", "ETH", "SOL", "VVV", "WLD", "NEAR", "ENA", "ZEC", "UNI", "HYPE",
]

TS_SOURCE_VENUE = "VENUE_EXCHANGE"
TS_SOURCE_RECV_LOCAL = "RECV_LOCAL_ONLY"


@dataclass
class TimestampedBBOTick(BBOTick):
    """BBOTick subclass explicitly separating exchange venue timestamp from local receive timestamp."""

    venue_ts_ms: int | None = None
    recv_ts_ms: int = 0
    ts_source: str = TS_SOURCE_RECV_LOCAL


def _parse_finite_float(raw: Any) -> float | None:
    """Parse a finite float, rejecting None, bool, non-numeric values, NaN, and +/-Inf."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def _parse_optional_nonneg_qty(raw: Any) -> float | None:
    """Parse an optional non-negative finite quantity; missing/blank maps to 0.0, invalid returns None."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return 0.0
    val = _parse_finite_float(raw)
    if val is None or val < 0.0:
        return None
    return val


def _parse_venue_ts_ms(raw: Any) -> int | None:
    """Parse a positive integer exchange timestamp in ms; absent/invalid returns None (never local clock)."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            ts_int = int(text)
        except ValueError:
            fv = _parse_finite_float(text)
            if fv is None or fv <= 0.0:
                return None
            return int(fv)
        return ts_int if ts_int > 0 else None
    fv = _parse_finite_float(raw)
    if fv is None or fv <= 0.0:
        return None
    return int(fv)


def _is_valid_bbo_quote(
    bid: float | None,
    ask: float | None,
    bid_qty: float | None,
    ask_qty: float | None,
    lat_ms: float | None,
) -> bool:
    """Validate finite positive non-crossed BBO prices, non-negative quantities/latency, and overflow safety."""
    if (
        bid is None
        or ask is None
        or bid_qty is None
        or ask_qty is None
        or lat_ms is None
    ):
        return False
    if not (
        math.isfinite(bid)
        and math.isfinite(ask)
        and math.isfinite(bid_qty)
        and math.isfinite(ask_qty)
        and math.isfinite(lat_ms)
    ):
        return False
    if not (bid > 0.0 and ask > 0.0 and ask >= bid and bid_qty >= 0.0 and ask_qty >= 0.0 and lat_ms >= 0.0):
        return False
    mid = (bid + ask) * 0.5
    spread = ask - bid
    return math.isfinite(mid) and mid > 0.0 and math.isfinite(spread)


def _http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
    max_retries: int = 3,
) -> tuple[Any, float]:
    """Perform HTTP GET/POST with exponential backoff + jitter and return (json, latency_ms)."""
    data = json.dumps(payload, allow_nan=False).encode("utf-8") if payload is not None else None
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
            if isinstance(data, dict):
                for item in data.get("order_books", []):
                    if isinstance(item, dict) and item.get("status") == "active" and item.get("market_type") == "perp":
                        mapping[str(item.get("symbol", "")).upper()] = int(item["market_id"])
            self._lighter_markets = mapping
        return self._lighter_markets

    def fetch_okx_swap_bbo(self) -> list[BBOTick]:
        data, lat_ms = _http_json("https://www.okx.com/api/v5/market/tickers?instType=SWAP")
        recv_ts_ms = utc_now_ms()
        lat_val = _parse_finite_float(lat_ms)
        want = {f"{s}-USDT-SWAP": s for s in self.symbols}
        ticks: list[BBOTick] = []
        if not isinstance(data, dict):
            return ticks
        for row in data.get("data", []):
            if not isinstance(row, dict):
                continue
            inst = row.get("instId", "")
            if inst not in want:
                continue
            bid = _parse_finite_float(row.get("bidPx"))
            ask = _parse_finite_float(row.get("askPx"))
            bid_q = _parse_optional_nonneg_qty(row.get("bidSz"))
            ask_q = _parse_optional_nonneg_qty(row.get("askSz"))
            if not _is_valid_bbo_quote(bid, ask, bid_q, ask_q, lat_val):
                continue
            venue_ts_ms = _parse_venue_ts_ms(row.get("ts"))
            ts_source = TS_SOURCE_VENUE if venue_ts_ms is not None else TS_SOURCE_RECV_LOCAL
            ts_ms = venue_ts_ms if venue_ts_ms is not None else recv_ts_ms
            ticks.append(
                TimestampedBBOTick(
                    "okx",
                    want[inst],
                    ts_ms,
                    float(bid),
                    float(ask),
                    float(bid_q),
                    float(ask_q),
                    float(lat_val),
                    PROVENANCE_REAL,
                    venue_ts_ms=venue_ts_ms,
                    recv_ts_ms=recv_ts_ms,
                    ts_source=ts_source,
                )
            )
        return ticks

    def fetch_gate_futures_bbo(self) -> list[BBOTick]:
        data, lat_ms = _http_json("https://api.gateio.ws/api/v4/futures/usdt/tickers")
        recv_ts_ms = utc_now_ms()
        lat_val = _parse_finite_float(lat_ms)
        want = {f"{s}_USDT": s for s in self.symbols}
        ticks: list[BBOTick] = []
        if not isinstance(data, list):
            return ticks
        for row in data:
            if not isinstance(row, dict):
                continue
            contract = row.get("contract", "")
            if contract not in want:
                continue
            bid = _parse_finite_float(row.get("highest_bid"))
            ask = _parse_finite_float(row.get("lowest_ask"))
            bid_q = _parse_optional_nonneg_qty(row.get("highest_size"))
            ask_q = _parse_optional_nonneg_qty(row.get("lowest_size"))
            if not _is_valid_bbo_quote(bid, ask, bid_q, ask_q, lat_val):
                continue
            raw_vts = row.get("ts_ms") if "ts_ms" in row else row.get("time_ms")
            venue_ts_ms = _parse_venue_ts_ms(raw_vts)
            ts_source = TS_SOURCE_VENUE if venue_ts_ms is not None else TS_SOURCE_RECV_LOCAL
            ts_ms = venue_ts_ms if venue_ts_ms is not None else recv_ts_ms
            ticks.append(
                TimestampedBBOTick(
                    "gate",
                    want[contract],
                    ts_ms,
                    float(bid),
                    float(ask),
                    float(bid_q),
                    float(ask_q),
                    float(lat_val),
                    PROVENANCE_REAL,
                    venue_ts_ms=venue_ts_ms,
                    recv_ts_ms=recv_ts_ms,
                    ts_source=ts_source,
                )
            )
        return ticks

    def fetch_hyperliquid_bbo(self) -> list[BBOTick]:
        ticks: list[BBOTick] = []
        for sym in self.symbols:
            try:
                data, lat_ms = _http_json("https://api.hyperliquid.xyz/info", payload={"type": "l2Book", "coin": sym})
                recv_ts_ms = utc_now_ms()
                lat_val = _parse_finite_float(lat_ms)
                if not isinstance(data, dict):
                    continue
                levels = data.get("levels") or []
                if not isinstance(levels, list) or len(levels) < 2 or not levels[0] or not levels[1]:
                    continue
                top_bid = levels[0][0]
                top_ask = levels[1][0]
                if not isinstance(top_bid, dict) or not isinstance(top_ask, dict):
                    continue
                bid = _parse_finite_float(top_bid.get("px"))
                bid_q = _parse_finite_float(top_bid.get("sz"))
                ask = _parse_finite_float(top_ask.get("px"))
                ask_q = _parse_finite_float(top_ask.get("sz"))
                if not _is_valid_bbo_quote(bid, ask, bid_q, ask_q, lat_val):
                    continue
                venue_ts_ms = _parse_venue_ts_ms(data.get("time"))
                ts_source = TS_SOURCE_VENUE if venue_ts_ms is not None else TS_SOURCE_RECV_LOCAL
                ts_ms = venue_ts_ms if venue_ts_ms is not None else recv_ts_ms
                ticks.append(
                    TimestampedBBOTick(
                        "hyperliquid",
                        sym,
                        ts_ms,
                        float(bid),
                        float(ask),
                        float(bid_q),
                        float(ask_q),
                        float(lat_val),
                        PROVENANCE_REAL,
                        venue_ts_ms=venue_ts_ms,
                        recv_ts_ms=recv_ts_ms,
                        ts_source=ts_source,
                    )
                )
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
                recv_ts_ms = utc_now_ms()
                lat_val = _parse_finite_float(lat_ms)
                if not isinstance(data, dict):
                    continue
                asks = data.get("asks") or []
                bids = data.get("bids") or []
                if not isinstance(asks, list) or not isinstance(bids, list) or not asks or not bids:
                    continue
                # Validate EVERY returned level before picking max bid / min ask so NaN/Inf cannot bypass max()/min()
                parsed_bids: list[tuple[float, dict[str, Any]]] = []
                for lvl in bids:
                    if not isinstance(lvl, dict) or "price" not in lvl:
                        parsed_bids = []
                        break
                    px = _parse_finite_float(lvl.get("price"))
                    if px is None or px <= 0.0:
                        parsed_bids = []
                        break
                    parsed_bids.append((px, lvl))
                if not parsed_bids:
                    continue

                parsed_asks: list[tuple[float, dict[str, Any]]] = []
                for lvl in asks:
                    if not isinstance(lvl, dict) or "price" not in lvl:
                        parsed_asks = []
                        break
                    px = _parse_finite_float(lvl.get("price"))
                    if px is None or px <= 0.0:
                        parsed_asks = []
                        break
                    parsed_asks.append((px, lvl))
                if not parsed_asks:
                    continue

                bid, best_bid_row = max(parsed_bids, key=lambda item: item[0])
                ask, best_ask_row = min(parsed_asks, key=lambda item: item[0])
                bid_q = _parse_optional_nonneg_qty(best_bid_row.get("remaining_base_amount"))
                ask_q = _parse_optional_nonneg_qty(best_ask_row.get("remaining_base_amount"))
                if not _is_valid_bbo_quote(bid, ask, bid_q, ask_q, lat_val):
                    continue
                raw_vts = data.get("ts_ms") if "ts_ms" in data else data.get("timestamp_ms")
                venue_ts_ms = _parse_venue_ts_ms(raw_vts)
                ts_source = TS_SOURCE_VENUE if venue_ts_ms is not None else TS_SOURCE_RECV_LOCAL
                ts_ms = venue_ts_ms if venue_ts_ms is not None else recv_ts_ms
                ticks.append(
                    TimestampedBBOTick(
                        "lighter",
                        sym,
                        ts_ms,
                        float(bid),
                        float(ask),
                        float(bid_q),
                        float(ask_q),
                        float(lat_val),
                        PROVENANCE_REAL,
                        venue_ts_ms=venue_ts_ms,
                        recv_ts_ms=recv_ts_ms,
                        ts_source=ts_source,
                    )
                )
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
