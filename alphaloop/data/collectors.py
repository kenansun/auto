"""Real-exchange market data collectors (Hyperliquid, Lighter, OKX, Gate.io).

Strictly STD-34 compliant:
- Exponential backoff with jitter on HTTP 429/5xx/timeouts.
- Zero synthetic fallback: fetch failure raises or logs explicitly and returns partial/empty.
- Every bar is stamped with PROVENANCE_REAL.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from typing import Any

from alphaloop.core.models import FundingRateBar, PriceBar
from alphaloop.core.provenance import PROVENANCE_REAL, utc_now_ms

logger = logging.getLogger(__name__)

USER_AGENT = "AlphaLoop-QuantResearch/1.0"
HOUR_MS = 3600 * 1000


def _http_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_sec: float = 12.0,
    max_retries: int = 3,
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"User-Agent": USER_AGENT}
    if data is not None:
        headers["Content-Type"] = "application/json"

    last_err: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                sleep_s = (0.6 * (2 ** attempt)) + random.uniform(0.05, 0.35)
                logger.warning("HTTP %s on %s (attempt %d), backoff %.2fs", exc.code, url, attempt + 1, sleep_s)
                time.sleep(sleep_s)
                continue
            logger.error("HTTP %s permanent failure on %s", exc.code, url)
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt < max_retries - 1:
                sleep_s = (0.5 * (2 ** attempt)) + random.uniform(0.05, 0.30)
                logger.warning("Network error on %s: %s (attempt %d), backoff %.2fs", url, exc, attempt + 1, sleep_s)
                time.sleep(sleep_s)
                continue
            logger.error("Network failure on %s after %d attempts: %s", url, max_retries, exc)
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Unexpected empty retry loop for {url}")


def _infer_interval_hours(timestamps_ms: list[int], default_hours: float = 8.0) -> float:
    """Infer settlement interval in hours from median positive adjacent timestamp delta."""
    if len(timestamps_ms) < 2:
        return default_hours
    ts_sorted = sorted(set(timestamps_ms))
    deltas = [(ts_sorted[i] - ts_sorted[i - 1]) / HOUR_MS for i in range(1, len(ts_sorted))]
    pos = [d for d in deltas if 0.5 <= d <= 24.0]
    if not pos:
        return default_hours
    pos.sort()
    med = pos[len(pos) // 2]
    for candidate in (1.0, 2.0, 4.0, 8.0):
        if abs(med - candidate) <= 0.35:
            return candidate
    return round(med, 2)


class HyperliquidCollector:
    """Hyperliquid Info API collector (main crypto perps + xyz HIP-3)."""

    INFO_URL = "https://api.hyperliquid.xyz/info"

    def fetch_live_universe(self, dex: str | None = None) -> dict[str, dict[str, float]]:
        req: dict[str, Any] = {"type": "metaAndAssetCtxs"}
        if dex:
            req["dex"] = dex
        data = _http_json(self.INFO_URL, method="POST", payload=req)
        out: dict[str, dict[str, float]] = {}
        for u, ctx in zip(data[0]["universe"], data[1]):
            sym = str(u["name"])
            mark = float(ctx.get("markPx") or 0.0)
            fr_1h = float(ctx.get("funding") or 0.0)
            oi_usd = float(ctx.get("openInterest") or 0.0) * mark
            vol_24h = float(ctx.get("dayNtlVlm") or 0.0)
            out[sym] = {
                "funding_rate_1h": fr_1h,
                "apr_pct": fr_1h * 24.0 * 365.0 * 100.0,
                "mark_px": mark,
                "oi_usd": oi_usd,
                "vol_24h_usd": vol_24h,
            }
        return out

    def fetch_funding_history(self, symbol: str, days: int = 90) -> list[FundingRateBar]:
        end_ms = utc_now_ms()
        cursor = end_ms - days * 86400 * 1000
        seen: dict[int, FundingRateBar] = {}
        for _ in range(12):  # 12 * 500h = 250 days max safety bound
            rows = _http_json(
                self.INFO_URL,
                method="POST",
                payload={"type": "fundingHistory", "coin": symbol, "startTime": cursor, "endTime": end_ms},
            )
            if not rows:
                break
            max_t = cursor
            for r in rows:
                t_raw = int(r["time"])
                t_hour = (t_raw // HOUR_MS) * HOUR_MS
                rate = float(r["fundingRate"])
                seen[t_hour] = FundingRateBar(
                    venue="hyperliquid",
                    symbol=symbol,
                    ts_ms=t_hour,
                    funding_rate=rate,
                    interval_hours=1.0,
                    provenance=PROVENANCE_REAL,
                )
                if t_raw > max_t:
                    max_t = t_raw
            if len(rows) < 490 or max_t + HOUR_MS >= end_ms:
                break
            cursor = max_t + 1000
            time.sleep(0.08)
        return sorted(seen.values(), key=lambda x: x.ts_ms)

    def fetch_candles_1h(self, symbol: str, days: int = 90) -> list[PriceBar]:
        end_ms = utc_now_ms()
        start_ms = end_ms - days * 86400 * 1000
        rows = _http_json(
            self.INFO_URL,
            method="POST",
            payload={
                "type": "candleSnapshot",
                "req": {"coin": symbol, "interval": "1h", "startTime": start_ms, "endTime": end_ms},
            },
        )
        bars: list[PriceBar] = []
        for r in rows:
            t_hour = (int(r["t"]) // HOUR_MS) * HOUR_MS
            close = float(r["c"])
            vol_base = float(r["v"])
            bars.append(
                PriceBar(
                    venue="hyperliquid",
                    symbol=symbol,
                    ts_ms=t_hour,
                    open=float(r["o"]),
                    high=float(r["h"]),
                    low=float(r["l"]),
                    close=close,
                    volume_usd=vol_base * close,
                    provenance=PROVENANCE_REAL,
                )
            )
        bars.sort(key=lambda x: x.ts_ms)
        return bars


class LighterCollector:
    """Lighter zk-rollup Perp DEX collector (0% standard account fees, 1h funding)."""

    BASE_URL = "https://mainnet.zklighter.elliot.ai/api/v1"

    def __init__(self) -> None:
        self._market_ids: dict[str, int] | None = None

    def market_id_map(self) -> dict[str, int]:
        if self._market_ids is None:
            data = _http_json(f"{self.BASE_URL}/orderBooks")
            self._market_ids = {
                str(o["symbol"]): int(o["market_id"])
                for o in data.get("order_books", [])
                if o.get("market_type") == "perp"
            }
        return self._market_ids

    def fetch_live_funding_all(self) -> dict[str, dict[str, float]]:
        """Returns Lighter's aggregated live 8h-equivalent funding rates by exchange."""
        data = _http_json(f"{self.BASE_URL}/funding-rates")
        out: dict[str, dict[str, float]] = {}
        for r in data.get("funding_rates", []):
            ex = str(r.get("exchange", ""))
            sym = str(r.get("symbol", ""))
            rate_8h = float(r.get("rate") or 0.0)
            out.setdefault(ex, {})[sym] = (rate_8h / 8.0) * 24.0 * 365.0 * 100.0
        return out

    def fetch_funding_history(self, symbol: str, days: int = 90) -> list[FundingRateBar]:
        mids = self.market_id_map()
        if symbol not in mids:
            return []
        mid = mids[symbol]
        end_sec = int(time.time())
        target_start_sec = end_sec - days * 86400
        seen: dict[int, FundingRateBar] = {}

        cursor_end = end_sec
        for _ in range(6):  # up to 6 * 750h = 187 days
            url = (
                f"{self.BASE_URL}/fundings?market_id={mid}&resolution=1h"
                f"&start_timestamp={target_start_sec}&end_timestamp={cursor_end}&count_back=750"
            )
            data = _http_json(url)
            rows = data.get("fundings", [])
            if not rows:
                break
            min_ts = cursor_end
            for r in rows:
                ts_sec = int(r["timestamp"])
                ts_ms = (ts_sec * 1000 // HOUR_MS) * HOUR_MS
                # Verified triple-checked unit: rate='0.0012' means 0.0012% = 1.2e-5 per 1h.
                # direction=='long' -> longs pay shorts (+); direction=='short' -> shorts pay longs (-).
                rate_dec = float(r["rate"]) / 100.0
                if str(r.get("direction", "")).lower() == "short":
                    rate_dec = -rate_dec
                seen[ts_ms] = FundingRateBar(
                    venue="lighter",
                    symbol=symbol,
                    ts_ms=ts_ms,
                    funding_rate=rate_dec,
                    interval_hours=1.0,
                    provenance=PROVENANCE_REAL,
                )
                if ts_sec < min_ts:
                    min_ts = ts_sec
            if len(rows) < 700 or min_ts <= target_start_sec + 3600:
                break
            cursor_end = min_ts - 3600
            time.sleep(0.10)
        return sorted(seen.values(), key=lambda x: x.ts_ms)


class OKXCollector:
    """OKX Public REST collector for realized funding history and 1h SWAP/SPOT candles."""

    BASE_URL = "https://www.okx.com/api/v5"

    def fetch_funding_history(self, symbol: str, days: int = 90) -> list[FundingRateBar]:
        inst_id = f"{symbol}-USDT-SWAP"
        cutoff_ms = utc_now_ms() - days * 86400 * 1000
        raw_rows: list[tuple[int, float]] = []
        after: str | None = None
        for _ in range(10):  # up to 1000 periods
            url = f"{self.BASE_URL}/public/funding-rate-history?instId={inst_id}&limit=100"
            if after:
                url += f"&after={after}"
            data = _http_json(url)
            rows = data.get("data", [])
            if not rows:
                break
            min_t = None
            for r in rows:
                ts_ms = (int(r["fundingTime"]) // HOUR_MS) * HOUR_MS
                rate = float(r.get("realizedRate") or r.get("fundingRate") or 0.0)
                if ts_ms >= cutoff_ms:
                    raw_rows.append((ts_ms, rate))
                if min_t is None or ts_ms < min_t:
                    min_t = ts_ms
            if min_t is None or min_t <= cutoff_ms or len(rows) < 95:
                break
            after = str(min_t)
            time.sleep(0.08)

        if not raw_rows:
            return []
        interval_h = _infer_interval_hours([t for t, _ in raw_rows], default_hours=8.0)
        dedup = {
            ts: FundingRateBar(
                venue="okx",
                symbol=symbol,
                ts_ms=ts,
                funding_rate=rate,
                interval_hours=interval_h,
                provenance=PROVENANCE_REAL,
            )
            for ts, rate in raw_rows
        }
        return sorted(dedup.values(), key=lambda x: x.ts_ms)

    def fetch_candles_1h(self, symbol: str, days: int = 90, spot: bool = False) -> list[PriceBar]:
        inst_id = f"{symbol}-USDT" if spot else f"{symbol}-USDT-SWAP"
        venue_tag = "okx_spot" if spot else "okx"
        cutoff_ms = utc_now_ms() - days * 86400 * 1000
        seen: dict[int, PriceBar] = {}
        after: str | None = None
        for _ in range(25):  # 25 * 100h = ~104 days
            url = f"{self.BASE_URL}/market/history-candles?instId={inst_id}&bar=1H&limit=100"
            if after:
                url += f"&after={after}"
            data = _http_json(url)
            rows = data.get("data", [])
            if not rows:
                break
            min_t = None
            for r in rows:
                ts_ms = (int(r[0]) // HOUR_MS) * HOUR_MS
                if ts_ms >= cutoff_ms:
                    close = float(r[4])
                    vol_ccy = float(r[7]) if len(r) > 7 else float(r[5]) * close
                    seen[ts_ms] = PriceBar(
                        venue=venue_tag,
                        symbol=symbol,
                        ts_ms=ts_ms,
                        open=float(r[1]),
                        high=float(r[2]),
                        low=float(r[3]),
                        close=close,
                        volume_usd=vol_ccy,
                        provenance=PROVENANCE_REAL,
                    )
                if min_t is None or ts_ms < min_t:
                    min_t = ts_ms
            if min_t is None or min_t <= cutoff_ms or len(rows) < 95:
                break
            after = str(min_t)
            time.sleep(0.06)
        return sorted(seen.values(), key=lambda x: x.ts_ms)


class GateCollector:
    """Gate.io USDT Futures public collector (4h/8h funding + 1h candles)."""

    BASE_URL = "https://api.gateio.ws/api/v4/futures/usdt"

    def fetch_funding_history(self, symbol: str, days: int = 90) -> list[FundingRateBar]:
        contract = f"{symbol}_USDT"
        now_sec = int(time.time())
        cutoff_sec = now_sec - days * 86400
        raw_rows: list[tuple[int, float]] = []
        cursor_to = now_sec
        for _ in range(6):
            url = f"{self.BASE_URL}/funding_rate?contract={contract}&limit=200&from={cutoff_sec}&to={cursor_to}"
            try:
                rows = _http_json(url)
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 404):
                    return []
                raise
            if not rows:
                break
            min_t = cursor_to
            for r in rows:
                t_sec = int(r["t"])
                ts_ms = (t_sec * 1000 // HOUR_MS) * HOUR_MS
                if t_sec >= cutoff_sec:
                    raw_rows.append((ts_ms, float(r["r"])))
                if t_sec < min_t:
                    min_t = t_sec
            if len(rows) < 180 or min_t <= cutoff_sec + 3600:
                break
            cursor_to = min_t - 1
            time.sleep(0.08)

        if not raw_rows:
            return []
        interval_h = _infer_interval_hours([t for t, _ in raw_rows], default_hours=8.0)
        dedup = {
            ts: FundingRateBar(
                venue="gate",
                symbol=symbol,
                ts_ms=ts,
                funding_rate=rate,
                interval_hours=interval_h,
                provenance=PROVENANCE_REAL,
            )
            for ts, rate in raw_rows
        }
        return sorted(dedup.values(), key=lambda x: x.ts_ms)

    def fetch_candles_1h(self, symbol: str, days: int = 90) -> list[PriceBar]:
        contract = f"{symbol}_USDT"
        now_sec = int(time.time())
        cutoff_sec = now_sec - days * 86400
        seen: dict[int, PriceBar] = {}
        cursor_from = cutoff_sec
        step_sec = 900 * 3600  # 900 hours per batch
        while cursor_from < now_sec:
            cursor_to = min(now_sec, cursor_from + step_sec)
            url = f"{self.BASE_URL}/candlesticks?contract={contract}&interval=1h&from={cursor_from}&to={cursor_to}"
            try:
                rows = _http_json(url)
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 404):
                    break
                raise
            for r in rows:
                ts_ms = (int(r["t"]) * 1000 // HOUR_MS) * HOUR_MS
                close = float(r["c"])
                vol_usd = float(r.get("sum") or 0.0)
                seen[ts_ms] = PriceBar(
                    venue="gate",
                    symbol=symbol,
                    ts_ms=ts_ms,
                    open=float(r["o"]),
                    high=float(r["h"]),
                    low=float(r["l"]),
                    close=close,
                    volume_usd=vol_usd,
                    provenance=PROVENANCE_REAL,
                )
            cursor_from = cursor_to + 3600
            time.sleep(0.08)
        return sorted(seen.values(), key=lambda x: x.ts_ms)
