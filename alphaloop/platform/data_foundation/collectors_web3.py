"""100% Real Public Web3 Data Collectors for Strategy R&D Platform v4 (STD-32 / STD-34 Compliant).

Verified live-probed public sources (zero API keys, zero synthetic fallback):
1. Hyperliquid (`/info`): native 4h & 12h `candleSnapshot` (filters unclosed bars & pre-launch v=0 backfill)
   + paginated `fundingHistory` (with v3 1h seed reuse, +48ms jitter snapping, and free `premium_basis_bps`)
2. Binance Vision S3 (`data.binance.vision`) & Spot Tail API (`data-api.binance.vision`):
   - UM monthly/daily `klines` (4h/12h), `fundingRate` (supporting 8h->4h switches & exact carry_interval_h),
     `premiumIndexKlines`, and daily 5m `metrics` (sorted from shuffled Spark CSVs + causal 4h downsampling)
   - Spot monthly `klines` (normalizing 2025-01+ 16-digit microsecond switchover) + REST current-month tail
3. Gate.io Futures v4 (strict non-Chrome UA, `sum` USD volume, 179d funding cap)
4. OKX Public API v5 (`4H` / `12Hutc` UTC-aligned bars, `confirm == "1"`, `realizedRate` funding)
5. Deribit Public API v2 (`BTC_DVOL` daily with backward `continuation` pagination)
6. DefiLlama Stablecoins (`STABLECOIN_MCAP_USD` summing `totalCirculatingUSD.values()` only)
"""

from __future__ import annotations

import http.client
import io
import json
import logging
import math
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from alphaloop.core.provenance import PROVENANCE_REAL, sha256_obj, utc_now_ms
from alphaloop.data.collectors import USER_AGENT as ALPHALOOP_USER_AGENT
from alphaloop.platform.assets import (
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_CRYPTO_SPOT,
    ASSET_CLASS_GOLD_PERP,
)
from alphaloop.platform.data_foundation.models import (
    DAY_MS,
    HOUR_MS,
    CanonicalBar,
    Web3DerivativesMetric,
    Web3MarketMacroPoint,
)

logger = logging.getLogger(__name__)

_TF_TO_HOURS: dict[str, float] = {
    "1h": 1.0,
    "4h": 4.0,
    "8h": 8.0,
    "12h": 12.0,
    "1d": 24.0,
}

_OKX_BAR_MAP: dict[str, str] = {
    "1h": "1H",
    "4h": "4H",
    "12h": "12Hutc",  # CRITICAL: OKX "12H" is UTC+8; "12Hutc" is UTC 00:00/12:00!
    "1d": "1Dutc",
}


def snap_to_nearest_hour_ms(ts_ms: int) -> int:
    """Snap minor exchange settlement jitter (e.g. +1..+48ms) to exact UTC hour boundary."""
    return ((int(ts_ms) + (HOUR_MS // 2)) // HOUR_MS) * HOUR_MS


def normalize_binance_ts_ms(raw_ts: int) -> int:
    """Normalize Binance Vision timestamps: 16-digit microseconds (Spot >=2025-01) -> 13-digit ms."""
    val = int(raw_ts)
    return val // 1000 if val >= 10**15 else val


def parse_zip_csv_rows(zip_bytes: bytes) -> list[list[str]]:
    """Extract CSV rows from a Binance Vision ZIP, auto-skipping UM header while keeping headerless Spot row 0."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if not names:
            return []
        csv_name = next((n for n in names if n.endswith(".csv")), names[0])
        text = zf.read(csv_name).decode("utf-8-sig")
    rows: list[list[str]] = []
    for idx, line in enumerate(text.splitlines()):
        s = line.strip()
        if not s:
            continue
        if idx == 0 and not s[0].isdigit():
            continue
        rows.append(s.split(","))
    return rows


def iter_year_months(start_ym: str, end_ym: str) -> list[tuple[int, int]]:
    """Generate inclusive list of (year, month) tuples from 'YYYY-MM' to 'YYYY-MM'."""
    sy, sm = (int(p) for p in start_ym.split("-"))
    ey, em = (int(p) for p in end_ym.split("-"))
    out: list[tuple[int, int]] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def strip_usdt_quote(symbol: str) -> str:
    s = symbol.upper().strip()
    if s.endswith("USDT") and len(s) > 4:
        return s[:-4]
    if s.endswith("_USDT") and len(s) > 5:
        return s[:-5]
    if s.endswith("-USDT-SWAP") and len(s) > 10:
        return s[:-10]
    return s


def _default_perp_asset_class(canonical_symbol: str) -> str:
    return ASSET_CLASS_GOLD_PERP if canonical_symbol.upper() in ("PAXG", "XAUT", "XAU") else ASSET_CLASS_CRYPTO_PERP


def _slice_fingerprint(
    venue: str,
    symbol: str,
    tf_sec: int,
    endpoint: str,
    row_count: int,
    min_ts: int,
    max_ts: int,
) -> str:
    return sha256_obj(
        {
            "venue": venue,
            "symbol": symbol,
            "tf_sec": tf_sec,
            "endpoint": endpoint,
            "rows": row_count,
            "min_ts": min_ts,
            "max_ts": max_ts,
        }
    )[:16]


class RateLimitedHTTPClient:
    """Thread-safe stdlib HTTP client with per-host pacing, Retry-After, and jittered backoff (STD-34)."""

    def __init__(
        self,
        user_agent: str = ALPHALOOP_USER_AGENT,
        default_timeout: float = 20.0,
    ) -> None:
        self.user_agent = user_agent
        self.default_timeout = float(default_timeout)
        self._host_locks: dict[str, threading.Lock] = {}
        self._host_next_pace: dict[str, float] = {}
        self._meta_lock = threading.Lock()

    def _pace(self, host: str, min_interval_sec: float) -> None:
        if min_interval_sec <= 0.0:
            return
        with self._meta_lock:
            lock = self._host_locks.setdefault(host, threading.Lock())
        with lock:
            now_mono = time.monotonic()
            wait = self._host_next_pace.get(host, 0.0) - now_mono
            if wait > 0.0:
                time.sleep(wait)
            self._host_next_pace[host] = time.monotonic() + min_interval_sec

    def fetch_bytes(
        self,
        url: str,
        *,
        method: str = "GET",
        json_payload: dict[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout_sec: float | None = None,
        max_retries: int = 5,
        min_interval_sec: float = 0.0,
        allow_404: bool = False,
    ) -> bytes | None:
        host = urllib.parse.urlsplit(url).netloc
        data = json.dumps(json_payload).encode("utf-8") if json_payload is not None else None
        headers: dict[str, str] = {"User-Agent": self.user_agent}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if extra_headers:
            headers.update(extra_headers)
        timeout = float(timeout_sec) if timeout_sec is not None else self.default_timeout
        last_err: Exception | None = None

        for attempt in range(max_retries):
            self._pace(host, min_interval_sec)
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and allow_404:
                    return None
                last_err = exc
                if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    base = float(retry_after) if (retry_after and retry_after.strip().isdigit()) else (1.5 * (2**attempt))
                    sleep_s = min(30.0, base) + random.uniform(0.15, 0.65)
                    logger.warning(
                        "HTTP %s on %s (attempt %d/%d), backoff %.2fs",
                        exc.code,
                        url,
                        attempt + 1,
                        max_retries,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                logger.error("HTTP %s permanent failure on %s", exc.code, url)
                raise
            except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
                last_err = exc
                if attempt < max_retries - 1:
                    sleep_s = min(20.0, 0.8 * (2**attempt)) + random.uniform(0.1, 0.4)
                    logger.warning(
                        "Network error on %s: %s (attempt %d/%d), backoff %.2fs",
                        url,
                        exc,
                        attempt + 1,
                        max_retries,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                logger.error("Network failure on %s after %d attempts: %s", url, max_retries, exc)
                raise
        raise last_err if last_err is not None else RuntimeError(f"Exhausted retries for {url}")

    def fetch_json(
        self,
        url: str,
        *,
        method: str = "GET",
        json_payload: dict[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout_sec: float | None = None,
        max_retries: int = 5,
        min_interval_sec: float = 0.0,
        allow_404: bool = False,
    ) -> Any:
        raw = self.fetch_bytes(
            url,
            method=method,
            json_payload=json_payload,
            extra_headers=extra_headers,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            min_interval_sec=min_interval_sec,
            allow_404=allow_404,
        )
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))


class HyperliquidWeb3Collector:
    """Native 4h/12h Hyperliquid candleSnapshot + multi-year paginated fundingHistory collector."""

    INFO_URL = "https://api.hyperliquid.xyz/info"

    def __init__(self, client: RateLimitedHTTPClient | None = None) -> None:
        self.client = client or RateLimitedHTTPClient()

    def fetch_raw_candles(
        self,
        symbol: str,
        interval: str = "12h",
        *,
        days: int = 1095,
        start_ms: int | None = None,
        end_ms: int | None = None,
        now_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        if interval not in _TF_TO_HOURS:
            raise ValueError(f"Unsupported Hyperliquid interval: {interval!r}")
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        tf_ms = int(round(_TF_TO_HOURS[interval] * HOUR_MS))
        end_t = int(end_ms) if end_ms is not None else ref_now
        start_t = int(start_ms) if start_ms is not None else max(0, end_t - int(days) * DAY_MS)

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol.upper(),
                "interval": interval,
                "startTime": start_t,
                "endTime": end_t,
            },
        }
        rows = self.client.fetch_json(
            self.INFO_URL,
            method="POST",
            json_payload=payload,
            max_retries=6,
            min_interval_sec=0.15,
        )
        if not isinstance(rows, list):
            return []
        return self.filter_raw_candles(rows, interval=interval, now_ms=ref_now)

    @staticmethod
    def filter_raw_candles(
        rows: Sequence[Mapping[str, Any]],
        *,
        interval: str,
        now_ms: int,
    ) -> list[dict[str, Any]]:
        """Filter out unclosed in-progress bars and pre-launch zero-volume index backfill bars."""
        tf_ms = int(round(_TF_TO_HOURS[interval] * HOUR_MS))
        by_ts: dict[int, dict[str, Any]] = {}
        for r in rows:
            try:
                t_open = int(r["t"])
                t_close = int(r.get("T", t_open + tf_ms - 1))
                o = float(r["o"])
                h = float(r["h"])
                l = float(r["l"])
                c = float(r["c"])
                v = float(r.get("v", 0.0))
                n_trades = int(r.get("n", 0))
            except (KeyError, TypeError, ValueError):
                continue
            # Trap #1: exclude unclosed in-progress current candle
            if (t_close + 1) > now_ms or (t_open + tf_ms) > now_ms:
                continue
            # Trap #3: exclude pre-launch zero-volume index backfill (e.g. BTC 2022, PAXG 2025-03-27..04-06)
            if v <= 0.0 and n_trades == 0:
                continue
            if not (math.isfinite(o) and math.isfinite(h) and math.isfinite(l) and math.isfinite(c) and math.isfinite(v)):
                continue
            if o <= 0.0 or h <= 0.0 or l <= 0.0 or c <= 0.0:
                continue
            by_ts[t_open] = {
                "t": t_open,
                "T": t_close,
                "o": o,
                "h": max(h, o, c),
                "l": min(l, o, c),
                "c": c,
                "v": max(0.0, v),
                "n": n_trades,
            }
        return [by_ts[ts] for ts in sorted(by_ts)]

    def _paginate_funding_segment(
        self,
        symbol: str,
        seg_start_ms: int,
        seg_end_ms: int,
        *,
        pace_sec: float = 0.35,
        max_retries: int = 6,
    ) -> list[dict[str, Any]]:
        if seg_start_ms > seg_end_ms:
            return []
        collected: list[dict[str, Any]] = []
        cursor = int(seg_start_ms)
        sym_up = symbol.upper()
        while cursor <= seg_end_ms:
            payload = {
                "type": "fundingHistory",
                "coin": sym_up,
                "startTime": cursor,
                "endTime": int(seg_end_ms),
            }
            page = self.client.fetch_json(
                self.INFO_URL,
                method="POST",
                json_payload=payload,
                max_retries=max_retries,
                min_interval_sec=pace_sec,
            )
            if not isinstance(page, list) or not page:
                break
            collected.extend(page)
            max_raw_t = max(int(r["time"]) for r in page if "time" in r)
            if len(page) < 500 or max_raw_t + HOUR_MS > seg_end_ms:
                break
            next_cursor = max_raw_t + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
        return collected

    @staticmethod
    def parse_funding_rows(
        rows: Sequence[Mapping[str, Any]],
        *,
        seed_hourly_map: Mapping[int, tuple[float, ...]] | None = None,
    ) -> dict[int, tuple[float, float, float]]:
        """Parse Hyperliquid fundingHistory rows into {ts_hour_ms: (rate, interval_h, premium_dec)}."""
        # Start from optional seed map (e.g. migrated v3 1h bars)
        merged_raw: dict[int, tuple[float, float, float | None]] = {}
        if seed_hourly_map:
            for ts_k, vals in seed_hourly_map.items():
                ts_snap = snap_to_nearest_hour_ms(int(ts_k))
                rate = float(vals[0])
                iv_h = float(vals[1]) if len(vals) >= 2 and float(vals[1]) > 0.0 else 1.0
                prem = float(vals[2]) if len(vals) >= 3 else 0.0
                if math.isfinite(rate) and math.isfinite(iv_h) and math.isfinite(prem):
                    merged_raw[ts_snap] = (rate, iv_h, prem)

        # Parse fresh API rows (API rows overwrite seed rows because API also includes `premium`)
        api_points: list[tuple[int, float, float]] = []
        for r in rows:
            try:
                ts_snap = snap_to_nearest_hour_ms(int(r["time"]))
                rate = float(r["fundingRate"])
                prem = float(r.get("premium", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(rate) and math.isfinite(prem)):
                continue
            api_points.append((ts_snap, rate, prem))

        api_points.sort(key=lambda x: x[0])
        dedup_api: dict[int, tuple[float, float]] = {}
        for ts_snap, rate, prem in api_points:
            dedup_api[ts_snap] = (rate, prem)
        sorted_ts = sorted(dedup_api)

        for idx, ts_snap in enumerate(sorted_ts):
            rate, prem = dedup_api[ts_snap]
            # Handle HL's May-Jun 2023 8h -> 1h transition accurately from adjacent spacing
            iv_h = 1.0
            if idx > 0:
                delta_h = round((ts_snap - sorted_ts[idx - 1]) / float(HOUR_MS))
                if delta_h == 8:
                    iv_h = 8.0
            elif len(sorted_ts) > 1:
                delta_h = round((sorted_ts[1] - ts_snap) / float(HOUR_MS))
                if delta_h == 8:
                    iv_h = 8.0
            merged_raw[ts_snap] = (rate, iv_h, prem)

        out: dict[int, tuple[float, float, float]] = {}
        for ts_snap in sorted(merged_raw):
            rate, iv_h, prem = merged_raw[ts_snap]
            out[ts_snap] = (rate, iv_h, float(prem or 0.0))
        return out

    def fetch_funding_history(
        self,
        symbol: str,
        *,
        start_ms: int,
        end_ms: int,
        seed_hourly_map: Mapping[int, tuple[float, ...]] | None = None,
        pace_sec: float = 0.35,
        max_retries: int = 6,
    ) -> dict[int, tuple[float, float, float]]:
        """Paginate Hyperliquid fundingHistory, optionally skipping ranges already covered by seed_hourly_map."""
        raw_rows: list[dict[str, Any]] = []
        if seed_hourly_map:
            seed_ts = sorted(snap_to_nearest_hour_ms(int(k)) for k in seed_hourly_map)
            seed_min, seed_max = seed_ts[0], seed_ts[-1]
            # Fetch pre-seed history
            if start_ms < seed_min:
                raw_rows.extend(
                    self._paginate_funding_segment(
                        symbol,
                        start_ms,
                        min(end_ms, seed_min - 1),
                        pace_sec=pace_sec,
                        max_retries=max_retries,
                    )
                )
            # Fetch post-seed tail
            if seed_max + HOUR_MS <= end_ms:
                raw_rows.extend(
                    self._paginate_funding_segment(
                        symbol,
                        seed_max + HOUR_MS,
                        end_ms,
                        pace_sec=pace_sec,
                        max_retries=max_retries,
                    )
                )
        else:
            raw_rows.extend(
                self._paginate_funding_segment(
                    symbol,
                    start_ms,
                    end_ms,
                    pace_sec=pace_sec,
                    max_retries=max_retries,
                )
            )
        return self.parse_funding_rows(raw_rows, seed_hourly_map=seed_hourly_map)

    @staticmethod
    def build_canonical_bars(
        symbol: str,
        interval: str,
        raw_candles: Sequence[Mapping[str, Any]],
        funding_map: Mapping[int, tuple[float, float, float]],
        *,
        asset_class: str | None = None,
        trim_to_funding_span: bool = True,
    ) -> list[CanonicalBar]:
        """Join filtered HL candles with snapped hourly funding + premium on [t_open, t_open + tf_ms)."""
        if not raw_candles:
            return []
        sym_up = symbol.upper()
        ac = asset_class or _default_perp_asset_class(sym_up)
        tf_h = _TF_TO_HOURS[interval]
        tf_sec = int(round(tf_h * 3600.0))
        tf_ms = int(round(tf_h * HOUR_MS))

        sorted_f_ts = sorted(funding_map.keys())
        if trim_to_funding_span and not sorted_f_ts:
            return []
        min_f_ts = sorted_f_ts[0] if sorted_f_ts else None
        max_f_ts = sorted_f_ts[-1] if sorted_f_ts else None
        tail_step_ms = (
            int(round(max(1.0, float(funding_map[max_f_ts][1])) * HOUR_MS))
            if max_f_ts is not None
            else HOUR_MS
        )

        # Pre-bucket funding by candle open timestamp for O(N) alignment
        f_by_bucket: dict[int, list[tuple[float, float, float]]] = {}
        for ts_f in sorted_f_ts:
            triple = funding_map[ts_f]
            b_open = (int(ts_f) // tf_ms) * tf_ms
            f_by_bucket.setdefault(b_open, []).append(triple)

        selected: list[Mapping[str, Any]] = []
        for c in raw_candles:
            t_open = int(c["t"])
            if trim_to_funding_span and min_f_ts is not None and max_f_ts is not None:
                if t_open < min_f_ts or (t_open + tf_ms - tail_step_ms) > max_f_ts:
                    continue
            selected.append(c)

        if not selected:
            return []
        fp = _slice_fingerprint(
            "hyperliquid",
            sym_up,
            tf_sec,
            f"hl:candleSnapshot:{interval}",
            len(selected),
            int(selected[0]["t"]),
            int(selected[-1]["t"]),
        )
        out: list[CanonicalBar] = []
        for c in selected:
            t_open = int(c["t"])
            matched = f_by_bucket.get(t_open, [])
            if matched:
                carry_rate = sum(x[0] for x in matched)
                carry_iv = sum(x[1] for x in matched)
                prem_bps = (sum(x[2] for x in matched) / float(len(matched))) * 10_000.0
            else:
                carry_rate = 0.0
                carry_iv = tf_h
                prem_bps = 0.0
            close_px = float(c["c"])
            out.append(
                CanonicalBar(
                    asset_class=ac,
                    venue="hyperliquid",
                    symbol=sym_up,
                    timeframe_h=tf_h,
                    ts_event_ms=t_open,
                    ts_avail_ms=t_open + tf_ms,
                    open=float(c["o"]),
                    high=float(c["h"]),
                    low=float(c["l"]),
                    close=close_px,
                    volume_usd=max(0.0, float(c["v"])) * close_px,
                    carry_rate_period=carry_rate,
                    carry_interval_h=max(1.0, carry_iv),
                    spread_bps=0.0,
                    provenance=PROVENANCE_REAL,
                    source_fingerprint=fp,
                    timeframe_sec=tf_sec,
                    session_tag="ALL_24_7",
                    premium_basis_bps=round(prem_bps, 4),
                    contract_multiplier=1.0,
                )
            )
        return out

    def fetch_bars(
        self,
        symbol: str,
        interval: str = "12h",
        *,
        days: int = 1095,
        asset_class: str | None = None,
        seed_hourly_map: Mapping[int, tuple[float, ...]] | None = None,
        funding_map: Mapping[int, tuple[float, float, float]] | None = None,
        trim_to_funding_span: bool = True,
        pace_sec: float = 0.35,
        now_ms: int | None = None,
    ) -> tuple[list[CanonicalBar], dict[int, tuple[float, float, float]]]:
        """Fetch real HL bars + funding (accepts pre-fetched funding_map so 12h & 4h share one funding pass)."""
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        raw_candles = self.fetch_raw_candles(symbol, interval=interval, days=days, now_ms=ref_now)
        if not raw_candles:
            return [], dict(funding_map or {})
        if funding_map is None:
            f_start = int(raw_candles[0]["t"])
            f_end = int(raw_candles[-1]["t"]) + int(round(_TF_TO_HOURS[interval] * HOUR_MS))
            fmap = self.fetch_funding_history(
                symbol,
                start_ms=f_start,
                end_ms=f_end,
                seed_hourly_map=seed_hourly_map,
                pace_sec=pace_sec,
            )
        else:
            fmap = dict(funding_map)
        bars = self.build_canonical_bars(
            symbol,
            interval,
            raw_candles,
            fmap,
            asset_class=asset_class,
            trim_to_funding_span=trim_to_funding_span,
        )
        return bars, fmap


class BinanceVisionWeb3Collector:
    """Binance Vision S3 archive + `data-api.binance.vision` collector (UM klines/funding/basis/metrics + Spot)."""

    S3_BASE = "https://data.binance.vision/data"
    SPOT_DATA_API = "https://data-api.binance.vision/api/v3/klines"

    def __init__(self, client: RateLimitedHTTPClient | None = None) -> None:
        self.client = client or RateLimitedHTTPClient()

    def _fetch_zips_parallel(self, urls: Sequence[str], *, max_workers: int = 8) -> dict[str, bytes]:
        if not urls:
            return {}
        out: dict[str, bytes] = {}
        workers = max(1, min(int(max_workers), len(urls)))

        def _get(u: str) -> tuple[str, bytes | None]:
            return u, self.client.fetch_bytes(u, allow_404=True, max_retries=4, timeout_sec=25.0)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_get, u) for u in urls]
            for fut in as_completed(futs):
                u, blob = fut.result()
                if blob is not None:
                    out[u] = blob
        return out

    @staticmethod
    def parse_kline_rows(
        rows: Sequence[Sequence[str]],
        *,
        tf_ms: int,
        now_ms: int,
    ) -> dict[int, tuple[float, float, float, float, float]]:
        """Parse Binance Vision kline CSV rows (handles both 13-digit ms & 16-digit µs Spot timestamps)."""
        out: dict[int, tuple[float, float, float, float, float]] = {}
        for r in rows:
            if len(r) < 8:
                continue
            try:
                t_open = normalize_binance_ts_ms(int(r[0]))
                o = float(r[1])
                h = float(r[2])
                l = float(r[3])
                c = float(r[4])
                quote_vol = float(r[7])
            except (TypeError, ValueError):
                continue
            if (t_open + tf_ms) > now_ms:
                continue
            if not (math.isfinite(o) and math.isfinite(h) and math.isfinite(l) and math.isfinite(c) and math.isfinite(quote_vol)):
                continue
            if o <= 0.0 or h <= 0.0 or l <= 0.0 or c <= 0.0:
                continue
            out[t_open] = (o, max(h, o, c), min(l, o, c), c, max(0.0, quote_vol))
        return out

    @staticmethod
    def parse_um_funding_rows(rows: Sequence[Sequence[str]]) -> dict[int, tuple[float, float]]:
        """Parse UM fundingRate CSV rows into {ts_hour_ms: (rate_decimal, funding_interval_hours)}."""
        out: dict[int, tuple[float, float]] = {}
        for r in rows:
            if len(r) < 3:
                continue
            try:
                ts_snap = snap_to_nearest_hour_ms(normalize_binance_ts_ms(int(r[0])))
                iv_h = float(r[1]) if r[1] else 8.0
                rate = float(r[2])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(iv_h) and iv_h > 0.0 and math.isfinite(rate)):
                continue
            out[ts_snap] = (rate, iv_h)
        return out

    @staticmethod
    def parse_premium_index_rows(rows: Sequence[Sequence[str]]) -> dict[int, float]:
        """Parse UM premiumIndexKlines CSV rows into {ts_open_ms: premium_basis_bps}."""
        out: dict[int, float] = {}
        for r in rows:
            if len(r) < 5:
                continue
            try:
                t_open = normalize_binance_ts_ms(int(r[0]))
                prem_close = float(r[4])
            except (TypeError, ValueError):
                continue
            if math.isfinite(prem_close):
                out[t_open] = prem_close * 10_000.0
        return out

    @staticmethod
    def parse_um_metrics_rows(
        rows: Sequence[Sequence[str]],
        *,
        venue: str = "binance_um",
        symbol: str,
        downsample_tf_h: float = 4.0,
        now_ms: int | None = None,
    ) -> list[Web3DerivativesMetric]:
        """Parse & sort unsorted Binance daily metrics CSV rows, then causally downsample 5m -> downsample_tf_h."""
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        canonical_sym = strip_usdt_quote(symbol)
        raw_pts: list[tuple[int, float, float, float, float]] = []
        for r in rows:
            if len(r) < 8:
                continue
            try:
                dt = datetime.strptime(r[0].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                ts_ms = int(dt.timestamp() * 1000)
                oi_usd = float(r[3])
                top_lsr = float(r[5]) if r[5] else float(r[4])
                acc_lsr = float(r[6])
                taker_ratio = float(r[7])
            except (TypeError, ValueError):
                continue
            if not (
                math.isfinite(oi_usd)
                and math.isfinite(top_lsr)
                and math.isfinite(acc_lsr)
                and math.isfinite(taker_ratio)
            ):
                continue
            if oi_usd < 0.0 or top_lsr < 0.0 or acc_lsr < 0.0 or taker_ratio < 0.0:
                continue
            raw_pts.append((ts_ms, oi_usd, top_lsr, acc_lsr, taker_ratio))

        if not raw_pts:
            return []
        # CRITICAL: Binance Vision daily metrics CSVs arrive in shuffled Spark partition order!
        raw_pts.sort(key=lambda x: x[0])

        tf_ms = int(round(float(downsample_tf_h) * HOUR_MS))
        if tf_ms <= 300_000:
            fp = _slice_fingerprint(venue, canonical_sym, 300, "binance_vision:metrics:5m", len(raw_pts), raw_pts[0][0], raw_pts[-1][0])
            return [
                Web3DerivativesMetric(
                    venue=venue,
                    symbol=canonical_sym,
                    ts_event_ms=ts_ms,
                    ts_avail_ms=ts_ms + 300_000,
                    open_interest_usd=oi_usd,
                    top_trader_lsr=top_lsr,
                    account_lsr=acc_lsr,
                    taker_ls_vol_ratio=taker_r,
                    provenance=PROVENANCE_REAL,
                    source_fingerprint=fp,
                )
                for ts_ms, oi_usd, top_lsr, acc_lsr, taker_r in raw_pts
                if (ts_ms + 300_000) <= ref_now
            ]

        buckets: dict[int, list[tuple[int, float, float, float, float]]] = {}
        for pt in raw_pts:
            b_open = (pt[0] // tf_ms) * tf_ms
            buckets.setdefault(b_open, []).append(pt)

        tf_sec = int(round(float(downsample_tf_h) * 3600.0))
        sorted_b = [b for b in sorted(buckets) if (b + tf_ms) <= ref_now]
        if not sorted_b:
            return []
        fp = _slice_fingerprint(
            venue,
            canonical_sym,
            tf_sec,
            f"binance_vision:metrics:{downsample_tf_h:.1f}h",
            len(sorted_b),
            sorted_b[0],
            sorted_b[-1],
        )
        out: list[Web3DerivativesMetric] = []
        for b_open in sorted_b:
            grp = buckets[b_open]
            # Causal within bucket: last point-in-time stock variables (OI, LSRs), mean flow variable (taker vol ratio)
            last = grp[-1]
            taker_avg = sum(x[4] for x in grp) / float(len(grp))
            out.append(
                Web3DerivativesMetric(
                    venue=venue,
                    symbol=canonical_sym,
                    ts_event_ms=b_open,
                    ts_avail_ms=b_open + tf_ms,
                    open_interest_usd=last[1],
                    top_trader_lsr=last[2],
                    account_lsr=last[3],
                    taker_ls_vol_ratio=round(taker_avg, 6),
                    provenance=PROVENANCE_REAL,
                    source_fingerprint=fp,
                )
            )
        return out

    def fetch_um_bars(
        self,
        symbol: str,
        interval: str = "4h",
        *,
        start_ym: str = "2023-01",
        end_ym: str = "2026-10",
        asset_class: str | None = None,
        trim_to_funding_span: bool = True,
        max_workers: int = 8,
        now_ms: int | None = None,
    ) -> list[CanonicalBar]:
        """Fetch Binance Vision UM monthly klines + fundingRate + premiumIndexKlines."""
        if interval not in _TF_TO_HOURS:
            raise ValueError(f"Unsupported Binance UM interval: {interval!r}")
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        exch_sym = symbol.upper()
        canonical_sym = strip_usdt_quote(exch_sym)
        ac = asset_class or _default_perp_asset_class(canonical_sym)
        tf_h = _TF_TO_HOURS[interval]
        tf_sec = int(round(tf_h * 3600.0))
        tf_ms = int(round(tf_h * HOUR_MS))

        yms = iter_year_months(start_ym, end_ym)
        kline_urls = [
            f"{self.S3_BASE}/futures/um/monthly/klines/{exch_sym}/{interval}/{exch_sym}-{interval}-{y:04d}-{m:02d}.zip"
            for y, m in yms
        ]
        funding_urls = [
            f"{self.S3_BASE}/futures/um/monthly/fundingRate/{exch_sym}/{exch_sym}-fundingRate-{y:04d}-{m:02d}.zip"
            for y, m in yms
        ]
        prem_urls = [
            f"{self.S3_BASE}/futures/um/monthly/premiumIndexKlines/{exch_sym}/{interval}/{exch_sym}-{interval}-{y:04d}-{m:02d}.zip"
            for y, m in yms
        ]

        blobs = self._fetch_zips_parallel(kline_urls + funding_urls + prem_urls, max_workers=max_workers)
        kline_rows: list[list[str]] = []
        for u in kline_urls:
            if u in blobs:
                kline_rows.extend(parse_zip_csv_rows(blobs[u]))
        funding_rows: list[list[str]] = []
        for u in funding_urls:
            if u in blobs:
                funding_rows.extend(parse_zip_csv_rows(blobs[u]))
        prem_rows: list[list[str]] = []
        for u in prem_urls:
            if u in blobs:
                prem_rows.extend(parse_zip_csv_rows(blobs[u]))

        klines = self.parse_kline_rows(kline_rows, tf_ms=tf_ms, now_ms=ref_now)
        fmap = self.parse_um_funding_rows(funding_rows)
        pmap = self.parse_premium_index_rows(prem_rows)
        return self.assemble_um_bars(
            canonical_sym=canonical_sym,
            interval=interval,
            klines=klines,
            funding_map=fmap,
            premium_map=pmap,
            asset_class=ac,
            trim_to_funding_span=trim_to_funding_span,
        )

    @staticmethod
    def assemble_um_bars(
        *,
        canonical_sym: str,
        interval: str,
        klines: Mapping[int, tuple[float, float, float, float, float]],
        funding_map: Mapping[int, tuple[float, float]],
        premium_map: Mapping[int, float],
        asset_class: str | None = None,
        trim_to_funding_span: bool = True,
    ) -> list[CanonicalBar]:
        if not klines:
            return []
        ac = asset_class or _default_perp_asset_class(canonical_sym)
        tf_h = _TF_TO_HOURS[interval]
        tf_sec = int(round(tf_h * 3600.0))
        tf_ms = int(round(tf_h * HOUR_MS))

        sorted_f = sorted(funding_map.keys())
        if trim_to_funding_span and not sorted_f:
            return []
        min_f = sorted_f[0] if sorted_f else None
        max_f = sorted_f[-1] if sorted_f else None
        tail_iv = float(funding_map[max_f][1]) if max_f is not None else 8.0

        f_by_bucket: dict[int, list[tuple[float, float]]] = {}
        for ts_f in sorted_f:
            pair = funding_map[ts_f]
            b_open = (int(ts_f) // tf_ms) * tf_ms
            f_by_bucket.setdefault(b_open, []).append(pair)

        selected_ts: list[int] = []
        upper_f = max(max_f, max_f + int(round(tail_iv * HOUR_MS)) - tf_ms) if max_f is not None else None
        for t_open in sorted(klines):
            if trim_to_funding_span and min_f is not None and upper_f is not None:
                if t_open < min_f or t_open > upper_f:
                    continue
            selected_ts.append(t_open)
        if not selected_ts:
            return []

        fp = _slice_fingerprint(
            "binance_um",
            canonical_sym,
            tf_sec,
            f"binance_vision:um:{interval}",
            len(selected_ts),
            selected_ts[0],
            selected_ts[-1],
        )
        # Strictly causal funding-interval state (zero peeking at future 8h -> 4h transitions)
        if sorted_f:
            current_iv = float(funding_map[sorted_f[0]][1])
            for ts_f in sorted_f:
                if ts_f <= selected_ts[0]:
                    current_iv = float(funding_map[ts_f][1])
                else:
                    break
        else:
            current_iv = 8.0
        out: list[CanonicalBar] = []
        for t_open in selected_ts:
            o, h, l, c, qvol = klines[t_open]
            matched = f_by_bucket.get(t_open, [])
            if matched:
                carry_rate = sum(x[0] for x in matched)
                carry_iv = sum(x[1] for x in matched)
                current_iv = float(matched[-1][1])
            else:
                carry_rate = 0.0
                carry_iv = current_iv
            prem_bps = premium_map.get(t_open, 0.0)
            out.append(
                CanonicalBar(
                    asset_class=ac,
                    venue="binance_um",
                    symbol=canonical_sym,
                    timeframe_h=tf_h,
                    ts_event_ms=t_open,
                    ts_avail_ms=t_open + tf_ms,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume_usd=qvol,
                    carry_rate_period=carry_rate,
                    carry_interval_h=max(1.0, carry_iv),
                    spread_bps=0.0,
                    provenance=PROVENANCE_REAL,
                    source_fingerprint=fp,
                    timeframe_sec=tf_sec,
                    session_tag="ALL_24_7",
                    premium_basis_bps=round(prem_bps, 4),
                    contract_multiplier=1.0,
                )
            )
        return out

    def fetch_spot_rest_tail(
        self,
        exch_symbol: str,
        interval: str,
        *,
        start_ms: int,
        now_ms: int,
    ) -> dict[int, tuple[float, float, float, float, float]]:
        """Fetch recent Spot klines from `data-api.binance.vision/api/v3/klines` (filters unclosed bar)."""
        tf_ms = int(round(_TF_TO_HOURS[interval] * HOUR_MS))
        out: dict[int, tuple[float, float, float, float, float]] = {}
        cursor = int(start_ms)
        while cursor + tf_ms <= now_ms:
            q = urllib.parse.urlencode(
                {
                    "symbol": exch_symbol.upper(),
                    "interval": interval,
                    "startTime": cursor,
                    "limit": 1000,
                }
            )
            url = f"{self.SPOT_DATA_API}?{q}"
            rows = self.client.fetch_json(url, max_retries=5, min_interval_sec=0.1)
            if not isinstance(rows, list) or not rows:
                break
            advanced = False
            for r in rows:
                if not isinstance(r, list) or len(r) < 8:
                    continue
                t_open = normalize_binance_ts_ms(int(r[0]))
                t_close = normalize_binance_ts_ms(int(r[6]))
                if (t_close + 1) > now_ms or (t_open + tf_ms) > now_ms:
                    continue
                o, h, l, c, qvol = float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[7])
                if not (math.isfinite(o) and math.isfinite(h) and math.isfinite(l) and math.isfinite(c) and math.isfinite(qvol)):
                    continue
                if o <= 0.0 or h <= 0.0 or l <= 0.0 or c <= 0.0:
                    continue
                out[t_open] = (o, max(h, o, c), min(l, o, c), c, max(0.0, qvol))
                if t_open + tf_ms > cursor:
                    cursor = t_open + tf_ms
                    advanced = True
            if len(rows) < 1000 or not advanced:
                break
        return out

    def fetch_spot_bars(
        self,
        symbol: str,
        interval: str = "4h",
        *,
        start_ym: str = "2023-01",
        end_ym: str = "2026-10",
        include_rest_tail: bool = True,
        asset_class: str = ASSET_CLASS_CRYPTO_SPOT,
        max_workers: int = 8,
        now_ms: int | None = None,
    ) -> list[CanonicalBar]:
        """Fetch Binance Vision Spot monthly klines (13/16-digit normalized) + `data-api` current-month tail."""
        if interval not in _TF_TO_HOURS:
            raise ValueError(f"Unsupported Binance Spot interval: {interval!r}")
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        exch_sym = symbol.upper()
        canonical_sym = strip_usdt_quote(exch_sym)
        tf_h = _TF_TO_HOURS[interval]
        tf_sec = int(round(tf_h * 3600.0))
        tf_ms = int(round(tf_h * HOUR_MS))

        yms = iter_year_months(start_ym, end_ym)
        urls = [
            f"{self.S3_BASE}/spot/monthly/klines/{exch_sym}/{interval}/{exch_sym}-{interval}-{y:04d}-{m:02d}.zip"
            for y, m in yms
        ]
        blobs = self._fetch_zips_parallel(urls, max_workers=max_workers)
        csv_rows: list[list[str]] = []
        for u in urls:
            if u in blobs:
                csv_rows.extend(parse_zip_csv_rows(blobs[u]))

        klines = self.parse_kline_rows(csv_rows, tf_ms=tf_ms, now_ms=ref_now)
        if include_rest_tail:
            if klines:
                tail_start = max(klines) + tf_ms
            else:
                sy, sm = (int(p) for p in end_ym.split("-"))
                tail_start = int(datetime(sy, sm, 1, tzinfo=timezone.utc).timestamp() * 1000)
            tail = self.fetch_spot_rest_tail(exch_sym, interval, start_ms=tail_start, now_ms=ref_now)
            klines.update(tail)

        if not klines:
            return []
        sorted_ts = sorted(klines)
        fp = _slice_fingerprint(
            "binance_spot",
            canonical_sym,
            tf_sec,
            f"binance_vision:spot:{interval}",
            len(sorted_ts),
            sorted_ts[0],
            sorted_ts[-1],
        )
        return [
            CanonicalBar(
                asset_class=asset_class,
                venue="binance_spot",
                symbol=canonical_sym,
                timeframe_h=tf_h,
                ts_event_ms=t_open,
                ts_avail_ms=t_open + tf_ms,
                open=klines[t_open][0],
                high=klines[t_open][1],
                low=klines[t_open][2],
                close=klines[t_open][3],
                volume_usd=klines[t_open][4],
                carry_rate_period=0.0,
                carry_interval_h=tf_h,
                spread_bps=0.0,
                provenance=PROVENANCE_REAL,
                source_fingerprint=fp,
                timeframe_sec=tf_sec,
                session_tag="ALL_24_7",
                premium_basis_bps=0.0,
                contract_multiplier=1.0,
            )
            for t_open in sorted_ts
        ]

    def fetch_um_metrics(
        self,
        symbol: str,
        *,
        days: int = 365,
        end_date: date | None = None,
        downsample_tf_h: float = 4.0,
        max_workers: int = 8,
        now_ms: int | None = None,
    ) -> list[Web3DerivativesMetric]:
        """Fetch Binance Vision UM daily 5m metrics ZIPs, sort shuffled rows, and downsample causally."""
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        exch_sym = symbol.upper()
        last_day = end_date or (datetime.fromtimestamp(ref_now / 1000.0, tz=timezone.utc).date() - timedelta(days=1))
        first_day = last_day - timedelta(days=max(1, int(days)) - 1)
        urls: list[str] = []
        d = first_day
        while d <= last_day:
            ds = d.strftime("%Y-%m-%d")
            urls.append(f"{self.S3_BASE}/futures/um/daily/metrics/{exch_sym}/{exch_sym}-metrics-{ds}.zip")
            d += timedelta(days=1)

        blobs = self._fetch_zips_parallel(urls, max_workers=max_workers)
        all_rows: list[list[str]] = []
        for u in urls:
            if u in blobs:
                all_rows.extend(parse_zip_csv_rows(blobs[u]))
        return self.parse_um_metrics_rows(
            all_rows,
            venue="binance_um",
            symbol=exch_sym,
            downsample_tf_h=downsample_tf_h,
            now_ms=ref_now,
        )


class GateWeb3Collector:
    """Gate.io Futures v4 public collector (strict non-Chrome UA, `sum` USD volume, 179d funding cap)."""

    BASE_URL = "https://api.gateio.ws/api/v4/futures/usdt"
    MAX_FUNDING_LOOKBACK_DAYS = 179

    def __init__(self, client: RateLimitedHTTPClient | None = None) -> None:
        # Gate.io Cloudflare WAF returns HTTP 403 on Chrome UAs; enforce non-Chrome UA
        self.client = client or RateLimitedHTTPClient(user_agent=ALPHALOOP_USER_AGENT)

    def fetch_bars(
        self,
        symbol: str,
        interval: str = "12h",
        *,
        days: int = 179,
        asset_class: str | None = None,
        trim_to_funding_span: bool = True,
        now_ms: int | None = None,
    ) -> list[CanonicalBar]:
        if interval not in _TF_TO_HOURS:
            raise ValueError(f"Unsupported Gate interval: {interval!r}")
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        canonical_sym = strip_usdt_quote(symbol)
        contract = f"{canonical_sym}_USDT"
        ac = asset_class or _default_perp_asset_class(canonical_sym)
        tf_h = _TF_TO_HOURS[interval]
        tf_ms = int(round(tf_h * HOUR_MS))
        tf_sec = int(round(tf_h * 3600.0))

        end_sec = ref_now // 1000
        start_sec = max(0, end_sec - int(days) * 86_400)
        chunk_sec = 175 * 86_400
        raw_candles: list[dict[str, Any]] = []
        cur = start_sec
        while cur < end_sec:
            nxt = min(end_sec, cur + chunk_sec)
            q = urllib.parse.urlencode(
                {"contract": contract, "interval": interval, "from": cur, "to": nxt}
            )
            page = self.client.fetch_json(
                f"{self.BASE_URL}/candlesticks?{q}",
                max_retries=5,
                min_interval_sec=0.12,
            )
            if isinstance(page, list):
                raw_candles.extend(page)
            cur = nxt + 1

        fund_days = min(int(days), self.MAX_FUNDING_LOOKBACK_DAYS)
        f_from = max(0, end_sec - fund_days * 86_400)
        fq = urllib.parse.urlencode(
            {"contract": contract, "limit": 1000, "from": f_from, "to": end_sec}
        )
        raw_funding = self.client.fetch_json(
            f"{self.BASE_URL}/funding_rate?{fq}",
            max_retries=5,
            min_interval_sec=0.12,
        )
        return self.parse_candles_and_funding(
            canonical_sym=canonical_sym,
            interval=interval,
            raw_candles=raw_candles,
            raw_funding=raw_funding if isinstance(raw_funding, list) else [],
            asset_class=ac,
            trim_to_funding_span=trim_to_funding_span,
            now_ms=ref_now,
        )

    @staticmethod
    def parse_candles_and_funding(
        *,
        canonical_sym: str,
        interval: str,
        raw_candles: Sequence[Mapping[str, Any]],
        raw_funding: Sequence[Mapping[str, Any]],
        asset_class: str | None = None,
        trim_to_funding_span: bool = True,
        now_ms: int,
    ) -> list[CanonicalBar]:
        ac = asset_class or _default_perp_asset_class(canonical_sym)
        tf_h = _TF_TO_HOURS[interval]
        tf_ms = int(round(tf_h * HOUR_MS))
        tf_sec = int(round(tf_h * 3600.0))

        # Parse funding (Gate returns descending order)
        f_points: list[tuple[int, float]] = []
        for r in raw_funding:
            try:
                ts_ms = snap_to_nearest_hour_ms(int(r["t"]) * 1000)
                rate = float(r["r"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(rate):
                f_points.append((ts_ms, rate))
        f_points.sort(key=lambda x: x[0])
        f_map: dict[int, tuple[float, float]] = {}
        for idx, (ts_ms, rate) in enumerate(f_points):
            iv_h = 8.0
            if idx > 0:
                dh = round((ts_ms - f_points[idx - 1][0]) / float(HOUR_MS))
                if dh in (1, 4, 8):
                    iv_h = float(dh)
            f_map[ts_ms] = (rate, iv_h)

        sorted_f = sorted(f_map)
        if trim_to_funding_span and not sorted_f:
            return []
        f_by_bucket: dict[int, list[tuple[float, float]]] = {}
        for ts_f in sorted_f:
            f_by_bucket.setdefault((ts_f // tf_ms) * tf_ms, []).append(f_map[ts_f])
        min_f = sorted_f[0] if sorted_f else None
        max_f = sorted_f[-1] if sorted_f else None

        by_ts: dict[int, tuple[float, float, float, float, float]] = {}
        for r in raw_candles:
            try:
                t_open = int(r["t"]) * 1000
                o, h, l, c = float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"])
                vol_usd = float(r.get("sum") or 0.0)  # Gate `sum` is quote USD volume, NOT `v` contracts!
            except (KeyError, TypeError, ValueError):
                continue
            if (t_open + tf_ms) > now_ms:
                continue
            if not (math.isfinite(o) and math.isfinite(h) and math.isfinite(l) and math.isfinite(c) and math.isfinite(vol_usd)):
                continue
            if o <= 0.0 or h <= 0.0 or l <= 0.0 or c <= 0.0:
                continue
            by_ts[t_open] = (o, max(h, o, c), min(l, o, c), c, max(0.0, vol_usd))

        upper_f = max(max_f, max_f + 8 * HOUR_MS - tf_ms) if max_f is not None else None
        sel_ts = [
            t
            for t in sorted(by_ts)
            if (not trim_to_funding_span or min_f is None or upper_f is None or (min_f <= t <= upper_f))
        ]
        if not sel_ts:
            return []
        fp = _slice_fingerprint("gate", canonical_sym, tf_sec, f"gate:v4:{interval}", len(sel_ts), sel_ts[0], sel_ts[-1])
        current_iv = float(f_map[sorted_f[0]][1]) if sorted_f else 8.0
        out: list[CanonicalBar] = []
        for t_open in sel_ts:
            o, h, l, c, vol_usd = by_ts[t_open]
            matched = f_by_bucket.get(t_open, [])
            if matched:
                carry_rate = sum(x[0] for x in matched)
                carry_iv = sum(x[1] for x in matched)
                current_iv = float(matched[-1][1])
            else:
                carry_rate = 0.0
                carry_iv = current_iv
            out.append(
                CanonicalBar(
                    asset_class=ac,
                    venue="gate",
                    symbol=canonical_sym,
                    timeframe_h=tf_h,
                    ts_event_ms=t_open,
                    ts_avail_ms=t_open + tf_ms,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume_usd=vol_usd,
                    carry_rate_period=carry_rate,
                    carry_interval_h=max(1.0, carry_iv),
                    spread_bps=0.0,
                    provenance=PROVENANCE_REAL,
                    source_fingerprint=fp,
                    timeframe_sec=tf_sec,
                    session_tag="ALL_24_7",
                    premium_basis_bps=0.0,
                    contract_multiplier=1.0,
                )
            )
        return out


class OKXWeb3Collector:
    """OKX Public API v5 collector (`4H` / `12Hutc` UTC alignment, `confirm == "1"`, `realizedRate`)."""

    BASE_URL = "https://www.okx.com/api/v5"

    def __init__(self, client: RateLimitedHTTPClient | None = None) -> None:
        self.client = client or RateLimitedHTTPClient()

    @staticmethod
    def resolve_okx_bar(interval: str) -> str:
        if interval not in _OKX_BAR_MAP:
            raise ValueError(f"Unsupported OKX interval: {interval!r}")
        return _OKX_BAR_MAP[interval]

    def fetch_bars(
        self,
        symbol: str,
        interval: str = "12h",
        *,
        days: int = 365,
        asset_class: str | None = None,
        trim_to_funding_span: bool = False,
        now_ms: int | None = None,
    ) -> list[CanonicalBar]:
        bar_param = self.resolve_okx_bar(interval)
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        canonical_sym = strip_usdt_quote(symbol)
        inst_id = f"{canonical_sym}-USDT-SWAP"
        ac = asset_class or _default_perp_asset_class(canonical_sym)
        cutoff_ms = max(0, ref_now - int(days) * DAY_MS)

        raw_candles: list[list[str]] = []
        after: str | None = None
        for _ in range(50):
            params: dict[str, Any] = {"instId": inst_id, "bar": bar_param, "limit": 100}
            if after is not None:
                params["after"] = after
            resp = self.client.fetch_json(
                f"{self.BASE_URL}/market/history-candles?{urllib.parse.urlencode(params)}",
                max_retries=5,
                min_interval_sec=0.08,
            )
            data = resp.get("data") if isinstance(resp, dict) and str(resp.get("code")) == "0" else []
            if not data:
                break
            raw_candles.extend(data)
            oldest = min(int(r[0]) for r in data)
            if oldest <= cutoff_ms or len(data) < 100:
                break
            after = str(oldest)

        raw_funding: list[dict[str, Any]] = []
        f_after: str | None = None
        for _ in range(10):
            fparams: dict[str, Any] = {"instId": inst_id, "limit": 100}
            if f_after is not None:
                fparams["after"] = f_after
            fresp = self.client.fetch_json(
                f"{self.BASE_URL}/public/funding-rate-history?{urllib.parse.urlencode(fparams)}",
                max_retries=5,
                min_interval_sec=0.08,
            )
            fdata = fresp.get("data") if isinstance(fresp, dict) and str(fresp.get("code")) == "0" else []
            if not fdata:
                break
            raw_funding.extend(fdata)
            oldest_f = min(int(r["fundingTime"]) for r in fdata if "fundingTime" in r)
            if oldest_f <= cutoff_ms or len(fdata) < 100:
                break
            f_after = str(oldest_f)

        return self.parse_candles_and_funding(
            canonical_sym=canonical_sym,
            interval=interval,
            raw_candles=raw_candles,
            raw_funding=raw_funding,
            asset_class=ac,
            cutoff_ms=cutoff_ms,
            trim_to_funding_span=trim_to_funding_span,
            now_ms=ref_now,
        )

    @staticmethod
    def parse_candles_and_funding(
        *,
        canonical_sym: str,
        interval: str,
        raw_candles: Sequence[Sequence[str]],
        raw_funding: Sequence[Mapping[str, Any]],
        asset_class: str | None = None,
        cutoff_ms: int = 0,
        trim_to_funding_span: bool = False,
        now_ms: int,
    ) -> list[CanonicalBar]:
        ac = asset_class or _default_perp_asset_class(canonical_sym)
        tf_h = _TF_TO_HOURS[interval]
        tf_ms = int(round(tf_h * HOUR_MS))
        tf_sec = int(round(tf_h * 3600.0))

        f_points: list[tuple[int, float]] = []
        for r in raw_funding:
            try:
                ts_ms = snap_to_nearest_hour_ms(int(r["fundingTime"]))
                rate_str = r.get("realizedRate")
                if rate_str in (None, ""):
                    rate_str = r.get("fundingRate")
                rate = float(rate_str)  # type: ignore[arg-type]
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(rate):
                f_points.append((ts_ms, rate))
        f_points.sort(key=lambda x: x[0])
        f_map: dict[int, tuple[float, float]] = {}
        for idx, (ts_ms, rate) in enumerate(f_points):
            iv_h = 8.0
            if idx > 0:
                dh = round((ts_ms - f_points[idx - 1][0]) / float(HOUR_MS))
                if dh in (1, 4, 8):
                    iv_h = float(dh)
            f_map[ts_ms] = (rate, iv_h)

        sorted_f = sorted(f_map)
        if trim_to_funding_span and not sorted_f:
            return []
        f_by_bucket: dict[int, list[tuple[float, float]]] = {}
        for ts_f in sorted_f:
            f_by_bucket.setdefault((ts_f // tf_ms) * tf_ms, []).append(f_map[ts_f])
        min_f = sorted_f[0] if sorted_f else None
        max_f = sorted_f[-1] if sorted_f else None

        by_ts: dict[int, tuple[float, float, float, float, float]] = {}
        for r in raw_candles:
            if len(r) < 9:
                continue
            # CRITICAL: OKX returns confirm="0" for unclosed candle even on history-candles!
            if str(r[8]) != "1":
                continue
            try:
                t_open = int(r[0])
                o, h, l, c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
                vol_usd = float(r[7])
            except (TypeError, ValueError):
                continue
            if t_open < cutoff_ms or (t_open + tf_ms) > now_ms:
                continue
            if not (math.isfinite(o) and math.isfinite(h) and math.isfinite(l) and math.isfinite(c) and math.isfinite(vol_usd)):
                continue
            if o <= 0.0 or h <= 0.0 or l <= 0.0 or c <= 0.0:
                continue
            by_ts[t_open] = (o, max(h, o, c), min(l, o, c), c, max(0.0, vol_usd))

        upper_f = max(max_f, max_f + 8 * HOUR_MS - tf_ms) if max_f is not None else None
        sel_ts = [
            t
            for t in sorted(by_ts)
            if (not trim_to_funding_span or min_f is None or upper_f is None or (min_f <= t <= upper_f))
        ]
        if not sel_ts:
            return []
        fp = _slice_fingerprint("okx", canonical_sym, tf_sec, f"okx:v5:{interval}", len(sel_ts), sel_ts[0], sel_ts[-1])
        current_iv = float(f_map[sorted_f[0]][1]) if sorted_f else 8.0
        out: list[CanonicalBar] = []
        for t_open in sel_ts:
            o, h, l, c, vol_usd = by_ts[t_open]
            matched = f_by_bucket.get(t_open, [])
            if matched:
                carry_rate = sum(x[0] for x in matched)
                carry_iv = sum(x[1] for x in matched)
                current_iv = float(matched[-1][1])
            else:
                carry_rate = 0.0
                carry_iv = current_iv
            out.append(
                CanonicalBar(
                    asset_class=ac,
                    venue="okx",
                    symbol=canonical_sym,
                    timeframe_h=tf_h,
                    ts_event_ms=t_open,
                    ts_avail_ms=t_open + tf_ms,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume_usd=vol_usd,
                    carry_rate_period=carry_rate,
                    carry_interval_h=max(1.0, carry_iv),
                    spread_bps=0.0,
                    provenance=PROVENANCE_REAL,
                    source_fingerprint=fp,
                    timeframe_sec=tf_sec,
                    session_tag="ALL_24_7",
                    premium_basis_bps=0.0,
                    contract_multiplier=1.0,
                )
            )
        return out


class DeribitDVOLCollector:
    """Deribit Public API v2 BTC DVOL collector with backward `continuation` pagination."""

    ENDPOINT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"

    def __init__(self, client: RateLimitedHTTPClient | None = None) -> None:
        self.client = client or RateLimitedHTTPClient()

    def fetch_dvol(
        self,
        *,
        currency: str = "BTC",
        resolution_sec: int = 86_400,
        start_ms: int = 1_609_459_200_000,  # 2021-01-01T00:00:00Z
        end_ms: int | None = None,
        now_ms: int | None = None,
    ) -> list[Web3MarketMacroPoint]:
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        cur_end = int(end_ms) if end_ms is not None else ref_now
        raw_rows: list[list[Any]] = []

        for _ in range(30):
            q = urllib.parse.urlencode(
                {
                    "currency": currency.upper(),
                    "resolution": int(resolution_sec),
                    "start_timestamp": int(start_ms),
                    "end_timestamp": int(cur_end),
                }
            )
            resp = self.client.fetch_json(f"{self.ENDPOINT}?{q}", max_retries=5, min_interval_sec=0.1)
            result = resp.get("result") if isinstance(resp, dict) else None
            if not isinstance(result, dict):
                break
            data = result.get("data") or []
            if not data:
                break
            raw_rows.extend(data)
            cont = result.get("continuation")
            if cont is None:
                break
            cont_int = int(cont)
            if cont_int <= start_ms or cont_int >= cur_end:
                break
            cur_end = cont_int

        return self.parse_dvol_rows(
            raw_rows,
            currency=currency.upper(),
            resolution_sec=int(resolution_sec),
            now_ms=ref_now,
        )

    @staticmethod
    def parse_dvol_rows(
        rows: Sequence[Sequence[Any]],
        *,
        currency: str = "BTC",
        resolution_sec: int = 86_400,
        now_ms: int,
    ) -> list[Web3MarketMacroPoint]:
        res_ms = int(resolution_sec) * 1000
        by_ts: dict[int, float] = {}
        for r in rows:
            if len(r) < 5:
                continue
            try:
                ts_ev = int(r[0])
                close_dvol = float(r[4])
            except (TypeError, ValueError):
                continue
            if (ts_ev + res_ms) > now_ms:
                continue
            if math.isfinite(close_dvol) and close_dvol > 0.0:
                by_ts[ts_ev] = close_dvol
        if not by_ts:
            return []
        sorted_ts = sorted(by_ts)
        series_id = f"{currency}_DVOL"
        fp = _slice_fingerprint("deribit", series_id, resolution_sec, "deribit:dvol", len(sorted_ts), sorted_ts[0], sorted_ts[-1])
        return [
            Web3MarketMacroPoint(
                source="deribit",
                series_id=series_id,
                ts_event_ms=ts_ev,
                ts_avail_ms=ts_ev + res_ms,
                value=by_ts[ts_ev],
                provenance=PROVENANCE_REAL,
                source_fingerprint=fp,
            )
            for ts_ev in sorted_ts
        ]


class DefiLlamaStablecoinCollector:
    """DefiLlama global stablecoin supply collector (`sum(totalCirculatingUSD.values())` ONLY)."""

    ENDPOINT = "https://stablecoins.llama.fi/stablecoincharts/all"

    def __init__(self, client: RateLimitedHTTPClient | None = None) -> None:
        self.client = client or RateLimitedHTTPClient()

    def fetch_stablecoin_mcap(self, *, now_ms: int | None = None) -> list[Web3MarketMacroPoint]:
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        payload = self.client.fetch_json(self.ENDPOINT, max_retries=5, timeout_sec=25.0)
        if not isinstance(payload, list):
            return []
        return self.parse_stablecoin_rows(payload, now_ms=ref_now)

    @staticmethod
    def parse_stablecoin_rows(
        rows: Sequence[Mapping[str, Any]],
        *,
        now_ms: int,
    ) -> list[Web3MarketMacroPoint]:
        by_ts: dict[int, float] = {}
        for r in rows:
            usd_map = r.get("totalCirculatingUSD")
            if not isinstance(usd_map, Mapping):
                continue
            try:
                ts_ev = int(r["date"]) * 1000
            except (KeyError, TypeError, ValueError):
                continue
            if (ts_ev + DAY_MS) > now_ms:
                continue
            total_usd = 0.0
            for val in usd_map.values():
                if val is None or isinstance(val, bool):
                    continue
                try:
                    vf = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(vf) and vf > 0.0:
                    total_usd += vf
            if total_usd > 0.0:
                by_ts[ts_ev] = total_usd
        if not by_ts:
            return []
        sorted_ts = sorted(by_ts)
        fp = _slice_fingerprint(
            "defillama",
            "STABLECOIN_MCAP_USD",
            86_400,
            "defillama:stablecoincharts:all",
            len(sorted_ts),
            sorted_ts[0],
            sorted_ts[-1],
        )
        return [
            Web3MarketMacroPoint(
                source="defillama",
                series_id="STABLECOIN_MCAP_USD",
                ts_event_ms=ts_ev,
                ts_avail_ms=ts_ev + DAY_MS,
                value=round(by_ts[ts_ev], 2),
                provenance=PROVENANCE_REAL,
                source_fingerprint=fp,
            )
            for ts_ev in sorted_ts
        ]
