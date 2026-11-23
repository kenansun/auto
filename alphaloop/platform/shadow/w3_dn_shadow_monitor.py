"""W3-DN Dual-Leg Spot+Perp Delta-Neutral Read-Only Shadow Blind-Run Monitor (>=48h Gate).

Hardened across formal Red-Blue Adversarial Review (`CC-20261123-006`, closing `RB-01..03`,
`RB-06..10`, `RB-16`, `RB-19`):
  - Hard Security Lock: `SHADOW_LIVE_EXECUTION_ENABLED: Final[bool] = False`.
    Zero private keys, zero API keys, zero order signing, zero execution client imports.
  - STD-32 Mandatory Reuse:
    * Reuses `Web3DeltaNeutralCarryKernel` (G7 signal parity + state checkpointing),
      `RateLimitedHTTPClient`, `connect_wal()`, and P8 MD-Gateway BBO feed contract
      (`P8MDGatewayFileQuoteProvider`, O(1) EOF seek, zero `wss://stream.binance.com`).
    * Public REST fallback uses Waterfall (`data-api.binance.vision` -> `api.binance.com`
      failing over on HTTP 451/403/429 or invalid <=0/crossed BBO) + Hyperliquid `info`
      (`metaAndAssetCtxs` + `l2Book`) + Deribit DVOL (with 6h staleness expiry).
  - STD-34 Six Pillars + Full Restart Continuity:
    1. Internal service reuse (STD-32)
    2. Waterfall + Jitter + explicit HTTP timeouts
    3. Structured rotating JSONL logs with `trace_id` & `allow_nan=False`
    4. Tiered `PlatformErrorCode` alerts + cooldown anti-storm + strict Fail-Closed guards
       (crossed book, stale quote, future wall-clock/quote skew, leg desync, unpaired leg,
       NaN/Inf/<=0 qty/price poison)
    5. Monotonic dead-man switch (`time.monotonic()`, active from tick 0) + atomic `0644`
       P16 Watchdog Sentinel heartbeat
    6. SQLite WAL (`connect_wal()`, `busy_timeout=30000`), atomic state+kernel checkpointing
       across restarts, and SHA-256 tamper-evident audit chain
"""

from __future__ import annotations

import csv
import enum
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Final, Protocol, Sequence, runtime_checkable

from alphaloop.core.db import LockedConnection, connect_wal, transaction
from alphaloop.core.provenance import PROVENANCE_REAL, utc_now_ms
from alphaloop.platform.assets import (
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_CRYPTO_SPOT,
    COST_BINANCE_SPOT,
    COST_HYPERLIQUID_PERP,
)
from alphaloop.platform.data_foundation.collectors_web3 import RateLimitedHTTPClient
from alphaloop.platform.data_foundation.models import CanonicalBar
from alphaloop.platform.strategies.web3.delta_neutral_carry import (
    Web3DeltaNeutralCarryKernel,
    perp_leg_symbol,
    spot_leg_symbol,
)

SHADOW_LIVE_EXECUTION_ENABLED: Final[bool] = False
DEFAULT_W3_DN_SHADOW_COINS: Final[tuple[str, ...]] = ("BTC", "ETH", "SOL", "AVAX")

_BINANCE_SPOT_WATERFALL_BASES: Final[tuple[str, ...]] = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
)
_HYPERLIQUID_INFO_URL: Final[str] = "https://api.hyperliquid.xyz/info"
_DERIBIT_DVOL_URL: Final[str] = "https://www.deribit.com/api/v2/public/get_volatility_index_data"

logger = logging.getLogger(__name__)


class PlatformErrorCode(str, enum.Enum):
    """Standardized STD-34 tiered error & risk codes for W3-DN shadow observation."""

    OK = "OK"
    STALE_QUOTE = "ERR_W3DN_STALE_QUOTE"
    CROSSED_BOOK = "ERR_W3DN_CROSSED_BOOK"
    CLOCK_SKEW = "ERR_W3DN_CLOCK_SKEW"
    LEG_DESYNC = "ERR_W3DN_LEG_DESYNC"
    UNPAIRED_LEG = "ERR_W3DN_UNPAIRED_LEG"
    NON_FINITE_DATA = "ERR_W3DN_NON_FINITE_DATA"
    BASIS_BLOWOUT = "ERR_W3DN_BASIS_BLOWOUT"
    MARGIN_STORM = "ERR_W3DN_MARGIN_STORM"
    FUNDING_FLIP = "WARN_W3DN_FUNDING_FLIP"
    FEED_OUTAGE = "ERR_W3DN_FEED_OUTAGE"
    DEAD_MAN_TRIGGERED = "ERR_W3DN_DEAD_MAN_SWITCH"
    DELTA_INVARIANT_BREACH = "ERR_W3DN_DELTA_INVARIANT"
    LIVE_SWITCH_BLOCKED = "ERR_W3DN_LIVE_EXECUTION_FORBIDDEN"


_CODE_SEVERITY: Final[dict[PlatformErrorCode, str]] = {
    PlatformErrorCode.OK: "INFO",
    PlatformErrorCode.FUNDING_FLIP: "WARN",
    PlatformErrorCode.STALE_QUOTE: "HIGH",
    PlatformErrorCode.CROSSED_BOOK: "HIGH",
    PlatformErrorCode.CLOCK_SKEW: "HIGH",
    PlatformErrorCode.LEG_DESYNC: "HIGH",
    PlatformErrorCode.UNPAIRED_LEG: "HIGH",
    PlatformErrorCode.NON_FINITE_DATA: "HIGH",
    PlatformErrorCode.BASIS_BLOWOUT: "CRITICAL",
    PlatformErrorCode.MARGIN_STORM: "CRITICAL",
    PlatformErrorCode.FEED_OUTAGE: "HIGH",
    PlatformErrorCode.DEAD_MAN_TRIGGERED: "CRITICAL",
    PlatformErrorCode.DELTA_INVARIANT_BREACH: "CRITICAL",
    PlatformErrorCode.LIVE_SWITCH_BLOCKED: "CRITICAL",
}


def _is_pos_finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0


def _is_finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _normalize_coin_symbol(raw_sym: str) -> str:
    """Normalize `BTCUSDT`, `BTC-USDT`, `BTC/USDT`, `BTC_USDT` -> `BTC` (RB-09)."""
    cleaned = re.sub(r"[-_/\s]+", "", str(raw_sym).strip().upper())
    for suffix in ("USDT", "USDC", "USD", "PERP"):
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            return cleaned[: -len(suffix)]
    return cleaned


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist JSON (`allow_nan=False`, mode `0644`) via mkstemp + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def estimate_l2_vwap_slippage_bps(
    levels: Sequence[tuple[float, float]],
    *,
    notional_usd: float,
    mid_price: float,
    side: str,
) -> float:
    """Estimate one-way VWAP execution slippage (bps vs mid_price) from L2 orderbook levels."""
    if not levels or not _is_pos_finite(notional_usd) or not _is_pos_finite(mid_price):
        return 50.0
    remaining_usd = float(notional_usd)
    filled_qty = 0.0
    spent_usd = 0.0
    for px_raw, sz_raw in levels:
        if not (_is_pos_finite(px_raw) and _is_pos_finite(sz_raw)):
            continue
        px = float(px_raw)
        sz = float(sz_raw)
        lvl_usd = px * sz
        take_usd = min(remaining_usd, lvl_usd)
        take_qty = take_usd / px
        filled_qty += take_qty
        spent_usd += take_usd
        remaining_usd -= take_usd
        if remaining_usd <= 1e-6:
            break
    if filled_qty <= 0.0 or remaining_usd > 1e-3:
        return 50.0
    vwap = spent_usd / filled_qty
    if side.lower() == "buy":
        slip = ((vwap / mid_price) - 1.0) * 10_000.0
    else:
        slip = (1.0 - (vwap / mid_price)) * 10_000.0
    return max(0.0, float(slip))


@dataclass(frozen=True)
class PairMarketSnapshot:
    """Single-coin dual-leg (Binance Spot + Hyperliquid Perp) live market snapshot."""

    coin: str
    ts_observe_ms: int
    spot_bid: float | None
    spot_ask: float | None
    spot_bid_qty: float | None
    spot_ask_qty: float | None
    spot_ts_ms: int | None
    perp_bid: float | None
    perp_ask: float | None
    perp_mark: float | None
    perp_oracle: float | None
    perp_funding_1h: float | None
    perp_ts_ms: int | None
    perp_l2_bids: tuple[tuple[float, float], ...] = ()
    perp_l2_asks: tuple[tuple[float, float], ...] = ()
    dvol: float | None = None
    spot_source: str = ""
    perp_source: str = ""
    provenance: str = PROVENANCE_REAL

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["perp_l2_bids"] = [list(lv) for lv in self.perp_l2_bids[:5]]
        d["perp_l2_asks"] = [list(lv) for lv in self.perp_l2_asks[:5]]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PairMarketSnapshot:
        return cls(
            coin=str(d["coin"]),
            ts_observe_ms=int(d["ts_observe_ms"]),
            spot_bid=d.get("spot_bid"),
            spot_ask=d.get("spot_ask"),
            spot_bid_qty=d.get("spot_bid_qty"),
            spot_ask_qty=d.get("spot_ask_qty"),
            spot_ts_ms=d.get("spot_ts_ms"),
            perp_bid=d.get("perp_bid"),
            perp_ask=d.get("perp_ask"),
            perp_mark=d.get("perp_mark"),
            perp_oracle=d.get("perp_oracle"),
            perp_funding_1h=d.get("perp_funding_1h"),
            perp_ts_ms=d.get("perp_ts_ms"),
            perp_l2_bids=tuple((float(a), float(b)) for a, b in (d.get("perp_l2_bids") or ())),
            perp_l2_asks=tuple((float(a), float(b)) for a, b in (d.get("perp_l2_asks") or ())),
            dvol=d.get("dvol"),
            spot_source=str(d.get("spot_source", "")),
            perp_source=str(d.get("perp_source", "")),
            provenance=str(d.get("provenance", PROVENANCE_REAL)),
        )


@dataclass(frozen=True)
class ShadowMonitorConfig:
    """Configuration for the strictly read-only W3-DN >=48h shadow blind-run monitor."""

    state_dir: Path
    coins: tuple[str, ...] = DEFAULT_W3_DN_SHADOW_COINS
    capital_usd: float = 10_000.0
    max_spot_gross: float = 0.65
    max_pair_leg_weight: float = 0.35
    basis_stop_bps: float = 350.0
    margin_storm_exit_ratio: float = 0.50
    max_stale_ms: int = 120_000
    max_future_skew_ms: int = 5_000
    max_leg_desync_ms: int = 60_000
    max_tick_gap_sec: float = 300.0
    max_consecutive_errors: int = 5
    alert_cooldown_sec: float = 300.0
    kernel_bar_interval_sec: int = 43_200  # 12h canonical strategy bar cadence
    required_shadow_hours: float = 48.0
    min_uptime_ratio: float = 0.99
    min_healthy_coin_ratio: float = 1.0  # RB-08: require full configured universe valid per healthy tick
    enforce_wall_clock_bounds: bool = True  # RB-10: reject spoofed future ts_observe_ms in live runs
    spot_taker_fee_bps: float = COST_BINANCE_SPOT.taker_fee_bps
    perp_taker_fee_bps: float = COST_HYPERLIQUID_PERP.taker_fee_bps
    log_max_bytes: int = 5 * 1024 * 1024
    log_backup_count: int = 3
    p16_heartbeat_url: str = ""
    service_name: str = "w3_dn_shadow_monitor"


@dataclass
class VirtualPairPosition:
    """Paper-only dual-leg position enforcing atomic w_spot + w_perp == 0.0."""

    coin: str
    spot_qty: float = 0.0
    perp_qty: float = 0.0  # Negative = short perp
    spot_entry_px: float = 0.0
    perp_entry_px: float = 0.0
    weight_spot: float = 0.0
    weight_perp: float = 0.0
    opened_ts_ms: int = 0
    last_funding_ts_ms: int = 0

    @property
    def is_open(self) -> bool:
        return abs(self.weight_spot) > 1e-9 or abs(self.weight_perp) > 1e-9

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VirtualPairPosition:
        return cls(
            coin=str(d["coin"]),
            spot_qty=float(d.get("spot_qty", 0.0)),
            perp_qty=float(d.get("perp_qty", 0.0)),
            spot_entry_px=float(d.get("spot_entry_px", 0.0)),
            perp_entry_px=float(d.get("perp_entry_px", 0.0)),
            weight_spot=float(d.get("weight_spot", 0.0)),
            weight_perp=float(d.get("weight_perp", 0.0)),
            opened_ts_ms=int(d.get("opened_ts_ms", 0)),
            last_funding_ts_ms=int(d.get("last_funding_ts_ms", 0)),
        )


@dataclass(frozen=True)
class ShadowReadinessReport:
    """Machine-verifiable >=48h shadow blind-run readiness assessment."""

    verdict: str  # "READY_FOR_OWNER_REVIEW" | "SHADOW_OBSERVATION_IN_PROGRESS" | "FAIL_CLOSED_BLOCKED"
    live_execution_enabled: bool
    health_status: str
    elapsed_hours: float
    required_hours: float
    total_ticks: int
    healthy_ticks: int
    uptime_ratio: float
    audit_chain_intact: bool
    audit_chain_length: int
    max_abs_delta_weight: float
    crossed_book_bypass_count: int
    max_observed_abs_basis_bps: float
    basis_stop_bps: float
    min_perp_margin_buffer_ratio: float
    margin_storm_exit_ratio: float
    mean_observed_4leg_cost_bps: float
    backtest_assumed_4leg_cost_bps: float
    shadow_equity_usd: float
    shadow_net_pnl_usd: float
    shadow_funding_usd: float
    shadow_cost_usd: float
    alert_counts_by_code: dict[str, int]
    alert_suppressed_by_code: dict[str, int]
    blocking_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class W3DNQuoteProvider(Protocol):
    """Pluggable strictly read-only quote provider (P8 MD-Gateway, Public REST, or Replay)."""

    def fetch_snapshots(self, coins: Sequence[str], *, trace_id: str) -> dict[str, PairMarketSnapshot]:
        ...


class PublicRestWaterfallQuoteProvider:
    """STD-32/STD-34 compliant public REST Waterfall quote provider (zero raw Binance WS)."""

    def __init__(
        self,
        *,
        client: RateLimitedHTTPClient | None = None,
        spot_bases: Sequence[str] = _BINANCE_SPOT_WATERFALL_BASES,
        hl_info_url: str = _HYPERLIQUID_INFO_URL,
        deribit_dvol_url: str = _DERIBIT_DVOL_URL,
        timeout_sec: float = 10.0,
        max_dvol_stale_sec: float = 21_600.0,
    ) -> None:
        self.client = client or RateLimitedHTTPClient(default_timeout=float(timeout_sec))
        self.spot_bases = tuple(str(b).rstrip("/") for b in spot_bases)
        self.hl_info_url = str(hl_info_url)
        self.deribit_dvol_url = str(deribit_dvol_url)
        self.timeout_sec = float(timeout_sec)
        self.max_dvol_stale_sec = max(300.0, float(max_dvol_stale_sec))
        self._last_dvol: float | None = None
        self._last_dvol_fetch_mono: float = 0.0

    def _fetch_binance_spot_bbo(self, coin: str) -> tuple[float, float, float, float, int, str] | None:
        sym = f"{coin.upper()}USDT"
        for base in self.spot_bases:
            url = f"{base}/api/v3/ticker/bookTicker?symbol={sym}"
            try:
                payload = self.client.fetch_json(
                    url,
                    method="GET",
                    timeout_sec=self.timeout_sec,
                    max_retries=2,
                    min_interval_sec=0.15,
                )
                if not isinstance(payload, dict):
                    continue
                bid = float(payload["bidPrice"])
                ask = float(payload["askPrice"])
                bid_qty = float(payload.get("bidQty", 0.0))
                ask_qty = float(payload.get("askQty", 0.0))
                # RB-16: fail over to next host on <=0 or crossed BBO!
                if not (_is_pos_finite(bid) and _is_pos_finite(ask) and bid < ask):
                    logger.warning("Spot waterfall endpoint %s returned invalid BBO for %s: %s/%s", base, sym, bid, ask)
                    continue
                if not (_is_pos_finite(bid_qty) and _is_pos_finite(ask_qty)):
                    continue
                ts_ms = utc_now_ms()
                return bid, ask, bid_qty, ask_qty, ts_ms, base
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                logger.warning("Spot waterfall endpoint %s failed for %s: %s", base, sym, exc)
                continue
        return None

    def _fetch_hl_meta_and_ctxs(self) -> dict[str, dict[str, Any]]:
        payload = self.client.fetch_json(
            self.hl_info_url,
            method="POST",
            json_payload={"type": "metaAndAssetCtxs"},
            timeout_sec=self.timeout_sec,
            max_retries=3,
            min_interval_sec=0.35,
        )
        if not isinstance(payload, list) or len(payload) < 2:
            return {}
        meta, ctxs = payload[0], payload[1]
        universe = meta.get("universe", []) if isinstance(meta, dict) else []
        out: dict[str, dict[str, Any]] = {}
        for idx, item in enumerate(universe):
            if idx >= len(ctxs) or not isinstance(item, dict) or not isinstance(ctxs[idx], dict):
                continue
            name = str(item.get("name", "")).upper()
            if name:
                out[name] = ctxs[idx]
        return out

    def _fetch_hl_l2_book(
        self, coin: str
    ) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...], int] | None:
        payload = self.client.fetch_json(
            self.hl_info_url,
            method="POST",
            json_payload={"type": "l2Book", "coin": coin.upper()},
            timeout_sec=self.timeout_sec,
            max_retries=3,
            min_interval_sec=0.35,
        )
        if not isinstance(payload, dict):
            return None
        levels = payload.get("levels")
        if not isinstance(levels, list) or len(levels) < 2:
            return None
        raw_bids, raw_asks = levels[0], levels[1]
        bids = tuple(
            (float(lv["px"]), float(lv["sz"]))
            for lv in raw_bids[:5]
            if isinstance(lv, dict) and "px" in lv and "sz" in lv
        )
        asks = tuple(
            (float(lv["px"]), float(lv["sz"]))
            for lv in raw_asks[:5]
            if isinstance(lv, dict) and "px" in lv and "sz" in lv
        )
        ts_ms = int(payload.get("time") or utc_now_ms())
        return bids, asks, ts_ms

    def _fetch_dvol_cached(self) -> float | None:
        now_mono = time.monotonic()
        if self._last_dvol is not None and (now_mono - self._last_dvol_fetch_mono) < 300.0:
            return self._last_dvol
        end_ms = utc_now_ms()
        start_ms = end_ms - 6 * 3_600_000
        url = (
            f"{self.deribit_dvol_url}?currency=BTC&resolution=3600"
            f"&start_timestamp={start_ms}&end_timestamp={end_ms}"
        )
        try:
            payload = self.client.fetch_json(
                url,
                method="GET",
                timeout_sec=self.timeout_sec,
                max_retries=2,
                min_interval_sec=0.25,
            )
            data = payload.get("result", {}).get("data", []) if isinstance(payload, dict) else []
            if data and isinstance(data[-1], list) and len(data[-1]) >= 5:
                val = float(data[-1][4])
                if _is_pos_finite(val):
                    self._last_dvol = val
                    self._last_dvol_fetch_mono = now_mono
                    return val
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            logger.warning("Deribit DVOL fetch failed: %s", exc)
        # RB-16: expire cached DVOL after max_dvol_stale_sec rather than reusing indefinitely
        if self._last_dvol is not None and (now_mono - self._last_dvol_fetch_mono) <= self.max_dvol_stale_sec:
            return self._last_dvol
        self._last_dvol = None
        return None

    def fetch_snapshots(self, coins: Sequence[str], *, trace_id: str) -> dict[str, PairMarketSnapshot]:
        _ = trace_id
        hl_ctxs: dict[str, dict[str, Any]] = {}
        try:
            hl_ctxs = self._fetch_hl_meta_and_ctxs()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            logger.warning("Hyperliquid metaAndAssetCtxs failed: %s", exc)

        dvol_val = self._fetch_dvol_cached()
        out: dict[str, PairMarketSnapshot] = {}

        for raw_coin in coins:
            coin = str(raw_coin).strip().upper()
            spot = self._fetch_binance_spot_bbo(coin)
            ctx = hl_ctxs.get(coin)
            l2 = None
            try:
                l2 = self._fetch_hl_l2_book(coin)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
                logger.warning("Hyperliquid l2Book failed for %s: %s", coin, exc)

            # Capture observation timestamp AFTER fetching this coin's quotes (prevents false CLOCK_SKEW on retry)
            coin_obs_ms = utc_now_ms()
            perp_bids = l2[0] if l2 else ()
            perp_asks = l2[1] if l2 else ()
            perp_mark: float | None = None
            perp_oracle: float | None = None
            perp_funding_1h: float | None = None
            if ctx:
                try:
                    perp_mark = float(ctx["markPx"]) if ctx.get("markPx") is not None else None
                    perp_oracle = float(ctx["oraclePx"]) if ctx.get("oraclePx") is not None else None
                    perp_funding_1h = float(ctx["funding"]) if ctx.get("funding") is not None else None
                except (TypeError, ValueError):
                    perp_mark = None
                    perp_oracle = None
                    perp_funding_1h = None

            out[coin] = PairMarketSnapshot(
                coin=coin,
                ts_observe_ms=coin_obs_ms,
                spot_bid=spot[0] if spot else None,
                spot_ask=spot[1] if spot else None,
                spot_bid_qty=spot[2] if spot else None,
                spot_ask_qty=spot[3] if spot else None,
                spot_ts_ms=spot[4] if spot else None,
                perp_bid=perp_bids[0][0] if perp_bids else None,
                perp_ask=perp_asks[0][0] if perp_asks else None,
                perp_mark=perp_mark,
                perp_oracle=perp_oracle,
                perp_funding_1h=perp_funding_1h,
                perp_ts_ms=l2[2] if l2 else (coin_obs_ms if ctx else None),
                perp_l2_bids=perp_bids,
                perp_l2_asks=perp_asks,
                dvol=dvol_val,
                spot_source=spot[5] if spot else "",
                perp_source=self.hl_info_url if (ctx or l2) else "",
                provenance=PROVENANCE_REAL,
            )
        return out


class P8MDGatewayFileQuoteProvider:
    """STD-32 P8 MD-Gateway consumer with O(1) EOF seek, empty-file safety, and strict ts_ms (RB-09)."""

    def __init__(
        self,
        p8_feed_path: Path | str,
        *,
        perp_provider: PublicRestWaterfallQuoteProvider | None = None,
        tail_bytes: int = 262_144,
    ) -> None:
        self.p8_feed_path = Path(p8_feed_path)
        self.perp_provider = perp_provider or PublicRestWaterfallQuoteProvider()
        self.tail_bytes = max(4_096, int(tail_bytes))

    def _read_bounded_tail_lines(self) -> tuple[str | None, list[str]]:
        """Read optional header line + bounded EOF tail bytes in O(1) memory even for multi-GB logs."""
        if not self.p8_feed_path.is_file():
            return None, []
        size = self.p8_feed_path.stat().st_size
        if size <= 0:
            return None, []
        with self.p8_feed_path.open("rb") as fh:
            header_raw = fh.readline()
            header = header_raw.decode("utf-8", errors="ignore").strip() if header_raw else None
            if size <= self.tail_bytes:
                fh.seek(0)
                text = fh.read().decode("utf-8", errors="ignore")
                return header, [ln for ln in text.splitlines() if ln.strip()]
            fh.seek(max(0, size - self.tail_bytes))
            _ = fh.readline()  # discard partial line at seek offset
            text = fh.read().decode("utf-8", errors="ignore")
            return header, [ln for ln in text.splitlines() if ln.strip()]

    def _read_latest_p8_spot_bbo(self, coins: Sequence[str]) -> dict[str, tuple[float, float, float, float, int]]:
        latest: dict[str, tuple[float, float, float, float, int]] = {}
        header, lines = self._read_bounded_tail_lines()
        if not lines:
            return latest
        wanted = {str(c).strip().upper() for c in coins}
        if self.p8_feed_path.suffix.lower() == ".csv":
            if not header:
                return latest
            csv_lines = lines if lines[0].strip() == header else ([header] + lines)
            reader = csv.DictReader(csv_lines)
            for row in reader:
                if not row:
                    continue
                sym = _normalize_coin_symbol(str(row.get("symbol", row.get("coin", ""))))
                if sym not in wanted:
                    continue
                ts_raw = row.get("ts_ms")
                if ts_raw is None or str(ts_raw).strip() == "":
                    continue
                try:
                    latest[sym] = (
                        float(row["bid"]),
                        float(row["ask"]),
                        float(row.get("bid_qty", 1.0)),
                        float(row.get("ask_qty", 1.0)),
                        int(ts_raw),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            for line in lines:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                sym = _normalize_coin_symbol(str(obj.get("symbol", obj.get("coin", ""))))
                if sym not in wanted:
                    continue
                # RB-09: strictly require explicit ts_ms (never fabricate utc_now_ms() on missing timestamp!)
                if "ts_ms" not in obj or obj["ts_ms"] is None:
                    continue
                try:
                    latest[sym] = (
                        float(obj["bid"]),
                        float(obj["ask"]),
                        float(obj.get("bid_qty", 1.0)),
                        float(obj.get("ask_qty", 1.0)),
                        int(obj["ts_ms"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        return latest

    def fetch_snapshots(self, coins: Sequence[str], *, trace_id: str) -> dict[str, PairMarketSnapshot]:
        _ = trace_id
        obs_ms = utc_now_ms()
        p8_spot = self._read_latest_p8_spot_bbo(coins)
        hl_ctxs: dict[str, dict[str, Any]] = {}
        try:
            hl_ctxs = self.perp_provider._fetch_hl_meta_and_ctxs()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
            hl_ctxs = {}
        dvol_val = self.perp_provider._fetch_dvol_cached()

        out: dict[str, PairMarketSnapshot] = {}
        for raw_coin in coins:
            coin = str(raw_coin).strip().upper()
            sp = p8_spot.get(coin)
            ctx = hl_ctxs.get(coin)
            l2 = None
            try:
                l2 = self.perp_provider._fetch_hl_l2_book(coin)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
                l2 = None

            perp_bids = l2[0] if l2 else ()
            perp_asks = l2[1] if l2 else ()
            out[coin] = PairMarketSnapshot(
                coin=coin,
                ts_observe_ms=obs_ms,
                spot_bid=sp[0] if sp else None,
                spot_ask=sp[1] if sp else None,
                spot_bid_qty=sp[2] if sp else None,
                spot_ask_qty=sp[3] if sp else None,
                spot_ts_ms=sp[4] if sp else None,
                perp_bid=perp_bids[0][0] if perp_bids else None,
                perp_ask=perp_asks[0][0] if perp_asks else None,
                perp_mark=float(ctx["markPx"]) if (ctx and ctx.get("markPx") is not None) else None,
                perp_oracle=float(ctx["oraclePx"]) if (ctx and ctx.get("oraclePx") is not None) else None,
                perp_funding_1h=float(ctx["funding"]) if (ctx and ctx.get("funding") is not None) else None,
                perp_ts_ms=l2[2] if l2 else (obs_ms if ctx else None),
                perp_l2_bids=perp_bids,
                perp_l2_asks=perp_asks,
                dvol=dvol_val,
                spot_source=f"p8_md_gateway:{self.p8_feed_path.name}",
                perp_source=self.perp_provider.hl_info_url if (ctx or l2) else "",
                provenance=PROVENANCE_REAL,
            )
        return out


class ReplayQuoteProvider:
    """Deterministic offline quote provider for unit & Red-Blue adversarial stress tests."""

    def __init__(self, frames: Sequence[dict[str, PairMarketSnapshot]]) -> None:
        self._frames = list(frames)
        self._cursor = 0

    def fetch_snapshots(self, coins: Sequence[str], *, trace_id: str) -> dict[str, PairMarketSnapshot]:
        _ = (coins, trace_id)
        if not self._frames:
            return {}
        idx = min(self._cursor, len(self._frames) - 1)
        self._cursor += 1
        return dict(self._frames[idx])


class W3DNShadowMonitor:
    """Strictly read-only >=48h shadow blind-run monitor with WAL checkpoint recovery."""

    def __init__(
        self,
        config: ShadowMonitorConfig,
        *,
        provider: W3DNQuoteProvider | None = None,
        kernel: Web3DeltaNeutralCarryKernel | None = None,
    ) -> None:
        if SHADOW_LIVE_EXECUTION_ENABLED:
            raise RuntimeError(f"{PlatformErrorCode.LIVE_SWITCH_BLOCKED.value}: live execution forbidden")

        self.config = config
        self.state_dir = Path(config.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "w3_dn_shadow.sqlite"
        self.log_path = self.state_dir / "w3_dn_shadow.jsonl"
        self.heartbeat_path = self.state_dir / "w3_dn_shadow_heartbeat.json"

        self.provider: W3DNQuoteProvider = provider or PublicRestWaterfallQuoteProvider()
        self.kernel: Web3DeltaNeutralCarryKernel = kernel or Web3DeltaNeutralCarryKernel(
            max_spot_gross=config.max_spot_gross,
            max_pair_leg_weight=config.max_pair_leg_weight,
            basis_stop_bps=config.basis_stop_bps,
            margin_storm_exit_ratio=config.margin_storm_exit_ratio,
            capital_budget_usd=config.capital_usd,
        )

        self.conn: LockedConnection = connect_wal(self.db_path)
        self._init_schema()
        self._log_handler: RotatingFileHandler | None = None
        self._trace_logger = self._init_rotating_logger()

        # Virtual accounting & telemetry state (hydrated from SQLite checkpoint if restarting — RB-01)
        self._cash_usd: float = float(config.capital_usd)
        self._perp_collateral_base_usd: float = float(config.capital_usd) * max(
            0.20, 1.0 - float(config.max_spot_gross)
        )
        self._perp_realized_pnl_usd: float = 0.0
        self._total_funding_usd: float = 0.0
        self._total_cost_usd: float = 0.0
        self._positions: dict[str, VirtualPairPosition] = {}
        self._last_valid_snaps: dict[str, PairMarketSnapshot] = {}

        self._start_ts_ms: int | None = None
        self._last_ts_ms: int | None = None
        # RB-10: initialize monotonic anchor at startup so tick-0 outages trigger dead-man switch
        self._last_good_mono: float = time.monotonic()
        self._consecutive_errors: int = 0
        self._total_ticks: int = 0
        self._healthy_ticks: int = 0
        self._crossed_book_bypass_count: int = 0
        self._max_abs_delta_weight: float = 0.0
        self._max_observed_abs_basis_bps: float = 0.0
        self._min_perp_margin_buffer_ratio: float = 1.0
        self._observed_4leg_cost_sum: float = 0.0
        self._observed_4leg_cost_count: int = 0
        self._last_kernel_bucket: int | None = None
        self._basis_cooldown_until_bucket: dict[str, int] = {}
        self._last_chain_hash: str = self._load_last_chain_hash()

        self._alert_state: dict[tuple[str, str], tuple[float, int]] = {}
        self._alert_counts: dict[str, int] = {}
        self._alert_suppressed_counts: dict[str, int] = {}
        self._health_status: str = "INITIALIZING"

        self._hydrate_checkpoint()

    def __enter__(self) -> W3DNShadowMonitor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._log_handler is not None:
            self._trace_logger.removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None
        self.conn.close()

    def _init_rotating_logger(self) -> logging.Logger:
        lg = logging.getLogger(f"alphaloop.shadow.w3dn.{id(self)}")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        handler = RotatingFileHandler(
            str(self.log_path),
            maxBytes=max(65_536, int(self.config.log_max_bytes)),
            backupCount=max(1, int(self.config.log_backup_count)),
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(handler)
        self._log_handler = handler
        return lg

    def _log_event(self, *, trace_id: str, event: str, level: str, payload: dict[str, Any]) -> None:
        rec = {
            "ts_utc_ms": utc_now_ms(),
            "monotonic_s": round(time.monotonic(), 6),
            "trace_id": trace_id,
            "service": self.config.service_name,
            "live_execution_enabled": SHADOW_LIVE_EXECUTION_ENABLED,
            "event": event,
            "level": level,
            "payload": payload,
        }
        line = json.dumps(rec, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._trace_logger.info(line)

    def _init_schema(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    ts_observe_ms INTEGER NOT NULL,
                    coin TEXT NOT NULL,
                    spot_bid REAL,
                    spot_ask REAL,
                    perp_bid REAL,
                    perp_ask REAL,
                    perp_mark REAL,
                    perp_funding_1h REAL,
                    basis_mid_bps REAL,
                    basis_mark_bps REAL,
                    exec_4leg_cost_bps REAL,
                    dvol REAL,
                    guard_status TEXT NOT NULL,
                    error_code TEXT NOT NULL,
                    provenance TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shadow_ticks_ts ON shadow_ticks (ts_observe_ms, coin);"
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    coin TEXT NOT NULL,
                    action TEXT NOT NULL,
                    weight_spot REAL NOT NULL,
                    weight_perp REAL NOT NULL,
                    net_delta_weight REAL NOT NULL,
                    spot_exec_px REAL NOT NULL,
                    perp_exec_px REAL NOT NULL,
                    est_cost_usd REAL NOT NULL,
                    equity_usd REAL NOT NULL,
                    reason TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    code TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    suppressed_prior INTEGER NOT NULL,
                    message TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_audit_chain (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    trace_id TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    chain_hash TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_state_checkpoint (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    updated_ts_ms INTEGER NOT NULL,
                    state_json TEXT NOT NULL
                );
                """
            )

    def _build_checkpoint_dict(self) -> dict[str, Any]:
        return {
            "cash_usd": self._cash_usd,
            "perp_collateral_base_usd": self._perp_collateral_base_usd,
            "perp_realized_pnl_usd": self._perp_realized_pnl_usd,
            "total_funding_usd": self._total_funding_usd,
            "total_cost_usd": self._total_cost_usd,
            "positions": {c: p.to_dict() for c, p in sorted(self._positions.items()) if p.is_open},
            "last_valid_snaps": {c: s.to_dict() for c, s in sorted(self._last_valid_snaps.items())},
            "start_ts_ms": self._start_ts_ms,
            "last_ts_ms": self._last_ts_ms,
            "consecutive_errors": self._consecutive_errors,
            "total_ticks": self._total_ticks,
            "healthy_ticks": self._healthy_ticks,
            "crossed_book_bypass_count": self._crossed_book_bypass_count,
            "max_abs_delta_weight": self._max_abs_delta_weight,
            "max_observed_abs_basis_bps": self._max_observed_abs_basis_bps,
            "min_perp_margin_buffer_ratio": self._min_perp_margin_buffer_ratio,
            "observed_4leg_cost_sum": self._observed_4leg_cost_sum,
            "observed_4leg_cost_count": self._observed_4leg_cost_count,
            "last_kernel_bucket": self._last_kernel_bucket,
            "basis_cooldown_until_bucket": dict(self._basis_cooldown_until_bucket),
            "alert_counts": dict(self._alert_counts),
            "alert_suppressed_counts": dict(self._alert_suppressed_counts),
            "health_status": self._health_status,
            "kernel_state": self.kernel.export_state(),
        }

    def _hydrate_checkpoint(self) -> None:
        """Restore complete virtual portfolio, telemetry & kernel state on restart (RB-01)."""
        row = self.conn.fetch_one("SELECT state_json FROM shadow_state_checkpoint WHERE id = 1")
        if row is None:
            return
        st = json.loads(str(row["state_json"]))
        self._cash_usd = float(st.get("cash_usd", self._cash_usd))
        self._perp_collateral_base_usd = float(
            st.get("perp_collateral_base_usd", self._perp_collateral_base_usd)
        )
        self._perp_realized_pnl_usd = float(st.get("perp_realized_pnl_usd", 0.0))
        self._total_funding_usd = float(st.get("total_funding_usd", 0.0))
        self._total_cost_usd = float(st.get("total_cost_usd", 0.0))
        self._positions = {
            c: VirtualPairPosition.from_dict(pd) for c, pd in (st.get("positions") or {}).items()
        }
        self._last_valid_snaps = {
            c: PairMarketSnapshot.from_dict(sd) for c, sd in (st.get("last_valid_snaps") or {}).items()
        }
        self._start_ts_ms = st.get("start_ts_ms")
        self._last_ts_ms = st.get("last_ts_ms")
        self._consecutive_errors = int(st.get("consecutive_errors", 0))
        self._total_ticks = int(st.get("total_ticks", 0))
        self._healthy_ticks = int(st.get("healthy_ticks", 0))
        self._crossed_book_bypass_count = int(st.get("crossed_book_bypass_count", 0))
        self._max_abs_delta_weight = float(st.get("max_abs_delta_weight", 0.0))
        self._max_observed_abs_basis_bps = float(st.get("max_observed_abs_basis_bps", 0.0))
        self._min_perp_margin_buffer_ratio = float(st.get("min_perp_margin_buffer_ratio", 1.0))
        self._observed_4leg_cost_sum = float(st.get("observed_4leg_cost_sum", 0.0))
        self._observed_4leg_cost_count = int(st.get("observed_4leg_cost_count", 0))
        self._last_kernel_bucket = st.get("last_kernel_bucket")
        self._basis_cooldown_until_bucket = {
            str(k): int(v) for k, v in (st.get("basis_cooldown_until_bucket") or {}).items()
        }
        self._alert_counts = {str(k): int(v) for k, v in (st.get("alert_counts") or {}).items()}
        self._alert_suppressed_counts = {
            str(k): int(v) for k, v in (st.get("alert_suppressed_counts") or {}).items()
        }
        self._health_status = str(st.get("health_status", "INITIALIZING"))
        if isinstance(st.get("kernel_state"), dict):
            self.kernel.restore_state(st["kernel_state"])

    def _persist_checkpoint_only(self, ts_ms: int) -> None:
        state_json = json.dumps(self._build_checkpoint_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT INTO shadow_state_checkpoint (id, updated_ts_ms, state_json)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET updated_ts_ms=excluded.updated_ts_ms, state_json=excluded.state_json
                """,
                (int(ts_ms), state_json),
            )

    def _load_last_chain_hash(self) -> str:
        row = self.conn.fetch_one("SELECT chain_hash FROM shadow_audit_chain ORDER BY seq DESC LIMIT 1")
        return str(row["chain_hash"]) if row is not None else ("0" * 64)

    def _append_audit_chain_and_checkpoint(
        self, *, ts_ms: int, trace_id: str, payload: dict[str, Any]
    ) -> str:
        """Atomically append tamper-evident audit hash AND persist state checkpoint in one WAL txn (RB-01)."""
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        preimage = f"{self._last_chain_hash}|{int(ts_ms)}|{trace_id}|{payload_json}".encode("utf-8")
        chain_hash = hashlib.sha256(preimage).hexdigest()
        state_json = json.dumps(self._build_checkpoint_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT INTO shadow_audit_chain (ts_ms, trace_id, prev_hash, payload_json, chain_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (int(ts_ms), trace_id, self._last_chain_hash, payload_json, chain_hash),
            )
            self.conn.execute(
                """
                INSERT INTO shadow_state_checkpoint (id, updated_ts_ms, state_json)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET updated_ts_ms=excluded.updated_ts_ms, state_json=excluded.state_json
                """,
                (int(ts_ms), state_json),
            )
        self._last_chain_hash = chain_hash
        return chain_hash

    def verify_audit_chain(self) -> tuple[bool, int]:
        """Verify complete SHA-256 tamper-evident audit chain integrity from genesis."""
        rows = self.conn.fetch_all(
            "SELECT seq, ts_ms, trace_id, prev_hash, payload_json, chain_hash "
            "FROM shadow_audit_chain ORDER BY seq ASC"
        )
        expected_prev = "0" * 64
        for r in rows:
            if str(r["prev_hash"]) != expected_prev:
                return False, len(rows)
            preimage = (
                f"{expected_prev}|{int(r['ts_ms'])}|{str(r['trace_id'])}|{str(r['payload_json'])}"
            ).encode("utf-8")
            calc = hashlib.sha256(preimage).hexdigest()
            if calc != str(r["chain_hash"]):
                return False, len(rows)
            expected_prev = calc
        return True, len(rows)

    def _emit_alert(
        self,
        *,
        trace_id: str,
        ts_ms: int,
        code: PlatformErrorCode,
        coin: str,
        message: str,
    ) -> bool:
        """Emit tiered alert with per-(code, coin) anti-storm cooldown throttling (RB-16 accurate counts)."""
        key = (code.value, coin)
        now_mono = time.monotonic()
        prev = self._alert_state.get(key)
        if prev is not None:
            last_mono, suppressed = prev
            if (now_mono - last_mono) < self.config.alert_cooldown_sec:
                self._alert_state[key] = (last_mono, suppressed + 1)
                self._alert_suppressed_counts[code.value] = self._alert_suppressed_counts.get(code.value, 0) + 1
                return False
            prior_suppressed = suppressed
        else:
            prior_suppressed = 0

        self._alert_state[key] = (now_mono, 0)
        self._alert_counts[code.value] = self._alert_counts.get(code.value, 0) + 1
        severity = _CODE_SEVERITY.get(code, "HIGH")
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT INTO shadow_alerts (trace_id, ts_ms, code, severity, coin, suppressed_prior, message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (trace_id, int(ts_ms), code.value, severity, coin, int(prior_suppressed), message),
            )
        self._log_event(
            trace_id=trace_id,
            event="SHADOW_ALERT",
            level=severity,
            payload={
                "code": code.value,
                "coin": coin,
                "suppressed_prior": prior_suppressed,
                "message": message,
            },
        )
        return True

    def validate_snapshot(self, snap: PairMarketSnapshot) -> tuple[bool, PlatformErrorCode, str]:
        """Strict Fail-Closed anti-poisoning validation on every live dual-leg snapshot (RB-10)."""
        spot_present = snap.spot_bid is not None and snap.spot_ask is not None and snap.spot_ts_ms is not None
        perp_present = (
            snap.perp_bid is not None
            and snap.perp_ask is not None
            and snap.perp_ts_ms is not None
            and snap.perp_funding_1h is not None
        )
        if not (spot_present and perp_present):
            return False, PlatformErrorCode.UNPAIRED_LEG, f"Missing spot or perp leg for {snap.coin}"

        if not all(_is_pos_finite(x) for x in (snap.spot_bid, snap.spot_ask, snap.perp_bid, snap.perp_ask)):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Non-finite/non-positive BBO for {snap.coin}"
        if not _is_finite(snap.perp_funding_1h):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Non-finite funding rate for {snap.coin}"
        # RB-10: validate quantities & oracle/mark/dvol so NaN never leaks into CanonicalBar
        if snap.spot_bid_qty is not None and not _is_pos_finite(snap.spot_bid_qty):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Invalid spot_bid_qty for {snap.coin}"
        if snap.spot_ask_qty is not None and not _is_pos_finite(snap.spot_ask_qty):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Invalid spot_ask_qty for {snap.coin}"
        if snap.perp_mark is not None and not _is_pos_finite(snap.perp_mark):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Invalid perp_mark for {snap.coin}"
        if snap.perp_oracle is not None and not _is_pos_finite(snap.perp_oracle):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Invalid perp_oracle for {snap.coin}"
        if snap.dvol is not None and not _is_pos_finite(snap.dvol):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Invalid dvol for {snap.coin}"

        if float(snap.spot_bid) >= float(snap.spot_ask):
            return (
                False,
                PlatformErrorCode.CROSSED_BOOK,
                f"Spot crossed book {snap.coin}: bid={snap.spot_bid} >= ask={snap.spot_ask}",
            )
        if float(snap.perp_bid) >= float(snap.perp_ask):
            return (
                False,
                PlatformErrorCode.CROSSED_BOOK,
                f"Perp crossed book {snap.coin}: bid={snap.perp_bid} >= ask={snap.perp_ask}",
            )

        obs_ms = int(snap.ts_observe_ms)
        spot_ts = int(snap.spot_ts_ms)
        perp_ts = int(snap.perp_ts_ms)
        # RB-10: bound ts_observe_ms against local wall clock when live bounds are enabled
        if self.config.enforce_wall_clock_bounds:
            wall_ms = utc_now_ms()
            if (obs_ms - wall_ms) > self.config.max_future_skew_ms:
                return (
                    False,
                    PlatformErrorCode.CLOCK_SKEW,
                    f"Future ts_observe_ms skew on {snap.coin}: obs={obs_ms} > wall={wall_ms}",
                )
        if (spot_ts - obs_ms) > self.config.max_future_skew_ms or (perp_ts - obs_ms) > self.config.max_future_skew_ms:
            return (
                False,
                PlatformErrorCode.CLOCK_SKEW,
                f"Future quote timestamp skew on {snap.coin}: obs={obs_ms} spot={spot_ts} perp={perp_ts}",
            )
        if (obs_ms - spot_ts) > self.config.max_stale_ms or (obs_ms - perp_ts) > self.config.max_stale_ms:
            return (
                False,
                PlatformErrorCode.STALE_QUOTE,
                f"Stale quote on {snap.coin}: spot_age={obs_ms - spot_ts}ms perp_age={obs_ms - perp_ts}ms",
            )
        if abs(spot_ts - perp_ts) > self.config.max_leg_desync_ms:
            return (
                False,
                PlatformErrorCode.LEG_DESYNC,
                f"Leg timestamp desync on {snap.coin}: |{spot_ts} - {perp_ts}| > {self.config.max_leg_desync_ms}ms",
            )

        return True, PlatformErrorCode.OK, "OK"

    def _resolve_mark_prices_for_coin(
        self,
        coin: str,
        pos: VirtualPairPosition,
        valid_snaps: dict[str, PairMarketSnapshot],
        raw_snaps: dict[str, PairMarketSnapshot] | None = None,
    ) -> tuple[float, float]:
        """Resolve conservative liquidation MTM prices (never hiding a live partial-leg surge — RB-02)."""
        v_snap = valid_snaps.get(coin)
        raw_snap = raw_snaps.get(coin) if raw_snaps else None
        last_snap = self._last_valid_snaps.get(coin)

        spot_px = pos.spot_entry_px
        if v_snap and _is_pos_finite(v_snap.spot_bid):
            spot_px = float(v_snap.spot_bid)
        elif raw_snap and _is_pos_finite(raw_snap.spot_bid) and (
            raw_snap.spot_ask is None or float(raw_snap.spot_bid) < float(raw_snap.spot_ask)
        ):
            spot_px = float(raw_snap.spot_bid)
        elif last_snap and _is_pos_finite(last_snap.spot_bid):
            spot_px = float(last_snap.spot_bid)

        perp_px = pos.perp_entry_px
        if v_snap and _is_pos_finite(v_snap.perp_ask):
            perp_px = float(v_snap.perp_ask)
        elif raw_snap and _is_pos_finite(raw_snap.perp_mark):
            # Partial outage where spot timed out but live HL perp surged -> mark short perp loss!
            perp_px = float(raw_snap.perp_mark)
        elif raw_snap and _is_pos_finite(raw_snap.perp_ask) and (
            raw_snap.perp_bid is None or float(raw_snap.perp_bid) < float(raw_snap.perp_ask)
        ):
            perp_px = float(raw_snap.perp_ask)
        elif last_snap and _is_pos_finite(last_snap.perp_ask):
            perp_px = float(last_snap.perp_ask)

        return spot_px, perp_px

    def _compute_shadow_equity_and_margin(
        self,
        valid_snaps: dict[str, PairMarketSnapshot],
        raw_snaps: dict[str, PairMarketSnapshot] | None = None,
    ) -> tuple[float, float]:
        """Conservative liquidation MTM equity and simulated perp margin buffer ratio (RB-02, RB-07)."""
        spot_mtm_value = 0.0
        perp_unrealized = 0.0
        has_open = False
        for coin, pos in self._positions.items():
            if not pos.is_open:
                continue
            has_open = True
            spot_px, perp_px = self._resolve_mark_prices_for_coin(coin, pos, valid_snaps, raw_snaps)
            spot_mtm_value += pos.spot_qty * spot_px
            perp_unrealized += pos.perp_qty * (perp_px - pos.perp_entry_px)

        equity = max(0.0, self._cash_usd + spot_mtm_value + perp_unrealized)
        if not has_open:
            # RB-02: when 100% flat in cash, margin buffer is healthy (1.0) so monitor never wedges!
            return equity, 1.0
        perp_margin_equity = self._perp_collateral_base_usd + self._perp_realized_pnl_usd + perp_unrealized
        margin_ratio = (
            perp_margin_equity / self._perp_collateral_base_usd
            if self._perp_collateral_base_usd > 0.0
            else 1.0
        )
        return equity, margin_ratio

    def _rebalance_collateral_if_flat(self) -> None:
        """Reset virtual perp collateral pool from current cash equity once all pairs are closed (RB-02)."""
        if all(not p.is_open for p in self._positions.values()):
            self._perp_collateral_base_usd = max(
                1.0, self._cash_usd * max(0.20, 1.0 - float(self.config.max_spot_gross))
            )
            self._perp_realized_pnl_usd = 0.0

    def _close_virtual_pair(
        self,
        coin: str,
        mark_snaps: dict[str, PairMarketSnapshot],
        *,
        raw_snaps: dict[str, PairMarketSnapshot] | None = None,
        trace_id: str,
        ts_ms: int,
        reason: str,
    ) -> None:
        pos = self._positions.get(coin)
        if pos is None or not pos.is_open:
            return
        spot_exit_px, perp_exit_px = self._resolve_mark_prices_for_coin(coin, pos, mark_snaps, raw_snaps)

        spot_notional = abs(pos.spot_qty) * spot_exit_px
        perp_notional = abs(pos.perp_qty) * perp_exit_px
        spot_fee = spot_notional * (self.config.spot_taker_fee_bps / 10_000.0)
        perp_fee = perp_notional * (self.config.perp_taker_fee_bps / 10_000.0)
        total_fee = spot_fee + perp_fee

        perp_pnl = pos.perp_qty * (perp_exit_px - pos.perp_entry_px)
        self._cash_usd += (pos.spot_qty * spot_exit_px) + perp_pnl - total_fee
        self._perp_realized_pnl_usd += perp_pnl - perp_fee
        self._total_cost_usd += total_fee
        self._positions[coin] = VirtualPairPosition(coin=coin)
        self._rebalance_collateral_if_flat()

        self.kernel.sync_executed_weights({spot_leg_symbol(coin): 0.0, perp_leg_symbol(coin): 0.0})
        # RB-16: compute accurate portfolio equity across ALL valid snaps, not just the closed coin
        eq, _ = self._compute_shadow_equity_and_margin(mark_snaps, raw_snaps)
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT INTO shadow_events
                (trace_id, ts_ms, coin, action, weight_spot, weight_perp, net_delta_weight,
                 spot_exec_px, perp_exec_px, est_cost_usd, equity_usd, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    int(ts_ms),
                    coin,
                    "VIRTUAL_CLOSE_PAIR",
                    0.0,
                    0.0,
                    0.0,
                    spot_exit_px,
                    perp_exit_px,
                    total_fee,
                    eq,
                    reason,
                ),
            )

    def _apply_atomic_target_pair(
        self,
        coin: str,
        target_spot_w: float,
        target_perp_w: float,
        valid_snaps: dict[str, PairMarketSnapshot],
        *,
        trace_id: str,
        ts_ms: int,
        reason: str,
    ) -> None:
        """Apply strictly atomic paired virtual trade enforcing w_spot + w_perp == 0.0 & cash >= 0."""
        snap = valid_snaps[coin]
        w_s = max(0.0, float(target_spot_w))
        w_p = float(target_perp_w)
        if abs(w_s + w_p) > 1e-9:
            self._emit_alert(
                trace_id=trace_id,
                ts_ms=ts_ms,
                code=PlatformErrorCode.DELTA_INVARIANT_BREACH,
                coin=coin,
                message=f"Rejected non-atomic pair weights: w_spot={w_s}, w_perp={w_p}",
            )
            return

        self._max_abs_delta_weight = max(self._max_abs_delta_weight, abs(w_s + w_p))
        pos = self._positions.get(coin) or VirtualPairPosition(coin=coin)
        if w_s <= 1e-9:
            if pos.is_open:
                self._close_virtual_pair(coin, valid_snaps, trace_id=trace_id, ts_ms=ts_ms, reason=reason)
            return

        if pos.is_open and abs(w_s - pos.weight_spot) < 0.02:
            return

        if pos.is_open:
            self._close_virtual_pair(coin, valid_snaps, trace_id=trace_id, ts_ms=ts_ms, reason="REBALANCE_ROTATE")

        # RB-16: refresh portfolio equity & cap notional by available cash net of 2-leg entry fees
        eq_now, _ = self._compute_shadow_equity_and_margin(valid_snaps)
        fee_rate = (self.config.spot_taker_fee_bps + self.config.perp_taker_fee_bps) / 10_000.0
        max_affordable = max(0.0, self._cash_usd / (1.0 + fee_rate))
        target_notional = min(max(0.0, eq_now * w_s), max_affordable)
        if target_notional <= 1.0:
            return

        effective_w = target_notional / max(eq_now, 1.0)
        spot_entry_px = float(snap.spot_ask)
        perp_entry_px = float(snap.perp_bid)
        spot_fee = target_notional * (self.config.spot_taker_fee_bps / 10_000.0)
        perp_fee = target_notional * (self.config.perp_taker_fee_bps / 10_000.0)
        total_fee = spot_fee + perp_fee

        spot_qty = target_notional / spot_entry_px
        perp_qty = -(target_notional / perp_entry_px)

        self._cash_usd = max(0.0, self._cash_usd - target_notional - total_fee)
        self._perp_realized_pnl_usd -= perp_fee
        self._total_cost_usd += total_fee
        self._positions[coin] = VirtualPairPosition(
            coin=coin,
            spot_qty=spot_qty,
            perp_qty=perp_qty,
            spot_entry_px=spot_entry_px,
            perp_entry_px=perp_entry_px,
            weight_spot=effective_w,
            weight_perp=-effective_w,
            opened_ts_ms=int(ts_ms),
            last_funding_ts_ms=int(ts_ms),
        )
        self.kernel.sync_executed_weights({spot_leg_symbol(coin): effective_w, perp_leg_symbol(coin): -effective_w})
        eq_after, _ = self._compute_shadow_equity_and_margin(valid_snaps)
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT INTO shadow_events
                (trace_id, ts_ms, coin, action, weight_spot, weight_perp, net_delta_weight,
                 spot_exec_px, perp_exec_px, est_cost_usd, equity_usd, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    int(ts_ms),
                    coin,
                    "VIRTUAL_OPEN_PAIR",
                    effective_w,
                    -effective_w,
                    0.0,
                    spot_entry_px,
                    perp_entry_px,
                    total_fee,
                    eq_after,
                    reason,
                ),
            )

    def _accrue_funding(self, valid_snaps: dict[str, PairMarketSnapshot], ts_ms: int) -> float:
        accrued_total = 0.0
        for coin, pos in self._positions.items():
            if not pos.is_open:
                continue
            snap = valid_snaps.get(coin)
            if snap is None or snap.perp_funding_1h is None:
                continue
            dt_ms = max(0, int(ts_ms) - int(pos.last_funding_ts_ms))
            dt_hours = min(24.0, dt_ms / 3_600_000.0)
            if dt_hours <= 0.0:
                continue
            perp_mark = float(snap.perp_mark or (0.5 * (float(snap.perp_bid) + float(snap.perp_ask))))
            cashflow = (-pos.perp_qty) * perp_mark * float(snap.perp_funding_1h) * dt_hours
            self._cash_usd += cashflow
            self._perp_realized_pnl_usd += cashflow
            self._total_funding_usd += cashflow
            pos.last_funding_ts_ms = int(ts_ms)
            accrued_total += cashflow
        return accrued_total

    def _write_heartbeat(self, *, trace_id: str, ts_ms: int, equity_usd: float, margin_ratio: float) -> None:
        hb = {
            "service": self.config.service_name,
            "status": self._health_status,
            "live_execution_enabled": SHADOW_LIVE_EXECUTION_ENABLED,
            "ts_utc_ms": int(ts_ms),
            "ts_utc_iso": datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat(),
            "monotonic_s": round(time.monotonic(), 3),
            "trace_id": trace_id,
            "total_ticks": self._total_ticks,
            "healthy_ticks": self._healthy_ticks,
            "consecutive_errors": self._consecutive_errors,
            "shadow_equity_usd": round(equity_usd, 4),
            "perp_margin_buffer_ratio": round(margin_ratio, 6),
            "max_observed_abs_basis_bps": round(self._max_observed_abs_basis_bps, 4),
            "open_pairs": sorted(c for c, p in self._positions.items() if p.is_open),
            "last_chain_hash": self._last_chain_hash,
        }
        _atomic_write_json(self.heartbeat_path, hb)
        if self.config.p16_heartbeat_url:
            try:
                body = json.dumps(hb, allow_nan=False).encode("utf-8")
                req = urllib.request.Request(
                    self.config.p16_heartbeat_url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3.0):
                    pass
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                logger.warning("Optional P16 heartbeat POST failed (non-fatal): %s", exc)

    def run_once(self) -> dict[str, Any]:
        """Execute one deterministic read-only shadow observation, risk check, and paper tick."""
        if SHADOW_LIVE_EXECUTION_ENABLED:
            raise RuntimeError(f"{PlatformErrorCode.LIVE_SWITCH_BLOCKED.value}: live execution forbidden")

        trace_id = f"w3dn-{uuid.uuid4().hex[:12]}"
        now_mono = time.monotonic()
        self._total_ticks += 1

        # RB-10: monotonic dead-man switch active from tick 0
        dead_man_gap = now_mono - self._last_good_mono
        if dead_man_gap > self.config.max_tick_gap_sec:
            self._health_status = "DEGRADED"
            self._emit_alert(
                trace_id=trace_id,
                ts_ms=utc_now_ms(),
                code=PlatformErrorCode.DEAD_MAN_TRIGGERED,
                coin="ALL",
                message=f"Tick gap {dead_man_gap:.1f}s exceeded max_tick_gap_sec={self.config.max_tick_gap_sec}",
            )

        try:
            snapshots = self.provider.fetch_snapshots(self.config.coins, trace_id=trace_id)
        except Exception as exc:
            self._consecutive_errors += 1
            self._health_status = (
                "HALTED_FAIL_CLOSED"
                if self._consecutive_errors >= self.config.max_consecutive_errors
                else "DEGRADED"
            )
            now_ms = utc_now_ms()
            self._emit_alert(
                trace_id=trace_id,
                ts_ms=now_ms,
                code=PlatformErrorCode.FEED_OUTAGE,
                coin="ALL",
                message=f"Quote provider raised {type(exc).__name__}: {exc}",
            )
            eq, margin_r = self._compute_shadow_equity_and_margin(self._last_valid_snaps)
            self._persist_checkpoint_only(now_ms)
            self._write_heartbeat(trace_id=trace_id, ts_ms=now_ms, equity_usd=eq, margin_ratio=margin_r)
            return {
                "trace_id": trace_id,
                "status": self._health_status,
                "valid_coins": 0,
                "error": str(exc),
            }

        valid_snaps: dict[str, PairMarketSnapshot] = {}
        tick_summary_by_coin: dict[str, dict[str, Any]] = {}
        dvol_obs: float | None = None
        fallback_ts_ms = utc_now_ms()

        for raw_coin in self.config.coins:
            coin = str(raw_coin).upper()
            snap = snapshots.get(coin)
            if snap is None:
                self._emit_alert(
                    trace_id=trace_id,
                    ts_ms=fallback_ts_ms,
                    code=PlatformErrorCode.UNPAIRED_LEG,
                    coin=coin,
                    message=f"Snapshot missing completely for {coin}",
                )
                with transaction(self.conn):
                    self.conn.execute(
                        """
                        INSERT INTO shadow_ticks
                        (trace_id, ts_observe_ms, coin, guard_status, error_code, provenance)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (trace_id, fallback_ts_ms, coin, "FAIL_CLOSED", PlatformErrorCode.UNPAIRED_LEG.value, PROVENANCE_REAL),
                    )
                continue

            ok, code, msg = self.validate_snapshot(snap)
            if not ok:
                self._emit_alert(trace_id=trace_id, ts_ms=fallback_ts_ms, code=code, coin=coin, message=msg)
                with transaction(self.conn):
                    self.conn.execute(
                        """
                        INSERT INTO shadow_ticks
                        (trace_id, ts_observe_ms, coin, spot_bid, spot_ask, perp_bid, perp_ask,
                         perp_mark, perp_funding_1h, dvol, guard_status, error_code, provenance)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            trace_id,
                            fallback_ts_ms,
                            coin,
                            snap.spot_bid if _is_finite(snap.spot_bid) else None,
                            snap.spot_ask if _is_finite(snap.spot_ask) else None,
                            snap.perp_bid if _is_finite(snap.perp_bid) else None,
                            snap.perp_ask if _is_finite(snap.perp_ask) else None,
                            snap.perp_mark if _is_finite(snap.perp_mark) else None,
                            snap.perp_funding_1h if _is_finite(snap.perp_funding_1h) else None,
                            snap.dvol if _is_finite(snap.dvol) else None,
                            "FAIL_CLOSED",
                            code.value,
                            snap.provenance,
                        ),
                    )
                continue

            # Belt-and-suspenders crossed-book invariant assertion (RB-10)
            if float(snap.spot_bid) >= float(snap.spot_ask) or float(snap.perp_bid) >= float(snap.perp_ask):
                self._crossed_book_bypass_count += 1
                continue

            valid_snaps[coin] = snap
            self._last_valid_snaps[coin] = snap
            if snap.dvol is not None:
                dvol_obs = float(snap.dvol)

            mid_spot = 0.5 * (float(snap.spot_bid) + float(snap.spot_ask))
            mid_perp = 0.5 * (float(snap.perp_bid) + float(snap.perp_ask))
            mark_perp = float(snap.perp_mark) if snap.perp_mark is not None else mid_perp
            basis_mid_bps = ((mid_perp / mid_spot) - 1.0) * 10_000.0
            basis_mark_bps = ((mark_perp / mid_spot) - 1.0) * 10_000.0
            self._max_observed_abs_basis_bps = max(
                self._max_observed_abs_basis_bps,
                abs(basis_mid_bps),
                abs(basis_mark_bps),
            )

            spot_spread_half_bps = (((float(snap.spot_ask) - float(snap.spot_bid)) / mid_spot) * 10_000.0) * 0.5
            perp_slip_buy_bps = (
                estimate_l2_vwap_slippage_bps(
                    snap.perp_l2_asks,
                    notional_usd=self.config.capital_usd * self.config.max_pair_leg_weight,
                    mid_price=mid_perp,
                    side="buy",
                )
                if snap.perp_l2_asks
                else ((((float(snap.perp_ask) - float(snap.perp_bid)) / mid_perp) * 10_000.0) * 0.5)
            )
            perp_slip_sell_bps = (
                estimate_l2_vwap_slippage_bps(
                    snap.perp_l2_bids,
                    notional_usd=self.config.capital_usd * self.config.max_pair_leg_weight,
                    mid_price=mid_perp,
                    side="sell",
                )
                if snap.perp_l2_bids
                else ((((float(snap.perp_ask) - float(snap.perp_bid)) / mid_perp) * 10_000.0) * 0.5)
            )
            est_4leg_rt_bps = (
                2.0 * (self.config.spot_taker_fee_bps + spot_spread_half_bps)
                + 2.0 * self.config.perp_taker_fee_bps
                + perp_slip_buy_bps
                + perp_slip_sell_bps
            )
            self._observed_4leg_cost_sum += est_4leg_rt_bps
            self._observed_4leg_cost_count += 1

            if float(snap.perp_funding_1h) < 0.0:
                self._emit_alert(
                    trace_id=trace_id,
                    ts_ms=int(snap.ts_observe_ms),
                    code=PlatformErrorCode.FUNDING_FLIP,
                    coin=coin,
                    message=f"Negative 1h funding on {coin}: {float(snap.perp_funding_1h) * 8760.0 * 100.0:.2f}% APR",
                )

            with transaction(self.conn):
                self.conn.execute(
                    """
                    INSERT INTO shadow_ticks
                    (trace_id, ts_observe_ms, coin, spot_bid, spot_ask, perp_bid, perp_ask,
                     perp_mark, perp_funding_1h, basis_mid_bps, basis_mark_bps,
                     exec_4leg_cost_bps, dvol, guard_status, error_code, provenance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace_id,
                        int(snap.ts_observe_ms),
                        coin,
                        float(snap.spot_bid),
                        float(snap.spot_ask),
                        float(snap.perp_bid),
                        float(snap.perp_ask),
                        mark_perp,
                        float(snap.perp_funding_1h),
                        basis_mid_bps,
                        basis_mark_bps,
                        est_4leg_rt_bps,
                        snap.dvol,
                        "PASS",
                        PlatformErrorCode.OK.value,
                        snap.provenance,
                    ),
                )

            tick_summary_by_coin[coin] = {
                "spot_mid": round(mid_spot, 6),
                "perp_mid": round(mid_perp, 6),
                "basis_mid_bps": round(basis_mid_bps, 4),
                "basis_mark_bps": round(basis_mark_bps, 4),
                "funding_apr_pct": round(float(snap.perp_funding_1h) * 8760.0 * 100.0, 4),
                "est_4leg_rt_cost_bps": round(est_4leg_rt_bps, 4),
            }

        # RB-10: only derive tick timestamp from VALIDATED snapshots (prevents spoofed 48h clock jump!)
        ts_ms = max((int(s.ts_observe_ms) for s in valid_snaps.values()), default=fallback_ts_ms)
        if valid_snaps:
            if self._start_ts_ms is None:
                self._start_ts_ms = ts_ms
            self._last_ts_ms = max(self._last_ts_ms or ts_ms, ts_ms)

        # RB-08: require configured healthy coin ratio (default 100% of universe) to count healthy tick
        min_needed = max(1, int(math.ceil(len(self.config.coins) * float(self.config.min_healthy_coin_ratio))))
        if not valid_snaps:
            self._consecutive_errors += 1
            self._health_status = (
                "HALTED_FAIL_CLOSED"
                if self._consecutive_errors >= self.config.max_consecutive_errors
                else "DEGRADED"
            )
            # RB-02 / RB-21: even if 0 valid pairs, pass {} (NOT self._last_valid_snaps!) so live
            # raw_snaps (e.g., +150% Hyperliquid perp_mark surge during a full Binance Spot outage)
            # are NEVER shadowed by stale cached snapshots!
            eq_d, margin_d = self._compute_shadow_equity_and_margin({}, snapshots)
            self._min_perp_margin_buffer_ratio = min(self._min_perp_margin_buffer_ratio, margin_d)
            if margin_d < self.config.margin_storm_exit_ratio:
                bucket_sec_d = max(60, int(self.config.kernel_bar_interval_sec))
                cur_bucket_d = int(ts_ms // (bucket_sec_d * 1000))
                self._emit_alert(
                    trace_id=trace_id,
                    ts_ms=ts_ms,
                    code=PlatformErrorCode.MARGIN_STORM,
                    coin="ALL",
                    message=f"Partial-outage margin storm ratio={margin_d:.4f}; force-flattening ALL open pairs",
                )
                for c_open in list(self._positions.keys()):
                    if self._positions[c_open].is_open:
                        self._close_virtual_pair(
                            c_open,
                            {},
                            raw_snaps=snapshots,
                            trace_id=trace_id,
                            ts_ms=ts_ms,
                            reason="MARGIN_STORM_PARTIAL_OUTAGE_EXIT",
                        )
                        self._basis_cooldown_until_bucket[c_open] = cur_bucket_d + 1
                eq_d, margin_d = self._compute_shadow_equity_and_margin({}, snapshots)
            self._persist_checkpoint_only(ts_ms)
            self._write_heartbeat(trace_id=trace_id, ts_ms=ts_ms, equity_usd=eq_d, margin_ratio=margin_d)
            return {
                "trace_id": trace_id,
                "status": self._health_status,
                "valid_coins": 0,
                "equity_usd": round(eq_d, 4),
            }

        if len(valid_snaps) >= min_needed:
            self._consecutive_errors = 0
            self._healthy_ticks += 1
            self._last_good_mono = now_mono
            self._health_status = "HEALTHY"
        else:
            self._health_status = "DEGRADED"

        # 1. Accrue real time-weighted funding on open virtual pairs
        funding_delta = self._accrue_funding(valid_snaps, ts_ms)

        # 2. Intraday fast-path risk supervisor (margin storm across ALL open pairs + basis blowout)
        equity_usd, margin_ratio = self._compute_shadow_equity_and_margin(valid_snaps, snapshots)
        self._min_perp_margin_buffer_ratio = min(self._min_perp_margin_buffer_ratio, margin_ratio)

        bucket_sec = max(60, int(self.config.kernel_bar_interval_sec))
        cur_bucket = int(ts_ms // (bucket_sec * 1000))
        stopped_this_tick: set[str] = set()

        storm_triggered = margin_ratio < self.config.margin_storm_exit_ratio
        if storm_triggered:
            self._emit_alert(
                trace_id=trace_id,
                ts_ms=ts_ms,
                code=PlatformErrorCode.MARGIN_STORM,
                coin="ALL",
                message=(
                    f"Perp margin buffer ratio {margin_ratio:.4f} < storm threshold "
                    f"{self.config.margin_storm_exit_ratio:.2f}; flattening ALL open virtual pairs"
                ),
            )
            # RB-02: flatten ALL open positions including coins missing from valid_snaps!
            for c_open in list(self._positions.keys()):
                if self._positions[c_open].is_open:
                    self._close_virtual_pair(
                        c_open,
                        valid_snaps,
                        raw_snaps=snapshots,
                        trace_id=trace_id,
                        ts_ms=ts_ms,
                        reason="MARGIN_STORM_EXIT",
                    )
                    stopped_this_tick.add(c_open)
                    self._basis_cooldown_until_bucket[c_open] = cur_bucket + 1
        else:
            for coin, snap in valid_snaps.items():
                mid_spot = 0.5 * (float(snap.spot_bid) + float(snap.spot_ask))
                mid_perp = 0.5 * (float(snap.perp_bid) + float(snap.perp_ask))
                mark_perp = float(snap.perp_mark) if snap.perp_mark is not None else mid_perp
                b_mid = ((mid_perp / mid_spot) - 1.0) * 10_000.0
                b_mark = ((mark_perp / mid_spot) - 1.0) * 10_000.0
                if abs(b_mid) >= self.config.basis_stop_bps or abs(b_mark) >= self.config.basis_stop_bps:
                    self._emit_alert(
                        trace_id=trace_id,
                        ts_ms=ts_ms,
                        code=PlatformErrorCode.BASIS_BLOWOUT,
                        coin=coin,
                        message=(
                            f"Intraday basis blowout on {coin}: mid={b_mid:.1f}bp mark={b_mark:.1f}bp "
                            f"(limit={self.config.basis_stop_bps:.1f}bp)"
                        ),
                    )
                    self._close_virtual_pair(
                        coin,
                        valid_snaps,
                        raw_snaps=snapshots,
                        trace_id=trace_id,
                        ts_ms=ts_ms,
                        reason="INTRADAY_BASIS_STOP",
                    )
                    # RB-06: block same-tick & next-bucket re-entry after intraday basis stop!
                    stopped_this_tick.add(coin)
                    self._basis_cooldown_until_bucket[coin] = cur_bucket + 1

        # 3. Kernel decision evaluation on canonical bar bucket (reusing exact W3-DN kernel for G7 parity!)
        if not storm_triggered and (self._last_kernel_bucket is None or cur_bucket > self._last_kernel_bucket):
            self._last_kernel_bucket = cur_bucket
            bars_at_ts: dict[str, CanonicalBar] = {}
            tf_h = bucket_sec / 3600.0
            for coin, snap in valid_snaps.items():
                mid_spot = 0.5 * (float(snap.spot_bid) + float(snap.spot_ask))
                mid_perp = 0.5 * (float(snap.perp_bid) + float(snap.perp_ask))
                s_sym = spot_leg_symbol(coin)
                p_sym = perp_leg_symbol(coin)
                spot_vol = float(snap.spot_bid_qty or 1.0) * mid_spot
                bars_at_ts[s_sym] = CanonicalBar(
                    asset_class=ASSET_CLASS_CRYPTO_SPOT,
                    venue="binance_spot",
                    symbol=s_sym,
                    timeframe_h=tf_h,
                    timeframe_sec=bucket_sec,
                    ts_event_ms=int(ts_ms),
                    ts_avail_ms=int(ts_ms),
                    open=mid_spot,
                    high=float(snap.spot_ask),
                    low=float(snap.spot_bid),
                    close=mid_spot,
                    volume_usd=spot_vol,
                    carry_rate_period=0.0,
                    carry_interval_h=tf_h,
                    spread_bps=((float(snap.spot_ask) - float(snap.spot_bid)) / mid_spot) * 10_000.0,
                    provenance=snap.provenance,
                )
                bars_at_ts[p_sym] = CanonicalBar(
                    asset_class=ASSET_CLASS_CRYPTO_PERP,
                    venue="hyperliquid",
                    symbol=p_sym,
                    timeframe_h=tf_h,
                    timeframe_sec=bucket_sec,
                    ts_event_ms=int(ts_ms),
                    ts_avail_ms=int(ts_ms),
                    open=mid_perp,
                    high=float(snap.perp_ask),
                    low=float(snap.perp_bid),
                    close=mid_perp,
                    volume_usd=float(self.config.capital_usd),
                    carry_rate_period=float(snap.perp_funding_1h) * tf_h,
                    carry_interval_h=tf_h,
                    spread_bps=((float(snap.perp_ask) - float(snap.perp_bid)) / mid_perp) * 10_000.0,
                    provenance=snap.provenance,
                )

            # RB-03: pass exact "margin_ratio" key expected by Web3DeltaNeutralCarryKernel
            ctx: dict[str, Any] = {
                "margin_ratio": margin_ratio,
                "perp_margin_buffer_ratio": margin_ratio,
            }
            if dvol_obs is not None:
                ctx["dvol"] = dvol_obs
            decisions = self.kernel.on_bar_step(ts_ms, bars_at_ts, ctx)

            # RB-08: preserve open gross budget for held coins currently absent from valid_snaps
            locked_absent_spot_w = math.fsum(
                p.weight_spot for c, p in self._positions.items() if p.is_open and c not in valid_snaps
            )
            avail_spot_budget = max(0.0, float(self.config.max_spot_gross) - locked_absent_spot_w)
            proposed_spot_w: dict[str, float] = {}
            proposed_perp_w: dict[str, float] = {}
            proposed_regime: dict[str, str] = {}
            for coin in sorted(valid_snaps.keys()):
                # RB-06: suppress opening on same tick or cooldown bucket after basis stop
                if coin in stopped_this_tick or cur_bucket < self._basis_cooldown_until_bucket.get(coin, -1):
                    proposed_spot_w[coin] = 0.0
                    proposed_perp_w[coin] = 0.0
                    proposed_regime[coin] = "BASIS_STOP_COOLDOWN"
                    continue
                dec_s = decisions.get(spot_leg_symbol(coin))
                dec_p = decisions.get(perp_leg_symbol(coin))
                w_s = float(dec_s.target_weight) if dec_s is not None else 0.0
                w_p = float(dec_p.target_weight) if dec_p is not None else 0.0
                proposed_spot_w[coin] = max(0.0, w_s)
                proposed_perp_w[coin] = w_p
                proposed_regime[coin] = str(dec_s.regime_label) if dec_s is not None else "FLAT"

            tot_prop = math.fsum(proposed_spot_w.values())
            if tot_prop > avail_spot_budget and tot_prop > 1e-9:
                sc = avail_spot_budget / tot_prop
                for coin in proposed_spot_w:
                    proposed_spot_w[coin] *= sc
                    proposed_perp_w[coin] *= sc

            # Close exits first to free cash before opening new pairs (RB-16)
            ordered_coins = sorted(valid_snaps.keys(), key=lambda c: (proposed_spot_w[c] > 1e-9, c))
            for coin in ordered_coins:
                self._apply_atomic_target_pair(
                    coin,
                    proposed_spot_w[coin],
                    proposed_perp_w[coin],
                    valid_snaps,
                    trace_id=trace_id,
                    ts_ms=ts_ms,
                    reason=proposed_regime[coin],
                )

        # 4. Post-tick equity, atomic tamper-evident hash chain + WAL checkpoint & heartbeat
        eq_final, margin_final = self._compute_shadow_equity_and_margin(valid_snaps, snapshots)
        self._min_perp_margin_buffer_ratio = min(self._min_perp_margin_buffer_ratio, margin_final)
        audit_payload = {
            "valid_coins": sorted(valid_snaps.keys()),
            "equity_usd": round(eq_final, 6),
            "funding_delta_usd": round(funding_delta, 6),
            "margin_ratio": round(margin_final, 6),
            "max_abs_delta_weight": round(self._max_abs_delta_weight, 12),
            "ticks": tick_summary_by_coin,
        }
        chain_hash = self._append_audit_chain_and_checkpoint(
            ts_ms=ts_ms, trace_id=trace_id, payload=audit_payload
        )
        self._log_event(
            trace_id=trace_id,
            event="SHADOW_TICK",
            level="INFO",
            payload={"chain_hash": chain_hash, **audit_payload},
        )
        self._write_heartbeat(trace_id=trace_id, ts_ms=ts_ms, equity_usd=eq_final, margin_ratio=margin_final)
        return {
            "trace_id": trace_id,
            "status": self._health_status,
            "ts_ms": ts_ms,
            "valid_coins": len(valid_snaps),
            "equity_usd": round(eq_final, 4),
            "margin_ratio": round(margin_final, 6),
            "chain_hash": chain_hash,
            "coins": tick_summary_by_coin,
        }

    def evaluate_48h_readiness(self) -> ShadowReadinessReport:
        """Compute machine-verifiable >=48h blind-run readiness verdict (never auto-enables live)."""
        elapsed_h = (
            max(0.0, (float(self._last_ts_ms) - float(self._start_ts_ms)) / 3_600_000.0)
            if (self._start_ts_ms is not None and self._last_ts_ms is not None)
            else 0.0
        )
        uptime = (self._healthy_ticks / self._total_ticks) if self._total_ticks > 0 else 0.0
        chain_ok, chain_len = self.verify_audit_chain()
        # RB-07: mark open pairs to cached latest valid market quotes (never stale entry prices!)
        eq_now, _ = self._compute_shadow_equity_and_margin(self._last_valid_snaps)
        mean_4leg = (
            self._observed_4leg_cost_sum / self._observed_4leg_cost_count
            if self._observed_4leg_cost_count > 0
            else 0.0
        )

        blockers: list[str] = []
        hard_fail = False
        if SHADOW_LIVE_EXECUTION_ENABLED:
            blockers.append("LIVE_EXECUTION_ENABLED must remain False")
            hard_fail = True
        if not chain_ok:
            blockers.append("SHA-256 shadow audit chain integrity failed")
            hard_fail = True
        if self._max_abs_delta_weight > 1e-9:
            blockers.append(f"Delta neutrality breached: max_abs_delta_weight={self._max_abs_delta_weight}")
            hard_fail = True
        if self._crossed_book_bypass_count > 0:
            blockers.append(f"Crossed-book bypass detected: {self._crossed_book_bypass_count}")
            hard_fail = True
        if self._health_status == "HALTED_FAIL_CLOSED":
            blockers.append("Monitor halted in FAIL_CLOSED state due to consecutive errors")
            hard_fail = True
        if self._total_ticks > 0 and uptime < self.config.min_uptime_ratio:
            blockers.append(f"Uptime {uptime:.4f} < required {self.config.min_uptime_ratio:.4f}")
            hard_fail = True

        if hard_fail:
            verdict = "FAIL_CLOSED_BLOCKED"
        elif elapsed_h < self.config.required_shadow_hours:
            blockers.append(
                f"Observation elapsed {elapsed_h:.2f}h < required {self.config.required_shadow_hours:.1f}h"
            )
            verdict = "SHADOW_OBSERVATION_IN_PROGRESS"
        else:
            verdict = "READY_FOR_OWNER_REVIEW"

        return ShadowReadinessReport(
            verdict=verdict,
            live_execution_enabled=SHADOW_LIVE_EXECUTION_ENABLED,
            health_status=self._health_status,
            elapsed_hours=round(elapsed_h, 4),
            required_hours=float(self.config.required_shadow_hours),
            total_ticks=self._total_ticks,
            healthy_ticks=self._healthy_ticks,
            uptime_ratio=round(uptime, 6),
            audit_chain_intact=chain_ok,
            audit_chain_length=chain_len,
            max_abs_delta_weight=self._max_abs_delta_weight,
            crossed_book_bypass_count=self._crossed_book_bypass_count,
            max_observed_abs_basis_bps=round(self._max_observed_abs_basis_bps, 4),
            basis_stop_bps=float(self.config.basis_stop_bps),
            min_perp_margin_buffer_ratio=round(self._min_perp_margin_buffer_ratio, 6),
            margin_storm_exit_ratio=float(self.config.margin_storm_exit_ratio),
            mean_observed_4leg_cost_bps=round(mean_4leg, 4),
            backtest_assumed_4leg_cost_bps=float(self.kernel.full_4leg_rt_cost_bps),
            shadow_equity_usd=round(eq_now, 4),
            shadow_net_pnl_usd=round(eq_now - float(self.config.capital_usd), 4),
            shadow_funding_usd=round(self._total_funding_usd, 4),
            shadow_cost_usd=round(self._total_cost_usd, 4),
            alert_counts_by_code=dict(sorted(self._alert_counts.items())),
            alert_suppressed_by_code=dict(sorted(self._alert_suppressed_counts.items())),
            blocking_reasons=blockers,
        )
