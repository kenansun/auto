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
from dataclasses import asdict, dataclass, field
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
SHADOW_STATUS_ACTIVE: Final[str] = "ACTIVE_EXECUTION"
SHADOW_STATUS_VACUOUS: Final[str] = "SHADOW_VACUOUS_ZERO_TRADES"
_MIN_VALID_EPOCH_MS: Final[int] = 1_577_836_800_000  # 2020-01-01T00:00:00Z
_MAX_VALID_EPOCH_MS: Final[int] = 4_102_444_800_000  # 2100-01-01T00:00:00Z

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
    if isinstance(x, bool):
        return False
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0


def _is_finite(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _percentile_nearest_rank(sorted_vals: Sequence[float], q: float) -> float:
    """Deterministic nearest-rank percentile on a pre-sorted finite sequence."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(float(q) * len(sorted_vals)) - 1))
    return float(sorted_vals[idx])


def _compute_4leg_cost_distribution(values: Sequence[float], threshold_bps: float) -> dict[str, Any]:
    """Compute finite 4-leg cost tail distribution (mean/p95/p99/max and budget-breach spike stats)."""
    finite_vals = sorted(float(v) for v in values if _is_finite(v))
    cnt = len(finite_vals)
    if cnt == 0:
        return {
            "count": 0,
            "mean_bps": 0.0,
            "p95_bps": 0.0,
            "p99_bps": 0.0,
            "max_bps": 0.0,
            "spike_count": 0,
            "spike_rate": 0.0,
        }
    mean_v = math.fsum(finite_vals) / cnt
    p95_v = _percentile_nearest_rank(finite_vals, 0.95)
    p99_v = _percentile_nearest_rank(finite_vals, 0.99)
    max_v = float(finite_vals[-1])
    thr = float(threshold_bps) if _is_pos_finite(threshold_bps) else 36.0
    spikes = sum(1 for v in finite_vals if v > thr)
    return {
        "count": cnt,
        "mean_bps": round(mean_v, 4),
        "p95_bps": round(p95_v, 4),
        "p99_bps": round(p99_v, 4),
        "max_bps": round(max_v, 4),
        "spike_count": int(spikes),
        "spike_rate": round(spikes / cnt, 6),
    }


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
    p16_heartbeat_path: Path | None = None
    readiness_report_path: Path | None = None
    observe_only_coins: tuple[str, ...] = ()
    service_name: str = "w3_dn_shadow_monitor"
    enable_prewarm: bool = False
    prewarm_bars: int = 12


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

    verdict: str  # "READY_FOR_OWNER_REVIEW" | "SHADOW_OBSERVATION_IN_PROGRESS" | "SHADOW_VACUOUS_ZERO_TRADES" | "FAIL_CLOSED_BLOCKED"
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
    observe_only_coin_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    executed_trades: int = 0
    prewarmed_bars: int = 0
    non_vacuous: bool = False
    non_vacuous_parity: bool = False
    shadow_status: str = SHADOW_STATUS_VACUOUS
    parity_depth: str = SHADOW_STATUS_VACUOUS
    p95_observed_4leg_cost_bps: float = 0.0
    p99_observed_4leg_cost_bps: float = 0.0
    max_observed_4leg_cost_bps: float = 0.0
    tail_cost_spike_threshold_bps: float = 36.0
    tail_cost_spike_count: int = 0
    tail_cost_spike_rate: float = 0.0
    primary_coin_cost_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

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
                raw_bid = payload["bidPrice"]
                raw_ask = payload["askPrice"]
                raw_bid_qty = payload.get("bidQty", 0.0)
                raw_ask_qty = payload.get("askQty", 0.0)
                if any(isinstance(v, bool) for v in (raw_bid, raw_ask, raw_bid_qty, raw_ask_qty)):
                    continue
                bid = float(raw_bid)
                ask = float(raw_ask)
                bid_qty = float(raw_bid_qty)
                ask_qty = float(raw_ask_qty)
                # RB-16 + STD-46: explicit math.isfinite guard before ordering comparison; fail over on invalid/crossed BBO
                if not (math.isfinite(bid) and math.isfinite(ask) and bid > 0.0 and ask > 0.0 and bid < ask):
                    logger.warning("Spot waterfall endpoint %s returned invalid BBO for %s: %s/%s", base, sym, bid, ask)
                    continue
                if not (math.isfinite(bid_qty) and math.isfinite(ask_qty) and bid_qty > 0.0 and ask_qty > 0.0):
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
        prewarm_frames: Sequence[Any] | None = None,
    ) -> None:
        if SHADOW_LIVE_EXECUTION_ENABLED:
            raise RuntimeError(f"{PlatformErrorCode.LIVE_SWITCH_BLOCKED.value}: live execution forbidden")

        self.config = config
        self._primary_coins: tuple[str, ...] = tuple(dict.fromkeys(str(c).strip().upper() for c in config.coins if str(c).strip()))
        self._observe_only_coins: tuple[str, ...] = tuple(
            dict.fromkeys(str(c).strip().upper() for c in config.observe_only_coins if str(c).strip())
        )
        if not self._primary_coins:
            raise ValueError("ShadowMonitorConfig.coins must contain at least one primary coin")
        overlap = set(self._primary_coins).intersection(self._observe_only_coins)
        if overlap:
            raise ValueError(
                f"observe_only_coins must NOT overlap primary coins: {sorted(overlap)}"
            )
        self.state_dir = Path(config.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "w3_dn_shadow.sqlite"
        self.log_path = self.state_dir / "w3_dn_shadow.jsonl"
        self.heartbeat_path = (
            Path(config.p16_heartbeat_path)
            if config.p16_heartbeat_path is not None
            else (self.state_dir / "w3_dn_shadow_heartbeat.json")
        )
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self.readiness_report_path: Path | None = (
            Path(config.readiness_report_path) if config.readiness_report_path is not None else None
        )
        if self.readiness_report_path is not None:
            self.readiness_report_path.parent.mkdir(parents=True, exist_ok=True)

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
        self._observe_only_stats: dict[str, dict[str, float]] = {}
        self._last_kernel_bucket: int | None = None
        self._basis_cooldown_until_bucket: dict[str, int] = {}
        self._last_chain_hash: str = self._load_last_chain_hash()

        self._alert_state: dict[tuple[str, str], tuple[float, int]] = {}
        self._alert_counts: dict[str, int] = {}
        self._alert_suppressed_counts: dict[str, int] = {}
        self._health_status: str = "INITIALIZING"
        self._executed_trades: int = 0
        self._prewarmed_bars: int = 0
        self._last_prewarm_bucket: int | None = None

        self._hydrate_checkpoint()

        if prewarm_frames is not None and self._prewarmed_bars == 0:
            cap = int(self.config.prewarm_bars) if (self.config.enable_prewarm and int(self.config.prewarm_bars) > 0) else None
            self.prewarm_from_snapshots(prewarm_frames, max_bars=cap)

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
            "observe_only_stats": {
                c: dict(stats) for c, stats in self._observe_only_stats.items()
            },
            "last_kernel_bucket": self._last_kernel_bucket,
            "basis_cooldown_until_bucket": dict(self._basis_cooldown_until_bucket),
            "alert_counts": dict(self._alert_counts),
            "alert_suppressed_counts": dict(self._alert_suppressed_counts),
            "health_status": self._health_status,
            "executed_trades": self._executed_trades,
            "prewarmed_bars": self._prewarmed_bars,
            "last_prewarm_bucket": self._last_prewarm_bucket,
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
        self._observe_only_stats = {
            str(c): {str(k): float(v) for k, v in (d or {}).items()}
            for c, d in (st.get("observe_only_stats") or {}).items()
        }
        self._last_kernel_bucket = st.get("last_kernel_bucket")
        self._basis_cooldown_until_bucket = {
            str(k): int(v) for k, v in (st.get("basis_cooldown_until_bucket") or {}).items()
        }
        self._alert_counts = {str(k): int(v) for k, v in (st.get("alert_counts") or {}).items()}
        self._alert_suppressed_counts = {
            str(k): int(v) for k, v in (st.get("alert_suppressed_counts") or {}).items()
        }
        self._health_status = str(st.get("health_status", "INITIALIZING"))
        if "executed_trades" in st and st["executed_trades"] is not None:
            self._executed_trades = max(0, int(st["executed_trades"]))
        else:
            cnt_row = self.conn.fetch_one(
                "SELECT COUNT(*) AS cnt FROM shadow_events WHERE action IN ('VIRTUAL_OPEN_PAIR', 'VIRTUAL_CLOSE_PAIR')"
            )
            self._executed_trades = int(cnt_row["cnt"]) if cnt_row is not None else 0
        self._prewarmed_bars = max(0, int(st.get("prewarmed_bars", 0)))
        raw_lpw = st.get("last_prewarm_bucket")
        self._last_prewarm_bucket = int(raw_lpw) if raw_lpw is not None else None
        if isinstance(st.get("kernel_state"), dict):
            self.kernel.restore_state(st["kernel_state"])

        # RB-D06: Fail closed if an open virtual pair position was moved out of primary coins (e.g. into observe_only_coins)
        orphans = sorted(
            c for c, pos in self._positions.items() if pos.is_open and c not in self._primary_coins
        )
        if orphans:
            raise ValueError(
                f"Orphaned open shadow positions not in primary coins {list(self._primary_coins)}: {orphans}. "
                "Close open positions or start a fresh state_dir before moving coins to --observe-only-coins."
            )

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

        # RB-D03: strictly validate epoch timestamps against None/bool/NaN/Inf/non-positive BEFORE int()
        for ts_name, ts_val in (
            ("ts_observe_ms", snap.ts_observe_ms),
            ("spot_ts_ms", snap.spot_ts_ms),
            ("perp_ts_ms", snap.perp_ts_ms),
        ):
            if (
                ts_val is None
                or isinstance(ts_val, bool)
                or not isinstance(ts_val, (int, float))
                or not math.isfinite(float(ts_val))
                or float(ts_val) <= 0.0
            ):
                return (
                    False,
                    PlatformErrorCode.NON_FINITE_DATA,
                    f"Invalid timestamp {ts_name}={ts_val!r} for {snap.coin}",
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

    def _build_kernel_bars(
        self,
        valid_snaps: dict[str, PairMarketSnapshot],
        ts_ms: int,
    ) -> dict[str, CanonicalBar]:
        """Construct canonical paired {COIN}_SPOT and {COIN}_PERP bars shared by live ticks and pre-warming."""
        bucket_sec = max(60, int(self.config.kernel_bar_interval_sec))
        tf_h = bucket_sec / 3600.0
        bars_at_ts: dict[str, CanonicalBar] = {}
        for coin, snap in valid_snaps.items():
            mid_spot = 0.5 * (float(snap.spot_bid) + float(snap.spot_ask))
            mid_perp = 0.5 * (float(snap.perp_bid) + float(snap.perp_ask))
            s_sym = spot_leg_symbol(coin)
            p_sym = perp_leg_symbol(coin)
            spot_vol = float(snap.spot_bid_qty) * mid_spot if snap.spot_bid_qty is not None else mid_spot
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
        return bars_at_ts

    def _validate_prewarm_snapshot(self, snap: PairMarketSnapshot) -> tuple[bool, PlatformErrorCode, str]:
        """Validate a historical pre-warming snapshot without rejecting past timestamps against wall clock."""
        spot_present = snap.spot_bid is not None and snap.spot_ask is not None and snap.spot_ts_ms is not None
        perp_present = (
            snap.perp_bid is not None
            and snap.perp_ask is not None
            and snap.perp_ts_ms is not None
            and snap.perp_funding_1h is not None
        )
        if not (spot_present and perp_present):
            return False, PlatformErrorCode.UNPAIRED_LEG, f"Missing spot or perp leg for prewarm {snap.coin}"
        if not all(_is_pos_finite(x) for x in (snap.spot_bid, snap.spot_ask, snap.perp_bid, snap.perp_ask)):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Non-finite/non-positive prewarm BBO for {snap.coin}"
        if not _is_finite(snap.perp_funding_1h):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Non-finite prewarm funding rate for {snap.coin}"
        if snap.spot_bid_qty is not None and not _is_pos_finite(snap.spot_bid_qty):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Invalid prewarm spot_bid_qty for {snap.coin}"
        if snap.spot_ask_qty is not None and not _is_pos_finite(snap.spot_ask_qty):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Invalid prewarm spot_ask_qty for {snap.coin}"
        if snap.perp_mark is not None and not _is_pos_finite(snap.perp_mark):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Invalid prewarm perp_mark for {snap.coin}"
        if snap.perp_oracle is not None and not _is_pos_finite(snap.perp_oracle):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Invalid prewarm perp_oracle for {snap.coin}"
        if snap.dvol is not None and not _is_pos_finite(snap.dvol):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Invalid prewarm dvol for {snap.coin}"

        s_bid = float(snap.spot_bid)
        s_ask = float(snap.spot_ask)
        p_bid = float(snap.perp_bid)
        p_ask = float(snap.perp_ask)
        if not (math.isfinite(s_bid) and math.isfinite(s_ask) and math.isfinite(p_bid) and math.isfinite(p_ask)):
            return False, PlatformErrorCode.NON_FINITE_DATA, f"Non-finite prewarm quotes for {snap.coin}"
        if s_bid >= s_ask or p_bid >= p_ask:
            return False, PlatformErrorCode.CROSSED_BOOK, f"Crossed book in prewarm snapshot for {snap.coin}"

        for ts_name, ts_val in (
            ("ts_observe_ms", snap.ts_observe_ms),
            ("spot_ts_ms", snap.spot_ts_ms),
            ("perp_ts_ms", snap.perp_ts_ms),
        ):
            if (
                ts_val is None
                or isinstance(ts_val, bool)
                or not isinstance(ts_val, (int, float))
                or not math.isfinite(float(ts_val))
                or int(ts_val) < _MIN_VALID_EPOCH_MS
                or int(ts_val) > _MAX_VALID_EPOCH_MS
            ):
                return False, PlatformErrorCode.NON_FINITE_DATA, f"Invalid prewarm {ts_name} for {snap.coin}"

        obs_ms = int(snap.ts_observe_ms)
        spot_ts = int(snap.spot_ts_ms)
        perp_ts = int(snap.perp_ts_ms)
        if self.config.enforce_wall_clock_bounds:
            wall_ms = utc_now_ms()
            if (obs_ms - wall_ms) > self.config.max_future_skew_ms:
                return False, PlatformErrorCode.CLOCK_SKEW, f"Future prewarm ts_observe_ms for {snap.coin}"
        if (spot_ts - obs_ms) > self.config.max_future_skew_ms or (perp_ts - obs_ms) > self.config.max_future_skew_ms:
            return False, PlatformErrorCode.CLOCK_SKEW, f"Future prewarm quote timestamp for {snap.coin}"
        if (obs_ms - spot_ts) > self.config.max_stale_ms or (obs_ms - perp_ts) > self.config.max_stale_ms:
            return False, PlatformErrorCode.STALE_QUOTE, f"Stale prewarm leg quote for {snap.coin}"
        if abs(spot_ts - perp_ts) > self.config.max_leg_desync_ms:
            return False, PlatformErrorCode.LEG_DESYNC, f"Prewarm leg desync for {snap.coin}"
        return True, PlatformErrorCode.OK, "OK"

    def _coerce_prewarm_snapshot(self, raw: Any, *, fallback_coin: str | None = None) -> PairMarketSnapshot | None:
        """Coerce a PairMarketSnapshot or serialized/minimal dict into a PairMarketSnapshot (fail-closed)."""
        if isinstance(raw, PairMarketSnapshot):
            norm_coin = _normalize_coin_symbol(raw.coin or (fallback_coin or ""))
            if not norm_coin:
                return None
            if norm_coin == raw.coin:
                return raw
            return PairMarketSnapshot(
                coin=norm_coin,
                ts_observe_ms=raw.ts_observe_ms,
                spot_bid=raw.spot_bid,
                spot_ask=raw.spot_ask,
                spot_bid_qty=raw.spot_bid_qty,
                spot_ask_qty=raw.spot_ask_qty,
                spot_ts_ms=raw.spot_ts_ms,
                perp_bid=raw.perp_bid,
                perp_ask=raw.perp_ask,
                perp_mark=raw.perp_mark,
                perp_oracle=raw.perp_oracle,
                perp_funding_1h=raw.perp_funding_1h,
                perp_ts_ms=raw.perp_ts_ms,
                perp_l2_bids=raw.perp_l2_bids,
                perp_l2_asks=raw.perp_l2_asks,
                dvol=raw.dvol,
                spot_source=raw.spot_source,
                perp_source=raw.perp_source,
                provenance=raw.provenance,
            )
        if not isinstance(raw, dict):
            return None

        coin_raw = raw.get("coin") or raw.get("symbol") or fallback_coin
        if not coin_raw or not isinstance(coin_raw, str):
            return None
        coin = _normalize_coin_symbol(coin_raw)
        if not coin:
            return None

        ts_raw = next(
            (raw.get(k) for k in ("ts_observe_ms", "ts_ms", "ts_event_ms", "timestamp_ms") if raw.get(k) is not None),
            None,
        )
        if ts_raw is None or isinstance(ts_raw, bool) or not _is_pos_finite(ts_raw):
            return None
        obs_ms = int(ts_raw)

        s_bid: float | None = None
        s_ask: float | None = None
        if "spot_bid" in raw or "spot_ask" in raw:
            if not (_is_pos_finite(raw.get("spot_bid")) and _is_pos_finite(raw.get("spot_ask"))):
                return None
            s_bid = float(raw["spot_bid"])
            s_ask = float(raw["spot_ask"])
        else:
            s_mid_raw = next((raw.get(k) for k in ("spot_mid", "spot_close", "spot_price") if raw.get(k) is not None), None)
            if not _is_pos_finite(s_mid_raw):
                return None
            sm = float(s_mid_raw)
            half_s = sm * 0.00005
            s_bid = sm - half_s
            s_ask = sm + half_s

        p_bid: float | None = None
        p_ask: float | None = None
        if "perp_bid" in raw or "perp_ask" in raw:
            if not (_is_pos_finite(raw.get("perp_bid")) and _is_pos_finite(raw.get("perp_ask"))):
                return None
            p_bid = float(raw["perp_bid"])
            p_ask = float(raw["perp_ask"])
        else:
            p_mid_raw = next((raw.get(k) for k in ("perp_mid", "perp_close", "perp_price") if raw.get(k) is not None), None)
            if not _is_pos_finite(p_mid_raw):
                return None
            pm = float(p_mid_raw)
            half_p = pm * 0.00005
            p_bid = pm - half_p
            p_ask = pm + half_p

        fund_raw = next(
            (raw.get(k) for k in ("perp_funding_1h", "funding_1h", "funding_rate_1h") if raw.get(k) is not None),
            None,
        )
        if fund_raw is None or not _is_finite(fund_raw):
            return None

        spot_ts_raw = raw.get("spot_ts_ms")
        perp_ts_raw = raw.get("perp_ts_ms")
        spot_ts_ms = int(spot_ts_raw) if (spot_ts_raw is not None and not isinstance(spot_ts_raw, bool) and _is_pos_finite(spot_ts_raw)) else obs_ms
        perp_ts_ms = int(perp_ts_raw) if (perp_ts_raw is not None and not isinstance(perp_ts_raw, bool) and _is_pos_finite(perp_ts_raw)) else obs_ms

        return PairMarketSnapshot(
            coin=coin,
            ts_observe_ms=obs_ms,
            spot_bid=s_bid,
            spot_ask=s_ask,
            spot_bid_qty=raw.get("spot_bid_qty"),
            spot_ask_qty=raw.get("spot_ask_qty"),
            spot_ts_ms=spot_ts_ms,
            perp_bid=p_bid,
            perp_ask=p_ask,
            perp_mark=raw.get("perp_mark"),
            perp_oracle=raw.get("perp_oracle"),
            perp_funding_1h=float(fund_raw),
            perp_ts_ms=perp_ts_ms,
            dvol=raw.get("dvol"),
            spot_source=str(raw.get("spot_source", "prewarm")),
            perp_source=str(raw.get("perp_source", "prewarm")),
            provenance=str(raw.get("provenance", PROVENANCE_REAL)),
        )

    def _is_kernel_warm_ready(self) -> bool:
        need = int(self.kernel.warmup_bars)
        if self._prewarmed_bars >= need:
            return True
        st_map = self.kernel.export_state().get("states", {})
        if isinstance(st_map, dict):
            return any(
                isinstance(s, dict) and int(s.get("bar_count", 0)) >= need
                for s in st_map.values()
            )
        return False

    def prewarm_from_bars(
        self,
        bar_frames: Sequence[dict[str, CanonicalBar] | tuple[int, dict[str, CanonicalBar]]],
        *,
        max_bars: int | None = None,
    ) -> dict[str, Any]:
        """Read-only historical CanonicalBar pre-warming (ZERO virtual trades, ticks, or audit-chain mutations)."""
        if SHADOW_LIVE_EXECUTION_ENABLED:
            raise RuntimeError(f"{PlatformErrorCode.LIVE_SWITCH_BLOCKED.value}: live execution forbidden")
        if isinstance(bar_frames, (str, bytes, dict)):
            raise TypeError("bar_frames must be a sequence of CanonicalBar frame mappings")
        if max_bars is not None and (isinstance(max_bars, bool) or int(max_bars) < 0):
            raise ValueError("max_bars must be a non-negative integer")

        bucket_sec = max(60, int(self.config.kernel_bar_interval_sec))
        bucket_ms = bucket_sec * 1000
        primary_allowed_syms = {
            sym
            for coin in self._primary_coins
            for sym in (spot_leg_symbol(coin), perp_leg_symbol(coin))
        }
        bucket_bars: dict[int, tuple[int, dict[str, CanonicalBar]]] = {}
        rejected_frames = 0

        for item in bar_frames:
            explicit_ts: int | None = None
            raw_map: Any = item
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                if isinstance(item[0], bool) or not _is_pos_finite(item[0]):
                    rejected_frames += 1
                    continue
                explicit_ts = int(item[0])
                raw_map = item[1]
            if not isinstance(raw_map, dict) or not raw_map:
                rejected_frames += 1
                continue

            clean_map: dict[str, CanonicalBar] = {}
            frame_ts: int | None = explicit_ts
            invalid_found = False
            for sym, bar in raw_map.items():
                if not isinstance(bar, CanonicalBar):
                    invalid_found = True
                    continue
                sym_u = str(sym).strip().upper()
                if sym_u not in primary_allowed_syms:
                    continue
                if not (
                    _is_pos_finite(bar.open)
                    and _is_pos_finite(bar.high)
                    and _is_pos_finite(bar.low)
                    and _is_pos_finite(bar.close)
                    and float(bar.high) >= float(bar.low)
                    and _is_finite(bar.carry_rate_period)
                    and not isinstance(bar.ts_event_ms, bool)
                    and _MIN_VALID_EPOCH_MS <= int(bar.ts_event_ms) <= _MAX_VALID_EPOCH_MS
                ):
                    invalid_found = True
                    continue
                clean_map[sym_u] = bar
                if frame_ts is None or int(bar.ts_event_ms) > frame_ts:
                    frame_ts = int(bar.ts_event_ms)

            if invalid_found and not clean_map:
                rejected_frames += 1
                continue
            if not clean_map or frame_ts is None:
                rejected_frames += 1
                continue
            if not (_MIN_VALID_EPOCH_MS <= frame_ts <= _MAX_VALID_EPOCH_MS):
                rejected_frames += 1
                continue
            if self.config.enforce_wall_clock_bounds and (frame_ts - utc_now_ms()) > self.config.max_future_skew_ms:
                rejected_frames += 1
                continue
            b_idx = int(frame_ts // bucket_ms)
            prev = bucket_bars.get(b_idx)
            if prev is None or frame_ts >= prev[0]:
                bucket_bars[b_idx] = (frame_ts, clean_map)

        sorted_buckets = sorted(bucket_bars.keys())
        if max_bars is not None:
            limit = int(max_bars)
            sorted_buckets = sorted_buckets[-limit:] if limit > 0 else []

        live_start_bucket = int(self._start_ts_ms // bucket_ms) if self._start_ts_ms is not None else None
        added_bars = 0
        last_prewarm_ts_ms: int | None = None
        for b_idx in sorted_buckets:
            if self._last_prewarm_bucket is not None and b_idx <= self._last_prewarm_bucket:
                continue
            if live_start_bucket is not None and b_idx >= live_start_bucket:
                continue
            b_ts_ms, b_map = bucket_bars[b_idx]
            ctx: dict[str, Any] = {"margin_ratio": 1.0, "perp_margin_buffer_ratio": 1.0}
            _ = self.kernel.on_bar_step(b_ts_ms, b_map, ctx)
            self._prewarmed_bars += 1
            self._last_prewarm_bucket = b_idx
            added_bars += 1
            last_prewarm_ts_ms = b_ts_ms

        if added_bars > 0 and last_prewarm_ts_ms is not None:
            self._persist_checkpoint_only(last_prewarm_ts_ms)
            self._log_event(
                trace_id=f"w3dn-prewarm-{uuid.uuid4().hex[:10]}",
                event="SHADOW_PREWARM",
                level="INFO",
                payload={
                    "source": "canonical_bars",
                    "prewarmed_bars_added": added_bars,
                    "prewarmed_bars_total": self._prewarmed_bars,
                    "rejected_frames": rejected_frames,
                    "kernel_warmup_bars": int(self.kernel.warmup_bars),
                    "last_prewarm_bucket": self._last_prewarm_bucket,
                },
            )

        return {
            "prewarmed_bars_added": added_bars,
            "prewarmed_bars_total": self._prewarmed_bars,
            "rejected_frames": rejected_frames,
            "kernel_warmup_bars": int(self.kernel.warmup_bars),
            "kernel_warm_ready": self._is_kernel_warm_ready(),
            "executed_trades": self._executed_trades,
            "total_ticks": self._total_ticks,
            "live_execution_enabled": SHADOW_LIVE_EXECUTION_ENABLED,
        }

    def prewarm_from_snapshots(
        self,
        frames: Sequence[Any],
        *,
        max_bars: int | None = None,
    ) -> dict[str, Any]:
        """Read-only historical snapshot pre-warming closing the 144h-in-48h zero-trade warmup gap.

        Accepts:
          - Sequence[dict[str, PairMarketSnapshot]] (ReplayQuoteProvider format)
          - Sequence[dict[str, dict]] (coin-keyed snapshot dicts)
          - Flat Sequence[PairMarketSnapshot | dict]
          - Sequence[dict[str, CanonicalBar]] (delegated to `prewarm_from_bars`)
        Feeds kernel state causally without opening virtual positions, counting live ticks,
        modifying cash/audit chain, or advancing `_start_ts_ms`/`_last_kernel_bucket`.
        """
        if SHADOW_LIVE_EXECUTION_ENABLED:
            raise RuntimeError(f"{PlatformErrorCode.LIVE_SWITCH_BLOCKED.value}: live execution forbidden")
        if isinstance(frames, (str, bytes, dict)):
            raise TypeError("frames must be a sequence of historical snapshot frames")
        if max_bars is not None and (isinstance(max_bars, bool) or int(max_bars) < 0):
            raise ValueError("max_bars must be a non-negative integer")

        frame_list = list(frames)
        if frame_list:
            first = frame_list[0]
            if isinstance(first, dict) and first and all(isinstance(v, CanonicalBar) for v in first.values()):
                return self.prewarm_from_bars(frame_list, max_bars=max_bars)
            if (
                isinstance(first, tuple)
                and len(first) == 2
                and isinstance(first[1], dict)
                and first[1]
                and all(isinstance(v, CanonicalBar) for v in first[1].values())
            ):
                return self.prewarm_from_bars(frame_list, max_bars=max_bars)

        bucket_sec = max(60, int(self.config.kernel_bar_interval_sec))
        bucket_ms = bucket_sec * 1000
        bucket_map: dict[int, dict[str, PairMarketSnapshot]] = {}
        rejected_frames = 0
        _single_snap_keys = frozenset(
            {"coin", "symbol", "ts_observe_ms", "ts_ms", "ts_event_ms", "spot_bid", "spot_mid", "spot_close"}
        )

        def _ingest_one(raw_obj: Any, fallback_coin: str | None = None) -> None:
            nonlocal rejected_frames
            snap = self._coerce_prewarm_snapshot(raw_obj, fallback_coin=fallback_coin)
            if snap is None:
                rejected_frames += 1
                return
            coin = _normalize_coin_symbol(snap.coin)
            if coin not in self._primary_coins:
                return
            ok, _code, _msg = self._validate_prewarm_snapshot(snap)
            if not ok:
                rejected_frames += 1
                return
            b_idx = int(snap.ts_observe_ms) // bucket_ms
            b_snaps = bucket_map.setdefault(b_idx, {})
            prev = b_snaps.get(coin)
            if prev is None or int(snap.ts_observe_ms) >= int(prev.ts_observe_ms):
                b_snaps[coin] = snap

        for item in frame_list:
            if isinstance(item, PairMarketSnapshot):
                _ingest_one(item)
            elif isinstance(item, dict):
                if not item:
                    rejected_frames += 1
                    continue
                is_coin_map = any(isinstance(v, (PairMarketSnapshot, dict)) for v in item.values())
                if (not is_coin_map) and any(k in item for k in _single_snap_keys):
                    _ingest_one(item)
                else:
                    for raw_k, raw_v in item.items():
                        _ingest_one(raw_v, fallback_coin=str(raw_k))
            else:
                rejected_frames += 1

        sorted_buckets = sorted(bucket_map.keys())
        if max_bars is not None:
            limit = int(max_bars)
            sorted_buckets = sorted_buckets[-limit:] if limit > 0 else []

        live_start_bucket = int(self._start_ts_ms // bucket_ms) if self._start_ts_ms is not None else None
        added_bars = 0
        last_prewarm_ts_ms: int | None = None
        for b_idx in sorted_buckets:
            if self._last_prewarm_bucket is not None and b_idx <= self._last_prewarm_bucket:
                continue
            if live_start_bucket is not None and b_idx >= live_start_bucket:
                continue
            b_snaps = bucket_map[b_idx]
            if not b_snaps:
                continue
            b_ts_ms = max(int(s.ts_observe_ms) for s in b_snaps.values())
            bars_at_ts = self._build_kernel_bars(b_snaps, b_ts_ms)
            ctx: dict[str, Any] = {"margin_ratio": 1.0, "perp_margin_buffer_ratio": 1.0}
            dvol_vals = [float(s.dvol) for s in b_snaps.values() if s.dvol is not None and _is_pos_finite(s.dvol)]
            if dvol_vals:
                ctx["dvol"] = dvol_vals[-1]
            # Read-only kernel warm-up: discard trade decisions (zero virtual executions / zero live mutations)
            _ = self.kernel.on_bar_step(b_ts_ms, bars_at_ts, ctx)
            self._prewarmed_bars += 1
            self._last_prewarm_bucket = b_idx
            added_bars += 1
            last_prewarm_ts_ms = b_ts_ms

        if added_bars > 0 and last_prewarm_ts_ms is not None:
            self._persist_checkpoint_only(last_prewarm_ts_ms)
            self._log_event(
                trace_id=f"w3dn-prewarm-{uuid.uuid4().hex[:10]}",
                event="SHADOW_PREWARM",
                level="INFO",
                payload={
                    "source": "snapshots",
                    "prewarmed_bars_added": added_bars,
                    "prewarmed_bars_total": self._prewarmed_bars,
                    "rejected_frames": rejected_frames,
                    "kernel_warmup_bars": int(self.kernel.warmup_bars),
                    "last_prewarm_bucket": self._last_prewarm_bucket,
                },
            )

        return {
            "prewarmed_bars_added": added_bars,
            "prewarmed_bars_total": self._prewarmed_bars,
            "rejected_frames": rejected_frames,
            "kernel_warmup_bars": int(self.kernel.warmup_bars),
            "kernel_warm_ready": self._is_kernel_warm_ready(),
            "executed_trades": self._executed_trades,
            "total_ticks": self._total_ticks,
            "live_execution_enabled": SHADOW_LIVE_EXECUTION_ENABLED,
        }

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
        self._executed_trades += 1

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
        self._executed_trades += 1

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
            "executed_trades": self._executed_trades,
            "prewarmed_bars": self._prewarmed_bars,
            "non_vacuous": self._executed_trades > 0,
            "non_vacuous_parity": self._executed_trades > 0,
            "shadow_status": SHADOW_STATUS_ACTIVE if self._executed_trades > 0 else SHADOW_STATUS_VACUOUS,
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

    def _record_observe_only_ticks(
        self,
        snapshots: dict[str, PairMarketSnapshot],
        *,
        trace_id: str,
        fallback_ts_ms: int,
    ) -> dict[str, dict[str, Any]]:
        """Record L2 slippage/basis/funding diagnostics for observe-only coins (ZERO virtual positions or uptime impact)."""
        out: dict[str, dict[str, Any]] = {}
        for coin in self._observe_only_coins:
            stats = self._observe_only_stats.setdefault(
                coin,
                {"count": 0.0, "fail_count": 0.0, "cost_sum_bps": 0.0, "max_abs_basis_bps": 0.0, "last_funding_apr_pct": 0.0},
            )
            snap = snapshots.get(coin)
            if snap is None:
                stats["fail_count"] += 1.0
                out[coin] = {"mode": "OBSERVE_ONLY", "status": "MISSING"}
                continue
            ok, code, _msg = self.validate_snapshot(snap)
            if not ok:
                stats["fail_count"] += 1.0
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
                            "OBSERVE_ONLY_FAIL",
                            code.value,
                            snap.provenance,
                        ),
                    )
                out[coin] = {"mode": "OBSERVE_ONLY", "status": "FAIL_CLOSED", "error_code": code.value}
                continue

            mid_spot = 0.5 * (float(snap.spot_bid) + float(snap.spot_ask))
            mid_perp = 0.5 * (float(snap.perp_bid) + float(snap.perp_ask))
            mark_perp = float(snap.perp_mark) if snap.perp_mark is not None else mid_perp
            basis_mid_bps = ((mid_perp / mid_spot) - 1.0) * 10_000.0
            basis_mark_bps = ((mark_perp / mid_spot) - 1.0) * 10_000.0
            spot_spread_half_bps = (((float(snap.spot_ask) - float(snap.spot_bid)) / mid_spot) * 10_000.0) * 0.5
            notional = self.config.capital_usd * self.config.max_pair_leg_weight
            perp_slip_buy_bps = (
                estimate_l2_vwap_slippage_bps(snap.perp_l2_asks, notional_usd=notional, mid_price=mid_perp, side="buy")
                if snap.perp_l2_asks
                else ((((float(snap.perp_ask) - float(snap.perp_bid)) / mid_perp) * 10_000.0) * 0.5)
            )
            perp_slip_sell_bps = (
                estimate_l2_vwap_slippage_bps(snap.perp_l2_bids, notional_usd=notional, mid_price=mid_perp, side="sell")
                if snap.perp_l2_bids
                else ((((float(snap.perp_ask) - float(snap.perp_bid)) / mid_perp) * 10_000.0) * 0.5)
            )
            est_4leg_rt_bps = (
                2.0 * (self.config.spot_taker_fee_bps + spot_spread_half_bps)
                + 2.0 * self.config.perp_taker_fee_bps
                + perp_slip_buy_bps
                + perp_slip_sell_bps
            )
            funding_apr = float(snap.perp_funding_1h) * 8760.0 * 100.0
            stats["count"] += 1.0
            stats["cost_sum_bps"] += est_4leg_rt_bps
            stats["max_abs_basis_bps"] = max(stats["max_abs_basis_bps"], abs(basis_mid_bps), abs(basis_mark_bps))
            stats["last_funding_apr_pct"] = funding_apr

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
                        "OBSERVE_ONLY",
                        PlatformErrorCode.OK.value,
                        snap.provenance,
                    ),
                )
            out[coin] = {
                "mode": "OBSERVE_ONLY",
                "spot_mid": round(mid_spot, 6),
                "perp_mid": round(mid_perp, 6),
                "basis_mid_bps": round(basis_mid_bps, 4),
                "basis_mark_bps": round(basis_mark_bps, 4),
                "funding_apr_pct": round(funding_apr, 4),
                "est_4leg_rt_cost_bps": round(est_4leg_rt_bps, 4),
            }
        return out

    def _write_live_readiness_snapshot(self) -> None:
        """Atomically persist world-readable (0644) live readiness JSON when configured."""
        if self.readiness_report_path is None:
            return
        rep = self.evaluate_48h_readiness().to_dict()
        fd, tmp_name = tempfile.mkstemp(
            prefix=".ready_", suffix=".json", dir=str(self.readiness_report_path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=2, sort_keys=True, allow_nan=False)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp_name, 0o644)
            os.replace(tmp_name, self.readiness_report_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

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

        fetch_coins = self._primary_coins + self._observe_only_coins
        try:
            snapshots = self.provider.fetch_snapshots(fetch_coins, trace_id=trace_id)
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
            self._write_live_readiness_snapshot()
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

        observe_only_summary = self._record_observe_only_ticks(
            snapshots, trace_id=trace_id, fallback_ts_ms=fallback_ts_ms
        )

        for coin in self._primary_coins:
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

        # RB-08: require configured healthy coin ratio (default 100% of primary universe) to count healthy tick
        min_needed = max(1, int(math.ceil(len(self._primary_coins) * float(self.config.min_healthy_coin_ratio))))
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
            self._write_live_readiness_snapshot()
            ret_d: dict[str, Any] = {
                "trace_id": trace_id,
                "status": self._health_status,
                "valid_coins": 0,
                "equity_usd": round(eq_d, 4),
            }
            if observe_only_summary:
                ret_d["observe_only"] = observe_only_summary
            return ret_d

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
            bars_at_ts = self._build_kernel_bars(valid_snaps, ts_ms)

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
        self._write_live_readiness_snapshot()
        ret_ok: dict[str, Any] = {
            "trace_id": trace_id,
            "status": self._health_status,
            "ts_ms": ts_ms,
            "valid_coins": len(valid_snaps),
            "equity_usd": round(eq_final, 4),
            "margin_ratio": round(margin_final, 6),
            "chain_hash": chain_hash,
            "coins": tick_summary_by_coin,
        }
        if observe_only_summary:
            ret_ok["observe_only"] = observe_only_summary
        return ret_ok

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

        spike_threshold_bps = float(self.kernel.full_4leg_rt_cost_bps)
        primary_rows = self.conn.fetch_all(
            "SELECT coin, exec_4leg_cost_bps FROM shadow_ticks "
            "WHERE guard_status = 'PASS' AND exec_4leg_cost_bps IS NOT NULL"
        )
        primary_all_costs: list[float] = []
        primary_costs_by_coin: dict[str, list[float]] = {}
        for r in primary_rows:
            val = r["exec_4leg_cost_bps"]
            if _is_finite(val):
                fv = float(val)
                primary_all_costs.append(fv)
                primary_costs_by_coin.setdefault(str(r["coin"]), []).append(fv)
        primary_dist = _compute_4leg_cost_distribution(primary_all_costs, spike_threshold_bps)

        primary_coin_cost_stats: dict[str, dict[str, Any]] = {}
        for c in sorted(set(self._primary_coins) | set(primary_costs_by_coin.keys())):
            c_dist = _compute_4leg_cost_distribution(primary_costs_by_coin.get(c, ()), spike_threshold_bps)
            if c_dist["count"] > 0:
                primary_coin_cost_stats[c] = {
                    "valid_ticks": c_dist["count"],
                    "mean_4leg_cost_bps": c_dist["mean_bps"],
                    "p95_4leg_cost_bps": c_dist["p95_bps"],
                    "p99_4leg_cost_bps": c_dist["p99_bps"],
                    "max_4leg_cost_bps": c_dist["max_bps"],
                    "spike_count": c_dist["spike_count"],
                    "spike_rate": c_dist["spike_rate"],
                }

        obs_rows = self.conn.fetch_all(
            "SELECT coin, exec_4leg_cost_bps FROM shadow_ticks "
            "WHERE guard_status = 'OBSERVE_ONLY' AND exec_4leg_cost_bps IS NOT NULL"
        )
        obs_costs_by_coin: dict[str, list[float]] = {}
        for r in obs_rows:
            val = r["exec_4leg_cost_bps"]
            if _is_finite(val):
                obs_costs_by_coin.setdefault(str(r["coin"]), []).append(float(val))

        executed_trades = int(self._executed_trades)
        non_vacuous = executed_trades > 0
        parity_depth = SHADOW_STATUS_ACTIVE if non_vacuous else SHADOW_STATUS_VACUOUS

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
        if not non_vacuous:
            blockers.append(
                f"SHADOW_VACUOUS_ZERO_TRADES: executed_trades=0 "
                f"(warmup_bars={self.kernel.warmup_bars}, prewarmed_bars={self._prewarmed_bars})"
            )

        if hard_fail:
            verdict = "FAIL_CLOSED_BLOCKED"
            shadow_status = SHADOW_STATUS_VACUOUS if not non_vacuous else "FAIL_CLOSED_BLOCKED"
        elif not non_vacuous:
            if elapsed_h < self.config.required_shadow_hours:
                blockers.append(
                    f"Observation elapsed {elapsed_h:.2f}h < required {self.config.required_shadow_hours:.1f}h"
                )
            verdict = SHADOW_STATUS_VACUOUS
            shadow_status = SHADOW_STATUS_VACUOUS
        elif elapsed_h < self.config.required_shadow_hours:
            blockers.append(
                f"Observation elapsed {elapsed_h:.2f}h < required {self.config.required_shadow_hours:.1f}h"
            )
            verdict = "SHADOW_OBSERVATION_IN_PROGRESS"
            shadow_status = SHADOW_STATUS_ACTIVE
        else:
            verdict = "READY_FOR_OWNER_REVIEW"
            shadow_status = SHADOW_STATUS_ACTIVE

        obs_stats_out: dict[str, dict[str, Any]] = {}
        for c, st in sorted(self._observe_only_stats.items()):
            cnt = int(st.get("count", 0.0))
            mean_obs = (float(st.get("cost_sum_bps", 0.0)) / cnt) if cnt > 0 else 0.0
            c_obs_dist = _compute_4leg_cost_distribution(obs_costs_by_coin.get(c, ()), spike_threshold_bps)
            obs_stats_out[c] = {
                "valid_ticks": cnt,
                "fail_ticks": int(st.get("fail_count", 0.0)),
                "mean_4leg_cost_bps": round(mean_obs, 4),
                "max_abs_basis_bps": round(float(st.get("max_abs_basis_bps", 0.0)), 4),
                "last_funding_apr_pct": round(float(st.get("last_funding_apr_pct", 0.0)), 4),
                "p95_4leg_cost_bps": c_obs_dist["p95_bps"],
                "p99_4leg_cost_bps": c_obs_dist["p99_bps"],
                "max_4leg_cost_bps": c_obs_dist["max_bps"],
                "spike_count": c_obs_dist["spike_count"],
                "spike_rate": c_obs_dist["spike_rate"],
            }

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
            backtest_assumed_4leg_cost_bps=spike_threshold_bps,
            shadow_equity_usd=round(eq_now, 4),
            shadow_net_pnl_usd=round(eq_now - float(self.config.capital_usd), 4),
            shadow_funding_usd=round(self._total_funding_usd, 4),
            shadow_cost_usd=round(self._total_cost_usd, 4),
            alert_counts_by_code=dict(sorted(self._alert_counts.items())),
            alert_suppressed_by_code=dict(sorted(self._alert_suppressed_counts.items())),
            blocking_reasons=blockers,
            observe_only_coin_stats=obs_stats_out,
            executed_trades=executed_trades,
            prewarmed_bars=self._prewarmed_bars,
            non_vacuous=non_vacuous,
            non_vacuous_parity=non_vacuous,
            shadow_status=shadow_status,
            parity_depth=parity_depth,
            p95_observed_4leg_cost_bps=primary_dist["p95_bps"],
            p99_observed_4leg_cost_bps=primary_dist["p99_bps"],
            max_observed_4leg_cost_bps=primary_dist["max_bps"],
            tail_cost_spike_threshold_bps=spike_threshold_bps,
            tail_cost_spike_count=primary_dist["spike_count"],
            tail_cost_spike_rate=primary_dist["spike_rate"],
            primary_coin_cost_stats=primary_coin_cost_stats,
        )
