"""Crypto-Native Web3 Market Regime Engine (Track W, F-12 / Fixes D7).

100% crypto-native point-in-time market regime classifier with strict TradFi isolation:
  - NEVER reads Yahoo `^VIX`, `^GVZ`, `GC=F`, `DXY`, US Treasury yields, or any TradFi macro key.
  - Even when TradFi symbols or Yahoo `^VIX` keys are injected into inputs or `context`,
    they are strictly filtered/ignored so outputs remain bit-identical.

Five crypto-native drivers:
  1. BTC structural trend (`close` vs 20d/60d EMA & volatility-adjusted momentum)
  2. Universe market breadth (`% > 20d EMA` and `% > 60d EMA` across eligible crypto symbols)
  3. BTC vs Altcoin dominance rotation (alt median 20d return minus BTC 20d return)
  4. Deribit `BTC_DVOL` crypto implied volatility level & rolling stress z-score
  5. DefiLlama 30-day Stablecoin supply flow (`STABLECOIN_MCAP_USD`) + HL/Binance funding & OI crowding

Supports BOTH:
  - Stateless PIT queries via `snapshot_as_of(as_of_ms)` using `bisect.bisect_right` (D6-safe:
    out-of-order queries are bit-identical with zero cursor state mutation).
  - Causal streaming bar-by-bar evaluation via `update(ts_ms, bars_at_ts, macro_ctx)` for
    embedded strategy usage and G7 streaming-vs-batch bit-parity.
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import asdict, dataclass
import logging
import math
from typing import Any, Sequence

from alphaloop.platform.data_foundation.models import (
    CanonicalBar,
    DAY_MS,
    Web3DerivativesMetric,
    Web3MarketMacroPoint,
)

logger = logging.getLogger(__name__)

# Canonical Web3 Regime Labels
WEB3_REGIME_ALT_SEASON = "ALT_SEASON"
WEB3_REGIME_EXPANSION = "EXPANSION"
WEB3_REGIME_NEUTRAL = "NEUTRAL"
WEB3_REGIME_CROWDED_STRESS = "CROWDED_STRESS"
WEB3_REGIME_RISK_OFF_BEAR = "RISK_OFF_BEAR"

VALID_WEB3_REGIMES: frozenset[str] = frozenset(
    {
        WEB3_REGIME_ALT_SEASON,
        WEB3_REGIME_EXPANSION,
        WEB3_REGIME_NEUTRAL,
        WEB3_REGIME_CROWDED_STRESS,
        WEB3_REGIME_RISK_OFF_BEAR,
    }
)

_ALLOWED_CRYPTO_ASSET_CLASSES: frozenset[str] = frozenset({"CRYPTO_PERP", "CRYPTO_SPOT"})
_FORBIDDEN_TRADFI_VENUES: frozenset[str] = frozenset({"yahoo", "us_treasury", "bls", "fed", "comex", "nyse", "sse", "szse"})
_FORBIDDEN_TRADFI_SYMBOLS: frozenset[str] = frozenset(
    {
        "^VIX",
        "^GVZ",
        "^TNX",
        "VIX",
        "GVZ",
        "TNX",
        "GC=F",
        "SI=F",
        "DX-Y.NYB",
        "DXY",
        "GLD",
        "SLV",
        "IEF",
        "TIP",
        "TLT",
        "GDX",
        "UUP",
        "SPY",
        "QQQ",
        "PAXG",
        "XAUT",
    }
)


def _clamp(val: float, lo: float, hi: float) -> float:
    if not math.isfinite(val):
        return lo
    return max(lo, min(hi, val))


def _finite_or(val: Any, default: float) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return default
    return fv if math.isfinite(fv) else default


def _fmedian(vals: Sequence[float]) -> float:
    n = len(vals)
    if n == 0:
        return 0.0
    ordered = sorted(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def _fmean_std(vals: Sequence[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mu = math.fsum(vals) / n
    if n < 2:
        return mu, 0.0
    ss = math.fsum((x - mu) * (x - mu) for x in vals)
    var = max(0.0, ss / n)
    return mu, math.sqrt(var)


def is_crypto_bar(symbol: str, bar: CanonicalBar) -> bool:
    """Return True strictly for Web3 crypto spot/perp bars; reject all Yahoo/TradFi/Gold bars (D7)."""
    sym_u = str(symbol or bar.symbol).strip().upper()
    if not sym_u or sym_u.startswith("^") or sym_u in _FORBIDDEN_TRADFI_SYMBOLS:
        return False
    if "=" in sym_u or sym_u.endswith(".SH") or sym_u.endswith(".SZ") or sym_u.endswith(".NYB"):
        return False
    venue_l = str(getattr(bar, "venue", "")).strip().lower()
    if venue_l in _FORBIDDEN_TRADFI_VENUES:
        return False
    ac = str(getattr(bar, "asset_class", "")).strip().upper()
    if ac not in _ALLOWED_CRYPTO_ASSET_CLASSES:
        return False
    close_val = getattr(bar, "close", 0.0)
    return isinstance(close_val, (int, float)) and math.isfinite(float(close_val)) and float(close_val) > 0.0


def annualized_carry_apr_pct(bar: CanonicalBar) -> float:
    """Compute exact D11 annualized funding APR (%) handling both 1h/8h/12h/16h carry intervals."""
    tf_h = max(0.01, _finite_or(getattr(bar, "timeframe_h", 12.0), 12.0))
    ci_raw = _finite_or(getattr(bar, "carry_interval_h", tf_h), tf_h)
    ci_h = ci_raw if ci_raw > 0.0 else tf_h
    denom_h = max(tf_h, ci_h)
    rate = _finite_or(getattr(bar, "carry_rate_period", 0.0), 0.0)
    return (rate / denom_h) * 8760.0 * 100.0


@dataclass(frozen=True)
class Web3RegimeSnapshot:
    """Point-in-time crypto-native market regime state & exposure multipliers."""

    ts_ms: int
    regime_label: str
    exposure_scalar: float
    net_exposure_scalar: float
    risk_scalar: float
    composite_score: float
    btc_trend_score: float
    btc_alt_dominance: float
    breadth_pct_above_20d: float
    breadth_pct_above_60d: float
    dvol: float
    dvol_z: float
    stablecoin_flow_30d: float
    funding_apr_median: float
    crowding_z: float
    oi_change_7d_median: float
    active_universe_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _IndexedBarPoint:
    ts_avail_ms: int
    ts_event_ms: int
    close: float
    ema_20d: float
    ema_60d: float
    ret_20d: float
    funding_apr_24h: float


class _StreamingSymbolTracker:
    """Strictly causal per-symbol EMA, 20d return, and D11 funding tracker."""

    __slots__ = (
        "tf_h",
        "span_20d",
        "span_60d",
        "alpha_20d",
        "alpha_60d",
        "carry_win",
        "ema_20d",
        "ema_60d",
        "closes",
        "carry_events",
        "last_ts_ms",
        "last_close",
        "last_ret_20d",
        "last_funding_apr",
    )

    def __init__(self, timeframe_h: float) -> None:
        self.tf_h = max(0.25, float(timeframe_h))
        bars_per_day = max(0.5, 24.0 / self.tf_h)
        self.span_20d = max(5, int(round(20.0 * bars_per_day)))
        self.span_60d = max(15, int(round(60.0 * bars_per_day)))
        self.alpha_20d = 2.0 / (self.span_20d + 1.0)
        self.alpha_60d = 2.0 / (self.span_60d + 1.0)
        self.carry_win = max(2, int(round(48.0 / self.tf_h)))
        self.ema_20d: float | None = None
        self.ema_60d: float | None = None
        self.closes: deque[float] = deque(maxlen=self.span_20d + 2)
        self.carry_events: deque[tuple[float, float, float]] = deque(maxlen=self.carry_win)
        self.last_ts_ms: int = 0
        self.last_close: float = 0.0
        self.last_ret_20d: float = 0.0
        self.last_funding_apr: float = 0.0

    def step(self, bar: CanonicalBar) -> _IndexedBarPoint:
        c = float(bar.close)
        if self.ema_20d is None:
            self.ema_20d = c
            self.ema_60d = c
        else:
            self.ema_20d = self.alpha_20d * c + (1.0 - self.alpha_20d) * self.ema_20d
            self.ema_60d = self.alpha_60d * c + (1.0 - self.alpha_60d) * (
                self.ema_60d if self.ema_60d is not None else c
            )

        self.closes.append(c)
        if len(self.closes) > 1:
            ref_idx = max(0, len(self.closes) - 1 - self.span_20d)
            ref_c = self.closes[ref_idx]
            ret_20d = (c / ref_c) - 1.0 if ref_c > 0.0 else 0.0
        else:
            ret_20d = 0.0

        tf_h = max(0.01, _finite_or(bar.timeframe_h, self.tf_h))
        ci_raw = _finite_or(bar.carry_interval_h, tf_h)
        ci_h = ci_raw if ci_raw > 0.0 else tf_h
        rate = _finite_or(bar.carry_rate_period, 0.0)
        self.carry_events.append((rate, tf_h, ci_h))
        sum_r = math.fsum(r for r, _, _ in self.carry_events)
        sum_tf = math.fsum(t for _, t, _ in self.carry_events)
        sum_cov = math.fsum(max(t, ci) for r, t, ci in self.carry_events if abs(r) > 1e-15)
        denom_h = max(sum_tf, sum_cov, 0.01)
        funding_apr = (sum_r / denom_h) * 8760.0 * 100.0

        self.last_ts_ms = int(bar.ts_event_ms)
        self.last_close = c
        self.last_ret_20d = ret_20d
        self.last_funding_apr = funding_apr

        return _IndexedBarPoint(
            ts_avail_ms=int(bar.ts_avail_ms),
            ts_event_ms=int(bar.ts_event_ms),
            close=c,
            ema_20d=float(self.ema_20d),
            ema_60d=float(self.ema_60d if self.ema_60d is not None else c),
            ret_20d=ret_20d,
            funding_apr_24h=funding_apr,
        )


def _classify_web3_regime(
    *,
    ts_ms: int,
    btc_trend_score: float,
    btc_alt_dominance: float,
    breadth_20d: float,
    breadth_60d: float,
    dvol: float,
    dvol_z: float,
    stablecoin_flow_30d: float,
    funding_apr_median: float,
    oi_change_7d_median: float,
    active_count: int,
    dvol_stress_threshold: float,
) -> Web3RegimeSnapshot:
    """Pure deterministic crypto-native regime scoring function (zero TradFi inputs)."""
    btc_t = _clamp(btc_trend_score, -1.0, 1.0)
    alt_dom = _clamp(btc_alt_dominance, -1.0, 1.0)
    b20 = _clamp(breadth_20d, 0.0, 1.0)
    b60 = _clamp(breadth_60d, 0.0, 1.0)
    dvol_val = _clamp(dvol, 5.0, 250.0)
    dvol_z_c = _clamp(dvol_z, -4.0, 6.0)
    st_flow = _clamp(stablecoin_flow_30d, -0.50, 0.50)
    fund_med = _clamp(funding_apr_median, -500.0, 500.0)
    oi_med = _clamp(oi_change_7d_median, -0.90, 3.0)

    breadth_score = 2.0 * (0.65 * b20 + 0.35 * b60) - 1.0
    dvol_score = _clamp((60.0 - dvol_val) / 30.0, -1.0, 0.60)
    stable_score = math.tanh(st_flow / 0.05)

    # Funding + OI crowding index (> 2.0 indicates leveraged overheating)
    fund_crowd = max(0.0, (fund_med - 20.0) / 30.0)
    oi_crowd = max(0.0, (oi_med - 0.08) / 0.15)
    crowding_z = _clamp(0.70 * fund_crowd + 0.30 * oi_crowd + max(0.0, dvol_z_c - 1.5) * 0.35, -3.0, 5.0)
    overheat_pen = 0.15 * max(0.0, crowding_z - 1.5)

    composite = _clamp(
        0.35 * btc_t
        + 0.25 * breadth_score
        + 0.20 * dvol_score
        + 0.15 * stable_score
        + 0.05 * alt_dom
        - overheat_pen,
        -2.0,
        2.0,
    )

    if dvol_val >= dvol_stress_threshold and (btc_t < 0.0 or b20 < 0.45):
        label = WEB3_REGIME_RISK_OFF_BEAR
    elif composite <= -0.25 or (btc_t <= -0.25 and b20 < 0.40):
        label = WEB3_REGIME_RISK_OFF_BEAR
    elif dvol_val >= dvol_stress_threshold or crowding_z >= 2.2:
        label = WEB3_REGIME_CROWDED_STRESS
    elif composite >= 0.40 and b20 >= 0.65 and alt_dom >= 0.02:
        label = WEB3_REGIME_ALT_SEASON
    elif composite >= 0.15 and btc_t > 0.0 and b20 >= 0.50:
        label = WEB3_REGIME_EXPANSION
    else:
        label = WEB3_REGIME_NEUTRAL

    # Smooth monotonic exposure scaling anchored inside non-overlapping regime bands
    raw_scalar = 0.60 + 0.45 * composite
    if dvol_val > 70.0:
        raw_scalar *= max(0.25, 1.0 - (dvol_val - 70.0) / 60.0)

    if label == WEB3_REGIME_ALT_SEASON:
        exp_scalar = _clamp(raw_scalar + 0.15, 0.95, 1.20)
        net_scalar = _clamp(0.75 + 0.25 * max(0.0, composite), 0.75, 1.00)
    elif label == WEB3_REGIME_EXPANSION:
        exp_scalar = _clamp(raw_scalar + 0.05, 0.80, 1.05)
        net_scalar = _clamp(0.45 + 0.35 * max(0.0, composite), 0.45, 0.80)
    elif label == WEB3_REGIME_NEUTRAL:
        exp_scalar = _clamp(raw_scalar, 0.45, 0.70)
        net_scalar = _clamp(0.15 + 0.20 * composite, 0.05, 0.35)
    elif label == WEB3_REGIME_CROWDED_STRESS:
        exp_scalar = _clamp(raw_scalar, 0.25, 0.48)
        net_scalar = _clamp(0.10 + 0.10 * composite, 0.0, 0.15)
    else:
        # RISK_OFF_BEAR: strictly market-neutral net exposure (0.0) & low risk scalar
        exp_scalar = _clamp(raw_scalar, 0.10, 0.24)
        net_scalar = 0.0

    return Web3RegimeSnapshot(
        ts_ms=int(ts_ms),
        regime_label=label,
        exposure_scalar=round(exp_scalar, 6),
        net_exposure_scalar=round(net_scalar, 6),
        risk_scalar=round(exp_scalar, 6),
        composite_score=round(composite, 6),
        btc_trend_score=round(btc_t, 6),
        btc_alt_dominance=round(alt_dom, 6),
        breadth_pct_above_20d=round(b20, 6),
        breadth_pct_above_60d=round(b60, 6),
        dvol=round(dvol_val, 4),
        dvol_z=round(dvol_z_c, 4),
        stablecoin_flow_30d=round(st_flow, 6),
        funding_apr_median=round(fund_med, 4),
        crowding_z=round(crowding_z, 4),
        oi_change_7d_median=round(oi_med, 6),
        active_universe_count=int(active_count),
    )


class Web3MarketRegimeEngine:
    """Crypto-native Point-in-Time & Streaming Market Regime Engine (Zero Yahoo/TradFi coupling)."""

    def __init__(
        self,
        *,
        bars_by_symbol: dict[str, Sequence[CanonicalBar]] | None = None,
        dvol_series: Sequence[Web3MarketMacroPoint] | None = None,
        stablecoin_series: Sequence[Web3MarketMacroPoint] | None = None,
        web3_macro: Sequence[Web3MarketMacroPoint] | None = None,
        metrics_by_symbol: dict[str, Sequence[Web3DerivativesMetric]] | None = None,
        btc_symbol: str = "BTC",
        dvol_stress_threshold: float = 80.0,
        dvol_calm_threshold: float = 55.0,
    ) -> None:
        self.btc_symbol = str(btc_symbol).upper()
        self.dvol_stress_threshold = float(dvol_stress_threshold)
        self.dvol_calm_threshold = float(dvol_calm_threshold)

        # Immutable pre-indexed PIT series (stateless bisect_right lookups -> D6 safe)
        self._pit_bars: dict[str, tuple[list[int], list[_IndexedBarPoint]]] = {}
        if bars_by_symbol:
            for sym in sorted(bars_by_symbol):
                raw_list = [b for b in bars_by_symbol[sym] if is_crypto_bar(sym, b)]
                if not raw_list:
                    continue
                ordered = sorted(raw_list, key=lambda b: (int(b.ts_avail_ms), int(b.ts_event_ms)))
                tracker = _StreamingSymbolTracker(float(ordered[0].timeframe_h))
                avails: list[int] = []
                pts: list[_IndexedBarPoint] = []
                for b in ordered:
                    pt = tracker.step(b)
                    avails.append(pt.ts_avail_ms)
                    pts.append(pt)
                self._pit_bars[sym.upper()] = (avails, pts)

        # Split combined web3_macro if explicit series not supplied
        dvol_raw: list[Web3MarketMacroPoint] = list(dvol_series or ())
        stable_raw: list[Web3MarketMacroPoint] = list(stablecoin_series or ())
        if web3_macro:
            for pt in web3_macro:
                sid = str(getattr(pt, "series_id", "")).strip().upper()
                if sid == "BTC_DVOL" and not dvol_series:
                    dvol_raw.append(pt)
                elif sid == "STABLECOIN_MCAP_USD" and not stablecoin_series:
                    stable_raw.append(pt)

        self._dvol_avails, self._dvol_vals = self._index_macro_points(dvol_raw)
        self._stable_avails, self._stable_vals = self._index_macro_points(stable_raw)

        # Pre-index crypto derivatives metrics (OI) strictly by ts_avail_ms
        self._oi_index: dict[str, tuple[list[int], list[float]]] = {}
        if metrics_by_symbol:
            for sym in sorted(metrics_by_symbol):
                sym_u = sym.upper()
                if sym_u in _FORBIDDEN_TRADFI_SYMBOLS or sym_u.startswith("^"):
                    continue
                m_ordered = sorted(
                    metrics_by_symbol[sym],
                    key=lambda m: (int(m.ts_avail_ms), int(m.ts_event_ms)),
                )
                m_avails: list[int] = []
                m_ois: list[float] = []
                for m in m_ordered:
                    oi = _finite_or(getattr(m, "open_interest_usd", 0.0), 0.0)
                    if oi > 0.0:
                        m_avails.append(int(m.ts_avail_ms))
                        m_ois.append(oi)
                if m_avails:
                    self._oi_index[sym_u] = (m_avails, m_ois)

        # Mutable streaming state (cleared cleanly on reset())
        self._stream_trackers: dict[str, _StreamingSymbolTracker] = {}
        self._stream_dvol_hist: deque[float] = deque(maxlen=120)
        self._stream_stable_hist: deque[tuple[int, float]] = deque(maxlen=180)
        self._stream_oi_hist: dict[str, deque[tuple[int, float]]] = {}
        self._last_snapshot: Web3RegimeSnapshot | None = None

    @staticmethod
    def _index_macro_points(points: Sequence[Web3MarketMacroPoint]) -> tuple[list[int], list[float]]:
        valid = [
            p
            for p in points
            if math.isfinite(float(p.value))
            and str(getattr(p, "source", "")).strip().lower() not in _FORBIDDEN_TRADFI_VENUES
        ]
        ordered = sorted(valid, key=lambda p: (int(p.ts_avail_ms), int(p.ts_event_ms)))
        return [int(p.ts_avail_ms) for p in ordered], [float(p.value) for p in ordered]

    def reset(self) -> None:
        """Reset all mutable streaming state without altering immutable PIT pre-loaded series."""
        self._stream_trackers.clear()
        self._stream_dvol_hist.clear()
        self._stream_stable_hist.clear()
        self._stream_oi_hist.clear()
        self._last_snapshot = None

    def clone_fresh(self) -> Web3MarketRegimeEngine:
        """Return a fresh engine sharing immutable PIT indices with empty streaming state."""
        clone = Web3MarketRegimeEngine(
            btc_symbol=self.btc_symbol,
            dvol_stress_threshold=self.dvol_stress_threshold,
            dvol_calm_threshold=self.dvol_calm_threshold,
        )
        clone._pit_bars = self._pit_bars
        clone._dvol_avails = self._dvol_avails
        clone._dvol_vals = self._dvol_vals
        clone._stable_avails = self._stable_avails
        clone._stable_vals = self._stable_vals
        clone._oi_index = self._oi_index
        return clone

    def _lookup_dvol_pit(self, as_of_ms: int, fallback: float = 55.0) -> tuple[float, float]:
        if not self._dvol_avails:
            return fallback, 0.0
        idx = bisect.bisect_right(self._dvol_avails, int(as_of_ms)) - 1
        if idx < 0:
            return fallback, 0.0
        cur = self._dvol_vals[idx]
        win_start = max(0, idx - 59)
        window = self._dvol_vals[win_start : idx + 1]
        if len(window) >= 10:
            mu, sd = _fmean_std(window)
            dz = (cur - mu) / sd if sd > 1e-6 else 0.0
        else:
            dz = (cur - 55.0) / 15.0
        return cur, dz

    def _lookup_stablecoin_flow_30d_pit(self, as_of_ms: int) -> float:
        if len(self._stable_avails) < 2:
            return 0.0
        idx_now = bisect.bisect_right(self._stable_avails, int(as_of_ms)) - 1
        if idx_now < 1:
            return 0.0
        val_now = self._stable_vals[idx_now]
        target_prev_ms = int(as_of_ms) - 30 * DAY_MS
        idx_prev = bisect.bisect_right(self._stable_avails, target_prev_ms) - 1
        if idx_prev < 0:
            idx_prev = 0
        val_prev = self._stable_vals[idx_prev]
        if val_prev <= 0.0:
            return 0.0
        return (val_now / val_prev) - 1.0

    def _lookup_oi_change_7d_pit(self, as_of_ms: int) -> float:
        if not self._oi_index:
            return 0.0
        chgs: list[float] = []
        target_7d_ms = int(as_of_ms) - 7 * DAY_MS
        for sym in sorted(self._oi_index):
            avails, vals = self._oi_index[sym]
            i_now = bisect.bisect_right(avails, int(as_of_ms)) - 1
            if i_now < 1:
                continue
            i_prev = bisect.bisect_right(avails, target_7d_ms) - 1
            if i_prev < 0:
                i_prev = 0
            v_prev = vals[i_prev]
            if v_prev > 0.0:
                chgs.append((vals[i_now] / v_prev) - 1.0)
        return _fmedian(chgs) if chgs else 0.0

    def snapshot_as_of(
        self,
        as_of_ms: int,
        *,
        context: dict[str, Any] | None = None,
    ) -> Web3RegimeSnapshot:
        """Stateless Point-in-Time regime query (`ts_avail_ms <= as_of_ms`) via `bisect_right`.

        Strictly ignores any Yahoo `^VIX` or TradFi keys in `context`.
        """
        ts_q = int(as_of_ms)
        ctx = context if isinstance(context, dict) else {}

        active_pts: dict[str, _IndexedBarPoint] = {}
        for sym in sorted(self._pit_bars):
            avails, pts = self._pit_bars[sym]
            idx = bisect.bisect_right(avails, ts_q) - 1
            if idx >= 0:
                active_pts[sym] = pts[idx]

        # Compute BTC trend, breadth, alt dominance, and median funding across active crypto universe
        above_20 = 0
        above_60 = 0
        fund_aprs: list[float] = []
        alt_rets: list[float] = []
        btc_ret_20d = 0.0
        btc_trend = 0.0

        for sym, pt in active_pts.items():
            if pt.close > pt.ema_20d:
                above_20 += 1
            if pt.close > pt.ema_60d:
                above_60 += 1
            fund_aprs.append(pt.funding_apr_24h)
            if sym == self.btc_symbol:
                btc_ret_20d = pt.ret_20d
                t1 = math.tanh(((pt.close / pt.ema_20d) - 1.0) / 0.06) if pt.ema_20d > 0.0 else 0.0
                t2 = math.tanh(((pt.ema_20d / pt.ema_60d) - 1.0) / 0.08) if pt.ema_60d > 0.0 else 0.0
                t3 = math.tanh(pt.ret_20d / 0.15)
                btc_trend = 0.45 * t1 + 0.30 * t2 + 0.25 * t3
            else:
                alt_rets.append(pt.ret_20d)

        n_act = len(active_pts)
        if n_act > 0:
            breadth_20 = above_20 / float(n_act)
            breadth_60 = above_60 / float(n_act)
            if self.btc_symbol not in active_pts:
                # Fallback structural trend from universe median if BTC not present
                med_ret = _fmedian([p.ret_20d for p in active_pts.values()])
                btc_trend = math.tanh(med_ret / 0.15)
                btc_ret_20d = med_ret
            alt_dom = math.tanh((_fmedian(alt_rets) - btc_ret_20d) / 0.15) if alt_rets else 0.0
            fund_med = _fmedian(fund_aprs)
        else:
            breadth_20 = 0.5
            breadth_60 = 0.5
            alt_dom = 0.0
            fund_med = 0.0

        # Crypto-native macro drivers: prefer pre-loaded PIT series; fallback ONLY to crypto-native ctx keys
        if self._dvol_avails:
            dvol, dvol_z = self._lookup_dvol_pit(ts_q)
        else:
            dvol_raw = ctx.get("BTC_DVOL", ctx.get("dvol"))
            dvol = _finite_or(dvol_raw, 55.0)
            dvol_z = (dvol - 55.0) / 15.0

        if len(self._stable_avails) >= 2:
            st_flow = self._lookup_stablecoin_flow_30d_pit(ts_q)
        else:
            st_flow = _finite_or(ctx.get("stablecoin_flow_30d"), 0.0)

        if self._oi_index:
            oi_med = self._lookup_oi_change_7d_pit(ts_q)
        else:
            oi_raw = ctx.get("oi_change_7d")
            if isinstance(oi_raw, dict):
                oi_vals = [_finite_or(v, 0.0) for k, v in oi_raw.items() if str(k).upper() not in _FORBIDDEN_TRADFI_SYMBOLS]
                oi_med = _fmedian(oi_vals) if oi_vals else 0.0
            else:
                oi_med = _finite_or(oi_raw, 0.0)

        return _classify_web3_regime(
            ts_ms=ts_q,
            btc_trend_score=btc_trend,
            btc_alt_dominance=alt_dom,
            breadth_20d=breadth_20,
            breadth_60d=breadth_60,
            dvol=dvol,
            dvol_z=dvol_z,
            stablecoin_flow_30d=st_flow,
            funding_apr_median=fund_med,
            oi_change_7d_median=oi_med,
            active_count=n_act,
            dvol_stress_threshold=self.dvol_stress_threshold,
        )

    def evaluate(
        self,
        ts_ms: int,
        *,
        bars_at_ts: dict[str, CanonicalBar] | None = None,
        bars_by_symbol: dict[str, Sequence[CanonicalBar]] | None = None,
        dvol: float | None = None,
        stablecoin_flow_30d: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> Web3RegimeSnapshot:
        """Evaluate PIT regime at `ts_ms`, optionally building a temporary PIT view if series are passed."""
        if bars_by_symbol and not self._pit_bars:
            tmp = Web3MarketRegimeEngine(
                bars_by_symbol=bars_by_symbol,
                btc_symbol=self.btc_symbol,
                dvol_stress_threshold=self.dvol_stress_threshold,
                dvol_calm_threshold=self.dvol_calm_threshold,
            )
            merged_ctx: dict[str, Any] = {}
            if isinstance(context, dict):
                for key in ("BTC_DVOL", "dvol", "stablecoin_flow_30d", "oi_change_7d"):
                    if key in context:
                        merged_ctx[key] = context[key]
            if dvol is not None:
                merged_ctx["dvol"] = float(dvol)
            if stablecoin_flow_30d is not None:
                merged_ctx["stablecoin_flow_30d"] = float(stablecoin_flow_30d)
            return tmp.snapshot_as_of(ts_ms, context=merged_ctx)

        if self._pit_bars:
            return self.snapshot_as_of(ts_ms, context=context)

        if bars_at_ts:
            merged_ctx2: dict[str, Any] = {}
            if isinstance(context, dict):
                for key in ("BTC_DVOL", "dvol", "STABLECOIN_MCAP_USD", "stablecoin_flow_30d", "oi_change_7d", "web3_metrics"):
                    if key in context:
                        merged_ctx2[key] = context[key]
            if dvol is not None:
                merged_ctx2["dvol"] = float(dvol)
            if stablecoin_flow_30d is not None:
                merged_ctx2["stablecoin_flow_30d"] = float(stablecoin_flow_30d)
            return self.update(ts_ms, bars_at_ts, merged_ctx2)

        return self.snapshot_as_of(ts_ms, context=context)

    def update(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        macro_ctx: dict[str, Any] | None = None,
    ) -> Web3RegimeSnapshot:
        """Strictly causal streaming bar-by-bar regime update (ignores all Yahoo/TradFi inputs)."""
        ctx = macro_ctx if isinstance(macro_ctx, dict) else {}

        for sym in sorted(bars_at_ts):
            bar = bars_at_ts[sym]
            if not is_crypto_bar(sym, bar):
                continue
            sym_u = sym.upper()
            tracker = self._stream_trackers.get(sym_u)
            if tracker is None:
                tracker = _StreamingSymbolTracker(float(bar.timeframe_h))
                self._stream_trackers[sym_u] = tracker
            tracker.step(bar)

        # Update streaming crypto-native DVOL
        dvol_val: float
        dvol_z: float
        if self._dvol_avails:
            dvol_val, dvol_z = self._lookup_dvol_pit(int(ts_ms))
        else:
            raw_dvol = ctx.get("BTC_DVOL", ctx.get("dvol"))
            if raw_dvol is not None:
                fv = _finite_or(raw_dvol, 55.0)
                self._stream_dvol_hist.append(fv)
            if self._stream_dvol_hist:
                dvol_val = self._stream_dvol_hist[-1]
                if len(self._stream_dvol_hist) >= 10:
                    mu, sd = _fmean_std(list(self._stream_dvol_hist))
                    dvol_z = (dvol_val - mu) / sd if sd > 1e-6 else 0.0
                else:
                    dvol_z = (dvol_val - 55.0) / 15.0
            else:
                dvol_val = 55.0
                dvol_z = 0.0

        # Update streaming DefiLlama stablecoin supply flow
        if len(self._stable_avails) >= 2:
            st_flow = self._lookup_stablecoin_flow_30d_pit(int(ts_ms))
        elif "stablecoin_flow_30d" in ctx and ctx["stablecoin_flow_30d"] is not None:
            st_flow = _finite_or(ctx["stablecoin_flow_30d"], 0.0)
        else:
            raw_mcap = ctx.get("STABLECOIN_MCAP_USD")
            if raw_mcap is not None:
                mv = _finite_or(raw_mcap, 0.0)
                if mv > 0.0:
                    self._stream_stable_hist.append((int(ts_ms), mv))
            if len(self._stream_stable_hist) >= 2:
                cur_ts, cur_val = self._stream_stable_hist[-1]
                ref_val = self._stream_stable_hist[0][1]
                cutoff = cur_ts - 30 * DAY_MS
                for h_ts, h_val in self._stream_stable_hist:
                    if h_ts <= cutoff:
                        ref_val = h_val
                    else:
                        break
                st_flow = (cur_val / ref_val) - 1.0 if ref_val > 0.0 else 0.0
            else:
                st_flow = 0.0

        # Update streaming crypto derivatives OI metrics (if present)
        raw_metrics = ctx.get("web3_metrics", ctx.get("derivatives_metrics"))
        if isinstance(raw_metrics, dict):
            for m_sym in sorted(raw_metrics):
                m_u = str(m_sym).upper()
                if m_u in _FORBIDDEN_TRADFI_SYMBOLS or m_u.startswith("^"):
                    continue
                m_obj = raw_metrics[m_sym]
                oi_v = _finite_or(getattr(m_obj, "open_interest_usd", None), 0.0)
                if oi_v > 0.0:
                    hist = self._stream_oi_hist.setdefault(m_u, deque(maxlen=120))
                    hist.append((int(ts_ms), oi_v))

        # Compute cross-sectional crypto breadth, BTC trend, alt dominance, and median funding
        above_20 = 0
        above_60 = 0
        fund_aprs: list[float] = []
        alt_rets: list[float] = []
        btc_ret_20d = 0.0
        btc_trend = 0.0

        for sym_u in sorted(self._stream_trackers):
            tr = self._stream_trackers[sym_u]
            if tr.ema_20d is None or tr.ema_60d is None:
                continue
            if tr.last_close > tr.ema_20d:
                above_20 += 1
            if tr.last_close > tr.ema_60d:
                above_60 += 1
            fund_aprs.append(tr.last_funding_apr)
            if sym_u == self.btc_symbol:
                btc_ret_20d = tr.last_ret_20d
                t1 = math.tanh(((tr.last_close / tr.ema_20d) - 1.0) / 0.06) if tr.ema_20d > 0.0 else 0.0
                t2 = math.tanh(((tr.ema_20d / tr.ema_60d) - 1.0) / 0.08) if tr.ema_60d > 0.0 else 0.0
                t3 = math.tanh(tr.last_ret_20d / 0.15)
                btc_trend = 0.45 * t1 + 0.30 * t2 + 0.25 * t3
            else:
                alt_rets.append(tr.last_ret_20d)

        n_act = len(self._stream_trackers)
        if n_act > 0:
            breadth_20 = above_20 / float(n_act)
            breadth_60 = above_60 / float(n_act)
            if self.btc_symbol not in self._stream_trackers:
                med_r = _fmedian([t.last_ret_20d for t in self._stream_trackers.values()])
                btc_trend = math.tanh(med_r / 0.15)
                btc_ret_20d = med_r
            alt_dom = math.tanh((_fmedian(alt_rets) - btc_ret_20d) / 0.15) if alt_rets else 0.0
            fund_med = _fmedian(fund_aprs)
        else:
            breadth_20 = 0.5
            breadth_60 = 0.5
            alt_dom = 0.0
            fund_med = 0.0

        # Resolve 7d OI change median
        if self._oi_index:
            oi_med = self._lookup_oi_change_7d_pit(int(ts_ms))
        elif "oi_change_7d" in ctx and ctx["oi_change_7d"] is not None:
            oi_raw = ctx["oi_change_7d"]
            if isinstance(oi_raw, dict):
                oi_vals = [_finite_or(v, 0.0) for k, v in oi_raw.items() if str(k).upper() not in _FORBIDDEN_TRADFI_SYMBOLS]
                oi_med = _fmedian(oi_vals) if oi_vals else 0.0
            else:
                oi_med = _finite_or(oi_raw, 0.0)
        elif self._stream_oi_hist:
            oi_chgs: list[float] = []
            cutoff_7d = int(ts_ms) - 7 * DAY_MS
            for m_u in sorted(self._stream_oi_hist):
                dq = self._stream_oi_hist[m_u]
                if len(dq) < 2:
                    continue
                cur_oi = dq[-1][1]
                ref_oi = dq[0][1]
                for h_ts, h_v in dq:
                    if h_ts <= cutoff_7d:
                        ref_oi = h_v
                    else:
                        break
                if ref_oi > 0.0:
                    oi_chgs.append((cur_oi / ref_oi) - 1.0)
            oi_med = _fmedian(oi_chgs) if oi_chgs else 0.0
        else:
            oi_med = 0.0

        snap = _classify_web3_regime(
            ts_ms=int(ts_ms),
            btc_trend_score=btc_trend,
            btc_alt_dominance=alt_dom,
            breadth_20d=breadth_20,
            breadth_60d=breadth_60,
            dvol=dvol_val,
            dvol_z=dvol_z,
            stablecoin_flow_30d=st_flow,
            funding_apr_median=fund_med,
            oi_change_7d_median=oi_med,
            active_count=n_act,
            dvol_stress_threshold=self.dvol_stress_threshold,
        )
        self._last_snapshot = snap
        return snap
