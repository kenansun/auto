"""W2: Web3 Volatility-Managed Adaptive Breakout/Trend CTA + Funding/OI Crowding Filter (F-14).

100% crypto-native directional trend/breakout engine implementing `AssetNativeStrategy`:
  - Uses ONLY G2-verified trend/breakout factors (Carver vol-normalized EWMAC + strictly prior
    dynamic Donchian channels + TSMOM). ZERO unverified/hardcoded `squeeze_breakout_dir` boost
    (fixes V-04 / D2 / P3-10).
  - Volatility-managed position sizing inversely scaled to causal realized annualized volatility.
  - Asymmetric funding & OI crowding throttle: extreme crowded-long funding + rising OI throttles
    long exposure by ~40-50% (and symmetric short-squeeze throttle on extreme negative funding).
  - Cross-sectional market breadth short-side gate: suppresses/blocks shorts when universe breadth
    (`% > 20d EMA`) is strong and enables short breakdowns when breadth collapses (`< 25%`) in
    `RISK_OFF_BEAR` (solving decaying-alt / short-squeeze traps).
  - Causal trailing Wilder ATR `stop_price` on active positions + Carver inertia via
    `sync_executed_weights` for exact G7 streaming-vs-batch bit-parity.
  - Strictly ignores all Yahoo `^VIX` and TradFi macro keys (D7).
"""

from __future__ import annotations

import bisect
from collections import deque
import logging
import math
from typing import Any, Sequence

from alphaloop.platform.core_engine.interfaces import BarDecision
from alphaloop.platform.data_foundation.models import (
    CanonicalBar,
    DAY_MS,
    Web3DerivativesMetric,
)
from alphaloop.platform.strategies.web3.regime_engine import (
    WEB3_REGIME_RISK_OFF_BEAR,
    Web3MarketRegimeEngine,
    is_crypto_bar,
)

logger = logging.getLogger(__name__)

# Strictly G2-verified directional trend/breakout factor whitelist (ZERO squeeze_breakout_dir!)
ALLOWED_W2_FACTORS: frozenset[str] = frozenset(
    {
        "ewmac_8_32",
        "ewmac_16_64",
        "ewmac_32_128",
        "trend_composite",
        "donchian_pos_20",
        "donchian_pos_40",
        "donchian_pos_entry",
        "tsmom_20",
        "tsmom_60",
    }
)
DEFAULT_W2_FACTORS: tuple[str, ...] = (
    "ewmac_8_32",
    "ewmac_16_64",
    "ewmac_32_128",
    "donchian_pos_40",
)

_EWMAC_SCALARS: dict[tuple[int, int], float] = {
    (8, 32): 5.3,
    (16, 64): 3.75,
    (32, 128): 2.65,
}


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


def _mean_std(vals: Sequence[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mu = math.fsum(vals) / n
    if n < 2:
        return mu, 0.0
    ss = math.fsum((x - mu) * (x - mu) for x in vals)
    return mu, math.sqrt(max(0.0, ss / n))


class _W2SymbolState:
    """Strictly causal per-symbol EWMAC, Donchian, Wilder ATR, vol, carry, and OI tracker."""

    __slots__ = (
        "tf_h",
        "bars_per_year",
        "donchian_entry_n",
        "donchian_exit_n",
        "emas",
        "ema_20d",
        "ema_60d",
        "alpha_20d",
        "alpha_60d",
        "closes",
        "highs",
        "lows",
        "log_rets",
        "carry_events",
        "carry_aprs",
        "oi_hist",
        "atr_14",
        "prev_close",
        "bar_count",
        "prior_high_entry",
        "prior_low_entry",
        "prior_high_exit",
        "prior_low_exit",
        "prior_high_20",
        "prior_low_20",
        "prior_high_40",
        "prior_low_40",
    )

    def __init__(self, timeframe_h: float, donchian_entry_n: int, donchian_exit_n: int) -> None:
        self.tf_h = max(0.25, float(timeframe_h))
        self.bars_per_year = (24.0 * 365.0) / self.tf_h
        self.donchian_entry_n = max(4, int(donchian_entry_n))
        self.donchian_exit_n = max(2, int(donchian_exit_n))

        bars_per_day = max(0.5, 24.0 / self.tf_h)
        span_20d = max(5, int(round(20.0 * bars_per_day)))
        span_60d = max(15, int(round(60.0 * bars_per_day)))
        self.alpha_20d = 2.0 / (span_20d + 1.0)
        self.alpha_60d = 2.0 / (span_60d + 1.0)

        self.emas: dict[int, float] = {}
        self.ema_20d: float | None = None
        self.ema_60d: float | None = None

        max_ch = max(65, self.donchian_entry_n + 5, self.donchian_exit_n + 5)
        self.closes: deque[float] = deque(maxlen=max_ch)
        self.highs: deque[float] = deque(maxlen=max_ch)
        self.lows: deque[float] = deque(maxlen=max_ch)
        self.log_rets: deque[float] = deque(maxlen=80)
        self.carry_events: deque[tuple[float, float, float]] = deque(maxlen=12)
        self.carry_aprs: deque[float] = deque(maxlen=80)
        self.oi_hist: deque[tuple[int, float]] = deque(maxlen=120)
        self.atr_14: float | None = None
        self.prev_close: float | None = None
        self.bar_count: int = 0

        self.prior_high_entry: float = 0.0
        self.prior_low_entry: float = 0.0
        self.prior_high_exit: float = 0.0
        self.prior_low_exit: float = 0.0
        self.prior_high_20: float = 0.0
        self.prior_low_20: float = 0.0
        self.prior_high_40: float = 0.0
        self.prior_low_40: float = 0.0

    @staticmethod
    def _prior_hl(highs: deque[float], lows: deque[float], lookback: int, cur_h: float, cur_l: float) -> tuple[float, float]:
        if not highs:
            return cur_h, cur_l
        k = max(2, int(lookback))
        h_slice = list(highs)[-k:]
        l_slice = list(lows)[-k:]
        return max(h_slice), min(l_slice)

    def update(self, bar: CanonicalBar) -> dict[str, float]:
        """Update state with one closed bar and return strictly causal verified factor values."""
        c = float(bar.close)
        h = float(bar.high)
        lo = float(bar.low)
        self.bar_count += 1

        # Strictly PRIOR channel highs/lows BEFORE appending current bar (zero lookahead!)
        self.prior_high_entry, self.prior_low_entry = self._prior_hl(
            self.highs, self.lows, self.donchian_entry_n, h, lo
        )
        self.prior_high_exit, self.prior_low_exit = self._prior_hl(
            self.highs, self.lows, self.donchian_exit_n, h, lo
        )
        self.prior_high_20, self.prior_low_20 = self._prior_hl(self.highs, self.lows, 20, h, lo)
        self.prior_high_40, self.prior_low_40 = self._prior_hl(self.highs, self.lows, 40, h, lo)

        # Update EMAs
        for span in (8, 16, 32, 64, 128):
            if span not in self.emas:
                self.emas[span] = c
            else:
                alpha = 2.0 / (span + 1.0)
                self.emas[span] = alpha * c + (1.0 - alpha) * self.emas[span]

        if self.ema_20d is None:
            self.ema_20d = c
            self.ema_60d = c
        else:
            self.ema_20d = self.alpha_20d * c + (1.0 - self.alpha_20d) * self.ema_20d
            self.ema_60d = self.alpha_60d * c + (1.0 - self.alpha_60d) * (
                self.ema_60d if self.ema_60d is not None else c
            )

        # True Range, Wilder ATR(14), and log return
        if self.prev_close is None:
            tr = h - lo
        else:
            tr = max(h - lo, abs(h - self.prev_close), abs(lo - self.prev_close))
            if self.prev_close > 0.0 and c > 0.0:
                self.log_rets.append(math.log(c / self.prev_close))

        if self.atr_14 is None:
            self.atr_14 = tr
        else:
            self.atr_14 = ((self.atr_14 * 13.0) + tr) / 14.0
        atr_val = max(self.atr_14, c * 1e-5)

        # D11 rolling funding APR & causal z-score
        tf_h = max(0.01, _finite_or(bar.timeframe_h, self.tf_h))
        ci_raw = _finite_or(bar.carry_interval_h, tf_h)
        ci_h = ci_raw if ci_raw > 0.0 else tf_h
        rate = _finite_or(bar.carry_rate_period, 0.0)
        self.carry_events.append((rate, tf_h, ci_h))
        tail_c = list(self.carry_events)[-4:]
        sum_r = math.fsum(r for r, _, _ in tail_c)
        sum_tf = math.fsum(t for _, t, _ in tail_c)
        sum_cov = math.fsum(max(t, ci) for r, t, ci in tail_c if abs(r) > 1e-15)
        carry_apr = (sum_r / max(sum_tf, sum_cov, 0.01)) * 8760.0 * 100.0
        self.carry_aprs.append(carry_apr)

        self.closes.append(c)
        self.highs.append(h)
        self.lows.append(lo)
        self.prev_close = c

        # Realized bar & annualized volatility
        if len(self.log_rets) >= 8:
            _, sd_ret = _mean_std(list(self.log_rets)[-20:])
            bar_vol = max(sd_ret, atr_val / max(c, 1e-8) * 0.25, 1e-4)
        else:
            bar_vol = max(atr_val / max(c, 1e-8), 1e-4)
        realized_vol_ann = max(bar_vol * math.sqrt(self.bars_per_year), 0.12)

        # Carver vol-normalized EWMAC forecasts in [-20, +20]
        price_vol_unit = max(c * bar_vol, atr_val * 0.5, 1e-8)
        ewmac_8_32 = _clamp(((self.emas[8] - self.emas[32]) / price_vol_unit) * _EWMAC_SCALARS[(8, 32)], -20.0, 20.0)
        ewmac_16_64 = _clamp(((self.emas[16] - self.emas[64]) / price_vol_unit) * _EWMAC_SCALARS[(16, 64)], -20.0, 20.0)
        ewmac_32_128 = _clamp(((self.emas[32] - self.emas[128]) / price_vol_unit) * _EWMAC_SCALARS[(32, 128)], -20.0, 20.0)
        trend_comp = _clamp(1.12 * (0.30 * ewmac_8_32 + 0.40 * ewmac_16_64 + 0.30 * ewmac_32_128), -20.0, 20.0)

        def _dpos(hi_p: float, lo_p: float) -> float:
            span = hi_p - lo_p
            if span <= 1e-12:
                return 0.0
            return _clamp(2.0 * ((c - lo_p) / span) - 1.0, -1.5, 1.5)

        donch_20 = _dpos(self.prior_high_20, self.prior_low_20)
        donch_40 = _dpos(self.prior_high_40, self.prior_low_40)
        donch_entry = _dpos(self.prior_high_entry, self.prior_low_entry)

        def _tsmom(lb: int) -> float:
            if len(self.closes) <= lb:
                return 0.0
            ref = self.closes[-1 - lb]
            if ref <= 0.0:
                return 0.0
            return _clamp(math.log(c / ref) / max(bar_vol * math.sqrt(float(lb)), 1e-4), -4.0, 4.0)

        # Endogenous causal funding crowding z-score (rolling z + absolute APR pressure)
        if len(self.carry_aprs) >= 15:
            f_mu, f_sd = _mean_std(list(self.carry_aprs)[:-1] if len(self.carry_aprs) > 15 else list(self.carry_aprs))
            z_hist = _clamp((carry_apr - f_mu) / f_sd, -4.0, 5.0) if f_sd > 1.0 else 0.0
        else:
            z_hist = 0.0
        if carry_apr >= 0.0:
            z_lvl = max(0.0, (carry_apr - 25.0) / 35.0)
            endo_funding_z = max(z_hist, z_lvl)
        else:
            z_lvl = min(0.0, (carry_apr + 25.0) / 35.0)
            endo_funding_z = min(z_hist, z_lvl)

        return {
            "ewmac_8_32": ewmac_8_32,
            "ewmac_16_64": ewmac_16_64,
            "ewmac_32_128": ewmac_32_128,
            "trend_composite": trend_comp,
            "donchian_pos_20": donch_20,
            "donchian_pos_40": donch_40,
            "donchian_pos_entry": donch_entry,
            "tsmom_20": _tsmom(20),
            "tsmom_60": _tsmom(60),
            "atr_14": atr_val,
            "realized_vol_ann": realized_vol_ann,
            "carry_apr_pct": carry_apr,
            "endo_funding_z": endo_funding_z,
        }

    def update_oi(self, ts_ms: int, oi_usd: float) -> None:
        if math.isfinite(oi_usd) and oi_usd > 0.0:
            self.oi_hist.append((int(ts_ms), float(oi_usd)))

    def compute_stream_oi_change_7d(self, ts_ms: int) -> float | None:
        if len(self.oi_hist) < 2:
            return None
        cur = self.oi_hist[-1][1]
        cutoff = int(ts_ms) - 7 * DAY_MS
        ref = self.oi_hist[0][1]
        for h_ts, h_v in self.oi_hist:
            if h_ts <= cutoff:
                ref = h_v
            else:
                break
        if ref <= 0.0:
            return None
        return (cur / ref) - 1.0


class Web3AdaptiveTrendKernel:
    """W2: Crypto-Native Vol-Managed Adaptive Breakout/Trend CTA with Crowding & Breadth Gates."""

    asset_domain: str = "web3"

    def __init__(
        self,
        *,
        donchian_entry_n: int = 20,
        donchian_exit_n: int = 10,
        target_annual_vol_pct: float = 20.0,
        max_symbol_weight: float = 1.0,
        atr_stop_mult: float = 3.0,
        inertia_buffer: float = 0.08,
        warmup_bars: int = 35,
        entry_forecast_threshold: float = 1.5,
        selected_factors: Sequence[str] | None = None,
        metrics_by_symbol: dict[str, Sequence[Web3DerivativesMetric]] | None = None,
        regime_engine: Web3MarketRegimeEngine | None = None,
        strategy_id: str = "web3_w2_adaptive_trend",
    ) -> None:
        self.strategy_id = str(strategy_id)
        self.asset_domain = "web3"
        self.donchian_entry_n = max(4, int(donchian_entry_n))
        self.donchian_exit_n = max(2, int(donchian_exit_n))
        self.target_annual_vol_pct = _clamp(float(target_annual_vol_pct), 5.0, 100.0)
        self.max_symbol_weight = _clamp(float(max_symbol_weight), 0.10, 2.0)
        self.atr_stop_mult = _clamp(float(atr_stop_mult), 1.5, 8.0)
        self.inertia_buffer = _clamp(float(inertia_buffer), 0.0, 0.50)
        self.warmup_bars = max(5, int(warmup_bars))
        self.entry_forecast_threshold = _clamp(float(entry_forecast_threshold), 0.2, 10.0)

        # Enforce G2 factor whitelist: NEVER allow unverified squeeze_breakout_dir (V-04)
        if selected_factors is not None:
            filtered = tuple(
                str(f) for f in selected_factors if str(f) in ALLOWED_W2_FACTORS
            )
            self.selected_factors: tuple[str, ...] = filtered if filtered else ("ewmac_16_64",)
        else:
            self.selected_factors = DEFAULT_W2_FACTORS

        self._pit_oi: dict[str, tuple[list[int], list[float]]] = {}
        self._raw_metrics_ref = metrics_by_symbol
        if metrics_by_symbol:
            for sym in sorted(metrics_by_symbol):
                ordered = sorted(
                    metrics_by_symbol[sym],
                    key=lambda m: (int(m.ts_avail_ms), int(m.ts_event_ms)),
                )
                avails: list[int] = []
                vals: list[float] = []
                for m in ordered:
                    oi_v = _finite_or(getattr(m, "open_interest_usd", 0.0), 0.0)
                    if oi_v > 0.0:
                        avails.append(int(m.ts_avail_ms))
                        vals.append(oi_v)
                if avails:
                    self._pit_oi[sym.upper()] = (avails, vals)

        self._external_regime_template = regime_engine
        self._regime_engine: Web3MarketRegimeEngine = (
            regime_engine.clone_fresh() if regime_engine is not None else Web3MarketRegimeEngine()
        )

        # Mutable streaming state
        self._states: dict[str, _W2SymbolState] = {}
        self._executed_weights: dict[str, float] = {}
        self._trailing_stops: dict[str, float | None] = {}
        self._step_count: int = 0

    def reset(self) -> None:
        """Reset all mutable internal state cleanly."""
        self._states.clear()
        self._executed_weights.clear()
        self._trailing_stops.clear()
        self._step_count = 0
        self._regime_engine.reset()

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        """Sync actual post-allocator/post-fill weights for Carver inertia, trailing stops & G7 parity."""
        for sym, w in executed_weights.items():
            fw = _finite_or(w, 0.0)
            s_key = str(sym)
            prev_w = self._executed_weights.get(s_key, 0.0)
            self._executed_weights[s_key] = fw
            if abs(fw) <= 1e-9 or (abs(prev_w) > 1e-9 and prev_w * fw < 0.0):
                self._trailing_stops[s_key] = None

    def clone_fresh(self) -> Web3AdaptiveTrendKernel:
        """Return a brand-new freshly reset instance with identical configuration."""
        return Web3AdaptiveTrendKernel(
            donchian_entry_n=self.donchian_entry_n,
            donchian_exit_n=self.donchian_exit_n,
            target_annual_vol_pct=self.target_annual_vol_pct,
            max_symbol_weight=self.max_symbol_weight,
            atr_stop_mult=self.atr_stop_mult,
            inertia_buffer=self.inertia_buffer,
            warmup_bars=self.warmup_bars,
            entry_forecast_threshold=self.entry_forecast_threshold,
            selected_factors=self.selected_factors,
            metrics_by_symbol=self._raw_metrics_ref,
            regime_engine=self._external_regime_template,
            strategy_id=self.strategy_id,
        )

    def _lookup_pit_oi_change_7d(self, sym_u: str, ts_ms: int) -> float | None:
        entry = self._pit_oi.get(sym_u)
        if not entry:
            return None
        avails, vals = entry
        i_now = bisect.bisect_right(avails, int(ts_ms)) - 1
        if i_now < 1:
            return None
        cutoff = int(ts_ms) - 7 * DAY_MS
        i_prev = bisect.bisect_right(avails, cutoff) - 1
        if i_prev < 0:
            i_prev = 0
        v_prev = vals[i_prev]
        if v_prev <= 0.0:
            return None
        return (vals[i_now] / v_prev) - 1.0

    def _compute_verified_forecast(self, fvals: dict[str, float]) -> float:
        """Combine strictly G2-verified trend/breakout factors onto Carver [-20, +20] scale."""
        parts: list[float] = []
        for fname in self.selected_factors:
            if fname in ("ewmac_8_32", "ewmac_16_64", "ewmac_32_128", "trend_composite"):
                parts.append(fvals[fname])
            elif fname in ("donchian_pos_20", "donchian_pos_40", "donchian_pos_entry"):
                parts.append(fvals[fname] * 10.0)
            elif fname in ("tsmom_20", "tsmom_60"):
                parts.append(fvals[fname] * 5.0)
        if not parts:
            return 0.0
        raw_mean = math.fsum(parts) / len(parts)
        return _clamp(raw_mean, -20.0, 20.0)

    def _resolve_crowding_inputs(
        self,
        sym: str,
        sym_u: str,
        ts_ms: int,
        state: _W2SymbolState,
        endo_funding_z: float,
        ctx: dict[str, Any],
    ) -> tuple[float, float]:
        # Update streaming OI history if Web3DerivativesMetric supplied in context
        raw_metrics = ctx.get("web3_metrics", ctx.get("derivatives_metrics"))
        if isinstance(raw_metrics, dict):
            m_obj = raw_metrics.get(sym, raw_metrics.get(sym_u))
            if m_obj is not None:
                if isinstance(m_obj, Sequence) and not isinstance(m_obj, (str, bytes)):
                    for item in m_obj:
                        item_av = int(getattr(item, "ts_avail_ms", getattr(item, "ts_event_ms", ts_ms)))
                        if item_av <= int(ts_ms):
                            oi_v = _finite_or(getattr(item, "open_interest_usd", 0.0), 0.0)
                            if oi_v > 0.0:
                                state.update_oi(int(getattr(item, "ts_event_ms", ts_ms)), oi_v)
                else:
                    oi_v = _finite_or(getattr(m_obj, "open_interest_usd", 0.0), 0.0)
                    if oi_v > 0.0:
                        state.update_oi(int(getattr(m_obj, "ts_event_ms", ts_ms)), oi_v)

        # Resolve funding_z
        if "funding_z" in ctx and ctx["funding_z"] is not None:
            fz_raw = ctx["funding_z"]
            if isinstance(fz_raw, dict):
                eff_fz = _finite_or(fz_raw.get(sym, fz_raw.get(sym_u)), endo_funding_z)
            else:
                eff_fz = _finite_or(fz_raw, endo_funding_z)
        else:
            eff_fz = endo_funding_z

        # Resolve oi_change_7d
        if "oi_change_7d" in ctx and ctx["oi_change_7d"] is not None:
            oi_raw = ctx["oi_change_7d"]
            if isinstance(oi_raw, dict):
                eff_oi = _finite_or(oi_raw.get(sym, oi_raw.get(sym_u)), 0.0)
            else:
                eff_oi = _finite_or(oi_raw, 0.0)
        else:
            stream_oi = state.compute_stream_oi_change_7d(ts_ms)
            if stream_oi is not None:
                eff_oi = stream_oi
            else:
                pit_oi = self._lookup_pit_oi_change_7d(sym_u, ts_ms)
                eff_oi = pit_oi if pit_oi is not None else 0.0

        return eff_fz, eff_oi

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        """Consume closed crypto bars at `ts_ms` and emit strictly causal W2 trend decisions."""
        self._step_count += 1
        ctx = context if isinstance(context, dict) else {}

        crypto_bars: dict[str, CanonicalBar] = {}
        non_crypto_syms: set[str] = set()
        for sym in sorted(bars_at_ts):
            bar = bars_at_ts[sym]
            if is_crypto_bar(sym, bar):
                crypto_bars[sym] = bar
            else:
                non_crypto_syms.add(sym)

        reg_snap = self._regime_engine.update(int(ts_ms), crypto_bars, ctx)
        raw_reg = ctx.get("regime_label")
        regime_label = (
            str(raw_reg).strip().upper()
            if isinstance(raw_reg, str) and raw_reg.strip()
            else reg_snap.regime_label
        )
        is_bear_regime = ("BEAR" in regime_label) or ("RISK_OFF" in regime_label)

        decisions: dict[str, BarDecision] = {}
        for sym in sorted(non_crypto_syms):
            decisions[sym] = BarDecision(
                symbol=sym,
                ts_event_ms=int(ts_ms),
                target_weight=0.0,
                raw_signal=0.0,
                regime_label=regime_label,
                stop_price=None,
                metadata={"ignored_non_crypto": True},
            )

        n_crypto = max(1, len(crypto_bars))
        target_sym_vol = (self.target_annual_vol_pct / 100.0) / math.sqrt(float(n_crypto))

        for sym in sorted(crypto_bars):
            bar = crypto_bars[sym]
            state = self._states.get(sym)
            if state is None:
                state = _W2SymbolState(
                    timeframe_h=float(bar.timeframe_h),
                    donchian_entry_n=self.donchian_entry_n,
                    donchian_exit_n=self.donchian_exit_n,
                )
                self._states[sym] = state

            fvals = state.update(bar)

            # Resolve market breadth (context override -> multi-coin regime breadth -> own EMA proxy)
            if "breadth_pct_above_20d" in ctx and ctx["breadth_pct_above_20d"] is not None:
                breadth_20d = _clamp(_finite_or(ctx["breadth_pct_above_20d"], 0.5), 0.0, 1.0)
            elif len(crypto_bars) >= 2:
                breadth_20d = reg_snap.breadth_pct_above_20d
            else:
                ema20 = state.ema_20d if state.ema_20d is not None else float(bar.close)
                ema60 = state.ema_60d if state.ema_60d is not None else float(bar.close)
                if float(bar.close) > ema20 and ema20 >= ema60:
                    breadth_20d = 0.80
                elif float(bar.close) < ema20 and float(bar.close) < ema60:
                    breadth_20d = 0.20
                else:
                    breadth_20d = 0.50

            if state.bar_count < self.warmup_bars:
                decisions[sym] = BarDecision(
                    symbol=sym,
                    ts_event_ms=int(ts_ms),
                    target_weight=0.0,
                    raw_signal=0.0,
                    regime_label=regime_label,
                    stop_price=None,
                    metadata={"warmup": True, "bar_count": state.bar_count},
                )
                continue

            # 1. Strictly G2-verified trend/breakout forecast (ZERO squeeze_breakout_dir!)
            forecast = self._compute_verified_forecast(fvals)
            prev_w = self._executed_weights.get(sym, 0.0)

            # Check Donchian exit channel on existing positions
            c = float(bar.close)
            donch_exit_triggered = False
            if prev_w > 1e-9 and c < state.prior_low_exit and forecast <= self.entry_forecast_threshold:
                donch_exit_triggered = True
            elif prev_w < -1e-9 and c > state.prior_high_exit and forecast >= -self.entry_forecast_threshold:
                donch_exit_triggered = True

            if donch_exit_triggered or abs(forecast) < self.entry_forecast_threshold:
                desired_w = 0.0
            else:
                vol_ann = fvals["realized_vol_ann"]
                vol_scale = target_sym_vol / max(vol_ann, 0.12)
                desired_w = _clamp((forecast / 10.0) * vol_scale, -self.max_symbol_weight, self.max_symbol_weight)

            # 2. Asymmetric Funding / OI Crowding Throttle (~40-50% cut when heavily crowded)
            eff_fz, eff_oi = self._resolve_crowding_inputs(
                sym, sym.upper(), int(ts_ms), state, fvals["endo_funding_z"], ctx
            )
            crowding_mult = 1.0
            if desired_w > 1e-9:
                f_pen = _clamp((eff_fz - 0.8) / 1.8, 0.0, 1.0)
                oi_pen = _clamp((eff_oi - 0.05) / 0.25, 0.0, 1.0)
                crowd_intensity = _clamp(max(0.85 * f_pen, 0.60 * f_pen + 0.40 * oi_pen), 0.0, 1.0)
                crowding_mult = 1.0 - 0.50 * crowd_intensity
                desired_w *= crowding_mult
            elif desired_w < -1e-9:
                f_spen = _clamp((-eff_fz - 0.8) / 1.8, 0.0, 1.0)
                oi_spen = _clamp((eff_oi - 0.05) / 0.25, 0.0, 1.0)
                short_crowd = _clamp(max(0.85 * f_spen, 0.60 * f_spen + 0.40 * oi_spen), 0.0, 1.0)
                crowding_mult = 1.0 - 0.48 * short_crowd
                desired_w *= crowding_mult

            # 3. Breadth Short-Side Gate (blocks/suppresses shorts in strong breadth; allows in bear <25%)
            short_gate_mult = 1.0
            if desired_w < -1e-9:
                if breadth_20d >= 0.55:
                    short_gate_mult = 0.0
                elif breadth_20d >= 0.25:
                    short_gate_mult = 0.50 if is_bear_regime else (0.0 if breadth_20d >= 0.45 else 0.25)
                else:
                    short_gate_mult = 1.0
                desired_w *= short_gate_mult

            # 4. Carver inertia buffer driven strictly by synced weights
            inertia_held = False
            if abs(prev_w) > 1e-9 and prev_w * desired_w > 0.0 and self.inertia_buffer > 0.0:
                band = max(0.015, self.inertia_buffer * max(abs(desired_w), abs(prev_w)))
                if abs(desired_w - prev_w) <= band:
                    final_w = prev_w
                    inertia_held = True
                else:
                    final_w = desired_w
            else:
                final_w = desired_w

            # 5. Causal trailing Wilder ATR stop price for active positions
            atr_14 = fvals["atr_14"]
            prev_stop = self._trailing_stops.get(sym)
            stop_price: float | None = None
            if final_w > 1e-9:
                cand_stop = max(c * 0.05, c - self.atr_stop_mult * atr_14)
                if prev_w > 1e-9 and prev_stop is not None:
                    stop_price = max(prev_stop, cand_stop)
                else:
                    stop_price = cand_stop
                stop_price = min(stop_price, c * 0.995)
            elif final_w < -1e-9:
                cand_stop = c + self.atr_stop_mult * atr_14
                if prev_w < -1e-9 and prev_stop is not None:
                    stop_price = min(prev_stop, cand_stop)
                else:
                    stop_price = cand_stop
                stop_price = max(stop_price, c * 1.005)
            self._trailing_stops[sym] = stop_price

            decisions[sym] = BarDecision(
                symbol=sym,
                ts_event_ms=int(ts_ms),
                target_weight=float(final_w),
                raw_signal=float(forecast),
                regime_label=regime_label,
                stop_price=None if stop_price is None else round(float(stop_price), 8),
                metadata={
                    "forecast": round(forecast, 4),
                    "realized_vol_ann_pct": round(fvals["realized_vol_ann"] * 100.0, 4),
                    "atr_14": round(atr_14, 6),
                    "funding_z": round(eff_fz, 4),
                    "oi_change_7d": round(eff_oi, 4),
                    "crowding_mult": round(crowding_mult, 4),
                    "breadth_pct_above_20d": round(breadth_20d, 4),
                    "short_gate_mult": round(short_gate_mult, 4),
                    "inertia_held": bool(inertia_held),
                    "selected_factors": list(self.selected_factors),
                },
            )

        return decisions
