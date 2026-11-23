"""Single-Kernel Cross-Asset Factor Engine & G2 Direction/Redundancy Auditor (v4 Remediated).

Synthesizes:
- NautilusTrader single-kernel parity: `StreamingFactorEngine.update(bar)` is the ONE
  implementation used by both historical batch evaluation and live bar streaming.
- Rob Carver (`pysystemtrade`): vol-normalized EWMAC forecasts scaled to [-20, +20] with
  published forecast scalar discipline (zero curve-fitting of raw indicator scales).
- QuantPedia (2026): Cross-Asset Gold-Treasury joint momentum state (`Gold ↑ & IEF ↑`
  = falling real rates / easing tailwind) plus Gold/Silver Ratio (`SI=F`) and 10Y yield
  (`^TNX`) strictly Point-in-Time aligned via stateless `bisect.bisect_right` (fixes D6).
- D2 Dynamic Donchian Channels: configurable `donchian_entry_n`/`donchian_exit_n` and
  causal `prior_channel(lookback)` / `donchian_pos(lookback)` queries.
- D11 Carry Normalization: exact `carry_interval_h` rolling 24h normalization fixing the
  8x Gate settlement overstatement and `carry_zscore_60` settlement spikes.
- D12 Upgraded G2 Auditor: segment-isolated Newey-West Bartlett HAC t-stat + H-phase-
  averaged Spearman IC + conditional non-overlapping sparse event expectancy + RD-Agent
  pairwise correlation redundancy pruning.
"""

from __future__ import annotations

import bisect
import logging
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from alphaloop.platform.assets import TradingCalendar
from alphaloop.platform.core_engine.validation import (
    compute_hac_factor_ic,
    compute_sparse_event_expectancy,
)
from alphaloop.platform.data_lake import CanonicalBar

logger = logging.getLogger(__name__)

# Economic hypothesis prior for directional factors:
#  +1 = higher factor value should predict higher forward return
#  -1 = higher factor value should predict lower forward return (e.g. mean-reversion / crowding)
#   0 = non-directional regime/state variable (volatility, efficiency, Hurst, macro state)
FACTOR_ECONOMIC_PRIORS: dict[str, int] = {
    "ewmac_8_32": +1,
    "ewmac_16_64": +1,
    "ewmac_32_128": +1,
    "trend_composite": +1,
    "donchian_pos_20": +1,
    "donchian_pos_40": +1,
    "di_spread_14": +1,
    "tsmom_20": +1,
    "tsmom_60": +1,
    "squeeze_breakout_dir": +1,
    "carry_apr_24h": -1,     # Extremely positive funding = crowded longs (contrarian drag on spot/perp longs)
    "zscore_20": -1,         # Short-term stretch mean-reversion prior
    "rsi_14_norm": +1,       # Momentum regime prior (evaluated honestly; fails if mean-reverting)
    "macro_gold_composite": +1,
    # Non-directional structural/regime descriptors
    "adx_14": 0,
    "kaufman_er_20": 0,
    "atr_pct_14": 0,
    "realized_vol_ann_20": 0,
    "vol_pctile_60": 0,
    "squeeze_ratio_20": 0,
    "autocorr_1_20": 0,
    "variance_ratio_20": 0,
    "hurst_proxy_40": 0,
    "rel_volume_20": 0,
    "carry_zscore_60": 0,
    "macro_gold_treasury_state": 0,
    "macro_vix_stress_z": 0,
    "macro_gsr_trend": 0,
    "macro_gsr_z": 0,
    "macro_tnx_mom": 0,
    "macro_tnx_z": 0,
}

# Carver-style published forecast scalars normalizing raw vol-scaled EWMAC to E[|f|] ~= 10
_EWMAC_SCALARS: dict[tuple[int, int], float] = {
    (8, 32): 5.3,
    (16, 64): 3.75,
    (32, 128): 2.65,
}


@dataclass(frozen=True)
class FactorSnapshot:
    """Strictly causal factor vector emitted at bar close (`ts_avail_ms`)."""

    symbol: str
    venue: str
    timeframe_h: float
    ts_event_ms: int
    ts_avail_ms: int
    close: float
    atr_14: float
    prior_high_40: float
    prior_low_40: float
    prior_high_20: float
    prior_low_20: float
    values: dict[str, float]
    # v4 D2 optional defaulted fields (100% backwards-compatible with 12-kwarg callers)
    prior_high_entry: float = 0.0
    prior_low_entry: float = 0.0
    prior_high_exit: float = 0.0
    prior_low_exit: float = 0.0

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.values.get(key, default))


@dataclass
class FactorAuditResult:
    """Single-factor G2 Direction, HAC IC / Sparse Event, & Monotonicity Audit outcome."""

    factor_name: str
    expected_sign: int
    sample_count: int
    spearman_ic: float
    t_stat: float
    icir: float
    hit_rate: float
    quintile_means_bps: list[float]
    monotonicity_rho: float
    sign_ok: bool
    monotonic_ok: bool
    passed_g2: bool
    pruned_as_redundant: bool = False
    redundant_with: str = ""
    detail: str = ""
    # v4 D12 diagnostic fields (defaulted for 100% v3 compatibility)
    hac_t_stat: float = 0.0
    phase_ic_std: float = 0.0
    audit_mode: str = "continuous_ic"  # "continuous_ic" | "sparse_event"
    event_count: int = 0
    event_mean_return_bps: float = 0.0
    event_t_stat: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FactorAuditSummary:
    """Universe-level G2 Factor Audit + RD-Agent Redundancy Pruning summary."""

    track_name: str
    horizon_bars: int
    total_factors_evaluated: int
    passed_g2_factors: list[str]
    selected_non_redundant_factors: list[str]
    pruned_redundant_pairs: list[dict[str, Any]]
    results: list[FactorAuditResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_name": self.track_name,
            "horizon_bars": self.horizon_bars,
            "total_factors_evaluated": self.total_factors_evaluated,
            "passed_g2_factors": list(self.passed_g2_factors),
            "selected_non_redundant_factors": list(self.selected_non_redundant_factors),
            "pruned_redundant_pairs": list(self.pruned_redundant_pairs),
            "results": [r.to_dict() for r in self.results],
        }


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _fmean(xs: Sequence[float]) -> float:
    return (math.fsum(xs) / len(xs)) if xs else 0.0


def _fpvar(xs: Sequence[float], mu: float | None = None) -> float:
    n = len(xs)
    if n <= 0:
        return 0.0
    m = (math.fsum(xs) / n) if mu is None else mu
    return max(0.0, math.fsum((x - m) * (x - m) for x in xs) / n)


def _fpstdev(xs: Sequence[float], mu: float | None = None) -> float:
    v = _fpvar(xs, mu)
    return math.sqrt(v) if v > 0.0 else 0.0


def _fstdev(xs: Sequence[float], mu: float | None = None) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = (math.fsum(xs) / n) if mu is None else mu
    v = max(0.0, math.fsum((x - m) * (x - m) for x in xs) / (n - 1))
    return math.sqrt(v) if v > 0.0 else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    """Compute average mid-ranks (1-indexed) with exact tie handling in pure Python."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        v = values[order[i]]
        while j + 1 < n and values[order[j + 1]] == v:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg_rank
        i = j + 1
    return out


def spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pure-stdlib Spearman rank correlation; returns 0.0 on constant/degenerate vectors."""
    n = min(len(xs), len(ys))
    if n < 5:
        return 0.0
    rx = _ranks(xs[:n])
    ry = _ranks(ys[:n])
    mx = math.fsum(rx) / n
    my = math.fsum(ry) / n
    vx = math.fsum((a - mx) * (a - mx) for a in rx)
    vy = math.fsum((b - my) * (b - my) for b in ry)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return _clamp(cov / math.sqrt(vx * vy), -1.0, 1.0)


class MacroRegimeTimeline:
    """Stateless, strictly Point-in-Time causal macro regime provider (Fixes D6).

    Implements QuantPedia (2026) Cross-Asset Price-Based Regimes for Gold:
      - Joint (Gold Mom > 0 & IEF Treasury Mom > 0) => State +1 (Falling Real Rates / Easing)
      - Joint (Gold Mom < 0 & IEF Treasury Mom < 0) => State -1 (Rising Real Rates / Tightening)
      - Mixed states => State 0
    Combined with DXY inverse momentum, TIP & ^TNX real-rate signals, Gold/Silver Ratio
    (`GC=F / SI=F`), and VIX stress z-score.

    Every lookup uses pure stateless `bisect.bisect_right(ts_avail, as_of_ms)` so non-monotonic,
    repeated, or concurrent queries are mathematically guaranteed bit-identical with zero lookahead.
    """

    def __init__(self, macro_bars_by_symbol: dict[str, list[CanonicalBar]]) -> None:
        self._ts_avail: dict[str, list[int]] = {}
        self._closes: dict[str, list[float]] = {}
        for sym, bars in macro_bars_by_symbol.items():
            ordered = sorted(
                (b for b in bars if math.isfinite(b.close) and b.close > 0.0),
                key=lambda b: (int(b.ts_avail_ms), int(b.ts_event_ms)),
            )
            dedup_ts: list[int] = []
            dedup_closes: list[float] = []
            for b in ordered:
                t_av = int(b.ts_avail_ms)
                c_val = float(b.close)
                if dedup_ts and dedup_ts[-1] == t_av:
                    dedup_closes[-1] = c_val
                else:
                    dedup_ts.append(t_av)
                    dedup_closes.append(c_val)
            self._ts_avail[sym] = dedup_ts
            self._closes[sym] = dedup_closes

        # Precompute strictly PIT-aligned Gold/Silver Ratio (GC=F/SI=F or GLD/SI=F|SLV)
        self._gsr_ts: list[int] = []
        self._gsr_vals: list[float] = []
        gold_sym = "GC=F" if self._ts_avail.get("GC=F") else ("GLD" if self._ts_avail.get("GLD") else "")
        silver_sym = "SI=F" if self._ts_avail.get("SI=F") else ("SLV" if self._ts_avail.get("SLV") else "")
        if gold_sym and silver_sym:
            g_map = dict(zip(self._ts_avail[gold_sym], self._closes[gold_sym]))
            s_map = dict(zip(self._ts_avail[silver_sym], self._closes[silver_sym]))
            for ts in sorted(set(g_map) & set(s_map)):
                sv = s_map[ts]
                if sv > 1e-12:
                    self._gsr_ts.append(ts)
                    self._gsr_vals.append(g_map[ts] / sv)

    @staticmethod
    def _window_from(ts_list: list[int], val_list: list[float], as_of_ms: int, need: int) -> list[float]:
        if not ts_list:
            return []
        idx = bisect.bisect_right(ts_list, int(as_of_ms))
        if idx <= 0:
            return []
        start = max(0, idx - need)
        return val_list[start:idx]

    def _window_at(self, sym: str, as_of_ms: int, need: int) -> list[float]:
        return self._window_from(
            self._ts_avail.get(sym, []),
            self._closes.get(sym, []),
            as_of_ms,
            need,
        )

    @staticmethod
    def _mom_sign_strength(closes: list[float], windows: tuple[int, ...] = (21, 63, 126, 252)) -> float:
        """Multi-horizon trailing return score in [-1, +1] using strictly past closes."""
        if len(closes) < 22:
            return 0.0
        cur = closes[-1]
        scores: list[float] = []
        for w in windows:
            if len(closes) > w and closes[-1 - w] > 0.0:
                ret = cur / closes[-1 - w] - 1.0
                scores.append(math.tanh(ret / 0.05))
        return _fmean(scores) if scores else 0.0

    @staticmethod
    def _yield_mom_strength(yields_pct: list[float], windows: tuple[int, ...] = (21, 63, 126)) -> float:
        """Multi-horizon trailing yield change score in [-1, +1] (+1 = yields rising, 50bp scale)."""
        if len(yields_pct) < 22:
            return 0.0
        cur = yields_pct[-1]
        scores: list[float] = []
        for w in windows:
            if len(yields_pct) > w:
                delta_pct = cur - yields_pct[-1 - w]
                scores.append(math.tanh(delta_pct / 0.50))
        return _fmean(scores) if scores else 0.0

    @staticmethod
    def _causal_zscore(vals: list[float], min_len: int = 25, lo: float = -4.0, hi: float = 4.0) -> float:
        if len(vals) < min_len:
            return 0.0
        hist = vals[:-1]
        mu = _fmean(hist)
        sd = _fpstdev(hist, mu)
        if sd <= 1e-6:
            return 0.0
        return _clamp((vals[-1] - mu) / sd, lo, hi)

    def snapshot_as_of(self, as_of_ms: int) -> dict[str, float]:
        """Return causal macro regime vector strictly visible at `as_of_ms` (100% stateless)."""
        gold_sym = "GC=F" if self._ts_avail.get("GC=F") else ("GLD" if self._ts_avail.get("GLD") else "")
        gold_c = self._window_at(gold_sym, as_of_ms, 260) if gold_sym else []
        ief_c = self._window_at("IEF", as_of_ms, 260)
        dxy_c = self._window_at("DX-Y.NYB", as_of_ms, 260)
        tip_c = self._window_at("TIP", as_of_ms, 260)
        vix_c = self._window_at("^VIX", as_of_ms, 130)
        gsr_c = self._window_from(self._gsr_ts, self._gsr_vals, as_of_ms, 135)
        tnx_c = self._window_at("^TNX", as_of_ms, 135)

        gold_mom = self._mom_sign_strength(gold_c)
        ief_mom = self._mom_sign_strength(ief_c)
        dxy_mom = self._mom_sign_strength(dxy_c)
        tip_mom = self._mom_sign_strength(tip_c)
        gsr_trend = self._mom_sign_strength(gsr_c, windows=(21, 63, 126))
        gsr_z = self._causal_zscore(gsr_c, min_len=25, lo=-4.0, hi=4.0)
        tnx_mom = self._yield_mom_strength(tnx_c, windows=(21, 63, 126))
        tnx_z = self._causal_zscore(tnx_c, min_len=25, lo=-4.0, hi=4.0)
        vix_stress_z = self._causal_zscore(vix_c, min_len=25, lo=-4.0, hi=5.0)

        if gold_mom > 0.05 and ief_mom > 0.05:
            gt_state = 1.0   # QuantPedia State 1: Gold up + Treasuries up (falling real yields)
        elif gold_mom < -0.05 and ief_mom < -0.05:
            gt_state = -1.0  # QuantPedia State 4: Gold down + Treasuries down (rising real yields)
        else:
            gt_state = 0.0

        dxy_tailwind = -dxy_mom
        if len(tnx_c) >= 22:
            real_rate_tailwind = 0.40 * ief_mom + 0.40 * tip_mom + 0.20 * (-tnx_mom)
        else:
            real_rate_tailwind = 0.50 * ief_mom + 0.50 * tip_mom

        composite = _clamp(
            0.45 * gt_state + 0.25 * dxy_tailwind + 0.30 * real_rate_tailwind,
            -1.0,
            1.0,
        )
        return {
            "macro_gold_treasury_state": round(gt_state, 4),
            "macro_dxy_tailwind": round(dxy_tailwind, 4),
            "macro_real_rate_tailwind": round(real_rate_tailwind, 4),
            "macro_vix_stress_z": round(vix_stress_z, 4),
            "macro_gold_composite": round(composite, 4),
            "macro_gsr_trend": round(gsr_trend, 4),
            "macro_gsr_z": round(gsr_z, 4),
            "macro_tnx_mom": round(tnx_mom, 4),
            "macro_tnx_z": round(tnx_z, 4),
        }


class StreamingFactorEngine:
    """Single-kernel strictly causal multi-factor calculator (exact batch == streaming parity)."""

    def __init__(
        self,
        calendar: TradingCalendar,
        timeframe_h: float,
        donchian_entry_n: int = 40,
        donchian_exit_n: int = 20,
    ) -> None:
        self.calendar = calendar
        self.timeframe_h = float(timeframe_h)
        self.bars_per_year = calendar.bars_per_year(timeframe_h)
        self.donchian_entry_n = max(2, int(donchian_entry_n))
        self.donchian_exit_n = max(2, int(donchian_exit_n))
        self._bar_count = 0
        self._prev_bar: CanonicalBar | None = None

        # EMA trackers for Carver EWMAC (8, 16, 32, 64, 128)
        self._emas: dict[int, float] = {}
        self._ema_spans = (8, 16, 32, 64, 128)

        # Wilder ATR / ADX / RSI state
        self._atr14: float | None = None
        self._sm_tr: float = 0.0
        self._sm_pdm: float = 0.0
        self._sm_mdm: float = 0.0
        self._dx_buf: list[float] = []
        self._adx14: float | None = None
        self._avg_gain14: float | None = None
        self._avg_loss14: float | None = None
        self._gain_seed: list[float] = []
        self._loss_seed: list[float] = []

        # Rolling causal buffers
        chan_maxlen = max(130, self.donchian_entry_n + 5, self.donchian_exit_n + 5)
        self._closes: deque[float] = deque(maxlen=max(130, chan_maxlen))
        self._highs: deque[float] = deque(maxlen=chan_maxlen)
        self._lows: deque[float] = deque(maxlen=chan_maxlen)
        self._vols: deque[float] = deque(maxlen=65)
        self._log_rets: deque[float] = deque(maxlen=65)
        self._atr_pcts: deque[float] = deque(maxlen=65)

        # D11 Carry rolling window & smoothed APR history
        self._carry_win: int = max(1, int(round(24.0 / max(0.01, self.timeframe_h))))
        self._carry_events: deque[tuple[float, float, float]] = deque(maxlen=max(65, self._carry_win + 4))
        self._carry_aprs: deque[float] = deque(maxlen=65)

    @staticmethod
    def _prior_hl(buf_h: deque[float], buf_l: deque[float], lb: int, cur_h: float, cur_l: float) -> tuple[float, float]:
        if len(buf_h) < 2:
            return cur_h, cur_l
        k = max(2, int(lb))
        return max(list(buf_h)[-k:]), min(list(buf_l)[-k:])

    def prior_channel(self, lookback: int) -> tuple[float, float]:
        """Return strictly causal `(prior_high, prior_low)` over trailing `lookback` bars
        EXCLUDING the most recently ingested bar (zero lookahead)."""
        if not self._highs:
            return 0.0, 0.0
        prior_h = list(self._highs)[:-1]
        prior_l = list(self._lows)[:-1]
        if len(prior_h) < 2:
            return self._highs[-1], self._lows[-1]
        k = max(2, int(lookback))
        return max(prior_h[-k:]), min(prior_l[-k:])

    def donchian_pos(self, lookback: int, close: float | None = None) -> float:
        """Return strictly causal Donchian channel position in [-1.5, +1.5] for arbitrary `lookback`."""
        if not self._closes:
            return 0.0
        c = self._closes[-1] if close is None else float(close)
        hi, lo = self.prior_channel(lookback)
        span = hi - lo
        if span <= 1e-12:
            return 0.0
        return _clamp(2.0 * ((c - lo) / span) - 1.0, -1.5, 1.5)

    def update(
        self,
        bar: CanonicalBar,
        macro_ctx: dict[str, float] | None = None,
    ) -> FactorSnapshot:
        """Consume one closed bar and emit its strictly causal FactorSnapshot."""
        self._bar_count += 1
        c = float(bar.close)
        h = float(bar.high)
        l = float(bar.low)

        # Capture strictly PRIOR channel highs/lows BEFORE appending current bar (zero lookahead!)
        prior_high_40, prior_low_40 = self._prior_hl(self._highs, self._lows, 40, h, l)
        prior_high_20, prior_low_20 = self._prior_hl(self._highs, self._lows, 20, h, l)
        prior_high_entry, prior_low_entry = self._prior_hl(self._highs, self._lows, self.donchian_entry_n, h, l)
        prior_high_exit, prior_low_exit = self._prior_hl(self._highs, self._lows, self.donchian_exit_n, h, l)

        # 1. Update EMAs
        for span in self._ema_spans:
            if span not in self._emas:
                self._emas[span] = c
            else:
                alpha = 2.0 / (span + 1.0)
                self._emas[span] = alpha * c + (1.0 - alpha) * self._emas[span]

        # 2. Update True Range, Wilder ATR(14), ADX(14), RSI(14), log return
        if self._prev_bar is None:
            tr = h - l
            log_ret = 0.0
            di_spread = 0.0
        else:
            prev_c = float(self._prev_bar.close)
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            log_ret = math.log(c / prev_c) if prev_c > 0.0 else 0.0
            self._log_rets.append(log_ret)

            # Wilder RSI(14)
            delta = c - prev_c
            gain = max(0.0, delta)
            loss = max(0.0, -delta)
            if self._avg_gain14 is None:
                self._gain_seed.append(gain)
                self._loss_seed.append(loss)
                if len(self._gain_seed) == 14:
                    self._avg_gain14 = math.fsum(self._gain_seed) / 14.0
                    self._avg_loss14 = math.fsum(self._loss_seed) / 14.0
            else:
                self._avg_gain14 = ((self._avg_gain14 * 13.0) + gain) / 14.0
                self._avg_loss14 = ((self._avg_loss14 * 13.0) + loss) / 14.0

            # Wilder Directional Movement & ADX(14)
            up_move = h - float(self._prev_bar.high)
            dn_move = float(self._prev_bar.low) - l
            p_dm = up_move if (up_move > dn_move and up_move > 0.0) else 0.0
            m_dm = dn_move if (dn_move > up_move and dn_move > 0.0) else 0.0
            if self._bar_count <= 15:
                self._sm_tr += tr
                self._sm_pdm += p_dm
                self._sm_mdm += m_dm
            else:
                self._sm_tr = self._sm_tr - (self._sm_tr / 14.0) + tr
                self._sm_pdm = self._sm_pdm - (self._sm_pdm / 14.0) + p_dm
                self._sm_mdm = self._sm_mdm - (self._sm_mdm / 14.0) + m_dm

            if self._bar_count >= 15 and self._sm_tr > 1e-12:
                pdi = 100.0 * self._sm_pdm / self._sm_tr
                mdi = 100.0 * self._sm_mdm / self._sm_tr
                di_spread = (pdi - mdi) / 100.0
                denom = pdi + mdi
                dx = 100.0 * abs(pdi - mdi) / denom if denom > 1e-12 else 0.0
                if self._adx14 is None:
                    self._dx_buf.append(dx)
                    if len(self._dx_buf) == 14:
                        self._adx14 = math.fsum(self._dx_buf) / 14.0
                else:
                    self._adx14 = ((self._adx14 * 13.0) + dx) / 14.0
            else:
                di_spread = 0.0

        if self._atr14 is None:
            self._atr14 = tr
        else:
            self._atr14 = ((self._atr14 * 13.0) + tr) / 14.0

        atr_val = max(self._atr14, c * 1e-5)
        atr_pct = (atr_val / c) * 100.0 if c > 0.0 else 0.0
        self._atr_pcts.append(atr_pct)

        # D11: Exact carry_interval_h-aware rolling 24h Carry APR (fixes 8x Gate bug without double-dividing!)
        tf_h = max(0.01, float(bar.timeframe_h))
        ci_h_raw = getattr(bar, "carry_interval_h", tf_h)
        ci_h = max(0.01, float(ci_h_raw if (ci_h_raw is not None and float(ci_h_raw) > 0.0) else tf_h))
        carry_rate = float(bar.carry_rate_period)
        self._carry_events.append((carry_rate, tf_h, ci_h))

        recent_carry = list(self._carry_events)[-self._carry_win :]
        sum_carry_rate = math.fsum(r for r, _, _ in recent_carry)
        elapsed_h = math.fsum(t for _, t, _ in recent_carry)
        covered_settlement_h = math.fsum(max(t, ci) for r, t, ci in recent_carry if abs(r) > 1e-15)
        denom_h = max(elapsed_h, covered_settlement_h, 0.01)
        carry_apr_24h = (sum_carry_rate / denom_h) * 24.0 * 365.0 * 100.0
        self._carry_aprs.append(carry_apr_24h)

        # Append current bar to rolling buffers
        self._closes.append(c)
        self._highs.append(h)
        self._lows.append(l)
        self._vols.append(float(bar.volume_usd))
        self._prev_bar = bar

        closes_list = list(self._closes)
        rets_list = list(self._log_rets)

        # 3. Volatility & Squeeze
        if len(rets_list) >= 10:
            r20 = rets_list[-20:]
            bar_vol = _fpstdev(r20)
        else:
            bar_vol = max(atr_pct / 100.0, 1e-4)
        realized_vol_ann = bar_vol * math.sqrt(self.bars_per_year) * 100.0

        # Rolling volatility percentile rank [0, 1]
        if len(self._atr_pcts) >= 10:
            ap_list = list(self._atr_pcts)
            vol_pctile = sum(1 for x in ap_list if x <= atr_pct) / len(ap_list)
        else:
            vol_pctile = 0.5

        # Bollinger / Keltner squeeze ratio & Z-score
        if len(closes_list) >= 20:
            w20 = closes_list[-20:]
            sma20 = math.fsum(w20) / 20.0
            sd20 = _fpstdev(w20, sma20)
            zscore_20 = _clamp((c - sma20) / sd20, -4.0, 4.0) if sd20 > 1e-12 else 0.0
            bb_half = 2.0 * sd20
            kc_half = 1.5 * atr_val
            squeeze_ratio_20 = _clamp(bb_half / kc_half, 0.2, 3.0) if kc_half > 1e-12 else 1.0
        else:
            zscore_20 = 0.0
            squeeze_ratio_20 = 1.0

        # 4. Carver vol-normalized EWMAC forecasts in [-20, +20]
        price_vol_unit = max(c * max(bar_vol, 1e-4), atr_val * 0.5, 1e-8)
        ewmac_vals: dict[str, float] = {}
        for fast, slow in ((8, 32), (16, 64), (32, 128)):
            raw = (self._emas[fast] - self._emas[slow]) / price_vol_unit
            scaled = _clamp(raw * _EWMAC_SCALARS[(fast, slow)], -20.0, 20.0)
            ewmac_vals[f"ewmac_{fast}_{slow}"] = scaled
        trend_composite = _clamp(
            1.12
            * (
                0.30 * ewmac_vals["ewmac_8_32"]
                + 0.40 * ewmac_vals["ewmac_16_64"]
                + 0.30 * ewmac_vals["ewmac_32_128"]
            ),
            -20.0,
            20.0,
        )

        # 5. Donchian normalized channel positions [-1.5, +1.5] using strictly PRIOR high/low!
        def _donch_pos(hi: float, lo: float) -> float:
            span = hi - lo
            if span <= 1e-12:
                return 0.0
            return _clamp(2.0 * ((c - lo) / span) - 1.0, -1.5, 1.5)

        donchian_pos_20 = _donch_pos(prior_high_20, prior_low_20)
        donchian_pos_40 = _donch_pos(prior_high_40, prior_low_40)
        donchian_pos_entry = _donch_pos(prior_high_entry, prior_low_entry)
        donchian_pos_exit = _donch_pos(prior_high_exit, prior_low_exit)

        # Squeeze breakout directional signal
        squeeze_breakout_dir = 0.0
        if squeeze_ratio_20 < 1.05:
            if c > prior_high_20:
                squeeze_breakout_dir = 1.0
            elif c < prior_low_20:
                squeeze_breakout_dir = -1.0

        # 6. Kaufman Efficiency Ratio (20)
        if len(closes_list) >= 21:
            net_move = abs(closes_list[-1] - closes_list[-21])
            path_len = math.fsum(
                abs(closes_list[i] - closes_list[i - 1])
                for i in range(len(closes_list) - 20, len(closes_list))
            )
            kaufman_er_20 = _clamp(net_move / path_len, 0.0, 1.0) if path_len > 1e-12 else 0.0
        else:
            kaufman_er_20 = 0.3

        # 7. Time-series momentum (vol-normalized)
        def _tsmom(lb: int) -> float:
            if len(closes_list) <= lb or closes_list[-1 - lb] <= 0.0:
                return 0.0
            ret = math.log(c / closes_list[-1 - lb])
            denom = max(bar_vol * math.sqrt(lb), 1e-4)
            return _clamp(ret / denom, -4.0, 4.0)

        tsmom_20 = _tsmom(20)
        tsmom_60 = _tsmom(60)

        # 8. Wilder RSI normalized [-1, +1]
        if self._avg_gain14 is not None and self._avg_loss14 is not None:
            if self._avg_loss14 <= 1e-12:
                rsi = 100.0 if self._avg_gain14 > 0.0 else 50.0
            else:
                rs = self._avg_gain14 / self._avg_loss14
                rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi_norm = (rsi - 50.0) / 50.0
        else:
            rsi_norm = 0.0

        # 9. Autocorrelation, Variance Ratio (Lo-MacKinlay), and R/S Hurst proxy
        autocorr_1_20 = 0.0
        variance_ratio_20 = 1.0
        if len(rets_list) >= 20:
            r20 = rets_list[-20:]
            x0, x1 = r20[:-1], r20[1:]
            mx0 = math.fsum(x0) / len(x0)
            mx1 = math.fsum(x1) / len(x1)
            vx0 = math.fsum((a - mx0) * (a - mx0) for a in x0)
            vx1 = math.fsum((b - mx1) * (b - mx1) for b in x1)
            if vx0 > 1e-12 and vx1 > 1e-12:
                autocorr_1_20 = _clamp(
                    math.fsum((a - mx0) * (b - mx1) for a, b in zip(x0, x1)) / math.sqrt(vx0 * vx1),
                    -1.0,
                    1.0,
                )
            var1 = _fpvar(r20)
            r2_bars = [r20[i] + r20[i + 1] for i in range(0, len(r20) - 1, 2)]
            if var1 > 1e-12 and len(r2_bars) >= 4:
                var2 = _fpvar(r2_bars)
                variance_ratio_20 = _clamp(var2 / (2.0 * var1), 0.2, 2.5)

        hurst_proxy_40 = 0.5
        if len(rets_list) >= 40:
            r40 = rets_list[-40:]
            mu40 = math.fsum(r40) / 40.0
            sd40 = _fpstdev(r40, mu40)
            if sd40 > 1e-12:
                cum = 0.0
                cums: list[float] = []
                for x in r40:
                    cum += x - mu40
                    cums.append(cum)
                rs_val = (max(cums) - min(cums)) / sd40
                if rs_val > 1e-9:
                    hurst_proxy_40 = _clamp(math.log(rs_val) / math.log(40.0), 0.1, 0.95)

        # 10. Relative Volume & Carry stats
        if len(self._vols) >= 10:
            v20 = list(self._vols)[-20:]
            avg_v = math.fsum(v20) / len(v20)
            rel_vol = _clamp(float(bar.volume_usd) / avg_v, 0.0, 10.0) if avg_v > 1e-9 else 1.0
        else:
            rel_vol = 1.0

        carry_list = list(self._carry_aprs)
        if len(carry_list) >= 15:
            cmu = _fmean(carry_list)
            csd = _fpstdev(carry_list, cmu)
            carry_zscore_60 = _clamp((carry_apr_24h - cmu) / csd, -4.0, 4.0) if csd > 1e-6 else 0.0
        else:
            carry_zscore_60 = 0.0

        values: dict[str, float] = {
            **ewmac_vals,
            "trend_composite": trend_composite,
            "donchian_pos_20": donchian_pos_20,
            "donchian_pos_40": donchian_pos_40,
            "donchian_pos_entry": donchian_pos_entry,
            "donchian_pos_exit": donchian_pos_exit,
            "di_spread_14": di_spread,
            "adx_14": self._adx14 if self._adx14 is not None else 15.0,
            "kaufman_er_20": kaufman_er_20,
            "tsmom_20": tsmom_20,
            "tsmom_60": tsmom_60,
            "atr_pct_14": atr_pct,
            "realized_vol_ann_20": realized_vol_ann,
            "vol_pctile_60": vol_pctile,
            "squeeze_ratio_20": squeeze_ratio_20,
            "squeeze_breakout_dir": squeeze_breakout_dir,
            "rsi_14_norm": rsi_norm,
            "zscore_20": zscore_20,
            "autocorr_1_20": autocorr_1_20,
            "variance_ratio_20": variance_ratio_20,
            "hurst_proxy_40": hurst_proxy_40,
            "rel_volume_20": rel_vol,
            "carry_apr_24h": carry_apr_24h,
            "carry_zscore_60": carry_zscore_60,
        }
        if macro_ctx:
            for k, v in macro_ctx.items():
                values[k] = float(v)
        else:
            values.setdefault("macro_gold_treasury_state", 0.0)
            values.setdefault("macro_dxy_tailwind", 0.0)
            values.setdefault("macro_real_rate_tailwind", 0.0)
            values.setdefault("macro_vix_stress_z", 0.0)
            values.setdefault("macro_gold_composite", 0.0)
            values.setdefault("macro_gsr_trend", 0.0)
            values.setdefault("macro_gsr_z", 0.0)
            values.setdefault("macro_tnx_mom", 0.0)
            values.setdefault("macro_tnx_z", 0.0)

        return FactorSnapshot(
            symbol=bar.symbol,
            venue=bar.venue,
            timeframe_h=bar.timeframe_h,
            ts_event_ms=bar.ts_event_ms,
            ts_avail_ms=bar.ts_avail_ms,
            close=c,
            atr_14=atr_val,
            prior_high_40=prior_high_40,
            prior_low_40=prior_low_40,
            prior_high_20=prior_high_20,
            prior_low_20=prior_low_20,
            values=values,
            prior_high_entry=prior_high_entry,
            prior_low_entry=prior_low_entry,
            prior_high_exit=prior_high_exit,
            prior_low_exit=prior_low_exit,
        )

    def compute_series(
        self,
        bars: list[CanonicalBar],
        macro_timeline: MacroRegimeTimeline | None = None,
    ) -> list[FactorSnapshot]:
        """Batch helper that calls the exact same `update()` kernel bar-by-bar."""
        ordered = sorted(bars, key=lambda b: b.ts_event_ms)
        out: list[FactorSnapshot] = []
        for bar in ordered:
            mctx = macro_timeline.snapshot_as_of(bar.ts_avail_ms) if macro_timeline is not None else None
            out.append(self.update(bar, macro_ctx=mctx))
        return out


class FactorDirectionAuditor:
    """G2 Factor Direction Audit with D12 HAC / Phase-Averaged IC + Sparse Event Expectancy."""

    def __init__(
        self,
        min_abs_ic: float = 0.02,
        min_monotonicity_rho: float = 0.50,
        min_t_stat: float = 1.28,
        redundancy_corr_threshold: float = 0.85,
        warmup_bars: int = 45,
        min_event_count: int = 12,
        sparse_nonzero_threshold: float = 0.25,
    ) -> None:
        self.min_abs_ic = float(min_abs_ic)
        self.min_monotonicity_rho = float(min_monotonicity_rho)
        self.min_t_stat = float(min_t_stat)
        self.redundancy_corr_threshold = float(redundancy_corr_threshold)
        self.warmup_bars = int(warmup_bars)
        self.min_event_count = max(2, int(min_event_count))
        self.sparse_nonzero_threshold = float(sparse_nonzero_threshold)

    def audit(
        self,
        track_name: str,
        bars_by_symbol: dict[str, list[CanonicalBar]],
        snaps_by_symbol: dict[str, list[FactorSnapshot]],
        horizon_bars: int = 8,
        candidate_factors: list[str] | None = None,
        priors: dict[str, int] | None = None,
    ) -> FactorAuditSummary:
        """Evaluate directional alpha factors via phase-averaged HAC IC or sparse event expectancy."""
        prior_map = dict(FACTOR_ECONOMIC_PRIORS)
        if priors:
            prior_map.update(priors)
        factors = candidate_factors or [k for k, sgn in prior_map.items() if sgn != 0]

        h_bars = max(1, int(horizon_bars))

        # Collect ALL causal overlapping pairs per symbol: decision at i close -> T+1 open -> T+1+H close
        sym_pairs: dict[str, dict[str, list[tuple[int, float, float]]]] = {f: {} for f in factors}
        pooled_f: dict[str, list[float]] = {f: [] for f in factors}
        pooled_r: dict[str, list[float]] = {f: [] for f in factors}
        seg_lens_by_f: dict[str, list[int]] = {f: [] for f in factors}
        chunk_ics: dict[str, list[float]] = {f: [] for f in factors}

        for sym, bars in sorted(bars_by_symbol.items()):
            snaps = snaps_by_symbol.get(sym) or []
            n = min(len(bars), len(snaps))
            if n <= self.warmup_bars + h_bars + 2:
                continue
            sym_f: dict[str, list[float]] = {f: [] for f in factors}
            sym_r: dict[str, list[float]] = {f: [] for f in factors}
            for f in factors:
                sym_pairs[f][sym] = []

            for i in range(self.warmup_bars, n - h_bars - 1):
                entry_px = bars[i + 1].open
                exit_px = bars[i + 1 + h_bars].close
                if entry_px <= 0.0 or exit_px <= 0.0:
                    continue
                fwd_bps = (exit_px / entry_px - 1.0) * 10_000.0
                if not math.isfinite(fwd_bps):
                    continue
                snap = snaps[i]
                for f in factors:
                    val = snap.get(f, 0.0)
                    if math.isfinite(val):
                        sym_pairs[f][sym].append((i, val, fwd_bps))
                        pooled_f[f].append(val)
                        pooled_r[f].append(fwd_bps)
                        sym_f[f].append(val)
                        sym_r[f].append(fwd_bps)

            for f in factors:
                cnt = len(sym_f[f])
                if cnt > 0:
                    seg_lens_by_f[f].append(cnt)
                if cnt >= 24:
                    n_chunks = 4 if cnt >= 80 else 2
                    step_c = cnt // n_chunks
                    for c_i in range(n_chunks):
                        a = c_i * step_c
                        b = cnt if c_i == n_chunks - 1 else (c_i + 1) * step_c
                        if b - a >= 10:
                            chunk_ics[f].append(spearman_corr(sym_f[f][a:b], sym_r[f][a:b]))

        results: list[FactorAuditResult] = []
        passed_names: list[str] = []

        for f in factors:
            xs = pooled_f[f]
            ys = pooled_r[f]
            exp_sign = prior_map.get(f, 0)
            n_obs = len(xs)
            nonzero_n = sum(1 for x in xs if abs(x) > 1e-12)
            distinct_vals = len(set(round(x, 8) for x in xs))

            if n_obs < 20 or nonzero_n == 0 or distinct_vals < 2:
                results.append(
                    FactorAuditResult(
                        factor_name=f,
                        expected_sign=exp_sign,
                        sample_count=n_obs,
                        spearman_ic=0.0,
                        t_stat=0.0,
                        icir=0.0,
                        hit_rate=0.0,
                        quintile_means_bps=[],
                        monotonicity_rho=0.0,
                        sign_ok=False,
                        monotonic_ok=False,
                        passed_g2=False,
                        detail="Insufficient sample or constant/zero-variance factor (hard FAIL).",
                    )
                )
                continue

            is_sparse = ((nonzero_n / float(n_obs)) < self.sparse_nonzero_threshold) or (distinct_vals == 2)

            if is_sparse:
                # Branch A: Non-overlapping conditional sparse event expectancy (D12)
                dedup_sigs: list[float] = []
                dedup_rets: list[float] = []
                pos_rets: list[float] = []
                neg_rets: list[float] = []
                zero_rets: list[float] = []
                for sym, seq in sorted(sym_pairs[f].items()):
                    next_allowed_i = -1
                    for bar_i, val, fwd_bps in seq:
                        if abs(val) <= 1e-12:
                            zero_rets.append(fwd_bps)
                            continue
                        if bar_i < next_allowed_i:
                            continue
                        dedup_sigs.append(val)
                        dedup_rets.append(fwd_bps)
                        if val > 0.0:
                            pos_rets.append(fwd_bps)
                        else:
                            neg_rets.append(fwd_bps)
                        next_allowed_i = bar_i + h_bars + 1

                ev_rep = compute_sparse_event_expectancy(
                    dedup_sigs,
                    dedup_rets,
                    signal_name=f,
                    expected_sign=exp_sign,
                    event_threshold=1e-12,
                    horizon_bars=1,  # Already strictly cooldown-deduplicated per symbol
                    min_events=self.min_event_count,
                    min_mean_fwd_ret=0.0,
                    min_t_stat=self.min_t_stat,
                    min_win_rate=0.45,
                    returns_in_bps=True,
                )
                raw_ic = spearman_corr(xs, ys)
                q_states = [
                    round(_fmean(neg_rets), 2),
                    round(_fmean(zero_rets), 2),
                    round(_fmean(pos_rets), 2),
                ]
                ev_icir = (ev_rep.mean_fwd_ret / ev_rep.std_fwd_ret) if ev_rep.std_fwd_ret > 1e-12 else 0.0
                sign_ok = (
                    exp_sign != 0
                    and ev_rep.n_events >= self.min_event_count
                    and ev_rep.mean_fwd_ret > 0.0
                    and ev_rep.t_stat >= self.min_t_stat
                )
                monotonic_ok = sign_ok and (ev_rep.win_rate >= 0.45)
                passed = sign_ok and monotonic_ok
                if passed:
                    passed_names.append(f)

                results.append(
                    FactorAuditResult(
                        factor_name=f,
                        expected_sign=exp_sign,
                        sample_count=n_obs,
                        spearman_ic=round(raw_ic, 4),
                        t_stat=round(ev_rep.t_stat, 3),
                        icir=round(ev_icir, 3),
                        hit_rate=round(ev_rep.win_rate, 3),
                        quintile_means_bps=q_states,
                        monotonicity_rho=1.0 if monotonic_ok else 0.0,
                        sign_ok=sign_ok,
                        monotonic_ok=monotonic_ok,
                        passed_g2=passed,
                        detail=(
                            f"PASS G2 sparse_event (n={ev_rep.n_events}, mean={ev_rep.mean_fwd_ret_bps:+.1f}bp, t={ev_rep.t_stat:.2f})"
                            if passed
                            else f"FAIL G2 sparse_event (n={ev_rep.n_events}, mean={ev_rep.mean_fwd_ret_bps:+.1f}bp, t={ev_rep.t_stat:.2f})"
                        ),
                        hac_t_stat=round(ev_rep.t_stat, 3),
                        phase_ic_std=0.0,
                        audit_mode="sparse_event",
                        event_count=ev_rep.n_events,
                        event_mean_return_bps=ev_rep.mean_fwd_ret_bps,
                        event_t_stat=round(ev_rep.t_stat, 3),
                    )
                )
                continue

            # Branch B: Continuous factor -> Phase-averaged Rank IC + Segment-isolated Newey-West HAC t-stat
            hac_rep = compute_hac_factor_ic(
                xs,
                ys,
                horizon_bars=h_bars,
                factor_name=f,
                segment_lengths=seg_lens_by_f[f] if seg_lens_by_f[f] else None,
            )
            ic = hac_rep.phase_avg_ic
            if hac_rep.hac_se > 1e-12:
                hac_t = ic / hac_rep.hac_se
            elif abs(ic) > 1e-12:
                hac_t = math.copysign(max(abs(hac_rep.hac_t_stat), 50.0), ic)
            else:
                hac_t = 0.0

            ics = chunk_ics[f]
            if len(ics) >= 2:
                ic_mu = _fmean(ics)
                ic_sd = _fpstdev(ics, ic_mu)
                icir = ic_mu / ic_sd if ic_sd > 1e-6 else (ic_mu / 0.1)
                hit_rate = sum(1 for v in ics if (v * exp_sign) > 0.0) / len(ics)
            else:
                icir = ic / 0.1
                hit_rate = 1.0 if (ic * exp_sign) > 0.0 else 0.0

            # Q1..Q5 quintile means across ALL overlapping pairs (phase-invariant monotonicity)
            pairs = sorted(zip(xs, ys), key=lambda p: p[0])
            q_means: list[float] = []
            for q in range(5):
                lo_i = int(round(q * n_obs / 5.0))
                hi_i = int(round((q + 1) * n_obs / 5.0))
                slice_y = [p[1] for p in pairs[lo_i:hi_i]]
                q_means.append(round(_fmean(slice_y), 2) if slice_y else 0.0)

            mono_rho = spearman_corr([1.0, 2.0, 3.0, 4.0, 5.0], q_means)
            if exp_sign > 0:
                q_spread_ok = (q_means[4] > q_means[0]) and (q_means[4] >= q_means[2] * 0.5 or q_means[4] >= q_means[3])
            elif exp_sign < 0:
                q_spread_ok = (q_means[0] > q_means[4]) and (q_means[0] >= q_means[2] * 0.5 or q_means[0] >= q_means[1])
            else:
                q_spread_ok = False

            sign_ok = (exp_sign != 0) and ((ic * exp_sign) >= self.min_abs_ic) and ((hac_t * exp_sign) >= self.min_t_stat)
            monotonic_ok = (exp_sign != 0) and ((mono_rho * exp_sign) >= self.min_monotonicity_rho) and q_spread_ok
            passed = sign_ok and monotonic_ok
            if passed:
                passed_names.append(f)

            results.append(
                FactorAuditResult(
                    factor_name=f,
                    expected_sign=exp_sign,
                    sample_count=n_obs,
                    spearman_ic=round(ic, 4),
                    t_stat=round(hac_t, 3),
                    icir=round(icir, 3),
                    hit_rate=round(hit_rate, 3),
                    quintile_means_bps=q_means,
                    monotonicity_rho=round(mono_rho, 3),
                    sign_ok=sign_ok,
                    monotonic_ok=monotonic_ok,
                    passed_g2=passed,
                    detail=(
                        f"PASS G2 (phase_ic={ic:+.4f}, hac_t={hac_t:.2f}, nw_lags={hac_rep.nw_lags})"
                        if passed
                        else f"FAIL G2 (sign_ok={sign_ok}, mono_ok={monotonic_ok}, hac_t={hac_t:.2f})"
                    ),
                    hac_t_stat=round(hac_t, 3),
                    phase_ic_std=round(hac_rep.phase_ic_std, 4),
                    audit_mode="continuous_ic",
                )
            )

        # RD-Agent(Q) pairwise redundancy pruning among G2-passing factors (keep higher |ICIR|+|IC|)
        res_by_name = {r.factor_name: r for r in results}
        ranked_pass = sorted(
            passed_names,
            key=lambda fn: abs(res_by_name[fn].icir) + abs(res_by_name[fn].spearman_ic),
            reverse=True,
        )
        selected: list[str] = []
        pruned_pairs: list[dict[str, Any]] = []
        for fn in ranked_pass:
            is_dup = False
            for kept in selected:
                c_pair = abs(spearman_corr(pooled_f[fn], pooled_f[kept]))
                if c_pair >= self.redundancy_corr_threshold:
                    res_by_name[fn].pruned_as_redundant = True
                    res_by_name[fn].redundant_with = kept
                    res_by_name[fn].detail = f"Pruned redundant with {kept} (|corr|={c_pair:.3f})"
                    pruned_pairs.append({"dropped": fn, "kept": kept, "abs_corr": round(c_pair, 4)})
                    is_dup = True
                    break
            if not is_dup:
                selected.append(fn)

        return FactorAuditSummary(
            track_name=track_name,
            horizon_bars=h_bars,
            total_factors_evaluated=len(results),
            passed_g2_factors=passed_names,
            selected_non_redundant_factors=selected,
            pruned_redundant_pairs=pruned_pairs,
            results=results,
        )
