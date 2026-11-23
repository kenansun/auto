"""Single-Kernel Cross-Asset Factor Engine & G2 Direction/Redundancy Auditor.

Synthesizes:
- NautilusTrader single-kernel parity: `StreamingFactorEngine.update(bar)` is the ONE
  implementation used by both historical batch evaluation and live bar streaming.
- Rob Carver (`pysystemtrade`): vol-normalized EWMAC forecasts scaled to [-20, +20] with
  published forecast scalar discipline (zero curve-fitting of raw indicator scales).
- QuantPedia (2026): Cross-Asset Gold-Treasury joint momentum state (`Gold ↑ & IEF ↑`
  = falling real rates / easing tailwind) strictly Point-in-Time aligned via `ts_avail_ms`.
- BAISYS_QUANT + CC-20261117-003 G2 Gate: Spearman Rank IC sign audit against declared
  economic hypotheses + Q1..Q5 quintile monotonicity + pure-noise hard rejection.
- Microsoft RD-Agent(Q): Pairwise factor correlation redundancy pruning (|corr| >= threshold).
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from alphaloop.platform.assets import TradingCalendar
from alphaloop.platform.data_lake import CanonicalBar

logger = logging.getLogger(__name__)

# Economic hypothesis prior for directional factors:
#  +1 = higher factor value should predict higher forward return
#  -1 = higher factor value should predict lower forward return (e.g. mean-reversion / crowding)
#   0 = non-directional regime/state variable (volatility, efficiency, Hurst)
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

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.values.get(key, default))


@dataclass
class FactorAuditResult:
    """Single-factor G2 Direction & Monotonicity Audit outcome."""

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


def _ranks(values: list[float]) -> list[float]:
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


def spearman_corr(xs: list[float], ys: list[float]) -> float:
    """Pure-stdlib Spearman rank correlation; returns 0.0 on constant/degenerate vectors."""
    n = min(len(xs), len(ys))
    if n < 5:
        return 0.0
    rx = _ranks(xs[:n])
    ry = _ranks(ys[:n])
    mx = sum(rx) / n
    my = sum(ry) / n
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return _clamp(cov / math.sqrt(vx * vy), -1.0, 1.0)


class MacroRegimeTimeline:
    """Strictly Point-in-Time causal macro regime provider (Gold + Treasury + DXY + TIP + VIX).

    Implements QuantPedia (2026) Cross-Asset Price-Based Regimes for Gold:
      - Joint (Gold Mom > 0 & IEF Treasury Mom > 0) => State +1 (Falling Real Rates / Easing)
      - Joint (Gold Mom < 0 & IEF Treasury Mom < 0) => State -1 (Rising Real Rates / Tightening)
      - Mixed states => State 0
    Combined with DXY inverse momentum, TIP real-yield momentum, and VIX stress z-score.
    Every lookup for a target bar uses ONLY macro bars with `ts_avail_ms <= bar.ts_avail_ms`.
    """

    def __init__(self, macro_bars_by_symbol: dict[str, list[CanonicalBar]]) -> None:
        self._series: dict[str, list[tuple[int, float]]] = {}
        for sym, bars in macro_bars_by_symbol.items():
            ordered = sorted(bars, key=lambda b: b.ts_avail_ms)
            self._series[sym] = [(b.ts_avail_ms, b.close) for b in ordered if b.close > 0]
        self._cursors: dict[str, int] = {sym: 0 for sym in self._series}
        self._last_ts: int = -1

    def _window_at(self, sym: str, as_of_ms: int, need: int) -> list[float]:
        pts = self._series.get(sym)
        if not pts:
            return []
        if as_of_ms < self._last_ts:
            # Non-monotonic query: reset cursor safely so lookups remain deterministic
            idx = 0
            while idx < len(pts) and pts[idx][0] <= as_of_ms:
                idx += 1
        else:
            idx = self._cursors.get(sym, 0)
            while idx < len(pts) and pts[idx][0] <= as_of_ms:
                idx += 1
            self._cursors[sym] = idx
        if idx <= 0:
            return []
        start = max(0, idx - need)
        return [pts[k][1] for k in range(start, idx)]

    @staticmethod
    def _mom_sign_strength(closes: list[float], windows: tuple[int, ...] = (21, 63, 126, 252)) -> float:
        """Multi-horizon trailing return score in [-1, +1] using strictly past closes."""
        if len(closes) < 22:
            return 0.0
        cur = closes[-1]
        scores: list[float] = []
        for w in windows:
            if len(closes) > w and closes[-1 - w] > 0:
                ret = cur / closes[-1 - w] - 1.0
                # Smooth tanh normalization around 5% move
                scores.append(math.tanh(ret / 0.05))
        return sum(scores) / len(scores) if scores else 0.0

    def snapshot_as_of(self, as_of_ms: int) -> dict[str, float]:
        """Return causal macro regime vector strictly visible at `as_of_ms`."""
        gold_sym = "GC=F" if "GC=F" in self._series else ("GLD" if "GLD" in self._series else "")
        gold_c = self._window_at(gold_sym, as_of_ms, 260) if gold_sym else []
        ief_c = self._window_at("IEF", as_of_ms, 260)
        dxy_c = self._window_at("DX-Y.NYB", as_of_ms, 260)
        tip_c = self._window_at("TIP", as_of_ms, 260)
        vix_c = self._window_at("^VIX", as_of_ms, 130)
        self._last_ts = as_of_ms

        gold_mom = self._mom_sign_strength(gold_c)
        ief_mom = self._mom_sign_strength(ief_c)
        dxy_mom = self._mom_sign_strength(dxy_c)
        tip_mom = self._mom_sign_strength(tip_c)

        if gold_mom > 0.05 and ief_mom > 0.05:
            gt_state = 1.0   # QuantPedia State 1: Gold up + Treasuries up (falling real yields)
        elif gold_mom < -0.05 and ief_mom < -0.05:
            gt_state = -1.0  # QuantPedia State 4: Gold down + Treasuries down (rising real yields)
        else:
            gt_state = 0.0

        dxy_tailwind = -dxy_mom  # Weak USD = Gold & Risk asset tailwind
        real_rate_tailwind = 0.5 * ief_mom + 0.5 * tip_mom

        vix_stress_z = 0.0
        if len(vix_c) >= 25:
            mu = statistics.mean(vix_c[:-1])
            sd = statistics.pstdev(vix_c[:-1])
            if sd > 1e-6:
                vix_stress_z = _clamp((vix_c[-1] - mu) / sd, -4.0, 5.0)

        # Interpretable composite Gold macro score in [-1, +1]
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
        }


class StreamingFactorEngine:
    """Single-kernel strictly causal multi-factor calculator (exact batch == streaming parity)."""

    def __init__(self, calendar: TradingCalendar, timeframe_h: float) -> None:
        self.calendar = calendar
        self.timeframe_h = float(timeframe_h)
        self.bars_per_year = calendar.bars_per_year(timeframe_h)
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
        self._closes: deque[float] = deque(maxlen=130)
        self._highs: deque[float] = deque(maxlen=65)
        self._lows: deque[float] = deque(maxlen=65)
        self._vols: deque[float] = deque(maxlen=65)
        self._log_rets: deque[float] = deque(maxlen=65)
        self._atr_pcts: deque[float] = deque(maxlen=65)
        self._carry_aprs: deque[float] = deque(maxlen=65)

    def update(
        self,
        bar: CanonicalBar,
        macro_ctx: dict[str, float] | None = None,
    ) -> FactorSnapshot:
        """Consume one closed bar and emit its strictly causal FactorSnapshot."""
        self._bar_count += 1
        c = bar.close
        h = bar.high
        l = bar.low

        # Capture strictly PRIOR channel highs/lows BEFORE appending current bar (zero lookahead!)
        prior_high_40 = max(list(self._highs)[-40:]) if len(self._highs) >= 2 else h
        prior_low_40 = min(list(self._lows)[-40:]) if len(self._lows) >= 2 else l
        prior_high_20 = max(list(self._highs)[-20:]) if len(self._highs) >= 2 else h
        prior_low_20 = min(list(self._lows)[-20:]) if len(self._lows) >= 2 else l

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
            prev_c = self._prev_bar.close
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            log_ret = math.log(c / prev_c) if prev_c > 0 else 0.0
            self._log_rets.append(log_ret)

            # Wilder RSI(14)
            delta = c - prev_c
            gain = max(0.0, delta)
            loss = max(0.0, -delta)
            if self._avg_gain14 is None:
                self._gain_seed.append(gain)
                self._loss_seed.append(loss)
                if len(self._gain_seed) == 14:
                    self._avg_gain14 = sum(self._gain_seed) / 14.0
                    self._avg_loss14 = sum(self._loss_seed) / 14.0
            else:
                self._avg_gain14 = ((self._avg_gain14 * 13.0) + gain) / 14.0
                self._avg_loss14 = ((self._avg_loss14 * 13.0) + loss) / 14.0

            # Wilder Directional Movement & ADX(14)
            up_move = h - self._prev_bar.high
            dn_move = self._prev_bar.low - l
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
                        self._adx14 = sum(self._dx_buf) / 14.0
                else:
                    self._adx14 = ((self._adx14 * 13.0) + dx) / 14.0
            else:
                di_spread = 0.0

        if self._atr14 is None:
            self._atr14 = tr
        else:
            self._atr14 = ((self._atr14 * 13.0) + tr) / 14.0

        atr_val = max(self._atr14, c * 1e-5)
        atr_pct = (atr_val / c) * 100.0 if c > 0 else 0.0
        self._atr_pcts.append(atr_pct)

        # Carry APR annualized
        carry_apr = (bar.carry_rate_period / max(0.01, bar.timeframe_h)) * 24.0 * 365.0 * 100.0
        self._carry_aprs.append(carry_apr)

        # Append current bar to rolling buffers
        self._closes.append(c)
        self._highs.append(h)
        self._lows.append(l)
        self._vols.append(bar.volume_usd)
        self._prev_bar = bar

        closes_list = list(self._closes)
        rets_list = list(self._log_rets)

        # 3. Volatility & Squeeze
        if len(rets_list) >= 10:
            r20 = rets_list[-20:]
            bar_vol = statistics.pstdev(r20)
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
            sma20 = sum(w20) / 20.0
            sd20 = statistics.pstdev(w20)
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
        # Forecast Diversification Multiplier (FDM ~= 1.12 for 3 correlated trend speeds)
        trend_composite = _clamp(
            1.12 * (0.30 * ewmac_vals["ewmac_8_32"] + 0.40 * ewmac_vals["ewmac_16_64"] + 0.30 * ewmac_vals["ewmac_32_128"]),
            -20.0,
            20.0,
        )

        # 5. Donchian normalized channel positions [-1, +1] using strictly PRIOR high/low!
        def _donch_pos(hi: float, lo: float) -> float:
            span = hi - lo
            if span <= 1e-12:
                return 0.0
            return _clamp(2.0 * ((c - lo) / span) - 1.0, -1.5, 1.5)

        donchian_pos_20 = _donch_pos(prior_high_20, prior_low_20)
        donchian_pos_40 = _donch_pos(prior_high_40, prior_low_40)

        # Squeeze breakout directional signal (CC-20261117-003 §八 D2: compression -> breakout)
        squeeze_breakout_dir = 0.0
        if squeeze_ratio_20 < 1.05:
            if c > prior_high_20:
                squeeze_breakout_dir = 1.0
            elif c < prior_low_20:
                squeeze_breakout_dir = -1.0

        # 6. Kaufman Efficiency Ratio (20)
        if len(closes_list) >= 21:
            net_move = abs(closes_list[-1] - closes_list[-21])
            path_len = sum(abs(closes_list[i] - closes_list[i - 1]) for i in range(len(closes_list) - 20, len(closes_list)))
            kaufman_er_20 = _clamp(net_move / path_len, 0.0, 1.0) if path_len > 1e-12 else 0.0
        else:
            kaufman_er_20 = 0.3

        # 7. Time-series momentum (vol-normalized)
        def _tsmom(lb: int) -> float:
            if len(closes_list) <= lb or closes_list[-1 - lb] <= 0:
                return 0.0
            ret = math.log(c / closes_list[-1 - lb])
            denom = max(bar_vol * math.sqrt(lb), 1e-4)
            return _clamp(ret / denom, -4.0, 4.0)

        tsmom_20 = _tsmom(20)
        tsmom_60 = _tsmom(60)

        # 8. Wilder RSI normalized [-1, +1]
        if self._avg_gain14 is not None and self._avg_loss14 is not None:
            if self._avg_loss14 <= 1e-12:
                rsi = 100.0 if self._avg_gain14 > 0 else 50.0
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
            mx0 = sum(x0) / len(x0)
            mx1 = sum(x1) / len(x1)
            vx0 = sum((a - mx0) ** 2 for a in x0)
            vx1 = sum((b - mx1) ** 2 for b in x1)
            if vx0 > 1e-12 and vx1 > 1e-12:
                autocorr_1_20 = _clamp(sum((a - mx0) * (b - mx1) for a, b in zip(x0, x1)) / math.sqrt(vx0 * vx1), -1.0, 1.0)
            var1 = statistics.pvariance(r20)
            r2_bars = [r20[i] + r20[i + 1] for i in range(0, len(r20) - 1, 2)]
            if var1 > 1e-12 and len(r2_bars) >= 4:
                var2 = statistics.pvariance(r2_bars)
                variance_ratio_20 = _clamp(var2 / (2.0 * var1), 0.2, 2.5)

        hurst_proxy_40 = 0.5
        if len(rets_list) >= 40:
            r40 = rets_list[-40:]
            mu40 = sum(r40) / 40.0
            sd40 = statistics.pstdev(r40)
            if sd40 > 1e-12:
                cum = 0.0
                cums: list[float] = []
                for x in r40:
                    cum += (x - mu40)
                    cums.append(cum)
                rs_val = (max(cums) - min(cums)) / sd40
                if rs_val > 1e-9:
                    hurst_proxy_40 = _clamp(math.log(rs_val) / math.log(40.0), 0.1, 0.95)

        # 10. Relative Volume & Carry stats
        if len(self._vols) >= 10:
            avg_v = sum(list(self._vols)[-20:]) / min(20, len(self._vols))
            rel_vol = _clamp(bar.volume_usd / avg_v, 0.0, 10.0) if avg_v > 1e-9 else 1.0
        else:
            rel_vol = 1.0

        carry_list = list(self._carry_aprs)
        carry_win = max(1, int(round(24.0 / max(1.0, self.timeframe_h))))
        carry_apr_24h = sum(carry_list[-carry_win:]) / min(carry_win, len(carry_list))
        if len(carry_list) >= 15:
            cmu = statistics.mean(carry_list)
            csd = statistics.pstdev(carry_list)
            carry_zscore_60 = _clamp((carry_apr - cmu) / csd, -4.0, 4.0) if csd > 1e-6 else 0.0
        else:
            carry_zscore_60 = 0.0

        values: dict[str, float] = {
            **ewmac_vals,
            "trend_composite": trend_composite,
            "donchian_pos_20": donchian_pos_20,
            "donchian_pos_40": donchian_pos_40,
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
    """G2 Factor Direction Audit + Q1..Q5 Monotonicity + RD-Agent Redundancy Pruning."""

    def __init__(
        self,
        min_abs_ic: float = 0.02,
        min_monotonicity_rho: float = 0.50,
        min_t_stat: float = 1.28,
        redundancy_corr_threshold: float = 0.85,
        warmup_bars: int = 45,
    ) -> None:
        self.min_abs_ic = float(min_abs_ic)
        self.min_monotonicity_rho = float(min_monotonicity_rho)
        self.min_t_stat = float(min_t_stat)
        self.redundancy_corr_threshold = float(redundancy_corr_threshold)
        self.warmup_bars = int(warmup_bars)

    def audit(
        self,
        track_name: str,
        bars_by_symbol: dict[str, list[CanonicalBar]],
        snaps_by_symbol: dict[str, list[FactorSnapshot]],
        horizon_bars: int = 8,
        candidate_factors: list[str] | None = None,
        priors: dict[str, int] | None = None,
    ) -> FactorAuditSummary:
        """Evaluate directional alpha factors strictly against non-overlapping forward returns."""
        prior_map = dict(FACTOR_ECONOMIC_PRIORS)
        if priors:
            prior_map.update(priors)
        factors = candidate_factors or [k for k, sgn in prior_map.items() if sgn != 0]

        # Collect pooled (factor_val, forward_ret_bps) pairs using strictly causal T+1 open -> T+1+H close
        pooled_f: dict[str, list[float]] = {f: [] for f in factors}
        pooled_r: dict[str, list[float]] = {f: [] for f in factors}
        chunk_ics: dict[str, list[float]] = {f: [] for f in factors}

        step = max(1, horizon_bars)
        for sym, bars in sorted(bars_by_symbol.items()):
            snaps = snaps_by_symbol.get(sym) or []
            n = min(len(bars), len(snaps))
            if n <= self.warmup_bars + horizon_bars + 2:
                continue
            sym_f: dict[str, list[float]] = {f: [] for f in factors}
            sym_r: dict[str, list[float]] = {f: [] for f in factors}
            # Non-overlapping forward samples from warmup onward: decision at i close, enter at i+1 open, exit at i+1+H close
            for i in range(self.warmup_bars, n - horizon_bars - 1, step):
                entry_px = bars[i + 1].open
                exit_px = bars[i + 1 + horizon_bars].close
                if entry_px <= 0 or exit_px <= 0:
                    continue
                fwd_bps = (exit_px / entry_px - 1.0) * 10_000.0
                snap = snaps[i]
                for f in factors:
                    val = snap.get(f, 0.0)
                    if math.isfinite(val) and math.isfinite(fwd_bps):
                        pooled_f[f].append(val)
                        pooled_r[f].append(fwd_bps)
                        sym_f[f].append(val)
                        sym_r[f].append(fwd_bps)
            # Rolling / per-symbol IC chunks for ICIR stability
            for f in factors:
                if len(sym_f[f]) >= 12:
                    mid = len(sym_f[f]) // 2
                    for a, b in ((0, mid), (mid, len(sym_f[f]))):
                        if b - a >= 6:
                            chunk_ics[f].append(spearman_corr(sym_f[f][a:b], sym_r[f][a:b]))

        results: list[FactorAuditResult] = []
        passed_names: list[str] = []

        for f in factors:
            xs = pooled_f[f]
            ys = pooled_r[f]
            exp_sign = prior_map.get(f, 0)
            n_obs = len(xs)
            if n_obs < 20 or len(set(round(x, 8) for x in xs)) < 3:
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

            ic = spearman_corr(xs, ys)
            denom_t = max(1e-9, 1.0 - ic * ic)
            t_stat = ic * math.sqrt(max(1.0, float(n_obs - 2)) / denom_t)
            ics = chunk_ics[f]
            if len(ics) >= 2:
                ic_mu = statistics.mean(ics)
                ic_sd = statistics.pstdev(ics)
                icir = ic_mu / ic_sd if ic_sd > 1e-6 else (ic_mu / 0.1)
                hit_rate = sum(1 for v in ics if (v * exp_sign) > 0) / len(ics)
            else:
                icir = ic / 0.1
                hit_rate = 1.0 if (ic * exp_sign) > 0 else 0.0

            # Q1 .. Q5 quintile mean returns
            pairs = sorted(zip(xs, ys), key=lambda p: p[0])
            q_means: list[float] = []
            for q in range(5):
                lo_i = int(round(q * n_obs / 5.0))
                hi_i = int(round((q + 1) * n_obs / 5.0))
                slice_y = [p[1] for p in pairs[lo_i:hi_i]]
                q_means.append(round(statistics.mean(slice_y), 2) if slice_y else 0.0)

            mono_rho = spearman_corr([1.0, 2.0, 3.0, 4.0, 5.0], q_means)
            # Reject hump-shaped / tail-collapsed quintile profiles (e.g. Q3 peak, Q5 collapse)
            if exp_sign > 0:
                q_spread_ok = (q_means[4] > q_means[0]) and (q_means[4] >= q_means[2] * 0.5 or q_means[4] >= q_means[3])
            elif exp_sign < 0:
                q_spread_ok = (q_means[0] > q_means[4]) and (q_means[0] >= q_means[2] * 0.5 or q_means[0] >= q_means[1])
            else:
                q_spread_ok = False
            # Strict sign + statistical significance check against economic hypothesis (never flip sign!)
            sign_ok = (exp_sign != 0) and ((ic * exp_sign) >= self.min_abs_ic) and ((t_stat * exp_sign) >= self.min_t_stat)
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
                    t_stat=round(t_stat, 3),
                    icir=round(icir, 3),
                    hit_rate=round(hit_rate, 3),
                    quintile_means_bps=q_means,
                    monotonicity_rho=round(mono_rho, 3),
                    sign_ok=sign_ok,
                    monotonic_ok=monotonic_ok,
                    passed_g2=passed,
                    detail="PASS G2" if passed else f"FAIL G2 (sign_ok={sign_ok}, mono_ok={monotonic_ok}, t={t_stat:.2f})",
                )
            )

        # RD-Agent(Q) pairwise redundancy pruning among G2-passing factors (keep higher |ICIR|)
        res_by_name = {r.factor_name: r for r in results}
        ranked_pass = sorted(passed_names, key=lambda fn: abs(res_by_name[fn].icir) + abs(res_by_name[fn].spearman_ic), reverse=True)
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
            horizon_bars=horizon_bars,
            total_factors_evaluated=len(results),
            passed_g2_factors=passed_names,
            selected_non_redundant_factors=selected,
            pruned_redundant_pairs=pruned_pairs,
            results=results,
        )
