"""Stateless Point-in-Time Gold Macro Regime Engine (Track G, Feature F-16).

Consumes all 7 Gold macro drivers strictly causally via stateless `bisect.bisect_right`
over `ts_avail_ms` (resolving v3 defect D6 so out-of-order, repeated, or concurrent
timestamp queries are guaranteed bit-identical with zero lookahead):
  1. US Treasury 10Y TIPS Real Yield (`MacroYieldPoint.real_10y`, fallback `^TNX`)
  2. QuantPedia `Gold x IEF` & `TIP` real-rate / bond-inflation joint trend
  3. US Dollar Index inverse trend (`DX-Y.NYB`, fallback `UUP`)
  4. Gold/Silver Ratio confirmation (`GC=F / SI=F` or `GLD / SLV`)
  5. Gold Miners relative strength (`GDX / GLD`, fallback `GDX / GC=F`)
  6. Gold implied volatility level & z-score exhaustion (`^GVZ`)
  7. Safe-haven asymmetric `^VIX` response (constructive safe-haven bid alongside
     rising Gold & falling real yields, contrasting with liquidation dash-for-cash)
"""

from __future__ import annotations

import bisect
import math
from typing import Any, Sequence

from alphaloop.platform.data_foundation.models import CanonicalBar, MacroYieldPoint


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _fmean(vals: Sequence[float]) -> float:
    return (math.fsum(vals) / len(vals)) if vals else 0.0


def _fpstdev(vals: Sequence[float], mu: float) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    var = math.fsum((x - mu) * (x - mu) for x in vals) / n
    return math.sqrt(var) if (math.isfinite(var) and var > 0.0) else 0.0


def _dedup_sorted_series(pairs: Sequence[tuple[int, float]]) -> tuple[list[int], list[float]]:
    """Sort ascending by timestamp and keep latest finite observation per timestamp."""
    if not pairs:
        return [], []
    ordered = sorted(pairs, key=lambda item: item[0])
    ts_out: list[int] = []
    val_out: list[float] = []
    for ts_ms, v in ordered:
        if not math.isfinite(v):
            continue
        if ts_out and ts_out[-1] == ts_ms:
            val_out[-1] = v
        else:
            ts_out.append(ts_ms)
            val_out.append(v)
    return ts_out, val_out


def _build_pit_ratio_series(
    num_ts: list[int],
    num_vals: list[float],
    den_ts: list[int],
    den_vals: list[float],
) -> tuple[list[int], list[float]]:
    """Build strictly causal ratio series (num / den) aligned at union of PIT availability."""
    if not num_ts or not den_ts:
        return [], []
    all_ts = sorted(set(num_ts) | set(den_ts))
    ratio_ts: list[int] = []
    ratio_vals: list[float] = []
    for ts in all_ts:
        i_n = bisect.bisect_right(num_ts, ts) - 1
        i_d = bisect.bisect_right(den_ts, ts) - 1
        if i_n < 0 or i_d < 0:
            continue
        nv = num_vals[i_n]
        dv = den_vals[i_d]
        if dv > 1e-12 and nv > 1e-12:
            ratio_ts.append(ts)
            ratio_vals.append(nv / dv)
    return ratio_ts, ratio_vals


class GoldMacroEngine:
    """100% stateless Point-in-Time Gold & Macro regime engine (bisect_right, zero cursors)."""

    def __init__(
        self,
        macro_bars_by_symbol: dict[str, Sequence[CanonicalBar]] | None = None,
        *,
        bars_by_symbol: dict[str, Sequence[CanonicalBar]] | None = None,
        yield_curve: Sequence[MacroYieldPoint] | None = None,
        yield_points: Sequence[MacroYieldPoint] | None = None,
    ) -> None:
        merged_bars: dict[str, Sequence[CanonicalBar]] = {}
        if bars_by_symbol:
            merged_bars.update(bars_by_symbol)
        if macro_bars_by_symbol:
            merged_bars.update(macro_bars_by_symbol)

        self._ts_avail: dict[str, list[int]] = {}
        self._closes: dict[str, list[float]] = {}
        for sym, bars in merged_bars.items():
            pairs = [
                (int(b.ts_avail_ms), float(b.close))
                for b in bars
                if math.isfinite(float(b.close)) and float(b.close) > 0.0
            ]
            ts_list, close_list = _dedup_sorted_series(pairs)
            if ts_list:
                self._ts_avail[sym] = ts_list
                self._closes[sym] = close_list

        raw_yields = yield_curve if yield_curve else (yield_points if yield_points else ())
        y_pairs: list[tuple[int, float]] = []
        for pt in raw_yields:
            r10 = pt.real_10y
            if r10 is None:
                for fallback in (pt.real_7y, pt.real_5y, pt.real_20y, pt.real_30y):
                    if fallback is not None:
                        r10 = fallback
                        break
            if r10 is not None and math.isfinite(float(r10)):
                y_pairs.append((int(pt.ts_avail_ms), float(r10)))
        self._real_10y_ts, self._real_10y_vals = _dedup_sorted_series(y_pairs)

        # Driver 4: PIT-aligned Gold / Silver Ratio (GC=F / SI=F, fallback GLD / SLV)
        self._gold_sym = (
            "GC=F" if "GC=F" in self._ts_avail else ("GLD" if "GLD" in self._ts_avail else "")
        )
        silver_sym = (
            "SI=F" if "SI=F" in self._ts_avail else ("SLV" if "SLV" in self._ts_avail else "")
        )
        if self._gold_sym and silver_sym:
            self._gsr_ts, self._gsr_vals = _build_pit_ratio_series(
                self._ts_avail[self._gold_sym],
                self._closes[self._gold_sym],
                self._ts_avail[silver_sym],
                self._closes[silver_sym],
            )
        else:
            self._gsr_ts, self._gsr_vals = [], []

        # Driver 5: PIT-aligned Gold Miners Relative Strength (GDX / GLD, fallback GDX / GC=F)
        miners_denom = (
            "GLD" if "GLD" in self._ts_avail else ("GC=F" if "GC=F" in self._ts_avail else "")
        )
        if "GDX" in self._ts_avail and miners_denom:
            self._miners_ts, self._miners_vals = _build_pit_ratio_series(
                self._ts_avail["GDX"],
                self._closes["GDX"],
                self._ts_avail[miners_denom],
                self._closes[miners_denom],
            )
        else:
            self._miners_ts, self._miners_vals = [], []

    @staticmethod
    def _window_from(
        ts_list: list[int],
        val_list: list[float],
        as_of_ms: int,
        need: int,
    ) -> list[float]:
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

    def latest_close_as_of(self, symbol: str, as_of_ms: int) -> float | None:
        """Return strictly causal latest close visible at `as_of_ms` (stateless bisect_right)."""
        ts_list = self._ts_avail.get(symbol)
        if not ts_list:
            return None
        idx = bisect.bisect_right(ts_list, int(as_of_ms)) - 1
        if idx < 0:
            return None
        return float(self._closes[symbol][idx])

    @staticmethod
    def _mom_score(
        vals: list[float],
        windows: tuple[int, ...] = (21, 63, 126),
        scale: float = 0.06,
    ) -> float:
        """Multi-horizon trailing return score in [-1, +1] using strictly past observations."""
        n = len(vals)
        if n < 2:
            return 0.0
        cur = vals[-1]
        scores: list[float] = []
        for w in windows:
            if n > w and vals[-1 - w] > 1e-12:
                ret = (cur / vals[-1 - w]) - 1.0
                scores.append(math.tanh(ret / scale))
        if not scores and n >= 5 and vals[0] > 1e-12:
            ret = (cur / vals[0]) - 1.0
            scores.append(math.tanh(ret / scale))
        return _clamp(_fmean(scores), -1.0, 1.0) if scores else 0.0

    @staticmethod
    def _yield_delta_score(
        yields_pct: list[float],
        windows: tuple[int, ...] = (21, 63, 126),
        scale_pct: float = 0.50,
    ) -> tuple[float, float]:
        """Return `(avg_yield_change_pct, score_in_minus1_plus1)` where +1 = falling real yields."""
        n = len(yields_pct)
        if n < 2:
            return 0.0, 0.0
        cur = yields_pct[-1]
        deltas: list[float] = []
        scores: list[float] = []
        for w in windows:
            if n > w:
                d = cur - yields_pct[-1 - w]
                deltas.append(d)
                scores.append(-math.tanh(d / scale_pct))
        if not deltas and n >= 5:
            d = cur - yields_pct[0]
            deltas.append(d)
            scores.append(-math.tanh(d / scale_pct))
        if not deltas:
            return 0.0, 0.0
        return _fmean(deltas), _clamp(_fmean(scores), -1.0, 1.0)

    @staticmethod
    def _causal_zscore(
        vals: list[float],
        min_len: int = 20,
        lo: float = -4.0,
        hi: float = 4.0,
    ) -> float:
        if len(vals) < min_len:
            return 0.0
        hist = vals[:-1]
        mu = _fmean(hist)
        sd = _fpstdev(hist, mu)
        if sd <= 1e-6:
            return 0.0
        return _clamp((vals[-1] - mu) / sd, lo, hi)

    def snapshot_as_of(self, ts_ms: int) -> dict[str, Any]:
        """Compute strictly causal 7-driver Gold macro regime snapshot at `ts_ms`."""
        as_of = int(ts_ms)
        drivers_active = 0

        # Reference Gold momentum (GC=F, fallback GLD)
        gold_c = self._window_at(self._gold_sym, as_of, 140) if self._gold_sym else []
        gold_mom = self._mom_score(gold_c, windows=(21, 63, 126), scale=0.06)

        # ------------------------------------------------------------------
        # Driver 1: US Treasury 10Y TIPS Real Yield (falling = bullish Gold)
        # ------------------------------------------------------------------
        ry_w = self._window_from(self._real_10y_ts, self._real_10y_vals, as_of, 140)
        tnx_w = self._window_at("^TNX", as_of, 140)
        real_10y_level = 0.0
        real_10y_change = 0.0
        real_yield_score = 0.0
        has_d1 = False
        if len(ry_w) >= 2:
            real_10y_level = ry_w[-1]
            real_10y_change, delta_score = self._yield_delta_score(ry_w, windows=(21, 63, 126), scale_pct=0.45)
            level_tilt = -math.tanh((real_10y_level - 1.50) / 1.75)
            real_yield_score = _clamp(0.80 * delta_score + 0.20 * level_tilt, -1.0, 1.0)
            has_d1 = True
        elif len(ry_w) == 1:
            real_10y_level = ry_w[-1]
            real_yield_score = _clamp(-0.50 * math.tanh((real_10y_level - 1.50) / 1.75), -1.0, 1.0)
            has_d1 = True
        elif len(tnx_w) >= 2:
            real_10y_level = tnx_w[-1]
            real_10y_change, real_yield_score = self._yield_delta_score(
                tnx_w, windows=(21, 63, 126), scale_pct=0.60
            )
            has_d1 = True
        if has_d1:
            drivers_active += 1

        # ------------------------------------------------------------------
        # Driver 2: QuantPedia Gold x IEF & TIP bond/inflation trend
        # ------------------------------------------------------------------
        ief_c = self._window_at("IEF", as_of, 140)
        tip_c = self._window_at("TIP", as_of, 140)
        ief_mom = self._mom_score(ief_c, windows=(21, 63, 126), scale=0.05)
        tip_mom = self._mom_score(tip_c, windows=(21, 63, 126), scale=0.05)
        has_d2 = len(ief_c) >= 2 or len(tip_c) >= 2
        if gold_mom > 0.04 and ief_mom > 0.04:
            gt_state = 1.0
        elif gold_mom < -0.04 and ief_mom < -0.04:
            gt_state = -1.0
        else:
            gt_state = _clamp(0.5 * (gold_mom + ief_mom), -1.0, 1.0)

        if len(ief_c) >= 2 and len(tip_c) >= 2:
            bond_inflation_score = _clamp(0.35 * gt_state + 0.35 * ief_mom + 0.30 * tip_mom, -1.0, 1.0)
        elif len(ief_c) >= 2:
            bond_inflation_score = _clamp(0.50 * gt_state + 0.50 * ief_mom, -1.0, 1.0)
        elif len(tip_c) >= 2:
            bond_inflation_score = _clamp(tip_mom, -1.0, 1.0)
        else:
            bond_inflation_score = 0.0
        if has_d2:
            drivers_active += 1

        # ------------------------------------------------------------------
        # Driver 3: US Dollar Trend (DX-Y.NYB, fallback UUP; falling = bullish)
        # ------------------------------------------------------------------
        dxy_sym = "DX-Y.NYB" if "DX-Y.NYB" in self._ts_avail else ("UUP" if "UUP" in self._ts_avail else "")
        dxy_c = self._window_at(dxy_sym, as_of, 140) if dxy_sym else []
        dxy_mom = self._mom_score(dxy_c, windows=(21, 63, 126), scale=0.045)
        dxy_score = _clamp(-dxy_mom, -1.0, 1.0) if len(dxy_c) >= 2 else 0.0
        has_d3 = len(dxy_c) >= 2
        if has_d3:
            drivers_active += 1

        # ------------------------------------------------------------------
        # Driver 4: Gold/Silver Ratio (GC=F / SI=F or GLD / SLV)
        # Falling GSR (Silver confirming Gold expansion) => bullish breadth
        # ------------------------------------------------------------------
        gsr_c = self._window_from(self._gsr_ts, self._gsr_vals, as_of, 140)
        gsr_trend = self._mom_score(gsr_c, windows=(21, 63, 126), scale=0.06)
        gsr_z = self._causal_zscore(gsr_c, min_len=20, lo=-4.0, hi=4.0)
        has_d4 = len(gsr_c) >= 2
        gsr_score = (
            _clamp(-0.75 * gsr_trend - 0.25 * math.tanh(gsr_z / 2.5), -1.0, 1.0)
            if has_d4
            else 0.0
        )
        if has_d4:
            drivers_active += 1

        # ------------------------------------------------------------------
        # Driver 5: Gold Miners Relative Strength (GDX / GLD)
        # ------------------------------------------------------------------
        miners_c = self._window_from(self._miners_ts, self._miners_vals, as_of, 140)
        miners_rs_mom = self._mom_score(miners_c, windows=(21, 63, 126), scale=0.08)
        has_d5 = len(miners_c) >= 2
        miners_rs_score = _clamp(miners_rs_mom, -1.0, 1.0) if has_d5 else 0.0
        if has_d5:
            drivers_active += 1

        # ------------------------------------------------------------------
        # Driver 6: Gold Volatility (^GVZ level & z-score exhaustion)
        # ------------------------------------------------------------------
        gvz_c = self._window_at("^GVZ", as_of, 140)
        gvz_level = gvz_c[-1] if gvz_c else 0.0
        gvz_z = self._causal_zscore(gvz_c, min_len=20, lo=-4.0, hi=5.0)
        has_d6 = len(gvz_c) >= 1
        if has_d6:
            drivers_active += 1
            if gvz_level <= 20.0:
                level_comp = 0.20 * ((20.0 - gvz_level) / 20.0)
            else:
                level_comp = -0.80 * math.tanh((gvz_level - 20.0) / 16.0)
            z_comp = -0.25 * math.tanh(max(0.0, gvz_z - 1.0) / 2.0)
            gvz_score = _clamp(level_comp + z_comp, -1.0, 1.0)
        else:
            gvz_score = 0.0

        # ------------------------------------------------------------------
        # Driver 7: Safe-Haven Asymmetric ^VIX Response
        # Elevated ^VIX is bullish safe-haven tailwind when Gold & real-rate
        # tailwind confirm; flips negative only under dash-for-cash liquidity
        # liquidation (Gold falling + real yields rising OR extreme VIX >= 65).
        # ------------------------------------------------------------------
        vix_c = self._window_at("^VIX", as_of, 140)
        vix_level = vix_c[-1] if vix_c else 0.0
        vix_z = self._causal_zscore(vix_c, min_len=20, lo=-4.0, hi=6.0)
        has_d7 = len(vix_c) >= 1
        if has_d7:
            drivers_active += 1
            rates_constructive = (real_yield_score >= -0.05) or (bond_inflation_score >= 0.0)
            gold_holding = gold_mom >= -0.02
            stress_mag = max(0.0, math.tanh((vix_level - 18.0) / 18.0))
            z_mag = max(0.0, math.tanh(max(0.0, vix_z - 0.5) / 2.5))
            if vix_level >= 65.0 and not gold_holding:
                # Extreme systemic liquidity crunch (March 2020 style dash-for-cash)
                vix_safe_haven_score = _clamp(-0.75 * stress_mag - 0.25 * z_mag, -1.0, 0.0)
            elif vix_level >= 20.0 or vix_z >= 0.75:
                if gold_holding and rates_constructive:
                    # Confirmed safe-haven bid into Gold
                    liquidity_damp = 0.65 if vix_level >= 65.0 else 1.0
                    vix_safe_haven_score = _clamp(
                        liquidity_damp * (0.65 * stress_mag + 0.35 * z_mag), 0.0, 1.0
                    )
                elif (not gold_holding) and (real_yield_score < -0.10):
                    # Dash-for-cash liquidation: rising yields + falling gold + VIX stress
                    vix_safe_haven_score = _clamp(-0.70 * stress_mag - 0.30 * z_mag, -1.0, 0.0)
                else:
                    vix_safe_haven_score = _clamp(0.20 * stress_mag, -0.5, 0.5)
            else:
                # Calm equity volatility
                vix_safe_haven_score = 0.0
        else:
            vix_safe_haven_score = 0.0

        # ------------------------------------------------------------------
        # Weighted Composite Across Active Drivers (No Premature Clipping)
        # ------------------------------------------------------------------
        weighted_pairs = (
            (has_d1, 0.28, real_yield_score),
            (has_d2, 0.20, bond_inflation_score),
            (has_d3, 0.16, dxy_score),
            (has_d4, 0.10, gsr_score),
            (has_d5, 0.10, miners_rs_score),
            (has_d6, 0.08, gvz_score),
            (has_d7, 0.08, vix_safe_haven_score),
        )
        active_w = math.fsum(w for present, w, _ in weighted_pairs if present)
        if active_w > 1e-12:
            composite = math.fsum(w * s for present, w, s in weighted_pairs if present) / active_w
            composite = _clamp(composite, -1.0, 1.0)
        else:
            composite = 0.0

        # Regime classification
        if vix_level >= 28.0 and vix_safe_haven_score >= 0.25 and composite > 0.0:
            regime_label = "GOLD_SAFE_HAVEN_BID"
        elif vix_level >= 32.0 and vix_safe_haven_score <= -0.35 and composite <= -0.20:
            regime_label = "GOLD_LIQUIDITY_STRESS"
        elif composite >= 0.20:
            regime_label = "GOLD_BULL_TAILWIND"
        elif composite <= -0.20:
            regime_label = "GOLD_BEAR_HEADWIND"
        else:
            regime_label = "GOLD_NEUTRAL"

        # Asymmetric exposure scalar with ^GVZ vol exhaustion throttle
        gvz_throttle = 1.0
        if has_d6 and gvz_level > 24.0:
            gvz_throttle = max(0.60, 1.0 - (gvz_level - 24.0) / 55.0)
        if composite >= 0.0:
            raw_scalar = 1.0 + 0.25 * composite
        else:
            raw_scalar = max(0.25, 1.0 + 0.70 * composite)
        exposure_scalar = _clamp(raw_scalar * gvz_throttle, 0.20, 1.25)

        return {
            "as_of_ms": as_of,
            "macro_gold_composite": round(composite, 6),
            "regime_label": regime_label,
            "exposure_scalar": round(exposure_scalar, 6),
            "real_10y_level": round(real_10y_level, 6),
            "real_10y_change": round(real_10y_change, 6),
            "real_yield_score": round(real_yield_score, 6),
            "gold_treasury_state": round(gt_state, 6),
            "bond_inflation_score": round(bond_inflation_score, 6),
            "dxy_score": round(dxy_score, 6),
            "gsr_trend": round(gsr_trend, 6),
            "gsr_z": round(gsr_z, 6),
            "gsr_score": round(gsr_score, 6),
            "miners_rs_mom": round(miners_rs_mom, 6),
            "miners_rs_score": round(miners_rs_score, 6),
            "gvz_level": round(gvz_level, 6),
            "gvz_z": round(gvz_z, 6),
            "gvz_score": round(gvz_score, 6),
            "vix_level": round(vix_level, 6),
            "vix_z": round(vix_z, 6),
            "vix_safe_haven_score": round(vix_safe_haven_score, 6),
            "gold_mom": round(gold_mom, 6),
            "drivers_active": int(drivers_active),
        }
