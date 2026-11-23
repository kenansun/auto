"""Causal Market Regime Profiler & Physics-Based Automatic Parameter Inference Engine (v4 Remediated).

Directly answers the core R&D question:
  "How do we infer parameters automatically from market dynamics and customize strategies
   WITHOUT falling into high-dimensional curve-fitting (PBO = 1.0)?"

Two-layer institutional design (Carver + AFML + QuantPedia):
  1. Causal MarketRegimeProfiler: Classifies each bar strictly causally (`t <= now`,
     ZERO Viterbi lookahead) with hysteresis persistence into interpretable states:
     TREND_EXPANSION / TREND_CALM / SQUEEZE_COIL / RANGE_CHOP / HIGH_VOL_STRESS / CARRY_RICH.
     **D7 Hard Isolation**: Yahoo `^VIX` (`macro_vix_stress_z`) NEVER touches Web3/crypto
     assets or any track with `use_macro_overlay=False`.
  2. AutoParameterInferrer: Derives baseline parameters from market physics BEFORE any grid scan:
     - Cost-to-Volatility Horizon Law (prevents ATB H6 / CFD swap churn disasters!)
     - Carver Volatility Targeting (auto-scales leverage inversely to realized volatility)
     - Persistence-Adaptive Lookbacks (scales Donchian / EWMAC speed to ER & Hurst)
     - Noise-Calibrated Chandelier ATR Stop & Carver Inertia Position Buffer
     - Regime & Gold-Macro Conditioning Exposure Schedule
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from alphaloop.platform.assets import (
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_CRYPTO_SPOT,
    ASSET_CLASS_GOLD_PERP,
    ASSET_CLASS_GOLD_TRADFI,
    AssetSpec,
)
from alphaloop.platform.data_lake import CanonicalBar
from alphaloop.platform.factors import FactorSnapshot

logger = logging.getLogger(__name__)

REGIME_TREND_EXPANSION = "TREND_EXPANSION"
REGIME_TREND_CALM = "TREND_CALM"
REGIME_SQUEEZE_COIL = "SQUEEZE_COIL"
REGIME_RANGE_CHOP = "RANGE_CHOP"
REGIME_HIGH_VOL_STRESS = "HIGH_VOL_STRESS"
REGIME_CARRY_RICH = "CARRY_RICH"

ARCHETYPE_MULTI_FACTOR_TREND = "MULTI_FACTOR_TREND_CTA"
ARCHETYPE_GOLD_MACRO_TREND = "GOLD_MACRO_TREND_SQUEEZE"
ARCHETYPE_CARRY_HARVEST = "VOL_TIERED_CARRY"
ARCHETYPE_DEFENSIVE_OBSERVE = "DEFENSIVE_OBSERVE_ONLY"

_WEB3_ASSET_CLASSES: frozenset[str] = frozenset(
    {ASSET_CLASS_CRYPTO_PERP, ASSET_CLASS_CRYPTO_SPOT, "WEB3", "CRYPTO"}
)
_GOLD_MACRO_ASSET_CLASSES: frozenset[str] = frozenset(
    {ASSET_CLASS_GOLD_PERP, ASSET_CLASS_GOLD_TRADFI}
)


def is_web3_asset_class(asset_class: str | None) -> bool:
    """Return True if `asset_class` belongs to Web3/crypto (strictly isolated from TradFi Yahoo ^VIX)."""
    if not asset_class:
        return False
    ac = str(asset_class).strip().upper()
    return ac in _WEB3_ASSET_CLASSES or "CRYPTO" in ac or "WEB3" in ac


@dataclass(frozen=True)
class RegimePoint:
    ts_event_ms: int
    ts_avail_ms: int
    raw_regime: str
    confirmed_regime: str
    adx_14: float
    kaufman_er_20: float
    vol_pctile_60: float
    squeeze_ratio_20: float
    hurst_proxy_40: float
    macro_gold_composite: float


@dataclass
class MarketProfile:
    """Empirical structural fingerprint of an asset/universe over a historical window."""

    symbol: str
    venue: str
    asset_class: str
    timeframe_h: float
    sample_bars: int
    realized_vol_ann_median_pct: float
    realized_vol_ann_p75_pct: float
    bar_median_abs_ret_bps: float
    round_trip_cost_bps: float
    cost_to_bar_move_ratio: float
    min_cost_feasible_horizon_bars: int
    cost_feasible: bool
    median_adx_14: float
    median_kaufman_er: float
    median_hurst_proxy: float
    median_variance_ratio: float
    excess_kurtosis: float
    median_carry_apr_pct: float
    p90_abs_carry_apr_pct: float
    current_regime: str
    regime_distribution_pct: dict[str, float]
    macro_gold_latest: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InferredParameterSet:
    """Physics-derived baseline strategy parameters + explainable causal reasoning trace."""

    symbol_or_track: str
    asset_class: str
    timeframe_h: float
    recommended_archetype: str
    cost_feasible: bool
    recommended_min_timeframe_h: float
    target_annual_vol_pct: float
    max_symbol_leverage: float
    risk_per_trade_pct: float
    donchian_entry_n: int
    donchian_exit_n: int
    atr_stop_mult: float
    min_hold_bars: int
    position_buffer_pct: float
    ewmac_weight_fast: float
    ewmac_weight_mid: float
    ewmac_weight_slow: float
    use_macro_overlay: bool
    allow_short: bool
    regime_exposure_map: dict[str, float] = field(default_factory=dict)
    inference_trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _fmedian(vals: Sequence[float]) -> float:
    n = len(vals)
    if n == 0:
        return 0.0
    ordered = sorted(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def classify_raw_regime(
    snap: FactorSnapshot,
    *,
    use_macro_overlay: bool = False,
    asset_class: str | None = None,
) -> str:
    """Strictly causal point-in-time classification of one FactorSnapshot.

    D7 Hard Guard: Yahoo `^VIX` (`macro_vix_stress_z`) NEVER affects regime classification
    when `use_macro_overlay=False` OR when `asset_class` is Web3/crypto.
    """
    vol_pct = snap.get("vol_pctile_60", 0.5)
    allow_vix = bool(use_macro_overlay) and not is_web3_asset_class(asset_class)
    vix_z = snap.get("macro_vix_stress_z", 0.0) if allow_vix else 0.0
    sq = snap.get("squeeze_ratio_20", 1.0)
    adx = snap.get("adx_14", 15.0)
    er = snap.get("kaufman_er_20", 0.3)
    hurst = snap.get("hurst_proxy_40", 0.5)
    vr = snap.get("variance_ratio_20", 1.0)
    abs_carry = abs(snap.get("carry_apr_24h", 0.0))

    if vol_pct >= 0.88 or (allow_vix and vix_z >= 2.2):
        return REGIME_HIGH_VOL_STRESS
    if sq < 0.85 and vol_pct <= 0.30:
        return REGIME_SQUEEZE_COIL
    if adx >= 25.0 and er >= 0.32 and (hurst >= 0.50 or vr >= 1.02):
        return REGIME_TREND_EXPANSION
    if adx >= 18.0 and er >= 0.22 and vol_pct < 0.78:
        return REGIME_TREND_CALM
    if abs_carry >= 15.0 and adx < 20.0:
        return REGIME_CARRY_RICH
    return REGIME_RANGE_CHOP


class CausalRegimeTracker:
    """Streaming-identical causal regime state machine with hysteresis anti-whipsaw."""

    def __init__(
        self,
        confirm_bars: int = 2,
        *,
        use_macro_overlay: bool = False,
        asset_class: str | None = None,
    ) -> None:
        self.confirm_bars = max(1, int(confirm_bars))
        self.asset_class = asset_class
        self.use_macro_overlay = bool(use_macro_overlay) and not is_web3_asset_class(asset_class)
        self.confirmed: str = REGIME_TREND_CALM
        self._pending: str = ""
        self._pending_count: int = 0

    def step(self, snap: FactorSnapshot) -> RegimePoint:
        raw = classify_raw_regime(
            snap,
            use_macro_overlay=self.use_macro_overlay,
            asset_class=self.asset_class,
        )
        # Immediate emergency escalation into HIGH_VOL_STRESS (fail-safe risk defense)
        if raw == REGIME_HIGH_VOL_STRESS:
            self.confirmed = REGIME_HIGH_VOL_STRESS
            self._pending = ""
            self._pending_count = 0
        elif raw == self.confirmed:
            self._pending = ""
            self._pending_count = 0
        else:
            if raw == self._pending:
                self._pending_count += 1
            else:
                self._pending = raw
                self._pending_count = 1
            if self._pending_count >= self.confirm_bars:
                self.confirmed = raw
                self._pending = ""
                self._pending_count = 0
        return RegimePoint(
            ts_event_ms=snap.ts_event_ms,
            ts_avail_ms=snap.ts_avail_ms,
            raw_regime=raw,
            confirmed_regime=self.confirmed,
            adx_14=round(snap.get("adx_14", 0.0), 2),
            kaufman_er_20=round(snap.get("kaufman_er_20", 0.0), 3),
            vol_pctile_60=round(snap.get("vol_pctile_60", 0.5), 3),
            squeeze_ratio_20=round(snap.get("squeeze_ratio_20", 1.0), 3),
            hurst_proxy_40=round(snap.get("hurst_proxy_40", 0.5), 3),
            macro_gold_composite=round(snap.get("macro_gold_composite", 0.0), 3) if self.use_macro_overlay else 0.0,
        )

    def classify_series(self, snaps: list[FactorSnapshot]) -> list[RegimePoint]:
        return [self.step(s) for s in snaps]


def _excess_kurtosis(vals: list[float]) -> float:
    n = len(vals)
    if n < 10:
        return 0.0
    mu = math.fsum(vals) / n
    diffs = [x - mu for x in vals]
    m2 = math.fsum(d * d for d in diffs) / n
    if m2 <= 1e-18:
        return 0.0
    m4 = math.fsum((d * d) * (d * d) for d in diffs) / n
    return _clamp((m4 / (m2 * m2)) - 3.0, -2.0, 30.0)


class MarketRegimeProfiler:
    """Profiles market physics & cost feasibility from real CanonicalBars + FactorSnapshots."""

    def profile_asset(
        self,
        spec: AssetSpec,
        bars: list[CanonicalBar],
        snaps: list[FactorSnapshot],
        execution_mode: str = "TAKER_TAKER",
        warmup_bars: int = 35,
        use_macro_overlay: bool | None = None,
    ) -> tuple[MarketProfile, list[RegimePoint]]:
        if use_macro_overlay is None:
            eff_macro = (
                spec.asset_class in _GOLD_MACRO_ASSET_CLASSES
                and not is_web3_asset_class(spec.asset_class)
            )
        else:
            eff_macro = bool(use_macro_overlay) and not is_web3_asset_class(spec.asset_class)

        tracker = CausalRegimeTracker(
            confirm_bars=2,
            use_macro_overlay=eff_macro,
            asset_class=spec.asset_class,
        )
        regime_pts = tracker.classify_series(snaps)
        valid_snaps = snaps[warmup_bars:] if len(snaps) > warmup_bars + 10 else snaps
        valid_regs = regime_pts[warmup_bars:] if len(regime_pts) > warmup_bars + 10 else regime_pts

        ordered = sorted(bars, key=lambda b: b.ts_event_ms)
        log_rets = [
            math.log(ordered[i].close / ordered[i - 1].close)
            for i in range(1, len(ordered))
            if ordered[i - 1].close > 0.0 and ordered[i].close > 0.0
        ]
        abs_ret_bps = [abs(r) * 10_000.0 for r in log_rets]
        bar_med_move = _fmedian(abs_ret_bps) if abs_ret_bps else 10.0
        rt_cost = spec.cost_model.round_trip_cost_bps(execution_mode=execution_mode)

        # Cost-to-Volatility Horizon Law:
        # Find smallest holding horizon H where median |H-bar move| >= 3.0 * round_trip_cost_bps.
        min_h = 1
        cost_feasible = False
        closes = [b.close for b in ordered]
        for h in (1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48):
            if len(closes) <= h + 5:
                break
            hmoves = [
                abs(closes[i] / closes[i - h] - 1.0) * 10_000.0
                for i in range(h, len(closes))
                if closes[i - h] > 0.0
            ]
            if hmoves and _fmedian(hmoves) >= max(3.0 * rt_cost, 15.0):
                min_h = h
                cost_feasible = True
                break
        if not cost_feasible:
            min_h = 24

        vols_ann = sorted(s.get("realized_vol_ann_20", 20.0) for s in valid_snaps)
        vol_med = _fmedian(vols_ann) if vols_ann else 20.0
        vol_p75 = vols_ann[min(len(vols_ann) - 1, int(0.75 * len(vols_ann)))] if vols_ann else vol_med

        adx_med = _fmedian([s.get("adx_14", 15.0) for s in valid_snaps]) if valid_snaps else 15.0
        er_med = _fmedian([s.get("kaufman_er_20", 0.3) for s in valid_snaps]) if valid_snaps else 0.3
        hurst_med = _fmedian([s.get("hurst_proxy_40", 0.5) for s in valid_snaps]) if valid_snaps else 0.5
        vr_med = _fmedian([s.get("variance_ratio_20", 1.0) for s in valid_snaps]) if valid_snaps else 1.0
        carry_vals = [s.get("carry_apr_24h", 0.0) for s in valid_snaps]
        abs_carries = sorted(abs(x) for x in carry_vals)
        carry_med = _fmedian(carry_vals) if carry_vals else 0.0
        carry_p90 = abs_carries[min(len(abs_carries) - 1, int(0.90 * len(abs_carries)))] if abs_carries else 0.0

        reg_counts: dict[str, int] = {}
        for rp in valid_regs:
            reg_counts[rp.confirmed_regime] = reg_counts.get(rp.confirmed_regime, 0) + 1
        total_r = max(1, len(valid_regs))
        reg_dist = {k: round(100.0 * v / total_r, 1) for k, v in sorted(reg_counts.items())}

        tf_h = ordered[0].timeframe_h if ordered else 12.0
        profile = MarketProfile(
            symbol=spec.symbol,
            venue=spec.venue,
            asset_class=spec.asset_class,
            timeframe_h=tf_h,
            sample_bars=len(ordered),
            realized_vol_ann_median_pct=round(vol_med, 2),
            realized_vol_ann_p75_pct=round(vol_p75, 2),
            bar_median_abs_ret_bps=round(bar_med_move, 2),
            round_trip_cost_bps=round(rt_cost, 2),
            cost_to_bar_move_ratio=round(rt_cost / max(bar_med_move, 1e-6), 3),
            min_cost_feasible_horizon_bars=min_h,
            cost_feasible=cost_feasible,
            median_adx_14=round(adx_med, 2),
            median_kaufman_er=round(er_med, 3),
            median_hurst_proxy=round(hurst_med, 3),
            median_variance_ratio=round(vr_med, 3),
            excess_kurtosis=round(_excess_kurtosis(log_rets), 2),
            median_carry_apr_pct=round(carry_med, 2),
            p90_abs_carry_apr_pct=round(carry_p90, 2),
            current_regime=valid_regs[-1].confirmed_regime if valid_regs else REGIME_RANGE_CHOP,
            regime_distribution_pct=reg_dist,
            macro_gold_latest=(
                round(valid_snaps[-1].get("macro_gold_composite", 0.0), 3)
                if (valid_snaps and eff_macro)
                else 0.0
            ),
        )
        return profile, regime_pts


class AutoParameterInferrer:
    """Infers physically grounded, anti-overfit strategy parameters from MarketProfile(s)."""

    def __init__(self, target_annual_vol_pct: float = 18.0) -> None:
        self.target_annual_vol_pct = float(target_annual_vol_pct)

    def infer_from_profiles(
        self,
        track_name: str,
        specs: list[AssetSpec],
        profiles: list[MarketProfile],
    ) -> InferredParameterSet:
        if not profiles or not specs:
            raise ValueError("At least one AssetSpec and MarketProfile required for inference")

        tf_h = profiles[0].timeframe_h
        asset_class = specs[0].asset_class
        is_gold_track = all(s.asset_class in _GOLD_MACRO_ASSET_CLASSES for s in specs)
        allow_short = all(s.calendar.allow_short for s in specs)
        max_lev_cap = min(s.max_leverage_cap for s in specs)

        vol_p75 = _fmedian([p.realized_vol_ann_p75_pct for p in profiles])
        vol_med = _fmedian([p.realized_vol_ann_median_pct for p in profiles])
        er_med = _fmedian([p.median_kaufman_er for p in profiles])
        adx_med = _fmedian([p.median_adx_14 for p in profiles])
        hurst_med = _fmedian([p.median_hurst_proxy for p in profiles])
        vr_med = _fmedian([p.median_variance_ratio for p in profiles])
        kurt_med = _fmedian([p.excess_kurtosis for p in profiles])
        rt_cost = _fmedian([p.round_trip_cost_bps for p in profiles])
        bar_move = _fmedian([p.bar_median_abs_ret_bps for p in profiles])
        min_hold = int(max(2, round(_fmedian([p.min_cost_feasible_horizon_bars for p in profiles]))))
        cost_feasible = any(p.cost_feasible for p in profiles)
        carry_p90 = _fmedian([p.p90_abs_carry_apr_pct for p in profiles])

        trace: list[str] = []

        # 1. Cost-to-Volatility Horizon Law
        rec_tf_h = tf_h
        if rt_cost / max(bar_move, 1e-6) > 0.35:
            rec_tf_h = min(24.0, tf_h * 2.0)
            trace.append(
                f"Cost-Horizon Law: RT cost {rt_cost:.1f}bp is >35% of single-bar median move {bar_move:.1f}bp "
                f"-> enforce min_hold={min_hold} bars and recommend >= {rec_tf_h:.0f}h timeframe."
            )
        else:
            trace.append(
                f"Cost-Horizon Law: RT cost {rt_cost:.1f}bp vs bar move {bar_move:.1f}bp "
                f"(ratio={rt_cost/max(bar_move, 1e-6):.2f}) -> cost-feasible at {tf_h:.0f}h with min_hold={min_hold} bars."
            )

        # 2. Carver Volatility Targeting -> Leverage & Risk/Trade
        max_sym_lev = _clamp(self.target_annual_vol_pct / max(vol_p75, 10.0), 0.30, max_lev_cap)
        risk_per_trade = _clamp(0.75 * (20.0 / max(vol_med, 12.0)), 0.35, 1.25)
        trace.append(
            f"Carver Vol-Target ({self.target_annual_vol_pct:.1f}% ann): p75 realized vol={vol_p75:.1f}% "
            f"-> max_symbol_leverage={max_sym_lev:.2f}x (cap {max_lev_cap:.1f}x), risk_per_trade={risk_per_trade:.2f}%."
        )

        # 3. Persistence & Noise -> Adaptive Donchian & EWMAC Speed Weights
        persistence_score = (
            0.45 * er_med
            + 0.35 * _clamp((hurst_med - 0.35) / 0.35, 0.0, 1.0)
            + 0.20 * _clamp((adx_med - 12.0) / 25.0, 0.0, 1.0)
        )
        if tf_h >= 24.0:
            entry_n = int(round(_clamp(55.0 - 25.0 * persistence_score, 20.0, 55.0)))
        elif tf_h >= 8.0:
            entry_n = int(round(_clamp(48.0 - 20.0 * persistence_score, 24.0, 48.0)))
        else:
            entry_n = int(round(_clamp(36.0 - 18.0 * persistence_score, 12.0, 36.0)))
        exit_n = max(6, int(round(entry_n * 0.40)))

        if persistence_score >= 0.55:
            w_fast, w_mid, w_slow = 0.35, 0.40, 0.25
        elif persistence_score >= 0.38:
            w_fast, w_mid, w_slow = 0.25, 0.40, 0.35
        else:
            w_fast, w_mid, w_slow = 0.15, 0.35, 0.50
        trace.append(
            f"Persistence Inference (ER={er_med:.2f}, Hurst={hurst_med:.2f}, ADX={adx_med:.1f}, VR={vr_med:.2f} "
            f"-> score={persistence_score:.2f}): Donchian ({entry_n}/{exit_n}), EWMAC weights ({w_fast}/{w_mid}/{w_slow})."
        )

        # 4. Noise & Fat-Tail Calibrated Chandelier ATR Stop + Carver Position Buffer
        atr_stop_mult = _clamp(2.6 + 1.4 * (1.0 - er_med) + 0.20 * max(0.0, kurt_med / 4.0), 2.5, 4.5)
        pos_buffer = _clamp(0.08 + 0.06 * (rt_cost / max(bar_move, 1.0)), 0.08, 0.25)
        trace.append(
            f"Stop & Buffer Calibration (kurtosis={kurt_med:.1f}): Chandelier ATR mult={atr_stop_mult:.2f}, "
            f"Carver inertia buffer={pos_buffer * 100:.1f}%."
        )

        # 5. Regime Exposure Schedule & Archetype Selection
        if not cost_feasible:
            archetype = ARCHETYPE_DEFENSIVE_OBSERVE
            trace.append("Archetype=DEFENSIVE_OBSERVE_ONLY because expected moves fail 3x round-trip cost floor.")
        elif is_gold_track:
            archetype = ARCHETYPE_GOLD_MACRO_TREND
            trace.append(
                "Archetype=GOLD_MACRO_TREND_SQUEEZE: combines Donchian/EWMAC trend "
                "conditioned on QuantPedia Gold-Treasury/DXY/TIP real-rate macro regime."
            )
        elif adx_med < 17.0 and carry_p90 >= 15.0 and er_med < 0.25:
            archetype = ARCHETYPE_CARRY_HARVEST
            trace.append(
                f"Archetype=VOL_TIERED_CARRY: choppy price action (ADX={adx_med:.1f}) with rich carry (p90={carry_p90:.1f}% APR)."
            )
        else:
            archetype = ARCHETYPE_MULTI_FACTOR_TREND
            trace.append(
                "Archetype=MULTI_FACTOR_TREND_CTA: cross-asset vol-targeted multi-speed trend + breakout "
                "with regime-adaptive exposure scaling."
            )

        regime_exposure_map = {
            REGIME_TREND_EXPANSION: 1.00,
            REGIME_TREND_CALM: 0.90,
            REGIME_SQUEEZE_COIL: 0.95,
            REGIME_CARRY_RICH: 0.70,
            REGIME_RANGE_CHOP: 0.45,
            REGIME_HIGH_VOL_STRESS: 0.50,
        }

        return InferredParameterSet(
            symbol_or_track=track_name,
            asset_class=asset_class,
            timeframe_h=tf_h,
            recommended_archetype=archetype,
            cost_feasible=cost_feasible,
            recommended_min_timeframe_h=rec_tf_h,
            target_annual_vol_pct=round(self.target_annual_vol_pct, 2),
            max_symbol_leverage=round(max_sym_lev, 3),
            risk_per_trade_pct=round(risk_per_trade, 3),
            donchian_entry_n=entry_n,
            donchian_exit_n=exit_n,
            atr_stop_mult=round(atr_stop_mult, 2),
            min_hold_bars=min_hold,
            position_buffer_pct=round(pos_buffer, 3),
            ewmac_weight_fast=w_fast,
            ewmac_weight_mid=w_mid,
            ewmac_weight_slow=w_slow,
            use_macro_overlay=is_gold_track,
            allow_short=allow_short,
            regime_exposure_map=regime_exposure_map,
            inference_trace=trace,
        )
