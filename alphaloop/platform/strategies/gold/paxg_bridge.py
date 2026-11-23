"""G3: Dedicated On-Chain PAXG Dual-Basis, Weekend Guard & Funding Bridge (Track G, Feature F-19).

Built specifically for 24/7 on-chain tokenized Gold perpetuals (`PAXG` on Hyperliquid /
Binance UM / Gate) with strict physical microstructure separation from TradFi COMEX:
  1. Dual-Basis Decomposition emitted in every `BarDecision.metadata`:
       - `perp_spot_basis_bps = (paxg_perp / paxg_spot - 1.0) * 10000.0`
       - `spot_comex_basis_bps = (paxg_spot / comex_ref - 1.0) * 10000.0`
  2. Own-Bar Stop Independence: Trailing/protective stops are calculated and triggered
     STRICTLY from `PAXG`'s own OHLC bars (100% invariant to COMEX `GC=F` intraday noise).
  3. Friday-Close-Frozen Weekend Guard (`[Fri 22:00 UTC, Sun 22:00 UTC)`):
       - Freezes COMEX reference at Friday's close (`is_comex_weekend_utc`).
       - Widens Carver inertia buffer by `+50%` (`1.5x`) during weekends to suppress churn.
       - Blocks fresh non-stop entries when weekend basis dislocation exceeds `> 2.5 sigma`,
         while NEVER blocking PAXG own-bar protective stop-outs.
  4. Funding Carry Awareness: Properly annualizes `carry_rate_period` via `carry_interval_h`
     and trims long perp exposure whenever annualized funding exceeds `> 25%` APR.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from alphaloop.platform.core_engine.interfaces import BarDecision
from alphaloop.platform.data_foundation.models import CanonicalBar
from alphaloop.platform.data_foundation.router import is_comex_weekend_utc
from alphaloop.platform.strategies.gold.macro_engine import GoldMacroEngine


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _sample_std(vals: deque[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mu = math.fsum(vals) / n
    var = math.fsum((x - mu) * (x - mu) for x in vals) / (n - 1)
    return math.sqrt(var) if (math.isfinite(var) and var > 0.0) else 0.0


def _extract_field(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class GoldPAXGBridgeKernel:
    """G3 On-Chain PAXG Dual-Basis, Weekend Guard & Funding-Aware Bridge Strategy."""

    def __init__(
        self,
        symbol: str = "PAXG",
        *,
        macro_engine: GoldMacroEngine | None = None,
        fast_ema: int = 8,
        slow_ema: int = 24,
        donchian_n: int = 16,
        stop_atr_mult: float = 2.5,
        stop_pct_floor: float = 0.03,
        weekday_inertia_band: float = 0.10,
        weekend_inertia_mult: float = 1.50,
        weekend_dislocation_z: float = 2.5,
        funding_trim_apr_pct: float = 25.0,
        funding_max_trim: float = 0.65,
        allow_short: bool = True,
        target_annual_vol_pct: float = 18.0,
        max_weight: float = 1.0,
        warmup_bars: int = 20,
    ) -> None:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("GoldPAXGBridgeKernel requires a non-empty symbol")
        if fast_ema < 2 or slow_ema <= fast_ema:
            raise ValueError(f"Invalid EMA spans: fast_ema={fast_ema}, slow_ema={slow_ema}")
        if weekend_inertia_mult < 1.0:
            raise ValueError("weekend_inertia_mult must be >= 1.0")

        self.symbol: str = symbol.strip()
        self.strategy_id: str = "gold_g3_paxg_bridge"
        self.asset_domain: str = "gold"
        self.warmup_bars: int = max(2, int(warmup_bars))
        self.macro_engine: GoldMacroEngine | None = macro_engine

        self.fast_ema: int = int(fast_ema)
        self.slow_ema: int = int(slow_ema)
        self.donchian_n: int = max(3, int(donchian_n))
        self.stop_atr_mult: float = max(1.0, float(stop_atr_mult))
        self.stop_pct_floor: float = max(0.005, float(stop_pct_floor))
        self.weekday_inertia_band: float = max(0.01, float(weekday_inertia_band))
        self.weekend_inertia_mult: float = float(weekend_inertia_mult)
        self.weekend_dislocation_z: float = max(0.5, float(weekend_dislocation_z))
        self.funding_trim_apr_pct: float = max(1.0, float(funding_trim_apr_pct))
        self.funding_max_trim: float = _clamp(funding_max_trim, 0.05, 0.95)
        self.allow_short: bool = bool(allow_short)
        self.target_annual_vol_pct: float = float(target_annual_vol_pct)
        self.max_weight: float = max(0.1, float(max_weight))

        self._bar_count: int = 0
        self._ema_fast: float | None = None
        self._ema_slow: float | None = None
        self._atr14: float | None = None
        self._prev_close: float | None = None
        self._highs: deque[float] = deque(maxlen=max(64, self.donchian_n + 6))
        self._lows: deque[float] = deque(maxlen=max(64, self.donchian_n + 6))
        self._closes: deque[float] = deque(maxlen=max(64, self.donchian_n + 6))
        self._log_rets: deque[float] = deque(maxlen=40)
        self._basis_hist: deque[float] = deque(maxlen=60)
        self._funding_apr_ema: float | None = None
        self._frozen_comex_price: float | None = None
        self._active_stop_price: float | None = None
        self._executed_weights: dict[str, float] = {}

    def reset(self) -> None:
        """Reset all mutable streaming state cleanly."""
        self._bar_count = 0
        self._ema_fast = None
        self._ema_slow = None
        self._atr14 = None
        self._prev_close = None
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._log_rets.clear()
        self._basis_hist.clear()
        self._funding_apr_ema = None
        self._frozen_comex_price = None
        self._active_stop_price = None
        self._executed_weights = {}

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        """Synchronize actual post-allocator/post-fill weights for weekend inertia & G7 parity."""
        prev_w = float(self._executed_weights.get(self.symbol, 0.0))
        self._executed_weights = {str(k): float(v) for k, v in executed_weights.items()}
        new_w = float(self._executed_weights.get(self.symbol, 0.0))
        if abs(new_w) <= 1e-9:
            self._active_stop_price = None
        elif (prev_w * new_w <= 0.0) and self._prev_close is not None and self._atr14 is not None:
            # Freshly established/flipped position -> anchor own-bar protective stop
            dist = max(self.stop_atr_mult * self._atr14, self.stop_pct_floor * self._prev_close)
            if new_w > 0.0:
                self._active_stop_price = max(1e-6, self._prev_close - dist)
            else:
                self._active_stop_price = self._prev_close + dist

    def clone_fresh(self) -> GoldPAXGBridgeKernel:
        """Return a freshly reset clone with identical configuration."""
        return GoldPAXGBridgeKernel(
            symbol=self.symbol,
            macro_engine=self.macro_engine,
            fast_ema=self.fast_ema,
            slow_ema=self.slow_ema,
            donchian_n=self.donchian_n,
            stop_atr_mult=self.stop_atr_mult,
            stop_pct_floor=self.stop_pct_floor,
            weekday_inertia_band=self.weekday_inertia_band,
            weekend_inertia_mult=self.weekend_inertia_mult,
            weekend_dislocation_z=self.weekend_dislocation_z,
            funding_trim_apr_pct=self.funding_trim_apr_pct,
            funding_max_trim=self.funding_max_trim,
            allow_short=self.allow_short,
            target_annual_vol_pct=self.target_annual_vol_pct,
            max_weight=self.max_weight,
            warmup_bars=self.warmup_bars,
        )

    def _resolve_macro(
        self,
        ts_ms: int,
        context: dict[str, Any] | None,
    ) -> tuple[float, str]:
        macro_obj = context.get("gold_macro") if context else None
        if macro_obj is None and self.macro_engine is not None:
            # Query at ts_ms (strictly <= bar open/weekend timestamp, never peeking Monday!)
            macro_obj = self.macro_engine.snapshot_as_of(int(ts_ms))
        if macro_obj is None:
            return 0.0, "GOLD_PAXG_NEUTRAL"
        raw_c = _extract_field(macro_obj, "macro_gold_composite", 0.0)
        comp = _clamp(float(raw_c), -1.5, 1.5) if (raw_c is not None and math.isfinite(float(raw_c))) else 0.0
        lbl = str(_extract_field(macro_obj, "regime_label", ""))
        if not lbl:
            if comp >= 0.20:
                lbl = "GOLD_BULL_TAILWIND"
            elif comp <= -0.20:
                lbl = "GOLD_BEAR_HEADWIND"
            else:
                lbl = "GOLD_PAXG_NEUTRAL"
        return comp, lbl

    def _resolve_dual_basis(
        self,
        ts_ms: int,
        bar: CanonicalBar,
        is_weekend: bool,
        context: dict[str, Any] | None,
    ) -> tuple[float, float, float, float, float, float]:
        """Return `(paxg_spot, comex_ref, perp_spot_bps, spot_comex_bps, perp_comex_bps, basis_z)`."""
        paxg_perp = float(bar.close)
        ctx = context or {}
        basis_pt = ctx.get("paxg_basis")

        # Resolve PAXG Spot price
        spot_raw = ctx.get("paxg_spot_price", ctx.get("paxg_spot_close"))
        if spot_raw is None and basis_pt is not None:
            spot_raw = _extract_field(basis_pt, "paxg_spot_close", None)
        if spot_raw is not None and math.isfinite(float(spot_raw)) and float(spot_raw) > 0.0:
            paxg_spot = float(spot_raw)
        elif math.isfinite(float(bar.premium_basis_bps)) and abs(float(bar.premium_basis_bps)) > 1e-9:
            denom = 1.0 + (float(bar.premium_basis_bps) / 10_000.0)
            paxg_spot = paxg_perp / denom if denom > 0.1 else paxg_perp
        else:
            paxg_spot = paxg_perp

        # Resolve COMEX GC=F reference (strictly frozen at Friday close during weekend window!)
        ctx_comex_raw = ctx.get("comex_price", ctx.get("gc_f_price", ctx.get("gc_ref_close")))
        if ctx_comex_raw is None and basis_pt is not None:
            ctx_comex_raw = _extract_field(basis_pt, "gc_ref_close", None)

        ctx_comex: float | None = None
        if ctx_comex_raw is not None and math.isfinite(float(ctx_comex_raw)) and float(ctx_comex_raw) > 0.0:
            ctx_comex = float(ctx_comex_raw)

        macro_comex: float | None = None
        if ctx_comex is None and self.macro_engine is not None:
            mc = self.macro_engine.latest_close_as_of("GC=F", int(ts_ms))
            if mc is not None and math.isfinite(mc) and mc > 0.0:
                macro_comex = float(mc)

        if not is_weekend:
            comex_candidate = ctx_comex if ctx_comex is not None else macro_comex
            if comex_candidate is not None:
                self._frozen_comex_price = comex_candidate
                comex_ref = comex_candidate
            elif self._frozen_comex_price is not None:
                comex_ref = self._frozen_comex_price
            else:
                comex_ref = paxg_spot
        else:
            # Friday-close-frozen COMEX reference during [Fri 22:00 UTC, Sun 22:00 UTC).
            # If caller explicitly supplies a reference comex_price in context, honor it for
            # dual-basis reporting without mutating the weekday Friday-close anchor.
            if ctx_comex is not None:
                comex_ref = ctx_comex
            elif self._frozen_comex_price is not None:
                comex_ref = self._frozen_comex_price
            elif macro_comex is not None:
                comex_ref = macro_comex
            else:
                comex_ref = paxg_spot

        perp_spot_bps = ((paxg_perp / paxg_spot) - 1.0) * 10_000.0
        spot_comex_bps = ((paxg_spot / comex_ref) - 1.0) * 10_000.0
        perp_comex_bps = ((paxg_perp / comex_ref) - 1.0) * 10_000.0

        # Resolve causal basis dislocation z-score
        ctx_bz = ctx.get("basis_z", ctx.get("weekend_basis_z"))
        if ctx_bz is not None and math.isfinite(float(ctx_bz)):
            basis_z = float(ctx_bz)
        elif len(self._basis_hist) >= 12:
            mu_b = math.fsum(self._basis_hist) / len(self._basis_hist)
            sd_b = _sample_std(self._basis_hist)
            if sd_b > 1e-6:
                basis_z = _clamp((spot_comex_bps - mu_b) / sd_b, -10.0, 10.0)
            else:
                basis_z = _clamp((spot_comex_bps - mu_b) / 35.0, -10.0, 10.0)
        else:
            basis_z = _clamp(spot_comex_bps / 80.0, -10.0, 10.0)

        self._basis_hist.append(spot_comex_bps)
        return paxg_spot, comex_ref, perp_spot_bps, spot_comex_bps, perp_comex_bps, basis_z

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        """Process PAXG bar strictly causally with own-bar stops, weekend guard, and funding trim."""
        decisions: dict[str, BarDecision] = {}
        bar = bars_at_ts.get(self.symbol)
        if bar is None:
            return decisions

        c = float(bar.close)
        h = float(bar.high)
        lo = float(bar.low)
        self._bar_count += 1

        # Strictly causal prior PAXG channel BEFORE appending current bar
        if len(self._highs) >= 2:
            k = min(len(self._highs), self.donchian_n)
            prior_hi = max(list(self._highs)[-k:])
            prior_lo = min(list(self._lows)[-k:])
            span = prior_hi - prior_lo
            donchian_pos = _clamp((2.0 * ((c - prior_lo) / span)) - 1.0, -1.2, 1.2) if span > 1e-9 else 0.0
        else:
            prior_hi, prior_lo = h, lo
            donchian_pos = 0.0

        # Update PAXG own-bar ATR-14, returns, and EMAs
        if self._prev_close is not None and self._prev_close > 0.0:
            tr = max(h - lo, abs(h - self._prev_close), abs(lo - self._prev_close))
            self._log_rets.append(math.log(max(c, 1e-9) / self._prev_close))
        else:
            tr = max(h - lo, c * 0.004)
        if self._atr14 is None:
            self._atr14 = tr
        else:
            self._atr14 = (self._atr14 * 13.0 + tr) / 14.0

        a_fast = 2.0 / (self.fast_ema + 1.0)
        a_slow = 2.0 / (self.slow_ema + 1.0)
        if self._ema_fast is None or self._ema_slow is None:
            self._ema_fast = c
            self._ema_slow = c
        else:
            self._ema_fast = a_fast * c + (1.0 - a_fast) * self._ema_fast
            self._ema_slow = a_slow * c + (1.0 - a_slow) * self._ema_slow

        self._highs.append(h)
        self._lows.append(lo)
        self._closes.append(c)
        self._prev_close = c

        atr = max(float(self._atr14), c * 0.003)

        # Weekend detection, dual basis, macro regime, and properly normalized funding APR
        is_weekend = is_comex_weekend_utc(int(ts_ms))
        paxg_spot, comex_ref, perp_spot_bps, spot_comex_bps, perp_comex_bps, basis_z = (
            self._resolve_dual_basis(ts_ms, bar, is_weekend, context)
        )
        macro_comp, regime_lbl = self._resolve_macro(ts_ms, context)

        # Normalize funding rate strictly by bar.carry_interval_h (D11 exact unit discipline)
        eff_iv_h = max(0.01, float(bar.carry_interval_h))
        cur_funding_apr = (float(bar.carry_rate_period) / eff_iv_h) * 8760.0 * 100.0
        if self._funding_apr_ema is None:
            self._funding_apr_ema = cur_funding_apr
        else:
            self._funding_apr_ema = 0.40 * cur_funding_apr + 0.60 * self._funding_apr_ema
        eff_funding_apr = max(cur_funding_apr, self._funding_apr_ema)

        eff_inertia = (
            self.weekday_inertia_band * self.weekend_inertia_mult
            if is_weekend
            else self.weekday_inertia_band
        )
        prev_w = float(self._executed_weights.get(self.symbol, 0.0))

        # Warmup bars: strictly flat
        if self._bar_count <= self.warmup_bars:
            decisions[self.symbol] = BarDecision(
                symbol=self.symbol,
                ts_event_ms=int(ts_ms),
                target_weight=0.0,
                raw_signal=0.0,
                regime_label=regime_lbl,
                stop_price=None,
                metadata={
                    "warmup": True,
                    "perp_spot_basis_bps": round(perp_spot_bps, 4),
                    "spot_comex_basis_bps": round(spot_comex_bps, 4),
                    "perp_comex_basis_bps": round(perp_comex_bps, 4),
                    "paxg_spot_price": round(paxg_spot, 6),
                    "comex_ref_price": round(comex_ref, 6),
                    "is_weekend_guard": bool(is_weekend),
                    "weekend_frozen_comex": bool(is_weekend),
                    "basis_z": round(basis_z, 4),
                    "weekend_dislocation_blocked": False,
                    "effective_inertia_band": round(eff_inertia, 6),
                    "funding_apr_pct": round(cur_funding_apr, 4),
                    "funding_trim_scalar": 1.0,
                    "own_bar_stop_triggered": False,
                },
            )
            return decisions

        # 1. PRIORITY 1: PAXG Own-Bar Protective Stop Check (100% independent of GC=F & NEVER blocked by weekend guard!)
        own_bar_stop_triggered = False
        if abs(prev_w) > 1e-9:
            stop_dist = max(self.stop_atr_mult * atr, self.stop_pct_floor * c)
            if self._active_stop_price is None:
                self._active_stop_price = max(1e-6, c - stop_dist) if prev_w > 0.0 else (c + stop_dist)
            if prev_w > 0.0 and lo <= self._active_stop_price:
                own_bar_stop_triggered = True
            elif prev_w < 0.0 and h >= self._active_stop_price:
                own_bar_stop_triggered = True

        weekend_dislocation_blocked = False
        funding_trim_scalar = 1.0

        if own_bar_stop_triggered:
            chosen_w = 0.0
            raw_sig = 0.0
            self._active_stop_price = None
        else:
            # 2. Pure PAXG Own-Bar Trend Signal + Gold Macro Alignment
            ewmac_norm = math.tanh(((self._ema_fast - self._ema_slow) / atr) / 1.4)
            lb = min(len(self._closes) - 1, 16)
            if lb >= 4 and self._closes[-1 - lb] > 0.0:
                ret_lb = (c / self._closes[-1 - lb]) - 1.0
                ret_norm = math.tanh(ret_lb / 0.025)
            else:
                ret_norm = 0.0
            tech_sig = _clamp(
                0.45 * ewmac_norm + 0.30 * _clamp(donchian_pos, -1.0, 1.0) + 0.25 * ret_norm,
                -1.0,
                1.0,
            )

            bpy = (24.0 * 365.0) / max(0.25, float(bar.timeframe_h))
            ann_vol = (
                max(0.06, _sample_std(self._log_rets) * math.sqrt(bpy))
                if len(self._log_rets) >= 5
                else 0.16
            )
            vol_scale = _clamp((self.target_annual_vol_pct / 100.0) / ann_vol, 0.35, 1.15)

            if tech_sig > 0.08 and macro_comp >= -0.35:
                macro_boost = 1.0 + 0.25 * max(0.0, min(1.5, macro_comp))
                desired_w = _clamp(
                    (0.35 + 0.65 * tech_sig) * vol_scale * macro_boost,
                    0.0,
                    self.max_weight,
                )
            elif tech_sig < -0.15 and self.allow_short and macro_comp <= 0.35:
                macro_boost = 1.0 + 0.25 * max(0.0, min(1.5, -macro_comp))
                desired_w = -_clamp(
                    abs(tech_sig) * vol_scale * macro_boost * 0.75,
                    0.0,
                    self.max_weight * 0.80,
                )
            else:
                desired_w = 0.0

            # 3. Funding Carry Awareness: Trim long perp exposure when PAXG funding > 25% APR
            if desired_w > 1e-9 and eff_funding_apr > self.funding_trim_apr_pct:
                excess_apr = eff_funding_apr - self.funding_trim_apr_pct
                trim_frac = min(self.funding_max_trim, excess_apr / 100.0)
                funding_trim_scalar = max(1.0 - self.funding_max_trim, 1.0 - trim_frac)
                desired_w *= funding_trim_scalar

            # Mild perp-over-spot froth dampener on extreme perp premium (never based on GC=F!)
            if desired_w > 1e-9 and perp_spot_bps > 75.0:
                perp_damp = max(0.60, 1.0 - (perp_spot_bps - 75.0) / 300.0)
                desired_w *= perp_damp

            # 4. Weekend Guard (>2.5 sigma dislocation blocks fresh entries; +50% inertia buffer)
            if is_weekend and abs(basis_z) > self.weekend_dislocation_z:
                if abs(prev_w) <= 1e-9:
                    desired_w = 0.0
                    weekend_dislocation_blocked = True
                elif abs(desired_w) > abs(prev_w) and (desired_w * prev_w > 0.0):
                    desired_w = prev_w
                    weekend_dislocation_blocked = True

            # Apply proportional Carver inertia buffer (+50% wider on weekends!)
            chosen_w = desired_w
            if abs(prev_w) > 1e-9 and abs(desired_w) > 1e-9 and (prev_w * desired_w > 0.0):
                ref_scale = max(abs(desired_w), abs(prev_w))
                if abs(desired_w - prev_w) <= eff_inertia * ref_scale + 1e-12:
                    chosen_w = prev_w

            raw_sig = _clamp(tech_sig * 12.0 + macro_comp * 4.0, -20.0, 20.0)
            if weekend_dislocation_blocked and abs(chosen_w) <= 1e-9:
                raw_sig = 0.0

            # Ratchet own-bar trailing stop strictly from PAXG's own price & ATR
            if chosen_w > 1e-9:
                cand_stop = max(1e-6, c - max(self.stop_atr_mult * atr, self.stop_pct_floor * c))
                if self._active_stop_price is None or prev_w <= 1e-9:
                    self._active_stop_price = cand_stop
                else:
                    self._active_stop_price = max(self._active_stop_price, cand_stop)
            elif chosen_w < -1e-9:
                cand_stop = c + max(self.stop_atr_mult * atr, self.stop_pct_floor * c)
                if self._active_stop_price is None or prev_w >= -1e-9:
                    self._active_stop_price = cand_stop
                else:
                    self._active_stop_price = min(self._active_stop_price, cand_stop)
            else:
                self._active_stop_price = None

        out_stop = (
            round(self._active_stop_price, 6)
            if (self._active_stop_price is not None and abs(chosen_w) > 1e-9)
            else None
        )
        decisions[self.symbol] = BarDecision(
            symbol=self.symbol,
            ts_event_ms=int(ts_ms),
            target_weight=float(chosen_w),
            raw_signal=float(raw_sig),
            regime_label=f"{regime_lbl}_WEEKEND" if is_weekend else regime_lbl,
            stop_price=out_stop,
            metadata={
                "perp_spot_basis_bps": round(perp_spot_bps, 4),
                "spot_comex_basis_bps": round(spot_comex_bps, 4),
                "perp_comex_basis_bps": round(perp_comex_bps, 4),
                "paxg_spot_price": round(paxg_spot, 6),
                "comex_ref_price": round(comex_ref, 6),
                "is_weekend_guard": bool(is_weekend),
                "weekend_frozen_comex": bool(is_weekend),
                "basis_z": round(basis_z, 4),
                "weekend_dislocation_blocked": bool(weekend_dislocation_blocked),
                "effective_inertia_band": round(eff_inertia, 6),
                "funding_apr_pct": round(cur_funding_apr, 4),
                "funding_trim_scalar": round(funding_trim_scalar, 6),
                "own_bar_stop_triggered": bool(own_bar_stop_triggered),
                "macro_gold_composite": round(macro_comp, 6),
            },
        )
        return decisions
