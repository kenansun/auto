"""G1: Single-Instrument TradFi Gold Macro-Regime Core Strategy (Track G, Feature F-17).

Strict architectural & economic invariants:
  - Single-Instrument Guard (D8): Executes strictly on ONE configured primary Gold proxy
    (`GC=F` OR `GLD`). NEVER double-counts simultaneous `GC=F + GLD` legs.
  - Asymmetric Macro-Regime Core: Scales long exposure upward in falling-real-yield /
    bullish Gold macro tailwind regimes, throttles long exposure in rising-real-yield
    headwinds, strictly suppresses counter-macro shorts during bullish falling-real-yield
    regimes, and exits / shifts defensive during joint macro + trend breakdowns.
  - Pure Gold/Macro Domain Isolation: Zero Web3/crypto signal imports or formulas.
  - Full G7 Bit-Parity: Deterministic causal streaming state synchronized via
    `sync_executed_weights` and `clone_fresh`.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from alphaloop.platform.core_engine.interfaces import BarDecision
from alphaloop.platform.data_foundation.models import CanonicalBar
from alphaloop.platform.strategies.gold.macro_engine import GoldMacroEngine

_GOLD_PROXY_SYMBOLS: frozenset[str] = frozenset(
    {"GC=F", "GLD", "IAU", "MGC=F", "XAUUSD", "XAU", "PAXG"}
)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _sample_std(vals: deque[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mu = math.fsum(vals) / n
    var = math.fsum((x - mu) * (x - mu) for x in vals) / (n - 1)
    return math.sqrt(var) if (math.isfinite(var) and var > 0.0) else 0.0


def _extract_macro_field(macro_obj: Any, key: str, default: Any) -> Any:
    if isinstance(macro_obj, dict):
        return macro_obj.get(key, default)
    return getattr(macro_obj, key, default)


class GoldTradFiMacroKernel:
    """G1 Single-Instrument TradFi Gold Macro-Regime Core (`GC=F` OR `GLD`, never both)."""

    def __init__(
        self,
        symbol: str = "GC=F",
        *,
        macro_engine: GoldMacroEngine | None = None,
        allow_short: bool = True,
        fast_ema: int = 12,
        slow_ema: int = 42,
        donchian_n: int = 20,
        vol_lookback: int = 20,
        target_annual_vol_pct: float = 15.0,
        max_long_weight: float = 1.15,
        max_short_weight: float = 0.45,
        inertia_band: float = 0.04,
        stop_atr_mult: float = 3.0,
        warmup_bars: int = 20,
        reject_multi_gold_input: bool = False,
    ) -> None:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("GoldTradFiMacroKernel requires a single non-empty primary symbol")
        if any(sep in symbol for sep in (",", "+", " ", ";")):
            raise ValueError(
                f"D8 single-instrument guard: composite gold symbol {symbol!r} is forbidden"
            )
        if fast_ema < 2 or slow_ema <= fast_ema:
            raise ValueError(f"Invalid EMA spans: fast_ema={fast_ema}, slow_ema={slow_ema}")
        if donchian_n < 3 or vol_lookback < 5:
            raise ValueError(f"Invalid lookbacks: donchian_n={donchian_n}, vol_lookback={vol_lookback}")

        self.symbol: str = symbol.strip()
        self.strategy_id: str = "gold_g1_tradfi_macro_core"
        self.asset_domain: str = "gold"
        self.warmup_bars: int = max(2, int(warmup_bars))
        self.macro_engine: GoldMacroEngine | None = macro_engine
        self.allow_short: bool = bool(allow_short)
        self.fast_ema: int = int(fast_ema)
        self.slow_ema: int = int(slow_ema)
        self.donchian_n: int = int(donchian_n)
        self.vol_lookback: int = int(vol_lookback)
        self.target_annual_vol_pct: float = float(target_annual_vol_pct)
        self.max_long_weight: float = max(0.1, float(max_long_weight))
        self.max_short_weight: float = max(0.0, float(max_short_weight))
        self.inertia_band: float = max(0.0, float(inertia_band))
        self.stop_atr_mult: float = max(1.0, float(stop_atr_mult))
        self.reject_multi_gold_input: bool = bool(reject_multi_gold_input)

        self._bar_count: int = 0
        self._ema_fast: float | None = None
        self._ema_slow: float | None = None
        self._atr14: float | None = None
        self._prev_close: float | None = None
        self._closes: deque[float] = deque(maxlen=max(64, self.donchian_n + 5))
        self._highs: deque[float] = deque(maxlen=max(64, self.donchian_n + 5))
        self._lows: deque[float] = deque(maxlen=max(64, self.donchian_n + 5))
        self._log_rets: deque[float] = deque(maxlen=max(32, self.vol_lookback + 2))
        self._executed_weights: dict[str, float] = {}

    def reset(self) -> None:
        """Reset all mutable internal state for clean batch/streaming replay."""
        self._bar_count = 0
        self._ema_fast = None
        self._ema_slow = None
        self._atr14 = None
        self._prev_close = None
        self._closes.clear()
        self._highs.clear()
        self._lows.clear()
        self._log_rets.clear()
        self._executed_weights = {}

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        """Synchronize actual post-allocator/post-fill weights for Carver inertia & G7 parity."""
        self._executed_weights = {str(k): float(v) for k, v in executed_weights.items()}

    def clone_fresh(self) -> GoldTradFiMacroKernel:
        """Return a freshly reset clone sharing only the stateless GoldMacroEngine reference."""
        return GoldTradFiMacroKernel(
            symbol=self.symbol,
            macro_engine=self.macro_engine,
            allow_short=self.allow_short,
            fast_ema=self.fast_ema,
            slow_ema=self.slow_ema,
            donchian_n=self.donchian_n,
            vol_lookback=self.vol_lookback,
            target_annual_vol_pct=self.target_annual_vol_pct,
            max_long_weight=self.max_long_weight,
            max_short_weight=self.max_short_weight,
            inertia_band=self.inertia_band,
            stop_atr_mult=self.stop_atr_mult,
            warmup_bars=self.warmup_bars,
            reject_multi_gold_input=self.reject_multi_gold_input,
        )

    def _resolve_macro(
        self,
        bar: CanonicalBar,
        context: dict[str, Any] | None,
    ) -> tuple[float, float, str, float]:
        """Resolve `(macro_composite, exposure_scalar, regime_label, gvz_z)` strictly causally."""
        macro_obj = None
        if context:
            macro_obj = context.get("gold_macro")
        if macro_obj is None and self.macro_engine is not None:
            macro_obj = self.macro_engine.snapshot_as_of(bar.ts_avail_ms)

        if macro_obj is None:
            return 0.0, 1.0, "GOLD_NEUTRAL", 0.0

        comp = float(_extract_macro_field(macro_obj, "macro_gold_composite", 0.0))
        if not math.isfinite(comp):
            comp = 0.0
        comp = _clamp(comp, -1.5, 1.5)

        raw_scalar = _extract_macro_field(macro_obj, "exposure_scalar", None)
        if raw_scalar is not None and math.isfinite(float(raw_scalar)):
            exp_scalar = _clamp(float(raw_scalar), 0.20, 1.30)
        else:
            exp_scalar = _clamp(1.0 + 0.25 * comp if comp >= 0.0 else max(0.25, 1.0 + 0.65 * comp), 0.20, 1.30)

        regime_lbl = str(_extract_macro_field(macro_obj, "regime_label", ""))
        if not regime_lbl:
            if comp >= 0.20:
                regime_lbl = "GOLD_BULL_TAILWIND"
            elif comp <= -0.20:
                regime_lbl = "GOLD_BEAR_HEADWIND"
            else:
                regime_lbl = "GOLD_NEUTRAL"

        gvz_z = float(_extract_macro_field(macro_obj, "gvz_z", 0.0))
        if not math.isfinite(gvz_z):
            gvz_z = 0.0
        return comp, exp_scalar, regime_lbl, gvz_z

    def _update_indicators(self, bar: CanonicalBar) -> tuple[float, float, float, float]:
        """Update strictly causal single-instrument trend, channel, ATR, and vol state."""
        c = float(bar.close)
        h = float(bar.high)
        lo = float(bar.low)
        self._bar_count += 1

        # Compute prior Donchian channel BEFORE appending current bar (zero lookahead)
        if len(self._highs) >= 2:
            k = min(len(self._highs), self.donchian_n)
            prior_hi = max(list(self._highs)[-k:])
            prior_lo = min(list(self._lows)[-k:])
            span = prior_hi - prior_lo
            if span > 1e-9:
                donchian_pos = _clamp((2.0 * ((c - prior_lo) / span)) - 1.0, -1.2, 1.2)
            else:
                donchian_pos = 0.0
        else:
            donchian_pos = 0.0

        # True Range & Wilder ATR-14
        if self._prev_close is not None and self._prev_close > 0.0:
            tr = max(h - lo, abs(h - self._prev_close), abs(lo - self._prev_close))
            log_r = math.log(max(c, 1e-9) / self._prev_close)
            self._log_rets.append(log_r)
        else:
            tr = max(h - lo, c * 0.005)
        if self._atr14 is None:
            self._atr14 = tr
        else:
            self._atr14 = (self._atr14 * 13.0 + tr) / 14.0

        # Fast & Slow EMAs
        a_fast = 2.0 / (self.fast_ema + 1.0)
        a_slow = 2.0 / (self.slow_ema + 1.0)
        if self._ema_fast is None or self._ema_slow is None:
            self._ema_fast = c
            self._ema_slow = c
        else:
            self._ema_fast = a_fast * c + (1.0 - a_fast) * self._ema_fast
            self._ema_slow = a_slow * c + (1.0 - a_slow) * self._ema_slow

        self._closes.append(c)
        self._highs.append(h)
        self._lows.append(lo)
        self._prev_close = c

        atr = max(float(self._atr14), c * 0.004)
        ewmac_norm = math.tanh(((self._ema_fast - self._ema_slow) / atr) / 1.5)

        # Multi-week trailing return normalized by realized vol
        lb = min(len(self._closes) - 1, 20)
        if lb >= 5 and self._closes[-1 - lb] > 0.0:
            ret_lb = (c / self._closes[-1 - lb]) - 1.0
            d_vol = max(0.004, _sample_std(self._log_rets))
            ret_norm = math.tanh(ret_lb / max(0.015, d_vol * math.sqrt(float(lb))))
        else:
            ret_norm = 0.0

        tech_signal = _clamp(
            0.45 * ewmac_norm + 0.30 * _clamp(donchian_pos, -1.0, 1.0) + 0.25 * ret_norm,
            -1.0,
            1.0,
        )
        ann_vol = max(0.06, _sample_std(self._log_rets) * math.sqrt(252.0)) if len(self._log_rets) >= 5 else 0.15
        return tech_signal, atr, ann_vol, donchian_pos

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        """Emit strictly single-instrument Gold decision (suppressing any secondary Gold proxy)."""
        gold_present = [s for s in sorted(bars_at_ts) if s in _GOLD_PROXY_SYMBOLS]
        if len(gold_present) > 1 and self.reject_multi_gold_input:
            raise ValueError(
                f"D8 single-instrument violation on {self.strategy_id}: "
                f"simultaneous Gold proxies {gold_present} supplied"
            )

        decisions: dict[str, BarDecision] = {}

        # Explicitly emit 0.0 for any secondary Gold proxy present (prevents D8 double-counting)
        for sym in sorted(bars_at_ts):
            if sym == self.symbol:
                continue
            decisions[sym] = BarDecision(
                symbol=sym,
                ts_event_ms=int(ts_ms),
                target_weight=0.0,
                raw_signal=0.0,
                regime_label="GOLD_SECONDARY_SUPPRESSED",
                stop_price=None,
                metadata={
                    "d8_single_leg_suppressed": True,
                    "primary_symbol": self.symbol,
                },
            )

        bar = bars_at_ts.get(self.symbol)
        if bar is None:
            return decisions

        tech_signal, atr, ann_vol, donchian_pos = self._update_indicators(bar)
        macro_comp, exp_scalar, regime_lbl, gvz_z = self._resolve_macro(bar, context)
        prev_w = float(self._executed_weights.get(self.symbol, 0.0))

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
                    "macro_gold_composite": round(macro_comp, 6),
                    "exposure_scalar": round(exp_scalar, 6),
                    "primary_symbol": self.symbol,
                },
            )
            return decisions

        # Asymmetric macro-regime position sizing
        vol_scale = _clamp((self.target_annual_vol_pct / 100.0) / max(ann_vol, 0.06), 0.35, 1.35)
        gvz_damp = max(0.70, 1.0 - max(0.0, gvz_z - 1.5) * 0.10)

        if macro_comp >= 0.25:
            # Falling-real-yield / bullish macro tailwind:
            # Boost longs asymmetrically and STRICTLY suppress counter-macro shorts!
            macro_long_mult = 1.0 + 0.35 * min(1.5, macro_comp)
            if tech_signal > 0.02:
                base_w = (0.35 + 0.65 * tech_signal) * macro_long_mult
            elif tech_signal >= -0.25:
                # Mild dip inside bullish macro: hold reduced positive core, NEVER short
                base_w = max(0.0, 0.25 * (1.0 + tech_signal * 3.0)) * min(1.0, macro_comp)
            else:
                # Deep technical pullback inside macro bull: flatten to cash, NEVER short
                base_w = 0.0
            desired_w = _clamp(base_w * vol_scale * gvz_damp, 0.0, self.max_long_weight)
        elif macro_comp <= -0.25:
            # Rising-real-yield / bearish macro headwind:
            # Heavily throttle longs; exit/defensive short on confirmed trend breakdown
            macro_long_throttle = max(0.20, 1.0 + 0.55 * max(-1.5, macro_comp))
            if tech_signal > 0.15:
                base_w = tech_signal * macro_long_throttle * 0.70
                desired_w = _clamp(base_w * vol_scale * gvz_damp, 0.0, self.max_long_weight * 0.60)
            elif tech_signal < -0.15 and self.allow_short:
                short_strength = min(1.0, (-tech_signal - 0.10) / 0.75) * min(1.2, abs(macro_comp))
                desired_w = -_clamp(
                    short_strength * self.max_short_weight * min(1.0, vol_scale),
                    0.0,
                    self.max_short_weight,
                )
            else:
                desired_w = 0.0
        else:
            # Neutral macro regime: clean trend-following with conservative sizing
            if tech_signal > 0.08:
                base_w = tech_signal * 0.85 * exp_scalar
                desired_w = _clamp(base_w * vol_scale * gvz_damp, 0.0, self.max_long_weight)
            elif tech_signal < -0.35 and self.allow_short:
                short_strength = min(1.0, (-tech_signal - 0.30) / 0.65)
                desired_w = -_clamp(
                    short_strength * (self.max_short_weight * 0.65) * min(1.0, vol_scale),
                    0.0,
                    self.max_short_weight,
                )
            else:
                desired_w = 0.0

        # Apply Carver inertia buffer (skipping exits to flat or sign flips)
        chosen_w = desired_w
        if (
            self.inertia_band > 0.0
            and abs(prev_w) > 1e-9
            and abs(desired_w) > 1e-9
            and (prev_w * desired_w > 0.0)
            and abs(desired_w - prev_w) < self.inertia_band
        ):
            chosen_w = prev_w

        # Protective ATR stop on primary instrument's own bars
        c = float(bar.close)
        stop_price: float | None = None
        if chosen_w > 1e-9:
            sp = c - self.stop_atr_mult * atr
            if sp > 1e-6:
                stop_price = round(sp, 6)
        elif chosen_w < -1e-9:
            sp = c + self.stop_atr_mult * atr
            if math.isfinite(sp) and sp > 1e-6:
                stop_price = round(sp, 6)

        raw_signal = _clamp(tech_signal * 10.0 + macro_comp * 5.0, -20.0, 20.0)
        decisions[self.symbol] = BarDecision(
            symbol=self.symbol,
            ts_event_ms=int(ts_ms),
            target_weight=float(chosen_w),
            raw_signal=float(raw_signal),
            regime_label=regime_lbl,
            stop_price=stop_price,
            metadata={
                "primary_symbol": self.symbol,
                "macro_gold_composite": round(macro_comp, 6),
                "exposure_scalar": round(exp_scalar, 6),
                "tech_signal": round(tech_signal, 6),
                "donchian_pos": round(donchian_pos, 6),
                "ann_vol": round(ann_vol, 6),
                "gvz_z": round(gvz_z, 6),
                "d8_single_leg_enforced": True,
            },
        )
        return decisions
