"""G4: Gold vs Gold Miners (`GLD` / `GDX`) Causal Volatility-Matched Mean-Reversion Kernel.

Track G — Physically isolated from Web3 and CN A-Share domains (zero cross-domain imports).

Economic Rationale (Orthogonal to `G1` Directional Gold Macro Trend):
  - Physical Gold (`GLD`) and Gold Mining Equities (`GDX`) share bullion revenue fundamentals
    yet `GDX` exhibits ~2x realized volatility due to operational leverage and equity-market
    risk-on/risk-off overshoots.
  - Strictly causal trailing realized-volatility matching (`w_gld * σ_gld == -w_gdx * σ_gdx`)
    neutralizes directional gold & equity beta while harvesting persistent mean-reversion of
    the `log(GLD / GDX)` relative valuation spread.
  - Full `G7` Bit-Exact Batch/Streaming Parity (`reset()`, `clone_fresh()`, `sync_executed_weights()`).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from alphaloop.platform.core_engine.interfaces import BarDecision
from alphaloop.platform.data_foundation.models import CanonicalBar


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _sample_mean_std(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n < 2:
        return (vals[0] if vals else 0.0), 0.0
    mu = math.fsum(vals) / n
    var = math.fsum((x - mu) * (x - mu) for x in vals) / (n - 1)
    return mu, (math.sqrt(var) if (math.isfinite(var) and var > 0.0) else 0.0)


class GoldMinerSpreadMRKernel:
    """G4: Causal Volatility-Matched `GLD` vs `GDX` Mean-Reversion Orthogonal Gold Sleeve."""

    asset_domain: str = "gold"

    def __init__(
        self,
        *,
        gold_symbol: str = "GLD",
        miner_symbol: str = "GDX",
        lookback_days: int = 60,
        vol_lookback_days: int = 40,
        z_enter: float = 1.0,
        z_exit: float = 0.25,
        z_scale: float = 2.5,
        max_gross_weight: float = 0.90,
        max_single_leg_weight: float = 0.65,
        min_vol_ratio: float = 1.0,
        max_vol_ratio: float = 3.0,
        inertia_band: float = 0.08,
        warmup_bars: int = 62,
        strategy_id: str = "gold_g4_miner_spread_mr",
    ) -> None:
        if gold_symbol.strip() == miner_symbol.strip():
            raise ValueError("gold_symbol and miner_symbol must be distinct instruments")
        self.strategy_id: str = str(strategy_id)
        self.asset_domain: str = "gold"
        self.gold_symbol: str = gold_symbol.strip().upper()
        self.miner_symbol: str = miner_symbol.strip().upper()
        self.lookback_days: int = max(10, int(lookback_days))
        self.vol_lookback_days: int = max(10, int(vol_lookback_days))
        self.z_enter: float = max(0.25, float(z_enter))
        self.z_exit: float = max(0.0, min(self.z_enter * 0.8, float(z_exit)))
        self.z_scale: float = max(0.5, float(z_scale))
        self.max_gross_weight: float = _clamp(float(max_gross_weight), 0.10, 1.50)
        self.max_single_leg_weight: float = _clamp(float(max_single_leg_weight), 0.05, 1.0)
        self.min_vol_ratio: float = max(0.5, float(min_vol_ratio))
        self.max_vol_ratio: float = max(self.min_vol_ratio, float(max_vol_ratio))
        self.inertia_band: float = _clamp(float(inertia_band), 0.0, 0.35)
        self.warmup_bars: int = max(self.lookback_days + 2, self.vol_lookback_days + 2, int(warmup_bars))

        self._step_count: int = 0
        self._prev_gld: float | None = None
        self._prev_gdx: float | None = None
        self._last_paired_ts_ms: int | None = None
        self._gld_rets: deque[float] = deque(maxlen=max(64, self.vol_lookback_days + 5))
        self._gdx_rets: deque[float] = deque(maxlen=max(64, self.vol_lookback_days + 5))
        self._log_ratios: deque[float] = deque(maxlen=max(140, self.lookback_days + 5))
        self._executed_weights: dict[str, float] = {}

    def reset(self) -> None:
        self._step_count = 0
        self._prev_gld = None
        self._prev_gdx = None
        self._last_paired_ts_ms = None
        self._gld_rets.clear()
        self._gdx_rets.clear()
        self._log_ratios.clear()
        self._executed_weights.clear()

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        for sym, w in executed_weights.items():
            wf = float(w)
            self._executed_weights[str(sym)] = wf if math.isfinite(wf) else 0.0

    def clone_fresh(self) -> GoldMinerSpreadMRKernel:
        return GoldMinerSpreadMRKernel(
            gold_symbol=self.gold_symbol,
            miner_symbol=self.miner_symbol,
            lookback_days=self.lookback_days,
            vol_lookback_days=self.vol_lookback_days,
            z_enter=self.z_enter,
            z_exit=self.z_exit,
            z_scale=self.z_scale,
            max_gross_weight=self.max_gross_weight,
            max_single_leg_weight=self.max_single_leg_weight,
            min_vol_ratio=self.min_vol_ratio,
            max_vol_ratio=self.max_vol_ratio,
            inertia_band=self.inertia_band,
            warmup_bars=self.warmup_bars,
            strategy_id=self.strategy_id,
        )

    def _hold_or_flat_unpaired(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        *,
        flat_label: str,
    ) -> dict[str, BarDecision]:
        """RB-05: never close only one leg when the partner leg is missing; preserve hedge until paired."""
        prev_w_g = float(self._executed_weights.get(self.gold_symbol, 0.0))
        prev_w_m = float(self._executed_weights.get(self.miner_symbol, 0.0))
        has_open_pair = abs(prev_w_g) > 1e-9 or abs(prev_w_m) > 1e-9
        label = "G4_UNPAIRED_HOLD_EXISTING" if has_open_pair else flat_label
        out: dict[str, BarDecision] = {}
        for sym in sorted(bars_at_ts):
            held_w = float(self._executed_weights.get(sym, 0.0)) if has_open_pair else 0.0
            out[sym] = BarDecision(
                symbol=sym,
                ts_event_ms=int(ts_ms),
                target_weight=held_w,
                raw_signal=0.0,
                regime_label=label,
                stop_price=None,
                metadata={"paired": False, "preserved_open_hedge": has_open_pair},
            )
        return out

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        _ = context
        g_bar = bars_at_ts.get(self.gold_symbol)
        m_bar = bars_at_ts.get(self.miner_symbol)
        if g_bar is None or m_bar is None:
            return self._hold_or_flat_unpaired(ts_ms, bars_at_ts, flat_label="G4_UNPAIRED_HOLD_FLAT")

        g_px = float(g_bar.close)
        m_px = float(m_bar.close)
        if not (math.isfinite(g_px) and g_px > 0.0 and math.isfinite(m_px) and m_px > 0.0):
            return self._hold_or_flat_unpaired(ts_ms, bars_at_ts, flat_label="G4_INVALID_PX")

        self._step_count += 1
        if self._prev_gld is not None and self._prev_gdx is not None:
            # RB-17: normalize or skip extreme multi-day missing-bar gaps so 1-day vol std stays accurate
            gap_days = (
                max(1.0, float(int(ts_ms) - self._last_paired_ts_ms) / 86_400_000.0)
                if self._last_paired_ts_ms is not None
                else 1.0
            )
            if gap_days <= 5.0:
                dt_scale = math.sqrt(max(1.0, gap_days / 3.0)) if gap_days > 3.5 else 1.0
                self._gld_rets.append(((g_px / self._prev_gld) - 1.0) / dt_scale)
                self._gdx_rets.append(((m_px / self._prev_gdx) - 1.0) / dt_scale)
        self._prev_gld = g_px
        self._prev_gdx = m_px
        self._last_paired_ts_ms = int(ts_ms)
        log_ratio = math.log(g_px / m_px)
        self._log_ratios.append(log_ratio)

        prev_w_g = float(self._executed_weights.get(self.gold_symbol, 0.0))
        prev_w_m = float(self._executed_weights.get(self.miner_symbol, 0.0))

        w_g = 0.0
        w_m = 0.0
        z_val = 0.0
        vol_ratio = 2.0
        regime = "G4_WARMUP"

        if (
            self._step_count >= self.warmup_bars
            and len(self._log_ratios) >= self.lookback_days
            and len(self._gld_rets) >= self.vol_lookback_days
            and len(self._gdx_rets) >= self.vol_lookback_days
        ):
            win = list(self._log_ratios)[-self.lookback_days :]
            mu, sd = _sample_mean_std(win)
            z_val = (log_ratio - mu) / sd if sd > 1e-9 else 0.0

            _, vol_g = _sample_mean_std(list(self._gld_rets)[-self.vol_lookback_days :])
            _, vol_m = _sample_mean_std(list(self._gdx_rets)[-self.vol_lookback_days :])
            vol_ratio = _clamp(vol_m / max(vol_g, 1e-6), self.min_vol_ratio, self.max_vol_ratio)

            if abs(z_val) >= self.z_enter:
                sig = _clamp(-z_val / self.z_scale, -1.0, 1.0)
                scale = self.max_gross_weight / (vol_ratio + 1.0)
                w_g = sig * vol_ratio * scale
                w_m = -sig * 1.0 * scale
                # RB-17: proportionally clamp BOTH legs if either exceeds max_single_leg_weight
                max_leg = max(abs(w_g), abs(w_m))
                if max_leg > self.max_single_leg_weight and max_leg > 1e-9:
                    shrink = self.max_single_leg_weight / max_leg
                    w_g *= shrink
                    w_m *= shrink
                regime = "G4_LONG_GLD_SHORT_GDX" if w_g > 0.0 else "G4_SHORT_GLD_LONG_GDX"
            elif abs(z_val) <= self.z_exit:
                w_g = 0.0
                w_m = 0.0
                regime = "G4_MEAN_REVERTED_FLAT"
            else:
                # Hysteresis zone: hold existing paired spread until mean-reversion exit
                w_g = prev_w_g
                w_m = prev_w_m
                regime = "G4_HYSTERESIS_HOLD" if (abs(w_g) > 1e-6 or abs(w_m) > 1e-6) else "G4_NEUTRAL_FLAT"

            # Carver inertia suppression for small continuous adjustments
            if (
                self.inertia_band > 0.0
                and abs(w_g) > 1e-6
                and abs(prev_w_g) > 1e-6
                and abs(w_g - prev_w_g) <= self.inertia_band
                and abs(w_m - prev_w_m) <= self.inertia_band
            ):
                w_g = prev_w_g
                w_m = prev_w_m

        # Raw signal carries causal spread mean-reversion direction (-z) for deterministic allocator ranking
        raw_sig = _clamp(-z_val / self.z_scale, -1.0, 1.0)
        meta = {
            "log_ratio": round(log_ratio, 6),
            "spread_z": round(z_val, 4),
            "causal_vol_ratio": round(vol_ratio, 4),
            "paired_spread": True,
        }
        return {
            self.gold_symbol: BarDecision(
                symbol=self.gold_symbol,
                ts_event_ms=int(ts_ms),
                target_weight=float(w_g),
                raw_signal=float(raw_sig),
                regime_label=regime,
                stop_price=None,
                metadata={"leg": "GOLD_BULLION", **meta},
            ),
            self.miner_symbol: BarDecision(
                symbol=self.miner_symbol,
                ts_event_ms=int(ts_ms),
                target_weight=float(w_m),
                raw_signal=float(-raw_sig),
                regime_label=regime,
                stop_price=None,
                metadata={"leg": "GOLD_MINERS", **meta},
            ),
        }
