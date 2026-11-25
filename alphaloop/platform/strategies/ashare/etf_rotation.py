"""A1: CN A-Share Native Risk-Adjusted ETF Momentum & Defensive CGB Bond Rotation Kernel.

Physically isolated from Web3 and Gold strategy modules (zero cross-domain imports).
Subclasses `AShareStrategyInterface` so EVERY decision strictly enforces CN A-Share physics:
  - Strict Long-Only (`target_weight >= 0.0`)
  - Exchange-Local CST (`Asia/Shanghai`) `T+1` sell lock (shares bought on CST trade date D
    cannot be sold or reduced until CST trade date > D)
  - Board-Aware Daily Price Limits (`±10%` Mainboard, `±20%` ChiNext `300/301` & STAR `688/689`
    & `159915.SZ`/`588000.SS` ETF overrides; blocks buys at limit-up and sells at limit-down)
  - Sell-Only Stamp Duty metadata (`5.0 bps` conservative stress default via `COST_CN_A_SHARE`)
  - Defensive rotation into China Government Bond ETFs (`511010.SS` / `511260.SS`) or cash
    when equity risk ETFs fail absolute trend & positive risk-adjusted momentum filters
  - Full `G7` Bit-Exact Batch/Streaming Parity (`reset()`, `clone_fresh()`, `sync_executed_weights()`)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Sequence

from alphaloop.platform.assets import COST_CN_A_SHARE
from alphaloop.platform.core_engine.interfaces import BarDecision
from alphaloop.platform.data_foundation.models import CanonicalBar
from alphaloop.platform.strategies.ashare_contract import (
    AShareStrategyInterface,
    cn_exchange_trade_date,
)

_CHINEXT_STAR_ETF_20PCT: frozenset[str] = frozenset({"159915.SZ", "588000.SS", "588080.SS", "159949.SZ"})
_DEFAULT_DEFENSIVE_BONDS: tuple[str, ...] = ("511010.SS", "511260.SS")


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _sample_std(vals: Sequence[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mu = math.fsum(vals) / n
    var = math.fsum((x - mu) * (x - mu) for x in vals) / (n - 1)
    return math.sqrt(var) if (math.isfinite(var) and var > 0.0) else 0.0


class AShareETFRotationKernel(AShareStrategyInterface):
    """A1 CN A-Share Native Long-Only T+1 Risk-Adjusted Momentum + CGB Bond Defensive Rotation."""

    def __init__(
        self,
        *,
        strategy_id: str = "ashare_a1_etf_rotation",
        defensive_symbols: Sequence[str] = _DEFAULT_DEFENSIVE_BONDS,
        top_k: int = 2,
        fast_mom_days: int = 20,
        slow_mom_days: int = 60,
        vol_lookback: int = 20,
        trend_ema_span: int = 50,
        rank_hysteresis: int = 1,
        inertia_buffer: float = 0.25,
        rebalance_every_bars: int = 5,
        max_single_weight: float = 0.55,
        max_total_weight: float = 0.95,
        target_annual_vol_pct: float = 12.0,
        defensive_fallback_weight: float = 0.85,
        sell_stamp_duty_bps: float = COST_CN_A_SHARE.stamp_duty_sell_bps,
        warmup_bars: int = 62,
    ) -> None:
        super().__init__(
            strategy_id=strategy_id,
            sell_stamp_duty_bps=sell_stamp_duty_bps,
            max_single_weight=max_single_weight,
            warmup_bars=warmup_bars,
        )
        self.defensive_symbols: tuple[str, ...] = tuple(str(s).strip().upper() for s in defensive_symbols)
        self.top_k: int = max(1, int(top_k))
        self.fast_mom_days: int = max(5, int(fast_mom_days))
        self.slow_mom_days: int = max(self.fast_mom_days + 5, int(slow_mom_days))
        self.vol_lookback: int = max(5, int(vol_lookback))
        self.trend_ema_span: int = max(10, int(trend_ema_span))
        self.rank_hysteresis: int = max(0, int(rank_hysteresis))
        self.inertia_buffer: float = _clamp(float(inertia_buffer), 0.0, 0.50)
        self.rebalance_every_bars: int = max(1, int(rebalance_every_bars))
        self.max_total_weight: float = _clamp(float(max_total_weight), 0.10, 1.0)
        self.target_annual_vol_pct: float = max(2.0, float(target_annual_vol_pct))
        self.defensive_fallback_weight: float = _clamp(float(defensive_fallback_weight), 0.0, 1.0)

        self._step_count: int = 0
        self._closes: dict[str, deque[float]] = {}
        self._log_rets: dict[str, deque[float]] = {}
        self._ema_trend: dict[str, float] = {}
        self._pending_exit_targets: dict[str, float] = {}

    def reset(self) -> None:
        super().reset()
        self._step_count = 0
        self._closes.clear()
        self._log_rets.clear()
        self._ema_trend.clear()
        self._pending_exit_targets.clear()

    def clone_fresh(self) -> AShareETFRotationKernel:
        return AShareETFRotationKernel(
            strategy_id=self.strategy_id,
            defensive_symbols=self.defensive_symbols,
            top_k=self.top_k,
            fast_mom_days=self.fast_mom_days,
            slow_mom_days=self.slow_mom_days,
            vol_lookback=self.vol_lookback,
            trend_ema_span=self.trend_ema_span,
            rank_hysteresis=self.rank_hysteresis,
            inertia_buffer=self.inertia_buffer,
            rebalance_every_bars=self.rebalance_every_bars,
            max_single_weight=self.max_single_weight,
            max_total_weight=self.max_total_weight,
            target_annual_vol_pct=self.target_annual_vol_pct,
            defensive_fallback_weight=self.defensive_fallback_weight,
            sell_stamp_duty_bps=self.sell_stamp_duty_bps,
            warmup_bars=self.warmup_bars,
        )

    def resolve_price_limit_pct(self, symbol: str) -> float:
        """Recognize ±20% ChiNext/STAR ETFs (`159915.SZ`, `588000.SS`) alongside stock prefixes."""
        s = symbol.strip().upper()
        if self.daily_price_limit_pct is None and s in _CHINEXT_STAR_ETF_20PCT:
            return 20.0
        return super().resolve_price_limit_pct(symbol)

    def _update_symbol_history(self, sym: str, close_px: float) -> None:
        dq_c = self._closes.get(sym)
        if dq_c is None:
            dq_c = deque(maxlen=self.slow_mom_days + 5)
            self._closes[sym] = dq_c
        dq_r = self._log_rets.get(sym)
        if dq_r is None:
            dq_r = deque(maxlen=max(32, self.vol_lookback + 5))
            self._log_rets[sym] = dq_r

        if dq_c and dq_c[-1] > 0.0 and close_px > 0.0:
            dq_r.append(math.log(close_px / dq_c[-1]))
        dq_c.append(close_px)

        alpha = 2.0 / (self.trend_ema_span + 1.0)
        prev_ema = self._ema_trend.get(sym)
        self._ema_trend[sym] = close_px if prev_ema is None else (alpha * close_px + (1.0 - alpha) * prev_ema)

    def _score_symbol(self, sym: str) -> tuple[float, float, bool]:
        """Return `(risk_adjusted_mom_score, annualized_vol_frac, trend_pass)` strictly causally."""
        closes = self._closes.get(sym)
        rets = self._log_rets.get(sym)
        if not closes or len(closes) <= self.slow_mom_days or not rets or len(rets) < self.vol_lookback:
            return 0.0, 0.20, False

        c_now = closes[-1]
        c_fast = closes[-1 - self.fast_mom_days]
        c_slow = closes[-1 - self.slow_mom_days]
        if c_now <= 0.0 or c_fast <= 0.0 or c_slow <= 0.0:
            return 0.0, 0.20, False

        r_fast = math.log(c_now / c_fast)
        r_slow = math.log(c_now / c_slow)
        daily_vol = _sample_std(list(rets)[-self.vol_lookback :])
        ann_vol = max(0.04, daily_vol * math.sqrt(242.0))
        # Blend 20d + 60d momentum scaled by realized vol (A-share high-beta penalty)
        blended = 0.55 * (r_fast * (242.0 / self.fast_mom_days)) + 0.45 * (r_slow * (242.0 / self.slow_mom_days))
        score = blended / ann_vol
        ema_val = self._ema_trend.get(sym, c_now)
        trend_ok = (c_now > ema_val) and (blended > 0.0)
        return float(score), float(ann_vol), bool(trend_ok)

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        """Compute cross-ETF A-share weights and enforce all CN A-share market physics via super()."""
        self._step_count += 1
        ordered_syms = sorted(bars_at_ts.keys())
        printing_set = set(ordered_syms)
        for sym in ordered_syms:
            bar = bars_at_ts[sym]
            if math.isfinite(float(bar.close)) and float(bar.close) > 0.0:
                self._update_symbol_history(sym, float(bar.close))

        # Clear any pending exits that have now actually filled
        for sym in list(self._pending_exit_targets.keys()):
            cur_exec = max(0.0, float(self._executed_weights.get(sym, 0.0)))
            if cur_exec <= self._pending_exit_targets[sym] + 0.02:
                self._pending_exit_targets.pop(sym, None)

        # RB-15: reserve gross capacity already locked in non-printing/suspended held ETFs
        locked_non_printing_w = math.fsum(
            max(0.0, float(w)) for s, w in self._executed_weights.items() if s not in printing_set
        )
        available_total_w = max(0.0, self.max_total_weight - locked_non_printing_w)

        desired_weights: dict[str, float] = {sym: 0.0 for sym in ordered_syms}
        raw_scores: dict[str, float] = {sym: 0.0 for sym in ordered_syms}
        regime_label = "ASHARE_WARMUP"

        if self._step_count >= self.warmup_bars:
            is_rebal_bar = (
                self.rebalance_every_bars <= 1
                or ((self._step_count - self.warmup_bars) % self.rebalance_every_bars == 0)
                or all(self._executed_weights.get(s, 0.0) <= 1e-6 for s in ordered_syms)
            )
            def_set = set(self.defensive_symbols)
            risk_candidates: list[tuple[float, str, float]] = []
            def_candidates: list[tuple[float, str, float]] = []

            for sym in ordered_syms:
                sc, ann_v, trend_ok = self._score_symbol(sym)
                raw_scores[sym] = sc
                if sym in def_set:
                    if sc >= -0.25:
                        def_candidates.append((sc, sym, ann_v))
                elif trend_ok and sc > 0.15:
                    risk_candidates.append((sc, sym, ann_v))

            if not is_rebal_bar:
                # Hold existing long positions between scheduled rebalances, EXCEPT immediately retry
                # any prior rebalance exit/reduction blocked by limit-down or T+1 lock (RB-14)!
                retried_any = False
                for sym in ordered_syms:
                    cur_w = max(0.0, float(self._executed_weights.get(sym, 0.0)))
                    if sym in self._pending_exit_targets and cur_w > self._pending_exit_targets[sym] + 1e-6:
                        desired_weights[sym] = self._pending_exit_targets[sym]
                        retried_any = True
                    else:
                        desired_weights[sym] = cur_w
                regime_label = "ASHARE_RETRY_BLOCKED_EXIT" if retried_any else "ASHARE_HOLD_CADENCE"
            else:
                risk_candidates.sort(key=lambda item: (-item[0], item[1]))
                def_candidates.sort(key=lambda item: (-item[0], item[1]))

                selected: list[tuple[str, float]] = []
                if risk_candidates:
                    regime_label = "ASHARE_RISK_ON_MOM"
                    hyst_cutoff = self.top_k + self.rank_hysteresis
                    hyst_pool = {sym for _, sym, _ in risk_candidates[:hyst_cutoff]}
                    vol_by_sym = {sym: ann_v for _, sym, ann_v in risk_candidates}

                    held_kept = [
                        sym
                        for _, sym, _ in risk_candidates[:hyst_cutoff]
                        if self._executed_weights.get(sym, 0.0) > 0.05 and sym in hyst_pool
                    ][: self.top_k]
                    for sym in held_kept:
                        selected.append((sym, vol_by_sym[sym]))
                    for _, sym, ann_v in risk_candidates:
                        if len(selected) >= self.top_k:
                            break
                        if all(s != sym for s, _ in selected):
                            selected.append((sym, ann_v))

                    target_vol = self.target_annual_vol_pct / 100.0
                    inv_vols = {sym: 1.0 / max(0.06, ann_v) for sym, ann_v in selected}
                    inv_sum = math.fsum(inv_vols.values())
                    raw_w: dict[str, float] = {}
                    for sym, ann_v in selected:
                        base_share = inv_vols[sym] / inv_sum if inv_sum > 0.0 else (1.0 / len(selected))
                        scaled = base_share * min(1.0, target_vol / max(0.06, ann_v))
                        raw_w[sym] = _clamp(scaled, 0.0, self.max_single_weight)

                    tot_risk = math.fsum(raw_w.values())
                    if tot_risk > available_total_w and tot_risk > 1e-9:
                        scale = available_total_w / tot_risk
                        raw_w = {s: w * scale for s, w in raw_w.items()}
                        tot_risk = available_total_w

                    for s, w in raw_w.items():
                        desired_weights[s] = w

                    residual = max(0.0, available_total_w - tot_risk)
                    if residual > 0.10 and def_candidates:
                        best_bond = def_candidates[0][1]
                        desired_weights[best_bond] = min(self.max_single_weight, residual)
                elif def_candidates:
                    regime_label = "ASHARE_DEFENSIVE_CGB_BOND"
                    n_bonds = min(2, len(def_candidates))
                    cap_def = min(self.defensive_fallback_weight, available_total_w)
                    per_bond = min(self.max_single_weight, cap_def / n_bonds)
                    for _, b_sym, _ in def_candidates[:n_bonds]:
                        desired_weights[b_sym] = per_bond
                else:
                    regime_label = "ASHARE_DEFENSIVE_CASH"

                if self.inertia_buffer > 0.0:
                    for sym in ordered_syms:
                        prev_w = max(0.0, float(self._executed_weights.get(sym, 0.0)))
                        des_w = desired_weights[sym]
                        if prev_w > 1e-6 and des_w > 1e-6:
                            band = max(0.04, self.inertia_buffer * max(prev_w, des_w))
                            if abs(des_w - prev_w) <= band:
                                desired_weights[sym] = prev_w

                gross_after = math.fsum(desired_weights.values())
                if gross_after > available_total_w and gross_after > 1e-9:
                    scale = available_total_w / gross_after
                    desired_weights = {s: w * scale for s, w in desired_weights.items()}

                # Record any requested exits/reductions so blocked fills retry immediately next bar (RB-14)
                for sym in ordered_syms:
                    prev_w = max(0.0, float(self._executed_weights.get(sym, 0.0)))
                    if desired_weights[sym] < prev_w - 0.02:
                        self._pending_exit_targets[sym] = desired_weights[sym]
                    else:
                        self._pending_exit_targets.pop(sym, None)

        # Delegate final T+1 lock, ±10%/±20% limit-lock, long-only & stamp-duty guards to AShareStrategyInterface!
        merged_ctx: dict[str, Any] = dict(context) if context else {}
        merged_ctx["desired_weights"] = desired_weights
        decisions = super().on_bar_step(ts_ms, bars_at_ts, merged_ctx)

        # Preserve exact A1 raw_signal & regime label while retaining all AShareStrategyInterface physics metadata
        enriched: dict[str, BarDecision] = {}
        for sym, dec in decisions.items():
            meta = dict(dec.metadata or {})
            meta["a1_regime"] = regime_label
            meta["a1_score"] = round(raw_scores.get(sym, 0.0), 6)
            meta["a1_is_defensive_bond"] = sym in self.defensive_symbols
            enriched[sym] = BarDecision(
                symbol=dec.symbol,
                ts_event_ms=dec.ts_event_ms,
                target_weight=dec.target_weight,
                raw_signal=float(raw_scores.get(sym, 0.0)),
                regime_label=regime_label,
                stop_price=dec.stop_price,
                metadata=meta,
            )
        return enriched
