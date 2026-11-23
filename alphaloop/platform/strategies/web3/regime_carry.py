"""W3: Web3 Regime-Gated Cross-Venue & Spot-Perp Funding Carry Engine (F-15).

100% crypto-native funding & basis carry engine implementing `AssetNativeStrategy`:
  - Cost-Horizon Carry Floor: blocks new entry unless expected carry over the holding horizon
    is strictly `>= 3.0 * rt_cost_bps` (eliminating ATB H6 `-2,366bp` churn).
  - Exact D11 `carry_interval_h`-normalized funding APR calculation + conservative same-sign
    `premium_basis_bps` convergence component + optional cross-venue spread input.
  - Exact perpetual funding sign convention: positive funding (longs pay shorts) takes SHORT
    perp carry exposure (`target_weight < 0`) to collect positive funding cashflows; deeply
    negative funding takes LONG perp carry exposure (`target_weight > 0`).
  - Deribit `BTC DVOL` & realized-volatility dynamic leverage throttle (`DVOL=100` sharply
    throttles leverage below `DVOL=40`; ZERO Yahoo `^VIX` or TradFi coupling).
  - Hard 50% protective margin storm-exit (`margin_ratio <= 0.50` -> immediate flat `0.0`,
    bypassing inertia buffer to prevent intrabar liquidation).
  - Explicit separated carry/margin ledger telemetry in `BarDecision.metadata` (REA-001).
  - Full G7 streaming-vs-batch bit-parity via `sync_executed_weights` & `clone_fresh()`.
"""

from __future__ import annotations

import bisect
from collections import deque
import logging
import math
from typing import Any, Sequence

from alphaloop.platform.core_engine.interfaces import BarDecision
from alphaloop.platform.data_foundation.models import CanonicalBar, Web3MarketMacroPoint
from alphaloop.platform.strategies.web3.regime_engine import is_crypto_bar

logger = logging.getLogger(__name__)

REGIME_CARRY_ACTIVE = "CARRY_ACTIVE"
REGIME_CARRY_DVOL_THROTTLED = "CARRY_DVOL_THROTTLED"
REGIME_CARRY_SUB_FLOOR = "CARRY_SUB_FLOOR"
REGIME_MARGIN_STORM_EXIT = "MARGIN_STORM_EXIT"
REGIME_CARRY_WARMUP = "CARRY_WARMUP"


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


class _W3SymbolState:
    """Strictly causal per-symbol funding carry & realized-volatility tracker."""

    __slots__ = (
        "tf_h",
        "bars_per_year",
        "carry_events",
        "carry_ema_apr",
        "log_rets",
        "prev_close",
        "bar_count",
    )

    def __init__(self, timeframe_h: float) -> None:
        self.tf_h = max(0.25, float(timeframe_h))
        self.bars_per_year = (24.0 * 365.0) / self.tf_h
        self.carry_events: deque[tuple[float, float, float]] = deque(maxlen=max(3, int(round(36.0 / self.tf_h))))
        self.carry_ema_apr: float | None = None
        self.log_rets: deque[float] = deque(maxlen=60)
        self.prev_close: float | None = None
        self.bar_count: int = 0

    def update(self, bar: CanonicalBar) -> tuple[float, float]:
        """Consume one closed bar and return `(smoothed_carry_apr_pct, realized_vol_ann_pct)`."""
        self.bar_count += 1
        c = float(bar.close)
        if self.prev_close is not None and self.prev_close > 0.0 and c > 0.0:
            self.log_rets.append(math.log(c / self.prev_close))
        self.prev_close = c

        tf_h = max(0.01, _finite_or(bar.timeframe_h, self.tf_h))
        ci_raw = _finite_or(bar.carry_interval_h, tf_h)
        ci_h = ci_raw if ci_raw > 0.0 else tf_h
        rate = _finite_or(bar.carry_rate_period, 0.0)
        self.carry_events.append((rate, tf_h, ci_h))

        sum_r = math.fsum(r for r, _, _ in self.carry_events)
        sum_tf = math.fsum(t for _, t, _ in self.carry_events)
        sum_cov = math.fsum(max(t, ci) for r, t, ci in self.carry_events if abs(r) > 1e-15)
        window_apr = (sum_r / max(sum_tf, sum_cov, 0.01)) * 8760.0 * 100.0

        alpha = 0.50  # Fast 3-bar EMA smoothing single-bar noise without missing carry regimes
        if self.carry_ema_apr is None:
            self.carry_ema_apr = window_apr
        else:
            self.carry_ema_apr = alpha * window_apr + (1.0 - alpha) * self.carry_ema_apr

        if len(self.log_rets) >= 6:
            _, sd = _mean_std(list(self.log_rets)[-20:])
            realized_vol_pct = max(5.0, sd * math.sqrt(self.bars_per_year) * 100.0)
        else:
            realized_vol_pct = 15.0

        return float(self.carry_ema_apr), realized_vol_pct


class Web3RegimeCarryKernel:
    """W3: Regime-Gated Cross-Venue & Spot-Perp Funding Carry with Cost Floor & Margin Guard."""

    asset_domain: str = "web3"

    def __init__(
        self,
        *,
        warmup_bars: int = 14,
        hold_horizon_days: float = 7.0,
        cost_floor_multiplier: float = 3.0,
        default_rt_cost_bps: float = 14.0,
        basis_weight: float = 0.25,
        base_leverage: float = 1.0,
        dvol_reference: float = 40.0,
        dvol_max_hard_cap: float = 115.0,
        margin_storm_exit_ratio: float = 0.50,
        inertia_buffer: float = 0.12,
        capital_budget_usd: float = 10_000.0,
        allow_negative_funding_carry: bool = True,
        dvol_series: Sequence[Web3MarketMacroPoint] | None = None,
        strategy_id: str = "web3_w3_regime_carry",
    ) -> None:
        self.strategy_id = str(strategy_id)
        self.asset_domain = "web3"
        self.warmup_bars = max(2, int(warmup_bars))
        self.hold_horizon_days = _clamp(float(hold_horizon_days), 1.0, 60.0)
        self.cost_floor_multiplier = max(1.0, float(cost_floor_multiplier))
        self.default_rt_cost_bps = max(0.0, float(default_rt_cost_bps))
        self.basis_weight = _clamp(float(basis_weight), 0.0, 1.0)
        self.base_leverage = _clamp(float(base_leverage), 0.10, 3.0)
        self.dvol_reference = _clamp(float(dvol_reference), 15.0, 80.0)
        self.dvol_max_hard_cap = _clamp(float(dvol_max_hard_cap), 85.0, 200.0)
        self.margin_storm_exit_ratio = _clamp(float(margin_storm_exit_ratio), 0.10, 0.90)
        self.inertia_buffer = _clamp(float(inertia_buffer), 0.0, 0.50)
        self.capital_budget_usd = max(100.0, float(capital_budget_usd))
        self.allow_negative_funding_carry = bool(allow_negative_funding_carry)

        # Optional pre-loaded PIT Deribit BTC_DVOL series (stateless bisect_right)
        self._raw_dvol_series = list(dvol_series) if dvol_series else None
        if dvol_series:
            ordered_dv = sorted(
                (p for p in dvol_series if math.isfinite(float(p.value))),
                key=lambda p: (int(p.ts_avail_ms), int(p.ts_event_ms)),
            )
            self._dvol_avails: list[int] = [int(p.ts_avail_ms) for p in ordered_dv]
            self._dvol_vals: list[float] = [float(p.value) for p in ordered_dv]
        else:
            self._dvol_avails = []
            self._dvol_vals = []

        # Mutable streaming state
        self._states: dict[str, _W3SymbolState] = {}
        self._executed_weights: dict[str, float] = {}
        self._step_count: int = 0

    def reset(self) -> None:
        """Reset all mutable internal state cleanly."""
        self._states.clear()
        self._executed_weights.clear()
        self._step_count = 0

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        """Sync actual post-allocator/post-fill weights for Carver inertia & G7 parity."""
        for sym, w in executed_weights.items():
            self._executed_weights[str(sym)] = _finite_or(w, 0.0)

    def clone_fresh(self) -> Web3RegimeCarryKernel:
        """Return a brand-new freshly reset instance with identical configuration."""
        return Web3RegimeCarryKernel(
            warmup_bars=self.warmup_bars,
            hold_horizon_days=self.hold_horizon_days,
            cost_floor_multiplier=self.cost_floor_multiplier,
            default_rt_cost_bps=self.default_rt_cost_bps,
            basis_weight=self.basis_weight,
            base_leverage=self.base_leverage,
            dvol_reference=self.dvol_reference,
            dvol_max_hard_cap=self.dvol_max_hard_cap,
            margin_storm_exit_ratio=self.margin_storm_exit_ratio,
            inertia_buffer=self.inertia_buffer,
            capital_budget_usd=self.capital_budget_usd,
            allow_negative_funding_carry=self.allow_negative_funding_carry,
            dvol_series=self._raw_dvol_series,
            strategy_id=self.strategy_id,
        )

    def _resolve_dvol(self, ts_ms: int, ctx: dict[str, Any]) -> float:
        """Resolve crypto-native Deribit BTC DVOL (strictly ignoring Yahoo ^VIX)."""
        raw = ctx.get("dvol", ctx.get("BTC_DVOL"))
        if raw is not None:
            return _clamp(_finite_or(raw, 50.0), 5.0, 250.0)
        if self._dvol_avails:
            idx = bisect.bisect_right(self._dvol_avails, int(ts_ms)) - 1
            if idx >= 0:
                return _clamp(self._dvol_vals[idx], 5.0, 250.0)
        return 50.0

    def _resolve_margin_ratio(self, sym: str, ctx: dict[str, Any]) -> float:
        raw = ctx.get("margin_ratio")
        if raw is None:
            return 1.0
        if isinstance(raw, dict):
            return _clamp(_finite_or(raw.get(sym, raw.get(sym.upper())), 1.0), 0.0, 2.0)
        return _clamp(_finite_or(raw, 1.0), 0.0, 2.0)

    def _resolve_rt_cost_bps(self, sym: str, ctx: dict[str, Any]) -> float:
        raw = ctx.get("rt_cost_bps")
        if raw is None:
            return self.default_rt_cost_bps
        if isinstance(raw, dict):
            return max(0.0, _finite_or(raw.get(sym, raw.get(sym.upper())), self.default_rt_cost_bps))
        return max(0.0, _finite_or(raw, self.default_rt_cost_bps))

    def _resolve_cross_venue_apr(self, sym: str, ctx: dict[str, Any]) -> float | None:
        raw = ctx.get("cross_venue_carry_apr_pct", ctx.get("venue_spread_apr_pct"))
        if raw is None:
            return None
        if isinstance(raw, dict):
            val = raw.get(sym, raw.get(sym.upper()))
            return _finite_or(val, 0.0) if val is not None else None
        return _finite_or(raw, 0.0)

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        """Consume closed crypto perp bars at `ts_ms` and emit strictly causal W3 carry decisions."""
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

        dvol = self._resolve_dvol(int(ts_ms), ctx)

        decisions: dict[str, BarDecision] = {}
        for sym in sorted(non_crypto_syms):
            decisions[sym] = BarDecision(
                symbol=sym,
                ts_event_ms=int(ts_ms),
                target_weight=0.0,
                raw_signal=0.0,
                regime_label=REGIME_CARRY_SUB_FLOOR,
                stop_price=None,
                metadata={
                    "separated_margin_ledger": True,
                    "ignored_non_crypto": True,
                    "allocated_margin_usd": 0.0,
                },
            )

        # First pass: update causal symbol states & evaluate cost-horizon carry floor
        eval_rows: dict[str, dict[str, Any]] = {}
        active_candidates: list[str] = []

        for sym in sorted(crypto_bars):
            bar = crypto_bars[sym]
            state = self._states.get(sym)
            if state is None:
                state = _W3SymbolState(float(bar.timeframe_h))
                self._states[sym] = state

            carry_apr_pct, realized_vol_pct = state.update(bar)
            cv_apr = self._resolve_cross_venue_apr(sym, ctx)
            eff_carry_apr = (
                cv_apr
                if (cv_apr is not None and abs(cv_apr) > abs(carry_apr_pct))
                else carry_apr_pct
            )

            funding_horizon_bps = (eff_carry_apr / 100.0) * (self.hold_horizon_days / 365.0) * 10_000.0
            prem_bps = _finite_or(getattr(bar, "premium_basis_bps", 0.0), 0.0)
            basis_contrib_bps = (
                self.basis_weight * prem_bps
                if (abs(eff_carry_apr) > 1e-6 and prem_bps * eff_carry_apr > 0.0)
                else 0.0
            )
            expected_carry_bps = funding_horizon_bps + basis_contrib_bps

            rt_cost_bps = self._resolve_rt_cost_bps(sym, ctx)
            cost_floor_bps = self.cost_floor_multiplier * rt_cost_bps
            margin_ratio = self._resolve_margin_ratio(sym, ctx)
            storm_exit = margin_ratio <= self.margin_storm_exit_ratio

            # DVOL + realized-vol dynamic leverage scalar
            eff_vol = max(dvol, realized_vol_pct * 0.80)
            if dvol >= self.dvol_max_hard_cap:
                dvol_scalar = 0.0
            else:
                dvol_scalar = min(1.0, (self.dvol_reference / max(self.dvol_reference, eff_vol)) ** 1.25)

            prev_w = self._executed_weights.get(sym, 0.0)
            in_warmup = state.bar_count < self.warmup_bars

            # Direction: positive funding -> SHORT perp (-1); negative funding -> LONG perp (+1)
            if expected_carry_bps > 0.0:
                direction = -1.0
            elif expected_carry_bps < 0.0 and self.allow_negative_funding_carry:
                direction = +1.0
            else:
                direction = 0.0

            abs_exp = abs(expected_carry_bps)
            holding_same_dir = (abs(prev_w) > 1e-9) and (prev_w * direction > 0.0)

            if in_warmup or storm_exit or direction == 0.0 or dvol_scalar <= 1e-9:
                floor_passed = False
            elif holding_same_dir:
                # Hold hysteresis: remain in carry position while >= 50% of entry floor (1.5x RT fee)
                floor_passed = abs_exp >= (0.50 * cost_floor_bps)
            else:
                # Strict entry gate: expected carry over hold horizon MUST be >= 3x RT fee!
                floor_passed = abs_exp >= cost_floor_bps

            if floor_passed:
                active_candidates.append(sym)

            eval_rows[sym] = {
                "carry_apr_pct": eff_carry_apr,
                "expected_carry_bps": expected_carry_bps,
                "cost_floor_bps": cost_floor_bps,
                "rt_cost_bps": rt_cost_bps,
                "cost_floor_passed": floor_passed,
                "dvol": dvol,
                "dvol_scalar": dvol_scalar,
                "realized_vol_pct": realized_vol_pct,
                "margin_ratio": margin_ratio,
                "storm_exit": storm_exit,
                "in_warmup": in_warmup,
                "direction": direction,
                "prev_w": prev_w,
                "premium_basis_bps": prem_bps,
            }

        # Second pass: allocate separated carry margin budget across active passing symbols
        n_active = max(1, len(active_candidates))
        per_sym_cap = self.base_leverage / float(n_active)

        for sym in sorted(crypto_bars):
            row = eval_rows[sym]
            prev_w = float(row["prev_w"])
            exp_bps = float(row["expected_carry_bps"])
            floor_bps = float(row["cost_floor_bps"])
            dvol_scalar = float(row["dvol_scalar"])
            storm_exit = bool(row["storm_exit"])
            in_warmup = bool(row["in_warmup"])
            floor_passed = bool(row["cost_floor_passed"])

            if storm_exit:
                desired_w = 0.0
                final_w = 0.0
                inertia_held = False
                regime_label = REGIME_MARGIN_STORM_EXIT
            elif in_warmup:
                desired_w = 0.0
                final_w = 0.0
                inertia_held = False
                regime_label = REGIME_CARRY_WARMUP
            elif not floor_passed:
                desired_w = 0.0
                final_w = 0.0
                inertia_held = False
                regime_label = REGIME_CARRY_SUB_FLOOR
            else:
                carry_strength = _clamp(abs(exp_bps) / max(floor_bps * 2.5, 1.0), 0.35, 1.0)
                desired_w = float(row["direction"]) * per_sym_cap * dvol_scalar * carry_strength

                # Carver inertia ONLY when not storm-exiting, same direction, and inside band
                if abs(prev_w) > 1e-9 and prev_w * desired_w > 0.0 and self.inertia_buffer > 0.0:
                    band = max(0.015, self.inertia_buffer * max(abs(desired_w), abs(prev_w)))
                    if abs(desired_w - prev_w) <= band:
                        final_w = prev_w
                        inertia_held = True
                    else:
                        final_w = desired_w
                        inertia_held = False
                else:
                    final_w = desired_w
                    inertia_held = False

                regime_label = (
                    REGIME_CARRY_DVOL_THROTTLED if dvol_scalar < 0.70 else REGIME_CARRY_ACTIVE
                )

            allocated_margin_usd = abs(final_w) * self.capital_budget_usd
            reserve_margin_usd = max(0.0, self.capital_budget_usd - allocated_margin_usd)
            raw_sig = 0.0 if (in_warmup or storm_exit) else float(row["direction"]) * (abs(exp_bps) / max(floor_bps, 1.0))

            decisions[sym] = BarDecision(
                symbol=sym,
                ts_event_ms=int(ts_ms),
                target_weight=float(final_w),
                raw_signal=float(raw_sig),
                regime_label=regime_label,
                stop_price=None,
                metadata={
                    "separated_margin_ledger": True,
                    "carry_apr_pct": round(float(row["carry_apr_pct"]), 4),
                    "expected_carry_bps": round(exp_bps, 4),
                    "cost_floor_bps": round(floor_bps, 4),
                    "rt_cost_bps": round(float(row["rt_cost_bps"]), 4),
                    "cost_floor_passed": bool(floor_passed),
                    "dvol": round(float(row["dvol"]), 4),
                    "dvol_leverage_scalar": round(dvol_scalar, 4),
                    "realized_vol_ann_pct": round(float(row["realized_vol_pct"]), 4),
                    "margin_ratio": round(float(row["margin_ratio"]), 4),
                    "storm_exit_triggered": bool(storm_exit),
                    "allocated_margin_usd": round(allocated_margin_usd, 2),
                    "reserve_margin_usd": round(reserve_margin_usd, 2),
                    "premium_basis_bps": round(float(row["premium_basis_bps"]), 4),
                    "inertia_held": bool(inertia_held),
                },
            )

        return decisions
