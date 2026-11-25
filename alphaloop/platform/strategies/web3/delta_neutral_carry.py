"""W3-DN: Web3 True Dual-Leg Spot+Perp Delta-Neutral Funding Carry Engine (Phase B).

Eliminates Phase A single-leg W3 directional price drag while accounting for 100% of real
execution economics inside `PortfolioAccountingEngine`:
  - Atomic Spot+Perp Pairing: for every active coin `SYM`, emits simultaneous LONG spot
    (`{SYM}_SPOT`, `target_weight = +w_leg`, Binance Spot fee+spread, `allow_short=False`)
    and SHORT perp (`{SYM}_PERP`, `target_weight = -w_leg`, Hyperliquid/Binance UM fee+spread
    and real periodic funding cashflow), guaranteeing `w_spot + w_perp == 0.0` on every bar.
  - Real Mark-to-Market Basis P&L: portfolio equity absorbs real `(spot_ret - perp_ret)`
    basis fluctuations plus full 4-leg entry/exit round-trip costs (ZERO synthetic free hedge).
  - Cost-Horizon Entry Floor + Exit Hysteresis: opens a new pair only when smoothed positive
    funding over `hold_horizon_days` strictly exceeds `cost_floor_multiplier * full_4leg_rt_cost_bps`;
    stays held while carry remains above `exit_apr_pct` (preventing ATB H6-style churn).
  - Basis Blow-Up Stop, Deribit `BTC DVOL` Dynamic Throttle (ZERO Yahoo `^VIX`), and hard 50%
    margin storm-exit (`margin_ratio <= 0.50` -> immediate atomic flat on both legs).
  - Full G7 streaming-vs-batch bit-parity (`sync_executed_weights` + `clone_fresh`).
"""

from __future__ import annotations

import bisect
from collections import deque
import logging
import math
from typing import Any, Sequence

from alphaloop.platform.assets import (
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_CRYPTO_SPOT,
    AssetSpec,
    CALENDAR_CRYPTO_24_7,
    COST_BINANCE_PERP,
    COST_BINANCE_SPOT,
    COST_HYPERLIQUID_PERP,
)
from alphaloop.platform.core_engine.interfaces import BarDecision
from alphaloop.platform.data_foundation.models import CanonicalBar, Web3MarketMacroPoint
from alphaloop.platform.strategies.web3.regime_engine import is_crypto_bar

logger = logging.getLogger(__name__)

DN_REGIME_ACTIVE = "DN_CARRY_ACTIVE"
DN_REGIME_DVOL_THROTTLED = "DN_CARRY_DVOL_THROTTLED"
DN_REGIME_SUB_FLOOR = "DN_CARRY_SUB_FLOOR"
DN_REGIME_BASIS_STOP = "DN_BASIS_STOP"
DN_REGIME_MARGIN_STORM_EXIT = "DN_MARGIN_STORM_EXIT"
DN_REGIME_WARMUP = "DN_CARRY_WARMUP"


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


def spot_leg_symbol(coin: str) -> str:
    return f"{coin.strip().upper()}_SPOT"


def perp_leg_symbol(coin: str) -> str:
    return f"{coin.strip().upper()}_PERP"


def parse_pair_leg(sym: str) -> tuple[str, str] | None:
    """Parse `{COIN}_SPOT` / `{COIN}_PERP` into `(coin, leg_kind)` or None."""
    u = sym.strip().upper()
    if u.endswith("_SPOT") and len(u) > 5:
        return u[:-5], "SPOT"
    if u.endswith("_PERP") and len(u) > 5:
        return u[:-5], "PERP"
    return None


def build_dual_leg_carry_universe(
    spot_bars_by_coin: dict[str, Sequence[CanonicalBar]],
    perp_bars_by_coin: dict[str, Sequence[CanonicalBar]],
    *,
    spot_venue: str = "binance_spot",
    perp_venue: str = "hyperliquid",
    spot_cost_model: Any = COST_BINANCE_SPOT,
    perp_cost_model: Any | None = None,
) -> tuple[dict[str, AssetSpec], dict[str, list[CanonicalBar]]]:
    """Build timestamp-aligned `(specs, bars_by_symbol)` for `{COIN}_SPOT` + `{COIN}_PERP` legs.

    Only retains timestamps where BOTH Spot and Perp bars exist for a given coin so dual-leg
    accounting never encounters an unhedged half-pair bar.
    """
    eff_perp_cost = (
        perp_cost_model
        if perp_cost_model is not None
        else (COST_BINANCE_PERP if perp_venue == "binance_um" else COST_HYPERLIQUID_PERP)
    )
    specs: dict[str, AssetSpec] = {}
    bars_out: dict[str, list[CanonicalBar]] = {}

    common_coins = sorted(set(spot_bars_by_coin) & set(perp_bars_by_coin))
    for coin in common_coins:
        s_map = {int(b.ts_event_ms): b for b in spot_bars_by_coin[coin]}
        p_map = {int(b.ts_event_ms): b for b in perp_bars_by_coin[coin]}
        shared_ts = sorted(set(s_map) & set(p_map))
        if not shared_ts:
            continue

        s_key = spot_leg_symbol(coin)
        p_key = perp_leg_symbol(coin)
        specs[s_key] = AssetSpec(
            symbol=s_key,
            venue=spot_venue,
            asset_class=ASSET_CLASS_CRYPTO_SPOT,
            calendar=CALENDAR_CRYPTO_24_7,
            cost_model=spot_cost_model,
            max_leverage_cap=1.0,
        )
        specs[p_key] = AssetSpec(
            symbol=p_key,
            venue=perp_venue,
            asset_class=ASSET_CLASS_CRYPTO_PERP,
            calendar=CALENDAR_CRYPTO_24_7,
            cost_model=eff_perp_cost,
            max_leverage_cap=2.0,
        )
        bars_out[s_key] = [s_map[ts] for ts in shared_ts]
        bars_out[p_key] = [p_map[ts] for ts in shared_ts]

    return specs, bars_out


class _DualLegCoinState:
    """Strictly causal per-coin funding APR and Spot-vs-Perp basis tracker."""

    __slots__ = (
        "tf_h",
        "carry_events",
        "carry_ema_apr",
        "ema_alpha",
        "basis_bps_hist",
        "bar_count",
        "is_active_pair",
    )

    def __init__(self, timeframe_h: float, *, carry_ema_span: int = 6) -> None:
        self.tf_h = max(0.25, float(timeframe_h))
        self.carry_events: deque[tuple[float, float, float]] = deque(
            maxlen=max(4, int(round(72.0 / self.tf_h)))
        )
        self.carry_ema_apr: float | None = None
        self.ema_alpha: float = 2.0 / (max(2, int(carry_ema_span)) + 1.0)
        self.basis_bps_hist: deque[float] = deque(maxlen=60)
        self.bar_count: int = 0
        self.is_active_pair: bool = False

    def update(self, spot_bar: CanonicalBar, perp_bar: CanonicalBar) -> tuple[float, float]:
        """Update with paired closed Spot + Perp bars; return `(smoothed_carry_apr_pct, basis_bps)`."""
        self.bar_count += 1
        tf_h = max(0.01, _finite_or(perp_bar.timeframe_h, self.tf_h))
        ci_raw = _finite_or(perp_bar.carry_interval_h, tf_h)
        ci_h = ci_raw if ci_raw > 0.0 else tf_h
        rate = _finite_or(perp_bar.carry_rate_period, 0.0)
        self.carry_events.append((rate, tf_h, ci_h))

        sum_r = math.fsum(r for r, _, _ in self.carry_events)
        sum_tf = math.fsum(t for _, t, _ in self.carry_events)
        sum_cov = math.fsum(max(t, ci) for r, t, ci in self.carry_events if abs(r) > 1e-15)
        window_apr = (sum_r / max(sum_tf, sum_cov, 0.01)) * 8760.0 * 100.0

        if self.carry_ema_apr is None:
            self.carry_ema_apr = window_apr
        else:
            self.carry_ema_apr = self.ema_alpha * window_apr + (1.0 - self.ema_alpha) * self.carry_ema_apr

        s_close = max(1e-9, _finite_or(spot_bar.close, 0.0))
        p_close = max(1e-9, _finite_or(perp_bar.close, 0.0))
        basis_bps = ((p_close / s_close) - 1.0) * 10_000.0
        self.basis_bps_hist.append(basis_bps)
        return float(self.carry_ema_apr), float(basis_bps)


class Web3DeltaNeutralCarryKernel:
    """W3-DN: Crypto-Native Dual-Leg Spot+Perp Delta-Neutral Funding Carry Engine."""

    asset_domain: str = "web3"

    def __init__(
        self,
        *,
        warmup_bars: int = 12,
        carry_ema_span: int = 6,
        hold_horizon_days: float = 21.0,
        cost_floor_multiplier: float = 3.0,
        full_4leg_rt_cost_bps: float = 36.0,
        exit_apr_pct: float = 2.0,
        max_entry_negative_basis_bps: float = -45.0,
        basis_stop_bps: float = 350.0,
        max_spot_gross: float = 0.90,
        max_pair_leg_weight: float = 0.45,
        dvol_reference: float = 55.0,
        dvol_max_hard_cap: float = 115.0,
        margin_storm_exit_ratio: float = 0.50,
        inertia_buffer: float = 0.15,
        capital_budget_usd: float = 10_000.0,
        dvol_series: Sequence[Web3MarketMacroPoint] | None = None,
        strategy_id: str = "web3_w3_delta_neutral_carry",
    ) -> None:
        self.strategy_id = str(strategy_id)
        self.asset_domain = "web3"
        self.warmup_bars = max(2, int(warmup_bars))
        self.carry_ema_span = max(2, int(carry_ema_span))
        self.hold_horizon_days = _clamp(float(hold_horizon_days), 1.0, 90.0)
        self.cost_floor_multiplier = max(1.0, float(cost_floor_multiplier))
        self.full_4leg_rt_cost_bps = max(0.0, float(full_4leg_rt_cost_bps))
        self.exit_apr_pct = float(exit_apr_pct)
        self.max_entry_negative_basis_bps = float(max_entry_negative_basis_bps)
        self.basis_stop_bps = max(50.0, float(basis_stop_bps))
        self.max_spot_gross = _clamp(float(max_spot_gross), 0.10, 1.0)
        self.max_pair_leg_weight = _clamp(float(max_pair_leg_weight), 0.05, 1.0)
        self.dvol_reference = _clamp(float(dvol_reference), 20.0, 90.0)
        self.dvol_max_hard_cap = _clamp(float(dvol_max_hard_cap), 85.0, 200.0)
        self.margin_storm_exit_ratio = _clamp(float(margin_storm_exit_ratio), 0.10, 0.90)
        self.inertia_buffer = _clamp(float(inertia_buffer), 0.0, 0.50)
        self.capital_budget_usd = max(100.0, float(capital_budget_usd))

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

        self._states: dict[str, _DualLegCoinState] = {}
        self._executed_weights: dict[str, float] = {}
        self._step_count: int = 0

    def reset(self) -> None:
        self._states.clear()
        self._executed_weights.clear()
        self._step_count = 0

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        for sym, w in executed_weights.items():
            self._executed_weights[str(sym)] = _finite_or(w, 0.0)

    def export_state(self) -> dict[str, Any]:
        """Serialize mutable causal kernel state for atomic WAL checkpointing across restarts (RB-01)."""
        states_out: dict[str, Any] = {}
        for coin, st in sorted(self._states.items()):
            states_out[coin] = {
                "tf_h": float(st.tf_h),
                "carry_events": [list(ev) for ev in st.carry_events],
                "carry_ema_apr": st.carry_ema_apr,
                "ema_alpha": float(st.ema_alpha),
                "basis_bps_hist": list(st.basis_bps_hist),
                "bar_count": int(st.bar_count),
                "is_active_pair": bool(st.is_active_pair),
            }
        return {
            "step_count": int(self._step_count),
            "executed_weights": dict(self._executed_weights),
            "states": states_out,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore mutable causal kernel state from a persisted WAL checkpoint (RB-01)."""
        self.reset()
        self._step_count = max(0, int(state.get("step_count", 0)))
        for sym, w in (state.get("executed_weights") or {}).items():
            self._executed_weights[str(sym)] = _finite_or(w, 0.0)
        for coin, st_d in (state.get("states") or {}).items():
            st = _DualLegCoinState(
                float(st_d.get("tf_h", 12.0)),
                carry_ema_span=self.carry_ema_span,
            )
            st.ema_alpha = float(st_d.get("ema_alpha", st.ema_alpha))
            st.carry_ema_apr = (
                float(st_d["carry_ema_apr"]) if st_d.get("carry_ema_apr") is not None else None
            )
            st.bar_count = max(0, int(st_d.get("bar_count", 0)))
            st.is_active_pair = bool(st_d.get("is_active_pair", False))
            for ev in st_d.get("carry_events") or []:
                if isinstance(ev, (list, tuple)) and len(ev) == 3:
                    st.carry_events.append((float(ev[0]), float(ev[1]), float(ev[2])))
            for b in st_d.get("basis_bps_hist") or []:
                st.basis_bps_hist.append(float(b))
            self._states[str(coin)] = st

    def clone_fresh(self) -> Web3DeltaNeutralCarryKernel:
        return Web3DeltaNeutralCarryKernel(
            warmup_bars=self.warmup_bars,
            carry_ema_span=self.carry_ema_span,
            hold_horizon_days=self.hold_horizon_days,
            cost_floor_multiplier=self.cost_floor_multiplier,
            full_4leg_rt_cost_bps=self.full_4leg_rt_cost_bps,
            exit_apr_pct=self.exit_apr_pct,
            max_entry_negative_basis_bps=self.max_entry_negative_basis_bps,
            basis_stop_bps=self.basis_stop_bps,
            max_spot_gross=self.max_spot_gross,
            max_pair_leg_weight=self.max_pair_leg_weight,
            dvol_reference=self.dvol_reference,
            dvol_max_hard_cap=self.dvol_max_hard_cap,
            margin_storm_exit_ratio=self.margin_storm_exit_ratio,
            inertia_buffer=self.inertia_buffer,
            capital_budget_usd=self.capital_budget_usd,
            dvol_series=self._raw_dvol_series,
            strategy_id=self.strategy_id,
        )

    def _resolve_dvol(self, ts_ms: int, ctx: dict[str, Any]) -> float:
        raw = ctx.get("dvol", ctx.get("BTC_DVOL"))
        if raw is not None:
            return _clamp(_finite_or(raw, 50.0), 5.0, 250.0)
        if self._dvol_avails:
            idx = bisect.bisect_right(self._dvol_avails, int(ts_ms)) - 1
            if idx >= 0:
                return _clamp(self._dvol_vals[idx], 5.0, 250.0)
        return 50.0

    def _resolve_margin_ratio(self, coin: str, ctx: dict[str, Any]) -> float:
        raw = ctx.get("margin_ratio", ctx.get("perp_margin_buffer_ratio"))
        if raw is None:
            return 1.0
        if isinstance(raw, dict):
            return _clamp(
                _finite_or(
                    raw.get(coin, raw.get(perp_leg_symbol(coin), raw.get(coin.upper()))),
                    1.0,
                ),
                0.0,
                2.0,
            )
        return _clamp(_finite_or(raw, 1.0), 0.0, 2.0)

    def _resolve_4leg_rt_cost_bps(self, coin: str, ctx: dict[str, Any]) -> float:
        raw = ctx.get("full_4leg_rt_cost_bps", ctx.get("rt_cost_bps"))
        if raw is None:
            return self.full_4leg_rt_cost_bps
        if isinstance(raw, dict):
            return max(0.0, _finite_or(raw.get(coin, raw.get(coin.upper())), self.full_4leg_rt_cost_bps))
        return max(0.0, _finite_or(raw, self.full_4leg_rt_cost_bps))

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        """Consume paired `{COIN}_SPOT` and `{COIN}_PERP` bars and emit atomic delta-neutral decisions."""
        self._step_count += 1
        ctx = context if isinstance(context, dict) else {}
        dvol = self._resolve_dvol(int(ts_ms), ctx)

        spot_by_coin: dict[str, tuple[str, CanonicalBar]] = {}
        perp_by_coin: dict[str, tuple[str, CanonicalBar]] = {}
        unpaired_or_non_crypto: list[str] = []

        for sym in sorted(bars_at_ts):
            bar = bars_at_ts[sym]
            parsed = parse_pair_leg(sym)
            if parsed is None:
                unpaired_or_non_crypto.append(sym)
                continue
            coin, leg_kind = parsed
            if not is_crypto_bar(coin, bar):
                unpaired_or_non_crypto.append(sym)
                continue
            if leg_kind == "SPOT":
                spot_by_coin[coin] = (sym, bar)
            else:
                perp_by_coin[coin] = (sym, bar)

        decisions: dict[str, BarDecision] = {}
        for sym in unpaired_or_non_crypto:
            decisions[sym] = BarDecision(
                symbol=sym,
                ts_event_ms=int(ts_ms),
                target_weight=0.0,
                raw_signal=0.0,
                regime_label=DN_REGIME_SUB_FLOOR,
                stop_price=None,
                metadata={"ignored_unpaired_or_non_crypto": True},
            )

        paired_coins = sorted(set(spot_by_coin) & set(perp_by_coin))
        for coin in sorted((set(spot_by_coin) ^ set(perp_by_coin))):
            sym_only = spot_by_coin[coin][0] if coin in spot_by_coin else perp_by_coin[coin][0]
            decisions[sym_only] = BarDecision(
                symbol=sym_only,
                ts_event_ms=int(ts_ms),
                target_weight=0.0,
                raw_signal=0.0,
                regime_label=DN_REGIME_SUB_FLOOR,
                stop_price=None,
                metadata={"pair_coin": coin, "unpaired_leg_suppressed": True},
            )

        # DVOL dynamic leverage multiplier (smoothly throttles above dvol_reference, zero at hard cap)
        if dvol >= self.dvol_max_hard_cap:
            dvol_mult = 0.0
        elif dvol <= self.dvol_reference:
            dvol_mult = 1.0
        else:
            frac = (dvol - self.dvol_reference) / max(1.0, self.dvol_max_hard_cap - self.dvol_reference)
            dvol_mult = max(0.15, 1.0 - 0.85 * frac)

        # First pass: update state and evaluate entry/hold/exit eligibility per coin pair
        pair_scores: dict[str, float] = {}
        pair_diag: dict[str, dict[str, Any]] = {}
        active_coins: list[str] = []

        for coin in paired_coins:
            s_sym, s_bar = spot_by_coin[coin]
            p_sym, p_bar = perp_by_coin[coin]
            st = self._states.get(coin)
            if st is None:
                st = _DualLegCoinState(
                    _finite_or(p_bar.timeframe_h, 12.0),
                    carry_ema_span=self.carry_ema_span,
                )
                self._states[coin] = st

            carry_apr, basis_bps = st.update(s_bar, p_bar)
            rt_cost_bps = self._resolve_4leg_rt_cost_bps(coin, ctx)
            margin_ratio = self._resolve_margin_ratio(coin, ctx)
            expected_carry_bps = (carry_apr * 100.0) * (self.hold_horizon_days / 365.0)
            min_entry_bps = self.cost_floor_multiplier * rt_cost_bps

            # Check actual synced executed position for hysteresis & G7 parity
            prev_s_w = abs(self._executed_weights.get(s_sym, 0.0))
            prev_p_w = abs(self._executed_weights.get(p_sym, 0.0))
            currently_held = min(prev_s_w, prev_p_w) > 1e-6

            regime = DN_REGIME_SUB_FLOOR
            eligible = False

            if st.bar_count < self.warmup_bars:
                regime = DN_REGIME_WARMUP
                eligible = False
            elif margin_ratio <= self.margin_storm_exit_ratio:
                regime = DN_REGIME_MARGIN_STORM_EXIT
                eligible = False
            elif dvol_mult <= 0.0:
                regime = DN_REGIME_DVOL_THROTTLED
                eligible = False
            elif abs(basis_bps) >= self.basis_stop_bps:
                regime = DN_REGIME_BASIS_STOP
                eligible = False
            elif currently_held:
                # Hysteresis hold: remain delta-neutral while carry stays above exit floor
                if carry_apr >= self.exit_apr_pct:
                    eligible = True
                    regime = DN_REGIME_DVOL_THROTTLED if dvol_mult < 0.85 else DN_REGIME_ACTIVE
                else:
                    eligible = False
                    regime = DN_REGIME_SUB_FLOOR
            else:
                # Strict cost-horizon entry gate (3x full 4-leg RT cost) + basis guard
                if expected_carry_bps >= min_entry_bps and basis_bps >= self.max_entry_negative_basis_bps:
                    eligible = True
                    regime = DN_REGIME_DVOL_THROTTLED if dvol_mult < 0.85 else DN_REGIME_ACTIVE
                else:
                    eligible = False
                    regime = DN_REGIME_SUB_FLOOR

            st.is_active_pair = eligible
            score = max(0.0, carry_apr) if eligible else 0.0
            pair_scores[coin] = score
            if eligible:
                active_coins.append(coin)
            pair_diag[coin] = {
                "carry_apr_pct": round(carry_apr, 4),
                "expected_carry_bps": round(expected_carry_bps, 2),
                "min_entry_bps": round(min_entry_bps, 2),
                "full_4leg_rt_cost_bps": round(rt_cost_bps, 2),
                "basis_bps": round(basis_bps, 2),
                "dvol": round(dvol, 2),
                "dvol_mult": round(dvol_mult, 4),
                "margin_ratio": round(margin_ratio, 4),
                "regime": regime,
                "currently_held": currently_held,
            }

        # Second pass: allocate spot/perp atomic leg weights across active pairs
        target_leg_w: dict[str, float] = {c: 0.0 for c in paired_coins}
        if active_coins:
            eff_spot_gross = self.max_spot_gross * dvol_mult
            # 60% equal weight + 40% carry-tilted across active pairs for stability
            sum_sc = math.fsum(max(1.0, pair_scores[c]) for c in active_coins)
            raw_alloc: dict[str, float] = {}
            for c in active_coins:
                eq_share = 1.0 / len(active_coins)
                sc_share = max(1.0, pair_scores[c]) / max(sum_sc, 1e-9)
                w_c = eff_spot_gross * (0.60 * eq_share + 0.40 * sc_share)
                raw_alloc[c] = min(self.max_pair_leg_weight, w_c)
            # Re-check total spot gross ceiling
            tot_alloc = math.fsum(raw_alloc.values())
            if tot_alloc > eff_spot_gross + 1e-12:
                sc = eff_spot_gross / tot_alloc
                for c in active_coins:
                    raw_alloc[c] *= sc
            for c in active_coins:
                target_leg_w[c] = raw_alloc[c]

        # Third pass: apply symmetric Carver inertia band to BOTH legs atomically (preserving w_spot + w_perp == 0)
        for coin in paired_coins:
            s_sym, _ = spot_by_coin[coin]
            p_sym, _ = perp_by_coin[coin]
            diag = pair_diag[coin]
            des_leg = target_leg_w[coin]
            prev_s = self._executed_weights.get(s_sym, 0.0)
            prev_p = self._executed_weights.get(p_sym, 0.0)
            # Only apply inertia if prior execution is genuinely symmetric and neither leg is exiting
            prev_sym_leg = min(max(0.0, prev_s), max(0.0, -prev_p))
            is_prev_sym = abs(prev_s + prev_p) <= 1e-6 and prev_sym_leg > 1e-6
            inertia_held = False
            chosen_leg = des_leg
            if (
                des_leg > 1e-6
                and is_prev_sym
                and self.inertia_buffer > 0.0
                and diag["regime"] not in (DN_REGIME_MARGIN_STORM_EXIT, DN_REGIME_BASIS_STOP)
            ):
                band = max(0.02, self.inertia_buffer * max(des_leg, prev_sym_leg))
                if abs(des_leg - prev_sym_leg) <= band:
                    chosen_leg = prev_sym_leg
                    inertia_held = True

            # Identical raw_signal magnitude for both legs guarantees symmetric allocator ranking
            sig = round(pair_scores[coin], 6)
            base_meta = {
                "pair_coin": coin,
                "delta_neutral_pair": True,
                "inertia_held": inertia_held,
                "pair_capital_usd": round(self.capital_budget_usd * chosen_leg, 2),
                **{k: v for k, v in diag.items() if k != "regime"},
            }
            decisions[s_sym] = BarDecision(
                symbol=s_sym,
                ts_event_ms=int(ts_ms),
                target_weight=float(chosen_leg),
                raw_signal=float(sig),
                regime_label=str(diag["regime"]),
                stop_price=None,
                metadata={"leg": "SPOT", **base_meta},
            )
            decisions[p_sym] = BarDecision(
                symbol=p_sym,
                ts_event_ms=int(ts_ms),
                target_weight=float(-chosen_leg),
                raw_signal=float(-sig),
                regime_label=str(diag["regime"]),
                stop_price=None,
                metadata={"leg": "PERP", **base_meta},
            )

        return decisions
