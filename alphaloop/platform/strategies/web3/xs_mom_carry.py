"""W1: Web3 Cross-Sectional Momentum + Funding Carry/Crowding Tilt + OI Confirmation (F-13).

100% crypto-native multi-coin L/S strategy implementing `AssetNativeStrategy`:
  - Cross-sectional multi-horizon volatility-adjusted momentum ranking (`top_k` long, `bottom_k` short).
  - Funding crowding penalty / carry tilt (penalizes extreme crowded-long positive funding, favors
    un-crowded / negative-funding momentum using exact D11 `carry_interval_h` normalization).
  - Open Interest (OI) directional confirmation (`oi_change_7d` / `Web3DerivativesMetric`).
  - Point-in-Time listing guard (`pit_listing_dates`, defaulting `HYPE` to `2024-12-05T00:00:00Z`
    with strict zero weight and zero history accumulation prior to listing).
  - Crypto-native regime-gated net exposure (`RISK_OFF_BEAR` -> market-neutral net ~0;
    `EXPANSION`/`ALT_SEASON` -> constructive net-long tilt; ZERO Yahoo `^VIX` or TradFi coupling).
  - Carver inertia buffer driven strictly by `sync_executed_weights` for full G7 bit-parity.
"""

from __future__ import annotations

import bisect
from collections import deque
from datetime import datetime, timezone
import logging
import math
from typing import Any, Sequence

from alphaloop.platform.core_engine.interfaces import BarDecision
from alphaloop.platform.data_foundation.models import (
    CanonicalBar,
    DAY_MS,
    Web3DerivativesMetric,
)
from alphaloop.platform.strategies.web3.regime_engine import (
    WEB3_REGIME_ALT_SEASON,
    WEB3_REGIME_CROWDED_STRESS,
    WEB3_REGIME_EXPANSION,
    WEB3_REGIME_NEUTRAL,
    WEB3_REGIME_RISK_OFF_BEAR,
    Web3MarketRegimeEngine,
    is_crypto_bar,
)

logger = logging.getLogger(__name__)

# Official Hyperliquid HYPE perpetual listing timestamp: 2024-12-05T00:00:00Z
HYPE_PIT_LISTING_MS: int = int(
    round(datetime(2024, 12, 5, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000.0)
)
DEFAULT_WEB3_PIT_LISTING_DATES: dict[str, int] = {
    "HYPE": HYPE_PIT_LISTING_MS,
}


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
    var = max(0.0, ss / n)
    return mu, math.sqrt(var)


def _normalize_regime_label(raw_label: Any, fallback: str) -> str:
    if not isinstance(raw_label, str) or not raw_label.strip():
        return fallback
    lbl = raw_label.strip().upper()
    if "BEAR" in lbl or "RISK_OFF" in lbl:
        return WEB3_REGIME_RISK_OFF_BEAR
    if "STRESS" in lbl or "CROWD" in lbl:
        return WEB3_REGIME_CROWDED_STRESS
    if "ALT" in lbl:
        return WEB3_REGIME_ALT_SEASON
    if "EXPAN" in lbl or "BULL" in lbl or "TREND" in lbl:
        return WEB3_REGIME_EXPANSION
    if "NEUT" in lbl or "CHOP" in lbl or "RANGE" in lbl:
        return WEB3_REGIME_NEUTRAL
    return lbl


class _W1SymbolState:
    """Strictly causal per-symbol rolling return, volatility, carry, and OI buffer."""

    __slots__ = ("closes", "log_rets", "carry_events", "oi_hist", "bar_count")

    def __init__(self) -> None:
        self.closes: deque[float] = deque(maxlen=160)
        self.log_rets: deque[float] = deque(maxlen=80)
        self.carry_events: deque[tuple[float, float, float]] = deque(maxlen=16)
        self.oi_hist: deque[tuple[int, float]] = deque(maxlen=120)
        self.bar_count: int = 0

    def update_bar(self, bar: CanonicalBar) -> None:
        c = float(bar.close)
        if self.closes:
            prev_c = self.closes[-1]
            if prev_c > 0.0 and c > 0.0:
                self.log_rets.append(math.log(c / prev_c))
        self.closes.append(c)
        tf_h = max(0.01, _finite_or(bar.timeframe_h, 12.0))
        ci_raw = _finite_or(bar.carry_interval_h, tf_h)
        ci_h = ci_raw if ci_raw > 0.0 else tf_h
        rate = _finite_or(bar.carry_rate_period, 0.0)
        self.carry_events.append((rate, tf_h, ci_h))
        self.bar_count += 1

    def update_oi(self, ts_ms: int, oi_usd: float) -> None:
        if math.isfinite(oi_usd) and oi_usd > 0.0:
            self.oi_hist.append((int(ts_ms), float(oi_usd)))

    def compute_vol_adj_momentum(self) -> tuple[float, float]:
        """Return `(risk_adj_mom, realized_bar_vol)` using strictly past closes."""
        n = len(self.closes)
        if n < 6:
            return 0.0, 0.02
        c = self.closes[-1]
        if len(self.log_rets) >= 5:
            recent_rets = list(self.log_rets)[-30:]
            _, sd = _mean_std(recent_rets)
            bar_vol = max(sd, 0.0025)
        else:
            bar_vol = 0.02

        horizons = ((10, 0.20), (20, 0.35), (40, 0.30), (60, 0.15))
        weighted_sum = 0.0
        weight_total = 0.0
        for h, w in horizons:
            if n > h:
                ref = self.closes[-1 - h]
                if ref > 0.0 and c > 0.0:
                    log_r = math.log(c / ref)
                    scaled = _clamp(log_r / (bar_vol * math.sqrt(float(h))), -4.0, 4.0)
                    weighted_sum += w * scaled
                    weight_total += w
        if weight_total <= 0.0:
            ref = self.closes[0]
            h_eff = max(1, n - 1)
            log_r = math.log(c / ref) if ref > 0.0 else 0.0
            mom = _clamp(log_r / (bar_vol * math.sqrt(float(h_eff))), -4.0, 4.0)
        else:
            mom = weighted_sum / weight_total
        return mom, bar_vol

    def compute_funding_apr_pct(self, lookback_bars: int = 6) -> float:
        """Return exact D11 rolling annualized funding APR (%) over recent bars."""
        if not self.carry_events:
            return 0.0
        tail = list(self.carry_events)[-max(1, lookback_bars) :]
        sum_r = math.fsum(r for r, _, _ in tail)
        sum_tf = math.fsum(t for _, t, _ in tail)
        sum_cov = math.fsum(max(t, ci) for r, t, ci in tail if abs(r) > 1e-15)
        denom_h = max(sum_tf, sum_cov, 0.01)
        return (sum_r / denom_h) * 8760.0 * 100.0

    def compute_stream_oi_change_7d(self, ts_ms: int) -> float | None:
        if len(self.oi_hist) < 2:
            return None
        cur_oi = self.oi_hist[-1][1]
        cutoff = int(ts_ms) - 7 * DAY_MS
        ref_oi = self.oi_hist[0][1]
        for h_ts, h_v in self.oi_hist:
            if h_ts <= cutoff:
                ref_oi = h_v
            else:
                break
        if ref_oi <= 0.0:
            return None
        return (cur_oi / ref_oi) - 1.0


class Web3XSMomCarryKernel:
    """W1: Crypto-Native Cross-Sectional Momentum + Funding Crowding Tilt + OI Confirmation."""

    asset_domain: str = "web3"

    def __init__(
        self,
        *,
        top_k: int = 2,
        bottom_k: int = 2,
        warmup_bars: int = 40,
        min_symbol_bars: int = 20,
        inertia_buffer: float = 0.10,
        gross_target: float = 1.0,
        funding_penalty_weight: float = 0.35,
        oi_confirm_weight: float = 0.25,
        score_ema_alpha: float = 0.35,
        pit_listing_dates: dict[str, int] | None = None,
        metrics_by_symbol: dict[str, Sequence[Web3DerivativesMetric]] | None = None,
        regime_engine: Web3MarketRegimeEngine | None = None,
        strategy_id: str = "web3_w1_xs_mom_carry",
    ) -> None:
        self.strategy_id = str(strategy_id)
        self.asset_domain = "web3"
        self.top_k = max(1, int(top_k))
        self.bottom_k = max(0, int(bottom_k))
        self.warmup_bars = max(5, int(warmup_bars))
        self.min_symbol_bars = max(5, min(int(min_symbol_bars), self.warmup_bars))
        self.inertia_buffer = _clamp(float(inertia_buffer), 0.0, 0.50)
        self.gross_target = _clamp(float(gross_target), 0.10, 2.0)
        self.funding_penalty_weight = _clamp(float(funding_penalty_weight), 0.0, 2.0)
        self.oi_confirm_weight = _clamp(float(oi_confirm_weight), 0.0, 2.0)
        self.score_ema_alpha = _clamp(float(score_ema_alpha), 0.05, 1.0)

        merged_listings = dict(DEFAULT_WEB3_PIT_LISTING_DATES)
        if pit_listing_dates:
            for k, v in pit_listing_dates.items():
                merged_listings[str(k).upper()] = int(v)
        self.pit_listing_dates: dict[str, int] = merged_listings

        # Optional pre-loaded PIT OI metrics indexed strictly by ts_avail_ms
        self._pit_oi: dict[str, tuple[list[int], list[float]]] = {}
        self._raw_metrics_ref = metrics_by_symbol
        if metrics_by_symbol:
            for sym in sorted(metrics_by_symbol):
                ordered = sorted(
                    metrics_by_symbol[sym],
                    key=lambda m: (int(m.ts_avail_ms), int(m.ts_event_ms)),
                )
                avails: list[int] = []
                vals: list[float] = []
                for m in ordered:
                    oi_v = _finite_or(getattr(m, "open_interest_usd", 0.0), 0.0)
                    if oi_v > 0.0:
                        avails.append(int(m.ts_avail_ms))
                        vals.append(oi_v)
                if avails:
                    self._pit_oi[sym.upper()] = (avails, vals)

        self._external_regime_template = regime_engine
        self._regime_engine: Web3MarketRegimeEngine = (
            regime_engine.clone_fresh() if regime_engine is not None else Web3MarketRegimeEngine()
        )

        # Mutable bar-by-bar state
        self._states: dict[str, _W1SymbolState] = {}
        self._score_emas: dict[str, float] = {}
        self._executed_weights: dict[str, float] = {}
        self._step_count: int = 0

    def reset(self) -> None:
        """Reset all mutable internal state to clean initial conditions."""
        self._states.clear()
        self._score_emas.clear()
        self._executed_weights.clear()
        self._step_count = 0
        self._regime_engine.reset()

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        """Sync actual post-allocator/post-fill weights for Carver inertia & G7 parity."""
        for sym, w in executed_weights.items():
            fw = _finite_or(w, 0.0)
            self._executed_weights[str(sym)] = fw

    def clone_fresh(self) -> Web3XSMomCarryKernel:
        """Return a brand-new freshly reset instance with identical parameters."""
        return Web3XSMomCarryKernel(
            top_k=self.top_k,
            bottom_k=self.bottom_k,
            warmup_bars=self.warmup_bars,
            min_symbol_bars=self.min_symbol_bars,
            inertia_buffer=self.inertia_buffer,
            gross_target=self.gross_target,
            funding_penalty_weight=self.funding_penalty_weight,
            oi_confirm_weight=self.oi_confirm_weight,
            score_ema_alpha=self.score_ema_alpha,
            pit_listing_dates=dict(self.pit_listing_dates),
            metrics_by_symbol=self._raw_metrics_ref,
            regime_engine=self._external_regime_template,
            strategy_id=self.strategy_id,
        )

    def _lookup_pit_oi_change_7d(self, sym_u: str, ts_ms: int) -> float | None:
        entry = self._pit_oi.get(sym_u)
        if not entry:
            return None
        avails, vals = entry
        i_now = bisect.bisect_right(avails, int(ts_ms)) - 1
        if i_now < 1:
            return None
        cutoff = int(ts_ms) - 7 * DAY_MS
        i_prev = bisect.bisect_right(avails, cutoff) - 1
        if i_prev < 0:
            i_prev = 0
        v_prev = vals[i_prev]
        if v_prev <= 0.0:
            return None
        return (vals[i_now] / v_prev) - 1.0

    def _resolve_symbol_oi_change(
        self,
        sym: str,
        sym_u: str,
        ts_ms: int,
        state: _W1SymbolState,
        ctx: dict[str, Any],
    ) -> float:
        # Update streaming OI history if Web3DerivativesMetric objects are supplied in context
        raw_metrics = ctx.get("web3_metrics", ctx.get("derivatives_metrics"))
        if isinstance(raw_metrics, dict):
            m_obj = raw_metrics.get(sym, raw_metrics.get(sym_u))
            if m_obj is not None:
                if isinstance(m_obj, Sequence) and not isinstance(m_obj, (str, bytes)):
                    for item in m_obj:
                        item_av = int(getattr(item, "ts_avail_ms", getattr(item, "ts_event_ms", ts_ms)))
                        if item_av <= int(ts_ms):
                            oi_v = _finite_or(getattr(item, "open_interest_usd", 0.0), 0.0)
                            if oi_v > 0.0:
                                state.update_oi(int(getattr(item, "ts_event_ms", ts_ms)), oi_v)
                else:
                    oi_v = _finite_or(getattr(m_obj, "open_interest_usd", 0.0), 0.0)
                    if oi_v > 0.0:
                        state.update_oi(int(getattr(m_obj, "ts_event_ms", ts_ms)), oi_v)

        # Priority 1: explicit per-symbol or scalar oi_change_7d in context
        if "oi_change_7d" in ctx and ctx["oi_change_7d"] is not None:
            oi_ctx = ctx["oi_change_7d"]
            if isinstance(oi_ctx, dict):
                if sym in oi_ctx:
                    return _finite_or(oi_ctx[sym], 0.0)
                if sym_u in oi_ctx:
                    return _finite_or(oi_ctx[sym_u], 0.0)
            else:
                return _finite_or(oi_ctx, 0.0)

        # Priority 2: streaming OI metric history
        stream_chg = state.compute_stream_oi_change_7d(ts_ms)
        if stream_chg is not None:
            return stream_chg

        # Priority 3: pre-loaded PIT Web3DerivativesMetric index
        pit_chg = self._lookup_pit_oi_change_7d(sym_u, ts_ms)
        if pit_chg is not None:
            return pit_chg

        return 0.0

    def _regime_leg_budgets(
        self,
        regime_label: str,
        exposure_scalar: float,
    ) -> tuple[float, float]:
        """Return `(long_gross_budget, short_gross_budget)` gated by crypto-native regime."""
        if regime_label == WEB3_REGIME_RISK_OFF_BEAR:
            # Market-neutral L/S book in bear/stress: strictly net ~ 0.0, active gross > 0.05
            eff_gross = max(0.40, min(0.70, exposure_scalar * 2.0)) * self.gross_target
            return 0.50 * eff_gross, 0.50 * eff_gross
        if regime_label == WEB3_REGIME_CROWDED_STRESS:
            eff_gross = max(0.45, min(0.75, exposure_scalar * 1.4)) * self.gross_target
            return 0.52 * eff_gross, 0.48 * eff_gross
        if regime_label == WEB3_REGIME_ALT_SEASON:
            eff_gross = _clamp(exposure_scalar, 0.85, 1.15) * self.gross_target
            return 0.72 * eff_gross, 0.28 * eff_gross
        if regime_label == WEB3_REGIME_EXPANSION:
            eff_gross = _clamp(exposure_scalar, 0.75, 1.05) * self.gross_target
            return 0.65 * eff_gross, 0.35 * eff_gross
        # NEUTRAL
        eff_gross = _clamp(exposure_scalar, 0.50, 0.85) * self.gross_target
        return 0.55 * eff_gross, 0.45 * eff_gross

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        """Consume closed crypto bars at `ts_ms` and emit strictly causal W1 L/S decisions."""
        self._step_count += 1
        ctx = context if isinstance(context, dict) else {}

        # 1. Partition incoming bars strictly into eligible crypto vs pre-listing / non-crypto
        crypto_bars: dict[str, CanonicalBar] = {}
        pre_listing_syms: set[str] = set()
        non_crypto_syms: set[str] = set()

        for sym in sorted(bars_at_ts):
            bar = bars_at_ts[sym]
            if not is_crypto_bar(sym, bar):
                non_crypto_syms.add(sym)
                continue
            sym_u = sym.upper()
            listing_ms = self.pit_listing_dates.get(sym_u)
            if listing_ms is not None and int(ts_ms) < listing_ms:
                pre_listing_syms.add(sym)
                continue
            crypto_bars[sym] = bar

        # 2. Update embedded crypto-native regime engine (strictly ignores Yahoo/TradFi keys)
        reg_snap = self._regime_engine.update(int(ts_ms), crypto_bars, ctx)
        regime_label = _normalize_regime_label(ctx.get("regime_label"), reg_snap.regime_label)
        if "exposure_scalar" in ctx and ctx["exposure_scalar"] is not None:
            exposure_scalar = _clamp(_finite_or(ctx["exposure_scalar"], reg_snap.exposure_scalar), 0.10, 1.30)
        elif "net_exposure_scalar" in ctx and ctx["net_exposure_scalar"] is not None:
            exposure_scalar = _clamp(_finite_or(ctx["net_exposure_scalar"], reg_snap.exposure_scalar), 0.10, 1.30)
        else:
            exposure_scalar = reg_snap.exposure_scalar

        # 3. Update per-symbol causal state ONLY for post-listing crypto bars
        for sym in sorted(crypto_bars):
            state = self._states.get(sym)
            if state is None:
                state = _W1SymbolState()
                self._states[sym] = state
            state.update_bar(crypto_bars[sym])

        decisions: dict[str, BarDecision] = {}

        # Always emit zero for pre-listing and non-crypto symbols
        for sym in sorted(pre_listing_syms):
            sym_u = sym.upper()
            decisions[sym] = BarDecision(
                symbol=sym,
                ts_event_ms=int(ts_ms),
                target_weight=0.0,
                raw_signal=0.0,
                regime_label=regime_label,
                stop_price=None,
                metadata={
                    "pit_pre_listing": True,
                    "listing_ts_ms": int(self.pit_listing_dates[sym_u]),
                },
            )
        for sym in sorted(non_crypto_syms):
            decisions[sym] = BarDecision(
                symbol=sym,
                ts_event_ms=int(ts_ms),
                target_weight=0.0,
                raw_signal=0.0,
                regime_label=regime_label,
                stop_price=None,
                metadata={"ignored_non_crypto": True},
            )

        # 4. Warmup gate
        eligible_syms = [
            sym
            for sym in sorted(crypto_bars)
            if self._states[sym].bar_count >= self.min_symbol_bars
        ]
        if self._step_count < self.warmup_bars or not eligible_syms:
            for sym in sorted(crypto_bars):
                decisions[sym] = BarDecision(
                    symbol=sym,
                    ts_event_ms=int(ts_ms),
                    target_weight=0.0,
                    raw_signal=0.0,
                    regime_label=regime_label,
                    stop_price=None,
                    metadata={
                        "warmup": True,
                        "step_count": self._step_count,
                        "symbol_bars": self._states[sym].bar_count,
                    },
                )
            return decisions

        # 5. Compute cross-sectional momentum, funding crowding penalty, and OI confirmation
        raw_moms: dict[str, float] = {}
        bar_vols: dict[str, float] = {}
        fund_aprs: dict[str, float] = {}
        oi_chgs: dict[str, float] = {}

        for sym in eligible_syms:
            st = self._states[sym]
            m_val, b_vol = st.compute_vol_adj_momentum()
            raw_moms[sym] = m_val
            bar_vols[sym] = b_vol
            fund_aprs[sym] = st.compute_funding_apr_pct(lookback_bars=6)
            oi_chgs[sym] = self._resolve_symbol_oi_change(sym, sym.upper(), int(ts_ms), st, ctx)

        mom_vals = [raw_moms[s] for s in eligible_syms]
        mom_mu, mom_sd = _mean_std(mom_vals)
        fund_vals = [fund_aprs[s] for s in eligible_syms]
        fund_mu, fund_sd = _mean_std(fund_vals)

        scores: dict[str, float] = {}
        mom_z_map: dict[str, float] = {}
        fund_score_map: dict[str, float] = {}
        oi_score_map: dict[str, float] = {}

        for sym in eligible_syms:
            if mom_sd > 1e-9:
                m_z = _clamp((raw_moms[sym] - mom_mu) / mom_sd, -3.0, 3.0)
            else:
                m_z = 0.0
            # Blend cross-sectional z-score with absolute momentum sign stability
            mom_comp = 0.75 * m_z + 0.25 * math.tanh(raw_moms[sym] / 1.5)

            # Funding crowding penalty: high positive funding penalized, negative funding rewarded
            f_level = -math.tanh(fund_aprs[sym] / 50.0)
            f_xs = -_clamp((fund_aprs[sym] - fund_mu) / fund_sd, -2.5, 2.5) / 2.5 if fund_sd > 1e-6 else 0.0
            fund_comp = 0.65 * f_level + 0.35 * f_xs

            # OI directional confirmation
            oi_comp = math.tanh(oi_chgs[sym] / 0.25)

            raw_sig = (
                mom_comp
                + self.funding_penalty_weight * fund_comp
                + self.oi_confirm_weight * oi_comp
            )
            prev_ema = self._score_emas.get(sym)
            if prev_ema is None:
                sm_sig = raw_sig
            else:
                sm_sig = self.score_ema_alpha * raw_sig + (1.0 - self.score_ema_alpha) * prev_ema
            self._score_emas[sym] = sm_sig
            scores[sym] = sm_sig
            mom_z_map[sym] = m_z
            fund_score_map[sym] = fund_comp
            oi_score_map[sym] = oi_comp

        # 6. Rank top-K long / bottom-K short cleanly without overlap
        n_elig = len(eligible_syms)
        ranked_desc = sorted(eligible_syms, key=lambda s: (-scores[s], s))
        if n_elig == 1:
            eff_top = 1 if scores[ranked_desc[0]] > 0.0 else 0
            eff_bot = 0
        else:
            eff_top = min(self.top_k, max(1, n_elig // 2))
            eff_bot = min(self.bottom_k, max(0, n_elig - eff_top))

        long_syms = ranked_desc[:eff_top]
        short_syms = ranked_desc[n_elig - eff_bot :] if eff_bot > 0 else []

        long_budget, short_budget = self._regime_leg_budgets(regime_label, exposure_scalar)
        desired_weights: dict[str, float] = {sym: 0.0 for sym in crypto_bars}

        if long_syms:
            inv_vols_l = {s: 1.0 / max(bar_vols[s], 0.005) for s in long_syms}
            sum_iv_l = math.fsum(inv_vols_l.values())
            for s in long_syms:
                # 50% equal-weight + 50% inverse-vol within long leg for stability
                w_share = 0.50 * (1.0 / len(long_syms)) + 0.50 * (inv_vols_l[s] / max(sum_iv_l, 1e-12))
                desired_weights[s] = long_budget * w_share

        if short_syms:
            inv_vols_s = {s: 1.0 / max(bar_vols[s], 0.005) for s in short_syms}
            sum_iv_s = math.fsum(inv_vols_s.values())
            for s in short_syms:
                w_share = 0.50 * (1.0 / len(short_syms)) + 0.50 * (inv_vols_s[s] / max(sum_iv_s, 1e-12))
                desired_weights[s] = -short_budget * w_share

        # 7. Apply Carver inertia buffer using synced post-allocator weights
        final_weights: dict[str, float] = {}
        inertia_held_map: dict[str, bool] = {}

        for sym in sorted(crypto_bars):
            des_w = desired_weights[sym]
            prev_w = self._executed_weights.get(sym, 0.0)
            # Never hold through exits (des_w == 0) or sign flips
            if abs(prev_w) > 1e-9 and prev_w * des_w > 0.0 and self.inertia_buffer > 0.0:
                band = max(0.012, self.inertia_buffer * max(abs(des_w), abs(prev_w)))
                if abs(des_w - prev_w) <= band:
                    final_weights[sym] = prev_w
                    inertia_held_map[sym] = True
                    continue
            final_weights[sym] = des_w
            inertia_held_map[sym] = False

        # Guarantee market-neutrality ceiling in RISK_OFF_BEAR even after inertia
        if regime_label == WEB3_REGIME_RISK_OFF_BEAR:
            gross_now = math.fsum(abs(w) for w in final_weights.values())
            net_now = math.fsum(final_weights.values())
            if gross_now > 1e-9 and abs(net_now) > 0.10 * gross_now:
                for sym in sorted(crypto_bars):
                    final_weights[sym] = desired_weights[sym]
                    inertia_held_map[sym] = False

        # 8. Emit canonical BarDecision objects
        for sym in sorted(crypto_bars):
            if sym in eligible_syms:
                decisions[sym] = BarDecision(
                    symbol=sym,
                    ts_event_ms=int(ts_ms),
                    target_weight=float(final_weights[sym]),
                    raw_signal=float(scores[sym]),
                    regime_label=regime_label,
                    stop_price=None,
                    metadata={
                        "mom_z": round(mom_z_map[sym], 4),
                        "funding_apr_pct": round(fund_aprs[sym], 4),
                        "funding_score": round(fund_score_map[sym], 4),
                        "oi_change_7d": round(oi_chgs[sym], 4),
                        "oi_score": round(oi_score_map[sym], 4),
                        "desired_weight": round(desired_weights[sym], 6),
                        "inertia_held": bool(inertia_held_map[sym]),
                        "exposure_scalar": round(exposure_scalar, 4),
                    },
                )
            else:
                decisions[sym] = BarDecision(
                    symbol=sym,
                    ts_event_ms=int(ts_ms),
                    target_weight=0.0,
                    raw_signal=0.0,
                    regime_label=regime_label,
                    stop_price=None,
                    metadata={
                        "warmup": True,
                        "symbol_bars": self._states[sym].bar_count,
                    },
                )

        return decisions
