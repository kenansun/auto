"""A2: CN A-Share Native Pure Stock-Bond Macro-ERP, CGB Yield, Margin-Flow & FX Rotation Kernel.

Physically isolated from Web3 and Gold strategy modules (zero cross-domain imports).
Explicitly FORBIDS domestic Gold ETF (`518880.SS`) so every basis point of return comes
strictly from pure Chinese domestic equities, high-dividend/low-vol equities, and China
Government Bond (CGB) ETFs under authentic CN A-Share trading rules:
  - Strict Long-Only (`target_weight >= 0.0`)
  - Exchange-Local CST (`Asia/Shanghai`) `T+1` sell lock
  - Board-Aware Daily Price Limits (`±10%` Mainboard, `±20%` ChiNext/STAR)
  - Sell-Only Stamp Duty (`5.0 bps` via `COST_CN_A_SHARE`)
  - Zero lookahead: `CNMacroRegimeTimeline` resolves ChinaBond CGB yields (`18:00 CST` PIT),
    SSE/SZSE margin financing balances (`T+1 08:30 CST` PIT), and USD/CNY FX (`T+1 06:00 CST` PIT)
    via stateless `bisect_right(..., ts_avail_ms)` (D6 non-monotonic reuse safe)
  - Full `G7` Bit-Exact Batch/Streaming Parity (`reset()`, `clone_fresh()`, `sync_executed_weights()`)
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from alphaloop.platform.assets import COST_CN_A_SHARE
from alphaloop.platform.core_engine.interfaces import BarDecision
from alphaloop.platform.data_foundation.models import CNAshareMacroPoint, CanonicalBar
from alphaloop.platform.strategies.ashare_contract import (
    AShareStrategyInterface,
    cn_exchange_trade_date,
)

_FORBIDDEN_GOLD_ETFS: frozenset[str] = frozenset({"518880.SS", "159934.SZ", "159937.SZ", "518800.SS"})
_DEFAULT_BOND_ETFS: tuple[str, ...] = ("511260.SS", "511010.SS")
_DEFAULT_DIVIDEND_ETFS: tuple[str, ...] = ("512890.SS", "515080.SS", "510880.SS")
_DEFAULT_BROAD_ETFS: tuple[str, ...] = (
    "510300.SS",
    "510050.SS",
    "510500.SS",
    "512100.SS",
    "159915.SZ",
    "588000.SS",
)


def _is_forbidden_gold_etf(symbol: str) -> bool:
    """Case- and whitespace-normalized Gold ETF exclusion guard (RB-D09)."""
    return str(symbol).strip().upper() in _FORBIDDEN_GOLD_ETFS


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _sample_std(vals: Sequence[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mu = math.fsum(vals) / n
    var = math.fsum((x - mu) * (x - mu) for x in vals) / (n - 1)
    return math.sqrt(var) if (math.isfinite(var) and var > 0.0) else 0.0


@dataclass(frozen=True)
class CNMacroRegimeSnapshot:
    """Point-in-Time domestic CN macro, CGB yield curve, contrarian margin-crowding & FX snapshot."""

    cgb_10y: float | None = None
    cgb_10y_chg_20d: float = 0.0
    cgb_10y_chg_60d: float = 0.0
    cgb_slope_10y_2y: float | None = None
    margin_mom_20d: float = 0.0
    margin_mom_60d: float = 0.0
    margin_ratio_pct: float | None = None
    fx_cny_strength_20d: float = 0.0
    risk_on_score: float = 0.0
    bond_easing_score: float = 0.0
    regime_label: str = "CN_MACRO_NEUTRAL"


class CNMacroRegimeTimeline:
    """Stateless `bisect_right` PIT resolver for `cn_ashare_macro` (D6 non-monotonic safe)."""

    def __init__(self, macro_points: Sequence[CNAshareMacroPoint] = ()) -> None:
        by_series: dict[str, dict[int, float]] = {}
        for pt in macro_points:
            if not math.isfinite(float(pt.value)):
                continue
            by_series.setdefault(str(pt.series_id), {})[int(pt.ts_avail_ms)] = float(pt.value)

        self._ts: dict[str, list[int]] = {}
        self._vals: dict[str, list[float]] = {}
        for sid, mp in by_series.items():
            ordered = sorted(mp.items())
            self._ts[sid] = [k for k, _ in ordered]
            self._vals[sid] = [v for _, v in ordered]

    def get_value(self, series_id: str, as_of_ms: int, lag_obs: int = 0) -> float | None:
        ts_arr = self._ts.get(series_id)
        if not ts_arr:
            return None
        idx = bisect.bisect_right(ts_arr, int(as_of_ms)) - 1 - max(0, int(lag_obs))
        if idx < 0:
            return None
        return self._vals[series_id][idx]

    def resolve(self, as_of_ms: int) -> CNMacroRegimeSnapshot:
        """Resolve strictly causal CN macro state at `as_of_ms` (`ts_avail_ms <= as_of_ms`)."""
        cgb_now = self.get_value("CN_CGB_10Y", as_of_ms, 0)
        cgb_20 = self.get_value("CN_CGB_10Y", as_of_ms, 20)
        cgb_60 = self.get_value("CN_CGB_10Y", as_of_ms, 60)
        slope = self.get_value("CN_CGB_10Y_2Y_SLOPE", as_of_ms, 0)

        m_now = self.get_value("CN_MARGIN_BALANCE_RMB", as_of_ms, 0)
        m_20 = self.get_value("CN_MARGIN_BALANCE_RMB", as_of_ms, 20)
        m_60 = self.get_value("CN_MARGIN_BALANCE_RMB", as_of_ms, 60)
        m_rat = self.get_value("CN_MARGIN_RATIO_PCT", as_of_ms, 0)

        fx_now = self.get_value("CN_FX_USDCNY", as_of_ms, 0)
        fx_20 = self.get_value("CN_FX_USDCNY", as_of_ms, 20)

        cgb_chg_20 = (cgb_now - cgb_20) if (cgb_now is not None and cgb_20 is not None) else 0.0
        cgb_chg_60 = (cgb_now - cgb_60) if (cgb_now is not None and cgb_60 is not None) else 0.0
        margin_mom_20 = ((m_now / m_20) - 1.0) if (m_now is not None and m_20 is not None and m_20 > 0.0) else 0.0
        margin_mom_60 = ((m_now / m_60) - 1.0) if (m_now is not None and m_60 is not None and m_60 > 0.0) else 0.0
        # Positive fx_cny_strength_20d means USD/CNY fell -> RMB strengthened (capital inflow)
        fx_str_20 = -((fx_now / fx_20) - 1.0) if (fx_now is not None and fx_20 is not None and fx_20 > 0.0) else 0.0

        # Smooth, bounded domestic macro signals in [-1, +1]
        fx_sig = _clamp(fx_str_20 / 0.015, -1.0, 1.0)
        # RB-D04: In retail-dominated CN A-shares, rapid leveraged margin balance surges signal
        # crowding/overheating (contrarian -1 prior); deleveraging washes out speculative froth.
        margin_decrowd_sig = _clamp(-(0.6 * margin_mom_20 + 0.4 * margin_mom_60) / 0.04, -1.0, 1.0)
        # Falling 10Y CGB yield (PBOC easing) boosts sovereign bonds & high-dividend bond proxies
        bond_easing = _clamp((-0.55 * cgb_chg_20 - 0.45 * cgb_chg_60) / 0.25, -1.0, 1.0)
        slope_bonus = _clamp(((slope if slope is not None else 0.35) - 0.20) / 0.40, -0.5, 0.5)
        bond_easing_final = _clamp(bond_easing + 0.25 * slope_bonus, -1.0, 1.0)

        # Composite domestic equity risk-on score: PBOC liquidity easing + RMB strength + contrarian margin de-crowding
        risk_on = _clamp(0.45 * bond_easing_final + 0.30 * fx_sig + 0.25 * margin_decrowd_sig, -1.0, 1.0)
        if risk_on >= 0.30:
            label = "CN_MACRO_EASING_DECROWDED"
        elif risk_on <= -0.25:
            label = "CN_MACRO_OVERHEATED_TIGHTENING"
        elif bond_easing_final >= 0.20:
            label = "CN_MACRO_PBOC_EASING_CARRY"
        else:
            label = "CN_MACRO_NEUTRAL_BALANCED"

        return CNMacroRegimeSnapshot(
            cgb_10y=cgb_now,
            cgb_10y_chg_20d=cgb_chg_20,
            cgb_10y_chg_60d=cgb_chg_60,
            cgb_slope_10y_2y=slope,
            margin_mom_20d=margin_mom_20,
            margin_mom_60d=margin_mom_60,
            margin_ratio_pct=m_rat,
            fx_cny_strength_20d=fx_str_20,
            risk_on_score=risk_on,
            bond_easing_score=bond_easing_final,
            regime_label=label,
        )


class AShareMacroERPRotationKernel(AShareStrategyInterface):
    """A2: Pure Domestic CN Stock-Bond Macro-ERP, CGB Yield, Contrarian Margin & FX Rotation Kernel.

    Strictly forbids Gold ETFs (`518880.SS` etc.), enforcing a 100% pure CN domestic
    Sovereign Bond + High-Dividend/Low-Vol + Broad Equity allocation governed by:
      1. ChinaBond 10Y/2Y CGB Sovereign Yield Curve (PIT 18:00 CST)
      2. SSE/SZSE Margin Financing Contrarian De-Crowding (PIT T+1 08:30 CST)
      3. USD/CNY Exchange Rate Pressure (PIT T+1 09:00 CST)
      4. Causal Inverse-Volatility Risk Budgeting + Low-Turnover Rebalance Hysteresis
    """

    def __init__(
        self,
        *,
        strategy_id: str = "ashare_a2_macro_erp_rotation",
        macro_timeline: CNMacroRegimeTimeline | None = None,
        bond_symbols: Sequence[str] = _DEFAULT_BOND_ETFS,
        dividend_symbols: Sequence[str] = _DEFAULT_DIVIDEND_ETFS,
        broad_equity_symbols: Sequence[str] = _DEFAULT_BROAD_ETFS,
        base_bond_weight: float = 0.55,
        base_dividend_weight: float = 0.30,
        base_broad_weight: float = 0.10,
        macro_tilt_strength: float = 0.20,
        mom_lookback_days: int = 60,
        vol_lookback_days: int = 20,
        rebalance_every_bars: int = 10,
        inertia_buffer: float = 0.08,
        max_single_weight: float = 0.55,
        max_total_weight: float = 0.95,
        sell_stamp_duty_bps: float = COST_CN_A_SHARE.stamp_duty_sell_bps,
        warmup_bars: int = 62,
    ) -> None:
        super().__init__(
            strategy_id=strategy_id,
            sell_stamp_duty_bps=sell_stamp_duty_bps,
            max_single_weight=max_single_weight,
            warmup_bars=warmup_bars,
        )
        self.macro_timeline: CNMacroRegimeTimeline = macro_timeline or CNMacroRegimeTimeline()
        self.bond_symbols: tuple[str, ...] = tuple(str(s).strip().upper() for s in bond_symbols)
        self.dividend_symbols: tuple[str, ...] = tuple(str(s).strip().upper() for s in dividend_symbols)
        self.broad_equity_symbols: tuple[str, ...] = tuple(str(s).strip().upper() for s in broad_equity_symbols)
        for grp in (self.bond_symbols, self.dividend_symbols, self.broad_equity_symbols):
            bad = [s for s in grp if _is_forbidden_gold_etf(s)]
            if bad:
                raise ValueError(f"A2 pure CN stock-bond strategy forbids Gold ETFs: {sorted(bad)}")

        self.base_bond_weight: float = _clamp(float(base_bond_weight), 0.15, 0.85)
        self.base_dividend_weight: float = _clamp(float(base_dividend_weight), 0.05, 0.60)
        self.base_broad_weight: float = _clamp(float(base_broad_weight), 0.0, 0.50)
        self.macro_tilt_strength: float = _clamp(float(macro_tilt_strength), 0.0, 0.35)
        self.mom_lookback_days: int = max(15, int(mom_lookback_days))
        self.vol_lookback_days: int = max(10, int(vol_lookback_days))
        self.rebalance_every_bars: int = max(1, int(rebalance_every_bars))
        self.inertia_buffer: float = _clamp(float(inertia_buffer), 0.0, 0.30)
        self.max_total_weight: float = _clamp(float(max_total_weight), 0.20, 1.0)

        self._step_count: int = 0
        self._closes: dict[str, deque[float]] = {}
        self._log_rets: dict[str, deque[float]] = {}
        self._pending_exit_targets: dict[str, float] = {}

    def reset(self) -> None:
        super().reset()
        self._step_count = 0
        self._closes.clear()
        self._log_rets.clear()
        self._pending_exit_targets.clear()

    def clone_fresh(self) -> AShareMacroERPRotationKernel:
        return AShareMacroERPRotationKernel(
            strategy_id=self.strategy_id,
            macro_timeline=self.macro_timeline,
            bond_symbols=self.bond_symbols,
            dividend_symbols=self.dividend_symbols,
            broad_equity_symbols=self.broad_equity_symbols,
            base_bond_weight=self.base_bond_weight,
            base_dividend_weight=self.base_dividend_weight,
            base_broad_weight=self.base_broad_weight,
            macro_tilt_strength=self.macro_tilt_strength,
            mom_lookback_days=self.mom_lookback_days,
            vol_lookback_days=self.vol_lookback_days,
            rebalance_every_bars=self.rebalance_every_bars,
            inertia_buffer=self.inertia_buffer,
            max_single_weight=self.max_single_weight,
            max_total_weight=self.max_total_weight,
            sell_stamp_duty_bps=self.sell_stamp_duty_bps,
            warmup_bars=self.warmup_bars,
        )

    def resolve_price_limit_pct(self, symbol: str) -> float:
        """Recognize ±20% ChiNext/STAR ETFs (`159915.SZ`, `588000.SS`) alongside stock prefixes."""
        s = symbol.strip().upper()
        if self.daily_price_limit_pct is None and s in {"159915.SZ", "588000.SS", "588080.SS", "159949.SZ"}:
            return 20.0
        return super().resolve_price_limit_pct(symbol)

    def _update_symbol_history(self, sym: str, bar: CanonicalBar) -> tuple[float | None, float]:
        buf = self._closes.setdefault(sym, deque(maxlen=max(self.mom_lookback_days + 5, 80)))
        rbuf = self._log_rets.setdefault(sym, deque(maxlen=max(self.vol_lookback_days + 5, 40)))
        prev_close = buf[-1] if buf else None
        close = float(bar.close)
        # RB-D09: guard against NaN/Inf/non-positive closes contaminating momentum/volatility state
        if not (math.isfinite(close) and close > 0.0):
            return prev_close, (prev_close if prev_close is not None else 0.0)
        if prev_close is not None and prev_close > 0.0:
            rbuf.append(math.log(close / prev_close))
        buf.append(close)
        return prev_close, close

    def _symbol_vol_and_mom(self, sym: str) -> tuple[float, float, float]:
        """Return (annualized_vol, mom_long, mom_20d) strictly from causal history."""
        buf = self._closes.get(sym)
        rbuf = self._log_rets.get(sym)
        if not buf or len(buf) < 22 or not rbuf or len(rbuf) < self.vol_lookback_days:
            return 0.15, 0.0, 0.0
        vol_daily = _sample_std(list(rbuf)[-self.vol_lookback_days :])
        ann_vol = max(0.015, vol_daily * math.sqrt(242.0))
        c_now = buf[-1]
        c_20 = buf[-21]
        mom_20 = (c_now / c_20) - 1.0 if c_20 > 0.0 else 0.0
        lb = min(self.mom_lookback_days, len(buf) - 1)
        c_lb = buf[-(lb + 1)]
        mom_lb = (c_now / c_lb) - 1.0 if c_lb > 0.0 else 0.0
        if not (math.isfinite(ann_vol) and math.isfinite(mom_lb) and math.isfinite(mom_20)):
            return 0.15, 0.0, 0.0
        return ann_vol, mom_lb, mom_20

    def _allocate_bucket(
        self,
        candidates: Sequence[str],
        bucket_weight: float,
        *,
        top_k: int,
        require_positive_trend: bool,
    ) -> dict[str, float]:
        """Allocate `bucket_weight` across eligible printing symbols via inverse-vol + trend filter."""
        if bucket_weight <= 1e-6 or not candidates:
            return {}
        scored: list[tuple[float, float, str]] = []
        for sym in candidates:
            ann_vol, mom_lb, mom_20 = self._symbol_vol_and_mom(sym)
            trend_score = 0.65 * mom_lb + 0.35 * mom_20
            if not math.isfinite(trend_score) or not math.isfinite(ann_vol):
                continue
            if require_positive_trend and trend_score <= 0.0:
                continue
            risk_adj = trend_score / max(ann_vol, 0.02)
            scored.append((risk_adj, ann_vol, sym))
        if not scored:
            return {}
        scored.sort(key=lambda x: (-x[0], x[2]))
        chosen = scored[: max(1, int(top_k))]
        inv_vols = [(1.0 / max(av, 0.02), sym) for _, av, sym in chosen]
        denom = math.fsum(iv for iv, _ in inv_vols)
        if denom <= 0.0:
            return {}
        out: dict[str, float] = {}
        for iv, sym in inv_vols:
            w = min(self.max_single_weight, bucket_weight * (iv / denom))
            out[sym] = w
        return out

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any],
    ) -> dict[str, BarDecision]:
        self._step_count += 1
        ordered_syms = sorted(bars_at_ts.keys())
        for sym in ordered_syms:
            self._update_symbol_history(sym, bars_at_ts[sym])

        max_avail_ms = max((int(bars_at_ts[s].ts_avail_ms) for s in ordered_syms), default=int(ts_ms))
        macro_active = self.macro_tilt_strength > 1e-9
        macro = (
            self.macro_timeline.resolve(max_avail_ms)
            if macro_active
            else CNMacroRegimeSnapshot(regime_label="CN_STATIC_NO_MACRO")
        )

        # Hard case-insensitive exclusion of Gold ETFs if ever injected into bars_at_ts (RB-D09)
        printing_valid = [s for s in ordered_syms if not _is_forbidden_gold_etf(s)]
        upper_to_raw = {s.strip().upper(): s for s in printing_valid}

        # Reserve capacity already locked in non-printing/suspended held ETFs (RB-15)
        held_non_printing = math.fsum(
            max(0.0, float(w))
            for s, w in self._executed_weights.items()
            if s not in bars_at_ts and float(w) > 1e-6
        )
        available_total_w = max(0.0, self.max_total_weight - held_non_printing)

        desired_weights: dict[str, float] = {sym: 0.0 for sym in ordered_syms}
        regime_by_sym: dict[str, str] = {}
        for sym in ordered_syms:
            if _is_forbidden_gold_etf(sym):
                regime_by_sym[sym] = "ASHARE_A2_GOLD_ETF_EXCLUDED"

        if self._step_count < self.warmup_bars:
            for sym in printing_valid:
                regime_by_sym[sym] = "ASHARE_A2_WARMUP"
        else:
            is_rebalance_bar = (
                ((self._step_count - self.warmup_bars) % self.rebalance_every_bars) == 0
                or all(self._executed_weights.get(s, 0.0) <= 1e-6 for s in printing_valid)
            )
            if not is_rebalance_bar:
                for sym in printing_valid:
                    cur_w = max(0.0, float(self._executed_weights.get(sym, 0.0)))
                    pending_t = self._pending_exit_targets.get(sym)
                    if pending_t is not None and cur_w > pending_t + 1e-6:
                        desired_weights[sym] = pending_t
                        regime_by_sym[sym] = "ASHARE_A2_RETRY_BLOCKED_EXIT"
                    else:
                        if pending_t is not None:
                            self._pending_exit_targets.pop(sym, None)
                        desired_weights[sym] = cur_w
                        regime_by_sym[sym] = f"{macro.regime_label}_HOLD"
            else:
                # RB-D04: When macro_tilt_strength == 0.0, ZERO macro tilt or easing shift applies!
                tilt = (self.macro_tilt_strength * macro.risk_on_score) if macro_active else 0.0
                easing_shift = (0.40 * self.macro_tilt_strength * macro.bond_easing_score) if macro_active else 0.0

                w_bond_target = _clamp(self.base_bond_weight - 0.65 * tilt + easing_shift, 0.25, 0.85)
                w_div_target = _clamp(self.base_dividend_weight + 0.35 * tilt + 0.35 * easing_shift, 0.10, 0.55)
                w_broad_target = _clamp(self.base_broad_weight + 0.30 * tilt, 0.0, 0.40)

                tot_req = w_bond_target + w_div_target + w_broad_target
                if tot_req > available_total_w and tot_req > 0.0:
                    scale = available_total_w / tot_req
                    w_bond_target *= scale
                    w_div_target *= scale
                    w_broad_target *= scale

                bonds_avail = [upper_to_raw[s] for s in self.bond_symbols if s in upper_to_raw]
                divs_avail = [upper_to_raw[s] for s in self.dividend_symbols if s in upper_to_raw]
                broad_avail = [upper_to_raw[s] for s in self.broad_equity_symbols if s in upper_to_raw]

                alloc_broad = self._allocate_bucket(broad_avail, w_broad_target, top_k=2, require_positive_trend=True)
                unused_broad = max(0.0, w_broad_target - math.fsum(alloc_broad.values()))

                require_div_trend = bool(macro_active and macro.risk_on_score < -0.35)
                alloc_div = self._allocate_bucket(divs_avail, w_div_target, top_k=2, require_positive_trend=require_div_trend)
                unused_div = max(0.0, w_div_target - math.fsum(alloc_div.values()))

                w_bond_final = min(available_total_w, w_bond_target + unused_broad + unused_div)
                alloc_bond = self._allocate_bucket(bonds_avail, w_bond_final, top_k=2, require_positive_trend=False)

                raw_targets: dict[str, float] = {}
                for d_part in (alloc_bond, alloc_div, alloc_broad):
                    for s, w in d_part.items():
                        raw_targets[s] = min(self.max_single_weight, raw_targets.get(s, 0.0) + w)

                sum_raw = math.fsum(raw_targets.values())
                if sum_raw > available_total_w and sum_raw > 0.0:
                    sc = available_total_w / sum_raw
                    raw_targets = {s: w * sc for s, w in raw_targets.items()}

                for sym in printing_valid:
                    cur_w = max(0.0, float(self._executed_weights.get(sym, 0.0)))
                    des_w = max(0.0, float(raw_targets.get(sym, 0.0)))
                    if des_w > 1e-6 and cur_w > 1e-6 and abs(des_w - cur_w) < self.inertia_buffer:
                        des_w = cur_w
                    desired_weights[sym] = des_w
                    regime_by_sym[sym] = macro.regime_label

                gross_after = math.fsum(desired_weights.values())
                if gross_after > available_total_w and gross_after > 1e-9:
                    sc2 = available_total_w / gross_after
                    desired_weights = {s: w * sc2 for s, w in desired_weights.items()}

                for sym in printing_valid:
                    prev_w = max(0.0, float(self._executed_weights.get(sym, 0.0)))
                    if desired_weights[sym] < prev_w - 0.02:
                        self._pending_exit_targets[sym] = desired_weights[sym]
                    else:
                        self._pending_exit_targets.pop(sym, None)

        # Delegate final T+1 lock, ±10%/±20% board price-limit locks, long-only & stamp duty to AShareStrategyInterface!
        merged_ctx: dict[str, Any] = dict(context) if context else {}
        merged_ctx["desired_weights"] = desired_weights
        base_decisions = super().on_bar_step(ts_ms, bars_at_ts, merged_ctx)

        enriched: dict[str, BarDecision] = {}
        for sym, dec in base_decisions.items():
            sym_u = sym.strip().upper()
            is_gold = _is_forbidden_gold_etf(sym)
            meta = dict(dec.metadata or {})
            meta["a2_regime"] = regime_by_sym.get(sym, macro.regime_label)
            meta["cn_risk_on_score"] = round(macro.risk_on_score, 4)
            meta["cn_bond_easing_score"] = round(macro.bond_easing_score, 4)
            meta["cn_cgb_10y"] = macro.cgb_10y
            meta["cn_margin_mom_20d"] = round(macro.margin_mom_20d, 6)
            meta["cn_fx_cny_strength_20d"] = round(macro.fx_cny_strength_20d, 6)
            meta["excluded_gold_etf"] = is_gold
            raw_sig = (
                0.0
                if is_gold
                else (
                    macro.bond_easing_score
                    if sym_u in self.bond_symbols
                    else (
                        0.5 * macro.bond_easing_score + 0.5 * macro.risk_on_score
                        if sym_u in self.dividend_symbols
                        else macro.risk_on_score
                    )
                )
            )
            enriched[sym] = BarDecision(
                symbol=dec.symbol,
                ts_event_ms=dec.ts_event_ms,
                target_weight=0.0 if is_gold else dec.target_weight,
                raw_signal=float(raw_sig),
                regime_label=regime_by_sym.get(sym, macro.regime_label),
                stop_price=dec.stop_price,
                metadata=meta,
            )
        return enriched
