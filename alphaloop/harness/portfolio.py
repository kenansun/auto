"""Multi-Strategy Causal Portfolio Combiner & Capacity-Aware Scaling Engine (Phase 2 Module C).

Combines heterogeneous verified strategy sleeves (e.g., Funding Carry + Gold Trend CTA +
Crypto Majors Trend CTA) under strict zero-lookahead accounting:
  - Exact overlapping UTC daily date alignment (fail-closed if <min_overlap_days or >3 interior missing days).
  - Capital conservation: weights >= 0, sum(weights) == 1.0 (no hidden leverage).
  - Supports ex-ante fixed weights OR strictly causal trailing inverse-volatility weights (< day d, warmup >= 20d).
  - Deducts explicit rebalance turnover costs on rebalance dates.
  - Computes continuous calendar-time 70/30 IS/OOS retention aligned with SixGateVerifier.
  - Models capacity-aware capital scaling ($3k / $10k / $50k) anchored to measured L2 BBO depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

DAY_MS = 86_400 * 1000


@dataclass
class SleeveInput:
    """Input equity curve & metadata for one strategy sleeve."""

    sleeve_id: str
    equity_curve: list[tuple[int, float]]   # (ts_ms, equity_usd)
    capital_usd: float = 3000.0
    weight: float = 0.0                     # Used when weighting_mode == "fixed"
    avg_monthly_turnover_x: float = 2.0     # Estimated monthly notional turnover / allocated capital
    measured_top_depth_usd: float = 25000.0 # Measured L2 top-of-book depth from live bbo_ticks


@dataclass
class PortfolioResult:
    """Comprehensive multi-sleeve portfolio scorecard."""

    valid: bool
    error: str
    weighting_mode: str
    overlap_days: int
    start_ts_ms: int
    end_ts_ms: int
    capital_usd: float
    sleeve_weights_final: dict[str, float]
    total_return_pct: float
    annualized_apr_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    worst_30d_return_pct: float
    rebalance_cost_total_pct: float
    is_metrics: dict[str, float]
    oos_metrics: dict[str, float]
    oos_retention: float
    correlation_matrix: dict[str, dict[str, float]]
    sleeve_summaries: dict[str, dict[str, float]]
    scaling_projections: list[dict[str, Any]]
    equity_curve: list[tuple[int, float]] = field(default_factory=list)

    def to_dict(self, include_curve: bool = False) -> dict[str, Any]:
        out = {
            "valid": self.valid,
            "error": self.error,
            "weighting_mode": self.weighting_mode,
            "overlap_days": self.overlap_days,
            "start_ts_ms": self.start_ts_ms,
            "end_ts_ms": self.end_ts_ms,
            "capital_usd": self.capital_usd,
            "sleeve_weights_final": dict(self.sleeve_weights_final),
            "total_return_pct": self.total_return_pct,
            "annualized_apr_pct": self.annualized_apr_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "worst_30d_return_pct": self.worst_30d_return_pct,
            "rebalance_cost_total_pct": self.rebalance_cost_total_pct,
            "is_metrics": dict(self.is_metrics),
            "oos_metrics": dict(self.oos_metrics),
            "oos_retention": self.oos_retention,
            "correlation_matrix": self.correlation_matrix,
            "sleeve_summaries": self.sleeve_summaries,
            "scaling_projections": self.scaling_projections,
        }
        if include_curve:
            out["equity_curve"] = [[ts, round(eq, 4)] for ts, eq in self.equity_curve]
        return out


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    vx = sum((x - mx) ** 2 for x in xs[:n])
    vy = sum((y - my) ** 2 for y in ys[:n])
    if vx <= 1e-18 or vy <= 1e-18:
        return 0.0
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return max(-1.0, min(1.0, cov / math.sqrt(vx * vy)))


class PortfolioCombiner:
    """Combines daily equity curves across heterogeneous sleeves without lookahead."""

    @staticmethod
    def _invalid(error: str, mode: str, capital_usd: float) -> PortfolioResult:
        return PortfolioResult(
            valid=False,
            error=error,
            weighting_mode=mode,
            overlap_days=0,
            start_ts_ms=0,
            end_ts_ms=0,
            capital_usd=capital_usd,
            sleeve_weights_final={},
            total_return_pct=0.0,
            annualized_apr_pct=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown_pct=0.0,
            worst_30d_return_pct=0.0,
            rebalance_cost_total_pct=0.0,
            is_metrics={"net_pnl_usd": 0.0, "apr_pct": 0.0, "days": 0.0},
            oos_metrics={"net_pnl_usd": 0.0, "apr_pct": 0.0, "days": 0.0},
            oos_retention=0.0,
            correlation_matrix={},
            sleeve_summaries={},
            scaling_projections=[],
            equity_curve=[],
        )

    def combine(
        self,
        sleeves: list[SleeveInput],
        *,
        capital_usd: float = 10000.0,
        weighting_mode: str = "fixed",           # "fixed" | "causal_inv_vol"
        rebalance_every_days: int = 30,
        rebalance_cost_bps: float = 8.0,
        vol_window_days: int = 20,
        min_overlap_days: int = 60,
        max_interior_missing_days: int = 3,
        scaling_capitals: tuple[float, ...] = (3000.0, 10000.0, 50000.0),
    ) -> PortfolioResult:
        if not sleeves:
            return self._invalid("NO_SLEEVES_PROVIDED", weighting_mode, capital_usd)
        if not math.isfinite(capital_usd) or capital_usd <= 0.0:
            raise ValueError("capital_usd must be finite and > 0")
        if weighting_mode not in ("fixed", "causal_inv_vol"):
            raise ValueError(f"Unsupported weighting_mode: {weighting_mode!r}")

        ids = [s.sleeve_id for s in sleeves]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate sleeve_id detected in portfolio")

        # Validate ex-ante fixed weights strictly (capital conservation: w_i >= 0, sum == 1.0)
        fixed_weights: dict[str, float] = {}
        if weighting_mode == "fixed":
            w_sum = 0.0
            for s in sleeves:
                if isinstance(s.weight, bool) or not math.isfinite(s.weight) or s.weight < 0.0:
                    raise ValueError(f"Invalid weight for sleeve {s.sleeve_id}: {s.weight!r}")
                fixed_weights[s.sleeve_id] = float(s.weight)
                w_sum += float(s.weight)
            if abs(w_sum - 1.0) > 1e-6:
                raise ValueError(f"Fixed portfolio weights must sum to 1.0 (got {w_sum:.6f})")

        # Align each sleeve to UTC daily end-of-day equity (last snapshot <= 23:59:59 UTC)
        # Red-Team HIGH #6: Fail closed on ANY bankrupt (eq <= 0) or non-finite equity value (no survivorship truncation!)
        daily_maps: dict[str, dict[int, float]] = {}
        for s in sleeves:
            dmap: dict[int, float] = {}
            for ts, eq in sorted(s.equity_curve, key=lambda x: x[0]):
                if isinstance(eq, bool) or not isinstance(eq, (int, float)) or not math.isfinite(float(eq)) or float(eq) <= 0.0:
                    return self._invalid(f"INVALID_OR_BANKRUPT_EQUITY:{s.sleeve_id}", weighting_mode, capital_usd)
                d_idx = ts // DAY_MS
                dmap[d_idx] = float(eq)
            if not dmap:
                return self._invalid(f"EMPTY_SLEEVE:{s.sleeve_id}", weighting_mode, capital_usd)
            daily_maps[s.sleeve_id] = dmap

        common_days_set = set.intersection(*(set(m.keys()) for m in daily_maps.values()))
        common_days = sorted(common_days_set)
        if len(common_days) < min_overlap_days:
            return self._invalid(
                f"INSUFFICIENT_OVERLAP({len(common_days)}d<{min_overlap_days}d)",
                weighting_mode,
                capital_usd,
            )

        calendar_span_days = (common_days[-1] - common_days[0]) + 1
        interior_missing = calendar_span_days - len(common_days)
        if interior_missing > max_interior_missing_days:
            return self._invalid(
                f"INTERIOR_DATE_GAPS_EXCEEDED({interior_missing}>{max_interior_missing_days})",
                weighting_mode,
                capital_usd,
            )

        # Compute aligned daily returns per sleeve for days[1..N-1]
        sleeve_rets: dict[str, list[float]] = {sid: [] for sid in ids}
        for i in range(1, len(common_days)):
            d_prev, d_cur = common_days[i - 1], common_days[i]
            for sid in ids:
                e0 = daily_maps[sid][d_prev]
                e1 = daily_maps[sid][d_cur]
                sleeve_rets[sid].append((e1 / e0) - 1.0)

        # Simulate daily portfolio compounding with explicit rebalance drift & turnover costs
        rebal_every = max(1, int(rebalance_every_days))
        vol_win = max(20, int(vol_window_days))
        if weighting_mode == "fixed":
            cur_w = dict(fixed_weights)
        else:
            eq_w = 1.0 / len(ids)
            cur_w = {sid: eq_w for sid in ids}

        port_eq = capital_usd
        port_curve: list[tuple[int, float]] = [(common_days[0] * DAY_MS, port_eq)]
        port_daily_rets: list[float] = []
        total_rebal_cost_frac = 0.0

        for day_i in range(len(common_days) - 1):
            # Step 1: Portfolio gross return on day_i using weights established BEFORE day_i
            gross_r = sum(cur_w[sid] * sleeve_rets[sid][day_i] for sid in ids)

            # Update drifted weights after market move
            denom = 1.0 + gross_r
            if denom > 1e-9:
                drifted_w = {sid: (cur_w[sid] * (1.0 + sleeve_rets[sid][day_i])) / denom for sid in ids}
            else:
                drifted_w = dict(cur_w)

            # Step 2: Check scheduled rebalance (strictly causal: uses returns[0..day_i] only AFTER day_i!)
            rebal_drag = 0.0
            if (day_i + 1) % rebal_every == 0:
                if weighting_mode == "fixed":
                    target_w = dict(fixed_weights)
                else:
                    if (day_i + 1) < vol_win:
                        eq_w = 1.0 / len(ids)
                        target_w = {sid: eq_w for sid in ids}
                    else:
                        inv_vols: dict[str, float] = {}
                        for sid in ids:
                            window_r = sleeve_rets[sid][day_i + 1 - vol_win : day_i + 1]
                            mr = sum(window_r) / len(window_r)
                            vr = sum((r - mr) ** 2 for r in window_r) / max(1, len(window_r) - 1)
                            sd = math.sqrt(max(0.0, vr))
                            # Floor daily vol at 1bp so near-flat cash/carry sleeves don't blow up to infinity
                            inv_vols[sid] = 1.0 / max(0.0001, sd)
                        iv_sum = sum(inv_vols.values())
                        target_w = {sid: inv_vols[sid] / iv_sum for sid in ids}

                one_way_turnover = 0.5 * sum(abs(target_w[sid] - drifted_w[sid]) for sid in ids)
                rebal_drag = one_way_turnover * (max(0.0, rebalance_cost_bps) / 10000.0)
                total_rebal_cost_frac += rebal_drag
                cur_w = target_w
            else:
                cur_w = drifted_w

            net_r = gross_r - rebal_drag
            port_daily_rets.append(net_r)
            port_eq = max(1e-6, port_eq * (1.0 + net_r))
            port_curve.append((common_days[day_i + 1] * DAY_MS, port_eq))

        # Summary metrics
        span_days = max(1.0, float(common_days[-1] - common_days[0]))
        total_ret_pct = ((port_eq / capital_usd) - 1.0) * 100.0
        ann_apr_pct = total_ret_pct * (365.0 / span_days)

        if len(port_daily_rets) >= 3:
            mr = sum(port_daily_rets) / len(port_daily_rets)
            vr = sum((r - mr) ** 2 for r in port_daily_rets) / (len(port_daily_rets) - 1)
            sd = math.sqrt(max(0.0, vr))
            sharpe = (mr / sd) * math.sqrt(365.0) if sd > 1e-12 else 0.0
        else:
            sharpe = 0.0

        peak = capital_usd
        max_dd_pct = 0.0
        for _, eq in port_curve:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = ((peak - eq) / peak) * 100.0
                if dd > max_dd_pct:
                    max_dd_pct = dd

        calmar = (ann_apr_pct / max_dd_pct) if max_dd_pct > 0.05 else (ann_apr_pct / 0.05 if ann_apr_pct > 0 else 0.0)

        # Worst rolling 30-day return
        worst_30d = 0.0
        if len(port_curve) > 30:
            for i in range(30, len(port_curve)):
                e0, e1 = port_curve[i - 30][1], port_curve[i][1]
                if e0 > 0:
                    r30 = ((e1 / e0) - 1.0) * 100.0
                    if r30 < worst_30d:
                        worst_30d = r30

        # Continuous calendar-time 70/30 IS/OOS (exact same formula as SixGateVerifier)
        split_ts = port_curve[0][0] + int((port_curve[-1][0] - port_curve[0][0]) * 0.70)
        split_idx = next((i for i, (ts, _) in enumerate(port_curve) if ts >= split_ts), len(port_curve))
        split_idx = max(2, min(split_idx, len(port_curve) - 1))
        is_seg = port_curve[:split_idx]
        oos_seg = port_curve[split_idx - 1:]

        def _seg(seg: list[tuple[int, float]]) -> dict[str, float]:
            if len(seg) < 2 or seg[0][1] <= 0 or seg[-1][0] <= seg[0][0]:
                return {"net_pnl_usd": 0.0, "apr_pct": 0.0, "days": 0.0}
            d_days = max(1.0, (seg[-1][0] - seg[0][0]) / float(DAY_MS))
            pnl = seg[-1][1] - seg[0][1]
            apr = ((seg[-1][1] / seg[0][1]) - 1.0) * 100.0 * (365.0 / d_days)
            return {"net_pnl_usd": round(pnl, 2), "apr_pct": round(apr, 2), "days": round(d_days, 1)}

        is_info = _seg(is_seg)
        oos_info = _seg(oos_seg)
        oos_ret = (oos_info["apr_pct"] / is_info["apr_pct"]) if is_info["apr_pct"] > 1.0 else (1.0 if oos_info["apr_pct"] > 0 else 0.0)

        # Pairwise daily return correlation matrix + individual sleeve summaries over overlap window
        corr_mat: dict[str, dict[str, float]] = {}
        sleeve_sums: dict[str, dict[str, float]] = {}
        for sid in ids:
            corr_mat[sid] = {other: round(_pearson_corr(sleeve_rets[sid], sleeve_rets[other]), 4) for other in ids}
            s_eq0 = daily_maps[sid][common_days[0]]
            s_eq1 = daily_maps[sid][common_days[-1]]
            s_ret_pct = ((s_eq1 / s_eq0) - 1.0) * 100.0 if s_eq0 > 0 else 0.0
            s_apr = s_ret_pct * (365.0 / span_days)
            sr = sleeve_rets[sid]
            sm = sum(sr) / len(sr)
            sv = sum((x - sm) ** 2 for x in sr) / max(1, len(sr) - 1)
            s_sh = (sm / math.sqrt(sv)) * math.sqrt(365.0) if sv > 1e-18 else 0.0
            sp, s_mdd = s_eq0, 0.0
            for d in common_days:
                ev = daily_maps[sid][d]
                if ev > sp:
                    sp = ev
                if sp > 0:
                    s_mdd = max(s_mdd, ((sp - ev) / sp) * 100.0)
            sleeve_sums[sid] = {
                "weight_final": round(cur_w[sid], 4),
                "total_return_pct": round(s_ret_pct, 2),
                "apr_pct": round(s_apr, 2),
                "sharpe": round(s_sh, 3),
                "max_dd_pct": round(s_mdd, 3),
            }

        # Capacity-aware capital scaling anchored to measured BBO top-of-book depth
        scaling: list[dict[str, Any]] = []
        sleeve_by_id = {s.sleeve_id: s for s in sleeves}
        for cap in scaling_capitals:
            # Estimate annual incremental market-impact drag (sqrt law relative to measured L2 top depth)
            ann_impact_drag_pct = 0.0
            for sid in ids:
                s_obj = sleeve_by_id[sid]
                alloc_cap = cap * cur_w[sid]
                # Assume typical single-order slice is 20% of sleeve capital
                order_slice_usd = alloc_cap * 0.20
                top_depth = max(1000.0, float(s_obj.measured_top_depth_usd))
                # Conservative 1.5bp baseline impact at 100% of top-of-book depth, scaled by sqrt(slice/depth)
                extra_slip_bps = 1.5 * math.sqrt(max(0.0, order_slice_usd / top_depth))
                ann_turnover = max(0.5, float(s_obj.avg_monthly_turnover_x)) * 12.0
                # Drag on total portfolio = weight * ann_turnover * extra_slip_bps / 100 (% APR)
                ann_impact_drag_pct += cur_w[sid] * ann_turnover * (extra_slip_bps / 100.0)

            net_apr_est = ann_apr_pct - ann_impact_drag_pct
            monthly_usd_est = cap * (net_apr_est / 100.0) / 12.0
            scaling.append({
                "capital_usd": round(cap, 2),
                "gross_backtest_apr_pct": round(ann_apr_pct, 2),
                "capacity_impact_drag_apr_pct": round(ann_impact_drag_pct, 2),
                "net_capacity_adjusted_apr_pct": round(net_apr_est, 2),
                "est_monthly_net_usd": round(monthly_usd_est, 2),
            })

        return PortfolioResult(
            valid=True,
            error="",
            weighting_mode=weighting_mode,
            overlap_days=len(common_days),
            start_ts_ms=common_days[0] * DAY_MS,
            end_ts_ms=common_days[-1] * DAY_MS,
            capital_usd=capital_usd,
            sleeve_weights_final={k: round(v, 4) for k, v in cur_w.items()},
            total_return_pct=round(total_ret_pct, 3),
            annualized_apr_pct=round(ann_apr_pct, 2),
            sharpe_ratio=round(sharpe, 3),
            calmar_ratio=round(calmar, 3),
            max_drawdown_pct=round(max_dd_pct, 3),
            worst_30d_return_pct=round(worst_30d, 3),
            rebalance_cost_total_pct=round(total_rebal_cost_frac * 100.0, 4),
            is_metrics=is_info,
            oos_metrics=oos_info,
            oos_retention=round(oos_ret, 3),
            correlation_matrix=corr_mat,
            sleeve_summaries=sleeve_sums,
            scaling_projections=scaling,
            equity_curve=port_curve,
        )
