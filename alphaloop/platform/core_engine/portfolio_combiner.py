"""Top-Level Return-Stream Portfolio Combiner (F-11).

Unites independently validated asset-native sleeves (e.g., Web3 + Gold) STRICTLY at the
return-stream layer — ZERO shared signal kernels, ZERO parameter-grid coupling, and explicit
REA-001 capital & margin bucket separation across domains.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from alphaloop.platform.core_engine.accounting import SimulationResult
from alphaloop.platform.core_engine.validation import (
    fast_pearson_corr,
    fast_var_std,
    sharpe_from_returns,
)

logger = logging.getLogger(__name__)

_DAY_MS = 86_400_000


def _margin_bucket_for_domain(domain: str) -> str:
    d = str(domain).strip().lower()
    if d == "web3":
        return "web3_perp_margin"
    if d == "gold":
        return "gold_tradfi_margin"
    if d == "ashare":
        return "ashare_cash"
    return f"{d or 'generic'}_margin"


@dataclass(frozen=True)
class SleeveCapitalAllocation:
    """Explicit REA-001 capital and margin bucket allocation for one strategy sleeve."""

    sleeve_id: str
    asset_domain: str
    margin_bucket: str
    weight_final: float
    allocated_capital_usd: float
    max_sleeve_leverage: float
    effective_notional_cap_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CombinedPortfolioResult:
    """Return-stream combination output with correlation, DR, CAGR, and REA-001 margin separation."""

    valid: bool
    error: str
    method: str
    overlap_days: int
    span_calendar_days: float
    start_ts_ms: int
    end_ts_ms: int
    total_capital_usd: float
    final_equity_usd: float
    total_return_pct: float
    cagr_pct: float  # PRIMARY (D13)
    simple_apr_pct: float  # SECONDARY
    sharpe_ratio: float
    calmar_ratio: float
    realized_annual_vol_pct: float
    max_drawdown_pct: float
    rebalance_cost_total_pct: float
    diversification_ratio: float
    pairwise_corr_matrix: dict[str, dict[str, float]]
    sleeve_weights_final: dict[str, float]
    cash_reserve_weight: float
    sleeve_capital_allocations: dict[str, SleeveCapitalAllocation]
    sleeve_summaries: dict[str, dict[str, float]]
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    period_returns: list[float] = field(default_factory=list)
    sleeve_weights_history: list[tuple[int, dict[str, float]]] = field(default_factory=list)

    def to_dict(self, include_curve: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "valid": self.valid,
            "error": self.error,
            "method": self.method,
            "overlap_days": self.overlap_days,
            "span_calendar_days": round(self.span_calendar_days, 2),
            "start_ts_ms": self.start_ts_ms,
            "end_ts_ms": self.end_ts_ms,
            "total_capital_usd": round(self.total_capital_usd, 2),
            "final_equity_usd": round(self.final_equity_usd, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "cagr_pct": round(self.cagr_pct, 4),
            "simple_apr_pct": round(self.simple_apr_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "realized_annual_vol_pct": round(self.realized_annual_vol_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "rebalance_cost_total_pct": round(self.rebalance_cost_total_pct, 4),
            "diversification_ratio": round(self.diversification_ratio, 4),
            "pairwise_corr_matrix": {
                k: dict(v) for k, v in sorted(self.pairwise_corr_matrix.items())
            },
            "sleeve_weights_final": dict(sorted(self.sleeve_weights_final.items())),
            "cash_reserve_weight": round(self.cash_reserve_weight, 6),
            "sleeve_capital_allocations": {
                k: v.to_dict() for k, v in sorted(self.sleeve_capital_allocations.items())
            },
            "sleeve_summaries": {k: dict(v) for k, v in sorted(self.sleeve_summaries.items())},
        }
        if include_curve:
            out["equity_curve"] = list(self.equity_curve)
            out["period_returns"] = list(self.period_returns)
            out["sleeve_weights_history"] = [
                (ts, dict(w)) for ts, w in self.sleeve_weights_history
            ]
        return out


def _invalid_result(method: str, error: str, capital_usd: float) -> CombinedPortfolioResult:
    return CombinedPortfolioResult(
        valid=False,
        error=error,
        method=method,
        overlap_days=0,
        span_calendar_days=0.0,
        start_ts_ms=0,
        end_ts_ms=0,
        total_capital_usd=capital_usd,
        final_equity_usd=capital_usd,
        total_return_pct=0.0,
        cagr_pct=0.0,
        simple_apr_pct=0.0,
        sharpe_ratio=0.0,
        calmar_ratio=0.0,
        realized_annual_vol_pct=0.0,
        max_drawdown_pct=0.0,
        rebalance_cost_total_pct=0.0,
        diversification_ratio=1.0,
        pairwise_corr_matrix={},
        sleeve_weights_final={},
        cash_reserve_weight=1.0,
        sleeve_capital_allocations={},
        sleeve_summaries={},
    )


class PortfolioCombiner:
    """Strictly causal return-stream combiner across decoupled asset-native strategy sleeves."""

    def combine_sleeves(
        self,
        sleeve_results: dict[str, SimulationResult],
        *,
        method: str = "causal_inv_vol",  # "causal_inv_vol" | "causal_corr_aware" | "equal_weight"
        target_vol_pct: float = 15.0,
        vol_lookback_days: int = 25,
        corr_shrinkage: float = 0.25,
        rebalance_every_days: int = 5,
        rebalance_cost_bps: float = 4.0,
        min_overlap_days: int = 20,
        min_sleeve_ann_vol_pct: float = 2.0,
        capital_usd: float | None = None,
    ) -> CombinedPortfolioResult:
        """Combine independently validated sleeve return streams with zero signal-level coupling."""
        m_name = str(method).strip().lower()
        if m_name not in ("causal_inv_vol", "causal_corr_aware", "equal_weight"):
            raise ValueError(f"Unsupported combination method: {method!r}")

        if not sleeve_results:
            return _invalid_result(m_name, "sleeve_results is empty", float(capital_usd or 10_000.0))

        sleeve_ids = sorted(sleeve_results.keys())
        total_cap = (
            float(capital_usd)
            if capital_usd is not None
            else math.fsum(float(sleeve_results[s].capital_usd) for s in sleeve_ids)
        )
        if not math.isfinite(total_cap) or total_cap <= 0.0:
            return _invalid_result(m_name, f"Invalid total capital_usd={total_cap}", 10_000.0)

        # 1. Collapse each sleeve's equity_curve to UTC calendar day (handles 24/7 Web3 + 252d Gold!)
        daily_maps: dict[str, dict[int, tuple[int, float]]] = {}
        for sid in sleeve_ids:
            res = sleeve_results[sid]
            if not res.equity_curve:
                return _invalid_result(m_name, f"Sleeve {sid!r} has empty equity_curve", total_cap)
            dmap: dict[int, tuple[int, float]] = {}
            for ts_ms, eq in sorted(res.equity_curve, key=lambda pt: int(pt[0])):
                eq_f = float(eq)
                if not math.isfinite(eq_f) or eq_f <= 0.0:
                    return _invalid_result(
                        m_name, f"Sleeve {sid!r} contains non-positive/non-finite equity {eq_f}", total_cap
                    )
                day_key = int(ts_ms) // _DAY_MS
                dmap[day_key] = (int(ts_ms), eq_f)
            daily_maps[sid] = dmap

        common_days = sorted(set.intersection(*(set(m.keys()) for m in daily_maps.values())))
        if len(common_days) < max(2, int(min_overlap_days)):
            return _invalid_result(
                m_name,
                f"Insufficient overlapping UTC days ({len(common_days)} < {min_overlap_days})",
                total_cap,
            )

        # Aligned daily-endpoint returns (Friday->Monday naturally bridges weekend crypto vs TradFi)
        sleeve_rets: dict[str, list[float]] = {sid: [] for sid in sleeve_ids}
        ret_ts_ms: list[int] = []
        for idx in range(1, len(common_days)):
            d_prev = common_days[idx - 1]
            d_cur = common_days[idx]
            ts_cur = max(daily_maps[sid][d_cur][0] for sid in sleeve_ids)
            ret_ts_ms.append(ts_cur)
            for sid in sleeve_ids:
                e0 = daily_maps[sid][d_prev][1]
                e1 = daily_maps[sid][d_cur][1]
                sleeve_rets[sid].append((e1 / e0) - 1.0)

        n_rets = len(ret_ts_ms)
        m_sleeves = len(sleeve_ids)
        lb = max(5, int(vol_lookback_days))
        warmup_min = min(10, lb)
        reb_every = max(1, int(rebalance_every_days))
        lam = max(0.0, min(1.0, float(corr_shrinkage)))
        target_vol_frac = max(0.01, float(target_vol_pct) / 100.0)
        rebal_cost_frac = max(0.0, float(rebalance_cost_bps) / 10_000.0)
        min_vol_pct_f = float(min_sleeve_ann_vol_pct)
        if not math.isfinite(min_vol_pct_f) or min_vol_pct_f <= 0.0:
            raise ValueError(
                f"min_sleeve_ann_vol_pct must be finite and > 0, got {min_sleeve_ann_vol_pct!r}"
            )
        min_vol_frac = max(1e-4, min_vol_pct_f / 100.0)

        # 2. Strictly causal step-by-step portfolio simulation (uses ONLY returns[:i] and days[:i+1])
        eq = total_cap
        peak_eq = total_cap
        max_dd = 0.0
        cur_w: dict[str, float] = {sid: 0.0 for sid in sleeve_ids}
        drifted_w: dict[str, float] = {sid: 0.0 for sid in sleeve_ids}

        start_ts = max(daily_maps[sid][common_days[0]][0] for sid in sleeve_ids)
        eq_curve: list[tuple[int, float]] = [(start_ts, eq)]
        port_rets: list[float] = []
        weights_hist: list[tuple[int, dict[str, float]]] = []
        total_rebal_cost_pct = 0.0

        for i in range(n_rets):
            rebal_drag = 0.0
            if i == 0 or (i % reb_every == 0):
                if i < warmup_min:
                    target_w = {sid: 1.0 / float(m_sleeves) for sid in sleeve_ids}
                else:
                    win_lo = max(0, i - lb)
                    # Strictly causal annualization factor from past common_days[win_lo .. i]
                    win_days = max(1.0, float(common_days[i] - common_days[win_lo]))
                    causal_ppy = max(50.0, min(365.0, float(i - win_lo) * (365.0 / win_days)))
                    ann_scale_i = math.sqrt(causal_ppy)

                    hists = {sid: sleeve_rets[sid][win_lo:i] for sid in sleeve_ids}
                    ann_vols: dict[str, float] = {}
                    for sid in sleeve_ids:
                        sd_i = fast_var_std(hists[sid], ddof=1)[2]
                        ann_vols[sid] = max(min_vol_frac, sd_i * ann_scale_i)

                    # Causal shrunk correlation matrix over trailing lookback
                    r_mat: dict[str, dict[str, float]] = {
                        a: {b: (1.0 if a == b else 0.0) for b in sleeve_ids}
                        for a in sleeve_ids
                    }
                    for a_i in range(m_sleeves):
                        sa = sleeve_ids[a_i]
                        for b_i in range(a_i + 1, m_sleeves):
                            sb = sleeve_ids[b_i]
                            rho = fast_pearson_corr(hists[sa], hists[sb])
                            shrunk = (1.0 - lam) * rho
                            r_mat[sa][sb] = shrunk
                            r_mat[sb][sa] = shrunk

                    if m_name == "equal_weight":
                        w_tilde = {sid: 1.0 / float(m_sleeves) for sid in sleeve_ids}
                    elif m_name == "causal_inv_vol":
                        inv_v = {sid: 1.0 / max(ann_vols[sid], min_vol_frac) for sid in sleeve_ids}
                        inv_sum = math.fsum(inv_v.values())
                        w_tilde = {sid: inv_v[sid] / inv_sum for sid in sleeve_ids}
                    else:
                        # causal_corr_aware: downweight positively correlated redundant sleeves
                        raw_u: dict[str, float] = {}
                        for sid in sleeve_ids:
                            pos_corr_sum = math.fsum(max(0.0, r_mat[sid][other]) for other in sleeve_ids)
                            raw_u[sid] = 1.0 / (max(ann_vols[sid], min_vol_frac) * math.sqrt(max(1.0, pos_corr_sum)))
                        u_sum = math.fsum(raw_u.values())
                        w_tilde = {sid: raw_u[sid] / u_sum for sid in sleeve_ids}

                    # Causal ex-ante portfolio volatility targeting (vol_scale <= 1.0, zero borrowing)
                    p_var = 0.0
                    for sa in sleeve_ids:
                        wa = w_tilde[sa] * ann_vols[sa]
                        for sb in sleeve_ids:
                            p_var += wa * (w_tilde[sb] * ann_vols[sb]) * r_mat[sa][sb]
                    hat_sigma_ann = math.sqrt(max(1e-12, p_var))
                    vol_scale = min(1.0, target_vol_frac / max(hat_sigma_ann, 1e-4))
                    target_w = {sid: vol_scale * w_tilde[sid] for sid in sleeve_ids}

                one_way_turn = 0.5 * math.fsum(abs(target_w[s] - drifted_w[s]) for s in sleeve_ids)
                rebal_drag = one_way_turn * rebal_cost_frac
                total_rebal_cost_pct += rebal_drag * 100.0
                cur_w = dict(target_w)
            else:
                cur_w = dict(drifted_w)

            weights_hist.append((ret_ts_ms[i], dict(cur_w)))

            gross_ret = math.fsum(cur_w[sid] * sleeve_rets[sid][i] for sid in sleeve_ids)
            net_ret = (1.0 - rebal_drag) * (1.0 + gross_ret) - 1.0
            port_rets.append(net_ret)
            eq = max(0.0, eq * max(0.0, 1.0 + net_ret))
            eq_curve.append((ret_ts_ms[i], eq))

            if eq > peak_eq:
                peak_eq = eq
            dd = ((peak_eq - eq) / peak_eq) * 100.0 if peak_eq > 0.0 else 100.0
            if dd > max_dd:
                max_dd = dd

            gross_denom = max(1e-9, 1.0 + gross_ret)
            drifted_w = {sid: (cur_w[sid] * (1.0 + sleeve_rets[sid][i])) / gross_denom for sid in sleeve_ids}

        # 3. Full-sample summary metrics, pairwise correlation matrix, and Diversification Ratio
        span_days = max(1.0, float(common_days[-1] - common_days[0]))
        full_ppy = max(1.0, float(n_rets) * (365.0 / span_days))
        full_ann_scale = math.sqrt(full_ppy)

        corr_matrix: dict[str, dict[str, float]] = {sid: {} for sid in sleeve_ids}
        sleeve_ann_vols: dict[str, float] = {}
        sleeve_summaries: dict[str, dict[str, float]] = {}

        for sid in sleeve_ids:
            sr_s = sharpe_from_returns(sleeve_rets[sid], full_ppy)
            vol_s = fast_var_std(sleeve_rets[sid], ddof=1)[2] * full_ann_scale
            sleeve_ann_vols[sid] = vol_s
            sleeve_summaries[sid] = {
                "aligned_sharpe": round(sr_s, 4),
                "aligned_ann_vol_pct": round(vol_s * 100.0, 4),
                "source_cagr_pct": round(float(sleeve_results[sid].cagr_pct), 4),
                "source_max_dd_pct": round(float(sleeve_results[sid].max_drawdown_pct), 4),
            }
            for other in sleeve_ids:
                if sid == other:
                    corr_matrix[sid][other] = 1.0
                else:
                    corr_matrix[sid][other] = round(
                        fast_pearson_corr(sleeve_rets[sid], sleeve_rets[other]), 4
                    )

        final_w = dict(cur_w)
        w_sum = math.fsum(final_w.values())
        cash_w = max(0.0, 1.0 - w_sum)

        # Choueifaty-Coignard Diversification Ratio on normalized active weights
        if w_sum > 1e-9:
            w_norm = {sid: final_w[sid] / w_sum for sid in sleeve_ids}
        else:
            w_norm = {sid: 1.0 / float(m_sleeves) for sid in sleeve_ids}
        num_dr = math.fsum(w_norm[sid] * sleeve_ann_vols[sid] for sid in sleeve_ids)
        den_var = 0.0
        for sa in sleeve_ids:
            for sb in sleeve_ids:
                den_var += (
                    w_norm[sa]
                    * w_norm[sb]
                    * sleeve_ann_vols[sa]
                    * sleeve_ann_vols[sb]
                    * corr_matrix[sa][sb]
                )
        den_dr = math.sqrt(max(1e-12, den_var))
        div_ratio = (num_dr / den_dr) if (den_dr > 1e-9 and num_dr > 0.0) else 1.0

        # 4. REA-001 Capital & Margin Bucket Separation (strict conservation!)
        allocations: dict[str, SleeveCapitalAllocation] = {}
        for sid in sleeve_ids:
            res = sleeve_results[sid]
            w_f = float(final_w[sid])
            alloc_cap = total_cap * w_f
            max_lev = max(1.0, float(res.max_gross_leverage))
            allocations[sid] = SleeveCapitalAllocation(
                sleeve_id=sid,
                asset_domain=str(res.asset_domain),
                margin_bucket=_margin_bucket_for_domain(res.asset_domain),
                weight_final=round(w_f, 6),
                allocated_capital_usd=alloc_cap,
                max_sleeve_leverage=round(max_lev, 4),
                effective_notional_cap_usd=alloc_cap * max_lev,
            )

        total_ret_pct = ((eq / total_cap) - 1.0) * 100.0
        cagr_pct = (((eq / total_cap) ** (365.0 / span_days)) - 1.0) * 100.0 if eq > 0.0 else -100.0
        simple_apr_pct = total_ret_pct * (365.0 / span_days)
        comb_sr = sharpe_from_returns(port_rets, full_ppy)
        comb_vol_pct = fast_var_std(port_rets, ddof=1)[2] * full_ann_scale * 100.0
        calmar = (
            cagr_pct / max_dd
            if max_dd > 0.1
            else (cagr_pct / 0.1 if cagr_pct > 0.0 else 0.0)
        )

        return CombinedPortfolioResult(
            valid=True,
            error="",
            method=m_name,
            overlap_days=len(common_days),
            span_calendar_days=span_days,
            start_ts_ms=start_ts,
            end_ts_ms=ret_ts_ms[-1],
            total_capital_usd=total_cap,
            final_equity_usd=eq,
            total_return_pct=total_ret_pct,
            cagr_pct=cagr_pct,
            simple_apr_pct=simple_apr_pct,
            sharpe_ratio=comb_sr,
            calmar_ratio=calmar,
            realized_annual_vol_pct=comb_vol_pct,
            max_drawdown_pct=max_dd,
            rebalance_cost_total_pct=total_rebal_cost_pct,
            diversification_ratio=div_ratio,
            pairwise_corr_matrix=corr_matrix,
            sleeve_weights_final={k: round(v, 6) for k, v in sorted(final_w.items())},
            cash_reserve_weight=cash_w,
            sleeve_capital_allocations=allocations,
            sleeve_summaries=sleeve_summaries,
            equity_curve=eq_curve,
            period_returns=port_rets,
            sleeve_weights_history=weights_hist,
        )
