"""Causal Correlation-Aware Portfolio Risk Allocator (Fixes D9).

Enforces portfolio-level risk physics identically in historical batch simulation and
bar-by-bar live streaming (verified by Gate G7):
  1. Hard `max_positions` cardinality cap ranked deterministically by `abs(raw_signal)`.
  2. Per-symbol weight & leverage caps plus calendar `allow_short` enforcement.
  3. Causal Ledoit-Wolf-style shrunk correlation matrix -> Instrument Diversification
     Multiplier (IDM) & ex-ante portfolio volatility targeting (eliminating v3's 2.03x
     vol overshoot on correlated crypto majors).
  4. Hard portfolio gross leverage cap.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from alphaloop.platform.assets import (
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_CRYPTO_SPOT,
    AssetSpec,
)
from alphaloop.platform.core_engine.interfaces import BarDecision

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:
    _np = None  # benign: optional Research Plane accelerator

logger = logging.getLogger(__name__)


def _fast_sample_std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    if _np is not None and n >= 128:
        arr = _np.asarray(xs, dtype=_np.float64)
        mu = float(arr.mean())
        diff = arr - mu
        var = float(_np.dot(diff, diff)) / (n - 1)
        return math.sqrt(var) if (math.isfinite(var) and var > 0.0) else 0.0
    mu = math.fsum(xs) / n
    ss = math.fsum((x - mu) * (x - mu) for x in xs)
    var = ss / (n - 1)
    return math.sqrt(var) if (math.isfinite(var) and var > 0.0) else 0.0


def _fast_pearson_tail(xs: Sequence[float], ys: Sequence[float], m: int) -> float:
    if m < 3:
        return 0.0
    tx = xs[-m:]
    ty = ys[-m:]
    if _np is not None and m >= 64:
        ax = _np.asarray(tx, dtype=_np.float64)
        ay = _np.asarray(ty, dtype=_np.float64)
        dx = ax - float(ax.mean())
        dy = ay - float(ay.mean())
        vx = float(_np.dot(dx, dx))
        vy = float(_np.dot(dy, dy))
        if vx <= 1e-24 or vy <= 1e-24:
            return 0.0
        corr = float(_np.dot(dx, dy)) / math.sqrt(vx * vy)
        return max(-0.95, min(0.95, corr)) if math.isfinite(corr) else 0.0
    mx = math.fsum(tx) / m
    my = math.fsum(ty) / m
    vx = math.fsum((a - mx) * (a - mx) for a in tx)
    vy = math.fsum((b - my) * (b - my) for b in ty)
    if vx <= 1e-24 or vy <= 1e-24:
        return 0.0
    cov = math.fsum((a - mx) * (b - my) for a, b in zip(tx, ty))
    corr = cov / math.sqrt(vx * vy)
    return max(-0.95, min(0.95, corr)) if math.isfinite(corr) else 0.0


@dataclass(frozen=True)
class RiskAllocatorConfig:
    """Immutable configuration for `CausalRiskAllocator`."""

    max_positions: int = 6
    target_annual_vol_pct: float = 18.0
    max_portfolio_leverage: float = 2.0
    max_single_weight: float = 1.0
    use_correlation_idm: bool = True
    corr_lookback_bars: int = 60
    corr_shrinkage: float = 0.25
    min_corr_obs: int = 10
    default_same_class_corr: float = 0.50
    min_asset_ann_vol_pct: float = 5.0

    def __post_init__(self) -> None:
        if int(self.max_positions) < 1:
            raise ValueError(f"max_positions must be >= 1, got {self.max_positions}")
        if not math.isfinite(self.target_annual_vol_pct) or self.target_annual_vol_pct <= 0.0:
            raise ValueError(f"target_annual_vol_pct must be > 0, got {self.target_annual_vol_pct}")
        if not math.isfinite(self.max_portfolio_leverage) or self.max_portfolio_leverage <= 0.0:
            raise ValueError(f"max_portfolio_leverage must be > 0, got {self.max_portfolio_leverage}")
        if not math.isfinite(self.max_single_weight) or self.max_single_weight <= 0.0:
            raise ValueError(f"max_single_weight must be > 0, got {self.max_single_weight}")
        if int(self.corr_lookback_bars) < 5:
            raise ValueError(f"corr_lookback_bars must be >= 5, got {self.corr_lookback_bars}")
        if not math.isfinite(self.corr_shrinkage) or not (0.0 <= self.corr_shrinkage <= 1.0):
            raise ValueError(f"corr_shrinkage must be in [0, 1], got {self.corr_shrinkage}")
        if int(self.min_corr_obs) < 3:
            raise ValueError(f"min_corr_obs must be >= 3, got {self.min_corr_obs}")
        if not math.isfinite(self.min_asset_ann_vol_pct) or self.min_asset_ann_vol_pct <= 0.0:
            raise ValueError(f"min_asset_ann_vol_pct must be > 0, got {self.min_asset_ann_vol_pct}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_bars_per_year(
    dec: BarDecision | None,
    spec: AssetSpec | None,
    timeframe_h: float | None,
    sym_timeframe_h: float | None = None,
) -> float:
    meta_bpy = dec.metadata.get("bars_per_year") if (dec is not None and dec.metadata) else None
    if meta_bpy is not None:
        val = float(meta_bpy)
        if math.isfinite(val) and val > 0.0:
            return val
    meta_tf = dec.metadata.get("timeframe_h") if (dec is not None and dec.metadata) else None
    eff_tf: float | None = None
    for cand in (meta_tf, sym_timeframe_h, timeframe_h):
        if cand is not None:
            cval = float(cand)
            if math.isfinite(cval) and cval > 0.0:
                eff_tf = cval
                break
    if spec is not None:
        return spec.calendar.bars_per_year(eff_tf if eff_tf is not None else 24.0)
    if eff_tf is not None:
        return (24.0 * 365.0) / eff_tf
    return 252.0


def _prior_ann_vol(spec: AssetSpec | None) -> float:
    if spec is not None and spec.asset_class in (ASSET_CLASS_CRYPTO_PERP, ASSET_CLASS_CRYPTO_SPOT):
        return 0.50
    return 0.16


class CausalRiskAllocator:
    """Stateless, 100% causal portfolio risk allocator shared identically by batch & streaming (G7)."""

    def __init__(self, cfg: RiskAllocatorConfig) -> None:
        self.cfg = cfg

    def _build_asset_vols_and_corr(
        self,
        syms: list[str],
        raw_decisions: dict[str, BarDecision],
        recent_returns: dict[str, Sequence[float]],
        specs: dict[str, AssetSpec],
        *,
        timeframe_h: float | None = None,
        timeframe_h_by_symbol: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], list[list[float]]]:
        """Compute per-asset annualized vols and shrunk correlation matrix over `syms`."""
        k = len(syms)
        min_vol = self.cfg.min_asset_ann_vol_pct / 100.0
        lookback = int(self.cfg.corr_lookback_bars)
        asset_vols: dict[str, float] = {}
        tails: dict[str, list[float]] = {}

        for sym in syms:
            spec = specs.get(sym)
            dec = raw_decisions.get(sym)
            sym_tf = timeframe_h_by_symbol.get(sym) if timeframe_h_by_symbol else None
            bpy = _resolve_bars_per_year(dec, spec, timeframe_h, sym_tf)
            seq = recent_returns.get(sym, ())
            seq_slice = seq[-lookback:] if isinstance(seq, (list, tuple)) else list(seq)[-lookback:]
            tail = [float(x) for x in seq_slice if math.isfinite(float(x))]
            tails[sym] = tail
            if len(tail) >= 5:
                sd = _fast_sample_std(tail)
                ann_v = sd * math.sqrt(max(1.0, bpy))
                asset_vols[sym] = max(min_vol, ann_v) if ann_v > 1e-9 else max(min_vol, _prior_ann_vol(spec))
            else:
                asset_vols[sym] = max(min_vol, _prior_ann_vol(spec))

        # Shrunk correlation matrix R_shrunk = (1 - lambda) * R_hat + lambda * I
        lam = float(self.cfg.corr_shrinkage)
        one_minus_lam = 1.0 - lam
        r_mat: list[list[float]] = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]

        if self.cfg.use_correlation_idm and k > 1:
            for i in range(k):
                si = syms[i]
                ti = tails[si]
                spec_i = specs.get(si)
                for j in range(i + 1, k):
                    sj = syms[j]
                    tj = tails[sj]
                    m = min(len(ti), len(tj), lookback)
                    if m >= self.cfg.min_corr_obs:
                        rho_hat = _fast_pearson_tail(ti, tj, m)
                    else:
                        spec_j = specs.get(sj)
                        same_cls = (
                            spec_i is not None
                            and spec_j is not None
                            and spec_i.asset_class == spec_j.asset_class
                        )
                        rho_hat = float(self.cfg.default_same_class_corr) if same_cls else 0.0
                    shrunk = one_minus_lam * rho_hat
                    r_mat[i][j] = shrunk
                    r_mat[j][i] = shrunk

        return asset_vols, r_mat

    def compute_portfolio_vol_and_idm(
        self,
        weights: dict[str, float],
        raw_decisions: dict[str, BarDecision],
        recent_returns: dict[str, Sequence[float]],
        specs: dict[str, AssetSpec],
        *,
        timeframe_h: float | None = None,
        locked_weights: dict[str, float] | None = None,
        timeframe_h_by_symbol: dict[str, float] | None = None,
    ) -> tuple[float, float, dict[str, float]]:
        """Compute causal ex-ante annualized portfolio vol, IDM, and per-asset ann vols."""
        combined_weights: dict[str, float] = {}
        if locked_weights:
            for s, w in locked_weights.items():
                fw = float(w)
                if s not in raw_decisions and math.isfinite(fw) and abs(fw) > 1e-9:
                    combined_weights[s] = fw
        for s, w in weights.items():
            fw = float(w)
            if math.isfinite(fw) and abs(fw) > 1e-9:
                combined_weights[s] = fw

        syms = sorted(combined_weights)
        k = len(syms)
        if k == 0:
            return 0.0, 1.0, {}

        asset_vols, r_mat = self._build_asset_vols_and_corr(
            syms,
            raw_decisions,
            recent_returns,
            specs,
            timeframe_h=timeframe_h,
            timeframe_h_by_symbol=timeframe_h_by_symbol,
        )

        # Ex-ante portfolio variance: w^T (D R D) w
        wv = [combined_weights[s] * asset_vols[s] for s in syms]
        port_var = 0.0
        if self.cfg.use_correlation_idm:
            for i in range(k):
                wi = wv[i]
                row = r_mat[i]
                port_var += wi * wi
                for j in range(i + 1, k):
                    port_var += 2.0 * wi * wv[j] * row[j]
        else:
            # Diagonal (uncorrelated) variance approximation when IDM is explicitly disabled
            for wi in wv:
                port_var += wi * wi

        sigma_p = math.sqrt(max(1e-12, port_var))
        sum_abs_risk = math.fsum(abs(x) for x in wv)
        idm = min(2.5, sum_abs_risk / sigma_p) if sigma_p > 1e-9 else 1.0
        return sigma_p, idm, asset_vols

    def allocate_with_audit(
        self,
        ts_ms: int,
        raw_decisions: dict[str, BarDecision],
        recent_returns: dict[str, Sequence[float]],
        specs: dict[str, AssetSpec],
        *,
        timeframe_h: float | None = None,
        locked_weights: dict[str, float] | None = None,
        timeframe_h_by_symbol: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], list[tuple[str, float, float, str]]]:
        """Return `(allocated_weights, suppression_events[(sym, req_w, eff_w, reason)])`."""
        _ = ts_ms  # Causal point-in-time timestamp kept for signature parity & auditing
        suppressions: list[tuple[str, float, float, str]] = []
        weights: dict[str, float] = {sym: 0.0 for sym in sorted(raw_decisions)}

        # Non-printing locked holdings (holidays / missing bars / omitted symbols) consume slots, vol & leverage
        non_printing_locked: dict[str, float] = {}
        if locked_weights:
            for sym, w in sorted(locked_weights.items()):
                fw = float(w)
                if sym not in raw_decisions and math.isfinite(fw) and abs(fw) > 1e-9:
                    non_printing_locked[sym] = fw

        # 1. Finite check & allow_short enforcement
        active_syms: list[str] = []
        for sym in sorted(raw_decisions):
            dec = raw_decisions[sym]
            req_w = float(dec.target_weight)
            if not math.isfinite(req_w):
                suppressions.append((sym, 0.0, 0.0, "NON_FINITE_WEIGHT"))
                weights[sym] = 0.0
                continue
            spec = specs.get(sym)
            if spec is not None and not spec.calendar.allow_short and req_w < -1e-12:
                suppressions.append((sym, req_w, 0.0, "SHORT_NOT_ALLOWED"))
                weights[sym] = 0.0
                continue
            weights[sym] = req_w
            if abs(req_w) > 1e-9:
                active_syms.append(sym)

        if not active_syms:
            return weights, suppressions

        # 2. Hard max_positions cap accounting for non-printing locked holdings
        remaining_slots = max(0, int(self.cfg.max_positions) - len(non_printing_locked))
        if len(active_syms) > remaining_slots:
            ranked = sorted(
                active_syms,
                key=lambda s: (-abs(float(raw_decisions[s].raw_signal)), -abs(weights[s]), s),
            )
            kept = set(ranked[:remaining_slots])
            for sym in ranked[remaining_slots:]:
                old_w = weights[sym]
                weights[sym] = 0.0
                suppressions.append((sym, old_w, 0.0, "MAX_POSITIONS_CAP"))
            active_syms = sorted(kept)

        if not active_syms:
            return weights, suppressions

        # 3. Single-symbol weight cap
        for sym in active_syms:
            spec = specs.get(sym)
            cap = float(self.cfg.max_single_weight)
            if spec is not None:
                cap = min(cap, float(spec.max_leverage_cap))
            old_w = weights[sym]
            if abs(old_w) > cap + 1e-12:
                new_w = math.copysign(cap, old_w)
                weights[sym] = new_w
                suppressions.append((sym, old_w, new_w, "WEIGHT_CAP"))

        # 4. Causal shrunk-correlation ex-ante portfolio volatility targeting (downward-only),
        #    solving exact quadratic s^2 * v_adj + 2 * s * c_adj_lock + v_lock = target_vol^2
        #    so ONLY adjustable printing weights are scaled when locked holdings exist!
        target_vol = float(self.cfg.target_annual_vol_pct) / 100.0
        target_var = target_vol * target_vol
        all_syms = sorted(set(active_syms) | set(non_printing_locked))
        asset_vols, r_mat = self._build_asset_vols_and_corr(
            all_syms,
            raw_decisions,
            recent_returns,
            specs,
            timeframe_h=timeframe_h,
            timeframe_h_by_symbol=timeframe_h_by_symbol,
        )
        k_all = len(all_syms)
        adj_rv = [weights[s] * asset_vols[s] if s not in non_printing_locked else 0.0 for s in all_syms]
        lock_rv = [non_printing_locked[s] * asset_vols[s] if s in non_printing_locked else 0.0 for s in all_syms]

        v_adj = 0.0
        v_lock = 0.0
        c_adj_lock = 0.0
        if self.cfg.use_correlation_idm:
            for i in range(k_all):
                ai = adj_rv[i]
                li = lock_rv[i]
                row = r_mat[i]
                v_adj += ai * ai
                v_lock += li * li
                c_adj_lock += ai * li
                for j in range(i + 1, k_all):
                    rij = row[j]
                    v_adj += 2.0 * ai * adj_rv[j] * rij
                    v_lock += 2.0 * li * lock_rv[j] * rij
                    c_adj_lock += (ai * lock_rv[j] + li * adj_rv[j]) * rij
        else:
            for i in range(k_all):
                v_adj += adj_rv[i] * adj_rv[i]
                v_lock += lock_rv[i] * lock_rv[i]

        v_adj = max(0.0, v_adj)
        v_lock = max(0.0, v_lock)
        total_var = max(0.0, v_adj + 2.0 * c_adj_lock + v_lock)
        sigma_p = math.sqrt(max(1e-12, total_var))

        if sigma_p > target_vol + 1e-12:
            if not non_printing_locked:
                vol_scale = target_vol / sigma_p
            elif v_lock >= target_var:
                if c_adj_lock >= 0.0:
                    vol_scale = 0.0
                else:
                    # Hedge reduces variance: clamp minimizer -c / v_adj to [0, 1]
                    vol_scale = max(0.0, min(1.0, (-c_adj_lock / v_adj) if v_adj > 1e-18 else 1.0))
            else:
                # v_lock < target_var < V(1): unique positive root in (0, 1)
                if v_adj > 1e-18:
                    disc = max(0.0, c_adj_lock * c_adj_lock - v_adj * (v_lock - target_var))
                    vol_scale = (-c_adj_lock + math.sqrt(disc)) / v_adj
                elif c_adj_lock > 1e-18:
                    vol_scale = (target_var - v_lock) / (2.0 * c_adj_lock)
                else:
                    vol_scale = 0.0
                vol_scale = max(0.0, min(1.0, vol_scale))

            if vol_scale < 1.0 - 1e-12:
                for sym in active_syms:
                    old_w = weights[sym]
                    new_w = old_w * vol_scale
                    weights[sym] = new_w
                    if abs(old_w - new_w) > 1e-6:
                        suppressions.append((sym, old_w, new_w, "VOL_TARGET_SCALE"))

        # 5. Hard portfolio gross leverage cap accounting for non-printing locked holdings
        locked_gross = math.fsum(abs(w) for w in non_printing_locked.values())
        remaining_lev = max(0.0, float(self.cfg.max_portfolio_leverage) - locked_gross)
        printing_gross = math.fsum(abs(weights[s]) for s in active_syms)
        if printing_gross > remaining_lev + 1e-12:
            lev_scale = (remaining_lev / printing_gross) if remaining_lev > 1e-12 else 0.0
            for sym in active_syms:
                old_w = weights[sym]
                new_w = old_w * lev_scale
                weights[sym] = new_w
                if abs(old_w - new_w) > 1e-6:
                    suppressions.append((sym, old_w, new_w, "LEVERAGE_SCALE"))

        return weights, suppressions

    def allocate(
        self,
        ts_ms: int,
        raw_decisions: dict[str, BarDecision],
        recent_returns: dict[str, Sequence[float]],
        specs: dict[str, AssetSpec],
        *,
        timeframe_h: float | None = None,
        locked_weights: dict[str, float] | None = None,
        timeframe_h_by_symbol: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute strictly causal portfolio weights for the active bar close."""
        weights, _ = self.allocate_with_audit(
            ts_ms,
            raw_decisions,
            recent_returns,
            specs,
            timeframe_h=timeframe_h,
            locked_weights=locked_weights,
            timeframe_h_by_symbol=timeframe_h_by_symbol,
        )
        return weights
