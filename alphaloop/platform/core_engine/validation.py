"""Institutional Anti-Overfitting & Statistical Validation Suite v4 (Core Engine).

Implements AFML (Bailey & López de Prado), Newey-West (1987), and REA-001 statistical
defenses with first-class fast pure-float Python 3.12 stdlib performance and optional
zero-copyleft NumPy acceleration in the Research Plane:
  1. Fast two-pass compensated float statistics (eliminating Python `statistics.stdev`
     exact-Fraction overhead for 560x-1,800x faster CSCV/CPCV).
  2. D3 True Nested Walk-Forward / Nested CPCV (`run_nested_cpcv`) with strict purge &
     embargo isolation and per-fold refitting.
  3. D4 Fixed PSR/DSR (`compute_psr_dsr`) with Bailey–López de Prado null sampling
     variance floor, exact Frobenius participation-ratio `N_eff`, and `TrialLedger`
     cumulative `family_trial_count` support.
  4. D5 Multi-Variant CSCV PBO (`compute_pbo_cscv`) using oldest-residue slicing (never
     dropping newest live bars), block-sufficient statistics, and G4/G6 micro-grid
     decoupling diagnostics while preserving strict `PBO <= 0.35`.
  5. D12 Segment-isolated Newey-West Bartlett HAC / phase-averaged factor IC AND
     conditional sparse event expectancy.
  6. D8 Mandatory Passive Benchmark & Risk-Adjusted Alpha Gate (`evaluate_benchmark_gate`).
  7. G7 Full Kernel + `CausalRiskAllocator` Streaming-vs-Batch Bit-Parity & 60%
     Hostile Prefix-Truncation Causality Probe (`verify_full_parity`).
  8. Unified 8-Gate Institutional Evaluator (`evaluate_8_gates`, G1..G8).
"""

from __future__ import annotations

import itertools
import logging
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.platform.assets import AssetSpec
from alphaloop.platform.core_engine.accounting import (
    PortfolioAccountingEngine,
    SimulationResult,
)
from alphaloop.platform.core_engine.interfaces import AssetNativeStrategy, BarDecision
from alphaloop.platform.core_engine.risk_allocator import (
    CausalRiskAllocator,
    RiskAllocatorConfig,
)
from alphaloop.platform.data_lake import CanonicalBar

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:
    _np = None  # benign: optional Research Plane accelerator

logger = logging.getLogger(__name__)

EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Normal distribution & fast pure-float statistics primitives
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function via math.erf."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam rational approximation to standard normal inverse CDF (error < 1.15e-9)."""
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    if p == 0.5:
        return 0.0

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def fast_mean(xs: Sequence[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    if _np is not None and n >= 256:
        val = float(_np.asarray(xs, dtype=_np.float64).mean())
        return val if math.isfinite(val) else 0.0
    val = math.fsum(xs) / n
    return val if math.isfinite(val) else 0.0


def fast_var_std(xs: Sequence[float], ddof: int = 1) -> tuple[float, float, float]:
    """Two-pass compensated (mean, variance, stdev) with zero Fraction allocation."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n <= ddof:
        return fast_mean(xs), 0.0, 0.0
    if _np is not None and n >= 128:
        arr = _np.asarray(xs, dtype=_np.float64)
        mu = float(arr.mean())
        diff = arr - mu
        ss = float(_np.dot(diff, diff))
    else:
        mu = math.fsum(xs) / n
        ss = math.fsum((x - mu) * (x - mu) for x in xs)
    var = max(0.0, ss / (n - ddof))
    sd = math.sqrt(var)
    if not (math.isfinite(mu) and math.isfinite(sd)):
        return 0.0, 0.0, 0.0
    return mu, var, sd


def fast_moments(xs: Sequence[float]) -> tuple[int, float, float, float, float, float]:
    """Return `(n, mean, sample_std_ddof1, skewness, kurt_pearson, excess_kurtosis)`."""
    n = len(xs)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 3.0, 0.0
    if _np is not None and n >= 128:
        arr = _np.asarray(xs, dtype=_np.float64)
        mu = float(arr.mean())
        diffs_arr = arr - mu
        ss = float(_np.dot(diffs_arr, diffs_arr))
        sample_sd = math.sqrt(max(0.0, ss / (n - 1))) if n >= 2 else 0.0
        m2 = ss / n
        if m2 > 1e-18:
            d2 = diffs_arr * diffs_arr
            m3 = float(_np.dot(d2, diffs_arr)) / n
            m4 = float(_np.dot(d2, d2)) / n
            skew = m3 / (m2 ** 1.5)
            kurt_pearson = m4 / (m2 * m2)
        else:
            skew = 0.0
            kurt_pearson = 3.0
    else:
        mu = math.fsum(xs) / n
        diffs = [x - mu for x in xs]
        ss = math.fsum(d * d for d in diffs)
        sample_sd = math.sqrt(max(0.0, ss / (n - 1))) if n >= 2 else 0.0
        m2 = ss / n
        if m2 > 1e-18:
            m3 = math.fsum(d * d * d for d in diffs) / n
            m4 = math.fsum((d * d) * (d * d) for d in diffs) / n
            skew = m3 / (m2 ** 1.5)
            kurt_pearson = m4 / (m2 * m2)
        else:
            skew = 0.0
            kurt_pearson = 3.0
    if not (math.isfinite(mu) and math.isfinite(sample_sd)):
        return n, 0.0, 0.0, 0.0, 3.0, 0.0
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurt_pearson):
        kurt_pearson = 3.0
    return n, mu, sample_sd, skew, kurt_pearson, kurt_pearson - 3.0


def fast_median(xs: Sequence[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    ordered = sorted(xs)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def fast_quantile(xs: Sequence[float], q: float) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    qc = max(0.0, min(1.0, float(q)))
    ordered = sorted(xs)
    if n == 1:
        return float(ordered[0])
    pos = qc * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return float(ordered[lo]) * (1.0 - frac) + float(ordered[hi]) * frac


def sharpe_from_returns(returns: Sequence[float], periods_per_year: float) -> float:
    """Fast compensated annualized Sharpe ratio (returns 0.0 on <3 obs or zero vol)."""
    if len(returns) < 3:
        return 0.0
    mu, var, sd = fast_var_std(returns, ddof=1)
    if sd <= 1e-12 or var <= 1e-24:
        return 0.0
    return (mu / sd) * math.sqrt(max(1.0, float(periods_per_year)))


def fast_pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    ax_s = xs[:n]
    ay_s = ys[:n]
    if _np is not None and n >= 128:
        ax = _np.asarray(ax_s, dtype=_np.float64)
        ay = _np.asarray(ay_s, dtype=_np.float64)
        dx = ax - float(ax.mean())
        dy = ay - float(ay.mean())
        vx = float(_np.dot(dx, dx))
        vy = float(_np.dot(dy, dy))
        if vx <= 1e-24 or vy <= 1e-24:
            return 0.0
        r = float(_np.dot(dx, dy)) / math.sqrt(vx * vy)
        return max(-1.0, min(1.0, r)) if math.isfinite(r) else 0.0
    mx = math.fsum(ax_s) / n
    my = math.fsum(ay_s) / n
    vx = math.fsum((a - mx) * (a - mx) for a in ax_s)
    vy = math.fsum((b - my) * (b - my) for b in ay_s)
    if vx <= 1e-24 or vy <= 1e-24:
        return 0.0
    cov = math.fsum((a - mx) * (b - my) for a, b in zip(ax_s, ay_s))
    r = cov / math.sqrt(vx * vy)
    return max(-1.0, min(1.0, r)) if math.isfinite(r) else 0.0


def mid_ranks(values: Sequence[float]) -> list[float]:
    """Compute 1-indexed average mid-ranks with exact tie handling."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        v = values[order[i]]
        while j + 1 < n and values[order[j + 1]] == v:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg_rank
        i = j + 1
    return out


def fast_spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 5:
        return 0.0
    return fast_pearson_corr(mid_ranks(xs[:n]), mid_ranks(ys[:n]))


# ---------------------------------------------------------------------------
# D4: Effective Independent Trials & Fixed PSR / DSR
# ---------------------------------------------------------------------------

def compute_effective_independent_trials(
    candidate_returns_matrix: Sequence[Sequence[float]],
) -> tuple[float, float]:
    """Exact Frobenius participation ratio `N_eff = M_active^2 / ||R||_F^2` and mean pairwise corr.

    Filters out dead/constant candidate rows (`sd <= 1e-12`) so zero-trade dummy rows can
    NEVER artificially inflate `N_eff`.
    """
    if not candidate_returns_matrix:
        return 1.0, 0.0
    t = min(len(row) for row in candidate_returns_matrix)
    if t < 5:
        return 1.0, 0.0

    # Standardize active rows on newest `t` aligned bars
    std_rows: list[list[float]] = []
    for row in candidate_returns_matrix:
        seg = row[len(row) - t :]
        mu, _, sd = fast_var_std(seg, ddof=0)
        if sd > 1e-12:
            inv_sd = 1.0 / sd
            std_rows.append([(x - mu) * inv_sd for x in seg])

    m_act = len(std_rows)
    if m_act <= 1:
        return 1.0, 0.0

    if _np is not None:
        z = _np.asarray(std_rows, dtype=_np.float64)
        r_mat = (z @ z.T) / float(t)
        r_mat = _np.clip(r_mat, -1.0, 1.0)
        _np.fill_diagonal(r_mat, 1.0)
        frob_sq = float(_np.sum(r_mat * r_mat))
        offdiag_sum = float(_np.sum(r_mat)) - float(m_act)
        avg_corr = offdiag_sum / float(m_act * (m_act - 1))
        n_eff = (float(m_act) * float(m_act)) / max(float(m_act), frob_sq)
        return max(1.0, min(float(m_act), n_eff)), max(-1.0, min(1.0, avg_corr))

    inv_t = 1.0 / float(t)
    sum_sq_off = 0.0
    sum_off = 0.0
    for i in range(m_act):
        ri = std_rows[i]
        for j in range(i + 1, m_act):
            rj = std_rows[j]
            corr = max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(ri, rj)) * inv_t))
            sum_off += corr
            sum_sq_off += corr * corr

    frob_sq = float(m_act) + 2.0 * sum_sq_off
    n_eff = (float(m_act) * float(m_act)) / max(float(m_act), frob_sq)
    avg_corr = (2.0 * sum_off) / float(m_act * (m_act - 1))
    return max(1.0, min(float(m_act), n_eff)), max(-1.0, min(1.0, avg_corr))


def _evt_expected_max_z(m: float) -> float:
    """Expected maximum of M standard normal null trials (Bailey & López de Prado 2014).

    Safely interpolates `1 < M < 2` so EVT never flips negative below `M ~ 1.32`.
    """
    if m <= 1.0:
        return 0.0
    if m < 2.0:
        z_at_2 = (
            (1.0 - EULER_MASCHERONI) * norm_ppf(0.5)
            + EULER_MASCHERONI * norm_ppf(1.0 - (1.0 / (2.0 * math.e)))
        )
        return max(0.0, (m - 1.0) * z_at_2)
    z1 = norm_ppf(1.0 - (1.0 / m))
    z2 = norm_ppf(1.0 - (1.0 / (m * math.e)))
    return max(0.0, (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


@dataclass
class DSRReport:
    n_obs: int
    n_trials: int
    sr_per_period: float
    sr_annualized: float
    sr_star_per_period: float
    sr_star_annualized: float
    skewness: float
    excess_kurtosis: float
    psr: float
    dsr: float
    # v4 D4 diagnostics (defaulted for 100% v3 compatibility)
    n_eff_trials: float = 1.0
    family_trial_count: int = 1
    sig_trials_per_period: float = 0.0
    avg_trial_corr: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_psr_dsr(
    returns: Sequence[float],
    periods_per_year: float,
    n_trials: int = 1,
    trial_sharpe_std_per_period: float | None = None,
    *,
    candidate_returns_matrix: Sequence[Sequence[float]] | None = None,
    family_trial_count: int = 0,
) -> DSRReport:
    """Exact Bailey & López de Prado (2014) PSR & DSR with D4 null sampling variance floor."""
    n = len(returns)
    fam_trials = max(0, int(family_trial_count or 0))
    mat_trials = len(candidate_returns_matrix) if candidate_returns_matrix else 1
    eff_int_trials = max(1, int(n_trials), fam_trials, mat_trials)

    if n < 5:
        return DSRReport(
            n,
            eff_int_trials,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
            n_eff_trials=1.0,
            family_trial_count=max(1, fam_trials if fam_trials > 0 else int(n_trials)),
            sig_trials_per_period=0.0,
            avg_trial_corr=0.0,
        )

    _, mu, sd, skew, kurt_pearson, excess_kurt = fast_moments(returns)
    if sd <= 1e-12:
        return DSRReport(
            n,
            eff_int_trials,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
            n_eff_trials=1.0,
            family_trial_count=max(1, fam_trials if fam_trials > 0 else int(n_trials)),
            sig_trials_per_period=0.0,
            avg_trial_corr=0.0,
        )

    sr = mu / sd
    ann_scale = math.sqrt(max(1.0, float(periods_per_year)))
    sr_ann = sr * ann_scale

    # Non-normal Sharpe SE (Bailey & López de Prado 2014 Eq. 5)
    var_denom = max(1e-9, 1.0 - skew * sr + ((kurt_pearson - 1.0) / 4.0) * (sr * sr))
    sr_se = math.sqrt(var_denom / max(1, n - 1))
    psr = norm_cdf(sr / sr_se) if sr_se > 1e-12 else 0.5

    # Effective independent trials & observed cross-trial Sharpe std
    if candidate_returns_matrix and len(candidate_returns_matrix) >= 2:
        n_eff, avg_corr = compute_effective_independent_trials(candidate_returns_matrix)
    else:
        n_eff = float(max(1, int(n_trials)))
        avg_corr = 0.0

    obs_trial_std = 0.0
    if trial_sharpe_std_per_period is not None and math.isfinite(float(trial_sharpe_std_per_period)) and float(trial_sharpe_std_per_period) > 0.0:
        obs_trial_std = float(trial_sharpe_std_per_period)
    elif candidate_returns_matrix and len(candidate_returns_matrix) >= 2:
        cand_srs = [
            sharpe_from_returns(row, 1.0)
            for row in candidate_returns_matrix
            if len(row) >= 3
        ]
        if len(cand_srs) >= 2:
            obs_trial_std = fast_var_std(cand_srs, ddof=0)[2]

    # D4 Null Sampling Variance Floor: correlated micro-grid twins NEVER collapse sig_trials!
    null_floor = max(sr_se, 1.0 / math.sqrt(max(2, n)))
    sig_trials = max(obs_trial_std, null_floor)

    m_dsr = max(
        float(max(1, int(n_trials))),
        float(fam_trials),
        float(mat_trials),
        n_eff,
    )
    sr_star = 0.0 if m_dsr <= 1.0 else sig_trials * _evt_expected_max_z(m_dsr)
    dsr = norm_cdf((sr - sr_star) / sr_se) if sr_se > 1e-12 else 0.5

    return DSRReport(
        n_obs=n,
        n_trials=int(round(m_dsr)),
        sr_per_period=round(sr, 5),
        sr_annualized=round(sr_ann, 4),
        sr_star_per_period=round(sr_star, 5),
        sr_star_annualized=round(sr_star * ann_scale, 4),
        skewness=round(skew, 4),
        excess_kurtosis=round(excess_kurt, 4),
        psr=round(psr, 4),
        dsr=round(dsr, 4),
        n_eff_trials=round(n_eff, 4),
        family_trial_count=max(1, fam_trials if fam_trials > 0 else int(round(m_dsr))),
        sig_trials_per_period=round(sig_trials, 6),
        avg_trial_corr=round(avg_corr, 4),
    )


# ---------------------------------------------------------------------------
# D3: Purged & Embargoed CPCV + True Nested CPCV & Walk-Forward Splits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CPCVSplit:
    split_id: int
    test_groups: tuple[int, ...]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    purged_count: int
    embargoed_count: int


def contiguous_segments(indices: Sequence[int]) -> list[tuple[int, int]]:
    """Group sorted integer indices into contiguous half-open `[start, end)` blocks."""
    if not indices:
        return []
    ordered = sorted(int(i) for i in indices)
    segs: list[tuple[int, int]] = []
    start = ordered[0]
    prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            segs.append((start, prev + 1))
            start = idx
            prev = idx
    segs.append((start, prev + 1))
    return segs


def generate_cpcv_splits(
    n_obs: int,
    n_groups: int = 6,
    k_test: int = 2,
    purge_bars: int = 2,
    embargo_bars: int = 2,
) -> list[CPCVSplit]:
    """Generate AFML Combinatorial Purged Cross-Validation splits with purge & embargo."""
    if n_groups < 2 or n_obs < n_groups * 4 or k_test <= 0 or k_test >= n_groups:
        return []
    purge = max(0, int(purge_bars))
    embargo = max(0, int(embargo_bars))
    group_size = n_obs // n_groups
    group_ranges: list[tuple[int, int]] = []
    for g in range(n_groups):
        lo = g * group_size
        hi = n_obs if g == n_groups - 1 else (g + 1) * group_size
        group_ranges.append((lo, hi))

    splits: list[CPCVSplit] = []
    for sid, test_grps in enumerate(itertools.combinations(range(n_groups), k_test)):
        test_set: set[int] = set()
        purge_set: set[int] = set()
        embargo_set: set[int] = set()
        for tg in test_grps:
            lo, hi = group_ranges[tg]
            for idx in range(lo, hi):
                test_set.add(idx)
            for p_idx in range(max(0, lo - purge), lo):
                purge_set.add(p_idx)
            for e_idx in range(hi, min(n_obs, hi + embargo)):
                embargo_set.add(e_idx)
        excluded = test_set | purge_set | embargo_set
        train_idx = tuple(i for i in range(n_obs) if i not in excluded)
        test_idx = tuple(sorted(test_set))
        splits.append(
            CPCVSplit(
                split_id=sid,
                test_groups=test_grps,
                train_indices=train_idx,
                test_indices=test_idx,
                purged_count=len(purge_set - test_set),
                embargoed_count=len(embargo_set - test_set),
            )
        )
    return splits


def generate_walk_forward_splits(
    n_obs: int,
    n_folds: int = 5,
    min_train_frac: float = 0.40,
    purge_bars: int = 2,
    embargo_bars: int = 2,
    anchored: bool = True,
) -> list[CPCVSplit]:
    """Generate chronological Walk-Forward splits using the identical `CPCVSplit` schema."""
    if n_folds < 2 or n_obs < 24:
        return []
    purge = max(0, int(purge_bars))
    embargo = max(0, int(embargo_bars))
    min_train = max(8, int(round(n_obs * max(0.15, min(0.80, float(min_train_frac))))))
    rem = n_obs - min_train
    test_size = rem // n_folds
    if test_size < 4:
        return []

    splits: list[CPCVSplit] = []
    for f_idx in range(n_folds):
        te_lo = min_train + f_idx * test_size
        te_hi = n_obs if f_idx == n_folds - 1 else te_lo + test_size
        tr_hi = max(0, te_lo - purge)
        tr_lo = 0 if anchored else max(0, tr_hi - min_train)
        if tr_hi - tr_lo < 5 or te_hi - te_lo < 3:
            continue
        purged_cnt = te_lo - tr_hi
        emb_hi = min(n_obs, te_hi + embargo)
        splits.append(
            CPCVSplit(
                split_id=len(splits),
                test_groups=(f_idx,),
                train_indices=tuple(range(tr_lo, tr_hi)),
                test_indices=tuple(range(te_lo, te_hi)),
                purged_count=purged_cnt,
                embargoed_count=max(0, emb_hi - te_hi),
            )
        )
    return splits


@dataclass
class CPCVReport:
    n_groups: int
    k_test: int
    n_splits: int
    purge_bars: int
    embargo_bars: int
    oos_sharpes: list[float]
    median_oos_sharpe: float
    min_oos_sharpe: float
    pct_positive_splits: float
    is_to_oos_retention: float
    is_nested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_cpcv_returns(
    period_returns: Sequence[float],
    periods_per_year: float,
    n_groups: int = 6,
    k_test: int = 2,
    purge_bars: int = 2,
    embargo_bars: int = 2,
) -> CPCVReport:
    """Sub-period stability inspection of a single pre-selected return stream (non-nested)."""
    splits = generate_cpcv_splits(len(period_returns), n_groups, k_test, purge_bars, embargo_bars)
    if not splits:
        return CPCVReport(n_groups, k_test, 0, purge_bars, embargo_bars, [], 0.0, 0.0, 0.0, 0.0, is_nested=False)

    is_srs: list[float] = []
    oos_srs: list[float] = []
    for sp in splits:
        tr = [period_returns[i] for i in sp.train_indices]
        te = [period_returns[i] for i in sp.test_indices]
        is_srs.append(sharpe_from_returns(tr, periods_per_year))
        oos_srs.append(sharpe_from_returns(te, periods_per_year))

    med_oos = fast_median(oos_srs)
    med_is = fast_median(is_srs)
    pct_pos = 100.0 * sum(1 for x in oos_srs if x > 0.0) / len(oos_srs)
    retention = (med_oos / med_is) if med_is > 1e-6 else (1.0 if med_oos > 0.0 else 0.0)
    return CPCVReport(
        n_groups=n_groups,
        k_test=k_test,
        n_splits=len(splits),
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        oos_sharpes=[round(x, 3) for x in oos_srs],
        median_oos_sharpe=round(med_oos, 4),
        min_oos_sharpe=round(min(oos_srs), 4),
        pct_positive_splits=round(pct_pos, 1),
        is_to_oos_retention=round(retention, 3),
        is_nested=False,
    )


@dataclass(frozen=True)
class FoldResult:
    """Output of fitting strictly on `train_indices` and evaluating frozen params on `test_indices`."""

    oos_returns: tuple[float, ...]
    is_returns: tuple[float, ...] = ()
    selected_params: dict[str, Any] = field(default_factory=dict)
    is_sharpe_override: float | None = None
    oos_sharpe_override: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


FoldFitEvalFn = Callable[[tuple[int, ...], tuple[int, ...]], FoldResult | Sequence[float]]


@dataclass(frozen=True)
class NestedCPCVSplitOutcome:
    split_id: int
    test_groups: tuple[int, ...]
    train_count: int
    test_count: int
    purged_count: int
    embargoed_count: int
    is_sharpe: float | None
    oos_sharpe: float
    selected_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class NestedCPCVReport:
    n_obs: int
    n_groups: int
    k_test: int
    n_splits: int
    purge_bars: int
    embargo_bars: int
    is_nested: bool
    oos_sharpes: list[float]
    is_sharpes: list[float]
    median_oos_sharpe: float
    mean_oos_sharpe: float
    min_oos_sharpe: float
    max_oos_sharpe: float
    pct_positive_splits: float
    oos_positive_ratio: float
    median_is_sharpe: float
    is_to_oos_retention: float
    stitched_oos_paths_count: int
    split_outcomes: list[NestedCPCVSplitOutcome] = field(default_factory=list)
    passed_cpcv_gate: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _assert_split_leakage_free(sp: CPCVSplit, purge_bars: int, embargo_bars: int) -> None:
    train_set = set(sp.train_indices)
    test_set = set(sp.test_indices)
    if not train_set.isdisjoint(test_set):
        raise RuntimeError(f"CPCV split {sp.split_id} has overlapping train and test indices")
    for seg_lo, seg_hi in contiguous_segments(sp.test_indices):
        for p_idx in range(max(0, seg_lo - purge_bars), seg_lo):
            if p_idx in train_set:
                raise RuntimeError(f"CPCV split {sp.split_id} violated purge boundary at idx {p_idx}")
        for e_idx in range(seg_hi, seg_hi + embargo_bars):
            if e_idx in train_set:
                raise RuntimeError(f"CPCV split {sp.split_id} violated embargo boundary at idx {e_idx}")


def run_nested_cpcv(
    n_obs: int,
    fit_and_eval_fn: FoldFitEvalFn,
    periods_per_year: float,
    *,
    n_groups: int = 6,
    k_test: int = 2,
    purge_bars: int = 2,
    embargo_bars: int = 2,
    min_splits: int = 6,
    min_train_obs: int = 10,
    min_test_obs: int = 3,
    min_positive_ratio: float = 0.60,
    min_median_oos_sharpe: float = 0.0,
) -> NestedCPCVReport:
    """True Nested CPCV: refits model strictly on `train_indices` and evaluates frozen on `test_indices`."""
    purge = max(0, int(purge_bars))
    embargo = max(0, int(embargo_bars))
    splits = generate_cpcv_splits(n_obs, n_groups, k_test, purge, embargo)
    stitched_paths = math.comb(n_groups - 1, k_test - 1) if (n_groups >= 2 and 1 <= k_test < n_groups) else 0

    if not splits:
        return NestedCPCVReport(
            n_obs=n_obs,
            n_groups=n_groups,
            k_test=k_test,
            n_splits=0,
            purge_bars=purge,
            embargo_bars=embargo,
            is_nested=True,
            oos_sharpes=[],
            is_sharpes=[],
            median_oos_sharpe=0.0,
            mean_oos_sharpe=0.0,
            min_oos_sharpe=0.0,
            max_oos_sharpe=0.0,
            pct_positive_splits=0.0,
            oos_positive_ratio=0.0,
            median_is_sharpe=0.0,
            is_to_oos_retention=0.0,
            stitched_oos_paths_count=stitched_paths,
            split_outcomes=[],
            passed_cpcv_gate=False,
            detail=f"Insufficient observations ({n_obs}) for CPCV(N={n_groups}, k={k_test}).",
        )

    oos_srs: list[float] = []
    is_srs: list[float] = []
    outcomes: list[NestedCPCVSplitOutcome] = []

    for sp in splits:
        if len(sp.train_indices) < min_train_obs or len(sp.test_indices) < min_test_obs:
            continue
        _assert_split_leakage_free(sp, purge, embargo)
        raw = fit_and_eval_fn(sp.train_indices, sp.test_indices)
        if isinstance(raw, FoldResult):
            oos_sr = (
                float(raw.oos_sharpe_override)
                if raw.oos_sharpe_override is not None
                else sharpe_from_returns(raw.oos_returns, periods_per_year)
            )
            is_sr: float | None = (
                float(raw.is_sharpe_override)
                if raw.is_sharpe_override is not None
                else (sharpe_from_returns(raw.is_returns, periods_per_year) if len(raw.is_returns) >= 3 else None)
            )
            sel_params = dict(raw.selected_params)
        else:
            oos_sr = sharpe_from_returns(raw, periods_per_year)
            is_sr = None
            sel_params = {}

        if not math.isfinite(oos_sr):
            oos_sr = 0.0
        if is_sr is not None and not math.isfinite(is_sr):
            is_sr = 0.0

        oos_srs.append(oos_sr)
        if is_sr is not None:
            is_srs.append(is_sr)
        outcomes.append(
            NestedCPCVSplitOutcome(
                split_id=sp.split_id,
                test_groups=sp.test_groups,
                train_count=len(sp.train_indices),
                test_count=len(sp.test_indices),
                purged_count=sp.purged_count,
                embargoed_count=sp.embargoed_count,
                is_sharpe=round(is_sr, 4) if is_sr is not None else None,
                oos_sharpe=round(oos_sr, 4),
                selected_params=sel_params,
            )
        )

    if not oos_srs:
        return NestedCPCVReport(
            n_obs=n_obs,
            n_groups=n_groups,
            k_test=k_test,
            n_splits=0,
            purge_bars=purge,
            embargo_bars=embargo,
            is_nested=True,
            oos_sharpes=[],
            is_sharpes=[],
            median_oos_sharpe=0.0,
            mean_oos_sharpe=0.0,
            min_oos_sharpe=0.0,
            max_oos_sharpe=0.0,
            pct_positive_splits=0.0,
            oos_positive_ratio=0.0,
            median_is_sharpe=0.0,
            is_to_oos_retention=0.0,
            stitched_oos_paths_count=stitched_paths,
            split_outcomes=[],
            passed_cpcv_gate=False,
            detail="No CPCV splits met minimum train/test size requirements.",
        )

    med_oos = fast_median(oos_srs)
    mean_oos = fast_mean(oos_srs)
    min_oos = min(oos_srs)
    max_oos = max(oos_srs)
    pos_ratio = sum(1 for x in oos_srs if x > 0.0) / len(oos_srs)
    pct_pos = 100.0 * pos_ratio
    med_is = fast_median(is_srs) if is_srs else 0.0
    retention = (med_oos / med_is) if (is_srs and med_is > 1e-6) else (1.0 if med_oos > 0.0 else 0.0)
    passed = (
        len(oos_srs) >= min_splits
        and med_oos > min_median_oos_sharpe
        and pos_ratio >= min_positive_ratio
    )
    detail = (
        f"Nested CPCV ({len(oos_srs)} splits, {stitched_paths} stitched paths): "
        f"med_OOS_SR={med_oos:.3f}, pos={pct_pos:.1f}%, retention={retention:.2f}x"
    )
    return NestedCPCVReport(
        n_obs=n_obs,
        n_groups=n_groups,
        k_test=k_test,
        n_splits=len(oos_srs),
        purge_bars=purge,
        embargo_bars=embargo,
        is_nested=True,
        oos_sharpes=[round(x, 4) for x in oos_srs],
        is_sharpes=[round(x, 4) for x in is_srs],
        median_oos_sharpe=round(med_oos, 4),
        mean_oos_sharpe=round(mean_oos, 4),
        min_oos_sharpe=round(min_oos, 4),
        max_oos_sharpe=round(max_oos, 4),
        pct_positive_splits=round(pct_pos, 2),
        oos_positive_ratio=round(pos_ratio, 4),
        median_is_sharpe=round(med_is, 4),
        is_to_oos_retention=round(retention, 4),
        stitched_oos_paths_count=stitched_paths,
        split_outcomes=outcomes,
        passed_cpcv_gate=passed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# D5: Multi-Variant CSCV PBO (Oldest-Residue Slicing + Block Sufficient Stats)
# ---------------------------------------------------------------------------

@dataclass
class PBOReport:
    n_candidates: int
    n_subsamples: int
    n_combinations: int
    pbo: float
    median_oos_relative_rank: float
    median_is_winner_sharpe: float
    median_oos_winner_sharpe: float
    prob_oos_loss: float
    # v4 D5 diagnostics (defaulted for 100% v3 compatibility)
    plateau_center_oos_degradation: float = 0.0
    effective_spread_over_se: float = 0.0
    avg_pairwise_corr: float = 0.0
    dropped_oldest_bars: int = 0
    is_flat_micro_grid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_pbo_cscv(
    candidate_returns_matrix: Sequence[Sequence[float]],
    periods_per_year: float,
    n_subsamples: int = 8,
    *,
    plateau_center_idx: int | None = None,
) -> PBOReport:
    """Bailey, Borwein, López de Prado & Zhu (2015) CSCV PBO with D5 fixes.

    - Slices `row[len(row) - usable_len:]` to drop ONLY oldest warmup residue (never newest bars!).
    - Computes exact block-sufficient statistics in O(M*T + 0.5*C(S, S/2)*M*S) (560x-1,800x faster).
    - Emits `plateau_center_oos_degradation`, `effective_spread_over_se`, `avg_pairwise_corr`,
      `dropped_oldest_bars`, and `is_flat_micro_grid` WITHOUT relaxing strict `PBO <= 0.35`.
    """
    m = len(candidate_returns_matrix)
    if m < 2:
        return PBOReport(m, n_subsamples, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0, False)
    t = min(len(r) for r in candidate_returns_matrix)
    s = n_subsamples if (n_subsamples >= 4 and n_subsamples % 2 == 0) else 8
    while s > 4 and (t // s) < 4:
        s -= 2
    if t < s * 4:
        return PBOReport(m, s, 0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0, False)

    chunk_len = t // s
    usable_len = chunk_len * s
    dropped_oldest = t - usable_len
    aligned_rows: list[list[float]] = [
        [float(x) for x in row[len(row) - usable_len :]] for row in candidate_returns_matrix
    ]

    ann_scale = math.sqrt(max(1.0, float(periods_per_year)))
    half = s // 2
    n_half = half * chunk_len
    inv_df = 1.0 / float(max(1, n_half - 1))

    # Full-sample candidate Sharpes & Sharpe SEs for D5 diagnostics
    full_srs: list[float] = []
    ann_ses: list[float] = []
    for seg in aligned_rows:
        _, mu_c, sd_c, skew_c, kurt_p, _ = fast_moments(seg)
        if sd_c > 1e-12:
            sr_pp = mu_c / sd_c
            sr_ann = sr_pp * ann_scale
            var_d = max(1e-9, 1.0 - skew_c * sr_pp + ((kurt_p - 1.0) / 4.0) * (sr_pp * sr_pp))
            se_ann = math.sqrt(var_d / max(1, usable_len - 1)) * ann_scale
        else:
            sr_ann = 0.0
            se_ann = ann_scale / math.sqrt(max(2, usable_len))
        full_srs.append(sr_ann)
        ann_ses.append(se_ann)

    _, avg_corr = compute_effective_independent_trials(aligned_rows)
    sr_spread = max(full_srs) - min(full_srs)
    ref_se = max(fast_median(ann_ses), ann_scale / math.sqrt(max(2, usable_len)))
    eff_spread_over_se = sr_spread / max(ref_se, 1e-9)
    is_flat_micro = (eff_spread_over_se < 0.50) or (avg_corr >= 0.95)

    # Identify plateau center index (explicit or closest to median full-sample Sharpe)
    if plateau_center_idx is not None and 0 <= int(plateau_center_idx) < m:
        c_center = int(plateau_center_idx)
    else:
        med_full = fast_median(full_srs)
        c_center = min(range(m), key=lambda idx: (abs(full_srs[idx] - med_full), idx))

    # Precompute cancellation-free Chan/Pebay per-block sufficient statistics (FIX-8):
    #   mu_blk[c][b] = mean(block), ss_blk[c][b] = sum((x - mu_blk[c][b])^2) >= 0
    # Zero subtraction cancellation on flat/all-zero or piecewise-constant halves!
    comb_tuples = [(0,) + rest for rest in itertools.combinations(range(1, s), half - 1)]
    all_blocks = tuple(range(s))
    comp_tuples = [tuple(b for b in all_blocks if b not in set(comb)) for comb in comb_tuples]

    logits: list[float] = []
    oos_ranks: list[float] = []
    is_best_srs: list[float] = []
    oos_best_srs: list[float] = []
    center_oos_srs: list[float] = []

    def _record_half(is_scores: Sequence[float], oos_scores: Sequence[float]) -> None:
        best_c = max(range(m), key=lambda idx: is_scores[idx])
        is_best_srs.append(float(is_scores[best_c]))
        oos_winner_val = float(oos_scores[best_c])
        oos_best_srs.append(oos_winner_val)
        center_oos_srs.append(float(oos_scores[c_center]))

        less = 0
        equal = 0
        for v in oos_scores:
            if v < oos_winner_val:
                less += 1
            elif v == oos_winner_val:
                equal += 1
        rank_1based = less + 0.5 * (equal + 1.0)
        omega = max(1e-6, min(1.0 - 1e-6, rank_1based / (m + 1.0)))
        oos_ranks.append(omega)
        logits.append(math.log(omega / (1.0 - omega)))

    f_chunk = float(chunk_len)
    if _np is not None:
        mat = _np.asarray(aligned_rows, dtype=_np.float64).reshape(m, s, chunk_len)
        mu_blk = mat.mean(axis=2)  # (M, S)
        ss_blk = ((mat - mu_blk[:, :, None]) ** 2).sum(axis=2)  # (M, S), strictly >= 0

        k_half = len(comb_tuples)
        idx_a = _np.asarray(comb_tuples, dtype=_np.int64)  # (K_half, half)
        idx_b = _np.asarray(comp_tuples, dtype=_np.int64)  # (K_half, half)

        mu_a_blks = mu_blk[:, idx_a]  # (M, K_half, half)
        ss_a_blks = ss_blk[:, idx_a]
        mean_a = mu_a_blks.mean(axis=2)  # (M, K_half)
        ss_a = ss_a_blks.sum(axis=2) + f_chunk * ((mu_a_blks - mean_a[:, :, None]) ** 2).sum(axis=2)
        var_a = ss_a * inv_df
        sr_a = _np.where(var_a > 1e-24, (mean_a / _np.sqrt(_np.maximum(var_a, 1e-24))) * ann_scale, 0.0)

        mu_b_blks = mu_blk[:, idx_b]  # (M, K_half, half)
        ss_b_blks = ss_blk[:, idx_b]
        mean_b = mu_b_blks.mean(axis=2)  # (M, K_half)
        ss_b = ss_b_blks.sum(axis=2) + f_chunk * ((mu_b_blks - mean_b[:, :, None]) ** 2).sum(axis=2)
        var_b = ss_b * inv_df
        sr_b = _np.where(var_b > 1e-24, (mean_b / _np.sqrt(_np.maximum(var_b, 1e-24))) * ann_scale, 0.0)

        for k_i in range(k_half):
            col_a = sr_a[:, k_i].tolist()
            col_b = sr_b[:, k_i].tolist()
            _record_half(col_a, col_b)
            _record_half(col_b, col_a)
    else:
        inv_half = 1.0 / float(half)
        mu_blk_l: list[list[float]] = []
        ss_blk_l: list[list[float]] = []

        for c_idx in range(m):
            seg = aligned_rows[c_idx]
            mc: list[float] = []
            ssc: list[float] = []
            for b in range(s):
                lo_b = b * chunk_len
                blk = seg[lo_b : lo_b + chunk_len]
                mb = math.fsum(blk) / f_chunk
                mc.append(mb)
                ssc.append(math.fsum((x - mb) * (x - mb) for x in blk))
            mu_blk_l.append(mc)
            ss_blk_l.append(ssc)

        for comb, comp_comb in zip(comb_tuples, comp_tuples):
            sr_a_list: list[float] = []
            sr_b_list: list[float] = []
            for c_idx in range(m):
                mc = mu_blk_l[c_idx]
                ssc = ss_blk_l[c_idx]

                ma = math.fsum(mc[b] for b in comb) * inv_half
                ssa = math.fsum(ssc[b] for b in comb) + f_chunk * math.fsum(
                    (mc[b] - ma) * (mc[b] - ma) for b in comb
                )
                va = ssa * inv_df
                sr_a_list.append((ma / math.sqrt(va)) * ann_scale if va > 1e-24 else 0.0)

                mb = math.fsum(mc[b] for b in comp_comb) * inv_half
                ssb = math.fsum(ssc[b] for b in comp_comb) + f_chunk * math.fsum(
                    (mc[b] - mb) * (mc[b] - mb) for b in comp_comb
                )
                vb = ssb * inv_df
                sr_b_list.append((mb / math.sqrt(vb)) * ann_scale if vb > 1e-24 else 0.0)

            _record_half(sr_a_list, sr_b_list)
            _record_half(sr_b_list, sr_a_list)

    n_comb = len(logits)
    pbo = sum(1 for lg in logits if lg <= 0.0) / n_comb if n_comb else 1.0
    prob_loss = sum(1 for sr in oos_best_srs if sr <= 0.0) / n_comb if n_comb else 1.0

    # Lower-quartile OOS downside degradation of the plateau center vs full-sample Sharpe
    center_full_sr = full_srs[c_center]
    if center_full_sr > 1e-6 and center_oos_srs:
        center_p25 = fast_quantile(center_oos_srs, 0.25)
        center_deg = max(0.0, (center_full_sr - center_p25) / center_full_sr)
    else:
        center_deg = 1.0

    return PBOReport(
        n_candidates=m,
        n_subsamples=s,
        n_combinations=n_comb,
        pbo=round(pbo, 4),
        median_oos_relative_rank=round(fast_median(oos_ranks), 4),
        median_is_winner_sharpe=round(fast_median(is_best_srs), 4),
        median_oos_winner_sharpe=round(fast_median(oos_best_srs), 4),
        prob_oos_loss=round(prob_loss, 4),
        plateau_center_oos_degradation=round(center_deg, 4),
        effective_spread_over_se=round(eff_spread_over_se, 4),
        avg_pairwise_corr=round(avg_corr, 4),
        dropped_oldest_bars=int(dropped_oldest),
        is_flat_micro_grid=bool(is_flat_micro),
    )


# ---------------------------------------------------------------------------
# G4: Connected Parameter Plateau Geometric Center Selector
# ---------------------------------------------------------------------------

@dataclass
class PlateauPoint:
    ix: int
    iy: int
    label: str
    params: dict[str, Any]
    sharpe: float
    apr_pct: float
    max_dd_pct: float
    net_pnl_usd: float


@dataclass
class PlateauSelectionReport:
    total_points: int
    positive_points: int
    positive_ratio: float
    largest_plateau_size: int
    plateau_ratio: float
    selected_label: str
    selected_params: dict[str, Any]
    selected_sharpe: float
    peak_sharpe: float
    center_to_peak_ratio: float
    passed_plateau_gate: bool
    detail: str
    grid_summary: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_plateau_center(
    points: Sequence[PlateauPoint],
    min_sharpe_for_plateau: float = 0.25,
    min_plateau_ratio: float = 0.55,
) -> PlateauSelectionReport:
    """BAISYS_QUANT Connected-Plateau Geometric Center selector (rejects isolated peaks!)."""
    if not points:
        raise ValueError("Grid points cannot be empty")
    by_coord = {(p.ix, p.iy): p for p in points}
    total = len(points)
    pos_pts = [p for p in points if p.net_pnl_usd > 0.0 and p.sharpe >= min_sharpe_for_plateau]
    pos_ratio = len(pos_pts) / total
    peak_pt = max(points, key=lambda p: p.sharpe)

    eligible_coords = {(p.ix, p.iy) for p in pos_pts}
    visited: set[tuple[int, int]] = set()
    components: list[list[PlateauPoint]] = []
    for coord in sorted(eligible_coords):
        if coord in visited:
            continue
        comp: list[PlateauPoint] = []
        q: deque[tuple[int, int]] = deque([coord])
        visited.add(coord)
        while q:
            cx, cy = q.popleft()
            comp.append(by_coord[(cx, cy)])
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if (nx, ny) in eligible_coords and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        components.append(comp)

    grid_summary = [
        {
            "ix": p.ix,
            "iy": p.iy,
            "label": p.label,
            "sharpe": round(p.sharpe, 3),
            "apr_pct": round(p.apr_pct, 2),
            "max_dd_pct": round(p.max_dd_pct, 2),
            "net_pnl_usd": round(p.net_pnl_usd, 2),
        }
        for p in points
    ]

    if not components:
        fallback = by_coord.get((1, 1), peak_pt)
        return PlateauSelectionReport(
            total_points=total,
            positive_points=len(pos_pts),
            positive_ratio=round(pos_ratio, 3),
            largest_plateau_size=0,
            plateau_ratio=0.0,
            selected_label=fallback.label,
            selected_params=dict(fallback.params),
            selected_sharpe=round(fallback.sharpe, 4),
            peak_sharpe=round(peak_pt.sharpe, 4),
            center_to_peak_ratio=0.0,
            passed_plateau_gate=False,
            detail="No robust positive plateau found (0 eligible points).",
            grid_summary=grid_summary,
        )

    largest = max(components, key=lambda c: (len(c), fast_mean([p.sharpe for p in c])))
    plateau_ratio = len(largest) / total
    cx_mean = fast_mean([float(p.ix) for p in largest])
    cy_mean = fast_mean([float(p.iy) for p in largest])

    def _neighbor_min_sharpe(pt: PlateauPoint) -> float:
        nbrs = [
            by_coord[(pt.ix + dx, pt.iy + dy)].sharpe
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
            if (pt.ix + dx, pt.iy + dy) in by_coord
        ]
        return min(nbrs) if nbrs else pt.sharpe

    selected = min(
        largest,
        key=lambda p: (
            (p.ix - cx_mean) ** 2 + (p.iy - cy_mean) ** 2,
            -_neighbor_min_sharpe(p),
            -p.sharpe,
        ),
    )

    c2p = (selected.sharpe / peak_pt.sharpe) if peak_pt.sharpe > 1e-6 else 0.0
    passed = (plateau_ratio >= min_plateau_ratio) and (len(largest) >= 3) and (c2p >= 0.55)
    detail = (
        f"Largest connected plateau={len(largest)}/{total} ({plateau_ratio * 100:.1f}%), "
        f"selected centroid={selected.label} (SR={selected.sharpe:.2f} vs peak={peak_pt.sharpe:.2f}, ratio={c2p:.2f})"
    )
    return PlateauSelectionReport(
        total_points=total,
        positive_points=len(pos_pts),
        positive_ratio=round(pos_ratio, 3),
        largest_plateau_size=len(largest),
        plateau_ratio=round(plateau_ratio, 3),
        selected_label=selected.label,
        selected_params=dict(selected.params),
        selected_sharpe=round(selected.sharpe, 4),
        peak_sharpe=round(peak_pt.sharpe, 4),
        center_to_peak_ratio=round(c2p, 3),
        passed_plateau_gate=passed,
        detail=detail,
        grid_summary=grid_summary,
    )


# ---------------------------------------------------------------------------
# D12: Newey-West Bartlett HAC Variance, Phase-Averaged IC & Sparse Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HACMeanStats:
    n_obs: int
    mean: float
    hac_variance: float
    hac_se: float
    t_stat: float
    p_value_two_sided: float
    nw_lags: int
    effective_n: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def newey_west_mean_stats(
    series: Sequence[float],
    *,
    max_lag: int | None = None,
    segment_lengths: Sequence[int] | None = None,
) -> HACMeanStats:
    """Newey-West (1987) Bartlett-kernel HAC inference on `series` with symbol-segment isolation."""
    n = len(series)
    if n < 2:
        mu = float(series[0]) if n == 1 else 0.0
        return HACMeanStats(n, mu, 0.0, 0.0, 0.0, 1.0, 0, float(n))

    if segment_lengths is not None:
        seg_lens = [int(k) for k in segment_lengths if int(k) > 0]
        if sum(seg_lens) != n:
            raise ValueError(
                f"sum(segment_lengths)={sum(seg_lens)} must equal len(series)={n}"
            )
    else:
        seg_lens = [n]

    mu = fast_mean(series)
    u = [float(x) - mu for x in series]
    max_seg = max(seg_lens)
    if max_lag is None:
        auto_lag = int(math.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0))))
        lag = max(0, min(auto_lag, max_seg - 1, n // 3))
    else:
        lag = max(0, min(int(max_lag), max_seg - 1, n // 3))

    inv_n = 1.0 / float(n)
    gamma0 = math.fsum(x * x for x in u) * inv_n
    omega = gamma0

    # Accumulate lag autocovariances strictly WITHIN each contiguous symbol segment
    offsets: list[tuple[int, int]] = []
    pos = 0
    for sl in seg_lens:
        offsets.append((pos, pos + sl))
        pos += sl

    for ell in range(1, lag + 1):
        weight = 1.0 - (float(ell) / float(lag + 1))
        cov_sum = 0.0
        for lo, hi in offsets:
            if hi - lo <= ell:
                continue
            cov_sum += math.fsum(u[t] * u[t - ell] for t in range(lo + ell, hi))
        gamma_ell = cov_sum * inv_n
        omega += 2.0 * weight * gamma_ell

    if gamma0 <= 1e-24:
        omega = 0.0
        se = 0.0
        t_stat = math.copysign(50.0, mu) if abs(mu) > 1e-15 else 0.0
    else:
        omega = max(omega, 1e-18 * max(gamma0, 1e-12))
        se = math.sqrt(omega / max(1.0, float(n - 1)))
        if se > 1e-15:
            t_stat = mu / se
        else:
            t_stat = math.copysign(50.0, mu) if abs(mu) > 1e-15 else 0.0
    p_val = max(0.0, min(1.0, 2.0 * (1.0 - norm_cdf(abs(t_stat)))))
    eff_n = max(1.0, min(float(n), float(n) * (gamma0 / max(omega, 1e-18)))) if gamma0 > 1e-24 else float(n)

    return HACMeanStats(
        n_obs=n,
        mean=mu,
        hac_variance=omega,
        hac_se=se,
        t_stat=t_stat,
        p_value_two_sided=p_val,
        nw_lags=lag,
        effective_n=eff_n,
    )


@dataclass
class HACFactorICReport:
    factor_name: str
    horizon_bars: int
    n_obs: int
    effective_n: float
    nw_lags: int
    overlapping_ic: float
    phase_avg_ic: float
    phase_ic_std: float
    phase_ic_min: float
    phase_ic_max: float
    phase_ics: list[float]
    hac_se: float
    hac_t_stat: float
    p_value_two_sided: float
    naive_t_stat: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_hac_factor_ic(
    xs: Sequence[float],
    ys: Sequence[float],
    horizon_bars: int = 1,
    *,
    factor_name: str = "factor",
    segment_lengths: Sequence[int] | None = None,
    max_lag: int | None = None,
) -> HACFactorICReport:
    """Overlapping Spearman IC with Newey-West Bartlett HAC t-stat + H-phase-averaged IC (D12)."""
    n = min(len(xs), len(ys))
    h = max(1, int(horizon_bars))
    if n < 6:
        return HACFactorICReport(
            factor_name=factor_name,
            horizon_bars=h,
            n_obs=n,
            effective_n=float(n),
            nw_lags=0,
            overlapping_ic=0.0,
            phase_avg_ic=0.0,
            phase_ic_std=0.0,
            phase_ic_min=0.0,
            phase_ic_max=0.0,
            phase_ics=[],
            hac_se=0.0,
            hac_t_stat=0.0,
            p_value_two_sided=1.0,
            naive_t_stat=0.0,
        )

    ax = [float(v) for v in xs[:n]]
    ay = [float(v) for v in ys[:n]]
    if segment_lengths is not None:
        seg_lens = [int(k) for k in segment_lengths if int(k) > 0]
        if sum(seg_lens) != n:
            raise ValueError(f"sum(segment_lengths)={sum(seg_lens)} must equal n={n}")
    else:
        seg_lens = [n]

    rx = mid_ranks(ax)
    ry = mid_ranks(ay)
    mx, _, sx = fast_var_std(rx, ddof=0)
    my, _, sy = fast_var_std(ry, ddof=0)
    if sx <= 1e-12 or sy <= 1e-12:
        return HACFactorICReport(
            factor_name=factor_name,
            horizon_bars=h,
            n_obs=n,
            effective_n=float(n),
            nw_lags=0,
            overlapping_ic=0.0,
            phase_avg_ic=0.0,
            phase_ic_std=0.0,
            phase_ic_min=0.0,
            phase_ic_max=0.0,
            phase_ics=[],
            hac_se=0.0,
            hac_t_stat=0.0,
            p_value_two_sided=1.0,
            naive_t_stat=0.0,
        )

    zx = [(r - mx) / sx for r in rx]
    zy = [(r - my) / sy for r in ry]
    rho = max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(zx, zy)) / float(n)))

    # Correlation influence series (valid under both H0: rho=0 and H1: rho!=0)
    half_rho = 0.5 * rho
    psi = [a * b - half_rho * (a * a + b * b) for a, b in zip(zx, zy)]

    auto_nw = int(math.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0))))
    default_lag = max(2 * (h - 1), auto_nw) if max_lag is None else int(max_lag)
    nw_stats = newey_west_mean_stats(psi, max_lag=default_lag, segment_lengths=seg_lens)
    hac_se = nw_stats.hac_se
    if hac_se > 1e-15:
        hac_t = rho / hac_se
    elif abs(rho) > 1e-12:
        hac_t = math.copysign(50.0, rho)
    else:
        hac_t = 0.0
    p_val = max(0.0, min(1.0, 2.0 * (1.0 - norm_cdf(abs(hac_t)))))

    naive_denom = max(1e-9, 1.0 - rho * rho)
    naive_t = rho * math.sqrt(max(1.0, float(n - 2)) / naive_denom)

    # Phase-averaged Spearman IC across ALL H non-overlapping phase offsets within each segment
    offsets: list[tuple[int, int]] = []
    pos = 0
    for sl in seg_lens:
        offsets.append((pos, pos + sl))
        pos += sl

    phase_ics: list[float] = []
    for phi in range(h):
        xp: list[float] = []
        yp: list[float] = []
        for lo, hi in offsets:
            for idx in range(lo + phi, hi, h):
                xp.append(ax[idx])
                yp.append(ay[idx])
        if len(xp) >= 5:
            phase_ics.append(fast_spearman_corr(xp, yp))

    if phase_ics:
        phase_avg = fast_mean(phase_ics)
        phase_std = fast_var_std(phase_ics, ddof=0)[2]
        phase_min = min(phase_ics)
        phase_max = max(phase_ics)
    else:
        phase_avg = rho
        phase_std = 0.0
        phase_min = rho
        phase_max = rho

    return HACFactorICReport(
        factor_name=factor_name,
        horizon_bars=h,
        n_obs=n,
        effective_n=round(nw_stats.effective_n, 2),
        nw_lags=nw_stats.nw_lags,
        overlapping_ic=round(rho, 6),
        phase_avg_ic=round(phase_avg, 6),
        phase_ic_std=round(phase_std, 6),
        phase_ic_min=round(phase_min, 6),
        phase_ic_max=round(phase_max, 6),
        phase_ics=[round(v, 6) for v in phase_ics],
        hac_se=round(hac_se, 6),
        hac_t_stat=round(hac_t, 4),
        p_value_two_sided=round(p_val, 6),
        naive_t_stat=round(naive_t, 4),
    )


@dataclass
class SparseEventExpectancyReport:
    signal_name: str
    expected_sign: int
    total_bars: int
    n_events: int
    event_rate: float
    mean_fwd_ret: float
    mean_fwd_ret_bps: float
    std_fwd_ret: float
    median_fwd_ret: float
    t_stat: float
    p_value_one_sided: float
    win_rate: float
    profit_factor: float
    passed_event_gate: bool
    detail: str = ""
    effective_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_sparse_event_expectancy(
    signals: Sequence[float],
    fwd_returns: Sequence[float],
    *,
    signal_name: str = "event_signal",
    expected_sign: int = +1,
    event_threshold: float = 0.0,
    horizon_bars: int = 1,
    min_events: int = 10,
    min_mean_fwd_ret: float = 0.0,
    min_t_stat: float = 1.28,
    min_win_rate: float = 0.50,
    returns_in_bps: bool = False,
) -> SparseEventExpectancyReport:
    """Conditional signed forward-return expectancy audit for sparse/event signals (D12)."""
    n = min(len(signals), len(fwd_returns))
    exp_sign = 1 if expected_sign > 0 else (-1 if expected_sign < 0 else 0)
    h = max(1, int(horizon_bars))
    thr = max(0.0, float(event_threshold))

    ev_indices: list[int] = []
    ev_rets: list[float] = []
    for idx in range(n):
        sig = float(signals[idx])
        ret = float(fwd_returns[idx])
        if not (math.isfinite(sig) and math.isfinite(ret)):
            continue
        if abs(sig) > thr:
            sgn = 1.0 if sig > 0.0 else -1.0
            ev_indices.append(idx)
            ev_rets.append(sgn * float(exp_sign) * ret)

    k = len(ev_rets)
    ev_rate = (k / float(n)) if n > 0 else 0.0

    # Greedy non-overlapping effective event count (min bar separation >= max(1, horizon_bars))
    min_sep = max(1, h)
    n_eff_events = 0
    last_kept_idx = -min_sep - 1
    for ev_i in ev_indices:
        if (ev_i - last_kept_idx) >= min_sep:
            n_eff_events += 1
            last_kept_idx = ev_i

    if exp_sign == 0 or k < max(2, int(min_events)):
        return SparseEventExpectancyReport(
            signal_name=signal_name,
            expected_sign=exp_sign,
            total_bars=n,
            n_events=k,
            event_rate=round(ev_rate, 4),
            mean_fwd_ret=0.0,
            mean_fwd_ret_bps=0.0,
            std_fwd_ret=0.0,
            median_fwd_ret=0.0,
            t_stat=0.0,
            p_value_one_sided=0.5,
            win_rate=0.0,
            profit_factor=0.0,
            passed_event_gate=False,
            detail=f"Insufficient sparse events (n={k}, eff={n_eff_events} < {min_events}) or neutral prior.",
            effective_events=n_eff_events,
        )

    mu, _, sd = fast_var_std(ev_rets, ddof=1)
    med = fast_median(ev_rets)

    # Overlap-aware HAC lag: non-zero ONLY when consecutive events fall within horizon_bars
    max_cand_lag = min(k - 1, h)
    l_ev = 0
    for lag in range(1, max_cand_lag + 1):
        if any((ev_indices[j] - ev_indices[j - lag]) < h for j in range(lag, k)):
            l_ev = lag

    # Handle constant/zero-variance returns consistently across spaced & overlapping clusters (FIX-10)
    nw_lag_used = 0
    if sd <= 1e-15:
        t_stat = (10.0 * math.copysign(1.0, mu)) if abs(mu) > 1e-15 else 0.0
    elif l_ev > 0:
        # 2x Bartlett bandwidth matching compute_hac_factor_ic
        nw_lag_req = min(max(1, k - 1), max(2 * l_ev, h))
        nw = newey_west_mean_stats(ev_rets, max_lag=nw_lag_req)
        t_stat = nw.t_stat
        nw_lag_used = nw.nw_lags
    else:
        se = sd / math.sqrt(float(k))
        t_stat = mu / se

    p_one = max(0.0, min(1.0, 1.0 - norm_cdf(t_stat)))
    win_rate = sum(1 for r in ev_rets if r > 0.0) / float(k)
    pos_sum = math.fsum(r for r in ev_rets if r > 0.0)
    neg_sum = math.fsum(-r for r in ev_rets if r < 0.0)
    if neg_sum > 1e-12:
        pf = min(99.0, pos_sum / neg_sum)
    else:
        pf = 99.0 if pos_sum > 0.0 else 0.0

    mean_bps = mu if returns_in_bps else mu * 10_000.0
    passed = (
        k >= int(min_events)
        and n_eff_events >= int(min_events)
        and mu > float(min_mean_fwd_ret)
        and t_stat >= float(min_t_stat)
        and win_rate >= float(min_win_rate)
    )
    detail = (
        f"Sparse event audit: n={k} (eff={n_eff_events}, {ev_rate * 100:.1f}%), mean={mean_bps:+.2f}bp, "
        f"t={t_stat:.2f} (nw_lag={nw_lag_used}), win={win_rate * 100:.1f}%, pf={pf:.2f}"
    )
    return SparseEventExpectancyReport(
        signal_name=signal_name,
        expected_sign=exp_sign,
        total_bars=n,
        n_events=k,
        event_rate=round(ev_rate, 4),
        mean_fwd_ret=round(mu, 6),
        mean_fwd_ret_bps=round(mean_bps, 2),
        std_fwd_ret=round(sd, 6),
        median_fwd_ret=round(med, 6),
        t_stat=round(t_stat, 4),
        p_value_one_sided=round(p_one, 6),
        win_rate=round(win_rate, 4),
        profit_factor=round(pf, 3),
        passed_event_gate=passed,
        detail=detail,
        effective_events=n_eff_events,
    )


# ---------------------------------------------------------------------------
# D8: Passive Buy-and-Hold Benchmark & Risk-Adjusted Alpha Gate (G8)
# ---------------------------------------------------------------------------

def compute_buy_and_hold_returns(
    bars_or_universe: Sequence[CanonicalBar] | dict[str, Sequence[CanonicalBar]],
    *,
    timestamps_ms: Sequence[int] | None = None,
) -> list[float]:
    """Compute strictly passive single-asset or drifting equal-weight multi-asset B&H returns."""
    if isinstance(bars_or_universe, dict):
        universe = {
            sym: sorted(bs, key=lambda b: int(b.ts_event_ms))
            for sym, bs in sorted(bars_or_universe.items())
            if bs
        }
    else:
        ordered = sorted(bars_or_universe, key=lambda b: int(b.ts_event_ms))
        universe = {"__single__": ordered} if ordered else {}

    if not universe:
        return [0.0] * len(timestamps_ms) if timestamps_ms is not None else []

    k = len(universe)
    close_maps: dict[str, dict[int, float]] = {}
    all_ts_set: set[int] = set()
    for sym, ordered in universe.items():
        cmap: dict[int, float] = {}
        for b in ordered:
            if b.close > 0.0 and math.isfinite(b.close):
                ts = int(b.ts_event_ms)
                cmap[ts] = float(b.close)
                all_ts_set.add(ts)
        close_maps[sym] = cmap

    sorted_ts = sorted(all_ts_set)
    if len(sorted_ts) < 2:
        return [0.0] * len(timestamps_ms) if timestamps_ms is not None else []

    # True drifting equal-weight passive portfolio (1/K per symbol, zero hidden rebalance alpha)
    vals: dict[str, float] = {sym: 1.0 / float(k) for sym in universe}
    last_close: dict[str, float] = {}
    eq_by_ts: dict[int, float] = {}
    ret_by_ts: dict[int, float] = {}
    prev_eq: float | None = None

    for ts in sorted_ts:
        for sym in universe:
            c = close_maps[sym].get(ts)
            if c is None:
                continue
            prev_c = last_close.get(sym)
            if prev_c is not None and prev_c > 0.0:
                vals[sym] *= c / prev_c
            last_close[sym] = c
        eq = math.fsum(vals.values())
        eq_by_ts[ts] = eq
        if prev_eq is not None and prev_eq > 1e-12:
            ret_by_ts[ts] = (eq / prev_eq) - 1.0
        prev_eq = eq

    if timestamps_ms is not None:
        return [float(ret_by_ts.get(int(ts), 0.0)) for ts in timestamps_ms]
    return [float(ret_by_ts[ts]) for ts in sorted_ts[1:] if ts in ret_by_ts]


@dataclass
class BenchmarkGateReport:
    benchmark_name: str
    strategy_mode: str
    n_aligned_obs: int
    strategy_sharpe: float
    benchmark_sharpe: float
    strategy_cagr_pct: float
    benchmark_cagr_pct: float
    strategy_max_dd_pct: float
    benchmark_max_dd_pct: float
    strategy_calmar: float
    benchmark_calmar: float
    corr_to_benchmark: float
    beta_to_benchmark: float
    annualized_alpha_pct: float
    beats_sharpe: bool
    beats_calmar: bool
    qualifies_orthogonal_alpha: bool
    passed_benchmark_gate: bool
    pass_pathway: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stream_risk_stats(returns: Sequence[float], periods_per_year: float) -> tuple[float, float, float, float]:
    """Return `(sharpe, cagr_pct, max_dd_pct, calmar)` on an aligned return stream."""
    n = len(returns)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    sr = sharpe_from_returns(returns, periods_per_year)
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        eq *= max(0.0, 1.0 + float(r))
        if eq > peak:
            peak = eq
        dd = ((peak - eq) / peak) * 100.0 if peak > 0.0 else 100.0
        if dd > max_dd:
            max_dd = dd
    years = max(float(n) / max(1.0, float(periods_per_year)), 1.0 / 365.0)
    cagr_pct = ((eq ** (1.0 / years)) - 1.0) * 100.0 if eq > 0.0 else -100.0
    calmar = (cagr_pct / max(max_dd, 0.1)) if cagr_pct > 0.0 else 0.0
    return sr, cagr_pct, max_dd, calmar


def evaluate_benchmark_gate(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    periods_per_year: float,
    *,
    benchmark_name: str = "PASSIVE_BH",
    strategy_mode: str = "auto",  # "directional" | "orthogonal" | "auto"
    min_sharpe_margin: float = 0.0,
    min_calmar_margin: float = 0.0,
    max_orthogonal_corr: float = 0.30,
    min_orthogonal_sharpe: float = 0.0,
) -> BenchmarkGateReport:
    """Gate G8: Passive Buy-and-Hold Benchmark & Risk-Adjusted Alpha Verification (D8)."""
    n = min(len(strategy_returns), len(benchmark_returns))
    mode = str(strategy_mode).strip().lower()
    if mode not in ("directional", "orthogonal", "auto"):
        raise ValueError(f"Unsupported strategy_mode: {strategy_mode!r}")

    if n < 10:
        return BenchmarkGateReport(
            benchmark_name=benchmark_name,
            strategy_mode=mode,
            n_aligned_obs=n,
            strategy_sharpe=0.0,
            benchmark_sharpe=0.0,
            strategy_cagr_pct=0.0,
            benchmark_cagr_pct=0.0,
            strategy_max_dd_pct=0.0,
            benchmark_max_dd_pct=0.0,
            strategy_calmar=0.0,
            benchmark_calmar=0.0,
            corr_to_benchmark=0.0,
            beta_to_benchmark=0.0,
            annualized_alpha_pct=0.0,
            beats_sharpe=False,
            beats_calmar=False,
            qualifies_orthogonal_alpha=False,
            passed_benchmark_gate=False,
            pass_pathway="FAILED",
            detail=f"Insufficient aligned observations ({n} < 10) for benchmark gate.",
        )

    s_ret = [float(x) for x in strategy_returns[len(strategy_returns) - n :]]
    b_ret = [float(x) for x in benchmark_returns[len(benchmark_returns) - n :]]

    s_sr, s_cagr, s_dd, s_cal = _stream_risk_stats(s_ret, periods_per_year)
    b_sr, b_cagr, b_dd, b_cal = _stream_risk_stats(b_ret, periods_per_year)

    corr = fast_pearson_corr(s_ret, b_ret)
    mu_s = fast_mean(s_ret)
    mu_b, var_b, _ = fast_var_std(b_ret, ddof=1)
    if var_b > 1e-24:
        cov_sb = math.fsum((a - mu_s) * (b - mu_b) for a, b in zip(s_ret, b_ret)) / float(n - 1)
        beta = cov_sb / var_b
    else:
        beta = 0.0
    ann_alpha_pct = (mu_s - beta * mu_b) * max(1.0, float(periods_per_year)) * 100.0

    beats_sharpe = (s_sr > 0.0) and (s_sr > b_sr + float(min_sharpe_margin))
    beats_calmar = (s_sr > 0.0) and (s_cagr > 0.0) and (s_cal > b_cal + float(min_calmar_margin))
    qualifies_orth = (abs(corr) <= float(max_orthogonal_corr)) and (s_sr > float(min_orthogonal_sharpe))

    if mode == "directional":
        passed = beats_sharpe or beats_calmar
        pathway = "DIRECTIONAL_OUTPERFORMANCE" if passed else "FAILED"
    elif mode == "orthogonal":
        passed = qualifies_orth
        pathway = "ORTHOGONAL_ALPHA" if passed else "FAILED"
    else:
        if beats_sharpe or beats_calmar:
            passed = True
            pathway = "DIRECTIONAL_OUTPERFORMANCE"
        elif qualifies_orth:
            passed = True
            pathway = "ORTHOGONAL_ALPHA"
        else:
            passed = False
            pathway = "FAILED"

    detail = (
        f"Strategy (SR={s_sr:.3f}, CAGR={s_cagr:+.2f}%, Calmar={s_cal:.3f}) vs "
        f"{benchmark_name} (SR={b_sr:.3f}, CAGR={b_cagr:+.2f}%, Calmar={b_cal:.3f}), "
        f"corr={corr:+.3f}, beta={beta:+.3f}, alpha={ann_alpha_pct:+.2f}%/y -> {pathway}"
    )
    return BenchmarkGateReport(
        benchmark_name=benchmark_name,
        strategy_mode=mode,
        n_aligned_obs=n,
        strategy_sharpe=round(s_sr, 4),
        benchmark_sharpe=round(b_sr, 4),
        strategy_cagr_pct=round(s_cagr, 4),
        benchmark_cagr_pct=round(b_cagr, 4),
        strategy_max_dd_pct=round(s_dd, 4),
        benchmark_max_dd_pct=round(b_dd, 4),
        strategy_calmar=round(s_cal, 4),
        benchmark_calmar=round(b_cal, 4),
        corr_to_benchmark=round(corr, 4),
        beta_to_benchmark=round(beta, 4),
        annualized_alpha_pct=round(ann_alpha_pct, 4),
        beats_sharpe=beats_sharpe,
        beats_calmar=beats_calmar,
        qualifies_orthogonal_alpha=qualifies_orth,
        passed_benchmark_gate=passed,
        pass_pathway=pathway,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# G7: Full Kernel + RiskAllocator Streaming-vs-Batch Bit-Parity Harness
# ---------------------------------------------------------------------------

@dataclass
class ParityReport:
    passed: bool
    checked_timestamps: int
    checked_symbol_steps: int
    max_abs_raw_weight_delta: float
    max_abs_weight_delta: float
    max_abs_equity_delta: float
    prefix_max_weight_delta: float
    prefix_max_equity_delta: float
    regime_mismatch_count: int
    first_mismatch_detail: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_equity_val(point: tuple[int, float] | float) -> float:
    if isinstance(point, tuple):
        return float(point[1])
    return float(point)


def verify_full_parity(
    strategy: AssetNativeStrategy,
    specs: dict[str, AssetSpec],
    bars_by_symbol: dict[str, Sequence[CanonicalBar]],
    *,
    risk_cfg: RiskAllocatorConfig,
    batch_result: SimulationResult | None = None,
    capital_usd: float = 10_000.0,
    execution_mode: str = "TAKER_TAKER",
    extra_slippage_bps_per_leg: float = 0.0,
    context_provider: Callable[[int], dict[str, Any]] | None = None,
    weight_tol: float = 1e-10,
    equity_tol: float = 1e-8,
    accounting_engine: PortfolioAccountingEngine | None = None,
) -> ParityReport:
    """Gate G7: Verify BOTH per-symbol strategy AND `CausalRiskAllocator` bit-parity + causality.

    Runs three independent checks against `batch_result`:
      1. Bar-by-bar `StreamingAccountingSession` replay with a fresh `strategy.clone_fresh()`,
         verifying `raw_decisions`, `allocated_weights`, `executed_weights`, `regime_label`,
         and `equity_curve`.
      2. Direct independent `strategy.clone_fresh()` + fresh `CausalRiskAllocator(risk_cfg)`
         replay (without relying on `PortfolioAccountingEngine` internals) verifying allocated
         weights and raw decisions bit-for-bit.
      3. Hostile 60% prefix-truncation causality probe where future bars are physically
         removed from memory before simulation.
    """
    engine = accounting_engine if accounting_engine is not None else PortfolioAccountingEngine()
    if batch_result is None:
        batch_result = engine.simulate(
            strategy.clone_fresh(),
            specs,
            bars_by_symbol,
            risk_cfg=risk_cfg,
            capital_usd=capital_usd,
            execution_mode=execution_mode,
            extra_slippage_bps_per_leg=extra_slippage_bps_per_leg,
            context_provider=context_provider,
        )

    # Index bars strictly by timestamp
    bars_by_ts: dict[int, dict[str, CanonicalBar]] = {}
    for sym in sorted(bars_by_symbol):
        if sym not in specs:
            continue
        for b in sorted(bars_by_symbol[sym], key=lambda x: int(x.ts_event_ms)):
            ts = int(b.ts_event_ms)
            slot = bars_by_ts.get(ts)
            if slot is None:
                slot = {}
                bars_by_ts[ts] = slot
            slot[sym] = b

    sorted_ts = sorted(bars_by_ts)
    if not sorted_ts:
        return ParityReport(
            passed=False,
            checked_timestamps=0,
            checked_symbol_steps=0,
            max_abs_raw_weight_delta=1.0,
            max_abs_weight_delta=1.0,
            max_abs_equity_delta=1.0,
            prefix_max_weight_delta=1.0,
            prefix_max_equity_delta=1.0,
            regime_mismatch_count=0,
            first_mismatch_detail="No bars available for parity verification.",
            detail="FAIL: empty bar stream.",
        )

    if len(batch_result.timestamps_ms) != len(sorted_ts) or len(batch_result.equity_curve) != len(sorted_ts):
        return ParityReport(
            passed=False,
            checked_timestamps=0,
            checked_symbol_steps=0,
            max_abs_raw_weight_delta=1.0,
            max_abs_weight_delta=1.0,
            max_abs_equity_delta=1.0,
            prefix_max_weight_delta=1.0,
            prefix_max_equity_delta=1.0,
            regime_mismatch_count=0,
            first_mismatch_detail=(
                f"Length mismatch: batch_ts={len(batch_result.timestamps_ms)}, "
                f"batch_eq={len(batch_result.equity_curve)}, input_ts={len(sorted_ts)}"
            ),
            detail="FAIL: batch result length mismatch.",
        )

    # 1. Independent StreamingAccountingSession bar-by-bar replay
    session_s = engine.create_session(
        strategy.clone_fresh(),
        specs,
        risk_cfg=risk_cfg,
        capital_usd=capital_usd,
        execution_mode=execution_mode,
        extra_slippage_bps_per_leg=extra_slippage_bps_per_leg,
    )

    # 2. Direct independent strategy + CausalRiskAllocator replay (zero accounting engine shortcut)
    strat_direct = strategy.clone_fresh()
    strat_direct.reset()
    alloc_direct = CausalRiskAllocator(risk_cfg)
    ret_maxlen = max(128, int(risk_cfg.corr_lookback_bars) + 5)
    direct_rets: dict[str, deque[float]] = {s: deque(maxlen=ret_maxlen) for s in sorted(specs)}
    direct_last_close: dict[str, float] = {}
    direct_held: dict[str, float] = {s: 0.0 for s in sorted(specs)}
    direct_pending: dict[str, float] = {s: 0.0 for s in sorted(specs)}
    direct_seen_tfs: dict[str, float] = {}

    max_raw_delta = 0.0
    max_w_delta = 0.0
    max_eq_delta = 0.0
    regime_mismatches = 0
    checked_sym_steps = 0
    first_mismatch = ""

    for idx, ts in enumerate(sorted_ts):
        bar_slice = bars_by_ts[ts]
        ctx = context_provider(ts) if context_provider is not None else None
        step_res = session_s.step_bar(ts, bar_slice, context=ctx)

        # Compare StreamingAccountingSession vs batch_result
        batch_eq = _extract_equity_val(batch_result.equity_curve[idx])
        eq_delta = abs(step_res.equity_usd - batch_eq)
        if eq_delta > max_eq_delta:
            max_eq_delta = eq_delta
            if eq_delta > equity_tol and not first_mismatch:
                first_mismatch = f"Equity drift at ts={ts}: batch={batch_eq:.8f} vs stream={step_res.equity_usd:.8f}"

        batch_raw = batch_result.raw_decisions_history[idx] if idx < len(batch_result.raw_decisions_history) else {}
        batch_alloc = (
            batch_result.allocated_weights_history[idx][1]
            if idx < len(batch_result.allocated_weights_history)
            else {}
        )
        batch_exec = (
            batch_result.executed_weights_history[idx]
            if idx < len(batch_result.executed_weights_history)
            else {}
        )

        # Direct independent allocator check (and direct strategy check when neither halted nor fill-overridden)
        for sym, b in sorted(bar_slice.items()):
            direct_seen_tfs[sym] = float(b.timeframe_h)
            prev_c = direct_last_close.get(sym)
            if prev_c is not None and prev_c > 0.0 and b.close > 0.0:
                direct_rets.setdefault(sym, deque(maxlen=ret_maxlen)).append((b.close / prev_c) - 1.0)
            direct_last_close[sym] = float(b.close)

        if session_s.is_halted:
            direct_raw: dict[str, BarDecision] = {}
            direct_full_alloc: dict[str, float] = {s: 0.0 for s in sorted(step_res.allocated_weights)}
        elif not session_s.had_any_fill_override:
            for sym in sorted(bar_slice):
                direct_held[sym] = direct_pending.get(sym, 0.0)
            direct_raw = strat_direct.on_bar_step(ts, bar_slice, context=ctx)
            direct_locked = {
                s: float(w)
                for s, w in direct_held.items()
                if s not in direct_raw and abs(w) > 1e-9
            }
            direct_alloc = alloc_direct.allocate(
                ts,
                direct_raw,
                direct_rets,
                specs,
                locked_weights=direct_locked,
                timeframe_h_by_symbol=dict(direct_seen_tfs),
            )
            direct_full_alloc = dict(direct_held)
            direct_full_alloc.update(direct_alloc)
            strat_direct.sync_executed_weights(dict(direct_full_alloc))
            for sym in sorted(direct_raw):
                direct_pending[sym] = float(direct_alloc.get(sym, 0.0))
        else:
            direct_raw = step_res.raw_decisions
            held_now = dict(step_res.executed_weights)
            direct_locked = {
                s: float(w)
                for s, w in held_now.items()
                if s not in direct_raw and abs(w) > 1e-9
            }
            direct_alloc = alloc_direct.allocate(
                ts,
                direct_raw,
                direct_rets,
                specs,
                locked_weights=direct_locked,
                timeframe_h_by_symbol=dict(direct_seen_tfs),
            )
            direct_full_alloc = dict(held_now)
            direct_full_alloc.update(direct_alloc)

        all_syms = sorted(
            set(bar_slice)
            | set(batch_raw)
            | set(batch_alloc)
            | set(batch_exec)
            | set(step_res.allocated_weights)
            | set(step_res.executed_weights)
            | set(direct_full_alloc)
        )
        for sym in all_syms:
            checked_sym_steps += 1
            b_dec = batch_raw.get(sym)
            s_dec = step_res.raw_decisions.get(sym)
            d_dec = direct_raw.get(sym)
            if b_dec is not None and s_dec is not None:
                rd1 = abs(s_dec.target_weight - b_dec.target_weight)
                rd2 = abs((d_dec.target_weight if d_dec is not None else 0.0) - b_dec.target_weight)
                rd = max(rd1, rd2)
                if rd > max_raw_delta:
                    max_raw_delta = rd
                if rd > weight_tol and not first_mismatch:
                    first_mismatch = f"Raw weight drift on {sym}@{ts}: delta={rd:.3e}"
                if s_dec.regime_label != b_dec.regime_label or (d_dec is not None and d_dec.regime_label != b_dec.regime_label):
                    regime_mismatches += 1
                    if not first_mismatch:
                        first_mismatch = f"Regime mismatch on {sym}@{ts}: batch={b_dec.regime_label} vs stream={s_dec.regime_label}"
            elif (b_dec is None) != (s_dec is None):
                max_raw_delta = max(max_raw_delta, 1.0)
                if not first_mismatch:
                    first_mismatch = f"Missing raw decision on {sym}@{ts}"

            wd_alloc_s = abs(step_res.allocated_weights.get(sym, 0.0) - batch_alloc.get(sym, 0.0))
            wd_alloc_d = abs(direct_full_alloc.get(sym, 0.0) - batch_alloc.get(sym, 0.0))
            wd_exec_s = abs(step_res.executed_weights.get(sym, 0.0) - batch_exec.get(sym, 0.0))
            wd = max(wd_alloc_s, wd_alloc_d, wd_exec_s)
            if wd > max_w_delta:
                max_w_delta = wd
                if wd > weight_tol and not first_mismatch:
                    first_mismatch = f"Allocated/executed weight drift on {sym}@{ts}: delta={wd:.3e}"

    # 3. Hostile 60% Prefix-Truncation Causality Probe (future bars physically absent!)
    prefix_max_w = 0.0
    prefix_max_eq = 0.0
    if len(sorted_ts) >= 10:
        cut_idx = max(3, (len(sorted_ts) * 3) // 5)
        ts_mid = sorted_ts[cut_idx]
        prefix_bars = {
            sym: [b for b in bs if int(b.ts_event_ms) <= ts_mid]
            for sym, bs in bars_by_symbol.items()
        }
        prefix_res = engine.simulate(
            strategy.clone_fresh(),
            specs,
            prefix_bars,
            risk_cfg=risk_cfg,
            capital_usd=capital_usd,
            execution_mode=execution_mode,
            extra_slippage_bps_per_leg=extra_slippage_bps_per_leg,
            context_provider=context_provider,
        )
        for p_idx in range(len(prefix_res.timestamps_ms)):
            p_eq = _extract_equity_val(prefix_res.equity_curve[p_idx])
            b_eq = _extract_equity_val(batch_result.equity_curve[p_idx])
            pe_d = abs(p_eq - b_eq)
            if pe_d > prefix_max_eq:
                prefix_max_eq = pe_d
                if pe_d > equity_tol and not first_mismatch:
                    first_mismatch = f"Prefix lookahead equity drift at idx={p_idx}: delta={pe_d:.3e}"

            p_aw = prefix_res.allocated_weights_history[p_idx][1]
            b_aw = batch_result.allocated_weights_history[p_idx][1]
            p_ew = prefix_res.executed_weights_history[p_idx]
            b_ew = batch_result.executed_weights_history[p_idx]
            for sym in set(p_aw) | set(b_aw) | set(p_ew) | set(b_ew):
                pw_d = max(
                    abs(p_aw.get(sym, 0.0) - b_aw.get(sym, 0.0)),
                    abs(p_ew.get(sym, 0.0) - b_ew.get(sym, 0.0)),
                )
                if pw_d > prefix_max_w:
                    prefix_max_w = pw_d
                    if pw_d > weight_tol and not first_mismatch:
                        first_mismatch = f"Prefix lookahead weight drift on {sym}@idx={p_idx}: delta={pw_d:.3e}"

            p_raw = prefix_res.raw_decisions_history[p_idx] if p_idx < len(prefix_res.raw_decisions_history) else {}
            b_raw = batch_result.raw_decisions_history[p_idx] if p_idx < len(batch_result.raw_decisions_history) else {}
            for sym in sorted(set(p_raw) | set(b_raw)):
                pr_dec = p_raw.get(sym)
                br_dec = b_raw.get(sym)
                if pr_dec is not None and br_dec is not None:
                    pr_d = abs(pr_dec.target_weight - br_dec.target_weight)
                    if pr_d > prefix_max_w:
                        prefix_max_w = pr_d
                        if pr_d > weight_tol and not first_mismatch:
                            first_mismatch = f"Prefix lookahead raw weight drift on {sym}@idx={p_idx}: delta={pr_d:.3e}"
                    if pr_dec.regime_label != br_dec.regime_label:
                        regime_mismatches += 1
                        if not first_mismatch:
                            first_mismatch = (
                                f"Prefix lookahead regime mismatch on {sym}@idx={p_idx}: "
                                f"batch={br_dec.regime_label} vs prefix={pr_dec.regime_label}"
                            )
                else:
                    prefix_max_w = max(prefix_max_w, 1.0)
                    if not first_mismatch:
                        first_mismatch = f"Prefix lookahead missing raw decision on {sym}@idx={p_idx}"

    passed = (
        len(sorted_ts) > 0
        and max_raw_delta <= weight_tol
        and max_w_delta <= weight_tol
        and max_eq_delta <= equity_tol
        and prefix_max_w <= weight_tol
        and prefix_max_eq <= equity_tol
        and regime_mismatches == 0
    )
    detail = (
        f"Full kernel+allocator parity across {len(sorted_ts)} timestamps: "
        f"weight_delta={max_w_delta:.2e}, equity_delta={max_eq_delta:.2e}, "
        f"prefix_w={prefix_max_w:.2e}, prefix_eq={prefix_max_eq:.2e}, "
        f"regime_mismatches={regime_mismatches}"
    )
    if not passed and first_mismatch:
        detail = f"{detail} | first_mismatch={first_mismatch}"

    return ParityReport(
        passed=passed,
        checked_timestamps=len(sorted_ts),
        checked_symbol_steps=checked_sym_steps,
        max_abs_raw_weight_delta=max_raw_delta,
        max_abs_weight_delta=max_w_delta,
        max_abs_equity_delta=max_eq_delta,
        prefix_max_weight_delta=prefix_max_w,
        prefix_max_equity_delta=prefix_max_eq,
        regime_mismatch_count=regime_mismatches,
        first_mismatch_detail=first_mismatch,
        detail=detail,
    )


verify_full_pipeline_parity = verify_full_parity


# ---------------------------------------------------------------------------
# Unified 8-Gate Institutional Evaluator (G1 .. G8)
# ---------------------------------------------------------------------------

class GateVerdict(str, Enum):
    PROMOTE_SHADOW_CANDIDATE = "PROMOTE_SHADOW_CANDIDATE"
    CONDITIONAL_RESEARCH = "CONDITIONAL_RESEARCH"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class InstitutionalGateOutcome:
    gate_id: str
    name: str
    passed: bool
    value: str
    threshold: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalGateReport:
    strategy_id: str
    asset_domain: str
    verdict: str
    passed_gates: int
    total_gates: int
    gates: list[InstitutionalGateOutcome]
    gate_map: dict[str, bool]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "asset_domain": self.asset_domain,
            "verdict": str(self.verdict),
            "passed_gates": self.passed_gates,
            "total_gates": self.total_gates,
            "gates": [g.to_dict() for g in self.gates],
            "gate_map": dict(self.gate_map),
            "summary": dict(self.summary),
        }


def _metric_get(obj: SimulationResult | dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(key, default)
    else:
        val = getattr(obj, key, default)
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def evaluate_8_gates(
    strategy_id: str,
    asset_domain: str,
    *,
    dq_reports: Sequence[Any] | None = None,
    factor_audit: Any | None = None,
    event_reports: Sequence[SparseEventExpectancyReport] | None = None,
    baseline_metrics: SimulationResult | dict[str, Any] | None = None,
    stress_metrics: SimulationResult | dict[str, Any] | None = None,
    plateau_report: PlateauSelectionReport | None = None,
    nested_cpcv_report: NestedCPCVReport | CPCVReport | None = None,
    dsr_report: DSRReport | None = None,
    pbo_report: PBOReport | None = None,
    parity_report: ParityReport | None = None,
    benchmark_report: BenchmarkGateReport | None = None,
    allow_synthetic_in_test: bool = False,
    min_trades: int = 5,
    min_dsr: float = 0.50,
    max_pbo: float = 0.35,
) -> InstitutionalGateReport:
    """Evaluate all 8 institutional gates (`G1..G8`) fail-closed with zero threshold relaxation."""
    # G1: Data Quality & Provenance
    if dq_reports:
        dq_list = list(dq_reports)
        prov_ok = True
        if not allow_synthetic_in_test:
            for r in dq_list:
                prov = getattr(r, "provenance", "")
                if prov != PROVENANCE_REAL:
                    prov_ok = False
        dq_pass_n = sum(1 for r in dq_list if bool(getattr(r, "passed", False)))
        g1_ok = (dq_pass_n == len(dq_list)) and prov_ok
        g1_val = f"{dq_pass_n}/{len(dq_list)} DQ pass (real_prov={prov_ok})"
        g1_det = "All DQ checks passed" if g1_ok else "DQ failure or non-real provenance"
    else:
        g1_ok = False
        g1_val = "MISSING"
        g1_det = "Fail-closed: dq_reports missing or empty."

    # G2: Factor / Signal Audit (Continuous HAC IC or Sparse Event Expectancy)
    sel_factors = list(getattr(factor_audit, "selected_non_redundant_factors", ())) if factor_audit is not None else []
    passed_events = [r.signal_name for r in (event_reports or ()) if r.passed_event_gate]
    g2_ok = (len(sel_factors) >= 1) or (len(passed_events) >= 1)
    g2_val = f"factors={len(sel_factors)}, events={len(passed_events)}"
    g2_det = f"selected={sel_factors[:6]}, passed_events={passed_events[:4]}"

    # G3: Cost-First Economic Edge + Stress (+2bp/leg) + Zero Liquidations
    if baseline_metrics is not None and stress_metrics is not None:
        base_net = _metric_get(baseline_metrics, "net_pnl_usd")
        base_sr = _metric_get(baseline_metrics, "sharpe_ratio")
        base_liq = int(round(_metric_get(baseline_metrics, "liquidation_count")))
        base_trades = int(round(_metric_get(baseline_metrics, "trade_count")))
        stress_net = _metric_get(stress_metrics, "net_pnl_usd")
        g3_ok = (
            base_net > 0.0
            and base_sr > 0.0
            and stress_net > 0.0
            and base_liq == 0
            and base_trades >= int(min_trades)
        )
        g3_val = f"net=${base_net:+.2f}, SR={base_sr:.2f}, stress=${stress_net:+.2f}, liq={base_liq}, trades={base_trades}"
        g3_det = "Positive net edge after full costs & stress" if g3_ok else "Failed full-cost economic edge gate"
    else:
        g3_ok = False
        g3_val = "MISSING"
        g3_det = "Fail-closed: baseline_metrics or stress_metrics missing."

    # G4: Local Parameter Stability / Plateau
    if plateau_report is not None:
        g4_ok = bool(plateau_report.passed_plateau_gate)
        g4_val = (
            f"plateau={plateau_report.largest_plateau_size}/{plateau_report.total_points} "
            f"({plateau_report.plateau_ratio * 100:.0f}%), c2p={plateau_report.center_to_peak_ratio:.2f}"
        )
        g4_det = plateau_report.detail
    else:
        g4_ok = False
        g4_val = "MISSING"
        g4_det = "Fail-closed: plateau_report missing."

    # G5: True Nested CPCV + Fixed DSR (strictly rejects non-nested CPCVReport!)
    if nested_cpcv_report is not None and dsr_report is not None:
        is_nested = bool(getattr(nested_cpcv_report, "is_nested", False))
        cpcv_pass = bool(getattr(nested_cpcv_report, "passed_cpcv_gate", False))
        g5_ok = is_nested and cpcv_pass and (dsr_report.dsr >= float(min_dsr))
        med_oos = float(getattr(nested_cpcv_report, "median_oos_sharpe", 0.0))
        pct_pos = float(getattr(nested_cpcv_report, "pct_positive_splits", 0.0))
        g5_val = f"nested={is_nested}, med_OOS_SR={med_oos:.2f}, pos={pct_pos:.0f}%, DSR={dsr_report.dsr:.3f}"
        g5_det = (
            f"SR={dsr_report.sr_annualized:.2f} vs SR*={dsr_report.sr_star_annualized:.2f} "
            f"(N={dsr_report.n_trials}, N_eff={dsr_report.n_eff_trials:.2f})"
        )
        if not is_nested:
            g5_det = "REJECTED: non-nested CPCV report supplied (D3 requires true nested fold refitting)."
    else:
        g5_ok = False
        g5_val = "MISSING"
        g5_det = "Fail-closed: nested_cpcv_report or dsr_report missing."

    # G6: Multi-Variant CSCV PBO (strict PBO <= 0.35, never relaxed)
    if pbo_report is not None:
        g6_ok = (pbo_report.n_combinations > 0) and (pbo_report.pbo <= float(max_pbo))
        g6_val = f"PBO={pbo_report.pbo:.3f}, P(OOS_loss)={pbo_report.prob_oos_loss:.3f}"
        g6_det = (
            f"combos={pbo_report.n_combinations}, center_deg={pbo_report.plateau_center_oos_degradation:.3f}, "
            f"spread/SE={pbo_report.effective_spread_over_se:.3f}, avg_corr={pbo_report.avg_pairwise_corr:.3f}, "
            f"flat_micro_grid={pbo_report.is_flat_micro_grid}"
        )
    else:
        g6_ok = False
        g6_val = "MISSING"
        g6_det = "Fail-closed: pbo_report missing."

    # G7: Full Kernel + RiskAllocator Streaming-vs-Batch Bit-Parity
    if parity_report is not None:
        g7_ok = (
            bool(parity_report.passed)
            and parity_report.max_abs_weight_delta <= 1e-10
            and parity_report.max_abs_equity_delta <= 1e-8
            and parity_report.prefix_max_weight_delta <= 1e-10
            and parity_report.prefix_max_equity_delta <= 1e-8
            and parity_report.regime_mismatch_count == 0
        )
        g7_val = (
            f"w_delta={parity_report.max_abs_weight_delta:.2e}, "
            f"eq_delta={parity_report.max_abs_equity_delta:.2e}"
        )
        g7_det = parity_report.detail
    else:
        g7_ok = False
        g7_val = "MISSING"
        g7_det = "Fail-closed: parity_report missing."

    # G8: Mandatory Passive Benchmark & Risk-Adjusted Alpha Gate
    if benchmark_report is not None:
        g8_ok = bool(benchmark_report.passed_benchmark_gate)
        g8_val = (
            f"{benchmark_report.pass_pathway} "
            f"(SR {benchmark_report.strategy_sharpe:.2f} vs {benchmark_report.benchmark_sharpe:.2f}, "
            f"corr={benchmark_report.corr_to_benchmark:+.2f})"
        )
        g8_det = benchmark_report.detail
    else:
        g8_ok = False
        g8_val = "MISSING"
        g8_det = "Fail-closed: benchmark_report missing."

    gates = [
        InstitutionalGateOutcome("G1", "Data Quality & Provenance (DQ-MA-01..07)", g1_ok, g1_val, "100% REAL & DQ PASS", g1_det),
        InstitutionalGateOutcome("G2", "Factor HAC IC / Sparse Event Audit", g2_ok, g2_val, ">=1 non-redundant factor or event PASS", g2_det),
        InstitutionalGateOutcome("G3", "Cost-First Economic Edge & Stress", g3_ok, g3_val, f"net>0 & SR>0 & stress>0 & liq=0 & trades>={min_trades}", g3_det),
        InstitutionalGateOutcome("G4", "Local Parameter Plateau Stability", g4_ok, g4_val, "plateau>=55% & size>=3 & c2p>=0.55", g4_det),
        InstitutionalGateOutcome("G5", "Nested CPCV Generalization & Fixed DSR", g5_ok, g5_val, f"nested CPCV PASS & DSR>={min_dsr:.2f}", g5_det),
        InstitutionalGateOutcome("G6", "Multi-Variant CSCV PBO", g6_ok, g6_val, f"PBO<={max_pbo:.2f}", g6_det),
        InstitutionalGateOutcome("G7", "Full Kernel + RiskAllocator Bit-Parity", g7_ok, g7_val, "w<=1e-10 & eq<=1e-8 & prefix=0", g7_det),
        InstitutionalGateOutcome("G8", "Passive Benchmark & Risk-Adjusted Alpha", g8_ok, g8_val, "beat B&H (SR or Calmar) OR orthogonal alpha (|corr|<=0.30, SR>0)", g8_det),
    ]

    passed_n = sum(1 for g in gates if g.passed)
    if passed_n == 8:
        verdict = GateVerdict.PROMOTE_SHADOW_CANDIDATE.value
    elif g1_ok and g3_ok and g7_ok and passed_n >= 6:
        verdict = GateVerdict.CONDITIONAL_RESEARCH.value
    else:
        verdict = GateVerdict.REJECTED.value

    return InstitutionalGateReport(
        strategy_id=strategy_id,
        asset_domain=asset_domain,
        verdict=verdict,
        passed_gates=passed_n,
        total_gates=8,
        gates=gates,
        gate_map={g.gate_id: g.passed for g in gates},
        summary={
            "verdict": verdict,
            "passed_gates": passed_n,
            "total_gates": 8,
            "failed_gates": [g.gate_id for g in gates if not g.passed],
        },
    )
