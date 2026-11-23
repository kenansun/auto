"""Institutional Anti-Overfitting Validation Suite (Pure Python Stdlib).

Implements core AFML (López de Prado) & BAISYS_QUANT statistical defenses without external
binary dependencies so the suite runs identically on 768MB cloud servers and dev machines:
  1. Exact Probabilistic Sharpe Ratio (PSR) & Deflated Sharpe Ratio (DSR) adjusting for
     sample length T, skewness, excess kurtosis, and multiple-testing trial count N.
  2. Combinatorial Purged Cross-Validation (CPCV) with explicit purge & embargo windows.
  3. Combinatorially Symmetric Cross-Validation Probability of Backtest Overfitting (PBO).
  4. Connected Parameter Plateau Geometric Center selector (BAISYS_QUANT rule: never pick
     an isolated peak; select geometric centroid of the largest 4-connected robust plateau).
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

EULER_MASCHERONI = 0.5772156649015329


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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


def sharpe_from_returns(returns: list[float], periods_per_year: float) -> float:
    if len(returns) < 3:
        return 0.0
    mu = statistics.mean(returns)
    sd = statistics.stdev(returns)
    if sd <= 1e-12:
        return 0.0
    return (mu / sd) * math.sqrt(max(1.0, periods_per_year))


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_psr_dsr(
    returns: list[float],
    periods_per_year: float,
    n_trials: int = 1,
    trial_sharpe_std_per_period: float | None = None,
) -> DSRReport:
    """Exact Bailey & López de Prado (2014) Probabilistic & Deflated Sharpe Ratio."""
    n = len(returns)
    if n < 5:
        return DSRReport(n, max(1, n_trials), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5)

    mu = statistics.mean(returns)
    sd = statistics.stdev(returns)
    if sd <= 1e-12:
        return DSRReport(n, max(1, n_trials), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5)

    sr = mu / sd
    ann_scale = math.sqrt(max(1.0, periods_per_year))
    sr_ann = sr * ann_scale

    # Higher moments
    m2 = sum((r - mu) ** 2 for r in returns) / n
    m3 = sum((r - mu) ** 3 for r in returns) / n
    m4 = sum((r - mu) ** 4 for r in returns) / n
    skew = m3 / (m2 ** 1.5) if m2 > 1e-18 else 0.0
    kurt_pearson = m4 / (m2 * m2) if m2 > 1e-18 else 3.0
    excess_kurt = kurt_pearson - 3.0

    # Non-normal Sharpe variance (Bailey & López de Prado 2014 Eq. 5)
    var_denom = max(1e-9, 1.0 - skew * sr + ((kurt_pearson - 1.0) / 4.0) * (sr * sr))
    sr_se = math.sqrt(var_denom / max(1, n - 1))

    psr = norm_cdf(sr / sr_se) if sr_se > 1e-12 else 0.5

    trials = max(1, int(n_trials))
    if trials <= 1:
        sr_star = 0.0
    else:
        # Expected maximum of N null trials under extreme-value theory
        z1 = norm_ppf(1.0 - (1.0 / trials))
        z2 = norm_ppf(1.0 - (1.0 / (trials * math.e)))
        # Conservative default cross-trial Sharpe std when trial matrix is unavailable
        sig_trials = (
            float(trial_sharpe_std_per_period)
            if (trial_sharpe_std_per_period is not None and trial_sharpe_std_per_period > 0)
            else max(sr_se, 1.0 / math.sqrt(max(2, n)))
        )
        sr_star = sig_trials * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)

    dsr = norm_cdf((sr - sr_star) / sr_se) if sr_se > 1e-12 else 0.5
    return DSRReport(
        n_obs=n,
        n_trials=trials,
        sr_per_period=round(sr, 5),
        sr_annualized=round(sr_ann, 4),
        sr_star_per_period=round(sr_star, 5),
        sr_star_annualized=round(sr_star * ann_scale, 4),
        skewness=round(skew, 4),
        excess_kurtosis=round(excess_kurt, 4),
        psr=round(psr, 4),
        dsr=round(dsr, 4),
    )


@dataclass(frozen=True)
class CPCVSplit:
    split_id: int
    test_groups: tuple[int, ...]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    purged_count: int
    embargoed_count: int


def generate_cpcv_splits(
    n_obs: int,
    n_groups: int = 6,
    k_test: int = 2,
    purge_bars: int = 2,
    embargo_bars: int = 2,
) -> list[CPCVSplit]:
    """Generate AFML Combinatorial Purged Cross-Validation splits with purge & embargo."""
    if n_obs < n_groups * 4 or k_test <= 0 or k_test >= n_groups:
        return []
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
            for p_idx in range(max(0, lo - purge_bars), lo):
                purge_set.add(p_idx)
            for e_idx in range(hi, min(n_obs, hi + embargo_bars)):
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_cpcv_returns(
    period_returns: list[float],
    periods_per_year: float,
    n_groups: int = 6,
    k_test: int = 2,
    purge_bars: int = 2,
    embargo_bars: int = 2,
) -> CPCVReport:
    """Evaluate a single strategy's return stream across all purged/embargoed CPCV splits."""
    splits = generate_cpcv_splits(len(period_returns), n_groups, k_test, purge_bars, embargo_bars)
    if not splits:
        return CPCVReport(n_groups, k_test, 0, purge_bars, embargo_bars, [], 0.0, 0.0, 0.0, 0.0)

    is_srs: list[float] = []
    oos_srs: list[float] = []
    for sp in splits:
        tr = [period_returns[i] for i in sp.train_indices]
        te = [period_returns[i] for i in sp.test_indices]
        is_srs.append(sharpe_from_returns(tr, periods_per_year))
        oos_srs.append(sharpe_from_returns(te, periods_per_year))

    med_oos = statistics.median(oos_srs)
    med_is = statistics.median(is_srs)
    pct_pos = 100.0 * sum(1 for x in oos_srs if x > 0.0) / len(oos_srs)
    retention = (med_oos / med_is) if med_is > 1e-6 else (1.0 if med_oos > 0 else 0.0)
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
    )


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_pbo_cscv(
    candidate_returns_matrix: list[list[float]],
    periods_per_year: float,
    n_subsamples: int = 8,
) -> PBOReport:
    """Bailey, Borwein, López de Prado & Zhu (2015) CSCV Probability of Backtest Overfitting.

    `candidate_returns_matrix`: list of M candidates, each a list of T aligned period returns.
    """
    m = len(candidate_returns_matrix)
    if m < 2:
        return PBOReport(m, n_subsamples, 0, 1.0, 0.0, 0.0, 0.0, 1.0)
    t = min(len(r) for r in candidate_returns_matrix)
    s = n_subsamples if (n_subsamples >= 4 and n_subsamples % 2 == 0) else 8
    while s > 4 and (t // s) < 4:
        s -= 2
    if t < s * 4:
        return PBOReport(m, s, 0, 1.0, 0.0, 0.0, 0.0, 1.0)

    chunk_len = t // s
    blocks: list[list[list[float]]] = []
    for c_idx in range(m):
        row = candidate_returns_matrix[c_idx][: chunk_len * s]
        blocks.append([row[k * chunk_len : (k + 1) * chunk_len] for k in range(s)])

    half = s // 2
    logits: list[float] = []
    oos_ranks: list[float] = []
    is_best_srs: list[float] = []
    oos_best_srs: list[float] = []

    all_blocks = set(range(s))
    for is_comb in itertools.combinations(range(s), half):
        oos_comb = tuple(sorted(all_blocks - set(is_comb)))
        is_scores: list[float] = []
        oos_scores: list[float] = []
        for c_idx in range(m):
            is_ret = [r for b in is_comb for r in blocks[c_idx][b]]
            oos_ret = [r for b in oos_comb for r in blocks[c_idx][b]]
            is_scores.append(sharpe_from_returns(is_ret, periods_per_year))
            oos_scores.append(sharpe_from_returns(oos_ret, periods_per_year))

        best_c = max(range(m), key=lambda idx: is_scores[idx])
        is_best_srs.append(is_scores[best_c])
        oos_best_srs.append(oos_scores[best_c])

        # Relative rank of IS-winner in OOS (1 = worst, M = best) -> omega in (0, 1)
        oos_winner_val = oos_scores[best_c]
        less = sum(1 for v in oos_scores if v < oos_winner_val)
        equal = sum(1 for v in oos_scores if v == oos_winner_val)
        rank_1based = less + 0.5 * (equal + 1.0)
        omega = max(1e-6, min(1.0 - 1e-6, rank_1based / (m + 1.0)))
        oos_ranks.append(omega)
        logits.append(math.log(omega / (1.0 - omega)))

    n_comb = len(logits)
    pbo = sum(1 for lg in logits if lg <= 0.0) / n_comb if n_comb else 1.0
    prob_loss = sum(1 for sr in oos_best_srs if sr <= 0.0) / n_comb if n_comb else 1.0
    return PBOReport(
        n_candidates=m,
        n_subsamples=s,
        n_combinations=n_comb,
        pbo=round(pbo, 4),
        median_oos_relative_rank=round(statistics.median(oos_ranks), 4),
        median_is_winner_sharpe=round(statistics.median(is_best_srs), 4),
        median_oos_winner_sharpe=round(statistics.median(oos_best_srs), 4),
        prob_oos_loss=round(prob_loss, 4),
    )


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
    points: list[PlateauPoint],
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

    # Find largest 4-connected component among eligible robust points
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
        # Honest fallback to center coordinate (1, 1) if present, else best available, with gate=FAIL
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

    # Pick largest connected plateau (tie-break by highest mean Sharpe)
    largest = max(components, key=lambda c: (len(c), statistics.mean(p.sharpe for p in c)))
    plateau_ratio = len(largest) / total
    cx_mean = statistics.mean(p.ix for p in largest)
    cy_mean = statistics.mean(p.iy for p in largest)

    def _neighbor_min_sharpe(pt: PlateauPoint) -> float:
        nbrs = [
            by_coord[(pt.ix + dx, pt.iy + dy)].sharpe
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
            if (pt.ix + dx, pt.iy + dy) in by_coord
        ]
        return min(nbrs) if nbrs else pt.sharpe

    # Select geometric centroid of largest connected plateau, tie-breaking by safest neighborhood floor
    selected = min(
        largest,
        key=lambda p: (
            (p.ix - cx_mean) ** 2 + (p.iy - cy_mean) ** 2,
            -_neighbor_min_sharpe(p),
            -p.sharpe,
        ),
    )

    c2p = (selected.sharpe / peak_pt.sharpe) if peak_pt.sharpe > 1e-6 else 0.0
    # Reject isolated spikes (e.g., plateau_size < 55% of grid OR isolated peak > 2x plateau center)
    passed = (plateau_ratio >= min_plateau_ratio) and (len(largest) >= 3) and (c2p >= 0.55)
    detail = (
        f"Largest connected plateau={len(largest)}/{total} ({plateau_ratio*100:.1f}%), "
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
