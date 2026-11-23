"""Institutional Anti-Overfitting Validation Suite (Backwards-Compatible v3/v4 Facade).

Re-exports all canonical statistical validation primitives from
`alphaloop.platform.core_engine.validation` so both the thin Edge/Execution Plane
(pure Python 3.12 stdlib fast-float path) and the scalable multi-server Research Plane
(optional zero-copyleft NumPy acceleration) share one single source of truth:
  1. Exact Probabilistic Sharpe Ratio (PSR) & Deflated Sharpe Ratio (DSR) with null
     sampling variance floor, effective independent trials `N_eff`, and `TrialLedger`
     family trial count support.
  2. Purged & Embargoed CPCV plus true Nested CPCV (`run_nested_cpcv`).
  3. Combinatorially Symmetric Cross-Validation PBO (`compute_pbo_cscv`) with oldest-
     residue slicing and micro-grid diagnostics.
  4. Connected Parameter Plateau Geometric Center selector (`select_plateau_center`).
  5. Newey-West HAC / phase-averaged factor IC, sparse event expectancy, passive
     benchmark gate G8, full kernel + RiskAllocator G7 parity, and 8-gate evaluation.
"""

from __future__ import annotations

from alphaloop.platform.core_engine.validation import (
    EULER_MASCHERONI,
    BenchmarkGateReport,
    CPCVReport,
    CPCVSplit,
    DSRReport,
    FoldFitEvalFn,
    FoldResult,
    GateVerdict,
    HACFactorICReport,
    HACMeanStats,
    InstitutionalGateOutcome,
    InstitutionalGateReport,
    NestedCPCVReport,
    NestedCPCVSplitOutcome,
    ParityReport,
    PBOReport,
    PlateauPoint,
    PlateauSelectionReport,
    SparseEventExpectancyReport,
    compute_buy_and_hold_returns,
    compute_effective_independent_trials,
    compute_hac_factor_ic,
    compute_pbo_cscv,
    compute_psr_dsr,
    compute_sparse_event_expectancy,
    contiguous_segments,
    evaluate_8_gates,
    evaluate_benchmark_gate,
    evaluate_cpcv_returns,
    fast_mean,
    fast_median,
    fast_moments,
    fast_pearson_corr,
    fast_quantile,
    fast_spearman_corr,
    fast_var_std,
    generate_cpcv_splits,
    generate_walk_forward_splits,
    mid_ranks,
    newey_west_mean_stats,
    norm_cdf,
    norm_ppf,
    run_nested_cpcv,
    select_plateau_center,
    sharpe_from_returns,
    verify_full_parity,
    verify_full_pipeline_parity,
)

__all__ = [
    "EULER_MASCHERONI",
    "norm_cdf",
    "norm_ppf",
    "fast_mean",
    "fast_var_std",
    "fast_moments",
    "fast_median",
    "fast_quantile",
    "sharpe_from_returns",
    "fast_pearson_corr",
    "mid_ranks",
    "fast_spearman_corr",
    "compute_effective_independent_trials",
    "DSRReport",
    "compute_psr_dsr",
    "CPCVSplit",
    "contiguous_segments",
    "generate_cpcv_splits",
    "generate_walk_forward_splits",
    "CPCVReport",
    "evaluate_cpcv_returns",
    "FoldResult",
    "FoldFitEvalFn",
    "NestedCPCVSplitOutcome",
    "NestedCPCVReport",
    "run_nested_cpcv",
    "PBOReport",
    "compute_pbo_cscv",
    "PlateauPoint",
    "PlateauSelectionReport",
    "select_plateau_center",
    "HACMeanStats",
    "newey_west_mean_stats",
    "HACFactorICReport",
    "compute_hac_factor_ic",
    "SparseEventExpectancyReport",
    "compute_sparse_event_expectancy",
    "BenchmarkGateReport",
    "compute_buy_and_hold_returns",
    "evaluate_benchmark_gate",
    "ParityReport",
    "verify_full_parity",
    "verify_full_pipeline_parity",
    "GateVerdict",
    "InstitutionalGateOutcome",
    "InstitutionalGateReport",
    "evaluate_8_gates",
]
