"""Unit & adversarial tests for Platform v3 Regime Inference and AFML Anti-Overfitting Math."""

from __future__ import annotations

import math
import random
import unittest

from alphaloop.platform.assets import (
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_GOLD_TRADFI,
    CALENDAR_CRYPTO_24_7,
    CALENDAR_GOLD_TRADFI_23_5,
    COST_GOLD_FUTURES_ETF,
    COST_HYPERLIQUID_PERP,
    AssetSpec,
    CostModel,
)
from alphaloop.platform.data_lake import CanonicalBar
from alphaloop.platform.factors import StreamingFactorEngine
from alphaloop.platform.regime_inference import (
    ARCHETYPE_DEFENSIVE_OBSERVE,
    ARCHETYPE_GOLD_MACRO_TREND,
    AutoParameterInferrer,
    MarketRegimeProfiler,
)
from alphaloop.platform.validation import (
    PlateauPoint,
    compute_pbo_cscv,
    compute_psr_dsr,
    evaluate_cpcv_returns,
    generate_cpcv_splits,
    norm_cdf,
    norm_ppf,
    select_plateau_center,
)

BASE_TS_MS = 1_735_689_600_000
HOUR_MS = 3_600_000


def _synth_series(n: int, tf_h: float, vol: float, trend_drift: float, asset_class: str, sym: str, seed: int = 11) -> list[CanonicalBar]:
    rng = random.Random(seed)
    px = 100.0
    step_ms = int(tf_h * HOUR_MS)
    out: list[CanonicalBar] = []
    for i in range(n):
        ret = trend_drift + rng.gauss(0.0, vol)
        o = px
        c = max(1.0, o * math.exp(ret))
        h = max(o, c) * (1.0 + abs(rng.gauss(0.0, vol * 0.3)))
        l = min(o, c) * max(0.5, 1.0 - abs(rng.gauss(0.0, vol * 0.3)))
        out.append(
            CanonicalBar(
                asset_class=asset_class,
                venue="test",
                symbol=sym,
                timeframe_h=tf_h,
                ts_event_ms=BASE_TS_MS + i * step_ms,
                ts_avail_ms=BASE_TS_MS + (i + 1) * step_ms,
                open=o,
                high=h,
                low=l,
                close=c,
                volume_usd=500_000.0,
            )
        )
        px = c
    return out


class TestPlatformV3InferenceAndValidation(unittest.TestCase):
    def test_norm_cdf_and_ppf_accuracy(self) -> None:
        for z in (-2.5, -1.0, 0.0, 1.0, 1.96, 2.576):
            p = norm_cdf(z)
            self.assertAlmostEqual(norm_ppf(p), z, places=5)

    def test_dsr_penalizes_multiple_testing_and_fat_tails(self) -> None:
        rng = random.Random(42)
        rets = [0.002 + rng.gauss(0.0, 0.01) for _ in range(250)]
        r_single = compute_psr_dsr(rets, periods_per_year=252.0, n_trials=1)
        r_many = compute_psr_dsr(rets, periods_per_year=252.0, n_trials=500)
        self.assertGreater(r_single.dsr, r_many.dsr)
        self.assertGreater(r_many.sr_star_annualized, r_single.sr_star_annualized)

    def test_cpcv_purging_and_embargo_invariants(self) -> None:
        splits = generate_cpcv_splits(n_obs=120, n_groups=6, k_test=2, purge_bars=3, embargo_bars=3)
        self.assertEqual(len(splits), 15)  # C(6, 2) = 15
        for sp in splits:
            train_s = set(sp.train_indices)
            test_s = set(sp.test_indices)
            # Zero train/test overlap AND zero train index within purge/embargo window of any test index
            self.assertEqual(len(train_s & test_s), 0)
            for ti in test_s:
                for offset in (-3, -2, -1, 1, 2, 3):
                    self.assertNotIn(ti + offset, train_s)

        # Positive drift stream passes CPCV; pure noise does not reliably hold high OOS SR
        rng = random.Random(7)
        strong = [0.006 + rng.gauss(0.0, 0.008) for _ in range(180)]
        rep = evaluate_cpcv_returns(strong, periods_per_year=252.0)
        self.assertEqual(rep.n_splits, 15)
        self.assertGreater(rep.median_oos_sharpe, 2.0)
        self.assertEqual(rep.pct_positive_splits, 100.0)

    def test_pbo_detects_overfit_noise_matrix_vs_true_edge(self) -> None:
        rng = random.Random(101)
        # 16 zero-mean pure-noise strategies (exact zero true drift -> selection bias yields PBO >= 0.50)
        noise_mat = []
        for _ in range(16):
            raw = [rng.gauss(0.0, 0.01) for _ in range(160)]
            mu = sum(raw) / len(raw)
            noise_mat.append([r - mu for r in raw])
        pbo_noise = compute_pbo_cscv(noise_mat, periods_per_year=252.0, n_subsamples=8)
        self.assertEqual(pbo_noise.n_combinations, 70)
        self.assertGreaterEqual(pbo_noise.pbo, 0.50)

        # Matrix where candidate 0 has genuine persistent edge across all subperiods -> PBO ~ 0.0
        true_mat = [[0.008 + rng.gauss(0.0, 0.008) for _ in range(160)]] + [
            [-0.002 + rng.gauss(0.0, 0.01) for _ in range(160)] for _ in range(8)
        ]
        pbo_true = compute_pbo_cscv(true_mat, periods_per_year=252.0, n_subsamples=8)
        self.assertLessEqual(pbo_true.pbo, 0.05)

    def test_plateau_selector_rejects_isolated_spike_and_picks_centroid(self) -> None:
        # Hostile case 1: Single isolated spike surrounded by negative losses -> MUST FAIL!
        spike_grid = [
            PlateauPoint(ix, iy, f"p_{ix}_{iy}", {"x": ix, "y": iy}, 4.5 if (ix, iy) == (0, 0) else -0.8, 40.0 if (ix, iy) == (0, 0) else -10.0, 5.0, 500.0 if (ix, iy) == (0, 0) else -100.0)
            for ix in range(3)
            for iy in range(3)
        ]
        spike_rep = select_plateau_center(spike_grid)
        self.assertFalse(spike_rep.passed_plateau_gate)
        self.assertEqual(spike_rep.largest_plateau_size, 1)

        # Robust case 2: 9/9 positive plateau with a corner peak -> picks geometric center (1, 1), NOT corner (2, 2)!
        flat_grid = [
            PlateauPoint(
                ix,
                iy,
                f"p_{ix}_{iy}",
                {"x": ix, "y": iy},
                2.1 if (ix, iy) == (2, 2) else 1.8,
                25.0,
                4.0,
                300.0,
            )
            for ix in range(3)
            for iy in range(3)
        ]
        flat_rep = select_plateau_center(flat_grid)
        self.assertTrue(flat_rep.passed_plateau_gate)
        self.assertEqual(flat_rep.selected_label, "p_1_1")

    def test_auto_parameter_inferrer_vol_targeting_and_cost_feasibility(self) -> None:
        # High-vol crypto vs lower-vol gold -> Carver vol-targeting MUST assign lower leverage to high-vol crypto!
        high_vol_spec = AssetSpec("SOL", ASSET_CLASS_CRYPTO_PERP, "hyperliquid", CALENDAR_CRYPTO_24_7, COST_HYPERLIQUID_PERP, max_leverage_cap=2.0)
        gold_spec = AssetSpec("GC=F", ASSET_CLASS_GOLD_TRADFI, "yahoo", CALENDAR_GOLD_TRADFI_23_5, COST_GOLD_FUTURES_ETF, max_leverage_cap=2.0)

        sol_bars = _synth_series(150, 12.0, vol=0.045, trend_drift=0.002, asset_class=ASSET_CLASS_CRYPTO_PERP, sym="SOL", seed=1)
        gc_bars = _synth_series(150, 24.0, vol=0.009, trend_drift=0.001, asset_class=ASSET_CLASS_GOLD_TRADFI, sym="GC=F", seed=2)

        profiler = MarketRegimeProfiler()
        inferrer = AutoParameterInferrer(target_annual_vol_pct=18.0)

        sol_snaps = StreamingFactorEngine(CALENDAR_CRYPTO_24_7, 12.0).compute_series(sol_bars)
        gc_snaps = StreamingFactorEngine(CALENDAR_GOLD_TRADFI_23_5, 24.0).compute_series(gc_bars)

        sol_prof, _ = profiler.profile_asset(high_vol_spec, sol_bars, sol_snaps)
        gc_prof, _ = profiler.profile_asset(gold_spec, gc_bars, gc_snaps)

        sol_inf = inferrer.infer_from_profiles("sol_track", [high_vol_spec], [sol_prof])
        gc_inf = inferrer.infer_from_profiles("gold_track", [gold_spec], [gc_prof])

        self.assertLess(sol_inf.max_symbol_leverage, gc_inf.max_symbol_leverage)
        self.assertEqual(gc_inf.recommended_archetype, ARCHETYPE_GOLD_MACRO_TREND)
        self.assertTrue(gc_inf.use_macro_overlay)
        self.assertGreaterEqual(len(sol_inf.inference_trace), 4)

        # Hostile cost-infeasible asset (absurd 500bp round-trip friction on 10bp moves) -> DEFENSIVE_OBSERVE_ONLY
        expensive_spec = AssetSpec(
            "TINY",
            ASSET_CLASS_CRYPTO_PERP,
            "bad_dex",
            CALENDAR_CRYPTO_24_7,
            CostModel(maker_fee_bps=200.0, taker_fee_bps=250.0, slippage_bps_per_leg=100.0),
            max_leverage_cap=1.0,
        )
        tiny_bars = _synth_series(100, 1.0, vol=0.0005, trend_drift=0.0, asset_class=ASSET_CLASS_CRYPTO_PERP, sym="TINY", seed=3)
        tiny_snaps = StreamingFactorEngine(CALENDAR_CRYPTO_24_7, 1.0).compute_series(tiny_bars)
        tiny_prof, _ = profiler.profile_asset(expensive_spec, tiny_bars, tiny_snaps)
        tiny_inf = inferrer.infer_from_profiles("tiny_track", [expensive_spec], [tiny_prof])
        self.assertFalse(tiny_inf.cost_feasible)
        self.assertEqual(tiny_inf.recommended_archetype, ARCHETYPE_DEFENSIVE_OBSERVE)


if __name__ == "__main__":
    unittest.main()
