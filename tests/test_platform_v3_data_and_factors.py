"""Unit & hostile-adversarial tests for Platform v3 Multi-Asset Data Lake and Factor Engine."""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.platform.assets import (
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_GOLD_PERP,
    ASSET_CLASS_GOLD_TRADFI,
    CALENDAR_CN_A_SHARE,
    CALENDAR_CRYPTO_24_7,
    CALENDAR_GOLD_TRADFI_23_5,
    COST_CN_A_SHARE,
    COST_HYPERLIQUID_PERP,
    default_ashare_template_spec,
)
from alphaloop.platform.data_lake import (
    CanonicalBar,
    MultiAssetDataLake,
    audit_bars_quality,
    audit_cross_source_divergence,
    resample_canonical_bars,
)
from alphaloop.platform.factors import (
    FactorDirectionAuditor,
    MacroRegimeTimeline,
    StreamingFactorEngine,
    spearman_corr,
)

BASE_TS_MS = 1_735_689_600_000  # 2025-01-01T00:00:00Z
HOUR_MS = 3_600_000
DAY_MS = 86_400_000


def _make_bars(
    n: int,
    symbol: str = "BTC",
    venue: str = "hyperliquid",
    asset_class: str = ASSET_CLASS_CRYPTO_PERP,
    tf_h: float = 12.0,
    drift: float = 0.004,
    vol: float = 0.012,
    seed: int = 42,
    provenance: str = PROVENANCE_REAL,
) -> list[CanonicalBar]:
    rng = random.Random(seed)
    px = 100.0
    step_ms = int(tf_h * HOUR_MS)
    bars: list[CanonicalBar] = []
    for i in range(n):
        ts = BASE_TS_MS + i * step_ms
        # Trending synthetic wave + noise (strictly for unit tests)
        wave = math.sin(i / 18.0) * drift
        ret = wave + drift * 0.4 + rng.gauss(0.0, vol)
        o = px
        c = max(1.0, o * math.exp(ret))
        h = max(o, c) * (1.0 + abs(rng.gauss(0.0, vol * 0.4)))
        l = min(o, c) * max(0.5, 1.0 - abs(rng.gauss(0.0, vol * 0.4)))
        bars.append(
            CanonicalBar(
                asset_class=asset_class,
                venue=venue,
                symbol=symbol,
                timeframe_h=tf_h,
                ts_event_ms=ts,
                ts_avail_ms=ts + step_ms,
                open=o,
                high=h,
                low=l,
                close=c,
                volume_usd=1_000_000.0 * (1.0 + abs(ret) * 20.0),
                carry_rate_period=0.0001,
                carry_interval_h=tf_h,
                provenance=provenance,
            )
        )
        px = c
    return bars


class TestPlatformV3DataAndFactors(unittest.TestCase):
    def test_asset_calendars_and_costs(self) -> None:
        self.assertAlmostEqual(CALENDAR_CRYPTO_24_7.bars_per_year(12.0), 730.0, places=3)
        self.assertAlmostEqual(CALENDAR_GOLD_TRADFI_23_5.bars_per_year(24.0), 252.0, places=3)
        self.assertAlmostEqual(CALENDAR_CN_A_SHARE.bars_per_year(24.0), 242.0, places=3)
        # Round-trip taker cost HL = 2 * (4.5 + 1.5) = 12bp
        self.assertAlmostEqual(COST_HYPERLIQUID_PERP.round_trip_cost_bps("TAKER_TAKER"), 12.0, places=4)
        # A-share includes sell-side stamp duty (5bp)
        self.assertAlmostEqual(COST_CN_A_SHARE.round_trip_cost_bps("TAKER_TAKER"), 20.0, places=4)
        ash = default_ashare_template_spec()
        self.assertFalse(ash.calendar.allow_short)
        self.assertEqual(ash.calendar.t_plus_days, 1)

    def test_data_lake_pit_guard_and_poison_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lake = MultiAssetDataLake(Path(td) / "lake.sqlite")
            try:
                good_bars = _make_bars(20, tf_h=1.0)
                # Hostile bars: lookahead ts_avail < ts_event, NaN close, OHLC envelope violation
                bad_lookahead = CanonicalBar(
                    asset_class=ASSET_CLASS_CRYPTO_PERP,
                    venue="hyperliquid",
                    symbol="BTC",
                    timeframe_h=1.0,
                    ts_event_ms=BASE_TS_MS + 999 * HOUR_MS,
                    ts_avail_ms=BASE_TS_MS + 998 * HOUR_MS,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume_usd=1000.0,
                )
                bad_nan = CanonicalBar(
                    asset_class=ASSET_CLASS_CRYPTO_PERP,
                    venue="hyperliquid",
                    symbol="BTC",
                    timeframe_h=1.0,
                    ts_event_ms=BASE_TS_MS + 1000 * HOUR_MS,
                    ts_avail_ms=BASE_TS_MS + 1001 * HOUR_MS,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=float("nan"),
                    volume_usd=1000.0,
                )
                bad_env = CanonicalBar(
                    asset_class=ASSET_CLASS_CRYPTO_PERP,
                    venue="hyperliquid",
                    symbol="BTC",
                    timeframe_h=1.0,
                    ts_event_ms=BASE_TS_MS + 1001 * HOUR_MS,
                    ts_avail_ms=BASE_TS_MS + 1002 * HOUR_MS,
                    open=100.0,
                    high=99.0,  # high < open -> MUST reject
                    low=98.0,
                    close=100.0,
                    volume_usd=1000.0,
                )
                acc, rej = lake.upsert_bars(good_bars + [bad_lookahead, bad_nan, bad_env])
                self.assertEqual(acc, 20)
                self.assertEqual(rej, 3)

                # Strict Point-in-Time as_of_ms check: bar #5 opens at +5h and is ONLY available at +6h
                visible = lake.load_bars("hyperliquid", "BTC", 1.0, as_of_ms=BASE_TS_MS + 5 * HOUR_MS)
                self.assertEqual(len(visible), 5)
                self.assertTrue(all(b.ts_avail_ms <= BASE_TS_MS + 5 * HOUR_MS for b in visible))

                # JSONL export works cleanly
                out_jsonl = Path(td) / "btc.jsonl"
                self.assertEqual(lake.export_jsonl("hyperliquid", "BTC", 1.0, out_jsonl), 20)
                self.assertEqual(len(out_jsonl.read_text(encoding="utf-8").splitlines()), 20)
            finally:
                lake.close()

    def test_dq_rejects_synthetic_provenance_by_default(self) -> None:
        fake_bars = _make_bars(80, provenance="MOCK_SYNTHETIC")
        rep = audit_bars_quality(fake_bars, CALENDAR_CRYPTO_24_7, allow_synthetic_in_test=False)
        self.assertFalse(rep.passed)
        self.assertFalse(rep.checks[0].passed)

    def test_resample_and_cross_source_divergence(self) -> None:
        bars_1h = _make_bars(24 * 60, symbol="PAXG", asset_class=ASSET_CLASS_GOLD_PERP, tf_h=1.0, drift=0.0005, vol=0.002)
        bars_12h = resample_canonical_bars(bars_1h, 12.0)
        self.assertEqual(len(bars_12h), 120)
        self.assertAlmostEqual(bars_12h[0].carry_rate_period, 12 * 0.0001, places=8)

        # Create aligned TradFi GC=F series with close correlation
        bars_gc = [
            CanonicalBar(
                asset_class=ASSET_CLASS_GOLD_TRADFI,
                venue="yahoo",
                symbol="GC=F",
                timeframe_h=24.0,
                ts_event_ms=b.ts_event_ms,
                ts_avail_ms=b.ts_avail_ms,
                open=b.open * 1.001,
                high=b.high * 1.001,
                low=b.low * 1.001,
                close=b.close * 1.001,
                volume_usd=b.volume_usd,
            )
            for b in resample_canonical_bars(bars_1h, 24.0)
        ]
        div = audit_cross_source_divergence(bars_1h, bars_gc, "PAXG", "GC=F")
        self.assertEqual(div.overlap_days, 60)
        self.assertGreater(div.daily_return_corr, 0.99)
        self.assertEqual(div.arbitration_verdict, "ALIGNED")

    def test_streaming_factor_engine_zero_lookahead_and_macro_pit(self) -> None:
        bars = _make_bars(160, symbol="PAXG", asset_class=ASSET_CLASS_GOLD_PERP, tf_h=12.0)
        # Macro daily series where future jump happens ONLY after day 60
        macro_gc = _make_bars(120, symbol="GC=F", venue="yahoo", asset_class=ASSET_CLASS_GOLD_TRADFI, tf_h=24.0, drift=0.005, vol=0.005, seed=7)
        macro_ief = _make_bars(120, symbol="IEF", venue="yahoo", asset_class=ASSET_CLASS_GOLD_TRADFI, tf_h=24.0, drift=0.003, vol=0.003, seed=8)
        tl = MacroRegimeTimeline({"GC=F": macro_gc, "IEF": macro_ief})

        # Compute full series vs truncated prefix -> snapshots at bar k MUST be bit-identical (zero lookahead!)
        eng_full = StreamingFactorEngine(CALENDAR_CRYPTO_24_7, 12.0)
        full_snaps = eng_full.compute_series(bars, macro_timeline=tl)

        eng_prefix = StreamingFactorEngine(CALENDAR_CRYPTO_24_7, 12.0)
        tl_prefix = MacroRegimeTimeline({"GC=F": macro_gc, "IEF": macro_ief})
        prefix_snaps = eng_prefix.compute_series(bars[:90], macro_timeline=tl_prefix)

        for k in range(90):
            for fname, val in prefix_snaps[k].values.items():
                self.assertAlmostEqual(val, full_snaps[k].values[fname], places=12, msg=f"Lookahead drift in {fname} at bar {k}")

    def test_g2_factor_direction_auditor_rejects_pure_noise_and_flipped_sign(self) -> None:
        rng = random.Random(123)
        bars = _make_bars(800, drift=0.0, vol=0.015, seed=99)
        eng = StreamingFactorEngine(CALENDAR_CRYPTO_24_7, 12.0)
        snaps = eng.compute_series(bars)
        noise_names = [f"pure_noise_{k}" for k in range(5)]
        poisoned_snaps = []
        for i, s in enumerate(snaps):
            vals = dict(s.values)
            vals["zero_const"] = 0.0
            for nn in noise_names:
                vals[nn] = rng.gauss(0.0, 1.0)
            # Deliberately wrong-sign factor (anti-correlated with forward move)
            fwd_hint = (bars[min(len(bars) - 1, i + 5)].close / bars[min(len(bars) - 1, i + 1)].open - 1.0)
            vals["wrong_sign"] = -fwd_hint
            poisoned_snaps.append(
                type(s)(
                    symbol=s.symbol,
                    venue=s.venue,
                    timeframe_h=s.timeframe_h,
                    ts_event_ms=s.ts_event_ms,
                    ts_avail_ms=s.ts_avail_ms,
                    close=s.close,
                    atr_14=s.atr_14,
                    prior_high_40=s.prior_high_40,
                    prior_low_40=s.prior_low_40,
                    prior_high_20=s.prior_high_20,
                    prior_low_20=s.prior_low_20,
                    values=vals,
                )
            )
        auditor = FactorDirectionAuditor(min_abs_ic=0.05, min_monotonicity_rho=0.60, min_t_stat=1.65)
        cands = ["zero_const", "wrong_sign"] + noise_names
        summary = auditor.audit(
            "noise_test",
            {"BTC": bars},
            {"BTC": poisoned_snaps},
            horizon_bars=4,
            candidate_factors=cands,
            priors={c: +1 for c in cands},
        )
        by_name = {r.factor_name: r for r in summary.results}
        self.assertFalse(by_name["zero_const"].passed_g2)
        self.assertFalse(by_name["wrong_sign"].passed_g2)
        self.assertLess(by_name["wrong_sign"].spearman_ic, -0.5)
        noise_rejected = sum(1 for nn in noise_names if not by_name[nn].passed_g2)
        self.assertGreaterEqual(noise_rejected, 4)
        self.assertAlmostEqual(spearman_corr([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
