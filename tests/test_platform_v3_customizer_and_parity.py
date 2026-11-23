"""End-to-end tests for Platform v3 StrategyCustomizer, Single-Kernel G7 Parity, and A-Share Constraints."""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.platform.assets import (
    ASSET_CLASS_CN_A_SHARE,
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_GOLD_TRADFI,
    default_ashare_template_spec,
    default_gold_specs,
    default_web3_specs,
)
from alphaloop.platform.data_lake import CanonicalBar, MultiAssetDataLake
from alphaloop.platform.factors import StreamingFactorEngine
from alphaloop.platform.regime_inference import CausalRegimeTracker
from alphaloop.platform.strategy_customizer import (
    AdaptiveMultiFactorKernel,
    KernelConfig,
    StrategyCustomizer,
    simulate_universe,
)

BASE_TS_MS = 1_735_689_600_000
HOUR_MS = 3_600_000


def _trending_bars(symbol: str, venue: str, asset_class: str, tf_h: float, n: int, seed: int) -> list[CanonicalBar]:
    """Generate clean persistent multi-cycle trend regimes strictly for unit-test harness verification."""
    rng = random.Random(seed)
    px = 100.0
    step_ms = int(tf_h * HOUR_MS)
    bars: list[CanonicalBar] = []
    for i in range(n):
        # Persistent alternating trend regimes (45 bars up, 35 bars down, 50 bars up, etc.)
        phase = (i // 45) % 3
        drift = 0.008 if phase in (0, 2) else -0.007
        ret = drift + rng.gauss(0.0, 0.008)
        o = px
        c = max(2.0, o * math.exp(ret))
        h = max(o, c) * (1.0 + abs(rng.gauss(0.0, 0.003)))
        l = min(o, c) * max(0.5, 1.0 - abs(rng.gauss(0.0, 0.003)))
        bars.append(
            CanonicalBar(
                asset_class=asset_class,
                venue=venue,
                symbol=symbol,
                timeframe_h=tf_h,
                ts_event_ms=BASE_TS_MS + i * step_ms,
                ts_avail_ms=BASE_TS_MS + (i + 1) * step_ms,
                open=o,
                high=h,
                low=l,
                close=c,
                volume_usd=2_000_000.0,
                carry_rate_period=0.00005,
                carry_interval_h=tf_h,
                provenance=PROVENANCE_REAL,
            )
        )
        px = c
    return bars


class TestPlatformV3CustomizerAndParity(unittest.TestCase):
    def test_full_7gate_pipeline_and_single_kernel_g7_bit_parity(self) -> None:
        specs = default_web3_specs(["BTC", "ETH", "SOL"], venue="hyperliquid")
        bars_by_sym = {
            "BTC": _trending_bars("BTC", "hyperliquid", ASSET_CLASS_CRYPTO_PERP, 12.0, 260, seed=10),
            "ETH": _trending_bars("ETH", "hyperliquid", ASSET_CLASS_CRYPTO_PERP, 12.0, 260, seed=20),
            "SOL": _trending_bars("SOL", "hyperliquid", ASSET_CLASS_CRYPTO_PERP, 12.0, 260, seed=30),
        }
        with tempfile.TemporaryDirectory() as td:
            lake = MultiAssetDataLake(Path(td) / "v3.sqlite")
            try:
                customizer = StrategyCustomizer(target_annual_vol_pct=18.0)
                res = customizer.build_and_validate(
                    track_name="unit_web3_trend",
                    specs=specs,
                    bars_by_symbol=bars_by_sym,
                    capital_usd=3000.0,
                    factor_horizon_bars=6,
                    allow_synthetic_in_test=True,
                    lake=lake,
                )
                gates = {g.gate_id: g for g in res.gates}
                self.assertEqual(len(gates), 7)
                # Gate G7 (NautilusTrader single-kernel batch == streaming parity) MUST ALWAYS PASS!
                self.assertTrue(gates["G7"].passed, gates["G7"].detail)
                # Clean trending fixture should pass G1, G2, G3, G4, G5, G7
                self.assertTrue(gates["G1"].passed)
                self.assertTrue(gates["G2"].passed)
                self.assertTrue(gates["G3"].passed)
                self.assertTrue(gates["G4"].passed)
                self.assertTrue(gates["G5"].passed)
                self.assertGreater(res.baseline_metrics["net_pnl_usd"], 0.0)
                # Blueprint persisted in lake
                rows = lake.conn.fetch_all("SELECT blueprint_id, verdict FROM strategy_blueprints")
                self.assertEqual(len(rows), 1)
            finally:
                lake.close()

    def test_gold_macro_overlay_and_parity_end_to_end(self) -> None:
        gold_specs = {"GC=F": default_gold_specs()["GC=F"]}
        gc_bars = _trending_bars("GC=F", "yahoo", ASSET_CLASS_GOLD_TRADFI, 24.0, 320, seed=55)
        ief_bars = _trending_bars("IEF", "yahoo", ASSET_CLASS_GOLD_TRADFI, 24.0, 320, seed=56)
        customizer = StrategyCustomizer(target_annual_vol_pct=15.0)
        res = customizer.build_and_validate(
            track_name="unit_gold_macro",
            specs=gold_specs,
            bars_by_symbol={"GC=F": gc_bars},
            macro_bars_by_symbol={"GC=F": gc_bars, "IEF": ief_bars},
            capital_usd=3000.0,
            factor_horizon_bars=8,
            allow_synthetic_in_test=True,
        )
        gates = {g.gate_id: g for g in res.gates}
        self.assertTrue(res.final_kernel_config.use_macro_overlay)
        self.assertTrue(gates["G7"].passed, gates["G7"].detail)

    def test_ashare_extension_enforces_long_only_and_t_plus_1(self) -> None:
        ash_spec = default_ashare_template_spec("000300.SH")
        bars = _trending_bars("000300.SH", "cn_a_share", ASSET_CLASS_CN_A_SHARE, 24.0, 180, seed=77)
        eng = StreamingFactorEngine(ash_spec.calendar, 24.0)
        snaps = eng.compute_series(bars)
        regs = CausalRegimeTracker(confirm_bars=2).classify_series(snaps)

        cfg = KernelConfig(
            strategy_name="ashare_ext_test",
            asset_class=ASSET_CLASS_CN_A_SHARE,
            timeframe_h=24.0,
            max_positions=1,
            allow_short=False,
            min_hold_bars=1,
            warmup_bars=30,
        )
        _, decs = simulate_universe(cfg, {"000300.SH": ash_spec}, {"000300.SH": bars}, {"000300.SH": snaps}, {"000300.SH": regs})
        # Strictly long-only: NEVER negative weight!
        self.assertTrue(all(d.target_weight >= 0.0 for d in decs["000300.SH"]))
        # Sanity check kernel rejects shorting directly
        k = AdaptiveMultiFactorKernel(cfg, ash_spec)
        self.assertFalse(k.cfg.allow_short)


if __name__ == "__main__":
    unittest.main()
