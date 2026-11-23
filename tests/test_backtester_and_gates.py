"""Unit & adversarial tests for causal backtester, anti-lookahead, hysteresis, and 6-Gate verifier."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from alphaloop.core.models import FundingRateBar, PriceBar, StrategyConfig
from alphaloop.core.provenance import PROVENANCE_REAL, PROVENANCE_SYNTHETIC
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import HOUR_MS, FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier, TrialLedger


def _seed_synthetic_pair(
    store: MarketDataStore,
    symbol: str,
    hours: int,
    rate_a_1h: float,
    rate_b_1h: float,
    base_price: float = 100.0,
    basis_drift_end_pct: float = 0.0,
    provenance: str = PROVENANCE_SYNTHETIC,
) -> tuple[int, int]:
    start = 1_700_000_000_000
    start = (start // HOUR_MS) * HOUR_MS
    fa, fb, ca, cb = [], [], [], []
    for h in range(hours):
        ts = start + h * HOUR_MS
        fa.append(FundingRateBar("hyperliquid", symbol, ts, rate_a_1h, 1.0, provenance))
        fb.append(FundingRateBar("gate", symbol, ts, rate_b_1h, 1.0, provenance))
        frac = h / max(1, hours - 1)
        px_a = base_price * (1.0 + basis_drift_end_pct * frac)
        px_b = base_price
        ca.append(PriceBar("hyperliquid", symbol, ts, px_a, px_a, px_a, px_a, 1e6, provenance))
        cb.append(PriceBar("gate", symbol, ts, px_b, px_b, px_b, px_b, 1e6, provenance))
    store.upsert_funding(fa)
    store.upsert_funding(fb)
    store.upsert_candles(ca)
    store.upsert_candles(cb)
    return start, start + (hours - 1) * HOUR_MS


class TestBacktesterAndGates(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = MarketDataStore(self.root / "market.sqlite")
        self.ledger = TrialLedger(self.root / "trials.sqlite")
        self.bt = FundingCarryBacktester(self.store)
        self.verifier = SixGateVerifier(self.bt, self.ledger)

    def tearDown(self) -> None:
        self.store.close()
        self.ledger.close()
        self.tmp.cleanup()

    def test_zero_lookahead_does_not_collect_entry_hour_funding(self) -> None:
        """Entering at hour t must NOT accrue funding from ts == entry_ts_ms."""
        # 50 hours: HL pays +0.0005/hr (438% APR), Gate pays 0 -> enter at warmup hour 24, collect 25 hours (25..49)
        start, end = _seed_synthetic_pair(self.store, "SOL", hours=50, rate_a_1h=0.0005, rate_b_1h=0.0)
        cfg = StrategyConfig(
            strategy_id="t-lookahead",
            version="v1",
            family="unit",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="gate",
            mode="perp_perp",
            symbols=["SOL"],
            capital_usd=1000.0,
            leverage=2.0,
            max_positions=1,
            per_position_weight=1.0,
            entry_spread_apr_pct=20.0,
            exit_spread_apr_pct=4.0,
            lookback_hours=24,
            min_hold_hours=24,
            fee_a_bps=0.0,
            fee_b_bps=0.0,
            slippage_per_leg_bps=0.0,
            rebalance_drag_bps_per_month=0.0,
        )
        res = self.bt.run(cfg, start_ms=start, end_ms=end)
        self.assertEqual(res.metrics.trade_count, 1)
        tr = res.trades[0]
        self.assertEqual(tr.entry_ts_ms, start + 24 * HOUR_MS)
        self.assertEqual(tr.hold_hours, 25.0)
        # Exactly 25 hourly settlements accrued, NOT 26!
        expected_gross = 1000.0 * 0.0005 * 25
        self.assertAlmostEqual(tr.gross_funding_usd, expected_gross, places=4)

    def test_real_basis_pnl_and_basis_stop_loss(self) -> None:
        """Adverse basis widening on SHORT_A_LONG_B (A rises 5% above B) triggers BASIS_STOP."""
        start, end = _seed_synthetic_pair(
            self.store, "HYPE", hours=80, rate_a_1h=0.0001, rate_b_1h=0.0, basis_drift_end_pct=0.05
        )
        cfg = StrategyConfig(
            strategy_id="t-basis",
            version="v1",
            family="unit",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="gate",
            mode="perp_perp",
            symbols=["HYPE"],
            capital_usd=1000.0,
            leverage=2.0,
            max_positions=1,
            per_position_weight=1.0,
            entry_spread_apr_pct=15.0,
            exit_spread_apr_pct=2.0,
            lookback_hours=24,
            stop_loss_basis_bps=150.0,
            fee_a_bps=1.5,
            fee_b_bps=2.0,
            slippage_per_leg_bps=1.0,
            rebalance_drag_bps_per_month=0.0,
        )
        res = self.bt.run(cfg, start_ms=start, end_ms=end)
        self.assertGreaterEqual(res.metrics.trade_count, 1)
        self.assertEqual(res.trades[0].exit_reason, "BASIS_STOP")
        self.assertLess(res.trades[0].basis_pnl_usd, 0.0)
        self.assertEqual(res.basis_coverage_pct, 100.0)

    def test_synthetic_data_blocked_by_g1_provenance_gate(self) -> None:
        """G1 MUST reject synthetic data unless explicitly allowed in unit test flag."""
        start, end = _seed_synthetic_pair(self.store, "BTC", hours=200, rate_a_1h=0.0001, rate_b_1h=-0.00005)
        cfg = StrategyConfig(
            strategy_id="t-g1-block",
            version="v1",
            family="unit-prov",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="gate",
            mode="perp_perp",
            symbols=["BTC"],
            capital_usd=3000.0,
            entry_spread_apr_pct=15.0,
            exit_spread_apr_pct=4.0,
        )
        verdict, _, _ = self.verifier.verify(cfg, start_ms=start, end_ms=end, allow_synthetic_in_unit_test=False)
        g1 = next(c for c in verdict.checks if c.gate_id == "G1")
        self.assertFalse(g1.passed)
        self.assertFalse(verdict.overall_pass)
        self.assertEqual(self.ledger.count_family_trials("unit-prov"), 1)

    def test_atb_buggy_rules_reproduce_fee_churn_loss(self) -> None:
        """ATB H6 inverted hysteresis (enter >= 10.95% APR, exit < 54.75% APR x3) loses money on positive carry."""
        # Constant +20% APR funding: profitable with hysteresis, catastrophic under ATB buggy rules!
        rate_20apr = 0.20 / (365.0 * 24.0)
        start, end = _seed_synthetic_pair(self.store, "ETH", hours=300, rate_a_1h=rate_20apr, rate_b_1h=0.0, provenance=PROVENANCE_REAL)
        buggy = StrategyConfig(
            strategy_id="atb-bug",
            version="v0",
            family="churn",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="gate",
            mode="spot_perp",
            symbols=["ETH"],
            capital_usd=3000.0,
            leverage=1.0,
            max_positions=1,
            per_position_weight=1.0,
            lookback_hours=24,
            fee_a_bps=5.0,
            fee_b_bps=10.0,
            slippage_per_leg_bps=1.0,
            use_atb_buggy_rules=True,
        )
        res_bug = self.bt.run(buggy, start_ms=start, end_ms=end)
        self.assertGreater(res_bug.metrics.gross_funding_usd, 0.0)
        self.assertLess(res_bug.metrics.net_pnl_usd, 0.0)
        self.assertGreater(res_bug.metrics.trade_count, 50)


if __name__ == "__main__":
    unittest.main()
