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
        self.assertEqual(self.ledger.count_family_trials("funding_carry:hyperliquid<->gate"), 1)

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

    def test_zero_candle_close_lookahead_at_decision_hour(self) -> None:
        """Architect P0-1: mutating c.close on decision bar t must have ZERO effect on entry decision or entry price."""
        start = (1_700_000_000_000 // HOUR_MS) * HOUR_MS
        hours = 60

        def _run_with_close_shock(shock_mult: float) -> tuple[int, float, float]:
            store = MarketDataStore(self.root / f"lh_{shock_mult}.sqlite")
            fa, fb, ca, cb = [], [], [], []
            for h in range(hours):
                ts = start + h * HOUR_MS
                fa.append(FundingRateBar("hyperliquid", "SOL", ts, 0.0004, 1.0, PROVENANCE_REAL))
                fb.append(FundingRateBar("gate", "SOL", ts, 0.0, 1.0, PROVENANCE_REAL))
                # Entry happens at h == 24; shock c.close at h == 24 massively while keeping c.open identical!
                cl_a = 100.0 * shock_mult if h == 24 else 100.0
                hi_a = max(100.0, cl_a)
                lo_a = min(100.0, cl_a)
                ca.append(PriceBar("hyperliquid", "SOL", ts, 100.0, hi_a, lo_a, cl_a, 1e6, PROVENANCE_REAL))
                cb.append(PriceBar("gate", "SOL", ts, 100.0, 100.0, 100.0, 100.0, 1e6, PROVENANCE_REAL))
            store.upsert_funding(fa)
            store.upsert_funding(fb)
            store.upsert_candles(ca)
            store.upsert_candles(cb)
            cfg = StrategyConfig(
                strategy_id="t-open-only",
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
                entry_spread_apr_pct=15.0,
                exit_spread_apr_pct=2.0,
                lookback_hours=24,
                max_basis_entry_bps=30.0,
                stop_loss_basis_bps=250.0,
                fee_a_bps=0.0,
                fee_b_bps=0.0,
                slippage_per_leg_bps=0.0,
                rebalance_drag_bps_per_month=0.0,
            )
            res = FundingCarryBacktester(store).run(cfg, start_ms=start, end_ms=start + (hours - 1) * HOUR_MS)
            store.close()
            self.assertEqual(res.metrics.trade_count, 1)
            return res.trades[0].entry_ts_ms, res.trades[0].basis_pnl_usd, res.trades[0].net_pnl_usd

        e1, b1, n1 = _run_with_close_shock(1.0)
        e2, b2, n2 = _run_with_close_shock(1.10)
        self.assertEqual(e1, e2)
        self.assertAlmostEqual(b1, b2, places=6)
        self.assertAlmostEqual(n1, n2, places=6)

    def test_margin_liquidation_trigger_and_g2_rejection(self) -> None:
        """Architect P1-1: 48% adverse rally at 5.0x per-leg leverage MUST trigger MARGIN_LIQUIDATION, realize loss & fail G2."""
        start = (1_700_000_000_000 // HOUR_MS) * HOUR_MS
        hours = 80
        fa, fb, ca, cb = [], [], [], []
        for h in range(hours):
            ts = start + h * HOUR_MS
            fa.append(FundingRateBar("hyperliquid", "WLD", ts, 0.0005, 1.0, PROVENANCE_REAL))
            fb.append(FundingRateBar("gate", "WLD", ts, 0.0, 1.0, PROVENANCE_REAL))
            px = 100.0 if h < 30 else 145.0
            hi = 148.0 if h == 30 else px
            ca.append(PriceBar("hyperliquid", "WLD", ts, px, hi, px, px, 1e6, PROVENANCE_REAL))
            cb.append(PriceBar("gate", "WLD", ts, px, hi, px, px, 1e6, PROVENANCE_REAL))
        self.store.upsert_funding(fa)
        self.store.upsert_funding(fb)
        self.store.upsert_candles(ca)
        self.store.upsert_candles(cb)

        cfg = StrategyConfig(
            strategy_id="t-liq",
            version="v1",
            family="unit-liq",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="gate",
            mode="perp_perp",
            symbols=["WLD"],
            capital_usd=1000.0,
            leverage=5.0,
            max_positions=1,
            per_position_weight=1.0,
            entry_spread_apr_pct=15.0,
            exit_spread_apr_pct=2.0,
            lookback_hours=24,
        )
        res = self.bt.run(cfg, start_ms=start, end_ms=start + (hours - 1) * HOUR_MS)
        self.assertGreaterEqual(res.metrics.liquidation_count, 1)
        liq_trades = [t for t in res.trades if t.exit_reason == "MARGIN_LIQUIDATION"]
        self.assertTrue(liq_trades)
        # Red-Team #3: liquidation must incur real realized penalty (not $0!)
        self.assertLess(liq_trades[0].net_pnl_usd, -100.0)
        verdict, _, _ = self.verifier.verify(cfg, start_ms=start, end_ms=start + (hours - 1) * HOUR_MS)
        g2 = next(c for c in verdict.checks if c.gate_id == "G2")
        self.assertFalse(g2.passed)

    def test_missing_candle_short_leg_liquidation_proxy_and_cooldown(self) -> None:
        """Red-Team #3 BLOCKER: when Lighter (venue_a) has ZERO candles, Gate (venue_b) +30% rally MUST liquidate short_a at 4x."""
        start = (1_700_000_000_000 // HOUR_MS) * HOUR_MS
        hours = 100
        fa, fb, cb = [], [], []
        for h in range(hours):
            ts = start + h * HOUR_MS
            fa.append(FundingRateBar("lighter", "ENA", ts, 0.0005, 1.0, PROVENANCE_REAL))
            fb.append(FundingRateBar("gate", "ENA", ts, 0.0, 1.0, PROVENANCE_REAL))
            px = 1.00 if h < 30 else 1.30
            hi = 1.32 if h == 30 else px * 1.005
            lo = px * 0.995
            cb.append(PriceBar("gate", "ENA", ts, px, hi, lo, px, 1e6, PROVENANCE_REAL))
        self.store.upsert_funding(fa)
        self.store.upsert_funding(fb)
        # Intentionally upsert ONLY venue_b candles (simulating Lighter funding-only!)
        self.store.upsert_candles(cb)

        cfg = StrategyConfig(
            strategy_id="t-lt-proxy-liq",
            version="v1",
            family="lt-proxy",
            status="DRAFT",
            venue_a="lighter",
            venue_b="gate",
            mode="perp_perp",
            symbols=["ENA"],
            capital_usd=1000.0,
            leverage=4.0,  # 22.5% trigger -> +32% rally MUST liquidate short_a via venue_b proxy!
            max_positions=1,
            per_position_weight=1.0,
            entry_spread_apr_pct=15.0,
            exit_spread_apr_pct=2.0,
            lookback_hours=24,
        )
        res = self.bt.run(cfg, start_ms=start, end_ms=start + (hours - 1) * HOUR_MS)
        self.assertGreaterEqual(res.metrics.liquidation_count, 1)
        self.assertEqual(res.trades[0].exit_reason, "MARGIN_LIQUIDATION")
        self.assertLess(res.trades[0].net_pnl_usd, -100.0)
        # Cooldown prevents immediate re-entry within 24h of liquidation!
        if len(res.trades) >= 2:
            self.assertGreaterEqual(res.trades[1].entry_ts_ms - res.trades[0].exit_ts_ms, 24 * HOUR_MS)

    def test_dsr_family_rename_and_version_bump_cannot_bypass_trial_counter(self) -> None:
        """Red-Team #10: renaming family/version/notes cannot reset trial #1 or bypass DSR dedup."""
        start = (1_700_000_000_000 // HOUR_MS) * HOUR_MS
        hours = 720
        fa, fb, ca, cb = [], [], [], []
        for h in range(hours):
            ts = start + h * HOUR_MS
            # Alternating 60h positive (+43.8% APR) / 60h negative (-8.8% APR) waves -> 6 trades & passes DQ-06!
            active = (h % 120) < 60
            fa.append(FundingRateBar("hyperliquid", "BTC", ts, (0.00005 if active else -0.00001), 1.0, PROVENANCE_REAL))
            fb.append(FundingRateBar("gate", "BTC", ts, 0.0, 1.0, PROVENANCE_REAL))
            px = 100.0 + (h % 5) * 0.2
            ca.append(PriceBar("hyperliquid", "BTC", ts, px, px * 1.003, px * 0.997, px, 1e6, PROVENANCE_REAL))
            cb.append(PriceBar("gate", "BTC", ts, px, px * 1.003, px * 0.997, px, 1e6, PROVENANCE_REAL))
        self.store.upsert_funding(fa)
        self.store.upsert_funding(fb)
        self.store.upsert_candles(ca)
        self.store.upsert_candles(cb)

        cfg1 = StrategyConfig(
            strategy_id="s1", version="v1", family="fake_fam_1", status="DRAFT",
            venue_a="hyperliquid", venue_b="gate", mode="perp_perp", symbols=["BTC"],
            entry_spread_apr_pct=10.0, exit_spread_apr_pct=2.0, min_hold_hours=12, exit_confirm_hours=4, lookback_hours=12,
        )
        # Same economic params, renamed family/version/notes -> MUST deduplicate (still trial #1)
        cfg2 = StrategyConfig.from_dict({**cfg1.to_dict(), "strategy_id": "s2", "version": "v9", "family": "fake_fam_2", "notes": "renamed"})
        # Different economic param with renamed family -> MUST increment to trial #2 under canonical family!
        cfg3 = StrategyConfig.from_dict({**cfg1.to_dict(), "strategy_id": "s3", "version": "v3", "family": "fake_fam_3", "entry_spread_apr_pct": 12.0})

        v1, _, _ = self.verifier.verify(cfg1, start_ms=start, end_ms=start + (hours - 1) * HOUR_MS)
        v2, _, _ = self.verifier.verify(cfg2, start_ms=start, end_ms=start + (hours - 1) * HOUR_MS)
        v3, _, _ = self.verifier.verify(cfg3, start_ms=start, end_ms=start + (hours - 1) * HOUR_MS)
        self.assertEqual(v1.trial_number, 1)
        self.assertEqual(v2.trial_number, 1)
        self.assertEqual(v3.trial_number, 2)


if __name__ == "__main__":
    unittest.main()
