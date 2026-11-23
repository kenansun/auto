"""Tests for atomic rename(2) HotConfigStore, TelemetryMonitor, MultiModelRouter, and AgentEvolutionOrchestrator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from alphaloop.agent_loop.model_router import MultiModelRouter
from alphaloop.agent_loop.orchestrator import AgentEvolutionOrchestrator
from alphaloop.core.config import HotConfigStore
from alphaloop.core.models import FundingRateBar, PriceBar, StrategyConfig
from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import HOUR_MS, FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier, TrialLedger
from alphaloop.harness.telemetry import TelemetryMonitor


class TestHotConfigAndAgentLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = MarketDataStore(self.root / "market.sqlite")
        self.ledger = TrialLedger(self.root / "trials.sqlite")
        self.cfg_store = HotConfigStore(self.root / "state")
        self.bt = FundingCarryBacktester(self.store)
        self.verifier = SixGateVerifier(self.bt, self.ledger)
        self.router = MultiModelRouter()
        self.orch = AgentEvolutionOrchestrator(self.root / "state", self.bt, self.verifier, self.cfg_store, self.router)

    def tearDown(self) -> None:
        self.store.close()
        self.ledger.close()
        self.cfg_store.close()
        self.orch.close()
        self.tmp.cleanup()

    def test_hot_config_publish_rollback_and_human_live_gate(self) -> None:
        cfg_v1 = StrategyConfig(
            strategy_id="carry_alpha",
            version="v1.0",
            family="test",
            status="SHADOW_LIVE",
            venue_a="hyperliquid",
            venue_b="gate",
            mode="perp_perp",
            symbols=["BTC", "ETH"],
            entry_spread_apr_pct=20.0,
            exit_spread_apr_pct=4.0,
        )
        sha1 = self.cfg_store.register_version(cfg_v1, actor="unit", reason="init")
        self.assertEqual(len(sha1), 64)
        self.assertEqual(self.cfg_store.read_active("carry_alpha").version, "v1.0")

        # Autonomous LIVE promotion without human approval token MUST raise PermissionError
        cfg_live = StrategyConfig.from_dict({**cfg_v1.to_dict(), "version": "v2.0", "status": "CANARY_APPROVED"})
        with self.assertRaises(PermissionError):
            self.cfg_store.register_version(cfg_live, actor="agent", reason="unauthorized live attempt")

        # With explicit human token -> succeeds, and rollback restores v1.0 cleanly
        self.cfg_store.register_version(
            cfg_live, actor="owner", reason="manual signoff", human_approval_token="HUMAN_APPROVED:OWNER-001"
        )
        self.assertEqual(self.cfg_store.read_active("carry_alpha").version, "v2.0")

        rolled = self.cfg_store.rollback("carry_alpha", "v1.0", actor="owner", reason="revert test")
        self.assertEqual(rolled.version, "v1.0")
        self.assertEqual(self.cfg_store.read_active("carry_alpha").version, "v1.0")

    def test_cross_family_red_team_independence(self) -> None:
        rt_for_ds = self.router.select_red_team_route("deepseek")
        self.assertNotEqual(rt_for_ds.family, "deepseek")
        rt_for_claude = self.router.select_red_team_route("claude")
        self.assertNotEqual(rt_for_claude.family, "claude")

    def test_closed_loop_agent_detects_churn_and_evolves_strategy(self) -> None:
        """Seed multi-symbol persistent carry wave -> buggy seed triggers CHURN_EXCESS -> Agent fixes & improves."""
        start = (1_700_000_000_000 // HOUR_MS) * HOUR_MS
        hours = 2160
        for sym in ("SOL", "HYPE", "AAVE"):
            fa, fb, ca, cb = [], [], [], []
            for h in range(hours):
                ts = start + h * HOUR_MS
                # Alternating 240h active carry waves (+30% APR spread, inside ATB's 10.95%..54.75% churn trap) and 240h quiet periods (-4% APR)
                active = (h % 480) < 240
                r_a = (0.28 if active else -0.04) / (365.0 * 24.0)
                r_b = (-0.02 if active else 0.0) / (365.0 * 24.0)
                fa.append(FundingRateBar("hyperliquid", sym, ts, r_a, 1.0, PROVENANCE_REAL))
                fb.append(FundingRateBar("gate", sym, ts, r_b, 1.0, PROVENANCE_REAL))
                ca.append(PriceBar("hyperliquid", sym, ts, 100.0, 100.0, 100.0, 100.0, 1e6, PROVENANCE_REAL))
                cb.append(PriceBar("gate", sym, ts, 100.0, 100.0, 100.0, 100.0, 1e6, PROVENANCE_REAL))
            self.store.upsert_funding(fa)
            self.store.upsert_funding(fb)
            self.store.upsert_candles(ca)
            self.store.upsert_candles(cb)

        buggy = StrategyConfig(
            strategy_id="evol_demo",
            version="v0.1",
            family="evol_family",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="gate",
            mode="perp_perp",
            symbols=["SOL", "HYPE", "AAVE"],
            capital_usd=3000.0,
            leverage=2.0,
            max_positions=3,
            per_position_weight=0.33,
            entry_spread_apr_pct=12.0,
            exit_spread_apr_pct=10.0,
            lookback_hours=12,
            fee_a_bps=1.5,
            fee_b_bps=2.0,
            slippage_per_leg_bps=1.0,
            rebalance_drag_bps_per_month=5.0,
            use_atb_buggy_rules=True,
        )
        self.cfg_store.register_version(buggy, actor="seed", reason="buggy seed")
        rec = self.orch.run_cycle(buggy, next_version="v0.2", start_ms=start, end_ms=start + (hours - 1) * HOUR_MS)
        self.assertEqual(rec["feedback"]["anomaly_type"], "CHURN_EXCESS")
        self.assertFalse(rec["candidate_config"]["use_atb_buggy_rules"])
        self.assertGreater(
            rec["candidate_metrics"]["annualized_apr_pct"],
            rec["baseline_metrics"]["annualized_apr_pct"],
        )
        self.assertTrue(rec["gate_verdict"]["overall_pass"])
        self.assertEqual(rec["red_team_verdict"]["decision"], "APPROVE_SHADOW")
        self.assertTrue(rec["promoted_shadow"])
        self.assertEqual(self.cfg_store.read_active("evol_demo").version, "v0.2")
        self.assertEqual(self.cfg_store.read_active("evol_demo").status, "SHADOW_LIVE")

    def test_failed_gates_block_shadow_promotion(self) -> None:
        """Negative-EV market data must fail 6-Gate and prevent HotConfigStore promotion."""
        start = (1_700_000_000_000 // HOUR_MS) * HOUR_MS
        hours = 400
        fa, fb, ca, cb = [], [], [], []
        for h in range(hours):
            ts = start + h * HOUR_MS
            # Zero spread -> fees guarantee negative net PnL
            fa.append(FundingRateBar("hyperliquid", "DOGE", ts, 0.0, 1.0, PROVENANCE_REAL))
            fb.append(FundingRateBar("gate", "DOGE", ts, 0.0, 1.0, PROVENANCE_REAL))
            ca.append(PriceBar("hyperliquid", "DOGE", ts, 0.1, 0.1, 0.1, 0.1, 1e6, PROVENANCE_REAL))
            cb.append(PriceBar("gate", "DOGE", ts, 0.1, 0.1, 0.1, 0.1, 1e6, PROVENANCE_REAL))
        self.store.upsert_funding(fa)
        self.store.upsert_funding(fb)
        self.store.upsert_candles(ca)
        self.store.upsert_candles(cb)

        seed = StrategyConfig(
            strategy_id="zero_edge",
            version="v0.1",
            family="zero_family",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="gate",
            mode="perp_perp",
            symbols=["DOGE"],
        )
        self.cfg_store.register_version(seed, actor="seed", reason="zero edge seed")
        rec = self.orch.run_cycle(seed, next_version="v0.2", start_ms=start, end_ms=start + (hours - 1) * HOUR_MS)
        self.assertFalse(rec["gate_verdict"]["overall_pass"])
        self.assertEqual(rec["red_team_verdict"]["decision"], "REJECT")
        self.assertFalse(rec["promoted_shadow"])
        self.assertEqual(self.cfg_store.read_active("zero_edge").version, "v0.1")


if __name__ == "__main__":
    unittest.main()
