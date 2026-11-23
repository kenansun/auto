"""Adversarial unit tests for MicroRunSession Dual-Layer Early-Kill, ReadOnlyFleetAuditor security, and Web v2 endpoints."""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from alphaloop.core.models import BBOTick, StrategyConfig
from alphaloop.core.provenance import PROVENANCE_REAL, utc_now_ms
from alphaloop.data.store import MarketDataStore
from alphaloop.fleet.ssh_auditor import ReadOnlyFleetAuditor, validate_safe_path
from alphaloop.harness.micro_runner import MicroRunSession, MicroRunStore
from alphaloop.plugins import FundingCarryPlugin
from alphaloop.web.server import create_server


class TestMicroRunnerAndFleetSecurity(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = MarketDataStore(self.root / "market.sqlite")
        self.mstore = MicroRunStore(self.root / "micro_runs.sqlite")
        self.session = MicroRunSession(self.store, self.mstore)

    def tearDown(self) -> None:
        self.store.close()
        self.mstore.close()
        self.tmp.cleanup()

    def _seed_bbo(self, venue_a: str, venue_b: str, spread_a_bps: float, spread_b_bps: float, n_rounds: int = 15) -> None:
        now = utc_now_ms() - n_rounds * 20_000
        ticks: list[BBOTick] = []
        for i in range(n_rounds):
            ts = now + i * 20_000
            for sym in ("BTC", "ETH"):
                mid = 1000.0
                ba = mid * (1.0 - spread_a_bps / 20000.0)
                aa = mid * (1.0 + spread_a_bps / 20000.0)
                bb = mid * (1.0 - spread_b_bps / 20000.0)
                ab = mid * (1.0 + spread_b_bps / 20000.0)
                ticks.append(BBOTick(venue_a, sym, ts, ba, aa, 5.0, 5.0, 50.0, PROVENANCE_REAL))
                ticks.append(BBOTick(venue_b, sym, ts + 50, bb, ab, 5.0, 5.0, 55.0, PROVENANCE_REAL))
        self.store.upsert_bbo_ticks(ticks)

    def test_micro_runner_execution_kill_fires_immediately_on_wide_live_spread(self) -> None:
        """Layer 1 Execution-Quality Kill MUST fire immediately when live combined half-spread > 2x assumed slippage."""
        # Assumed slippage_per_leg = 1.5bp -> two-leg assumed = 3.0bp -> kill > 6.0bp!
        # Seed 16bp spread on venue_a + 4bp on venue_b -> combined half-spread = 10.0bp (3.33x assumed)
        self._seed_bbo("lighter", "gate", spread_a_bps=16.0, spread_b_bps=4.0, n_rounds=12)
        cfg = FundingCarryPlugin().default_config(
            strategy_id="mr_wide_spread",
            symbols=["BTC", "ETH"],
            slippage_per_leg_bps=1.5,
        )
        rec = self.session.evaluate_bbo_window(cfg, target_hours=24.0, simulated_elapsed_hours=1.0)
        self.assertEqual(rec.status, "KILLED_EARLY")
        self.assertIn("EXECUTION_KILL:LIVE_SPREAD_EXCEEDS", rec.kill_reason)
        self.assertGreater(rec.slippage_ratio, 2.0)

    def test_micro_runner_does_not_false_kill_carry_during_hour_1_rt_amortization(self) -> None:
        """Architect P1-4: Negative paper PnL at hour 1 (RT cost amortization) MUST NOT kill a 96h-horizon carry strategy."""
        self._seed_bbo("lighter", "gate", spread_a_bps=1.0, spread_b_bps=1.0, n_rounds=12)
        cfg = FundingCarryPlugin().default_config(
            strategy_id="mr_carry_amort",
            symbols=["BTC", "ETH"],
            min_hold_hours=96,
            slippage_per_leg_bps=1.5,
        )
        # Hour 2 with -$5 RT opening drag: MUST stay ACTIVE (NOT killed!)
        rec_early = self.session.evaluate_bbo_window(
            cfg, target_hours=24.0, paper_net_pnl_usd=-5.0, paper_drawdown_pct=0.2, simulated_elapsed_hours=2.0
        )
        self.assertEqual(rec_early.status, "ACTIVE")
        self.assertEqual(rec_early.kill_reason, "")

        # After full 96h horizon with persistent -$12 negative PnL -> triggers ITERATE_AGENT feedback!
        rec_late = self.session.evaluate_bbo_window(
            cfg, target_hours=96.0, paper_net_pnl_usd=-12.0, paper_drawdown_pct=0.5, simulated_elapsed_hours=96.5
        )
        self.assertEqual(rec_late.status, "ITERATE_AGENT")
        self.assertIn("ECONOMIC_FEEDBACK", rec_late.kill_reason)

        # Red-Team #6: Catastrophic >5% DD at hour 0.5 MUST kill immediately (never wait 96h!)
        rec_cat = self.session.evaluate_bbo_window(
            cfg, target_hours=96.0, paper_net_pnl_usd=-200.0, paper_drawdown_pct=6.5, simulated_elapsed_hours=0.5
        )
        self.assertEqual(rec_cat.status, "KILLED_EARLY")
        self.assertIn("CATASTROPHIC_KILL", rec_cat.kill_reason)

        # Red-Team #6: Setting target_hours=1.0 on a 96h strategy MUST NOT grant premature COMPLETED_PASS at hour 2!
        rec_short = self.session.evaluate_bbo_window(
            cfg, target_hours=1.0, paper_net_pnl_usd=5.0, paper_drawdown_pct=0.1, simulated_elapsed_hours=2.0
        )
        self.assertEqual(rec_short.status, "ACTIVE")

        # Red-Team HIGH-R2: 96h live replay with ZERO emitted intents MUST NOT earn COMPLETED_PASS!
        rec_zero_intents = self.session.evaluate_bbo_window(
            cfg, target_hours=96.0, paper_net_pnl_usd=None, paper_drawdown_pct=None, simulated_elapsed_hours=97.0
        )
        self.assertEqual(rec_zero_intents.status, "ITERATE_AGENT")
        self.assertEqual(rec_zero_intents.kill_reason, "ZERO_INTENTS_EXECUTED_IN_MICRO_RUN")

    def test_ssh_fleet_auditor_rejects_shell_injection_and_unknown_probes(self) -> None:
        """ReadOnlyFleetAuditor MUST strictly reject non-allowlisted probes, invalid IPs, privileged users, and non-/opt/ or secret paths."""
        roles_file = self.root / "roles.json"
        roles_file.write_text(json.dumps({"ssh_readonly_user": "agent-ro", "nodes": []}), encoding="utf-8")
        aud = ReadOnlyFleetAuditor(roles_file, self.root / "dummy_key")

        with self.assertRaises(PermissionError):
            aud.run_probe("127.0.0.1", "arbitrary_rm_rf")
        with self.assertRaises(ValueError):
            aud.run_probe("127.0.0.1; rm -rf /", "health_snapshot")
        for bad_path in (
            "relative/path.csv",
            "/opt/data/../../etc/passwd",
            "/opt/data/a.csv; rm -rf /",
            "/opt/data/$(whoami).csv",
            "/opt/data/a.csv | cat",
        ):
            with self.assertRaises(ValueError, msg=f"Path should be rejected: {bad_path}"):
                validate_safe_path(bad_path)
        for forbidden_path in ("/etc/passwd", "/opt/app/.env", "/opt/app/security.json", "/opt/keys/id_ed25519"):
            with self.assertRaises(PermissionError, msg=f"Forbidden path should be blocked: {forbidden_path}"):
                validate_safe_path(forbidden_path)
        self.assertEqual(validate_safe_path("/opt/research_app/data/spreads.csv"), "/opt/research_app/data/spreads.csv")

        # Red-Team #8: Privileged or flag-injected SSH username MUST be rejected
        for bad_user, exc_t in (("root", PermissionError), ("-oProxyCommand=id", ValueError)):
            bad_roles = self.root / f"roles_{abs(hash(bad_user))}.json"
            bad_roles.write_text(json.dumps({"ssh_readonly_user": bad_user, "nodes": []}), encoding="utf-8")
            with self.assertRaises(exc_t):
                ReadOnlyFleetAuditor(bad_roles, self.root / "dummy_key")

    def test_web_v2_new_endpoints_return_200_and_clean_json(self) -> None:
        srv, ctx = create_server(self.root / "web_v2", host="127.0.0.1", port=0)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            for ep in ("/api/plugins", "/api/dq/reports", "/api/micro-runs", "/api/fleet/audit"):
                conn.request("GET", ep)
                resp = conn.getresponse()
                body = resp.read()
                self.assertEqual(resp.status, 200, f"Endpoint {ep} failed")
                parsed = json.loads(body.decode("utf-8"))
                self.assertIsInstance(parsed, dict)
            conn.close()
        finally:
            srv.shutdown()
            srv.server_close()
            ctx.close()


if __name__ == "__main__":
    unittest.main()
