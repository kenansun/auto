"""Closed-Loop 4-Role Agent Strategy Evolution Orchestrator (RD-Agent Pattern).

Pipeline:
  1. TelemetryMonitor compresses live/paper/backtest results into a compact FeedbackEvent.
  2. QuantHypothesisAgent (DeepSeek Reasoner / fallback) proposes a single bounded config mutation.
  3. SixGateVerifier deterministically evaluates candidate on REAL exchange history (no LLM self-grading).
  4. RedTeamCriticAgent (Cross-family Claude Opus / GPT-6.1 / fallback) audits compact diff + 6-Gate report.
  5. HotConfigStore atomically stages SHADOW_LIVE config via rename(2) ONLY if all 6 gates PASS
     AND Red-Team approves. LIVE money deployment remains strictly human-gated.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from alphaloop.agent_loop.model_router import MultiModelRouter
from alphaloop.core.config import HotConfigStore
from alphaloop.core.db import connect_wal, transaction
from alphaloop.core.models import (
    AgentProposal,
    BacktestMetrics,
    FeedbackEvent,
    GateVerdict,
    RedTeamVerdict,
    StrategyConfig,
)
from alphaloop.core.provenance import sha256_obj, utc_now_ms
from alphaloop.harness.backtester import FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier
from alphaloop.harness.telemetry import TelemetryMonitor

logger = logging.getLogger(__name__)

PROPOSER_SYSTEM_PROMPT = (
    "You are a quantitative strategy researcher. Given compact telemetry diagnostics and current "
    "StrategyConfig, output a single JSON object with keys: hypothesis (str), economic_rationale (str), "
    "param_changes (dict of allowed numeric/bool fields), kill_criteria (str). "
    "Focus on fixing the exact diagnosed failure mode without overfitting."
)

RED_TEAM_SYSTEM_PROMPT = (
    "You are an adversarial quantitative risk auditor. Inspect ONLY the proposed param_changes, "
    "baseline vs candidate metrics, and 6-Gate verification verdict. Return a single JSON object with keys: "
    "decision ('APPROVE_SHADOW' | 'REVISE' | 'REJECT'), findings (list of {severity, item, detail}), summary (str). "
    "NEVER approve if any machine gate failed or if costs/concentration remain unsafe."
)

# Bounded mutation whitelist prevents an LLM from modifying capital/safety/venue invariants
MUTABLE_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "entry_spread_apr_pct": (4.0, 60.0),
    "exit_spread_apr_pct": (0.0, 20.0),
    "lookback_hours": (8.0, 240.0),
    "exit_confirm_hours": (4.0, 96.0),
    "min_hold_hours": (8.0, 240.0),
    "max_basis_entry_bps": (10.0, 100.0),
    "stop_loss_basis_bps": (80.0, 400.0),
    "max_positions": (1.0, 10.0),
}


class AgentEvolutionOrchestrator:
    """Runs and persists end-to-end closed-loop Agent strategy iteration cycles."""

    def __init__(
        self,
        state_dir: str | Path,
        backtester: FundingCarryBacktester,
        verifier: SixGateVerifier,
        config_store: HotConfigStore,
        router: MultiModelRouter | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.backtester = backtester
        self.verifier = verifier
        self.config_store = config_store
        self.router = router or MultiModelRouter()
        self.telemetry = TelemetryMonitor()
        self.conn = connect_wal(self.state_dir / "agent_cycles.sqlite")
        self._init_schema()

    def _init_schema(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_cycles (
                    cycle_id TEXT PRIMARY KEY,
                    ts_ms INTEGER NOT NULL,
                    strategy_id TEXT NOT NULL,
                    from_version TEXT NOT NULL,
                    to_version TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    proposer_model TEXT NOT NULL,
                    red_team_model TEXT NOT NULL,
                    gates_passed INTEGER NOT NULL,
                    gates_total INTEGER NOT NULL,
                    red_team_decision TEXT NOT NULL,
                    promoted_shadow INTEGER NOT NULL,
                    baseline_apr_pct REAL NOT NULL,
                    candidate_apr_pct REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def run_cycle(
        self,
        current_cfg: StrategyConfig,
        next_version: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        allow_synthetic_in_unit_test: bool = False,
    ) -> dict[str, Any]:
        """Execute 1 complete closed-loop iteration: Telemetry -> Propose -> 6-Gate -> Red-Team -> Hot-Stage."""
        now = utc_now_ms()
        base_res = self.backtester.run(current_cfg, start_ms=start_ms, end_ms=end_ms)
        feedback = self.telemetry.diagnose(current_cfg, base_res.metrics)

        # Step 2: Invoke Quant Hypothesis Proposer
        proposal = self._propose_mutation(current_cfg, feedback, base_res.metrics)

        # Apply strictly bounded parameter changes
        cand_cfg = self._apply_bounded_changes(current_cfg, next_version, proposal.param_changes)

        # Step 3: Deterministic 6-Gate Machine Verification (ZERO LLM self-grading)
        verdict, cand_res, extras = self.verifier.verify(
            cand_cfg,
            start_ms=start_ms,
            end_ms=end_ms,
            allow_synthetic_in_unit_test=allow_synthetic_in_unit_test,
        )

        # Step 4: Adversarial Cross-Model Red-Team Audit
        proposer_family = proposal.model_used.split("/")[0]
        red_team = self._run_red_team(current_cfg, cand_cfg, base_res.metrics, cand_res.metrics, verdict, proposer_family)

        # Step 5: Atomic Hot-Config Staging (SHADOW_LIVE max; LIVE strictly human-gated)
        promoted = False
        staged_sha = None
        if verdict.overall_pass and red_team.decision == "APPROVE_SHADOW":
            staged_cfg = dataclasses.replace(cand_cfg, status="SHADOW_LIVE")
            staged_sha = self.config_store.register_version(
                staged_cfg,
                actor=f"alphaloop:{proposal.model_used}+rt:{red_team.model_used}",
                reason=f"Auto-staged SHADOW_LIVE after {verdict.passed_count}/{verdict.total_count} gates + RedTeam APPROVE_SHADOW",
            )
            promoted = True
        else:
            rejected_cfg = dataclasses.replace(cand_cfg, status="BACKTESTED")
            logger.info(
                "Candidate %s@%s NOT promoted (gates=%d/%d, red_team=%s)",
                rejected_cfg.strategy_id,
                rejected_cfg.version,
                verdict.passed_count,
                verdict.total_count,
                red_team.decision,
            )

        cycle_id = f"CYC-{sha256_obj({'s': current_cfg.strategy_id, 'v': next_version, 't': now})[:10]}"
        record = {
            "cycle_id": cycle_id,
            "ts_ms": now,
            "strategy_id": current_cfg.strategy_id,
            "from_version": current_cfg.version,
            "to_version": next_version,
            "feedback": feedback.to_dict(),
            "proposal": proposal.to_dict(),
            "baseline_metrics": base_res.metrics.to_dict(),
            "candidate_metrics": cand_res.metrics.to_dict(),
            "gate_verdict": verdict.to_dict(),
            "red_team_verdict": red_team.to_dict(),
            "promoted_shadow": promoted,
            "staged_config_sha256": staged_sha,
            "plateau_grid": extras["plateau_grid"],
            "is_metrics": extras["is_metrics"],
            "oos_metrics": extras["oos_metrics"],
            "stress_metrics": extras["stress_metrics"],
            "candidate_config": cand_cfg.to_dict(),
        }

        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT INTO agent_cycles
                (cycle_id, ts_ms, strategy_id, from_version, to_version, anomaly_type,
                 proposer_model, red_team_model, gates_passed, gates_total,
                 red_team_decision, promoted_shadow, baseline_apr_pct, candidate_apr_pct, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    now,
                    current_cfg.strategy_id,
                    current_cfg.version,
                    next_version,
                    feedback.anomaly_type,
                    proposal.model_used,
                    red_team.model_used,
                    verdict.passed_count,
                    verdict.total_count,
                    red_team.decision,
                    1 if promoted else 0,
                    base_res.metrics.annualized_apr_pct,
                    cand_res.metrics.annualized_apr_pct,
                    json.dumps(record, sort_keys=True),
                ),
            )
        return record

    def _propose_mutation(
        self, cfg: StrategyConfig, feedback: FeedbackEvent, metrics: BacktestMetrics
    ) -> AgentProposal:
        now = utc_now_ms()
        payload = {
            "feedback": feedback.to_dict(),
            "current_config": {
                k: getattr(cfg, k)
                for k in (
                    "venue_a", "venue_b", "mode", "entry_spread_apr_pct", "exit_spread_apr_pct",
                    "lookback_hours", "exit_confirm_hours", "min_hold_hours", "max_positions",
                    "fee_a_bps", "fee_b_bps", "use_atb_buggy_rules",
                )
            },
            "allowed_bounds": MUTABLE_PARAM_BOUNDS,
        }
        llm_out, model_used = self.router.invoke_json("QUANT_HYPOTHESIS", PROPOSER_SYSTEM_PROMPT, payload)
        if llm_out and isinstance(llm_out.get("param_changes"), dict):
            return AgentProposal(
                proposal_id=f"PROP-{sha256_obj({'t': now, 'm': model_used})[:8]}",
                ts_ms=now,
                model_used=model_used,
                role="QUANT_HYPOTHESIS",
                hypothesis=str(llm_out.get("hypothesis", "")),
                economic_rationale=str(llm_out.get("economic_rationale", "")),
                param_changes=dict(llm_out["param_changes"]),
                kill_criteria=str(llm_out.get("kill_criteria", "Any G1-G6 failure")),
            )

        # Transparent deterministic quant fallback driven directly by telemetry diagnosis
        changes: dict[str, Any] = {}
        if cfg.use_atb_buggy_rules or feedback.anomaly_type == "CHURN_EXCESS":
            changes = {
                "use_atb_buggy_rules": False,
                "entry_spread_apr_pct": 6.0,
                "exit_spread_apr_pct": 0.5,
                "lookback_hours": 120,
                "min_hold_hours": 96,
                "exit_confirm_hours": 36,
            }
            hyp = (
                "Eliminate inverted hysteresis fee churn (ATB H6 failure mode) by disabling buggy "
                "3-settlement exit, smoothing signal over 120h (5d) structural window, holding positive "
                "carry down to 0.5% exit threshold with 96h min-hold + 36h exit confirmation."
            )
            econ = (
                "Captures structural Perp-DEX (Hyperliquid/Lighter) baseline premium over CEXs (Gate/OKX) "
                "while amortizing 4-leg round-trip costs across 15-20d average holds, cutting cost/gross below 35%."
            )
        elif feedback.anomaly_type == "BASIS_SHOCK":
            changes = {
                "max_basis_entry_bps": max(15.0, round(cfg.max_basis_entry_bps * 0.6, 1)),
                "stop_loss_basis_bps": max(100.0, round(cfg.stop_loss_basis_bps * 0.75, 1)),
                "lookback_hours": min(72, cfg.lookback_hours + 12),
            }
            hyp = "Tighten entry basis dislocation guard and basis stop-loss to cap adverse mark-price divergence."
            econ = "Prevents entering stretched premiums right before mean-reversion hurts the short-perp leg."
        elif feedback.anomaly_type == "EDGE_DECAY":
            changes = {
                "entry_spread_apr_pct": max(12.0, round(cfg.entry_spread_apr_pct * 0.85, 1)),
                "exit_spread_apr_pct": max(2.0, round(cfg.exit_spread_apr_pct * 0.75, 1)),
                "lookback_hours": 48,
                "min_hold_hours": 48,
            }
            hyp = "Smooth trailing signal to 48h and moderate entry hurdle to capture persistent mid-tier carry spreads."
            econ = "Captures longer-lived Hyperliquid baseline vs lower-funding venue spreads while keeping turnover low."
        else:
            changes = {
                "lookback_hours": min(72, cfg.lookback_hours + 12),
                "min_hold_hours": min(72, cfg.min_hold_hours + 12),
            }
            hyp = "Incremental turnover reduction on healthy carry engine to improve Sharpe and cost efficiency."
            econ = "Longer smoothing window filters brief 1h funding spikes without sacrificing structural spread carry."

        return AgentProposal(
            proposal_id=f"PROP-{sha256_obj({'t': now, 'c': changes})[:8]}",
            ts_ms=now,
            model_used=model_used,
            role="QUANT_HYPOTHESIS",
            hypothesis=hyp,
            economic_rationale=econ,
            param_changes=changes,
            kill_criteria="Reject immediately if STRESS net PnL <= 0, OOS retention < 40%, or plateau < 75%.",
        )

    def _apply_bounded_changes(
        self, cfg: StrategyConfig, next_version: str, changes: dict[str, Any]
    ) -> StrategyConfig:
        d = cfg.to_dict()
        d["version"] = next_version
        d["status"] = "BACKTESTED"
        for key, val in changes.items():
            if key == "use_atb_buggy_rules" and isinstance(val, bool):
                d[key] = val
            elif key in MUTABLE_PARAM_BOUNDS and isinstance(val, (int, float)):
                lo, hi = MUTABLE_PARAM_BOUNDS[key]
                clamped = max(lo, min(hi, float(val)))
                if isinstance(getattr(cfg, key), int):
                    d[key] = int(round(clamped))
                else:
                    d[key] = round(clamped, 3)
        # Invariant: entry hurdle must stay at least 4% APR above exit hurdle
        if d["exit_spread_apr_pct"] >= d["entry_spread_apr_pct"] - 4.0:
            d["exit_spread_apr_pct"] = max(0.5, round(d["entry_spread_apr_pct"] - 6.0, 2))
        return StrategyConfig.from_dict(d)

    def _run_red_team(
        self,
        base_cfg: StrategyConfig,
        cand_cfg: StrategyConfig,
        base_m: BacktestMetrics,
        cand_m: BacktestMetrics,
        verdict: GateVerdict,
        proposer_family: str,
    ) -> RedTeamVerdict:
        now = utc_now_ms()
        rt_spec = self.router.select_red_team_route(proposer_family)
        compact_diff = {
            k: {"before": getattr(base_cfg, k), "after": getattr(cand_cfg, k)}
            for k in cand_cfg.to_dict()
            if getattr(base_cfg, k) != getattr(cand_cfg, k) and k not in ("version", "status")
        }
        payload = {
            "param_diff": compact_diff,
            "baseline": {"apr_pct": base_m.annualized_apr_pct, "sharpe": base_m.sharpe_ratio, "net_usd": base_m.net_pnl_usd, "fee_ratio": base_m.fee_to_gross_ratio},
            "candidate": {"apr_pct": cand_m.annualized_apr_pct, "sharpe": cand_m.sharpe_ratio, "net_usd": cand_m.net_pnl_usd, "fee_ratio": cand_m.fee_to_gross_ratio, "max_dd_pct": cand_m.max_drawdown_pct},
            "six_gate_verdict": verdict.to_dict(),
        }
        llm_out, model_used = self.router.invoke_json(
            "RED_TEAM_CRITIC", RED_TEAM_SYSTEM_PROMPT, payload, override_spec=rt_spec
        )
        if llm_out and llm_out.get("decision") in ("APPROVE_SHADOW", "REVISE", "REJECT"):
            decision = str(llm_out["decision"])
            # Hard mechanical veto: LLM CANNOT override failed machine gates!
            if not verdict.overall_pass and decision == "APPROVE_SHADOW":
                decision = "REJECT"
            return RedTeamVerdict(
                verdict_id=f"RT-{sha256_obj({'t': now, 'd': decision})[:8]}",
                ts_ms=now,
                model_used=model_used,
                decision=decision,
                findings=list(llm_out.get("findings", [])),
                summary=str(llm_out.get("summary", "")),
            )

        # Deterministic adversarial red-team audit
        findings: list[dict[str, str]] = []
        for chk in verdict.checks:
            if not chk.passed:
                findings.append({
                    "severity": "BLOCKER",
                    "item": f"{chk.gate_id}:{chk.name}",
                    "detail": f"Observed {chk.value} vs required {chk.threshold} ({chk.detail})",
                })
        if cand_m.fee_to_gross_ratio > 0.50:
            findings.append({
                "severity": "MAJOR",
                "item": "COST_DRAG",
                "detail": f"Costs consume {cand_m.fee_to_gross_ratio:.1%} of gross funding.",
            })
        if cand_m.net_pnl_usd <= base_m.net_pnl_usd and base_m.net_pnl_usd > 0:
            findings.append({
                "severity": "MINOR",
                "item": "NO_NET_LIFT",
                "detail": f"Candidate net (${cand_m.net_pnl_usd:+.2f}) did not beat baseline (${base_m.net_pnl_usd:+.2f}).",
            })

        blockers = [f for f in findings if f["severity"] == "BLOCKER"]
        if not blockers and verdict.overall_pass:
            decision = "APPROVE_SHADOW"
            summary = (
                f"All 6 deterministic gates passed (APR={cand_m.annualized_apr_pct:+.1f}%, "
                f"Sharpe={cand_m.sharpe_ratio:.2f}, MaxDD={cand_m.max_drawdown_pct:.2f}%). "
                f"Cleared for SHADOW_LIVE hot-reload; real capital still requires human sign-off."
            )
        elif len(blockers) <= 2 and cand_m.net_pnl_usd > base_m.net_pnl_usd:
            decision = "REVISE"
            summary = f"Improved over baseline but blocked by {len(blockers)} gate(s); needs another bounded iteration."
        else:
            decision = "REJECT"
            summary = f"Rejected with {len(blockers)} blocker gate failure(s); baseline retained."

        return RedTeamVerdict(
            verdict_id=f"RT-{sha256_obj({'t': now, 'd': decision, 'b': len(blockers)})[:8]}",
            ts_ms=now,
            model_used=model_used,
            decision=decision,
            findings=findings,
            summary=summary,
        )

    def recent_cycles(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT payload_json FROM agent_cycles ORDER BY ts_ms DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def close(self) -> None:
        self.conn.close()
