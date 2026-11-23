"""Closed-Loop Plugin-Agnostic Agent Strategy Evolution Orchestrator (AlphaLoop v2).

Upgrades (Architect P0-5, P1-3, P1-5):
  - 100% strategy-agnostic via PluginRegistry (works on funding_carry, basis_reversion, structural_arb, etc.).
  - Resets per-round LLM budget cleanly before each cycle (P1-5).
  - Enforces Architect P1-3 DQ Promotion Cap:
      * DATA_QUALIFIED_FULL (or verified live BBO FULL cert for the venue pair) -> SHADOW_LIVE.
      * DATA_QUALIFIED_FUNDING_ONLY -> CAPPED at MICRO_RUN_PROBE_REQUIRED (cannot auto-stage SHADOW_LIVE).
      * DATA_REJECTED -> blocked at G1.
  - Supports multi-round autonomous improvement sprints (`run_multi_round_sprint`) with convergence stop.
  - LIVE capital promotion remains strictly human-gated via HotConfigStore.
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
from alphaloop.harness.micro_runner import MicroRunSession, MicroRunStore
from alphaloop.plugins import BaseStrategyPlugin, PluginRegistry, default_registry

logger = logging.getLogger(__name__)

PROPOSER_SYSTEM_PROMPT = (
    "You are a quantitative strategy researcher. Given compact telemetry diagnostics, plugin metadata, "
    "and current StrategyConfig, output a single JSON object with keys: hypothesis (str), "
    "economic_rationale (str), param_changes (dict of allowed numeric fields), kill_criteria (str). "
    "Focus on fixing the exact diagnosed failure mode without overfitting."
)

RED_TEAM_SYSTEM_PROMPT = (
    "You are an adversarial quantitative risk auditor. Inspect ONLY the proposed param_changes, "
    "baseline vs candidate metrics, DQ tier, and 6-Gate verification verdict. Return a single JSON object "
    "with keys: decision ('APPROVE_SHADOW' | 'REVISE' | 'REJECT'), findings (list of {severity, item, detail}), "
    "summary (str). NEVER approve if any machine gate failed, liquidation_count > 0, or DQ is DATA_REJECTED."
)


class AgentEvolutionOrchestrator:
    """Runs and persists plugin-agnostic closed-loop Agent strategy evolution cycles."""

    def __init__(
        self,
        state_dir: str | Path,
        backtester: FundingCarryBacktester,
        verifier: SixGateVerifier,
        config_store: HotConfigStore,
        router: MultiModelRouter | None = None,
        registry: PluginRegistry | None = None,
        micro_store: MicroRunStore | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.backtester = backtester
        self.verifier = verifier
        self.config_store = config_store
        self.router = router or MultiModelRouter()
        self.registry = registry or verifier.registry or default_registry()
        self.micro_store = micro_store
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

    def _has_verified_micro_run_lift(self, cand_cfg: StrategyConfig, max_age_ms: int = 7 * 86400 * 1000) -> bool:
        """Return True ONLY if a fresh COMPLETED_PASS MicroRun exists AND covers 100% of cand_cfg.symbols."""
        if self.micro_store is None:
            return False
        passed_mr = self.micro_store.latest_passed_run(
            cand_cfg.strategy_id, version=cand_cfg.version, max_age_ms=max_age_ms
        )
        if passed_mr is None or passed_mr.dq_tier != "DATA_QUALIFIED_FULL":
            return False
        observed_syms = set(passed_mr.details.get("symbols_observed") or [])
        req_syms = set(cand_cfg.symbols)
        if not req_syms or not req_syms.issubset(observed_syms):
            return False
        bbo_cert = self.backtester.store.latest_dq_certificate(f"bbo:{cand_cfg.venue_a}<->{cand_cfg.venue_b}")
        if not bbo_cert or bbo_cert.get("tier") != "DATA_QUALIFIED_FULL":
            return False
        if (utc_now_ms() - int(bbo_cert.get("created_ts_ms", 0))) > max_age_ms:
            return False
        cert_syms = set((bbo_cert.get("symbol_scores") or {}).keys())
        return req_syms.issubset(cert_syms)

    def run_cycle(
        self,
        current_cfg: StrategyConfig,
        next_version: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        allow_synthetic_in_unit_test: bool = False,
    ) -> dict[str, Any]:
        """Execute 1 complete plugin-agnostic closed-loop iteration."""
        now = utc_now_ms()
        self.router.reset_round_budget()
        store = self.backtester.store
        plugin = self.registry.resolve_for_config(current_cfg)

        # Step 1: Evaluate baseline & diagnose structured anomaly via plugin
        base_res = plugin.evaluate(store, current_cfg, start_ms=start_ms, end_ms=end_ms)
        feedback = plugin.diagnose_telemetry(current_cfg, base_res.metrics)

        # Step 2: Invoke Quant Hypothesis Proposer & clamp strictly via plugin invariants
        proposal = self._propose_mutation(plugin, current_cfg, feedback, base_res.metrics)
        cand_cfg = plugin.validate_and_clamp(current_cfg, next_version, proposal.param_changes)

        # Step 3: Deterministic DQ + 6-Gate Machine Verification (ZERO LLM self-grading)
        verdict, cand_res, extras = self.verifier.verify(
            cand_cfg,
            start_ms=start_ms,
            end_ms=end_ms,
            allow_synthetic_in_unit_test=allow_synthetic_in_unit_test,
            plugin=plugin,
        )

        # Step 4: Adversarial Cross-Model Red-Team Audit
        proposer_family = proposal.model_used.split("/")[0]
        red_team = self._run_red_team(
            current_cfg, cand_cfg, base_res.metrics, cand_res.metrics, verdict, proposer_family
        )

        # Step 5: Atomic Hot-Config Staging with Hardened Architect P1-3 DQ Promotion Cap (Red-Team #5.1)
        promoted = False
        promotion_stage = "HELD"
        staged_sha = None
        if verdict.overall_pass and red_team.decision == "APPROVE_SHADOW":
            has_verified_micro_lift = self._has_verified_micro_run_lift(cand_cfg)
            if verdict.dq_tier == "DATA_QUALIFIED_FULL" or has_verified_micro_lift:
                staged_cfg = dataclasses.replace(cand_cfg, status="SHADOW_LIVE")
                staged_sha = self.config_store.register_version(
                    staged_cfg,
                    actor=f"alphaloop:{proposal.model_used}+rt:{red_team.model_used}",
                    reason=(
                        f"Auto-staged SHADOW_LIVE ({verdict.passed_count}/{verdict.total_count} gates, "
                        f"DQ={verdict.dq_tier}, micro_run_lift={has_verified_micro_lift}, RedTeam=APPROVE_SHADOW)"
                    ),
                )
                promoted = True
                promotion_stage = "SHADOW_LIVE"
            else:
                # Architect P1-3: FUNDING_ONLY MUST pass full-symbol COMPLETED_PASS MicroRun before SHADOW_LIVE!
                probe_cfg = dataclasses.replace(cand_cfg, status="MICRO_RUN_PROBE_REQUIRED")
                staged_sha = self.config_store.register_version(
                    probe_cfg,
                    actor=f"alphaloop:{proposal.model_used}+rt:{red_team.model_used}",
                    reason=(
                        f"Capped at MICRO_RUN_PROBE_REQUIRED (DQ={verdict.dq_tier}: requires full-symbol COMPLETED_PASS MicroRun)"
                    ),
                )
                promoted = False
                promotion_stage = "MICRO_RUN_PROBE_REQUIRED"
        else:
            logger.info(
                "Candidate %s@%s NOT promoted (gates=%d/%d, dq=%s, red_team=%s)",
                cand_cfg.strategy_id,
                cand_cfg.version,
                verdict.passed_count,
                verdict.total_count,
                verdict.dq_tier,
                red_team.decision,
            )

        diff_scorecard = MicroRunSession.compare_versions(
            base_res.metrics.to_dict(), cand_res.metrics.to_dict()
        )
        cycle_id = f"CYC-{sha256_obj({'s': current_cfg.strategy_id, 'v': next_version, 't': now})[:10]}"
        record = {
            "cycle_id": cycle_id,
            "ts_ms": now,
            "plugin_id": plugin.plugin_id,
            "strategy_id": current_cfg.strategy_id,
            "from_version": current_cfg.version,
            "to_version": next_version,
            "feedback": feedback.to_dict(),
            "proposal": proposal.to_dict(),
            "baseline_metrics": base_res.metrics.to_dict(),
            "candidate_metrics": cand_res.metrics.to_dict(),
            "diff_scorecard": diff_scorecard,
            "dq_report": extras["dq_report"],
            "gate_verdict": verdict.to_dict(),
            "red_team_verdict": red_team.to_dict(),
            "promoted_shadow": promoted,
            "promotion_stage": promotion_stage,
            "staged_config_sha256": staged_sha,
            "plateau_grid": extras["plateau_grid"],
            "is_metrics": extras["is_metrics"],
            "oos_metrics": extras["oos_metrics"],
            "stress_metrics": extras["stress_metrics"],
            "taker_hedge_metrics": extras["taker_hedge_metrics"],
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

    def run_multi_round_sprint(
        self,
        seed_cfg: StrategyConfig,
        max_rounds: int = 3,
        version_prefix: str = "v-sprint",
        start_ms: int | None = None,
        end_ms: int | None = None,
        allow_synthetic_in_unit_test: bool = False,
    ) -> dict[str, Any]:
        """Run up to `max_rounds` bounded evolution iterations with early convergence stop."""
        rounds: list[dict[str, Any]] = []
        cur_cfg = seed_cfg
        non_improve_streak = 0
        stop_reason = "MAX_ROUNDS_REACHED"

        for r_idx in range(1, max(1, max_rounds) + 1):
            rec = self.run_cycle(
                cur_cfg,
                next_version=f"{version_prefix}-r{r_idx}",
                start_ms=start_ms,
                end_ms=end_ms,
                allow_synthetic_in_unit_test=allow_synthetic_in_unit_test,
            )
            rounds.append(rec)
            if rec["gate_verdict"]["overall_pass"] and rec["red_team_verdict"]["decision"] == "APPROVE_SHADOW":
                stop_reason = f"CONVERGED_AT_ROUND_{r_idx}({rec['promotion_stage']})"
                cur_cfg = StrategyConfig.from_dict(rec["candidate_config"])
                break
            if rec["diff_scorecard"]["improved"]:
                cur_cfg = StrategyConfig.from_dict(rec["candidate_config"])
                non_improve_streak = 0
            else:
                non_improve_streak += 1
                if non_improve_streak >= 2:
                    stop_reason = f"EARLY_STOP_NO_IMPROVEMENT_AFTER_ROUND_{r_idx}"
                    break

        return {
            "strategy_id": seed_cfg.strategy_id,
            "plugin_id": seed_cfg.plugin_id,
            "rounds_executed": len(rounds),
            "stop_reason": stop_reason,
            "final_version": cur_cfg.version,
            "final_promotion_stage": rounds[-1]["promotion_stage"] if rounds else "NONE",
            "rounds": rounds,
        }

    def _propose_mutation(
        self,
        plugin: BaseStrategyPlugin,
        cfg: StrategyConfig,
        feedback: FeedbackEvent,
        metrics: BacktestMetrics,
    ) -> AgentProposal:
        import math

        now = utc_now_ms()
        bounds = plugin.mutable_param_bounds()
        payload = {
            "plugin": plugin.metadata(),
            "feedback": feedback.to_dict(),
            "current_config": cfg.to_dict(),
            "allowed_bounds": bounds,
        }
        llm_out, model_used = self.router.invoke_json("QUANT_HYPOTHESIS", PROPOSER_SYSTEM_PROMPT, payload)
        if isinstance(llm_out, dict) and isinstance(llm_out.get("param_changes"), dict):
            raw_changes = llm_out["param_changes"]
            sanitized: dict[str, Any] = {}
            rejected_keys: list[str] = []
            for k, v in raw_changes.items():
                if k not in bounds:
                    rejected_keys.append(str(k))
                    continue
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    rejected_keys.append(str(k))
                    continue
                lo, hi = bounds[k]
                if float(v) < float(lo) or float(v) > float(hi):
                    rejected_keys.append(f"{k}={v}∉[{lo},{hi}]")
                    continue
                sanitized[k] = v

            if sanitized and not rejected_keys:
                return AgentProposal(
                    proposal_id=f"PROP-{sha256_obj({'t': now, 'm': model_used})[:8]}",
                    ts_ms=now,
                    model_used=model_used,
                    role="QUANT_HYPOTHESIS",
                    hypothesis=str(llm_out.get("hypothesis", "")),
                    economic_rationale=str(llm_out.get("economic_rationale", "")),
                    param_changes=sanitized,
                    kill_criteria=str(llm_out.get("kill_criteria", "Any G1-G6 failure")),
                )
            logger.warning(
                "LLM proposal rejected fail-closed (invalid/out-of-bounds keys=%s); falling back deterministically",
                rejected_keys or ["EMPTY_CHANGES"],
            )
            model_used = f"deterministic-fallback(sanitized_reject={model_used})"

        changes, hyp, econ, kill = plugin.fallback_proposal(cfg, feedback, metrics)
        return AgentProposal(
            proposal_id=f"PROP-{sha256_obj({'t': now, 'c': changes})[:8]}",
            ts_ms=now,
            model_used=model_used,
            role="QUANT_HYPOTHESIS",
            hypothesis=hyp,
            economic_rationale=econ,
            param_changes=changes,
            kill_criteria=kill,
        )

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
        if rt_spec.family == proposer_family:
            raise RuntimeError(f"Cross-family Red-Team violation: proposer={proposer_family} == critic={rt_spec.family}")
        compact_diff = {
            k: {"before": getattr(base_cfg, k), "after": getattr(cand_cfg, k)}
            for k in cand_cfg.to_dict()
            if getattr(base_cfg, k) != getattr(cand_cfg, k) and k not in ("version", "status")
        }
        payload = {
            "param_diff": compact_diff,
            "baseline": {
                "apr_pct": base_m.annualized_apr_pct,
                "sharpe": base_m.sharpe_ratio,
                "net_usd": base_m.net_pnl_usd,
                "fee_ratio": base_m.fee_to_gross_ratio,
                "liquidations": base_m.liquidation_count,
            },
            "candidate": {
                "apr_pct": cand_m.annualized_apr_pct,
                "sharpe": cand_m.sharpe_ratio,
                "net_usd": cand_m.net_pnl_usd,
                "fee_ratio": cand_m.fee_to_gross_ratio,
                "max_dd_pct": cand_m.max_drawdown_pct,
                "liquidations": cand_m.liquidation_count,
            },
            "dq_tier": verdict.dq_tier,
            "six_gate_verdict": verdict.to_dict(),
        }
        llm_out, model_used = self.router.invoke_json(
            "RED_TEAM_CRITIC", RED_TEAM_SYSTEM_PROMPT, payload, override_spec=rt_spec
        )
        if llm_out and llm_out.get("decision") in ("APPROVE_SHADOW", "REVISE", "REJECT"):
            decision = str(llm_out["decision"])
            if (not verdict.overall_pass or cand_m.liquidation_count > 0) and decision == "APPROVE_SHADOW":
                decision = "REJECT"
            raw_findings = llm_out.get("findings", [])
            if isinstance(raw_findings, list):
                clean_findings = [f for f in raw_findings if isinstance(f, dict)]
            else:
                clean_findings = [
                    {
                        "severity": "MINOR",
                        "item": "MALFORMED_FINDINGS",
                        "detail": str(raw_findings)[:200],
                    }
                ]
            return RedTeamVerdict(
                verdict_id=f"RT-{sha256_obj({'t': now, 'd': decision})[:8]}",
                ts_ms=now,
                model_used=model_used,
                decision=decision,
                findings=clean_findings,
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
        if cand_m.liquidation_count > 0:
            findings.append({
                "severity": "BLOCKER",
                "item": "MARGIN_LIQUIDATION",
                "detail": f"Candidate suffered {cand_m.liquidation_count} single-leg margin liquidation(s).",
            })
        if verdict.dq_tier == "DATA_QUALIFIED_FUNDING_ONLY":
            findings.append({
                "severity": "MAJOR",
                "item": "FUNDING_ONLY_DQ_CAP",
                "detail": "Dataset lacks dual-venue 1h basis coverage; capped at MICRO_RUN_PROBE_REQUIRED until live BBO FULL cert.",
            })
        if cand_m.fee_to_gross_ratio > 0.50:
            findings.append({
                "severity": "MAJOR",
                "item": "COST_DRAG",
                "detail": f"Costs consume {cand_m.fee_to_gross_ratio:.1%} of gross alpha.",
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
                f"Sharpe={cand_m.sharpe_ratio:.2f}, MaxDD={cand_m.max_drawdown_pct:.2f}%, DQ={verdict.dq_tier}). "
                f"Real capital deployment remains strictly human-gated."
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
        rows = self.conn.fetch_all(
            "SELECT payload_json FROM agent_cycles ORDER BY ts_ms DESC LIMIT ?",
            (limit,),
        )
        return [json.loads(r["payload_json"]) for r in rows]

    def close(self) -> None:
        self.conn.close()
