"""Fast Micro-Run Validation & Iteration Diff Engine (Architect P1-4 / P2-2).

Pure replayable state machine persisted to SQLite WAL (no hidden background daemon threads).
Supports the Owner's rapid iteration doctrine:
  "推出一个策略就让它去跑，快速小跑，对比分析，不断推出优化版本"

Dual-layer Early-Kill Guard (Architect P1-4):
  Layer 1 — Execution-Quality Kill (fires immediately once ticks >= 20):
    - Data Quality cert is DATA_REJECTED
    - Realized half-spread / slippage > 2.0x assumed slippage_per_leg_bps
    - Live mark basis breaches stop_loss_basis_bps
  Layer 2 — Economic Edge-Decay Kill (fires ONLY after elapsed_hours >= plugin.min_economic_kill_horizon_hours):
    - Prevents falsely killing multi-day carry strategies on hour-1 round-trip cost amortization!
    - Fires if paper drawdown exceeds max_drawdown_kill_pct or realized APR collapses negative after full horizon.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

from alphaloop.core.db import connect_wal, transaction
from alphaloop.core.models import (
    DataQualityReport,
    MarketEvent,
    MicroRunRecord,
    OrderIntent,
    StrategyConfig,
)
from alphaloop.core.provenance import sha256_obj, utc_now_ms
from alphaloop.data.quality import DataQualityAuditor
from alphaloop.data.store import MarketDataStore
from alphaloop.plugins import PluginRegistry, default_registry

logger = logging.getLogger(__name__)


class MicroRunStore:
    """SQLite WAL state persistence for replayable Fast Micro-Run sessions."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = connect_wal(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS micro_runs (
                    run_id TEXT PRIMARY KEY,
                    plugin_id TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_ts_ms INTEGER NOT NULL,
                    updated_ts_ms INTEGER NOT NULL,
                    target_hours REAL NOT NULL,
                    elapsed_hours REAL NOT NULL,
                    dq_cert_id TEXT NOT NULL,
                    dq_tier TEXT NOT NULL,
                    ticks_observed INTEGER NOT NULL,
                    realized_spread_bps_median REAL NOT NULL,
                    assumed_slippage_bps REAL NOT NULL,
                    slippage_ratio REAL NOT NULL,
                    paper_net_pnl_usd REAL NOT NULL,
                    paper_drawdown_pct REAL NOT NULL,
                    kill_reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_micro_runs_strat
                ON micro_runs (strategy_id, updated_ts_ms DESC);
                """
            )

    def save(self, rec: MicroRunRecord) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT OR REPLACE INTO micro_runs
                (run_id, plugin_id, strategy_id, version, status, started_ts_ms, updated_ts_ms,
                 target_hours, elapsed_hours, dq_cert_id, dq_tier, ticks_observed,
                 realized_spread_bps_median, assumed_slippage_bps, slippage_ratio,
                 paper_net_pnl_usd, paper_drawdown_pct, kill_reason, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.run_id,
                    rec.plugin_id,
                    rec.strategy_id,
                    rec.version,
                    rec.status,
                    rec.started_ts_ms,
                    rec.updated_ts_ms,
                    rec.target_hours,
                    rec.elapsed_hours,
                    rec.dq_cert_id,
                    rec.dq_tier,
                    rec.ticks_observed,
                    rec.realized_spread_bps_median,
                    rec.assumed_slippage_bps,
                    rec.slippage_ratio,
                    rec.paper_net_pnl_usd,
                    rec.paper_drawdown_pct,
                    rec.kill_reason,
                    json.dumps(rec.to_dict(), sort_keys=True),
                ),
            )

    def get(self, run_id: str) -> MicroRunRecord | None:
        row = self.conn.fetch_one("SELECT payload_json FROM micro_runs WHERE run_id=?", (run_id,))
        if not row:
            return None
        d = json.loads(row["payload_json"])
        return MicroRunRecord(**d)

    def latest_passed_run(
        self,
        strategy_id: str,
        version: str | None = None,
        max_age_ms: int | None = None,
    ) -> MicroRunRecord | None:
        """Return latest COMPLETED_PASS micro-run matching strategy (and optionally version/freshness)."""
        sql = "SELECT payload_json FROM micro_runs WHERE strategy_id=? AND status='COMPLETED_PASS'"
        params: list[Any] = [strategy_id]
        if version is not None:
            sql += " AND version=?"
            params.append(version)
        if max_age_ms is not None:
            sql += " AND updated_ts_ms >= ?"
            params.append(utc_now_ms() - int(max_age_ms))
        sql += " ORDER BY updated_ts_ms DESC LIMIT 1"
        row = self.conn.fetch_one(sql, params)
        if not row:
            return None
        return MicroRunRecord(**json.loads(row["payload_json"]))

    def list_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        rows = self.conn.fetch_all(
            "SELECT payload_json FROM micro_runs ORDER BY updated_ts_ms DESC LIMIT ?",
            (limit,),
        )
        return [json.loads(r["payload_json"]) for r in rows]

    def close(self) -> None:
        self.conn.close()


class MicroRunSession:
    """Deterministic, replayable Fast Micro-Run state machine with Dual-Layer Early-Kill Guard."""

    def __init__(
        self,
        market_store: MarketDataStore,
        run_store: MicroRunStore,
        registry: PluginRegistry | None = None,
    ) -> None:
        self.market_store = market_store
        self.run_store = run_store
        self.registry = registry or default_registry()
        self.auditor = DataQualityAuditor(market_store)

    @staticmethod
    def _simulate_paper_intents(
        events: list[MarketEvent],
        plugin: Any,
        cfg: StrategyConfig,
    ) -> tuple[list[OrderIntent], float, float]:
        """Replay events chronologically and compute realistic mark-to-market paper PnL & drawdown."""
        state: dict[str, Any] = {}
        intents: list[OrderIntent] = []
        latest_bbo: dict[tuple[str, str], dict[str, float]] = {}
        open_pos: dict[str, dict[str, Any]] = {}
        realized_pnl = 0.0
        peak_eq = cfg.capital_usd
        max_dd_pct = 0.0
        fee_one_side = (cfg.fee_a_bps + cfg.fee_b_bps) / 10000.0

        def _mtm_unrealized() -> float:
            unr = 0.0
            for sym, pos in open_pos.items():
                qa = latest_bbo.get((cfg.venue_a, sym))
                qb = latest_bbo.get((cfg.venue_b, sym))
                if not qa or not qb:
                    continue
                # Conservative mark-to-exit using bid/ask crossing
                if pos["action"] == "OPEN_SHORT_A_LONG_B":
                    pnl_a = ((pos["entry_a"] - qa["ask"]) / pos["entry_a"]) * pos["notional"]
                    pnl_b = ((qb["bid"] - pos["entry_b"]) / pos["entry_b"]) * pos["notional"]
                else:
                    pnl_a = ((qa["bid"] - pos["entry_a"]) / pos["entry_a"]) * pos["notional"]
                    pnl_b = ((pos["entry_b"] - qb["ask"]) / pos["entry_b"]) * pos["notional"]
                unr += pnl_a + pnl_b + pos["accrued_funding"]
            return unr

        for ev in events:
            if ev.kind == "BBO":
                latest_bbo[(ev.venue, ev.symbol)] = {
                    "bid": float(ev.payload["bid"]),
                    "ask": float(ev.payload["ask"]),
                }
            elif ev.kind == "FUNDING" and ev.symbol in open_pos:
                pos = open_pos[ev.symbol]
                rate = float(ev.payload.get("funding_rate", 0.0))
                if ev.venue == cfg.venue_a:
                    sign = 1.0 if pos["action"] == "OPEN_SHORT_A_LONG_B" else -1.0
                    pos["accrued_funding"] += sign * rate * pos["notional"]
                elif ev.venue == cfg.venue_b:
                    sign = -1.0 if pos["action"] == "OPEN_SHORT_A_LONG_B" else 1.0
                    pos["accrued_funding"] += sign * rate * pos["notional"]

            step_intents = plugin.on_event(ev, state, cfg)
            for intent in step_intents:
                intents.append(intent)
                qa = latest_bbo.get((cfg.venue_a, intent.symbol))
                qb = latest_bbo.get((cfg.venue_b, intent.symbol))
                if intent.action in ("OPEN_SHORT_A_LONG_B", "OPEN_LONG_A_SHORT_B"):
                    if intent.symbol in open_pos or not qa or not qb:
                        continue
                    entry_a = qa["bid"] if intent.action == "OPEN_SHORT_A_LONG_B" else qa["ask"]
                    entry_b = qb["ask"] if intent.action == "OPEN_SHORT_A_LONG_B" else qb["bid"]
                    if entry_a > 0 and entry_b > 0:
                        notional = max(1.0, float(intent.notional_usd))
                        realized_pnl -= notional * fee_one_side
                        open_pos[intent.symbol] = {
                            "action": intent.action,
                            "entry_a": entry_a,
                            "entry_b": entry_b,
                            "notional": notional,
                            "accrued_funding": 0.0,
                        }
                elif intent.action == "CLOSE" and intent.symbol in open_pos and qa and qb:
                    pos = open_pos.pop(intent.symbol)
                    if pos["action"] == "OPEN_SHORT_A_LONG_B":
                        pnl_a = ((pos["entry_a"] - qa["ask"]) / pos["entry_a"]) * pos["notional"]
                        pnl_b = ((qb["bid"] - pos["entry_b"]) / pos["entry_b"]) * pos["notional"]
                    else:
                        pnl_a = ((qa["bid"] - pos["entry_a"]) / pos["entry_a"]) * pos["notional"]
                        pnl_b = ((pos["entry_b"] - qb["ask"]) / pos["entry_b"]) * pos["notional"]
                    realized_pnl += pnl_a + pnl_b + pos["accrued_funding"] - pos["notional"] * fee_one_side

            eq = cfg.capital_usd + realized_pnl + _mtm_unrealized()
            if eq > peak_eq:
                peak_eq = eq
            if peak_eq > 0:
                dd = max(0.0, (peak_eq - eq) / peak_eq * 100.0)
                if dd > max_dd_pct:
                    max_dd_pct = dd

        net_pnl = realized_pnl + _mtm_unrealized()
        return intents, net_pnl, max_dd_pct

    def evaluate_bbo_window(
        self,
        cfg: StrategyConfig,
        target_hours: float = 24.0,
        start_ms: int | None = None,
        end_ms: int | None = None,
        max_stale_sec: float | None = 3600.0,
        assumed_slippage_multiplier_kill: float = 2.0,
        paper_net_pnl_usd: float | None = None,
        paper_drawdown_pct: float | None = None,
        simulated_elapsed_hours: float | None = None,
    ) -> MicroRunRecord:
        """Replay real captured BBO + Funding events for (venue_a, venue_b) & apply Dual-Layer Early-Kill."""
        now = utc_now_ms()
        plugin = self.registry.resolve_for_config(cfg)
        dq_report, bbo_summary = self.auditor.audit_bbo_ticks(
            cfg.venue_a,
            cfg.venue_b,
            list(cfg.symbols),
            start_ms=start_ms,
            end_ms=end_ms,
            dataset_name=f"micro_run:{cfg.strategy_id}@{cfg.version}",
            persist=True,
            max_stale_sec=max_stale_sec,
        )

        # Replay BOTH BBO ticks AND overlapping FUNDING bars chronologically (Red-Team #6)
        events: list[MarketEvent] = []
        min_ts, max_ts = 0, 0
        for sym in cfg.symbols:
            for v in (cfg.venue_a, cfg.venue_b):
                for t in self.market_store.load_bbo_ticks(v, sym, start_ms, end_ms):
                    if min_ts == 0 or t.ts_ms < min_ts:
                        min_ts = t.ts_ms
                    if t.ts_ms > max_ts:
                        max_ts = t.ts_ms
                    events.append(
                        MarketEvent(
                            kind="BBO",
                            ts_ms=t.ts_ms,
                            venue=t.venue,
                            symbol=t.symbol,
                            payload={
                                "bid": t.bid,
                                "ask": t.ask,
                                "bid_qty": t.bid_qty,
                                "ask_qty": t.ask_qty,
                                "spread_bps": t.spread_bps,
                            },
                        )
                    )
        if min_ts > 0 and max_ts >= min_ts:
            for sym in cfg.symbols:
                for v in (cfg.venue_a, cfg.venue_b):
                    if cfg.mode == "spot_perp" and v == cfg.venue_b:
                        continue
                    for f in self.market_store.load_funding(v, sym, min_ts, max_ts):
                        events.append(
                            MarketEvent(
                                kind="FUNDING",
                                ts_ms=f.ts_ms,
                                venue=f.venue,
                                symbol=f.symbol,
                                payload={
                                    "funding_rate": f.funding_rate,
                                    "interval_hours": f.interval_hours,
                                    "apr_pct": f.apr_pct,
                                },
                            )
                        )
        events.sort(key=lambda e: (e.ts_ms, 0 if e.kind == "BBO" else 1))
        intents, sim_pnl, sim_dd = self._simulate_paper_intents(events, plugin, cfg)
        eff_pnl = sim_pnl if paper_net_pnl_usd is None else float(paper_net_pnl_usd)
        eff_dd = sim_dd if paper_drawdown_pct is None else float(paper_drawdown_pct)

        elapsed_h = (
            float(simulated_elapsed_hours)
            if simulated_elapsed_hours is not None
            else (max(0.0, (max_ts - min_ts) / (3600.0 * 1000.0)) if max_ts > min_ts else 0.0)
        )
        ticks_n = int(bbo_summary["total_ticks"])
        realized_half_spread_bps = float(bbo_summary["combined_half_spread_cost_bps"])
        assumed_two_leg_slip = max(0.5, 2.0 * cfg.slippage_per_leg_bps)
        slip_ratio = round(realized_half_spread_bps / assumed_two_leg_slip, 3)

        status, kill_reason = self._evaluate_early_kill(
            cfg=cfg,
            plugin_min_horizon_h=plugin.min_economic_kill_horizon_hours(cfg),
            dq_report=dq_report,
            ticks_observed=ticks_n,
            slippage_ratio=slip_ratio,
            multiplier_kill=assumed_slippage_multiplier_kill,
            elapsed_hours=elapsed_h,
            target_hours=target_hours,
            paper_net_pnl_usd=eff_pnl,
            paper_drawdown_pct=eff_dd,
            bbo_summary=bbo_summary,
            intents_observed=len(intents),
            is_simulated_pnl=(paper_net_pnl_usd is None),
        )

        run_id = f"MR-{sha256_obj({'s': cfg.strategy_id, 'v': cfg.version, 't': now})[:10]}"
        rec = MicroRunRecord(
            run_id=run_id,
            plugin_id=plugin.plugin_id,
            strategy_id=cfg.strategy_id,
            version=cfg.version,
            status=status,
            started_ts_ms=min_ts or now,
            updated_ts_ms=now,
            target_hours=round(target_hours, 2),
            elapsed_hours=round(elapsed_h, 3),
            dq_cert_id=dq_report.cert_id,
            dq_tier=dq_report.tier,
            ticks_observed=ticks_n,
            realized_spread_bps_median=round(realized_half_spread_bps, 3),
            assumed_slippage_bps=round(assumed_two_leg_slip, 3),
            slippage_ratio=slip_ratio,
            paper_net_pnl_usd=round(eff_pnl, 4),
            paper_drawdown_pct=round(eff_dd, 4),
            kill_reason=kill_reason,
            details={
                "bbo_summary": bbo_summary,
                "intents_emitted": len(intents),
                "funding_events_replayed": sum(1 for e in events if e.kind == "FUNDING"),
                "symbols_observed": sorted(bbo_summary.get("per_symbol", {}).keys()),
                "min_economic_kill_horizon_hours": plugin.min_economic_kill_horizon_hours(cfg),
                "dq_checks_passed": f"{dq_report.passed_count}/{dq_report.total_count}",
            },
        )
        self.run_store.save(rec)
        return rec

    @staticmethod
    def _evaluate_early_kill(
        *,
        cfg: StrategyConfig,
        plugin_min_horizon_h: float,
        dq_report: DataQualityReport,
        ticks_observed: int,
        slippage_ratio: float,
        multiplier_kill: float,
        elapsed_hours: float,
        target_hours: float,
        paper_net_pnl_usd: float,
        paper_drawdown_pct: float,
        bbo_summary: dict[str, Any],
        intents_observed: int = 0,
        is_simulated_pnl: bool = True,
    ) -> tuple[str, str]:
        """Dual-layer Early-Kill Guard separating Execution-Quality/Catastrophic-DD from Economic-Horizon checks."""
        # LAYER 1: Execution-Quality & Catastrophic Drawdown Kill (fires IMMEDIATELY at ANY hour!)
        if dq_report.tier == "DATA_REJECTED":
            failed = [c.gate_id for c in dq_report.checks if not c.passed and c.severity == "BLOCKING"]
            return "KILLED_EARLY", f"EXECUTION_KILL:DQ_REJECTED({','.join(failed)})"

        if paper_drawdown_pct > 5.0:
            return "KILLED_EARLY", f"CATASTROPHIC_KILL:DRAWDOWN_BREACH({paper_drawdown_pct:.2f}%>5.0%)"

        if ticks_observed >= 20 and slippage_ratio > multiplier_kill:
            return (
                "KILLED_EARLY",
                f"EXECUTION_KILL:LIVE_SPREAD_EXCEEDS_{multiplier_kill:.1f}X_ASSUMED(ratio={slippage_ratio:.2f}x)",
            )

        # Check live basis stop breach across observed symbols
        for sym, st in (bbo_summary.get("per_symbol") or {}).items():
            if float(st.get("p95_abs_basis_bps", 0.0)) > cfg.stop_loss_basis_bps:
                return (
                    "KILLED_EARLY",
                    f"EXECUTION_KILL:LIVE_BASIS_STOP_BREACHED({sym}={st['p95_abs_basis_bps']:.1f}bp>{cfg.stop_loss_basis_bps}bp)",
                )

        # LAYER 2: Economic Edge-Decay Kill (strictly gated behind plugin min horizon!)
        if elapsed_hours >= plugin_min_horizon_h:
            if paper_net_pnl_usd < 0.0:
                return "ITERATE_AGENT", f"ECONOMIC_FEEDBACK:NEGATIVE_NET_AFTER_{elapsed_hours:.1f}H(${paper_net_pnl_usd:+.2f})"

        # COMPLETED_PASS requires elapsed_hours >= BOTH target_hours AND plugin_min_horizon_h
        # AND non-zero simulated intents when running live replay (Red-Team #6 & HIGH-R2)
        required_pass_horizon = max(float(target_hours), float(plugin_min_horizon_h))
        if elapsed_hours >= required_pass_horizon:
            if is_simulated_pnl and intents_observed == 0:
                return "ITERATE_AGENT", "ZERO_INTENTS_EXECUTED_IN_MICRO_RUN"
            if dq_report.is_eligible_for_live and paper_net_pnl_usd >= 0.0:
                return "COMPLETED_PASS", ""
            return "ITERATE_AGENT", "TARGET_HORIZON_REACHED_NEEDS_REVIEW"

        return "ACTIVE", ""

    @staticmethod
    def compare_versions(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        """Produce structured Iteration Diff Scorecard comparing two strategy versions."""
        def _f(d: dict[str, Any], k: str) -> float:
            return float(d.get(k, 0.0) or 0.0)

        apr_delta = round(_f(candidate, "annualized_apr_pct") - _f(baseline, "annualized_apr_pct"), 2)
        sharpe_delta = round(_f(candidate, "sharpe_ratio") - _f(baseline, "sharpe_ratio"), 3)
        dd_delta = round(_f(candidate, "max_drawdown_pct") - _f(baseline, "max_drawdown_pct"), 3)
        net_delta = round(_f(candidate, "net_pnl_usd") - _f(baseline, "net_pnl_usd"), 2)
        fee_delta = round(_f(candidate, "fee_to_gross_ratio") - _f(baseline, "fee_to_gross_ratio"), 4)
        improved = (net_delta > 0 and sharpe_delta >= 0 and dd_delta <= 1.0)
        return {
            "improved": improved,
            "net_pnl_delta_usd": net_delta,
            "apr_delta_pct": apr_delta,
            "sharpe_delta": sharpe_delta,
            "max_dd_delta_pct": dd_delta,
            "fee_ratio_delta": fee_delta,
            "recommendation": "PROMOTE_NEXT_STAGE" if improved else "HOLD_OR_ITERATE",
        }
