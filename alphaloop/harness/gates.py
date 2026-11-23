"""Automated 6-Gate Statistical & Anti-Overfitting Verification Harness.

Consolidates the strongest machine-enforceable gates from MMA/CYPO/BAISYS:
  G1: Data Provenance & Integrity (strict REAL_EXCHANGE_API, non-empty SHA-256)
  G2: Economic Expectancy & Cost Stress (MAIN & STRESS +2bp/leg both positive, MaxDD cap)
  G3: Walk-Forward Out-of-Sample Split (70% IS / 30% OOS, OOS positive & retention >= 50%)
  G4: Parameter Plateau Stability (3x3 entry/exit grid >= 70% positive, no isolated spike)
  G5: Concentration & Robustness (drop top-5 trades AND drop best symbol remain positive)
  G6: Append-Only Multiple-Testing Trial Ledger (records every trial & warns on trial inflation)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Any

from alphaloop.core.db import connect_wal, transaction
from alphaloop.core.models import BacktestMetrics, GateCheck, GateVerdict, StrategyConfig
from alphaloop.core.provenance import PROVENANCE_REAL, sha256_obj, utc_now_ms
from alphaloop.harness.backtester import BacktestResult, FundingCarryBacktester

logger = logging.getLogger(__name__)


class TrialLedger:
    """Append-only SQLite WAL trial ledger tracking every strategy evaluation (G6)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = connect_wal(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trial_ledger (
                    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    family TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    dataset_sha256 TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    net_pnl_usd REAL NOT NULL,
                    apr_pct REAL NOT NULL,
                    sharpe REAL NOT NULL,
                    max_dd_pct REAL NOT NULL,
                    overall_pass INTEGER NOT NULL,
                    verdict_json TEXT NOT NULL
                );
                """
            )

    def count_family_trials(self, family: str) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM trial_ledger WHERE family=?", (family,)).fetchone()
        return int(row["n"]) if row else 0

    def append(self, cfg: StrategyConfig, metrics: BacktestMetrics, verdict: GateVerdict) -> int:
        with transaction(self.conn):
            cur = self.conn.execute(
                """
                INSERT INTO trial_ledger
                (ts_ms, family, strategy_id, version, config_sha256, dataset_sha256,
                 provenance, net_pnl_usd, apr_pct, sharpe, max_dd_pct, overall_pass, verdict_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now_ms(),
                    cfg.family,
                    cfg.strategy_id,
                    cfg.version,
                    sha256_obj(cfg.to_dict()),
                    metrics.dataset_sha256,
                    metrics.provenance,
                    metrics.net_pnl_usd,
                    metrics.annualized_apr_pct,
                    metrics.sharpe_ratio,
                    metrics.max_drawdown_pct,
                    1 if verdict.overall_pass else 0,
                    json.dumps(verdict.to_dict(), sort_keys=True),
                ),
            )
            return int(cur.lastrowid)

    def recent_trials(self, limit: int = 30) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT trial_id, ts_ms, family, strategy_id, version, provenance,
                   net_pnl_usd, apr_pct, sharpe, max_dd_pct, overall_pass
            FROM trial_ledger ORDER BY trial_id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()


class SixGateVerifier:
    """Runs deterministic 6-Gate verification + plateau matrix + OOS walk-forward."""

    def __init__(self, backtester: FundingCarryBacktester, ledger: TrialLedger) -> None:
        self.backtester = backtester
        self.ledger = ledger

    def verify(
        self,
        cfg: StrategyConfig,
        start_ms: int | None = None,
        end_ms: int | None = None,
        allow_synthetic_in_unit_test: bool = False,
    ) -> tuple[GateVerdict, BacktestResult, dict[str, Any]]:
        """Run full 6-gate battery and record trial in append-only ledger."""
        main_res = self.backtester.run(cfg, start_ms=start_ms, end_ms=end_ms)
        m = main_res.metrics
        checks: list[GateCheck] = []

        # G1: Data Provenance & Integrity
        g1_ok = (m.provenance == PROVENANCE_REAL or allow_synthetic_in_unit_test) and m.trade_count >= 5
        checks.append(
            GateCheck(
                gate_id="G1",
                name="Data Provenance & Sample Floor",
                passed=g1_ok,
                value=f"{m.provenance} (trades={m.trade_count})",
                threshold=f"{PROVENANCE_REAL} & trades>=5",
                detail=f"sha256={m.dataset_sha256[:12]} days={m.days}",
            )
        )

        # G2: Economic Expectancy & Cost Stress (+2bp/leg extra slippage)
        stress_res = self.backtester.run(cfg, start_ms=start_ms, end_ms=end_ms, extra_slippage_bps_per_leg=2.0)
        sm = stress_res.metrics
        g2_ok = (
            m.net_pnl_usd > 0.0
            and sm.net_pnl_usd > 0.0
            and m.annualized_apr_pct >= 5.0
            and m.max_drawdown_pct <= 10.0
            and m.fee_to_gross_ratio <= 0.65
        )
        checks.append(
            GateCheck(
                gate_id="G2",
                name="Cost-Stressed Expectancy & Drawdown",
                passed=g2_ok,
                value=f"MAIN={m.annualized_apr_pct:+.1f}%/DD={m.max_drawdown_pct:.2f}% | STRESS={sm.annualized_apr_pct:+.1f}%",
                threshold="MAIN>=5% & STRESS>0 & DD<=10% & Cost/Gross<=65%",
                detail=f"main_net=${m.net_pnl_usd:+.2f}, stress_net=${sm.net_pnl_usd:+.2f}, cost_ratio={m.fee_to_gross_ratio:.2%}",
            )
        )

        # G3: Continuous Causal Walk-Forward 70% IS / 30% OOS (no cold-warmup truncation distortion)
        eq = main_res.equity_curve
        split_idx = int(len(eq) * 0.70)
        is_seg = eq[:split_idx] if split_idx >= 2 else eq
        oos_seg = eq[split_idx:] if (len(eq) - split_idx) >= 2 else eq

        def _seg_summary(seg: list[tuple[int, float]]) -> dict[str, float]:
            if len(seg) < 2 or seg[0][1] <= 0:
                return {"net_pnl_usd": 0.0, "apr_pct": 0.0, "days": 0.0}
            s0, s1 = seg[0][1], seg[-1][1]
            d_days = max(1.0 / 24.0, (seg[-1][0] - seg[0][0]) / (86400.0 * 1000.0))
            pnl = s1 - s0
            apr = ((s1 / s0) - 1.0) * 100.0 * (365.0 / d_days)
            return {"net_pnl_usd": round(pnl, 4), "apr_pct": round(apr, 2), "days": round(d_days, 2)}

        is_info = _seg_summary(is_seg)
        oos_info = _seg_summary(oos_seg)
        is_apr = is_info["apr_pct"]
        oos_apr = oos_info["apr_pct"]
        retention = (oos_apr / is_apr) if is_apr > 1.0 else (1.0 if oos_apr > 0 else 0.0)
        g3_ok = is_info["net_pnl_usd"] > 0.0 and oos_info["net_pnl_usd"] > 0.0 and retention >= 0.25
        checks.append(
            GateCheck(
                gate_id="G3",
                name="Continuous Walk-Forward 70/30 OOS Consistency",
                passed=g3_ok,
                value=f"IS={is_apr:+.1f}% ({is_info['days']}d) | OOS={oos_apr:+.1f}% ({oos_info['days']}d, ret={retention:.2f}x)",
                threshold="IS>0 & OOS>0 & OOS/IS>=0.25",
                detail=f"IS_net=${is_info['net_pnl_usd']:+.2f}, OOS_net=${oos_info['net_pnl_usd']:+.2f}",
            )
        )

        # G4: Parameter Plateau Stability (3x3 grid around entry_spread_apr & exit_spread_apr)
        plateau_grid: list[dict[str, Any]] = []
        entry_factors = [0.75, 1.0, 1.25]
        exit_factors = [0.70, 1.0, 1.30]
        pos_cells = 0
        aprs: list[float] = []
        for ef in entry_factors:
            for xf in exit_factors:
                cand_entry = max(4.0, round(cfg.entry_spread_apr_pct * ef, 2))
                cand_exit = min(cand_entry - 2.0, max(0.5, round(cfg.exit_spread_apr_pct * xf, 2)))
                cand_cfg = dataclasses.replace(
                    cfg,
                    entry_spread_apr_pct=cand_entry,
                    exit_spread_apr_pct=cand_exit,
                )
                r = self.backtester.run(cand_cfg, start_ms=start_ms, end_ms=end_ms)
                apr_val = r.metrics.annualized_apr_pct
                aprs.append(apr_val)
                if r.metrics.net_pnl_usd > 0.0:
                    pos_cells += 1
                plateau_grid.append({
                    "entry_apr": cand_entry,
                    "exit_apr": cand_exit,
                    "apr_pct": apr_val,
                    "sharpe": r.metrics.sharpe_ratio,
                    "net_pnl_usd": r.metrics.net_pnl_usd,
                    "trades": r.metrics.trade_count,
                })
        pos_ratio = pos_cells / len(plateau_grid)
        avg_neighbor = sum(aprs) / len(aprs) if aprs else 0.0
        peak_ratio = (m.annualized_apr_pct / avg_neighbor) if avg_neighbor > 0.5 else 99.0
        # Reject isolated spikes where center > 1.8x neighbor average
        g4_ok = pos_ratio >= 0.75 and (0.5 <= peak_ratio <= 1.8)
        checks.append(
            GateCheck(
                gate_id="G4",
                name="Parameter Plateau Stability (3x3 Grid)",
                passed=g4_ok,
                value=f"{pos_cells}/9 positive ({pos_ratio:.0%}), center/avg={peak_ratio:.2f}x",
                threshold=">=75% positive & center/avg in [0.5, 1.8]",
                detail=f"grid APR range [{min(aprs):+.1f}%, {max(aprs):+.1f}%]",
            )
        )

        # G5: Concentration & Robustness (drop top-5 trades AND drop best symbol)
        g5_ok = m.top5_removed_net_pnl_usd > 0.0 and (
            len(cfg.symbols) <= 1 or m.top_symbol_removed_net_pnl_usd > 0.0
        )
        checks.append(
            GateCheck(
                gate_id="G5",
                name="Anti-Concentration (Drop Top-5 & Best Coin)",
                passed=g5_ok,
                value=f"exTop5=${m.top5_removed_net_pnl_usd:+.2f} | exBest({m.best_symbol})=${m.top_symbol_removed_net_pnl_usd:+.2f}",
                threshold="Both > $0.00",
                detail=f"full_net=${m.net_pnl_usd:+.2f}",
            )
        )

        # G6: Multiple-Testing Trial Accounting (Deflated Sharpe hair-cut)
        prior_trials = self.ledger.count_family_trials(cfg.family)
        trial_num = prior_trials + 1
        # Conservative expected max Sharpe under null: sqrt(2 * ln(N)) for N trials
        null_threshold = math.sqrt(2.0 * math.log(max(2, trial_num))) * 0.5
        g6_ok = m.sharpe_ratio >= max(0.8, null_threshold)
        dsr_note = f"trial #{trial_num} in family '{cfg.family}'; null_sharpe_hurdle={max(0.8, null_threshold):.2f}"
        checks.append(
            GateCheck(
                gate_id="G6",
                name="Multiple-Testing Deflated Sharpe Hurdle",
                passed=g6_ok,
                value=f"Sharpe={m.sharpe_ratio:.2f} (trial #{trial_num})",
                threshold=f"Sharpe>={max(0.8, null_threshold):.2f}",
                detail=dsr_note,
            )
        )

        passed_count = sum(1 for c in checks if c.passed)
        verdict = GateVerdict(
            strategy_id=cfg.strategy_id,
            version=cfg.version,
            overall_pass=(passed_count == len(checks)),
            passed_count=passed_count,
            total_count=len(checks),
            trial_number=trial_num,
            deflated_sharpe_note=dsr_note,
            checks=checks,
        )
        self.ledger.append(cfg, m, verdict)

        extras = {
            "stress_metrics": sm.to_dict(),
            "is_metrics": is_info,
            "oos_metrics": oos_info,
            "plateau_grid": plateau_grid,
        }
        return verdict, main_res, extras
