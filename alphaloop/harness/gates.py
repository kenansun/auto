"""Automated 6-Gate Statistical, DQ & Anti-Overfitting Verification Harness (AlphaLoop v2).

Upgrades (Architect P0-5, P1-1, P1-2, P1-3, P1-5):
  G1: Hard Data Quality Certification (DQ-01..DQ-08) + Provenance + Sample Floor
      Blocks DATA_REJECTED; attaches dq_cert_id & dq_tier to GateVerdict.
  G2: Economic Expectancy, Cost Stress (+2bp/leg) & Zero-Liquidation Margin Safety (P1-1)
      Also reports MAKER_THEN_TAKER_HEDGE execution sensitivity (P1-2).
  G3: Continuous Causal Walk-Forward 70% IS / 30% OOS Consistency.
  G4: Plugin-Delegated Parameter Plateau Stability (3x3 neighborhood, P0-5).
  G5: Anti-Concentration Robustness (drop top-5 trades AND drop best symbol > $0).
  G6: Append-Only Multiple-Testing Trial Ledger with DSR deduplication & G1-reject exclusion (P1-5).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from alphaloop.core.db import connect_wal, transaction
from alphaloop.core.models import BacktestMetrics, GateCheck, GateVerdict, StrategyConfig
from alphaloop.core.provenance import PROVENANCE_REAL, sha256_obj, utc_now_ms
from alphaloop.data.quality import ALLOWED_REAL_PROVENANCES, DataQualityAuditor
from alphaloop.harness.backtester import BacktestResult, FundingCarryBacktester
from alphaloop.plugins import BaseStrategyPlugin, PluginRegistry, default_registry

logger = logging.getLogger(__name__)


NON_ECONOMIC_CFG_FIELDS = frozenset({"version", "status", "notes", "strategy_id", "family"})


def economic_config_sha(cfg: StrategyConfig) -> str:
    """Hash ONLY economic strategy parameters so renaming version/notes/family never bypasses G6 DSR."""
    d = {k: v for k, v in cfg.to_dict().items() if k not in NON_ECONOMIC_CFG_FIELDS}
    if isinstance(d.get("symbols"), list):
        d["symbols"] = sorted(set(d["symbols"]))
    return sha256_obj(d)


def dsr_family_key(cfg: StrategyConfig) -> str:
    """Canonical immutable DSR family key based on plugin + venue pair (prevents family-rename reset)."""
    return f"{cfg.plugin_id or 'funding_carry'}:{cfg.venue_a}<->{cfg.venue_b}"


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
                    counted_for_dsr INTEGER NOT NULL DEFAULT 1,
                    verdict_json TEXT NOT NULL
                );
                """
            )
            cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(trial_ledger)").fetchall()}
            if "counted_for_dsr" not in cols:
                self.conn.execute("ALTER TABLE trial_ledger ADD COLUMN counted_for_dsr INTEGER NOT NULL DEFAULT 1")

    def count_family_trials(self, family: str) -> int:
        row = self.conn.fetch_one("SELECT COUNT(*) AS n FROM trial_ledger WHERE family=?", (family,))
        return int(row["n"]) if row else 0

    def count_effective_trials(self, family: str) -> int:
        """Count distinct (config_sha256, dataset_sha256) valid evaluations for G6 DSR (Architect P1-5)."""
        row = self.conn.fetch_one(
            """
            SELECT COUNT(DISTINCT config_sha256 || ':' || dataset_sha256) AS n
            FROM trial_ledger
            WHERE family=? AND counted_for_dsr=1
            """,
            (family,),
        )
        return int(row["n"]) if row else 0

    def has_effective_trial(self, family: str, config_sha256: str, dataset_sha256: str) -> bool:
        row = self.conn.fetch_one(
            """
            SELECT 1 FROM trial_ledger
            WHERE family=? AND config_sha256=? AND dataset_sha256=? AND counted_for_dsr=1
            LIMIT 1
            """,
            (family, config_sha256, dataset_sha256),
        )
        return row is not None

    def append(
        self,
        cfg: StrategyConfig,
        metrics: BacktestMetrics,
        verdict: GateVerdict,
        counted_for_dsr: bool = True,
        family_override: str | None = None,
        config_sha_override: str | None = None,
    ) -> int:
        fam = family_override or dsr_family_key(cfg)
        csha = config_sha_override or economic_config_sha(cfg)
        with transaction(self.conn):
            cur = self.conn.execute(
                """
                INSERT INTO trial_ledger
                (ts_ms, family, strategy_id, version, config_sha256, dataset_sha256,
                 provenance, net_pnl_usd, apr_pct, sharpe, max_dd_pct, overall_pass, counted_for_dsr, verdict_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now_ms(),
                    fam,
                    cfg.strategy_id,
                    cfg.version,
                    csha,
                    metrics.dataset_sha256,
                    metrics.provenance,
                    metrics.net_pnl_usd,
                    metrics.annualized_apr_pct,
                    metrics.sharpe_ratio,
                    metrics.max_drawdown_pct,
                    1 if verdict.overall_pass else 0,
                    1 if counted_for_dsr else 0,
                    json.dumps(verdict.to_dict(), sort_keys=True),
                ),
            )
            return int(cur.lastrowid)

    def evaluate_and_record_dsr_atomic(
        self,
        cfg: StrategyConfig,
        metrics: BacktestMetrics,
        checks_g1_to_g5: list[GateCheck],
        g1_ok: bool,
        dq_cert_id: str,
        dq_tier: str,
    ) -> GateVerdict:
        """Atomically compute G6 DSR hurdle and insert trial row in one transaction (Red-Team #4 TOCTOU fix)."""
        fam = dsr_family_key(cfg)
        csha = economic_config_sha(cfg)
        with transaction(self.conn):
            already_counted = self.has_effective_trial(fam, csha, metrics.dataset_sha256)
            count_this_for_dsr = g1_ok and (not already_counted)
            prior_eff = self.count_effective_trials(fam)
            trial_num = prior_eff + (1 if count_this_for_dsr else 0)
            trial_num_for_hurdle = max(1, trial_num)
            # Horizon-scaled Deflated Sharpe null hurdle: shorter windows have higher SR sampling variance
            sr_se_scale = math.sqrt(365.0 / max(30.0, float(metrics.days or 30.0)))
            null_threshold = math.sqrt(2.0 * math.log(max(2, trial_num_for_hurdle))) * 0.5 * sr_se_scale
            hurdle = max(0.8, null_threshold)
            g6_ok = metrics.sharpe_ratio >= hurdle
            dsr_note = (
                f"effective trial #{trial_num_for_hurdle} in family '{fam}' "
                f"(counted={count_this_for_dsr}, dup={already_counted}, se_scale={sr_se_scale:.2f}x); hurdle={hurdle:.2f}"
            )
            checks = list(checks_g1_to_g5) + [
                GateCheck(
                    gate_id="G6",
                    name="Multiple-Testing Deflated Sharpe Hurdle",
                    passed=g6_ok,
                    value=f"Sharpe={metrics.sharpe_ratio:.2f} (eff_trial #{trial_num_for_hurdle})",
                    threshold=f"Sharpe>={hurdle:.2f}",
                    detail=dsr_note,
                )
            ]
            passed_count = sum(1 for c in checks if c.passed)
            verdict = GateVerdict(
                strategy_id=cfg.strategy_id,
                version=cfg.version,
                overall_pass=(passed_count == len(checks)),
                passed_count=passed_count,
                total_count=len(checks),
                trial_number=trial_num_for_hurdle,
                deflated_sharpe_note=dsr_note,
                checks=checks,
                dq_cert_id=dq_cert_id,
                dq_tier=dq_tier,
            )
            self.conn.execute(
                """
                INSERT INTO trial_ledger
                (ts_ms, family, strategy_id, version, config_sha256, dataset_sha256,
                 provenance, net_pnl_usd, apr_pct, sharpe, max_dd_pct, overall_pass, counted_for_dsr, verdict_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now_ms(),
                    fam,
                    cfg.strategy_id,
                    cfg.version,
                    csha,
                    metrics.dataset_sha256,
                    metrics.provenance,
                    metrics.net_pnl_usd,
                    metrics.annualized_apr_pct,
                    metrics.sharpe_ratio,
                    metrics.max_drawdown_pct,
                    1 if verdict.overall_pass else 0,
                    1 if count_this_for_dsr else 0,
                    json.dumps(verdict.to_dict(), sort_keys=True),
                ),
            )
            return verdict

    def recent_trials(self, limit: int = 30) -> list[dict[str, Any]]:
        rows = self.conn.fetch_all(
            """
            SELECT trial_id, ts_ms, family, strategy_id, version, provenance,
                   net_pnl_usd, apr_pct, sharpe, max_dd_pct, overall_pass, counted_for_dsr
            FROM trial_ledger ORDER BY trial_id DESC LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()


class SixGateVerifier:
    """Plugin-agnostic 6-Gate verifier with integrated DQ-01..DQ-08 hard gate."""

    def __init__(
        self,
        backtester: FundingCarryBacktester,
        ledger: TrialLedger,
        registry: PluginRegistry | None = None,
    ) -> None:
        self.backtester = backtester
        self.ledger = ledger
        self.registry = registry or default_registry()
        self.dq_auditor = DataQualityAuditor(backtester.store)

    def verify(
        self,
        cfg: StrategyConfig,
        start_ms: int | None = None,
        end_ms: int | None = None,
        allow_synthetic_in_unit_test: bool = False,
        plugin: BaseStrategyPlugin | None = None,
    ) -> tuple[GateVerdict, BacktestResult, dict[str, Any]]:
        """Run DQ certification + full 6-Gate verification battery and log trial atomically."""
        store = self.backtester.store
        active_plugin = plugin or self.registry.resolve_for_config(cfg)

        # Pre-flight Data Quality Certification (DQ-01..DQ-08)
        csv_path = str(cfg.params.get("event_csv_path", "")).strip()
        if csv_path and Path(csv_path).exists():
            csv_prov = str(cfg.params.get("csv_provenance", "UNVERIFIED_LOCAL_CSV"))
            dq_report = self.dq_auditor.audit_csv_sample(
                csv_path,
                dataset_name=f"{cfg.strategy_id}@{cfg.version}",
                venue_pair=f"csv:{Path(csv_path).name}",
                persist=True,
                provenance=csv_prov,
            )
        else:
            vb_dq = f"{cfg.venue_a}_spot" if cfg.mode == "spot_perp" else cfg.venue_b
            dq_report = self.dq_auditor.audit_funding_pair(
                cfg.venue_a,
                vb_dq,
                list(cfg.symbols),
                start_ms=start_ms,
                end_ms=end_ms,
                dataset_name=f"{cfg.strategy_id}@{cfg.version}",
                persist=True,
                allow_synthetic_in_unit_test=allow_synthetic_in_unit_test,
            )

        main_res = active_plugin.evaluate(store, cfg, start_ms=start_ms, end_ms=end_ms)
        m = main_res.metrics
        checks: list[GateCheck] = []

        # G1: Data Quality Certification, Provenance & Sample Floor
        prov_ok = (m.provenance in ALLOWED_REAL_PROVENANCES) or allow_synthetic_in_unit_test
        g1_ok = dq_report.is_usable_for_research and prov_ok and m.trade_count >= 5
        dq_failed_ids = [c.gate_id for c in dq_report.checks if not c.passed and c.severity == "BLOCKING"]
        checks.append(
            GateCheck(
                gate_id="G1",
                name="Data Quality Cert, Provenance & Sample Floor",
                passed=g1_ok,
                value=f"{dq_report.tier} | {m.provenance} (trades={m.trade_count})",
                threshold=f"NOT DATA_REJECTED & {PROVENANCE_REAL} & trades>=5",
                detail=(
                    f"dq={dq_report.cert_id} ({dq_report.passed_count}/{dq_report.total_count}, "
                    f"basis_cov={dq_report.basis_coverage_pct:.1f}%, failed={dq_failed_ids})"
                ),
            )
        )

        # G2: Economic Expectancy, Cost Stress (+2bp/leg), Taker-Hedge Survivability & Zero-Liquidation (P1-1/P1-2/RT#10)
        stress_res = active_plugin.evaluate(
            store, cfg, start_ms=start_ms, end_ms=end_ms, extra_slippage_bps_per_leg=2.0
        )
        sm = stress_res.metrics
        taker_hedge_res = active_plugin.evaluate(
            store, cfg, start_ms=start_ms, end_ms=end_ms, execution_mode_override="MAKER_THEN_TAKER_HEDGE"
        )
        thm = taker_hedge_res.metrics
        g2_ok = (
            m.net_pnl_usd > 0.0
            and sm.net_pnl_usd > 0.0
            and thm.net_pnl_usd > 0.0
            and m.annualized_apr_pct >= 5.0
            and m.max_drawdown_pct <= 10.0
            and m.fee_to_gross_ratio <= 0.65
            and m.liquidation_count == 0
            and sm.liquidation_count == 0
            and thm.liquidation_count == 0
        )
        checks.append(
            GateCheck(
                gate_id="G2",
                name="Cost-Stressed Expectancy, Taker-Hedge & Margin Safety",
                passed=g2_ok,
                value=f"MAIN={m.annualized_apr_pct:+.1f}%/DD={m.max_drawdown_pct:.2f}%/Liq={m.liquidation_count} | STRESS={sm.annualized_apr_pct:+.1f}% | TH=${thm.net_pnl_usd:+.1f}",
                threshold="MAIN>=5% & STRESS>0 & TH>0 & DD<=10% & Cost/Gross<=65% & Liq=0",
                detail=(
                    f"main_net=${m.net_pnl_usd:+.2f}, stress_net=${sm.net_pnl_usd:+.2f}, "
                    f"taker_hedge_net=${thm.net_pnl_usd:+.2f}, cost_ratio={m.fee_to_gross_ratio:.2%}, "
                    f"forced_rebal={m.forced_rebalance_count}"
                ),
            )
        )

        # G3: Calendar-Time Causal Walk-Forward 70% IS / 30% OOS (continuous boundary, fail-closed on short curves)
        eq = main_res.equity_curve
        if len(eq) >= 4 and eq[-1][0] > eq[0][0]:
            split_ts = eq[0][0] + int((eq[-1][0] - eq[0][0]) * 0.70)
            split_idx = next((i for i, (ts, _) in enumerate(eq) if ts >= split_ts), len(eq))
            split_idx = max(2, min(split_idx, len(eq) - 1))
            is_seg = eq[:split_idx]
            # Anchor OOS at split_idx - 1 so the boundary transition return is preserved continuously
            oos_seg = eq[split_idx - 1:]
        else:
            is_seg = []
            oos_seg = []

        def _seg_summary(seg: list[tuple[int, float]]) -> dict[str, float]:
            if len(seg) < 2 or seg[0][1] <= 0 or seg[-1][0] <= seg[0][0]:
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
        g3_ok = (
            len(is_seg) >= 2
            and len(oos_seg) >= 2
            and is_info["net_pnl_usd"] > 0.0
            and oos_info["net_pnl_usd"] > 0.0
            and retention >= 0.25
        )
        checks.append(
            GateCheck(
                gate_id="G3",
                name="Continuous Calendar-Time 70/30 OOS Consistency",
                passed=g3_ok,
                value=f"IS={is_apr:+.1f}% ({is_info['days']}d) | OOS={oos_apr:+.1f}% ({oos_info['days']}d, ret={retention:.2f}x)",
                threshold="IS>0 & OOS>0 & OOS/IS>=0.25",
                detail=f"IS_net=${is_info['net_pnl_usd']:+.2f}, OOS_net=${oos_info['net_pnl_usd']:+.2f}",
            )
        )

        # G4: Plugin-Delegated Parameter Plateau Stability (3x3 Neighborhood, P0-5)
        plateau_grid: list[dict[str, Any]] = []
        variants = active_plugin.plateau_variants(cfg)
        pos_cells = 0
        aprs: list[float] = []
        for label, cand_cfg in variants:
            r = active_plugin.evaluate(store, cand_cfg, start_ms=start_ms, end_ms=end_ms)
            apr_val = r.metrics.annualized_apr_pct
            aprs.append(apr_val)
            if r.metrics.net_pnl_usd > 0.0:
                pos_cells += 1
            plateau_grid.append({
                "label": label,
                "entry_apr": cand_cfg.entry_spread_apr_pct,
                "exit_apr": cand_cfg.exit_spread_apr_pct,
                "apr_pct": apr_val,
                "sharpe": r.metrics.sharpe_ratio,
                "net_pnl_usd": r.metrics.net_pnl_usd,
                "trades": r.metrics.trade_count,
            })
        pos_ratio = (pos_cells / len(plateau_grid)) if plateau_grid else 0.0
        avg_neighbor = (sum(aprs) / len(aprs)) if aprs else 0.0
        peak_ratio = (m.annualized_apr_pct / avg_neighbor) if avg_neighbor > 0.5 else 99.0
        g4_ok = pos_ratio >= 0.75 and (0.5 <= peak_ratio <= 1.8)
        checks.append(
            GateCheck(
                gate_id="G4",
                name=f"Parameter Plateau Stability ({len(plateau_grid)}-Cell Grid)",
                passed=g4_ok,
                value=f"{pos_cells}/{len(plateau_grid)} positive ({pos_ratio:.0%}), center/avg={peak_ratio:.2f}x",
                threshold=">=75% positive & center/avg in [0.5, 1.8]",
                detail=f"grid APR range [{min(aprs):+.1f}%, {max(aprs):+.1f}%]" if aprs else "empty grid",
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

        # G6: Atomic Multiple-Testing DSR Evaluation + Append (Red-Team #4 & #10)
        verdict = self.ledger.evaluate_and_record_dsr_atomic(
            cfg=cfg,
            metrics=m,
            checks_g1_to_g5=checks,
            g1_ok=g1_ok,
            dq_cert_id=dq_report.cert_id,
            dq_tier=dq_report.tier,
        )

        extras = {
            "dq_report": dq_report.to_dict(),
            "stress_metrics": sm.to_dict(),
            "taker_hedge_metrics": thm.to_dict(),
            "is_metrics": is_info,
            "oos_metrics": oos_info,
            "plateau_grid": plateau_grid,
        }
        return verdict, main_res, extras
