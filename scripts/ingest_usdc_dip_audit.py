#!/usr/bin/env python3
"""Pillar 3 Event-Pulse (USDC_DIP) Automated DQ & Sample-Threshold Evaluator (Stage 3 S3-01, Red-Team #7 Hardened).

Hard Quantitative & Provenance Invariants:
  - NEVER forges `SERVER_COLLECTED_LIVE` provenance: defaults to `UNVERIFIED_LOCAL_CSV`
    unless explicitly verified via `--provenance SERVER_COLLECTED_LIVE`.
  - Auto-detects `ts_iso` vs `ts_ms` timestamp header (or explicit `--ts-col`).
  - Uses actual cryptographic file hash `dq_rep.dataset_sha256` (zero summary-dict hash forgery).
  - Deducts full friction: fill-rate haircut `(1 - revert_prob)`, explicit DEX+CEX swap/taker
    fees & slippage, edge decay, rebalance drag, gas USD, and revert gas cost; never trusts
    raw reported PnL without conservative re-pricing; never invents hidden `$500` notional.
  - Strictly enforces `DQ-01` sample sufficiency (`min_samples = 30`) before gate passage and
    records evaluated trials into the canonical `trial_ledger.sqlite` (G6 anti-selection-bias).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.core.provenance import utc_now_ms
from alphaloop.daemon.shadow_runner import _atomic_write_json
from alphaloop.data.quality import DataQualityAuditor
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.gates import TrialLedger
from alphaloop.plugins import StructuralArbPlugin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("usdc-dip-audit")


def _safe_float(val: Any) -> float | None:
    if val is None or str(val).strip() == "":
        return None
    try:
        x = float(val)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def evaluate_usdc_dip_csv(
    csv_path: str | Path,
    *,
    data_dir: str | Path = "data",
    min_samples: int = 30,
    provenance: str = "UNVERIFIED_LOCAL_CSV",
    ts_col: str | None = None,
    default_notional_usd: float | None = None,
    dex_cex_fee_slip_bps: float = 8.0,
    gas_usd_per_trade: float = 0.08,
    revert_prob: float = 0.06,
    revert_gas_usd: float = 0.05,
    edge_decay_bps: float = 3.0,
    rebalance_bps: float = 4.0,
) -> dict[str, Any]:
    """Audit a burst paper-trade CSV for USDC_DIP sample sufficiency and full-friction EV."""
    path = Path(csv_path)
    if not path.exists():
        return {
            "status": "FILE_NOT_FOUND",
            "csv_path": str(path),
            "valid_samples": 0,
            "min_samples": int(min_samples),
            "gate_eligible": False,
        }

    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh_hdr:
        hdr_reader = csv.reader(fh_hdr)
        headers = [h.strip() for h in (next(hdr_reader, None) or [])]
    resolved_ts_col = ts_col or ("ts_iso" if "ts_iso" in headers else "ts_ms")

    root_dir = Path(data_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    store = MarketDataStore(root_dir / "market_data.sqlite")
    auditor = DataQualityAuditor(store)
    try:
        dq_rep = auditor.audit_csv_sample(
            path,
            dataset_name=f"pillar3_usdc_dip:{path.name}",
            ts_col=resolved_ts_col,
            venue_pair="pillar3_usdc_dip",
            persist=False,
            provenance=provenance,
        )
    finally:
        store.close()

    rows: list[dict[str, Any]] = []
    skipped_missing_notional = 0
    notional_assumed = False

    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            event_type = str(
                raw.get("track") or raw.get("strategy") or raw.get("event_type") or raw.get("signal_type") or ""
            ).upper()
            symbol = str(raw.get("symbol") or raw.get("pair") or "").upper()
            if "USDC_DIP" not in event_type and "USDC" not in symbol:
                continue

            notional = _safe_float(raw.get("notional_usd") or raw.get("size_usd"))
            if notional is None:
                if default_notional_usd is not None and default_notional_usd > 0:
                    notional = float(default_notional_usd)
                    notional_assumed = True
                else:
                    skipped_missing_notional += 1
                    continue
            if notional <= 0:
                skipped_missing_notional += 1
                continue

            gross_bps = _safe_float(
                raw.get("realized_net_bp")
                or raw.get("raw_net_bp")
                or raw.get("edge_bps")
                or raw.get("gross_bps")
                or raw.get("spread_bps")
            )
            if gross_bps is None:
                continue

            # Full conservative friction re-pricing: fill haircut (1 - revert_prob) + swap/taker/slip + decay + rebalance + gas
            fill_prob = max(0.0, min(1.0, 1.0 - float(revert_prob)))
            gross_usd = notional * ((gross_bps * fill_prob) / 10_000.0)
            fee_slip_usd = (
                notional * ((dex_cex_fee_slip_bps + edge_decay_bps) / 10_000.0)
                + (fill_prob * gas_usd_per_trade)
                + (float(revert_prob) * revert_gas_usd)
            )
            rebalance_usd = notional * (rebalance_bps / 10_000.0)
            repriced_usd = gross_usd - fee_slip_usd - rebalance_usd
            rep_val = _safe_float(raw.get("net_pnl_usd") or raw.get("pnl_usd"))
            realized_usd = min(repriced_usd, rep_val) if rep_val is not None else repriced_usd

            rows.append({
                "event_type": event_type or "USDC_DIP",
                "symbol": symbol or "USDC",
                "notional_usd": round(notional, 2),
                "gross_bps": round(gross_bps, 2),
                "gross_usd": round(gross_usd, 4),
                "fee_slip_usd": round(fee_slip_usd, 4),
                "rebalance_usd": round(rebalance_usd, 4),
                "conservative_net_usd": round(realized_usd, 4),
            })

    n_valid = len(rows)
    pnls = [r["conservative_net_usd"] for r in rows]
    total_net = sum(pnls)
    wins = sum(1 for x in pnls if x > 0)
    win_rate = (wins / n_valid) if n_valid > 0 else 0.0
    med_net = statistics.median(pnls) if n_valid > 0 else 0.0
    ex_top5 = sum(sorted(pnls, reverse=True)[5:]) if n_valid > 5 else 0.0

    # Chronological 70/30 OOS split on realized event sequence
    split_idx = int(n_valid * 0.7)
    is_net = sum(pnls[:split_idx]) if split_idx > 0 else 0.0
    oos_net = sum(pnls[split_idx:]) if split_idx < n_valid else 0.0

    base_out: dict[str, Any] = {
        "ts_ms": utc_now_ms(),
        "provenance": provenance,
        "ts_col": resolved_ts_col,
        "csv_sha256": dq_rep.dataset_sha256,
        "dq_tier": dq_rep.tier,
        "dq_passed": f"{dq_rep.passed_count}/{dq_rep.total_count}",
        "valid_samples": n_valid,
        "skipped_missing_notional": skipped_missing_notional,
        "notional_assumed": notional_assumed,
        "min_samples": int(min_samples),
        "progress_pct": round(100.0 * min(n_valid, int(min_samples)) / max(1, int(min_samples)), 1),
        "conservative_net_pnl_usd": round(total_net, 4),
        "win_rate": round(win_rate, 4),
        "median_trade_net_usd": round(med_net, 4),
        "ex_top5_net_pnl_usd": round(ex_top5, 4),
        "is_70pct_net_usd": round(is_net, 4),
        "oos_30pct_net_usd": round(oos_net, 4),
    }

    if n_valid < int(min_samples):
        base_out.update({
            "status": f"WAITING_FOR_SAMPLES ({n_valid}/{min_samples})",
            "gate_eligible": False,
            "verdict": "HOLD_ACCUMULATE_MORE_REAL_EVENTS",
        })
        return base_out

    # Sample threshold met -> enforce DQ provenance + OOS + ex-top5 + win-rate gates and record in TrialLedger
    passed_gates = (
        dq_rep.tier == "DATA_QUALIFIED_FULL"
        and total_net > 0
        and ex_top5 > 0
        and is_net > 0
        and oos_net > 0
        and win_rate >= 0.55
    )
    plugin = StructuralArbPlugin()
    plugin_meta = plugin.metadata()
    ledger = TrialLedger(root_dir / "trial_ledger.sqlite")
    try:
        from alphaloop.core.models import BacktestMetrics, GateCheck, GateVerdict

        ev_cfg = plugin.default_config(
            symbols=["USDC"],
            strategy_id="pillar3_usdc_dip",
            version=f"csv-{dq_rep.dataset_sha256[:8]}",
            venue_a="dex",
            venue_b="cex",
            capital_usd=3000.0,
        )
        win_sum = sum(x for x in pnls if x > 0)
        loss_sum = abs(sum(x for x in pnls if x < 0))
        real_pf = (win_sum / loss_sum) if loss_sum > 0 else (99.0 if win_sum > 0 else 0.0)
        eq_cur = 3000.0
        eq_peak = 3000.0
        max_dd = 0.0
        for p in pnls:
            eq_cur += p
            if eq_cur > eq_peak:
                eq_peak = eq_cur
            if eq_peak > 0:
                dd = ((eq_peak - eq_cur) / eq_peak) * 100.0
                if dd > max_dd:
                    max_dd = dd
        real_fees = sum(float(r.get("fee_slip_usd", 0.0)) for r in rows)
        real_rebal = sum(float(r.get("rebalance_usd", 0.0)) for r in rows)
        real_gross = sum(float(r.get("gross_usd", 0.0)) for r in rows)
        mean_p = (total_net / n_valid) if n_valid > 0 else 0.0
        std_p = statistics.stdev(pnls) if n_valid >= 3 else 0.0
        real_sharpe = ((mean_p / std_p) * math.sqrt(n_valid)) if std_p > 1e-12 else 0.0
        ret_pct = (total_net / 3000.0) * 100.0
        synthetic_m = BacktestMetrics(
            window_start_ms=0,
            window_end_ms=0,
            days=max(1.0, float(n_valid)),
            capital_usd=3000.0,
            trade_count=n_valid,
            win_count=wins,
            win_rate=round(win_rate, 4),
            avg_hold_hours=0.1,
            gross_funding_usd=0.0,
            basis_pnl_usd=round(real_gross, 4),
            total_fees_usd=round(real_fees, 4),
            total_slippage_usd=0.0,
            total_rebalance_usd=round(real_rebal, 4),
            net_pnl_usd=round(total_net, 4),
            total_return_pct=round(ret_pct, 4),
            annualized_apr_pct=round(ret_pct, 4),
            sharpe_ratio=round(real_sharpe, 4),
            calmar_ratio=round(ret_pct / max_dd, 4) if max_dd > 1e-9 else 0.0,
            max_drawdown_pct=round(max_dd, 4),
            profit_factor=round(real_pf, 4),
            fee_to_gross_ratio=round((real_fees + real_rebal) / real_gross, 4) if real_gross > 1e-9 else 0.0,
            avg_capital_utilization_pct=0.0,
            monthly_usd_est=round(total_net, 4),
            top5_removed_net_pnl_usd=round(ex_top5, 4),
            top_symbol_removed_net_pnl_usd=round(ex_top5, 4),
            best_symbol="USDC",
            provenance=provenance,
            dataset_sha256=dq_rep.dataset_sha256,
            gross_alpha_usd=round(real_gross, 4),
        )
        checks = [
            GateCheck("G1", "DQ_PROVENANCE", dq_rep.tier == "DATA_QUALIFIED_FULL", dq_rep.tier, "DATA_QUALIFIED_FULL", "CSV DQ"),
            GateCheck("G2", "NET_EV", total_net > 0, f"{total_net:.4f}", ">0", "Full-friction EV"),
            GateCheck("G3", "EX_TOP5_EV", ex_top5 > 0, f"{ex_top5:.4f}", ">0", "Top-5 concentration"),
            GateCheck("G4", "OOS_CONSISTENCY", is_net > 0 and oos_net > 0, f"IS={is_net:.2f}/OOS={oos_net:.2f}", ">0", "70/30 OOS"),
            GateCheck("G5", "WIN_RATE", win_rate >= 0.55, f"{win_rate:.3f}", ">=0.55", "Win rate"),
            GateCheck("G6", "SAMPLE_COUNT", n_valid >= min_samples, str(n_valid), f">={min_samples}", "Min samples"),
        ]
        ev_verdict = GateVerdict(
            strategy_id=ev_cfg.strategy_id,
            version=ev_cfg.version,
            overall_pass=passed_gates,
            passed_count=sum(1 for c in checks if c.passed),
            total_count=len(checks),
            trial_number=ledger.count_effective_trials("structural_arb:dex<->cex") + 1,
            deflated_sharpe_note=f"event_pulse_n={n_valid}",
            checks=checks,
            dq_cert_id=dq_rep.cert_id,
            dq_tier=dq_rep.tier,
        )
        trial_id = ledger.append(
            ev_cfg,
            synthetic_m,
            ev_verdict,
            counted_for_dsr=(dq_rep.tier == "DATA_QUALIFIED_FULL"),
        )
    finally:
        ledger.close()

    base_out.update({
        "status": "READY_FOR_SHADOW_PROMOTION_REVIEW" if passed_gates else "REJECTED_INSUFFICIENT_EDGE",
        "gate_eligible": True,
        "passed_event_gates": passed_gates,
        "plugin_id": plugin_meta["plugin_id"],
        "trial_id": trial_id,
        "verdict": "PASS_EVENT_GATES" if passed_gates else "FAIL_EVENT_GATES",
    })
    return base_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pillar 3 USDC_DIP Automated DQ & Sample Gate Auditor")
    parser.add_argument("--csv", required=True, help="Path to read-only burst_paper_trades.csv sample")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--provenance", default="UNVERIFIED_LOCAL_CSV", help="Explicit verified provenance label")
    parser.add_argument("--ts-col", default="", help="Optional timestamp column name override")
    parser.add_argument("--default-notional-usd", type=float, default=None, help="Explicit notional for bps-only CSVs")
    parser.add_argument("--out", default="", help="Optional atomic JSON output path")
    args = parser.parse_args(argv)

    result = evaluate_usdc_dip_csv(
        args.csv,
        data_dir=args.data_dir,
        min_samples=args.min_samples,
        provenance=args.provenance,
        ts_col=args.ts_col or None,
        default_notional_usd=args.default_notional_usd,
    )
    if args.out:
        _atomic_write_json(Path(args.out), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
