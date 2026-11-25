#!/usr/bin/env python3
"""Empirical Data Quality Audit (DQ-01..DQ-08) across:
  1. 90-day 4-exchange SQLite lake (funding + 1h OHLCV candles)
  2. Live collected 4-exchange BBO orderbook ticks (Hyperliquid / Lighter / OKX / Gate)
  3. Bounded read-only CSV samples pulled from configured fleet nodes via read-only SSH

Zero synthetic data; zero mutations on remote servers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.core.provenance import utc_iso
from alphaloop.data.bbo_collector import DEFAULT_BBO_SYMBOLS
from alphaloop.data.quality import DataQualityAuditor
from alphaloop.data.store import MarketDataStore
from alphaloop.fleet.ssh_auditor import ReadOnlyFleetAuditor
from alphaloop.plugins.structural_arb_plugin import StructuralArbPlugin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dq_empirical")


def _pull_csv_with_header(
    auditor: ReadOnlyFleetAuditor, ip: str, remote_path: str, out_file: Path, tail_rows: int = 500
) -> tuple[bool, int, str]:
    """Pull CSV header + tail N rows strictly read-only via ReadOnlyFleetAuditor allowlisted helpers."""
    ok_h, header, err_h = auditor.read_remote_head_line(ip, remote_path)
    if not ok_h or not header:
        return False, 0, err_h or "head failed"
    ok_t, tail_out, err_t = auditor.read_remote_tail(ip, remote_path, lines=tail_rows)
    if not ok_t:
        return False, 0, err_t or "tail failed"
    lines = [ln for ln in tail_out.splitlines() if ln.strip()]
    body_lines = lines[1:] if (lines and lines[0].strip() == header) else lines
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(header + "\n" + "\n".join(body_lines) + "\n", encoding="utf-8")
    return True, len(body_lines), header


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=".dq.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Empirical DQ-01..DQ-08 audit across lake, BBO ticks, and fleet CSVs")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--roles", default=str(ROOT / "config" / "fleet_roles.json"))
    p.add_argument("--ssh-key", default=os.environ.get("ALPHALOOP_SSH_KEY") or str(Path.home() / ".ssh" / "id_ed25519"))
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    roles_path = Path(args.roles)
    ssh_key = Path(args.ssh_key)

    store = MarketDataStore(data_dir / "market_data.sqlite")
    dq = DataQualityAuditor(store)
    fleet = ReadOnlyFleetAuditor(roles_path, ssh_key) if roles_path.exists() else None
    node_ips = {n["node_id"]: n["ip"] for n in (fleet.config.get("nodes", []) if fleet else [])}
    csv_targets = list(fleet.config.get("csv_targets", [])) if fleet else []

    results: dict[str, Any] = {
        "generated_utc": utc_iso(),
        "funding_pairs": {},
        "bbo_live_pairs": {},
        "server_csv_samples": {},
        "structural_arb_burst": {},
    }

    # 1. Audit 90-day 4-exchange SQLite lake pairs
    def valid_syms(va: str, vb: str | None) -> list[str]:
        all_s = [r["symbol"] for r in store.conn.execute("SELECT DISTINCT symbol FROM funding_rates WHERE venue=?", (va,)).fetchall()]
        out = []
        for s in sorted(all_s):
            if len(store.load_funding(va, s)) < 100:
                continue
            if vb and not vb.endswith("_spot") and len(store.load_funding(vb, s)) < 100:
                continue
            out.append(s)
        return out

    pairs = [
        ("lighter", "gate"),
        ("lighter", "okx"),
        ("hyperliquid", "okx"),
        ("hyperliquid", "gate"),
        ("hyperliquid", "lighter"),
        ("okx", "okx_spot"),
    ]
    logger.info("=== 1. AUDITING 90-DAY 4-EXCHANGE SQLITE LAKE ===")
    for va, vb in pairs:
        syms = valid_syms(va, vb)
        rep = dq.audit_funding_pair(va, vb, syms, dataset_name=f"lake_90d:{va}<->{vb}", persist=True)
        results["funding_pairs"][f"{va}<->{vb}"] = rep.to_dict()
        logger.info(
            "  %-22s | syms=%2d | rows=%6d | basis_cov=%5.1f%% | checks=%d/%d | tier=%s",
            f"{va}<->{vb}", len(syms), rep.row_count, rep.basis_coverage_pct, rep.passed_count, rep.total_count, rep.tier,
        )

    # 2. Audit real live collected BBO ticks
    logger.info("=== 2. AUDITING REAL LIVE BBO ORDERBOOK TICKS ===")
    bbo_pairs = [
        ("lighter", "gate"),
        ("lighter", "okx"),
        ("hyperliquid", "okx"),
        ("hyperliquid", "gate"),
        ("hyperliquid", "lighter"),
    ]
    for va, vb in bbo_pairs:
        rep, bbo_sum = dq.audit_bbo_ticks(
            va, vb, DEFAULT_BBO_SYMBOLS,
            dataset_name=f"live_bbo:{va}<->{vb}",
            persist=True,
            max_stale_sec=None,
        )
        results["bbo_live_pairs"][f"{va}<->{vb}"] = {
            "report": rep.to_dict(),
            "summary": bbo_sum,
        }
        logger.info(
            "  BBO %-19s | ticks=%4d | aligned=%3d | spA=%5.2fbp | spB=%5.2fbp | combHalf=%5.2fbp | medBasis=%+6.2fbp | tier=%s",
            f"{va}<->{vb}",
            bbo_sum["total_ticks"],
            bbo_sum["aligned_pairs"],
            bbo_sum["venue_a_med_spread_bps"],
            bbo_sum["venue_b_med_spread_bps"],
            bbo_sum["combined_half_spread_cost_bps"],
            bbo_sum["med_basis_bps"],
            rep.tier,
        )

    # 3. Pull bounded read-only CSV samples configured in local gitignored fleet_roles.json
    logger.info("=== 3. AUDITING CONFIGURED SERVER CSV SAMPLES (READ-ONLY) ===")
    sample_dir = data_dir / "fleet_audit" / "csv_samples"
    burst_csv_local: Path | None = None
    for target in csv_targets:
        name = str(target["name"])
        node_id = str(target["node_id"])
        rpath = str(target["remote_path"])
        ts_col = str(target.get("ts_col", "ts_ms"))
        pcols = list(target.get("price_cols", []))
        ip = node_ips.get(node_id, "")
        if not fleet or not ip or not ssh_key.exists():
            results["server_csv_samples"][name] = {"ok": False, "error": f"No local fleet config/SSH key for {node_id}"}
            continue
        out_f = sample_dir / f"{name}.csv"
        ok, _n_rows, info = _pull_csv_with_header(fleet, ip, rpath, out_f, tail_rows=500)
        if not ok:
            logger.warning("  SKIP %-28s -> %s", name, info)
            results["server_csv_samples"][name] = {"ok": False, "error": info}
            continue
        cols = [c.strip() for c in info.split(",")]
        eff_ts = ts_col if ts_col in cols else next((c for c in cols if "ts" in c.lower() or "time" in c.lower() or "date" in c.lower()), cols[0])
        eff_pcols = [c for c in pcols if c in cols]
        rep = dq.audit_csv_sample(
            out_f,
            dataset_name=name,
            ts_col=eff_ts,
            price_cols=eff_pcols,
            venue_pair=f"server:{Path(rpath).name}",
            persist=True,
            provenance="SERVER_COLLECTED_LIVE",
        )
        results["server_csv_samples"][name] = {
            "ok": True,
            "header_cols": cols,
            "detected_ts_col": eff_ts,
            "report": rep.to_dict(),
        }
        if target.get("evaluate_structural_arb"):
            burst_csv_local = out_f
        logger.info(
            "  CSV %-28s | rows=%4d | ts_col=%-14s | checks=%d/%d | tier=%s",
            name, rep.row_count, eff_ts, rep.passed_count, rep.total_count, rep.tier,
        )

    # 4. Evaluate StructuralArbPlugin on burst CSV sample if configured & pulled
    if burst_csv_local and burst_csv_local.exists():
        plug = StructuralArbPlugin()
        cfg_all = plug.default_config(
            strategy_id="burst_all_events",
            params={"event_csv_path": str(burst_csv_local), "csv_provenance": "SERVER_COLLECTED_LIVE", "min_edge_bps": 10.0, "gas_usd_per_tx": 0.15, "revert_rate_pct": 5.0, "rebalance_cost_bps": 4.0, "confirm_decay_bps": 2.0, "notional_per_trade_usd": 500.0},
        )
        res_all = plug.evaluate(store, cfg_all)
        cfg_usdc = plug.default_config(
            strategy_id="burst_usdc_dip_only",
            params={"event_csv_path": str(burst_csv_local), "csv_provenance": "SERVER_COLLECTED_LIVE", "event_type_filter": "USDC_DIP", "min_edge_bps": 10.0, "gas_usd_per_tx": 0.15, "revert_rate_pct": 5.0, "rebalance_cost_bps": 4.0, "confirm_decay_bps": 2.0, "notional_per_trade_usd": 500.0},
        )
        res_usdc = plug.evaluate(store, cfg_usdc)
        results["structural_arb_burst"] = {
            "all_events": res_all.metrics.to_dict(),
            "usdc_dip_only": res_usdc.metrics.to_dict(),
        }
        logger.info(
            "  StructuralArb Burst ALL: trades=%d net=$%+.2f | USDC_DIP: trades=%d net=$%+.2f",
            res_all.metrics.trade_count, res_all.metrics.net_pnl_usd,
            res_usdc.metrics.trade_count, res_usdc.metrics.net_pnl_usd,
        )

    out_path = data_dir / "fleet_audit" / "dq_empirical_report.json"
    _atomic_write_json(out_path, results)
    logger.info("Saved empirical DQ & BBO report -> %s", out_path)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
