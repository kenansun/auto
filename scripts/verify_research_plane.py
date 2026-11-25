#!/usr/bin/env python3
"""Research Plane Accelerator Verification & Benchmark CLI (`requirements-research.txt`).

Dual-Plane Architecture (Verdict A):
  - Edge / Shadow nodes run pure Python 3.12 stdlib (`EDGE_STDLIB_MODE`, EXIT 0).
  - Elastic Research nodes with `requirements-research.txt` installed run permissive-license
    acceleration verification (Parquet export, `:memory:` DuckDB SQL sandbox + 0 disk `.duckdb`,
    and vectorized stats benchmark) without polluting production edge nodes.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.platform.data_foundation.router import DataFoundationRouter

OPTIONAL_ACCELERATORS: tuple[tuple[str, str], ...] = (
    ("numpy", "BSD-3-Clause"),
    ("scipy", "BSD-3-Clause"),
    ("pyarrow", "Apache-2.0"),
    ("duckdb", "MIT"),
    ("polars", "MIT"),
    ("pytest", "MIT"),
)

FORBIDDEN_COPYLEFT_TOKENS: tuple[str, ...] = ("GPL", "AGPL", "LGPL", "MPL")


def probe_accelerators() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for pkg, expected_lic in OPTIONAL_ACCELERATORS:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", None)
            if not ver:
                try:
                    ver = importlib.metadata.version(pkg)
                except importlib.metadata.PackageNotFoundError:
                    ver = "unknown"
            lic_text = ""
            try:
                meta = importlib.metadata.metadata(pkg)
                lic_text = str(meta.get("License-Expression") or meta.get("License") or "")[:120]
            except importlib.metadata.PackageNotFoundError:
                lic_text = expected_lic
            copyleft_hit = any(tok in lic_text.upper() for tok in FORBIDDEN_COPYLEFT_TOKENS)
            status[pkg] = {
                "installed": True,
                "version": str(ver),
                "expected_license": expected_lic,
                "reported_license": lic_text,
                "copyleft_clean": not copyleft_hit,
            }
        except ImportError:
            status[pkg] = {
                "installed": False,
                "version": None,
                "expected_license": expected_lic,
                "reported_license": None,
                "copyleft_clean": True,
            }
    return status


def benchmark_research_plane(lake_dir: Path) -> dict[str, Any]:
    accel = probe_accelerators()
    all_core_installed = all(accel[k]["installed"] for k in ("numpy", "pyarrow", "duckdb"))
    copyleft_clean = all(v["copyleft_clean"] for v in accel.values())

    report: dict[str, Any] = {
        "mode": "RESEARCH_ACCELERATED_MODE" if all_core_installed else "EDGE_STDLIB_MODE",
        "copyleft_clean": copyleft_clean,
        "accelerators": accel,
        "benchmarks": {},
    }

    if not copyleft_clean:
        report["status"] = "FAIL_COPYLEFT_VIOLATION"
        return report

    if not all_core_installed:
        report["status"] = "PASS_EDGE_STDLIB_MODE"
        return report

    if not (lake_dir / "catalog.sqlite").is_file():
        report["status"] = "PASS_PACKAGES_READY_NO_LOCAL_LAKE"
        return report

    import shutil

    with tempfile.TemporaryDirectory(prefix="research_plane_bench_") as td:
        tmp_lake = Path(td) / "lake"
        (tmp_lake / "domains").mkdir(parents=True, exist_ok=True)
        shutil.copy2(lake_dir / "catalog.sqlite", tmp_lake / "catalog.sqlite")
        for db in (lake_dir / "domains").glob("*.sqlite"):
            shutil.copy2(db, tmp_lake / "domains" / db.name)

        with DataFoundationRouter(tmp_lake) as router:
            t0 = time.perf_counter()
            exp_w3 = router.export_domain_partition("web3", fmt="parquet")
            t_export = time.perf_counter() - t0

            t1 = time.perf_counter()
            rows = router.query_domain_readonly_duckdb(
                "web3",
                "SELECT venue, COUNT(*) AS n_bars, ROUND(AVG(close), 4) AS avg_close "
                "FROM web3_bars GROUP BY venue ORDER BY venue",
            )
            t_query = time.perf_counter() - t1

            disk_duckdb_files = list(tmp_lake.rglob("*.duckdb*"))
            report["benchmarks"]["web3_parquet_export"] = {
                "format": exp_w3["format"],
                "total_rows": exp_w3["total_rows"],
                "elapsed_sec": round(t_export, 4),
            }
            report["benchmarks"]["duckdb_memory_sql"] = {
                "elapsed_sec": round(t_query, 4),
                "result_rows": rows,
                "on_disk_duckdb_files": len(disk_duckdb_files),
            }

            if (lake_dir / "domains" / "cn_ashare_lake.sqlite").is_file():
                t2 = time.perf_counter()
                exp_a = router.export_domain_partition("cn_ashare", fmt="parquet")
                a_rows = router.query_domain_readonly_duckdb(
                    "cn_ashare",
                    "SELECT COUNT(*) AS n_bars, COUNT(DISTINCT symbol) AS n_syms FROM cn_ashare_bars",
                )
                report["benchmarks"]["cn_ashare_parquet_duckdb"] = {
                    "format": exp_a["format"],
                    "total_rows": exp_a["total_rows"],
                    "query_result": a_rows,
                    "elapsed_sec": round(time.perf_counter() - t2, 4),
                }

    report["status"] = "PASS_RESEARCH_ACCELERATED"
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verify AlphaLoop v4 Research Plane optional accelerators")
    p.add_argument("--lake-dir", type=Path, default=ROOT / "data" / "platform_v4_lake")
    p.add_argument("--require-research", action="store_true", help="Exit non-zero unless numpy/pyarrow/duckdb installed")
    args = p.parse_args(argv)

    rep = benchmark_research_plane(args.lake_dir)
    print(json.dumps(rep, indent=2, sort_keys=True, allow_nan=False))
    if not rep.get("copyleft_clean", False):
        return 2
    if args.require_research and rep.get("mode") != "RESEARCH_ACCELERATED_MODE":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
