#!/usr/bin/env python3
"""Research Lake Immutable Snapshot & SHA-256 Integrity Verifier (`OBS-07` Closure).

Uses SQLite's online `.backup()` API to create crash-consistent, WAL-clean archive snapshots
of `data/platform_v4_lake` (including `catalog.sqlite`, `domains/*.sqlite`, and immutable
`parquet/` partitions) with a deterministic `SNAPSHOT_MANIFEST.json` SHA-256 ledger.

Enables strict separation between frozen official research lake snapshots and daily
diagnostic/CI regressions (`--trial-ledger-mode sandbox|readonly`), preventing routine runs
from inflating official `trial_ledger` counts while preserving REA-001 immutability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaloop.core.db import connect_readonly, connect_wal


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sqlite_online_backup(src_db: Path, dst_db: Path, *, post_query: str | None = None) -> Any:
    """Create a WAL-consistent SQLite copy via SQLite's online backup API (STD-34 safe)."""
    dst_db.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".snap_", suffix=".sqlite", dir=str(dst_db.parent))
    os.close(fd)
    src_conn = connect_readonly(src_db, timeout_sec=30.0)
    dst_conn = connect_wal(Path(tmp_name), timeout_sec=30.0)
    query_res: Any = None
    try:
        src_conn.backup(dst_conn)
        if post_query:
            row = dst_conn.execute(post_query).fetchone()
            query_res = row[0] if row is not None else None
        dst_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        dst_conn.commit()
    except Exception:
        dst_conn.close()
        src_conn.close()
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    else:
        dst_conn.close()
        src_conn.close()
        # Clean sidecar WAL/SHM created on temp file before atomic publish
        for suffix in ("-wal", "-shm"):
            sidecar = Path(tmp_name + suffix)
            if sidecar.exists():
                sidecar.unlink()
        os.chmod(tmp_name, 0o444)
        os.replace(tmp_name, dst_db)
        return query_res


def _assert_not_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing to snapshot symlink: {path}")


def create_lake_snapshot(src_lake: Path, dst_dir: Path, *, label: str = "") -> dict[str, Any]:
    """Create an immutable, SHA-256 verified snapshot of a Platform v4 research lake."""
    src_resolved = src_lake.resolve()
    dst_resolved = dst_dir.resolve()
    cat_src = src_resolved / "catalog.sqlite"
    if not cat_src.exists():
        raise FileNotFoundError(f"Source lake missing catalog.sqlite: {src_resolved}")
    _assert_not_symlink(cat_src)
    if dst_resolved == src_resolved or src_resolved in dst_resolved.parents:
        raise ValueError("Snapshot destination must NOT overlap or sit inside the source lake directory")
    if dst_resolved.exists() and any(dst_resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty snapshot directory: {dst_resolved}")

    dst_resolved.mkdir(parents=True, exist_ok=True)
    files_manifest: dict[str, dict[str, Any]] = {}

    # 1. Online backup of catalog.sqlite + capture official trial_ledger count pre-chmod
    cat_dst = dst_resolved / "catalog.sqlite"
    raw_cnt = _sqlite_online_backup(
        cat_src,
        cat_dst,
        post_query="SELECT COUNT(*) FROM trial_ledger",
    )
    trial_total = int(raw_cnt or 0)

    files_manifest["catalog.sqlite"] = {
        "bytes": cat_dst.stat().st_size,
        "sha256": _sha256_file(cat_dst),
    }

    # 2. Online backup of every domain SQLite database (reject symlinks — RB-D05)
    domains_src = src_resolved / "domains"
    if domains_src.is_dir():
        _assert_not_symlink(domains_src)
        for db_file in sorted(domains_src.glob("*.sqlite")):
            _assert_not_symlink(db_file)
            rel = f"domains/{db_file.name}"
            target = dst_resolved / rel
            _sqlite_online_backup(db_file, target)
            files_manifest[rel] = {
                "bytes": target.stat().st_size,
                "sha256": _sha256_file(target),
            }

    # 3. Copy immutable exported partition files (.jsonl.gz / .parquet), rejecting symlinks (RB-D05)
    parquet_src = src_resolved / "parquet"
    if parquet_src.is_dir():
        _assert_not_symlink(parquet_src)
        for part in sorted(parquet_src.rglob("*")):
            if part.is_symlink():
                raise ValueError(f"Refusing to follow symlink inside research lake parquet/: {part}")
            if part.is_file() and not part.name.startswith("."):
                rel = part.relative_to(src_resolved).as_posix()
                target = dst_resolved / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(part, target)
                os.chmod(target, 0o444)
                files_manifest[rel] = {
                    "bytes": target.stat().st_size,
                    "sha256": _sha256_file(target),
                }

    manifest: dict[str, Any] = {
        "schema_version": "obs07-lake-snapshot-v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": label,
        "source_lake_dir": str(src_resolved),
        "trial_ledger_total_rows": trial_total,
        "file_count": len(files_manifest),
        "files": files_manifest,
    }
    manifest_path = dst_resolved / "SNAPSHOT_MANIFEST.json"
    fd, tmp_m = tempfile.mkstemp(prefix=".manifest_", suffix=".json", dir=str(dst_resolved))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(tmp_m, 0o444)
    os.replace(tmp_m, manifest_path)
    return manifest


def verify_lake_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    """Verify snapshot manifest completeness, SHA-256 checksums, zero unmanifested rogue files & trial count (RB-D05)."""
    snap_resolved = snapshot_dir.resolve()
    manifest_path = snap_resolved / "SNAPSHOT_MANIFEST.json"
    if not manifest_path.exists() or manifest_path.is_symlink():
        return {"ok": False, "error": f"Missing or symlinked SNAPSHOT_MANIFEST.json in {snap_resolved}", "mismatches": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") or {}
    mismatches: list[str] = []
    if not isinstance(files, dict) or "catalog.sqlite" not in files:
        mismatches.append("MISSING_REQUIRED_MANIFEST_ENTRY:catalog.sqlite")
    checked = 0
    expected_rels: set[str] = set()
    if isinstance(files, dict):
        for rel, meta in sorted(files.items()):
            rel_s = str(rel)
            expected_rels.add(rel_s)
            raw_target = snap_resolved / rel_s
            if raw_target.is_symlink():
                mismatches.append(f"SYMLINK_FORBIDDEN:{rel_s}")
                continue
            target = raw_target.resolve()
            if snap_resolved not in target.parents:
                mismatches.append(f"PATH_TRAVERSAL:{rel_s}")
                continue
            if not target.exists():
                mismatches.append(f"MISSING:{rel_s}")
                continue
            actual_sha = _sha256_file(target)
            if not isinstance(meta, dict) or actual_sha != meta.get("sha256"):
                mismatches.append(f"SHA_MISMATCH:{rel_s}")
                continue
            checked += 1

    # RB-D05: fail-closed if ANY unexpected unmanifested file or symlink exists inside snapshot_dir
    for disk_item in sorted(snap_resolved.rglob("*")):
        if disk_item == manifest_path:
            continue
        rel_disk = disk_item.relative_to(snap_resolved).as_posix()
        if disk_item.is_symlink():
            mismatches.append(f"SYMLINK_FORBIDDEN:{rel_disk}")
        elif disk_item.is_file() and rel_disk not in expected_rels:
            mismatches.append(f"UNMANIFESTED_FILE:{rel_disk}")

    # RB-D05: cross-check catalog.sqlite trial_ledger row count against manifest claim
    cat_file = snap_resolved / "catalog.sqlite"
    actual_trial_rows: int | None = None
    if cat_file.is_file() and not cat_file.is_symlink():
        chk_conn = connect_readonly(cat_file, timeout_sec=15.0, immutable=True)
        try:
            row = chk_conn.execute("SELECT COUNT(*) FROM trial_ledger").fetchone()
            actual_trial_rows = int(row[0]) if row is not None else 0
        finally:
            chk_conn.close()
        if manifest.get("trial_ledger_total_rows") != actual_trial_rows:
            mismatches.append(
                f"TRIAL_COUNT_MISMATCH:manifest={manifest.get('trial_ledger_total_rows')}!=db={actual_trial_rows}"
            )

    return {
        "ok": len(mismatches) == 0 and checked == len(expected_rels) and checked > 0,
        "snapshot_dir": str(snap_resolved),
        "files_checked": checked,
        "expected_files": len(expected_rels),
        "trial_ledger_total_rows": actual_trial_rows,
        "mismatches": mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Create or verify immutable Platform v4 research lake snapshots (OBS-07)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("create", help="Create a new SHA-256 verified snapshot")
    pc.add_argument("--src", type=Path, required=True, help="Source research lake directory")
    pc.add_argument("--dst", type=Path, required=True, help="New empty destination snapshot directory")
    pc.add_argument("--label", type=str, default="", help="Optional snapshot label")

    pv = sub.add_parser("verify", help="Verify an existing snapshot directory against SNAPSHOT_MANIFEST.json")
    pv.add_argument("--snapshot", type=Path, required=True, help="Snapshot directory to verify")

    args = p.parse_args(argv)
    if args.cmd == "create":
        res = create_lake_snapshot(args.src, args.dst, label=args.label)
        print(json.dumps(res, indent=2, sort_keys=True, allow_nan=False))
        return 0
    res = verify_lake_snapshot(args.snapshot)
    print(json.dumps(res, indent=2, sort_keys=True, allow_nan=False))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
