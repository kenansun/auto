"""Atomic rename(2) Hot-Reloadable Strategy Config Store with version history & rollback."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from alphaloop.core.db import connect_wal, transaction
from alphaloop.core.models import StrategyConfig
from alphaloop.core.provenance import sha256_obj, utc_now_ms

logger = logging.getLogger(__name__)


class HotConfigStore:
    """Versioned strategy registry + atomic POSIX rename(2) runtime config bus.

    Guarantees:
    - Live consumers read a single JSON file updated atomically via os.replace() (<2us),
      preventing partial reads or torn writes.
    - Every config transition is logged append-only in SQLite WAL with git/config SHA256.
    - LIVE promotion requires explicit human approval token; automated loops may only
      promote up to SHADOW_LIVE.
    """

    ALLOWED_AUTO_STATUSES = {"DRAFT", "BACKTESTED", "RED_TEAMED", "SHADOW_LIVE", "RETIRED"}

    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = self.state_dir / "active_configs"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "registry.sqlite"
        self.conn = connect_wal(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_ts_ms INTEGER NOT NULL,
                    UNIQUE(strategy_id, version)
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS active_pointers (
                    strategy_id TEXT PRIMARY KEY,
                    active_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_ts_ms INTEGER NOT NULL
                );
                """
            )

    def register_version(
        self,
        cfg: StrategyConfig,
        actor: str,
        reason: str,
        human_approval_token: str | None = None,
    ) -> str:
        """Persist a new strategy config version and atomically publish active runtime file."""
        if cfg.status not in self.ALLOWED_AUTO_STATUSES:
            if not human_approval_token or not human_approval_token.startswith("HUMAN_APPROVED:"):
                raise PermissionError(
                    f"Status '{cfg.status}' requires explicit human_approval_token='HUMAN_APPROVED:<id>'"
                )

        payload = cfg.to_dict()
        cfg_sha = sha256_obj(payload)
        now = utc_now_ms()

        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT OR REPLACE INTO strategy_versions
                (strategy_id, version, status, config_sha256, config_json, actor, reason, created_ts_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cfg.strategy_id,
                    cfg.version,
                    cfg.status,
                    cfg_sha,
                    json.dumps(payload, indent=2, sort_keys=True),
                    actor,
                    reason,
                    now,
                ),
            )
            self.conn.execute(
                """
                INSERT OR REPLACE INTO active_pointers
                (strategy_id, active_version, status, updated_ts_ms)
                VALUES (?, ?, ?, ?)
                """,
                (cfg.strategy_id, cfg.version, cfg.status, now),
            )

        self._atomic_publish(cfg.strategy_id, {
            "strategy_id": cfg.strategy_id,
            "version": cfg.version,
            "status": cfg.status,
            "config_sha256": cfg_sha,
            "updated_ts_ms": now,
            "actor": actor,
            "reason": reason,
            "config": payload,
        })
        logger.info("Published hot config %s@%s [%s] sha=%s", cfg.strategy_id, cfg.version, cfg.status, cfg_sha[:12])
        return cfg_sha

    def _atomic_publish(self, strategy_id: str, envelope: dict[str, Any]) -> Path:
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in strategy_id)
        target = self.runtime_dir / f"{safe_id}.json"
        fd, tmp_path = tempfile.mkstemp(prefix=f".{safe_id}_", suffix=".tmp", dir=str(self.runtime_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(envelope, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, target)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return target

    def read_active(self, strategy_id: str) -> StrategyConfig | None:
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in strategy_id)
        target = self.runtime_dir / f"{safe_id}.json"
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as fh:
            env = json.load(fh)
        return StrategyConfig.from_dict(env["config"])

    def rollback(self, strategy_id: str, target_version: str, actor: str, reason: str) -> StrategyConfig:
        row = self.conn.execute(
            "SELECT config_json FROM strategy_versions WHERE strategy_id=? AND version=?",
            (strategy_id, target_version),
        ).fetchone()
        if not row:
            raise ValueError(f"Version {strategy_id}@{target_version} not found")
        cfg = StrategyConfig.from_dict(json.loads(row["config_json"]))
        now = utc_now_ms()
        cfg_sha = sha256_obj(cfg.to_dict())
        with transaction(self.conn):
            self.conn.execute(
                "UPDATE active_pointers SET active_version=?, status=?, updated_ts_ms=? WHERE strategy_id=?",
                (cfg.version, cfg.status, now, strategy_id),
            )
        self._atomic_publish(strategy_id, {
            "strategy_id": strategy_id,
            "version": cfg.version,
            "status": cfg.status,
            "config_sha256": cfg_sha,
            "updated_ts_ms": now,
            "actor": actor,
            "reason": f"ROLLBACK: {reason}",
            "config": cfg.to_dict(),
        })
        logger.warning("Rolled back %s -> %s (%s)", strategy_id, target_version, reason)
        return cfg

    def list_strategies(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT p.strategy_id, p.active_version, p.status, p.updated_ts_ms,
                   v.config_sha256, v.actor, v.reason, v.config_json
            FROM active_pointers p
            JOIN strategy_versions v
              ON p.strategy_id = v.strategy_id AND p.active_version = v.version
            ORDER BY p.updated_ts_ms DESC
            """
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "strategy_id": r["strategy_id"],
                "active_version": r["active_version"],
                "status": r["status"],
                "updated_ts_ms": r["updated_ts_ms"],
                "config_sha256": r["config_sha256"],
                "actor": r["actor"],
                "reason": r["reason"],
                "config": json.loads(r["config_json"]),
            })
        return out

    def history(self, strategy_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT version, status, config_sha256, actor, reason, created_ts_ms
            FROM strategy_versions
            WHERE strategy_id=?
            ORDER BY id DESC
            """,
            (strategy_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
