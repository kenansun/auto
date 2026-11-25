"""Atomic rename(2) Hot-Reloadable Strategy Config Store with version history & rollback.

Red-Team hardened (#5 & #16):
- Validates strategy_id against `^[A-Za-z0-9_-]{1,64}$` (no path collision or empty filename).
- Enforces strict `^HUMAN_APPROVED:[A-Za-z0-9._-]{4,64}$` token on BOTH `register_version()`
  AND `rollback()` whenever the target status is outside `ALLOWED_AUTO_STATUSES`.
- Append-only `strategy_versions` table (never silently overwrites existing version with different config).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from alphaloop.core.db import connect_wal, get_conn_lock, transaction
from alphaloop.core.models import StrategyConfig
from alphaloop.core.provenance import sha256_obj, utc_now_ms

logger = logging.getLogger(__name__)

STRATEGY_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
HUMAN_TOKEN_RE = re.compile(r"^HUMAN_APPROVED:[A-Za-z0-9._-]{4,64}$")


def validate_strategy_id(strategy_id: str) -> str:
    if not strategy_id or not STRATEGY_ID_RE.fullmatch(strategy_id):
        raise ValueError(f"Invalid strategy_id {strategy_id!r}; must match {STRATEGY_ID_RE.pattern}")
    return strategy_id


def validate_human_token(status: str, allowed_auto: set[str], token: str | None) -> None:
    if status not in allowed_auto:
        if not token or not HUMAN_TOKEN_RE.fullmatch(token):
            raise PermissionError(
                f"Status '{status}' requires valid human_approval_token matching 'HUMAN_APPROVED:<4..64 id chars>'"
            )


def validate_strategy_config_params(cfg: StrategyConfig) -> None:
    """Fail-closed validation of plugin-specific `cfg.params` schema across registry & hot-reload (CC-20261123-001)."""
    pid = str(cfg.plugin_id or "funding_carry").strip().lower()
    if pid == "funding_carry":
        from alphaloop.harness.backtester import validate_funding_carry_params

        validate_funding_carry_params(cfg.params)
    elif pid == "trend_cta":
        from alphaloop.plugins.trend_cta_plugin import validate_trend_cta_params

        validate_trend_cta_params(cfg.params)
    elif cfg.params is not None and not isinstance(cfg.params, dict):
        raise ValueError(f"StrategyConfig.params must be a dict, got {type(cfg.params).__name__}")


class HotConfigStore:
    """Versioned strategy registry + atomic POSIX rename(2) runtime config bus."""

    ALLOWED_AUTO_STATUSES = {
        "DRAFT",
        "BACKTESTED",
        "RED_TEAMED",
        "MICRO_RUN_PROBE_REQUIRED",
        "SHADOW_LIVE",
        "RETIRED",
    }

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
        """Persist a new strategy config version (append-only) and atomically publish active runtime file."""
        validate_strategy_id(cfg.strategy_id)
        validate_human_token(cfg.status, self.ALLOWED_AUTO_STATUSES, human_approval_token)
        validate_strategy_config_params(cfg)

        payload = cfg.to_dict()
        cfg_sha = sha256_obj(payload)
        now = utc_now_ms()

        # Hold connection lock across BOTH DB transaction and runtime os.replace() to preserve strict ordering (HIGH-R1)
        with get_conn_lock(self.conn):
            with transaction(self.conn):
                existing = self.conn.fetch_one(
                    "SELECT config_sha256 FROM strategy_versions WHERE strategy_id=? AND version=?",
                    (cfg.strategy_id, cfg.version),
                )
                if existing is not None:
                    if existing["config_sha256"] != cfg_sha:
                        raise ValueError(
                            f"Append-only violation: {cfg.strategy_id}@{cfg.version} already exists with different SHA"
                        )
                else:
                    self.conn.execute(
                        """
                        INSERT INTO strategy_versions
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
            self._atomic_publish_runtime(cfg, cfg_sha, now)
        logger.info("Published hot config %s@%s [%s] sha=%s", cfg.strategy_id, cfg.version, cfg.status, cfg_sha[:12])
        return cfg_sha

    def _atomic_publish_runtime(self, cfg: StrategyConfig, cfg_sha: str, ts_ms: int) -> Path:
        """Write to temp file in same directory + fsync + POSIX os.replace() (atomic rename(2))."""
        safe_id = validate_strategy_id(cfg.strategy_id)
        target = self.runtime_dir / f"{safe_id}.json"
        envelope = {
            "published_ts_ms": ts_ms,
            "config_sha256": cfg_sha,
            "config": cfg.to_dict(),
        }
        raw = json.dumps(envelope, indent=2, sort_keys=True).encode("utf-8")
        fd, tmp_name = tempfile.mkstemp(prefix=f".{safe_id}.", suffix=".tmp", dir=str(self.runtime_dir))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
        return target

    def read_active(self, strategy_id: str) -> StrategyConfig | None:
        safe_id = validate_strategy_id(strategy_id)
        target = self.runtime_dir / f"{safe_id}.json"
        if not target.exists():
            return None
        data = json.loads(target.read_text(encoding="utf-8"))
        return StrategyConfig.from_dict(data["config"])

    def rollback(
        self,
        strategy_id: str,
        target_version: str,
        actor: str,
        reason: str,
        human_approval_token: str | None = None,
    ) -> StrategyConfig:
        """Atomically roll back active runtime config (enforces human token if target is LIVE/CANARY)."""
        validate_strategy_id(strategy_id)
        with get_conn_lock(self.conn):
            row = self.conn.fetch_one(
                "SELECT status, config_sha256, config_json FROM strategy_versions WHERE strategy_id=? AND version=?",
                (strategy_id, target_version),
            )
            if row is None:
                raise ValueError(f"Version {strategy_id}@{target_version} not found")
            cfg = StrategyConfig.from_dict(json.loads(row["config_json"]))
            validate_human_token(cfg.status, self.ALLOWED_AUTO_STATUSES, human_approval_token)
            validate_strategy_config_params(cfg)

            now = utc_now_ms()
            with transaction(self.conn):
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO active_pointers
                    (strategy_id, active_version, status, updated_ts_ms)
                    VALUES (?, ?, ?, ?)
                    """,
                    (strategy_id, target_version, row["status"], now),
                )
            self._atomic_publish_runtime(cfg, row["config_sha256"], now)
        logger.warning("Rolled back %s -> %s by %s (%s)", strategy_id, target_version, actor, reason)
        return cfg

    def list_strategies(self) -> list[dict[str, Any]]:
        rows = self.conn.fetch_all(
            """
            SELECT p.strategy_id, p.active_version, p.status, p.updated_ts_ms,
                   v.config_sha256, v.actor, v.reason, v.config_json
            FROM active_pointers p
            JOIN strategy_versions v
              ON p.strategy_id = v.strategy_id AND p.active_version = v.version
            ORDER BY p.updated_ts_ms DESC
            """
        )
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

    def version_history(self, strategy_id: str) -> list[dict[str, Any]]:
        rows = self.conn.fetch_all(
            """
            SELECT version, status, config_sha256, actor, reason, created_ts_ms
            FROM strategy_versions WHERE strategy_id=? ORDER BY id DESC
            """,
            (strategy_id,),
        )
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
