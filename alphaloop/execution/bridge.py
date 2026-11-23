"""Shadow/Paper-Only Execution Bridge with Hash-Chained Audit Log & Persistent Kill Switch (Module D).

Strictly DRY-RUN / PAPER-ONLY (`LIVE_EXECUTION_ENABLED: Final[bool] = False`):
  - Zero private keys, zero live exchange order submission, zero real-money movement.
  - Translates plugin `OrderIntent` + live `BBOTick` + hot-reloaded `StrategyConfig` into
    deterministic `ExecutionPlan` objects aligned with institutional DEX-CEX execution contracts.
  - Enforces fail-closed pre-trade risk guards (status gate, strategy_id match, finite config/intent
    bounds, persistent kill switch, stale/crossed/mismatched BBO, spread ceiling, notional cap)
    and emits explicit `PLAN_BLOCKED` audit rows.
  - Deterministic canonical-JSON `idempotency_key` (NO wall-clock timestamp!) with `UNIQUE(idempotency_key)`.
  - Atomic `BEGIN IMMEDIATE` `LockedConnection` transactions protecting `ExecutionKillSwitch`,
    `ExecutionAuditChain` (`seq`, `ts_ms`, `idempotency_key`, `prev_hash`, `record_hash`), and `execution_plans`.
  - `HotConfigWatcher` validates symlinks, size <=1MB, filename==strategy_id, envelope SHA-256,
    evicts deleted/corrupt configs, and auto-trips `CONFIG_CORRUPT` kill switch on malformed JSON.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

from alphaloop.core.config import validate_strategy_id
from alphaloop.core.db import LockedConnection, connect_wal, transaction
from alphaloop.core.models import BBOTick, OrderIntent, StrategyConfig
from alphaloop.core.provenance import sha256_obj, utc_now_ms

logger = logging.getLogger(__name__)

LIVE_EXECUTION_ENABLED: Final[bool] = False
GENESIS_PREV_HASH: Final[str] = "0" * 64
ALLOWED_SHADOW_STATUSES: Final[frozenset[str]] = frozenset({"SHADOW_LIVE", "PAPER_LIVE"})
HUMAN_RESET_RE = re.compile(r"^HUMAN_RESET:[A-Za-z0-9._-]{4,64}$")
MAX_CONFIG_BYTES: Final[int] = 1024 * 1024  # 1MB hard cap


class KillTrigger(str, Enum):
    """Canonical kill-switch triggers aligned with institutional execution safety contracts."""

    OPERATOR = "OPERATOR"
    RISK_DRAWDOWN = "RISK_DRAWDOWN"
    RISK_SINGLE_LEG_LIMIT = "RISK_SINGLE_LEG_LIMIT"
    SLIPPAGE_EXCEEDED = "SLIPPAGE_EXCEEDED"
    STALE_FEED = "STALE_FEED"
    CONFIG_CORRUPT = "CONFIG_CORRUPT"


@dataclass(frozen=True)
class LegSpec:
    """Single leg instruction inside a shadow ExecutionPlan."""

    leg_id: str          # LEG_A | LEG_B
    venue: str
    symbol: str
    side: str            # BUY | SELL | CLOSE
    notional_usd: float
    ref_price: float
    limit_price: float
    spread_bps: float
    bbo_ts_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    """Deterministic shadow execution plan derived from a plugin OrderIntent."""

    plan_id: str
    idempotency_key: str
    run_mode: str         # Always "SHADOW_DRY_RUN"
    status: str           # PLANNED_SHADOW | BLOCKED
    block_reason: str
    ts_ms: int
    strategy_id: str
    version: str
    config_sha256: str
    symbol: str
    action: str
    notional_usd: float
    expected_edge_bps: float
    legs: list[LegSpec] = field(default_factory=list)
    audit_seq: int = 0
    duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["legs"] = [leg.to_dict() if isinstance(leg, LegSpec) else leg for leg in self.legs]
        return d


def _is_finite_num(val: Any) -> bool:
    return (not isinstance(val, bool)) and isinstance(val, (int, float)) and math.isfinite(float(val))


def _canonical_json(obj: Any) -> str:
    """RFC-8259 canonical JSON rejecting NaN/Infinity fail-closed."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _compute_record_hash(
    seq: int,
    ts_ms: int,
    prev_hash: str,
    event_type: str,
    plan_id: str,
    idempotency_key: str,
    canonical_payload: str,
) -> str:
    """Collision-free canonical JSON array hash covering ALL audit columns including ts_ms & idempotency_key."""
    frame = _canonical_json([seq, ts_ms, prev_hash, event_type, plan_id, idempotency_key, canonical_payload])
    return hashlib.sha256(frame.encode("utf-8")).hexdigest()


class ExecutionAuditChain:
    """Append-only SHA-256 hash-chained audit ledger persisted in SQLite WAL."""

    def __init__(self, conn: LockedConnection) -> None:
        self.conn = conn
        self._init_schema()

    def _init_schema(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_audit_chain (
                    seq INTEGER PRIMARY KEY,
                    ts_ms INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    record_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def next_seq_and_prev_in_tx(self) -> tuple[int, str]:
        """Read next sequence number and previous record_hash INSIDE an active transaction."""
        row = self.conn.execute(
            "SELECT seq, record_hash FROM execution_audit_chain ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return 1, GENESIS_PREV_HASH
        return int(row["seq"]) + 1, str(row["record_hash"])

    def append_in_tx(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        plan_id: str = "",
        idempotency_key: str = "",
        ts_ms: int | None = None,
    ) -> tuple[int, str]:
        """Append one hash-chained audit row INSIDE an active BEGIN IMMEDIATE transaction."""
        now = int(ts_ms) if ts_ms is not None else utc_now_ms()
        next_seq, prev_hash = self.next_seq_and_prev_in_tx()
        cjson = _canonical_json(payload)
        rec_hash = _compute_record_hash(next_seq, now, prev_hash, event_type, plan_id, idempotency_key, cjson)
        self.conn.execute(
            """
            INSERT INTO execution_audit_chain
            (seq, ts_ms, event_type, plan_id, idempotency_key, prev_hash, record_hash, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (next_seq, now, event_type, plan_id, idempotency_key, prev_hash, rec_hash, cjson),
        )
        return next_seq, rec_hash

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        plan_id: str = "",
        idempotency_key: str = "",
        ts_ms: int | None = None,
    ) -> tuple[int, str]:
        with transaction(self.conn):
            return self.append_in_tx(
                event_type,
                payload,
                plan_id=plan_id,
                idempotency_key=idempotency_key,
                ts_ms=ts_ms,
            )

    def head(self) -> tuple[int, str]:
        row = self.conn.fetch_one(
            "SELECT seq, record_hash FROM execution_audit_chain ORDER BY seq DESC LIMIT 1"
        )
        if row is None:
            return 0, GENESIS_PREV_HASH
        return int(row["seq"]), str(row["record_hash"])

    def verify_chain(self) -> tuple[bool, str, int]:
        """Verify contiguous seq 1..N, genesis prev_hash, ts_ms, idempotency_key, hash linking, and payload integrity."""
        rows = self.conn.fetch_all(
            "SELECT seq, ts_ms, event_type, plan_id, idempotency_key, prev_hash, record_hash, payload_json "
            "FROM execution_audit_chain ORDER BY seq ASC"
        )
        expected_seq = 1
        expected_prev = GENESIS_PREV_HASH
        for r in rows:
            seq = int(r["seq"])
            if seq != expected_seq:
                return False, f"NON_CONTIGUOUS_SEQ(expected={expected_seq}, got={seq})", len(rows)
            prev_h = str(r["prev_hash"])
            if prev_h != expected_prev:
                return False, f"BROKEN_PREV_HASH_AT_SEQ_{seq}", len(rows)
            recomp = _compute_record_hash(
                seq,
                int(r["ts_ms"]),
                prev_h,
                str(r["event_type"]),
                str(r["plan_id"]),
                str(r["idempotency_key"]),
                str(r["payload_json"]),
            )
            if recomp != str(r["record_hash"]):
                return False, f"TAMPERED_RECORD_HASH_AT_SEQ_{seq}", len(rows)
            expected_prev = recomp
            expected_seq += 1
        return True, "OK", len(rows)

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.fetch_all(
            """
            SELECT seq, ts_ms, event_type, plan_id, idempotency_key, prev_hash, record_hash, payload_json
            FROM execution_audit_chain ORDER BY seq DESC LIMIT ?
            """,
            (limit,),
        )
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d.pop("payload_json"))
            except json.JSONDecodeError:
                d["payload"] = {}
            out.append(d)
        return out


class ExecutionKillSwitch:
    """Persistent fail-closed SQLite kill latch linked atomically to ExecutionAuditChain."""

    def __init__(self, conn: LockedConnection, audit: ExecutionAuditChain) -> None:
        self.conn = conn
        self.audit = audit
        self._init_schema()

    def _init_schema(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kill_switch (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    engaged INTEGER NOT NULL CHECK (engaged IN (0, 1)),
                    trigger_name TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    engaged_ts_ms INTEGER NOT NULL,
                    cleared_ts_ms INTEGER NOT NULL DEFAULT 0,
                    clear_actor TEXT NOT NULL DEFAULT '',
                    clear_reason TEXT NOT NULL DEFAULT '',
                    engage_seq INTEGER NOT NULL DEFAULT 0
                );
                """
            )

    def is_engaged_in_tx(self) -> bool:
        """Read kill latch inside an already-locked transaction (fail-closed)."""
        row = self.conn.execute("SELECT engaged FROM kill_switch WHERE id = 1").fetchone()
        return bool(row and int(row["engaged"]) == 1)

    def is_engaged(self) -> bool:
        try:
            row = self.conn.fetch_one("SELECT engaged FROM kill_switch WHERE id = 1")
            return bool(row and int(row["engaged"]) == 1)
        except Exception as exc:
            logger.error("KillSwitch read failed (fail-closed -> ENGAGED): %s", exc)
            return True

    def state(self) -> dict[str, Any]:
        row = self.conn.fetch_one("SELECT * FROM kill_switch WHERE id = 1")
        if row is None:
            return {"engaged": False, "engage_seq": 0, "trigger": "", "actor": "", "detail": ""}
        d = dict(row)
        return {
            "engaged": bool(d["engaged"]),
            "engage_seq": int(d["engage_seq"]),
            "trigger": str(d["trigger_name"]) if d["engaged"] else "",
            "actor": str(d["actor"]) if d["engaged"] else "",
            "detail": str(d["detail"]) if d["engaged"] else "",
            "engaged_ts_ms": int(d["engaged_ts_ms"]),
            "cleared_ts_ms": int(d["cleared_ts_ms"]),
            "clear_actor": str(d["clear_actor"]),
            "clear_reason": str(d["clear_reason"]),
        }

    def engage(self, trigger: KillTrigger, actor: str, detail: str) -> int:
        """Engage kill latch and record KILL_ENGAGED in one atomic transaction."""
        now = utc_now_ms()
        with transaction(self.conn):
            row = self.conn.execute(
                "SELECT engaged, trigger_name, actor, detail, engaged_ts_ms, engage_seq FROM kill_switch WHERE id = 1"
            ).fetchone()
            next_engage_seq = (int(row["engage_seq"]) + 1) if row else 1
            if row and int(row["engaged"]) == 1:
                self.conn.execute(
                    "UPDATE kill_switch SET engage_seq=? WHERE id = 1",
                    (next_engage_seq,),
                )
            else:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO kill_switch
                    (id, engaged, trigger_name, actor, detail, engaged_ts_ms, cleared_ts_ms, clear_actor, clear_reason, engage_seq)
                    VALUES (1, 1, ?, ?, ?, ?, 0, '', '', ?)
                    """,
                    (trigger.value, actor, detail, now, next_engage_seq),
                )
            self.audit.append_in_tx(
                "KILL_ENGAGED",
                {
                    "trigger": trigger.value,
                    "actor": actor,
                    "detail": detail,
                    "engage_seq": next_engage_seq,
                },
                ts_ms=now,
            )
            return next_engage_seq

    def clear(self, actor: str, human_reset_token: str, reason: str) -> int:
        """Clear an engaged kill switch; strictly requires HUMAN_RESET:<id> token (fullmatch) and non-empty reason."""
        if not HUMAN_RESET_RE.fullmatch(human_reset_token or ""):
            raise PermissionError(
                "Clearing ExecutionKillSwitch requires explicit human approval token matching 'HUMAN_RESET:<id>'."
            )
        if not (reason or "").strip():
            raise ValueError("Non-empty reason is required to clear ExecutionKillSwitch.")
        now = utc_now_ms()
        with transaction(self.conn):
            row = self.conn.execute(
                "SELECT engaged, engage_seq FROM kill_switch WHERE id = 1"
            ).fetchone()
            if not row or int(row["engaged"]) != 1:
                raise ValueError("ExecutionKillSwitch is not currently engaged.")
            eseq = int(row["engage_seq"])
            self.conn.execute(
                """
                UPDATE kill_switch
                SET engaged=0, cleared_ts_ms=?, clear_actor=?, clear_reason=?
                WHERE id=1 AND engage_seq=?
                """,
                (now, actor, f"{human_reset_token}: {reason.strip()}", eseq),
            )
            self.audit.append_in_tx(
                "KILL_CLEARED",
                {
                    "actor": actor,
                    "human_reset_token": human_reset_token,
                    "reason": reason.strip(),
                    "engage_seq": eseq,
                },
                ts_ms=now,
            )
            return eseq


def _validate_cfg_risk_fields(cfg: StrategyConfig) -> str:
    """Validate finite, non-bool risk fields on StrategyConfig; return empty string if valid."""
    if not cfg.strategy_id or not cfg.symbols:
        return "INVALID_STRATEGY_OR_SYMBOLS"
    try:
        validate_strategy_id(cfg.strategy_id)
    except ValueError as exc:
        return f"INVALID_STRATEGY_ID({exc})"
    if not _is_finite_num(cfg.capital_usd) or float(cfg.capital_usd) <= 0.0:
        return f"INVALID_CFG_CAPITAL({cfg.capital_usd})"
    if not _is_finite_num(cfg.leverage) or float(cfg.leverage) <= 0.0:
        return f"INVALID_CFG_LEVERAGE({cfg.leverage})"
    if not _is_finite_num(cfg.per_position_weight) or not (0.0 < float(cfg.per_position_weight) <= 1.0):
        return f"INVALID_CFG_WEIGHT({cfg.per_position_weight})"
    if isinstance(cfg.max_positions, bool) or not isinstance(cfg.max_positions, int) or cfg.max_positions < 1:
        return f"INVALID_CFG_MAX_POSITIONS({cfg.max_positions})"
    if not _is_finite_num(cfg.slippage_per_leg_bps) or float(cfg.slippage_per_leg_bps) < 0.0:
        return f"INVALID_CFG_SLIPPAGE({cfg.slippage_per_leg_bps})"
    return ""


class HotConfigWatcher:
    """Thread-safe watcher for atomic active_configs/*.json failing closed on symlinks/oversize/SHA mismatch/corruption."""

    def __init__(
        self,
        active_dir: str | Path,
        audit: ExecutionAuditChain,
        kill_switch: ExecutionKillSwitch,
    ) -> None:
        self.active_dir = Path(active_dir)
        self.audit = audit
        self.kill_switch = kill_switch
        self._lock = threading.RLock()
        self._fingerprints: dict[str, tuple[int, str]] = {}  # filename -> (mtime_ns, sha256)
        self._corrupt_seen: dict[str, tuple[int, int]] = {}  # filename -> (mtime_ns, st_size)
        self._file_to_sid: dict[str, str] = {}               # filename -> strategy_id
        self._loaded: dict[str, tuple[StrategyConfig, str]] = {}  # strategy_id -> (cfg, config_sha)

    def _evict_file_locked(self, fname: str) -> str | None:
        self._fingerprints.pop(fname, None)
        old_sid = self._file_to_sid.pop(fname, None)
        if old_sid is not None:
            self._loaded.pop(old_sid, None)
        return old_sid

    def poll(self) -> list[dict[str, Any]]:
        """Scan active_configs/*.json for new/modified/deleted configs; trips kill switch on any corruption."""
        events: list[dict[str, Any]] = []
        with self._lock:
            if not self.active_dir.exists():
                self._corrupt_seen.clear()
                for fname in list(self._fingerprints.keys()):
                    old_sid = self._evict_file_locked(fname)
                    self.audit.append("CONFIG_REMOVED", {"file": fname, "strategy_id": old_sid or ""})
                    events.append({"file": fname, "strategy_id": old_sid or "", "status": "REMOVED"})
                return events

            current_paths = sorted(self.active_dir.glob("*.json"))
            current_names = {p.name for p in current_paths}

            # Evict deleted files so stale configs never linger in memory
            for fname in sorted(set(self._corrupt_seen.keys()) - current_names):
                self._corrupt_seen.pop(fname, None)
            for fname in sorted(set(self._fingerprints.keys()) - current_names):
                old_sid = self._evict_file_locked(fname)
                self.audit.append("CONFIG_REMOVED", {"file": fname, "strategy_id": old_sid or ""})
                events.append({"file": fname, "strategy_id": old_sid or "", "status": "REMOVED"})

            for path in current_paths:
                # Reject symlinks and oversized files BEFORE reading bytes
                sig: tuple[int, int] | None = None
                try:
                    st = path.lstat()
                    sig = (st.st_mtime_ns, st.st_size)
                    if path.is_symlink():
                        raise ValueError("Symlink refused in active_configs")
                    if st.st_size > MAX_CONFIG_BYTES:
                        raise ValueError(f"Config exceeds {MAX_CONFIG_BYTES}B limit ({st.st_size}B)")
                    if self._corrupt_seen.get(path.name) == sig:
                        continue
                    raw = path.read_bytes()
                except Exception as exc:
                    self._evict_file_locked(path.name)
                    if sig is not None and self._corrupt_seen.get(path.name) == sig:
                        continue
                    if sig is not None:
                        self._corrupt_seen[path.name] = sig
                    self.kill_switch.engage(
                        KillTrigger.CONFIG_CORRUPT,
                        actor="hot_config_watcher",
                        detail=f"Rejected active config {path.name}: {exc}",
                    )
                    self.audit.append("CONFIG_CORRUPT_BLOCKED", {"file": path.name, "error": str(exc)})
                    events.append({"file": path.name, "status": "CORRUPT_KILL_ENGAGED", "error": str(exc)})
                    continue

                fsha = hashlib.sha256(raw).hexdigest()
                prev = self._fingerprints.get(path.name)
                if prev == (st.st_mtime_ns, fsha):
                    continue

                try:
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
                        raise ValueError("Missing 'config' object in active config payload")
                    cfg = StrategyConfig.from_dict(payload["config"])
                    err = _validate_cfg_risk_fields(cfg)
                    if err:
                        raise ValueError(err)
                    if path.stem != cfg.strategy_id:
                        raise ValueError(f"Filename stem {path.stem!r} != strategy_id {cfg.strategy_id!r}")
                    computed_sha = sha256_obj(cfg.to_dict())
                    declared_sha = payload.get("config_sha256")
                    if declared_sha is not None and str(declared_sha) != computed_sha:
                        raise ValueError(
                            f"config_sha256 mismatch (declared={str(declared_sha)[:12]}, computed={computed_sha[:12]})"
                        )
                    cfg_sha = computed_sha
                except Exception as exc:
                    self._evict_file_locked(path.name)
                    if sig is not None:
                        self._corrupt_seen[path.name] = sig
                    self.kill_switch.engage(
                        KillTrigger.CONFIG_CORRUPT,
                        actor="hot_config_watcher",
                        detail=f"Corrupted active config {path.name}: {exc}",
                    )
                    self.audit.append(
                        "CONFIG_CORRUPT_BLOCKED",
                        {"file": path.name, "file_sha256": fsha, "error": str(exc)},
                    )
                    events.append({"file": path.name, "status": "CORRUPT_KILL_ENGAGED", "error": str(exc)})
                    continue

                self._corrupt_seen.pop(path.name, None)
                self._fingerprints[path.name] = (st.st_mtime_ns, fsha)
                self._file_to_sid[path.name] = cfg.strategy_id
                self._loaded[cfg.strategy_id] = (cfg, cfg_sha)
                self.audit.append(
                    "CONFIG_RELOADED",
                    {
                        "file": path.name,
                        "strategy_id": cfg.strategy_id,
                        "version": cfg.version,
                        "status": cfg.status,
                        "config_sha256": cfg_sha,
                    },
                )
                events.append({
                    "file": path.name,
                    "strategy_id": cfg.strategy_id,
                    "version": cfg.version,
                    "status": "RELOADED",
                    "config_sha256": cfg_sha,
                })
        return events

    def get_loaded(self, strategy_id: str) -> tuple[StrategyConfig, str] | None:
        with self._lock:
            return self._loaded.get(strategy_id)

    def all_loaded(self) -> dict[str, StrategyConfig]:
        with self._lock:
            return {sid: cfg for sid, (cfg, _sha) in self._loaded.items()}


class ExecutionBridge:
    """Strictly SHADOW_DRY_RUN execution bridge translating OrderIntent -> audited ExecutionPlan."""

    def __init__(
        self,
        state_dir: str | Path,
        *,
        max_notional_cap_usd: float = 5000.0,
        max_bbo_age_ms: int = 30_000,
        intent_bucket_ms: int = 60_000,
        live_mode: bool = False,
    ) -> None:
        if live_mode or LIVE_EXECUTION_ENABLED:
            raise RuntimeError("LIVE_EXECUTION_DISABLED: ExecutionBridge is strictly SHADOW_DRY_RUN only.")
        if not _is_finite_num(max_notional_cap_usd) or float(max_notional_cap_usd) <= 0.0:
            raise ValueError("max_notional_cap_usd must be finite and > 0")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.max_notional_cap_usd = max(10.0, float(max_notional_cap_usd))
        self.max_bbo_age_ms = max(1000, int(max_bbo_age_ms))
        self.intent_bucket_ms = max(1000, int(intent_bucket_ms))

        self.conn = connect_wal(self.state_dir / "execution_bridge.sqlite")
        self.audit = ExecutionAuditChain(self.conn)
        self.kill_switch = ExecutionKillSwitch(self.conn, self.audit)
        self.watcher = HotConfigWatcher(self.state_dir / "active_configs", self.audit, self.kill_switch)
        self._init_schema()

    def _init_schema(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_plans (
                    plan_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    strategy_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    notional_usd REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_exec_plans_ts
                ON execution_plans (ts_ms DESC);
                """
            )

    def _idempotency_key(self, cfg_strategy_id: str, intent: OrderIntent, config_sha: str, bucket_ts: int) -> str:
        """Collision-free deterministic idempotency key (canonical JSON array; never wall-clock)."""
        if not _is_finite_num(intent.notional_usd):
            notional_str = "INVALID_NOTIONAL"
        else:
            notional_str = f"{round(float(intent.notional_usd), 2):.2f}"
        frame = _canonical_json([
            cfg_strategy_id,
            str(intent.strategy_id),
            config_sha,
            bucket_ts,
            str(intent.symbol),
            str(intent.action),
            notional_str,
        ])
        return hashlib.sha256(frame.encode("utf-8")).hexdigest()

    def submit_intent(
        self,
        cfg: StrategyConfig,
        intent: OrderIntent,
        bbo_by_venue: dict[str, BBOTick],
        *,
        now_ms: int | None = None,
        max_spread_bps: float | None = None,
    ) -> ExecutionPlan:
        """Validate intent against fail-closed shadow gates and persist plan + hash-chained audit atomically."""
        if LIVE_EXECUTION_ENABLED:
            raise RuntimeError("LIVE_EXECUTION_DISABLED")

        eval_now = int(now_ms) if (now_ms is not None and _is_finite_num(now_ms)) else utc_now_ms()
        # Hash config safely even if caller injected NaN fields into cfg (repr fallback avoids crash before fail-closed)
        try:
            config_sha = sha256_obj(cfg.to_dict())
        except (TypeError, ValueError):
            config_sha = hashlib.sha256(repr(cfg).encode("utf-8")).hexdigest()

        ts_valid = _is_finite_num(intent.ts_ms) and int(intent.ts_ms) > 0
        bucket_ts = (int(intent.ts_ms) // self.intent_bucket_ms) * self.intent_bucket_ms if ts_valid else 0
        idem_key = self._idempotency_key(str(cfg.strategy_id), intent, config_sha, bucket_ts)
        plan_id = f"SHD-{idem_key[:16]}"

        notional_valid = _is_finite_num(intent.notional_usd) and float(intent.notional_usd) > 0.0
        safe_notional = round(float(intent.notional_usd), 2) if notional_valid else 0.0
        safe_edge = round(float(intent.expected_edge_bps), 3) if _is_finite_num(intent.expected_edge_bps) else 0.0

        cfg_err = _validate_cfg_risk_fields(cfg)
        spread_ceiling_valid = (
            True if max_spread_bps is None else (_is_finite_num(max_spread_bps) and float(max_spread_bps) > 0.0)
        )

        block_reason = ""
        legs: list[LegSpec] = []

        if cfg_err:
            block_reason = cfg_err
        elif not ts_valid:
            block_reason = f"INVALID_INTENT_TS({intent.ts_ms})"
        elif str(intent.strategy_id) != str(cfg.strategy_id):
            block_reason = f"STRATEGY_ID_MISMATCH(intent={intent.strategy_id},cfg={cfg.strategy_id})"
        elif cfg.status not in ALLOWED_SHADOW_STATUSES:
            block_reason = f"STATUS_NOT_SHADOW_ELIGIBLE({cfg.status})"
        elif not spread_ceiling_valid:
            block_reason = f"INVALID_MAX_SPREAD_BPS({max_spread_bps})"
        elif not notional_valid:
            block_reason = f"INVALID_NOTIONAL({intent.notional_usd})"
        elif intent.symbol not in cfg.symbols:
            block_reason = f"SYMBOL_NOT_IN_CONFIG({intent.symbol})"
        else:
            slot_cap = (
                float(cfg.capital_usd)
                * min(float(cfg.per_position_weight), 1.0 / max(1, int(cfg.max_positions)))
                * max(1.0, float(cfg.leverage))
            )
            notional_limit = min(slot_cap, self.max_notional_cap_usd)
            spread_ceiling = (
                float(max_spread_bps)
                if max_spread_bps is not None
                else max(25.0, 4.0 * max(1.0, float(cfg.slippage_per_leg_bps)))
            )
            if not math.isfinite(notional_limit) or float(intent.notional_usd) > notional_limit + 1e-6:
                block_reason = f"NOTIONAL_CAP_EXCEEDED({float(intent.notional_usd):.2f}>{notional_limit:.2f})"
            else:
                is_single_venue = (
                    (cfg.mode == "single_venue_perp")
                    or (cfg.venue_a == cfg.venue_b)
                    or intent.action in ("OPEN_LONG", "OPEN_SHORT")
                )
                if is_single_venue:
                    side_a = "BUY" if intent.action == "OPEN_LONG" else ("SELL" if intent.action == "OPEN_SHORT" else "CLOSE")
                    leg_reqs = [("LEG_A", cfg.venue_a, side_a)]
                elif intent.action == "OPEN_SHORT_A_LONG_B":
                    leg_reqs = [("LEG_A", cfg.venue_a, "SELL"), ("LEG_B", cfg.venue_b, "BUY")]
                elif intent.action == "OPEN_LONG_A_SHORT_B":
                    leg_reqs = [("LEG_A", cfg.venue_a, "BUY"), ("LEG_B", cfg.venue_b, "SELL")]
                elif intent.action == "CLOSE":
                    leg_reqs = [("LEG_A", cfg.venue_a, "CLOSE"), ("LEG_B", cfg.venue_b, "CLOSE")]
                else:
                    block_reason = f"UNSUPPORTED_ACTION({intent.action})"
                    leg_reqs = []

                for leg_id, venue, side in leg_reqs:
                    tick = bbo_by_venue.get(venue)
                    if tick is None:
                        block_reason = f"MISSING_BBO({venue}:{intent.symbol})"
                        break
                    if tick.venue != venue or tick.symbol != intent.symbol:
                        block_reason = f"BBO_IDENTITY_MISMATCH(want={venue}:{intent.symbol},got={tick.venue}:{tick.symbol})"
                        break
                    if (
                        not _is_finite_num(tick.bid)
                        or not _is_finite_num(tick.ask)
                        or float(tick.bid) <= 0.0
                        or float(tick.ask) < float(tick.bid)
                        or not _is_finite_num(tick.ts_ms)
                    ):
                        block_reason = f"INVALID_OR_CROSSED_BBO({venue}:bid={tick.bid},ask={tick.ask})"
                        break
                    age_ms = eval_now - int(tick.ts_ms)
                    if age_ms > self.max_bbo_age_ms or age_ms < -5000:
                        block_reason = f"STALE_BBO({venue}:age={age_ms}ms>{self.max_bbo_age_ms}ms)"
                        break
                    sp_bps = tick.spread_bps
                    if not math.isfinite(sp_bps) or sp_bps > spread_ceiling:
                        block_reason = f"SPREAD_CEILING_EXCEEDED({venue}:{sp_bps:.2f}bp>{spread_ceiling:.2f}bp)"
                        break
                    ref_px = tick.ask if side == "BUY" else (tick.bid if side == "SELL" else tick.mid)
                    slip_mult = float(cfg.slippage_per_leg_bps) / 10000.0
                    limit_px = ref_px * (1.0 + slip_mult) if side == "BUY" else ref_px * (1.0 - slip_mult)
                    if not (math.isfinite(ref_px) and math.isfinite(limit_px) and limit_px > 0.0):
                        block_reason = f"INVALID_COMPUTED_PRICE({venue})"
                        break
                    legs.append(
                        LegSpec(
                            leg_id=leg_id,
                            venue=venue,
                            symbol=intent.symbol,
                            side=side,
                            notional_usd=safe_notional,
                            ref_price=round(ref_px, 8),
                            limit_price=round(limit_px, 8),
                            spread_bps=round(sp_bps, 4),
                            bbo_ts_ms=int(tick.ts_ms),
                        )
                    )

        # Atomic transaction: deduplicate by idempotency_key -> check kill switch -> persist plan & hash-chained audit
        with transaction(self.conn):
            existing = self.conn.execute(
                "SELECT payload_json FROM execution_plans WHERE idempotency_key=?",
                (idem_key,),
            ).fetchone()
            if existing is not None:
                prev_dict = json.loads(existing["payload_json"])
                prev_legs = [LegSpec(**lg) for lg in prev_dict.get("legs", [])]
                prev_dict["legs"] = prev_legs
                prev_dict["duplicate"] = True
                return ExecutionPlan(**prev_dict)

            if self.kill_switch.is_engaged_in_tx():
                block_reason = "KILL_SWITCH_ENGAGED"

            status = "BLOCKED" if block_reason else "PLANNED_SHADOW"
            event_type = "PLAN_BLOCKED" if block_reason else "PLAN_CREATED_SHADOW"
            next_seq, _ = self.audit.next_seq_and_prev_in_tx()
            plan = ExecutionPlan(
                plan_id=plan_id,
                idempotency_key=idem_key,
                run_mode="SHADOW_DRY_RUN",
                status=status,
                block_reason=block_reason,
                ts_ms=eval_now,
                strategy_id=str(cfg.strategy_id),
                version=str(cfg.version),
                config_sha256=config_sha,
                symbol=str(intent.symbol),
                action=str(intent.action),
                notional_usd=safe_notional,
                expected_edge_bps=safe_edge,
                legs=[] if block_reason else legs,
                audit_seq=next_seq,
            )
            seq, _ = self.audit.append_in_tx(
                event_type,
                plan.to_dict(),
                plan_id=plan_id,
                idempotency_key=idem_key,
                ts_ms=eval_now,
            )
            plan.audit_seq = seq
            self.conn.execute(
                """
                INSERT INTO execution_plans
                (plan_id, idempotency_key, ts_ms, strategy_id, version, config_sha256,
                 symbol, action, status, notional_usd, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.plan_id,
                    plan.idempotency_key,
                    plan.ts_ms,
                    plan.strategy_id,
                    plan.version,
                    plan.config_sha256,
                    plan.symbol,
                    plan.action,
                    plan.status,
                    plan.notional_usd,
                    _canonical_json(plan.to_dict()),
                ),
            )
            return plan

    def recent_plans(self, limit: int = 30) -> list[dict[str, Any]]:
        rows = self.conn.fetch_all(
            "SELECT payload_json FROM execution_plans ORDER BY ts_ms DESC LIMIT ?",
            (limit,),
        )
        return [json.loads(r["payload_json"]) for r in rows]

    def summary(self) -> dict[str, Any]:
        ok, detail, chain_len = self.audit.verify_chain()
        head_seq, head_hash = self.audit.head()
        plans = self.recent_plans(limit=30)
        return {
            "live_execution_enabled": LIVE_EXECUTION_ENABLED,
            "run_mode": "SHADOW_DRY_RUN",
            "kill_switch": self.kill_switch.state(),
            "audit_chain_valid": ok,
            "audit_chain_detail": detail,
            "audit_chain_length": chain_len,
            "audit_head_seq": head_seq,
            "audit_head_hash": head_hash,
            "recent_plans": plans,
        }

    def close(self) -> None:
        self.conn.close()
