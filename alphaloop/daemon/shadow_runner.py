"""Always-On Shadow Portfolio Daemon & Shadow-vs-Backtest Parity Auditor (Stage 1 S1-02, Red-Team Hardened).

Strictly DRY-RUN / PAPER-ONLY (`LIVE_EXECUTION_ENABLED: Final[bool] = False`):
  - Zero live order placement, zero private keys, zero mainnet fund movement.
  - Drives active `SHADOW_LIVE` / `PAPER_LIVE` strategy plugins on real exchange BBO &
    funding feeds via `ExecutionBridge` + `HotConfigWatcher`, with persistent SQLite runtime
    state (`shadow_runtime_state`) surviving restarts cleanly.
  - Computes continuous Shadow-vs-Backtest parity scorecards persisted to SQLite WAL
    (`shadow_parity_snapshots`) and exported atomically to `shadow_parity_report.json`.
  - Enforces 4 Fail-Closed Automatic Kill / Halt Guards:
      1. Hash-chain integrity failure (`verify_chain() == False`) -> KillSwitch + P0 alert.
      2. Per-venue feed staleness (ANY required venue age > `max_bbo_stale_sec`) -> `DEGRADED_HALT`.
      3. Sustained wide live spread (`spread_ratio > 1.5x` using exact 1-leg/2-leg assumed slip
         for `>= wide_spread_kill_sec`) -> `KillSwitch(SLIPPAGE_EXCEEDED)` + P0 alert.
      4. Shadow-vs-Backtest PnL drift (`|drift_bps| > 150bp` after `drift_check_min_hours`) ->
         `KillSwitch(RISK_DRAWDOWN)` + P0 alert.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import signal
import statistics
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from alphaloop.core.db import connect_wal, transaction
from alphaloop.core.models import BBOTick, MarketEvent, OrderIntent, StrategyConfig
from alphaloop.core.provenance import sha256_obj, utc_now_ms
from alphaloop.daemon.watchdog_hook import AlertNotifier, WatchdogHook
from alphaloop.data.bbo_collector import MultiVenueBBOCollector
from alphaloop.data.store import MarketDataStore
from alphaloop.execution.bridge import (
    ALLOWED_SHADOW_STATUSES,
    LIVE_EXECUTION_ENABLED,
    ExecutionBridge,
    KillTrigger,
)
from alphaloop.plugins import PluginRegistry, default_registry

logger = logging.getLogger(__name__)

DAEMON_LIVE_EXECUTION_ENABLED: Final[bool] = False
TAKER_FEE_BPS_DEFAULT: Final[float] = 4.5


def _is_finite(val: Any) -> bool:
    return (not isinstance(val, bool)) and isinstance(val, (int, float)) and math.isfinite(float(val))


def _is_dual_venue(cfg: StrategyConfig) -> bool:
    return bool(cfg.venue_b and cfg.venue_b != cfg.venue_a)


def _assumed_total_slip_bps(cfg: StrategyConfig) -> float:
    mult = 2.0 if _is_dual_venue(cfg) else 1.0
    return max(0.1, mult * float(cfg.slippage_per_leg_bps))


def _effective_leg_fee_bps(cfg: StrategyConfig, leg_fee_bps: float) -> float:
    mode = str(cfg.execution_mode or "MAKER_MAKER").upper()
    if mode == "TAKER_TAKER":
        return max(float(leg_fee_bps), TAKER_FEE_BPS_DEFAULT)
    return float(leg_fee_bps)


def _atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
    """Persist JSON atomically via temp file + fsync + os.replace (STD-34 compliant)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError as cleanup_exc:
            logger.warning("Failed to cleanup temp file %s after write error: %s", tmp_name, cleanup_exc)
        raise


@dataclass(frozen=True)
class ShadowParityRecord:
    """Point-in-time Shadow vs same-window Backtest parity verification record."""

    snapshot_id: str
    ts_ms: int
    strategy_id: str
    plugin_id: str
    version: str
    window_start_ms: int
    window_end_ms: int
    elapsed_hours: float
    shadow_equity_usd: float
    shadow_return_pct: float
    backtest_return_pct: float
    backtest_comparable: bool
    drift_bps: float
    live_median_half_spread_bps: float
    assumed_slippage_bps: float
    spread_ratio: float
    plans_created: int
    plans_blocked: int
    open_positions: int
    bbo_freshness_sec: float
    audit_chain_valid: bool
    kill_switch_engaged: bool
    status: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ShadowParityAuditor:
    """Persists Shadow equity, runtime state, and Shadow-vs-Backtest parity snapshots in SQLite WAL."""

    def __init__(
        self,
        db_path: str | Path,
        market_store: MarketDataStore,
        registry: PluginRegistry | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.market_store = market_store
        self.registry = registry or default_registry()
        self.conn = connect_wal(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_parity_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    ts_ms INTEGER NOT NULL,
                    strategy_id TEXT NOT NULL,
                    plugin_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    drift_bps REAL NOT NULL,
                    spread_ratio REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shadow_parity_strat
                ON shadow_parity_snapshots (strategy_id, ts_ms DESC);
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_equity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    strategy_id TEXT NOT NULL,
                    equity_usd REAL NOT NULL,
                    realized_pnl_usd REAL NOT NULL,
                    unrealized_pnl_usd REAL NOT NULL,
                    open_positions INTEGER NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_runtime_state (
                    strategy_id TEXT PRIMARY KEY,
                    started_ts_ms INTEGER NOT NULL,
                    realized_pnl_usd REAL NOT NULL,
                    plans_created INTEGER NOT NULL,
                    plans_blocked INTEGER NOT NULL,
                    open_positions_json TEXT NOT NULL,
                    plugin_state_json TEXT NOT NULL DEFAULT '{}',
                    last_funding_ts_json TEXT NOT NULL DEFAULT '{}',
                    updated_ts_ms INTEGER NOT NULL
                );
                """
            )
            cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(shadow_runtime_state)").fetchall()}
            if "plugin_state_json" not in cols:
                self.conn.execute("ALTER TABLE shadow_runtime_state ADD COLUMN plugin_state_json TEXT NOT NULL DEFAULT '{}'")
            if "last_funding_ts_json" not in cols:
                self.conn.execute("ALTER TABLE shadow_runtime_state ADD COLUMN last_funding_ts_json TEXT NOT NULL DEFAULT '{}'")

    @staticmethod
    def _encode_state(obj: Any) -> Any:
        if isinstance(obj, dict):
            if any(isinstance(k, tuple) for k in obj):
                return {
                    "__tuple_map__": [
                        [list(k) if isinstance(k, tuple) else k, ShadowParityAuditor._encode_state(v)]
                        for k, v in obj.items()
                    ]
                }
            return {str(k): ShadowParityAuditor._encode_state(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ShadowParityAuditor._encode_state(x) for x in obj]
        if isinstance(obj, tuple):
            return {"__tuple__": [ShadowParityAuditor._encode_state(x) for x in obj]}
        return obj

    @staticmethod
    def _decode_state(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "__tuple_map__" in obj and len(obj) == 1:
                out: dict[Any, Any] = {}
                for k, v in obj["__tuple_map__"]:
                    key = tuple(k) if isinstance(k, list) else k
                    out[key] = ShadowParityAuditor._decode_state(v)
                return out
            if "__tuple__" in obj and len(obj) == 1:
                return tuple(ShadowParityAuditor._decode_state(x) for x in obj["__tuple__"])
            return {k: ShadowParityAuditor._decode_state(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ShadowParityAuditor._decode_state(x) for x in obj]
        return obj

    def save_runtime_state(
        self,
        strategy_id: str,
        *,
        started_ts_ms: int,
        realized_pnl_usd: float,
        plans_created: int,
        plans_blocked: int,
        open_positions: dict[str, dict[str, Any]],
        updated_ts_ms: int,
        plugin_state: dict[str, Any] | None = None,
        last_funding_ts: dict[str, int] | None = None,
    ) -> None:
        encoded_pstate = self._encode_state(plugin_state or {})
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT INTO shadow_runtime_state
                (strategy_id, started_ts_ms, realized_pnl_usd, plans_created, plans_blocked,
                 open_positions_json, plugin_state_json, last_funding_ts_json, updated_ts_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy_id) DO UPDATE SET
                    started_ts_ms = excluded.started_ts_ms,
                    realized_pnl_usd = excluded.realized_pnl_usd,
                    plans_created = excluded.plans_created,
                    plans_blocked = excluded.plans_blocked,
                    open_positions_json = excluded.open_positions_json,
                    plugin_state_json = excluded.plugin_state_json,
                    last_funding_ts_json = excluded.last_funding_ts_json,
                    updated_ts_ms = excluded.updated_ts_ms
                """,
                (
                    strategy_id,
                    int(started_ts_ms),
                    float(realized_pnl_usd),
                    int(plans_created),
                    int(plans_blocked),
                    json.dumps(open_positions, sort_keys=True, allow_nan=False),
                    json.dumps(encoded_pstate, sort_keys=True, default=str, allow_nan=False),
                    json.dumps(last_funding_ts or {}, sort_keys=True, allow_nan=False),
                    int(updated_ts_ms),
                ),
            )

    def load_all_runtime_states(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.fetch_all(
            "SELECT strategy_id, started_ts_ms, realized_pnl_usd, plans_created, plans_blocked, "
            "open_positions_json, plugin_state_json, last_funding_ts_json "
            "FROM shadow_runtime_state"
        )
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            out[str(r["strategy_id"])] = {
                "started_ts_ms": int(r["started_ts_ms"]),
                "realized_pnl_usd": float(r["realized_pnl_usd"]),
                "plans_created": int(r["plans_created"]),
                "plans_blocked": int(r["plans_blocked"]),
                "open_positions": json.loads(str(r["open_positions_json"])),
                "plugin_state": self._decode_state(json.loads(str(r["plugin_state_json"] or "{}"))),
                "last_funding_ts": json.loads(str(r["last_funding_ts_json"] or "{}")),
            }
        return out

    def record_equity(
        self,
        ts_ms: int,
        strategy_id: str,
        equity_usd: float,
        realized_pnl_usd: float,
        unrealized_pnl_usd: float,
        open_positions: int,
    ) -> None:
        if not (_is_finite(equity_usd) and _is_finite(realized_pnl_usd) and _is_finite(unrealized_pnl_usd)):
            raise ValueError("Non-finite shadow equity metrics rejected fail-closed")
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT INTO shadow_equity_log
                (ts_ms, strategy_id, equity_usd, realized_pnl_usd, unrealized_pnl_usd, open_positions)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(ts_ms),
                    str(strategy_id),
                    round(float(equity_usd), 4),
                    round(float(realized_pnl_usd), 4),
                    round(float(unrealized_pnl_usd), 4),
                    int(open_positions),
                ),
            )

    def save_snapshot(self, rec: ShadowParityRecord) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT OR REPLACE INTO shadow_parity_snapshots
                (snapshot_id, ts_ms, strategy_id, plugin_id, version, status, drift_bps, spread_ratio, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.snapshot_id,
                    rec.ts_ms,
                    rec.strategy_id,
                    rec.plugin_id,
                    rec.version,
                    rec.status,
                    rec.drift_bps,
                    rec.spread_ratio,
                    json.dumps(rec.to_dict(), sort_keys=True, allow_nan=False),
                ),
            )

    def evaluate_strategy(
        self,
        cfg: StrategyConfig,
        *,
        window_start_ms: int,
        window_end_ms: int,
        shadow_equity_usd: float,
        live_half_spreads_bps: list[float],
        plans_created: int,
        plans_blocked: int,
        open_positions: int,
        bbo_freshness_sec: float,
        audit_chain_valid: bool,
        kill_switch_engaged: bool,
        max_spread_ratio: float = 1.5,
        max_drift_bps: float = 150.0,
        drift_check_min_hours: float = 24.0,
    ) -> ShadowParityRecord:
        """Compare live Shadow equity & realized spread against same-window plugin backtest."""
        cap = float(cfg.capital_usd)
        if not (_is_finite(cap) and cap > 0 and _is_finite(shadow_equity_usd)):
            raise ValueError("Invalid capital or non-finite shadow equity")

        elapsed_hours = max(0.0, (window_end_ms - window_start_ms) / 3_600_000.0)
        shadow_ret_pct = ((float(shadow_equity_usd) - cap) / cap) * 100.0

        finite_spreads = [float(x) for x in live_half_spreads_bps if _is_finite(x) and float(x) >= 0.0]
        med_half_spread = statistics.median(finite_spreads) if finite_spreads else 0.0
        assumed_slip = _assumed_total_slip_bps(cfg)
        spread_ratio = med_half_spread / assumed_slip if finite_spreads else 0.0

        backtest_ret_pct = 0.0
        backtest_comparable = False
        detail_parts: list[str] = []

        if elapsed_hours >= 1.0:
            try:
                plugin = self.registry.resolve_for_config(cfg)
                bt = plugin.evaluate(
                    self.market_store,
                    cfg,
                    start_ms=int(window_start_ms),
                    end_ms=int(window_end_ms),
                )
                if bt.metrics and bt.metrics.days > 0 and _is_finite(bt.metrics.total_return_pct):
                    backtest_ret_pct = float(bt.metrics.total_return_pct)
                    backtest_comparable = True
                else:
                    detail_parts.append("INSUFFICIENT_HISTORY_IN_LIVE_WINDOW")
            except Exception as exc:
                logger.warning("Same-window parity backtest skipped for %s: %s", cfg.strategy_id, exc)
                detail_parts.append(f"BACKTEST_UNAVAILABLE:{type(exc).__name__}")
        else:
            detail_parts.append("WARMUP_WINDOW_LT_1H")

        drift_bps = (shadow_ret_pct - backtest_ret_pct) * 100.0 if backtest_comparable else 0.0

        status = "PARITY_OK"
        if kill_switch_engaged or not audit_chain_valid:
            status = "KILL_TRIPPED"
            detail_parts.append("KILL_OR_CHAIN_INVALID")
        elif finite_spreads and spread_ratio > max_spread_ratio:
            status = "WARN_WIDE_SPREAD"
            detail_parts.append(f"SPREAD_RATIO={spread_ratio:.2f}x>{max_spread_ratio:.2f}x")
        elif backtest_comparable and elapsed_hours >= drift_check_min_hours and abs(drift_bps) > max_drift_bps:
            status = "DRIFT_EXCEEDED"
            detail_parts.append(f"DRIFT={drift_bps:+.1f}bp>|{max_drift_bps:.1f}bp|")

        snap_id = "PAR-" + sha256_obj({
            "sid": cfg.strategy_id,
            "ver": cfg.version,
            "ts": int(window_end_ms),
            "eq": round(shadow_equity_usd, 4),
            "plans": int(plans_created),
        })[:14]

        rec = ShadowParityRecord(
            snapshot_id=snap_id,
            ts_ms=int(window_end_ms),
            strategy_id=cfg.strategy_id,
            plugin_id=cfg.plugin_id or "funding_carry",
            version=cfg.version,
            window_start_ms=int(window_start_ms),
            window_end_ms=int(window_end_ms),
            elapsed_hours=round(elapsed_hours, 4),
            shadow_equity_usd=round(float(shadow_equity_usd), 4),
            shadow_return_pct=round(shadow_ret_pct, 4),
            backtest_return_pct=round(backtest_ret_pct, 4),
            backtest_comparable=backtest_comparable,
            drift_bps=round(drift_bps, 2),
            live_median_half_spread_bps=round(med_half_spread, 4),
            assumed_slippage_bps=round(assumed_slip, 4),
            spread_ratio=round(spread_ratio, 4),
            plans_created=int(plans_created),
            plans_blocked=int(plans_blocked),
            open_positions=int(open_positions),
            bbo_freshness_sec=round(float(bbo_freshness_sec), 2),
            audit_chain_valid=bool(audit_chain_valid),
            kill_switch_engaged=bool(kill_switch_engaged),
            status=status,
            detail="; ".join(detail_parts) if detail_parts else "OK",
        )
        self.save_snapshot(rec)
        return rec

    def recent_snapshots(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.fetch_all(
            "SELECT payload_json FROM shadow_parity_snapshots ORDER BY ts_ms DESC LIMIT ?",
            (max(1, int(limit)),),
        )
        return [json.loads(r["payload_json"]) for r in rows]

    def close(self) -> None:
        self.conn.close()


class ShadowPortfolioDaemon:
    """Always-on SHADOW_DRY_RUN daemon driving active SHADOW_LIVE strategies with parity guards."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        registry: PluginRegistry | None = None,
        watchdog: WatchdogHook | None = None,
        notifier: AlertNotifier | None = None,
        max_bbo_stale_sec: float = 180.0,
        max_spread_ratio: float = 1.5,
        wide_spread_kill_sec: float = 7200.0,
        max_drift_bps: float = 150.0,
        drift_check_min_hours: float = 24.0,
        bbo_symbols: list[str] | None = None,
    ) -> None:
        if LIVE_EXECUTION_ENABLED or DAEMON_LIVE_EXECUTION_ENABLED:
            raise RuntimeError("ShadowPortfolioDaemon refuses non-shadow execution mode")

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.data_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.registry = registry or default_registry()
        self.store = MarketDataStore(self.data_dir / "market_data.sqlite")
        self.bridge = ExecutionBridge(self.state_dir, max_bbo_age_ms=int(max_bbo_stale_sec * 1000.0))
        self.watcher = self.bridge.watcher
        self.parity = ShadowParityAuditor(self.state_dir / "shadow_parity.sqlite", self.store, self.registry)
        self.watchdog = watchdog or WatchdogHook(async_mode=True)
        self.notifier = notifier or AlertNotifier(cooldown_sec=1800.0)
        self._owns_watchdog = watchdog is None

        self.max_bbo_stale_sec = max(5.0, float(max_bbo_stale_sec))
        self.max_spread_ratio = max(1.0, float(max_spread_ratio))
        self.wide_spread_kill_sec = max(0.0, float(wide_spread_kill_sec))
        self.max_drift_bps = max(10.0, float(max_drift_bps))
        self.drift_check_min_hours = max(0.0, float(drift_check_min_hours))
        self.bbo_symbols = [s.upper() for s in (bbo_symbols or [
            "PAXG", "BTC", "ETH", "SOL", "XRP", "DOGE", "LINK", "AAVE", "SUI", "AVAX", "HYPE", "LTC",
        ])]
        self._collector = MultiVenueBBOCollector(self.bbo_symbols)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._seq = 0
        self._plugin_states: dict[str, dict[str, Any]] = {}
        self._open_positions: dict[str, dict[str, dict[str, Any]]] = {}
        self._realized_pnl: dict[str, float] = {}
        self._started_ts_ms: dict[str, int] = {}
        self._plans_created: dict[str, int] = {}
        self._plans_blocked: dict[str, int] = {}
        self._spread_history: dict[str, list[float]] = {}
        self._wide_spread_since_ms: dict[str, int | None] = {}
        self._latest_bbo_cache: dict[tuple[str, str], BBOTick] = {}
        self._last_delivered_bbo_ts: dict[tuple[str, str, str], int] = {}
        self._last_delivered_funding_ts: dict[tuple[str, str, str], int] = {}
        self._warmed_strategies: set[str] = set()

        # Hydrate persisted shadow runtime state across restarts (Red-Team HIGH #8)
        for sid, saved in self.parity.load_all_runtime_states().items():
            self._started_ts_ms[sid] = saved["started_ts_ms"]
            self._realized_pnl[sid] = saved["realized_pnl_usd"]
            self._plans_created[sid] = saved["plans_created"]
            self._plans_blocked[sid] = saved["plans_blocked"]
            self._open_positions[sid] = saved["open_positions"]
            if isinstance(saved.get("plugin_state"), dict):
                self._plugin_states[sid] = saved["plugin_state"]
            for fkey_s, fts in (saved.get("last_funding_ts") or {}).items():
                if ":" in str(fkey_s):
                    v, sym = str(fkey_s).split(":", 1)
                    self._last_delivered_funding_ts[(sid, v, sym)] = int(fts)

    def _warmup_plugin_state_from_candles(self, cfg: StrategyConfig, state: dict[str, Any], ts_now: int) -> None:
        """Warm up TrendCTA resampled ohlc_hist (high, low, close) buffers from completed 1h candle buckets (R2-1)."""
        sid = cfg.strategy_id
        if sid in self._warmed_strategies:
            return
        self._warmed_strategies.add(sid)
        if (cfg.plugin_id or "") != "trend_cta":
            return
        params = cfg.params or {}
        tf_h = max(1, int(params.get("timeframe_hours", 12)))
        entry_n = max(5, int(params.get("entry_n", 40)))
        atr_n = max(5, int(params.get("atr_n", 14)))
        cap_n = max(entry_n, atr_n) + 5
        tf_ms = tf_h * 3_600_000
        current_open_bucket = (int(ts_now) // tf_ms) * tf_ms
        lookback_ms = max(tf_h * (cap_n + 10) * 3_600_000, 30 * 86_400_000)
        ohlc_hist: dict[str, list[tuple[float, float, float]]] = state.setdefault("ohlc_hist", {})
        for sym in cfg.symbols:
            if len(ohlc_hist.get(sym, [])) >= entry_n:
                continue
            candles = self.store.load_candles(cfg.venue_a, sym, start_ms=ts_now - lookback_ms, end_ms=ts_now)
            if not candles:
                continue
            buckets: dict[int, list[Any]] = {}
            for c in candles:
                b = (int(c.ts_ms) // tf_ms) * tf_ms
                # Strictly exclude currently open/incomplete bucket (zero lookahead!)
                if b >= current_open_bucket:
                    continue
                buckets.setdefault(b, []).append(c)
            completed: list[tuple[float, float, float]] = []
            for bts in sorted(buckets.keys()):
                grp = sorted(buckets[bts], key=lambda x: x.ts_ms)
                if len(grp) < tf_h:
                    continue
                completed.append((
                    max(float(x.high) for x in grp),
                    min(float(x.low) for x in grp),
                    float(grp[-1].close),
                ))
            if completed:
                ohlc_hist[sym] = completed[-cap_n:]

    def _load_latest_bbo_map(self, symbols: list[str], venues: set[str], now_ms: int) -> dict[tuple[str, str], BBOTick]:
        lookback_ms = max(int(self.max_bbo_stale_sec * 1000.0 * 4), 3_600_000)
        start_ms = max(0, int(now_ms) - lookback_ms)
        merged: dict[tuple[str, str], BBOTick] = {}
        for venue in sorted(venues):
            if not venue:
                continue
            for sym in symbols:
                ticks = self.store.load_bbo_ticks(venue, sym, start_ms=start_ms, end_ms=now_ms + 30_000)
                if ticks:
                    merged[(venue, sym)] = ticks[-1]
        # Merge cache strictly by max ts_ms & prune expired cache entries (Red-Team HIGH #8)
        stale_cutoff = int(now_ms - self.max_bbo_stale_sec * 1000.0)
        for key, cached in list(self._latest_bbo_cache.items()):
            if cached.ts_ms < stale_cutoff:
                self._latest_bbo_cache.pop(key, None)
                continue
            prev = merged.get(key)
            if prev is None or cached.ts_ms >= prev.ts_ms:
                merged[key] = cached
        return merged

    def _accrue_funding_on_open_positions(
        self,
        cfg: StrategyConfig,
        venue: str,
        sym: str,
        funding_rate: float,
        fb_ts_ms: int,
    ) -> None:
        """Accrue realized perpetual funding cashflows ONLY for bars strictly after entry_ts_ms (R2-2)."""
        pos = self._open_positions.get(cfg.strategy_id, {}).get(sym)
        if pos is None:
            return
        if int(fb_ts_ms) <= int(pos.get("entry_ts_ms", 0)):
            return
        notional = float(pos.get("notional", 0.0))
        if notional <= 0 or not _is_finite(funding_rate):
            return
        cashflow = 0.0
        if pos.get("single_venue"):
            if venue == cfg.venue_a:
                # Positive funding: LONG pays, SHORT receives
                sign = -1.0 if pos["action"] == "OPEN_LONG" else 1.0
                cashflow = sign * notional * float(funding_rate)
        else:
            if pos["action"] == "OPEN_SHORT_A_LONG_B":
                # Short A receives +rate_a, Long B pays -rate_b
                if venue == cfg.venue_a:
                    cashflow = notional * float(funding_rate)
                elif venue == cfg.venue_b:
                    cashflow = -notional * float(funding_rate)
            else:
                if venue == cfg.venue_a:
                    cashflow = -notional * float(funding_rate)
                elif venue == cfg.venue_b:
                    cashflow = notional * float(funding_rate)
        pos["accrued_funding"] = float(pos.get("accrued_funding", 0.0)) + cashflow

    def _apply_shadow_fill(
        self,
        cfg: StrategyConfig,
        intent: OrderIntent,
        bbo_a: BBOTick | None,
        bbo_b: BBOTick | None,
        fill_ts_ms: int,
    ) -> None:
        """Maintain realistic paper positions, fees, and funding accrual for PLANNED_SHADOW fills."""
        sid = cfg.strategy_id
        positions = self._open_positions.setdefault(sid, {})
        realized = self._realized_pnl.setdefault(sid, 0.0)
        sym = intent.symbol
        action = str(intent.action)
        notional = float(intent.notional_usd)
        fee_a = _effective_leg_fee_bps(cfg, cfg.fee_a_bps) / 10_000.0
        fee_b = _effective_leg_fee_bps(cfg, cfg.fee_b_bps) / 10_000.0
        dual_fee = fee_a + fee_b

        if action in ("OPEN_LONG", "OPEN_SHORT"):
            if bbo_a is None or sym in positions:
                return
            entry_px = bbo_a.ask if action == "OPEN_LONG" else bbo_a.bid
            if entry_px <= 0:
                return
            positions[sym] = {
                "single_venue": True,
                "action": action,
                "notional": notional,
                "entry_ts_ms": int(fill_ts_ms),
                "entry_a": entry_px,
                "last_mark_a": bbo_a.mid,
                "accrued_funding": 0.0,
            }
            self._realized_pnl[sid] = realized - notional * fee_a
            return

        if action in ("OPEN_SHORT_A_LONG_B", "OPEN_LONG_A_SHORT_B"):
            if bbo_a is None or bbo_b is None or sym in positions:
                return
            entry_a = bbo_a.bid if action == "OPEN_SHORT_A_LONG_B" else bbo_a.ask
            entry_b = bbo_b.ask if action == "OPEN_SHORT_A_LONG_B" else bbo_b.bid
            if entry_a <= 0 or entry_b <= 0:
                return
            positions[sym] = {
                "single_venue": False,
                "action": action,
                "notional": notional,
                "entry_ts_ms": int(fill_ts_ms),
                "entry_a": entry_a,
                "entry_b": entry_b,
                "last_mark_a": bbo_a.mid,
                "last_mark_b": bbo_b.mid,
                "accrued_funding": 0.0,
            }
            self._realized_pnl[sid] = realized - notional * dual_fee
            return

        if action in ("CLOSE", "CLOSE_PAIR"):
            pos = positions.pop(sym, None)
            if pos is None:
                return
            accrued_funding = float(pos.get("accrued_funding", 0.0))
            if pos.get("single_venue"):
                if bbo_a is None:
                    positions[sym] = pos
                    return
                if pos["action"] == "OPEN_LONG":
                    gross = ((bbo_a.bid - pos["entry_a"]) / pos["entry_a"]) * pos["notional"]
                else:
                    gross = ((pos["entry_a"] - bbo_a.ask) / pos["entry_a"]) * pos["notional"]
                self._realized_pnl[sid] = realized + gross + accrued_funding - pos["notional"] * fee_a
            else:
                if bbo_a is None or bbo_b is None:
                    positions[sym] = pos
                    return
                if pos["action"] == "OPEN_SHORT_A_LONG_B":
                    pnl_a = ((pos["entry_a"] - bbo_a.ask) / pos["entry_a"]) * pos["notional"]
                    pnl_b = ((bbo_b.bid - pos["entry_b"]) / pos["entry_b"]) * pos["notional"]
                else:
                    pnl_a = ((bbo_a.bid - pos["entry_a"]) / pos["entry_a"]) * pos["notional"]
                    pnl_b = ((pos["entry_b"] - bbo_b.ask) / pos["entry_b"]) * pos["notional"]
                self._realized_pnl[sid] = realized + pnl_a + pnl_b + accrued_funding - pos["notional"] * dual_fee

    def _compute_mtm_equity(
        self,
        cfg: StrategyConfig,
        bbo_map: dict[tuple[str, str], BBOTick],
    ) -> tuple[float, float, float]:
        """Compute MTM equity carrying forward last known mark price on transient missing ticks."""
        sid = cfg.strategy_id
        realized = self._realized_pnl.get(sid, 0.0)
        unr = 0.0
        for sym, pos in self._open_positions.get(sid, {}).items():
            qa = bbo_map.get((cfg.venue_a, sym))
            if qa is not None:
                pos["last_mark_a"] = qa.bid if pos["action"] in ("OPEN_LONG", "OPEN_LONG_A_SHORT_B") else qa.ask
            mark_a = float(pos.get("last_mark_a", pos["entry_a"]))
            accrued_f = float(pos.get("accrued_funding", 0.0))

            if pos.get("single_venue"):
                px_a = qa.bid if (qa is not None and pos["action"] == "OPEN_LONG") else (
                    qa.ask if qa is not None else mark_a
                )
                if pos["action"] == "OPEN_LONG":
                    unr += ((px_a - pos["entry_a"]) / pos["entry_a"]) * pos["notional"] + accrued_f
                else:
                    unr += ((pos["entry_a"] - px_a) / pos["entry_a"]) * pos["notional"] + accrued_f
            else:
                qb = bbo_map.get((cfg.venue_b, sym))
                if qb is not None:
                    pos["last_mark_b"] = qb.bid if pos["action"] == "OPEN_SHORT_A_LONG_B" else qb.ask
                mark_b = float(pos.get("last_mark_b", pos["entry_b"]))
                if pos["action"] == "OPEN_SHORT_A_LONG_B":
                    px_a = qa.ask if qa is not None else mark_a
                    px_b = qb.bid if qb is not None else mark_b
                    unr += ((pos["entry_a"] - px_a) / pos["entry_a"]) * pos["notional"]
                    unr += ((px_b - pos["entry_b"]) / pos["entry_b"]) * pos["notional"] + accrued_f
                else:
                    px_a = qa.bid if qa is not None else mark_a
                    px_b = qb.ask if qb is not None else mark_b
                    unr += ((px_a - pos["entry_a"]) / pos["entry_a"]) * pos["notional"]
                    unr += ((pos["entry_b"] - px_b) / pos["entry_b"]) * pos["notional"] + accrued_f
        return float(cfg.capital_usd) + realized + unr, realized, unr

    def run_once(
        self,
        *,
        now_ms: int | None = None,
        live_ticks: list[BBOTick] | None = None,
        collect_live: bool = False,
        parity_report_path: Path | None = None,
    ) -> dict[str, Any]:
        """Execute one deterministic SHADOW_DRY_RUN cycle and update parity scorecard."""
        if LIVE_EXECUTION_ENABLED or DAEMON_LIVE_EXECUTION_ENABLED:
            raise RuntimeError("ShadowPortfolioDaemon forbids live execution")

        with self._lock:
            ts_now = int(now_ms if now_ms is not None else utc_now_ms())
            self._seq += 1
            trace_id = f"SHDCYC-{ts_now}-{self._seq}"

            # 1) Verify cryptographic audit chain integrity BEFORE doing any work
            chain_ok, chain_detail, _chain_len = self.bridge.audit.verify_chain()
            if not chain_ok:
                if not self.bridge.kill_switch.is_engaged():
                    self.bridge.kill_switch.engage(
                        KillTrigger.CONFIG_CORRUPT,
                        actor="shadow-daemon",
                        detail=f"AUDIT_CHAIN_CORRUPT:{chain_detail}",
                    )
                self.notifier.emit(
                    "P0",
                    "AUDIT_CHAIN_CORRUPT",
                    "Shadow Audit Chain Integrity Failure",
                    f"trace={trace_id} detail={chain_detail}",
                )
                return self._finalize_report(
                    ts_now,
                    trace_id,
                    status="KILL_TRIPPED",
                    bbo_freshness_sec=-1.0,
                    watcher_events=[],
                    strategy_records=[],
                    report_path=parity_report_path,
                    detail=f"AUDIT_CHAIN_CORRUPT:{chain_detail}",
                )

            # 2) Hot-reload & validate active configs (auto-trips CONFIG_CORRUPT once per corrupt file)
            watcher_events = self.watcher.poll()
            if any(ev.get("status") == "CORRUPT_KILL_ENGAGED" for ev in watcher_events):
                self.notifier.emit(
                    "P0",
                    "CONFIG_CORRUPT",
                    "Corrupt Active Strategy Config Detected",
                    f"trace={trace_id} events={watcher_events}",
                )

            loaded_cfgs = self.watcher.all_loaded()
            shadow_cfgs = {
                sid: cfg
                for sid, cfg in loaded_cfgs.items()
                if str(cfg.status).upper() in ALLOWED_SHADOW_STATUSES
            }

            # 3) Gather live BBO ticks (injected, live-collected, or latest stored)
            if collect_live:
                collected, _errs = self._collector.capture_round()
                if collected:
                    self.store.upsert_bbo_ticks(collected)
                    live_ticks = list(live_ticks or []) + collected

            if live_ticks:
                self.store.upsert_bbo_ticks(live_ticks)
                for t in live_ticks:
                    if t.bid > 0 and t.ask >= t.bid and _is_finite(t.bid) and _is_finite(t.ask):
                        prev_c = self._latest_bbo_cache.get((t.venue, t.symbol))
                        if prev_c is None or t.ts_ms >= prev_c.ts_ms:
                            self._latest_bbo_cache[(t.venue, t.symbol)] = t

            venues_needed: set[str] = set()
            symbols_needed: set[str] = set(self.bbo_symbols)
            for cfg in shadow_cfgs.values():
                if cfg.venue_a:
                    venues_needed.add(cfg.venue_a)
                if _is_dual_venue(cfg):
                    venues_needed.add(cfg.venue_b)
                for s in cfg.symbols:
                    symbols_needed.add(s.upper())
            if not venues_needed:
                venues_needed.update({"hyperliquid", "gate"})

            bbo_map = self._load_latest_bbo_map(sorted(symbols_needed), venues_needed, ts_now)
            if not bbo_map:
                self.notifier.emit(
                    "P1",
                    "BBO_FEED_MISSING",
                    "No Valid BBO Ticks Available for Shadow Daemon",
                    f"trace={trace_id} venues={sorted(venues_needed)}",
                )
                return self._finalize_report(
                    ts_now,
                    trace_id,
                    status="DEGRADED_HALT",
                    bbo_freshness_sec=999_999.0,
                    watcher_events=watcher_events,
                    strategy_records=[],
                    report_path=parity_report_path,
                    detail="NO_BBO_TICKS_AVAILABLE",
                )

            # Fail-closed PER-VENUE freshness check (Red-Team HIGH #8):
            # Every required venue for active strategies must have a fresh BBO tick!
            venue_freshest_ts: dict[str, int] = {}
            for (v, _s), tick in bbo_map.items():
                prev_ts = venue_freshest_ts.get(v, 0)
                if tick.ts_ms > prev_ts:
                    venue_freshest_ts[v] = tick.ts_ms

            missing_venues = sorted(v for v in venues_needed if v not in venue_freshest_ts)
            if missing_venues:
                self.notifier.emit(
                    "P1",
                    "VENUE_BBO_MISSING",
                    "Required Exchange Venue BBO Feed Missing",
                    f"trace={trace_id} missing_venues={missing_venues}",
                )
                return self._finalize_report(
                    ts_now,
                    trace_id,
                    status="DEGRADED_HALT",
                    bbo_freshness_sec=999_999.0,
                    watcher_events=watcher_events,
                    strategy_records=[],
                    report_path=parity_report_path,
                    detail=f"MISSING_VENUE_BBO({','.join(missing_venues)})",
                )

            worst_venue_ts = min(venue_freshest_ts[v] for v in venues_needed)
            bbo_freshness_sec = max(0.0, (ts_now - worst_venue_ts) / 1000.0)
            if bbo_freshness_sec > self.max_bbo_stale_sec:
                self.notifier.emit(
                    "P1",
                    "STALE_BBO_FEED",
                    "Shadow Daemon Halted on Stale BBO Feed",
                    f"trace={trace_id} worst_venue_age={bbo_freshness_sec:.1f}s > {self.max_bbo_stale_sec:.1f}s",
                )
                return self._finalize_report(
                    ts_now,
                    trace_id,
                    status="DEGRADED_HALT",
                    bbo_freshness_sec=bbo_freshness_sec,
                    watcher_events=watcher_events,
                    strategy_records=[],
                    report_path=parity_report_path,
                    detail=f"STALE_BBO_FEED({bbo_freshness_sec:.1f}s>{self.max_bbo_stale_sec:.1f}s)",
                )

            # Genuine business-payload heartbeat ONLY after ALL required venues verified fresh!
            self.watchdog.heartbeat(
                "alphaloop_shadow_bbo",
                ts_event_ms=worst_venue_ts,
                seq=self._seq,
                meta={"symbols": len(bbo_map), "venues": len(venues_needed), "strategies": len(shadow_cfgs)},
            )

            # 4) Step each active SHADOW_LIVE strategy
            records: list[ShadowParityRecord] = []
            for sid, cfg in sorted(shadow_cfgs.items()):
                self._started_ts_ms.setdefault(sid, ts_now)
                self._plans_created.setdefault(sid, 0)
                self._plans_blocked.setdefault(sid, 0)
                self._spread_history.setdefault(sid, [])
                state = self._plugin_states.setdefault(sid, {})
                plugin = self.registry.resolve_for_config(cfg)
                self._warmup_plugin_state_from_candles(cfg, state, ts_now)

                step_half_spreads: list[float] = []
                dual = _is_dual_venue(cfg)

                # Deliver ONLY new BBO ticks per (sid, venue, sym) & snapshot plugin state for atomic rollback on BLOCKED
                for sym in cfg.symbols:
                    ta = bbo_map.get((cfg.venue_a, sym))
                    tb = bbo_map.get((cfg.venue_b, sym)) if dual else None

                    if ta is not None and not dual:
                        step_half_spreads.append(ta.spread_bps * 0.5)
                    elif ta is not None and tb is not None and dual:
                        step_half_spreads.append((ta.spread_bps + tb.spread_bps) * 0.5)

                    for tick in (ta, tb):
                        if tick is None:
                            continue
                        dedup_key = (sid, tick.venue, sym)
                        if self._last_delivered_bbo_ts.get(dedup_key, -1) >= tick.ts_ms:
                            continue
                        self._last_delivered_bbo_ts[dedup_key] = tick.ts_ms
                        ev = MarketEvent(
                            kind="BBO",
                            ts_ms=tick.ts_ms,
                            venue=tick.venue,
                            symbol=sym,
                            payload={"bid": tick.bid, "ask": tick.ask, "mid": tick.mid, "spread_bps": tick.spread_bps},
                        )
                        self._step_plugin_and_submit(cfg, plugin, state, ev, bbo_map, ts_now)

                # Deliver NEW realized funding bars sorted strictly chronologically across venues (R2-2)
                funding_venues = [cfg.venue_a, cfg.venue_b] if dual else [cfg.venue_a]
                for sym in cfg.symbols:
                    pending_bars: list[tuple[int, str, Any]] = []
                    for fvenue in funding_venues:
                        if not fvenue:
                            continue
                        f_key = (sid, fvenue, sym)
                        last_f_ts = self._last_delivered_funding_ts.get(f_key, ts_now - 86_400_000)
                        for fb in self.store.load_funding(fvenue, sym, start_ms=last_f_ts + 1, end_ms=ts_now):
                            if fb.ts_ms > self._last_delivered_funding_ts.get(f_key, -1):
                                pending_bars.append((int(fb.ts_ms), fvenue, fb))
                    pending_bars.sort(key=lambda item: (item[0], item[1]))
                    for fb_ts, fvenue, fb in pending_bars:
                        f_key = (sid, fvenue, sym)
                        if fb_ts <= self._last_delivered_funding_ts.get(f_key, -1):
                            continue
                        self._last_delivered_funding_ts[f_key] = fb_ts
                        self._accrue_funding_on_open_positions(cfg, fvenue, sym, fb.funding_rate, fb_ts)
                        ev_f = MarketEvent(
                            kind="FUNDING",
                            ts_ms=fb_ts,
                            venue=fvenue,
                            symbol=sym,
                            payload={
                                "funding_rate": fb.funding_rate,
                                "interval_hours": fb.interval_hours,
                                "apr_pct": fb.apr_pct,
                            },
                        )
                        self._step_plugin_and_submit(cfg, plugin, state, ev_f, bbo_map, ts_now)

                # Keep bounded spread history (last 240 samples)
                hist = self._spread_history[sid]
                hist.extend(step_half_spreads)
                if len(hist) > 240:
                    del hist[:-240]

                # Sustained wide live spread kill guard (exact 1-leg / 2-leg scaling, Red-Team BLOCKER #6)
                assumed_slip = _assumed_total_slip_bps(cfg)
                if step_half_spreads:
                    cur_ratio = statistics.median(step_half_spreads) / assumed_slip
                    if cur_ratio > self.max_spread_ratio:
                        since = self._wide_spread_since_ms.get(sid)
                        if since is None:
                            self._wide_spread_since_ms[sid] = ts_now
                            since = ts_now
                        elapsed_wide_sec = (ts_now - since) / 1000.0
                        if elapsed_wide_sec >= self.wide_spread_kill_sec and not self.bridge.kill_switch.is_engaged():
                            self.bridge.kill_switch.engage(
                                KillTrigger.SLIPPAGE_EXCEEDED,
                                actor="shadow-daemon",
                                detail=f"{sid}: spread_ratio={cur_ratio:.2f}x sustained {elapsed_wide_sec:.0f}s",
                            )
                            self.notifier.emit(
                                "P0",
                                "SPREAD_KILL_ENGAGED",
                                f"Shadow KillSwitch Engaged ({sid})",
                                f"Live spread {cur_ratio:.2f}x > {self.max_spread_ratio:.2f}x for {elapsed_wide_sec:.0f}s",
                            )
                    else:
                        self._wide_spread_since_ms[sid] = None

                # Compute & persist MTM equity + runtime state + Shadow-vs-Backtest Parity record
                eq_usd, realized_usd, unr_usd = self._compute_mtm_equity(cfg, bbo_map)
                open_cnt = len(self._open_positions.get(sid, {}))
                self.parity.record_equity(ts_now, sid, eq_usd, realized_usd, unr_usd, open_cnt)
                sid_funding_ts = {
                    f"{v}:{sym}": ts
                    for (s_id, v, sym), ts in self._last_delivered_funding_ts.items()
                    if s_id == sid
                }
                self.parity.save_runtime_state(
                    sid,
                    started_ts_ms=self._started_ts_ms[sid],
                    realized_pnl_usd=realized_usd,
                    plans_created=self._plans_created[sid],
                    plans_blocked=self._plans_blocked[sid],
                    open_positions=self._open_positions.get(sid, {}),
                    updated_ts_ms=ts_now,
                    plugin_state=self._plugin_states.get(sid, {}),
                    last_funding_ts=sid_funding_ts,
                )

                chain_ok_now, _, _ = self.bridge.audit.verify_chain()
                ks_now = self.bridge.kill_switch.is_engaged()
                rec = self.parity.evaluate_strategy(
                    cfg,
                    window_start_ms=self._started_ts_ms[sid],
                    window_end_ms=ts_now,
                    shadow_equity_usd=eq_usd,
                    live_half_spreads_bps=hist,
                    plans_created=self._plans_created[sid],
                    plans_blocked=self._plans_blocked[sid],
                    open_positions=open_cnt,
                    bbo_freshness_sec=bbo_freshness_sec,
                    audit_chain_valid=chain_ok_now,
                    kill_switch_engaged=ks_now,
                    max_spread_ratio=self.max_spread_ratio,
                    max_drift_bps=self.max_drift_bps,
                    drift_check_min_hours=self.drift_check_min_hours,
                )

                if rec.status == "DRIFT_EXCEEDED" and not self.bridge.kill_switch.is_engaged():
                    self.bridge.kill_switch.engage(
                        KillTrigger.RISK_DRAWDOWN,
                        actor="shadow-daemon",
                        detail=f"{sid}: shadow-vs-backtest drift {rec.drift_bps:+.1f}bp > {self.max_drift_bps:.1f}bp",
                    )
                    self.notifier.emit(
                        "P0",
                        "PARITY_DRIFT_KILL",
                        f"Shadow Parity Drift Kill ({sid})",
                        f"drift={rec.drift_bps:+.1f}bp shadow={rec.shadow_return_pct:+.2f}% bt={rec.backtest_return_pct:+.2f}%",
                    )
                    rec = self.parity.evaluate_strategy(
                        cfg,
                        window_start_ms=self._started_ts_ms[sid],
                        window_end_ms=ts_now,
                        shadow_equity_usd=eq_usd,
                        live_half_spreads_bps=hist,
                        plans_created=self._plans_created[sid],
                        plans_blocked=self._plans_blocked[sid],
                        open_positions=open_cnt,
                        bbo_freshness_sec=bbo_freshness_sec,
                        audit_chain_valid=chain_ok_now,
                        kill_switch_engaged=True,
                        max_spread_ratio=self.max_spread_ratio,
                        max_drift_bps=self.max_drift_bps,
                        drift_check_min_hours=self.drift_check_min_hours,
                    )
                records.append(rec)

            overall_status = "KILL_TRIPPED" if self.bridge.kill_switch.is_engaged() else (
                "WARN_WIDE_SPREAD" if any(r.status == "WARN_WIDE_SPREAD" for r in records) else "PARITY_OK"
            )
            return self._finalize_report(
                ts_now,
                trace_id,
                status=overall_status,
                bbo_freshness_sec=bbo_freshness_sec,
                watcher_events=watcher_events,
                strategy_records=records,
                report_path=parity_report_path,
                detail="OK" if overall_status == "PARITY_OK" else overall_status,
            )

    def _step_plugin_and_submit(
        self,
        cfg: StrategyConfig,
        plugin: Any,
        state: dict[str, Any],
        ev: MarketEvent,
        bbo_map: dict[tuple[str, str], BBOTick],
        ts_now: int,
    ) -> None:
        """Invoke plugin.on_event(), submit intents via ExecutionBridge, and rollback position state if BLOCKED."""
        sid = cfg.strategy_id
        dual = _is_dual_venue(cfg)
        pos_keys = ("open_pos", "positions")
        pos_snapshots = {k: copy.deepcopy(state[k]) for k in pos_keys if k in state}
        intents = plugin.on_event(ev, state, cfg)
        if not intents:
            return

        any_blocked = False
        for intent in intents:
            ta = bbo_map.get((cfg.venue_a, intent.symbol))
            tb = bbo_map.get((cfg.venue_b, intent.symbol)) if dual else None
            bbo_by_venue: dict[str, BBOTick] = {}
            if ta is not None:
                bbo_by_venue[cfg.venue_a] = ta
            if tb is not None and dual:
                bbo_by_venue[cfg.venue_b] = tb

            plan = self.bridge.submit_intent(cfg, intent, bbo_by_venue, now_ms=ts_now)
            if plan.status == "PLANNED_SHADOW":
                if not plan.duplicate:
                    self._plans_created[sid] = self._plans_created.get(sid, 0) + 1
                    self._apply_shadow_fill(cfg, intent, ta, tb, fill_ts_ms=ts_now)
            else:
                self._plans_blocked[sid] = self._plans_blocked.get(sid, 0) + 1
                any_blocked = True

        # Restore only position dicts on BLOCKED so updated market telemetry (latest_apr/latest_mids/ohlc_hist) is kept!
        if any_blocked:
            for k in pos_keys:
                if k in pos_snapshots:
                    state[k] = pos_snapshots[k]
                else:
                    state.pop(k, None)

    def _finalize_report(
        self,
        ts_ms: int,
        trace_id: str,
        *,
        status: str,
        bbo_freshness_sec: float,
        watcher_events: list[dict[str, Any]],
        strategy_records: list[ShadowParityRecord],
        report_path: Path | None,
        detail: str,
    ) -> dict[str, Any]:
        bridge_sum = self.bridge.summary()
        payload: dict[str, Any] = {
            "valid": True,
            "run_mode": "SHADOW_DRY_RUN",
            "live_execution_enabled": LIVE_EXECUTION_ENABLED,
            "trace_id": trace_id,
            "ts_ms": int(ts_ms),
            "status": status,
            "detail": detail,
            "bbo_freshness_sec": round(float(bbo_freshness_sec), 2),
            "kill_switch": bridge_sum["kill_switch"],
            "audit_chain_valid": bridge_sum["audit_chain_valid"],
            "audit_chain_length": bridge_sum["audit_chain_length"],
            "audit_head_seq": bridge_sum["audit_head_seq"],
            "watcher_events": watcher_events,
            "strategies": [r.to_dict() for r in strategy_records],
            "watchdog": self.watchdog.stats(),
            "alerts_recent": self.notifier.recent_alerts(limit=15),
        }
        target = report_path or (self.data_dir / "shadow_parity_report.json")
        _atomic_write_json(target, payload)
        return payload

    def run_forever(self, poll_interval_sec: float = 30.0, collect_live: bool = True) -> None:
        """Blocking loop for systemd daemon execution with graceful SIGINT/SIGTERM handling."""
        interval = max(2.0, float(poll_interval_sec))

        def _handle_signal(signum: int, _frame: Any) -> None:
            logger.info("ShadowPortfolioDaemon received signal %s; stopping cleanly...", signum)
            self._stop_event.set()

        prev_int = signal.signal(signal.SIGINT, _handle_signal)
        prev_term = signal.signal(signal.SIGTERM, _handle_signal)
        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                try:
                    summary = self.run_once(collect_live=collect_live)
                    logger.info(
                        "Shadow cycle %s status=%s strategies=%d bbo_age=%.1fs chain=%s kill=%s",
                        summary["trace_id"],
                        summary["status"],
                        len(summary["strategies"]),
                        summary["bbo_freshness_sec"],
                        summary["audit_chain_valid"],
                        summary["kill_switch"]["engaged"],
                    )
                except Exception as exc:
                    logger.exception("Shadow cycle error (fail-closed continue): %s", exc)
                    self.notifier.emit("P1", "SHADOW_CYCLE_ERROR", "Shadow Daemon Cycle Error", str(exc))
                elapsed = time.monotonic() - t0
                wait_sec = max(0.5, interval - elapsed)
                self._stop_event.wait(timeout=wait_sec)
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)
            self.close()

    def close(self) -> None:
        self._stop_event.set()
        if self._owns_watchdog:
            self.watchdog.close()
        self.parity.close()
        self.bridge.close()
        self.store.close()
