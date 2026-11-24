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
from alphaloop.harness.backtester import (
    HOUR_MS,
    TAKER_FEE_BPS,
    _param_int,
    compute_carry_slot_sizing,
)
from alphaloop.plugins import PluginRegistry, default_registry
from alphaloop.plugins.trend_cta_plugin import (
    compute_causal_atr_and_ema,
    compute_cta_entry_notional,
    compute_true_range,
    parse_cta_params,
)

logger = logging.getLogger(__name__)

DAEMON_LIVE_EXECUTION_ENABLED: Final[bool] = False
SHADOW_LIVE_EXECUTION_ENABLED: Final[bool] = False
PARITY_DEPTH_ACTIVE: Final[str] = "ACTIVE_EXECUTION"
PARITY_DEPTH_VACUOUS: Final[str] = "VACUOUS_ZERO_PLANS"
TAKER_FEE_BPS_DEFAULT: Final[float] = 4.5


def _is_finite(val: Any) -> bool:
    return (not isinstance(val, bool)) and isinstance(val, (int, float)) and math.isfinite(float(val))


def _is_dual_venue(cfg: StrategyConfig) -> bool:
    return bool(cfg.venue_b and cfg.venue_b != cfg.venue_a)


def _assumed_total_slip_bps(cfg: StrategyConfig) -> float:
    mult = 2.0 if _is_dual_venue(cfg) else 1.0
    return max(0.1, mult * float(cfg.slippage_per_leg_bps))


def _effective_leg_fee_bps(
    cfg: StrategyConfig,
    leg_fee_bps: float,
    venue: str = "",
    *,
    is_leg_b: bool = False,
    is_exit: bool = False,
) -> float:
    """Venue- and execution-mode-aware leg fee matching backtester.py & trend_cta_plugin.py."""
    mode = str(cfg.execution_mode or "MAKER_MAKER").upper()
    taker = float(TAKER_FEE_BPS.get(venue, TAKER_FEE_BPS_DEFAULT))
    if mode == "TAKER_TAKER":
        return max(float(leg_fee_bps), taker)
    if mode == "MAKER_THEN_TAKER_HEDGE":
        dual = _is_dual_venue(cfg)
        if (dual and is_leg_b) or ((not dual) and is_exit):
            return max(float(leg_fee_bps), taker)
    return float(leg_fee_bps)


def _normalize_int_ts_dict(raw: Any) -> dict[int, float]:
    """Normalize any plain JSON numeric-string timestamp keys to int (preferring native int on collision)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[int, float] = {}
    for k, v in raw.items():
        if isinstance(k, bool):
            continue
        try:
            tk = int(k)
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        if tk not in out or isinstance(k, int):
            out[tk] = fv
    return out


def _normalize_plugin_state(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize hydrated plugin state dicts so int timestamp keys and OHLC tuples never drift after JSON load."""
    if not isinstance(state, dict):
        return {}
    fhist = state.get("funding_hist")
    if isinstance(fhist, dict):
        for fk, fmap in list(fhist.items()):
            fhist[fk] = _normalize_int_ts_dict(fmap)
    hrets = state.get("hourly_log_rets")
    if isinstance(hrets, dict):
        for sym, rmap in list(hrets.items()):
            hrets[sym] = _normalize_int_ts_dict(rmap)
    ohist = state.get("ohlc_hist")
    if isinstance(ohist, dict):
        for sym, bars in list(ohist.items()):
            if isinstance(bars, list):
                ohist[sym] = [
                    (float(x[0]), float(x[1]), float(x[2]))
                    for x in bars
                    if isinstance(x, (list, tuple)) and len(x) == 3
                ]
    return state


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
    parity_depth: str = PARITY_DEPTH_VACUOUS
    non_vacuous_parity: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_prewindow_funding_warmup(
    market_store: MarketDataStore,
    cfg: StrategyConfig,
    window_start_ms: int,
) -> dict[str, Any]:
    """Validate pre-window funding history across required venues (both venue_a and venue_b for dual-venue carry)."""
    if isinstance(window_start_ms, bool) or not isinstance(window_start_ms, int) or window_start_ms < 0:
        raise ValueError("window_start_ms must be a non-negative integer")
    lb_h = max(1, int(cfg.lookback_hours))
    params = cfg.params or {}
    vol_lb_h = _param_int(params, "vol_lookback_hours", 168) if params.get("vol_target_leverage_enabled") else 0
    warm_h = max(lb_h, vol_lb_h) + 24
    warmup_start_ms = max(0, int(window_start_ms) - warm_h * HOUR_MS)
    req_oldest_ms = int(window_start_ms) - lb_h * HOUR_MS

    req_venues = [cfg.venue_a, cfg.venue_b] if _is_dual_venue(cfg) else [cfg.venue_a]
    symbols = list(cfg.symbols or [])

    per_symbol: dict[str, dict[str, Any]] = {}
    warmed_symbols: list[str] = []
    unwarmed_symbols: list[str] = []
    sym_obs_hours_list: list[float] = []

    for sym in symbols:
        venue_details: dict[str, dict[str, Any]] = {}
        venue_obs_hours: list[float] = []
        all_venues_ok = True
        for venue in req_venues:
            bars = market_store.load_funding(venue, sym, start_ms=warmup_start_ms, end_ms=int(window_start_ms))
            valid_bars = [b for b in bars if _is_finite(b.funding_rate)]
            if valid_bars:
                oldest_ts: int | None = min(int(b.ts_ms) for b in valid_bars)
                obs_h = max(0.0, (int(window_start_ms) - oldest_ts) / float(HOUR_MS))
                v_ok = (lb_h <= 1) or (oldest_ts <= req_oldest_ms)
            else:
                oldest_ts = None
                obs_h = 0.0
                v_ok = lb_h <= 1
            venue_obs_hours.append(obs_h)
            if not v_ok:
                all_venues_ok = False
            venue_details[venue] = {
                "bar_count": len(valid_bars),
                "oldest_ts_ms": oldest_ts,
                "observed_hours": round(obs_h, 2),
                "sufficient": bool(v_ok),
            }

        sym_bottleneck_h = min(venue_obs_hours) if venue_obs_hours else 0.0
        sym_obs_hours_list.append(sym_bottleneck_h)
        if all_venues_ok:
            warmed_symbols.append(sym)
        else:
            unwarmed_symbols.append(sym)
        per_symbol[sym] = {
            "sufficient": bool(all_venues_ok),
            "observed_hours": round(sym_bottleneck_h, 2),
            "required_hours": lb_h,
            "coverage_ratio": round(min(1.0, sym_bottleneck_h / float(lb_h)), 4) if lb_h > 0 else 1.0,
            "venues": venue_details,
        }

    sufficient = (lb_h <= 1) or bool(warmed_symbols)
    best_obs_h = max(sym_obs_hours_list) if sym_obs_hours_list else 0.0
    min_obs_h = min(sym_obs_hours_list) if sym_obs_hours_list else 0.0
    cov_ratio = 1.0 if lb_h <= 1 else round(min(1.0, best_obs_h / float(lb_h)), 4)
    sym_cov_ratio = round(len(warmed_symbols) / len(symbols), 4) if symbols else (1.0 if lb_h <= 1 else 0.0)
    reason = "" if sufficient else "INSUFFICIENT_PREWINDOW_WARMUP"

    return {
        "sufficient": bool(sufficient),
        "warmup_ok": bool(sufficient),
        "warm_ok": bool(sufficient),
        "reason": reason,
        "warmup_start_ms": warmup_start_ms,
        "window_start_ms": int(window_start_ms),
        "required_hours": lb_h,
        "vol_lookback_hours": vol_lb_h,
        "observed_hours": round(best_obs_h, 2),
        "min_symbol_observed_hours": round(min_obs_h, 2),
        "coverage_ratio": cov_ratio,
        "symbol_coverage_ratio": sym_cov_ratio,
        "required_venues": req_venues,
        "warmed_symbols": warmed_symbols,
        "unwarmed_symbols": unwarmed_symbols,
        "per_symbol": per_symbol,
    }


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
            if obj and all(isinstance(k, int) and not isinstance(k, bool) for k in obj):
                return {
                    "__int_map__": [
                        [int(k), ShadowParityAuditor._encode_state(v)]
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
            if "__int_map__" in obj and len(obj) == 1:
                return {int(k): ShadowParityAuditor._decode_state(v) for k, v in obj["__int_map__"]}
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
            decoded = self._decode_state(json.loads(str(r["plugin_state_json"] or "{}")))
            out[str(r["strategy_id"])] = {
                "started_ts_ms": int(r["started_ts_ms"]),
                "realized_pnl_usd": float(r["realized_pnl_usd"]),
                "plans_created": int(r["plans_created"]),
                "plans_blocked": int(r["plans_blocked"]),
                "open_positions": json.loads(str(r["open_positions_json"])),
                "plugin_state": _normalize_plugin_state(decoded) if isinstance(decoded, dict) else {},
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

    def validate_prewindow_warmup(
        self,
        cfg: StrategyConfig,
        window_start_ms: int,
    ) -> dict[str, Any]:
        """Return structured pre-window warmup diagnostics for funding_carry and trend_cta parity backtests."""
        pid = str(cfg.plugin_id or "funding_carry").strip().lower()
        if pid == "funding_carry":
            return validate_prewindow_funding_warmup(self.market_store, cfg, int(window_start_ms))

        if pid == "trend_cta":
            cp = parse_cta_params(cfg)
            tf_h = cp["tf_h"]
            tf_ms = cp["tf_ms"]
            req_bars = int(cp["warmup_bars"])
            need_bars = max(req_bars, cp["ema_n"] * 3 if cp["ema_n"] > 0 else 0) + 10
            warmup_start = max(0, int(window_start_ms) - need_bars * tf_ms)
            eval_start_bucket = (int(window_start_ms) // tf_ms) * tf_ms
            symbols = list(cfg.symbols or [])
            per_symbol: dict[str, dict[str, Any]] = {}
            warmed_symbols: list[str] = []
            unwarmed_symbols: list[str] = []
            obs_bars_list: list[int] = []
            for sym in symbols:
                c1h = self.market_store.load_candles(cfg.venue_a, sym, start_ms=warmup_start, end_ms=eval_start_bucket)
                bcounts: dict[int, int] = {}
                for c in c1h:
                    b = (int(c.ts_ms) // tf_ms) * tf_ms
                    if b < eval_start_bucket:
                        bcounts[b] = bcounts.get(b, 0) + 1
                completed_n = sum(1 for cnt in bcounts.values() if cnt >= tf_h)
                obs_bars_list.append(completed_n)
                sym_ok = completed_n >= req_bars
                if sym_ok:
                    warmed_symbols.append(sym)
                else:
                    unwarmed_symbols.append(sym)
                per_symbol[sym] = {
                    "sufficient": bool(sym_ok),
                    "observed_bars": int(completed_n),
                    "required_bars": req_bars,
                    "observed_hours": round(float(completed_n * tf_h), 2),
                    "required_hours": int(req_bars * tf_h),
                    "coverage_ratio": round(min(1.0, completed_n / float(req_bars)), 4) if req_bars > 0 else 1.0,
                }
            sufficient = bool(warmed_symbols) if req_bars > 0 else True
            best_bars = max(obs_bars_list) if obs_bars_list else 0
            min_bars = min(obs_bars_list) if obs_bars_list else 0
            return {
                "sufficient": bool(sufficient),
                "warmup_ok": bool(sufficient),
                "warm_ok": bool(sufficient),
                "reason": "" if sufficient else "INSUFFICIENT_PREWINDOW_WARMUP",
                "warmup_start_ms": warmup_start,
                "window_start_ms": int(window_start_ms),
                "required_bars": req_bars,
                "observed_bars": int(best_bars),
                "required_hours": int(req_bars * tf_h),
                "observed_hours": round(float(best_bars * tf_h), 2),
                "min_symbol_observed_hours": round(float(min_bars * tf_h), 2),
                "coverage_ratio": round(min(1.0, best_bars / float(req_bars)), 4) if req_bars > 0 else 1.0,
                "symbol_coverage_ratio": round(len(warmed_symbols) / len(symbols), 4) if symbols else 0.0,
                "required_venues": [cfg.venue_a],
                "warmed_symbols": warmed_symbols,
                "unwarmed_symbols": unwarmed_symbols,
                "per_symbol": per_symbol,
            }

        return {
            "sufficient": True,
            "warmup_ok": True,
            "warm_ok": True,
            "reason": "",
            "warmup_start_ms": None,
            "window_start_ms": int(window_start_ms),
            "required_hours": 0,
            "observed_hours": 0.0,
            "min_symbol_observed_hours": 0.0,
            "coverage_ratio": 1.0,
            "symbol_coverage_ratio": 1.0,
            "required_venues": [cfg.venue_a],
            "warmed_symbols": list(cfg.symbols or []),
            "unwarmed_symbols": [],
            "per_symbol": {},
        }

    def _compute_prewindow_warmup(
        self,
        cfg: StrategyConfig,
        window_start_ms: int,
    ) -> tuple[int | None, bool, str]:
        """Return (warmup_start_ms, has_sufficient_history, reason) for apples-to-apples parity backtest."""
        res = self.validate_prewindow_warmup(cfg, int(window_start_ms))
        return res["warmup_start_ms"], bool(res["sufficient"]), str(res["reason"])

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
        """Compare live Shadow equity & realized spread against causally pre-warmed same-window backtest."""
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
                warmup_start_ms, warm_ok, warm_reason = self._compute_prewindow_warmup(cfg, int(window_start_ms))
                if not warm_ok:
                    detail_parts.append(warm_reason)
                else:
                    eval_kwargs: dict[str, Any] = {
                        "start_ms": int(window_start_ms),
                        "end_ms": int(window_end_ms),
                    }
                    if warmup_start_ms is not None:
                        eval_kwargs["warmup_start_ms"] = int(warmup_start_ms)
                    try:
                        bt = plugin.evaluate(self.market_store, cfg, **eval_kwargs)
                    except TypeError:
                        eval_kwargs.pop("warmup_start_ms", None)
                        bt = plugin.evaluate(self.market_store, cfg, **eval_kwargs)
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

        plans_created_int = int(plans_created)
        non_vacuous = plans_created_int > 0
        parity_depth = PARITY_DEPTH_ACTIVE if non_vacuous else PARITY_DEPTH_VACUOUS
        if not non_vacuous:
            detail_parts.append(PARITY_DEPTH_VACUOUS)

        snap_id = "PAR-" + sha256_obj({
            "sid": cfg.strategy_id,
            "ver": cfg.version,
            "ts": int(window_end_ms),
            "eq": round(shadow_equity_usd, 4),
            "plans": plans_created_int,
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
            plans_created=plans_created_int,
            plans_blocked=int(plans_blocked),
            open_positions=int(open_positions),
            bbo_freshness_sec=round(float(bbo_freshness_sec), 2),
            audit_chain_valid=bool(audit_chain_valid),
            kill_switch_engaged=bool(kill_switch_engaged),
            status=status,
            detail="; ".join(detail_parts) if detail_parts else "OK",
            parity_depth=parity_depth,
            non_vacuous_parity=non_vacuous,
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
        if LIVE_EXECUTION_ENABLED or DAEMON_LIVE_EXECUTION_ENABLED or SHADOW_LIVE_EXECUTION_ENABLED:
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
        self._warmed_strategies: set[tuple[str, str]] = set()
        self._unwarmed_funding_keys: dict[tuple[str, str], set[tuple[str, str]]] = {}
        self._last_vol_refresh_hour: dict[tuple[str, str], int] = {}

        # Hydrate persisted shadow runtime state across restarts (Red-Team HIGH #8)
        for sid, saved in self.parity.load_all_runtime_states().items():
            self._started_ts_ms[sid] = saved["started_ts_ms"]
            self._realized_pnl[sid] = saved["realized_pnl_usd"]
            self._plans_created[sid] = saved["plans_created"]
            self._plans_blocked[sid] = saved["plans_blocked"]
            self._open_positions[sid] = saved["open_positions"]
            if isinstance(saved.get("plugin_state"), dict):
                self._plugin_states[sid] = _normalize_plugin_state(saved["plugin_state"])
            for fkey_s, fts in (saved.get("last_funding_ts") or {}).items():
                if ":" in str(fkey_s):
                    v, sym = str(fkey_s).split(":", 1)
                    self._last_delivered_funding_ts[(sid, v, sym)] = int(fts)

    def _warmup_plugin_state_from_candles(
        self,
        cfg: StrategyConfig,
        cfg_sha: str,
        state: dict[str, Any],
        ts_now: int,
    ) -> None:
        """Warm up TrendCTA (ohlc_hist/TR/EMA/missed buckets/active_bucket) & FundingCarry (funding_hist/vol rets)."""
        sid = cfg.strategy_id
        warm_key = (sid, cfg_sha)
        plugin_id = str(cfg.plugin_id or "funding_carry")
        params = cfg.params or {}
        hour_floor = (int(ts_now) // HOUR_MS) * HOUR_MS

        # Retry ONLY still-unwarmed cold-start funding keys (zero 96h lockout if funding DB fills after daemon boot!)
        if warm_key in self._warmed_strategies:
            if plugin_id == "funding_carry":
                pending_keys = self._unwarmed_funding_keys.get(warm_key)
                if pending_keys:
                    lb_h = max(1, int(cfg.lookback_hours))
                    f_lookback_ms = (lb_h + 24) * HOUR_MS
                    funding_hist: dict[tuple[str, str], dict[int, float]] = state.setdefault("funding_hist", {})
                    first_funding_ts: dict[str, int] = state.setdefault("first_funding_ts", {})
                    for fv, sym in list(pending_keys):
                        f_key = (sid, fv, sym)
                        fbars = self.store.load_funding(fv, sym, start_ms=ts_now - f_lookback_ms, end_ms=ts_now)
                        max_warm_ts = -1
                        fmap = funding_hist.setdefault((fv, sym), {})
                        for fb in fbars:
                            if _is_finite(fb.funding_rate):
                                fts = int(fb.ts_ms)
                                fmap[fts] = float(fb.funding_rate)
                                if fts > max_warm_ts:
                                    max_warm_ts = fts
                                prev_f = first_funding_ts.get(sym)
                                if prev_f is None or fts < prev_f:
                                    first_funding_ts[sym] = fts
                        if max_warm_ts >= 0:
                            self._last_delivered_funding_ts[f_key] = max_warm_ts
                            pending_keys.discard((fv, sym))
            return

        self._warmed_strategies.add(warm_key)

        if plugin_id == "trend_cta":
            cp = parse_cta_params(cfg)
            tf_h = cp["tf_h"]
            tf_ms = cp["tf_ms"]
            entry_n = cp["entry_n"]
            exit_n = cp["exit_n"]
            atr_n = cp["atr_n"]
            ema_n = cp["ema_n"]
            cap_n = max(entry_n, exit_n, atr_n) + 5
            alpha = (2.0 / (ema_n + 1.0)) if ema_n > 0 else 0.0
            current_open_bucket = (int(ts_now) // tf_ms) * tf_ms
            need_bars = max(cap_n, ema_n * 3 if ema_n > 0 else 0) + 10
            lookback_ms = max(tf_h * need_bars * HOUR_MS, 30 * 86_400_000)

            ohlc_hist: dict[str, list[tuple[float, float, float]]] = state.setdefault("ohlc_hist", {})
            tr_hist: dict[str, list[float]] = state.setdefault("tr_hist", {})
            ema_state: dict[str, float] = state.setdefault("ema_state", {})
            bars_seen: dict[str, int] = state.setdefault("bars_seen", {})
            last_warmed_bts: dict[str, int] = state.setdefault("last_warmed_bucket_ts", {})
            active_bucket: dict[str, dict[str, Any]] = state.setdefault("active_bucket", {})

            for sym in cfg.symbols:
                # ALWAYS evict any persisted stale active_bucket from an earlier bucket BEFORE seeding
                # (prevents double-committing an old bucket out of order when cur_open_closed_1h is empty!)
                cur_ab = active_bucket.get(sym)
                if cur_ab is not None and int(cur_ab.get("bts", -1)) < current_open_bucket:
                    active_bucket.pop(sym, None)
                    cur_ab = None

                candles = self.store.load_candles(cfg.venue_a, sym, start_ms=ts_now - lookback_ms, end_ms=ts_now)
                if not candles:
                    continue
                completed_buckets: dict[int, list[Any]] = {}
                cur_open_closed_1h: list[Any] = []
                for c in candles:
                    c_ts = int(c.ts_ms)
                    b = (c_ts // tf_ms) * tf_ms
                    if b < current_open_bucket:
                        completed_buckets.setdefault(b, []).append(c)
                    elif b == current_open_bucket and c_ts < hour_floor:
                        # Strictly closed 1h bars inside current_open_bucket (excludes open hour_floor candle!)
                        cur_open_closed_1h.append(c)

                completed_ordered: list[tuple[int, tuple[float, float, float]]] = []
                for bts in sorted(completed_buckets.keys()):
                    grp = sorted(completed_buckets[bts], key=lambda x: x.ts_ms)
                    if len(grp) < tf_h:
                        continue
                    completed_ordered.append((
                        bts,
                        (
                            max(float(x.high) for x in grp),
                            min(float(x.low) for x in grp),
                            float(grp[-1].close),
                        ),
                    ))

                existing_series = ohlc_hist.get(sym, [])
                if not existing_series or len(existing_series) < entry_n:
                    if completed_ordered:
                        comp_bars = [bar for _, bar in completed_ordered]
                        ohlc_hist[sym] = comp_bars[-cap_n:]
                        bars_seen[sym] = len(comp_bars)
                        trs: list[float] = []
                        for idx_i, (b_hi, b_lo, _b_cl) in enumerate(comp_bars):
                            pc_i = comp_bars[idx_i - 1][2] if idx_i > 0 else None
                            trs.append(compute_true_range(b_hi, b_lo, pc_i))
                        tr_hist[sym] = trs[-(atr_n + 5) :]
                        _, emas = compute_causal_atr_and_ema(comp_bars, atr_n, ema_n)
                        ema_state[sym] = emas[-1]
                        last_warmed_bts[sym] = completed_ordered[-1][0]
                else:
                    # Restart catch-up: reconcile any completed buckets missed during downtime
                    prev_bts = int(last_warmed_bts.get(sym, -1))
                    missed = [(bts, bar) for bts, bar in completed_ordered if bts > prev_bts]
                    if missed:
                        series = ohlc_hist.setdefault(sym, [])
                        tr_s = tr_hist.setdefault(sym, [])
                        for bts, (b_hi, b_lo, b_cl) in missed:
                            prev_c = series[-1][2] if series else None
                            tr_s.append(compute_true_range(b_hi, b_lo, prev_c))
                            if len(tr_s) > atr_n + 5:
                                del tr_s[: len(tr_s) - (atr_n + 5)]
                            if sym not in ema_state:
                                ema_state[sym] = b_cl
                            elif ema_n > 0:
                                ema_state[sym] = alpha * b_cl + (1.0 - alpha) * float(ema_state[sym])
                            else:
                                ema_state[sym] = b_cl
                            series.append((b_hi, b_lo, b_cl))
                            if len(series) > cap_n:
                                del series[: len(series) - cap_n]
                            bars_seen[sym] = int(bars_seen.get(sym, 0)) + 1
                            last_warmed_bts[sym] = bts

                # Seed current_open_bucket from closed 1h bars in [current_open_bucket, hour_floor) if missing
                if cur_open_closed_1h and cur_ab is None:
                    cur_open_closed_1h.sort(key=lambda x: x.ts_ms)
                    expected_h = max(0, (hour_floor - current_open_bucket) // HOUR_MS)
                    is_partial = (int(cur_open_closed_1h[0].ts_ms) != current_open_bucket) or (
                        len(cur_open_closed_1h) < expected_h
                    )
                    active_bucket[sym] = {
                        "bts": float(current_open_bucket),
                        "high": max(float(x.high) for x in cur_open_closed_1h),
                        "low": min(float(x.low) for x in cur_open_closed_1h),
                        "close": float(cur_open_closed_1h[-1].close),
                        "partial": is_partial,
                    }
            return

        if plugin_id == "funding_carry":
            lb_h = max(1, int(cfg.lookback_hours))
            vol_lb_h = max(24, min(336, _param_int(params, "vol_lookback_hours", 168)))
            funding_hist = state.setdefault("funding_hist", {})
            first_funding_ts = state.setdefault("first_funding_ts", {})
            dual = _is_dual_venue(cfg)
            f_venues = [cfg.venue_a, cfg.venue_b] if dual else [cfg.venue_a]
            f_lookback_ms = (lb_h + 24) * HOUR_MS
            unwarmed_set = self._unwarmed_funding_keys.setdefault(warm_key, set())
            for sym in cfg.symbols:
                for fv in f_venues:
                    if not fv:
                        continue
                    fmap = funding_hist.setdefault((fv, sym), {})
                    f_key = (sid, fv, sym)
                    has_persisted_cursor = f_key in self._last_delivered_funding_ts
                    # On fresh start, warm ALL historical funding strictly <= ts_now silently and pin cursor
                    # ONLY when rows exist (if DB is still cold/empty, retry silently on next cycle — zero 96h lockout!)
                    cutoff_end = self._last_delivered_funding_ts[f_key] if has_persisted_cursor else ts_now
                    fbars = self.store.load_funding(fv, sym, start_ms=ts_now - f_lookback_ms, end_ms=cutoff_end)
                    max_warm_ts = -1
                    for fb in fbars:
                        if _is_finite(fb.funding_rate):
                            fts = int(fb.ts_ms)
                            fmap[fts] = float(fb.funding_rate)
                            if fts > max_warm_ts:
                                max_warm_ts = fts
                            prev_f = first_funding_ts.get(sym)
                            if prev_f is None or fts < prev_f:
                                first_funding_ts[sym] = fts
                    if not has_persisted_cursor:
                        if max_warm_ts >= 0:
                            self._last_delivered_funding_ts[f_key] = max_warm_ts
                            unwarmed_set.discard((fv, sym))
                        else:
                            unwarmed_set.add((fv, sym))
            self._refresh_carry_hourly_log_rets(cfg, state, ts_now, lookback_h=vol_lb_h + 24)
            self._last_vol_refresh_hour[warm_key] = hour_floor

    def _refresh_carry_hourly_log_rets(
        self,
        cfg: StrategyConfig,
        state: dict[str, Any],
        ts_now: int,
        *,
        lookback_h: int = 192,
    ) -> None:
        """Refresh strictly closed 1h candle log-returns (ts <= hour_floor - 1h) for causal vol-tiered leverage."""
        if str(cfg.plugin_id or "funding_carry") != "funding_carry":
            return
        params = cfg.params or {}
        if not bool(params.get("vol_target_leverage_enabled", False)):
            return
        vol_lb_h = max(24, min(336, _param_int(params, "vol_lookback_hours", 168)))
        hour_floor = (int(ts_now) // HOUR_MS) * HOUR_MS
        last_closed_bar_ts = hour_floor - HOUR_MS
        if last_closed_bar_ts <= 0:
            return
        start_ms = last_closed_bar_ts - max(lookback_h, vol_lb_h + 24) * HOUR_MS
        hourly_log_rets: dict[str, dict[int, float]] = state.setdefault("hourly_log_rets", {})
        dual = _is_dual_venue(cfg)
        for sym in cfg.symbols:
            rmap = hourly_log_rets.setdefault(sym, {})
            if dual and cfg.venue_b:
                for c in self.store.load_candles(cfg.venue_b, sym, start_ms=start_ms, end_ms=last_closed_bar_ts):
                    if c.ts_ms <= last_closed_bar_ts and c.open > 0 and c.close > 0:
                        rmap[int(c.ts_ms)] = math.log(float(c.close) / float(c.open))
            for c in self.store.load_candles(cfg.venue_a, sym, start_ms=start_ms, end_ms=last_closed_bar_ts):
                if c.ts_ms <= last_closed_bar_ts and c.open > 0 and c.close > 0:
                    rmap[int(c.ts_ms)] = math.log(float(c.close) / float(c.open))
            cutoff_r = last_closed_bar_ts - (vol_lb_h + 48) * HOUR_MS
            if len(rmap) > vol_lb_h + 72:
                for old_t in [t for t in rmap if int(t) < cutoff_r]:
                    rmap.pop(old_t, None)

    def _load_latest_bbo_map(self, symbols: list[str], venues: set[str], now_ms: int) -> dict[tuple[str, str], BBOTick]:
        lookback_ms = max(int(self.max_bbo_stale_sec * 1000.0 * 4), 3_600_000)
        start_ms = max(0, int(now_ms) - lookback_ms)
        merged: dict[tuple[str, str], BBOTick] = {}
        for venue in sorted(venues):
            if not venue:
                continue
            for sym in symbols:
                latest_tick = self.store.load_latest_bbo_tick(
                    venue, sym, min_ts_ms=start_ms, max_ts_ms=now_ms + 30_000
                )
                if latest_tick is not None:
                    merged[(venue, sym)] = latest_tick
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
        """Maintain realistic paper positions, venue-accurate fees, liquidation penalties, and funding accrual."""
        sid = cfg.strategy_id
        positions = self._open_positions.setdefault(sid, {})
        realized = self._realized_pnl.setdefault(sid, 0.0)
        sym = intent.symbol
        action = str(intent.action)
        reason = str(intent.reason or "")
        notional = float(intent.notional_usd)
        fee_a_entry = _effective_leg_fee_bps(cfg, cfg.fee_a_bps, cfg.venue_a, is_leg_b=False, is_exit=False) / 10_000.0
        fee_b_entry = _effective_leg_fee_bps(cfg, cfg.fee_b_bps, cfg.venue_b, is_leg_b=True, is_exit=False) / 10_000.0
        fee_a_exit = _effective_leg_fee_bps(cfg, cfg.fee_a_bps, cfg.venue_a, is_leg_b=False, is_exit=True) / 10_000.0
        fee_b_exit = _effective_leg_fee_bps(cfg, cfg.fee_b_bps, cfg.venue_b, is_leg_b=True, is_exit=True) / 10_000.0
        dual_fee_entry = fee_a_entry + fee_b_entry
        dual_fee_exit = fee_a_exit + fee_b_exit
        eff_weight = min(float(cfg.per_position_weight), 1.0 / max(1, int(cfg.max_positions)))
        slot_cap = float(cfg.capital_usd) * eff_weight

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
                "slot_margin": slot_cap,
                "entry_ts_ms": int(fill_ts_ms),
                "entry_a": entry_px,
                "last_mark_a": bbo_a.mid,
                "accrued_funding": 0.0,
            }
            self._realized_pnl[sid] = realized - notional * fee_a_entry
            return

        if action in ("OPEN_SHORT_A_LONG_B", "OPEN_LONG_A_SHORT_B"):
            if bbo_a is None or bbo_b is None or sym in positions:
                return
            entry_a = bbo_a.bid if action == "OPEN_SHORT_A_LONG_B" else bbo_a.ask
            entry_b = bbo_b.ask if action == "OPEN_SHORT_A_LONG_B" else bbo_b.bid
            if entry_a <= 0 or entry_b <= 0:
                return
            _, carry_slot_margin, _ = compute_carry_slot_sizing(cfg, cfg.leverage)
            positions[sym] = {
                "single_venue": False,
                "action": action,
                "notional": notional,
                "slot_margin": carry_slot_margin,
                "entry_ts_ms": int(fill_ts_ms),
                "entry_a": entry_a,
                "entry_b": entry_b,
                "last_mark_a": bbo_a.mid,
                "last_mark_b": bbo_b.mid,
                "accrued_funding": 0.0,
            }
            self._realized_pnl[sid] = realized - notional * dual_fee_entry
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
                exit_fee_a = (
                    max(fee_a_exit, TAKER_FEE_BPS.get(cfg.venue_a, TAKER_FEE_BPS_DEFAULT) / 10_000.0)
                    if reason in ("CHANDELIER_STOP", "MARGIN_LIQUIDATION")
                    else fee_a_exit
                )
                extra_pen = 0.0
                if reason == "MARGIN_LIQUIDATION":
                    pos_slot_m = float(pos.get("slot_margin", slot_cap))
                    extra_pen = max(pos_slot_m * 0.10, float(pos["notional"]) * 0.02)
                self._realized_pnl[sid] = realized + gross + accrued_funding - pos["notional"] * exit_fee_a - extra_pen
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
                exit_cost = dual_fee_exit
                extra_pen = 0.0
                if reason == "MARGIN_LIQUIDATION":
                    _, default_cm, _ = compute_carry_slot_sizing(cfg, cfg.leverage)
                    pos_slot_m = float(pos.get("slot_margin", default_cm))
                    extra_pen = max(pos_slot_m * 0.90, float(pos["notional"]) * 0.05)
                elif reason == "MARGIN_PROTECT_EXIT":
                    taker_a = max(fee_a_exit, TAKER_FEE_BPS.get(cfg.venue_a, TAKER_FEE_BPS_DEFAULT) / 10_000.0)
                    taker_b = max(fee_b_exit, TAKER_FEE_BPS.get(cfg.venue_b, TAKER_FEE_BPS_DEFAULT) / 10_000.0)
                    raw_ps = (cfg.params or {}).get("protective_exit_slip_bps", 5.0)
                    storm_slip_bps = max(5.0, float(raw_ps) if _is_finite(raw_ps) else 5.0)
                    exit_cost = taker_a + taker_b + (2.0 * storm_slip_bps) / 10_000.0
                self._realized_pnl[sid] = realized + pnl_a + pnl_b + accrued_funding - pos["notional"] * exit_cost - extra_pen

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
        if LIVE_EXECUTION_ENABLED or DAEMON_LIVE_EXECUTION_ENABLED or SHADOW_LIVE_EXECUTION_ENABLED:
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

            # Compute required venues & full symbol union BEFORE collect_live so collector covers all 24 symbols!
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
            sorted_syms_needed = sorted(symbols_needed)
            if self._collector.symbols != sorted_syms_needed:
                self._collector.symbols = sorted_syms_needed

            # 3) Gather live BBO ticks (injected, live-collected, or latest stored)
            if collect_live:
                collected, _errs = self._collector.capture_round()
                if collected:
                    self.store.upsert_bbo_ticks(collected)
                    live_ticks = list(live_ticks or []) + collected

            if live_ticks:
                self.store.upsert_bbo_ticks(live_ticks)
                for t in live_ticks:
                    if (
                        not isinstance(t.bid, bool)
                        and not isinstance(t.ask, bool)
                        and isinstance(t.bid, (int, float))
                        and isinstance(t.ask, (int, float))
                        and math.isfinite(float(t.bid))
                        and math.isfinite(float(t.ask))
                        and t.bid > 0
                        and t.ask >= t.bid
                    ):
                        prev_c = self._latest_bbo_cache.get((t.venue, t.symbol))
                        if prev_c is None or t.ts_ms >= prev_c.ts_ms:
                            self._latest_bbo_cache[(t.venue, t.symbol)] = t

            bbo_map = self._load_latest_bbo_map(sorted_syms_needed, venues_needed, ts_now)
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

            # 4) Step each active SHADOW_LIVE strategy with per-strategy exception isolation
            records: list[ShadowParityRecord] = []
            hour_floor = (int(ts_now) // HOUR_MS) * HOUR_MS
            for sid, cfg in sorted(shadow_cfgs.items()):
                try:
                    loaded_entry = self.watcher.get_loaded(sid)
                    cfg_sha = loaded_entry[1] if (isinstance(loaded_entry, tuple) and len(loaded_entry) == 2) else sha256_obj(cfg.to_dict())

                    self._started_ts_ms.setdefault(sid, ts_now)
                    self._plans_created.setdefault(sid, 0)
                    self._plans_blocked.setdefault(sid, 0)
                    self._spread_history.setdefault(sid, [])
                    state = self._plugin_states.setdefault(sid, {})
                    plugin = self.registry.resolve_for_config(cfg)
                    self._warmup_plugin_state_from_candles(cfg, cfg_sha, state, ts_now)

                    # Refresh closed 1h log-returns whenever hour_floor advances (keeps 7d vol leverage live!)
                    warm_key = (sid, cfg_sha)
                    if str(cfg.plugin_id or "funding_carry") == "funding_carry":
                        if self._last_vol_refresh_hour.get(warm_key) != hour_floor:
                            vol_lb_h = max(24, min(336, _param_int(cfg.params or {}, "vol_lookback_hours", 168)))
                            self._refresh_carry_hourly_log_rets(cfg, state, ts_now, lookback_h=vol_lb_h + 24)
                            self._last_vol_refresh_hour[warm_key] = hour_floor

                    step_half_spreads = self._step_strategy_portfolio_cycle(
                        cfg, plugin, state, bbo_map, ts_now, hour_floor
                    )

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
                except Exception as strat_exc:
                    logger.exception("Isolated strategy error for %s (continuing sibling strategies): %s", sid, strat_exc)
                    self.notifier.emit(
                        "P1",
                        "SHADOW_STRATEGY_ERROR",
                        f"Shadow Strategy Execution Error ({sid})",
                        f"trace={trace_id} strategy={sid} error={type(strat_exc).__name__}: {strat_exc}",
                    )

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

    def _submit_intent_to_bridge(
        self,
        cfg: StrategyConfig,
        intent: OrderIntent,
        bbo_map: dict[tuple[str, str], BBOTick],
        ts_now: int,
    ) -> tuple[str, bool]:
        """Submit one OrderIntent to ExecutionBridge & apply shadow fill. Returns (plan.status, plan.duplicate)."""
        sid = cfg.strategy_id
        dual = _is_dual_venue(cfg)
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
        return plan.status, bool(plan.duplicate)

    def _step_strategy_portfolio_cycle(
        self,
        cfg: StrategyConfig,
        plugin: Any,
        state: dict[str, Any],
        bbo_map: dict[tuple[str, str], BBOTick],
        ts_now: int,
        hour_floor: int,
    ) -> list[float]:
        """Deliver FUNDING-before-BBO with Pass-1 exit-before-entry barrier and Pass-2 edge-ranked entries."""
        sid = cfg.strategy_id
        dual = _is_dual_venue(cfg)
        plugin_id = str(cfg.plugin_id or "funding_carry")
        step_half_spreads: list[float] = []

        # A) Gather new FUNDING bars (silently catch up missed historical bars < hour_floor; group live bars by ts)
        funding_venues = [cfg.venue_a, cfg.venue_b] if dual else [cfg.venue_a]
        live_funding_events: dict[str, list[MarketEvent]] = {sym: [] for sym in cfg.symbols}
        f_hist_state = state.setdefault("funding_hist", {}) if plugin_id == "funding_carry" else None
        first_f_state = state.setdefault("first_funding_ts", {}) if plugin_id == "funding_carry" else None

        for sym in cfg.symbols:
            pending_bars: list[tuple[int, str, Any]] = []
            for fvenue in funding_venues:
                if not fvenue:
                    continue
                f_key = (sid, fvenue, sym)
                if plugin_id == "funding_carry" and f_key not in self._last_delivered_funding_ts:
                    # Cold-start empty funding DB: wait for _warmup_plugin_state_from_candles retry
                    # so historical rows warm funding_hist silently without pinning cursor to ts_now.
                    continue
                last_f_ts = self._last_delivered_funding_ts.get(f_key, self._started_ts_ms.get(sid, ts_now))
                for fb in self.store.load_funding(fvenue, sym, start_ms=last_f_ts + 1, end_ms=ts_now):
                    if fb.ts_ms > self._last_delivered_funding_ts.get(f_key, -1):
                        pending_bars.append((int(fb.ts_ms), fvenue, fb))
            pending_bars.sort(key=lambda item: (item[0], item[1]))

            live_grouped: dict[int, list[tuple[str, Any]]] = {}
            for fb_ts, fvenue, fb in pending_bars:
                f_key = (sid, fvenue, sym)
                if fb_ts <= self._last_delivered_funding_ts.get(f_key, -1):
                    continue
                self._last_delivered_funding_ts[f_key] = fb_ts
                self._accrue_funding_on_open_positions(cfg, fvenue, sym, fb.funding_rate, fb_ts)
                if fb_ts < hour_floor:
                    # Missed historical bar after downtime: ingest into funding_hist silently (no phantom historical orders)
                    if f_hist_state is not None and _is_finite(fb.funding_rate):
                        fmap = f_hist_state.setdefault((fvenue, sym), {})
                        fmap[fb_ts] = float(fb.funding_rate)
                        if first_f_state is not None:
                            prev_f = first_f_state.get(sym)
                            if prev_f is None or fb_ts < prev_f:
                                first_f_state[sym] = fb_ts
                else:
                    live_grouped.setdefault(fb_ts, []).append((fvenue, fb))

            for fb_ts in sorted(live_grouped.keys()):
                group = live_grouped[fb_ts]
                for idx_g, (fvenue, fb) in enumerate(group):
                    live_funding_events[sym].append(
                        MarketEvent(
                            kind="FUNDING",
                            ts_ms=fb_ts,
                            venue=fvenue,
                            symbol=sym,
                            payload={
                                "funding_rate": fb.funding_rate,
                                "interval_hours": fb.interval_hours,
                                "apr_pct": fb.apr_pct,
                                "defer_decision": idx_g < (len(group) - 1),
                            },
                        )
                    )

        # B) Gather new BBO ticks & spread telemetry
        live_bbo_events: dict[str, list[MarketEvent]] = {sym: [] for sym in cfg.symbols}
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
                live_bbo_events[sym].append(
                    MarketEvent(
                        kind="BBO",
                        ts_ms=tick.ts_ms,
                        venue=tick.venue,
                        symbol=sym,
                        payload={"bid": tick.bid, "ask": tick.ask, "mid": tick.mid, "spread_bps": tick.spread_bps},
                    )
                )

        # Custom plugins not using open_pos fallback directly to standard single-step submission (FUNDING before BBO)
        if plugin_id not in ("funding_carry", "trend_cta"):
            for sym in cfg.symbols:
                for ev in live_funding_events[sym] + live_bbo_events[sym]:
                    self._step_plugin_and_submit(cfg, plugin, state, ev, bbo_map, ts_now)
            return step_half_spreads

        pos_keys = ("open_pos", "positions", "pending_actions", "cooldown_until")
        open_pos_map: dict[str, Any] = state.setdefault("open_pos", {})
        open_syms = [
            sym
            for sym in cfg.symbols
            if sym in open_pos_map or sym in self._open_positions.get(sid, {})
        ]
        open_sym_set = set(open_syms)
        reversal_candidates: dict[str, tuple[OrderIntent, dict[str, Any] | None]] = {}

        # C) PASS 1 — Open positions FIRST (exits, stops, reversals) freeing slots & gross leverage before any entries!
        for sym in open_syms:
            if plugin_id == "funding_carry" and len(live_bbo_events[sym]) > 1:
                pass1_bbo_evs = [
                    MarketEvent(
                        kind=ev.kind,
                        ts_ms=ev.ts_ms,
                        venue=ev.venue,
                        symbol=ev.symbol,
                        payload={**ev.payload, "defer_decision": idx_b < (len(live_bbo_events[sym]) - 1)},
                    )
                    for idx_b, ev in enumerate(live_bbo_events[sym])
                ]
            else:
                pass1_bbo_evs = live_bbo_events[sym]
            for ev in live_funding_events[sym] + pass1_bbo_evs:
                pos_snapshots = {k: copy.deepcopy(state[k]) for k in pos_keys if k in state}
                intents = plugin.on_event(ev, state, cfg)
                if not intents:
                    continue

                close_intents = [i for i in intents if str(i.action) in ("CLOSE", "CLOSE_PAIR")]
                open_intents = [i for i in intents if str(i.action).startswith("OPEN_")]
                close_blocked = False
                for c_int in close_intents:
                    c_status, c_dup = self._submit_intent_to_bridge(cfg, c_int, bbo_map, ts_now)
                    if c_status != "PLANNED_SHADOW":
                        close_blocked = True
                        break
                    if c_dup and sym in self._open_positions.get(sid, {}):
                        close_blocked = True
                        break

                if close_blocked:
                    # Atomic rollback: if CLOSE is BLOCKED, abort any paired reversal OPEN_* too!
                    for k in pos_keys:
                        if k in pos_snapshots:
                            state[k] = pos_snapshots[k]
                        else:
                            state.pop(k, None)
                    continue

                if open_intents:
                    # Queue reversal OPEN_* into Pass 2's cross-symbol edge-ranked candidate pool
                    rev_int = open_intents[-1]
                    rev_pos = copy.deepcopy(state.get("open_pos", {}).pop(sym, None))
                    reversal_candidates[sym] = (rev_int, rev_pos)

        # D) PASS 2 — Flat symbols telemetry & edge-ranked candidate collection (no alphabetical slot stealing!)
        base_open = copy.deepcopy(state.get("open_pos", {}))
        entry_candidates: dict[str, tuple[OrderIntent, dict[str, Any] | None]] = dict(reversal_candidates)
        for sym in cfg.symbols:
            if sym in open_sym_set:
                continue
            sym_events = live_funding_events[sym] + live_bbo_events[sym]
            if not sym_events:
                continue
            if plugin_id == "funding_carry" and len(sym_events) > 1:
                pass2_events = [
                    MarketEvent(
                        kind=ev.kind,
                        ts_ms=ev.ts_ms,
                        venue=ev.venue,
                        symbol=ev.symbol,
                        payload={**ev.payload, "defer_decision": idx_e < (len(sym_events) - 1)},
                    )
                    for idx_e, ev in enumerate(sym_events)
                ]
            else:
                pass2_events = sym_events
            for ev in pass2_events:
                # Evaluate every flat symbol against the same post-Pass-1 open_pos snapshot
                state["open_pos"] = copy.deepcopy(base_open)
                intents = plugin.on_event(ev, state, cfg)
                for intent in intents:
                    if str(intent.action).startswith("OPEN_"):
                        cand_pos = copy.deepcopy(state.get("open_pos", {}).get(sym))
                        prev = entry_candidates.get(sym)
                        if prev is None or float(intent.expected_edge_bps) >= float(prev[0].expected_edge_bps):
                            entry_candidates[sym] = (intent, cand_pos)
            state["open_pos"] = copy.deepcopy(base_open)

        state["open_pos"] = base_open

        # Rank ALL candidate entries strictly by (-expected_edge_bps, symbol) matching backtester.py & evaluate()
        ranked = sorted(
            entry_candidates.items(),
            key=lambda kv: (-float(kv[1][0].expected_edge_bps), kv[0]),
        )
        cta_atr_n = parse_cta_params(cfg)["atr_n"] if plugin_id == "trend_cta" else 14
        for sym, (intent, cand_pos) in ranked:
            cur_open = state.setdefault("open_pos", {})
            if sym in cur_open or len(cur_open) >= int(cfg.max_positions):
                continue

            submit_intent = intent
            commit_pos = copy.deepcopy(cand_pos) if isinstance(cand_pos, dict) else None
            if plugin_id == "trend_cta" and commit_pos is not None:
                cur_gross = sum(float(p_item.get("notional", 0.0)) for p_item in cur_open.values())
                tr_s = state.get("tr_hist", {}).get(sym, [])
                atr_prev = (sum(tr_s[-cta_atr_n:]) / float(len(tr_s[-cta_atr_n:]))) if tr_s else 0.0
                recomp_notional, slot_cap = compute_cta_entry_notional(
                    cfg,
                    entry_px=float(commit_pos.get("entry_px", 0.0)),
                    atr_prev=atr_prev,
                    cur_gross_notional=cur_gross,
                )
                if recomp_notional < 10.0:
                    continue
                commit_pos["notional"] = recomp_notional
                commit_pos["slot_margin"] = slot_cap
                submit_intent = OrderIntent(
                    intent.strategy_id,
                    intent.symbol,
                    intent.action,
                    recomp_notional,
                    intent.expected_edge_bps,
                    intent.reason,
                    intent.ts_ms,
                )

            status, is_dup = self._submit_intent_to_bridge(cfg, submit_intent, bbo_map, ts_now)
            if status == "PLANNED_SHADOW":
                if not is_dup:
                    if commit_pos is not None:
                        cur_open[sym] = commit_pos
                else:
                    # Reconcile duplicate intent against actual shadow ledger so state["open_pos"] never drifts
                    if sym in self._open_positions.get(sid, {}) and commit_pos is not None:
                        cur_open[sym] = commit_pos
                    else:
                        cur_open.pop(sym, None)

        return step_half_spreads

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
        pos_keys = ("open_pos", "positions", "pending_actions", "cooldown_until")
        pos_snapshots = {k: copy.deepcopy(state[k]) for k in pos_keys if k in state}
        intents = plugin.on_event(ev, state, cfg)
        if not intents:
            return

        any_blocked = False
        any_planned = False
        blocked_open_syms: list[str] = []
        for intent in intents:
            status, is_dup = self._submit_intent_to_bridge(cfg, intent, bbo_map, ts_now)
            if status == "PLANNED_SHADOW":
                any_planned = True
                if is_dup and str(intent.action).startswith("OPEN_") and intent.symbol not in self._open_positions.get(sid, {}):
                    for k in ("open_pos", "positions"):
                        if isinstance(state.get(k), dict):
                            state[k].pop(intent.symbol, None)
            else:
                any_blocked = True
                if str(intent.action).startswith("OPEN_"):
                    blocked_open_syms.append(intent.symbol)
                elif str(intent.action) in ("CLOSE", "CLOSE_PAIR"):
                    # Abort any subsequent reversal OPEN_* if CLOSE was BLOCKED
                    break

        # Restore position/action state on BLOCKED while keeping updated market telemetry (ohlc_hist/funding_hist)!
        if any_blocked:
            if not any_planned:
                for k in pos_keys:
                    if k in pos_snapshots:
                        state[k] = pos_snapshots[k]
                    else:
                        state.pop(k, None)
            else:
                # Partial execution (e.g., CLOSE succeeded, reversal OPEN blocked): evict only blocked opens
                for sym_b in blocked_open_syms:
                    if sym_b not in self._open_positions.get(sid, {}):
                        for k in ("open_pos", "positions"):
                            if isinstance(state.get(k), dict):
                                state[k].pop(sym_b, None)

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
        total_plans_created = sum(max(0, int(r.plans_created)) for r in strategy_records)
        non_vacuous_agg = total_plans_created > 0
        parity_depth_agg = PARITY_DEPTH_ACTIVE if non_vacuous_agg else PARITY_DEPTH_VACUOUS
        all_strategies_non_vacuous = bool(strategy_records) and all(
            bool(r.non_vacuous_parity) for r in strategy_records
        )
        vacuous_strategies = [r.strategy_id for r in strategy_records if not r.non_vacuous_parity]
        payload: dict[str, Any] = {
            "valid": True,
            "run_mode": "SHADOW_DRY_RUN",
            "live_execution_enabled": LIVE_EXECUTION_ENABLED,
            "trace_id": trace_id,
            "ts_ms": int(ts_ms),
            "status": status,
            "detail": detail,
            "plans_created": total_plans_created,
            "parity_depth": parity_depth_agg,
            "non_vacuous_parity": non_vacuous_agg,
            "all_strategies_non_vacuous": all_strategies_non_vacuous,
            "vacuous_strategies": vacuous_strategies,
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
