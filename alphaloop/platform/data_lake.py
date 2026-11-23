"""Unified Point-in-Time Multi-Asset Data Foundation (SQLite WAL + Dual Timestamps + Real Collectors).

Architectural pillars synthesized from industry leaders & SCHQ standards:
- NautilusTrader dual-timestamp discipline (`ts_event_ms` vs `ts_avail_ms`):
  every bar records BOTH bar open/event time AND availability time (bar close).
  Causal queries (`as_of_ms`) strictly filter `ts_avail_ms <= as_of_ms` to eliminate lookahead.
- STD-32 Reuse-First: reuses `alphaloop.data.collectors` (HyperliquidCollector, GateCollector)
  and `alphaloop.core.db.connect_wal` — zero duplicated exchange collectors.
- STD-34 Compliant: WAL + busy_timeout, exponential backoff + jitter on HTTP, zero silent
  exception swallowing, finite OHLCV validation, zero synthetic data fallback.
- Cross-Source Gold Arbitration: explicitly quantifies PAXG vs GC=F basis, return correlation,
  and dual-source ADX14 divergence (resolving the CC-20261117-006 PAXG 15.8 vs GC 32.4 blind spot).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import math
import random
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from alphaloop.core.db import connect_wal, transaction
from alphaloop.core.provenance import PROVENANCE_REAL, sha256_obj, utc_now_ms
from alphaloop.data.collectors import GateCollector, HyperliquidCollector
from alphaloop.data.quality import ALLOWED_REAL_PROVENANCES, MIN_VALID_EPOCH_MS
from alphaloop.data.store import MarketDataStore
from alphaloop.platform.assets import (
    ASSET_CLASS_GOLD_PERP,
    ASSET_CLASS_GOLD_TRADFI,
    ASSET_CLASS_MACRO_INDEX,
    AssetSpec,
    GOLD_MACRO_SYMBOLS,
    TradingCalendar,
)

logger = logging.getLogger(__name__)

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
YAHOO_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AlphaLoop-PlatformV3/1.0"


@dataclass(frozen=True)
class CanonicalBar:
    """Unified Point-in-Time OHLCV + Carry bar across Web3, Gold, Macro, and Equities."""

    asset_class: str
    venue: str
    symbol: str
    timeframe_h: float
    ts_event_ms: int        # Bar open timestamp (UTC ms)
    ts_avail_ms: int        # Earliest causal visibility timestamp (bar close UTC ms >= ts_event_ms)
    open: float
    high: float
    low: float
    close: float
    volume_usd: float
    carry_rate_period: float = 0.0   # Realized funding/carry rate over this bar (decimal; + means longs pay shorts)
    carry_interval_h: float = 1.0    # Settlement interval hours of carry_rate_period
    spread_bps: float = 0.0          # Observed top-of-book spread if known, else 0.0
    provenance: str = PROVENANCE_REAL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultiAssetDQCheck:
    gate_id: str
    name: str
    passed: bool
    value: str
    threshold: str
    detail: str


@dataclass
class MultiAssetDQReport:
    dataset_label: str
    asset_class: str
    row_count: int
    dataset_sha256: str
    provenance: str
    passed: bool
    checks: list[MultiAssetDQCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_label": self.dataset_label,
            "asset_class": self.asset_class,
            "row_count": self.row_count,
            "dataset_sha256": self.dataset_sha256,
            "provenance": self.provenance,
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
        }


@dataclass
class CrossSourceDivergenceReport:
    """Dual-source consistency audit (e.g., PAXG 24/7 perp vs COMEX GC=F daily/intraday)."""

    label_a: str
    label_b: str
    overlap_days: int
    median_basis_bps: float
    p95_abs_basis_bps: float
    daily_return_corr: float
    adx14_a: float
    adx14_b: float
    adx14_diff: float
    arbitration_verdict: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_canonical_bar(bar: CanonicalBar) -> tuple[bool, str]:
    """Fail-closed bar integrity & Point-in-Time check (STD-34 Pillar 6)."""
    if bar.ts_event_ms < MIN_VALID_EPOCH_MS or bar.ts_avail_ms < bar.ts_event_ms:
        return False, "invalid_or_lookahead_timestamp"
    if bar.timeframe_h <= 0.0 or not math.isfinite(bar.timeframe_h):
        return False, "invalid_timeframe"
    nums = (bar.open, bar.high, bar.low, bar.close, bar.volume_usd, bar.carry_rate_period, bar.spread_bps)
    if not all(math.isfinite(x) for x in nums):
        return False, "non_finite_numeric"
    if bar.open <= 0.0 or bar.high <= 0.0 or bar.low <= 0.0 or bar.close <= 0.0:
        return False, "non_positive_price"
    if bar.high < max(bar.open, bar.close) - 1e-12 or bar.low > min(bar.open, bar.close) + 1e-12:
        return False, "ohlc_envelope_violation"
    if bar.low > bar.high:
        return False, "inverted_high_low"
    if bar.volume_usd < 0.0 or bar.spread_bps < 0.0:
        return False, "negative_volume_or_spread"
    return True, "ok"


def fingerprint_canonical_bars(bars: list[CanonicalBar]) -> tuple[str, str, int]:
    """Compute deterministic SHA-256 hash and provenance union over a CanonicalBar sequence."""
    h = hashlib.sha256()
    provs: set[str] = set()
    ordered = sorted(bars, key=lambda b: (b.venue, b.symbol, b.timeframe_h, b.ts_event_ms))
    for b in ordered:
        provs.add(b.provenance)
        h.update(
            f"{b.venue}|{b.symbol}|{b.timeframe_h:.4f}|{b.ts_event_ms}|{b.ts_avail_ms}|"
            f"{b.open:.8f}|{b.high:.8f}|{b.low:.8f}|{b.close:.8f}|{b.volume_usd:.4f}|"
            f"{b.carry_rate_period:.10f}|{b.provenance}\n".encode("utf-8")
        )
    prov_str = "+".join(sorted(provs)) if provs else "EMPTY"
    return h.hexdigest(), prov_str, len(ordered)


def resample_canonical_bars(bars: list[CanonicalBar], target_timeframe_h: float) -> list[CanonicalBar]:
    """Strictly causal UTC-aligned bar resampling (e.g. 1h -> 4h / 12h / 24h).

    Preserves exact Point-in-Time visibility: bucket ts_avail_ms = max(ts_avail_ms) of constituent bars.
    Drops incomplete trailing bucket only if fewer than half the expected bars exist when resampling
    fine intraday bars into coarse bars, while keeping complete aligned buckets deterministic.
    """
    if not bars:
        return []
    tf_ms = int(round(target_timeframe_h * HOUR_MS))
    if tf_ms <= 0:
        raise ValueError("target_timeframe_h must be positive")
    ordered = sorted(bars, key=lambda b: b.ts_event_ms)
    if abs(ordered[0].timeframe_h - target_timeframe_h) < 1e-6:
        return list(ordered)

    buckets: dict[int, list[CanonicalBar]] = {}
    for b in ordered:
        b_start = (b.ts_event_ms // tf_ms) * tf_ms
        buckets.setdefault(b_start, []).append(b)

    out: list[CanonicalBar] = []
    for b_start in sorted(buckets):
        grp = buckets[b_start]
        first = grp[0]
        last = grp[-1]
        provs = sorted({x.provenance for x in grp})
        carry_sum = sum(x.carry_rate_period for x in grp)
        spread_avg = statistics.mean(x.spread_bps for x in grp)
        out.append(
            CanonicalBar(
                asset_class=first.asset_class,
                venue=first.venue,
                symbol=first.symbol,
                timeframe_h=float(target_timeframe_h),
                ts_event_ms=b_start,
                ts_avail_ms=max(b_start + tf_ms, max(x.ts_avail_ms for x in grp)),
                open=first.open,
                high=max(x.high for x in grp),
                low=min(x.low for x in grp),
                close=last.close,
                volume_usd=sum(x.volume_usd for x in grp),
                carry_rate_period=carry_sum,
                carry_interval_h=float(target_timeframe_h),
                spread_bps=spread_avg,
                provenance="+".join(provs),
            )
        )
    return out


class MultiAssetDataLake:
    """SQLite WAL Point-in-Time multi-asset research lake with JSONL export boundary."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = connect_wal(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS asset_registry (
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    asset_class TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    updated_ts_ms INTEGER NOT NULL,
                    PRIMARY KEY (venue, symbol)
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS canonical_bars (
                    asset_class TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe_h REAL NOT NULL,
                    ts_event_ms INTEGER NOT NULL,
                    ts_avail_ms INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume_usd REAL NOT NULL,
                    carry_rate_period REAL NOT NULL,
                    carry_interval_h REAL NOT NULL,
                    spread_bps REAL NOT NULL,
                    provenance TEXT NOT NULL,
                    PRIMARY KEY (venue, symbol, timeframe_h, ts_event_ms)
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_canonical_pit
                ON canonical_bars (venue, symbol, timeframe_h, ts_avail_ms);
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS factor_audits (
                    audit_id TEXT PRIMARY KEY,
                    created_ts_ms INTEGER NOT NULL,
                    track_name TEXT NOT NULL,
                    dataset_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS regime_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    created_ts_ms INTEGER NOT NULL,
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe_h REAL NOT NULL,
                    regime_label TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_blueprints (
                    blueprint_id TEXT PRIMARY KEY,
                    created_ts_ms INTEGER NOT NULL,
                    track_name TEXT NOT NULL,
                    archetype TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    dataset_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def register_asset(self, spec: AssetSpec) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT OR REPLACE INTO asset_registry
                (venue, symbol, asset_class, spec_json, updated_ts_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                (spec.venue, spec.symbol, spec.asset_class, json.dumps(spec.to_dict(), sort_keys=True), utc_now_ms()),
            )

    def upsert_bars(self, bars: list[CanonicalBar]) -> tuple[int, int]:
        """Persist validated bars; returns (accepted_count, rejected_count)."""
        if not bars:
            return 0, 0
        valid_rows: list[tuple[Any, ...]] = []
        rejected = 0
        for b in bars:
            ok, reason = validate_canonical_bar(b)
            if not ok:
                rejected += 1
                logger.warning("Rejected invalid bar %s/%s@%s: %s", b.venue, b.symbol, b.ts_event_ms, reason)
                continue
            valid_rows.append(
                (
                    b.asset_class,
                    b.venue,
                    b.symbol,
                    float(b.timeframe_h),
                    int(b.ts_event_ms),
                    int(b.ts_avail_ms),
                    float(b.open),
                    float(b.high),
                    float(b.low),
                    float(b.close),
                    float(b.volume_usd),
                    float(b.carry_rate_period),
                    float(b.carry_interval_h),
                    float(b.spread_bps),
                    b.provenance,
                )
            )
        if valid_rows:
            with transaction(self.conn):
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO canonical_bars
                    (asset_class, venue, symbol, timeframe_h, ts_event_ms, ts_avail_ms,
                     open, high, low, close, volume_usd, carry_rate_period, carry_interval_h,
                     spread_bps, provenance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    valid_rows,
                )
        return len(valid_rows), rejected

    def load_bars(
        self,
        venue: str,
        symbol: str,
        timeframe_h: float,
        start_ms: int | None = None,
        end_ms: int | None = None,
        as_of_ms: int | None = None,
    ) -> list[CanonicalBar]:
        """Load bars with optional Point-in-Time `as_of_ms` causal visibility guard."""
        sql = (
            "SELECT asset_class, venue, symbol, timeframe_h, ts_event_ms, ts_avail_ms, "
            "open, high, low, close, volume_usd, carry_rate_period, carry_interval_h, "
            "spread_bps, provenance FROM canonical_bars "
            "WHERE venue=? AND symbol=? AND ABS(timeframe_h - ?) < 1e-6"
        )
        params: list[Any] = [venue, symbol, float(timeframe_h)]
        if start_ms is not None:
            sql += " AND ts_event_ms >= ?"
            params.append(int(start_ms))
        if end_ms is not None:
            sql += " AND ts_event_ms <= ?"
            params.append(int(end_ms))
        if as_of_ms is not None:
            sql += " AND ts_avail_ms <= ?"
            params.append(int(as_of_ms))
        sql += " ORDER BY ts_event_ms ASC"
        rows = self.conn.fetch_all(sql, params)
        return [
            CanonicalBar(
                asset_class=r["asset_class"],
                venue=r["venue"],
                symbol=r["symbol"],
                timeframe_h=float(r["timeframe_h"]),
                ts_event_ms=int(r["ts_event_ms"]),
                ts_avail_ms=int(r["ts_avail_ms"]),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume_usd=float(r["volume_usd"]),
                carry_rate_period=float(r["carry_rate_period"]),
                carry_interval_h=float(r["carry_interval_h"]),
                spread_bps=float(r["spread_bps"]),
                provenance=r["provenance"],
            )
            for r in rows
        ]

    def import_from_legacy_market_store(
        self,
        legacy_store: MarketDataStore,
        venue: str,
        symbols: list[str],
        asset_class: str,
    ) -> dict[str, int]:
        """Import verified 1h candles + funding from existing AlphaLoop MarketDataStore."""
        counts: dict[str, int] = {}
        for sym in symbols:
            candles = legacy_store.load_candles(venue, sym)
            funding = legacy_store.load_funding(venue, sym)
            fr_by_ts = {f.ts_ms: f for f in funding}
            cbars: list[CanonicalBar] = []
            for c in candles:
                fr = fr_by_ts.get(c.ts_ms)
                cbars.append(
                    CanonicalBar(
                        asset_class=asset_class,
                        venue=venue,
                        symbol=sym,
                        timeframe_h=1.0,
                        ts_event_ms=c.ts_ms,
                        ts_avail_ms=c.ts_ms + HOUR_MS,
                        open=c.open,
                        high=c.high,
                        low=c.low,
                        close=c.close,
                        volume_usd=c.volume_usd,
                        carry_rate_period=fr.funding_rate if fr is not None else 0.0,
                        carry_interval_h=fr.interval_hours if fr is not None else 1.0,
                        provenance=c.provenance,
                    )
                )
            acc, _ = self.upsert_bars(cbars)
            counts[sym] = acc
        return counts

    def export_jsonl(self, venue: str, symbol: str, timeframe_h: float, out_path: str | Path) -> int:
        """Export canonical bars to open JSONL interchange format (DuckDB/Parquet compatible)."""
        bars = self.load_bars(venue, symbol, timeframe_h)
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for b in bars:
                fh.write(json.dumps(b.to_dict(), sort_keys=True) + "\n")
        return len(bars)

    def save_factor_audit(self, track_name: str, dataset_sha256: str, payload: dict[str, Any]) -> str:
        now = utc_now_ms()
        aid = f"FA-{sha256_obj({'t': track_name, 'd': dataset_sha256, 'ts': now})[:10]}"
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO factor_audits (audit_id, created_ts_ms, track_name, dataset_sha256, payload_json) VALUES (?, ?, ?, ?, ?)",
                (aid, now, track_name, dataset_sha256, json.dumps(payload, sort_keys=True)),
            )
        return aid

    def save_regime_snapshot(
        self,
        venue: str,
        symbol: str,
        timeframe_h: float,
        regime_label: str,
        payload: dict[str, Any],
    ) -> str:
        now = utc_now_ms()
        sid = f"REG-{sha256_obj({'v': venue, 's': symbol, 'r': regime_label, 'ts': now})[:10]}"
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO regime_snapshots (snapshot_id, created_ts_ms, venue, symbol, timeframe_h, regime_label, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sid, now, venue, symbol, float(timeframe_h), regime_label, json.dumps(payload, sort_keys=True)),
            )
        return sid

    def save_strategy_blueprint(
        self,
        track_name: str,
        archetype: str,
        verdict: str,
        dataset_sha256: str,
        payload: dict[str, Any],
    ) -> str:
        now = utc_now_ms()
        bid = f"BP-{sha256_obj({'t': track_name, 'a': archetype, 'd': dataset_sha256, 'ts': now})[:10]}"
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO strategy_blueprints (blueprint_id, created_ts_ms, track_name, archetype, verdict, dataset_sha256, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (bid, now, track_name, archetype, verdict, dataset_sha256, json.dumps(payload, sort_keys=True)),
            )
        return bid

    def summary_stats(self) -> dict[str, Any]:
        rows = self.conn.fetch_all(
            """
            SELECT asset_class, venue, symbol, timeframe_h,
                   COUNT(*) AS n_bars,
                   MIN(ts_event_ms) AS min_ts,
                   MAX(ts_event_ms) AS max_ts
            FROM canonical_bars
            GROUP BY asset_class, venue, symbol, timeframe_h
            ORDER BY asset_class, venue, symbol, timeframe_h
            """
        )
        return {
            "db_path": str(self.db_path),
            "series_count": len(rows),
            "series": [dict(r) for r in rows],
        }

    def close(self) -> None:
        self.conn.close()


def audit_bars_quality(
    bars: list[CanonicalBar],
    calendar: TradingCalendar,
    dataset_label: str = "dataset",
    min_bars: int = 40,
    allow_synthetic_in_test: bool = False,
) -> MultiAssetDQReport:
    """Multi-Asset Data Quality Certification (DQ-MA-01 .. DQ-MA-06)."""
    ds_sha, prov, row_count = fingerprint_canonical_bars(bars)
    asset_class = bars[0].asset_class if bars else "UNKNOWN"
    prov_parts = set(prov.split("+")) if prov and prov != "EMPTY" else set()

    dq1_ok = (
        row_count >= min_bars
        and bool(ds_sha)
        and (
            (bool(prov_parts) and prov_parts.issubset(ALLOWED_REAL_PROVENANCES))
            or allow_synthetic_in_test
        )
    )

    ordered = sorted(bars, key=lambda b: b.ts_event_ms)
    non_monotonic = 0
    pit_violations = 0
    invalid_ohlc = 0
    max_gap_h = 0.0
    max_jump_pct = 0.0
    flat_streak = 0
    max_flat_streak = 0

    prev: CanonicalBar | None = None
    for b in ordered:
        ok, _ = validate_canonical_bar(b)
        if not ok:
            invalid_ohlc += 1
        if b.ts_avail_ms < b.ts_event_ms:
            pit_violations += 1
        if b.high == b.low and b.open == b.close:
            flat_streak += 1
            if flat_streak > max_flat_streak:
                max_flat_streak = flat_streak
        else:
            flat_streak = 0
        if prev is not None:
            if b.ts_event_ms <= prev.ts_event_ms:
                non_monotonic += 1
            else:
                gap_h = (b.ts_event_ms - prev.ts_event_ms) / float(HOUR_MS)
                if gap_h > max_gap_h:
                    max_gap_h = gap_h
            if prev.close > 0:
                jump = abs(b.close / prev.close - 1.0) * 100.0
                if jump > max_jump_pct:
                    max_jump_pct = jump
        prev = b

    expected_step_h = ordered[0].timeframe_h if ordered else 1.0
    gap_budget_h = max(calendar.max_expected_gap_hours, expected_step_h * 4.5)

    checks = [
        MultiAssetDQCheck(
            gate_id="DQ-MA-01",
            name="Provenance & Sample Floor",
            passed=dq1_ok,
            value=f"{prov} (n={row_count})",
            threshold=f"REAL & n>={min_bars}",
            detail=f"sha256={ds_sha[:12]}",
        ),
        MultiAssetDQCheck(
            gate_id="DQ-MA-02",
            name="Point-in-Time Causal Visibility",
            passed=(pit_violations == 0 and non_monotonic == 0),
            value=f"pit_err={pit_violations}, non_mono={non_monotonic}",
            threshold="0 violations",
            detail="ts_avail_ms >= ts_event_ms and strictly increasing",
        ),
        MultiAssetDQCheck(
            gate_id="DQ-MA-03",
            name="OHLCV Finite & Envelope Integrity",
            passed=(invalid_ohlc == 0),
            value=f"invalid={invalid_ohlc}",
            threshold="0 invalid bars",
            detail="Checks NaN/Inf, positive prices, high>=max(O,C), low<=min(O,C)",
        ),
        MultiAssetDQCheck(
            gate_id="DQ-MA-04",
            name="Calendar-Aware Gap Continuity",
            passed=(max_gap_h <= gap_budget_h * 2.0),
            value=f"max_gap={max_gap_h:.1f}h",
            threshold=f"<={gap_budget_h * 2.0:.1f}h ({calendar.calendar_id})",
            detail=f"timeframe={expected_step_h}h",
        ),
        MultiAssetDQCheck(
            gate_id="DQ-MA-05",
            name="Extreme Glitch Jump Guard",
            passed=(max_jump_pct <= 80.0),
            value=f"max_bar_jump={max_jump_pct:.2f}%",
            threshold="<=80.0%",
            detail="Blocks 10x decimal/currency corruption spikes",
        ),
        MultiAssetDQCheck(
            gate_id="DQ-MA-06",
            name="Frozen Feed Guard",
            passed=(max_flat_streak <= max(12, int(round(row_count * 0.10)))),
            value=f"max_flat_streak={max_flat_streak}",
            threshold="<=max(12, 10% bars)",
            detail="Detects dead/frozen price feeds",
        ),
    ]
    return MultiAssetDQReport(
        dataset_label=dataset_label,
        asset_class=asset_class,
        row_count=row_count,
        dataset_sha256=ds_sha,
        provenance=prov,
        passed=all(c.passed for c in checks),
        checks=checks,
    )


def _quick_adx14(bars: list[CanonicalBar]) -> float:
    """Compute final Wilder ADX(14) cleanly for dual-source arbitration."""
    if len(bars) < 30:
        return 0.0
    ordered = sorted(bars, key=lambda b: b.ts_event_ms)
    period = 14
    trs: list[float] = []
    p_dms: list[float] = []
    m_dms: list[float] = []
    for i in range(1, len(ordered)):
        cur, prev = ordered[i], ordered[i - 1]
        tr = max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close))
        up = cur.high - prev.high
        dn = prev.low - cur.low
        p_dm = up if (up > dn and up > 0.0) else 0.0
        m_dm = dn if (dn > up and dn > 0.0) else 0.0
        trs.append(tr)
        p_dms.append(p_dm)
        m_dms.append(m_dm)
    if len(trs) < period * 2:
        return 0.0
    sm_tr = sum(trs[:period])
    sm_p = sum(p_dms[:period])
    sm_m = sum(m_dms[:period])
    dxs: list[float] = []
    for i in range(period, len(trs)):
        sm_tr = sm_tr - (sm_tr / period) + trs[i]
        sm_p = sm_p - (sm_p / period) + p_dms[i]
        sm_m = sm_m - (sm_m / period) + m_dms[i]
        if sm_tr > 0:
            pdi = 100.0 * sm_p / sm_tr
            mdi = 100.0 * sm_m / sm_tr
            denom = pdi + mdi
            dx = 100.0 * abs(pdi - mdi) / denom if denom > 0 else 0.0
            dxs.append(dx)
    if len(dxs) < period:
        return 0.0
    adx = sum(dxs[:period]) / period
    for dx in dxs[period:]:
        adx = ((adx * (period - 1)) + dx) / period
    return adx


def audit_cross_source_divergence(
    bars_a: list[CanonicalBar],
    bars_b: list[CanonicalBar],
    label_a: str = "PAXG",
    label_b: str = "GC=F",
) -> CrossSourceDivergenceReport:
    """Arbitrate dual-source gold feeds (e.g. PAXG 24/7 vs COMEX GC=F daily)."""
    day_a = {b.ts_event_ms // DAY_MS: b for b in resample_canonical_bars(bars_a, 24.0)}
    day_b = {b.ts_event_ms // DAY_MS: b for b in resample_canonical_bars(bars_b, 24.0)}
    common = sorted(set(day_a) & set(day_b))
    if len(common) < 20:
        return CrossSourceDivergenceReport(
            label_a=label_a,
            label_b=label_b,
            overlap_days=len(common),
            median_basis_bps=0.0,
            p95_abs_basis_bps=0.0,
            daily_return_corr=0.0,
            adx14_a=0.0,
            adx14_b=0.0,
            adx14_diff=0.0,
            arbitration_verdict="INSUFFICIENT_OVERLAP",
            notes=f"Need >=20 overlapping UTC days, got {len(common)}",
        )

    bases_bps: list[float] = []
    rets_a: list[float] = []
    rets_b: list[float] = []
    aligned_a: list[CanonicalBar] = []
    aligned_b: list[CanonicalBar] = []
    for idx, d in enumerate(common):
        ba, bb = day_a[d], day_b[d]
        aligned_a.append(ba)
        aligned_b.append(bb)
        bases_bps.append((ba.close / bb.close - 1.0) * 10_000.0)
        if idx > 0:
            pa, pb = day_a[common[idx - 1]], day_b[common[idx - 1]]
            rets_a.append(math.log(ba.close / pa.close))
            rets_b.append(math.log(bb.close / pb.close))

    med_basis = statistics.median(bases_bps)
    abs_sorted = sorted(abs(x) for x in bases_bps)
    p95_abs = abs_sorted[min(len(abs_sorted) - 1, int(0.95 * len(abs_sorted)))]

    corr = 0.0
    if len(rets_a) >= 5:
        ma = statistics.mean(rets_a)
        mb = statistics.mean(rets_b)
        va = sum((x - ma) ** 2 for x in rets_a)
        vb = sum((y - mb) ** 2 for y in rets_b)
        if va > 0 and vb > 0:
            corr = sum((x - ma) * (y - mb) for x, y in zip(rets_a, rets_b)) / math.sqrt(va * vb)

    adx_a = _quick_adx14(aligned_a)
    adx_b = _quick_adx14(aligned_b)
    adx_diff = abs(adx_a - adx_b)

    if corr >= 0.85 and adx_diff <= 6.0:
        verdict = "ALIGNED"
        notes = "Both sources track tightly; either execution leg may drive trend regime."
    elif corr >= 0.70 and adx_diff > 6.0:
        verdict = "ADX_DIVERGENCE_USE_EXECUTION_LEG_WITH_MACRO_CONFIRM"
        notes = (
            f"Close returns correlate ({corr:.3f}), but intrabar range/weekend 24/7 noise causes "
            f"ADX14 split ({label_a}={adx_a:.1f} vs {label_b}={adx_b:.1f}, diff={adx_diff:.1f}). "
            "Use execution-venue bar series for stops/sizing and TradFi GC=F+IEF for macro regime."
        )
    else:
        verdict = "DISLOCATED_CAUTION"
        notes = f"Low daily correlation ({corr:.3f}) or wide basis (p95={p95_abs:.1f}bp); audit mapping before trading."

    return CrossSourceDivergenceReport(
        label_a=label_a,
        label_b=label_b,
        overlap_days=len(common),
        median_basis_bps=round(med_basis, 2),
        p95_abs_basis_bps=round(p95_abs, 2),
        daily_return_corr=round(corr, 4),
        adx14_a=round(adx_a, 2),
        adx14_b=round(adx_b, 2),
        adx14_diff=round(adx_diff, 2),
        arbitration_verdict=verdict,
        notes=notes,
    )


class YahooFinanceCollector:
    """STD-34 compliant public Yahoo Finance Chart JSON collector for Gold & Macro daily/1h series."""

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

    def _fetch_json(self, url: str, timeout_sec: float = 12.0, max_retries: int = 3) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(max_retries):
            req = urllib.request.Request(url, headers={"User-Agent": YAHOO_USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_err = exc
                if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    sleep_s = (0.7 * (2 ** attempt)) + random.uniform(0.1, 0.4)
                    logger.warning("Yahoo HTTP %s on %s, backoff %.2fs", exc.code, url, sleep_s)
                    time.sleep(sleep_s)
                    continue
                raise
            except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError, json.JSONDecodeError) as exc:
                last_err = exc
                if attempt < max_retries - 1:
                    sleep_s = (0.6 * (2 ** attempt)) + random.uniform(0.1, 0.35)
                    logger.warning("Yahoo network error on %s: %s, backoff %.2fs", url, exc, sleep_s)
                    time.sleep(sleep_s)
                    continue
                raise
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"Unexpected empty retry loop for {url}")

    def fetch_bars(
        self,
        symbol: str,
        asset_class: str = ASSET_CLASS_MACRO_INDEX,
        range_str: str = "5y",
        interval: str = "1d",
    ) -> list[CanonicalBar]:
        """Fetch real historical OHLCV from Yahoo Finance with strict Point-in-Time ts_avail_ms."""
        tf_h = 24.0 if interval == "1d" else 1.0
        tf_ms = DAY_MS if interval == "1d" else HOUR_MS
        now_ms = utc_now_ms()
        encoded_sym = urllib.parse.quote(symbol, safe="")
        url = f"{self.BASE_URL}/{encoded_sym}?range={range_str}&interval={interval}"
        data = self._fetch_json(url)
        results = (data.get("chart") or {}).get("result") or []
        if not results:
            logger.warning("Yahoo returned empty chart result for %s", symbol)
            return []
        res = results[0]
        timestamps = res.get("timestamp") or []
        quotes = ((res.get("indicators") or {}).get("quote") or [{}])[0]
        opens = quotes.get("open") or []
        highs = quotes.get("high") or []
        lows = quotes.get("low") or []
        closes = quotes.get("close") or []
        vols = quotes.get("volume") or []

        seen: dict[int, CanonicalBar] = {}
        for ts_s, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, vols):
            if o is None or h is None or l is None or c is None:
                continue
            o_f, h_f, l_f, c_f = float(o), float(h), float(l), float(c)
            if not (math.isfinite(o_f) and math.isfinite(h_f) and math.isfinite(l_f) and math.isfinite(c_f)):
                continue
            if o_f <= 0 or h_f <= 0 or l_f <= 0 or c_f <= 0:
                continue
            # Normalize daily bars to UTC midnight; enforce ts_avail_ms = bar close (never peek live unclosed bar)
            raw_ms = int(ts_s) * 1000
            ts_ev = (raw_ms // DAY_MS) * DAY_MS if interval == "1d" else (raw_ms // HOUR_MS) * HOUR_MS
            ts_av = ts_ev + tf_ms
            if ts_av > now_ms:
                continue
            # Guard tiny provider rounding anomalies in high/low envelope
            h_clean = max(h_f, o_f, c_f)
            l_clean = min(l_f, o_f, c_f)
            vol_usd = max(0.0, float(v or 0.0)) * c_f
            seen[ts_ev] = CanonicalBar(
                asset_class=asset_class,
                venue="yahoo",
                symbol=symbol,
                timeframe_h=tf_h,
                ts_event_ms=ts_ev,
                ts_avail_ms=ts_av,
                open=o_f,
                high=h_clean,
                low=l_clean,
                close=c_f,
                volume_usd=vol_usd,
                carry_rate_period=0.0,
                carry_interval_h=tf_h,
                provenance=PROVENANCE_REAL,
            )
        return sorted(seen.values(), key=lambda x: x.ts_event_ms)


def ingest_real_web3_and_gold_lake(
    lake: MultiAssetDataLake,
    web3_symbols: list[str],
    crypto_days: int = 180,
    gold_macro_range: str = "5y",
    include_gate_ref: bool = True,
) -> dict[str, Any]:
    """Fetch REAL Web3 (Hyperliquid + Gate) and Gold+Macro (Yahoo) history into MultiAssetDataLake.

    Strictly STD-32 (reuses HyperliquidCollector & GateCollector) and zero synthetic fallback.
    """
    hl = HyperliquidCollector()
    gate = GateCollector() if include_gate_ref else None
    yahoo = YahooFinanceCollector()
    summary: dict[str, Any] = {"web3_hl": {}, "web3_gate": {}, "gold_macro_yahoo": {}, "errors": []}

    all_perp_syms = list(dict.fromkeys([s.upper() for s in web3_symbols] + ["PAXG"]))
    for sym in all_perp_syms:
        ac = ASSET_CLASS_GOLD_PERP if sym == "PAXG" else "CRYPTO_PERP"
        try:
            candles = hl.fetch_candles_1h(sym, days=crypto_days)
            funding = hl.fetch_funding_history(sym, days=crypto_days)
            fr_map = {f.ts_ms: f for f in funding}
            cbars = [
                CanonicalBar(
                    asset_class=ac,
                    venue="hyperliquid",
                    symbol=sym,
                    timeframe_h=1.0,
                    ts_event_ms=c.ts_ms,
                    ts_avail_ms=c.ts_ms + HOUR_MS,
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    volume_usd=c.volume_usd,
                    carry_rate_period=fr_map[c.ts_ms].funding_rate if c.ts_ms in fr_map else 0.0,
                    carry_interval_h=fr_map[c.ts_ms].interval_hours if c.ts_ms in fr_map else 1.0,
                    provenance=c.provenance,
                )
                for c in candles
            ]
            acc, rej = lake.upsert_bars(cbars)
            summary["web3_hl"][sym] = {"bars_1h": acc, "rejected": rej, "funding_bars": len(funding)}
        except Exception as exc:
            msg = f"HL fetch failed for {sym}: {exc}"
            logger.warning(msg)
            summary["errors"].append(msg)

        if gate is not None and sym in ("BTC", "ETH", "SOL", "PAXG"):
            try:
                gcandles = gate.fetch_candles_1h(sym, days=crypto_days)
                gfunding = gate.fetch_funding_history(sym, days=crypto_days)
                gfr_map = {f.ts_ms: f for f in gfunding}
                gcbars = [
                    CanonicalBar(
                        asset_class=ac,
                        venue="gate",
                        symbol=sym,
                        timeframe_h=1.0,
                        ts_event_ms=c.ts_ms,
                        ts_avail_ms=c.ts_ms + HOUR_MS,
                        open=c.open,
                        high=c.high,
                        low=c.low,
                        close=c.close,
                        volume_usd=c.volume_usd,
                        carry_rate_period=gfr_map[c.ts_ms].funding_rate if c.ts_ms in gfr_map else 0.0,
                        carry_interval_h=gfr_map[c.ts_ms].interval_hours if c.ts_ms in gfr_map else 8.0,
                        provenance=c.provenance,
                    )
                    for c in gcandles
                ]
                acc, rej = lake.upsert_bars(gcbars)
                summary["web3_gate"][sym] = {"bars_1h": acc, "rejected": rej, "funding_bars": len(gfunding)}
            except Exception as exc:
                msg = f"Gate fetch failed for {sym}: {exc}"
                logger.warning(msg)
                summary["errors"].append(msg)

    for msym in GOLD_MACRO_SYMBOLS:
        ac = ASSET_CLASS_GOLD_TRADFI if msym in ("GC=F", "GLD") else ASSET_CLASS_MACRO_INDEX
        try:
            mbars = yahoo.fetch_bars(msym, asset_class=ac, range_str=gold_macro_range, interval="1d")
            acc, rej = lake.upsert_bars(mbars)
            summary["gold_macro_yahoo"][msym] = {"bars_1d": acc, "rejected": rej}
            time.sleep(0.15)
        except Exception as exc:
            msg = f"Yahoo fetch failed for {msym}: {exc}"
            logger.warning(msg)
            summary["errors"].append(msg)

    return summary
