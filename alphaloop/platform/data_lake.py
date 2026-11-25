"""Unified Point-in-Time Multi-Asset Data Lake Facade (SQLite WAL + Exact timeframe_sec Index + DQ-MA-01..07).

Preserves 100% v3 backwards compatibility for `MultiAssetDataLake` and `YahooFinanceCollector`
while upgrading:
- D10: `resample_canonical_bars` delegates to `data_foundation.models.resample_bars(..., require_complete=True)`
- D11: Tightened 24/7 calendar gap budget (`DQ-MA-04`), new blocking `DQ-MA-07` perp funding coverage
  & unit sanity check, and preserved `carry_interval_h` across resampling
- D14: Exact `timeframe_sec INTEGER` index (`idx_canonical_pit_sec` / `idx_canonical_ev_sec`) with
  automatic idempotent legacy v3 schema upgrade + collision-proof UUID-nonce audit upserts
- D15d: Contract multiplier propagation in `YahooFinanceCollector` (`GC=F=100.0`, `SI=F=5000.0`)
"""

from __future__ import annotations

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
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from alphaloop.core.db import connect_wal, transaction
from alphaloop.core.provenance import PROVENANCE_REAL, sha256_obj, utc_now_ms
from alphaloop.data.collectors import GateCollector, HyperliquidCollector
from alphaloop.data.quality import ALLOWED_REAL_PROVENANCES
from alphaloop.data.store import MarketDataStore
from alphaloop.platform.assets import (
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_GOLD_PERP,
    ASSET_CLASS_GOLD_TRADFI,
    ASSET_CLASS_MACRO_INDEX,
    AssetSpec,
    DEFAULT_CONTRACT_MULTIPLIERS,
    GOLD_MACRO_SYMBOLS,
    TradingCalendar,
)
from alphaloop.platform.data_foundation.models import (
    DAY_MS,
    HOUR_MS,
    CanonicalBar,
    DomainStoreProtocol,
    MacroEventRelease,
    MacroYieldPoint,
    MultiAssetDQCheck,
    MultiAssetDQReport,
    PAXGBasisPoint,
    Web3DerivativesMetric,
    Web3MarketMacroPoint,
    audit_bars_quality,
    check_canonical_bar,
    classify_tradfi_session_tag,
    fingerprint_canonical_bars,
    resample_bars,
    validate_canonical_bar,
)

logger = logging.getLogger(__name__)

YAHOO_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AlphaLoop-PlatformV4/1.0"

__all__ = [
    "HOUR_MS",
    "DAY_MS",
    "YAHOO_USER_AGENT",
    "CanonicalBar",
    "Web3DerivativesMetric",
    "Web3MarketMacroPoint",
    "MacroYieldPoint",
    "MacroEventRelease",
    "PAXGBasisPoint",
    "DomainStoreProtocol",
    "MultiAssetDQCheck",
    "MultiAssetDQReport",
    "CrossSourceDivergenceReport",
    "check_canonical_bar",
    "validate_canonical_bar",
    "fingerprint_canonical_bars",
    "resample_bars",
    "resample_canonical_bars",
    "MultiAssetDataLake",
    "audit_bars_quality",
    "audit_cross_source_divergence",
    "YahooFinanceCollector",
    "ingest_real_web3_and_gold_lake",
]


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


def resample_canonical_bars(
    bars: Sequence[CanonicalBar],
    target_timeframe_h: float,
    *,
    require_complete: bool = True,
    min_subbars: int | None = None,
) -> list[CanonicalBar]:
    """Strictly causal UTC-aligned bar resampling delegating to data_foundation.models.resample_bars."""
    return resample_bars(
        bars,
        target_timeframe_h,
        require_complete=require_complete,
        min_subbars=min_subbars,
    )


class MultiAssetDataLake:
    """SQLite WAL Point-in-Time multi-asset research lake (v3-compatible facade with v4 integer indexing)."""

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
                    timeframe_sec INTEGER NOT NULL,
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
                    source_fingerprint TEXT NOT NULL DEFAULT '',
                    session_tag TEXT NOT NULL DEFAULT 'ALL_24_7',
                    premium_basis_bps REAL NOT NULL DEFAULT 0.0,
                    contract_multiplier REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY (venue, symbol, timeframe_sec, ts_event_ms)
                );
                """
            )
            # Idempotent in-place schema upgrade for legacy v3 databases lacking v4 columns
            cols = {
                str(r["name"])
                for r in self.conn.fetch_all("PRAGMA table_info(canonical_bars)")
            }
            if "timeframe_sec" not in cols:
                self.conn.execute(
                    "ALTER TABLE canonical_bars ADD COLUMN timeframe_sec INTEGER NOT NULL DEFAULT 0;"
                )
            if "source_fingerprint" not in cols:
                self.conn.execute(
                    "ALTER TABLE canonical_bars ADD COLUMN source_fingerprint TEXT NOT NULL DEFAULT '';"
                )
            if "session_tag" not in cols:
                self.conn.execute(
                    "ALTER TABLE canonical_bars ADD COLUMN session_tag TEXT NOT NULL DEFAULT 'ALL_24_7';"
                )
            if "premium_basis_bps" not in cols:
                self.conn.execute(
                    "ALTER TABLE canonical_bars ADD COLUMN premium_basis_bps REAL NOT NULL DEFAULT 0.0;"
                )
            if "contract_multiplier" not in cols:
                self.conn.execute(
                    "ALTER TABLE canonical_bars ADD COLUMN contract_multiplier REAL NOT NULL DEFAULT 1.0;"
                )
            self.conn.execute(
                """
                UPDATE canonical_bars
                SET timeframe_sec = CAST(ROUND(timeframe_h * 3600.0) AS INTEGER)
                WHERE timeframe_sec <= 0;
                """
            )
            # Exact integer timeframe_sec indexes (fixes D14 ABS(timeframe_h - ?) scan defeat)
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_canonical_pit_sec
                ON canonical_bars (venue, symbol, timeframe_sec, ts_avail_ms);
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_canonical_ev_sec
                ON canonical_bars (venue, symbol, timeframe_sec, ts_event_ms, ts_avail_ms);
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

    def upsert_bars(
        self,
        bars: Sequence[CanonicalBar],
        *,
        allow_synthetic_in_test: bool = True,
    ) -> tuple[int, int]:
        """Persist validated bars; returns (accepted_count, rejected_count)."""
        if not bars:
            return 0, 0
        valid_rows: list[tuple[Any, ...]] = []
        rejected = 0
        for b in bars:
            ok, reason = check_canonical_bar(b, allow_synthetic_in_test=allow_synthetic_in_test)
            if not ok:
                rejected += 1
                logger.warning("Rejected invalid bar %s/%s@%s: %s", b.venue, b.symbol, b.ts_event_ms, reason)
                continue
            tf_sec = int(b.timeframe_sec) if b.timeframe_sec > 0 else int(round(float(b.timeframe_h) * 3600.0))
            valid_rows.append(
                (
                    b.asset_class,
                    b.venue,
                    b.symbol,
                    float(b.timeframe_h),
                    tf_sec,
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
                    b.source_fingerprint,
                    b.session_tag,
                    float(b.premium_basis_bps),
                    float(b.contract_multiplier),
                )
            )
        if valid_rows:
            with transaction(self.conn):
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO canonical_bars
                    (asset_class, venue, symbol, timeframe_h, timeframe_sec, ts_event_ms, ts_avail_ms,
                     open, high, low, close, volume_usd, carry_rate_period, carry_interval_h,
                     spread_bps, provenance, source_fingerprint, session_tag, premium_basis_bps,
                     contract_multiplier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        """Load bars using exact integer timeframe_sec index with optional PIT `as_of_ms` guard."""
        tf_sec = int(round(float(timeframe_h) * 3600.0))
        sql = (
            "SELECT asset_class, venue, symbol, timeframe_h, timeframe_sec, ts_event_ms, ts_avail_ms, "
            "open, high, low, close, volume_usd, carry_rate_period, carry_interval_h, "
            "spread_bps, provenance, source_fingerprint, session_tag, premium_basis_bps, "
            "contract_multiplier FROM canonical_bars "
            "WHERE venue=? AND symbol=? AND timeframe_sec=?"
        )
        params: list[Any] = [venue, symbol, tf_sec]
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
                source_fingerprint=str(r["source_fingerprint"] or ""),
                timeframe_sec=int(r["timeframe_sec"]),
                session_tag=str(r["session_tag"] or "ALL_24_7"),
                premium_basis_bps=float(r["premium_basis_bps"]),
                contract_multiplier=float(r["contract_multiplier"]),
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
        nonce = uuid.uuid4().hex[:8]
        aid = f"FA-{sha256_obj({'t': track_name, 'd': dataset_sha256, 'ts': now, 'n': nonce})[:12]}"
        with transaction(self.conn):
            self.conn.execute(
                "INSERT OR REPLACE INTO factor_audits (audit_id, created_ts_ms, track_name, dataset_sha256, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
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
        nonce = uuid.uuid4().hex[:8]
        sid = f"REG-{sha256_obj({'v': venue, 's': symbol, 'r': regime_label, 'ts': now, 'n': nonce})[:12]}"
        with transaction(self.conn):
            self.conn.execute(
                "INSERT OR REPLACE INTO regime_snapshots "
                "(snapshot_id, created_ts_ms, venue, symbol, timeframe_h, regime_label, payload_json) "
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
        nonce = uuid.uuid4().hex[:8]
        bid = f"BP-{sha256_obj({'t': track_name, 'a': archetype, 'd': dataset_sha256, 'ts': now, 'n': nonce})[:12]}"
        with transaction(self.conn):
            self.conn.execute(
                "INSERT OR REPLACE INTO strategy_blueprints "
                "(blueprint_id, created_ts_ms, track_name, archetype, verdict, dataset_sha256, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (bid, now, track_name, archetype, verdict, dataset_sha256, json.dumps(payload, sort_keys=True)),
            )
        return bid

    def summary_stats(self) -> dict[str, Any]:
        rows = self.conn.fetch_all(
            """
            SELECT asset_class, venue, symbol, timeframe_h, timeframe_sec,
                   COUNT(*) AS n_bars,
                   MIN(ts_event_ms) AS min_ts,
                   MAX(ts_event_ms) AS max_ts
            FROM canonical_bars
            GROUP BY asset_class, venue, symbol, timeframe_h, timeframe_sec
            ORDER BY asset_class, venue, symbol, timeframe_sec
            """
        )
        return {
            "db_path": str(self.db_path),
            "series_count": len(rows),
            "series": [dict(r) for r in rows],
        }

    def close(self) -> None:
        self.conn.close()


def _quick_adx14(bars: Sequence[CanonicalBar]) -> float:
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
    bars_a: Sequence[CanonicalBar],
    bars_b: Sequence[CanonicalBar],
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

    med_basis = float(statistics.median(bases_bps))
    abs_sorted = sorted(abs(x) for x in bases_bps)
    p95_abs = float(abs_sorted[min(len(abs_sorted) - 1, int(0.95 * len(abs_sorted)))])

    corr = 0.0
    if len(rets_a) >= 5:
        ma = sum(rets_a) / len(rets_a)
        mb = sum(rets_b) / len(rets_b)
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
    """STD-34 compliant public Yahoo Finance Chart JSON collector (v3-compatible + D15d multipliers)."""

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
        contract_multiplier: float | None = None,
    ) -> list[CanonicalBar]:
        """Fetch real historical OHLCV from Yahoo Finance with strict Point-in-Time ts_avail_ms."""
        tf_h = 24.0 if interval == "1d" else 1.0
        tf_ms = DAY_MS if interval == "1d" else HOUR_MS
        mult = (
            float(contract_multiplier)
            if contract_multiplier is not None
            else float(DEFAULT_CONTRACT_MULTIPLIERS.get(symbol, 1.0))
        )
        now_ms = utc_now_ms()
        encoded_sym = urllib.parse.quote(symbol, safe="")
        url = f"{self.BASE_URL}/{encoded_sym}?range={range_str}&interval={interval}"
        data = self._fetch_json(url)
        results = (data.get("chart") or {}).get("result") or []
        if not results:
            logger.warning("Yahoo returned empty chart result for %s", symbol)
            return []
        res = results[0]
        meta = res.get("meta") or {}
        inst_type = str(meta.get("instrumentType") or "")
        timestamps = res.get("timestamp") or []
        quotes = ((res.get("indicators") or {}).get("quote") or [{}])[0]
        opens = quotes.get("open") or []
        highs = quotes.get("high") or []
        lows = quotes.get("low") or []
        closes = quotes.get("close") or []
        vols = quotes.get("volume") or []

        seen: dict[int, CanonicalBar] = {}
        for idx, (ts_s, o, h, l, c, v) in enumerate(zip(timestamps, opens, highs, lows, closes, vols)):
            if o is None or h is None or l is None or c is None:
                continue
            o_f, h_f, l_f, c_f = float(o), float(h), float(l), float(c)
            if not (math.isfinite(o_f) and math.isfinite(h_f) and math.isfinite(l_f) and math.isfinite(c_f)):
                continue
            if o_f <= 0 or h_f <= 0 or l_f <= 0 or c_f <= 0:
                continue
            ts_int = int(ts_s)
            v_f = max(0.0, float(v or 0.0))
            if interval == "1h":
                if ts_int % 60 != 0:
                    continue
                if (
                    idx > 0
                    and (ts_int - int(timestamps[idx - 1])) == 1800
                    and (ts_int % 3600) == 0
                    and v_f == 0.0
                    and inst_type.upper() in ("ETF", "EQUITY", "INDEX")
                ):
                    continue
                ts_ev = ts_int * 1000
            else:
                raw_ms = ts_int * 1000
                ts_ev = (raw_ms // DAY_MS) * DAY_MS
            ts_av = ts_ev + tf_ms
            if ts_av > now_ms:
                continue
            if ts_ev in seen:
                continue
            h_clean = max(h_f, o_f, c_f)
            l_clean = min(l_f, o_f, c_f)
            vol_usd = v_f * c_f * mult
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
                source_fingerprint=f"yahoo:v8:{symbol}:{interval}:{range_str}",
                session_tag=classify_tradfi_session_tag(ts_ev, symbol, inst_type, tf_h),
                contract_multiplier=mult,
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
        ac = ASSET_CLASS_GOLD_PERP if sym == "PAXG" else ASSET_CLASS_CRYPTO_PERP
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
            acc, rej = lake.upsert_bars(cbars, allow_synthetic_in_test=False)
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
                acc, rej = lake.upsert_bars(gcbars, allow_synthetic_in_test=False)
                summary["web3_gate"][sym] = {"bars_1h": acc, "rejected": rej, "funding_bars": len(gfunding)}
            except Exception as exc:
                msg = f"Gate fetch failed for {sym}: {exc}"
                logger.warning(msg)
                summary["errors"].append(msg)

    for msym in GOLD_MACRO_SYMBOLS:
        ac = ASSET_CLASS_GOLD_TRADFI if msym in ("GC=F", "GLD") else ASSET_CLASS_MACRO_INDEX
        try:
            mbars = yahoo.fetch_bars(msym, asset_class=ac, range_str=gold_macro_range, interval="1d")
            acc, rej = lake.upsert_bars(mbars, allow_synthetic_in_test=False)
            summary["gold_macro_yahoo"][msym] = {"bars_1d": acc, "rejected": rej}
            time.sleep(0.15)
        except Exception as exc:
            msg = f"Yahoo fetch failed for {msym}: {exc}"
            logger.warning(msg)
            summary["errors"].append(msg)

    return summary
