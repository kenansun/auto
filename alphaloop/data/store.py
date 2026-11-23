"""Immutable SQLite WAL market data store with SHA-256 dataset fingerprinting."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

from alphaloop.core.db import connect_wal, transaction
from alphaloop.core.models import BBOTick, DataQualityReport, FundingRateBar, PriceBar
from alphaloop.core.provenance import PROVENANCE_REAL, utc_now_ms

logger = logging.getLogger(__name__)


class MarketDataStore:
    """Stores verified historical funding bars, 1h candles, live BBO ticks, and DQ certificates."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = connect_wal(self.db_path)
        self._fp_cache: dict[tuple[Any, ...], tuple[str, str, int]] = {}
        self._init_schema()

    def _init_schema(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS funding_rates (
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    funding_rate REAL NOT NULL,
                    interval_hours REAL NOT NULL,
                    provenance TEXT NOT NULL,
                    PRIMARY KEY (venue, symbol, ts_ms)
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_funding_venue_sym_ts
                ON funding_rates (venue, symbol, ts_ms);
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candles_1h (
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume_usd REAL NOT NULL,
                    provenance TEXT NOT NULL,
                    PRIMARY KEY (venue, symbol, ts_ms)
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_candles_venue_sym_ts
                ON candles_1h (venue, symbol, ts_ms);
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bbo_ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    bid_qty REAL NOT NULL,
                    ask_qty REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    provenance TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bbo_venue_sym_ts
                ON bbo_ticks (venue, symbol, ts_ms, id);
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dq_certificates (
                    cert_id TEXT PRIMARY KEY,
                    created_ts_ms INTEGER NOT NULL,
                    dataset_name TEXT NOT NULL,
                    venue_pair TEXT NOT NULL,
                    dataset_sha256 TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    basis_coverage_pct REAL NOT NULL,
                    tier TEXT NOT NULL,
                    passed_count INTEGER NOT NULL,
                    total_count INTEGER NOT NULL,
                    report_json TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS live_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_ts_ms INTEGER NOT NULL,
                    venue TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_manifests (
                    manifest_id TEXT PRIMARY KEY,
                    created_ts_ms INTEGER NOT NULL,
                    provenance TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    dataset_sha256 TEXT NOT NULL,
                    meta_json TEXT NOT NULL
                );
                """
            )

    def upsert_funding(self, bars: list[FundingRateBar]) -> int:
        if not bars:
            return 0
        with transaction(self.conn):
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO funding_rates
                (venue, symbol, ts_ms, funding_rate, interval_hours, provenance)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (b.venue, b.symbol, b.ts_ms, b.funding_rate, b.interval_hours, b.provenance)
                    for b in bars
                ],
            )
        return len(bars)

    def upsert_candles(self, bars: list[PriceBar]) -> int:
        if not bars:
            return 0
        with transaction(self.conn):
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO candles_1h
                (venue, symbol, ts_ms, open, high, low, close, volume_usd, provenance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (b.venue, b.symbol, b.ts_ms, b.open, b.high, b.low, b.close, b.volume_usd, b.provenance)
                    for b in bars
                ],
            )
        return len(bars)

    def upsert_bbo_ticks(self, ticks: list[BBOTick]) -> int:
        """Persist validated BBO ticks (STD-34 Pillar 6: rejects NaN/Inf and crossed books)."""
        if not ticks:
            return 0
        valid_rows = []
        rejected = 0
        for t in ticks:
            if (
                not math.isfinite(t.bid)
                or not math.isfinite(t.ask)
                or t.bid <= 0.0
                or t.ask <= 0.0
                or t.bid > t.ask
                or t.ts_ms <= 0
            ):
                rejected += 1
                continue
            valid_rows.append(
                (t.venue, t.symbol, t.ts_ms, t.bid, t.ask, max(0.0, t.bid_qty), max(0.0, t.ask_qty), max(0.0, t.latency_ms), t.provenance)
            )
        if rejected:
            logger.warning("Rejected %d invalid/crossed BBO ticks before DB write", rejected)
        if not valid_rows:
            return 0
        with transaction(self.conn):
            self.conn.executemany(
                """
                INSERT INTO bbo_ticks
                (venue, symbol, ts_ms, bid, ask, bid_qty, ask_qty, latency_ms, provenance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                valid_rows,
            )
        return len(valid_rows)

    def save_dq_certificate(self, report: DataQualityReport) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT OR REPLACE INTO dq_certificates
                (cert_id, created_ts_ms, dataset_name, venue_pair, dataset_sha256,
                 provenance, row_count, basis_coverage_pct, tier, passed_count, total_count, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.cert_id,
                    report.created_ts_ms,
                    report.dataset_name,
                    report.venue_pair,
                    report.dataset_sha256,
                    report.provenance,
                    report.row_count,
                    report.basis_coverage_pct,
                    report.tier,
                    report.passed_count,
                    report.total_count,
                    json.dumps(report.to_dict(), sort_keys=True),
                ),
            )

    def save_live_snapshot(self, venue: str, payload: Any) -> None:
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO live_snapshots (captured_ts_ms, venue, payload_json) VALUES (?, ?, ?)",
                (utc_now_ms(), venue, json.dumps(payload, separators=(",", ":"))),
            )

    def load_funding(
        self,
        venue: str,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[FundingRateBar]:
        sql = "SELECT venue, symbol, ts_ms, funding_rate, interval_hours, provenance FROM funding_rates WHERE venue=? AND symbol=?"
        params: list[Any] = [venue, symbol]
        if start_ms is not None:
            sql += " AND ts_ms >= ?"
            params.append(start_ms)
        if end_ms is not None:
            sql += " AND ts_ms <= ?"
            params.append(end_ms)
        sql += " ORDER BY ts_ms ASC"
        rows = self.conn.fetch_all(sql, params)
        return [
            FundingRateBar(
                venue=r["venue"],
                symbol=r["symbol"],
                ts_ms=r["ts_ms"],
                funding_rate=r["funding_rate"],
                interval_hours=r["interval_hours"],
                provenance=r["provenance"],
            )
            for r in rows
        ]

    def load_candles(
        self,
        venue: str,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[PriceBar]:
        sql = "SELECT venue, symbol, ts_ms, open, high, low, close, volume_usd, provenance FROM candles_1h WHERE venue=? AND symbol=?"
        params: list[Any] = [venue, symbol]
        if start_ms is not None:
            sql += " AND ts_ms >= ?"
            params.append(start_ms)
        if end_ms is not None:
            sql += " AND ts_ms <= ?"
            params.append(end_ms)
        sql += " ORDER BY ts_ms ASC"
        rows = self.conn.fetch_all(sql, params)
        return [
            PriceBar(
                venue=r["venue"],
                symbol=r["symbol"],
                ts_ms=r["ts_ms"],
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
                volume_usd=r["volume_usd"],
                provenance=r["provenance"],
            )
            for r in rows
        ]

    def fingerprint_slice(
        self,
        venues: list[str],
        symbols: list[str],
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> tuple[str, str, int]:
        """Compute deterministic SHA-256 & provenance across BOTH funding_rates and candles_1h (snapshot-consistent)."""
        v_key = tuple(sorted(set(venues)))
        s_key = tuple(sorted(set(symbols)))
        with self.conn.lock:
            for _attempt in range(3):
                dv_row = self.conn.execute("PRAGMA data_version").fetchone()
                data_ver = int(dv_row[0]) if dv_row else 0
                total_ch = int(self.conn.total_changes)
                cache_key = (v_key, s_key, start_ms, end_ms, data_ver, total_ch)
                cached = self._fp_cache.get(cache_key)
                if cached is not None:
                    return cached

                # Read inside a single SQLite read transaction when not already inside a write tx
                started_read_tx = False
                if not self.conn.in_transaction:
                    self.conn.execute("BEGIN DEFERRED;")
                    started_read_tx = True
                try:
                    h = hashlib.sha256()
                    provs: set[str] = set()
                    count = 0
                    for v in v_key:
                        for s in s_key:
                            bars = self.load_funding(v, s, start_ms, end_ms)
                            for b in bars:
                                provs.add(b.provenance)
                                h.update(f"F|{b.venue}|{b.symbol}|{b.ts_ms}|{b.funding_rate:.10f}|{b.interval_hours}\n".encode())
                                count += 1
                            cbars = self.load_candles(v, s, start_ms, end_ms)
                            for c in cbars:
                                provs.add(c.provenance)
                                h.update(f"C|{c.venue}|{c.symbol}|{c.ts_ms}|{c.open:.8f}|{c.high:.8f}|{c.low:.8f}|{c.close:.8f}\n".encode())
                                count += 1
                finally:
                    if started_read_tx and self.conn.in_transaction:
                        self.conn.execute("COMMIT;")

                dv_after_row = self.conn.execute("PRAGMA data_version").fetchone()
                data_ver_after = int(dv_after_row[0]) if dv_after_row else 0
                total_ch_after = int(self.conn.total_changes)

                if not provs:
                    prov = "EMPTY_DATASET"
                elif provs == {PROVENANCE_REAL}:
                    prov = PROVENANCE_REAL
                else:
                    prov = "+".join(sorted(provs))
                res = (h.hexdigest(), prov, count)

                if data_ver_after == data_ver and total_ch_after == total_ch:
                    if len(self._fp_cache) >= 64:
                        self._fp_cache.clear()
                    self._fp_cache[cache_key] = res
                    return res
                # Concurrent external writer committed during scan; retry for an un-torn cacheable snapshot
            return res

    def load_bbo_ticks(
        self,
        venue: str,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = None,
        order_by_id: bool = False,
    ) -> list[BBOTick]:
        sql = (
            "SELECT venue, symbol, ts_ms, bid, ask, bid_qty, ask_qty, latency_ms, provenance "
            "FROM bbo_ticks WHERE venue=? AND symbol=?"
        )
        params: list[Any] = [venue, symbol]
        if start_ms is not None:
            sql += " AND ts_ms >= ?"
            params.append(start_ms)
        if end_ms is not None:
            sql += " AND ts_ms <= ?"
            params.append(end_ms)
        sql += " ORDER BY rowid ASC" if order_by_id else " ORDER BY ts_ms ASC, rowid ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = self.conn.fetch_all(sql, params)
        return [
            BBOTick(
                venue=r["venue"],
                symbol=r["symbol"],
                ts_ms=r["ts_ms"],
                bid=r["bid"],
                ask=r["ask"],
                bid_qty=r["bid_qty"],
                ask_qty=r["ask_qty"],
                latency_ms=r["latency_ms"],
                provenance=r["provenance"],
            )
            for r in rows
        ]

    def list_dq_certificates(self, limit: int = 30) -> list[dict[str, Any]]:
        rows = self.conn.fetch_all(
            "SELECT report_json FROM dq_certificates ORDER BY created_ts_ms DESC LIMIT ?",
            (limit,),
        )
        return [json.loads(r["report_json"]) for r in rows]

    def latest_dq_certificate(self, venue_pair: str) -> dict[str, Any] | None:
        row = self.conn.fetch_one(
            "SELECT report_json FROM dq_certificates WHERE venue_pair=? ORDER BY created_ts_ms DESC LIMIT 1",
            (venue_pair,),
        )
        return json.loads(row["report_json"]) if row else None

    def summary_stats(self) -> dict[str, Any]:
        f_rows = self.conn.fetch_all(
            """
            SELECT venue, COUNT(DISTINCT symbol) AS symbols, COUNT(*) AS rows,
                   MIN(ts_ms) AS min_ts, MAX(ts_ms) AS max_ts
            FROM funding_rates GROUP BY venue ORDER BY venue
            """
        )
        c_rows = self.conn.fetch_all(
            """
            SELECT venue, COUNT(DISTINCT symbol) AS symbols, COUNT(*) AS rows,
                   MIN(ts_ms) AS min_ts, MAX(ts_ms) AS max_ts
            FROM candles_1h GROUP BY venue ORDER BY venue
            """
        )
        b_rows = self.conn.fetch_all(
            """
            SELECT venue, COUNT(DISTINCT symbol) AS symbols, COUNT(*) AS rows,
                   MIN(ts_ms) AS min_ts, MAX(ts_ms) AS max_ts,
                   ROUND(AVG(latency_ms), 1) AS avg_latency_ms
            FROM bbo_ticks GROUP BY venue ORDER BY venue
            """
        )
        return {
            "funding": [dict(r) for r in f_rows],
            "candles_1h": [dict(r) for r in c_rows],
            "bbo_ticks": [dict(r) for r in b_rows],
        }

    def close(self) -> None:
        self.conn.close()
