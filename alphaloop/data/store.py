"""Immutable SQLite WAL market data store with SHA-256 dataset fingerprinting."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from alphaloop.core.db import connect_wal, transaction
from alphaloop.core.models import FundingRateBar, PriceBar
from alphaloop.core.provenance import PROVENANCE_REAL, utc_now_ms

logger = logging.getLogger(__name__)


class MarketDataStore:
    """Stores verified historical funding bars, 1h candles, and dataset manifests."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = connect_wal(self.db_path)
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
        rows = self.conn.execute(sql, params).fetchall()
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
        rows = self.conn.execute(sql, params).fetchall()
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
        """Compute deterministic SHA-256 & provenance for exact funding slice used in backtest."""
        h = hashlib.sha256()
        provs: set[str] = set()
        count = 0
        for v in sorted(set(venues)):
            for s in sorted(set(symbols)):
                bars = self.load_funding(v, s, start_ms, end_ms)
                for b in bars:
                    provs.add(b.provenance)
                    h.update(f"{b.venue}|{b.symbol}|{b.ts_ms}|{b.funding_rate:.10f}|{b.interval_hours}\n".encode())
                    count += 1
        if not provs:
            prov = "EMPTY_DATASET"
        elif provs == {PROVENANCE_REAL}:
            prov = PROVENANCE_REAL
        else:
            prov = "+".join(sorted(provs))
        return h.hexdigest(), prov, count

    def summary_stats(self) -> dict[str, Any]:
        f_rows = self.conn.execute(
            """
            SELECT venue, COUNT(DISTINCT symbol) AS symbols, COUNT(*) AS rows,
                   MIN(ts_ms) AS min_ts, MAX(ts_ms) AS max_ts
            FROM funding_rates GROUP BY venue ORDER BY venue
            """
        ).fetchall()
        c_rows = self.conn.execute(
            """
            SELECT venue, COUNT(DISTINCT symbol) AS symbols, COUNT(*) AS rows,
                   MIN(ts_ms) AS min_ts, MAX(ts_ms) AS max_ts
            FROM candles_1h GROUP BY venue ORDER BY venue
            """
        ).fetchall()
        return {
            "funding": [dict(r) for r in f_rows],
            "candles_1h": [dict(r) for r in c_rows],
        }

    def close(self) -> None:
        self.conn.close()
