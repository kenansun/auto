"""Option B Partitioned PIT Data Foundation Router (`catalog.sqlite` + Physically Isolated Domain Stores).

Architecture (Verdict B — Unified Logical PIT Envelope + Physically Partitioned Domain Stores):
- Control Plane (`base_dir/catalog.sqlite`, opened via `connect_wal()` enforcing `PRAGMA journal_mode=WAL`):
  * `asset_registry`: universal instrument registry across domains
  * `dataset_manifests`: SHA-256 provenance manifests for immutable Parquet / `.jsonl.gz` exports
  * `dq_certificates`: persisted `DQ-MA-01..07` certification records
  * `trial_ledger`: REA-001 strictly append-only hypothesis-family ledger enforced by SQLite
    `BEFORE UPDATE`, `BEFORE DELETE`, and `BEFORE INSERT` (no-replace) `RAISE(ABORT, ...)` triggers
  * `factor_audits`, `regime_snapshots`, `strategy_blueprints`: collision-proof UUID-nonce upserts
- Physically Isolated Domain Stores (zero cross-domain lock contention, zero mutable DuckDB WAL):
  * `base_dir/domains/web3_lake.sqlite` (`web3_bars`, `web3_derivatives_metrics`, `web3_market_macro`)
  * `base_dir/domains/gold_macro_lake.sqlite` (`tradfi_bars`, `macro_yield_curve`, `macro_events`,
    `paxg_basis_arbitration`)
  * Pluggable `DomainStoreProtocol` via `register_domain_store()` for future CN A-Shares or new domains
- Immutable Partition Export (`base_dir/parquet/<domain>/`):
  * Uses optional `pyarrow` for `.parquet` when installed, or deterministic `mtime=0` `.jsonl.gz`
  * Optional read-only `:memory:` DuckDB analytical query helper (never writes `.duckdb.wal` files)
"""

from __future__ import annotations

import bisect
import gzip
import json
import logging
import os
import re
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from alphaloop.core.db import LockedConnection, connect_memory, connect_readonly, connect_wal, transaction
from alphaloop.core.provenance import PROVENANCE_REAL, sha256_file, sha256_obj, utc_now_ms
from alphaloop.platform.assets import (
    ASSET_CLASS_CN_A_SHARE,
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_CRYPTO_SPOT,
    ASSET_CLASS_GOLD_PERP,
    ASSET_CLASS_GOLD_TRADFI,
    ASSET_CLASS_MACRO_INDEX,
    AssetSpec,
    DEFAULT_CONTRACT_MULTIPLIERS,
)
from alphaloop.platform.data_foundation.models import (
    DAY_MS,
    CNAshareMacroPoint,
    CanonicalBar,
    DomainStoreProtocol,
    MacroEventRelease,
    MacroYieldPoint,
    MultiAssetDQReport,
    PAXGBasisPoint,
    Web3DerivativesMetric,
    Web3MarketMacroPoint,
    audit_bars_quality,
    classify_tradfi_session_tag,
    validate_canonical_bar,
    validate_cn_ashare_macro_point,
    validate_macro_event_release,
    validate_macro_yield_point,
    validate_paxg_basis_point,
    validate_web3_derivatives_metric,
    validate_web3_market_macro_point,
)

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    _HAS_PYARROW = True
except ImportError:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]
    _HAS_PYARROW = False

try:
    import duckdb  # type: ignore[import-not-found]

    _HAS_DUCKDB = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    _HAS_DUCKDB = False


_DOMAIN_ALIASES: dict[str, str] = {
    "web3": "web3",
    "web3_perp": "web3",
    "gold_macro": "gold_macro",
    "gold": "gold_macro",
    "gold_tradfi": "gold_macro",
    "cn_ashare": "cn_ashare",
    "ashare": "cn_ashare",
    "a_share": "cn_ashare",
    "cn_a_share": "cn_ashare",
}

ASSET_CLASS_TO_DOMAIN: dict[str, str] = {
    ASSET_CLASS_CRYPTO_PERP: "web3",
    ASSET_CLASS_CRYPTO_SPOT: "web3",
    ASSET_CLASS_GOLD_TRADFI: "gold_macro",
    ASSET_CLASS_GOLD_PERP: "gold_macro",
    ASSET_CLASS_MACRO_INDEX: "gold_macro",
    ASSET_CLASS_CN_A_SHARE: "cn_ashare",
}

_FORBIDDEN_SQL_RE = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|REPLACE|ATTACH|DETACH|COPY|PRAGMA|"
    r"EXPORT|IMPORT|CALL|LOAD|INSTALL|SET|RESET|FORCE|CHECKPOINT|VACUUM|"
    r"DESCRIBE|SHOW|SUMMARIZE|EXPLAIN|PIVOT|UNPIVOT|"
    r"read_[a-z0-9_]+|[a-z0-9_]*_scan|[a-z0-9_]*_metadata|"
    r"parquet_[a-z0-9_]+|iceberg_[a-z0-9_]+|delta_[a-z0-9_]+|sniff_[a-z0-9_]+|"
    r"duckdb_[a-z0-9_]+|pragma_[a-z0-9_]+|sqlite_[a-z0-9_]+|"
    r"pg_[a-z0-9_]+|test_[a-z0-9_]+|show_[a-z0-9_]+|substrait_[a-z0-9_]+|"
    r"which_secret|secret_[a-z0-9_]+|database_size|database_list|table_info|storage_info|all_tables|"
    r"st_read|glob|httpfs|getenv|getvariable|query_table|"
    r"information_schema|current_setting|current_database|current_schemas?|"
    r"current_catalog|current_user|session_user|current_role|current_query|txid_current"
    r")\b|\b(?:version|query)\s*\(|\b(?:system|memory|temp)\s*\.",
    re.IGNORECASE,
)

_SQL_TOKEN_RE = re.compile(
    r"""
    '(?:''|[^'])*'           | # single-quoted string literal
    "(?:""|[^"])*"           | # double-quoted identifier / literal
    [A-Za-z_][A-Za-z0-9_]*   | # identifier / keyword
    [(),.]                   | # punctuation
    \S                         # any other non-space token
    """,
    re.VERBOSE,
)

_FROM_CLAUSE_END_KW: frozenset[str] = frozenset(
    {
        "WHERE",
        "GROUP",
        "ORDER",
        "HAVING",
        "LIMIT",
        "OFFSET",
        "QUALIFY",
        "WINDOW",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "FETCH",
        "FOR",
        "INTO",
        "RETURNING",
    }
)


def _assert_no_table_source_literals(cleaned_sql: str) -> None:
    """Reject string/file-path literals anywhere inside FROM/JOIN/TABLE source lists (RB-11, RB-20).

    Uses a parenthesis-depth state stack (`NONE` | `TABLE` | `JOIN_COND`) so derived
    subqueries `(SELECT ...)` and join predicates `ON ...` / `USING (...)` NEVER wipe out
    the enclosing `FROM` table-source state. Consequently:
      - `FROM (SELECT 1) AS t, '/etc/passwd'` -> REJECTED
      - `FROM web3_bars a JOIN web3_bars b ON a.symbol = b.symbol, '/etc/passwd'` -> REJECTED
      - `FROM web3_bars a JOIN web3_bars b USING (symbol), '/etc/passwd'` -> REJECTED
      - `FROM (SELECT 1) t, ('/etc/passwd')` & `(('/etc/passwd'))` -> REJECTED
      - `WITH c AS (SELECT * FROM (SELECT 1) x, '/etc/passwd') SELECT * FROM c` -> REJECTED
      - `WITH c AS (TABLE '/etc/passwd') SELECT * FROM c` -> REJECTED
    while legitimate analytical predicates (`ON a.symbol = 'BTC'`, `ON a.symbol IN ('BTC', 'ETH')`,
    `WHERE symbol IN ('BTC', 'ETH')`, `SELECT 'tag' AS label`) remain permitted.
    """
    tokens = _SQL_TOKEN_RE.findall(cleaned_sql)
    n = len(tokens)
    stack: list[str] = ["NONE"]

    for i, tok in enumerate(tokens):
        up = tok.upper()
        cur = stack[-1]

        if tok == "(":
            nxt = tokens[i + 1] if (i + 1) < n else ""
            if cur == "TABLE":
                if nxt.startswith("'") or nxt.startswith('"'):
                    raise ValueError(
                        "Read-only DuckDB query forbids parenthesized file-path literal in FROM/JOIN/TABLE"
                    )
                if nxt.upper() in ("SELECT", "WITH"):
                    stack.append("NONE")
                else:
                    # Parenthesized table expression / join / nested literal e.g. FROM (('/etc/passwd'))
                    stack.append("TABLE")
            else:
                stack.append("NONE")
            continue

        if tok == ")":
            if len(stack) <= 1:
                raise ValueError("Read-only DuckDB query contains unbalanced parentheses")
            stack.pop()
            continue

        if up in ("FROM", "JOIN", "TABLE"):
            stack[-1] = "TABLE"
            continue

        if cur == "TABLE":
            if up in ("ON", "USING"):
                stack[-1] = "JOIN_COND"
                continue
            if up in _FROM_CLAUSE_END_KW:
                stack[-1] = "NONE"
                continue
            if tok.startswith("'") or tok.startswith('"'):
                raise ValueError(
                    "Read-only DuckDB query forbids quoted file-path literals in FROM/JOIN/TABLE list"
                )
        elif cur == "JOIN_COND":
            if tok == ",":
                # Comma at same paren depth after ON/USING begins a new comma-joined table source!
                stack[-1] = "TABLE"
                continue
            if up in _FROM_CLAUSE_END_KW:
                stack[-1] = "NONE"
                continue

    if len(stack) != 1:
        raise ValueError("Read-only DuckDB query contains unbalanced parentheses")


def has_pyarrow() -> bool:
    """Return True if optional pyarrow accelerator is installed."""
    return _HAS_PYARROW


def has_duckdb() -> bool:
    """Return True if optional duckdb read-only analytical engine is installed."""
    return _HAS_DUCKDB


def normalize_domain(domain: str) -> str:
    key = domain.strip().lower()
    if not key:
        raise ValueError("Domain name must be non-empty")
    return _DOMAIN_ALIASES.get(key, key)


def validate_readonly_sql(sql: str) -> str:
    """Fail-closed SQL guard permitting strictly single-statement SELECT / WITH read-only queries."""
    cleaned = sql.strip()
    if not cleaned:
        raise ValueError("SQL query must be non-empty")
    # Strip trailing semicolon only; reject embedded semicolons (multi-statement injection)
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    if ";" in cleaned or "--" in cleaned or "/*" in cleaned or "$" in cleaned or "`" in cleaned:
        raise ValueError("Read-only DuckDB query forbids multi-statements, comments, dollar-quotes, and backticks")
    if not re.match(r"^(SELECT|WITH)\s", cleaned, re.IGNORECASE):
        raise ValueError("Read-only DuckDB query must start with SELECT or WITH")
    if _FORBIDDEN_SQL_RE.search(cleaned):
        raise ValueError("Read-only DuckDB query contains forbidden write/DDL/external-IO/metadata keyword")
    if re.search(r"\b(?:FROM|JOIN|TABLE)\s*\(?\s*['\"]", cleaned, re.IGNORECASE):
        raise ValueError("Read-only DuckDB query forbids direct file-path literals in FROM/JOIN/TABLE")
    _assert_no_table_source_literals(cleaned)
    return cleaned


def is_comex_weekend_utc(ts_ms: int) -> bool:
    """Return True when UTC timestamp lies in Friday-close weekend window [Fri 22:00 UTC, Sun 22:00 UTC)."""
    dt = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
    wd = dt.weekday()  # Mon=0 .. Fri=4, Sat=5, Sun=6
    if wd == 4 and dt.hour >= 22:
        return True
    if wd == 5:
        return True
    if wd == 6 and dt.hour < 22:
        return True
    return False


def compute_paxg_basis_points(
    perp_bars: Sequence[CanonicalBar],
    spot_bars: Sequence[CanonicalBar],
    gc_bars: Sequence[CanonicalBar],
    *,
    venue_perp: str | None = None,
    venue_spot: str | None = None,
    ref_tradfi_symbol: str = "GC=F",
    max_spot_stale_ms: int = DAY_MS,
    max_gc_stale_ms: int = 5 * DAY_MS,
) -> list[PAXGBasisPoint]:
    """Build strictly causal dual-basis points decomposing PAXG Perp vs PAXG Spot vs COMEX GC=F.

    Uses `bisect_right` strictly over `ts_avail_ms <= perp_bar.ts_avail_ms` so weekend PAXG bars
    causally freeze `gc_ref_close` at Friday's close (`is_weekend_frozen=True`) with zero lookahead.
    """
    if not perp_bars:
        return []
    ordered_perp = sorted(perp_bars, key=lambda b: b.ts_event_ms)
    v_perp = venue_perp or ordered_perp[0].venue
    v_spot = venue_spot or (spot_bars[0].venue if spot_bars else "binance_spot")

    ordered_spot = sorted(spot_bars, key=lambda b: b.ts_avail_ms)
    spot_avails = [int(b.ts_avail_ms) for b in ordered_spot]

    ordered_gc = sorted(gc_bars, key=lambda b: b.ts_avail_ms)
    gc_avails = [int(b.ts_avail_ms) for b in ordered_gc]

    out: list[PAXGBasisPoint] = []
    for pb in ordered_perp:
        ref_avail = int(pb.ts_avail_ms)
        spot_close: float | None = None
        gc_close: float | None = None
        provs = {pb.provenance}

        if ordered_spot:
            idx_s = bisect.bisect_right(spot_avails, ref_avail) - 1
            if idx_s >= 0:
                sb = ordered_spot[idx_s]
                if ref_avail - int(sb.ts_avail_ms) <= max_spot_stale_ms and sb.close > 0.0:
                    spot_close = float(sb.close)
                    provs.add(sb.provenance)

        if ordered_gc:
            idx_g = bisect.bisect_right(gc_avails, ref_avail) - 1
            if idx_g >= 0:
                gb = ordered_gc[idx_g]
                if ref_avail - int(gb.ts_avail_ms) <= max_gc_stale_ms and gb.close > 0.0:
                    gc_close = float(gb.close)
                    provs.add(gb.provenance)

        perp_spot_bps = ((pb.close / spot_close) - 1.0) * 10_000.0 if spot_close is not None else None
        spot_gc_bps = (
            ((spot_close / gc_close) - 1.0) * 10_000.0
            if (spot_close is not None and gc_close is not None)
            else None
        )
        perp_gc_bps = ((pb.close / gc_close) - 1.0) * 10_000.0 if gc_close is not None else None
        # Evaluate weekend freeze at the bar's closing instant (ref_avail - 1) so 1h/4h/12h bars
        # align strictly with bisect_right(gc_avails, ref_avail) and never drift on Sun reopen.
        weekend_flag = is_comex_weekend_utc(ref_avail - 1)
        tf_sec = int(pb.timeframe_sec) if pb.timeframe_sec > 0 else int(round(pb.timeframe_h * 3600.0))

        out.append(
            PAXGBasisPoint(
                venue_perp=v_perp,
                venue_spot=v_spot,
                ref_tradfi_symbol=ref_tradfi_symbol,
                timeframe_sec=tf_sec,
                ts_event_ms=int(pb.ts_event_ms),
                ts_avail_ms=ref_avail,
                paxg_perp_close=float(pb.close),
                paxg_spot_close=spot_close,
                gc_ref_close=gc_close,
                paxg_perp_spot_bps=round(perp_spot_bps, 4) if perp_spot_bps is not None else None,
                paxg_spot_gc_bps=round(spot_gc_bps, 4) if spot_gc_bps is not None else None,
                paxg_perp_gc_bps=round(perp_gc_bps, 4) if perp_gc_bps is not None else None,
                is_weekend_frozen=weekend_flag,
                provenance="+".join(sorted(provs)),
                source_fingerprint=f"paxg_basis:{v_perp}:{v_spot}:{ref_tradfi_symbol}:{tf_sec}",
            )
        )
    return out


class _SQLiteBarDomainStore:
    """Base physically isolated SQLite WAL domain store satisfying DomainStoreProtocol."""

    def __init__(
        self,
        domain_name: str,
        db_path: Path,
        bars_table: str,
        pit_index_name: str,
        *,
        readonly: bool = False,
    ) -> None:
        self.domain_name = domain_name
        self.db_path = Path(db_path)
        self.bars_table = bars_table
        self.pit_index_name = pit_index_name
        self.readonly: bool = bool(readonly)
        if self.readonly:
            if self.db_path.is_file():
                self.conn: LockedConnection = connect_readonly(self.db_path)
            else:
                self.conn = connect_memory()
                self._init_bars_table()
        else:
            self.conn = connect_wal(self.db_path)
            self._init_bars_table()

    def _init_bars_table(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.bars_table} (
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
            self.conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.pit_index_name}
                ON {self.bars_table} (venue, symbol, timeframe_sec, ts_avail_ms);
                """
            )

    def upsert_bars(
        self,
        bars: Sequence[CanonicalBar],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        if not bars:
            return 0
        payload: list[tuple[Any, ...]] = []
        for b in bars:
            validate_canonical_bar(b, allow_synthetic_in_test=allow_synthetic_in_test)
            tf_sec = int(b.timeframe_sec) if b.timeframe_sec > 0 else int(round(float(b.timeframe_h) * 3600.0))
            payload.append(
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
        with transaction(self.conn):
            self.conn.executemany(
                f"""
                INSERT OR REPLACE INTO {self.bars_table}
                (asset_class, venue, symbol, timeframe_h, timeframe_sec, ts_event_ms, ts_avail_ms,
                 open, high, low, close, volume_usd, carry_rate_period, carry_interval_h,
                 spread_bps, provenance, source_fingerprint, session_tag, premium_basis_bps,
                 contract_multiplier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return len(payload)

    def load_bars(
        self,
        venue: str,
        symbol: str,
        timeframe_sec: int,
        *,
        as_of_ms: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[CanonicalBar]:
        sql = (
            f"SELECT asset_class, venue, symbol, timeframe_h, timeframe_sec, ts_event_ms, ts_avail_ms, "
            f"open, high, low, close, volume_usd, carry_rate_period, carry_interval_h, "
            f"spread_bps, provenance, source_fingerprint, session_tag, premium_basis_bps, "
            f"contract_multiplier FROM {self.bars_table} "
            f"WHERE venue=? AND symbol=? AND timeframe_sec=?"
        )
        params: list[Any] = [venue, symbol, int(timeframe_sec)]
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
                asset_class=str(r["asset_class"]),
                venue=str(r["venue"]),
                symbol=str(r["symbol"]),
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
                provenance=str(r["provenance"]),
                source_fingerprint=str(r["source_fingerprint"] or ""),
                timeframe_sec=int(r["timeframe_sec"]),
                session_tag=str(r["session_tag"] or "ALL_24_7"),
                premium_basis_bps=float(r["premium_basis_bps"]),
                contract_multiplier=float(r["contract_multiplier"]),
            )
            for r in rows
        ]

    def list_series(self) -> list[dict[str, Any]]:
        rows = self.conn.fetch_all(
            f"""
            SELECT asset_class, venue, symbol, timeframe_h, timeframe_sec,
                   COUNT(*) AS n_bars,
                   MIN(ts_event_ms) AS min_ts,
                   MAX(ts_event_ms) AS max_ts
            FROM {self.bars_table}
            GROUP BY asset_class, venue, symbol, timeframe_h, timeframe_sec
            ORDER BY asset_class, venue, symbol, timeframe_sec
            """
        )
        return [dict(r) for r in rows]

    def iter_export_rows(self) -> Iterator[dict[str, Any]]:
        rows = self.conn.fetch_all(
            f"""
            SELECT asset_class, venue, symbol, timeframe_h, timeframe_sec, ts_event_ms, ts_avail_ms,
                   open, high, low, close, volume_usd, carry_rate_period, carry_interval_h,
                   spread_bps, provenance, source_fingerprint, session_tag, premium_basis_bps,
                   contract_multiplier
            FROM {self.bars_table}
            ORDER BY venue, symbol, timeframe_sec, ts_event_ms
            """
        )
        for r in rows:
            yield dict(r)

    def close(self) -> None:
        self.conn.close()


class Web3DomainStore(_SQLiteBarDomainStore):
    """Physically isolated Web3 SQLite WAL domain store (`domains/web3_lake.sqlite`)."""

    def __init__(self, db_path: Path, *, readonly: bool = False) -> None:
        super().__init__(
            domain_name="web3",
            db_path=db_path,
            bars_table="web3_bars",
            pit_index_name="idx_web3_bars_pit_sec",
            readonly=readonly,
        )
        if (not self.readonly) or (not self.db_path.is_file()):
            self._init_web3_tables()

    def _init_web3_tables(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web3_derivatives_metrics (
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    ts_event_ms INTEGER NOT NULL,
                    ts_avail_ms INTEGER NOT NULL,
                    open_interest_usd REAL NOT NULL,
                    top_trader_lsr REAL NOT NULL,
                    account_lsr REAL NOT NULL,
                    taker_ls_vol_ratio REAL NOT NULL,
                    provenance TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (venue, symbol, ts_event_ms)
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_w3_metrics_pit
                ON web3_derivatives_metrics (venue, symbol, ts_avail_ms);
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web3_market_macro (
                    source TEXT NOT NULL,
                    series_id TEXT NOT NULL,
                    ts_event_ms INTEGER NOT NULL,
                    ts_avail_ms INTEGER NOT NULL,
                    value REAL NOT NULL,
                    provenance TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (source, series_id, ts_event_ms)
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_w3_macro_pit
                ON web3_market_macro (series_id, ts_avail_ms);
                """
            )

    def upsert_metrics(
        self,
        rows: Sequence[Web3DerivativesMetric],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        if not rows:
            return 0
        payload: list[tuple[Any, ...]] = []
        for r in rows:
            validate_web3_derivatives_metric(r, allow_synthetic_in_test=allow_synthetic_in_test)
            payload.append(
                (
                    r.venue,
                    r.symbol,
                    int(r.ts_event_ms),
                    int(r.ts_avail_ms),
                    float(r.open_interest_usd),
                    float(r.top_trader_lsr),
                    float(r.account_lsr),
                    float(r.taker_ls_vol_ratio),
                    r.provenance,
                    r.source_fingerprint,
                )
            )
        with transaction(self.conn):
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO web3_derivatives_metrics
                (venue, symbol, ts_event_ms, ts_avail_ms, open_interest_usd,
                 top_trader_lsr, account_lsr, taker_ls_vol_ratio, provenance, source_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return len(payload)

    def load_metrics(
        self,
        venue: str,
        symbol: str,
        *,
        as_of_ms: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[Web3DerivativesMetric]:
        sql = (
            "SELECT venue, symbol, ts_event_ms, ts_avail_ms, open_interest_usd, "
            "top_trader_lsr, account_lsr, taker_ls_vol_ratio, provenance, source_fingerprint "
            "FROM web3_derivatives_metrics WHERE venue=? AND symbol=?"
        )
        params: list[Any] = [venue, symbol]
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
            Web3DerivativesMetric(
                venue=str(r["venue"]),
                symbol=str(r["symbol"]),
                ts_event_ms=int(r["ts_event_ms"]),
                ts_avail_ms=int(r["ts_avail_ms"]),
                open_interest_usd=float(r["open_interest_usd"]),
                top_trader_lsr=float(r["top_trader_lsr"]),
                account_lsr=float(r["account_lsr"]),
                taker_ls_vol_ratio=float(r["taker_ls_vol_ratio"]),
                provenance=str(r["provenance"]),
                source_fingerprint=str(r["source_fingerprint"] or ""),
            )
            for r in rows
        ]

    def upsert_macro(
        self,
        rows: Sequence[Web3MarketMacroPoint],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        if not rows:
            return 0
        payload: list[tuple[Any, ...]] = []
        for r in rows:
            validate_web3_market_macro_point(r, allow_synthetic_in_test=allow_synthetic_in_test)
            payload.append(
                (
                    r.source,
                    r.series_id,
                    int(r.ts_event_ms),
                    int(r.ts_avail_ms),
                    float(r.value),
                    r.provenance,
                    r.source_fingerprint,
                )
            )
        with transaction(self.conn):
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO web3_market_macro
                (source, series_id, ts_event_ms, ts_avail_ms, value, provenance, source_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return len(payload)

    def load_macro(
        self,
        series_id: str,
        *,
        as_of_ms: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        source: str | None = None,
    ) -> list[Web3MarketMacroPoint]:
        sql = (
            "SELECT source, series_id, ts_event_ms, ts_avail_ms, value, provenance, source_fingerprint "
            "FROM web3_market_macro WHERE series_id=?"
        )
        params: list[Any] = [series_id]
        if source is not None:
            sql += " AND source=?"
            params.append(source)
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
            Web3MarketMacroPoint(
                source=str(r["source"]),
                series_id=str(r["series_id"]),
                ts_event_ms=int(r["ts_event_ms"]),
                ts_avail_ms=int(r["ts_avail_ms"]),
                value=float(r["value"]),
                provenance=str(r["provenance"]),
                source_fingerprint=str(r["source_fingerprint"] or ""),
            )
            for r in rows
        ]

    def export_tables(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "web3_bars": list(self.iter_export_rows()),
            "web3_derivatives_metrics": [
                dict(r)
                for r in self.conn.fetch_all(
                    "SELECT venue, symbol, ts_event_ms, ts_avail_ms, open_interest_usd, "
                    "top_trader_lsr, account_lsr, taker_ls_vol_ratio, provenance, source_fingerprint "
                    "FROM web3_derivatives_metrics ORDER BY venue, symbol, ts_event_ms"
                )
            ],
            "web3_market_macro": [
                dict(r)
                for r in self.conn.fetch_all(
                    "SELECT source, series_id, ts_event_ms, ts_avail_ms, value, provenance, source_fingerprint "
                    "FROM web3_market_macro ORDER BY source, series_id, ts_event_ms"
                )
            ],
        }


class GoldMacroDomainStore(_SQLiteBarDomainStore):
    """Physically isolated Gold & Macro SQLite WAL domain store (`domains/gold_macro_lake.sqlite`)."""

    def __init__(self, db_path: Path, *, readonly: bool = False) -> None:
        super().__init__(
            domain_name="gold_macro",
            db_path=db_path,
            bars_table="tradfi_bars",
            pit_index_name="idx_tradfi_bars_pit_sec",
            readonly=readonly,
        )
        if (not self.readonly) or (not self.db_path.is_file()):
            self._init_gold_macro_tables()

    def _init_gold_macro_tables(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS macro_yield_curve (
                    source TEXT NOT NULL,
                    ts_event_ms INTEGER NOT NULL,
                    ts_avail_ms INTEGER NOT NULL,
                    real_5y REAL,
                    real_7y REAL,
                    real_10y REAL,
                    real_20y REAL,
                    real_30y REAL,
                    provenance TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (source, ts_event_ms)
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_yield_pit
                ON macro_yield_curve (ts_avail_ms);
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS macro_events (
                    event_type TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    ref_period TEXT NOT NULL,
                    ts_release_ms INTEGER NOT NULL,
                    ts_avail_ms INTEGER NOT NULL,
                    is_sep_meeting INTEGER NOT NULL DEFAULT 0,
                    actual REAL,
                    prior REAL,
                    surprise_z REAL,
                    source TEXT NOT NULL DEFAULT '',
                    provenance TEXT NOT NULL,
                    PRIMARY KEY (event_type, event_id)
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_pit
                ON macro_events (event_type, ts_avail_ms);
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paxg_basis_arbitration (
                    venue_perp TEXT NOT NULL,
                    venue_spot TEXT NOT NULL,
                    ref_tradfi_symbol TEXT NOT NULL DEFAULT 'GC=F',
                    timeframe_sec INTEGER NOT NULL,
                    ts_event_ms INTEGER NOT NULL,
                    ts_avail_ms INTEGER NOT NULL,
                    paxg_perp_close REAL NOT NULL,
                    paxg_spot_close REAL,
                    gc_ref_close REAL,
                    paxg_perp_spot_bps REAL,
                    paxg_spot_gc_bps REAL,
                    paxg_perp_gc_bps REAL,
                    is_weekend_frozen INTEGER NOT NULL DEFAULT 0,
                    provenance TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (venue_perp, venue_spot, ref_tradfi_symbol, timeframe_sec, ts_event_ms)
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_paxg_basis_pit
                ON paxg_basis_arbitration (venue_perp, timeframe_sec, ts_avail_ms);
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_paxg_basis_ev
                ON paxg_basis_arbitration (venue_perp, ref_tradfi_symbol, timeframe_sec, ts_event_ms, ts_avail_ms);
                """
            )

    def upsert_yield_curve(
        self,
        rows: Sequence[MacroYieldPoint],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        if not rows:
            return 0
        payload: list[tuple[Any, ...]] = []
        for r in rows:
            validate_macro_yield_point(r, allow_synthetic_in_test=allow_synthetic_in_test)
            payload.append(
                (
                    r.source,
                    int(r.ts_event_ms),
                    int(r.ts_avail_ms),
                    float(r.real_5y) if r.real_5y is not None else None,
                    float(r.real_7y) if r.real_7y is not None else None,
                    float(r.real_10y) if r.real_10y is not None else None,
                    float(r.real_20y) if r.real_20y is not None else None,
                    float(r.real_30y) if r.real_30y is not None else None,
                    r.provenance,
                    r.source_fingerprint,
                )
            )
        with transaction(self.conn):
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO macro_yield_curve
                (source, ts_event_ms, ts_avail_ms, real_5y, real_7y, real_10y, real_20y, real_30y,
                 provenance, source_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return len(payload)

    def load_yield_curve(
        self,
        *,
        as_of_ms: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        source: str | None = None,
    ) -> list[MacroYieldPoint]:
        sql = (
            "SELECT source, ts_event_ms, ts_avail_ms, real_5y, real_7y, real_10y, real_20y, real_30y, "
            "provenance, source_fingerprint FROM macro_yield_curve WHERE 1=1"
        )
        params: list[Any] = []
        if source is not None:
            sql += " AND source=?"
            params.append(source)
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
            MacroYieldPoint(
                source=str(r["source"]),
                ts_event_ms=int(r["ts_event_ms"]),
                ts_avail_ms=int(r["ts_avail_ms"]),
                real_5y=float(r["real_5y"]) if r["real_5y"] is not None else None,
                real_7y=float(r["real_7y"]) if r["real_7y"] is not None else None,
                real_10y=float(r["real_10y"]) if r["real_10y"] is not None else None,
                real_20y=float(r["real_20y"]) if r["real_20y"] is not None else None,
                real_30y=float(r["real_30y"]) if r["real_30y"] is not None else None,
                provenance=str(r["provenance"]),
                source_fingerprint=str(r["source_fingerprint"] or ""),
            )
            for r in rows
        ]

    def upsert_macro_events(
        self,
        rows: Sequence[MacroEventRelease],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        if not rows:
            return 0
        payload: list[tuple[Any, ...]] = []
        for r in rows:
            validate_macro_event_release(r, allow_synthetic_in_test=allow_synthetic_in_test)
            payload.append(
                (
                    r.event_type,
                    r.event_id,
                    r.ref_period,
                    int(r.ts_release_ms),
                    int(r.ts_avail_ms),
                    1 if r.is_sep_meeting else 0,
                    float(r.actual) if r.actual is not None else None,
                    float(r.prior) if r.prior is not None else None,
                    float(r.surprise_z) if r.surprise_z is not None else None,
                    r.source,
                    r.provenance,
                )
            )
        with transaction(self.conn):
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO macro_events
                (event_type, event_id, ref_period, ts_release_ms, ts_avail_ms, is_sep_meeting,
                 actual, prior, surprise_z, source, provenance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return len(payload)

    def load_macro_events(
        self,
        event_types: Sequence[str] | None = None,
        *,
        as_of_ms: int | None = None,
        lookahead_schedule_ms: int = 0,
    ) -> list[MacroEventRelease]:
        sql = (
            "SELECT event_type, event_id, ref_period, ts_release_ms, ts_avail_ms, is_sep_meeting, "
            "actual, prior, surprise_z, source, provenance FROM macro_events WHERE 1=1"
        )
        params: list[Any] = []
        if event_types:
            types_norm = [str(t).upper() for t in event_types]
            placeholders = ",".join("?" for _ in types_norm)
            sql += f" AND event_type IN ({placeholders})"
            params.extend(types_norm)
        if as_of_ms is not None:
            if lookahead_schedule_ms > 0:
                sql += " AND (ts_avail_ms <= ? OR ts_release_ms <= ?)"
                params.extend([int(as_of_ms), int(as_of_ms) + int(lookahead_schedule_ms)])
            else:
                sql += " AND ts_avail_ms <= ?"
                params.append(int(as_of_ms))
        sql += " ORDER BY ts_release_ms ASC, event_type ASC, event_id ASC"
        rows = self.conn.fetch_all(sql, params)
        out: list[MacroEventRelease] = []
        for r in rows:
            ts_rel = int(r["ts_release_ms"])
            ts_av = int(r["ts_avail_ms"])
            actual = float(r["actual"]) if r["actual"] is not None else None
            prior = float(r["prior"]) if r["prior"] is not None else None
            surprise_z = float(r["surprise_z"]) if r["surprise_z"] is not None else None
            # Strict PIT redaction for pre-scheduled upcoming events when lookahead_schedule_ms > 0
            if as_of_ms is not None and ts_av > int(as_of_ms):
                actual = None
                surprise_z = None
                if ts_rel - int(as_of_ms) > 20 * DAY_MS:
                    prior = None
            out.append(
                MacroEventRelease(
                    event_type=str(r["event_type"]),
                    event_id=str(r["event_id"]),
                    ref_period=str(r["ref_period"]),
                    ts_release_ms=ts_rel,
                    ts_avail_ms=ts_av,
                    is_sep_meeting=bool(r["is_sep_meeting"]),
                    actual=actual,
                    prior=prior,
                    surprise_z=surprise_z,
                    source=str(r["source"] or ""),
                    provenance=str(r["provenance"]),
                )
            )
        return out

    def upsert_paxg_basis(
        self,
        rows: Sequence[PAXGBasisPoint],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        if not rows:
            return 0
        payload: list[tuple[Any, ...]] = []
        for r in rows:
            validate_paxg_basis_point(r, allow_synthetic_in_test=allow_synthetic_in_test)
            payload.append(
                (
                    r.venue_perp,
                    r.venue_spot,
                    r.ref_tradfi_symbol,
                    int(r.timeframe_sec),
                    int(r.ts_event_ms),
                    int(r.ts_avail_ms),
                    float(r.paxg_perp_close),
                    float(r.paxg_spot_close) if r.paxg_spot_close is not None else None,
                    float(r.gc_ref_close) if r.gc_ref_close is not None else None,
                    float(r.paxg_perp_spot_bps) if r.paxg_perp_spot_bps is not None else None,
                    float(r.paxg_spot_gc_bps) if r.paxg_spot_gc_bps is not None else None,
                    float(r.paxg_perp_gc_bps) if r.paxg_perp_gc_bps is not None else None,
                    1 if r.is_weekend_frozen else 0,
                    r.provenance,
                    r.source_fingerprint,
                )
            )
        with transaction(self.conn):
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO paxg_basis_arbitration
                (venue_perp, venue_spot, ref_tradfi_symbol, timeframe_sec, ts_event_ms, ts_avail_ms,
                 paxg_perp_close, paxg_spot_close, gc_ref_close, paxg_perp_spot_bps,
                 paxg_spot_gc_bps, paxg_perp_gc_bps, is_weekend_frozen, provenance, source_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return len(payload)

    def load_paxg_basis(
        self,
        venue_perp: str,
        timeframe_sec: int,
        *,
        as_of_ms: int | None = None,
        venue_spot: str | None = None,
        ref_tradfi_symbol: str = "GC=F",
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[PAXGBasisPoint]:
        sql = (
            "SELECT venue_perp, venue_spot, ref_tradfi_symbol, timeframe_sec, ts_event_ms, ts_avail_ms, "
            "paxg_perp_close, paxg_spot_close, gc_ref_close, paxg_perp_spot_bps, paxg_spot_gc_bps, "
            "paxg_perp_gc_bps, is_weekend_frozen, provenance, source_fingerprint "
            "FROM paxg_basis_arbitration WHERE venue_perp=? AND timeframe_sec=? AND ref_tradfi_symbol=?"
        )
        params: list[Any] = [venue_perp, int(timeframe_sec), ref_tradfi_symbol]
        if venue_spot is not None:
            sql += " AND venue_spot=?"
            params.append(venue_spot)
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
            PAXGBasisPoint(
                venue_perp=str(r["venue_perp"]),
                venue_spot=str(r["venue_spot"]),
                ref_tradfi_symbol=str(r["ref_tradfi_symbol"]),
                timeframe_sec=int(r["timeframe_sec"]),
                ts_event_ms=int(r["ts_event_ms"]),
                ts_avail_ms=int(r["ts_avail_ms"]),
                paxg_perp_close=float(r["paxg_perp_close"]),
                paxg_spot_close=float(r["paxg_spot_close"]) if r["paxg_spot_close"] is not None else None,
                gc_ref_close=float(r["gc_ref_close"]) if r["gc_ref_close"] is not None else None,
                paxg_perp_spot_bps=float(r["paxg_perp_spot_bps"]) if r["paxg_perp_spot_bps"] is not None else None,
                paxg_spot_gc_bps=float(r["paxg_spot_gc_bps"]) if r["paxg_spot_gc_bps"] is not None else None,
                paxg_perp_gc_bps=float(r["paxg_perp_gc_bps"]) if r["paxg_perp_gc_bps"] is not None else None,
                is_weekend_frozen=bool(r["is_weekend_frozen"]),
                provenance=str(r["provenance"]),
                source_fingerprint=str(r["source_fingerprint"] or ""),
            )
            for r in rows
        ]

    def export_tables(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "tradfi_bars": list(self.iter_export_rows()),
            "macro_yield_curve": [
                dict(r)
                for r in self.conn.fetch_all(
                    "SELECT source, ts_event_ms, ts_avail_ms, real_5y, real_7y, real_10y, real_20y, real_30y, "
                    "provenance, source_fingerprint FROM macro_yield_curve ORDER BY source, ts_event_ms"
                )
            ],
            "macro_events": [
                dict(r)
                for r in self.conn.fetch_all(
                    "SELECT event_type, event_id, ref_period, ts_release_ms, ts_avail_ms, is_sep_meeting, "
                    "actual, prior, surprise_z, source, provenance "
                    "FROM macro_events ORDER BY ts_release_ms, event_type, event_id"
                )
            ],
            "paxg_basis_arbitration": [
                dict(r)
                for r in self.conn.fetch_all(
                    "SELECT venue_perp, venue_spot, ref_tradfi_symbol, timeframe_sec, ts_event_ms, ts_avail_ms, "
                    "paxg_perp_close, paxg_spot_close, gc_ref_close, paxg_perp_spot_bps, paxg_spot_gc_bps, "
                    "paxg_perp_gc_bps, is_weekend_frozen, provenance, source_fingerprint "
                    "FROM paxg_basis_arbitration ORDER BY venue_perp, venue_spot, ref_tradfi_symbol, timeframe_sec, ts_event_ms"
                )
            ],
        }


class CNAshareDomainStore(_SQLiteBarDomainStore):
    """Physically isolated CN A-Share SQLite WAL domain store (`domains/cn_ashare_lake.sqlite`).

    Zero cross-domain coupling or lock contention with Web3 or Gold/Macro SQLite stores.
    Stores both `cn_ashare_bars` (OHLCV ETFs) and `cn_ashare_macro` (ChinaBond CGB yields,
    SSE/SZSE margin financing balances, USD/CNY & offshore proxies) under strict PIT lags.
    """

    def __init__(self, db_path: Path, *, readonly: bool = False) -> None:
        super().__init__(
            domain_name="cn_ashare",
            db_path=db_path,
            bars_table="cn_ashare_bars",
            pit_index_name="idx_cn_ashare_bars_pit_sec",
            readonly=readonly,
        )
        if (not self.readonly) or (not self.db_path.is_file()):
            self._init_cn_macro_schema()

    def _init_cn_macro_schema(self) -> None:
        with transaction(self.conn):
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cn_ashare_macro (
                    source TEXT NOT NULL,
                    series_id TEXT NOT NULL,
                    ts_event_ms INTEGER NOT NULL,
                    ts_avail_ms INTEGER NOT NULL,
                    value REAL NOT NULL,
                    provenance TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    PRIMARY KEY (series_id, ts_event_ms)
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cn_ashare_macro_pit
                ON cn_ashare_macro (series_id, ts_avail_ms);
                """
            )

    def upsert_cn_macro(
        self,
        rows: Sequence[CNAshareMacroPoint],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        if not rows:
            return 0
        payload: list[tuple[Any, ...]] = []
        for r in rows:
            validate_cn_ashare_macro_point(r, allow_synthetic_in_test=allow_synthetic_in_test)
            payload.append(
                (
                    r.source,
                    r.series_id,
                    int(r.ts_event_ms),
                    int(r.ts_avail_ms),
                    float(r.value),
                    r.provenance,
                    r.source_fingerprint,
                )
            )
        with transaction(self.conn):
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO cn_ashare_macro
                (source, series_id, ts_event_ms, ts_avail_ms, value, provenance, source_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return len(payload)

    def load_cn_macro(
        self,
        series_ids: Sequence[str] | None = None,
        *,
        as_of_ms: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[CNAshareMacroPoint]:
        sql = (
            "SELECT source, series_id, ts_event_ms, ts_avail_ms, value, provenance, source_fingerprint "
            "FROM cn_ashare_macro WHERE 1=1"
        )
        params: list[Any] = []
        if series_ids:
            placeholders = ",".join("?" for _ in series_ids)
            sql += f" AND series_id IN ({placeholders})"
            params.extend(str(s) for s in series_ids)
        if start_ms is not None:
            sql += " AND ts_event_ms >= ?"
            params.append(int(start_ms))
        if end_ms is not None:
            sql += " AND ts_event_ms <= ?"
            params.append(int(end_ms))
        if as_of_ms is not None:
            sql += " AND ts_avail_ms <= ?"
            params.append(int(as_of_ms))
        sql += " ORDER BY series_id ASC, ts_event_ms ASC"
        rows = self.conn.fetch_all(sql, params)
        return [
            CNAshareMacroPoint(
                source=str(r["source"]),
                series_id=str(r["series_id"]),
                ts_event_ms=int(r["ts_event_ms"]),
                ts_avail_ms=int(r["ts_avail_ms"]),
                value=float(r["value"]),
                provenance=str(r["provenance"]),
                source_fingerprint=str(r["source_fingerprint"] or ""),
            )
            for r in rows
        ]

    def export_tables(self) -> dict[str, list[dict[str, Any]]]:
        tables: dict[str, list[dict[str, Any]]] = {
            "cn_ashare_bars": list(self.iter_export_rows()),
        }
        macro_rows = [
            dict(r)
            for r in self.conn.fetch_all(
                "SELECT source, series_id, ts_event_ms, ts_avail_ms, value, provenance, source_fingerprint "
                "FROM cn_ashare_macro ORDER BY series_id ASC, ts_event_ms ASC"
            )
        ]
        if macro_rows:
            tables["cn_ashare_macro"] = macro_rows
        return tables


def _atomic_write_jsonl_gz(rows: Sequence[dict[str, Any]], target: Path) -> int:
    """Write deterministic gzip-compressed JSONL (`mtime=0, filename=''`) via atomic os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".jsonl.gz", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as raw_fh:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, mtime=0) as gz_fh:
                for row in rows:
                    line = (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode(
                        "utf-8"
                    )
                    gz_fh.write(line)
            raw_fh.flush()
            os.fsync(raw_fh.fileno())
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return len(rows)


def _atomic_write_parquet(rows: Sequence[dict[str, Any]], target: Path) -> int:
    """Write immutable Parquet file via optional pyarrow and atomic os.replace."""
    if not _HAS_PYARROW or pa is None or pq is None:
        raise RuntimeError("pyarrow is required for fmt='parquet'; use fmt='auto' or fmt='jsonl.gz'")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet", dir=str(target.parent))
    os.close(fd)
    try:
        table = pa.Table.from_pylist(list(rows)) if rows else pa.table({"_empty": pa.array([], type=pa.int64())})
        pq.write_table(table, tmp_name)
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return len(rows)


_VALID_TRIAL_LEDGER_MODES: frozenset[str] = frozenset({"official", "sandbox", "readonly"})


class DataFoundationRouter:
    """Unified PIT Control Plane (`catalog.sqlite`) + Physically Partitioned Domain Stores Router.

    Supports `trial_ledger_mode` (`OBS-07` isolation):
      - `"official"` (default): appends trials directly into immutable `catalog.sqlite` `trial_ledger`.
      - `"sandbox"`: writes new trials into an isolated sandbox SQLite WAL DB (0 rows written to
        official `catalog.sqlite`), while `get_family_trial_count()` returns `official + sandbox`
        so DSR is NEVER under-penalized.
      - `"readonly"`: appends trials strictly in memory (0 disk writes to `catalog.sqlite`), while
        `get_family_trial_count()` returns `official + in_memory` so DSR stays strictly conservative.
    """

    def __init__(
        self,
        base_dir: Path | str,
        *,
        trial_ledger_mode: str = "official",
        sandbox_trial_db_path: Path | str | None = None,
    ) -> None:
        mode_norm = str(trial_ledger_mode).strip().lower()
        if mode_norm not in _VALID_TRIAL_LEDGER_MODES:
            raise ValueError(
                f"Invalid trial_ledger_mode {trial_ledger_mode!r}; expected one of {sorted(_VALID_TRIAL_LEDGER_MODES)}"
            )
        self.trial_ledger_mode: str = mode_norm
        self._sandbox_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._sandbox_trial_conn: LockedConnection | None = None
        self._sandbox_trial_path: Path | None = None
        self._readonly_trials: list[dict[str, Any]] = []

        self.base_dir = Path(base_dir)
        self.domains_dir = self.base_dir / "domains"
        self.parquet_dir = self.base_dir / "parquet"
        self.catalog_path = self.base_dir / "catalog.sqlite"
        self._official_lake_readonly: bool = self.trial_ledger_mode in ("readonly", "sandbox")

        if self._official_lake_readonly:
            if not self.catalog_path.is_file():
                raise FileNotFoundError(
                    f"Official catalog.sqlite not found at {self.catalog_path} "
                    f"for trial_ledger_mode={self.trial_ledger_mode!r}"
                )
            self.catalog_conn: LockedConnection = connect_readonly(self.catalog_path)
        else:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.domains_dir.mkdir(parents=True, exist_ok=True)
            self.catalog_conn = connect_wal(self.catalog_path)
            self._init_catalog_schema()

        if self.trial_ledger_mode == "sandbox":
            if sandbox_trial_db_path is not None:
                sb_path = Path(sandbox_trial_db_path)
                sb_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                self._sandbox_tmpdir = tempfile.TemporaryDirectory(prefix="alphaloop_trial_sandbox_")
                sb_path = Path(self._sandbox_tmpdir.name) / "trial_sandbox.sqlite"
            self._sandbox_trial_path = sb_path
            self._sandbox_trial_conn = connect_wal(sb_path)
            self._init_trial_ledger_table(self._sandbox_trial_conn)

        self.web3_store = Web3DomainStore(
            self.domains_dir / "web3_lake.sqlite",
            readonly=self._official_lake_readonly,
        )
        self.gold_macro_store = GoldMacroDomainStore(
            self.domains_dir / "gold_macro_lake.sqlite",
            readonly=self._official_lake_readonly,
        )
        self.cn_ashare_store = CNAshareDomainStore(
            self.domains_dir / "cn_ashare_lake.sqlite",
            readonly=self._official_lake_readonly,
        )
        self._domain_stores: dict[str, DomainStoreProtocol] = {
            "web3": self.web3_store,
            "gold_macro": self.gold_macro_store,
            "cn_ashare": self.cn_ashare_store,
        }

    def __enter__(self) -> DataFoundationRouter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @staticmethod
    def has_pyarrow() -> bool:
        return has_pyarrow()

    @staticmethod
    def has_duckdb() -> bool:
        return has_duckdb()

    @staticmethod
    def _init_trial_ledger_table(conn: LockedConnection) -> None:
        """Create immutable REA-001 `trial_ledger` table + append-only triggers on a SQLite WAL connection."""
        with transaction(conn):
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trial_ledger (
                    trial_id TEXT PRIMARY KEY,
                    created_ts_ms INTEGER NOT NULL,
                    family_id TEXT NOT NULL,
                    track_name TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    dataset_sha256 TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    schema_version TEXT NOT NULL DEFAULT 'rea-001-v1'
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trial_family
                ON trial_ledger (family_id, created_ts_ms);
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_trial_ledger_no_update
                BEFORE UPDATE ON trial_ledger
                BEGIN
                    SELECT RAISE(ABORT, 'trial_ledger is append-only (REA-001): UPDATE forbidden');
                END;
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_trial_ledger_no_delete
                BEFORE DELETE ON trial_ledger
                BEGIN
                    SELECT RAISE(ABORT, 'trial_ledger is append-only (REA-001): DELETE forbidden');
                END;
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_trial_ledger_no_replace
                BEFORE INSERT ON trial_ledger
                WHEN EXISTS (SELECT 1 FROM trial_ledger WHERE trial_id = NEW.trial_id)
                BEGIN
                    SELECT RAISE(ABORT, 'trial_ledger is append-only (REA-001): REPLACE/overwrite forbidden');
                END;
                """
            )

    def _init_catalog_schema(self) -> None:
        with transaction(self.catalog_conn):
            self.catalog_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS asset_registry (
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    asset_class TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    updated_ts_ms INTEGER NOT NULL,
                    PRIMARY KEY (venue, symbol)
                );
                """
            )
            self.catalog_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_manifests (
                    manifest_id TEXT PRIMARY KEY,
                    created_ts_ms INTEGER NOT NULL,
                    domain TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    format TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    file_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    min_ts_ms INTEGER,
                    max_ts_ms INTEGER,
                    provenance TEXT NOT NULL
                );
                """
            )
            self.catalog_conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_manifests_domain
                ON dataset_manifests (domain, created_ts_ms);
                """
            )
            self.catalog_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dq_certificates (
                    cert_id TEXT PRIMARY KEY,
                    created_ts_ms INTEGER NOT NULL,
                    domain TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe_sec INTEGER NOT NULL,
                    dataset_sha256 TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            self.catalog_conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dq_cert_series
                ON dq_certificates (domain, venue, symbol, timeframe_sec, created_ts_ms);
                """
            )
            self.catalog_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trial_ledger (
                    trial_id TEXT PRIMARY KEY,
                    created_ts_ms INTEGER NOT NULL,
                    family_id TEXT NOT NULL,
                    track_name TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    dataset_sha256 TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    schema_version TEXT NOT NULL DEFAULT 'rea-001-v1'
                );
                """
            )
            self.catalog_conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trial_family
                ON trial_ledger (family_id, created_ts_ms);
                """
            )
            self.catalog_conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_trial_ledger_no_update
                BEFORE UPDATE ON trial_ledger
                BEGIN
                    SELECT RAISE(ABORT, 'trial_ledger is append-only (REA-001): UPDATE forbidden');
                END;
                """
            )
            self.catalog_conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_trial_ledger_no_delete
                BEFORE DELETE ON trial_ledger
                BEGIN
                    SELECT RAISE(ABORT, 'trial_ledger is append-only (REA-001): DELETE forbidden');
                END;
                """
            )
            self.catalog_conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_trial_ledger_no_replace
                BEFORE INSERT ON trial_ledger
                WHEN EXISTS (SELECT 1 FROM trial_ledger WHERE trial_id = NEW.trial_id)
                BEGIN
                    SELECT RAISE(ABORT, 'trial_ledger is append-only (REA-001): REPLACE/overwrite forbidden');
                END;
                """
            )
            self.catalog_conn.execute(
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
            self.catalog_conn.execute(
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
            self.catalog_conn.execute(
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

    def register_domain_store(self, store: DomainStoreProtocol) -> None:
        """Register an additional isolated domain store (e.g. CN A-Shares) without modifying Web3/Gold."""
        if not isinstance(store, DomainStoreProtocol):
            raise TypeError(f"Store {type(store).__name__} does not satisfy DomainStoreProtocol")
        key = normalize_domain(store.domain_name)
        prev = self._domain_stores.get(key)
        if prev is not None and prev is not store:
            prev.close()
        self._domain_stores[key] = store

    def get_domain_store(self, domain: str) -> DomainStoreProtocol:
        key = normalize_domain(domain)
        store = self._domain_stores.get(key)
        if store is None:
            raise KeyError(
                f"Unknown domain {domain!r} (normalized {key!r}); registered={sorted(self._domain_stores)}"
            )
        return store

    def _assert_official_writable(self, op_name: str) -> None:
        if self._official_lake_readonly:
            raise PermissionError(
                f"Official lake write {op_name!r} forbidden in trial_ledger_mode={self.trial_ledger_mode!r}"
            )

    def register_asset(self, spec: AssetSpec, *, domain: str | None = None) -> None:
        self._assert_official_writable("register_asset")
        dom = normalize_domain(domain) if domain else ASSET_CLASS_TO_DOMAIN.get(spec.asset_class, "web3")
        with transaction(self.catalog_conn):
            self.catalog_conn.execute(
                """
                INSERT OR REPLACE INTO asset_registry
                (venue, symbol, asset_class, domain, spec_json, updated_ts_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.venue,
                    spec.symbol,
                    spec.asset_class,
                    dom,
                    json.dumps(spec.to_dict(), sort_keys=True),
                    utc_now_ms(),
                ),
            )

    def upsert_bars(
        self,
        domain: str,
        bars: Sequence[CanonicalBar],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        store = self.get_domain_store(domain)
        return store.upsert_bars(bars, allow_synthetic_in_test=allow_synthetic_in_test)

    def load_bars(
        self,
        domain: str,
        venue: str,
        symbol: str,
        timeframe_sec: int,
        *,
        as_of_ms: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[CanonicalBar]:
        store = self.get_domain_store(domain)
        return store.load_bars(
            venue,
            symbol,
            int(timeframe_sec),
            as_of_ms=as_of_ms,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def upsert_web3_metrics(
        self,
        rows: Sequence[Web3DerivativesMetric],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        return self.web3_store.upsert_metrics(rows, allow_synthetic_in_test=allow_synthetic_in_test)

    def load_web3_metrics(
        self,
        venue: str,
        symbol: str,
        *,
        as_of_ms: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[Web3DerivativesMetric]:
        return self.web3_store.load_metrics(
            venue,
            symbol,
            as_of_ms=as_of_ms,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def upsert_web3_macro(
        self,
        rows: Sequence[Web3MarketMacroPoint],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        return self.web3_store.upsert_macro(rows, allow_synthetic_in_test=allow_synthetic_in_test)

    def load_web3_macro(
        self,
        series_id: str,
        *,
        as_of_ms: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        source: str | None = None,
    ) -> list[Web3MarketMacroPoint]:
        return self.web3_store.load_macro(
            series_id,
            as_of_ms=as_of_ms,
            start_ms=start_ms,
            end_ms=end_ms,
            source=source,
        )

    def upsert_yield_curve(
        self,
        rows: Sequence[MacroYieldPoint],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        return self.gold_macro_store.upsert_yield_curve(rows, allow_synthetic_in_test=allow_synthetic_in_test)

    def load_yield_curve(
        self,
        *,
        as_of_ms: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        source: str | None = None,
    ) -> list[MacroYieldPoint]:
        return self.gold_macro_store.load_yield_curve(
            as_of_ms=as_of_ms,
            start_ms=start_ms,
            end_ms=end_ms,
            source=source,
        )

    def upsert_macro_events(
        self,
        rows: Sequence[MacroEventRelease],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        return self.gold_macro_store.upsert_macro_events(rows, allow_synthetic_in_test=allow_synthetic_in_test)

    def load_macro_events(
        self,
        event_types: Sequence[str] | None = None,
        *,
        as_of_ms: int | None = None,
        lookahead_schedule_ms: int = 0,
    ) -> list[MacroEventRelease]:
        return self.gold_macro_store.load_macro_events(
            event_types,
            as_of_ms=as_of_ms,
            lookahead_schedule_ms=lookahead_schedule_ms,
        )

    def upsert_paxg_basis(
        self,
        rows: Sequence[PAXGBasisPoint],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        return self.gold_macro_store.upsert_paxg_basis(rows, allow_synthetic_in_test=allow_synthetic_in_test)

    def load_paxg_basis(
        self,
        venue_perp: str,
        timeframe_sec: int,
        *,
        as_of_ms: int | None = None,
        venue_spot: str | None = None,
        ref_tradfi_symbol: str = "GC=F",
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[PAXGBasisPoint]:
        return self.gold_macro_store.load_paxg_basis(
            venue_perp,
            int(timeframe_sec),
            as_of_ms=as_of_ms,
            venue_spot=venue_spot,
            ref_tradfi_symbol=ref_tradfi_symbol,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def upsert_cn_ashare_macro(
        self,
        rows: Sequence[CNAshareMacroPoint],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int:
        return self.cn_ashare_store.upsert_cn_macro(rows, allow_synthetic_in_test=allow_synthetic_in_test)

    def load_cn_ashare_macro(
        self,
        series_ids: Sequence[str] | None = None,
        *,
        as_of_ms: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[CNAshareMacroPoint]:
        return self.cn_ashare_store.load_cn_macro(
            series_ids=series_ids,
            as_of_ms=as_of_ms,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def record_trial(
        self,
        family_id: str,
        track_name: str,
        strategy_name: str,
        params_hash: str,
        metrics: dict[str, Any],
        verdict: str,
        *,
        dataset_sha256: str = "",
    ) -> str:
        """Append an immutable hypothesis-family trial record respecting `trial_ledger_mode` (OBS-07)."""
        now = utc_now_ms()
        nonce = uuid.uuid4().hex[:8]
        digest = sha256_obj(
            {
                "f": family_id,
                "t": track_name,
                "s": strategy_name,
                "p": params_hash,
                "ts": now,
                "n": nonce,
            }
        )[:12]
        prefix = "TR-RO" if self.trial_ledger_mode == "readonly" else ("TR-SB" if self.trial_ledger_mode == "sandbox" else "TR")
        trial_id = f"{prefix}-{digest}-{nonce[:6]}"
        metrics_json = json.dumps(metrics, sort_keys=True)

        if self.trial_ledger_mode == "readonly":
            self._readonly_trials.append(
                {
                    "trial_id": trial_id,
                    "created_ts_ms": now,
                    "family_id": family_id,
                    "track_name": track_name,
                    "strategy_name": strategy_name,
                    "params_hash": params_hash,
                    "dataset_sha256": dataset_sha256,
                    "metrics": dict(metrics),
                    "verdict": verdict,
                    "schema_version": "rea-001-v1",
                }
            )
            return trial_id

        target_conn = (
            self._sandbox_trial_conn
            if (self.trial_ledger_mode == "sandbox" and self._sandbox_trial_conn is not None)
            else self.catalog_conn
        )
        with transaction(target_conn):
            target_conn.execute(
                """
                INSERT INTO trial_ledger
                (trial_id, created_ts_ms, family_id, track_name, strategy_name,
                 params_hash, dataset_sha256, metrics_json, verdict, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'rea-001-v1')
                """,
                (
                    trial_id,
                    now,
                    family_id,
                    track_name,
                    strategy_name,
                    params_hash,
                    dataset_sha256,
                    metrics_json,
                    verdict,
                ),
            )
        return trial_id

    def get_official_family_trial_count(self, family_id: str) -> int:
        row = self.catalog_conn.fetch_one(
            "SELECT COUNT(*) AS cnt FROM trial_ledger WHERE family_id=?",
            (family_id,),
        )
        return int(row["cnt"]) if row is not None else 0

    def get_family_trial_count(self, family_id: str) -> int:
        """Cumulative trial count (official + sandbox/readonly) so DSR is NEVER under-penalized."""
        official_cnt = self.get_official_family_trial_count(family_id)
        if self.trial_ledger_mode == "sandbox" and self._sandbox_trial_conn is not None:
            sb_row = self._sandbox_trial_conn.fetch_one(
                "SELECT COUNT(*) AS cnt FROM trial_ledger WHERE family_id=?",
                (family_id,),
            )
            return official_cnt + (int(sb_row["cnt"]) if sb_row is not None else 0)
        if self.trial_ledger_mode == "readonly":
            ro_cnt = sum(1 for r in self._readonly_trials if r["family_id"] == family_id)
            return official_cnt + ro_cnt
        return official_cnt

    def list_family_trials(self, family_id: str) -> list[dict[str, Any]]:
        rows = self.catalog_conn.fetch_all(
            "SELECT trial_id, created_ts_ms, family_id, track_name, strategy_name, params_hash, "
            "dataset_sha256, metrics_json, verdict, schema_version "
            "FROM trial_ledger WHERE family_id=? ORDER BY created_ts_ms ASC, trial_id ASC",
            (family_id,),
        )
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d.pop("metrics_json"))
            out.append(d)
        if self.trial_ledger_mode == "sandbox" and self._sandbox_trial_conn is not None:
            sb_rows = self._sandbox_trial_conn.fetch_all(
                "SELECT trial_id, created_ts_ms, family_id, track_name, strategy_name, params_hash, "
                "dataset_sha256, metrics_json, verdict, schema_version "
                "FROM trial_ledger WHERE family_id=? ORDER BY created_ts_ms ASC, trial_id ASC",
                (family_id,),
            )
            for r in sb_rows:
                d = dict(r)
                d["metrics"] = json.loads(d.pop("metrics_json"))
                out.append(d)
        elif self.trial_ledger_mode == "readonly":
            for r in self._readonly_trials:
                if r["family_id"] == family_id:
                    out.append(dict(r))
        return out

    def save_factor_audit(self, track_name: str, dataset_sha256: str, payload: dict[str, Any]) -> str:
        self._assert_official_writable("save_factor_audit")
        now = utc_now_ms()
        nonce = uuid.uuid4().hex[:8]
        aid = f"FA-{sha256_obj({'t': track_name, 'd': dataset_sha256, 'ts': now, 'n': nonce})[:12]}"
        with transaction(self.catalog_conn):
            self.catalog_conn.execute(
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
        self._assert_official_writable("save_regime_snapshot")
        now = utc_now_ms()
        nonce = uuid.uuid4().hex[:8]
        sid = f"REG-{sha256_obj({'v': venue, 's': symbol, 'r': regime_label, 'ts': now, 'n': nonce})[:12]}"
        with transaction(self.catalog_conn):
            self.catalog_conn.execute(
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
        self._assert_official_writable("save_strategy_blueprint")
        now = utc_now_ms()
        nonce = uuid.uuid4().hex[:8]
        bid = f"BP-{sha256_obj({'t': track_name, 'a': archetype, 'd': dataset_sha256, 'ts': now, 'n': nonce})[:12]}"
        with transaction(self.catalog_conn):
            self.catalog_conn.execute(
                "INSERT OR REPLACE INTO strategy_blueprints "
                "(blueprint_id, created_ts_ms, track_name, archetype, verdict, dataset_sha256, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (bid, now, track_name, archetype, verdict, dataset_sha256, json.dumps(payload, sort_keys=True)),
            )
        return bid

    def audit_domain_dq(
        self,
        domain: str,
        venue: str,
        symbol: str,
        timeframe_sec: int,
        spec: AssetSpec,
        *,
        allow_synthetic_in_test: bool = False,
        min_bars: int = 40,
    ) -> MultiAssetDQReport:
        """Run DQ-MA-01..07 on a domain series and persist the DQ certificate to `catalog.sqlite` in official mode."""
        dom = normalize_domain(domain)
        bars = self.load_bars(dom, venue, symbol, int(timeframe_sec))
        label = f"{dom}:{venue}:{symbol}:{int(timeframe_sec)}s"
        report = audit_bars_quality(
            bars,
            spec.calendar,
            dataset_label=label,
            min_bars=min_bars,
            allow_synthetic_in_test=allow_synthetic_in_test,
        )
        # RB-D05: Never mutate catalog.sqlite in readonly/sandbox modes (enables running on 0444 snapshots!)
        if not self._official_lake_readonly:
            now = utc_now_ms()
            nonce = uuid.uuid4().hex[:8]
            cert_id = f"DQ-{sha256_obj({'l': label, 'sha': report.dataset_sha256, 'ts': now, 'n': nonce})[:12]}"
            with transaction(self.catalog_conn):
                self.catalog_conn.execute(
                    """
                    INSERT OR REPLACE INTO dq_certificates
                    (cert_id, created_ts_ms, domain, venue, symbol, timeframe_sec, dataset_sha256, passed, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cert_id,
                        now,
                        dom,
                        venue,
                        symbol,
                        int(timeframe_sec),
                        report.dataset_sha256,
                        1 if report.passed else 0,
                        json.dumps(report.to_dict(), sort_keys=True),
                    ),
                )
        return report

    def export_domain_partition(self, domain: str, fmt: str = "auto") -> dict[str, Any]:
        """Atomically export domain tables to immutable Parquet (if pyarrow) or deterministic `.jsonl.gz`."""
        self._assert_official_writable("export_domain_partition")
        dom = normalize_domain(domain)
        store = self.get_domain_store(dom)
        fmt_norm = fmt.strip().lower()
        if fmt_norm == "auto":
            resolved_fmt = "parquet" if _HAS_PYARROW else "jsonl.gz"
        elif fmt_norm == "parquet":
            if not _HAS_PYARROW:
                raise RuntimeError("pyarrow is required for fmt='parquet'; use fmt='auto' or fmt='jsonl.gz'")
            resolved_fmt = "parquet"
        elif fmt_norm in ("jsonl.gz", "jsonl"):
            resolved_fmt = "jsonl.gz"
        else:
            raise ValueError(f"Unsupported partition export format: {fmt!r}")

        if isinstance(store, (Web3DomainStore, GoldMacroDomainStore, CNAshareDomainStore)):
            tables = store.export_tables()
        else:
            tables = {f"{dom}_bars": list(store.iter_export_rows())}

        out_dir = self.parquet_dir / dom
        out_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now_ms()

        files_meta: list[dict[str, Any]] = []
        manifest_ids: list[str] = []
        sha256_by_table: dict[str, str] = {}
        total_rows = 0

        for table_name, rows in tables.items():
            ext = "parquet" if resolved_fmt == "parquet" else "jsonl.gz"
            target = out_dir / f"{table_name}.{ext}"
            if resolved_fmt == "parquet":
                row_count = _atomic_write_parquet(rows, target)
            else:
                row_count = _atomic_write_jsonl_gz(rows, target)
            digest = sha256_file(str(target))
            file_bytes = target.stat().st_size
            rel_path = str(target.relative_to(self.base_dir))

            ts_vals: list[int] = []
            provs: set[str] = set()
            for r in rows:
                ts = r.get("ts_event_ms") if "ts_event_ms" in r else r.get("ts_release_ms")
                if ts is not None:
                    ts_vals.append(int(ts))
                pv = r.get("provenance")
                if pv:
                    provs.add(str(pv))
            min_ts = min(ts_vals) if ts_vals else None
            max_ts = max(ts_vals) if ts_vals else None
            prov_str = "+".join(sorted(provs)) if provs else "EMPTY"
            manifest_id = f"MF-{sha256_obj({'d': dom, 't': table_name, 's': digest, 'ts': now, 'n': uuid.uuid4().hex[:6]})[:12]}"

            with transaction(self.catalog_conn):
                # Keep dataset_manifests 1:1 with the current file at (domain, rel_path) on disk
                self.catalog_conn.execute(
                    "DELETE FROM dataset_manifests WHERE domain=? AND rel_path=?",
                    (dom, rel_path),
                )
                self.catalog_conn.execute(
                    """
                    INSERT OR REPLACE INTO dataset_manifests
                    (manifest_id, created_ts_ms, domain, table_name, format, rel_path,
                     row_count, file_bytes, sha256, min_ts_ms, max_ts_ms, provenance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        manifest_id,
                        now,
                        dom,
                        table_name,
                        resolved_fmt,
                        rel_path,
                        row_count,
                        file_bytes,
                        digest,
                        min_ts,
                        max_ts,
                        prov_str,
                    ),
                )
            total_rows += row_count
            manifest_ids.append(manifest_id)
            sha256_by_table[table_name] = digest
            files_meta.append(
                {
                    "manifest_id": manifest_id,
                    "table_name": table_name,
                    "format": resolved_fmt,
                    "rel_path": rel_path,
                    "row_count": row_count,
                    "file_bytes": file_bytes,
                    "sha256": digest,
                    "min_ts_ms": min_ts,
                    "max_ts_ms": max_ts,
                    "provenance": prov_str,
                }
            )

        return {
            "domain": dom,
            "format": resolved_fmt,
            "total_rows": total_rows,
            "manifest_ids": manifest_ids,
            "sha256_by_table": sha256_by_table,
            "files": files_meta,
        }

    def list_dataset_manifests(self, domain: str | None = None) -> list[dict[str, Any]]:
        if domain is not None:
            dom = normalize_domain(domain)
            rows = self.catalog_conn.fetch_all(
                "SELECT * FROM dataset_manifests WHERE domain=? ORDER BY created_ts_ms DESC, table_name ASC",
                (dom,),
            )
        else:
            rows = self.catalog_conn.fetch_all(
                "SELECT * FROM dataset_manifests ORDER BY created_ts_ms DESC, domain ASC, table_name ASC"
            )
        return [dict(r) for r in rows]

    def query_domain_readonly_duckdb(self, domain: str, sql: str) -> list[dict[str, Any]]:
        """Execute a strictly read-only SELECT/WITH query over exported partitions in `:memory:` DuckDB.

        Never creates any on-disk `.duckdb` or `.duckdb.wal` file (ADR-0013 compliant).
        """
        dom = normalize_domain(domain)
        _ = self.get_domain_store(dom)
        safe_sql = validate_readonly_sql(sql)
        if not _HAS_DUCKDB or duckdb is None:
            raise RuntimeError(
                "duckdb is not installed; install requirements-research.txt or use router.load_bars()"
            )
        part_dir = self.parquet_dir / dom
        if not part_dir.exists():
            raise FileNotFoundError(
                f"No exported partition directory at {part_dir}; call export_domain_partition({dom!r}) first"
            )
        con = duckdb.connect(":memory:")
        try:
            # Prefer Parquet over jsonl.gz if both exist for the same table name; materialize
            # strictly into :memory: TEMP TABLEs BEFORE disabling external_access so user SQL
            # can query loaded partitions while external filesystem/network access stays blocked.
            selected_files: dict[str, tuple[str, Path]] = {}
            for p in sorted(part_dir.iterdir()):
                if not p.is_file():
                    continue
                if p.suffix == ".parquet":
                    tname = p.stem
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tname):
                        selected_files[tname] = ("parquet", p)
                elif p.name.endswith(".jsonl.gz"):
                    tname = p.name[: -len(".jsonl.gz")]
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tname) and tname not in selected_files:
                        selected_files[tname] = ("jsonl.gz", p)

            for tname, (kind, p) in sorted(selected_files.items()):
                escaped_path = str(p.resolve()).replace("'", "''")
                if kind == "parquet":
                    con.execute(
                        f"CREATE TEMP TABLE {tname} AS SELECT * FROM read_parquet('{escaped_path}')"
                    )
                else:
                    # Skip 0-row gzip files (empty ndjson) so non-empty sibling tables stay queryable
                    with gzip.open(p, "rt", encoding="utf-8") as fh:
                        first_line = fh.readline().strip()
                    if first_line:
                        con.execute(
                            f"CREATE TEMP TABLE {tname} AS SELECT * FROM read_json_auto('{escaped_path}')"
                        )
            # Lock down external filesystem/network access and freeze DuckDB settings before running user SQL
            con.execute("SET enable_external_access = false")
            con.execute("SET lock_configuration = true")
            cur = con.execute(safe_sql)
            cols = [desc[0] for desc in (cur.description or [])]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            con.close()

    def migrate_from_v3_sqlite(
        self,
        v3_sqlite_path: Path | str,
        *,
        allow_synthetic_in_test: bool = False,
    ) -> dict[str, Any]:
        """Idempotently migrate bars and metadata from legacy v3 SQLite strictly read-only (`?mode=ro`).

        Never modifies `v3_sqlite_path` (opens with `?mode=ro` + `PRAGMA query_only=ON`).
        """
        v3_path = Path(v3_sqlite_path).resolve()
        if not v3_path.exists():
            raise FileNotFoundError(f"v3 SQLite database not found: {v3_path}")

        uri = f"file:{v3_path}?mode=ro"
        src = sqlite3.connect(uri, uri=True, timeout=30.0)
        src.row_factory = sqlite3.Row
        try:
            src.execute("PRAGMA query_only=ON;")
            existing_tables = {
                str(r["name"])
                for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            migrated_bars: dict[str, int] = {"web3": 0, "gold_macro": 0, "skipped": 0}

            if "canonical_bars" in existing_tables:
                bar_cols = {
                    str(r["name"])
                    for r in src.execute("PRAGMA table_info(canonical_bars)").fetchall()
                }
                rows = src.execute("SELECT * FROM canonical_bars ORDER BY venue, symbol, timeframe_h, ts_event_ms").fetchall()
                web3_batch: list[CanonicalBar] = []
                gold_batch: list[CanonicalBar] = []
                for r in rows:
                    ac = str(r["asset_class"])
                    sym = str(r["symbol"])
                    tf_h = float(r["timeframe_h"])
                    tf_sec = (
                        int(r["timeframe_sec"])
                        if ("timeframe_sec" in bar_cols and int(r["timeframe_sec"] or 0) > 0)
                        else int(round(tf_h * 3600.0))
                    )
                    mult = (
                        float(r["contract_multiplier"])
                        if ("contract_multiplier" in bar_cols and float(r["contract_multiplier"] or 0.0) > 0.0)
                        else float(DEFAULT_CONTRACT_MULTIPLIERS.get(sym, 1.0))
                    )
                    stag = (
                        str(r["session_tag"])
                        if ("session_tag" in bar_cols and r["session_tag"])
                        else (
                            "ALL_24_7"
                            if ac in (ASSET_CLASS_CRYPTO_PERP, ASSET_CLASS_CRYPTO_SPOT, ASSET_CLASS_GOLD_PERP)
                            else classify_tradfi_session_tag(int(r["ts_event_ms"]), sym, "", tf_h)
                        )
                    )
                    src_fp = (
                        str(r["source_fingerprint"])
                        if ("source_fingerprint" in bar_cols and r["source_fingerprint"])
                        else f"migrated_v3:{r['venue']}:{sym}:{tf_sec}"
                    )
                    basis_bps = (
                        float(r["premium_basis_bps"])
                        if ("premium_basis_bps" in bar_cols and r["premium_basis_bps"] is not None)
                        else 0.0
                    )
                    bar = CanonicalBar(
                        asset_class=ac,
                        venue=str(r["venue"]),
                        symbol=sym,
                        timeframe_h=tf_h,
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
                        provenance=str(r["provenance"]),
                        source_fingerprint=src_fp,
                        timeframe_sec=tf_sec,
                        session_tag=stag,
                        premium_basis_bps=basis_bps,
                        contract_multiplier=mult,
                    )
                    target_dom = ASSET_CLASS_TO_DOMAIN.get(ac)
                    if target_dom == "web3":
                        web3_batch.append(bar)
                    elif target_dom == "gold_macro":
                        gold_batch.append(bar)
                    else:
                        migrated_bars["skipped"] += 1

                if web3_batch:
                    migrated_bars["web3"] = self.upsert_bars(
                        "web3", web3_batch, allow_synthetic_in_test=allow_synthetic_in_test
                    )
                if gold_batch:
                    migrated_bars["gold_macro"] = self.upsert_bars(
                        "gold_macro", gold_batch, allow_synthetic_in_test=allow_synthetic_in_test
                    )

            asset_count = 0
            if "asset_registry" in existing_tables:
                ar_rows = src.execute(
                    "SELECT venue, symbol, asset_class, spec_json, updated_ts_ms FROM asset_registry"
                ).fetchall()
                with transaction(self.catalog_conn):
                    for r in ar_rows:
                        sym = str(r["symbol"])
                        dom = ASSET_CLASS_TO_DOMAIN.get(str(r["asset_class"]), "web3")
                        raw_spec = str(r["spec_json"])
                        try:
                            parsed_spec = json.loads(raw_spec)
                        except (TypeError, ValueError):
                            parsed_spec = None
                        if isinstance(parsed_spec, dict) and "contract_multiplier" not in parsed_spec:
                            parsed_spec["contract_multiplier"] = float(
                                DEFAULT_CONTRACT_MULTIPLIERS.get(sym, 1.0)
                            )
                            spec_json_out = json.dumps(parsed_spec, sort_keys=True)
                        else:
                            spec_json_out = raw_spec
                        self.catalog_conn.execute(
                            """
                            INSERT OR IGNORE INTO asset_registry
                            (venue, symbol, asset_class, domain, spec_json, updated_ts_ms)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(r["venue"]),
                                sym,
                                str(r["asset_class"]),
                                dom,
                                spec_json_out,
                                int(r["updated_ts_ms"]),
                            ),
                        )
                        asset_count += 1

            fa_count = 0
            if "factor_audits" in existing_tables:
                fa_rows = src.execute(
                    "SELECT audit_id, created_ts_ms, track_name, dataset_sha256, payload_json FROM factor_audits"
                ).fetchall()
                with transaction(self.catalog_conn):
                    for r in fa_rows:
                        self.catalog_conn.execute(
                            """
                            INSERT OR IGNORE INTO factor_audits
                            (audit_id, created_ts_ms, track_name, dataset_sha256, payload_json)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                str(r["audit_id"]),
                                int(r["created_ts_ms"]),
                                str(r["track_name"]),
                                str(r["dataset_sha256"]),
                                str(r["payload_json"]),
                            ),
                        )
                        fa_count += 1

            rs_count = 0
            if "regime_snapshots" in existing_tables:
                rs_rows = src.execute(
                    "SELECT snapshot_id, created_ts_ms, venue, symbol, timeframe_h, regime_label, payload_json "
                    "FROM regime_snapshots"
                ).fetchall()
                with transaction(self.catalog_conn):
                    for r in rs_rows:
                        self.catalog_conn.execute(
                            """
                            INSERT OR IGNORE INTO regime_snapshots
                            (snapshot_id, created_ts_ms, venue, symbol, timeframe_h, regime_label, payload_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(r["snapshot_id"]),
                                int(r["created_ts_ms"]),
                                str(r["venue"]),
                                str(r["symbol"]),
                                float(r["timeframe_h"]),
                                str(r["regime_label"]),
                                str(r["payload_json"]),
                            ),
                        )
                        rs_count += 1

            bp_count = 0
            if "strategy_blueprints" in existing_tables:
                bp_rows = src.execute(
                    "SELECT blueprint_id, created_ts_ms, track_name, archetype, verdict, dataset_sha256, payload_json "
                    "FROM strategy_blueprints"
                ).fetchall()
                with transaction(self.catalog_conn):
                    for r in bp_rows:
                        self.catalog_conn.execute(
                            """
                            INSERT OR IGNORE INTO strategy_blueprints
                            (blueprint_id, created_ts_ms, track_name, archetype, verdict, dataset_sha256, payload_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(r["blueprint_id"]),
                                int(r["created_ts_ms"]),
                                str(r["track_name"]),
                                str(r["archetype"]),
                                str(r["verdict"]),
                                str(r["dataset_sha256"]),
                                str(r["payload_json"]),
                            ),
                        )
                        bp_count += 1

            return {
                "source_v3_path": str(v3_path),
                "bars_web3": migrated_bars["web3"],
                "bars_gold_macro": migrated_bars["gold_macro"],
                "bars_skipped": migrated_bars["skipped"],
                "asset_registry": asset_count,
                "factor_audits": fa_count,
                "regime_snapshots": rs_count,
                "strategy_blueprints": bp_count,
            }
        finally:
            src.close()

    def summary_stats(self) -> dict[str, Any]:
        return {
            "base_dir": str(self.base_dir),
            "trial_ledger_mode": self.trial_ledger_mode,
            "domains": {
                name: store.list_series()
                for name, store in sorted(self._domain_stores.items())
            },
            "web3_metrics_count": int(
                (self.web3_store.conn.fetch_one("SELECT COUNT(*) AS c FROM web3_derivatives_metrics") or {"c": 0})["c"]
            ),
            "web3_macro_count": int(
                (self.web3_store.conn.fetch_one("SELECT COUNT(*) AS c FROM web3_market_macro") or {"c": 0})["c"]
            ),
            "macro_yield_curve_count": int(
                (self.gold_macro_store.conn.fetch_one("SELECT COUNT(*) AS c FROM macro_yield_curve") or {"c": 0})["c"]
            ),
            "macro_events_count": int(
                (self.gold_macro_store.conn.fetch_one("SELECT COUNT(*) AS c FROM macro_events") or {"c": 0})["c"]
            ),
            "paxg_basis_count": int(
                (self.gold_macro_store.conn.fetch_one("SELECT COUNT(*) AS c FROM paxg_basis_arbitration") or {"c": 0})["c"]
            ),
            "cn_ashare_macro_count": int(
                (self.cn_ashare_store.conn.fetch_one("SELECT COUNT(*) AS c FROM cn_ashare_macro") or {"c": 0})["c"]
            ),
            "trial_ledger_count": int(
                (self.catalog_conn.fetch_one("SELECT COUNT(*) AS c FROM trial_ledger") or {"c": 0})["c"]
            ),
            "manifest_count": int(
                (self.catalog_conn.fetch_one("SELECT COUNT(*) AS c FROM dataset_manifests") or {"c": 0})["c"]
            ),
            "dq_certificates_count": int(
                (self.catalog_conn.fetch_one("SELECT COUNT(*) AS c FROM dq_certificates") or {"c": 0})["c"]
            ),
        }

    def close(self) -> None:
        seen_stores: set[int] = set()
        for store in self._domain_stores.values():
            if id(store) not in seen_stores:
                seen_stores.add(id(store))
                store.close()
        if self._sandbox_trial_conn is not None:
            self._sandbox_trial_conn.close()
            self._sandbox_trial_conn = None
        if self._sandbox_tmpdir is not None:
            self._sandbox_tmpdir.cleanup()
            self._sandbox_tmpdir = None
        self.catalog_conn.close()
