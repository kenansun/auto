"""Thread-safe SQLite WAL connection factory compliant with SCHQ STD-34 Pillar 6.

Uses a `LockedConnection(sqlite3.Connection)` subclass that serializes BOTH reads and writes
via a per-connection `threading.RLock`, paired with explicit `BEGIN IMMEDIATE` write
transactions so concurrent `ThreadingHTTPServer` workers never interleave unlocked SELECTs
with active transactions or hit deferred-upgrade SQLITE_BUSY races (Red-Team #4).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class LockedConnection(sqlite3.Connection):
    """sqlite3.Connection subclass serializing execute/executemany/fetch_all/fetch_one with an RLock."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.lock = threading.RLock()

    def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:  # type: ignore[override]
        with self.lock:
            return super().execute(sql, parameters)

    def executemany(self, sql: str, seq_of_parameters: Any) -> sqlite3.Cursor:  # type: ignore[override]
        with self.lock:
            return super().executemany(sql, seq_of_parameters)

    def fetch_all(self, sql: str, parameters: Any = ()) -> list[sqlite3.Row]:
        """Execute query and materialize all rows while holding self.lock (Red-Team HIGH-R1)."""
        with self.lock:
            return super().execute(sql, parameters).fetchall()

    def fetch_one(self, sql: str, parameters: Any = ()) -> sqlite3.Row | None:
        """Execute query and materialize single row while holding self.lock (Red-Team HIGH-R1)."""
        with self.lock:
            return super().execute(sql, parameters).fetchone()


def get_conn_lock(conn: sqlite3.Connection) -> Any:
    lock = getattr(conn, "lock", None)
    if lock is not None and hasattr(lock, "__enter__"):
        return lock
    raise TypeError(f"Expected LockedConnection opened via connect_wal(), got {type(conn).__name__}")


def connect_wal(db_path: str | Path, timeout_sec: float = 30.0) -> LockedConnection:
    """Open a thread-safe SQLite WAL connection with autocommit isolation."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        timeout=timeout_sec,
        isolation_level=None,
        check_same_thread=False,
        factory=LockedConnection,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)};")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def connect_readonly(
    db_path: str | Path,
    timeout_sec: float = 30.0,
    *,
    immutable: bool = False,
) -> LockedConnection:
    """Open a thread-safe strictly read-only SQLite URI connection with busy_timeout.

    When `immutable=True` OR the target file is read-only (`0444` snapshot) without an active
    `-wal` sidecar, opens with `immutable=1` so SQLite never attempts to create `-shm` files
    inside a read-only snapshot directory (RB-D05).
    """
    import os
    import urllib.parse

    resolved = Path(db_path).resolve()
    wal_sidecar = Path(str(resolved) + "-wal")
    use_immutable = bool(immutable) or ((not os.access(resolved, os.W_OK)) and (not wal_sidecar.exists()))
    quoted = urllib.parse.quote(str(resolved), safe="/:")
    uri = f"file:{quoted}?mode=ro" + ("&immutable=1" if use_immutable else "")
    conn = sqlite3.connect(
        uri,
        uri=True,
        timeout=timeout_sec,
        isolation_level=None,
        check_same_thread=False,
        factory=LockedConnection,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON;")
    conn.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)};")
    return conn


def connect_memory(timeout_sec: float = 30.0) -> LockedConnection:
    """Open an isolated thread-safe in-memory SQLite connection (zero disk writes)."""
    conn = sqlite3.connect(
        ":memory:",
        timeout=timeout_sec,
        isolation_level=None,
        check_same_thread=False,
        factory=LockedConnection,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Thread-safe atomic write transaction using BEGIN IMMEDIATE with rollback logging."""
    lock = get_conn_lock(conn)
    with lock:
        nested = conn.in_transaction
        if not nested:
            conn.execute("BEGIN IMMEDIATE;")
        try:
            yield conn
            if not nested:
                conn.execute("COMMIT;")
        except Exception as exc:
            if not nested and conn.in_transaction:
                conn.execute("ROLLBACK;")
            logger.error("SQLite transaction rolled back: %s", exc)
            raise
