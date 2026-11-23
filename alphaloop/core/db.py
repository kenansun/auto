"""SQLite WAL connection factory compliant with SCHQ STD-34 Pillar 6."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def connect_wal(db_path: str | Path, timeout_sec: float = 30.0) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode, busy_timeout, and foreign keys."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=timeout_sec, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)};")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Atomic transaction helper with explicit rollback logging."""
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("SQLite transaction rolled back: %s", exc)
        raise
