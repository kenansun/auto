"""Deterministic pure-stdlib fixtures, SQLite WAL helpers, and opaque-box contract helpers.

All Platform v4 module imports are strictly lazy (performed inside helper calls, NEVER at
module import time) so `run_e2e.py --list` discovers all test IDs cleanly before parallel
implementation milestones complete, while executing tests genuinely fails with ImportError
or AssertionError until target contracts exist.

Zero third-party dependencies (pure Python 3.12 stdlib only) and 100% STD-34 compliant.
"""

from __future__ import annotations

import atexit
from collections import deque
from contextlib import contextmanager
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from types import UnionType
import typing
from typing import Any, Callable, Iterator, Sequence

# ---------------------------------------------------------------------------
# Canonical Paths & sys.path Bootstrapping
# ---------------------------------------------------------------------------

E2E_SUITE_DIR = Path(__file__).resolve().parent
AUTO_REPO_ROOT = E2E_SUITE_DIR.parents[1]
WORKSPACE_PARENT = AUTO_REPO_ROOT.parent
SCHQ_REPO_ROOT = WORKSPACE_PARENT / "Strategic-Command-Headquarters"

if str(AUTO_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTO_REPO_ROOT))

AUDIT_SCRIPT_PATH = SCHQ_REPO_ROOT / "scripts" / "audit_service_readiness.py"
V4_LAKE_DIR = AUTO_REPO_ROOT / "data" / "platform_v4_lake"
V3_LAKE_PATH = AUTO_REPO_ROOT / "data" / "platform_v3_lake.sqlite"
V4_RUNNER_PATH = AUTO_REPO_ROOT / "scripts" / "run_strategy_platform_v4.py"
REQUIREMENTS_RESEARCH_PATH = AUTO_REPO_ROOT / "requirements-research.txt"


def resolve_v4_lake_dir() -> Path:
    """Resolve Platform v4 data lake path, honoring PLATFORM_V4_LAKE_DIR when set."""
    env_dir = os.environ.get("PLATFORM_V4_LAKE_DIR", "").strip()
    return Path(env_dir) if env_dir else V4_LAKE_DIR


def has_real_v4_lake(lake_dir: Path | None = None) -> bool:
    """Return True iff all three Option B SQLite databases exist under the resolved lake dir."""
    root = resolve_v4_lake_dir() if lake_dir is None else Path(lake_dir)
    return (
        (root / "catalog.sqlite").is_file()
        and (root / "domains" / "web3_lake.sqlite").is_file()
        and (root / "domains" / "gold_macro_lake.sqlite").is_file()
    )


def resolve_v3_lake_path() -> Path:
    """Resolve legacy Platform v3 SQLite path, honoring PLATFORM_V3_LAKE_PATH when set."""
    env_path = os.environ.get("PLATFORM_V3_LAKE_PATH", "").strip()
    return Path(env_path) if env_path else V3_LAKE_PATH

ACTIVE_WORKSPACE_REPOS: tuple[Path, ...] = (
    AUTO_REPO_ROOT,
    SCHQ_REPO_ROOT,
    WORKSPACE_PARENT / "agent_wallet_test",
    WORKSPACE_PARENT / "arb-executor",
    WORKSPACE_PARENT / "auto_trade_bot",
    WORKSPACE_PARENT / "multi_market_analyzer",
    WORKSPACE_PARENT / "web3-opportunity-brain",
)

# Exact Unicode paths for all 9 F-23 SCHQ deliverables (U+FF5C '｜', U+FF1A '：', U+FF08/FF09 '（）')
SCHQ_DELIVERABLES: dict[str, Path] = {
    "ticket": (
        SCHQ_REPO_ROOT
        / "02_项目组合与任务"
        / "CC-20261123-004｜工单：策略研发平台v4升级（数据底座分域+多品种策略解耦+多服务器架构与未来规划）_20261123.md"
    ),
    "architecture": (
        SCHQ_REPO_ROOT
        / "docs"
        / "architecture"
        / "MULTI_ASSET_STRATEGY_RD_PLATFORM_V4_AND_ROADMAP_20261123.md"
    ),
    "empirical_report": (
        SCHQ_REPO_ROOT
        / "docs"
        / "evidence"
        / "STRATEGY_PLATFORM_V4_EMPIRICAL_AUDIT_AND_AB_REPORT_20261123.md"
    ),
    "own_repos": (
        SCHQ_REPO_ROOT
        / "05_ASSET_REGISTRY｜统一资产与代币管理"
        / "01_权威资产台账"
        / "OWN_REPOS_REGISTRY｜自有仓库索引.md"
    ),
    "shared_services": (
        SCHQ_REPO_ROOT
        / "05_ASSET_REGISTRY｜统一资产与代币管理"
        / "01_权威资产台账"
        / "SHARED_SERVICES_AND_SUBSCRIPTION_CATALOG｜公共服务与订阅接入总目录.md"
    ),
    "oss_ledger": (
        SCHQ_REPO_ROOT
        / "05_ASSET_REGISTRY｜统一资产与代币管理"
        / "01_权威资产台账"
        / "OPEN_SOURCE_RESEARCH_LEDGER｜开源仓库调研台账.md"
    ),
    "change_ledger": SCHQ_REPO_ROOT / "governance" / "CHANGE_NOTIFICATION_LEDGER.md",
    "kanban": (
        SCHQ_REPO_ROOT
        / "02_项目组合与任务"
        / "CC-20261113-003｜平台研发进度统一看板_20261113.md"
    ),
    "maintenance": SCHQ_REPO_ROOT / "MAINTENANCE.md",
}

# ---------------------------------------------------------------------------
# Deterministic UTC Epoch Millisecond Constants
# ---------------------------------------------------------------------------

HOUR_MS = 3_600_000
DAY_MS = 86_400_000


def utc_ms(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> int:
    """Compute exact integer UTC epoch milliseconds."""
    dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    return int(round(dt.timestamp() * 1000.0))


BASE_TS_MS = utc_ms(2024, 1, 1, 0, 0)
HYPE_LISTING_MS = utc_ms(2024, 12, 5, 0, 0)
DEC_2026_NFP_MS = utc_ms(2026, 12, 4, 13, 30)
DEC_2026_FOMC_MS = utc_ms(2026, 12, 9, 19, 0)
DEC_2026_CPI_MS = utc_ms(2026, 12, 10, 13, 30)

# Verified calendar: 2026-11-18 Wed, 2026-11-20 Fri, 2026-11-21 Sat, 2026-11-22 Sun, 2026-11-23 Mon
WED_2026_11_18_1200_MS = utc_ms(2026, 11, 18, 12, 0)
FRI_2026_11_20_1800_MS = utc_ms(2026, 11, 20, 18, 0)
FRI_2026_11_20_2200_MS = utc_ms(2026, 11, 20, 22, 0)
SAT_2026_11_21_1200_MS = utc_ms(2026, 11, 21, 12, 0)
SAT_2026_11_21_1400_MS = utc_ms(2026, 11, 21, 14, 0)
SUN_2026_11_22_1400_MS = utc_ms(2026, 11, 22, 14, 0)
SUN_2026_11_22_2200_MS = utc_ms(2026, 11, 22, 22, 0)
MON_2026_11_23_0800_MS = utc_ms(2026, 11, 23, 8, 0)

# ---------------------------------------------------------------------------
# Centralized STD-34 Compliant SQLite Helpers (ONLY file calling sqlite3.connect)
# ---------------------------------------------------------------------------


def open_sqlite_rw(db_path: Path | str) -> sqlite3.Connection:
    """Open a writable SQLite connection with explicit timeout and WAL mode."""
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def open_sqlite_ro(db_path: Path | str) -> sqlite3.Connection:
    """Open a strictly read-only SQLite connection without mutating journal mode."""
    resolved = Path(db_path).resolve()
    uri = f"file:{resolved}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def copy_sqlite_readonly_snapshot(src_path: Path | str, dst_path: Path | str) -> Path:
    """Safely copy an existing SQLite DB via read-only backup API into an isolated file."""
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_conn = open_sqlite_ro(src_path)
    try:
        dst_conn = open_sqlite_rw(dst)
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()
    return dst


def query_explain_plan(db_path: Path | str, sql: str, params: Sequence[Any] = ()) -> str:
    """Return formatted EXPLAIN QUERY PLAN text from a read-only SQLite connection."""
    conn = open_sqlite_ro(db_path)
    try:
        rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}", tuple(params)).fetchall()
        return "\n".join(" | ".join(str(col) for col in row) for row in rows)
    finally:
        conn.close()


def discover_bar_table(
    db_path: Path | str,
    *,
    preferred: Sequence[str] = ("canonical_bars", "web3_bars", "tradfi_bars"),
) -> str:
    """Dynamically resolve the bar table name in a domain SQLite DB via sqlite_master."""
    conn = open_sqlite_ro(db_path)
    try:
        all_tables = [
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
    finally:
        conn.close()
    for cand in preferred:
        if cand in all_tables:
            return cand
    bar_tables = [t for t in all_tables if "bar" in t.lower()]
    if not bar_tables:
        raise AssertionError(f"No bars table found in {Path(db_path).name}; tables={all_tables}")
    chosen = bar_tables[0]
    if not chosen.replace("_", "").isalnum():
        raise AssertionError(f"Unexpected SQLite bar table identifier: {chosen!r}")
    return chosen


def resolve_paxg_bars(
    router: Any,
    *,
    router_mod: Any | None = None,
    timeframe_sec: int = 43200,
    venues: Sequence[str] = ("hyperliquid", "binance_um", "binance_spot", "gate"),
    min_bars: int = 1,
) -> list[Any]:
    """Load PAXG bars checking gold_macro domain first, falling back to web3 domain."""
    gold_dom = resolve_gold_domain(router_mod)
    web3_dom = resolve_web3_domain(router_mod)
    for dom in (gold_dom, web3_dom):
        for venue in venues:
            cand = router.load_bars(dom, venue, "PAXG", int(timeframe_sec))
            if len(cand) >= min_bars:
                return list(cand)
    return []


def create_synthetic_v3_lake_db(
    db_path: Path | str,
    *,
    n_web3_bars: int = 80,
    n_gold_bars: int = 80,
    provenance: str = "REAL_EXCHANGE_API",
) -> Path:
    """Create a temp v3-schema SQLite DB matching auto/data/platform_v3_lake.sqlite."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = open_sqlite_rw(db_file)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS asset_registry (
                venue TEXT,
                symbol TEXT,
                asset_class TEXT,
                spec_json TEXT,
                updated_ts_ms INTEGER,
                PRIMARY KEY (venue, symbol)
            );
            CREATE TABLE IF NOT EXISTS canonical_bars (
                asset_class TEXT,
                venue TEXT,
                symbol TEXT,
                timeframe_h REAL,
                ts_event_ms INTEGER,
                ts_avail_ms INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume_usd REAL,
                carry_rate_period REAL,
                carry_interval_h REAL,
                spread_bps REAL,
                provenance TEXT,
                PRIMARY KEY (venue, symbol, timeframe_h, ts_event_ms)
            );
            CREATE INDEX IF NOT EXISTS idx_canonical_pit
                ON canonical_bars (venue, symbol, timeframe_h, ts_avail_ms);
            CREATE TABLE IF NOT EXISTS factor_audits (
                audit_id TEXT PRIMARY KEY,
                created_ts_ms INTEGER,
                track_name TEXT,
                dataset_sha256 TEXT,
                payload_json TEXT
            );
            CREATE TABLE IF NOT EXISTS regime_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                created_ts_ms INTEGER,
                venue TEXT,
                symbol TEXT,
                timeframe_h REAL,
                regime_label TEXT,
                payload_json TEXT
            );
            CREATE TABLE IF NOT EXISTS strategy_blueprints (
                blueprint_id TEXT PRIMARY KEY,
                created_ts_ms INTEGER,
                track_name TEXT,
                archetype TEXT,
                verdict TEXT,
                dataset_sha256 TEXT,
                payload_json TEXT
            );
            """
        )
        start_ms = BASE_TS_MS
        conn.execute(
            "INSERT OR REPLACE INTO asset_registry VALUES (?, ?, ?, ?, ?)",
            ("hyperliquid", "BTC", "CRYPTO_PERP", json.dumps({"symbol": "BTC"}), start_ms),
        )
        conn.execute(
            "INSERT OR REPLACE INTO asset_registry VALUES (?, ?, ?, ?, ?)",
            ("yahoo", "GC=F", "GOLD_TRADFI", json.dumps({"symbol": "GC=F"}), start_ms),
        )
        web3_rows = []
        price = 42000.0
        for i in range(n_web3_bars):
            ts_ev = start_ms + i * 12 * HOUR_MS
            ts_av = ts_ev + 12 * HOUR_MS
            nxt = price * (1.0 + 0.002 * math.sin(i / 5.0) + 0.001)
            hi = max(price, nxt) * 1.005
            lo = min(price, nxt) * 0.995
            web3_rows.append(
                (
                    "CRYPTO_PERP",
                    "hyperliquid",
                    "BTC",
                    12.0,
                    ts_ev,
                    ts_av,
                    price,
                    hi,
                    lo,
                    nxt,
                    25_000_000.0,
                    0.00012,
                    1.0,
                    1.5,
                    provenance,
                )
            )
            price = nxt
        conn.executemany(
            "INSERT OR REPLACE INTO canonical_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            web3_rows,
        )

        gold_rows = []
        gpx = 2050.0
        for i in range(n_gold_bars):
            ts_ev = start_ms + i * DAY_MS
            ts_av = ts_ev + DAY_MS
            nxt = gpx * (1.0 + 0.0015 * math.cos(i / 7.0) + 0.0005)
            hi = max(gpx, nxt) * 1.004
            lo = min(gpx, nxt) * 0.996
            gold_rows.append(
                (
                    "GOLD_TRADFI",
                    "yahoo",
                    "GC=F",
                    24.0,
                    ts_ev,
                    ts_av,
                    gpx,
                    hi,
                    lo,
                    nxt,
                    150_000_000.0,
                    0.0,
                    24.0,
                    2.0,
                    provenance,
                )
            )
            gpx = nxt
        conn.executemany(
            "INSERT OR REPLACE INTO canonical_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            gold_rows,
        )
        conn.commit()
    finally:
        conn.close()
    return db_file


# ---------------------------------------------------------------------------
# Sandboxes, Strict JSON, Subprocess & STD-34 Audit Helpers
# ---------------------------------------------------------------------------


@contextmanager
def temp_sandbox(prefix: str = "e2e_v4_") -> Iterator[Path]:
    """Isolated temporary directory sandbox (never uses dot-prefixed paths)."""
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        yield Path(tmp)


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"Non-standard JSON constant rejected: {token!r}")


def strict_json_loads(raw: str) -> Any:
    """Parse strict RFC 8259 JSON, rejecting NaN, Infinity, and -Infinity."""
    return json.loads(raw, parse_constant=_reject_json_constant)


def strict_json_dumps(obj: Any) -> str:
    """Serialize strict RFC 8259 JSON, rejecting NaN, Infinity, and -Infinity."""
    return json.dumps(obj, allow_nan=False, sort_keys=True)


def sha256_file(path: Path | str) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_audit_script(
    target_path: Path | str,
    *,
    strict: bool = True,
    include_tests: bool = False,
    timeout_s: float = 60.0,
) -> tuple[int, dict[str, Any], str]:
    """Execute SCHQ audit_service_readiness.py against a single target path."""
    cmd = [sys.executable, str(AUDIT_SCRIPT_PATH), "--json"]
    if strict:
        cmd.append("--strict")
    if include_tests:
        cmd.append("--include-tests")
    cmd.append(str(target_path))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    payload: dict[str, Any] = {}
    if proc.stdout.strip():
        payload = strict_json_loads(proc.stdout)
    return proc.returncode, payload, proc.stderr


# ---------------------------------------------------------------------------
# Pure-Stdlib Quantitative Math & Deterministic Generators
# ---------------------------------------------------------------------------


def sample_pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pure-stdlib Pearson correlation coefficient."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    cov = 0.0
    vx = 0.0
    vy = 0.0
    for i in range(n):
        dx = xs[i] - mx
        dy = ys[i] - my
        cov += dx * dy
        vx += dx * dx
        vy += dy * dy
    denom = math.sqrt(vx * vy)
    if denom <= 1e-18:
        return 0.0
    return cov / denom


def annualized_vol(returns: Sequence[float], periods_per_year: float) -> float:
    """Pure-stdlib sample annualized volatility (decimal, e.g. 0.15 = 15%)."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    return math.sqrt(max(var, 0.0) * periods_per_year)


def generate_correlated_return_matrix(
    n_assets: int,
    n_obs: int,
    rho: float,
    *,
    annual_vol: float = 0.30,
    periods_per_year: float = 365.0,
    drift_annual: float = 0.0,
    seed: int = 42,
) -> list[list[float]]:
    """Generate deterministic multi-asset returns with exact common-factor correlation rho.

    Uses r_i(t) = mu + sigma * (sqrt(rho) * Z(t) + sqrt(1 - rho) * eps_i(t))
    where Z and eps_i are standardized independent Gaussian series (pure stdlib).
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    rng = random.Random(seed)
    sigma = annual_vol / math.sqrt(periods_per_year)
    mu = drift_annual / periods_per_year

    def _std_normal_vec() -> list[float]:
        raw = [rng.gauss(0.0, 1.0) for _ in range(n_obs)]
        m = sum(raw) / n_obs
        centered = [x - m for x in raw]
        sd = math.sqrt(sum(x * x for x in centered) / max(n_obs - 1, 1))
        if sd <= 1e-12:
            return [0.0] * n_obs
        return [x / sd for x in centered]

    common = _std_normal_vec()
    a = math.sqrt(rho)
    b = math.sqrt(max(1.0 - rho, 0.0))
    matrix: list[list[float]] = []
    for _ in range(n_assets):
        eps = _std_normal_vec()
        # Orthogonalize eps against common factor so sample corr matches rho tightly
        dot_ce = sum(common[t] * eps[t] for t in range(n_obs))
        dot_cc = sum(common[t] * common[t] for t in range(n_obs))
        proj = dot_ce / dot_cc if dot_cc > 1e-12 else 0.0
        orth = [eps[t] - proj * common[t] for t in range(n_obs)]
        sd_o = math.sqrt(sum(x * x for x in orth) / max(n_obs - 1, 1))
        orth_std = [x / sd_o for x in orth] if sd_o > 1e-12 else [0.0] * n_obs
        series = [mu + sigma * (a * common[t] + b * orth_std[t]) for t in range(n_obs)]
        matrix.append(series)
    return matrix


def closes_from_returns(start_price: float, returns: Sequence[float]) -> list[float]:
    """Convert period returns into strictly positive close prices."""
    prices = [float(start_price)]
    cur = float(start_price)
    for r in returns:
        cur = max(cur * (1.0 + float(r)), 1e-4)
        prices.append(cur)
    return prices


def build_trending_closes(
    n_bars: int,
    *,
    start_price: float = 100.0,
    step_ret: float = 0.004,
    noise_std: float = 0.005,
    seed: int = 7,
) -> list[float]:
    """Build deterministic trending close price sequence."""
    rng = random.Random(seed)
    prices = [float(start_price)]
    cur = float(start_price)
    for _ in range(max(n_bars - 1, 0)):
        r = step_ret + rng.gauss(0.0, noise_std)
        cur = max(cur * (1.0 + r), 1e-3)
        prices.append(cur)
    return prices


# ---------------------------------------------------------------------------
# Lazy Target Importers & Non-Weakening Contract Helpers (A-01..A-10)
# ---------------------------------------------------------------------------

_MISSING = object()


def import_provenance() -> tuple[str, str, str]:
    mod = importlib.import_module("alphaloop.core.provenance")
    return mod.PROVENANCE_REAL, mod.PROVENANCE_PAPER, mod.PROVENANCE_SYNTHETIC


def import_assets() -> Any:
    return importlib.import_module("alphaloop.platform.assets")


def import_data_lake() -> Any:
    return importlib.import_module("alphaloop.platform.data_lake")


def import_df_models() -> Any:
    return importlib.import_module("alphaloop.platform.data_foundation.models")


def import_df_router() -> Any:
    return importlib.import_module("alphaloop.platform.data_foundation.router")


def import_resample_bars() -> Callable[..., list[Any]]:
    """Locate public resample_bars contract (Ambiguity A-05)."""
    candidates = (
        ("alphaloop.platform.data_foundation.router", "resample_bars"),
        ("alphaloop.platform.data_foundation", "resample_bars"),
        ("alphaloop.platform.data_foundation.models", "resample_bars"),
    )
    errors: list[str] = []
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:
            errors.append(f"{mod_name}: {exc}")
            continue
        fn = getattr(mod, attr, None)
        if callable(fn):
            return fn
        errors.append(f"{mod_name}.{attr}: missing")
    raise ImportError("Could not import resample_bars from data_foundation: " + "; ".join(errors))


def import_collectors_web3() -> Any:
    return importlib.import_module("alphaloop.platform.data_foundation.collectors_web3")


def import_collectors_gold_macro() -> Any:
    return importlib.import_module("alphaloop.platform.data_foundation.collectors_gold_macro")


def import_core_interfaces() -> Any:
    return importlib.import_module("alphaloop.platform.core_engine.interfaces")


def import_risk_allocator_mod() -> Any:
    return importlib.import_module("alphaloop.platform.core_engine.risk_allocator")


def import_accounting_mod() -> Any:
    return importlib.import_module("alphaloop.platform.core_engine.accounting")


def import_core_validation_mod() -> Any:
    return importlib.import_module("alphaloop.platform.core_engine.validation")


def import_portfolio_combiner_mod() -> Any:
    return importlib.import_module("alphaloop.platform.core_engine.portfolio_combiner")


def import_factors_mod() -> Any:
    return importlib.import_module("alphaloop.platform.factors")


def import_regime_inference_mod() -> Any:
    return importlib.import_module("alphaloop.platform.regime_inference")


def import_v3_validation_mod() -> Any:
    return importlib.import_module("alphaloop.platform.validation")


def import_strategy_customizer_mod() -> Any:
    return importlib.import_module("alphaloop.platform.strategy_customizer")


def import_web3_regime_mod() -> Any:
    return importlib.import_module("alphaloop.platform.strategies.web3.regime_engine")


def import_w1_mod() -> Any:
    return importlib.import_module("alphaloop.platform.strategies.web3.xs_mom_carry")


def import_w2_mod() -> Any:
    return importlib.import_module("alphaloop.platform.strategies.web3.adaptive_trend")


def import_w3_mod() -> Any:
    return importlib.import_module("alphaloop.platform.strategies.web3.regime_carry")


def import_gold_macro_mod() -> Any:
    return importlib.import_module("alphaloop.platform.strategies.gold.macro_engine")


def import_g1_mod() -> Any:
    return importlib.import_module("alphaloop.platform.strategies.gold.tradfi_macro_core")


def import_g2_mod() -> Any:
    return importlib.import_module("alphaloop.platform.strategies.gold.session_event_4h")


def import_g3_mod() -> Any:
    return importlib.import_module("alphaloop.platform.strategies.gold.paxg_bridge")


def import_ashare_contract_mod() -> Any:
    return importlib.import_module("alphaloop.platform.strategies.ashare_contract")


def close_router(router: Any) -> None:
    """Cleanly close DataFoundationRouter SQLite handles (Ambiguity A-10)."""
    close_fn = getattr(router, "close", None)
    if callable(close_fn):
        close_fn()


def resolve_gold_domain(router_mod: Any | None = None) -> str:
    """Resolve Gold domain identifier (Ambiguity A-03: defaults strictly to 'gold_macro')."""
    if router_mod is not None:
        for attr in ("DOMAIN_GOLD_MACRO", "DOMAIN_GOLD"):
            val = getattr(router_mod, attr, None)
            if isinstance(val, str) and val:
                return val
    return "gold_macro"


def resolve_web3_domain(router_mod: Any | None = None) -> str:
    """Resolve Web3 domain identifier (defaults strictly to 'web3')."""
    if router_mod is not None:
        val = getattr(router_mod, "DOMAIN_WEB3", None)
        if isinstance(val, str) and val:
            return val
    return "web3"


def make_bar(
    *,
    venue: str,
    symbol: str,
    asset_class: str,
    timeframe_h: float,
    ts_event_ms: int,
    ts_avail_ms: int,
    open: float,
    high: float,
    low: float,
    close: float,
    volume_usd: float,
    carry_rate_period: float = 0.0,
    carry_interval_h: float = 1.0,
    spread_bps: float = 0.0,
    provenance: str | None = None,
    source_fingerprint: str = "e2e_fp_sha256",
    timeframe_sec: int | None = None,
    session_tag: str = "ALL_24_7",
    premium_basis_bps: float = 0.0,
    contract_multiplier: float = 1.0,
    use_v3_model: bool = False,
) -> Any:
    """Construct CanonicalBar strictly via keyword arguments (Ambiguity A-01)."""
    prov_real, _, _ = import_provenance()
    prov = prov_real if provenance is None else provenance
    if use_v3_model:
        cls = import_data_lake().CanonicalBar
    else:
        cls = import_df_models().CanonicalBar

    kwargs: dict[str, Any] = {
        "venue": venue,
        "symbol": symbol,
        "asset_class": asset_class,
        "timeframe_h": float(timeframe_h),
        "ts_event_ms": int(ts_event_ms),
        "ts_avail_ms": int(ts_avail_ms),
        "open": float(open),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume_usd": float(volume_usd),
        "carry_rate_period": float(carry_rate_period),
        "carry_interval_h": float(carry_interval_h),
        "spread_bps": float(spread_bps),
        "provenance": prov,
    }
    if not use_v3_model:
        sig = inspect.signature(cls)
        optional_v4: dict[str, Any] = {
            "source_fingerprint": source_fingerprint,
            "session_tag": session_tag,
            "premium_basis_bps": float(premium_basis_bps),
            "contract_multiplier": float(contract_multiplier),
        }
        if timeframe_sec is not None:
            optional_v4["timeframe_sec"] = int(timeframe_sec)
        for k, v in optional_v4.items():
            if k in sig.parameters:
                kwargs[k] = v
    return cls(**kwargs)


def build_bar_series(
    *,
    symbol: str,
    venue: str,
    asset_class: str,
    timeframe_h: float,
    start_ms: int,
    closes: Sequence[float],
    opens: Sequence[float] | None = None,
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
    volume_usd: float = 10_000_000.0,
    carry_rates: Sequence[float] | None = None,
    carry_interval_h: float = 1.0,
    spread_bps: float = 1.5,
    provenance: str | None = None,
    session_tag: str = "ALL_24_7",
    contract_multiplier: float = 1.0,
    premium_basis_bps: float = 0.0,
    step_ms: int | None = None,
    use_v3_model: bool = False,
) -> list[Any]:
    """Build a deterministic causal sequence of CanonicalBar objects."""
    tf_ms = int(round(timeframe_h * HOUR_MS)) if step_ms is None else int(step_ms)
    bars: list[Any] = []
    for i, c_val in enumerate(closes):
        c = float(c_val)
        o = float(opens[i]) if opens is not None else (float(closes[i - 1]) if i > 0 else c)
        h = float(highs[i]) if highs is not None else max(o, c) * 1.003
        lo = float(lows[i]) if lows is not None else min(o, c) * 0.997
        ts_ev = int(start_ms + i * tf_ms)
        ts_av = int(ts_ev + int(round(timeframe_h * HOUR_MS)))
        cr = float(carry_rates[i]) if carry_rates is not None else 0.0
        bars.append(
            make_bar(
                venue=venue,
                symbol=symbol,
                asset_class=asset_class,
                timeframe_h=timeframe_h,
                ts_event_ms=ts_ev,
                ts_avail_ms=ts_av,
                open=o,
                high=h,
                low=lo,
                close=c,
                volume_usd=volume_usd,
                carry_rate_period=cr,
                carry_interval_h=carry_interval_h,
                spread_bps=spread_bps,
                provenance=provenance,
                session_tag=session_tag,
                contract_multiplier=contract_multiplier,
                premium_basis_bps=premium_basis_bps,
                use_v3_model=use_v3_model,
            )
        )
    return bars


def build_ashare_multiday_1h_bars(
    *,
    symbol: str = "600519.SH",
    start_year: int = 2026,
    start_month: int = 11,
    start_day: int = 23,
    n_days: int = 3,
    daily_close_paths: Sequence[Sequence[float]] | None = None,
    base_price: float = 100.0,
) -> list[Any]:
    """Build intraday 1h CN A-Share bars across CST trading sessions (09:30..14:30 CST).

    CST = UTC+8, so 4 intraday 1h bars per day start at 01:30, 02:30, 05:00, 06:00 UTC.
    """
    assets = import_assets()
    bars: list[Any] = []
    utc_hours = ((1, 30), (2, 30), (5, 0), (6, 0))
    prev_close = float(base_price)
    for d in range(n_days):
        day_ts_base = utc_ms(start_year, start_month, start_day, 0, 0) + d * DAY_MS
        dt_day = datetime.fromtimestamp(day_ts_base / 1000.0, tz=timezone.utc)
        for idx, (hh, mm) in enumerate(utc_hours):
            ts_ev = utc_ms(dt_day.year, dt_day.month, dt_day.day, hh, mm)
            ts_av = ts_ev + HOUR_MS
            if daily_close_paths is not None:
                c = float(daily_close_paths[d][idx])
            else:
                c = prev_close * (1.0 + 0.003 * (1 if (d + idx) % 2 == 0 else -0.3))
            o = prev_close
            hi = max(o, c) * 1.002
            lo = min(o, c) * 0.998
            bars.append(
                make_bar(
                    venue="sse",
                    symbol=symbol,
                    asset_class=assets.ASSET_CLASS_CN_A_SHARE,
                    timeframe_h=1.0,
                    ts_event_ms=ts_ev,
                    ts_avail_ms=ts_av,
                    open=o,
                    high=hi,
                    low=lo,
                    close=c,
                    volume_usd=50_000_000.0,
                    session_tag="CN_A_SHARE",
                )
            )
            prev_close = c
    return bars


def build_gold_seven_driver_fixture(
    *,
    n_days: int = 160,
    start_ms: int = BASE_TS_MS,
    real_10y_start: float = 2.40,
    real_10y_end: float = 1.40,
    dxy_step: float = -0.0008,
    ief_step: float = 0.0009,
    tip_step: float = 0.0010,
    si_step: float = 0.0022,
    gdx_step: float = 0.0024,
    gvz_level: float = 16.0,
    vix_level: float = 16.0,
    gold_step: float = 0.0016,
    seed: int = 19,
) -> tuple[dict[str, list[Any]], list[Any]]:
    """Build deterministic 7-driver Gold + TradFi macro fixture and US Treasury TIPS yields."""
    assets = import_assets()
    models = import_df_models()
    prov_real, _, _ = import_provenance()

    def _series(sym: str, ac: str, start_p: float, step: float, s_off: int, mult: float = 1.0) -> list[Any]:
        closes = build_trending_closes(n_days, start_price=start_p, step_ret=step, noise_std=0.003, seed=seed + s_off)
        return build_bar_series(
            symbol=sym,
            venue="yahoo",
            asset_class=ac,
            timeframe_h=24.0,
            start_ms=start_ms,
            closes=closes,
            volume_usd=200_000_000.0,
            contract_multiplier=mult,
            session_tag="TRADFI_DAILY",
        )

    bars: dict[str, list[Any]] = {
        "GC=F": _series("GC=F", assets.ASSET_CLASS_GOLD_TRADFI, 2050.0, gold_step, 1, mult=100.0),
        "GLD": _series("GLD", assets.ASSET_CLASS_GOLD_TRADFI, 190.0, gold_step, 2, mult=1.0),
        "SI=F": _series("SI=F", assets.ASSET_CLASS_MACRO_INDEX, 24.0, si_step, 3),
        "DX-Y.NYB": _series("DX-Y.NYB", assets.ASSET_CLASS_MACRO_INDEX, 104.0, dxy_step, 4),
        "IEF": _series("IEF", assets.ASSET_CLASS_MACRO_INDEX, 94.0, ief_step, 5),
        "TIP": _series("TIP", assets.ASSET_CLASS_MACRO_INDEX, 106.0, tip_step, 6),
        "GDX": _series("GDX", assets.ASSET_CLASS_MACRO_INDEX, 31.0, gdx_step, 7),
        "^GVZ": build_bar_series(
            symbol="^GVZ",
            venue="yahoo",
            asset_class=assets.ASSET_CLASS_MACRO_INDEX,
            timeframe_h=24.0,
            start_ms=start_ms,
            closes=[gvz_level * (1.0 + 0.02 * math.sin(i / 6.0)) for i in range(n_days)],
            session_tag="TRADFI_DAILY",
        ),
        "^VIX": build_bar_series(
            symbol="^VIX",
            venue="yahoo",
            asset_class=assets.ASSET_CLASS_MACRO_INDEX,
            timeframe_h=24.0,
            start_ms=start_ms,
            closes=[vix_level * (1.0 + 0.02 * math.cos(i / 5.0)) for i in range(n_days)],
            session_tag="TRADFI_DAILY",
        ),
    }

    yields: list[Any] = []
    for i in range(n_days):
        frac = i / max(n_days - 1, 1)
        r10 = real_10y_start + frac * (real_10y_end - real_10y_start)
        ts_ev = start_ms + i * DAY_MS
        ts_av = ts_ev + DAY_MS
        yields.append(
            models.MacroYieldPoint(
                source="us_treasury",
                ts_event_ms=ts_ev,
                ts_avail_ms=ts_av,
                real_5y=r10 - 0.15,
                real_7y=r10 - 0.07,
                real_10y=r10,
                real_20y=r10 + 0.12,
                real_30y=r10 + 0.20,
                provenance=prov_real,
                source_fingerprint="tips_e2e_fp",
            )
        )
    return bars, yields


# ---------------------------------------------------------------------------
# RiskAllocator, Metrics, Strategy & Validation Contract Helpers (A-04..A-08)
# ---------------------------------------------------------------------------


def make_risk_allocator(cfg: Any) -> Any:
    """Instantiate CausalRiskAllocator cleanly per Ambiguity A-04."""
    mod = import_risk_allocator_mod()
    cls = mod.CausalRiskAllocator
    sig = inspect.signature(cls.__init__)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    if not params:
        return cls()
    first = params[0]
    if first.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
        return cls(cfg)
    for kw in ("cfg", "config", "risk_cfg"):
        if kw in sig.parameters:
            return cls(**{kw: cfg})
    return cls()


def call_risk_allocate(
    allocator: Any,
    cfg: Any,
    ts_ms: int,
    raw_decisions: dict[str, Any],
    recent_returns: dict[str, Sequence[float]],
    specs: dict[str, Any],
    *,
    timeframe_h: float | None = None,
) -> dict[str, float]:
    """Call CausalRiskAllocator.allocate cleanly per Ambiguity A-04 (with optional timeframe_h)."""
    fn = allocator.allocate
    sig = inspect.signature(fn)
    extra_kw: dict[str, Any] = {}
    if "timeframe_h" in sig.parameters and timeframe_h is not None:
        extra_kw["timeframe_h"] = float(timeframe_h)
    if "cfg" in sig.parameters:
        return fn(ts_ms, raw_decisions, recent_returns, specs, cfg=cfg, **extra_kw)
    if "config" in sig.parameters:
        return fn(ts_ms, raw_decisions, recent_returns, specs, config=cfg, **extra_kw)
    return fn(ts_ms, raw_decisions, recent_returns, specs, **extra_kw)


def normalize_verdict(verdict: Any) -> str:
    """Normalize Verdict Enum or string into canonical string value."""
    if hasattr(verdict, "value") and isinstance(verdict.value, str):
        return verdict.value
    if hasattr(verdict, "name") and isinstance(verdict.name, str):
        return verdict.name
    return str(verdict)


def get_metric(obj: Any, *candidate_names: str, default: Any = _MISSING) -> Any:
    """Strictly retrieve a named metric from object, nested .metrics, or .to_dict() (A-06)."""
    search_targets: list[Any] = [obj]
    nested_metrics = getattr(obj, "metrics", None)
    if nested_metrics is not None:
        search_targets.append(nested_metrics)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        d = to_dict()
        if isinstance(d, dict):
            search_targets.append(d)
            if isinstance(d.get("metrics"), dict):
                search_targets.append(d["metrics"])

    for name in candidate_names:
        for tgt in search_targets:
            if isinstance(tgt, dict) and name in tgt:
                return tgt[name]
            if not isinstance(tgt, dict) and hasattr(tgt, name):
                return getattr(tgt, name)

    if default is not _MISSING:
        return default
    available: list[str] = []
    for tgt in search_targets:
        if isinstance(tgt, dict):
            available.extend(sorted(str(k) for k in tgt.keys()))
        else:
            available.extend(sorted(k for k in dir(tgt) if not k.startswith("_")))
    raise AssertionError(
        f"Missing required metric(s) {candidate_names!r} on {type(obj).__name__}. "
        f"Available attributes/keys: {sorted(set(available))}"
    )


def measure_strategy_activity(sim_result: Any) -> tuple[int, float]:
    """Return (executed_trades, max_abs_executed_weight) to defeat zero-weight dummy stubs."""
    trade_count = int(get_metric(sim_result, "trade_count", "executed_trades", "total_trades", default=0))
    ledger = get_metric(sim_result, "trade_ledger", default=None)
    ledger_exec = len(getattr(ledger, "executed", ())) if ledger is not None else 0
    executed_trades = max(trade_count, ledger_exec)

    max_abs_w = 0.0
    hist = get_metric(
        sim_result,
        "executed_weights_history",
        "allocated_weights_history",
        default=(),
    )
    for entry in hist:
        wmap = entry[1] if isinstance(entry, tuple) and len(entry) >= 2 and isinstance(entry[1], dict) else entry
        if isinstance(wmap, dict):
            for val in wmap.values():
                av = abs(float(val))
                if av > max_abs_w:
                    max_abs_w = av
    return executed_trades, max_abs_w


def count_distinct_executed_weights(sim_result: Any, *, decimals: int = 5) -> int:
    """Count distinct rounded executed weight vectors across a simulation to detect constant stubs."""
    hist = get_metric(
        sim_result,
        "executed_weights_history",
        "allocated_weights_history",
        default=(),
    )
    seen: set[tuple[tuple[str, float], ...]] = set()
    for entry in hist:
        wmap = entry[1] if isinstance(entry, tuple) and len(entry) >= 2 and isinstance(entry[1], dict) else entry
        if isinstance(wmap, dict):
            vec = tuple((str(k), round(float(v), decimals)) for k, v in sorted(wmap.items()))
            seen.add(vec)
    return len(seen)


def count_distinct_raw_decisions(sim_result: Any, *, decimals: int = 5) -> int:
    """Count distinct raw strategy (target_weight, raw_signal) vectors to defeat constant-output stubs."""
    raw_hist = get_metric(sim_result, "raw_decisions_history", default=())
    seen: set[tuple[tuple[str, float, float], ...]] = set()
    for entry in raw_hist:
        if isinstance(entry, dict):
            vec = tuple(
                (
                    str(sym),
                    round(float(getattr(dec, "target_weight", 0.0)), decimals),
                    round(float(getattr(dec, "raw_signal", 0.0)), decimals),
                )
                for sym, dec in sorted(entry.items())
            )
            seen.add(vec)
    return len(seen)


def normalize_snapshot(snap: Any) -> dict[str, Any]:
    """Normalize dict, .to_dict(), frozen dataclass, or attribute snapshot into a plain dict."""
    if isinstance(snap, dict):
        return dict(snap)
    to_dict = getattr(snap, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
        if isinstance(out, dict):
            return dict(out)
    if dataclasses.is_dataclass(snap) and not isinstance(snap, type):
        return dataclasses.asdict(snap)
    result: dict[str, Any] = {}
    for k in dir(snap):
        if k.startswith("_"):
            continue
        v = getattr(snap, k)
        if not callable(v):
            result[k] = v
    return result


_KWARG_ALIASES: dict[str, tuple[str, ...]] = {
    "policy": ("policy", "event_policy", "policy_mode", "policy_type", "event_mode", "mode"),
    "symbol": ("symbol", "instrument", "instrument_symbol", "target_symbol", "primary_symbol", "gold_symbol", "ticker"),
    "top_k": ("top_k", "k_long", "n_long", "long_k", "max_long"),
    "bottom_k": ("bottom_k", "k_short", "n_short", "short_k", "max_short"),
    "daily_price_limit_pct": ("daily_price_limit_pct", "price_limit_pct", "limit_pct"),
    "selected_factors": ("selected_factors", "active_factors", "active_g2_factors", "allowed_factors"),
    "macro_engine": ("macro_engine", "gold_macro_engine", "regime_engine"),
    "macro_bars_by_symbol": ("macro_bars_by_symbol", "bars_by_symbol", "macro_bars"),
    "yield_curve": ("yield_curve", "yield_points", "tips_yields", "macro_yields"),
    "dvol_series": ("dvol_series", "dvol_points", "btc_dvol"),
    "stablecoin_series": ("stablecoin_series", "stablecoin_points", "stablecoin_mcap"),
    "web3_macro": ("web3_macro", "macro_points", "web3_macro_points"),
    "metrics_by_symbol": ("metrics_by_symbol", "derivatives_metrics", "web3_metrics"),
}
CRITICAL_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {"policy", "symbol", "top_k", "bottom_k", "daily_price_limit_pct"}
)


def _extract_dataclass_type(hint: Any) -> type | None:
    """Resolve a dataclass type from a type hint (including Optional[X] / X | None)."""
    if inspect.isclass(hint) and dataclasses.is_dataclass(hint):
        return hint
    origin = typing.get_origin(hint)
    if origin in (typing.Union, UnionType):
        for arg in typing.get_args(hint):
            if inspect.isclass(arg) and dataclasses.is_dataclass(arg):
                return arg
    return None


def build_strategy_instance(
    cls: type,
    *,
    required_overrides: Sequence[str] = (),
    **preferred_kwargs: Any,
) -> Any:
    """Instantiate a strategy/engine class binding aliases & config dataclasses, failing loudly on dropped overrides."""
    sig = inspect.signature(cls.__init__)
    params = {name: p for name, p in sig.parameters.items() if name != "self"}
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    filtered: dict[str, Any] = {}
    bound_keys: set[str] = set()

    # Step A: Direct or alias match against __init__ parameters
    for key, val in preferred_kwargs.items():
        aliases = _KWARG_ALIASES.get(key, (key,))
        matched_param = None
        for cand in aliases:
            if cand in params and cand not in filtered:
                matched_param = cand
                break
        if matched_param is not None:
            filtered[matched_param] = val
            bound_keys.add(key)

    # Step B: Match remaining unbound kwargs against a config dataclass parameter if present
    unbound = {k: v for k, v in preferred_kwargs.items() if k not in bound_keys}
    cfg_fields_debug: list[str] = []
    if unbound:
        type_hints: dict[str, Any] = {}
        try:
            type_hints = typing.get_type_hints(cls.__init__)
        except (NameError, TypeError, AttributeError) as exc:
            _hint_err = str(exc)
            type_hints = {}

        for pname, param in params.items():
            if pname in filtered:
                continue
            cfg_default = param.default if (param.default is not inspect.Parameter.empty and dataclasses.is_dataclass(param.default)) else None
            cfg_cls = type(cfg_default) if cfg_default is not None else _extract_dataclass_type(type_hints.get(pname))
            if cfg_cls is None:
                continue
            fnames = {f.name for f in dataclasses.fields(cfg_cls)}
            cfg_fields_debug.extend(sorted(fnames))
            cfg_kw: dict[str, Any] = {}
            for key, val in list(unbound.items()):
                aliases = _KWARG_ALIASES.get(key, (key,))
                for cand in aliases:
                    if cand in fnames and cand not in cfg_kw:
                        cfg_kw[cand] = val
                        bound_keys.add(key)
                        break
            if cfg_kw:
                if cfg_default is not None and not isinstance(cfg_default, type):
                    filtered[pname] = dataclasses.replace(cfg_default, **cfg_kw)
                else:
                    filtered[pname] = cfg_cls(**cfg_kw)
                unbound = {k: v for k, v in preferred_kwargs.items() if k not in bound_keys}
                if not unbound:
                    break

    if has_var_kw and unbound:
        for key, val in unbound.items():
            filtered[key] = val
            bound_keys.add(key)

    # Step C: Fail loudly if any critical or explicitly required override could not be bound
    must_bind = set(required_overrides) | (set(preferred_kwargs) & CRITICAL_OVERRIDE_KEYS)
    missing_critical = sorted(k for k in must_bind if k not in bound_keys)
    if missing_critical:
        raise AssertionError(
            f"Critical override kwarg(s) {missing_critical!r} could not be bound on "
            f"{cls.__name__}.__init__(params={list(params.keys())}, cfg_fields={sorted(set(cfg_fields_debug))})"
        )

    return cls(**filtered)


def invoke_web3_regime_engine(
    cls: type,
    bars_by_symbol: dict[str, Sequence[Any]],
    dvol_series: Sequence[Any],
    stablecoin_series: Sequence[Any],
    ts_ms: int,
    *,
    extra_context: dict[str, Any] | None = None,
    metrics_by_symbol: dict[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Invoke Web3MarketRegimeEngine across snapshot/evaluate/streaming contracts without silent fallbacks."""
    init_kw: dict[str, Any] = {
        "bars_by_symbol": bars_by_symbol,
        "dvol_series": list(dvol_series),
        "stablecoin_series": list(stablecoin_series),
        "web3_macro": list(dvol_series) + list(stablecoin_series),
    }
    if metrics_by_symbol is not None:
        init_kw["metrics_by_symbol"] = metrics_by_symbol
    inst = build_strategy_instance(cls, **init_kw)

    method_name = None
    fn = None
    for cand in ("snapshot_as_of", "evaluate", "classify", "infer_regime", "update", "on_bar_step"):
        attr = getattr(inst, cand, None)
        if callable(attr):
            method_name = cand
            fn = attr
            break
    if fn is None or method_name is None:
        raise AssertionError(
            f"{cls.__name__} must expose a public evaluation method "
            f"(snapshot_as_of/evaluate/classify/infer_regime/update/on_bar_step)"
        )

    sig = inspect.signature(fn)
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    extra = dict(extra_context or {})

    # Determine BTC or primary symbol timeline up to ts_ms
    ref_sym = "BTC" if "BTC" in bars_by_symbol else sorted(bars_by_symbol.keys())[0]
    ref_bars = [b for b in bars_by_symbol[ref_sym] if int(b.ts_event_ms) <= int(ts_ms)]
    if not ref_bars:
        ref_bars = list(bars_by_symbol[ref_sym])

    elig_dvol = [p for p in dvol_series if int(getattr(p, "ts_event_ms", 0)) <= int(ts_ms)]
    if not elig_dvol:
        elig_dvol = list(dvol_series)
    elig_stable = [p for p in stablecoin_series if int(getattr(p, "ts_event_ms", 0)) <= int(ts_ms)]
    if not elig_stable:
        elig_stable = list(stablecoin_series)

    dvol_val = float(elig_dvol[-1].value) if elig_dvol else 50.0
    if elig_stable:
        st_idx0 = max(0, len(elig_stable) - 60)
        st_v0 = float(elig_stable[st_idx0].value)
        st_flow_30d = (float(elig_stable[-1].value) / st_v0 - 1.0) if st_v0 > 0 else 0.0
    else:
        st_flow_30d = 0.0

    # Streaming style: update / on_bar_step accepting bars_at_ts
    if method_name in ("update", "on_bar_step"):
        reset_fn = getattr(inst, "reset", None)
        if callable(reset_fn):
            reset_fn()
        dvol_by_ts = {int(p.ts_event_ms): float(p.value) for p in elig_dvol}
        st_by_ts = {int(p.ts_event_ms): float(p.value) for p in elig_stable}
        by_ts_bars: dict[int, dict[str, Any]] = {}
        for sym, blist in bars_by_symbol.items():
            for b in blist:
                t_b = int(b.ts_event_ms)
                if t_b <= int(ts_ms):
                    by_ts_bars.setdefault(t_b, {})[sym] = b
        timeline = sorted(by_ts_bars.keys())
        out: Any = None
        last_dvol = float(elig_dvol[0].value) if elig_dvol else 50.0
        last_st = float(elig_stable[0].value) if elig_stable else 150e9
        st_hist: list[float] = []
        n_pos = sum(
            1
            for p in params.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        for t_i in timeline:
            if t_i in dvol_by_ts:
                last_dvol = dvol_by_ts[t_i]
            if t_i in st_by_ts:
                last_st = st_by_ts[t_i]
            st_hist.append(last_st)
            base_st = st_hist[max(0, len(st_hist) - 60)]
            flow_i = (last_st / base_st - 1.0) if base_st > 0 else 0.0
            ctx_i: dict[str, Any] = {
                "BTC_DVOL": last_dvol,
                "dvol": last_dvol,
                "STABLECOIN_MCAP_USD": last_st,
                "stablecoin_flow_30d": flow_i,
            }
            ctx_i.update(extra)
            if n_pos >= 3 or has_var_kw:
                out = fn(t_i, by_ts_bars[t_i], ctx_i)
            elif n_pos == 2:
                out = fn(t_i, by_ts_bars[t_i])
            else:
                out = fn(t_i)
        return normalize_snapshot(out)

    # Stateless / snapshot / evaluate style: bind strictly accepted parameters
    bars_at_ts = {}
    for sym, blist in bars_by_symbol.items():
        elig = [b for b in blist if int(b.ts_event_ms) <= int(ts_ms)]
        if elig:
            bars_at_ts[sym] = elig[-1]
    ctx_payload: dict[str, Any] = {
        "BTC_DVOL": dvol_val,
        "dvol": dvol_val,
        "stablecoin_flow_30d": st_flow_30d,
    }
    ctx_payload.update(extra)

    candidate_kw: dict[str, Any] = {
        "ts_ms": int(ts_ms),
        "as_of_ms": int(ts_ms),
        "ts_event_ms": int(ts_ms),
        "bars_at_ts": bars_at_ts,
        "bars_by_symbol": bars_by_symbol,
        "dvol": dvol_val,
        "dvol_series": list(dvol_series),
        "stablecoin_series": list(stablecoin_series),
        "stablecoin_flow_30d": st_flow_30d,
        "context": ctx_payload,
        "ctx": ctx_payload,
    }
    if metrics_by_symbol is not None:
        candidate_kw["metrics_by_symbol"] = metrics_by_symbol

    call_kw: dict[str, Any] = {}
    for pname in params:
        if pname in candidate_kw:
            call_kw[pname] = candidate_kw[pname]
    if has_var_kw:
        for ek, ev in extra.items():
            if ek not in call_kw:
                call_kw[ek] = ev

    if not call_kw and len(params) >= 1:
        out = fn(int(ts_ms))
    else:
        out = fn(**call_kw)
    return normalize_snapshot(out)


def instantiate_ashare_interface(
    cls: type,
    *,
    default_signal: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Instantiate AShareStrategyInterface cleanly whether concrete or ABC (dynamic __abstractmethods__)."""
    abstract_names = getattr(cls, "__abstractmethods__", None)
    if not abstract_names:
        return build_strategy_instance(cls, **kwargs)

    def _extract_ctx_signal(args: tuple[Any, ...], kw: dict[str, Any]) -> float:
        ctx_dict: dict[str, Any] | None = None
        for v in kw.values():
            if isinstance(v, dict) and any(k in v for k in ("raw_signal", "desired_weight", "signal")):
                ctx_dict = v
                break
        if ctx_dict is None:
            for a in reversed(args):
                if isinstance(a, dict) and any(k in a for k in ("raw_signal", "desired_weight", "signal")):
                    ctx_dict = a
                    break
        if ctx_dict is not None:
            for key in ("raw_signal", "desired_weight", "signal"):
                if key in ctx_dict:
                    return float(ctx_dict[key])
        return float(default_signal)

    method_impls: dict[str, Any] = {}
    for m_name in sorted(abstract_names):
        orig = getattr(cls, m_name, None)
        ret_str = ""
        if orig is not None:
            ret_ann = inspect.signature(orig).return_annotation
            ret_str = str(ret_ann).lower() if ret_ann is not inspect.Signature.empty else ""

        if m_name == "clone_fresh":
            def _clone_fresh(self: Any) -> Any:
                return instantiate_ashare_interface(cls, default_signal=default_signal, **kwargs)
            method_impls[m_name] = _clone_fresh
        elif m_name == "on_bar_step" or "dict" in ret_str:
            def _on_bar_step_impl(self: Any, *args: Any, **kw: Any) -> dict[str, Any]:
                iface = import_core_interfaces()
                ts_val = int(args[0]) if args else int(kw.get("ts_ms", 0))
                bars_arg = args[1] if len(args) > 1 else kw.get("bars_at_ts", {})
                sig_val = _extract_ctx_signal(args, kw)
                w_val = max(0.0, min(1.0, sig_val))
                out: dict[str, Any] = {}
                if isinstance(bars_arg, dict):
                    for sym in sorted(bars_arg.keys()):
                        out[sym] = iface.BarDecision(
                            symbol=sym,
                            ts_event_ms=ts_val,
                            target_weight=w_val,
                            raw_signal=sig_val,
                            regime_label="ASHARE",
                            metadata={"source": "dynamic_ashare"},
                        )
                return out
            method_impls[m_name] = _on_bar_step_impl
        elif "tuple" in ret_str:
            def _tuple_impl(self: Any, *args: Any, **kw: Any) -> tuple[float, dict[str, Any]]:
                return (_extract_ctx_signal(args, kw), {})
            method_impls[m_name] = _tuple_impl
        else:
            def _float_impl(self: Any, *args: Any, **kw: Any) -> float:
                return _extract_ctx_signal(args, kw)
            method_impls[m_name] = _float_impl

    dyn_cls = type(f"_Dynamic{cls.__name__}", (cls,), method_impls)
    return build_strategy_instance(dyn_cls, **kwargs)


def resolve_validation_callable(mod: Any, *candidate_names: str) -> Callable[..., Any]:
    """Resolve a public validation function or validator method cleanly (A-08)."""
    for name in candidate_names:
        attr = getattr(mod, name, None)
        if callable(attr) and not inspect.isclass(attr):
            return attr
    # Also inspect canonical Validator classes if exposed
    for cls_name in (
        "ValidationSuiteV4",
        "InstitutionalValidator",
        "InstitutionalValidationSuite",
        "ValidatorV4",
        "EightGateValidator",
    ):
        cls = getattr(mod, cls_name, None)
        if inspect.isclass(cls):
            for name in candidate_names:
                method = getattr(cls, name, None)
                if callable(method):
                    inst = cls()
                    return getattr(inst, name)
    available = sorted(k for k in dir(mod) if not k.startswith("_"))
    raise AssertionError(
        f"Required validation contract callable {candidate_names!r} not found in {mod.__name__}. "
        f"Available exports: {available}"
    )


def resolve_verdict_enum(val_mod: Any) -> Any:
    """Resolve canonical 3-value Verdict Enum from core_engine.validation (A-08)."""
    for cname in ("GateVerdict", "Verdict", "ValidationVerdict", "GateVerdictV4", "PlatformVerdict"):
        cand = getattr(val_mod, cname, None)
        if cand is not None:
            return cand
    available = sorted(k for k in dir(val_mod) if not k.startswith("_"))
    raise AssertionError(f"Missing Verdict Enum in {val_mod.__name__}; available={available}")


def _synth_return_series_for_sharpe(
    target_sharpe: float,
    *,
    n_obs: int = 252,
    periods_per_year: float = 252.0,
    annual_vol: float = 0.16,
    base_series: Sequence[float] | None = None,
    target_corr: float = 0.0,
    seed: int = 777,
) -> list[float]:
    """Synthesize pure-stdlib return vector with controlled Sharpe and correlation to base_series."""
    rng = random.Random(seed)
    sigma = annual_vol / math.sqrt(periods_per_year)
    if base_series is not None and len(base_series) == n_obs:
        bm = sum(base_series) / n_obs
        bc = [x - bm for x in base_series]
        bsd = math.sqrt(sum(x * x for x in bc) / max(n_obs - 1, 1))
        bstd = [x / bsd for x in bc] if bsd > 1e-12 else [0.0] * n_obs
    else:
        bstd = [0.0] * n_obs
        target_corr = 0.0

    raw = [rng.gauss(0.0, 1.0) for _ in range(n_obs)]
    rm = sum(raw) / n_obs
    rc = [x - rm for x in raw]
    # Orthogonalize against bstd
    dot_rb = sum(rc[i] * bstd[i] for i in range(n_obs))
    dot_bb = sum(bstd[i] * bstd[i] for i in range(n_obs))
    proj = dot_rb / dot_bb if dot_bb > 1e-12 else 0.0
    orth = [rc[i] - proj * bstd[i] for i in range(n_obs)]
    osd = math.sqrt(sum(x * x for x in orth) / max(n_obs - 1, 1))
    ostd = [x / osd for x in orth] if osd > 1e-12 else [0.0] * n_obs

    rho = max(-0.99, min(0.99, float(target_corr)))
    z = [rho * bstd[i] + math.sqrt(max(1.0 - rho * rho, 0.0)) * ostd[i] for i in range(n_obs)]
    zm = sum(z) / n_obs
    zc = [x - zm for x in z]
    zsd = math.sqrt(sum(x * x for x in zc) / max(n_obs - 1, 1))
    zstd = [x / zsd for x in zc] if zsd > 1e-12 else [0.0] * n_obs

    mu = (target_sharpe * annual_vol) / periods_per_year
    return [mu + sigma * x for x in zstd]


def _standardize_vec(xs: Sequence[float]) -> list[float]:
    """Center and scale vector to zero mean and unit sample standard deviation."""
    n = len(xs)
    if n < 2:
        return [0.0] * n
    mean = sum(xs) / n
    centered = [float(x) - mean for x in xs]
    sd = math.sqrt(sum(c * c for c in centered) / max(n - 1, 1))
    if sd <= 1e-12:
        return [0.0] * n
    return [c / sd for c in centered]


def _match_corr_preserving_shape(
    target_base: Sequence[float],
    ref_std: Sequence[float],
    target_corr: float,
    seed: int,
) -> list[float]:
    """Construct a unit-std series preserving target_base's drawdown shape while matching target_corr to ref_std."""
    n = len(target_base)
    base_std = _standardize_vec(target_base)
    dot_br = sum(base_std[i] * ref_std[i] for i in range(n))
    dot_rr = sum(ref_std[i] * ref_std[i] for i in range(n))
    proj = dot_br / dot_rr if dot_rr > 1e-12 else 0.0
    orth = [base_std[i] - proj * ref_std[i] for i in range(n)]
    orth_ss = sum(x * x for x in orth)
    if orth_ss <= 1e-12:
        rng = random.Random(seed)
        noise = _standardize_vec([rng.gauss(0.0, 1.0) for _ in range(n)])
        dot_nr = sum(noise[i] * ref_std[i] for i in range(n))
        proj_n = dot_nr / dot_rr if dot_rr > 1e-12 else 0.0
        orth = [noise[i] - proj_n * ref_std[i] for i in range(n)]
    orth_std = _standardize_vec(orth)
    rho = max(-0.99, min(0.99, float(target_corr)))
    # Preserve sign alignment of orth with original target_base so crash-avoidance stays positive
    combined = [rho * ref_std[i] + math.sqrt(max(1.0 - rho * rho, 0.0)) * orth_std[i] for i in range(n)]
    return _standardize_vec(combined)


def _empirical_sr_calmar(returns: Sequence[float], periods_per_year: float) -> tuple[float, float]:
    """Pure-stdlib empirical annualized Sharpe and Calmar ratios."""
    n = len(returns)
    if n < 2:
        return 0.0, 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    sd = math.sqrt(max(var, 0.0))
    sr = (mean / sd) * math.sqrt(periods_per_year) if sd > 1e-12 else 0.0

    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        eq *= max(1.0 + float(r), 1e-6)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0.0 else 0.0
        if dd > max_dd:
            max_dd = dd
    years = max(n / periods_per_year, 1e-6)
    cagr = (eq ** (1.0 / years)) - 1.0 if eq > 0.0 else -1.0
    calmar = cagr / max_dd if max_dd > 1e-9 else (10.0 if cagr > 0.0 else 0.0)
    return sr, calmar


def _build_crash_avoidance_pair(
    crash_sr: float,
    smooth_sr: float,
    target_corr: float,
    *,
    periods_per_year: float = 252.0,
    n_obs: int = 360,
    seed: int = 803,
) -> tuple[list[float], list[float]]:
    """Build (crash_series, smooth_series) where smooth_series has strictly higher Calmar at exact Sharpe & corr."""
    rng = random.Random(seed)
    # 1) Smooth unit-std series: alternating sign + mild jitter -> tiny cumulative drawdown, high Calmar
    raw_u = [(1.0 if i % 2 == 0 else -1.0) + rng.gauss(0.0, 0.15) for i in range(n_obs)]
    u_std = _standardize_vec(raw_u)

    # 2) Concentrated consecutive crash-and-rebound vector orthogonalized to u_std -> deep MaxDD, low Calmar
    raw_v = [0.0] * n_obs
    crash_start = min(80, n_obs // 4)
    crash_mid = min(crash_start + 60, (n_obs * 2) // 4)
    crash_end = min(crash_mid + 60, (n_obs * 3) // 4)
    for i in range(crash_start, crash_mid):
        raw_v[i] = -1.0
    for i in range(crash_mid, crash_end):
        raw_v[i] = +1.0
    dot_vu = sum(raw_v[i] * u_std[i] for i in range(n_obs))
    dot_uu = sum(u_std[i] * u_std[i] for i in range(n_obs))
    proj = dot_vu / dot_uu if dot_uu > 1e-12 else 0.0
    v_std = _standardize_vec([raw_v[i] - proj * u_std[i] for i in range(n_obs)])

    rho = max(-0.95, min(0.95, float(target_corr)))
    c_std = _standardize_vec(
        [rho * u_std[i] + math.sqrt(max(1.0 - rho * rho, 0.0)) * v_std[i] for i in range(n_obs)]
    )

    c_vol = 0.25
    sm_vol = 0.16
    c_sigma = c_vol / math.sqrt(periods_per_year)
    c_mu = (crash_sr * c_vol) / periods_per_year
    crash_rets = [c_mu + c_sigma * x for x in c_std]

    sm_sigma = sm_vol / math.sqrt(periods_per_year)
    sm_mu = (smooth_sr * sm_vol) / periods_per_year
    smooth_rets = [sm_mu + sm_sigma * x for x in u_std]
    return crash_rets, smooth_rets


def evaluate_g8_contract(
    val_mod: Any,
    *,
    strategy_sharpe: float,
    strategy_calmar: float,
    benchmark_sharpe: float,
    benchmark_calmar: float,
    corr_to_benchmark: float,
    periods_per_year: float = 252.0,
) -> Any:
    """Invoke public G8 Passive Benchmark Gate honoring empirical SR, Calmar, and corr invariants (A-08)."""
    fn = resolve_validation_callable(
        val_mod,
        "evaluate_benchmark_gate",
        "evaluate_g8_benchmark",
        "evaluate_g8",
        "check_g8_benchmark",
        "evaluate_passive_benchmark_gate",
    )
    sig = inspect.signature(fn)
    pnames = list(sig.parameters.keys())
    if any("return" in p for p in pnames):
        n_obs = 360
        if strategy_sharpe > 0.0 and benchmark_sharpe > 0.0 and (strategy_calmar > benchmark_calmar):
            bm_rets, st_rets = _build_crash_avoidance_pair(
                benchmark_sharpe,
                strategy_sharpe,
                corr_to_benchmark,
                periods_per_year=periods_per_year,
                n_obs=n_obs,
                seed=803,
            )
        elif strategy_sharpe > 0.0 and benchmark_sharpe > 0.0 and (strategy_calmar < benchmark_calmar):
            st_rets, bm_rets = _build_crash_avoidance_pair(
                strategy_sharpe,
                benchmark_sharpe,
                corr_to_benchmark,
                periods_per_year=periods_per_year,
                n_obs=n_obs,
                seed=805,
            )
        else:
            bm_rets = _synth_return_series_for_sharpe(
                benchmark_sharpe, n_obs=n_obs, periods_per_year=periods_per_year, annual_vol=0.16, seed=801
            )
            st_rets = _synth_return_series_for_sharpe(
                strategy_sharpe,
                n_obs=n_obs,
                periods_per_year=periods_per_year,
                annual_vol=0.16,
                base_series=bm_rets,
                target_corr=corr_to_benchmark,
                seed=802,
            )

        # Built-in mathematical self-check preventing silent synthesis inversion
        real_s_sr, real_s_cal = _empirical_sr_calmar(st_rets, periods_per_year)
        real_b_sr, real_b_cal = _empirical_sr_calmar(bm_rets, periods_per_year)
        real_corr = sample_pearson_corr(st_rets, bm_rets)
        if abs(strategy_sharpe - benchmark_sharpe) > 1e-6:
            if (real_s_sr > real_b_sr) != (strategy_sharpe > benchmark_sharpe):
                raise AssertionError(
                    f"evaluate_g8_contract SR synthesis drift: real ({real_s_sr:.3f} vs {real_b_sr:.3f}) "
                    f"!= requested ({strategy_sharpe:.3f} vs {benchmark_sharpe:.3f})"
                )
        if strategy_sharpe > 0.0 and benchmark_sharpe > 0.0 and abs(strategy_calmar - benchmark_calmar) > 1e-6:
            if (real_s_cal > real_b_cal) != (strategy_calmar > benchmark_calmar):
                raise AssertionError(
                    f"evaluate_g8_contract Calmar synthesis drift: real ({real_s_cal:.3f} vs {real_b_cal:.3f}) "
                    f"!= requested ({strategy_calmar:.3f} vs {benchmark_calmar:.3f})"
                )
        if abs(real_corr - corr_to_benchmark) > 0.05:
            raise AssertionError(
                f"evaluate_g8_contract corr synthesis drift: real {real_corr:.3f} vs requested {corr_to_benchmark:.3f}"
            )

        kw: dict[str, Any] = {}
        for p in pnames:
            if "strat" in p and "ret" in p:
                kw[p] = st_rets
            elif ("bench" in p or "bh" in p) and "ret" in p:
                kw[p] = bm_rets
            elif "periods_per_year" in p:
                kw[p] = periods_per_year
        if len(kw) >= 2:
            return fn(**kw)
        return fn(st_rets, bm_rets, periods_per_year)
    if "strategy_sharpe" in sig.parameters:
        return fn(
            strategy_sharpe=strategy_sharpe,
            strategy_calmar=strategy_calmar,
            benchmark_sharpe=benchmark_sharpe,
            benchmark_calmar=benchmark_calmar,
            corr_to_benchmark=corr_to_benchmark,
        )
    return fn(strategy_sharpe, strategy_calmar, benchmark_sharpe, benchmark_calmar, corr_to_benchmark)


def _make_genuine_plateau_report(val_mod: Any, passed: bool) -> Any:
    """Build a real PlateauSelectionReport via select_plateau_center."""
    PlateauPoint = getattr(val_mod, "PlateauPoint", None) or import_v3_validation_mod().PlateauPoint
    select_fn = resolve_validation_callable(val_mod, "select_plateau_center")
    pts = []
    for x in range(3):
        for y in range(3):
            if passed:
                sr = 1.25 - 0.04 * (abs(x - 1) + abs(y - 1))
            else:
                sr = 2.0 if (x == 1 and y == 1) else -0.4
            pts.append(
                PlateauPoint(
                    ix=x, iy=y, label=f"p_{x}_{y}",
                    params={"x": x, "y": y}, sharpe=sr, apr_pct=sr * 15.0,
                    max_dd_pct=10.0, net_pnl_usd=sr * 1500.0,
                )
            )
    return select_fn(pts, min_sharpe_for_plateau=0.25, min_plateau_ratio=0.55)


def evaluate_8_gates_with_overrides(
    val_mod: Any,
    *,
    g1_passed: bool = True,
    g2_passed: bool = True,
    baseline_sharpe: float = 1.45,
    baseline_net_pnl: float = 3200.0,
    stress_sharpe: float = 1.10,
    stress_net_pnl: float = 2400.0,
    g4_passed: bool = True,
    g5_passed: bool = True,
    g6_passed: bool = True,
    g7_passed: bool = True,
    g8_passed: bool = True,
) -> tuple[dict[str, bool], str]:
    """Evaluate G1..G8 & Verdict Enum via standalone or holistic evaluate_8_gates contract (A-08)."""
    from types import SimpleNamespace

    eval_all = getattr(val_mod, "evaluate_8_gates", None) or getattr(val_mod, "evaluate_eight_gates", None)
    if callable(eval_all):
        sig = inspect.signature(eval_all)
        dl_mod = import_data_lake()
        fac_mod = import_factors_mod()
        prov_real, _, prov_synth = import_provenance()
        prov = prov_real if g1_passed else prov_synth

        dq_obj = dl_mod.MultiAssetDQReport(
            dataset_label="e2e_ds",
            asset_class="CRYPTO_PERP",
            row_count=120,
            dataset_sha256="a" * 64,
            provenance=prov,
            passed=g1_passed,
            checks=[],
        )
        fac_obj = fac_mod.FactorAuditSummary(
            track_name="e2e_track",
            horizon_bars=8,
            total_factors_evaluated=1,
            passed_g2_factors=["ewmac_16_64"] if g2_passed else [],
            selected_non_redundant_factors=["ewmac_16_64"] if g2_passed else [],
            pruned_redundant_pairs=[],
            results=[
                fac_mod.FactorAuditResult(
                    factor_name="ewmac_16_64", expected_sign=1, sample_count=100,
                    spearman_ic=0.12 if g2_passed else 0.0, t_stat=2.5 if g2_passed else 0.1,
                    icir=0.8 if g2_passed else 0.0, hit_rate=0.6 if g2_passed else 0.5,
                    quintile_means_bps=[-10.0, -5.0, 0.0, 5.0, 15.0] if g2_passed else [0.0] * 5,
                    monotonicity_rho=0.9 if g2_passed else 0.0,
                    sign_ok=g2_passed, monotonic_ok=g2_passed, passed_g2=g2_passed,
                )
            ],
        )
        plateau_obj = _make_genuine_plateau_report(val_mod, g4_passed)
        dsr_fn = resolve_validation_callable(val_mod, "compute_psr_dsr")
        rng = random.Random(901)
        dsr_rets = (
            [0.0055 + rng.gauss(0.0, 0.008) for _ in range(400)]
            if g5_passed
            else [rng.gauss(0.0, 0.01) for _ in range(400)]
        )
        dsr_obj = dsr_fn(dsr_rets, periods_per_year=252.0, n_trials=4, trial_sharpe_std_per_period=0.04)
        pbo_fn = resolve_validation_callable(val_mod, "compute_pbo_cscv")
        if g6_passed:
            pbo_mat = (
                [[0.008 + rng.gauss(0.0, 0.006) for _ in range(320)]]
                + [[-0.002 + rng.gauss(0.0, 0.010) for _ in range(320)] for _ in range(7)]
            )
        else:
            # Deterministic 8-block regime-overfit matrix: candidate c wins ONLY in block c (40 bars/block)
            # -> IS winner always has 0 winning blocks in OOS -> guaranteed PBO == 1.0 (>= 0.85)
            pbo_mat = []
            for c_idx in range(8):
                row_c: list[float] = []
                for blk in range(8):
                    drift = 0.015 if blk == c_idx else -0.003
                    for _ in range(40):
                        row_c.append(drift + rng.gauss(0.0, 0.004))
                pbo_mat.append(row_c)
        pbo_obj = pbo_fn(pbo_mat, periods_per_year=252.0, n_subsamples=8)
        if (float(pbo_obj.pbo) <= 0.35) != g6_passed:
            raise AssertionError(
                f"evaluate_8_gates_with_overrides G6 matrix drift: g6_passed={g6_passed} but PBO={pbo_obj.pbo:.4f}"
            )
        bench_obj = evaluate_g8_contract(
            val_mod,
            strategy_sharpe=1.30 if g8_passed else 0.50,
            strategy_calmar=1.10 if g8_passed else 0.25,
            benchmark_sharpe=0.80,
            benchmark_calmar=0.55,
            corr_to_benchmark=0.80,
        )
        def _build_contract_report(cls_name: str, overrides: dict[str, Any], fallback_ns: Any) -> Any:
            cls = getattr(val_mod, cls_name, None)
            if not inspect.isclass(cls):
                return fallback_ns
            c_sig = inspect.signature(cls)
            is_pass = bool(overrides.get("passed", True))
            c_kw: dict[str, Any] = {}
            for pname, p in c_sig.parameters.items():
                low = pname.lower()
                if pname in overrides:
                    c_kw[pname] = overrides[pname]
                elif "pass" in low or low.endswith("_ok") or low.startswith("is_") or low == "parity_ok":
                    c_kw[pname] = is_pass
                elif "delta" in low or "diff" in low:
                    c_kw[pname] = 0.0 if is_pass else 0.05
                elif "mismatch" in low or "error_count" in low or "fail_count" in low:
                    c_kw[pname] = 0 if is_pass else 5
                elif p.default is inspect.Parameter.empty:
                    if "ratio" in low or "sharpe" in low or "retention" in low:
                        c_kw[pname] = 0.9 if is_pass else -0.2
                    elif "n_" in low or "count" in low or "bars" in low or "splits" in low or "checked" in low:
                        c_kw[pname] = 15
                    elif "outcomes" in low or "splits" in low or "errors" in low or "returns" in low:
                        c_kw[pname] = []
                    else:
                        c_kw[pname] = "e2e"
            return cls(**c_kw)

        nested_ns = SimpleNamespace(
            passed=g5_passed,
            passed_nested_cpcv=g5_passed,
            passed_cpcv_gate=g5_passed,
            n_splits=15,
            n_groups=6,
            k_test=2,
            oos_positive_ratio=0.87 if g5_passed else 0.20,
            oos_sharpe_retention=0.85 if g5_passed else 0.10,
            mean_oos_sharpe=1.15 if g5_passed else -0.25,
            median_oos_sharpe=1.10 if g5_passed else -0.25,
            mean_is_sharpe=1.35 if g5_passed else 0.50,
            split_outcomes=[],
            detail="e2e_nested_cpcv",
        )
        nested_obj = _build_contract_report("NestedCPCVReport", vars(nested_ns), nested_ns)

        parity_ns = SimpleNamespace(
            passed=g7_passed,
            passed_parity=g7_passed,
            passed_parity_gate=g7_passed,
            max_abs_weight_delta=0.0 if g7_passed else 0.05,
            max_abs_equity_delta=0.0 if g7_passed else 50.0,
            prefix_max_weight_delta=0.0 if g7_passed else 0.05,
            n_bars_checked=100,
            detail="e2e_parity",
        )
        parity_obj = _build_contract_report("ParityReport", vars(parity_ns), parity_ns)
        base_sim = SimpleNamespace(
            sharpe_ratio=baseline_sharpe,
            net_pnl_usd=baseline_net_pnl,
            cagr_pct=baseline_sharpe * 10.0,
            simple_apr_pct=baseline_sharpe * 10.0,
            max_drawdown_pct=8.0,
            calmar_ratio=max(baseline_sharpe, 0.0),
            total_trades=25,
            executed_trades=25,
            trade_count=25,
            liquidation_count=0,
            trade_ledger=SimpleNamespace(executed_count=25, candidate_count=30, suppressed_count=5),
        )
        stress_sim = SimpleNamespace(
            sharpe_ratio=stress_sharpe,
            net_pnl_usd=stress_net_pnl,
            cagr_pct=stress_sharpe * 10.0,
            simple_apr_pct=stress_sharpe * 10.0,
            max_drawdown_pct=10.0,
            calmar_ratio=max(stress_sharpe, 0.0),
            total_trades=25,
            executed_trades=25,
            trade_count=25,
            liquidation_count=0,
            trade_ledger=SimpleNamespace(executed_count=25, candidate_count=30, suppressed_count=5),
        )

        kw: dict[str, Any] = {}
        for pname, param in sig.parameters.items():
            low = pname.lower()
            if "strategy_id" in low or "track" in low or "name" in low or "label" in low:
                kw[pname] = "e2e_track"
            elif "domain" in low:
                kw[pname] = "web3"
            elif "provenance" in low:
                kw[pname] = prov
            elif low in ("g1_passed", "dq_passed", "g1_ok", "g1_dq_passed"):
                kw[pname] = g1_passed
            elif low in ("g2_passed", "factor_passed", "g2_ok", "g2_factor_passed"):
                kw[pname] = g2_passed
            elif "stress" in low and "sharpe" in low:
                kw[pname] = stress_sharpe
            elif "stress" in low and "pnl" in low:
                kw[pname] = stress_net_pnl
            elif "sharpe" in low:
                kw[pname] = baseline_sharpe
            elif "pnl" in low:
                kw[pname] = baseline_net_pnl
            elif low in ("g4_passed", "plateau_passed", "g4_ok", "g4_plateau_passed"):
                kw[pname] = g4_passed
            elif low in ("g5_passed", "cpcv_dsr_passed", "g5_ok", "g5_nested_passed"):
                kw[pname] = g5_passed
            elif low in ("g6_passed", "pbo_passed", "g6_ok", "g6_pbo_passed"):
                kw[pname] = g6_passed
            elif low in ("g7_passed", "parity_passed", "g7_ok", "g7_parity_passed"):
                kw[pname] = g7_passed
            elif low in ("g8_passed", "benchmark_passed", "g8_ok", "g8_benchmark_passed"):
                kw[pname] = g8_passed
            elif "detail" in low:
                kw[pname] = "e2e_probe"
            elif "dq_reports" in low:
                kw[pname] = [dq_obj]
            elif "dq" in low and "report" in low:
                kw[pname] = dq_obj
            elif "factor_audit" in low:
                kw[pname] = fac_obj
            elif "dsr" in low and "report" in low:
                kw[pname] = dsr_obj
            elif ("cpcv" in low or "nested" in low) and "report" in low:
                kw[pname] = nested_obj
            elif "pbo" in low and "report" in low:
                kw[pname] = pbo_obj
            elif "plateau" in low and "report" in low:
                kw[pname] = plateau_obj
            elif "parity" in low and "report" in low:
                kw[pname] = parity_obj
            elif "bench" in low and "report" in low:
                kw[pname] = bench_obj
            elif ("stress" in low) and ("sim" in low or "res" in low or "metric" in low):
                kw[pname] = stress_sim
            elif ("sim" in low or "baseline" in low) and ("res" in low or "metric" in low or "sim" in low):
                kw[pname] = base_sim
            elif low == "allow_synthetic_in_test":
                kw[pname] = False
            elif param.default is inspect.Parameter.empty:
                if "dq" in low or "g1" in low:
                    kw[pname] = g1_passed
                elif "factor" in low or "g2" in low:
                    kw[pname] = g2_passed
                elif "plateau" in low or "g4" in low:
                    kw[pname] = g4_passed
                elif "dsr" in low or "cpcv" in low or "nested" in low or "g5" in low:
                    kw[pname] = g5_passed
                elif "pbo" in low or "g6" in low:
                    kw[pname] = g6_passed
                elif "parity" in low or "g7" in low:
                    kw[pname] = g7_passed
                elif "bench" in low or "g8" in low:
                    kw[pname] = g8_passed
        rep = eval_all(**kw)
        gates_out: dict[str, bool] = {}
        raw_gates = getattr(rep, "gates", getattr(rep, "outcomes", []))
        if isinstance(raw_gates, dict):
            for gid, val in raw_gates.items():
                gates_out[str(gid)] = bool(getattr(val, "passed", val))
        else:
            for g in raw_gates:
                gid = str(getattr(g, "gate_id", getattr(g, "id", "")))
                gates_out[gid] = bool(getattr(g, "passed", False))
        verdict_str = normalize_verdict(get_metric(rep, "verdict"))
        return gates_out, verdict_str

    g3_fn = resolve_validation_callable(
        val_mod, "evaluate_g3_cost_edge", "evaluate_g3", "check_g3_cost_edge", "evaluate_cost_first_edge"
    )
    verdict_fn = resolve_validation_callable(
        val_mod, "determine_verdict", "compute_verdict", "decide_verdict", "evaluate_verdict"
    )
    sig3 = inspect.signature(g3_fn)
    g3_out = (
        g3_fn(
            baseline_sharpe=baseline_sharpe,
            baseline_net_pnl=baseline_net_pnl,
            stress_sharpe=stress_sharpe,
            stress_net_pnl=stress_net_pnl,
        )
        if "stress_sharpe" in sig3.parameters
        else g3_fn(baseline_sharpe, baseline_net_pnl, stress_sharpe, stress_net_pnl)
    )
    g3_ok = bool(get_metric(g3_out, "passed"))
    gate_map = {
        "G1": g1_passed,
        "G2": g2_passed,
        "G3": g3_ok,
        "G4": g4_passed,
        "G5": g5_passed,
        "G6": g6_passed,
        "G7": g7_passed,
        "G8": g8_passed,
    }
    return gate_map, normalize_verdict(verdict_fn(gate_map))


def build_scripted_strategy(
    weights_by_step: Sequence[dict[str, float]],
    *,
    strategy_id: str = "scripted_e2e_strategy",
    asset_domain: str = "web3",
    warmup_bars: int = 0,
    inertia_band: float = 0.0,
    stop_by_step: Sequence[dict[str, float | None]] | None = None,
    signals_by_step: Sequence[dict[str, float]] | None = None,
) -> Any:
    """Construct a minimal AssetNativeStrategy concrete implementation for Core Engine tests."""
    iface = import_core_interfaces()
    BarDecision = iface.BarDecision
    BaseCls = getattr(iface, "AssetNativeStrategy", object)
    # If Protocol/ABC allows subclassing, inherit cleanly; otherwise use plain structural class
    use_base = BaseCls if inspect.isclass(BaseCls) and not getattr(BaseCls, "_is_protocol", False) else object

    class _ScriptedStrategy(use_base):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            self.strategy_id = strategy_id
            self.asset_domain = asset_domain
            self.warmup_bars = warmup_bars
            self._step = 0
            self._exec_weights: dict[str, float] = {}
            self.sync_history: list[dict[str, float]] = []

        def reset(self) -> None:
            self._step = 0
            self._exec_weights = {}
            self.sync_history = []

        def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
            self._exec_weights = {k: float(v) for k, v in executed_weights.items()}
            self.sync_history.append(dict(self._exec_weights))

        def clone_fresh(self) -> Any:
            return build_scripted_strategy(
                weights_by_step,
                strategy_id=strategy_id,
                asset_domain=asset_domain,
                warmup_bars=warmup_bars,
                inertia_band=inertia_band,
                stop_by_step=stop_by_step,
                signals_by_step=signals_by_step,
            )

        def on_bar_step(
            self,
            ts_ms: int,
            bars_at_ts: dict[str, Any],
            context: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            idx = self._step
            self._step += 1
            desired_map = weights_by_step[idx] if idx < len(weights_by_step) else {}
            stop_map = stop_by_step[idx] if (stop_by_step is not None and idx < len(stop_by_step)) else {}
            sig_map = signals_by_step[idx] if (signals_by_step is not None and idx < len(signals_by_step)) else {}
            out: dict[str, Any] = {}
            for sym in sorted(bars_at_ts.keys()):
                desired = float(desired_map.get(sym, 0.0))
                prev_w = float(self._exec_weights.get(sym, 0.0))
                if inertia_band > 0.0 and abs(desired - prev_w) < inertia_band and desired != 0.0:
                    chosen = prev_w
                else:
                    chosen = desired
                raw_sig = float(sig_map.get(sym, chosen * 10.0))
                out[sym] = BarDecision(
                    symbol=sym,
                    ts_event_ms=int(ts_ms),
                    target_weight=chosen,
                    raw_signal=raw_sig,
                    regime_label="SCRIPTED",
                    stop_price=stop_map.get(sym),
                    metadata={"step": idx, "synced_prev_w": prev_w},
                )
            return out

    return _ScriptedStrategy()


class _ParityObserverStrategy:
    """Structural AssetNativeStrategy wrapper counting on_bar_step and sync_executed_weights calls."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.strategy_id: str = str(getattr(inner, "strategy_id", "observer"))
        self.asset_domain: str = str(getattr(inner, "asset_domain", "web3"))
        self.warmup_bars: int = int(getattr(inner, "warmup_bars", 0))
        self.on_bar_step_calls: int = 0
        self.sync_calls: int = 0

    def reset(self) -> None:
        self.on_bar_step_calls = 0
        self.sync_calls = 0
        self._inner.reset()

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.on_bar_step_calls += 1
        return self._inner.on_bar_step(ts_ms, bars_at_ts, context)

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        self.sync_calls += 1
        self._inner.sync_executed_weights(executed_weights)

    def clone_fresh(self) -> Any:
        return _ParityObserverStrategy(self._inner.clone_fresh())


def _extract_weight_map(entry: Any) -> dict[str, float]:
    raw = entry[1] if isinstance(entry, tuple) and len(entry) >= 2 and isinstance(entry[1], dict) else entry
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    return {}


def _extract_equity_val(entry: Any) -> float:
    return float(entry[1]) if isinstance(entry, (tuple, list)) else float(entry)


def _max_dict_diff(a: dict[str, float], b: dict[str, float]) -> float:
    worst = 0.0
    for sym in set(a.keys()) | set(b.keys()):
        d = abs(float(a.get(sym, 0.0)) - float(b.get(sym, 0.0)))
        if d > worst:
            worst = d
    return worst


def run_streaming_vs_batch_check(
    strategy: Any,
    specs: dict[str, Any],
    bars_by_symbol: dict[str, Sequence[Any]],
    *,
    risk_cfg: Any,
    capital_usd: float = 10_000.0,
    context_provider: Callable[[int], dict[str, Any]] | None = None,
    require_active: bool = True,
    accounting_engine: Any | None = None,
) -> tuple[float, float]:
    """4-layer opaque-box G7 streaming-vs-batch & causality oracle across kernel + RiskAllocator.

    Layer 1: Instrumented batch simulate() verifying on_bar_step & sync_executed_weights invocation
             counts (>= non-halted bars, killing lookahead M6A & no-sync M6B) + optional activity guard.
    Layer 2: Production core_engine.validation.verify_full_parity verification.
    Layer 3: Independent StreamingAccountingSession + manual CausalRiskAllocator cross-check DIRECTLY
             against res_batch.allocated_weights_history / executed_weights_history / equity_curve.
    Layer 4: Hostile 60% prefix-truncation AND same-length 10x future-tail shock causality probes.

    Returns (max_abs_weight_delta, max_abs_equity_delta).
    """
    acct_mod = import_accounting_mod()
    val_mod = import_core_validation_mod()
    engine = accounting_engine if accounting_engine is not None else acct_mod.PortfolioAccountingEngine()

    timeline = sorted(
        {int(b.ts_event_ms) for sym, bars in bars_by_symbol.items() if sym in specs for b in bars}
    )
    if not timeline:
        raise AssertionError("run_streaming_vs_batch_check requires non-empty bars_by_symbol")

    by_ts: dict[int, dict[str, Any]] = {ts: {} for ts in timeline}
    for sym, bars in bars_by_symbol.items():
        if sym not in specs:
            continue
        for b in bars:
            by_ts[int(b.ts_event_ms)][sym] = b

    # --- Layer 1: Instrumented batch simulate() ---
    obs_batch = _ParityObserverStrategy(strategy.clone_fresh())
    res_batch = engine.simulate(
        obs_batch,
        specs,
        bars_by_symbol,
        risk_cfg=risk_cfg,
        capital_usd=capital_usd,
        context_provider=context_provider,
    )
    eq_batch = list(get_metric(res_batch, "equity_curve"))
    alloc_hist_batch = list(get_metric(res_batch, "allocated_weights_history"))
    exec_hist_batch = list(get_metric(res_batch, "executed_weights_history"))

    if len(eq_batch) != len(timeline):
        raise AssertionError(
            f"G7 violation: equity_curve length {len(eq_batch)} != timeline length {len(timeline)}"
        )
    if len(alloc_hist_batch) != len(timeline):
        raise AssertionError(
            f"G7 violation: allocated_weights_history length {len(alloc_hist_batch)} != timeline {len(timeline)}"
        )
    if len(exec_hist_batch) != len(timeline):
        raise AssertionError(
            f"G7 violation: executed_weights_history length {len(exec_hist_batch)} != timeline {len(timeline)}"
        )

    # Count non-halted bars (starting equity > 1e-6) to accommodate early bankruptcy halts
    non_halted = 1
    for idx in range(len(eq_batch) - 1):
        if _extract_equity_val(eq_batch[idx]) > 1e-6:
            non_halted += 1
    non_halted = min(non_halted, len(timeline))

    if obs_batch.on_bar_step_calls < non_halted:
        raise AssertionError(
            f"G7 violation (M6A): engine.simulate invoked on_bar_step {obs_batch.on_bar_step_calls} < {non_halted} bars"
        )
    if obs_batch.sync_calls < non_halted:
        raise AssertionError(
            f"G7 violation (M6B): engine.simulate invoked sync_executed_weights {obs_batch.sync_calls} < {non_halted} bars"
        )

    if require_active:
        exec_trades, max_abs_exec_w = measure_strategy_activity(res_batch)
        if exec_trades <= 0 or max_abs_exec_w <= 1e-6:
            raise AssertionError(
                f"G7 anti-stub activity failure: strategy produced trade_count={exec_trades}, "
                f"max_abs_exec_w={max_abs_exec_w:.6e} (expected active non-zero execution)"
            )

    max_w_delta = 0.0
    max_eq_delta = 0.0

    # --- Layer 2: Production verify_full_parity ---
    vfp = getattr(val_mod, "verify_full_parity", None) or getattr(val_mod, "verify_full_pipeline_parity", None)
    if callable(vfp):
        vfp_sig = inspect.signature(vfp)
        vfp_kw: dict[str, Any] = {
            "risk_cfg": risk_cfg,
            "batch_result": res_batch,
            "capital_usd": capital_usd,
            "context_provider": context_provider,
        }
        if "accounting_engine" in vfp_sig.parameters:
            vfp_kw["accounting_engine"] = engine
        prep = vfp(strategy.clone_fresh(), specs, bars_by_symbol, **vfp_kw)
        for attr in ("max_abs_raw_weight_delta", "max_abs_weight_delta", "prefix_max_weight_delta"):
            val = getattr(prep, attr, None)
            if val is not None:
                max_w_delta = max(max_w_delta, float(val))
        for attr in ("max_abs_equity_delta", "prefix_max_equity_delta"):
            val = getattr(prep, attr, None)
            if val is not None:
                max_eq_delta = max(max_eq_delta, float(val))
        if not bool(getattr(prep, "passed", False)):
            raise AssertionError(
                f"G7 violation: verify_full_parity failed: detail={getattr(prep, 'detail', prep)}"
            )
        if int(getattr(prep, "regime_mismatch_count", 0)) != 0:
            raise AssertionError(
                f"G7 violation: verify_full_parity regime_mismatch_count={prep.regime_mismatch_count}"
            )

    # --- Layer 3: Independent StreamingAccountingSession + manual CausalRiskAllocator vs res_batch ---
    create_sess = getattr(engine, "create_session", None)
    session = (
        create_sess(strategy.clone_fresh(), specs, risk_cfg=risk_cfg, capital_usd=capital_usd)
        if callable(create_sess)
        else None
    )
    s_manual = strategy.clone_fresh()
    s_manual.reset()
    allocator = make_risk_allocator(risk_cfg)
    maxlen = max(128, int(getattr(risk_cfg, "corr_lookback_bars", 60)) + 5)
    recent_rets: dict[str, deque[float]] = {sym: deque(maxlen=maxlen) for sym in specs}
    last_close: dict[str, float] = {}
    had_override = False

    for idx, ts in enumerate(timeline):
        bars_at_ts = by_ts[ts]
        ctx = context_provider(ts) if context_provider is not None else None
        batch_alloc = _extract_weight_map(alloc_hist_batch[idx])
        batch_exec = _extract_weight_map(exec_hist_batch[idx])
        batch_eq = _extract_equity_val(eq_batch[idx])

        if session is not None:
            step_res = session.step_bar(ts, bars_at_ts, context=ctx)
            max_eq_delta = max(max_eq_delta, abs(float(step_res.equity_usd) - batch_eq))
            max_w_delta = max(max_w_delta, _max_dict_diff(_extract_weight_map(step_res.allocated_weights), batch_alloc))
            max_w_delta = max(max_w_delta, _max_dict_diff(_extract_weight_map(step_res.executed_weights), batch_exec))
            if bool(getattr(step_res, "had_fill_override", False)):
                had_override = True

        # Independent manual causal kernel + RiskAllocator cross-check directly against res_batch
        for sym, bar in bars_at_ts.items():
            if sym in last_close and last_close[sym] > 0.0:
                recent_rets[sym].append(float(bar.close) / last_close[sym] - 1.0)
            last_close[sym] = float(bar.close)
        step_tf_h = float(next(iter(bars_at_ts.values())).timeframe_h)
        decisions = s_manual.on_bar_step(ts, bars_at_ts, ctx)
        manual_alloc = call_risk_allocate(
            allocator, risk_cfg, ts, decisions, recent_rets, specs, timeframe_h=step_tf_h
        )
        s_manual.sync_executed_weights(manual_alloc)
        if not had_override and idx < non_halted:
            max_w_delta = max(max_w_delta, _max_dict_diff(manual_alloc, batch_alloc))

    # --- Layer 4: Hostile Prefix-Truncation AND Same-Length 10x Future-Tail Shock Causality ---
    if len(timeline) >= 6:
        cut = max(3, (len(timeline) * 3) // 5)
        ts_cut = timeline[cut - 1]
        prefix_bars = {
            sym: [b for b in bs if int(b.ts_event_ms) <= ts_cut]
            for sym, bs in bars_by_symbol.items()
        }
        mut_bars: dict[str, list[Any]] = {}
        for sym, bs in bars_by_symbol.items():
            mut_list: list[Any] = []
            for b in bs:
                if int(b.ts_event_ms) > ts_cut and dataclasses.is_dataclass(b):
                    mut_list.append(
                        dataclasses.replace(
                            b,
                            open=float(b.open) * 10.0,
                            high=float(b.high) * 10.0,
                            low=float(b.low) * 10.0,
                            close=float(b.close) * 10.0,
                            volume_usd=float(b.volume_usd) * 5.0,
                            carry_rate_period=float(b.carry_rate_period) * 5.0,
                        )
                    )
                else:
                    mut_list.append(b)
            mut_bars[sym] = mut_list

        for probe_label, probe_bars in (("prefix_truncation", prefix_bars), ("future_tail_10x_shock", mut_bars)):
            probe_res = engine.simulate(
                strategy.clone_fresh(),
                specs,
                probe_bars,
                risk_cfg=risk_cfg,
                capital_usd=capital_usd,
                context_provider=context_provider,
            )
            p_eq = list(get_metric(probe_res, "equity_curve"))
            p_alloc = list(get_metric(probe_res, "allocated_weights_history"))
            p_exec = list(get_metric(probe_res, "executed_weights_history"))
            if len(p_eq) < cut or len(p_alloc) < cut or len(p_exec) < cut:
                raise AssertionError(
                    f"G7 causality probe {probe_label} returned too few bars ({len(p_eq)} < {cut})"
                )
            for i in range(cut):
                max_eq_delta = max(max_eq_delta, abs(_extract_equity_val(p_eq[i]) - _extract_equity_val(eq_batch[i])))
                max_w_delta = max(
                    max_w_delta,
                    _max_dict_diff(_extract_weight_map(p_alloc[i]), _extract_weight_map(alloc_hist_batch[i])),
                    _max_dict_diff(_extract_weight_map(p_exec[i]), _extract_weight_map(exec_hist_batch[i])),
                )

    return max_w_delta, max_eq_delta


# ---------------------------------------------------------------------------
# Realistic Offline Sample Payloads for Collector Parsers (Tiers 1–2)
# ---------------------------------------------------------------------------

SAMPLE_TREASURY_TIPS_CSV = (
    "Date,5 YR,7 YR,10 YR,20 YR,30 YR\n"
    "11/18/2026,2.41,2.55,2.71,2.88,2.95\n"
    "11/19/2026,2.44,2.58,2.74,2.90,2.97\n"
    "11/20/2026,2.46,2.61,2.77,2.92,2.99\n"
)

SAMPLE_FED_FOMC_HTML = """<!DOCTYPE html>
<html>
<body>
<div class="panel panel-default">
  <div class="panel-heading"><h4>2026 FOMC Meetings</h4></div>
  <div class="row fomc-meeting">
    <div class="fomc-meeting__month">January</div>
    <div class="fomc-meeting__date">27-28</div>
  </div>
  <div class="row fomc-meeting">
    <div class="fomc-meeting__month">March</div>
    <div class="fomc-meeting__date">17-18*</div>
  </div>
  <div class="row fomc-meeting">
    <div class="fomc-meeting__month">December</div>
    <div class="fomc-meeting__date">8-9*</div>
  </div>
</div>
</body>
</html>
"""

SAMPLE_BLS_V2_JSON = json.dumps(
    {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "CUSR0000SA0",
                    "data": [
                        {"year": "2026", "period": "M10", "periodName": "October", "value": "315.45"},
                        {"year": "2026", "period": "M09", "periodName": "September", "value": "314.60"},
                    ],
                },
                {
                    "seriesID": "CES0000000001",
                    "data": [
                        {"year": "2026", "period": "M10", "periodName": "October", "value": "159210"},
                        {"year": "2026", "period": "M09", "periodName": "September", "value": "159030"},
                    ],
                },
            ]
        },
    }
)

SAMPLE_HL_CANDLES_JSON = json.dumps(
    [
        {
            "t": BASE_TS_MS,
            "T": BASE_TS_MS + 12 * HOUR_MS,
            "s": "BTC",
            "i": "12h",
            "o": "42000.0",
            "h": "42800.0",
            "l": "41900.0",
            "c": "42500.0",
            "v": "1250.5",
        },
        {
            "t": BASE_TS_MS + 12 * HOUR_MS,
            "T": BASE_TS_MS + 24 * HOUR_MS,
            "s": "BTC",
            "i": "12h",
            "o": "42500.0",
            "h": "43100.0",
            "l": "42400.0",
            "c": "43000.0",
            "v": "1420.0",
        },
    ]
)

SAMPLE_HL_FUNDING_JSON = json.dumps(
    [
        {"coin": "BTC", "fundingRate": "0.0000125", "premium": "0.0002", "time": BASE_TS_MS},
        {"coin": "BTC", "fundingRate": "0.0000150", "premium": "0.0003", "time": BASE_TS_MS + HOUR_MS},
    ]
)

SAMPLE_DERIBIT_DVOL_JSON = json.dumps(
    {
        "result": {
            "data": [
                [BASE_TS_MS, 51.2, 52.0, 50.5, 51.8],
                [BASE_TS_MS + DAY_MS, 51.8, 54.0, 51.0, 53.5],
            ]
        }
    }
)

SAMPLE_DEFILLAMA_STABLECOINS_JSON = json.dumps(
    [
        {"date": str(BASE_TS_MS // 1000), "totalCirculatingUSD": {"peggedUSD": 165_000_000_000.0}},
        {"date": str((BASE_TS_MS + DAY_MS) // 1000), "totalCirculatingUSD": {"peggedUSD": 166_200_000_000.0}},
    ]
)


# ---------------------------------------------------------------------------
# Clean-Clone Deterministic Sandbox Builders (DEF-01; NEVER spoofs REAL_EXCHANGE_API)
# ---------------------------------------------------------------------------

_CLEAN_CLONE_V4_TEMPLATE_DIR: Path | None = None


def _cleanup_clean_clone_template() -> None:
    global _CLEAN_CLONE_V4_TEMPLATE_DIR
    if _CLEAN_CLONE_V4_TEMPLATE_DIR is not None and _CLEAN_CLONE_V4_TEMPLATE_DIR.exists():
        shutil.rmtree(_CLEAN_CLONE_V4_TEMPLATE_DIR, ignore_errors=True)
        _CLEAN_CLONE_V4_TEMPLATE_DIR = None


atexit.register(_cleanup_clean_clone_template)


def _generate_weekday_dates_ms(count: int, end_ms: int) -> list[int]:
    """Generate `count` ascending UTC midnight weekday timestamps ending at or before `end_ms`."""
    cur_ms = int(end_ms)
    out: list[int] = []
    while len(out) < count:
        dt = datetime.fromtimestamp(cur_ms / 1000.0, tz=timezone.utc)
        if dt.weekday() < 5:
            out.append(cur_ms)
        cur_ms -= DAY_MS
    out.reverse()
    return out


def _populate_clean_clone_v4_lake(dst_dir: Path) -> Path:
    """Populate a deterministic, multi-year Option B sandbox lake marked SYNTHETIC_TEST_ONLY."""
    prov_mod = importlib.import_module("alphaloop.core.provenance")
    assets_mod = import_assets()
    models_mod = import_df_models()
    router_mod = import_df_router()

    prov_syn = str(prov_mod.PROVENANCE_SYNTHETIC)
    router = router_mod.DataFoundationRouter(dst_dir)
    try:
        # 1) Web3 domain: 4 liquid coins x 1,300 12h bars (649.5d >= 600d) with realistic cyclical trends & funding
        w3_end_ms = utc_ms(2026, 11, 22, 12, 0)
        w3_n = 1300
        w3_start_ms = w3_end_ms - (w3_n - 1) * 12 * HOUR_MS
        coin_cfg = {
            "BTC": (42000.0, 0.00055, 0.19, 34.0, 0.0, 0.00085, 0.00055, 11),
            "ETH": (2400.0, 0.00035, 0.24, 29.0, 1.1, 0.00075, 0.00060, 17),
            "SOL": (95.0, 0.00080, 0.30, 23.0, 2.3, 0.00105, 0.00065, 23),
            "AVAX": (32.0, 0.00020, 0.27, 27.0, 3.6, 0.00065, 0.00050, 31),
        }
        w3_bars: list[Any] = []
        for sym, (p0, drift, amp, period, phase, fund_base, fund_amp, seed) in coin_cfg.items():
            rng = random.Random(seed)
            px = p0
            for i in range(w3_n):
                ts_ev = w3_start_ms + i * 12 * HOUR_MS
                ts_av = ts_ev + 12 * HOUR_MS
                wave = amp * (
                    math.sin(2.0 * math.pi * (i + 1) / period + phase)
                    - math.sin(2.0 * math.pi * i / period + phase)
                )
                slow = 0.08 * (
                    math.cos(2.0 * math.pi * (i + 1) / 140.0 + phase * 0.5)
                    - math.cos(2.0 * math.pi * i / 140.0 + phase * 0.5)
                )
                ret = drift + wave + slow + rng.gauss(0.0, 0.011)
                o = px
                c = max(0.5, o * math.exp(max(-0.18, min(0.18, ret))))
                hi = max(o, c) * 1.008
                lo = min(o, c) * 0.992
                # Strong positive carry regime in middle & upper waves so W3 passes 3x RT fee floor & trades cleanly
                carry = fund_base + fund_amp * math.sin(2.0 * math.pi * i / 90.0 + phase)
                if abs(carry) < 1e-6:
                    carry = 0.0001
                w3_bars.append(
                    models_mod.CanonicalBar(
                        asset_class=assets_mod.ASSET_CLASS_CRYPTO_PERP,
                        venue="hyperliquid",
                        symbol=sym,
                        timeframe_h=12.0,
                        timeframe_sec=43_200,
                        ts_event_ms=ts_ev,
                        ts_avail_ms=ts_av,
                        open=round(o, 4),
                        high=round(hi, 4),
                        low=round(lo, 4),
                        close=round(c, 4),
                        volume_usd=round(50_000_000.0 * (1.0 + 0.3 * math.sin(i / 15.0)), 2),
                        carry_rate_period=round(carry, 8),
                        carry_interval_h=12.0,
                        spread_bps=2.0,
                        provenance=prov_syn,
                        source_fingerprint=f"clean_clone_sandbox:web3:{sym}",
                    )
                )
                # Matching Binance Spot 12h bar with small mean-reverting basis for W3-DN dual-leg tests
                basis_bps = 6.0 * math.sin(2.0 * math.pi * i / 30.0 + phase)
                s_mult = 1.0 / (1.0 + basis_bps / 10_000.0)
                w3_bars.append(
                    models_mod.CanonicalBar(
                        asset_class=assets_mod.ASSET_CLASS_CRYPTO_SPOT,
                        venue="binance_spot",
                        symbol=sym,
                        timeframe_h=12.0,
                        timeframe_sec=43_200,
                        ts_event_ms=ts_ev,
                        ts_avail_ms=ts_av,
                        open=round(o * s_mult, 4),
                        high=round(hi * s_mult, 4),
                        low=round(lo * s_mult, 4),
                        close=round(c * s_mult, 4),
                        volume_usd=round(80_000_000.0 * (1.0 + 0.25 * math.sin(i / 17.0)), 2),
                        carry_rate_period=0.0,
                        carry_interval_h=12.0,
                        spread_bps=1.5,
                        provenance=prov_syn,
                        source_fingerprint=f"clean_clone_sandbox:binance_spot:{sym}:12h",
                    )
                )
                px = c
        router.upsert_bars("web3", w3_bars, allow_synthetic_in_test=True)

        # Web3 derivatives metrics (OI/LSR) & macro (BTC_DVOL, STABLECOIN_MCAP_USD)
        w3_metrics: list[Any] = []
        w3_macro: list[Any] = []
        for d_idx in range(650):
            ts_ev = w3_start_ms + d_idx * DAY_MS
            ts_av = ts_ev + DAY_MS
            w3_metrics.append(
                models_mod.Web3DerivativesMetric(
                    venue="binance_um",
                    symbol="BTC",
                    ts_event_ms=ts_ev,
                    ts_avail_ms=ts_av,
                    open_interest_usd=round(8_000_000_000.0 + 2_500_000_000.0 * math.sin(d_idx / 28.0), 2),
                    top_trader_lsr=round(1.15 + 0.35 * math.cos(d_idx / 21.0), 4),
                    account_lsr=round(1.25 + 0.30 * math.sin(d_idx / 19.0), 4),
                    taker_ls_vol_ratio=round(1.02 + 0.18 * math.sin(d_idx / 11.0), 4),
                    provenance=prov_syn,
                    source_fingerprint="clean_clone_sandbox:binance_um:BTC",
                )
            )
            w3_macro.append(
                models_mod.Web3MarketMacroPoint(
                    source="deribit",
                    series_id="BTC_DVOL",
                    ts_event_ms=ts_ev,
                    ts_avail_ms=ts_av,
                    value=round(52.0 + 14.0 * math.sin(d_idx / 35.0), 2),
                    provenance=prov_syn,
                    source_fingerprint="clean_clone_sandbox:deribit:BTC_DVOL",
                )
            )
            w3_macro.append(
                models_mod.Web3MarketMacroPoint(
                    source="defillama",
                    series_id="STABLECOIN_MCAP_USD",
                    ts_event_ms=ts_ev,
                    ts_avail_ms=ts_av,
                    value=round(165_000_000_000.0 + d_idx * 85_000_000.0 + 4_000_000_000.0 * math.sin(d_idx / 40.0), 2),
                    provenance=prov_syn,
                    source_fingerprint="clean_clone_sandbox:defillama:STABLECOIN_MCAP_USD",
                )
            )
        router.upsert_web3_metrics(w3_metrics, allow_synthetic_in_test=True)
        router.upsert_web3_macro(w3_macro, allow_synthetic_in_test=True)

        # 2) Gold/Macro domain: 1,350 weekday 1d bars for GC=F (>1,850 calendar days) + 600 tail days for macro basket
        gold_1d_ts = _generate_weekday_dates_ms(1350, utc_ms(2026, 11, 20, 0, 0))
        gold_tail_ts = gold_1d_ts[-600:]
        gold_bars: list[Any] = []
        gc_close_by_ts: dict[int, float] = {}

        # Generate GC=F 1d with multi-regime oscillating trends (uncorrelated phase vs BTC)
        rng_gc = random.Random(101)
        px_gc = 1820.0
        for idx, ts_ev in enumerate(gold_1d_ts):
            wave = 0.16 * (
                math.sin(2.0 * math.pi * (idx + 1) / 95.0 + 0.7)
                - math.sin(2.0 * math.pi * idx / 95.0 + 0.7)
            )
            ret = 0.00032 + wave + rng_gc.gauss(0.0, 0.0075)
            o = px_gc
            c = max(1200.0, o * math.exp(max(-0.06, min(0.06, ret))))
            gc_close_by_ts[ts_ev] = c
            gold_bars.append(
                models_mod.CanonicalBar(
                    asset_class=assets_mod.ASSET_CLASS_GOLD_TRADFI,
                    venue="yahoo",
                    symbol="GC=F",
                    timeframe_h=24.0,
                    timeframe_sec=86_400,
                    ts_event_ms=ts_ev,
                    ts_avail_ms=ts_ev + 22 * HOUR_MS,
                    open=round(o, 2),
                    high=round(max(o, c) * 1.006, 2),
                    low=round(min(o, c) * 0.994, 2),
                    close=round(c, 2),
                    volume_usd=180_000_000.0,
                    carry_rate_period=0.0,
                    carry_interval_h=24.0,
                    spread_bps=1.5,
                    provenance=prov_syn,
                    source_fingerprint="clean_clone_sandbox:yahoo:GC=F:1d",
                    contract_multiplier=100.0,
                )
            )
            px_gc = c

        # Complementary 1d macro series aligned on last 600 weekdays
        macro_specs_cfg = (
            ("GLD", assets_mod.ASSET_CLASS_GOLD_TRADFI, 1.0, 0.095, 1.0, 102),
            ("SI=F", assets_mod.ASSET_CLASS_MACRO_INDEX, 5000.0, 0.0115, 1.15, 103),
            ("DX-Y.NYB", assets_mod.ASSET_CLASS_MACRO_INDEX, 1.0, 103.5, -0.45, 104),
            ("IEF", assets_mod.ASSET_CLASS_MACRO_INDEX, 1.0, 96.0, 0.35, 105),
            ("TIP", assets_mod.ASSET_CLASS_MACRO_INDEX, 1.0, 108.0, 0.40, 106),
            ("GDX", assets_mod.ASSET_CLASS_MACRO_INDEX, 1.0, 36.0, 1.30, 107),
            ("^GVZ", assets_mod.ASSET_CLASS_MACRO_INDEX, 1.0, 16.5, -0.20, 108),
            ("^VIX", assets_mod.ASSET_CLASS_MACRO_INDEX, 1.0, 18.0, -0.25, 109),
        )
        gc_tail_base = gc_close_by_ts[gold_tail_ts[0]]
        for sym, ac, mult, scale_or_base, beta, seed in macro_specs_cfg:
            rng_m = random.Random(seed)
            prev_c = gc_tail_base * scale_or_base if sym in ("GLD", "SI=F") else scale_or_base
            for idx, ts_ev in enumerate(gold_tail_ts):
                gc_rel = gc_close_by_ts[ts_ev] / gc_tail_base
                if sym == "GLD":
                    target = gc_close_by_ts[ts_ev] * scale_or_base * (1.0 + 0.002 * math.sin(idx / 12.0))
                elif sym == "SI=F":
                    target = gc_close_by_ts[ts_ev] * scale_or_base * (1.0 + 0.05 * math.sin(idx / 30.0))
                else:
                    target = scale_or_base * (gc_rel ** beta) * (1.0 + 0.06 * math.sin(idx / 24.0 + seed))
                o = prev_c
                c = max(2.0, target * math.exp(rng_m.gauss(0.0, 0.004)))
                gold_bars.append(
                    models_mod.CanonicalBar(
                        asset_class=ac,
                        venue="yahoo",
                        symbol=sym,
                        timeframe_h=24.0,
                        timeframe_sec=86_400,
                        ts_event_ms=ts_ev,
                        ts_avail_ms=ts_ev + 22 * HOUR_MS,
                        open=round(o, 2),
                        high=round(max(o, c) * 1.005, 2),
                        low=round(min(o, c) * 0.995, 2),
                        close=round(c, 2),
                        volume_usd=80_000_000.0,
                        carry_rate_period=0.0,
                        carry_interval_h=24.0,
                        spread_bps=1.5,
                        provenance=prov_syn,
                        source_fingerprint=f"clean_clone_sandbox:yahoo:{sym}:1d",
                        contract_multiplier=mult,
                    )
                )
                prev_c = c

        # GC=F 4h bars: 640 weekdays x 5 bars/day = 3,200 bars (>880 calendar days >= 700d) ending 2026-12-11
        gc_4h_days = _generate_weekday_dates_ms(640, utc_ms(2026, 12, 11, 0, 0))
        rng_4h = random.Random(202)
        px_4h = 2350.0
        bar_idx = 0
        for day_ms in gc_4h_days:
            for slot_h in (0, 4, 8, 12, 16):
                ts_ev = day_ms + slot_h * HOUR_MS
                ts_av = ts_ev + 4 * HOUR_MS
                wave = 0.055 * (
                    math.sin(2.0 * math.pi * (bar_idx + 1) / 55.0)
                    - math.sin(2.0 * math.pi * bar_idx / 55.0)
                )
                ret = 0.00012 + wave + rng_4h.gauss(0.0, 0.0038)
                o = px_4h
                c = max(1600.0, o * math.exp(max(-0.035, min(0.035, ret))))
                gold_bars.append(
                    models_mod.CanonicalBar(
                        asset_class=assets_mod.ASSET_CLASS_GOLD_TRADFI,
                        venue="yahoo",
                        symbol="GC=F",
                        timeframe_h=4.0,
                        timeframe_sec=14_400,
                        ts_event_ms=ts_ev,
                        ts_avail_ms=ts_av,
                        open=round(o, 2),
                        high=round(max(o, c) * 1.0035, 2),
                        low=round(min(o, c) * 0.9965, 2),
                        close=round(c, 2),
                        volume_usd=35_000_000.0,
                        carry_rate_period=0.0,
                        carry_interval_h=4.0,
                        spread_bps=1.5,
                        provenance=prov_syn,
                        source_fingerprint="clean_clone_sandbox:yahoo:GC=F:4h",
                        contract_multiplier=100.0,
                    )
                )
                px_4h = c
                bar_idx += 1

        # PAXG 12h perp (hyperliquid) + spot (binance_spot): 1,150 bars (574.5d >= 550d) in gold_macro domain
        paxg_n = 1150
        paxg_end_ms = utc_ms(2026, 11, 22, 12, 0)
        paxg_start_ms = paxg_end_ms - (paxg_n - 1) * 12 * HOUR_MS
        rng_px = random.Random(303)
        px_spot = 2380.0
        basis_rows: list[Any] = []
        for i in range(paxg_n):
            ts_ev = paxg_start_ms + i * 12 * HOUR_MS
            ts_av = ts_ev + 12 * HOUR_MS
            wave = 0.14 * (
                math.sin(2.0 * math.pi * (i + 1) / 80.0 + 0.4)
                - math.sin(2.0 * math.pi * i / 80.0 + 0.4)
            )
            ret = 0.00025 + wave + rng_px.gauss(0.0, 0.0055)
            o_s = px_spot
            c_s = max(1600.0, o_s * math.exp(max(-0.05, min(0.05, ret))))
            basis_bps = 18.0 * math.sin(2.0 * math.pi * i / 28.0)
            o_p = o_s * (1.0 + basis_bps / 10_000.0)
            c_p = c_s * (1.0 + basis_bps / 10_000.0)
            gold_bars.append(
                models_mod.CanonicalBar(
                    asset_class=assets_mod.ASSET_CLASS_GOLD_PERP,
                    venue="hyperliquid",
                    symbol="PAXG",
                    timeframe_h=12.0,
                    timeframe_sec=43_200,
                    ts_event_ms=ts_ev,
                    ts_avail_ms=ts_av,
                    open=round(o_p, 2),
                    high=round(max(o_p, c_p) * 1.004, 2),
                    low=round(min(o_p, c_p) * 0.996, 2),
                    close=round(c_p, 2),
                    volume_usd=14_000_000.0,
                    carry_rate_period=round(0.00012 + 0.00008 * math.sin(i / 25.0), 8),
                    carry_interval_h=12.0,
                    spread_bps=3.0,
                    provenance=prov_syn,
                    source_fingerprint="clean_clone_sandbox:hyperliquid:PAXG:12h",
                    premium_basis_bps=round(basis_bps, 2),
                )
            )
            gold_bars.append(
                models_mod.CanonicalBar(
                    asset_class=assets_mod.ASSET_CLASS_CRYPTO_SPOT,
                    venue="binance_spot",
                    symbol="PAXG",
                    timeframe_h=12.0,
                    timeframe_sec=43_200,
                    ts_event_ms=ts_ev,
                    ts_avail_ms=ts_av,
                    open=round(o_s, 2),
                    high=round(max(o_s, c_s) * 1.003, 2),
                    low=round(min(o_s, c_s) * 0.997, 2),
                    close=round(c_s, 2),
                    volume_usd=18_000_000.0,
                    carry_rate_period=0.0,
                    carry_interval_h=12.0,
                    spread_bps=2.5,
                    provenance=prov_syn,
                    source_fingerprint="clean_clone_sandbox:binance_spot:PAXG:12h",
                )
            )
            if i >= paxg_n - 120:
                dt = datetime.fromtimestamp(ts_ev / 1000.0, tz=timezone.utc)
                basis_rows.append(
                    models_mod.PAXGBasisPoint(
                        venue_perp="hyperliquid",
                        venue_spot="binance_spot",
                        ref_tradfi_symbol="GC=F",
                        timeframe_sec=43_200,
                        ts_event_ms=ts_ev,
                        ts_avail_ms=ts_av,
                        paxg_perp_close=round(c_p, 2),
                        paxg_spot_close=round(c_s, 2),
                        gc_ref_close=round(c_s, 2),
                        paxg_perp_spot_bps=round(basis_bps, 2),
                        paxg_spot_gc_bps=0.0,
                        paxg_perp_gc_bps=round(basis_bps, 2),
                        is_weekend_frozen=(dt.weekday() >= 5),
                        provenance=prov_syn,
                        source_fingerprint="clean_clone_sandbox:paxg_basis",
                    )
                )
            px_spot = c_s

        router.upsert_bars("gold_macro", gold_bars, allow_synthetic_in_test=True)
        router.upsert_paxg_basis(basis_rows, allow_synthetic_in_test=True)

        # US Treasury TIPS real yield curve (600 weekdays)
        yield_rows: list[Any] = []
        for idx, ts_ev in enumerate(gold_tail_ts):
            r10 = 1.85 - 0.35 * math.sin(2.0 * math.pi * idx / 95.0 + 0.7)
            yield_rows.append(
                models_mod.MacroYieldPoint(
                    source="us_treasury_tips",
                    ts_event_ms=ts_ev,
                    ts_avail_ms=ts_ev + 22 * HOUR_MS,
                    real_5y=round(r10 - 0.15, 3),
                    real_7y=round(r10 - 0.07, 3),
                    real_10y=round(r10, 3),
                    real_20y=round(r10 + 0.14, 3),
                    real_30y=round(r10 + 0.22, 3),
                    provenance=prov_syn,
                    source_fingerprint="clean_clone_sandbox:us_treasury_tips",
                )
            )
        router.upsert_yield_curve(yield_rows, allow_synthetic_in_test=True)

        # Scheduled 2025-2026 FOMC / NFP / CPI macro events (including exact Dec 2026 event week)
        event_rows: list[Any] = []
        for yr in (2025, 2026):
            for mo in range(1, 13):
                if yr == 2026 and mo == 12:
                    continue
                nfp_ts = utc_ms(yr, mo, 5, 13, 30)
                cpi_ts = utc_ms(yr, mo, 12, 13, 30)
                event_rows.append(
                    models_mod.MacroEventRelease(
                        event_type="NFP",
                        event_id=f"NFP-{yr}-{mo:02d}-05",
                        ref_period=f"{yr}-{mo:02d}",
                        ts_release_ms=nfp_ts,
                        ts_avail_ms=nfp_ts,
                        is_sep_meeting=False,
                        actual=175.0,
                        prior=165.0,
                        surprise_z=0.35,
                        source="bls_sandbox",
                        provenance=prov_syn,
                    )
                )
                event_rows.append(
                    models_mod.MacroEventRelease(
                        event_type="CPI",
                        event_id=f"CPI-{yr}-{mo:02d}-12",
                        ref_period=f"{yr}-{mo:02d}",
                        ts_release_ms=cpi_ts,
                        ts_avail_ms=cpi_ts,
                        is_sep_meeting=False,
                        actual=2.7,
                        prior=2.6,
                        surprise_z=0.20,
                        source="bls_sandbox",
                        provenance=prov_syn,
                    )
                )
                if mo in (1, 3, 5, 6, 7, 9, 11):
                    fomc_ts = utc_ms(yr, mo, 18, 18, 0)
                    event_rows.append(
                        models_mod.MacroEventRelease(
                            event_type="FOMC",
                            event_id=f"FOMC-{yr}-{mo:02d}-18",
                            ref_period=f"{yr}-{mo:02d}",
                            ts_release_ms=fomc_ts,
                            ts_avail_ms=fomc_ts,
                            is_sep_meeting=(mo in (3, 6, 9)),
                            actual=None,
                            prior=None,
                            surprise_z=None,
                            source="fed_sandbox",
                            provenance=prov_syn,
                        )
                    )
        # Exact Dec 2026 event week trio
        event_rows.extend(
            [
                models_mod.MacroEventRelease(
                    event_type="NFP",
                    event_id="NFP-2026-12-04",
                    ref_period="2026-11",
                    ts_release_ms=DEC_2026_NFP_MS,
                    ts_avail_ms=DEC_2026_NFP_MS,
                    is_sep_meeting=False,
                    source="bls_sandbox",
                    provenance=prov_syn,
                ),
                models_mod.MacroEventRelease(
                    event_type="FOMC",
                    event_id="FOMC-2026-12-09",
                    ref_period="2026-12",
                    ts_release_ms=DEC_2026_FOMC_MS,
                    ts_avail_ms=DEC_2026_FOMC_MS,
                    is_sep_meeting=True,
                    source="fed_sandbox",
                    provenance=prov_syn,
                ),
                models_mod.MacroEventRelease(
                    event_type="CPI",
                    event_id="CPI-2026-12-10",
                    ref_period="2026-11",
                    ts_release_ms=DEC_2026_CPI_MS,
                    ts_avail_ms=DEC_2026_CPI_MS,
                    is_sep_meeting=False,
                    source="bls_sandbox",
                    provenance=prov_syn,
                ),
            ]
        )
        router.upsert_macro_events(event_rows, allow_synthetic_in_test=True)

        # Export immutable jsonl.gz partitions & SHA-256 dataset_manifests
        router.export_domain_partition("web3", fmt="jsonl.gz")
        router.export_domain_partition("gold_macro", fmt="jsonl.gz")
    finally:
        close_router(router)
    return dst_dir


def build_clean_clone_v4_sandbox_lake(dst_dir: Path) -> Path:
    """Return an isolated sandbox copy of the deterministic clean-clone v4 lake (SYNTHETIC_TEST_ONLY)."""
    global _CLEAN_CLONE_V4_TEMPLATE_DIR
    if _CLEAN_CLONE_V4_TEMPLATE_DIR is None or not (_CLEAN_CLONE_V4_TEMPLATE_DIR / "catalog.sqlite").is_file():
        tpl = Path(tempfile.mkdtemp(prefix="e2e_v4_clean_clone_tpl_"))
        _populate_clean_clone_v4_lake(tpl)
        _CLEAN_CLONE_V4_TEMPLATE_DIR = tpl

    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "domains").mkdir(parents=True, exist_ok=True)
    for rel in ("catalog.sqlite", "domains/web3_lake.sqlite", "domains/gold_macro_lake.sqlite"):
        copy_sqlite_readonly_snapshot(_CLEAN_CLONE_V4_TEMPLATE_DIR / rel, dst_dir / rel)
    src_pq = _CLEAN_CLONE_V4_TEMPLATE_DIR / "parquet"
    dst_pq = dst_dir / "parquet"
    if src_pq.is_dir():
        if dst_pq.exists():
            shutil.rmtree(dst_pq)
        shutil.copytree(src_pq, dst_pq)
    return dst_dir


def build_clean_clone_v3_sandbox_lake(v3_path: Path) -> Path:
    """Construct a genuine legacy v3 SQLite WAL lake (SYNTHETIC_TEST_ONLY) for clean-clone migration tests."""
    prov_mod = importlib.import_module("alphaloop.core.provenance")
    assets_mod = import_assets()
    v3_mod = importlib.import_module("alphaloop.platform.data_lake")

    prov_syn = str(prov_mod.PROVENANCE_SYNTHETIC)
    lake = v3_mod.MultiAssetDataLake(v3_path)
    try:
        btc_spec = assets_mod.default_web3_specs(("BTC",))["BTC"]
        gc_spec = assets_mod.default_gold_specs()["GC=F"]
        lake.register_asset(btc_spec)
        lake.register_asset(gc_spec)

        btc_start = utc_ms(2026, 9, 1, 0, 0)
        btc_bars = []
        px = 62000.0
        for i in range(1050):
            ts_ev = btc_start + i * HOUR_MS
            c = px * (1.0 + 0.0008 * math.sin(i / 18.0))
            btc_bars.append(
                v3_mod.CanonicalBar(
                    asset_class=assets_mod.ASSET_CLASS_CRYPTO_PERP,
                    venue="hyperliquid",
                    symbol="BTC",
                    timeframe_h=1.0,
                    ts_event_ms=ts_ev,
                    ts_avail_ms=ts_ev + HOUR_MS,
                    open=round(px, 2),
                    high=round(max(px, c) * 1.003, 2),
                    low=round(min(px, c) * 0.997, 2),
                    close=round(c, 2),
                    volume_usd=25_000_000.0,
                    carry_rate_period=0.000012,
                    carry_interval_h=1.0,
                    spread_bps=2.0,
                    provenance=prov_syn,
                )
            )
            px = c

        gc_days = _generate_weekday_dates_ms(550, utc_ms(2026, 11, 20, 0, 0))
        gc_bars = []
        gpx = 2150.0
        for i, ts_ev in enumerate(gc_days):
            c = gpx * (1.0 + 0.002 * math.sin(i / 25.0))
            gc_bars.append(
                v3_mod.CanonicalBar(
                    asset_class=assets_mod.ASSET_CLASS_GOLD_TRADFI,
                    venue="yahoo",
                    symbol="GC=F",
                    timeframe_h=24.0,
                    ts_event_ms=ts_ev,
                    ts_avail_ms=ts_ev + 22 * HOUR_MS,
                    open=round(gpx, 2),
                    high=round(max(gpx, c) * 1.005, 2),
                    low=round(min(gpx, c) * 0.995, 2),
                    close=round(c, 2),
                    volume_usd=120_000_000.0,
                    carry_rate_period=0.0,
                    carry_interval_h=24.0,
                    spread_bps=1.5,
                    provenance=prov_syn,
                )
            )
            gpx = c

        lake.upsert_bars(btc_bars, allow_synthetic_in_test=True)
        lake.upsert_bars(gc_bars, allow_synthetic_in_test=True)
    finally:
        lake.close()
    return v3_path

