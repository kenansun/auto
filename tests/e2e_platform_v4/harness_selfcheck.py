#!/usr/bin/env python3
"""Independent self-verification harness for the Platform v4 E2E test suite.

Validates:
1. py_compile & AST validity of every .py file in tests/e2e_platform_v4/.
2. AST forensic policy scan on all test_tier*.py files:
   - ZERO @unittest.skip*, skipTest, @unittest.expectedFailure
   - ZERO third-party imports (pytest, numpy, pandas, scipy, pyarrow, duckdb, polars)
   - ZERO silent except: pass / except: ... swallows
   - ZERO time.monotonic / time.monotonic_ns on UTC/epoch lines
   - ZERO tautological assertTrue(True) / assertEqual(1, 1) stubs
   - ZERO top-level imports of target v4 modules (ensuring clean --list discovery)
   - ZERO direct sqlite3.connect calls outside fixtures.py
3. Deterministic fixtures.py math & helper accuracy (correlation, UTC calendar, WAL SQLite, strict JSON).
4. run_e2e.py --list exits 0 with 0 load errors and satisfies all per-tier, total (>=144),
   and per-Feature-Area (>=5 in Tier 1 AND >=5 in Tier 2 for FA-01..FA-11) counts.
5. Negative exit-code enforcement in an isolated TemporaryDirectory:
   - failing test -> exit 1
   - erroring test -> exit 1
   - skipped test -> exit 1
   - expectedFailure test -> exit 1
   - empty suite -> exit 1
   - genuinely passing suite -> exit 0
6. STD-34 audit_service_readiness.py --strict --include-tests --json exits 0 with
   files_scanned == len(all_py_files) and 0 violations.
7. Oracle mutation guard (anti-regression): verifies fixtures.run_streaming_vs_batch_check
   catches sabotaged lookahead & no-sync engines, evaluate_8_gates_with_overrides(g6_passed=False)
   returns G6=False & non-PROMOTE verdict, and evaluate_g8_contract satisfies exact SR/Calmar ordering.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import io
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import unittest
from typing import Any

SUITE_DIR = Path(__file__).resolve().parent
AUTO_ROOT = SUITE_DIR.parents[1]
if str(AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTO_ROOT))
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

if __package__:
    from . import fixtures, run_e2e
else:
    import fixtures
    import run_e2e

FORBIDDEN_THIRD_PARTY = frozenset(
    {"pytest", "numpy", "pandas", "scipy", "pyarrow", "duckdb", "polars", "sklearn", "statsmodels"}
)
FORBIDDEN_TOP_LEVEL_V4_PREFIXES = (
    "alphaloop.platform.data_foundation",
    "alphaloop.platform.core_engine",
    "alphaloop.platform.strategies",
)
UTC_EPOCH_SUBSTRINGS = ("ts_exch_ms", "ts_utc", "utc_ms", "epoch")
FA_IDS = tuple(f"fa{i:02d}" for i in range(1, 12))


def _check_compile_and_ast(py_files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
            src = path.read_text(encoding="utf-8")
            ast.parse(src, filename=str(path))
        except Exception as exc:
            errors.append(f"compile/parse failed for {path.name}: {exc}")
    return errors


def _check_ast_forensics(py_files: list[Path]) -> list[str]:
    violations: list[str] = []
    for path in py_files:
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines()
        tree = ast.parse(src, filename=str(path))
        is_tier_test = path.name.startswith("test_tier")

        # Check top-level imports in test_tier*.py for eager v4 imports
        if is_tier_test:
            for stmt in tree.body:
                if isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        if any(alias.name.startswith(p) for p in FORBIDDEN_TOP_LEVEL_V4_PREFIXES):
                            violations.append(
                                f"{path.name}:{stmt.lineno} eager top-level v4 import: {alias.name}"
                            )
                elif isinstance(stmt, ast.ImportFrom) and stmt.module:
                    if any(stmt.module.startswith(p) for p in FORBIDDEN_TOP_LEVEL_V4_PREFIXES):
                        violations.append(
                            f"{path.name}:{stmt.lineno} eager top-level v4 import: {stmt.module}"
                        )

        for node in ast.walk(tree):
            # Forbidden third-party imports everywhere in suite
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in FORBIDDEN_THIRD_PARTY:
                        violations.append(f"{path.name}:{node.lineno} forbidden third-party import {root}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_THIRD_PARTY:
                    violations.append(f"{path.name}:{node.lineno} forbidden third-party import {root}")

            # Silent except swallows anywhere in suite
            if isinstance(node, ast.ExceptHandler):
                if len(node.body) == 1:
                    first = node.body[0]
                    is_pass = isinstance(first, ast.Pass)
                    is_ellipsis = (
                        isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and first.value.value is Ellipsis
                    )
                    if is_pass or is_ellipsis:
                        violations.append(f"{path.name}:{node.lineno} silent except swallow")

            # time.monotonic on UTC/epoch line anywhere
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "time"
                    and node.func.attr in ("monotonic", "monotonic_ns")
                ):
                    line_txt = lines[node.lineno - 1] if 0 < node.lineno <= len(lines) else ""
                    if any(sub in line_txt for sub in UTC_EPOCH_SUBSTRINGS):
                        violations.append(f"{path.name}:{node.lineno} time.monotonic on UTC/epoch line")

            # Raw sqlite3.connect forbidden outside fixtures.py
            if path.name != "fixtures.py" and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "sqlite3"
                    and node.func.attr == "connect"
                ):
                    violations.append(f"{path.name}:{node.lineno} direct sqlite3.connect outside fixtures.py")

            if is_tier_test:
                # Zero skip / expectedFailure decorators
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    for dec in node.decorator_list:
                        dec_name = ""
                        if isinstance(dec, ast.Attribute):
                            dec_name = dec.attr
                        elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                            dec_name = dec.func.attr
                        elif isinstance(dec, ast.Name):
                            dec_name = dec.id
                        elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                            dec_name = dec.func.id
                        if dec_name in ("skip", "skipIf", "skipUnless", "expectedFailure"):
                            violations.append(f"{path.name}:{dec.lineno} forbidden decorator @{dec_name}")

                # Zero skipTest calls or tautological assertTrue(True) stubs
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "skipTest":
                        violations.append(f"{path.name}:{node.lineno} forbidden skipTest() call")
                    if node.func.attr == "assertTrue" and len(node.args) == 1:
                        arg0 = node.args[0]
                        if isinstance(arg0, ast.Constant) and arg0.value is True:
                            violations.append(f"{path.name}:{node.lineno} tautological assertTrue(True)")
                    if node.func.attr == "assertEqual" and len(node.args) == 2:
                        a0, a1 = node.args
                        if (
                            isinstance(a0, ast.Constant)
                            and isinstance(a1, ast.Constant)
                            and a0.value == a1.value
                        ):
                            violations.append(f"{path.name}:{node.lineno} tautological constant assertEqual")

    return violations


def _check_fixtures_integrity() -> list[str]:
    errors: list[str] = []
    # Exact UTC timestamp checks
    expected_ts = {
        "HYPE_LISTING_MS": (fixtures.HYPE_LISTING_MS, 1_733_356_800_000),
        "DEC_2026_NFP_MS": (fixtures.DEC_2026_NFP_MS, 1_796_391_000_000),
        "DEC_2026_FOMC_MS": (fixtures.DEC_2026_FOMC_MS, 1_796_842_800_000),
        "DEC_2026_CPI_MS": (fixtures.DEC_2026_CPI_MS, 1_796_909_400_000),
    }
    for label, (actual, expected) in expected_ts.items():
        if actual != expected:
            errors.append(f"Timestamp mismatch for {label}: {actual} != {expected}")

    # Verify calendar weekdays (Mon=0..Sun=6)
    weekdays = (
        ("WED_2026_11_18_1200_MS", fixtures.WED_2026_11_18_1200_MS, 2),
        ("FRI_2026_11_20_2200_MS", fixtures.FRI_2026_11_20_2200_MS, 4),
        ("SAT_2026_11_21_1200_MS", fixtures.SAT_2026_11_21_1200_MS, 5),
        ("SUN_2026_11_22_2200_MS", fixtures.SUN_2026_11_22_2200_MS, 6),
        ("MON_2026_11_23_0800_MS", fixtures.MON_2026_11_23_0800_MS, 0),
    )
    for label, ts_val, expected_wd in weekdays:
        wd = datetime.fromtimestamp(ts_val / 1000.0, tz=timezone.utc).weekday()
        if wd != expected_wd:
            errors.append(f"Weekday mismatch for {label}: got {wd}, expected {expected_wd}")

    # Correlated return matrix accuracy
    mat = fixtures.generate_correlated_return_matrix(10, 400, 0.90, annual_vol=0.30, periods_per_year=365.0, seed=77)
    if len(mat) != 10 or len(mat[0]) != 400:
        errors.append("generate_correlated_return_matrix shape mismatch")
    else:
        corrs = [fixtures.sample_pearson_corr(mat[0], mat[j]) for j in range(1, 10)]
        avg_corr = sum(corrs) / len(corrs)
        if abs(avg_corr - 0.90) > 0.04:
            errors.append(f"Correlated matrix mean pairwise corr {avg_corr:.4f} far from 0.90")
        vol0 = fixtures.annualized_vol(mat[0], 365.0)
        if abs(vol0 - 0.30) > 0.03:
            errors.append(f"Correlated matrix annual vol {vol0:.4f} far from 0.30")

    # Strict JSON NaN rejection
    dumps_rejected = False
    try:
        fixtures.strict_json_dumps({"bad": float("nan")})
    except ValueError:
        dumps_rejected = True
    if not dumps_rejected:
        errors.append("strict_json_dumps failed to reject NaN")

    loads_rejected = False
    try:
        fixtures.strict_json_loads('{"bad": NaN}')
    except ValueError:
        loads_rejected = True
    if not loads_rejected:
        errors.append("strict_json_loads failed to reject NaN")

    # SQLite WAL helpers inside isolated sandbox
    with fixtures.temp_sandbox(prefix="e2e_selfcheck_db_") as tmp:
        db_file = tmp / "sample.sqlite"
        conn = fixtures.open_sqlite_rw(db_file)
        try:
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
            conn.execute("INSERT INTO t VALUES (1, 'ok')")
            conn.commit()
        finally:
            conn.close()
        if str(mode).lower() != "wal":
            errors.append(f"open_sqlite_rw journal_mode expected wal, got {mode!r}")

        ro_conn = fixtures.open_sqlite_ro(db_file)
        try:
            row = ro_conn.execute("SELECT v FROM t WHERE id = 1").fetchone()
        finally:
            ro_conn.close()
        if row != ("ok",):
            errors.append(f"open_sqlite_ro returned unexpected row: {row!r}")

        v3_db = fixtures.create_synthetic_v3_lake_db(tmp / "v3.sqlite", n_web3_bars=20, n_gold_bars=20)
        snap_db = fixtures.copy_sqlite_readonly_snapshot(v3_db, tmp / "v3_copy.sqlite")
        c2 = fixtures.open_sqlite_ro(snap_db)
        try:
            cnt = c2.execute("SELECT COUNT(*) FROM canonical_bars").fetchone()[0]
        finally:
            c2.close()
        if cnt != 40:
            errors.append(f"v3 snapshot copy expected 40 bars, got {cnt}")

    return errors


def _check_discovery_and_fa_counts() -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    proc = subprocess.run(
        [sys.executable, str(SUITE_DIR / "run_e2e.py"), "--list", "--tier", "all"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        errors.append(f"run_e2e.py --list exited {proc.returncode}: {proc.stdout}\n{proc.stderr}")

    tier_counts: dict[str, int] = {}
    fa_counts_t1: dict[str, int] = {fa: 0 for fa in FA_IDS}
    fa_counts_t2: dict[str, int] = {fa: 0 for fa in FA_IDS}
    total = 0

    for tk in ("1", "2", "3", "4"):
        _, tids, load_errs = run_e2e.discover_tier(tk)
        tier_counts[f"tier_{tk}"] = len(tids)
        total += len(tids)
        if load_errs:
            errors.append(f"Tier {tk} load errors: {load_errs}")
        if len(tids) < run_e2e.MIN_TIER_COUNTS[tk]:
            errors.append(f"Tier {tk} count {len(tids)} < min {run_e2e.MIN_TIER_COUNTS[tk]}")
        if tk == "1":
            for tid in tids:
                m = re.search(r"fa(\d{2})", tid.lower())
                if m:
                    key = f"fa{m.group(1)}"
                    if key in fa_counts_t1:
                        fa_counts_t1[key] += 1
        elif tk == "2":
            for tid in tids:
                m = re.search(r"fa(\d{2})", tid.lower())
                if m:
                    key = f"fa{m.group(1)}"
                    if key in fa_counts_t2:
                        fa_counts_t2[key] += 1

    if total < run_e2e.MIN_TOTAL_COUNT:
        errors.append(f"Total discovered tests {total} < required {run_e2e.MIN_TOTAL_COUNT}")

    for fa in FA_IDS:
        if fa_counts_t1[fa] < 5:
            errors.append(f"Tier 1 {fa.upper()} count {fa_counts_t1[fa]} < 5")
        if fa_counts_t2[fa] < 5:
            errors.append(f"Tier 2 {fa.upper()} count {fa_counts_t2[fa]} < 5")

    return errors, {
        "total": total,
        "tier_counts": tier_counts,
        "fa_counts_t1": fa_counts_t1,
        "fa_counts_t2": fa_counts_t2,
    }


def _check_negative_exit_codes() -> list[str]:
    """Verify run_e2e.run_suite_strict strictly returns 1 on fail/error/skip/expectedFail/empty."""
    errors: list[str] = []

    class _PassCase(unittest.TestCase):
        def test_ok(self) -> None:
            self.assertEqual(2 + 2, 4)

    class _FailCase(unittest.TestCase):
        def test_fail(self) -> None:
            self.assertEqual(2 + 2, 5)

    class _ErrorCase(unittest.TestCase):
        def test_err(self) -> None:
            raise RuntimeError("intentional error")

    class _SkipCase(unittest.TestCase):
        def test_skip(self) -> None:
            self.skipTest("intentional skip")

    class _ExpFailCase(unittest.TestCase):
        @unittest.expectedFailure
        def test_xfail(self) -> None:
            self.assertEqual(1, 2)

    scenarios = (
        ("passing", unittest.TestSuite([_PassCase("test_ok")]), 0),
        ("failing", unittest.TestSuite([_FailCase("test_fail")]), 1),
        ("erroring", unittest.TestSuite([_ErrorCase("test_err")]), 1),
        ("skipping", unittest.TestSuite([_SkipCase("test_skip")]), 1),
        ("expected_failure", unittest.TestSuite([_ExpFailCase("test_xfail")]), 1),
        ("empty", unittest.TestSuite([]), 1),
    )
    for label, suite, expected_rc in scenarios:
        buf = io.StringIO()
        rc, summary = run_e2e.run_suite_strict(suite, verbosity=0, stream=buf)
        if rc != expected_rc:
            errors.append(
                f"Negative exit-code check {label!r}: got rc={rc}, expected {expected_rc} (summary={summary})"
            )
    return errors


def _check_std34_audit(py_files: list[Path]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    rc, payload, stderr = fixtures.run_audit_script(SUITE_DIR, strict=True, include_tests=True)
    if rc != 0:
        errors.append(f"STD-34 audit_service_readiness.py exited {rc}: {payload} {stderr}")
    scanned = int(payload.get("files_scanned", 0))
    if scanned != len(py_files):
        errors.append(f"STD-34 scanned {scanned} files, expected exact {len(py_files)}")
    if int(payload.get("high_severity_count", 0)) != 0:
        errors.append(f"STD-34 high_severity_count != 0: {payload.get('findings')}")
    if int(payload.get("medium_severity_count", 0)) != 0:
        errors.append(f"STD-34 medium_severity_count != 0: {payload.get('findings')}")
    if payload.get("findings"):
        errors.append(f"STD-34 non-empty findings: {payload.get('findings')}")
    return errors, payload


def _check_oracle_mutation_guard() -> list[str]:
    """Verify core fixture oracles fail loudly on sabotaged engines and preserve mathematical invariants."""
    errors: list[str] = []
    from types import SimpleNamespace

    assets = fixtures.import_assets()
    risk_mod = fixtures.import_risk_allocator_mod()
    acct_mod = fixtures.import_accounting_mod()
    val_mod = fixtures.import_core_validation_mod()

    # Guard 1: run_streaming_vs_batch_check must pass on real engine and FAIL on lookahead & no-sync engines
    specs = assets.default_web3_specs(("BTC",))
    bars = fixtures.build_bar_series(
        symbol="BTC",
        venue="hyperliquid",
        asset_class=assets.ASSET_CLASS_CRYPTO_PERP,
        timeframe_h=12.0,
        start_ms=fixtures.BASE_TS_MS,
        closes=[100.0 + float(i) for i in range(30)],
        carry_rates=[0.00005] * 30,
        carry_interval_h=12.0,
    )
    risk_cfg = risk_mod.RiskAllocatorConfig(
        max_positions=1,
        target_annual_vol_pct=200.0,
        max_portfolio_leverage=0.5,
        max_single_weight=0.5,
        use_correlation_idm=False,
    )
    strat = fixtures.build_scripted_strategy([{"BTC": 1.0}, {"BTC": 0.55}] * 15, inertia_band=0.15)

    w_ok, eq_ok = fixtures.run_streaming_vs_batch_check(
        strat, specs, {"BTC": bars}, risk_cfg=risk_cfg, require_active=True
    )
    if w_ok > 1e-10 or eq_ok > 1e-8:
        errors.append(f"oracle_mutation_guard: clean real engine failed G7 baseline ({w_ok}, {eq_ok})")

    class _SabotagedLookaheadEngine:
        def simulate(
            self,
            strategy: Any,
            sim_specs: dict[str, Any],
            bars_by_symbol: dict[str, Any],
            *,
            risk_cfg: Any,
            capital_usd: float = 10_000.0,
            context_provider: Any = None,
            **kwargs: Any,
        ) -> Any:
            b_list = list(next(iter(bars_by_symbol.values())))
            future_mult = float(b_list[-1].close) / max(float(b_list[0].close), 1e-9)
            return SimpleNamespace(
                equity_curve=[(int(b.ts_event_ms), capital_usd * (1.0 + 0.01 * future_mult)) for b in b_list],
                allocated_weights_history=[(int(b.ts_event_ms), {"BTC": 0.0}) for b in b_list],
                executed_weights_history=[{"BTC": 0.0} for _ in b_list],
                raw_decisions_history=[{} for _ in b_list],
                trade_count=0,
                trade_ledger=SimpleNamespace(executed=[]),
            )

    lookahead_caught = False
    try:
        w_bad, eq_bad = fixtures.run_streaming_vs_batch_check(
            strat,
            specs,
            {"BTC": bars},
            risk_cfg=risk_cfg,
            require_active=False,
            accounting_engine=_SabotagedLookaheadEngine(),
        )
        lookahead_caught = (w_bad > 1e-10) or (eq_bad > 1e-8)
    except (AssertionError, AttributeError, RuntimeError, ValueError) as exc:
        _caught_lookahead_msg = str(exc)
        lookahead_caught = bool(_caught_lookahead_msg)
    if not lookahead_caught:
        errors.append("oracle_mutation_guard: run_streaming_vs_batch_check tautologically PASSED sabotaged lookahead engine!")

    class _DropSyncProxy:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.strategy_id = str(getattr(inner, "strategy_id", "proxy"))
            self.asset_domain = str(getattr(inner, "asset_domain", "web3"))
            self.warmup_bars = int(getattr(inner, "warmup_bars", 0))
            self.dropped_sync_calls = 0

        def reset(self) -> None:
            self._inner.reset()

        def on_bar_step(self, ts_ms: int, bars_at_ts: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
            return self._inner.on_bar_step(ts_ms, bars_at_ts, context)

        def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
            self.dropped_sync_calls += 1

        def clone_fresh(self) -> Any:
            return _DropSyncProxy(self._inner.clone_fresh())

    class _SabotagedNoSyncEngine(acct_mod.PortfolioAccountingEngine):
        def simulate(self, strategy: Any, *args: Any, **kwargs: Any) -> Any:
            return super().simulate(_DropSyncProxy(strategy), *args, **kwargs)

    nosync_caught = False
    try:
        w_ns, eq_ns = fixtures.run_streaming_vs_batch_check(
            strat,
            specs,
            {"BTC": bars},
            risk_cfg=risk_cfg,
            require_active=False,
            accounting_engine=_SabotagedNoSyncEngine(),
        )
        nosync_caught = (w_ns > 1e-10) or (eq_ns > 1e-8)
    except (AssertionError, AttributeError, RuntimeError, ValueError) as exc:
        _caught_nosync_msg = str(exc)
        nosync_caught = bool(_caught_nosync_msg)
    if not nosync_caught:
        errors.append("oracle_mutation_guard: run_streaming_vs_batch_check tautologically PASSED sabotaged no-sync engine!")

    # Guard 2: evaluate_8_gates_with_overrides(g6_passed=False) must return G6 == False & non-PROMOTE verdict
    gates_fail, verdict_fail = fixtures.evaluate_8_gates_with_overrides(val_mod, g6_passed=False)
    if gates_fail.get("G6") is not False or verdict_fail == "PROMOTE_SHADOW_CANDIDATE":
        errors.append(f"oracle_mutation_guard: g6_passed=False drifted ({gates_fail}, {verdict_fail})")
    gates_ok, verdict_ok = fixtures.evaluate_8_gates_with_overrides(val_mod, g6_passed=True)
    if gates_ok.get("G6") is not True or verdict_ok != "PROMOTE_SHADOW_CANDIDATE":
        errors.append(f"oracle_mutation_guard: g6_passed=True baseline failed ({gates_ok}, {verdict_ok})")

    # Guard 3: evaluate_g8_contract Calmar-beat AND Sharpe-beat quadrants honor exact empirical ordering
    g8_cal = fixtures.evaluate_g8_contract(
        val_mod,
        strategy_sharpe=0.75,
        strategy_calmar=0.90,
        benchmark_sharpe=0.80,
        benchmark_calmar=0.55,
        corr_to_benchmark=0.85,
    )
    if not (
        float(fixtures.get_metric(g8_cal, "strategy_sharpe")) < float(fixtures.get_metric(g8_cal, "benchmark_sharpe"))
        and float(fixtures.get_metric(g8_cal, "strategy_calmar")) > float(fixtures.get_metric(g8_cal, "benchmark_calmar"))
        and bool(fixtures.get_metric(g8_cal, "passed_benchmark_gate", "passed"))
    ):
        errors.append(f"oracle_mutation_guard: evaluate_g8_contract Calmar-beat invariant failed: {g8_cal}")

    g8_sr = fixtures.evaluate_g8_contract(
        val_mod,
        strategy_sharpe=1.15,
        strategy_calmar=0.50,
        benchmark_sharpe=0.80,
        benchmark_calmar=0.55,
        corr_to_benchmark=0.85,
    )
    if not (
        float(fixtures.get_metric(g8_sr, "strategy_sharpe")) > float(fixtures.get_metric(g8_sr, "benchmark_sharpe"))
        and float(fixtures.get_metric(g8_sr, "strategy_calmar")) < float(fixtures.get_metric(g8_sr, "benchmark_calmar"))
        and bool(fixtures.get_metric(g8_sr, "passed_benchmark_gate", "passed"))
    ):
        errors.append(f"oracle_mutation_guard: evaluate_g8_contract Sharpe-beat invariant failed: {g8_sr}")

    return errors


def run_self_check(*, verbose: bool = False) -> tuple[int, dict[str, Any]]:
    """Run all 7 harness integrity checks and return (exit_code, report_dict)."""
    py_files = sorted(SUITE_DIR.glob("*.py"))
    compile_errors = _check_compile_and_ast(py_files)
    forensic_errors = _check_ast_forensics(py_files)
    fixture_errors = _check_fixtures_integrity()
    discovery_errors, discovery_meta = _check_discovery_and_fa_counts()
    neg_exit_errors = _check_negative_exit_codes()
    std34_errors, std34_payload = _check_std34_audit(py_files)
    oracle_guard_errors = _check_oracle_mutation_guard()

    checks = {
        "compile_and_ast": {"passed": len(compile_errors) == 0, "errors": compile_errors},
        "ast_forensics": {"passed": len(forensic_errors) == 0, "errors": forensic_errors},
        "fixtures_integrity": {"passed": len(fixture_errors) == 0, "errors": fixture_errors},
        "discovery_and_counts": {
            "passed": len(discovery_errors) == 0,
            "errors": discovery_errors,
            "meta": discovery_meta,
        },
        "negative_exit_codes": {"passed": len(neg_exit_errors) == 0, "errors": neg_exit_errors},
        "std34_strict_audit": {
            "passed": len(std34_errors) == 0,
            "errors": std34_errors,
            "files_scanned": std34_payload.get("files_scanned", 0),
        },
        "oracle_mutation_guard": {"passed": len(oracle_guard_errors) == 0, "errors": oracle_guard_errors},
    }
    all_passed = all(c["passed"] for c in checks.values())
    report: dict[str, Any] = {
        "mode": "self_check",
        "passed": all_passed,
        "exit_code": 0 if all_passed else 1,
        "py_files_checked": [p.name for p in py_files],
        "checks": checks,
    }

    print(f"=== E2E Harness Self-Check ({len(py_files)} Python files) ===")
    for name, info in checks.items():
        status = "PASS" if info["passed"] else "FAIL"
        print(f"  [{status}] {name}")
        if verbose or not info["passed"]:
            for err in info["errors"]:
                print(f"    - {err}")
    print(
        f"Discovery: total={discovery_meta['total']} "
        f"{discovery_meta['tier_counts']} | "
        f"STD-34 files_scanned={std34_payload.get('files_scanned', 0)}"
    )
    print(f"Self-Check Overall: {'PASS' if all_passed else 'FAIL'}")
    return (0 if all_passed else 1), report


if __name__ == "__main__":
    code, _ = run_self_check(verbose=True)
    raise SystemExit(code)
