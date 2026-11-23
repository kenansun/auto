#!/usr/bin/env python3
"""Independent pure-stdlib CLI runner for Platform v4 4-Tier Opaque-Box E2E Suite.

Usage:
    python3 run_e2e.py --tier 1|2|3|4|all [--verbose] [--json-out <path>]
    python3 run_e2e.py --list [--tier 1|2|3|4|all] [--json-out <path>]
    python3 run_e2e.py --self-check [--json-out <path>]

Strict Exit Code Contract:
    Exits 0 ONLY when tests_run > 0, load_errors == 0, failures == 0, errors == 0,
    skipped == 0, expected_failures == 0, and unexpected_successes == 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import unittest
from typing import Any, Sequence

SUITE_DIR = Path(__file__).resolve().parent
AUTO_ROOT = SUITE_DIR.parents[1]
if str(AUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTO_ROOT))
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

if __package__:
    from . import fixtures
else:
    import fixtures

TIER_PATTERNS: dict[str, str] = {
    "1": "test_tier1_*.py",
    "2": "test_tier2_*.py",
    "3": "test_tier3_*.py",
    "4": "test_tier4_*.py",
}

MIN_TIER_COUNTS: dict[str, int] = {
    "1": 55,
    "2": 55,
    "3": 22,
    "4": 10,
}
MIN_TOTAL_COUNT = 144


def iter_test_cases(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    """Flatten a nested unittest.TestSuite into concrete TestCase instances."""
    cases: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            cases.extend(iter_test_cases(item))
        else:
            cases.append(item)
    return cases


def discover_tier(tier_key: str, *, start_dir: Path = SUITE_DIR) -> tuple[unittest.TestSuite, list[str], list[str]]:
    """Discover all tests for a single tier ('1'..'4').

    Returns (suite, test_ids, load_errors).
    """
    pattern = TIER_PATTERNS[tier_key]
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(start_dir), pattern=pattern, top_level_dir=str(start_dir))
    test_ids: list[str] = []
    load_errors: list[str] = []
    for case in iter_test_cases(suite):
        tid = case.id()
        if case.__class__.__name__ == "_FailedTest" or "unittest.loader._FailedTest" in tid:
            load_errors.append(tid)
        else:
            test_ids.append(tid)
    return suite, test_ids, load_errors


def evaluate_result(
    result: unittest.TestResult,
    *,
    load_errors: Sequence[str] = (),
    elapsed_sec: float = 0.0,
) -> tuple[int, dict[str, Any]]:
    """Enforce the strict zero-skip, zero-failure exit-code policy."""
    tests_run = int(result.testsRun)
    n_failures = len(result.failures)
    n_errors = len(result.errors)
    n_skipped = len(result.skipped)
    n_exp_fail = len(result.expectedFailures)
    n_unexp_succ = len(result.unexpectedSuccesses)
    n_load_err = len(load_errors)

    passed = (
        tests_run > 0
        and n_load_err == 0
        and n_failures == 0
        and n_errors == 0
        and n_skipped == 0
        and n_exp_fail == 0
        and n_unexp_succ == 0
    )
    summary: dict[str, Any] = {
        "passed": passed,
        "exit_code": 0 if passed else 1,
        "tests_run": tests_run,
        "failures": n_failures,
        "errors": n_errors,
        "skipped": n_skipped,
        "expected_failures": n_exp_fail,
        "unexpected_successes": n_unexp_succ,
        "load_errors": list(load_errors),
        "elapsed_sec": round(float(elapsed_sec), 4),
        "failure_ids": [case.id() for case, _ in result.failures],
        "error_ids": [case.id() for case, _ in result.errors],
        "skipped_ids": [case.id() for case, _ in result.skipped],
    }
    return (0 if passed else 1), summary


def run_suite_strict(
    suite: unittest.TestSuite,
    *,
    verbosity: int = 1,
    stream: Any = None,
    load_errors: Sequence[str] = (),
) -> tuple[int, dict[str, Any]]:
    """Execute a TestSuite and evaluate strictly with zero tolerance for skips."""
    t0 = time.perf_counter()
    runner = unittest.TextTestRunner(
        stream=stream if stream is not None else sys.stdout,
        verbosity=verbosity,
    )
    result = runner.run(suite)
    elapsed = time.perf_counter() - t0
    return evaluate_result(result, load_errors=load_errors, elapsed_sec=elapsed)


def _selected_tiers(tier_arg: str) -> list[str]:
    if tier_arg == "all":
        return ["1", "2", "3", "4"]
    if tier_arg not in TIER_PATTERNS:
        raise ValueError(f"Unsupported tier: {tier_arg!r}")
    return [tier_arg]


def run_list_mode(
    tier_arg: str,
    *,
    verbose: bool = False,
    start_dir: Path = SUITE_DIR,
) -> tuple[int, dict[str, Any]]:
    """Discover and list tests across selected tier(s), validating coverage thresholds."""
    tiers = _selected_tiers(tier_arg)
    per_tier: dict[str, list[str]] = {}
    all_load_errors: list[str] = []
    total_count = 0
    threshold_failures: list[str] = []

    for tk in tiers:
        _, tids, load_errs = discover_tier(tk, start_dir=start_dir)
        per_tier[tk] = tids
        all_load_errors.extend(load_errs)
        total_count += len(tids)
        min_req = MIN_TIER_COUNTS[tk]
        if len(tids) < min_req:
            threshold_failures.append(f"Tier {tk}: discovered {len(tids)} < required {min_req}")

    if tier_arg == "all" and total_count < MIN_TOTAL_COUNT:
        threshold_failures.append(f"Total: discovered {total_count} < required {MIN_TOTAL_COUNT}")

    for tk in tiers:
        tids = per_tier[tk]
        print(f"=== Tier {tk} ({len(tids)} tests, min >= {MIN_TIER_COUNTS[tk]}) ===")
        for tid in tids:
            print(f"  {tid}")

    print(
        f"\nSummary: total={total_count}, "
        + ", ".join(f"T{tk}={len(per_tier[tk])}" for tk in tiers)
        + f", load_errors={len(all_load_errors)}"
    )
    if all_load_errors:
        print("LOAD ERRORS:")
        for err in all_load_errors:
            print(f"  {err}")
    if threshold_failures:
        print("THRESHOLD FAILURES:")
        for msg in threshold_failures:
            print(f"  {msg}")

    ok = len(all_load_errors) == 0 and len(threshold_failures) == 0 and total_count > 0
    payload: dict[str, Any] = {
        "mode": "list",
        "tier": tier_arg,
        "passed": ok,
        "exit_code": 0 if ok else 1,
        "total_tests": total_count,
        "per_tier_counts": {f"tier_{tk}": len(per_tier[tk]) for tk in tiers},
        "per_tier_test_ids": {f"tier_{tk}": per_tier[tk] for tk in tiers} if verbose else {},
        "load_errors": all_load_errors,
        "threshold_failures": threshold_failures,
    }
    return (0 if ok else 1), payload


def run_exec_mode(
    tier_arg: str,
    *,
    verbosity: int = 1,
    start_dir: Path = SUITE_DIR,
    enforce_min_counts: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Execute tests for requested tier(s) in strict order with minimum-count enforcement."""
    tiers = _selected_tiers(tier_arg)
    combined = unittest.TestSuite()
    all_load_errors: list[str] = []
    per_tier_counts: dict[str, int] = {}
    threshold_failures: list[str] = []
    total_discovered = 0

    for tk in tiers:
        tier_suite, tids, load_errs = discover_tier(tk, start_dir=start_dir)
        per_tier_counts[f"tier_{tk}"] = len(tids)
        total_discovered += len(tids)
        all_load_errors.extend(load_errs)
        if enforce_min_counts and len(tids) < MIN_TIER_COUNTS[tk]:
            threshold_failures.append(f"Tier {tk}: discovered {len(tids)} < required {MIN_TIER_COUNTS[tk]}")
        combined.addTests(tier_suite)

    if enforce_min_counts and tier_arg == "all" and total_discovered < MIN_TOTAL_COUNT:
        threshold_failures.append(f"Total: discovered {total_discovered} < required {MIN_TOTAL_COUNT}")

    rc, summary = run_suite_strict(combined, verbosity=verbosity, load_errors=all_load_errors)
    summary["mode"] = "run"
    summary["tier"] = tier_arg
    summary["per_tier_discovered"] = per_tier_counts
    summary["threshold_failures"] = threshold_failures
    if threshold_failures:
        rc = 1
        summary["passed"] = False
        summary["exit_code"] = 1
    print(
        f"\n[E2E Platform v4] tier={tier_arg} run={summary['tests_run']} "
        f"fail={summary['failures']} err={summary['errors']} skip={summary['skipped']} "
        f"exp_fail={summary['expected_failures']} unexp_succ={summary['unexpected_successes']} "
        f"load_err={len(all_load_errors)} thresh_err={len(threshold_failures)} "
        f"elapsed={summary['elapsed_sec']}s -> "
        f"{'PASS' if rc == 0 else 'FAIL'}"
    )
    return rc, summary


def _write_json_out(path_str: str, payload: dict[str, Any]) -> None:
    out_path = Path(path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(fixtures.strict_json_dumps(payload) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Platform v4 Independent 4-Tier Opaque-Box E2E Test Runner"
    )
    parser.add_argument(
        "--tier",
        choices=("1", "2", "3", "4", "all"),
        default="all",
        help="Tier to discover or execute (default: all)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="List discovered tests and verify tier counts without executing targets",
    )
    mode_group.add_argument(
        "--self-check",
        action="store_true",
        dest="self_check",
        help="Run independent E2E harness integrity self-check",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write strict NaN-free JSON summary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    verbosity = 2 if args.verbose else 1

    if args.self_check:
        if __package__:
            from . import harness_selfcheck
        else:
            import harness_selfcheck

        rc, payload = harness_selfcheck.run_self_check(verbose=args.verbose)
    elif args.list_only:
        rc, payload = run_list_mode(args.tier, verbose=args.verbose)
    else:
        rc, payload = run_exec_mode(args.tier, verbosity=verbosity)

    if args.json_out:
        _write_json_out(args.json_out, payload)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
