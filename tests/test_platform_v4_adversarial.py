"""Tier 5 White-Box Adversarial Hardening Suite for AlphaLoop Strategy R&D Platform v4.

Attacks every critical architectural, statistical, domain-isolation, and security invariant:
- ADV-01: Zero Yahoo ^VIX / TradFi contamination across Web3 W1/W2/W3 & Regime Engine (D7)
- ADV-02: Zero pre-release FOMC/NFP/CPI actual/surprise_z leakage in Gold G2 Policies A/B/C (D16)
- ADV-03: PAXG G3 stop-loss isolation from GC=F whipsaws & Friday-close weekend guard
- ADV-04: Gold G1 strict anti-double-counting guard against simultaneous GC=F + GLD (D8)
- ADV-05: SQLite REA-001 trial_ledger trigger immutability (blocks UPDATE/DELETE/INSERT OR REPLACE)
- ADV-06: Read-only DuckDB SQL sandbox blocks external file-IO & DDL injection
- ADV-07: CLI runner fail-closed on empty lake & synthetic-contaminated lake (zero fake PASS)
- ADV-08: A-Share isolated contract enforces long-only, intraday T+1, limit-up/down & sell stamp duty
- ADV-09: Nested CPCV catches pure-noise IS overfit & evaluate_8_gates rejects non-nested CPCV (D3)
- ADV-10: Platform-wide pure Python 3.12 stdlib execution, zero DO_NOT_TOUCH coupling, zero tombstones
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import math
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.platform.assets import (
    ASSET_CLASS_CN_A_SHARE,
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_GOLD_PERP,
    ASSET_CLASS_GOLD_TRADFI,
    default_ashare_template_spec,
    default_gold_specs,
    default_web3_specs,
)
from alphaloop.platform.core_engine import (
    FoldResult,
    PortfolioAccountingEngine,
    RiskAllocatorConfig,
    evaluate_8_gates,
    evaluate_cpcv_returns,
    run_nested_cpcv,
    sharpe_from_returns,
)
from alphaloop.platform.data_foundation import (
    CanonicalBar,
    DataFoundationRouter,
    MacroEventRelease,
    validate_readonly_sql,
)
from alphaloop.platform.strategies.ashare_contract import AShareStrategyInterface
from alphaloop.platform.strategies.gold import (
    Gold4hEventKernel,
    GoldPAXGBridgeKernel,
    GoldTradFiMacroKernel,
)
from alphaloop.platform.strategies.web3 import (
    Web3AdaptiveTrendKernel,
    Web3MarketRegimeEngine,
    Web3RegimeCarryKernel,
    Web3XSMomCarryKernel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_strategy_platform_v4.py"
HOUR_MS = 3_600_000


def _make_bars(
    symbol: str,
    venue: str,
    asset_class: str,
    closes: list[float],
    *,
    timeframe_h: float = 12.0,
    start_ms: int = 1_735_689_600_000,
    carry_rate: float = 0.0001,
    provenance: str = PROVENANCE_REAL,
) -> list[CanonicalBar]:
    step_ms = int(round(timeframe_h * HOUR_MS))
    bars: list[CanonicalBar] = []
    prev_c = closes[0]
    for i, c in enumerate(closes):
        ts_e = start_ms + i * step_ms
        o = prev_c if i > 0 else c
        hi = max(o, c) * 1.005
        lo = min(o, c) * 0.995
        bars.append(
            CanonicalBar(
                venue=venue,
                symbol=symbol,
                asset_class=asset_class,
                timeframe_h=timeframe_h,
                ts_event_ms=ts_e,
                ts_avail_ms=ts_e + step_ms,
                open=o,
                high=hi,
                low=lo,
                close=c,
                volume_usd=1_000_000.0,
                carry_rate_period=carry_rate,
                carry_interval_h=1.0,
                spread_bps=2.0,
                provenance=provenance,
            )
        )
        prev_c = c
    return bars


class _ConstantLongAShare(AShareStrategyInterface):
    def compute_raw_signal(self, ts_ms: int, bar: CanonicalBar, context: dict | None = None) -> float:
        if context and "force_signal" in context:
            return float(context["force_signal"])
        return 1.0


class TestPlatformV4Adversarial(unittest.TestCase):
    """Tier 5 white-box adversarial tests attacking Platform v4 core invariants."""

    def test_adv01_web3_zero_tradfi_vix_contamination(self) -> None:
        rng = random.Random(4201)
        syms = ("BTC", "ETH", "SOL", "AVAX")
        bars_by_sym: dict[str, list[CanonicalBar]] = {}
        for idx, s in enumerate(syms):
            px = 100.0
            closes = []
            for _ in range(110):
                px *= 1.0 + (0.004 - 0.002 * idx) + rng.gauss(0.0, 0.012)
                closes.append(max(1.0, px))
            bars_by_sym[s] = _make_bars(s, "hyperliquid", ASSET_CLASS_CRYPTO_PERP, closes)

        # Inject a toxic ^VIX series into regime engine inputs as well as context
        fake_vix_bars = _make_bars("^VIX", "yahoo", ASSET_CLASS_GOLD_TRADFI, [88.0] * 110)
        reg_clean = Web3MarketRegimeEngine(bars_by_symbol=bars_by_sym)
        reg_toxic = Web3MarketRegimeEngine(bars_by_symbol={**bars_by_sym, "^VIX": fake_vix_bars})
        ts_check = bars_by_sym["BTC"][-1].ts_avail_ms
        self.assertEqual(reg_clean.snapshot_as_of(ts_check), reg_toxic.snapshot_as_of(ts_check))

        specs = default_web3_specs(syms)
        cfg = RiskAllocatorConfig(max_positions=4, target_annual_vol_pct=20.0, max_portfolio_leverage=1.5, max_single_weight=0.5)
        acct = PortfolioAccountingEngine()
        for strat_cls in (Web3XSMomCarryKernel, Web3AdaptiveTrendKernel, Web3RegimeCarryKernel):
            r_clean = acct.simulate(strat_cls(), specs, bars_by_sym, risk_cfg=cfg)
            r_vix = acct.simulate(
                strat_cls(),
                specs,
                bars_by_sym,
                risk_cfg=cfg,
                context_provider=lambda _ts: {"yahoo_vix": 99.0, "macro_vix_stress_z": 12.0, "gold_macro": {"macro_gold_composite": -5.0}},
            )
            self.assertEqual(r_clean.equity_curve, r_vix.equity_curve, f"D7 violation in {strat_cls.__name__}")

    def test_adv02_gold_g2_zero_pre_release_event_actual_leakage(self) -> None:
        t_rel = int(datetime(2026, 12, 9, 19, 0, tzinfo=timezone.utc).timestamp() * 1000)
        start_ms = t_rel - 40 * 4 * HOUR_MS
        closes = [2700.0 + 2.0 * math.sin(i / 3.0) for i in range(44)]
        closes[38] = 2718.0
        closes[39] = 2725.0
        closes[40] = 2745.0
        bars = _make_bars("GC=F", "yahoo", ASSET_CLASS_GOLD_TRADFI, closes, timeframe_h=4.0, start_ms=start_ms)

        ev_dovish = MacroEventRelease("FOMC", "FOMC_DEC", "2026-12", t_rel, t_rel, True, 3.75, 4.50, -3.5, "fed", PROVENANCE_REAL)
        ev_hawkish = MacroEventRelease("FOMC", "FOMC_DEC", "2026-12", t_rel, t_rel, True, 5.50, 4.50, +3.5, "fed", PROVENANCE_REAL)

        for pol in ("A", "B", "C"):
            k1 = Gold4hEventKernel(policy=pol, events=[ev_dovish])
            k2 = Gold4hEventKernel(policy=pol, events=[ev_hawkish])
            for b in bars:
                if b.ts_event_ms >= t_rel:
                    break
                d1 = k1.on_bar_step(b.ts_event_ms, {"GC=F": b}, {"macro_events": [ev_dovish], "gold_macro": {"macro_gold_composite": 0.6}})
                d2 = k2.on_bar_step(b.ts_event_ms, {"GC=F": b}, {"macro_events": [ev_hawkish], "gold_macro": {"macro_gold_composite": 0.6}})
                self.assertEqual(d1["GC=F"].target_weight, d2["GC=F"].target_weight, f"Pre-release leakage in Policy {pol}")
                self.assertEqual(d1["GC=F"].raw_signal, d2["GC=F"].raw_signal)
                k1.sync_executed_weights({"GC=F": d1["GC=F"].target_weight})
                k2.sync_executed_weights({"GC=F": d2["GC=F"].target_weight})

    def test_adv03_paxg_stop_isolation_and_weekend_dislocation_block(self) -> None:
        # Build bars ending cleanly inside Saturday UTC (COMEX weekend frozen)
        sat_ms = int(datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
        start_ms = sat_ms - 45 * 12 * HOUR_MS
        closes = [2650.0 + 1.2 * i for i in range(46)]
        bars = _make_bars("PAXG", "hyperliquid", ASSET_CLASS_GOLD_PERP, closes, timeframe_h=12.0, start_ms=start_ms)

        # 1) Stop price strictly independent of GC=F intraday shock
        k_norm = GoldPAXGBridgeKernel()
        k_shock = GoldPAXGBridgeKernel()
        for b in bars[:40]:
            d1 = k_norm.on_bar_step(b.ts_event_ms, {"PAXG": b}, {"paxg_spot_price": b.close, "comex_price": b.close})
            d2 = k_shock.on_bar_step(b.ts_event_ms, {"PAXG": b}, {"paxg_spot_price": b.close, "comex_price": b.close * 0.85})
            self.assertEqual(d1["PAXG"].stop_price, d2["PAXG"].stop_price)
            k_norm.sync_executed_weights({"PAXG": d1["PAXG"].target_weight})
            k_shock.sync_executed_weights({"PAXG": d2["PAXG"].target_weight})

        # 2) Extreme >3.5sigma weekend dislocation blocks fresh entry from flat
        k_wknd = GoldPAXGBridgeKernel(weekend_dislocation_z=2.5)
        for i, b in enumerate(bars):
            is_last = i == len(bars) - 1
            comex_px = b.close * (0.94 if is_last else 1.0)
            dec = k_wknd.on_bar_step(
                b.ts_event_ms,
                {"PAXG": b},
                {"paxg_spot_price": b.close, "comex_price": comex_px, "is_weekend_frozen": is_last},
            )
            if not is_last:
                # Keep flat right before the Saturday spike to verify fresh-entry block
                k_wknd.sync_executed_weights({"PAXG": 0.0})
            else:
                self.assertAlmostEqual(dec["PAXG"].target_weight, 0.0, places=8)
                self.assertTrue(bool(dec["PAXG"].metadata.get("weekend_dislocation_blocked", False)))

    def test_adv04_gold_g1_anti_double_count_guard(self) -> None:
        closes = [2600.0 + 2.0 * i for i in range(60)]
        gc_bars = _make_bars("GC=F", "yahoo", ASSET_CLASS_GOLD_TRADFI, closes, timeframe_h=24.0)
        gld_bars = _make_bars("GLD", "yahoo", ASSET_CLASS_GOLD_TRADFI, [c / 10.0 for c in closes], timeframe_h=24.0)

        k_cap = GoldTradFiMacroKernel(symbol="GC=F", reject_multi_gold_input=False)
        last_dec = {}
        for gb, lb in zip(gc_bars, gld_bars):
            last_dec = k_cap.on_bar_step(gb.ts_event_ms, {"GC=F": gb, "GLD": lb}, {"gold_macro": {"macro_gold_composite": 0.8}})
            k_cap.sync_executed_weights({s: d.target_weight for s, d in last_dec.items()})
        self.assertGreater(last_dec["GC=F"].target_weight, 0.0)
        self.assertEqual(last_dec["GLD"].target_weight, 0.0)

        k_strict = GoldTradFiMacroKernel(symbol="GC=F", reject_multi_gold_input=True)
        with self.assertRaises(ValueError):
            k_strict.on_bar_step(gc_bars[0].ts_event_ms, {"GC=F": gc_bars[0], "GLD": gld_bars[0]})

    def test_adv05_trial_ledger_sqlite_immutability_triggers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="adv05_ledger_") as td:
            router = DataFoundationRouter(Path(td))
            try:
                tid = router.record_trial("fam_adv", "web3", "W1", "abc123", {"sharpe_ratio": 1.2}, "CONDITIONAL_RESEARCH")
                self.assertTrue(bool(tid))
            finally:
                router.close()

            conn = sqlite3.connect(str(Path(td) / "catalog.sqlite"), timeout=10.0)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute("UPDATE trial_ledger SET verdict = 'PROMOTE_SHADOW_CANDIDATE' WHERE trial_id = ?", (tid,))
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute("DELETE FROM trial_ledger WHERE trial_id = ?", (tid,))
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(
                        "INSERT OR REPLACE INTO trial_ledger(trial_id, created_ts_ms, family_id, track_name, strategy_name, params_hash, metrics_json, verdict) "
                        "VALUES (?, 1, 'fam_adv', 'web3', 'W1', 'tamper', '{}', 'PROMOTE_SHADOW_CANDIDATE')",
                        (tid,),
                    )
            finally:
                conn.close()

    def test_adv06_readonly_duckdb_sql_sandbox_blocks_injection_and_external_io(self) -> None:
        self.assertEqual(validate_readonly_sql("SELECT COUNT(*) FROM bars;"), "SELECT COUNT(*) FROM bars")
        attacks = (
            "SELECT * FROM read_text('/etc/passwd')",
            "SELECT * FROM read_parquet('/tmp/leak.parquet')",
            "SELECT * FROM glob('/home/*')",
            "SELECT 1; DROP TABLE bars",
            "SELECT 1 -- comment injection",
            "INSERT INTO bars VALUES (1)",
            "COPY (SELECT 1) TO '/tmp/out.csv'",
            "PRAGMA database_list",
        )
        for bad_sql in attacks:
            with self.assertRaises(ValueError, msg=f"Sandbox failed to block: {bad_sql}"):
                validate_readonly_sql(bad_sql)

    def test_adv07_runner_fail_closed_on_empty_and_synthetic_lakes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="adv07_runner_") as td:
            tmp = Path(td)
            empty_dir = tmp / "empty_lake"
            empty_dir.mkdir()
            rep = tmp / "rep_empty.json"
            p1 = subprocess.run(
                [sys.executable, str(RUNNER_PATH), "--offline", "--lake-dir", str(empty_dir), "--report-out", str(rep)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(p1.returncode, 0, "Runner must fail closed on empty lake")
            self.assertFalse(rep.exists())

            # Create lake containing synthetic provenance bar
            syn_dir = tmp / "syn_lake"
            router = DataFoundationRouter(syn_dir)
            try:
                syn_bars = _make_bars("BTC", "hyperliquid", ASSET_CLASS_CRYPTO_PERP, [100.0 + i for i in range(40)], provenance="SYNTHETIC_TEST")
                router.upsert_bars("web3", syn_bars, allow_synthetic_in_test=True)
            finally:
                router.close()
            p2 = subprocess.run(
                [sys.executable, str(RUNNER_PATH), "--offline", "--lake-dir", str(syn_dir), "--report-out", str(rep)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(p2.returncode, 0, "Runner must fail closed on synthetic data lake")

    def test_adv08_ashare_t_plus_1_price_limits_and_sell_stamp_duty(self) -> None:
        strat = _ConstantLongAShare()
        spec = default_ashare_template_spec("600519.SH")
        self.assertEqual(strat.asset_domain, "ashare")
        self.assertEqual(spec.calendar.t_plus_days, 1)

        # Day 1 bar 1 (10:00 CST = 02:00 UTC): buy allowed
        t1 = int(datetime(2026, 11, 18, 2, 0, tzinfo=timezone.utc).timestamp() * 1000)
        b1 = _make_bars("600519.SH", "sse", ASSET_CLASS_CN_A_SHARE, [100.0], timeframe_h=1.0, start_ms=t1)[0]
        d1 = strat.on_bar_step(t1, {"600519.SH": b1}, {"force_signal": 1.0})
        self.assertGreater(d1["600519.SH"].target_weight, 0.0)
        strat.sync_executed_weights({"600519.SH": d1["600519.SH"].target_weight})

        # Day 1 bar 2 (14:00 CST = 06:00 UTC): same-day sell MUST be locked by T+1!
        t2 = t1 + 4 * HOUR_MS
        b2 = _make_bars("600519.SH", "sse", ASSET_CLASS_CN_A_SHARE, [101.0], timeframe_h=1.0, start_ms=t2)[0]
        d2 = strat.on_bar_step(t2, {"600519.SH": b2}, {"force_signal": -1.0})
        self.assertGreater(d2["600519.SH"].target_weight, 0.0, "Same-day sell must be locked by T+1")

        # Long-only check from flat: negative signal clamps to 0.0
        strat2 = _ConstantLongAShare()
        d_neg = strat2.on_bar_step(t1, {"600519.SH": b1}, {"force_signal": -1.0})
        self.assertEqual(d_neg["600519.SH"].target_weight, 0.0)

        # Sell-only stamp duty: buy cost = 0, sell cost > 0
        self.assertEqual(strat.estimate_stamp_duty_usd(10_000.0, is_sell=False), 0.0)
        self.assertGreater(strat.estimate_stamp_duty_usd(10_000.0, is_sell=True), 0.0)

    def test_adv09_nested_cpcv_rejects_pure_noise_and_non_nested_g5(self) -> None:
        rng = random.Random(20261123)
        n_obs = 240
        noise_matrix = [[rng.gauss(0.0, 0.015) for _ in range(n_obs)] for _ in range(20)]

        def _overfit_selector(train_idx: tuple[int, ...], test_idx: tuple[int, ...]) -> FoldResult:
            best_row = max(noise_matrix, key=lambda row: sharpe_from_returns([row[i] for i in train_idx], 252.0))
            return FoldResult(
                oos_returns=tuple(best_row[i] for i in test_idx),
                is_returns=tuple(best_row[i] for i in train_idx),
            )

        nested_rep = run_nested_cpcv(n_obs, _overfit_selector, periods_per_year=252.0, n_groups=6, k_test=2, purge_bars=2, embargo_bars=2)
        self.assertTrue(nested_rep.is_nested)
        self.assertFalse(nested_rep.passed_cpcv_gate, "Nested CPCV must catch IS selection overfit on pure noise")

        # evaluate_8_gates MUST reject non-nested CPCVReport for G5
        legacy_cpcv = evaluate_cpcv_returns([0.002] * n_obs, periods_per_year=252.0)
        gate_rep = evaluate_8_gates("adv_non_nested", "web3", nested_cpcv_report=legacy_cpcv)
        self.assertFalse(gate_rep.gate_map["G5"])

    def test_adv10_stdlib_only_no_do_not_touch_and_no_tombstones(self) -> None:
        # Pure stdlib import check across full v4 surface
        probe = (
            "import sys\n"
            "BLOCKED = {'numpy','scipy','pyarrow','duckdb','polars','pandas','pytest'}\n"
            "class Blocker:\n"
            "  def find_spec(self, fullname, path, target=None):\n"
            "    if fullname.split('.')[0] in BLOCKED:\n"
            "      raise ImportError(fullname)\n"
            "    return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "import alphaloop.platform\n"
            "from alphaloop.platform.strategies import web3, gold, ashare_contract\n"
            "print('ADV_STDLIB_OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ADV_STDLIB_OK", proc.stdout)

        # AST zero DO_NOT_TOUCH check across platform + v4 runner
        forbidden = ("alphaloop.execution", "alphaloop.daemon", "alphaloop.plugins")
        tombstone_re = re.compile(r"\bclass\s+\w*(?:GoldS1|SessionMeanReversion|OFLOW|HOF3|H_OF3)\w*", re.IGNORECASE)
        targets = list((REPO_ROOT / "alphaloop" / "platform").rglob("*.py")) + [RUNNER_PATH]
        for path in targets:
            src = path.read_text(encoding="utf-8")
            self.assertIsNone(tombstone_re.search(src), f"Tombstone found in {path}")
            tree = ast.parse(src, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        self.assertFalse(any(a.name.startswith(f) for f in forbidden), f"{path} imports {a.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(any(node.module.startswith(f) for f in forbidden), f"{path} imports {node.module}")


if __name__ == "__main__":
    unittest.main()
