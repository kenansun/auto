# AlphaLoop v2.2 — Strategy Factory, DQ Gate, Portfolio Combiner & Always-On Shadow Platform

Zero-dependency (pure Python 3.12 stdlib) quantitative strategy R&D factory engineered for **hard data-quality certification (`DQ-01..DQ-08`), pluggable heterogeneous alpha strategies (Funding Carry + Trend CTA + Basis Reversion + Structural Arb), multi-sleeve portfolio combination, always-on `SHADOW_DRY_RUN` daemon with Shadow-vs-Backtest parity auditing, closed-loop multi-model Agent evolution, and strict Decimal execution-plane contract adaptation**.

## Core Capabilities (v2.2)

1. **Real-Exchange Market Data Lake & Live BBO Collector (`alphaloop/data/`)**:
   - Paginated **180d–365d** historical funding & `1h` OHLCV collectors plus live synchronized top-of-book BBO orderbook collector (`bbo_collector.py`) across **Hyperliquid**, **Lighter**, **OKX**, and **Gate.io**.
   - Thread-safe `LockedConnection` SQLite WAL storage with `BEGIN IMMEDIATE` write isolation, crossed-book rejection, and SHA-256 dataset fingerprinting (`PROVENANCE_REAL`).
2. **Hard Data Quality Certification Engine (`alphaloop/data/quality.py`, `DQ-01..DQ-08`)**:
   - Certifies cross-venue funding pairs (`audit_funding_pair`), single-venue perpetual datasets (`audit_single_venue_perp` with `>=98%` candle↔funding overlap enforcement so directional strategies cannot skip funding drag), live BBO feeds (`audit_bbo_ticks`), and server CSVs (`audit_csv_sample`) into `DATA_QUALIFIED_FULL`, `DATA_QUALIFIED_FUNDING_ONLY`, or `DATA_REJECTED`.
3. **Strategy Factory v2.2 Pluggable Registry (`alphaloop/plugins/`)**:
   - Strategy-agnostic `BaseStrategyPlugin` contract & `PluginRegistry` shipping 4 heterogeneous strategy families (`funding_carry`, `trend_cta`, `basis_reversion`, `structural_arb`).
4. **Automated DQ + 6-Gate Verifier, Portfolio Combiner & Fast Micro-Runs (`alphaloop/harness/`)**:
   - `SixGateVerifier` integrates `DQ-01..DQ-08` in `G1`, zero-liquidation + `+2bp/leg` cost stress + `MAKER_THEN_TAKER_HEDGE` survivability in `G2`, calendar-time continuous 70/30 IS/OOS walk-forward in `G3`, plugin-delegated 3×3 plateau in `G4`, anti-concentration in `G5`, and atomic horizon-scaled DSR-deduplicated trial ledger in `G6`.
   - `PortfolioCombiner` (`portfolio.py`) combines heterogeneous sleeves on exact overlapping UTC daily grids (fail-closed on insufficient overlap, interior date gaps, or bankrupt/non-finite equity), enforces capital conservation (`sum(w) == 1.0`, `w_i >= 0`), deducts explicit rebalance turnover costs, and projects capacity-aware scaling anchored to measured live BBO top-of-book depth.
5. **Shadow-Only Execution Bridge, Always-On Daemon & Parity Auditor (`alphaloop/execution/`, `alphaloop/daemon/`)**:
   - Hard-coded `LIVE_EXECUTION_ENABLED = False` (`SHADOW_DRY_RUN` only; zero private keys, zero live order placement).
   - `ExecutionBridge` + canonical-JSON SHA-256 hash-chained `ExecutionAuditChain` + persistent `ExecutionKillSwitch` (`HUMAN_RESET:<id>`) + thread-safe `HotConfigWatcher`.
   - `ShadowPortfolioDaemon` + `ShadowParityAuditor` (`daemon/shadow_runner.py`) drive active `SHADOW_LIVE` strategies continuously on live BBO feeds, compare live Shadow equity against same-window deterministic backtests, and enforce 4 fail-closed kill/halt guards (chain tamper, stale feed, sustained wide spread `>1.5x`, and parity drift `>150bp`).
   - `WatchdogHook` & `AlertNotifier` (`daemon/watchdog_hook.py`): bounded async dead-man's-switch heartbeat gated strictly on fresh business timestamps + 30-minute alert deduplication.
   - `arb_executor_adapter.py`: exact `Decimal` wire-envelope translator (`to_arb_plan_risk_envelope`, `to_arb_audit_record_envelope`) with secret-key masking for downstream execution planes.
6. **Closed-Loop Multi-Model Agent Orchestrator & Web Console v2.2 (`alphaloop/agent_loop/`, `alphaloop/web/`)**:
   - Consequence-severity router with per-round + SQLite persistent daily USD budget caps (`$3.00/day` default), strict LLM parameter whitelist/bounds sanitization, and mandatory cross-family Red-Team independence.
   - Loopback-only (`127.0.0.1:8796`) Web Console v2.2 exposing `/api/shadow/parity`, `/api/portfolio`, `/api/execution/plans`, `/api/plugins`, `/api/dq/reports`, `/api/micro-runs`, `/api/fleet/audit`, `/api/strategies`, `/api/cycles`, `/api/trials`, and `/api/models` with zero inline `onclick`, strict CSRF guards, and `STD-34` 0-violation compliance.

## Quickstart

```bash
# Run full unit & Red-Team adversarial test suite
python3 -m unittest discover -s tests -v

# Bootstrap verified Balanced 50/50 SHADOW_LIVE configs and run 1 shadow cycle
python3 scripts/run_shadow_daemon.py --data-dir data --bootstrap-balanced --once

# Inspect Shadow Execution Bridge audit chain & Shadow Parity report
python3 -m alphaloop.cli exec-status
python3 -m alphaloop.cli shadow-parity

# Run unattended nightly incremental backfill & 6-Gate factory worker
python3 scripts/run_nightly_factory.py --data-dir data --days 3

# Audit Pillar 3 USDC_DIP event sample progress (enforces DQ-01 n>=30)
python3 scripts/ingest_usdc_dip_audit.py --csv <path/to/burst_paper_trades.csv>

# Launch local Strategy R&D Web Console v2.2 (loopback only)
python3 -m alphaloop.cli serve --host 127.0.0.1 --port 8796
```
