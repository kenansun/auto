# AlphaLoop v2 — Strategy Factory, DQ Gate & Multi-Model Agent R&D Platform

Zero-dependency (pure Python 3.12 stdlib) quantitative strategy R&D factory engineered for **hard data-quality certification (`DQ-01..DQ-08`), pluggable heterogeneous alpha strategies, fast micro-run validation with dual-layer early-kill, and closed-loop multi-model Agent iteration**.

## Core Capabilities (v2)

1. **Real-Exchange Market Data Lake & Live BBO Collector (`alphaloop/data/`)**:
   - Paginated 90-day historical funding & 1h OHLCV collectors plus live top-of-book BBO orderbook collector (`bbo_collector.py`) across **Hyperliquid**, **Lighter**, **OKX**, and **Gate.io**.
   - Thread-safe `LockedConnection` SQLite WAL storage with `BEGIN IMMEDIATE` write isolation, crossed-book rejection, and SHA-256 dataset fingerprinting (`PROVENANCE_REAL`).
2. **Hard Data Quality Certification Engine (`alphaloop/data/quality.py`, `DQ-01..DQ-08`)**:
   - Certifies every SQLite lake slice, live BBO feed, and server CSV into `DATA_QUALIFIED_FULL`, `DATA_QUALIFIED_FUNDING_ONLY`, or `DATA_REJECTED`.
   - Enforces cadence-scaled gaps, clock/monotonicity hygiene, coarser-grid alignment, cross-venue symbol mapping price ratios (catches `1000PEPE`/`USDCUSDT` errors), distribution funding unit bounds (catches 100x bugs without penalizing real spikes), dual-venue basis completeness, and OHLCV-only anti-frozen checks.
3. **Strategy Factory v2 Pluggable Registry (`alphaloop/plugins/`)**:
   - Strategy-agnostic `BaseStrategyPlugin` contract & `PluginRegistry` shipping 3 heterogeneous strategy families:
     - `funding_carry`: Cross-venue perp-perp / spot-perp carry with causal `c.open` zero-lookahead pricing, missing-leg directional liquidation proxy & 24h post-stop cooldown, and explicit `MAKER_MAKER` / `MAKER_THEN_TAKER_HEDGE` / `TAKER_TAKER` execution modes.
     - `basis_reversion`: Cross-venue perp-perp mark basis mean-reversion with causal rolling z-score, default `execution_lag_bars=1` anti-phantom lag on 1h candles, last-seen price carry-forward, and 4-leg cost & funding drag accounting.
     - `structural_arb`: Event-driven DEX-CEX / stablecoin-dip dislocation evaluator with strict ex-ante signal vs ex-post realized PnL separation, immutable physical friction parameters (gas, revert rate, confirm decay, rebalance drag), and 4-leg execution-mode fees.
4. **Automated DQ + 6-Gate Verifier & Replayable Fast Micro-Runs (`alphaloop/harness/`)**:
   - `SixGateVerifier` integrates `DQ-01..DQ-08` in `G1`, zero-liquidation + `+2bp/leg` cost stress + `MAKER_THEN_TAKER_HEDGE` survivability in `G2`, calendar-time continuous 70/30 IS/OOS walk-forward in `G3`, plugin-delegated 3×3 plateau in `G4`, anti-concentration in `G5`, and atomic horizon-scaled DSR-deduplicated trial ledger (keyed by immutable `(plugin_id, venue_pair)` + economic parameter SHA-256) in `G6`.
   - `MicroRunSession` replays live BBO + funding events chronologically through `plugin.on_event()` with simulated mark-to-market paper PnL/DD and a **Dual-Layer Early-Kill Guard**: immediate execution-quality / catastrophic drawdown (`>5%`) kill on `DATA_REJECTED` / live spread >2× assumed / basis stop breach, while gating economic PnL feedback and `COMPLETED_PASS` behind `max(target_hours, plugin.min_economic_kill_horizon_hours())`.
5. **Closed-Loop Multi-Model Agent Orchestrator (`alphaloop/agent_loop/`)**:
   - Consequence-severity router (`Gemini Flash`, `DeepSeek Reasoner`, `Claude Sonnet`, cross-family `Claude Opus` / `GPT-6.1 Sol`) with thread-local per-round USD budget isolation (`$0.50`/round).
   - Enforces DQ promotion cap (`DATA_QUALIFIED_FUNDING_ONLY` capped at `MICRO_RUN_PROBE_REQUIRED` until a fresh `COMPLETED_PASS` `MicroRunRecord` + `DATA_QUALIFIED_FULL` BBO certificate covers 100% of `symbols`); `CANARY_APPROVED` / `LIVE` registration and rollback strictly require `HUMAN_APPROVED:<id>`.
6. **Hardened Read-Only SSH Fleet Auditor & Web Console v2 (`alphaloop/fleet/`, `alphaloop/web/`)**:
   - Allowlist-only read-only SSH probes (non-root user, validated IPv4, `--` delimiter, `/opt/` prefix allowlist blocking secret files) configured via local gitignored `config/fleet_roles.json` and `ALPHALOOP_SSH_KEY`.
   - Loopback-only (`127.0.0.1:8796`) Web Console v2 exposing `/api/plugins`, `/api/dq/reports`, `/api/micro-runs`, `/api/fleet/audit`, `/api/strategies`, `/api/cycles`, `/api/trials`, and `/api/models` with delegated DOM event listeners (zero inline `onclick`), CSRF `Content-Type`/`Origin` guards, and `L-034`/`STD-34` 0-violation compliance.

## Quickstart

```bash
# Run full unit & Red-Team adversarial test suite (29 tests)
python3 -m unittest discover -s tests -v

# List Strategy Factory v2 plugins
python3 -m alphaloop.cli plugins

# Run DQ-01..DQ-08 audit on SQLite lake or live BBO ticks
python3 -m alphaloop.cli dq-audit --venue-a hyperliquid --venue-b okx
python3 -m alphaloop.cli dq-audit --venue-a lighter --venue-b gate --bbo

# Capture live 4-exchange BBO orderbook ticks
python3 scripts/run_bbo_collector.py --duration-minutes 10 --interval-sec 20

# Run full 90-day empirical research + heterogeneous plugins + live BBO micro-runs
python3 scripts/run_full_research.py

# Launch local Strategy R&D Web Console v2
python3 -m alphaloop.cli serve --host 127.0.0.1 --port 8796
```
