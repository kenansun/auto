# AlphaLoop v2.1 — Strategy Factory, DQ Gate, Portfolio Combiner & Shadow Execution Platform

Zero-dependency (pure Python 3.12 stdlib) quantitative strategy R&D factory engineered for **hard data-quality certification (`DQ-01..DQ-08`), pluggable heterogeneous alpha strategies (Funding Carry + Trend CTA + Basis Reversion + Structural Arb), multi-sleeve portfolio combination, fast micro-run validation, shadow execution bridging, and closed-loop multi-model Agent evolution**.

## Core Capabilities (v2.1)

1. **Real-Exchange Market Data Lake & Live BBO Collector (`alphaloop/data/`)**:
   - Paginated **180d–365d** historical funding & `1h` OHLCV collectors plus live synchronized top-of-book BBO orderbook collector (`bbo_collector.py`) across **Hyperliquid**, **Lighter**, **OKX**, and **Gate.io**.
   - Thread-safe `LockedConnection` SQLite WAL storage with `BEGIN IMMEDIATE` write isolation, crossed-book rejection, and SHA-256 dataset fingerprinting (`PROVENANCE_REAL`).
2. **Hard Data Quality Certification Engine (`alphaloop/data/quality.py`, `DQ-01..DQ-08`)**:
   - Certifies cross-venue funding pairs (`audit_funding_pair`), single-venue perpetual datasets (`audit_single_venue_perp` with `>=98%` candle↔funding overlap enforcement so directional strategies cannot skip funding drag), live BBO feeds (`audit_bbo_ticks`), and server CSVs (`audit_csv_sample`) into `DATA_QUALIFIED_FULL`, `DATA_QUALIFIED_FUNDING_ONLY`, or `DATA_REJECTED`.
3. **Strategy Factory v2.1 Pluggable Registry (`alphaloop/plugins/`)**:
   - Strategy-agnostic `BaseStrategyPlugin` contract & `PluginRegistry` shipping 4 heterogeneous strategy families:
     - `funding_carry`: Cross-venue perp-perp / spot-perp carry with causal `c.open` zero-lookahead pricing, optional causal 7-day volatility-tiered per-symbol leverage capping, storm-priced protective margin exit guard (`[protective_margin_exit_pct, 0.90)` at `TAKER_TAKER` + `>=5bp/leg` slip + 24h cooldown while `>=0.90` single-bar jumps still liquidate), and explicit `MAKER_MAKER` / `MAKER_THEN_TAKER_HEDGE` / `TAKER_TAKER` execution modes.
     - `trend_cta`: Multi-timeframe (`4h`/`12h`/`24h`, deterministically resampled from verified `1h` bars with incomplete buckets dropped) bidirectional Donchian Channel breakout + strictly prior-bar Chandelier ATR trailing stop + inverse-volatility sizing + `T+1` open fills + real perpetual funding carry/drag on mark notional (`qty * cur_1h_open`).
     - `basis_reversion`: Cross-venue perp-perp mark basis mean-reversion with causal rolling z-score, default `execution_lag_bars=1` anti-phantom lag, and 4-leg cost & funding drag accounting.
     - `structural_arb`: Event-driven dislocation evaluator with strict ex-ante signal vs ex-post realized PnL separation and immutable physical friction parameters.
4. **Automated DQ + 6-Gate Verifier, Portfolio Combiner & Fast Micro-Runs (`alphaloop/harness/`)**:
   - `SixGateVerifier` integrates `DQ-01..DQ-08` in `G1`, zero-liquidation + `+2bp/leg` cost stress + `MAKER_THEN_TAKER_HEDGE` survivability in `G2`, calendar-time continuous 70/30 IS/OOS walk-forward in `G3`, plugin-delegated 3×3 plateau in `G4`, anti-concentration in `G5`, and atomic horizon-scaled DSR-deduplicated trial ledger in `G6`.
   - `PortfolioCombiner` (`portfolio.py`) combines heterogeneous sleeves on exact overlapping UTC daily grids (fail-closed on insufficient overlap or interior date gaps), enforces capital conservation (`sum(w) == 1.0`, `w_i >= 0`), deducts explicit rebalance turnover costs, computes pairwise correlation & 70/30 OOS retention, and projects capacity-aware scaling anchored to measured live BBO top-of-book depth.
   - `MicroRunSession` replays live BBO + causal funding context chronologically through `plugin.on_event()` with mark-to-market paper PnL/DD and Dual-Layer Early-Kill Guard.
5. **Shadow-Only Execution Bridge (`alphaloop/execution/bridge.py`)**:
   - Hard-coded `LIVE_EXECUTION_ENABLED = False` (`SHADOW_DRY_RUN` only; zero private keys, zero live order placement).
   - Translates plugin `OrderIntent` + live BBO into deterministic `ExecutionPlan` objects with non-wall-clock `idempotency_key`, fail-closed pre-trade gates (status, stale/crossed/mismatched BBO, spread ceiling, notional cap), SHA-256 hash-chained `ExecutionAuditChain` (`verify_chain()`), persistent `ExecutionKillSwitch` (`HUMAN_RESET:<id>` clear gate), and `HotConfigWatcher` (auto-trips kill switch on corrupt active config JSON).
6. **Closed-Loop Multi-Model Agent Orchestrator & Web Console v2.1 (`alphaloop/agent_loop/`, `alphaloop/web/`)**:
   - Consequence-severity router with thread-local per-round USD budget isolation, DQ promotion caps, and human-gated live/canary controls (`HUMAN_APPROVED:<id>`).
   - Loopback-only (`127.0.0.1:8796`) Web Console v2.1 exposing `/api/portfolio`, `/api/execution/plans`, `/api/plugins`, `/api/dq/reports`, `/api/micro-runs`, `/api/fleet/audit`, `/api/strategies`, `/api/cycles`, `/api/trials`, and `/api/models` with zero inline `onclick`, strict CSRF `Content-Type`/`Origin` guards, and `STD-34` 0-violation compliance.

## Quickstart

```bash
# Run full unit & Red-Team adversarial test suite
python3 -m unittest discover -s tests -v

# List Strategy Factory v2.1 plugins
python3 -m alphaloop.cli plugins

# Backfill 180d-365d real multi-exchange history (PAXG + 24 liquid crypto perps)
python3 scripts/backfill_phase2_lake.py --funding-days 190 --cta-candle-days 365 --alt-candle-days 185

# Run DQ-01..DQ-08 audits (cross-venue funding pair, single-venue perp, live BBO)
python3 -m alphaloop.cli dq-audit --venue-a hyperliquid --venue-b gate
python3 -m alphaloop.cli dq-audit --venue-a hyperliquid --single-perp --ref-venue gate
python3 -m alphaloop.cli dq-audit --venue-a lighter --venue-b gate --bbo

# Inspect Shadow Execution Bridge audit chain & kill-switch state
python3 -m alphaloop.cli exec-status

# Capture live 4-exchange BBO orderbook ticks
python3 scripts/run_bbo_collector.py --duration-minutes 10 --interval-sec 20

# Launch local Strategy R&D Web Console v2.1
python3 -m alphaloop.cli serve --host 127.0.0.1 --port 8796
```
