# AlphaLoop v1 — Strategy R&D & Multi-Model Agent Feedback Platform

Zero-dependency (pure Python 3.12 stdlib) quantitative strategy R&D platform designed to replace manual multi-session markdown bureaucracy with a **closed-loop, machine-verified, real-time strategy iteration engine**.

## Core Capabilities

1. **Real-Exchange Market Data Lake (`alphaloop/data/`)**:
   - Paginated 90-day historical & live collectors for **Hyperliquid**, **Lighter**, **OKX**, and **Gate.io** (realized funding settlements + 1h OHLCV candles).
   - Immutable SQLite WAL storage with SHA-256 dataset fingerprinting (`PROVENANCE_REAL`) — zero synthetic fallback.
2. **Causal Event-Driven Backtester (`alphaloop/harness/backtester.py`)**:
   - Strictly zero-lookahead (`available_at <= decision_at`; funding collected strictly `ts_ms > entry_ts_ms`).
   - Full 4-leg Maker/Taker fee, slippage, dual-venue real 1h mark-basis PnL, basis stop-loss, and amortized rebalance drag accounting.
3. **Automated 6-Gate Anti-Overfitting Verifier (`alphaloop/harness/gates.py`)**:
   - `G1` Provenance & sample floor · `G2` MAIN + STRESS cost expectancy & drawdown · `G3` 70/30 Walk-Forward OOS · `G4` 3×3 Parameter Plateau stability · `G5` Drop top-5 trades & best coin · `G6` Append-only SQLite Trial Ledger with multiple-testing Sharpe hurdle.
4. **Atomic `rename(2)` Hot-Reload Config Bus (`alphaloop/core/config.py`)**:
   - Sub-microsecond lock-free config updates with one-click version rollback.
   - Autonomous loops may stage at most `SHADOW_LIVE`; `CANARY_APPROVED`/`LIVE` requires an explicit `HUMAN_APPROVED:<id>` token.
5. **Consequence-Severity Multi-Model Agent Loop (`alphaloop/agent_loop/`)**:
   - Routes roles across `Gemini Flash` (telemetry), `DeepSeek Reasoner` (quant hypothesis), `Claude Sonnet` (code builder), and cross-family `Claude Opus` / `GPT-6.1 Sol` (adversarial Red-Team critic) with a hard per-cycle USD budget cap.
   - Closed loop: `TelemetryMonitor → Proposer → Deterministic 6-Gate → Cross-Model Red-Team → Atomic SHADOW_LIVE Stage`.
6. **Hardened Strategy R&D Web Console (`alphaloop/web/server.py`)**:
   - Loopback-only (`127.0.0.1:8796`), `L-034` Content-Length hardened, XSS-escaped, `STD-34` 0-violation compliant.

## Quickstart

```bash
# Run unit & adversarial tests
python3 -m unittest discover -s tests -v

# Scan live cross-venue funding spreads across Hyperliquid, Lighter, Binance
python3 -m alphaloop.cli scan --min-oi-m 5

# Run full 90-day real exchange research + Agent evolution demo
python3 scripts/run_full_research.py

# Launch local Strategy R&D Web Console
python3 -m alphaloop.cli serve --host 127.0.0.1 --port 8796
```
