"""Reference Strategy Plugin #3: Event-Driven Structural / DEX-CEX / Dislocation Arbitrage.

Red-Team hardened (#1, #5.3, #9, #12, #15):
- Strictly separates ex-ante `entry_signal_bps` (known at decision time) from ex-post
  `realized_edge_bps` (settled later) — NEVER filters on future outcomes or deletes losses!
- Applies T+1 execution lag on 1h OHLCV candles to block async candle-open bounce mirages.
- Locks ALL physical friction parameters (`gas_usd_per_tx`, `revert_rate_pct`, `confirm_decay_bps`,
  `rebalance_cost_bps`, `notional_per_trade_usd`) out of `STRUCTURAL_ARB_BOUNDS` so autonomous
  AI Agents CANNOT reward-hack by erasing transaction costs!
- Charges full 4-leg execution-mode-aware fees (`MAKER_MAKER` / `MAKER_THEN_TAKER_HEDGE` / `TAKER_TAKER`).
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from alphaloop.core.models import (
    BacktestMetrics,
    FeedbackEvent,
    MarketEvent,
    OrderIntent,
    StrategyConfig,
    TradeRecord,
)
from alphaloop.core.provenance import sha256_obj, utc_now_ms
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import HOUR_MS, TAKER_FEE_BPS, BacktestResult
from alphaloop.plugins.base import BaseStrategyPlugin

# Red-Team #5.3: ONLY strategy entry threshold is agent-tunable; physical friction is IMMUTABLE!
STRUCTURAL_ARB_BOUNDS: dict[str, tuple[float, float]] = {
    "min_edge_bps": (8.0, 120.0),
}


def _p(cfg: StrategyConfig, key: str, default: float) -> float:
    val = cfg.params.get(key, default)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return default
    f = float(val)
    return f if math.isfinite(f) else default


class StructuralArbPlugin(BaseStrategyPlugin):
    """Event-driven structural dislocation & stablecoin-dip arbitrage evaluator."""

    plugin_id = "structural_arb"
    display_name = "Structural DEX-CEX / Dislocation Arb"
    strategy_family = "structural_dislocation_arb"
    description = (
        "Evaluates atomic/short-window DEX-CEX dislocations & stablecoin-dip micro-events "
        "with immutable physical gas, tx-revert rate, confirmation decay, and inventory rebalance costs."
    )
    required_data_spec = {
        "tables": ["candles_1h"],
        "optional_csv_events": True,
        "requires_dual_candles_for_live": True,
        "supported_modes": ["event_arb", "perp_perp"],
    }

    def default_config(self, symbols: list[str] | None = None, **overrides: Any) -> StrategyConfig:
        params: dict[str, Any] = {
            "min_edge_bps": 22.0,
            "confirm_decay_bps": 4.0,
            "rebalance_cost_bps": 6.0,
            "gas_usd_per_tx": 0.40,
            "revert_rate_pct": 8.0,
            "notional_per_trade_usd": 500.0,
            "execution_lag_bars": 1,
            "event_type_filter": "",
            "event_csv_path": "",
            "csv_provenance": "UNVERIFIED_LOCAL_CSV",
        }
        if "params" in overrides and isinstance(overrides["params"], dict):
            params.update(overrides.pop("params"))
        base = StrategyConfig(
            strategy_id="structural_arb_default",
            version="v1.0",
            family="structural_dislocation_arb",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="okx",
            mode="perp_perp",
            symbols=list(symbols or ["BTC", "ETH", "SOL", "WLD", "ENA", "UNI"]),
            capital_usd=3000.0,
            leverage=1.0,
            max_positions=3,
            per_position_weight=0.20,
            entry_spread_apr_pct=0.0,
            exit_spread_apr_pct=0.0,
            lookback_hours=1,
            exit_confirm_hours=1,
            min_hold_hours=1,
            fee_a_bps=1.5,
            fee_b_bps=2.0,
            slippage_per_leg_bps=1.5,
            rebalance_drag_bps_per_month=0.0,
            plugin_id=self.plugin_id,
            execution_mode="TAKER_TAKER",
            params=params,
        )
        if overrides:
            d = base.to_dict()
            d.update(overrides)
            return StrategyConfig.from_dict(d)
        return base

    def mutable_param_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(STRUCTURAL_ARB_BOUNDS)

    def validate_and_clamp(
        self, cfg: StrategyConfig, next_version: str, changes: dict[str, Any]
    ) -> StrategyConfig:
        d = cfg.to_dict()
        d["version"] = next_version
        d["status"] = "BACKTESTED"
        d["plugin_id"] = self.plugin_id
        params = dict(d.get("params") or {})
        for k, v in changes.items():
            if k in STRUCTURAL_ARB_BOUNDS and isinstance(v, (int, float)) and not isinstance(v, bool):
                fv = float(v)
                if not math.isfinite(fv):
                    continue
                lo, hi = STRUCTURAL_ARB_BOUNDS[k]
                params[k] = round(max(lo, min(hi, fv)), 3)
        d["params"] = params
        return StrategyConfig.from_dict(d)

    def evaluate(
        self,
        store: MarketDataStore,
        cfg: StrategyConfig,
        start_ms: int | None = None,
        end_ms: int | None = None,
        extra_slippage_bps_per_leg: float = 0.0,
        execution_mode_override: str | None = None,
    ) -> BacktestResult:
        run_cfg = dataclasses.replace(cfg, execution_mode=execution_mode_override) if execution_mode_override else cfg
        min_edge = _p(run_cfg, "min_edge_bps", 22.0)
        decay_bps = _p(run_cfg, "confirm_decay_bps", 4.0)
        rebal_bps = _p(run_cfg, "rebalance_cost_bps", 6.0)
        gas_usd = _p(run_cfg, "gas_usd_per_tx", 0.40)
        revert_pct = _p(run_cfg, "revert_rate_pct", 8.0)
        eff_weight = min(run_cfg.per_position_weight, 1.0 / max(1, run_cfg.max_positions))
        max_slot_notional = max(10.0, run_cfg.capital_usd * eff_weight)
        notional = min(_p(run_cfg, "notional_per_trade_usd", 500.0), max_slot_notional)
        exec_lag = max(0, int(round(_p(run_cfg, "execution_lag_bars", 1.0))))
        ev_filter = str(run_cfg.params.get("event_type_filter", "")).strip()
        csv_path_str = str(run_cfg.params.get("event_csv_path", "")).strip()
        csv_prov = str(run_cfg.params.get("csv_provenance", "UNVERIFIED_LOCAL_CSV")).strip() or "UNVERIFIED_LOCAL_CSV"

        # (ts_ms, symbol, ex_ante_signal_bps, ex_post_realized_bps, label)
        events: list[tuple[int, str, float, float, str]] = []
        is_csv = False
        if csv_path_str:
            p_csv = Path(csv_path_str).resolve()
            if p_csv.exists() and p_csv.is_file():
                is_csv = True
                events, ds_sha, prov = self._load_csv_events(p_csv, ev_filter, start_ms, end_ms, csv_prov)
            else:
                events, ds_sha, prov = [], "", "EMPTY_DATASET"
        else:
            events, ds_sha, prov = self._extract_candle_dislocations(store, run_cfg, start_ms, end_ms, min_edge, exec_lag)

        # Red-Team #12: Full execution-mode-aware fee schedule (4-leg RT for perp-perp, 2-leg for atomic CSV)
        eff_fee_a = run_cfg.fee_a_bps
        eff_fee_b = run_cfg.fee_b_bps
        if run_cfg.execution_mode == "MAKER_THEN_TAKER_HEDGE":
            eff_fee_b = max(eff_fee_b, TAKER_FEE_BPS.get(run_cfg.venue_b, eff_fee_b))
        elif run_cfg.execution_mode == "TAKER_TAKER":
            eff_fee_a = max(eff_fee_a, TAKER_FEE_BPS.get(run_cfg.venue_a, eff_fee_a))
            eff_fee_b = max(eff_fee_b, TAKER_FEE_BPS.get(run_cfg.venue_b, eff_fee_b))
        rt_mult = 1.0 if is_csv else 2.0
        fee_bps = rt_mult * (eff_fee_a + eff_fee_b)
        slip_bps = rt_mult * 2.0 * (run_cfg.slippage_per_leg_bps + extra_slippage_bps_per_leg) + decay_bps
        revert_frac = revert_pct / 100.0

        trades: list[TradeRecord] = []
        equity_curve: list[tuple[int, float]] = []
        eq = run_cfg.capital_usd
        seq = 0
        for ts_ms, sym, ex_ante_signal_bps, realized_bps, label in events:
            # Red-Team #1: filter STRICTLY on ex-ante signal; book ALL realized outcomes (positive & negative)!
            if ex_ante_signal_bps < min_edge:
                continue
            seq += 1
            gross_usd = notional * (realized_bps / 10000.0)
            fee_usd = notional * (fee_bps / 10000.0) + gas_usd * (1.0 + revert_frac)
            slip_usd = notional * (slip_bps / 10000.0)
            rebal_usd = notional * (rebal_bps / 10000.0)
            # Only non-reverted fraction captures gross (losses on filled leg still hit at (1 - revert_frac))
            realized_gross = gross_usd * (1.0 - revert_frac)
            net_usd = realized_gross - fee_usd - slip_usd - rebal_usd
            eq += net_usd
            trades.append(
                TradeRecord(
                    trade_id=f"SA-{seq:04d}",
                    symbol=sym,
                    direction=label or "STRUCTURAL_ARB",
                    entry_ts_ms=ts_ms,
                    exit_ts_ms=ts_ms + (60_000 if is_csv else HOUR_MS),
                    hold_hours=0.02 if is_csv else 1.0,
                    notional_per_leg_usd=round(notional, 2),
                    entry_spread_apr_pct=round(ex_ante_signal_bps, 2),
                    gross_funding_usd=0.0,
                    basis_pnl_usd=round(realized_gross, 4),
                    fee_usd=round(fee_usd, 4),
                    slippage_usd=round(slip_usd, 4),
                    rebalance_usd=round(rebal_usd, 4),
                    net_pnl_usd=round(net_usd, 4),
                    exit_reason="ATOMIC_CAPTURE" if is_csv else "DISLOCATION_1H_CLOSE",
                )
            )
            equity_curve.append((ts_ms, eq))

        if not events:
            min_ts, max_ts = start_ms or 0, end_ms or 0
        else:
            min_ts = start_ms if start_ms is not None else events[0][0]
            max_ts = end_ms if end_ms is not None else events[-1][0]
            if not equity_curve:
                equity_curve = [(min_ts, run_cfg.capital_usd), (max_ts, run_cfg.capital_usd)]

        # Build continuous daily equity timeline across [min_ts, max_ts] so zero-trade days are NOT omitted!
        daily_curve: list[tuple[int, float]] = []
        if max_ts > min_ts and min_ts > 0:
            day_ms = 86400 * 1000
            ev_idx = 0
            cur_eq = run_cfg.capital_usd
            t_cur = min_ts
            while t_cur <= max_ts:
                while ev_idx < len(equity_curve) and equity_curve[ev_idx][0] <= t_cur:
                    cur_eq = equity_curve[ev_idx][1]
                    ev_idx += 1
                daily_curve.append((t_cur, cur_eq))
                t_cur += day_ms
            if daily_curve and daily_curve[-1][0] < max_ts:
                daily_curve.append((max_ts, eq))
        else:
            daily_curve = list(equity_curve)

        days = max(1.0 / 24.0, (max_ts - min_ts) / (86400.0 * 1000.0)) if max_ts > min_ts else 1.0
        gross_alpha = sum(t.basis_pnl_usd for t in trades)
        fees = sum(t.fee_usd for t in trades)
        slips = sum(t.slippage_usd for t in trades)
        rebals = sum(t.rebalance_usd for t in trades)
        net_pnl = sum(t.net_pnl_usd for t in trades)
        wins = [t.net_pnl_usd for t in trades if t.net_pnl_usd > 0]
        losses = [-t.net_pnl_usd for t in trades if t.net_pnl_usd < 0]
        pf = (sum(wins) / sum(losses)) if sum(losses) > 0 else (99.0 if wins else 0.0)
        tot_ret = (net_pnl / run_cfg.capital_usd) * 100.0 if run_cfg.capital_usd > 0 else 0.0
        apr = tot_ret * (365.0 / days)

        peak = run_cfg.capital_usd
        mdd = 0.0
        for _, e in equity_curve:
            if e > peak:
                peak = e
            if peak > 0:
                mdd = max(mdd, ((peak - e) / peak) * 100.0)

        vals = [e for _, e in daily_curve]
        drets = [(vals[i] / vals[i - 1]) - 1.0 for i in range(1, len(vals)) if vals[i - 1] > 0]
        if len(drets) >= 3:
            mu = sum(drets) / len(drets)
            sd = math.sqrt(sum((r - mu) ** 2 for r in drets) / (len(drets) - 1))
            sharpe = (mu / sd) * math.sqrt(365.0) if sd > 1e-12 else 0.0
        else:
            sharpe = 0.0

        calmar = (apr / mdd) if mdd > 0.05 else (apr / 0.05 if apr > 0 else 0.0)
        fee_ratio = ((fees + slips + rebals) / gross_alpha) if gross_alpha > 0 else 99.0
        sorted_p = sorted((t.net_pnl_usd for t in trades), reverse=True)
        top5_rem = sum(sorted_p[5:]) if len(sorted_p) > 5 else 0.0

        by_sym: dict[str, dict[str, float]] = {}
        for t in trades:
            b = by_sym.setdefault(t.symbol, {"trades": 0.0, "wins": 0.0, "net_pnl_usd": 0.0})
            b["trades"] += 1.0
            b["wins"] += 1.0 if t.net_pnl_usd > 0 else 0.0
            b["net_pnl_usd"] += t.net_pnl_usd
        best_s = max(by_sym, key=lambda s: by_sym[s]["net_pnl_usd"]) if by_sym else ""
        top_s_rem = net_pnl - max(0.0, by_sym[best_s]["net_pnl_usd"]) if best_s else 0.0

        metrics = BacktestMetrics(
            window_start_ms=min_ts,
            window_end_ms=max_ts,
            days=round(days, 2),
            capital_usd=run_cfg.capital_usd,
            trade_count=len(trades),
            win_count=len(wins),
            win_rate=round(len(wins) / len(trades), 4) if trades else 0.0,
            avg_hold_hours=round(sum(t.hold_hours for t in trades) / len(trades), 2) if trades else 0.0,
            gross_funding_usd=0.0,
            basis_pnl_usd=round(gross_alpha, 4),
            total_fees_usd=round(fees, 4),
            total_slippage_usd=round(slips, 4),
            total_rebalance_usd=round(rebals, 4),
            net_pnl_usd=round(net_pnl, 4),
            total_return_pct=round(tot_ret, 4),
            annualized_apr_pct=round(apr, 2),
            sharpe_ratio=round(sharpe, 3),
            calmar_ratio=round(calmar, 3),
            max_drawdown_pct=round(mdd, 4),
            profit_factor=round(pf, 3),
            fee_to_gross_ratio=round(fee_ratio, 4),
            avg_capital_utilization_pct=round((notional / run_cfg.capital_usd) * 100.0, 2) if (trades and run_cfg.capital_usd > 0) else 0.0,
            monthly_usd_est=round(net_pnl * (30.0 / days), 2),
            top5_removed_net_pnl_usd=round(top5_rem, 4),
            top_symbol_removed_net_pnl_usd=round(top_s_rem, 4),
            best_symbol=best_s,
            provenance=prov,
            dataset_sha256=ds_sha,
            by_symbol=by_sym,
            gross_alpha_usd=round(gross_alpha, 4),
            liquidation_count=0,
            forced_rebalance_count=0,
        )
        return BacktestResult(run_cfg, metrics, trades, daily_curve, 100.0 if trades else 0.0, [])

    @staticmethod
    def _load_csv_events(
        path: Path, ev_filter: str, start_ms: int | None, end_ms: int | None, provenance: str
    ) -> tuple[list[tuple[int, str, float, float, str]], str, str]:
        h = hashlib.sha256()
        events: list[tuple[int, str, float, float, str]] = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                h.update(str(sorted((str(k), v) for k, v in row.items())).encode("utf-8"))
                label = str(row.get("track") or row.get("strategy") or row.get("event_type") or row.get("direction") or "CSV_EVENT")
                if ev_filter and ev_filter not in label:
                    continue
                raw_ts = row.get("ts_ms") or row.get("ts") or row.get("timestamp_ms") or row.get("ts_iso") or "0"
                ts = 0
                try:
                    ts = int(float(raw_ts))
                    if 0 < ts < 10_000_000_000:
                        ts *= 1000
                except ValueError:
                    try:
                        ts = int(datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).timestamp() * 1000)
                    except ValueError:
                        continue
                if start_ms is not None and ts < start_ms:
                    continue
                if end_ms is not None and ts > end_ms:
                    continue
                # Separate ex-ante signal (raw_net_bp / peak_edge_bp) from ex-post realized outcome (realized_net_bp)
                sig_str = row.get("raw_net_bp") or row.get("peak_edge_bp") or row.get("edge_bps") or row.get("spread_bp") or "0"
                real_str = row.get("realized_net_bp") or row.get("net_bp") or sig_str
                try:
                    sig_bps = float(sig_str)
                    real_bps = float(real_str)
                except ValueError:
                    continue
                sym = str(row.get("symbol") or row.get("pair") or "USDC").upper()
                events.append((ts, sym, abs(sig_bps), real_bps, label))
        events.sort(key=lambda x: x[0])
        return events, h.hexdigest(), provenance

    @staticmethod
    def _extract_candle_dislocations(
        store: MarketDataStore,
        cfg: StrategyConfig,
        start_ms: int | None,
        end_ms: int | None,
        min_edge: float,
        exec_lag: int = 1,
    ) -> tuple[list[tuple[int, str, float, float, str]], str, str]:
        """Extract causal dual-venue dislocation events separating ex-ante signal from ex-post realized PnL (Red-Team #1)."""
        ds_sha, prov, _ = store.fingerprint_slice([cfg.venue_a, cfg.venue_b], list(cfg.symbols), start_ms, end_ms)
        events: list[tuple[int, str, float, float, str]] = []
        for sym in cfg.symbols:
            ca = {c.ts_ms: c.open for c in store.load_candles(cfg.venue_a, sym, start_ms, end_ms)}
            cb = {c.ts_ms: c.open for c in store.load_candles(cfg.venue_b, sym, start_ms, end_ms)}
            common_ts = sorted(set(ca) & set(cb))
            step = max(0, exec_lag)
            for idx in range(len(common_ts) - (step + 1)):
                t_sig = common_ts[idx]
                t_ent = common_ts[idx + step]
                t_ext = common_ts[idx + step + 1]
                if (t_ent - t_sig != step * HOUR_MS) or (t_ext - t_ent != HOUR_MS):
                    continue
                sa, sb = ca[t_sig], cb[t_sig]
                ea, eb = ca[t_ent], cb[t_ent]
                xa, xb = ca[t_ext], cb[t_ext]
                if min(sa, sb, ea, eb, xa, xb) <= 0:
                    continue
                sig_basis = ((sa - sb) / sb) * 10000.0
                if abs(sig_basis) < min_edge:
                    continue
                # Realized PnL strictly from fill bar (t_ent) to exit bar (t_ext) — can be positive OR negative!
                if sig_basis > 0:
                    realized_bps = (((xb / eb) - 1.0) - ((xa / ea) - 1.0)) * 10000.0
                    label = "SHORT_A_LONG_B"
                else:
                    realized_bps = (((xa / ea) - 1.0) - ((xb / eb) - 1.0)) * 10000.0
                    label = "LONG_A_SHORT_B"
                events.append((t_ent, sym, abs(sig_basis), realized_bps, label))
        events.sort(key=lambda x: x[0])
        return events, ds_sha, prov

    def plateau_variants(self, cfg: StrategyConfig) -> list[tuple[str, StrategyConfig]]:
        me = _p(cfg, "min_edge_bps", 22.0)
        dc = _p(cfg, "confirm_decay_bps", 4.0)
        variants: list[tuple[str, StrategyConfig]] = []
        for ef in (0.8, 1.0, 1.2):
            for df in (0.75, 1.0, 1.25):
                p = dict(cfg.params)
                p["min_edge_bps"] = round(max(8.0, me * ef), 2)
                p["confirm_decay_bps"] = round(max(0.0, dc * df), 2)
                variants.append((f"edge={p['min_edge_bps']}bp/decay={p['confirm_decay_bps']}bp", dataclasses.replace(cfg, params=p)))
        return variants

    def diagnose_telemetry(self, cfg: StrategyConfig, metrics: BacktestMetrics) -> FeedbackEvent:
        now = utc_now_ms()
        anomaly, sev = "HEALTHY", "INFO"
        summary = f"Structural arb healthy: trades={metrics.trade_count}, net=${metrics.net_pnl_usd:+.2f}."
        if metrics.trade_count < 5:
            anomaly, sev = "LOW_SAMPLE", "P1"
            summary = f"Insufficient structural arb events ({metrics.trade_count} < 5); expand synchronized event feed."
        elif metrics.fee_to_gross_ratio > 0.70 or metrics.net_pnl_usd <= 0:
            anomaly, sev = "COST_OVERRUN", "P0"
            summary = f"Gas/rebalance/decay friction overwhelmed raw edge (net=${metrics.net_pnl_usd:+.2f}, cost_ratio={metrics.fee_to_gross_ratio:.2f}x)."
        return FeedbackEvent(
            event_id=f"FB-{sha256_obj({'s': cfg.strategy_id, 't': now, 'a': anomaly})[:10]}",
            ts_ms=now,
            strategy_id=cfg.strategy_id,
            severity=sev,
            anomaly_type=anomaly,
            summary=summary,
            metrics_snapshot=metrics.to_dict(),
        )

    def fallback_proposal(
        self, cfg: StrategyConfig, feedback: FeedbackEvent, metrics: BacktestMetrics
    ) -> tuple[dict[str, Any], str, str, str]:
        me = _p(cfg, "min_edge_bps", 22.0)
        if feedback.anomaly_type == "LOW_SAMPLE":
            return (
                {"min_edge_bps": max(10.0, round(me * 0.75, 1))},
                "Lower min_edge_bps threshold slightly to observe sufficient dislocation sample size.",
                "Expands sample coverage while preserving positive net hurdle above immutable gas + rebalance drag.",
                "Reject if net PnL turns negative or cost_ratio exceeds 65%.",
            )
        return (
            {"min_edge_bps": min(100.0, round(me * 1.35, 1))},
            "Raise min_edge_bps filter to require larger dislocation cushion over immutable gas & inventory rebalance cost.",
            "Eliminates marginal fills whose raw edge gets eaten by gas, reverts, and one-way rebalancing.",
            "Reject if trade_count < 5 or STRESS net PnL <= 0.",
        )

    def on_event(self, event: MarketEvent, state: dict[str, Any], cfg: StrategyConfig) -> list[OrderIntent]:
        if event.kind not in ("DEX_QUOTE", "BBO"):
            return []
        edge = float(event.payload.get("edge_bps", 0.0))
        min_edge = _p(cfg, "min_edge_bps", 22.0)
        if edge >= min_edge:
            eff_weight = min(cfg.per_position_weight, 1.0 / max(1, cfg.max_positions))
            notional = min(_p(cfg, "notional_per_trade_usd", 500.0), cfg.capital_usd * eff_weight)
            return [OrderIntent(cfg.strategy_id, event.symbol, "OPEN_SHORT_A_LONG_B", notional, edge, "STRUCTURAL_EDGE_HIT", event.ts_ms)]
        return []

    def min_economic_kill_horizon_hours(self, cfg: StrategyConfig) -> float:
        return 12.0
