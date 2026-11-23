"""Reference Strategy Plugin #1: Cross-Venue Perp-Perp & Spot-Perp Funding Carry."""

from __future__ import annotations

import dataclasses
from typing import Any

from alphaloop.core.models import (
    BacktestMetrics,
    FeedbackEvent,
    MarketEvent,
    OrderIntent,
    StrategyConfig,
)
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import BacktestResult, FundingCarryBacktester
from alphaloop.harness.telemetry import TelemetryMonitor
from alphaloop.plugins.base import BaseStrategyPlugin

FUNDING_CARRY_BOUNDS: dict[str, tuple[float, float]] = {
    "entry_spread_apr_pct": (4.0, 60.0),
    "exit_spread_apr_pct": (0.0, 20.0),
    "lookback_hours": (8.0, 240.0),
    "exit_confirm_hours": (4.0, 96.0),
    "min_hold_hours": (8.0, 240.0),
    "max_basis_entry_bps": (10.0, 100.0),
    "stop_loss_basis_bps": (80.0, 400.0),
    "max_positions": (1.0, 10.0),
}


class FundingCarryPlugin(BaseStrategyPlugin):
    """Delta-neutral cross-venue funding carry strategy with hysteresis & margin safety."""

    plugin_id = "funding_carry"
    display_name = "Cross-Venue Funding Carry"
    strategy_family = "funding_carry"
    description = (
        "Captures structural DEX-vs-CEX funding rate differentials (e.g. Lighter/Hyperliquid vs Gate/OKX) "
        "with causal rolling APR smoothing, hysteresis anti-churn, basis guard, and per-leg margin protection."
    )
    required_data_spec = {
        "tables": ["funding_rates", "candles_1h"],
        "requires_dual_candles_for_live": True,
        "min_history_days": 30,
        "supported_modes": ["perp_perp", "spot_perp"],
    }

    def __init__(self) -> None:
        self._telemetry = TelemetryMonitor()

    def default_config(self, symbols: list[str] | None = None, **overrides: Any) -> StrategyConfig:
        base = StrategyConfig(
            strategy_id="funding_carry_default",
            version="v2.2",
            family="dex_cex_funding_carry",
            status="DRAFT",
            venue_a="lighter",
            venue_b="gate",
            mode="perp_perp",
            symbols=list(symbols or ["BTC", "ETH", "SOL", "WLD", "NEAR", "ENA", "ZEC", "UNI"]),
            capital_usd=3000.0,
            leverage=4.0,
            max_positions=8,
            per_position_weight=0.125,
            entry_spread_apr_pct=6.0,
            exit_spread_apr_pct=0.5,
            lookback_hours=120,
            exit_confirm_hours=36,
            min_hold_hours=96,
            max_basis_entry_bps=40.0,
            stop_loss_basis_bps=200.0,
            fee_a_bps=0.0,
            fee_b_bps=2.0,
            slippage_per_leg_bps=1.5,
            rebalance_drag_bps_per_month=3.0,
            allow_bidirectional=True,
            plugin_id=self.plugin_id,
            execution_mode="MAKER_MAKER",
        )
        if overrides:
            d = base.to_dict()
            d.update(overrides)
            return StrategyConfig.from_dict(d)
        return base

    def mutable_param_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(FUNDING_CARRY_BOUNDS)

    def validate_and_clamp(
        self,
        cfg: StrategyConfig,
        next_version: str,
        changes: dict[str, Any],
    ) -> StrategyConfig:
        import math

        d = cfg.to_dict()
        d["version"] = next_version
        d["status"] = "BACKTESTED"
        d["plugin_id"] = self.plugin_id
        params = dict(d.get("params") or {})
        param_bounds: dict[str, tuple[float, float]] = {
            "protective_margin_exit_pct": (0.35, 0.75),
            "vol_target_ann_pct": (60.0, 220.0),
            "min_symbol_leverage": (1.0, 2.5),
            "vol_lookback_hours": (48.0, 240.0),
        }
        for key, val in changes.items():
            if key == "use_atb_buggy_rules" and isinstance(val, bool):
                d[key] = val
            elif key == "vol_target_leverage_enabled" and isinstance(val, bool):
                params[key] = val
            elif key in param_bounds and isinstance(val, (int, float)) and not isinstance(val, bool):
                fval = float(val)
                if not math.isfinite(fval):
                    continue
                lo, hi = param_bounds[key]
                clamped = max(lo, min(hi, fval))
                params[key] = int(round(clamped)) if key == "vol_lookback_hours" else round(clamped, 3)
            elif key in FUNDING_CARRY_BOUNDS and isinstance(val, (int, float)) and not isinstance(val, bool):
                fval = float(val)
                if not math.isfinite(fval):
                    continue
                lo, hi = FUNDING_CARRY_BOUNDS[key]
                clamped = max(lo, min(hi, fval))
                if isinstance(getattr(cfg, key), int):
                    d[key] = int(round(clamped))
                else:
                    d[key] = round(clamped, 3)
        # Preserve physical storm-exit floor (immutable to Agent)
        if "protective_exit_slip_bps" in params:
            raw_s = params["protective_exit_slip_bps"]
            if isinstance(raw_s, bool) or not isinstance(raw_s, (int, float)) or not math.isfinite(float(raw_s)):
                params["protective_exit_slip_bps"] = 5.0
            else:
                params["protective_exit_slip_bps"] = max(5.0, float(raw_s))
        d["params"] = params
        # Invariant 1: Prevent >100% capital over-allocation when max_positions increases (Red-Team #5.3)
        d["per_position_weight"] = round(min(float(d["per_position_weight"]), 1.0 / max(1, int(d["max_positions"]))), 4)
        # Invariant 2: entry hurdle must stay >= 4% APR above exit hurdle (prevents ATB H6 bug)
        if d["exit_spread_apr_pct"] >= d["entry_spread_apr_pct"] - 4.0:
            d["exit_spread_apr_pct"] = max(0.5, round(d["entry_spread_apr_pct"] - 6.0, 2))
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
        run_cfg = cfg
        if execution_mode_override:
            run_cfg = dataclasses.replace(cfg, execution_mode=execution_mode_override)
        bt = FundingCarryBacktester(store)
        return bt.run(
            run_cfg,
            start_ms=start_ms,
            end_ms=end_ms,
            extra_slippage_bps_per_leg=extra_slippage_bps_per_leg,
        )

    def plateau_variants(self, cfg: StrategyConfig) -> list[tuple[str, StrategyConfig]]:
        variants: list[tuple[str, StrategyConfig]] = []
        for ef in (0.75, 1.0, 1.25):
            for xf in (0.70, 1.0, 1.30):
                cand_entry = max(4.0, round(cfg.entry_spread_apr_pct * ef, 2))
                cand_exit = min(cand_entry - 2.0, max(0.5, round(cfg.exit_spread_apr_pct * xf, 2)))
                cand = dataclasses.replace(
                    cfg,
                    entry_spread_apr_pct=cand_entry,
                    exit_spread_apr_pct=cand_exit,
                )
                variants.append((f"entry={cand_entry}%/exit={cand_exit}%", cand))
        return variants

    def diagnose_telemetry(self, cfg: StrategyConfig, metrics: BacktestMetrics) -> FeedbackEvent:
        return self._telemetry.diagnose(cfg, metrics)

    def fallback_proposal(
        self,
        cfg: StrategyConfig,
        feedback: FeedbackEvent,
        metrics: BacktestMetrics,
    ) -> tuple[dict[str, Any], str, str, str]:
        if feedback.anomaly_type == "MARGIN_LIQUIDATION":
            p = cfg.params or {}
            cur_prot = float(p.get("protective_margin_exit_pct", 0.65) or 0.65)
            cur_vt = float(p.get("vol_target_ann_pct", 130.0) or 130.0)
            next_prot = max(0.40, round(cur_prot - 0.10, 2))
            next_vt = max(80.0, round(cur_vt - 20.0, 1))
            changes = {
                "vol_target_leverage_enabled": True,
                "vol_target_ann_pct": next_vt,
                "min_symbol_leverage": 1.5,
                "protective_margin_exit_pct": next_prot,
                "stop_loss_basis_bps": max(80.0, round(cfg.stop_loss_basis_bps * 0.75, 1)),
            }
            return (
                changes,
                f"Enable/tighten causal 7d volatility-tiered leverage (target {next_vt:.0f}%) and {next_prot:.0%} protective margin exit after liquidation.",
                "Caps leverage on high-volatility altcoins and atomically closes both legs before maintenance margin liquidation.",
                "Reject immediately if any liquidation_count > 0 remains.",
            )
        if cfg.use_atb_buggy_rules or feedback.anomaly_type == "CHURN_EXCESS":
            changes = {
                "use_atb_buggy_rules": False,
                "entry_spread_apr_pct": 6.0,
                "exit_spread_apr_pct": 0.5,
                "lookback_hours": 120,
                "min_hold_hours": 96,
                "exit_confirm_hours": 36,
            }
            return (
                changes,
                "Eliminate inverted hysteresis fee churn by smoothing signal over 120h and holding carry down to 0.5% exit.",
                "Captures structural DEX-vs-CEX funding differential while amortizing 4-leg round-trip costs across multi-day holds.",
                "Reject immediately if STRESS net PnL <= 0, OOS retention < 25%, or plateau < 75%.",
            )
        if feedback.anomaly_type == "BASIS_SHOCK":
            changes = {
                "max_basis_entry_bps": max(15.0, round(cfg.max_basis_entry_bps * 0.6, 1)),
                "stop_loss_basis_bps": max(100.0, round(cfg.stop_loss_basis_bps * 0.75, 1)),
                "lookback_hours": min(144, cfg.lookback_hours + 24),
            }
            return (
                changes,
                "Tighten entry basis dislocation guard and basis stop-loss to cap adverse mark-price divergence.",
                "Prevents entering stretched premiums right before mean-reversion hurts the short-perp leg.",
                "Reject if basis_pnl_usd continues to erode > 50% of gross carry.",
            )
        if feedback.anomaly_type == "EDGE_DECAY":
            changes = {
                "entry_spread_apr_pct": max(5.0, round(cfg.entry_spread_apr_pct * 0.85, 1)),
                "exit_spread_apr_pct": max(0.5, round(cfg.exit_spread_apr_pct * 0.75, 1)),
                "lookback_hours": 72,
                "min_hold_hours": 72,
            }
            return (
                changes,
                "Smooth trailing signal to 72h and calibrate entry hurdle to capture persistent mid-tier carry spreads.",
                "Improves capital utilization while maintaining low turnover.",
                "Reject if STRESS APR <= 0 or Cost/Gross > 65%.",
            )
        changes = {
            "lookback_hours": min(144, cfg.lookback_hours + 12),
            "min_hold_hours": min(120, cfg.min_hold_hours + 12),
        }
        return (
            changes,
            "Incremental turnover reduction on healthy carry engine to improve Sharpe and cost efficiency.",
            "Longer smoothing window filters brief 1h funding spikes without sacrificing structural spread carry.",
            "Reject if Sharpe or STRESS net PnL declines relative to baseline.",
        )

    def on_event(
        self,
        event: MarketEvent,
        state: dict[str, Any],
        cfg: StrategyConfig,
    ) -> list[OrderIntent]:
        """Pure event step for MicroRunSession: tracks live BBO basis stops & funding spreads."""
        intents: list[OrderIntent] = []
        sym = event.symbol
        if sym not in cfg.symbols:
            return intents

        latest_mids: dict[tuple[str, str], float] = state.setdefault("latest_mids", {})
        latest_apr: dict[tuple[str, str], float] = state.setdefault("latest_apr", {})
        open_pos: dict[str, dict[str, Any]] = state.setdefault("open_pos", {})

        if event.kind == "BBO":
            bid = float(event.payload.get("bid", 0.0))
            ask = float(event.payload.get("ask", 0.0))
            if bid > 0 and ask >= bid:
                latest_mids[(event.venue, sym)] = (bid + ask) * 0.5
        elif event.kind == "FUNDING":
            latest_apr[(event.venue, sym)] = float(event.payload.get("apr_pct", 0.0))

        px_a = latest_mids.get((cfg.venue_a, sym))
        px_b = latest_mids.get((cfg.venue_b, sym))
        if sym in open_pos and px_a and px_b:
            pos = open_pos[sym]
            ea, eb = pos["entry_px_a"], pos["entry_px_b"]
            if ea > 0 and eb > 0:
                ret_a = (px_a / ea) - 1.0
                ret_b = (px_b / eb) - 1.0
                rel = (ret_b - ret_a) if pos["direction"] == "SHORT_A_LONG_B" else (ret_a - ret_b)
                if rel * 10000.0 <= -cfg.stop_loss_basis_bps:
                    intents.append(OrderIntent(cfg.strategy_id, sym, "CLOSE", pos["notional_usd"], rel * 10000.0, "BASIS_STOP", event.ts_ms))
                    open_pos.pop(sym, None)
                    return intents

        apr_a = latest_apr.get((cfg.venue_a, sym))
        apr_b = latest_apr.get((cfg.venue_b, sym))
        if event.kind == "FUNDING" and sym in open_pos and apr_a is not None and apr_b is not None:
            pos = open_pos[sym]
            held_h = (event.ts_ms - int(pos.get("entry_ts_ms", event.ts_ms))) / 3_600_000.0
            signed_spread = (apr_a - apr_b) if pos["direction"] == "SHORT_A_LONG_B" else (apr_b - apr_a)
            if held_h >= float(cfg.min_hold_hours):
                if signed_spread < float(cfg.exit_spread_apr_pct):
                    low_since = pos.get("low_since_ms")
                    if low_since is None:
                        pos["low_since_ms"] = int(event.ts_ms)
                        low_since = int(event.ts_ms)
                    low_h = (int(event.ts_ms) - int(low_since)) / 3_600_000.0
                    if low_h >= float(cfg.exit_confirm_hours):
                        intents.append(
                            OrderIntent(cfg.strategy_id, sym, "CLOSE", pos["notional_usd"], signed_spread, "HYSTERESIS_EXIT", event.ts_ms)
                        )
                        open_pos.pop(sym, None)
                        return intents
                else:
                    pos["low_since_ms"] = None

        # Red-Team BLOCKER #3: NEVER open a carry position before BOTH venues have valid live BBO prices!
        if (
            px_a is not None
            and px_b is not None
            and px_a > 0.0
            and px_b > 0.0
            and apr_a is not None
            and apr_b is not None
            and sym not in open_pos
            and len(open_pos) < cfg.max_positions
        ):
            spread = apr_a - apr_b
            if abs(spread) >= cfg.entry_spread_apr_pct:
                direction = "SHORT_A_LONG_B" if spread > 0 else "LONG_A_SHORT_B"
                action = "OPEN_SHORT_A_LONG_B" if spread > 0 else "OPEN_LONG_A_SHORT_B"
                notional = cfg.capital_usd * cfg.per_position_weight * (cfg.leverage / 2.0)
                raw_basis = ((px_a - px_b) / px_b) * 10000.0
                adverse = -raw_basis if spread > 0 else raw_basis
                if adverse > cfg.max_basis_entry_bps:
                    return intents
                open_pos[sym] = {
                    "direction": direction,
                    "notional_usd": notional,
                    "entry_px_a": px_a,
                    "entry_px_b": px_b,
                    "entry_ts_ms": event.ts_ms,
                    "low_streak": 0,
                }
                intents.append(OrderIntent(cfg.strategy_id, sym, action, notional, abs(spread), "FUNDING_SPREAD_ENTRY", event.ts_ms))
        return intents

    def min_economic_kill_horizon_hours(self, cfg: StrategyConfig) -> float:
        return max(48.0, float(cfg.min_hold_hours))
