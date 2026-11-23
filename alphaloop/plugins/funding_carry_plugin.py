"""Reference Strategy Plugin #1: Cross-Venue Perp-Perp & Spot-Perp Funding Carry."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from alphaloop.core.models import (
    BacktestMetrics,
    FeedbackEvent,
    MarketEvent,
    OrderIntent,
    StrategyConfig,
)
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import (
    HOUR_MS,
    KNOWN_FUNDING_CARRY_PARAM_KEYS,
    BacktestResult,
    FundingCarryBacktester,
    _basis_pnl_usd,
    _param_int,
    compute_carry_slot_sizing,
    compute_causal_symbol_leverage,
    compute_rolling_funding_apr,
    evaluate_carry_entry_signal,
    evaluate_carry_margin_guard,
    validate_funding_carry_params,
)
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
        valid_fields = set(StrategyConfig.__dataclass_fields__.keys())
        unknown_overrides = sorted(k for k in overrides.keys() if k not in valid_fields)
        if unknown_overrides:
            raise ValueError(
                f"Unknown StrategyConfig override keys for funding_carry: {unknown_overrides}"
            )
        if "params" in overrides:
            validate_funding_carry_params(overrides["params"])
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
            cfg = StrategyConfig.from_dict(d)
            validate_funding_carry_params(cfg.params)
            return cfg
        return base

    def mutable_param_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(FUNDING_CARRY_BOUNDS)

    def validate_and_clamp(
        self,
        cfg: StrategyConfig,
        next_version: str,
        changes: dict[str, Any],
    ) -> StrategyConfig:
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
        # Strip any unknown keys defensively before strict schema check
        params = {k: v for k, v in params.items() if k in KNOWN_FUNDING_CARRY_PARAM_KEYS}
        validate_funding_carry_params(params)
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
        warmup_start_ms: int | None = None,
    ) -> BacktestResult:
        validate_funding_carry_params(cfg.params)
        run_cfg = cfg
        if execution_mode_override:
            run_cfg = dataclasses.replace(cfg, execution_mode=execution_mode_override)
        bt = FundingCarryBacktester(store)
        return bt.run(
            run_cfg,
            start_ms=start_ms,
            end_ms=end_ms,
            extra_slippage_bps_per_leg=extra_slippage_bps_per_leg,
            warmup_start_ms=warmup_start_ms,
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
        """Single-kernel streaming step matching FundingCarryBacktester.run() math identically.

        Enforces:
        - Strict param validation (no dead/unconsumed keys or invalid types).
        - Causal rolling funding APR over `cfg.lookback_hours` (`compute_rolling_funding_apr`).
        - Causal volatility-tiered leverage (`compute_causal_symbol_leverage`) + exact slot sizing.
        - Isolated margin guard (`evaluate_carry_margin_guard`: liquidation >=90% > protective exit > 65% rebalance).
        - 24h post-stop cooldown, hysteresis exit confirmation, and paired-venue `defer_decision` barrier.
        """
        validate_funding_carry_params(cfg.params)
        intents: list[OrderIntent] = []
        sym = event.symbol
        if sym not in cfg.symbols:
            return intents

        ts_ms = int(event.ts_ms)
        decision_ts = ts_ms
        is_spot_perp = cfg.mode == "spot_perp"
        venue_b_px = f"{cfg.venue_a}_spot" if is_spot_perp else cfg.venue_b
        lb_h = max(1, int(cfg.lookback_hours))
        vol_lb_h = max(24, min(336, _param_int(cfg.params, "vol_lookback_hours", 168)))
        _, default_slot_margin_per_leg, _ = compute_carry_slot_sizing(cfg, cfg.leverage)

        latest_mids: dict[tuple[str, str], float] = state.setdefault("latest_mids", {})
        latest_hi_lo: dict[tuple[str, str], tuple[float, float]] = state.setdefault("latest_hi_lo", {})
        funding_hist: dict[tuple[str, str], dict[int, float]] = state.setdefault("funding_hist", {})
        first_funding_ts: dict[str, int] = state.setdefault("first_funding_ts", {})
        hourly_log_rets: dict[str, dict[int, float]] = state.setdefault("hourly_log_rets", {})
        cooldown_until: dict[str, int] = state.setdefault("cooldown_until", {})
        open_pos: dict[str, dict[str, Any]] = state.setdefault("open_pos", {})

        defer_decision = bool(event.payload.get("defer_decision", False))

        if event.kind == "BBO":
            bid = float(event.payload.get("bid", 0.0))
            ask = float(event.payload.get("ask", 0.0))
            if bid > 0.0 and ask >= bid and math.isfinite(bid) and math.isfinite(ask):
                mid = (bid + ask) * 0.5
                latest_mids[(event.venue, sym)] = mid
                latest_hi_lo[(event.venue, sym)] = (ask, bid)
        elif event.kind == "CANDLE":
            op = float(event.payload.get("open", 0.0))
            hi = float(event.payload.get("high", 0.0))
            lo = float(event.payload.get("low", 0.0))
            cl = float(event.payload.get("close", 0.0))
            decision_ts = int(event.payload.get("end_ts_ms", ts_ms))
            if cl > 0.0 and hi >= lo > 0.0 and math.isfinite(cl):
                latest_mids[(event.venue, sym)] = cl
                latest_hi_lo[(event.venue, sym)] = (hi, lo)
                if op > 0.0 and math.isfinite(op):
                    rmap = hourly_log_rets.setdefault(sym, {})
                    # Match backtester.py: store under bar-open ts_ms; prefer venue_a, fallback venue_b per bar
                    if event.venue == cfg.venue_a or ts_ms not in rmap:
                        rmap[ts_ms] = math.log(cl / op)
                    cutoff_r = decision_ts - (vol_lb_h + 48) * HOUR_MS
                    if len(rmap) > vol_lb_h + 72:
                        for old_t in [t for t in rmap if int(t) < cutoff_r]:
                            rmap.pop(old_t, None)
        elif event.kind == "FUNDING":
            if "funding_rate" in event.payload:
                rate = float(event.payload["funding_rate"])
            else:
                apr_in = float(event.payload.get("apr_pct", 0.0))
                rate = apr_in / (24.0 * 365.0 * 100.0)
            if math.isfinite(rate):
                fmap = funding_hist.setdefault((event.venue, sym), {})
                fmap[ts_ms] = rate
                prev_first = first_funding_ts.get(sym)
                if prev_first is None or ts_ms < prev_first:
                    first_funding_ts[sym] = ts_ms
                cutoff_f = ts_ms - (lb_h + 24) * HOUR_MS
                if len(fmap) > lb_h + 48:
                    for old_t in [t for t in fmap if int(t) < cutoff_f]:
                        fmap.pop(old_t, None)

                # Track ATB buggy low-settlement streak parity on venue_a funding strictly after entry
                if cfg.use_atb_buggy_rules and event.venue == cfg.venue_a and sym in open_pos:
                    pos_bug = open_pos[sym]
                    if ts_ms > int(pos_bug.get("entry_ts_ms", ts_ms)) and ts_ms > int(pos_bug.get("last_atb_ts", -1)):
                        pos_bug["last_atb_ts"] = ts_ms
                        apr_inst = (
                            (rate * 24.0 * 365.0 * 100.0)
                            if cfg.venue_a in ("hyperliquid", "lighter")
                            else (rate * 3.0 * 365.0 * 100.0)
                        )
                        if abs(apr_inst) < 54.75:
                            pos_bug["low_streak_settlements"] = int(pos_bug.get("low_streak_settlements", 0)) + 1
                        else:
                            pos_bug["low_streak_settlements"] = 0
        else:
            return intents

        if defer_decision:
            return intents

        px_a = latest_mids.get((cfg.venue_a, sym))
        px_b = latest_mids.get((venue_b_px, sym))

        # 1) Margin guard (liquidation >=0.90 > protective exit > 0.65 rebalance) + Basis Stop
        if sym in open_pos and decision_ts > int(open_pos[sym].get("entry_ts_ms", 0)):
            pos = open_pos[sym]
            pos_slot_margin = float(pos.get("slot_margin_per_leg", default_slot_margin_per_leg))
            anc_a = pos.get("anc_a", pos.get("entry_px_a"))
            anc_b = pos.get("anc_b", pos.get("entry_px_b"))
            hl_a = latest_hi_lo.get((cfg.venue_a, sym))
            hl_b = latest_hi_lo.get((venue_b_px, sym)) if not is_spot_perp else None
            eff_hl_a = hl_a or hl_b
            eff_anc_a = anc_a if (hl_a and anc_a) else anc_b
            eff_hl_b = hl_b or hl_a
            eff_anc_b = anc_b if (hl_b and anc_b) else anc_a

            guard_event, margin_used = evaluate_carry_margin_guard(
                cfg,
                direction=pos["direction"],
                notional_usd=float(pos["notional_usd"]),
                slot_margin_per_leg=pos_slot_margin,
                anc_a=eff_anc_a,
                anc_b=eff_anc_b,
                hi_a=eff_hl_a[0] if eff_hl_a else None,
                lo_a=eff_hl_a[1] if eff_hl_a else None,
                hi_b=eff_hl_b[0] if eff_hl_b else None,
                lo_b=eff_hl_b[1] if eff_hl_b else None,
            )
            if guard_event in ("MARGIN_LIQUIDATION", "MARGIN_PROTECT_EXIT"):
                intents.append(
                    OrderIntent(
                        cfg.strategy_id,
                        sym,
                        "CLOSE",
                        float(pos["notional_usd"]),
                        round(margin_used * 100.0, 2),
                        guard_event,
                        decision_ts,
                    )
                )
                open_pos.pop(sym, None)
                cooldown_until[sym] = decision_ts + 24 * HOUR_MS
                return intents
            elif guard_event == "FORCED_REBALANCE":
                pos["anc_a"] = px_a or anc_a
                pos["anc_b"] = px_b or anc_b

            if px_a and px_b:
                _, basis_bps = _basis_pnl_usd(
                    pos["direction"], float(pos["notional_usd"]), pos.get("entry_px_a"), pos.get("entry_px_b"), px_a, px_b
                )
                if basis_bps <= -float(cfg.stop_loss_basis_bps):
                    intents.append(
                        OrderIntent(cfg.strategy_id, sym, "CLOSE", float(pos["notional_usd"]), basis_bps, "BASIS_STOP", decision_ts)
                    )
                    open_pos.pop(sym, None)
                    cooldown_until[sym] = decision_ts + 24 * HOUR_MS
                    return intents

        # 2) Causal rolling funding APR (shared kernel formula)
        apr_a = compute_rolling_funding_apr(funding_hist.get((cfg.venue_a, sym), {}), decision_ts, lb_h)
        apr_b = 0.0 if is_spot_perp else compute_rolling_funding_apr(funding_hist.get((cfg.venue_b, sym), {}), decision_ts, lb_h)

        # 3) Hysteresis / ATB buggy exit check on FUNDING events strictly after entry & non-regressing ts
        if event.kind == "FUNDING" and sym in open_pos and decision_ts > int(open_pos[sym].get("entry_ts_ms", decision_ts)):
            pos = open_pos[sym]
            if cfg.use_atb_buggy_rules:
                if int(pos.get("low_streak_settlements", 0)) >= 3:
                    intents.append(
                        OrderIntent(cfg.strategy_id, sym, "CLOSE", float(pos["notional_usd"]), 0.0, "ATB_BUGGY_LOW_STREAK", decision_ts)
                    )
                    open_pos.pop(sym, None)
                    return intents
            elif apr_a is not None and apr_b is not None:
                last_hyst_ts = int(pos.get("last_hyst_ts", -1))
                if decision_ts >= last_hyst_ts:
                    held_h = (decision_ts - int(pos.get("entry_ts_ms", decision_ts))) / float(HOUR_MS)
                    signed_spread = (
                        (apr_a - apr_b)
                        if pos["direction"] in ("SHORT_A_LONG_B", "SHORT_A_LONG_SPOT")
                        else (apr_b - apr_a)
                    )
                    if decision_ts > last_hyst_ts:
                        pos["low_streak_prev"] = int(pos.get("low_streak_hours", 0))
                        pos["last_hyst_ts"] = decision_ts
                    if signed_spread < float(cfg.exit_spread_apr_pct):
                        pos["low_streak_hours"] = int(pos.get("low_streak_prev", 0)) + 1
                    else:
                        pos["low_streak_hours"] = 0

                    if held_h >= float(cfg.min_hold_hours) and int(pos["low_streak_hours"]) >= int(cfg.exit_confirm_hours):
                        intents.append(
                            OrderIntent(cfg.strategy_id, sym, "CLOSE", float(pos["notional_usd"]), signed_spread, "HYSTERESIS_EXIT", decision_ts)
                        )
                        open_pos.pop(sym, None)
                        return intents

        # 4) Entry check: requires BOTH valid prices, warmup readiness, cooldown clear, and slot availability
        first_ts = first_funding_ts.get(sym)
        warmup_ready = (lb_h <= 1) or (first_ts is not None and (decision_ts - first_ts) >= lb_h * HOUR_MS)
        if (
            warmup_ready
            and decision_ts >= int(cooldown_until.get(sym, 0))
            and px_a is not None
            and px_a > 0.0
            and (is_spot_perp or (px_b is not None and px_b > 0.0))
            and sym not in open_pos
            and len(open_pos) < int(cfg.max_positions)
        ):
            sig = evaluate_carry_entry_signal(cfg, apr_a, apr_b, px_a, px_b)
            if sig is not None:
                direction, action, spread_abs = sig
                eff_lev = compute_causal_symbol_leverage(cfg, hourly_log_rets.get(sym, {}), decision_ts)
                _, pos_slot_margin, pos_notional = compute_carry_slot_sizing(cfg, eff_lev)
                open_pos[sym] = {
                    "direction": direction,
                    "notional_usd": pos_notional,
                    "slot_margin_per_leg": pos_slot_margin,
                    "eff_leverage": eff_lev,
                    "entry_px_a": px_a,
                    "entry_px_b": px_b,
                    "anc_a": px_a,
                    "anc_b": px_b,
                    "entry_ts_ms": decision_ts,
                    "low_streak_hours": 0,
                    "low_streak_prev": 0,
                    "low_streak_settlements": 0,
                    "last_hyst_ts": decision_ts,
                    "last_atb_ts": decision_ts,
                }
                intents.append(
                    OrderIntent(cfg.strategy_id, sym, action, pos_notional, spread_abs, "FUNDING_SPREAD_ENTRY", decision_ts)
                )
        return intents

    def min_economic_kill_horizon_hours(self, cfg: StrategyConfig) -> float:
        return max(48.0, float(cfg.min_hold_hours))
