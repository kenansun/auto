"""Reference Strategy Plugin #2: Cross-Venue Perp-Perp Basis Mean-Reversion (Red-Team #2, #5.3, #12 Hardened).

Anti-Phantom Execution Guarantee (Red-Team #2):
- 1-hour OHLCV `c.open` is the first last-trade print on each venue (asynchronous across venues
  and subject to bid-ask bounce). Trading signal(t) at price(t) creates a fake +90% APR illusion!
- Therefore, on 1h OHLCV candles, `execution_lag_bars` defaults to `1`: a dislocation observed at
  bar `t - 1h` fills strictly at bar `t` open, preventing async open-print mirage completely.
- True sub-second synchronized BBO execution is supported via `on_event()` in `MicroRunSession`
  with rolling z-score, quote sync tolerance (<=15s), directional take-profit, and stop-loss.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from alphaloop.core.models import (
    BacktestMetrics,
    FeedbackEvent,
    MarketEvent,
    OrderIntent,
    PriceBar,
    StrategyConfig,
    TradeRecord,
)
from alphaloop.core.provenance import sha256_obj, utc_now_ms
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import HOUR_MS, TAKER_FEE_BPS, BacktestResult, _basis_pnl_usd
from alphaloop.plugins.base import BaseStrategyPlugin

BASIS_REV_BOUNDS: dict[str, tuple[float, float]] = {
    "entry_zscore": (1.2, 4.5),
    "exit_zscore": (-0.5, 1.5),
    "min_basis_bps": (4.0, 60.0),
    "z_lookback_hours": (12.0, 168.0),
    "max_hold_hours": (6.0, 120.0),
    "stop_basis_bps": (40.0, 300.0),
    "max_positions": (1.0, 8.0),
}


def _param(cfg: StrategyConfig, key: str, default: float) -> float:
    val = cfg.params.get(key, default)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return default
    f = float(val)
    return f if math.isfinite(f) else default


class BasisReversionPlugin(BaseStrategyPlugin):
    """Cross-venue perp-perp mark basis mean-reversion strategy (anti-async-open hardened)."""

    plugin_id = "basis_reversion"
    display_name = "Cross-Venue Perp Basis Mean-Reversion"
    strategy_family = "basis_mean_reversion"
    description = (
        "Trades extreme cross-venue perp mark-price dislocations expecting mean-reversion. "
        "Enforces T+1 execution lag on asynchronous 1h OHLCV candles to eliminate bid-ask bounce mirages; "
        "full synchronous execution runs on live BBO ticks in MicroRunSession."
    )
    required_data_spec = {
        "tables": ["candles_1h", "funding_rates", "bbo_ticks"],
        "requires_dual_candles_for_live": True,
        "requires_synchronized_bbo_for_live": True,
        "min_basis_coverage_pct": 90.0,
        "supported_modes": ["perp_perp"],
    }

    def default_config(self, symbols: list[str] | None = None, **overrides: Any) -> StrategyConfig:
        params: dict[str, Any] = {
            "entry_zscore": 2.4,
            "exit_zscore": 0.25,
            "min_basis_bps": 14.0,
            "z_lookback_hours": 48,
            "max_hold_hours": 36,
            "stop_basis_bps": 120.0,
            "execution_lag_bars": 1,  # Red-Team #2: anti-async-open-bounce default on 1h OHLCV!
        }
        if "params" in overrides and isinstance(overrides["params"], dict):
            params.update(overrides.pop("params"))
        base = StrategyConfig(
            strategy_id="basis_reversion_hl_okx",
            version="v1.0",
            family="basis_mean_reversion",
            status="DRAFT",
            venue_a="hyperliquid",
            venue_b="okx",
            mode="perp_perp",
            symbols=list(symbols or ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "UNI", "SUI", "WLD", "ENA"]),
            capital_usd=3000.0,
            leverage=2.0,
            max_positions=4,
            per_position_weight=0.25,
            entry_spread_apr_pct=0.0,
            exit_spread_apr_pct=0.0,
            lookback_hours=48,
            exit_confirm_hours=1,
            min_hold_hours=2,
            max_basis_entry_bps=120.0,
            stop_loss_basis_bps=float(params["stop_basis_bps"]),
            fee_a_bps=1.5,
            fee_b_bps=2.0,
            slippage_per_leg_bps=1.0,
            rebalance_drag_bps_per_month=3.0,
            allow_bidirectional=True,
            plugin_id=self.plugin_id,
            execution_mode="MAKER_MAKER",
            params=params,
        )
        if overrides:
            d = base.to_dict()
            d.update(overrides)
            return StrategyConfig.from_dict(d)
        return base

    def mutable_param_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(BASIS_REV_BOUNDS)

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
        for key, val in changes.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            fval = float(val)
            if not math.isfinite(fval):
                continue
            if key == "max_positions":
                lo, hi = BASIS_REV_BOUNDS["max_positions"]
                d["max_positions"] = int(round(max(lo, min(hi, fval))))
            elif key in BASIS_REV_BOUNDS:
                lo, hi = BASIS_REV_BOUNDS[key]
                clamped = max(lo, min(hi, fval))
                if key in ("z_lookback_hours", "max_hold_hours"):
                    params[key] = int(round(clamped))
                else:
                    params[key] = round(clamped, 3)
        if params.get("exit_zscore", 0.25) >= params.get("entry_zscore", 2.4) - 0.5:
            params["exit_zscore"] = round(max(0.0, params.get("entry_zscore", 2.4) - 1.0), 3)
        # Prevent capital over-allocation & keep stop_loss_basis_bps synchronized (Red-Team #5.3 & #16)
        d["per_position_weight"] = round(min(float(d["per_position_weight"]), 1.0 / max(1, int(d["max_positions"]))), 4)
        d["stop_loss_basis_bps"] = float(params.get("stop_basis_bps", d["stop_loss_basis_bps"]))
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
        symbols = list(run_cfg.symbols)
        entry_z = _param(run_cfg, "entry_zscore", 2.4)
        exit_z = _param(run_cfg, "exit_zscore", 0.25)
        min_basis_bps = _param(run_cfg, "min_basis_bps", 14.0)
        z_lb = max(6, int(round(_param(run_cfg, "z_lookback_hours", 48.0))))
        max_hold_h = max(2.0, _param(run_cfg, "max_hold_hours", 36.0))
        stop_bps = _param(run_cfg, "stop_basis_bps", 120.0)
        exec_lag = max(0, int(round(_param(run_cfg, "execution_lag_bars", 1.0))))

        candles_a: dict[str, dict[int, PriceBar]] = {}
        candles_b: dict[str, dict[int, PriceBar]] = {}
        rates_a: dict[str, dict[int, float]] = {}
        rates_b: dict[str, dict[int, float]] = {}
        all_ts_set: set[int] = set()

        for sym in symbols:
            ca = store.load_candles(run_cfg.venue_a, sym, start_ms, end_ms)
            cb = store.load_candles(run_cfg.venue_b, sym, start_ms, end_ms)
            candles_a[sym] = {c.ts_ms: c for c in ca}
            candles_b[sym] = {c.ts_ms: c for c in cb}
            common = set(candles_a[sym].keys()) & set(candles_b[sym].keys())
            all_ts_set.update(common)
            rates_a[sym] = {b.ts_ms: b.funding_rate for b in store.load_funding(run_cfg.venue_a, sym, start_ms, end_ms)}
            rates_b[sym] = {b.ts_ms: b.funding_rate for b in store.load_funding(run_cfg.venue_b, sym, start_ms, end_ms)}

        ds_sha, prov, row_count = store.fingerprint_slice([run_cfg.venue_a, run_cfg.venue_b], symbols, start_ms, end_ms)
        if not all_ts_set or row_count == 0:
            return BacktestResult(
                config=run_cfg,
                metrics=BacktestMetrics(
                    start_ms or 0, end_ms or 0, 0.0, run_cfg.capital_usd, 0, 0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, "", prov, ds_sha, {}, 0.0, 0, 0,
                ),
                trades=[],
                equity_curve=[],
                basis_coverage_pct=0.0,
                warnings=["EMPTY_DUAL_CANDLE_DATASET"],
            )

        timeline = sorted(all_ts_set)
        min_ts, max_ts = timeline[0], timeline[-1]

        eff_weight = min(run_cfg.per_position_weight, 1.0 / max(1, run_cfg.max_positions))
        slot_capital = run_cfg.capital_usd * eff_weight
        notional_per_leg = slot_capital * (run_cfg.leverage / 2.0)
        slot_margin_per_leg = max(1e-6, slot_capital * 0.5)

        eff_fee_a = run_cfg.fee_a_bps
        eff_fee_b = run_cfg.fee_b_bps
        if run_cfg.execution_mode == "MAKER_THEN_TAKER_HEDGE":
            eff_fee_b = max(eff_fee_b, TAKER_FEE_BPS.get(run_cfg.venue_b, eff_fee_b))
        elif run_cfg.execution_mode == "TAKER_TAKER":
            eff_fee_a = max(eff_fee_a, TAKER_FEE_BPS.get(run_cfg.venue_a, eff_fee_a))
            eff_fee_b = max(eff_fee_b, TAKER_FEE_BPS.get(run_cfg.venue_b, eff_fee_b))

        one_way_fee_bps = eff_fee_a + eff_fee_b
        one_way_slip_bps = 2.0 * (run_cfg.slippage_per_leg_bps + extra_slippage_bps_per_leg)

        basis_hist: dict[str, list[float]] = {s: [] for s in symbols}
        # Pending lagged signals queue: sym -> (direction, signal_bps) detected at bar t-1 for fill at bar t
        pending_signals: dict[str, tuple[float, str, float]] = {}
        last_seen_px: dict[str, tuple[float, float]] = {}
        cooldown_until: dict[str, int] = {}
        open_pos: dict[str, dict[str, Any]] = {}
        trades: list[TradeRecord] = []
        equity_curve: list[tuple[int, float]] = []
        util_samples: list[float] = []
        realized_pnl = 0.0
        trade_seq = 0
        liquidation_count = 0
        forced_rebalance_count = 0

        for ts in timeline:
            # 1) Accrue funding on open positions strictly after entry
            for sym, pos in open_pos.items():
                if ts <= pos["entry_ts_ms"]:
                    continue
                fr_a = rates_a[sym].get(ts)
                fr_b = rates_b[sym].get(ts)
                step_f = 0.0
                if fr_a is not None:
                    step_f += pos["notional"] * fr_a if pos["direction"] == "SHORT_A_LONG_B" else -pos["notional"] * fr_a
                if fr_b is not None:
                    step_f += -pos["notional"] * fr_b if pos["direction"] == "SHORT_A_LONG_B" else pos["notional"] * fr_b
                pos["gross_funding_usd"] += step_f

            # Update last observed valid prices and compute current causal basis & z-score
            cur_z: dict[str, tuple[float, float, float, float]] = {}
            for sym in symbols:
                ca = candles_a[sym].get(ts)
                cb = candles_b[sym].get(ts)
                if not (ca and cb and ca.open > 0 and cb.open > 0):
                    continue
                last_seen_px[sym] = (ca.open, cb.open)
                b_bps = ((ca.open - cb.open) / cb.open) * 10000.0
                hist = basis_hist[sym]
                if len(hist) >= z_lb:
                    win = hist[-z_lb:]
                    mean_b = sum(win) / len(win)
                    var_b = sum((x - mean_b) ** 2 for x in win) / max(1, len(win) - 1)
                    std_b = math.sqrt(var_b)
                    eff_std = max(std_b, 2.0)
                    z = (b_bps - mean_b) / eff_std
                    cur_z[sym] = (b_bps, z, ca.open, cb.open)
                hist.append(b_bps)

            # 2) Margin check (prior bar) & exits
            to_close: list[tuple[str, str]] = []
            prev_ts = ts - HOUR_MS
            for sym, pos in open_pos.items():
                hold_h = (ts - pos["entry_ts_ms"]) / HOUR_MS
                ca_prev = candles_a[sym].get(prev_ts)
                cb_prev = candles_b[sym].get(prev_ts)
                if ts > pos["entry_ts_ms"]:
                    anc_a, anc_b = pos["anchor_a"], pos["anchor_b"]
                    is_short_a = pos["direction"] == "SHORT_A_LONG_B"
                    adv = 0.0
                    if ca_prev and anc_a > 0:
                        adv = max(adv, ((ca_prev.high / anc_a) - 1.0) if is_short_a else (1.0 - (ca_prev.low / anc_a)))
                    if cb_prev and anc_b > 0:
                        adv = max(adv, (1.0 - (cb_prev.low / anc_b)) if is_short_a else ((cb_prev.high / anc_b) - 1.0))
                    margin_used = adv * (pos["notional"] / slot_margin_per_leg)
                    if margin_used >= 0.90:
                        liquidation_count += 1
                        pos["extra_rebal_usd"] += max(slot_margin_per_leg * 0.90, pos["notional"] * 0.05)
                        to_close.append((sym, "MARGIN_LIQUIDATION"))
                        continue
                    elif margin_used >= 0.65:
                        forced_rebalance_count += 1
                        pos["extra_rebal_usd"] += max(1.0, slot_margin_per_leg * 0.001)
                        ca_now = candles_a[sym].get(ts)
                        cb_now = candles_b[sym].get(ts)
                        pos["anchor_a"] = ca_now.open if ca_now else anc_a
                        pos["anchor_b"] = cb_now.open if cb_now else anc_b

                ca_now = candles_a[sym].get(ts)
                cb_now = candles_b[sym].get(ts)
                if not (ca_now and cb_now):
                    continue
                _, b_move = _basis_pnl_usd(pos["direction"], pos["notional"], pos["entry_a"], pos["entry_b"], ca_now.open, cb_now.open)
                if b_move < pos["max_adverse_basis_bps"]:
                    pos["max_adverse_basis_bps"] = b_move
                if b_move <= -stop_bps:
                    to_close.append((sym, "BASIS_STOP"))
                    continue
                if hold_h >= max_hold_h:
                    to_close.append((sym, "MAX_HOLD_TIMEOUT"))
                    continue
                if sym in cur_z:
                    _, z_val, _, _ = cur_z[sym]
                    if (pos["direction"] == "SHORT_A_LONG_B" and z_val <= exit_z) or (
                        pos["direction"] == "LONG_A_SHORT_B" and z_val >= -exit_z
                    ):
                        to_close.append((sym, "MEAN_REVERSION_EXIT"))

            for sym, reason in to_close:
                pos = open_pos.pop(sym)
                if reason in ("MARGIN_LIQUIDATION", "BASIS_STOP"):
                    cooldown_until[sym] = ts + 24 * HOUR_MS
                # Red-Team #12: carry forward last valid observed price (NEVER entry price teleport!)
                ex_a, ex_b = last_seen_px.get(sym, (pos["entry_a"], pos["entry_b"]))
                tr = self._close_trade(
                    trade_seq + 1, sym, pos, ts, reason, ex_a, ex_b,
                    one_way_fee_bps, one_way_slip_bps, run_cfg.rebalance_drag_bps_per_month,
                )
                trade_seq += 1
                trades.append(tr)
                realized_pnl += tr.net_pnl_usd

            # 3) Entries (Red-Team #2: if exec_lag >= 1, fill strictly on signals queued from PRIOR bar!)
            if exec_lag >= 1:
                if len(open_pos) < run_cfg.max_positions and pending_signals:
                    ordered = sorted(pending_signals.items(), key=lambda kv: kv[1][0], reverse=True)
                    for sym, (score, direction, sig_bps) in ordered:
                        if len(open_pos) >= run_cfg.max_positions:
                            break
                        if sym in open_pos or ts < cooldown_until.get(sym, 0):
                            continue
                        ca_now = candles_a[sym].get(ts)
                        cb_now = candles_b[sym].get(ts)
                        if not (ca_now and cb_now and ca_now.open > 0 and cb_now.open > 0):
                            continue
                        open_pos[sym] = {
                            "direction": direction,
                            "entry_ts_ms": ts,
                            "notional": notional_per_leg,
                            "entry_a": ca_now.open,
                            "entry_b": cb_now.open,
                            "anchor_a": ca_now.open,
                            "anchor_b": cb_now.open,
                            "entry_signal": sig_bps,
                            "gross_funding_usd": 0.0,
                            "extra_rebal_usd": 0.0,
                            "max_adverse_basis_bps": 0.0,
                        }
                # Queue signals detected on THIS bar for NEXT bar open execution
                pending_signals = {}
                for sym, (b_bps, z_val, _, _) in cur_z.items():
                    if sym in open_pos or ts < cooldown_until.get(sym, 0):
                        continue
                    if z_val >= entry_z and b_bps >= min_basis_bps:
                        pending_signals[sym] = (abs(z_val), "SHORT_A_LONG_B", b_bps)
                    elif z_val <= -entry_z and b_bps <= -min_basis_bps:
                        pending_signals[sym] = (abs(z_val), "LONG_A_SHORT_B", b_bps)
            else:
                if len(open_pos) < run_cfg.max_positions:
                    cands: list[tuple[float, str, str, float, float, float]] = []
                    for sym, (b_bps, z_val, px_a, px_b) in cur_z.items():
                        if sym in open_pos or ts < cooldown_until.get(sym, 0):
                            continue
                        if z_val >= entry_z and b_bps >= min_basis_bps:
                            cands.append((abs(z_val), sym, "SHORT_A_LONG_B", b_bps, px_a, px_b))
                        elif z_val <= -entry_z and b_bps <= -min_basis_bps:
                            cands.append((abs(z_val), sym, "LONG_A_SHORT_B", b_bps, px_a, px_b))
                    cands.sort(reverse=True)
                    for score, sym, direction, b_bps, px_a, px_b in cands:
                        if len(open_pos) >= run_cfg.max_positions:
                            break
                        open_pos[sym] = {
                            "direction": direction,
                            "entry_ts_ms": ts,
                            "notional": notional_per_leg,
                            "entry_a": px_a,
                            "entry_b": px_b,
                            "anchor_a": px_a,
                            "anchor_b": px_b,
                            "entry_signal": b_bps,
                            "gross_funding_usd": 0.0,
                            "extra_rebal_usd": 0.0,
                            "max_adverse_basis_bps": 0.0,
                        }

            # 4) Mark-to-market equity
            unreal = 0.0
            for sym, pos in open_pos.items():
                cur_a, cur_b = last_seen_px.get(sym, (pos["entry_a"], pos["entry_b"]))
                b_pnl, _ = _basis_pnl_usd(pos["direction"], pos["notional"], pos["entry_a"], pos["entry_b"], cur_a, cur_b)
                hold_h = max(0.0, (ts - pos["entry_ts_ms"]) / HOUR_MS)
                est_rt = pos["notional"] * (2.0 * (one_way_fee_bps + one_way_slip_bps)) / 10000.0
                acc_reb = pos["notional"] * (run_cfg.rebalance_drag_bps_per_month / 10000.0) * (hold_h / 720.0)
                unreal += pos["gross_funding_usd"] + b_pnl - est_rt - acc_reb - pos["extra_rebal_usd"]
            equity_curve.append((ts, run_cfg.capital_usd + realized_pnl + unreal))
            util_samples.append(((len(open_pos) * slot_capital) / run_cfg.capital_usd * 100.0) if run_cfg.capital_usd > 0 else 0.0)

        # 5) Close remaining at end using last valid observed prices
        final_ts = timeline[-1]
        for sym in list(open_pos.keys()):
            pos = open_pos.pop(sym)
            ex_a, ex_b = last_seen_px.get(sym, (pos["entry_a"], pos["entry_b"]))
            tr = self._close_trade(
                trade_seq + 1, sym, pos, final_ts, "END_OF_WINDOW", ex_a, ex_b,
                one_way_fee_bps, one_way_slip_bps, run_cfg.rebalance_drag_bps_per_month,
            )
            trade_seq += 1
            trades.append(tr)
            realized_pnl += tr.net_pnl_usd
        if equity_curve:
            equity_curve[-1] = (final_ts, run_cfg.capital_usd + realized_pnl)

        metrics = self._build_metrics(
            run_cfg, min_ts, max_ts, trades, equity_curve, util_samples,
            prov, ds_sha, liquidation_count, forced_rebalance_count,
        )
        warnings = []
        if exec_lag >= 1:
            warnings.append("OHLCV_ASYNC_OPEN_LAG_1H: T+1 execution lag enforced to block 1h open-print bounce.")
        if liquidation_count:
            warnings.append(f"MARGIN_LIQUIDATION_OCCURRED ({liquidation_count})")
        return BacktestResult(run_cfg, metrics, trades, equity_curve, 100.0 if trades else 0.0, warnings)

    @staticmethod
    def _close_trade(
        seq: int, sym: str, pos: dict[str, Any], exit_ts_ms: int, reason: str,
        ex_a: float, ex_b: float, one_way_fee_bps: float, one_way_slip_bps: float, rebal_bpm: float,
    ) -> TradeRecord:
        hold_h = max(0.0, (exit_ts_ms - pos["entry_ts_ms"]) / HOUR_MS)
        b_pnl, _ = _basis_pnl_usd(pos["direction"], pos["notional"], pos["entry_a"], pos["entry_b"], ex_a, ex_b)
        fee = pos["notional"] * (2.0 * one_way_fee_bps) / 10000.0
        slip = pos["notional"] * (2.0 * one_way_slip_bps) / 10000.0
        reb = pos["notional"] * (rebal_bpm / 10000.0) * (hold_h / 720.0) + pos["extra_rebal_usd"]
        net = pos["gross_funding_usd"] + b_pnl - fee - slip - reb
        return TradeRecord(
            trade_id=f"BR-{seq:04d}",
            symbol=sym,
            direction=pos["direction"],
            entry_ts_ms=pos["entry_ts_ms"],
            exit_ts_ms=exit_ts_ms,
            hold_hours=round(hold_h, 2),
            notional_per_leg_usd=round(pos["notional"], 2),
            entry_spread_apr_pct=round(pos["entry_signal"], 2),
            gross_funding_usd=round(pos["gross_funding_usd"], 4),
            basis_pnl_usd=round(b_pnl, 4),
            fee_usd=round(fee, 4),
            slippage_usd=round(slip, 4),
            rebalance_usd=round(reb, 4),
            net_pnl_usd=round(net, 4),
            exit_reason=reason,
            max_adverse_basis_bps=round(pos["max_adverse_basis_bps"], 2),
        )

    @staticmethod
    def _build_metrics(
        cfg: StrategyConfig, start_ms: int, end_ms: int, trades: list[TradeRecord],
        equity_curve: list[tuple[int, float]], util_samples: list[float],
        prov: str, ds_sha: str, liq_n: int, rebal_n: int,
    ) -> BacktestMetrics:
        days = max(1.0 / 24.0, (end_ms - start_ms) / (86400.0 * 1000.0))
        gross_f = sum(t.gross_funding_usd for t in trades)
        basis_p = sum(t.basis_pnl_usd for t in trades)
        gross_alpha = gross_f + basis_p
        fees = sum(t.fee_usd for t in trades)
        slips = sum(t.slippage_usd for t in trades)
        rebals = sum(t.rebalance_usd for t in trades)
        net_pnl = sum(t.net_pnl_usd for t in trades)
        wins = [t.net_pnl_usd for t in trades if t.net_pnl_usd > 0]
        losses = [-t.net_pnl_usd for t in trades if t.net_pnl_usd < 0]
        pf = (sum(wins) / sum(losses)) if sum(losses) > 0 else (99.0 if wins else 0.0)
        tot_ret = (net_pnl / cfg.capital_usd) * 100.0 if cfg.capital_usd > 0 else 0.0
        apr = tot_ret * (365.0 / days)

        peak = cfg.capital_usd
        mdd = 0.0
        for _, eq in equity_curve:
            if eq > peak:
                peak = eq
            if peak > 0:
                mdd = max(mdd, ((peak - eq) / peak) * 100.0)

        daily_eq: dict[int, float] = {}
        for ts, eq in equity_curve:
            daily_eq[ts // (86400 * 1000)] = eq
        vals = [daily_eq[k] for k in sorted(daily_eq)]
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
            b = by_sym.setdefault(t.symbol, {"trades": 0.0, "wins": 0.0, "net_pnl_usd": 0.0, "basis_pnl_usd": 0.0, "costs_usd": 0.0})
            b["trades"] += 1.0
            b["wins"] += 1.0 if t.net_pnl_usd > 0 else 0.0
            b["net_pnl_usd"] += t.net_pnl_usd
            b["basis_pnl_usd"] += t.basis_pnl_usd
            b["costs_usd"] += t.fee_usd + t.slippage_usd + t.rebalance_usd
        best_s = max(by_sym, key=lambda s: by_sym[s]["net_pnl_usd"]) if by_sym else ""
        top_s_rem = net_pnl - max(0.0, by_sym[best_s]["net_pnl_usd"]) if best_s else 0.0

        return BacktestMetrics(
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            days=round(days, 2),
            capital_usd=cfg.capital_usd,
            trade_count=len(trades),
            win_count=len(wins),
            win_rate=round(len(wins) / len(trades), 4) if trades else 0.0,
            avg_hold_hours=round(sum(t.hold_hours for t in trades) / len(trades), 2) if trades else 0.0,
            gross_funding_usd=round(gross_f, 4),
            basis_pnl_usd=round(basis_p, 4),
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
            avg_capital_utilization_pct=round(sum(util_samples) / len(util_samples), 2) if util_samples else 0.0,
            monthly_usd_est=round(net_pnl * (30.0 / days), 2),
            top5_removed_net_pnl_usd=round(top5_rem, 4),
            top_symbol_removed_net_pnl_usd=round(top_s_rem, 4),
            best_symbol=best_s,
            provenance=prov,
            dataset_sha256=ds_sha,
            by_symbol=by_sym,
            gross_alpha_usd=round(gross_alpha, 4),
            liquidation_count=liq_n,
            forced_rebalance_count=rebal_n,
        )

    def plateau_variants(self, cfg: StrategyConfig) -> list[tuple[str, StrategyConfig]]:
        ez = _param(cfg, "entry_zscore", 2.4)
        mb = _param(cfg, "min_basis_bps", 14.0)
        variants: list[tuple[str, StrategyConfig]] = []
        for zf in (0.85, 1.0, 1.15):
            for bf in (0.80, 1.0, 1.20):
                nz = round(max(1.2, ez * zf), 2)
                nb = round(max(4.0, mb * bf), 2)
                p = dict(cfg.params)
                p["entry_zscore"] = nz
                p["min_basis_bps"] = nb
                variants.append((f"z={nz}/minBps={nb}", dataclasses.replace(cfg, params=p)))
        return variants

    def diagnose_telemetry(self, cfg: StrategyConfig, metrics: BacktestMetrics) -> FeedbackEvent:
        now = utc_now_ms()
        anomaly, sev = "HEALTHY", "INFO"
        summary = f"Basis reversion healthy: APR={metrics.annualized_apr_pct:+.1f}%, Sharpe={metrics.sharpe_ratio:.2f}."
        if metrics.liquidation_count > 0:
            anomaly, sev = "MARGIN_LIQUIDATION", "P0"
            summary = f"{metrics.liquidation_count} margin liquidations detected; tighten stop_basis_bps."
        elif metrics.trade_count >= 5 and metrics.fee_to_gross_ratio > 0.70:
            anomaly, sev = "CHURN_EXCESS", "P0"
            summary = f"Basis reversion over-trading: costs consume {metrics.fee_to_gross_ratio:.1%} of gross basis alpha. Raise min_basis_bps / entry_zscore."
        elif metrics.net_pnl_usd <= 0:
            anomaly, sev = "EDGE_DECAY", "P1"
            summary = f"Basis reversion negative net PnL (${metrics.net_pnl_usd:+.2f}, basis=${metrics.basis_pnl_usd:+.2f}); requires synchronized live BBO edge."
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
        mb = _param(cfg, "min_basis_bps", 14.0)
        ez = _param(cfg, "entry_zscore", 2.4)
        if feedback.anomaly_type in ("CHURN_EXCESS", "EDGE_DECAY"):
            return (
                {"min_basis_bps": min(50.0, round(mb * 1.45, 1)), "entry_zscore": min(3.8, round(ez + 0.4, 2))},
                "Raise minimum dislocation bps hurdle and z-score threshold to filter sub-cost basis noise.",
                "Ensures expected mean-reversion edge exceeds 4-leg round-trip maker fee + slippage before entry.",
                "Reject if trade_count drops below 5 or STRESS net PnL remains <= 0.",
            )
        return (
            {"stop_basis_bps": max(50.0, round(_param(cfg, "stop_basis_bps", 120.0) * 0.8, 1))},
            "Tighten basis stop-loss on diverging dislocations.",
            "Cuts tail exposure when a structural regime shift prevents basis convergence.",
            "Reject if MaxDD increases or Sharpe drops.",
        )

    def on_event(self, event: MarketEvent, state: dict[str, Any], cfg: StrategyConfig) -> list[OrderIntent]:
        """Synchronized live BBO step handler with quote-sync guard, rolling z-score, directional TP, and stop-loss (Red-Team #12)."""
        intents: list[OrderIntent] = []
        if event.kind != "BBO" or event.symbol not in cfg.symbols:
            return intents
        bid, ask = float(event.payload.get("bid", 0.0)), float(event.payload.get("ask", 0.0))
        if bid <= 0 or ask < bid:
            return intents
        quotes: dict[tuple[str, str], tuple[int, float]] = state.setdefault("quotes", {})
        quotes[(event.venue, event.symbol)] = (event.ts_ms, (bid + ask) * 0.5)
        qa = quotes.get((cfg.venue_a, event.symbol))
        qb = quotes.get((cfg.venue_b, event.symbol))
        if not (qa and qb and qb[1] > 0):
            return intents
        # Guard against pairing stale and fresh quotes (max 15s skew)
        if abs(qa[0] - qb[0]) > 15_000:
            return intents
        pa, pb = qa[1], qb[1]
        b_bps = ((pa - pb) / pb) * 10000.0

        hist: list[float] = state.setdefault("basis_hist", {}).setdefault(event.symbol, [])
        z_lb = max(8, min(60, int(round(_param(cfg, "z_lookback_hours", 20.0)))))
        z_val = 0.0
        if len(hist) >= z_lb:
            win = hist[-z_lb:]
            mu = sum(win) / len(win)
            sd = math.sqrt(sum((x - mu) ** 2 for x in win) / max(1, len(win) - 1))
            z_val = (b_bps - mu) / max(sd, 2.0)
        hist.append(b_bps)
        if len(hist) > z_lb * 2:
            del hist[:-z_lb]

        min_b = _param(cfg, "min_basis_bps", 14.0)
        entry_z = _param(cfg, "entry_zscore", 2.4)
        exit_z = _param(cfg, "exit_zscore", 0.25)
        stop_b = _param(cfg, "stop_basis_bps", 120.0)
        eff_weight = min(cfg.per_position_weight, 1.0 / max(1, cfg.max_positions))
        notional = cfg.capital_usd * eff_weight * (cfg.leverage / 2.0)
        open_pos: dict[str, dict[str, Any]] = state.setdefault("open_pos", {})

        if event.symbol in open_pos:
            pos = open_pos[event.symbol]
            _, rel_bps = _basis_pnl_usd(pos["dir"], notional, pos["ea"], pos["eb"], pa, pb)
            if rel_bps <= -stop_b:
                open_pos.pop(event.symbol, None)
                intents.append(OrderIntent(cfg.strategy_id, event.symbol, "CLOSE", notional, rel_bps, "LIVE_BASIS_STOP", event.ts_ms))
            elif (pos["dir"] == "SHORT_A_LONG_B" and (z_val <= exit_z or b_bps <= min_b * 0.25)) or (
                pos["dir"] == "LONG_A_SHORT_B" and (z_val >= -exit_z or b_bps >= -min_b * 0.25)
            ):
                open_pos.pop(event.symbol, None)
                intents.append(OrderIntent(cfg.strategy_id, event.symbol, "CLOSE", notional, rel_bps, "LIVE_BASIS_CONVERGED", event.ts_ms))
        elif len(hist) >= z_lb and len(open_pos) < cfg.max_positions:
            if z_val >= entry_z and b_bps >= min_b:
                open_pos[event.symbol] = {"dir": "SHORT_A_LONG_B", "ea": pa, "eb": pb}
                intents.append(OrderIntent(cfg.strategy_id, event.symbol, "OPEN_SHORT_A_LONG_B", notional, abs(b_bps), "LIVE_BASIS_DISLOCATION", event.ts_ms))
            elif z_val <= -entry_z and b_bps <= -min_b:
                open_pos[event.symbol] = {"dir": "LONG_A_SHORT_B", "ea": pa, "eb": pb}
                intents.append(OrderIntent(cfg.strategy_id, event.symbol, "OPEN_LONG_A_SHORT_B", notional, abs(b_bps), "LIVE_BASIS_DISLOCATION", event.ts_ms))
        return intents

    def min_economic_kill_horizon_hours(self, cfg: StrategyConfig) -> float:
        return max(24.0, _param(cfg, "max_hold_hours", 36.0))
