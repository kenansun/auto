"""Causal hourly event-driven backtester for Cross-Venue Perp-Perp & Spot-Perp Funding Carry.

Zero-lookahead guarantees:
- Decisions at hour `t` only observe funding settlements and candles with `ts_ms <= t`.
- Funding collection for an open position begins strictly at `ts_ms > entry_ts_ms`.
- Full 4-leg fee, slippage, dual-venue real basis PnL, and amortized rebalance accounting.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from alphaloop.core.models import (
    BacktestMetrics,
    FundingRateBar,
    PriceBar,
    StrategyConfig,
    TradeRecord,
)
from alphaloop.data.store import MarketDataStore

logger = logging.getLogger(__name__)

HOUR_MS = 3600 * 1000


@dataclass
class _OpenPosition:
    symbol: str
    direction: str          # SHORT_A_LONG_B | LONG_A_SHORT_B | SHORT_A_LONG_SPOT
    entry_ts_ms: int
    notional_usd: float
    entry_spread_apr_pct: float
    entry_px_a: float | None
    entry_px_b: float | None
    gross_funding_usd: float = 0.0
    low_streak_hours: int = 0
    low_streak_settlements: int = 0
    max_adverse_basis_bps: float = 0.0


@dataclass
class BacktestResult:
    config: StrategyConfig
    metrics: BacktestMetrics
    trades: list[TradeRecord]
    equity_curve: list[tuple[int, float]]   # (ts_ms, equity_usd)
    basis_coverage_pct: float
    warnings: list[str] = field(default_factory=list)


TAKER_FEE_BPS: dict[str, float] = {
    "hyperliquid": 4.5,
    "lighter": 0.0,
    "okx": 5.0,
    "gate": 5.0,
    "binance": 5.0,
}


def _basis_pnl_usd(
    direction: str,
    notional_usd: float,
    entry_a: float | None,
    entry_b: float | None,
    cur_a: float | None,
    cur_b: float | None,
) -> tuple[float, float]:
    """Return (basis_pnl_usd, basis_move_bps) using real aligned dual-venue prices."""
    if not (entry_a and entry_b and cur_a and cur_b and entry_a > 0 and entry_b > 0):
        return 0.0, 0.0
    ret_a = (cur_a / entry_a) - 1.0
    ret_b = (cur_b / entry_b) - 1.0
    if direction in ("SHORT_A_LONG_B", "SHORT_A_LONG_SPOT"):
        # Short A, Long B
        rel = ret_b - ret_a
    else:
        # Long A, Short B
        rel = ret_a - ret_b
    return notional_usd * rel, rel * 10000.0


class FundingCarryBacktester:
    """Deterministic hourly causal backtest engine backed by SQLite MarketDataStore."""

    def __init__(self, store: MarketDataStore) -> None:
        self.store = store

    def run(
        self,
        cfg: StrategyConfig,
        start_ms: int | None = None,
        end_ms: int | None = None,
        extra_slippage_bps_per_leg: float = 0.0,
    ) -> BacktestResult:
        """Execute deterministic causal backtest over [start_ms, end_ms]."""
        symbols = list(cfg.symbols)
        is_spot_perp = cfg.mode == "spot_perp"

        rates_a: dict[str, dict[int, float]] = {}
        rates_b: dict[str, dict[int, float]] = {}
        prices_a: dict[str, dict[int, float]] = {}
        prices_b: dict[str, dict[int, float]] = {}
        candles_a: dict[str, dict[int, PriceBar]] = {}
        candles_b: dict[str, dict[int, PriceBar]] = {}

        all_ts_set: set[int] = set()
        for sym in symbols:
            fa = self.store.load_funding(cfg.venue_a, sym, start_ms, end_ms)
            rates_a[sym] = {b.ts_ms: b.funding_rate for b in fa}
            all_ts_set.update(rates_a[sym].keys())

            if not is_spot_perp:
                fb = self.store.load_funding(cfg.venue_b, sym, start_ms, end_ms)
                rates_b[sym] = {b.ts_ms: b.funding_rate for b in fb}
                all_ts_set.update(rates_b[sym].keys())
            else:
                rates_b[sym] = {}

            # P0-1 ZERO LOOKAHEAD: Exchanges stamp 1h bar [t, t+1h) with OPEN timestamp t.
            # At decision time t, ONLY c.open is observable (c.close is not known until t+1h!).
            ca = self.store.load_candles(cfg.venue_a, sym, start_ms, end_ms)
            prices_a[sym] = {c.ts_ms: c.open for c in ca}
            candles_a[sym] = {c.ts_ms: c for c in ca}

            venue_b_px = f"{cfg.venue_a}_spot" if is_spot_perp else cfg.venue_b
            cb = self.store.load_candles(venue_b_px, sym, start_ms, end_ms)
            prices_b[sym] = {c.ts_ms: c.open for c in cb}
            candles_b[sym] = {c.ts_ms: c for c in cb}

        # Include venue_b_px in fingerprint so spot_perp spot candles are tamper-evident (Red-Team #9)
        venues_for_fp = [cfg.venue_a, f"{cfg.venue_a}_spot"] if is_spot_perp else [cfg.venue_a, cfg.venue_b]
        dataset_sha, prov, row_count = self.store.fingerprint_slice(venues_for_fp, symbols, start_ms, end_ms)

        if not all_ts_set or row_count == 0:
            empty_metrics = BacktestMetrics(
                window_start_ms=start_ms or 0,
                window_end_ms=end_ms or 0,
                days=0.0,
                capital_usd=cfg.capital_usd,
                trade_count=0,
                win_count=0,
                win_rate=0.0,
                avg_hold_hours=0.0,
                gross_funding_usd=0.0,
                basis_pnl_usd=0.0,
                total_fees_usd=0.0,
                total_slippage_usd=0.0,
                total_rebalance_usd=0.0,
                net_pnl_usd=0.0,
                total_return_pct=0.0,
                annualized_apr_pct=0.0,
                sharpe_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown_pct=0.0,
                profit_factor=0.0,
                fee_to_gross_ratio=0.0,
                avg_capital_utilization_pct=0.0,
                monthly_usd_est=0.0,
                top5_removed_net_pnl_usd=0.0,
                top_symbol_removed_net_pnl_usd=0.0,
                best_symbol="",
                provenance=prov,
                dataset_sha256=dataset_sha,
                gross_alpha_usd=0.0,
            )
            return BacktestResult(
                config=cfg,
                metrics=empty_metrics,
                trades=[],
                equity_curve=[],
                basis_coverage_pct=0.0,
                warnings=["EMPTY_DATASET"],
            )

        min_ts = min(all_ts_set)
        max_ts = max(all_ts_set)
        timeline: list[int] = []
        t = min_ts
        while t <= max_ts:
            timeline.append(t)
            t += HOUR_MS

        n_steps = len(timeline)
        prefix_a: dict[str, list[float]] = {}
        hits_a: dict[str, list[int]] = {}
        prefix_b: dict[str, list[float]] = {}
        hits_b: dict[str, list[int]] = {}
        for sym in symbols:
            pa = [0.0] * (n_steps + 1)
            ha = [0] * (n_steps + 1)
            pb = [0.0] * (n_steps + 1)
            hb = [0] * (n_steps + 1)
            ra_map = rates_a[sym]
            rb_map = rates_b[sym]
            for i, ts_i in enumerate(timeline):
                va = ra_map.get(ts_i)
                pa[i + 1] = pa[i] + (va if va is not None else 0.0)
                ha[i + 1] = ha[i] + (1 if va is not None else 0)
                if not is_spot_perp:
                    vb = rb_map.get(ts_i)
                    pb[i + 1] = pb[i] + (vb if vb is not None else 0.0)
                    hb[i + 1] = hb[i] + (1 if vb is not None else 0)
            prefix_a[sym] = pa
            hits_a[sym] = ha
            prefix_b[sym] = pb
            hits_b[sym] = hb

        def fast_rolling_apr(sym: str, step_idx: int, is_b: bool = False) -> float | None:
            lb = cfg.lookback_hours
            lo = max(0, step_idx + 1 - lb)
            hi = step_idx + 1
            if is_b:
                if hits_b[sym][hi] - hits_b[sym][lo] == 0:
                    return None
                total = prefix_b[sym][hi] - prefix_b[sym][lo]
            else:
                if hits_a[sym][hi] - hits_a[sym][lo] == 0:
                    return None
                total = prefix_a[sym][hi] - prefix_a[sym][lo]
            return (total / float(lb)) * 24.0 * 365.0 * 100.0

        # Perp-Perp: slot capital is split 50/50 across Venue A & B margin.
        # Effective per-leg leverage = notional_per_leg / (slot_capital * 0.5) = cfg.leverage.
        eff_weight = min(cfg.per_position_weight, 1.0 / max(1, cfg.max_positions))
        slot_capital = cfg.capital_usd * eff_weight
        if is_spot_perp:
            notional_per_leg = slot_capital * (cfg.leverage / (1.0 + cfg.leverage))
            slot_margin_per_leg = max(1e-6, slot_capital / (1.0 + cfg.leverage))
        else:
            notional_per_leg = slot_capital * (cfg.leverage / 2.0)
            slot_margin_per_leg = max(1e-6, slot_capital * 0.5)

        eff_fee_a = cfg.fee_a_bps
        eff_fee_b = cfg.fee_b_bps
        if cfg.execution_mode == "MAKER_THEN_TAKER_HEDGE":
            eff_fee_b = max(eff_fee_b, TAKER_FEE_BPS.get(cfg.venue_b, eff_fee_b))
        elif cfg.execution_mode == "TAKER_TAKER":
            eff_fee_a = max(eff_fee_a, TAKER_FEE_BPS.get(cfg.venue_a, eff_fee_a))
            eff_fee_b = max(eff_fee_b, TAKER_FEE_BPS.get(cfg.venue_b, eff_fee_b))

        slip_bps = cfg.slippage_per_leg_bps + extra_slippage_bps_per_leg
        one_way_fee_bps = eff_fee_a + eff_fee_b
        one_way_slip_bps = 2.0 * slip_bps

        open_pos: dict[str, _OpenPosition] = {}
        anchor_px: dict[str, tuple[float | None, float | None]] = {}
        extra_rebal_usd: dict[str, float] = {}
        cooldown_until: dict[str, int] = {}
        closed_trades: list[TradeRecord] = []
        realized_pnl_usd = 0.0
        equity_curve: list[tuple[int, float]] = []
        util_samples: list[float] = []
        basis_priced_trades = 0
        trade_seq = 0
        liquidation_count = 0
        forced_rebalance_count = 0
        warnings: list[str] = []

        warmup_end = min_ts + cfg.lookback_hours * HOUR_MS

        for step_idx, ts in enumerate(timeline):
            # 1) Accrue realized funding settlements strictly AFTER entry_ts_ms
            for sym, pos in open_pos.items():
                if ts <= pos.entry_ts_ms:
                    continue
                fr_a = rates_a[sym].get(ts)
                fr_b = 0.0 if is_spot_perp else rates_b[sym].get(ts)
                step_funding = 0.0
                if fr_a is not None:
                    if pos.direction in ("SHORT_A_LONG_B", "SHORT_A_LONG_SPOT"):
                        step_funding += pos.notional_usd * fr_a
                    else:
                        step_funding -= pos.notional_usd * fr_a
                if not is_spot_perp and fr_b is not None:
                    if pos.direction == "SHORT_A_LONG_B":
                        step_funding -= pos.notional_usd * fr_b
                    else:
                        step_funding += pos.notional_usd * fr_b
                pos.gross_funding_usd += step_funding

                if cfg.use_atb_buggy_rules and fr_a is not None:
                    apr_inst = (fr_a * 24.0 * 365.0 * 100.0) if cfg.venue_a in ("hyperliquid", "lighter") else (fr_a * 3.0 * 365.0 * 100.0)
                    if abs(apr_inst) < 54.75:
                        pos.low_streak_settlements += 1
                    else:
                        pos.low_streak_settlements = 0

            # 2) Check per-leg margin safety (P1-1 + Red-Team #3 proxy for missing-candle leg) & exits
            to_close: list[tuple[str, str]] = []
            prev_bar_ts = ts - HOUR_MS
            for sym, pos in open_pos.items():
                hold_h = (ts - pos.entry_ts_ms) / HOUR_MS
                cur_a = prices_a[sym].get(ts)
                cur_b = prices_b[sym].get(ts)

                if ts > pos.entry_ts_ms:
                    bar_a_prev = candles_a[sym].get(prev_bar_ts)
                    bar_b_prev = candles_b[sym].get(prev_bar_ts)
                    anc_a, anc_b = anchor_px.get(sym, (pos.entry_px_a, pos.entry_px_b))
                    # Red-Team #3: When one venue lacks historical candles (e.g. Lighter),
                    # conservatively proxy missing-leg directional move from the present venue's candle!
                    eff_bar_a = bar_a_prev or bar_b_prev
                    eff_anc_a = anc_a if (bar_a_prev and anc_a) else anc_b
                    eff_bar_b = bar_b_prev or bar_a_prev
                    eff_anc_b = anc_b if (bar_b_prev and anc_b) else anc_a

                    adverse_leg_move = 0.0
                    is_short_a = pos.direction in ("SHORT_A_LONG_B", "SHORT_A_LONG_SPOT")
                    if eff_bar_a and eff_anc_a and eff_anc_a > 0:
                        mv_a = ((eff_bar_a.high / eff_anc_a) - 1.0) if is_short_a else (1.0 - (eff_bar_a.low / eff_anc_a))
                        adverse_leg_move = max(adverse_leg_move, mv_a)
                    if (not is_spot_perp) and eff_bar_b and eff_anc_b and eff_anc_b > 0:
                        mv_b = (1.0 - (eff_bar_b.low / eff_anc_b)) if is_short_a else ((eff_bar_b.high / eff_anc_b) - 1.0)
                        adverse_leg_move = max(adverse_leg_move, mv_b)

                    leg_lev = pos.notional_usd / slot_margin_per_leg
                    margin_consumed = adverse_leg_move * leg_lev
                    if margin_consumed >= 0.90:
                        liquidation_count += 1
                        # Red-Team #3: realize maintenance-margin liquidation penalty + unhedged exit slip
                        extra_rebal_usd[sym] = extra_rebal_usd.get(sym, 0.0) + max(
                            slot_margin_per_leg * 0.90, pos.notional_usd * 0.05
                        )
                        to_close.append((sym, "MARGIN_LIQUIDATION"))
                        continue
                    elif margin_consumed >= 0.65:
                        forced_rebalance_count += 1
                        extra_rebal_usd[sym] = extra_rebal_usd.get(sym, 0.0) + max(1.0, slot_margin_per_leg * 0.001)
                        anchor_px[sym] = (cur_a or anc_a, cur_b or anc_b)

                _, basis_bps = _basis_pnl_usd(
                    pos.direction, pos.notional_usd, pos.entry_px_a, pos.entry_px_b, cur_a, cur_b
                )
                if basis_bps < pos.max_adverse_basis_bps:
                    pos.max_adverse_basis_bps = basis_bps

                if basis_bps <= -cfg.stop_loss_basis_bps:
                    to_close.append((sym, "BASIS_STOP"))
                    continue

                if cfg.use_atb_buggy_rules:
                    if pos.low_streak_settlements >= 3:
                        to_close.append((sym, "ATB_BUGGY_LOW_STREAK"))
                    continue

                apr_a = fast_rolling_apr(sym, step_idx, is_b=False)
                apr_b = 0.0 if is_spot_perp else fast_rolling_apr(sym, step_idx, is_b=True)
                if apr_a is not None and apr_b is not None:
                    cur_spread_apr = (apr_a - apr_b) if pos.direction in ("SHORT_A_LONG_B", "SHORT_A_LONG_SPOT") else (apr_b - apr_a)
                    if cur_spread_apr < cfg.exit_spread_apr_pct:
                        pos.low_streak_hours += 1
                    else:
                        pos.low_streak_hours = 0

                    if hold_h >= cfg.min_hold_hours and pos.low_streak_hours >= cfg.exit_confirm_hours:
                        to_close.append((sym, "HYSTERESIS_EXIT"))

            for sym, reason in to_close:
                pos = open_pos.pop(sym)
                anchor_px.pop(sym, None)
                forced_cost = extra_rebal_usd.pop(sym, 0.0)
                if reason in ("MARGIN_LIQUIDATION", "BASIS_STOP"):
                    cooldown_until[sym] = ts + 24 * HOUR_MS
                trade_seq += 1
                tr, priced = self._finalize_trade(
                    trade_seq, pos, ts, reason, prices_a[sym].get(ts), prices_b[sym].get(ts),
                    one_way_fee_bps, one_way_slip_bps, cfg.rebalance_drag_bps_per_month, forced_cost,
                )
                if priced:
                    basis_priced_trades += 1
                closed_trades.append(tr)
                realized_pnl_usd += tr.net_pnl_usd

            # 3) Check entries if after warmup and slots available (respecting post-stop cooldown)
            if ts >= warmup_end and len(open_pos) < cfg.max_positions:
                candidates: list[tuple[float, str, str, float]] = []
                for sym in symbols:
                    if sym in open_pos or ts < cooldown_until.get(sym, 0):
                        continue
                    apr_a = fast_rolling_apr(sym, step_idx, is_b=False)
                    apr_b = 0.0 if is_spot_perp else fast_rolling_apr(sym, step_idx, is_b=True)
                    if apr_a is None or apr_b is None:
                        continue

                    if cfg.use_atb_buggy_rules:
                        if apr_a >= 10.95:
                            candidates.append((apr_a, sym, "SHORT_A_LONG_SPOT", apr_a))
                        continue

                    spread_ab = apr_a - apr_b
                    if is_spot_perp or not cfg.allow_bidirectional:
                        if spread_ab >= cfg.entry_spread_apr_pct:
                            candidates.append((spread_ab, sym, "SHORT_A_LONG_SPOT" if is_spot_perp else "SHORT_A_LONG_B", spread_ab))
                    else:
                        if abs(spread_ab) >= cfg.entry_spread_apr_pct:
                            direction = "SHORT_A_LONG_B" if spread_ab > 0 else "LONG_A_SHORT_B"
                            candidates.append((abs(spread_ab), sym, direction, abs(spread_ab)))

                candidates.sort(key=lambda x: x[0], reverse=True)
                for score, sym, direction, spread_apr in candidates:
                    if len(open_pos) >= cfg.max_positions:
                        break
                    px_a = prices_a[sym].get(ts)
                    px_b = prices_b[sym].get(ts)
                    if px_a and px_b and px_a > 0 and px_b > 0:
                        raw_basis_bps = ((px_a - px_b) / px_b) * 10000.0
                        adverse_entry = -raw_basis_bps if direction in ("SHORT_A_LONG_B", "SHORT_A_LONG_SPOT") else raw_basis_bps
                        if adverse_entry > cfg.max_basis_entry_bps:
                            continue
                    open_pos[sym] = _OpenPosition(
                        symbol=sym,
                        direction=direction,
                        entry_ts_ms=ts,
                        notional_usd=notional_per_leg,
                        entry_spread_apr_pct=spread_apr,
                        entry_px_a=px_a,
                        entry_px_b=px_b,
                    )
                    anchor_px[sym] = (px_a, px_b)
                    extra_rebal_usd[sym] = 0.0

            # 4) Mark-to-market equity snapshot
            unrealized = 0.0
            for sym, pos in open_pos.items():
                hold_h = max(0.0, (ts - pos.entry_ts_ms) / HOUR_MS)
                cur_a = prices_a[sym].get(ts)
                cur_b = prices_b[sym].get(ts)
                b_pnl, _ = _basis_pnl_usd(
                    pos.direction, pos.notional_usd, pos.entry_px_a, pos.entry_px_b, cur_a, cur_b
                )
                est_rt_cost = pos.notional_usd * (2.0 * (one_way_fee_bps + one_way_slip_bps)) / 10000.0
                accrued_rebal = pos.notional_usd * (cfg.rebalance_drag_bps_per_month / 10000.0) * (hold_h / (30.0 * 24.0))
                unrealized += pos.gross_funding_usd + b_pnl - est_rt_cost - accrued_rebal - extra_rebal_usd.get(sym, 0.0)

            equity_curve.append((ts, cfg.capital_usd + realized_pnl_usd + unrealized))
            util_samples.append(((len(open_pos) * slot_capital) / cfg.capital_usd * 100.0) if cfg.capital_usd > 0 else 0.0)

        # 5) Close remaining positions at end of window (mark-to-market settlement)
        final_ts = timeline[-1]
        for sym in list(open_pos.keys()):
            pos = open_pos.pop(sym)
            anchor_px.pop(sym, None)
            forced_cost = extra_rebal_usd.pop(sym, 0.0)
            trade_seq += 1
            tr, priced = self._finalize_trade(
                trade_seq, pos, final_ts, "END_OF_WINDOW",
                prices_a[sym].get(final_ts), prices_b[sym].get(final_ts),
                one_way_fee_bps, one_way_slip_bps, cfg.rebalance_drag_bps_per_month, forced_cost,
            )
            if priced:
                basis_priced_trades += 1
            closed_trades.append(tr)
            realized_pnl_usd += tr.net_pnl_usd

        if equity_curve:
            equity_curve[-1] = (final_ts, cfg.capital_usd + realized_pnl_usd)

        basis_cov = (basis_priced_trades / len(closed_trades) * 100.0) if closed_trades else 0.0
        if closed_trades and basis_cov < 50.0:
            warnings.append(
                f"LOW_DUAL_CANDLE_COVERAGE ({basis_cov:.1f}%): venue_b lacked 1h candles; basis PnL partially unpriced."
            )
        if liquidation_count > 0:
            warnings.append(f"MARGIN_LIQUIDATION_OCCURRED ({liquidation_count} events)")

        metrics = self._compute_metrics(
            cfg=cfg,
            start_ms=min_ts,
            end_ms=max_ts,
            trades=closed_trades,
            equity_curve=equity_curve,
            util_samples=util_samples,
            provenance=prov,
            dataset_sha=dataset_sha,
            liquidation_count=liquidation_count,
            forced_rebalance_count=forced_rebalance_count,
        )
        return BacktestResult(
            config=cfg,
            metrics=metrics,
            trades=closed_trades,
            equity_curve=equity_curve,
            basis_coverage_pct=round(basis_cov, 2),
            warnings=warnings,
        )

    def _finalize_trade(
        self,
        seq: int,
        pos: _OpenPosition,
        exit_ts_ms: int,
        reason: str,
        exit_px_a: float | None,
        exit_px_b: float | None,
        one_way_fee_bps: float,
        one_way_slip_bps: float,
        rebalance_drag_bps_per_month: float,
        extra_rebal_usd: float = 0.0,
    ) -> tuple[TradeRecord, bool]:
        hold_h = max(0.0, (exit_ts_ms - pos.entry_ts_ms) / HOUR_MS)
        priced = bool(pos.entry_px_a and pos.entry_px_b and exit_px_a and exit_px_b)
        b_pnl, _ = _basis_pnl_usd(
            pos.direction, pos.notional_usd, pos.entry_px_a, pos.entry_px_b, exit_px_a, exit_px_b
        )
        fee_usd = pos.notional_usd * (2.0 * one_way_fee_bps) / 10000.0
        slip_usd = pos.notional_usd * (2.0 * one_way_slip_bps) / 10000.0
        rebal_usd = pos.notional_usd * (rebalance_drag_bps_per_month / 10000.0) * (hold_h / (30.0 * 24.0)) + extra_rebal_usd
        net_usd = pos.gross_funding_usd + b_pnl - fee_usd - slip_usd - rebal_usd
        tr = TradeRecord(
            trade_id=f"T-{seq:04d}",
            symbol=pos.symbol,
            direction=pos.direction,
            entry_ts_ms=pos.entry_ts_ms,
            exit_ts_ms=exit_ts_ms,
            hold_hours=round(hold_h, 2),
            notional_per_leg_usd=round(pos.notional_usd, 2),
            entry_spread_apr_pct=round(pos.entry_spread_apr_pct, 2),
            gross_funding_usd=round(pos.gross_funding_usd, 4),
            basis_pnl_usd=round(b_pnl, 4),
            fee_usd=round(fee_usd, 4),
            slippage_usd=round(slip_usd, 4),
            rebalance_usd=round(rebal_usd, 4),
            net_pnl_usd=round(net_usd, 4),
            exit_reason=reason,
            max_adverse_basis_bps=round(pos.max_adverse_basis_bps, 2),
        )
        return tr, priced

    def _compute_metrics(
        self,
        cfg: StrategyConfig,
        start_ms: int,
        end_ms: int,
        trades: list[TradeRecord],
        equity_curve: list[tuple[int, float]],
        util_samples: list[float],
        provenance: str,
        dataset_sha: str,
        liquidation_count: int = 0,
        forced_rebalance_count: int = 0,
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
        win_rate = (len(wins) / len(trades)) if trades else 0.0
        avg_hold = (sum(t.hold_hours for t in trades) / len(trades)) if trades else 0.0
        pf = (sum(wins) / sum(losses)) if sum(losses) > 0 else (99.0 if sum(wins) > 0 else 0.0)

        total_ret_pct = (net_pnl / cfg.capital_usd) * 100.0 if cfg.capital_usd > 0 else 0.0
        ann_apr_pct = total_ret_pct * (365.0 / days)
        monthly_usd = net_pnl * (30.0 / days)

        # Underwater Max Drawdown from hourly equity curve
        peak = cfg.capital_usd
        max_dd_pct = 0.0
        for _, eq in equity_curve:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = ((peak - eq) / peak) * 100.0
                if dd > max_dd_pct:
                    max_dd_pct = dd

        # Daily equity returns for annualized Sharpe (sqrt(365) for 24/7 crypto)
        daily_eq: dict[int, float] = {}
        for ts, eq in equity_curve:
            day_idx = ts // (86400 * 1000)
            daily_eq[day_idx] = eq
        day_vals = [daily_eq[k] for k in sorted(daily_eq.keys())]
        daily_rets: list[float] = []
        for i in range(1, len(day_vals)):
            if day_vals[i - 1] > 0:
                daily_rets.append((day_vals[i] / day_vals[i - 1]) - 1.0)
        if len(daily_rets) >= 3:
            mean_r = sum(daily_rets) / len(daily_rets)
            var_r = sum((r - mean_r) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
            std_r = math.sqrt(var_r)
            sharpe = (mean_r / std_r) * math.sqrt(365.0) if std_r > 1e-12 else 0.0
        else:
            sharpe = 0.0

        calmar = (ann_apr_pct / max_dd_pct) if max_dd_pct > 0.05 else (ann_apr_pct / 0.05 if ann_apr_pct > 0 else 0.0)
        gross_denom = gross_alpha if gross_alpha > 0 else gross_f
        fee_to_gross = ((fees + slips + rebals) / gross_denom) if gross_denom > 0 else 99.0
        avg_util = (sum(util_samples) / len(util_samples)) if util_samples else 0.0

        # Concentration stress: drop top 5 winning trades & drop best symbol
        sorted_pnls = sorted((t.net_pnl_usd for t in trades), reverse=True)
        top5_removed = sum(sorted_pnls[5:]) if len(sorted_pnls) > 5 else 0.0

        by_sym: dict[str, dict[str, float]] = {}
        for t in trades:
            bucket = by_sym.setdefault(t.symbol, {
                "trades": 0.0, "wins": 0.0, "gross_funding_usd": 0.0,
                "basis_pnl_usd": 0.0, "costs_usd": 0.0, "net_pnl_usd": 0.0, "hold_hours": 0.0,
            })
            bucket["trades"] += 1.0
            if t.net_pnl_usd > 0:
                bucket["wins"] += 1.0
            bucket["gross_funding_usd"] += t.gross_funding_usd
            bucket["basis_pnl_usd"] += t.basis_pnl_usd
            bucket["costs_usd"] += t.fee_usd + t.slippage_usd + t.rebalance_usd
            bucket["net_pnl_usd"] += t.net_pnl_usd
            bucket["hold_hours"] += t.hold_hours

        for s_stats in by_sym.values():
            s_stats["win_rate"] = round(s_stats["wins"] / s_stats["trades"], 4) if s_stats["trades"] else 0.0
            s_stats["avg_hold_hours"] = round(s_stats["hold_hours"] / s_stats["trades"], 2) if s_stats["trades"] else 0.0
            for k in ("gross_funding_usd", "basis_pnl_usd", "costs_usd", "net_pnl_usd"):
                s_stats[k] = round(s_stats[k], 4)

        best_sym = ""
        best_sym_pnl = -1e18
        for s, st in by_sym.items():
            if st["net_pnl_usd"] > best_sym_pnl:
                best_sym_pnl = st["net_pnl_usd"]
                best_sym = s
        top_sym_removed = net_pnl - max(0.0, best_sym_pnl) if by_sym else 0.0

        return BacktestMetrics(
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            days=round(days, 2),
            capital_usd=cfg.capital_usd,
            trade_count=len(trades),
            win_count=len(wins),
            win_rate=round(win_rate, 4),
            avg_hold_hours=round(avg_hold, 2),
            gross_funding_usd=round(gross_f, 4),
            basis_pnl_usd=round(basis_p, 4),
            total_fees_usd=round(fees, 4),
            total_slippage_usd=round(slips, 4),
            total_rebalance_usd=round(rebals, 4),
            net_pnl_usd=round(net_pnl, 4),
            total_return_pct=round(total_ret_pct, 4),
            annualized_apr_pct=round(ann_apr_pct, 2),
            sharpe_ratio=round(sharpe, 3),
            calmar_ratio=round(calmar, 3),
            max_drawdown_pct=round(max_dd_pct, 4),
            profit_factor=round(pf, 3),
            fee_to_gross_ratio=round(fee_to_gross, 4),
            avg_capital_utilization_pct=round(avg_util, 2),
            monthly_usd_est=round(monthly_usd, 2),
            top5_removed_net_pnl_usd=round(top5_removed, 4),
            top_symbol_removed_net_pnl_usd=round(top_sym_removed, 4),
            best_symbol=best_sym,
            provenance=provenance,
            dataset_sha256=dataset_sha,
            by_symbol=by_sym,
            gross_alpha_usd=round(gross_alpha, 4),
            liquidation_count=liquidation_count,
            forced_rebalance_count=forced_rebalance_count,
        )
