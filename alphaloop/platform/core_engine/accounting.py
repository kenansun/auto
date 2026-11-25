"""Asset-Agnostic Portfolio Accounting Engine & REA-001 3-Tier Trade Ledger (Fixes D1, D13, D15).

Core guarantees:
  - D1 Exact Gap Conservation: Tracks per-symbol `last_bar[sym]` and computes overnight /
    exit-bar / holiday gap PnL `prev_w * eq_start * (bar.open / last_bar[sym].close - 1.0)`
    strictly BEFORE applying open-fill trade costs or overwriting `active_weights[sym]`.
  - D13 True Compounding Metrics: Reports true `cagr_pct` as primary annualized return
    alongside `simple_apr_pct`.
  - D15 Market Physics: Exchange-local trading-date `T+1`/`T+N` lock (never `bar_index + N`),
    per-symbol calendar annualization for holding carry drag, defensive `contract_multiplier`
    lookup, and conservative intrabar stop & liquidation guards distinguishing leveraged
    perpetuals/margin accounts from unleveraged spot/ETF holdings.
  - REA-001 3-Tier Auditability: Retains full `candidates`, `executed`, and `suppressed`
    records plus explicit per-leg fee, slippage, stamp-duty, funding, and holding-drag totals.
  - Streaming-vs-Batch Parity (G7): `PortfolioAccountingEngine.simulate(...)` drives the exact
    same `StreamingAccountingSession.step_bar(...)` runner verified by `verify_full_parity`.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

from alphaloop.platform.assets import (
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_GOLD_PERP,
    AssetSpec,
    TradingCalendar,
)
from alphaloop.platform.core_engine.interfaces import AssetNativeStrategy, BarDecision
from alphaloop.platform.core_engine.risk_allocator import CausalRiskAllocator, RiskAllocatorConfig
from alphaloop.platform.data_lake import CanonicalBar

logger = logging.getLogger(__name__)


def exchange_trading_date(ts_ms: int, calendar: TradingCalendar) -> date:
    """Resolve exchange local trading date from UTC ms for T+1 / T+N settlement enforcement."""
    tz_name = getattr(calendar, "timezone_name", None)
    tz_off = getattr(calendar, "tz_offset_hours", None)
    if tz_name:
        try:
            tz: Any = ZoneInfo(str(tz_name))
        except Exception as exc:
            logger.debug("Falling back from unknown timezone %s to UTC: %s", tz_name, exc)
            tz = timezone.utc
    elif tz_off is not None:
        tz = timezone(timedelta(hours=float(tz_off)))
    elif calendar.calendar_id == "CN_SSE_SZSE":
        # Asia/Shanghai fixed UTC+8 (no DST since 1991; zero tzdata failure risk)
        tz = timezone(timedelta(hours=8))
    elif calendar.calendar_id in ("US_EQUITY_DAILY", "GOLD_TRADFI_23_5"):
        try:
            tz = ZoneInfo("America/New_York")
        except Exception as exc:
            logger.debug("ZoneInfo America/New_York unavailable, using UTC-5: %s", exc)
            tz = timezone(timedelta(hours=-5))
    else:
        tz = timezone.utc
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=tz).date()


def _fast_sample_std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = math.fsum(xs) / n
    ss = math.fsum((x - mu) * (x - mu) for x in xs)
    var = ss / (n - 1)
    return math.sqrt(var) if (math.isfinite(var) and var > 0.0) else 0.0


def _fast_sharpe(returns: Sequence[float], periods_per_year: float) -> float:
    if len(returns) < 3:
        return 0.0
    mu = math.fsum(returns) / len(returns)
    sd = _fast_sample_std(returns)
    if sd <= 1e-12:
        return 0.0
    return (mu / sd) * math.sqrt(max(1.0, float(periods_per_year)))


def _is_leveraged_exposure(spec: AssetSpec, gross_leverage: float, weight: float = 0.0) -> bool:
    if weight < -1e-9:
        return True
    if spec.asset_class in (ASSET_CLASS_CRYPTO_PERP, ASSET_CLASS_GOLD_PERP):
        return True
    return gross_leverage > 1.0 + 1e-6


def _resolve_contract_multiplier(spec: AssetSpec, bar: CanonicalBar | None) -> float:
    spec_raw = getattr(spec, "contract_multiplier", None)
    bar_raw = getattr(bar, "contract_multiplier", None) if bar is not None else None
    spec_m: float | None = None
    bar_m: float | None = None
    if spec_raw is not None:
        try:
            v = float(spec_raw)
            if math.isfinite(v) and v > 0.0:
                spec_m = v
        except (TypeError, ValueError):
            spec_m = None
    if bar_raw is not None:
        try:
            v = float(bar_raw)
            if math.isfinite(v) and v > 0.0:
                bar_m = v
        except (TypeError, ValueError):
            bar_m = None
    if spec_m is not None and abs(spec_m - 1.0) > 1e-12:
        return spec_m
    if bar_m is not None and abs(bar_m - 1.0) > 1e-12:
        return bar_m
    return spec_m if spec_m is not None else (bar_m if bar_m is not None else 1.0)


def _decompose_one_way_costs(
    spec: AssetSpec,
    execution_mode: str,
    is_sell: bool,
    extra_slippage_bps: float,
    notional_usd: float,
) -> tuple[float, float, float]:
    """Decompose CostModel into exact (fee_usd, slippage_usd, stamp_duty_usd)."""
    cm = spec.cost_model
    mode = execution_mode.upper()
    if mode == "MAKER_MAKER":
        fee_bps = cm.maker_fee_bps
    elif mode == "MAKER_THEN_TAKER_HEDGE":
        fee_bps = 0.5 * (cm.maker_fee_bps + cm.taker_fee_bps)
    else:
        fee_bps = cm.taker_fee_bps
    slip_bps = max(0.0, float(cm.slippage_bps_per_leg) + float(extra_slippage_bps))
    stamp_bps = float(cm.stamp_duty_sell_bps) if is_sell else 0.0
    fee_usd = notional_usd * (max(0.0, float(fee_bps)) / 10_000.0)
    slippage_usd = notional_usd * (slip_bps / 10_000.0)
    stamp_duty_usd = notional_usd * (max(0.0, stamp_bps) / 10_000.0)
    return fee_usd, slippage_usd, stamp_duty_usd


@dataclass(frozen=True)
class CandidateRecord:
    """REA-001 Tier 1: Every bar-close candidate decision before open-fill execution."""

    ts_ms: int
    symbol: str
    raw_signal: float
    raw_target_weight: float
    allocated_weight: float
    regime_label: str
    stop_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutedTrade:
    """REA-001 Tier 2: Every executed fill (open rebalance, intrabar stop, or liquidation)."""

    ts_fill_ms: int
    symbol: str
    side: str  # "BUY" | "SELL"
    prev_weight: float
    new_weight: float
    delta_weight: float
    fill_price: float
    notional_usd: float
    contracts: float
    contract_multiplier: float
    fee_usd: float
    slippage_usd: float
    stamp_duty_usd: float
    total_cost_usd: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SuppressedOrder:
    """REA-001 Tier 3: Every order suppressed, capped, or locked by risk/market constraints."""

    ts_ms: int
    symbol: str
    requested_weight: float
    effective_weight: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TradeLedger:
    """REA-001 3-Tier Audit Ledger (candidates / executed / suppressed + cost decomposition)."""

    candidates: list[CandidateRecord] = field(default_factory=list)
    executed: list[ExecutedTrade] = field(default_factory=list)
    suppressed: list[SuppressedOrder] = field(default_factory=list)
    total_fee_usd: float = 0.0
    total_slippage_usd: float = 0.0
    total_stamp_duty_usd: float = 0.0
    total_funding_usd: float = 0.0
    total_holding_drag_usd: float = 0.0

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def executed_count(self) -> int:
        return len(self.executed)

    @property
    def suppressed_count(self) -> int:
        return len(self.suppressed)

    def suppression_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.suppressed:
            counts[row.reason] = counts.get(row.reason, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self, include_rows: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "candidate_count": self.candidate_count,
            "executed_count": self.executed_count,
            "suppressed_count": self.suppressed_count,
            "suppression_counts": self.suppression_counts(),
            "total_fee_usd": round(self.total_fee_usd, 6),
            "total_slippage_usd": round(self.total_slippage_usd, 6),
            "total_stamp_duty_usd": round(self.total_stamp_duty_usd, 6),
            "total_funding_usd": round(self.total_funding_usd, 6),
            "total_holding_drag_usd": round(self.total_holding_drag_usd, 6),
        }
        if include_rows:
            out["candidates"] = [c.to_dict() for c in self.candidates]
            out["executed"] = [e.to_dict() for e in self.executed]
            out["suppressed"] = [s.to_dict() for s in self.suppressed]
        return out


@dataclass(frozen=True)
class BarStepResult:
    """Per-bar output emitted by `StreamingAccountingSession.step_bar(...)`."""

    ts_ms: int
    equity_usd: float
    raw_decisions: dict[str, BarDecision]
    allocated_weights: dict[str, float]
    executed_weights: dict[str, float]
    had_fill_override: bool = False


@dataclass
class SimulationResult:
    """Harmonized institutional backtest/streaming result for Core Engine v4."""

    strategy_id: str
    asset_domain: str
    capital_usd: float
    final_equity_usd: float
    days: float
    bars_simulated: int
    bars_per_year: float
    trade_count: int
    total_turnover: float
    gross_price_pnl_usd: float
    gap_pnl_usd: float
    intraday_pnl_usd: float
    funding_pnl_usd: float
    holding_drag_usd: float
    funding_carry_pnl_usd: float
    total_fee_usd: float
    total_slippage_usd: float
    total_stamp_duty_usd: float
    total_cost_usd: float
    net_pnl_usd: float
    total_return_pct: float
    cagr_pct: float  # PRIMARY annualized return (D13)
    simple_apr_pct: float  # SECONDARY linear annualized return
    sharpe_ratio: float
    calmar_ratio: float
    realized_annual_vol_pct: float
    max_drawdown_pct: float
    max_active_positions: int
    mean_active_positions: float
    max_gross_leverage: float
    liquidation_count: int
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    timestamps_ms: list[int] = field(default_factory=list)
    period_returns: list[float] = field(default_factory=list)
    period_timestamps_ms: list[int] = field(default_factory=list)
    by_symbol_net_usd: dict[str, float] = field(default_factory=dict)
    raw_decisions_history: list[dict[str, BarDecision]] = field(default_factory=list)
    allocated_weights_history: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    executed_weights_history: list[dict[str, float]] = field(default_factory=list)
    trade_ledger: TradeLedger = field(default_factory=TradeLedger)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "asset_domain": self.asset_domain,
            "capital_usd": round(self.capital_usd, 2),
            "final_equity_usd": round(self.final_equity_usd, 2),
            "days": round(self.days, 2),
            "bars_simulated": self.bars_simulated,
            "bars_per_year": round(self.bars_per_year, 2),
            "trade_count": self.trade_count,
            "total_turnover": round(self.total_turnover, 4),
            "gross_price_pnl_usd": round(self.gross_price_pnl_usd, 4),
            "gap_pnl_usd": round(self.gap_pnl_usd, 4),
            "intraday_pnl_usd": round(self.intraday_pnl_usd, 4),
            "funding_pnl_usd": round(self.funding_pnl_usd, 4),
            "holding_drag_usd": round(self.holding_drag_usd, 4),
            "funding_carry_pnl_usd": round(self.funding_carry_pnl_usd, 4),
            "total_fee_usd": round(self.total_fee_usd, 4),
            "total_slippage_usd": round(self.total_slippage_usd, 4),
            "total_stamp_duty_usd": round(self.total_stamp_duty_usd, 4),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "net_pnl_usd": round(self.net_pnl_usd, 4),
            "total_return_pct": round(self.total_return_pct, 4),
            "cagr_pct": round(self.cagr_pct, 4),
            "simple_apr_pct": round(self.simple_apr_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "realized_annual_vol_pct": round(self.realized_annual_vol_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "max_active_positions": self.max_active_positions,
            "mean_active_positions": round(self.mean_active_positions, 3),
            "max_gross_leverage": round(self.max_gross_leverage, 4),
            "liquidation_count": self.liquidation_count,
            "by_symbol_net_usd": {k: round(v, 4) for k, v in sorted(self.by_symbol_net_usd.items())},
            "trade_ledger": self.trade_ledger.to_dict(include_rows=False),
        }

    def to_dict(self, include_curves: bool = False) -> dict[str, Any]:
        out = self.summary_dict()
        if include_curves:
            out["equity_curve"] = list(self.equity_curve)
            out["timestamps_ms"] = list(self.timestamps_ms)
            out["period_returns"] = list(self.period_returns)
            out["period_timestamps_ms"] = list(self.period_timestamps_ms)
            out["allocated_weights_history"] = [
                (ts, dict(w)) for ts, w in self.allocated_weights_history
            ]
            out["executed_weights_history"] = [dict(w) for w in self.executed_weights_history]
        return out


class StreamingAccountingSession:
    """Stateful bar-by-bar accounting & risk runner shared identically by batch and G7 streaming."""

    def __init__(
        self,
        strategy: AssetNativeStrategy,
        specs: dict[str, AssetSpec],
        *,
        risk_cfg: RiskAllocatorConfig,
        capital_usd: float = 10_000.0,
        execution_mode: str = "TAKER_TAKER",
        extra_slippage_bps_per_leg: float = 0.0,
        maint_margin_rate: float = 0.05,
        liquidation_fee_bps: float = 20.0,
    ) -> None:
        if capital_usd <= 0.0 or not math.isfinite(capital_usd):
            raise ValueError(f"capital_usd must be finite and > 0, got {capital_usd}")
        self.strategy = strategy
        self.specs = dict(specs)
        self.risk_cfg = risk_cfg
        self.risk_allocator = CausalRiskAllocator(risk_cfg)
        self.capital_usd = float(capital_usd)
        self.execution_mode = str(execution_mode)
        self.extra_slippage_bps_per_leg = float(extra_slippage_bps_per_leg)
        self.maint_margin_rate = max(0.005, float(maint_margin_rate))
        self.liquidation_fee_bps = max(0.0, float(liquidation_fee_bps))

        self.strategy.reset()

        self._step_idx: int = 0
        self._equity: float = float(capital_usd)
        self._peak_eq: float = float(capital_usd)
        self._max_dd: float = 0.0
        self._halted: bool = False

        self._active_weights: dict[str, float] = {s: 0.0 for s in sorted(specs)}
        self._pending_weights: dict[str, float] = {s: 0.0 for s in sorted(specs)}
        self._active_stop_prices: dict[str, float | None] = {s: None for s in sorted(specs)}
        self._last_bar: dict[str, CanonicalBar] = {}
        self._last_seen_trade_date: dict[str, date | None] = {}
        self._trading_days_since_buy: dict[str, int] = {}

        ret_maxlen = max(128, int(risk_cfg.corr_lookback_bars) + 5)
        self._recent_returns: dict[str, deque[float]] = {
            s: deque(maxlen=ret_maxlen) for s in sorted(specs)
        }
        self._seen_timeframes: dict[str, float] = {}
        self._last_raw_signal: dict[str, float] = {}

        self._gross_price_pnl: float = 0.0
        self._gap_pnl: float = 0.0
        self._intraday_pnl: float = 0.0
        self._funding_pnl: float = 0.0
        self._holding_drag_pnl: float = 0.0
        self._total_fee_usd: float = 0.0
        self._total_slippage_usd: float = 0.0
        self._total_stamp_duty_usd: float = 0.0
        self._total_turnover: float = 0.0
        self._trade_count: int = 0
        self._liquidation_count: int = 0
        self._max_active_positions: int = 0
        self._sum_active_positions: int = 0
        self._active_pos_obs: int = 0
        self._max_gross_leverage: float = 0.0
        self._by_symbol_net_usd: dict[str, float] = {s: 0.0 for s in sorted(specs)}

        self._equity_curve: list[tuple[int, float]] = []
        self._timestamps_ms: list[int] = []
        self._period_returns: list[float] = []
        self._period_timestamps_ms: list[int] = []
        self._raw_decisions_history: list[dict[str, BarDecision]] = []
        self._allocated_weights_history: list[tuple[int, dict[str, float]]] = []
        self._executed_weights_history: list[dict[str, float]] = []
        self._ledger = TradeLedger()
        self._had_any_fill_override: bool = False

    @property
    def had_any_fill_override(self) -> bool:
        return self._had_any_fill_override

    @property
    def is_halted(self) -> bool:
        return self._halted

    def step_bar(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> BarStepResult:
        """Advance portfolio accounting and strategy state strictly causally by one timestamp."""
        idx = self._step_idx
        self._step_idx += 1
        eq_start = self._equity

        # Filter to known specs in deterministic sorted symbol order
        active_bars: dict[str, CanonicalBar] = {
            sym: bars_at_ts[sym] for sym in sorted(bars_at_ts) if sym in self.specs
        }
        for sym, bar in active_bars.items():
            self._seen_timeframes[sym] = float(bar.timeframe_h)
            if sym not in self._active_weights:
                self._active_weights[sym] = 0.0
                self._pending_weights[sym] = 0.0
                self._active_stop_prices[sym] = None
                self._by_symbol_net_usd[sym] = 0.0
                self._recent_returns[sym] = deque(maxlen=max(128, int(self.risk_cfg.corr_lookback_bars) + 5))

        if self._halted or eq_start <= 1e-6:
            self._halted = True
            self._had_any_fill_override = True
            for s in self._active_weights:
                self._active_weights[s] = 0.0
                self._pending_weights[s] = 0.0
                self._active_stop_prices[s] = None
            for sym, bar in active_bars.items():
                prev_b = self._last_bar.get(sym)
                if prev_b is not None and prev_b.close > 0.0 and bar.close > 0.0:
                    self._recent_returns[sym].append((bar.close / prev_b.close) - 1.0)
                self._last_bar[sym] = bar
            empty_decs: dict[str, BarDecision] = {}
            zero_w = {s: 0.0 for s in sorted(self._active_weights)}
            self._equity_curve.append((ts_ms, self._equity))
            self._timestamps_ms.append(ts_ms)
            self._raw_decisions_history.append(empty_decs)
            self._allocated_weights_history.append((ts_ms, dict(zero_w)))
            self._executed_weights_history.append(dict(zero_w))
            if idx >= self.strategy.warmup_bars:
                self._period_returns.append(0.0)
                self._period_timestamps_ms.append(ts_ms)
            return BarStepResult(ts_ms, self._equity, empty_decs, zero_w, zero_w, False)

        # Update exchange-local trading date counters for T+1 / T+N settlement tracking (D15a)
        for sym, bar in active_bars.items():
            spec = self.specs[sym]
            cur_date = exchange_trading_date(bar.ts_event_ms, spec.calendar)
            prev_date = self._last_seen_trade_date.get(sym)
            if prev_date is not None and cur_date > prev_date:
                self._trading_days_since_buy[sym] = self._trading_days_since_buy.get(sym, 0) + 1
            self._last_seen_trade_date[sym] = cur_date

        had_override_this_bar = False

        # ------------------------------------------------------------------
        # STEP A: D1 Overnight / Exit-Bar / Holiday Gap PnL BEFORE weight overwrite!
        # ------------------------------------------------------------------
        gap_usd_by_sym: dict[str, float] = {}
        gap_rets: dict[str, float] = {}
        gross_prev_lev = math.fsum(abs(w) for w in self._active_weights.values())
        for sym, bar in active_bars.items():
            prev_w = self._active_weights.get(sym, 0.0)
            prev_b = self._last_bar.get(sym)
            if abs(prev_w) > 1e-9 and prev_b is not None and prev_b.close > 0.0 and bar.open > 0.0:
                g_ret = (bar.open / prev_b.close) - 1.0
                gap_rets[sym] = g_ret
                gap_usd_by_sym[sym] = (prev_w * eq_start) * g_ret

        raw_bar_gap_pnl = math.fsum(gap_usd_by_sym.values())
        if eq_start + raw_bar_gap_pnl < 0.0:
            # Proportionally clip ONLY negative gap losses so eq_open == 0.0 and exact conservation holds
            pos_gap = math.fsum(v for v in gap_usd_by_sym.values() if v > 0.0)
            neg_gap_abs = math.fsum(-v for v in gap_usd_by_sym.values() if v < 0.0)
            loss_budget = max(0.0, eq_start + pos_gap)
            neg_scale = (loss_budget / neg_gap_abs) if neg_gap_abs > 1e-12 else 0.0
            for sym in list(gap_usd_by_sym):
                if gap_usd_by_sym[sym] < 0.0:
                    gap_usd_by_sym[sym] *= neg_scale

        bar_gap_pnl = math.fsum(gap_usd_by_sym.values())
        for sym, gap_usd in gap_usd_by_sym.items():
            self._gap_pnl += gap_usd
            self._gross_price_pnl += gap_usd
            self._by_symbol_net_usd[sym] = self._by_symbol_net_usd.get(sym, 0.0) + gap_usd

        eq_open = max(0.0, eq_start + bar_gap_pnl)

        # Check overnight maintenance-margin liquidation for leveraged OR short exposure (D15c)
        lev_open_notional = 0.0
        lev_syms_open: list[str] = []
        for sym in sorted(self._active_weights):
            pw = self._active_weights[sym]
            if abs(pw) > 1e-9 and _is_leveraged_exposure(self.specs[sym], gross_prev_lev, pw):
                g_ret = gap_rets.get(sym, 0.0)
                lev_open_notional += abs(pw * eq_start * (1.0 + g_ret))
                lev_syms_open.append(sym)

        if lev_syms_open and eq_open <= lev_open_notional * self.maint_margin_rate:
            self._liquidation_count += 1
            had_override_this_bar = True
            self._had_any_fill_override = True
            liq_fee_total = 0.0
            for sym in lev_syms_open:
                pw = self._active_weights[sym]
                bar_for_sym = active_bars.get(sym)
                if bar_for_sym is not None:
                    fill_px = bar_for_sym.open
                else:
                    last_b = self._last_bar.get(sym)
                    bar_for_sym = last_b
                    fill_px = last_b.close if last_b is not None else 1.0
                spec = self.specs[sym]
                notional = abs(pw * eq_start * (1.0 + gap_rets.get(sym, 0.0)))
                liq_fee = min(eq_open - liq_fee_total, notional * (self.liquidation_fee_bps / 10_000.0))
                liq_fee = max(0.0, liq_fee)
                liq_fee_total += liq_fee
                mult = _resolve_contract_multiplier(spec, bar_for_sym) if bar_for_sym is not None else 1.0
                contracts = notional / max(fill_px * mult, 1e-12)
                self._total_fee_usd += liq_fee
                self._ledger.total_fee_usd += liq_fee
                self._by_symbol_net_usd[sym] = self._by_symbol_net_usd.get(sym, 0.0) - liq_fee
                if eq_open > 1e-9:
                    self._total_turnover += notional / eq_open
                self._ledger.executed.append(
                    ExecutedTrade(
                        ts_fill_ms=ts_ms,
                        symbol=sym,
                        side="SELL" if pw > 0.0 else "BUY",
                        prev_weight=pw,
                        new_weight=0.0,
                        delta_weight=-pw,
                        fill_price=fill_px,
                        notional_usd=notional,
                        contracts=contracts,
                        contract_multiplier=mult,
                        fee_usd=liq_fee,
                        slippage_usd=0.0,
                        stamp_duty_usd=0.0,
                        total_cost_usd=liq_fee,
                        reason="OVERNIGHT_LIQUIDATION",
                    )
                )
                self._active_weights[sym] = 0.0
                self._pending_weights[sym] = 0.0
                self._active_stop_prices[sym] = None
                self._trade_count += 1
            eq_open = max(0.0, eq_open - liq_fee_total)

        # Clean wipeout halt when eq_open <= 1e-9: zero ALL weights (never write ghost weights!)
        if eq_open <= 1e-9:
            self._equity = 0.0
            self._halted = True
            self._had_any_fill_override = True
            if self._peak_eq > 0.0:
                self._max_dd = 100.0
            for s in self._active_weights:
                self._active_weights[s] = 0.0
                self._pending_weights[s] = 0.0
                self._active_stop_prices[s] = None
            for sym, bar in active_bars.items():
                prev_b = self._last_bar.get(sym)
                if prev_b is not None and prev_b.close > 0.0 and bar.close > 0.0:
                    self._recent_returns[sym].append((bar.close / prev_b.close) - 1.0)
                self._last_bar[sym] = bar
            self.strategy.sync_executed_weights(dict(self._active_weights))
            zero_w = {s: 0.0 for s in sorted(self._active_weights)}
            empty_decs_w: dict[str, BarDecision] = {}
            self._equity_curve.append((ts_ms, 0.0))
            self._timestamps_ms.append(ts_ms)
            self._executed_weights_history.append(dict(zero_w))
            self._raw_decisions_history.append(empty_decs_w)
            self._allocated_weights_history.append((ts_ms, dict(zero_w)))
            if idx >= self.strategy.warmup_bars:
                self._active_pos_obs += 1
                self._period_returns.append(-1.0 if eq_start > 1e-9 else 0.0)
                self._period_timestamps_ms.append(ts_ms)
            return BarStepResult(ts_ms, 0.0, empty_decs_w, dict(zero_w), dict(zero_w), True)

        # ------------------------------------------------------------------
        # STEP B: Execute Pending Open-Fill Orders at bar.open (2-Pass with Post-Lock Guards)
        # ------------------------------------------------------------------
        trade_cost_bar = 0.0
        held_notionals_open: dict[str, float] = {}
        traded_symbols_at_open: set[str] = set()

        # Pass 1: Resolve allow_short, T+1, and PRICE_LIMIT constraints & legal floor per printing symbol
        planned_exec_w: dict[str, float] = {}
        floor_abs_w: dict[str, float] = {}
        for sym, bar in active_bars.items():
            spec = self.specs[sym]
            prev_w = self._active_weights.get(sym, 0.0)
            desired_w = self._pending_weights.get(sym, 0.0)
            exec_w = desired_w

            # Enforce allow_short
            if not spec.calendar.allow_short and exec_w < -1e-12:
                self._ledger.suppressed.append(
                    SuppressedOrder(ts_ms, sym, desired_w, 0.0, "SHORT_NOT_ALLOWED")
                )
                exec_w = 0.0
                had_override_this_bar = True

            # Enforce exchange-local trading-date T+1 / T+N lock (D15a)
            t_plus = max(0, int(spec.calendar.t_plus_days))
            t_plus_cannot_reduce = (
                t_plus > 0
                and prev_w > 1e-9
                and self._trading_days_since_buy.get(sym, 0) < t_plus
            )
            if t_plus_cannot_reduce and exec_w < prev_w - 1e-9:
                self._ledger.suppressed.append(
                    SuppressedOrder(ts_ms, sym, desired_w, prev_w, "T_PLUS_1_LOCK")
                )
                exec_w = prev_w
                had_override_this_bar = True

            # Enforce daily price limit lock (e.g., CN A-share ±10% one-tick lock bar)
            prev_b = self._last_bar.get(sym)
            lim_pct = spec.calendar.daily_price_limit_pct
            is_limit_up_bar = False
            is_limit_dn_bar = False
            if (
                lim_pct is not None
                and prev_b is not None
                and prev_b.close > 0.0
                and abs(bar.high - bar.low) <= 1e-9 * max(1.0, abs(bar.open))
            ):
                move_pct = ((bar.open / prev_b.close) - 1.0) * 100.0
                is_limit_up_bar = move_pct >= (lim_pct - 0.15)
                is_limit_dn_bar = move_pct <= -(lim_pct - 0.15)
                if abs(exec_w - prev_w) > 1e-9:
                    limit_up_locked = is_limit_up_bar and exec_w > prev_w
                    limit_dn_locked = is_limit_dn_bar and exec_w < prev_w
                    if limit_up_locked or limit_dn_locked:
                        self._ledger.suppressed.append(
                            SuppressedOrder(
                                ts_ms,
                                sym,
                                desired_w,
                                prev_w,
                                "PRICE_LIMIT_UP_LOCK" if limit_up_locked else "PRICE_LIMIT_DOWN_LOCK",
                            )
                        )
                        exec_w = prev_w
                        had_override_this_bar = True

            # Determine irreducible floor magnitude under T+1 or price-limit lock
            cannot_reduce_held = (
                t_plus_cannot_reduce
                or (is_limit_dn_bar and prev_w > 1e-9)
                or (is_limit_up_bar and prev_w < -1e-9)
            )
            floor_abs_w[sym] = abs(prev_w) if cannot_reduce_held else 0.0
            planned_exec_w[sym] = exec_w

        # Post-Lock Guard 1: Enforce hard max_positions across non-printing held + locked/continuing + new entries
        non_printing_held = {
            s: float(w)
            for s, w in self._active_weights.items()
            if s not in active_bars and abs(w) > 1e-9
        }
        continuing_printing = [
            s for s in active_bars
            if abs(self._active_weights.get(s, 0.0)) > 1e-9 and abs(planned_exec_w[s]) > 1e-9
        ]
        new_entries = [
            s for s in active_bars
            if abs(self._active_weights.get(s, 0.0)) <= 1e-9 and abs(planned_exec_w[s]) > 1e-9
        ]
        current_active_count = len(non_printing_held) + len(continuing_printing)
        slots_for_new = max(0, int(self.risk_cfg.max_positions) - current_active_count)
        if len(new_entries) > slots_for_new:
            ranked_new = sorted(
                new_entries,
                key=lambda s: (-abs(self._last_raw_signal.get(s, 0.0)), -abs(planned_exec_w[s]), s),
            )
            for sym in ranked_new[slots_for_new:]:
                old_w = planned_exec_w[sym]
                self._ledger.suppressed.append(
                    SuppressedOrder(ts_ms, sym, old_w, 0.0, "POST_LOCK_MAX_POSITIONS_CAP")
                )
                planned_exec_w[sym] = 0.0
                self._pending_weights[sym] = 0.0
                had_override_this_bar = True
                self._had_any_fill_override = True

        # Post-Lock Guard 2: Enforce hard max_portfolio_leverage across non-printing + locked + adjustable orders
        non_printing_gross = math.fsum(abs(w) for w in non_printing_held.values())
        printing_gross = math.fsum(abs(planned_exec_w[s]) for s in active_bars)
        max_port_lev = float(self.risk_cfg.max_portfolio_leverage)
        if (non_printing_gross + printing_gross) > max_port_lev + 1e-9:
            pre_lev_w = dict(planned_exec_w)
            # Stage 1: Trim new increases above abs(prev_w) first
            excess = (non_printing_gross + printing_gross) - max_port_lev
            inc_by_sym: dict[str, float] = {}
            for s in active_bars:
                prev_abs = abs(self._active_weights.get(s, 0.0))
                cur_abs = abs(planned_exec_w[s])
                if cur_abs > prev_abs + 1e-12:
                    inc_by_sym[s] = cur_abs - prev_abs
            total_inc = math.fsum(inc_by_sym.values())
            if total_inc > 1e-12:
                cut_ratio = min(1.0, excess / total_inc)
                for s, inc in inc_by_sym.items():
                    old_w = planned_exec_w[s]
                    new_abs = max(floor_abs_w[s], abs(old_w) - inc * cut_ratio)
                    planned_exec_w[s] = math.copysign(new_abs, old_w) if new_abs > 1e-12 else 0.0

            # Stage 2: If still above cap, proportionally trim remaining adjustable weight above legal floor
            printing_gross = math.fsum(abs(planned_exec_w[s]) for s in active_bars)
            if (non_printing_gross + printing_gross) > max_port_lev + 1e-9:
                excess = (non_printing_gross + printing_gross) - max_port_lev
                adj_room: dict[str, float] = {}
                for s in active_bars:
                    room = max(0.0, abs(planned_exec_w[s]) - floor_abs_w[s])
                    if room > 1e-12:
                        adj_room[s] = room
                total_room = math.fsum(adj_room.values())
                if total_room > 1e-12:
                    cut_ratio = min(1.0, excess / total_room)
                    for s, room in adj_room.items():
                        old_w = planned_exec_w[s]
                        new_abs = max(floor_abs_w[s], abs(old_w) - room * cut_ratio)
                        planned_exec_w[s] = math.copysign(new_abs, old_w) if new_abs > 1e-12 else 0.0

            for s in active_bars:
                if abs(planned_exec_w[s] - pre_lev_w[s]) > 1e-9:
                    self._ledger.suppressed.append(
                        SuppressedOrder(ts_ms, s, pre_lev_w[s], planned_exec_w[s], "POST_LOCK_LEVERAGE_CAP")
                    )
                    self._pending_weights[s] = planned_exec_w[s]
                    had_override_this_bar = True
                    self._had_any_fill_override = True

        # Pass 2: Execute resolved open fills & record costs / ledger trades
        for sym, bar in active_bars.items():
            spec = self.specs[sym]
            t_plus = max(0, int(spec.calendar.t_plus_days))
            prev_w = self._active_weights.get(sym, 0.0)
            exec_w = planned_exec_w[sym]
            prev_notional_open = prev_w * eq_start * (1.0 + gap_rets.get(sym, 0.0))
            delta_w = exec_w - prev_w

            if abs(delta_w) > 1e-9 and eq_open > 1e-9 and bar.open > 0.0:
                target_notional_open = exec_w * eq_open
                notional_traded = abs(target_notional_open - prev_notional_open)
                if notional_traded > 1e-9:
                    is_sell = target_notional_open < prev_notional_open
                    fee_usd, slip_usd, stamp_usd = _decompose_one_way_costs(
                        spec,
                        self.execution_mode,
                        is_sell=is_sell,
                        extra_slippage_bps=self.extra_slippage_bps_per_leg,
                        notional_usd=notional_traded,
                    )
                    cost_usd = fee_usd + slip_usd + stamp_usd
                    rem_eq_for_cost = max(0.0, eq_open - trade_cost_bar)
                    if cost_usd > rem_eq_for_cost and cost_usd > 1e-12:
                        c_scale = rem_eq_for_cost / cost_usd
                        fee_usd *= c_scale
                        slip_usd *= c_scale
                        stamp_usd *= c_scale
                        cost_usd = fee_usd + slip_usd + stamp_usd
                    trade_cost_bar += cost_usd
                    self._total_fee_usd += fee_usd
                    self._total_slippage_usd += slip_usd
                    self._total_stamp_duty_usd += stamp_usd
                    self._ledger.total_fee_usd += fee_usd
                    self._ledger.total_slippage_usd += slip_usd
                    self._ledger.total_stamp_duty_usd += stamp_usd
                    self._by_symbol_net_usd[sym] = self._by_symbol_net_usd.get(sym, 0.0) - cost_usd
                    self._total_turnover += notional_traded / max(eq_open, 1e-9)

                    mult = _resolve_contract_multiplier(spec, bar)
                    contracts = notional_traded / max(bar.open * mult, 1e-12)
                    side = "SELL" if is_sell else "BUY"
                    if abs(prev_w) <= 1e-9 and abs(exec_w) > 1e-9:
                        trade_reason = "OPEN_ENTRY"
                        self._trade_count += 1
                    elif abs(prev_w) > 1e-9 and abs(exec_w) <= 1e-9:
                        trade_reason = "OPEN_EXIT"
                        self._trade_count += 1
                    elif prev_w * exec_w < 0.0:
                        trade_reason = "OPEN_FLIP"
                        self._trade_count += 1
                    else:
                        trade_reason = "OPEN_REBALANCE"

                    self._ledger.executed.append(
                        ExecutedTrade(
                            ts_fill_ms=ts_ms,
                            symbol=sym,
                            side=side,
                            prev_weight=prev_w,
                            new_weight=exec_w,
                            delta_weight=delta_w,
                            fill_price=bar.open,
                            notional_usd=notional_traded,
                            contracts=contracts,
                            contract_multiplier=mult,
                            fee_usd=fee_usd,
                            slippage_usd=slip_usd,
                            stamp_duty_usd=stamp_usd,
                            total_cost_usd=cost_usd,
                            reason=trade_reason,
                        )
                    )

                self._active_weights[sym] = exec_w
                held_notionals_open[sym] = target_notional_open
                traded_symbols_at_open.add(sym)

                # Reset exchange trading-day counter on new/increased long purchase
                if t_plus > 0 and exec_w > prev_w + 1e-9:
                    self._trading_days_since_buy[sym] = 0
            elif abs(delta_w) <= 1e-9:
                self._active_weights[sym] = exec_w
                held_notionals_open[sym] = prev_notional_open if abs(exec_w) > 1e-9 else 0.0
            else:
                # Unfillable bar (e.g., bar.open <= 0.0): preserve prev_w rather than writing a ghost weight
                self._active_weights[sym] = prev_w
                held_notionals_open[sym] = prev_notional_open if abs(prev_w) > 1e-9 else 0.0

        # Track non-printing holiday positions' notional for gross leverage checks
        for sym, w in self._active_weights.items():
            if sym not in held_notionals_open and abs(w) > 1e-9:
                held_notionals_open[sym] = w * eq_start

        if had_override_this_bar:
            self._had_any_fill_override = True
            self.strategy.sync_executed_weights(dict(self._active_weights))

        # ------------------------------------------------------------------
        # STEP C: Intrabar Stop, Leveraged Liquidation, Open->Close PnL & Carry
        # ------------------------------------------------------------------
        intra_bar_pnl = -trade_cost_bar
        stopped_symbols: set[str] = set()

        # C1. Conservative intrabar stop checks (enforcing both T+1 and PRICE_LIMIT locks!)
        for sym, bar in active_bars.items():
            held_w = self._active_weights.get(sym, 0.0)
            stop_px = self._active_stop_prices.get(sym)
            if abs(held_w) <= 1e-9 or stop_px is None or bar.open <= 0.0:
                continue
            spec = self.specs[sym]
            t_plus = max(0, int(spec.calendar.t_plus_days))
            stop_hit = (held_w > 1e-9 and bar.low <= stop_px) or (held_w < -1e-9 and bar.high >= stop_px)
            if not stop_hit:
                continue
            if t_plus > 0 and held_w > 1e-9 and self._trading_days_since_buy.get(sym, 0) < t_plus:
                self._ledger.suppressed.append(
                    SuppressedOrder(ts_ms, sym, 0.0, held_w, "T_PLUS_1_STOP_LOCK")
                )
                had_override_this_bar = True
                self._had_any_fill_override = True
                continue

            # Enforce 1-tick price-limit lock on intrabar stops (FIX-5)
            prev_b = self._last_bar.get(sym)
            lim_pct = spec.calendar.daily_price_limit_pct
            if (
                lim_pct is not None
                and prev_b is not None
                and prev_b.close > 0.0
                and abs(bar.high - bar.low) <= 1e-9 * max(1.0, abs(bar.open))
            ):
                move_pct = ((bar.open / prev_b.close) - 1.0) * 100.0
                limit_dn_stop = held_w > 1e-9 and move_pct <= -(lim_pct - 0.15)
                limit_up_stop = held_w < -1e-9 and move_pct >= (lim_pct - 0.15)
                if limit_dn_stop or limit_up_stop:
                    self._ledger.suppressed.append(
                        SuppressedOrder(
                            ts_ms,
                            sym,
                            0.0,
                            held_w,
                            "PRICE_LIMIT_DOWN_STOP_LOCK" if limit_dn_stop else "PRICE_LIMIT_UP_STOP_LOCK",
                        )
                    )
                    had_override_this_bar = True
                    self._had_any_fill_override = True
                    continue

            fill_px = min(bar.open, stop_px) if held_w > 0.0 else max(bar.open, stop_px)
            notional = held_notionals_open.get(sym, held_w * eq_open)
            stop_ret = (fill_px / bar.open) - 1.0
            stop_pnl = notional * stop_ret
            exit_notional = abs(notional * (1.0 + stop_ret))
            is_sell = held_w > 0.0
            fee_usd, slip_usd, stamp_usd = _decompose_one_way_costs(
                spec,
                self.execution_mode,
                is_sell=is_sell,
                extra_slippage_bps=self.extra_slippage_bps_per_leg,
                notional_usd=exit_notional,
            )
            cost_usd = fee_usd + slip_usd + stamp_usd
            avail_before_stop = max(0.0, eq_open + intra_bar_pnl)
            if stop_pnl - cost_usd < -avail_before_stop:
                if stop_pnl < 0.0:
                    stop_pnl = max(-avail_before_stop, stop_pnl)
                rem_after_pnl = max(0.0, avail_before_stop + stop_pnl)
                if cost_usd > rem_after_pnl and cost_usd > 1e-12:
                    c_scale = rem_after_pnl / cost_usd
                    fee_usd *= c_scale
                    slip_usd *= c_scale
                    stamp_usd *= c_scale
                    cost_usd = fee_usd + slip_usd + stamp_usd

            self._intraday_pnl += stop_pnl
            self._gross_price_pnl += stop_pnl
            self._total_fee_usd += fee_usd
            self._total_slippage_usd += slip_usd
            self._total_stamp_duty_usd += stamp_usd
            self._ledger.total_fee_usd += fee_usd
            self._ledger.total_slippage_usd += slip_usd
            self._ledger.total_stamp_duty_usd += stamp_usd
            intra_bar_pnl += (stop_pnl - cost_usd)
            self._by_symbol_net_usd[sym] = self._by_symbol_net_usd.get(sym, 0.0) + stop_pnl - cost_usd
            self._total_turnover += exit_notional / max(eq_open, 1e-9)
            self._trade_count += 1

            mult = _resolve_contract_multiplier(spec, bar)
            contracts = abs(notional) / max(bar.open * mult, 1e-12)
            self._ledger.executed.append(
                ExecutedTrade(
                    ts_fill_ms=ts_ms,
                    symbol=sym,
                    side="SELL" if is_sell else "BUY",
                    prev_weight=held_w,
                    new_weight=0.0,
                    delta_weight=-held_w,
                    fill_price=fill_px,
                    notional_usd=exit_notional,
                    contracts=contracts,
                    contract_multiplier=mult,
                    fee_usd=fee_usd,
                    slippage_usd=slip_usd,
                    stamp_duty_usd=stamp_usd,
                    total_cost_usd=cost_usd,
                    reason="INTRABAR_STOP",
                )
            )
            self._active_weights[sym] = 0.0
            self._pending_weights[sym] = 0.0
            self._active_stop_prices[sym] = None
            held_notionals_open[sym] = 0.0
            stopped_symbols.add(sym)
            had_override_this_bar = True
            self._had_any_fill_override = True

        # C2. Leveraged/short vs unleveraged-long intrabar liquidation check (D15c, FIX-4, FIX-6, FIX-7)
        gross_held_lev = math.fsum(abs(w) for w in self._active_weights.values())
        lev_syms_intra: list[str] = []
        lev_adverse_pnl = 0.0
        lev_notional_sum = 0.0
        for sym, bar in active_bars.items():
            if sym in stopped_symbols:
                continue
            held_w = self._active_weights.get(sym, 0.0)
            if abs(held_w) <= 1e-9 or bar.open <= 0.0:
                continue
            spec = self.specs[sym]
            if _is_leveraged_exposure(spec, gross_held_lev, held_w):
                notional = held_notionals_open.get(sym, held_w * eq_open)
                worst_px = bar.low if held_w > 0.0 else bar.high
                adv = notional * ((worst_px / bar.open) - 1.0)
                wick_notional = abs(notional * (worst_px / bar.open))
                lev_adverse_pnl += adv
                lev_notional_sum += wick_notional
                lev_syms_intra.append(sym)

        liquidated_symbols: set[str] = set()
        if lev_syms_intra and (eq_open + intra_bar_pnl + lev_adverse_pnl) <= (lev_notional_sum * self.maint_margin_rate):
            self._liquidation_count += 1
            had_override_this_bar = True
            self._had_any_fill_override = True
            available_for_liq = max(0.0, eq_open + intra_bar_pnl)
            # Conservatively realize adverse move down to floor of available equity
            realized_liq_loss = max(-available_for_liq, lev_adverse_pnl)
            scale_loss = (realized_liq_loss / lev_adverse_pnl) if abs(lev_adverse_pnl) > 1e-12 else 1.0
            rem_eq = max(0.0, available_for_liq + realized_liq_loss)
            for sym in lev_syms_intra:
                bar = active_bars[sym]
                spec = self.specs[sym]
                held_w = self._active_weights[sym]
                notional = held_notionals_open.get(sym, held_w * eq_open)
                worst_px = bar.low if held_w > 0.0 else bar.high
                wick_notional = abs(notional * (worst_px / bar.open))
                sym_loss = notional * ((worst_px / bar.open) - 1.0) * scale_loss
                liq_fee = min(rem_eq, wick_notional * (self.liquidation_fee_bps / 10_000.0))
                liq_fee = max(0.0, liq_fee)
                rem_eq = max(0.0, rem_eq - liq_fee)

                self._intraday_pnl += sym_loss
                self._gross_price_pnl += sym_loss
                self._total_fee_usd += liq_fee
                self._ledger.total_fee_usd += liq_fee
                intra_bar_pnl += (sym_loss - liq_fee)
                self._by_symbol_net_usd[sym] = self._by_symbol_net_usd.get(sym, 0.0) + sym_loss - liq_fee
                self._total_turnover += wick_notional / max(eq_open, 1e-9)
                self._trade_count += 1

                mult = _resolve_contract_multiplier(spec, bar)
                contracts = abs(notional) / max(bar.open * mult, 1e-12)
                self._ledger.executed.append(
                    ExecutedTrade(
                        ts_fill_ms=ts_ms,
                        symbol=sym,
                        side="SELL" if held_w > 0.0 else "BUY",
                        prev_weight=held_w,
                        new_weight=0.0,
                        delta_weight=-held_w,
                        fill_price=worst_px,
                        notional_usd=wick_notional,
                        contracts=contracts,
                        contract_multiplier=mult,
                        fee_usd=liq_fee,
                        slippage_usd=0.0,
                        stamp_duty_usd=0.0,
                        total_cost_usd=liq_fee,
                        reason="INTRABAR_LIQUIDATION",
                    )
                )
                self._active_weights[sym] = 0.0
                self._pending_weights[sym] = 0.0
                self._active_stop_prices[sym] = None
                held_notionals_open[sym] = 0.0
                liquidated_symbols.add(sym)

        # C3. Open->Close price PnL + per-symbol calendar funding & carry drag (D15b) with exact equity-floor conservation
        c3_entries: list[list[Any]] = []
        for sym, bar in active_bars.items():
            if sym in stopped_symbols or sym in liquidated_symbols:
                continue
            held_w = self._active_weights.get(sym, 0.0)
            if abs(held_w) <= 1e-9 or bar.open <= 0.0:
                continue
            spec = self.specs[sym]
            notional = held_notionals_open.get(sym, held_w * eq_open)
            intra_ret = (bar.close / bar.open) - 1.0
            p_usd = notional * intra_ret
            # Perp/futures periodic funding: positive carry_rate_period means LONGS pay SHORTS
            carry_usd = -notional * float(bar.carry_rate_period)
            # Holding drag annualized strictly by THIS symbol's own calendar (D15b)
            sym_bpy = spec.calendar.bars_per_year(float(bar.timeframe_h))
            drag_usd = -abs(notional) * (
                (float(spec.cost_model.annual_carry_drag_bps) / 10_000.0) / max(1.0, sym_bpy)
            )
            c3_entries.append([sym, p_usd, carry_usd, drag_usd])

        if c3_entries:
            pre_c3_eq = eq_open + intra_bar_pnl
            raw_c3_total = math.fsum(e[1] + e[2] + e[3] for e in c3_entries)
            if pre_c3_eq + raw_c3_total < 0.0:
                pos_c3 = math.fsum(v for e in c3_entries for v in (e[1], e[2], e[3]) if v > 0.0)
                neg_c3_abs = math.fsum(-v for e in c3_entries for v in (e[1], e[2], e[3]) if v < 0.0)
                c3_budget = max(0.0, pre_c3_eq + pos_c3)
                c3_scale = (c3_budget / neg_c3_abs) if neg_c3_abs > 1e-12 else 0.0
                for e in c3_entries:
                    if e[1] < 0.0:
                        e[1] *= c3_scale
                    if e[2] < 0.0:
                        e[2] *= c3_scale
                    if e[3] < 0.0:
                        e[3] *= c3_scale

            for sym, p_usd, carry_usd, drag_usd in c3_entries:
                self._intraday_pnl += p_usd
                self._gross_price_pnl += p_usd
                self._funding_pnl += carry_usd
                self._holding_drag_pnl += drag_usd
                self._ledger.total_funding_usd += carry_usd
                self._ledger.total_holding_drag_usd += drag_usd
                intra_bar_pnl += p_usd + carry_usd + drag_usd
                self._by_symbol_net_usd[sym] = (
                    self._by_symbol_net_usd.get(sym, 0.0) + p_usd + carry_usd + drag_usd
                )

        # Update end-of-bar equity, drawdown, and active position diagnostics
        self._equity = max(0.0, eq_open + intra_bar_pnl)
        if self._equity > self._peak_eq:
            self._peak_eq = self._equity
        dd = ((self._peak_eq - self._equity) / self._peak_eq) * 100.0 if self._peak_eq > 0.0 else 100.0
        if dd > self._max_dd:
            self._max_dd = dd

        # Halt cleanly if equity drained to <= 1e-6 (including fee-only wipeouts -> zero ghost weights!)
        if self._equity <= 1e-6:
            self._halted = True
            had_override_this_bar = True
            self._had_any_fill_override = True
            for s in self._active_weights:
                self._active_weights[s] = 0.0
                self._pending_weights[s] = 0.0
                self._active_stop_prices[s] = None
            for sym, bar in active_bars.items():
                prev_b = self._last_bar.get(sym)
                if prev_b is not None and prev_b.close > 0.0 and bar.close > 0.0:
                    self._recent_returns[sym].append((bar.close / prev_b.close) - 1.0)
                self._last_bar[sym] = bar
            self.strategy.sync_executed_weights(dict(self._active_weights))
            zero_w = {s: 0.0 for s in sorted(self._active_weights)}
            empty_decs_h: dict[str, BarDecision] = {}
            self._equity_curve.append((ts_ms, self._equity))
            self._timestamps_ms.append(ts_ms)
            self._executed_weights_history.append(dict(zero_w))
            self._raw_decisions_history.append(empty_decs_h)
            self._allocated_weights_history.append((ts_ms, dict(zero_w)))
            if idx >= self.strategy.warmup_bars:
                self._active_pos_obs += 1
                ret_bar = (self._equity - eq_start) / eq_start if eq_start > 1e-9 else 0.0
                self._period_returns.append(ret_bar)
                self._period_timestamps_ms.append(ts_ms)
            return BarStepResult(ts_ms, self._equity, empty_decs_h, dict(zero_w), dict(zero_w), True)

        active_count = sum(1 for w in self._active_weights.values() if abs(w) > 1e-9)
        gross_lev = math.fsum(abs(w) for w in self._active_weights.values())
        if active_count > self._max_active_positions:
            self._max_active_positions = active_count
        if gross_lev > self._max_gross_leverage:
            self._max_gross_leverage = gross_lev
        if idx >= self.strategy.warmup_bars:
            self._sum_active_positions += active_count
            self._active_pos_obs += 1
            ret_bar = (self._equity - eq_start) / eq_start if eq_start > 1e-9 else 0.0
            self._period_returns.append(ret_bar)
            self._period_timestamps_ms.append(ts_ms)

        self._equity_curve.append((ts_ms, self._equity))
        self._timestamps_ms.append(ts_ms)
        executed_snapshot = dict(self._active_weights)
        self._executed_weights_history.append(executed_snapshot)

        if had_override_this_bar and (stopped_symbols or liquidated_symbols):
            self.strategy.sync_executed_weights(dict(self._active_weights))

        # ------------------------------------------------------------------
        # STEP D: Update Causal Returns -> Strategy Decision -> RiskAllocator -> Weight Sync
        # ------------------------------------------------------------------
        for sym, bar in active_bars.items():
            prev_b = self._last_bar.get(sym)
            if prev_b is not None and prev_b.close > 0.0 and bar.close > 0.0:
                self._recent_returns[sym].append((bar.close / prev_b.close) - 1.0)
            self._last_bar[sym] = bar

        raw_decisions = self.strategy.on_bar_step(ts_ms, active_bars, context=context)
        for sym, dec in raw_decisions.items():
            self._last_raw_signal[sym] = float(dec.raw_signal)

        non_printing_locked = {
            s: float(w)
            for s, w in self._active_weights.items()
            if s not in raw_decisions and abs(w) > 1e-9
        }

        alloc_w, suppressions = self.risk_allocator.allocate_with_audit(
            ts_ms,
            raw_decisions,
            self._recent_returns,
            self.specs,
            locked_weights=non_printing_locked,
            timeframe_h_by_symbol=dict(self._seen_timeframes),
        )

        # Record REA-001 Tier 1 candidates and Tier 3 allocator suppressions
        for sym in sorted(raw_decisions):
            dec = raw_decisions[sym]
            self._ledger.candidates.append(
                CandidateRecord(
                    ts_ms=ts_ms,
                    symbol=sym,
                    raw_signal=float(dec.raw_signal),
                    raw_target_weight=float(dec.target_weight),
                    allocated_weight=float(alloc_w.get(sym, 0.0)),
                    regime_label=str(dec.regime_label),
                    stop_price=dec.stop_price,
                )
            )
        for sym, req_w, eff_w, reason in suppressions:
            self._ledger.suppressed.append(
                SuppressedOrder(
                    ts_ms=ts_ms,
                    symbol=sym,
                    requested_weight=float(req_w),
                    effective_weight=float(eff_w),
                    reason=str(reason),
                )
            )

        # Feed COMPLETE portfolio weight map (held non-printing + newly allocated) back into strategy state (FIX-1 / G7)
        full_synced = dict(self._active_weights)
        full_synced.update(alloc_w)
        self.strategy.sync_executed_weights(dict(full_synced))

        # Queue allocated weights & stops for NEXT bar open execution (zero lookahead!)
        for sym in sorted(raw_decisions):
            self._pending_weights[sym] = float(alloc_w.get(sym, 0.0))
            self._active_stop_prices[sym] = raw_decisions[sym].stop_price

        self._raw_decisions_history.append(dict(raw_decisions))
        self._allocated_weights_history.append((ts_ms, dict(full_synced)))

        return BarStepResult(
            ts_ms=ts_ms,
            equity_usd=self._equity,
            raw_decisions=dict(raw_decisions),
            allocated_weights=dict(full_synced),
            executed_weights=executed_snapshot,
            had_fill_override=had_override_this_bar,
        )

    def finalize(self) -> SimulationResult:
        """Compute final D13 CAGR, Sharpe, Calmar, vol, and return complete `SimulationResult`."""
        n_ts = len(self._timestamps_ms)
        if n_ts < 2:
            days = 1.0
        else:
            days = max(1.0, (self._timestamps_ms[-1] - self._timestamps_ms[0]) / 86_400_000.0)

        # Portfolio bars_per_year: exact calendar if homogeneous, empirical if mixed calendars (D15b)
        cal_ids = {
            self.specs[s].calendar.calendar_id
            for s in self._seen_timeframes
            if s in self.specs
        }
        tfs = {round(tf, 6) for tf in self._seen_timeframes.values()}
        if len(cal_ids) == 1 and len(tfs) == 1 and self._seen_timeframes:
            any_sym = next(iter(self._seen_timeframes))
            bpy = self.specs[any_sym].calendar.bars_per_year(self._seen_timeframes[any_sym])
        elif len(self._period_timestamps_ms) >= 2:
            active_days = max(
                1.0,
                (self._period_timestamps_ms[-1] - self._period_timestamps_ms[0]) / 86_400_000.0,
            )
            bpy = max(1.0, len(self._period_returns) * (365.0 / active_days))
        elif self.specs:
            any_spec = next(iter(self.specs.values()))
            bpy = any_spec.calendar.bars_per_year(24.0)
        else:
            bpy = 252.0

        final_eq = self._equity
        net_pnl = final_eq - self.capital_usd
        total_ret_pct = (net_pnl / self.capital_usd) * 100.0 if self.capital_usd > 0.0 else 0.0
        if final_eq > 0.0 and self.capital_usd > 0.0:
            cagr_pct = (((final_eq / self.capital_usd) ** (365.0 / days)) - 1.0) * 100.0
        else:
            cagr_pct = -100.0
        simple_apr_pct = total_ret_pct * (365.0 / days)

        sr = _fast_sharpe(self._period_returns, bpy)
        realized_vol_ann_pct = _fast_sample_std(self._period_returns) * math.sqrt(max(1.0, bpy)) * 100.0
        calmar = (
            cagr_pct / self._max_dd
            if self._max_dd > 0.1
            else (cagr_pct / 0.1 if cagr_pct > 0.0 else 0.0)
        )
        total_cost = self._total_fee_usd + self._total_slippage_usd + self._total_stamp_duty_usd
        funding_carry_pnl = self._funding_pnl + self._holding_drag_pnl
        mean_pos = (
            self._sum_active_positions / self._active_pos_obs
            if self._active_pos_obs > 0
            else 0.0
        )

        return SimulationResult(
            strategy_id=str(getattr(self.strategy, "strategy_id", "unknown")),
            asset_domain=str(getattr(self.strategy, "asset_domain", "unknown")),
            capital_usd=self.capital_usd,
            final_equity_usd=final_eq,
            days=days,
            bars_simulated=len(self._period_returns),
            bars_per_year=bpy,
            trade_count=self._trade_count,
            total_turnover=self._total_turnover,
            gross_price_pnl_usd=self._gross_price_pnl,
            gap_pnl_usd=self._gap_pnl,
            intraday_pnl_usd=self._intraday_pnl,
            funding_pnl_usd=self._funding_pnl,
            holding_drag_usd=self._holding_drag_pnl,
            funding_carry_pnl_usd=funding_carry_pnl,
            total_fee_usd=self._total_fee_usd,
            total_slippage_usd=self._total_slippage_usd,
            total_stamp_duty_usd=self._total_stamp_duty_usd,
            total_cost_usd=total_cost,
            net_pnl_usd=net_pnl,
            total_return_pct=total_ret_pct,
            cagr_pct=cagr_pct,
            simple_apr_pct=simple_apr_pct,
            sharpe_ratio=sr,
            calmar_ratio=calmar,
            realized_annual_vol_pct=realized_vol_ann_pct,
            max_drawdown_pct=self._max_dd,
            max_active_positions=self._max_active_positions,
            mean_active_positions=mean_pos,
            max_gross_leverage=self._max_gross_leverage,
            liquidation_count=self._liquidation_count,
            equity_curve=list(self._equity_curve),
            timestamps_ms=list(self._timestamps_ms),
            period_returns=list(self._period_returns),
            period_timestamps_ms=list(self._period_timestamps_ms),
            by_symbol_net_usd=dict(sorted(self._by_symbol_net_usd.items())),
            raw_decisions_history=list(self._raw_decisions_history),
            allocated_weights_history=list(self._allocated_weights_history),
            executed_weights_history=list(self._executed_weights_history),
            trade_ledger=self._ledger,
        )


class PortfolioAccountingEngine:
    """Asset-agnostic T+1 open-fill portfolio simulator backed by `StreamingAccountingSession`."""

    def __init__(
        self,
        *,
        maint_margin_rate: float = 0.05,
        liquidation_fee_bps: float = 20.0,
    ) -> None:
        self.maint_margin_rate = max(0.005, float(maint_margin_rate))
        self.liquidation_fee_bps = max(0.0, float(liquidation_fee_bps))

    def create_session(
        self,
        strategy: AssetNativeStrategy,
        specs: dict[str, AssetSpec],
        *,
        risk_cfg: RiskAllocatorConfig,
        capital_usd: float = 10_000.0,
        execution_mode: str = "TAKER_TAKER",
        extra_slippage_bps_per_leg: float = 0.0,
    ) -> StreamingAccountingSession:
        """Create a bar-by-bar streaming accounting session (used by both simulate and G7 parity)."""
        return StreamingAccountingSession(
            strategy=strategy,
            specs=specs,
            risk_cfg=risk_cfg,
            capital_usd=capital_usd,
            execution_mode=execution_mode,
            extra_slippage_bps_per_leg=extra_slippage_bps_per_leg,
            maint_margin_rate=self.maint_margin_rate,
            liquidation_fee_bps=self.liquidation_fee_bps,
        )

    def simulate(
        self,
        strategy: AssetNativeStrategy,
        specs: dict[str, AssetSpec],
        bars_by_symbol: dict[str, Sequence[CanonicalBar]],
        *,
        risk_cfg: RiskAllocatorConfig,
        capital_usd: float = 10_000.0,
        execution_mode: str = "TAKER_TAKER",
        extra_slippage_bps_per_leg: float = 0.0,
        context_provider: Callable[[int], dict[str, Any]] | None = None,
    ) -> SimulationResult:
        """Run strictly causal T+1 open-fill multi-asset portfolio simulation."""
        session = self.create_session(
            strategy=strategy,
            specs=specs,
            risk_cfg=risk_cfg,
            capital_usd=capital_usd,
            execution_mode=execution_mode,
            extra_slippage_bps_per_leg=extra_slippage_bps_per_leg,
        )

        bars_by_ts: dict[int, dict[str, CanonicalBar]] = {}
        for sym in sorted(bars_by_symbol):
            if sym not in specs:
                continue
            ordered = sorted(bars_by_symbol[sym], key=lambda b: int(b.ts_event_ms))
            for bar in ordered:
                ts = int(bar.ts_event_ms)
                slot = bars_by_ts.get(ts)
                if slot is None:
                    slot = {}
                    bars_by_ts[ts] = slot
                slot[sym] = bar

        for ts in sorted(bars_by_ts):
            ctx = context_provider(ts) if context_provider is not None else None
            session.step_bar(ts, bars_by_ts[ts], context=ctx)

        return session.finalize()
