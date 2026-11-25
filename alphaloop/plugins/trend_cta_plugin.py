"""Reference Strategy Plugin #4: Multi-Timeframe Causal Trend CTA (Donchian + Chandelier ATR).

Generalizes Gold & Liquid Crypto Major trend-following (Donchian Channel Breakout + ATR Chandelier
Trailing Stop + Inverse-Volatility Risk Sizing) with institutional zero-lookahead accounting:
  - Deterministic UTC resampling from verified 1h OHLCV bars (incomplete buckets dropped).
  - Strictly causal T+1 open execution (signal computed at closed bar k, filled at bar k+1 open).
  - Strictly causal Chandelier trailing stop (computed through bar k-1 BEFORE testing bar k).
  - Conservative gap & same-bar entry+stop fill rules (min(open, stop) long / max(open, stop) short).
  - Real perpetual funding carry/drag accrued on mark notional (`qty * cur_1h_open`) at every
    settlement timestamp inside `(entry_ts, exit_ts]`.
  - Stop-loss exits ALWAYS pay TAKER fee; G2 MAKER_THEN_TAKER_HEDGE charges taker on all exits;
    immutable physical fee & slippage floors.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
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
from alphaloop.harness.backtester import BacktestResult, TAKER_FEE_BPS
from alphaloop.plugins.base import BaseStrategyPlugin

HOUR_MS = 3600 * 1000

TREND_CTA_BOUNDS: dict[str, tuple[float, float]] = {
    "entry_n": (10.0, 80.0),
    "exit_n": (5.0, 40.0),
    "atr_n": (7.0, 28.0),
    "atr_mult": (1.5, 5.0),
    "ema_filter_n": (0.0, 120.0),
    "risk_per_trade_pct": (0.25, 2.5),
    "max_symbol_leverage": (0.5, 2.5),
    "max_positions": (1.0, 10.0),
}

ALLOWED_TIMEFRAMES_H = (4, 12, 24)

KNOWN_TREND_CTA_PARAM_KEYS: frozenset[str] = frozenset({
    "timeframe_hours",
    "entry_n",
    "exit_n",
    "atr_n",
    "atr_mult",
    "ema_filter_n",
    "risk_per_trade_pct",
    "max_symbol_leverage",
    "cooldown_hours",
    "ref_venue",
})


def validate_trend_cta_params(params: dict[str, Any] | None) -> None:
    """Fail-closed validation blocking unknown/dead param keys and invalid value types (CC-20261123-001 A1)."""
    if params is None:
        return
    if not isinstance(params, dict):
        raise ValueError(f"trend_cta params must be a dict, got {type(params).__name__}")
    unknown = sorted(str(k) for k in params.keys() if k not in KNOWN_TREND_CTA_PARAM_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown/unconsumed trend_cta params keys: {unknown}. "
            f"Allowed keys: {sorted(KNOWN_TREND_CTA_PARAM_KEYS)}"
        )
    for k, v in params.items():
        if k == "ref_venue":
            if not isinstance(v, str):
                raise ValueError(f"trend_cta param 'ref_venue' must be str, got {type(v).__name__} ({v!r})")
        else:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(
                    f"trend_cta param '{k}' must be finite numeric, got {type(v).__name__} ({v!r})"
                )


def parse_cta_params(cfg: StrategyConfig) -> dict[str, Any]:
    """Single-source CTA parameter parser shared identically by evaluate(), on_event(), and Shadow warmup."""
    p = cfg.params or {}
    tf_h = _p_int(p, "timeframe_hours", 12)
    if tf_h not in ALLOWED_TIMEFRAMES_H:
        tf_h = 12
    tf_ms = tf_h * HOUR_MS
    entry_n = max(5, _p_int(p, "entry_n", 30))
    exit_n = max(3, min(entry_n - 1, _p_int(p, "exit_n", 12)))
    atr_n = max(5, _p_int(p, "atr_n", 14))
    atr_mult = max(1.5, _p_float(p, "atr_mult", 3.0))
    ema_n = max(0, _p_int(p, "ema_filter_n", 0))
    cooldown_h = max(0, _p_int(p, "cooldown_hours", 24))
    warmup_bars = max(entry_n, exit_n, atr_n, ema_n) + 1
    return {
        "tf_h": tf_h,
        "tf_ms": tf_ms,
        "entry_n": entry_n,
        "exit_n": exit_n,
        "atr_n": atr_n,
        "atr_mult": atr_mult,
        "ema_n": ema_n,
        "cooldown_h": cooldown_h,
        "warmup_bars": warmup_bars,
    }


@dataclass
class _ResampledBar:
    ts_ms: int        # Bucket open UTC ms
    end_ts_ms: int    # Bucket close UTC ms (ts_ms + tf_ms)
    open: float
    high: float
    low: float
    close: float
    volume_usd: float


@dataclass
class _CTAOpenPosition:
    symbol: str
    direction: str         # LONG | SHORT
    entry_ts_ms: int
    entry_px: float
    qty: float
    entry_notional: float
    slot_margin: float
    entry_fee_usd: float
    entry_slip_usd: float
    peak_px: float         # Highest favorable price known strictly through k-1
    trough_px: float       # Lowest favorable price known strictly through k-1
    active_stop: float     # Chandelier stop computed strictly through k-1
    gross_funding_usd: float = 0.0
    last_funding_ts: int = 0
    max_adverse_bps: float = 0.0


def _p_float(params: dict[str, Any], key: str, default: float) -> float:
    raw = params.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return default
    fv = float(raw)
    return fv if math.isfinite(fv) else default


def _p_int(params: dict[str, Any], key: str, default: int) -> int:
    raw = params.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return default
    fv = float(raw)
    return int(round(fv)) if math.isfinite(fv) else default


def compute_true_range(high: float, low: float, prev_close: float | None) -> float:
    """Standard Wilder True Range including overnight/inter-bar price gaps."""
    hl = max(0.0, float(high) - float(low))
    if prev_close is None or prev_close <= 0.0:
        return hl
    return max(hl, abs(float(high) - float(prev_close)), abs(float(low) - float(prev_close)))


def compute_causal_atr_and_ema(
    bars_hlc: list[tuple[float, float, float]],
    atr_n: int,
    ema_n: int,
) -> tuple[list[float], list[float]]:
    """Compute strictly causal SMA(TrueRange, atr_n) and EMA(close, ema_n) identically for batch & stream."""
    an = max(1, int(atr_n))
    en = max(0, int(ema_n))
    trs: list[float] = []
    atrs: list[float] = []
    emas: list[float] = []
    alpha = (2.0 / (en + 1.0)) if en > 0 else 0.0
    ema_val = 0.0
    for i, (hi, lo, cl) in enumerate(bars_hlc):
        pc = bars_hlc[i - 1][2] if i > 0 else None
        tr = compute_true_range(hi, lo, pc)
        if i == 0:
            ema_val = cl
        else:
            ema_val = (alpha * cl + (1.0 - alpha) * ema_val) if en > 0 else cl
        trs.append(tr)
        if i + 1 >= an:
            atrs.append(sum(trs[i + 1 - an : i + 1]) / float(an))
        else:
            atrs.append(sum(trs) / float(len(trs)))
        emas.append(ema_val)
    return atrs, emas


def compute_cta_entry_notional(
    cfg: StrategyConfig,
    *,
    entry_px: float,
    atr_prev: float,
    cur_gross_notional: float,
) -> tuple[float, float]:
    """Single-kernel inverse-volatility ATR risk sizing. Returns (notional_usd, slot_margin)."""
    params = cfg.params or {}
    atr_mult = max(1.5, _p_float(params, "atr_mult", 3.0))
    risk_pct = max(0.0025, min(0.03, _p_float(params, "risk_per_trade_pct", 1.0) / 100.0))
    max_sym_lev = max(0.5, min(float(cfg.leverage), _p_float(params, "max_symbol_leverage", 1.5)))
    eff_weight = min(float(cfg.per_position_weight), 1.0 / max(1, int(cfg.max_positions)))
    slot_cap = float(cfg.capital_usd) * eff_weight
    if entry_px <= 0.0 or not math.isfinite(entry_px) or not math.isfinite(atr_prev):
        return 0.0, slot_cap
    stop_dist_pct = max(0.01, (atr_mult * max(0.0, atr_prev)) / entry_px)
    risk_notional = (float(cfg.capital_usd) * risk_pct) / stop_dist_pct
    sym_cap_notional = slot_cap * max_sym_lev
    rem_gross = max(0.0, float(cfg.capital_usd) * float(cfg.leverage) - float(cur_gross_notional))
    notional = min(risk_notional, sym_cap_notional, rem_gross)
    if notional < 10.0:
        return 0.0, slot_cap
    return notional, slot_cap


def evaluate_cta_intrabar_risk(
    *,
    direction: str,
    entry_px: float,
    entry_notional: float,
    slot_margin: float,
    active_stop: float,
    bar_high: float,
    bar_low: float,
) -> str | None:
    """Single-kernel intrabar risk guard: MARGIN_LIQUIDATION (>=0.90) checked BEFORE CHANDELIER_STOP."""
    if entry_px <= 0.0:
        return None
    adverse_frac = (
        max(0.0, 1.0 - (bar_low / entry_px))
        if direction == "LONG"
        else max(0.0, (bar_high / entry_px) - 1.0)
    )
    leg_lev = float(entry_notional) / max(1e-6, float(slot_margin))
    if adverse_frac * leg_lev >= 0.90:
        return "MARGIN_LIQUIDATION"
    if active_stop > 0.0 and math.isfinite(active_stop):
        stopped = (bar_low <= active_stop) if direction == "LONG" else (bar_high >= active_stop)
        if stopped:
            return "CHANDELIER_STOP"
    return None


def evaluate_cta_bar_signal(
    *,
    close: float,
    prev_hi: float,
    prev_lo: float,
    prev_ex_hi: float,
    prev_ex_lo: float,
    atr_cur: float,
    ema_cur: float,
    ema_n: int,
    allow_short: bool,
    open_direction: str | None,
) -> tuple[str, float] | None:
    """Single-kernel end-of-bar Donchian + EMA trend-filter signal evaluator."""
    ema_ok_long = (ema_n == 0) or (close > ema_cur)
    ema_ok_short = (ema_n == 0) or (close < ema_cur)
    denom = max(1e-9, float(atr_cur))
    if open_direction == "LONG":
        if allow_short and close < prev_lo and ema_ok_short:
            return "OPEN_SHORT", (prev_lo - close) / denom
        if close < prev_ex_lo:
            return "EXIT", 0.0
        return None
    if open_direction == "SHORT":
        if close > prev_hi and ema_ok_long:
            return "OPEN_LONG", (close - prev_hi) / denom
        if close > prev_ex_hi:
            return "EXIT", 0.0
        return None
    if close > prev_hi and ema_ok_long:
        return "OPEN_LONG", (close - prev_hi) / denom
    if allow_short and close < prev_lo and ema_ok_short:
        return "OPEN_SHORT", (prev_lo - close) / denom
    return None


def resample_ohlcv_1h(bars_1h: list[PriceBar], tf_hours: int) -> list[_ResampledBar]:
    """Deterministically resample 1h bars into fixed UTC buckets, dropping incomplete buckets."""
    tf_h = tf_hours if tf_hours in ALLOWED_TIMEFRAMES_H else 12
    tf_ms = tf_h * HOUR_MS
    buckets: dict[int, list[PriceBar]] = {}
    for b in bars_1h:
        bts = (b.ts_ms // tf_ms) * tf_ms
        buckets.setdefault(bts, []).append(b)

    out: list[_ResampledBar] = []
    for bts in sorted(buckets.keys()):
        grp = sorted(buckets[bts], key=lambda x: x.ts_ms)
        if len(grp) < tf_h:
            continue  # Drop incomplete bucket to prevent lookahead/partial-bar distortion
        out.append(
            _ResampledBar(
                ts_ms=bts,
                end_ts_ms=bts + tf_ms,
                open=grp[0].open,
                high=max(x.high for x in grp),
                low=min(x.low for x in grp),
                close=grp[-1].close,
                volume_usd=sum(x.volume_usd for x in grp),
            )
        )
    return out


class TrendCTAPlugin(BaseStrategyPlugin):
    """Multi-timeframe Donchian Breakout + ATR Chandelier Trailing Stop directional CTA plugin."""

    plugin_id = "trend_cta"
    display_name = "Multi-Timeframe Trend CTA (Donchian + Chandelier)"
    strategy_family = "directional_trend_cta"
    description = (
        "Bidirectional Donchian Channel breakout with causal ATR Chandelier trailing stops, "
        "inverse-volatility sizing, T+1 open fills, and real perpetual funding carry/drag accounting."
    )
    required_data_spec = {
        "tables": ["candles_1h", "funding_rates"],
        "requires_dual_candles_for_live": False,
        "min_history_days": 60,
        "supported_modes": ["single_venue_perp"],
    }

    def default_config(self, symbols: list[str] | None = None, **overrides: Any) -> StrategyConfig:
        valid_fields = set(StrategyConfig.__dataclass_fields__.keys())
        unknown_overrides = sorted(k for k in overrides.keys() if k not in valid_fields)
        if unknown_overrides:
            raise ValueError(
                f"Unknown StrategyConfig override keys for trend_cta: {unknown_overrides}"
            )
        if "params" in overrides:
            validate_trend_cta_params(overrides["params"])
        base = StrategyConfig(
            strategy_id="trend_cta_default",
            version="v1.0",
            family="trend_cta_majors",
            status="DRAFT",
            venue_a="gate",
            venue_b="gate",
            mode="single_venue_perp",
            symbols=list(symbols or ["PAXG", "BTC", "ETH", "SOL", "XRP", "LINK", "AAVE", "SUI", "AVAX", "DOGE", "LTC"]),
            capital_usd=3000.0,
            leverage=2.0,
            max_positions=6,
            per_position_weight=0.1667,
            entry_spread_apr_pct=0.0,
            exit_spread_apr_pct=0.0,
            lookback_hours=120,
            exit_confirm_hours=0,
            min_hold_hours=24,
            max_basis_entry_bps=50.0,
            stop_loss_basis_bps=250.0,
            fee_a_bps=2.0,                # Maker limit entry; stops/taker-hedge pay TAKER_FEE_BPS
            fee_b_bps=0.0,
            slippage_per_leg_bps=2.0,     # Conservative single-leg execution slip
            rebalance_drag_bps_per_month=0.0,
            allow_bidirectional=True,
            plugin_id=self.plugin_id,
            execution_mode="MAKER_MAKER",
            params={
                "timeframe_hours": 12,
                "entry_n": 30,
                "exit_n": 12,
                "atr_n": 14,
                "atr_mult": 3.0,
                "ema_filter_n": 0,
                "risk_per_trade_pct": 1.0,
                "max_symbol_leverage": 1.5,
                "cooldown_hours": 24,
                "ref_venue": "hyperliquid",
            },
        )
        if overrides:
            d = base.to_dict()
            if "params" in overrides and isinstance(overrides["params"], dict):
                merged = dict(d["params"])
                merged.update(overrides["params"])
                overrides = {**overrides, "params": merged}
            d.update(overrides)
            cfg = StrategyConfig.from_dict(d)
            validate_trend_cta_params(cfg.params)
            return cfg
        return base

    def mutable_param_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(TREND_CTA_BOUNDS)

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
        d["mode"] = "single_venue_perp"
        params = dict(d.get("params") or {})

        # Preserve immutable physical friction params from baseline
        d["fee_a_bps"] = max(1.0, float(cfg.fee_a_bps))
        d["slippage_per_leg_bps"] = max(1.0, float(cfg.slippage_per_leg_bps))

        for key, val in changes.items():
            if key == "timeframe_hours" and isinstance(val, int) and not isinstance(val, bool) and val in ALLOWED_TIMEFRAMES_H:
                params["timeframe_hours"] = val
            elif key == "max_positions" and isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
                lo, hi = TREND_CTA_BOUNDS["max_positions"]
                d["max_positions"] = int(round(max(lo, min(hi, float(val)))))
            elif key in TREND_CTA_BOUNDS and isinstance(val, (int, float)) and not isinstance(val, bool):
                fval = float(val)
                if not math.isfinite(fval):
                    continue
                lo, hi = TREND_CTA_BOUNDS[key]
                clamped = max(lo, min(hi, fval))
                if key in ("entry_n", "exit_n", "atr_n", "ema_filter_n"):
                    params[key] = int(round(clamped))
                else:
                    params[key] = round(clamped, 3)

        # Invariant 1: exit_n strictly < entry_n - 2
        entry_n = max(10, _p_int(params, "entry_n", 30))
        exit_n = max(5, min(entry_n - 3, _p_int(params, "exit_n", 12)))
        params["entry_n"] = entry_n
        params["exit_n"] = exit_n
        # Strip any unknown/dead keys defensively and enforce strict param schema
        params = {k: v for k, v in params.items() if k in KNOWN_TREND_CTA_PARAM_KEYS}
        validate_trend_cta_params(params)
        # Invariant 2: per_position_weight cannot exceed 1 / max_positions
        d["per_position_weight"] = round(min(float(d["per_position_weight"]), 1.0 / max(1, int(d["max_positions"]))), 4)
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
        warmup_start_ms: int | None = None,
    ) -> BacktestResult:
        validate_trend_cta_params(cfg.params)
        cp = parse_cta_params(cfg)
        tf_h = cp["tf_h"]
        tf_ms = cp["tf_ms"]
        entry_n = cp["entry_n"]
        exit_n = cp["exit_n"]
        atr_n = cp["atr_n"]
        atr_mult = cp["atr_mult"]
        ema_n = cp["ema_n"]
        cooldown_h = cp["cooldown_h"]
        warmup_bars = cp["warmup_bars"]
        allow_short = bool(cfg.allow_bidirectional)

        use_prewarm = warmup_start_ms is not None and start_ms is not None and int(warmup_start_ms) < int(start_ms)
        load_start_ms = int(warmup_start_ms) if use_prewarm else start_ms
        eval_start_bucket = ((int(start_ms) // tf_ms) * tf_ms) if use_prewarm else None

        symbols = list(cfg.symbols)
        venue = cfg.venue_a
        ds_sha, prov, row_count = store.fingerprint_slice([venue], symbols, load_start_ms, end_ms)

        exec_mode = execution_mode_override or cfg.execution_mode
        taker_bps = max(float(cfg.fee_a_bps), TAKER_FEE_BPS.get(venue, 5.0))
        if exec_mode == "TAKER_TAKER":
            entry_fee_bps = taker_bps
            normal_exit_fee_bps = taker_bps
        elif exec_mode == "MAKER_THEN_TAKER_HEDGE":
            entry_fee_bps = float(cfg.fee_a_bps)
            normal_exit_fee_bps = taker_bps
        else:
            entry_fee_bps = float(cfg.fee_a_bps)
            normal_exit_fee_bps = float(cfg.fee_a_bps)
        # Stop-loss & liquidation exits ALWAYS pay taker fee regardless of execution mode
        stop_exit_fee_bps = taker_bps
        slip_bps = max(0.0, float(cfg.slippage_per_leg_bps) + float(extra_slippage_bps_per_leg))

        # Load & resample 1h bars and real funding settlements per symbol
        resampled_by_sym: dict[str, dict[int, _ResampledBar]] = {}
        ordered_bars_by_sym: dict[str, list[_ResampledBar]] = {}
        bar_idx_by_sym: dict[str, dict[int, int]] = {}
        hourly_open_by_sym: dict[str, dict[int, float]] = {}
        import bisect

        funding_by_sym: dict[str, list[tuple[int, float]]] = {}
        funding_ts_by_sym: dict[str, list[int]] = {}
        all_buckets_set: set[int] = set()

        for sym in symbols:
            c_1h = store.load_candles(venue, sym, load_start_ms, end_ms)
            hourly_open_by_sym[sym] = {c.ts_ms: c.open for c in c_1h}
            rbars = resample_ohlcv_1h(c_1h, tf_h)
            ordered_bars_by_sym[sym] = rbars
            resampled_by_sym[sym] = {b.ts_ms: b for b in rbars}
            bar_idx_by_sym[sym] = {b.ts_ms: i for i, b in enumerate(rbars)}
            all_buckets_set.update(b.ts_ms for b in rbars)

            fbars = store.load_funding(venue, sym, load_start_ms, end_ms)
            f_sorted = sorted((f.ts_ms, f.funding_rate) for f in fbars)
            funding_by_sym[sym] = f_sorted
            funding_ts_by_sym[sym] = [x[0] for x in f_sorted]

        if not all_buckets_set or row_count == 0 or (eval_start_bucket is not None and max(all_buckets_set) < eval_start_bucket):
            return self._empty_result(cfg, start_ms or 0, end_ms or 0, prov, ds_sha)

        # Precompute strictly causal ATR and EMA arrays per symbol via shared single-kernel helper
        atr_arr: dict[str, list[float]] = {}
        ema_arr: dict[str, list[float]] = {}
        for sym in symbols:
            rbars = ordered_bars_by_sym[sym]
            atrs, emas = compute_causal_atr_and_ema(
                [(b.high, b.low, b.close) for b in rbars], atr_n, ema_n
            )
            atr_arr[sym] = atrs
            ema_arr[sym] = emas

        timeline = sorted(all_buckets_set)

        open_pos: dict[str, _CTAOpenPosition] = {}
        pending_actions: dict[str, tuple[str, float]] = {}  # sym -> (action, strength)
        cooldown_until: dict[str, int] = {}
        trades: list[TradeRecord] = []
        equity_curve: list[tuple[int, float]] = []
        util_samples: list[float] = []
        realized_pnl_usd = 0.0
        trade_seq = 0
        liquidation_count = 0

        def _funding_in_range(sym: str, ts_lo_excl: int, ts_hi_incl: int, direction: str, qty: float, fallback_px: float, adverse_only: bool = False) -> float:
            total = 0.0
            h_px = hourly_open_by_sym[sym]
            ts_list = funding_ts_by_sym[sym]
            f_list = funding_by_sym[sym]
            idx = bisect.bisect_right(ts_list, ts_lo_excl)
            n_f = len(f_list)
            while idx < n_f:
                f_ts, f_rate = f_list[idx]
                if f_ts > ts_hi_incl:
                    break
                mark_px = h_px.get(f_ts, fallback_px)
                mark_notional = qty * mark_px
                # Positive funding rate: longs PAY (-), shorts RECEIVE (+)
                cf = (-mark_notional * f_rate) if direction == "LONG" else (mark_notional * f_rate)
                if not (adverse_only and cf > 0.0):
                    total += cf
                idx += 1
            return total

        def _close_pos(sym: str, pos: _CTAOpenPosition, exit_ts: int, exit_px: float, reason: str, fee_bps: float, extra_penalty: float = 0.0) -> None:
            nonlocal realized_pnl_usd, trade_seq
            trade_seq += 1
            exit_notional = pos.qty * exit_px
            price_pnl = (
                (exit_px - pos.entry_px) * pos.qty
                if pos.direction == "LONG"
                else (pos.entry_px - exit_px) * pos.qty
            )
            exit_fee = exit_notional * (fee_bps / 10000.0)
            exit_slip = exit_notional * (slip_bps / 10000.0)
            tot_fee = pos.entry_fee_usd + exit_fee
            tot_slip = pos.entry_slip_usd + exit_slip
            net = price_pnl + pos.gross_funding_usd - tot_fee - tot_slip - extra_penalty
            hold_h = max(0.0, (exit_ts - pos.entry_ts_ms) / float(HOUR_MS))
            trades.append(
                TradeRecord(
                    trade_id=f"CTA-{trade_seq:04d}",
                    symbol=sym,
                    direction=pos.direction,
                    entry_ts_ms=pos.entry_ts_ms,
                    exit_ts_ms=exit_ts,
                    hold_hours=round(hold_h, 2),
                    notional_per_leg_usd=round(pos.entry_notional, 2),
                    entry_spread_apr_pct=0.0,
                    gross_funding_usd=round(pos.gross_funding_usd, 4),
                    basis_pnl_usd=round(price_pnl, 4),
                    fee_usd=round(tot_fee, 4),
                    slippage_usd=round(tot_slip, 4),
                    rebalance_usd=round(extra_penalty, 4),
                    net_pnl_usd=round(net, 4),
                    exit_reason=reason,
                    max_adverse_basis_bps=round(pos.max_adverse_bps, 2),
                )
            )
            realized_pnl_usd += net

        last_seen_close: dict[str, float] = {}
        for bts in timeline:
            if eval_start_bucket is not None and bts < eval_start_bucket:
                for sym in symbols:
                    bar_w = resampled_by_sym[sym].get(bts)
                    if bar_w is not None:
                        last_seen_close[sym] = bar_w.close
                continue
            # --- PHASE 1: Execute pending T+1 open exits/reversals & entries at bar.open ---
            # 1a) Exits / reversals first; preserve unexecuted EXIT/reversal if symbol's bucket was missing (Red-Team #7)
            retained_pending: dict[str, tuple[str, float]] = {}
            for sym in list(pending_actions.keys()):
                act, strength = pending_actions[sym]
                bar = resampled_by_sym[sym].get(bts)
                if bar is None:
                    # Preserve EXIT or reversal for open positions until next available bar
                    if sym in open_pos and (
                        act == "EXIT"
                        or (act == "OPEN_LONG" and open_pos[sym].direction == "SHORT")
                        or (act == "OPEN_SHORT" and open_pos[sym].direction == "LONG")
                    ):
                        retained_pending[sym] = (act, strength)
                    continue
                if sym in open_pos and (
                    act == "EXIT"
                    or (act == "OPEN_LONG" and open_pos[sym].direction == "SHORT")
                    or (act == "OPEN_SHORT" and open_pos[sym].direction == "LONG")
                ):
                    pos = open_pos.pop(sym)
                    pos.gross_funding_usd += _funding_in_range(sym, pos.last_funding_ts, bts, pos.direction, pos.qty, bar.open)
                    _close_pos(sym, pos, bts, bar.open, "DONCHIAN_EXIT" if act == "EXIT" else "DONCHIAN_REVERSE", normal_exit_fee_bps)
                    if act == "EXIT":
                        pending_actions.pop(sym, None)

            # 1b) Entries sorted deterministically by breakout strength
            entry_cands: list[tuple[float, str, str]] = []
            for sym, (act, strength) in list(pending_actions.items()):
                if sym in retained_pending:
                    continue
                if act in ("OPEN_LONG", "OPEN_SHORT"):
                    entry_cands.append((strength, sym, "LONG" if act == "OPEN_LONG" else "SHORT"))
            pending_actions.clear()
            pending_actions.update(retained_pending)
            entry_cands.sort(key=lambda x: (-x[0], x[1]))

            cur_gross = sum(p.entry_notional for p in open_pos.values())
            for _, sym, direction in entry_cands:
                if sym in open_pos or len(open_pos) >= cfg.max_positions:
                    continue
                if bts < cooldown_until.get(sym, 0):
                    continue
                bar = resampled_by_sym[sym].get(bts)
                k = bar_idx_by_sym[sym].get(bts)
                if bar is None or k is None or k < warmup_bars or bar.open <= 0:
                    continue
                # ATR known strictly BEFORE bar k opens (k - 1)
                atr_prev = atr_arr[sym][k - 1]
                notional, slot_cap = compute_cta_entry_notional(
                    cfg,
                    entry_px=bar.open,
                    atr_prev=atr_prev,
                    cur_gross_notional=cur_gross,
                )
                if notional < 10.0:
                    continue

                qty = notional / bar.open
                entry_fee = notional * (entry_fee_bps / 10000.0)
                entry_slip = notional * (slip_bps / 10000.0)
                init_stop = (bar.open - atr_mult * atr_prev) if direction == "LONG" else (bar.open + atr_mult * atr_prev)
                open_pos[sym] = _CTAOpenPosition(
                    symbol=sym,
                    direction=direction,
                    entry_ts_ms=bts,
                    entry_px=bar.open,
                    qty=qty,
                    entry_notional=notional,
                    slot_margin=slot_cap,
                    entry_fee_usd=entry_fee,
                    entry_slip_usd=entry_slip,
                    peak_px=bar.open,
                    trough_px=bar.open,
                    active_stop=init_stop,
                    gross_funding_usd=0.0,
                    last_funding_ts=bts,
                )
                cur_gross += notional

            # --- PHASE 2: Intrabar liquidation & strictly-prior Chandelier stop checks + real funding ---
            for sym in list(open_pos.keys()):
                pos = open_pos[sym]
                bar = resampled_by_sym[sym].get(bts)
                if bar is None:
                    continue

                adverse_frac = (
                    max(0.0, 1.0 - (bar.low / pos.entry_px))
                    if pos.direction == "LONG"
                    else max(0.0, (bar.high / pos.entry_px) - 1.0)
                )
                if -adverse_frac * 10000.0 < pos.max_adverse_bps:
                    pos.max_adverse_bps = -adverse_frac * 10000.0

                risk_evt = evaluate_cta_intrabar_risk(
                    direction=pos.direction,
                    entry_px=pos.entry_px,
                    entry_notional=pos.entry_notional,
                    slot_margin=pos.slot_margin,
                    active_stop=pos.active_stop,
                    bar_high=bar.high,
                    bar_low=bar.low,
                )
                # 2a) Liquidation check FIRST
                if risk_evt == "MARGIN_LIQUIDATION":
                    liquidation_count += 1
                    open_pos.pop(sym)
                    pos.gross_funding_usd += _funding_in_range(
                        sym, pos.last_funding_ts, bar.end_ts_ms, pos.direction, pos.qty, bar.close, adverse_only=True
                    )
                    liq_px = bar.low if pos.direction == "LONG" else bar.high
                    liq_pen = max(pos.slot_margin * 0.10, pos.entry_notional * 0.02)
                    _close_pos(sym, pos, bar.end_ts_ms, liq_px, "MARGIN_LIQUIDATION", stop_exit_fee_bps, extra_penalty=liq_pen)
                    cooldown_until[sym] = bar.end_ts_ms + cooldown_h * HOUR_MS
                    continue

                # 2b) Strictly causal Chandelier stop check (active_stop computed ONLY through k-1!)
                if risk_evt == "CHANDELIER_STOP":
                    open_pos.pop(sym)
                    # Conservative gap fill & REA-001 same-bar entry+stop worst-case exit
                    fill_px = min(bar.open, pos.active_stop) if pos.direction == "LONG" else max(bar.open, pos.active_stop)
                    fill_px = max(1e-9, fill_px)
                    pos.gross_funding_usd += _funding_in_range(
                        sym, pos.last_funding_ts, bar.end_ts_ms, pos.direction, pos.qty, fill_px, adverse_only=True
                    )
                    _close_pos(sym, pos, bar.end_ts_ms, fill_px, "CHANDELIER_STOP", stop_exit_fee_bps)
                    cooldown_until[sym] = bar.end_ts_ms + cooldown_h * HOUR_MS
                    continue

                # 2c) Accrue real perp funding settlements strictly inside (last_funding_ts, bar.end_ts_ms]
                pos.gross_funding_usd += _funding_in_range(
                    sym, pos.last_funding_ts, bar.end_ts_ms, pos.direction, pos.qty, bar.close, adverse_only=False
                )
                pos.last_funding_ts = bar.end_ts_ms

            # --- PHASE 3: End-of-bar stop ratchet + causal Donchian signal scheduling for T+1 open ---
            for sym in symbols:
                bar = resampled_by_sym[sym].get(bts)
                if bar is not None:
                    last_seen_close[sym] = bar.close
                k = bar_idx_by_sym[sym].get(bts)
                if bar is None or k is None or k < warmup_bars:
                    continue
                rbars = ordered_bars_by_sym[sym]
                atr_cur = atr_arr[sym][k]

                # Update Chandelier trailing stop ONLY AFTER bar k closes (for bar k+1)
                if sym in open_pos:
                    pos = open_pos[sym]
                    if pos.direction == "LONG":
                        pos.peak_px = max(pos.peak_px, bar.high)
                        pos.active_stop = max(pos.active_stop, pos.peak_px - atr_mult * atr_cur)
                    else:
                        pos.trough_px = min(pos.trough_px, bar.low)
                        pos.active_stop = min(pos.active_stop, pos.trough_px + atr_mult * atr_cur)

                # Strictly prior channels [k - n, k - 1] (excluding current bar k)
                don_hi = max(rbars[j].high for j in range(k - entry_n, k))
                don_lo = min(rbars[j].low for j in range(k - entry_n, k))
                ex_lo = min(rbars[j].low for j in range(k - exit_n, k))
                ex_hi = max(rbars[j].high for j in range(k - exit_n, k))

                sig = evaluate_cta_bar_signal(
                    close=bar.close,
                    prev_hi=don_hi,
                    prev_lo=don_lo,
                    prev_ex_hi=ex_hi,
                    prev_ex_lo=ex_lo,
                    atr_cur=atr_cur,
                    ema_cur=ema_arr[sym][k],
                    ema_n=ema_n,
                    allow_short=allow_short,
                    open_direction=open_pos[sym].direction if sym in open_pos else None,
                )
                if sig is not None:
                    pending_actions[sym] = sig

            # Continuous mark-to-market equity snapshot at bar close (bts + tf_ms)
            unr = 0.0
            for sym, pos in open_pos.items():
                bar = resampled_by_sym[sym].get(bts)
                cur_px = bar.close if bar else last_seen_close.get(sym, pos.entry_px)
                px_pnl = (cur_px - pos.entry_px) * pos.qty if pos.direction == "LONG" else (pos.entry_px - cur_px) * pos.qty
                est_exit = (pos.qty * cur_px) * ((normal_exit_fee_bps + slip_bps) / 10000.0)
                unr += px_pnl + pos.gross_funding_usd - pos.entry_fee_usd - pos.entry_slip_usd - est_exit
            equity_curve.append((bts + tf_ms, cfg.capital_usd + realized_pnl_usd + unr))
            used_gross = sum(p.entry_notional for p in open_pos.values())
            util_samples.append((used_gross / max(1.0, cfg.capital_usd)) * 100.0)

        # Close remaining positions at end of window
        final_ts = timeline[-1] + tf_ms
        for sym in list(open_pos.keys()):
            pos = open_pos.pop(sym)
            last_bar = ordered_bars_by_sym[sym][-1] if ordered_bars_by_sym[sym] else None
            exit_px = last_bar.close if last_bar else pos.entry_px
            _close_pos(sym, pos, final_ts, exit_px, "END_OF_WINDOW", normal_exit_fee_bps)

        if equity_curve:
            equity_curve[-1] = (final_ts, cfg.capital_usd + realized_pnl_usd)

        report_start_ms = max(timeline[0], eval_start_bucket) if eval_start_bucket is not None else timeline[0]
        metrics = self._compute_metrics(
            cfg=cfg,
            start_ms=report_start_ms,
            end_ms=final_ts,
            trades=trades,
            equity_curve=equity_curve,
            util_samples=util_samples,
            provenance=prov,
            dataset_sha=ds_sha,
            liquidation_count=liquidation_count,
        )
        warnings: list[str] = []
        if liquidation_count > 0:
            warnings.append(f"MARGIN_LIQUIDATION_OCCURRED ({liquidation_count} events)")
        return BacktestResult(
            config=cfg,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            basis_coverage_pct=100.0,
            warnings=warnings,
        )

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
    ) -> BacktestMetrics:
        days = max(1.0 / 24.0, (end_ms - start_ms) / (86400.0 * 1000.0))
        gross_f = sum(t.gross_funding_usd for t in trades)
        price_p = sum(t.basis_pnl_usd for t in trades)
        gross_alpha = gross_f + price_p
        fees = sum(t.fee_usd for t in trades)
        slips = sum(t.slippage_usd for t in trades)
        penalties = sum(t.rebalance_usd for t in trades)
        net_pnl = sum(t.net_pnl_usd for t in trades)

        wins = [t.net_pnl_usd for t in trades if t.net_pnl_usd > 0]
        losses = [-t.net_pnl_usd for t in trades if t.net_pnl_usd < 0]
        win_rate = (len(wins) / len(trades)) if trades else 0.0
        avg_hold = (sum(t.hold_hours for t in trades) / len(trades)) if trades else 0.0
        pf = (sum(wins) / sum(losses)) if sum(losses) > 0 else (99.0 if sum(wins) > 0 else 0.0)

        total_ret_pct = (net_pnl / cfg.capital_usd) * 100.0 if cfg.capital_usd > 0 else 0.0
        ann_apr_pct = total_ret_pct * (365.0 / days)
        monthly_usd = net_pnl * (30.0 / days)

        peak = cfg.capital_usd
        max_dd_pct = 0.0
        for _, eq in equity_curve:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = ((peak - eq) / peak) * 100.0
                if dd > max_dd_pct:
                    max_dd_pct = dd

        daily_eq: dict[int, float] = {}
        for ts, eq in equity_curve:
            daily_eq[ts // (86400 * 1000)] = eq
        day_vals = [daily_eq[k] for k in sorted(daily_eq.keys())]
        daily_rets = [
            (day_vals[i] / day_vals[i - 1]) - 1.0
            for i in range(1, len(day_vals))
            if day_vals[i - 1] > 0
        ]
        if len(daily_rets) >= 3:
            mean_r = sum(daily_rets) / len(daily_rets)
            var_r = sum((r - mean_r) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
            std_r = math.sqrt(var_r)
            sharpe = (mean_r / std_r) * math.sqrt(365.0) if std_r > 1e-12 else 0.0
        else:
            sharpe = 0.0

        calmar = (ann_apr_pct / max_dd_pct) if max_dd_pct > 0.05 else (ann_apr_pct / 0.05 if ann_apr_pct > 0 else 0.0)
        fee_to_gross = ((fees + slips + penalties) / gross_alpha) if gross_alpha > 0 else 99.0
        avg_util = (sum(util_samples) / len(util_samples)) if util_samples else 0.0

        sorted_pnls = sorted((t.net_pnl_usd for t in trades), reverse=True)
        top5_removed = sum(sorted_pnls[5:]) if len(sorted_pnls) > 5 else 0.0

        by_sym: dict[str, dict[str, float]] = {}
        for t in trades:
            b = by_sym.setdefault(t.symbol, {
                "trades": 0.0, "wins": 0.0, "gross_funding_usd": 0.0,
                "basis_pnl_usd": 0.0, "costs_usd": 0.0, "net_pnl_usd": 0.0, "hold_hours": 0.0,
            })
            b["trades"] += 1.0
            if t.net_pnl_usd > 0:
                b["wins"] += 1.0
            b["gross_funding_usd"] += t.gross_funding_usd
            b["basis_pnl_usd"] += t.basis_pnl_usd
            b["costs_usd"] += t.fee_usd + t.slippage_usd + t.rebalance_usd
            b["net_pnl_usd"] += t.net_pnl_usd
            b["hold_hours"] += t.hold_hours

        for st in by_sym.values():
            st["win_rate"] = round(st["wins"] / st["trades"], 4) if st["trades"] else 0.0
            st["avg_hold_hours"] = round(st["hold_hours"] / st["trades"], 2) if st["trades"] else 0.0
            for k in ("gross_funding_usd", "basis_pnl_usd", "costs_usd", "net_pnl_usd"):
                st[k] = round(st[k], 4)

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
            basis_pnl_usd=round(price_p, 4),
            total_fees_usd=round(fees, 4),
            total_slippage_usd=round(slips, 4),
            total_rebalance_usd=round(penalties, 4),
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
            forced_rebalance_count=0,
        )

    def _empty_result(
        self, cfg: StrategyConfig, start_ms: int, end_ms: int, prov: str, ds_sha: str
    ) -> BacktestResult:
        m = BacktestMetrics(
            window_start_ms=start_ms,
            window_end_ms=end_ms,
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
            dataset_sha256=ds_sha,
            gross_alpha_usd=0.0,
        )
        return BacktestResult(config=cfg, metrics=m, trades=[], equity_curve=[], basis_coverage_pct=0.0, warnings=["EMPTY_DATASET"])

    def plateau_variants(self, cfg: StrategyConfig) -> list[tuple[str, StrategyConfig]]:
        p = cfg.params or {}
        base_en = max(10, _p_int(p, "entry_n", 30))
        base_ex = max(5, _p_int(p, "exit_n", 12))
        base_am = max(1.5, _p_float(p, "atr_mult", 3.0))
        variants: list[tuple[str, StrategyConfig]] = []
        for nf in (0.80, 1.0, 1.20):
            for af in (0.80, 1.0, 1.20):
                en = max(10, int(round(base_en * nf)))
                ex = max(5, min(en - 3, int(round(base_ex * nf))))
                am = round(max(1.5, min(5.0, base_am * af)), 2)
                np = dict(p)
                np["entry_n"] = en
                np["exit_n"] = ex
                np["atr_mult"] = am
                cand = dataclasses.replace(cfg, params=np)
                variants.append((f"don={en}/{ex}|atr={am}x", cand))
        return variants

    def diagnose_telemetry(self, cfg: StrategyConfig, metrics: BacktestMetrics) -> FeedbackEvent:
        now = utc_now_ms()
        if metrics.liquidation_count > 0:
            sev, anom = "P0", "MARGIN_LIQUIDATION"
            summary = f"TrendCTA suffered {metrics.liquidation_count} liquidation(s); reduce risk/leverage."
        elif metrics.max_drawdown_pct > 10.0:
            sev, anom = "P0", "DRAWDOWN_BREACH"
            summary = f"TrendCTA MaxDD {metrics.max_drawdown_pct:.2f}% exceeds 10% ceiling."
        elif metrics.trade_count < 5:
            sev, anom = "P1", "LOW_SAMPLE_COUNT"
            summary = f"Only {metrics.trade_count} trades; entry channel too wide for sample floor."
        elif metrics.fee_to_gross_ratio > 0.65:
            sev, anom = "P1", "CHURN_EXCESS"
            summary = f"Costs consume {metrics.fee_to_gross_ratio:.1%} of gross alpha; widen channels/stop."
        elif metrics.net_pnl_usd <= 0.0 or metrics.annualized_apr_pct < 5.0:
            sev, anom = "P1", "EDGE_DECAY"
            summary = f"TrendCTA APR {metrics.annualized_apr_pct:+.2f}% (net ${metrics.net_pnl_usd:+.2f}) below 5% hurdle."
        else:
            sev, anom = "INFO", "HEALTHY"
            summary = f"TrendCTA healthy: APR {metrics.annualized_apr_pct:+.2f}%, Sharpe {metrics.sharpe_ratio:.2f}, MaxDD {metrics.max_drawdown_pct:.2f}%."

        eid = f"FB-{sha256_obj({'s': cfg.strategy_id, 'a': anom, 't': now})[:8]}"
        return FeedbackEvent(
            event_id=eid,
            ts_ms=now,
            strategy_id=cfg.strategy_id,
            severity=sev,
            anomaly_type=anom,
            summary=summary,
            metrics_snapshot={
                "apr_pct": metrics.annualized_apr_pct,
                "sharpe": metrics.sharpe_ratio,
                "max_dd_pct": metrics.max_drawdown_pct,
                "trade_count": metrics.trade_count,
                "fee_ratio": metrics.fee_to_gross_ratio,
                "funding_usd": metrics.gross_funding_usd,
                "price_pnl_usd": metrics.basis_pnl_usd,
            },
        )

    def fallback_proposal(
        self,
        cfg: StrategyConfig,
        feedback: FeedbackEvent,
        metrics: BacktestMetrics,
    ) -> tuple[dict[str, Any], str, str, str]:
        p = cfg.params or {}
        en = _p_int(p, "entry_n", 30)
        ex = _p_int(p, "exit_n", 12)
        am = _p_float(p, "atr_mult", 3.0)
        rp = _p_float(p, "risk_per_trade_pct", 1.0)
        msl = _p_float(p, "max_symbol_leverage", 1.5)

        if feedback.anomaly_type in ("MARGIN_LIQUIDATION", "DRAWDOWN_BREACH"):
            return (
                {"risk_per_trade_pct": max(0.35, round(rp * 0.7, 2)), "max_symbol_leverage": max(0.75, round(msl * 0.75, 2))},
                "Reduce per-trade ATR risk budget and symbol leverage cap to bring MaxDD safely below 10%.",
                "Volatility-scaled downsizing preserves trend Sharpe while compressing left-tail drawdowns.",
                "Reject if MaxDD remains > 10% or liquidation_count > 0.",
            )
        if feedback.anomaly_type == "LOW_SAMPLE_COUNT":
            return (
                {"entry_n": max(14, int(round(en * 0.7))), "exit_n": max(6, int(round(ex * 0.75)))},
                "Shorten Donchian entry/exit lookback to capture medium-horizon trends with sufficient trade sample size.",
                "Increases trade count cleanly without sacrificing breakout momentum persistence.",
                "Reject if trade_count < 10 or cost ratio > 65%.",
            )
        if feedback.anomaly_type == "CHURN_EXCESS":
            return (
                {"entry_n": min(60, en + 8), "exit_n": min(30, ex + 4), "atr_mult": min(4.5, round(am + 0.5, 2))},
                "Widen Donchian channel and Chandelier ATR stop to filter whipsaw noise and cut fee drag.",
                "Reduces false breakouts and taker stop turnover during sideways consolidation.",
                "Reject if fee_to_gross_ratio remains > 65%.",
            )
        return (
            {"entry_n": min(55, max(16, en + 4)), "atr_mult": min(4.2, round(am + 0.3, 2))},
            "Refine breakout horizon and trailing stop cushion to improve trend capture vs whipsaw ratio.",
            "Lets winning multi-day trends breathe longer while capping adverse excursions.",
            "Reject if Sharpe or STRESS net PnL declines relative to baseline.",
        )

    def on_event(
        self,
        event: MarketEvent,
        state: dict[str, Any],
        cfg: StrategyConfig,
    ) -> list[OrderIntent]:
        """Single-kernel streaming step matching TrendCTAPlugin.evaluate() math identically.

        Enforces:
        - Strict param validation (no dead/unconsumed keys).
        - True Range (gap-aware) + SMA(ATR) + incremental EMA trend filter (`ema_filter_n`).
        - Inverse-volatility ATR risk sizing (`compute_cta_entry_notional`) & gross leverage cap.
        - Intrabar `MARGIN_LIQUIDATION` (>=0.90) before `CHANDELIER_STOP` (`evaluate_cta_intrabar_risk`)
          plus `cooldown_hours` after stops/liquidations.
        - Strictly causal T+1 open execution (Phase 1 open -> Phase 2 intrabar risk -> Phase 3 bar close).
        """
        validate_trend_cta_params(cfg.params)
        intents: list[OrderIntent] = []
        sym = event.symbol
        if sym not in cfg.symbols or event.venue != cfg.venue_a:
            return intents

        cp = parse_cta_params(cfg)
        tf_ms = cp["tf_ms"]
        en = cp["entry_n"]
        ex = cp["exit_n"]
        atr_n = cp["atr_n"]
        atr_mult = cp["atr_mult"]
        ema_n = cp["ema_n"]
        cooldown_h = cp["cooldown_h"]
        warmup_bars = cp["warmup_bars"]
        hist_cap = max(en, ex, atr_n) + 5
        alpha = (2.0 / (ema_n + 1.0)) if ema_n > 0 else 0.0

        hist: dict[str, list[tuple[float, float, float]]] = state.setdefault("ohlc_hist", {})
        tr_hist: dict[str, list[float]] = state.setdefault("tr_hist", {})
        ema_state: dict[str, float] = state.setdefault("ema_state", {})
        bars_seen: dict[str, int] = state.setdefault("bars_seen", {})
        active_bucket: dict[str, dict[str, Any]] = state.setdefault("active_bucket", {})
        open_pos: dict[str, dict[str, Any]] = state.setdefault("open_pos", {})
        pending_actions: dict[str, list[Any]] = state.setdefault("pending_actions", {})
        cooldown_until: dict[str, int] = state.setdefault("cooldown_until", {})

        series = hist.setdefault(sym, [])
        tr_s = tr_hist.setdefault(sym, [])
        # Lazily hydrate TR/EMA/count if ohlc_hist was pre-populated by warmup or external state
        if sym not in bars_seen:
            bars_seen[sym] = len(series)
            if series:
                trs_init: list[float] = []
                for idx_i, (b_hi, b_lo, _b_cl) in enumerate(series):
                    pc_i = series[idx_i - 1][2] if idx_i > 0 else None
                    trs_init.append(compute_true_range(b_hi, b_lo, pc_i))
                tr_s[:] = trs_init[-(atr_n + 5) :]
                if sym not in ema_state:
                    _, emas_init = compute_causal_atr_and_ema(series, atr_n, ema_n)
                    ema_state[sym] = emas_init[-1]

        def _current_atr() -> float:
            if not tr_s:
                return 0.0
            window = tr_s[-atr_n:]
            return sum(window) / float(len(window))

        def _execute_pending(
            open_px: float,
            open_ts: int,
            atr_prev: float,
            cur_k: int,
            cooldown_check_ts: int | None = None,
        ) -> None:
            pending = pending_actions.get(sym)
            if not pending:
                return
            act = str(pending[0])
            strength = float(pending[1]) if len(pending) > 1 else 0.0
            if sym in open_pos and (
                act == "EXIT"
                or (act == "OPEN_LONG" and open_pos[sym]["direction"] == "SHORT")
                or (act == "OPEN_SHORT" and open_pos[sym]["direction"] == "LONG")
            ):
                pos = open_pos.pop(sym)
                intents.append(
                    OrderIntent(
                        cfg.strategy_id,
                        sym,
                        "CLOSE",
                        float(pos["notional"]),
                        0.0,
                        "DONCHIAN_EXIT" if act == "EXIT" else "DONCHIAN_REVERSE",
                        open_ts,
                    )
                )
                if act == "EXIT":
                    pending_actions.pop(sym, None)
                    return

            pending_actions.pop(sym, None)
            if act not in ("OPEN_LONG", "OPEN_SHORT"):
                return
            direction = "LONG" if act == "OPEN_LONG" else "SHORT"
            cd_ts = open_ts if cooldown_check_ts is None else int(cooldown_check_ts)
            if (
                sym in open_pos
                or len(open_pos) >= int(cfg.max_positions)
                or cd_ts < int(cooldown_until.get(sym, 0))
                or cur_k < warmup_bars
                or open_px <= 0.0
                or not math.isfinite(open_px)
            ):
                return
            cur_gross = sum(float(p_item["notional"]) for p_item in open_pos.values())
            notional, slot_cap = compute_cta_entry_notional(
                cfg,
                entry_px=open_px,
                atr_prev=atr_prev,
                cur_gross_notional=cur_gross,
            )
            if notional < 10.0:
                return
            init_stop = (open_px - atr_mult * atr_prev) if direction == "LONG" else (open_px + atr_mult * atr_prev)
            open_pos[sym] = {
                "direction": direction,
                "notional": notional,
                "slot_margin": slot_cap,
                "entry_px": open_px,
                "entry_ts_ms": open_ts,
                "peak": open_px,
                "trough": open_px,
                "stop_px": init_stop,
            }
            intents.append(
                OrderIntent(
                    cfg.strategy_id,
                    sym,
                    act,
                    notional,
                    round(strength, 4),
                    "DONCHIAN_LONG_BREAKOUT" if direction == "LONG" else "DONCHIAN_SHORT_BREAKOUT",
                    open_ts,
                )
            )

        def _check_intrabar(
            bar_high: float,
            bar_low: float,
            exit_ts: int,
            cooldown_anchor_ts: int | None = None,
        ) -> bool:
            if sym not in open_pos:
                return False
            pos = open_pos[sym]
            eff_w = min(float(cfg.per_position_weight), 1.0 / max(1, int(cfg.max_positions)))
            slot_m = float(pos.get("slot_margin", float(cfg.capital_usd) * eff_w))
            risk_evt = evaluate_cta_intrabar_risk(
                direction=str(pos["direction"]),
                entry_px=float(pos["entry_px"]),
                entry_notional=float(pos["notional"]),
                slot_margin=slot_m,
                active_stop=float(pos.get("stop_px", 0.0)),
                bar_high=bar_high,
                bar_low=bar_low,
            )
            if risk_evt in ("MARGIN_LIQUIDATION", "CHANDELIER_STOP"):
                intents.append(
                    OrderIntent(cfg.strategy_id, sym, "CLOSE", float(pos["notional"]), 0.0, risk_evt, exit_ts)
                )
                open_pos.pop(sym, None)
                anchor_ts = exit_ts if cooldown_anchor_ts is None else int(cooldown_anchor_ts)
                cooldown_until[sym] = anchor_ts + cooldown_h * HOUR_MS
                return True
            return False

        def _close_bar_phase3(hi: float, lo: float, cl: float) -> float:
            k_cur = int(bars_seen.get(sym, 0))
            prev_close = series[-1][2] if series else None
            tr_k = compute_true_range(hi, lo, prev_close)
            tr_s.append(tr_k)
            if len(tr_s) > atr_n + 5:
                del tr_s[: len(tr_s) - (atr_n + 5)]
            atr_cur = _current_atr()
            if k_cur == 0 or sym not in ema_state:
                ema_cur = cl
            else:
                ema_cur = (alpha * cl + (1.0 - alpha) * float(ema_state[sym])) if ema_n > 0 else cl
            ema_state[sym] = ema_cur

            if k_cur >= warmup_bars and len(series) >= max(en, ex):
                if sym in open_pos:
                    pos = open_pos[sym]
                    if pos["direction"] == "LONG":
                        pos["peak"] = max(float(pos.get("peak", cl)), hi)
                        pos["stop_px"] = max(float(pos.get("stop_px", 0.0)), pos["peak"] - atr_mult * atr_cur)
                    else:
                        pos["trough"] = min(float(pos.get("trough", cl)), lo)
                        pos["stop_px"] = min(float(pos.get("stop_px", float("inf"))), pos["trough"] + atr_mult * atr_cur)

                prev_hi = max(x[0] for x in series[-en:])
                prev_lo = min(x[1] for x in series[-en:])
                prev_ex_lo = min(x[1] for x in series[-ex:])
                prev_ex_hi = max(x[0] for x in series[-ex:])
                sig = evaluate_cta_bar_signal(
                    close=cl,
                    prev_hi=prev_hi,
                    prev_lo=prev_lo,
                    prev_ex_hi=prev_ex_hi,
                    prev_ex_lo=prev_ex_lo,
                    atr_cur=atr_cur,
                    ema_cur=ema_cur,
                    ema_n=ema_n,
                    allow_short=bool(cfg.allow_bidirectional),
                    open_direction=str(open_pos[sym]["direction"]) if sym in open_pos else None,
                )
                if sig is not None:
                    pending_actions[sym] = [sig[0], sig[1]]

            series.append((hi, lo, cl))
            if len(series) > hist_cap:
                del series[: len(series) - hist_cap]
            bars_seen[sym] = k_cur + 1
            return atr_cur

        ts_ms = int(event.ts_ms)
        if event.kind == "CANDLE":
            hi = float(event.payload.get("high", 0.0))
            lo = float(event.payload.get("low", 0.0))
            cl = float(event.payload.get("close", 0.0))
            op = float(event.payload.get("open", cl))
            if op <= 0.0 or not math.isfinite(op):
                op = cl
            if cl <= 0.0 or hi < lo or lo <= 0.0 or not (math.isfinite(hi) and math.isfinite(lo) and math.isfinite(cl)):
                return intents
            bar_end_ts = int(event.payload.get("end_ts_ms", ts_ms + tf_ms))
            k_cur = int(bars_seen.get(sym, 0))
            atr_prev = _current_atr()
            # Phase 1: Execute T+1 open exits/reversals & entries
            _execute_pending(op, ts_ms, atr_prev, k_cur)
            # Phase 2: Intrabar liquidation & prior-bar Chandelier stop
            _check_intrabar(hi, lo, bar_end_ts)
            # Phase 3: End-of-bar stop ratchet + schedule T+1 open signal
            _close_bar_phase3(hi, lo, cl)
            return intents

        if event.kind == "BBO":
            bid = float(event.payload.get("bid", 0.0))
            ask = float(event.payload.get("ask", 0.0))
            if bid <= 0.0 or ask < bid or not (math.isfinite(bid) and math.isfinite(ask)):
                return intents
            mid = (bid + ask) * 0.5
            bts = (ts_ms // tf_ms) * tf_ms
            cur = active_bucket.get(sym)
            if cur is None:
                _check_intrabar(ask, bid, ts_ms, cooldown_anchor_ts=bts + tf_ms)
                # Mark mid-bucket unseeded starts (>= 1h into bucket) as partial so stub bars don't pollute ohlc_hist
                is_partial = (ts_ms - bts) >= HOUR_MS
                active_bucket[sym] = {"bts": float(bts), "high": ask, "low": bid, "close": mid, "partial": is_partial}
                return intents
            if bts < int(cur["bts"]):
                return intents
            if bts == int(cur["bts"]):
                _check_intrabar(ask, bid, ts_ms, cooldown_anchor_ts=bts + tf_ms)
                cur["high"] = max(float(cur["high"]), ask)
                cur["low"] = min(float(cur["low"]), bid)
                cur["close"] = mid
                return intents
            # New timeframe bucket started: close prior bucket ONLY if complete (not a partial cold-start stub),
            # then Phase 1 executes at new bucket open (cooldown tested vs bts), then Phase 2 intrabar check!
            if bool(cur.get("partial", False)):
                atr_closed = _current_atr()
            else:
                atr_closed = _close_bar_phase3(float(cur["high"]), float(cur["low"]), float(cur["close"]))
            active_bucket[sym] = {"bts": float(bts), "high": ask, "low": bid, "close": mid, "partial": False}
            _execute_pending(mid, ts_ms, atr_closed, int(bars_seen.get(sym, 0)), cooldown_check_ts=bts)
            _check_intrabar(ask, bid, ts_ms, cooldown_anchor_ts=bts + tf_ms)
            return intents

        return intents

    def min_economic_kill_horizon_hours(self, cfg: StrategyConfig) -> float:
        tf_h = _p_int(cfg.params or {}, "timeframe_hours", 12)
        return max(48.0, float(tf_h * 6))
