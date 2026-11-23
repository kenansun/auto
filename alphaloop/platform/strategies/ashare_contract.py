"""Track A — Isolated CN A-Share Native Strategy Extension Contract (Feature F-20).

Provides `AShareStrategyInterface`, an isolated `AssetNativeStrategy`-compliant base
contract ready for future Shenzhen / Cloud Research Node A-Share alpha modules with
ZERO coupling to Web3 or Gold strategy domains.

Enforced CN A-Share Market Physics:
  - Strict Long-Only (`target_weight >= 0.0`, clamping negative signals to `0.0`)
  - Exchange-Local Trading-Date `T+1` Lock (`Asia/Shanghai` UTC+8; shares purchased on
    trading date D cannot be reduced until trading date > D)
  - Board-Aware Daily Price Limits (`±10%` SSE/SZSE Mainboard default; `±20%` ChiNext
    `300`/`301` & STAR Market `688`/`689`; blocks buys at limit-up and sells at limit-down)
  - Sell-Only Stamp Duty Accounting (`5.0 bps` on sell notional only, `0.0 bps` on buys)
  - First-Class Extension Hook Slots for A-Share microstructure & flow alpha:
      * `score_call_auction(ts_ms, bar, context)` (09:15-09:25 CST auction imbalance)
      * `score_order_flow_imbalance(ts_ms, bar, context)` / `score_ofi(...)` (L2 OFI)
      * `score_theme_rotation(ts_ms, bar, context)` (concept/sector theme heat rotation)
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any

from alphaloop.platform.assets import COST_CN_A_SHARE
from alphaloop.platform.core_engine.interfaces import BarDecision
from alphaloop.platform.data_lake import CanonicalBar

_CST_TZ = timezone(timedelta(hours=8))
_CHINEXT_STAR_PREFIXES: tuple[str, ...] = ("300", "301", "688", "689")
_BSE_PREFIXES: tuple[str, ...] = ("43", "83", "87", "920")


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def cn_exchange_trade_date(ts_ms: int) -> date:
    """Resolve China Standard Time (`Asia/Shanghai`, fixed UTC+8) exchange trading date."""
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=_CST_TZ).date()


def default_board_price_limit_pct(symbol: str) -> float:
    """Infer CN A-Share daily price limit (`±10%` Mainboard, `±20%` ChiNext/STAR, `±30%` BSE)."""
    code = symbol.strip().upper().split(".")[0]
    if code.startswith(_CHINEXT_STAR_PREFIXES):
        return 20.0
    if code.startswith(_BSE_PREFIXES):
        return 30.0
    return 10.0


class AShareStrategyInterface:
    """Isolated CN A-Share native strategy contract enforcing T+1, long-only, limits & stamp duty."""

    def __init__(
        self,
        strategy_id: str = "ashare_native_interface",
        *,
        daily_price_limit_pct: float | None = None,
        sell_stamp_duty_bps: float = COST_CN_A_SHARE.stamp_duty_sell_bps,
        buy_stamp_duty_bps: float = 0.0,
        commission_bps: float = COST_CN_A_SHARE.taker_fee_bps,
        slippage_bps: float = COST_CN_A_SHARE.slippage_bps_per_leg,
        max_single_weight: float = 1.0,
        limit_lock_tolerance_pct: float = 0.15,
        auction_hook_weight: float = 0.35,
        ofi_hook_weight: float = 0.35,
        theme_hook_weight: float = 0.30,
        warmup_bars: int = 0,
    ) -> None:
        if daily_price_limit_pct is not None and (
            not math.isfinite(float(daily_price_limit_pct)) or float(daily_price_limit_pct) <= 0.0
        ):
            raise ValueError(f"daily_price_limit_pct must be > 0, got {daily_price_limit_pct}")
        if sell_stamp_duty_bps < 0.0 or buy_stamp_duty_bps < 0.0:
            raise ValueError("Stamp duty rates must be non-negative")
        if buy_stamp_duty_bps > 0.0:
            raise ValueError("CN A-Share stamp duty is strictly sell-only (buy_stamp_duty_bps must be 0.0)")

        self.strategy_id: str = str(strategy_id)
        self.asset_domain: str = "ashare"
        self.warmup_bars: int = max(0, int(warmup_bars))
        self.daily_price_limit_pct: float | None = (
            float(daily_price_limit_pct) if daily_price_limit_pct is not None else None
        )
        self.sell_stamp_duty_bps: float = float(sell_stamp_duty_bps)
        self.buy_stamp_duty_bps: float = 0.0
        self.commission_bps: float = max(0.0, float(commission_bps))
        self.slippage_bps: float = max(0.0, float(slippage_bps))
        self.max_single_weight: float = max(0.05, float(max_single_weight))
        self.limit_lock_tolerance_pct: float = max(0.0, float(limit_lock_tolerance_pct))
        self.auction_hook_weight: float = float(auction_hook_weight)
        self.ofi_hook_weight: float = float(ofi_hook_weight)
        self.theme_hook_weight: float = float(theme_hook_weight)

        self._executed_weights: dict[str, float] = {}
        self._current_trade_date: dict[str, date] = {}
        self._last_buy_trade_date: dict[str, date] = {}
        self._last_bar_date: dict[str, date] = {}
        self._last_bar_close: dict[str, float] = {}
        self._prev_day_close: dict[str, float] = {}

    def reset(self) -> None:
        """Reset all mutable T+1, reference-close, and executed-weight tracking state."""
        self._executed_weights = {}
        self._current_trade_date = {}
        self._last_buy_trade_date = {}
        self._last_bar_date = {}
        self._last_bar_close = {}
        self._prev_day_close = {}

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        """Synchronize post-allocator/post-fill weights and record same-day T+1 buy locks."""
        for sym, raw_w in executed_weights.items():
            s = str(sym)
            new_w = max(0.0, float(raw_w))
            prev_w = float(self._executed_weights.get(s, 0.0))
            if new_w > prev_w + 1e-9:
                cur_dt = self._current_trade_date.get(s)
                if cur_dt is not None:
                    self._last_buy_trade_date[s] = cur_dt
            self._executed_weights[s] = new_w

    def clone_fresh(self) -> AShareStrategyInterface:
        """Return a brand-new freshly reset instance preserving subclass type and configuration."""
        return type(self)(
            strategy_id=self.strategy_id,
            daily_price_limit_pct=self.daily_price_limit_pct,
            sell_stamp_duty_bps=self.sell_stamp_duty_bps,
            buy_stamp_duty_bps=self.buy_stamp_duty_bps,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            max_single_weight=self.max_single_weight,
            limit_lock_tolerance_pct=self.limit_lock_tolerance_pct,
            auction_hook_weight=self.auction_hook_weight,
            ofi_hook_weight=self.ofi_hook_weight,
            theme_hook_weight=self.theme_hook_weight,
            warmup_bars=self.warmup_bars,
        )

    def resolve_price_limit_pct(self, symbol: str) -> float:
        """Return configured override or board-inferred CN A-Share daily price limit (%)."""
        if self.daily_price_limit_pct is not None:
            return self.daily_price_limit_pct
        return default_board_price_limit_pct(symbol)

    def estimate_one_way_cost_bps(self, *, is_sell: bool) -> float:
        """Return exact one-way cost bps applying stamp duty strictly on sells."""
        stamp = self.sell_stamp_duty_bps if is_sell else 0.0
        return self.commission_bps + self.slippage_bps + stamp

    def estimate_stamp_duty_usd(self, notional_usd: float, *, is_sell: bool) -> float:
        """Compute sell-only CN A-Share stamp duty in currency units (0.0 on buys)."""
        if not is_sell or notional_usd <= 0.0:
            return 0.0
        return float(notional_usd) * (self.sell_stamp_duty_bps / 10_000.0)

    # ------------------------------------------------------------------
    # Explicit A-Share Microstructure & Flow Extension Hook Slots
    # ------------------------------------------------------------------

    def score_call_auction(
        self,
        ts_ms: int,
        bar: CanonicalBar,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Extension hook: 09:15-09:25 CST call-auction imbalance / gap alpha score."""
        _ = (ts_ms, bar)
        if not context:
            return 0.0
        raw = context.get("call_auction_score", context.get("auction_score", 0.0))
        return float(raw) if (raw is not None and math.isfinite(float(raw))) else 0.0

    def score_order_flow_imbalance(
        self,
        ts_ms: int,
        bar: CanonicalBar,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Extension hook: intraday L2 order-flow imbalance (OFI) alpha score."""
        _ = (ts_ms, bar)
        if not context:
            return 0.0
        raw = context.get("ofi_score", context.get("order_flow_imbalance", 0.0))
        return float(raw) if (raw is not None and math.isfinite(float(raw))) else 0.0

    def score_ofi(
        self,
        ts_ms: int,
        bar: CanonicalBar,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Alias for `score_order_flow_imbalance`."""
        return self.score_order_flow_imbalance(ts_ms, bar, context)

    def score_theme_rotation(
        self,
        ts_ms: int,
        bar: CanonicalBar,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Extension hook: concept/sector theme rotation heat & leadership score."""
        _ = (ts_ms, bar)
        if not context:
            return 0.0
        raw = context.get("theme_rotation_score", context.get("theme_score", 0.0))
        return float(raw) if (raw is not None and math.isfinite(float(raw))) else 0.0

    def compute_raw_signal(
        self,
        ts_ms: int,
        bar: CanonicalBar,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Compute raw directional score from explicit context or A-Share extension hooks."""
        if context and "raw_signal" in context and context["raw_signal"] is not None:
            sig = float(context["raw_signal"])
            return sig if math.isfinite(sig) else 0.0
        auc = self.score_call_auction(ts_ms, bar, context)
        ofi = self.score_order_flow_imbalance(ts_ms, bar, context)
        thm = self.score_theme_rotation(ts_ms, bar, context)
        return (
            self.auction_hook_weight * auc
            + self.ofi_hook_weight * ofi
            + self.theme_hook_weight * thm
        )

    def _resolve_desired_weight(
        self,
        sym: str,
        raw_signal: float,
        context: dict[str, Any] | None,
    ) -> float:
        """Map raw signal or explicit context desired_weight to non-negative long-only weight."""
        if context:
            dw_map = context.get("desired_weights")
            if isinstance(dw_map, dict) and sym in dw_map and dw_map[sym] is not None:
                return _clamp(float(dw_map[sym]), 0.0, self.max_single_weight)
            if "desired_weight" in context and context["desired_weight"] is not None:
                return _clamp(float(context["desired_weight"]), 0.0, self.max_single_weight)
        if raw_signal <= 0.0:
            return 0.0
        return _clamp(math.tanh(raw_signal / 3.0) * self.max_single_weight, 0.0, self.max_single_weight)

    def _resolve_prev_day_close(
        self,
        sym: str,
        cur_date: date,
        context: dict[str, Any] | None,
    ) -> float | None:
        if context:
            pc_map = context.get("prev_close_by_symbol")
            if isinstance(pc_map, dict) and sym in pc_map and pc_map[sym] is not None:
                val = float(pc_map[sym])
                if math.isfinite(val) and val > 0.0:
                    return val
            if "prev_close" in context and context["prev_close"] is not None:
                val = float(context["prev_close"])
                if math.isfinite(val) and val > 0.0:
                    return val
        last_dt = self._last_bar_date.get(sym)
        if last_dt is not None and cur_date > last_dt and sym in self._last_bar_close:
            self._prev_day_close[sym] = self._last_bar_close[sym]
        return self._prev_day_close.get(sym)

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        """Enforce long-only, CST T+1 lock, ±10%/±20% limits, and sell-only stamp duty metadata."""
        decisions: dict[str, BarDecision] = {}

        for sym in sorted(bars_at_ts):
            bar = bars_at_ts[sym]
            cur_date = cn_exchange_trade_date(int(bar.ts_event_ms))
            self._current_trade_date[sym] = cur_date

            prev_w = max(0.0, float(self._executed_weights.get(sym, 0.0)))
            ref_prev_close = self._resolve_prev_day_close(sym, cur_date, context)

            auc_score = self.score_call_auction(ts_ms, bar, context)
            ofi_score = self.score_order_flow_imbalance(ts_ms, bar, context)
            theme_score = self.score_theme_rotation(ts_ms, bar, context)
            raw_sig = float(self.compute_raw_signal(ts_ms, bar, context))
            if not math.isfinite(raw_sig):
                raw_sig = 0.0

            # 1. Strict long-only target weight clamp (>= 0.0)
            target_w = self._resolve_desired_weight(sym, raw_sig, context)

            # 2. Exchange-date T+1 sell lock (CST Asia/Shanghai)
            last_buy_dt = self._last_buy_trade_date.get(sym)
            t_plus_1_locked = (
                prev_w > 1e-9
                and last_buy_dt is not None
                and cur_date <= last_buy_dt
                and target_w < prev_w - 1e-9
            )
            if t_plus_1_locked:
                target_w = prev_w

            # 3. Daily Price-Limit Check (±10% Mainboard / ±20% ChiNext & STAR)
            lim_pct = self.resolve_price_limit_pct(sym)
            price_limit_state = "NORMAL"
            move_pct = 0.0
            if ref_prev_close is not None and ref_prev_close > 0.0 and bar.close > 0.0:
                move_pct = ((float(bar.close) / ref_prev_close) - 1.0) * 100.0
                threshold = lim_pct - self.limit_lock_tolerance_pct
                is_limit_up = move_pct >= threshold
                is_limit_down = move_pct <= -threshold
                if is_limit_up and target_w > prev_w + 1e-9:
                    target_w = prev_w
                    price_limit_state = "LIMIT_UP_LOCK"
                elif is_limit_down and target_w < prev_w - 1e-9:
                    target_w = prev_w
                    price_limit_state = "LIMIT_DOWN_LOCK"

            # Final unbreakable long-only invariant
            target_w = max(0.0, min(self.max_single_weight, target_w))
            is_sell_order = target_w < prev_w - 1e-9

            # Update causal close history AFTER decision logic
            self._last_bar_date[sym] = cur_date
            self._last_bar_close[sym] = float(bar.close)

            decisions[sym] = BarDecision(
                symbol=sym,
                ts_event_ms=int(ts_ms),
                target_weight=float(target_w),
                raw_signal=float(raw_sig),
                regime_label="ASHARE_LONG_ONLY",
                stop_price=None,
                metadata={
                    "trade_date_cst": cur_date.isoformat(),
                    "t_plus_1_locked": bool(t_plus_1_locked),
                    "price_limit_state": price_limit_state,
                    "daily_price_limit_pct": round(lim_pct, 2),
                    "move_from_prev_close_pct": round(move_pct, 4),
                    "sell_stamp_duty_bps": round(self.sell_stamp_duty_bps, 4),
                    "estimated_stamp_duty_bps": round(
                        self.sell_stamp_duty_bps if is_sell_order else 0.0, 4
                    ),
                    "estimated_one_way_cost_bps": round(
                        self.estimate_one_way_cost_bps(is_sell=is_sell_order), 4
                    ),
                    "call_auction_score": round(auc_score, 6),
                    "ofi_score": round(ofi_score, 6),
                    "theme_rotation_score": round(theme_score, 6),
                },
            )
        return decisions
