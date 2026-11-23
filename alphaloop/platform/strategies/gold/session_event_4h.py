"""G2: Gold 4h Tactical Breakout + PIT Macro Event-Window Engine (Track G, Feature F-18).

Pre-registered Policy Ablation (`A` / `B` / `C`):
  - Policy `"A"`: Baseline 4h Donchian breakout (`N=12`) + `GoldMacroEngine` directional filter.
  - Policy `"B"`: Event-Whipsaw Shield — blocks fresh entries and shields short-horizon
    tactical exposure to flat inside `[T - 4h, T]` around scheduled FOMC / NFP / CPI prints.
  - Policy `"C"`: Post-Event Expansion — shields pre-release `[T - 4h, T)` whipsaw AND
    enables aligned post-event `[T, T + 8h]` breakout expansion boosted by `gvz_z` & macro.

Strict Causality & Anti-Tombstone Guarantees:
  - ZERO pre-release leakage: NEVER inspects `event.actual` or `event.surprise_z` when
    `ts_ms < event.ts_avail_ms`. Only public schedule times (`ts_release_ms`) are visible
    before `ts_avail_ms`.
  - Session tagging (`LONDON_NY` / `ASIAN` / `OFF_HOURS`) emitted in strict-JSON metadata.
  - Pure directional breakout architecture — strictly ZERO session mean-reversion fading.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Sequence

from alphaloop.platform.core_engine.interfaces import BarDecision
from alphaloop.platform.data_foundation.models import CanonicalBar, MacroEventRelease
from alphaloop.platform.strategies.gold.macro_engine import GoldMacroEngine

_HOUR_MS = 3_600_000
_VALID_POLICIES: frozenset[str] = frozenset({"A", "B", "C"})
_DEFAULT_EVENT_TYPES: tuple[str, ...] = ("FOMC", "NFP", "CPI")


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _sample_std(vals: deque[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mu = math.fsum(vals) / n
    var = math.fsum((x - mu) * (x - mu) for x in vals) / (n - 1)
    return math.sqrt(var) if (math.isfinite(var) and var > 0.0) else 0.0


def _classify_4h_session(ts_ms: int) -> str:
    """Deterministic UTC session tag separating London/NY peak liquidity from Asian/Off-Hours."""
    hour_utc = (int(ts_ms) // _HOUR_MS) % 24
    if 8 <= hour_utc < 20:
        return "LONDON_NY"
    if 0 <= hour_utc < 8:
        return "ASIAN"
    return "OFF_HOURS"


def _extract_macro_field(macro_obj: Any, key: str, default: Any) -> Any:
    if isinstance(macro_obj, dict):
        return macro_obj.get(key, default)
    return getattr(macro_obj, key, default)


class Gold4hEventKernel:
    """G2 Gold 4h Tactical Breakout + PIT FOMC/NFP/CPI Event-Window Engine (Policies A/B/C)."""

    def __init__(
        self,
        policy: str = "B",
        *,
        symbol: str = "GC=F",
        donchian_n: int = 12,
        exit_n: int = 6,
        pre_event_shield_hours: float = 4.0,
        post_event_window_hours: float = 8.0,
        macro_engine: GoldMacroEngine | None = None,
        events: Sequence[MacroEventRelease] | None = None,
        event_types: Sequence[str] = _DEFAULT_EVENT_TYPES,
        allow_short: bool = True,
        target_annual_vol_pct: float = 15.0,
        max_weight: float = 1.0,
        stop_atr_mult: float = 4.5,
        stop_pct_floor: float = 0.035,
        warmup_bars: int = 14,
    ) -> None:
        pol = str(policy).strip().upper()
        if pol not in _VALID_POLICIES:
            raise ValueError(f"Gold4hEventKernel policy must be one of A/B/C, got {policy!r}")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Gold4hEventKernel requires a non-empty symbol")
        if donchian_n < 3 or exit_n < 2:
            raise ValueError(f"Invalid channel lengths: donchian_n={donchian_n}, exit_n={exit_n}")

        self.policy: str = pol
        self.symbol: str = symbol.strip()
        self.strategy_id: str = f"gold_g2_4h_event_policy_{self.policy}"
        self.asset_domain: str = "gold"
        self.warmup_bars: int = max(2, int(warmup_bars))

        self.donchian_n: int = int(donchian_n)
        self.exit_n: int = int(exit_n)
        self.pre_event_shield_hours: float = max(0.0, float(pre_event_shield_hours))
        self.post_event_window_hours: float = max(0.0, float(post_event_window_hours))
        self.macro_engine: GoldMacroEngine | None = macro_engine
        self.events: tuple[MacroEventRelease, ...] = tuple(events) if events else ()
        self.event_types: frozenset[str] = frozenset(str(t).strip().upper() for t in event_types)
        self.allow_short: bool = bool(allow_short)
        self.target_annual_vol_pct: float = float(target_annual_vol_pct)
        self.max_weight: float = max(0.1, float(max_weight))
        self.stop_atr_mult: float = max(1.0, float(stop_atr_mult))
        self.stop_pct_floor: float = max(0.005, float(stop_pct_floor))

        buf_len = max(64, self.donchian_n + 6, self.exit_n + 6)
        self._bar_count: int = 0
        self._highs: deque[float] = deque(maxlen=buf_len)
        self._lows: deque[float] = deque(maxlen=buf_len)
        self._closes: deque[float] = deque(maxlen=buf_len)
        self._log_rets: deque[float] = deque(maxlen=40)
        self._atr14: float | None = None
        self._prev_close: float | None = None
        self._executed_weights: dict[str, float] = {}

    def reset(self) -> None:
        """Reset mutable streaming state."""
        self._bar_count = 0
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._log_rets.clear()
        self._atr14 = None
        self._prev_close = None
        self._executed_weights = {}

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        """Synchronize actual post-allocator/post-fill weights for G7 parity."""
        self._executed_weights = {str(k): float(v) for k, v in executed_weights.items()}

    def clone_fresh(self) -> Gold4hEventKernel:
        """Return a brand-new freshly reset instance with identical parameters."""
        return Gold4hEventKernel(
            policy=self.policy,
            symbol=self.symbol,
            donchian_n=self.donchian_n,
            exit_n=self.exit_n,
            pre_event_shield_hours=self.pre_event_shield_hours,
            post_event_window_hours=self.post_event_window_hours,
            macro_engine=self.macro_engine,
            events=self.events,
            event_types=tuple(sorted(self.event_types)),
            allow_short=self.allow_short,
            target_annual_vol_pct=self.target_annual_vol_pct,
            max_weight=self.max_weight,
            stop_atr_mult=self.stop_atr_mult,
            stop_pct_floor=self.stop_pct_floor,
            warmup_bars=self.warmup_bars,
        )

    def _resolve_macro_and_gvz(
        self,
        bar: CanonicalBar,
        context: dict[str, Any] | None,
    ) -> tuple[float, str, float]:
        macro_obj = context.get("gold_macro") if context else None
        if macro_obj is None and self.macro_engine is not None:
            macro_obj = self.macro_engine.snapshot_as_of(bar.ts_avail_ms)

        macro_comp = 0.0
        regime_lbl = "GOLD_NEUTRAL"
        gvz_z = 0.0
        if macro_obj is not None:
            raw_c = _extract_macro_field(macro_obj, "macro_gold_composite", 0.0)
            if raw_c is not None and math.isfinite(float(raw_c)):
                macro_comp = _clamp(float(raw_c), -1.5, 1.5)
            lbl = str(_extract_macro_field(macro_obj, "regime_label", ""))
            if lbl:
                regime_lbl = lbl
            elif macro_comp >= 0.20:
                regime_lbl = "GOLD_BULL_TAILWIND"
            elif macro_comp <= -0.20:
                regime_lbl = "GOLD_BEAR_HEADWIND"
            raw_g = _extract_macro_field(macro_obj, "gvz_z", 0.0)
            if raw_g is not None and math.isfinite(float(raw_g)):
                gvz_z = float(raw_g)

        if context and "gvz_z" in context and context["gvz_z"] is not None:
            val_g = float(context["gvz_z"])
            if math.isfinite(val_g):
                gvz_z = val_g
        return macro_comp, regime_lbl, gvz_z

    def _inspect_event_windows(
        self,
        ts_ms: int,
        context: dict[str, Any] | None,
    ) -> tuple[bool, bool, str, float]:
        """Classify PIT event window strictly without reading unreleased actual/surprise_z.

        Returns `(in_pre_shield, in_post_expansion, active_event_id, released_surprise_z)`.
        """
        ctx_events: Sequence[MacroEventRelease] = ()
        if context:
            raw_ev = context.get("macro_events")
            if raw_ev is None:
                raw_ev = context.get("events")
            if raw_ev:
                ctx_events = raw_ev

        all_events: list[MacroEventRelease] = list(self.events)
        if ctx_events:
            seen_ids = {ev.event_id for ev in all_events}
            for ev in ctx_events:
                if ev.event_id not in seen_ids:
                    all_events.append(ev)
                    seen_ids.add(ev.event_id)

        if not all_events:
            return False, False, "", 0.0

        pre_ms = int(round(self.pre_event_shield_hours * _HOUR_MS))
        post_ms = int(round(self.post_event_window_hours * _HOUR_MS))
        cur_ts = int(ts_ms)

        in_pre_shield = False
        in_post_expansion = False
        active_event_id = ""
        released_surprise_z = 0.0

        for ev in all_events:
            ev_type = str(ev.event_type).strip().upper()
            if self.event_types and ev_type not in self.event_types:
                continue

            # Scheduled release time is public calendar knowledge; NEVER read actual/surprise_z here!
            t_rel = int(ev.ts_release_ms)
            t_avail = int(ev.ts_avail_ms)

            if self.policy == "B":
                # Policy B: Event-Whipsaw Shield covers [T - 4h, T] inclusive
                if (t_rel - pre_ms) <= cur_ts <= t_rel:
                    in_pre_shield = True
                    active_event_id = str(ev.event_id)
            elif self.policy == "C":
                # Policy C: Shield pre-release [T - 4h, T) strictly before print
                if (t_rel - pre_ms) <= cur_ts < t_rel:
                    in_pre_shield = True
                    active_event_id = str(ev.event_id)
                # Post-event expansion [T, T + 8h]: strictly causal check `cur_ts >= t_avail`
                elif t_rel <= cur_ts <= (t_rel + post_ms) and cur_ts >= t_avail:
                    in_post_expansion = True
                    active_event_id = str(ev.event_id)
                    # PIT-SAFE: Only read released surprise_z strictly AFTER ts_avail_ms!
                    if ev.surprise_z is not None and math.isfinite(float(ev.surprise_z)):
                        released_surprise_z = _clamp(float(ev.surprise_z), -5.0, 5.0)

        return in_pre_shield, in_post_expansion, active_event_id, released_surprise_z

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        """Consume 4h bar close and emit strictly causal next-open tactical Gold decision."""
        decisions: dict[str, BarDecision] = {}
        bar = bars_at_ts.get(self.symbol)
        if bar is None:
            return decisions

        c = float(bar.close)
        h = float(bar.high)
        lo = float(bar.low)
        self._bar_count += 1

        # Strictly causal prior entry & exit Donchian channels BEFORE appending current bar
        if len(self._highs) >= 2:
            k_ent = min(len(self._highs), self.donchian_n)
            k_ext = min(len(self._highs), self.exit_n)
            prior_hi = max(list(self._highs)[-k_ent:])
            prior_lo = min(list(self._lows)[-k_ent:])
            prior_close_hi = max(list(self._closes)[-k_ent:])
            prior_close_lo = min(list(self._closes)[-k_ent:])
            exit_hi = max(list(self._highs)[-k_ext:])
            exit_lo = min(list(self._lows)[-k_ext:])
        else:
            prior_hi, prior_lo = h, lo
            prior_close_hi, prior_close_lo = c, c
            exit_hi, exit_lo = h, lo

        if self._prev_close is not None and self._prev_close > 0.0:
            tr = max(h - lo, abs(h - self._prev_close), abs(lo - self._prev_close))
            self._log_rets.append(math.log(max(c, 1e-9) / self._prev_close))
        else:
            tr = max(h - lo, c * 0.003)
        if self._atr14 is None:
            self._atr14 = tr
        else:
            self._atr14 = (self._atr14 * 13.0 + tr) / 14.0

        self._highs.append(h)
        self._lows.append(lo)
        self._closes.append(c)
        self._prev_close = c

        atr = max(float(self._atr14), c * 0.002)
        session_tag = _classify_4h_session(ts_ms)
        macro_comp, regime_lbl, gvz_z = self._resolve_macro_and_gvz(bar, context)
        in_pre_shield, in_post_expansion, active_ev_id, released_surprise_z = (
            self._inspect_event_windows(ts_ms, context)
        )
        prev_w = float(self._executed_weights.get(self.symbol, 0.0))

        # Warmup phase: strictly flat
        if self._bar_count <= self.warmup_bars:
            decisions[self.symbol] = BarDecision(
                symbol=self.symbol,
                ts_event_ms=int(ts_ms),
                target_weight=0.0,
                raw_signal=0.0,
                regime_label=regime_lbl,
                stop_price=None,
                metadata={
                    "policy": self.policy,
                    "session_tag": session_tag,
                    "event_window_state": "WARMUP",
                    "active_event_id": active_ev_id,
                    "macro_gold_composite": round(macro_comp, 6),
                    "gvz_z": round(gvz_z, 6),
                },
            )
            return decisions

        # Pure directional Donchian breakout signals (ZERO session mean-reversion!)
        eps = max(1e-9, c * 1e-9)
        up_breakout = c > (prior_hi + eps)
        dn_breakout = c < (prior_lo - eps)
        long_exit_hit = c < (exit_lo - eps)
        short_exit_hit = c > (exit_hi + eps)

        # Annualized vol scaling on 4h TradFi bars (~23h/day * 252 / 4 = 1449 bars/yr)
        bpy = (23.0 * 252.0) / max(0.25, float(bar.timeframe_h))
        ann_vol = (
            max(0.06, _sample_std(self._log_rets) * math.sqrt(bpy))
            if len(self._log_rets) >= 5
            else 0.15
        )
        vol_unit = _clamp((self.target_annual_vol_pct / 100.0) / ann_vol, 0.35, 1.0)
        session_mult = 1.0 if session_tag == "LONDON_NY" else (0.85 if session_tag == "ASIAN" else 0.75)

        # Event-window state & policy dispatch
        event_state = "NORMAL"
        desired_w = prev_w
        raw_sig = 0.0

        if self.policy in ("B", "C") and in_pre_shield:
            # Event-Whipsaw Shield: block fresh entries & shield tactical exposure ahead of print
            event_state = "PRE_EVENT_SHIELD"
            desired_w = 0.0
            raw_sig = 0.0
        else:
            # Compute baseline breakout direction filtered by GoldMacroEngine
            if up_breakout and macro_comp >= -0.35:
                macro_boost = 1.0 + 0.25 * max(0.0, min(1.5, macro_comp))
                strength = _clamp(0.55 + 0.45 * math.tanh((c - prior_hi) / atr), 0.35, 1.0)
                desired_w = _clamp(strength * vol_unit * macro_boost * session_mult, 0.0, self.max_weight)
                raw_sig = _clamp(5.0 + 10.0 * math.tanh((c - prior_hi) / atr) + 3.0 * macro_comp, 0.1, 20.0)
            elif dn_breakout and self.allow_short and macro_comp <= 0.35:
                macro_boost = 1.0 + 0.25 * max(0.0, min(1.5, -macro_comp))
                strength = _clamp(0.55 + 0.45 * math.tanh((prior_lo - c) / atr), 0.35, 1.0)
                desired_w = -_clamp(
                    strength * vol_unit * macro_boost * session_mult * 0.85,
                    0.0,
                    self.max_weight,
                )
                raw_sig = -_clamp(5.0 + 10.0 * math.tanh((prior_lo - c) / atr) - 3.0 * macro_comp, 0.1, 20.0)
            elif prev_w > 1e-9:
                if long_exit_hit or macro_comp <= -0.50:
                    desired_w = 0.0
                    raw_sig = 0.0
                else:
                    desired_w = prev_w
                    raw_sig = max(0.5, 4.0 * max(0.0, macro_comp) + 2.0)
            elif prev_w < -1e-9:
                if short_exit_hit or macro_comp >= 0.50:
                    desired_w = 0.0
                    raw_sig = 0.0
                else:
                    desired_w = prev_w
                    raw_sig = min(-0.5, 4.0 * min(0.0, macro_comp) - 2.0)
            else:
                desired_w = 0.0
                raw_sig = 0.0

            # Policy C Post-Event Expansion [T, T + 8h] (strictly AFTER ts_avail_ms!)
            if self.policy == "C" and in_post_expansion:
                event_state = "POST_EVENT_EXPANSION"
                gvz_expansion_boost = 1.20 if gvz_z >= 0.5 else 1.08
                # Dovish / negative rate or inflation surprise adds mild confirmation on aligned Gold move
                surprise_adj = 1.0 + 0.05 * _clamp(-released_surprise_z, -1.0, 1.0)
                if desired_w > 1e-9 and macro_comp >= 0.0:
                    desired_w = _clamp(desired_w * gvz_expansion_boost * surprise_adj, 0.0, self.max_weight)
                    raw_sig = _clamp(raw_sig * gvz_expansion_boost, 0.1, 20.0)
                elif desired_w < -1e-9 and macro_comp <= 0.0:
                    desired_w = -_clamp(abs(desired_w) * gvz_expansion_boost, 0.0, self.max_weight)
                    raw_sig = -_clamp(abs(raw_sig) * gvz_expansion_boost, 0.1, 20.0)
                elif abs(desired_w) <= 1e-9 and len(self._closes) >= 2:
                    # Post-event aligned directional expansion (single-bar thrust OR close-channel expansion)
                    bar_thrust = (c - self._closes[-2]) / atr
                    close_up_exp = (c > prior_close_hi + eps) and (bar_thrust > 0.0)
                    close_dn_exp = (c < prior_close_lo - eps) and (bar_thrust < 0.0)
                    if (bar_thrust > 0.75 or close_up_exp) and macro_comp >= 0.25:
                        desired_w = _clamp(
                            0.65 * vol_unit * gvz_expansion_boost * surprise_adj,
                            0.0,
                            self.max_weight,
                        )
                        raw_sig = _clamp(
                            5.0 + 6.0 * math.tanh(max(0.25, bar_thrust)),
                            0.1,
                            20.0,
                        )
                    elif (bar_thrust < -0.75 or close_dn_exp) and self.allow_short and macro_comp <= -0.25:
                        desired_w = -_clamp(
                            0.55 * vol_unit * gvz_expansion_boost,
                            0.0,
                            self.max_weight,
                        )
                        raw_sig = -_clamp(
                            5.0 + 6.0 * math.tanh(max(0.25, -bar_thrust)),
                            0.1,
                            20.0,
                        )

        # Protective wide catastrophe stop on own 4h bars (routine exits handled by exit_n channel)
        stop_dist = max(self.stop_atr_mult * atr, self.stop_pct_floor * c)
        stop_price: float | None = None
        if desired_w > 1e-9:
            sp = c - stop_dist
            if sp > 1e-6:
                stop_price = round(sp, 6)
        elif desired_w < -1e-9:
            sp = c + stop_dist
            if math.isfinite(sp) and sp > 1e-6:
                stop_price = round(sp, 6)

        decisions[self.symbol] = BarDecision(
            symbol=self.symbol,
            ts_event_ms=int(ts_ms),
            target_weight=float(desired_w),
            raw_signal=float(raw_sig),
            regime_label=regime_lbl,
            stop_price=stop_price,
            metadata={
                "policy": self.policy,
                "session_tag": session_tag,
                "event_window_state": event_state,
                "active_event_id": active_ev_id,
                "macro_gold_composite": round(macro_comp, 6),
                "gvz_z": round(gvz_z, 6),
                "prior_high": round(prior_hi, 6),
                "prior_low": round(prior_lo, 6),
            },
        )
        return decisions
