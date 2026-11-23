"""Single-Kernel Adaptive Multi-Factor Strategy Engine & Automated Strategy Customizer.

Closes the entire Strategy R&D Platform v3 loop:
  Market Data (MultiAssetDataLake)
    -> Single-Kernel Causal Factors + G2 Direction & Redundancy Audit
    -> Causal Regime Profiler + Physics-Based AutoParameterInferrer
    -> Low-Dimensional 3x3 Plateau Scan + Connected Plateau Center Selection
    -> 7-Gate Institutional Verification (DQ + G2 + Full-Cost Stress + Plateau + CPCV + DSR/PBO + G7 Bit-Identical Parity)
    -> SHA-256 Fingerprinted CustomizedStrategyBlueprint

Design guarantees:
  - NautilusTrader Single-Kernel Parity: `AdaptiveMultiFactorKernel.decide_bar()` runs the
    EXACT same stateful signal, sizing, regime filter, buffer, and Chandelier stop logic in
    both historical batch evaluation and live bar-by-bar streaming (verified by Gate G7!).
  - Zero Lookahead: Decisions at bar `t` close fill strictly at bar `t+1` open (`T+1` fill).
  - Full Cost Realism: Taker/Maker fee + slippage + stamp duty + explicit bar funding/carry settlement.
  - Honest Verdict: Never relaxes thresholds; outputs `PROMOTE_SHADOW_CANDIDATE` only when
    all 7 gates pass, `CONDITIONAL_RESEARCH` when partially promising, or `REJECTED`.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from alphaloop.core.provenance import sha256_obj, utc_now_ms
from alphaloop.platform.assets import AssetSpec
from alphaloop.platform.data_lake import (
    CanonicalBar,
    MultiAssetDataLake,
    audit_bars_quality,
    fingerprint_canonical_bars,
)
from alphaloop.platform.factors import (
    FactorAuditSummary,
    FactorDirectionAuditor,
    FactorSnapshot,
    MacroRegimeTimeline,
    StreamingFactorEngine,
)
from alphaloop.platform.regime_inference import (
    AutoParameterInferrer,
    CausalRegimeTracker,
    InferredParameterSet,
    MarketProfile,
    MarketRegimeProfiler,
    RegimePoint,
)
from alphaloop.platform.validation import (
    CPCVReport,
    DSRReport,
    PBOReport,
    PlateauPoint,
    PlateauSelectionReport,
    compute_pbo_cscv,
    compute_psr_dsr,
    evaluate_cpcv_returns,
    select_plateau_center,
    sharpe_from_returns,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KernelConfig:
    """Immutable, SHA-256 hashable configuration for AdaptiveMultiFactorKernel."""

    strategy_name: str
    asset_class: str
    timeframe_h: float
    capital_usd: float = 3000.0
    max_positions: int = 6
    target_annual_vol_pct: float = 18.0
    max_symbol_leverage: float = 1.5
    entry_forecast_threshold: float = 4.0     # Enter/hold when scaled forecast |f| >= threshold
    exit_forecast_threshold: float = 1.0      # Hysteresis exit band
    atr_stop_mult: float = 3.5                # Chandelier ATR trailing stop multiplier
    min_hold_bars: int = 3                    # Cost-horizon anti-churn minimum hold (stop overrides!)
    position_buffer_pct: float = 0.12         # Carver inertia buffer on target weight changes
    ewmac_weight_fast: float = 0.25
    ewmac_weight_mid: float = 0.40
    ewmac_weight_slow: float = 0.35
    donchian_weight: float = 0.30             # Weight of Donchian/Squeeze vs EWMAC composite
    use_macro_overlay: bool = False           # Gold macro conditioning (QuantPedia Gold-Treasury/DXY/TIP)
    allow_short: bool = True
    execution_mode: str = "TAKER_TAKER"
    warmup_bars: int = 45
    regime_exposure_map: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KernelDecision:
    ts_event_ms: int
    ts_avail_ms: int
    symbol: str
    raw_forecast: float
    regime: str
    regime_scalar: float
    macro_scalar: float
    target_weight: float      # Signed notional / equity (e.g. +0.45 = 45% long, -0.30 = 30% short)
    stop_triggered: bool
    reason: str


@dataclass
class SymbolPositionState:
    weight: float = 0.0       # Signed notional / equity
    bars_held: int = 0
    highest_close_since_entry: float = 0.0
    lowest_close_since_entry: float = 0.0
    entry_price: float = 0.0
    t_plus_locked_until_bar: int = -1


class AdaptiveMultiFactorKernel:
    """Single-kernel causal signal & risk engine shared identically across batch and live."""

    def __init__(self, cfg: KernelConfig, spec: AssetSpec) -> None:
        self.cfg = cfg
        self.spec = spec
        self._bar_index: int = -1
        self.pos = SymbolPositionState()

    def decide_bar(
        self,
        bar: CanonicalBar,
        snap: FactorSnapshot,
        regime_pt: RegimePoint,
    ) -> KernelDecision:
        """Pure causal decision emitted at bar close (`ts_avail_ms`) to execute at next bar open."""
        self._bar_index += 1
        c = bar.close
        atr = snap.atr_14
        cur_w = self.pos.weight

        if self._bar_index < self.cfg.warmup_bars:
            return KernelDecision(
                ts_event_ms=bar.ts_event_ms,
                ts_avail_ms=bar.ts_avail_ms,
                symbol=bar.symbol,
                raw_forecast=0.0,
                regime=regime_pt.confirmed_regime,
                regime_scalar=0.0,
                macro_scalar=1.0,
                target_weight=0.0,
                stop_triggered=False,
                reason="WARMUP",
            )

        # 1. Multi-Factor Alpha Forecast in [-20, +20] (Carver forecast scaling)
        ewmac_blend = (
            self.cfg.ewmac_weight_fast * snap.get("ewmac_8_32", 0.0)
            + self.cfg.ewmac_weight_mid * snap.get("ewmac_16_64", 0.0)
            + self.cfg.ewmac_weight_slow * snap.get("ewmac_32_128", 0.0)
        )
        # Map Donchian channel [-1.5, +1.5] & Squeeze breakout [-1, +1] onto Carver [-20, +20] scale
        donch_f = max(-20.0, min(20.0, snap.get("donchian_pos_40", 0.0) * 14.0))
        squeeze_boost = snap.get("squeeze_breakout_dir", 0.0) * 4.0
        vol_confirm = 1.10 if snap.get("rel_volume_20", 1.0) >= 1.3 else 1.0

        w_d = max(0.0, min(0.8, self.cfg.donchian_weight))
        raw_forecast = max(
            -20.0,
            min(20.0, ((1.0 - w_d) * ewmac_blend + w_d * donch_f + squeeze_boost) * vol_confirm * 1.10),
        )

        # 2. Check Chandelier ATR trailing stop using strictly PRIOR high/low tracking
        stop_triggered = False
        if cur_w > 1e-6:
            self.pos.bars_held += 1
            self.pos.highest_close_since_entry = max(self.pos.highest_close_since_entry, c)
            stop_px = self.pos.highest_close_since_entry - self.cfg.atr_stop_mult * atr
            if c < stop_px and self._bar_index >= self.pos.t_plus_locked_until_bar:
                stop_triggered = True
        elif cur_w < -1e-6:
            self.pos.bars_held += 1
            self.pos.lowest_close_since_entry = (
                min(self.pos.lowest_close_since_entry, c) if self.pos.lowest_close_since_entry > 0 else c
            )
            stop_px = self.pos.lowest_close_since_entry + self.cfg.atr_stop_mult * atr
            if c > stop_px and self._bar_index >= self.pos.t_plus_locked_until_bar:
                stop_triggered = True

        # 3. Regime & Gold Macro Conditioning Scalar
        reg = regime_pt.confirmed_regime
        reg_scalar = float(self.cfg.regime_exposure_map.get(reg, 0.75))

        macro_scalar = 1.0
        if self.cfg.use_macro_overlay:
            m_comp = snap.get("macro_gold_composite", 0.0)
            gt_state = snap.get("macro_gold_treasury_state", 0.0)
            if raw_forecast > 0:
                if gt_state > 0.5 or m_comp >= 0.25:
                    macro_scalar = 1.05   # Falling real yields / easing tailwind (QuantPedia State 1)
                elif gt_state < -0.5 or m_comp <= -0.25:
                    macro_scalar = 0.25   # Rising real yields headwind -> defensive throttle
                else:
                    macro_scalar = 0.65
            elif raw_forecast < 0:
                if gt_state < -0.5 or m_comp <= -0.25:
                    macro_scalar = 1.00
                elif gt_state > 0.5:
                    macro_scalar = 0.20   # Never fight strong falling-real-rate gold bull blindly
                else:
                    macro_scalar = 0.55

        # 4. Carver Volatility-Targeted Position Sizing
        realized_vol = max(snap.get("realized_vol_ann_20", 25.0), 8.0)
        slot_cap = self.cfg.max_symbol_leverage / max(1.0, math.sqrt(float(self.cfg.max_positions)))
        vol_unit_weight = (self.cfg.target_annual_vol_pct / realized_vol) / max(1.0, float(self.cfg.max_positions) ** 0.5)
        scaled_strength = raw_forecast / 10.0  # Carver E[|f|] = 10 => unit risk = 1.0

        desired_w = max(-slot_cap, min(slot_cap, vol_unit_weight * scaled_strength * reg_scalar * macro_scalar))
        if not self.cfg.allow_short and desired_w < 0.0:
            desired_w = 0.0

        # 5. Entry/Exit Hysteresis + Chandelier Stop + Cost-Horizon Min-Hold + Carver Position Buffer
        reason = "HOLD"
        if stop_triggered:
            next_w = 0.0
            reason = "CHANDELIER_ATR_STOP"
        elif abs(cur_w) <= 1e-6:
            # Flat -> require entry threshold
            if abs(raw_forecast) >= self.cfg.entry_forecast_threshold and abs(desired_w) >= 0.03:
                next_w = desired_w
                reason = "ENTRY_SIGNAL"
            else:
                next_w = 0.0
                reason = "FLAT_BELOW_ENTRY"
        else:
            # In position
            t_plus_blocked = self._bar_index < self.pos.t_plus_locked_until_bar
            same_side = (cur_w > 0 and raw_forecast > 0) or (cur_w < 0 and raw_forecast < 0)
            if t_plus_blocked:
                next_w = cur_w
                reason = "T_PLUS_1_LOCK"
            elif not same_side and abs(raw_forecast) >= self.cfg.entry_forecast_threshold:
                if self.pos.bars_held >= self.cfg.min_hold_bars:
                    next_w = desired_w
                    reason = "REVERSAL_SIGNAL"
                else:
                    next_w = cur_w
                    reason = "MIN_HOLD_GUARD"
            elif abs(raw_forecast) < self.cfg.exit_forecast_threshold:
                if self.pos.bars_held >= self.cfg.min_hold_bars:
                    next_w = 0.0
                    reason = "HYSTERESIS_EXIT"
                else:
                    next_w = cur_w
                    reason = "MIN_HOLD_GUARD"
            else:
                # Carver inertia buffer: avoid tiny rebalances that burn fees
                rel_diff = abs(desired_w - cur_w) / max(abs(cur_w), 0.05)
                if rel_diff >= self.cfg.position_buffer_pct and self.pos.bars_held >= self.cfg.min_hold_bars:
                    next_w = desired_w
                    reason = "VOL_REBALANCE"
                else:
                    next_w = cur_w
                    reason = "BUFFER_HOLD"

        # Update stateful position tracker deterministically
        if abs(next_w) <= 1e-6:
            self.pos = SymbolPositionState()
        elif abs(cur_w) <= 1e-6 or (cur_w * next_w < 0):
            # New entry or flip
            t_lock = self._bar_index + max(0, int(self.spec.calendar.t_plus_days))
            self.pos = SymbolPositionState(
                weight=next_w,
                bars_held=0,
                highest_close_since_entry=c,
                lowest_close_since_entry=c,
                entry_price=c,
                t_plus_locked_until_bar=t_lock,
            )
        else:
            self.pos.weight = next_w

        return KernelDecision(
            ts_event_ms=bar.ts_event_ms,
            ts_avail_ms=bar.ts_avail_ms,
            symbol=bar.symbol,
            raw_forecast=round(raw_forecast, 4),
            regime=reg,
            regime_scalar=round(reg_scalar, 3),
            macro_scalar=round(macro_scalar, 3),
            target_weight=round(next_w, 6),
            stop_triggered=stop_triggered,
            reason=reason,
        )


@dataclass
class SimulationMetrics:
    days: float
    bars_simulated: int
    trade_count: int
    total_turnover: float
    gross_price_pnl_usd: float
    funding_carry_pnl_usd: float
    total_cost_usd: float
    net_pnl_usd: float
    total_return_pct: float
    apr_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    liquidation_count: int
    period_returns: list[float] = field(default_factory=list)
    by_symbol_net_usd: dict[str, float] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("period_returns", None)
        return d


def simulate_universe(
    cfg: KernelConfig,
    specs: dict[str, AssetSpec],
    bars_by_symbol: dict[str, list[CanonicalBar]],
    snaps_by_symbol: dict[str, list[FactorSnapshot]],
    regimes_by_symbol: dict[str, list[RegimePoint]],
    extra_slippage_bps_per_leg: float = 0.0,
) -> tuple[SimulationMetrics, dict[str, list[KernelDecision]]]:
    """Run strictly causal T+1 open-fill multi-asset portfolio simulation via AdaptiveMultiFactorKernel."""
    # Step 1: Run the single-kernel per symbol to generate causal bar-close target weights
    decisions_by_sym: dict[str, list[KernelDecision]] = {}
    decision_map: dict[tuple[str, int], KernelDecision] = {}
    bar_map: dict[tuple[str, int], CanonicalBar] = {}
    all_timestamps: set[int] = set()

    for sym, bars in sorted(bars_by_symbol.items()):
        if sym not in specs:
            continue
        snaps = snaps_by_symbol[sym]
        regs = regimes_by_symbol[sym]
        kernel = AdaptiveMultiFactorKernel(cfg, specs[sym])
        decs: list[KernelDecision] = []
        for b, s, r in zip(bars, snaps, regs):
            dec = kernel.decide_bar(b, s, r)
            decs.append(dec)
            decision_map[(sym, b.ts_event_ms)] = dec
            bar_map[(sym, b.ts_event_ms)] = b
            all_timestamps.add(b.ts_event_ms)
        decisions_by_sym[sym] = decs

    timeline = sorted(all_timestamps)
    if len(timeline) < 3:
        empty = SimulationMetrics(0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        return empty, decisions_by_sym

    equity = float(cfg.capital_usd)
    peak_eq = equity
    max_dd = 0.0
    # Signed held weight per symbol executed at bar t open from bar t-1 close decision!
    active_weights: dict[str, float] = {sym: 0.0 for sym in decisions_by_sym}
    pending_target: dict[str, float] = {sym: 0.0 for sym in decisions_by_sym}

    gross_price_pnl = 0.0
    funding_pnl = 0.0
    total_cost = 0.0
    total_turnover = 0.0
    trade_count = 0
    liquidations = 0
    by_sym_pnl: dict[str, float] = {sym: 0.0 for sym in decisions_by_sym}
    period_rets: list[float] = []

    first_spec = next(iter(specs.values()))
    bars_per_year = first_spec.calendar.bars_per_year(cfg.timeframe_h)

    for idx, ts in enumerate(timeline):
        eq_start = equity
        if eq_start <= 1e-6:
            liquidations += 1
            break

        # Cap gross portfolio leverage safely across active symbols
        gross_pending = sum(abs(w) for w in pending_target.values())
        port_lev_cap = max(1.0, cfg.max_symbol_leverage * 1.5)
        lev_scale = (port_lev_cap / gross_pending) if gross_pending > port_lev_cap else 1.0

        bar_pnl = 0.0
        for sym in decisions_by_sym:
            bar = bar_map.get((sym, ts))
            if bar is None:
                continue
            spec = specs[sym]
            target_w = pending_target[sym] * lev_scale
            prev_w = active_weights[sym]

            # Execute weight delta strictly at THIS bar's OPEN (decided at previous bar's close -> zero lookahead!)
            delta_w = target_w - prev_w
            if abs(delta_w) > 1e-6:
                is_sell = delta_w < 0.0
                one_way_bps = spec.cost_model.one_way_cost_bps(
                    execution_mode=cfg.execution_mode,
                    is_sell=is_sell,
                    extra_slippage_bps=extra_slippage_bps_per_leg,
                )
                notional_traded = abs(delta_w) * eq_start
                cost_usd = notional_traded * (one_way_bps / 10_000.0)
                total_cost += cost_usd
                bar_pnl -= cost_usd
                by_sym_pnl[sym] -= cost_usd
                total_turnover += abs(delta_w)
                if (abs(prev_w) <= 1e-6 and abs(target_w) > 1e-6) or (prev_w * target_w < 0) or (abs(prev_w) > 1e-6 and abs(target_w) <= 1e-6):
                    trade_count += 1
                active_weights[sym] = target_w

            held_w = active_weights[sym]
            if abs(held_w) > 1e-6 and bar.open > 0:
                notional = held_w * eq_start
                # Mark-to-market return from THIS bar's open to THIS bar's close (plus gap if held from prior close)
                # Using open->next_open approximation cleanly via open->close + prior_close->open when already held:
                prev_bar = bar_map.get((sym, timeline[idx - 1])) if idx > 0 else None
                if abs(prev_w) > 1e-6 and prev_bar is not None and prev_bar.close > 0:
                    gap_ret = (bar.open / prev_bar.close) - 1.0
                    gap_usd = (prev_w * eq_start) * gap_ret
                    gross_price_pnl += gap_usd
                    bar_pnl += gap_usd
                    by_sym_pnl[sym] += gap_usd

                intra_ret = (bar.close / bar.open) - 1.0
                p_usd = notional * intra_ret
                gross_price_pnl += p_usd
                bar_pnl += p_usd
                by_sym_pnl[sym] += p_usd

                # Funding / carry settlement: positive carry_rate_period means LONGS pay SHORTS
                carry_usd = -notional * bar.carry_rate_period
                # Plus annual holding drag (e.g., ETF expense ratio / roll)
                drag_usd = -abs(notional) * ((spec.cost_model.annual_carry_drag_bps / 10_000.0) / max(1.0, bars_per_year))
                funding_pnl += (carry_usd + drag_usd)
                bar_pnl += (carry_usd + drag_usd)
                by_sym_pnl[sym] += (carry_usd + drag_usd)

            # Queue THIS bar's close decision for NEXT bar's open execution
            dec = decision_map.get((sym, ts))
            if dec is not None:
                pending_target[sym] = dec.target_weight

        equity = max(0.0, eq_start + bar_pnl)
        if equity <= cfg.capital_usd * 0.25:
            liquidations += 1
            equity = 0.0
        if equity > peak_eq:
            peak_eq = equity
        dd = ((peak_eq - equity) / peak_eq) * 100.0 if peak_eq > 0 else 100.0
        if dd > max_dd:
            max_dd = dd

        if idx >= cfg.warmup_bars:
            period_rets.append((equity - eq_start) / eq_start if eq_start > 0 else 0.0)

    days = max(1.0, (timeline[-1] - timeline[0]) / 86_400_000.0)
    net_pnl = equity - cfg.capital_usd
    tot_ret = (net_pnl / cfg.capital_usd) * 100.0 if cfg.capital_usd > 0 else 0.0
    apr = tot_ret * (365.0 / days)
    sr = sharpe_from_returns(period_rets, bars_per_year)
    calmar = (apr / max_dd) if max_dd > 0.1 else (apr / 0.1 if apr > 0 else 0.0)

    metrics = SimulationMetrics(
        days=round(days, 2),
        bars_simulated=len(period_rets),
        trade_count=trade_count,
        total_turnover=round(total_turnover, 3),
        gross_price_pnl_usd=round(gross_price_pnl, 2),
        funding_carry_pnl_usd=round(funding_pnl, 2),
        total_cost_usd=round(total_cost, 2),
        net_pnl_usd=round(net_pnl, 2),
        total_return_pct=round(tot_ret, 2),
        apr_pct=round(apr, 2),
        sharpe_ratio=round(sr, 4),
        calmar_ratio=round(calmar, 3),
        max_drawdown_pct=round(max_dd, 2),
        liquidation_count=liquidations,
        period_returns=period_rets,
        by_symbol_net_usd={k: round(v, 2) for k, v in sorted(by_sym_pnl.items())},
    )
    return metrics, decisions_by_sym


def verify_single_kernel_parity(
    cfg: KernelConfig,
    specs: dict[str, AssetSpec],
    bars_by_symbol: dict[str, list[CanonicalBar]],
    batch_decisions: dict[str, list[KernelDecision]],
    macro_bars_by_symbol: dict[str, list[CanonicalBar]] | None = None,
) -> tuple[bool, float, str]:
    """Gate G7: Replay identical bars one-by-one through fresh streaming engines & assert bit parity."""
    max_delta = 0.0
    checked_bars = 0
    for sym, bars in sorted(bars_by_symbol.items()):
        if sym not in specs or sym not in batch_decisions:
            continue
        spec = specs[sym]
        ordered = sorted(bars, key=lambda b: b.ts_event_ms)
        f_eng = StreamingFactorEngine(spec.calendar, cfg.timeframe_h)
        r_trk = CausalRegimeTracker(confirm_bars=2)
        kernel = AdaptiveMultiFactorKernel(cfg, spec)
        m_tl = MacroRegimeTimeline(macro_bars_by_symbol) if macro_bars_by_symbol else None

        expected_list = batch_decisions[sym]
        if len(expected_list) != len(ordered):
            return False, 1.0, f"Length mismatch on {sym}: batch={len(expected_list)} vs stream={len(ordered)}"

        for bar, exp in zip(ordered, expected_list):
            mctx = m_tl.snapshot_as_of(bar.ts_avail_ms) if m_tl is not None else None
            snap = f_eng.update(bar, macro_ctx=mctx)
            rpt = r_trk.step(snap)
            got = kernel.decide_bar(bar, snap, rpt)
            checked_bars += 1
            delta = abs(got.target_weight - exp.target_weight)
            if delta > max_delta:
                max_delta = delta
            if delta > 1e-10 or got.regime != exp.regime or got.reason != exp.reason:
                return (
                    False,
                    delta,
                    f"Parity drift on {sym}@{bar.ts_event_ms}: batch=({exp.target_weight},{exp.regime},{exp.reason}) "
                    f"vs stream=({got.target_weight},{got.regime},{got.reason})",
                )
    return True, max_delta, f"Bit-identical single-kernel parity across {checked_bars} bars (max_delta={max_delta:.2e})."


@dataclass
class GateOutcome:
    gate_id: str
    name: str
    passed: bool
    value: str
    threshold: str
    detail: str


@dataclass
class CustomizationResult:
    """Complete auditable output of Strategy R&D Platform v3 automated strategy customization."""

    blueprint_id: str
    track_name: str
    created_ts_ms: int
    dataset_sha256: str
    provenance: str
    verdict: str  # PROMOTE_SHADOW_CANDIDATE | CONDITIONAL_RESEARCH | REJECTED
    passed_gates: int
    total_gates: int
    inferred_params: InferredParameterSet
    final_kernel_config: KernelConfig
    profiles: list[MarketProfile]
    factor_audit: FactorAuditSummary
    plateau_report: PlateauSelectionReport
    baseline_metrics: dict[str, Any]
    cost_stress_metrics: dict[str, Any]
    cpcv_report: CPCVReport
    pbo_report: PBOReport
    dsr_report: DSRReport
    gates: list[GateOutcome]

    def to_dict(self) -> dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "track_name": self.track_name,
            "created_ts_ms": self.created_ts_ms,
            "dataset_sha256": self.dataset_sha256,
            "provenance": self.provenance,
            "verdict": self.verdict,
            "passed_gates": self.passed_gates,
            "total_gates": self.total_gates,
            "inferred_params": self.inferred_params.to_dict(),
            "final_kernel_config": self.final_kernel_config.to_dict(),
            "profiles": [p.to_dict() for p in self.profiles],
            "factor_audit": self.factor_audit.to_dict(),
            "plateau_report": self.plateau_report.to_dict(),
            "baseline_metrics": self.baseline_metrics,
            "cost_stress_metrics": self.cost_stress_metrics,
            "cpcv_report": self.cpcv_report.to_dict(),
            "pbo_report": self.pbo_report.to_dict(),
            "dsr_report": self.dsr_report.to_dict(),
            "gates": [asdict(g) for g in self.gates],
        }


class StrategyCustomizer:
    """End-to-end automated strategy researcher, parameter inferrer, customizer, and 7-Gate verifier."""

    def __init__(self, target_annual_vol_pct: float = 18.0) -> None:
        self.profiler = MarketRegimeProfiler()
        self.inferrer = AutoParameterInferrer(target_annual_vol_pct=target_annual_vol_pct)
        self.factor_auditor = FactorDirectionAuditor()

    def build_and_validate(
        self,
        track_name: str,
        specs: dict[str, AssetSpec],
        bars_by_symbol: dict[str, list[CanonicalBar]],
        macro_bars_by_symbol: dict[str, list[CanonicalBar]] | None = None,
        capital_usd: float = 3000.0,
        execution_mode: str = "TAKER_TAKER",
        factor_horizon_bars: int = 8,
        allow_synthetic_in_test: bool = False,
        lake: MultiAssetDataLake | None = None,
    ) -> CustomizationResult:
        """Execute full 7-Gate Strategy R&D Platform v3 pipeline on real (or unit-test) multi-asset bars."""
        # 1. Dataset Fingerprint & Multi-Asset DQ Certification (Gate G1)
        all_bars = [b for sym in sorted(bars_by_symbol) for b in bars_by_symbol[sym]]
        ds_sha, prov, _ = fingerprint_canonical_bars(all_bars)
        dq_reports = [
            audit_bars_quality(
                bars_by_symbol[sym],
                specs[sym].calendar,
                dataset_label=f"{track_name}:{sym}",
                min_bars=50,
                allow_synthetic_in_test=allow_synthetic_in_test,
            )
            for sym in sorted(bars_by_symbol)
            if sym in specs
        ]
        g1_ok = bool(dq_reports) and all(r.passed for r in dq_reports)
        dq_fail_syms = [r.dataset_label for r in dq_reports if not r.passed]

        # 2. Single-Kernel Causal Factor Computation & Market Profiling
        snaps_by_sym: dict[str, list[FactorSnapshot]] = {}
        regimes_by_sym: dict[str, list[RegimePoint]] = {}
        profiles: list[MarketProfile] = []
        valid_specs: list[AssetSpec] = []

        for sym in sorted(bars_by_symbol):
            if sym not in specs:
                continue
            ordered = sorted(bars_by_symbol[sym], key=lambda b: b.ts_event_ms)
            if len(ordered) < 55:
                continue
            spec = specs[sym]
            tf_h = ordered[0].timeframe_h
            m_tl = MacroRegimeTimeline(macro_bars_by_symbol) if macro_bars_by_symbol else None
            f_eng = StreamingFactorEngine(spec.calendar, tf_h)
            snaps = f_eng.compute_series(ordered, macro_timeline=m_tl)
            prof, rpts = self.profiler.profile_asset(spec, ordered, snaps, execution_mode=execution_mode)
            snaps_by_sym[sym] = snaps
            regimes_by_sym[sym] = rpts
            profiles.append(prof)
            valid_specs.append(spec)
            if lake is not None:
                lake.save_regime_snapshot(spec.venue, spec.symbol, tf_h, prof.current_regime, prof.to_dict())

        # 3. G2 Factor Direction & Redundancy Audit
        factor_audit = self.factor_auditor.audit(
            track_name=track_name,
            bars_by_symbol={s: bars_by_symbol[s] for s in snaps_by_sym},
            snaps_by_symbol=snaps_by_sym,
            horizon_bars=factor_horizon_bars,
        )
        g2_ok = len(factor_audit.selected_non_redundant_factors) >= 1
        if lake is not None:
            lake.save_factor_audit(track_name, ds_sha, factor_audit.to_dict())

        # 4. Physics-Based Automatic Parameter Inference
        inferred = self.inferrer.infer_from_profiles(track_name, valid_specs, profiles)

        # Adapt blend weights cleanly from G2 audit without blind optimization
        has_donch = any(f.startswith("donchian") or f == "squeeze_breakout_dir" for f in factor_audit.passed_g2_factors)
        donch_weight = 0.35 if has_donch else 0.20

        base_cfg = KernelConfig(
            strategy_name=f"{track_name}_v3_custom",
            asset_class=inferred.asset_class,
            timeframe_h=inferred.timeframe_h,
            capital_usd=capital_usd,
            max_positions=min(6, max(1, len(valid_specs))),
            target_annual_vol_pct=inferred.target_annual_vol_pct,
            max_symbol_leverage=inferred.max_symbol_leverage,
            entry_forecast_threshold=4.0,
            exit_forecast_threshold=1.0,
            atr_stop_mult=inferred.atr_stop_mult,
            min_hold_bars=inferred.min_hold_bars,
            position_buffer_pct=inferred.position_buffer_pct,
            ewmac_weight_fast=inferred.ewmac_weight_fast,
            ewmac_weight_mid=inferred.ewmac_weight_mid,
            ewmac_weight_slow=inferred.ewmac_weight_slow,
            donchian_weight=donch_weight,
            use_macro_overlay=inferred.use_macro_overlay and bool(macro_bars_by_symbol),
            allow_short=inferred.allow_short,
            execution_mode=execution_mode,
            regime_exposure_map=dict(inferred.regime_exposure_map),
        )

        # 5. Low-Dimensional 3x3 Bounded Plateau Scan around inferred parameters (NEVER blind 8D!)
        entry_grid = (2.5, 4.0, 5.5)
        stop_deltas = (-0.5, 0.0, +0.5)
        grid_points: list[PlateauPoint] = []
        cand_returns_matrix: list[list[float]] = []
        cand_configs: dict[str, KernelConfig] = {}
        cand_sims: dict[str, tuple[SimulationMetrics, dict[str, list[KernelDecision]]]] = {}

        for ix, ent_thr in enumerate(entry_grid):
            for iy, s_delta in enumerate(stop_deltas):
                s_mult = round(max(2.0, min(5.0, base_cfg.atr_stop_mult + s_delta)), 2)
                lbl = f"ent{ent_thr:.1f}_atr{s_mult:.2f}"
                c_cfg = dataclasses.replace(
                    base_cfg,
                    entry_forecast_threshold=ent_thr,
                    exit_forecast_threshold=max(0.5, round(ent_thr * 0.25, 2)),
                    atr_stop_mult=s_mult,
                )
                sim_m, decs = simulate_universe(
                    c_cfg,
                    {s.symbol: s for s in valid_specs},
                    {s.symbol: bars_by_symbol[s.symbol] for s in valid_specs},
                    snaps_by_sym,
                    regimes_by_sym,
                )
                cand_configs[lbl] = c_cfg
                cand_sims[lbl] = (sim_m, decs)
                cand_returns_matrix.append(sim_m.period_returns)
                grid_points.append(
                    PlateauPoint(
                        ix=ix,
                        iy=iy,
                        label=lbl,
                        params={"entry_forecast_threshold": ent_thr, "atr_stop_mult": s_mult},
                        sharpe=sim_m.sharpe_ratio,
                        apr_pct=sim_m.apr_pct,
                        max_dd_pct=sim_m.max_drawdown_pct,
                        net_pnl_usd=sim_m.net_pnl_usd,
                    )
                )

        # Select geometric plateau center (BAISYS_QUANT discipline: never pick isolated spike!)
        plateau_rep = select_plateau_center(grid_points, min_sharpe_for_plateau=0.25, min_plateau_ratio=0.55)
        final_cfg = cand_configs[plateau_rep.selected_label]
        final_metrics, final_decs = cand_sims[plateau_rep.selected_label]

        # 6. Cost Stress Test (+2bp/leg extra slippage)
        stress_metrics, _ = simulate_universe(
            final_cfg,
            {s.symbol: s for s in valid_specs},
            {s.symbol: bars_by_symbol[s.symbol] for s in valid_specs},
            snaps_by_sym,
            regimes_by_sym,
            extra_slippage_bps_per_leg=2.0,
        )

        # 7. CPCV, CSCV PBO, and Exact PSR/DSR (AFML)
        bars_per_year = valid_specs[0].calendar.bars_per_year(final_cfg.timeframe_h)
        purge = max(2, final_cfg.min_hold_bars)
        cpcv_rep = evaluate_cpcv_returns(
            final_metrics.period_returns,
            periods_per_year=bars_per_year,
            n_groups=6,
            k_test=2,
            purge_bars=purge,
            embargo_bars=purge,
        )
        pbo_rep = compute_pbo_cscv(cand_returns_matrix, periods_per_year=bars_per_year, n_subsamples=8)
        trial_srs = [p.sharpe / math.sqrt(max(1.0, bars_per_year)) for p in grid_points]
        trial_std = statistics.pstdev(trial_srs) if len(trial_srs) >= 2 else None
        dsr_rep = compute_psr_dsr(
            final_metrics.period_returns,
            periods_per_year=bars_per_year,
            n_trials=len(grid_points),
            trial_sharpe_std_per_period=trial_std,
        )

        # 8. Gate G7: Single-Kernel Streaming vs Batch Bit-Parity Check
        g7_ok, max_delta, g7_detail = verify_single_kernel_parity(
            final_cfg,
            {s.symbol: s for s in valid_specs},
            {s.symbol: bars_by_symbol[s.symbol] for s in valid_specs},
            final_decs,
            macro_bars_by_symbol=macro_bars_by_symbol,
        )

        # 9. Assemble Honest 7-Gate Verdict
        g3_ok = (
            final_metrics.net_pnl_usd > 0.0
            and stress_metrics.net_pnl_usd > 0.0
            and final_metrics.liquidation_count == 0
            and final_metrics.trade_count >= 5
        )
        g4_ok = plateau_rep.passed_plateau_gate
        g5_ok = cpcv_rep.n_splits >= 6 and cpcv_rep.median_oos_sharpe > 0.0 and cpcv_rep.pct_positive_splits >= 60.0
        g6_ok = dsr_rep.dsr >= 0.50 and pbo_rep.pbo <= 0.35

        gates = [
            GateOutcome(
                gate_id="G1",
                name="Multi-Asset Data Quality & Provenance (DQ-MA)",
                passed=g1_ok,
                value=f"{len(dq_reports)-len(dq_fail_syms)}/{len(dq_reports)} clean ({prov})",
                threshold="100% REAL & DQ-MA PASS",
                detail="All clean" if g1_ok else f"Failed: {dq_fail_syms}",
            ),
            GateOutcome(
                gate_id="G2",
                name="Factor Direction & Redundancy Audit",
                passed=g2_ok,
                value=f"{len(factor_audit.selected_non_redundant_factors)} selected / {len(factor_audit.passed_g2_factors)} passed",
                threshold=">=1 directional factor passes IC sign + Q1-Q5 mono",
                detail=f"selected={factor_audit.selected_non_redundant_factors[:6]}",
            ),
            GateOutcome(
                gate_id="G3",
                name="Full-Cost Expectancy, Stress (+2bp/leg) & 0-Liq",
                passed=g3_ok,
                value=f"net=${final_metrics.net_pnl_usd:+.2f} (APR {final_metrics.apr_pct:+.1f}%), stress=${stress_metrics.net_pnl_usd:+.2f}, liq={final_metrics.liquidation_count}",
                threshold="net>0 & stress>0 & liq=0 & trades>=5",
                detail=f"trades={final_metrics.trade_count}, costs=${final_metrics.total_cost_usd:.2f}, funding=${final_metrics.funding_carry_pnl_usd:+.2f}",
            ),
            GateOutcome(
                gate_id="G4",
                name="Connected Parameter Plateau Stability",
                passed=g4_ok,
                value=f"plateau={plateau_rep.largest_plateau_size}/{plateau_rep.total_points} ({plateau_rep.plateau_ratio*100:.0f}%), center={plateau_rep.selected_label}",
                threshold="plateau>=55% & center/peak>=0.55",
                detail=plateau_rep.detail,
            ),
            GateOutcome(
                gate_id="G5",
                name="Purged & Embargoed CPCV Generalization",
                passed=g5_ok,
                value=f"med_OOS_SR={cpcv_rep.median_oos_sharpe:.2f}, pos={cpcv_rep.pct_positive_splits:.0f}% ({cpcv_rep.n_splits} splits)",
                threshold="med_OOS_SR>0 & >=60% splits positive",
                detail=f"min_OOS_SR={cpcv_rep.min_oos_sharpe:.2f}, IS->OOS retention={cpcv_rep.is_to_oos_retention:.2f}x",
            ),
            GateOutcome(
                gate_id="G6",
                name="Multiple-Testing Defense (Exact DSR + CSCV PBO)",
                passed=g6_ok,
                value=f"DSR={dsr_rep.dsr:.3f}, PSR={dsr_rep.psr:.3f}, PBO={pbo_rep.pbo:.3f}",
                threshold="DSR>=0.50 & PBO<=0.35",
                detail=f"SR={dsr_rep.sr_annualized:.2f} vs SR*={dsr_rep.sr_star_annualized:.2f} (N={dsr_rep.n_trials}, skew={dsr_rep.skewness:.2f}, kurt={dsr_rep.excess_kurtosis:.2f})",
            ),
            GateOutcome(
                gate_id="G7",
                name="Single-Kernel Research-to-Live Bit Parity",
                passed=g7_ok,
                value=f"max_delta={max_delta:.2e}",
                threshold="max_delta<=1e-10 & exact regime/reason",
                detail=g7_detail,
            ),
        ]

        passed_n = sum(1 for g in gates if g.passed)
        if passed_n == len(gates):
            verdict = "PROMOTE_SHADOW_CANDIDATE"
        elif g1_ok and g3_ok and g7_ok and passed_n >= 5:
            verdict = "CONDITIONAL_RESEARCH"
        else:
            verdict = "REJECTED"

        now = utc_now_ms()
        bp_id = f"BP-{sha256_obj({'t': track_name, 'd': ds_sha, 'c': final_cfg.to_dict()})[:10]}"
        result = CustomizationResult(
            blueprint_id=bp_id,
            track_name=track_name,
            created_ts_ms=now,
            dataset_sha256=ds_sha,
            provenance=prov,
            verdict=verdict,
            passed_gates=passed_n,
            total_gates=len(gates),
            inferred_params=inferred,
            final_kernel_config=final_cfg,
            profiles=profiles,
            factor_audit=factor_audit,
            plateau_report=plateau_rep,
            baseline_metrics=final_metrics.summary_dict(),
            cost_stress_metrics=stress_metrics.summary_dict(),
            cpcv_report=cpcv_rep,
            pbo_report=pbo_rep,
            dsr_report=dsr_rep,
            gates=gates,
        )
        if lake is not None:
            lake.save_strategy_blueprint(track_name, inferred.recommended_archetype, verdict, ds_sha, result.to_dict())
        return result
