"""Core strongly-typed domain models for AlphaLoop Strategy R&D Platform."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FundingRateBar:
    """Realized funding settlement bar normalized to UTC millisecond timestamp."""

    venue: str
    symbol: str
    ts_ms: int
    funding_rate: float          # Decimal rate paid over interval_hours (e.g. 0.0001 = 1 bp)
    interval_hours: float        # 1.0 (HL/Lighter), 4.0 or 8.0 (OKX/Gate)
    provenance: str

    @property
    def rate_per_hour(self) -> float:
        return self.funding_rate / self.interval_hours if self.interval_hours > 0 else 0.0

    @property
    def apr_pct(self) -> float:
        return self.rate_per_hour * 24.0 * 365.0 * 100.0


@dataclass
class PriceBar:
    """1-hour OHLCV price bar for aligned mark/basis/liquidation accounting."""

    venue: str
    symbol: str
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume_usd: float
    provenance: str


@dataclass
class StrategyConfig:
    """Declarative, hot-reloadable strategy configuration."""

    strategy_id: str
    version: str
    family: str                           # e.g. "cross_venue_funding_carry"
    status: str                           # DRAFT | BACKTESTED | RED_TEAMED | SHADOW_LIVE | CANARY_APPROVED | RETIRED
    venue_a: str                          # e.g. "hyperliquid"
    venue_b: str                          # e.g. "lighter" | "okx" | "gate"
    mode: str                             # "perp_perp" | "spot_perp"
    symbols: list[str]
    capital_usd: float = 3000.0
    leverage: float = 2.0                 # Perp-Perp effective notional = capital * leverage / 2 per leg
    max_positions: int = 3
    per_position_weight: float = 0.30     # Fraction of total capital per slot
    entry_spread_apr_pct: float = 18.0    # Enter when rolling spread APR >= threshold
    exit_spread_apr_pct: float = 4.0      # Hysteresis: exit only when rolling spread APR < threshold
    lookback_hours: int = 24              # Causal trailing window (strictly <= decision time)
    exit_confirm_hours: int = 12          # Consecutive hours below exit threshold before closing
    min_hold_hours: int = 24              # Anti-churn minimum hold unless stop triggered
    max_basis_entry_bps: float = 40.0     # Avoid entering into extreme adverse basis
    stop_loss_basis_bps: float = 200.0    # Emergency exit if mark basis widens adversely
    fee_a_bps: float = 1.5                # Maker 1.5bp (HL) / Taker 4.5bp
    fee_b_bps: float = 0.0                # Maker 0.0bp (Lighter) / 2.0bp (OKX/Gate)
    slippage_per_leg_bps: float = 1.5     # Execution slippage per leg entry/exit
    rebalance_drag_bps_per_month: float = 10.0  # Amortized USDC/USDT transfer/bridge friction
    allow_bidirectional: bool = True      # True for Perp-Perp; False for long-spot/short-perp
    use_atb_buggy_rules: bool = False     # Forensic mode reproducing ATB H6 churn bug
    notes: str = ""
    plugin_id: str = "funding_carry"      # Strategy Factory plugin identifier
    execution_mode: str = "MAKER_MAKER"   # MAKER_MAKER | MAKER_THEN_TAKER_HEDGE | TAKER_TAKER
    params: dict[str, Any] = field(default_factory=dict)  # Plugin-specific extensible parameters

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        if "params" in filtered and isinstance(filtered["params"], dict):
            filtered["params"] = dict(filtered["params"])
        return cls(**filtered)


@dataclass
class TradeRecord:
    """Single closed or open carry trade with full cost/PnL decomposition."""

    trade_id: str
    symbol: str
    direction: str                        # "SHORT_A_LONG_B" | "LONG_A_SHORT_B"
    entry_ts_ms: int
    exit_ts_ms: int
    hold_hours: float
    notional_per_leg_usd: float
    entry_spread_apr_pct: float
    gross_funding_usd: float
    basis_pnl_usd: float
    fee_usd: float
    slippage_usd: float
    rebalance_usd: float
    net_pnl_usd: float
    exit_reason: str                      # "HYSTERESIS_EXIT" | "BASIS_STOP" | "MARGIN_LIQUIDATION" | "END_OF_WINDOW"
    max_adverse_basis_bps: float = 0.0

    @property
    def net_bps_on_notional(self) -> float:
        if self.notional_per_leg_usd <= 0:
            return 0.0
        return (self.net_pnl_usd / self.notional_per_leg_usd) * 10000.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["net_bps_on_notional"] = round(self.net_bps_on_notional, 3)
        return d


@dataclass
class BacktestMetrics:
    """Comprehensive accounting metrics across all 3 REA-001 layers."""

    window_start_ms: int
    window_end_ms: int
    days: float
    capital_usd: float
    trade_count: int
    win_count: int
    win_rate: float
    avg_hold_hours: float
    gross_funding_usd: float
    basis_pnl_usd: float
    total_fees_usd: float
    total_slippage_usd: float
    total_rebalance_usd: float
    net_pnl_usd: float
    total_return_pct: float
    annualized_apr_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    fee_to_gross_ratio: float
    avg_capital_utilization_pct: float
    monthly_usd_est: float
    top5_removed_net_pnl_usd: float
    top_symbol_removed_net_pnl_usd: float
    best_symbol: str
    provenance: str
    dataset_sha256: str
    by_symbol: dict[str, dict[str, float]] = field(default_factory=dict)
    gross_alpha_usd: float = 0.0
    liquidation_count: int = 0
    forced_rebalance_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateCheck:
    gate_id: str
    name: str
    passed: bool
    value: str
    threshold: str
    detail: str


@dataclass
class GateVerdict:
    strategy_id: str
    version: str
    overall_pass: bool
    passed_count: int
    total_count: int
    trial_number: int
    deflated_sharpe_note: str
    checks: list[GateCheck] = field(default_factory=list)
    dq_cert_id: str = ""
    dq_tier: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "overall_pass": self.overall_pass,
            "passed_count": self.passed_count,
            "total_count": self.total_count,
            "trial_number": self.trial_number,
            "deflated_sharpe_note": self.deflated_sharpe_note,
            "dq_cert_id": self.dq_cert_id,
            "dq_tier": self.dq_tier,
            "checks": [asdict(c) for c in self.checks],
        }


@dataclass
class FeedbackEvent:
    event_id: str
    ts_ms: int
    strategy_id: str
    severity: str       # P0 | P1 | INFO
    anomaly_type: str   # CHURN_EXCESS | EDGE_DECAY | COST_OVERRUN | BASIS_SHOCK | HEALTHY
    summary: str
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentProposal:
    proposal_id: str
    ts_ms: int
    model_used: str
    role: str
    hypothesis: str
    economic_rationale: str
    param_changes: dict[str, Any]
    kill_criteria: str
    raw_reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RedTeamVerdict:
    verdict_id: str
    ts_ms: int
    model_used: str
    decision: str       # APPROVE_SHADOW | REVISE | REJECT
    findings: list[dict[str, str]]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BBOTick:
    """Real-time top-of-book quote snapshot normalized to UTC ms (STD-32/STD-34)."""

    venue: str
    symbol: str
    ts_ms: int
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    latency_ms: float
    provenance: str

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) * 0.5

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return ((self.ask - self.bid) / m) * 10000.0 if m > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mid"] = round(self.mid, 8)
        d["spread_bps"] = round(self.spread_bps, 4)
        return d


@dataclass
class DataQualityCheck:
    gate_id: str        # DQ-01 .. DQ-08
    name: str
    passed: bool
    severity: str       # BLOCKING | ADVISORY
    value: str
    threshold: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualityReport:
    cert_id: str        # DQ-<sha256[:10]>
    created_ts_ms: int
    dataset_name: str
    venue_pair: str
    dataset_sha256: str
    provenance: str
    row_count: int
    basis_coverage_pct: float
    tier: str           # DATA_QUALIFIED_FULL | DATA_QUALIFIED_FUNDING_ONLY | DATA_REJECTED
    passed_count: int
    total_count: int
    checks: list[DataQualityCheck] = field(default_factory=list)
    symbol_scores: dict[str, str] = field(default_factory=dict)

    @property
    def is_usable_for_research(self) -> bool:
        return self.tier in ("DATA_QUALIFIED_FULL", "DATA_QUALIFIED_FUNDING_ONLY")

    @property
    def is_eligible_for_live(self) -> bool:
        return self.tier == "DATA_QUALIFIED_FULL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cert_id": self.cert_id,
            "created_ts_ms": self.created_ts_ms,
            "dataset_name": self.dataset_name,
            "venue_pair": self.venue_pair,
            "dataset_sha256": self.dataset_sha256,
            "provenance": self.provenance,
            "row_count": self.row_count,
            "basis_coverage_pct": self.basis_coverage_pct,
            "tier": self.tier,
            "passed_count": self.passed_count,
            "total_count": self.total_count,
            "is_usable_for_research": self.is_usable_for_research,
            "is_eligible_for_live": self.is_eligible_for_live,
            "checks": [c.to_dict() for c in self.checks],
            "symbol_scores": dict(self.symbol_scores),
        }


@dataclass
class MicroRunRecord:
    """State tracking record for 24h-72h Fast Micro-Run validation sprints."""

    run_id: str
    plugin_id: str
    strategy_id: str
    version: str
    status: str         # ACTIVE | COMPLETED_PASS | KILLED_EARLY | ITERATE_AGENT | READY_FOR_HUMAN_LIVE
    started_ts_ms: int
    updated_ts_ms: int
    target_hours: float
    elapsed_hours: float
    dq_cert_id: str
    dq_tier: str
    ticks_observed: int
    realized_spread_bps_median: float
    assumed_slippage_bps: float
    slippage_ratio: float
    paper_net_pnl_usd: float
    paper_drawdown_pct: float
    kill_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketEvent:
    """Normalized market event for replayable MicroRunSession execution (P0-5/P2-2)."""

    kind: str           # BBO | FUNDING | CANDLE | DEX_QUOTE | MACRO_EVENT
    ts_ms: int
    venue: str
    symbol: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrderIntent:
    """Declarative non-live execution intent produced by a strategy plugin during micro-runs."""

    strategy_id: str
    symbol: str
    action: str         # OPEN_SHORT_A_LONG_B | OPEN_LONG_A_SHORT_B | CLOSE | HOLD
    notional_usd: float
    expected_edge_bps: float
    reason: str
    ts_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


