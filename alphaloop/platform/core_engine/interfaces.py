"""Asset-Agnostic Strategy Interfaces for AlphaLoop Strategy R&D Platform v4.

Defines the minimal, strictly causal contract shared across decoupled asset-native
strategy tracks (`web3`, `gold`, `ashare`) without leaking any asset-specific signal
formulas, regime maps, or parameter grids into the shared core engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from alphaloop.platform.data_lake import CanonicalBar

VALID_ASSET_DOMAINS: frozenset[str] = frozenset({"web3", "gold", "ashare"})


@dataclass(frozen=True)
class BarDecision:
    """Strictly causal decision emitted at bar close (`ts_event_ms`) to fill at next bar open."""

    symbol: str
    ts_event_ms: int
    target_weight: float
    raw_signal: float
    regime_label: str
    stop_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("BarDecision.symbol must be non-empty")
        if not math.isfinite(self.target_weight) or not math.isfinite(self.raw_signal):
            raise ValueError(
                f"Non-finite decision for {self.symbol}: "
                f"target_weight={self.target_weight}, raw_signal={self.raw_signal}"
            )
        if self.stop_price is not None and (not math.isfinite(self.stop_price) or self.stop_price <= 0.0):
            raise ValueError(f"Invalid stop_price for {self.symbol}: {self.stop_price}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ts_event_ms": int(self.ts_event_ms),
            "target_weight": float(self.target_weight),
            "raw_signal": float(self.raw_signal),
            "regime_label": str(self.regime_label),
            "stop_price": None if self.stop_price is None else float(self.stop_price),
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class AssetNativeStrategy(Protocol):
    """Asset-agnostic protocol implemented independently by Web3, Gold, and A-Share engines."""

    strategy_id: str
    asset_domain: str  # "web3" | "gold" | "ashare"
    warmup_bars: int

    def reset(self) -> None:
        """Reset all mutable internal state to clean initial conditions."""
        ...

    def on_bar_step(
        self,
        ts_ms: int,
        bars_at_ts: dict[str, CanonicalBar],
        context: dict[str, Any] | None = None,
    ) -> dict[str, BarDecision]:
        """Consume all closed bars at `ts_ms` and emit strictly causal next-open decisions."""
        ...

    def sync_executed_weights(self, executed_weights: dict[str, float]) -> None:
        """Synchronize actual post-allocator/post-fill weights for Carver inertia & G7 parity."""
        ...

    def clone_fresh(self) -> AssetNativeStrategy:
        """Return a brand-new, freshly reset strategy instance with identical parameters."""
        ...
