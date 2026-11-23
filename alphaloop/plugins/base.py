"""Strategy Factory v2 Plugin Interface & Registry (Architect P0-5 Decoupled).

Every strategy family implements BaseStrategyPlugin so SixGateVerifier,
AgentEvolutionOrchestrator, MicroRunSession, CLI, and Web Console remain
100% strategy-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from alphaloop.core.models import (
    BacktestMetrics,
    FeedbackEvent,
    MarketEvent,
    OrderIntent,
    StrategyConfig,
)
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import BacktestResult


class BaseStrategyPlugin(ABC):
    """Abstract contract for pluggable quantitative strategies in AlphaLoop v2."""

    plugin_id: str = ""
    display_name: str = ""
    strategy_family: str = ""
    description: str = ""
    required_data_spec: dict[str, Any] = {}

    @abstractmethod
    def default_config(self, symbols: list[str] | None = None, **overrides: Any) -> StrategyConfig:
        """Return a safe, valid default StrategyConfig for this plugin."""

    @abstractmethod
    def mutable_param_bounds(self) -> dict[str, tuple[float, float]]:
        """Return bounded numeric mutation whitelist allowed for AI Agent optimization."""

    @abstractmethod
    def validate_and_clamp(
        self,
        cfg: StrategyConfig,
        next_version: str,
        changes: dict[str, Any],
    ) -> StrategyConfig:
        """Apply bounded changes strictly within safe invariants and return new StrategyConfig."""

    @abstractmethod
    def evaluate(
        self,
        store: MarketDataStore,
        cfg: StrategyConfig,
        start_ms: int | None = None,
        end_ms: int | None = None,
        extra_slippage_bps_per_leg: float = 0.0,
        execution_mode_override: str | None = None,
    ) -> BacktestResult:
        """Run strictly causal historical evaluation on MarketDataStore (or declared dataset)."""

    @abstractmethod
    def plateau_variants(self, cfg: StrategyConfig) -> list[tuple[str, StrategyConfig]]:
        """Generate a 3x3 parameter neighborhood for Gate G4 stability verification."""

    @abstractmethod
    def diagnose_telemetry(self, cfg: StrategyConfig, metrics: BacktestMetrics) -> FeedbackEvent:
        """Compress evaluation/micro-run metrics into a structured anomaly FeedbackEvent."""

    @abstractmethod
    def fallback_proposal(
        self,
        cfg: StrategyConfig,
        feedback: FeedbackEvent,
        metrics: BacktestMetrics,
    ) -> tuple[dict[str, Any], str, str, str]:
        """Return deterministic fallback (param_changes, hypothesis, economic_rationale, kill_criteria)."""

    @abstractmethod
    def on_event(
        self,
        event: MarketEvent,
        state: dict[str, Any],
        cfg: StrategyConfig,
    ) -> list[OrderIntent]:
        """Pure event-driven step handler for replayable MicroRunSession (zero hidden I/O)."""

    def min_economic_kill_horizon_hours(self, cfg: StrategyConfig) -> float:
        """Minimum elapsed hours before economic edge-decay Early-Kill may fire (Architect P1-4)."""
        return max(24.0, float(cfg.min_hold_hours))

    def metadata(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "display_name": self.display_name,
            "strategy_family": self.strategy_family,
            "description": self.description,
            "required_data_spec": dict(self.required_data_spec),
            "mutable_param_bounds": {k: [v[0], v[1]] for k, v in self.mutable_param_bounds().items()},
        }


class PluginRegistry:
    """Thread-safe lookup registry for AlphaLoop strategy plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, BaseStrategyPlugin] = {}

    def register(self, plugin: BaseStrategyPlugin) -> None:
        if not plugin.plugin_id:
            raise ValueError("Plugin must declare a non-empty plugin_id")
        self._plugins[plugin.plugin_id] = plugin

    def get(self, plugin_id: str) -> BaseStrategyPlugin:
        pid = plugin_id or "funding_carry"
        if pid not in self._plugins:
            raise KeyError(f"Unknown strategy plugin_id: {pid!r} (registered: {sorted(self._plugins)})")
        return self._plugins[pid]

    def resolve_for_config(self, cfg: StrategyConfig) -> BaseStrategyPlugin:
        return self.get(cfg.plugin_id or "funding_carry")

    def list_plugins(self) -> list[dict[str, Any]]:
        return [p.metadata() for _, p in sorted(self._plugins.items())]
