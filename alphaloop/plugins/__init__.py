"""Strategy Factory v2 Plugin Registry & Reference Plugins."""

from __future__ import annotations

from alphaloop.plugins.base import BaseStrategyPlugin, PluginRegistry
from alphaloop.plugins.basis_reversion_plugin import BasisReversionPlugin
from alphaloop.plugins.funding_carry_plugin import FundingCarryPlugin
from alphaloop.plugins.structural_arb_plugin import StructuralArbPlugin
from alphaloop.plugins.trend_cta_plugin import TrendCTAPlugin


def default_registry() -> PluginRegistry:
    """Create a PluginRegistry populated with all 4 reference AlphaLoop v2 strategy plugins."""
    reg = PluginRegistry()
    reg.register(FundingCarryPlugin())
    reg.register(BasisReversionPlugin())
    reg.register(StructuralArbPlugin())
    reg.register(TrendCTAPlugin())
    return reg


__all__ = [
    "BaseStrategyPlugin",
    "PluginRegistry",
    "FundingCarryPlugin",
    "BasisReversionPlugin",
    "StructuralArbPlugin",
    "TrendCTAPlugin",
    "default_registry",
]
