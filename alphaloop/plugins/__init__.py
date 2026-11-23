"""Strategy Factory v2 Plugin Registry & Reference Plugins."""

from __future__ import annotations

from alphaloop.plugins.base import BaseStrategyPlugin, PluginRegistry
from alphaloop.plugins.basis_reversion_plugin import BasisReversionPlugin
from alphaloop.plugins.funding_carry_plugin import FundingCarryPlugin
from alphaloop.plugins.structural_arb_plugin import StructuralArbPlugin


def default_registry() -> PluginRegistry:
    """Create a PluginRegistry populated with all 3 reference AlphaLoop v2 strategy plugins."""
    reg = PluginRegistry()
    reg.register(FundingCarryPlugin())
    reg.register(BasisReversionPlugin())
    reg.register(StructuralArbPlugin())
    return reg


__all__ = [
    "BaseStrategyPlugin",
    "PluginRegistry",
    "FundingCarryPlugin",
    "BasisReversionPlugin",
    "StructuralArbPlugin",
    "default_registry",
]
