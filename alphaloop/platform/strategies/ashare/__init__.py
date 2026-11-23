"""Track A — Physically Isolated CN A-Share Native Strategy Package.

Zero imports from `alphaloop.platform.strategies.web3` or `alphaloop.platform.strategies.gold`.
"""

from alphaloop.platform.strategies.ashare.etf_rotation import AShareETFRotationKernel
from alphaloop.platform.strategies.ashare.macro_erp_rotation import (
    AShareMacroERPRotationKernel,
    CNMacroRegimeSnapshot,
    CNMacroRegimeTimeline,
)

__all__ = [
    "AShareETFRotationKernel",
    "AShareMacroERPRotationKernel",
    "CNMacroRegimeSnapshot",
    "CNMacroRegimeTimeline",
]
