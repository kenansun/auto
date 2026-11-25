"""Track G — Gold Dedicated Strategy Suite (Strictly Isolated from Web3 & A-Share Domains).

Exports:
  - `GoldMacroEngine`: Stateless 7-driver PIT Gold/Macro regime engine (F-16)
  - `GoldTradFiMacroKernel`: G1 Single-Instrument GC=F/GLD Asymmetric Macro Core (F-17)
  - `Gold4hEventKernel`: G2 Gold 4h Tactical Breakout + PIT Macro Event Engine A/B/C (F-18)
  - `GoldPAXGBridgeKernel`: G3 PAXG Dual-Basis, Weekend Guard & Funding Bridge (F-19)
  - `GoldMinerSpreadMRKernel`: G4 Causal Vol-Matched GLD/GDX Mean-Reversion Orthogonal Sleeve (Phase C)
"""

from alphaloop.platform.strategies.gold.macro_engine import GoldMacroEngine
from alphaloop.platform.strategies.gold.miner_spread_mr import GoldMinerSpreadMRKernel
from alphaloop.platform.strategies.gold.paxg_bridge import GoldPAXGBridgeKernel
from alphaloop.platform.strategies.gold.session_event_4h import Gold4hEventKernel
from alphaloop.platform.strategies.gold.tradfi_macro_core import GoldTradFiMacroKernel

__all__ = [
    "GoldMacroEngine",
    "GoldTradFiMacroKernel",
    "GoldMinerSpreadMRKernel",
    "Gold4hEventKernel",
    "GoldPAXGBridgeKernel",
]
