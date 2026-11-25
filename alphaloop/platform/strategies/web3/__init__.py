"""Track W: Crypto-Native Web3 Dedicated Strategy Suite (AlphaLoop Platform v4).

Strictly isolated from TradFi Gold, Yahoo `^VIX`, and CN A-Share contracts:
  - `Web3MarketRegimeEngine` (`regime_engine.py`): Crypto-native regime classification
  - `Web3XSMomCarryKernel` (`xs_mom_carry.py`): W1 Cross-Sectional Momentum + Funding/OI Tilt
  - `Web3AdaptiveTrendKernel` (`adaptive_trend.py`): W2 Vol-Managed Adaptive Breakout/Trend CTA
  - `Web3RegimeCarryKernel` (`regime_carry.py`): W3 Regime-Gated Funding & Basis Carry
"""

from alphaloop.platform.strategies.web3.adaptive_trend import (
    ALLOWED_W2_FACTORS,
    DEFAULT_W2_FACTORS,
    Web3AdaptiveTrendKernel,
)
from alphaloop.platform.strategies.web3.delta_neutral_carry import (
    DN_REGIME_ACTIVE,
    DN_REGIME_BASIS_STOP,
    DN_REGIME_DVOL_THROTTLED,
    DN_REGIME_MARGIN_STORM_EXIT,
    DN_REGIME_SUB_FLOOR,
    DN_REGIME_WARMUP,
    Web3DeltaNeutralCarryKernel,
    build_dual_leg_carry_universe,
    parse_pair_leg,
    perp_leg_symbol,
    spot_leg_symbol,
)
from alphaloop.platform.strategies.web3.regime_carry import (
    REGIME_CARRY_ACTIVE,
    REGIME_CARRY_DVOL_THROTTLED,
    REGIME_CARRY_SUB_FLOOR,
    REGIME_CARRY_WARMUP,
    REGIME_MARGIN_STORM_EXIT,
    Web3RegimeCarryKernel,
)
from alphaloop.platform.strategies.web3.regime_engine import (
    VALID_WEB3_REGIMES,
    WEB3_REGIME_ALT_SEASON,
    WEB3_REGIME_CROWDED_STRESS,
    WEB3_REGIME_EXPANSION,
    WEB3_REGIME_NEUTRAL,
    WEB3_REGIME_RISK_OFF_BEAR,
    Web3MarketRegimeEngine,
    Web3RegimeSnapshot,
    annualized_carry_apr_pct,
    is_crypto_bar,
)
from alphaloop.platform.strategies.web3.xs_mom_carry import (
    DEFAULT_WEB3_PIT_LISTING_DATES,
    HYPE_PIT_LISTING_MS,
    Web3XSMomCarryKernel,
)

__all__ = [
    "ALLOWED_W2_FACTORS",
    "DEFAULT_W2_FACTORS",
    "DEFAULT_WEB3_PIT_LISTING_DATES",
    "DN_REGIME_ACTIVE",
    "DN_REGIME_BASIS_STOP",
    "DN_REGIME_DVOL_THROTTLED",
    "DN_REGIME_MARGIN_STORM_EXIT",
    "DN_REGIME_SUB_FLOOR",
    "DN_REGIME_WARMUP",
    "HYPE_PIT_LISTING_MS",
    "REGIME_CARRY_ACTIVE",
    "REGIME_CARRY_DVOL_THROTTLED",
    "REGIME_CARRY_SUB_FLOOR",
    "REGIME_CARRY_WARMUP",
    "REGIME_MARGIN_STORM_EXIT",
    "VALID_WEB3_REGIMES",
    "WEB3_REGIME_ALT_SEASON",
    "WEB3_REGIME_CROWDED_STRESS",
    "WEB3_REGIME_EXPANSION",
    "WEB3_REGIME_NEUTRAL",
    "WEB3_REGIME_RISK_OFF_BEAR",
    "Web3AdaptiveTrendKernel",
    "Web3DeltaNeutralCarryKernel",
    "Web3MarketRegimeEngine",
    "Web3RegimeCarryKernel",
    "Web3RegimeSnapshot",
    "Web3XSMomCarryKernel",
    "annualized_carry_apr_pct",
    "build_dual_leg_carry_universe",
    "is_crypto_bar",
    "parse_pair_leg",
    "perp_leg_symbol",
    "spot_leg_symbol",
]
