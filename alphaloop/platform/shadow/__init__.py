"""AlphaLoop Platform v4 — Read-Only Shadow Paper Monitoring Package.

Strict Security & Governance Boundaries:
- `SHADOW_LIVE_EXECUTION_ENABLED = False` (hard-coded; zero private keys, zero order signing)
- STD-32 compliant: strictly forbids direct `wss://stream.binance.com` connections
  (reuses P8 MD-Gateway feed interface or public REST Waterfall + Jitter)
- STD-34 compliant: SQLite WAL + busy_timeout, rotating JSONL trace logs,
  PlatformErrorCode anti-storm throttling, monotonic dead-man switch, P16 heartbeat,
  and SHA-256 tamper-evident audit chain.
"""

from alphaloop.platform.shadow.w3_dn_shadow_monitor import (
    DEFAULT_W3_DN_SHADOW_COINS,
    SHADOW_LIVE_EXECUTION_ENABLED,
    P8MDGatewayFileQuoteProvider,
    PairMarketSnapshot,
    PlatformErrorCode,
    PublicRestWaterfallQuoteProvider,
    ReplayQuoteProvider,
    ShadowMonitorConfig,
    ShadowReadinessReport,
    W3DNShadowMonitor,
    estimate_l2_vwap_slippage_bps,
)

__all__ = [
    "DEFAULT_W3_DN_SHADOW_COINS",
    "SHADOW_LIVE_EXECUTION_ENABLED",
    "P8MDGatewayFileQuoteProvider",
    "PairMarketSnapshot",
    "PlatformErrorCode",
    "PublicRestWaterfallQuoteProvider",
    "ReplayQuoteProvider",
    "ShadowMonitorConfig",
    "ShadowReadinessReport",
    "W3DNShadowMonitor",
    "estimate_l2_vwap_slippage_bps",
]
