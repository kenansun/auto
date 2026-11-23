"""AlphaLoop v2.2 Always-On Shadow Daemon, Parity Auditor & Watchdog Integration."""

from alphaloop.daemon.shadow_runner import (
    LIVE_EXECUTION_ENABLED,
    ShadowParityAuditor,
    ShadowParityRecord,
    ShadowPortfolioDaemon,
)
from alphaloop.daemon.watchdog_hook import AlertNotifier, WatchdogHook

__all__ = [
    "AlertNotifier",
    "LIVE_EXECUTION_ENABLED",
    "ShadowParityAuditor",
    "ShadowParityRecord",
    "ShadowPortfolioDaemon",
    "WatchdogHook",
]
