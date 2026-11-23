"""Shadow/Paper-only Execution Bridge connecting AlphaLoop Strategy Factory to execution contracts."""

from __future__ import annotations

from alphaloop.execution.bridge import (
    GENESIS_PREV_HASH,
    LIVE_EXECUTION_ENABLED,
    ExecutionAuditChain,
    ExecutionBridge,
    ExecutionKillSwitch,
    ExecutionPlan,
    HotConfigWatcher,
    KillTrigger,
    LegSpec,
)

__all__ = [
    "GENESIS_PREV_HASH",
    "LIVE_EXECUTION_ENABLED",
    "KillTrigger",
    "LegSpec",
    "ExecutionPlan",
    "ExecutionKillSwitch",
    "ExecutionAuditChain",
    "HotConfigWatcher",
    "ExecutionBridge",
]
