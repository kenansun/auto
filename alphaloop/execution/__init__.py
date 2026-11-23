"""Shadow/Paper-only Execution Bridge connecting AlphaLoop Strategy Factory to execution contracts."""

from __future__ import annotations

from alphaloop.execution.arb_executor_adapter import (
    ADAPTER_LIVE_EXECUTION_ENABLED,
    to_arb_audit_record_envelope,
    to_arb_plan_risk_envelope,
)
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
    "ADAPTER_LIVE_EXECUTION_ENABLED",
    "GENESIS_PREV_HASH",
    "LIVE_EXECUTION_ENABLED",
    "KillTrigger",
    "LegSpec",
    "ExecutionPlan",
    "ExecutionKillSwitch",
    "ExecutionAuditChain",
    "ExecutionBridge",
    "HotConfigWatcher",
    "to_arb_plan_risk_envelope",
    "to_arb_audit_record_envelope",
]
