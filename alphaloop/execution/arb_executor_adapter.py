"""Strict Decimal Execution-Plane Contract Adapter (AlphaLoop -> Institutional DEX-CEX Executor, Stage 2 S2-02).

Translates verified AlphaLoop `ExecutionPlan` & `ExecutionAuditChain` records into exact
`Decimal`-stringified envelopes compatible with external execution-plane `PlanRiskInput` and
`AuditRecord` wire contracts (zero floating-point drift, zero live order placement).

Hard Safety Invariants:
  - `LIVE_EXECUTION_ENABLED: Final[bool] = False` (shadow/dry-run contract translation only).
  - Refuses `BLOCKED` plans, non-shadow run modes, empty legs, non-finite/non-positive prices or
    notionals, stale quotes (`now_ms - bbo_ts_ms > max_quote_age_ms`), and secret-like keys.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from alphaloop.execution.bridge import LIVE_EXECUTION_ENABLED, ExecutionPlan, LegSpec

ADAPTER_LIVE_EXECUTION_ENABLED: Final[bool] = False
_SECRET_KEY_SUBSTRINGS = (
    "secret",
    "privatekey",
    "privkey",
    "apikey",
    "passphrase",
    "mnemonic",
    "token",
    "password",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "seed",
)
_ALLOWED_LEG_SIDES = frozenset({"BUY", "SELL", "CLOSE"})
_MAX_DECIMAL_EXPONENT = 18


def _to_strict_positive_decimal_str(val: Any, field_name: str) -> str:
    """Convert numeric input to exact positive finite Decimal string (rejects bool/NaN/Inf/<=0/extreme exp)."""
    if isinstance(val, bool) or not isinstance(val, (int, float, Decimal, str)):
        raise ValueError(f"INVALID_DECIMAL_TYPE:{field_name}")
    if isinstance(val, float) and not math.isfinite(val):
        raise ValueError(f"NON_FINITE_DECIMAL:{field_name}")
    try:
        dec = val if isinstance(val, Decimal) else Decimal(str(val))
    except InvalidOperation as exc:
        raise ValueError(f"INVALID_DECIMAL_VALUE:{field_name}") from exc
    if not dec.is_finite() or dec <= Decimal("0"):
        raise ValueError(f"NON_POSITIVE_DECIMAL:{field_name}")
    if abs(dec.adjusted()) > _MAX_DECIMAL_EXPONENT:
        raise ValueError(f"OUT_OF_RANGE_DECIMAL:{field_name}")
    return format(dec.normalize(), "f")


def _sanitize_payload(obj: Any) -> Any:
    """Ensure simple JSON types only, finite floats, and mask any secret-like dictionary keys."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("NON_FINITE_AUDIT_PAYLOAD")
        return obj
    if isinstance(obj, Decimal):
        if not obj.is_finite() or abs(obj.adjusted()) > _MAX_DECIMAL_EXPONENT:
            raise ValueError("NON_FINITE_AUDIT_PAYLOAD")
        return format(obj.normalize(), "f")
    if isinstance(obj, (list, tuple)):
        return [_sanitize_payload(item) for item in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            norm = re.sub(r"[-_\s]", "", ks.lower())
            if any(sub in norm for sub in _SECRET_KEY_SUBSTRINGS):
                out[ks] = "***REDACTED***"
            else:
                out[ks] = _sanitize_payload(v)
        return out
    raise ValueError(f"UNSUPPORTED_AUDIT_PAYLOAD_TYPE:{type(obj).__name__}")


def to_arb_plan_risk_envelope(
    plan: ExecutionPlan,
    *,
    now_ms: int,
    max_quote_age_ms: int = 30_000,
) -> dict[str, Any]:
    """Translate a valid PLANNED_SHADOW ExecutionPlan to a strict Decimal PlanRiskInput wire envelope."""
    if LIVE_EXECUTION_ENABLED or ADAPTER_LIVE_EXECUTION_ENABLED:
        raise RuntimeError("Adapter forbids live execution mode")
    if not isinstance(plan, ExecutionPlan):
        raise ValueError("INVALID_PLAN_TYPE")
    if plan.run_mode != "SHADOW_DRY_RUN":
        raise ValueError(f"UNSUPPORTED_RUN_MODE:{plan.run_mode}")
    if plan.status != "PLANNED_SHADOW":
        raise ValueError(f"PLAN_NOT_ELIGIBLE:{plan.status}:{plan.block_reason}")
    if not plan.plan_id or not plan.idempotency_key or not plan.symbol:
        raise ValueError("MISSING_PLAN_IDENTIFIERS")
    if isinstance(plan.ts_ms, bool) or not isinstance(plan.ts_ms, int) or plan.ts_ms <= 0:
        raise ValueError("INVALID_PLAN_TS_MS")
    if not plan.legs:
        raise ValueError("EMPTY_PLAN_LEGS")
    if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms <= 0:
        raise ValueError("INVALID_NOW_MS")
    if isinstance(max_quote_age_ms, bool) or not isinstance(max_quote_age_ms, int) or max_quote_age_ms <= 0:
        raise ValueError("INVALID_MAX_QUOTE_AGE_MS")

    notional_dec = _to_strict_positive_decimal_str(plan.notional_usd, "notional_usd")
    quote_prices: list[str] = []
    leg_envelopes: list[dict[str, Any]] = []
    oldest_bbo_ms: int | None = None

    for leg in plan.legs:
        if not isinstance(leg, LegSpec):
            raise ValueError("INVALID_LEG_TYPE")
        if not str(leg.leg_id or "").strip() or not str(leg.venue or "").strip():
            raise ValueError("MISSING_LEG_IDENTIFIERS")
        if leg.symbol != plan.symbol:
            raise ValueError(f"LEG_SYMBOL_MISMATCH:{leg.symbol}!={plan.symbol}")
        if leg.side not in _ALLOWED_LEG_SIDES:
            raise ValueError(f"INVALID_LEG_SIDE:{leg.side}")
        if isinstance(leg.bbo_ts_ms, bool) or not isinstance(leg.bbo_ts_ms, int) or leg.bbo_ts_ms <= 0:
            raise ValueError(f"INVALID_LEG_BBO_TS:{leg.leg_id}")
        age_ms = now_ms - leg.bbo_ts_ms
        if age_ms < -5_000:
            raise ValueError(f"FUTURE_QUOTE_TS:{leg.leg_id}:lead_ms={-age_ms}")
        if age_ms > max_quote_age_ms:
            raise ValueError(f"STALE_QUOTE:{leg.leg_id}:age_ms={age_ms}>{max_quote_age_ms}")

        ref_px_str = _to_strict_positive_decimal_str(leg.ref_price, f"{leg.leg_id}.ref_price")
        lim_px_str = _to_strict_positive_decimal_str(leg.limit_price, f"{leg.leg_id}.limit_price")
        leg_notional_str = _to_strict_positive_decimal_str(leg.notional_usd, f"{leg.leg_id}.notional_usd")
        quote_prices.append(ref_px_str)
        oldest_bbo_ms = leg.bbo_ts_ms if oldest_bbo_ms is None else min(oldest_bbo_ms, leg.bbo_ts_ms)
        leg_envelopes.append({
            "leg_id": leg.leg_id,
            "venue": leg.venue,
            "symbol": leg.symbol,
            "side": leg.side,
            "notional_quote": leg_notional_str,
            "ref_price_quote": ref_px_str,
            "limit_price_quote": lim_px_str,
            "bbo_utc_ns": int(leg.bbo_ts_ms) * 1_000_000,
        })

    return {
        "schema_version": "alphaloop.plan_risk_envelope.v1",
        "live_execution_enabled": False,
        "run_mode": "SHADOW_DRY_RUN",
        "plan_id": plan.plan_id,
        "idempotency_key": plan.idempotency_key,
        "strategy_id": plan.strategy_id,
        "strategy_version": plan.version,
        "config_sha256": plan.config_sha256,
        "symbol": plan.symbol,
        "action": plan.action,
        "notional_quote": notional_dec,
        "quote_prices": tuple(quote_prices),
        "quote_age_ms": max(0, now_ms - int(oldest_bbo_ms or now_ms)),
        "created_utc_ns": int(plan.ts_ms) * 1_000_000,
        "legs": leg_envelopes,
    }


def to_arb_audit_record_envelope(audit_row: dict[str, Any]) -> dict[str, Any]:
    """Translate an AlphaLoop ExecutionAuditChain row into a sanitized AuditRecord wire envelope."""
    if not isinstance(audit_row, dict):
        raise ValueError("INVALID_AUDIT_ROW")
    run_mode = str(audit_row.get("run_mode", "SHADOW_DRY_RUN"))
    if run_mode != "SHADOW_DRY_RUN":
        raise ValueError(f"UNSUPPORTED_AUDIT_RUN_MODE:{run_mode}")
    ts_ms = audit_row.get("ts_ms")
    if isinstance(ts_ms, bool) or not isinstance(ts_ms, int) or ts_ms <= 0:
        raise ValueError("INVALID_AUDIT_TS_MS")
    event_type = str(audit_row.get("event_type", "")).strip()
    if not event_type:
        raise ValueError("MISSING_AUDIT_EVENT_TYPE")

    raw_payload = audit_row.get("payload")
    if raw_payload is None and "payload_json" in audit_row:
        import json
        raw_payload = json.loads(str(audit_row["payload_json"]))
    clean_payload = _sanitize_payload(raw_payload) if raw_payload is not None else {}
    if not isinstance(clean_payload, dict):
        raise ValueError("AUDIT_PAYLOAD_MUST_BE_OBJECT")

    return {
        "schema_version": "alphaloop.audit_record_envelope.v1",
        "event_type": event_type,
        "run_mode": "SHADOW_DRY_RUN",
        "rule_id": str(audit_row.get("rule_id", "")),
        "key_name": str(audit_row.get("key_name", "")),
        "utc_ns": int(ts_ms) * 1_000_000,
        "plan_id": str(audit_row.get("plan_id") or "") or None,
        "idempotency_key": str(audit_row.get("idempotency_key") or ""),
        "record_hash": str(audit_row.get("record_hash") or ""),
        "payload": clean_payload,
    }
