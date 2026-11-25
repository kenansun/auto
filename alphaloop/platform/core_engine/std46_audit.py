"""SCH-STD-46 Companion Audit Layer for AlphaLoop Strategy R&D Platform v4.

Pure Python 3.12 stdlib companion module providing fail-closed verification helpers
without modifying `alphaloop/platform/core_engine/validation.py` (0-byte untouched):
  - `VER-02` Dual Sharpe (`compute_dual_sharpe`: raw Rf=0% + risk-free excess Sharpe)
  - `VER-03` Static-Baseline Ablation Dominance (`evaluate_static_ablation`)
  - `VER-04` Composite Portfolio Production Eligibility & FX Disclosure (`audit_composite_portfolio`)
  - `STD-46` Report Assembly (`build_std46_audit_report`, compatible with SCHQ `--mode report`)
"""

from __future__ import annotations

import math
import re
from typing import Any, Final, Sequence

VERDICT_CAP_DOMINATED: Final[str] = "STATIC_ALLOCATION_DOMINATED"
VERDICT_CAP_PASS: Final[str] = "PASS"
CLASSIFICATION_PRODUCTION: Final[str] = "PRODUCTION_ELIGIBLE"
CLASSIFICATION_DIAGNOSTIC: Final[str] = "DIAGNOSTIC_HYPOTHETICAL_ONLY"
DEFAULT_USD_RF_PCT: Final[float] = 4.5
DEFAULT_RMB_RF_PCT: Final[float] = 1.5

_GATE_RATIO_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
_CURRENCY_SPLIT_RE = re.compile(r"[+/,;\-\s]+")

_DISQUALIFYING_SLEEVE_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "REJECTED",
        "FAIL",
        "FAILED",
        "CONDITIONAL_RESEARCH",
        "STATIC_ALLOCATION_DOMINATED",
        "DIAGNOSTIC_HYPOTHETICAL_ONLY",
    }
)

_CAGR_KEYS: Final[tuple[str, ...]] = ("cagr_pct", "cagr")
_SR_KEYS: Final[tuple[str, ...]] = ("excess_sharpe", "sharpe", "sharpe_ratio", "raw_sharpe")
_MAXDD_KEYS: Final[tuple[str, ...]] = (
    "max_dd_pct",
    "maxdd_pct",
    "max_drawdown_pct",
    "max_dd",
    "maxdd",
    "max_drawdown",
)
_CALMAR_KEYS: Final[tuple[str, ...]] = ("calmar", "calmar_ratio")

_VACUOUS_FORBIDDEN_CLAIMS: Final[frozenset[str]] = frozenset(
    {
        "SHADOW_READY",
        "PARITY_VERIFIED",
        "PARITY_OK",
        "READY_FOR_CANARY",
        "ACTIVE_EXECUTION",
        "READY_FOR_OWNER_REVIEW",
        "PROMOTE",
        "PROMOTE_SHADOW_CANDIDATE",
        "PRODUCTION",
        "PRODUCTION_ELIGIBLE",
        "APPROVE",
        "APPROVED",
        "LIVE",
        "CANARY",
    }
)

_SHADOW_PLAN_KEYS: Final[tuple[str, ...]] = ("plans_created", "orders_planned", "execution_plans")
_SHADOW_TRADE_KEYS: Final[tuple[str, ...]] = ("executed_trades", "trades_executed", "virtual_trades", "fills")
_VER04_BAD_SLEEVE_TOKENS: Final[frozenset[str]] = frozenset(
    {"REJECTED", "FAIL", "CONDITIONAL_RESEARCH", "STATIC_ALLOCATION_DOMINATED"}
)


def _require_finite_float(val: Any, name: str) -> float:
    """Validate and coerce a strictly finite numeric value, rejecting None/bool/NaN/Inf."""
    if val is None or isinstance(val, bool):
        raise ValueError(f"{name} must be a finite real number, got {val!r}")
    if not isinstance(val, (int, float)):
        if isinstance(val, str):
            text = val.strip()
            if not text:
                raise ValueError(f"{name} must be a finite real number, got empty string")
            try:
                out = float(text)
            except ValueError as exc:
                raise ValueError(f"{name} must be a finite real number, got {val!r}") from exc
        else:
            raise ValueError(f"{name} must be a finite real number, got {type(val).__name__}")
    else:
        out = float(val)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {val!r}")
    return out


def _extract_pair_metric(
    strategy_metrics: dict[str, Any],
    static_metrics: dict[str, Any],
    keys: tuple[str, ...],
    metric_label: str,
) -> tuple[float, float]:
    """Extract a metric pair using common-key priority first (apples-to-apples), fail-closed if missing."""
    for key in keys:
        if (
            key in strategy_metrics
            and key in static_metrics
            and strategy_metrics[key] is not None
            and static_metrics[key] is not None
        ):
            return (
                _require_finite_float(strategy_metrics[key], f"strategy_metrics.{key}"),
                _require_finite_float(static_metrics[key], f"static_metrics.{key}"),
            )

    s_key = next((k for k in keys if k in strategy_metrics and strategy_metrics[k] is not None), None)
    b_key = next((k for k in keys if k in static_metrics and static_metrics[k] is not None), None)
    if s_key is None:
        raise ValueError(
            f"strategy_metrics missing required {metric_label} metric (accepted keys: {', '.join(keys)})"
        )
    if b_key is None:
        raise ValueError(
            f"static_metrics missing required {metric_label} metric (accepted keys: {', '.join(keys)})"
        )
    return (
        _require_finite_float(strategy_metrics[s_key], f"strategy_metrics.{s_key}"),
        _require_finite_float(static_metrics[b_key], f"static_metrics.{b_key}"),
    )


def _normalize_currency_tokens(raw: Any) -> set[str]:
    """Normalize currency tokens to uppercase canonical ISO codes (mapping CNY -> RMB)."""
    tokens: set[str] = set()
    if raw is None:
        return tokens
    if isinstance(raw, str):
        parts = _CURRENCY_SPLIT_RE.split(raw.strip().upper())
        for part in parts:
            if not part:
                continue
            tokens.add("RMB" if part == "CNY" else part)
        return tokens
    if isinstance(raw, (set, frozenset, list, tuple)):
        for item in raw:
            if not isinstance(item, str):
                raise ValueError(f"Currency token must be str, got {type(item).__name__}")
            tokens.update(_normalize_currency_tokens(item))
        return tokens
    raise ValueError(f"Unsupported currencies container type: {type(raw).__name__}")


def _inspect_sleeve_gates(sleeve: dict[str, Any]) -> tuple[bool, str, int | None, int | None]:
    """Evaluate EVERY present gate indicator on a sleeve (fail-closed on any failure or contradiction)."""
    found_any = False
    failures: list[str] = []
    passing_ratios: list[tuple[int, int]] = []
    failing_ratios: list[tuple[int, int]] = []

    def _check_int_pair(gp_key: str, gt_key: str) -> None:
        nonlocal found_any
        has_gp = gp_key in sleeve and sleeve[gp_key] is not None
        has_gt = gt_key in sleeve and sleeve[gt_key] is not None
        if not has_gp:
            if has_gt:
                gt_only = sleeve[gt_key]
                if isinstance(gt_only, bool) or not isinstance(gt_only, int) or gt_only <= 0:
                    found_any = True
                    failures.append(f"invalid {gt_key}={gt_only!r}")
            return
        found_any = True
        gp_raw = sleeve[gp_key]
        gt_raw = sleeve[gt_key] if has_gt else 8
        if isinstance(gp_raw, bool) or not isinstance(gp_raw, int) or gp_raw < 0:
            failures.append(f"invalid {gp_key}={gp_raw!r}")
            return
        if isinstance(gt_raw, bool) or not isinstance(gt_raw, int) or gt_raw <= 0:
            failures.append(f"invalid {gt_key}={gt_raw!r}")
            return
        if gp_raw > gt_raw:
            failures.append(f"invalid gates {gp_raw}/{gt_raw} ({gp_key} > {gt_key})")
            failing_ratios.append((gp_raw, gt_raw))
            return
        if gp_raw < 8 or gp_raw < gt_raw:
            failures.append(f"gates {gp_raw}/{gt_raw} < 8/8")
            failing_ratios.append((gp_raw, gt_raw))
            return
        passing_ratios.append((gp_raw, gt_raw))

    _check_int_pair("gates_passed", "gates_total")
    _check_int_pair("passed_gates", "total_gates")

    for key in ("gates", "gate_score"):
        if key in sleeve and sleeve[key] is not None:
            found_any = True
            raw = sleeve[key]
            if not isinstance(raw, str):
                failures.append(f"malformed {key}={raw!r}")
                continue
            m = _GATE_RATIO_RE.match(raw)
            if not m:
                failures.append(f"malformed {key}={raw!r}")
                continue
            gp = int(m.group(1))
            gt = int(m.group(2))
            if gt <= 0:
                failures.append(f"invalid {key}={raw!r}")
            elif gp > gt:
                failures.append(f"invalid gates {gp}/{gt} ({key} passed > total)")
                failing_ratios.append((gp, gt))
            elif gp < 8 or gp < gt:
                failures.append(f"gates {gp}/{gt} < 8/8")
                failing_ratios.append((gp, gt))
            else:
                passing_ratios.append((gp, gt))

    if "passed_8_of_8" in sleeve and sleeve["passed_8_of_8"] is not None:
        found_any = True
        flag = sleeve["passed_8_of_8"]
        if not isinstance(flag, bool):
            failures.append(f"invalid passed_8_of_8={flag!r}")
        elif not flag:
            failures.append("passed_8_of_8=False")

    if not found_any:
        return False, "missing 8/8 gate evidence", None, 8
    if failures:
        if failing_ratios:
            fgp, fgt = failing_ratios[0]
            return False, "; ".join(failures), fgp, fgt
        return False, "; ".join(failures), None, 8

    if passing_ratios:
        gp_ok, gt_ok = passing_ratios[0]
        return True, f"gates {gp_ok}/{gt_ok}", gp_ok, gt_ok
    return True, "passed_8_of_8=True", 8, 8


def _parse_sleeve_gate_status(sleeve: dict[str, Any]) -> tuple[bool, str]:
    """Parse per-sleeve 8/8 gate eligibility from canonical Platform v4 / STD-46 schemas (fail-closed)."""
    ok, detail, _gp, _gt = _inspect_sleeve_gates(sleeve)
    return ok, detail


def _is_positive_int_counter(val: Any) -> bool:
    """Return True iff val is a strictly positive non-bool integer counter."""
    return (not isinstance(val, bool)) and isinstance(val, int) and val > 0


def _canonicalize_sleeve_for_report(sleeve: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize sleeve gate and disqualifying status fields for 1:1 SCHQ `--mode report` parity."""
    sl = dict(sleeve)
    gate_ok, _detail, gp, gt = _inspect_sleeve_gates(sl)
    if gate_ok:
        sl["gates_passed"] = gp if gp is not None else 8
        sl["gates_total"] = gt if gt is not None else 8
    else:
        if gp is not None and gt is not None and (gp < 8 or gp < gt):
            sl["gates_passed"] = gp
            sl["gates_total"] = gt
        else:
            cur_v = str(sl.get("verdict") or "").strip().upper()
            if cur_v not in _VER04_BAD_SLEEVE_TOKENS:
                sl["verdict"] = "FAIL"

    disq_token: str | None = None
    for status_key in ("verdict", "status", "verdict_cap", "classification"):
        raw_st = sl.get(status_key)
        if isinstance(raw_st, str):
            norm_st = raw_st.strip().upper()
            if norm_st in _DISQUALIFYING_SLEEVE_STATUSES:
                disq_token = norm_st if norm_st in _VER04_BAD_SLEEVE_TOKENS else "FAIL"
                break
    if disq_token is None and sl.get("dominated_by_static") is True:
        disq_token = VERDICT_CAP_DOMINATED
    elif disq_token is None and sl.get("eligible_for_production") is False:
        disq_token = "FAIL"

    if disq_token is not None:
        cur_v = str(sl.get("verdict") or "").strip().upper()
        cur_s = str(sl.get("status") or "").strip().upper()
        if cur_v not in _VER04_BAD_SLEEVE_TOKENS:
            sl["verdict"] = disq_token
        if cur_s not in _VER04_BAD_SLEEVE_TOKENS:
            sl["status"] = disq_token
    return sl


def compute_dual_sharpe(
    cagr_pct: float,
    ann_vol_pct: float,
    rf_pct: float = DEFAULT_USD_RF_PCT,
    raw_sharpe: float | None = None,
) -> dict[str, float]:
    """Compute STD-46 VER-02 Dual Sharpe (raw Rf=0% and risk-free excess Sharpe).

    When `raw_sharpe` is supplied (e.g., exact return-series arithmetic Sharpe from backtest),
    `excess_sharpe = raw_sharpe - rf_pct / ann_vol_pct` (exact risk-free shift invariance).
    When `raw_sharpe` is None, `raw_sharpe = cagr_pct / ann_vol_pct` and
    `excess_sharpe = (cagr_pct - rf_pct) / ann_vol_pct`.
    """
    cagr = _require_finite_float(cagr_pct, "cagr_pct")
    vol = _require_finite_float(ann_vol_pct, "ann_vol_pct")
    rf = _require_finite_float(rf_pct, "rf_pct")
    if vol <= 0.0:
        raise ValueError(f"ann_vol_pct must be strictly positive, got {vol!r}")
    if rf < 0.0:
        raise ValueError(f"rf_pct must be non-negative, got {rf!r}")

    excess_cagr = cagr - rf
    if raw_sharpe is None:
        raw_sr = cagr / vol
        excess_sr = excess_cagr / vol
    else:
        raw_sr = _require_finite_float(raw_sharpe, "raw_sharpe")
        excess_sr = raw_sr - (rf / vol)

    if not (math.isfinite(raw_sr) and math.isfinite(excess_sr) and math.isfinite(excess_cagr)):
        raise ValueError("Computed dual Sharpe metrics overflowed to non-finite values")

    return {
        "raw_sharpe": float(raw_sr),
        "excess_sharpe": float(excess_sr),
        "rf_pct": float(rf),
        "excess_cagr_pct": float(excess_cagr),
        "cagr_pct": float(cagr),
        "ann_vol_pct": float(vol),
    }


def evaluate_static_ablation(
    strategy_metrics: dict[str, Any],
    static_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate STD-46 VER-03 static-baseline ablation dominance across CAGR, Sharpe, MaxDD, Calmar.

    Advantage sign convention (positive = strategy better than static baseline):
      - delta_cagr   = strategy_cagr - static_cagr
      - delta_sharpe = strategy_sharpe - static_sharpe
      - delta_max_dd = abs(static_max_dd) - abs(strategy_max_dd)   (sign-invariant!)
      - delta_calmar = strategy_calmar - static_calmar

    If all 4 deltas are <= 0.0, the strategy is dominated by the zero-signal static baseline
    and its verdict is capped at `STATIC_ALLOCATION_DOMINATED`.
    """
    if not isinstance(strategy_metrics, dict):
        raise ValueError(f"strategy_metrics must be a dict, got {type(strategy_metrics).__name__}")
    if not isinstance(static_metrics, dict):
        raise ValueError(f"static_metrics must be a dict, got {type(static_metrics).__name__}")

    s_cagr, b_cagr = _extract_pair_metric(strategy_metrics, static_metrics, _CAGR_KEYS, "CAGR")
    s_sr, b_sr = _extract_pair_metric(strategy_metrics, static_metrics, _SR_KEYS, "Sharpe")
    s_dd, b_dd = _extract_pair_metric(strategy_metrics, static_metrics, _MAXDD_KEYS, "MaxDD")
    s_cal, b_cal = _extract_pair_metric(strategy_metrics, static_metrics, _CALMAR_KEYS, "Calmar")

    delta_cagr = s_cagr - b_cagr
    delta_sr = s_sr - b_sr
    delta_max_dd = abs(b_dd) - abs(s_dd)
    delta_calmar = s_cal - b_cal
    if not all(math.isfinite(x) for x in (delta_cagr, delta_sr, delta_max_dd, delta_calmar)):
        raise ValueError("Computed static ablation deltas overflowed to non-finite values")

    dominated = (
        delta_cagr <= 0.0
        and delta_sr <= 0.0
        and delta_max_dd <= 0.0
        and delta_calmar <= 0.0
    )
    verdict_cap = VERDICT_CAP_DOMINATED if dominated else VERDICT_CAP_PASS

    deltas = {
        "cagr": float(delta_cagr),
        "sharpe": float(delta_sr),
        "max_dd": float(delta_max_dd),
        "calmar": float(delta_calmar),
        "cagr_pct": float(delta_cagr),
        "max_dd_pct": float(delta_max_dd),
        "maxdd_pct": float(delta_max_dd),
    }
    return {
        "dominated_by_static": bool(dominated),
        "verdict_cap": verdict_cap,
        "deltas": deltas,
        "delta_vs_static": dict(deltas),
        "strategy_metrics": {
            "cagr_pct": float(s_cagr),
            "sharpe": float(s_sr),
            "max_dd_pct": float(s_dd),
            "maxdd_pct": float(s_dd),
            "calmar": float(s_cal),
        },
        "static_metrics": {
            "cagr_pct": float(b_cagr),
            "sharpe": float(b_sr),
            "max_dd_pct": float(b_dd),
            "maxdd_pct": float(b_dd),
            "calmar": float(b_cal),
        },
    }


def audit_composite_portfolio(
    sleeves: Sequence[dict[str, Any]],
    currencies: set[str] | frozenset[str] | Sequence[str],
    fx_hedged: bool,
) -> dict[str, Any]:
    """Audit STD-46 VER-04 composite portfolio production eligibility & cross-currency FX risk.

    A composite portfolio is `PRODUCTION_ELIGIBLE` if and only if:
      1. `sleeves` is non-empty and 100% of member sleeves pass 8/8 gates with no disqualifying verdict;
      2. Either all member & declared currencies resolve to <= 1 currency (with CNY normalized to RMB),
         or `fx_hedged` is explicitly `True`;
      3. Optional per-sleeve weights/capital allocations are finite, non-negative, and bounded (sum <= 1.0).
    Otherwise restricted to `DIAGNOSTIC_HYPOTHETICAL_ONLY`.
    """
    if not isinstance(fx_hedged, bool):
        raise ValueError(f"fx_hedged must be a bool, got {type(fx_hedged).__name__}")
    if isinstance(sleeves, (str, bytes, dict)):
        raise ValueError("sleeves must be a sequence of sleeve dicts")
    try:
        sleeve_list = list(sleeves)
    except TypeError as exc:
        raise ValueError("sleeves must be a sequence of sleeve dicts") from exc

    norm_currencies = _normalize_currency_tokens(currencies)
    failing_sleeves: list[str] = []
    failing_sleeve_details: dict[str, str] = {}
    reasons: list[str] = []
    capital_errors: list[str] = []

    if not sleeve_list:
        reasons.append("Empty sleeves list: at least 1 eligible 8/8 sleeve is required")

    weight_sum = 0.0
    has_weights = False

    for idx, sl in enumerate(sleeve_list):
        if not isinstance(sl, dict):
            raise ValueError(f"sleeves[{idx}] must be a dict, got {type(sl).__name__}")
        sid = str(
            sl.get("sleeve_id")
            or sl.get("strategy_id")
            or sl.get("id")
            or sl.get("name")
            or f"sleeve[{idx}]"
        )
        if "currency" in sl and sl["currency"] is not None:
            norm_currencies.update(_normalize_currency_tokens(sl["currency"]))
        if "currencies" in sl and sl["currencies"] is not None:
            norm_currencies.update(_normalize_currency_tokens(sl["currencies"]))

        sl_failures: list[str] = []
        gate_ok, gate_detail = _parse_sleeve_gate_status(sl)
        if not gate_ok:
            sl_failures.append(gate_detail)

        for status_key in ("status", "verdict", "verdict_cap", "classification"):
            raw_st = sl.get(status_key)
            if isinstance(raw_st, str) and raw_st.strip().upper() in _DISQUALIFYING_SLEEVE_STATUSES:
                sl_failures.append(f"{status_key}={raw_st.strip().upper()}")

        if "eligible_for_production" in sl and sl["eligible_for_production"] is not None:
            if not isinstance(sl["eligible_for_production"], bool):
                sl_failures.append(f"invalid eligible_for_production={sl['eligible_for_production']!r}")
            elif sl["eligible_for_production"] is False:
                sl_failures.append("eligible_for_production=False")

        if "dominated_by_static" in sl and sl["dominated_by_static"] is not None:
            if not isinstance(sl["dominated_by_static"], bool):
                sl_failures.append(f"invalid dominated_by_static={sl['dominated_by_static']!r}")
            elif sl["dominated_by_static"] is True:
                sl_failures.append("dominated_by_static=True")

        sleeve_weights: list[float] = []
        for w_key in ("weight", "weight_final"):
            if w_key in sl and sl[w_key] is not None:
                has_weights = True
                try:
                    w_val = _require_finite_float(sl[w_key], f"{sid}.{w_key}")
                except ValueError as exc:
                    capital_errors.append(str(exc))
                    sl_failures.append(f"invalid {w_key}")
                    continue
                if w_val < 0.0:
                    msg = f"negative {w_key} on {sid}: {w_val}"
                    capital_errors.append(msg)
                    sl_failures.append(msg)
                else:
                    sleeve_weights.append(w_val)
        if sleeve_weights:
            weight_sum += max(sleeve_weights)

        if "allocated_capital" in sl and sl["allocated_capital"] is not None:
            try:
                cap_val = _require_finite_float(sl["allocated_capital"], f"{sid}.allocated_capital")
                if cap_val < 0.0:
                    msg = f"negative allocated_capital on {sid}: {cap_val}"
                    capital_errors.append(msg)
                    sl_failures.append(msg)
            except ValueError as exc:
                capital_errors.append(str(exc))
                sl_failures.append("invalid allocated_capital")

        if sl_failures:
            failing_sleeves.append(sid)
            detail_str = "; ".join(sl_failures)
            failing_sleeve_details[sid] = detail_str
            reasons.append(f"Sleeve {sid} ineligible: {detail_str}")

    if has_weights and weight_sum > 1.0 + 1e-6:
        msg = f"Sleeve weights exceed 1.0 conservation cap: sum={weight_sum:.6f}"
        capital_errors.append(msg)
        reasons.append(msg)

    sorted_currencies = sorted(norm_currencies)
    is_multi_currency = len(sorted_currencies) > 1
    fx_warning: str | None = None
    if is_multi_currency and not fx_hedged:
        fx_warning = (
            f"Unhedged multi-currency composite ({', '.join(sorted_currencies)}) "
            "mixes currencies without FX hedging costs; restricted to DIAGNOSTIC_HYPOTHETICAL_ONLY."
        )
        reasons.append(fx_warning)

    eligible_for_production = (
        bool(sleeve_list)
        and len(failing_sleeves) == 0
        and fx_warning is None
        and len(capital_errors) == 0
    )
    classification = CLASSIFICATION_PRODUCTION if eligible_for_production else CLASSIFICATION_DIAGNOSTIC

    return {
        "eligible_for_production": bool(eligible_for_production),
        "classification": classification,
        "failing_sleeves": failing_sleeves,
        "failing_sleeve_details": failing_sleeve_details,
        "fx_warning": fx_warning,
        "currencies": sorted_currencies,
        "fx_hedged": bool(fx_hedged),
        "reasons": reasons,
    }


def build_std46_audit_report(
    *,
    report_name: str,
    strategies: Sequence[dict[str, Any]] | None = None,
    composites: Sequence[dict[str, Any]] | None = None,
    shadows: Sequence[dict[str, Any]] | None = None,
    evidence_status: str = "SELF_REPORTED",
    unverified_items: Sequence[str] = ("Independent cross-review pending (no review_id yet)",),
    review_id: str | None = None,
) -> dict[str, Any]:
    """Assemble a JSON-serializable (`allow_nan=False`) STD-46 audit report compliant with SCHQ `--mode report`."""
    if not isinstance(report_name, str) or not report_name.strip():
        raise ValueError("report_name must be a non-empty string")

    out_strategies: list[dict[str, Any]] = []
    for raw_s in strategies or ():
        if not isinstance(raw_s, dict):
            raise ValueError("Each strategy entry must be a dict")
        item = dict(raw_s)
        if "excess_sharpe" not in item or "raw_sharpe" not in item:
            if "cagr_pct" in item and "ann_vol_pct" in item:
                cur_tokens = _normalize_currency_tokens(item.get("currency") or item.get("currencies"))
                default_rf = DEFAULT_RMB_RF_PCT if cur_tokens == {"RMB"} else DEFAULT_USD_RF_PCT
                rf_val = item.get("rf_pct", default_rf)
                dual = compute_dual_sharpe(
                    cagr_pct=item["cagr_pct"],
                    ann_vol_pct=item["ann_vol_pct"],
                    rf_pct=rf_val,
                    raw_sharpe=item.get("raw_sharpe"),
                )
                item.update(dual)
        if "static_ablation" not in item and isinstance(item.get("static_metrics"), dict):
            static_m = item["static_metrics"]
            strat_for_ablation = dict(item)
            if "excess_sharpe" not in static_m:
                for bare_sr_key in ("sharpe", "sharpe_ratio"):
                    if (
                        bare_sr_key in static_m
                        and static_m[bare_sr_key] is not None
                        and bare_sr_key not in strat_for_ablation
                    ):
                        raw_sr_candidate = raw_s.get(bare_sr_key)
                        if raw_sr_candidate is None:
                            raw_sr_candidate = raw_s.get("raw_sharpe")
                        if raw_sr_candidate is None:
                            raw_sr_candidate = item.get("raw_sharpe")
                        if raw_sr_candidate is not None:
                            strat_for_ablation[bare_sr_key] = raw_sr_candidate
            item["static_ablation"] = evaluate_static_ablation(strat_for_ablation, static_m)
        ab = item.get("static_ablation")
        if isinstance(ab, dict) and ab.get("dominated_by_static") is True:
            item["status"] = VERDICT_CAP_DOMINATED
            item["verdict"] = VERDICT_CAP_DOMINATED
            item["classification"] = VERDICT_CAP_DOMINATED
            item["eligible_for_production"] = False
        out_strategies.append(item)

    out_composites: list[dict[str, Any]] = []
    for raw_c in composites or ():
        if not isinstance(raw_c, dict):
            raise ValueError("Each composite entry must be a dict")
        item = dict(raw_c)
        if "excess_sharpe" not in item or "raw_sharpe" not in item:
            if "cagr_pct" in item and "ann_vol_pct" in item:
                cur_tokens = _normalize_currency_tokens(item.get("currencies") or item.get("currency"))
                default_rf = DEFAULT_RMB_RF_PCT if cur_tokens == {"RMB"} else DEFAULT_USD_RF_PCT
                rf_val = item.get("rf_pct", default_rf)
                dual = compute_dual_sharpe(
                    cagr_pct=item["cagr_pct"],
                    ann_vol_pct=item["ann_vol_pct"],
                    rf_pct=rf_val,
                    raw_sharpe=item.get("raw_sharpe"),
                )
                item.update(dual)
        if "sleeves" in item:
            currencies = item.get("currencies") or item.get("currency") or ()
            raw_fx = item.get("fx_hedged")
            fx_hedged = False if raw_fx is None else raw_fx
            audit = audit_composite_portfolio(item["sleeves"], currencies, fx_hedged=fx_hedged)
            item["sleeves"] = [_canonicalize_sleeve_for_report(sl) for sl in item["sleeves"]]
            item["eligible_for_production"] = audit["eligible_for_production"]
            item["classification"] = audit["classification"]
            item["failing_sleeves"] = audit["failing_sleeves"]
            item["fx_warning"] = audit["fx_warning"]
            item["currencies"] = audit["currencies"]
            item["fx_hedged"] = audit["fx_hedged"]
            if not audit["eligible_for_production"]:
                item["status"] = CLASSIFICATION_DIAGNOSTIC
                item["verdict"] = CLASSIFICATION_DIAGNOSTIC
            elif "status" not in item:
                item["status"] = CLASSIFICATION_PRODUCTION
        out_composites.append(item)

    out_shadows: list[dict[str, Any]] = []
    for raw_sh in shadows or ():
        if not isinstance(raw_sh, dict):
            raise ValueError("Each shadow entry must be a dict")
        item = dict(raw_sh)
        present_plan_vals = [item[k] for k in _SHADOW_PLAN_KEYS if k in item and item[k] is not None]
        present_trade_vals = [item[k] for k in _SHADOW_TRADE_KEYS if k in item and item[k] is not None]
        plans_present = bool(present_plan_vals)
        trades_present = bool(present_trade_vals)
        plans_all_pos = plans_present and all(_is_positive_int_counter(v) for v in present_plan_vals)
        trades_all_pos = trades_present and all(_is_positive_int_counter(v) for v in present_trade_vals)

        is_vacuous = (
            (not plans_present and not trades_present)
            or (plans_present and not plans_all_pos)
            or (trades_present and not trades_all_pos)
        )
        if is_vacuous:
            vac_label = (
                "SHADOW_VACUOUS_ZERO_TRADES"
                if (trades_present and not trades_all_pos)
                else "VACUOUS_ZERO_PLANS"
            )
            item["non_vacuous_parity"] = False
            item["parity_depth"] = vac_label
            item["shadow_status"] = vac_label
            for k in ("status", "verdict", "parity_status", "readiness_status"):
                if k in item and str(item[k]).strip().upper() in _VACUOUS_FORBIDDEN_CLAIMS:
                    item[k] = vac_label
        out_shadows.append(item)

    report: dict[str, Any] = {
        "report_name": report_name.strip(),
        "standard": "SCH-STD-46",
        "evidence_status": str(evidence_status or "SELF_REPORTED"),
        "strategies": out_strategies,
        "composites": out_composites,
        "shadows": out_shadows,
    }
    if review_id is not None and str(review_id).strip():
        report["review_id"] = str(review_id).strip()
    else:
        items = [str(x).strip() for x in unverified_items if str(x).strip()]
        report["unverified_items"] = items or ["Independent cross-review pending (no review_id yet)"]
    return report


__all__ = [
    "CLASSIFICATION_DIAGNOSTIC",
    "CLASSIFICATION_PRODUCTION",
    "DEFAULT_RMB_RF_PCT",
    "DEFAULT_USD_RF_PCT",
    "VERDICT_CAP_DOMINATED",
    "VERDICT_CAP_PASS",
    "audit_composite_portfolio",
    "build_std46_audit_report",
    "compute_dual_sharpe",
    "evaluate_static_ablation",
]
