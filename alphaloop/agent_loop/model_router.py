"""Consequence-Severity Multi-Model Router with budget cap & cross-model enforcement.

Implements consequence-tiered multi-model routing as executable code:
- Routes by role & consequence severity (NOT one-size-fits-all model).
- Enforces cross-family Red-Team independence (Critic family != Proposer family).
- Enforces hard USD budget cap per cycle to prevent runaway token loops.
- Uses ONLY env-injected keys; falls back transparently to deterministic local logic when offline.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRouteSpec:
    role: str
    family: str            # "deepseek" | "gemini" | "claude" | "openai"
    model_id: str
    provider_url: str
    api_key_env: str
    input_usd_per_1m: float
    output_usd_per_1m: float
    rationale: str


DEFAULT_ROUTES: dict[str, ModelRouteSpec] = {
    "SCANNER_T0": ModelRouteSpec(
        role="SCANNER_T0",
        family="gemini",
        model_id="gemini-3.6-flash",
        provider_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        api_key_env="GEMINI_API_KEY",
        input_usd_per_1m=0.15,
        output_usd_per_1m=0.60,
        rationale="High-volume telemetry compression, log triage & daily patrol at ultra-low unit cost.",
    ),
    "QUANT_HYPOTHESIS": ModelRouteSpec(
        role="QUANT_HYPOTHESIS",
        family="deepseek",
        model_id="deepseek-reasoner",
        provider_url="https://api.deepseek.com/chat/completions",
        api_key_env="DEEPSEEK_API_KEY",
        input_usd_per_1m=0.55,
        output_usd_per_1m=2.19,
        rationale="Mathematical & statistical reasoning for factor/parameter hypothesis generation at ~1/10th frontier cost.",
    ),
    "CODE_BUILDER": ModelRouteSpec(
        role="CODE_BUILDER",
        family="claude",
        model_id="claude-sonnet-5.5",
        provider_url="https://api.anthropic.com/v1/messages",
        api_key_env="ANTHROPIC_API_KEY",
        input_usd_per_1m=2.00,
        output_usd_per_1m=10.00,
        rationale="Balanced long-context multi-file strategy engineering and unit test generation.",
    ),
    "RED_TEAM_CRITIC": ModelRouteSpec(
        role="RED_TEAM_CRITIC",
        family="claude",
        model_id="claude-opus-5.5",
        provider_url="https://api.anthropic.com/v1/messages",
        api_key_env="ANTHROPIC_API_KEY",
        input_usd_per_1m=4.00,
        output_usd_per_1m=20.00,
        rationale="Adversarial irreversible-consequence & statistical audit; strictly distinct family from DeepSeek/GPT proposer.",
    ),
    "RED_TEAM_ALT": ModelRouteSpec(
        role="RED_TEAM_ALT",
        family="openai",
        model_id="gpt-6.1-sol",
        provider_url="https://api.openai.com/v1/chat/completions",
        api_key_env="OPENAI_API_KEY",
        input_usd_per_1m=3.00,
        output_usd_per_1m=15.00,
        rationale="Cross-family escalation & hard-bug arbiter when Proposer is Claude.",
    ),
}


class MultiModelRouter:
    """Thread-safe structured single-turn JSON router with per-round & persistent daily USD caps."""

    def __init__(
        self,
        routes: dict[str, ModelRouteSpec] | None = None,
        max_cycle_budget_usd: float = 0.50,
        max_daily_budget_usd: float = 3.00,
        budget_db_path: str | None = None,
    ) -> None:
        self.routes = dict(routes or DEFAULT_ROUTES)
        self.max_cycle_budget_usd = max(0.0, float(max_cycle_budget_usd))
        self.max_daily_budget_usd = max(0.0, float(max_daily_budget_usd))
        self._lock = threading.Lock()
        self._tls = threading.local()
        self.spent_usd = 0.0
        self.call_log: list[dict[str, Any]] = []
        self._budget_conn = None
        if budget_db_path:
            from pathlib import Path
            from alphaloop.core.db import connect_wal, transaction

            self._budget_conn = connect_wal(Path(budget_db_path))
            with transaction(self._budget_conn):
                self._budget_conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS llm_daily_spend (
                        utc_day TEXT PRIMARY KEY,
                        spent_usd REAL NOT NULL,
                        call_count INTEGER NOT NULL,
                        updated_ts_ms INTEGER NOT NULL
                    );
                    """
                )

    def _utc_day(self) -> str:
        import datetime
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    def daily_spent_usd(self, utc_day: str | None = None) -> float:
        day = utc_day or self._utc_day()
        if self._budget_conn is None:
            with self._lock:
                return float(self.spent_usd)
        row = self._budget_conn.fetch_one("SELECT spent_usd FROM llm_daily_spend WHERE utc_day=?", (day,))
        return float(row["spent_usd"]) if row else 0.0

    def _reserve_daily_budget(self, reserve_usd: float) -> tuple[bool, float, str]:
        """Atomically reserve worst-case call spend in SQLite (BEGIN IMMEDIATE) before calling LLM."""
        reserve = max(0.0, float(reserve_usd))
        day = self._utc_day()
        if self._budget_conn is None:
            with self._lock:
                if self.spent_usd >= self.max_daily_budget_usd or (self.spent_usd + reserve) > self.max_daily_budget_usd + 1e-12:
                    return False, float(self.spent_usd), day
                self.spent_usd += reserve
                return True, float(self.spent_usd), day

        from alphaloop.core.db import transaction
        from alphaloop.core.provenance import utc_now_ms

        now_ms = utc_now_ms()
        with self._lock, transaction(self._budget_conn):
            row = self._budget_conn.fetch_one(
                "SELECT spent_usd FROM llm_daily_spend WHERE utc_day=?", (day,)
            )
            cur = float(row["spent_usd"]) if row else 0.0
            if cur >= self.max_daily_budget_usd or (cur + reserve) > self.max_daily_budget_usd + 1e-12:
                return False, cur, day
            self._budget_conn.execute(
                """
                INSERT INTO llm_daily_spend (utc_day, spent_usd, call_count, updated_ts_ms)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(utc_day) DO UPDATE SET
                    spent_usd = spent_usd + excluded.spent_usd,
                    call_count = call_count + 1,
                    updated_ts_ms = excluded.updated_ts_ms
                """,
                (day, reserve, now_ms),
            )
            return True, cur + reserve, day

    def _reconcile_daily_budget(self, delta_usd: float, *, utc_day: str | None = None) -> None:
        """Adjust reserved daily spend to exact post-call cost on the exact reserved UTC day."""
        delta = float(delta_usd)
        if abs(delta) < 1e-12:
            return
        if self._budget_conn is None:
            with self._lock:
                self.spent_usd = max(0.0, self.spent_usd + delta)
            return
        from alphaloop.core.db import transaction
        from alphaloop.core.provenance import utc_now_ms

        day = utc_day or self._utc_day()
        now_ms = utc_now_ms()
        with self._lock, transaction(self._budget_conn):
            self._budget_conn.execute(
                """
                UPDATE llm_daily_spend
                SET spent_usd = MAX(0.0, spent_usd + ?), updated_ts_ms = ?
                WHERE utc_day = ?
                """,
                (delta, now_ms, day),
            )

    @staticmethod
    def _validate_http_url(url: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Invalid LLM endpoint URL scheme/host: {url!r}")
        return url

    def close(self) -> None:
        if self._budget_conn is not None:
            self._budget_conn.close()
            self._budget_conn = None

    @property
    def round_spent_usd(self) -> float:
        return float(getattr(self._tls, "round_spent_usd", 0.0))

    @round_spent_usd.setter
    def round_spent_usd(self, value: float) -> None:
        self._tls.round_spent_usd = float(value)

    def reset_round_budget(self) -> None:
        """Reset current thread's per-round budget counter without disturbing concurrent threads."""
        self._tls.round_spent_usd = 0.0

    def select_red_team_route(self, proposer_family: str) -> ModelRouteSpec:
        """Guarantee adversarial Red-Team critic comes from a different model family."""
        primary = self.routes["RED_TEAM_CRITIC"]
        if primary.family != proposer_family:
            return primary
        return self.routes["RED_TEAM_ALT"]

    def invoke_json(
        self,
        role: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        override_spec: ModelRouteSpec | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """Call external LLM if API key is configured & within round/daily budgets; else fallback locally."""
        spec = override_spec or self.routes[role]
        gateway_url = os.environ.get("ALPHALOOP_AI_GATEWAY_URL", "").strip()
        api_key = os.environ.get("ALPHALOOP_AI_GATEWAY_KEY", "").strip() or os.environ.get(spec.api_key_env, "").strip()

        if not api_key:
            return None, f"deterministic-fallback(target={spec.family}/{spec.model_id},no_api_key)"

        if gateway_url:
            try:
                self._validate_http_url(gateway_url)
            except ValueError as exc:
                logger.warning("Rejected invalid ALPHALOOP_AI_GATEWAY_URL: %s", exc)
                return None, "deterministic-fallback(invalid_gateway_url)"

        compact_user = json.dumps(user_payload, separators=(",", ":"), ensure_ascii=True)
        if len(compact_user) > 24000:
            compact_user = compact_user[:24000] + "...[TRUNCATED_FOR_COST_GUARD]"

        est_in_tok = max(1, (len(system_prompt.encode("utf-8")) + len(compact_user.encode("utf-8"))) // 4)
        reserve_usd = max(
            1e-6,
            (est_in_tok * spec.input_usd_per_1m + 1200 * spec.output_usd_per_1m) / 1_000_000.0,
        )

        if self.round_spent_usd >= self.max_cycle_budget_usd or (self.round_spent_usd + reserve_usd) > self.max_cycle_budget_usd + 1e-12:
            logger.warning("Round LLM budget cap $%.4f reached; falling back locally", self.max_cycle_budget_usd)
            return None, f"deterministic-fallback(budget_cap_reached=${self.round_spent_usd:.4f})"

        ok_daily, day_spent, reserved_day = self._reserve_daily_budget(reserve_usd)
        if not ok_daily:
            logger.warning("Daily LLM budget cap $%.4f reached ($%.4f); falling back locally", self.max_daily_budget_usd, day_spent)
            return None, f"deterministic-fallback(daily_budget_cap_reached=${day_spent:.4f})"

        actual_cost: float | None = None
        try:
            if spec.family == "claude" and not gateway_url:
                self._validate_http_url(spec.provider_url)
                resp_text, in_tok, out_tok = self._call_anthropic(spec, api_key, system_prompt, compact_user)
            else:
                url = self._validate_http_url(gateway_url or spec.provider_url)
                resp_text, in_tok, out_tok = self._call_openai_compat(url, spec.model_id, api_key, system_prompt, compact_user)

            # Conservative byte-length fallback if provider omits or zeroes usage counts
            if in_tok <= 0:
                in_tok = est_in_tok
            if out_tok <= 0:
                out_tok = max(1, len(resp_text.encode("utf-8")) // 4)

            est_cost = (max(0, in_tok) * spec.input_usd_per_1m + max(0, out_tok) * spec.output_usd_per_1m) / 1_000_000.0
            actual_cost = est_cost
            self.round_spent_usd = self.round_spent_usd + est_cost
            with self._lock:
                if self._budget_conn is not None:
                    self.spent_usd += est_cost
                self.call_log.append({
                    "role": role,
                    "model": f"{spec.family}/{spec.model_id}",
                    "in_tokens": in_tok,
                    "out_tokens": out_tok,
                    "est_usd": round(est_cost, 6),
                })
            parsed = self._extract_json(resp_text)
            if not isinstance(parsed, dict):
                raise ValueError("Model JSON root must be an object")
            return parsed, f"{spec.family}/{spec.model_id}"
        except (urllib.error.URLError, urllib.error.HTTPError, http.client.HTTPException, OSError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            logger.warning("LLM call failed for %s (%s): %s — falling back deterministically", role, spec.model_id, exc)
            return None, f"deterministic-fallback(api_error={type(exc).__name__})"
        finally:
            final_cost = actual_cost if actual_cost is not None else 0.0
            self._reconcile_daily_budget(final_cost - reserve_usd, utc_day=reserved_day)

    @staticmethod
    def _read_bounded_json(resp: Any, max_bytes: int = 1_048_576) -> dict[str, Any]:
        raw = resp.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError(f"LLM_RESPONSE_EXCEEDS_{max_bytes}_BYTES")
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("LLM_HTTP_JSON_MUST_BE_OBJECT")
        return data

    def _call_openai_compat(
        self, url: str, model_id: str, api_key: str, system_prompt: str, user_msg: str
    ) -> tuple[str, int, int]:
        body = json.dumps({
            "model": model_id,
            "max_tokens": 1200,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = self._read_bounded_json(resp)
        content = str(data["choices"][0]["message"]["content"])
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return content, int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)

    def _call_anthropic(
        self, spec: ModelRouteSpec, api_key: str, system_prompt: str, user_msg: str
    ) -> tuple[str, int, int]:
        body = json.dumps({
            "model": spec.model_id,
            "max_tokens": 1200,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_msg}],
        }).encode("utf-8")
        req = urllib.request.Request(
            spec.provider_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = self._read_bounded_json(resp)
        text_blocks = [b["text"] for b in data.get("content", []) if isinstance(b, dict) and b.get("type") == "text"]
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return "\n".join(text_blocks), int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            clean = "\n".join(lines).strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        return json.loads(clean[start : end + 1])

    def routing_matrix(self) -> list[dict[str, Any]]:
        return [
            {
                "role": r.role,
                "family": r.family,
                "model_id": r.model_id,
                "api_key_env": r.api_key_env,
                "configured": bool(os.environ.get(r.api_key_env) or os.environ.get("ALPHALOOP_AI_GATEWAY_KEY")),
                "pricing_usd_1m": f"${r.input_usd_per_1m}/${r.output_usd_per_1m}",
                "rationale": r.rationale,
            }
            for r in self.routes.values()
        ]
