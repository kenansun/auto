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
    """Thread-safe structured single-turn JSON router with per-thread round budget isolation."""

    def __init__(
        self,
        routes: dict[str, ModelRouteSpec] | None = None,
        max_cycle_budget_usd: float = 0.50,
    ) -> None:
        self.routes = dict(routes or DEFAULT_ROUTES)
        self.max_cycle_budget_usd = max_cycle_budget_usd
        self._lock = threading.Lock()
        self._tls = threading.local()
        self.spent_usd = 0.0
        self.call_log: list[dict[str, Any]] = []

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
        """Call external LLM if API key is configured & within budget; otherwise return (None, fallback_tag)."""
        spec = override_spec or self.routes[role]
        gateway_url = os.environ.get("ALPHALOOP_AI_GATEWAY_URL", "").strip()
        api_key = os.environ.get("ALPHALOOP_AI_GATEWAY_KEY", "").strip() or os.environ.get(spec.api_key_env, "").strip()

        if not api_key:
            return None, f"deterministic-fallback(target={spec.family}/{spec.model_id},no_api_key)"

        if self.round_spent_usd >= self.max_cycle_budget_usd:
            logger.warning("Round LLM budget cap $%.2f reached; falling back locally", self.max_cycle_budget_usd)
            return None, f"deterministic-fallback(budget_cap_reached=${self.round_spent_usd:.2f})"

        compact_user = json.dumps(user_payload, separators=(",", ":"), ensure_ascii=True)
        if len(compact_user) > 24000:
            compact_user = compact_user[:24000] + "...[TRUNCATED_FOR_COST_GUARD]"

        try:
            if spec.family == "claude" and not gateway_url:
                resp_text, in_tok, out_tok = self._call_anthropic(spec, api_key, system_prompt, compact_user)
            else:
                url = gateway_url or spec.provider_url
                resp_text, in_tok, out_tok = self._call_openai_compat(url, spec.model_id, api_key, system_prompt, compact_user)

            est_cost = (in_tok * spec.input_usd_per_1m + out_tok * spec.output_usd_per_1m) / 1_000_000.0
            self.round_spent_usd = self.round_spent_usd + est_cost
            with self._lock:
                self.spent_usd += est_cost
                self.call_log.append({
                    "role": role,
                    "model": f"{spec.family}/{spec.model_id}",
                    "in_tokens": in_tok,
                    "out_tokens": out_tok,
                    "est_usd": round(est_cost, 6),
                })
            parsed = self._extract_json(resp_text)
            return parsed, f"{spec.family}/{spec.model_id}"
        except (urllib.error.URLError, urllib.error.HTTPError, http.client.HTTPException, OSError, TimeoutError, ValueError, KeyError) as exc:
            logger.warning("LLM call failed for %s (%s): %s — falling back deterministically", role, spec.model_id, exc)
            return None, f"deterministic-fallback(api_error={type(exc).__name__})"

    def _call_openai_compat(
        self, url: str, model_id: str, api_key: str, system_prompt: str, user_msg: str
    ) -> tuple[str, int, int]:
        body = json.dumps({
            "model": model_id,
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
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

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
            data = json.loads(resp.read().decode("utf-8"))
        text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        usage = data.get("usage", {})
        return "\n".join(text_blocks), int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

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
