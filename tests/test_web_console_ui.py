"""Comprehensive pure-stdlib contract & HTTP tests for AlphaLoop v2.2 Strategy R&D Console UI."""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from html.parser import HTMLParser
from pathlib import Path
from unittest import mock

import alphaloop.web.server as server_mod
from alphaloop.core.models import BBOTick, FundingRateBar, PriceBar, StrategyConfig
from alphaloop.core.provenance import PROVENANCE_REAL, utc_now_ms
from alphaloop.web.server import CONSOLE_HTML, ConsoleAppContext, create_server

REPO_ROOT = Path(__file__).resolve().parents[1]
LLM_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ALPHALOOP_AI_GATEWAY_KEY",
    "ALPHALOOP_AI_GATEWAY_URL",
)


class _ConsoleDOMInspector(HTMLParser):
    """Stdlib HTMLParser collecting landmarks, ARIA tabs, dialogs, forms, selects, scripts, and styles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.ids: set[str] = set()
        self.h1_count: int = 0
        self.dialogs: list[dict[str, str]] = []
        self.forms: list[dict[str, str]] = []
        self.tablists: list[dict[str, str]] = []
        self.tabs: list[dict[str, str]] = []
        self.tabpanels: list[dict[str, str]] = []
        self.selects: dict[str, list[str]] = {}
        self.inline_event_attrs: list[tuple[str, str]] = []
        self.scripts: list[str] = []
        self.styles: list[str] = []
        self._cur_select_id: str | None = None
        self._in_script = False
        self._in_style = False
        self._buf: list[str] = []
        self.toast_region_attrs: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        attr_map: dict[str, str] = {}
        for k, v in attrs:
            kl = k.lower()
            val = "" if v is None else v
            attr_map[kl] = val
            if kl.startswith("on"):
                self.inline_event_attrs.append((t, kl))

        elem_id = attr_map.get("id", "")
        if elem_id:
            self.ids.add(elem_id)
            if elem_id == "ui-toast-stack":
                self.toast_region_attrs = dict(attr_map)

        if t == "h1":
            self.h1_count += 1
        elif t == "dialog":
            self.dialogs.append(attr_map)
        elif t == "form":
            self.forms.append(attr_map)
        elif t == "select":
            self._cur_select_id = elem_id or f"select_{len(self.selects)}"
            self.selects.setdefault(self._cur_select_id, [])
        elif t == "option" and self._cur_select_id is not None:
            self.selects[self._cur_select_id].append(attr_map.get("value", ""))
        elif t == "script":
            self._in_script = True
            self._buf = []
        elif t == "style":
            self._in_style = True
            self._buf = []

        role = attr_map.get("role", "").lower()
        if role == "tablist":
            self.tablists.append(attr_map)
        elif role == "tab":
            self.tabs.append(attr_map)
        elif role == "tabpanel":
            self.tabpanels.append(attr_map)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "select":
            self._cur_select_id = None
        elif t == "script" and self._in_script:
            self.scripts.append("".join(self._buf))
            self._in_script = False
            self._buf = []
        elif t == "style" and self._in_style:
            self.styles.append("".join(self._buf))
            self._in_style = False
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_script or self._in_style:
            self._buf.append(data)


def _split_top_level_commas(expr: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _has_colored_glow_shadow(css_text: str) -> list[str]:
    strip_color = re.compile(r"(?:rgba?|hsla?|var)\([^)]*\)|#[0-9a-fA-F]{3,8}\b", re.I)
    length_tok = re.compile(r"(?<![\w.-])(?:-?(?:\d+\.?\d*|\.\d+)(?:px|rem|em)|0)(?![\w-])", re.I)
    non_black_hex = re.compile(r"#(?!(?:000|0000|000000(?:[0-9a-fA-F]{2})?)\b)[0-9a-fA-F]{3,8}\b")
    non_black_rgb = re.compile(r"rgba?\(\s*(?!0(?:\s*,\s*|\s+)0(?:\s*,\s*|\s+)0(?:\s*[,/)]|\s*$))", re.I)
    hsl_rx = re.compile(r"hsla?\(", re.I)
    colored_var = re.compile(r"var\(--(?:accent|color-|border-focus|market-|up-color|down-color|warning-color)[\w-]*\)", re.I)

    hits: list[str] = []
    for m in re.finditer(r"box-shadow\s*:\s*([^;}\"']+)", css_text, re.I):
        expr = m.group(1)
        for layer in _split_top_level_commas(expr):
            lengths = length_tok.findall(strip_color.sub(" ", layer))
            if len(lengths) < 3:
                continue
            blur_val = float(re.sub(r"[a-zA-Z]+$", "", lengths[2]))
            if blur_val <= 0:
                continue
            if (
                non_black_hex.search(layer)
                or non_black_rgb.search(layer)
                or hsl_rx.search(layer)
                or colored_var.search(layer)
            ):
                hits.append(layer)
    return hits


class TestConsoleUIContract(unittest.TestCase):
    """Static DOM, CSS Anti-Cliché, STD-37, L-034, JS syntax & Python warning contracts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dom = _ConsoleDOMInspector()
        cls.dom.feed(CONSOLE_HTML)
        cls.dom.close()
        cls.style_code = "\n".join(cls.dom.styles)
        cls.script_code = "\n".join(cls.dom.scripts)

    def test_console_html_export_v22_branding_and_preserved_substrings(self) -> None:
        self.assertIsInstance(CONSOLE_HTML, str)
        self.assertGreater(len(CONSOLE_HTML), 2000)
        self.assertNotIn("v2.1", CONSOLE_HTML.lower())
        self.assertNotIn("v2.1", (ConsoleAppContext.__doc__ or "").lower())
        self.assertIn("v2.2", CONSOLE_HTML)
        self.assertIn("ALPHALOOP V2.2", CONSOLE_HTML.upper())
        # Preserved cross-suite substring contracts
        self.assertIn("PortfolioCombiner", CONSOLE_HTML)
        self.assertIn("Shadow Execution Bridge", CONSOLE_HTML)
        self.assertIn("shadowParityBox", CONSOLE_HTML)
        self.assertIn("data-action", CONSOLE_HTML)
        # No accidental raw control-char decode corruption
        self.assertNotIn("\x00", CONSOLE_HTML)

    def test_l034_zero_inline_handlers_zero_native_dialogs_and_xss_escape(self) -> None:
        self.assertEqual(self.dom.inline_event_attrs, [], f"Found inline event attributes: {self.dom.inline_event_attrs}")
        self.assertIsNone(
            re.search(r'(?<!\.)\bon(?:click|change|input|submit|error|load|mouseover|focus)\s*=\s*[\'"]', CONSOLE_HTML, re.I),
            "Inline event handler attribute found in CONSOLE_HTML",
        )
        self.assertIsNone(
            re.search(r"(?<!ui\.)(?<!\.)\b(?:(?:window|globalThis)\.)?(?:alert|confirm|prompt)\s*\(", self.script_code),
            "Native blocking dialog call found in embedded <script>",
        )
        self.assertIsNotNone(re.search(r"\bfunction\s+(?:escapeHtml|esc)\s*\(", self.script_code))
        for entity in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
            self.assertIn(entity, self.script_code)
        self.assertIn("data-action", CONSOLE_HTML)
        self.assertIn("closest(", self.script_code)

    def test_anti_cliche_css_audit_zero_cdn_zero_purple_zero_forbidden_patterns(self) -> None:
        cdn_rx = re.compile(
            r"fonts\.googleapis\.com|fonts\.gstatic\.com|cdnjs\.cloudflare\.com|cdn\.jsdelivr\.net|unpkg\.com",
            re.I,
        )
        self.assertIsNone(cdn_rx.search(CONSOLE_HTML), "External CDN reference found")

        purple_rx = re.compile(
            r"(#(?:7c3aed|8b5cf6|a855f7|6366f1|9333ea|a78bfa|c084fc|818cf8|4f46e5|581c87|a371f7|bc8cff|8957e5|9b8afb|b388ff|d1b8ff|7c4dff|6d28d9|c4b5fd|d8b4fe|9d4edd|7f5af0|9945ff|e879f9|800080)\b"
            r"|rgba?\(\s*(?:124|139|168|99|147|167|192|129|79|163|137|155|179)\s*,\s*(?:58|92|85|102|51|139|132|140|70|113|87|138|136)\s*,"
            r"|--(?:accent-)?(?:purple|violet|indigo)\b"
            r"|:\s*[^;{}]*(?<![-\w])(?:purple|violet|rebeccapurple|blueviolet|mediumpurple)(?![-\w]))",
            re.I,
        )
        self.assertIsNone(purple_rx.search(CONSOLE_HTML), "Purple/violet/indigo token or color found")

        thick_border_rx = re.compile(
            r"border(?:-(?:left|right|top|bottom))?\s*:\s*(?:[3-9]|\d{2,})px\s+solid",
            re.I,
        )
        self.assertIsNone(thick_border_rx.search(self.style_code), "3px+ solid border found in CSS")

        glow_hits = _has_colored_glow_shadow(self.style_code)
        self.assertEqual(glow_hits, [], f"Colored glow box-shadow found: {glow_hits}")

        pulse_rx = re.compile(r"@keyframes\s+[\w-]*pulse|animation\s*:[^;]*pulse[^;]*infinite", re.I)
        self.assertIsNone(pulse_rx.search(self.style_code), "Infinite pulse animation found")

        grad_clip_rx = re.compile(r"background-clip\s*:\s*text|text-fill-color\s*:\s*transparent", re.I)
        self.assertIsNone(grad_clip_rx.search(self.style_code), "Gradient text clip found")

        self.assertNotIn("#64748b", CONSOLE_HTML.lower(), "Retired WCAG-failing slate #64748b found")

        for form_attrs in self.dom.forms:
            self.assertNotEqual(form_attrs.get("method", "").lower(), "dialog", "Forbidden <form method='dialog'> found")
        self.assertIsNone(re.search(r'method\s*=\s*["\']?dialog', CONSOLE_HTML, re.I))

    def test_canonical_design_tokens_responsive_breakpoints_and_reduced_motion(self) -> None:
        required_tokens = (
            r"--bg-base\s*:\s*#0b0f14",
            r"--bg-surface\s*:\s*#11161f",
            r"--bg-elevated\s*:\s*#161d28",
            r"--bg-hover\s*:\s*#1e2736",
            r"--text-strong\s*:\s*#f1f5f9",
            r"--text-primary\s*:\s*#e6edf3",
            r"--text-secondary\s*:\s*#94a3b8",
            r"--text-muted\s*:\s*#7d8fa6",
            r"--accent-primary\s*:\s*#38bdf8",
            r"--color-profit\s*:\s*#22c55e",
            r"--color-loss\s*:\s*#ef4444",
            r"--color-warn\s*:\s*#f59e0b",
            r"--surface-hairline-top\s*:",
        )
        for rx in required_tokens:
            self.assertIsNotNone(re.search(rx, self.style_code, re.I), f"Missing canonical CSS token: {rx}")

        required_rules = (
            r"@media\s*\(\s*max-width\s*:\s*1280px\s*\)",
            r"@media\s*\(\s*max-width\s*:\s*1024px\s*\)",
            r"@media\s*\(\s*max-width\s*:\s*768px\s*\)",
            r"@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)",
            r":focus-visible",
            r"tabular-nums",
        )
        for rx in required_rules:
            self.assertIsNotNone(re.search(rx, self.style_code, re.I), f"Missing required responsive/a11y rule: {rx}")

    def test_wai_aria_five_tabs_panels_and_all_13_data_containers(self) -> None:
        self.assertEqual(self.dom.h1_count, 1, "Page must contain exactly one <h1>")
        self.assertGreaterEqual(len(self.dom.tablists), 1)
        self.assertEqual(len(self.dom.tabs), 5)
        self.assertEqual(len(self.dom.tabpanels), 5)

        panel_ids = {p.get("id", "") for p in self.dom.tabpanels}
        expected_panels = {"panel-overview", "panel-execution", "panel-factory", "panel-evolution", "panel-fleet"}
        self.assertEqual(panel_ids, expected_panels)

        selected_tabs = [t for t in self.dom.tabs if t.get("aria-selected") == "true"]
        self.assertEqual(len(selected_tabs), 1)
        for tab in self.dom.tabs:
            self.assertEqual(tab.get("data-action"), "tab")
            self.assertIn(tab.get("aria-controls", ""), panel_ids)

        expected_data_ids = {
            "kpiGrid",
            "shadowParityBox",
            "portfolioBox",
            "marketSummaryBox",
            "execBox",
            "strategiesBox",
            "pluginsBox",
            "dqBox",
            "microBox",
            "cyclesBox",
            "modelsBox",
            "trialsBox",
            "fleetBox",
        }
        self.assertTrue(expected_data_ids <= self.dom.ids, f"Missing data IDs: {expected_data_ids - self.dom.ids}")

        required_chrome_ids = {"healthBadge", "refreshBtn", "actionDialog", "ui-toast-stack"}
        self.assertTrue(required_chrome_ids <= self.dom.ids, f"Missing chrome IDs: {required_chrome_ids - self.dom.ids}")
        self.assertEqual(self.dom.toast_region_attrs.get("aria-live"), "polite")

        action_dialogs = [d for d in self.dom.dialogs if d.get("id") == "actionDialog"]
        self.assertEqual(len(action_dialogs), 1)
        self.assertEqual(action_dialogs[0].get("aria-modal"), "true")

    def test_auto_refresh_controls_std37_states_and_action_wiring(self) -> None:
        self.assertIn("autoRefreshSelect", self.dom.selects)
        opts = set(self.dom.selects["autoRefreshSelect"])
        self.assertTrue({"0", "10000", "30000", "60000"} <= opts, f"Unexpected auto-refresh options: {opts}")
        self.assertIn("document.hidden", self.script_code)
        self.assertIn("visibilitychange", self.script_code)

        # STD-37 state vocabulary
        self.assertIn('data-state="sim"', CONSOLE_HTML)
        self.assertIn("no-channel", CONSOLE_HTML)
        self.assertIn("upstream-error", CONSOLE_HTML)
        self.assertIn("Promise.allSettled", self.script_code)

        # All 4 strategy actions & endpoints
        for token in (
            "/api/agent/run-cycle",
            "/api/agent/run-sprint",
            "/api/micro-runs/evaluate",
            "/api/strategies/rollback",
            "/api/strategies/versions",
        ):
            self.assertIn(token, self.script_code)

        # HUMAN_APPROVED regex helper & LLM budget caps
        self.assertIn("HUMAN_APPROVED:[A-Za-z0-9._-]{4,64}", self.script_code)
        self.assertIn("round_spent_usd", self.script_code)
        self.assertIn("spent_usd", self.script_code)
        self.assertIn("0.50", self.script_code)
        self.assertIn("3.00", self.script_code)

    def test_embedded_javascript_syntax_valid_via_node(self) -> None:
        self.assertGreaterEqual(len(self.dom.scripts), 1)
        node_bin = shutil.which("node")
        if not node_bin:
            self.skipTest("node not installed")
        for idx, script in enumerate(self.dom.scripts):
            proc = subprocess.run(
                [node_bin, "--check", "-"],
                input=script,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"Script #{idx} JS syntax error:\n{proc.stderr}")
            self.assertEqual(proc.stderr.strip(), "")

    def test_server_module_zero_python_syntax_warnings(self) -> None:
        server_path = Path(server_mod.__file__).resolve()
        source = server_path.read_text(encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error", SyntaxWarning)
            compile(source, str(server_path), "exec")
        self.assertEqual(caught, [])

        proc = subprocess.run(
            [sys.executable, "-B", "-W", "error::SyntaxWarning", "-c", "import alphaloop.web.server"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"Subprocess import emitted SyntaxWarning:\n{proc.stderr}")
        self.assertEqual(proc.stderr.strip(), "")


class TestConsoleV22Endpoints(unittest.TestCase):
    """Live ephemeral loopback HTTP tests for AlphaLoop v2.2 console endpoints."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.srv, self.ctx = create_server(self.root, host="127.0.0.1", port=0)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()
        self.ctx.close()
        self.tmp.cleanup()

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            raw = resp.read()
            hdrs = {k.lower(): v for k, v in resp.getheaders()}
            return resp.status, hdrs, raw
        finally:
            conn.close()

    def _post_json(
        self,
        path: str,
        payload: dict[str, object],
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], dict[str, object]]:
        hdrs = {"Content-Type": "application/json"}
        if extra_headers:
            hdrs.update(extra_headers)
        status, resp_hdrs, raw = self._request("POST", path, body=json.dumps(payload).encode("utf-8"), headers=hdrs)
        return status, resp_hdrs, json.loads(raw.decode("utf-8"))

    def _assert_security_headers(self, hdrs: dict[str, str], raw: bytes) -> None:
        self.assertEqual(int(hdrs["content-length"]), len(raw))
        self.assertEqual(hdrs.get("cache-control"), "no-cache")
        self.assertEqual(hdrs.get("x-content-type-options"), "nosniff")

    def _make_cfg(self, sid: str, version: str, status: str) -> StrategyConfig:
        return StrategyConfig(
            strategy_id=sid,
            version=version,
            family="test_family",
            status=status,
            venue_a="hyperliquid",
            venue_b="gate",
            mode="perp_perp",
            symbols=["BTC"],
            entry_spread_apr_pct=18.0,
            exit_spread_apr_pct=4.0,
        )

    def test_get_strategies_versions_valid_missing_unknown_headers_and_host_guard(self) -> None:
        self.ctx.config_store.register_version(self._make_cfg("ver_strat", "v1.0", "SHADOW_LIVE"), actor="unit", reason="init")
        self.ctx.config_store.register_version(
            self._make_cfg("ver_strat", "v2.0", "CANARY_APPROVED"),
            actor="owner",
            reason="canary promote",
            human_approval_token="HUMAN_APPROVED:TEST-001",
        )

        # Valid query -> 200 newest-first
        st, hdrs, raw = self._request("GET", "/api/strategies/versions?strategy_id=ver_strat")
        self.assertEqual(st, 200)
        self._assert_security_headers(hdrs, raw)
        data = json.loads(raw.decode("utf-8"))
        self.assertEqual(data["strategy_id"], "ver_strat")
        self.assertEqual(data["active_version"], "v2.0")
        self.assertIn("SHADOW_LIVE", data["allowed_auto_statuses"])
        self.assertEqual([v["version"] for v in data["versions"]], ["v2.0", "v1.0"])
        for v in data["versions"]:
            for key in ("version", "status", "config_sha256", "actor", "reason", "created_ts_ms"):
                self.assertIn(key, v)

        # Missing strategy_id -> 400
        for bad_path in ("/api/strategies/versions", "/api/strategies/versions?strategy_id="):
            st_b, hdrs_b, raw_b = self._request("GET", bad_path)
            self.assertEqual(st_b, 400)
            self._assert_security_headers(hdrs_b, raw_b)
            self.assertIn("error", json.loads(raw_b.decode("utf-8")))

        # Path traversal / invalid strategy_id -> 400
        st_inv, hdrs_inv, raw_inv = self._request("GET", "/api/strategies/versions?strategy_id=../etc")
        self.assertEqual(st_inv, 400)
        self._assert_security_headers(hdrs_inv, raw_inv)

        # Unknown strategy -> 404
        st_404, hdrs_404, raw_404 = self._request("GET", "/api/strategies/versions?strategy_id=does_not_exist")
        self.assertEqual(st_404, 404)
        self._assert_security_headers(hdrs_404, raw_404)

        # DNS rebinding Host guard -> 403
        st_host, hdrs_host, raw_host = self._request(
            "GET",
            "/api/strategies/versions?strategy_id=ver_strat",
            headers={"Host": "evil.attacker.example"},
        )
        self.assertEqual(st_host, 403)
        self._assert_security_headers(hdrs_host, raw_host)
        self.assertEqual(json.loads(raw_host.decode("utf-8"))["error"], "FORBIDDEN_HOST_HEADER")

    def test_post_agent_run_sprint_valid_clamping_and_validation(self) -> None:
        self.ctx.config_store.register_version(self._make_cfg("sprint_strat", "v0.1", "DRAFT"), actor="unit", reason="seed")

        with mock.patch.dict(os.environ, {k: "" for k in LLM_ENV_KEYS}):
            st, hdrs, raw = self._request(
                "POST",
                "/api/agent/run-sprint",
                body=json.dumps({"strategy_id": "sprint_strat", "max_rounds": 2}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(st, 200)
            self._assert_security_headers(hdrs, raw)
            data = json.loads(raw.decode("utf-8"))
            self.assertEqual(data["strategy_id"], "sprint_strat")
            self.assertEqual(data["plugin_id"], "funding_carry")
            self.assertGreaterEqual(data["rounds_executed"], 1)
            self.assertLessEqual(data["rounds_executed"], 2)
            self.assertEqual(len(data["rounds"]), data["rounds_executed"])
            self.assertTrue(bool(data["stop_reason"]))
            self.assertIn("final_version", data)
            self.assertIn("final_promotion_stage", data)

            # Cycles ledger reflects sprint execution
            st_c, _, raw_c = self._request("GET", "/api/cycles")
            self.assertEqual(st_c, 200)
            cycles = json.loads(raw_c.decode("utf-8"))["cycles"]
            self.assertGreaterEqual(len(cycles), data["rounds_executed"])

            # Invalid max_rounds -> 400
            for bad_rounds in (True, False, 2.5, "3", None):
                st_b, _, d_b = self._post_json("/api/agent/run-sprint", {"strategy_id": "sprint_strat", "max_rounds": bad_rounds})
                self.assertEqual(st_b, 400, f"Expected 400 for max_rounds={bad_rounds!r}")
                self.assertEqual(d_b["error"], "INVALID_MAX_ROUNDS")

            # Invalid / unknown strategy_id
            st_bad_sid, _, _ = self._post_json("/api/agent/run-sprint", {"strategy_id": "../bad", "max_rounds": 1})
            self.assertEqual(st_bad_sid, 400)
            st_miss, _, _ = self._post_json("/api/agent/run-sprint", {"strategy_id": "no_such_strat", "max_rounds": 1})
            self.assertEqual(st_miss, 404)

            # CSRF guards
            st_ct, _, _ = self._request("POST", "/api/agent/run-sprint", body=b"{}", headers={"Content-Type": "text/plain"})
            self.assertEqual(st_ct, 415)
            st_orig, _, _ = self._post_json(
                "/api/agent/run-sprint",
                {"strategy_id": "sprint_strat", "max_rounds": 1},
                extra_headers={"Origin": "https://evil.example"},
            )
            self.assertEqual(st_orig, 403)

    def test_post_micro_runs_evaluate_custom_hours_clamping_and_bounds(self) -> None:
        self.ctx.config_store.register_version(self._make_cfg("micro_strat", "v1.0", "SHADOW_LIVE"), actor="unit", reason="seed")
        now = utc_now_ms() - 10 * 20_000
        ticks: list[BBOTick] = []
        for i in range(10):
            ts = now + i * 20_000
            ticks.append(BBOTick("hyperliquid", "BTC", ts, 999.9, 1000.1, 5.0, 5.0, 20.0, PROVENANCE_REAL))
            ticks.append(BBOTick("gate", "BTC", ts + 25, 999.9, 1000.1, 5.0, 5.0, 25.0, PROVENANCE_REAL))
        self.ctx.store.upsert_bbo_ticks(ticks)

        st, hdrs, raw = self._request(
            "POST",
            "/api/micro-runs/evaluate",
            body=json.dumps({"strategy_id": "micro_strat", "target_hours": 72}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(st, 200)
        self._assert_security_headers(hdrs, raw)
        d = json.loads(raw.decode("utf-8"))
        self.assertEqual(d["strategy_id"], "micro_strat")
        self.assertEqual(d["target_hours"], 72.0)
        for field in ("run_id", "elapsed_hours", "paper_net_pnl_usd", "paper_drawdown_pct", "slippage_ratio", "kill_reason"):
            self.assertIn(field, d)

        # Upper clamp (>720 -> 720.0)
        st_cl, _, d_cl = self._post_json("/api/micro-runs/evaluate", {"strategy_id": "micro_strat", "target_hours": 1000})
        self.assertEqual(st_cl, 200)
        self.assertEqual(d_cl["target_hours"], 720.0)

        # Persisted in GET /api/micro-runs
        st_list, _, raw_list = self._request("GET", "/api/micro-runs")
        self.assertEqual(st_list, 200)
        self.assertGreaterEqual(len(json.loads(raw_list.decode("utf-8"))["runs"]), 2)

        # Invalid target_hours -> 400
        for bad_h in (0, -6, True, False, "24", None):
            st_b, _, d_b = self._post_json("/api/micro-runs/evaluate", {"strategy_id": "micro_strat", "target_hours": bad_h})
            self.assertEqual(st_b, 400, f"Expected 400 for target_hours={bad_h!r}")
            self.assertEqual(d_b["error"], "INVALID_TARGET_HOURS")

        for raw_bad in (
            b'{"strategy_id":"micro_strat","target_hours":NaN}',
            b'{"strategy_id":"micro_strat","target_hours":Infinity}',
        ):
            st_nf, _, raw_nf = self._request(
                "POST",
                "/api/micro-runs/evaluate",
                body=raw_bad,
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(st_nf, 400)
            self.assertEqual(json.loads(raw_nf.decode("utf-8"))["error"], "INVALID_TARGET_HOURS")

        st_inv, _, _ = self._post_json("/api/micro-runs/evaluate", {"strategy_id": "../evil", "target_hours": 24})
        self.assertEqual(st_inv, 400)
        st_404, _, _ = self._post_json("/api/micro-runs/evaluate", {"strategy_id": "ghost_strat", "target_hours": 24})
        self.assertEqual(st_404, 404)

    def test_post_strategies_rollback_auto_allowed_and_human_gated_canary_live(self) -> None:
        self.ctx.config_store.register_version(self._make_cfg("rb_strat", "v1.0", "SHADOW_LIVE"), actor="unit", reason="v1")
        self.ctx.config_store.register_version(
            self._make_cfg("rb_strat", "v2.0", "CANARY_APPROVED"),
            actor="owner",
            reason="v2 canary",
            human_approval_token="HUMAN_APPROVED:SEED-001",
        )
        self.ctx.config_store.register_version(
            self._make_cfg("rb_strat", "v3.0", "LIVE"),
            actor="owner",
            reason="v3 live",
            human_approval_token="HUMAN_APPROVED:SEED-002",
        )
        self.assertEqual(self.ctx.config_store.read_active("rb_strat").version, "v3.0")

        # 1. Auto-allowed SHADOW_LIVE rollback (no token needed -> 200)
        st, hdrs, raw = self._request(
            "POST",
            "/api/strategies/rollback",
            body=json.dumps({"strategy_id": "rb_strat", "target_version": "v1.0", "reason": "revert to shadow"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(st, 200)
        self._assert_security_headers(hdrs, raw)
        self.assertEqual(json.loads(raw.decode("utf-8")), {"rolled_back_to": "v1.0", "strategy_id": "rb_strat"})
        self.assertEqual(self.ctx.config_store.read_active("rb_strat").version, "v1.0")

        # 2. CANARY_APPROVED without token -> 403
        st_no, _, _ = self._post_json("/api/strategies/rollback", {"strategy_id": "rb_strat", "target_version": "v2.0"})
        self.assertEqual(st_no, 403)
        self.assertEqual(self.ctx.config_store.read_active("rb_strat").version, "v1.0")

        # 3. CANARY_APPROVED with malformed tokens -> 403
        for bad_tok in ("HUMAN_APPROVED:", "HUMAN_APPROVED:abc", "HUMAN_APPROVED:bad token", "INVALID:12345"):
            st_bad, _, _ = self._post_json(
                "/api/strategies/rollback",
                {"strategy_id": "rb_strat", "target_version": "v2.0", "human_approval_token": bad_tok},
            )
            self.assertEqual(st_bad, 403, f"Expected 403 for token={bad_tok!r}")
            self.assertEqual(self.ctx.config_store.read_active("rb_strat").version, "v1.0")

        # 4. CANARY_APPROVED with valid token -> 200
        st_ok2, _, d_ok2 = self._post_json(
            "/api/strategies/rollback",
            {"strategy_id": "rb_strat", "target_version": "v2.0", "human_approval_token": "HUMAN_APPROVED:OPS-2026-01"},
        )
        self.assertEqual(st_ok2, 200)
        self.assertEqual(d_ok2["rolled_back_to"], "v2.0")
        self.assertEqual(self.ctx.config_store.read_active("rb_strat").version, "v2.0")

        # 5. LIVE without token -> 403, with valid token -> 200
        st_live_no, _, _ = self._post_json("/api/strategies/rollback", {"strategy_id": "rb_strat", "target_version": "v3.0"})
        self.assertEqual(st_live_no, 403)
        self.assertEqual(self.ctx.config_store.read_active("rb_strat").version, "v2.0")

        st_live_ok, _, d_live_ok = self._post_json(
            "/api/strategies/rollback",
            {"strategy_id": "rb_strat", "target_version": "v3.0", "human_approval_token": "HUMAN_APPROVED:LIVE-SIGN-09"},
        )
        self.assertEqual(st_live_ok, 200)
        self.assertEqual(d_live_ok["rolled_back_to"], "v3.0")
        self.assertEqual(self.ctx.config_store.read_active("rb_strat").version, "v3.0")

        # 6. Unknown version / unknown strategy / invalid strategy_id -> 404
        for payload in (
            {"strategy_id": "rb_strat", "target_version": "v99.9"},
            {"strategy_id": "no_such_strat", "target_version": "v1.0"},
            {"strategy_id": "../escape", "target_version": "v1.0"},
        ):
            st_404, _, _ = self._post_json("/api/strategies/rollback", payload)
            self.assertEqual(st_404, 404)

        # 7. CSRF & Host guards
        st_ct, _, _ = self._request("POST", "/api/strategies/rollback", body=b"{}", headers={"Content-Type": "text/plain"})
        self.assertEqual(st_ct, 415)
        st_orig, _, _ = self._post_json(
            "/api/strategies/rollback",
            {"strategy_id": "rb_strat", "target_version": "v1.0"},
            extra_headers={"Origin": "https://evil.example"},
        )
        self.assertEqual(st_orig, 403)
        st_host, _, _ = self._post_json(
            "/api/strategies/rollback",
            {"strategy_id": "rb_strat", "target_version": "v1.0"},
            extra_headers={"Host": "evil.attacker.example"},
        )
        self.assertEqual(st_host, 403)

    def test_all_get_endpoints_include_security_headers(self) -> None:
        endpoints_200 = (
            "/",
            "/index.html",
            "/api/health",
            "/api/plugins",
            "/api/strategies",
            "/api/dq/reports",
            "/api/micro-runs",
            "/api/fleet/audit",
            "/api/portfolio",
            "/api/shadow/parity",
            "/api/execution/plans",
            "/api/trials",
            "/api/cycles",
            "/api/models",
            "/api/market/summary",
        )
        for ep in endpoints_200:
            st, hdrs, raw = self._request("GET", ep)
            self.assertEqual(st, 200, f"GET {ep} returned {st}")
            self._assert_security_headers(hdrs, raw)

        st_404, hdrs_404, raw_404 = self._request("GET", "/api/nonexistent")
        self.assertEqual(st_404, 404)
        self._assert_security_headers(hdrs_404, raw_404)

    def test_models_and_market_summary_payload_schemas_match_ui_expectations(self) -> None:
        st_m, hdrs_m, raw_m = self._request("GET", "/api/models")
        self.assertEqual(st_m, 200)
        self._assert_security_headers(hdrs_m, raw_m)
        m = json.loads(raw_m.decode("utf-8"))
        self.assertIsInstance(m["routes"], list)
        self.assertGreaterEqual(len(m["routes"]), 1)
        self.assertIsInstance(m["round_spent_usd"], (int, float))
        self.assertIsInstance(m["spent_usd"], (int, float))
        self.assertIsInstance(m["daily_spent_usd"], (int, float))
        self.assertAlmostEqual(float(m["max_cycle_budget_usd"]), 0.50, places=4)
        self.assertAlmostEqual(float(m["max_daily_budget_usd"]), 3.00, places=4)

        now = utc_now_ms()
        self.ctx.store.upsert_funding([FundingRateBar("hyperliquid", "BTC", now, 0.0001, 1.0, PROVENANCE_REAL)])
        self.ctx.store.upsert_candles([PriceBar("hyperliquid", "BTC", now, 100.0, 101.0, 99.0, 100.5, 50000.0, PROVENANCE_REAL)])
        self.ctx.store.upsert_bbo_ticks([BBOTick("hyperliquid", "BTC", now, 100.4, 100.6, 2.0, 2.0, 12.5, PROVENANCE_REAL)])

        st_mk, hdrs_mk, raw_mk = self._request("GET", "/api/market/summary")
        self.assertEqual(st_mk, 200)
        self._assert_security_headers(hdrs_mk, raw_mk)
        mk = json.loads(raw_mk.decode("utf-8"))
        for key in ("funding", "candles_1h", "bbo_ticks"):
            self.assertEqual(len(mk[key]), 1)
            row = mk[key][0]
            for f in ("venue", "symbols", "rows", "min_ts", "max_ts"):
                self.assertIn(f, row)
        self.assertIn("avg_latency_ms", mk["bbo_ticks"][0])
        self.assertAlmostEqual(float(mk["bbo_ticks"][0]["avg_latency_ms"]), 12.5, places=1)


if __name__ == "__main__":
    unittest.main()
