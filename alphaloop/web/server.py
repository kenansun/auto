"""Zero-dependency HTTP Strategy R&D Console (STD-34 & L-034 hardened).

Security & reliability invariants:
- Default loopback bind 127.0.0.1 (no accidental public exposure).
- Strict Content-Length validation (rejects negative, malformed, or >1MB bodies with 400/413).
- Explicit Content-Length & Cache-Control: no-cache on every response.
- Client-side escapeHtml() on all dynamic strings (prevents DOM XSS).
- Zero hardcoded fake data (STD-37).
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from alphaloop.agent_loop.model_router import MultiModelRouter
from alphaloop.agent_loop.orchestrator import AgentEvolutionOrchestrator
from alphaloop.core.config import HotConfigStore
from alphaloop.core.provenance import utc_now_ms
from alphaloop.data.store import MarketDataStore
from alphaloop.harness.backtester import FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier, TrialLedger

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 1024 * 1024  # 1MB hard cap (L-034 / RB3-05)

CONSOLE_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AlphaLoop v1 — Strategy R&amp;D &amp; Agent Feedback Console</title>
<style>
  :root{--bg:#0b0f17;--card:#131a26;--border:#233044;--text:#e6edf7;--muted:#8fa2bd;--green:#22c55e;--red:#ef4444;--amber:#f59e0b;--blue:#38bdf8;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.5}
  header{display:flex;justify-content:space-between;align-items:center;padding:14px 22px;border-bottom:1px solid var(--border);background:#0f1623}
  h1{font-size:17px;margin:0;letter-spacing:.3px} .badge{font-size:12px;padding:3px 9px;border-radius:999px;background:#1e293b;color:var(--blue)}
  main{max-width:1280px;margin:0 auto;padding:18px;display:grid;gap:16px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px}
  .card h2{font-size:14px;color:var(--muted);margin:0 0 10px;text-transform:uppercase;letter-spacing:.5px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--border)}
  th{color:var(--muted);font-weight:600}
  .ok{color:var(--green);font-weight:600} .bad{color:var(--red);font-weight:600} .warn{color:var(--amber);font-weight:600}
  button{background:var(--blue);color:#04111d;border:none;padding:7px 12px;border-radius:6px;font-weight:600;cursor:pointer;font-size:12px}
  button:hover{opacity:.9} button.sec{background:#1f2d42;color:var(--text)}
  pre{background:#090d14;border:1px solid var(--border);border-radius:8px;padding:10px;overflow:auto;font-size:12px;max-height:260px}
</style>
</head>
<body>
<header>
  <h1>AlphaLoop v1 — Strategy R&amp;D &amp; Multi-Model Agent Feedback Platform</h1>
  <div><span id="healthBadge" class="badge">Loading...</span> <button class="sec" onclick="refreshAll()">Refresh</button></div>
</header>
<main>
  <div class="grid" id="kpiGrid"></div>
  <div class="card">
    <h2>Active Hot-Reloadable Strategies (Atomic rename(2) Bus)</h2>
    <div id="strategiesBox">Loading...</div>
  </div>
  <div class="card">
    <h2>Closed-Loop Agent Evolution Cycles (Diagnose → Propose → 6-Gate → Cross-Model Red-Team)</h2>
    <div id="cyclesBox">Loading...</div>
  </div>
  <div class="card">
    <h2>Multi-Model Consequence Router Matrix</h2>
    <div id="modelsBox">Loading...</div>
  </div>
  <div class="card">
    <h2>Append-Only Multiple-Testing Trial Ledger (G6)</h2>
    <div id="trialsBox">Loading...</div>
  </div>
</main>
<script>
function esc(v){return String(v??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));}
async function getJson(u){const r=await fetch(u,{cache:'no-store'});if(!r.ok)throw new Error('HTTP '+r.status);return r.json();}
async function triggerCycle(sid){
  const ver='v-auto-'+Math.floor(Date.now()/1000);
  const r=await fetch('/api/agent/run-cycle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({strategy_id:sid,next_version:ver})});
  const d=await r.json();
  alert('Cycle '+esc(d.cycle_id||'ERR')+': gates='+(d.gate_verdict?d.gate_verdict.passed_count+'/'+d.gate_verdict.total_count:'?')+' red_team='+(d.red_team_verdict?d.red_team_verdict.decision:d.error));
  refreshAll();
}
async function refreshAll(){
  try{
    const [h,s,c,m,t,mk]=await Promise.all([
      getJson('/api/health'),getJson('/api/strategies'),getJson('/api/cycles'),
      getJson('/api/models'),getJson('/api/trials'),getJson('/api/market/summary')
    ]);
    document.getElementById('healthBadge').textContent='ONLINE | '+h.status+' | ts='+new Date(h.ts_ms).toISOString();
    const totalFundingRows=(mk.funding||[]).reduce((a,x)=>a+x.rows,0);
    document.getElementById('kpiGrid').innerHTML=[
      ['Registered Strategies', (s.strategies||[]).length],
      ['Real Funding Rows', totalFundingRows.toLocaleString()],
      ['Agent Evolution Cycles', (c.cycles||[]).length],
      ['Recorded G6 Trials', (t.trials||[]).length]
    ].map(([k,v])=>'<div class="card"><h2>'+esc(k)+'</h2><div style="font-size:26px;font-weight:700">'+esc(v)+'</div></div>').join('');

    const sRows=(s.strategies||[]).map(x=>'<tr><td><b>'+esc(x.strategy_id)+'</b></td><td>'+esc(x.active_version)+'</td><td><span class="badge">'+esc(x.status)+'</span></td><td>'+esc(x.config.venue_a)+' / '+esc(x.config.venue_b)+'</td><td>'+esc(x.config.entry_spread_apr_pct)+'% / '+esc(x.config.exit_spread_apr_pct)+'%</td><td><code>'+esc(x.config_sha256.slice(0,12))+'</code></td><td><button onclick="triggerCycle(\\''+esc(x.strategy_id)+'\\')">Run Agent Cycle</button></td></tr>').join('');
    document.getElementById('strategiesBox').innerHTML='<table><thead><tr><th>Strategy</th><th>Version</th><th>Status</th><th>Venues</th><th>Entry/Exit APR</th><th>SHA256</th><th>Action</th></tr></thead><tbody>'+(sRows||'<tr><td colspan="7">No strategies registered</td></tr>')+'</tbody></table>';

    const cRows=(c.cycles||[]).map(x=>'<tr><td><code>'+esc(x.cycle_id)+'</code></td><td>'+esc(x.strategy_id)+' ('+esc(x.from_version)+'→'+esc(x.to_version)+')</td><td>'+esc(x.feedback.anomaly_type)+'</td><td>'+esc(x.baseline_metrics.annualized_apr_pct)+'% → <b>'+esc(x.candidate_metrics.annualized_apr_pct)+'%</b></td><td class="'+(x.gate_verdict.overall_pass?'ok':'bad')+'">'+esc(x.gate_verdict.passed_count)+'/'+esc(x.gate_verdict.total_count)+'</td><td>'+esc(x.red_team_verdict.decision)+'</td><td>'+(x.promoted_shadow?'<span class="ok">STAGED SHADOW</span>':'<span class="warn">HELD</span>')+'</td></tr>').join('');
    document.getElementById('cyclesBox').innerHTML='<table><thead><tr><th>Cycle</th><th>Strategy</th><th>Anomaly</th><th>APR Shift</th><th>6-Gate</th><th>Red-Team</th><th>Outcome</th></tr></thead><tbody>'+(cRows||'<tr><td colspan="7">No cycles yet</td></tr>')+'</tbody></table>';

    const mRows=(m.routes||[]).map(x=>'<tr><td><b>'+esc(x.role)+'</b></td><td>'+esc(x.family)+' / '+esc(x.model_id)+'</td><td>'+esc(x.pricing_usd_1m)+'</td><td>'+(x.configured?'<span class="ok">KEY SET</span>':'<span class="warn">FALLBACK</span>')+'</td><td>'+esc(x.rationale)+'</td></tr>').join('');
    document.getElementById('modelsBox').innerHTML='<table><thead><tr><th>Role</th><th>Model</th><th>Price (In/Out 1M)</th><th>API Status</th><th>Rationale</th></tr></thead><tbody>'+mRows+'</tbody></table>';

    const tRows=(t.trials||[]).map(x=>'<tr><td>#'+esc(x.trial_id)+'</td><td>'+esc(x.strategy_id)+'@'+esc(x.version)+'</td><td>'+esc(x.provenance)+'</td><td>$'+esc(x.net_pnl_usd)+'</td><td>'+esc(x.apr_pct)+'%</td><td>'+esc(x.sharpe)+'</td><td>'+esc(x.max_dd_pct)+'%</td><td class="'+(x.overall_pass?'ok':'bad')+'">'+(x.overall_pass?'PASS':'FAIL')+'</td></tr>').join('');
    document.getElementById('trialsBox').innerHTML='<table><thead><tr><th>Trial</th><th>Strategy</th><th>Provenance</th><th>Net PnL</th><th>APR</th><th>Sharpe</th><th>MaxDD</th><th>6-Gate</th></tr></thead><tbody>'+(tRows||'<tr><td colspan="8">No trials recorded</td></tr>')+'</tbody></table>';
  }catch(e){
    document.getElementById('healthBadge').textContent='ERROR: '+e.message;
  }
}
refreshAll();
</script>
</body>
</html>
"""


class ConsoleAppContext:
    """Shared dependency container for AlphaLoop HTTP server."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = MarketDataStore(self.data_dir / "market_data.sqlite")
        self.ledger = TrialLedger(self.data_dir / "trial_ledger.sqlite")
        self.config_store = HotConfigStore(self.data_dir / "state")
        self.backtester = FundingCarryBacktester(self.store)
        self.verifier = SixGateVerifier(self.backtester, self.ledger)
        self.router = MultiModelRouter()
        self.orchestrator = AgentEvolutionOrchestrator(
            state_dir=self.data_dir / "state",
            backtester=self.backtester,
            verifier=self.verifier,
            config_store=self.config_store,
            router=self.router,
        )

    def close(self) -> None:
        self.store.close()
        self.ledger.close()
        self.config_store.close()
        self.orchestrator.close()


def _make_handler(ctx: ConsoleAppContext) -> type[BaseHTTPRequestHandler]:
    class AlphaLoopHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("HTTP %s - %s", self.address_string(), fmt % args)

        def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, obj: Any) -> None:
            raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", raw)

        def _read_body(self) -> tuple[bytes | None, int | None, str | None]:
            raw_len = self.headers.get("Content-Length")
            if raw_len is None:
                return b"", None, None
            try:
                length = int(raw_len.strip())
            except ValueError:
                return None, 400, "INVALID_CONTENT_LENGTH"
            if length < 0:
                return None, 400, "NEGATIVE_CONTENT_LENGTH"
            if length > MAX_BODY_BYTES:
                return None, 413, f"PAYLOAD_TOO_LARGE_MAX_{MAX_BODY_BYTES}"
            return self.rfile.read(length), None, None

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send_bytes(200, "text/html; charset=utf-8", CONSOLE_HTML.encode("utf-8"))
                return
            if path == "/api/health":
                self._send_json(200, {"status": "HEALTHY", "service": "alphaloop-v1", "ts_ms": utc_now_ms()})
                return
            if path == "/api/strategies":
                self._send_json(200, {"strategies": ctx.config_store.list_strategies()})
                return
            if path == "/api/trials":
                self._send_json(200, {"trials": ctx.ledger.recent_trials(limit=50)})
                return
            if path == "/api/cycles":
                self._send_json(200, {"cycles": ctx.orchestrator.recent_cycles(limit=25)})
                return
            if path == "/api/models":
                self._send_json(200, {"routes": ctx.router.routing_matrix(), "spent_usd": round(ctx.router.spent_usd, 6)})
                return
            if path == "/api/market/summary":
                self._send_json(200, ctx.store.summary_stats())
                return
            self._send_json(404, {"error": "NOT_FOUND", "path": path})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            body, err_code, err_msg = self._read_body()
            if err_code is not None or body is None:
                self._send_json(err_code or 400, {"error": err_msg})
                return
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": f"INVALID_JSON: {exc}"})
                return

            if path == "/api/agent/run-cycle":
                sid = str(payload.get("strategy_id", "")).strip()
                next_ver = str(payload.get("next_version", f"v-{utc_now_ms()}")).strip()
                cfg = ctx.config_store.read_active(sid)
                if cfg is None:
                    self._send_json(404, {"error": f"STRATEGY_NOT_FOUND: {sid}"})
                    return
                rec = ctx.orchestrator.run_cycle(cfg, next_version=next_ver)
                self._send_json(200, rec)
                return

            if path == "/api/strategies/rollback":
                sid = str(payload.get("strategy_id", "")).strip()
                target_ver = str(payload.get("target_version", "")).strip()
                reason = str(payload.get("reason", "Manual console rollback")).strip()
                try:
                    cfg = ctx.config_store.rollback(sid, target_ver, actor="console-operator", reason=reason)
                except ValueError as exc:
                    self._send_json(404, {"error": str(exc)})
                    return
                self._send_json(200, {"rolled_back_to": cfg.version, "strategy_id": cfg.strategy_id})
                return

            self._send_json(404, {"error": "NOT_FOUND", "path": path})

    return AlphaLoopHandler


def create_server(data_dir: str | Path, host: str = "127.0.0.1", port: int = 8796) -> tuple[ThreadingHTTPServer, ConsoleAppContext]:
    ctx = ConsoleAppContext(data_dir)
    srv = ThreadingHTTPServer((host, port), _make_handler(ctx))
    return srv, ctx
