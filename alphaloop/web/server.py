"""Zero-dependency HTTP Strategy R&D Console v2 (STD-34 & L-034 hardened).

Security & reliability invariants:
- Default loopback bind 127.0.0.1 (no accidental public exposure).
- Strict Content-Length validation (rejects negative, malformed, or >1MB bodies with 400/413).
- Explicit Content-Length & Cache-Control: no-cache on every response.
- Client-side escapeHtml() on all dynamic strings (prevents DOM XSS).
- Zero hardcoded fake data (STD-37); fleet panel reads latest saved read-only snapshot only.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from alphaloop.agent_loop.model_router import MultiModelRouter
from alphaloop.agent_loop.orchestrator import AgentEvolutionOrchestrator
from alphaloop.core.config import HotConfigStore
from alphaloop.core.provenance import utc_now_ms
from alphaloop.data.store import MarketDataStore
from alphaloop.execution.bridge import ExecutionBridge
from alphaloop.harness.backtester import FundingCarryBacktester
from alphaloop.harness.gates import SixGateVerifier, TrialLedger
from alphaloop.harness.micro_runner import MicroRunSession, MicroRunStore
from alphaloop.plugins import PluginRegistry, default_registry

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 1024 * 1024  # 1MB hard cap (L-034 / RB3-05)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
LOOPBACK_ORIGIN_RE = re.compile(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d{1,5})?$")

CONSOLE_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AlphaLoop v2.1 — Strategy Factory, Portfolio, DQ Gate &amp; Shadow Execution Console</title>
<style>
  :root{--bg:#0b0f17;--card:#131a26;--border:#233044;--text:#e6edf7;--muted:#8fa2bd;--green:#22c55e;--red:#ef4444;--amber:#f59e0b;--blue:#38bdf8;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.5}
  header{display:flex;justify-content:space-between;align-items:center;padding:14px 22px;border-bottom:1px solid var(--border);background:#0f1623}
  h1{font-size:17px;margin:0;letter-spacing:.3px} .badge{font-size:12px;padding:3px 9px;border-radius:999px;background:#1e293b;color:var(--blue)}
  main{max-width:1360px;margin:0 auto;padding:18px;display:grid;gap:16px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;overflow-x:auto}
  .card h2{font-size:13px;color:var(--muted);margin:0 0 10px;text-transform:uppercase;letter-spacing:.5px}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--border)}
  th{color:var(--muted);font-weight:600}
  .ok{color:var(--green);font-weight:600} .bad{color:var(--red);font-weight:600} .warn{color:var(--amber);font-weight:600}
  button{background:var(--blue);color:#04111d;border:none;padding:6px 10px;border-radius:6px;font-weight:600;cursor:pointer;font-size:12px;margin-right:4px}
  button:hover{opacity:.9} button.sec{background:#1f2d42;color:var(--text)}
</style>
</head>
<body>
<header>
  <h1>AlphaLoop v2.2 — Strategy Factory, Portfolio Combiner, DQ-01..08 Gate &amp; Always-On Shadow Parity</h1>
  <div><span id="healthBadge" class="badge">Loading...</span> <button class="sec" id="refreshBtn" type="button">Refresh</button></div>
</header>
<main>
  <div class="grid" id="kpiGrid"></div>
  <div class="card">
    <h2>Live Shadow Parity &amp; Drift Guard (SHADOW_DRY_RUN vs Backtest)</h2>
    <div id="shadowParityBox">Loading...</div>
  </div>
  <div class="card">
    <h2>Multi-Pillar Portfolio Scorecard &amp; Capacity Scaling (PortfolioCombiner)</h2>
    <div id="portfolioBox">Loading...</div>
  </div>
  <div class="card">
    <h2>Shadow Execution Bridge (SHADOW_DRY_RUN · Hash-Chained Audit · Kill Switch)</h2>
    <div id="execBox">Loading...</div>
  </div>
  <div class="card">
    <h2>Strategy Factory v2.2 Plugins (Pluggable Alpha Registry)</h2>
    <div id="pluginsBox">Loading...</div>
  </div>
  <div class="card">
    <h2>Active Hot-Reloadable Strategies (Atomic rename(2) Bus)</h2>
    <div id="strategiesBox">Loading...</div>
  </div>
  <div class="card">
    <h2>Data Quality Certification Ledger (DQ-01 .. DQ-08)</h2>
    <div id="dqBox">Loading...</div>
  </div>
  <div class="card">
    <h2>Fast Micro-Run Validation &amp; Dual-Layer Early-Kill Guard</h2>
    <div id="microBox">Loading...</div>
  </div>
  <div class="card">
    <h2>5-Node Prepaid Server Fleet Health (Read-Only SSH Snapshot)</h2>
    <div id="fleetBox">Loading...</div>
  </div>
  <div class="card">
    <h2>Closed-Loop Agent Evolution &amp; Version Diff (Diagnose → Propose → DQ + 6-Gate → Red-Team)</h2>
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
  alert('Cycle '+(d.cycle_id||'ERR')+': stage='+(d.promotion_stage||'?')+' gates='+(d.gate_verdict?d.gate_verdict.passed_count+'/'+d.gate_verdict.total_count:'?')+' red_team='+(d.red_team_verdict?d.red_team_verdict.decision:d.error));
  refreshAll();
}
async function triggerMicroRun(sid){
  const r=await fetch('/api/micro-runs/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({strategy_id:sid})});
  const d=await r.json();
  alert('MicroRun '+(d.run_id||'ERR')+': status='+(d.status||d.error)+' slip_ratio='+(d.slippage_ratio)+'x kill='+(d.kill_reason||'none'));
  refreshAll();
}
document.getElementById('refreshBtn').addEventListener('click', refreshAll);
document.getElementById('strategiesBox').addEventListener('click', function(ev){
  const btn = ev.target.closest('button[data-action]');
  if(!btn) return;
  const sid = btn.getAttribute('data-strategy-id') || '';
  if(!sid) return;
  if(btn.getAttribute('data-action') === 'evolve') triggerCycle(sid);
  if(btn.getAttribute('data-action') === 'micro') triggerMicroRun(sid);
});
async function refreshAll(){
  try{
    const [h,s,c,m,t,mk,pl,dq,mr,fl,pf,ex,sp]=await Promise.all([
      getJson('/api/health'),getJson('/api/strategies'),getJson('/api/cycles'),
      getJson('/api/models'),getJson('/api/trials'),getJson('/api/market/summary'),
      getJson('/api/plugins'),getJson('/api/dq/reports'),getJson('/api/micro-runs'),
      getJson('/api/fleet/audit'),getJson('/api/portfolio'),getJson('/api/execution/plans'),
      getJson('/api/shadow/parity')
    ]);
    document.getElementById('healthBadge').textContent='ONLINE | '+h.status+' | '+h.service+' | '+new Date(h.ts_ms).toISOString();
    const totalFundingRows=(mk.funding||[]).reduce((a,x)=>a+x.rows,0);
    const totalBboRows=(mk.bbo_ticks||[]).reduce((a,x)=>a+x.rows,0);
    document.getElementById('kpiGrid').innerHTML=[
      ['Strategy Plugins', (pl.plugins||[]).length],
      ['Active Strategies', (s.strategies||[]).length],
      ['Real Funding Rows', totalFundingRows.toLocaleString()],
      ['Live BBO Ticks', totalBboRows.toLocaleString()],
      ['DQ Certificates', (dq.reports||[]).length],
      ['Shadow Parity', sp.valid ? (sp.status||'OK') : 'IDLE'],
      ['Shadow Audit Seq', '#'+(ex.audit_head_seq??0)+' ('+(ex.audit_chain_valid?'VALID':'BROKEN')+')'],
      ['Fleet Reachable', (fl.nodes_reachable??0)+'/'+(fl.nodes_total??0)]
    ].map(([k,v])=>'<div class="card"><h2>'+esc(k)+'</h2><div style="font-size:22px;font-weight:700">'+esc(v)+'</div></div>').join('');

    if(sp.valid){
      const spRows=(sp.strategies||[]).map(x=>'<tr><td><b>'+esc(x.strategy_id)+'</b>@'+esc(x.version)+'</td><td>'+esc(x.plugin_id)+'</td><td>$'+esc(x.shadow_equity_usd)+' ('+esc(x.shadow_return_pct)+'%)</td><td>'+esc(x.backtest_return_pct)+'%</td><td class="'+(Math.abs(x.drift_bps||0)<=150?'ok':'bad')+'">'+esc(x.drift_bps)+' bp</td><td class="'+((x.spread_ratio||0)<=1.5?'ok':'warn')+'">'+esc(x.live_median_half_spread_bps)+'bp ('+esc(x.spread_ratio)+'x)</td><td>'+esc(x.plans_created)+' / '+esc(x.plans_blocked)+'</td><td class="'+(x.status==='PARITY_OK'?'ok':(x.status==='WARN_WIDE_SPREAD'?'warn':'bad'))+'">'+esc(x.status)+'</td></tr>').join('');
      document.getElementById('shadowParityBox').innerHTML='<div style="margin-bottom:8px">Status: <b class="'+(sp.status==='PARITY_OK'?'ok':(sp.status==='DEGRADED_HALT'?'warn':'bad'))+'">'+esc(sp.status)+'</b> | BBO Age: <b>'+esc(sp.bbo_freshness_sec)+'s</b> | Live Enabled: <b>'+esc(sp.live_execution_enabled)+'</b> | Trace: <code>'+esc(sp.trace_id||'')+'</code></div><table><thead><tr><th>Strategy</th><th>Plugin</th><th>Shadow Equity</th><th>Backtest Ret</th><th>Drift</th><th>Live Spread</th><th>Plans (OK/Blk)</th><th>Parity</th></tr></thead><tbody>'+(spRows||'<tr><td colspan="8">No active SHADOW_LIVE strategies evaluated yet</td></tr>')+'</tbody></table>';
    }else{
      document.getElementById('shadowParityBox').innerHTML='<span class="warn">'+esc(sp.error||'Shadow daemon has not written shadow_parity_report.json yet')+'</span>';
    }

    if(pf.valid){
      const scRows=(pf.scaling_projections||[]).map(x=>'<tr><td>$'+esc(x.capital_usd)+'</td><td>'+esc(x.gross_backtest_apr_pct)+'%</td><td>-'+esc(x.capacity_impact_drag_apr_pct)+'%</td><td class="'+(x.net_capacity_adjusted_apr_pct>0?'ok':'bad')+'"><b>'+esc(x.net_capacity_adjusted_apr_pct)+'%</b></td><td>$'+esc(x.est_monthly_net_usd)+'/mo</td></tr>').join('');
      document.getElementById('portfolioBox').innerHTML='<div style="margin-bottom:8px">Mode: <b>'+esc(pf.weighting_mode)+'</b> | Overlap: <b>'+esc(pf.overlap_days)+'d</b> | APR: <b class="'+(pf.annualized_apr_pct>0?'ok':'bad')+'">'+esc(pf.annualized_apr_pct)+'%</b> | Sharpe: <b>'+esc(pf.sharpe_ratio)+'</b> | MaxDD: <b>'+esc(pf.max_drawdown_pct)+'%</b> | OOS Ret: <b>'+esc(pf.oos_retention)+'x</b></div><table><thead><tr><th>Capital</th><th>Backtest APR</th><th>Capacity Drag</th><th>Net Adjusted APR</th><th>Est Monthly</th></tr></thead><tbody>'+scRows+'</tbody></table>';
    }else{
      document.getElementById('portfolioBox').innerHTML='<span class="warn">'+esc(pf.error||'No portfolio report saved yet')+'</span>';
    }

    const ks=ex.kill_switch||{};
    const exRows=(ex.recent_plans||[]).slice(0,12).map(x=>'<tr><td><code>'+esc(x.plan_id)+'</code></td><td>'+esc(x.strategy_id)+'@'+esc(x.version)+'</td><td>'+esc(x.symbol)+'</td><td>'+esc(x.action)+'</td><td>$'+esc(x.notional_usd)+'</td><td class="'+(x.status==='PLANNED_SHADOW'?'ok':'bad')+'">'+esc(x.status)+'</td><td>'+esc(x.block_reason||'—')+'</td></tr>').join('');
    document.getElementById('execBox').innerHTML='<div style="margin-bottom:8px">Mode: <span class="badge">'+esc(ex.run_mode)+'</span> | Live Enabled: <b>'+esc(ex.live_execution_enabled)+'</b> | Kill Switch: <b class="'+(ks.engaged?'bad':'ok')+'">'+(ks.engaged?'ENGAGED ('+esc(ks.trigger)+')':'CLEAR')+'</b> | Audit Chain: <b class="'+(ex.audit_chain_valid?'ok':'bad')+'">'+esc(ex.audit_chain_detail)+' ('+esc(ex.audit_chain_length)+' events)</b></div><table><thead><tr><th>Plan ID</th><th>Strategy</th><th>Symbol</th><th>Action</th><th>Notional</th><th>Status</th><th>Block Reason</th></tr></thead><tbody>'+(exRows||'<tr><td colspan="7">No shadow execution plans recorded yet</td></tr>')+'</tbody></table>';

    const plRows=(pl.plugins||[]).map(x=>'<tr><td><b>'+esc(x.plugin_id)+'</b></td><td>'+esc(x.display_name)+'</td><td>'+esc(x.strategy_family)+'</td><td>'+esc(Object.keys(x.mutable_param_bounds||{}).join(', '))+'</td><td>'+esc(x.description)+'</td></tr>').join('');
    document.getElementById('pluginsBox').innerHTML='<table><thead><tr><th>Plugin ID</th><th>Name</th><th>Family</th><th>Mutable Params</th><th>Description</th></tr></thead><tbody>'+(plRows||'<tr><td colspan="5">No plugins</td></tr>')+'</tbody></table>';

    const sRows=(s.strategies||[]).map(x=>'<tr><td><b>'+esc(x.strategy_id)+'</b></td><td>'+esc(x.config.plugin_id||'funding_carry')+'</td><td>'+esc(x.active_version)+'</td><td><span class="badge">'+esc(x.status)+'</span></td><td>'+esc(x.config.venue_a)+' / '+esc(x.config.venue_b)+'</td><td><code>'+esc(x.config_sha256.slice(0,12))+'</code></td><td><button type="button" data-action="evolve" data-strategy-id="'+esc(x.strategy_id)+'">Evolve</button><button type="button" class="sec" data-action="micro" data-strategy-id="'+esc(x.strategy_id)+'">MicroRun</button></td></tr>').join('');
    document.getElementById('strategiesBox').innerHTML='<table><thead><tr><th>Strategy</th><th>Plugin</th><th>Version</th><th>Status</th><th>Venues</th><th>SHA256</th><th>Action</th></tr></thead><tbody>'+(sRows||'<tr><td colspan="7">No strategies registered</td></tr>')+'</tbody></table>';

    const dqRows=(dq.reports||[]).slice(0,15).map(x=>'<tr><td><code>'+esc(x.cert_id)+'</code></td><td>'+esc(x.dataset_name)+'</td><td>'+esc(x.venue_pair)+'</td><td>'+esc(x.row_count)+'</td><td>'+esc(x.basis_coverage_pct)+'%</td><td>'+esc(x.passed_count)+'/'+esc(x.total_count)+'</td><td class="'+(x.tier==='DATA_QUALIFIED_FULL'?'ok':(x.tier==='DATA_REJECTED'?'bad':'warn'))+'">'+esc(x.tier)+'</td></tr>').join('');
    document.getElementById('dqBox').innerHTML='<table><thead><tr><th>Cert ID</th><th>Dataset</th><th>Pair</th><th>Rows</th><th>Basis Cov</th><th>Checks</th><th>Tier</th></tr></thead><tbody>'+(dqRows||'<tr><td colspan="7">No DQ certificates yet</td></tr>')+'</tbody></table>';

    const mrRows=(mr.runs||[]).slice(0,15).map(x=>'<tr><td><code>'+esc(x.run_id)+'</code></td><td>'+esc(x.strategy_id)+'@'+esc(x.version)+'</td><td>'+esc(x.plugin_id)+'</td><td class="'+(x.status==='COMPLETED_PASS'?'ok':(x.status==='KILLED_EARLY'?'bad':'warn'))+'">'+esc(x.status)+'</td><td>'+esc(x.ticks_observed)+'</td><td>'+esc(x.realized_spread_bps_median)+'bp / '+esc(x.assumed_slippage_bps)+'bp ('+esc(x.slippage_ratio)+'x)</td><td>'+esc(x.dq_tier)+'</td><td>'+esc(x.kill_reason||'—')+'</td></tr>').join('');
    document.getElementById('microBox').innerHTML='<table><thead><tr><th>Run ID</th><th>Strategy</th><th>Plugin</th><th>Status</th><th>BBO Ticks</th><th>Live vs Assumed Slip</th><th>DQ Tier</th><th>Kill / Note</th></tr></thead><tbody>'+(mrRows||'<tr><td colspan="8">No micro-runs yet</td></tr>')+'</tbody></table>';

    const flRows=(fl.nodes||[]).map(x=>'<tr><td><b>'+esc(x.label)+'</b></td><td><code>'+esc(x.ip)+'</code></td><td class="'+(x.reachable?'ok':'bad')+'">'+(x.reachable?'OK':'DOWN')+'</td><td>'+esc(x.root_disk_pct)+'% ('+esc(x.root_disk_free_gb)+'GB free)</td><td>'+esc(x.mem_used_pct)+'% ('+esc(x.mem_avail_mb)+'MB avail)</td><td>'+esc((x.alerts||[]).map(a=>a.severity+':'+a.code).join(', ')||'NONE')+'</td></tr>').join('');
    document.getElementById('fleetBox').innerHTML='<table><thead><tr><th>Node</th><th>IP</th><th>SSH</th><th>Root Disk</th><th>RAM</th><th>Alerts</th></tr></thead><tbody>'+(flRows||'<tr><td colspan="6">No fleet snapshot found</td></tr>')+'</tbody></table>';

    const cRows=(c.cycles||[]).map(x=>{const df=x.diff_scorecard||{};return '<tr><td><code>'+esc(x.cycle_id)+'</code></td><td>'+esc(x.strategy_id)+' ('+esc(x.from_version)+'→'+esc(x.to_version)+')</td><td>'+esc(x.feedback.anomaly_type)+'</td><td>'+esc(x.baseline_metrics.annualized_apr_pct)+'% → <b>'+esc(x.candidate_metrics.annualized_apr_pct)+'%</b> ('+(df.apr_delta_pct>=0?'+':'')+esc(df.apr_delta_pct??0)+'%)</td><td>'+esc(df.sharpe_delta??0)+' / '+esc(df.max_dd_delta_pct??0)+'%</td><td class="'+(x.gate_verdict.overall_pass?'ok':'bad')+'">'+esc(x.gate_verdict.passed_count)+'/'+esc(x.gate_verdict.total_count)+'</td><td>'+esc(x.red_team_verdict.decision)+'</td><td><span class="badge">'+esc(x.promotion_stage||(x.promoted_shadow?'SHADOW_LIVE':'HELD'))+'</span></td></tr>';}).join('');
    document.getElementById('cyclesBox').innerHTML='<table><thead><tr><th>Cycle</th><th>Strategy</th><th>Anomaly</th><th>APR Shift</th><th>ΔSharpe / ΔDD</th><th>6-Gate</th><th>Red-Team</th><th>Stage</th></tr></thead><tbody>'+(cRows||'<tr><td colspan="8">No cycles yet</td></tr>')+'</tbody></table>';

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
    """Shared dependency container for AlphaLoop v2.1 HTTP server."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry: PluginRegistry = default_registry()
        self.store = MarketDataStore(self.data_dir / "market_data.sqlite")
        self.ledger = TrialLedger(self.data_dir / "trial_ledger.sqlite")
        self.config_store = HotConfigStore(self.data_dir / "state")
        self.micro_store = MicroRunStore(self.data_dir / "state" / "micro_runs.sqlite")
        self.micro_session = MicroRunSession(self.store, self.micro_store, self.registry)
        self.exec_bridge = ExecutionBridge(self.data_dir / "state")
        self.backtester = FundingCarryBacktester(self.store)
        self.verifier = SixGateVerifier(self.backtester, self.ledger, registry=self.registry)
        self.router = MultiModelRouter(budget_db_path=str(self.data_dir / "state" / "llm_budget.sqlite"))
        self.orchestrator = AgentEvolutionOrchestrator(
            state_dir=self.data_dir / "state",
            backtester=self.backtester,
            verifier=self.verifier,
            config_store=self.config_store,
            router=self.router,
            registry=self.registry,
            micro_store=self.micro_store,
        )

    def read_fleet_snapshot(self) -> dict[str, Any]:
        snap = self.data_dir / "fleet_audit" / "fleet_snapshot.json"
        if not snap.exists():
            return {"nodes_total": 0, "nodes_reachable": 0, "p0_alerts": 0, "p1_alerts": 0, "nodes": []}
        try:
            return json.loads(snap.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"nodes_total": 0, "nodes_reachable": 0, "nodes": [], "error": "UNREADABLE_SNAPSHOT"}

    def read_portfolio_snapshot(self) -> dict[str, Any]:
        snap = self.data_dir / "portfolio_report.json"
        if not snap.exists():
            return {"valid": False, "error": "NO_PORTFOLIO_REPORT_SAVED", "scaling_projections": []}
        try:
            data = json.loads(snap.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"valid": False, "error": "INVALID_PORTFOLIO_REPORT"}
        except (OSError, json.JSONDecodeError):
            return {"valid": False, "error": "UNREADABLE_PORTFOLIO_REPORT"}

    def read_shadow_parity_snapshot(self) -> dict[str, Any]:
        snap = self.data_dir / "shadow_parity_report.json"
        if not snap.exists():
            return {
                "valid": False,
                "live_execution_enabled": False,
                "run_mode": "SHADOW_DRY_RUN",
                "error": "NO_SHADOW_PARITY_REPORT_SAVED",
                "strategies": [],
            }
        try:
            data = json.loads(snap.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"valid": False, "error": "INVALID_SHADOW_PARITY_REPORT"}
        except (OSError, json.JSONDecodeError):
            return {"valid": False, "error": "UNREADABLE_SHADOW_PARITY_REPORT"}

    def close(self) -> None:
        self.store.close()
        self.ledger.close()
        self.config_store.close()
        self.micro_store.close()
        self.exec_bridge.close()
        self.orchestrator.close()
        self.router.close()


def _make_handler(ctx: ConsoleAppContext) -> type[BaseHTTPRequestHandler]:
    class AlphaLoopHandler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(15.0)

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("HTTP %s - %s", self.address_string(), fmt % args)

        def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _check_loopback_host(self) -> bool:
            bind_host = str(self.server.server_address[0])
            if bind_host not in LOOPBACK_HOSTS:
                return True
            host_hdr = (self.headers.get("Host") or "").strip()
            if not host_hdr:
                return True
            return bool(re.fullmatch(r"(127\.0\.0\.1|localhost|\[::1\])(:\d{1,5})?", host_hdr, re.IGNORECASE))

        def _send_json(self, status: int, obj: Any) -> None:
            raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", raw)

        def _read_body(self) -> tuple[bytes | None, int | None, str | None]:
            if self.headers.get("Transfer-Encoding"):
                return None, 411, "CHUNKED_ENCODING_UNSUPPORTED"
            cl_all = self.headers.get_all("Content-Length") or []
            if len(cl_all) > 1:
                return None, 400, "DUPLICATE_CONTENT_LENGTH"
            raw_len = cl_all[0] if cl_all else None
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
            try:
                if not self._check_loopback_host():
                    self._send_json(403, {"error": "FORBIDDEN_HOST_HEADER"})
                    return
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    self._send_bytes(200, "text/html; charset=utf-8", CONSOLE_HTML.encode("utf-8"))
                    return
                if path == "/api/health":
                    self._send_json(200, {"status": "HEALTHY", "service": "alphaloop-v2.2", "ts_ms": utc_now_ms()})
                    return
                if path == "/api/plugins":
                    self._send_json(200, {"plugins": ctx.registry.list_plugins()})
                    return
                if path == "/api/strategies":
                    self._send_json(200, {"strategies": ctx.config_store.list_strategies()})
                    return
                if path == "/api/dq/reports":
                    self._send_json(200, {"reports": ctx.store.list_dq_certificates(limit=50)})
                    return
                if path == "/api/micro-runs":
                    self._send_json(200, {"runs": ctx.micro_store.list_runs(limit=50)})
                    return
                if path == "/api/fleet/audit":
                    self._send_json(200, ctx.read_fleet_snapshot())
                    return
                if path == "/api/portfolio":
                    self._send_json(200, ctx.read_portfolio_snapshot())
                    return
                if path == "/api/shadow/parity":
                    self._send_json(200, ctx.read_shadow_parity_snapshot())
                    return
                if path == "/api/execution/plans":
                    self._send_json(200, ctx.exec_bridge.summary())
                    return
                if path == "/api/trials":
                    self._send_json(200, {"trials": ctx.ledger.recent_trials(limit=50)})
                    return
                if path == "/api/cycles":
                    self._send_json(200, {"cycles": ctx.orchestrator.recent_cycles(limit=25)})
                    return
                if path == "/api/models":
                    self._send_json(
                        200,
                        {
                            "routes": ctx.router.routing_matrix(),
                            "round_spent_usd": round(ctx.router.round_spent_usd, 6),
                            "spent_usd": round(ctx.router.spent_usd, 6),
                        },
                    )
                    return
                if path == "/api/market/summary":
                    self._send_json(200, ctx.store.summary_stats())
                    return
                self._send_json(404, {"error": "NOT_FOUND", "path": path})
            except Exception as exc:
                logger.exception("Unhandled GET error on %s", self.path)
                self._send_json(500, {"error": "INTERNAL_SERVER_ERROR", "detail": type(exc).__name__})

        def do_POST(self) -> None:  # noqa: N802
            try:
                if not self._check_loopback_host():
                    self._send_json(403, {"error": "FORBIDDEN_HOST_HEADER"})
                    return
                path = urlparse(self.path).path
                # CSRF Defense (Red-Team #11): require JSON Content-Type and loopback Origin if sent
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if ctype != "application/json":
                    self._send_json(415, {"error": "UNSUPPORTED_MEDIA_TYPE_REQUIRE_APPLICATION_JSON"})
                    return
                origin = (self.headers.get("Origin") or "").strip()
                if origin and not LOOPBACK_ORIGIN_RE.fullmatch(origin):
                    self._send_json(403, {"error": "FORBIDDEN_ORIGIN"})
                    return

                body, err_code, err_msg = self._read_body()
                if err_code is not None or body is None:
                    self._send_json(err_code or 400, {"error": err_msg})
                    return
                try:
                    payload = json.loads(body.decode("utf-8")) if body else {}
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self._send_json(400, {"error": f"INVALID_JSON: {exc}"})
                    return
                if not isinstance(payload, dict):
                    self._send_json(400, {"error": "JSON_OBJECT_REQUIRED"})
                    return

                if path == "/api/agent/run-cycle":
                    sid = str(payload.get("strategy_id", "")).strip()
                    next_ver = str(payload.get("next_version", f"v-{utc_now_ms()}")).strip()
                    if not next_ver or len(next_ver) > 64 or any(ord(ch) < 32 for ch in next_ver):
                        self._send_json(400, {"error": "INVALID_NEXT_VERSION"})
                        return
                    try:
                        cfg = ctx.config_store.read_active(sid)
                    except ValueError as exc:
                        self._send_json(400, {"error": str(exc)})
                        return
                    if cfg is None:
                        self._send_json(404, {"error": f"STRATEGY_NOT_FOUND: {sid}"})
                        return
                    rec = ctx.orchestrator.run_cycle(cfg, next_version=next_ver)
                    self._send_json(200, rec)
                    return

                if path == "/api/agent/run-sprint":
                    sid = str(payload.get("strategy_id", "")).strip()
                    raw_rounds = payload.get("max_rounds", 3)
                    if isinstance(raw_rounds, bool) or not isinstance(raw_rounds, int):
                        self._send_json(400, {"error": "INVALID_MAX_ROUNDS"})
                        return
                    max_rounds = max(1, min(raw_rounds, 5))
                    try:
                        cfg = ctx.config_store.read_active(sid)
                    except ValueError as exc:
                        self._send_json(400, {"error": str(exc)})
                        return
                    if cfg is None:
                        self._send_json(404, {"error": f"STRATEGY_NOT_FOUND: {sid}"})
                        return
                    sprint = ctx.orchestrator.run_multi_round_sprint(cfg, max_rounds=max_rounds)
                    self._send_json(200, sprint)
                    return

                if path == "/api/micro-runs/evaluate":
                    sid = str(payload.get("strategy_id", "")).strip()
                    raw_h = payload.get("target_hours", 24.0)
                    if isinstance(raw_h, bool) or not isinstance(raw_h, (int, float)) or not math.isfinite(float(raw_h)) or float(raw_h) <= 0.0:
                        self._send_json(400, {"error": "INVALID_TARGET_HOURS"})
                        return
                    target_h = min(720.0, float(raw_h))
                    try:
                        cfg = ctx.config_store.read_active(sid)
                    except ValueError as exc:
                        self._send_json(400, {"error": str(exc)})
                        return
                    if cfg is None:
                        self._send_json(404, {"error": f"STRATEGY_NOT_FOUND: {sid}"})
                        return
                    rec = ctx.micro_session.evaluate_bbo_window(cfg, target_hours=target_h)
                    self._send_json(200, rec.to_dict())
                    return

                if path == "/api/strategies/rollback":
                    sid = str(payload.get("strategy_id", "")).strip()
                    target_ver = str(payload.get("target_version", "")).strip()
                    reason = str(payload.get("reason", "Manual console rollback")).strip()
                    token = payload.get("human_approval_token")
                    token_s = str(token).strip() if token is not None else None
                    try:
                        cfg = ctx.config_store.rollback(
                            sid,
                            target_ver,
                            actor="console-operator",
                            reason=reason,
                            human_approval_token=token_s or None,
                        )
                    except PermissionError as exc:
                        self._send_json(403, {"error": str(exc)})
                        return
                    except ValueError as exc:
                        self._send_json(404, {"error": str(exc)})
                        return
                    self._send_json(200, {"rolled_back_to": cfg.version, "strategy_id": cfg.strategy_id})
                    return

                self._send_json(404, {"error": "NOT_FOUND", "path": path})
            except Exception as exc:
                logger.exception("Unhandled POST error on %s", self.path)
                self._send_json(500, {"error": "INTERNAL_SERVER_ERROR", "detail": type(exc).__name__})

    return AlphaLoopHandler


def create_server(data_dir: str | Path, host: str = "127.0.0.1", port: int = 8796) -> tuple[ThreadingHTTPServer, ConsoleAppContext]:
    if host not in LOOPBACK_HOSTS and os.environ.get("ALPHALOOP_ALLOW_NON_LOOPBACK") != "1":
        raise PermissionError(
            f"Refusing non-loopback bind host={host!r}; set ALPHALOOP_ALLOW_NON_LOOPBACK=1 to override explicitly"
        )
    ctx = ConsoleAppContext(data_dir)
    srv = ThreadingHTTPServer((host, port), _make_handler(ctx))
    return srv, ctx
