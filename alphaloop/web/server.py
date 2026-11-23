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
from urllib.parse import parse_qs, urlparse

from alphaloop.agent_loop.model_router import MultiModelRouter
from alphaloop.agent_loop.orchestrator import AgentEvolutionOrchestrator
from alphaloop.core.config import HotConfigStore, validate_strategy_id
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

CONSOLE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AlphaLoop v2.2 — Strategy Factory, Portfolio, DQ Gate &amp; Shadow Execution Console</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%2311161f'/%3E%3Cpath d='M8 23L16 8l8 15h-3.6L16 14.5 11.6 23H8z' fill='%2338bdf8'/%3E%3C/svg%3E"/>
<style>
/* ==========================================================================
   INSTITUTIONAL DARK QUANT TERMINAL — CANONICAL DESIGN TOKENS v1.0
   AlphaLoop v2.2 Strategy R&D Console (Zero CDN · WCAG 2.1 AA · STD-37)
   ========================================================================== */
:root {
  color-scheme: dark;

  /* 1. 4-Level Obsidian / Charcoal Surface Elevation Ladder */
  --bg-base:     #0b0f14;
  --bg-surface:  #11161f;
  --bg-elevated: #161d28;
  --bg-hover:    #1e2736;
  --bg-scrim:    rgba(7, 10, 15, 0.72);

  /* 2. Surface Depth & Specular Hairline */
  --surface-canvas-bg:
    radial-gradient(1200px 600px at 50% -10%, rgba(56, 189, 248, 0.035), transparent 65%),
    var(--bg-base);
  --surface-sheen:        linear-gradient(180deg, rgba(255, 255, 255, 0.024) 0%, rgba(255, 255, 255, 0) 42%);
  --surface-hairline-top: inset 0 1px 0 rgba(255, 255, 255, 0.04);
  --shadow-panel:         0 1px 2px rgba(0, 0, 0, 0.35), var(--surface-hairline-top);
  --shadow-popover:       0 8px 24px rgba(0, 0, 0, 0.48), var(--surface-hairline-top);
  --shadow-dialog:        0 20px 56px rgba(0, 0, 0, 0.65), var(--surface-hairline-top);

  /* 3. Neutral Hairline Borders & Focus Ring */
  --border-subtle:   rgba(255, 255, 255, 0.08);
  --border-default:  rgba(255, 255, 255, 0.12);
  --border-strong:   rgba(255, 255, 255, 0.18);
  --border-focus:    #38bdf8;
  --focus-ring:      0 0 0 2px rgba(56, 189, 248, 0.45);

  /* 4. Text Ramp (WCAG 2.1 AA Verified L0-L3) */
  --text-strong:     #f1f5f9;
  --text-primary:    #e6edf3;
  --text-secondary:  #94a3b8;
  --text-muted:      #7d8fa6;
  --text-disabled:   #475569;
  --text-on-accent:  #0b0f14;
  --text-on-solid:   #ffffff;

  /* 5. Interactive Accent & Semantic Palette */
  --accent-primary:         #38bdf8;
  --accent-primary-hover:   #7dd3fc;
  --accent-strong:          #0369a1;
  --accent-strong-hover:    #075985;
  --accent-hover:           #0284c7;
  --accent-tint:            rgba(56, 189, 248, 0.12);
  --accent-tint-strong:     rgba(56, 189, 248, 0.20);

  --color-profit:           #22c55e;
  --color-profit-text:      #4ade80;
  --color-profit-tint:      rgba(34, 197, 94, 0.12);

  --color-loss:             #ef4444;
  --color-loss-text:        #f87171;
  --color-loss-solid:       #dc2626;
  --color-loss-solid-hover: #b91c1c;
  --color-loss-tint:        rgba(239, 68, 68, 0.12);

  --color-warn:             #f59e0b;
  --color-warn-text:        #fbbf24;
  --color-warn-tint:        rgba(245, 158, 11, 0.12);

  --color-info:             #60a5fa;
  --color-info-tint:        rgba(96, 165, 250, 0.12);
  --color-teal:             #2dd4bf;
  --color-teal-tint:        rgba(45, 212, 191, 0.12);
  --color-orange:           #fb923c;
  --color-orange-tint:      rgba(251, 146, 60, 0.12);

  /* 6. Market-Direction Aliases */
  --market-up:              var(--color-profit);
  --market-up-text:         var(--color-profit-text);
  --market-up-tint:         var(--color-profit-tint);
  --market-down:            var(--color-loss);
  --market-down-text:       var(--color-loss-text);
  --market-down-tint:       var(--color-loss-tint);

  /* 7. Typography */
  --font-sans: "PingFang SC", "Hiragino Sans GB", "Noto Sans SC", "Microsoft YaHei", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", "SF Mono", "Cascadia Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;

  --text-2xs: 0.6875rem;
  --text-xs:  0.75rem;
  --text-sm:  0.8125rem;
  --text-md:  0.875rem;
  --text-lg:  1rem;
  --text-xl:  1.25rem;
  --text-2xl: 1.5rem;

  --tracking-tight:  -0.025em;
  --tracking-normal:  0em;
  --tracking-label:   0.05em;

  --leading-tight:   1.15;
  --leading-snug:    1.35;
  --leading-normal:  1.5;

  /* 8. Spacing Scale (4px grid) */
  --space-1:  4px;
  --space-2:  8px;
  --space-3:  12px;
  --space-4:  16px;
  --space-5:  20px;
  --space-6:  24px;
  --space-8:  32px;

  /* 9. Console Radius Profile */
  --radius-none:    0px;
  --radius-xs:      2px;
  --radius-sm:      4px;
  --radius-md:      6px;
  --radius-lg:      8px;
  --radius-pill:    9999px;
  --radius-panel:   var(--radius-lg);
  --radius-control: var(--radius-md);
  --radius-badge:   var(--radius-sm);

  /* 10. Chrome, Z-Index & Motion */
  --topbar-h: 48px;
  --subbar-h: 38px;
  --z-sticky:  10;
  --z-topbar:  20;
  --z-rail:    30;
  --z-popover: 40;
  --z-drawer:  50;
  --z-modal:   60;
  --z-toast:   80;

  --ease-ui:  cubic-bezier(0.2, 0, 0.1, 1);
  --dur-fast: 120ms;
  --dur-base: 180ms;
}

*, *::before, *::after { box-sizing: border-box; }

html {
  background: var(--bg-base);
}

body {
  margin: 0;
  min-height: 100vh;
  background: var(--surface-canvas-bg);
  background-repeat: no-repeat;
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: var(--text-sm);
  line-height: var(--leading-snug);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  overflow-x: hidden;
}

h1, h2 {
  letter-spacing: var(--tracking-tight);
  line-height: var(--leading-tight);
}

code {
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  color: var(--accent-primary-hover);
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-xs);
  padding: 1px 5px;
}

.mono, .num, [data-numeric="true"] {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

.kpi-value {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  letter-spacing: var(--tracking-tight);
  line-height: var(--leading-tight);
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

:where(a, button, input, select, textarea, summary, [tabindex]):focus-visible {
  outline: 2px solid var(--border-focus);
  outline-offset: 2px;
  box-shadow: var(--focus-ring);
}

@keyframes ui-spin {
  to { transform: rotate(360deg); }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* ── Sticky Topbar (48px) ── */
.ui-topbar {
  position: sticky;
  top: 0;
  z-index: var(--z-topbar);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  min-height: var(--topbar-h);
  padding: 0 var(--space-4);
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-subtle);
  box-shadow: var(--shadow-panel);
}
.ui-topbar__brand {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  min-width: 0;
}
.ui-topbar__title {
  margin: 0;
  font-size: var(--text-md);
  font-weight: 700;
  color: var(--text-strong);
  letter-spacing: var(--tracking-tight);
  white-space: nowrap;
}
.ui-topbar__sub {
  font-size: var(--text-2xs);
  color: var(--text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ui-topbar__meta {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: nowrap;
  min-width: 0;
  overflow-x: auto;
}
.ui-sync-clock {
  font-size: var(--text-2xs);
  color: var(--text-secondary);
  white-space: nowrap;
}

/* ── Segmented WAI-ARIA Tab Bar ── */
.ui-tablist {
  position: sticky;
  top: var(--topbar-h);
  z-index: calc(var(--z-topbar) - 1);
  display: flex;
  align-items: center;
  gap: var(--space-1);
  padding: 0 var(--space-4);
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-subtle);
  overflow-x: auto;
}
.ui-tab {
  appearance: none;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: transparent;
  border: 0;
  border-bottom: 2px solid transparent;
  color: var(--text-secondary);
  padding: 10px 12px;
  font-family: var(--font-sans);
  font-size: var(--text-xs);
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
  transition: color var(--dur-fast) var(--ease-ui), border-color var(--dur-fast) var(--ease-ui);
}
.ui-tab:hover { color: var(--text-primary); }
.ui-tab[aria-selected="true"] {
  color: var(--accent-primary);
  border-bottom-color: var(--accent-primary);
}
.ui-tab__badge {
  display: inline-flex;
  align-items: center;
  padding: 1px 5px;
  border-radius: var(--radius-badge);
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 700;
  color: var(--color-loss-text);
  background: var(--color-loss-tint);
  border: 1px solid rgba(239, 68, 68, 0.34);
}
.ui-tab__badge[hidden] { display: none; }

/* ── Main Content Shell ── */
.console-main {
  max-width: 1680px;
  margin: 0 auto;
  padding: var(--space-4);
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}
.tab-panel {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
  min-width: 0;
}
.tab-panel[hidden] { display: none; }

/* ── 8-Card KPI Strip (4x2 Desktop / 2x4 Mobile) ── */
.ui-kpi-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--space-2);
}
.ui-kpi {
  background: var(--surface-sheen), var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-panel);
  padding: 10px 12px;
  box-shadow: var(--shadow-panel);
  min-width: 0;
}
.ui-kpi__top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
}
.ui-kpi__label {
  font-size: var(--text-2xs);
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: var(--tracking-label);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ui-kpi__value {
  margin-top: 6px;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-size: var(--text-xl);
  font-weight: 700;
  color: var(--text-strong);
  letter-spacing: var(--tracking-tight);
  line-height: var(--leading-tight);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ui-kpi__sub {
  margin-top: 4px;
  font-size: var(--text-2xs);
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ── Primary Panel & Sticky-Header Data Table ── */
.ui-panel {
  background: var(--surface-sheen), var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-panel);
  box-shadow: var(--shadow-panel);
  min-width: 0;
}
.ui-panel__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: var(--space-2);
  padding: 10px 14px;
  border-bottom: 1px solid var(--border-subtle);
  border-radius: var(--radius-panel) var(--radius-panel) 0 0;
}
.ui-panel__title {
  margin: 0;
  font-size: var(--text-md);
  font-weight: 600;
  color: var(--text-strong);
}
.ui-panel__tools {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}
.ui-panel__body {
  padding: 0;
  min-width: 0;
}
.ui-panel__pad {
  padding: 12px 14px;
}

/* Hairline summary strip inside L1 panel (avoids L3 nested cards) */
.ui-meta-bar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px 16px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border-subtle);
  font-size: var(--text-xs);
  color: var(--text-secondary);
}
.ui-meta-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  white-space: nowrap;
}
.ui-meta-item__k {
  color: var(--text-muted);
  font-size: var(--text-2xs);
  text-transform: uppercase;
  letter-spacing: var(--tracking-label);
}
.ui-meta-item__v {
  color: var(--text-strong);
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}

.ui-table-wrap {
  overflow: auto;
  max-height: 540px;
  border-radius: 0 0 var(--radius-panel) var(--radius-panel);
}
.ui-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: var(--text-xs);
}
.ui-table thead th {
  position: sticky;
  top: 0;
  z-index: var(--z-sticky);
  background: var(--bg-elevated);
  color: var(--text-secondary);
  font-size: var(--text-2xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: var(--tracking-label);
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border-default);
  white-space: nowrap;
}
.ui-table td {
  padding: 7px 10px;
  border-bottom: 1px solid var(--border-subtle);
  color: var(--text-primary);
  white-space: nowrap;
  vertical-align: middle;
}
.ui-table td.wrap-cell {
  white-space: normal;
  min-width: 180px;
  max-width: 420px;
}
.ui-table tbody tr:last-child td { border-bottom: 0; }
.ui-table th.num, .ui-table td.num {
  text-align: right;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}
.ui-table tbody tr:hover td { background: var(--bg-hover); }
.ui-table tbody tr.detail-row:hover td { background: var(--bg-elevated); }
.ui-table tbody tr.detail-row td {
  background: var(--bg-elevated);
  white-space: normal;
  padding: 12px 14px;
}

/* ── Semantic Value Colors & Chips ── */
.val-pos { color: var(--color-profit-text); font-weight: 600; }
.val-neg { color: var(--color-loss-text); font-weight: 600; }
.val-warn { color: var(--color-warn-text); font-weight: 600; }
.val-info { color: var(--accent-primary); font-weight: 600; }
.val-muted { color: var(--text-muted); }

.ui-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 7px;
  border-radius: var(--radius-badge);
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  border: 1px solid var(--border-subtle);
  background: var(--bg-elevated);
  color: var(--text-secondary);
  white-space: nowrap;
}
.ui-pill--ok   { color: var(--color-profit-text); background: var(--color-profit-tint); border-color: rgba(34, 197, 94, 0.28); }
.ui-pill--bad  { color: var(--color-loss-text); background: var(--color-loss-tint); border-color: rgba(239, 68, 68, 0.34); }
.ui-pill--warn { color: var(--color-warn-text); background: var(--color-warn-tint); border-color: rgba(245, 158, 11, 0.30); }
.ui-pill--info { color: var(--accent-primary); background: var(--accent-tint); border-color: rgba(56, 189, 248, 0.30); }
.ui-pill--teal { color: var(--color-teal); background: var(--color-teal-tint); border-color: rgba(45, 212, 191, 0.30); }

.ui-chip-list {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.ui-chip {
  display: inline-flex;
  align-items: center;
  padding: 1px 6px;
  border-radius: var(--radius-xs);
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  font-size: var(--text-2xs);
  color: var(--text-primary);
  white-space: nowrap;
}

/* ── Inline Meters & Budget Progress Bars ── */
.ui-meter {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-width: 112px;
}
.ui-meter__track {
  position: relative;
  flex: 1;
  min-width: 54px;
  height: 6px;
  border-radius: var(--radius-pill);
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  overflow: hidden;
}
.ui-meter__fill {
  height: 100%;
  border-radius: var(--radius-pill);
  background: var(--accent-primary);
  transition: width var(--dur-base) var(--ease-ui);
}
.ui-meter__fill--ok   { background: var(--color-profit); }
.ui-meter__fill--warn { background: var(--color-warn); }
.ui-meter__fill--bad  { background: var(--color-loss); }
.ui-meter__val {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-size: var(--text-2xs);
  white-space: nowrap;
}

.budget-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-3);
  padding: 12px 14px;
  border-bottom: 1px solid var(--border-subtle);
}
.budget-item {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.budget-item__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: var(--text-xs);
}
.budget-item__label {
  color: var(--text-secondary);
  font-weight: 600;
}
.budget-item__val {
  color: var(--text-strong);
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

/* ── Sticky Kill-Switch Alert Strip (1px border, zero thick border) ── */
.kill-strip {
  position: sticky;
  top: calc(var(--topbar-h) + var(--subbar-h));
  z-index: calc(var(--z-sticky) + 1);
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px 14px;
  margin: 12px 14px 0;
  padding: 10px 12px;
  border-radius: var(--radius-control);
  background: var(--color-loss-tint);
  border: 1px solid rgba(239, 68, 68, 0.40);
  color: var(--color-loss-text);
  font-size: var(--text-xs);
}
.kill-strip__dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--color-loss);
  flex-shrink: 0;
}

/* ── Cycle Expanded Diff Detail Grid ── */
.cycle-detail-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-3);
}
.cycle-detail-block {
  border-top: 1px solid var(--border-subtle);
  padding-top: 8px;
}
.cycle-detail-block h3 {
  margin: 0 0 6px;
  font-size: var(--text-2xs);
  text-transform: uppercase;
  letter-spacing: var(--tracking-label);
  color: var(--text-muted);
}
.cycle-detail-block p {
  margin: 0 0 6px;
  font-size: var(--text-xs);
  color: var(--text-primary);
}
.cycle-detail-kv {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  font-size: var(--text-xs);
  color: var(--text-secondary);
}

/* ── Buttons, Inputs & Filter Pills ── */
.ui-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 28px;
  padding: 4px 10px;
  border-radius: var(--radius-control);
  border: 1px solid var(--border-default);
  background: var(--bg-elevated);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: var(--text-xs);
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
  transition: background var(--dur-fast) var(--ease-ui), border-color var(--dur-fast) var(--ease-ui), color var(--dur-fast) var(--ease-ui);
}
.ui-btn:hover:not(:disabled) { background: var(--bg-hover); border-color: var(--border-strong); }
.ui-btn:disabled { opacity: 0.45; cursor: not-allowed; }
.ui-btn--primary { background: var(--accent-primary); border-color: transparent; color: var(--text-on-accent); }
.ui-btn--primary:hover:not(:disabled) { background: var(--accent-primary-hover); border-color: transparent; }
.ui-btn--solid { background: var(--accent-strong); border-color: transparent; color: var(--text-on-solid); }
.ui-btn--solid:hover:not(:disabled) { background: var(--accent-strong-hover); border-color: transparent; }
.ui-btn--profit { background: var(--color-profit); border-color: transparent; color: var(--text-on-accent); }
.ui-btn--profit:hover:not(:disabled) { background: var(--color-profit-text); border-color: transparent; }
.ui-btn--danger { background: var(--color-loss-solid); border-color: transparent; color: var(--text-on-solid); }
.ui-btn--danger:hover:not(:disabled) { background: var(--color-loss-solid-hover); border-color: transparent; }
.ui-btn--warn { background: var(--color-warn); border-color: transparent; color: var(--text-on-accent); }
.ui-btn--warn:hover:not(:disabled) { background: var(--color-warn-text); border-color: transparent; }
.ui-btn--ghost { background: transparent; border-color: var(--border-subtle); color: var(--text-secondary); }
.ui-btn--ghost:hover:not(:disabled) { background: var(--bg-hover); border-color: var(--border-default); color: var(--text-primary); }
.ui-btn--sm { min-height: 24px; padding: 2px 8px; font-size: var(--text-2xs); }
.ui-btn[aria-pressed="true"] {
  background: var(--accent-tint);
  border-color: rgba(56, 189, 248, 0.42);
  color: var(--accent-primary);
}

.ui-btn-group {
  display: inline-flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
}

.ui-spinner {
  display: none;
  width: 12px;
  height: 12px;
  border: 2px solid currentColor;
  border-top-color: transparent;
  border-radius: 50%;
  animation: ui-spin 0.7s linear infinite;
  flex-shrink: 0;
}
[data-loading="true"] .ui-spinner { display: inline-block; }

.ui-select, .ui-input {
  min-height: 28px;
  padding: 4px 8px;
  border-radius: var(--radius-control);
  border: 1px solid var(--border-default);
  background: var(--bg-elevated);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: var(--text-xs);
}
.ui-input { width: 100%; }
.ui-select:hover, .ui-input:hover { border-color: var(--border-strong); }

/* ── STD-37 Honest-UI State Badges & Empty Panels ── */
.ui-state-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 8px;
  border-radius: var(--radius-badge);
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.03em;
  border: 1px solid var(--border-subtle);
  white-space: nowrap;
}
.ui-state-badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
.ui-state-badge[data-state="live"]             { color: #4ade80; background: rgba(34, 197, 94, 0.12);  border-color: rgba(34, 197, 94, 0.28); }
.ui-state-badge[data-state="stale"]            { color: #fbbf24; background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.30); }
.ui-state-badge[data-state="no-channel"]       { color: #94a3b8; background: rgba(148, 163, 184, 0.10); border-color: rgba(148, 163, 184, 0.22); }
.ui-state-badge[data-state="no-channel"]::before { background: transparent; border: 1.5px solid #94a3b8; }
.ui-state-badge[data-state="upstream-error"]   { color: #f87171; background: rgba(239, 68, 68, 0.12);  border-color: rgba(239, 68, 68, 0.34); }
.ui-state-badge[data-state="sim"]              { color: #fbbf24; background: rgba(245, 158, 11, 0.14); border-color: rgba(245, 158, 11, 0.38); }
.ui-state-badge[data-state="paused"]           { color: #cbd5e1; background: rgba(148, 163, 184, 0.14); border-color: rgba(148, 163, 184, 0.30); }
.ui-state-badge[data-state="paused"][data-severity="critical"] {
  color: var(--color-loss-text);
  background: var(--color-loss-tint);
  border-color: rgba(239, 68, 68, 0.34);
}
.ui-state-badge[data-state="observation-only"] { color: #38bdf8; background: rgba(56, 189, 248, 0.12); border-color: rgba(56, 189, 248, 0.30); }

.ui-kpi[data-state="stale"] .ui-kpi__value { color: var(--text-muted); }
.ui-kpi[data-state="stale"] .ui-state-badge[data-state="stale"] { opacity: 1; color: var(--color-warn-text); }

.ui-honest-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin: 14px;
  padding: 20px 16px;
  border-radius: var(--radius-control);
  background: var(--bg-surface);
  border: 1px dashed rgba(148, 163, 184, 0.26);
  text-align: center;
  color: var(--text-secondary);
  font-size: var(--text-xs);
}
.ui-honest-empty[data-state="upstream-error"] {
  border-color: rgba(239, 68, 68, 0.40);
  background: rgba(239, 68, 68, 0.05);
  color: var(--text-secondary);
}

/* ── Non-Blocking Toast Stack & Native <dialog> ── */
.ui-toast-stack {
  position: fixed;
  right: 16px;
  bottom: 16px;
  z-index: var(--z-toast);
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-width: 400px;
  pointer-events: none;
}
.ui-toast {
  pointer-events: auto;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-control);
  padding: 10px 12px;
  box-shadow: var(--shadow-popover);
  color: var(--text-primary);
  font-size: var(--text-xs);
}
.ui-toast::before {
  content: "";
  width: 6px;
  height: 6px;
  margin-top: 5px;
  border-radius: 50%;
  flex-shrink: 0;
  background: var(--accent-primary);
}
.ui-toast[data-level="success"]::before { background: var(--color-profit); }
.ui-toast[data-level="warn"]::before    { background: var(--color-warn); }
.ui-toast[data-level="error"]::before   { background: var(--color-loss); }
.ui-toast__body  { flex: 1; min-width: 0; }
.ui-toast__title { font-weight: 600; color: var(--text-strong); margin-bottom: 2px; }
.ui-toast__msg   { color: var(--text-secondary); word-break: break-word; }

dialog.ui-dialog {
  background: var(--bg-surface);
  color: var(--text-primary);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-panel);
  padding: 0;
  width: min(560px, 94vw);
  max-height: 90vh;
  box-shadow: var(--shadow-dialog);
}
dialog.ui-dialog::backdrop { background: var(--bg-scrim); }
.ui-dialog__inner {
  display: flex;
  flex-direction: column;
  max-height: 90vh;
}
.ui-dialog__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-2);
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-subtle);
}
.ui-dialog__body {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  padding: 14px 16px;
  font-size: var(--text-sm);
  color: var(--text-secondary);
  overflow-y: auto;
}
.ui-dialog__field {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.ui-dialog__label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: var(--text-xs);
  font-weight: 600;
  color: var(--text-primary);
}
.ui-dialog__hint {
  font-size: var(--text-2xs);
  color: var(--text-muted);
}
.ui-dialog__error {
  padding: 8px 10px;
  border-radius: var(--radius-control);
  background: var(--color-loss-tint);
  border: 1px solid rgba(239, 68, 68, 0.34);
  color: var(--color-loss-text);
  font-size: var(--text-xs);
  word-break: break-word;
}
.ui-dialog__error[hidden] { display: none; }
.ui-dialog__result {
  border-top: 1px solid var(--border-subtle);
  padding-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.ui-dialog__result[hidden] { display: none; }
.ui-dialog__actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: var(--space-2);
  padding: 12px 16px;
  border-top: 1px solid var(--border-subtle);
}

/* ── Canonical Responsive Breakpoints (1280px / 1024px / 768px) ── */
@media (max-width: 1280px) {
  .ui-topbar { padding: 0 var(--space-3); gap: var(--space-2); }
  .ui-topbar__sub { display: none; }
  .ui-tablist { padding: 0 var(--space-3); }
  .console-main { padding: var(--space-3); }
}
@media (max-width: 1024px) {
  .budget-grid { grid-template-columns: 1fr; }
  .cycle-detail-grid { grid-template-columns: 1fr; }
  .ui-table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .ui-table--sticky-first th:first-child,
  .ui-table--sticky-first td:first-child {
    position: sticky;
    left: 0;
    z-index: 2;
    background: var(--bg-surface);
    border-right: 1px solid var(--border-subtle);
  }
  .ui-table--sticky-first thead th:first-child {
    z-index: calc(var(--z-sticky) + 2);
    background: var(--bg-elevated);
  }
  .ui-table--sticky-first tbody tr:hover td:first-child {
    background: var(--bg-hover);
  }
}
@media (max-width: 768px) {
  .ui-topbar { flex-wrap: wrap; padding: 6px var(--space-3); }
  .ui-topbar__meta { width: 100%; justify-content: flex-end; }
  .ui-tablist { top: 0; position: static; }
  .kill-strip { top: var(--topbar-h); }
  .ui-kpi-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ui-btn, .ui-tab, .ui-select { min-height: 36px; }
  .ui-toast-stack { left: 12px; right: 12px; bottom: 12px; max-width: none; }
}
</style>
</head>
<body>
<header class="ui-topbar">
  <div class="ui-topbar__brand">
    <h1 class="ui-topbar__title">ALPHALOOP v2.2</h1>
    <span id="modePill" class="ui-state-badge" data-state="sim" data-live="false">SHADOW_DRY_RUN</span>
    <span class="ui-topbar__sub">Strategy Factory · Portfolio · DQ Gate · Shadow Execution</span>
  </div>
  <div class="ui-topbar__meta">
    <span id="healthBadge" class="ui-state-badge" data-state="no-channel" data-live="false">CONNECTING</span>
    <span id="lastSyncClock" class="ui-sync-clock mono" title="Last successful sync UTC">UTC —</span>
    <label class="sr-only" for="autoRefreshSelect">Auto refresh interval</label>
    <select id="autoRefreshSelect" class="ui-select mono" data-change-action="auto-refresh" aria-label="Auto refresh interval">
      <option value="0">OFF</option>
      <option value="10000">10s</option>
      <option value="30000" selected>30s</option>
      <option value="60000">60s</option>
    </select>
    <button id="refreshBtn" type="button" class="ui-btn ui-btn--ghost" data-action="refresh">
      <span class="ui-spinner" aria-hidden="true"></span>
      <span class="ui-btn__label">Refresh</span>
    </button>
  </div>
</header>

<nav class="ui-tablist" role="tablist" aria-label="AlphaLoop workspaces">
  <button id="tab-overview" type="button" role="tab" class="ui-tab" data-action="tab" data-tab="overview" aria-controls="panel-overview" aria-selected="true" tabindex="0">Overview &amp; Parity</button>
  <button id="tab-execution" type="button" role="tab" class="ui-tab" data-action="tab" data-tab="execution" aria-controls="panel-execution" aria-selected="false" tabindex="-1">Execution &amp; Audit <span id="tabExecBadge" class="ui-tab__badge" hidden></span></button>
  <button id="tab-factory" type="button" role="tab" class="ui-tab" data-action="tab" data-tab="factory" aria-controls="panel-factory" aria-selected="false" tabindex="-1">Strategy Factory &amp; DQ</button>
  <button id="tab-evolution" type="button" role="tab" class="ui-tab" data-action="tab" data-tab="evolution" aria-controls="panel-evolution" aria-selected="false" tabindex="-1">Evolution &amp; Models</button>
  <button id="tab-fleet" type="button" role="tab" class="ui-tab" data-action="tab" data-tab="fleet" aria-controls="panel-fleet" aria-selected="false" tabindex="-1">5-Node Fleet</button>
</nav>

<main class="console-main">
  <section aria-label="System KPI overview">
    <div id="kpiGrid" class="ui-kpi-strip kpi-8"></div>
  </section>

  <!-- Tab 1: Overview & Parity -->
  <section id="panel-overview" class="tab-panel" role="tabpanel" aria-labelledby="tab-overview">
    <section class="ui-panel" aria-labelledby="heading-parity">
      <div class="ui-panel__head">
        <h2 id="heading-parity" class="ui-panel__title">Live Shadow Parity &amp; Drift Guard (SHADOW_DRY_RUN vs Backtest)</h2>
        <span class="ui-pill">Threshold: |Drift| &le; 150bp · Spread &le; 1.5x</span>
      </div>
      <div id="shadowParityBox" class="ui-panel__body"></div>
    </section>

    <section class="ui-panel" aria-labelledby="heading-portfolio">
      <div class="ui-panel__head">
        <h2 id="heading-portfolio" class="ui-panel__title">Multi-Pillar Portfolio Scorecard &amp; Capacity Scaling (PortfolioCombiner)</h2>
      </div>
      <div id="portfolioBox" class="ui-panel__body"></div>
    </section>

    <section class="ui-panel" aria-labelledby="heading-market">
      <div class="ui-panel__head">
        <h2 id="heading-market" class="ui-panel__title">Market Data Coverage (Funding · 1h Candles · Live BBO)</h2>
      </div>
      <div id="marketSummaryBox" class="ui-panel__body"></div>
    </section>
  </section>

  <!-- Tab 2: Execution & Audit -->
  <section id="panel-execution" class="tab-panel" role="tabpanel" aria-labelledby="tab-execution" hidden>
    <section class="ui-panel" aria-labelledby="heading-exec">
      <div class="ui-panel__head">
        <h2 id="heading-exec" class="ui-panel__title">Shadow Execution Bridge (SHADOW_DRY_RUN · Hash-Chained Audit · Kill Switch)</h2>
        <span class="ui-pill">LIVE_EXECUTION_ENABLED = false</span>
      </div>
      <div id="execBox" class="ui-panel__body"></div>
    </section>
  </section>

  <!-- Tab 3: Strategy Factory & DQ -->
  <section id="panel-factory" class="tab-panel" role="tabpanel" aria-labelledby="tab-factory" hidden>
    <section class="ui-panel" aria-labelledby="heading-strategies">
      <div class="ui-panel__head">
        <h2 id="heading-strategies" class="ui-panel__title">Active Hot-Reloadable Strategies (Atomic rename(2) Bus)</h2>
        <span class="ui-pill">Actions: Evolve · Sprint · MicroRun · Rollback</span>
      </div>
      <div id="strategiesBox" class="ui-panel__body"></div>
    </section>

    <section class="ui-panel" aria-labelledby="heading-plugins">
      <div class="ui-panel__head">
        <h2 id="heading-plugins" class="ui-panel__title">Strategy Factory v2.2 Plugins (Pluggable Alpha Registry)</h2>
      </div>
      <div id="pluginsBox" class="ui-panel__body"></div>
    </section>

    <section class="ui-panel" aria-labelledby="heading-dq">
      <div class="ui-panel__head">
        <h2 id="heading-dq" class="ui-panel__title">Data Quality Certification Ledger (DQ-01 .. DQ-08)</h2>
        <div id="dqFilterBar" class="ui-panel__tools"></div>
      </div>
      <div id="dqBox" class="ui-panel__body"></div>
    </section>

    <section class="ui-panel" aria-labelledby="heading-micro">
      <div class="ui-panel__head">
        <h2 id="heading-micro" class="ui-panel__title">Fast Micro-Run Validation &amp; Dual-Layer Early-Kill Guard</h2>
      </div>
      <div id="microBox" class="ui-panel__body"></div>
    </section>
  </section>

  <!-- Tab 4: Evolution & Models -->
  <section id="panel-evolution" class="tab-panel" role="tabpanel" aria-labelledby="tab-evolution" hidden>
    <section class="ui-panel" aria-labelledby="heading-cycles">
      <div class="ui-panel__head">
        <h2 id="heading-cycles" class="ui-panel__title">Closed-Loop Agent Evolution &amp; Version Diff (Diagnose &rarr; Propose &rarr; DQ + 6-Gate &rarr; Red-Team)</h2>
      </div>
      <div id="cyclesBox" class="ui-panel__body"></div>
    </section>

    <section class="ui-panel" aria-labelledby="heading-models">
      <div class="ui-panel__head">
        <h2 id="heading-models" class="ui-panel__title">Multi-Model Consequence Router &amp; Persistent LLM Budget</h2>
      </div>
      <div id="modelsBox" class="ui-panel__body"></div>
    </section>

    <section class="ui-panel" aria-labelledby="heading-trials">
      <div class="ui-panel__head">
        <h2 id="heading-trials" class="ui-panel__title">Append-Only Multiple-Testing Trial Ledger (G6)</h2>
        <div id="trialsFilterBar" class="ui-panel__tools"></div>
      </div>
      <div id="trialsBox" class="ui-panel__body"></div>
    </section>
  </section>

  <!-- Tab 5: 5-Node Fleet -->
  <section id="panel-fleet" class="tab-panel" role="tabpanel" aria-labelledby="tab-fleet" hidden>
    <section class="ui-panel" aria-labelledby="heading-fleet">
      <div class="ui-panel__head">
        <h2 id="heading-fleet" class="ui-panel__title">5-Node Prepaid Server Fleet Health (Read-Only SSH Snapshot)</h2>
      </div>
      <div id="fleetBox" class="ui-panel__body"></div>
    </section>
  </section>
</main>

<dialog id="actionDialog" class="ui-dialog" aria-modal="true" aria-labelledby="actionDialogTitle">
  <div class="ui-dialog__inner">
    <div class="ui-dialog__head">
      <h2 id="actionDialogTitle" class="ui-panel__title" data-dlg-title>Action</h2>
      <button type="button" class="ui-btn ui-btn--ghost ui-btn--sm" data-dlg-cancel aria-label="Close dialog">&times;</button>
    </div>
    <div class="ui-dialog__body">
      <div data-dlg-content></div>
      <div class="ui-dialog__error" data-dlg-error role="alert" hidden></div>
      <div class="ui-dialog__result" data-dlg-result hidden></div>
    </div>
    <div class="ui-dialog__actions">
      <button type="button" class="ui-btn ui-btn--ghost" data-dlg-cancel>Cancel</button>
      <button type="button" class="ui-btn ui-btn--primary" data-dlg-submit>
        <span class="ui-spinner" aria-hidden="true"></span>
        <span class="ui-btn__label" data-dlg-submit-label>Execute</span>
      </button>
    </div>
  </div>
</dialog>

<div id="ui-toast-stack" class="ui-toast-stack" role="region" aria-label="System notifications" aria-live="polite"></div>

<script>
(function () {
  'use strict';

  const TABS = ['overview', 'execution', 'factory', 'evolution', 'fleet'];
  const DEFAULT_ALLOWED_AUTO_STATUSES = [
    'DRAFT',
    'BACKTESTED',
    'RED_TEAMED',
    'MICRO_RUN_PROBE_REQUIRED',
    'SHADOW_LIVE',
    'RETIRED',
  ];
  const HUMAN_TOKEN_RE = /^HUMAN_APPROVED:[A-Za-z0-9._-]{4,64}$/;
  const TAB_STORAGE_KEY = 'alphaloop_v22_tab';
  const POLL_STORAGE_KEY = 'alphaloop_v22_poll_ms';

  const state = {
    activeTab: 'overview',
    autoRefreshMs: 30000,
    pollTimer: null,
    refreshSeq: 0,
    dqFilter: 'ALL',
    trialStrategy: 'ALL',
    trialResult: 'ALL',
    expandedCycles: new Set(),
    inFlight: new Set(),
    inFlightAction: Object.create(null),
    last: Object.create(null),
  };

  /* ── L-034 XSS Escape & Safe URL Helpers ── */
  function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    let str;
    try {
      str = String(value);
    } catch {
      return '';
    }
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
  const esc = escapeHtml;

  function safeUrl(raw, { allowRelative = true } = {}) {
    if (typeof raw !== 'string') return null;
    const cleaned = raw.trim();
    if (!cleaned || /[\u0000-\u001F\u007F]/.test(cleaned)) return null;
    if (/^(?:\/\/|\/\\|\\)/.test(cleaned)) return null;
    if (cleaned.includes('\\')) return null;
    if (allowRelative && /^(?:\/(?!\/|\\)|\.\/|\.\.\/|#)/.test(cleaned)) {
      if (typeof window !== 'undefined' && window.location && window.location.origin) {
        try {
          if (new URL(cleaned, window.location.origin).origin !== window.location.origin) return null;
        } catch {
          return null;
        }
      }
      return cleaned;
    }
    if (!/^https?:\/\//i.test(cleaned)) return null;
    try {
      const parsed = new URL(cleaned);
      if (parsed.protocol === 'https:' || parsed.protocol === 'http:') return parsed.href;
    } catch {
      return null;
    }
    return null;
  }

  /* ── Formatting & Numeric Utilities ── */
  function isFiniteNum(v) {
    return typeof v === 'number' && Number.isFinite(v);
  }
  function numOr(v, fallback) {
    const n = typeof v === 'number' ? v : Number(v);
    return Number.isFinite(n) ? n : fallback;
  }
  function fmtNum(v, digits = 2) {
    const n = Number(v);
    if (v === null || v === undefined || v === '' || !Number.isFinite(n)) return '—';
    return n.toFixed(digits);
  }
  function fmtSigned(v, digits = 2, suffix = '') {
    const n = Number(v);
    if (v === null || v === undefined || v === '' || !Number.isFinite(n)) return '—';
    return (n > 0 ? '+' : '') + n.toFixed(digits) + suffix;
  }
  function fmtUsd(v, digits = 2, signed = false) {
    const n = Number(v);
    if (v === null || v === undefined || v === '' || !Number.isFinite(n)) return '—';
    const abs = Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
    if (n < 0) return '-$' + abs;
    return (signed && n > 0 ? '+$' : '$') + abs;
  }
  function fmtUtc(tsMs) {
    const n = Number(tsMs);
    if (!Number.isFinite(n) || n <= 0) return '—';
    try {
      return new Date(n).toISOString().replace('T', ' ').replace(/\.\d{3}Z$/, ' UTC');
    } catch {
      return '—';
    }
  }
  function fmtUtcShort(tsMs) {
    const n = Number(tsMs);
    if (!Number.isFinite(n) || n <= 0) return '—';
    try {
      return new Date(n).toISOString().slice(0, 16).replace('T', ' ');
    } catch {
      return '—';
    }
  }
  function fmtFreshness(sec) {
    const n = Number(sec);
    if (sec === null || sec === undefined || !Number.isFinite(n) || n < 0 || n >= 999999) return 'N/A';
    return n.toFixed(1) + 's';
  }
  function clampPct(val) {
    if (!Number.isFinite(val)) return 0;
    return Math.max(0, Math.min(100, val));
  }
  function storageGet(key) {
    try { return window.localStorage.getItem(key); } catch { return null; }
  }
  function storageSet(key, val) {
    try { window.localStorage.setItem(key, String(val)); } catch { /* ignore */ }
  }

  /* ── Non-Blocking Toast Stack ── */
  function showToast({ level = 'info', title = '', message = '', durationMs = 5000 } = {}) {
    let stack = document.getElementById('ui-toast-stack');
    if (!stack) {
      stack = document.createElement('div');
      stack.id = 'ui-toast-stack';
      stack.className = 'ui-toast-stack';
      stack.setAttribute('role', 'region');
      stack.setAttribute('aria-label', 'System notifications');
      stack.setAttribute('aria-live', 'polite');
      document.body.appendChild(stack);
    }
    while (stack.children.length >= 5) stack.firstElementChild.remove();
    const el = document.createElement('div');
    el.className = 'ui-toast';
    el.dataset.level = level;
    el.setAttribute('role', level === 'error' ? 'alert' : 'status');
    el.innerHTML =
      '<div class="ui-toast__body">' +
        (title ? '<div class="ui-toast__title">' + esc(title) + '</div>' : '') +
        '<div class="ui-toast__msg">' + esc(message) + '</div>' +
      '</div>' +
      '<button type="button" class="ui-btn ui-btn--ghost ui-btn--sm" data-toast-close aria-label="Dismiss notification">&times;</button>';
    let timer = durationMs > 0 ? setTimeout(() => el.remove(), durationMs) : null;
    el.addEventListener('mouseenter', () => {
      if (timer) { clearTimeout(timer); timer = null; }
    });
    el.addEventListener('mouseleave', () => {
      if (timer) { clearTimeout(timer); timer = null; }
      if (durationMs > 0) timer = setTimeout(() => el.remove(), 2200);
    });
    const closeBtn = el.querySelector('[data-toast-close]');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        if (timer) { clearTimeout(timer); timer = null; }
        el.remove();
      });
    }
    stack.appendChild(el);
  }

  /* ── Fetch Helpers ── */
  async function getJson(url) {
    const r = await fetch(url, { cache: 'no-store' });
    let data = null;
    try { data = await r.json(); } catch { data = null; }
    if (!r.ok) {
      const msg = data && data.error ? String(data.error) : ('HTTP ' + r.status);
      const err = new Error(msg);
      err.status = r.status;
      err.payload = data;
      throw err;
    }
    return data;
  }

  async function postJson(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      cache: 'no-store',
    });
    let data = null;
    try { data = await r.json(); } catch { data = null; }
    if (!r.ok) {
      const msg = data && data.error ? String(data.error) : ('HTTP ' + r.status);
      const err = new Error('HTTP ' + r.status + ': ' + msg);
      err.status = r.status;
      err.payload = data;
      throw err;
    }
    return data;
  }

  function renderHonestEmpty(stateAttr, badgeText, detailText) {
    return (
      '<div class="ui-honest-empty" data-state="' + esc(stateAttr) + '" data-live="false">' +
        '<span class="ui-state-badge" data-state="' + esc(stateAttr) + '" data-live="false">' + esc(badgeText) + '</span>' +
        '<div>' + esc(detailText) + '</div>' +
      '</div>'
    );
  }

  /* ── WAI-ARIA Tab Switching ── */
  function setActiveTab(tabId, { focusTab = false, persist = true } = {}) {
    const next = TABS.includes(tabId) ? tabId : 'overview';
    state.activeTab = next;
    TABS.forEach((id) => {
      const btn = document.getElementById('tab-' + id);
      const panel = document.getElementById('panel-' + id);
      const active = id === next;
      if (btn) {
        btn.setAttribute('aria-selected', active ? 'true' : 'false');
        btn.setAttribute('tabindex', active ? '0' : '-1');
        if (active && focusTab) btn.focus();
      }
      if (panel) panel.hidden = !active;
    });
    if (persist) {
      storageSet(TAB_STORAGE_KEY, next);
      if (window.location.hash !== '#' + next) {
        try { window.history.replaceState(null, '', '#' + next); } catch { /* ignore */ }
      }
    }
  }

  function initTabs() {
    const hashTab = (window.location.hash || '').replace(/^#/, '');
    const savedTab = storageGet(TAB_STORAGE_KEY) || '';
    const initial = TABS.includes(hashTab) ? hashTab : (TABS.includes(savedTab) ? savedTab : 'overview');
    setActiveTab(initial, { persist: false });

    const tablist = document.querySelector('[role="tablist"]');
    if (tablist) {
      tablist.addEventListener('keydown', (ev) => {
        const cur = ev.target instanceof Element ? ev.target.closest('[role="tab"]') : null;
        if (!cur) return;
        const idx = TABS.indexOf(cur.dataset.tab || '');
        if (idx < 0) return;
        let nextIdx = -1;
        if (ev.key === 'ArrowRight') nextIdx = (idx + 1) % TABS.length;
        else if (ev.key === 'ArrowLeft') nextIdx = (idx - 1 + TABS.length) % TABS.length;
        else if (ev.key === 'Home') nextIdx = 0;
        else if (ev.key === 'End') nextIdx = TABS.length - 1;
        if (nextIdx >= 0) {
          ev.preventDefault();
          setActiveTab(TABS[nextIdx], { focusTab: true });
        }
      });
    }
    window.addEventListener('hashchange', () => {
      const h = (window.location.hash || '').replace(/^#/, '');
      if (TABS.includes(h) && h !== state.activeTab) setActiveTab(h, { persist: false });
    });
  }

  /* ── Visibility-Aware Auto-Refresh Scheduler ── */
  function stopPolling() {
    if (state.pollTimer !== null) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }
  function schedulePolling() {
    stopPolling();
    if (state.autoRefreshMs > 0 && !document.hidden) {
      state.pollTimer = setInterval(() => {
        if (!document.hidden) refreshAll();
      }, state.autoRefreshMs);
    }
  }
  function initAutoRefresh() {
    const sel = document.getElementById('autoRefreshSelect');
    const saved = storageGet(POLL_STORAGE_KEY);
    if (sel && saved !== null && ['0', '10000', '30000', '60000'].includes(saved)) {
      sel.value = saved;
      state.autoRefreshMs = Number(saved);
    } else if (sel) {
      state.autoRefreshMs = Number(sel.value) || 30000;
    }
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        stopPolling();
      } else {
        schedulePolling();
        if (state.autoRefreshMs > 0) refreshAll();
      }
    });
    schedulePolling();
  }

  /* ── Renderers ── */
  function renderTopbarHealth(resHealth, resExec) {
    const hb = document.getElementById('healthBadge');
    const clk = document.getElementById('lastSyncClock');
    const modePill = document.getElementById('modePill');

    if (resExec && resExec.ok && resExec.data) {
      const rm = resExec.data.run_mode || 'SHADOW_DRY_RUN';
      if (modePill) {
        modePill.textContent = rm;
        modePill.dataset.state = 'sim';
        modePill.dataset.live = 'false';
      }
    }

    if (!hb) return;
    if (!resHealth.ok) {
      hb.dataset.state = 'upstream-error';
      hb.dataset.live = 'false';
      hb.textContent = 'UPSTREAM_ERROR';
      hb.title = resHealth.error;
      return;
    }
    const h = resHealth.data || {};
    const healthy = String(h.status || '').toUpperCase() === 'HEALTHY';
    hb.dataset.state = healthy ? 'live' : 'stale';
    hb.dataset.live = healthy ? 'true' : 'false';
    hb.textContent = (h.status || 'UNKNOWN') + ' · ' + (h.service || 'alphaloop-v2.2');
    hb.title = 'Service healthy';
    if (clk) clk.textContent = fmtUtc(h.ts_ms || Date.now());
  }

  function renderKpiCard({ label, value, stateAttr, badgeText, sub, severity = '' }) {
    const sevAttr = severity ? ' data-severity="' + esc(severity) + '"' : '';
    const liveAttr = stateAttr === 'live' ? 'true' : 'false';
    return (
      '<div class="ui-kpi" data-state="' + esc(stateAttr) + '" data-live="' + liveAttr + '">' +
        '<div class="ui-kpi__top">' +
          '<span class="ui-kpi__label">' + esc(label) + '</span>' +
          '<span class="ui-state-badge" data-state="' + esc(stateAttr) + '" data-live="' + liveAttr + '"' + sevAttr + '>' + esc(badgeText) + '</span>' +
        '</div>' +
        '<div class="ui-kpi__value kpi-value">' + esc(value) + '</div>' +
        '<div class="ui-kpi__sub" title="' + esc(sub) + '">' + esc(sub) + '</div>' +
      '</div>'
    );
  }

  function renderKpiGrid(R) {
    const cards = [];

    // 1. Plugins
    if (!R.plugins.ok) {
      cards.push({ label: 'Plugins', value: '—', stateAttr: 'upstream-error', badgeText: 'ERR', sub: R.plugins.error });
    } else {
      const list = (R.plugins.data && R.plugins.data.plugins) || [];
      cards.push(list.length > 0
        ? { label: 'Plugins', value: String(list.length), stateAttr: 'live', badgeText: 'LIVE', sub: list.length + ' alpha families' }
        : { label: 'Plugins', value: '—', stateAttr: 'no-channel', badgeText: 'NO_CHANNEL', sub: 'No plugins loaded' });
    }

    // 2. Strategies
    if (!R.strategies.ok) {
      cards.push({ label: 'Strategies', value: '—', stateAttr: 'upstream-error', badgeText: 'ERR', sub: R.strategies.error });
    } else {
      const list = (R.strategies.data && R.strategies.data.strategies) || [];
      const shadowCnt = list.filter((x) => x.status === 'SHADOW_LIVE').length;
      const otherCnt = list.length - shadowCnt;
      cards.push(list.length > 0
        ? { label: 'Strategies', value: String(list.length), stateAttr: 'live', badgeText: 'LIVE', sub: shadowCnt + ' SHADOW · ' + otherCnt + ' other' }
        : { label: 'Strategies', value: '0', stateAttr: 'no-channel', badgeText: 'NO_CHANNEL', sub: 'No active configs' });
    }

    // 3. Funding Rows & 4. BBO Ticks
    if (!R.market.ok) {
      cards.push({ label: 'Funding Rows', value: '—', stateAttr: 'upstream-error', badgeText: 'ERR', sub: R.market.error });
      cards.push({ label: 'BBO Ticks', value: '—', stateAttr: 'upstream-error', badgeText: 'ERR', sub: R.market.error });
    } else {
      const mk = R.market.data || {};
      const funding = Array.isArray(mk.funding) ? mk.funding : [];
      const bbo = Array.isArray(mk.bbo_ticks) ? mk.bbo_ticks : [];
      const fRows = funding.reduce((a, x) => a + numOr(x.rows, 0), 0);
      const fSyms = funding.reduce((a, x) => a + numOr(x.symbols, 0), 0);
      cards.push(funding.length > 0
        ? { label: 'Funding Rows', value: fRows.toLocaleString('en-US'), stateAttr: 'live', badgeText: 'LIVE', sub: funding.length + ' venues · ' + fSyms + ' venue-syms' }
        : { label: 'Funding Rows', value: '—', stateAttr: 'no-channel', badgeText: 'NO_CHANNEL', sub: 'No funding bars' });

      const bRows = bbo.reduce((a, x) => a + numOr(x.rows, 0), 0);
      const bLatWeighted = bbo.reduce((a, x) => a + numOr(x.avg_latency_ms, 0) * Math.max(1, numOr(x.rows, 0)), 0);
      const bWeight = bbo.reduce((a, x) => a + Math.max(1, numOr(x.rows, 0)), 0);
      const avgLat = bWeight > 0 ? (bLatWeighted / bWeight).toFixed(1) : '—';
      cards.push(bbo.length > 0
        ? { label: 'BBO Ticks', value: bRows.toLocaleString('en-US'), stateAttr: 'live', badgeText: 'LIVE', sub: bbo.length + ' venues · ' + avgLat + 'ms avg' }
        : { label: 'BBO Ticks', value: '—', stateAttr: 'no-channel', badgeText: 'NO_CHANNEL', sub: 'No live BBO ticks' });
    }

    // 5. DQ Certs
    if (!R.dq.ok) {
      cards.push({ label: 'DQ Certs', value: '—', stateAttr: 'upstream-error', badgeText: 'ERR', sub: R.dq.error });
    } else {
      const reps = (R.dq.data && R.dq.data.reports) || [];
      const fullCnt = reps.filter((x) => x.tier === 'DATA_QUALIFIED_FULL').length;
      const rejCnt = reps.filter((x) => x.tier === 'DATA_REJECTED').length;
      const warnCnt = reps.length - fullCnt - rejCnt;
      cards.push(reps.length > 0
        ? { label: 'DQ Certs', value: String(reps.length), stateAttr: 'live', badgeText: 'LIVE', sub: fullCnt + ' FULL · ' + warnCnt + ' WARN · ' + rejCnt + ' REJ' }
        : { label: 'DQ Certs', value: '—', stateAttr: 'no-channel', badgeText: 'NO_CHANNEL', sub: 'No DQ certs yet' });
    }

    // 6. Shadow Parity
    if (!R.parity.ok) {
      cards.push({ label: 'Shadow Parity', value: '—', stateAttr: 'upstream-error', badgeText: 'ERR', sub: R.parity.error });
    } else {
      const sp = R.parity.data || {};
      if (!sp.valid) {
        cards.push({ label: 'Shadow Parity', value: '—', stateAttr: 'no-channel', badgeText: 'NO_CHANNEL', sub: sp.error || 'No shadow report' });
      } else {
        const st = String(sp.status || 'PARITY_OK');
        const stratCnt = (sp.strategies || []).length;
        let stateAttr = 'live';
        let badgeText = 'LIVE';
        let severity = '';
        if (st === 'PARITY_OK') {
          stateAttr = 'live'; badgeText = 'LIVE';
        } else if (st.startsWith('WARN') || st.startsWith('DEGRADED')) {
          stateAttr = 'stale'; badgeText = 'WARN';
        } else {
          stateAttr = 'paused'; badgeText = 'HALT'; severity = 'critical';
        }
        cards.push({
          label: 'Shadow Parity',
          value: st,
          stateAttr,
          badgeText,
          severity,
          sub: 'BBO ' + fmtFreshness(sp.bbo_freshness_sec) + ' · ' + stratCnt + ' strat',
        });
      }
    }

    // 7. Audit Chain
    if (!R.exec.ok) {
      cards.push({ label: 'Audit Chain', value: '—', stateAttr: 'upstream-error', badgeText: 'ERR', sub: R.exec.error });
    } else {
      const ex = R.exec.data || {};
      const ks = ex.kill_switch || {};
      const valid = Boolean(ex.audit_chain_valid);
      const engaged = Boolean(ks.engaged);
      cards.push({
        label: 'Audit Chain',
        value: '#' + String(ex.audit_head_seq ?? 0),
        stateAttr: valid && !engaged ? 'live' : (valid ? 'paused' : 'upstream-error'),
        badgeText: valid ? (engaged ? 'KS' : 'VALID') : 'BROKEN',
        severity: engaged ? 'critical' : '',
        sub: String(ex.audit_chain_length ?? 0) + ' events · ' + (engaged ? 'KS ENGAGED' : 'KS CLEAR'),
      });
    }

    // 8. Fleet Reachable
    if (!R.fleet.ok) {
      cards.push({ label: 'Fleet Reachable', value: '—', stateAttr: 'upstream-error', badgeText: 'ERR', sub: R.fleet.error });
    } else {
      const fl = R.fleet.data || {};
      if (fl.error) {
        cards.push({ label: 'Fleet Reachable', value: '—', stateAttr: 'upstream-error', badgeText: 'ERR', sub: String(fl.error) });
      } else if (!fl.nodes_total) {
        cards.push({ label: 'Fleet Reachable', value: '—', stateAttr: 'no-channel', badgeText: 'NO_CHANNEL', sub: 'No SSH snapshot' });
      } else {
        const up = numOr(fl.nodes_reachable, 0);
        const tot = numOr(fl.nodes_total, 0);
        const p0 = numOr(fl.p0_alerts, 0);
        const p1 = numOr(fl.p1_alerts, 0);
        let stateAttr = 'live';
        let badgeText = 'LIVE';
        if (p0 > 0 || up === 0) { stateAttr = 'upstream-error'; badgeText = 'P0'; }
        else if (up < tot || p1 > 0) { stateAttr = 'stale'; badgeText = 'WARN'; }
        cards.push({
          label: 'Fleet Reachable',
          value: up + '/' + tot,
          stateAttr,
          badgeText,
          sub: 'P0:' + p0 + ' · P1:' + p1,
        });
      }
    }

    const grid = document.getElementById('kpiGrid');
    if (grid) grid.innerHTML = cards.map(renderKpiCard).join('');
  }

  function renderShadowParity(res) {
    const box = document.getElementById('shadowParityBox');
    if (!box) return;
    if (!res.ok) {
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', res.error);
      return;
    }
    const sp = res.data || {};
    if (!sp.valid) {
      box.innerHTML = renderHonestEmpty('no-channel', 'NO_CHANNEL', sp.error || 'Shadow daemon has not written shadow_parity_report.json yet');
      return;
    }
    const statusStr = String(sp.status || 'PARITY_OK');
    const statusCls = statusStr === 'PARITY_OK' ? 'ui-pill--ok' : ((statusStr.startsWith('WARN') || statusStr.startsWith('DEGRADED')) ? 'ui-pill--warn' : 'ui-pill--bad');
    const strategies = Array.isArray(sp.strategies) ? sp.strategies : [];

    const rowsHtml = strategies.map((x) => {
      const drift = numOr(x.drift_bps, 0);
      const absDrift = Math.abs(drift);
      const driftPct = clampPct((absDrift / 150) * 100);
      const driftFill = absDrift <= 150 ? 'ui-meter__fill--ok' : 'ui-meter__fill--bad';
      const driftTextCls = absDrift <= 150 ? 'val-pos' : 'val-neg';
      const comparable = x.backtest_comparable !== false;
      const spreadRatio = numOr(x.spread_ratio, 0);
      const spreadPillCls = spreadRatio <= 1.5 ? 'ui-pill--ok' : (spreadRatio <= 2.0 ? 'ui-pill--warn' : 'ui-pill--bad');
      const rowStatus = String(x.status || '');
      const rowStatusCls = rowStatus === 'PARITY_OK' ? 'ui-pill--ok' : (rowStatus === 'WARN_WIDE_SPREAD' ? 'ui-pill--warn' : 'ui-pill--bad');
      const shRet = numOr(x.shadow_return_pct, 0);

      return (
        '<tr>' +
          '<td><strong>' + esc(x.strategy_id) + '</strong><span class="val-muted">@' + esc(x.version) + '</span></td>' +
          '<td><span class="ui-pill ui-pill--info">' + esc(x.plugin_id || '—') + '</span></td>' +
          '<td class="num">' + esc(fmtUsd(x.shadow_equity_usd, 2)) + ' <span class="' + (shRet >= 0 ? 'val-pos' : 'val-neg') + '">(' + esc(fmtSigned(shRet, 2, '%')) + ')</span></td>' +
          '<td class="num">' + (comparable ? esc(fmtSigned(x.backtest_return_pct, 2, '%')) : '<span class="val-muted" title="' + esc(x.detail || 'Warmup window < 1h') + '">N/A (Warmup)</span>') + '</td>' +
          '<td>' +
            '<div class="ui-meter">' +
              '<div class="ui-meter__track"><div class="ui-meter__fill ' + driftFill + '" style="width:' + driftPct.toFixed(1) + '%"></div></div>' +
              '<span class="ui-meter__val ' + driftTextCls + '">' + esc(fmtSigned(drift, 1, ' bp')) + '</span>' +
            '</div>' +
          '</td>' +
          '<td class="num">' + esc(fmtNum(x.live_median_half_spread_bps, 2)) + 'bp <span class="ui-pill ' + spreadPillCls + '">' + esc(fmtNum(spreadRatio, 2)) + 'x</span></td>' +
          '<td class="num">' + esc(x.plans_created ?? 0) + ' / ' + esc(x.plans_blocked ?? 0) + '</td>' +
          '<td><span class="ui-pill ' + rowStatusCls + '">' + esc(rowStatus || '—') + '</span></td>' +
        '</tr>'
      );
    }).join('');

    box.innerHTML =
      '<div class="ui-meta-bar">' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Status</span><span class="ui-pill ' + statusCls + '">' + esc(statusStr) + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">BBO Age</span><span class="ui-meta-item__v">' + esc(fmtFreshness(sp.bbo_freshness_sec)) + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Live Enabled</span><span class="ui-meta-item__v">' + esc(sp.live_execution_enabled ? 'TRUE' : 'false') + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Trace</span><code>' + esc(sp.trace_id || '—') + '</code></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Updated</span><span class="ui-meta-item__v">' + esc(fmtUtc(sp.ts_ms)) + '</span></span>' +
      '</div>' +
      '<div class="ui-table-wrap">' +
        '<table class="ui-table ui-table--sticky-first">' +
          '<thead><tr><th>Strategy</th><th>Plugin</th><th class="num">Shadow Equity</th><th class="num">Backtest Ret</th><th>Drift (&le;150bp)</th><th class="num">Live Spread (&le;1.5x)</th><th class="num">Plans (OK/Blk)</th><th>Parity</th></tr></thead>' +
          '<tbody>' + (rowsHtml || '<tr><td colspan="8" class="val-muted">No active SHADOW_LIVE strategies evaluated yet</td></tr>') + '</tbody>' +
        '</table>' +
      '</div>';
  }

  function renderPortfolio(res) {
    const box = document.getElementById('portfolioBox');
    if (!box) return;
    if (!res.ok) {
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', res.error);
      return;
    }
    const pf = res.data || {};
    if (!pf.valid) {
      box.innerHTML = renderHonestEmpty('no-channel', 'NO_CHANNEL', pf.error || 'No portfolio report saved yet');
      return;
    }
    const projections = Array.isArray(pf.scaling_projections) ? pf.scaling_projections : [];
    const sleeves = pf.sleeve_summaries && typeof pf.sleeve_summaries === 'object' ? Object.entries(pf.sleeve_summaries) : [];

    const sleeveBar = sleeves.length > 0
      ? '<div class="ui-meta-bar">' +
          '<span class="ui-meta-item__k">Sleeves</span>' +
          sleeves.map(([name, s]) => (
            '<span class="ui-chip mono">' + esc(name) +
              ' · w=' + esc(fmtNum(numOr(s.weight_final, 0) * 100, 1)) + '%' +
              ' · APR ' + esc(fmtNum(s.apr_pct, 1)) + '%' +
              ' · SR ' + esc(fmtNum(s.sharpe, 2)) +
            '</span>'
          )).join('') +
        '</div>'
      : '';

    const scRows = projections.map((x) => {
      const gross = numOr(x.gross_backtest_apr_pct, 0);
      const drag = Math.max(0, numOr(x.capacity_impact_drag_apr_pct, 0));
      const net = numOr(x.net_capacity_adjusted_apr_pct, 0);
      const denom = Math.max(Math.abs(gross), drag, 1);
      const dragPct = clampPct((drag / denom) * 100);
      const dragFill = (dragPct >= 30 || net <= 0) ? 'ui-meter__fill--bad' : 'ui-meter__fill--warn';
      return (
        '<tr>' +
          '<td class="num"><strong>' + esc(fmtUsd(x.capital_usd, 0)) + '</strong></td>' +
          '<td class="num">' + esc(fmtNum(gross, 2)) + '%</td>' +
          '<td>' +
            '<div class="ui-meter">' +
              '<div class="ui-meter__track"><div class="ui-meter__fill ' + dragFill + '" style="width:' + dragPct.toFixed(1) + '%"></div></div>' +
              '<span class="ui-meter__val val-warn">-' + esc(fmtNum(drag, 2)) + '%</span>' +
            '</div>' +
          '</td>' +
          '<td class="num ' + (net > 0 ? 'val-pos' : 'val-neg') + '">' + esc(fmtNum(net, 2)) + '%</td>' +
          '<td class="num">' + esc(fmtUsd(x.est_monthly_net_usd, 2)) + '/mo</td>' +
        '</tr>'
      );
    }).join('');

    const aprVal = numOr(pf.annualized_apr_pct, 0);
    box.innerHTML =
      '<div class="ui-meta-bar">' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Mode</span><span class="ui-pill ui-pill--info">' + esc(pf.weighting_mode || '—') + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Overlap</span><span class="ui-meta-item__v">' + esc(pf.overlap_days ?? '—') + 'd</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Annualized APR</span><span class="ui-meta-item__v ' + (aprVal > 0 ? 'val-pos' : 'val-neg') + '">' + esc(fmtNum(aprVal, 2)) + '%</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Sharpe</span><span class="ui-meta-item__v">' + esc(fmtNum(pf.sharpe_ratio, 2)) + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Calmar</span><span class="ui-meta-item__v">' + esc(fmtNum(pf.calmar_ratio, 2)) + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">MaxDD</span><span class="ui-meta-item__v">' + esc(fmtNum(pf.max_drawdown_pct, 2)) + '%</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">OOS Retention</span><span class="ui-meta-item__v">' + esc(fmtNum(pf.oos_retention, 2)) + 'x</span></span>' +
      '</div>' +
      sleeveBar +
      '<div class="ui-table-wrap">' +
        '<table class="ui-table ui-table--sticky-first">' +
          '<thead><tr><th class="num">Capital</th><th class="num">Backtest APR</th><th>Capacity Drag</th><th class="num">Net Adjusted APR</th><th class="num">Est Monthly</th></tr></thead>' +
          '<tbody>' + (scRows || '<tr><td colspan="5" class="val-muted">No capacity scaling projections available</td></tr>') + '</tbody>' +
        '</table>' +
      '</div>';
  }

  function renderMarketSummary(res) {
    const box = document.getElementById('marketSummaryBox');
    if (!box) return;
    if (!res.ok) {
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', res.error);
      return;
    }
    const mk = res.data || {};
    const rows = [];
    (Array.isArray(mk.funding) ? mk.funding : []).forEach((r) => rows.push({ feed: 'FUNDING', chip: 'ui-pill--info', ...r, hasLat: false }));
    (Array.isArray(mk.candles_1h) ? mk.candles_1h : []).forEach((r) => rows.push({ feed: '1H CANDLES', chip: 'ui-pill--teal', ...r, hasLat: false }));
    (Array.isArray(mk.bbo_ticks) ? mk.bbo_ticks : []).forEach((r) => rows.push({ feed: 'LIVE BBO', chip: 'ui-pill--ok', ...r, hasLat: true }));

    if (rows.length === 0) {
      box.innerHTML = renderHonestEmpty('no-channel', 'NO_CHANNEL', 'No market data ingested in SQLite store');
      return;
    }

    const htmlRows = rows.map((r) => (
      '<tr>' +
        '<td><span class="ui-pill ' + r.chip + '">' + esc(r.feed) + '</span></td>' +
        '<td><strong>' + esc(r.venue) + '</strong></td>' +
        '<td class="num">' + esc(numOr(r.symbols, 0).toLocaleString('en-US')) + '</td>' +
        '<td class="num">' + esc(numOr(r.rows, 0).toLocaleString('en-US')) + '</td>' +
        '<td class="num">' + (r.hasLat && isFiniteNum(r.avg_latency_ms) ? esc(fmtNum(r.avg_latency_ms, 1)) + ' ms' : '<span class="val-muted">—</span>') + '</td>' +
        '<td class="mono">' + esc(fmtUtcShort(r.min_ts)) + ' &rarr; ' + esc(fmtUtcShort(r.max_ts)) + '</td>' +
      '</tr>'
    )).join('');

    box.innerHTML =
      '<div class="ui-table-wrap">' +
        '<table class="ui-table ui-table--sticky-first">' +
          '<thead><tr><th>Feed</th><th>Venue</th><th class="num">Symbols</th><th class="num">Rows</th><th class="num">Avg Latency</th><th>UTC Date Span</th></tr></thead>' +
          '<tbody>' + htmlRows + '</tbody>' +
        '</table>' +
      '</div>';
  }

  function renderExecution(res) {
    const box = document.getElementById('execBox');
    const tabBadge = document.getElementById('tabExecBadge');
    if (!box) return;
    if (!res.ok) {
      if (tabBadge) tabBadge.hidden = true;
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', res.error);
      return;
    }
    const ex = res.data || {};
    const ks = ex.kill_switch || {};
    const engaged = Boolean(ks.engaged);
    const chainValid = Boolean(ex.audit_chain_valid);

    if (tabBadge) {
      if (engaged || !chainValid) {
        tabBadge.textContent = engaged ? 'KILL' : 'AUDIT';
        tabBadge.hidden = false;
      } else {
        tabBadge.hidden = true;
      }
    }

    const killStrip = engaged
      ? '<div class="kill-strip" role="alert">' +
          '<span class="kill-strip__dot" aria-hidden="true"></span>' +
          '<strong>KILL SWITCH ENGAGED</strong>' +
          '<span>Trigger: <code class="mono">' + esc(ks.trigger || 'UNKNOWN') + '</code></span>' +
          '<span>Actor: ' + esc(ks.actor || '—') + '</span>' +
          '<span>Detail: ' + esc(ks.detail || '—') + '</span>' +
          '<span class="mono">Seq #' + esc(ks.engage_seq ?? 0) + '</span>' +
        '</div>'
      : '';

    const plans = Array.isArray(ex.recent_plans) ? ex.recent_plans.slice(0, 30) : [];
    const exRows = plans.map((x) => {
      const ok = x.status === 'PLANNED_SHADOW';
      return (
        '<tr>' +
          '<td><code>' + esc(x.plan_id) + '</code></td>' +
          '<td class="mono">' + esc(fmtUtcShort(x.ts_ms)) + '</td>' +
          '<td><strong>' + esc(x.strategy_id) + '</strong><span class="val-muted">@' + esc(x.version) + '</span></td>' +
          '<td class="mono">' + esc(x.symbol) + '</td>' +
          '<td>' + esc(x.action) + '</td>' +
          '<td class="num">' + esc(fmtUsd(x.notional_usd, 2)) + '</td>' +
          '<td class="num">' + esc(fmtNum(x.expected_edge_bps, 2)) + ' bp</td>' +
          '<td><span class="ui-pill ' + (ok ? 'ui-pill--ok' : 'ui-pill--bad') + '">' + esc(x.status) + '</span></td>' +
          '<td class="wrap-cell">' + esc(x.block_reason || '—') + '</td>' +
        '</tr>'
      );
    }).join('');

    const headHashShort = String(ex.audit_head_hash || '').slice(0, 12) || '—';
    box.innerHTML =
      killStrip +
      '<div class="ui-meta-bar">' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Run Mode</span><span class="ui-state-badge" data-state="sim" data-live="false">' + esc(ex.run_mode || 'SHADOW_DRY_RUN') + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Live Enabled</span><span class="ui-meta-item__v">' + esc(ex.live_execution_enabled ? 'TRUE' : 'false') + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Kill Switch</span><span class="ui-pill ' + (engaged ? 'ui-pill--bad' : 'ui-pill--ok') + '">' + (engaged ? 'ENGAGED' : 'CLEAR') + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Hash Audit</span><span class="ui-pill ' + (chainValid ? 'ui-pill--ok' : 'ui-pill--bad') + '">' + (chainValid ? 'CHAIN VALID' : 'CHAIN BROKEN') + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Detail</span><span class="ui-meta-item__v">' + esc(ex.audit_chain_detail || 'OK') + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Head</span><span class="ui-meta-item__v">#' + esc(ex.audit_head_seq ?? 0) + ' (' + esc(ex.audit_chain_length ?? 0) + ' events)</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Hash</span><code>' + esc(headHashShort) + '</code></span>' +
      '</div>' +
      '<div class="ui-table-wrap">' +
        '<table class="ui-table ui-table--sticky-first">' +
          '<thead><tr><th>Plan ID</th><th>Time (UTC)</th><th>Strategy</th><th>Symbol</th><th>Action</th><th class="num">Notional</th><th class="num">Edge</th><th>Status</th><th>Block Reason</th></tr></thead>' +
          '<tbody>' + (exRows || '<tr><td colspan="9" class="val-muted">No shadow execution plans recorded yet</td></tr>') + '</tbody>' +
        '</table>' +
      '</div>';
  }

  function renderStrategies(res) {
    const box = document.getElementById('strategiesBox');
    if (!box) return;
    if (!res.ok) {
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', res.error);
      return;
    }
    const list = (res.data && Array.isArray(res.data.strategies)) ? res.data.strategies : [];
    if (list.length === 0) {
      box.innerHTML = renderHonestEmpty('no-channel', 'NO_CHANNEL', 'No strategies registered in HotConfigStore');
      return;
    }

    const sRows = list.map((x) => {
      const sid = String(x.strategy_id || '');
      const cfg = x.config || {};
      const plugin = cfg.plugin_id || 'funding_carry';
      const venueA = cfg.venue_a || '—';
      const venueB = cfg.venue_b || '';
      const venues = venueB && venueB !== venueA ? (venueA + ' / ' + venueB) : venueA;
      const busy = state.inFlight.has(sid);
      const activeAct = state.inFlightAction[sid] || '';
      const st = String(x.status || '');
      const stCls = st === 'SHADOW_LIVE' ? 'ui-pill--ok' : ((st === 'LIVE' || st === 'CANARY_APPROVED' || st === 'CANARY') ? 'ui-pill--warn' : 'ui-pill--info');

      const btn = (action, label, variant) => {
        const loading = busy && activeAct === action;
        return (
          '<button type="button" class="ui-btn ui-btn--sm ' + variant + '"' +
          ' data-action="' + action + '" data-strategy-id="' + esc(sid) + '"' +
          (busy ? ' disabled' : '') +
          (loading ? ' data-loading="true"' : '') + '>' +
          '<span class="ui-spinner" aria-hidden="true"></span>' +
          '<span class="ui-btn__label">' + esc(label) + '</span>' +
          '</button>'
        );
      };

      return (
        '<tr>' +
          '<td><strong>' + esc(sid) + '</strong></td>' +
          '<td><span class="ui-pill ui-pill--teal">' + esc(plugin) + '</span></td>' +
          '<td class="mono">' + esc(x.active_version || '—') + '</td>' +
          '<td><span class="ui-pill ' + stCls + '">' + esc(st || '—') + '</span></td>' +
          '<td class="mono">' + esc(venues) + '</td>' +
          '<td><code>' + esc(String(x.config_sha256 || '').slice(0, 12) || '—') + '</code></td>' +
          '<td class="mono">' + esc(fmtUtcShort(x.updated_ts_ms)) + '</td>' +
          '<td><div class="ui-btn-group">' +
            btn('evolve', 'Evolve', 'ui-btn--primary') +
            btn('sprint', 'Sprint', 'ui-btn--solid') +
            btn('micro', 'MicroRun', '') +
            btn('rollback', 'Rollback', 'ui-btn--ghost') +
          '</div></td>' +
        '</tr>'
      );
    }).join('');

    box.innerHTML =
      '<div class="ui-table-wrap">' +
        '<table class="ui-table ui-table--sticky-first">' +
          '<thead><tr><th>Strategy</th><th>Plugin</th><th>Version</th><th>Status</th><th>Venues</th><th>Config SHA256</th><th>Updated (UTC)</th><th>Actions</th></tr></thead>' +
          '<tbody>' + sRows + '</tbody>' +
        '</table>' +
      '</div>';
  }

  function renderPlugins(res) {
    const box = document.getElementById('pluginsBox');
    if (!box) return;
    if (!res.ok) {
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', res.error);
      return;
    }
    const list = (res.data && Array.isArray(res.data.plugins)) ? res.data.plugins : [];
    if (list.length === 0) {
      box.innerHTML = renderHonestEmpty('no-channel', 'NO_CHANNEL', 'No plugins registered');
      return;
    }

    const plRows = list.map((x) => {
      const bounds = x.mutable_param_bounds && typeof x.mutable_param_bounds === 'object' ? Object.entries(x.mutable_param_bounds) : [];
      const chips = bounds.length > 0
        ? '<div class="ui-chip-list">' + bounds.map(([k, v]) => {
            const rng = Array.isArray(v) ? (esc(v[0]) + '..' + esc(v[1])) : esc(v);
            return '<span class="ui-chip mono">' + esc(k) + ': ' + rng + '</span>';
          }).join('') + '</div>'
        : '<span class="val-muted">—</span>';
      return (
        '<tr>' +
          '<td><strong>' + esc(x.plugin_id) + '</strong></td>' +
          '<td>' + esc(x.display_name) + '</td>' +
          '<td><span class="ui-pill ui-pill--info">' + esc(x.strategy_family) + '</span></td>' +
          '<td class="wrap-cell">' + chips + '</td>' +
          '<td class="wrap-cell">' + esc(x.description) + '</td>' +
        '</tr>'
      );
    }).join('');

    box.innerHTML =
      '<div class="ui-table-wrap">' +
        '<table class="ui-table ui-table--sticky-first">' +
          '<thead><tr><th>Plugin ID</th><th>Display Name</th><th>Family</th><th>Mutable Param Bounds</th><th>Description</th></tr></thead>' +
          '<tbody>' + plRows + '</tbody>' +
        '</table>' +
      '</div>';
  }

  function dqTierBucket(tier) {
    if (tier === 'DATA_QUALIFIED_FULL') return 'FULL';
    if (tier === 'DATA_REJECTED') return 'REJECTED';
    return 'WARN';
  }

  function renderDqReports(res) {
    const box = document.getElementById('dqBox');
    const bar = document.getElementById('dqFilterBar');
    if (!box) return;
    if (!res.ok) {
      if (bar) bar.innerHTML = '';
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', res.error);
      return;
    }
    const all = (res.data && Array.isArray(res.data.reports)) ? res.data.reports.slice(0, 50) : [];
    const counts = { ALL: all.length, FULL: 0, WARN: 0, REJECTED: 0 };
    all.forEach((x) => { counts[dqTierBucket(x.tier)] += 1; });

    if (bar) {
      bar.innerHTML = ['ALL', 'FULL', 'WARN', 'REJECTED'].map((tierKey) => (
        '<button type="button" class="ui-btn ui-btn--sm ui-btn--ghost" data-action="dq-filter" data-dq-tier="' + tierKey + '" aria-pressed="' + (state.dqFilter === tierKey ? 'true' : 'false') + '">' +
          tierKey + ' <span class="mono">(' + counts[tierKey] + ')</span>' +
        '</button>'
      )).join('');
    }

    if (all.length === 0) {
      box.innerHTML = renderHonestEmpty('no-channel', 'NO_CHANNEL', 'No DQ certificates issued yet');
      return;
    }

    const filtered = state.dqFilter === 'ALL' ? all : all.filter((x) => dqTierBucket(x.tier) === state.dqFilter);
    const dqRows = filtered.map((x) => {
      const b = dqTierBucket(x.tier);
      const cls = b === 'FULL' ? 'ui-pill--ok' : (b === 'REJECTED' ? 'ui-pill--bad' : 'ui-pill--warn');
      return (
        '<tr>' +
          '<td><code>' + esc(x.cert_id) + '</code></td>' +
          '<td>' + esc(x.dataset_name) + '</td>' +
          '<td class="mono">' + esc(x.venue_pair) + '</td>' +
          '<td class="num">' + esc(numOr(x.row_count, 0).toLocaleString('en-US')) + '</td>' +
          '<td class="num">' + esc(fmtNum(x.basis_coverage_pct, 1)) + '%</td>' +
          '<td class="num">' + esc(x.passed_count ?? 0) + '/' + esc(x.total_count ?? 0) + '</td>' +
          '<td><span class="ui-pill ' + cls + '">' + esc(x.tier) + '</span></td>' +
        '</tr>'
      );
    }).join('');

    box.innerHTML =
      '<div class="ui-table-wrap">' +
        '<table class="ui-table ui-table--sticky-first">' +
          '<thead><tr><th>Cert ID</th><th>Dataset</th><th>Venue Pair</th><th class="num">Rows</th><th class="num">Basis Cov</th><th class="num">Checks</th><th>Tier</th></tr></thead>' +
          '<tbody>' + (dqRows || '<tr><td colspan="7" class="val-muted">No certificates match filter "' + esc(state.dqFilter) + '"</td></tr>') + '</tbody>' +
        '</table>' +
      '</div>';
  }

  function renderMicroRuns(res) {
    const box = document.getElementById('microBox');
    if (!box) return;
    if (!res.ok) {
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', res.error);
      return;
    }
    const runs = (res.data && Array.isArray(res.data.runs)) ? res.data.runs.slice(0, 50) : [];
    if (runs.length === 0) {
      box.innerHTML = renderHonestEmpty('no-channel', 'NO_CHANNEL', 'No micro-runs evaluated yet');
      return;
    }

    const mrRows = runs.map((x) => {
      const st = String(x.status || '');
      const stCls = (st === 'COMPLETED_PASS' || st === 'READY_FOR_HUMAN_LIVE') ? 'ui-pill--ok' : (st === 'KILLED_EARLY' ? 'ui-pill--bad' : 'ui-pill--warn');
      const elapsed = numOr(x.elapsed_hours, 0);
      const target = Math.max(0.01, numOr(x.target_hours, 24));
      const pct = clampPct((elapsed / target) * 100);
      const barCls = st === 'KILLED_EARLY' ? 'ui-meter__fill--bad' : (pct >= 100 ? 'ui-meter__fill--ok' : '');
      const pnl = numOr(x.paper_net_pnl_usd, 0);
      const slipRatio = numOr(x.slippage_ratio, 0);
      const slipCls = slipRatio <= 1.5 ? 'val-pos' : (slipRatio <= 2.0 ? 'val-warn' : 'val-neg');

      return (
        '<tr>' +
          '<td><code>' + esc(x.run_id) + '</code></td>' +
          '<td><strong>' + esc(x.strategy_id) + '</strong><span class="val-muted">@' + esc(x.version) + '</span></td>' +
          '<td><span class="ui-pill ui-pill--info">' + esc(x.plugin_id) + '</span></td>' +
          '<td><span class="ui-pill ' + stCls + '">' + esc(st) + '</span></td>' +
          '<td>' +
            '<div class="ui-meter">' +
              '<div class="ui-meter__track"><div class="ui-meter__fill ' + barCls + '" style="width:' + pct.toFixed(1) + '%"></div></div>' +
              '<span class="ui-meter__val">' + esc(fmtNum(elapsed, 1)) + 'h / ' + esc(fmtNum(target, 1)) + 'h</span>' +
            '</div>' +
          '</td>' +
          '<td class="num">' + esc(numOr(x.ticks_observed, 0).toLocaleString('en-US')) + '</td>' +
          '<td class="num">' + esc(fmtNum(x.realized_spread_bps_median, 2)) + 'bp / ' + esc(fmtNum(x.assumed_slippage_bps, 2)) + 'bp <span class="' + slipCls + '">(' + esc(fmtNum(slipRatio, 2)) + 'x)</span></td>' +
          '<td class="num ' + (pnl >= 0 ? 'val-pos' : 'val-neg') + '">' + esc(fmtUsd(pnl, 2, true)) + '</td>' +
          '<td class="num">' + esc(fmtNum(x.paper_drawdown_pct, 2)) + '%</td>' +
          '<td>' + esc(x.dq_tier || '—') + '</td>' +
          '<td class="wrap-cell">' + esc(x.kill_reason || '—') + '</td>' +
        '</tr>'
      );
    }).join('');

    box.innerHTML =
      '<div class="ui-table-wrap">' +
        '<table class="ui-table ui-table--sticky-first">' +
          '<thead><tr><th>Run ID</th><th>Strategy</th><th>Plugin</th><th>Status</th><th>Elapsed / Target</th><th class="num">BBO Ticks</th><th class="num">Slip Ratio</th><th class="num">Paper Net PnL</th><th class="num">Paper MaxDD</th><th>DQ Tier</th><th>Kill / Note</th></tr></thead>' +
          '<tbody>' + mrRows + '</tbody>' +
        '</table>' +
      '</div>';
  }

  function renderCycleDetail(x) {
    const df = x.diff_scorecard || {};
    const prop = x.proposal || {};
    const rt = x.red_team_verdict || {};
    const gv = x.gate_verdict || {};
    const params = prop.param_changes && typeof prop.param_changes === 'object' ? Object.entries(prop.param_changes) : [];
    const findings = Array.isArray(rt.findings) ? rt.findings : [];
    const checks = Array.isArray(gv.checks) ? gv.checks : [];

    return (
      '<div class="cycle-detail-grid">' +
        '<div class="cycle-detail-block">' +
          '<h3>Diff Scorecard</h3>' +
          '<div class="cycle-detail-kv mono">' +
            '<span>Net PnL &Delta;: <strong>' + esc(fmtUsd(df.net_pnl_delta_usd, 2, true)) + '</strong></span>' +
            '<span>APR &Delta;: <strong>' + esc(fmtSigned(df.apr_delta_pct, 2, '%')) + '</strong></span>' +
            '<span>Sharpe &Delta;: <strong>' + esc(fmtSigned(df.sharpe_delta, 2)) + '</strong></span>' +
            '<span>MaxDD &Delta;: <strong>' + esc(fmtSigned(df.max_dd_delta_pct, 2, '%')) + '</strong></span>' +
            '<span>Fee Ratio &Delta;: <strong>' + esc(fmtSigned(df.fee_ratio_delta, 3)) + '</strong></span>' +
            '<span>Rec: <strong>' + esc(df.recommendation || '—') + '</strong></span>' +
          '</div>' +
        '</div>' +
        '<div class="cycle-detail-block">' +
          '<h3>Proposal (' + esc(prop.model_used || 'deterministic') + ')</h3>' +
          '<p>' + esc(prop.hypothesis || '—') + '</p>' +
          (params.length > 0
            ? '<div class="ui-chip-list">' + params.map(([k, v]) => '<span class="ui-chip mono">' + esc(k) + ' = ' + esc(v) + '</span>').join('') + '</div>'
            : '<span class="val-muted">No param changes</span>') +
        '</div>' +
        '<div class="cycle-detail-block">' +
          '<h3>Red-Team Verdict (' + esc(rt.model_used || '—') + ' · ' + esc(rt.decision || '—') + ')</h3>' +
          '<p>' + esc(rt.summary || '—') + '</p>' +
          (findings.length > 0
            ? '<div class="ui-chip-list">' + findings.map((f) => {
                const cls = f.severity === 'BLOCKER' ? 'ui-pill--bad' : (f.severity === 'MAJOR' ? 'ui-pill--warn' : 'ui-pill--info');
                return '<span class="ui-pill ' + cls + '" title="' + esc(f.detail || '') + '">' + esc(f.severity) + ': ' + esc(f.item) + '</span>';
              }).join('') + '</div>'
            : '<span class="val-muted">No red-team findings</span>') +
        '</div>' +
        '<div class="cycle-detail-block">' +
          '<h3>6-Gate Verification (Trial #' + esc(gv.trial_number ?? '—') + ' · ' + esc(gv.dq_tier || '—') + ')</h3>' +
          (checks.length > 0
            ? '<div class="ui-chip-list">' + checks.map((c) => (
                '<span class="ui-pill ' + (c.passed ? 'ui-pill--ok' : 'ui-pill--bad') + '" title="' + esc((c.value ?? '') + ' vs ' + (c.threshold ?? '') + ' ' + (c.detail || '')) + '">' +
                  esc(c.gate_id) + ' ' + (c.passed ? 'PASS' : 'FAIL') +
                '</span>'
              )).join('') + '</div>'
            : '<span class="val-muted">No gate details</span>') +
        '</div>' +
      '</div>'
    );
  }

  function renderCycles(res) {
    const box = document.getElementById('cyclesBox');
    if (!box) return;
    if (!res.ok) {
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', res.error);
      return;
    }
    const list = (res.data && Array.isArray(res.data.cycles)) ? res.data.cycles : [];
    if (list.length === 0) {
      box.innerHTML = renderHonestEmpty('no-channel', 'NO_CHANNEL', 'No agent evolution cycles recorded yet');
      return;
    }

    const cRows = list.map((x) => {
      const cid = String(x.cycle_id || '');
      const df = x.diff_scorecard || {};
      const fb = x.feedback || {};
      const bm = x.baseline_metrics || {};
      const cm = x.candidate_metrics || {};
      const gv = x.gate_verdict || {};
      const rt = x.red_team_verdict || {};
      const expanded = state.expandedCycles.has(cid);
      const aprDelta = numOr(df.apr_delta_pct, 0);
      const stage = x.promotion_stage || (x.promoted_shadow ? 'SHADOW_LIVE' : 'HELD');
      const stageCls = stage === 'SHADOW_LIVE' ? 'ui-pill--ok' : (stage === 'HELD' ? 'ui-pill--bad' : 'ui-pill--warn');
      const rtDec = String(rt.decision || '—');
      const rtCls = rtDec === 'APPROVE_SHADOW' ? 'ui-pill--ok' : (rtDec === 'REJECT' ? 'ui-pill--bad' : 'ui-pill--warn');

      const mainRow =
        '<tr>' +
          '<td><code>' + esc(cid) + '</code></td>' +
          '<td><strong>' + esc(x.strategy_id) + '</strong> <span class="mono val-muted">(' + esc(x.from_version) + ' &rarr; ' + esc(x.to_version) + ')</span></td>' +
          '<td><span class="ui-pill">' + esc(fb.anomaly_type || '—') + '</span></td>' +
          '<td class="num">' + esc(fmtNum(bm.annualized_apr_pct, 1)) + '% &rarr; <strong>' + esc(fmtNum(cm.annualized_apr_pct, 1)) + '%</strong> <span class="' + (aprDelta >= 0 ? 'val-pos' : 'val-neg') + '">(' + esc(fmtSigned(aprDelta, 1, '%')) + ')</span></td>' +
          '<td class="num">' + esc(fmtSigned(df.sharpe_delta, 2)) + ' / ' + esc(fmtSigned(df.max_dd_delta_pct, 2, '%')) + '</td>' +
          '<td><span class="ui-pill ' + (gv.overall_pass ? 'ui-pill--ok' : 'ui-pill--bad') + '">' + esc(gv.passed_count ?? 0) + '/' + esc(gv.total_count ?? 0) + '</span></td>' +
          '<td><span class="ui-pill ' + rtCls + '">' + esc(rtDec) + '</span></td>' +
          '<td><span class="ui-pill ' + stageCls + '">' + esc(stage) + '</span></td>' +
          '<td><button type="button" class="ui-btn ui-btn--sm ui-btn--ghost" data-action="toggle-cycle" data-cycle-id="' + esc(cid) + '" aria-expanded="' + (expanded ? 'true' : 'false') + '">' + (expanded ? 'Hide Diff' : 'Inspect Diff') + '</button></td>' +
        '</tr>';

      const detailRow = expanded
        ? '<tr class="detail-row"><td colspan="9">' + renderCycleDetail(x) + '</td></tr>'
        : '';

      return mainRow + detailRow;
    }).join('');

    box.innerHTML =
      '<div class="ui-table-wrap">' +
        '<table class="ui-table ui-table--sticky-first">' +
          '<thead><tr><th>Cycle</th><th>Strategy (From &rarr; To)</th><th>Anomaly</th><th class="num">APR Shift</th><th class="num">&Delta;Sharpe / &Delta;DD</th><th>6-Gate</th><th>Red-Team</th><th>Stage</th><th>Details</th></tr></thead>' +
          '<tbody>' + cRows + '</tbody>' +
        '</table>' +
      '</div>';
  }

  function budgetBarClass(pct) {
    if (pct >= 90) return 'ui-meter__fill--bad';
    if (pct >= 70) return 'ui-meter__fill--warn';
    return '';
  }

  function renderModels(res) {
    const box = document.getElementById('modelsBox');
    if (!box) return;
    if (!res.ok) {
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', res.error);
      return;
    }
    const m = res.data || {};
    const roundSpent = Math.max(0, numOr(m.round_spent_usd, 0));
    const roundCap = Math.max(0.01, numOr(m.max_cycle_budget_usd, 0.50));
    const dailySpent = Math.max(0, Math.max(numOr(m.spent_usd, 0), numOr(m.daily_spent_usd, 0)));
    const dailyCap = Math.max(0.01, numOr(m.max_daily_budget_usd, 3.00));
    const roundPct = clampPct((roundSpent / roundCap) * 100);
    const dailyPct = clampPct((dailySpent / dailyCap) * 100);

    const routes = Array.isArray(m.routes) ? m.routes : [];
    const mRows = routes.map((x) => (
      '<tr>' +
        '<td><strong>' + esc(x.role) + '</strong></td>' +
        '<td class="mono">' + esc(x.family) + ' / ' + esc(x.model_id) + '</td>' +
        '<td class="num">' + esc(x.pricing_usd_1m) + '</td>' +
        '<td><span class="ui-pill ' + (x.configured ? 'ui-pill--ok' : 'ui-pill--warn') + '">' + (x.configured ? 'KEY SET' : 'FALLBACK') + '</span></td>' +
        '<td class="wrap-cell">' + esc(x.rationale) + '</td>' +
      '</tr>'
    )).join('');

    box.innerHTML =
      '<div class="budget-grid">' +
        '<div class="budget-item">' +
          '<div class="budget-item__head">' +
            '<span class="budget-item__label">Per-Round LLM Spend Cap</span>' +
            '<span class="budget-item__val">' + esc(fmtUsd(roundSpent, 4)) + ' / ' + esc(fmtUsd(roundCap, 2)) + ' (' + roundPct.toFixed(1) + '%)</span>' +
          '</div>' +
          '<div class="ui-meter__track"><div class="ui-meter__fill ' + budgetBarClass(roundPct) + '" style="width:' + roundPct.toFixed(1) + '%"></div></div>' +
        '</div>' +
        '<div class="budget-item">' +
          '<div class="budget-item__head">' +
            '<span class="budget-item__label">Persistent UTC Daily LLM Budget</span>' +
            '<span class="budget-item__val">' + esc(fmtUsd(dailySpent, 4)) + ' / ' + esc(fmtUsd(dailyCap, 2)) + ' (' + dailyPct.toFixed(1) + '%)</span>' +
          '</div>' +
          '<div class="ui-meter__track"><div class="ui-meter__fill ' + budgetBarClass(dailyPct) + '" style="width:' + dailyPct.toFixed(1) + '%"></div></div>' +
        '</div>' +
      '</div>' +
      '<div class="ui-table-wrap">' +
        '<table class="ui-table ui-table--sticky-first">' +
          '<thead><tr><th>Role</th><th>Family / Model</th><th class="num">Price (In/Out 1M)</th><th>API Status</th><th>Rationale</th></tr></thead>' +
          '<tbody>' + (mRows || '<tr><td colspan="5" class="val-muted">No routes configured</td></tr>') + '</tbody>' +
        '</table>' +
      '</div>';
  }

  function renderTrials(res) {
    const box = document.getElementById('trialsBox');
    const bar = document.getElementById('trialsFilterBar');
    if (!box) return;
    if (!res.ok) {
      if (bar) bar.innerHTML = '';
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', res.error);
      return;
    }
    const all = (res.data && Array.isArray(res.data.trials)) ? res.data.trials : [];
    const stratSet = Array.from(new Set(all.map((x) => String(x.strategy_id || '')).filter(Boolean))).sort();
    if (state.trialStrategy !== 'ALL' && !stratSet.includes(state.trialStrategy)) {
      state.trialStrategy = 'ALL';
    }

    const byStrat = state.trialStrategy === 'ALL' ? all : all.filter((x) => x.strategy_id === state.trialStrategy);
    const passCnt = byStrat.filter((x) => Boolean(x.overall_pass)).length;
    const failCnt = byStrat.length - passCnt;

    if (bar) {
      const opts = ['<option value="ALL">All Strategies (' + all.length + ')</option>']
        .concat(stratSet.map((sid) => '<option value="' + esc(sid) + '"' + (state.trialStrategy === sid ? ' selected' : '') + '>' + esc(sid) + '</option>'))
        .join('');
      const btns = [
        ['ALL', byStrat.length],
        ['PASS', passCnt],
        ['FAIL', failCnt],
      ].map(([k, cnt]) => (
        '<button type="button" class="ui-btn ui-btn--sm ui-btn--ghost" data-action="trial-status-filter" data-trial-status="' + k + '" aria-pressed="' + (state.trialResult === k ? 'true' : 'false') + '">' +
          k + ' <span class="mono">(' + cnt + ')</span>' +
        '</button>'
      )).join('');
      bar.innerHTML =
        '<label class="sr-only" for="trialStrategyFilter">Filter trials by strategy</label>' +
        '<select id="trialStrategyFilter" class="ui-select" data-change-action="trial-strategy-filter">' + opts + '</select>' +
        btns;
    }

    if (all.length === 0) {
      box.innerHTML = renderHonestEmpty('no-channel', 'NO_CHANNEL', 'No backtest trials recorded in G6 ledger yet');
      return;
    }

    const filtered = byStrat.filter((x) => {
      const passed = Boolean(x.overall_pass);
      if (state.trialResult === 'PASS') return passed;
      if (state.trialResult === 'FAIL') return !passed;
      return true;
    });

    const tRows = filtered.map((x) => {
      const passed = Boolean(x.overall_pass);
      const pnl = numOr(x.net_pnl_usd, 0);
      return (
        '<tr>' +
          '<td class="mono">#' + esc(x.trial_id) + '</td>' +
          '<td><strong>' + esc(x.strategy_id) + '</strong><span class="val-muted">@' + esc(x.version) + '</span></td>' +
          '<td class="mono">' + esc(x.provenance) + '</td>' +
          '<td class="num ' + (pnl >= 0 ? 'val-pos' : 'val-neg') + '">' + esc(fmtUsd(pnl, 2, true)) + '</td>' +
          '<td class="num">' + esc(fmtNum(x.apr_pct, 2)) + '%</td>' +
          '<td class="num">' + esc(fmtNum(x.sharpe, 2)) + '</td>' +
          '<td class="num">' + esc(fmtNum(x.max_dd_pct, 2)) + '%</td>' +
          '<td><span class="ui-pill ' + (passed ? 'ui-pill--ok' : 'ui-pill--bad') + '">' + (passed ? 'PASS' : 'FAIL') + '</span></td>' +
        '</tr>'
      );
    }).join('');

    box.innerHTML =
      '<div class="ui-table-wrap">' +
        '<table class="ui-table ui-table--sticky-first">' +
          '<thead><tr><th>Trial</th><th>Strategy</th><th>Provenance</th><th class="num">Net PnL</th><th class="num">APR</th><th class="num">Sharpe</th><th class="num">MaxDD</th><th>6-Gate</th></tr></thead>' +
          '<tbody>' + (tRows || '<tr><td colspan="8" class="val-muted">No trials match current filter</td></tr>') + '</tbody>' +
        '</table>' +
      '</div>';
  }

  function renderFleetBar(pct, subText) {
    const val = clampPct(numOr(pct, 0));
    const fillCls = val >= 85 ? 'ui-meter__fill--bad' : (val >= 70 ? 'ui-meter__fill--warn' : 'ui-meter__fill--ok');
    return (
      '<div class="ui-meter">' +
        '<div class="ui-meter__track"><div class="ui-meter__fill ' + fillCls + '" style="width:' + val.toFixed(1) + '%"></div></div>' +
        '<span class="ui-meter__val">' + val.toFixed(1) + '% <span class="val-muted">(' + esc(subText) + ')</span></span>' +
      '</div>'
    );
  }

  function renderFleet(res) {
    const box = document.getElementById('fleetBox');
    if (!box) return;
    if (!res.ok) {
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', res.error);
      return;
    }
    const fl = res.data || {};
    if (fl.error) {
      box.innerHTML = renderHonestEmpty('upstream-error', 'UPSTREAM_ERROR', String(fl.error));
      return;
    }
    const nodes = Array.isArray(fl.nodes) ? fl.nodes : [];
    if (nodes.length === 0) {
      box.innerHTML = renderHonestEmpty('no-channel', 'NO_CHANNEL', 'No read-only SSH fleet snapshot found');
      return;
    }

    const p0 = numOr(fl.p0_alerts, 0);
    const p1 = numOr(fl.p1_alerts, 0);
    const flRows = nodes.map((x) => {
      const up = Boolean(x.reachable);
      const alerts = Array.isArray(x.alerts) ? x.alerts : [];
      const alertHtml = alerts.length > 0
        ? '<div class="ui-chip-list">' + alerts.map((a) => {
            const cls = a.severity === 'P0' ? 'ui-pill--bad' : 'ui-pill--warn';
            return '<span class="ui-pill ' + cls + '" title="' + esc(a.detail || '') + '">' + esc(a.severity) + ':' + esc(a.code) + '</span>';
          }).join('') + '</div>'
        : '<span class="ui-pill ui-pill--ok">NONE</span>';

      return (
        '<tr>' +
          '<td><strong>' + esc(x.label || x.node_id || '—') + '</strong>' + (x.primary_role ? '<div class="val-muted" style="font-size:11px">' + esc(x.primary_role) + '</div>' : '') + '</td>' +
          '<td><code>' + esc(x.ip || '—') + '</code></td>' +
          '<td><span class="ui-pill ' + (up ? 'ui-pill--ok' : 'ui-pill--bad') + '">' + (up ? 'OK' : 'DOWN') + '</span></td>' +
          '<td>' + (up ? renderFleetBar(x.root_disk_pct, fmtNum(x.root_disk_free_gb, 1) + 'GB free') : '<span class="val-muted">—</span>') + '</td>' +
          '<td>' + (up ? renderFleetBar(x.mem_used_pct, fmtNum(x.mem_avail_mb, 0) + 'MB avail') : '<span class="val-muted">—</span>') + '</td>' +
          '<td class="wrap-cell">' + alertHtml + '</td>' +
        '</tr>'
      );
    }).join('');

    box.innerHTML =
      '<div class="ui-meta-bar">' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Reachable</span><span class="ui-meta-item__v">' + esc(fl.nodes_reachable ?? 0) + '/' + esc(fl.nodes_total ?? nodes.length) + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">P0</span><span class="ui-pill ' + (p0 > 0 ? 'ui-pill--bad' : 'ui-pill--ok') + '">' + esc(p0) + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">P1</span><span class="ui-pill ' + (p1 > 0 ? 'ui-pill--warn' : 'ui-pill--ok') + '">' + esc(p1) + '</span></span>' +
        '<span class="ui-meta-item"><span class="ui-meta-item__k">Audited</span><span class="ui-meta-item__v">' + esc(fmtUtc(fl.generated_ts_ms)) + '</span></span>' +
      '</div>' +
      '<div class="ui-table-wrap">' +
        '<table class="ui-table ui-table--sticky-first">' +
          '<thead><tr><th>Node</th><th>IP</th><th>SSH</th><th>Root Disk</th><th>RAM</th><th>Alerts</th></tr></thead>' +
          '<tbody>' + flRows + '</tbody>' +
        '</table>' +
      '</div>';
  }

  /* ── Fault-Isolated Refresh (STD-37 Promise.allSettled) ── */
  async function refreshAll() {
    const seq = ++state.refreshSeq;
    const refBtn = document.getElementById('refreshBtn');
    if (refBtn) { refBtn.disabled = true; refBtn.dataset.loading = 'true'; }

    const specs = [
      ['health', '/api/health'],
      ['strategies', '/api/strategies'],
      ['cycles', '/api/cycles'],
      ['models', '/api/models'],
      ['trials', '/api/trials'],
      ['market', '/api/market/summary'],
      ['plugins', '/api/plugins'],
      ['dq', '/api/dq/reports'],
      ['micro', '/api/micro-runs'],
      ['fleet', '/api/fleet/audit'],
      ['portfolio', '/api/portfolio'],
      ['exec', '/api/execution/plans'],
      ['parity', '/api/shadow/parity'],
    ];

    try {
      const settled = await Promise.allSettled(specs.map(([, url]) => getJson(url)));
      if (seq !== state.refreshSeq) return;

      const R = Object.create(null);
      specs.forEach(([key], idx) => {
        const s = settled[idx];
        if (s.status === 'fulfilled') {
          R[key] = { ok: true, data: s.value, error: '' };
        } else {
          const msg = s.reason && s.reason.message ? String(s.reason.message) : 'Request failed';
          R[key] = { ok: false, data: null, error: msg };
        }
      });
      state.last = R;

      renderTopbarHealth(R.health, R.exec);
      renderKpiGrid(R);
      renderShadowParity(R.parity);
      renderPortfolio(R.portfolio);
      renderMarketSummary(R.market);
      renderExecution(R.exec);
      renderStrategies(R.strategies);
      renderPlugins(R.plugins);
      renderDqReports(R.dq);
      renderMicroRuns(R.micro);
      renderCycles(R.cycles);
      renderModels(R.models);
      renderTrials(R.trials);
      renderFleet(R.fleet);
    } finally {
      if (seq === state.refreshSeq && refBtn) {
        refBtn.disabled = false;
        refBtn.dataset.loading = 'false';
      }
    }
  }

  /* ── Per-Row In-Flight Lock Helper ── */
  function setStrategyInFlight(sid, actionName) {
    if (actionName) {
      state.inFlight.add(sid);
      state.inFlightAction[sid] = actionName;
    } else {
      state.inFlight.delete(sid);
      delete state.inFlightAction[sid];
    }
    if (state.last.strategies) renderStrategies(state.last.strategies);
  }

  /* ── Direct Single-Cycle Evolution Action ── */
  async function runDirectEvolve(sid) {
    if (!sid || state.inFlight.has(sid)) return;
    setStrategyInFlight(sid, 'evolve');
    try {
      const ver = 'v-auto-' + Date.now();
      const d = await postJson('/api/agent/run-cycle', { strategy_id: sid, next_version: ver });
      const gates = d.gate_verdict ? (d.gate_verdict.passed_count + '/' + d.gate_verdict.total_count) : '?';
      const rt = d.red_team_verdict ? d.red_team_verdict.decision : '?';
      showToast({
        level: d.promoted_shadow ? 'success' : 'info',
        title: 'Cycle ' + (d.cycle_id || 'Completed') + ' (' + sid + ')',
        message: 'Stage: ' + (d.promotion_stage || 'HELD') + ' · Gates: ' + gates + ' · Red-Team: ' + rt,
      });
      await refreshAll();
    } catch (err) {
      showToast({
        level: 'error',
        title: 'Evolve Failed (' + sid + ')',
        message: err && err.message ? err.message : 'Request failed',
      });
    } finally {
      setStrategyInFlight(sid, '');
    }
  }

  /* ── Single Native Accessible <dialog id="actionDialog"> Controller ── */
  let _dlgAbort = null;

  function closeActionDialog(prevFocus) {
    const dlg = document.getElementById('actionDialog');
    if (_dlgAbort) {
      _dlgAbort.abort();
      _dlgAbort = null;
    }
    if (dlg && dlg.open) dlg.close();
    if (prevFocus && typeof prevFocus.focus === 'function') {
      try { prevFocus.focus(); } catch { /* ignore */ }
    }
  }

  async function openActionDialog(mode, sid) {
    const dlg = document.getElementById('actionDialog');
    if (!dlg || !sid) return;
    if (_dlgAbort) {
      _dlgAbort.abort();
      _dlgAbort = null;
    }
    if (dlg.open) dlg.close();

    const prevFocus = document.activeElement;
    const ac = new AbortController();
    _dlgAbort = ac;

    const titleEl = dlg.querySelector('[data-dlg-title]');
    const contentEl = dlg.querySelector('[data-dlg-content]');
    const errEl = dlg.querySelector('[data-dlg-error]');
    const resEl = dlg.querySelector('[data-dlg-result]');
    const submitBtn = dlg.querySelector('[data-dlg-submit]');
    const submitLabel = dlg.querySelector('[data-dlg-submit-label]');
    const cancelBtns = dlg.querySelectorAll('[data-dlg-cancel]');

    const setDlgError = (msg) => {
      if (!errEl) return;
      if (msg) {
        errEl.textContent = String(msg);
        errEl.hidden = false;
      } else {
        errEl.textContent = '';
        errEl.hidden = true;
      }
    };
    const setDlgLoading = (loading) => {
      if (!submitBtn) return;
      submitBtn.disabled = loading;
      submitBtn.dataset.loading = loading ? 'true' : 'false';
    };

    setDlgError('');
    if (resEl) { resEl.innerHTML = ''; resEl.hidden = true; }
    setDlgLoading(false);

    let onSubmit = null;

    if (mode === 'sprint') {
      titleEl.textContent = 'Run Multi-Round Sprint — ' + sid;
      submitLabel.textContent = 'Run Sprint';
      submitBtn.className = 'ui-btn ui-btn--primary';
      contentEl.innerHTML =
        '<div class="ui-dialog__field">' +
          '<label class="ui-dialog__label" for="dlgSprintRounds"><span>Max Rounds (1–5)</span><span class="ui-pill ui-pill--info">POST /api/agent/run-sprint</span></label>' +
          '<select id="dlgSprintRounds" class="ui-select mono">' +
            '<option value="1">1 round</option>' +
            '<option value="2">2 rounds</option>' +
            '<option value="3" selected>3 rounds (default)</option>' +
            '<option value="4">4 rounds</option>' +
            '<option value="5">5 rounds (max)</option>' +
          '</select>' +
          '<span class="ui-dialog__hint">Runs closed-loop Diagnose &rarr; Propose &rarr; DQ + 6-Gate &rarr; Red-Team iterations with early stop.</span>' +
        '</div>';

      onSubmit = async () => {
        setDlgError('');
        const raw = document.getElementById('dlgSprintRounds').value;
        const maxRounds = Number(raw);
        if (!Number.isInteger(maxRounds) || maxRounds < 1 || maxRounds > 5) {
          setDlgError('max_rounds must be an integer between 1 and 5.');
          return;
        }
        setDlgLoading(true);
        setStrategyInFlight(sid, 'sprint');
        try {
          const sp = await postJson('/api/agent/run-sprint', { strategy_id: sid, max_rounds: maxRounds });
          const rounds = Array.isArray(sp.rounds) ? sp.rounds : [];
          const rows = rounds.map((r, i) => {
            const df = r.diff_scorecard || {};
            const gv = r.gate_verdict || {};
            const rt = r.red_team_verdict || {};
            return (
              '<tr>' +
                '<td class="mono">R' + (i + 1) + '</td>' +
                '<td class="mono">' + esc(r.to_version) + '</td>' +
                '<td class="num">' + esc(fmtSigned(df.apr_delta_pct, 1, '%')) + '</td>' +
                '<td class="num">' + esc(gv.passed_count ?? 0) + '/' + esc(gv.total_count ?? 0) + '</td>' +
                '<td>' + esc(rt.decision || '—') + '</td>' +
                '<td><span class="ui-pill">' + esc(r.promotion_stage || 'HELD') + '</span></td>' +
              '</tr>'
            );
          }).join('');
          resEl.innerHTML =
            '<div class="cycle-detail-kv mono">' +
              '<span>Rounds: <strong>' + esc(sp.rounds_executed) + '</strong></span>' +
              '<span>Final: <strong>' + esc(sp.final_version || '—') + '</strong></span>' +
              '<span>Stage: <strong>' + esc(sp.final_promotion_stage || 'NONE') + '</strong></span>' +
              '<span>Stop: <code>' + esc(sp.stop_reason || '—') + '</code></span>' +
            '</div>' +
            '<div class="ui-table-wrap">' +
              '<table class="ui-table">' +
                '<thead><tr><th>#</th><th>Version</th><th class="num">&Delta;APR</th><th class="num">Gates</th><th>Red-Team</th><th>Stage</th></tr></thead>' +
                '<tbody>' + (rows || '<tr><td colspan="6" class="val-muted">No rounds returned</td></tr>') + '</tbody>' +
              '</table>' +
            '</div>';
          resEl.hidden = false;
          submitLabel.textContent = 'Run Again';
          showToast({
            level: sp.final_promotion_stage === 'SHADOW_LIVE' ? 'success' : 'info',
            title: 'Sprint Completed (' + sid + ')',
            message: 'Rounds: ' + sp.rounds_executed + ' · Stage: ' + sp.final_promotion_stage + ' · ' + sp.stop_reason,
          });
          await refreshAll();
        } catch (err) {
          setDlgError(err && err.message ? err.message : 'Sprint request failed');
        } finally {
          setDlgLoading(false);
          setStrategyInFlight(sid, '');
        }
      };
    } else if (mode === 'micro') {
      titleEl.textContent = 'Evaluate Micro-Run Window — ' + sid;
      submitLabel.textContent = 'Evaluate Micro-Run';
      submitBtn.className = 'ui-btn ui-btn--primary';
      contentEl.innerHTML =
        '<div class="ui-dialog__field">' +
          '<span class="ui-dialog__label"><span>Presets</span><span class="ui-pill ui-pill--info">POST /api/micro-runs/evaluate</span></span>' +
          '<div class="ui-btn-group">' +
            '<button type="button" class="ui-btn ui-btn--sm" data-action="micro-preset" data-hours="6">6h</button>' +
            '<button type="button" class="ui-btn ui-btn--sm" data-action="micro-preset" data-hours="24" aria-pressed="true">24h</button>' +
            '<button type="button" class="ui-btn ui-btn--sm" data-action="micro-preset" data-hours="72">72h</button>' +
            '<button type="button" class="ui-btn ui-btn--sm" data-action="micro-preset" data-hours="168">168h</button>' +
          '</div>' +
        '</div>' +
        '<div class="ui-dialog__field">' +
          '<label class="ui-dialog__label" for="dlgMicroHours"><span>Target Hours (&gt; 0 and &le; 720)</span></label>' +
          '<input id="dlgMicroHours" class="ui-input mono" type="number" min="0.1" max="720" step="any" value="24"/>' +
          '<span class="ui-dialog__hint">Evaluates live BBO tick window against assumed slippage &amp; dual-layer early-kill guard.</span>' +
        '</div>';

      onSubmit = async () => {
        setDlgError('');
        const raw = document.getElementById('dlgMicroHours').value;
        const hours = Number(raw);
        if (!Number.isFinite(hours) || hours <= 0 || hours > 720) {
          setDlgError('target_hours must be a finite number strictly > 0 and <= 720.');
          return;
        }
        setDlgLoading(true);
        setStrategyInFlight(sid, 'micro');
        try {
          const d = await postJson('/api/micro-runs/evaluate', { strategy_id: sid, target_hours: hours });
          closeActionDialog(prevFocus);
          showToast({
            level: d.status === 'COMPLETED_PASS' ? 'success' : (d.status === 'KILLED_EARLY' ? 'error' : 'info'),
            title: 'MicroRun ' + (d.run_id || 'Done') + ' (' + sid + ')',
            message: 'Status: ' + (d.status || '?') +
              ' · ' + fmtNum(d.elapsed_hours, 1) + 'h/' + fmtNum(d.target_hours, 1) + 'h' +
              ' · Slip: ' + fmtNum(d.slippage_ratio, 2) + 'x' +
              ' · PnL: ' + fmtUsd(d.paper_net_pnl_usd, 2, true) +
              (d.kill_reason ? ' · ' + d.kill_reason : ''),
          });
          await refreshAll();
        } catch (err) {
          setDlgError(err && err.message ? err.message : 'MicroRun evaluation failed');
        } finally {
          setDlgLoading(false);
          setStrategyInFlight(sid, '');
        }
      };
    } else if (mode === 'rollback') {
      titleEl.textContent = 'Rollback Strategy Version — ' + sid;
      submitLabel.textContent = 'Confirm Rollback';
      submitBtn.className = 'ui-btn ui-btn--danger';
      contentEl.innerHTML =
        '<div class="ui-dialog__field">' +
          '<label class="ui-dialog__label" for="dlgRollbackSelect"><span>Registered Versions</span><span id="dlgRollbackVerBadge" class="ui-pill">Loading...</span></label>' +
          '<select id="dlgRollbackSelect" class="ui-select mono"></select>' +
        '</div>' +
        '<div class="ui-dialog__field">' +
          '<label class="ui-dialog__label" for="dlgRollbackManual"><span>Manual Version Override (optional)</span></label>' +
          '<input id="dlgRollbackManual" class="ui-input mono" type="text" autocomplete="off" placeholder="Leave blank to use selected version above"/>' +
        '</div>' +
        '<div class="ui-dialog__field">' +
          '<label class="ui-dialog__label" for="dlgRollbackReason"><span>Audit Reason</span></label>' +
          '<input id="dlgRollbackReason" class="ui-input" type="text" autocomplete="off" value="Manual console rollback"/>' +
        '</div>' +
        '<div class="ui-dialog__field">' +
          '<label class="ui-dialog__label" for="dlgRollbackToken"><span>Human Approval Token</span><span id="dlgTokenReqBadge" class="ui-pill">OPTIONAL</span></label>' +
          '<input id="dlgRollbackToken" class="ui-input mono" type="text" autocomplete="off" placeholder="HUMAN_APPROVED:OPS-2026-01"/>' +
          '<span class="ui-dialog__hint">Format: <code>^HUMAN_APPROVED:[A-Za-z0-9._-]{4,64}$</code>. Required when target version status is CANARY / CANARY_APPROVED / LIVE.</span>' +
        '</div>';

      let allowedAuto = new Set(DEFAULT_ALLOWED_AUTO_STATUSES);
      const statusByVer = Object.create(null);

      const updateTokenReq = () => {
        const manual = (document.getElementById('dlgRollbackManual').value || '').trim();
        const selEl = document.getElementById('dlgRollbackSelect');
        const target = manual || (selEl ? selEl.value : '');
        const st = statusByVer[target] || '';
        const badge = document.getElementById('dlgTokenReqBadge');
        if (!badge) return;
        if (st && !allowedAuto.has(st)) {
          badge.className = 'ui-pill ui-pill--bad';
          badge.textContent = 'REQUIRED (' + st + ')';
        } else if (st) {
          badge.className = 'ui-pill ui-pill--ok';
          badge.textContent = 'OPTIONAL (' + st + ')';
        } else {
          badge.className = 'ui-pill ui-pill--warn';
          badge.textContent = 'IF CANARY/LIVE';
        }
      };

      const selEl = document.getElementById('dlgRollbackSelect');
      const manualEl = document.getElementById('dlgRollbackManual');
      if (selEl) selEl.addEventListener('change', updateTokenReq, { signal: ac.signal });
      if (manualEl) manualEl.addEventListener('input', updateTokenReq, { signal: ac.signal });

      // Hydrate version dropdown asynchronously after opening dialog
      getJson('/api/strategies/versions?strategy_id=' + encodeURIComponent(sid)).then((vd) => {
        if (ac.signal.aborted) return;
        if (Array.isArray(vd.allowed_auto_statuses) && vd.allowed_auto_statuses.length > 0) {
          allowedAuto = new Set(vd.allowed_auto_statuses);
        }
        const versions = Array.isArray(vd.versions) ? vd.versions : [];
        const activeVer = vd.active_version || '';
        const verBadge = document.getElementById('dlgRollbackVerBadge');
        if (verBadge) verBadge.textContent = versions.length + ' version(s)';
        if (selEl) {
          let defaultIdx = 0;
          const nonActiveIdx = versions.findIndex((v) => v.version !== activeVer);
          if (nonActiveIdx >= 0) defaultIdx = nonActiveIdx;
          selEl.innerHTML = versions.map((v, idx) => {
            statusByVer[v.version] = v.status || '';
            const tag = v.version === activeVer ? ' (ACTIVE)' : '';
            const sha = String(v.config_sha256 || '').slice(0, 8);
            return '<option value="' + esc(v.version) + '"' + (idx === defaultIdx ? ' selected' : '') + '>' +
              esc(v.version) + tag + ' [' + esc(v.status) + '] · ' + esc(sha) + ' · ' + esc(fmtUtcShort(v.created_ts_ms)) +
            '</option>';
          }).join('');
        }
        updateTokenReq();
      }).catch((err) => {
        if (ac.signal.aborted) return;
        const verBadge = document.getElementById('dlgRollbackVerBadge');
        if (verBadge) verBadge.textContent = 'Unavailable — use manual input';
        setDlgError('Could not load version history: ' + (err && err.message ? err.message : 'error'));
      });

      onSubmit = async () => {
        setDlgError('');
        const manual = (document.getElementById('dlgRollbackManual').value || '').trim();
        const selected = (document.getElementById('dlgRollbackSelect').value || '').trim();
        const targetVer = manual || selected;
        const reason = (document.getElementById('dlgRollbackReason').value || '').trim() || 'Manual console rollback';
        const token = (document.getElementById('dlgRollbackToken').value || '').trim();
        if (!targetVer) {
          setDlgError('Please select or enter a target_version.');
          return;
        }
        const targetStatus = statusByVer[targetVer] || '';
        if (targetStatus && !allowedAuto.has(targetStatus) && !token) {
          setDlgError('Target version ' + targetVer + ' has status ' + targetStatus + ' and requires a HUMAN_APPROVED:<id> token.');
          return;
        }
        if (token && !HUMAN_TOKEN_RE.test(token)) {
          setDlgError('human_approval_token must match ^HUMAN_APPROVED:[A-Za-z0-9._-]{4,64}$');
          return;
        }
        const payload = { strategy_id: sid, target_version: targetVer, reason };
        if (token) payload.human_approval_token = token;

        setDlgLoading(true);
        setStrategyInFlight(sid, 'rollback');
        try {
          const d = await postJson('/api/strategies/rollback', payload);
          closeActionDialog(prevFocus);
          showToast({
            level: 'success',
            title: 'Rollback Succeeded (' + sid + ')',
            message: 'Active version reverted to ' + (d.rolled_back_to || targetVer),
          });
          await refreshAll();
        } catch (err) {
          setDlgError(err && err.message ? err.message : 'Rollback failed');
        } finally {
          setDlgLoading(false);
          setStrategyInFlight(sid, '');
        }
      };
    }

    submitBtn.addEventListener('click', () => {
      if (typeof onSubmit === 'function') onSubmit();
    }, { signal: ac.signal });

    cancelBtns.forEach((b) => {
      b.addEventListener('click', () => closeActionDialog(prevFocus), { signal: ac.signal });
    });

    dlg.addEventListener('cancel', (ev) => {
      ev.preventDefault();
      closeActionDialog(prevFocus);
    }, { signal: ac.signal });

    dlg.addEventListener('click', (ev) => {
      if (ev.target === dlg) closeActionDialog(prevFocus);
    }, { signal: ac.signal });

    dlg.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' && ev.target instanceof HTMLInputElement && !submitBtn.disabled) {
        ev.preventDefault();
        if (typeof onSubmit === 'function') onSubmit();
      }
    }, { signal: ac.signal });

    dlg.showModal();
  }

  /* ── Delegated Event Routers (L-034 Zero Inline Handlers) ── */
  const ACTION_HANDLERS = Object.create(null);
  ACTION_HANDLERS.refresh = () => { refreshAll(); };
  ACTION_HANDLERS.tab = (el) => { setActiveTab(el.dataset.tab || 'overview'); };
  ACTION_HANDLERS.evolve = (el) => { runDirectEvolve(el.dataset.strategyId || ''); };
  ACTION_HANDLERS.sprint = (el) => { openActionDialog('sprint', el.dataset.strategyId || ''); };
  ACTION_HANDLERS.micro = (el) => { openActionDialog('micro', el.dataset.strategyId || ''); };
  ACTION_HANDLERS.rollback = (el) => { openActionDialog('rollback', el.dataset.strategyId || ''); };
  ACTION_HANDLERS['dq-filter'] = (el) => {
    state.dqFilter = el.dataset.dqTier || 'ALL';
    if (state.last.dq) renderDqReports(state.last.dq);
  };
  ACTION_HANDLERS['trial-status-filter'] = (el) => {
    state.trialResult = el.dataset.trialStatus || 'ALL';
    if (state.last.trials) renderTrials(state.last.trials);
  };
  ACTION_HANDLERS['toggle-cycle'] = (el) => {
    const cid = el.dataset.cycleId || '';
    if (!cid) return;
    if (state.expandedCycles.has(cid)) state.expandedCycles.delete(cid);
    else state.expandedCycles.add(cid);
    if (state.last.cycles) renderCycles(state.last.cycles);
  };
  ACTION_HANDLERS['micro-preset'] = (el) => {
    const hrs = el.dataset.hours || '24';
    const inp = document.getElementById('dlgMicroHours');
    if (inp) inp.value = hrs;
    document.querySelectorAll('[data-action="micro-preset"]').forEach((b) => {
      b.setAttribute('aria-pressed', b === el ? 'true' : 'false');
    });
  };

  const CHANGE_HANDLERS = Object.create(null);
  CHANGE_HANDLERS['auto-refresh'] = (el) => {
    const ms = Number(el.value);
    state.autoRefreshMs = Number.isFinite(ms) && ms >= 0 ? ms : 30000;
    storageSet(POLL_STORAGE_KEY, state.autoRefreshMs);
    schedulePolling();
  };
  CHANGE_HANDLERS['trial-strategy-filter'] = (el) => {
    state.trialStrategy = el.value || 'ALL';
    if (state.last.trials) renderTrials(state.last.trials);
  };

  document.addEventListener('click', (ev) => {
    const el = ev.target instanceof Element ? ev.target.closest('[data-action]') : null;
    if (!el || el.disabled || el.getAttribute('aria-disabled') === 'true') return;
    const fn = ACTION_HANDLERS[el.dataset.action];
    if (typeof fn === 'function') fn(el, ev);
  });

  document.addEventListener('change', (ev) => {
    const el = ev.target instanceof Element ? ev.target.closest('[data-change-action]') : null;
    if (!el || el.disabled) return;
    const fn = CHANGE_HANDLERS[el.dataset.changeAction];
    if (typeof fn === 'function') fn(el, ev);
  });

  /* ── Boot ── */
  initTabs();
  initAutoRefresh();
  refreshAll();
})();
</script>
</body>
</html>
"""


class ConsoleAppContext:
    """Shared dependency container for AlphaLoop v2.2 HTTP server."""

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
                parsed = urlparse(self.path)
                path = parsed.path
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
                if path == "/api/strategies/versions":
                    qs = parse_qs(parsed.query, keep_blank_values=False)
                    raw_sid = (qs.get("strategy_id") or [""])[0].strip()
                    if not raw_sid:
                        self._send_json(400, {"error": "MISSING_STRATEGY_ID"})
                        return
                    try:
                        sid = validate_strategy_id(raw_sid)
                    except ValueError as exc:
                        self._send_json(400, {"error": str(exc)})
                        return
                    versions = ctx.config_store.version_history(sid)
                    if not versions:
                        self._send_json(404, {"error": f"STRATEGY_NOT_FOUND: {sid}"})
                        return
                    active_cfg = ctx.config_store.read_active(sid)
                    self._send_json(
                        200,
                        {
                            "strategy_id": sid,
                            "active_version": active_cfg.version if active_cfg is not None else "",
                            "allowed_auto_statuses": sorted(ctx.config_store.ALLOWED_AUTO_STATUSES),
                            "versions": versions,
                        },
                    )
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
                            "daily_spent_usd": round(ctx.router.daily_spent_usd(), 6),
                            "max_cycle_budget_usd": round(ctx.router.max_cycle_budget_usd, 6),
                            "max_daily_budget_usd": round(ctx.router.max_daily_budget_usd, 6),
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
