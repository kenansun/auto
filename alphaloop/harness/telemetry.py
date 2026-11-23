"""Real-time strategy telemetry monitor & structured anomaly feedback emitter."""

from __future__ import annotations

import logging
from typing import Any

from alphaloop.core.models import BacktestMetrics, FeedbackEvent, StrategyConfig
from alphaloop.core.provenance import sha256_obj, utc_now_ms

logger = logging.getLogger(__name__)


class TelemetryMonitor:
    """Evaluates live/paper/backtest telemetry and emits structured FeedbackEvents.

    Instead of dumping raw logs into an LLM, this compresses operational performance
    into concrete anomaly classifications that trigger the Agent Evolution Loop.
    """

    def diagnose(self, cfg: StrategyConfig, metrics: BacktestMetrics) -> FeedbackEvent:
        now = utc_now_ms()
        snap: dict[str, Any] = {
            "strategy_id": cfg.strategy_id,
            "version": cfg.version,
            "days": metrics.days,
            "trade_count": metrics.trade_count,
            "avg_hold_hours": metrics.avg_hold_hours,
            "gross_funding_usd": metrics.gross_funding_usd,
            "basis_pnl_usd": metrics.basis_pnl_usd,
            "total_costs_usd": round(metrics.total_fees_usd + metrics.total_slippage_usd + metrics.total_rebalance_usd, 4),
            "net_pnl_usd": metrics.net_pnl_usd,
            "apr_pct": metrics.annualized_apr_pct,
            "sharpe": metrics.sharpe_ratio,
            "max_dd_pct": metrics.max_drawdown_pct,
            "fee_to_gross_ratio": metrics.fee_to_gross_ratio,
            "win_rate": metrics.win_rate,
            "entry_spread_apr_pct": cfg.entry_spread_apr_pct,
            "exit_spread_apr_pct": cfg.exit_spread_apr_pct,
            "use_atb_buggy_rules": cfg.use_atb_buggy_rules,
        }

        anomaly = "HEALTHY"
        severity = "INFO"
        summary = f"Healthy carry profile: APR={metrics.annualized_apr_pct:+.1f}%, Sharpe={metrics.sharpe_ratio:.2f}, Cost/Gross={metrics.fee_to_gross_ratio:.1%}."

        # Rule 1: Churn / Fee Overrun (catches ATB H6 exact pathology!)
        if metrics.trade_count >= 5 and (
            metrics.fee_to_gross_ratio > 0.75
            or (metrics.gross_funding_usd > 0 and metrics.net_pnl_usd < 0 and (metrics.total_fees_usd + metrics.total_slippage_usd) > metrics.gross_funding_usd)
        ):
            anomaly = "CHURN_EXCESS"
            severity = "P0"
            summary = (
                f"Fee churn destroying positive gross carry: gross=${metrics.gross_funding_usd:+.2f} "
                f"vs costs=${snap['total_costs_usd']:.2f} (ratio={metrics.fee_to_gross_ratio:.2f}x, "
                f"avg_hold={metrics.avg_hold_hours:.1f}h, trades={metrics.trade_count}). "
                f"Action needed: widen entry/exit hysteresis gap, increase min_hold_hours, or switch to lower-fee venue/maker."
            )
        # Rule 2: Basis Shock
        elif metrics.basis_pnl_usd < -abs(metrics.gross_funding_usd) * 0.6 and metrics.basis_pnl_usd < -10.0:
            anomaly = "BASIS_SHOCK"
            severity = "P0"
            summary = (
                f"Adverse mark-basis divergence (${metrics.basis_pnl_usd:+.2f}) eroded funding carry "
                f"(${metrics.gross_funding_usd:+.2f}). Tighten max_basis_entry_bps / stop_loss_basis_bps."
            )
        # Rule 3: Edge Decay / Low APR
        elif metrics.net_pnl_usd <= 0 or metrics.annualized_apr_pct < 5.0:
            anomaly = "EDGE_DECAY"
            severity = "P1"
            summary = (
                f"Weak or negative net return (APR={metrics.annualized_apr_pct:+.1f}%, net=${metrics.net_pnl_usd:+.2f}, "
                f"util={metrics.avg_capital_utilization_pct:.1f}%). Re-evaluate entry threshold, lookback, or symbol universe."
            )
        # Rule 4: Drawdown Breach
        elif metrics.max_drawdown_pct > 8.0:
            anomaly = "DRAWDOWN_BREACH"
            severity = "P0"
            summary = f"Max drawdown {metrics.max_drawdown_pct:.2f}% exceeds 8.0% budget. Reduce leverage or per_position_weight."

        eid = f"FB-{sha256_obj({'s': cfg.strategy_id, 't': now, 'a': anomaly})[:10]}"
        logger.info("Telemetry diagnosis [%s/%s]: %s", severity, anomaly, summary)
        return FeedbackEvent(
            event_id=eid,
            ts_ms=now,
            strategy_id=cfg.strategy_id,
            severity=severity,
            anomaly_type=anomaly,
            summary=summary,
            metrics_snapshot=snap,
        )
