"""Track A — CN A-Share Point-in-Time Public Market Data Collector (`cn_ashare` domain).

Strictly isolated from Web3 and Gold/Macro data pipelines (`domains/cn_ashare_lake.sqlite`).
Uses pure stdlib `RateLimitedHTTPClient` (Waterfall + Jitter + explicit timeout) to ingest
real multi-year CN A-Share broad-market, sector, and CGB Treasury ETF daily OHLCV bars with
exact China Standard Time (`Asia/Shanghai`, UTC+8) 15:00 close availability timestamps:

  - Every bar's `ts_avail_ms` is computed as 15:00 CST (`07:00 UTC`) on its CST trade date
    (strictly `>= ts_event_ms + 4h`), guaranteeing zero intraday lookahead.
  - Liquid ETFs avoid single-stock survivorship bias while natively exercising both SSE/SZSE
    Mainboard (`±10%`) and ChiNext/STAR (`±20%`) daily price-limit physics, `T+1` sell locks,
    long-only constraints, and sell-only stamp duty accounting (`COST_CN_A_SHARE`).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from alphaloop.core.provenance import PROVENANCE_REAL, sha256_obj, utc_now_ms
from alphaloop.platform.assets import (
    ASSET_CLASS_CN_A_SHARE,
    CALENDAR_CN_A_SHARE,
    COST_CN_A_SHARE,
    AssetSpec,
    TradingCalendar,
)
from alphaloop.platform.data_foundation.collectors_web3 import RateLimitedHTTPClient
from alphaloop.platform.data_foundation.models import CanonicalBar

logger = logging.getLogger(__name__)

_CST_TZ = timezone(timedelta(hours=8))
_FOUR_HOURS_MS = 4 * 3_600_000
_DAY_MS = 86_400_000

_CHINEXT_STAR_20PCT_ETFS: frozenset[str] = frozenset({"159915.SZ", "588000.SS", "588080.SS", "159949.SZ"})
_CHINEXT_STAR_PREFIXES: tuple[str, ...] = ("300", "301", "688", "689")
_BSE_PREFIXES: tuple[str, ...] = ("43", "83", "87", "920")

CALENDAR_CN_A_SHARE_20PCT = TradingCalendar(
    calendar_id="CN_CHINEXT_STAR_20PCT",
    trading_hours_per_day=CALENDAR_CN_A_SHARE.trading_hours_per_day,
    trading_days_per_year=CALENDAR_CN_A_SHARE.trading_days_per_year,
    allow_short=False,
    t_plus_days=1,
    daily_price_limit_pct=20.0,
    max_expected_gap_hours=CALENDAR_CN_A_SHARE.max_expected_gap_hours,
)

CALENDAR_CN_A_SHARE_30PCT = TradingCalendar(
    calendar_id="CN_BSE_30PCT",
    trading_hours_per_day=CALENDAR_CN_A_SHARE.trading_hours_per_day,
    trading_days_per_year=CALENDAR_CN_A_SHARE.trading_days_per_year,
    allow_short=False,
    t_plus_days=1,
    daily_price_limit_pct=30.0,
    max_expected_gap_hours=CALENDAR_CN_A_SHARE.max_expected_gap_hours,
)

# 11 liquid CN A-Share broad, growth, sector & CGB Treasury ETFs (multi-year, zero single-stock survivorship bias)
DEFAULT_CN_ASHARE_ETF_UNIVERSE: tuple[str, ...] = (
    "510300.SS",  # CSI 300 ETF (Mainboard ±10%)
    "510500.SS",  # CSI 500 ETF (Mainboard ±10%)
    "159915.SZ",  # ChiNext ETF (ChiNext ±20%)
    "588000.SS",  # STAR 50 ETF (STAR ±20%)
    "510050.SS",  # SSE 50 ETF (Mainboard ±10%)
    "512880.SS",  # Securities ETF (Mainboard ±10%)
    "512010.SS",  # Healthcare ETF (Mainboard ±10%)
    "512480.SS",  # Semiconductor ETF (Mainboard ±10%)
    "518880.SS",  # Domestic CNY Gold ETF (Mainboard ±10%)
    "511010.SS",  # 5Y China Govt Bond ETF (Defensive)
    "511260.SS",  # 10Y China Govt Bond ETF (Defensive)
)

CN_ASHARE_DEFENSIVE_BOND_ETFS: tuple[str, ...] = ("511010.SS", "511260.SS")
CN_ASHARE_RISK_ETFS: tuple[str, ...] = (
    "510300.SS",
    "510500.SS",
    "159915.SZ",
    "588000.SS",
    "510050.SS",
    "512880.SS",
    "512010.SS",
    "512480.SS",
    "518880.SS",
)


def cn_ashare_close_avail_ms(ts_event_ms: int) -> int:
    """Resolve exact 15:00 CST (07:00 UTC) post-close Point-in-Time availability timestamp."""
    dt_cst = datetime.fromtimestamp(int(ts_event_ms) / 1000.0, tz=_CST_TZ)
    close_cst = datetime(dt_cst.year, dt_cst.month, dt_cst.day, 15, 0, 0, tzinfo=_CST_TZ)
    close_ms = int(close_cst.timestamp() * 1000.0)
    # Guarantee ts_avail_ms >= ts_event_ms + 4h (A-share 4h cash session duration)
    return max(close_ms, int(ts_event_ms) + _FOUR_HOURS_MS)


def resolve_cn_ashare_calendar(symbol: str) -> TradingCalendar:
    """Assign board-accurate CN A-Share TradingCalendar (±10% Mainboard, ±20% ChiNext/STAR, ±30% BSE) (RB-12)."""
    s = str(symbol).strip().upper()
    code = s.split(".")[0]
    if s in _CHINEXT_STAR_20PCT_ETFS or code.startswith(_CHINEXT_STAR_PREFIXES):
        return CALENDAR_CN_A_SHARE_20PCT
    if code.startswith(_BSE_PREFIXES):
        return CALENDAR_CN_A_SHARE_30PCT
    return CALENDAR_CN_A_SHARE


def default_cn_ashare_specs(
    symbols: Sequence[str] = DEFAULT_CN_ASHARE_ETF_UNIVERSE,
    venue: str = "cn_a_share",
) -> dict[str, AssetSpec]:
    """Build canonical CN A-Share AssetSpecs with board-accurate limits, T+1, long-only, and COST_CN_A_SHARE."""
    out: dict[str, AssetSpec] = {}
    for sym in symbols:
        s = str(sym).strip().upper()
        tags = ("cn_ashare", "defensive_bond") if s in CN_ASHARE_DEFENSIVE_BOND_ETFS else ("cn_ashare", "etf")
        out[s] = AssetSpec(
            symbol=s,
            asset_class=ASSET_CLASS_CN_A_SHARE,
            venue=venue,
            calendar=resolve_cn_ashare_calendar(s),
            cost_model=COST_CN_A_SHARE,
            max_leverage_cap=1.0,
            base_currency=s.split(".")[0],
            quote_currency="CNY",
            tags=tags,
            contract_multiplier=1.0,
        )
    return out


class CNAsharePublicCollector:
    """Pure stdlib CN A-Share multi-year daily OHLCV collector with strict CST PIT timestamps."""

    YAHOO_HOSTS: tuple[str, ...] = (
        "https://query1.finance.yahoo.com",
        "https://query2.finance.yahoo.com",
    )

    def __init__(self, client: RateLimitedHTTPClient | None = None) -> None:
        self.client = client or RateLimitedHTTPClient(default_timeout=15.0)

    def parse_chart_payload(
        self,
        symbol: str,
        payload: Mapping[str, Any],
        *,
        venue: str = "cn_a_share",
        now_ms: int | None = None,
    ) -> list[CanonicalBar]:
        """Parse Yahoo Finance chart JSON into strictly validated PIT CN A-Share CanonicalBars.

        RB-13: filters out zero-volume / unclosed holiday/intraday stub bars (`volume <= 0` or
        `ts_avail_ms > now_ms`) so all ETFs align cleanly on genuine completed trading days.
        """
        chart = payload.get("chart")
        if not isinstance(chart, Mapping):
            return []
        results = chart.get("result")
        if not isinstance(results, list) or not results or not isinstance(results[0], Mapping):
            return []
        res = results[0]
        timestamps = res.get("timestamp") or []
        indicators = res.get("indicators") or {}
        quotes = indicators.get("quote") or []
        if not timestamps or not quotes or not isinstance(quotes[0], Mapping):
            return []
        q = quotes[0]
        opens = q.get("open") or []
        highs = q.get("high") or []
        lows = q.get("low") or []
        closes = q.get("close") or []
        vols = q.get("volume") or []

        sym = str(symbol).strip().upper()
        fingerprint = sha256_obj({"collector": "CNAsharePublicCollector", "symbol": sym, "n": len(timestamps)})
        cutoff_ms = int(now_ms) if now_ms is not None else utc_now_ms()

        by_cst_day: dict[str, CanonicalBar] = {}
        for i, raw_ts in enumerate(timestamps):
            if i >= len(closes) or raw_ts is None:
                continue
            c_raw = closes[i]
            o_raw = opens[i] if i < len(opens) else None
            h_raw = highs[i] if i < len(highs) else None
            l_raw = lows[i] if i < len(lows) else None
            v_raw = vols[i] if i < len(vols) else None
            if c_raw is None or o_raw is None or h_raw is None or l_raw is None or v_raw is None:
                continue
            try:
                o_f = float(o_raw)
                h_f = float(h_raw)
                l_f = float(l_raw)
                c_f = float(c_raw)
                v_f = float(v_raw)
            except (TypeError, ValueError):
                continue
            if not (
                math.isfinite(o_f)
                and math.isfinite(h_f)
                and math.isfinite(l_f)
                and math.isfinite(c_f)
                and math.isfinite(v_f)
            ):
                continue
            # RB-13: reject non-positive price or zero-volume phantom/holiday stub bars
            if min(o_f, h_f, l_f, c_f) <= 0.0 or v_f <= 0.0:
                continue
            hi = max(o_f, h_f, l_f, c_f)
            lo = min(o_f, h_f, l_f, c_f)
            ts_ev_ms = int(raw_ts) * 1000
            # Normalize event timestamp to 09:30 CST (01:30 UTC) session open for clean daily alignment
            dt_cst = datetime.fromtimestamp(ts_ev_ms / 1000.0, tz=_CST_TZ)
            day_key = dt_cst.strftime("%Y-%m-%d")
            open_cst_ms = int(
                datetime(dt_cst.year, dt_cst.month, dt_cst.day, 9, 30, 0, tzinfo=_CST_TZ).timestamp() * 1000.0
            )
            avail_ms = cn_ashare_close_avail_ms(open_cst_ms)
            # Reject unclosed in-progress trading session stubs whose 15:00 CST close is in the future
            if avail_ms > cutoff_ms:
                continue
            notional_cny = max(0.0, v_f * c_f)

            by_cst_day[day_key] = CanonicalBar(
                asset_class=ASSET_CLASS_CN_A_SHARE,
                venue=venue,
                symbol=sym,
                timeframe_h=24.0,
                timeframe_sec=86_400,
                ts_event_ms=open_cst_ms,
                ts_avail_ms=avail_ms,
                open=o_f,
                high=hi,
                low=lo,
                close=c_f,
                volume_usd=notional_cny,
                carry_rate_period=0.0,
                carry_interval_h=24.0,
                spread_bps=5.0,
                provenance=PROVENANCE_REAL,
                session_tag="CN_A_SHARE_DAY",
                source_fingerprint=fingerprint,
            )

        return sorted(by_cst_day.values(), key=lambda b: b.ts_event_ms)

    def fetch_daily_bars(
        self,
        symbol: str,
        *,
        range_str: str = "5y",
        venue: str = "cn_a_share",
    ) -> list[CanonicalBar]:
        """Fetch real multi-year CN A-Share daily OHLCV via host waterfall + rate limiting."""
        sym = str(symbol).strip().upper()
        last_err: Exception | None = None
        for host in self.YAHOO_HOSTS:
            url = f"{host}/v8/finance/chart/{sym}?range={range_str}&interval=1d"
            try:
                payload = self.client.fetch_json(
                    url,
                    method="GET",
                    timeout_sec=15.0,
                    max_retries=3,
                    min_interval_sec=0.25,
                )
                if isinstance(payload, Mapping):
                    bars = self.parse_chart_payload(sym, payload, venue=venue)
                    if bars:
                        return bars
            except Exception as exc:
                last_err = exc
                logger.warning("CNAsharePublicCollector host %s failed for %s: %s", host, sym, exc)
                continue
        if last_err is not None:
            raise RuntimeError(f"All CN A-Share hosts failed for {sym}: {last_err}") from last_err
        return []
