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
from alphaloop.platform.data_foundation.models import CNAshareMacroPoint, CanonicalBar

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

# 17 liquid CN A-Share broad, dividend/low-vol, growth, sector & CGB Treasury ETFs
# (multi-year, zero single-stock survivorship bias)
DEFAULT_CN_ASHARE_ETF_UNIVERSE: tuple[str, ...] = (
    "510300.SS",  # CSI 300 ETF (Mainboard ±10%)
    "510500.SS",  # CSI 500 ETF (Mainboard ±10%)
    "159915.SZ",  # ChiNext ETF (ChiNext ±20%)
    "588000.SS",  # STAR 50 ETF (STAR ±20%)
    "510050.SS",  # SSE 50 ETF (Mainboard ±10%)
    "512100.SS",  # CSI 1000 ETF (Mainboard ±10%)
    "510880.SS",  # SSE Dividend ETF (Mainboard ±10%, 10y defensive equity carry)
    "512890.SS",  # CSI Dividend Low-Vol ETF (Mainboard ±10%, 8y defensive equity)
    "515080.SS",  # CSI Dividend ETF (Mainboard ±10%, 7y defensive equity)
    "512800.SS",  # CSI Bank ETF (Mainboard ±10%, 9y high-dividend value)
    "512880.SS",  # Securities ETF (Mainboard ±10%)
    "512010.SS",  # Healthcare ETF (Mainboard ±10%)
    "512480.SS",  # Semiconductor ETF (Mainboard ±10%)
    "518880.SS",  # Domestic CNY Gold ETF (ONLY for A1 historical attribution audit; EXCLUDED from A2!)
    "511010.SS",  # 5Y China Govt Bond ETF (Defensive sovereign bond)
    "511260.SS",  # 10Y China Govt Bond ETF (Defensive sovereign bond)
    "511380.SS",  # CSI Convertible Bond ETF (Hybrid defensive credit)
)

CN_ASHARE_DEFENSIVE_BOND_ETFS: tuple[str, ...] = ("511010.SS", "511260.SS")
CN_ASHARE_DIVIDEND_ETFS: tuple[str, ...] = ("510880.SS", "512890.SS", "515080.SS", "512800.SS")
CN_ASHARE_BROAD_EQUITY_ETFS: tuple[str, ...] = (
    "510300.SS",
    "510050.SS",
    "510500.SS",
    "512100.SS",
    "159915.SZ",
    "588000.SS",
)
# Strictly pure domestic CN Stock + Dividend + Sovereign Bond universe (ZERO Gold ETF 518880.SS!)
CN_ASHARE_PURE_STOCK_BOND_UNIVERSE: tuple[str, ...] = (
    "510300.SS",
    "510050.SS",
    "510500.SS",
    "512100.SS",
    "159915.SZ",
    "588000.SS",
    "510880.SS",
    "512890.SS",
    "515080.SS",
    "512800.SS",
    "511010.SS",
    "511260.SS",
)
CN_ASHARE_RISK_ETFS: tuple[str, ...] = (
    "510300.SS",
    "510500.SS",
    "159915.SZ",
    "588000.SS",
    "510050.SS",
    "512100.SS",
    "510880.SS",
    "512890.SS",
    "515080.SS",
    "512800.SS",
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


def _cst_date_open_ms(date_str: str) -> int:
    """Parse YYYY-MM-DD (or 'YYYY-MM-DD HH:MM:SS') into 09:30 CST epoch milliseconds."""
    day_part = str(date_str).strip()[:10]
    dt = datetime.strptime(day_part, "%Y-%m-%d").replace(hour=9, minute=30, second=0, tzinfo=_CST_TZ)
    return int(dt.timestamp() * 1000.0)


def _cst_post_close_avail_ms(date_str: str, *, hour: int = 18, minute: int = 0, plus_days: int = 0) -> int:
    """Resolve strict post-close CST availability timestamp (never <= 15:00 CST on trade_date)."""
    day_part = str(date_str).strip()[:10]
    base = datetime.strptime(day_part, "%Y-%m-%d").replace(tzinfo=_CST_TZ) + timedelta(days=plus_days)
    avail = base.replace(hour=hour, minute=minute, second=0)
    return int(avail.timestamp() * 1000.0)


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
        if s in CN_ASHARE_DEFENSIVE_BOND_ETFS:
            tags = ("cn_ashare", "defensive_bond")
        elif s in CN_ASHARE_DIVIDEND_ETFS:
            tags = ("cn_ashare", "dividend_value")
        else:
            tags = ("cn_ashare", "etf")
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
    """Pure stdlib CN A-Share multi-year OHLCV & native domestic macro/flow PIT collector."""

    YAHOO_HOSTS: tuple[str, ...] = (
        "https://query1.finance.yahoo.com",
        "https://query2.finance.yahoo.com",
    )
    EASTMONEY_DATA_URL: str = "https://datacenter-web.eastmoney.com/api/data/v1/get"

    def __init__(self, client: RateLimitedHTTPClient | None = None) -> None:
        self.client = client or RateLimitedHTTPClient(default_timeout=15.0)

    def parse_chart_payload(
        self,
        symbol: str,
        payload: Mapping[str, Any],
        *,
        venue: str = "cn_a_share",
        use_adj_close: bool = True,
        now_ms: int | None = None,
    ) -> list[CanonicalBar]:
        """Parse Yahoo Finance chart JSON into strictly validated PIT CN A-Share CanonicalBars.

        RB-13: filters out zero-volume / unclosed holiday/intraday stub bars (`volume <= 0` or
        `ts_avail_ms > now_ms`) so all ETFs align cleanly on genuine completed trading days.
        Phase D: applies symmetric proportional total-return `adjclose` scaling across all CN ETFs
        (matching `YahooGoldMacroCollector`) so cash dividend distributions on high-yield & broad
        ETFs (including `510300.SS` benchmark) never create artificial ex-dividend crash bars.
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
        adj_objs = indicators.get("adjclose") or []
        adj_list = (
            adj_objs[0].get("adjclose") or []
            if isinstance(adj_objs, list) and adj_objs and isinstance(adj_objs[0], Mapping)
            else []
        )

        sym = str(symbol).strip().upper()
        fingerprint = sha256_obj(
            {"collector": "CNAsharePublicCollector", "symbol": sym, "n": len(timestamps), "adj": bool(use_adj_close)}
        )
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

            # Proportional total-return adjustment when Yahoo adjclose is present and valid
            if use_adj_close and i < len(adj_list) and adj_list[i] is not None:
                try:
                    adj_f = float(adj_list[i])
                except (TypeError, ValueError):
                    adj_f = float("nan")
                if math.isfinite(adj_f) and adj_f > 0.0:
                    scale = adj_f / c_f
                    o_f *= scale
                    h_f *= scale
                    l_f *= scale
                    c_f = adj_f

            hi = max(o_f, h_f, l_f, c_f)
            lo = min(o_f, h_f, l_f, c_f)

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
        range_str: str = "10y",
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

    @staticmethod
    def parse_eastmoney_cgb_payload(
        payload: Mapping[str, Any],
        *,
        now_ms: int | None = None,
    ) -> list[CNAshareMacroPoint]:
        """Parse ChinaBond CGB sovereign yields (2Y/5Y/10Y/30Y + 10Y-2Y slope) with strict 18:00 CST PIT."""
        res = payload.get("result")
        if not isinstance(res, Mapping):
            return []
        rows = res.get("data")
        if not isinstance(rows, list):
            return []
        cutoff_ms = int(now_ms) if now_ms is not None else utc_now_ms()
        fp = sha256_obj({"collector": "eastmoney_cgb", "n": len(rows)})
        field_map = (
            ("EMM00588704", "CN_CGB_2Y"),
            ("EMM00166462", "CN_CGB_5Y"),
            ("EMM00166466", "CN_CGB_10Y"),
            ("EMM00166469", "CN_CGB_30Y"),
            ("EMM01276014", "CN_CGB_10Y_2Y_SLOPE"),
        )
        out: list[CNAshareMacroPoint] = []
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            raw_date = item.get("SOLAR_DATE")
            if not raw_date:
                continue
            try:
                ev_ms = _cst_date_open_ms(str(raw_date))
                # Official ChinaBond daily curve publishes ~17:30 CST -> 18:00 CST strictly after 15:00 close!
                avail_ms = _cst_post_close_avail_ms(str(raw_date), hour=18, minute=0, plus_days=0)
            except ValueError:
                continue
            if avail_ms > cutoff_ms or avail_ms <= ev_ms:
                continue
            for col, series_id in field_map:
                raw_v = item.get(col)
                if raw_v is None:
                    continue
                try:
                    val = float(raw_v)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(val):
                    continue
                out.append(
                    CNAshareMacroPoint(
                        source="eastmoney_cgb",
                        series_id=series_id,
                        ts_event_ms=ev_ms,
                        ts_avail_ms=avail_ms,
                        value=val,
                        provenance=PROVENANCE_REAL,
                        source_fingerprint=fp,
                    )
                )
        return out

    def fetch_eastmoney_cgb_yields(
        self,
        *,
        max_pages: int = 7,
        page_size: int = 500,
    ) -> list[CNAshareMacroPoint]:
        """Fetch ~10+ years of official ChinaBond CGB sovereign yields via Eastmoney public JSON API."""
        dedup: dict[tuple[str, int], CNAshareMacroPoint] = {}
        for page in range(1, max(1, int(max_pages)) + 1):
            url = (
                f"{self.EASTMONEY_DATA_URL}"
                f"?reportName=RPTA_WEB_TREASURYYIELD"
                f"&columns=SOLAR_DATE,EMM00588704,EMM00166462,EMM00166466,EMM00166469,EMM01276014"
                f"&pageNumber={page}&pageSize={int(page_size)}&sortColumns=SOLAR_DATE&sortTypes=-1"
            )
            payload = self.client.fetch_json(
                url,
                method="GET",
                timeout_sec=15.0,
                max_retries=3,
                min_interval_sec=0.25,
            )
            if not isinstance(payload, Mapping):
                break
            pts = self.parse_eastmoney_cgb_payload(payload)
            if not pts:
                break
            for pt in pts:
                dedup[(pt.series_id, pt.ts_event_ms)] = pt
            res = payload.get("result") or {}
            total_pages = int(res.get("pages") or 1)
            if page >= total_pages:
                break
        return sorted(dedup.values(), key=lambda p: (p.series_id, p.ts_event_ms))

    @staticmethod
    def parse_eastmoney_margin_payload(
        payload: Mapping[str, Any],
        *,
        now_ms: int | None = None,
    ) -> list[CNAshareMacroPoint]:
        """Parse SSE/SZSE full-market margin financing balance (`RPTA_RZRQ_LSHJ`) with strict T+1 08:30 CST PIT."""
        res = payload.get("result")
        if not isinstance(res, Mapping):
            return []
        rows = res.get("data")
        if not isinstance(rows, list):
            return []
        cutoff_ms = int(now_ms) if now_ms is not None else utc_now_ms()
        fp = sha256_obj({"collector": "eastmoney_rzrq", "n": len(rows)})
        field_map = (
            ("RZYE", "CN_MARGIN_BALANCE_RMB"),
            ("RZYEZB", "CN_MARGIN_RATIO_PCT"),
            ("RZJME", "CN_MARGIN_NET_BUY_RMB"),
            ("RQYE", "CN_SHORT_BALANCE_RMB"),
        )
        out: list[CNAshareMacroPoint] = []
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            raw_date = item.get("DIM_DATE")
            if not raw_date:
                continue
            try:
                ev_ms = _cst_date_open_ms(str(raw_date))
                # Official SSE/SZSE margin balances publish next morning ~08:30 CST (zero same-day lookahead!)
                avail_ms = _cst_post_close_avail_ms(str(raw_date), hour=8, minute=30, plus_days=1)
            except ValueError:
                continue
            if avail_ms > cutoff_ms or avail_ms <= ev_ms:
                continue
            for col, series_id in field_map:
                raw_v = item.get(col)
                if raw_v is None:
                    continue
                try:
                    val = float(raw_v)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(val):
                    continue
                if series_id in ("CN_MARGIN_BALANCE_RMB", "CN_MARGIN_RATIO_PCT") and val <= 0.0:
                    continue
                out.append(
                    CNAshareMacroPoint(
                        source="eastmoney_rzrq",
                        series_id=series_id,
                        ts_event_ms=ev_ms,
                        ts_avail_ms=avail_ms,
                        value=val,
                        provenance=PROVENANCE_REAL,
                        source_fingerprint=fp,
                    )
                )
        return out

    def fetch_eastmoney_margin_balance(
        self,
        *,
        max_pages: int = 6,
        page_size: int = 500,
    ) -> list[CNAshareMacroPoint]:
        """Fetch ~10+ years of SSE/SZSE full-market margin financing & short balances."""
        dedup: dict[tuple[str, int], CNAshareMacroPoint] = {}
        for page in range(1, max(1, int(max_pages)) + 1):
            url = (
                f"{self.EASTMONEY_DATA_URL}"
                f"?reportName=RPTA_RZRQ_LSHJ"
                f"&columns=DIM_DATE,RZYE,RZYEZB,RZJME,RQYE"
                f"&pageNumber={page}&pageSize={int(page_size)}&sortColumns=DIM_DATE&sortTypes=-1"
            )
            payload = self.client.fetch_json(
                url,
                method="GET",
                timeout_sec=15.0,
                max_retries=3,
                min_interval_sec=0.25,
            )
            if not isinstance(payload, Mapping):
                break
            pts = self.parse_eastmoney_margin_payload(payload)
            if not pts:
                break
            for pt in pts:
                dedup[(pt.series_id, pt.ts_event_ms)] = pt
            res = payload.get("result") or {}
            total_pages = int(res.get("pages") or 1)
            if page >= total_pages:
                break
        return sorted(dedup.values(), key=lambda p: (p.series_id, p.ts_event_ms))

    @staticmethod
    def parse_cn_macro_yahoo_payload(
        series_id: str,
        payload: Mapping[str, Any],
        *,
        avail_hour_cst: int = 23,
        avail_plus_days: int = 0,
        now_ms: int | None = None,
    ) -> list[CNAshareMacroPoint]:
        """Parse Yahoo daily close for CN FX (`CNY=X`) / Offshore (`^HSI`) with strict post-15:00 CST PIT."""
        chart = payload.get("chart")
        if not isinstance(chart, Mapping):
            return []
        results = chart.get("result")
        if not isinstance(results, list) or not results or not isinstance(results[0], Mapping):
            return []
        res = results[0]
        timestamps = res.get("timestamp") or []
        quotes = (res.get("indicators") or {}).get("quote") or []
        if not timestamps or not quotes or not isinstance(quotes[0], Mapping):
            return []
        closes = quotes[0].get("close") or []
        cutoff_ms = int(now_ms) if now_ms is not None else utc_now_ms()
        fp = sha256_obj({"collector": "yahoo_cn_macro", "series_id": series_id, "n": len(timestamps)})
        by_day: dict[str, CNAshareMacroPoint] = {}
        for i, raw_ts in enumerate(timestamps):
            if i >= len(closes) or raw_ts is None or closes[i] is None:
                continue
            try:
                val = float(closes[i])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(val) or val <= 0.0:
                continue
            dt_cst = datetime.fromtimestamp(int(raw_ts), tz=_CST_TZ)
            day_str = dt_cst.strftime("%Y-%m-%d")
            ev_ms = _cst_date_open_ms(day_str)
            avail_ms = _cst_post_close_avail_ms(day_str, hour=avail_hour_cst, minute=0, plus_days=avail_plus_days)
            if avail_ms > cutoff_ms or avail_ms <= ev_ms:
                continue
            by_day[day_str] = CNAshareMacroPoint(
                source="yahoo_cn_macro",
                series_id=series_id,
                ts_event_ms=ev_ms,
                ts_avail_ms=avail_ms,
                value=val,
                provenance=PROVENANCE_REAL,
                source_fingerprint=fp,
            )
        return sorted(by_day.values(), key=lambda p: p.ts_event_ms)

    def fetch_cn_fx_and_offshore(self, *, range_str: str = "10y") -> list[CNAshareMacroPoint]:
        """Fetch 10y USD/CNY (`CNY=X`, T+1 09:00 CST PIT) and Hang Seng (`^HSI`, T 17:00 CST PIT)."""
        import urllib.parse

        targets = (
            # RB-D10: Yahoo CNY=X uses Europe/London midnight bars (closes 07:00 CST BST / 08:00 CST GMT) -> T+1 09:00 CST!
            ("CNY=X", "CN_FX_USDCNY", 9, 1),
            ("^HSI", "CN_OFFSHORE_HSI", 17, 0),  # HKEX closes 16:00 CST -> 17:00 CST (strictly after 15:00 A-share close!)
        )
        out: list[CNAshareMacroPoint] = []
        for y_sym, series_id, hr, plus_d in targets:
            enc = urllib.parse.quote(y_sym, safe="")
            for host in self.YAHOO_HOSTS:
                url = f"{host}/v8/finance/chart/{enc}?range={range_str}&interval=1d"
                try:
                    payload = self.client.fetch_json(
                        url,
                        method="GET",
                        timeout_sec=15.0,
                        max_retries=3,
                        min_interval_sec=0.25,
                    )
                    if isinstance(payload, Mapping):
                        pts = self.parse_cn_macro_yahoo_payload(
                            series_id,
                            payload,
                            avail_hour_cst=hr,
                            avail_plus_days=plus_d,
                        )
                        if pts:
                            out.extend(pts)
                            break
                except Exception as exc:
                    logger.warning("Yahoo CN macro host %s failed for %s: %s", host, y_sym, exc)
                    continue
        return out
