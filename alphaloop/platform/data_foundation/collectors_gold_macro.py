"""100% Real Public Gold & Macro Data Collectors + PIT Event Calendar (STD-34 Compliant).

Verified live-probed public sources (zero API keys, zero synthetic fallback):
1. Yahoo Finance v8 (`YahooGoldMacroCollector`):
   - All 13 symbols (`GOLD_MACRO_SYMBOLS_V4`): `1d` (5y–10y, total-return `adjclose` scaling on
     dividend-bearing ETFs `UUP/TIP/IEF/TLT/GDX`) and `1h -> 4h` (`730d`, `min_subbars=2`)
   - Drops live non-minute ticks (`ts_s % 60 != 0`) and ETF/Index `:00 V=0` 1800s closing prints
   - Preserves exact `:30`/`:20` timestamps (`ts_avail_ms = ts_event_ms + HOUR_MS`, zero lookahead!)
   - Applies COMEX contract multipliers (`GC=F=100.0`, `SI=F=5000.0`) and deterministic `session_tag`
2. US Treasury Official TIPS Real Yield Curve CSV (`USTreasuryRealYieldCollector`):
   - `home.treasury.gov` `2021..2026` (`5Y/7Y/10Y/20Y/30Y`), `utf-8-sig`, safe `None` on `N/A`/`ND`,
     ascending sort, `ts_avail_ms = ts_event_ms + 22 * HOUR_MS`
3. Federal Reserve Official FOMC Calendar (`FedFOMCCalendarCollector`):
   - Live HTML parser for `federalreserve.gov/monetarypolicy/fomccalendars.htm` (`2021–2027`),
     skips `notation vote`/`cancelled`/`unscheduled`, `is_sep_meeting = ("*" in dt_text)` ONLY,
     `ZoneInfo("America/New_York")` 14:00 local time with pure-Python US DST fallback
4. BLS Public API v2 + Deterministic PIT NFP/CPI Release Calendar (`BLSMacroEventCollector`):
   - Single batched POST (`CUSR0000SA0`, `CES0000000001`, `2020..2026`) + optional 12h JSON cache
   - Handles `2025-M10 value="-"` (footnote `X` lapse in appropriations) cleanly
   - Client-side MoM/change + strictly causal trailing 12-month `surprise_z`
   - Statutory BLS NFP release formula + curated official `2021–2026` CPI schedule
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
import tempfile
import time
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from alphaloop.core.provenance import PROVENANCE_REAL, utc_now_ms
from alphaloop.platform.assets import (
    ASSET_CLASS_GOLD_TRADFI,
    ASSET_CLASS_MACRO_INDEX,
    DEFAULT_CONTRACT_MULTIPLIERS,
    GOLD_MACRO_SYMBOLS_V4,
)
from alphaloop.platform.data_foundation.collectors_web3 import RateLimitedHTTPClient
from alphaloop.platform.data_foundation.models import (
    DAY_MS,
    HOUR_MS,
    CanonicalBar,
    MacroEventRelease,
    MacroYieldPoint,
    classify_tradfi_session_tag,
    resample_bars,
)

logger = logging.getLogger(__name__)

YAHOO_V4_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AlphaLoop-PlatformV4/1.0"

# Dividend-distributing ETFs where 5y raw close lags total-return adjclose by 11-18pp
DIVIDEND_ADJ_CLOSE_SYMBOLS = frozenset({"UUP", "TIP", "IEF", "TLT", "GDX"})

_MONTH_NAME_MAP: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

# Verified official BLS CPI-U release dates (08:30 America/New_York) keyed by reference period YYYY-MM.
# Note: 2025-10 was omitted/unavailable by BLS (footnote X: 2025 lapse in appropriations).
OFFICIAL_CPI_RELEASE_DATES: dict[str, str] = {
    # 2021 releases
    "2020-12": "2021-01-13",
    "2021-01": "2021-02-10",
    "2021-02": "2021-03-10",
    "2021-03": "2021-04-13",
    "2021-04": "2021-05-12",
    "2021-05": "2021-06-10",
    "2021-06": "2021-07-13",
    "2021-07": "2021-08-11",
    "2021-08": "2021-09-14",
    "2021-09": "2021-10-13",
    "2021-10": "2021-11-10",
    "2021-11": "2021-12-10",
    # 2022 releases
    "2021-12": "2022-01-12",
    "2022-01": "2022-02-10",
    "2022-02": "2022-03-10",
    "2022-03": "2022-04-12",
    "2022-04": "2022-05-11",
    "2022-05": "2022-06-10",
    "2022-06": "2022-07-13",
    "2022-07": "2022-08-10",
    "2022-08": "2022-09-13",
    "2022-09": "2022-10-13",
    "2022-10": "2022-11-10",
    "2022-11": "2022-12-13",
    # 2023 releases
    "2022-12": "2023-01-12",
    "2023-01": "2023-02-14",
    "2023-02": "2023-03-14",
    "2023-03": "2023-04-12",
    "2023-04": "2023-05-10",
    "2023-05": "2023-06-13",
    "2023-06": "2023-07-12",
    "2023-07": "2023-08-10",
    "2023-08": "2023-09-13",
    "2023-09": "2023-10-12",
    "2023-10": "2023-11-14",
    "2023-11": "2023-12-12",
    # 2024 releases
    "2023-12": "2024-01-11",
    "2024-01": "2024-02-13",
    "2024-02": "2024-03-12",
    "2024-03": "2024-04-10",
    "2024-04": "2024-05-15",
    "2024-05": "2024-06-12",
    "2024-06": "2024-07-11",
    "2024-07": "2024-08-14",
    "2024-08": "2024-09-11",
    "2024-09": "2024-10-10",
    "2024-10": "2024-11-13",
    "2024-11": "2024-12-11",
    # 2025 releases (2025-09 delayed to Oct 24; 2025-10 unavailable; 2025-11 released Dec 18)
    "2024-12": "2025-01-15",
    "2025-01": "2025-02-12",
    "2025-02": "2025-03-12",
    "2025-03": "2025-04-10",
    "2025-04": "2025-05-13",
    "2025-05": "2025-06-11",
    "2025-06": "2025-07-15",
    "2025-07": "2025-08-12",
    "2025-08": "2025-09-11",
    "2025-09": "2025-10-24",
    "2025-11": "2025-12-18",
    # 2026 releases
    "2025-12": "2026-01-13",
    "2026-01": "2026-02-13",
    "2026-02": "2026-03-11",
    "2026-03": "2026-04-10",
    "2026-04": "2026-05-12",
    "2026-05": "2026-06-10",
    "2026-06": "2026-07-14",
    "2026-07": "2026-08-12",
    "2026-08": "2026-09-11",
    "2026-09": "2026-10-14",
    "2026-10": "2026-11-12",
    "2026-11": "2026-12-10",
}


def _nth_weekday_of_month(year: int, month: int, weekday: int, nth: int) -> date:
    """Return the nth (1-based) occurrence of weekday (Mon=0..Sun=6) in (year, month)."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (nth - 1))


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    last_day = nxt - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _is_us_ny_dst_fallback(year: int, month: int, day: int, hour: int, minute: int = 0) -> bool:
    """Pure-Python US Eastern DST rule (2nd Sunday in March 02:00 .. 1st Sunday in November 02:00)."""
    dst_start_d = _nth_weekday_of_month(year, 3, 6, 2)  # Sunday=6
    dst_end_d = _nth_weekday_of_month(year, 11, 6, 1)
    cur = (month, day, hour, minute)
    start_t = (dst_start_d.month, dst_start_d.day, 2, 0)
    end_t = (dst_end_d.month, dst_end_d.day, 2, 0)
    return start_t <= cur < end_t


def ny_local_to_utc_ms(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    """Convert America/New_York local wall time to UTC milliseconds with pure-Python DST fallback."""
    try:
        tz = ZoneInfo("America/New_York")
        dt_local = datetime(year, month, day, hour, minute, 0, tzinfo=tz)
        return int(dt_local.astimezone(timezone.utc).timestamp() * 1000)
    except ZoneInfoNotFoundError:
        offset_h = -4 if _is_us_ny_dst_fallback(year, month, day, hour, minute) else -5
        tz_fb = timezone(timedelta(hours=offset_h))
        dt_local = datetime(year, month, day, hour, minute, 0, tzinfo=tz_fb)
        return int(dt_local.astimezone(timezone.utc).timestamp() * 1000)


def _observed_fixed_holiday(d: date) -> date:
    if d.weekday() == 5:  # Saturday -> Friday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday -> Monday
        return d + timedelta(days=1)
    return d


def us_federal_holidays(year: int) -> set[date]:
    """Return US federal holidays (including observed dates) for a given calendar year."""
    hols: set[date] = {
        _observed_fixed_holiday(date(year, 1, 1)),
        _nth_weekday_of_month(year, 1, 0, 3),    # MLK Day
        _nth_weekday_of_month(year, 2, 0, 3),    # Presidents Day
        _last_weekday_of_month(year, 5, 0),      # Memorial Day
        _observed_fixed_holiday(date(year, 7, 4)),
        _nth_weekday_of_month(year, 9, 0, 1),    # Labor Day
        _nth_weekday_of_month(year, 10, 0, 2),   # Columbus Day
        _observed_fixed_holiday(date(year, 11, 11)),
        _nth_weekday_of_month(year, 11, 3, 4),   # Thanksgiving
        _observed_fixed_holiday(date(year, 12, 25)),
    }
    if year >= 2021:
        hols.add(_observed_fixed_holiday(date(year, 6, 19)))  # Juneteenth
    # If next year's Jan 1 falls on Saturday, Dec 31 of `year` is observed
    next_ny = _observed_fixed_holiday(date(year + 1, 1, 1))
    if next_ny.year == year:
        hols.add(next_ny)
    return hols


def bls_nfp_release_date(ref_year: int, ref_month: int) -> date:
    """Statutory BLS Employment Situation (NFP) release date: 3rd Friday after Sun-Sat week containing the 12th."""
    d12 = date(ref_year, ref_month, 12)
    sat_end = d12 + timedelta(days=(5 - d12.weekday()) % 7)
    rel = sat_end + timedelta(days=20)
    if rel.month == 1 and rel.day <= 3:
        rel = rel + timedelta(days=7)
    hols = us_federal_holidays(rel.year)
    while rel in hols or rel.weekday() >= 5:
        rel -= timedelta(days=1)
    return rel


def bls_cpi_release_date(ref_year: int, ref_month: int) -> date | None:
    """Return official curated 2021-2026 CPI release date (None for 2025-10 lapse), else algorithmic fallback."""
    key = f"{ref_year:04d}-{ref_month:02d}"
    if key == "2025-10":
        return None
    if key in OFFICIAL_CPI_RELEASE_DATES:
        return date.fromisoformat(OFFICIAL_CPI_RELEASE_DATES[key])
    # Algorithmic fallback for >=2027 or <2020-12: 2nd Wednesday of the following month (shifted off holidays)
    rel_y = ref_year + (1 if ref_month == 12 else 0)
    rel_m = 1 if ref_month == 12 else (ref_month + 1)
    cand = _nth_weekday_of_month(rel_y, rel_m, 2, 2)  # 2nd Wednesday
    hols = us_federal_holidays(rel_y)
    while cand in hols or cand.weekday() >= 5:
        cand += timedelta(days=1)
    return cand


def _parse_optional_float(val: Any) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.upper() in ("N/A", "NA", "ND", "-", "--", "NULL", "NONE"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


class YahooGoldMacroCollector:
    """Yahoo Finance v8 collector for all 13 Gold & Macro symbols (`1d` 5y-10y & `1h->4h` 730d)."""

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(self, client: RateLimitedHTTPClient | None = None) -> None:
        self.client = client or RateLimitedHTTPClient(user_agent=YAHOO_V4_USER_AGENT)

    @staticmethod
    def parse_chart_payload(
        payload: Mapping[str, Any],
        *,
        symbol: str,
        asset_class: str | None = None,
        interval: str = "1d",
        range_str: str = "5y",
        contract_multiplier: float | None = None,
        use_adj_close: bool | None = None,
        now_ms: int | None = None,
    ) -> list[CanonicalBar]:
        """Parse raw Yahoo v8 chart JSON into strictly causal CanonicalBar sequence."""
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        ac = asset_class or (ASSET_CLASS_GOLD_TRADFI if symbol in ("GC=F", "GLD") else ASSET_CLASS_MACRO_INDEX)
        mult = (
            float(contract_multiplier)
            if contract_multiplier is not None
            else float(DEFAULT_CONTRACT_MULTIPLIERS.get(symbol, 1.0))
        )
        adj_enabled = (symbol in DIVIDEND_ADJ_CLOSE_SYMBOLS) if use_adj_close is None else bool(use_adj_close)
        tf_h = 24.0 if interval == "1d" else 1.0
        tf_ms = DAY_MS if interval == "1d" else HOUR_MS
        tf_sec = int(round(tf_h * 3600.0))

        results = (payload.get("chart") or {}).get("result") or []
        if not results:
            return []
        res = results[0]
        meta = res.get("meta") or {}
        inst_type = str(meta.get("instrumentType") or "")
        timestamps = res.get("timestamp") or []
        indicators = res.get("indicators") or {}
        quotes = (indicators.get("quote") or [{}])[0]
        opens = quotes.get("open") or []
        highs = quotes.get("high") or []
        lows = quotes.get("low") or []
        closes = quotes.get("close") or []
        vols = quotes.get("volume") or []
        adj_list = ((indicators.get("adjclose") or [{}])[0]).get("adjclose") or []

        n = min(len(timestamps), len(opens), len(highs), len(lows), len(closes), len(vols))
        seen: dict[int, CanonicalBar] = {}
        fp = f"yahoo:v8:{symbol}:{interval}:{range_str}"

        for i in range(n):
            o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], vols[i]
            if o is None or h is None or l is None or c is None:
                continue
            o_f, h_f, l_f, c_f = float(o), float(h), float(l), float(c)
            if not (math.isfinite(o_f) and math.isfinite(h_f) and math.isfinite(l_f) and math.isfinite(c_f)):
                continue
            if o_f <= 0.0 or h_f <= 0.0 or l_f <= 0.0 or c_f <= 0.0:
                continue
            ts_int = int(timestamps[i])
            v_f = max(0.0, float(v or 0.0))

            if interval == "1h":
                # Drop trailing live regularMarketTime snapshot ticks (non-zero seconds)
                if ts_int % 60 != 0:
                    continue
                # Drop Yahoo ETF/Index :00 V=0 closing-print ticks 1800s after preceding :30 bar
                if (
                    i > 0
                    and (ts_int - int(timestamps[i - 1])) == 1800
                    and (ts_int % 3600) == 0
                    and v_f == 0.0
                    and inst_type.upper() in ("ETF", "EQUITY", "INDEX")
                ):
                    continue
                # Preserve exact :30/:20 bar open time (ZERO 30-40min lookahead!)
                ts_ev = ts_int * 1000
                ts_av = ts_ev + HOUR_MS
            else:
                ts_ev = ((ts_int * 1000) // DAY_MS) * DAY_MS
                ts_av = ts_ev + DAY_MS

            if ts_av > ref_now:
                continue
            if ts_ev in seen:
                continue

            vol_usd = v_f * c_f * mult
            if interval == "1d" and adj_enabled and i < len(adj_list) and adj_list[i] is not None:
                adj_f = float(adj_list[i])
                if math.isfinite(adj_f) and adj_f > 0.0:
                    scale = adj_f / c_f
                    o_f *= scale
                    h_f *= scale
                    l_f *= scale
                    c_f = adj_f

            h_clean = max(h_f, o_f, c_f)
            l_clean = min(l_f, o_f, c_f)
            seen[ts_ev] = CanonicalBar(
                asset_class=ac,
                venue="yahoo",
                symbol=symbol,
                timeframe_h=tf_h,
                ts_event_ms=ts_ev,
                ts_avail_ms=ts_av,
                open=o_f,
                high=h_clean,
                low=l_clean,
                close=c_f,
                volume_usd=vol_usd,
                carry_rate_period=0.0,
                carry_interval_h=tf_h,
                spread_bps=0.0,
                provenance=PROVENANCE_REAL,
                source_fingerprint=fp,
                timeframe_sec=tf_sec,
                session_tag=classify_tradfi_session_tag(ts_ev, symbol, inst_type, tf_h),
                premium_basis_bps=0.0,
                contract_multiplier=mult,
            )
        return sorted(seen.values(), key=lambda b: b.ts_event_ms)

    def fetch_bars(
        self,
        symbol: str,
        *,
        asset_class: str | None = None,
        range_str: str = "5y",
        interval: str = "1d",
        contract_multiplier: float | None = None,
        use_adj_close: bool | None = None,
        now_ms: int | None = None,
    ) -> list[CanonicalBar]:
        encoded = urllib.parse.quote(symbol, safe="")
        url = f"{self.BASE_URL}/{encoded}?range={range_str}&interval={interval}"
        data = self.client.fetch_json(url, max_retries=4, min_interval_sec=0.15)
        if not isinstance(data, Mapping):
            return []
        return self.parse_chart_payload(
            data,
            symbol=symbol,
            asset_class=asset_class,
            interval=interval,
            range_str=range_str,
            contract_multiplier=contract_multiplier,
            use_adj_close=use_adj_close,
            now_ms=now_ms,
        )

    def fetch_4h_bars(
        self,
        symbol: str,
        *,
        asset_class: str | None = None,
        range_str: str = "730d",
        min_subbars: int = 2,
        contract_multiplier: float | None = None,
        now_ms: int | None = None,
    ) -> tuple[list[CanonicalBar], list[CanonicalBar]]:
        """Fetch Yahoo 1h history and causally resample to 4h (`min_subbars=2` empirically verified)."""
        bars_1h = self.fetch_bars(
            symbol,
            asset_class=asset_class,
            range_str=range_str,
            interval="1h",
            contract_multiplier=contract_multiplier,
            use_adj_close=False,
            now_ms=now_ms,
        )
        bars_4h = resample_bars(bars_1h, 4.0, require_complete=True, min_subbars=min_subbars)
        return bars_1h, bars_4h


class USTreasuryRealYieldCollector:
    """US Treasury official Daily Par Real Yield Curve Rates CSV collector (`home.treasury.gov`)."""

    URL_TEMPLATE = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        "daily-treasury-rates.csv/{year}/all?type=daily_treasury_real_yield_curve"
        "&field_tdr_date_value={year}&page&_format=csv"
    )

    def __init__(self, client: RateLimitedHTTPClient | None = None) -> None:
        self.client = client or RateLimitedHTTPClient(user_agent=YAHOO_V4_USER_AGENT)

    @staticmethod
    def parse_csv_text(
        csv_text: str,
        *,
        source_fingerprint: str = "us_treasury:real_yield_curve",
        now_ms: int | None = None,
    ) -> list[MacroYieldPoint]:
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        cleaned = csv_text.lstrip("\ufeff")
        reader = csv.DictReader(io.StringIO(cleaned))
        by_ts: dict[int, MacroYieldPoint] = {}

        for row in reader:
            raw_date = (row.get("Date") or "").strip()
            if not raw_date:
                continue
            try:
                d = datetime.strptime(raw_date, "%m/%d/%Y").date()
            except ValueError:
                continue
            r5 = _parse_optional_float(row.get("5 YR"))
            r7 = _parse_optional_float(row.get("7 YR"))
            r10 = _parse_optional_float(row.get("10 YR"))
            r20 = _parse_optional_float(row.get("20 YR"))
            r30 = _parse_optional_float(row.get("30 YR"))
            if all(v is None for v in (r5, r7, r10, r20, r30)):
                continue
            ts_ev = int(datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
            # US Treasury publishes ~15:30-16:30 ET; 22:00 UTC is strictly causal in both EST and EDT
            ts_av = ts_ev + 22 * HOUR_MS
            if ts_av > ref_now:
                continue
            by_ts[ts_ev] = MacroYieldPoint(
                source="us_treasury",
                ts_event_ms=ts_ev,
                ts_avail_ms=ts_av,
                real_5y=r5,
                real_7y=r7,
                real_10y=r10,
                real_20y=r20,
                real_30y=r30,
                provenance=PROVENANCE_REAL,
                source_fingerprint=source_fingerprint,
            )
        return [by_ts[ts] for ts in sorted(by_ts)]

    def fetch_years(
        self,
        start_year: int = 2021,
        end_year: int = 2026,
        *,
        now_ms: int | None = None,
    ) -> list[MacroYieldPoint]:
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        fp = f"us_treasury:real_yield_curve:{start_year}-{end_year}"
        by_ts: dict[int, MacroYieldPoint] = {}
        for yr in range(int(start_year), int(end_year) + 1):
            url = self.URL_TEMPLATE.format(year=yr)
            blob = self.client.fetch_bytes(url, max_retries=4, min_interval_sec=0.2, allow_404=True)
            if blob is None:
                continue
            pts = self.parse_csv_text(blob.decode("utf-8-sig"), source_fingerprint=fp, now_ms=ref_now)
            for p in pts:
                by_ts[p.ts_event_ms] = p
        return [by_ts[ts] for ts in sorted(by_ts)]


class FedFOMCCalendarCollector:
    """Federal Reserve official FOMC meeting calendar HTML parser (`2021–2027`)."""

    URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    _PANEL_SPLIT_RE = re.compile(
        r'<div[^>]*class="[^"]*\bpanel-heading\b[^"]*"[^>]*>(.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    )
    _YEAR_HEAD_RE = re.compile(r"\b(20\d{2})\s+FOMC\s+Meetings\b", re.IGNORECASE)
    _ROW_PAIR_RE = re.compile(
        r'<div[^>]*class="[^"]*\bfomc-meeting__month\b[^"]*"[^>]*>(.*?)</div>\s*'
        r'<div[^>]*class="[^"]*\bfomc-meeting__date\b[^"]*"[^>]*>(.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    )
    _SUP_STRIP_RE = re.compile(r"<sup\b[^>]*>.*?</sup>", re.IGNORECASE | re.DOTALL)
    _TAG_STRIP_RE = re.compile(r"<[^>]+>")
    _SKIP_KEYWORDS = ("notation vote", "cancelled", "canceled", "unscheduled")

    def __init__(self, client: RateLimitedHTTPClient | None = None) -> None:
        self.client = client or RateLimitedHTTPClient(user_agent=YAHOO_V4_USER_AGENT)

    @classmethod
    def _clean_text(cls, html_frag: str) -> str:
        no_sup = cls._SUP_STRIP_RE.sub("", html_frag)
        return " ".join(cls._TAG_STRIP_RE.sub(" ", no_sup).split()).strip()

    @classmethod
    def parse_fomc_html(
        cls,
        html_text: str,
        *,
        min_year: int = 2021,
        max_year: int = 2027,
    ) -> list[MacroEventRelease]:
        """Parse official Federal Reserve FOMC HTML into PIT MacroEventRelease entries (14:00 America/New_York)."""
        matches = list(cls._PANEL_SPLIT_RE.finditer(html_text))
        events_by_id: dict[str, MacroEventRelease] = {}

        for idx, m in enumerate(matches):
            heading = cls._clean_text(m.group(1))
            ym = cls._YEAR_HEAD_RE.search(heading)
            if not ym:
                continue
            panel_year = int(ym.group(1))
            if panel_year < min_year or panel_year > max_year:
                continue
            seg_start = m.end()
            seg_end = matches[idx + 1].start() if (idx + 1) < len(matches) else len(html_text)
            section = html_text[seg_start:seg_end]

            for mo_raw, dt_raw in cls._ROW_PAIR_RE.findall(section):
                mo_text = cls._clean_text(mo_raw)
                dt_text = cls._clean_text(dt_raw)
                dt_low = dt_text.lower()
                if any(kw in dt_low for kw in cls._SKIP_KEYWORDS):
                    continue

                # CRITICAL: ONLY '*' in date text marks SEP/dot-plot (fomc-meeting--shaded is zebra striping!)
                is_sep = "*" in dt_text

                # Determine decision month: check month names embedded in dt_text first, else last token of mo_text
                months_in_dt = [
                    _MONTH_NAME_MAP[tok.lower()]
                    for tok in re.findall(r"[A-Za-z]+", dt_text)
                    if tok.lower() in _MONTH_NAME_MAP
                ]
                months_in_mo = [
                    _MONTH_NAME_MAP[tok.lower()]
                    for tok in re.split(r"[/\-\s]+", mo_text)
                    if tok.lower() in _MONTH_NAME_MAP
                ]
                if months_in_dt:
                    dec_month = months_in_dt[-1]
                    first_month = months_in_dt[0]
                elif months_in_mo:
                    dec_month = months_in_mo[-1]
                    first_month = months_in_mo[0]
                else:
                    continue

                day_nums = [int(x) for x in re.findall(r"\b(\d{1,2})\b", dt_text)]
                if not day_nums:
                    continue
                dec_day = day_nums[-1]
                dec_year = panel_year + (1 if (first_month == 12 and dec_month == 1) else 0)

                try:
                    _ = date(dec_year, dec_month, dec_day)
                except ValueError:
                    continue

                ts_rel_ms = ny_local_to_utc_ms(dec_year, dec_month, dec_day, 14, 0)
                event_id = f"FOMC-{dec_year:04d}-{dec_month:02d}-{dec_day:02d}"
                events_by_id[event_id] = MacroEventRelease(
                    event_type="FOMC",
                    event_id=event_id,
                    ref_period=f"{dec_year:04d}-{dec_month:02d}",
                    ts_release_ms=ts_rel_ms,
                    ts_avail_ms=ts_rel_ms,
                    is_sep_meeting=is_sep,
                    actual=None,
                    prior=None,
                    surprise_z=None,
                    source="federalreserve.gov",
                    provenance=PROVENANCE_REAL,
                )

        return sorted(events_by_id.values(), key=lambda e: (e.ts_release_ms, e.event_id))

    def fetch_calendar(self, *, min_year: int = 2021, max_year: int = 2027) -> list[MacroEventRelease]:
        blob = self.client.fetch_bytes(self.URL, max_retries=4, min_interval_sec=0.2)
        if not blob:
            return []
        return self.parse_fomc_html(blob.decode("utf-8", errors="replace"), min_year=min_year, max_year=max_year)


class BLSMacroEventCollector:
    """Single-request batched BLS Public API v2 collector + deterministic PIT NFP/CPI schedule."""

    API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    SERIES_CPI = "CUSR0000SA0"
    SERIES_NFP = "CES0000000001"
    CACHE_TTL_SEC = 12 * 3600

    def __init__(
        self,
        client: RateLimitedHTTPClient | None = None,
        cache_path: Path | str | None = None,
    ) -> None:
        self.client = client or RateLimitedHTTPClient(user_agent=YAHOO_V4_USER_AGENT)
        self.cache_path = Path(cache_path) if cache_path is not None else None

    def _read_cache(self, start_year: int, end_year: int) -> dict[str, Any] | None:
        if self.cache_path is None or not self.cache_path.exists():
            return None
        try:
            age = time.time() - self.cache_path.stat().st_mtime
            if age > self.CACHE_TTL_SEC:
                return None
            cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("_start_year") == int(start_year)
                and cached.get("_end_year") == int(end_year)
                and isinstance(cached.get("payload"), dict)
            ):
                return cached["payload"]
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable BLS cache %s: %s", self.cache_path, exc)
        return None

    def _write_cache(self, start_year: int, end_year: int, payload: dict[str, Any]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        wrapper = {
            "_start_year": int(start_year),
            "_end_year": int(end_year),
            "_cached_utc_ms": utc_now_ms(),
            "payload": payload,
        }
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp_bls_", suffix=".json", dir=str(self.cache_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(wrapper, fh, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.cache_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def fetch_raw_bls_payload(
        self,
        *,
        start_year: int = 2020,
        end_year: int = 2026,
        registration_key: str | None = None,
    ) -> dict[str, Any]:
        cached = self._read_cache(start_year, end_year)
        if cached is not None:
            return cached
        body: dict[str, Any] = {
            "seriesid": [self.SERIES_CPI, self.SERIES_NFP],
            "startyear": str(int(start_year)),
            "endyear": str(int(end_year)),
        }
        if registration_key:
            body["registrationkey"] = registration_key
        resp = self.client.fetch_json(
            self.API_URL,
            method="POST",
            json_payload=body,
            max_retries=4,
            timeout_sec=25.0,
        )
        if not isinstance(resp, dict) or resp.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS API v2 request did not succeed: {resp!r}")
        self._write_cache(start_year, end_year, resp)
        return resp

    @classmethod
    def parse_bls_series_values(cls, payload: Mapping[str, Any]) -> dict[str, dict[str, float | None]]:
        """Extract {series_id: {'YYYY-MM': float | None}} cleanly handling 2025-M10 value='-' footnote X."""
        out: dict[str, dict[str, float | None]] = {}
        series_list = ((payload.get("Results") or {}).get("series")) or []
        for s_obj in series_list:
            sid = str(s_obj.get("seriesID") or "")
            if not sid:
                continue
            monthly: dict[str, float | None] = {}
            for item in s_obj.get("data") or []:
                period = str(item.get("period") or "")
                if not re.fullmatch(r"M(0[1-9]|1[0-2])", period):
                    continue
                yr = int(item["year"])
                mo = int(period[1:])
                ref_key = f"{yr:04d}-{mo:02d}"
                footnotes = item.get("footnotes") or []
                has_lapse_flag = any(
                    isinstance(fn, Mapping) and str(fn.get("code") or "").upper() == "X"
                    for fn in footnotes
                )
                val = None if has_lapse_flag else _parse_optional_float(item.get("value"))
                monthly[ref_key] = val
            out[sid] = monthly
        return out

    @staticmethod
    def _prev_ref_period(year: int, month: int) -> str:
        return f"{year - 1:04d}-12" if month == 1 else f"{year:04d}-{month - 1:02d}"

    @classmethod
    def build_events_from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        schedule_start_year: int = 2021,
        schedule_end_year: int = 2026,
        now_ms: int | None = None,
    ) -> list[MacroEventRelease]:
        """Build strictly causal NFP and CPI MacroEventRelease records (`2021..2026`)."""
        ref_now = int(now_ms) if now_ms is not None else utc_now_ms()
        parsed = cls.parse_bls_series_values(payload)
        cpi_levels = parsed.get(cls.SERIES_CPI, {})
        nfp_levels = parsed.get(cls.SERIES_NFP, {})

        # Compute monthly CPI MoM % and NFP monthly job change (thousands) across 2020..schedule_end_year
        cpi_mom: dict[str, float | None] = {}
        nfp_chg: dict[str, float | None] = {}
        for y in range(schedule_start_year - 1, schedule_end_year + 1):
            for m in range(1, 13):
                rk = f"{y:04d}-{m:02d}"
                pk = cls._prev_ref_period(y, m)
                c_cur, c_prev = cpi_levels.get(rk), cpi_levels.get(pk)
                if c_cur is not None and c_prev is not None and c_prev > 0.0:
                    cpi_mom[rk] = round(((c_cur / c_prev) - 1.0) * 100.0, 4)
                else:
                    cpi_mom[rk] = None
                n_cur, n_prev = nfp_levels.get(rk), nfp_levels.get(pk)
                if n_cur is not None and n_prev is not None:
                    nfp_chg[rk] = round(n_cur - n_prev, 2)
                else:
                    nfp_chg[rk] = None

        events: list[MacroEventRelease] = []

        # Reference periods whose releases fall in schedule_start_year .. schedule_end_year:
        # 2020-12 releases in Jan 2021 .. 2026-11 releases in Dec 2026.
        ref_pairs: list[tuple[int, int]] = [(schedule_start_year - 1, 12)]
        for y in range(schedule_start_year, schedule_end_year + 1):
            max_m = 11 if y == schedule_end_year else 12
            for m in range(1, max_m + 1):
                ref_pairs.append((y, m))

        # 1. NFP events (strictly causal trailing 12-release surprise_z)
        nfp_history: list[float] = []
        # Warm trailing history with 2020-02..2020-11 if present (all strictly before Jan 2021!)
        for m in range(2, 12):
            rk_pre = f"{schedule_start_year - 1:04d}-{m:02d}"
            if nfp_chg.get(rk_pre) is not None:
                nfp_history.append(float(nfp_chg[rk_pre]))  # type: ignore[arg-type]

        for ry, rm in ref_pairs:
            rk = f"{ry:04d}-{rm:02d}"
            rel_d = bls_nfp_release_date(ry, rm)
            if rel_d.year < schedule_start_year or rel_d.year > schedule_end_year:
                continue
            ts_rel_ms = ny_local_to_utc_ms(rel_d.year, rel_d.month, rel_d.day, 8, 30)
            pk = cls._prev_ref_period(ry, rm)
            is_future = ts_rel_ms > ref_now
            raw_act = None if is_future else nfp_chg.get(rk)
            raw_prior = nfp_chg.get(pk)
            if is_future and ts_rel_ms - ref_now > 20 * DAY_MS:
                raw_prior = None
            surprise_z: float | None = None
            if raw_act is not None and raw_prior is not None and len(nfp_history) >= 6:
                window = nfp_history[-12:]
                mean_w = sum(window) / float(len(window))
                var_w = sum((x - mean_w) ** 2 for x in window) / float(max(1, len(window) - 1))
                std_w = math.sqrt(var_w)
                if std_w > 1e-9:
                    surprise_z = round((raw_act - raw_prior) / std_w, 4)
            if raw_act is not None:
                nfp_history.append(raw_act)

            events.append(
                MacroEventRelease(
                    event_type="NFP",
                    event_id=f"NFP-{rel_d.isoformat()}",
                    ref_period=rk,
                    ts_release_ms=ts_rel_ms,
                    ts_avail_ms=ts_rel_ms,
                    is_sep_meeting=False,
                    actual=raw_act,
                    prior=raw_prior,
                    surprise_z=surprise_z,
                    source="bls_schedule_official" if is_future else "bls_v2_latest_vintage",
                    provenance=PROVENANCE_REAL,
                )
            )

        # 2. CPI events (strictly causal trailing 12-release surprise_z)
        cpi_history: list[float] = []
        for m in range(2, 12):
            rk_pre = f"{schedule_start_year - 1:04d}-{m:02d}"
            if cpi_mom.get(rk_pre) is not None:
                cpi_history.append(float(cpi_mom[rk_pre]))  # type: ignore[arg-type]

        for ry, rm in ref_pairs:
            rk = f"{ry:04d}-{rm:02d}"
            rel_d = bls_cpi_release_date(ry, rm)
            if rel_d is None:
                # 2025-10 lapse in appropriations: no CPI release occurred
                continue
            if rel_d.year < schedule_start_year or rel_d.year > schedule_end_year:
                continue
            ts_rel_ms = ny_local_to_utc_ms(rel_d.year, rel_d.month, rel_d.day, 8, 30)
            pk = cls._prev_ref_period(ry, rm)
            is_future = ts_rel_ms > ref_now
            raw_act = None if is_future else cpi_mom.get(rk)
            raw_prior = cpi_mom.get(pk)
            if is_future and ts_rel_ms - ref_now > 20 * DAY_MS:
                raw_prior = None
            surprise_z = None
            if raw_act is not None and len(cpi_history) >= 6:
                window = cpi_history[-12:]
                mean_w = sum(window) / float(len(window))
                var_w = sum((x - mean_w) ** 2 for x in window) / float(max(1, len(window) - 1))
                std_w = math.sqrt(var_w)
                if std_w > 1e-9:
                    surprise_z = round((raw_act - mean_w) / std_w, 4)
            if raw_act is not None:
                cpi_history.append(raw_act)

            events.append(
                MacroEventRelease(
                    event_type="CPI",
                    event_id=f"CPI-{rel_d.isoformat()}",
                    ref_period=rk,
                    ts_release_ms=ts_rel_ms,
                    ts_avail_ms=ts_rel_ms,
                    is_sep_meeting=False,
                    actual=raw_act,
                    prior=raw_prior,
                    surprise_z=surprise_z,
                    source="bls_schedule_official" if is_future else "bls_v2_latest_vintage",
                    provenance=PROVENANCE_REAL,
                )
            )

        return sorted(events, key=lambda e: (e.ts_release_ms, e.event_type, e.event_id))

    def fetch_events(
        self,
        *,
        start_year: int = 2020,
        end_year: int = 2026,
        schedule_start_year: int = 2021,
        schedule_end_year: int = 2026,
        registration_key: str | None = None,
        now_ms: int | None = None,
    ) -> list[MacroEventRelease]:
        payload = self.fetch_raw_bls_payload(
            start_year=start_year,
            end_year=end_year,
            registration_key=registration_key,
        )
        return self.build_events_from_payload(
            payload,
            schedule_start_year=schedule_start_year,
            schedule_end_year=schedule_end_year,
            now_ms=now_ms,
        )


def default_gold_macro_symbols_v4() -> tuple[str, ...]:
    """Return the canonical 13-symbol Gold & Macro universe for Platform v4."""
    return GOLD_MACRO_SYMBOLS_V4
