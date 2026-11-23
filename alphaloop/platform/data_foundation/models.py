"""Universal Point-in-Time Envelope & Domain Data Models for Strategy R&D Platform v4.

Provides:
- CanonicalBar (backwards-compatible with v3 15-field positional layout + v4 PIT/session/basis/multiplier fields)
- Fail-closed validators: check_canonical_bar() and raising validate_canonical_bar()
- Domain-specific PIT dataclasses: Web3DerivativesMetric, Web3MarketMacroPoint,
  MacroYieldPoint, MacroEventRelease, PAXGBasisPoint + fail-closed validators
- DomainStoreProtocol for clean pluggable future asset domains (e.g. CN A-Shares)
- resample_bars() with D10 complete-bucket enforcement and D11 carry_interval_h preservation
"""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.data.quality import ALLOWED_REAL_PROVENANCES
from alphaloop.platform.assets import (
    ASSET_CLASS_CRYPTO_PERP,
    ASSET_CLASS_CRYPTO_SPOT,
    ASSET_CLASS_GOLD_PERP,
    TradingCalendar,
)

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
# 2000-01-01T00:00:00Z .. 2100-01-01T00:00:00Z:
# Blocks CLOCK_MONOTONIC leaks, Unix-seconds mistakes, and 16-digit microsecond leaks
# while permitting 10y TradFi Gold/Macro (2016+) and DefiLlama Stablecoin (2017+) histories.
MIN_PLATFORM_EPOCH_MS = 946_684_800_000
MAX_PLATFORM_EPOCH_MS = 4_102_444_800_000

_TWENTY_FOUR_SEVEN_CLASSES = frozenset(
    {
        ASSET_CLASS_CRYPTO_PERP,
        ASSET_CLASS_CRYPTO_SPOT,
        ASSET_CLASS_GOLD_PERP,
    }
)

_US_EQUITY_RTH_SYMBOLS = frozenset(
    {
        "GLD",
        "SLV",
        "UUP",
        "TIP",
        "IEF",
        "TLT",
        "GDX",
        "^GVZ",
        "^TNX",
    }
)


def _is_strict_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_finite_real(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _is_nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _is_provenance_allowed(provenance: str, allow_synthetic_in_test: bool) -> bool:
    if not _is_nonempty_str(provenance):
        return False
    if allow_synthetic_in_test:
        return True
    prov_parts = {p.strip() for p in provenance.split("+") if p.strip()}
    return bool(prov_parts) and prov_parts.issubset(ALLOWED_REAL_PROVENANCES)


def classify_tradfi_session_tag(
    ts_event_ms: int,
    symbol: str,
    instrument_type: str = "",
    timeframe_h: float = 1.0,
) -> str:
    """Deterministic session_tag for TradFi daily/intraday/4h bars."""
    inst = instrument_type.upper()
    if inst in ("ETF", "EQUITY") or symbol in _US_EQUITY_RTH_SYMBOLS:
        return "US_EQUITY_RTH"
    if timeframe_h >= 24.0:
        return "DAILY_SESSION"
    hour_utc = (int(ts_event_ms) // HOUR_MS) % 24
    if abs(float(timeframe_h) - 4.0) < 1e-6:
        # 4h UTC slots 08:00, 12:00, 16:00 cover 08:00-20:00 UTC (London AM/PM + NY COMEX)
        return "LONDON_NY" if hour_utc in (8, 12, 16) else "ASIAN_ROLLOVER"
    return "LONDON_NY" if 7 <= hour_utc <= 20 else "ASIAN_ROLLOVER"


@dataclass(frozen=True)
class CanonicalBar:
    """Unified Point-in-Time OHLCV + Carry bar across Web3, Gold, Macro, and Equities.

    Preserves the exact v3 15-field positional layout first and appends v4 metadata fields
    with safe defaults so all v3 positional & keyword callers remain 100% compatible.
    """

    asset_class: str
    venue: str
    symbol: str
    timeframe_h: float
    ts_event_ms: int        # Bar open timestamp (UTC ms)
    ts_avail_ms: int        # Earliest causal visibility timestamp (bar close UTC ms >= ts_event_ms)
    open: float
    high: float
    low: float
    close: float
    volume_usd: float
    carry_rate_period: float = 0.0   # Realized funding/carry rate over this bar (decimal; + means longs pay shorts)
    carry_interval_h: float = 1.0    # Settlement interval hours of carry_rate_period
    spread_bps: float = 0.0          # Observed top-of-book spread if known, else 0.0
    provenance: str = PROVENANCE_REAL
    source_fingerprint: str = ""
    timeframe_sec: int = 0
    session_tag: str = "ALL_24_7"
    premium_basis_bps: float = 0.0
    contract_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if (
            _is_strict_int(self.timeframe_sec)
            and self.timeframe_sec == 0
            and _is_finite_real(self.timeframe_h)
            and float(self.timeframe_h) > 0.0
        ):
            object.__setattr__(self, "timeframe_sec", int(round(float(self.timeframe_h) * 3600.0)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_canonical_bar(
    bar: CanonicalBar,
    *,
    allow_synthetic_in_test: bool = False,
) -> tuple[bool, str]:
    """Return (is_valid, reason) for fail-closed CanonicalBar integrity & PIT verification."""
    if not (_is_nonempty_str(bar.venue) and _is_nonempty_str(bar.symbol) and _is_nonempty_str(bar.asset_class)):
        return False, "missing_envelope_identifier"
    if not (_is_strict_int(bar.ts_event_ms) and _is_strict_int(bar.ts_avail_ms)):
        return False, "invalid_or_lookahead_timestamp"
    if (
        bar.ts_event_ms < MIN_PLATFORM_EPOCH_MS
        or bar.ts_event_ms > MAX_PLATFORM_EPOCH_MS
        or bar.ts_avail_ms < bar.ts_event_ms
        or bar.ts_avail_ms > MAX_PLATFORM_EPOCH_MS
    ):
        return False, "invalid_or_lookahead_timestamp"
    if (
        not _is_finite_real(bar.timeframe_h)
        or float(bar.timeframe_h) <= 0.0
        or not _is_strict_int(bar.timeframe_sec)
        or bar.timeframe_sec <= 0
        or abs(int(bar.timeframe_sec) - int(round(float(bar.timeframe_h) * 3600.0))) > 1
    ):
        return False, "invalid_timeframe"
    if not _is_finite_real(bar.carry_interval_h) or float(bar.carry_interval_h) <= 0.0:
        return False, "invalid_carry_interval"
    if not _is_finite_real(bar.contract_multiplier) or float(bar.contract_multiplier) <= 0.0:
        return False, "invalid_contract_multiplier"
    nums = (
        bar.open,
        bar.high,
        bar.low,
        bar.close,
        bar.volume_usd,
        bar.carry_rate_period,
        bar.spread_bps,
        bar.premium_basis_bps,
    )
    if not all(_is_finite_real(x) for x in nums):
        return False, "non_finite_numeric"
    if bar.open <= 0.0 or bar.high <= 0.0 or bar.low <= 0.0 or bar.close <= 0.0:
        return False, "non_positive_price"
    if bar.low > bar.high:
        return False, "inverted_high_low"
    if bar.high < max(bar.open, bar.close) - 1e-12 or bar.low > min(bar.open, bar.close) + 1e-12:
        return False, "ohlc_envelope_violation"
    if bar.volume_usd < 0.0 or bar.spread_bps < 0.0:
        return False, "negative_volume_or_spread"
    if not _is_provenance_allowed(bar.provenance, allow_synthetic_in_test):
        return False, f"disallowed_provenance:{bar.provenance}"
    return True, "ok"


def validate_canonical_bar(bar: CanonicalBar, allow_synthetic_in_test: bool = False) -> None:
    """Fail-closed validator raising ValueError on any PIT, OHLCV, unit, or provenance violation."""
    ok, reason = check_canonical_bar(bar, allow_synthetic_in_test=allow_synthetic_in_test)
    if not ok:
        raise ValueError(
            f"Invalid CanonicalBar ({bar.venue}/{bar.symbol}@{bar.ts_event_ms}): {reason}"
        )


def fingerprint_canonical_bars(bars: Sequence[CanonicalBar]) -> tuple[str, str, int]:
    """Compute deterministic SHA-256 hash and provenance union over a CanonicalBar sequence."""
    h = hashlib.sha256()
    provs: set[str] = set()
    ordered = sorted(bars, key=lambda b: (b.venue, b.symbol, b.timeframe_sec, b.ts_event_ms))
    for b in ordered:
        provs.add(b.provenance)
        h.update(
            f"{b.venue}|{b.symbol}|{b.timeframe_h:.4f}|{b.ts_event_ms}|{b.ts_avail_ms}|"
            f"{b.open:.8f}|{b.high:.8f}|{b.low:.8f}|{b.close:.8f}|{b.volume_usd:.4f}|"
            f"{b.carry_rate_period:.10f}|{b.provenance}\n".encode("utf-8")
        )
    prov_str = "+".join(sorted(provs)) if provs else "EMPTY"
    return h.hexdigest(), prov_str, len(ordered)


@dataclass(frozen=True)
class Web3DerivativesMetric:
    """Point-in-Time derivatives positioning & open interest metric (Binance UM / CEX)."""

    venue: str
    symbol: str
    ts_event_ms: int
    ts_avail_ms: int
    open_interest_usd: float
    top_trader_lsr: float
    account_lsr: float
    taker_ls_vol_ratio: float
    provenance: str = PROVENANCE_REAL
    source_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_web3_derivatives_metric(
    row: Web3DerivativesMetric,
    allow_synthetic_in_test: bool = False,
) -> None:
    if not (_is_nonempty_str(row.venue) and _is_nonempty_str(row.symbol)):
        raise ValueError("Web3DerivativesMetric requires non-empty venue and symbol")
    if not (_is_strict_int(row.ts_event_ms) and _is_strict_int(row.ts_avail_ms)) or (
        row.ts_event_ms < MIN_PLATFORM_EPOCH_MS
        or row.ts_event_ms > MAX_PLATFORM_EPOCH_MS
        or row.ts_avail_ms < row.ts_event_ms
        or row.ts_avail_ms > MAX_PLATFORM_EPOCH_MS
    ):
        raise ValueError(
            f"Invalid/lookahead timestamp on Web3DerivativesMetric {row.venue}/{row.symbol}@{row.ts_event_ms}"
        )
    vals = (
        row.open_interest_usd,
        row.top_trader_lsr,
        row.account_lsr,
        row.taker_ls_vol_ratio,
    )
    if not all(_is_finite_real(v) for v in vals):
        raise ValueError(f"Non-finite metric value on {row.venue}/{row.symbol}@{row.ts_event_ms}")
    if any(float(v) < 0.0 for v in vals):
        raise ValueError(f"Negative metric value on {row.venue}/{row.symbol}@{row.ts_event_ms}")
    if not _is_provenance_allowed(row.provenance, allow_synthetic_in_test):
        raise ValueError(f"Disallowed provenance on Web3DerivativesMetric: {row.provenance!r}")


@dataclass(frozen=True)
class Web3MarketMacroPoint:
    """Point-in-Time crypto-native market macro series (Deribit BTC DVOL, DefiLlama Stablecoins)."""

    source: str
    series_id: str
    ts_event_ms: int
    ts_avail_ms: int
    value: float
    provenance: str = PROVENANCE_REAL
    source_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_web3_market_macro_point(
    row: Web3MarketMacroPoint,
    allow_synthetic_in_test: bool = False,
) -> None:
    if not (_is_nonempty_str(row.source) and _is_nonempty_str(row.series_id)):
        raise ValueError("Web3MarketMacroPoint requires non-empty source and series_id")
    if not (_is_strict_int(row.ts_event_ms) and _is_strict_int(row.ts_avail_ms)) or (
        row.ts_event_ms < MIN_PLATFORM_EPOCH_MS
        or row.ts_event_ms > MAX_PLATFORM_EPOCH_MS
        or row.ts_avail_ms < row.ts_event_ms
        or row.ts_avail_ms > MAX_PLATFORM_EPOCH_MS
    ):
        raise ValueError(
            f"Invalid/lookahead timestamp on Web3MarketMacroPoint {row.source}/{row.series_id}@{row.ts_event_ms}"
        )
    if not _is_finite_real(row.value):
        raise ValueError(f"Non-finite value on Web3MarketMacroPoint {row.series_id}@{row.ts_event_ms}")
    if not _is_provenance_allowed(row.provenance, allow_synthetic_in_test):
        raise ValueError(f"Disallowed provenance on Web3MarketMacroPoint: {row.provenance!r}")


@dataclass(frozen=True)
class CNAshareMacroPoint:
    """Point-in-Time domestic CN A-Share macro, sovereign yield curve, margin-flow & FX observation.

    Stored strictly inside `domains/cn_ashare_lake.sqlite` (`cn_ashare_macro` table),
    physically isolated from Web3 and Gold/Macro data stores.
    Enforces strict `ts_avail_ms > ts_event_ms` so post-close ChinaBond yields (18:00 CST)
    and next-morning SSE/SZSE margin balances (T+1 08:30 CST) never leak intraday.
    """

    source: str
    series_id: str
    ts_event_ms: int
    ts_avail_ms: int
    value: float
    provenance: str = PROVENANCE_REAL
    source_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_cn_ashare_macro_point(
    row: CNAshareMacroPoint,
    allow_synthetic_in_test: bool = False,
) -> None:
    if not (_is_nonempty_str(row.source) and _is_nonempty_str(row.series_id)):
        raise ValueError("CNAshareMacroPoint requires non-empty source and series_id")
    if not (_is_strict_int(row.ts_event_ms) and _is_strict_int(row.ts_avail_ms)) or (
        row.ts_event_ms < MIN_PLATFORM_EPOCH_MS
        or row.ts_event_ms > MAX_PLATFORM_EPOCH_MS
        or row.ts_avail_ms <= row.ts_event_ms
        or row.ts_avail_ms > MAX_PLATFORM_EPOCH_MS
    ):
        raise ValueError(
            f"Invalid/lookahead timestamp on CNAshareMacroPoint {row.source}/{row.series_id}@{row.ts_event_ms}"
        )
    if not _is_finite_real(row.value):
        raise ValueError(f"Non-finite value on CNAshareMacroPoint {row.series_id}@{row.ts_event_ms}")
    if not _is_provenance_allowed(row.provenance, allow_synthetic_in_test):
        raise ValueError(f"Disallowed provenance on CNAshareMacroPoint: {row.provenance!r}")


@dataclass(frozen=True)
class MacroYieldPoint:
    """Point-in-Time US Treasury Daily Par Real Yield Curve observation (5Y/7Y/10Y/20Y/30Y)."""

    source: str
    ts_event_ms: int
    ts_avail_ms: int
    real_5y: float | None
    real_7y: float | None
    real_10y: float | None
    real_20y: float | None
    real_30y: float | None
    provenance: str = PROVENANCE_REAL
    source_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_macro_yield_point(
    row: MacroYieldPoint,
    allow_synthetic_in_test: bool = False,
) -> None:
    if not _is_nonempty_str(row.source):
        raise ValueError("MacroYieldPoint requires non-empty source")
    if not (_is_strict_int(row.ts_event_ms) and _is_strict_int(row.ts_avail_ms)) or (
        row.ts_event_ms < MIN_PLATFORM_EPOCH_MS
        or row.ts_event_ms > MAX_PLATFORM_EPOCH_MS
        or row.ts_avail_ms < row.ts_event_ms
        or row.ts_avail_ms > MAX_PLATFORM_EPOCH_MS
    ):
        raise ValueError(f"Invalid/lookahead timestamp on MacroYieldPoint {row.source}@{row.ts_event_ms}")
    tenors = (row.real_5y, row.real_7y, row.real_10y, row.real_20y, row.real_30y)
    if all(t is None for t in tenors):
        raise ValueError(f"MacroYieldPoint has all-null tenors at {row.ts_event_ms}")
    for val in tenors:
        if val is not None and not _is_finite_real(val):
            raise ValueError(f"Non-finite tenor value on MacroYieldPoint at {row.ts_event_ms}")
    if not _is_provenance_allowed(row.provenance, allow_synthetic_in_test):
        raise ValueError(f"Disallowed provenance on MacroYieldPoint: {row.provenance!r}")


@dataclass(frozen=True)
class MacroEventRelease:
    """Point-in-Time macro event release schedule & realized actuals (FOMC / NFP / CPI)."""

    event_type: str
    event_id: str
    ref_period: str
    ts_release_ms: int
    ts_avail_ms: int
    is_sep_meeting: bool = False
    actual: float | None = None
    prior: float | None = None
    surprise_z: float | None = None
    source: str = ""
    provenance: str = PROVENANCE_REAL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_macro_event_release(
    row: MacroEventRelease,
    allow_synthetic_in_test: bool = False,
) -> None:
    if not (_is_nonempty_str(row.event_type) and _is_nonempty_str(row.event_id)):
        raise ValueError("MacroEventRelease requires non-empty event_type and event_id")
    if not (_is_strict_int(row.ts_release_ms) and _is_strict_int(row.ts_avail_ms)) or (
        row.ts_release_ms < MIN_PLATFORM_EPOCH_MS
        or row.ts_release_ms > MAX_PLATFORM_EPOCH_MS
        or row.ts_avail_ms < row.ts_release_ms
        or row.ts_avail_ms > MAX_PLATFORM_EPOCH_MS
    ):
        raise ValueError(
            f"Invalid/lookahead timestamp on MacroEventRelease {row.event_type}/{row.event_id}"
        )
    for field_name, val in (("actual", row.actual), ("prior", row.prior), ("surprise_z", row.surprise_z)):
        if val is not None and not _is_finite_real(val):
            raise ValueError(f"Non-finite {field_name} on MacroEventRelease {row.event_id}")
    if not _is_provenance_allowed(row.provenance, allow_synthetic_in_test):
        raise ValueError(f"Disallowed provenance on MacroEventRelease: {row.provenance!r}")


@dataclass(frozen=True)
class PAXGBasisPoint:
    """Causal dual-basis snapshot decomposing PAXG Perp vs PAXG Spot vs COMEX GC=F."""

    venue_perp: str
    venue_spot: str
    ref_tradfi_symbol: str
    timeframe_sec: int
    ts_event_ms: int
    ts_avail_ms: int
    paxg_perp_close: float
    paxg_spot_close: float | None = None
    gc_ref_close: float | None = None
    paxg_perp_spot_bps: float | None = None
    paxg_spot_gc_bps: float | None = None
    paxg_perp_gc_bps: float | None = None
    is_weekend_frozen: bool = False
    provenance: str = PROVENANCE_REAL
    source_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_paxg_basis_point(
    row: PAXGBasisPoint,
    allow_synthetic_in_test: bool = False,
) -> None:
    if not (
        _is_nonempty_str(row.venue_perp)
        and _is_nonempty_str(row.venue_spot)
        and _is_nonempty_str(row.ref_tradfi_symbol)
    ):
        raise ValueError("PAXGBasisPoint requires non-empty venue_perp, venue_spot, and ref_tradfi_symbol")
    if not _is_strict_int(row.timeframe_sec) or row.timeframe_sec <= 0:
        raise ValueError("PAXGBasisPoint timeframe_sec must be positive")
    if not (_is_strict_int(row.ts_event_ms) and _is_strict_int(row.ts_avail_ms)) or (
        row.ts_event_ms < MIN_PLATFORM_EPOCH_MS
        or row.ts_event_ms > MAX_PLATFORM_EPOCH_MS
        or row.ts_avail_ms < row.ts_event_ms
        or row.ts_avail_ms > MAX_PLATFORM_EPOCH_MS
    ):
        raise ValueError(
            f"Invalid/lookahead timestamp on PAXGBasisPoint {row.venue_perp}@{row.ts_event_ms}"
        )
    if not _is_finite_real(row.paxg_perp_close) or float(row.paxg_perp_close) <= 0.0:
        raise ValueError("PAXGBasisPoint paxg_perp_close must be finite and positive")
    for name, val in (
        ("paxg_spot_close", row.paxg_spot_close),
        ("gc_ref_close", row.gc_ref_close),
    ):
        if val is not None and (not _is_finite_real(val) or float(val) <= 0.0):
            raise ValueError(f"PAXGBasisPoint {name} must be finite and positive when present")
    for name, val in (
        ("paxg_perp_spot_bps", row.paxg_perp_spot_bps),
        ("paxg_spot_gc_bps", row.paxg_spot_gc_bps),
        ("paxg_perp_gc_bps", row.paxg_perp_gc_bps),
    ):
        if val is not None and not _is_finite_real(val):
            raise ValueError(f"PAXGBasisPoint {name} must be finite when present")
    if not _is_provenance_allowed(row.provenance, allow_synthetic_in_test):
        raise ValueError(f"Disallowed provenance on PAXGBasisPoint: {row.provenance!r}")


@runtime_checkable
class DomainStoreProtocol(Protocol):
    """Pluggable storage contract for physically partitioned asset domains (Web3, Gold/Macro, future CN A-Shares)."""

    domain_name: str
    db_path: Path

    def upsert_bars(
        self,
        bars: Sequence[CanonicalBar],
        *,
        allow_synthetic_in_test: bool = False,
    ) -> int: ...

    def load_bars(
        self,
        venue: str,
        symbol: str,
        timeframe_sec: int,
        *,
        as_of_ms: int | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[CanonicalBar]: ...

    def list_series(self) -> list[dict[str, Any]]: ...

    def iter_export_rows(self) -> Iterator[dict[str, Any]]: ...

    def close(self) -> None: ...


def resample_bars(
    bars: Sequence[CanonicalBar],
    target_tf_h: float,
    *,
    require_complete: bool = True,
    min_subbars: int | None = None,
) -> list[CanonicalBar]:
    """Strictly causal UTC-aligned bar resampling (D10 complete-bucket + D11 carry_interval_h fixes).

    - Deduplicates by ts_event_ms (keeping last) and sorts ascending.
    - Drops incomplete leading/trailing buckets when require_complete=True:
      * 24/7 assets (CRYPTO_PERP, CRYPTO_SPOT, GOLD_PERP) require 100% of expected subbars
        (e.g. 4,800 aligned 1h bars -> exact 400 12h bars; 4,800 offset-11h bars -> 399 12h bars).
      * TradFi assets default to min_subbars=2 for 1h->4h (or ceil(expected * 0.5) when expected >= 2)
        so COMEX 1h daily Globex maintenance breaks & NYSE RTH sessions stay intact without 1-bar stubs.
      * Callers can explicitly override with min_subbars.
    - Strictly causal ts_avail_ms = max(b_start + tf_ms, max(x.ts_avail_ms for x in grp)).
    - Preserves carry_interval_h = max(target_tf_h, max(x.carry_interval_h for x in grp)).
    """
    if not bars:
        return []
    if not _is_finite_real(target_tf_h) or float(target_tf_h) <= 0.0:
        raise ValueError("target_tf_h must be finite and positive")

    tf_ms = int(round(float(target_tf_h) * HOUR_MS))
    if tf_ms <= 0:
        raise ValueError("target_tf_h must produce positive timeframe milliseconds")

    if not _is_finite_real(bars[0].timeframe_h) or float(bars[0].timeframe_h) <= 0.0:
        raise ValueError("resample_bars requires finite positive source timeframe_h")
    src_tf_h = float(bars[0].timeframe_h)
    if any(
        (not _is_finite_real(b.timeframe_h)) or abs(float(b.timeframe_h) - src_tf_h) > 1e-6
        for b in bars
    ):
        raise ValueError("resample_bars requires uniform source timeframe_h across input bars")

    # Deduplicate by ts_event_ms so duplicate timestamps can never fake bucket completeness
    dedup: dict[int, CanonicalBar] = {}
    for b in bars:
        dedup[int(b.ts_event_ms)] = b
    ordered = [dedup[ts] for ts in sorted(dedup)]

    if src_tf_h > float(target_tf_h) + 1e-6:
        raise ValueError(
            f"Cannot downsample coarser timeframe {src_tf_h}h to finer {target_tf_h}h"
        )
    if abs(src_tf_h - float(target_tf_h)) < 1e-6:
        return list(ordered)

    expected_subbars = max(1, int(round(float(target_tf_h) / max(1e-6, src_tf_h))))
    first_bar = ordered[0]
    is_24_7 = first_bar.asset_class in _TWENTY_FOUR_SEVEN_CLASSES
    is_us_rth = (
        first_bar.symbol in _US_EQUITY_RTH_SYMBOLS
        or first_bar.session_tag == "US_EQUITY_RTH"
    )

    if min_subbars is not None:
        required_n = max(1, int(min_subbars))
    elif require_complete:
        if is_24_7:
            required_n = expected_subbars
        elif is_us_rth and float(target_tf_h) >= 24.0:
            # US Equity RTH is 6.5h/day (7 x 1h bars or 2 x 4h bars); ceil avoids banker's rounding on 6.5
            rth_expected = max(1, math.ceil(6.5 / max(1e-6, src_tf_h)))
            required_n = max(1, math.ceil(rth_expected * 0.5))
        else:
            required_n = max(2, math.ceil(expected_subbars * 0.5)) if expected_subbars >= 2 else 1
    else:
        required_n = 1

    buckets: dict[int, list[CanonicalBar]] = {}
    for b in ordered:
        b_start = (int(b.ts_event_ms) // tf_ms) * tf_ms
        buckets.setdefault(b_start, []).append(b)

    target_tf_sec = int(round(float(target_tf_h) * 3600.0))
    out: list[CanonicalBar] = []
    for b_start in sorted(buckets):
        grp = buckets[b_start]
        if len(grp) < required_n:
            continue
        first = grp[0]
        last = grp[-1]
        n_grp = float(len(grp))
        provs = sorted({x.provenance for x in grp})
        carry_sum = sum(x.carry_rate_period for x in grp)
        carry_iv = max(float(target_tf_h), max(x.carry_interval_h for x in grp))
        spread_avg = sum(x.spread_bps for x in grp) / n_grp
        basis_avg = sum(x.premium_basis_bps for x in grp) / n_grp
        fps = {x.source_fingerprint for x in grp if x.source_fingerprint}
        src_fp = next(iter(fps)) if len(fps) == 1 else f"resampled:{first.venue}:{first.symbol}:{target_tf_sec}"
        if is_24_7:
            stag = "ALL_24_7"
        else:
            stag = classify_tradfi_session_tag(
                b_start,
                first.symbol,
                instrument_type="",
                timeframe_h=float(target_tf_h),
            )
        out.append(
            CanonicalBar(
                asset_class=first.asset_class,
                venue=first.venue,
                symbol=first.symbol,
                timeframe_h=float(target_tf_h),
                ts_event_ms=b_start,
                ts_avail_ms=max(b_start + tf_ms, max(x.ts_avail_ms for x in grp)),
                open=first.open,
                high=max(x.high for x in grp),
                low=min(x.low for x in grp),
                close=last.close,
                volume_usd=sum(x.volume_usd for x in grp),
                carry_rate_period=carry_sum,
                carry_interval_h=carry_iv,
                spread_bps=spread_avg,
                provenance="+".join(provs),
                source_fingerprint=src_fp,
                timeframe_sec=target_tf_sec,
                session_tag=stag,
                premium_basis_bps=basis_avg,
                contract_multiplier=first.contract_multiplier,
            )
        )
    return out


@dataclass
class MultiAssetDQCheck:
    gate_id: str
    name: str
    passed: bool
    value: str
    threshold: str
    detail: str


@dataclass
class MultiAssetDQReport:
    dataset_label: str
    asset_class: str
    row_count: int
    dataset_sha256: str
    provenance: str
    passed: bool
    checks: list[MultiAssetDQCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_label": self.dataset_label,
            "asset_class": self.asset_class,
            "row_count": self.row_count,
            "dataset_sha256": self.dataset_sha256,
            "provenance": self.provenance,
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
        }


def audit_bars_quality(
    bars: Sequence[CanonicalBar],
    calendar: TradingCalendar,
    dataset_label: str = "dataset",
    min_bars: int = 40,
    allow_synthetic_in_test: bool = False,
) -> MultiAssetDQReport:
    """Multi-Asset Data Quality Certification (DQ-MA-01 .. DQ-MA-07)."""
    ds_sha, prov, row_count = fingerprint_canonical_bars(bars)
    prov_parts = set(prov.split("+")) if prov and prov != "EMPTY" else set()

    dq1_ok = (
        row_count >= min_bars
        and bool(ds_sha)
        and (
            (bool(prov_parts) and prov_parts.issubset(ALLOWED_REAL_PROVENANCES))
            or allow_synthetic_in_test
        )
    )

    ordered = sorted(bars, key=lambda b: b.ts_event_ms)
    asset_class = ordered[0].asset_class if ordered else "UNKNOWN"
    non_monotonic = 0
    pit_violations = 0
    invalid_ohlc = 0
    max_gap_h = 0.0
    max_jump_pct = 0.0
    flat_streak = 0
    max_flat_streak = 0

    prev: CanonicalBar | None = None
    for b in ordered:
        # Provenance is isolated in DQ-MA-01 so DQ-MA-03 checks pure structural/OHLCV/PIT integrity
        ok, _ = check_canonical_bar(b, allow_synthetic_in_test=True)
        if not ok:
            invalid_ohlc += 1
        if not (_is_strict_int(b.ts_event_ms) and _is_strict_int(b.ts_avail_ms)) or b.ts_avail_ms < b.ts_event_ms:
            pit_violations += 1
        if b.high == b.low and b.open == b.close:
            flat_streak += 1
            if flat_streak > max_flat_streak:
                max_flat_streak = flat_streak
        else:
            flat_streak = 0
        if prev is not None:
            if b.ts_event_ms <= prev.ts_event_ms:
                non_monotonic += 1
            else:
                gap_h = (b.ts_event_ms - prev.ts_event_ms) / float(HOUR_MS)
                if gap_h > max_gap_h:
                    max_gap_h = gap_h
            if prev.close > 0:
                jump = abs(b.close / prev.close - 1.0) * 100.0
                if jump > max_jump_pct:
                    max_jump_pct = jump
        prev = b

    expected_step_h = float(ordered[0].timeframe_h) if ordered else 1.0
    is_24_7_cal = (
        calendar.calendar_id == "CRYPTO_24_7"
        or (calendar.trading_hours_per_day >= 24.0 and calendar.trading_days_per_year >= 365.0)
    )
    if is_24_7_cal:
        # Tightened D11 budget: removes the old 2x (108h on 12h) loophole
        max_allowed_gap_h = max(calendar.max_expected_gap_hours, expected_step_h * 3.0)
    else:
        max_allowed_gap_h = max(calendar.max_expected_gap_hours * 1.25, expected_step_h * 4.5)

    # DQ-MA-07: Perp Funding Coverage (>=98.0% expected settlement buckets) & Unit Sanity (D11)
    perp_bars = [
        b for b in ordered if b.asset_class in (ASSET_CLASS_CRYPTO_PERP, ASSET_CLASS_GOLD_PERP)
    ]
    non_perp_bars = [
        b for b in ordered if b.asset_class not in (ASSET_CLASS_CRYPTO_PERP, ASSET_CLASS_GOLD_PERP)
    ]
    if perp_bars and non_perp_bars:
        dq7_check = MultiAssetDQCheck(
            gate_id="DQ-MA-07",
            name="Perp Funding Coverage & Unit Sanity",
            passed=False,
            value=f"mixed_asset_classes (perp={len(perp_bars)}, non_perp={len(non_perp_bars)})",
            threshold="cov>=98.0%, med_apr<=150%, p95_apr<=1500%, max_8h<=0.05",
            detail="mixed_perp_and_non_perp_asset_classes",
        )
    elif perp_bars:
        ci_valid = all(
            _is_finite_real(b.carry_interval_h)
            and _is_finite_real(b.timeframe_h)
            and 0.0 < float(b.carry_interval_h) <= max(24.0, float(b.timeframe_h) + 8.0)
            for b in perp_bars
        )
        nz_bars = [
            b for b in perp_bars if _is_finite_real(b.carry_rate_period) and abs(float(b.carry_rate_period)) > 0.0
        ]
        if not ci_valid:
            dq7_ok = False
            dq7_val = f"cov=0.00% (ci_invalid, nz={len(nz_bars)}/{len(perp_bars)})"
            dq7_detail = f"ci_valid=False, nz_bars={len(nz_bars)}/{len(perp_bars)}"
        else:
            if expected_step_h >= 8.0:
                expected_total = float(len(perp_bars))
                covered_total = float(len(nz_bars))
                funding_cov_pct = min(100.0, (covered_total / max(1e-9, expected_total)) * 100.0)
                seg_count = 1
            else:
                expected_total = 0.0
                covered_total = 0.0
                seg_count = 0
                cur_cad_h: float | None = None
                cur_seg: list[CanonicalBar] = []

                def _flush_seg(seg: list[CanonicalBar], cad_h: float) -> tuple[float, float]:
                    cad_ms = max(1, int(round(cad_h * HOUR_MS)))
                    seg_expected_frac = sum(float(x.timeframe_h) / cad_h for x in seg)
                    seg_nz = [
                        x for x in seg if _is_finite_real(x.carry_rate_period) and abs(float(x.carry_rate_period)) > 0.0
                    ]
                    seg_unique_buckets = len({int(x.ts_event_ms) // cad_ms for x in seg_nz})
                    seg_covered = min(float(seg_unique_buckets), float(len(seg_nz)), seg_expected_frac)
                    return seg_expected_frac, seg_covered

                for b in perp_bars:
                    cad_h = round(max(float(b.timeframe_h), float(b.carry_interval_h)), 2)
                    if cur_cad_h is None or cad_h != cur_cad_h:
                        if cur_seg and cur_cad_h is not None:
                            se, sc = _flush_seg(cur_seg, cur_cad_h)
                            expected_total += se
                            covered_total += sc
                            seg_count += 1
                        cur_cad_h = cad_h
                        cur_seg = [b]
                    else:
                        cur_seg.append(b)
                if cur_seg and cur_cad_h is not None:
                    se, sc = _flush_seg(cur_seg, cur_cad_h)
                    expected_total += se
                    covered_total += sc
                    seg_count += 1
                funding_cov_pct = min(100.0, (covered_total / max(1e-9, expected_total)) * 100.0)

            abs_aprs: list[float] = []
            rates_8h: list[float] = []
            for b in nz_bars:
                eff_h = max(0.01, float(b.carry_interval_h))
                abs_aprs.append(abs(float(b.carry_rate_period)) / eff_h * 8760.0 * 100.0)
                rates_8h.append(abs(float(b.carry_rate_period)) * (8.0 / eff_h))
            med_apr = float(statistics.median(abs_aprs)) if abs_aprs else 0.0
            if abs_aprs:
                sorted_aprs = sorted(abs_aprs)
                p95_apr = float(sorted_aprs[min(len(sorted_aprs) - 1, int(0.95 * len(sorted_aprs)))])
            else:
                p95_apr = 0.0
            max_rate_8h = max(rates_8h) if rates_8h else 0.0

            dq7_ok = (
                ci_valid
                and funding_cov_pct >= 98.0
                and med_apr <= 150.0
                and p95_apr <= 1500.0
                and max_rate_8h <= 0.05
            )
            dq7_val = (
                f"cov={funding_cov_pct:.2f}% ({covered_total:.1f}/{expected_total:.1f}), "
                f"med_apr={med_apr:.1f}%, p95_apr={p95_apr:.1f}%, max_8h={max_rate_8h:.5f}"
            )
            dq7_detail = (
                f"step_h={expected_step_h:.1f}h, segments={seg_count}, "
                f"nz_bars={len(nz_bars)}/{len(perp_bars)}, ci_valid={ci_valid}"
            )
        dq7_check = MultiAssetDQCheck(
            gate_id="DQ-MA-07",
            name="Perp Funding Coverage & Unit Sanity",
            passed=dq7_ok,
            value=dq7_val,
            threshold="cov>=98.0%, med_apr<=150%, p95_apr<=1500%, max_8h<=0.05",
            detail=dq7_detail,
        )
    else:
        dq7_check = MultiAssetDQCheck(
            gate_id="DQ-MA-07",
            name="Perp Funding Coverage & Unit Sanity",
            passed=True,
            value="N/A (non-perp)",
            threshold="perp-only",
            detail=f"asset_class={asset_class}",
        )

    checks = [
        MultiAssetDQCheck(
            gate_id="DQ-MA-01",
            name="Provenance & Sample Floor",
            passed=dq1_ok,
            value=f"{prov} (n={row_count})",
            threshold=f"REAL & n>={min_bars}",
            detail=f"sha256={ds_sha[:12]}",
        ),
        MultiAssetDQCheck(
            gate_id="DQ-MA-02",
            name="Point-in-Time Causal Visibility",
            passed=(pit_violations == 0 and non_monotonic == 0),
            value=f"pit_err={pit_violations}, non_mono={non_monotonic}",
            threshold="0 violations",
            detail="ts_avail_ms >= ts_event_ms and strictly increasing",
        ),
        MultiAssetDQCheck(
            gate_id="DQ-MA-03",
            name="OHLCV Finite & Envelope Integrity",
            passed=(invalid_ohlc == 0),
            value=f"invalid={invalid_ohlc}",
            threshold="0 invalid bars",
            detail="Checks NaN/Inf, positive prices, high>=max(O,C), low<=min(O,C)",
        ),
        MultiAssetDQCheck(
            gate_id="DQ-MA-04",
            name="Calendar-Aware Gap Continuity",
            passed=(max_gap_h <= max_allowed_gap_h),
            value=f"max_gap={max_gap_h:.1f}h",
            threshold=f"<={max_allowed_gap_h:.1f}h ({calendar.calendar_id})",
            detail=f"timeframe={expected_step_h}h",
        ),
        MultiAssetDQCheck(
            gate_id="DQ-MA-05",
            name="Extreme Glitch Jump Guard",
            passed=(max_jump_pct <= 80.0),
            value=f"max_bar_jump={max_jump_pct:.2f}%",
            threshold="<=80.0%",
            detail="Blocks 10x decimal/currency corruption spikes",
        ),
        MultiAssetDQCheck(
            gate_id="DQ-MA-06",
            name="Frozen Feed Guard",
            passed=(max_flat_streak <= max(12, int(round(row_count * 0.10)))),
            value=f"max_flat_streak={max_flat_streak}",
            threshold="<=max(12, 10% bars)",
            detail="Detects dead/frozen price feeds",
        ),
        dq7_check,
    ]
    return MultiAssetDQReport(
        dataset_label=dataset_label,
        asset_class=asset_class,
        row_count=row_count,
        dataset_sha256=ds_sha,
        provenance=prov,
        passed=all(c.passed for c in checks),
        checks=checks,
    )
