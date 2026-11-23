"""Multi-Asset Domain Specifications: Web3 Perps/Spot, Gold (PAXG & TradFi), Macro, and A-Share Extension.

Implements QuantConnect LEAN's asset-modeling discipline so the Strategy R&D Platform v3
shares one unified data lake, factor engine, regime profiler, parameter inferrer, and
anti-overfitting harness across Web3 and Gold while preserving exact venue/calendar/cost physics
(and leaving a clean extension point for CN A-shares).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ASSET_CLASS_CRYPTO_PERP = "CRYPTO_PERP"
ASSET_CLASS_CRYPTO_SPOT = "CRYPTO_SPOT"
ASSET_CLASS_GOLD_PERP = "GOLD_PERP"
ASSET_CLASS_GOLD_TRADFI = "GOLD_TRADFI"
ASSET_CLASS_MACRO_INDEX = "MACRO_INDEX"
ASSET_CLASS_CN_A_SHARE = "CN_A_SHARE"

VALID_ASSET_CLASSES = frozenset(
    {
        ASSET_CLASS_CRYPTO_PERP,
        ASSET_CLASS_CRYPTO_SPOT,
        ASSET_CLASS_GOLD_PERP,
        ASSET_CLASS_GOLD_TRADFI,
        ASSET_CLASS_MACRO_INDEX,
        ASSET_CLASS_CN_A_SHARE,
    }
)


@dataclass(frozen=True)
class TradingCalendar:
    """Market session & annualization specification per asset class."""

    calendar_id: str
    trading_hours_per_day: float
    trading_days_per_year: float
    allow_short: bool = True
    t_plus_days: int = 0
    daily_price_limit_pct: float | None = None
    max_expected_gap_hours: float = 4.0  # Weekend-aware gap budget for DQ

    def bars_per_year(self, timeframe_hours: float) -> float:
        """Return exact annual bar count for Sharpe & realized-volatility scaling."""
        tf = max(0.01, float(timeframe_hours))
        if tf >= 24.0:
            return self.trading_days_per_year / (tf / 24.0)
        return (self.trading_hours_per_day * self.trading_days_per_year) / tf

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CostModel:
    """Explicit transaction, slippage, stamp-duty, and holding-cost model (Carver 'model costs first')."""

    maker_fee_bps: float
    taker_fee_bps: float
    slippage_bps_per_leg: float
    stamp_duty_sell_bps: float = 0.0
    annual_carry_drag_bps: float = 0.0
    rebalance_drag_bps_per_month: float = 0.0

    def one_way_cost_bps(
        self,
        execution_mode: str = "TAKER_TAKER",
        is_sell: bool = False,
        extra_slippage_bps: float = 0.0,
    ) -> float:
        mode = execution_mode.upper()
        if mode == "MAKER_MAKER":
            fee = self.maker_fee_bps
        elif mode == "MAKER_THEN_TAKER_HEDGE":
            fee = 0.5 * (self.maker_fee_bps + self.taker_fee_bps)
        else:
            fee = self.taker_fee_bps
        stamp = self.stamp_duty_sell_bps if is_sell else 0.0
        return max(0.0, fee + self.slippage_bps_per_leg + extra_slippage_bps + stamp)

    def round_trip_cost_bps(
        self,
        execution_mode: str = "TAKER_TAKER",
        extra_slippage_bps_per_leg: float = 0.0,
    ) -> float:
        return self.one_way_cost_bps(execution_mode, is_sell=False, extra_slippage_bps=extra_slippage_bps_per_leg) + (
            self.one_way_cost_bps(execution_mode, is_sell=True, extra_slippage_bps=extra_slippage_bps_per_leg)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssetSpec:
    """Canonical instrument descriptor binding asset class, venue, calendar, costs, and proxy links."""

    symbol: str
    asset_class: str
    venue: str
    calendar: TradingCalendar
    cost_model: CostModel
    max_leverage_cap: float = 2.0
    base_currency: str = ""
    quote_currency: str = "USD"
    proxy_symbol: str = ""  # E.g. PAXG -> GC=F for cross-source arbitration
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.asset_class not in VALID_ASSET_CLASSES:
            raise ValueError(f"Unsupported asset_class: {self.asset_class!r}")
        if self.max_leverage_cap <= 0.0:
            raise ValueError("max_leverage_cap must be positive")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


# Canonical calendars
CALENDAR_CRYPTO_24_7 = TradingCalendar(
    calendar_id="CRYPTO_24_7",
    trading_hours_per_day=24.0,
    trading_days_per_year=365.0,
    allow_short=True,
    t_plus_days=0,
    daily_price_limit_pct=None,
    max_expected_gap_hours=4.0,
)

CALENDAR_GOLD_TRADFI_23_5 = TradingCalendar(
    calendar_id="GOLD_TRADFI_23_5",
    trading_hours_per_day=23.0,
    trading_days_per_year=252.0,
    allow_short=True,
    t_plus_days=0,
    daily_price_limit_pct=None,
    max_expected_gap_hours=96.0,  # Long holidays + weekends
)

CALENDAR_US_DAILY = TradingCalendar(
    calendar_id="US_EQUITY_DAILY",
    trading_hours_per_day=6.5,
    trading_days_per_year=252.0,
    allow_short=True,
    t_plus_days=0,
    daily_price_limit_pct=None,
    max_expected_gap_hours=120.0,  # 4-day holiday weekend
)

CALENDAR_CN_A_SHARE = TradingCalendar(
    calendar_id="CN_SSE_SZSE",
    trading_hours_per_day=4.0,
    trading_days_per_year=242.0,
    allow_short=False,
    t_plus_days=1,
    daily_price_limit_pct=10.0,
    max_expected_gap_hours=240.0,  # Golden Week / Spring Festival
)


# Canonical cost models (conservative real-world fees)
COST_HYPERLIQUID_PERP = CostModel(
    maker_fee_bps=1.5,
    taker_fee_bps=4.5,
    slippage_bps_per_leg=1.5,
    annual_carry_drag_bps=0.0,  # Funding settled explicitly per bar
)

COST_GATE_PERP = CostModel(
    maker_fee_bps=2.0,
    taker_fee_bps=5.0,
    slippage_bps_per_leg=2.0,
    annual_carry_drag_bps=0.0,
)

COST_GOLD_FUTURES_ETF = CostModel(
    maker_fee_bps=1.0,
    taker_fee_bps=2.0,
    slippage_bps_per_leg=2.0,
    annual_carry_drag_bps=40.0,  # GLD 0.40% ER / conservative futures roll friction
)

COST_CN_A_SHARE = CostModel(
    maker_fee_bps=2.5,
    taker_fee_bps=2.5,
    slippage_bps_per_leg=5.0,
    stamp_duty_sell_bps=5.0,
)


WEB3_LIQUID_UNIVERSE: tuple[str, ...] = (
    "BTC", "ETH", "SOL", "XRP", "DOGE", "AVAX",
    "LINK", "AAVE", "UNI", "SUI", "HYPE", "LTC",
)

GOLD_MACRO_SYMBOLS: tuple[str, ...] = (
    "GC=F",      # COMEX Gold Futures
    "GLD",       # SPDR Gold Trust ETF
    "IEF",       # iShares 7-10Y US Treasury ETF (QuantPedia real-rate regime proxy)
    "DX-Y.NYB",  # US Dollar Index (DXY)
    "^TNX",      # CBOE 10Y Treasury Yield
    "TIP",       # iShares TIPS Bond ETF (real-yield proxy)
    "^VIX",      # CBOE Volatility Index (global stress)
    "SI=F",      # COMEX Silver Futures (Gold/Silver ratio)
)


def default_web3_specs(
    symbols: tuple[str, ...] | list[str] = WEB3_LIQUID_UNIVERSE,
    venue: str = "hyperliquid",
) -> dict[str, AssetSpec]:
    cost = COST_HYPERLIQUID_PERP if venue == "hyperliquid" else COST_GATE_PERP
    out: dict[str, AssetSpec] = {}
    for sym in symbols:
        s = sym.upper()
        is_gold_token = s in ("PAXG", "XAUT", "XAU")
        out[s] = AssetSpec(
            symbol=s,
            asset_class=ASSET_CLASS_GOLD_PERP if is_gold_token else ASSET_CLASS_CRYPTO_PERP,
            venue=venue,
            calendar=CALENDAR_CRYPTO_24_7,
            cost_model=cost,
            max_leverage_cap=2.0,
            base_currency=s,
            quote_currency="USD",
            proxy_symbol="GC=F" if is_gold_token else "",
            tags=("gold", "perp_24_7") if is_gold_token else ("web3", "crypto_major", "perp_24_7"),
        )
    return out


def default_gold_specs() -> dict[str, AssetSpec]:
    return {
        "PAXG": AssetSpec(
            symbol="PAXG",
            asset_class=ASSET_CLASS_GOLD_PERP,
            venue="hyperliquid",
            calendar=CALENDAR_CRYPTO_24_7,
            cost_model=COST_HYPERLIQUID_PERP,
            max_leverage_cap=2.0,
            base_currency="XAU",
            quote_currency="USD",
            proxy_symbol="GC=F",
            tags=("gold", "paxg", "perp_24_7"),
        ),
        "GC=F": AssetSpec(
            symbol="GC=F",
            asset_class=ASSET_CLASS_GOLD_TRADFI,
            venue="yahoo",
            calendar=CALENDAR_GOLD_TRADFI_23_5,
            cost_model=COST_GOLD_FUTURES_ETF,
            max_leverage_cap=2.0,
            base_currency="XAU",
            quote_currency="USD",
            proxy_symbol="PAXG",
            tags=("gold", "comex", "tradfi"),
        ),
        "GLD": AssetSpec(
            symbol="GLD",
            asset_class=ASSET_CLASS_GOLD_TRADFI,
            venue="yahoo",
            calendar=CALENDAR_US_DAILY,
            cost_model=COST_GOLD_FUTURES_ETF,
            max_leverage_cap=1.5,
            base_currency="XAU",
            quote_currency="USD",
            proxy_symbol="GC=F",
            tags=("gold", "etf", "tradfi"),
        ),
    }


def default_macro_specs() -> dict[str, AssetSpec]:
    out: dict[str, AssetSpec] = {}
    for sym in GOLD_MACRO_SYMBOLS:
        ac = ASSET_CLASS_GOLD_TRADFI if sym in ("GC=F", "GLD") else ASSET_CLASS_MACRO_INDEX
        cal = CALENDAR_GOLD_TRADFI_23_5 if sym in ("GC=F", "SI=F") else CALENDAR_US_DAILY
        out[sym] = AssetSpec(
            symbol=sym,
            asset_class=ac,
            venue="yahoo",
            calendar=cal,
            cost_model=COST_GOLD_FUTURES_ETF,
            max_leverage_cap=1.0,
            base_currency=sym,
            quote_currency="USD",
            tags=("macro", "gold_regime"),
        )
    return out


def default_ashare_template_spec(symbol: str = "000300.SH") -> AssetSpec:
    """Ready-to-plug CN A-Share AssetSpec template (T+1, long-only, +-10% limit, stamp duty)."""
    return AssetSpec(
        symbol=symbol,
        asset_class=ASSET_CLASS_CN_A_SHARE,
        venue="cn_a_share",
        calendar=CALENDAR_CN_A_SHARE,
        cost_model=COST_CN_A_SHARE,
        max_leverage_cap=1.0,
        base_currency=symbol,
        quote_currency="CNY",
        tags=("cn_a_share", "t_plus_1", "long_only"),
    )
