"""Strategy R&D Platform v4 — Option B Partitioned Point-in-Time Data Foundation.

Exports:
- Models & Validators (`CanonicalBar`, `Web3DerivativesMetric`, `Web3MarketMacroPoint`,
  `MacroYieldPoint`, `MacroEventRelease`, `PAXGBasisPoint`, `DomainStoreProtocol`, `resample_bars`)
- Router (`DataFoundationRouter`, `Web3DomainStore`, `GoldMacroDomainStore`, `compute_paxg_basis_points`)
- Real Public Web3 Collectors (`HyperliquidWeb3Collector`, `BinanceVisionWeb3Collector`,
  `GateWeb3Collector`, `OKXWeb3Collector`, `DeribitDVOLCollector`, `DefiLlamaStablecoinCollector`)
- Real Public Gold & Macro Collectors (`YahooGoldMacroCollector`, `USTreasuryRealYieldCollector`,
  `FedFOMCCalendarCollector`, `BLSMacroEventCollector`)
- Full Real Multi-Year Ingestion Entrypoint (`ingest_platform_v4_real_lake`)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from alphaloop.platform.assets import (
    ASSET_CLASS_CRYPTO_SPOT,
    ASSET_CLASS_GOLD_PERP,
    ASSET_CLASS_GOLD_TRADFI,
    ASSET_CLASS_MACRO_INDEX,
    GOLD_MACRO_SYMBOLS_V4,
    WEB3_LIQUID_UNIVERSE,
    default_gold_specs,
    default_macro_specs_v4,
    default_web3_specs,
)
from alphaloop.platform.data_foundation.collectors_gold_macro import (
    BLSMacroEventCollector,
    DIVIDEND_ADJ_CLOSE_SYMBOLS,
    FedFOMCCalendarCollector,
    OFFICIAL_CPI_RELEASE_DATES,
    USTreasuryRealYieldCollector,
    YahooGoldMacroCollector,
    bls_cpi_release_date,
    bls_nfp_release_date,
    ny_local_to_utc_ms,
    us_federal_holidays,
)
from alphaloop.platform.data_foundation.collectors_web3 import (
    BinanceVisionWeb3Collector,
    DefiLlamaStablecoinCollector,
    DeribitDVOLCollector,
    GateWeb3Collector,
    HyperliquidWeb3Collector,
    OKXWeb3Collector,
    RateLimitedHTTPClient,
    normalize_binance_ts_ms,
    parse_zip_csv_rows,
    snap_to_nearest_hour_ms,
)
from alphaloop.platform.data_foundation.models import (
    DAY_MS,
    HOUR_MS,
    MAX_PLATFORM_EPOCH_MS,
    MIN_PLATFORM_EPOCH_MS,
    CanonicalBar,
    DomainStoreProtocol,
    MacroEventRelease,
    MacroYieldPoint,
    MultiAssetDQCheck,
    MultiAssetDQReport,
    PAXGBasisPoint,
    Web3DerivativesMetric,
    Web3MarketMacroPoint,
    audit_bars_quality,
    check_canonical_bar,
    classify_tradfi_session_tag,
    fingerprint_canonical_bars,
    resample_bars,
    validate_canonical_bar,
    validate_macro_event_release,
    validate_macro_yield_point,
    validate_paxg_basis_point,
    validate_web3_derivatives_metric,
    validate_web3_market_macro_point,
)
from alphaloop.platform.data_foundation.router import (
    ASSET_CLASS_TO_DOMAIN,
    DataFoundationRouter,
    GoldMacroDomainStore,
    Web3DomainStore,
    compute_paxg_basis_points,
    has_duckdb,
    has_pyarrow,
    is_comex_weekend_utc,
    normalize_domain,
    validate_readonly_sql,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HOUR_MS",
    "DAY_MS",
    "MIN_PLATFORM_EPOCH_MS",
    "MAX_PLATFORM_EPOCH_MS",
    "CanonicalBar",
    "Web3DerivativesMetric",
    "Web3MarketMacroPoint",
    "MacroYieldPoint",
    "MacroEventRelease",
    "PAXGBasisPoint",
    "MultiAssetDQCheck",
    "MultiAssetDQReport",
    "audit_bars_quality",
    "DomainStoreProtocol",
    "check_canonical_bar",
    "validate_canonical_bar",
    "validate_web3_derivatives_metric",
    "validate_web3_market_macro_point",
    "validate_macro_yield_point",
    "validate_macro_event_release",
    "validate_paxg_basis_point",
    "fingerprint_canonical_bars",
    "classify_tradfi_session_tag",
    "resample_bars",
    "ASSET_CLASS_TO_DOMAIN",
    "DataFoundationRouter",
    "Web3DomainStore",
    "GoldMacroDomainStore",
    "compute_paxg_basis_points",
    "has_pyarrow",
    "has_duckdb",
    "is_comex_weekend_utc",
    "normalize_domain",
    "validate_readonly_sql",
    "RateLimitedHTTPClient",
    "HyperliquidWeb3Collector",
    "BinanceVisionWeb3Collector",
    "GateWeb3Collector",
    "OKXWeb3Collector",
    "DeribitDVOLCollector",
    "DefiLlamaStablecoinCollector",
    "normalize_binance_ts_ms",
    "parse_zip_csv_rows",
    "snap_to_nearest_hour_ms",
    "YahooGoldMacroCollector",
    "USTreasuryRealYieldCollector",
    "FedFOMCCalendarCollector",
    "BLSMacroEventCollector",
    "DIVIDEND_ADJ_CLOSE_SYMBOLS",
    "OFFICIAL_CPI_RELEASE_DATES",
    "bls_nfp_release_date",
    "bls_cpi_release_date",
    "ny_local_to_utc_ms",
    "us_federal_holidays",
    "ingest_platform_v4_real_lake",
]


def ingest_platform_v4_real_lake(
    base_dir: Path | str,
    *,
    v3_sqlite_path: Path | str | None = None,
    hl_days_12h: int = 1095,
    hl_days_4h: int = 833,
    binance_start_ym: str = "2023-01",
    binance_end_ym: str = "2026-10",
    binance_metrics_days: int = 365,
    gate_days: int = 179,
    okx_days: int = 365,
    yahoo_daily_range: str = "10y",
    yahoo_intraday_range: str = "730d",
    skip_existing_hl: bool = True,
) -> dict[str, Any]:
    """Execute 100% real multi-year public ingestion into Option B partitioned Data Foundation v4.

    Zero synthetic data fallback. Any upstream error is logged and surfaced in `summary["errors"]`.
    Checkpoints every coin/series immediately to SQLite WAL stores so interrupted runs resume cleanly.
    """
    base = Path(base_dir)
    summary: dict[str, Any] = {
        "base_dir": str(base),
        "v3_migration": {},
        "hyperliquid": {},
        "binance_um": {},
        "binance_spot": {},
        "binance_metrics": {},
        "gate": {},
        "okx": {},
        "web3_macro": {},
        "yahoo_gold_macro": {},
        "us_treasury": {},
        "macro_events": {},
        "paxg_basis": {},
        "exports": {},
        "dq_audit": {},
        "errors": [],
    }

    client = RateLimitedHTTPClient()
    hl = HyperliquidWeb3Collector(client=client)
    bvision = BinanceVisionWeb3Collector(client=client)
    gate = GateWeb3Collector(client=client)
    okx = OKXWeb3Collector(client=client)
    deribit = DeribitDVOLCollector(client=client)
    llama = DefiLlamaStablecoinCollector(client=client)
    yahoo = YahooGoldMacroCollector()
    treasury = USTreasuryRealYieldCollector()
    fomc = FedFOMCCalendarCollector()
    bls = BLSMacroEventCollector(cache_path=base / "cache" / "bls_v2_cache.json")

    with DataFoundationRouter(base) as router:
        # 1. Idempotent strictly read-only migration from v3 SQLite BEFORE v4 spec registration
        #    so authoritative v4 specs (contract_multiplier, max_leverage_cap, tags) always take precedence.
        if v3_sqlite_path is not None and Path(v3_sqlite_path).exists():
            summary["v3_migration"] = router.migrate_from_v3_sqlite(v3_sqlite_path)
            logger.info("Migrated v3 SQLite: %s", summary["v3_migration"])

        # 2. Register authoritative v4 asset specifications
        hl_web3_specs = default_web3_specs(WEB3_LIQUID_UNIVERSE, venue="hyperliquid")
        bm_um_specs = default_web3_specs(("BTC", "ETH", "SOL", "PAXG"), venue="binance_um")
        bm_spot_specs = default_web3_specs(("BTC", "PAXG"), venue="binance_spot")
        gate_specs = default_web3_specs(("BTC", "ETH", "SOL", "PAXG"), venue="gate")
        okx_specs = default_web3_specs(("BTC", "ETH", "SOL"), venue="okx")
        gold_specs = default_gold_specs()
        macro_v4_specs = default_macro_specs_v4()

        for spec in hl_web3_specs.values():
            router.register_asset(spec, domain="web3")
        for sym, spec in bm_um_specs.items():
            router.register_asset(spec, domain="gold_macro" if sym == "PAXG" else "web3")
        for sym, spec in bm_spot_specs.items():
            router.register_asset(spec, domain="gold_macro" if sym == "PAXG" else "web3")
        for sym, spec in gate_specs.items():
            router.register_asset(spec, domain="gold_macro" if sym == "PAXG" else "web3")
        for spec in okx_specs.values():
            router.register_asset(spec, domain="web3")
        for spec in gold_specs.values():
            router.register_asset(spec, domain="gold_macro")
        for spec in macro_v4_specs.values():
            router.register_asset(spec, domain="gold_macro")

        # 3. Hyperliquid 12h (3y) + 4h (833d) + paginated fundingHistory across ALL 13 coins (checkpointed per coin)
        all_hl_coins = list(WEB3_LIQUID_UNIVERSE) + ["PAXG"]
        # Core coins fetch full fresh fundingHistory (for 100% real `premium_basis_bps` right up to latest bar);
        # remaining 9 altcoins reuse migrated v3 1h funding seed to save ~90 HL API pages cleanly.
        full_premium_coins = {"BTC", "ETH", "SOL", "PAXG"}
        for sym in all_hl_coins:
            dom = "gold_macro" if sym == "PAXG" else "web3"
            ac = ASSET_CLASS_GOLD_PERP if sym == "PAXG" else "CRYPTO_PERP"
            if skip_existing_hl:
                existing_12h = router.load_bars(dom, "hyperliquid", sym, 43_200)
                existing_4h = router.load_bars(dom, "hyperliquid", sym, 14_400)
                min_12h_ok = 1_100 if sym == "PAXG" else (1_350 if sym == "HYPE" else 2_000)
                min_4h_ok = 3_400 if sym == "PAXG" else (4_100 if sym == "HYPE" else 4_800)
                if len(existing_12h) >= min_12h_ok and len(existing_4h) >= min_4h_ok:
                    summary["hyperliquid"][sym] = {
                        "bars_12h": len(existing_12h),
                        "bars_4h": len(existing_4h),
                        "reused_checkpoint": True,
                    }
                    continue
            try:
                seed_map: dict[int, tuple[float, float]] | None = None
                if sym not in full_premium_coins:
                    v3_1h = router.load_bars(dom, "hyperliquid", sym, 3_600)
                    if v3_1h:
                        seed_map = {
                            b.ts_event_ms: (b.carry_rate_period, b.carry_interval_h)
                            for b in v3_1h
                            if b.carry_interval_h > 0.0 and abs(b.carry_rate_period) > 0.0
                        }
                bars_12h, fmap = hl.fetch_bars(
                    sym,
                    interval="12h",
                    days=hl_days_12h,
                    asset_class=ac,
                    seed_hourly_map=seed_map,
                    trim_to_funding_span=True,
                )
                bars_4h, _ = hl.fetch_bars(
                    sym,
                    interval="4h",
                    days=hl_days_4h,
                    asset_class=ac,
                    funding_map=fmap,
                    trim_to_funding_span=True,
                )
                n12 = router.upsert_bars(dom, bars_12h, allow_synthetic_in_test=False)
                n4 = router.upsert_bars(dom, bars_4h, allow_synthetic_in_test=False)
                summary["hyperliquid"][sym] = {
                    "domain": dom,
                    "bars_12h": n12,
                    "bars_4h": n4,
                    "funding_rows": len(fmap),
                    "seeded_from_v3": bool(seed_map),
                }
                logger.info("HL %s checkpointed: 12h=%d, 4h=%d, funding=%d", sym, n12, n4, len(fmap))
            except Exception as exc:
                msg = f"HL ingestion failed for {sym}: {exc}"
                logger.error(msg)
                summary["errors"].append(msg)

        # 4. Binance Vision S3: UM 4h+12h (klines + funding + premiumIndex)
        um_targets = [
            ("BTCUSDT", "BTC", "web3", binance_start_ym),
            ("ETHUSDT", "ETH", "web3", binance_start_ym),
            ("SOLUSDT", "SOL", "web3", binance_start_ym),
            ("PAXGUSDT", "PAXG", "gold_macro", "2025-03"),
        ]
        for exch_sym, sym, dom, s_ym in um_targets:
            ac = ASSET_CLASS_GOLD_PERP if sym == "PAXG" else "CRYPTO_PERP"
            sym_res: dict[str, int] = {}
            for iv in ("4h", "12h"):
                try:
                    bars = bvision.fetch_um_bars(
                        exch_sym,
                        interval=iv,
                        start_ym=s_ym,
                        end_ym=binance_end_ym,
                        asset_class=ac,
                        trim_to_funding_span=True,
                    )
                    n_up = router.upsert_bars(dom, bars, allow_synthetic_in_test=False)
                    sym_res[f"bars_{iv}"] = n_up
                except Exception as exc:
                    msg = f"Binance UM {exch_sym} {iv} failed: {exc}"
                    logger.error(msg)
                    summary["errors"].append(msg)
            summary["binance_um"][sym] = sym_res

        # 5. Binance Vision S3 + data-api Spot 4h+12h (BTCUSDT -> web3, PAXGUSDT -> gold_macro)
        spot_targets = [
            ("BTCUSDT", "BTC", "web3"),
            ("PAXGUSDT", "PAXG", "gold_macro"),
        ]
        for exch_sym, sym, dom in spot_targets:
            sym_res = {}
            for iv in ("4h", "12h"):
                try:
                    bars = bvision.fetch_spot_bars(
                        exch_sym,
                        interval=iv,
                        start_ym=binance_start_ym,
                        end_ym=binance_end_ym,
                        include_rest_tail=True,
                        asset_class=ASSET_CLASS_CRYPTO_SPOT,
                    )
                    n_up = router.upsert_bars(dom, bars, allow_synthetic_in_test=False)
                    sym_res[f"bars_{iv}"] = n_up
                except Exception as exc:
                    msg = f"Binance Spot {exch_sym} {iv} failed: {exc}"
                    logger.error(msg)
                    summary["errors"].append(msg)
            summary["binance_spot"][sym] = sym_res

        # 6. Binance Vision daily 5m metrics (365d -> causal 4h) into web3_derivatives_metrics
        for exch_sym, sym in (("BTCUSDT", "BTC"), ("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"), ("PAXGUSDT", "PAXG")):
            try:
                metrics = bvision.fetch_um_metrics(
                    exch_sym,
                    days=binance_metrics_days,
                    downsample_tf_h=4.0,
                )
                n_m = router.upsert_web3_metrics(metrics, allow_synthetic_in_test=False)
                summary["binance_metrics"][sym] = n_m
            except Exception as exc:
                msg = f"Binance metrics {exch_sym} failed: {exc}"
                logger.error(msg)
                summary["errors"].append(msg)

        # 7. Gate.io v4 (BTC, ETH, SOL, PAXG 4h+12h, 179d funding-matched)
        for sym in ("BTC", "ETH", "SOL", "PAXG"):
            dom = "gold_macro" if sym == "PAXG" else "web3"
            ac = ASSET_CLASS_GOLD_PERP if sym == "PAXG" else "CRYPTO_PERP"
            sym_res = {}
            for iv in ("4h", "12h"):
                try:
                    bars = gate.fetch_bars(sym, interval=iv, days=gate_days, asset_class=ac, trim_to_funding_span=True)
                    sym_res[f"bars_{iv}"] = router.upsert_bars(dom, bars, allow_synthetic_in_test=False)
                except Exception as exc:
                    msg = f"Gate {sym} {iv} failed: {exc}"
                    logger.error(msg)
                    summary["errors"].append(msg)
            summary["gate"][sym] = sym_res

        # 8. OKX v5 (BTC, ETH, SOL 4h+12h, 365d bars / ~95d funding)
        for sym in ("BTC", "ETH", "SOL"):
            sym_res = {}
            for iv in ("4h", "12h"):
                try:
                    bars = okx.fetch_bars(sym, interval=iv, days=okx_days, trim_to_funding_span=False)
                    sym_res[f"bars_{iv}"] = router.upsert_bars("web3", bars, allow_synthetic_in_test=False)
                except Exception as exc:
                    msg = f"OKX {sym} {iv} failed: {exc}"
                    logger.error(msg)
                    summary["errors"].append(msg)
            summary["okx"][sym] = sym_res

        # 9. Deribit BTC DVOL & DefiLlama Stablecoin Supply
        try:
            dvol_pts = deribit.fetch_dvol()
            summary["web3_macro"]["BTC_DVOL"] = router.upsert_web3_macro(dvol_pts, allow_synthetic_in_test=False)
        except Exception as exc:
            msg = f"Deribit BTC_DVOL failed: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

        try:
            llama_pts = llama.fetch_stablecoin_mcap()
            summary["web3_macro"]["STABLECOIN_MCAP_USD"] = router.upsert_web3_macro(
                llama_pts, allow_synthetic_in_test=False
            )
        except Exception as exc:
            msg = f"DefiLlama STABLECOIN_MCAP_USD failed: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

        # 10. Yahoo Finance v8 ALL 13 Gold & Macro symbols: 1d (10y fallback 5y) + 1h->4h (730d, min_subbars=2)
        for sym in GOLD_MACRO_SYMBOLS_V4:
            ac = ASSET_CLASS_GOLD_TRADFI if sym in ("GC=F", "GLD") else ASSET_CLASS_MACRO_INDEX
            sym_res = {}
            try:
                bars_1d = yahoo.fetch_bars(sym, asset_class=ac, range_str=yahoo_daily_range, interval="1d")
                if len(bars_1d) < 100 and yahoo_daily_range != "5y":
                    bars_1d = yahoo.fetch_bars(sym, asset_class=ac, range_str="5y", interval="1d")
                sym_res["bars_1d"] = router.upsert_bars("gold_macro", bars_1d, allow_synthetic_in_test=False)
            except Exception as exc:
                msg = f"Yahoo 1d {sym} failed: {exc}"
                logger.error(msg)
                summary["errors"].append(msg)

            try:
                bars_1h, bars_4h = yahoo.fetch_4h_bars(
                    sym, asset_class=ac, range_str=yahoo_intraday_range, min_subbars=2
                )
                sym_res["bars_1h_clean"] = len(bars_1h)
                sym_res["bars_4h"] = router.upsert_bars("gold_macro", bars_4h, allow_synthetic_in_test=False)
            except Exception as exc:
                msg = f"Yahoo 1h->4h {sym} failed: {exc}"
                logger.error(msg)
                summary["errors"].append(msg)
            summary["yahoo_gold_macro"][sym] = sym_res

        # 11. US Treasury Official TIPS Real Yield Curve (2021..2026)
        try:
            yields = treasury.fetch_years(2021, 2026)
            summary["us_treasury"]["rows"] = router.upsert_yield_curve(yields, allow_synthetic_in_test=False)
        except Exception as exc:
            msg = f"US Treasury real yield fetch failed: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

        # 12. Federal Reserve FOMC Calendar (2021..2027) + BLS API v2 CPI/NFP (2021..2026)
        try:
            fomc_events = fomc.fetch_calendar(min_year=2021, max_year=2027)
            summary["macro_events"]["FOMC"] = router.upsert_macro_events(fomc_events, allow_synthetic_in_test=False)
        except Exception as exc:
            msg = f"Fed FOMC calendar fetch failed: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

        try:
            bls_events = bls.fetch_events(start_year=2020, end_year=2026, schedule_start_year=2021, schedule_end_year=2026)
            summary["macro_events"]["BLS_NFP_CPI"] = router.upsert_macro_events(
                bls_events, allow_synthetic_in_test=False
            )
        except Exception as exc:
            msg = f"BLS NFP/CPI fetch failed: {exc}"
            logger.error(msg)
            summary["errors"].append(msg)

        # 13. Compute & persist PAXG dual-basis arbitration (4h & 12h, HL & Binance UM vs Binance Spot vs GC=F)
        gc_4h = router.load_bars("gold_macro", "yahoo", "GC=F", 14_400)
        gc_1d = router.load_bars("gold_macro", "yahoo", "GC=F", 86_400)
        for v_perp in ("hyperliquid", "binance_um"):
            for tf_sec, gc_ref in ((14_400, gc_4h if gc_4h else gc_1d), (43_200, gc_4h if gc_4h else gc_1d)):
                perp_bars = router.load_bars("gold_macro", v_perp, "PAXG", tf_sec)
                spot_bars = router.load_bars("gold_macro", "binance_spot", "PAXG", tf_sec)
                if perp_bars and (spot_bars or gc_ref):
                    bpts = compute_paxg_basis_points(
                        perp_bars,
                        spot_bars,
                        gc_ref,
                        venue_perp=v_perp,
                        venue_spot="binance_spot",
                        ref_tradfi_symbol="GC=F",
                    )
                    n_bp = router.upsert_paxg_basis(bpts, allow_synthetic_in_test=False)
                    summary["paxg_basis"][f"{v_perp}:{tf_sec}s"] = n_bp

        # 14. Export deterministic partitions + SHA-256 manifests
        summary["exports"]["web3"] = router.export_domain_partition("web3", fmt="auto")
        summary["exports"]["gold_macro"] = router.export_domain_partition("gold_macro", fmt="auto")

        # 15. Run DQ-MA-01..07 audits & persist certificates to catalog.sqlite
        dq_summary: dict[str, dict[str, Any]] = {}
        for sym, spec in hl_web3_specs.items():
            for tf_sec in (14_400, 43_200):
                rep = router.audit_domain_dq("web3", "hyperliquid", sym, tf_sec, spec)
                dq_summary[f"web3:hyperliquid:{sym}:{tf_sec}s"] = {
                    "passed": rep.passed,
                    "rows": rep.row_count,
                    "failed_gates": [c.gate_id for c in rep.checks if not c.passed],
                }
        for sym in ("BTC", "ETH", "SOL"):
            spec = bm_um_specs[sym]
            for tf_sec in (14_400, 43_200):
                rep = router.audit_domain_dq("web3", "binance_um", sym, tf_sec, spec)
                dq_summary[f"web3:binance_um:{sym}:{tf_sec}s"] = {
                    "passed": rep.passed,
                    "rows": rep.row_count,
                    "failed_gates": [c.gate_id for c in rep.checks if not c.passed],
                }
        for v_paxg, spec in (("hyperliquid", gold_specs["PAXG"]), ("binance_um", bm_um_specs["PAXG"]), ("binance_spot", bm_spot_specs["PAXG"])):
            for tf_sec in (14_400, 43_200):
                rep = router.audit_domain_dq("gold_macro", v_paxg, "PAXG", tf_sec, spec)
                dq_summary[f"gold_macro:{v_paxg}:PAXG:{tf_sec}s"] = {
                    "passed": rep.passed,
                    "rows": rep.row_count,
                    "failed_gates": [c.gate_id for c in rep.checks if not c.passed],
                }
        for sym, spec in macro_v4_specs.items():
            for tf_sec in (86_400, 14_400):
                rep = router.audit_domain_dq("gold_macro", "yahoo", sym, tf_sec, spec)
                dq_summary[f"gold_macro:yahoo:{sym}:{tf_sec}s"] = {
                    "passed": rep.passed,
                    "rows": rep.row_count,
                    "failed_gates": [c.gate_id for c in rep.checks if not c.passed],
                }
        summary["dq_audit"] = dq_summary
        summary["stats"] = router.summary_stats()

    return summary
