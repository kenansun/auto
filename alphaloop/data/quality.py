"""Deterministic Data Quality Certification Engine (DQ-01 .. DQ-08, Architect P0-2 Hardened).

Provides machine-verifiable certification before any dataset may enter backtesting,
fast micro-runs, or strategy promotion:
  - DATA_QUALIFIED_FULL:         All blocking gates + DQ-07 dual-venue price/basis >= 90%.
  - DATA_QUALIFIED_FUNDING_ONLY: All blocking gates pass, but dual-venue basis < 90%
                                 (usable for research/micro-run; BLOCKED from direct LIVE).
  - DATA_REJECTED:               One or more blocking gates (DQ-01..DQ-06, DQ-08) failed.

Architect P0-2 defenses:
  - DQ-02: Gap tolerance scales cleanly with venue settlement cadence (1h vs 4h vs 8h).
  - DQ-03: Non-decreasing per (venue, symbol) for BBO burst ticks; blocks <2020 monotonic leaks.
  - DQ-04: Coarser-grid alignment for 1h vs 8h venue pairs (matches >=95% of coarse bars).
  - DQ-05: Cross-venue median price ratio sanity (<=2% perp-perp) catches USDCUSDT/1000PEPE
           mismapping + cross-venue jump divergence avoids false positives on synchronized crashes.
  - DQ-06: Distribution checks (median<=150% APR, p95<=1500%, max_single_period<=0.05) catch
           100x unit bugs without false-positiving legitimate altcoin funding spikes.
  - DQ-08: Frozen-feed check runs ONLY on OHLCV/BBO (NEVER on clamped baseline funding rates),
           and staleness SLA applies only in LIVE mode.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import statistics
from pathlib import Path
from typing import Any

from alphaloop.core.models import DataQualityCheck, DataQualityReport
from alphaloop.core.provenance import PROVENANCE_REAL, sha256_obj, utc_now_ms
from alphaloop.data.store import MarketDataStore

logger = logging.getLogger(__name__)

ALLOWED_REAL_PROVENANCES = {PROVENANCE_REAL, "SERVER_COLLECTED_LIVE"}
MIN_VALID_EPOCH_MS = 1_577_836_800_000  # 2020-01-01T00:00:00Z (blocks CLOCK_MONOTONIC leakage)
HOUR_MS = 3600 * 1000


class DataQualityAuditor:
    """Runs deterministic DQ-01..DQ-08 audits across SQLite lakes, BBO ticks, and server CSVs."""

    def __init__(self, store: MarketDataStore | None = None) -> None:
        self.store = store

    def audit_funding_pair(
        self,
        venue_a: str,
        venue_b: str,
        symbols: list[str],
        start_ms: int | None = None,
        end_ms: int | None = None,
        dataset_name: str | None = None,
        persist: bool = True,
        allow_synthetic_in_unit_test: bool = False,
        require_fresh_hours: float | None = None,
        max_price_ratio_dev_pct: float = 2.5,
    ) -> DataQualityReport:
        """Audit a cross-venue funding + 1h candle slice in MarketDataStore."""
        if self.store is None:
            raise ValueError("MarketDataStore is required for audit_funding_pair")

        now_ms = utc_now_ms()
        is_spot_perp = venue_b.endswith("_spot")
        # Include BOTH venues in dataset fingerprint so spot candles are cryptographically bound too (Red-Team #9)
        fp_venues = sorted({venue_a, venue_b})
        ds_sha, prov, row_count = self.store.fingerprint_slice(fp_venues, symbols, start_ms, end_ms)

        checks: list[DataQualityCheck] = []
        symbol_scores: dict[str, str] = {}

        # DQ-01: Provenance & Cryptographic Fingerprint (covers BOTH funding_rates & candles_1h)
        prov_parts = set(prov.split("+")) if prov else set()
        dq1_ok = (
            row_count >= 50
            and bool(ds_sha)
            and (
                (bool(prov_parts) and prov_parts.issubset(ALLOWED_REAL_PROVENANCES))
                or allow_synthetic_in_unit_test
            )
        )
        checks.append(
            DataQualityCheck(
                gate_id="DQ-01",
                name="Provenance & Cryptographic Fingerprint",
                passed=dq1_ok,
                severity="BLOCKING",
                value=f"{prov} ({row_count} rows)",
                threshold="REAL_EXCHANGE_API & rows>=50",
                detail=f"sha256={ds_sha[:12]}",
            )
        )

        coverages: list[float] = []
        max_gap_ratio = 0.0
        bad_ts_count = 0
        non_monotonic_count = 0
        interval_mismatches = 0
        abs_aprs: list[float] = []
        per_sym_unit_anomalies: list[str] = []
        max_single_period_rate = 0.0
        coarse_align_ratios: list[float] = []
        candle_corrupt = 0
        poison_divergent_jumps = 0
        symbol_mapping_mismatches: list[str] = []
        max_frozen_hours = 0
        max_flat_ohlc_hours = 0
        dual_price_hits = 0
        dual_price_total = 0
        latest_ts_seen = 0

        for sym in symbols:
            fa = self.store.load_funding(venue_a, sym, start_ms, end_ms)
            fb = [] if is_spot_perp else self.store.load_funding(venue_b, sym, start_ms, end_ms)
            ca = self.store.load_candles(venue_a, sym, start_ms, end_ms)
            cb = self.store.load_candles(venue_b, sym, start_ms, end_ms)

            sym_issues: list[str] = []

            for vname, bars in ((venue_a, fa), (venue_b, fb)):
                if is_spot_perp and vname == venue_b:
                    continue
                if len(bars) < 2:
                    sym_issues.append(f"{vname}:sparse({len(bars)})")
                    coverages.append(0.0)
                    continue

                deltas_h: list[float] = []
                sym_abs_aprs: list[float] = []
                declared_h = statistics.median([b.interval_hours for b in bars if b.interval_hours > 0] or [1.0])
                prev_ts = -1
                for b in bars:
                    if b.ts_ms > latest_ts_seen:
                        latest_ts_seen = b.ts_ms
                    if b.ts_ms < MIN_VALID_EPOCH_MS or b.ts_ms > now_ms + 300_000:
                        bad_ts_count += 1
                    if prev_ts >= 0:
                        if b.ts_ms <= prev_ts:
                            non_monotonic_count += 1
                        else:
                            deltas_h.append((b.ts_ms - prev_ts) / float(HOUR_MS))
                    prev_ts = b.ts_ms
                    if not math.isfinite(b.funding_rate) or not math.isfinite(b.interval_hours) or b.interval_hours <= 0:
                        candle_corrupt += 1
                    else:
                        a_apr = abs(b.apr_pct)
                        abs_aprs.append(a_apr)
                        sym_abs_aprs.append(a_apr)
                        if abs(b.funding_rate) > max_single_period_rate:
                            max_single_period_rate = abs(b.funding_rate)

                if sym_abs_aprs:
                    s_sorted = sorted(sym_abs_aprs)
                    s_med = statistics.median(s_sorted)
                    s_p95 = s_sorted[int(0.95 * (len(s_sorted) - 1))]
                    if s_med > 150.0 or s_p95 > 1500.0:
                        per_sym_unit_anomalies.append(f"{vname}:{sym}(med={s_med:.0f}%,p95={s_p95:.0f}%)")
                        sym_issues.append(f"{vname}:funding_unit_anomaly")

                if deltas_h:
                    med_delta = statistics.median(deltas_h)
                    max_delta = max(deltas_h)
                    # Cadence-switch-aware coverage: handles CEX 8h<->4h switches cleanly without masking real gaps
                    base_h = max(declared_h, med_delta)
                    normal_cap = max(8.0, base_h) if base_h >= 2.0 else base_h
                    expected_intervals = sum(
                        1.0 if d <= normal_cap + 0.05 else (d / max(0.5, base_h))
                        for d in deltas_h
                    )
                    # Also account for missing head/tail coverage when an explicit [start_ms, end_ms] window is audited
                    if start_ms is not None and end_ms is not None and end_ms > start_ms:
                        head_gap_h = max(0.0, (bars[0].ts_ms - start_ms) / float(HOUR_MS) - normal_cap)
                        tail_gap_h = max(0.0, (end_ms - bars[-1].ts_ms) / float(HOUR_MS) - normal_cap)
                        if head_gap_h > 0.0:
                            expected_intervals += head_gap_h / max(0.5, base_h)
                        if tail_gap_h > 0.0:
                            expected_intervals += tail_gap_h / max(0.5, base_h)
                    cov = min(1.0, len(bars) / max(1.0, expected_intervals + 1.0))
                    coverages.append(cov)
                    gap_limit = max(6.0, 2.5 * base_h)
                    max_gap_ratio = max(max_gap_ratio, max_delta / gap_limit)
                    if abs(med_delta - declared_h) / max(declared_h, 0.1) > 0.25:
                        interval_mismatches += 1
                        sym_issues.append(f"{vname}:interval_mismatch(med={med_delta:.2f}h vs {declared_h:.2f}h)")

            # DQ-04: Coarser-grid alignment (works accurately for 1h DEX vs 8h CEX!)
            if not is_spot_perp and len(fa) >= 2 and len(fb) >= 2:
                coarse, fine = (fb, fa) if len(fb) <= len(fa) else (fa, fb)
                fine_ts_set = {(b.ts_ms // HOUR_MS) * HOUR_MS for b in fine}
                matched = sum(1 for b in coarse if ((b.ts_ms // HOUR_MS) * HOUR_MS) in fine_ts_set)
                coarse_align_ratios.append(matched / float(len(coarse)))
            elif is_spot_perp and len(fa) >= 2:
                coarse_align_ratios.append(1.0)

            # Candle sanity, intra-bar wick anomalies, cross-venue mapping ratio, divergent jumps, frozen feed
            ca_map = {c.ts_ms: c for c in ca}
            cb_map = {c.ts_ms: c for c in cb}
            for cname, cbars in ((venue_a, ca), (venue_b, cb)):
                prev_bar = None
                frozen = 0
                flat_ohlc = 0
                prev_cts = -1
                other_map = cb_map if cname == venue_a else ca_map
                for c in cbars:
                    if c.ts_ms > latest_ts_seen:
                        latest_ts_seen = c.ts_ms
                    if c.ts_ms < MIN_VALID_EPOCH_MS or c.ts_ms > now_ms + 300_000:
                        bad_ts_count += 1
                    if prev_cts >= 0 and c.ts_ms <= prev_cts:
                        non_monotonic_count += 1
                    prev_cts = c.ts_ms

                    vals = (c.open, c.high, c.low, c.close)
                    if any((not math.isfinite(v) or v <= 0.0) for v in vals):
                        candle_corrupt += 1
                        continue
                    if c.low > min(c.open, c.close) * 1.0001 or c.high < max(c.open, c.close) * 0.9999:
                        candle_corrupt += 1

                    # Intra-candle extreme wick sanity (Red-Team #9): catches >80% bad prints inside 1h candle
                    wick_range = (c.high - c.low) / c.open
                    if wick_range > 0.80:
                        o_cur = other_map.get(c.ts_ms)
                        if o_cur and o_cur.open > 0:
                            o_wick = (o_cur.high - o_cur.low) / o_cur.open
                            if o_wick < 0.25:
                                poison_divergent_jumps += 1
                        else:
                            poison_divergent_jumps += 1

                    if prev_bar is not None and prev_bar.close > 0:
                        jump = abs(c.close - prev_bar.close) / prev_bar.close
                        if jump > 0.35:
                            # Cross-venue jump divergence check (P0-2.4): only poison if other venue didn't move!
                            o_cur = other_map.get(c.ts_ms)
                            o_prev = other_map.get(prev_bar.ts_ms)
                            if o_cur and o_prev and o_prev.close > 0:
                                o_jump = abs(o_cur.close - o_prev.close) / o_prev.close
                                if o_jump < 0.10:
                                    poison_divergent_jumps += 1
                            elif jump > 0.65:
                                poison_divergent_jumps += 1

                        # Anti-frozen check on OHLC: zero-vol >6h OR prolonged identical flatline >12h even with vol>0
                        if c.open == c.high == c.low == c.close == prev_bar.close:
                            flat_ohlc += 1
                            if flat_ohlc > max_flat_ohlc_hours:
                                max_flat_ohlc_hours = flat_ohlc
                            if c.volume_usd == 0.0:
                                frozen += 1
                                if frozen > max_frozen_hours:
                                    max_frozen_hours = frozen
                            else:
                                frozen = 0
                        else:
                            frozen = 0
                            flat_ohlc = 0
                    prev_bar = c

            # Cross-venue median price ratio sanity (catches WOB USDCUSDT & kPEPE/1000PEPE bugs!)
            common_px_ts = sorted(set(ca_map.keys()) & set(cb_map.keys()))
            if len(common_px_ts) >= 10:
                devs = [
                    abs((ca_map[t].open / cb_map[t].open) - 1.0) * 100.0
                    for t in common_px_ts
                    if cb_map[t].open > 0
                ]
                if devs:
                    med_dev_pct = statistics.median(devs)
                    if med_dev_pct > max_price_ratio_dev_pct:
                        symbol_mapping_mismatches.append(f"{sym}:{med_dev_pct:.1f}%")
                        sym_issues.append(f"price_ratio_mismatch({med_dev_pct:.1f}%)")

            # Anchor dual-candle completeness on hourly candle union (or funding bars when both lack candles)
            ref_hours = len(set(ca_map.keys()) | set(cb_map.keys()))
            if ref_hours == 0:
                ref_hours = len({(b.ts_ms // HOUR_MS) * HOUR_MS for b in fa})
            if ref_hours > 0:
                dual_price_total += ref_hours
                dual_price_hits += len(common_px_ts)

            symbol_scores[sym] = "OK" if not sym_issues else ",".join(sym_issues)

        avg_cov = (sum(coverages) / len(coverages) * 100.0) if coverages else 0.0
        min_cov = (min(coverages) * 100.0) if coverages else 0.0
        # Enforce both average coverage >=95%, per-symbol floor >=85%, and max_gap <= gap_limit (ratio<=1.0x)
        dq2_ok = avg_cov >= 95.0 and min_cov >= 85.0 and max_gap_ratio <= 1.0
        checks.append(
            DataQualityCheck(
                gate_id="DQ-02",
                name="Bar Coverage & Cadence-Scaled Gap Continuity",
                passed=dq2_ok,
                severity="BLOCKING",
                value=f"avg_cov={avg_cov:.2f}%, min_cov={min_cov:.2f}%, max_gap_ratio={max_gap_ratio:.2f}x",
                threshold="avg_cov>=95.0% & min_cov>=85.0% & max_gap_ratio<=1.0x",
                detail=f"symbols_checked={len(symbols)}",
            )
        )

        dq3_ok = bad_ts_count == 0 and non_monotonic_count == 0
        checks.append(
            DataQualityCheck(
                gate_id="DQ-03",
                name="Timestamp Monotonicity & Clock Hygiene",
                passed=dq3_ok,
                severity="BLOCKING",
                value=f"bad_ts={bad_ts_count}, non_mono={non_monotonic_count}",
                threshold="bad_ts=0 & non_mono=0",
                detail="Rejects CLOCK_MONOTONIC leakage (<2020) and future timestamps",
            )
        )

        avg_align = (sum(coarse_align_ratios) / len(coarse_align_ratios) * 100.0) if coarse_align_ratios else 0.0
        dq4_ok = avg_align >= 95.0
        checks.append(
            DataQualityCheck(
                gate_id="DQ-04",
                name="Coarser-Grid Cross-Venue Settlement Alignment",
                passed=dq4_ok,
                severity="BLOCKING",
                value=f"coarse_align={avg_align:.2f}%",
                threshold=">=95.0%",
                detail=f"pairs_checked={len(coarse_align_ratios)}",
            )
        )

        dq5_ok = candle_corrupt == 0 and poison_divergent_jumps == 0 and len(symbol_mapping_mismatches) == 0
        checks.append(
            DataQualityCheck(
                gate_id="DQ-05",
                name="OHLCV Integrity, Poison Jump & Symbol Mapping Ratio",
                passed=dq5_ok,
                severity="BLOCKING",
                value=f"corrupt={candle_corrupt}, poison_jumps={poison_divergent_jumps}, px_mismatch={len(symbol_mapping_mismatches)}",
                threshold=f"all 0 & med_px_dev<={max_price_ratio_dev_pct}%",
                detail=f"mismatches={symbol_mapping_mismatches[:5]}",
            )
        )

        # DQ-06: Per-symbol & aggregate funding unit check (P0-2.5 + Red-Team #9)
        sorted_aprs = sorted(abs_aprs)
        med_apr = statistics.median(sorted_aprs) if sorted_aprs else 0.0
        p95_apr = sorted_aprs[int(0.95 * (len(sorted_aprs) - 1))] if sorted_aprs else 0.0
        dq6_ok = (
            interval_mismatches == 0
            and len(per_sym_unit_anomalies) == 0
            and med_apr <= 150.0
            and p95_apr <= 1500.0
            and max_single_period_rate <= 0.05
        )
        checks.append(
            DataQualityCheck(
                gate_id="DQ-06",
                name="Funding Unit & Settlement Cadence Distribution",
                passed=dq6_ok,
                severity="BLOCKING",
                value=f"mismatches={interval_mismatches}, sym_anom={len(per_sym_unit_anomalies)}, med={med_apr:.1f}%, p95={p95_apr:.1f}%, max_bar={max_single_period_rate:.4f}",
                threshold="mismatches=0 & sym_anom=0 & med<=150% & p95<=1500% & max_bar<=0.05",
                detail=f"anomalies={per_sym_unit_anomalies[:4]}",
            )
        )

        basis_cov_pct = round((dual_price_hits / dual_price_total * 100.0), 2) if dual_price_total > 0 else 0.0
        dq7_ok = basis_cov_pct >= 90.0
        checks.append(
            DataQualityCheck(
                gate_id="DQ-07",
                name="Dual-Venue Price & Basis Completeness",
                passed=dq7_ok,
                severity="ADVISORY",
                value=f"basis_cov={basis_cov_pct:.2f}%",
                threshold=">=90.0% for FULL (else FUNDING_ONLY)",
                detail=f"aligned_dual_bars={dual_price_hits}/{dual_price_total}",
            )
        )

        age_hours = (now_ms - latest_ts_seen) / float(HOUR_MS) if latest_ts_seen > 0 else 9999.0
        fresh_ok = True if require_fresh_hours is None else (age_hours <= require_fresh_hours)
        dq8_ok = max_frozen_hours <= 6 and max_flat_ohlc_hours <= 12 and fresh_ok
        fresh_thresh = "archive" if require_fresh_hours is None else f"{require_fresh_hours}h"
        checks.append(
            DataQualityCheck(
                gate_id="DQ-08",
                name="Anti-Frozen OHLCV Feed & Optional Live Freshness",
                passed=dq8_ok,
                severity="BLOCKING",
                value=f"frozen_0vol={max_frozen_hours}h, flat_ohlc={max_flat_ohlc_hours}h, age_h={age_hours:.1f}",
                threshold=f"frozen<=6h & flat<=12h & age<={fresh_thresh}",
                detail="Runs on OHLCV flatlines only (never penalizes clamped funding baselines)",
            )
        )

        blocking_ok = all(c.passed for c in checks if c.severity == "BLOCKING")
        if not blocking_ok:
            tier = "DATA_REJECTED"
        elif dq7_ok:
            tier = "DATA_QUALIFIED_FULL"
        else:
            tier = "DATA_QUALIFIED_FUNDING_ONLY"

        vpair = f"{venue_a}<->{venue_b}"
        dname = dataset_name or f"funding_pair:{vpair}"
        cert_id = f"DQ-{sha256_obj({'d': dname, 'sha': ds_sha, 't': now_ms})[:10]}"
        report = DataQualityReport(
            cert_id=cert_id,
            created_ts_ms=now_ms,
            dataset_name=dname,
            venue_pair=vpair,
            dataset_sha256=ds_sha,
            provenance=prov,
            row_count=row_count,
            basis_coverage_pct=basis_cov_pct,
            tier=tier,
            passed_count=sum(1 for c in checks if c.passed),
            total_count=len(checks),
            checks=checks,
            symbol_scores=symbol_scores,
        )
        if persist:
            self.store.save_dq_certificate(report)
        return report

    def audit_bbo_ticks(
        self,
        venue_a: str,
        venue_b: str,
        symbols: list[str],
        start_ms: int | None = None,
        end_ms: int | None = None,
        dataset_name: str | None = None,
        persist: bool = True,
        max_stale_sec: float | None = 3600.0,
        sync_tolerance_ms: int = 15_000,
    ) -> tuple[DataQualityReport, dict[str, Any]]:
        """Audit live collected BBO ticks and compute empirical spread, basis, and mapping stats."""
        if self.store is None:
            raise ValueError("MarketDataStore is required for audit_bbo_ticks")

        now_ms = utc_now_ms()
        h = hashlib.sha256()
        provs: set[str] = set()
        total_rows = 0
        bad_ts = 0
        backwards_ts = 0
        crossed_or_bad = 0
        latencies: list[float] = []
        spreads_a: list[float] = []
        spreads_b: list[float] = []
        dual_basis_bps: list[float] = []
        px_ratio_mismatches: list[str] = []
        aligned_pairs = 0
        total_a_ticks = 0
        max_gap_sec = 0.0
        latest_ts = 0
        per_symbol_stats: dict[str, dict[str, float]] = {}

        for sym in symbols:
            # Load in true ingestion order (ORDER BY id ASC) so DQ-03 catches timestamp inversions (Red-Team #9)
            ta = self.store.load_bbo_ticks(venue_a, sym, start_ms, end_ms, order_by_id=True)
            tb = self.store.load_bbo_ticks(venue_b, sym, start_ms, end_ms, order_by_id=True)
            for vname, ticks, sp_list in ((venue_a, ta, spreads_a), (venue_b, tb, spreads_b)):
                prev = -1
                for t in ticks:
                    total_rows += 1
                    provs.add(t.provenance)
                    h.update(f"{t.venue}|{t.symbol}|{t.ts_ms}|{t.bid:.8f}|{t.ask:.8f}\n".encode())
                    if t.ts_ms > latest_ts:
                        latest_ts = t.ts_ms
                    if t.ts_ms < MIN_VALID_EPOCH_MS or t.ts_ms > now_ms + 300_000:
                        bad_ts += 1
                    # Non-decreasing allows same-ms burst ticks cleanly (P0-2.2)
                    if prev >= 0:
                        if t.ts_ms < prev:
                            backwards_ts += 1
                        else:
                            max_gap_sec = max(max_gap_sec, (t.ts_ms - prev) / 1000.0)
                    prev = t.ts_ms
                    if not (math.isfinite(t.bid) and math.isfinite(t.ask) and t.bid > 0 and t.ask >= t.bid):
                        crossed_or_bad += 1
                        continue
                    latencies.append(t.latency_ms)
                    sp_list.append(t.spread_bps)

            # Use time-ordered valid-quote views for cross-venue sync & stats (Red-Team LOW-R2: prevents NaN in JSON)
            ta_sorted = sorted(
                (t for t in ta if math.isfinite(t.bid) and math.isfinite(t.ask) and t.bid > 0 and t.ask >= t.bid),
                key=lambda x: x.ts_ms,
            )
            tb_sorted = sorted(
                (t for t in tb if math.isfinite(t.bid) and math.isfinite(t.ask) and t.bid > 0 and t.ask >= t.bid),
                key=lambda x: x.ts_ms,
            )
            sym_basis: list[float] = []
            sym_sp_a = [t.spread_bps for t in ta_sorted]
            sym_sp_b = [t.spread_bps for t in tb_sorted]
            j = 0
            for t in ta_sorted:
                total_a_ticks += 1
                while j + 1 < len(tb_sorted) and abs(tb_sorted[j + 1].ts_ms - t.ts_ms) <= abs(tb_sorted[j].ts_ms - t.ts_ms):
                    j += 1
                if j < len(tb_sorted) and abs(tb_sorted[j].ts_ms - t.ts_ms) <= sync_tolerance_ms:
                    aligned_pairs += 1
                    mid_a, mid_b = t.mid, tb_sorted[j].mid
                    if mid_b > 0:
                        b_bps = ((mid_a - mid_b) / mid_b) * 10000.0
                        dual_basis_bps.append(b_bps)
                        sym_basis.append(b_bps)
            if sym_basis and abs(statistics.median(sym_basis)) > 250.0:
                px_ratio_mismatches.append(f"{sym}:{statistics.median(sym_basis):+.0f}bp")
            if sym_sp_a or sym_sp_b:
                sorted_abs = sorted(abs(x) for x in sym_basis)
                p95_b = sorted_abs[int(0.95 * (len(sorted_abs) - 1))] if sorted_abs else 0.0
                per_symbol_stats[sym] = {
                    f"{venue_a}_ticks": len(ta_sorted),
                    f"{venue_b}_ticks": len(tb_sorted),
                    f"{venue_a}_med_spread_bps": round(statistics.median(sym_sp_a), 3) if sym_sp_a else 0.0,
                    f"{venue_b}_med_spread_bps": round(statistics.median(sym_sp_b), 3) if sym_sp_b else 0.0,
                    "med_basis_bps": round(statistics.median(sym_basis), 3) if sym_basis else 0.0,
                    "p95_abs_basis_bps": round(p95_b, 3),
                }

        ds_sha = h.hexdigest()
        prov = "+".join(sorted(provs)) if provs else "EMPTY_DATASET"
        align_pct = round((aligned_pairs / total_a_ticks * 100.0), 2) if total_a_ticks > 0 else 0.0
        med_lat = round(statistics.median(latencies), 1) if latencies else 0.0
        med_sp_a = round(statistics.median(spreads_a), 3) if spreads_a else 0.0
        med_sp_b = round(statistics.median(spreads_b), 3) if spreads_b else 0.0
        age_sec = (now_ms - latest_ts) / 1000.0 if latest_ts > 0 else 999999.0
        fresh_ok = True if max_stale_sec is None else (age_sec <= max_stale_sec)
        stale_thresh = "archive" if max_stale_sec is None else f"{max_stale_sec}s"

        checks = [
            DataQualityCheck("DQ-01", "BBO Provenance & SHA-256", total_rows >= 20 and provs.issubset(ALLOWED_REAL_PROVENANCES) and bool(provs), "BLOCKING", f"{prov} ({total_rows} ticks)", "REAL & ticks>=20", f"sha256={ds_sha[:12]}"),
            DataQualityCheck("DQ-02", "BBO Capture Continuity", max_gap_sec <= 300.0 and total_rows >= 20, "BLOCKING", f"max_gap={max_gap_sec:.1f}s", "max_gap<=300s", f"symbols={len(symbols)}"),
            DataQualityCheck("DQ-03", "BBO Non-Decreasing Timestamp Hygiene", bad_ts == 0 and backwards_ts == 0, "BLOCKING", f"bad_ts={bad_ts}, backwards={backwards_ts}", "0 & 0", "Non-decreasing UTC ms in ingestion order"),
            DataQualityCheck("DQ-04", "Dual-Venue BBO Round Sync", align_pct >= 90.0, "BLOCKING", f"sync={align_pct:.1f}%", ">=90.0%", f"aligned={aligned_pairs}/{total_a_ticks}"),
            DataQualityCheck("DQ-05", "Zero Crossed Book & Symbol Price Ratio", crossed_or_bad == 0 and med_sp_a <= 50.0 and med_sp_b <= 50.0 and len(px_ratio_mismatches) == 0, "BLOCKING", f"crossed={crossed_or_bad}, spA={med_sp_a}bp, spB={med_sp_b}bp, mismatch={len(px_ratio_mismatches)}", "crossed=0 & sp<=50bp & ratio<=2.5%", f"mismatches={px_ratio_mismatches[:4]}"),
            DataQualityCheck("DQ-06", "Public API Fetch Latency", med_lat <= 3000.0, "BLOCKING", f"med_lat={med_lat:.1f}ms", "<=3000ms", "REST/Book snapshot latency"),
            DataQualityCheck("DQ-07", "Live Dual-Venue Basis Coverage", align_pct >= 90.0 and len(dual_basis_bps) >= 10, "ADVISORY", f"cov={align_pct:.1f}%, n_basis={len(dual_basis_bps)}", ">=90% & n>=10", f"med_basis={statistics.median(dual_basis_bps):+.2f}bp" if dual_basis_bps else "no basis"),
            DataQualityCheck("DQ-08", "Live Freshness SLA", fresh_ok, "BLOCKING", f"age={age_sec:.1f}s", f"<={stale_thresh}", "Confirms active collector"),
        ]
        blocking_ok = all(c.passed for c in checks if c.severity == "BLOCKING")
        tier = "DATA_REJECTED" if not blocking_ok else ("DATA_QUALIFIED_FULL" if checks[6].passed else "DATA_QUALIFIED_FUNDING_ONLY")
        vpair = f"bbo:{venue_a}<->{venue_b}"
        dname = dataset_name or vpair
        cert_id = f"DQ-{sha256_obj({'d': dname, 'sha': ds_sha, 't': now_ms})[:10]}"
        report = DataQualityReport(
            cert_id=cert_id,
            created_ts_ms=now_ms,
            dataset_name=dname,
            venue_pair=vpair,
            dataset_sha256=ds_sha,
            provenance=prov,
            row_count=total_rows,
            basis_coverage_pct=align_pct,
            tier=tier,
            passed_count=sum(1 for c in checks if c.passed),
            total_count=len(checks),
            checks=checks,
            symbol_scores={s: "OK" for s in per_symbol_stats},
        )
        if persist:
            self.store.save_dq_certificate(report)

        summary = {
            "venue_a": venue_a,
            "venue_b": venue_b,
            "total_ticks": total_rows,
            "aligned_pairs": aligned_pairs,
            "venue_a_med_spread_bps": med_sp_a,
            "venue_b_med_spread_bps": med_sp_b,
            "combined_half_spread_cost_bps": round((med_sp_a + med_sp_b) * 0.5, 3),
            "med_basis_bps": round(statistics.median(dual_basis_bps), 3) if dual_basis_bps else 0.0,
            "std_basis_bps": round(statistics.pstdev(dual_basis_bps), 3) if len(dual_basis_bps) >= 2 else 0.0,
            "per_symbol": per_symbol_stats,
        }
        return report, summary

    def audit_csv_sample(
        self,
        csv_path: str | Path,
        dataset_name: str,
        ts_col: str = "ts_ms",
        price_cols: list[str] | None = None,
        venue_pair: str = "server_csv",
        persist: bool = True,
        provenance: str = "UNVERIFIED_LOCAL_CSV",
    ) -> DataQualityReport:
        """Audit a CSV dataset; defaults to UNVERIFIED_LOCAL_CSV (rejected by DQ-01 unless certified)."""
        p = Path(csv_path)
        now_ms = utc_now_ms()
        h = hashlib.sha256()
        row_count = 0
        parsed_ts_count = 0
        bad_ts = 0
        non_mono = 0
        bad_px = 0
        crossed = 0
        prev_ts = -1
        pcols = price_cols or []

        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row_count += 1
                h.update(str(sorted((str(k), str(v)) for k, v in row.items())).encode("utf-8"))
                raw_ts = (row.get(ts_col) or "").strip()
                ts_val: int | None = None
                if not raw_ts:
                    bad_ts += 1
                else:
                    try:
                        ts_val = int(float(raw_ts))
                        if ts_val < 10_000_000_000:
                            ts_val *= 1000
                    except ValueError:
                        from datetime import datetime
                        try:
                            dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                            ts_val = int(dt.timestamp() * 1000)
                        except ValueError:
                            ts_val = None
                            bad_ts += 1
                if ts_val is not None:
                    parsed_ts_count += 1
                    if ts_val < MIN_VALID_EPOCH_MS or ts_val > now_ms + 300_000:
                        bad_ts += 1
                    if prev_ts >= 0 and ts_val < prev_ts:
                        non_mono += 1
                    prev_ts = ts_val
                for c in pcols:
                    raw_v = row.get(c, "")
                    if raw_v == "" or raw_v is None:
                        continue
                    try:
                        v = float(raw_v)
                        if not math.isfinite(v) or v <= 0:
                            bad_px += 1
                    except ValueError:
                        bad_px += 1
                if "bid" in row and "ask" in row and row["bid"] and row["ask"]:
                    try:
                        bid_f, ask_f = float(row["bid"]), float(row["ask"])
                        if not (math.isfinite(bid_f) and math.isfinite(ask_f) and bid_f > 0 and ask_f > 0):
                            bad_px += 1
                        elif bid_f > ask_f:
                            crossed += 1
                    except ValueError:
                        bad_px += 1

        ds_sha = h.hexdigest()
        # Red-Team MEDIUM-R1: DQ-01 MUST reject UNVERIFIED_LOCAL_CSV (aligned with audit_funding_pair & audit_bbo_ticks)
        checks = [
            DataQualityCheck(
                "DQ-01",
                "CSV Provenance & Sample Floor",
                row_count >= 10 and provenance in ALLOWED_REAL_PROVENANCES,
                "BLOCKING",
                f"{provenance} ({row_count} rows)",
                "ALLOWED_REAL_PROVENANCES & rows>=10",
                f"file={p.name} sha={ds_sha[:12]}",
            ),
            DataQualityCheck(
                "DQ-03",
                "CSV Non-Decreasing Timestamp Hygiene",
                bad_ts == 0 and non_mono == 0 and parsed_ts_count == row_count and row_count > 0,
                "BLOCKING",
                f"bad_ts={bad_ts}, inversions={non_mono}, parsed={parsed_ts_count}/{row_count}",
                "0 & 0 & all_rows_parsed",
                f"ts_col={ts_col}",
            ),
            DataQualityCheck(
                "DQ-05",
                "CSV Price & Crossed Book Sanity",
                bad_px == 0 and crossed == 0,
                "BLOCKING",
                f"bad_px={bad_px}, crossed={crossed}",
                "0 & 0",
                f"cols={pcols}",
            ),
        ]
        tier = "DATA_QUALIFIED_FULL" if all(c.passed for c in checks) else "DATA_REJECTED"
        cert_id = f"DQ-{sha256_obj({'d': dataset_name, 'sha': ds_sha, 't': now_ms})[:10]}"
        report = DataQualityReport(
            cert_id=cert_id,
            created_ts_ms=now_ms,
            dataset_name=dataset_name,
            venue_pair=venue_pair,
            dataset_sha256=ds_sha,
            provenance=provenance,
            row_count=row_count,
            basis_coverage_pct=100.0 if tier == "DATA_QUALIFIED_FULL" else 0.0,
            tier=tier,
            passed_count=sum(1 for c in checks if c.passed),
            total_count=len(checks),
            checks=checks,
        )
        if persist and self.store is not None:
            self.store.save_dq_certificate(report)
        return report
