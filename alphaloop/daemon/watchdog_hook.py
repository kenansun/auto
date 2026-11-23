"""Non-Blocking Business-Payload Watchdog Hook & Deduplicated Alert Notifier (Stage 1 S1-01, Red-Team Hardened).

STD-32 / STD-34 Pillar 4 & Pillar 5 Compliance:
  - Emits heartbeats ONLY for real, non-stale validated business payloads (`ts_event_ms` guard).
    Wall-clock fake heartbeats are rejected fail-closed (`STALE_EVENT_TS` / `FUTURE_EVENT_TS`).
  - Uses canonical `/api/v1/heartbeat` route and `"metrics"` payload key compatible with
    the node watchdog sentinel.
  - Bounded ring buffer (`maxlen=256`) + guarded daemon worker thread with strict HTTP timeout
    (`0.5s`) and `try/except` protection so transport exceptions never kill the drain worker.
  - `AlertNotifier` enforces `(level, code)` cooldown ONLY after successful sink delivery,
    so transient sink failures never suppress P0 alerts for 30 minutes.
"""

from __future__ import annotations

import collections
import http.client
import json
import logging
import math
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from alphaloop.core.provenance import utc_now_ms

logger = logging.getLogger(__name__)

ALLOWED_ALERT_LEVELS = frozenset({"P0", "P1", "P2", "REPORT"})


def _validate_http_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid HTTP(S) watchdog URL: {url!r}")
    return cleaned


class WatchdogHook:
    """Async non-blocking dead-man's-switch heartbeat emitter gated on real business timestamps."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        node_name: str | None = None,
        *,
        max_stale_ms: int = 300_000,
        max_future_ms: int = 30_000,
        timeout_sec: float = 0.5,
        queue_maxsize: int = 256,
        async_mode: bool = True,
        clock_ms_fn: Callable[[], int] | None = None,
        transport: Callable[[str, dict[str, Any], float], tuple[bool, str]] | None = None,
    ) -> None:
        raw_url = endpoint_url if endpoint_url is not None else os.environ.get("ALPHALOOP_WATCHDOG_URL", "")
        self.endpoint_url = _validate_http_url(raw_url)
        self.node_name = (
            node_name
            or os.environ.get("ALPHALOOP_NODE_NAME")
            or socket.gethostname()
        )
        self.max_stale_ms = max(1_000, int(max_stale_ms))
        self.max_future_ms = max(1_000, int(max_future_ms))
        self.timeout_sec = max(0.05, min(float(timeout_sec), 5.0))
        self.queue_maxsize = max(8, min(int(queue_maxsize), 2048))
        self.async_mode = bool(async_mode)
        self._clock_ms = clock_ms_fn or utc_now_ms
        self._transport = transport or self._default_http_post

        self._lock = threading.Lock()
        self._queue: collections.deque[dict[str, Any]] = collections.deque(maxlen=self.queue_maxsize)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._accepted = 0
        self._rejected = 0
        self._dropped = 0
        self._sent_ok = 0
        self._sent_err = 0
        self._last_event_ts_ms: dict[str, int] = {}
        self._last_reject_reason = ""
        self._worker: threading.Thread | None = None

        if self.async_mode:
            self._worker = threading.Thread(
                target=self._drain_loop,
                name="alphaloop-watchdog-hook",
                daemon=True,
            )
            self._worker.start()

    @staticmethod
    def resolve_heartbeat_url(url: str) -> str:
        cleaned = url.rstrip("/")
        if not cleaned:
            return ""
        if cleaned.endswith("/api/v1/heartbeat") or cleaned.endswith("/heartbeat"):
            return cleaned
        return f"{cleaned}/api/v1/heartbeat"

    @classmethod
    def _default_http_post(cls, url: str, payload: dict[str, Any], timeout_sec: float) -> tuple[bool, str]:
        if not url:
            return True, "DISABLED_NO_ENDPOINT"
        target = cls.resolve_heartbeat_url(url)
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        req = urllib.request.Request(
            target,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "AlphaLoop-WatchdogHook/2.2"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                code = int(getattr(resp, "status", 200))
                return (200 <= code < 300), f"HTTP_{code}"
        except (urllib.error.URLError, urllib.error.HTTPError, http.client.HTTPException, OSError, TimeoutError) as exc:
            logger.warning("Watchdog heartbeat POST failed (%s): %s", target, exc)
            return False, f"ERR:{type(exc).__name__}"

    def _validate_payload(
        self,
        feed_id: str,
        ts_event_ms: int,
        seq: int,
        now_ms: int,
    ) -> tuple[bool, str]:
        fid = str(feed_id or "").strip()
        if not fid or len(fid) > 128:
            return False, "INVALID_FEED_ID"
        if isinstance(ts_event_ms, bool) or not isinstance(ts_event_ms, int) or ts_event_ms <= 0:
            return False, "INVALID_EVENT_TS"
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            return False, "INVALID_SEQ"
        age_ms = now_ms - ts_event_ms
        if age_ms > self.max_stale_ms:
            return False, f"STALE_EVENT_TS(age_ms={age_ms}>{self.max_stale_ms})"
        if age_ms < -self.max_future_ms:
            return False, f"FUTURE_EVENT_TS(lead_ms={-age_ms}>{self.max_future_ms})"
        return True, "OK"

    def heartbeat(
        self,
        feed_id: str,
        ts_event_ms: int,
        seq: int,
        meta: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Validate business payload timestamp and enqueue (or send sync) a watchdog heartbeat."""
        now_ms = int(self._clock_ms())
        ok, reason = self._validate_payload(feed_id, ts_event_ms, seq, now_ms)
        if not ok:
            with self._lock:
                self._rejected += 1
                self._last_reject_reason = reason
            logger.warning("WatchdogHook rejected heartbeat feed=%r: %s", feed_id, reason)
            return False, reason

        safe_metrics: dict[str, Any] = {}
        if meta:
            for k, v in meta.items():
                if isinstance(v, float) and not math.isfinite(v):
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    safe_metrics[str(k)] = v

        payload: dict[str, Any] = {
            "feed_id": str(feed_id).strip(),
            "node": self.node_name,
            "ts_event_ms": int(ts_event_ms),
            "ts_sent_ms": now_ms,
            "seq": int(seq),
            "metrics": safe_metrics,
        }

        if not self.async_mode:
            try:
                sent_ok, detail = self._transport(self.endpoint_url, payload, self.timeout_sec)
            except Exception as exc:
                logger.warning("Watchdog sync transport exception: %s", exc)
                sent_ok, detail = False, f"TRANSPORT_EXC:{type(exc).__name__}"
            with self._lock:
                self._accepted += 1
                self._last_event_ts_ms[payload["feed_id"]] = int(ts_event_ms)
                if sent_ok:
                    self._sent_ok += 1
                else:
                    self._sent_err += 1
            return sent_ok, detail

        with self._lock:
            self._accepted += 1
            self._last_event_ts_ms[payload["feed_id"]] = int(ts_event_ms)
            if len(self._queue) >= self.queue_maxsize:
                self._dropped += 1
            self._queue.append(payload)
        self._wake.set()
        return True, "QUEUED"

    def _drain_loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    item = self._queue.popleft()
                try:
                    ok, _ = self._transport(self.endpoint_url, item, self.timeout_sec)
                except Exception as exc:
                    logger.warning("Watchdog async transport error (worker kept alive): %s", exc)
                    ok = False
                with self._lock:
                    if ok:
                        self._sent_ok += 1
                    else:
                        self._sent_err += 1
                if self._stop.is_set():
                    break

    def flush(self, timeout_sec: float = 2.0) -> None:
        """Wait briefly for queued heartbeats to drain (mainly for shutdown and tests)."""
        deadline = time.monotonic() + max(0.05, float(timeout_sec))
        while time.monotonic() < deadline:
            with self._lock:
                if not self._queue:
                    return
            self._wake.set()
            time.sleep(0.02)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": bool(self.endpoint_url),
                "node": self.node_name,
                "accepted": self._accepted,
                "rejected": self._rejected,
                "dropped": self._dropped,
                "sent_ok": self._sent_ok,
                "sent_err": self._sent_err,
                "queued": len(self._queue),
                "last_reject_reason": self._last_reject_reason,
                "last_event_ts_ms": dict(self._last_event_ts_ms),
            }

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=1.5)


class AlertNotifier:
    """Tiered P0/P1/P2/REPORT alert sink with strict cooldown committed ONLY on delivery success."""

    def __init__(
        self,
        cooldown_sec: float = 1800.0,
        *,
        clock_ms_fn: Callable[[], int] | None = None,
        sink: Callable[[dict[str, Any]], None] | None = None,
        history_size: int = 100,
    ) -> None:
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self._clock_ms = clock_ms_fn or utc_now_ms
        self._sink = sink
        self._lock = threading.Lock()
        self._last_sent_ms: dict[tuple[str, str], int] = {}
        self._history: collections.deque[dict[str, Any]] = collections.deque(maxlen=max(10, int(history_size)))
        self._emitted_total = 0
        self._suppressed = 0
        self._sink_errors = 0

    def emit(
        self,
        level: str,
        code: str,
        title: str,
        detail: str,
        *,
        source: str = "alphaloop/shadow",
    ) -> tuple[bool, str]:
        lvl = str(level or "P2").upper()
        if lvl not in ALLOWED_ALERT_LEVELS:
            lvl = "P2"
        clean_code = str(code or "GENERIC").strip()[:64]
        now_ms = int(self._clock_ms())
        key = (lvl, clean_code)

        with self._lock:
            prev_ms = self._last_sent_ms.get(key)
            if prev_ms is not None and (now_ms - prev_ms) < int(self.cooldown_sec * 1000.0):
                self._suppressed += 1
                return False, "COOLDOWN_SUPPRESSED"
            record = {
                "ts_ms": now_ms,
                "level": lvl,
                "code": clean_code,
                "source": str(source)[:128],
                "title": str(title)[:200],
                "detail": str(detail)[:1000],
            }

        logger.log(
            logging.ERROR if lvl == "P0" else (logging.WARNING if lvl == "P1" else logging.INFO),
            "[ALERT:%s][%s] %s — %s",
            lvl,
            clean_code,
            record["title"],
            record["detail"],
        )
        if self._sink is not None:
            try:
                self._sink(dict(record))
            except Exception as exc:
                logger.warning("AlertNotifier sink error (cooldown NOT committed so alert retries): %s", exc)
                with self._lock:
                    self._sink_errors += 1
                return False, f"SINK_ERROR:{type(exc).__name__}"

        with self._lock:
            self._last_sent_ms[key] = now_ms
            self._emitted_total += 1
            self._history.append(record)
        return True, "EMITTED"

    def recent_alerts(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._history)
        return list(reversed(items[-max(1, int(limit)):]))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "emitted": self._emitted_total,
                "suppressed": self._suppressed,
                "sink_errors": self._sink_errors,
            }
