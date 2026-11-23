"""Tests for collector helpers, dataset fingerprinting, and STD-34/L-034 hardened Web Console API."""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from alphaloop.core.models import FundingRateBar
from alphaloop.core.provenance import PROVENANCE_REAL
from alphaloop.data.collectors import _infer_interval_hours
from alphaloop.data.store import MarketDataStore
from alphaloop.web.server import create_server


class TestCollectorsAndWebSecurity(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.srv, self.ctx = create_server(self.root, host="127.0.0.1", port=0)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()
        self.ctx.close()
        self.tmp.cleanup()

    def test_interval_inference(self) -> None:
        h = 3600 * 1000
        self.assertEqual(_infer_interval_hours([0, 4 * h, 8 * h, 12 * h]), 4.0)
        self.assertEqual(_infer_interval_hours([0, 8 * h, 16 * h, 24 * h]), 8.0)
        self.assertEqual(_infer_interval_hours([0, h, 2 * h]), 1.0)

    def test_dataset_fingerprint_is_deterministic_and_tamper_evident(self) -> None:
        store = MarketDataStore(self.root / "fp.sqlite")
        store.upsert_funding([FundingRateBar("hyperliquid", "BTC", 3600000, 0.0001, 1.0, PROVENANCE_REAL)])
        sha1, prov1, cnt1 = store.fingerprint_slice(["hyperliquid"], ["BTC"])
        self.assertEqual(prov1, PROVENANCE_REAL)
        self.assertEqual(cnt1, 1)
        # Mutate funding rate by 1e-6 -> SHA256 MUST change!
        store.upsert_funding([FundingRateBar("hyperliquid", "BTC", 3600000, 0.000101, 1.0, PROVENANCE_REAL)])
        sha2, _, _ = store.fingerprint_slice(["hyperliquid"], ["BTC"])
        self.assertNotEqual(sha1, sha2)
        store.close()

    def test_web_health_and_api_endpoints(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        for ep in ("/", "/api/health", "/api/strategies", "/api/trials", "/api/cycles", "/api/models", "/api/market/summary"):
            conn.request("GET", ep)
            resp = conn.getresponse()
            body = resp.read()
            self.assertEqual(resp.status, 200, f"Endpoint {ep} failed")
            self.assertEqual(int(resp.getheader("Content-Length")), len(body))
            self.assertEqual(resp.getheader("Cache-Control"), "no-cache")
        conn.close()

    def test_web_rejects_negative_and_oversized_content_length_l034(self) -> None:
        """L-034 / RB3-08 regression: negative or >1MB Content-Length must immediately return 400/413."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", "/api/agent/run-cycle")
        conn.putheader("Content-Length", "-1")
        conn.endheaders()
        resp = conn.getresponse()
        resp.read()
        self.assertEqual(resp.status, 400)
        conn.close()

        conn2 = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn2.putrequest("POST", "/api/agent/run-cycle")
        conn2.putheader("Content-Length", str(2 * 1024 * 1024))
        conn2.endheaders()
        resp2 = conn2.getresponse()
        resp2.read()
        self.assertEqual(resp2.status, 413)
        conn2.close()


if __name__ == "__main__":
    unittest.main()
