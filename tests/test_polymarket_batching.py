import os, sys
from unittest import TestCase, mock
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
from scripts.data_collection.polymarket_client import PolymarketClient

class TestPolymarketBatching(TestCase):
    def test_prices_batched_on_large_payload(self):
        client = PolymarketClient()
        # Mock session.post to simulate 400 when payload too large and 200 otherwise
        def fake_post(url, json=None, timeout=None):
            class Resp:
                def __init__(self, status_code, data):
                    self.status_code = status_code
                    self._data = data
                def json(self):
                    return self._data
            # Each token produces two entries in payload (BUY/SELL)
            n = len(json or [])
            if n > 2*100:  # MAX_CLOB_PRICE_BATCH * sides
                return Resp(400, {})
            # Return a mapping for the tokens present in payload
            out = {}
            toks = {d.get("token_id") for d in (json or [])}
            for t in toks:
                if not t: continue
                out[str(t)] = {"BUY": "0.50", "SELL":"0.51"}
            return Resp(200, out)

        with mock.patch.object(client, "session") as msess:
            msess.post.side_effect = fake_post
            # Build 250 tokens → requires batching
            tokens = [f"T{i:03d}" for i in range(250)]
            res = client.get_multiple_prices(tokens)
            # Should have coverage for all tokens (or at least > 0)
            self.assertGreater(len(res), 0)
            self.assertIn("T000", res)
            self.assertIn("BUY", res["T000"])