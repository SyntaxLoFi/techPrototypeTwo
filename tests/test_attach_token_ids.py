import os, sys, unittest
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
from scripts.data_collection.polymarket_fetcher_v2 import PolymarketFetcher

class DummyClient:
    def iter_clob_markets(self):
        # Market slugs lowercased as in CLOB
        return iter([
            { "market_slug": "btc-above-50k", "tokens": [
                {"token_id":"AAA", "outcome":"YES"}, {"token_id":"BBB","outcome":"NO"}
            ]},
            { "slug": "eth-above-4k", "tokens": [
                {"token_id":"CCC", "outcome":"UP"}, {"token_id":"DDD","outcome":"DOWN"}
            ]},
        ])

class TestAttachTokenIds(unittest.TestCase):
    def test_attach_by_slug_case_insensitive(self):
        fetcher = PolymarketFetcher(debug_dir="debug_runs")
        markets = [
            {"marketSlug": "BTC-Above-50k"},
            {"marketSlug": "eth-above-4k"},
            {"marketSlug": "unknown"},
        ]
        out = fetcher._attach_token_ids_via_clob(DummyClient(), markets)
        m0 = out[0].get("tokenIds"); m1 = out[1].get("tokenIds"); m2 = out[2].get("tokenIds")
        self.assertEqual(m0["YES"], "AAA"); self.assertEqual(m0["NO"], "BBB")
        self.assertEqual(m1["UP"], "CCC"); self.assertEqual(m1["DOWN"], "DDD")
        self.assertIsNone(m2)