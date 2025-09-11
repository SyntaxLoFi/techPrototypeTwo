import os
import json
import shutil
import unittest
from unittest import mock

import sys
BASE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.data_collection.polymarket_client import PolymarketClient
from scripts.data_collection.polymarket_fetcher_v2 import PolymarketFetcher

class TestPolymarketClientBooks(unittest.TestCase):
    @mock.patch("requests.post")
    def test_get_books_maps_asset_id(self, mpost):
        class R:
            status_code = 200
            def json(self_inner):
                return [
                    {"asset_id": "AAA", "bids":[{"price":"0.45","size":"10"}], "asks":[{"price":"0.55","size":"5"}]},
                    {"asset_id": "BBB", "bids":[{"price":"0.15","size":"7"}],  "asks":[{"price":"0.85","size":"3"}]},
                ]
        mpost.return_value = R()
        client = PolymarketClient()
        out = client.get_books(["AAA","BBB"])
        self.assertIn("AAA", out)
        self.assertIn("BBB", out)
        self.assertIn("bids", out["AAA"])
        args, kwargs = mpost.call_args
        self.assertTrue(args[0].endswith("/books"))
        self.assertIsInstance(kwargs.get("json"), list)
        self.assertTrue(all("token_id" in item for item in kwargs["json"]))

    @mock.patch("requests.post")
    def test_get_prices_by_request(self, mpost):
        class R:
            status_code = 200
            def json(self_inner):
                return {"AAA":{"BUY":"0.55","SELL":"0.45"}, "BBB":{"BUY":"0.85","SELL":"0.15"}}
        mpost.return_value = R()
        client = PolymarketClient()
        out = client.get_prices_by_request(["AAA","BBB"])
        self.assertEqual(out["AAA"]["BUY"], "0.55")
        args, kwargs = mpost.call_args
        body = kwargs["json"]
        self.assertEqual(sum(1 for x in body if x["token_id"]=="AAA"), 2)
        sides = {x["side"] for x in body if x["token_id"]=="AAA"}
        self.assertEqual(sides, {"BUY","SELL"})

class DummyClient:
    def __init__(self, books, prices):
        self._books = books
        self._prices = prices
    def get_books(self, token_ids):
        return {tid: self._books[tid] for tid in token_ids if tid in self._books}
    def get_prices_by_request(self, token_ids, sides=None):
        return {tid: self._prices[tid] for tid in token_ids if tid in self._prices}

class TestFetcherEnrichmentDebug(unittest.TestCase):
    def setUp(self):
        self.run_id = "UNITTEST_PM"
        os.environ["DEBUG"] = "1"
        os.environ["APP_RUN_ID"] = self.run_id
        os.chdir(ROOT)
        self.debug_root = os.path.join(ROOT, "debug_runs", self.run_id)
        shutil.rmtree(self.debug_root, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.debug_root, ignore_errors=True)
        os.environ.pop("DEBUG", None)
        os.environ.pop("APP_RUN_ID", None)

    def test_books_best_and_prices_written(self):
        markets = [
            {"tokenIds": {"YES":"AAA", "NO":"BBB"}, "question":"Q1"},
        ]
        books = {
            "AAA": {"bids":[{"price":"0.49","size":"10"},{"price":"0.48","size":"20"}],
                    "asks":[{"price":"0.51","size":"5"},{"price":"0.52","size":"6"}]},
            "BBB": {"bids":[{"price":"0.53","size":"4"}],
                    "asks":[{"price":"0.47","size":"11"},{"price":"0.48","size":"12"}]},
        }
        prices = {"AAA":{"BUY":"0.51", "SELL":"0.49"}, "BBB":{"BUY":"0.47", "SELL":"0.53"}}
        fetcher = PolymarketFetcher(debug_dir="debug_runs")
        client = DummyClient(books, prices)
        _ = fetcher._enrich_with_orderbooks(client, markets)
        poly_dir = os.path.join(self.debug_root, "polymarket")
        self.assertTrue(os.path.exists(os.path.join(poly_dir, "books.json")))
        self.assertTrue(os.path.exists(os.path.join(poly_dir, "books_best.json")))
        self.assertTrue(os.path.exists(os.path.join(poly_dir, "prices.json")))
        best = json.load(open(os.path.join(poly_dir, "books_best.json")))
        self.assertEqual(best["AAA"]["BUY"], "0.51")
        self.assertEqual(best["AAA"]["SELL"], "0.49")
        prices_json = json.load(open(os.path.join(poly_dir, "prices.json")))
        self.assertIn("AAA", prices_json)
        self.assertIn("SELL", prices_json["AAA"])