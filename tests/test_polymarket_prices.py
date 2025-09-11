import os
import json
import shutil
from unittest import TestCase, mock

# Ensure project dir in path
import sys
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.data_collection.polymarket_client import PolymarketClient
import main_scanner

class TestPolymarketClient(TestCase):
    def setUp(self):
        self.client = PolymarketClient()

    @mock.patch("requests.Session.post")
    def test_get_multiple_prices(self, mpost):
        class Resp:
            status_code = 200
            def json(self):
                return {
                    "AAA": {"BUY": "0.45", "SELL": "0.46"},
                    "BBB": {"BUY": "0.10", "SELL": "0.11"},
                }
        mpost.return_value = Resp()
        out = self.client.get_multiple_prices(["AAA", "BBB"])
        self.assertEqual(out["AAA"]["BUY"], "0.45")
        self.assertIn("SELL", out["BBB"])
        # Ensure we POSTed to /prices with the expected shape
        args, kwargs = mpost.call_args
        self.assertTrue(args[0].endswith("/prices"))
        self.assertIsInstance(kwargs.get("json"), list)
        sides = [d.get("side") for d in kwargs["json"] if d.get("token_id") == "AAA"]
        self.assertIn("BUY", sides)
        self.assertIn("SELL", sides)

    @mock.patch("requests.Session.get")
    def test_get_all_prices(self, mget):
        class Resp:
            status_code = 200
            def json(self):
                # lower-case keys should get normalized to BUY/SELL
                return {"CCC": {"buy": "0.20", "sell": "0.21"}}
        mget.return_value = Resp()
        out = self.client.get_all_prices()
        self.assertEqual(out["CCC"]["BUY"], "0.20")
        self.assertEqual(out["CCC"]["SELL"], "0.21")


class TestDumpDebug(TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure debug files land under project directory
        os.chdir(PROJECT_DIR)

    def setUp(self):
        # Isolate & enable debug recorder
        os.environ["DEBUG"] = "1"
        os.environ["APP_RUN_ID"] = "UNITTEST"
        self.debug_root = os.path.join(PROJECT_DIR, "debug_runs", "UNITTEST")
        shutil.rmtree(self.debug_root, ignore_errors=True)

    def tearDown(self):
        os.environ.pop("DEBUG", None)
        os.environ.pop("APP_RUN_ID", None)
        shutil.rmtree(self.debug_root, ignore_errors=True)

    def test_dump_polymarket_clob_debug(self):
        markets = [
            {"tokenIds": {"yes": "AAA", "no": "BBB"}},
            {"tokenIds": {"UP": "CCC", "DOWN": "DDD"}},
        ]
        # Patch client used by the helper
        with mock.patch.object(main_scanner, "PolymarketClient") as MockClient:
            inst = MockClient.return_value
            inst.get_books.return_value = {
                "AAA": {"bids":[{"price":"0.45","size":"100"}], "asks":[{"price":"0.46","size":"200"}]},
                "BBB": {"bids":[{"price":"0.55","size":"50"}],  "asks":[{"price":"0.56","size":"60"}]},
                "CCC": {"bids":[], "asks":[]},
                "DDD": {"bids":[{"price":"0.12","size":"30"}],  "asks":[{"price":"0.13","size":"40"}]},
            }
            inst.get_multiple_prices.return_value = {
                "AAA": {"BUY":"0.46","SELL":"0.45"},
                "BBB": {"BUY":"0.56","SELL":"0.55"},
                "CCC": {"BUY":"0.14","SELL":"0.13"},
                "DDD": {"BUY":"0.13","SELL":"0.12"},
            }
            # Use an explicit recorder so we always write in test
            from utils.debug_recorder import RawDataRecorder
            rec = RawDataRecorder(enabled=True)
            main_scanner._dump_polymarket_clob_debug(markets, recorder=rec, logger=None)

        # Verify files created anywhere under debug_runs
        found_token_ids = []
        for root, dirs, files in os.walk(os.path.join(PROJECT_DIR, "debug_runs")):
            if "token_ids.json" in files and os.path.basename(root) == "polymarket":
                found_token_ids.append(os.path.join(root, "token_ids.json"))
        self.assertTrue(found_token_ids, "expected at least one polymarket/token_ids.json to be created")
        # Validate contents
        price_paths = []
        for root, dirs, files in os.walk(os.path.join(PROJECT_DIR, "debug_runs")):
            if "prices.json" in files and os.path.basename(root) == "polymarket":
                price_paths.append(os.path.join(root, "prices.json"))
        self.assertTrue(price_paths, "expected polymarket/prices.json to be created")
        data = json.load(open(price_paths[0]))
        self.assertIn("AAA", data)
        self.assertIn("BUY", data["AAA"])