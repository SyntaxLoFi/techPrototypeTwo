# scripts/data_collection/test_perps_derive_fields.py
import math
import json
import types

import importlib

def _fake_resp(payload):
    class R:
        status_code = 200
        def json(self): return payload
        @property
        def text(self): return json.dumps(payload)
    return R()

def test_fetch_sets_hourly_cadence_and_index_premium(monkeypatch):
    # Lazy import so monkeypatch lands before the module under test does network
    import scripts.data_collection.perps_data_collector as pdc_mod

    # Patch whichever client is used (requests or utils.http_client)
    tried = {"requests": False, "rest": False}

    def fake_post(url, json=None, timeout=None):
        if url.endswith("/public/get_ticker"):
            return _fake_resp({
                "result": {
                    "instrument_name": "ETH-PERP",
                    "mark_price": "1050",
                    "index_price": "1000",
                    "best_bid_price": "1049",
                    "best_ask_price": "1051",
                    "maker_fee_rate": "0.0001",
                    "taker_fee_rate": "0.0003",
                    "base_fee": "0.10",
                    "perp_details": {"funding_rate": "0.001"}  # hourly
                }
            })
        if url.endswith("/public/get_instrument"):
            return _fake_resp({"result": {"instrument_name": "ETH-PERP", "contract_size": "1"}})
        if url.endswith("/public/get_funding_rate_history"):
            return _fake_resp({"result": {"series": [], "percentiles": {}}})
        return _fake_resp({"result": {}})

    # Try to patch utils.http_client if present
    try:
        import utils.http_client as rest
        monkeypatch.setattr(rest, "post", fake_post)
        tried["rest"] = True
    except Exception:
        pass
    # Also patch requests as fallback
    try:
        import requests
        monkeypatch.setattr(requests, "post", fake_post)
        tried["requests"] = True
    except Exception:
        pass

    PerpsDataCollector = pdc_mod.PerpsDataCollector
    c = PerpsDataCollector()
    data = c.fetch_perp_info("ETH", fetch_funding_history=False)

    assert data is not None
    # funding cadence should be hourly on Derive
    assert math.isclose(float(data["funding_interval_hours"]), 1.0), f"wrong cadence: {data['funding_interval_hours']}"
    # premium index must be (mark - index) / index
    expected = (float(data["mark_price"]) - float(data["index_price"])) / float(data["index_price"])
    assert math.isclose(float(data["premium_index"]), expected, rel_tol=1e-12)