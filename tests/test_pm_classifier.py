# -*- coding: utf-8 -*-
import json
import os
import re

from scripts.data_collection.pm_ingest import tag_from_local_markets
from scripts.data_collection.pm_classifier import classify_market


def test_classify_thresholds():
    e = {"id": "e1", "title": "Ethereum above 4500 on Sep 8", "description": ""}
    m = {
        "id": "m1",
        "question": "Ethereum above $4,500 on September 8?",
        "description": "",
        "endDate": "2025-09-08T20:00:00Z",
        "clobTokenIds": {"YES": "123", "NO": "456"},
    }
    t = classify_market(e, m)
    assert t["marketClass"] == "SINGLE_THRESHOLD"
    assert t["relation"] == ">="
    assert t["threshold"] == 4500.0
    assert "variance_swap" in t["strategies"]


def test_classify_range():
    e = {"id": "e2", "title": "Ethereum Price - September 6, 4PM ET", "description": ""}
    m = {"id": "m2", "question": "4100-4200", "description": ""}
    t = classify_market(e, m)
    assert t["marketClass"] == "RANGE_BUCKET"
    assert t["rangeLow"] == 4100.0 and t["rangeHigh"] == 4200.0


def test_classify_directional():
    e = {"id": "e3", "title": "Ethereum up or down - September 5, 7PM EST", "description": ""}
    m = {"id": "m3", "question": "Will ETH finish UP or DOWN from 7-8pm?", "description": ""}
    t = classify_market(e, m)
    assert t["marketClass"] == "DIRECTIONAL_PERIOD"
    assert "perpetuals" in t["strategies"]

def test_strategy_filtering_sol_perpetuals_only():
    # SOL single-threshold should be forced to perpetuals-only
    e = {"id": "e4", "title": "Solana above 250 on Sep 8", "description": ""}
    m = {
        "id": "m4",
        "question": "Solana above $250 on September 8?",
        "description": "Resolves on Binance SOLUSDT Close.",
        "endDate": "2025-09-08T20:00:00Z",
        "clobTokenIds": {"YES": "s1", "NO": "s2"},
    }
    t = classify_market(e, m)
    assert t["asset"] == "SOL"
    assert t["marketClass"] == "SINGLE_THRESHOLD"
    assert t["strategies"] == ["perpetuals"]

def test_strategy_filtering_xrp_range_perpetuals_only():
    # XRP range bucket should be forced to perpetuals-only (no options_spreads)
    e = {"id": "e5", "title": "XRP Price - September 6, 4PM ET", "description": ""}
    m = {"id": "m5", "question": "0.60-0.65", "description": "Resolves via Binance XRPUSDT Close."}
    t = classify_market(e, m)
    assert t["asset"] == "XRP"
    assert t["marketClass"] == "RANGE_BUCKET"
    assert t["strategies"] == ["perpetuals"]

def test_strategy_filtering_btc_keeps_options_and_variance_swap():
    # BTC should keep the richer strategy set for single-threshold
    e = {"id": "e6", "title": "Bitcoin above 60000 on Sep 8", "description": ""}
    m = {"id": "m6", "question": "$60,000", "description": "Will BTC hit 60k by Sep 8 (Binance BTCUSDT Close)?"}
    t = classify_market(e, m)
    assert t["asset"] == "BTC"
    assert t["marketClass"] == "SINGLE_THRESHOLD"
    # order-preserving check
    assert t["strategies"] == ["variance_swap", "perpetuals", "options_vanilla"]


def test_offline_tagging_fixture_exists_and_tags():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(repo_root, "debug_runs", "markets.json")
    assert os.path.exists(path), "Expected debug_runs/markets.json fixture to exist"
    with open(path, "r", encoding="utf-8") as f:
        markets = json.load(f)
    tagged = tag_from_local_markets(markets)
    assert len(tagged) >= 1
    assert all(
        t["marketClass"] in ("SINGLE_THRESHOLD", "DIRECTIONAL_PERIOD", "RANGE_BUCKET", "UNKNOWN")
        for t in tagged
    )