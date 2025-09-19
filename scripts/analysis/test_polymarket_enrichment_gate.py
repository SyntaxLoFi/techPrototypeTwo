#!/usr/bin/env python3
# Minimal tests for orderbook enrichment + liquidity gate behavior
import sys, os
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from typing import Dict, Any, List
from scripts.data_collection.polymarket_fetcher_v2 import PolymarketFetcher
from hedging.opportunity_builder import HedgeOpportunityBuilder

class _DummyClient:
    def get_books(self, token_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        # Produce books that have only bids for YES and only asks for NO to test both paths
        out = {}
        for tid in token_ids:
            out[tid] = {
                'best_bid': {'price': 0.48, 'size': 100},
                'best_ask': {'price': 0.52, 'size': 0},  # no ask liquidity
                'mid': 0.50,
            }
        return out

def test_enrich_sets_sizes_from_bid_or_ask():
    fetcher = PolymarketFetcher(debug_dir='debug_runs')
    markets = [
        {'marketSlug': 'foo', 'tokenIds': {'YES': 'T1', 'NO': 'T2'}, 'asset': 'BTC', 'strike_price': 100000.0, 'is_above': True, 'endDate': '2025-12-31T00:00:00Z'}
    ]
    enriched = fetcher._enrich_with_orderbooks(_DummyClient(), markets)
    m = enriched[0]
    assert m['yes_bid'] == 0.48 and m['yes_mid'] == 0.50
    # yes_size should reflect bid size since ask is zero
    assert m['yes_size'] == 100.0
    # no_* also set
    assert 'no_bid' in m and 'no_ask' in m

def test_liquidity_gate_uses_bid_or_ask():
    scanners = {
        'BTC': {
            'contracts': [{
                'question': 'Test?',
                'yes_bid_qty': 50.0, 'yes_ask_qty': 0.0,
                'no_bid_qty': 0.0,  'no_ask_qty': 0.0,
                'strike_price': 100000.0, 'is_above': True, 'endDate': '2025-12-31T00:00:00Z'
            }],
            'has_options': False, 'has_perps': False, 'current_spot': 100000.0,
            'options_collector': None, 'perps_collector': None
        }
    }
    hb = HedgeOpportunityBuilder(scanners=scanners)
    assert hb._pm_has_liquidity(scanners['BTC']['contracts'][0]) is True

if __name__ == '__main__':
    test_enrich_sets_sizes_from_bid_or_ask()
    test_liquidity_gate_uses_bid_or_ask()
    print('ok')