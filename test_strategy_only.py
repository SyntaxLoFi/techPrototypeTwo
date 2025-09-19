#!/usr/bin/env python3
"""
Test that only strategy-produced opportunities are created after removing fallback logic.
"""

import asyncio
import json
import logging
from pathlib import Path

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

async def test_opportunity_creation():
    """Run a minimal test of the opportunity builder."""
    try:
        from hedging.opportunity_builder import HedgeOpportunityBuilder
        from scripts.data_collection.polymarket_fetcher import PolymarketFetcher
        from scripts.data_collection.options_chain_collector import OptionsChainCollector
        
        # Create minimal scanner setup for BTC
        oc = OptionsChainCollector()
        try:
            oc.fetch_all_options('BTC')
        except:
            pass
            
        scanner = {
            'BTC': {
                'currency': 'BTC',
                'has_options': True,
                'options_collector': oc,
                'current_spot': 100000,  # dummy value
                'contracts': []  # Will be populated below
            }
        }
        
        # Get some test contracts
        pf = PolymarketFetcher()
        try:
            markets = pf.fetch_crypto_markets()[:10]  # Just first 10 for speed
        except:
            markets = []
            
        # Add BTC contracts
        btc_contracts = []
        for m in markets:
            if 'BTC' in str(m.get('question', '')).upper():
                m['currency'] = 'BTC'
                m['yes_price'] = 0.5  # Dummy price
                m['no_price'] = 0.5
                m['yes_size'] = 100
                m['no_size'] = 100
                # Only tag some with variance swap
                if len(btc_contracts) < 3:
                    m['strategyTags'] = ['options.variance_swap']
                    m['marketClass'] = 'SINGLE_THRESHOLD'
                btc_contracts.append(m)
                
        scanner['BTC']['contracts'] = btc_contracts
        
        # Build opportunities
        builder = HedgeOpportunityBuilder(scanner)
        opportunities = list(builder.build(scanner))
        
        # Check results
        print(f"\nTest Results:")
        print(f"Total contracts tested: {len(btc_contracts)}")
        print(f"Total opportunities created: {len(opportunities)}")
        
        # Check strategies
        strategies = {}
        for opp in opportunities:
            strat = opp.get('strategy', 'unknown')
            strategies[strat] = strategies.get(strat, 0) + 1
            
        print(f"\nStrategies found:")
        for strat, count in strategies.items():
            print(f"  {strat}: {count}")
            
        # Verify no fallback opportunities
        if 'no_strategy_available' in strategies:
            print("\n❌ FAIL: Found 'no_strategy_available' opportunities!")
            return False
        else:
            print("\n✅ PASS: No fallback opportunities found!")
            return True
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_opportunity_creation())
    exit(0 if success else 1)