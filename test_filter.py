#!/usr/bin/env python3
import json
import sys
sys.path.append('/Users/wade/Documents/techPrototypeTwo')

from strategies.options.variance_swap_strategy import VarianceSwapStrategy
from datetime import datetime, timezone

# Load sample options
with open('/Users/wade/Documents/techPrototypeTwo/debug_runs/20250918-142750Z/options/BTC_options.json', 'r') as f:
    options = json.load(f)

# Create test PM contract
pm_contract = {
    'question': 'Will Bitcoin reach $200,000 by December 31, 2025?',
    'strike_price': 200000,
    'yes_price': 0.05,
    'no_price': 0.95,
    'is_above': True,
    'days_to_expiry': 103.3,  # Dec 31, 2025
    'currency': 'BTC',
    'ticker': 'BTC'
}

# Initialize strategy
import logging
logger = logging.getLogger('test')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(handler)

cfg = {}  # Empty config
strategy = VarianceSwapStrategy(cfg, logger)
# Disable live quotes requirement for testing
strategy.require_live_quotes = False

# Test filter
print(f"Total options before filter: {len(options)}")
print(f"PM days to expiry: {pm_contract['days_to_expiry']}")

# Call evaluate_opportunities which will test the filtering
hedge_instruments = {'options': options}

# Add more debug info
print("\nSample option data:")
if options:
    sample = options[0]
    print(f"  expiry_date: {sample.get('expiry_date')}")
    print(f"  days_to_expiry: {sample.get('days_to_expiry')}")
    print(f"  bid: {sample.get('bid')}")
    print(f"  ask: {sample.get('ask')}")

opportunities = strategy.evaluate_opportunities(pm_contract, hedge_instruments, 117000)
print(f"\nOpportunities found: {len(opportunities)}")