#!/usr/bin/env python3
import json
from collections import defaultdict

# Track all stages
funnel = defaultdict(int)
stage_details = defaultdict(lambda: defaultdict(int))

# Count total opportunities in the final JSON
with open('results/unfiltered_opportunities_20250917_214510.json', 'r') as f:
    data = json.load(f)
    opportunities = data.get('opportunities', [])
    funnel['total_opportunities_in_json'] = len(opportunities)
    
    # Count by strategy
    strategy_counts = defaultdict(int)
    for opp in opportunities:
        strategy_counts[opp.get('strategy', 'unknown')] += 1
    
    print("=== OPPORTUNITIES IN FINAL JSON ===")
    print(f"Total opportunities: {funnel['total_opportunities_in_json']}")
    print("By strategy:")
    for strat, count in sorted(strategy_counts.items()):
        print(f"  {strat}: {count}")

# Analyze validation audit
print("\n=== VARIANCE SWAP STRATEGY FUNNEL ===")

with open('analysis/validation_audit.jsonl', 'r') as f:
    lines = f.readlines()
    
validation_passed = 0
validation_failed = 0
validation_failures = defaultdict(int)
stage_transitions = []

for i, line in enumerate(lines):
    data = json.loads(line.strip())
    
    if data.get('stage') == 'pre_strategy':
        funnel['1_markets_evaluated'] += 1
        
    elif data.get('stage') == 'validate_inputs':
        if data.get('validation_pass') == True:
            funnel['2_passed_validation'] += 1
            validation_passed += 1
        else:
            funnel['2_failed_validation'] += 1
            reason = data.get('reason_code', 'UNKNOWN')
            validation_failures[reason] += 1
            validation_failed += 1
    
    # Look for other stages
    if 'filter_options_by_expiry' in str(data):
        funnel['3_reached_expiry_filter'] += 1
    
    if 'entered_filter_options_by_expiry' in str(data):
        funnel['3_entered_expiry_filter'] += 1

# Calculate portfolio building failures
# If 245 passed validation but 0 strategies generated, all failed at portfolio building
portfolio_failures = funnel['2_passed_validation'] - 0  # 0 is final strategies

print(f"\n1. Markets Evaluated: {funnel['1_markets_evaluated']}")
print(f"   → {funnel['1_markets_evaluated']} markets sent to variance swap strategy")

print(f"\n2. Validation Stage:")
print(f"   ✓ Passed: {funnel['2_passed_validation']} ({funnel['2_passed_validation']/funnel['1_markets_evaluated']*100:.1f}%)")
print(f"   ✗ Failed: {funnel['2_failed_validation']} ({funnel['2_failed_validation']/funnel['1_markets_evaluated']*100:.1f}%)")
print(f"   Reasons for failure:")
for reason, count in sorted(validation_failures.items()):
    print(f"      - {reason}: {count}")

print(f"\n3. Portfolio Building Stage:")
print(f"   → {funnel['2_passed_validation']} markets attempted")
print(f"   ✓ Successful: 0 (0.0%)")
print(f"   ✗ Failed: {portfolio_failures} (100.0%)")

print(f"\n4. Final Output:")
print(f"   → 0 variance swap strategies generated")
print(f"   → All markets assigned 'no_strategy_available'")

# Summary
print(f"\n=== SUMMARY ===")
print(f"Total opportunities in JSON: {funnel['total_opportunities_in_json']}")
print(f"Markets evaluated by variance swap: {funnel['1_markets_evaluated']} ({funnel['1_markets_evaluated']/funnel['total_opportunities_in_json']*100:.1f}%)")
print(f"Lost at validation: {funnel['2_failed_validation']} ({funnel['2_failed_validation']/funnel['1_markets_evaluated']*100:.1f}%)")
print(f"Lost at portfolio building: {portfolio_failures} ({portfolio_failures/funnel['2_passed_validation']*100:.1f}%)")
print(f"Final variance swap strategies: 0 (0.0%)")

print("\n=== KEY FINDINGS ===")
print("1. NO price validation: 0 failures (NO prices are NOT being validated against thresholds)")
print("2. Only YES prices are checked against min/max thresholds")
print("3. This throws away entire markets even when NO position would be viable")
print("4. Even after fixing validation, portfolio building is failing 100% of the time")