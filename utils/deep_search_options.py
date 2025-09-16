#!/usr/bin/env python3
import json
from collections import defaultdict
import re

def deep_search_for_options(file_path):
    """Deep search for any option-related data in the JSON."""
    
    print("Reading JSON file...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    opportunities = data.get('opportunities', [])
    print(f"Total opportunities: {len(opportunities)}")
    
    # Find all unique keys in the data
    all_keys = set()
    
    def collect_keys(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                all_keys.add(full_key)
                if isinstance(value, (dict, list)):
                    collect_keys(value, full_key)
        elif isinstance(obj, list) and obj:
            if isinstance(obj[0], dict):
                collect_keys(obj[0], f"{prefix}[0]")
    
    # Collect keys from first few variance swap strategies
    variance_swaps_found = 0
    for opp in opportunities:
        pm_contract = opp.get('polymarket_contract', {})
        strategy_eligibility = pm_contract.get('strategyEligibility', {})
        options_eligibility = strategy_eligibility.get('options', {})
        
        if options_eligibility.get('variance_swap') == True:
            if variance_swaps_found < 3:
                print(f"\n\nVariance Swap #{variance_swaps_found + 1} - All fields:")
                collect_keys(opp)
                
                # Print the entire structure with focus on option/strike/expiry related fields
                def print_option_fields(obj, indent=0):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if any(term in str(key).lower() for term in ['option', 'strike', 'expir', 'maturity', 'leg', 'hedge']):
                                print(" " * indent + f"{key}: ", end="")
                                if isinstance(value, (dict, list)):
                                    print()
                                    print_option_fields(value, indent + 2)
                                else:
                                    print(value)
                
                print("\nOption-related fields found:")
                print_option_fields(opp)
                
                # Save this variance swap for detailed inspection
                with open(f'variance_swap_example_{variance_swaps_found + 1}.json', 'w') as f:
                    json.dump(opp, f, indent=2)
                print(f"\nSaved to variance_swap_example_{variance_swaps_found + 1}.json")
                
            variance_swaps_found += 1
    
    # Print all unique keys that might contain option data
    print("\n\nAll unique keys containing option/strike/expiry/maturity/leg:")
    option_keys = [key for key in sorted(all_keys) if any(term in key.lower() for term in ['option', 'strike', 'expir', 'maturity', 'leg'])]
    for key in option_keys:
        print(f"  {key}")

if __name__ == "__main__":
    file_path = "results/unfiltered_opportunities_20250916_103329.json"
    deep_search_for_options(file_path)