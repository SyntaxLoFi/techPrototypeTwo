#!/usr/bin/env python3
import json
from collections import defaultdict

def analyze_strategy_structures(file_path):
    """Analyze the structure of different strategy types."""
    
    print("Reading JSON file...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    opportunities = data.get('opportunities', [])
    print(f"Total opportunities: {len(opportunities)}")
    
    # Group strategies by type
    strategies_by_type = defaultdict(list)
    
    for opp in opportunities:
        strategy = opp.get('strategy', 'unknown')
        strategies_by_type[strategy].append(opp)
    
    print(f"\nStrategy types found: {list(strategies_by_type.keys())}")
    print("\nStrategy counts:")
    for strategy, items in strategies_by_type.items():
        print(f"  {strategy}: {len(items)}")
    
    # Check for variance swaps in each strategy type
    print("\nVariance swap eligibility by strategy type:")
    for strategy, items in strategies_by_type.items():
        variance_swap_count = 0
        for item in items:
            pm_contract = item.get('polymarket_contract', {})
            strategy_eligibility = pm_contract.get('strategyEligibility', {})
            options_eligibility = strategy_eligibility.get('options', {})
            if options_eligibility.get('variance_swap') == True:
                variance_swap_count += 1
        
        if variance_swap_count > 0:
            print(f"  {strategy}: {variance_swap_count} variance swaps")
    
    # Analyze a non-variance swap strategy to see if it has option dates
    print("\n\nLooking for strategies with actual option data...")
    
    # Find strategies that might have option dates
    for strategy_type, items in strategies_by_type.items():
        for item in items[:10]:  # Check first 10 of each type
            # Look for any field containing 'option' or 'strike'
            if has_option_data(item):
                print(f"\nFound potential option data in {strategy_type} strategy")
                print(f"Index in file: {opportunities.index(item)}")
                
                # Save this example
                with open(f'example_{strategy_type}_with_options.json', 'w') as f:
                    json.dump(item, f, indent=2)
                print(f"Saved to example_{strategy_type}_with_options.json")
                
                # Look for date fields
                dates = find_all_dates(item)
                if dates:
                    print("Dates found in this strategy:")
                    for path, date in dates[:10]:  # Show first 10
                        print(f"  {path}: {date}")
                
                break

def has_option_data(obj, path=""):
    """Check if object contains option-related data."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if any(term in str(key).lower() for term in ['option', 'strike', 'expir', 'maturity']):
                if key not in ['strategyEligibility', 'strategies', 'strategyCategories', 'strategyTags']:
                    return True
            if isinstance(value, (dict, list)):
                if has_option_data(value, f"{path}.{key}"):
                    return True
    elif isinstance(obj, list):
        for item in obj:
            if has_option_data(item, path):
                return True
    return False

def find_all_dates(obj, path=""):
    """Find all date strings in an object."""
    import re
    dates = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, str) and re.match(r'\d{4}-\d{2}-\d{2}', value):
                dates.append((new_path, value))
            elif isinstance(value, (dict, list)):
                dates.extend(find_all_dates(value, new_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            dates.extend(find_all_dates(item, f"{path}[{i}]"))
    
    return dates

if __name__ == "__main__":
    file_path = "results/unfiltered_opportunities_20250916_103329.json"
    analyze_strategy_structures(file_path)