#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from datetime import datetime
import re

def find_option_dates_in_all_strategies(file_path):
    """Find all option dates across all strategies in the file."""
    
    print("Reading JSON file...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    opportunities = data.get('opportunities', [])
    print(f"Total opportunities: {len(opportunities)}")
    
    # Track different date fields found
    date_field_paths = Counter()
    variance_swap_count = 0
    strategies_with_option_dates = []
    
    for i, opp in enumerate(opportunities):
        # Check if it's a variance swap
        pm_contract = opp.get('polymarket_contract', {})
        strategy_eligibility = pm_contract.get('strategyEligibility', {})
        options_eligibility = strategy_eligibility.get('options', {})
        is_variance_swap = options_eligibility.get('variance_swap') == True
        
        if is_variance_swap:
            variance_swap_count += 1
        
        # Look for option-related date fields recursively
        def search_for_option_dates(obj, path=""):
            found_dates = []
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    # Check if key contains option-related terms
                    if any(term in key.lower() for term in ['option', 'expir', 'maturity', 'strike']):
                        if isinstance(value, str) and re.match(r'\d{4}-\d{2}-\d{2}', value):
                            found_dates.append((new_path, value))
                            date_field_paths[new_path] += 1
                        elif isinstance(value, (dict, list)):
                            found_dates.extend(search_for_option_dates(value, new_path))
                    else:
                        if isinstance(value, (dict, list)):
                            found_dates.extend(search_for_option_dates(value, new_path))
            
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    found_dates.extend(search_for_option_dates(item, f"{path}[{idx}]"))
            
            return found_dates
        
        option_dates = search_for_option_dates(opp)
        if option_dates:
            strategies_with_option_dates.append({
                'index': i,
                'is_variance_swap': is_variance_swap,
                'strategy': opp.get('strategy'),
                'hedge_type': opp.get('hedge_type'),
                'dates': option_dates,
                'polymarket_end_date': pm_contract.get('endDate')
            })
    
    print(f"\nFound {variance_swap_count} variance swap strategies")
    print(f"Found {len(strategies_with_option_dates)} strategies with option dates")
    
    print("\nMost common option date field paths:")
    for path, count in date_field_paths.most_common(10):
        print(f"  {path}: {count} occurrences")
    
    # Show some examples
    print("\nExample strategies with option dates:")
    for i, strategy in enumerate(strategies_with_option_dates[:5]):
        print(f"\n{i+1}. Strategy index {strategy['index']}:")
        print(f"   Variance swap: {strategy['is_variance_swap']}")
        print(f"   Strategy type: {strategy['strategy']}")
        print(f"   Hedge type: {strategy['hedge_type']}")
        print(f"   Option dates found:")
        for path, date in strategy['dates']:
            print(f"     {path}: {date}")
    
    return strategies_with_option_dates, date_field_paths

if __name__ == "__main__":
    file_path = "results/unfiltered_opportunities_20250916_103329.json"
    
    try:
        strategies, paths = find_option_dates_in_all_strategies(file_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()