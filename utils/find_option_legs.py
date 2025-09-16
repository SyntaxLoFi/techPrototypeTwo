#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from datetime import datetime

def find_option_legs_in_variance_swaps(file_path):
    """Look for option leg data in variance swap strategies."""
    
    print("Reading JSON file...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    opportunities = data.get('opportunities', [])
    print(f"Total opportunities: {len(opportunities)}")
    
    variance_swaps = []
    option_dates_counter = Counter()
    date_details = defaultdict(list)
    
    # Look through all opportunities
    for opp in opportunities:
        pm_contract = opp.get('polymarket_contract', {})
        strategy_eligibility = pm_contract.get('strategyEligibility', {})
        options_eligibility = strategy_eligibility.get('options', {})
        
        if options_eligibility.get('variance_swap') == True:
            variance_swaps.append(opp)
            
            # Check all top-level keys for option/leg data
            for key in opp.keys():
                if 'leg' in key.lower() or 'option' in key.lower():
                    print(f"\nFound key '{key}' in variance swap")
                    print(f"Value: {opp[key]}")
            
            # Look for any field that might contain dates
            # Common patterns: legs, options_legs, hedge_legs, etc.
            potential_leg_fields = [
                'legs', 'option_legs', 'options_legs', 'hedge_legs',
                'long_legs', 'short_legs', 'buy_legs', 'sell_legs',
                'calls', 'puts', 'options', 'strikes'
            ]
            
            for field in potential_leg_fields:
                if field in opp:
                    print(f"\nFound '{field}' field in variance swap!")
                    leg_data = opp[field]
                    
                    # If it's a list of legs
                    if isinstance(leg_data, list):
                        for i, leg in enumerate(leg_data):
                            print(f"  Leg {i}: {leg}")
                            
                            # Look for dates in the leg
                            if isinstance(leg, dict):
                                for k, v in leg.items():
                                    if any(term in k.lower() for term in ['date', 'expir', 'maturity']):
                                        print(f"    Found date field '{k}': {v}")
                                        option_dates_counter[v] += 1
                                        
                                        # Store details
                                        pm_date = pm_contract.get('endDate', 'Unknown')
                                        if pm_date:
                                            try:
                                                pm_date_obj = datetime.fromisoformat(pm_date.replace('Z', '+00:00'))
                                                pm_date_str = pm_date_obj.strftime('%Y-%m-%d')
                                            except:
                                                pm_date_str = pm_date
                                        else:
                                            pm_date_str = 'Unknown'
                                        
                                        date_details[v].append({
                                            'polymarket_date': pm_date_str,
                                            'field': f"{field}[{i}].{k}",
                                            'question': pm_contract.get('question', 'Unknown')
                                        })
    
    print(f"\n\nTotal variance swaps analyzed: {len(variance_swaps)}")
    
    if len(variance_swaps) > 0 and len(option_dates_counter) == 0:
        # No option dates found in expected places
        # Let's check the first variance swap's complete structure
        print("\nNo option dates found in expected fields.")
        print("Checking complete structure of first variance swap...")
        
        first_vs = variance_swaps[0]
        print("\nAll top-level keys in first variance swap:")
        for key in sorted(first_vs.keys()):
            value = first_vs[key]
            if isinstance(value, (dict, list)):
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"  {key}: {value}")
    
    return option_dates_counter, date_details

if __name__ == "__main__":
    file_path = "results/unfiltered_opportunities_20250916_103329.json"
    
    dates_counter, details = find_option_legs_in_variance_swaps(file_path)
    
    if dates_counter:
        print("\n\nOption Expiry Dates Summary:")
        print("Option Date | Count | Example Polymarket Dates")
        print("-" * 60)
        
        for date, count in sorted(dates_counter.items()):
            pm_dates = set(d['polymarket_date'] for d in details[date])
            pm_dates_str = ", ".join(sorted(list(pm_dates))[:3])
            if len(pm_dates) > 3:
                pm_dates_str += "..."
            print(f"{date:11} | {count:5} | {pm_dates_str}")
    else:
        print("\nNo option expiry dates found in standard locations.")