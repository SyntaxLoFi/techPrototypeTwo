#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from datetime import datetime
import re

def find_dates_in_object(obj, path=""):
    """Recursively find all date-like strings in an object."""
    dates_found = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, str):
                # Look for date patterns (YYYY-MM-DD or ISO format)
                if re.match(r'\d{4}-\d{2}-\d{2}', value):
                    dates_found.append((new_path, value))
            else:
                dates_found.extend(find_dates_in_object(value, new_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            dates_found.extend(find_dates_in_object(item, f"{path}[{i}]"))
    
    return dates_found

def extract_variance_swap_strategies(file_path):
    """Extract all strategies with variance_swap = true and their option expiry dates."""
    
    variance_swap_strategies = []
    option_dates_counter = Counter()
    polymarket_dates_map = defaultdict(set)  # Map option date to polymarket dates
    
    print("Reading JSON file...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    opportunities = data.get('opportunities', [])
    print(f"Total opportunities: {len(opportunities)}")
    
    # Extract variance swap strategies
    for opp in opportunities:
        pm_contract = opp.get('polymarket_contract', {})
        strategy_eligibility = pm_contract.get('strategyEligibility', {})
        options_eligibility = strategy_eligibility.get('options', {})
        
        if options_eligibility.get('variance_swap') == True:
            variance_swap_strategies.append(opp)
            
            # Debug: analyze first variance swap strategy
            if len(variance_swap_strategies) == 1:
                print("\nFirst variance swap strategy - analyzing for dates:")
                all_dates = find_dates_in_object(opp)
                print(f"Found {len(all_dates)} date fields:")
                for path, date in all_dates:
                    print(f"  {path}: {date}")
                
                with open('first_variance_swap.json', 'w') as f:
                    json.dump(opp, f, indent=2)
                print("\nSaved full strategy to first_variance_swap.json")
            
            # Since variance swaps might not have explicit option dates,
            # let's check if they reference option dates in any field
            # Check if there's an 'option_expiry' field
            if 'option_expiry' in opp:
                option_date = opp['option_expiry']
                option_dates_counter[option_date] += 1
                pm_end_date = pm_contract.get('endDate', 'Unknown')
                if pm_end_date:
                    try:
                        pm_date_obj = datetime.fromisoformat(pm_end_date.replace('Z', '+00:00'))
                        pm_date_str = pm_date_obj.strftime('%Y-%m-%d')
                    except:
                        pm_date_str = pm_end_date
                    polymarket_dates_map[option_date].add(pm_date_str)
            
            # Check in 'options' field
            if 'options' in opp:
                options = opp['options']
                if isinstance(options, dict):
                    for key in ['expiry', 'expiry_date', 'maturity', 'date']:
                        if key in options:
                            option_date = options[key]
                            option_dates_counter[option_date] += 1
                            pm_end_date = pm_contract.get('endDate', 'Unknown')
                            if pm_end_date:
                                try:
                                    pm_date_obj = datetime.fromisoformat(pm_end_date.replace('Z', '+00:00'))
                                    pm_date_str = pm_date_obj.strftime('%Y-%m-%d')
                                except:
                                    pm_date_str = pm_end_date
                                polymarket_dates_map[option_date].add(pm_date_str)
                elif isinstance(options, list):
                    for opt in options:
                        if isinstance(opt, dict):
                            for key in ['expiry', 'expiry_date', 'maturity', 'date']:
                                if key in opt:
                                    option_date = opt[key]
                                    option_dates_counter[option_date] += 1
                                    pm_end_date = pm_contract.get('endDate', 'Unknown')
                                    if pm_end_date:
                                        try:
                                            pm_date_obj = datetime.fromisoformat(pm_end_date.replace('Z', '+00:00'))
                                            pm_date_str = pm_date_obj.strftime('%Y-%m-%d')
                                        except:
                                            pm_date_str = pm_end_date
                                        polymarket_dates_map[option_date].add(pm_date_str)
    
    print(f"\nFound {len(variance_swap_strategies)} variance swap strategies")
    print(f"Found {len(option_dates_counter)} unique option expiry dates")
    
    # Create sorted list of dates with counts
    date_list = []
    for option_date, count in sorted(option_dates_counter.items()):
        pm_dates = sorted(list(polymarket_dates_map[option_date]))
        for pm_date in pm_dates:
            date_list.append({
                'polymarket_date': pm_date,
                'option_date': option_date,
                'count': count
            })
    
    return variance_swap_strategies, date_list

def save_results(strategies, date_list, output_prefix='variance_swap'):
    """Save the results to files."""
    
    # Save date list
    date_file = f"{output_prefix}_dates.json"
    with open(date_file, 'w') as f:
        json.dump(date_list, f, indent=2)
    print(f"\nSaved date list to {date_file}")
    
    # Save date list as CSV for easy viewing
    csv_file = f"{output_prefix}_dates.csv"
    with open(csv_file, 'w') as f:
        f.write("Polymarket Date,Option Date,Count\n")
        for item in date_list:
            f.write(f"{item['polymarket_date']},{item['option_date']},{item['count']}\n")
    print(f"Saved date list CSV to {csv_file}")
    
    # Print summary
    print("\nDate Summary:")
    print("Polymarket Date | Option Date | Count")
    print("-" * 50)
    for item in date_list[:20]:  # Show first 20
        print(f"{item['polymarket_date']:15} | {item['option_date']:11} | {item['count']:5}")
    
    if len(date_list) > 20:
        print(f"... and {len(date_list) - 20} more entries")
    
    return date_file, csv_file

if __name__ == "__main__":
    file_path = "results/unfiltered_opportunities_20250916_103329.json"
    
    try:
        strategies, date_list = extract_variance_swap_strategies(file_path)
        save_results(strategies, date_list)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()