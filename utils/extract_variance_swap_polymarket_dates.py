#!/usr/bin/env python3
import json
from collections import Counter
from datetime import datetime

def extract_variance_swap_polymarket_dates(file_path):
    """Extract Polymarket end dates from variance swap strategies."""
    
    print("Reading JSON file...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    opportunities = data.get('opportunities', [])
    print(f"Total opportunities: {len(opportunities)}")
    
    # Extract variance swap strategies and their Polymarket dates
    variance_swap_strategies = []
    polymarket_dates_counter = Counter()
    
    for opp in opportunities:
        pm_contract = opp.get('polymarket_contract', {})
        strategy_eligibility = pm_contract.get('strategyEligibility', {})
        options_eligibility = strategy_eligibility.get('options', {})
        
        if options_eligibility.get('variance_swap') == True:
            variance_swap_strategies.append(opp)
            
            # Get Polymarket end date
            pm_end_date = pm_contract.get('endDate', 'Unknown')
            if pm_end_date and pm_end_date != 'Unknown':
                # Convert to simple date format
                try:
                    pm_date_obj = datetime.fromisoformat(pm_end_date.replace('Z', '+00:00'))
                    pm_date_str = pm_date_obj.strftime('%Y-%m-%d')
                    polymarket_dates_counter[pm_date_str] += 1
                except:
                    polymarket_dates_counter[pm_end_date] += 1
    
    print(f"\nFound {len(variance_swap_strategies)} variance swap strategies")
    print(f"Found {len(polymarket_dates_counter)} unique Polymarket end dates")
    
    # Create sorted list with counts
    date_list = []
    for pm_date, count in sorted(polymarket_dates_counter.items()):
        date_list.append({
            'polymarket_date': pm_date,
            'option_date': pm_date,  # Assuming Polymarket date IS the option date
            'count': count
        })
    
    return variance_swap_strategies, date_list

def save_results(date_list, output_prefix='variance_swap_pm'):
    """Save the results to files."""
    
    # Save as CSV for easy viewing
    csv_file = f"{output_prefix}_dates.csv"
    with open(csv_file, 'w') as f:
        f.write("Polymarket Date,Option Date,Count\n")
        for item in date_list:
            f.write(f"{item['polymarket_date']},{item['option_date']},{item['count']}\n")
    print(f"\nSaved date list CSV to {csv_file}")
    
    # Save as JSON
    json_file = f"{output_prefix}_dates.json"
    with open(json_file, 'w') as f:
        json.dump(date_list, f, indent=2)
    print(f"Saved date list JSON to {json_file}")
    
    # Print full list as requested
    print("\nFull list of dates with counts:")
    print("Polymarket Date | Option Date | Count")
    print("-" * 50)
    total_count = 0
    for item in date_list:
        print(f"{item['polymarket_date']:15} | {item['option_date']:11} | {item['count']:5}")
        total_count += item['count']
    
    print(f"\nTotal variance swap strategies: {total_count}")
    
    return csv_file, json_file

if __name__ == "__main__":
    file_path = "results/unfiltered_opportunities_20250916_103329.json"
    
    try:
        strategies, date_list = extract_variance_swap_polymarket_dates(file_path)
        save_results(date_list)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()