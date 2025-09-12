#!/usr/bin/env python3
"""
Script to analyze variance swap strategies and their option expiry dates
"""

import json
from collections import Counter, defaultdict
from datetime import datetime

def analyze_variance_swaps(filename):
    """
    Extract variance swap strategies and analyze their expiry dates
    """
    # Store full records for detailed analysis
    variance_swap_records = []
    variance_swap_count = 0
    
    print(f"Loading data from {filename}...")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"Total opportunities: {data.get('total_opportunities', 0)}")
    
    # Process each opportunity
    for opportunity in data.get('opportunities', []):
        # Check if it has variance_swap = true
        # The polymarket data is nested within the opportunity
        polymarket = opportunity.get('polymarket', {})
        strategy_eligibility = polymarket.get('strategyEligibility', {})
        options_eligibility = strategy_eligibility.get('options', {})
        
        if options_eligibility.get('variance_swap', False):
            variance_swap_count += 1
            
            # Extract end date from polymarket
            end_date = polymarket.get('endDate')
            
            if end_date:
                # Convert to date string (remove time part for grouping)
                try:
                    date_obj = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    date_str = date_obj.strftime('%Y-%m-%d')
                    
                    # Store the record with market info
                    # Check if there's a separate options expiry date
                    # For now, we'll look for any other date field in the opportunity
                    options_date = date_str  # Default to same as polymarket
                    
                    # Check if there's an options field at the opportunity level
                    if 'options' in opportunity:
                        options_data = opportunity.get('options', {})
                        if 'expiry' in options_data or 'expiry_date' in options_data or 'expiryDate' in options_data:
                            opt_exp = options_data.get('expiry') or options_data.get('expiry_date') or options_data.get('expiryDate')
                            if opt_exp:
                                try:
                                    opt_date_obj = datetime.fromisoformat(opt_exp.replace('Z', '+00:00'))
                                    options_date = opt_date_obj.strftime('%Y-%m-%d')
                                except:
                                    pass
                    
                    variance_swap_records.append({
                        'polymarket_date': date_str,
                        'options_date': options_date,
                        'market_title': polymarket.get('eventTitle', 'Unknown'),
                        'asset': polymarket.get('asset', 'Unknown')
                    })
                except Exception as e:
                    print(f"Error parsing date {end_date}: {e}")
    
    print(f"\nFound {variance_swap_count} variance swap strategies")
    
    # Count occurrences by date combination
    date_combinations = defaultdict(int)
    for record in variance_swap_records:
        key = (record['polymarket_date'], record['options_date'])
        date_combinations[key] += 1
    
    # Sort by polymarket date
    sorted_combinations = sorted(date_combinations.items(), key=lambda x: x[0][0])
    
    print(f"\nFull list of dates used:")
    print("-" * 60)
    print(f"{'Polymarket Date':<18} {'Options Date':<18} {'Count':>10}")
    print("-" * 60)
    
    total_count = 0
    for (polymarket_date, options_date), count in sorted_combinations:
        print(f"{polymarket_date:<18} {options_date:<18} {count:>10}")
        total_count += count
    
    print("-" * 60)
    print(f"{'Total':<37} {total_count:>10}")
    
    # Show summary by unique polymarket dates
    polymarket_dates_count = Counter([record['polymarket_date'] for record in variance_swap_records])
    
    print(f"\nSummary by Polymarket date (total unique dates: {len(polymarket_dates_count)}):")
    print("-" * 40)
    print(f"{'Date':<15} {'Count':>10}")
    print("-" * 40)
    
    for date, count in sorted(polymarket_dates_count.items()):
        print(f"{date:<15} {count:>10}")
    
    # Highlight September 12, 2025 specifically
    sept_12_count = polymarket_dates_count.get('2025-09-12', 0)
    print(f"\n** September 12, 2025 is used {sept_12_count} times **")
    
    # Check for any differences between polymarket and options dates
    different_dates = []
    for record in variance_swap_records:
        if record['polymarket_date'] != record['options_date']:
            different_dates.append(record)
    
    if different_dates:
        print(f"\nFound {len(different_dates)} strategies with different PM and Options dates:")
        print("-" * 80)
        for record in different_dates[:10]:  # Show first 10
            print(f"PM: {record['polymarket_date']} | Options: {record['options_date']} | {record['asset']} | {record['market_title'][:50]}...")
    else:
        print("\n** All strategies have the same Polymarket and Options expiration dates **")

if __name__ == "__main__":
    analyze_variance_swaps("results/detailed_opportunities_20250911_210005.json")