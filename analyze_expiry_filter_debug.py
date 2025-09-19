#!/usr/bin/env python3
"""Analyze the expiry filter debug output to understand why options are being filtered."""

import json
import os
from collections import defaultdict, Counter
from datetime import datetime

def analyze_expiry_filter_debug():
    """Analyze expiry filter debug log."""
    debug_file = "debug_runs/expiry_filter_debug.jsonl"
    
    if not os.path.exists(debug_file):
        print(f"Debug file not found: {debug_file}")
        print("Please run the variance swap strategy first to generate debug data.")
        return
    
    # Aggregate statistics
    total_markets = 0
    rejection_reasons = Counter()
    expiry_patterns = defaultdict(int)
    markets_with_zero_options = []
    
    with open(debug_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                entry = json.loads(line)
                stats = entry['stats']
                pm_dte = entry['pm_dte']
                
                total_markets += 1
                
                # Track rejection reasons
                rejection_reasons['missing_expiry'] += stats['missing_expiry']
                rejection_reasons['synthetic_or_flagged'] += stats['synthetic_or_flagged']
                rejection_reasons['failed_quote_validation'] += stats['failed_quote_validation']
                rejection_reasons['expired_before_pm'] += stats['expired_before_pm']
                rejection_reasons['missing_dte'] += stats['missing_dte']
                
                # Check if all options were filtered
                if stats.get('final_options_count', 0) == 0:
                    markets_with_zero_options.append({
                        'pm_dte': pm_dte,
                        'total_options': stats['total_options'],
                        'rejections': {
                            'missing_expiry': stats['missing_expiry'],
                            'synthetic_or_flagged': stats['synthetic_or_flagged'],
                            'failed_quote_validation': stats['failed_quote_validation'],
                            'expired_before_pm': stats['expired_before_pm'],
                            'missing_dte': stats['missing_dte']
                        },
                        'expiry_groups': stats['expiry_groups'],
                        'rejected_expiries': stats['rejected_expiries']
                    })
                
                # Track expiry patterns
                for expiry in stats['expiry_groups']:
                    expiry_patterns[expiry] += 1
                    
            except json.JSONDecodeError:
                continue
            except KeyError as e:
                print(f"Warning: Missing key {e} in entry")
                continue
    
    # Print analysis
    print(f"\n{'='*80}")
    print("EXPIRY FILTER DEBUG ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Total markets analyzed: {total_markets}")
    print(f"Markets with zero options after filtering: {len(markets_with_zero_options)}")
    print(f"Success rate: {((total_markets - len(markets_with_zero_options)) / total_markets * 100):.1f}%\n")
    
    print("REJECTION REASONS (across all options):")
    print("-" * 40)
    total_rejections = sum(rejection_reasons.values())
    for reason, count in rejection_reasons.most_common():
        pct = (count / total_rejections * 100) if total_rejections > 0 else 0
        print(f"  {reason:30s}: {count:8d} ({pct:5.1f}%)")
    
    print(f"\nTOTAL REJECTIONS: {total_rejections}")
    
    # Analyze specific failure patterns
    if markets_with_zero_options:
        print("\n" + "="*80)
        print("DETAILED ANALYSIS OF FAILED MARKETS")
        print("="*80)
        
        # Sample the first few failed markets
        for i, market in enumerate(markets_with_zero_options[:5]):
            print(f"\nFailed Market {i+1}:")
            print(f"  PM days to expiry: {market['pm_dte']:.1f}")
            print(f"  Total options available: {market['total_options']}")
            
            print("  Rejections breakdown:")
            for reason, count in market['rejections'].items():
                if count > 0:
                    print(f"    - {reason}: {count}")
            
            if market['expiry_groups']:
                print(f"  Expiry groups found: {len(market['expiry_groups'])}")
                for expiry, count in sorted(market['expiry_groups'].items())[:3]:
                    print(f"    - {expiry}: {count} options")
            
            if market['rejected_expiries']:
                print(f"  Rejected expiries: {len(market['rejected_expiries'])}")
                for expiry, info in sorted(market['rejected_expiries'].items())[:3]:
                    print(f"    - {expiry}: {info}")
    
    # Most common expiry dates
    print("\n" + "="*80)
    print("MOST COMMON OPTION EXPIRY DATES")
    print("="*80)
    for expiry, count in Counter(expiry_patterns).most_common(10):
        print(f"  {expiry}: appeared in {count} markets")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if rejection_reasons['failed_quote_validation'] > total_rejections * 0.5:
        print("- High quote validation failures: Check if options have bid/ask data")
        print("- Consider setting require_live_quotes=False if quotes are missing")
    
    if rejection_reasons['expired_before_pm'] > total_rejections * 0.3:
        print("- Many options expire before PM date: Check option expiry alignment")
        print("- Consider markets with shorter time horizons")
    
    if rejection_reasons['synthetic_or_flagged'] > total_rejections * 0.2:
        print("- Many synthetic/flagged options: Review data source quality")

if __name__ == "__main__":
    analyze_expiry_filter_debug()