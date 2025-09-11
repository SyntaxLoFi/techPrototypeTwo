#!/usr/bin/env python3
"""
Utility to convert between detailed and summary opportunity data formats

This allows you to:
1. Extract summary from detailed data (for smaller files)
2. View specific parts of detailed data
3. Export options chains separately
"""

import json
import gzip
import argparse
import os
from datetime import datetime
from typing import Dict, List


def load_json_file(filepath: str) -> Dict:
    """Load JSON file (handles both regular and gzip compressed)"""
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)


def save_json_file(data: Dict, filepath: str, compress: bool = False):
    """Save JSON file (optionally compressed)"""
    if compress:
        filepath = filepath.replace('.json', '.json.gz')
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    else:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    print(f"Saved to {filepath}")


def extract_summary(detailed_data: Dict) -> Dict:
    """Extract summary version from detailed data"""
    summary = {
        'timestamp': detailed_data.get('timestamp'),
        'total_opportunities': detailed_data.get('total_opportunities'),
        'ranking_method': detailed_data.get('ranking_method'),
        'opportunities': []
    }
    
    for opp in detailed_data.get('opportunities', []):
        # Copy basic fields
        summary_opp = {
            'rank': opp.get('rank'),
            'quality_tier': opp.get('quality_tier'),
            'currency': opp.get('currency'),
            'hedge_type': opp.get('hedge_type'),
            'strategy': opp.get('strategy'),
            'max_profit': opp.get('max_profit'),
            'max_loss': opp.get('max_loss'),
            'risk_reward_ratio': opp.get('risk_reward_ratio'),
            'mispricing': opp.get('mispricing'),
            'polymarket': opp.get('polymarket'),
            'probability_metrics': opp.get('probability_metrics'),
            'probabilities': opp.get('probabilities')
        }
        
        # Add minimal Lyra/perp data
        if 'lyra' in opp:
            summary_opp['lyra'] = {
                'expiry': opp['lyra'].get('expiry'),
                'strikes': opp['lyra'].get('strikes'),
                'spread_cost': opp['lyra'].get('spread_cost')
            }
        
        if 'perp' in opp:
            summary_opp['perp'] = {
                'initial_position': opp['perp'].get('initial_position'),
                'current_funding': opp['perp'].get('current_funding'),
                'expected_funding_pnl': opp['perp'].get('expected_funding_pnl'),
                'funding_uncertainty': opp['perp'].get('funding_uncertainty')
            }
        
        summary['opportunities'].append(summary_opp)
    
    return summary


def extract_options_chains(detailed_data: Dict) -> Dict:
    """Extract all options chain data from detailed opportunities"""
    chains_data = {
        'timestamp': detailed_data.get('timestamp'),
        'opportunities_with_options': [],
        'unique_chains': {}
    }
    
    for opp in detailed_data.get('opportunities', []):
        if opp.get('hedge_type') == 'options' and 'available_options' in opp:
            chains_info = {
                'rank': opp.get('rank'),
                'currency': opp.get('currency'),
                'strategy': opp.get('strategy'),
                'polymarket_strike': opp.get('polymarket', {}).get('strike'),
                'expiries': opp['available_options'].get('expiries', []),
                'total_instruments': opp['available_options'].get('total_instruments'),
                'evaluated_options_sample': opp.get('evaluated_options_sample', [])
            }
            chains_data['opportunities_with_options'].append(chains_info)
            
            # Collect unique chains
            for expiry, chain in opp['available_options'].get('chains', {}).items():
                chain_key = f"{opp.get('currency')}_{expiry}"
                if chain_key not in chains_data['unique_chains']:
                    chains_data['unique_chains'][chain_key] = {
                        'currency': opp.get('currency'),
                        'expiry': expiry,
                        'strikes': list(chain.keys()),
                        'num_strikes': len(chain)
                    }
    
    return chains_data


def analyze_detailed_data(detailed_data: Dict):
    """Analyze and print statistics about detailed data"""
    print("\n=== Detailed Data Analysis ===")
    print(f"Timestamp: {detailed_data.get('timestamp')}")
    print(f"Total Opportunities: {detailed_data.get('total_opportunities')}")
    
    # Count by type
    options_count = 0
    perps_count = 0
    has_detailed_strategy = 0
    has_options_chains = 0
    total_options_evaluated = 0
    
    for opp in detailed_data.get('opportunities', []):
        if opp.get('hedge_type') == 'options':
            options_count += 1
            if 'detailed_strategy' in opp:
                has_detailed_strategy += 1
            if 'available_options' in opp:
                has_options_chains += 1
            if 'total_options_evaluated' in opp:
                total_options_evaluated += opp['total_options_evaluated']
        elif opp.get('hedge_type') == 'perpetuals':
            perps_count += 1
    
    print(f"\nStrategy Breakdown:")
    print(f"  Options Strategies: {options_count}")
    print(f"  Perpetual Strategies: {perps_count}")
    print(f"\nDetailed Data Available:")
    print(f"  With Detailed Strategy Info: {has_detailed_strategy}")
    print(f"  With Options Chains: {has_options_chains}")
    print(f"  Total Options Evaluated: {total_options_evaluated:,}")
    
    # Show sample of available data
    if detailed_data.get('opportunities'):
        first_opp = detailed_data['opportunities'][0]
        print(f"\nSample Opportunity Fields:")
        print(f"  Basic Fields: {list(first_opp.keys())[:10]}...")
        
        if 'detailed_strategy' in first_opp:
            print(f"  Detailed Strategy Fields: {list(first_opp['detailed_strategy'].keys())}")
        
        if 'available_options' in first_opp:
            avail_opts = first_opp['available_options']
            print(f"  Available Options:")
            print(f"    Expiries: {avail_opts.get('expiries', [])[:5]}...")
            print(f"    Total Instruments: {avail_opts.get('total_instruments')}")
            if 'chains' in avail_opts and avail_opts['chains']:
                first_expiry = list(avail_opts['chains'].keys())[0]
                num_strikes = len(avail_opts['chains'][first_expiry])
                print(f"    Sample Chain ({first_expiry}): {num_strikes} strikes")


def main():
    parser = argparse.ArgumentParser(description='Convert between detailed and summary opportunity data formats')
    parser.add_argument('input_file', help='Input JSON file (can be .json or .json.gz)')
    parser.add_argument('--to-summary', action='store_true', help='Convert detailed to summary format')
    parser.add_argument('--extract-chains', action='store_true', help='Extract options chains to separate file')
    parser.add_argument('--analyze', action='store_true', help='Analyze detailed data and show statistics')
    parser.add_argument('--output', '-o', help='Output filename (default: auto-generated)')
    parser.add_argument('--compress', '-c', action='store_true', help='Compress output file')
    
    args = parser.parse_args()
    
    # Load input file
    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} not found")
        return
    
    print(f"Loading {args.input_file}...")
    data = load_json_file(args.input_file)
    
    # Perform requested operation
    if args.analyze:
        analyze_detailed_data(data)
    
    if args.to_summary:
        print("Extracting summary...")
        summary_data = extract_summary(data)
        
        output_file = args.output or args.input_file.replace('detailed_', 'summary_').replace('.gz', '')
        save_json_file(summary_data, output_file, args.compress)
        
        # Show size reduction
        import sys
        original_size = sys.getsizeof(json.dumps(data))
        summary_size = sys.getsizeof(json.dumps(summary_data))
        print(f"Size reduction: {original_size:,} → {summary_size:,} bytes ({summary_size/original_size*100:.1f}%)")
    
    if args.extract_chains:
        print("Extracting options chains...")
        chains_data = extract_options_chains(data)
        
        output_file = args.output or args.input_file.replace('detailed_opportunities', 'options_chains').replace('.gz', '')
        save_json_file(chains_data, output_file, args.compress)
        
        print(f"Extracted {len(chains_data['opportunities_with_options'])} opportunities with options")
        print(f"Found {len(chains_data['unique_chains'])} unique option chains")


if __name__ == "__main__":
    main()