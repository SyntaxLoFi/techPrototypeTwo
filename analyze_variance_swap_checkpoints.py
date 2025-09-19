#!/usr/bin/env python3
"""
Analyze variance_swap_checkpoints.log to provide a summary of where opportunities are being filtered.
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_checkpoint_log(filepath: str) -> Dict:
    """Parse the checkpoint log and extract filtering statistics."""
    
    stats = {
        'total_markets': 0,
        'checkpoint_failures': defaultdict(int),
        'checkpoint_details': defaultdict(list),
        'final_opportunities': 0,
        'markets_with_opportunities': 0,
        'expiry_filter_stats': {'total': 0, 'zero_options': 0, 'option_reduction': []},
        'price_gate_stats': {'yes_invalid': 0, 'no_invalid': 0, 'both_invalid': 0},
        'too_close_expiry': 0,
        'portfolio_build_failures': defaultdict(int),
        'forward_pm_gap_failures': 0,
        'cost_recovery_failures': 0,
        'digital_bounds_failures': 0
    }
    
    current_market = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # New market entry
            if line.startswith('==') and line.endswith('=='):
                continue
            elif 'Market:' in line and 'Starting with' not in line:
                stats['total_markets'] += 1
                current_market = re.search(r'Market: (.+?)$', line).group(1)
            
            # Basic validation failed (0B)
            elif 'Checkpoint 0B:' in line:
                stats['checkpoint_failures']['0B_basic_validation'] += 1
                stats['checkpoint_details']['0B_basic_validation'].append(current_market)
            
            # Missing NO price (0A)
            elif 'Checkpoint 0A:' in line:
                stats['checkpoint_failures']['0A_missing_no_price'] += 1
                stats['checkpoint_details']['0A_missing_no_price'].append(current_market)
            
            # PM price-range gate (1)
            elif 'Checkpoint 1:' in line:
                match = re.search(r'YES price: ([\d.]+) \((valid|invalid)\), NO price: ([\d.]+) \((valid|invalid)\)', line)
                if match:
                    yes_valid = match.group(2) == 'valid'
                    no_valid = match.group(4) == 'valid'
                    if not yes_valid:
                        stats['price_gate_stats']['yes_invalid'] += 1
                    if not no_valid:
                        stats['price_gate_stats']['no_invalid'] += 1
                    if not yes_valid and not no_valid:
                        stats['price_gate_stats']['both_invalid'] += 1
                        stats['checkpoint_failures']['1_price_gate_both_invalid'] += 1
                        stats['checkpoint_details']['1_price_gate_both_invalid'].append(current_market)
            
            # Expiry selection filter (2)
            elif 'Checkpoint 2:' in line:
                match = re.search(r'Options: (\d+) -> (\d+)', line)
                if match:
                    before = int(match.group(1))
                    after = int(match.group(2))
                    stats['expiry_filter_stats']['total'] += 1
                    stats['expiry_filter_stats']['option_reduction'].append((before, after))
                    if after == 0:
                        stats['expiry_filter_stats']['zero_options'] += 1
                        stats['checkpoint_failures']['2_expiry_filter'] += 1
                        stats['checkpoint_details']['2_expiry_filter'].append(current_market)
            
            # Too close expiry (3)
            elif 'Checkpoint 3:' in line:
                stats['too_close_expiry'] += 1
                stats['checkpoint_failures']['3_too_close_expiry'] += 1
                match = re.search(r'(\d{4}-\d{2}-\d{2}) has ([\d.]+) days', line)
                if match:
                    stats['checkpoint_details']['3_too_close_expiry'].append(f"{current_market} ({match.group(2)} days)")
            
            # Portfolio build failures (4)
            elif 'Checkpoint 4:' in line and 'DROPPED' in line:
                stats['checkpoint_failures']['4_portfolio_build'] += 1
                low = line.lower()
                if ('insufficient st' in low and 'normal' in low):
                    stats['portfolio_build_failures']['insufficient_strikes_normalization'] += 1
                elif ('no strikes on' in low and 'both sides' in low):
                    stats['portfolio_build_failures']['no_strikes_both_sides'] += 1
                elif 'dedup' in low:
                    stats['portfolio_build_failures']['insufficient_strikes_dedup'] += 1
                elif 'forward outside strike range' in low:
                    stats['portfolio_build_failures']['forward_outside_range'] += 1
                elif 'wing coverage' in low or 'wing coverage after truncation' in low:
                    stats['portfolio_build_failures']['insufficient_wing_coverage'] += 1
                elif 'wings entirely missing' in low:
                    stats['portfolio_build_failures']['wings_entirely_missing'] += 1
                else:
                    stats['portfolio_build_failures']['other'] += 1
                stats['checkpoint_details']['4_portfolio_build'].append(current_market)
            
            # Forward/PM gap failures (4A)
            elif 'Checkpoint 4A:' in line and 'DROPPED' in line:
                stats['forward_pm_gap_failures'] += 1
                stats['checkpoint_failures']['4A_forward_pm_gap'] += 1
                stats['checkpoint_details']['4A_forward_pm_gap'].append(current_market)
            
            # Cost recovery failures (5)
            elif 'Checkpoint 5:' in line and ('DROPPED' in line or 'FAILED' in line):
                stats['cost_recovery_failures'] += 1
                stats['checkpoint_failures']['5_cost_recovery'] += 1
                stats['checkpoint_details']['5_cost_recovery'].append(current_market)
            
            # Digital bounds failures (6) — count only if DROPPED
            elif 'Checkpoint 6:' in line and 'DROPPED' in line:
                stats['digital_bounds_failures'] += 1
                stats['checkpoint_failures']['6_digital_bounds'] += 1
                stats['checkpoint_details']['6_digital_bounds'].append(current_market)
            
            # Final result (legacy) + current line produced by strategy
            elif 'Final result:' in line:
                match = re.search(r'(\d+) opportunities returned', line)
                if match:
                    opps = int(match.group(1))
                    stats['final_opportunities'] += opps
                    if opps > 0:
                        stats['markets_with_opportunities'] += 1
            elif 'Total opportunities returning to HedgeOpportunityBuilder' in line:
                match = re.search(r':\s*(\d+)$', line)
                if match:
                    opps = int(match.group(1))
                    stats['final_opportunities'] += opps
                    if opps > 0:
                        stats['markets_with_opportunities'] += 1
    
    return stats


def print_summary(stats: Dict):
    """Print a formatted summary of the checkpoint analysis."""
    
    print("=" * 80)
    print("VARIANCE SWAP CHECKPOINT ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    
    print(f"Total markets evaluated: {stats['total_markets']}")
    print(f"Markets with opportunities returned: {stats['markets_with_opportunities']}")
    print(f"Total opportunities returned: {stats['final_opportunities']}")
    print()
    
    print("FILTERING FUNNEL:")
    print("-" * 60)
    
    # Calculate cumulative drops
    remaining = stats['total_markets']
    
    # 0A & 0B: Early drops
    early_drops = stats['checkpoint_failures'].get('0A_missing_no_price', 0) + \
                  stats['checkpoint_failures'].get('0B_basic_validation', 0)
    if early_drops > 0:
        print(f"0. Early gates (missing data/validation): {early_drops} dropped")
        print(f"   - Missing NO price: {stats['checkpoint_failures'].get('0A_missing_no_price', 0)}")
        print(f"   - Basic validation failed: {stats['checkpoint_failures'].get('0B_basic_validation', 0)}")
        remaining -= early_drops
        print(f"   → {remaining} markets remaining")
        print()
    
    # 1: Price range gate
    price_drops = stats['checkpoint_failures'].get('1_price_gate_both_invalid', 0)
    if price_drops > 0 or stats['price_gate_stats']['yes_invalid'] > 0 or stats['price_gate_stats']['no_invalid'] > 0:
        print(f"1. PM price-range gate:")
        print(f"   - YES prices invalid: {stats['price_gate_stats']['yes_invalid']}")
        print(f"   - NO prices invalid: {stats['price_gate_stats']['no_invalid']}")
        print(f"   - Both invalid (dropped): {price_drops}")
        remaining -= price_drops
        print(f"   → {remaining} markets remaining")
        print()
    
    # 2: Expiry filter
    expiry_drops = stats['checkpoint_failures'].get('2_expiry_filter', 0)
    if expiry_drops > 0:
        print(f"2. Expiry selection filter: {expiry_drops} dropped")
        print(f"   - Markets with zero suitable options: {stats['expiry_filter_stats']['zero_options']}")
        if stats['expiry_filter_stats']['option_reduction']:
            avg_before = sum(x[0] for x in stats['expiry_filter_stats']['option_reduction']) / len(stats['expiry_filter_stats']['option_reduction'])
            avg_after = sum(x[1] for x in stats['expiry_filter_stats']['option_reduction']) / len(stats['expiry_filter_stats']['option_reduction'])
            print(f"   - Average options: {avg_before:.1f} → {avg_after:.1f}")
        remaining -= expiry_drops
        print(f"   → {remaining} markets remaining")
        print()
    
    # 3: Too close expiry (always show)
    too_close_drops = stats['checkpoint_failures'].get('3_too_close_expiry', 0)
    print(f"3. Too close expiry gate: {too_close_drops} dropped")
    if too_close_drops > 0:
        remaining -= too_close_drops
    print(f"   → {remaining} markets remaining")
    print()
    
    # 4: Portfolio build
    portfolio_drops = stats['checkpoint_failures'].get('4_portfolio_build', 0)
    if portfolio_drops > 0:
        print(f"4. Portfolio build gates: {portfolio_drops} dropped")
        for reason, count in stats['portfolio_build_failures'].items():
            if count > 0:
                print(f"   - {reason.replace('_', ' ').title()}: {count}")
        remaining -= portfolio_drops
        print(f"   → {remaining} markets remaining")
        print()
    
    # 4A: Forward/PM gap
    gap_drops = stats['checkpoint_failures'].get('4A_forward_pm_gap', 0)
    if gap_drops > 0:
        print(f"4A. Forward/PM gap check (>5%): {gap_drops} dropped")
        remaining -= gap_drops
        print(f"   → {remaining} markets remaining")
        print()
    
    # 5: Cost recovery
    cost_drops = stats['checkpoint_failures'].get('5_cost_recovery', 0)
    if cost_drops > 0:
        print(f"5. Cost recovery gate: {cost_drops} dropped")
        remaining -= cost_drops
        print(f"   → {remaining} markets remaining")
        print()
    
    # 6: Digital bounds (always show)
    bounds_drops = stats['checkpoint_failures'].get('6_digital_bounds', 0)
    print(f"6. Digital bounds gate: {bounds_drops} dropped")
    if bounds_drops > 0:
        remaining -= bounds_drops
    print(f"   → {remaining} markets remaining")
    print()

    # 7: Successes (always show)
    print(f"7. Opportunities created: {stats['final_opportunities']}")
    print()
    
    print("=" * 80)
    print("SUMMARY:")
    print(f"- Started with: {stats['total_markets']} markets")
    print(f"- Ended with: {stats['markets_with_opportunities']} markets producing opportunities")
    print(f"- Total opportunities: {stats['final_opportunities']}")
    print()
    
    # Show top failure reasons
    print("TOP FAILURE REASONS:")
    sorted_failures = sorted(stats['checkpoint_failures'].items(), key=lambda x: x[1], reverse=True)[:5]
    for checkpoint, count in sorted_failures:
        pct = (count / stats['total_markets']) * 100
        print(f"- {checkpoint}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    log_file = "debug_runs/variance_swap_checkpoints.log"
    stats = parse_checkpoint_log(log_file)
    print_summary(stats)