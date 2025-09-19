#!/usr/bin/env python3
import json
import csv
from collections import defaultdict
import re

def extract_date_from_title(title):
    """Extract date from market title"""
    patterns = [
        (r"by (\w+ \d{1,2}, \d{4})", lambda m: m.group(1)),
        (r"on (\w+ \d{1,2}, \d{4})", lambda m: m.group(1)),
        (r"(\w+ \d{1,2}, \d{4})", lambda m: m.group(1)),
        (r"by (\w+ \d{1,2})\?", lambda m: m.group(1)),
        (r"on (\w+ \d{1,2})\?", lambda m: m.group(1)),
        (r"on (\w+ \d{1,2}) at", lambda m: m.group(1)),
        (r"before (\d{4})", lambda m: m.group(1)),
        (r"in (\d{4})\?", lambda m: m.group(1)),
        (r"in (\w+)\?", lambda m: m.group(1)),
        (r"(\$\d+[Kk]?) in (\w+)", lambda m: m.group(2)),
        (r"reach.*in (\w+)", lambda m: m.group(1)),
        (r"dip.*in (\w+)", lambda m: m.group(1)),
    ]
    
    for pattern, extractor in patterns:
        match = re.search(pattern, title)
        if match:
            return extractor(match)
    return ""

# Track all stages
stage_counts = defaultdict(int)
validation_failures = defaultdict(list)
market_titles = {}
all_failures = []

# Read and process the audit file
with open('analysis/validation_audit.jsonl', 'r') as f:
    lines = f.readlines()
    
for i, line in enumerate(lines):
    data = json.loads(line.strip())
    
    # Track pre_strategy entries (markets evaluated)
    if data.get('stage') == 'pre_strategy':
        stage_counts['markets_evaluated'] += 1
        market_titles[data.get('pm_question', '')] = data
        
    # Track validation passes
    elif data.get('stage') == 'validate_inputs':
        if data.get('validation_pass') == True:
            stage_counts['validation_passed'] += 1
        else:
            stage_counts['validation_failed'] += 1
            reason = data.get('reason_code', 'UNKNOWN')
            
            # Get the market info from previous line
            if i > 0:
                prev_data = json.loads(lines[i-1].strip())
                if prev_data.get('stage') == 'pre_strategy':
                    title = prev_data.get('pm_question', '')
                    date = extract_date_from_title(title)
                    currency = prev_data.get('pm_ticker', '')
                    
                    failure_info = {
                        'Market Title': title,
                        'Date': date,
                        'Currency': currency,
                        'Reason': reason,
                        'YES Price': '',
                        'NO Price': '',
                        'YES Too High': '',
                        'YES Too Low': '',
                        'NO Too High': '',
                        'NO Too Low': '',
                        'Missing Field': ''
                    }
                    
                    # Fill in specific failure details
                    if reason == 'YES_TOO_LOW':
                        failure_info['YES Price'] = data.get('yes_price', '')
                        failure_info['YES Too Low'] = data.get('yes_price', '')
                    elif reason == 'YES_TOO_HIGH':
                        failure_info['YES Price'] = data.get('yes_price', '')
                        failure_info['YES Too High'] = data.get('yes_price', '')
                    elif reason == 'NO_TOO_LOW':
                        failure_info['NO Price'] = data.get('no_price', '')
                        failure_info['NO Too Low'] = data.get('no_price', '')
                    elif reason == 'NO_TOO_HIGH':
                        failure_info['NO Price'] = data.get('no_price', '')
                        failure_info['NO Too High'] = data.get('no_price', '')
                    elif reason.startswith('MISSING_'):
                        failure_info['Missing Field'] = reason.replace('MISSING_', '')
                    
                    validation_failures[reason].append(failure_info)
                    all_failures.append(failure_info)

# Write comprehensive CSV with all failures
with open('all_validation_failures.csv', 'w', newline='') as f:
    fieldnames = ['Market Title', 'Date', 'Currency', 'Reason', 'YES Price', 'NO Price', 
                  'YES Too High', 'YES Too Low', 'NO Too High', 'NO Too Low', 'Missing Field']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_failures)

# Create summary by reason
print("\n=== VALIDATION FAILURE SUMMARY ===")
print(f"Total markets evaluated: {stage_counts['markets_evaluated']}")
print(f"Total validation passed: {stage_counts['validation_passed']}")
print(f"Total validation failed: {stage_counts['validation_failed']}")
print("\nFailures by reason:")
for reason, failures in sorted(validation_failures.items()):
    print(f"  {reason}: {len(failures)}")

# Check for NO price validations
no_failures = [f for f in all_failures if f['Reason'] in ['NO_TOO_HIGH', 'NO_TOO_LOW']]
print(f"\nNO price failures found: {len(no_failures)}")

# Create filtering funnel
print("\n=== FILTERING FUNNEL ===")
print(f"1. Markets evaluated: {stage_counts['markets_evaluated']}")
print(f"2. After validation: {stage_counts['validation_passed']} passed ({stage_counts['validation_passed']/stage_counts['markets_evaluated']*100:.1f}%)")
print(f"   - YES_TOO_LOW: {len(validation_failures['YES_TOO_LOW'])}")
print(f"   - YES_TOO_HIGH: {len(validation_failures['YES_TOO_HIGH'])}")
print(f"   - NO_TOO_LOW: {len(validation_failures['NO_TOO_LOW'])}")
print(f"   - NO_TOO_HIGH: {len(validation_failures['NO_TOO_HIGH'])}")
print(f"   - MISSING_DAYS_TO_EXPIRY: {len(validation_failures['MISSING_DAYS_TO_EXPIRY'])}")
print(f"   - Other failures: {stage_counts['validation_failed'] - sum(len(v) for v in validation_failures.values())}")

# Write funnel summary to CSV
with open('validation_funnel_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Stage', 'Count', 'Percentage', 'Lost'])
    
    total = stage_counts['markets_evaluated']
    writer.writerow(['Markets Evaluated', total, '100.0%', '-'])
    
    # Validation stage
    passed = stage_counts['validation_passed']
    lost_validation = stage_counts['validation_failed']
    writer.writerow(['Passed Validation', passed, f'{passed/total*100:.1f}%', f'-{lost_validation}'])
    
    # Breakdown of validation failures
    writer.writerow(['', '', '', ''])
    writer.writerow(['Validation Failure Reasons', 'Count', '% of Total', '% of Failures'])
    for reason, failures in sorted(validation_failures.items()):
        count = len(failures)
        writer.writerow([f'  {reason}', count, f'{count/total*100:.1f}%', f'{count/lost_validation*100:.1f}%'])

print(f"\nFiles created:")
print("  - all_validation_failures.csv (detailed breakdown)")
print("  - validation_funnel_summary.csv (summary statistics)")