#!/usr/bin/env python3
import json
import re
import csv

def extract_date_from_title(title):
    """Extract date from market title"""
    # First try specific date patterns
    patterns = [
        (r"by (\w+ \d{1,2}, \d{4})", lambda m: m.group(1)),  # by December 31, 2025
        (r"on (\w+ \d{1,2}, \d{4})", lambda m: m.group(1)),  # on September 18, 2025
        (r"(\w+ \d{1,2}, \d{4})", lambda m: m.group(1)),  # December 31, 2025
        (r"by (\w+ \d{1,2})\?", lambda m: m.group(1)),  # by September 30?
        (r"on (\w+ \d{1,2})\?", lambda m: m.group(1)),  # on September 18?
        (r"on (\w+ \d{1,2}) at", lambda m: m.group(1)),  # on September 17 at 8PM ET
        (r"before (\d{4})", lambda m: m.group(1)),  # before 2026
        (r"in (\d{4})\?", lambda m: m.group(1)),  # in 2025?
    ]
    
    for pattern, extractor in patterns:
        match = re.search(pattern, title)
        if match:
            return extractor(match)
    
    # Then try month-only patterns
    month_patterns = [
        (r"in (\w+)\?", lambda m: m.group(1)),  # in September?
        (r"(\$\d+[Kk]?) in (\w+)", lambda m: m.group(2)),  # $200K in September
        (r"reach.*in (\w+)", lambda m: m.group(1)),  # reach in September
        (r"dip.*in (\w+)", lambda m: m.group(1)),  # dip in September
    ]
    
    for pattern, extractor in month_patterns:
        match = re.search(pattern, title)
        if match:
            return extractor(match)
    
    return ""

# Process validation audit file
validation_data = []
with open('analysis/validation_audit.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        if 'validation_pass' in data and data['validation_pass'] == False:
            validation_data.append(data)

# Create consolidated CSV
output_data = []

# Process each validation failure
for i, fail_data in enumerate(validation_data):
    # Get the previous line to get the market title
    with open('analysis/validation_audit.jsonl', 'r') as f:
        lines = f.readlines()
        for j, line in enumerate(lines):
            current = json.loads(line.strip())
            if current == fail_data and j > 0:
                prev_data = json.loads(lines[j-1].strip())
                title = prev_data.get('pm_question', '')
                date = extract_date_from_title(title)
                
                reason = fail_data.get('reason_code', '')
                yes_price = fail_data.get('yes_price', '')
                
                output_data.append({
                    'Market Title': title,
                    'Date': date,
                    'YES Too High': yes_price if reason == 'YES_TOO_HIGH' else '',
                    'YES Too Low': yes_price if reason == 'YES_TOO_LOW' else '',
                    'Missing Days to Expiry': 'X' if reason == 'MISSING_DAYS_TO_EXPIRY' else ''
                })
                break

# Write consolidated CSV
with open('validation_failures_consolidated.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Market Title', 'Date', 'YES Too High', 'YES Too Low', 'Missing Days to Expiry'])
    writer.writeheader()
    writer.writerows(output_data)

# Also create separate CSVs for each failure type
yes_too_low = [row for row in output_data if row['YES Too Low']]
yes_too_high = [row for row in output_data if row['YES Too High']]
missing_days = [row for row in output_data if row['Missing Days to Expiry']]

# YES Too Low CSV
with open('validation_failures_yes_too_low_formatted.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Market Title', 'Date', 'YES Price'])
    writer.writeheader()
    for row in yes_too_low:
        writer.writerow({
            'Market Title': row['Market Title'],
            'Date': row['Date'],
            'YES Price': row['YES Too Low']
        })

# YES Too High CSV
with open('validation_failures_yes_too_high_formatted.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Market Title', 'Date', 'YES Price'])
    writer.writeheader()
    for row in yes_too_high:
        writer.writerow({
            'Market Title': row['Market Title'],
            'Date': row['Date'],
            'YES Price': row['YES Too High']
        })

# Missing Days CSV
with open('validation_failures_missing_days_formatted.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Market Title', 'Date'])
    writer.writeheader()
    for row in missing_days:
        writer.writerow({
            'Market Title': row['Market Title'],
            'Date': row['Date']
        })

print(f"Created consolidated CSV with {len(output_data)} total failures")
print(f"- YES Too Low: {len(yes_too_low)}")
print(f"- YES Too High: {len(yes_too_high)}")
print(f"- Missing Days to Expiry: {len(missing_days)}")