#!/usr/bin/env python3
"""
Verify opportunity data structure is consistent throughout the pipeline
"""

import json
import os
from typing import Dict, List


def verify_opportunity_structure(opp: Dict) -> List[str]:
    """Verify an opportunity has all required fields"""
    errors = []
    
    # Core fields that must exist
    required_fields = [
        'strategy',
        'max_profit', 
        'max_loss',
        'currency',
        'hedge_type'
    ]
    
    for field in required_fields:
        if field not in opp:
            errors.append(f"Missing required field: {field}")
    
    # Check polymarket data (either format)
    if 'polymarket' not in opp and 'polymarket_contract' not in opp:
        errors.append("Missing both 'polymarket' and 'polymarket_contract'")
    
    pm_data = opp.get('polymarket', opp.get('polymarket_contract', {}))
    if not pm_data.get('question'):
        errors.append("Missing polymarket question")
    
    # Check strike price (either format)
    strike = pm_data.get('strike', pm_data.get('strike_price', 0))
    if strike == 0:
        errors.append("Missing or zero strike price")
    
    # Check metrics (either format) 
    metrics = opp.get('metrics', opp.get('probability_metrics', {}))
    if not metrics:
        errors.append("Missing both 'metrics' and 'probability_metrics'")
    
    # For ranked opportunities
    if 'rank' in opp and opp['rank'] <= 0:
        errors.append(f"Invalid rank: {opp['rank']}")
    
    return errors


def check_saved_files():
    """Check all saved opportunity files for consistency"""
    print("\n📂 Checking Saved Opportunity Files...")
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print("❌ No results directory found")
        return
    
    # Find all opportunity files
    files_checked = 0
    total_opportunities = 0
    issues_found = 0
    
    for filename in os.listdir(results_dir):
        if 'opportunities' in filename and filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            files_checked += 1
            
            print(f"\nChecking: {filename}")
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                opportunities = data.get('opportunities', [])
                print(f"  - Opportunities: {len(opportunities)}")
                total_opportunities += len(opportunities)
                
                # Check each opportunity
                file_issues = 0
                for i, opp in enumerate(opportunities[:5]):  # Check first 5
                    errors = verify_opportunity_structure(opp)
                    if errors:
                        file_issues += len(errors)
                        print(f"  - Opportunity #{i+1} issues: {errors}")
                
                if file_issues == 0:
                    print(f"  ✅ Structure valid")
                else:
                    print(f"  ⚠️  {file_issues} issues found")
                    issues_found += file_issues
                    
            except Exception as e:
                print(f"  ❌ Error reading file: {e}")
    
    print(f"\n📊 Summary:")
    print(f"  - Files checked: {files_checked}")
    print(f"  - Total opportunities: {total_opportunities}")
    print(f"  - Issues found: {issues_found}")
    
    return issues_found == 0


def verify_streamlit_compatibility():
    """Verify data works with Streamlit's expected structure"""
    print("\n🎨 Verifying Streamlit Compatibility...")
    
    # Test accessing data the way Streamlit does
    test_cases = [
        {
            'name': 'Old format',
            'data': {
                'probability_metrics': {'prob_of_profit': 0.75, 'expected_value': 100},
                'polymarket': {'question': 'Test?', 'strike': 50000, 'yes_price': 0.6}
            }
        },
        {
            'name': 'New format', 
            'data': {
                'metrics': {'prob_of_profit': 0.75, 'expected_value': 100},
                'polymarket_contract': {'question': 'Test?', 'strike_price': 50000, 'yes_price': 0.6}
            }
        },
        {
            'name': 'Mixed format',
            'data': {
                'metrics': {'prob_of_profit': 0.75},
                'probability_metrics': {'expected_value': 100},
                'polymarket': {'question': 'Test?', 'strike': 50000, 'yes_price': 0.6}
            }
        }
    ]
    
    all_passed = True
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        opp = test['data']
        
        try:
            # Dashboard access pattern
            prob = (opp.get('metrics', {}).get('prob_of_profit', 0) or 
                   opp.get('probability_metrics', {}).get('prob_of_profit', 0))
            assert prob == 0.75, f"Wrong probability: {prob}"
            
            # Strategy Analyzer pattern
            pm = opp.get('polymarket', opp.get('polymarket_contract', {}))
            question = pm.get('question', 'Unknown')
            assert question == 'Test?', f"Wrong question: {question}"
            
            strike = pm.get('strike', pm.get('strike_price', 0))
            assert strike == 50000, f"Wrong strike: {strike}"
            
            print(f"  ✅ Passed")
            
        except AssertionError as e:
            print(f"  ❌ Failed: {e}")
            all_passed = False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_passed = False
    
    return all_passed


def main():
    print("="*80)
    print("🔍 Data Structure Verification")
    print("="*80)
    
    # Check saved files
    files_ok = check_saved_files()
    
    # Check Streamlit compatibility
    streamlit_ok = verify_streamlit_compatibility()
    
    print("\n" + "="*80)
    if files_ok and streamlit_ok:
        print("✅ All data structure checks PASSED!")
    else:
        print("⚠️  Some issues found - review above")
    print("="*80)


if __name__ == "__main__":
    main()