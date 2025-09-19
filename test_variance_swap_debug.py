#!/usr/bin/env python3
"""Test script to run variance swap with enhanced expiry filter debugging."""

import logging
import sys
import os

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_runs/variance_swap_debug_detailed.log')
    ]
)

# Run main scanner with DEBUG enabled
os.environ['DEBUG'] = 'true'

print("Running main scanner with enhanced expiry filter debugging...")
print("Look for [EXPIRY_FILTER_DEBUG] messages in the output")
print("Detailed stats will be saved to debug_runs/expiry_filter_debug.jsonl")
print("-" * 80)

# Import and run main scanner
from main_scanner import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError running scanner: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-" * 80)
    print("Run complete. Check the following files:")
    print("  - debug_runs/variance_swap_debug_detailed.log")
    print("  - debug_runs/expiry_filter_debug.jsonl")
    print("  - debug_runs/variance_swap_checkpoints.log")