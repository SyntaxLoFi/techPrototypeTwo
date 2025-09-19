#!/usr/bin/env python3
"""
Analyze variance swap checkpoint logs and general pipeline checkpoints
to understand the filtering funnel.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import sys


def analyze_variance_swap_log(log_path: Path) -> Dict[str, Any]:
    """Analyze the variance swap checkpoint log file."""
    results = {
        "checkpoints": defaultdict(lambda: {"in": 0, "out": 0, "filtered": 0}),
        "filtering_reasons": defaultdict(int),
        "total_processed": 0,
        "total_opportunities": 0
    }
    
    if not log_path.exists():
        return results
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                checkpoint = data.get("checkpoint", "")
                
                if checkpoint == "evaluate_opportunities_start":
                    results["total_processed"] += 1
                    
                elif checkpoint == "opportunity_created":
                    results["total_opportunities"] += 1
                    
                elif checkpoint.startswith("filter_"):
                    # Extract filtering info
                    before = data.get("before_count", 0)
                    after = data.get("after_count", 0)
                    filtered = data.get("filtered_count", before - after)
                    
                    results["checkpoints"][checkpoint]["in"] = before
                    results["checkpoints"][checkpoint]["out"] = after
                    results["checkpoints"][checkpoint]["filtered"] = filtered
                    
                    # Track reasons if available
                    if "reason" in data:
                        results["filtering_reasons"][data["reason"]] += 1
                        
            except json.JSONDecodeError:
                continue
                
    return results


def analyze_pipeline_checkpoints(checkpoint_dir: Path) -> Dict[str, Any]:
    """Analyze the general pipeline checkpoint files."""
    results = {
        "stages": [],
        "opportunity_flow": {}
    }
    
    if not checkpoint_dir.exists():
        return results
    
    # Read all checkpoint files in order
    checkpoint_files = sorted(checkpoint_dir.glob("*.json"))
    
    for cp_file in checkpoint_files:
        if cp_file.name == "summary.json":
            continue
            
        try:
            with open(cp_file, 'r') as f:
                data = json.load(f)
                
            stage_name = data.get("name", "")
            metadata = data.get("metadata", {})
            
            # Extract opportunity counts at different stages
            count = None
            if "count" in metadata:
                count = metadata["count"]
            elif "total_opportunities" in metadata:
                count = metadata["total_opportunities"]
            elif "total_contracts" in metadata:
                count = metadata["total_contracts"]
                
            if stage_name in ["hedge_opportunities_raw", "pre_probability_ranking", 
                             "post_probability_ranking", "pre_ev_filter", 
                             "post_ev_filter", "final_opportunities"]:
                results["opportunity_flow"][stage_name] = {
                    "count": count,
                    "metadata": metadata
                }
                
            results["stages"].append({
                "file": cp_file.name,
                "stage": stage_name,
                "count": count,
                "metadata": metadata
            })
            
        except Exception as e:
            print(f"Error reading {cp_file}: {e}")
            
    return results


def print_funnel_summary(variance_results: Dict, pipeline_results: Dict):
    """Print a clear summary of the filtering funnel."""
    print("\n" + "="*80)
    print("VARIANCE SWAP & PIPELINE FILTERING FUNNEL ANALYSIS")
    print("="*80)
    
    # Variance Swap Strategy Internal Filtering
    if variance_results["checkpoints"]:
        print("\n📊 VARIANCE SWAP STRATEGY INTERNAL FILTERING:")
        print("-"*60)
        print(f"Total PM contracts processed: {variance_results['total_processed']}")
        print(f"Total opportunities created: {variance_results['total_opportunities']}")
        
        for checkpoint, stats in sorted(variance_results["checkpoints"].items()):
            if stats["in"] > 0:
                filter_rate = (stats["filtered"] / stats["in"]) * 100 if stats["in"] > 0 else 0
                print(f"\n  {checkpoint}:")
                print(f"    In:  {stats['in']}")
                print(f"    Out: {stats['out']}")
                print(f"    Filtered: {stats['filtered']} ({filter_rate:.1f}%)")
    
    # Pipeline-Level Filtering
    if pipeline_results["opportunity_flow"]:
        print("\n\n📊 MAIN PIPELINE FILTERING:")
        print("-"*60)
        
        flow = pipeline_results["opportunity_flow"]
        stages = [
            ("hedge_opportunities_raw", "Opportunities from all strategies"),
            ("pre_probability_ranking", "Before probability ranking"),
            ("post_probability_ranking", "After probability ranking"),
            ("pre_ev_filter", "Before EV filter"),
            ("post_ev_filter", "After EV filter"),
            ("final_opportunities", "Final opportunities")
        ]
        
        prev_count = None
        for stage_key, stage_name in stages:
            if stage_key in flow:
                count = flow[stage_key]["count"]
                if count is not None:
                    if prev_count is not None and prev_count > 0:
                        change = count - prev_count
                        pct_change = (change / prev_count) * 100
                        if change < 0:
                            print(f"\n  {stage_name}: {count}")
                            print(f"    ↓ Filtered: {abs(change)} ({abs(pct_change):.1f}% removed)")
                        elif change > 0:
                            print(f"\n  {stage_name}: {count}")
                            print(f"    ↑ Added: {change}")
                        else:
                            print(f"\n  {stage_name}: {count} (no change)")
                    else:
                        print(f"\n  {stage_name}: {count}")
                    prev_count = count
                    
                    # Add extra metadata if available
                    meta = flow[stage_key]["metadata"]
                    if "filtered_out" in meta:
                        print(f"    Explicitly filtered: {meta['filtered_out']}")
                    if "currencies" in meta:
                        print(f"    Currencies: {', '.join(meta['currencies'])}")
                    if "strategies" in meta and meta["strategies"]:
                        print(f"    Strategies: {', '.join(meta['strategies'])}")
    
    # Summary Stats
    print("\n\n📊 SUMMARY:")
    print("-"*60)
    
    # Calculate overall retention rate
    if pipeline_results["opportunity_flow"]:
        initial = flow.get("hedge_opportunities_raw", {}).get("count", 0)
        final = flow.get("final_opportunities", {}).get("count", 0)
        if initial > 0:
            retention_rate = (final / initial) * 100
            print(f"Overall pipeline retention: {final}/{initial} ({retention_rate:.1f}%)")
            print(f"Total filtered out: {initial - final}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main analysis function."""
    # Find the latest debug run
    debug_runs = Path("debug_runs")
    
    # Check for variance swap logs
    variance_log = debug_runs / "variance_swap_checkpoints.log"
    variance_results = analyze_variance_swap_log(variance_log)
    
    # Find latest checkpoint directory
    checkpoint_dirs = sorted([d for d in debug_runs.glob("*/checkpoints") if d.is_dir()])
    
    if checkpoint_dirs:
        latest_checkpoint_dir = checkpoint_dirs[-1]
        print(f"Analyzing checkpoints from: {latest_checkpoint_dir.parent.name}")
        pipeline_results = analyze_pipeline_checkpoints(latest_checkpoint_dir)
    else:
        print("No checkpoint directories found")
        pipeline_results = {"stages": [], "opportunity_flow": {}}
    
    # Print the funnel summary
    print_funnel_summary(variance_results, pipeline_results)
    
    # Additional details if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--verbose":
        print("\n\nDETAILED STAGE INFORMATION:")
        print("-"*60)
        for stage in pipeline_results["stages"]:
            print(f"\n{stage['file']}:")
            print(f"  Stage: {stage['stage']}")
            if stage['count'] is not None:
                print(f"  Count: {stage['count']}")
            if stage['metadata']:
                print(f"  Metadata: {json.dumps(stage['metadata'], indent=4)}")


if __name__ == "__main__":
    main()