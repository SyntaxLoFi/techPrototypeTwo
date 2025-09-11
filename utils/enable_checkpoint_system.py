#!/usr/bin/env python3
"""
Script to enable and configure the OptionHedgeBuilder checkpoint system.

This script provides utilities to:
1. Enable/disable the checkpoint system
2. Configure checkpoint settings
3. Analyze existing checkpoint data
4. Clean up old checkpoint files
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.debug_recorder import get_recorder
from utils.method_checkpoint_decorator import save_all_summaries


def enable_checkpoints():
    """Enable the checkpoint system by setting environment variables."""
    os.environ["DEBUG"] = "true"
    os.environ["DEBUG_CAPTURE_CHECKPOINT"] = "true"
    print("✓ Checkpoint system ENABLED")
    print("  - DEBUG=true")
    print("  - DEBUG_CAPTURE_CHECKPOINT=true")


def disable_checkpoints():
    """Disable the checkpoint system."""
    if "DEBUG" in os.environ:
        del os.environ["DEBUG"]
    if "DEBUG_CAPTURE_CHECKPOINT" in os.environ:
        del os.environ["DEBUG_CAPTURE_CHECKPOINT"]
    print("✓ Checkpoint system DISABLED")


def show_status():
    """Show current checkpoint system status."""
    recorder = get_recorder()
    print("Checkpoint System Status")
    print("=" * 30)
    print(f"Enabled: {recorder.enabled}")
    print(f"Run ID: {recorder.run_id}")
    print(f"Base directory: {recorder._base_dir()}")
    print(f"Capture settings: {recorder.capture}")
    
    if recorder.enabled:
        checkpoint_dir = recorder._base_dir() / "OptionsHedgeBuilder_checkpoints"
        if checkpoint_dir.exists():
            instances = list(checkpoint_dir.glob("*"))
            print(f"Active instances: {len(instances)}")
            for instance_dir in instances:
                files = list(instance_dir.glob("*.json"))
                print(f"  - {instance_dir.name}: {len(files)} checkpoint files")
        else:
            print("No checkpoint files found yet")


def analyze_checkpoints(run_id: Optional[str] = None):
    """Analyze checkpoint data for a specific run or the latest run."""
    debug_dir = Path("debug_runs")
    
    if not debug_dir.exists():
        print("No debug_runs directory found")
        return
    
    if run_id:
        target_dir = debug_dir / run_id
        if not target_dir.exists():
            print(f"Run directory {run_id} not found")
            return
    else:
        # Find the latest run
        run_dirs = [d for d in debug_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            print("No run directories found")
            return
        target_dir = max(run_dirs, key=lambda d: d.name)
    
    print(f"Analyzing checkpoint data for run: {target_dir.name}")
    print("=" * 60)
    
    checkpoint_base = target_dir / "OptionsHedgeBuilder_checkpoints"
    if not checkpoint_base.exists():
        print("No OptionsHedgeBuilder_checkpoints directory found")
        return
    
    instances = list(checkpoint_base.glob("*"))
    print(f"Found {len(instances)} OptionHedgeBuilder instances:")
    
    total_calls = 0
    total_duration = 0.0
    method_stats = {}
    
    for instance_dir in instances:
        print(f"\nInstance {instance_dir.name}:")
        
        summary_file = instance_dir / "method_trace_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                
                instance_calls = summary.get("total_calls", 0)
                total_calls += instance_calls
                print(f"  Total calls: {instance_calls}")
                
                method_statistics = summary.get("method_statistics", {})
                for method_name, stats in method_statistics.items():
                    call_count = stats.get("call_count", 0)
                    avg_duration = stats.get("avg_duration", 0.0)
                    exception_count = stats.get("exception_count", 0)
                    total_duration += stats.get("total_duration", 0.0)
                    
                    print(f"    {method_name}: {call_count} calls, "
                          f"avg {avg_duration:.4f}s, {exception_count} exceptions")
                    
                    # Aggregate method stats
                    if method_name not in method_stats:
                        method_stats[method_name] = {
                            "total_calls": 0,
                            "total_duration": 0.0,
                            "total_exceptions": 0
                        }
                    method_stats[method_name]["total_calls"] += call_count
                    method_stats[method_name]["total_duration"] += stats.get("total_duration", 0.0)
                    method_stats[method_name]["total_exceptions"] += exception_count
                
            except Exception as e:
                print(f"  Error reading summary: {e}")
        
        # Count checkpoint files
        checkpoint_files = list(instance_dir.glob("*.json"))
        print(f"  Checkpoint files: {len(checkpoint_files)}")
    
    print(f"\nOverall Statistics:")
    print(f"  Total method calls: {total_calls}")
    print(f"  Total execution time: {total_duration:.4f}s")
    print(f"  Average call duration: {total_duration/total_calls:.4f}s" if total_calls > 0 else "  No calls")
    
    print(f"\nMethod Summary:")
    for method_name, stats in method_stats.items():
        avg_duration = stats["total_duration"] / stats["total_calls"] if stats["total_calls"] > 0 else 0
        print(f"  {method_name}: {stats['total_calls']} calls, "
              f"avg {avg_duration:.4f}s, {stats['total_exceptions']} exceptions")


def list_runs():
    """List all available debug runs."""
    debug_dir = Path("debug_runs")
    if not debug_dir.exists():
        print("No debug_runs directory found")
        return
    
    run_dirs = [d for d in debug_dir.iterdir() if d.is_dir()]
    run_dirs.sort(key=lambda d: d.name, reverse=True)
    
    print(f"Available debug runs ({len(run_dirs)}):")
    for run_dir in run_dirs:
        checkpoint_dir = run_dir / "OptionsHedgeBuilder_checkpoints"
        if checkpoint_dir.exists():
            instances = len(list(checkpoint_dir.glob("*")))
            print(f"  {run_dir.name} - {instances} instances")
        else:
            print(f"  {run_dir.name} - no checkpoint data")


def cleanup_old_runs(days: int = 7):
    """Clean up debug runs older than specified days."""
    debug_dir = Path("debug_runs")
    if not debug_dir.exists():
        print("No debug_runs directory found")
        return
    
    cutoff_date = datetime.now() - timedelta(days=days)
    cleaned_count = 0
    
    for run_dir in debug_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Parse run ID timestamp (format: YYYYMMDD-HHMMSSZ)
        try:
            run_timestamp = datetime.strptime(run_dir.name[:15], "%Y%m%d-%H%M%S")
            if run_timestamp < cutoff_date:
                print(f"Cleaning up old run: {run_dir.name}")
                shutil.rmtree(run_dir)
                cleaned_count += 1
        except ValueError:
            # Skip directories that don't match the expected format
            continue
    
    print(f"Cleaned up {cleaned_count} old runs")


def export_summary(run_id: Optional[str] = None, output_file: Optional[str] = None):
    """Export checkpoint summary to a file."""
    debug_dir = Path("debug_runs")
    
    if run_id:
        target_dir = debug_dir / run_id
    else:
        run_dirs = [d for d in debug_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            print("No run directories found")
            return
        target_dir = max(run_dirs, key=lambda d: d.name)
    
    checkpoint_base = target_dir / "OptionsHedgeBuilder_checkpoints"
    if not checkpoint_base.exists():
        print("No checkpoint data found")
        return
    
    # Collect all summary data
    export_data = {
        "run_id": target_dir.name,
        "export_timestamp": datetime.now().isoformat(),
        "instances": []
    }
    
    for instance_dir in checkpoint_base.glob("*"):
        summary_file = instance_dir / "method_trace_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                export_data["instances"].append(summary)
            except Exception as e:
                print(f"Error reading {summary_file}: {e}")
    
    # Write export file
    if not output_file:
        output_file = f"checkpoint_summary_{target_dir.name}.json"
    
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Exported checkpoint summary to: {output_file}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="OptionHedgeBuilder Checkpoint System Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Enable command
    subparsers.add_parser("enable", help="Enable checkpoint system")
    
    # Disable command
    subparsers.add_parser("disable", help="Disable checkpoint system")
    
    # Status command
    subparsers.add_parser("status", help="Show checkpoint system status")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze checkpoint data")
    analyze_parser.add_argument("--run-id", help="Specific run ID to analyze")
    
    # List command
    subparsers.add_parser("list", help="List all debug runs")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old debug runs")
    cleanup_parser.add_argument("--days", type=int, default=7, help="Keep runs newer than N days")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export checkpoint summary")
    export_parser.add_argument("--run-id", help="Specific run ID to export")
    export_parser.add_argument("--output", help="Output file name")
    
    args = parser.parse_args()
    
    if args.command == "enable":
        enable_checkpoints()
    elif args.command == "disable":
        disable_checkpoints()
    elif args.command == "status":
        show_status()
    elif args.command == "analyze":
        analyze_checkpoints(args.run_id)
    elif args.command == "list":
        list_runs()
    elif args.command == "cleanup":
        cleanup_old_runs(args.days)
    elif args.command == "export":
        export_summary(args.run_id, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()