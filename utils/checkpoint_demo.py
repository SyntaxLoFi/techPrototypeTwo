#!/usr/bin/env python3
"""
Demo script to showcase the method checkpoint decorator system.
"""
from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.method_checkpoint_decorator import checkpoint_all_methods, save_all_summaries
from utils.debug_recorder import get_recorder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@checkpoint_all_methods
class DemoClass:
    """Demo class to show method checkpointing functionality."""
    
    def __init__(self, name: str):
        self.name = name
        self.data = []
    
    def add_item(self, item: Any) -> int:
        """Add an item to the data list."""
        self.data.append(item)
        return len(self.data)
    
    def process_items(self, filter_func=None) -> List[Any]:
        """Process items with optional filtering."""
        if filter_func:
            result = [item for item in self.data if filter_func(item)]
        else:
            result = self.data.copy()
        
        # Call another method to demonstrate nesting
        self._sort_items(result)
        return result
    
    def _sort_items(self, items: List[Any]) -> None:
        """Internal method to sort items."""
        if all(isinstance(item, (int, float, str)) for item in items):
            items.sort()
    
    def failing_method(self) -> str:
        """Method that raises an exception for testing."""
        raise ValueError("This is a test exception")
    
    def complex_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Method with complex input/output for serialization testing."""
        result = {
            "input_keys": list(data.keys()),
            "processed_data": {},
            "metadata": {
                "timestamp": "2025-09-10",
                "processor": self.name
            }
        }
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                result["processed_data"][key] = value * 2
            elif isinstance(value, str):
                result["processed_data"][key] = value.upper()
            else:
                result["processed_data"][key] = str(value)
        
        return result


def demo_checkpoint_system():
    """Demonstrate the checkpoint system functionality."""
    print("=" * 60)
    print("Method Checkpoint Decorator Demo")
    print("=" * 60)
    
    # Enable debug recording
    os.environ["DEBUG"] = "true"
    recorder = get_recorder()
    
    if not recorder.enabled:
        print("Warning: Debug recorder is not enabled!")
        print("Set DEBUG=true environment variable to enable checkpoints")
        return
    
    print(f"Debug recording enabled: {recorder.enabled}")
    print(f"Debug directory: {recorder._base_dir()}")
    print()
    
    # Create demo instance
    demo = DemoClass("TestProcessor")
    
    print("1. Testing basic method calls...")
    demo.add_item("hello")
    demo.add_item(42)
    demo.add_item(3.14)
    
    print("2. Testing nested method calls...")
    filtered_items = demo.process_items(lambda x: isinstance(x, (int, float)))
    print(f"Filtered items: {filtered_items}")
    
    print("3. Testing complex data serialization...")
    complex_data = {
        "string_value": "test",
        "number_value": 123,
        "float_value": 45.67,
        "list_value": [1, 2, 3],
        "nested_dict": {"inner_key": "inner_value"}
    }
    result = demo.complex_method(complex_data)
    print(f"Complex method result keys: {list(result.keys())}")
    
    print("4. Testing exception handling...")
    try:
        demo.failing_method()
    except ValueError as e:
        print(f"Caught expected exception: {e}")
    
    print("5. Saving method trace summary...")
    demo.save_method_trace_summary()
    
    print("\n6. Examining checkpoint files...")
    checkpoint_dir = recorder._base_dir() / "OptionsHedgeBuilder_checkpoints"
    if checkpoint_dir.exists():
        instance_dirs = list(checkpoint_dir.glob("*"))
        for instance_dir in instance_dirs:
            print(f"\nInstance directory: {instance_dir.name}")
            checkpoint_files = list(instance_dir.glob("*.json"))
            print(f"Checkpoint files ({len(checkpoint_files)}):")
            for file in sorted(checkpoint_files):
                print(f"  - {file.name}")
                
                # Show sample content
                try:
                    with open(file) as f:
                        data = json.load(f)
                    if "trace" in data:
                        trace = data["trace"]
                        print(f"    Method: {trace.get('method_name')}")
                        print(f"    Duration: {trace.get('duration', 'N/A')} seconds")
                        if trace.get('exception'):
                            print(f"    Exception: {trace['exception']['type']}")
                except Exception as e:
                    print(f"    Error reading file: {e}")
    else:
        print("No checkpoint directory found!")
    
    print("\nDemo completed!")
    
    # Clean up
    save_all_summaries()


def analyze_checkpoint_structure():
    """Analyze the structure of created checkpoint files."""
    print("\n" + "=" * 60)
    print("Checkpoint Structure Analysis")
    print("=" * 60)
    
    recorder = get_recorder()
    base_dir = recorder._base_dir()
    
    # Find all checkpoint files
    checkpoint_files = list(base_dir.rglob("*checkpoint*.json"))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files in {base_dir}")
    
    # Analyze file types
    file_types = {}
    for file in checkpoint_files:
        if "summary" in file.name:
            file_type = "summary"
        elif "start" in file.name:
            file_type = "start"
        elif "end" in file.name:
            file_type = "end"
        else:
            file_type = "other"
        
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    print("\nFile type distribution:")
    for file_type, count in file_types.items():
        print(f"  {file_type}: {count} files")
    
    # Analyze a sample summary file
    summary_files = [f for f in checkpoint_files if "summary" in f.name]
    if summary_files:
        print(f"\nAnalyzing summary file: {summary_files[0].name}")
        try:
            with open(summary_files[0]) as f:
                summary = json.load(f)
            
            print(f"  Class: {summary.get('class_name')}")
            print(f"  Total calls: {summary.get('total_calls')}")
            print(f"  Active calls: {summary.get('active_calls')}")
            
            if "method_statistics" in summary:
                stats = summary["method_statistics"]
                print(f"  Methods tracked: {len(stats)}")
                for method_name, method_stats in stats.items():
                    print(f"    {method_name}: {method_stats.get('call_count')} calls, "
                          f"avg duration: {method_stats.get('avg_duration', 0):.4f}s")
            
            if "call_hierarchy" in summary:
                hierarchy = summary["call_hierarchy"]
                root_calls = hierarchy.get("root_calls", [])
                print(f"  Root calls: {len(root_calls)}")
                
        except Exception as e:
            print(f"Error analyzing summary file: {e}")


if __name__ == "__main__":
    demo_checkpoint_system()
    analyze_checkpoint_structure()