from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Any, Optional, Dict, List, Union
from pathlib import Path

from utils.debug_recorder import get_recorder, RawDataRecorder

logger = logging.getLogger(__name__)


class StepDebugger:
    """
    A debugging utility that creates numbered checkpoints throughout the data pipeline.
    Helps identify where data is lost or transformed incorrectly.
    """
    
    def __init__(self, recorder: Optional[RawDataRecorder] = None, enabled: Optional[bool] = None):
        """
        Initialize the step debugger.
        
        Args:
            recorder: Optional RawDataRecorder instance. If None, gets the global recorder.
            enabled: Override the enabled state. If None, uses recorder's state.
        """
        self.recorder = recorder or get_recorder()
        self.enabled = enabled if enabled is not None else self.recorder.enabled
        self.step_counter = 0
        self.checkpoints: List[Dict[str, Any]] = []
        self.drop_reasons: Dict[str, int] = {}
        
    def checkpoint(self, 
                  name: str, 
                  data: Any, 
                  metadata: Optional[Dict[str, Any]] = None,
                  track_fields: Optional[List[str]] = None) -> None:
        """
        Create a numbered checkpoint with the current data state.
        
        Args:
            name: Descriptive name for this checkpoint (e.g., "polymarket_fetch")
            data: The data to capture at this point
            metadata: Optional additional context
            track_fields: Optional list of field names to specifically track for None values
        """
        if not self.enabled:
            return
            
        self.step_counter += 1
        
        # Calculate data statistics
        stats = self._calculate_stats(data, track_fields)
        
        checkpoint_data = {
            "step": self.step_counter,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "metadata": metadata or {},
            "data": data
        }
        
        # Store checkpoint info for summary
        self.checkpoints.append({
            "step": self.step_counter,
            "name": name,
            "stats": stats,
            "metadata": metadata or {}
        })
        
        # Save the checkpoint
        filename = f"checkpoints/{self.step_counter:03d}_{name}.json"
        try:
            self.recorder.dump_json(filename, checkpoint_data, category="checkpoint")
            logger.debug(f"Checkpoint {self.step_counter}: {name} - {stats.get('count', 0)} items")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint {name}: {e}")
    
    def track_drop(self, reason: str, item: Optional[Dict[str, Any]] = None) -> None:
        """
        Track why an item was dropped/filtered out.
        
        Args:
            reason: The reason for dropping (e.g., "DROP_EV_TOO_LOW")
            item: Optional item that was dropped for detailed logging
        """
        if not self.enabled:
            return
            
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1
        
        # Optionally log specific problematic items
        if item and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Dropped item - {reason}: {item.get('currency', '?')} - {item.get('question', '?')[:50]}")
    
    def save_summary(self) -> None:
        """Save a summary of the entire data flow."""
        if not self.enabled or not self.checkpoints:
            return
            
        summary = {
            "total_steps": self.step_counter,
            "checkpoints": self.checkpoints,
            "drop_reasons": self.drop_reasons,
            "data_flow": self._generate_flow_summary()
        }
        
        try:
            self.recorder.dump_json("checkpoints/summary.json", summary, category="checkpoint")
            logger.info(f"Saved debugging summary with {self.step_counter} checkpoints")
        except Exception as e:
            logger.warning(f"Failed to save summary: {e}")
    
    def _calculate_stats(self, data: Any, track_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate statistics about the data."""
        stats: Dict[str, Any] = {"type": type(data).__name__}
        
        if hasattr(data, '__len__'):
            stats["count"] = len(data)
            
        if isinstance(data, list) and data:
            # Sample the first item for structure
            first_item = data[0]
            if isinstance(first_item, dict):
                stats["sample_keys"] = list(first_item.keys())
                
                # Track None fields if requested
                if track_fields:
                    none_counts = {}
                    for field in track_fields:
                        none_count = sum(1 for item in data if isinstance(item, dict) and item.get(field) is None)
                        if none_count > 0:
                            none_counts[field] = none_count
                    if none_counts:
                        stats["none_fields"] = none_counts
                        
        elif isinstance(data, dict):
            stats["count"] = len(data)
            stats["keys"] = list(data.keys())
            
            # For scanner data, count contracts per currency
            if all(isinstance(v, dict) and "contracts" in v for v in data.values()):
                contract_counts = {k: len(v.get("contracts", [])) for k, v in data.items()}
                stats["contracts_per_currency"] = contract_counts
                
        return stats
    
    def _generate_flow_summary(self) -> List[str]:
        """Generate a human-readable summary of the data flow."""
        flow = []
        for i, cp in enumerate(self.checkpoints):
            count = cp["stats"].get("count", "?")
            flow.append(f"Step {cp['step']}: {cp['name']} ({count} items)")
            
            # Show data reduction between steps
            if i > 0:
                prev_count = self.checkpoints[i-1]["stats"].get("count", 0)
                curr_count = cp["stats"].get("count", 0)
                if isinstance(prev_count, int) and isinstance(curr_count, int) and curr_count < prev_count:
                    reduction = prev_count - curr_count
                    pct = (reduction / prev_count * 100) if prev_count > 0 else 0
                    flow.append(f"  → Reduced by {reduction} items ({pct:.1f}%)")
                    
        return flow


# Global instance for easy access
_step_debugger: Optional[StepDebugger] = None


def get_step_debugger(recorder: Optional[RawDataRecorder] = None) -> StepDebugger:
    """Get or create the global step debugger instance."""
    global _step_debugger
    if _step_debugger is None:
        _step_debugger = StepDebugger(recorder)
    return _step_debugger