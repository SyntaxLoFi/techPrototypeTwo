"""
Method checkpoint decorator system for comprehensive debugging and tracing.

This module provides decorators and utilities to checkpoint all method calls
with inputs, outputs, timestamps, and exception handling.
"""
from __future__ import annotations

import os
import json
import time
import logging
import functools
import traceback
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Type
from collections import defaultdict

from utils.debug_recorder import get_recorder, RawDataRecorder

logger = logging.getLogger(__name__)


class MethodCallTrace:
    """Represents a single method call with all relevant information."""
    
    def __init__(self, method_name: str, instance_id: str, call_id: str):
        self.method_name = method_name
        self.instance_id = instance_id
        self.call_id = call_id
        self.start_time = time.time()
        self.start_timestamp = datetime.now(timezone.utc).isoformat()
        self.end_time: Optional[float] = None
        self.end_timestamp: Optional[str] = None
        self.duration: Optional[float] = None
        self.args: List[Any] = []
        self.kwargs: Dict[str, Any] = {}
        self.result: Any = None
        self.exception: Optional[Dict[str, Any]] = None
        self.nested_calls: List[str] = []
        self.parent_call_id: Optional[str] = None
        
    def finish(self, result: Any = None, exception: Optional[Exception] = None):
        """Mark the call as finished with result or exception."""
        self.end_time = time.time()
        self.end_timestamp = datetime.now(timezone.utc).isoformat()
        self.duration = self.end_time - self.start_time
        
        if exception:
            self.exception = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        else:
            self.result = result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method_name": self.method_name,
            "instance_id": self.instance_id,
            "call_id": self.call_id,
            "start_time": self.start_time,
            "start_timestamp": self.start_timestamp,
            "end_time": self.end_time,
            "end_timestamp": self.end_timestamp,
            "duration": self.duration,
            "args": self._serialize_data(self.args),
            "kwargs": self._serialize_data(self.kwargs),
            "result": self._serialize_data(self.result),
            "exception": self.exception,
            "nested_calls": self.nested_calls,
            "parent_call_id": self.parent_call_id
        }
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON storage, handling complex types gracefully."""
        try:
            # Try direct JSON serialization first
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            return self._deep_serialize(data)
    
    def _deep_serialize(self, obj: Any, max_depth: int = 3, current_depth: int = 0) -> Any:
        """Deeply serialize objects with depth limiting to prevent infinite recursion."""
        if current_depth >= max_depth:
            return f"<truncated at depth {max_depth}: {type(obj).__name__}>"
        
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        
        if isinstance(obj, (list, tuple)):
            try:
                return [self._deep_serialize(item, max_depth, current_depth + 1) 
                       for item in obj[:10]]  # Limit to first 10 items
            except Exception:
                return f"<list/tuple with {len(obj)} items>"
        
        if isinstance(obj, dict):
            try:
                result = {}
                for k, v in list(obj.items())[:20]:  # Limit to first 20 items
                    if isinstance(k, str):
                        result[k] = self._deep_serialize(v, max_depth, current_depth + 1)
                    else:
                        result[str(k)] = self._deep_serialize(v, max_depth, current_depth + 1)
                return result
            except Exception:
                return f"<dict with {len(obj)} items>"
        
        # Handle common objects
        if hasattr(obj, '__dict__'):
            try:
                return {
                    "__type__": type(obj).__name__,
                    "__module__": getattr(type(obj), '__module__', 'unknown'),
                    "attributes": self._deep_serialize(
                        {k: v for k, v in obj.__dict__.items() 
                         if not k.startswith('_')}, 
                        max_depth, current_depth + 1
                    )
                }
            except Exception:
                pass
        
        # Fallback to string representation
        try:
            return str(obj)
        except Exception:
            return f"<unserializable: {type(obj).__name__}>"


class MethodCheckpointer:
    """Manages method call checkpointing for a class instance."""
    
    def __init__(self, class_name: str, instance_id: str, 
                 recorder: Optional[RawDataRecorder] = None):
        self.class_name = class_name
        self.instance_id = instance_id
        self.recorder = recorder or get_recorder()
        self.enabled = self.recorder.enabled
        
        # Thread-safe storage
        # Use RLock because method instrumentation can lead to nested acquisitions
        # (e.g., end_call() calling a helper that briefly grabs the same lock).
        self._lock = threading.RLock()
        self._call_counter = 0
        self._active_calls: Dict[str, MethodCallTrace] = {}
        self._completed_calls: List[MethodCallTrace] = []
        self._call_stack: List[str] = []  # Track nested calls
        
        # Global sequential counter for simplified numbering
        self._global_counter = 0
        
        # Method statistics
        self._method_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "call_count": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "exception_count": 0,
            "last_called": None
        })
    
    def start_call(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> str:
        """Start tracking a method call."""
        if not self.enabled:
            return ""
        
        with self._lock:
            self._call_counter += 1
            call_id = f"{self.instance_id}_{method_name}_{self._call_counter}"
            
            trace = MethodCallTrace(method_name, self.instance_id, call_id)
            trace.args = args
            trace.kwargs = kwargs
            
            # Set parent call if we're nested
            if self._call_stack:
                trace.parent_call_id = self._call_stack[-1]
                # Add to parent's nested calls
                if trace.parent_call_id in self._active_calls:
                    self._active_calls[trace.parent_call_id].nested_calls.append(call_id)
            
            self._active_calls[call_id] = trace
            self._call_stack.append(call_id)
            
            # Don't save start checkpoint - we'll save combined at the end
            
        return call_id
    
    def end_call(self, call_id: str, result: Any = None, exception: Optional[Exception] = None):
        """End tracking a method call."""
        if not self.enabled or not call_id:
            return
        
        # Decide whether to save a checkpoint while holding the lock,
        # but perform any heavy serialization / file I/O *after* releasing it.
        trace_to_save: Optional[MethodCallTrace] = None
        
        with self._lock:
            if call_id not in self._active_calls:
                return
            
            trace = self._active_calls.pop(call_id)
            trace.finish(result, exception)
            self._completed_calls.append(trace)
            
            # Remove from call stack
            if call_id in self._call_stack:
                self._call_stack.remove(call_id)
            
            # Update statistics
            stats = self._method_stats[trace.method_name]
            stats["call_count"] += 1
            if trace.duration:
                stats["total_duration"] += trace.duration
                stats["avg_duration"] = stats["total_duration"] / stats["call_count"]
            if exception:
                stats["exception_count"] += 1
            stats["last_called"] = trace.end_timestamp
            
            # For OptionHedgeBuilder, always save checkpoints to trace data flow
            if self.class_name == "OptionHedgeBuilder":
                trace_to_save = trace
            else:
                # For other classes, save checkpoint only if it contains meaningful data
                has_meaningful_data = self._has_meaningful_data(trace)
                if trace.method_name == "build" or has_meaningful_data:
                    trace_to_save = trace
        
        # Perform I/O outside the lock
        if trace_to_save is not None:
            self._save_combined_checkpoint(trace_to_save)
        else:
            logger.debug(
                f"Skipping checkpoint for {getattr(trace, 'method_name', '?')} - no meaningful options data detected"
            )
    
    def _has_meaningful_data(self, trace: MethodCallTrace) -> bool:
        """Check if the method call contains meaningful data, especially options-related."""
        # Always include methods that deal with options explicitly
        options_methods = ['_construct_options_hedge', 'build', '_transform_options_to_chain', 
                          '_select_best_expiry', '_get_quote', '_nearest_vertical', 
                          '_pm_has_liquidity', '_build_pm_pairs', '_calculate_liquidity_based_position_size']
        if trace.method_name in options_methods:
            logger.debug(f"Method {trace.method_name} is in options_methods list - will save checkpoint")
            return True
            
        # Check if args or kwargs contain options data
        def contains_options(obj: Any, depth: int = 0, seen: Optional[set] = None) -> bool:
            if depth > 3:  # Limit recursion depth
                return False
            if seen is None:
                seen = set()
            
            # Handle circular references
            obj_id = id(obj)
            if obj_id in seen:
                return False
            seen.add(obj_id)
            
            try:
                if isinstance(obj, str):
                    return 'option' in obj.lower()
                elif isinstance(obj, dict):
                    # Check for options-related keys
                    for key in ['options', 'option', 'options_collector', 'all_options', 
                               'option_type', 'strike', 'expiry', 'call', 'put']:
                        if key in obj:
                            return True
                    # Recursively check values with depth limit
                    return any(contains_options(v, depth + 1, seen) for v in obj.values())
                elif isinstance(obj, (list, tuple)):
                    # Limit to first 10 items for performance
                    return any(contains_options(item, depth + 1, seen) for item in obj[:10])
            except Exception:
                # Catch any errors during inspection
                pass
            
            return False
            
        # Check inputs
        if contains_options(trace.args) or contains_options(trace.kwargs):
            return True
            
        # Check outputs
        if contains_options(trace.result):
            return True
            
        # Skip if no meaningful data
        return False
    
    def _save_combined_checkpoint(self, trace: MethodCallTrace):
        """Save a single checkpoint file with both inputs and outputs."""
        try:
            with self._lock:
                self._global_counter += 1
                
            checkpoint_dir = f"OptionsHedgeBuilder_checkpoints"
            # Format with 3-digit zero padding for proper sorting
            filename = f"{checkpoint_dir}/{self._global_counter:03d}_{trace.method_name}.json"
            
            logger.debug(f"Saving checkpoint {self._global_counter:03d} for {trace.method_name}")
            
            # Use the trace's to_dict() method which already handles serialization
            trace_dict = trace.to_dict()
            
            checkpoint_data = {
                "sequence": self._global_counter,
                "method_name": trace.method_name,
                "timestamp": trace.start_timestamp,
                "duration_ms": trace.duration * 1000 if trace.duration else None,
                "inputs": {
                    "args": trace_dict.get("args", []),
                    "kwargs": trace_dict.get("kwargs", {})
                },
                "output": {
                    "result": trace_dict.get("result"),
                    "exception": trace_dict.get("exception")
                },
                "call_hierarchy": self._get_current_hierarchy(),
                "instance_id": self.instance_id
            }
            
            self.recorder.dump_json(filename, checkpoint_data, category="checkpoint")
            
        except Exception as e:
            logger.error(f"Failed to save method checkpoint: {e}", exc_info=True)
    
    def _get_current_hierarchy(self) -> List[Dict[str, str]]:
        """Get the current call hierarchy."""
        hierarchy = []
        for call_id in self._call_stack:
            if call_id in self._active_calls:
                trace = self._active_calls[call_id]
                hierarchy.append({
                    "call_id": call_id,
                    "method_name": trace.method_name,
                    "start_time": trace.start_timestamp
                })
        return hierarchy
    
    def save_summary(self):
        """Save a summary of all method calls."""
        if not self.enabled:
            return
        
        try:
            summary = {
                "class_name": self.class_name,
                "instance_id": self.instance_id,
                "total_calls": len(self._completed_calls),
                "active_calls": len(self._active_calls),
                "method_statistics": dict(self._method_stats),
                "call_trace": [trace.to_dict() for trace in self._completed_calls],
                "call_hierarchy": self._build_call_hierarchy()
            }
            
            filename = f"OptionsHedgeBuilder_checkpoints/method_trace_summary_{self.instance_id}.json"
            self.recorder.dump_json(filename, summary, category="checkpoint")
            
            logger.info(f"Saved method trace summary for {self.class_name} instance {self.instance_id}")
            
        except Exception as e:
            logger.warning(f"Failed to save method trace summary: {e}")
    
    def _build_call_hierarchy(self) -> Dict[str, Any]:
        """Build a hierarchical view of method calls."""
        hierarchy = {}
        root_calls = []
        
        for trace in self._completed_calls:
            if not trace.parent_call_id:
                root_calls.append(trace.call_id)
        
        def build_tree(call_id: str) -> Dict[str, Any]:
            trace = next((t for t in self._completed_calls if t.call_id == call_id), None)
            if not trace:
                return {}
            
            node = {
                "method_name": trace.method_name,
                "duration": trace.duration,
                "exception": trace.exception is not None,
                "children": []
            }
            
            for nested_call_id in trace.nested_calls:
                child_node = build_tree(nested_call_id)
                if child_node:
                    node["children"].append(child_node)
            
            return node
        
        hierarchy["root_calls"] = [build_tree(call_id) for call_id in root_calls]
        return hierarchy


# Global registry of checkpointers
_checkpointers: Dict[str, MethodCheckpointer] = {}
_checkpointer_lock = threading.RLock()


def get_method_checkpointer(instance: Any, class_name: Optional[str] = None) -> MethodCheckpointer:
    """Get or create a method checkpointer for an instance."""
    instance_id = f"{id(instance):x}"
    class_name = class_name or type(instance).__name__
    
    with _checkpointer_lock:
        if instance_id not in _checkpointers:
            _checkpointers[instance_id] = MethodCheckpointer(class_name, instance_id)
        return _checkpointers[instance_id]


def method_checkpoint(func: Callable) -> Callable:
    """
    Decorator to checkpoint method calls.
    
    Captures:
    - Method inputs (args and kwargs)
    - Method outputs (return values)
    - Timestamps and execution duration
    - Exception information
    - Call hierarchy for nested calls
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        checkpointer = get_method_checkpointer(self)
        
        if not checkpointer.enabled:
            # Direct call if checkpointing is disabled
            return func(self, *args, **kwargs)
        
        # Start tracking the call
        call_id = checkpointer.start_call(func.__name__, list(args), kwargs)
        
        try:
            result = func(self, *args, **kwargs)
            checkpointer.end_call(call_id, result=result)
            return result
        
        except Exception as e:
            checkpointer.end_call(call_id, exception=e)
            raise
    
    return wrapper


def checkpoint_all_methods(cls: Type) -> Type:
    """
    Class decorator to add method checkpointing to all methods.
    
    Usage:
        @checkpoint_all_methods
        class MyClass:
            def method1(self): ...
            def method2(self): ...
    """
    logger.info(f"Applying checkpoint_all_methods to class: {cls.__name__}")
    def should_checkpoint_method(name: str, method: Any) -> bool:
        """Determine if a method should be checkpointed."""
        # For OptionHedgeBuilder, include private methods (except __magic__)
        if cls.__name__ == "OptionHedgeBuilder":
            if name.startswith('__'):
                return False
        else:
            if name.startswith('_'):
                return False
        if not callable(method):
            return False
        if isinstance(method, (classmethod, staticmethod)):
            return False
        if hasattr(method, '__self__'):  # bound method
            return False
        return True
    
    # Get all methods to checkpoint
    methods_to_wrap = []
    for name in dir(cls):
        if name.startswith('__'):
            continue
        
        attr = getattr(cls, name)
        if should_checkpoint_method(name, attr):
            methods_to_wrap.append(name)
    
    # Wrap each method with the checkpoint decorator
    for method_name in methods_to_wrap:
        original_method = getattr(cls, method_name)
        wrapped_method = method_checkpoint(original_method)
        setattr(cls, method_name, wrapped_method)
    
    # Add a cleanup method to save summaries
    original_init = cls.__init__
    
    def enhanced_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Register cleanup on instance deletion
        import weakref
        instance_id = f"{id(self):x}"  # Capture instance_id in the closure
        def cleanup_callback(ref):
            try:
                if instance_id in _checkpointers:
                    _checkpointers[instance_id].save_summary()
                    del _checkpointers[instance_id]
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpointer: {e}")
        
        # weakref.ref(self, cleanup_callback)  # Temporarily disabled - may be causing issues
    
    cls.__init__ = enhanced_init
    
    # Add a method to manually save summary
    def save_method_trace_summary(self):
        """Save a summary of all method calls for this instance."""
        checkpointer = get_method_checkpointer(self)
        checkpointer.save_summary()
    
    cls.save_method_trace_summary = save_method_trace_summary
    
    return cls


def save_all_summaries():
    """Save summaries for all active checkpointers."""
    with _checkpointer_lock:
        for checkpointer in _checkpointers.values():
            try:
                checkpointer.save_summary()
            except Exception as e:
                logger.warning(f"Failed to save summary for {checkpointer.instance_id}: {e}")


# Register cleanup on module exit
import atexit
atexit.register(save_all_summaries)