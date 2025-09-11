# Method Checkpoint Decorator System

A comprehensive debugging and tracing system that captures all method calls with inputs, outputs, timestamps, and call hierarchy information.

## Features

- **Complete Method Tracing**: Captures all method calls with full context
- **Input/Output Serialization**: Safely serializes complex data structures
- **Exception Handling**: Gracefully handles and logs exceptions
- **Call Hierarchy**: Tracks nested method calls and their relationships  
- **Thread-Safe**: Safe for use in multi-threaded environments
- **Performance Statistics**: Tracks call counts, durations, and failure rates
- **Automatic Cleanup**: Handles resource management automatically

## Quick Start

### 1. Apply to a Class

```python
from utils.method_checkpoint_decorator import checkpoint_all_methods

@checkpoint_all_methods
class OptionHedgeBuilder:
    def __init__(self, scanners, market_analyzer=None, logger=None):
        # Your existing __init__ code
        pass
    
    def build(self, market_snapshot):
        # Your existing method code
        pass
    
    def _pm_has_liquidity(self, contract):
        # Private methods are NOT automatically checkpointed
        # (only public methods are tracked by default)
        pass
```

### 2. Enable Debug Mode

Set the `DEBUG` environment variable:

```bash
export DEBUG=true
```

Or in Python:

```python
import os
os.environ["DEBUG"] = "true"
```

### 3. Run Your Code

When debug mode is enabled, all method calls will be automatically checkpointed to:

```
debug_runs/YYYYMMDD-HHMMSSZ/OptionsHedgeBuilder_checkpoints/
```

## Checkpoint Directory Structure

```
debug_runs/
└── 20250910-201837Z/
    └── OptionsHedgeBuilder_checkpoints/
        └── a1b2c3d4e5f6/  # Instance ID (hex)
            ├── a1b2c3d4e5f6_build_1_start.json
            ├── a1b2c3d4e5f6_build_1_end.json
            ├── a1b2c3d4e5f6_construct_options_hedge_2_start.json
            ├── a1b2c3d4e5f6_construct_options_hedge_2_end.json
            └── method_trace_summary.json
```

## Checkpoint File Contents

### Start/End Files

Each method call generates two files:

**`{instance_id}_{method_name}_{call_number}_start.json`**
```json
{
  "phase": "start",
  "trace": {
    "method_name": "build",
    "instance_id": "a1b2c3d4e5f6",
    "call_id": "a1b2c3d4e5f6_build_1",
    "start_timestamp": "2025-09-10T20:18:37.123456Z",
    "args": [...],
    "kwargs": {...},
    "parent_call_id": null,
    "nested_calls": []
  },
  "call_hierarchy": [...],
  "stats": {
    "call_count": 1,
    "total_duration": 0.0,
    "avg_duration": 0.0,
    "exception_count": 0
  }
}
```

**`{instance_id}_{method_name}_{call_number}_end.json`**
```json
{
  "phase": "end", 
  "trace": {
    "method_name": "build",
    "end_timestamp": "2025-09-10T20:18:37.456789Z",
    "duration": 0.333333,
    "result": {...},
    "exception": null,
    "nested_calls": ["a1b2c3d4e5f6_construct_options_hedge_2"]
  },
  "call_hierarchy": [...],
  "stats": {
    "call_count": 1,
    "total_duration": 0.333333,
    "avg_duration": 0.333333,
    "exception_count": 0
  }
}
```

### Summary File

**`method_trace_summary.json`**
```json
{
  "class_name": "OptionHedgeBuilder",
  "instance_id": "a1b2c3d4e5f6", 
  "total_calls": 15,
  "active_calls": 0,
  "method_statistics": {
    "build": {
      "call_count": 3,
      "total_duration": 1.234,
      "avg_duration": 0.411,
      "exception_count": 0,
      "last_called": "2025-09-10T20:18:37.456789Z"
    }
  },
  "call_trace": [...],
  "call_hierarchy": {
    "root_calls": [
      {
        "method_name": "build",
        "duration": 0.333,
        "exception": false,
        "children": [
          {
            "method_name": "construct_options_hedge", 
            "duration": 0.123,
            "exception": false,
            "children": []
          }
        ]
      }
    ]
  }
}
```

## Data Serialization

The system safely serializes complex objects:

- **Basic types**: int, float, str, bool, None → preserved as-is
- **Collections**: list, tuple, dict → recursively serialized (with limits)
- **Objects with `__dict__`**: converted to `{"__type__": "ClassName", "attributes": {...}}`
- **Unserializable objects**: converted to string representation
- **Circular references**: prevented with depth limiting (max depth: 3)
- **Large collections**: truncated (lists to 10 items, dicts to 20 items)

## Method Selection Rules

By default, only **public methods** are checkpointed:

- ✅ `build()` - public method
- ✅ `construct_options_hedge()` - public method  
- ❌ `_pm_has_liquidity()` - private method (starts with `_`)
- ❌ `__init__()` - dunder method (starts with `__`)
- ❌ `@classmethod` or `@staticmethod` - not instance methods

## Manual Control

### Individual Method Decoration

For fine-grained control, decorate individual methods:

```python
from utils.method_checkpoint_decorator import method_checkpoint

class MyClass:
    @method_checkpoint 
    def important_method(self):
        pass
    
    def untracked_method(self):
        # This won't be checkpointed
        pass
```

### Manual Summary Generation

```python
# Generate summary on demand
instance.save_method_trace_summary()

# Or save all active summaries
from utils.method_checkpoint_decorator import save_all_summaries
save_all_summaries()
```

## Performance Considerations

- **Overhead**: Minimal when `DEBUG=false` (checkpointing disabled)
- **Storage**: Checkpoints can use significant disk space for large datasets
- **Serialization**: Complex objects may slow down method calls
- **Thread Safety**: All operations are thread-safe with internal locking

## Troubleshooting

### No Checkpoint Files Generated

1. Ensure `DEBUG=true` environment variable is set
2. Check that the debug recorder is enabled:
   ```python
   from utils.debug_recorder import get_recorder
   recorder = get_recorder()
   print(f"Recorder enabled: {recorder.enabled}")
   ```

### Serialization Errors

The system gracefully handles serialization failures by:
- Converting problematic objects to string representations
- Truncating large data structures
- Logging warnings for failed serializations

### Large File Sizes

To reduce checkpoint file sizes:
- Limit the size of method arguments
- Use data summaries instead of full datasets where possible
- Clean up old debug runs periodically

## Example Usage

```python
import os
from hedging.options import OptionHedgeBuilder

# Enable checkpointing
os.environ["DEBUG"] = "true"

# Create instance (automatically gets checkpoint decoration)
builder = OptionHedgeBuilder(scanners={}, market_analyzer=None)

# All method calls are now automatically checkpointed
opportunities = builder.build(market_snapshot={})

# Manually save summary (also happens automatically on instance deletion)
builder.save_method_trace_summary()
```

## Integration with Existing Debug System

The checkpoint system integrates with the existing `StepDebugger`:

- Uses the same `debug_runs/` directory structure
- Respects the same `DEBUG` environment variable
- Leverages the existing `RawDataRecorder` infrastructure
- Adds method-level detail to complement step-level checkpoints

This provides both high-level pipeline checkpoints (via `StepDebugger`) and detailed method-level tracing (via `MethodCheckpointer`) in a unified debugging environment.