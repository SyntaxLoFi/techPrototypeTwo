# Variance Swap Strategy Checkpoint Guide

This document describes the checkpoints implemented in the variance swap strategy to track where opportunities are filtered out during evaluation.

## Overview

The variance swap strategy evaluates each Polymarket opportunity through multiple filtering stages. Each checkpoint logs when opportunities are dropped and why.

### Files
- **Checkpoint logging code**: `/strategies/options/variance_swap_strategy.py`
- **Checkpoint log output**: `/debug_runs/variance_swap_checkpoints.log`
- **Analysis script**: `/analyze_variance_swap_checkpoints.py`
- **Summary output**: `/debug_runs/variance_swap_checkpoint_summary.txt`

## Checkpoints

### 0A: Missing NO price and empty NO book
- **Location**: `evaluate_opportunities()` method, lines 646-653
- **Function**: Early gate before any option processing
- **Trigger**: NO price is missing AND NO order book is empty (no bid/ask)
- **Result**: Market immediately dropped
- **Code**:
  ```python
  if no_price is None:
      # Check NO book for synthetic price
      if nb <= 0.0 and na <= 0.0:
          # Log checkpoint 0A
          return []
  ```

### 0B: Basic validation failed
- **Location**: `evaluate_opportunities()` method, lines 589-600
- **Function**: `_validate_basic_inputs()` 
- **Trigger**: Missing required fields (e.g., `days_to_expiry`) or invalid spot price
- **Result**: Market immediately dropped
- **Code**:
  ```python
  if not self._validate_basic_inputs(polymarket_contract, current_spot):
      # Log checkpoint 0B
      return []
  ```

### 1: PM price-range gate
- **Location**: `evaluate_opportunities()` method, lines 686-698
- **Function**: `_validate_position_price()` for both YES and NO positions
- **Trigger**: 
  - YES price ≤ 1% or ≥ 99%
  - NO price ≤ 1% or ≥ 99%
  - If BOTH are invalid, market is dropped
- **Result**: Individual position skipped (or market dropped if both invalid)
- **Code**:
  ```python
  yes_price_valid = self._validate_position_price('YES', yes_price)
  no_price_valid = self._validate_position_price('NO', no_price)
  if not yes_price_valid and not no_price_valid:
      return []
  ```

### 2: Expiry selection filter
- **Location**: `evaluate_opportunities()` method, lines 663-680
- **Function**: `filter_options_by_expiry()` (inherited from base class)
- **Trigger**:
  - Options don't expire on/after PM resolution date
  - Missing two-sided quotes when `require_live_quotes=True`
  - Expiry has fewer than `min_quotes_per_expiry` valid instruments
- **Result**: All options filtered out → market dropped
- **Code**:
  ```python
  suitable_options = self.filter_options_by_expiry(options_data, days_to_expiry)
  if not suitable_options:
      # Log checkpoint 2
      return []
  ```

### 3: Too close expiry gate
- **Location**: `_evaluate_variance_hedge()` method, lines 763-770
- **Function**: Time-to-expiry validation
- **Trigger**: Option expiry < 0.5 days away
- **Result**: Opportunity dropped for that expiry/position
- **Code**:
  ```python
  if seconds_to_option_expiry <= 0 or days_to_option_expiry < 0.5:
      # Log checkpoint 3
      return None
  ```

### 4: Portfolio build gates
- **Location**: `_build_variance_swap_portfolio()` method
- **Function**: Multiple validation checks during portfolio construction
- **Sub-checkpoints**:

#### 4.1: Insufficient strikes after normalization
- **Location**: Lines 1205-1212
- **Trigger**: Fewer than `min_strikes_required` (default: 6) valid options
- **Code**:
  ```python
  valid = self.normalize_options(options)
  if len(valid) < int(self.min_strikes_required):
      # Log checkpoint 4 - insufficient strikes
      return None
  ```

#### 4.2: No strikes on both sides of K0
- **Location**: Lines 1287-1301
- **Trigger**: No strikes below K0 OR no strikes above K0
- **Code**:
  ```python
  strikes_below = [k for k in strikes_all if k < K0]
  strikes_above = [k for k in strikes_all if k > K0]
  if not strikes_below or not strikes_above:
      # Log checkpoint 4 - no strikes both sides
      return None
  ```

#### 4.3: Insufficient strikes after deduplication
- **Location**: Lines 1414-1421
- **Trigger**: After removing duplicates, fewer than `min_strikes_required`
- **Code**:
  ```python
  if len(valid) < int(self.min_strikes_required):
      # Log checkpoint 4 - insufficient after dedup
      return None
  ```

### 4A: Forward/PM gap check
- **Location**: `_evaluate_variance_hedge()` method, lines 901-909
- **Function**: Checks gap between forward price and PM strike
- **Trigger**: Gap between forward and PM strike > 5% (when using forward anchor mode)
- **Result**: Opportunity dropped
- **Code**:
  ```python
  if (var_swap_portfolio.get('anchor_mode', 'forward') == 'forward') and (pm_gap is not None) and (pm_gap > max_gap):
      self.logger.info(f"[variance] skip: forward/PM gap {pm_gap:.2%} exceeds {max_gap:.2%}")
      # Log checkpoint 4A
      return None
  ```

### 5: Cost recovery gate
- **Location**: `_evaluate_variance_hedge()` method, lines 961-975
- **Function**: Validates hedge can recover PM entry cost
- **Trigger**: Hedge value at PM strike (with slippage) doesn't recover entry cost within tolerance (default: 2%)
- **Result**: Opportunity dropped
- **Code**:
  ```python
  if not passes_recovery:
      self.logger.info("[variance] reject: cost recovery at PM strike %s fails...")
      # Log checkpoint 5
      return None
  ```

### 6: Digital bounds gate
- **Location**: `_evaluate_variance_hedge()` method, lines 1011-1019
- **Function**: No-arbitrage bounds check at PM strike
- **Trigger**: PM YES price is within digital no-arb bounds (when `enforce_pm_digital_bounds=True`)
- **Result**: Opportunity dropped (no edge detected)
- **Code**:
  ```python
  if self.enforce_pm_digital_bounds and inside:
      self.logger.info("[variance] prune: PM YES within digital no-arb band...")
      # Log checkpoint 6
      return None
  ```

### Final: Total opportunities returned
- **Location**: `evaluate_opportunities()` method, lines 728-733
- **Function**: Summary logging
- **Output**: Total count of opportunities that passed all filters
- **Code**:
  ```python
  # Final checkpoint summary
  with open(checkpoint_log_path, "a") as f:
      f.write(f"Final result: {len(opportunities)} opportunities returned\n")
  ```

## Configuration Parameters

Key parameters that affect filtering:

- `min_price_threshold`: 0.01 (1%) - Minimum YES/NO price
- `max_price_threshold`: 0.99 (99%) - Maximum YES/NO price  
- `min_strikes_required`: 6 - Minimum distinct strikes needed
- `require_live_quotes_for_trades`: True - Require bid>0 and ask>0
- `min_quotes_per_expiry`: 4 - Minimum valid quotes per expiry
- `COST_RECOVERY_TOL`: 0.02 (2%) - Cost recovery tolerance
- `enforce_pm_digital_bounds`: True - Enable digital bounds check

## Usage

1. Run your variance swap strategy evaluation
2. Check `debug_runs/variance_swap_checkpoints.log` for detailed per-market filtering
3. Run `python3 analyze_variance_swap_checkpoints.py` for summary statistics
4. Review which checkpoints are causing the most opportunity drops