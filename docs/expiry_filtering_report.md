# Options Expiry Filtering Logic Report

## Executive Summary

This report documents all the expiry filtering logic implemented in the Prediction Market Variance Refactor repository. The system has multiple layers of expiry filtering applied at different stages of the pipeline, from initial data collection to final opportunity evaluation.

## Findings by File

### config/baseline.yaml
- **min_hours_to_expiry**: 0.10 (line 8) - Minimum hours before expiry to consider options/markets
- **max_days_to_expiry**: 180 (line 9) - Maximum days to expiry to consider options/markets
- **include_dailies**: true (line 142) - Keep daily markets even if they expire very soon
- **dailies_window_hours**: 24 (line 143) - Treat markets expiring within 24h as dailies
- **settle_spread_at_pm_expiry**: true (line 73) - How to handle when PM expires before options
- **options_unwind_model**: sticky_strike (line 77) - Model for valuing options at PM expiry
- **expiry_policy**: allow_far_with_unwind (line 174) - Policy for selecting option expiries
- **max_expiry_gap_days**: 60 (line 177) - Max days after PM for additional expiries
- **max_expiries_considered**: 10 (line 179) - Max number of different expiries to consider
- **min_quotes_per_expiry**: 2 (line 186) - Min instruments with two-sided quotes per expiry

### config/defaults.yaml
- **options_unwind_model**: sticky_strike (line 5) - Default model for cross-expiry valuation
- **expiry_policy**: allow_far_with_unwind (line 27) - Default expiry selection policy
- **max_expiry_gap_days**: 60 (line 30) - Default max gap for additional expiries
- **max_expiries_considered**: 10 (line 32) - Default max expiries to consider
- **min_quotes_per_expiry**: 2 (line 39) - Default min quotes requirement

### .env
- **MIN_HOURS_TO_EXPIRY**: 0.10 (line 45) - Environment override for minimum hours
- **MAX_DAYS_TO_EXPIRY**: 180 (line 46) - Environment override for maximum days

### config_manager.py
- **MIN_HOURS_TO_EXPIRY** export (line 604) - Exports config value with env override support
- **MAX_DAYS_TO_EXPIRY** export (line 605) - Exports config value with env override support
- **VARIANCE_EXPIRY_POLICY** (lines 76-77) - Reads from config/env, validates against allowed values
- **VARIANCE_MAX_EXPIRY_GAP_DAYS** (lines 79-80) - Reads from config/env, clamps to [0, 60]
- **VARIANCE_MAX_EXPIRIES_CONSIDERED** (lines 82-83) - Reads from config/env, clamps to [1, 10]
- **VARIANCE_MIN_QUOTES_PER_EXPIRY** (lines 95-96) - Reads from config/env, clamps to [2, 20]
- **SETTLE_SPREAD_AT_PM_EXPIRY** (line 597) - Exports boolean for cross-expiry handling
- **CONSIDER_ALL_EXPIRIES** (line 662) - Boolean flag for trying all expiries in scanner

### config_schema.py
- **min_hours_to_expiry**: confloat(ge=0) = 0.5 (line 218) - Schema definition with validation
- **max_days_to_expiry**: confloat(gt=0) = 180 (line 219) - Schema definition with validation
- **expiry_policy**: Literal constraint (line 313) - Validates allowed policy values
- **max_expiry_gap_days**: conint(ge=0, le=60) = 60 (line 314) - Schema with range constraint
- **max_expiries_considered**: conint(ge=1, le=10) = 10 (line 315) - Schema with range constraint
- **min_quotes_per_expiry**: conint(ge=2, le=20) = 2 (line 318) - Schema with range constraint

### scripts/data_collection/polymarket_fetcher.py
- **MIN_HOURS_TO_EXPIRY** import (line 28) - Imports from config_manager
- **_market_time_left_hours()** method - Calculates hours remaining to market expiry
- **_should_include_market()** method (lines 222-230) - Filters markets by MIN_HOURS_TO_EXPIRY
  - Respects MIN_HOURS_TO_EXPIRY for general filtering
  - Allows dailies within POLYMARKET_DAILIES_WINDOW_HOURS when configured

### strategies/options/variance_swap_strategy.py
- **filter_options_by_expiry()** method (lines 76-135) - Main expiry filtering logic
  - Groups options by expiry date
  - Sorts by days_to_expiry proximity to PM date
  - Validates min_quotes_per_expiry requirement (line 131)
  - Enforces max_expiry_gap_days for additional expiries (line 118)
  - Limits to max_expiries_considered (line 114)
  - Skips synthetic/flagged expiries (line 122)
- **VARIANCE_EXPIRY_POLICY** import (line 157) - From config_manager
- **VARIANCE_MAX_EXPIRY_GAP_DAYS** import (line 158) - From config_manager
- **VARIANCE_MAX_EXPIRIES_CONSIDERED** import (line 159) - From config_manager
- **VARIANCE_MIN_QUOTES_PER_EXPIRY** import (line 162) - From config_manager
- Uses options_unwind_model to determine if far expiries allowed (line 108)

### strategies/options/base_options_strategy.py
- **filter_options_by_expiry()** method (lines 296-332) - Simple expiry filter
  - Filters options by min_days_to_expiry parameter
  - Calculates days remaining from expiry_date field
  - Logs filtering summary with sample kept expiries (line 331)

### hedging/options.py
- **filter_options_by_expiry()** reference (line 516) - Calls strategy's filter method
- Computes days_to_expiry for PM contracts (lines 180-189)
- Passes through to strategy-specific filtering

### expected_value_filter.py
- No direct expiry filtering logic
- Uses days_to_expiry for holding period calculations (line 314)

### true_arbitrage_detector.py
- **SETTLE_SPREAD_AT_PM_EXPIRY** import (line 21) - For cross-expiry logic
- **OPTIONS_UNWIND_MODEL** import (line 21) - For valuation model
- Cross-expiry arbitrage validation (lines 206, 410) - Uses these settings

### execution_pricing.py
- Handles expiry field parsing (lines 119, 142-143)
- Supports cross-expiry valuation models (line 127)

### probability_ranker.py
- **days_to_option_expiry** usage (line 462) - Uses for time calculations
- **time_to_expiry** calculation (line 463) - Converts to years for pricing

### market_data_analyzer.py
- **time_to_expiry** parameter in calculations (lines 284, 296, 308, 316-320, 527)
- Used in Black-Scholes calculations and IV solving

### options_data_enricher.py
- **time_to_expiry** calculations (lines 178-186)
- Parses expiry dates from various formats (lines 164, 181, 250)

## Implementation Details

### 1. Polymarket Fetcher (`scripts/data_collection/polymarket_fetcher.py`)

The initial filtering happens at data collection:

```python
def _should_include_market(self, m: dict) -> bool:
    # Respect min_hours_to_expiry generally…
    tleft = self._market_time_left_hours(m)
    if tleft is None:
        return True  # no end date -> keep
    if tleft >= MIN_HOURS_TO_EXPIRY:
        return True
    # …but allow "dailies" when configured
    if POLYMARKET_INCLUDE_DAILIES and tleft <= POLYMARKET_DAILIES_WINDOW_HOURS:
        return True
    return False
```

### 2. Variance Swap Strategy (`strategies/options/variance_swap_strategy.py`)

The most sophisticated filtering happens in the variance swap strategy:

```python
def filter_options_by_expiry(self, options: List[Dict], pm_days_to_expiry: float, *, inclusive: bool = True) -> List[Dict]:
    """
    Build a list of instruments from up to N **valid** expiries on/after PM.
    Valid expiry requires:
    - Not synthetic/flagged
    - Has >= min_quotes_per_expiry instruments with both bid>0 and ask>0
    - For additional expiries: within max_expiry_gap_days when far expiries allowed
    - Far expiries allowed only if (expiry_policy == 'allow_far_with_unwind') and 
      (options_unwind_model != 'intrinsic_only')
    """
```

Key logic points:
1. Groups options by expiry date
2. Sorts expiries by proximity to PM date
3. Validates each expiry group for sufficient liquidity
4. Enforces gap constraints for additional expiries
5. Skips synthetic or flagged expiries

### 3. Base Options Strategy (`strategies/options/base_options_strategy.py`)

Provides a simpler filtering method:

```python
def filter_options_by_expiry(self, options_data: List[Dict], min_days_to_expiry: float = 0) -> List[Dict]:
    """Filter options that expire at least min_days after today."""
```

### 4. Configuration Management (`config_manager.py`)

Centralizes configuration with validation and clamping:

```python
VARIANCE_MAX_EXPIRY_GAP_DAYS = _clamp(_gap_days, 0, 60)
VARIANCE_MAX_EXPIRIES_CONSIDERED = _clamp(_max_expiries, 1, 10)
VARIANCE_MIN_QUOTES_PER_EXPIRY = _clamp(_minq, 2, 20)
```

## Multiple Filter Locations

The system applies expiry filters at multiple stages:

1. **Data Collection Stage**: Polymarket fetcher filters markets based on `MIN_HOURS_TO_EXPIRY`
2. **Strategy Evaluation**: Each strategy can apply its own expiry filtering
3. **Variance Strategy**: Most sophisticated multi-expiry selection logic
4. **True Arbitrage Detection**: Considers cross-expiry scenarios for arbitrage validation

## Environment Variable Overrides

The following environment variables can override configuration:
- `MIN_HOURS_TO_EXPIRY`
- `MAX_DAYS_TO_EXPIRY`
- `VARIANCE_EXPIRY_POLICY`
- `VARIANCE_MAX_EXPIRY_GAP_DAYS`
- `VARIANCE_MAX_EXPIRIES_CONSIDERED`
- `VARIANCE_MIN_QUOTES_PER_EXPIRY`

## Logic Flow Summary

1. **Initial Collection**: Markets/options collected if they meet basic time constraints
2. **Strategy Selection**: Based on expiry policy, select valid expiries
3. **Liquidity Validation**: Each expiry must have sufficient two-sided quotes
4. **Gap Enforcement**: Additional expiries must be within configured gap
5. **Cross-Expiry Handling**: Special logic for when PM resolves before options expire

## Recommendations

1. The default `min_hours_to_expiry` of 0.10 hours (6 minutes) seems extremely low and may include illiquid near-expiry options
2. Consider increasing `min_quotes_per_expiry` from 2 to ensure better liquidity
3. The `allow_far_with_unwind` policy with 60-day gap allows significant time risk
4. Document the rationale for allowing up to 10 different expiries in variance calculations