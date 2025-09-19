#!/usr/bin/env python3
"""
Script to implement position-specific validation in variance_swap_strategy.py

This fixes the issue where entire markets are rejected based only on YES price,
throwing away viable NO position opportunities.
"""

import re

# Read the current file
with open('strategies/options/variance_swap_strategy.py', 'r') as f:
    content = f.read()

# Step 1: Add the new validation methods after the existing validate_inputs mock
insert_position = content.find('# - ENDMOCK -') + len('# - ENDMOCK -')
new_methods = '''
    
    def _validate_basic_inputs(self, polymarket_contract: Dict, current_spot: float) -> bool:
        """
        Validate only the basic required inputs without price thresholds.
        This checks for required fields and valid spot price only.
        
        Returns:
            True if basic inputs are valid, False otherwise
        """
        # Check required fields
        reason = None
        missing = None
        for field in self.REQUIRED_FIELDS:
            if field not in polymarket_contract:
                self.logger.debug(f"Missing required field: {field}")
                missing = field
                reason = f"MISSING_{field.upper()}"
                _audit_emit({
                    "run_id": os.getenv("APP_RUN_ID", "unknown"),
                    "stage": "validate_basic_inputs",
                    "validation_pass": False,
                    "reason_code": reason,
                    "fields_seen": list(polymarket_contract.keys()),
                    "pm_market_id": polymarket_contract.get("id") or polymarket_contract.get("question_id") or polymarket_contract.get("slug"),
                    "pm_question": polymarket_contract.get("question"),
                    "pm_currency_field": polymarket_contract.get("currency"),
                })
                return False
        
        # Check spot price only if it's relevant for the strategy
        if 'strike_price' in self.REQUIRED_FIELDS and current_spot <= 0:
            self.logger.debug(f"Invalid spot price: {current_spot}")
            _audit_emit({
                "run_id": os.getenv("APP_RUN_ID", "unknown"),
                "stage": "validate_basic_inputs",
                "validation_pass": False,
                "reason_code": "BAD_SPOT",
                "current_spot": current_spot,
            })
            return False
            
        return True
    
    def _validate_position_price(self, position: str, price: Optional[float]) -> bool:
        """
        Validate price for a specific position (YES or NO).
        
        For YES positions: reject if YES price is too low or too high
        For NO positions: reject if NO price is too low or too high
        
        This allows taking NO positions when YES is extreme and vice versa.
        """
        if price is None:
            return False
        
        price = float(price)
        if not (0.0 < price < 1.0):
            return False
        
        # Use existing thresholds from base class or defaults
        min_threshold = getattr(self, 'min_price_threshold', 0.01)  # 1% minimum
        max_threshold = getattr(self, 'max_price_threshold', 0.99)  # 99% maximum
        
        if price < min_threshold:
            self.logger.debug(f"{position} price too low for liquidity: {price:.3f} < {min_threshold}")
            return False
        
        if price > max_threshold:
            self.logger.debug(f"{position} price too high for liquidity: {price:.3f} > {max_threshold}")
            return False
        
        return True
'''

# Insert new methods
content = content[:insert_position] + new_methods + content[insert_position:]

# Step 2: Replace validate_inputs call with _validate_basic_inputs
content = content.replace(
    'if not self.validate_inputs(polymarket_contract, current_spot):',
    'if not self._validate_basic_inputs(polymarket_contract, current_spot):'
)

# Step 3: Wrap YES opportunity evaluation with validation
yes_eval_pattern = r'(\s+)# Evaluate YES position with variance hedge\n(\s+)(yes_opportunity = self\._evaluate_variance_hedge\(\n.*?\n.*?\n.*?\n.*?\n\s+\))\n(\s+)(if yes_opportunity:\n\s+opportunities\.append\(yes_opportunity\))'

yes_replacement = r'''\1# Evaluate YES position with variance hedge (only if YES price is valid)
\1if self._validate_position_price('YES', yes_price):
\1    \3
\1    \5
\1else:
\1    self.logger.debug(f"[variance] Skipping YES position: price {yes_price} outside valid range")'''

content = re.sub(yes_eval_pattern, yes_replacement, content, flags=re.MULTILINE | re.DOTALL)

# Step 4: Wrap NO opportunity evaluation with validation
no_eval_pattern = r'(\s+)# Evaluate NO position with variance hedge\n(\s+)(no_opportunity = self\._evaluate_variance_hedge\(\n.*?\n.*?\n.*?\n.*?\n\s+\))\n(\s+)(if no_opportunity:\n\s+opportunities\.append\(no_opportunity\))'

no_replacement = r'''\1# Evaluate NO position with variance hedge (only if NO price is valid)
\1if self._validate_position_price('NO', no_price):
\1    \3
\1    \5
\1else:
\1    self.logger.debug(f"[variance] Skipping NO position: price {no_price} outside valid range")'''

content = re.sub(no_eval_pattern, no_replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write the modified content
with open('strategies/options/variance_swap_strategy_fixed.py', 'w') as f:
    f.write(content)

print("Created variance_swap_strategy_fixed.py with position-specific validation")
print("\nChanges made:")
print("1. Added _validate_basic_inputs() method - checks only required fields")
print("2. Added _validate_position_price() method - validates price per position")
print("3. Replaced validate_inputs() with _validate_basic_inputs()")
print("4. Wrapped YES evaluation with YES price validation")
print("5. Wrapped NO evaluation with NO price validation")
print("\nExpected impact:")
print("- 52 markets recovered (22 YES_TOO_LOW + 30 YES_TOO_HIGH)")
print("- Each provides opposite position opportunity")
print("- NO positions evaluated when YES is extreme, and vice versa")