"""
Base Strategy Class - Abstract base for all arbitrage strategies

This module defines the interface that all strategies must implement.
Ensures consistent behavior across different hedging approaches.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Sequence
import logging, os
from utils.validation_audit import emit as _audit_emit


class BaseStrategy(ABC):
    """
    Abstract base class for all arbitrage strategies.
    
    All strategies must implement:
    - evaluate_opportunities: Main entry point for strategy evaluation
    - get_strategy_name: Returns human-readable strategy name
    - get_strategy_type: Returns strategy category (options/perpetuals/hybrid)
    """
    
    # Subclasses may override this to declare only what they truly need.
    REQUIRED_FIELDS: Sequence[str] = ('strike_price', 'yes_price', 'days_to_expiry', 'currency')
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def evaluate_opportunities(
        self,
        polymarket_contract: Dict,
        hedge_instruments: Dict,
        current_spot: float,
        position_size: float = 100
    ) -> List[Dict]:
        """
        Evaluate arbitrage opportunities for this strategy.
        
        Args:
            polymarket_contract: PM contract details including:
                - strike_price: Strike price
                - yes_price: Current YES price
                - no_price: Current NO price
                - days_to_expiry: Days until PM expiry
                - currency: Underlying currency (BTC/ETH/etc)
                - is_above: True if "above" contract, False if "below"
            
            hedge_instruments: Available hedging instruments:
                - 'options': List of options data
                - 'perps': Perpetual futures data
                - 'spot': Spot market data
            
            current_spot: Current spot price of underlying
            position_size: Position size in USD
            
        Returns:
            List of opportunity dictionaries, each containing:
                - strategy_name: Name of the strategy
                - strategy_type: Type (options/perpetuals/hybrid)
                - pm_side: YES or NO
                - hedge_description: Description of hedge
                - max_profit: Maximum profit potential
                - max_loss: Maximum loss potential
                - probability_profit: Estimated probability of profit
                - required_capital: Capital required
                - ... (strategy-specific fields)
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return human-readable strategy name"""
        raise NotImplementedError
    
    @abstractmethod
    def get_strategy_type(self) -> str:
        """Return strategy type: 'options', 'perpetuals', or 'hybrid'"""
        raise NotImplementedError
    
    def validate_inputs(self, polymarket_contract: Dict, current_spot: float) -> bool:
        """
        Validate required inputs are present and valid.
        
        Implements enhanced price validation to avoid unrealistic opportunities:
        - Minimum price threshold to avoid extreme leverage
        - Maximum price threshold to ensure liquidity
        - Spread validation to ensure reasonable market making costs
        
        Returns:
            True if inputs are valid, False otherwise
        """
        # Strategy-specific required fields (subclass may override REQUIRED_FIELDS)
        reason = None
        missing = None
        for field in self.REQUIRED_FIELDS:
            if field not in polymarket_contract:
                self.logger.debug(f"Missing required field: {field}")
                missing = field
                reason = f"MISSING_{field.upper()}"
                _audit_emit({
                    "run_id": os.getenv("APP_RUN_ID", "unknown"),
                    "stage": "validate_inputs",
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
                "stage": "validate_inputs",
                "validation_pass": False,
                "reason_code": "BAD_SPOT",
                "current_spot": current_spot,
            })
            return False
        
        # Price sanity checks (only if present)
        yes_price = polymarket_contract.get('yes_price', None)
        no_price = polymarket_contract.get('no_price', None)
        
        if yes_price is not None:
            yes_price = float(yes_price)
            if not (0.0 < yes_price < 1.0):
                self.logger.debug(f"Invalid YES price: {yes_price}")
                return False
            
            # Enhanced validation: Minimum liquidity thresholds
            # These can be overridden by subclasses for strategy-specific needs
            min_price_threshold = getattr(self, 'min_price_threshold', 0.02)  # 2% minimum
            max_price_threshold = getattr(self, 'max_price_threshold', 0.98)  # 98% maximum
            
            # Check for extreme prices that likely have no liquidity
            if yes_price < min_price_threshold:
                self.logger.debug(f"YES price too low for liquidity: {yes_price:.3f} < {min_price_threshold}")
                _audit_emit({
                    "run_id": os.getenv("APP_RUN_ID", "unknown"),
                    "stage": "validate_inputs",
                    "validation_pass": False,
                    "reason_code": "YES_TOO_LOW",
                    "yes_price": yes_price,
                    "min_price_threshold": min_price_threshold,
                })
                return False
            
            if yes_price > max_price_threshold:
                self.logger.debug(f"YES price too high for liquidity: {yes_price:.3f} > {max_price_threshold}")
                _audit_emit({
                    "run_id": os.getenv("APP_RUN_ID", "unknown"),
                    "stage": "validate_inputs",
                    "validation_pass": False,
                    "reason_code": "YES_TOO_HIGH",
                    "yes_price": yes_price,
                    "max_price_threshold": max_price_threshold,
                })
                return False
        
        if no_price is not None:
            no_price = float(no_price)
            if not (0.0 < no_price < 1.0):
                self.logger.debug(f"Invalid NO price: {no_price}")
                return False
        
        # If both legs are present, optional sum-deviation guard
        if yes_price is not None and no_price is not None:
            max_sum_deviation = getattr(self, 'max_sum_deviation', 0.10)
            s = yes_price + no_price
            if abs(s - 1.0) > max_sum_deviation:
                self.logger.warning(f"Prices don't sum properly: YES={yes_price:.3f} + NO={no_price:.3f} = {(yes_price + no_price):.3f}")
                _audit_emit({
                    "run_id": os.getenv("APP_RUN_ID", "unknown"),
                    "stage": "validate_inputs",
                    "validation_pass": False,
                    "reason_code": "SUM_DEVIATION",
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "sum": s,
                    "max_sum_deviation": max_sum_deviation,
                })
                return False
        
        _audit_emit({
            "run_id": os.getenv("APP_RUN_ID", "unknown"),
            "stage": "validate_inputs",
            "validation_pass": True,
            "reason_code": "OK",
            "pm_market_id": polymarket_contract.get("id") or polymarket_contract.get("question_id") or polymarket_contract.get("slug"),
            "pm_question": polymarket_contract.get("question"),
            "currency": polymarket_contract.get("currency"),
        })
        return True
    
    def calculate_pm_payoffs(
        self, 
        pm_side: str, 
        pm_price: float, 
        position_size: float
    ) -> Tuple[float, float, float]:
        """
        Calculate Polymarket position payoffs.
        
        Args:
            pm_side: 'YES' or 'NO'
            pm_price: Entry price (0-1)
            position_size: Position size in USD
            
        Returns:
            Tuple of (shares, payoff_if_true, payoff_if_false)
        """
        shares = position_size / pm_price
        
        if pm_side == 'YES':
            payoff_if_true = shares * 1.0 - position_size  # Win
            payoff_if_false = -position_size  # Lose
        else:  # NO
            payoff_if_true = -position_size  # Lose
            payoff_if_false = shares * 1.0 - position_size  # Win
        
        return shares, payoff_if_true, payoff_if_false

    # --- New: canonical required capital computation ---
    @staticmethod
    def required_capital(
        pm_cash_out: float,
        option_entry_debit: float,
        option_entry_credit: float,
        is_short_vertical: bool,
        spread_width: float,
        spread_contracts: float,
    ) -> float:
        """
        Required capital ≥ PM cash out + net option debits + margin.
        For a short vertical, margin ≤ spread_width * spread_contracts (USD numeraire).
        """
        pm_cash_out = float(pm_cash_out)
        net_option_outflow = float(option_entry_debit) - float(option_entry_credit)
        margin = (float(spread_width) * float(spread_contracts)) if is_short_vertical else 0.0
        return max(0.0, pm_cash_out + net_option_outflow + margin)