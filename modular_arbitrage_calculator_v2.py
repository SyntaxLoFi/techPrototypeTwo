"""
Modular Arbitrage Calculator V2 - Uses new strategy architecture

This is the updated version that dynamically loads and executes
all available strategies from the modular strategy system.
"""

import numpy as np
from typing import Dict, List, Optional
import logging
from config_manager import DEFAULT_VOLATILITY, TRUE_ARBITRAGE_BONUS, NEAR_ARBITRAGE_BONUS, RISK_FREE_RATE

# Import strategy loader
from strategies.strategy_loader import StrategyLoader

logger = logging.getLogger(__name__)


class ModularArbitrageCalculatorV2:
    """
    Enhanced arbitrage calculator that uses the modular strategy system.
    Dynamically loads and executes all available strategies.
    """
    
    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger('ModularArbitrageCalculatorV2')
        self.volatility = DEFAULT_VOLATILITY
        
        # Load all strategies
        self.strategy_loader = StrategyLoader()
        self.strategies = self.strategy_loader.instantiate_all_strategies(risk_free_rate)
        
        # Log loaded strategies
        strategy_info = self.strategy_loader.get_strategy_info()
        self.logger.info(f"Loaded {len(self.strategies)} strategies:")
        for info in strategy_info:
            self.logger.info(f"  - {info['name']} ({info['type']})")
    
    def find_all_arbitrage_opportunities(
        self,
        polymarket_contract: Dict,
        hedge_instruments: Dict,
        current_spot: float,
        position_size: float = 100
    ) -> List[Dict]:
        """
        Evaluate all arbitrage opportunities using all loaded strategies.
        
        Args:
            polymarket_contract: PM contract details
            hedge_instruments: Available hedging instruments
            current_spot: Current spot price
            position_size: Position size in USD
            
        Returns:
            List of all evaluated arbitrage opportunities
        """
        all_opportunities = []
        
        # Log inputs
        self.logger.info(f"[CALC] Finding opportunities for: {polymarket_contract.get('question', 'Unknown')[:50]}...")
        self.logger.info(f"[CALC] Hedge instruments provided: {list(hedge_instruments.keys())}")
        for key, value in hedge_instruments.items():
            if isinstance(value, list):
                self.logger.info(f"[CALC]   {key}: {len(value)} items")
                if value and len(value) > 0:
                    self.logger.debug(f"[CALC]   {key} sample: {value[0]}")
            elif isinstance(value, dict):
                self.logger.info(f"[CALC]   {key}: dict with keys {list(value.keys())}")
        
        # Run each strategy
        for strategy in self.strategies:
            try:
                # Check if strategy can run with available instruments
                if not self._can_run_strategy(strategy, hedge_instruments):
                    self.logger.debug(f"[CALC] Skipping {strategy.get_strategy_name()} - missing required instruments")
                    continue
                
                # Evaluate opportunities
                self.logger.debug(f"[CALC] Evaluating {strategy.get_strategy_name()}...")
                opportunities = strategy.evaluate_opportunities(
                    polymarket_contract=polymarket_contract,
                    hedge_instruments=hedge_instruments,
                    current_spot=current_spot,
                    position_size=position_size
                )
                self.logger.info(f"[CALC] {strategy.get_strategy_name()} returned {len(opportunities)} opportunities")
                
                # Add strategy metadata to each opportunity
                for opp in opportunities:
                    opp['strategy_class'] = strategy.__class__.__name__
                    opp['strategy_module'] = strategy.__class__.__module__
                    
                    # Normalize Polymarket view so the ranker reads a price, not default 0.5
                    opp['polymarket'] = {
                        'question':       polymarket_contract.get('question'),
                        'strike':         polymarket_contract.get('strike_price'),
                        'yes_price':      polymarket_contract.get('yes_price'),
                        'no_price':       polymarket_contract.get('no_price'),
                        'end_date':       polymarket_contract.get('end_date'),
                        'days_to_expiry': polymarket_contract.get('days_to_expiry'),
                    }
                    
                    # Directional helpers used by the UI
                    is_above = polymarket_contract.get('is_above', True)
                    opp['pm_direction'] = 'bullish' if is_above else 'bearish'
                    opp['pm_wins_if']   = f"price {'>=' if is_above else '<='} ${polymarket_contract.get('strike_price', 0):,.0f}"
                
                all_opportunities.extend(opportunities)
                self.logger.debug(
                    f"{strategy.get_strategy_name()} returned {len(opportunities)} opportunities"
                )
            except Exception as e:
                self.logger.error(
                    f"Error running strategy {strategy.get_strategy_name()}: {str(e)}"
                )
        
        # Sort by profitability and arbitrage quality
        all_opportunities.sort(
            key=lambda x: self._score_opportunity(x),
            reverse=True
        )
        
        return all_opportunities
    
    def _can_run_strategy(self, strategy, hedge_instruments: Dict) -> bool:
        """
        Check if a strategy can run with available instruments.
        """
        strategy_type = strategy.get_strategy_type()
        strategy_name = strategy.get_strategy_name()
        
        if strategy_type == 'options':
            can_run = 'options' in hedge_instruments and bool(hedge_instruments['options'])
            self.logger.debug(f"[CALC] {strategy_name} (options): can_run={can_run}, has options={len(hedge_instruments.get('options', []))}")
            return can_run
        elif strategy_type == 'perpetuals':
            can_run = 'perps' in hedge_instruments and bool(hedge_instruments['perps'])
            self.logger.debug(f"[CALC] {strategy_name} (perpetuals): can_run={can_run}, has perps={'perps' in hedge_instruments}")
            return can_run
        elif strategy_type == 'hybrid':
            # Hybrid strategies might not need external instruments
            self.logger.debug(f"[CALC] {strategy_name} (hybrid): can_run=True (always)")
            return True
        else:
            self.logger.debug(f"[CALC] {strategy_name} ({strategy_type}): can_run=True (default)")
            return True
    
    def _score_opportunity(self, opportunity: Dict) -> float:
        """
        Score an opportunity for ranking.
        
        Prioritizes:
        1. True arbitrage
        2. High probability of profit
        3. Maximum profit potential
        """
        score = 0
        
        # True arbitrage gets highest priority
        if opportunity.get('is_true_arbitrage', False):
            score += TRUE_ARBITRAGE_BONUS
        
        # Funding arbitrage is also valuable
        if opportunity.get('is_funding_arbitrage', False):
            score += NEAR_ARBITRAGE_BONUS
        
        # Probability of profit
        prob_profit = opportunity.get('probability_profit', 0.5)
        score += prob_profit * 1000
        
        # Maximum profit (capped at 1000 to prevent outsized influence)
        max_profit = opportunity.get('max_profit', 0)
        score += min(max_profit, 1000)
        
        # Penalize high risk
        max_loss = opportunity.get('max_loss', 0)
        if max_loss < 0:
            score += max_loss * 0.5  # Negative value reduces score
        
        return score
    
    def get_opportunities_by_type(
        self,
        opportunities: List[Dict],
        strategy_type: str
    ) -> List[Dict]:
        """
        Filter opportunities by strategy type.
        """
        return [
            opp for opp in opportunities
            if opp.get('strategy_type') == strategy_type
        ]
    
    def get_true_arbitrage_opportunities(
        self,
        opportunities: List[Dict]
    ) -> List[Dict]:
        """
        Filter only true arbitrage opportunities.
        """
        return [
            opp for opp in opportunities
            if opp.get('is_true_arbitrage', False)
        ]
    
    def summarize_opportunities(self, opportunities: List[Dict]) -> Dict:
        """
        Create summary statistics of opportunities.
        """
        summary = {
            'total_count': len(opportunities),
            'by_type': {},
            'true_arbitrage_count': 0,
            'avg_max_profit': 0,
            'best_opportunity': None
        }
        
        # Count by type
        for opp in opportunities:
            strategy_type = opp.get('strategy_type', 'unknown')
            summary['by_type'][strategy_type] = summary['by_type'].get(strategy_type, 0) + 1
            
            if opp.get('is_true_arbitrage', False):
                summary['true_arbitrage_count'] += 1
        
        # Calculate averages
        if opportunities:
            total_profit = sum(opp.get('max_profit', 0) for opp in opportunities)
            summary['avg_max_profit'] = total_profit / len(opportunities)
            summary['best_opportunity'] = opportunities[0]  # Already sorted
        
        return summary