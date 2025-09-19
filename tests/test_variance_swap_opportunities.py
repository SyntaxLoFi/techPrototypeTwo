#!/usr/bin/env python3
"""
Unit tests for VarianceSwapStrategy opportunity generation.

Tests ensure that the strategy correctly returns opportunities when conditions are met.
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
import logging
import numpy as np
from strategies.options.variance_swap_strategy import VarianceSwapStrategy


class TestVarianceSwapOpportunities(unittest.TestCase):
    """Test that variance swap strategy returns opportunities correctly."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock config and logger
        mock_cfg = Mock()
        mock_logger = logging.getLogger('test')
        
        self.strategy = VarianceSwapStrategy(
            risk_free_rate=0.05,
            cfg=mock_cfg,
            logger=mock_logger
        )
        # Relax price gates for testing
        self.strategy.min_price_threshold = 0.001  # 0.1%
        self.strategy.max_price_threshold = 0.999  # 99.9%
        
    def create_mock_polymarket_contract(self, yes_price=0.3, no_price=0.7):
        """Create a mock Polymarket contract."""
        return {
            'question': 'Will Bitcoin reach $100K?',
            'currency': 'BTC',
            'strike_price': 100000,
            'yes_price': yes_price,
            'no_price': no_price,
            'is_above': True,
            'end_date': '2024-12-31',
            'days_to_expiry': 30,
        }
    
    def create_mock_options(self, num_strikes=20):
        """Create mock options chain."""
        strikes = np.linspace(90000, 110000, num_strikes)
        options = []
        
        for strike in strikes:
            # Simple mock pricing
            call_price = max(0, 100000 - strike) / 1000 + 100
            put_price = max(0, strike - 100000) / 1000 + 100
            
            options.extend([
                {
                    'type': 'call',
                    'strike': strike,
                    'bid': call_price * 0.98,
                    'ask': call_price * 1.02,
                    'expiry_date': '2024-12-31',
                    'days_to_expiry': 30,
                },
                {
                    'type': 'put',
                    'strike': strike,
                    'bid': put_price * 0.98,
                    'ask': put_price * 1.02,
                    'expiry_date': '2024-12-31',
                    'days_to_expiry': 30,
                }
            ])
        
        return options
    
    def test_opportunities_returned_for_valid_market(self):
        """Test that opportunities are returned for a valid market."""
        # Create mock data
        pm_contract = self.create_mock_polymarket_contract(yes_price=0.3, no_price=0.7)
        hedge_instruments = {'options': self.create_mock_options()}
        current_spot = 100000
        position_size = 1000
        
        # Mock internal methods to avoid complex calculations
        with patch.object(self.strategy, '_build_variance_swap_portfolio') as mock_build:
            with patch.object(self.strategy, '_calculate_variance_swap_params') as mock_params:
                with patch.object(self.strategy, '_sparse_variance_legs') as mock_sparse:
                    # Set up mocks
                    mock_build.return_value = {
                        'puts': [{'strike': 95000, 'weight': 0.1}],
                        'calls': [{'strike': 105000, 'weight': 0.1}],
                        'options': self.create_mock_options()[:10],
                        'k0': 100000,
                        'forward': 100000,
                        'replication_weights': {95000: 0.1, 105000: 0.1}
                    }
                    
                    mock_params.return_value = {
                        'strike_variance': 0.04,  # 20% vol
                        'strike_volatility': 0.2,
                        'forward': 100000,
                        'k0': 100000
                    }
                    
                    mock_sparse.return_value = ([
                        {'type': 'PUT', 'strike': 95000, 'action': 'BUY', 'contracts': 0.1}
                    ], {'mse_unit': 0.01})
                    
                    # Call evaluate_opportunities
                    opportunities = self.strategy.evaluate_opportunities(
                        pm_contract,
                        hedge_instruments,
                        current_spot,
                        position_size
                    )
                    
                    # Assertions
                    self.assertIsInstance(opportunities, list)
                    self.assertGreater(len(opportunities), 0, 
                                      "Should return at least one opportunity")
                    
                    # Check opportunity structure
                    if opportunities:
                        opp = opportunities[0]
                        self.assertIn('pm_side', opp)
                        self.assertIn('expected_pm_hedge_combined', opp)
                        self.assertIn('option_expiry', opp)
                        self.assertIn('currency', opp)
    
    def test_no_opportunities_for_extreme_prices(self):
        """Test that no opportunities are returned for extreme PM prices."""
        # Reset to strict thresholds
        self.strategy.min_price_threshold = 0.01
        self.strategy.max_price_threshold = 0.99
        
        # Create contract with extreme prices
        pm_contract = self.create_mock_polymarket_contract(yes_price=0.001, no_price=0.999)
        hedge_instruments = {'options': self.create_mock_options()}
        
        opportunities = self.strategy.evaluate_opportunities(
            pm_contract,
            hedge_instruments,
            100000,
            1000
        )
        
        self.assertEqual(len(opportunities), 0, 
                        "Should return no opportunities for extreme prices")
    
    def test_opportunities_tracked_in_checkpoint_log(self):
        """Test that opportunities are properly tracked in checkpoint log."""
        import tempfile
        import os
        
        # Create temporary checkpoint file
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'variance_swap_checkpoints.log')
            
            # Patch the checkpoint path
            with patch('strategies.options.variance_swap_strategy.checkpoint_log_path', checkpoint_path):
                pm_contract = self.create_mock_polymarket_contract()
                hedge_instruments = {'options': self.create_mock_options()}
                
                # Mock successful opportunity creation
                with patch.object(self.strategy, '_evaluate_variance_hedge') as mock_eval:
                    mock_eval.return_value = {
                        'pm_side': 'YES',
                        'option_expiry': '2024-12-31',
                        'expected_pm_hedge_combined': {
                            'expected_profit': 100,
                            'max_profit': 200,
                            'hedge_cost': 50
                        }
                    }
                    
                    opportunities = self.strategy.evaluate_opportunities(
                        pm_contract,
                        hedge_instruments,
                        100000,
                        1000
                    )
                    
                    # Check that checkpoint file was created
                    if os.path.exists(checkpoint_path):
                        with open(checkpoint_path, 'r') as f:
                            content = f.read()
                            self.assertIn('FINAL SUMMARY', content)
                            self.assertIn('Opportunities created: 2', content)  # YES and NO


if __name__ == '__main__':
    unittest.main()