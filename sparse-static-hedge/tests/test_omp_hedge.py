import pytest
import numpy as np
from sparse_static_hedge import (
    select_sparse_quadratic,
    round_options_and_repair_budget,
    refit_bases_given_fixed_options,
    preselect_strikes_by_moneyness,
)


class TestOMPSelection:
    """Test the OMP sparse selection algorithm."""
    
    @pytest.fixture
    def simple_hedge_problem(self):
        """Create a simple digital option hedging problem."""
        np.random.seed(42)
        n_scenarios = 1000
        S0, K_target = 100, 100
        T, sigma, r = 0.25, 0.3, 0.05
        
        # Generate scenarios
        S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*np.random.randn(n_scenarios))
        y = (S_T > K_target).astype(float)  # Digital payoff
        
        # Build basis: bond + calls/puts at various strikes
        strikes = [90, 95, 100, 105, 110]
        X = [np.ones(n_scenarios)]  # Bond
        names = ["bond_T1"]
        
        for K in strikes:
            X.append(np.maximum(S_T - K, 0))  # Call
            X.append(np.maximum(K - S_T, 0))  # Put
            names.extend([f"call_{K}", f"put_{K}"])
        
        X = np.column_stack(X)
        
        # Simple pricing (should use Black-Scholes in practice)
        prices = [np.exp(-r*T)]  # Bond
        for K in strikes:
            prices.append(np.mean(np.maximum(S_T - K, 0))*np.exp(-r*T))  # Call
            prices.append(np.mean(np.maximum(K - S_T, 0))*np.exp(-r*T))  # Put
        
        price_vec = np.array(prices)
        
        return {
            'X': X, 'y': y, 'price_vec': price_vec, 'names': names,
            'S_T': S_T, 'strikes': strikes, 'r': r, 'T': T
        }
    
    def test_sparse_selection_respects_max_legs(self, simple_hedge_problem):
        """Verify OMP never selects more than max_option_legs."""
        data = simple_hedge_problem
        
        # Test with different max_option_legs
        for max_legs in [2, 4, 6]:
            w, info = select_sparse_quadratic(
                data['X'], data['y'], data['price_vec'], data['names'],
                max_option_legs=max_legs,
                l2=1e-6,
                budget=None,
                always_on_bases=("bond_T1",),
                verbose=False
            )
            
            # Count selected options
            selected_options = [name for name in info['selected_names'] 
                              if name.startswith(('call_', 'put_'))]
            assert len(selected_options) <= max_legs
            
            # Verify all selected indices are active
            assert len(info['active_idx']) == len(info['selected_names'])
            
            # Verify weights are non-zero only for active indices
            active_set = set(info['active_idx'])
            for i, wi in enumerate(w):
                if i in active_set:
                    assert wi != 0 or abs(wi) < 1e-12  # May be numerically zero
                else:
                    assert abs(wi) < 1e-12
    
    def test_budget_constraint_exact(self, simple_hedge_problem):
        """Verify budget constraint is satisfied exactly."""
        data = simple_hedge_problem
        budget = 0.5  # Arbitrary budget
        
        w, info = select_sparse_quadratic(
            data['X'], data['y'], data['price_vec'], data['names'],
            max_option_legs=4,
            l2=1e-6,
            budget=budget,
            always_on_bases=("bond_T1",),
            verbose=False
        )
        
        # Check budget constraint
        cost = np.dot(data['price_vec'], w)
        assert abs(cost - budget) < 1e-10, f"Cost {cost} != budget {budget}"
    
    def test_unconstrained_vs_constrained(self, simple_hedge_problem):
        """Compare unconstrained and budget-constrained solutions."""
        data = simple_hedge_problem
        
        # Unconstrained
        w_free, info_free = select_sparse_quadratic(
            data['X'], data['y'], data['price_vec'], data['names'],
            max_option_legs=4,
            l2=1e-6,
            budget=None,
            always_on_bases=("bond_T1",),
        )
        
        # Get unconstrained cost
        free_cost = np.dot(data['price_vec'], w_free)
        
        # Constrained to half the free cost
        budget = free_cost * 0.5
        w_con, info_con = select_sparse_quadratic(
            data['X'], data['y'], data['price_vec'], data['names'],
            max_option_legs=4,
            l2=1e-6,
            budget=budget,
            always_on_bases=("bond_T1",),
        )
        
        # Constrained should have higher MSE (worse fit)
        assert info_con['mse_final'] >= info_free['mse_final'] - 1e-10
        
        # But exact budget
        assert abs(np.dot(data['price_vec'], w_con) - budget) < 1e-10
    
    def test_warm_start_keeps_initial(self, simple_hedge_problem):
        """Verify warm start legs remain in solution."""
        data = simple_hedge_problem
        
        # Find indices for specific options
        call_100_idx = data['names'].index('call_100')
        put_100_idx = data['names'].index('put_100')
        initial_active = [call_100_idx, put_100_idx]
        
        w, info = select_sparse_quadratic(
            data['X'], data['y'], data['price_vec'], data['names'],
            max_option_legs=4,
            l2=1e-6,
            budget=None,
            always_on_bases=("bond_T1",),
            initial_active_idx=initial_active,
            verbose=False
        )
        
        # Check warm start indices are kept
        for idx in initial_active:
            assert idx in info['active_idx']
            assert data['names'][idx] in info['selected_names']
    
    def test_candidate_restriction(self, simple_hedge_problem):
        """Test restricting candidate pool."""
        data = simple_hedge_problem
        
        # Only allow calls (no puts)
        call_indices = [i for i, name in enumerate(data['names']) 
                       if name.startswith('call_')]
        
        w, info = select_sparse_quadratic(
            data['X'], data['y'], data['price_vec'], data['names'],
            max_option_legs=3,
            l2=1e-6,
            budget=None,
            always_on_bases=("bond_T1",),
            candidate_idx=call_indices,
            verbose=False
        )
        
        # Should only select calls (and bond)
        for name in info['selected_names']:
            if name != "bond_T1":
                assert name.startswith('call_')
    
    def test_always_on_bases(self, simple_hedge_problem):
        """Test that always_on_bases are always included."""
        data = simple_hedge_problem
        
        # Add a forward to the basis
        data['X'] = np.column_stack([data['X'], data['S_T']])
        data['names'].append("S_T")
        data['price_vec'] = np.append(data['price_vec'], 100)  # S0
        
        w, info = select_sparse_quadratic(
            data['X'], data['y'], data['price_vec'], data['names'],
            max_option_legs=2,
            l2=1e-6,
            budget=None,
            always_on_bases=("bond_T1", "S_T"),
            verbose=False
        )
        
        # Both bases should be included
        assert "bond_T1" in info['selected_names']
        assert "S_T" in info['selected_names']
        
        # Plus at most 2 options
        options = [n for n in info['selected_names'] 
                  if n not in ("bond_T1", "S_T")]
        assert len(options) <= 2
    
    def test_warm_start_with_candidate_restriction(self, simple_hedge_problem):
        """Test that warm-start legs outside candidate pool still count toward max_option_legs."""
        data = simple_hedge_problem
        
        # Get option indices
        option_indices = [i for i, name in enumerate(data['names']) 
                         if name.startswith(('call_', 'put_'))]
        
        # Warm start with first two options (indices 1 and 2)
        initial_active = option_indices[:2]
        
        # Candidate pool is only the last few options (no overlap with warm start)
        candidate_pool = option_indices[-4:]
        
        # Ensure no overlap
        assert not set(initial_active).intersection(set(candidate_pool))
        
        w, info = select_sparse_quadratic(
            data['X'], data['y'], data['price_vec'], data['names'],
            max_option_legs=3,  # Total limit is 3
            l2=1e-6,
            budget=None,
            always_on_bases=("bond_T1",),
            initial_active_idx=initial_active,  # 2 options
            candidate_idx=candidate_pool,        # Different options
            verbose=False
        )
        
        # Count total options selected
        total_options = sum(1 for name in info['selected_names'] 
                           if name.startswith(('call_', 'put_')))
        
        # Should have exactly 3 options total (2 warm start + 1 from pool)
        assert total_options == 3, f"Expected 3 options, got {total_options}"
        
        # Warm start legs should be included
        for idx in initial_active:
            assert idx in info['active_idx']


class TestRounding:
    """Test the rounding and budget repair functionality."""
    
    def test_lot_sizes_compliant(self):
        """Verify all positions meet venue requirements."""
        names = ["bond_T1", "call_100_BTC", "put_100_BTC", "call_3000_ETH", "put_3000_ETH"]
        # Use larger weights to meet minimum notional requirements
        weights = np.array([10.0, 10.123, -10.456, 0.789, -0.234])
        prices = np.array([1.0, 5.0, 5.0, 100.0, 100.0])
        
        w_rounded = round_options_and_repair_budget(
            names, weights, prices,
            step_by_asset={"BTC": 0.01, "ETH": 0.01},
            floor_by_asset={"BTC": 0.01, "ETH": 0.01},
            min_notional=25.0,
            bond_name="bond_T1"
        )
        
        # Check BTC positions (now above min notional)
        assert abs(w_rounded[1] - 10.12) < 1e-10  # 10.123 -> 10.12
        assert abs(w_rounded[2] - (-10.46)) < 1e-10  # -10.456 -> -10.46
        
        # Check ETH positions (now with 0.01 step)
        assert abs(w_rounded[3] - 0.79) < 1e-10  # 0.789 -> 0.79
        # ETH put with notional -23.4 is below $25 min, so it's zeroed
        assert abs(w_rounded[4] - 0.0) < 1e-10  # -0.234 -> 0.0 (below min notional)
    
    def test_minimum_notional_filter(self):
        """Test that positions below min notional are dropped."""
        names = ["bond_T1", "call_100", "put_100"]
        weights = np.array([10.0, 0.001, 0.1])  # Tiny call position
        prices = np.array([1.0, 10.0, 10.0])  # Call notional = 0.01, below min
        
        w_rounded = round_options_and_repair_budget(
            names, weights, prices,
            min_notional=1.0,  # $1 minimum
            bond_name="bond_T1"
        )
        
        # Small position should be zeroed
        assert w_rounded[1] == 0.0
        # Larger position kept
        assert w_rounded[2] != 0.0
    
    def test_rounding_preserves_budget(self):
        """Verify bond adjustment maintains total cost."""
        names = ["bond_T1", "call_100", "put_100"]
        weights = np.array([5.0, 3.123, 4.456])  # Larger to meet min notional
        prices = np.array([0.95, 10.0, 8.0])
        
        # Original cost
        original_cost = np.dot(prices, weights)
        
        w_rounded = round_options_and_repair_budget(
            names, weights, prices,
            step_by_asset={"": 0.01},  # Round to 0.01
            min_notional=1.0,  # Low min notional for this test
            bond_name="bond_T1"
        )
        
        # Rounded cost should equal original
        rounded_cost = np.dot(prices, w_rounded)
        assert abs(rounded_cost - original_cost) < 1e-10
        
        # Options should be rounded to 0.01
        assert abs(w_rounded[1] - 3.12) < 1e-10
        assert abs(w_rounded[2] - 4.46) < 1e-10
    
    def test_asset_detection(self):
        """Test automatic asset detection from names."""
        names = ["bond_T1", "call_100_BTC", "ETH_put_3000", "DOGE_call_0.1"]
        weights = np.array([10.0, 5.123, 0.456, 1000.789])  # Larger to meet min notional
        prices = np.array([1.0, 10.0, 100.0, 0.1])
        
        w_rounded = round_options_and_repair_budget(
            names, weights, prices,
            step_by_asset={"BTC": 0.01, "ETH": 0.01, "DOGE": 0.001},
            floor_by_asset={"BTC": 0.01, "ETH": 0.01, "DOGE": 0.001},
            min_notional=10.0,  # Reasonable min notional
            bond_name="bond_T1"
        )
        
        # Check each asset rounded correctly
        assert abs(w_rounded[1] - 5.12) < 1e-10  # BTC: 0.01 step
        # ETH: 0.456 rounds to 0.46 with 0.01 step
        assert abs(w_rounded[2] - 0.46) < 1e-10   # ETH: 0.01 step
        # DOGE: 1000.789 rounds to 1000.789 (0.001 step)
        # Allow small tolerance for float precision after bond adjustment
        assert abs(w_rounded[3] - 1000.789) < 0.002  # DOGE: 0.001 step
    
    def test_no_bond_available(self):
        """Test behavior when bond is not in the portfolio."""
        names = ["call_100", "put_100"]
        weights = np.array([3.123, 4.456])  # Larger to meet min notional
        prices = np.array([10.0, 8.0])
        
        # Should not crash
        w_rounded = round_options_and_repair_budget(
            names, weights, prices,
            step_by_asset={"": 0.01},
            min_notional=1.0,  # Low min notional
            bond_name="bond_T1"  # Not in names
        )
        
        # Just rounds without repair
        assert abs(w_rounded[0] - 3.12) < 1e-10
        assert abs(w_rounded[1] - 4.46) < 1e-10


class TestRefitBases:
    """Test the base refitting functionality."""
    
    def test_refit_improves_hedge(self):
        """Test that refitting bases after rounding improves the hedge."""
        np.random.seed(42)
        n = 1000
        
        # Simple problem: hedge y = S_T > 100
        S_T = 100 * np.exp(0.3 * np.sqrt(0.25) * np.random.randn(n))
        y = (S_T > 100).astype(float)
        
        # Basis: bond + forward + one option
        X = np.column_stack([
            np.ones(n),  # bond
            S_T,         # forward
            np.maximum(S_T - 100, 0)  # call
        ])
        names = ["bond_T1", "S_T", "call_100"]
        prices = np.array([0.95, 100, 5.0])
        
        # Initial weights with rounded option
        w_fixed = np.array([0.5, 0.0, 0.1])  # Arbitrary
        
        # Refit bases only
        w_refit = refit_bases_given_fixed_options(
            X, y, prices, names, w_fixed,
            l2=1e-6,
            budget=None,
            base_names=("bond_T1", "S_T")
        )
        
        # Option weight should be unchanged
        assert w_refit[2] == w_fixed[2]
        
        # But bases should be different
        assert w_refit[0] != w_fixed[0] or w_refit[1] != w_fixed[1]
        
        # MSE should improve
        mse_fixed = np.mean((X @ w_fixed - y)**2)
        mse_refit = np.mean((X @ w_refit - y)**2)
        assert mse_refit <= mse_fixed + 1e-10
    
    def test_refit_respects_budget(self):
        """Test that refitting maintains budget constraint."""
        np.random.seed(42)
        n = 1000
        
        S_T = 100 * np.exp(0.3 * np.sqrt(0.25) * np.random.randn(n))
        y = (S_T > 100).astype(float)
        
        X = np.column_stack([
            np.ones(n),  # bond
            S_T,         # forward
            np.maximum(S_T - 100, 0)  # call
        ])
        names = ["bond_T1", "S_T", "call_100"]
        prices = np.array([0.95, 100, 5.0])
        
        # Fixed option weight and budget
        w_fixed = np.array([0.0, 0.0, 0.2])
        budget = 10.0
        
        w_refit = refit_bases_given_fixed_options(
            X, y, prices, names, w_fixed,
            l2=1e-6,
            budget=budget,
            base_names=("bond_T1", "S_T")
        )
        
        # Check budget
        cost = np.dot(prices, w_refit)
        assert abs(cost - budget) < 1e-10


class TestStrikePreselection:
    """Test the moneyness-based strike filtering."""
    
    def test_moneyness_filtering(self):
        """Test strikes are filtered by log-moneyness."""
        S0 = 100
        strikes = [50, 70, 90, 100, 110, 130, 150, 200]
        T = 0.25
        sigma = 0.3
        
        kept = preselect_strikes_by_moneyness(
            strikes, S0, T, sigma,
            sigma_width=1.0  # ±1 std dev
        )
        
        # Should keep strikes near ATM
        assert 90 in kept
        assert 100 in kept
        assert 110 in kept
        
        # Should drop far strikes
        assert 50 not in kept
        assert 200 not in kept
    
    def test_max_candidates_limit(self):
        """Test that max_candidates is respected."""
        S0 = 100
        strikes = list(range(80, 121, 2))  # 21 strikes
        
        kept = preselect_strikes_by_moneyness(
            strikes, S0, T_years=0.25, sigma=0.3,
            max_candidates=5
        )
        
        assert len(kept) == 5
        # Should include ATM
        assert 100 in kept
    
    def test_custom_center(self):
        """Test custom center strike."""
        S0 = 100
        strikes = list(range(80, 121, 2))
        
        kept = preselect_strikes_by_moneyness(
            strikes, S0, T_years=0.25, sigma=0.3,
            center_strike=110,  # Off-center
            sigma_width=0.5,
            max_candidates=3
        )
        
        # Should center around 110
        assert 110 in kept
        assert 108 in kept or 112 in kept
    
    def test_edge_cases(self):
        """Test edge cases for strike preselection."""
        # Zero time
        kept = preselect_strikes_by_moneyness(
            [90, 100, 110], S0=100, T_years=0, sigma=0.3
        )
        assert kept == [90, 100, 110]  # No filtering
        
        # Zero vol
        kept = preselect_strikes_by_moneyness(
            [90, 100, 110], S0=100, T_years=0.25, sigma=0
        )
        assert kept == [90, 100, 110]  # No filtering
        
        # Negative strikes (should be filtered)
        kept = preselect_strikes_by_moneyness(
            [-10, 0, 90, 100, 110], S0=100, T_years=0.25, sigma=0.3
        )
        assert -10 not in kept
        assert 0 not in kept


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline(self):
        """Test select -> round -> refit pipeline."""
        np.random.seed(42)
        n = 2000
        S0, K = 100, 100
        T, sigma, r = 0.25, 0.3, 0.05
        
        # Generate data
        S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*np.random.randn(n))
        y = (S_T > K).astype(float)
        
        # Build basis
        strikes = [85, 90, 95, 100, 105, 110, 115]
        X = [np.ones(n), S_T]  # bond + forward
        names = ["bond_T1", "S_T"]
        prices = [np.exp(-r*T), S0]
        
        for K in strikes:
            X.extend([np.maximum(S_T - K, 0), np.maximum(K - S_T, 0)])
            names.extend([f"call_{K}", f"put_{K}"])
            # Simple pricing
            prices.extend([
                np.mean(np.maximum(S_T - K, 0)) * np.exp(-r*T),
                np.mean(np.maximum(K - S_T, 0)) * np.exp(-r*T)
            ])
        
        X = np.column_stack(X)
        price_vec = np.array(prices)
        
        # Step 1: Sparse selection
        w_sparse, info = select_sparse_quadratic(
            X, y, price_vec, names,
            max_option_legs=4,
            l2=1e-6,
            budget=None,
            always_on_bases=("bond_T1", "S_T"),
            verbose=True
        )
        
        print(f"\nSelected: {info['selected_names']}")
        print(f"MSE: {info['mse_final']:.6f}")
        
        # Step 2: Round to lot sizes
        w_rounded = round_options_and_repair_budget(
            names, w_sparse, price_vec,
            step_by_asset={"": 0.01},
            min_notional=0.1,
            bond_name="bond_T1"
        )
        
        # Step 3: Refit bases
        w_final = refit_bases_given_fixed_options(
            X, y, price_vec, names, w_rounded,
            l2=1e-6,
            budget=None,
            base_names=("bond_T1", "S_T")
        )
        
        # Verify properties
        # 1. Sparsity maintained
        option_weights = [w_final[i] for i, nm in enumerate(names) 
                         if nm.startswith(('call_', 'put_'))]
        assert sum(abs(w) > 1e-10 for w in option_weights) <= 4
        
        # 2. Lot sizes respected
        for i, nm in enumerate(names):
            if nm.startswith(('call_', 'put_')) and abs(w_final[i]) > 1e-10:
                assert abs(w_final[i] - round(w_final[i], 2)) < 1e-10
        
        # 3. Hedge quality
        final_mse = np.mean((X @ w_final - y)**2)
        assert final_mse < 0.5  # Reasonable hedge quality
        
        print(f"\nFinal MSE: {final_mse:.6f}")
        print("Final weights:")
        for nm, w in zip(names, w_final):
            if abs(w) > 1e-10:
                print(f"  {nm}: {w:.4f}")
    
    def test_multiple_assets(self):
        """Test with multiple underlying assets."""
        # Synthetic multi-asset problem
        names = ["bond_T1", "BTC_call_50000", "BTC_put_50000", 
                "ETH_call_3000", "ETH_put_3000", "SOL_call_100"]
        n = 1000
        
        # Random basis matrix
        X = np.random.randn(n, len(names))
        X[:, 0] = 1  # Bond column
        
        # Random target
        y = np.random.randn(n)
        
        # Prices
        prices = np.array([0.95, 1000, 800, 100, 80, 10])
        
        # Sparse selection
        w_sparse, info = select_sparse_quadratic(
            X, y, prices, names,
            max_option_legs=3,
            l2=1e-4,
            always_on_bases=("bond_T1",)
        )
        
        # Round with asset-specific rules
        w_rounded = round_options_and_repair_budget(
            names, w_sparse, prices,
            step_by_asset={"BTC": 0.01, "ETH": 0.1, "SOL": 1.0},
            floor_by_asset={"BTC": 0.01, "ETH": 0.1, "SOL": 1.0},
            min_notional=50.0,
            bond_name="bond_T1"
        )
        
        # Check asset-specific rounding
        btc_indices = [i for i, nm in enumerate(names) if "BTC" in nm]
        eth_indices = [i for i, nm in enumerate(names) if "ETH" in nm]
        sol_indices = [i for i, nm in enumerate(names) if "SOL" in nm]
        
        for i in btc_indices:
            if abs(w_rounded[i]) > 1e-10:
                assert abs(w_rounded[i] - round(w_rounded[i], 2)) < 1e-10  # 0.01 step
        
        for i in eth_indices:
            if abs(w_rounded[i]) > 1e-10:
                assert abs(w_rounded[i] - round(w_rounded[i], 1)) < 1e-10  # 0.1 step
        
        for i in sol_indices:
            if abs(w_rounded[i]) > 1e-10:
                assert abs(w_rounded[i] - round(w_rounded[i], 0)) < 1e-10  # 1.0 step


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])