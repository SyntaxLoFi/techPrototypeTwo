# sparse-static-hedge

Dependency-light helpers for static quadratic hedging with **sparse leg selection** (Orthogonal Matching Pursuit + exact KKT refit), **lot-size rounding**, and **budget repair**.

Implements the Leung–Lorig objective:
> minimize (1/n)||X w - y||_2^2 + λ||w||_2^2, subject to p^T w = B (optional)

…but limits to ≤ L option legs, then rounds to venue increments (e.g., 0.01 BTC, 0.10 ETH) and repairs budget with the bond.

## Install (editable)
```bash
pip install -e ./sparse-static-hedge
```

## Quick start
```python
from sparse_static_hedge import select_sparse_quadratic, round_options_and_repair_budget

# Sparse selection with OMP
w_cont, info = select_sparse_quadratic(
    X, y, price_vec, names,
    max_option_legs=6, l2=1e-8, budget=0.0,
    always_on_bases=("bond_T1","S_T"),
    candidate_idx=None,       # or restrict to subset indices
    initial_active_idx=None,  # warm start with already-chosen legs
)

# Round to lot sizes and repair budget
w_final = round_options_and_repair_budget(
    names, w_cont, price_vec,
    underlyings=None,                   # optional parallel list per column
    step_by_asset={"BTC":0.01, "ETH":0.01},  # Lyra/Derive allows 0.01 increments
    floor_by_asset={"BTC":0.01, "ETH":0.01},  # Minimum position size
    min_notional=25.0,
    bond_name="bond_T1"
)
```

## API Reference

### `select_sparse_quadratic`

Greedy sparse selection (OMP) with exact KKT refit after each pick.

**Parameters:**
- `X` (ndarray): Terminal payoffs of basis columns [n_samples, n_basis]
- `y` (ndarray): Target payoff (e.g., digital option) [n_samples]
- `price_vec` (ndarray): t=0 prices of basis instruments [n_basis]
- `names` (list): Basis column names
- `max_option_legs` (int): Maximum number of option columns to select
- `l2` (float): Ridge regularization parameter
- `budget` (float or None): Budget constraint (if None, unconstrained)
- `always_on_bases` (tuple): Basis names always available (not counted toward leg cap)
- `candidate_idx` (iterable or None): Restrict option pool to these indices
- `initial_active_idx` (iterable or None): Warm start with these columns
- `verbose` (bool): Print diagnostics

**Returns:**
- `w` (ndarray): Continuous optimal weights [n_basis]
- `info` (dict): Diagnostics including active indices, MSE path, selected names

### `round_options_and_repair_budget`

Quantize option legs to venue increments and repair budget with bond adjustment.

**Parameters:**
- `names` (list): Instrument names
- `w` (ndarray): Continuous weights to round
- `price_vec` (ndarray): Instrument prices
- `underlyings` (list or None): Asset names parallel to instruments
- `asset_resolver` (dict or None): Map instrument name to asset
- `step_by_asset` (dict): Rounding increments by asset
- `floor_by_asset` (dict): Minimum position sizes by asset
- `default_step` (float): Default rounding increment
- `default_floor` (float): Default minimum position
- `min_notional` (float): Minimum notional value to keep
- `bond_name` (str): Name of bond instrument for budget repair

**Returns:**
- `wq` (ndarray): Rounded weights with budget preserved

### `refit_bases_given_fixed_options`

After rounding options, refit base columns (bond, forward) while keeping option weights fixed.

**Parameters:**
- `X`, `y`, `price_vec`, `names`: As in `select_sparse_quadratic`
- `w_fixed` (ndarray): Fixed weights (rounded options)
- `l2` (float): Ridge parameter
- `budget` (float or None): Budget constraint
- `base_names` (tuple): Names of base columns to refit

**Returns:**
- `w_new` (ndarray): Weights with refitted bases

### `preselect_strikes_by_moneyness`

Filter strikes by log-moneyness distance from center.

**Parameters:**
- `strikes` (list): Available strikes
- `S0` (float): Current spot price
- `T_years` (float): Time to expiry
- `sigma` (float): Volatility
- `center_strike` (float or None): Center for selection (default: ATM)
- `sigma_width` (float): Width in standard deviations
- `max_candidates` (int or None): Maximum strikes to keep

**Returns:**
- `kept` (list): Selected strikes sorted by proximity to center

## Mathematical Details

### OMP Algorithm
1. Start with base instruments (bond, forward)
2. Iteratively add the option most correlated with residual
3. After each addition, solve exact KKT system for all active weights
4. Stop when reaching `max_option_legs` options

### KKT System
For active set A, solve:
```
[G_A + λI    p_A] [w_A]   [c_A]
[p_A^T       0  ] [μ  ] = [B  ]
```
where G = X^T X / n, c = X^T y / n

### Lot-Size Rounding
1. Round each option to venue increment (e.g., 0.01 BTC)
2. Drop positions below floor or minimum notional
3. Adjust bond position to preserve total cost

## Example: Digital Option Hedge

```python
import numpy as np
from sparse_static_hedge import select_sparse_quadratic, round_options_and_repair_budget

# Setup: hedge a digital option paying 1 if S_T > 100
n_scenarios = 10000
S0, K_target = 100, 100
T, sigma, r = 0.25, 0.3, 0.05

# Generate scenarios
S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*np.random.randn(n_scenarios))
y = (S_T > K_target).astype(float)  # Digital payoff

# Build basis: bond + options at various strikes
strikes = [90, 95, 100, 105, 110]
X = [np.ones(n_scenarios)]  # Bond
names = ["bond_T1"]
prices = [np.exp(-r*T)]

for K in strikes:
    X.append(np.maximum(S_T - K, 0))  # Call
    X.append(np.maximum(K - S_T, 0))  # Put
    names.extend([f"call_{K}", f"put_{K}"])
    # Simplified pricing (use Black-Scholes in practice)
    prices.extend([np.mean(np.maximum(S_T - K, 0))*np.exp(-r*T),
                   np.mean(np.maximum(K - S_T, 0))*np.exp(-r*T)])

X = np.column_stack(X)
price_vec = np.array(prices)

# Sparse selection
w_cont, info = select_sparse_quadratic(
    X, y, price_vec, names,
    max_option_legs=4,
    l2=1e-6,
    budget=None,  # Unconstrained
    always_on_bases=("bond_T1",),
    verbose=True
)

print(f"Selected: {info['selected_names']}")
print(f"MSE: {info['mse_final']:.6f}")

# Round and repair
w_final = round_options_and_repair_budget(
    names, w_cont, price_vec,
    step_by_asset={"": 0.01},  # Generic asset
    min_notional=10.0,
    bond_name="bond_T1"
)

print(f"Final weights: {dict(zip(names, w_final))}")
```

## License

MIT