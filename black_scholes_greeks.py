"""
Black-Scholes Greeks Calculator

Provides accurate calculation of option Greeks (delta, gamma, vega, theta, rho)
for use in risk penalty calculations and hedging strategies.

Scaling Convention:
By default, vega and rho are scaled to represent the price change for a 1 percentage
point move (e.g., volatility from 30% to 31%, or interest rate from 5% to 6%).
This is the standard convention used by most trading desks. The raw mathematical
derivatives (representing 100% moves) can be obtained by setting scale_1pct=False.

Academic foundation:
- Black & Scholes (1973): "The Pricing of Options and Corporate Liabilities"
  https://www.jstor.org/stable/1831029
  Original derivation of the Black-Scholes formula
  
- Merton (1973): "Theory of Rational Option Pricing"
  https://www.jstor.org/stable/3003143
  Extensions including dividends and early exercise
  
- Hull (2022): "Options, Futures, and Other Derivatives" (11th Edition)
  Comprehensive textbook with Greeks formulas used here
  
- Rubinstein & Reiner (1991): "Breaking Down the Barriers"
  https://www.risk.net/derivatives/equity-derivatives/1500593/breaking-down-barriers
  Binary option pricing formulas
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple

# --- begin patch: safe constants & epsilons ---
try:
    from config_manager import DAYS_PER_YEAR  # existing behavior if present
except Exception:
    DAYS_PER_YEAR = 365.0  # safe fallback for tests/standalone use

_MIN_SIGMA = 1e-12
_MIN_TIME = 1e-12
# --- end patch ---


class BlackScholesGreeks:
    """
    Calculate Black-Scholes option prices and Greeks.
    
    All calculations assume European options on non-dividend paying assets.
    """
    
    @staticmethod
    def calculate_d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters for Black-Scholes formula.
        
        Args:
            S: Current spot price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility (annualized)
            T: Time to expiry (in years)
            
        Returns:
            Tuple of (d1, d2)
        """
        if T <= 0:
            # Preserve expired-option semantics for downstream logic
            if S > K:
                return float('inf'), float('inf')
            elif S < K:
                return float('-inf'), float('-inf')
            else:
                return 0.0, 0.0

        # Clamp to avoid divide-by-zero and runtime warnings
        sigma = max(float(sigma), _MIN_SIGMA)
        T = max(float(T), _MIN_TIME)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(d1), float(d2)
    
    @staticmethod
    def call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate Black-Scholes call option price."""
        if T <= 0:
            return max(S - K, 0)
            
        d1, d2 = BlackScholesGreeks.calculate_d1_d2(S, K, r, sigma, T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    @staticmethod
    def put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate Black-Scholes put option price."""
        if T <= 0:
            return max(K - S, 0)
            
        d1, d2 = BlackScholesGreeks.calculate_d1_d2(S, K, r, sigma, T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    @staticmethod
    def call_greeks(S: float, K: float, r: float, sigma: float, T: float, 
                    scale_1pct: bool = True, clip_gamma: bool = False) -> Dict[str, float]:
        """
        Calculate all Greeks for a call option.
        
        Args:
            S: Current spot price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility (annualized)
            T: Time to expiry (in years)
            scale_1pct: If True, vega and rho are scaled for 1% moves (default).
                       If False, returns raw derivatives.
            clip_gamma: If True, cap gamma for short-dated options (default False).
        
        Returns:
            Dictionary with keys: delta, gamma, vega, theta, rho
            
        Note:
            - theta: dollar change per day (negative for long positions)
            
            When scale_1pct=True (default):
            - vega: dollar change per 1 percentage point volatility move
            - rho: dollar change per 1 percentage point interest rate move
            
            When scale_1pct=False:
            - vega: dollar change per 1.0 (100%) volatility move
            - rho: dollar change per 1.0 (100%) interest rate move
        """
        if T <= 0:
            # Expired option Greeks
            if S > K:
                return {'delta': 1.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
            else:
                return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
        
        d1, d2 = BlackScholesGreeks.calculate_d1_d2(S, K, r, sigma, T)
        
        # Scaling factor for vega and rho
        factor = 0.01 if scale_1pct else 1.0
        
        # Delta: ∂C/∂S = N(d1)
        delta = norm.cdf(d1)
        
        # Gamma: ∂²C/∂S² = n(d1) / (S * σ * √T)
        # For very small T or sigma, gamma can explode. Cap it for stability.
        # Use clamped values for numerical stability
        clamped_sigma = max(sigma, _MIN_SIGMA)
        clamped_T = max(T, _MIN_TIME)
        sqrt_T = np.sqrt(clamped_T)
        gamma = norm.pdf(d1) / (S * clamped_sigma * sqrt_T)
        # Cap gamma for sub-day expiries to avoid numerical issues
        if clip_gamma and T < 0.003:  # Less than ~1 day
            max_gamma = 1.0 / (S * 0.01)  # Cap based on 1% of spot
            gamma = min(gamma, max_gamma)
        
        # Vega: ∂C/∂σ = S * n(d1) * √T
        # Vega naturally goes to 0 as T→0, but use stable sqrt
        vega = S * norm.pdf(d1) * sqrt_T * factor
        
        # Theta: -∂C/∂T
        # Note: Theta is typically quoted per day; divide by DAYS_PER_YEAR
        theta_annual = -(S * norm.pdf(d1) * clamped_sigma / (2 * sqrt_T) + 
                        r * K * np.exp(-r * T) * norm.cdf(d2))
        theta = theta_annual / DAYS_PER_YEAR
        
        # Rho: ∂C/∂r = K * T * e^(-rT) * N(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) * factor
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    @staticmethod
    def put_greeks(S: float, K: float, r: float, sigma: float, T: float,
                   scale_1pct: bool = True, clip_gamma: bool = False) -> Dict[str, float]:
        """
        Calculate all Greeks for a put option.
        
        Args:
            S: Current spot price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility (annualized)
            T: Time to expiry (in years)
            scale_1pct: If True, vega and rho are scaled for 1% moves (default).
                       If False, returns raw derivatives.
            clip_gamma: If True, cap gamma for short-dated options (default False).
        
        Returns:
            Dictionary with keys: delta, gamma, vega, theta, rho
            
        Note:
            - theta: dollar change per day (negative for long positions)
            
            When scale_1pct=True (default):
            - vega: dollar change per 1 percentage point volatility move
            - rho: dollar change per 1 percentage point interest rate move
            
            When scale_1pct=False:
            - vega: dollar change per 1.0 (100%) volatility move
            - rho: dollar change per 1.0 (100%) interest rate move
        """
        if T <= 0:
            # Expired option Greeks
            if S < K:
                return {'delta': -1.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
            else:
                return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
        
        d1, d2 = BlackScholesGreeks.calculate_d1_d2(S, K, r, sigma, T)
        
        # Scaling factor for vega and rho
        factor = 0.01 if scale_1pct else 1.0
        
        # Delta: ∂P/∂S = N(d1) - 1
        delta = norm.cdf(d1) - 1
        
        # Gamma: ∂²P/∂S² = n(d1) / (S * σ * √T) [same as call]
        # For very small T or sigma, gamma can explode. Cap it for stability.
        # Use clamped values for numerical stability
        clamped_sigma = max(sigma, _MIN_SIGMA)
        clamped_T = max(T, _MIN_TIME)
        sqrt_T = np.sqrt(clamped_T)
        gamma = norm.pdf(d1) / (S * clamped_sigma * sqrt_T)
        # Cap gamma for sub-day expiries to avoid numerical issues
        if clip_gamma and T < 0.003:  # Less than ~1 day
            max_gamma = 1.0 / (S * 0.01)  # Cap based on 1% of spot
            gamma = min(gamma, max_gamma)
        
        # Vega: ∂P/∂σ = S * n(d1) * √T [same as call]
        # Vega naturally goes to 0 as T→0, but use stable sqrt
        vega = S * norm.pdf(d1) * sqrt_T * factor
        
        # Theta: -∂P/∂T (per-day)
        theta_annual = -(S * norm.pdf(d1) * clamped_sigma / (2 * sqrt_T) - 
                        r * K * np.exp(-r * T) * norm.cdf(-d2))
        theta = theta_annual / DAYS_PER_YEAR
        
        # Rho: ∂P/∂r = -K * T * e^(-rT) * N(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * factor
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    @staticmethod
    def portfolio_greeks(positions: list, scale_1pct: bool = True) -> Dict[str, float]:
        """
        Aggregate Greeks for a portfolio.

        Each position dict may include:
          - 'type': 'call' or 'put' (case-insensitive)
          - 'position': number of units (contracts or shares)
          - optional 'contract_size': multiplier per unit (e.g., 100 for equity options)
          - 'S', 'K', 'r', 'sigma', 'T': parameters (str or float accepted)

        Returns dollar Greeks per position * contract_size (if provided).
        """
        total = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}

        for pos in positions:
            opt_type = str(pos.get('type', 'call')).strip().lower()
            qty = float(pos.get('position', 0.0))
            mult = float(pos.get('contract_size', 1.0))

            S = float(pos['S']); K = float(pos['K']); r = float(pos['r'])
            sigma = float(pos['sigma']); T = float(pos['T'])

            if opt_type == 'call':
                g = BlackScholesGreeks.call_greeks(S, K, r, sigma, T, scale_1pct=scale_1pct)
            elif opt_type == 'put':
                g = BlackScholesGreeks.put_greeks(S, K, r, sigma, T, scale_1pct=scale_1pct)
            else:
                raise ValueError(f"Unknown option type: {pos.get('type')}")

            weight = qty * mult
            for k, v in g.items():
                total[k] += weight * v

        return total
    
    @staticmethod
    def binary_option_vega(S: float, K: float, r: float, sigma: float, T: float,
                          scale_1pct: bool = True) -> float:
        """
        Vega for a cash-or-nothing digital call paying 1{S_T > K}.
        D = e^{-rT} N(d2)  ⇒  ∂D/∂σ = -e^{-rT} φ(d2) * (d1/σ).

        Note:
            We use a σ clamp consistent with calculate_d1_d2 to avoid
            inconsistency when σ is extremely small.
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        # Use the same effective sigma as in d1,d2 calculation
        sigma_eff = max(float(sigma), _MIN_SIGMA)
        d1, d2 = BlackScholesGreeks.calculate_d1_d2(S, K, r, sigma_eff, T)

        factor = 0.01 if scale_1pct else 1.0
        return float(-np.exp(-r * T) * norm.pdf(d2) * (d1 / sigma_eff) * factor)


def calculate_vega_penalty_accurate(
    options_data: list,
    current_spot: float,
    risk_free_rate: float = 0.05,
    default_volatility: float = 0.6,
    scale_1pct: bool = True,  # True = per 1% move (your documented default)
) -> float:
    """
    Accurate portfolio vega penalty. Returns |vega| aggregated across positions.
    Units:
      - scale_1pct=True: $ change per 1 percentage point vol move
      - scale_1pct=False: $ change per 1.00 (100%) vol move
    """
    bs = BlackScholesGreeks()

    positions = []
    for opt in options_data:
        # Strike and time
        strike = float(opt.get('K', opt.get('strike', current_spot)))
        if 'time_to_expiry_years' in opt:
            T = float(opt['time_to_expiry_years'])
        else:
            dte = float(opt.get('days_to_expiry', 30))
            T = dte / DAYS_PER_YEAR
        T = max(T, _MIN_TIME)

        # Vol and type/size
        sigma = float(opt.get('sigma', opt.get('implied_volatility', default_volatility)))
        opt_type = str(opt.get('type', opt.get('option_type', 'call'))).strip().lower()
        qty = float(opt.get('position', opt.get('quantity', 1)))
        mult = float(opt.get('contract_size', 1.0))  # use 100 for equity options if desired

        positions.append({
            'type': opt_type, 'position': qty, 'contract_size': mult,
            'S': float(current_spot), 'K': strike, 'r': float(risk_free_rate),
            'sigma': sigma, 'T': T,
        })

    portfolio = bs.portfolio_greeks(positions, scale_1pct=scale_1pct)
    return abs(portfolio['vega'])


# -------------------------------
# Self-tests (run with: python black_scholes_greeks.py)
# -------------------------------
def _run_self_tests() -> None:
    import math
    import numpy as _np

    bs = BlackScholesGreeks()

    # Canonical ATM case
    S, K, r, sigma, T = 100.0, 100.0, 0.00, 0.20, 1.0

    # --- Vega units: per 1%-pt (primary assertion from your notes) ---
    g_scaled = bs.call_greeks(S, K, r, sigma, T, scale_1pct=True)
    g_raw    = bs.call_greeks(S, K, r, sigma, T, scale_1pct=False)
    assert _np.isclose(g_scaled['vega'] * 100.0, g_raw['vega'], rtol=1e-12, atol=0.0), \
        "Vega (per 1%-pt) * 100 should equal raw vega."
    # Do the same for rho
    assert _np.isclose(g_scaled['rho'] * 100.0, g_raw['rho'], rtol=1e-12, atol=0.0), \
        "Rho (per 1%-pt) * 100 should equal raw rho."

    # --- Known price (sanity) ---
    cp = bs.call_price(S, K, r, sigma, T)
    pp = bs.put_price(S, K, r, sigma, T)
    # These are classic reference values for the ATM, r=0, σ=20%, T=1 case
    assert _np.isclose(cp, 7.965567455405804, rtol=1e-12), "Call price mismatch."
    assert _np.isclose(pp, cp, rtol=1e-12), "Put price should equal call price when r=0, ATM."

    # --- Put-call parity & Greeks relationships ---
    # Price parity with q=0: C - P = S - K e^{-rT}
    assert _np.isclose(cp - pp, S - K * _np.exp(-r * T), rtol=1e-12)
    # Delta: Δ_call - Δ_put = 1 (q=0)
    pg_scaled = bs.put_greeks(S, K, r, sigma, T, scale_1pct=True)
    assert _np.isclose(g_scaled['delta'] - pg_scaled['delta'], 1.0, rtol=1e-12)
    # Gamma equality, Vega equality (calls = puts)
    assert _np.isclose(g_scaled['gamma'], pg_scaled['gamma'], rtol=1e-12)
    assert _np.isclose(g_scaled['vega'], pg_scaled['vega'], rtol=1e-12)

    # --- Theta per day units: finite-difference check ---
    dT = 1.0 / DAYS_PER_YEAR
    theta_day = g_scaled['theta']
    V0 = bs.call_price(S, K, r, sigma, T)
    V1 = bs.call_price(S, K, r, sigma, T - dT)
    theta_fd = V1 - V0
    assert _np.isclose(theta_day, theta_fd, rtol=1e-3, atol=1e-6), \
        "Per-day theta should match 1-day roll-down FD."

    # --- Vega finite-diff: test both units (raw vs per 1%-pt) ---
    bump = 0.01  # absolute vol bump = 1 percentage point
    V_up = bs.call_price(S, K, r, sigma + bump, T)
    V_dn = bs.call_price(S, K, r, sigma - bump, T)

    # Raw vega (per 1.00 = 100% vol unit)
    vega_fd_raw = (V_up - V_dn) / (2 * bump)
    assert _np.isclose(vega_fd_raw, g_raw['vega'], rtol=1e-5, atol=1e-8), \
        "Raw vega should match FD with ±0.01 vol bump."

    # Scaled vega (per 1%-pt) = raw * 0.01
    vega_fd_scaled = vega_fd_raw * 0.01
    assert _np.isclose(vega_fd_scaled, g_scaled['vega'], rtol=1e-5, atol=1e-8), \
        "Per 1%-pt vega should match FD after 0.01 scaling."

    # --- Rho finite-diff: test both units too ---
    r_bump = 0.01  # 1%-pt rate bump
    C_up = bs.call_price(S, K, r + r_bump, sigma, T)
    C_dn = bs.call_price(S, K, r - r_bump, sigma, T)
    rho_fd_raw = (C_up - C_dn) / (2 * r_bump)
    rho_fd_scaled = rho_fd_raw * 0.01
    assert _np.isclose(rho_fd_raw, g_raw['rho'], rtol=1e-4, atol=1e-6)
    assert _np.isclose(rho_fd_scaled, g_scaled['rho'], rtol=1e-4, atol=1e-6)

    # --- Digital vega: analytic vs FD ---
    def _digital_price(S_, K_, r_, sigma_, T_):
        d1 = ( _np.log(S_ / K_) + (r_ + 0.5 * sigma_**2) * T_ ) / (sigma_ * _np.sqrt(T_))
        d2 = d1 - sigma_ * _np.sqrt(T_)
        return float(_np.exp(-r_ * T_) * norm.cdf(d2))

    def _digital_vega_fd(S_, K_, r_, sigma_, T_, eps=1e-5):
        return (_digital_price(S_, K_, r_, sigma_ + eps, T_) -
                _digital_price(S_, K_, r_, sigma_ - eps, T_)) / (2 * eps)

    S2, K2, r2, sigma2, T2 = 100.0, 95.0, 0.03, 0.25, 0.5
    dv_analytic = BlackScholesGreeks.binary_option_vega(S2, K2, r2, sigma2, T2, scale_1pct=False)
    dv_fd = _digital_vega_fd(S2, K2, r2, sigma2, T2)
    assert _np.isclose(dv_analytic, dv_fd, rtol=1e-6, atol=1e-10), \
        "Digital vega analytic should match FD."

    # --- Portfolio aggregation units ---
    pos = [{'type': 'call', 'position': 2, 'contract_size': 100,
            'S': S, 'K': K, 'r': r, 'sigma': sigma, 'T': T}]
    totals = bs.portfolio_greeks(pos, scale_1pct=True)
    # Expect totals ~ (qty * multiplier) times single-option greeks
    factor = 2 * 100
    for key in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        assert _np.isclose(totals[key], factor * g_scaled[key], rtol=1e-12, atol=1e-10), \
            f"Portfolio {key} mismatch."

    print("All Black–Scholes Greek tests PASSED ✔")

if __name__ == "__main__":
    _run_self_tests()