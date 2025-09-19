"""
Base Options Strategy - Abstract base for all options-based strategies

Extends BaseStrategy with options-specific functionality.
"""

from typing import List, Dict, Optional, Tuple, Literal, Any
import numpy as np
from scipy.stats import norm
from datetime import datetime, timezone, timedelta
import logging
try:
    from ..base_strategy import BaseStrategy
except (ImportError, ValueError):
    from base_strategy import BaseStrategy
from config_manager import SLIPPAGE_BPS, DAYS_PER_YEAR
from black_scholes_greeks import BlackScholesGreeks
#
# Shared option key helpers:
#
try:
    # preferred: project-level utils package
    from utils.opt_keys import normalize_type as _normalize_opt_type, is_valid_quote as _is_valid_quote
except Exception:
    # fallback shim keeps strategies runnable even if import path differs
    def _normalize_opt_type(t):
        t = (t or "").strip().lower()
        if t in ("c", "call"):
            return "call"
        if t in ("p", "put"):
            return "put"
        return None
    def _is_valid_quote(row):
        try:
            bid = float(row.get("bid", 0.0) or 0.0)
            ask = float(row.get("ask", 0.0) or 0.0)
        except Exception:
            return False
        return ask > 0.0 and 0.0 <= bid <= ask

logger = logging.getLogger(__name__)

# Type alias for option types
OptionType = Literal['call', 'put']

SECONDS_PER_YEAR = 365.0 * 86400.0


class BaseOptionsStrategy(BaseStrategy):
    # Optional external Greeks provider (your black_scholes_greeks.py via our wrapper is default)
    bs_greeks_fn = None
    # Theta units from bs_greeks_fn: 'per_year' (default) or 'per_day'
    greeks_theta_unit = 'per_year'
    """
    Base class for all options-based hedging strategies.
    
    Provides common options functionality:
    - Black-Scholes pricing
    - Greeks calculations
    - Options chain filtering
    - P&L calculations
    
    Units and Conventions:
    - theta: Annualized (per-year). For per-day theta, divide by DAYS_PER_YEAR
    - vega: Per 1.00 absolute vol change. For per-1% vol change, multiply by 0.01
    - risk_free_rate: Annualized continuous compounding rate (used in exp(-rT))
    - volatility: Annualized volatility (typically 0.0 to 3.0)
    
    Numerical Safeguards:
    - sigma (volatility) floor: 1e-8
    - time floor: 1e-12
    - spot price floor: 1e-12
    - All prices clamped to non-negative values after slippage
    """

    # If your project sets defaults elsewhere, these getattr calls in helpers
    # will still pick them up. You may override these on instances/config:
    #   self.exit_spread_pct_default: float = 0.01
    #   self.default_volatility: float = 0.0
    
    # Configuration for early expiry handling
    ALLOW_EARLY_EXPIRY_APPROX = True  # After an option's expiry, treat value at PM exit as 0
    
    # Time convention is centralized in config.DAYS_PER_YEAR
    
    def __init__(self, risk_free_rate: float = 0.05, min_delta_threshold: float = 0.10, 
                 max_delta_threshold: float = 0.90, max_spread: float = 0.03):
        """
        Initialize base options strategy with configurable thresholds.
        
        Args:
            risk_free_rate: Risk-free rate for Black-Scholes calculations
            min_delta_threshold: Minimum absolute delta for option filtering (default: 0.10)
            max_delta_threshold: Maximum absolute delta for option filtering (default: 0.90)
            max_spread: Maximum bid-ask spread for options (default: 0.03 = 3%)
        """
        super().__init__(risk_free_rate)
        # Options strategies use delta bands for filtering
        self.min_delta_threshold = min_delta_threshold
        self.max_delta_threshold = max_delta_threshold
        # Polymarket price thresholds - very permissive to only filter resolved markets
        self.min_price_threshold = 0.01  # 1% - only filter essentially resolved NO
        self.max_price_threshold = 0.99  # 99% - only filter essentially resolved YES
        self.max_spread = max_spread
        # Numerical stability parameters
        self._sigma_floor = 1e-8
        self._spot_floor = 1e-12
        self._time_floor = 1e-12
        self._strike_floor = 1e-12

    # -------- NEW: canonicalize option rows across venues --------
    def normalize_options(self, options: List[Dict]) -> List[Dict]:
        """
        Return a cleaned list where:
        - type is 'call' or 'put' (cp/right/etc. are normalized)
        - bid/ask are floats, ask>0, 0<=bid<=ask
        """
        out: List[Dict] = []
        for r in options or []:
            typ = _normalize_opt_type(
                r.get("type") or r.get("option_type") or r.get("right") or r.get("cp")
            )
            if typ is None:
                continue
            # coerce + validate quotes
            try:
                bid = float(r.get("bid", 0.0) or 0.0)
                ask = float(r.get("ask", 0.0) or 0.0)
            except Exception:
                continue
            bid = max(0.0, bid)
            if not (ask > 0.0 and bid <= ask):
                continue
            cleaned = dict(r)
            cleaned["type"] = typ
            cleaned["bid"] = bid
            cleaned["ask"] = ask
            out.append(cleaned)
        return out
    
    def get_strategy_type(self) -> str:
        """All options strategies return 'options' type"""
        return 'options'
    
    def _discount_factor(self, r: float, T: float) -> float:
        """Calculate discount factor e^(-rT)."""
        T = max(T, 0.0)
        return float(np.exp(-r * T))
    
    def _apply_slippage(self, mid_or_quote: float, side: str) -> float:
        """Apply slippage to a price based on side."""
        if side not in ('buy', 'sell'):
            raise ValueError("side must be 'buy' or 'sell'")
        adj = mid_or_quote * (1.0 + SLIPPAGE_BPS/1e4) if side == 'buy' else mid_or_quote * (1.0 - SLIPPAGE_BPS/1e4)
        return float(max(adj, 0.0))
    
    def _clamp(self, x: float, lo: float, hi: Optional[float] = None) -> float:
        """Clamp value between bounds for numerical stability."""
        if hi is not None:
            return float(min(max(x, lo), hi))
        return float(max(x, lo))
    
    def _safe_sigma(self, sigma: float) -> float:
        """Ensure sigma is above floor for numerical stability."""
        return self._clamp(sigma, self._sigma_floor)
    
    def _safe_strike(self, K: float) -> float:
        """Ensure strike is above floor to avoid log(S/K) blow-ups."""
        return self._clamp(K, self._strike_floor)
    
    def _validate_option_type(self, option_type: str) -> OptionType:
        """Validate option type is 'call' or 'put'."""
        if option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")
        return option_type
    
    def _d1(self, S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate d1 for Black-Scholes formula using centralized implementation."""
        d1, _ = BlackScholesGreeks.calculate_d1_d2(S, K, r, sigma, T)
        return d1
    
    def _d2(self, S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate d2 for Black-Scholes formula using centralized implementation."""
        _, d2 = BlackScholesGreeks.calculate_d1_d2(S, K, r, sigma, T)
        return d2
    
    def _rn_prob_ge(self, S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate risk-neutral probability that S_T >= K."""
        return float(norm.cdf(self._d2(S, K, r, sigma, T)))
    
    def _expected_spot_given_future_gbm(
        self,
        spot_today: float,
        spot_future: float,
        t_early: float,  # years to option expiry from now
        t_late: float,   # years to PM expiry from now (t_late > t_early)
        sigma: float
    ) -> float:
        """
        E[S_{t_early} | S_{t_late}] under risk‑neutral GBM with volatility `sigma` and risk‑free rate `r`.

        Model:
            dS_t = r S_t dt + sigma S_t dW_t,   S_t = S_0 * exp( (r - 0.5 sigma^2) t + sigma W_t )

        Let Y_t := ln S_t. Conditioning on S_{t_late} (i.e., Y_{t_late}) fixes W_{t_late}.
        Brownian‑bridge property (0 < t_early < t_late):
            W_{t_early} | W_{t_late}=w_T ~ N( (t_early/t_late) w_T,  t_early (1 - t_early/t_late) )

        Therefore ln S_{t_early} | ln S_{t_late} is Normal with
            m = ln S0 + (r - 0.5 sigma^2) t_early
                + (t_early / t_late) * ( ln S_{t_late} - [ ln S0 + (r - 0.5 sigma^2) t_late ] )
            v = sigma^2 * t_early * (1 - t_early / t_late)

        and E[S_{t_early} | S_{t_late}] = exp(m + 0.5 v).

        References: Shreve (SCF II), Revuz–Yor, Hull.
        """
        t_early = max(float(t_early), self._time_floor)
        t_late  = max(float(t_late),  self._time_floor)
        if not (t_late > t_early):
            return float(spot_future)  # degenerate/fallback

        S0 = max(float(spot_today), self._spot_floor)
        ST = max(float(spot_future), self._spot_floor)
        mu = self.risk_free_rate - 0.5 * sigma * sigma

        xT   = np.log(ST)
        mu_t = np.log(S0) + mu * t_early
        mu_T = np.log(S0) + mu * t_late

        cond_mean = mu_t + (t_early / t_late) * (xT - mu_T)
        cond_var  = (sigma * sigma) * t_early * (1.0 - t_early / t_late)

        return float(np.exp(cond_mean + 0.5 * cond_var))
    
    def _gbm_conditional_logn_params(
        self,
        spot_today: float,
        spot_at_pm: float,
        t_opt: float,
        t_pm: float,
        sigma: float
    ) -> Tuple[float, float]:
        """
        Return (m, v) for ln S_{t_opt} | S_{t_pm} under risk‑neutral GBM.
        Let mu = r - 0.5 sigma^2. Then
            m = ln S0 + mu t_opt + (t_opt / t_pm) * ( ln S_{t_pm} - [ln S0 + mu t_pm] )
            v = sigma^2 * t_opt * (1 - t_opt / t_pm)
        """
        # Floors
        t_opt = max(float(t_opt), self._time_floor)
        t_pm  = max(float(t_pm),  self._time_floor)
        S0 = max(float(spot_today), self._spot_floor)
        ST = max(float(spot_at_pm), self._spot_floor)
        sigma = self._safe_sigma(float(sigma))

        mu = self.risk_free_rate - 0.5 * sigma * sigma
        mu_t = np.log(S0) + mu * t_opt
        mu_T = np.log(S0) + mu * t_pm
        xT = np.log(ST)

        m = mu_t + (t_opt / t_pm) * (xT - mu_T)
        v = (sigma * sigma) * t_opt * (1.0 - t_opt / t_pm)
        return float(m), float(v)
    
    def _lognormal_call_partial_expectation(self, m: float, v: float, K: float) -> float:
        """
        E[(X - K)+] where X ~ LogNormal(m, v) (ln X ~ N(m, v)).
        Closed form: let s = sqrt(v), d2 = (m - ln K)/s, d1 = d2 + s.
            E[(X - K)+] = exp(m + 0.5 v) * Phi(d1) - K * Phi(d2).
        For v → 0, returns max(exp(m) - K, 0).
        """
        K = self._safe_strike(K)
        if v <= 1e-16:
            return max(0.0, np.exp(m) - K)
        s = np.sqrt(v)
        d2 = (m - np.log(K)) / s
        d1 = d2 + s
        return float(np.exp(m + 0.5 * v) * norm.cdf(d1) - K * norm.cdf(d2))

    def _lognormal_put_partial_expectation(self, m: float, v: float, K: float) -> float:
        """
        E[(K - X)+] where X ~ LogNormal(m, v) (ln X ~ N(m, v)).
        Closed form: let s = sqrt(v), d2 = (m - ln K)/s, d1 = d2 + s.
            E[(K - X)+] = K * Phi(-d2) - exp(m + 0.5 v) * Phi(-d1).
        For v → 0, returns max(K - exp(m), 0).
        """
        K = self._safe_strike(K)
        if v <= 1e-16:
            return max(0.0, K - np.exp(m))
        s = np.sqrt(v)
        d2 = (m - np.log(K)) / s
        d1 = d2 + s
        return float(K * norm.cdf(-d2) - np.exp(m + 0.5 * v) * norm.cdf(-d1))
    
    def filter_options_by_date(
        self, 
        options_data: List[Dict], 
        min_days_to_expiry: int = 0
    ) -> List[Dict]:
        """
        Simple filter for options that expire at least min_days after today.
        
        This is a basic date filter. For sophisticated PM-aware filtering,
        use filter_options_by_expiry instead.
        
        Args:
            options_data: List of options data
            min_days_to_expiry: Minimum days to expiry required from today
            
        Returns:
            Filtered list of options
        """
        min_expiry_date = datetime.now(timezone.utc).date() + timedelta(days=min_days_to_expiry)
        
        filtered = []
        total = len(options_data or [])
        for option in options_data:
            expiry_str = option.get('expiry_date')
            if not expiry_str:
                continue
                
            try:
                # Expect 'YYYY-MM-DD'
                expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
                if expiry_date >= min_expiry_date:
                    filtered.append(option)
            except Exception:
                continue
        try:
            kept = len(filtered)
            # Log a compact summary including a sample of the kept expiries
            kept_dates = sorted({ o.get('expiry_date') for o in filtered if o.get('expiry_date') })[:5]
            self.logger.info(f"[expiry] min_days={min_days_to_expiry} min_date={min_expiry_date} kept={kept}/{total} sample={kept_dates}")
        except Exception:
            pass
        return filtered
    
    def filter_options_by_expiry(
        self, 
        options: List[Dict], 
        pm_days_to_expiry: float,
        *, 
        inclusive: bool = True
    ) -> List[Dict]:
        """
        Filter options based on Polymarket expiry date with sophisticated validation.
        
        This method filters options to include only those from valid expiries
        that occur on or after the Polymarket resolution date.
        
        Args:
            options: List of option data dicts
            pm_days_to_expiry: Days until Polymarket resolution
            inclusive: If True, include options expiring on PM date
            
        Returns:
            List of filtered options suitable for hedging the PM position
            
        Valid expiry requirements:
        - Not synthetic/flagged for skip
        - Has valid quotes (bid > 0, ask > 0) if required
        - Expiry has minimum number of valid quotes
        - Expiry is within acceptable time window of PM date
        """
        if not options:
            return []
            
        pm_dte = float(pm_days_to_expiry or 0.0)
        
        # Initialize detailed logging
        debug_stats = {
            'total_options': len(options),
            'missing_expiry': 0,
            'synthetic_or_flagged': 0,
            'failed_quote_validation': 0,
            'expired_before_pm': 0,
            'missing_dte': 0,
            'expiry_groups': {},
            'rejected_expiries': {},
            'accepted_expiries': {}
        }
        
        # Get config parameters with defaults
        min_quotes_per_expiry = getattr(self, 'min_quotes_per_expiry', 5)
        require_live_quotes = getattr(self, 'require_live_quotes', True)
        max_expiries_considered = getattr(self, 'max_expiries_considered', 1)
        expiry_policy = getattr(self, 'expiry_policy', 'closest_only')
        max_expiry_gap_days = getattr(self, 'max_expiry_gap_days', 30.0)
        
        # Determine if far expiries are allowed
        allow_far = (expiry_policy == 'allow_far_with_unwind')
        
        # Group options by expiry
        by_expiry = {}
        for idx, opt in enumerate(options):
            # Accept multiple expiry field names
            expiry = opt.get('expiry_date') or opt.get('expiry') or opt.get('expiration')
            if not expiry:
                debug_stats['missing_expiry'] += 1
                continue
                
            # Skip synthetic or flagged options
            if opt.get('is_synthetic') or opt.get('skip_for_execution'):
                debug_stats['synthetic_or_flagged'] += 1
                continue
                
            # Validate quotes if required
            if require_live_quotes:
                bid = float(opt.get('bid', 0.0) or 0.0)
                ask = float(opt.get('ask', 0.0) or 0.0)
                has_live = opt.get('has_live_quotes', True)  # Default True if not specified
                
                if not ((bid > 0 or ask > 0) and has_live):
                    debug_stats['failed_quote_validation'] += 1
                    # Log specific quote failure details
                    if idx < 5:  # Log first 5 failures for debugging
                        self.logger.debug(
                            f"[expiry_filter] Quote validation failed for option {idx}: "
                            f"bid={bid}, ask={ask}, has_live={has_live}, "
                            f"strike={opt.get('strike')}, type={opt.get('type')}"
                        )
                    continue
            
            # Get days to expiry for this option
            opt_dte = opt.get('days_to_expiry')
            if opt_dte is None:
                # Calculate days_to_expiry from expiry_date if not present
                expiry_str = opt.get('expiry_date') or opt.get('expiry') or opt.get('expiration')
                if expiry_str:
                    try:
                        from datetime import datetime, timezone
                        # Parse the expiry date
                        expiry_dt = datetime.strptime(expiry_str, '%Y-%m-%d')
                        expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
                        now = datetime.now(timezone.utc)
                        opt_dte = max(0.0, (expiry_dt - now).total_seconds() / 86400.0)
                    except:
                        debug_stats['missing_dte'] += 1
                        continue
                else:
                    debug_stats['missing_dte'] += 1
                    continue
                
            try:
                opt_dte = float(opt_dte)
            except:
                debug_stats['missing_dte'] += 1
                continue
                
            # Check if option expires on/after PM date
            if inclusive:
                if opt_dte < pm_dte - 0.001:  # Small tolerance for float comparison
                    debug_stats['expired_before_pm'] += 1
                    if idx < 5:  # Log first 5 for debugging
                        self.logger.debug(
                            f"[expiry_filter] Option expires before PM: opt_dte={opt_dte:.2f}, "
                            f"pm_dte={pm_dte:.2f}, expiry={expiry}, strike={opt.get('strike')}"
                        )
                    continue
            else:
                if opt_dte <= pm_dte + 0.001:
                    debug_stats['expired_before_pm'] += 1
                    continue
                    
            # Add to expiry group
            if expiry not in by_expiry:
                by_expiry[expiry] = []
            by_expiry[expiry].append(opt)
        
        # Track expiry groups
        for expiry, opts in by_expiry.items():
            debug_stats['expiry_groups'][expiry] = len(opts)
        
        # Filter expiries by minimum quote count
        valid_expiries = {}
        for expiry, opts in by_expiry.items():
            if len(opts) >= min_quotes_per_expiry:
                valid_expiries[expiry] = opts
                debug_stats['accepted_expiries'][expiry] = len(opts)
            else:
                debug_stats['rejected_expiries'][expiry] = {
                    'count': len(opts),
                    'reason': f'too_few_quotes (< {min_quotes_per_expiry})'
                }
        
        if not valid_expiries:
            # Log complete debug summary when no valid expiries found
            self._log_expiry_filter_debug(debug_stats, pm_dte)
            return []
            
        # Sort expiries by proximity to PM date
        expiry_list = []
        for expiry, opts in valid_expiries.items():
            # Use the first option's DTE as representative
            rep_dte = opts[0].get('days_to_expiry', 0)
            distance = abs(rep_dte - pm_dte)
            expiry_list.append((expiry, opts, distance, rep_dte))
            
        expiry_list.sort(key=lambda x: x[2])  # Sort by distance
        
        # Select expiries based on policy
        result = []
        expiries_used = 0
        
        for expiry, opts, distance, rep_dte in expiry_list:
            if expiries_used >= max_expiries_considered:
                break
                
            # Check expiry gap constraint for additional expiries
            if expiries_used > 0 and not allow_far:
                break
                
            if expiries_used > 0 and allow_far:
                gap = abs(rep_dte - pm_dte)
                if gap > max_expiry_gap_days:
                    break
            
            result.extend(opts)
            expiries_used += 1
        
        # Update debug stats with final results
        debug_stats['final_options_count'] = len(result)
        debug_stats['expiries_used'] = expiries_used
        
        self.logger.debug(
            f"[filter_expiry] PM_dte={pm_dte:.1f} found {len(by_expiry)} expiries, "
            f"{len(valid_expiries)} valid, used {expiries_used}, returned {len(result)} options"
        )
        
        # Log debug stats if we filtered out a significant portion
        if len(result) < len(options) * 0.1:  # Lost >90% of options
            self._log_expiry_filter_debug(debug_stats, pm_dte)
        
        return result
    
    def filter_options_by_quality(
        self,
        options_data: List[Dict],
        *,
        use_mid: bool = True,
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
        max_rel_spread: Optional[float] = None,
        # NEW: require both sides to enable deterministic exits
        require_two_sided: bool = True,
        # NEW: size gates (contracts) and optional notional gate (quote currency, usually USDC)
        min_top_qty: float = 0.0,
        min_top_notional: float = 0.0,
    ) -> List[Dict]:
        """
        Filter options by delta band and relative spread.
        Requires: 'type', 'strike', 'expiry_years', and either:
          - ['bid','ask'] (+ sizes) when use_mid=True, or
          - 'price' when use_mid=False.
        
        Args:
            options_data: List of options data
            use_mid: Whether to use mid prices (requires bid/ask)
            min_delta: Minimum absolute delta (defaults to instance min_delta_threshold)
            max_delta: Maximum absolute delta (defaults to instance max_delta_threshold)
            max_rel_spread: Maximum relative spread as fraction of mid (defaults to instance max_spread)
            
        Returns:
            List of high-quality options
        """
        if min_delta is None:
            min_delta = self.min_delta_threshold
        if max_delta is None:
            max_delta = self.max_delta_threshold
        if max_rel_spread is None:
            # interpret self.max_spread as a relative fraction
            max_rel_spread = self.max_spread
        
        # Validate delta bounds
        if not (0 <= min_delta <= max_delta <= 1):
            raise ValueError(
                f"Invalid delta bounds: min_delta={min_delta}, max_delta={max_delta}. "
                "Must satisfy 0 <= min_delta <= max_delta <= 1"
            )
        
        out = []
        for opt in options_data:
            try:
                opt_type = self._validate_option_type(opt['type'])
                spot = float(opt['spot'])
                strike = float(opt['strike'])
                T = max(float(opt['expiry_years']), self._time_floor)
                iv = self._safe_sigma(float(opt.get('iv', opt.get('implied_volatility', 0.3))))
                
                # Validate spot and strike
                if not (np.isfinite(spot) and spot > 0):
                    continue
                if not (np.isfinite(strike) and strike > 0):
                    continue
                if not (np.isfinite(T) and T > 0):
                    continue
                if not (np.isfinite(iv) and iv > 0):
                    continue
            except Exception:
                continue

            # price + spread + two-sided exit liquidity
            if use_mid:
                bid = float(opt.get('bid', np.nan))
                ask = float(opt.get('ask', np.nan))
                bid_qty = float(opt.get('bid_qty', opt.get('best_bid_qty', 0.0)) or 0.0)
                ask_qty = float(opt.get('ask_qty', opt.get('best_ask_qty', 0.0)) or 0.0)
                # NOTE: previously checked `bid < 0`, which allowed zero-bid books through.
                if not np.isfinite(bid) or not np.isfinite(ask) or ask <= 0 or bid <= 0:
                    continue
                if require_two_sided:
                    # Both sides must have positive size; optionally require notional
                    if not (bid_qty > min_top_qty and ask_qty > min_top_qty):
                        continue
                    if min_top_notional > 0.0:
                        if (bid * bid_qty) < min_top_notional or (ask * ask_qty) < min_top_notional:
                            continue
                # Check for inverted bid/ask
                if bid > ask:
                    continue
                mid = 0.5 * (bid + ask)
                if mid <= 0:
                    continue
                rel_spread = (ask - bid) / mid
                if rel_spread > max_rel_spread:
                    continue
                price = mid
            else:
                price = float(opt.get('price', np.nan))
                if not np.isfinite(price) or price <= 0:
                    continue

            # delta band
            d1 = self._d1(spot, strike, self.risk_free_rate, iv, T)
            delta = norm.cdf(d1) if opt_type == 'call' else (norm.cdf(d1) - 1.0)
            if not (min_delta <= abs(delta) <= max_delta):
                continue

            # Preserve exit-side hints for downstream logic
            opt_out = {**opt, 'mid': price if use_mid else opt.get('price')}
            if use_mid:
                opt_out.setdefault('exit_bid', bid)
                opt_out.setdefault('exit_ask', ask)
                opt_out.setdefault('bid_qty', bid_qty)
                opt_out.setdefault('ask_qty', ask_qty)
            out.append(opt_out)
        return out

    # Convenience checker for reuse if you want it elsewhere in strategies
    def _two_sided_liquidity_ok(self, bid: float, ask: float, bid_qty: float, ask_qty: float,
                                min_qty: float = 0.0, min_notional: float = 0.0) -> bool:
        import numpy as _np
        if not (_np.isfinite(bid) and _np.isfinite(ask)) or bid <= 0 or ask <= 0:
            return False
        if not (bid_qty > min_qty and ask_qty > min_qty):
            return False
        if min_notional > 0.0 and ((bid * bid_qty) < min_notional or (ask * ask_qty) < min_notional):
            return False
        return True
    
    def calculate_black_scholes(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: OptionType = 'call'
    ) -> Dict[str, float]:
        """
        Calculate Black-Scholes price and Greeks using centralized implementation.
        
        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dict with:
            - price: Option price
            - delta: Price sensitivity to spot
            - gamma: Delta sensitivity to spot
            - vega: Price sensitivity per 1.00 absolute vol change (for per-1% vol, multiply by 0.01)
            - theta: Time decay per year (for per-day, divide by DAYS_PER_YEAR)
            - rho: Price sensitivity to interest rate (annualized)
        """
        # Validate option type
        option_type = self._validate_option_type(option_type)
        
        # Apply numerical safety
        volatility = self._safe_sigma(volatility)
        spot = max(spot, self._spot_floor)
        
        # Use centralized Black-Scholes implementation
        bs = BlackScholesGreeks()
        
        if option_type == 'call':
            price = bs.call_price(spot, strike, self.risk_free_rate, volatility, time_to_expiry)
            greeks = bs.call_greeks(spot, strike, self.risk_free_rate, volatility, time_to_expiry, 
                                   scale_1pct=False)  # We want raw derivatives to match existing behavior
        else:
            price = bs.put_price(spot, strike, self.risk_free_rate, volatility, time_to_expiry)
            greeks = bs.put_greeks(spot, strike, self.risk_free_rate, volatility, time_to_expiry,
                                  scale_1pct=False)  # We want raw derivatives to match existing behavior
        
        # The centralized implementation returns theta per day, but our API expects annualized
        theta_annual = greeks['theta'] * DAYS_PER_YEAR
        
        return {
            'price': float(price),
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'vega': greeks['vega'],      # per 1.00 vol (matches our existing API)
            'theta': theta_annual,        # annualized (matches our existing API)
            'rho': greeks['rho']
        }
    
    def _calculate_theta(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        d1: float,
        d2: float,
        option_type: OptionType
    ) -> float:
        """Calculate option theta (time decay)"""
        # Validate option type
        option_type = self._validate_option_type(option_type)
        first_term = -(spot * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry))
        
        if option_type == 'call':
            second_term = -self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:
            second_term = self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2)
        
        return first_term + second_term
    
    def find_options_around_strike(
        self,
        options_data: List[Dict],
        target_strike: float,
        num_strikes: int = 5
    ) -> List[Dict]:
        """
        Find options with strikes closest to target strike.
        
        Args:
            options_data: List of options
            target_strike: Target strike price
            num_strikes: Number of strikes to return
            
        Returns:
            List of options sorted by distance from target
        """
        # Create non-mutating decorated copies with distance
        decorated = [
            {**opt, 'distance_from_target': abs(opt.get('strike', 0) - target_strike)}
            for opt in options_data
        ]
        
        # Sort by distance and return closest
        sorted_options = sorted(decorated, key=lambda x: x['distance_from_target'])
        return sorted_options[:num_strikes]
    
    def calculate_spread_payoff(
        self,
        spot_at_expiry: float,
        long_strike: float,
        short_strike: float,
        spread_type: OptionType = 'call'
    ) -> float:
        """
        Calculate payoff of a vertical spread at expiry.
        
        Args:
            spot_at_expiry: Spot price at expiry
            long_strike: Strike of long option
            short_strike: Strike of short option
            spread_type: 'call' or 'put'
            
        Returns:
            Net payoff of spread
        """
        # Validate spread type
        spread_type = self._validate_option_type(spread_type)
        
        # Validate strike ordering for sensible spreads
        if spread_type == 'call' and long_strike >= short_strike:
            logger.warning(
                "Call spread may be inverted: long_strike (%.4f) >= short_strike (%.4f)",
                long_strike, short_strike
            )
        elif spread_type == 'put' and long_strike <= short_strike:
            logger.warning(
                "Put spread may be inverted: long_strike (%.4f) <= short_strike (%.4f)",
                long_strike, short_strike
            )
        
        if spread_type == 'call':
            long_payoff = max(0, spot_at_expiry - long_strike)
            short_payoff = max(0, spot_at_expiry - short_strike)
        else:
            long_payoff = max(0, long_strike - spot_at_expiry)
            short_payoff = max(0, short_strike - spot_at_expiry)
        
        return long_payoff - short_payoff
    
    def create_lyra_info(
        self,
        expiry_date: str,
        strikes: List[float],
        cost: float,
        days_to_expiry: int = None
    ) -> Dict:
        """
        Create standardized Lyra info structure with expiry dates.
        
        Args:
            expiry_date: Option expiry date (YYYY-MM-DD format)
            strikes: List of strikes used [lower, upper] or [strike]
            cost: Net cost of the strategy
            days_to_expiry: Days to option expiry
            
        Returns:
            Dict with standardized Lyra information
        """
        return {
            'expiry': expiry_date,
            'strikes': strikes,
            'spread_cost': cost,
            'days_to_expiry': days_to_expiry
        }
    
    def calculate_option_exit_value(
        self,
        option_strike: float,
        option_type: OptionType,
        days_to_option_expiry: int,
        days_to_pm_expiry: int,
        spot_at_pm_expiry: float,
        current_iv: float,
        iv_change_factor: float = 1.0,
        use_martingale_when_early: bool = False,
        *,
        spot_today: Optional[float] = None,            # NEW: needed for GBM backcast
        sigma_for_backcast: Optional[float] = None     # NEW: default to current_iv if None
    ) -> float:
        """
        Calculate option value when exiting at PM expiry.

        If the option lives beyond PM (days_remaining > 0), price with Black–Scholes
        using time_to_expiry = (days_to_option_expiry - days_to_pm_expiry)/DAYS_PER_YEAR
        and IV adjusted by iv_change_factor.

        If the option expires before PM (days_remaining <= 0):
          - Default: return 0 at PM exit (ALLOW_EARLY_EXPIRY_APPROX=True).
          - If use_martingale_when_early=True, compute the exact conditional payoff

                call: E[(S_{t_opt} - K)+ | S_{t_pm}]
                 put: E[(K - S_{t_opt})+ | S_{t_pm}]

            Under risk‑neutral GBM,
                ln S_{t_opt} | S_{t_pm} ~ Normal(m, v),
                m = ln S0 + (r - 0.5 sigma^2) t_opt + (t_opt/t_pm) * ( ln S_{t_pm} - [ln S0 + (r - 0.5 sigma^2) t_pm] )
                v = sigma^2 * t_opt * (1 - t_opt/t_pm)
            and the lognormal partial expectations above give the value.
        """
        # Validate option type
        option_type = self._validate_option_type(option_type)
        
        # Validate inputs
        if iv_change_factor <= 0:
            raise ValueError(f"iv_change_factor must be positive, got {iv_change_factor}")
        if days_to_pm_expiry < 0:
            raise ValueError(f"days_to_pm_expiry cannot be negative, got {days_to_pm_expiry}")
        days_to_option_expiry = max(0, days_to_option_expiry)  # Floor to 0
        
        # Calculate time remaining after PM expires
        days_remaining = days_to_option_expiry - days_to_pm_expiry
        if days_remaining <= 0:
            # Option expired before PM resolves
            if use_martingale_when_early:
                if spot_today is None:
                    raise ValueError("spot_today is required when use_martingale_when_early=True")
                sigma = float(sigma_for_backcast) if sigma_for_backcast is not None else float(current_iv)
                t_opt = max(days_to_option_expiry / DAYS_PER_YEAR, self._time_floor)
                t_pm  = max(days_to_pm_expiry    / DAYS_PER_YEAR, self._time_floor)
                # Only guard truly invalid ordering; allow equality (v -> 0 handles it)
                if t_pm < t_opt:
                    return 0.0

                m, v = self._gbm_conditional_logn_params(spot_today, spot_at_pm_expiry, t_opt, t_pm, sigma)
                return (
                    self._lognormal_call_partial_expectation(m, v, option_strike)
                    if option_type == 'call'
                    else self._lognormal_put_partial_expectation(m, v, option_strike)
                )

            # Default: at PM exit time, an already-expired option has zero value
            if self.ALLOW_EARLY_EXPIRY_APPROX:
                return 0.0

            # (Optional legacy behavior) — strongly discouraged:
            # intrinsic-at-PM-spot is not the actual value of an already expired option.
            intrinsic = max(0.0, spot_at_pm_expiry - option_strike) if option_type == 'call' \
                        else max(0.0, option_strike - spot_at_pm_expiry)
            return float(intrinsic)
        
        # Convert to years
        time_remaining = days_remaining / DAYS_PER_YEAR
        
        # Model IV change - simple factor for now
        # In production, this could be more sophisticated based on:
        # - Volatility smile/skew
        # - Event volatility crush
        # - Historical patterns
        new_iv = current_iv * iv_change_factor
        
        # Calculate Black-Scholes price with remaining time
        bs_result = self.calculate_black_scholes(
            spot=spot_at_pm_expiry,
            strike=option_strike,
            time_to_expiry=time_remaining,
            volatility=new_iv,
            option_type=option_type
        )
        
        return bs_result['price']
    
    def calculate_portfolio_exit_value(
        self,
        positions: List[Dict],
        days_to_pm_expiry: int,
        spot_at_pm_expiry: float,
        iv_change_factor: float = 1.0,
        apply_slippage: bool = False,
        side: Optional[str] = None,
        *,
        use_martingale_when_early: bool = False,
        spot_today: Optional[float] = None,
        sigma_for_backcast: Optional[float] = None,
    ) -> float:
        """
        Calculate total value of options portfolio at PM expiry.
        
        Args:
            positions: List of option positions, each with:
                - strike: Strike price
                - type: 'call' or 'put'
                - quantity: Number of contracts (negative for short)
                - days_to_expiry: Days to option expiry
                - implied_volatility: Current IV
            days_to_pm_expiry: Days to PM resolution
            spot_at_pm_expiry: Spot price at PM resolution
            iv_change_factor: IV change multiplier
            apply_slippage: Whether to apply slippage to exit values
            side: 'buy' or 'sell' for slippage direction (if apply_slippage=True)
            use_martingale_when_early: If True, use martingale expectation for early-expiry options
            spot_today: Current spot price (required if use_martingale_when_early=True)
            sigma_for_backcast: Volatility for GBM backcast (defaults to current_iv)
            
        Returns:
            Total portfolio value at exit
        """
        total_value = 0
        
        for position in positions:
            # Check if IV is missing and log warning
            current_iv = position.get('implied_volatility', 0.6)
            if 'implied_volatility' not in position:
                logger.warning(
                    "Missing implied volatility for %s option at strike %.4f, defaulting to %.4f",
                    position['type'], position['strike'], current_iv
                )
            
            exit_value = self.calculate_option_exit_value(
                option_strike=position['strike'],
                option_type=position['type'],
                days_to_option_expiry=position['days_to_expiry'],
                days_to_pm_expiry=days_to_pm_expiry,
                spot_at_pm_expiry=spot_at_pm_expiry,
                current_iv=current_iv,
                iv_change_factor=iv_change_factor,
                use_martingale_when_early=use_martingale_when_early,
                spot_today=spot_today,
                sigma_for_backcast=sigma_for_backcast
            )
            
            # Apply slippage if requested (auto-detect side if not provided)
            if apply_slippage and exit_value > 0:
                side_for_pos = side if side in ('buy', 'sell') else ('sell' if position['quantity'] > 0 else 'buy')
                exit_value = self._apply_slippage(exit_value, side_for_pos)
            
            # Multiply by quantity (negative for short positions)
            total_value += exit_value * position['quantity']
        
        return total_value

    # ---------------------------------------------------------------------
    # BEGIN: Centralized exit-valuation & cost-recovery helpers
    # ---------------------------------------------------------------------
    def _theoretical_option_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: str,
    ) -> float:
        """
        Return a theoretical *mid* price using the strategy's Black-Scholes
        if available, else fall back to intrinsic value.
        """
        if hasattr(self, 'calculate_black_scholes'):
            try:
                out = self.calculate_black_scholes(
                    spot=float(spot),
                    strike=float(strike),
                    time_to_expiry=max(0.0, float(time_to_expiry)),
                    volatility=max(0.0, float(volatility)),
                    option_type=str(option_type).lower(),
                )
                px = out['price'] if isinstance(out, dict) else float(out)
                if px is not None:
                    return float(px)
            except Exception:
                # fall through to intrinsic if the engine throws
                pass

        # Conservative intrinsic fallback if no pricing engine available
        if str(option_type).lower().startswith('c'):
            return max(0.0, float(spot) - float(strike))
        else:
            return max(0.0, float(strike) - float(spot))

    def calculate_hedge_value_at_price(
        self,
        target_price: float,
        hedge_positions: List[Dict],
        days_to_exit: float,
        current_spot: float,
        apply_exit_spreads: bool = True,
        default_exit_spread_pct: Optional[float] = None,
        *,
        apply_greeks_haircut: bool = False,
        haircut_dt_seconds: Optional[float] = None,
        delta_sigma_quantile: float = 1.0,   # z in N(0,1): 1.0≈1σ, 1.645≈95% one-sided
    ) -> float:
        """
        Cash proceeds (positive) or cost (negative) to unwind *the hedge* at S=target_price.
        Conventions:
          • Long options -> exit by *selling* at bid (cash inflow).
          • Short options -> exit by *buying* at ask (cash outflow).
        If quotes are missing, use theoretical mid and apply +/- spread pct.
        Each leg dict should include:
          {'type','strike','quantity','bid','ask','iv'(optional)}
        """
        T = max(0.0, float(days_to_exit)) / 365.0
        # Short execution horizon for the haircut (seconds -> years)
        try:
            dt_sec = float(haircut_dt_seconds) if haircut_dt_seconds is not None else 30.0
        except Exception:
            dt_sec = 30.0
        dt_years = dt_sec / SECONDS_PER_YEAR
        spread_pct = default_exit_spread_pct
        if spread_pct is None:
            spread_pct = getattr(self, 'exit_spread_pct_default', 0.01)  # 1% default

        total = 0.0
        for pos in hedge_positions or []:
            q = float(pos.get('quantity', 0.0) or 0.0)
            if q == 0.0:
                continue
            opt_type = str(pos.get('type') or pos.get('option_type') or '').lower()
            K        = float(pos.get('strike', 0.0) or 0.0)
            bid      = pos.get('bid', None)
            ask      = pos.get('ask', None)
            iv       = float(pos.get('iv', getattr(self, 'default_volatility', 0.0)) or 0.0)

            mid_theo = self._theoretical_option_price(
                spot=float(target_price), strike=K, time_to_expiry=T, volatility=iv, option_type=opt_type
            )

            if apply_exit_spreads:
                if (bid is not None and bid > 0) or (ask is not None and ask > 0):
                    if q > 0:
                        # long -> sell at bid
                        if bid is not None and bid > 0:
                            px = float(bid)
                        elif ask is not None and ask > 0:
                            px = max(0.0, float(ask) * (1.0 - 2.0 * spread_pct))
                        else:
                            px = max(0.0, mid_theo * (1.0 - spread_pct))
                    else:
                        # short -> buy at ask
                        if ask is not None and ask > 0:
                            px = float(ask)
                        elif bid is not None and bid > 0:
                            px = float(bid) * (1.0 + 2.0 * spread_pct)
                        else:
                            px = mid_theo * (1.0 + spread_pct)
                else:
                    px = mid_theo * (1.0 - spread_pct) if q > 0 else mid_theo * (1.0 + spread_pct)
            else:
                if (bid is not None and bid > 0) and (ask is not None and ask > 0):
                    px = 0.5 * (float(bid) + float(ask))
                else:
                    px = mid_theo

            # --- Optional Θ/Γ/Δ micro-haircut over a short execution horizon ---
            if apply_greeks_haircut and px > 0.0 and dt_years > 0.0:
                try:
                    # Greeks at PM resolution boundary (S=target_price, T=days_to_exit/365)
                    out = self.calculate_black_scholes(
                        spot=float(target_price),
                        strike=K,
                        time_to_expiry=T,
                        volatility=max(1e-8, iv),
                        option_type=opt_type,
                    )
                    theta_ann = float(out.get('theta', 0.0))  # per-year (our wrapper annualizes)
                    gamma     = float(out.get('gamma', 0.0))  # per $^2$
                    delta     = float(out.get('delta', 0.0))  # per $
                    sigma     = max(1e-8, float(iv))          # implied vol (ann.)
                    S         = float(target_price)           # here: K_PM
                    # Time-decay + curvature + directional (1σ or chosen z) adverse move
                    theta_term = abs(theta_ann) * dt_years
                    gamma_term = 0.5 * abs(gamma) * (sigma * S) * (sigma * S) * dt_years
                    delta_term = abs(delta) * sigma * S * (dt_years ** 0.5) * float(delta_sigma_quantile)
                    haircut    = theta_term + gamma_term + delta_term
                    if q > 0:   # selling a long -> reduce proceeds
                        px = max(0.0, px - haircut)
                    else:       # buying back a short -> increase cost
                        px = px + haircut
                except Exception:
                    # If anything goes wrong, skip the haircut rather than failing the valuation
                    pass

            cashflow = q * px if q > 0 else -abs(q) * px
            total += cashflow

        return float(total)

    def immediate_execution_debit(self, hedge_positions: List[Dict]) -> float:
        """Return immediate cash outlay to open the given option hedge.
        Positive values mean a net debit (bad for 'costless' hedges).
        Uses execution_pricing.option_exec_price with configured bps.
        """
        try:
            from execution_pricing import option_exec_price
        except Exception:
            return float('inf') if hedge_positions else 0.0
        total = 0.0
        for leg in hedge_positions:
            side = (leg.get('action') or 'BUY').upper()
            bid  = leg.get('bid'); ask = leg.get('ask'); mid = leg.get('mid')
            qty  = float(leg.get('contracts', 0.0) or 0.0) * float(leg.get('contract_size', 1.0) or 1.0)
            px   = option_exec_price('buy' if side=='BUY' else 'sell', bid, ask, mid=mid)
            total += px * qty
        return float(total)

    def validate_cost_recovery_at_strike(
        self,
        pm_strike: float,
        hedge_positions: List[Dict],
        hedge_entry_cost: float,
        days_to_pm_expiry: float,
        current_spot: float,
        default_exit_spread_pct: Optional[float] = None,
        *,
        apply_greeks_haircut: bool = False,
        haircut_dt_seconds: Optional[float] = None,
        delta_sigma_quantile: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Check that the hedge is self-financing at the PM strike:
            hedge_exit_value_at_pm >= entry_cost + exit_spread_loss
        Returns a dict with detailed metrics.
        """
        # Guard: strategy must not require a net debit to open the hedge at realistic execution prices
        try:
            _debit = self.immediate_execution_debit(hedge_positions)
            if _debit > 1e-8:
                return {
                    'hedge_exit_value': 0.0,
                    'entry_cost': float(hedge_entry_cost),
                    'exit_spread_loss': 0.0,
                    'total_cost_to_recover': float(hedge_entry_cost),
                    'net_recovery': -float(_debit),
                    'recovery_ratio': 0.0,
                    'is_adequate': False,
                }
        except Exception:
            pass

        hedge_value = self.calculate_hedge_value_at_price(
            target_price=float(pm_strike),
            hedge_positions=hedge_positions,
            days_to_exit=float(days_to_pm_expiry),
            current_spot=float(current_spot),
            apply_exit_spreads=True,
            default_exit_spread_pct=default_exit_spread_pct,
            apply_greeks_haircut=apply_greeks_haircut,
            haircut_dt_seconds=haircut_dt_seconds,
            delta_sigma_quantile=delta_sigma_quantile,
        )

        # Compare to mid to infer explicit spread loss component
        mid_value = self.calculate_hedge_value_at_price(
            target_price=float(pm_strike),
            hedge_positions=hedge_positions,
            days_to_exit=float(days_to_pm_expiry),
            current_spot=float(current_spot),
            apply_exit_spreads=False,
            default_exit_spread_pct=default_exit_spread_pct,
            apply_greeks_haircut=apply_greeks_haircut,
            haircut_dt_seconds=haircut_dt_seconds,
            delta_sigma_quantile=delta_sigma_quantile,
        )
        exit_spread_loss = max(0.0, float(mid_value) - float(hedge_value))

        entry_cost = abs(float(hedge_entry_cost))
        total_cost = entry_cost + exit_spread_loss

        return {
            'hedge_value_at_strike': float(hedge_value),
            'entry_cost': float(entry_cost),
            'exit_spread_loss': float(exit_spread_loss),
            'total_cost_to_recover': float(total_cost),
            'net_recovery': float(hedge_value) - float(total_cost),
            'recovery_ratio': (float(hedge_value) / float(total_cost)) if total_cost > 0 else 0.0,
            'is_adequate': bool(hedge_value >= total_cost),
        }
    # ---------------------------------------------------------------------
    # END: Centralized exit-valuation & cost-recovery helpers
    # ---------------------------------------------------------------------
    
    def _log_expiry_filter_debug(self, debug_stats: Dict, pm_dte: float) -> None:
        """Log detailed debug information about expiry filtering."""
        import json
        import os
        from datetime import datetime
        
        # Create debug directory if needed
        debug_dir = "debug_runs"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Log to both logger and file
        self.logger.info(
            f"[EXPIRY_FILTER_DEBUG] PM_dte={pm_dte:.1f} days, "
            f"Total options: {debug_stats['total_options']}, "
            f"Final: {debug_stats.get('final_options_count', 0)}"
        )
        
        # Log rejection reasons
        self.logger.info(
            f"[EXPIRY_FILTER_DEBUG] Rejections - "
            f"Missing expiry: {debug_stats['missing_expiry']}, "
            f"Synthetic/flagged: {debug_stats['synthetic_or_flagged']}, "
            f"Failed quotes: {debug_stats['failed_quote_validation']}, "
            f"Expired before PM: {debug_stats['expired_before_pm']}, "
            f"Missing DTE: {debug_stats['missing_dte']}"
        )
        
        # Log expiry group details
        if debug_stats['expiry_groups']:
            self.logger.info(
                f"[EXPIRY_FILTER_DEBUG] Found {len(debug_stats['expiry_groups'])} expiry groups"
            )
            for expiry, count in sorted(debug_stats['expiry_groups'].items())[:5]:
                self.logger.info(f"  - {expiry}: {count} options")
        
        # Log rejected expiries
        if debug_stats['rejected_expiries']:
            self.logger.info(
                f"[EXPIRY_FILTER_DEBUG] Rejected {len(debug_stats['rejected_expiries'])} expiries:"
            )
            for expiry, info in sorted(debug_stats['rejected_expiries'].items())[:5]:
                self.logger.info(f"  - {expiry}: {info['count']} options, reason: {info['reason']}")
        
        # Write detailed stats to file
        debug_file = os.path.join(debug_dir, "expiry_filter_debug.jsonl")
        with open(debug_file, "a") as f:
            debug_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'pm_dte': pm_dte,
                'stats': debug_stats
            }
            f.write(json.dumps(debug_entry) + "\n")