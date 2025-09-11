"""
Probability-Based Arbitrage Opportunity Ranker
==============================================

This module implements a sophisticated ranking system for arbitrage opportunities
based on probability-weighted expected value and risk metrics, grounded in 
peer-reviewed academic research.

Academic foundations:
1. Breeden & Litzenberger (1978) - Digital option replication via spreads
   https://www.jstor.org/stable/2352653
   
2. Wolfers & Zitzewitz (2004) - Prediction markets as probability estimators
   https://www.aeaweb.org/articles?id=10.1257%2F0895330041371321
   
3. Shleifer & Vishny (1997) - Limits to arbitrage and risk penalties
   https://scholar.harvard.edu/files/shleifer/files/limitsofarbitrage.pdf
   
4. Kelly (1956) - Optimal betting fractions and probabilistic sizing
   https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf
   
5. Aït-Sahalia & Lo (1998) - Option-implied probability densities
   https://www.princeton.edu/~yacine/aslo.pdf

Key innovations:
- Two-state payoff framework for all strategies
- Blended probability estimates (prediction market + options-implied)
- Distance-to-no-arbitrage (DNI) metric
- Risk-adjusted expected value with penalties
- Lexicographic sorting with dominance pruning
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import logging
from dataclasses import dataclass
from math import exp, sqrt, log
from scipy.stats import norm
from scipy.interpolate import interp1d
from black_scholes_greeks import BlackScholesGreeks, calculate_vega_penalty_accurate
from config_loader import get_config
from config_manager import DEFAULT_IMPLIED_VOLATILITY, RISK_FREE_RATE, DAYS_PER_YEAR
from market_data_analyzer import MarketDataAnalyzer


@dataclass
class PayoffProfile:
    """Two-state payoff structure for any arbitrage strategy"""
    upfront_cashflow: float  # Initial cash in/out
    payoff_if_yes: float    # Total P&L if event occurs
    payoff_if_no: float     # Total P&L if event doesn't occur
    
    @property
    def max_profit(self) -> float:
        return max(self.payoff_if_yes, self.payoff_if_no)
    
    @property
    def max_loss(self) -> float:
        return min(self.payoff_if_yes, self.payoff_if_no)
    
    @property
    def is_state_independent(self) -> bool:
        """True if profit is identical in both states (true arbitrage)"""
        return abs(self.payoff_if_yes - self.payoff_if_no) < 0.01


class ProbabilityRanker:
    """
    Ranks arbitrage opportunities by probability-weighted expected value
    with explicit penalties for residual risks.
    """
    
    @staticmethod
    def _clip_prob(x: float, eps: float = 1e-6) -> float:
        """Clip probabilities away from {0,1} to avoid infinities in sizing/logit math."""
        import numpy as np
        return float(np.clip(float(x), eps, 1 - eps))

    @staticmethod
    def _cfg(config, *names, default=None):
        """
        Read a config value with graceful fallback:
        - Try provided names (e.g., 'PROBABILITY_BLEND_MODE', 'probability_blend_mode')
        - Try lowercase variants
        - Try pydantic v1 .dict() or v2 .model_dump()
        """
        # Direct attribute access (handles alias properties too)
        for n in names:
            if hasattr(config, n):
                val = getattr(config, n)
                if val is not None:
                    return val
            low = n.lower()
            if hasattr(config, low):
                val = getattr(config, low)
                if val is not None:
                    return val

        # Try dict-like export (pydantic v1 / v2)
        for method in ("model_dump", "dict"):
            if hasattr(config, method):
                try:
                    data = getattr(config, method)()
                except Exception:
                    data = None
                if isinstance(data, dict):
                    for n in names:
                        if n in data and data[n] is not None:
                            return data[n]
                        low = n.lower()
                        if low in data and data[low] is not None:
                            return data[low]
        return default
    
    def __init__(
        self,
        polymarket_fee: float = 0.02,      # 2% winner's fee
        risk_free_rate: float = RISK_FREE_RATE,
        use_config: bool = True
    ):
        self.logger = logging.getLogger("ProbabilityRanker")
        self.pm_fee = polymarket_fee
        self.risk_free_rate = risk_free_rate

        if use_config:
            cfg = get_config().ranking

            # tolerate UPPER_SNAKE and lower_snake (pydantic v2)
            self.prob_blend_mode = self._cfg(
                cfg, "PROBABILITY_BLEND_MODE", "probability_blend_mode", "blend_mode", default="fixed"
            )
            
            # Allow common aliases (so .env can say "precision")
            mode_raw = self.prob_blend_mode
            aliases = {
                "precision": "fixed",
                "prec": "fixed",
                "exact": "fixed",
                'precise': 'fixed',
                'bayes': 'bayesian'
            }
            mode = aliases.get(mode_raw.lower(), mode_raw.lower())
            if mode != mode_raw.lower():
                self.logger.info("Mapping PROBABILITY_BLEND_MODE='%s' -> '%s' via alias", mode_raw, mode)
            self.prob_blend_mode = mode
                
            self.prob_blend_weight = float(self._cfg(
                cfg, "PROBABILITY_BLEND_WEIGHT", "probability_blend_weight", default=0.5
            ))

            self.vega_lambda = float(self._cfg(
                cfg, "VEGA_PENALTY_LAMBDA", "vega_penalty_lambda", default=0.5
            ))
            self.funding_kappa = float(self._cfg(
                cfg, "FUNDING_PENALTY_KAPPA", "funding_penalty_kappa", default=1.0
            ))
            self.kelly_cap = float(self._cfg(
                cfg, "KELLY_FRACTION_CAP", "kelly_fraction_cap", default=0.25
            ))
            self.transaction_cost_rate = float(self._cfg(
                cfg, "TRANSACTION_COST_RATE", "transaction_cost_rate", default=0.002
            ))

            self.market_impact_k = float(self._cfg(
                cfg, "MARKET_IMPACT_COEFFICIENT", "market_impact_coefficient", default=0.1
            ))
            # Aggregated counters for robust handling
            self._ranker_counts = {"missing_payoffs": 0, "missing_spot": 0, "invalid_fields": 0}

            # thresholds / tiers
            self.true_arb_dni_threshold = float(self._cfg(
                cfg, "TRUE_ARBITRAGE_DNI_THRESHOLD", "true_arbitrage_dni_threshold", default=0.0
            ))
            self.near_arb_dni_threshold = float(self._cfg(
                cfg, "NEAR_ARBITRAGE_DNI_THRESHOLD", "near_arbitrage_dni_threshold", default=-0.01
            ))
            self.near_arb_prob_threshold = float(self._cfg(
                cfg, "NEAR_ARBITRAGE_PROB_THRESHOLD", "near_arbitrage_prob_threshold", default=0.70
            ))
            self.high_prob_threshold = float(self._cfg(
                cfg, "HIGH_PROBABILITY_THRESHOLD", "high_probability_threshold", default=0.80
            ))
            self.moderate_prob_threshold = float(self._cfg(
                cfg, "MODERATE_PROBABILITY_THRESHOLD", "moderate_probability_threshold", default=0.60
            ))
        else:
            # sensible defaults (kept from your file)
            self.prob_blend_mode = "fixed"
            self.prob_blend_weight = 0.5
            self.vega_lambda = 0.5
            self.funding_kappa = 1.0
            self.kelly_cap = 0.25
            self.transaction_cost_rate = 0.002
            self.market_impact_k = 0.1

            self.true_arb_dni_threshold = 0.0
            self.near_arb_dni_threshold = -0.01
            self.near_arb_prob_threshold = 0.70
            self.high_prob_threshold = 0.80
            self.moderate_prob_threshold = 0.60

        # final guards
        if self.prob_blend_mode not in ("fixed", "bayesian"):
            self.logger.warning("Unknown PROBABILITY_BLEND_MODE=%r; falling back to 'fixed'", self.prob_blend_mode)
            self.prob_blend_mode = "fixed"

        # clamp key fractions
        self.prob_blend_weight = max(0.0, min(1.0, self.prob_blend_weight))
        self.kelly_cap = max(0.0, min(1.0, self.kelly_cap))

        # optional analyzers (lazy-init where used)
        self.market_analyzer = MarketDataAnalyzer()
        
    def rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Main ranking function - takes raw opportunities and returns sorted list
        with probability metrics and rankings.
        """
        self.logger.info(f"Ranking {len(opportunities)} opportunities by probability-weighted metrics")
        # Safety gate: reject any opportunity whose option hedge creates an immediate net debit
        # (buying at ask and selling at bid after fees). This prevents pseudo-arbitrage.
        try:
            from execution_pricing import violates_no_immediate_loss
            before = len(opportunities)
            safe = []
            dropped = 0
            for opp in opportunities:
                try:
                    bad = violates_no_immediate_loss(opp.get('execution_details', {}))
                except Exception:
                    # Missing quotes => not executable at entry → drop
                    bad = True
                if bad:
                    dropped += 1
                else:
                    safe.append(opp)
            opportunities = safe
            if dropped:
                self.logger.warning("Dropped %d opportunities (immediate-debit or non-executable at entry)", dropped)
        except Exception:
            # If pricing utilities are unavailable, proceed without gating
            pass
        
        # Step 1: Convert to two-state payoff profiles
        enriched_opportunities = []
        for opp in opportunities:
            try:
                # Extract payoff profile
                payoff = self._extract_payoff_profile(opp)
                
                # Calculate probability estimates
                pm_prob = self._get_pm_probability(opp)
                opt_prob = self._get_options_implied_probability(opp)
                
                # Keep current opp context available for cost-aware metrics
                self._current_opp = opp
                blended_prob = self._blend_probabilities(pm_prob, opt_prob)

                # Core metrics (some use actual costs, so they need _current_opp)
                dni = self._calculate_dni(opp, payoff)
                prob_of_profit = self._calculate_profit_probability(payoff, blended_prob)
                expected_value = self._calculate_expected_value(payoff, blended_prob)

                # Risk penalties & adjusted EV
                penalties = self._calculate_risk_penalties(opp, payoff)
                adjusted_ev = expected_value - penalties['total']

                # Advanced metrics
                edge_per_downside = self._calculate_edge_per_downside(payoff, expected_value)
                kelly_signal = self._calculate_kelly_signal(payoff, blended_prob)

                # Now it's safe to clear the handle
                del self._current_opp
                
                # Enrich opportunity with metrics
                enriched_opp = opp.copy()
                # Calculate actual costs for true arbitrage check
                # True arbitrage requires positive payoff in BOTH states after ALL costs
                actual_costs = self._calculate_actual_costs(opp, payoff)
                min_payoff_after_costs = min(payoff.payoff_if_yes, payoff.payoff_if_no) - actual_costs
                is_true_arbitrage = (dni >= 0) or (min_payoff_after_costs > 0)

                # Compute upfront cashflow once (from execution details when present)
                upfront_cashflow_val = payoff.upfront_cashflow
                if opp.get('execution_details'):
                    try:
                        upfront_cashflow_val = __import__('execution_pricing').execution_pricing.total_entry_cashflow(
                            opp.get('execution_details', {})
                        )
                    except Exception:
                        upfront_cashflow_val = payoff.upfront_cashflow

                enriched_opp.update({
                    'payoff_profile': {
                        # Upfront cashflow computed above for consistency
                        'upfront_cashflow': upfront_cashflow_val,
                        'payoff_if_yes': payoff.payoff_if_yes,
                        'payoff_if_no': payoff.payoff_if_no,
                        'is_state_independent': payoff.is_state_independent
                    },
                    # Top-level mirrors for UI panes that expect these keys without nesting
                    'upfront_cashflow': upfront_cashflow_val,
                    'payoff_if_yes': payoff.payoff_if_yes,
                    'payoff_if_no': payoff.payoff_if_no,
                    
                    'probabilities': {
                        'pm_implied': pm_prob,
                        'options_implied': opt_prob,
                        'blended': blended_prob
                    },
                    'metrics': {
                        'dni': dni,
                        'prob_of_profit': prob_of_profit,
                        'expected_value': expected_value,
                        'adjusted_ev': adjusted_ev,
                        'edge_per_downside': edge_per_downside,
                        'kelly_signal': kelly_signal,
                        'is_true_arbitrage': is_true_arbitrage,
                        'actual_costs': actual_costs,
                        'min_payoff_after_costs': min_payoff_after_costs,
                        'penalties': penalties,  # <- add this line (mirror)
                    },
                    'risk_penalties': penalties
                })
                
                enriched_opportunities.append(enriched_opp)
                
            except Exception as e:
                self.logger.warning(f"Failed to process opportunity: {e}")
                continue
        
        # Step 2: Remove dominated strategies
        non_dominated = self._remove_dominated_strategies(enriched_opportunities)
        
        # Step 3: Lexicographic sorting
        sorted_opps = self._lexicographic_sort(non_dominated)
        
        # Step 4: Add final rankings
        for i, opp in enumerate(sorted_opps):
            opp['rank'] = i + 1
            opp['quality_tier'] = self._assign_quality_tier(opp)
        
        # Aggregated summary logging
        try:
            self.logger.info(
                "ranker: opportunities=%d, skipped=%d, reasons=%s",
                len(opportunities),
                0,
                {k: int(v) for k, v in (self._ranker_counts or {}).items()}
            )
        except Exception:
            pass
        return sorted_opps
    
    def _extract_payoff_profile(self, opp: Dict) -> PayoffProfile:
        """
        Convert an opportunity to a standardized two-state payoff profile.

        IMPORTANT: We no longer infer orientation (YES/NO) or payoff shape from
        market heuristics like yes_price, pm_side, is_bearish, etc. Those guesses
        can invert bearish/bullish cases.

        The strategy builder must provide:
          - 'profit_if_yes': float (standard) or 'payoff_if_yes': float (legacy)
          - 'profit_if_no' : float (standard) or 'payoff_if_no' : float (legacy)
          - optional 'upfront_cashflow': float (defaults to 0)
        """
        # Try standardized fields first, then legacy fields
        py = opp.get('profit_if_yes', opp.get('payoff_if_yes'))
        pn = opp.get('profit_if_no', opp.get('payoff_if_no'))
        # Coalesce missing values to 0.0 and count for summary logging
        missing = False
        try:
            payoff_yes = float(0.0 if py is None else py)
            payoff_no  = float(0.0 if pn is None else pn)
            upfront    = float(opp.get("upfront_cashflow", 0.0))
        except (TypeError, ValueError):
            # If fields are non-numeric, treat as invalid and set to 0.0
            payoff_yes = 0.0; payoff_no = 0.0; upfront = float(opp.get("upfront_cashflow", 0.0))
            self._ranker_counts["invalid_fields"] = self._ranker_counts.get("invalid_fields", 0) + 1
        else:
            if py is None or pn is None:
                self._ranker_counts["missing_payoffs"] = self._ranker_counts.get("missing_payoffs", 0) + 1

        return PayoffProfile(
            upfront_cashflow=upfront,
            payoff_if_yes=payoff_yes,
            payoff_if_no=payoff_no
        )
    
    def _get_pm_probability(self, opp: Dict) -> float:
        """
        PM-implied probability. Prefer an explicit `p` if present,
        otherwise derive from yes_price / (1 - fee).
        """
        # 1) explicit p
        for key_path in (("p",), ("polymarket","p"), ("probability",)):
            cur, ok = opp, True
            for k in key_path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False; break
            if ok:
                return self._clip_prob(cur)

        # 2) price from normalized block
        if 'polymarket' in opp and 'yes_price' in opp['polymarket']:
            c = float(opp['polymarket']['yes_price'])
            f = float(self.pm_fee)
            # Correct formula for winner's fee: p = c / ((1-f) + f*c)
            raw = c / ((1.0 - f) + f * c)
            return self._clip_prob(raw)

        # 3) robust fallbacks for legacy shapes
        fallback_price = (
            opp.get('pm_price') or
            (opp.get('polymarket_contract', {}) or {}).get('yes_price')
        )
        if fallback_price is not None:
            c = float(fallback_price)
            f = float(self.pm_fee)
            # Correct formula for winner's fee: p = c / ((1-f) + f*c)
            raw = c / ((1.0 - f) + f * c)
            return self._clip_prob(raw)

        # 4) last resort
        return 0.5
    
    def _get_options_implied_probability(self, opp: Dict) -> Optional[float]:
        """
        Extract options-implied probability using Breeden-Litzenberger formula.
        
        Based on Breeden & Litzenberger (1978) and Malz (2014):
        P(S_T > K) = e^(rT) × (-∂C/∂K) at strike K
        
        For risk-neutral probability from market prices:
        - The derivative of call prices gives the (discounted) probability
        - Must apply risk-free rate adjustment for proper probability
        """
        # Check if this is an options strategy
        if opp.get('hedge_type') != 'options':
            return None
            
        # Try to extract options chain data if available
        options_data = opp.get('options_data', {})
        
        # If we have detailed options data with strikes and prices
        if options_data and 'strikes' in options_data and 'call_prices' in options_data:
            strikes = options_data.get('strikes', [])
            call_prices = options_data.get('call_prices', [])
            
            if len(strikes) < 3 or len(call_prices) < 3:
                # Insufficient data for derivative calculation
                return self._fallback_options_probability(opp)
            
            # Get PM strike and time to expiry
            pm_data = opp.get('polymarket', {})
            pm_strike = pm_data.get('strike_price', pm_data.get('strike', 0))
            days_to_option_expiry = opp.get('days_to_option_expiry', 30)
            time_to_expiry = days_to_option_expiry / DAYS_PER_YEAR
            
            if not pm_strike or pm_strike <= 0:
                return self._fallback_options_probability(opp)
            
            try:
                # Simple shape checks
                diffs = np.diff(call_prices)
                second_diffs = np.diff(diffs)
                is_nonmonotone = (diffs > 0).any()
                is_nonconvex   = (second_diffs < -1e-12).any()

                kind = 'linear' if (is_nonmonotone or is_nonconvex) else 'cubic'
                
                # Build interpolation function for smooth derivative
                price_func = interp1d(
                    strikes, call_prices, 
                    kind=kind,
                    fill_value='extrapolate',
                    bounds_error=False
                )
                
                # Calculate -∂C/∂K using central difference
                # Step size: min of strike spacing or 0.1% of strike
                h_grid = np.diff(strikes).min() if len(strikes) > 1 else pm_strike * 0.001
                h = max(h_grid, pm_strike * 1e-6)  # Add floor for numerical stability
                
                # Calculate derivative using central difference
                dC_dK = (price_func(pm_strike + h) - price_func(pm_strike - h)) / (2 * h)
                
                # Binary option value = -∂C/∂K
                binary_value = -dC_dK
                
                # Apply risk-neutral conversion with proper discount factor
                # Risk-neutral probability requires adjusting for time value
                # P(S_T > K) = e^(rT) × (-∂C/∂K)
                # This is because the option prices C(K) are discounted present values,
                # but we need the risk-neutral probability which is undiscounted
                
                risk_free_rate = getattr(self, 'risk_free_rate', RISK_FREE_RATE)
                discount_factor = np.exp(risk_free_rate * time_to_expiry)
                risk_neutral_prob = discount_factor * binary_value
                
                # Ensure probability is in valid range [0, 1]
                return np.clip(risk_neutral_prob, 0, 1)
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate options-implied probability: {e}")
                return self._fallback_options_probability(opp)
        else:
            # No detailed options data available
            return self._fallback_options_probability(opp)
    
    def _fallback_options_probability(self, opp: Dict) -> Optional[float]:
        """
        Risk-neutral fallback when a full options surface is unavailable.
        P(S_T > K) = N(d2) under Black–Scholes with default sigma if needed.
        """
        S = opp.get('current_spot') or opp.get('spot')
        pm = opp.get('polymarket', {})
        K = pm.get('strike_price', pm.get('strike'))
        days = opp.get('days_to_option_expiry', pm.get('days_to_expiry', 30))
        if not (S and K and days and days > 0):
            # count missing inputs (esp. spot)
            if not S:
                self._ranker_counts["missing_spot"] = self._ranker_counts.get("missing_spot", 0) + 1
            return None

        T = days / DAYS_PER_YEAR
        sigma = getattr(self, 'default_sigma', DEFAULT_IMPLIED_VOLATILITY)
        r = getattr(self, 'risk_free_rate', RISK_FREE_RATE)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(np.clip(norm.cdf(d2), 0.0, 1.0))
    
    def _estimate_pm_variance(self, opp: Dict) -> float:
        """
        Estimate variance of PM probability signal based on bid-ask spread and depth.
        Wider spreads and thinner books = higher variance.
        """
        pm_data = opp.get('polymarket', {})
        
        # Get bid-ask if available
        bid = pm_data.get('bid', pm_data.get('yes_price', 0.5) - 0.01)
        ask = pm_data.get('ask', pm_data.get('yes_price', 0.5) + 0.01)
        
        # Spread as proxy for uncertainty
        spread = ask - bid
        
        # Depth/liquidity if available
        liquidity = pm_data.get('liquidity', pm_data.get('volume_24h', 10000))
        
        # Variance estimate: spread^2 scaled by liquidity
        # More liquid markets have tighter, more reliable quotes
        liquidity_factor = min(liquidity / 100000, 1.0)  # Normalize to 100k baseline
        
        # Base variance from spread, reduced by liquidity
        variance = (spread ** 2) / (1 + liquidity_factor * 10)
        
        return max(variance, 0.0001)  # Minimum variance floor
    
    def _estimate_options_variance(self, opp: Dict) -> float:
        """
        Estimate variance of options-implied probability based on bid-ask spreads
        and spline interpolation uncertainty.
        """
        # Check if we have detailed options data
        if 'options_data' not in opp or 'strikes' not in opp['options_data']:
            # High variance if no proper options data
            return 0.1  # 10% std dev
            
        options_data = opp['options_data']
        strikes = options_data.get('strikes', [])
        
        # If sparse strikes, higher variance
        strike_density = len(strikes)
        if strike_density < 5:
            return 0.05  # 5% std dev for sparse data
            
        # Check bid-ask spreads if available
        if 'evaluated_options_sample' in opp:
            spread_sum = 0
            count = 0
            
            for opt in opp['evaluated_options_sample']:
                bid = opt.get('bid', 0)
                ask = opt.get('ask', 0)
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    spread_pct = (ask - bid) / mid
                    spread_sum += spread_pct
                    count += 1
                    
            if count > 0:
                avg_spread = spread_sum / count
                # Variance proportional to average spread
                return (avg_spread ** 2) / 4  # Empirical scaling
                
        # Default moderate variance
        return 0.01  # 1% std dev
    
    def _blend_probabilities(self, pm_prob: float, opt_prob: Optional[float]) -> float:
        """
        Blend prediction-market and options-implied probabilities.
        """
        import numpy as np

        pm_prob = self._clip_prob(pm_prob)
        if opt_prob is None:
            return pm_prob

        opt_prob = self._clip_prob(opt_prob)

        if self.prob_blend_mode == "fixed":
            blended = self.prob_blend_weight * pm_prob + (1 - self.prob_blend_weight) * opt_prob
            return float(np.clip(blended, 0.0, 1.0))

        if self.prob_blend_mode == "bayesian":
            # simple precision-weighted example (kept consistent with your code)
            pm_sigma = 0.08
            opt_sigma = 0.12
            precision_pm = 1.0 / (pm_sigma ** 2)
            precision_opt = 1.0 / (opt_sigma ** 2)
            total = precision_pm + precision_opt
            w_pm = precision_pm / total
            w_opt = precision_opt / total
            blended = w_pm * pm_prob + w_opt * opt_prob
            self.logger.debug("Precision blend: PM weight=%.3f, Opt weight=%.3f", w_pm, w_opt)
            return float(np.clip(blended, 0.0, 1.0))

        self.logger.warning("Unknown blend mode '%s', using fixed weights", self.prob_blend_mode)
        blended = self.prob_blend_weight * pm_prob + (1 - self.prob_blend_weight) * opt_prob
        return float(np.clip(blended, 0.0, 1.0))
    
    def _calculate_dni(self, opp: Dict, payoff: PayoffProfile) -> float:
        """
        Distance to No-Arbitrage (DNI)
        Positive DNI = guaranteed profit (true arbitrage)
        
        Based on static replication theory from Carr, Ellis & Gupta (1998)
        """
        # Get time to resolution
        days = opp.get('polymarket', {}).get('days_to_expiry', 30)
        T = max(days, 0) / DAYS_PER_YEAR  # use config constant
        
        # Discount factor
        df = np.exp(-self.risk_free_rate * T)
        
        # For state-independent profit (both payoffs equal)
        if payoff.is_state_independent:
            pv_payoff = payoff.payoff_if_yes * df  # Same as payoff_if_no
        else:
            # For state-dependent strategies
            # DNI = present value of minimum profit across states
            min_payoff = min(payoff.payoff_if_yes, payoff.payoff_if_no)
            pv_payoff = min_payoff * df
        
        # Account for actual transaction costs
        actual_costs = self._calculate_actual_costs(opp, payoff)
        
        return pv_payoff - actual_costs
    
    def _calculate_profit_probability(self, payoff: PayoffProfile, prob_yes: float) -> float:
        """
        Calculate probability of finishing with positive P&L.
        Cost-aware probability of finishing positive.
        """
        # Pull minimal opp context via a stored handle if present
        opp = getattr(self, '_current_opp', None)
        costs = 0.0
        if opp is not None:
            try:
                costs = self._calculate_actual_costs(opp, payoff)
            except Exception:
                costs = 0.0

        py = payoff.payoff_if_yes - costs
        pn = payoff.payoff_if_no  - costs
        if py > 0 and pn > 0: return 1.0
        if py <= 0 and pn <= 0: return 0.0
        return prob_yes if py > 0 else (1.0 - prob_yes)
    
    def _calculate_expected_value(self, payoff: PayoffProfile, prob_yes: float) -> float:
        """
        Expected value = probability-weighted average of payoffs.
        """
        ev = prob_yes * payoff.payoff_if_yes + (1 - prob_yes) * payoff.payoff_if_no
        return ev
    
    def _calculate_risk_penalties(self, opp: Dict, payoff: PayoffProfile) -> Dict[str, float]:
        """
        Calculate penalties for various risk factors.
        Based on limits-to-arbitrage literature (Shleifer & Vishny 1997).
        """
        penalties = {
            'vega': 0,
            'funding': 0,
            'liquidity': 0,
            'total': 0
        }
        
        hedge_type = opp.get('hedge_type', opp.get('strategy_type', ''))
        
        # 1. Vega penalty for options (time/volatility mismatch)
        if hedge_type == 'options':
            # Get option details from opportunity
            days_to_option_expiry = opp.get('days_to_option_expiry', 30)
            days_to_pm_expiry = opp.get('polymarket', {}).get('days_to_expiry', 30)
            
            # Calculate time mismatch factor
            time_mismatch_days = abs(days_to_option_expiry - days_to_pm_expiry)
            time_factor = time_mismatch_days / DAYS_PER_YEAR  # Normalize to years
            
            # Position value estimate
            position_value = abs(payoff.upfront_cashflow) if payoff.upfront_cashflow != 0 else 1000
            
            # Calculate actual vega using Black-Scholes
            vega_exposure = 0.0
            current_spot = opp.get('current_spot')
            
            # Extract option details if available
            if 'evaluated_options_sample' in opp and current_spot is not None:
                # Use actual options data for accurate vega
                options_sample = opp['evaluated_options_sample']
                
                vega_exposure = calculate_vega_penalty_accurate(
                    options_sample,
                    current_spot,
                    risk_free_rate=self.risk_free_rate
                )
            elif current_spot is not None:
                # Fallback to binary option vega approximation
                pm_strike = opp.get('polymarket', {}).get('strike', current_spot)
                time_to_expiry = days_to_option_expiry / DAYS_PER_YEAR
                
                # Try to get market-implied volatility
                currency = opp.get('currency', 'ETH')
                volatility = self.market_analyzer.get_market_implied_volatility(
                    currency=currency,
                    options_data=None,  # No options data in this path
                    use_cache=True
                )
                
                # Use default if market volatility not available
                if volatility is None:
                    volatility = DEFAULT_IMPLIED_VOLATILITY
                
                # Binary option vega for the PM contract
                binary_vega = BlackScholesGreeks.binary_option_vega(
                    S=current_spot,
                    K=pm_strike,
                    r=self.risk_free_rate,
                    sigma=volatility,  # Market-implied or default volatility
                    T=time_to_expiry
                )
                
                vega_exposure = abs(binary_vega) * position_value
            else:
                # Missing spot price, log warning but continue
                self.logger.warning("Missing current spot price for vega calculation, setting vega_exposure=0")
                vega_exposure = 0.0
            
            # Vega penalty based on time mismatch and exposure
            penalties['vega'] = self.vega_lambda * vega_exposure * time_factor
            
        # 2. Funding penalty for perpetuals
        elif hedge_type == 'perpetuals':
            # Get funding data
            funding_rate = opp.get('funding_rate', 0)
            funding_rate_annual = opp.get('funding_rate_annual', abs(funding_rate) * 3 * DAYS_PER_YEAR)
            
            # Calculate funding uncertainty as annualized volatility
            # Using 30% of the funding rate as uncertainty estimate
            funding_uncertainty = abs(funding_rate_annual) * 0.3
            
            # Position size estimate
            position_size = abs(opp.get('max_profit', 0) + opp.get('max_loss', 0)) / 2
            
            # Time factor based on actual holding period
            days_to_expiry = opp.get('polymarket', {}).get('days_to_expiry', 30)
            time_factor = days_to_expiry / DAYS_PER_YEAR  # Normalize to years
            
            # Funding risk = uncertainty * position * sqrt(time) due to volatility scaling
            penalties['funding'] = self.funding_kappa * funding_uncertainty * position_size * np.sqrt(time_factor)
            
        # 3. Liquidity penalty (bid-ask spreads, market impact)
        # More sophisticated liquidity cost model based on actual market data
        liquidity_penalty = 0.0
        
        # Extract market data for liquidity analysis
        hedge_instruments = opp.get('hedge_instruments', {})
        if not hedge_instruments:
            self.logger.debug("Opportunity missing 'hedge_instruments' - %s", opp.get('strategy', 'Unknown'))
        pm_data = opp.get('polymarket', {})
        
        # Use consistent notional based on Kelly stake
        notional = self._calculate_kelly_stake(payoff)
        if notional <= 0:
            return penalties  # No liquidity penalty if no stake
        
        # A. Hedge instrument liquidity costs (options or perps)
        if hedge_type == 'options':
            # For options, check bid-ask spread from actual data
            if 'evaluated_options_sample' in opp:
                total_spread_cost = 0
                option_count = 0
                
                for opt in opp.get('evaluated_options_sample', []):
                    bid = opt.get('bid', 0)
                    ask = opt.get('ask', 0)
                    if bid > 0 and ask > 0:
                        # Effective spread as percentage
                        spread_pct = (ask - bid) / ((ask + bid) / 2)
                        # Cost to cross half the spread
                        cross_cost = 0.5 * spread_pct * abs(opt.get('position', 1))
                        total_spread_cost += cross_cost
                        option_count += 1
                
                if option_count > 0:
                    # Average spread cost across options
                    avg_spread_cost = total_spread_cost / option_count
                    liquidity_penalty += avg_spread_cost * notional
                    
                    # Add market impact for options (typically less liquid than perps)
                    options_depth = opp.get('options_depth', 50000)  # Default 50k depth
                    options_depth = max(options_depth, 1e-9)
                    options_impact = self.market_impact_k * 1.5 * (notional / options_depth)  # 1.5x impact for options
                    liquidity_penalty += options_impact
            else:
                # Fallback: typical options spread + impact
                liquidity_penalty += 0.005 * notional  # 0.5% typical spread
                liquidity_penalty += 0.002 * notional  # 0.2% typical impact
                
        elif hedge_type == 'perpetuals':
            # For perpetuals, check orderbook depth
            perp_data = hedge_instruments.get('perps', opp.get('perps', {}))
            
            if perp_data.get('bid') and perp_data.get('ask'):
                bid = perp_data['bid']
                ask = perp_data['ask']
                mid = (bid + ask) / 2
                
                # Effective spread
                spread_pct = (ask - bid) / mid if mid > 0 else 0.001
                
                # Cost to cross half the spread
                cross_cost = 0.5 * spread_pct * notional
                
                # Market impact based on depth (if available)
                depth_at_mid = perp_data.get('depth_at_mid', 100000)  # Default 100k depth
                # Avoid division by zero
                depth_at_mid = max(depth_at_mid, 1e-9)
                
                # Amihud-style impact: proportional to size/depth
                market_impact = self.market_impact_k * (notional / depth_at_mid)
                
                liquidity_penalty += cross_cost + market_impact
            else:
                # Fallback for missing data
                liquidity_penalty += 0.002 * notional  # 0.2% typical perp spread
        
        # B. Polymarket liquidity costs
        pm_bid = pm_data.get('bid', pm_data.get('yes_price', 0.5) - 0.01)
        pm_ask = pm_data.get('ask', pm_data.get('yes_price', 0.5) + 0.01)
        
        if pm_bid > 0 and pm_ask < 1:
            pm_spread = pm_ask - pm_bid
            # Cost to cross PM spread (in outcome tokens)
            pm_cross_cost = 0.5 * pm_spread * notional
            liquidity_penalty += pm_cross_cost
        else:
            # Fallback: typical PM spread
            liquidity_penalty += 0.01 * notional  # 1% typical PM spread
        
        penalties['liquidity'] = liquidity_penalty
        
        # Total penalty
        penalties['total'] = penalties['vega'] + penalties['funding'] + penalties['liquidity']
        
        return penalties
    
    def _calculate_actual_costs(self, opp: Dict, payoff: PayoffProfile) -> float:
        """
        Calculate ACTUAL costs (not risk penalties) for true arbitrage determination.
        Includes:
        - Transaction costs (fees)
        - Bid-ask spreads
        - Market impact
        - Any known funding costs (for perps)
        
        Does NOT include risk penalties like vega or funding uncertainty.
        """
        total_costs = 0.0
        
        # Get hedge type for this opportunity
        hedge_type = opp.get('hedge_type', opp.get('strategy_type', ''))
        
        # 1. Transaction costs (exchange fees)
        # Use actual traded size, not P&L values
        total_notional = 0.0
        
        # Try proper size fields in priority order
        for k in ('required_capital', 'position_size', 'stake', 'portfolio_cost', 'notional'):
            if k in opp and opp[k] is not None:
                total_notional = max(total_notional, abs(float(opp[k])))
                
        # Fallback: use upfront cashflow if it represents real outlay
        if total_notional == 0.0 and payoff.upfront_cashflow < 0:
            total_notional = abs(payoff.upfront_cashflow)
            
        # Last resort fallback for edge cases
        if total_notional <= 1e-12:
            # This should rarely happen with properly structured opportunities
            self.logger.warning("No position size found, using P&L as last resort for costs")
            total_notional = max(abs(payoff.payoff_if_yes), abs(payoff.payoff_if_no))
        
        transaction_costs = total_notional * self.transaction_cost_rate
        total_costs += transaction_costs
        
        # 2. Bid-ask spread costs
        spread_costs = 0.0
        
        # A. Polymarket spread
        pm_data = opp.get('polymarket', {})
        pm_bid = pm_data.get('bid', pm_data.get('yes_price', 0.5) - 0.01)
        pm_ask = pm_data.get('ask', pm_data.get('yes_price', 0.5) + 0.01)
        
        if pm_bid > 0 and pm_ask < 1:
            pm_spread = pm_ask - pm_bid
            # Cost to cross PM spread (half-spread on entry)
            pm_notional = abs(payoff.upfront_cashflow) if payoff.upfront_cashflow != 0 else 1000
            spread_costs += 0.5 * pm_spread * pm_notional
        else:
            # Fallback: typical PM spread
            pm_notional = abs(payoff.upfront_cashflow) if payoff.upfront_cashflow != 0 else 1000
            spread_costs += 0.005 * pm_notional  # 0.5% half-spread
        
        # B. Hedge instrument spread
        if hedge_type == 'options':
            # Options spread from actual data
            if 'evaluated_options_sample' in opp:
                options_sample = opp['evaluated_options_sample']
                total_option_spread = 0
                for opt in options_sample:
                    if isinstance(opt, dict):
                        bid = opt.get('bid', 0)
                        ask = opt.get('ask', 0)
                        weight = abs(opt.get('weight', 0))
                        if bid > 0 and ask > 0 and weight > 0:
                            spread = (ask - bid) / ((ask + bid) / 2)
                            # Half-spread cost weighted by position
                            total_option_spread += 0.5 * spread * weight * ask * 100  # Assume $100 position
                spread_costs += total_option_spread
            else:
                # Fallback for options
                spread_costs += 0.01 * abs(opp.get('portfolio_cost', 1000))  # 1% half-spread
                
        elif hedge_type == 'perpetuals':
            # Perp spread
            perp_bid = opp.get('bid')
            perp_ask = opp.get('ask')
            
            if perp_bid and perp_ask and perp_bid > 0:
                perp_spread = (perp_ask - perp_bid) / perp_bid
                position_size = abs(opp.get('max_profit', 0) + opp.get('max_loss', 0)) / 2
                spread_costs += 0.5 * perp_spread * position_size
            else:
                # Fallback for perps
                position_size = abs(opp.get('max_profit', 0) + opp.get('max_loss', 0)) / 2
                spread_costs += 0.001 * position_size  # 0.1% half-spread
        
        total_costs += spread_costs
        
        # 3. Market impact (simplified model)
        # Based on notional size relative to typical liquidity
        impact_cost = 0.0
        
        # Rough impact model: cost = k * sqrt(size/ADV)
        # Using simplified version: impact = k * size
        total_size = total_notional
        impact_cost = self.market_impact_k * 0.0001 * total_size  # 0.01% per $1000
        
        total_costs += impact_cost
        
        # 4. Known funding costs (for perps, if holding to expiry)
        if hedge_type == 'perpetuals':
            funding_rate = opp.get('funding_rate', 0)
            days_to_expiry = opp.get('polymarket', {}).get('days_to_expiry', 30)
            funding_periods = days_to_expiry * 3  # 3 funding periods per day
            
            # Only include if we're paying funding (not receiving)
            direction = opp.get('direction', '')
            if (direction == 'LONG' and funding_rate > 0) or (direction == 'SHORT' and funding_rate < 0):
                position_size = abs(opp.get('max_profit', 0) + opp.get('max_loss', 0)) / 2
                funding_cost = abs(funding_rate) * funding_periods * position_size
                total_costs += funding_cost
        
        return total_costs
    
    def _calculate_edge_per_downside(self, payoff: PayoffProfile, expected_value: float) -> float:
        """
        Risk-adjusted metric: expected value per unit of downside risk.
        
        Returns -inf for negative EV to ensure losing trades sink to bottom.
        """
        # If negative expected value, return -infinity for clean sorting
        if expected_value < 0:
            return float('-inf')
            
        max_loss = abs(payoff.max_loss)
        if max_loss < 0.01:  # Avoid division by zero
            return float('inf') if expected_value > 0 else 0
            
        return expected_value / max_loss
    
    def _calculate_kelly_stake(self, payoff: PayoffProfile) -> float:
        """
        Calculate the Kelly stake (capital at risk) consistently.
        Used for Kelly criterion, dominance normalization, and liquidity penalties.
        
        Returns the absolute amount at risk.
        """
        # For options spreads, the stake is the upfront cash outlay if negative,
        # otherwise it's the maximum loss
        return -payoff.upfront_cashflow if payoff.upfront_cashflow < 0 else abs(payoff.max_loss)
    
    def _calculate_kelly_signal(self, payoff: PayoffProfile, prob_yes: float) -> float:
        """
        Kelly criterion signal for position sizing.
        Based on Kelly (1956) and modern applications.
        
        Returns fraction of bankroll to bet (capped at kelly_cap).
        """
        # Determine stake (capital at risk)
        stake = self._calculate_kelly_stake(payoff)
        
        if stake <= 0:
            return 0
            
        # Determine which side we're betting on and get net P&L
        if payoff.payoff_if_yes > payoff.payoff_if_no:
            # Betting on YES
            win_profit = payoff.payoff_if_yes  # Already net P&L per dataclass
            p = prob_yes
        else:
            # Betting on NO
            win_profit = payoff.payoff_if_no   # Already net P&L per dataclass
            p = 1 - prob_yes
            
        # Avoid negative or zero profits
        if win_profit <= 0:
            return 0
            
        # Kelly formula: f* = p - q/b
        # where p = win prob, q = loss prob, b = profit/stake ratio
        b = win_profit / stake
        q = 1 - p
        f = p - q / b
        
        # Cap at maximum fraction
        return min(max(f, 0), self.kelly_cap)
    
    def _normalize_payoffs(self, payoff: PayoffProfile) -> Tuple[float, float]:
        """
        Normalize payoffs by stake to enable size-independent comparison.
        Returns (normalized_if_yes, normalized_if_no).
        """
        stake = self._calculate_kelly_stake(payoff)
        # Avoid division by zero
        if stake < 1e-12:
            stake = 1e-12
            
        return payoff.payoff_if_yes / stake, payoff.payoff_if_no / stake
    
    def _remove_dominated_strategies(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Remove strategies that are strictly dominated by others.
        A dominates B if A has better normalized payoffs in all states for the same resolution event.
        
        Normalized dominance: Compare unit payoffs (per dollar at risk) to avoid size bias.
        """
        # Group by question id (fallback to question text; empty -> its own group)
        groups = {}
        for opp in opportunities:
            pm = opp.get('polymarket', opp.get('polymarket_contract', {}))
            qid = pm.get('question_id') or pm.get('question') or f"__ungrouped_{id(opp)}"
            groups.setdefault(qid, []).append(opp)

        non_dominated = []
        for qid, group in groups.items():
            for i, opp_a in enumerate(group):
                is_dominated = False
                pa = opp_a['payoff_profile']
                payoff_a = PayoffProfile(
                    pa['upfront_cashflow'], pa['payoff_if_yes'], pa['payoff_if_no']
                )
                norm_a_yes, norm_a_no = self._normalize_payoffs(payoff_a)

                for j, opp_b in enumerate(group):
                    if i == j: 
                        continue
                    pb = opp_b['payoff_profile']
                    payoff_b = PayoffProfile(
                        pb['upfront_cashflow'], pb['payoff_if_yes'], pb['payoff_if_no']
                    )
                    norm_b_yes, norm_b_no = self._normalize_payoffs(payoff_b)

                    # B dominates A if both states are weakly better and at least one strictly better
                    if (norm_b_yes >= norm_a_yes and norm_b_no >= norm_a_no and
                        (norm_b_yes > norm_a_yes or norm_b_no > norm_a_no)):
                        is_dominated = True
                        self.logger.debug(
                            f"Strategy {i} dominated by {j} for question: {qid[:50]}... "
                            f"(normalized: {norm_a_yes:.3f}/{norm_a_no:.3f} vs {norm_b_yes:.3f}/{norm_b_no:.3f})"
                        )
                        break
                if not is_dominated:
                    non_dominated.append(opp_a)

        self.logger.info(f"Removed {len(opportunities) - len(non_dominated)} dominated strategies")
        return non_dominated
    
    def _lexicographic_sort(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Sort opportunities using lexicographic ordering.
        Priority order:
        1. True arbitrage (DNI >= 0 and state-independent)
        2. Distance to no-arbitrage (DNI)
        3. Probability of profit
        4. Edge per downside
        5. Adjusted expected value
        6. Time to resolution
        """
        def sort_key(opp):
            metrics = opp['metrics']
            
            # Extract key values
            is_true_arb = metrics['is_true_arbitrage']
            dni = metrics['dni']
            prob_profit = metrics['prob_of_profit']
            edge_downside = metrics['edge_per_downside']
            adjusted_ev = metrics['adjusted_ev']
            
            # Time to resolution - use actual days to expiry
            pm = opp.get('polymarket', {})
            time_to_resolution = pm.get('days_to_expiry', opp.get('days_to_resolution', 9_999))
            
            # Handle infinities
            if edge_downside == float('inf'):
                edge_downside = 1e6
                
            return (
                -int(is_true_arb),  # True arbs first (negative for descending)
                -dni,               # Higher DNI first
                -prob_profit,       # Higher probability first
                -edge_downside,     # Higher edge/downside first
                -adjusted_ev,       # Higher adjusted EV first
                time_to_resolution  # Shorter time first
            )
            
        return sorted(opportunities, key=sort_key)
    
    
    def _assign_quality_tier(self, opp: Dict) -> str:
        """Assign quality tier using metrics **and** basic executability checks."""
        metrics = opp['metrics']

        # Guardrail: if required option legs lack live quotes, demote to SPECULATIVE
        req = opp.get('required_options') or []
        if req:
            for leg in req:
                bid = float(leg.get('bid') or 0.0)
                ask = float(leg.get('ask') or 0.0)
                side = (leg.get('action') or 'BUY').upper()
                need = ask if side == 'BUY' else bid
                # Also honor explicit flags from data enricher
                if leg.get('has_live_quotes') is False or leg.get('skip_for_execution') is True or need <= 0.0:
                    return "SPECULATIVE"

        if metrics.get('is_true_arbitrage'):
            return "TRUE_ARBITRAGE"
        elif metrics['dni'] > self.near_arb_dni_threshold and metrics['prob_of_profit'] > self.near_arb_prob_threshold:
            return "NEAR_ARBITRAGE"
        elif metrics['prob_of_profit'] > self.high_prob_threshold:
            return "HIGH_PROBABILITY"
        elif metrics['prob_of_profit'] > self.moderate_prob_threshold:
            return "MODERATE_PROBABILITY"
        else:
            return "SPECULATIVE"


def analyze_opportunities_file(filepath: str, output_path: Optional[str] = None) -> List[Dict]:
    """
    Analyze a JSON file of opportunities and rank them by probability metrics.
    """
    # Load opportunities
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    opportunities = data['opportunities']
    
    # Create ranker
    ranker = ProbabilityRanker()
    
    # Rank opportunities
    ranked = ranker.rank_opportunities(opportunities)
    
    # Save results if output path provided
    if output_path:
        output_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'original_count': len(opportunities),
            'ranked_count': len(ranked),
            'ranking_parameters': {
                'polymarket_fee': ranker.pm_fee,
                'probability_blend_weight': ranker.prob_blend_weight,
                'vega_penalty_lambda': ranker.vega_lambda,
                'funding_penalty_kappa': ranker.funding_kappa,
                'kelly_fraction_cap': ranker.kelly_cap
            },
            'opportunities': ranked
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
    return ranked


def display_top_opportunities(ranked_opportunities: List[Dict], top_n: int = 20):
    """
    Display the top N opportunities in a readable format.
    """
    print("\n" + "="*80)
    print("TOP ARBITRAGE OPPORTUNITIES (Ranked by Probability-Weighted Metrics)")
    print("="*80)
    
    # Group by quality tier
    tiers = {}
    for opp in ranked_opportunities[:top_n]:
        tier = opp['quality_tier']
        if tier not in tiers:
            tiers[tier] = []
        tiers[tier].append(opp)
    
    # Display by tier
    tier_order = ["TRUE_ARBITRAGE", "NEAR_ARBITRAGE", "HIGH_PROBABILITY", 
                  "MODERATE_PROBABILITY", "SPECULATIVE"]
    
    for tier in tier_order:
        if tier not in tiers:
            continue
            
        print(f"\n{tier}:")
        print("-" * 60)
        
        for opp in tiers[tier]:
            metrics = opp['metrics']
            # Handle both field names
            pm = opp.get('polymarket', opp.get('polymarket_contract', {}))
            
            print(f"\n#{opp['rank']}: {opp['strategy']}")
            print(f"   Market: {pm.get('question', 'Unknown')[:60]}...")
            print(f"   Strike: ${pm.get('strike', pm.get('strike_price', 0)):,.0f} | YES Price: {pm.get('yes_price', 0):.1%}")
            print(f"   Probability of Profit: {metrics['prob_of_profit']:.1%}")
            print(f"   Expected Value: ${metrics['expected_value']:,.2f}")
            print(f"   Adjusted EV (after penalties): ${metrics['adjusted_ev']:,.2f}")
            print(f"   Edge/Downside: {metrics['edge_per_downside']:.2f}")
            print(f"   Distance to No-Arb: ${metrics['dni']:,.2f}")
            
            if metrics['is_true_arbitrage']:
                print(f"   ✅ TRUE ARBITRAGE - Profit in all scenarios!")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"Analyzing opportunities from: {input_file}")
        ranked = analyze_opportunities_file(input_file, output_file)
        
        display_top_opportunities(ranked)
        
        if output_file:
            print(f"\nRanked opportunities saved to: {output_file}")
    else:
        print("Usage: python probability_ranker.py <input_json> [output_json]")