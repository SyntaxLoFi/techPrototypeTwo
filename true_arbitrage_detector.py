"""
True Arbitrage Detector - FIXED VERSION

This module implements proper arbitrage detection that:
1. Identifies opportunities with profit regardless of outcome (true arbitrage)
2. Sorts by probability of profit for directional strategies
3. Uses realistic position sizes and profit calculations
4. Validates that binary option payouts are mathematically possible

CRITICAL FIXES:
- Position sizes limited to realistic amounts
- Binary option payouts capped at $1 per share  
- True arbitrage prioritized over directional bets
- Probability-based sorting for non-arbitrage opportunities
"""

from typing import Dict, List, Optional, Tuple
import logging
import math
from config_manager import (
    SETTLE_SPREAD_AT_PM_EXPIRY, OPTIONS_UNWIND_MODEL,
    STRICT_ARBITRAGE_MODE, LABEL_TRUE_ARB_FROM_DETECTOR
)

# --- Config fallbacks (keeps module working if config isn't available) ---
try:
    from config_manager import TRUE_ARBITRAGE_BONUS
except Exception:
    TRUE_ARBITRAGE_BONUS = 10_000

try:
    from config_manager import DEFAULT_VOLATILITY
except Exception:
    DEFAULT_VOLATILITY = 0.60  # 60% annualized as a conservative default for crypto

# --- Normal CDF with SciPy fallback ---
try:
    from scipy.stats import norm as _norm
    def _ncdf(z: float) -> float:
        return float(_norm.cdf(z))
except Exception:
    def _ncdf(z: float) -> float:
        # Numerical stable-ish fallback
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

logger = logging.getLogger(__name__)

__all__ = ["TrueArbitrageDetector"]

# Single source of truth for event-boundary epsilon
EVENT_EPS = 1e-12


def _extract_spread_width_and_strike(opportunity: Dict) -> Tuple[float, float]:
    """
    Best‑effort extraction of (K, δ) from the opportunity.
    Falls back to (K, 0.0) if unavailable.
    """
    contract = (opportunity.get('polymarket_contract', {}) or {})
    K = float(contract.get('strike_price', 0.0)) or 0.0
    width = 0.0
    # Prefer explicit width from digital_quote
    dq = opportunity.get('digital_quote') or {}
    try:
        width = float(dq.get('width', 0.0)) or 0.0
    except Exception:
        width = 0.0
    if width <= 0.0:
        # Infer from required_options legs (two‑leg vertical)
        legs = opportunity.get('required_options') or []
        strikes = [float(l.get('strike', 0.0)) for l in legs if l.get('strike') is not None]
        if len(strikes) >= 2:
            width = abs(max(strikes) - min(strikes))
    return K, width

# Position size limits for realistic constraints
MAX_PM_SHARES = 100_000
MAX_HEDGE_NOTIONAL_USD = 2_000_000

def _clamp01(x: float) -> float:
    x = float(x)
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x


def _prob_event_black_scholes(S0: float, K: float, T: float, sigma: float, r: float, is_above: bool) -> float:
    """
    Calculate the risk-neutral probability of the event occurring using Black-Scholes formula.
    Returns N(d2) for above contracts, 1-N(d2) for below contracts.
    """
    S0 = float(S0)
    K = float(K)
    T = float(T)
    sigma = max(1e-12, float(sigma))
    if S0 <= 0.0 or K <= 0.0 or T <= 0.0:
        return 0.5
    d2 = (math.log(S0 / K) + (r - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    p_above = _ncdf(d2)
    return p_above if is_above else (1.0 - p_above)


def _check_yes_no_pair(yes: float, no: float, fees_bps: float = 30.0) -> dict:
    """
    Check for YES+NO pair arbitrage opportunities.
    Returns dict with sum, buy_both (if sum < 1-fees), sell_both (if sum > 1+fees).
    """
    yes = _clamp01(yes)
    no = _clamp01(no)
    fees = fees_bps / 10_000.0
    s = yes + no
    return {
        "sum": s,
        "buy_both": (s < 1.0 - fees),    # lock in $1, pay < 1
        "sell_both": (s > 1.0 + fees),   # collect > 1, owe $1
    }


class TrueArbitrageDetector:
    """
    Detects and validates true arbitrage opportunities vs directional strategies.
    """
    
    def __init__(self, risk_free_rate: float = 0.05, default_vol: float = DEFAULT_VOLATILITY):
        self.risk_free_rate = float(risk_free_rate)
        self.default_vol = float(default_vol)
        self.logger = logging.getLogger('TrueArbitrageDetector')
    
    def classify_opportunity(self, opportunity: Dict) -> Dict:
        """
        Classify an opportunity as true arbitrage vs directional strategy.
        
        Returns enhanced opportunity with:
        - is_true_arbitrage: bool
        - arbitrage_type: str  
        - profit_probability: float
        - classification_reason: str
        """
        # Compute min/max P&L if not already present
        min_band_pnl, max_band_pnl = self._min_max_pnl_over_band(opportunity)
        
        # Overwrite/expose canonical values for downstream ranking/UX
        opportunity['min_band_pnl'] = min_band_pnl
        opportunity['max_band_pnl'] = max_band_pnl
        opportunity['max_profit'] = max_band_pnl
        opportunity['max_loss'] = -min_band_pnl if min_band_pnl < 0 else 0.0
        
        max_profit = opportunity.get('max_profit', 0)
        max_loss = abs(opportunity.get('max_loss', 0))
        
        # Check for YES+NO pair arbitrage
        contract = opportunity.get('polymarket_contract', {}) or {}
        pricing = opportunity.get('pricing_details', {}) or {}
        yes_price = contract.get('yes_price', pricing.get('yes_price'))
        no_price = contract.get('no_price', pricing.get('no_price'))
        
        if yes_price is not None and no_price is not None:
            pair_check = _check_yes_no_pair(yes_price, no_price)
            opportunity['yes_no_pair_check'] = pair_check
        
        # TRUE ARBITRAGE CRITERIA (profit regardless of outcome)
        is_true_arbitrage = self._is_true_arbitrage(opportunity, max_profit, max_loss)
        
        if is_true_arbitrage:
            classification = self._classify_true_arbitrage(opportunity, max_profit, max_loss)
        else:
            classification = self._classify_directional_strategy(opportunity, max_profit, max_loss)
        
        # Add classification to opportunity
        opportunity.update(classification)
        # Promote nested metrics for UI/filters
        self._promote_strategy_metrics(opportunity)
        
        return opportunity
    
    def _is_true_arbitrage(self, opportunity: Dict, max_profit: float, max_loss: float) -> bool:
        """
        Strict definition: non-negative P&L in both outcomes, plus wide-band robustness.
        """
        try:
            contract = opportunity.get('polymarket_contract', {}) or {}
            pricing = opportunity.get('pricing_details', {}) or {}
            S0 = float(pricing.get('oracle_price', 0.0))
            K = float(contract.get('strike_price', 0.0))
            
            # NEW: require non-trivial exposure
            pm_shares = float(opportunity.get('pm_shares', 0.0))
            explicit_notional = opportunity.get('hedge_notional_usd')
            active_exposure = (pm_shares > 0.0) or (explicit_notional is not None and float(explicit_notional) > 0.0)
            if not active_exposure or S0 <= 0.0 or K <= 0.0:
                return False
            
            pnl_event, pnl_nonevent = self._event_outcome_pnls(opportunity)
            tol = 1e-8
            
            # Must be non-negative in both outcomes ...
            if pnl_event < -tol or pnl_nonevent < -tol:
                return False
            
            # ... and strictly positive in at least one
            if max(pnl_event, pnl_nonevent) <= tol:
                return False
            
            # In strict‑arb mode at PM resolution, the proof is exactly two‑state;
            # do not apply band‑scan robustness.
            if STRICT_ARBITRAGE_MODE and SETTLE_SPREAD_AT_PM_EXPIRY and str(OPTIONS_UNWIND_MODEL) == "intrinsic_only":
                return True
            # Otherwise keep the robustness check for non‑strict modes.
            min_pnl, _ = self._min_max_pnl_over_band(opportunity)
            return min_pnl >= -1e-6
        except Exception as e:
            self.logger.error(
                f"_is_true_arbitrage failed: {e} | opp_id={opportunity.get('id', 'unknown')}"
            )
            return False
    
    def _event_outcome_pnls(self, opportunity: Dict, eps: float = 1e-4) -> Tuple[float, float]:
        """
        P&L in the two binary outcomes at prices just across the strike.
        
        Uses a larger epsilon (1e-4) than _simulate_pnl_at_price (1e-12) to ensure
        we're clearly on either side of the strike boundary for unambiguous outcomes.
        """
        contract = opportunity.get('polymarket_contract', {}) or {}
        pricing = opportunity.get('pricing_details', {}) or {}
        K = float(contract.get('strike_price', 0.0))
        S0 = float(pricing.get('oracle_price', 0.0))
        is_above = bool(contract.get('is_above', True))
        if K <= 0.0 or S0 <= 0.0:
            return (float('-inf'), float('-inf'))
        _, width = _extract_spread_width_and_strike(opportunity)
        if width > 0.0:
            # Evaluate *outside the vertical's sliver* so the digital has settled.
            if is_above:
                # ABOVE: non‑event at K−1, event at K+δ+1
                s_event, s_nonevent = K + width + 1.0, max(1e-9, K - 1.0)
            else:
                # BELOW: event at K−δ−1, non‑event at K+δ+1
                s_event, s_nonevent = max(1e-9, K - width - 1.0), K + width + 1.0
        else:
            # Fallback to epsilon‑offset if width unavailable
            if is_above:
                s_event, s_nonevent = K * (1.0 + eps), max(1e-9, K * (1.0 - eps))
            else:
                s_event, s_nonevent = max(1e-9, K * (1.0 - eps)), K * (1.0 + eps)
        return (
            self._simulate_pnl_at_price(opportunity, s_event),
            self._simulate_pnl_at_price(opportunity, s_nonevent),
        )

    def _min_max_pnl_over_band(
        self, opportunity: Dict, lower_mult: float = 0.5, upper_mult: float = 2.0, steps: int = 121
    ) -> Tuple[float, float]:
        """Scan P&L over a wide band of final prices to test robustness."""
        S0 = float((opportunity.get('pricing_details', {}) or {}).get('oracle_price', 0.0))
        if S0 <= 0.0:
            return (float('-inf'), float('-inf'))
        min_p, max_p = float('inf'), float('-inf')
        for i in range(steps):
            f = lower_mult + (upper_mult - lower_mult) * (i / (steps - 1))
            ST = S0 * f
            pnl = self._simulate_pnl_at_price(opportunity, ST)
            min_p, max_p = min(min_p, pnl), max(max_p, pnl)
        return (min_p, max_p)

    def _count_profitable_scenarios(self, opportunity: Dict) -> Dict:
        """
        Count profitable scenarios by simulating price ranges.
        """
        try:
            contract = opportunity.get('polymarket_contract', {})
            pricing = opportunity.get('pricing_details', {})
            
            current_spot = pricing.get('oracle_price', 0)
            strike_price = contract.get('strike_price', current_spot)
            
            if current_spot <= 0:
                return {'total': 0, 'profitable': 0}
            
            # Test 21 price scenarios from -10% to +10%
            price_scenarios = []
            for i in range(21):
                factor = 0.9 + (i * 0.01)  # 0.90 to 1.10 in 0.01 increments
                price_scenarios.append(current_spot * factor)
            
            profitable_count = 0
            
            for test_price in price_scenarios:
                # Simulate P&L at this price
                pnl = self._simulate_pnl_at_price(opportunity, test_price)
                if pnl > 0:
                    profitable_count += 1
            
            return {
                'total': len(price_scenarios),
                'profitable': profitable_count
            }
            
        except Exception as e:
            self.logger.error(f"Error counting profitable scenarios: {e} | opp_id={opportunity.get('id', 'unknown')}")
            return {'total': 0, 'profitable': 0}
    
    def _simulate_pnl_at_price(self, opportunity: Dict, final_price: float, eps: float = EVENT_EPS) -> float:
        """
        Simulate total P&L at a specific final price. Clamps PM prices, validates pm_side,
        and uses $1-per-share as the natural perp notional unless overridden.
        
        Uses epsilon for consistent event boundary handling - ties default to "no event".
        """
        try:
            contract = opportunity.get('polymarket_contract', {}) or {}
            pricing = opportunity.get('pricing_details', {}) or {}
            
            pm_side = str(opportunity.get('pm_side', 'BUY_YES')).upper()
            pm_shares = min(MAX_PM_SHARES, max(0.0, float(opportunity.get('pm_shares', 0.0))))
            hedge_direction = str(opportunity.get('hedge_direction', 'LONG')).upper()
            strategy_type = str(opportunity.get('strategy_type', 'perps')).lower()
            
            # Validate pm_side early
            if pm_side not in ('BUY_YES', 'BUY_NO'):
                raise ValueError(f"pm_side must be BUY_YES or BUY_NO, got {pm_side!r}")
            
            K = float(contract.get('strike_price', 0.0))
            S0 = float(pricing.get('oracle_price', 0.0))
            if K <= 0.0 or S0 <= 0.0:
                return 0.0
            
            # Pull YES/NO quotes if present; do not force them unless shares must be derived
            y = contract.get('yes_price', pricing.get('yes_price'))
            n = contract.get('no_price', pricing.get('no_price'))
            yes_price = _clamp01(y) if y is not None else None
            no_price  = _clamp01(n) if n is not None else None

            # Side-aware share sizing fallback: only derive shares if needed
            position_size_usd = (
                opportunity.get('pm_position_size_usd')
                or opportunity.get('pm_position_size')
                or opportunity.get('position_size_usd')
            )
            if (pm_shares <= 0.0) and (position_size_usd is not None):
                try:
                    side_price = (yes_price if pm_side == 'BUY_YES' else no_price)
                    if side_price is None:
                        # Accept explicit side price from upstream if provided; still refuse 1-p complement inference
                        side_price = opportunity.get('pm_price', None)
                    if side_price is None:
                        self.logger.warning("Missing side price to derive pm_shares; cannot size position.")
                        return 0.0
                    pm_shares = min(
                        MAX_PM_SHARES,
                        float(position_size_usd) / max(1e-12, float(side_price))
                    )
                except Exception as _e:
                    self.logger.warning(f"Failed to derive pm_shares from position_size_usd: {_e}")
                    pm_shares = 0.0
            if pm_shares <= 0.0:
                return 0.0

            # Binary payoff with epsilon-consistent boundary handling
            is_above = bool(contract.get('is_above', True))
            if is_above:
                # Event occurs if price is strictly above strike (with epsilon tolerance)
                event_occurs = (final_price > K * (1.0 + eps))
            else:
                # Event occurs if price is strictly below strike (with epsilon tolerance)
                event_occurs = (final_price < K * (1.0 - eps))
            
            # PM leg P&L is **payoff only**; entry cash out is accounted for in `costs.total`.
            # BUY_YES pays $1 per share if the event occurs; BUY_NO pays $1 per share if the event does not occur.
            if pm_side == 'BUY_YES':
                pm_pnl = pm_shares * (1.0 if event_occurs else 0.0)
            else:  # BUY_NO
                pm_pnl = pm_shares * (1.0 if (not event_occurs) else 0.0)
            
            # Optional fee on profits for PM leg
            win_fee_rate = float(opportunity.get('win_fee_rate', 0.0))
            if win_fee_rate and pm_pnl > 0.0:
                pm_pnl *= (1.0 - win_fee_rate)

            # Hedge P&L
            if strategy_type == 'perps':
                explicit_notional = opportunity.get('hedge_notional_usd')
                if explicit_notional is not None and float(explicit_notional) > 0.0:
                    hedge_size_usd = min(MAX_HEDGE_NOTIONAL_USD, float(explicit_notional))
                else:
                    hedge_size_usd = min(MAX_HEDGE_NOTIONAL_USD, pm_shares * 1.0)  # natural unit for $1-capped PM payout
                
                price_change_pct = (final_price - S0) / S0
                hedge_pnl = hedge_size_usd * (price_change_pct if hedge_direction == 'LONG' else -price_change_pct)
                
                funding_pnl = float(opportunity.get('funding_income', 0.0)) - float(opportunity.get('funding_cost', 0.0))
                hedge_pnl += funding_pnl
            elif strategy_type == 'options':
                # --- Conservative cross-expiry hedge valuation (intrinsic only) ---
                hedge_pnl = 0.0
                legs = opportunity.get('required_options') or []
                # Optional: enforce vertical credit/debit bound in detector when width is inferable
                K, width = _extract_spread_width_and_strike(opportunity)
                if width and len(legs) >= 2:
                    # infer credit per unit if both legs are same qty and directions oppose
                    qs = { (l.get('type'), float(l.get('strike')), l.get('action')) for l in legs }
                    qtys = { float(l.get('contracts', 0.0)) for l in legs }
                    if len(qtys) == 1:
                        # rely on saved costs.total to test bound approximately
                        credit = -float((opportunity.get('costs', {}) or {}).get('options_entry', 0.0))
                        credit_per_unit = credit / max(1e-12, float(list(qtys)[0]))
                        if credit_per_unit < -1e-9 or credit_per_unit > width + 1e-9:
                            return 0.0  # suspect totals → not a true arb
                # In strict-arb mode we must be model-free at PM resolution; intrinsic only.
                if SETTLE_SPREAD_AT_PM_EXPIRY and str(OPTIONS_UNWIND_MODEL) == "intrinsic_only":
                    for leg in legs:
                        k = float(leg.get('strike', 0.0))
                        qty = float(leg.get('contracts', 0.0))
                        action = (leg.get('action') or 'BUY').upper()
                        typ = (leg.get('type') or 'CALL').upper()
                        intrinsic = max(0.0, final_price - k) if typ == 'CALL' else max(0.0, k - final_price)
                        leg_pnl = intrinsic * qty
                        hedge_pnl += leg_pnl if action == 'BUY' else -leg_pnl
            else:
                hedge_pnl = 0.0
            
            # total entry cashflows (USD numeraire)
            costs = float((opportunity.get('costs', {}) or {}).get('total', 0.0))
            # Strict USD‑numeraire arbitrage semantics (A1,A2,D9):
            #   P&L(S) = PM_payoff(S) + Σ leg_intrinsic(S) − costs.total
            return pm_pnl + hedge_pnl - costs
            
        except Exception as e:
            self.logger.error(f"_simulate_pnl_at_price failed: {e} | opp_id={opportunity.get('id', 'unknown')}")
            return 0.0

    def _rank_opportunities(self, candidates: List[Dict]) -> List[Dict]:
        """
        Rank opportunities with true arbitrage strictly prioritized.
        """
        ranked = []
        for opp in candidates:
            # Expose any existing fields, set defaults before ranking
            opp['max_profit'] = opp.get('max_profit', 0.0)
            opp['max_loss'] = abs(opp.get('max_loss', 0.0))
            
            # detector-driven label
            if LABEL_TRUE_ARB_FROM_DETECTOR:
                opp['is_true_arbitrage'] = bool(opp.get('is_true_arbitrage', False))
            else:
                opp['is_true_arbitrage'] = (opp.get('max_loss', 0.0) <= 100.0)  # Default max acceptable loss
            
            opp['profit_probability'] = opp.get('profit_probability', 0.0)
            opp['quality_score'] = opp.get('quality_score', 0.0)
            
            ranked.append(opp)
        
        # Sort by detector first, then quality; strict mode keeps pure arbs on top regardless of EV
        ranked.sort(key=lambda o: (
            not o.get('is_true_arbitrage', False),
            -(o.get('quality_score', 0)),
            -(o.get('profit_probability', 0.0)),
            -(o.get('max_profit', 0.0)),
        ))
        return ranked
    
    def _classify_true_arbitrage(self, opportunity: Dict, max_profit: float, max_loss: float) -> Dict:
        """
        Classify types of true arbitrage.
        """
        
        # Determine arbitrage quality
        if max_loss < 2:
            arbitrage_type = "PURE_ARBITRAGE"
            quality_score = 100
        elif max_loss < 5:
            arbitrage_type = "HIGH_QUALITY_ARBITRAGE"  
            quality_score = 90
        elif max_loss < 10:
            arbitrage_type = "GOOD_ARBITRAGE"
            quality_score = 80
        else:
            arbitrage_type = "MARGINAL_ARBITRAGE"
            quality_score = 70
        
        return {
            'is_true_arbitrage': True,
            'arbitrage_type': arbitrage_type,
            'profit_probability': 1.0,  # By definition
            'quality_score': quality_score,
            'classification_reason': f"True arbitrage: Max profit ${max_profit:.2f}, Max loss ${max_loss:.2f}",
            'trading_priority': 1  # Highest priority
        }
    
    def _classify_directional_strategy(self, opportunity: Dict, max_profit: float, max_loss: float) -> Dict:
        """
        Classify directional strategies by probability of profit.
        """
        
        # Calculate probability based on required price movement
        contract = opportunity.get('polymarket_contract', {})
        pricing = opportunity.get('pricing_details', {})
        
        current_spot = pricing.get('oracle_price', 0)
        strike_price = contract.get('strike_price', current_spot)
        
        if current_spot > 0 and strike_price > 0:
            price_distance = abs(strike_price - current_spot) / current_spot
            pm_side = opportunity.get('pm_side', 'BUY_YES')
            t_days = contract.get('days_to_event') or contract.get('time_to_event_days')
            if t_days and float(t_days) > 0.0:
                T = max(1e-6, float(t_days) / 365.0)
                sigma = max(1e-6, self.default_vol)
                is_above_contract = bool(contract.get('is_above', True))
                
                # Probability that the contract's condition is true under GBM
                p_event = _prob_event_black_scholes(current_spot, strike_price, T, sigma, self.risk_free_rate, is_above_contract)
                
                # If you're effectively "betting up", profit prob is p_event; otherwise it's 1 - p_event
                betting_up = ((pm_side == 'BUY_YES' and is_above_contract) or
                              (pm_side == 'BUY_NO' and not is_above_contract))
                profit_probability = float(p_event if betting_up else (1.0 - p_event))
                
                # Buckets as you have them
                if profit_probability >= 0.70:
                    strategy_type = "HIGH_PROBABILITY_DIRECTIONAL"
                elif profit_probability >= 0.50:
                    strategy_type = "MEDIUM_PROBABILITY_DIRECTIONAL"
                elif profit_probability >= 0.30:
                    strategy_type = "LOW_PROBABILITY_DIRECTIONAL"
                else:
                    strategy_type = "SPECULATIVE_DIRECTIONAL"
            else:
                # Keep your distance-only heuristic as a last-resort fallback
                if price_distance < 0.02:
                    profit_probability = 0.80
                    strategy_type = "HIGH_PROBABILITY_DIRECTIONAL"
                elif price_distance < 0.05:
                    profit_probability = 0.65
                    strategy_type = "MEDIUM_PROBABILITY_DIRECTIONAL"
                elif price_distance < 0.10:
                    profit_probability = 0.45
                    strategy_type = "LOW_PROBABILITY_DIRECTIONAL"
                else:
                    profit_probability = 0.25
                    strategy_type = "SPECULATIVE_DIRECTIONAL"
        else:
            price_distance = 0.0
            profit_probability = 0.30
            strategy_type = "SPECULATIVE_DIRECTIONAL"
        
        # Adjust for risk/reward ratio
        rr = (max_profit / max(max_loss, 1.0))
        if rr > 3:
            quality_score = 60
        elif rr > 2:
            quality_score = 50
        elif rr > 1:
            quality_score = 40
        else:
            quality_score = 20
        
        # Create classification reason
        if price_distance > 0:
            classification_reason = f"Directional strategy: {price_distance:.1%} movement required, P(profit)={profit_probability:.0%}"
        else:
            classification_reason = f"Directional strategy: Unknown price movement, P(profit)={profit_probability:.0%}"
        
        return {
            'is_true_arbitrage': False,
            'arbitrage_type': strategy_type,
            'profit_probability': profit_probability,
            'quality_score': quality_score,
            'price_movement_required': price_distance,
            'classification_reason': classification_reason,
            'trading_priority': 2 if profit_probability > 0.6 else 3  # Lower priority than true arbitrage
        }
    
    def _promote_strategy_metrics(self, opportunity: Dict) -> None:
        """
        Promote strategy-supplied nested metrics to top-level keys for dashboards/filters.
        Safe no-ops if missing. Mutates `opportunity` in-place.
        """
        # Cost-recovery metrics
        cr = opportunity.get('cost_recovery')
        if isinstance(cr, dict):
            opportunity['cr_is_adequate']        = bool(cr.get('is_adequate'))
            opportunity['cr_net_recovery']       = float(cr.get('net_recovery', 0.0))
            opportunity['cr_recovery_ratio']     = float(cr.get('recovery_ratio', 0.0))
            opportunity['cr_entry_cost']         = float(cr.get('entry_cost', 0.0))
            opportunity['cr_exit_spread_loss']   = float(cr.get('exit_spread_loss', 0.0))
            opportunity['cr_value_at_pm_strike'] = float(cr.get('hedge_value_at_strike', 0.0))

        # True-arb boundary metrics (portfolio-level at S=K)
        tab = opportunity.get('true_arb_boundary')
        if isinstance(tab, dict):
            opportunity['tab_is_true_arb_boundary']               = bool(tab.get('is_true_arb_boundary'))
            opportunity['tab_required_total_cost_at_boundary']    = float(tab.get('required_total_cost_at_boundary', 0.0))
            opportunity['tab_hedge_value_at_boundary']            = float(tab.get('hedge_value_at_boundary', 0.0))
            # pm_entry_cost may be missing/non-numeric; guard conversion
            try:
                opportunity['tab_pm_entry_cost'] = float(tab.get('pm_entry_cost', 0.0))
            except Exception:
                pass
            # Optional extras
            if 'evaluated_at_spot' in tab:
                try:
                    opportunity['tab_evaluated_at_spot'] = float(tab.get('evaluated_at_spot'))
                except Exception:
                    pass
            if 'pm_market_side' in tab and tab.get('pm_market_side') is not None:
                opportunity['tab_pm_market_side'] = str(tab.get('pm_market_side'))

    def rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Rank opportunities prioritizing true arbitrage, then probability of profit.
        """
        if not opportunities:
            return []
        
        # Classify all opportunities first
        classified = []
        for opp in opportunities:
            classified_opp = self.classify_opportunity(opp.copy())
            classified.append(classified_opp)
        
        # Sort by: 1) True arbitrage first, 2) Quality score, 3) Profit probability, 4) Max profit
        ranked = sorted(
            classified,
            key=lambda o: (
                not o.get('is_true_arbitrage', False),       # True arbitrage first
                -(o.get('quality_score', 0)),                # higher quality first
                -(o.get('profit_probability', 0.0)),         # higher prob first
                -(o.get('max_profit', 0.0)),                 # larger upside
            )
        )
        
        # Log ranking summary
        true_arb_count = sum(1 for opp in ranked if opp.get('is_true_arbitrage', False))
        directional_count = len(ranked) - true_arb_count
        
        self.logger.info(f"📊 RANKING SUMMARY:")
        self.logger.info(f"   🎯 True Arbitrage: {true_arb_count}")
        self.logger.info(f"   📈 Directional: {directional_count}")
        self.logger.info(f"   📊 Total: {len(ranked)}")
        
        return ranked