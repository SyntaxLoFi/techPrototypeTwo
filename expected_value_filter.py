"""
Expected Value Filter - Academic-based filtering for arbitrage opportunities
==========================================================================

This module implements filtering based on Expected Value (EV) and risk-adjusted
metrics rather than arbitrary profit thresholds. Based on academic literature:

- Wolfers & Zitzewitz (2004) - Prediction market probabilities
  https://www.aeaweb.org/articles?id=10.1257%2F0895330041371321
  
- Kelly (1956) - Optimal betting fractions
  https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf
  
- Sharpe (1994) - Risk-adjusted returns
  https://web.stanford.edu/~wfsharpe/art/sr/SR.htm
  
- MacLean, Thorp & Ziemba (2011) - Fractional Kelly
  https://www.worldscientific.com/worldscibooks/10.1142/7598

Author: Prediction Market Arbitrage System
Date: 2025-01-10
"""

import numpy as np
import os
from typing import Dict, Tuple, Optional, List, Any
import logging
from collections import Counter
from utils.log_gate import reason_debug, ev_summary_info_enabled  # type: ignore
from utils.step_debugger import get_step_debugger  # type: ignore

# Try to import from config with fallback defaults
try:
    from config_manager import (
        TRANSACTION_COST_RATE,
        KELLY_FRACTION_CAP,
        RISK_FREE_RATE,
        DAYS_PER_YEAR,
        PROBABILITY_CLIP_FLOOR,
        PROBABILITY_CLIP_CEILING,
    )
    # KELLY_FRACTION is not in config_manager, use default
    KELLY_FRACTION = 1.0  # 100% Kelly by default (no fractional scaling)
except ImportError:
    TRANSACTION_COST_RATE = 0.002  # 0.2% default
    KELLY_FRACTION_CAP = 0.25      # 25% maximum allocation default
    KELLY_FRACTION = 1.0           # 100% Kelly by default (no fractional scaling)
    RISK_FREE_RATE = 0.05          # 5% annual default
    DAYS_PER_YEAR = 365.25         # Calendar days per year
    PROBABILITY_CLIP_FLOOR = 0.001 # 0.1% minimum
    PROBABILITY_CLIP_CEILING = 0.999 # 99.9% maximum

logger = logging.getLogger(__name__)


class ExpectedValueFilter:
    """
    Filters arbitrage opportunities based on mathematical expectancy
    rather than arbitrary profit thresholds.
    """
    
    def __init__(self, 
                 min_ev: float = 0.10,
                 min_sharpe: float = 0.5,
                 min_kelly: float = 0.01,
                 max_downside: float = 100.0,
                 max_downside_pct: float = 0.10,  # 10% of position size
                 downside_scaling_mode: str = 'adaptive',  # 'fixed', 'percentage', or 'adaptive'
                 probability_blend_weight: float = 0.5,
                 probability_clip_floor: float = PROBABILITY_CLIP_FLOOR,
                 probability_clip_ceiling: float = PROBABILITY_CLIP_CEILING,
                 capital_at_risk: Optional[float] = None,
                 kelly_fraction: float = KELLY_FRACTION):
        """
        Initialize filter with configurable thresholds.
        """
        # Allow environment overrides without changing call sites
        try:
            if os.getenv("MIN_EV"):        min_ev = float(os.getenv("MIN_EV"))       # type: ignore
            if os.getenv("MIN_SHARPE"):    min_sharpe = float(os.getenv("MIN_SHARPE"))  # type: ignore
            if os.getenv("MIN_KELLY"):     min_kelly = float(os.getenv("MIN_KELLY"))  # type: ignore
        except Exception:
            pass
        self.min_ev = float(min_ev)
        self.min_sharpe = float(min_sharpe)
        self.min_kelly = float(min_kelly)
        self.max_downside = float(max_downside)
        self.max_downside_pct = float(max_downside_pct)
        self.downside_scaling_mode = downside_scaling_mode
        self.probability_blend_weight = float(probability_blend_weight)
        self.probability_clip_floor = float(probability_clip_floor)
        self.probability_clip_ceiling = float(probability_clip_ceiling)
        self.capital_at_risk = float(capital_at_risk) if capital_at_risk is not None else None
        self.transaction_cost = float(TRANSACTION_COST_RATE)
        # NEW: true fractional Kelly multiplier η in [0,1]
        self.kelly_fraction = float(max(0.0, min(kelly_fraction, 1.0)))
    
    def _effective_cost_rate(self, opp):
        """
        Global transaction cost rate plus any per-opportunity liquidity penalty
        supplied by the strategy (in basis points).
        """
        base = float(self.transaction_cost)
        bps  = float(opp.get("liquidity_penalty_bps", 0.0) or 0.0)
        return base + (bps / 10000.0)
    
    def _calculate_downside_threshold(self, position_size: Optional[float], 
                                     expected_value: float, 
                                     prob_of_loss: float) -> float:
        """
        Calculate intelligent downside threshold based on position size and risk metrics.
        
        Modes:
        - 'fixed': Use fixed dollar amount (original behavior)
        - 'percentage': Use percentage of position size
        - 'adaptive': Blend fixed and percentage, with adjustments for EV and probability
        
        Args:
            position_size: Size of the position in dollars
            expected_value: Expected value of the opportunity
            prob_of_loss: Probability of experiencing the max loss
            
        Returns:
            Maximum acceptable loss (positive number)
        """
        if self.downside_scaling_mode == 'fixed':
            return self.max_downside
            
        if position_size is None or position_size <= 0:
            # Fallback to fixed if no position size
            return self.max_downside
            
        if self.downside_scaling_mode == 'percentage':
            # Pure percentage-based threshold
            return position_size * self.max_downside_pct
            
        # Adaptive mode: intelligent scaling
        # Base threshold is the larger of fixed amount or percentage
        base_threshold = max(self.max_downside, position_size * self.max_downside_pct)
        
        # Adjust based on expected value
        # High positive EV justifies accepting larger downside
        if expected_value > 0:
            ev_multiplier = 1.0 + min(expected_value / position_size, 0.5)  # Cap at 50% boost
        else:
            ev_multiplier = 1.0
            
        # Adjust based on probability of loss
        # Low probability of loss allows larger acceptable downside
        if prob_of_loss < 0.1:  # Less than 10% chance
            prob_multiplier = 1.5
        elif prob_of_loss < 0.2:  # Less than 20% chance
            prob_multiplier = 1.25
        else:
            prob_multiplier = 1.0
            
        # For small positions, ensure minimum threshold
        min_threshold = min(self.max_downside, position_size * 0.5)  # At least 50% of small positions
        
        return max(min_threshold, base_threshold * ev_multiplier * prob_multiplier)
    
    def _blend_probabilities(self, pm_yes_price: float, options_implied_prob: Optional[float] = None) -> float:
        """Blend and clip probabilities from PM and options (if provided)."""
        # Coerce to floats safely
        try:
            pm = float(pm_yes_price)
        except Exception:
            pm = 0.5  # neutral fallback
        
        opt = None
        if options_implied_prob is not None:
            try:
                opt = float(options_implied_prob)
            except Exception:
                opt = None
        
        if opt is not None and 0.0 <= opt <= 1.0:
            prob_yes = self.probability_blend_weight * pm + (1.0 - self.probability_blend_weight) * opt
            logger.debug(f"Using blended probability: PM={pm:.3f}, Options={opt:.3f}, Blend={prob_yes:.3f}")
        else:
            prob_yes = pm
        
        # Ensure probability is in valid range
        return float(np.clip(prob_yes, self.probability_clip_floor, self.probability_clip_ceiling))
        
    def calculate_expected_value(self, 
                               payoff_if_yes: float, 
                               payoff_if_no: float, 
                               pm_yes_price: float,
                               options_implied_prob: Optional[float] = None) -> float:
        """
        Calculate expected value for binary outcome strategy.
        
        Based on: EV = P(event) × Payoff_if_event + (1 - P(event)) × Payoff_if_not_event
        
        Academic foundation:
        - Wolfers & Zitzewitz (2004): "Prediction Markets"
          Shows prediction market prices are efficient probability estimates
        
        Args:
            payoff_if_yes: Profit/loss if YES wins
            payoff_if_no: Profit/loss if NO wins  
            pm_yes_price: Prediction market YES price (implied probability)
            options_implied_prob: Options-implied probability (if available)
            
        Returns:
            Expected value in dollars
        """
        # Use centralized blending logic
        prob_yes = self._blend_probabilities(pm_yes_price, options_implied_prob)
        
        # Normalize payoffs if capital_at_risk is provided
        if self.capital_at_risk is not None and self.capital_at_risk > 0:
            normalized_yes = payoff_if_yes / self.capital_at_risk
            normalized_no = payoff_if_no / self.capital_at_risk
            ev = prob_yes * normalized_yes + (1 - prob_yes) * normalized_no
            # Return EV in dollar terms
            return ev * self.capital_at_risk
        else:
            ev = prob_yes * payoff_if_yes + (1 - prob_yes) * payoff_if_no
            return ev
    
    def calculate_binary_sharpe_ratio(self,
                                    payoff_if_yes: float,
                                    payoff_if_no: float,
                                    prob_yes: float,
                                    risk_free_rate: Optional[float] = None,
                                    holding_period_days: float = 1.0,
                                    position_cost: Optional[float] = None) -> Optional[float]:
        """Sharpe ratio for a binary outcome, computed in *return* space.

        Var[R] = p(1-p) * (r_yes - r_no)^2

        Returns inf when variance is (near) zero and expected return exceeds the risk-free return;
        returns 0.0 when variance is (near) zero and expected return is <= risk-free.
        Returns None if a capital base cannot be inferred.
        """
        # Determine capital base
        capital_base = None
        if self.capital_at_risk is not None and self.capital_at_risk > 0:
            capital_base = self.capital_at_risk
        elif position_cost is not None and position_cost > 0:
            capital_base = float(position_cost)
        else:
            return None  # cannot compute Sharpe without a base
        
        # Convert payoffs to returns
        r_yes = float(payoff_if_yes) / capital_base
        r_no = float(payoff_if_no) / capital_base
        
        # Period risk-free
        if risk_free_rate is None:
            risk_free_rate = RISK_FREE_RATE
        period_rf = float(risk_free_rate) * (float(holding_period_days) / DAYS_PER_YEAR)
        
        pr = float(prob_yes)
        exp_r = pr * r_yes + (1.0 - pr) * r_no
        delta = r_yes - r_no
        var = pr * (1.0 - pr) * (delta ** 2)  # <-- no epsilon
        
        # Degenerate variance → inf or 0
        if var <= 0.0 or abs(delta) < 1e-15 or pr in (0.0, 1.0):
            return float('inf') if exp_r > period_rf else 0.0
        
        std = (var) ** 0.5
        return (exp_r - period_rf) / std
    
    def calculate_kelly_fraction(self,
                               payoff_if_yes: float,
                               payoff_if_no: float,
                               prob_yes: float) -> float:
        """Kelly-optimal fraction for a binary bet with fractional scaling and capping."""
        # Normalize to per-dollar returns if capital_at_risk is known
        if self.capital_at_risk is not None and self.capital_at_risk > 0:
            y = float(payoff_if_yes) / self.capital_at_risk
            n = float(payoff_if_no) / self.capital_at_risk
        else:
            y = float(payoff_if_yes)
            n = float(payoff_if_no)

        # True arbitrage: both outcomes profitable → max allowed (after cap)
        if y > 0 and n > 0:
            raw = 1.0
        elif y > 0 > n:
            b = y / (abs(n) + 1e-15)
            raw = (float(prob_yes) * b - (1.0 - float(prob_yes))) / b
        elif n > 0 > y:
            b = n / (abs(y) + 1e-15)
            raw = ((1.0 - float(prob_yes)) * b - float(prob_yes)) / b
        else:
            return 0.0

        f = max(0.0, raw * self.kelly_fraction)
        return min(f, KELLY_FRACTION_CAP)
    
    def should_include_opportunity(self, opportunity: Dict) -> Tuple[bool, Dict]:
        """Decide inclusion based on EV and risk metrics with sensible gating."""
        # Payoffs - try standard fields first, then legacy
        payoff_if_yes = opportunity.get('profit_if_yes', opportunity.get('payoff_if_yes'))
        payoff_if_no = opportunity.get('profit_if_no', opportunity.get('payoff_if_no'))
        
        # Track missing data for debugging
        missing_fields = []
        if payoff_if_yes is None:
            missing_fields.append("profit_if_yes")
            # Last resort fallbacks only
            payoff_if_yes = opportunity.get('pnl_if_above', opportunity.get('max_profit', 0.0))
            
        if payoff_if_no is None:
            missing_fields.append("profit_if_no")
            payoff_if_no = opportunity.get('pnl_if_below', opportunity.get('max_loss', 0.0))

        # Probabilities
        pm_data = opportunity.get('polymarket_contract', opportunity.get('polymarket', {}))
        pm_yes_price = pm_data.get('yes_price', opportunity.get('pm_yes_price', 0.5))
        options_prob = opportunity.get('probabilities', {}).get('options_implied') if 'probabilities' in opportunity else None

        ev = self.calculate_expected_value(payoff_if_yes, payoff_if_no, pm_yes_price, options_prob)
        prob_yes = self._blend_probabilities(pm_yes_price, options_prob)

        # Holding period & capital base
        holding_period_days = max(float(opportunity.get('days_to_expiry', 1.0)), 1e-6)
        position_cost = None
        for key in ('stake', 'position_cost', 'portfolio_cost', 'notional', 'position_size'):
            if key in opportunity:
                position_cost = opportunity[key]
                break

        sharpe = self.calculate_binary_sharpe_ratio(
            payoff_if_yes, payoff_if_no, prob_yes,
            holding_period_days=holding_period_days,
            position_cost=position_cost
        )
        kelly = self.calculate_kelly_fraction(payoff_if_yes, payoff_if_no, prob_yes)

        # Transaction costs
        # Use actual traded size, not P&L values
        trading_notional = position_cost
        if trading_notional is None:
            # Try proper size fields in priority order
            for key in ('required_capital', 'position_size', 'stake', 'portfolio_cost', 'notional', 'position_cost'):
                if key in opportunity and opportunity[key] is not None:
                    trading_notional = float(opportunity[key])
                    break
        
        # Conservative fallback: use upfront_cashflow if it's a real outlay
        if trading_notional is None and 'upfront_cashflow' in opportunity and opportunity['upfront_cashflow'] < 0:
            trading_notional = abs(float(opportunity['upfront_cashflow']))

        if trading_notional and trading_notional > 0:
            tx_costs = self._effective_cost_rate(opportunity) * float(trading_notional) * 2.0
        else:
            # Don't use P&L as fallback - it massively overestimates costs
            logger.info("Cannot calculate transaction costs - no position size found for opportunity")
            tx_costs = 0.0

        ev_after_costs = ev - tx_costs

        # Calculate position size for downside scaling
        position_size = position_cost or trading_notional
        if position_size is None:
            # Try to infer from required capital or other fields
            for key in ('required_capital', 'position_size', 'stake', 'notional'):
                if key in opportunity and opportunity[key] is not None:
                    position_size = float(opportunity[key])
                    break
        
        # Calculate probability of loss (worst case scenario)
        worst_payoff = min(payoff_if_yes, payoff_if_no)
        if worst_payoff < 0:
            # Probability of hitting the worst case
            prob_of_loss = prob_yes if payoff_if_no < payoff_if_yes else (1 - prob_yes)
        else:
            prob_of_loss = 0.0  # No loss possible
        
        # Calculate intelligent downside threshold
        downside_threshold = self._calculate_downside_threshold(position_size, ev, prob_of_loss)
        
        # Gating
        true_arb = payoff_if_yes > 0 and payoff_if_no > 0
        good_downside = worst_payoff >= -downside_threshold

        if true_arb:
            min_profit = min(payoff_if_yes, payoff_if_no)
            should_include = (min_profit - tx_costs) >= max(self.min_ev, 0.0)
            inclusion_reason = ("True arbitrage: min_profit covers round-trip costs"
                                if should_include else "True arbitrage but profit < costs threshold")
        else:
            positive_ev = ev_after_costs >= self.min_ev
            good_sharpe = (sharpe is not None) and (sharpe >= self.min_sharpe)
            good_kelly = kelly >= self.min_kelly

            should_include = (positive_ev and good_downside) or ((good_sharpe or good_kelly) and good_downside)

            if should_include:
                if positive_ev:
                    inclusion_reason = f"Positive EV after costs: ${ev_after_costs:.2f}"
                elif good_sharpe:
                    inclusion_reason = f"Good risk-adjusted return: Sharpe={sharpe:.2f}"
                else:
                    inclusion_reason = f"Kelly suggests position: {kelly:.1%}"
            else:
                if worst_payoff < -downside_threshold:
                    inclusion_reason = (f"Downside too large: ${worst_payoff:.2f} "
                                      f"(threshold: ${-downside_threshold:.2f})")
                else:
                    inclusion_reason = (f"Failed criteria: EV=${ev_after_costs:.2f}, "
                                        f"Sharpe={'N/A' if sharpe is None else f'{sharpe:.2f}'}, "
                                        f"Kelly={kelly:.1%}")

        metrics = {
            'expected_value': ev,
            'ev_after_costs': ev_after_costs,
            'sharpe_ratio': sharpe,
            'kelly_fraction': kelly,
            'probability_used': prob_yes,
            'inclusion_reason': inclusion_reason,
            'worst_case': worst_payoff,
            'best_case': max(payoff_if_yes, payoff_if_no),
            'transaction_costs': tx_costs,
            'downside_threshold': downside_threshold,
            'position_size_used': position_size,
            'prob_of_loss': prob_of_loss,
        }
        
        # Add missing fields info if relevant
        if missing_fields:
            metrics['missing_fields'] = missing_fields

        # Per-dollar metrics if base known
        base = self.capital_at_risk if (self.capital_at_risk and self.capital_at_risk > 0) else (position_cost if (position_cost and position_cost > 0) else None)
        if base:
            metrics['expected_return_per_dollar'] = ev / base
            metrics['return_after_costs'] = ev_after_costs / base

        # Determine compact filter code for logging/aggregation
        try:
            if should_include:
                filter_code = "PASS_TRUE_ARB" if true_arb else "PASS"
            else:
                if true_arb:
                    filter_code = "DROP_TRUE_ARB_LOW_PROFIT"
                else:
                    if not good_downside:
                        filter_code = "DROP_DOWNSIDE_TOO_LARGE"
                    elif not positive_ev:
                        filter_code = "DROP_EV_BELOW_MIN"
                    elif not (good_sharpe or good_kelly):
                        if (sharpe is not None) and (sharpe < self.min_sharpe) and (kelly < self.min_kelly):
                            filter_code = "DROP_SHARPE_AND_KELLY"
                        elif (sharpe is not None) and (sharpe < self.min_sharpe):
                            filter_code = "DROP_SHARPE_BELOW_MIN"
                        else:
                            filter_code = "DROP_KELLY_BELOW_MIN"
                    else:
                        filter_code = "DROP_OTHER"
        except Exception:
            filter_code = "UNKNOWN"
        metrics['filter_code'] = filter_code

        # One-line DEBUG trace for each decision
        try:
            ccy = str(opportunity.get('currency')
                      or (opportunity.get('polymarket_contract') or {}).get('asset')
                      or (opportunity.get('polymarket') or {}).get('asset')
                      or "-").upper()
            side = str(opportunity.get('pm_side') or "-")
            try:
                K = float((opportunity.get('polymarket_contract') or {}).get('strike_price')
                          or (opportunity.get('polymarket') or {}).get('strike') or 0.0)
            except Exception:
                K = 0.0
            ctx = f"{ccy}:{side}:K={K:.0f}" if K else f"{ccy}:{side}"
            reason_debug(
                logger,
                "EV_FILTER %s %s code=%s ev_after_costs=%.6f costs=%.6f worst=%.4f thr=%.4f sharpe=%s kelly=%.4f reason=%s",
                "PASS" if should_include else "DROP", ctx, filter_code, ev_after_costs, tx_costs, worst_payoff, downside_threshold,
                "na" if sharpe is None else f"{sharpe:.2f}", kelly, inclusion_reason
            )
        except Exception:
            pass

        return should_include, metrics
    
    def filter_opportunities(self, opportunities: list, return_all: bool = False):
        """
        Annotate every opportunity with EV/Sharpe/Kelly and (optionally) return all.
        When return_all=True, returns (filtered, summary) where summary['annotated_all']
        contains *all* opps with a 'metrics' dict attached, and counts per-criterion.
        """
        annotated = []
        filtered = []
        counts = {"total": 0, "passed": 0, "failed": 0}
        reason_counts = Counter()
        debugger = get_step_debugger()
        
        # Checkpoint: Input to EV filter
        debugger.checkpoint("ev_filter_input", opportunities,
                          {"count": len(opportunities),
                           "has_profit_fields": sum(1 for o in opportunities if "profit_if_yes" in o and "profit_if_no" in o),
                           "has_metrics": sum(1 for o in opportunities if "metrics" in o)})

        for opp in opportunities or []:
            counts["total"] += 1
            # --- probabilities ---
            probs = (opp.get("probabilities") or {})
            pm_yes = probs.get("pm_yes") or (opp.get("polymarket") or {}).get("yes_price")
            pm_no  = probs.get("pm_no")  or (opp.get("polymarket") or {}).get("no_price")
            if pm_yes is not None and pm_no is None:
                pm_no = 1.0 - float(pm_yes)
            
            # Use the existing should_include_opportunity method
            include, metrics = self.should_include_opportunity(opp)
            
            # Extract calculated values from metrics
            prob_yes = metrics.get('probability_of_profit', 0.5)
            ev = metrics.get('expected_value', 0.0)
            sharpe = metrics.get('sharpe_ratio', 0.0)
            kelly = metrics.get('kelly_fraction', 0.0)
            
            # persist metrics on object (under a neutral key the rest of the code already reads)
            opp.setdefault("metrics", {}).update({
                "expected_value": metrics.get("ev_after_costs", metrics.get("expected_value")),
                "adjusted_ev": metrics.get("adjusted_ev", metrics.get("ev_after_costs")),
                "prob_of_profit": metrics.get("prob_of_profit", prob_yes),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "kelly_fraction": metrics.get("kelly_fraction"),
                "edge_per_downside": metrics.get("edge_per_downside"),
                "dni": metrics.get("dni"),
                "is_true_arbitrage": metrics.get("is_true_arbitrage", False),
                "filter_code": metrics.get("filter_code", "UNKNOWN"),
            })
            annotated.append(opp)
            if include:
                filtered.append(opp)
                counts["passed"] += 1
            else:
                filter_code = opp.get("metrics", {}).get('filter_code', 'DROP_OTHER')
                reason_counts[filter_code] += 1
                counts["failed"] += 1
                # Track drops in debugger with more detail
                debugger.track_drop(filter_code, opp)

        # Compact INFO summary with top drop reasons (to avoid log bloat)
        drop_counts = dict(reason_counts)
        total_failed = counts.get("failed", 0)
        top_drops = reason_counts.most_common(5)
        top_drops_str = ", ".join(f"{k}:{v}" for k, v in top_drops)
        
        try:
            if ev_summary_info_enabled() and counts.get("total", 0) > 0:
                logger.info(
                    "EV filter: %d passed / %d failed%s",
                    counts.get("passed", 0), counts.get("failed", 0),
                    f" | top drops: {top_drops_str}" if top_drops_str else ""
                )
        except Exception:
            pass
        
        # Checkpoint: Output from EV filter with detailed drop reasons
        debugger.checkpoint("ev_filter_output", filtered,
                          {"passed": len(filtered),
                           "failed": total_failed,
                           "drop_reasons": drop_counts,
                           "top_drops": top_drops_str})

        if return_all:
            return filtered, {"annotated_all": annotated, "counts": counts}
        return filtered


def create_normalized_filter(position_size: float, **kwargs) -> ExpectedValueFilter:
    """
    Create an ExpectedValueFilter that normalizes payoffs by position size.
    
    Args:
        position_size: The capital at risk for the position
        **kwargs: Additional parameters to pass to ExpectedValueFilter
        
    Returns:
        ExpectedValueFilter instance configured for normalized payoffs
    """
    return ExpectedValueFilter(capital_at_risk=position_size, **kwargs)


# Create default instance for easy import
default_filter = ExpectedValueFilter()