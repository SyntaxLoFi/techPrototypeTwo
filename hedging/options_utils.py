"""
Options-specific utility functions for hedge construction.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

from utils.log_gate import reason_debug  # type: ignore


@dataclass
class Quote:
    """Represents a bid/ask/mid quote for an option."""
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]


def transform_options_to_chain(options: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Transform flat options list to nested structure required by digital_hedge_builder.
    Returns: {expiry: {strike: {'call': {...}, 'put': {...}}}}
    """
    chains_by_expiry = {}
    
    for opt in options or []:
        # Get expiry date
        expiry = opt.get('expiry_date') or opt.get('expiry') or opt.get('expiration')
        if not expiry:
            continue
            
        # Get strike as string float
        try:
            strike = str(float(opt.get('strike', 0)))
        except (ValueError, TypeError):
            continue
            
        # Normalize option type
        opt_type_raw = str(opt.get('type', '')).upper()
        if opt_type_raw in ('C', 'CALL'):
            opt_type = 'call'
        elif opt_type_raw in ('P', 'PUT'):
            opt_type = 'put'
        else:
            continue
            
        # Build nested structure
        if expiry not in chains_by_expiry:
            chains_by_expiry[expiry] = {}
        if strike not in chains_by_expiry[expiry]:
            chains_by_expiry[expiry][strike] = {}
        
        chains_by_expiry[expiry][strike][opt_type] = opt
        
    return chains_by_expiry


def select_best_expiry(chains_by_expiry: Dict[str, Any], pm_days_to_expiry: float) -> Optional[str]:
    """
    Select the most appropriate expiry for digital hedge construction.
    Currently selects the nearest expiry that is on or after PM expiry.
    """
    if not chains_by_expiry:
        return None
        
    valid_expiries = []
    
    for expiry, chain in chains_by_expiry.items():
        # Check if this expiry has sufficient liquidity
        # Count instruments with two-sided quotes
        liquid_strikes = 0
        for strike, types in chain.items():
            call = types.get('call', {})
            put = types.get('put', {})
            
            # Check if either call or put has two-sided quotes
            call_liquid = (float(call.get('bid', 0)) > 0 and float(call.get('ask', 0)) > 0)
            put_liquid = (float(put.get('bid', 0)) > 0 and float(put.get('ask', 0)) > 0)
            
            if call_liquid or put_liquid:
                liquid_strikes += 1
                
        # Require at least 2 liquid strikes
        if liquid_strikes >= 2:
            valid_expiries.append(expiry)
            
    if not valid_expiries:
        return None
        
    # For now, return the first valid expiry (could be enhanced to sort by days to expiry)
    return valid_expiries[0]


def get_quote(items: List[Dict[str, Any]], otype: str, strike: float, expiry: Optional[str], logger: logging.Logger) -> Quote:
    """Extract a robust (bid, ask, mid) from raw option items."""
    otype = (otype or "").upper()
    best: Optional[Dict[str, Any]] = None
    for o in items or []:
        try:
            if str(o.get("type","")).upper() != otype:
                continue
            if float(o.get("strike")) != float(strike):
                continue
            if expiry and o.get("expiry_date") not in (expiry,):
                continue
            best = o
            break
        except Exception:
            continue
    if not best:
        reason_debug(logger, "REPL NO_OPTION_MATCH type=%s strike=%s expiry=%s", otype, str(strike), str(expiry))
        return Quote(None, None, None)
    bid = best.get("bid")
    ask = best.get("ask")
    mid = best.get("mid")
    try:
        bid = float(bid) if bid is not None else None
        ask = float(ask) if ask is not None else None
        if mid is None and bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        mid = float(mid) if mid is not None else None
    except Exception:
        bid = ask = mid = None
    # DEBUG: mark one‑sided quotes
    if best is not None and (bid is None or ask is None):
        reason_debug(logger, "REPL ONE_SIDED_QUOTE type=%s strike=%s has_bid=%s has_ask=%s",
                     otype, str(strike), str(bid is not None), str(ask is not None))
    return Quote(bid, ask, mid)


def nearest_vertical(all_options: List[Dict[str, Any]], K: float, is_above: bool, logger: logging.Logger) -> Optional[Tuple[str, float, float]]:
    """
    Find a small-width vertical around K using available strikes.
    Returns (otype, k_low, k_high) where otype in {"CALL","PUT"}.
    """
    strikes = sorted({float(o.get("strike")) for o in all_options if o.get("strike") is not None})
    if not strikes:
        reason_debug(logger, "REPL NO_OPTION_STRIKES")
        return None
    # Find nearest lower & higher strikes around K
    lower = max((s for s in strikes if s <= K), default=None)
    higher = min((s for s in strikes if s >= K and s != lower), default=None)
    if lower is None or higher is None or higher == lower:
        reason_debug(logger, "REPL INSUFFICIENT_STRIKES_AROUND_K K=%s", str(K))
        return None
    otype = "CALL" if is_above else "PUT"
    return (otype, float(lower), float(higher))