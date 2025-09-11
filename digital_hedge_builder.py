# digital_hedge_builder.py
from typing import Dict, Any, List, Tuple, Optional
from math import isfinite

# --- canonical option family helpers ---
try:
    # when running as a package
    from strategies.options.utils.opt_keys import key_for_is_above, chain_slice
except Exception:
    # robust fallback so local/script runs won't crash if PYTHONPATH isn't set
    def key_for_is_above(is_above: bool) -> str:
        return "call" if bool(is_above) else "put"
    def chain_slice(chain_for_expiry, strike, *, family=None, is_above=None):
        try:
            strike_key = str(float(strike))
        except Exception:
            strike_key = str(strike)
        fam = (family or ("call" if bool(is_above) else "put"))
        return (chain_for_expiry.get(strike_key) or {}).get(fam) or {}

from typing import Iterable
from execution_pricing import option_exec_price

def _safe(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if isfinite(v) else default
    except Exception:
        return default

def _bracket_strikes(strikes: List[float], K: float) -> Optional[tuple[float,float]]:
    """Return (K_low, K_high) such that K_low ≤ K < K_high. If K equals a strike,
    prefer (K, next_higher)."""
    xs = sorted({float(s) for s in strikes})
    if not xs:
        return None
    # if exact match, pick the next higher for finite-diff replication
    for i, s in enumerate(xs):
        if abs(s - K) < 1e-9:
            if i+1 < len(xs):
                return (s, xs[i+1])
            return None
    # otherwise find the bracket
    for i in range(len(xs)-1):
        if xs[i] <= K < xs[i+1]:
            return (xs[i], xs[i+1])
    return None

def build_digital_vertical_at_K(
    *,
    is_above: bool,
    K: float,
    expiry: str,
    chain_for_expiry: Dict[float, Dict[str, Dict[str, float]]],  # {strike: {'CALL': {...}, 'PUT': {...}}}
) -> Optional[Dict[str, Any]]:
    """
    Returns a dict with:
      width: δ
      digital_buy_per_1:  debit/δ  (to LONG the digital)
      digital_sell_per_1: credit/δ (to SHORT the digital)
      legs_longD:  executable legs (BUY digital)
      legs_shortD: executable legs (SELL digital)
    All prices are USD numeraire after slippage and fee deductions.
    """
    if not chain_for_expiry or K <= 0:
        return None

    strikes = list(chain_for_expiry.keys())
    br = _bracket_strikes(strikes, K)
    if not br:
        return None
    k_lo, k_hi = br
    width = float(k_hi - k_lo)
    if width <= 0:
        return None

    # no need to compute opt_key manually anymore
    K_neighbor = float(k_hi if is_above else k_lo)

    # Quotes
    def _q(strike: float) -> Dict[str, float]:
        # Use canonical chain slicing keyed by 'call'/'put'
        d = chain_slice(chain_for_expiry, strike, is_above=is_above)
        bid = _safe(d.get('bid', d.get('best_bid', d.get('mid', 0.0))))
        ask = _safe(d.get('ask', d.get('best_ask', d.get('mid', 0.0))))
        return {'bid': bid, 'ask': ask}

    qK   = _q(K)
    qN   = _q(K_neighbor)

    # BUY digital legs: (+1 @K, -1 @neighbor) in option-space mapped to USD via execution
    if is_above:
        # Digital(ABOVE): long = BUY C(K), SELL C(K+δ)
        buy_leg1  = {'type': 'CALL', 'strike': float(K),         'action': 'BUY',  'expiry': expiry}
        buy_leg2  = {'type': 'CALL', 'strike': float(K_neighbor),'action': 'SELL', 'expiry': expiry}
        sell_leg1 = {'type': 'CALL', 'strike': float(K),         'action': 'SELL', 'expiry': expiry}
        sell_leg2 = {'type': 'CALL', 'strike': float(K_neighbor),'action': 'BUY',  'expiry': expiry}
        long_cash  = option_exec_price('BUY',  qK['bid'], qK['ask']) + option_exec_price('SELL', qN['bid'], qN['ask'])
        short_cash = option_exec_price('SELL', qK['bid'], qK['ask']) + option_exec_price('BUY',  qN['bid'], qN['ask'])
        debit  = long_cash
        credit = -short_cash
    else:
        # Digital(BELOW): long = BUY P(K), SELL P(K-δ)
        buy_leg1  = {'type': 'PUT', 'strike': float(K),         'action': 'BUY',  'expiry': expiry}
        buy_leg2  = {'type': 'PUT', 'strike': float(K_neighbor),'action': 'SELL', 'expiry': expiry}
        sell_leg1 = {'type': 'PUT', 'strike': float(K),         'action': 'SELL', 'expiry': expiry}
        sell_leg2 = {'type': 'PUT', 'strike': float(K_neighbor),'action': 'BUY',  'expiry': expiry}
        long_cash  = option_exec_price('BUY',  qK['bid'], qK['ask']) + option_exec_price('SELL', qN['bid'], qN['ask'])
        short_cash = option_exec_price('SELL', qK['bid'], qK['ask']) + option_exec_price('BUY',  qN['bid'], qN['ask'])
        debit  = long_cash
        credit = -short_cash

    # Fees and slippage already applied in option_exec_price(); do not apply again.
    # debit and credit are positive magnitudes: debit (outflow), credit (inflow).

    # Exact-K anchoring check: at least one leg MUST be exactly at K to be "anchored".
    # (For ABOVE: [K, K+δ] CALL vertical; for BELOW: [K−δ, K] PUT vertical.)
    try:
        _all_legs = [buy_leg1, buy_leg2, sell_leg1, sell_leg2]
    except NameError:
        _all_legs = []
    has_exact_k = any(abs(float(leg.get('strike', float('nan'))) - float(K)) < 1e-9 for leg in _all_legs)

    return {
        'width': width,
        'k_low': k_lo,
        'k_high': k_hi,
        'digital_buy_per_1':  (debit  / width),
        'digital_sell_per_1': (credit / width),
        'legs_longD':  [buy_leg1, buy_leg2],
        'legs_shortD': [sell_leg1, sell_leg2],
        # New metadata for strict-arb gating & labeling:
        'has_exact_k': abs(k_lo - K) < 1e-9,
        'approximation_reason': None if abs(k_lo - K) < 1e-9 else 'no_exact_strike_at_K'
    }