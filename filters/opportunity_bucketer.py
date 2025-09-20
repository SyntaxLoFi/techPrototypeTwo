# filters/opportunity_bucketer.py
# Bucket classification + sub-ranking for opportunities.
# Buckets: TRUE_ARBITRAGE, NEAR_ARB_SUPERHEDGED, ROBUST_EV_PLUS, SPECULATIVE

from __future__ import annotations
from typing import Dict, List, Mapping, Optional
from math import log1p

# Tunables (can be overridden by env/config at import site)
DEFAULT_NEAR_ARB_GAP_PCT      = 0.05   # 5% of capital
DEFAULT_ES_CAP_PER_DOLLAR     = 0.50   # ES95 per $ cap for Robust EV+
DEFAULT_DAYS_FALLBACK         = 30.0

def _get_float(d: Mapping, *keys, default: float = 0.0) -> float:
    for k in keys:
        try:
            v = d.get(k)
            if v is None: 
                continue
            return float(v)
        except Exception:
            continue
    return float(default)

def _capital_at_risk(opp: Dict) -> float:
    # Prefer explicit capital fields; fall back to worst-case loss magnitude.
    for key in ('capital_at_risk','required_capital','position_size','stake','portfolio_cost','notional','position_cost'):
        v = _get_float(opp, key, default=None)
        if v and v > 0:
            return v
    y = _get_float(opp, 'payoff_if_yes','profit_if_yes')
    n = _get_float(opp, 'payoff_if_no','profit_if_no')
    wcl = -min(y, n, 0.0)
    return abs(wcl)

def _days_to_resolution(opp: Dict) -> float:
    for k in ('days_to_resolution','days_to_option_expiry','days_to_expiry'):
        v = _get_float(opp, k, default=None)
        if v and v > 0:
            return v
    return DEFAULT_DAYS_FALLBACK

def _prob_yes(opp: Dict) -> float:
    probs = opp.get('probabilities', {}) or {}
    p = _get_float(probs, 'blended', default=None)
    if p is None:
        p = _get_float(probs, 'pm_implied', default=0.5)
    # clip to avoid log issues
    return max(1e-6, min(1-1e-6, p))

def _min_payoff_after_costs(opp: Dict) -> float:
    y = _get_float(opp, 'payoff_if_yes','profit_if_yes')
    n = _get_float(opp, 'payoff_if_no','profit_if_no')
    costs = _get_float(opp.get('metrics', {}) if opp.get('metrics') else {}, 'actual_costs', default=0.0)
    return min(y, n) - costs

def _expected_shortfall_95_two_point(opp: Dict, p_yes: float) -> float:
    """Closed-form ES95 for a two-point P&L distribution.
    If the probability of the worse state >= 5%, ES = worst loss.
    Otherwise ES scales linearly by prob / 5% (conservative interpolation)."""
    y = _get_float(opp, 'payoff_if_yes','profit_if_yes')
    n = _get_float(opp, 'payoff_if_no','profit_if_no')
    worst = min(y, n)
    loss = max(0.0, -worst)
    if loss <= 0.0:
        return 0.0
    prob_loss = (1.0 - p_yes) if y < n else p_yes
    tail = 0.05
    return loss if prob_loss >= tail else loss * (prob_loss / tail)

def _growth_score(opp: Dict, p_yes: float, car: float) -> float:
    y = _get_float(opp, 'payoff_if_yes','profit_if_yes')
    n = _get_float(opp, 'payoff_if_no','profit_if_no')
    if car <= 0.0:
        return 0.0
    ry = max(-0.999999, y / car)
    rn = max(-0.999999, n / car)
    return p_yes * log1p(ry) + (1.0 - p_yes) * log1p(rn)

def tag_opportunities_in_place(ops: List[Dict],
                               near_gap_pct: float = DEFAULT_NEAR_ARB_GAP_PCT,
                               es_cap_per_dollar: float = DEFAULT_ES_CAP_PER_DOLLAR) -> None:
    """Annotate each opportunity with:
        - bucket: TRUE_ARBITRAGE | NEAR_ARB_SUPERHEDGED | ROBUST_EV_PLUS | SPECULATIVE
        - metrics: arb_slack, super_hedge_gap(_per_dollar), ev_per_dollar(_per_day),
                   es_95(_per_dollar), growth_score, profit_amounts, and convenience ratios.
    Idempotent and order-preserving."""
    for opp in ops:
        p = _prob_yes(opp)
        car  = _capital_at_risk(opp)
        days = _days_to_resolution(opp)
        y = _get_float(opp, 'payoff_if_yes','profit_if_yes')
        n = _get_float(opp, 'payoff_if_no','profit_if_no')

        # Core quantities
        min_after_costs = _min_payoff_after_costs(opp)
        arb_slack = min_after_costs
        super_gap = max(0.0, -min_after_costs)
        super_gap_per_dollar = (super_gap / car) if car > 0 else float('inf')
        ev = p*y + (1.0-p)*n
        ev_per_dollar = (ev / car) if car > 0 else 0.0
        ev_per_dollar_per_day = ev_per_dollar / max(1.0, days)
        es95 = _expected_shortfall_95_two_point(opp, p)
        es95_per_dollar = (es95 / car) if car > 0 else float('inf')
        growth = _growth_score(opp, p, car)

        # Profit amounts useful for sub-ranking
        profit_unhedged = ev
        profit_unhedged_per_day = profit_unhedged / max(1.0, days)
        profit_after_superhedge = ev - super_gap
        profit_after_superhedge_per_day = profit_after_superhedge / max(1.0, days)
        guaranteed_profit = max(0.0, arb_slack)  # floor if true arbitrage
        guaranteed_profit_per_day = guaranteed_profit / max(1.0, days)
        guaranteed_profit_per_dollar_per_day = (guaranteed_profit / car / max(1.0, days)) if car > 0 else 0.0

        # Record metrics
        opp.setdefault('metrics', {}).update({
            'arb_slack': arb_slack,
            'super_hedge_gap': super_gap,
            'super_hedge_gap_per_dollar': super_gap_per_dollar,
            'ev': ev,
            'ev_per_dollar': ev_per_dollar,
            'ev_per_dollar_per_day': ev_per_dollar_per_day,
            'es_95': es95,
            'es_95_per_dollar': es95_per_dollar,
            'growth_score': growth,
            'capital_at_risk': car,
            'time_to_resolution_days': days,
            # profit-focused fields for sub-ranking
            'profit_unhedged': profit_unhedged,
            'profit_unhedged_per_day': profit_unhedged_per_day,
            'profit_after_superhedge': profit_after_superhedge,
            'profit_after_superhedge_per_day': profit_after_superhedge_per_day,
            'guaranteed_profit': guaranteed_profit,
            'guaranteed_profit_per_day': guaranteed_profit_per_day,
            'guaranteed_profit_per_dollar_per_day': guaranteed_profit_per_dollar_per_day,
        })

        # Classification
        if arb_slack > 0.0:
            opp['bucket'] = 'TRUE_ARBITRAGE'
        elif super_gap_per_dollar <= near_gap_pct:
            opp['bucket'] = 'NEAR_ARB_SUPERHEDGED'
        elif (ev_per_dollar_per_day > 0.0) and (es95_per_dollar <= es_cap_per_dollar):
            opp['bucket'] = 'ROBUST_EV_PLUS'
        else:
            opp['bucket'] = 'SPECULATIVE'

def group_by_bucket(ops: List[Dict]) -> Dict[str, List[Dict]]:
    out = {'TRUE_ARBITRAGE': [], 'NEAR_ARB_SUPERHEDGED': [], 'ROBUST_EV_PLUS': [], 'SPECULATIVE': []}
    for o in ops:
        out.setdefault(o.get('bucket','SPECULATIVE'), []).append(o)
    return out

def bucket_sorted_list(ops: List[Dict],
                       sort_keys: Optional[Dict[str,str]] = None,
                       bucket_order: Optional[List[str]] = None) -> List[Dict]:
    """Return a single list sorted by bucket, with per-bucket sub-ranking.
    sort_keys maps bucket -> metric name. Defaults:
      TRUE_ARBITRAGE: 'guaranteed_profit' (desc)
      NEAR_ARB_SUPERHEDGED: 'super_hedge_gap' (asc)
      ROBUST_EV_PLUS: 'ev_per_dollar_per_day' (desc)
    Bucket order defaults to TRUE_ARBITRAGE, NEAR_ARB_SUPERHEDGED, ROBUST_EV_PLUS, SPECULATIVE."""
    sort_keys = sort_keys or {
        'TRUE_ARBITRAGE': 'guaranteed_profit',
        'NEAR_ARB_SUPERHEDGED': 'super_hedge_gap',
        'ROBUST_EV_PLUS': 'ev_per_dollar_per_day',
        'SPECULATIVE': 'ev_per_dollar_per_day',
    }
    bucket_order = bucket_order or ['TRUE_ARBITRAGE','NEAR_ARB_SUPERHEDGED','ROBUST_EV_PLUS','SPECULATIVE']
    grouped = group_by_bucket(ops)

    def _key(o, name):
        return o.get('metrics', {}).get(name, 0.0)

    result: List[Dict] = []
    for b in bucket_order:
        items = list(grouped.get(b, []))
        k = sort_keys.get(b, 'ev_per_dollar_per_day')
        # Direction per bucket
        if b in ('NEAR_ARB_SUPERHEDGED',):
            items.sort(key=lambda o: (_key(o, k), -_key(o, 'profit_after_superhedge_per_day')))
        elif b in ('TRUE_ARBITRAGE',):
            items.sort(key=lambda o: (-_key(o, k), -_key(o, 'profit_unhedged_per_day')))
        else:
            items.sort(key=lambda o: (-_key(o, k), -_key(o, 'growth_score'), _key(o, 'es_95_per_dollar')))
        result.extend(items)
    return result