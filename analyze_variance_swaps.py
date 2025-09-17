from __future__ import annotations
from datetime import datetime, date, timezone
from typing import List, Dict, Any
from core.expiry_window import enumerate_expiries, pm_date_to_default_cutoff_utc
from config_loader import load_config  # existing helper

def _filter_chain_by_expiry_candidates(options: List[dict], candidates) -> List[dict]:
    if not candidates:
        return options
    allowed = {c.date.isoformat() for c in candidates}
    out = []
    for o in options:
        exp = o.get("expiry")
        d = None
        if isinstance(exp, str):
            d = exp[:10]
        elif isinstance(exp, date):
            d = exp.isoformat()
        elif isinstance(exp, datetime):
            d = exp.date().isoformat()
        if d and d in allowed:
            out.append(o)
    return out

def run_variance_swaps(markets, options, *, cfg=None, pm_date_field: str="polymarket_date"):
    """
    Wire-in: for each PM market date, enumerate 0..N expiries (hour-level DTE; liquidity gates)
    and run the existing computation on the filtered per-expiry universe(s).
    """
    cfg = cfg or load_config()
    hedging = getattr(cfg, "hedging", None)
    varcfg = getattr(hedging, "variance", None)
    policy_cfg = dict(
        expiry_policy=getattr(varcfg, "expiry_policy", "allow_far_with_unwind"),
        max_expiry_gap_days=getattr(varcfg, "max_expiry_gap_days", 60),
        max_expiries_considered=getattr(varcfg, "max_expiries_considered", 10),
        min_quotes_per_expiry=getattr(varcfg, "min_quotes_per_expiry", 2),
        min_strikes_required=getattr(varcfg, "min_strikes_required", 6),
    )
    results = []
    for m in markets:
        pm_d: date = m[pm_date_field] if isinstance(m, dict) else getattr(m, pm_date_field)
        pm_ts = pm_date_to_default_cutoff_utc(pm_d)
        cands = enumerate_expiries(pm_ts, options, policy_cfg=policy_cfg)
        opt_filtered = _filter_chain_by_expiry_candidates(options, cands)
        # >>> existing VS logic over opt_filtered (unchanged) <<<
        res = _run_single_vs_market(m, opt_filtered, cfg=cfg)  # assume your helper exists
        results.append({"pm_date": pm_d.isoformat(), "selected_expiries": [c.date.isoformat() for c in cands], "result": res})
    return results