from __future__ import annotations
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from core.expiry_window import enumerate_expiries, pm_date_to_default_cutoff_utc
try:
    from config_loader import load_config
except Exception:
    load_config = None

MARKER_ATTRS = ("IS_OPTIONS_STRATEGY", "is_options_strategy")

def _extract_pm_date(m: Any) -> Optional[date]:
    """Best-effort Polymarket/PM date extraction."""
    keys = ("polymarket_date", "pm_date", "date")
    v = None
    if isinstance(m, dict):
        for k in keys:
            if k in m and m[k]:
                v = m[k]; break
    else:
        for k in keys:
            if hasattr(m, k):
                v = getattr(m, k); break
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str) and len(v) >= 10:
        try:
            return date(int(v[0:4]), int(v[5:7]), int(v[8:10]))
        except Exception:
            return None
    return None

def _filter_chain_to_date(options: Iterable[dict], d: date) -> List[dict]:
    allowed = d.isoformat()
    out = []
    for o in options:
        e = o.get("expiry")
        if isinstance(e, datetime):
            ee = e.date().isoformat()
        elif isinstance(e, date):
            ee = e.isoformat()
        else:
            ee = str(e)[:10] if e is not None else ""
        if ee == allowed:
            out.append(o)
    return out

def _build_policy_cfg(cfg: Any) -> Dict[str, Any]:
    default = dict(
        expiry_policy="allow_far_with_unwind",
        max_expiry_gap_days=60,
        max_expiries_considered=10,
        min_quotes_per_expiry=2,
        min_strikes_required=6,
    )
    try:
        hedging = getattr(cfg, "hedging", None)
        varcfg = getattr(hedging, "variance", None)
        if varcfg:
            default.update({
                "max_expiry_gap_days": getattr(varcfg, "max_expiry_gap_days", default["max_expiry_gap_days"]),
                "max_expiries_considered": getattr(varcfg, "max_expiries_considered", default["max_expiries_considered"]),
                "min_quotes_per_expiry": getattr(varcfg, "min_quotes_per_expiry", default["min_quotes_per_expiry"]),
                "min_strikes_required": getattr(varcfg, "min_strikes_required", default["min_strikes_required"]),
            })
    except Exception:
        pass
    return default

def _wrap_callable(strategy: Any, fn_name: str, cfg: Any) -> None:
    orig = getattr(strategy, fn_name, None)
    if not callable(orig):
        return
    def wrapped(*args, **kwargs):
        # Heuristic: (market, options, ...) positionally or via kwargs
        market = kwargs.get("market") if "market" in kwargs else (args[0] if len(args) >= 1 else None)
        options = kwargs.get("options") if "options" in kwargs else (args[1] if len(args) >= 2 else None)
        if market is None or options is None:
            return orig(*args, **kwargs)
        pm_d = _extract_pm_date(market)
        if pm_d is None:
            return orig(*args, **kwargs)
        pm_ts = pm_date_to_default_cutoff_utc(pm_d)
        policy = _build_policy_cfg(cfg)
        cands = enumerate_expiries(pm_ts, options, policy_cfg=policy)
        results = []
        for c in cands:
            filtered = _filter_chain_to_date(options, c.date)
            if "options" in kwargs:
                kw = dict(kwargs); kw["options"] = filtered
                res = orig(*args, **kw)
            else:
                arglist = list(args)
                if len(arglist) >= 2:
                    arglist[1] = filtered
                res = orig(*arglist, **kwargs)
            results.append({"expiry": c.date.isoformat(), "result": res})
        return {"selected_expiries": [c.date.isoformat() for c in cands], "per_expiry": results}
    setattr(strategy, fn_name, wrapped)

def try_wrap_expiry_layer(strategy: Any, cfg: Any = None) -> Any:
    """Wraps a single strategy instance in an expiry-aware layer if it looks like an options strategy."""
    is_options = False
    for a in MARKER_ATTRS:
        if getattr(strategy, a, False):
            is_options = True
            break
    if not is_options:
        # Fallback: detect by module path
        mod = getattr(strategy.__class__, "__module__", "") or ""
        if mod.startswith("strategies.options"):
            is_options = True
    if not is_options:
        return strategy
    if cfg is None and load_config:
        try:
            cfg = load_config()
        except Exception:
            cfg = None
    for name in ("evaluate", "run", "build", "build_opportunities", "execute"):
        _wrap_callable(strategy, name, cfg)
    return strategy

def wrap_all_options_strategies(strategies: List[Any], cfg: Any = None) -> List[Any]:
    out = []
    for s in strategies or []:
        out.append(try_wrap_expiry_layer(s, cfg=cfg))
    return out