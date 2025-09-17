from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, date, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json, os

@dataclass
class ExpiryCandidate:
    date: date
    expiry_ts_utc: datetime
    dte_hours: float
    is_monthly: bool
    is_weekly: bool
    quotes: int
    strikes: int
    kept: bool
    reason: str

def _expiry_dt_utc(d: date, cutoff=(23,59,59)) -> datetime:
    return datetime(d.year, d.month, d.day, cutoff[0], cutoff[1], cutoff[2], tzinfo=timezone.utc)

def _is_monthly(d: date) -> bool:
    # crude heuristic: if a week ahead crosses the month boundary, treat as monthly
    return d.month != (d + timedelta(days=7)).month

def _is_weekly(d: date) -> bool:
    return not _is_monthly(d)

def _collect_expiry_map(options: Iterable[dict]) -> Dict[date, List[dict]]:
    out: Dict[date, List[dict]] = {}
    for o in options:
        raw = o.get("expiry")
        if raw is None:
            continue
        if isinstance(raw, datetime):
            dd = raw.date()
        elif isinstance(raw, date):
            dd = raw
        elif isinstance(raw, str):
            try:
                y, m, d = int(raw[0:4]), int(raw[5:7]), int(raw[8:10])
                dd = date(y, m, d)
            except Exception:
                continue
        else:
            continue
        out.setdefault(dd, []).append(o)
    return out

def _count_quotes_and_strikes(opts_for_d: List[dict]) -> Tuple[int,int]:
    strikes = set()
    for r in opts_for_d:
        k = r.get("strike")
        if k is None:
            continue
        try:
            strikes.add(float(k))
        except Exception:
            try:
                strikes.add(float(str(k)))
            except Exception:
                pass
    return len(opts_for_d), len(strikes)

def enumerate_expiries(
    pm_settlement_ts_utc: datetime,
    options_universe: Sequence[dict],
    policy_cfg: Optional[Dict[str, Any]] = None,
    *,
    debug_jsonl: str = "debug_runs/expiry_debug.jsonl",
) -> List[ExpiryCandidate]:
    """
    Enumerate expiries relative to a PM settlement timestamp, honoring policy_cfg.
    Returns a chronologically ordered list (kept only) with metadata and reasons.
    Defaults mirror your hedging.variance intent.
    """
    if pm_settlement_ts_utc.tzinfo is None:
        pm_settlement_ts_utc = pm_settlement_ts_utc.replace(tzinfo=timezone.utc)
    pcfg = dict(
        expiry_policy="allow_far_with_unwind",
        max_expiry_gap_days=60,
        max_expiries_considered=10,
        min_quotes_per_expiry=2,
        min_strikes_required=6,
    )
    if policy_cfg:
        for k, v in policy_cfg.items():
            if k in pcfg:
                pcfg[k] = v

    e_map = _collect_expiry_map(options_universe)
    uniq = sorted(e_map.keys())
    all_candidates: List[ExpiryCandidate] = []
    kept: List[ExpiryCandidate] = []
    for d in uniq:
        e_ts = _expiry_dt_utc(d)
        dte_h = (e_ts - pm_settlement_ts_utc).total_seconds()/3600.0
        quotes, strikes = _count_quotes_and_strikes(e_map[d])
        cand = ExpiryCandidate(
            date=d,
            expiry_ts_utc=e_ts,
            dte_hours=dte_h,
            is_monthly=_is_monthly(d),
            is_weekly=_is_weekly(d),
            quotes=quotes,
            strikes=strikes,
            kept=False,
            reason="init",
        )
        all_candidates.append(cand)
        if dte_h < 0:
            cand.reason = "before_pm"
            continue
        if dte_h > pcfg["max_expiry_gap_days"]*24.0:
            cand.reason = "beyond_max_gap"
            continue
        if quotes < pcfg["min_quotes_per_expiry"]:
            cand.reason = "below_min_quotes"
            continue
        if strikes < pcfg["min_strikes_required"]:
            cand.reason = "below_min_strikes"
            continue
        cand.kept = True
        cand.reason = "kept_window_liquidity"
        kept.append(cand)

    kept_sorted = sorted(kept, key=lambda c: c.expiry_ts_utc)[: pcfg["max_expiries_considered"]]

    # structured debug
    try:
        os.makedirs(os.path.dirname(debug_jsonl), exist_ok=True)
        with open(debug_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "fn": "enumerate_expiries",
                "pm_settlement_ts_utc": pm_settlement_ts_utc.isoformat(),
                "policy_cfg": pcfg,
                "candidates": [asdict(c) for c in all_candidates],
                "kept": [asdict(c) for c in kept_sorted],
            }, default=str) + "\n")
    except Exception:
        pass
    return kept_sorted

def pm_date_to_default_cutoff_utc(pm_date: date) -> datetime:
    """Date-only PM market -> 23:59:59Z on that date."""
    return datetime(pm_date.year, pm_date.month, pm_date.day, 23, 59, 59, tzinfo=timezone.utc)