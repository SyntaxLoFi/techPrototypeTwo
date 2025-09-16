from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Iterable, List, Optional, Sequence, Tuple


def _as_date(raw) -> Optional[date]:
    if raw is None:
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, str):
        # Accept 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM...'
        try:
            y, m, d = int(raw[0:4]), int(raw[5:7]), int(raw[8:10])
            return date(y, m, d)
        except Exception:
            return None
    return None


def collect_unique_expiries(options: Iterable[dict]) -> List[date]:
    s = set()
    for o in options:
        d = _as_date(o.get("expiry"))
        if d:
            s.add(d)
    return sorted(s)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _write_debug(debug_path: Optional[str], payload: dict) -> None:
    if not debug_path:
        return
    try:
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        # best-effort only
        pass


def filter_options_by_expiry(
    options: Sequence[dict],
    pm_days_to_expiry: Optional[float] = None,
    *,
    inclusive: bool = True,
    pm_expiries: Optional[Sequence[date]] = None,
    window: Optional[Tuple[date, date]] = None,
    max_expiries: Optional[int] = None,
    now: Optional[datetime] = None,
    debug_path: str = "debug_runs/expiry_debug.jsonl",
) -> Tuple[List[dict], List[date]]:
    """
    Backward-compatible expansion of the legacy expiry filter.
    New behaviors:
      - If 'pm_expiries' provided: include options whose expiry is in that set.
      - If 'window' provided: include options with start <= expiry <= end.
      - Else (legacy): choose the single expiry whose DTE is closest to pm_days_to_expiry.
    Returns: (filtered_options, selected_expiries)
    """
    now = now or _now_utc()
    expiries_all = collect_unique_expiries(options)

    selected: List[date] = []
    mode = "legacy_single"
    if pm_expiries:
        mode = "explicit_set"
        selected = sorted(set(pm_expiries))
    elif window:
        mode = "window"
        start, end = window
        selected = [d for d in expiries_all if start <= d <= end]
    else:
        # Legacy behavior: closest expiry to pm_days_to_expiry
        if pm_days_to_expiry is None:
            # Fall back to "next available"
            selected = expiries_all[:1]
        else:
            # Build target date and pick closest
            target = (now.date())
            # pm_days_to_expiry is a float in days; round to nearest minute for comparability
            from datetime import timedelta
            target_d = target
            # Find the expiry whose (expiry - today).days is closest to pm_days_to_expiry
            # (uses calendar-day distance; good enough for bucket selection)
            def _dist(d: date) -> float:
                return abs((d - target_d).days - pm_days_to_expiry)
            chosen = sorted(expiries_all, key=_dist)[:1]
            selected = chosen

    if max_expiries and len(selected) > max_expiries:
        selected = selected[:max_expiries]

    filtered = [o for o in options if _as_date(o.get("expiry")) in set(selected)]

    # Debug trace (mirrors your existing JSONL for continuity)
    sample = []
    for i, o in enumerate(options[:3]):
        sample.append(
            {
                "i": i,
                "expiry": o.get("expiry"),
                "dte": "N/A",  # keep shape identical to your current log
            }
        )
    _write_debug(
        debug_path,
        {
            "ts": now.isoformat(),
            "fn": "filter_options_by_expiry",
            "mode": mode,
            "options_count": len(options),
            "inclusive": inclusive,
            "unique_expiries_found": len(expiries_all),
            "selected_expiries": [str(d) for d in selected],
            "sample_options": sample,
        },
    )

    return filtered, selected