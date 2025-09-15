from __future__ import annotations
from typing import Optional, Union
from datetime import datetime, timezone

Number = Union[int, float]

def _to_datetime_utc(value) -> Optional[datetime]:
    """Best-effort parse to timezone-aware UTC datetime."""
    if value is None:
        return None
    # Already a datetime
    if isinstance(value, datetime):
        if value.tzinfo is None:
            # Treat naive as UTC (do NOT assume local)
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    # Numeric epoch seconds/milliseconds
    if isinstance(value, (int, float)):
        # Heuristic: milliseconds if >= 1e12 (roughly post-2001 in ms)
        sec = float(value) / (1000.0 if float(value) >= 1e12 else 1.0)
        try:
            return datetime.fromtimestamp(sec, tz=timezone.utc)
        except Exception:
            return None
    # String (ISO8601)
    s = str(value).strip()
    if not s:
        return None
    try:
        # Normalize trailing 'Z' to '+00:00'
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        # Could extend with more formats here, but keep conservative
        return None

def compute_days_to_expiry(end_iso_or_ts, now: Optional[datetime] = None) -> Optional[float]:
    """
    Compute non-negative days-to-expiry from a variety of inputs:
      - ISO8601 string (with 'Z' or explicit offset)
      - epoch seconds or milliseconds
      - datetime (naive treated as UTC)
    Returns:
      float days >= 0.0, or None if parsing fails.
    """
    end_dt = _to_datetime_utc(end_iso_or_ts)
    if end_dt is None:
        return None
    ref = now.astimezone(timezone.utc) if isinstance(now, datetime) else datetime.now(timezone.utc)
    delta_sec = (end_dt - ref).total_seconds()
    days = max(0.0, delta_sec / 86400.0)
    return float(days)