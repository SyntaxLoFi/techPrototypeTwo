from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Optional, Sequence, Tuple

from .parsing import parse_market_expiry


@dataclass
class PMContract:
    """
    Represents a prediction-market contract's resolution time specification.
    Supports a single point-in-time ("on Sep 16 at 12pm ET"), an inclusive date
    range ("Sep 15-21"), month windows ("in September"), and deadlines
    ("by Dec 31").
    """
    market_text: str
    tz: str = "UTC"
    max_expiries: int = 5

    # Derived from parse_market_expiry():
    mode: str = "point"  # 'point' | 'range'
    point: Optional[datetime] = None
    window: Optional[Tuple[datetime, datetime]] = None

    @classmethod
    def from_market_text(
        cls, text: str, tz: str = "UTC", max_expiries: int = 5
    ) -> "PMContract":
        mode, point, window = parse_market_expiry(text, tz=tz)
        return cls(
            market_text=text,
            tz=tz,
            max_expiries=max_expiries,
            mode=mode,
            point=point,
            window=window,
        )

    def candidate_expiries(self, available: Sequence[date]) -> List[date]:
        """
        Select a set of option expiries for this PM contract from the
        exchange's available expiries.
        - range/window  -> all expiries within [start, end], optionally downsampled
        - point/deadline -> K nearest expiries (bracketing the point)
        """
        uniq = sorted(set(available))
        if not uniq:
            return []

        if self.mode == "range" and self.window:
            start_d = self.window[0].date()
            end_d = self.window[1].date()
            selected = [d for d in uniq if start_d <= d <= end_d]
            if self.max_expiries and len(selected) > self.max_expiries:
                # Downsample evenly to at most max_expiries
                step = max(1, len(selected) // self.max_expiries)
                selected = selected[::step][: self.max_expiries]
            return selected

        # point/deadline behavior
        if self.point is not None:
            target = self.point.date()
            k = max(2, self.max_expiries)  # at least bracket with 2 expiries if possible
            # Sort by absolute distance in days
            selected = sorted(uniq, key=lambda d: abs((d - target).days))[: k]
            return sorted(selected)

        # Fallback: just return up to max_expiries from available
        return uniq[: self.max_expiries or len(uniq)]


def collect_unique_expiries(options: Iterable[dict]) -> List[date]:
    """
    Extract and normalize unique expiry dates from option payloads.
    Accepts 'expiry' as either a date/datetime object or 'YYYY-MM-DD' string.
    """
    out: set[date] = set()
    for opt in options:
        raw = opt.get("expiry")
        if raw is None:
            continue
        if isinstance(raw, datetime):
            out.add(raw.date())
        elif isinstance(raw, date):
            out.add(raw)
        elif isinstance(raw, str):
            # Allow 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM...' etc.
            y, m, d = int(raw[0:4]), int(raw[5:7]), int(raw[8:10])
            out.add(date(y, m, d))
    return sorted(out)