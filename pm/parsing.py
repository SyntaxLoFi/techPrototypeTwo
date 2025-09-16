from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional, Tuple

# Minimal month map; avoids external deps.
_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_TZ_ALIASES = {
    "et": -4,  # naive handling: ET ~ UTC-4 during DST; good enough for expiry selection
    "est": -5, "edt": -4,
    "utc": 0, "gmt": 0,
}


def _now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)


def _to_tz_offset(tz: str) -> int:
    tz = tz.lower()
    return _TZ_ALIASES.get(tz, 0)


def _mk_dt(year: int, month: int, day: int, hh: int = 0, mm: int = 0, tz_off: int = 0) -> datetime:
    return datetime(year, month, day, hh, mm, tzinfo=timezone(timedelta(hours=tz_off)))


def parse_market_expiry(text: str, tz: str = "UTC") -> Tuple[Literal["point", "range"], Optional[datetime], Optional[Tuple[datetime, datetime]]]:
    """
    Parse common PM phrasing into a single point or an inclusive date range.

    Supported patterns (case-insensitive):
      - "... on <Month> <D>[, <YYYY>][ at <H>:<MM> <AM|PM> <TZ>]"  -> point
      - "... on <Month> <D>[, <YYYY>]"                             -> point (00:00)
      - "... <Month> <D>-<D>[, <YYYY>]"                            -> range [D1..D2]
      - "... in <Month>[ <YYYY>]"                                  -> range [month-start..month-end]
      - "... by <Month> <D>[, <YYYY>]"                             -> point (deadline)
    """
    s = text.strip()
    s_l = s.lower()
    now = _now_utc()
    year_default = now.year
    tz_off = _to_tz_offset(tz)

    # 1) Range like "September 15-21[, 2025]"
    m = re.search(r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})\s*-\s*(\d{1,2})(?:,\s*(\d{4}))?", s_l)
    if m:
        mon = _MONTHS[m.group(1)]
        d1 = int(m.group(2))
        d2 = int(m.group(3))
        yr = int(m.group(4)) if m.group(4) else year_default
        start = _mk_dt(yr, mon, d1, 0, 0, tz_off)
        end = _mk_dt(yr, mon, d2, 23, 59, tz_off)
        return "range", None, (start, end)

    # 2) "in September [2025]"
    m = re.search(r"in\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+(\d{4}))?", s_l)
    if m:
        mon = _MONTHS[m.group(1)]
        yr = int(m.group(2)) if m.group(2) else year_default
        # month start & end (naive month length set)
        month_len = [31, 29 if (yr % 400 == 0 or (yr % 4 == 0 and yr % 100 != 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mon - 1]
        start = _mk_dt(yr, mon, 1, 0, 0, tz_off)
        end = _mk_dt(yr, mon, month_len, 23, 59, tz_off)
        return "range", None, (start, end)

    # 3) "by December 31[, 2025]" -> treat as a point/deadline
    m = re.search(r"by\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:,\s*(\d{4}))?", s_l)
    if m:
        mon = _MONTHS[m.group(1)]
        day = int(m.group(2))
        yr = int(m.group(3)) if m.group(3) else year_default
        point = _mk_dt(yr, mon, day, 23, 59, tz_off)
        return "point", point, None

    # 4) "on September 16[, 2025][ at 12pm ET]"
    m = re.search(
        r"on\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:,\s*(\d{4}))?(?:\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*([A-Za-z]+)?)?",
        s_l,
    )
    if m:
        mon = _MONTHS[m.group(1)]
        day = int(m.group(2))
        yr = int(m.group(3)) if m.group(3) else year_default
        hh = 0
        mm = 0
        if m.group(4):
            hh = int(m.group(4))
            mm = int(m.group(5) or "0")
            ampm = m.group(6)
            if ampm == "pm" and hh != 12:
                hh += 12
            if ampm == "am" and hh == 12:
                hh = 0
        tz_txt = m.group(7) or tz
        tz_off_local = _to_tz_offset(tz_txt)
        point = _mk_dt(yr, mon, day, hh, mm, tz_off_local)
        return "point", point, None

    # Fallback: assume deadline "by <detected month end>" if a month is present
    m = re.search(r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)", s_l)
    if m:
        mon = _MONTHS[m.group(1)]
        yr = year_default
        month_len = [31, 29 if (yr % 400 == 0 or (yr % 4 == 0 and yr % 100 != 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mon - 1]
        point = _mk_dt(yr, mon, month_len, 23, 59, tz_off)
        return "point", point, None

    # Last resort: today as point
    return "point", now, None