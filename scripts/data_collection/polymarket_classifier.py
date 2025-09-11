# Polymarket market class classifier (price‑only, final)
# Classifies ONLY genuine price markets:
#  - "between $X and $Y", "from/within $X to/through $Y", "$X–$Y" (with currency context) → RANGE_BUCKET
#  - "greater/less than/above/below/over/under $X", "at least/at most $X", "$X or more/less", "reach/hit $X" → SINGLE_THRESHOLD
#  - DIRECTIONAL_PERIOD only when price context exists
# Blocks non‑price contexts (spending, employees, ETF approvals, "say/tweet/mention", reserve, elections, etc.)
# Guards against date ranges like "September 1–7" being parsed as price ranges.

from __future__ import annotations
import re
from typing import Optional, Literal, TypedDict


MarketClass = Literal['SINGLE_THRESHOLD', 'RANGE_BUCKET', 'DIRECTIONAL_PERIOD', 'UNKNOWN']

class Classification(TypedDict, total=False):
    marketClass: MarketClass
    relation: Literal['>', '<', '>=', '<=', '==']
    threshold: float
    rangeLow: float
    rangeHigh: float
    matched: str

_MONTH_RE = re.compile(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)', re.I)
_NON_PRICE_NEARBY = re.compile(
    r'\b('
    r'percent|percentage|employees?|headcount|people|users?|subscribers|seats|votes?|ballots|cases|orders|deliveries|units|'
    r'spending|budget|deficit|debt|revenue|tax(?:es)?|appropriation|funding|aid|'
    r'approval|approve|approved|nomination|election|primary|poll|'
    r'law|bill|act|statute|court|lawsuit|reserve|treasury|cb|central\s+bank|'
    r'say|says|said|mention|mentions|tweet|post|speech|address|'
    r'etfs?|ipo|merger|acquisition|'
    r'gdp|cpi|inflation|unemployment|jobs?\s+report'
    r')\b', re.I
)

_ASSET_TOKENS = {
    "BTC": ["btc", "bitcoin"],
    "ETH": ["eth", "ether", "ethereum"],
    "SOL": ["sol", "solana"],
    "XRP": ["xrp", "ripple"],
    "DOGE": ["doge", "dogecoin"],
}

def _norm_num(num: str, suffix: Optional[str] = None) -> Optional[float]:
    s = (num or '').replace(',', '').strip()
    if not s:
        return None
    try:
        val = float(s)
    except ValueError:
        m = re.match(r'^([0-9]+(?:\.[0-9]+)?)([kmbKMB])$', s)
        if not m:
            return None
        val = float(m.group(1))
        suffix = m.group(2).lower()
    mult = 1.0
    if suffix:
        ch = suffix.lower()
        if ch == 'k':
            mult = 1e3
        elif ch == 'm':
            mult = 1e6
        elif ch == 'b':
            mult = 1e9
    return val * mult

def _looks_like_date_range(text: str, a: int, b: int) -> bool:
    if a <= 31 and b <= 31 and _MONTH_RE.search(text):
        if re.search(rf"{_MONTH_RE.pattern}\s+\d{{1,2}}\s*(?:[-–—]|to)\s*\d{{1,2}}", text, re.I):
            return True
    return False

def _has_time_anchor(text: str) -> bool:
    """Detects presence of a scheduling anchor like a date or time (e.g., 'September 7, 7PM ET')."""
    months = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    dow = r"(Mon|Tue|Tues|Wed|Thu|Thur|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    time = r"(\b\d{1,2}(:\d{2})?\s*(AM|PM)\b)"
    tz = r"\b(ET|PT|CT|MT|UTC|GMT)\b"
    date_numeric = r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(/\d{2,4})?\b"
    rel_time = r"\b(today|tonight|tomorrow|this\s+(hour|day|week)|by|before|after|on)\b"
    return bool(re.search(
        months + r"\s+\d{1,2}(,\s*\d{4})?" + "|" + dow + "|" + time + r"\s*" + tz + "|" + date_numeric + "|" + rel_time,
        text, re.I
    ))

# Asset tokens used for DIRECTIONAL_PERIOD gating
_ASSET_TOKENS = re.compile(r"\b(bitcoin|btc|ethereum|eth|solana|sol|xrp|ripple|dogecoin|doge)\b", re.I)
_UP_OR_DOWN = re.compile(r"\bup\s*or\s*down\b", re.I)

def _extract_range(title: str) -> Optional[Classification]:
    # between $X and $Y
    m = re.search(r"between\s+\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*(k|m|b)?\s+(?:and|to)\s+\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*(k|m|b)?", title, re.I)
    if m:
        low = _norm_num(m.group(1), m.group(2))
        high = _norm_num(m.group(3), m.group(4))
        if low is not None and high is not None:
            lo, hi = (low, high) if low <= high else (high, low)
            return Classification(marketClass='RANGE_BUCKET', rangeLow=lo, rangeHigh=hi, matched=m.group(0))  # type: ignore
    # from/within/range $X to/through $Y
    m = re.search(r"(?:from|within|range)\s+\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*(k|m|b)?\s*(?:[-–—]|to|through|thru)\s*\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*(k|m|b)?", title, re.I)
    if m:
        low = _norm_num(m.group(1), m.group(2))
        high = _norm_num(m.group(3), m.group(4))
        if low is not None and high is not None:
            lo, hi = (low, high) if low <= high else (high, low)
            return Classification(marketClass='RANGE_BUCKET', rangeLow=lo, rangeHigh=hi, matched=m.group(0))  # type: ignore
    # X–Y with currency context (guard against dates)
    m = re.search(r"\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*(k|m|b)?\s*[-–—]\s*\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*(k|m|b)?", title, re.I)
    if m:
        n1 = _norm_num(m.group(1), m.group(2))
        n2 = _norm_num(m.group(3), m.group(4))
        if n1 is not None and n2 is not None:
            raw1 = (m.group(1) or '').replace(',', '')
            raw2 = (m.group(3) or '').replace(',', '')
            try:
                i1 = int(float(raw1)); i2 = int(float(raw2))
            except Exception:
                i1 = i2 = 1000
            had_currency = ('$' in m.group(0)) or bool(m.group(2)) or bool(m.group(4))
            if (not had_currency) and _looks_like_date_range(title, i1, i2):
                return None
            lo, hi = (n1, n2) if n1 <= n2 else (n2, n1)
            return Classification(marketClass='RANGE_BUCKET', rangeLow=lo, rangeHigh=hi, matched=m.group(0))  # type: ignore
    return None

def _extract_threshold(title: str) -> Optional[Classification]:
    re_main = re.compile("|".join([
        r"(?:greater\s+than|more\s+than|above|over)\s+\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*([kmb])?",
        r"(?:less\s+than|below|under)\s+\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*([kmb])?",
        r"(?:at\s+least|no\s+less\s+than|≥|>=)\s+\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*([kmb])?",
        r"(?:at\s+most|no\s+more\s+than|≤|<=)\s+\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*([kmb])?",
    ]), re.I)
    m = re_main.search(title)
    if m:
        g = m.groups()
        idx = next((i for i, v in enumerate(g) if v is not None and v != ''), -1)
        pair = idx // 2
        val = _norm_num(g[idx], g[idx+1] if idx+1 < len(g) else None)
        rel = ['>', '<', '>=', '<='][pair]
        return Classification(marketClass='SINGLE_THRESHOLD', relation=rel, threshold=val, matched=m.group(0))  # type: ignore
    # "$X or more/less"
    m2 = re.search(r"\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*([kmb])?\s*(?:or\s+more|or\s+higher)", title, re.I)
    if m2:
        v = _norm_num(m2.group(1), m2.group(2))
        return Classification(marketClass='SINGLE_THRESHOLD', relation='>=', threshold=v, matched=m2.group(0))  # type: ignore
    m2 = re.search(r"\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*([kmb])?\s*(?:or\s+less|or\s+lower)", title, re.I)
    if m2:
        v = _norm_num(m2.group(1), m2.group(2))
        return Classification(marketClass='SINGLE_THRESHOLD', relation='<=', threshold=v, matched=m2.group(0))  # type: ignore
    # "reach/hit $X"
    m3 = re.search(r"(?:reach|hit)\s+\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*([kmb])?", title, re.I)
    if m3:
        v = _norm_num(m3.group(1), m3.group(2))
        return Classification(marketClass='SINGLE_THRESHOLD', relation='==', threshold=v, matched=m3.group(0))  # type: ignore
    
    # NEW: verbs like "dip/fall/drop/rise/pump/dump to $X"
    # e.g., "Will SOL dip to $140 before 2026?"
    m5 = re.search(
        r"\b(dip|fall|drop|rise|pump|dump)\s+to\s+\$?\s*([0-9][\d,]*(?:\.\d+)?)\s*([kmb])?",
        title,
        re.I,
    )
    if m5:
        v = _norm_num(m5.group(2), m5.group(3))
        return Classification(marketClass='SINGLE_THRESHOLD', relation="==", threshold=v, matched=m5.group(0))  # type: ignore

    return None

def _tokens_for(asset: Optional[str]) -> list[str]:
    if not asset:
        return []
    return _ASSET_TOKENS.get(asset.upper(), [])

def _has_price_context(title: str, matched: Optional[str], asset: Optional[str]) -> bool:
    t = title.lower()
    m = (matched or "").lower()
    # Hard negatives anywhere → not price.
    if _NON_PRICE_NEARBY.search(t):
        return False
    # Strong positives.
    if 'price' in t:
        return True
    # Currency must be present AND near an asset token.
    if '$' in m or 'usd' in t:
        toks = _tokens_for(asset)
        start = t.find(m) if m else t.find('$')
        if start >= 0:
            for tok in toks:
                for mobj in re.finditer(tok, t, re.I):
                    if abs(mobj.start() - start) <= 40:
                        return True
        # Also allow patterns like "BTC $100k" or "$BTC".
        if re.search(r'\b(btc|bitcoin|eth|ether|ethereum|sol|solana|xrp|ripple|doge|dogecoin)\b\s*\$|\$\s*\b(btc|bitcoin|eth|ether|ethereum|sol|solana|xrp|ripple|doge|dogecoin)\b', t, re.I):
            return True
    return False

def _near_nonprice(title: str, matched: Optional[str]) -> bool:
    t = title
    m = matched or ''
    if '%' in m or re.search(r'\bpercent(age)?\b', m, re.I):
        return True
    if _NON_PRICE_NEARBY.search(t):
        return True
    return False

def classify_market_title(title: str, asset: Optional[str] = None) -> Classification:
    """Return a Classification dict for a Polymarket title."""
    if not isinstance(title, str) or not title.strip():
        return Classification(marketClass='UNKNOWN')  # type: ignore
    # Range
    r = _extract_range(title)
    if r:
        if _has_price_context(title, r.get('matched'), asset) and not _near_nonprice(title, r.get('matched')):
            return r
        return Classification(marketClass='UNKNOWN')  # type: ignore
    # Threshold
    t = _extract_threshold(title)
    if t:
        if _has_price_context(title, t.get('matched'), asset) and not _near_nonprice(title, t.get('matched')):
            return t
        return Classification(marketClass='UNKNOWN')  # type: ignore
    # Directional period for phrasing like "<ASSET> Up or Down - <time anchor>"
    # We intentionally **do not** require $/USD here because the phrase conveys price direction
    # and a time anchor. Still protected by the global non-price exclusion above.
    if _UP_OR_DOWN.search(title) and _ASSET_TOKENS.search(title) and _has_time_anchor(title):
        return Classification(marketClass='DIRECTIONAL_PERIOD')  # type: ignore
    return Classification(marketClass='UNKNOWN')  # type: ignore
