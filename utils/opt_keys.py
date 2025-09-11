from typing import Mapping, Optional

# Canonicalize option types and validate quotes in one place.
# This is intentionally dependency-light: pure stdlib and typing.

def normalize_type(value: Optional[str]) -> Optional[str]:
    """
    Map many vendor/right/cp spellings to {'call','put'}.
      Accepts: 'call','CALL','c','C' -> 'call'; 'put','PUT','p','P' -> 'put'
      Returns None if the input cannot be normalized.
    """
    t = (value or "").strip().lower()
    if t in ("c", "call"):
        return "call"
    if t in ("p", "put"):
        return "put"
    return None

def is_valid_quote(row: Mapping) -> bool:
    """
    A quote is valid if ask>0 and 0<=bid<=ask (floats parseable).
    Missing or non-numeric bid/ask => invalid.
    """
    try:
        bid = float(row.get("bid", 0.0) or 0.0)
        ask = float(row.get("ask", 0.0) or 0.0)
    except Exception:
        return False
    return ask > 0.0 and bid >= 0.0 and bid <= ask

def resolve_opt_key(strike_entry: Mapping, is_call: bool) -> str:
    """
    Given one strike's dictionary (e.g., chain[strike]), return the
    subkey containing call/put quotes:
      — try common variants: 'C','call','CALL','calls' or 'P','put','PUT','puts'
      — fall back to 'C'/'P' if nothing matches so callers don't blow up
    """
    if not isinstance(strike_entry, Mapping):
        return "C" if is_call else "P"
    keys = tuple(strike_entry.keys())
    if is_call:
        for k in ("C", "call", "CALL", "calls", "Calls", "CALLS"):
            if k in keys:
                return k
        return "C"
    else:
        for k in ("P", "put", "PUT", "puts", "Puts", "PUTS"):
            if k in keys:
                return k
        return "P"