# Canonical option family keys used by the options chain JSON:
# The chain nests under each strike as {..., "call": {...}, "put": {...}}
# See options_chain_structure_summary.json -> "option_type_keys": ["call", "put"]
# (We normalize any user/venue variety like C/P, CALL/PUT to these.)
from __future__ import annotations
from typing import Any, Mapping, Optional

CANONICAL_CALL_KEY: str = "call"
CANONICAL_PUT_KEY: str = "put"

_NORM = {
    "c": "call", "call": "call", "calls": "call", "CALL": "call", "C": "call",
    "p": "put",  "put":  "put",  "puts":  "put",  "PUT":  "put",  "P": "put",
}

def normalize_opt_type(x: Optional[str]) -> Optional[str]:
    """Return 'call' or 'put' for many common representations; None if unknown."""
    if x is None:
        return None
    return _NORM.get(str(x).strip(), None)

def family_for_is_above(is_above: bool) -> str:
    """Choose 'call' if the PM contract is an ABOVE type, else 'put'."""
    return CANONICAL_CALL_KEY if bool(is_above) else CANONICAL_PUT_KEY

# alias used by scripts
def key_for_is_above(is_above: bool) -> str:
    return family_for_is_above(is_above)

def is_call_key(k: Optional[str]) -> bool:
    return normalize_opt_type(k) == "call"

def is_put_key(k: Optional[str]) -> bool:
    return normalize_opt_type(k) == "put"

def choose_family(*, family: Optional[str] = None, is_above: Optional[bool] = None) -> str:
    """
    Choose a canonical family key given either:
      - family='call'/'put' (or any synonym), or
      - is_above=True/False.
    """
    if family is not None:
        f = normalize_opt_type(family)
        if f:
            return f
    if is_above is not None:
        return family_for_is_above(is_above)
    raise ValueError("choose_family: provide either family or is_above")

def chain_slice(chain_for_expiry: Mapping[str, Any],
                strike: Any,
                *,
                family: Optional[str] = None,
                is_above: Optional[bool] = None) -> Mapping[str, Any]:
    """
    Return the dict for a specific strike+family from the chain:
        chain_for_expiry[strike_str][family]
    - strike is normalized to a float string (the chain uses strings like '3200.0')
    """
    if strike is None:
        return {}
    try:
        strike_key = str(float(strike))
    except Exception:
        strike_key = str(strike)
    fam = choose_family(family=family, is_above=is_above)
    return (chain_for_expiry.get(strike_key) or {}).get(fam) or {}

__all__ = [
    "CANONICAL_CALL_KEY", "CANONICAL_PUT_KEY",
    "normalize_opt_type", "family_for_is_above", "key_for_is_above",
    "is_call_key", "is_put_key", "choose_family", "chain_slice",
]