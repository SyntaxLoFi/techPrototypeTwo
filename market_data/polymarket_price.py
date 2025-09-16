"""Polymarket pricing helpers.

This module provides a single responsibility: given a *Gamma* market payload
(dict) or a thin projection of it, return a best-estimate **YES** price in [0, 1].

Why this exists
---------------
We observed systematic 0.999 YES prices in our exports for otherwise liquid markets.
That pattern is consistent with a fallback that clamps missing values to ~1.0.
To prevent this class of error entirely, this helper:

* Never defaults a missing price to 1.0 (or any other constant);
* Understands Gamma's occasional JSON-string-encoded arrays (e.g. ``outcomePrices``);
* Normalizes percentages (0–100) vs probabilities (0–1);
* Prefers reliable sources in a documented order.

Precedence (documented strategy)
--------------------------------
1) If both ``outcomes`` and ``outcomePrices`` are present, parse them and return
   the price aligned to the 'Yes' outcome (case-insensitive).
2) Else, if quotes are present, return the **mid** of ``bestBid`` and ``bestAsk``.
3) Else, if ``lastTradePrice`` is present, return it (normalized).
4) Else, return ``None`` and let the caller decide how to handle missing data.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
import json


def _coerce_prob(x: Any) -> Optional[float]:
    """Coerce a value into a probability in [0, 1].

    Accepts strings or numbers. If a value is in [0, 100], assume it is a *percent*
    when > 1 and <= 100 and divide by 100. Reject (return None) for NaNs or
    obviously invalid inputs.
    """
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:  # NaN guard
        return None
    if v > 1.0 and v <= 100.0:
        v = v / 100.0
    if v < 0.0 or v > 1.0:
        return None
    return v


def _parse_outcome_prices(raw: Any) -> Optional[Iterable[float]]:
    """Parse ``outcomePrices`` which can be a list OR a JSON-encoded string.

    Returns an iterable of floats in [0,1] or None.
    """
    if raw is None:
        return None
    data = raw
    if isinstance(raw, str):
        raw = raw.strip()
        # Some Gamma responses (esp. via caches) encode arrays as strings.
        if raw.startswith("[") and raw.endswith("]"):
            try:
                data = json.loads(raw)
            except Exception:
                return None
        else:
            # Single scalar in a string
            v = _coerce_prob(raw)
            return [v] if v is not None else None
    if isinstance(data, (list, tuple)):
        vals = [_coerce_prob(x) for x in data]
        if any(v is None for v in vals):
            return None
        return vals  # type: ignore[return-value]
    return None


def derive_yes_price_from_gamma(market: Dict[str, Any]) -> Optional[float]:
    """Return a best-estimate YES price for a binary Gamma market.

    Parameters
    ----------
    market : Dict[str, Any]
        A Gamma *market* object (or a dict containing at least the relevant fields
        used below).

    Returns
    -------
    Optional[float]
        YES probability in [0, 1], or ``None`` if not derivable.
    """
    # 1) Try outcomes + outcomePrices first
    outcomes = market.get("outcomes")  # e.g., ["No", "Yes"] or ["Yes", "No"]
    prices_raw = market.get("outcomePrices")
    prices = _parse_outcome_prices(prices_raw) if prices_raw is not None else None
    if outcomes and prices:
        # Find YES index (case-insensitive, tolerate 'YES', 'Yes', etc.)
        try:
            yes_idx = [str(o).strip().lower() for o in outcomes].index("yes")
            yes_price = list(prices)[yes_idx]
            if yes_price is not None:
                return yes_price
        except ValueError:
            # If 'Yes' not present, but we have 2 outcomes assume index 1 is Yes
            # when the first is 'No' (common ordering).
            labels = [str(o).strip().lower() for o in outcomes]
            if len(labels) == 2 and labels[0] == "no" and prices is not None:
                vals = list(prices)
                if len(vals) >= 2:
                    return vals[1]

    # 2) Fall back to quotes mid if present
    bid = _coerce_prob(market.get("bestBid"))
    ask = _coerce_prob(market.get("bestAsk"))
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2.0
    # Accept single-sided quote if only one exists (rare)
    if bid is not None and ask is None:
        return bid
    if ask is not None and bid is None:
        return ask

    # 3) Last trade as a final fallback
    last = _coerce_prob(market.get("lastTradePrice"))
    if last is not None:
        return last

    # 4) Give up — DO NOT clamp to ~1.0
    return None


__all__ = ["derive_yes_price_from_gamma"]