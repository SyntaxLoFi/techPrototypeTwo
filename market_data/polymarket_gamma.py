# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional

# Core pricing helper (robust outcome/quotes parsing; never default to ~1.0)
from market_data.polymarket_price import derive_yes_price_from_gamma


def normalize_gamma_market(market: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize a single Polymarket gamma market for use in volatility calcs.
    This version relies on polymarket_price.py::derive_yes_price_from_gamma()
    to robustly parse outcomes/quotes without hardcoded 0.999 defaults.
    """
    if not market:
        return None

    out: Dict[str, Any] = {
        "pm_market_id": market.get("id") or market.get("marketId"),
        "title": market.get("question") or market.get("title"),
        # prices filled below (via robust core helper)
        "source": "polymarket",
    }

    # Core path: derive YES via outcomes→mid(bid,ask)→last, with %→prob normalization
    yes_price: Optional[float] = derive_yes_price_from_gamma(market)

    # Never fabricate; surface truthfully
    out["yes_price"] = yes_price
    out["no_price"] = (1.0 - yes_price) if yes_price is not None else None
    out["price_status"] = "ok" if yes_price is not None else "missing"

    # ... any remaining normalization (timestamps, ticker, etc.) stays the same
    return out