# -*- coding: utf-8 -*-
"""
Classifier and tagger for Polymarket crypto price markets.

This is an adapter that uses the local polymarket_classifier module
while maintaining backward compatibility with the existing API.
"""
from __future__ import annotations

import re
import json
from typing import Any, Dict, Optional, List

# Import the new classifier
try:
    from .polymarket_classifier import classify_market_title  # local, preferred
except Exception:
    try:
        from polymarket_classifier import classify_market_title  # fallback when package context missing
    except Exception:
        from polymarket.market_classifier import classify_market_title  # ultimate fallback for older envs

# -------------------- Asset patterns (maintained for compatibility) --------------------
ASSET_SYNONYMS: Dict[str, List[str]] = {
    "BTC": [r"\bBTC\b", r"\bBITCOIN\b"],
    "ETH": [r"\bETH\b", r"\bETHEREUM\b"],
    "SOL": [r"\bSOL\b", r"\bSOLANA\b"],
    "XRP": [r"\bXRP\b", r"\bRIPPLE\b"],
    "DOGE": [r"\bDOGE\b", r"\bDOGECOIN\b"],
}

ASSET_ORDER = ["BTC", "ETH", "SOL", "XRP", "DOGE"]

def _compile_asset_regexes() -> Dict[str, List[re.Pattern]]:
    out: Dict[str, List[re.Pattern]] = {}
    for a, pats in ASSET_SYNONYMS.items():
        out[a] = [re.compile(p, re.IGNORECASE) for p in pats]
    return out

ASSET_RE = _compile_asset_regexes()

# -------------------- Public helpers (maintained for compatibility) --------------------
def is_crypto_text(text: str) -> bool:
    up = (text or "").upper()
    for a in ASSET_ORDER:
        for rx in ASSET_RE[a]:
            if rx.search(up):
                return True
    return False

def find_asset(text: str) -> Optional[str]:
    up = (text or "").upper()
    for a in ASSET_ORDER:
        for rx in ASSET_RE[a]:
            if rx.search(up):
                return a
    return None

def detect_binance_symbol(text: str, asset: Optional[str]) -> Optional[str]:
    """
    Returns e.g. 'BTCUSDT' if the description references Binance + USDT.
    """
    if not text:
        return None
    up = text.upper()
    BINANCE_USDT = re.compile(r"\bBINANCE\b.*\bUSDT\b", re.IGNORECASE | re.DOTALL)
    PAIR_USDT = re.compile(r"\b([A-Z]{3,5})USDT\b")
    
    if not BINANCE_USDT.search(up):
        return None
    m = PAIR_USDT.search(up)
    if m:
        return f"{m.group(1)}USDT"
    if asset:
        return f"{asset}USDT"
    return None

def extract_token_ids(market: Dict[str, Any]) -> Optional[Dict[str, str]]:
    for key in ("clobTokenIds", "clob_token_ids", "tokenIds"):
        obj = market.get(key)
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:
                obj = None
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if v is None:
                    continue
                out[str(k).upper()] = str(v)
            return out or None
        if isinstance(obj, list):
            out = {}
            for item in obj:
                if isinstance(item, dict) and item.get("token_id") and item.get("outcome"):
                    out[str(item["outcome"]).upper()] = str(item["token_id"])
                elif isinstance(item, str):
                    idx = len(out)
                    out["YES" if idx == 0 else "NO"] = item
            return out or None
    return None

# -------------------- Strategy helpers --------------------
def _assign_strategy_categories(asset: Optional[str]) -> Dict[str, bool]:
    """
    Core policy (asset-level):
      - BTC/ETH: eligible for options + hybrid (and still perpetuals)
      - SOL/XRP/DOGE: perpetuals only
    """
    cats = {"perpetuals": True, "options": False, "hybrid": False}
    if asset in ("BTC", "ETH"):
        cats["options"] = True
        cats["hybrid"] = True
    return cats

def _assign_strategy_eligibility(asset: Optional[str], market_class: str) -> Dict[str, Dict[str, bool]]:
    """
    Sub-strategy eligibility flags.
    """
    elig: Dict[str, Dict[str, bool]] = {
        "options": {"variance_swap": False},
        "perpetuals": {},
        "hybrid": {},
    }
    if asset in ("BTC", "ETH") and market_class == "SINGLE_THRESHOLD":
        elig["options"]["variance_swap"] = True
    return elig

def _strategies_for_class(market_class: str) -> List[str]:
    if market_class == "SINGLE_THRESHOLD":
        return ["variance_swap", "perpetuals", "options_vanilla"]
    if market_class == "RANGE_BUCKET":
        return ["perpetuals", "options_spreads"]
    if market_class == "DIRECTIONAL_PERIOD":
        return ["perpetuals"]
    return []

def _filter_strategies_by_asset(asset: Optional[str], strategies: List[str]) -> List[str]:
    """
    Enforce asset-level availability:
      - BTC/ETH: keep as-is
      - SOL/XRP/DOGE: force to ['perpetuals']
    """
    if asset in ("SOL", "XRP", "DOGE"):
        return ["perpetuals"]
    # Deduplicate while preserving order
    out: List[str] = []
    seen = set()
    for s in strategies:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _make_base_row(event: Dict[str, Any], market: Dict[str, Any]) -> Dict[str, Any]:
    e_id = event.get("id") or event.get("_id") or event.get("eventId") or event.get("event_id")
    m_id = market.get("id") or market.get("_id") or market.get("market_id") or market.get("questionID")
    row = {
        "eventId": e_id,
        "eventTitle": event.get("title") or event.get("name") or "",
        "eventDescription": event.get("description") or "",
        "marketId": m_id,
        "marketSlug": market.get("slug") or "",
        "question": market.get("question") or market.get("title") or "",
        "marketDescription": market.get("description") or "",
        "endDate": market.get("endDate") or market.get("closeTime") or market.get("endTime") or "",
        "tokenIds": extract_token_ids(market),
    }
    return row

def classify_market(event: Dict[str, Any], market: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point: classify a market using the new classifier
    and adapt the results to the existing format.
    """
    row = _make_base_row(event, market)
    
    # Combine text for classification
    combined_text = " ".join([
        row["eventTitle"] or "",
        row["question"] or "",
        row["marketDescription"] or "",
        row["eventDescription"] or "",
    ])
    
    # Find asset
    asset = find_asset(combined_text)
    row["asset"] = asset
    
    # Use new classifier
    classification = classify_market_title(combined_text, asset)
    
    # Map classification results to existing format
    row["marketClass"] = classification["marketClass"]
    
    # Binance + USDT detector
    bsym = detect_binance_symbol(f"{row['marketDescription']} {row['eventDescription']}", asset)
    row["binanceSymbol"] = bsym
    
    # Map specific fields based on market class
    if classification["marketClass"] == "SINGLE_THRESHOLD":
        row["relation"] = classification.get("relation", ">=")
        row["threshold"] = classification.get("threshold", 0.0)
        # For downstream options hedger
        row["strike_price"] = float(classification.get("threshold", 0.0))
        row["is_above"] = row["relation"] in (">=", ">", "==")
    elif classification["marketClass"] == "RANGE_BUCKET":
        row["rangeLow"] = classification.get("rangeLow", 0.0)
        row["rangeHigh"] = classification.get("rangeHigh", 0.0)
        row["openSide"] = None  # New classifier doesn't support open-ended ranges
    
    # Assign strategies
    base_strategies = _strategies_for_class(row["marketClass"])
    row["strategies"] = _filter_strategies_by_asset(asset, base_strategies)
    # Parent categories (options / perpetuals / hybrid)
    row["strategyCategories"] = _assign_strategy_categories(asset)
    # Child-level eligibility (e.g., options → variance)
    row["strategyEligibility"] = _assign_strategy_eligibility(asset, row["marketClass"])
    # Hierarchical tags (e.g., "options.variance") for routing
    def _to_strategy_tags(elig: Dict[str, Dict[str, bool]]) -> List[str]:
        tags: List[str] = []
        for parent, children in (elig or {}).items():
            for child, ok in (children or {}).items():
                if ok:
                    tags.append(f"{parent}.{child}")
        return tags
    row["strategyTags"] = _to_strategy_tags(row["strategyEligibility"])
    return row