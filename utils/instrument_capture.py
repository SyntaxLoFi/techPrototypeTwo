# utils/instrument_capture.py
from __future__ import annotations
import os, datetime
from typing import Dict, Any, Optional

# Try to get from config first, then fall back to environment variables
try:
    from config_loader import get_config
    cfg = get_config()
    CAPTURE_ENABLED = bool(cfg.execution.capture_instruments)
    BASE_UNIT = float(cfg.execution.default_position_size)
except Exception:
    # Fall back to environment variables
    CAPTURE_ENABLED = os.getenv("CAPTURE_INSTRUMENTS", "0").lower() in ("1","true","yes","on")
    BASE_UNIT = float(os.getenv("DEFAULT_POSITION_SIZE", os.getenv("POSITION_BASE_UNIT", "100.0")))

def get_base_unit(default: float = None) -> float:
    """
    Helper for modules that prefer a function accessor.
    If config/env are unavailable, return provided default (or current BASE_UNIT).
    """
    if default is None:
        return BASE_UNIT
    return float(os.getenv("DEFAULT_POSITION_SIZE", os.getenv("POSITION_BASE_UNIT", str(default))))

def _ts() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def ensure_exec(opp: Dict[str, Any]) -> Dict[str, Any]:
    ed = opp.setdefault("execution_details", {})
    ed.setdefault("polymarket", {})
    ed.setdefault("options_legs", [])
    ed.setdefault("perp_legs", [])
    # keep a capture header for provenance
    meta = ed.setdefault("_meta", {})
    meta.setdefault("captured_at", _ts())
    meta.setdefault("base_unit", BASE_UNIT)
    return ed

def record_polymarket(opp: Dict[str, Any], *, side: str, price: float,
                      qty: float, notional: float, fee_bps: float = 100.0,
                      market_id: Optional[str] = None, question: Optional[str] = None):
    if not CAPTURE_ENABLED: return
    ed = ensure_exec(opp)
    pm = ed["polymarket"]
    pm.update({
        "market_id": market_id or opp.get("polymarket", {}).get("market_id"),
        "question": question or opp.get("polymarket", {}).get("question"),
        "side": side.upper(),
        "price": float(price),
        "qty": float(qty),
        "notional": float(notional),
        "fee_bps": float(fee_bps),
        "timestamp": _ts(),
        "shares_per_$1k": BASE_UNIT / float(price) if price else None
    })

def record_option_leg(opp: Dict[str, Any], *,
                      venue: str, symbol: str, type_: str, action: str,
                      contracts: float, expiry: str, strike: float, weight: float = 1.0,
                      quote: Optional[Dict[str,Any]] = None, greeks: Optional[Dict[str,Any]] = None,
                      fee_bps: float = 3.0, slippage_bps: float = 10.0,
                      instrument_id: Optional[str] = None, contract_size: float = 1.0,
                      source: str = "OptionsChainCollector"):
    if not CAPTURE_ENABLED: return
    ed = ensure_exec(opp)
    q = quote or {}
    g = greeks or {}
    bid, ask = q.get("bid"), q.get("ask")
    mid = q.get("mid")
    if mid is None and bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
    leg = {
        "venue": venue,
        "symbol": symbol,
        "type": type_.upper(),         # CALL/PUT
        "action": action.upper(),      # BUY/SELL
        "contracts": float(contracts),
        "weight": float(weight),
        "expiry": expiry,
        "strike": float(strike),
        "contract_size": float(contract_size),
        # quotes
        "bid": bid, "ask": ask,
        "mid": mid, "mark": q.get("mark", mid),
        "iv": (q.get("iv") or q.get("quote_iv") or
               ((lambda b,a: (float(b)+float(a))/2.0 if (b is not None and a is not None) else None)(q.get("iv_bid"), q.get("iv_ask")))),
        "quote_time": q.get("timestamp") or q.get("quote_time") or _ts(),
        "source": source,
        # greeks (flat for the leg, not position-sized)
        "delta": g.get("delta"), "gamma": g.get("gamma"),
        "vega": g.get("vega"), "theta": g.get("theta"), "rho": g.get("rho"),
        # costs/ids
        "fee_bps": float(fee_bps), "slippage_bps": float(slippage_bps),
        "instrument_id": instrument_id or q.get("instrument_id"),
    }
    ed["options_legs"].append(leg)

def record_perp_leg(opp: Dict[str, Any], *, venue: str, symbol: str, side: str, qty: float,
                    entry_price: float = None, index_price: float = None, mark_price: float = None,
                    leverage: float = None, fee_bps: float = None, spread_bps: float = None,
                    funding_rate: float = None, funding_interval: str = None, hedge_ratio: float = None,
                    instrument_id: str = None, contract_size: float = 1.0, source: str = "PerpsDataCollector"):
    if not CAPTURE_ENABLED: return
    ed = ensure_exec(opp)
    ed["perp_legs"].append({
        "venue": venue, "symbol": symbol, "side": side.upper(),
        "qty": float(qty), "contract_size": float(contract_size),
        "entry_price": entry_price, "index_price": index_price, "mark_price": mark_price,
        "leverage": leverage, "fee_bps": fee_bps, "spread_bps": spread_bps,
        "funding_rate": funding_rate, "funding_interval": funding_interval,
        "hedge_ratio": hedge_ratio, "instrument_id": instrument_id,
        "quote_time": _ts(), "source": source
    })