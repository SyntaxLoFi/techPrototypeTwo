# utils/capture_enrichment.py
from __future__ import annotations
from typing import Dict, Any, Optional, Callable, Sequence
from utils.instrument_capture import CAPTURE_ENABLED, ensure_exec, record_option_leg, record_perp_leg, record_polymarket, BASE_UNIT

def enrich_from_required_options(opp: Dict[str, Any],
                                 *,
                                 symbol: str,
                                 get_quote: Callable[[str,str,float,str], Optional[Dict[str,Any]]],
                                 get_greeks: Optional[Callable[[str,str,float,str], Optional[Dict[str,Any]]]] = None,
                                 fee_bps: float = 3.0,
                                 slippage_bps: float = 10.0) -> None:
    """
    If the strategy wrote opp['detailed_strategy']['required_options'],
    populate opp['execution_details']['options_legs'] with quotes/greeks.
    """
    if not CAPTURE_ENABLED:
        return
    ds = opp.get("detailed_strategy") or {}
    legs = list(ds.get("required_options") or [])
    if not legs:
        return
    for leg in legs:
        expiry = str(leg.get("expiry") or (opp.get("lyra") or {}).get("expiry") or "")
        strike = float(leg.get("strike"))
        type_  = (leg.get("type") or leg.get("option_type") or "").upper()  # CALL/PUT
        action = (leg.get("action") or leg.get("side") or "").upper()       # BUY/SELL
        contracts = float(leg.get("contracts") or leg.get("qty") or 1.0)
        q = get_quote(symbol, expiry, strike, type_) if get_quote else None
        g = get_greeks(symbol, expiry, strike, type_) if get_greeks else None
        record_option_leg(opp, venue="Lyra", symbol=symbol, type_=type_, action=action,
                          contracts=contracts, weight=float(leg.get("weight", 1.0)),
                          expiry=expiry, strike=strike, quote=q, greeks=g,
                          fee_bps=fee_bps, slippage_bps=slippage_bps,
                          instrument_id=(q or {}).get("instrument_id"))

def enrich_polymarket_ticket(opp: Dict[str, Any], *, pm_fee_bps: float = 100.0) -> None:
    """Write a unit-sized Polymarket ticket into execution_details."""
    from utils.instrument_capture import record_polymarket
    pm = (opp.get("polymarket") or {})
    price = pm.get("yes_price")
    if price:
        record_polymarket(opp, side="YES", price=float(price),
                          qty=(BASE_UNIT/float(price)), notional=BASE_UNIT,
                          fee_bps=float(pm_fee_bps),
                          market_id=pm.get("market_id"), question=pm.get("question"))

def attach_perp_snapshot(opp: Dict[str, Any],
                         *,
                         symbol: str,
                         get_perp_snapshot: Optional[Callable[[str], Optional[Dict[str,Any]]]] = None,
                         hedge_ratio: Optional[float] = None,
                         leverage: Optional[float] = None,
                         fee_bps: Optional[float] = None,
                         spread_bps: Optional[float] = None) -> None:
    """
    If your strategy uses a perp hedge but didn't save legs, snapshot the current perp.
    """
    if not CAPTURE_ENABLED or not get_perp_snapshot:
        return
    snap = get_perp_snapshot(symbol) or {}
    record_perp_leg(opp, venue=snap.get("venue","Perps"), symbol=symbol,
                    side=snap.get("side","SHORT" if (hedge_ratio or 0) > 0 else "LONG"),
                    qty=snap.get("qty", hedge_ratio or 0.0),
                    entry_price=snap.get("entry_price"), index_price=snap.get("index"),
                    mark_price=snap.get("mark_price"), leverage=leverage or snap.get("leverage"),
                    fee_bps=fee_bps or snap.get("fee_bps"), spread_bps=spread_bps or snap.get("spread_bps"),
                    funding_rate=snap.get("funding_rate"), funding_interval=snap.get("funding_interval"),
                    hedge_ratio=hedge_ratio, instrument_id=snap.get("instrument_id"))