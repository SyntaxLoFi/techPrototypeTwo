# utils/opportunity_postprocess.py
import os
import datetime
from typing import Dict, Any, List, Optional
from utils.instrument_capture import BASE_UNIT

try:
    from config_loader import get_config
    CAPTURE_ENABLED = bool(get_config().execution.capture_instruments)
except Exception:
    CAPTURE_ENABLED = os.getenv("CAPTURE_INSTRUMENTS", "0").lower() in ("1","true","yes","on")
SCHEMA_VERSION = os.getenv("SCHEMA_VERSION", "5.1-instruments")

def _ts():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def build_option_leg(venue: str, symbol: str, otype: str, action: str, contracts: float, weight: float,
                     expiry: str, strike: float, quote: Optional[Dict[str, Any]]=None,
                     greeks: Optional[Dict[str, Any]]=None, fees_bps: float=3.0, slippage_bps: float=10.0,
                     instrument_id: Optional[str]=None, contract_size: float=1.0, source: str="OptionsChainCollector") -> Dict[str, Any]:
    quote = quote or {}
    greeks = greeks or {}
    bid = quote.get("bid"); ask = quote.get("ask")
    mid = quote.get("mid")
    if mid is None:
        if (bid is not None and ask is not None and float(bid) > 0 and float(ask) > 0):
            mid = (float(bid) + float(ask)) / 2.0
        else:
            mid = bid if (bid or 0) > 0 else ask if (ask or 0) > 0 else quote.get("mark")
    mark = quote.get("mark", mid)
    return {
        "venue": venue,
        "symbol": symbol,
        "type": otype.upper(),
        "action": action.upper(),
        "contracts": float(contracts),
        "weight": float(weight),
        "expiry": expiry,
        "strike": float(strike),
        "contract_size": float(contract_size),
        "bid": bid, 
        "ask": ask, 
        "mid": mid,
        "mark": mark,
        "iv": quote.get("iv"),
        "delta": greeks.get("delta"), 
        "gamma": greeks.get("gamma"),
        "vega": greeks.get("vega"), 
        "theta": greeks.get("theta"),
        "fee_bps": float(fees_bps),
        "slippage_bps": float(slippage_bps),
        "instrument_id": instrument_id or quote.get("instrument_id"),
        "quote_time": quote.get("timestamp") or _ts(),
        "source": source
    }

def build_perp_leg(venue: str, symbol: str, side: str, qty: float, entry_price: Optional[float],
                   index_price: Optional[float], mark_price: Optional[float], leverage: Optional[float],
                   fee_bps: Optional[float], spread_bps: Optional[float],
                   funding_rate: Optional[float], funding_interval: Optional[str], hedge_ratio: Optional[float],
                   instrument_id: Optional[str]=None, contract_size: float=1.0,
                   source: str="PerpsDataCollector", quote_time: Optional[str]=None) -> Dict[str, Any]:
    return {
        "venue": venue, 
        "symbol": symbol, 
        "side": side.upper(),
        "qty": float(qty), 
        "contract_size": float(contract_size),
        "entry_price": float(entry_price) if entry_price is not None else None,
        "index_price": index_price, 
        "mark_price": mark_price,
        "leverage": leverage, 
        "fee_bps": fee_bps, 
        "spread_bps": spread_bps,
        "funding_rate": funding_rate, 
        "funding_interval": funding_interval,
        "hedge_ratio": hedge_ratio, 
        "instrument_id": instrument_id,
        "quote_time": quote_time or _ts(), 
        "source": source
    }

def _guess_symbol(opp: Dict[str, Any]) -> str:
    return opp.get("currency") or opp.get("symbol") or "BTC"

def _ensure_ed(opp: Dict[str, Any]) -> Dict[str, Any]:
    ed = opp.setdefault("execution_details", {})
    ed.setdefault("options_legs", [])
    ed.setdefault("perp_legs", [])
    return ed

def populate_execution_details(opportunities: List[Dict[str, Any]],
                               options_collector=None,
                               perps_collector=None,
                               enabled: bool | None = None) -> List[Dict[str, Any]]:
    """
    Post-process opportunities to fill opp['execution_details'] from existing fields.
    Optionally enrich quotes/greeks via collectors if provided.
    """
    # Respect explicit enable/disable override; default to CAPTURE_ENABLED
    if (enabled is False) or (enabled is None and not CAPTURE_ENABLED):
        return opportunities

    for opp in opportunities:
        ed = _ensure_ed(opp)

        # Mark schema version
        opp.setdefault("schema_version", SCHEMA_VERSION)

        # ---- Polymarket ticket (only if not already set) ----
        pm = opp.get("execution_details", {}).get("polymarket", {}) or {}
        if not pm:
            pm_block = opp.get("polymarket", {}) or {}
            price = pm_block.get("yes_price") or pm_block.get("price")
            if not price:
                price = (opp.get("probabilities") or {}).get("pm_implied")
            if price:
                side = str(opp.get("pm_side", pm_block.get("side", "YES"))).upper()
                # choose correct PM side price
                pm_yes = pm_block.get("yes_price")
                pm_no  = pm_block.get("no_price")
                side_price = float(price)
                if side == "NO" and pm_no is not None:
                    side_price = float(pm_no)
                if side == "YES" and pm_yes is not None:
                    side_price = float(pm_yes)
                base_unit = float(BASE_UNIT)
                opp.setdefault("execution_details", {})["polymarket"] = {
                    "market_id": pm_block.get("market_id"),
                    "question": pm_block.get("question"),
                    "side": side,
                    "price": float(side_price),
                    "qty": float(base_unit / max(1e-12, float(side_price))),  # respect config/env
                    "notional": base_unit,
                    "fee_bps": 100.0,
                    "timestamp": _ts(),
                }

        # ---- Options legs (from detailed_strategy or lyra hints) ----
        already = len(ed.get("options_legs") or [])
        if already == 0:
            symbol = _guess_symbol(opp)
            lyra = opp.get("lyra", {}) or {}
            dstrat = opp.get("detailed_strategy", {}) or {}
            req = dstrat.get("required_options")
            expiry = (opp.get("option_expiry") or lyra.get("expiry") or
                      (opp.get("options_details") or {}).get("expiry"))

            # Preferred: exact legs from required_options
            if isinstance(req, list) and req:
                for leg in req:
                    strike = float(leg.get("strike"))
                    otype = (leg.get("type") or "CALL").upper()
                    action = (leg.get("action") or ("BUY" if float(leg.get("weight",1)) >= 0 else "SELL")).upper()
                    contracts = float(leg.get("contracts", leg.get("weight", 1.0)))
                    quote, greeks = None, None
                    # Handle both single collector and dict of collectors by currency
                    collector = options_collector
                    if isinstance(options_collector, dict):
                        collector = options_collector.get(symbol)
                    if collector and hasattr(collector, 'get_quote'):
                        try:
                            quote = collector.get_quote(symbol=symbol, expiry=expiry, strike=strike, otype=otype)
                            greeks = collector.get_greeks(symbol=symbol, expiry=expiry, strike=strike, otype=otype) if hasattr(collector, 'get_greeks') else None
                        except Exception:
                            pass
                    ed["options_legs"].append(
                        build_option_leg("Lyra", symbol, otype, action, contracts, float(leg.get("weight", contracts)),
                                         expiry, strike, quote=quote, greeks=greeks)
                    )
            else:
                # Fallback: infer a vertical spread from lyra.strikes
                strikes = lyra.get("strikes")
                if strikes and len(strikes) == 2:
                    low, high = sorted([float(strikes[0]), float(strikes[1])])
                    question = (opp.get("polymarket", {}).get("question") or "").lower()
                    otype = "CALL" if any(w in question for w in ["reach", "hit", "above", "over", "exceed"]) else "PUT"
                    # assume debit spread by default (buy low / sell high)
                    for (strike, action) in ((low, "BUY"), (high, "SELL")):
                        quote, greeks = None, None
                        # Handle both single collector and dict of collectors by currency
                        collector = options_collector
                        if isinstance(options_collector, dict):
                            collector = options_collector.get(symbol)
                        if collector and hasattr(collector, 'get_quote'):
                            try:
                                quote = collector.get_quote(symbol=symbol, expiry=expiry, strike=strike, otype=otype)
                                greeks = collector.get_greeks(symbol=symbol, expiry=expiry, strike=strike, otype=otype) if hasattr(collector, 'get_greeks') else None
                            except Exception:
                                pass
                        ed["options_legs"].append(
                            build_option_leg("Lyra", symbol, otype, action, 1.0, 1.0, expiry, strike,
                                             quote=quote, greeks=greeks)
                        )

        # ---- Perp legs (if hedged via perps) ----
        if len(ed.get("perp_legs") or []) == 0:
            perp_hint = opp.get("perp") or opp.get("perps") or {}
            if perp_hint:
                try:
                    ed["perp_legs"].append(
                        build_perp_leg(
                            venue=perp_hint.get("venue","Lyra Perps"),
                            symbol=perp_hint.get("symbol","BTC-PERP"),
                            side=perp_hint.get("side","SHORT"),
                            qty=perp_hint.get("qty", 1.0),
                            entry_price=perp_hint.get("entry_price"),
                            index_price=perp_hint.get("index_price"),
                            mark_price=perp_hint.get("mark_price"),
                            leverage=perp_hint.get("leverage"),
                            fee_bps=perp_hint.get("fee_bps"),
                            spread_bps=perp_hint.get("spread_bps"),
                            funding_rate=perp_hint.get("funding_rate"),
                            funding_interval=perp_hint.get("funding_interval", "8h"),
                            hedge_ratio=perp_hint.get("hedge_ratio"),
                            instrument_id=perp_hint.get("instrument_id"),
                            contract_size=perp_hint.get("contract_size", 1.0)
                        )
                    )
                except Exception:
                    # never fail the pipeline
                    pass

    return opportunities