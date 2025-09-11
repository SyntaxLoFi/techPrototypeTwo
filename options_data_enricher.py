"""
Options Data Enricher - Adds bid/ask prices to options data

This module enriches options instrument data with current bid/ask prices
from the orderbook handler, making the data ready for strategy evaluation.
"""

import logging
import re
from typing import Dict, List, Optional
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)


class OptionsDataEnricher:
    """Enriches options data with current market prices"""
    
    def __init__(self, orderbook_handler=None):
        self.orderbook_handler = orderbook_handler
        self.logger = logging.getLogger('OptionsDataEnricher')
        
    def enrich_options_with_prices(
        self, 
        options_data: List[Dict],
        use_estimated_prices: bool = True,
        current_spot: float = None
    ) -> List[Dict]:
        """
        Enrich options data with bid/ask prices
        
        Args:
            options_data: List of options instruments
            use_estimated_prices: Whether to use estimated prices if orderbook unavailable
            current_spot: Current spot price for estimation
            
        Returns:
            List of options with bid/ask prices added
        """
        self.logger.info(f"[ENRICH] Starting enrichment of {len(options_data)} options")
        enriched_options = []
        skipped_unknown_type = 0
        
        for option in options_data:
            enriched_option = option.copy()
            
            # --- Normalize fields (type/strike/expiry) BEFORE doing pricing ---
            norm = self._normalize_option_fields(enriched_option)
            if norm is None:
                self.logger.warning(
                    f"[ENRICH] Skipping option with unknown type: "
                    f"{option.get('instrument_name') or option.get('symbol') or 'unknown'}"
                )
                skipped_unknown_type += 1
                continue
            enriched_option.update(norm)  # ensures 'type','strike','expiry_date' are present
            
            # Try to get real orderbook data
            instrument_name = option.get('instrument_name')
            if self.orderbook_handler and instrument_name:
                orderbook = self.orderbook_handler.get_latest_orderbook(instrument_name)
                if orderbook:
                    # Drop options with zero liquidity when we require real quotes
                    if not use_estimated_prices:
                        bbq = float(orderbook.get('best_bid_qty') or 0.0)
                        baq = float(orderbook.get('best_ask_qty') or 0.0)
                        if bbq <= 0.0 and baq <= 0.0:
                            continue
                    enriched_option['bid'] = orderbook['best_bid']
                    enriched_option['ask'] = orderbook['best_ask']
                    enriched_option['mid'] = orderbook['mid']
                    enriched_option['spread'] = orderbook['spread']
                    enriched_option['bid_qty'] = orderbook['best_bid_qty']
                    enriched_option['ask_qty'] = orderbook['best_ask_qty']
                    enriched_option['has_live_prices'] = True
                else:
                    # No orderbook data available
                    if use_estimated_prices:
                        ok = self._add_estimated_prices(enriched_option, current_spot)
                        if not ok:
                            # e.g., can't estimate because of missing critical fields
                            continue
                    else:
                        continue  # Skip options without prices
            else:
                # No orderbook handler or instrument name
                if use_estimated_prices:
                    ok = self._add_estimated_prices(enriched_option, current_spot)
                    if not ok:
                        continue
                else:
                    continue
                    
            enriched_options.append(enriched_option)
        
        if skipped_unknown_type:
            self.logger.warning(f"[ENRICH] Skipped {skipped_unknown_type} options with unknown type")
        self.logger.info(f"[ENRICH] Completed enrichment: {len(enriched_options)} options enriched")

        # Publish the freshest priced view so downstream strategies never see a stale/empty chain.
        try:
            # Infer currency from the options payload
            def _infer_currency(opts: List[Dict]) -> str:
                for o in opts:
                    cur = (o.get("currency")
                           or (o.get("instrument_name") or o.get("symbol") or "").split("-")[0])
                    if cur:
                        return str(cur).upper()
                return "ETH"
            currency = _infer_currency(enriched_options)

            # Tag quotes with a source if missing; default WS when has_live_prices set
            now_ms = int((datetime.now(timezone.utc)).timestamp() * 1000)
            for q in enriched_options:
                if "source" not in q:
                    q["source"] = "WS" if q.get("has_live_prices") else ("EST" if q.get("price_estimated") else "REST")
                if "ts" not in q:
                    q["ts"] = now_ms

            # 1) Process-local cache (new)
            try:
                from market_data.options_quotes_cache import OptionsQuotesCache
                OptionsQuotesCache().put_many(currency, enriched_options)
                self.logger.info(f"[ENRICH] Cached %d enriched %s options in OptionsQuotesCache", len(enriched_options), currency)
            except Exception:
                self.logger.exception("[ENRICH] Failed to write to OptionsQuotesCache")

            # 2) Back-compat: OptionsRepository
            try:
                from market_data.options_repository import OptionsRepository
                OptionsRepository().set_active(currency, enriched_options)
                self.logger.info(f"[ENRICH] Published %d enriched %s options to OptionsRepository", len(enriched_options), currency)
            except Exception:
                self.logger.exception("[ENRICH] Failed to publish enriched options to OptionsRepository")
        except Exception:
            self.logger.exception("[ENRICH] Failed during enriched options publication stage")
        return enriched_options
    
    def _add_estimated_prices(self, option: Dict, current_spot: float) -> bool:
        """
        Add estimated bid/ask prices using simplified Black-Scholes
        
        This is a fallback when real market data isn't available
        """
        # Expect normalized fields to already exist; still be defensive:
        strike = option.get('strike') or option.get('strike_price') or option.get('strikePrice')
        try:
            strike = float(strike) if strike is not None else None
        except Exception:
            strike = None
        option_type = (option.get('type') or option.get('option_type'))
        if option_type is None:
            # Cannot safely price without type
            self.logger.warning(f"[ENRICH] Missing option type; skipping estimate for "
                                f"{option.get('instrument_name') or option.get('symbol') or 'unknown'}")
            return False
        option_type = str(option_type).strip().lower()
        if option_type in ('c', 'p'):
            option_type = 'call' if option_type == 'c' else 'put'

        expiry_date_str = option.get('expiry_date') or option.get('expiration_date') or option.get('expiry')
        if isinstance(expiry_date_str, (int, float)):
            expiry_date_str = datetime.utcfromtimestamp(expiry_date_str).strftime('%Y-%m-%d')
        
        if not (strike and current_spot and expiry_date_str):
            # Can't estimate without basic info
            option['bid'] = 0.01
            option['ask'] = 0.05
            option['mid'] = 0.03
            option['spread'] = 0.04
            option['has_live_prices'] = False
            option['skip_for_execution'] = True
            option['skip_for_execution'] = True
            return True
            
        try:
            # Calculate precise time to expiry
            from config_manager import DAYS_PER_YEAR
            SECONDS_PER_YEAR = DAYS_PER_YEAR * 24 * 3600
            expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
            # Use timezone-aware datetime
            now = datetime.now(timezone.utc) if expiry_date.tzinfo else datetime.now()
            seconds_to_expiry = (expiry_date - now).total_seconds()
            days_to_expiry = seconds_to_expiry / (24 * 3600)
            time_to_expiry = max(seconds_to_expiry / SECONDS_PER_YEAR, 1e-8)
            
            # Simple intrinsic value calculation
            if option_type == 'call':
                intrinsic = max(current_spot - strike, 0)
            else:  # put
                intrinsic = max(strike - current_spot, 0)
            
            self.logger.debug(f"[ENRICH] Estimating {option_type} strike={strike}, spot={current_spot}, intrinsic={intrinsic}")
            
            # Add time value (very simplified)
            moneyness = current_spot / strike
            if 0.9 < moneyness < 1.1:  # Near the money
                time_value = current_spot * 0.02 * np.sqrt(time_to_expiry)
            else:  # Out of the money
                time_value = current_spot * 0.01 * np.sqrt(time_to_expiry)
            
            # Estimated option value
            fair_value = intrinsic + time_value
            
            # Add spread (wider for options)
            spread_pct = 0.05 if fair_value > 1.0 else 0.10
            half_spread = fair_value * spread_pct / 2
            
            option['bid'] = max(fair_value - half_spread, 0.01)
            option['ask'] = fair_value + half_spread
            option['mid'] = fair_value
            option['spread'] = option['ask'] - option['bid']
            option['has_live_prices'] = False
            option['skip_for_execution'] = True
            option['price_estimated'] = True
            return True
            
        except Exception as e:
            self.logger.warning(f"Error estimating prices for option: {e}")
            # Fallback to minimal prices
            option['bid'] = 0.01
            option['ask'] = 0.05
            option['mid'] = 0.03
            option['spread'] = 0.04
            option['has_live_prices'] = False
            option['skip_for_execution'] = True
            option['skip_for_execution'] = True
            option['price_estimated'] = True
            return True
    
    def _normalize_option_fields(self, opt: Dict) -> Optional[Dict]:
        """
        Return {'type','strike','expiry_date'} if type can be inferred; else None.
        """
        t = self._infer_option_type(opt)
        if not t:
            return None
        # Strike normalization
        strike = opt.get('strike') or opt.get('strike_price') or opt.get('strikePrice')
        try:
            strike = float(strike) if strike is not None else None
        except Exception:
            strike = None
        # Expiry normalization (YYYY-MM-DD)
        expiry = (opt.get('expiry_date') or opt.get('expiration_date')
                  or opt.get('expiry') or opt.get('expiration') or opt.get('expiryTimestamp'))
        expiry_date_str = None
        if isinstance(expiry, (int, float)):
            expiry_date_str = datetime.utcfromtimestamp(expiry).strftime('%Y-%m-%d')
        elif isinstance(expiry, str):
            expiry_date_str = expiry.split('T')[0]
        # Compose
        out = {'type': t}
        if strike is not None:
            out['strike'] = strike
        if expiry_date_str:
            out['expiry_date'] = expiry_date_str
        # Also mirror to 'option_type' for any legacy code
        out['option_type'] = t
        return out

    def _infer_option_type(self, opt: Dict) -> Optional[str]:
        """
        Try several common fields to identify call/put, including:
        - type / option_type / optionType ('call'|'put' or 'C'|'P')
        - right / putCall (OCC-style)
        - is_call / isCall (boolean)
        - instrument_name or symbol suffixes like '-C'/'-P'
        """
        # String fields first
        for key in ('type', 'option_type', 'optionType', 'right', 'putCall'):
            v = opt.get(key)
            if v is None:
                continue
            t = str(v).strip().lower()
            if t in ('call', 'put'):
                return t
            if t in ('c', 'p'):
                return 'call' if t == 'c' else 'put'
        # Boolean fields (Lyra-style)
        for key in ('is_call', 'isCall', 'iscall'):
            v = opt.get(key)
            if isinstance(v, bool):
                return 'call' if v else 'put'
        # Try symbol parsing (e.g., ETH-30AUG24-2500-C)
        name = (opt.get('instrument_name') or opt.get('symbol') or '')
        m = re.search(r'[-_/ ]([CP])$', name) or re.search(r'\b(call|put)\b', name, re.I)
        if m:
            grp = m.group(1).lower()
            return 'call' if grp in ('c', 'call') else 'put' if grp in ('p', 'put') else None
        return None