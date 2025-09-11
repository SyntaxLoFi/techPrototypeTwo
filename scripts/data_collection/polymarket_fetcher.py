"""
Polymarket data fetcher for crypto price predictions
Handles the correct nested event/market structure

FIXED ISSUES:
1. Timezone handling - now uses timezone.utc consistently
2. Better error handling for missing/malformed dates
"""
import logging
import os
import math
from typing import Dict, List, Tuple, Optional

import requests

from datetime import datetime, timezone, timedelta
import re
import time
import json
import os
from itertools import islice

from config_manager import (
    POLYMARKET_API_BASE, POLYMARKET_CLOB_BASE,
    HTTP_TIMEOUT_SEC, HTTP_RETRY_MAX, POLYMARKET_PRICES_CHUNK,
    POLYMARKET_FILTER_ACTIVE, POLYMARKET_FILTER_CLOSED,
    POLYMARKET_INCLUDE_DAILIES, POLYMARKET_DAILIES_WINDOW_HOURS,
    MIN_HOURS_TO_EXPIRY
)
from utils.http_client import get, post

logger = logging.getLogger(__name__)

DEFAULT_CLOB_BASE = os.getenv("POLYMARKET_CLOB_BASE", "https://clob.polymarket.com")


class PolymarketFetcher:
    """Fetch and parse Polymarket prediction markets with correct structure"""
    
    def __init__(self):
        self.api_base = POLYMARKET_API_BASE
        # Fall back to default if config is unset/blank
        self.clob_base = POLYMARKET_CLOB_BASE or DEFAULT_CLOB_BASE
        self.logger = logging.getLogger('PolymarketFetcher')
        # Rate limiting now handled by http_client
        self.prices_chunk = POLYMARKET_PRICES_CHUNK
        self.token_prices_cache = {}  # Cache token prices to reduce API calls
        
    def fetch_crypto_events(self) -> List[Dict]:
        """Fetch all crypto-related prediction events"""
        self.logger.info("Fetching crypto price prediction events...")
        
        all_events = []
        offset = 0
        batch_size = 100
        
        while True:
            params = {
                'limit': batch_size,
                'offset': offset,
                # Filter at the source: active events only
                'active': 'true' if POLYMARKET_FILTER_ACTIVE else 'false'
            }
            
            # Also filter out closed markets if configured
            if POLYMARKET_FILTER_CLOSED:
                params['closed'] = 'false'
            
            try:
                response = get(f"{self.api_base}/events", params=params)
                
                if response.status_code == 200:
                    events_batch = response.json()
                    
                    if not events_batch:
                        break
                    
                    all_events.extend(events_batch)
                    self.logger.debug(f"Fetched batch at offset {offset}, got {len(events_batch)} events")
                    
                    if len(events_batch) < batch_size:
                        break
                        
                    offset += batch_size
                    
                    if len(all_events) > 10000:
                        self.logger.warning("Reached safety limit of 10,000 events")
                        break
                else:
                    self.logger.error(f"API error: {response.status_code}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Request error: {e}")
                break
        
        self.logger.info(f"Fetched {len(all_events)} total active events")
        
        # Filter for crypto price predictions
        price_events = []
        for event in all_events:
            if self._is_crypto_price_event(event):
                price_events.append(event)
        
        self.logger.info(f"Found {len(price_events)} crypto price prediction events")
        
        # Count total markets
        total_markets = sum(len(event.get('markets', [])) for event in price_events)
        self.logger.info(f"Total crypto markets across all events: {total_markets}")
        
        # Validate we found data
        if not price_events:
            error_msg = "CRITICAL: No crypto price prediction events found on Polymarket. API may be down or filters too restrictive."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Sort by liquidity
        price_events.sort(key=lambda x: float(x.get('liquidity', 0) or 0), reverse=True)
        
        return price_events
    
    def fetch_crypto_markets(self) -> List[Dict]:
        """
        Fetch crypto markets directly from Gamma /markets endpoint
        This is an alternative approach that might have better token ID mapping
        """
        self.logger.info("Fetching crypto markets from Gamma /markets endpoint...")
        
        all_markets = []
        limit = 100
        offset = 0
        
        while True:
            # Filter at-source: keep active markets; if requested, exclude closed
            params = {
                'limit': limit,
                'offset': offset,
                'active': 'true' if POLYMARKET_FILTER_ACTIVE else 'false',
            }
            if POLYMARKET_FILTER_CLOSED:
                params['closed'] = 'false'
            
            try:
                response = get(f"{self.api_base}/markets", params=params)
                
                if response.status_code == 200:
                    markets_batch = response.json()
                    
                    if not markets_batch:
                        break
                    
                    # Final guard against closed/inactive/too-soon expiry
                    markets_batch = self.filter_markets(markets_batch)
                    
                    # Filter for crypto markets
                    for market in markets_batch:
                        if self._is_crypto_price_market(market):
                            all_markets.append(market)
                    
                    if len(markets_batch) < limit:
                        break
                        
                    offset += limit
                else:
                    self.logger.error(f"Markets API error: {response.status_code}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Request error fetching markets: {e}")
                break
        
        self.logger.info(f"Found {len(all_markets)} crypto price markets")
        return all_markets
    
    def _is_crypto_price_market(self, market: Dict) -> bool:
        """Check if a market is a crypto price prediction"""
        market_description = market.get('description', '').upper()
        supported_cryptos = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']
        
        # Check both patterns in market description
        for crypto in supported_cryptos:
            if f"{crypto}/USDT" in market_description or f"{crypto}USDT" in market_description:
                return True
        
        return False
    
    @staticmethod
    def _parse_iso_utc(s: str) -> Optional[datetime]:
        if not s:
            return None
        try:
            # Handles '...Z' or offset form
            return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _to_float(value) -> Optional[float]:
        """Convert value to float, returning None if invalid"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _market_time_left_hours(self, m: dict) -> Optional[float]:
        # Gamma uses end_date; legacy fields may be endDate or endDateIso
        end_s = m.get("end_date") or m.get("endDate") or m.get("endDateIso")
        end_dt = self._parse_iso_utc(end_s)
        if not end_dt:
            return None
        return (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0

    def _should_keep_market(self, m: dict) -> bool:
        # Extra client-side guard based on CLOB/Gamma booleans and expiry window
        if m.get("active") is False:
            return False
        if m.get("closed") is True and POLYMARKET_FILTER_CLOSED:
            return False
        # Accept both camelCase and snake_case flags from upstream
        if (m.get("enableOrderBook") is False) or (m.get("enable_order_book") is False):
            return False
        # Respect min_hours_to_expiry generally…
        tleft = self._market_time_left_hours(m)
        if tleft is None:
            return True  # no end date -> keep (let later filters handle)
        if tleft >= MIN_HOURS_TO_EXPIRY:
            return True
        # …but allow "dailies" (short-fuse markets) when configured:
        if POLYMARKET_INCLUDE_DAILIES and tleft >= -0.01:  # small negative tolerance
            # Limit to a window (e.g., within next 24h)
            return tleft <= POLYMARKET_DAILIES_WINDOW_HOURS
        return False

    def filter_markets(self, markets: List[Dict]) -> List[Dict]:
        return [m for m in markets or [] if self._should_keep_market(m)]

    def collect_all_markets(self, page_limit: int = 500) -> List[Dict]:
        offset = 0
        out: List[Dict] = []
        while True:
            resp = self.fetch_markets(limit=page_limit, offset=offset)
            if resp.status_code != 200:
                self.logger.error("Gamma /markets error %s", resp.status_code)
                break
            batch = resp.json() or []
            # Final guard against closed/inactive/too-soon expiry
            batch = self.filter_markets(batch)
            out.extend(batch)
            if len(batch) < page_limit:
                break
            offset += page_limit
        return out
    
    def fetch_markets(self, limit: int = 500, offset: int = 0, extra_params: dict = None) -> requests.Response:
        # Filter at-source: keep active markets; if requested, exclude closed
        params = {
            "limit": limit,
            "offset": offset,
            "active": 'true' if POLYMARKET_FILTER_ACTIVE else 'false',
        }
        if not POLYMARKET_FILTER_CLOSED:
            params['closed'] = 'false'
        if extra_params:
            params.update(extra_params)
        resp = get(f"{self.api_base}/markets", params=params)
        return resp
    
    def fetch_crypto_simplified_markets(self) -> List[Dict]:
        """
        Fetch crypto markets using CLOB /simplified-markets endpoint
        This provides token IDs directly without needing events
        """
        self.logger.info("Fetching crypto markets from CLOB /simplified-markets...")
        
        all_markets = []
        next_cursor = ""
        
        while True:
            try:
                params = {}
                if next_cursor:
                    params["next_cursor"] = next_cursor
                
                response = get(f"{self.clob_base}/simplified-markets", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    markets_batch = data.get("data", [])
                    
                    if not markets_batch:
                        break
                    
                    # Filter for crypto markets
                    for market in markets_batch:
                        if self._is_crypto_price_simplified_market(market):
                            all_markets.append(market)
                    
                    # Check for next page
                    next_cursor = data.get("next_cursor", "")
                    if not next_cursor or next_cursor == "LTE=":  # End marker
                        break
                else:
                    self.logger.error(f"Simplified markets API error: {response.status_code}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Request error fetching simplified markets: {e}")
                break
        
        self.logger.info(f"Found {len(all_markets)} crypto price markets")
        return all_markets
    
    def _is_crypto_price_simplified_market(self, market: Dict) -> bool:
        """Check if a simplified market is a crypto price prediction"""
        description = market.get("description", "").upper()
        supported_cryptos = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']
        
        # Check both patterns in description
        has_crypto_pattern = False
        for crypto in supported_cryptos:
            if f"{crypto}/USDT" in description or f"{crypto}USDT" in description:
                has_crypto_pattern = True
                break
        
        # Also check if active and has order book
        is_active = market.get("active", False)
        enable_order_book = market.get("enable_order_book", True)
        
        return has_crypto_pattern and is_active and enable_order_book
    
    def _is_crypto_price_event(self, event: Dict) -> bool:
        """Check if event contains crypto price prediction markets"""
        # Check if any market in the event has CRYPTO/USDT or CRYPTOUSDT patterns
        supported_cryptos = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']
        
        for market in event.get('markets', []):
            market_description = market.get('description', '').upper()
            
            for crypto in supported_cryptos:
                # Check both patterns: CRYPTO/USDT and CRYPTOUSDT
                if f"{crypto}/USDT" in market_description or f"{crypto}USDT" in market_description:
                    return True
        
        # Also check event description as fallback
        event_description = event.get('description', '').upper()
        for crypto in supported_cryptos:
            if f"{crypto}/USDT" in event_description or f"{crypto}USDT" in event_description:
                return True
        
        return False
    
    def _request_with_retries(self, method, url, **kwargs):
        """Delegate to http_client which handles retries."""
        if method.__name__ == 'post':
            return post(url, **kwargs)
        else:
            return get(url, **kwargs)

    def _fetch_books(self, token_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch order books for a batch of token_ids.
        Try the documented {"params":[...]} payload first; if that fails, retry with a raw array.
        Finally, GET /book for any stragglers.
        """
        if not token_ids:
            return {}
        out: Dict[str, Dict] = {}
        books: list[dict] = []
        # 1) POST /books with {"params":[...]}
        payload1 = {"params": [{"token_id": tid} for tid in token_ids]}
        try:
            resp1 = post(f"{self.clob_base}/books", json=payload1)
            resp1.raise_for_status()
            data1 = resp1.json()
            books = data1 if isinstance(data1, list) else (data1.get("orderbooks") or data1.get("books") or [])
        except Exception:
            # 2) POST /books with raw array body ([{"token_id":...}, ...])
            payload2 = [{"token_id": tid} for tid in token_ids]
            try:
                resp2 = post(f"{self.clob_base}/books", json=payload2)
                resp2.raise_for_status()
                data2 = resp2.json()
                books = data2 if isinstance(data2, list) else (data2.get("orderbooks") or data2.get("books") or [])
            except Exception:
                books = []

        # Convert any books we got
        try:
            for ob in books:
                # Accept 'asset_id' (canonical) or 'token_id' in response
                tid = ob.get("asset_id") or ob.get("token_id")
                if not tid:
                    continue
                bids = ob.get("bids") or []
                asks = ob.get("asks") or []
                best_bid, bb_sz = self._best(bids, best=True)
                best_ask, ba_sz = self._best(asks, best=False)
                mid = (best_bid + best_ask)/2 if (best_bid and best_ask) else (best_bid or best_ask or 0.0)
                out[str(tid)] = {
                    "best_bid": best_bid, "best_bid_size": bb_sz,
                    "best_ask": best_ask, "best_ask_size": ba_sz,
                    "midpoint": mid
                }
        except Exception:
            out = {}

        # 3) GET /book for any missing
        missing = [tid for tid in token_ids if tid and tid not in out]
        for tid in missing:
            try:
                r = get(f"{self.clob_base}/book", params={"token_id": tid})
                if r and r.status_code == 200:
                    ob = r.json() or {}
                    bids = ob.get("bids") or []
                    asks = ob.get("asks") or []
                    best_bid, bb_sz = self._best(bids, best=True)
                    best_ask, ba_sz = self._best(asks, best=False)
                    mid = (best_bid + best_ask)/2 if (best_bid and best_ask) else (best_bid or best_ask or 0.0)
                    out[str(tid)] = {
                        "best_bid": best_bid, "best_bid_size": bb_sz,
                        "best_ask": best_ask, "best_ask_size": ba_sz,
                        "midpoint": mid
                    }
            except Exception:
                continue
        self.logger.info(f"[CLOB] fetched books for {len(out)}/{len(set(token_ids))} tokens")
        return out

    def fetch_clob_prices(self, token_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch prices from CLOB for given token IDs
        Returns dict mapping token_id to price info
        """
        if not token_ids:
            return {}
        
        # Remove duplicates and filter out empty strings
        unique_token_ids = list(set(id for id in token_ids if id))
        
        if not unique_token_ids:
            return {}
        
        prices: Dict[str, Dict] = {}
        try:
            # Process in chunks to respect payload/infra limits
            for i in range(0, len(unique_token_ids), self.prices_chunk):
                batch = unique_token_ids[i : i + self.prices_chunk]
                batch_params = []
                for token_id in batch:
                    # Per docs: request uses token_id; response maps by asset_id
                    # BUY -> ask, SELL -> bid (per docs)
                    batch_params.append({"token_id": token_id, "side": "BUY"})
                    batch_params.append({"token_id": token_id, "side": "SELL"})
                # API expects a raw ARRAY body, not {"params": ...}
                payload = batch_params
                response = post(f"{self.clob_base}/prices", json=payload)
                if not response:
                    continue
                response.raise_for_status()
                # Response: {asset_id: {"BUY": ask_px_or_null, "SELL": bid_px_or_null}}
                price_map = response.json()
                
                # Process response - convert to our format
                for asset_id, sides in price_map.items():
                    if isinstance(sides, dict):
                        # BUY means "price to buy" -> best ASK; SELL means "price to sell" -> best BID
                        ask_value = sides.get("BUY")     # BUY -> best ask
                        bid_value = sides.get("SELL")    # SELL -> best bid
                        
                        # Convert to float, treating None as 0
                        best_bid = float(bid_value) if bid_value is not None else 0.0
                        best_ask = float(ask_value) if ask_value is not None else 0.0
                        
                        # Calculate midpoint only if both sides exist
                        if best_bid > 0 and best_ask > 0:
                            midpoint = (best_bid + best_ask) / 2
                        else:
                            midpoint = best_bid or best_ask or 0
                        
                        prices.setdefault(asset_id, {})
                        prices[asset_id].update({
                            "best_bid": best_bid,
                            "best_ask": best_ask,
                            "midpoint": midpoint
                        })
            # Fallback for any missing or empty tokens via /books
            missing = [
                tid for tid in unique_token_ids
                if tid not in prices or ((prices[tid].get("best_bid") or 0) == 0 and (prices[tid].get("best_ask") or 0) == 0)
            ]
            if missing:
                self.logger.debug(f"[CLOB] Falling back to /books for {len(missing)} tokens (sample={missing[:5]})")
                books = self._fetch_books(missing)
                for tid, v in books.items():
                    prices[tid] = v
            self.logger.info(f"Fetched CLOB prices for {len(prices)} tokens (with /books fallback if needed)")
        except Exception as e:
            self.logger.error(f"Error fetching CLOB prices: {str(e)}")
            return {}
        
        # Update cache
        self.token_prices_cache.update(prices)
        
        return prices
    
    def parse_contracts(self, events: List[Dict]) -> List[Dict]:
        """Parse events into standardized contract format"""
        contracts = []
        token_to_market_map = {}  # Map token_id -> market_info (for /prices)
        all_markets = []  # Track all valid markets
        
        for event in events:
            # Handle both event_id and id fields
            event_id = event.get('event_id', event.get('id', ''))
            event_title = event.get('title', '')
            
            # Process each market within the event
            markets = event.get('markets', [])
            for market in markets:
                # Skip inactive or closed markets
                if market.get('closed') or not market.get('active', True):
                    continue
                
                # Check time filter
                if not self._should_keep_market(market):
                    continue
                
                # Extract currency from market description
                market_description = market.get('description', '').upper()
                currency = None
                for crypto in ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']:
                    if f"{crypto}/USDT" in market_description or f"{crypto}USDT" in market_description:
                        currency = crypto
                        break
                
                if not currency:
                    continue
                
                # Store market for later processing
                # We'll create contracts regardless of token availability
                market_id = market.get('market_id', market.get('id', ''))
                market_info = {
                    'market': market,
                    'event': event,
                    'event_id': event_id,
                    'event_title': event_title,
                    'currency': currency,
                    'market_id': market_id
                }
                
                # Add to all markets list
                all_markets.append(market_info)
                
                # Extract token IDs from market (YES and NO)
                token_ids = []
                
                # Try different field names for token IDs
                # DO NOT treat string 'outcomes' (e.g., "YES"/"NO") as token_ids.
                # Only accept explicit token id fields.
                outcome_tokens = market.get('outcomes', [])
                if outcome_tokens:
                    for outcome in outcome_tokens:
                        if isinstance(outcome, dict):
                            token_id = outcome.get('token_id') or outcome.get('id')
                            if token_id:
                                token_ids.append(str(token_id))
                
                # Alternative: direct token fields
                if not token_ids:
                    yes_token = market.get('yes_token_id') or market.get('yesTokenId')
                    no_token  = market.get('no_token_id')  or market.get('noTokenId')
                    if yes_token: token_ids.append(yes_token)
                    if no_token:  token_ids.append(no_token)
                
                # Check for clobTokenIds field (common in new API)
                if not token_ids:
                    clob_tokens = market.get('clobTokenIds')  # Gamma often exposes this as a JSON string
                    if clob_tokens:
                        # Parse the clobTokenIds string that looks like "[\"token1\", \"token2\"]"
                        try:
                            import json as _json
                            token_list = _json.loads(clob_tokens)
                            if isinstance(token_list, list):
                                token_ids.extend([str(t) for t in token_list if t])
                        except Exception:
                            pass

                # Resolve YES/NO token IDs robustly and persist on market_info
                yes_id, no_id = self._resolve_yes_no_token_ids(market)
                resolved_ids = [x for x in (yes_id, no_id) if x]
                if not resolved_ids and token_ids:
                    # Keep unlabeled pair if that's all we have
                    resolved_ids = token_ids
                if resolved_ids:
                    market_info['token_ids'] = resolved_ids
                if yes_id:
                    market_info['yes_token_id'] = yes_id
                if no_id:
                    market_info['no_token_id']  = no_id

                # Also build the token -> market map for /prices batching
                for tid in resolved_ids:
                    if tid and tid not in token_to_market_map:
                        token_to_market_map[tid] = {
                            'market_id': market_id
                        }
        
        # Fetch all CLOB prices in one batch
        all_token_ids = list(token_to_market_map.keys())
        self.logger.debug(f"[CLOB] aggregated unique token ids: {len(all_token_ids)} (sample={all_token_ids[:10]})")
        if all_token_ids:
            self.logger.info(f"Fetching CLOB prices for {len(all_token_ids)} tokens...")
            clob_prices = self.fetch_clob_prices(all_token_ids)
        else:
            clob_prices = {}
        
        # Second pass: create contracts for all markets
        processed_markets = set()  # Track processed markets to avoid duplicates
        
        for market_info in all_markets:
            market = market_info['market']
            market_id = market_info['market_id']
            
            # Skip if we already processed this market
            if market_id in processed_markets:
                continue
            processed_markets.add(market_id)
            
            # Collect BOTH token prices for this market
            market_prices = {}
            for token_id, token_market_info in token_to_market_map.items():
                if token_market_info['market_id'] == market_id:
                    if token_id in clob_prices:
                        market_prices[token_id] = clob_prices[token_id]
            
            contract = self._parse_market_contract(
                market_info['event_id'],
                market_info['event_title'],
                market_info['currency'],
                market,
                market_info['event'],
                market_prices  # map of token_id -> price info (both sides if present)
            )
            # Attach token IDs so enrich_with_books() can query CLOB books and set top-of-book sizes
            toks = market_info.get('token_ids') or []
            if toks:
                contract['clob_token_ids'] = toks
                if 'yes_token_id' in market_info: contract['yes_token_id'] = market_info['yes_token_id']
                if 'no_token_id'  in market_info: contract['no_token_id']  = market_info['no_token_id']
            
            if contract:
                contracts.append(contract)
        
        self.logger.info(f"Parsed {len(contracts)} contracts from {len(all_markets)} crypto markets")
        
        # Validate we parsed contracts
        if not contracts:
            error_msg = f"CRITICAL: Failed to parse any contracts from {len(all_markets)} crypto markets. Check parsing logic or market data format."
            self.logger.error(error_msg)
            # Don't raise here as the fallback to simplified markets may work
            
        # Debug surface: how many contracts can be enriched with books?
        try:
            with_tokens = sum(1 for c in contracts if c.get('yes_token_id') or c.get('clob_token_ids'))
            self.logger.info(f"Parsed {len(contracts)} contracts; {with_tokens} carry token IDs for CLOB books")
        except Exception:
            pass
        return contracts
    
    def _extract_currency(self, title: str) -> Optional[str]:
        """Extract cryptocurrency from event title"""
        title_lower = title.lower()
        
        currency_map = {
            'bitcoin': 'BTC',
            'btc': 'BTC',
            'ethereum': 'ETH',
            'eth': 'ETH',
            'solana': 'SOL',
            'sol': 'SOL',
            'xrp': 'XRP',
            'ripple': 'XRP',
            'dogecoin': 'DOGE',
            'doge': 'DOGE'
        }
        
        for term, currency in currency_map.items():
            if term in title_lower:
                return currency
        
        return None
    
    # Price extraction patterns - require $ or k/m suffix to avoid matching dates
    _PRICE_RE = re.compile(
        r"""
        (?:                             # CASE A: $ followed by number (opt decimals), opt k/m
            \$\s*(?P<dol_num>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dol_dec>\d+))?
            \s*(?P<dol_suffix>[kKmM])?
        )
        |
        (?:                             # CASE B: bare number with required k/m suffix
            \b(?P<bare_num>\d+(?:\.\d+)?)\s*(?P<bare_suffix>[kKmM])\b
        )
        """,
        re.VERBOSE,
    )
    
    # Date patterns to scrub before looking for prices
    _DATE_LIKE_RE = re.compile(
        r"""\b(
            (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2} |
            \d{4}-\d{2}-\d{2} |
            \d{1,2}/\d{1,2}/\d{2,4}
        )\b""",
        re.IGNORECASE | re.VERBOSE,
    )
    
    # Asset aliases for multi-asset detection
    _ASSET_ALIASES = {
        "BTC": {"BTC", "BITCOIN"},
        "ETH": {"ETH", "ETHEREUM"},
        "SOL": {"SOL", "SOLANA"},
        "XRP": {"XRP", "RIPPLE"},
        "DOGE": {"DOGE", "DOGECOIN"},
    }
    
    def _extract_strike_from_text(self, text: str) -> Optional[float]:
        """Extract strike price from text, avoiding date numbers"""
        if not text:
            return None
            
        # Remove obvious date tokens to reduce false positives
        cleaned = self._DATE_LIKE_RE.sub(" ", text)
        
        m = self._PRICE_RE.search(cleaned)
        if not m:
            return None
        
        if m.group("dol_num"):
            num = float(m.group("dol_num").replace(",", ""))
            dec = m.group("dol_dec")
            if dec:
                num += float(f"0.{dec}")
            suffix = m.group("dol_suffix")
        else:
            num = float(m.group("bare_num"))
            suffix = m.group("bare_suffix")
        
        if suffix in ("k", "K"):
            num *= 1_000.0
        elif suffix in ("m", "M"):
            num *= 1_000_000.0
        
        return num
    
    def _count_assets_in_text(self, text: str) -> int:
        """Count distinct crypto assets mentioned in text"""
        if not text:
            return 0
            
        u = text.upper()
        found = 0
        for _, names in self._ASSET_ALIASES.items():
            if any(re.search(rf"\b{name}\b", u) for name in names):
                found += 1
        return found
    
    def _parse_market_contract(self, event_id: str, event_title: str, currency: str, market: Dict, event: Dict, clob_price_info: Dict = None) -> Optional[Dict]:
        """Parse a market into contract format"""
        question = market.get('question', '')
        
        # Skip multi-asset markets
        if self._count_assets_in_text(question) > 1 or self._count_assets_in_text(event_title) > 1:
            return None
        
        # Extract strike price from question
        strike_price = self._extract_strike_from_text(question)
        
        # For markets without explicit strike (Up/Down, ATH, etc), use current price as reference
        if strike_price is None:
            # These are still valid markets for strategies
            # Use a placeholder or current spot price
            strike_price = 0  # Will be handled by strategies appropriately
        
        # Determine direction from question
        question_lower = (question or '').strip().lower()
        
        # Default to checking if price goes above
        is_above = True
        
        # Check for specific patterns
        if any(word in question_lower for word in ['dip to', 'below', 'under', 'drop to']):
            is_above = False
        elif any(word in question_lower for word in ['reach', 'hit', 'above', 'over']):
            is_above = True
        
        # Special case: "dip to" means checking if price goes below
        if 'dip to' in question_lower:
            is_above = False
        
        # First try to get prices from outcomePrices field (most reliable)
        outcome_prices_raw = market.get('outcomePrices', [])
        
        # Handle case where outcomePrices is a JSON string
        if isinstance(outcome_prices_raw, str):
            try:
                import json
                outcome_prices = json.loads(outcome_prices_raw)
            except:
                outcome_prices = []
        else:
            outcome_prices = outcome_prices_raw
            
        if outcome_prices and len(outcome_prices) >= 2:
            # outcomePrices is [YES, NO] as strings
            try:
                yes_price = float(outcome_prices[0])
                no_price = float(outcome_prices[1])
                self.logger.debug(f"Using outcomePrices for '{question[:50]}...': YES={yes_price}, NO={no_price}")
            except (ValueError, IndexError):
                yes_price = 0
                no_price = 0
        else:
            yes_price = 0
            no_price = 0
        
        # If outcomePrices didn't work, try CLOB data
        if yes_price == 0:
            if clob_price_info:
                # Use CLOB prices (more accurate)
                best_bid = clob_price_info.get('best_bid', 0)
                best_ask = clob_price_info.get('best_ask', 0)
                yes_price = clob_price_info.get('midpoint', 0)
                
                if yes_price == 0:
                    # Fall back to lastTradePrice
                    yes_price = float(market.get('lastTradePrice', 0) or 0)
            else:
                # No CLOB data, use lastTradePrice
                yes_price = float(market.get('lastTradePrice', 0) or 0)
                best_bid = market.get('bestBid', 0)
                best_ask = market.get('bestAsk', 0)
        
        # If still no price data, use default 50/50 odds
        if yes_price == 0:
            yes_price = 0.5
            no_price = 0.5
            self.logger.debug(f"No price data for market '{question[:50]}...', using 50/50 default")
        elif ('no_price' not in locals()) or (no_price is None):
            # Do NOT infer NO from (1 - YES). If NO is missing, either:
            #  (a) derive from the NO order book if available, or
            #  (b) mark this market as incomplete / skip it.
            # Option (a): try order book mid; fall back to bid/ask.
            no_bid = self._to_float(market.get("no_best_bid"))
            no_ask = self._to_float(market.get("no_best_ask"))
            if (no_bid is not None) and (no_ask is not None):
                no_price = (no_bid + no_ask) / 2.0
            elif no_bid is not None:
                no_price = no_bid
            elif no_ask is not None:
                no_price = no_ask
            else:
                self.logger.warning("Skipping market %s: NO price unavailable; not inferring 1-YES", market.get('market_id', market.get('id', 'unknown')))
                return None  # or: set no_price=None; downstream must handle missing
        
        # Calculate days to expiry - fixed timezone handling
        # Constants for precise time calculations
        DAYS_PER_YEAR = 365.0  # Should import from config but avoiding large changes
        SECONDS_PER_YEAR = DAYS_PER_YEAR * 24 * 3600
        
        end_date_str = market.get('end_date') or market.get('endDate') or event.get('end_date', '')
        if end_date_str:
            try:
                # Parse the end date and ensure it's timezone-aware
                if end_date_str.endswith('Z'):
                    end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                else:
                    # If no timezone info, assume UTC
                    end_date = datetime.fromisoformat(end_date_str)
                    if end_date.tzinfo is None:
                        end_date = end_date.replace(tzinfo=timezone.utc)
                
                # Use timezone-aware current time
                current_time = datetime.now(timezone.utc)
                
                # Calculate time in seconds for precision
                seconds_to_expiry = (end_date - current_time).total_seconds()
                
                # Calculate both days and precise time
                days_to_expiry = seconds_to_expiry / (24 * 3600)
                time_to_expiry_years = max(seconds_to_expiry / SECONDS_PER_YEAR, 1e-8)  # Avoid zero
                
                # Extract just the date for options strategy generator
                expiry_date = end_date.strftime('%Y-%m-%d')
            except Exception as e:
                self.logger.warning(f"Error parsing date '{end_date_str}': {e}")
                days_to_expiry = 30  # Default
                time_to_expiry_years = 30 / DAYS_PER_YEAR
                expiry_date = None
        else:
            days_to_expiry = 30
            time_to_expiry_years = 30 / 365.0
            expiry_date = None
        
        return {
            'event_id': event_id,
            'market_id': market.get('market_id', market.get('id', '')),
            'event_title': event_title,
            'question': question,
            'currency': currency,
            'strike_price': strike_price,
            'is_above': is_above,
            'yes_price': yes_price,
            'no_price': no_price,
            'end_date': end_date_str,
            'expiry_date': expiry_date,  # ADDED: For options strategy generator
            'days_to_expiry': max(0.02, days_to_expiry),  # Allow sub-day expiries (min 30 minutes)
            'time_to_expiry_years': time_to_expiry_years,  # Precise time for Black-Scholes
            'volume_24hr': float(market.get('volume_24hr', 0) or market.get('volume24hr', 0) or 0),
            'liquidity': float(market.get('liquidity', 0) or 0),
            'best_bid': best_bid if 'best_bid' in locals() else float(market.get('bestBid', 0) or 0),
            'best_ask': best_ask if 'best_ask' in locals() else float(market.get('bestAsk', 0) or 0),
            'spread': float(market.get('spread', 0) or 0)
        }
    
    def get_active_currencies(self, contracts: List[Dict]) -> List[str]:
        """Get list of currencies that have active contracts"""
        currencies = set()
        for contract in contracts:
            if contract.get('currency'):
                currencies.add(contract['currency'])
        return list(currencies)
    
    def parse_contracts_from_simplified_markets(self, markets: List[Dict]) -> List[Dict]:
        """Parse simplified markets into standardized contract format (using CLOB data)"""
        return self.parse_simplified_markets(markets)
    
    def parse_simplified_markets(self, markets: List[Dict]) -> List[Dict]:
        """Parse simplified markets into standardized contract format"""
        contracts = []
        
        # Collect all token IDs for batch price fetching
        all_token_ids = []
        for market in markets:
            tokens = market.get("tokens", [])
            if len(tokens) >= 2:  # Binary markets have 2 tokens
                all_token_ids.extend([t["token_id"] for t in tokens if "token_id" in t])
        
        # Fetch all prices in batch
        clob_prices = {}
        if all_token_ids:
            self.logger.info(f"Fetching CLOB prices for {len(all_token_ids)} tokens...")
            clob_prices = self.fetch_clob_prices(all_token_ids)
        
        # Parse each market
        for market in markets:
            contract = self._parse_simplified_market(market, clob_prices)
            if contract:
                contracts.append(contract)
        
        return contracts
    
    def _parse_simplified_market(self, market: Dict, clob_prices: Dict) -> Optional[Dict]:
        """Parse a simplified market into contract format"""
        question = market.get("question", "")
        condition_id = market.get("condition_id", "")
        
        # Skip multi-asset markets
        if self._count_assets_in_text(question) > 1:
            return None
        
        # Extract currency from question
        currency = self._extract_currency(question)
        if not currency:
            return None
        
        # Extract strike price using improved regex
        strike_price = self._extract_strike_from_text(question)
        if strike_price is None:
            return None
        
        # Determine direction
        question_lower = (question or '').strip().lower()
        is_above = True
        
        if any(word in question_lower for word in ['dip to', 'below', 'under', 'drop to']):
            is_above = False
        elif any(word in question_lower for word in ['reach', 'hit', 'above', 'over']):
            is_above = True
        
        # Get tokens (binary markets have YES and NO tokens)
        tokens = market.get("tokens", [])
        if len(tokens) < 2:
            return None
        
        # Find YES token (outcome "Yes")
        yes_token = None
        no_token = None
        for token in tokens:
            if token.get("outcome", "").lower() == "yes":
                yes_token = token
            elif token.get("outcome", "").lower() == "no":
                no_token = token
        
        if not yes_token or not no_token:
            return None
        
        # Get prices from CLOB
        yes_token_id = yes_token.get("token_id")
        yes_price_info = clob_prices.get(yes_token_id, {})
        
        # Calculate YES price (midpoint)
        yes_price = yes_price_info.get("midpoint", 0)
        if yes_price == 0:
            # Fall back to market data if available
            yes_price = float(market.get("price", 0))
        
        if yes_price == 0:
            return None
        
        # Do NOT infer NO from (1 - YES). Prefer explicit NO or order book.
        no_price = self._to_float(market.get("no_price"))
        if no_price is None:
            no_bid = self._to_float(market.get("no_best_bid"))
            no_ask = self._to_float(market.get("no_best_ask"))
            if (no_bid is not None) and (no_ask is not None):
                no_price = (no_bid + no_ask) / 2.0
            elif no_bid is not None:
                no_price = no_bid
            elif no_ask is not None:
                no_price = no_ask
            else:
                self.logger.warning("Market %s missing NO; emitting as incomplete (no_price=None).", market.get("id"))
                no_price = None  # or: return None to drop it
        
        # Calculate expiry
        end_date_str = market.get("end_date_iso", "")
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                current_time = datetime.now(timezone.utc)
                seconds_to_expiry = (end_date - current_time).total_seconds()
                days_to_expiry = seconds_to_expiry / (24 * 3600)
                time_to_expiry_years = max(seconds_to_expiry / (365 * 24 * 3600), 1e-8)
                expiry_date = end_date.strftime('%Y-%m-%d')
            except Exception as e:
                self.logger.warning(f"Error parsing date '{end_date_str}': {e}")
                days_to_expiry = 30
                time_to_expiry_years = 30 / 365.0
                expiry_date = None
        else:
            days_to_expiry = 30
            time_to_expiry_years = 30 / 365.0
            expiry_date = None
        
        return {
            'market_id': condition_id,
            'condition_id': condition_id,
            'question': question,
            'currency': currency,
            'strike_price': strike_price,
            'is_above': is_above,
            'yes_price': yes_price,
            'no_price': no_price,
            'end_date': end_date_str,
            'expiry_date': expiry_date,
            'days_to_expiry': max(0.02, days_to_expiry),
            'time_to_expiry_years': time_to_expiry_years,
            'volume': float(market.get("volume", 0)),
            'liquidity': float(market.get("liquidity", 0)),
            'best_bid': yes_price_info.get("best_bid", 0),
            'best_ask': yes_price_info.get("best_ask", 0),
            'spread': yes_price_info.get("best_ask", 0) - yes_price_info.get("best_bid", 0) if yes_price_info else 0,
            'yes_token_id': yes_token_id,
            'no_token_id': no_token.get("token_id")
        }

    # ----------------------------
    # Orderbook enrichment (CLOB)
    # ----------------------------
    def enrich_with_books(self, contracts):
        """
        Populate top-of-book **prices AND sizes** for both tokens of each market.
        Request shape per docs: POST /books with {"params":[{"token_id":"..."}]}.
        Fallback: GET /book?token_id=... for any missed token.
        """
        # 1) Collect all YES/NO token_ids from every contract
        all_ids = []
        for c in contracts or []:
            y, n = self._extract_token_ids(c)
            if y: all_ids.append(y)
            if n: all_ids.append(n)
        all_ids = list({i for i in all_ids if i})
        if not all_ids:
            return contracts
        # 2) Fetch orderbooks in chunks
        books = {}
        chunk = int(os.getenv("POLYMARKET_BOOKS_CHUNK", "50"))
        for i in range(0, len(all_ids), chunk):
            books.update(self._fetch_books(all_ids[i:i+chunk]) or {})
        have = len(books)
        need = len(set(all_ids))
        self.logger.info(f"[CLOB] books coverage {have}/{need} tokens")
        # 3) Fallback: GET /book for any missing
        missing = [tid for tid in all_ids if tid not in books]
        for tid in missing:
            try:
                r = get(f"{self.clob_base}/book", params={"token_id": tid})
                if r and r.status_code == 200:
                    ob = r.json() or {}
                    bids = ob.get("bids") or []
                    asks = ob.get("asks") or []
                    best_bid, bb_sz = self._best(bids, best=True)
                    best_ask, ba_sz = self._best(asks, best=False)
                    mid = (best_bid + best_ask)/2 if (best_bid and best_ask) else (best_bid or best_ask or 0.0)
                    books[tid] = {"best_bid": best_bid, "best_bid_size": bb_sz,
                                  "best_ask": best_ask, "best_ask_size": ba_sz, "midpoint": mid}
            except Exception:
                continue
        # 4) Write explicit fields per side onto each contract
        for c in contracts or []:
            y, n = self._extract_token_ids(c)
            for side, tid in (("yes", y), ("no", n)):
                if tid and tid in books:
                    b = books[tid]
                    c[f"{side}_best_bid"] = b.get("best_bid", 0.0)
                    c[f"{side}_best_bid_size"] = b.get("best_bid_size", 0.0)
                    c[f"{side}_best_ask"] = b.get("best_ask", 0.0)
                    c[f"{side}_best_ask_size"] = b.get("best_ask_size", 0.0)
        return contracts

    def filter_two_sided_markets(self, contracts: List[Dict], *, min_size: float = 0.0, min_notional: float = 0.0) -> List[Dict]:
        """
        Keep a contract only if BOTH tokens (YES and NO) have a two-sided book:
          best_bid>0 with size>min_size and best_ask>0 with size>min_size,
          and optionally notional >= min_notional on both sides.
        """
        def ok(side: str, c: Dict) -> bool:
            bb = float(c.get(f"{side}_best_bid", 0.0) or 0.0)
            ba = float(c.get(f"{side}_best_ask", 0.0) or 0.0)
            bq = float(c.get(f"{side}_best_bid_size", 0.0) or 0.0)
            aq = float(c.get(f"{side}_best_ask_size", 0.0) or 0.0)
            if bb <= 0 or ba <= 0: return False
            if not (bq > min_size and aq > min_size): return False
            if min_notional > 0.0 and ((bb*bq) < min_notional or (ba*aq) < min_notional): return False
            return True
        out = []
        for c in contracts or []:
            if ok("yes", c) and ok("no", c):
                out.append(c)
        return out

    # ---- helpers ----
    def _extract_token_ids(self, c: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        Be liberal in what we accept; support multiple shapes coming from Gamma or simplified markets:
          - yes_token_id / no_token_id
          - clob_token_ids: [YES, NO]
          - tokens: [{'token_id': ...}, {'token_id': ...}]
        """
        yes_id = c.get("yes_token_id") or c.get("yesTokenId")
        no_id  = c.get("no_token_id")  or c.get("noTokenId")
        if (not yes_id or not no_id) and isinstance(c.get("clob_token_ids"), (list, tuple)):
            ids = list(c["clob_token_ids"])
            if len(ids) >= 2:
                yes_id = yes_id or ids[0]
                no_id  = no_id  or ids[1]
        # Accept camelCase / JSON-string form at the contract level too
        if (not yes_id or not no_id) and c.get("clobTokenIds"):
            raw = c.get("clobTokenIds")
            try:
                import json as _json
                ids = raw if isinstance(raw, list) else _json.loads(raw)
                if isinstance(ids, list) and len(ids) >= 2:
                    yes_id = yes_id or ids[0]
                    no_id  = no_id  or ids[1]
            except Exception:
                pass
        if (not yes_id or not no_id) and isinstance(c.get("tokens"), (list, tuple)):
            toks = list(c["tokens"])
            if len(toks) >= 2:
                # assume index 0 is YES per simplified schema examples
                yes_id = yes_id or toks[0].get("token_id") or toks[0].get("id")
                no_id  = no_id  or toks[1].get("token_id") or toks[1].get("id")
        return (str(yes_id) if yes_id else None, str(no_id) if no_id else None)

    # ---------- Token resolution helpers ----------
    def _resolve_yes_no_token_ids(self, market: dict) -> tuple[Optional[str], Optional[str]]:
        """
        Obtain (YES_token_id, NO_token_id) robustly:
          1) market['tokens'] with outcome labels
          2) clobTokenIds / clob_token_ids (list or JSON string); if unlabeled and conditionId present, query CLOB
        """
        # 1) Direct tokens array (best; used by /simplified-markets and CLOB /markets/<conditionId>)
        tokens = market.get("tokens") or []
        if isinstance(tokens, list) and tokens:
            yes_id = no_id = None
            for t in tokens:
                outcome = str(t.get("outcome", "")).lower()
                tid = t.get("token_id") or t.get("id") or t.get("asset_id")
                if not tid:
                    continue
                if outcome == "yes":
                    yes_id = str(tid)
                elif outcome == "no":
                    no_id = str(tid)
            if yes_id or no_id:
                return yes_id, no_id

        # 2) clobTokenIds / clob_token_ids (may be JSON string)
        raw_pair = market.get("clobTokenIds") or market.get("clob_token_ids")
        pair_ids: list[str] = []
        if raw_pair:
            try:
                import json as _json
                pair = raw_pair if isinstance(raw_pair, list) else _json.loads(raw_pair)
                if isinstance(pair, list):
                    pair_ids = [str(x) for x in pair if x]
            except Exception:
                pass

        # If we still don't have labels, try CLOB: GET /markets/<conditionId>
        condition_id = market.get("condition_id") or market.get("conditionId")
        if condition_id:
            try:
                r = get(f"{self.clob_base}/markets/{condition_id}")
                if r and r.status_code == 200:
                    mm = r.json() or {}
                    tkns = mm.get("tokens") or []
                    yes_id = no_id = None
                    for t in tkns:
                        outcome = str(t.get("outcome", "")).lower()
                        tid = t.get("token_id") or t.get("id") or t.get("asset_id")
                        if not tid:
                            continue
                        if outcome == "yes":
                            yes_id = str(tid)
                        elif outcome == "no":
                            no_id = str(tid)
                    if yes_id or no_id:
                        return yes_id, no_id
            except Exception:
                pass

        if len(pair_ids) >= 2:
            # Return the pair even if unlabeled (downstream can still fetch books for both)
            return pair_ids[0], pair_ids[1]
        return None, None

    def _best(self, levels, *, best=True) -> Tuple[float, float]:
        """
        Return (price, size) for best level from a list of {price,size} dicts (strings per CLOB docs).
        For bids → max price; for asks → min price.
        """
        if not levels:
            return (0.0, 0.0)
        try:
            if best:
                lvl = max(levels, key=lambda x: float(x.get("price")))
            else:
                lvl = min(levels, key=lambda x: float(x.get("price")))
            return float(lvl.get("price")), float(lvl.get("size"))
        except Exception:
            return (0.0, 0.0)
