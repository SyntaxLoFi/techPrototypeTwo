"""
Orderbook handler with memory management
"""
import time
import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict

from config_manager import (
    DATA_RETENTION_MINUTES,
    ALLOWED_ORDERBOOK_GROUPS, ALLOWED_ORDERBOOK_DEPTHS
)
from decimal import Decimal, InvalidOperation


class OrderbookHandler:
    """Handle orderbook data with automatic cleanup"""
    
    def __init__(self, spot_provider=None):
        self.logger = logging.getLogger('OrderbookHandler')
        self.latest_orderbooks = {}
        self.orderbook_history = defaultdict(list)
        self.subscribed_instruments = set()
        self.last_cleanup = time.time()
        self.spot_provider = spot_provider  # Inject spot price provider (Binance)
        
    def process_message(self, message: Dict, oracle_price: Optional[float] = None) -> Optional[Dict]:
        """Process orderbook message"""
        if message.get("method") != "subscription":
            return None
        
        params = message.get("params", {})
        channel = params.get("channel", "")
        
        if not channel.startswith("orderbook."):
            return None
        
        # Extract instrument name
        parts = channel.split(".")
        if len(parts) < 2:
            return None
        
        instrument = parts[1]
        data = params.get("data", {})
        
        # Parse orderbook
        orderbook = self._parse_orderbook(instrument, data, oracle_price)
        
        if orderbook:
            # Store latest
            self.latest_orderbooks[instrument] = orderbook
            
            # Store history
            self.orderbook_history[instrument].append(orderbook)
            
            # Cleanup old data periodically
            if time.time() - self.last_cleanup > 300:  # Every 5 minutes
                self.cleanup_old_data()
        
        return orderbook
    
    def _parse_orderbook(self, instrument: str, data: Dict, oracle_price: Optional[float]) -> Optional[Dict]:
        """Parse orderbook data"""
        try:
            timestamp = data.get("timestamp", time.time() * 1000) / 1000
            bids = (data.get("bids", []) or [])
            asks = (data.get("asks", []) or [])
            # Accept one‑sided books; options are often bid‑only or ask‑only on Lyra.
            
            # Best bid/ask (use Decimal for precision, cast to float for output)
            def _to_dec(x):
                try:
                    return Decimal(str(x))
                except (InvalidOperation, TypeError):
                    return Decimal(0)
            best_bid = _to_dec(bids[0][0]) if bids else Decimal(0)
            best_ask = _to_dec(asks[0][0]) if asks else Decimal(0)
            best_bid_qty = _to_dec(bids[0][1]) if bids else Decimal(0)
            best_ask_qty = _to_dec(asks[0][1]) if asks else Decimal(0)
            
            # Calculate metrics
            both_sides = (best_bid > 0 and best_ask > 0)
            mid = (best_bid + best_ask) / Decimal(2) if both_sides else Decimal(0)
            spread = (best_ask - best_bid) if both_sides else Decimal(0)
            spread_pct = (spread / mid * Decimal(100)) if both_sides and mid > 0 else Decimal(0)
            
            # Total liquidity
            bid_liquidity = sum(_to_dec(b[0]) * _to_dec(b[1]) for b in bids[:5]) if bids else Decimal(0)
            ask_liquidity = sum(_to_dec(a[0]) * _to_dec(a[1]) for a in asks[:5]) if asks else Decimal(0)
            
            # Get oracle price from spot provider if not provided
            if oracle_price is None and self.spot_provider:
                try:
                    # Extract base currency from instrument (e.g., "ETH" from "ETH-PERP" or "ETH-20240131-3000-C")
                    base_currency = instrument.split("-")[0]
                    oracle_price = self.spot_provider(base_currency)
                except Exception as e:
                    self.logger.debug(f"Failed to get oracle price for {instrument}: {e}")
                    oracle_price = None
            
            return {
                'instrument': instrument,
                'timestamp': float(timestamp),
                'best_bid': float(best_bid),
                'best_ask': float(best_ask),
                'best_bid_qty': float(best_bid_qty),
                'best_ask_qty': float(best_ask_qty),
                'mid': float(mid),
                'spread': float(spread),
                'spread_pct': float(spread_pct),
                'bid_depth': len(bids),
                'ask_depth': len(asks),
                'bid_liquidity': float(bid_liquidity),
                'ask_liquidity': float(ask_liquidity),
                'liquidity': float(bid_liquidity + ask_liquidity),
                'has_bid': bool(bids),
                'has_ask': bool(asks),
                'oracle_price': oracle_price
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing orderbook for {instrument}: {e}")
            return None
    
    def get_latest_orderbook(self, instrument: str) -> Optional[Dict]:
        """Get latest orderbook for an instrument"""
        return self.latest_orderbooks.get(instrument)
    
    def compute_mid_spread(self, best_bid: float, best_ask: float) -> tuple:
        """Compute mid price and spread from bid/ask"""
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        return mid, spread
    
    def cleanup_old_data(self):
        """Remove old data to prevent memory issues"""
        cutoff = time.time() - (DATA_RETENTION_MINUTES * 60)
        
        for instrument in list(self.orderbook_history.keys()):
            # Keep only recent data
            self.orderbook_history[instrument] = [
                ob for ob in self.orderbook_history[instrument]
                if ob['timestamp'] > cutoff
            ]
            
            # Remove empty entries
            if not self.orderbook_history[instrument]:
                del self.orderbook_history[instrument]
        
        self.last_cleanup = time.time()
        self.logger.debug(f"Cleaned up orderbook data, retained {len(self.orderbook_history)} instruments")
    
    def get_subscription_messages(self, instruments: Set[str], group: int = 1, depth: int = 20) -> List[Dict]:
        """
        Generate subscription messages for orderbook channels
        """
        if not instruments:
            return []
        # Validate per Derive channel spec
        if group not in ALLOWED_ORDERBOOK_GROUPS:
            group = min(ALLOWED_ORDERBOOK_GROUPS)
        if depth not in ALLOWED_ORDERBOOK_DEPTHS:
            depth = 20  # sensible default

        instrument_list = list(instruments)
        batch_size = 20
        messages = []
        
        for i in range(0, len(instrument_list), batch_size):
            batch = instrument_list[i:i + batch_size]
            # Format: orderbook.{instrument}.{group}.{depth}
            channels = [f"orderbook.{inst}.{group}.{depth}" for inst in batch]
            
            messages.append({
                "method": "subscribe",
                "params": {"channels": channels},
                "id": f"orderbook_batch_{i}"
            })
        
        self.subscribed_instruments.update(instruments)
        self.logger.info(f"Generated orderbook subscriptions for {len(instruments)} instruments with group={group}, depth={depth}")
        return messages