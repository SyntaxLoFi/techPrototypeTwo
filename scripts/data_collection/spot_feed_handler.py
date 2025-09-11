"""
Spot price feed handler - now using Binance as canonical source
"""
import logging
from typing import Dict, Optional

from .binance_spot_integration import get_spot_price


class SpotFeedHandler:
    """Handle spot price feed data"""
    
    def __init__(self, currency: str):
        self.currency = currency
        self.logger = logging.getLogger(f'SpotFeedHandler_{currency}')
        self.current_price = None
        self.price_history = []
        
    def process_message(self, message: Dict) -> Optional[Dict]:
        """Process spot feed message"""
        if message.get("method") != "subscription":
            return None
        
        params = message.get("params", {})
        channel = params.get("channel", "")
        
        if channel != f"spot_feed.{self.currency}":
            return None
        
        data = params.get("data", {})
        feeds = data.get("feeds", {})
        
        if self.currency in feeds:
            feed_data = feeds[self.currency]
            price = float(feed_data.get("price", 0))
            confidence = float(feed_data.get("confidence", 0))
            timestamp = data.get("timestamp", 0) / 1000
            
            self.current_price = price
            
            result = {
                'currency': self.currency,
                'oracle_price': price,
                'confidence': confidence,
                'timestamp': timestamp
            }
            
            self.price_history.append(result)
            
            # Keep only recent history (last 1000 points)
            if len(self.price_history) > 1000:
                self.price_history = self.price_history[-1000:]
            
            return result
        
        return None
    
    def get_current_price(self) -> Optional[float]:
        """
        Get current spot price - now delegates to Binance as canonical source
        
        Returns:
            Current spot price from Binance or None if unavailable
        """
        try:
            # Delegate to Binance as canonical source
            price = get_spot_price(self.currency)
            
            # Update internal state for compatibility
            if price is not None:
                self.current_price = price
                
            return price
        except Exception as e:
            self.logger.error(f"Failed to get Binance spot price: {e}")
            return None