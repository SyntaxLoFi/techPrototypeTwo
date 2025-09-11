"""
Ticker handler for real-time price updates from Derive/Lyra
Handles ticker.{instrument}.{interval} channel subscriptions
"""
import time
import logging
from typing import Dict, Optional, Set
from collections import defaultdict


class TickerHandler:
    """Handle ticker data from WebSocket subscriptions"""
    
    def __init__(self):
        self.logger = logging.getLogger('TickerHandler')
        self.latest_tickers = {}  # instrument -> latest ticker data
        self.ticker_history = defaultdict(list)
        self.subscribed_instruments = set()
        self.last_cleanup = time.time()
        
    def process_message(self, message: Dict) -> Optional[Dict]:
        """Process ticker message from WebSocket"""
        if message.get("method") != "subscription":
            return None
        
        params = message.get("params", {})
        channel = params.get("channel", "")
        
        if not channel.startswith("ticker."):
            return None
        
        # Extract instrument name and interval
        # Format: ticker.{instrument}.{interval}
        parts = channel.split(".")
        if len(parts) < 3:
            return None
        
        instrument = parts[1]
        interval = parts[2]  # Usually 100ms
        
        data = params.get("data", {})
        
        # Parse ticker data
        ticker = self._parse_ticker(instrument, data)
        
        if ticker:
            # Store latest ticker
            self.latest_tickers[instrument] = ticker
            
            # Store history
            self.ticker_history[instrument].append(ticker)
            
            # Keep only last 1000 entries per instrument
            if len(self.ticker_history[instrument]) > 1000:
                self.ticker_history[instrument] = self.ticker_history[instrument][-1000:]
            
            # Log significant price changes
            self._log_price_changes(instrument, ticker)
            
            # Cleanup old data periodically
            if time.time() - self.last_cleanup > 300:  # Every 5 minutes
                self.cleanup_old_data()
        
        return ticker
    
    def _parse_ticker(self, instrument: str, data: Dict) -> Optional[Dict]:
        """Parse ticker data into standardized format"""
        try:
            timestamp = data.get("timestamp", time.time() * 1000) / 1000
            
            # Price data
            mark_price = float(data.get("mark_price", 0))
            best_bid = float(data.get("best_bid_price", 0))
            best_ask = float(data.get("best_ask_price", 0))
            last_price = float(data.get("last_price", 0))
            
            # Volume data
            stats = data.get("stats", {}) or {}
            volume_24h_contracts = float(stats.get("contract_volume", 0))
            
            # For options - IV and greeks are in option_pricing per Derive docs
            option_pricing = data.get("option_pricing", {}) or {}
            implied_volatility = None
            delta = None
            gamma = None
            vega = None
            theta = None
            
            if option_pricing:
                # Extract IV and Greeks from option_pricing
                implied_volatility = float(option_pricing.get("iv") or 0)
                delta = float(option_pricing.get("delta") or 0)
                gamma = float(option_pricing.get("gamma") or 0)
                vega = float(option_pricing.get("vega") or 0)
                theta = float(option_pricing.get("theta") or 0)
                
                # Also available: ask_iv, bid_iv, mark_price (for options)
            
            # For perpetuals
            perp_details = data.get("perp_details", {})
            funding_rate = None
            
            if perp_details:
                funding_rate = float(perp_details.get("funding_rate", 0))
            
            # Calculate derived metrics (gracefully handle one-sided books)
            if best_bid and best_ask:
                mid = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
            elif best_bid:
                mid, spread = best_bid, 0
            elif best_ask:
                mid, spread = best_ask, 0
            else:
                mid, spread = mark_price, 0
            spread_pct = (spread / mid * 100) if mid > 0 else 0
            
            return {
                'instrument': instrument,
                'timestamp': timestamp,
                'mark_price': mark_price,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'last_price': last_price,
                'mid': mid,
                'spread': spread,
                'spread_pct': spread_pct,
                'volume_24h': volume_24h_contracts,
                'volume_24h_contracts': volume_24h_contracts,  # Add both keys for compatibility
                # Options specific
                'implied_volatility': implied_volatility,
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta,
                # Perps specific
                'funding_rate': funding_rate,
                # Instrument type - check option_pricing not option_details
                'instrument_type': 'option' if option_pricing else ('perp' if perp_details else 'unknown')
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing ticker for {instrument}: {e}")
            return None
    
    def _log_price_changes(self, instrument: str, ticker: Dict):
        """Log significant price changes"""
        history = self.ticker_history.get(instrument, [])
        if len(history) < 2:
            return
        
        prev_ticker = history[-2]
        
        # Check for significant mark price change
        prev_mark = prev_ticker.get('mark_price', 0)
        curr_mark = ticker.get('mark_price', 0)
        
        if prev_mark > 0:
            price_change_pct = abs((curr_mark - prev_mark) / prev_mark * 100)
            
            if price_change_pct > 1.0:  # More than 1% change
                self.logger.info(
                    f"{instrument} price change: ${prev_mark:.2f} -> ${curr_mark:.2f} "
                    f"({price_change_pct:.2f}%)"
                )
        
        # Check for significant IV change (options only)
        if ticker.get('instrument_type') == 'option':
            prev_iv = prev_ticker.get('implied_volatility', 0)
            curr_iv = ticker.get('implied_volatility', 0)
            
            if prev_iv > 0 and curr_iv > 0:
                iv_change = abs(curr_iv - prev_iv)
                
                if iv_change > 0.05:  # More than 5 vol points
                    self.logger.info(
                        f"{instrument} IV change: {prev_iv:.1%} -> {curr_iv:.1%} "
                        f"({iv_change:.1%} change)"
                    )
    
    def get_latest_ticker(self, instrument: str) -> Optional[Dict]:
        """Get latest ticker data for an instrument"""
        return self.latest_tickers.get(instrument)
    
    def get_all_tickers(self) -> Dict[str, Dict]:
        """Get all latest tickers"""
        return self.latest_tickers.copy()
    
    def get_subscribed_instruments(self) -> Set[str]:
        """Get set of subscribed instruments"""
        return self.subscribed_instruments.copy()
    
    def add_subscription(self, instrument: str):
        """Track a new subscription"""
        self.subscribed_instruments.add(instrument)
        self.logger.info(f"Added ticker subscription for {instrument}")
    
    def remove_subscription(self, instrument: str):
        """Remove a subscription"""
        self.subscribed_instruments.discard(instrument)
        self.logger.info(f"Removed ticker subscription for {instrument}")
    
    def cleanup_old_data(self):
        """Clean up old data to manage memory"""
        current_time = time.time()
        cutoff_time = current_time - (60 * 60)  # Keep last hour
        
        # Clean up history
        for instrument in list(self.ticker_history.keys()):
            # Filter out old entries
            self.ticker_history[instrument] = [
                t for t in self.ticker_history[instrument]
                if t['timestamp'] > cutoff_time
            ]
            
            # Remove instrument if no recent data
            if not self.ticker_history[instrument]:
                del self.ticker_history[instrument]
                if instrument in self.latest_tickers:
                    del self.latest_tickers[instrument]
        
        self.last_cleanup = current_time
        self.logger.info(f"Cleaned up ticker data, tracking {len(self.latest_tickers)} instruments")
    
    def get_funding_rates(self) -> Dict[str, float]:
        """Get current funding rates for all perpetuals"""
        funding_rates = {}
        
        for instrument, ticker in self.latest_tickers.items():
            if ticker.get('instrument_type') == 'perp' and ticker.get('funding_rate') is not None:
                funding_rates[instrument] = ticker['funding_rate']
        
        return funding_rates
    
    def get_options_ivs(self) -> Dict[str, float]:
        """Get current implied volatilities for all options"""
        ivs = {}
        
        for instrument, ticker in self.latest_tickers.items():
            if ticker.get('instrument_type') == 'option' and ticker.get('implied_volatility'):
                ivs[instrument] = ticker['implied_volatility']
        
        return ivs