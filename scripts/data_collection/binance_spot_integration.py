#!/usr/bin/env python
"""
Binance Spot Price Integration for Arbitrage Scanner
Maintains real-time crypto prices via WebSocket with time series logging
Supports multiple currencies (ETH, BTC, SOL, XRP, DOGE)
"""

import asyncio
import json
import websockets
import ssl
import certifi
from datetime import datetime
import threading
import time
from typing import Optional, Dict, Any
import logging
import pandas as pd
import os
from pathlib import Path
import requests
import os
from config_manager import (
    SUPPORTED_CURRENCIES, BINANCE_WS_URL, BINANCE_WS_FALLBACK, BINANCE_WS_US, BINANCE_SYMBOLS, SPOT_STALE_SECONDS,
    BINANCE_REST_PRIMARY, BINANCE_REST_SECONDARY, BINANCE_REST_HOSTS, HTTP_TIMEOUT_SEC, HTTP_RETRY_MAX
)
from utils.http_client import get, post

# Import WebSocket tracker
try:
    from utils.websocket_tracker import track_websocket, untrack_websocket
except ImportError:
    # Fallback if tracker not available
    def track_websocket(ws): pass
    def untrack_websocket(ws): pass

# Preferred public REST endpoint (market-data-only), with fallback
BINANCE_REST_API_PRIMARY = BINANCE_REST_PRIMARY
BINANCE_REST_API_FALLBACK = BINANCE_REST_SECONDARY

logger = logging.getLogger(__name__)

# Sticky global preferred WS URL once we detect geo restriction
_WS_URL_OVERRIDE = None


def get_binance_symbol(currency):
    """Get Binance trading symbol for currency"""
    return BINANCE_SYMBOLS.get(currency.upper(), f"{currency.upper()}USDT")


def get_spot_prices_path(currency):
    """Get path for spot price data"""
    return f"data/{currency.upper()}/spot_prices"


def validate_currency(currency):
    """Validate and normalize currency"""
    currency = currency.upper()
    if currency not in SUPPORTED_CURRENCIES:
        raise ValueError(f"Unsupported currency: {currency}")
    return currency


def fetch_spot_price_rest(currency: str, retries: int = HTTP_RETRY_MAX, timeout: int = HTTP_TIMEOUT_SEC) -> Optional[Dict[str, Any]]:
    """
    Fetch spot price via Binance REST API as fallback
    
    Args:
        currency: Currency code (ETH, BTC, etc.)
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        
    Returns:
        Dict with price data or None if failed
    """
    currency = validate_currency(currency)
    symbol = get_binance_symbol(currency)
    
    params = {"symbol": symbol}
    
    # Try each host in rotation
    for attempt, host in enumerate(BINANCE_REST_HOSTS[:retries]):
        try:
            price_resp = get(f"{host}/api/v3/ticker/price", params=params)
            
            if price_resp.status_code == 200:
                data = price_resp.json()
                price = float(data.get("price", 0))
                
                if price > 0:
                    # Fetch bookTicker for bid/ask (lighter than 24hr)
                    book_resp = get(f"{host}/api/v3/ticker/bookTicker", params=params)
                    
                    result = {
                        "price": price,
                        "source": "binance_rest",
                        "timestamp": time.time()
                    }
                    
                    if book_resp.status_code == 200:
                        book_data = book_resp.json()
                        result.update({
                            "bid": float(book_data.get("bidPrice", 0)),
                            "ask": float(book_data.get("askPrice", 0)),
                            "bid_qty": float(book_data.get("bidQty", 0)),
                            "ask_qty": float(book_data.get("askQty", 0))
                        })
                    
                    logger.info(f"Fetched {currency} price via Binance REST: ${price:.2f} (host: {host})")
                    return result
            
            elif price_resp.status_code == 429:  # Rate limited
                logger.warning("Binance rate limit hit, backing off...")
                time.sleep(2 ** attempt)  # Exponential backoff
            
            else:
                logger.error(f"Binance REST API error: {price_resp.status_code} - {price_resp.text}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"Binance REST API timeout (attempt {attempt + 1}/{retries})")
            
        except requests.exceptions.ConnectionError:
            logger.warning(f"Binance REST API connection error (attempt {attempt + 1}/{retries})")
            
        except Exception as e:
            logger.error(f"Unexpected error fetching Binance price: {e}")
            
        if attempt < retries - 1:
            time.sleep(1)  # Brief pause between retries
    
    logger.error(f"Failed to fetch {currency} price via REST API after {retries} attempts")
    return None


class BinanceSpotPriceFeed:
    """Maintains real-time Binance spot prices via WebSocket with logging for any supported currency"""
    
    def __init__(self, currency: str = "ETH", log_spot_prices: bool = True):
        """
        Initialize Binance spot price feed for specified currency
        
        Args:
            currency: Currency code (ETH, BTC, etc.)
            log_spot_prices: Whether to log prices to CSV
        """
        self.currency = validate_currency(currency)
        self.symbol = get_binance_symbol(self.currency).lower()
        # Use combined stream for both ticker and bookTicker for best accuracy
        # ticker gives 24h stats, bookTicker gives real-time best bid/ask
        # All stream symbols must be lowercase (Binance WS spec)
        
        # List of WebSocket URLs to try in order (primary, fallback, US)
        self.ws_urls = [
            BINANCE_WS_URL,
            BINANCE_WS_FALLBACK,
            BINANCE_WS_US
        ]
        # If a global preferred URL has been set (e.g., after one 451), start there
        global _WS_URL_OVERRIDE
        if _WS_URL_OVERRIDE and _WS_URL_OVERRIDE in self.ws_urls:
            self.current_url_index = self.ws_urls.index(_WS_URL_OVERRIDE)
            logger.info(f"Using global preferred Binance WS host: {_WS_URL_OVERRIDE}")
        else:
            # Check environment variable first
            env_ws_host = os.getenv("BINANCE_WS_HOST")
            if env_ws_host and env_ws_host in self.ws_urls:
                self.current_url_index = self.ws_urls.index(env_ws_host)
                logger.info(f"Using BINANCE_WS_HOST from environment: {env_ws_host}")
            else:
                self.current_url_index = 0
        self.stream_path = f"/stream?streams={self.symbol.lower()}@ticker/{self.symbol.lower()}@bookTicker"
        
        self.current_price = None
        self.last_update = None
        self.is_connected = False
        self.reconnect_delay = 5
        self.reconnect_attempts = 0
        self.max_reconnect_delay = 60  # Cap at 60 seconds
        self._stop_event = threading.Event()
        self._thread = None
        self._loop = None
        self.log_spot_prices = log_spot_prices
        self.session_start = time.time()
        
        # Connection health monitoring
        self.last_ping_time = None
        self.last_pong_time = None
        self.connection_start_time = None
        self.total_messages_received = 0
        self.connection_errors = []
        
        # Price data
        self.price_data = {
            'price': None,
            'bid': None,
            'ask': None,
            'volume_24h': None,
            'high_24h': None,
            'low_24h': None,
            'change_24h': None,
            'last_update': None
        }
        
        # Set up spot price logging
        if self.log_spot_prices:
            self._setup_spot_price_logging()
        
    def _setup_spot_price_logging(self):
        """Set up CSV logging for spot prices in currency-specific folder"""
        # Create currency-specific directory for spot price data
        self.spot_data_dir = Path(get_spot_prices_path(self.currency))
        self.spot_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create CSV filename with currency and date
        date_str = datetime.now().strftime("%Y%m%d")
        self.spot_price_file = self.spot_data_dir / f"{self.symbol}_spot_prices_{date_str}.csv"
        
        # Create file with headers if it doesn't exist
        if not self.spot_price_file.exists():
            df = pd.DataFrame(columns=['timestamp', 'price', 'bid', 'ask', 'volume_24h'])
            df.to_csv(self.spot_price_file, index=False)
            logger.info(f"Created spot price log for {self.currency}: {self.spot_price_file}")
        
    def _log_spot_price(self):
        """Log current spot price to CSV"""
        if not self.log_spot_prices or self.current_price is None:
            return
            
        try:
            # Prepare row data
            row_data = {
                'timestamp': datetime.now(),
                'price': self.current_price,
                'bid': self.price_data.get('bid'),
                'ask': self.price_data.get('ask'),
                'volume_24h': self.price_data.get('volume_24h')
            }
            
            # Append to CSV
            df = pd.DataFrame([row_data])
            df.to_csv(self.spot_price_file, mode='a', header=False, index=False)
            
        except Exception as e:
            logger.error(f"Failed to log spot price: {e}")
        
    def start(self):
        """Start the WebSocket connection in a background thread"""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_event_loop)
            self._thread.daemon = True
            self._thread.start()
            
            # Wait for initial connection
            timeout = 10
            start = time.time()
            while not self.is_connected and time.time() - start < timeout:
                time.sleep(0.1)
                
            if not self.is_connected:
                logger.error(f"Failed to connect to Binance WebSocket for {self.currency} within timeout")
                raise ConnectionError(f"Cannot establish Binance WebSocket connection for {self.currency}")
            else:
                logger.info(f"Binance WebSocket connected for {self.currency} ({self.symbol.upper()})")
    
    def stop(self):
        """Stop the WebSocket connection"""
        self._stop_event.set()
        
        # Cancel any pending tasks in the loop
        if hasattr(self, '_loop') and self._loop and not self._loop.is_closed():
            # Schedule the loop to stop
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning(f"Binance WebSocket thread for {self.currency} did not terminate within timeout")
        
        self.is_connected = False
        logger.info(f"Binance spot price feed stopped for {self.currency}")
    
    def _run_event_loop(self):
        """Run the asyncio event loop in a thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._maintain_connection())
        finally:
            # Cancel all pending tasks before closing the loop
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                logger.debug(f"Error canceling tasks: {e}")
            
            # Shutdown async generators
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception as e:
                logger.debug(f"Error shutting down async generators: {e}")
            
            self._loop.close()
    
    async def _maintain_connection(self):
        """Maintain WebSocket connection with auto-reconnect and URL fallback"""
        while not self._stop_event.is_set():
            try:
                # Reset connection monitoring
                self.connection_start_time = time.time()
                self.total_messages_received = 0
                
                await self._connect_and_stream()
                
                # Reset reconnect delay on successful connection
                self.reconnect_delay = 5
                self.reconnect_attempts = 0
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.is_connected = False
                self.reconnect_attempts += 1
                
                # Store error for diagnostics
                self.connection_errors.append({
                    'time': time.time(),
                    'error': str(e),
                    'url': self.ws_urls[self.current_url_index],
                    'attempt': self.reconnect_attempts
                })
                
                # Keep only last 10 errors
                if len(self.connection_errors) > 10:
                    self.connection_errors.pop(0)
                
                # Check if it's a 451 error (geographic restriction)
                error_str = str(e).lower()
                if "451" in error_str or "unavailable for legal reasons" in error_str:
                    # Jump directly to binance.us if present; otherwise advance to next
                    try:
                        us_idx = self.ws_urls.index(BINANCE_WS_US)
                        self.current_url_index = us_idx
                    except ValueError:
                        self.current_url_index = (self.current_url_index + 1) % len(self.ws_urls)
                    # Remember globally so subsequent feeds start on this host
                    global _WS_URL_OVERRIDE
                    _WS_URL_OVERRIDE = self.ws_urls[self.current_url_index]
                    logger.info(
                        f"Geographic restriction detected, switching to {_WS_URL_OVERRIDE} "
                        f"and caching as global preferred host"
                    )
                    # Don't delay for geographic restrictions, try immediately
                    continue
                
                if not self._stop_event.is_set():
                    # Implement exponential backoff
                    self.reconnect_delay = min(
                        self.reconnect_delay * 1.5,  # Increase by 50%
                        self.max_reconnect_delay
                    )
                    logger.info(f"Reconnecting in {self.reconnect_delay:.1f} seconds (attempt {self.reconnect_attempts})...")
                    
                    # Sleep with periodic checks for stop event
                    sleep_interval = 0.5  # Check every 500ms
                    remaining = self.reconnect_delay
                    while remaining > 0 and not self._stop_event.is_set():
                        sleep_time = min(sleep_interval, remaining)
                        await asyncio.sleep(sleep_time)
                        remaining -= sleep_time
    
    async def _connect_and_stream(self):
        """Connect to WebSocket and stream prices"""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Build the full URL with the current WebSocket endpoint
        current_url = self.ws_urls[self.current_url_index] + self.stream_path
        
        websocket = None
        secondary_websocket = None
        
        # Enhanced connection parameters for better stability
        # - open_timeout: Maximum time to wait for connection establishment
        # - ping_interval: More frequent pings to detect dead connections faster
        # - ping_timeout: Shorter timeout for ping responses
        # - close_timeout: Time to wait for graceful close
        try:
            async with websockets.connect(
                current_url, 
                ssl=ssl_context, 
                open_timeout=10,      # 10s to establish connection
                ping_interval=10,     # Send ping every 10s (was 20s)
                ping_timeout=5,       # Wait 5s for pong response
                close_timeout=10      # 10s for graceful close
            ) as websocket:
                # Track this WebSocket connection
                track_websocket(websocket)
                self.is_connected = True
                self.session_start = time.time()  # Reset session timer on connect
                connection_duration = time.time() - self.connection_start_time if self.connection_start_time else 0
                logger.info(f"Connected to Binance WebSocket ({self.ws_urls[self.current_url_index]}): {self.symbol.upper()} (took {connection_duration:.1f}s)")
                
                # Track successful connection
                self.last_ping_time = time.time()
                self.last_pong_time = time.time()
                
                while not self._stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                        await self._process_message(message)
                        self.total_messages_received += 1
                        
                        # Update ping tracking (WebSocket library handles pings internally)
                        self.last_pong_time = time.time()
                        
                    except asyncio.TimeoutError:
                        # Check connection health
                        time_since_last_message = time.time() - self.last_pong_time
                        if time_since_last_message > 90:  # No messages for 90s
                            logger.warning(f"No messages received for {time_since_last_message:.1f}s, connection may be stale")
                            raise websockets.ConnectionClosed(1000, "Connection appears stale")
                        
                        # Periodic reconnect safeguard (~23h) to avoid 24h server close
                        if time.time() - self.session_start > 23 * 3600:
                            logger.info("Proactive reconnect for 24h session limit")
                            raise websockets.ConnectionClosed(1000, "Session rollover")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        raise
                # Graceful shutdown when stop event is set
                if self._stop_event.is_set() and websocket:
                    untrack_websocket(websocket)
                    await websocket.close()
        except websockets.InvalidStatusCode as e:
            code = getattr(e, "status_code", None)
            if code == 451:
                logger.warning("WebSocket error: server rejected WebSocket connection: HTTP 451")
                logger.info("Geographic restriction detected, switching to %s and caching as global preferred host", BINANCE_WS_US)
                
                # Update global and environment variable
                global _WS_URL_OVERRIDE
                _WS_URL_OVERRIDE = BINANCE_WS_US
                os.environ["BINANCE_WS_HOST"] = BINANCE_WS_US  # sticky for current process
                
                # Update index to US host
                if BINANCE_WS_US in self.ws_urls:
                    self.current_url_index = self.ws_urls.index(BINANCE_WS_US)
                
                # Retry with the US host
                current_url = self.ws_urls[self.current_url_index] + self.stream_path
                secondary_websocket = await websockets.connect(
                    current_url, 
                    ssl=ssl_context, 
                    open_timeout=10,
                    ping_interval=10,
                    ping_timeout=5,
                    close_timeout=10
                )
                # Track this WebSocket connection
                track_websocket(secondary_websocket)
                try:
                    self.is_connected = True
                    self.session_start = time.time()
                    connection_duration = time.time() - self.connection_start_time if self.connection_start_time else 0
                    logger.info(f"Connected to Binance WebSocket ({self.ws_urls[self.current_url_index]}): {self.symbol.upper()} (took {connection_duration:.1f}s)")
                    
                    # Track successful connection
                    self.last_ping_time = time.time()
                    self.last_pong_time = time.time()
                    
                    while not self._stop_event.is_set():
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                            await self._process_message(message)
                            self.total_messages_received += 1
                            
                            # Update ping tracking (WebSocket library handles pings internally)
                            self.last_pong_time = time.time()
                            
                        except asyncio.TimeoutError:
                            # Check connection health
                            time_since_last_message = time.time() - self.last_pong_time
                            if time_since_last_message > 90:  # No messages for 90s
                                logger.warning(f"No messages received for {time_since_last_message:.1f}s, connection may be stale")
                                raise websockets.ConnectionClosed(1000, "Connection appears stale")
                            
                            # Periodic reconnect safeguard (~23h) to avoid 24h server close
                            if time.time() - self.session_start > 23 * 3600:
                                logger.info("Proactive reconnect for 24h session limit")
                                raise websockets.ConnectionClosed(1000, "Session rollover")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            raise
                finally:
                    if secondary_websocket:
                        untrack_websocket(secondary_websocket)
                        await secondary_websocket.close()
                        secondary_websocket = None
                return
            logger.error("WebSocket error: %s", e)
            raise
        finally:
            # Ensure any open websockets are closed
            if websocket:
                try:
                    untrack_websocket(websocket)
                    await websocket.close()
                except:
                    pass
            if secondary_websocket:
                try:
                    untrack_websocket(secondary_websocket)
                    await secondary_websocket.close()
                except:
                    pass
    
    async def _process_message(self, message: str):
        """Process incoming ticker or bookTicker message"""
        try:
            data = json.loads(message)
            
            # Combined stream format has stream name and data
            if 'stream' in data and 'data' in data:
                stream_name = data['stream']
                stream_data = data['data']
                
                if '@ticker' in stream_name:
                    # Full ticker update with 24h stats
                    self.price_data.update({
                        'price': float(stream_data.get('c', 0)),  # Current price
                        'volume_24h': float(stream_data.get('v', 0)),
                        'high_24h': float(stream_data.get('h', 0)),
                        'low_24h': float(stream_data.get('l', 0)),
                        'change_24h': float(stream_data.get('P', 0)),
                        'last_update': datetime.now()
                    })
                    # Ticker also has bid/ask but bookTicker is more accurate
                    if self.price_data.get('bid') is None:
                        self.price_data['bid'] = float(stream_data.get('b', 0))
                    if self.price_data.get('ask') is None:
                        self.price_data['ask'] = float(stream_data.get('a', 0))
                        
                elif '@bookTicker' in stream_name:
                    # Real-time best bid/ask update
                    self.price_data.update({
                        'bid': float(stream_data.get('b', 0)),    # Best bid price
                        'ask': float(stream_data.get('a', 0)),    # Best ask price
                        'bid_qty': float(stream_data.get('B', 0)), # Best bid quantity
                        'ask_qty': float(stream_data.get('A', 0)), # Best ask quantity
                        'last_update': datetime.now()
                    })
                    # Update current price as midpoint if we don't have ticker price
                    if self.price_data.get('price') is None and self.price_data['bid'] > 0 and self.price_data['ask'] > 0:
                        self.price_data['price'] = (self.price_data['bid'] + self.price_data['ask']) / 2
                        
            else:
                # Single stream format (backward compatibility)
                self.price_data = {
                    'price': float(data.get('c', 0)),  # Current price
                    'bid': float(data.get('b', 0)),    # Best bid
                    'ask': float(data.get('a', 0)),    # Best ask
                    'volume_24h': float(data.get('v', 0)),
                    'high_24h': float(data.get('h', 0)),
                    'low_24h': float(data.get('l', 0)),
                    'change_24h': float(data.get('P', 0)),
                    'last_update': datetime.now()
                }
            
            self.current_price = self.price_data.get('price')
            self.last_update = self.price_data.get('last_update')
            
            # Log the price
            if self.current_price:
                self._log_spot_price()
            
        except Exception as e:
            logger.error(f"Failed to parse message: {e}, Message: {message[:200]}")
    
    def get_current_price(self, require_fresh: bool = True, max_age_seconds: float = 5.0) -> Optional[float]:
        """
        Get the current spot price
        
        Args:
            require_fresh: If True, returns None if price is stale
            max_age_seconds: Maximum age for a price to be considered fresh
        """
        if self.current_price is None:
            return None
            
        # Check if price is stale
        if require_fresh and self.last_update:
            age = (datetime.now() - self.last_update).total_seconds()
            if age > max_age_seconds:
                logger.warning(f"Price data is stale ({age:.1f}s old)")
                return None
                
        return self.current_price
    
    def get_price_data(self) -> Dict[str, Any]:
        """Get all price data"""
        return self.price_data.copy()
    
    def is_price_fresh(self, max_age_seconds: float = 5.0) -> bool:
        """Check if price is fresh (not stale)"""
        if not self.last_update:
            return False
            
        age = (datetime.now() - self.last_update).total_seconds()
        return age <= max_age_seconds
    
    def get_connection_diagnostics(self) -> Dict[str, Any]:
        """Get connection health diagnostics"""
        diagnostics = {
            'is_connected': self.is_connected,
            'current_url': self.ws_urls[self.current_url_index] if self.current_url_index < len(self.ws_urls) else None,
            'reconnect_attempts': self.reconnect_attempts,
            'reconnect_delay': self.reconnect_delay,
            'total_messages': self.total_messages_received,
            'recent_errors': self.connection_errors[-5:] if self.connection_errors else []
        }
        
        if self.is_connected and self.connection_start_time:
            diagnostics['uptime_seconds'] = time.time() - self.connection_start_time
            
        if self.last_pong_time:
            diagnostics['seconds_since_last_message'] = time.time() - self.last_pong_time
            
        return diagnostics


# Multi-currency singleton instances
_binance_feeds = {}


def get_binance_spot_feed(currency: str = "ETH") -> BinanceSpotPriceFeed:
    """Get or create the Binance spot price feed for specified currency"""
    global _binance_feeds
    
    currency = validate_currency(currency)
    
    if currency not in _binance_feeds:
        _binance_feeds[currency] = BinanceSpotPriceFeed(currency, log_spot_prices=True)
        _binance_feeds[currency].start()
        
        # Wait for initial price
        timeout = 5
        start = time.time()
        while _binance_feeds[currency].get_current_price(require_fresh=False) is None and time.time() - start < timeout:
            time.sleep(0.1)
        
        logger.info(f"Initialized Binance feed for {currency}")
    
    return _binance_feeds[currency]


def stop_all_binance_feeds():
    """Stop all active Binance feeds"""
    global _binance_feeds
    
    for currency, feed in _binance_feeds.items():
        feed.stop()
        logger.info(f"Stopped Binance feed for {currency}")
    
    _binance_feeds.clear()


def get_spot_price(currency: str) -> Optional[float]:
    """
    Canonical spot price from Binance.
    Uses WS if fresh; otherwise falls back to Binance REST for resilience.
    """
    try:
        # Try WebSocket feed first
        feed = get_binance_spot_feed(currency)
        price = feed.get_current_price(require_fresh=True, max_age_seconds=SPOT_STALE_SECONDS)
        
        if price is not None:
            return price
        
        # Fall back to REST API (still Binance)
        logger.info(f"WebSocket price stale/unavailable, using REST API for {currency}")
        rest_data = fetch_spot_price_rest(currency)
        
        if rest_data:
            return rest_data.get("price")
            
    except Exception as e:
        logger.error(f"Failed to get Binance spot price for {currency}: {e}")
        
        # Last resort: try REST API directly
        try:
            rest_data = fetch_spot_price_rest(currency)
            if rest_data:
                return rest_data.get("price")
        except Exception as rest_e:
            logger.error(f"REST API fallback also failed: {rest_e}")
    
    return None


# Backward compatibility alias - remove after updating all imports
get_spot_price_fallback = get_spot_price


def get_spot_price_rest(symbol: str) -> Optional[float]:
    """
    Get spot price directly from Binance REST API
    
    Args:
        symbol: Trading symbol (e.g., "ETHUSDT", "BTCUSDT")
        
    Returns:
        Price as float or None if failed
    """
    # Extract currency from symbol (e.g., "BTCUSDT" -> "BTC")
    currency = None
    for curr in SUPPORTED_CURRENCIES:
        if symbol.upper().startswith(curr):
            currency = curr
            break
    
    if not currency:
        logger.error(f"Unsupported symbol: {symbol}")
        return None
    
    # Delegate to unified REST function
    result = fetch_spot_price_rest(currency)
    if result:
        return result.get("price")
    
    return None