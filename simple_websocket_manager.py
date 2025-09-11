"""
Simplified WebSocket manager based on working lyra-options-client implementation
"""

import ssl
import certifi
import asyncio
import websockets
import json
import logging
from typing import AsyncIterator, Dict

# Import WebSocket tracker
try:
    from utils.websocket_tracker import track_websocket, untrack_websocket
except ImportError:
    # Fallback if tracker not available
    def track_websocket(ws): pass
    def untrack_websocket(ws): pass

logger = logging.getLogger(__name__)


def get_ssl_context():
    """Get SSL context with proper certificates"""
    return ssl.create_default_context(cafile=certifi.where())


class SimpleWebSocketManager:
    """Simple WebSocket manager without complex reconnection - based on working implementation"""
    
    def __init__(self, uri: str = "wss://api.lyra.finance/ws"):
        self.uri = uri
        self.ssl_context = get_ssl_context()
        self.websocket = None
        self.is_connected = False
        
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.websocket = await websockets.connect(
                self.uri, 
                ssl=self.ssl_context,
                open_timeout=10,      # 10s to establish connection
                ping_interval=10,     # Send ping every 10s (was 20s)
                ping_timeout=5,       # Wait 5s for pong response (was 10s)
                close_timeout=10      # 10s for graceful close
            )
            # Track this WebSocket connection
            track_websocket(self.websocket)
            self.is_connected = True
            logger.info(f"Connected to {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.is_connected = False
            raise
            
    async def disconnect(self):
        """Close WebSocket connection gracefully"""
        if self.websocket:
            self.is_connected = False  # Set flag first to stop receive loops
            # Untrack before closing
            untrack_websocket(self.websocket)
            try:
                # Close with a short timeout to ensure cleanup
                await asyncio.wait_for(self.websocket.close(), timeout=2.5)
            except asyncio.TimeoutError:
                logger.warning("WebSocket close timed out, forcing closure")
                # Force close if normal close times out
                self.websocket = None
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None
                logger.info("Disconnected from WebSocket")
            
    async def send_message(self, message: dict):
        """Send a message to the WebSocket"""
        if not self.is_connected or not self.websocket:
            raise Exception("Not connected to WebSocket")
            
        await self.websocket.send(json.dumps(message))
        
    async def receive_messages(self, timeout_seconds: float = 1.0) -> AsyncIterator[dict]:
        """
        Async generator that yields received messages with simple timeout handling
        
        Args:
            timeout_seconds: Timeout for receiving messages
            
        Yields:
            Parsed JSON messages
        """
        if not self.is_connected or not self.websocket:
            raise Exception("Not connected to WebSocket")
            
        while self.is_connected:
            try:
                message = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=timeout_seconds
                )
                yield json.loads(message)
            except asyncio.TimeoutError:
                # Timeout is normal - just continue
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.is_connected = False
                break
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                self.is_connected = False
                break
                
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()


class SimpleSubscriptionManager:
    """Simple subscription manager"""
    
    def __init__(self, ws_manager: SimpleWebSocketManager):
        self.ws_manager = ws_manager
        self.active_subscriptions = set()
        
    async def subscribe(self, channels: list, subscription_id: str = None):
        """
        Subscribe to channels
        
        Args:
            channels: List of channel names
            subscription_id: Optional subscription ID
        """
        if not channels:
            return
            
        message = {
            "method": "subscribe",
            "params": {"channels": channels}
        }
        
        if subscription_id:
            message["id"] = subscription_id
            
        await self.ws_manager.send_message(message)
        self.active_subscriptions.update(channels)
        logger.info(f"Subscribed to {len(channels)} channels")
        
    async def subscribe_in_batches(self, channels: list, batch_size: int = 50, 
                                  delay: float = 0.1):
        """
        Subscribe to channels in batches with delay
        
        Args:
            channels: List of channel names
            batch_size: Number of channels per batch
            delay: Delay between batches in seconds
        """
        logger.info(f"Subscribing to {len(channels)} channels in batches of {batch_size}")
        
        for i in range(0, len(channels), batch_size):
            batch = channels[i:i + batch_size]
            await self.subscribe(batch, f"batch_{i}")
            if i + batch_size < len(channels):
                await asyncio.sleep(delay)

    async def resubscribe_all(self):
        """Resubscribe to all remembered channels after reconnect."""
        if not self.active_subscriptions:
            return
        await self.subscribe_in_batches(list(self.active_subscriptions))
                
    def get_active_subscriptions(self) -> set:
        """Get set of active subscription channels"""
        return self.active_subscriptions.copy()


async def collect_data_snapshot(currency: str, duration_seconds: int = 30,
                                option_instruments: list = None,
                                group: int = 1, depth: int = 10,
                                max_sub_batch: int = 200):
    """
    Collect a snapshot of data for a currency - similar to working implementation
    
    Args:
        currency: Currency to collect (BTC, ETH, etc.)
        duration_seconds: How long to collect data
        
    Returns:
        Dictionary with collected data
    """
    data = {
        'spot_prices': [],
        'orderbooks': [],
        'timestamp': None
    }
    
    try:
        async with SimpleWebSocketManager() as ws:
            sub_manager = SimpleSubscriptionManager(ws)
            
            # Subscribe to spot + perp + full option chain (batched)
            channels = [f"spot_feed.{currency}", f"orderbook.{currency}-PERP.{group}.{depth}"]
            if option_instruments:
                uniq = list(dict.fromkeys([i for i in option_instruments if isinstance(i, str)]))
                channels.extend([f"orderbook.{inst}.{group}.{depth}" for inst in uniq])
            await sub_manager.subscribe_in_batches(channels, batch_size=max_sub_batch, delay=0.05)
            
            # Collect data for the specified duration
            start_time = asyncio.get_event_loop().time()
            message_count = 0
            
            async for message in ws.receive_messages():
                message_count += 1
                
                # Process different message types
                if message.get("method") == "subscription":
                    params = message.get("params", {})
                    channel = params.get("channel", "")
                    
                    if channel.startswith("spot_feed"):
                        data['spot_prices'].append({
                            'timestamp': params.get("data", {}).get("timestamp"),
                            'data': params.get("data")
                        })
                    elif channel.startswith("orderbook"):
                        data['orderbooks'].append({
                            'timestamp': params.get("data", {}).get("timestamp"),
                            'instrument': channel.split('.')[1],
                            'data': params.get("data")
                        })
                
                # Check if we've collected enough
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= duration_seconds:
                    break
                    
                # Prevent infinite collection
                if message_count > 10000:
                    logger.warning("Collected too many messages, stopping")
                    break
            
            data['timestamp'] = asyncio.get_event_loop().time()
            logger.info(f"Collected {message_count} messages for {currency} in {elapsed:.1f}s")
            
    except Exception as e:
        logger.error(f"Error collecting data for {currency}: {e}")
        
    return data