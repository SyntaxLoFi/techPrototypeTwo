# scripts/data_collection/derive_ws_client.py
import asyncio
import json
import logging
from typing import AsyncIterator, Dict, Optional, List, Any
import time
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simple_websocket_manager import SimpleWebSocketManager

DERIVE_WS = "wss://api.lyra.finance/ws"
logger = logging.getLogger(__name__)

# Persistent client (avoids connect→subscribe→close loop across scans)
# Use 'Any' to avoid forward-ref issues if DeriveWSClient isn't resolved at import time.
_PERSISTENT_CLIENT: Optional[Any] = None

_ORDERBOOK_CH_PREFIX = "orderbook."
_SPOT_CH_PREFIX = "spot_feed."

def _parse_instrument_from_channel(channel: str) -> Optional[str]:
    # orderbook.{instrument}.{group}.{depth}
    if not channel.startswith(_ORDERBOOK_CH_PREFIX):
        return None
    parts = channel.split(".")
    return parts[1] if len(parts) >= 4 else None

class DeriveWSClient:
    """
    Minimal WS client for Derive (Lyra) public channels.
    * ticker.{instrument}.{interval}
    * orderbook.{instrument}.{group}.{depth}
    * spot_feed.{currency}
    """
    def __init__(self):
        self.ws = SimpleWebSocketManager(DERIVE_WS)
        self._subscription_id = 0

    # ----------------------------------------------------------------------
    # Utilities

    def _next_id(self) -> int:
        """Generate next subscription ID"""
        self._subscription_id += 1
        return self._subscription_id


    # ----------------------------------------------------------------------
    # Orderbook subscription (per instrument)
    async def subscribe_orderbook(self, instrument_name: str, group: int = 1, depth: int = 10) -> AsyncIterator[Dict]:
        """
        Subscribe to orderbook for a single instrument.
        Channel schema per docs: orderbook.{instrument}.{group}.{depth}
        """
        if group not in (1, 10, 100):
            raise ValueError("group must be one of 1, 10, 100")
        if depth not in (1, 10, 20, 100):
            raise ValueError("depth must be one of 1, 10, 20, 100")

        await self.ws.connect()
        channel = f"orderbook.{instrument_name}.{group}.{depth}"
        subscribe_msg = {
            "id": self._next_id(),
            "method": "subscribe",
            "params": {"channels": [channel]},
        }
        await self.ws.send_message(subscribe_msg)
        logger.info(f"Subscribed to {channel}")

        async for msg in self.ws.receive_messages():
            if not isinstance(msg, dict):
                continue
            # Confirmations
            if msg.get("id") and msg.get("result") is not None:
                continue
            # Notifications
            if msg.get("method") == "subscription":
                params = msg.get("params", {})
                ch = params.get("channel", "")
                if ch == channel:
                    data = params.get("data", {})
                    yield {
                        "channel": ch,
                        "instrument_name": data.get("instrument_name", instrument_name),
                        "timestamp": data.get("timestamp"),
                        "bids": data.get("bids", []),
                        "asks": data.get("asks", []),
                        "raw": data,
                    }

    # ----------------------------------------------------------------------
    # Bulk snapshot: orderbooks for many instruments + spot feed for the currency
    async def collect_data_snapshot(
        self,
        currency: str,
        option_instruments: Optional[List[str]] = None,
        *,
        duration_seconds: int = 8,
        group: int = 1,
        depth: int = 10,
        include_tickers: bool = False,
    ) -> Dict[str, List[Dict]]:
        """
        Subscribe to:
          - spot_feed.{currency}
          - orderbook.{instrument}.{group}.{depth} for each instrument
        Collect messages for `duration_seconds` and return them.
        """
        await self.ws.connect()
        channels: List[str] = [f"spot_feed.{currency}"]
        option_instruments = option_instruments or []
        channels += [f"orderbook.{inst}.{group}.{depth}" for inst in option_instruments]
        if include_tickers:
            # Not used by the scanner today, but handy to toggle
            channels += [f"ticker.{inst}.1000" for inst in option_instruments]

        sub = {"id": self._next_id(), "method": "subscribe", "params": {"channels": channels}}
        await self.ws.send_message(sub)
        logger.info(f"Subscribed to {len(channels)} channels for snapshot")

        orderbooks: List[Dict] = []
        spot_prices: List[Dict] = []
        started = time.time()

        async for msg in self.ws.receive_messages():
            if time.time() - started > duration_seconds:
                break
            if not isinstance(msg, dict):
                continue
            if msg.get("method") != "subscription":
                continue
            params = msg.get("params", {})
            ch = params.get("channel", "")
            data = params.get("data", {})
            if ch.startswith(_SPOT_CH_PREFIX):
                # Keep raw shape; scanner extracts timestamp/feeds later
                spot_prices.append({"channel": ch, "data": data})
            elif ch.startswith(_ORDERBOOK_CH_PREFIX):
                inst = data.get("instrument_name") or _parse_instrument_from_channel(ch)
                orderbooks.append({"instrument": inst, "channel": ch, "data": data})

        try:
            unsub = {"id": self._next_id(), "method": "unsubscribe", "params": {"channels": channels}}
            await self.ws.send_message(unsub)
        finally:
            await self.ws.disconnect()

        return {"orderbooks": orderbooks, "spot_prices": spot_prices}

    async def subscribe_ticker(self, instrument_name: str, interval_ms: int = 1000) -> AsyncIterator[Dict]:
        """
        Subscribe to ticker updates for an instrument.
        
        Args:
            instrument_name: Instrument to subscribe to (e.g., "ETH-PERP", "BTC-PERP")
            interval_ms: Update interval in milliseconds (100 or 1000)
            
        Yields:
            Normalized ticker snapshots with fields:
            - instrument_name: The instrument name
            - mark_price: Mark price
            - index_price: Index/spot price
            - best_bid: Best bid price
            - best_ask: Best ask price
            - maker_fee: Maker fee rate
            - taker_fee: Taker fee rate
            - base_fee: Base fee
            - perp_details: Perpetual-specific details (funding rate, etc.)
            - timestamp: Update timestamp
        """
        if interval_ms not in [100, 1000]:
            raise ValueError("interval_ms must be 100 or 1000")
            
        await self.ws.connect()
        
        # Subscribe to ticker channel
        channel = f"ticker.{instrument_name}.{interval_ms}"
        subscribe_msg = {
            "id": self._next_id(),
            "method": "subscribe",
            "params": {"channels": [channel]}
        }
        
        await self.ws.send_message(subscribe_msg)
        logger.info(f"Subscribed to {channel}")
        
        async for msg in self.ws.receive_messages():
            if not isinstance(msg, dict):
                continue
                
            # Handle subscription confirmation
            if msg.get("id") and msg.get("result") is not None:
                logger.debug(f"Subscription confirmed: {msg}")
                continue
                
            # Handle ticker notifications
            if msg.get("method") == "subscription":
                params = msg.get("params", {})
                ch = params.get("channel", "")
                
                if ch == channel:
                    data = params.get("data", {})
                    ticker = data.get("instrument_ticker", {})
                    
                    # Normalize ticker data
                    yield {
                        "instrument_name": ticker.get("instrument_name", instrument_name),
                        "mark_price": float(ticker.get("mark_price", 0.0)),
                        "index_price": float(ticker.get("index_price", 0.0)),
                        "best_bid": float(ticker.get("best_bid_price", 0.0)),
                        "best_ask": float(ticker.get("best_ask_price", 0.0)),
                        "maker_fee": float(ticker.get("maker_fee_rate", 0.0)),
                        "taker_fee": float(ticker.get("taker_fee_rate", 0.0)),
                        "base_fee": float(ticker.get("base_fee", 0.0)),
                        "perp_details": ticker.get("perp_details", {}),
                        "timestamp": data.get("timestamp"),
                    }

    async def subscribe_multiple_tickers(self, instruments: list[str], interval_ms: int = 1000) -> AsyncIterator[Dict]:
        """
        Subscribe to multiple ticker channels at once.
        
        Args:
            instruments: List of instrument names
            interval_ms: Update interval in milliseconds (100 or 1000)
            
        Yields:
            Normalized ticker snapshots from all subscribed instruments
        """
        if interval_ms not in [100, 1000]:
            raise ValueError("interval_ms must be 100 or 1000")
            
        await self.ws.connect()
        
        # Build channel list
        channels = [f"ticker.{instrument}.{interval_ms}" for instrument in instruments]
        
        # Subscribe to all channels
        subscribe_msg = {
            "id": self._next_id(),
            "method": "subscribe",
            "params": {"channels": channels}
        }
        
        await self.ws.send_message(subscribe_msg)
        logger.info(f"Subscribed to {len(channels)} ticker channels")
        
        # Track which channels we're listening to
        channel_set = set(channels)
        
        async for msg in self.ws.receive_messages():
            if not isinstance(msg, dict):
                continue
                
            # Handle subscription confirmation
            if msg.get("id") and msg.get("result") is not None:
                logger.debug(f"Subscription confirmed: {msg}")
                continue
                
            # Handle ticker notifications
            if msg.get("method") == "subscription":
                params = msg.get("params", {})
                ch = params.get("channel", "")
                
                if ch in channel_set:
                    data = params.get("data", {})
                    ticker = data.get("instrument_ticker", {})
                    
                    # Extract instrument name from channel if not in ticker
                    if not ticker.get("instrument_name") and ch.startswith("ticker."):
                        parts = ch.split(".")
                        if len(parts) >= 2:
                            instrument_name = parts[1]
                    else:
                        instrument_name = ticker.get("instrument_name", "")
                    
                    # Normalize ticker data
                    yield {
                        "instrument_name": instrument_name,
                        "mark_price": float(ticker.get("mark_price", 0.0)),
                        "index_price": float(ticker.get("index_price", 0.0)),
                        "best_bid": float(ticker.get("best_bid_price", 0.0)),
                        "best_ask": float(ticker.get("best_ask_price", 0.0)),
                        "maker_fee": float(ticker.get("maker_fee_rate", 0.0)),
                        "taker_fee": float(ticker.get("taker_fee_rate", 0.0)),
                        "base_fee": float(ticker.get("base_fee", 0.0)),
                        "perp_details": ticker.get("perp_details", {}),
                        "timestamp": data.get("timestamp"),
                    }

    async def close(self):
        """Close the WebSocket connection"""
        await self.ws.disconnect()

    async def __aenter__(self):
        """Async context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# ----------------------------------------------------------------------
# Backwards-compatible module-level helper for the scanner
async def collect_data_snapshot(
    currency: str,
    option_instruments: Optional[List[str]] = None,
    duration_seconds: int = 8,
    group: int = 1,
    depth: int = 10,
    include_tickers: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Thin wrapper to mirror the signature expected by main_scanner.
    """
    global _PERSISTENT_CLIENT
    _PERSISTENT_CLIENT = _PERSISTENT_CLIENT or DeriveWSClient()
    client = _PERSISTENT_CLIENT
    await client.ws.connect()
    try:
        return await client.collect_data_snapshot(currency, option_instruments, duration_seconds=duration_seconds, group=group, depth=depth, include_tickers=include_tickers)
    finally:
        await client.close()

# (Keep any example code below if you need it internally)


async def example_multiple_tickers():
    """Example: Subscribe to multiple tickers"""
    async with DeriveWSClient() as client:
        instruments = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
        count = 0
        
        async for tick in client.subscribe_multiple_tickers(instruments, interval_ms=1000):
            instrument = tick['instrument_name']
            print(f"{instrument}: ${tick['mark_price']:.2f} (spread: ${tick['best_ask'] - tick['best_bid']:.2f})")
            
            count += 1
            if count >= 15:  # Exit after 15 ticks total
                break


if __name__ == "__main__":
    # Run single ticker example
    print("=== Single Ticker Example ===")
    asyncio.run(example_single_ticker())
    
    # Run multiple ticker example
    print("\n=== Multiple Ticker Example ===")
    asyncio.run(example_multiple_tickers())