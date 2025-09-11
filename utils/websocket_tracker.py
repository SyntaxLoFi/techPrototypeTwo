"""
Global WebSocket connection tracker to ensure proper cleanup
"""

import asyncio
import weakref
import logging
from typing import Set, Any

logger = logging.getLogger(__name__)

# Global set to track all active WebSocket connections
_active_websockets: Set[weakref.ref] = set()


def track_websocket(websocket: Any) -> None:
    """Add a WebSocket connection to the global tracker"""
    _active_websockets.add(weakref.ref(websocket))
    logger.debug(f"Tracking WebSocket: {websocket}")


def untrack_websocket(websocket: Any) -> None:
    """Remove a WebSocket connection from the global tracker"""
    to_remove = None
    for ws_ref in _active_websockets:
        ws = ws_ref()
        if ws is websocket:
            to_remove = ws_ref
            break
    
    if to_remove:
        _active_websockets.remove(to_remove)
        logger.debug(f"Untracked WebSocket: {websocket}")


async def close_all_websockets() -> None:
    """Close all tracked WebSocket connections"""
    logger.info(f"Closing {len(_active_websockets)} tracked WebSocket connections...")
    
    # Create a copy to avoid modification during iteration
    websockets_to_close = []
    for ws_ref in list(_active_websockets):
        ws = ws_ref()
        if ws:
            websockets_to_close.append(ws)
    
    # Close all WebSockets
    close_tasks = []
    for ws in websockets_to_close:
        try:
            if hasattr(ws, 'close') and callable(ws.close):
                close_tasks.append(asyncio.create_task(ws.close()))
        except Exception as e:
            logger.debug(f"Error preparing to close WebSocket: {e}")
    
    # Wait for all close operations with timeout
    if close_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*close_tasks, return_exceptions=True),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout while closing WebSockets")
    
    # Clear the tracker
    _active_websockets.clear()
    logger.info("All WebSocket connections closed")


def get_active_websocket_count() -> int:
    """Get the number of active WebSocket connections"""
    # Clean up dead references
    dead_refs = []
    for ws_ref in _active_websockets:
        if ws_ref() is None:
            dead_refs.append(ws_ref)
    
    for ref in dead_refs:
        _active_websockets.remove(ref)
    
    return len(_active_websockets)