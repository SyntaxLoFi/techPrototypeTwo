from __future__ import annotations
from typing import Protocol, Mapping, Any, Dict, Optional, List
import asyncio
import logging
import time

# Import WebSocket snapshot collector with fallback (mirrors main_scanner behavior)
try:
    from scripts.data_collection.derive_ws_client import collect_data_snapshot as collect_data_snapshot
except Exception:  # pragma: no cover - legacy fallback
    try:
        from simple_websocket_manager import collect_data_snapshot as collect_data_snapshot
    except Exception:
        collect_data_snapshot = None  # type: ignore

from scripts.data_collection.binance_spot_integration import get_spot_price as get_binance_spot

class DataRefresher(Protocol):
    async def fetch_all(self) -> Mapping[str, Any]: ...

class DataRefresh:
    """Collects/refreshes external market data (spot + orderbooks) and updates scanners.

    This implementation mirrors the previous `_collect_fresh_market_data` method in main_scanner,
    but keeps IO at the boundary and avoids global singletons.
    """
    def __init__(self, scanners: Dict[str, Dict[str, Any]], logger: Optional[logging.Logger] = None) -> None:
        self.scanners = scanners
        self.logger = logger or logging.getLogger(__name__)

    async def fetch_all(self) -> Mapping[str, Any]:
        self.logger.info("Collecting fresh market data snapshots...")
        for currency, scanner in self.scanners.items():
            try:
                # Prefer Binance spot when available
                binance_price = get_binance_spot(currency)
                if binance_price:
                    scanner['current_spot'] = binance_price
                    self.logger.info(f"Spot (Binance): ${binance_price:,.2f}")
                    # Populate option orderbooks via a short snapshot (full chain)
                    option_instruments: List[str] = []
                    if scanner.get('has_options') and scanner.get('options_collector'):
                        all_opts = scanner['options_collector'].get_all_options() or []
                        option_instruments = [o['instrument_name'] for o in all_opts if o.get('instrument_name')]
                    if option_instruments and collect_data_snapshot:
                        try:
                            data_snapshot = await asyncio.wait_for(
                                collect_data_snapshot(
                                    currency,
                                    duration_seconds=8,
                                    option_instruments=option_instruments
                                ),
                                timeout=30.0
                            )
                            if data_snapshot and data_snapshot.get('orderbooks'):
                                for ob_data in data_snapshot['orderbooks']:
                                    instrument = ob_data.get('instrument')
                                    if instrument and scanner.get('orderbook_handler'):
                                        scanner['orderbook_handler'].process_message(
                                            {
                                                "method": "subscription",
                                                "params": {
                                                    "channel": f"orderbook.{instrument}.1.10",
                                                    "data": ob_data.get('data', {})
                                                }
                                            },
                                            oracle_price=scanner.get('current_spot')
                                        )
                        except Exception as e:
                            self.logger.warning(f"Failed to collect option orderbooks for {currency}: {e}")
                else:
                    # Fallback to WebSocket snapshot for both spot + orderbooks
                    self.logger.warning(f"Binance unavailable for {currency}, trying WebSocket snapshot...")
                    if not collect_data_snapshot:
                        raise RuntimeError("No snapshot collector available")
                    try:
                        self.logger.info(f"Fetching spot price for {currency} via WebSocket...")
                        option_instruments: List[str] = []
                        if scanner.get('has_options') and scanner.get('options_collector'):
                            try:
                                all_opts = scanner['options_collector'].get_all_options() or []
                                option_instruments = [o['instrument_name'] for o in all_opts if o.get('instrument_name')]
                            except Exception as e:
                                self.logger.debug(f"Unable to build option instrument list for {currency}: {e}")
                        data_snapshot = await asyncio.wait_for(
                            collect_data_snapshot(currency, duration_seconds=10, option_instruments=option_instruments),
                            timeout=30.0
                        )
                    except asyncio.TimeoutError:
                        error_msg = ("CRITICAL: WebSocket timeout while fetching spot price for "
                                     f"{currency}. Check network connectivity.")
                        self.logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    except Exception as e:
                        error_msg = f"CRITICAL: Failed to fetch spot price for {currency}: {e}"
                        self.logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    # Spot from snapshot (with staleness logging)
                    spot_info = (data_snapshot or {}).get('oracle_price', {})
                    price_val = spot_info.get('price')
                    timestamp_ms = spot_info.get('timestamp')
                    if price_val is not None:
                        try:
                            price_val = float(price_val)
                        except Exception:
                            price_val = None
                    if price_val:
                        scanner['current_spot'] = price_val
                        try:
                            age_seconds = time.time() - timestamp_ms/1000.0 if timestamp_ms else float('inf')
                        except Exception:
                            age_seconds = float('inf')
                        if age_seconds < 5.0:
                            self.logger.info(f"Spot (snapshot): ${price_val:,.2f}")
                        else:
                            self.logger.warning(f"Spot (snapshot, stale {age_seconds:.1f}s old): ${price_val:,.2f}")
                    # Update orderbooks from snapshot
                    if (data_snapshot or {}).get('orderbooks'):
                        for ob_data in data_snapshot['orderbooks']:
                            instrument = ob_data.get('instrument')
                            if instrument and scanner.get('orderbook_handler'):
                                scanner['orderbook_handler'].process_message(
                                    {
                                        "method": "subscription",
                                        "params": {
                                            "channel": f"orderbook.{instrument}.1.10",
                                            "data": ob_data.get('data', {})
                                        }
                                    },
                                    oracle_price=scanner.get('current_spot')
                                )
            except (RuntimeError, ValueError):
                raise
            except Exception as e:
                self.logger.warning(f"Failed to collect market data for {currency}: {e}")
        return self.scanners
