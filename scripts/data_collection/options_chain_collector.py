"""
Enhanced options chain collector that finds ALL possible option combinations
for replicating binary options

Key features:
1. Call spreads - traditional binary replication
2. Put spreads - using put-call parity
3. Single options with dynamic hedging
4. Synthetic strategies combining calls and puts

Author: Polymarket-Lyra Arbitrage System
Date: 2024
"""
import logging
from utils.debug_recorder import get_recorder  # type: ignore
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple

from config_manager import LYRA_API_BASE, DEFAULT_VOLATILITY
from utils.http_client import get, post


class OptionsChainCollector:
    """
    Collect and organize options chains with all possible strategies
    
    This class finds the optimal way to replicate a binary option
    by comparing all available strategies and their costs
    """
    
    def __init__(self):
        self.api_base = LYRA_API_BASE
        self.logger = logging.getLogger('OptionsChainCollector')
        self.chains_by_expiry = defaultdict(lambda: defaultdict(dict))
        self.all_options = []
        self.options_metadata = {}
        self.expiry_instruments = defaultdict(list)  # expiry -> list of instruments
        # Two-sided exit liquidity thresholds (contracts and USDC notional)
        self.min_top_qty_for_exit = 0.0; self.min_top_notional_for_exit = 0.0

    # --- New: simple read-side API used by populate_execution_details() ---
    def _select_option(self, items, *, symbol, expiry, strike, otype):
        t = (otype or "").upper()
        for o in items or []:
            try:
                if (str(o.get("type","")).upper() == t and
                    float(o.get("strike")) == float(strike) and
                    (o.get("expiry_date") == expiry or o.get("expiry") == expiry)):
                    return o
            except Exception:
                continue
        return None

    def get_quote(self, *, symbol: str, expiry: str, strike: float, otype: str) -> dict | None:
        """
        Return a normalized quote dict for (symbol, expiry, strike, type).
        Pulls from in-process OptionsQuotesCache; falls back to OptionsRepository or self.all_options.
        """
        try:
            from market_data.options_quotes_cache import OptionsQuotesCache
            items = OptionsQuotesCache().get_snapshot(symbol)  # returns []
        except Exception:
            items = []
        if not items:
            try:
                from market_data.options_repository import OptionsRepository
                items = OptionsRepository().get_active(symbol)
            except Exception:
                items = self.all_options or []

        o = self._select_option(items, symbol=symbol, expiry=expiry, strike=strike, otype=otype)
        if not o:
            return None

        bid = o.get("bid"); ask = o.get("ask")
        bid = float(bid) if bid is not None else None
        ask = float(ask) if ask is not None else None
        # Robust mid/mark (allow one-sided)
        mid = o.get("mid")
        if mid is None:
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
            else:
                mid = bid if (bid or 0) > 0 else ask if (ask or 0) > 0 else None
        mark = o.get("mark", mid)

        # IV: prefer explicit, else mid of iv_bid/iv_ask, else implied_volatility/sigma
        iv = o.get("iv") or o.get("quote_iv")
        if iv is None:
            ivb, iva = o.get("iv_bid"), o.get("iv_ask")
            try:
                if ivb is not None and iva is not None:
                    iv = (float(ivb) + float(iva)) / 2.0
            except Exception:
                iv = None
        if iv is None:
            iv = o.get("implied_volatility") or o.get("sigma")

        inst_id = o.get("instrument_name") or o.get("symbol") or f"{symbol}-{expiry}-{strike}-{otype[0:1].upper()}"
        ts = o.get("ts") or o.get("timestamp")
        return {"bid": bid, "ask": ask, "mid": mid, "mark": mark, "iv": iv,
                "instrument_id": inst_id, "timestamp": ts}

    def get_greeks(self, *, symbol: str, expiry: str, strike: float, otype: str) -> dict | None:
        """
        Return greeks if present; else compute via Black-Scholes when IV is available.
        """
        # Reuse the same option selection as get_quote()
        try:
            from market_data.options_quotes_cache import OptionsQuotesCache
            items = OptionsQuotesCache().get_snapshot(symbol)
        except Exception:
            items = []
        if not items:
            try:
                from market_data.options_repository import OptionsRepository
                items = OptionsRepository().get_active(symbol)
            except Exception:
                items = self.all_options or []
        o = self._select_option(items, symbol=symbol, expiry=expiry, strike=strike, otype=otype)
        if not o:
            return None
        if o.get("greeks"):
            return o["greeks"]
        try:
            from black_scholes_greeks import BlackScholesGreeks
            K = float(strike)
            # Time to expiry
            T_days = o.get("days_to_expiry")
            if T_days is None:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(expiry).replace(tzinfo=timezone.utc)
                T_days = max(0.1, (dt - datetime.now(timezone.utc)).total_seconds()/86400.0)
            T = float(T_days) / 365.25
            sigma = float(o.get("iv") or o.get("implied_volatility") or o.get("sigma") or 0.5)
            S = float(o.get("spot") or o.get("index") or o.get("underlying_price") or K)  # fallback on K
            r = 0.0
            bs = BlackScholesGreeks()
            return (bs.call_greeks if otype.upper()=="CALL" else bs.put_greeks)(S, K, r, sigma, T, scale_1pct=True)
        except Exception:
            return None

    def _two_sided_ok(self, ob: Dict, min_qty: float = 0.0, min_notional: float = 0.0) -> bool:
        """Require both best bid/ask *prices* and *sizes* to be positive; optionally enforce notional."""
        try:
            bb = float(ob.get('best_bid') or 0.0); ba = float(ob.get('best_ask') or 0.0)
            bbq = float(ob.get('best_bid_qty') or ob.get('bid_qty') or 0.0)
            baq = float(ob.get('best_ask_qty') or ob.get('ask_qty') or 0.0)
        except Exception:
            return False
        if bb <= 0 or ba <= 0: return False
        if not (bbq > min_qty and baq > min_qty): return False
        if min_notional > 0.0 and ((bb * bbq) < min_notional or (ba * baq) < min_notional): return False
        return True
    
    def get_all_options(self) -> List[Dict]:
        """Return all collected options data for comprehensive strategy generation."""
        return self.all_options
        
    def fetch_all_options(self, currency: str = "ETH", expired: bool = False):
        """Fetch all options for a currency"""
        try:
            # get_instruments returns all active instruments in a single request
            payload = {
                "currency": currency,
                "instrument_type": "option",
                "expired": expired
            }
            
            self.logger.info(f"Fetching options for {currency} with payload: {payload}")
            self.logger.info(f"API endpoint: {self.api_base}/public/get_instruments")
            
            response = post(
                f"{self.api_base}/public/get_instruments",
                json=payload
            )
            
            self.logger.info(f"API Response status: {response.status_code}")
            
            if response.ok:
                response_data = response.json()
                self.logger.info(f"Response keys: {list(response_data.keys())}")
                
                if "error" in response_data:
                    error_msg = f"CRITICAL: Lyra API returned error: {response_data['error']}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                else:
                    instruments = response_data.get("result", [])
                    all_options = instruments
                    self.logger.info(f"Raw API returned {len(instruments)} instruments")
                    
                    # Log sample instrument to check structure
                    if instruments:
                        self.logger.debug(f"Sample instrument: {instruments[0]}")
            else:
                error_msg = f"CRITICAL: Lyra API request failed with status {response.status_code}: {response.text}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            self.logger.info(f"Fetched {len(all_options)} total {currency} options from API")
            
            # Filter for active options only
            active_options = [opt for opt in all_options if opt.get("is_active", False)]
            self.logger.info(f"Found {len(active_options)} active options after filtering")
            
            # Log filtering details
            if len(active_options) != len(all_options):
                inactive_count = len(all_options) - len(active_options)
                self.logger.info(f"Filtered out {inactive_count} inactive options")
                # Show some examples of what was filtered
                for opt in all_options[:3]:
                    self.logger.debug(f"Option is_active={opt.get('is_active')}, name={opt.get('instrument_name')}")
            
            # Enhance with standardized fields
            enhanced_options = []
            for opt in active_options:
                # Extract option details
                option_details = opt.get("option_details", {})
                
                # Convert expiry timestamp to date string
                expiry_timestamp = option_details.get('expiry', 0)
                expiry_date_str = ''
                if expiry_timestamp > 0:
                    # Use UTC to avoid machine local-time drift
                    expiry_date_str = datetime.fromtimestamp(expiry_timestamp, tz=timezone.utc).strftime('%Y-%m-%d')
                
                # Convert P/C to put/call
                option_type_code = option_details.get('option_type', '')
                option_type_full = "put" if option_type_code == "P" else "call"
                
                # Create enhanced option with standardized fields for strategy generator
                enhanced_opt = opt.copy()
                strike_value = float(option_details.get('strike', 0))
                enhanced_opt.update({
                    'expiry_date': expiry_date_str,
                    'strike_price': strike_value,
                    'strike': strike_value,  # Add both for compatibility
                    'option_type': option_type_full,  # 'call' or 'put'
                    'type': option_type_full  # Add 'type' field that strategies expect
                })
                enhanced_options.append(enhanced_opt)
            
            self.all_options = enhanced_options
            
            self.logger.info(f"Enhanced {len(self.all_options)} active {currency} options")
            
            # Persist the snapshot so that ALL strategies can see the same data.
            try:
                from market_data.options_repository import OptionsRepository
                repo = OptionsRepository()
                snap = repo.set_active(currency, self.all_options)
                self.logger.info("Persisted %d %s options to OptionsRepository (asof=%s)",
                            len(self.all_options), currency, datetime.fromtimestamp(snap.asof).isoformat())
            except Exception:
                self.logger.exception("Failed to persist Lyra options to OptionsRepository")
            self.logger.info(f"Final count in self.all_options: {len(self.all_options)}")
            try:
                get_recorder().dump_json(f"options/{currency}_options.json", self.all_options, category="options")
                self.logger.info("Options chain collection successful (%d %s options)", len(self.all_options or []), currency)
            except Exception:
                pass
            
            # Validate we got options
            if not self.all_options:
                error_msg = f"CRITICAL: No active options found for {currency}. Lyra API returned {len(all_options)} total but all filtered out."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self._organize_options()
            
            # Print summary
            self.print_summary()
            
            return True
            
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            error_msg = f"CRITICAL: Failed to fetch options for {currency}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _organize_options(self):
        """Organize options by expiry/strike/type"""
        self.chains_by_expiry.clear()
        self.options_metadata.clear()
        self.expiry_instruments.clear()
        
        for option in self.all_options:
            instrument_name = option["instrument_name"]
            details = option["option_details"]
            
            expiry_timestamp = details["expiry"]
            expiry_date = datetime.fromtimestamp(expiry_timestamp, tz=timezone.utc).strftime('%Y-%m-%d')
            strike = float(details["strike"])
            option_type = details["option_type"]  # "P" or "C"
            
            # Convert P/C to put/call for consistency
            option_type_full = "put" if option_type == "P" else "call"
            
            # Store metadata
            self.options_metadata[instrument_name] = {
                "strike": strike,
                "type": option_type_full,
                "option_type_code": option_type,  # Keep original P/C
                "expiry_timestamp": expiry_timestamp,
                "expiry_date": expiry_date,
                "scheduled_activation": option.get("scheduled_activation"),
                "is_active": option.get("is_active", True),
                "tick_size": float(option.get("tick_size", 0.1)),
                "minimum_amount": float(option.get("minimum_amount", 0.1))
            }
            
            # Organize by expiry/strike/type
            self.chains_by_expiry[expiry_date][strike][option_type_full] = {
                "instrument_name": instrument_name,
                "metadata": self.options_metadata[instrument_name],
                "details": details,
                "raw_option": option  # Store full option data
            }
            
            # Track instruments by expiry
            self.expiry_instruments[expiry_date].append(instrument_name)
        
        # Sort strikes for each expiry
        for expiry_date in self.chains_by_expiry:
            self.chains_by_expiry[expiry_date] = dict(sorted(self.chains_by_expiry[expiry_date].items()))
    
    def get_expiry_dates(self) -> List[str]:
        """Get sorted list of expiry dates"""
        return sorted(self.chains_by_expiry.keys())
    
    def get_chain_for_expiry(self, expiry: str) -> Dict:
        """Get option chain for specific expiry"""
        return dict(self.chains_by_expiry.get(expiry, {}))
    
    def find_all_replication_strategies(
        self,
        polymarket_contract: Dict,
        orderbook_handler,
        max_strategies: int = 10
    ) -> List[Dict]:
        """
        Find ALL possible ways to replicate a binary option
        
        Args:
            polymarket_contract: The Polymarket contract to replicate
            orderbook_handler: Handler to get current bid/ask prices
            max_strategies: Maximum number of strategies to return
            
        Returns:
            List of strategies sorted by cost (cheapest first)
        """
        target_strike = polymarket_contract['strike_price']
        is_binary_call = polymarket_contract['is_above']  # "above" = binary call
        
        all_strategies = []
        
        # Check each expiry date
        for expiry_str in self.get_expiry_dates():
            # Parse expiry date and check if close enough to Polymarket expiry
            try:
                expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
                expiry_date = expiry_date.replace(tzinfo=timezone.utc)
                
                # Get Polymarket end date
                pm_end_str = polymarket_contract.get('end_date', '')
                if pm_end_str:
                    if pm_end_str.endswith('Z'):
                        pm_end = datetime.fromisoformat(pm_end_str.replace('Z', '+00:00'))
                    else:
                        pm_end = datetime.fromisoformat(pm_end_str)
                        if pm_end.tzinfo is None:
                            pm_end = pm_end.replace(tzinfo=timezone.utc)
                    
                    # Skip if expiry is too far from Polymarket date
                    if abs((expiry_date - pm_end).days) > 7:
                        continue
                
                # Find strategies for this expiry
                chain = self.get_chain_for_expiry(expiry_str)
                if not chain:
                    continue
                
                strikes = sorted(chain.keys())
                
                # 1. Call Spreads
                strategies = self._find_call_spreads(
                    strikes, target_strike, chain, expiry_str, 
                    is_binary_call, orderbook_handler
                )
                all_strategies.extend(strategies)
                
                # 2. Put Spreads
                strategies = self._find_put_spreads(
                    strikes, target_strike, chain, expiry_str,
                    is_binary_call, orderbook_handler
                )
                all_strategies.extend(strategies)
                
                # 3. Single Options (if close to strike)
                strategies = self._find_single_option_strategies(
                    strikes, target_strike, chain, expiry_str,
                    is_binary_call, orderbook_handler
                )
                all_strategies.extend(strategies)
                
            except Exception as e:
                self.logger.error(f"Error processing expiry {expiry_str}: {e}")
                continue
        
        # Sort by total cost (cheapest first)
        all_strategies.sort(key=lambda x: x.get('total_cost', float('inf')))
        
        # Return top strategies
        return all_strategies[:max_strategies]
    
    def _find_call_spreads(
        self,
        strikes: List[float],
        target_strike: float,
        chain: Dict,
        expiry: str,
        is_binary_call: bool,
        orderbook_handler
    ) -> List[Dict]:
        """
        Find call spread strategies to replicate binary
        
        For binary call: Buy Call(K-ε), Sell Call(K+ε)
        For binary put: Sell Call(K-ε), Buy Call(K+ε)
        """
        strategies = []
        
        # Find strikes bracketing the target
        for i in range(len(strikes) - 1):
            # Check if target strike is between these strikes
            if strikes[i] <= target_strike <= strikes[i + 1]:
                
                if is_binary_call:
                    # Binary call: Buy low strike, sell high strike
                    strategy = self._create_spread_strategy(
                        chain, strikes[i], strikes[i + 1],
                        'call', 'call', 1, -1,  # Buy low, sell high
                        expiry, "CALL_SPREAD_LONG", orderbook_handler,
                        target_strike
                    )
                else:
                    # Binary put: Sell low strike, buy high strike  
                    strategy = self._create_spread_strategy(
                        chain, strikes[i], strikes[i + 1],
                        'call', 'call', -1, 1,  # Sell low, buy high
                        expiry, "CALL_SPREAD_SHORT", orderbook_handler,
                        target_strike
                    )
                
                if strategy and strategy['total_cost'] > 0:
                    strategies.append(strategy)
                
                # Also try adjacent strikes for better pricing
                if i > 0:  # Try lower strike pair
                    if is_binary_call:
                        strategy = self._create_spread_strategy(
                            chain, strikes[i-1], strikes[i],
                            'call', 'call', 1, -1,
                            expiry, "CALL_SPREAD_LONG", orderbook_handler,
                            target_strike
                        )
                    else:
                        strategy = self._create_spread_strategy(
                            chain, strikes[i-1], strikes[i],
                            'call', 'call', -1, 1,
                            expiry, "CALL_SPREAD_SHORT", orderbook_handler,
                            target_strike
                        )
                    if strategy and strategy['total_cost'] > 0:
                        strategies.append(strategy)
        
        return strategies
    
    def _find_put_spreads(
        self,
        strikes: List[float],
        target_strike: float,
        chain: Dict,
        expiry: str,
        is_binary_call: bool,
        orderbook_handler
    ) -> List[Dict]:
        """
        Find put spread strategies to replicate binary
        
        For binary call: Sell Put(K-ε), Buy Put(K+ε)
        For binary put: Buy Put(K-ε), Sell Put(K+ε)
        """
        strategies = []
        
        # Find strikes bracketing the target
        for i in range(len(strikes) - 1):
            if strikes[i] <= target_strike <= strikes[i + 1]:
                
                if is_binary_call:
                    # Binary call via puts: Sell low strike put, buy high strike put
                    strategy = self._create_spread_strategy(
                        chain, strikes[i], strikes[i + 1],
                        'put', 'put', -1, 1,  # Sell low, buy high
                        expiry, "PUT_SPREAD_FOR_CALL", orderbook_handler,
                        target_strike
                    )
                else:
                    # Binary put via puts: Buy low strike put, sell high strike put
                    strategy = self._create_spread_strategy(
                        chain, strikes[i], strikes[i + 1],
                        'put', 'put', 1, -1,  # Buy low, sell high
                        expiry, "PUT_SPREAD_FOR_PUT", orderbook_handler,
                        target_strike
                    )
                
                if strategy and strategy['total_cost'] > 0:
                    strategies.append(strategy)
        
        return strategies
    
    def _find_single_option_strategies(
        self,
        strikes: List[float],
        target_strike: float,
        chain: Dict,
        expiry: str,
        is_binary_call: bool,
        orderbook_handler
    ) -> List[Dict]:
        """
        Find single option strategies (require dynamic hedging)
        
        Only viable if strike is very close to target
        """
        strategies = []
        
        # Find closest strike
        closest_strike = min(strikes, key=lambda x: abs(x - target_strike))
        
        # Only consider if within 2% of target
        if abs(closest_strike - target_strike) / target_strike > 0.02:
            return strategies
        
        if is_binary_call:
            # Strategy 1: Long call with dynamic hedge
            call_strategy = self._create_single_option_strategy(
                chain, closest_strike, 'call', 1,
                expiry, "LONG_CALL_DYNAMIC", orderbook_handler
            )
            if call_strategy:
                strategies.append(call_strategy)
            
            # Strategy 2: Short put with dynamic hedge (synthetically long)
            put_strategy = self._create_single_option_strategy(
                chain, closest_strike, 'put', -1,
                expiry, "SHORT_PUT_DYNAMIC", orderbook_handler
            )
            if put_strategy:
                strategies.append(put_strategy)
        else:
            # Strategy 1: Long put with dynamic hedge
            put_strategy = self._create_single_option_strategy(
                chain, closest_strike, 'put', 1,
                expiry, "LONG_PUT_DYNAMIC", orderbook_handler
            )
            if put_strategy:
                strategies.append(put_strategy)
            
            # Strategy 2: Short call with dynamic hedge
            call_strategy = self._create_single_option_strategy(
                chain, closest_strike, 'call', -1,
                expiry, "SHORT_CALL_DYNAMIC", orderbook_handler
            )
            if call_strategy:
                strategies.append(call_strategy)
        
        return strategies
    
    def _create_spread_strategy(
        self,
        chain: Dict,
        strike_low: float,
        strike_high: float,
        option_type_low: str,
        option_type_high: str,
        direction_low: int,  # 1 for buy, -1 for sell
        direction_high: int,
        expiry: str,
        strategy_name: str,
        orderbook_handler,
        binary_strike: float
    ) -> Optional[Dict]:
        """Create a spread strategy with current market prices"""
        # Get instruments
        opt_low = chain.get(strike_low, {}).get(option_type_low, {})
        opt_high = chain.get(strike_high, {}).get(option_type_high, {})
        
        if not opt_low or not opt_high:
            return None
        
        inst_low = opt_low.get('instrument_name')
        inst_high = opt_high.get('instrument_name')
        
        if not inst_low or not inst_high:
            return None
        
        # Get orderbooks
        ob_low = orderbook_handler.get_latest_orderbook(inst_low)
        ob_high = orderbook_handler.get_latest_orderbook(inst_high)
        
        if not ob_low or not ob_high:
            self.logger.debug(f"No orderbook for {inst_low} or {inst_high}")
            return None
        # Require two-sided books on both legs to guarantee exits
        if not self._two_sided_ok(ob_low,  self.min_top_qty_for_exit, self.min_top_notional_for_exit):  return None
        if not self._two_sided_ok(ob_high, self.min_top_qty_for_exit, self.min_top_notional_for_exit):  return None
        
        # Calculate costs based on direction
        if direction_low > 0:  # Buy
            cost_low = ob_low.get('best_ask', 0)
            if cost_low == 0:
                return None
        else:  # Sell
            cost_low = -ob_low.get('best_bid', 0)
            if ob_low.get('best_bid', 0) == 0:
                return None
            
        if direction_high > 0:  # Buy
            cost_high = ob_high.get('best_ask', 0)
            if cost_high == 0:
                return None
        else:  # Sell
            cost_high = -ob_high.get('best_bid', 0)
            if ob_high.get('best_bid', 0) == 0:
                return None
        
        # Net cost of the spread
        total_cost = cost_low + cost_high
        
        # TODO: Add Derive fees to cost calculation
        # - Taker base fee: $0.5 per options contract
        # - Taker/maker rates from ticker (taker_fee_rate, maker_fee_rate)
        # - If hedging with perps, include expected funding carry
        
        # For binary replication, normalize to $1 payoff
        spread_width = strike_high - strike_low
        if spread_width <= 0:
            return None
            
        normalized_cost = total_cost / spread_width
        
        # Check if cost is reasonable (0 < cost < 1 for normalized binary)
        if normalized_cost <= 0 or normalized_cost >= 1:
            return None
        
        return {
            'strategy_type': strategy_name,
            'expiry': expiry,
            'binary_strike': binary_strike,
            'legs': [
                {
                    'instrument': inst_low,
                    'strike': strike_low,
                    'type': option_type_low,
                    'direction': 'BUY' if direction_low > 0 else 'SELL',
                    'cost': abs(cost_low),
                    'price': ob_low.get('best_ask' if direction_low > 0 else 'best_bid', 0)
                },
                {
                    'instrument': inst_high,
                    'strike': strike_high,
                    'type': option_type_high,
                    'direction': 'BUY' if direction_high > 0 else 'SELL',
                    'cost': abs(cost_high),
                    'price': ob_high.get('best_ask' if direction_high > 0 else 'best_bid', 0)
                }
            ],
            'total_cost': normalized_cost,  # Cost per $1 of binary payoff
            'raw_cost': total_cost,  # Actual dollar cost
            'spread_width': spread_width,
            'strike_low': strike_low,
            'strike_high': strike_high,
            'bid_ask_spread': ob_low.get('spread', 0) + ob_high.get('spread', 0),
            'liquidity': min(
                ob_low.get('liquidity', 0),
                ob_high.get('liquidity', 0)
            ),
            'requires_dynamic_hedge': False
        }
    
    def get_instruments_for_expiry(self, expiry_date):
        """Get all instrument names for a specific expiry"""
        return self.expiry_instruments.get(expiry_date, [])
    
    def get_strikes_for_expiry(self, expiry_date):
        """Get all strikes for a specific expiry date"""
        chain = self.chains_by_expiry.get(expiry_date, {})
        return sorted(list(chain.keys()))
    
    def get_instrument_metadata(self, instrument_name):
        """Get metadata for a specific instrument"""
        return self.options_metadata.get(instrument_name)
    
    def get_expiries_within_days(self, days):
        """Get expiries within specified number of days"""
        cutoff = datetime.now(timezone.utc).timestamp() + (days * 24 * 60 * 60)
        expiries = []
        
        for expiry_date, chain in self.chains_by_expiry.items():
            # Get expiry timestamp from any option in the chain
            for strike_data in chain.values():
                if 'call' in strike_data:
                    expiry_ts = strike_data['call']['metadata']['expiry_timestamp']
                    if expiry_ts <= cutoff:
                        expiries.append(expiry_date)
                    break
        
        return sorted(expiries)
    
    def print_summary(self):
        """Print summary of available options chains"""
        print("\n=== Options Chain Summary ===")
        print(f"Total expiry dates: {len(self.chains_by_expiry)}")
        print(f"Total instruments: {len(self.options_metadata)}")
        
        # Calculate days to expiry for all dates
        now = datetime.now(timezone.utc)
        expiry_info = []
        
        for expiry_date in self.get_expiry_dates():
            chain = self.chains_by_expiry[expiry_date]
            num_strikes = len(chain)
            strikes = sorted(chain.keys())
            
            # Get expiry timestamp from any option
            for strike_data in chain.values():
                if 'call' in strike_data:
                    expiry_ts = strike_data['call']['metadata']['expiry_timestamp']
                    break
                elif 'put' in strike_data:
                    expiry_ts = strike_data['put']['metadata']['expiry_timestamp']
                    break
            
            expiry_dt = datetime.fromtimestamp(expiry_ts, tz=timezone.utc)
            days_to_expiry = (expiry_dt - now).days
            
            # Count puts and calls
            puts = sum(1 for s in chain.values() if 'put' in s)
            calls = sum(1 for s in chain.values() if 'call' in s)
            
            expiry_info.append({
                'date': expiry_date,
                'days': days_to_expiry,
                'strikes': num_strikes,
                'puts': puts,
                'calls': calls,
                'strike_range': (strikes[0], strikes[-1]) if strikes else (0, 0)
            })
        
        # Sort by days to expiry
        expiry_info.sort(key=lambda x: x['days'])
        
        # Show all expiries
        print("\nAll expiries (sorted by days to expiry):")
        for i, info in enumerate(expiry_info):
            print(f"\nExpiry {i+1}: {info['date']} ({info['days']} days)")
            print(f"  Strikes: {info['strikes']} (from {info['strike_range'][0]:.0f} to {info['strike_range'][1]:.0f})")
            print(f"  Options: {info['puts']} puts, {info['calls']} calls")
    
    def _create_single_option_strategy(
        self,
        chain: Dict,
        strike: float,
        option_type: str,
        direction: int,
        expiry: str,
        strategy_name: str,
        orderbook_handler
    ) -> Optional[Dict]:
        """Create single option strategy (requires dynamic hedging)"""
        opt = chain.get(strike, {}).get(option_type, {})
        if not opt:
            return None
            
        inst = opt.get('instrument_name')
        if not inst:
            return None
            
        ob = orderbook_handler.get_latest_orderbook(inst)
        if not ob:
            return None
        # Require two-sided book so we can both enter and exit
        if not self._two_sided_ok(ob, self.min_top_qty_for_exit, self.min_top_notional_for_exit):
            return None
        
        # Calculate cost
        if direction > 0:  # Buy
            cost = ob.get('best_ask', 0)
            if cost == 0:
                return None
        else:  # Sell
            cost = -ob.get('best_bid', 0)
            if ob.get('best_bid', 0) == 0:
                return None
        
        return {
            'strategy_type': strategy_name,
            'expiry': expiry,
            'requires_dynamic_hedge': True,
            'legs': [
                {
                    'instrument': inst,
                    'strike': strike,
                    'type': option_type,
                    'direction': 'BUY' if direction > 0 else 'SELL',
                    'cost': abs(cost),
                    'price': ob.get('best_ask' if direction > 0 else 'best_bid', 0)
                }
            ],
            'total_cost': abs(cost),
            'bid_ask_spread': ob.get('spread', 0),
            'liquidity': ob.get('liquidity', 0),
            'implied_volatility': ob.get('implied_volatility', DEFAULT_VOLATILITY)
        }