"""
Enhanced perpetuals data collector with funding rate integration
Fixed to use correct API endpoints and response structure

FIXED ISSUES:
1. Added execution price calculation based on bid/ask and direction
2. Better handling of missing data
3. Added slippage estimation based on order size
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from utils.debug_recorder import get_recorder  # type: ignore

from config_manager import LYRA_API_BASE, LYRA_PERPS_CURRENCIES, FUNDING_HISTORY_DAYS
from scripts.data_collection.funding_rate_collector import FundingRateCollector
from utils import http_client as rest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from market_data_analyzer import MarketDataAnalyzer


class PerpsDataCollector:
    """Collect and manage perpetual futures data from Lyra with funding history"""
    
    def __init__(self):
        self.api_base = LYRA_API_BASE
        self.logger = logging.getLogger('PerpsDataCollector')
        self.perps_data = {}
        self.funding_collector = FundingRateCollector(self.api_base)
        self.funding_history_days = FUNDING_HISTORY_DAYS  # Days of history to fetch (configurable)
        self.market_analyzer = MarketDataAnalyzer()
        
    def fetch_perp_info(self, currency: str, fetch_funding_history: bool = True) -> Optional[Dict]:
        """Fetch perpetual contract information with funding history"""
        if currency not in LYRA_PERPS_CURRENCIES:
            self.logger.warning(f"{currency} does not have perps on Lyra")
            return None
        
        perp_symbol = LYRA_PERPS_CURRENCIES[currency]
        
        try:
            # Get ticker data which includes best bid/ask, mark price, and instrument details
            ticker_response = rest.post(
                f"{self.api_base}/public/get_ticker",
                json={"instrument_name": perp_symbol}
            )
            
            if ticker_response.status_code != 200:
                self.logger.error(f"Failed to get ticker for {perp_symbol}: {ticker_response.status_code}")
                return None
            
            ticker_data = ticker_response.json()
            ticker_result = ticker_data.get('result', {})
            
            # Extract pricing data from ticker
            mark_price = float(ticker_result.get('mark_price', 0)) if ticker_result.get('mark_price') else None
            best_bid = float(ticker_result.get('best_bid_price', 0)) if ticker_result.get('best_bid_price') else None
            best_ask = float(ticker_result.get('best_ask_price', 0)) if ticker_result.get('best_ask_price') else None
            index_price = float(ticker_result.get('index_price', 0)) if ticker_result.get('index_price') else None
            
            # Get perp details from ticker
            perp_details = ticker_result.get('perp_details', {})
            
            # Get instrument details
            instrument_response = rest.post(
                f"{self.api_base}/public/get_instrument",
                json={"instrument_name": perp_symbol}
            )
            
            if instrument_response.status_code == 200:
                instrument_data = instrument_response.json()
                instrument = instrument_data.get('result', {})
            else:
                instrument = {}
                self.logger.warning(f"Failed to get instrument details for {perp_symbol}")
            
            # Fetch funding history if requested
            funding_stats = self._get_funding_statistics(perp_symbol) if fetch_funding_history else {}
            
            # Derive reports hourly funding rate in perp_details.funding_rate
            funding_rate_hourly = None
            if perp_details and 'funding_rate' in perp_details and perp_details['funding_rate'] is not None:
                try:
                    funding_rate_hourly = float(perp_details['funding_rate'])
                except Exception:
                    funding_rate_hourly = None
            
            # Fallback to funding stats if ticker doesn't have it
            if funding_rate_hourly is None:
                funding_rate_hourly = float(funding_stats.get('current_rate', 0) or 0)
            
            # Calculate execution prices based on bid/ask
            if best_bid and best_ask:
                spread = best_ask - best_bid
                spread_pct = (spread / mark_price * 100) if mark_price else 0
                
                # Estimate execution prices for different position sizes
                # These would be refined with orderbook depth data
                execution_prices = {
                    'buy_small': best_ask,  # Small orders hit the ask
                    'sell_small': best_bid,  # Small orders hit the bid
                    'buy_medium': best_ask + spread * 0.25,  # Medium orders face slippage
                    'sell_medium': best_bid - spread * 0.25,
                    'buy_large': best_ask + spread * 0.5,  # Large orders face more slippage
                    'sell_large': best_bid - spread * 0.5
                }
            else:
                spread = None
                spread_pct = None
                execution_prices = {}
            
            # Calculate premium index as (F - S) / S where F is mark price and S is index/oracle price
            premium_index = 0
            if index_price and index_price > 0:
                premium_index = (mark_price - index_price) / index_price if mark_price else 0
            
            perp_info = {
                'instrument_name': perp_symbol,
                'currency': currency,
                'mark_price': mark_price,
                'index_price': index_price,  # Oracle/spot price for Derive
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_pct': spread_pct,
                'execution_prices': execution_prices,
                'funding_rate': funding_rate_hourly,  # Hourly funding rate for Derive
                'funding_rate_8h': (funding_rate_hourly * 8) if funding_rate_hourly is not None else 0.0,  # 8h equivalent
                'premium_index': premium_index,  # (F-S)/S using index price
                'interest_rate_8h': 0.0001,  # Default 0.01% per 8h (typical exchange value)
                'funding_interval_hours': 1.0,  # Derive uses hourly funding
                'avg_funding_rate': funding_stats.get('avg_rate', funding_rate_hourly),  # Historical average
                'funding_rate_std': funding_stats.get('std_rate', 0),  # Standard deviation
                'funding_rate_min': funding_stats.get('min_rate', funding_rate_hourly),
                'funding_rate_max': funding_stats.get('max_rate', funding_rate_hourly),
                'funding_rate_percentiles': funding_stats.get('percentiles', {}),
                # Fees from Derive ticker
                'taker_fee': float(ticker_result.get('taker_fee_rate') or 0.0),
                'maker_fee': float(ticker_result.get('maker_fee_rate') or 0.0),
                'base_fee': float(ticker_result.get('base_fee') or 0.0),  # $0.10 per trade for Derive perps
                'tick_size': float(ticker_result.get('tick_size')) if ticker_result.get('tick_size') else float(instrument.get('tick_size')) if instrument.get('tick_size') else None,
                'contract_size': float(instrument.get('contract_size', 1)),
                'is_active': ticker_result.get('is_active', True),
                'timestamp': datetime.now().isoformat(),
                'funding_data_points': funding_stats.get('data_points', 0)
            }
            
            # Log warnings for missing critical data
            if perp_info['taker_fee'] == 0:
                self.logger.warning(f"Missing or zero taker fee for {currency} perpetual")
            if perp_info['maker_fee'] == 0:
                self.logger.warning(f"Missing or zero maker fee for {currency} perpetual")
            if perp_info['tick_size'] is None:
                self.logger.warning(f"Missing tick size for {currency} perpetual")
                
            self.perps_data[currency] = perp_info
            try:
                get_recorder().dump_json(f"perps/{currency}_perps.json", perp_info, category="perps")
                self.logger.info("Perps data collection successful for %s", currency)
            except Exception:
                pass
            # Format values for logging
            mark_str = f"${mark_price:,.2f}" if mark_price is not None else "N/A"
            bid_str = f"${best_bid:,.2f}" if best_bid is not None else "N/A"
            ask_str = f"${best_ask:,.2f}" if best_ask is not None else "N/A"
            spread_str = f"{spread_pct:.2f}%" if spread_pct is not None else "N/A"
            
            # Format funding rate with null check
            funding_str = f"{funding_rate_hourly:.4%}/hr" if funding_rate_hourly is not None else "N/A"
            avg_funding = funding_stats.get('avg_rate', funding_rate_hourly)
            avg_funding_str = f"{avg_funding:.4%}/hr" if avg_funding is not None else "N/A"
            
            self.logger.info(
                f"Fetched perp data for {currency}: "
                f"mark={mark_str}, "
                f"bid={bid_str}, "
                f"ask={ask_str}, "
                f"spread={spread_str}, "
                f"current_funding={funding_str}, "
                f"avg_funding={avg_funding_str}"
            )
            
            # Validate critical data
            if mark_price is None or mark_price <= 0:
                error_msg = f"CRITICAL: Invalid mark price for {currency} perpetual: {mark_price}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if 'funding_rate' not in perp_info:
                error_msg = f"CRITICAL: No funding rate data for {currency} perpetual"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            return perp_info
                
        except Exception as e:
            self.logger.error(f"Error fetching perp data for {currency}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _get_funding_statistics(self, instrument_name: str) -> Dict:
        """Get funding rate statistics from historical data"""
        try:
            # Fetch funding history
            success = self.funding_collector.fetch_funding_rate_history(
                instrument_name,
                period=3600  # Hourly funding
            )
            
            if not success:
                return {}
            
            # Get DataFrame
            df = self.funding_collector.get_funding_rates_df(instrument_name, period=3600)
            
            if df.empty:
                return {}
            
            # Calculate statistics
            funding_rates = df['funding_rate'].values
            
            stats = {
                'current_rate': float(df.iloc[-1]['funding_rate']),
                'avg_rate': float(funding_rates.mean()),
                'std_rate': float(funding_rates.std()),
                'min_rate': float(funding_rates.min()),
                'max_rate': float(funding_rates.max()),
                'percentiles': {
                    '5': float(np.percentile(funding_rates, 5)),
                    '25': float(np.percentile(funding_rates, 25)),
                    '50': float(np.percentile(funding_rates, 50)),
                    '75': float(np.percentile(funding_rates, 75)),
                    '95': float(np.percentile(funding_rates, 95))
                },
                'data_points': len(df),
                'date_range': {
                    'start': df['datetime'].min().isoformat(),
                    'end': df['datetime'].max().isoformat()
                }
            }
            
            # Calculate funding rate trends
            if len(df) > 24:  # At least 24 hours of data
                recent_24h = df.tail(24)['funding_rate'].mean()
                older = df.head(len(df) - 24)['funding_rate'].mean()
                stats['trend_24h'] = 'increasing' if recent_24h > older else 'decreasing'
                stats['recent_24h_avg'] = float(recent_24h)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating funding statistics: {e}")
            return {}
    
    def fetch_all_perps(self) -> Dict[str, Dict]:
        """Fetch data for all supported perpetuals with funding history"""
        # First fetch all funding histories
        self.logger.info("Fetching funding rate histories for all perps...")
        
        # Get all perpetual instruments first
        all_perp_instruments = []
        for currency in LYRA_PERPS_CURRENCIES.keys():
            if currency in LYRA_PERPS_CURRENCIES:
                all_perp_instruments.append(LYRA_PERPS_CURRENCIES[currency])
        
        # Fetch funding for each instrument
        for instrument in all_perp_instruments:
            try:
                self.funding_collector.fetch_funding_rate_history(
                    instrument,
                    period=3600,
                    start_timestamp=int((datetime.now() - timedelta(days=self.funding_history_days)).timestamp() * 1000)
                )
            except Exception as e:
                self.logger.error(f"Error fetching funding for {instrument}: {e}")
        
        # Then fetch perp info with stats
        for currency in LYRA_PERPS_CURRENCIES.keys():
            try:
                self.fetch_perp_info(currency, fetch_funding_history=True)
            except Exception as e:
                self.logger.error(f"Error fetching perp info for {currency}: {e}")
                # Continue with other currencies
                continue
        
        return self.perps_data
    
    def estimate_funding_cost_distribution(
        self,
        currency: str,
        position_size: float,
        days_to_expiry: float,
        confidence_level: float = 0.95
    ) -> Dict:
        """Estimate funding cost distribution based on historical data"""
        perp_data = self.get_perp_data(currency)
        if not perp_data:
            return {}
        
        instrument_name = perp_data['instrument_name']
        df = self.funding_collector.get_funding_rates_df(instrument_name, period=3600)
        
        if df.empty:
            return {}
        
        hours_to_expiry = days_to_expiry * 24
        funding_rates = df['funding_rate'].values
        
        # Monte Carlo simulation for funding cost
        n_simulations = 1000
        simulated_costs = []
        
        for _ in range(n_simulations):
            # Random walk through funding rates
            sampled_rates = np.random.choice(funding_rates, size=int(hours_to_expiry), replace=True)
            total_cost = position_size * sampled_rates.sum()
            simulated_costs.append(total_cost)
        
        simulated_costs = np.array(simulated_costs)
        
        # Calculate statistics
        return {
            'expected_cost': float(simulated_costs.mean()),
            'std_cost': float(simulated_costs.std()),
            'min_cost': float(simulated_costs.min()),
            'max_cost': float(simulated_costs.max()),
            'percentiles': {
                '5': float(np.percentile(simulated_costs, 5)),
                '50': float(np.percentile(simulated_costs, 50)),
                '95': float(np.percentile(simulated_costs, 95))
            },
            'var_95': float(np.percentile(simulated_costs, 95)),  # Value at Risk
            'probability_positive': float((simulated_costs > 0).mean()),
            'position_size': position_size,
            'hours_to_expiry': hours_to_expiry
        }
    
    def get_perp_data(self, currency: str) -> Optional[Dict]:
        """Get cached perp data"""
        return self.perps_data.get(currency)
    
    def get_funding_collector(self) -> FundingRateCollector:
        """Get the funding rate collector instance"""
        return self.funding_collector
    
    def get_margin_requirement(self, currency: str, position_size: float, side: str = 'long') -> Optional[Dict]:
        """
        Get actual margin requirement for a position using public/get_margin
        
        Args:
            currency: Currency symbol (e.g., 'BTC', 'ETH')
            position_size: Size of position in contracts
            side: 'long' or 'short'
            
        Returns:
            Dict with initial_margin, maintenance_margin, etc.
        """
        if currency not in LYRA_PERPS_CURRENCIES:
            return None
            
        perp_symbol = LYRA_PERPS_CURRENCIES[currency]
        
        try:
            # Build simulated portfolio for margin calculation
            position_value = position_size if side == 'long' else -position_size
            payload = {
                "positions": [
                    {
                        "instrument_name": perp_symbol,
                        "size": position_value,
                        "kind": "perp"
                    }
                ]
            }
            
            response = rest.post(
                f"{self.api_base}/public/get_margin",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('result', {})
                
                return {
                    'initial_margin': float(result.get('initial_margin', 0)),
                    'maintenance_margin': float(result.get('maintenance_margin', 0)),
                    'margin_balance': float(result.get('margin_balance', 0)),
                    'available_funds': float(result.get('available_funds', 0)),
                    'currency': currency,
                    'position_size': position_size,
                    'side': side
                }
            else:
                self.logger.error(f"Failed to get margin for {perp_symbol}: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting margin requirement: {e}")
            return None
    
    def calculate_delta_hedge_params(
        self,
        binary_delta: float,
        position_size: float,
        currency: str,
        days_to_expiry: float = 1.0
    ) -> Optional[Dict]:
        """Calculate parameters for delta hedging with perps"""
        perp_data = self.get_perp_data(currency)
        if not perp_data:
            return None
        
        mark_price = perp_data['mark_price']
        contract_size = perp_data['contract_size']
        
        if not mark_price:
            self.logger.error(f"No mark price available for {currency}")
            return None
        
        # Calculate required perp position
        # Binary option delta is in terms of underlying
        # Need to convert to perp contracts
        notional_hedge = abs(binary_delta) * position_size
        contracts_needed = notional_hedge / (mark_price * contract_size)
        
        # Round to tick size
        tick_size = perp_data['tick_size']
        contracts_rounded = round(contracts_needed / tick_size) * tick_size
        
        # Calculate costs
        notional_value = contracts_rounded * mark_price * contract_size
        # Note: For actual margin requirement, call public/get_margin with position
        # This is just an estimate - actual margin is portfolio-dependent
        margin_estimate = notional_value * 0.10  # 10% estimate, actual varies
        
        # Estimate execution price based on size using dynamic thresholds
        execution_prices = perp_data.get('execution_prices', {})
        
        # Get dynamic order size thresholds based on daily volume
        daily_volume = perp_data.get('volume_24h', 0)
        thresholds = self.market_analyzer.calculate_order_size_thresholds(
            daily_volume=daily_volume,
            currency=currency
        )
        
        if notional_value < thresholds['small']:  # Small order
            exec_price = execution_prices.get('buy_small' if binary_delta > 0 else 'sell_small', mark_price)
        elif notional_value < thresholds['medium']:  # Medium order
            exec_price = execution_prices.get('buy_medium' if binary_delta > 0 else 'sell_medium', mark_price)
        else:  # Large order
            exec_price = execution_prices.get('buy_large' if binary_delta > 0 else 'sell_large', mark_price)
        
        # Calculate slippage
        slippage_cost = abs(exec_price - mark_price) * contracts_rounded * contract_size
        
        # Entry fee using execution price
        entry_fee = notional_value * perp_data['taker_fee']
        
        # Estimate funding cost (funding_rate is hourly per Derive docs)
        hours_to_expiry = days_to_expiry * 24
        funding_cost = notional_value * perp_data['funding_rate'] * hours_to_expiry
        
        return {
            'contracts': contracts_rounded,
            'notional_value': notional_value,
            'margin_estimate': margin_estimate,  # Note: actual margin requires get_margin API
            'entry_fee': entry_fee,
            'estimated_funding_cost': funding_cost,
            'slippage_cost': slippage_cost,
            'total_cost': entry_fee + abs(funding_cost) + slippage_cost,
            'mark_price': mark_price,
            'execution_price': exec_price,
            'funding_rate_hourly': perp_data['funding_rate']
        }
    
    def print_funding_summary(self):
        """Print summary of funding rates across all perps"""
        print("\n=== Perpetual Funding Rate Summary ===")
        
        if not self.perps_data:
            print("No perpetual data available.")
            return
            
        for currency, data in self.perps_data.items():
            try:
                print(f"\n{currency} ({data.get('instrument_name', 'Unknown')}):")
                
                # Safely format mark price
                mark_price = data.get('mark_price')
                if mark_price is not None and mark_price > 0:
                    print(f"  Mark price: ${mark_price:,.2f}")
                else:
                    print("  Mark price: N/A")
                
                # Safely format bid/ask
                best_bid = data.get('best_bid')
                best_ask = data.get('best_ask')
                if best_bid and best_ask:
                    print(f"  Bid/Ask: ${best_bid:,.2f} / ${best_ask:,.2f}")
                    spread_pct = data.get('spread_pct', 0)
                    print(f"  Spread: {spread_pct:.2f}%")
                else:
                    print("  Bid/Ask: N/A")
                
                # Safely format funding rates
                funding_rate = data.get('funding_rate', 0)
                avg_funding_rate = data.get('avg_funding_rate', 0)
                funding_std = data.get('funding_rate_std', 0)
                funding_min = data.get('funding_rate_min', 0)
                funding_max = data.get('funding_rate_max', 0)
                
                print(f"  Current rate: {funding_rate:.4%}/hour")
                print(f"  Average rate: {avg_funding_rate:.4%}/hour")
                print(f"  Std deviation: {funding_std:.4%}")
                print(f"  Min/Max: {funding_min:.4%} / {funding_max:.4%}")
                
                data_points = data.get('funding_data_points', 0)
                if data_points > 0:
                    print(f"  Data points: {data_points}")
                    
                # Annualized rates (funding_rate is hourly per Derive docs)
                annual_current = funding_rate * 24 * 365
                annual_avg = avg_funding_rate * 24 * 365
                print(f"  Annualized current: {annual_current:.2%}")
                print(f"  Annualized average: {annual_avg:.2%}")
                
                # Show execution price estimates
                exec_prices = data.get('execution_prices', {})
                if exec_prices:
                    print("  Execution prices (estimated):")
                    buy_small = exec_prices.get('buy_small', 0)
                    sell_small = exec_prices.get('sell_small', 0)
                    buy_large = exec_prices.get('buy_large', 0)
                    sell_large = exec_prices.get('sell_large', 0)
                    if buy_small and sell_small:
                        print(f"    Small orders: Buy ${buy_small:,.2f} / Sell ${sell_small:,.2f}")
                    if buy_large and sell_large:
                        print(f"    Large orders: Buy ${buy_large:,.2f} / Sell ${sell_large:,.2f}")
                        
            except Exception as e:
                print(f"  Error displaying data for {currency}: {e}")
