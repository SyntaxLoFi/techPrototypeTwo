"""
Funding Rate Collector for Lyra Perpetual Futures

Fetches and stores funding rate history for perpetual futures
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from config_manager import HTTP_TIMEOUT_SEC
from utils import http_client as rest


class FundingRateCollector:
    """Fetch and store funding rate data from Lyra perpetuals"""
    
    def __init__(self, api_base="https://api.lyra.finance"):
        self.api_base = api_base
        self.funding_rates = {}
        self.perpetual_instruments = []
        
    def get_perpetual_instruments(self, currency="ETH") -> List[str]:
        """
        Get list of perpetual future instruments for a currency
        
        Args:
            currency: Base currency (default: ETH)
            
        Returns:
            List of perpetual instrument names
        """
        try:
            # Use POST with JSON body for Derive/Lyra API
            # Note: get_instruments doesn't support pagination
            payload = {
                "instrument_type": "perp",  # Must be instrument_type per docs
                "currency": currency,
                "expired": False
            }
            
            response = rest.post(
                f"{self.api_base}/public/get_instruments",  # Changed to get_instruments
                json=payload  # Changed from params to json
            )
            
            if response.status_code == 200:
                data = response.json()
                instruments = data["result"]  # CORRECTED: result is a list, not {"instruments": [...]}
                
                self.perpetual_instruments = [
                    inst["instrument_name"] for inst in instruments
                    if inst.get("is_active", False)
                ]
                
                print(f"Found {len(self.perpetual_instruments)} active perpetual instruments")
                return self.perpetual_instruments
            else:
                print(f"Failed to fetch perpetual instruments: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching perpetual instruments: {str(e)}")
            return []
    
    def fetch_funding_rate_history(self, instrument_name: str, 
                                 start_timestamp: Optional[int] = None,
                                 end_timestamp: Optional[int] = None,
                                 period: int = 3600) -> bool:
        """
        Fetch funding rate history for an instrument
        
        Args:
            instrument_name: Perpetual instrument name
            start_timestamp: Start time in ms (default: 30 days ago)
            end_timestamp: End time in ms (default: now)
            period: Funding period in seconds - must be one of: 900, 3600, 14400, 28800, 86400
            
        Returns:
            bool: Success status
        """
        try:
            # Default timestamps (Derive caps at ~30d history on this endpoint)
            if end_timestamp is None:
                end_timestamp = int(datetime.now().timestamp() * 1000)
            
            if start_timestamp is None:
                # Default to 30 days ago
                start_timestamp = end_timestamp - (30 * 24 * 60 * 60 * 1000)
            
            # Validate period (seconds)
            allowed = {900, 3600, 14400, 28800, 86400}
            if period not in allowed:
                period = 3600
            
            # Use POST with JSON body for Derive/Lyra API
            payload = {
                "instrument_name": instrument_name,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "period": int(period)
            }
            
            response = rest.post(
                f"{self.api_base}/public/get_funding_rate_history",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                funding_history = data["result"]["funding_rate_history"]
                
                # Store by instrument and period
                key = f"{instrument_name}_{period}"
                self.funding_rates[key] = funding_history
                
                print(f"Fetched {len(funding_history)} funding rates for {instrument_name}")
                return True
            else:
                print(f"Failed to fetch funding rates: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error fetching funding rates: {str(e)}")
            return False
    
    def fetch_all_funding_rates(self, currency="ETH", period: int = 3600,
                              days_back: int = 30) -> bool:
        """
        Fetch funding rates for all perpetual instruments
        
        Args:
            currency: Base currency
            period: Funding period in seconds
            days_back: Number of days to fetch
            
        Returns:
            bool: Success status
        """
        # First get all perpetual instruments
        instruments = self.get_perpetual_instruments(currency)
        
        if not instruments:
            print("No perpetual instruments found")
            return False
        
        # Calculate timestamps
        end_timestamp = int(datetime.now().timestamp() * 1000)
        start_timestamp = end_timestamp - (days_back * 24 * 60 * 60 * 1000)
        
        success_count = 0
        
        # Fetch funding rates for each instrument
        for instrument in instruments:
            if self.fetch_funding_rate_history(
                instrument, start_timestamp, end_timestamp, period
            ):
                success_count += 1
        
        print(f"Successfully fetched funding rates for {success_count}/{len(instruments)} instruments")
        return success_count > 0
    
    def get_funding_rates_df(self, instrument_name: str, period: int = 3600) -> pd.DataFrame:
        """
        Get funding rates as a DataFrame
        
        Args:
            instrument_name: Perpetual instrument name
            period: Funding period
            
        Returns:
            DataFrame with funding rate history
        """
        key = f"{instrument_name}_{period}"
        
        if key not in self.funding_rates:
            return pd.DataFrame()
        
        rates = self.funding_rates[key]
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        if not df.empty:
            # Convert timestamps to datetime
            df["datetime"] = pd.to_datetime(df["timestamp"], unit='ms')
            
            # Convert funding rate to float
            df["funding_rate"] = df["funding_rate"].astype(float)
            
            # Calculate annualized rate (funding_rate is hourly per Derive docs)
            # Need to convert from per-period rate to hourly first
            hours_per_period = period / 3600
            df["funding_rate_hourly"] = df["funding_rate"] / hours_per_period
            # Then annualize
            df["funding_rate_annualized"] = df["funding_rate_hourly"] * 24 * 365
            
            # Sort by timestamp
            df.sort_values("timestamp", inplace=True)
        
        return df
    
    def calculate_funding_cost(self, instrument_name: str, 
                             position_size: float,
                             start_time: datetime,
                             end_time: datetime,
                             period: int = 3600) -> Dict:
        """
        Calculate funding cost for a position over a time period
        
        Args:
            instrument_name: Perpetual instrument name
            position_size: Position size (positive for long, negative for short)
            start_time: Start datetime
            end_time: End datetime
            period: Funding period
            
        Returns:
            Dict with funding cost details
        """
        df = self.get_funding_rates_df(instrument_name, period)
        
        if df.empty:
            return {"error": "No funding rate data available"}
        
        # Filter to time period
        mask = (df["datetime"] >= start_time) & (df["datetime"] <= end_time)
        period_df = df[mask].copy()
        
        if period_df.empty:
            return {"error": "No funding rate data for specified period"}
        
        # Calculate funding payments
        # Positive funding rate: longs pay shorts
        # Negative funding rate: shorts pay longs
        period_df["funding_payment"] = -position_size * period_df["funding_rate"]
        
        total_funding = period_df["funding_payment"].sum()
        avg_funding_rate = period_df["funding_rate"].mean()
        
        return {
            "instrument": instrument_name,
            "position_size": position_size,
            "start_time": start_time,
            "end_time": end_time,
            "total_funding_cost": total_funding,
            "average_funding_rate": avg_funding_rate,
            "num_periods": len(period_df),
            "funding_payments": period_df[["datetime", "funding_rate", "funding_payment"]].to_dict('records')
        }
    
    def save_funding_data(self, output_dir: str = "data/funding_rates"):
        """
        Save funding rate data to CSV files
        
        Args:
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for key, rates in self.funding_rates.items():
            if not rates:
                continue
            
            # Parse instrument and period from key
            parts = key.rsplit('_', 1)
            instrument = parts[0]
            period = parts[1]
            
            # Get DataFrame
            df = self.get_funding_rates_df(instrument.split('_')[0], int(period))
            
            if not df.empty:
                filename = f"{instrument}_funding_{period}s_{timestamp}.csv"
                filepath = os.path.join(output_dir, filename)
                
                df.to_csv(filepath, index=False)
                print(f"Saved {len(df)} funding rates to {filepath}")
    
    def print_summary(self):
        """Print summary of funding rate data"""
        print("\n=== Funding Rate Summary ===")
        print(f"Perpetual instruments: {len(self.perpetual_instruments)}")
        
        for instrument in self.perpetual_instruments:
            print(f"\n{instrument}:")
            
            # Check different periods
            for period in [3600]:  # Default to hourly
                df = self.get_funding_rates_df(instrument, period)
                
                if not df.empty:
                    latest = df.iloc[-1]
                    avg_rate = df["funding_rate"].mean()
                    avg_annualized = df["funding_rate_annualized"].mean()
                    
                    print(f"  Period: {period}s ({period/3600:.1f} hours)")
                    print(f"  Latest rate: {latest['funding_rate']:.6f} ({latest['funding_rate_annualized']:.2%} annualized)")
                    print(f"  Average rate: {avg_rate:.6f} ({avg_annualized:.2%} annualized)")
                    print(f"  Data points: {len(df)}")
                    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
