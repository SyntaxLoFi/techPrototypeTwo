"""
Market Data Analyzer for Dynamic Configuration
Provides real-time market analysis for liquidity, volatility, and other metrics

Based on academic literature:
- Amihud (2002) - Illiquidity measure
- Kyle (1985) - Market depth and price impact
- Hasbrouck (2009) - Effective spread estimation
"""
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime, timedelta

from black_scholes_greeks import BlackScholesGreeks

from config_manager import (
    LIQUIDITY_SAFETY_FACTOR,
    MIN_PM_LIQUIDITY,
    MIN_PERP_LIQUIDITY,
    MIN_OPTIONS_LIQUIDITY,
    SMALL_ORDER_THRESHOLD_PCT,
    MEDIUM_ORDER_THRESHOLD_PCT,
    LARGE_ORDER_THRESHOLD_PCT,
    DEFAULT_IMPLIED_VOLATILITY,
    MARKET_IMPACT_GAMMA,
    ATM_STRIKE_WINDOW_LOW,
    ATM_STRIKE_WINDOW_HIGH,
    SECONDS_PER_YEAR
)

logger = logging.getLogger(__name__)


class MarketDataAnalyzer:
    """Analyzes market data to provide dynamic configuration parameters"""
    
    def __init__(self):
        self.logger = logging.getLogger('MarketDataAnalyzer')
        self._volatility_cache = {}  # Now keyed by (currency, expiry_bucket)
        self._greeks_cache = {}      # Cache for Greeks calculations
        self._volume_cache = {}
        self._liquidity_cache = {}
        self._cache_ttl = 300  # 5 minutes cache
        
    def calculate_dynamic_liquidity_constraints(
        self,
        currency: str,
        orderbook_data: Optional[Dict] = None,
        market_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Calculate liquidity constraints based on real market data
        
        Uses Kyle's Lambda and Amihud Illiquidity measures from academic literature
        """
        constraints = {
            'pm_max_usd': MIN_PM_LIQUIDITY,
            'perp_max_usd': MIN_PERP_LIQUIDITY,
            'options_max_usd': MIN_OPTIONS_LIQUIDITY
        }
        
        try:
            # 1. Order Book Depth Analysis (Kyle, 1985)
            if orderbook_data:
                depth_liquidity = self._analyze_orderbook_depth(orderbook_data)
                constraints['perp_max_usd'] = max(
                    MIN_PERP_LIQUIDITY,
                    depth_liquidity * LIQUIDITY_SAFETY_FACTOR
                )
            
            # 2. Amihud Illiquidity Adjustment
            # If we have historical price/volume data, calculate true Amihud measure
            if market_data:
                amihud_illiq = self._calculate_amihud_illiquidity(market_data)
                if amihud_illiq is not None:
                    # Higher illiquidity → lower position limits
                    # Amihud values typically range from 10^-9 to 10^-6 for liquid assets
                    if amihud_illiq > 1e-6:  # Illiquid
                        liquidity_multiplier = 0.3
                    elif amihud_illiq > 1e-7:  # Moderate liquidity
                        liquidity_multiplier = 0.6
                    else:  # Liquid
                        liquidity_multiplier = 1.0
                    
                    constraints['perp_max_usd'] *= liquidity_multiplier
                
                # Fallback: Volume-based cap if no price history
                elif 'volume_24h' in market_data:
                    volume_24h = market_data['volume_24h']
                    # Academic research suggests 1-5% of daily volume as max single trade
                    volume_based_limit = volume_24h * 0.02  # 2% of daily volume
                    
                    constraints['perp_max_usd'] = min(
                        constraints['perp_max_usd'],
                        volume_based_limit * LIQUIDITY_SAFETY_FACTOR
                    )
            
            # 3. Effective Spread Adjustment (Hasbrouck, 2009)
            # Calculate effective spread if we have recent trade data
            if market_data:
                effective_spread = self._calculate_effective_spread(market_data)
                if effective_spread is not None:
                    # Hasbrouck shows effective spread captures true trading costs
                    # Adjust liquidity based on effective spread vs quoted spread
                    quoted_spread = market_data.get('spread_percentage', effective_spread)
                    
                    if quoted_spread > 0:
                        # Ratio > 1 means effective spread > quoted (adverse selection)
                        spread_ratio = effective_spread / quoted_spread
                        
                        if spread_ratio > 2.0:  # High adverse selection
                            liquidity_factor = 0.3
                        elif spread_ratio > 1.5:
                            liquidity_factor = 0.5
                        elif spread_ratio > 1.2:
                            liquidity_factor = 0.7
                        else:
                            liquidity_factor = 1.0
                        
                        # Only scale perps liquidity from perps microstructure
                        constraints['perp_max_usd'] *= liquidity_factor
                
                # Fallback to quoted spread tiers if no trade data
                elif 'spread_percentage' in market_data:
                    spread_pct = market_data['spread_percentage']
                    
                    # Guard against None or non-numeric inputs
                    try:
                        spread_val = float(spread_pct)
                    except (TypeError, ValueError):
                        spread_val = None
                    
                    # Wide spreads indicate low liquidity
                    if spread_val is not None and spread_val > 1.0:  # >1% spread
                        liquidity_factor = 0.3
                    elif spread_val is not None and spread_val > 0.5:  # 0.5-1% spread
                        liquidity_factor = 0.5
                    elif spread_val is not None and spread_val > 0.2:  # 0.2-0.5% spread
                        liquidity_factor = 0.7
                    elif spread_val is not None:  # <0.2% spread (high liquidity)
                        liquidity_factor = 1.0
                    
                    # Only scale perps liquidity from perps microstructure
                    constraints['perp_max_usd'] *= liquidity_factor
            
            # 4/5. Do not cross-cap options or PM using perps; keep them independent.
            # Leave 'options_max_usd' and 'pm_max_usd' as independently configured unless we have venue-specific signals.
            
            self.logger.info(f"Dynamic liquidity constraints for {currency}: {constraints}")
            
        except Exception as e:
            self.logger.warning(f"Error calculating dynamic liquidity: {e}, using defaults")
        
        return constraints
    
    def _winsorize_array(self, arr, lower: float = 0.01, upper: float = 0.99):
        """
        In-place-safe winsorization that returns a clipped copy (nan-safe).
        """
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return a
        lo = np.nanpercentile(a, lower * 100.0)
        hi = np.nanpercentile(a, upper * 100.0)
        return np.clip(a, lo, hi)
    
    def _analyze_orderbook_depth(
        self,
        orderbook: Dict,
        target_move_pct: float = 0.005,
        max_levels: int = 50
    ) -> float:
        """
        Robustly approximate Kyle depth (inverse λ) by walking both sides of the book
        to reach a symmetric target move in the mid price.

        Returns the **USD notional** required to move the mid by `target_move_pct`.
        Higher notional => more depth (lower λ). See Kyle (1985).  # Δp = λ·q + ε

        Guards:
          - drops invalid/zero/negative levels
          - fixes locked/crossed books by trimming until ask > bid
          - caps levels considered
          - uses fractional level at the boundary when target is between ticks
        """
        try:
            bids = orderbook.get('bids', []) or []
            asks = orderbook.get('asks', []) or []
            if not bids or not asks:
                return 0.0

            def _clean(side):
                out = []
                for lvl in side[:max_levels]:
                    if not isinstance(lvl, (list, tuple)) or len(lvl) < 2:
                        continue
                    p, q = float(lvl[0]), float(lvl[1])
                    if np.isfinite(p) and np.isfinite(q) and p > 0 and q > 0:
                        out.append([p, q])
                return out

            bids = sorted(_clean(bids), key=lambda x: -x[0])
            asks = sorted(_clean(asks), key=lambda x: x[0])
            if not bids or not asks:
                return 0.0

            # Unlock/uncross by trimming the smaller-notional top until ask > bid
            while asks and bids and asks[0][0] <= bids[0][0]:
                if asks[0][0] * asks[0][1] <= bids[0][0] * bids[0][1]:
                    asks.pop(0)
                else:
                    bids.pop(0)
            if not bids or not asks:
                return 0.0

            mid = 0.5 * (asks[0][0] + bids[0][0])
            up_target = mid * (1.0 + target_move_pct)
            dn_target = mid * (1.0 - target_move_pct)

            # Walk up through asks
            buy_notional = 0.0
            last_p = bids[0][0]
            for p, q in asks:
                if p >= up_target:
                    # fraction needed within this level
                    denom = max(p - last_p, 1e-12)
                    frac = max(0.0, min(1.0, (up_target - last_p) / denom))
                    buy_notional += frac * q * p
                    break
                buy_notional += q * p
                last_p = p

            # Walk down through bids
            sell_notional = 0.0
            last_p = asks[0][0]
            for p, q in bids:
                if p <= dn_target:
                    denom = max(last_p - p, 1e-12)
                    frac = max(0.0, min(1.0, (last_p - dn_target) / denom))
                    sell_notional += frac * q * p
                    break
                sell_notional += q * p
                last_p = p

            notionals = [n for n in (buy_notional, sell_notional) if n > 0]
            return float(np.mean(notionals)) if notionals else 0.0

        except Exception as e:
            self.logger.error(f"Kyle depth failed: {e}")
            return 0.0
    
    def calculate_order_size_thresholds(
        self,
        daily_volume: float,
        currency: str
    ) -> Dict[str, float]:
        """
        Calculate order size thresholds based on daily volume
        Based on Admati & Pfleiderer (1988) - optimal trade size
        """
        if daily_volume <= 0:
            # Fallback to currency-specific defaults
            defaults = {
                'BTC': {'small': 1000, 'medium': 10000, 'large': 50000},
                'ETH': {'small': 500, 'medium': 5000, 'large': 25000},
                'SOL': {'small': 100, 'medium': 1000, 'large': 5000},
                'XRP': {'small': 50, 'medium': 500, 'large': 2500},
                'DOGE': {'small': 50, 'medium': 500, 'large': 2500}
            }
            return defaults.get(currency, {'small': 100, 'medium': 1000, 'large': 5000})
        
        return {
            'small': daily_volume * SMALL_ORDER_THRESHOLD_PCT,
            'medium': daily_volume * MEDIUM_ORDER_THRESHOLD_PCT,
            'large': daily_volume * LARGE_ORDER_THRESHOLD_PCT
        }
    
    def calculate_implied_volatility_from_prices(
        self,
        option_price: float,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.05,
        option_type: str = 'call',
        initial_guess: float = 0.5
    ) -> Optional[float]:
        """
        Compute implied volatility via Newton–Raphson using Black–Scholes forward pricing.

        Args:
            option_price: Observed market price (use mid = (bid+ask)/2)
            spot_price: Current underlying spot price
            strike_price: Option strike
            time_to_expiry: Time to expiry in years
            risk_free_rate: Annualized risk-free rate (continuous compounding)
            option_type: 'call' or 'put'
            initial_guess: Starting volatility guess (annualized, e.g., 0.5 = 50%)

        Returns:
            Implied volatility (annualized, as a decimal), or None if it fails to converge.
        """
        try:
            if option_price is None or option_price <= 0: return None
            if spot_price   is None or spot_price   <= 0: return None
            if strike_price is None or strike_price <= 0: return None
            if time_to_expiry is None or time_to_expiry <= 0: return None

            sigma   = max(float(initial_guess), 1e-6)
            max_it  = 100
            tol     = 1e-3  # |price error| < $0.001

            for _ in range(max_it):
                if option_type.lower() == 'call':
                    model_price = BlackScholesGreeks.call_price(spot_price, strike_price, risk_free_rate, sigma, time_to_expiry)
                    greeks      = BlackScholesGreeks.call_greeks(spot_price, strike_price, risk_free_rate, sigma, time_to_expiry, scale_1pct=True)
                else:
                    model_price = BlackScholesGreeks.put_price(spot_price, strike_price, risk_free_rate, sigma, time_to_expiry)
                    greeks      = BlackScholesGreeks.put_greeks(spot_price, strike_price, risk_free_rate, sigma, time_to_expiry, scale_1pct=True)

                diff = model_price - option_price
                if abs(diff) < tol:
                    return float(sigma)

                # Our greeks return vega per 1% vol move by default — convert to per 1.0 vol.
                vega_per_pct = greeks.get('vega', 0.0)
                vega = float(vega_per_pct) * 100.0

                if abs(vega) < 1e-10:
                    return None

                sigma_new = sigma - diff / vega
                if not np.isfinite(sigma_new) or sigma_new <= 0 or sigma_new > 5.0:
                    return None
                sigma = sigma_new

            return None  # didn't converge

        except Exception as e:
            # Log and fail gracefully
            try:
                self.logger.warning(f"IV solve failed (type={option_type}, K={strike_price}, T={time_to_expiry:.4f}y): {e}")
            except Exception:
                pass
            return None
    
    def get_market_implied_volatility(
        self,
        currency: str,
        options_data: Optional[List[Dict]] = None,
        use_cache: bool = True,
        expiry_days: Optional[int] = None,
        current_spot: Optional[float] = None,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Estimate the market-implied volatility using option prices.

        Process:
        1) Determine spot:
           - Try to extract from options_data (fields like 'spot', 'spot_price', 'underlying_price').
           - Else use current_spot param if provided.
           - Else try put–call parity on near-ATM matched pairs.
        2) For each option:
           - Extract strike, bid, ask, expiry_date ('YYYY-MM-DD'), and type.
           - Skip if bid/ask invalid (<=0) or already expired.
           - Compute mid = (bid+ask)/2 and time_to_expiry in years.
           - Solve for IV via calculate_implied_volatility_from_prices().
        3) Filter IVs to ATM window (95%–105% of spot). Take median.
           If none, expand to 90%–110%. If still none, fall back to default.
        4) Cache by (currency, expiry_bucket).

        Returns:
            Implied volatility as a decimal (e.g., 0.55). Falls back to DEFAULT_IMPLIED_VOLATILITY.
        """
        # Default IV from config, with safe fallback
        try:
            default_iv = DEFAULT_IMPLIED_VOLATILITY
        except Exception:
            default_iv = 0.5

        # Cache key by expiry bucket (same logic you use elsewhere)
        expiry_bucket = "general"
        if expiry_days is not None:
            if   expiry_days <= 7:  expiry_bucket = "7D"
            elif expiry_days <= 30: expiry_bucket = "30D"
            elif expiry_days <= 90: expiry_bucket = "90D"
            else:                   expiry_bucket = "180D+"

        cache_key = (currency, expiry_bucket)
        now = datetime.now()

        if use_cache and cache_key in self._volatility_cache:
            ts, val = self._volatility_cache[cache_key]
            if (now - ts).total_seconds() < self._cache_ttl and val is not None:
                return float(val)

        # Helper to read multiple possible keys
        def _get(d: Dict, *keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        # 1) Spot discovery
        spot = None
        try:
            if options_data:
                # Per-item spot if present
                for row in options_data:
                    cand = _get(row, 'spot', 'spot_price', 'underlying_price', 'underlyingSpot', 'underlying')
                    if isinstance(cand, (int, float)) and cand > 0:
                        spot = float(cand)
                        break
                # Try top-level style metadata on first row too
                if spot is None:
                    meta = options_data[0]
                    for k in ['spot', 'spot_price', 'underlying_price', 'lastUnderlyingPrice']:
                        if k in meta and isinstance(meta[k], (int, float)) and meta[k] > 0:
                            spot = float(meta[k])
                            break
        except Exception:
            pass

        if spot is None and current_spot is not None and current_spot > 0:
            spot = float(current_spot)

        if not options_data:
            self.logger.warning("No options_data provided; returning default implied volatility.")
            self._volatility_cache[cache_key] = (now, default_iv)
            return default_iv

        atm_pairs: List[Tuple[float, float]] = []     # (K, IV)
        all_pairs: List[Tuple[float, float]] = []     # (K, IV)

        # 2) Iterate contracts
        for opt in options_data:
            try:
                K   = _get(opt, 'strike_price', 'strike', 'K')
                bid = _get(opt, 'bid', 'bid_price', 'bidPrice')
                ask = _get(opt, 'ask', 'ask_price', 'askPrice')
                tpe = str(_get(opt, 'option_type', 'type', 'side', default='call')).lower()
                exp = _get(opt, 'expiry_date', 'expiry', 'expiration', 'expirationDate')

                if K is None or bid is None or ask is None:
                    continue
                K, bid, ask = float(K), float(bid), float(ask)
                if bid <= 0 or ask <= 0 or ask < bid:
                    continue
                mid = 0.5 * (bid + ask)

                # Expiry -> time in years
                if exp is None:
                    if expiry_days is None:
                        continue
                    T = float(expiry_days) / 365.0
                    exp_dt = None
                else:
                    try:
                        exp_dt = datetime.strptime(str(exp)[:10], "%Y-%m-%d")
                    except Exception:
                        try:
                            exp_dt = datetime.fromtimestamp(int(exp))
                        except Exception:
                            continue
                    if exp_dt <= now:
                        continue
                    T = (exp_dt - now).total_seconds() / float(SECONDS_PER_YEAR)

                S = spot

                # If no spot, try put–call parity on a matched pair (same K & expiry)
                if S is None:
                    want = 'put' if tpe == 'call' else 'call'
                    match = None
                    for other in options_data:
                        if other is opt:
                            continue
                        K2 = _get(other, 'strike_price', 'strike', 'K')
                        e2 = _get(other, 'expiry_date', 'expiry', 'expiration', 'expirationDate')
                        if K2 is None or e2 is None or float(K2) != K:
                            continue
                        try:
                            exp2 = datetime.strptime(str(e2)[:10], "%Y-%m-%d")
                        except Exception:
                            try:
                                exp2 = datetime.fromtimestamp(int(e2))
                            except Exception:
                                continue
                        if exp_dt and exp2.date() != exp_dt.date():
                            continue
                        typ2 = str(_get(other, 'option_type', 'type', 'side', default='')).lower()
                        if typ2 != want:
                            continue
                        b2 = _get(other, 'bid', 'bid_price', 'bidPrice')
                        a2 = _get(other, 'ask', 'ask_price', 'askPrice')
                        if b2 is None or a2 is None:
                            continue
                        b2, a2 = float(b2), float(a2)
                        if b2 <= 0 or a2 <= 0:
                            continue
                        mid2 = 0.5 * (b2 + a2)
                        match = (mid2, typ2)
                        break

                    if match is not None:
                        other_mid, other_type = match
                        if tpe == 'call' and other_type == 'put':
                            C, P = mid, other_mid
                        elif tpe == 'put' and other_type == 'call':
                            P, C = mid, other_mid
                        else:
                            C = P = None
                        if C is not None and P is not None:
                            S_est = C - P + K * np.exp(-risk_free_rate * T)
                            if np.isfinite(S_est) and S_est > 0:
                                S = float(S_est)

                if S is None or S <= 0:
                    continue

                iv = self.calculate_implied_volatility_from_prices(
                    option_price=mid,
                    spot_price=S,
                    strike_price=K,
                    time_to_expiry=T,
                    risk_free_rate=risk_free_rate,
                    option_type='call' if tpe.startswith('c') else 'put',
                    initial_guess=0.5
                )
                if iv is None or not np.isfinite(iv) or iv <= 0:
                    continue

                m = K / S
                all_pairs.append((K, iv))
                if ATM_STRIKE_WINDOW_LOW <= m <= ATM_STRIKE_WINDOW_HIGH:
                    atm_pairs.append((K, iv))

            except Exception as e:
                self.logger.debug(f"Skipping option due to error: {e}")
                continue

        def _median_iv(pairs: List[Tuple[float, float]]) -> Optional[float]:
            if not pairs:
                return None
            ivs = [iv for _, iv in pairs if iv is not None and np.isfinite(iv)]
            return float(np.median(ivs)) if ivs else None

        market_iv = _median_iv(atm_pairs)

        # widen to 90–110% if no ATM sample
        if market_iv is None and all_pairs and spot is not None and spot > 0:
            widened = [(K, iv) for (K, iv) in all_pairs if 0.90 <= (K / spot) <= 1.10]
            market_iv = _median_iv(widened)

        if market_iv is None:
            market_iv = float(default_iv)

        self._volatility_cache[cache_key] = (now, market_iv)
        try:
            self.logger.info(f"ATM implied volatility for {currency}: {market_iv:.2%}")
        except Exception:
            pass
        return market_iv
    
    def get_cached_greeks(
        self,
        currency: str,
        strike: float,
        expiry_days: int,
        option_type: str,
        spot: float,
        volatility: Optional[float] = None,
        risk_free_rate: float = 0.05
    ) -> Optional[Dict[str, float]]:
        """
        Get cached Greeks calculations with expiry-aware caching.
        
        This improves performance by avoiding redundant Black-Scholes calculations
        for similar options within the same expiry bucket.
        """
        # Determine expiry bucket
        if expiry_days <= 7:
            expiry_bucket = "7D"
        elif expiry_days <= 30:
            expiry_bucket = "30D"
        elif expiry_days <= 90:
            expiry_bucket = "90D"
        else:
            expiry_bucket = "180D+"
            
        # Round strike to nearest significant figure for cache efficiency
        strike_bucket = round(strike, -int(np.log10(strike)) + 2)
        
        # Create cache key
        cache_key = f"{currency}_{option_type}_{strike_bucket}_{expiry_bucket}"
        
        # Check cache
        if cache_key in self._greeks_cache:
            cached_time, cached_greeks, cached_params = self._greeks_cache[cache_key]
            
            # Verify parameters haven't changed significantly
            param_match = (
                abs(cached_params['spot'] - spot) / spot < 0.01 and  # 1% spot tolerance
                abs(cached_params.get('volatility', 0) - (volatility or 0)) < 0.05  # 5% vol tolerance
            )
            
            if (datetime.now() - cached_time).seconds < self._cache_ttl and param_match:
                self.logger.debug(f"Using cached Greeks for {cache_key}")
                return cached_greeks
        
        # If not in cache or expired, return None (caller should calculate)
        return None
    
    def cache_greeks(
        self,
        currency: str,
        strike: float,
        expiry_days: int,
        option_type: str,
        spot: float,
        greeks: Dict[str, float],
        volatility: Optional[float] = None
    ):
        """Store Greeks calculation in expiry-aware cache."""
        # Determine expiry bucket
        if expiry_days <= 7:
            expiry_bucket = "7D"
        elif expiry_days <= 30:
            expiry_bucket = "30D"
        elif expiry_days <= 90:
            expiry_bucket = "90D"
        else:
            expiry_bucket = "180D+"
            
        # Round strike for cache efficiency
        strike_bucket = round(strike, -int(np.log10(strike)) + 2)
        
        # Create cache key
        cache_key = f"{currency}_{option_type}_{strike_bucket}_{expiry_bucket}"
        
        # Store in cache with parameters
        self._greeks_cache[cache_key] = (
            datetime.now(),
            greeks,
            {'spot': spot, 'volatility': volatility}
        )
        
        # Limit cache size
        if len(self._greeks_cache) > 1000:
            # Remove oldest entries
            sorted_keys = sorted(
                self._greeks_cache.keys(),
                key=lambda k: self._greeks_cache[k][0]
            )
            for key in sorted_keys[:200]:  # Remove oldest 200
                del self._greeks_cache[key]
    
    def calculate_adaptive_spread_widths(
        self,
        spot_price: float,
        volatility: Optional[float],
        currency: str
    ) -> Optional[List[float]]:
        """
        Calculate adaptive spread widths based on spot price and volatility
        Based on Glosten & Milgrom (1985) - optimal spread theory
        
        Returns None if volatility data is not available
        """
        if spot_price <= 0:
            return [100, 500, 1000, 5000]  # Fallback for invalid spot
        
        if volatility is None:
            self.logger.warning(f"Cannot calculate spread widths without volatility data for {currency}")
            return None
        
        # Base spread width as function of price and volatility
        # Academic literature suggests spread ∝ √(volatility * time)
        volatility_factor = np.sqrt(volatility / 0.6)  # Normalize to 60% vol baseline
        
        # Different assets have different tick sizes and liquidity profiles
        currency_factors = {
            'BTC': 1.0,    # Baseline
            'ETH': 0.8,    # Tighter spreads due to higher liquidity
            'SOL': 1.5,    # Wider spreads
            'XRP': 2.0,    # Even wider
            'DOGE': 2.5    # Widest spreads
        }
        
        currency_factor = currency_factors.get(currency, 1.0)
        
        # Calculate base spread
        from config_manager import DEFAULT_SPREAD_WIDTH_PCT, SPREAD_WIDTH_MULTIPLIERS
        base_spread = spot_price * DEFAULT_SPREAD_WIDTH_PCT * volatility_factor * currency_factor
        
        # Generate multiple spread widths
        spread_widths = [base_spread * mult for mult in SPREAD_WIDTH_MULTIPLIERS]
        
        # Ensure within reasonable bounds
        from config_manager import MIN_SPREAD_WIDTH_PCT, MAX_SPREAD_WIDTH_PCT
        min_spread = spot_price * MIN_SPREAD_WIDTH_PCT
        max_spread = spot_price * MAX_SPREAD_WIDTH_PCT
        
        spread_widths = [
            max(min_spread, min(max_spread, spread))
            for spread in spread_widths
        ]
        
        # Round to reasonable precision based on asset price
        if spot_price > 10000:  # BTC
            spread_widths = [round(s, -2) for s in spread_widths]  # Round to 100s
        elif spot_price > 100:   # ETH
            spread_widths = [round(s, -1) for s in spread_widths]  # Round to 10s
        else:
            spread_widths = [round(s, 0) for s in spread_widths]   # Round to 1s
        
        return spread_widths
    
    def estimate_transaction_costs(
        self,
        currency: str,
        trade_size: float,
        market_data: Dict
    ) -> Optional[Dict[str, float]]:
        """
        Estimate transaction costs including spread and market impact
        Based on Almgren & Chriss (2001) - optimal execution
        
        Returns None if spread data is not available
        """
        costs = {
            'spread_cost': 0,
            'market_impact': 0,
            'total_cost_pct': 0
        }
        
        try:
            # 1. Half-spread cost
            spread_pct = market_data.get('spread_percentage')
            if spread_pct is None:
                self.logger.warning("Missing spread percentage data for transaction cost calculation")
                return None
            costs['spread_cost'] = trade_size * spread_pct / 2
            
            # 2. Market impact (square-root model from Almgren & Chriss)
            daily_volume = market_data.get('volume_24h', 0)
            if daily_volume > 0:
                # Impact = γ * √(trade_size / daily_volume)
                # γ is configurable per asset class
                gamma = MARKET_IMPACT_GAMMA.get(currency, 0.2)
                volume_fraction = trade_size / daily_volume
                impact_pct = gamma * np.sqrt(volume_fraction)
                costs['market_impact'] = trade_size * impact_pct
            
            # 3. Total cost as percentage
            total_cost = costs['spread_cost'] + costs['market_impact']
            costs['total_cost_pct'] = (total_cost / trade_size) if trade_size > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error estimating transaction costs: {e}")
            # Fallback to simple spread cost
            costs['spread_cost'] = trade_size * 0.002
            costs['total_cost_pct'] = 0.002
        
        return costs
    
    def _calculate_amihud_illiquidity(
        self,
        market_data: Dict,
        *,
        min_dollar_volume: float = 1.0,
        winsor_lower: float = 0.01,
        winsor_upper: float = 0.99,
        use_median: bool = True
    ) -> Optional[float]:
        """
        Amihud (2002) ILLIQ over the sample:
            ILLIQ = mean/median_t ( |log r_t| / DollarVolume_t )

        Requirements:
          - Either market_data['dollar_volume_history'] (per period)
            or price_history + volume_history (units) to compute dollar volume.
          - Skips periods with tiny or non-finite dollar volume.
          - Winsorizes the ratio to damp outliers (thin markets).

        Returns None if not enough data.
        """
        try:
            prices = np.asarray(market_data.get('price_history', []), dtype=float)
            if prices.size < 2:
                return None

            if 'dollar_volume_history' in market_data:
                dv = np.asarray(market_data['dollar_volume_history'], dtype=float)
            elif 'volume_history' in market_data:
                vol = np.asarray(market_data['volume_history'], dtype=float)
                n = min(len(prices), len(vol))
                prices, vol = prices[:n], vol[:n]
                dv = vol * prices
            else:
                return None

            n = min(len(prices), len(dv))
            prices, dv = prices[:n], dv[:n]

            # log-returns aligned to t -> (t-1, t)
            with np.errstate(divide='ignore', invalid='ignore'):
                rets = np.abs(np.diff(np.log(prices)))
            dv = dv[1:]  # align with returns

            mask = (dv >= min_dollar_volume) & np.isfinite(dv) & np.isfinite(rets)
            if not np.any(mask):
                return None

            illiq = rets[mask] / dv[mask]
            illiq = self._winsorize_array(illiq, winsor_lower, winsor_upper)

            value = float(np.nanmedian(illiq) if use_median else np.nanmean(illiq))
            return value if np.isfinite(value) else None

        except Exception as e:
            self.logger.error(f"Amihud ILLIQ failed: {e}")
            return None
    
    def calculate_effective_and_realized_spreads(
        self,
        market_data: Dict,
        *,
        post_trade_mid_lag_seconds: int = 300,   # 5 minutes by default
        max_outlier_spread_bps: float = 2000.0   # clip truly bad ticks
    ) -> Optional[Dict[str, float]]:
        """
        Computes:
          effective_t = 2 * |p_t - m_t| / m_t   (in bps)
          realized_t  = 2 * d_t * (p_t - m_{t+lag}) / m_t   (in bps)

        Where d_t is trade direction (+1 sell, -1 buy). If direction isn't provided,
        infer buy if p_t >= m_t.

        Guards:
          - drop locked/crossed quotes
          - drop trades far outside NBBO (|p-m| > 5 * spread)
          - clip outliers at `max_outlier_spread_bps`
          - return counts of used vs dropped
        """
        try:
            trades = market_data.get('recent_trades') or []
            quotes = market_data.get('quotes') or []
            if not trades or not quotes:
                return None

            # sanitize quotes, ensure ask > bid
            q = []
            for qt in quotes:
                try:
                    b = float(qt['bid']); a = float(qt['ask'])
                    if np.isfinite(b) and np.isfinite(a) and a > b and a > 0 and b > 0:
                        q.append({'bid': b, 'ask': a, 'timestamp': float(qt['timestamp'])})
                except Exception:
                    continue
            if not q:
                return None

            q = sorted(q, key=lambda x: x['timestamp'])
            q_times = np.array([x['timestamp'] for x in q], dtype=float)
            mids = np.array([(x['bid'] + x['ask']) / 2.0 for x in q], dtype=float)
            spreads = np.array([x['ask'] - x['bid'] for x in q], dtype=float)

            eff, rea = [], []
            used = dropped = 0

            for t in trades:
                try:
                    p = float(t['price'])
                    ts = float(t['timestamp'])
                except Exception:
                    dropped += 1
                    continue

                # quote at/just before trade
                idx = np.searchsorted(q_times, ts, side='right') - 1
                if idx < 0:
                    dropped += 1
                    continue

                m = mids[idx]; spr = spreads[idx]
                if not (np.isfinite(m) and np.isfinite(spr) and m > 0 and spr > 0):
                    dropped += 1
                    continue

                # outside-NBBO guard (relaxed)
                if abs(p - m) > 5.0 * spr:
                    dropped += 1
                    continue

                eff_bps = 2.0 * abs(p - m) / m * 1e4
                if eff_bps > max_outlier_spread_bps:
                    dropped += 1
                    continue

                # realized: use midpoint after lag
                lag_t = ts + post_trade_mid_lag_seconds
                j = np.searchsorted(q_times, lag_t, side='right') - 1
                if j < 0:
                    dropped += 1
                    continue
                m_lag = mids[j]

                # trade direction: +1 sell (takes bid), -1 buy (lifts ask)
                side = t.get('side')
                if side is None:
                    buyer_initiated = t.get('is_buyer_initiated')
                    if buyer_initiated is None:
                        buyer_initiated = (p >= m)
                    side = -1.0 if buyer_initiated else 1.0

                rea_bps = 2.0 * side * (p - m_lag) / m * 1e4
                if abs(rea_bps) > max_outlier_spread_bps:
                    dropped += 1
                    continue

                eff.append(eff_bps); rea.append(rea_bps); used += 1

            if used == 0:
                return None

            return {
                "effective_spread_bps_mean": float(np.mean(eff)),
                "effective_spread_bps_median": float(np.median(eff)),
                "realized_spread_bps_mean": float(np.mean(rea)),
                "realized_spread_bps_median": float(np.median(rea)),
                "num_trades_used": int(used),
                "num_trades_dropped": int(dropped),
            }

        except Exception as e:
            self.logger.error(f"Effective/realized spread failed: {e}")
            return None

    # Optional: keep your existing _calculate_effective_spread() signature working
    def _calculate_effective_spread(self, market_data: Dict) -> Optional[float]:
        res = self.calculate_effective_and_realized_spreads(market_data)
        return None if not res else res["effective_spread_bps_mean"] / 1e4  # back to fraction
    
    def calculate_breeden_litzenberger_density(
        self,
        options_chain: List[Dict],
        expiry_time: float,
        risk_free_rate: float = 0.05,
        smooth_before_diff: bool = False
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Calculate risk-neutral probability density using Breeden-Litzenberger (1978).
        State price density: q(K) = e^(rT) * ∂²C/∂K²
        
        Returns: List of (strike, density) tuples
        """
        try:
            if len(options_chain) < 5:  # Need sufficient strikes
                return None
            
            # Sort by strike
            sorted_options = sorted(options_chain, key=lambda x: x.get('strike', 0))
            
            strikes = np.array([opt['strike'] for opt in sorted_options])
            call_prices = np.array([opt.get('call_mid', 0) for opt in sorted_options])
            
            # Filter out invalid prices
            valid_mask = (strikes > 0) & (call_prices > 0)
            strikes = strikes[valid_mask]
            call_prices = call_prices[valid_mask]
            
            if len(strikes) < 5:
                return None
            
            # Optional: Smooth prices with cubic spline before differentiating
            if smooth_before_diff:
                try:
                    from scipy.interpolate import UnivariateSpline
                    # Fit cubic spline with smoothing
                    spline = UnivariateSpline(strikes, call_prices, s=0.001)
                    # Evaluate on finer grid
                    fine_strikes = np.linspace(strikes[0], strikes[-1], len(strikes) * 3)
                    call_prices = spline(fine_strikes)
                    strikes = fine_strikes
                except ImportError:
                    self.logger.warning("scipy not available for smoothing, using raw prices")
            
            # Calculate second derivative using central differences
            # For interior points: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
            densities = []
            
            for i in range(1, len(strikes) - 1):
                h1 = strikes[i] - strikes[i-1]
                h2 = strikes[i+1] - strikes[i]
                
                # Unequal spacing formula for second derivative
                d2C_dK2 = 2 * (
                    (call_prices[i+1] - call_prices[i]) / (h2 * (h1 + h2)) -
                    (call_prices[i] - call_prices[i-1]) / (h1 * (h1 + h2))
                )
                
                # Apply discount factor to get state price density
                T = (expiry_time - datetime.now().timestamp()) / SECONDS_PER_YEAR  # Years to expiry
                if T > 0:
                    density = np.exp(risk_free_rate * T) * d2C_dK2
                    if density > 0:  # Density must be positive
                        densities.append((strikes[i], density))
            
            return densities if densities else None
            
        except Exception as e:
            self.logger.error(f"Error calculating Breeden-Litzenberger density: {e}")
            return None
    
    def estimate_price_impact_lambda(
        self,
        returns: np.ndarray,
        signed_dollar_volume: np.ndarray,
        *,
        winsor_lower: float = 0.01,
        winsor_upper: float = 0.99
    ) -> Optional[float]:
        """
        Estimate λ from r_t = λ * S_t + ε_t   with S_t = signed dollar volume (or OFI).
        This is a simple single-equation proxy inspired by Hasbrouck-style setups,
        not the full VAR. Use it as a robust, fast impact coefficient.

        Returns λ (per USD notional). Caller should align and sign S_t with trade direction.
        """
        try:
            r = np.asarray(returns, dtype=float)
            s = np.asarray(signed_dollar_volume, dtype=float)
            n = min(r.size, s.size)
            if n < 10:
                return None
            r, s = r[:n], s[:n]
            m = np.isfinite(r) & np.isfinite(s)
            if not np.any(m):
                return None

            # winsorize both sides to tame fat tails
            r = self._winsorize_array(r[m], winsor_lower, winsor_upper)
            s = self._winsorize_array(s[m], winsor_lower, winsor_upper)

            var_s = np.var(s)
            if var_s <= 0:
                return None
            lam = float(np.cov(r, s, bias=True)[0, 1] / var_s)
            return lam if np.isfinite(lam) else None
        except Exception as e:
            self.logger.error(f"Price impact λ estimation failed: {e}")
            return None