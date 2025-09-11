"""Unified Configuration Manager

Provides a single source of truth for all configuration values,
consolidating config_loader.py (Pydantic) and config.py (constants).
"""

from __future__ import annotations

from dataclasses import dataclass, KW_ONLY
from typing import Dict, Tuple, Any, Optional
import os
import math
try:
    import yaml
except Exception:
    yaml = None
from config_loader import get_config, AppConfig
from config_schema import (
    LoggingSettings, LyraSettings, ExecutionSettings, 
    MarketImpactSettings, RankingSettings, TimeSettings, DataSettings,
    PolymarketConfig
)

# --------- Helpers for safe nested access and clamping ----------
def _get(d, path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _clamp(val, lo, hi):
    try:
        return max(lo, min(hi, val))
    except Exception:
        return lo

# --------- Load YAML (best-effort) ----------
_CFG = {}
_cfg_paths = []
try:
    _base_dir = os.path.dirname(os.path.abspath(__file__))
except Exception:
    _base_dir = os.getcwd()
# Candidate search order:
# 1) Explicit env
_cfg_paths.append(os.environ.get("APP_CONFIG_FILE"))
# 2) Repo-relative (alongside this file)
_cfg_paths.append(os.path.join(_base_dir, "config", "defaults.yaml"))
# 3) Parent repo dir (if config_manager.py lives under package root)
_cfg_paths.append(os.path.join(os.path.dirname(_base_dir), "config", "defaults.yaml"))
# 4) CWD-relative (for notebooks / scripts)
_cfg_paths.append(os.path.join(os.getcwd(), "config", "defaults.yaml"))
# Deduplicate while preserving order
_seen = set()
_cfg_paths = [p for p in _cfg_paths if p and not (p in _seen or _seen.add(p))]
for pth in _cfg_paths:
    if not pth:
        continue
    try:
        with open(pth, "r") as fh:
            _CFG = yaml.safe_load(fh) or {}
            break
    except Exception:
        continue

# --------- Execution: options unwind model ----------
_unwind_raw = str(_get(_CFG, ["execution", "options_unwind_model"], os.environ.get("OPTIONS_UNWIND_MODEL", "intrinsic_only")) or "intrinsic_only").lower()
_unwind_allowed = {"intrinsic_only", "sticky_strike", "sticky_delta", "sticky_moneyness"}
if _unwind_raw not in _unwind_allowed:
    _unwind_raw = "intrinsic_only"
OPTIONS_UNWIND_MODEL = _unwind_raw

# --------- Hedging / variance knobs (with clamps & defaults) ----------
_expiry_policy_raw = str(_get(_CFG, ["hedging", "variance", "expiry_policy"], os.environ.get("VARIANCE_EXPIRY_POLICY", "nearest_on_or_after")) or "nearest_on_or_after").lower()
VARIANCE_EXPIRY_POLICY = _expiry_policy_raw if _expiry_policy_raw in {"nearest_on_or_after", "allow_far_with_unwind"} else "nearest_on_or_after"

_gap_days = int(_get(_CFG, ["hedging", "variance", "max_expiry_gap_days"], os.environ.get("VARIANCE_MAX_EXPIRY_GAP_DAYS", 14)) or 14)
VARIANCE_MAX_EXPIRY_GAP_DAYS = _clamp(_gap_days, 0, 60)

_max_expiries = int(_get(_CFG, ["hedging", "variance", "max_expiries_considered"], os.environ.get("VARIANCE_MAX_EXPIRIES_CONSIDERED", 1)) or 1)
VARIANCE_MAX_EXPIRIES_CONSIDERED = _clamp(_max_expiries, 1, 10)

_rq = str(_get(_CFG, ["hedging", "variance", "require_live_quotes_for_trades"], os.environ.get("VARIANCE_REQUIRE_LIVE_QUOTES_FOR_TRADES", "true")) or "true").lower()
VARIANCE_REQUIRE_LIVE_QUOTES_FOR_TRADES = (_rq not in {"0","false","no"})

try:
    _spw = float(_get(_CFG, ["hedging", "variance", "strike_proximity_window"], os.environ.get("VARIANCE_STRIKE_PROXIMITY_WINDOW", 0.20)) or 0.20)
except Exception:
    _spw = 0.20
VARIANCE_STRIKE_PROXIMITY_WINDOW = float(_clamp(_spw, 0.05, 0.50))

# Minimum count of instruments with *two-sided* quotes required for an expiry to be "valid"
_minq = int(_get(_CFG, ["hedging", "variance", "min_quotes_per_expiry"], os.environ.get("VARIANCE_MIN_QUOTES_PER_EXPIRY", 4)) or 4)
VARIANCE_MIN_QUOTES_PER_EXPIRY = _clamp(_minq, 2, 20)

# (Other existing exports below…)

@dataclass(frozen=True)
class UnifiedConfig:
    """Unified configuration view that combines all settings"""
    
    # Core settings from AppConfig
    env: str
    time: TimeSettings
    logging: LoggingSettings
    lyra: LyraSettings
    execution: ExecutionSettings
    market_impact: MarketImpactSettings
    ranking: RankingSettings
    data: DataSettings
    polymarket: PolymarketConfig
    
    # Computed/derived values
    seconds_per_year: float
    binance_symbols: Dict[str, str]
    arbitrage_enabled_currencies: Tuple[str, ...]
    
    # API endpoints
    polymarket_api_base: str
    polymarket_clob_base: str
    binance_ws_url: str
    binance_ws_fallback: str
    binance_ws_us: str
    
    # Trading parameters
    default_volatility: float
    max_loss_tolerance: float
    
    # Polymarket execution
    pm_slippage_bps: float
    pm_fee_bps: float
    pm_profit_fee_bps: float
    hedge_alloc_fraction: float
    price_move_steps: Tuple[float, ...]
    
    # Other execution
    slippage_bps: float
    exit_spread_penalty: float
    execution_delay_ms: int
    exit_timing_pct: float
    min_exit_hours: float
    # Internal override fields; allow env to override properties
    _option_slippage_bps: Optional[float] = None
    _option_fee_bps: Optional[float] = None
    # All fields below are keyword-only (safe to mix defaults/non-defaults)
    _: KW_ONLY
    min_width_to_spread_ratio: float
    max_strike_asymmetry: float
    
    # WebSocket settings
    ws_reconnect_attempts: int
    ws_reconnect_delay: int
    ws_timeout: int
    
    # Market data
    funding_history_days: int
    variance_lookback_days: int
    default_implied_volatility: float
    atm_strike_window_low: float
    atm_strike_window_high: float
    
    # Liquidity configuration
    liquidity_safety_factor: float
    min_pm_liquidity: float
    min_perp_liquidity: float
    min_options_liquidity: float
    
    # Order size thresholds
    small_order_threshold_pct: float
    medium_order_threshold_pct: float
    large_order_threshold_pct: float
    
    # Spread width configuration
    default_spread_width_pct: float
    min_spread_width_pct: float
    max_spread_width_pct: float
    spread_width_multipliers: list[float]
    
    # Default market parameters
    default_spread_assumption: float
    default_pm_probability: float
    
    # Correlation and statistical
    default_crypto_correlation: float
    default_crypto_volatility: float
    
    # Arbitrage detection
    true_arbitrage_threshold: float
    min_profit_size: float
    
    # Scoring weights
    probability_score_weight: float
    risk_reward_weight: float
    profit_magnitude_weight: float
    directional_penalty_weight: float
    
    # Variance swap parameters
    min_strikes_required: int
    otm_cutoff_percentage: float
    min_funding_rate: float
    
    # Breeden-Litzenberger parameters
    bl_enforce_convex: bool
    bl_shape_projection_tol: float
    bl_strike_range_multiplier: float
    min_iv_floor: float
    min_dk_for_diff: float
    bl_max_bracket_bps: float
    bl_max_digital_rmse: float
    bl_diff_step_frac: float
    
    # IV crush parameters
    iv_crush_atm: float
    iv_crush_near: float
    iv_crush_far: float
    iv_crush_time_factor: float
    
    # Binary replication settings
    settle_spread_at_pm_expiry: bool
    min_spread_width: float
    max_spread_count: int
    
    # Conditional exit pricing
    use_conditional_exit_pricing: bool
    
    # Spot source configuration
    spot_source: str
    spot_stale_seconds: float
    
    # Scoring bonuses
    true_arbitrage_bonus: float
    near_arbitrage_bonus: float
    high_probability_bonus: float
    
    # Legacy/deprecated
    min_profit_threshold: float  # DEPRECATED
    
    @classmethod
    def from_app_config(cls, app_config: Optional[AppConfig] = None) -> UnifiedConfig:
        """Create unified config from AppConfig"""
        if app_config is None:
            app_config = get_config()
        
        # Import environment-based defaults from config.py
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        return cls(
            # Core settings
            env=app_config.env,
            time=app_config.time,
            logging=app_config.logging,
            lyra=app_config.lyra,
            execution=app_config.execution,
            market_impact=app_config.market_impact,
            ranking=app_config.ranking,
            data=app_config.data,
            polymarket=app_config.polymarket,
            
            # Computed values
            seconds_per_year=app_config.time.seconds_per_year,
            binance_symbols={
                "BTC": "BTCUSDT",
                "ETH": "ETHUSDT",
                "SOL": "SOLUSDT",
                "XRP": "XRPUSDT",
                "DOGE": "DOGEUSDT"
            },
            arbitrage_enabled_currencies=app_config.lyra.enabled_currencies,
            
            # API endpoints
            polymarket_api_base=os.getenv('POLYMARKET_API_BASE', 'https://gamma-api.polymarket.com'),
            polymarket_clob_base=os.getenv('POLYMARKET_CLOB_BASE', 'https://clob.polymarket.com'),
            binance_ws_url=os.getenv('BINANCE_WS_URL', 'wss://stream.binance.com:9443'),
            binance_ws_fallback=os.getenv('BINANCE_WS_FALLBACK', 'wss://data-stream.binance.vision:443'),
            binance_ws_us=os.getenv('BINANCE_WS_US', 'wss://stream.binance.us:9443'),
            
            # Trading parameters (not in schema yet)
            default_volatility=float(os.getenv('DEFAULT_VOLATILITY', '0.6')),
            max_loss_tolerance=float(os.getenv('MAX_LOSS_TOLERANCE', '0.20')),
            
            # Polymarket execution
            pm_slippage_bps=float(os.getenv('PM_SLIPPAGE_BPS', '5')),
            pm_fee_bps=float(os.getenv('PM_FEE_BPS', '0')),
            pm_profit_fee_bps=float(os.getenv('PM_PROFIT_FEE_BPS', os.getenv('PM_FEE_BPS', '0'))),
            hedge_alloc_fraction=float(os.getenv('HEDGE_ALLOC_FRACTION', '0.5')),
            price_move_steps=tuple(
                float(x) for x in os.getenv('PRICE_MOVE_STEPS', '0.025,0.05,0.10').split(',')
                if str(x).strip()
            ),
            
            # Other execution
            slippage_bps=float(os.getenv('SLIPPAGE_BPS', '10')),
            exit_spread_penalty=float(os.getenv('EXIT_SPREAD_PENALTY', '0.02')),
            execution_delay_ms=int(os.getenv('EXECUTION_DELAY_MS', '2000')),
            exit_timing_pct=float(os.getenv('EXIT_TIMING_PCT', '0.3')),
            min_exit_hours=float(os.getenv('MIN_EXIT_HOURS', '0.5')),
            
            # Option/Spread conditioning
            _option_slippage_bps=(lambda v=os.getenv('OPTION_SLIPPAGE_BPS'): float(v) if v is not None else None)(),
            _option_fee_bps=(lambda v=os.getenv('OPTION_FEE_BPS'): float(v) if v is not None else None)(),
            min_width_to_spread_ratio=float(os.getenv('MIN_WIDTH_TO_SPREAD_RATIO', '8.0')),
            max_strike_asymmetry=float(os.getenv('MAX_STRIKE_ASYMMETRY', '0.60')),
            
            # WebSocket settings
            ws_reconnect_attempts=int(os.getenv('WS_RECONNECT_ATTEMPTS', '5')),
            ws_reconnect_delay=int(os.getenv('WS_RECONNECT_DELAY', '2')),
            ws_timeout=int(os.getenv('WS_TIMEOUT', '30')),
            
            # Market data
            funding_history_days=int(os.getenv('FUNDING_HISTORY_DAYS', '30')),
            variance_lookback_days=int(os.getenv('VARIANCE_LOOKBACK_DAYS', '30')),
            default_implied_volatility=float(os.getenv('DEFAULT_IMPLIED_VOLATILITY', '0.6')),
            atm_strike_window_low=float(os.getenv('ATM_STRIKE_WINDOW_LOW', '0.95')),
            atm_strike_window_high=float(os.getenv('ATM_STRIKE_WINDOW_HIGH', '1.05')),
            
            # Liquidity configuration
            liquidity_safety_factor=float(os.getenv('LIQUIDITY_SAFETY_FACTOR', '0.7')),
            min_pm_liquidity=float(os.getenv('MIN_PM_LIQUIDITY', '1000')),
            min_perp_liquidity=float(os.getenv('MIN_PERP_LIQUIDITY', '5000')),
            min_options_liquidity=float(os.getenv('MIN_OPTIONS_LIQUIDITY', os.getenv('MIN_LIQUIDITY_REQUIRED', '25000'))),
            
            # Order size thresholds
            small_order_threshold_pct=float(os.getenv('SMALL_ORDER_THRESHOLD_PCT', '0.001')),
            medium_order_threshold_pct=float(os.getenv('MEDIUM_ORDER_THRESHOLD_PCT', '0.01')),
            large_order_threshold_pct=float(os.getenv('LARGE_ORDER_THRESHOLD_PCT', '0.05')),
            
            # Spread width configuration  
            default_spread_width_pct=app_config.execution.default_spread_width_pct,
            min_spread_width_pct=float(os.getenv('MIN_SPREAD_WIDTH_PCT', '0.002')),
            max_spread_width_pct=float(os.getenv('MAX_SPREAD_WIDTH_PCT', '0.10')),
            spread_width_multipliers=[0.5, 1.0, 2.0, 5.0],
            
            # Default market parameters
            default_spread_assumption=float(os.getenv('DEFAULT_SPREAD_ASSUMPTION', '0.002')),
            default_pm_probability=float(os.getenv('DEFAULT_PM_PROBABILITY', '0.5')),
            
            # Correlation and statistical
            default_crypto_correlation=float(os.getenv('DEFAULT_CRYPTO_CORRELATION', '0.75')),
            default_crypto_volatility=float(os.getenv('DEFAULT_CRYPTO_VOLATILITY', '0.6')),
            
            # Arbitrage detection
            true_arbitrage_threshold=float(os.getenv('TRUE_ARBITRAGE_THRESHOLD', '50')),
            min_profit_size=float(os.getenv('MIN_PROFIT_SIZE', '0.50')),
            
            # Scoring weights
            probability_score_weight=float(os.getenv('PROBABILITY_SCORE_WEIGHT', '1000')),
            risk_reward_weight=float(os.getenv('RISK_REWARD_WEIGHT', '100')),
            profit_magnitude_weight=float(os.getenv('PROFIT_MAGNITUDE_WEIGHT', '0.1')),
            directional_penalty_weight=float(os.getenv('DIRECTIONAL_PENALTY_WEIGHT', '500')),
            
            # Variance swap parameters
            min_strikes_required=int(os.getenv('MIN_STRIKES_REQUIRED', '10')),
            otm_cutoff_percentage=float(os.getenv('OTM_CUTOFF_PERCENTAGE', '0.3')),
            min_funding_rate=float(os.getenv('MIN_FUNDING_RATE', '0.000005')),
            
            # Breeden-Litzenberger parameters
            bl_enforce_convex=os.getenv('BL_ENFORCE_CONVEX', 'true').lower() in ('1', 'true', 'yes'),
            bl_shape_projection_tol=float(os.getenv('BL_SHAPE_PROJECTION_TOL', '1e-10')),
            bl_strike_range_multiplier=float(os.getenv('BL_STRIKE_RANGE_MULTIPLIER', '0.2')),
            min_iv_floor=float(os.getenv('MIN_IV_FLOOR', '0.05')),
            min_dk_for_diff=float(os.getenv('MIN_DK_FOR_DIFF', '0.5')),
            bl_max_bracket_bps=float(os.getenv('BL_MAX_BRACKET_BPS', '100')),
            bl_max_digital_rmse=float(os.getenv('BL_MAX_DIGITAL_RMSE', '0.02')),
            bl_diff_step_frac=float(os.getenv('BL_DIFF_STEP_FRAC', '0.5')),
            
            # IV crush parameters
            iv_crush_atm=float(os.getenv('IV_CRUSH_ATM', '0.20')),
            iv_crush_near=float(os.getenv('IV_CRUSH_NEAR', '0.15')),
            iv_crush_far=float(os.getenv('IV_CRUSH_FAR', '0.10')),
            iv_crush_time_factor=float(os.getenv('IV_CRUSH_TIME_FACTOR', '12.0')),
            
            # Binary replication settings
            settle_spread_at_pm_expiry=app_config.execution.settle_spread_at_pm_expiry,
            min_spread_width=app_config.execution.min_spread_width,
            max_spread_count=app_config.execution.max_spread_count,
            
            # Conditional exit pricing
            use_conditional_exit_pricing=os.getenv('USE_CONDITIONAL_EXIT_PRICING', 'true').lower() in ('1','true','yes'),
            
            # Spot source configuration
            spot_source=os.getenv('SPOT_SOURCE', 'binance').lower(),
            spot_stale_seconds=float(os.getenv('SPOT_STALE_SECONDS', '10')),
            
            # Scoring bonuses
            true_arbitrage_bonus=float(os.getenv('TRUE_ARBITRAGE_BONUS', '10000')),
            near_arbitrage_bonus=float(os.getenv('NEAR_ARBITRAGE_BONUS', '5000')),
            high_probability_bonus=float(os.getenv('HIGH_PROBABILITY_BONUS', '1000')),
            
            # Legacy/deprecated
            min_profit_threshold=float(os.getenv('MIN_PROFIT_THRESHOLD', '5')),
        )
    
    # Convenience properties for common access patterns
    @property
    def days_per_year(self) -> float:
        return self.time.days_per_year
    
    @property
    def risk_free_rate(self) -> float:
        return self.execution.risk_free_rate
    
    @property
    def min_position_size(self) -> float:
        return self.execution.min_position_size
    
    @property
    def default_position_size(self) -> float:
        return self.execution.default_position_size
    
    @property
    def max_position_size(self) -> float:
        return self.execution.max_position_size
    
    @property
    def min_hours_to_expiry(self) -> float:
        return self.time.min_hours_to_expiry
    
    @property
    def max_days_to_expiry(self) -> float:
        return self.time.max_days_to_expiry
    
    @property
    def save_detailed_data(self) -> bool:
        return self.data.save_detailed_data
    
    @property
    def save_unfiltered_opportunities(self) -> bool:
        return self.data.save_unfiltered_opportunities
    
    @property
    def lyra_api_base(self) -> str:
        return self.lyra.api_base
    
    @property
    def lyra_ws_uri(self) -> str:
        return self.lyra.ws_uri
    
    @property
    def lyra_options_currencies(self) -> Dict[str, str]:
        return self.lyra.options_currencies
    
    @property
    def lyra_perps_currencies(self) -> Dict[str, str]:
        return self.lyra.perps_currencies
    
    @property
    def supported_currencies(self) -> list[str]:
        # Return in the original order from config.py, not sorted
        return ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']
    
    @property
    def orderbook_depth(self) -> int:
        return self.data.orderbook_depth
    
    @property
    def scan_interval_seconds(self) -> int:
        return self.data.scan_interval_seconds
    
    @property
    def data_retention_minutes(self) -> int:
        return self.data.data_retention_minutes
    
    @property
    def transaction_cost_rate(self) -> float:
        return self.execution.transaction_cost_rate
    
    @property
    def min_expected_value(self) -> float:
        return self.execution.min_expected_value
    
    @property
    def min_sharpe_ratio(self) -> float:
        return self.execution.min_sharpe_ratio
    
    @property
    def min_kelly_fraction(self) -> float:
        return self.execution.min_kelly_fraction
    
    @property
    def kelly_fraction_cap(self) -> float:
        return self.execution.kelly_fraction_cap
    
    @property
    def max_acceptable_loss(self) -> float:
        return self.execution.max_acceptable_loss
    
    @property
    def probability_clip_floor(self) -> float:
        return self.ranking.probability_clip_floor
    
    @property
    def probability_clip_ceiling(self) -> float:
        return self.ranking.probability_clip_ceiling
    
    @property
    def min_correlation_threshold(self) -> float:
        return self.ranking.min_correlation_threshold
    
    @property
    def covariance_regularization(self) -> float:
        return self.ranking.covariance_regularization
    
    @property
    def log_level(self) -> str:
        return self.logging.level
    
    @property
    def log_to_file(self) -> bool:
        return self.logging.to_file
    
    @property
    def hedge_combination_mode(self) -> str:
        return self.execution.hedge_combination_mode
    
    @property
    def options_unwind_model(self) -> str:
        # Baseline & defaults use sticky_strike; prefer that when not set.
        return getattr(self.execution, 'options_unwind_model', 'sticky_strike')
    
    @property
    def option_fee_bps(self) -> float:
        # Prefer explicit override; otherwise fall back to ExecutionSettings
        if getattr(self, '_option_fee_bps', None) is not None:
            return float(self._option_fee_bps)
        return float(getattr(self.execution, 'option_fee_bps', 30.0))
    
    @property
    def option_slippage_bps(self) -> float:
        # Prefer explicit override; otherwise fall back to ExecutionSettings
        if getattr(self, '_option_slippage_bps', None) is not None:
            return float(self._option_slippage_bps)
        return float(getattr(self.execution, 'option_slippage_bps', 20.0))
    
    @property
    def max_spread_allowed(self) -> float:
        return self.execution.max_spread_allowed
    
    @property
    def min_spread_threshold(self) -> float:
        return self.execution.min_spread_threshold
    
    @property
    def market_impact_gamma(self) -> Dict[str, float]:
        return self.market_impact.gamma


# Module-level singleton
_unified_config: Optional[UnifiedConfig] = None


def get_unified_config(**kwargs: Any) -> UnifiedConfig:
    """Get the unified configuration instance (cached)"""
    global _unified_config
    if _unified_config is None:
        _unified_config = UnifiedConfig.from_app_config()
    return _unified_config


# Backward compatibility exports
# These allow existing code to import from config_manager instead of config.py
config = get_unified_config()

DAYS_PER_YEAR = config.days_per_year
SECONDS_PER_YEAR = config.seconds_per_year
RISK_FREE_RATE = config.risk_free_rate
DEFAULT_POSITION_SIZE = config.default_position_size
MIN_POSITION_SIZE = config.min_position_size
MAX_POSITION_SIZE = config.max_position_size
DEFAULT_IMPLIED_VOLATILITY = config.default_implied_volatility
LYRA_API_BASE = config.lyra_api_base
LYRA_WS_URI = config.lyra_ws_uri
LYRA_OPTIONS_CURRENCIES = config.lyra_options_currencies
LYRA_PERPS_CURRENCIES = config.lyra_perps_currencies
ARBITRAGE_ENABLED_CURRENCIES = config.arbitrage_enabled_currencies
BINANCE_SYMBOLS = config.binance_symbols
SAVE_DETAILED_DATA = config.save_detailed_data
SAVE_UNFILTERED_OPPORTUNITIES = config.save_unfiltered_opportunities
TRANSACTION_COST_RATE = config.transaction_cost_rate
MIN_EXPECTED_VALUE = config.min_expected_value
MIN_SHARPE_RATIO = config.min_sharpe_ratio
MIN_KELLY_FRACTION = config.min_kelly_fraction
KELLY_FRACTION_CAP = config.kelly_fraction_cap
MAX_ACCEPTABLE_LOSS = config.max_acceptable_loss
PROBABILITY_CLIP_FLOOR = config.probability_clip_floor
PROBABILITY_CLIP_CEILING = config.probability_clip_ceiling
MIN_CORRELATION_THRESHOLD = config.min_correlation_threshold
HEDGE_COMBINATION_MODE = config.hedge_combination_mode
POST_EVENT_VOL_DROP    = getattr(config, "post_event_vol_drop", 0.0)
OPTION_FEE_BPS         = config.option_fee_bps
OPTION_SLIPPAGE_BPS    = config.option_slippage_bps
MAX_SPREAD_ALLOWED = config.max_spread_allowed
MIN_SPREAD_THRESHOLD = config.min_spread_threshold
SETTLE_SPREAD_AT_PM_EXPIRY = config.settle_spread_at_pm_expiry
MIN_SPREAD_WIDTH = config.min_spread_width
MAX_SPREAD_COUNT = config.max_spread_count
USE_CONDITIONAL_EXIT_PRICING = config.use_conditional_exit_pricing
BL_ENFORCE_CONVEX = config.bl_enforce_convex
BL_SHAPE_PROJECTION_TOL = config.bl_shape_projection_tol
DEFAULT_SPREAD_WIDTH_PCT = config.default_spread_width_pct
MIN_HOURS_TO_EXPIRY = config.min_hours_to_expiry
MAX_DAYS_TO_EXPIRY = config.max_days_to_expiry
ORDERBOOK_DEPTH = config.orderbook_depth
SCAN_INTERVAL_SECONDS = config.scan_interval_seconds
DATA_RETENTION_MINUTES = config.data_retention_minutes
LOG_LEVEL = config.log_level
LOG_TO_FILE = config.log_to_file

# Additional exports needed by various modules
LIQUIDITY_SAFETY_FACTOR = config.liquidity_safety_factor
MIN_PM_LIQUIDITY = config.min_pm_liquidity
MIN_PERP_LIQUIDITY = config.min_perp_liquidity
MIN_OPTIONS_LIQUIDITY = config.min_options_liquidity
SMALL_ORDER_THRESHOLD_PCT = config.small_order_threshold_pct
MEDIUM_ORDER_THRESHOLD_PCT = config.medium_order_threshold_pct
LARGE_ORDER_THRESHOLD_PCT = config.large_order_threshold_pct
BINANCE_WS_URL = config.binance_ws_url
BINANCE_WS_FALLBACK = config.binance_ws_fallback
BINANCE_WS_US = config.binance_ws_us
VARIANCE_LOOKBACK_DAYS = config.variance_lookback_days
DEFAULT_VOLATILITY = config.default_volatility
ATM_STRIKE_WINDOW_LOW = config.atm_strike_window_low
ATM_STRIKE_WINDOW_HIGH = config.atm_strike_window_high
MIN_PROFIT_THRESHOLD = config.min_profit_threshold
TRUE_ARBITRAGE_BONUS = config.true_arbitrage_bonus
NEAR_ARBITRAGE_BONUS = config.near_arbitrage_bonus
HIGH_PROBABILITY_BONUS = config.high_probability_bonus
TRUE_ARBITRAGE_THRESHOLD = config.true_arbitrage_threshold
PROBABILITY_SCORE_WEIGHT = config.probability_score_weight
RISK_REWARD_WEIGHT = config.risk_reward_weight
PROFIT_MAGNITUDE_WEIGHT = config.profit_magnitude_weight
DIRECTIONAL_PENALTY_WEIGHT = config.directional_penalty_weight
POLYMARKET_API_BASE = config.polymarket_api_base
POLYMARKET_CLOB_BASE = config.polymarket_clob_base
FUNDING_HISTORY_DAYS = config.funding_history_days
EXIT_SPREAD_PENALTY = config.exit_spread_penalty
SLIPPAGE_BPS = config.slippage_bps
EXECUTION_DELAY_MS = config.execution_delay_ms
OPTION_SLIPPAGE_BPS = config.option_slippage_bps
OPTION_FEE_BPS = config.option_fee_bps

# --- Hedging rounding (venue increments) ---
try:
    _hedge_cfg = getattr(config.hedging, 'variance', None) or getattr(config.hedging, 'quadratic')
    CONTRACT_INCREMENT_BY_ASSET = dict(_hedge_cfg.contract_increment_by_asset)
    MIN_CONTRACTS_BY_ASSET      = dict(_hedge_cfg.min_contracts_by_asset)
except Exception:
    # Sensible defaults for Lyra/Deribit-style platforms
    CONTRACT_INCREMENT_BY_ASSET = {'BTC': 0.01, 'ETH': 0.01}
    MIN_CONTRACTS_BY_ASSET      = {'BTC': 0.01, 'ETH': 0.01}

# --- Cross-expiry settlement policy (already present in YAML) ---
SETTLE_SPREAD_AT_PM_EXPIRY = bool(getattr(config.execution, "settle_spread_at_pm_expiry", True))

# --- Strict arbitrage classification / policy flags ---
STRICT_ARBITRAGE_MODE = bool(getattr(config.execution, "strict_arbitrage_mode", True))
LABEL_TRUE_ARB_FROM_DETECTOR = bool(getattr(config.execution, "label_true_arb_from_detector", True))
EXACT_STRIKE_REQUIRED_FOR_TRUE_ARB = bool(getattr(config.execution, "exact_strike_required_for_true_arb", True))
CONSIDER_ALL_EXPIRIES = bool(getattr(config.execution, "consider_all_expiries", True))
MIN_WIDTH_TO_SPREAD_RATIO = config.min_width_to_spread_ratio
MAX_STRIKE_ASYMMETRY = config.max_strike_asymmetry
DEFAULT_SPREAD_ASSUMPTION = config.default_spread_assumption
WS_RECONNECT_ATTEMPTS = config.ws_reconnect_attempts
WS_RECONNECT_DELAY = config.ws_reconnect_delay
WS_TIMEOUT = config.ws_timeout
MIN_PROFIT_SIZE = config.min_profit_size
SPOT_SOURCE = config.spot_source
SPOT_STALE_SECONDS = config.spot_stale_seconds
DEFAULT_SPREAD_WIDTH_PCT = config.default_spread_width_pct
SPREAD_WIDTH_MULTIPLIERS = config.spread_width_multipliers
MARKET_IMPACT_GAMMA = config.market_impact_gamma
DEFAULT_CRYPTO_VOLATILITY = config.default_crypto_volatility
MIN_STRIKES_REQUIRED = config.min_strikes_required
OTM_CUTOFF_PERCENTAGE = config.otm_cutoff_percentage
COVARIANCE_REGULARIZATION = config.covariance_regularization
BL_STRIKE_RANGE_MULTIPLIER = config.bl_strike_range_multiplier
MIN_IV_FLOOR = config.min_iv_floor
MIN_DK_FOR_DIFF = config.min_dk_for_diff
BL_MAX_BRACKET_BPS = config.bl_max_bracket_bps
BL_MAX_DIGITAL_RMSE = config.bl_max_digital_rmse
BL_DIFF_STEP_FRAC = config.bl_diff_step_frac
DEFAULT_IMPLIED_VOLATILITY = config.default_implied_volatility
MAX_LOSS_TOLERANCE = config.max_loss_tolerance
RISK_FREE_RATE = config.risk_free_rate
PM_SLIPPAGE_BPS = config.pm_slippage_bps
PM_FEE_BPS = config.pm_fee_bps
PM_PROFIT_FEE_BPS = config.pm_profit_fee_bps
HEDGE_ALLOC_FRACTION = config.hedge_alloc_fraction
PRICE_MOVE_STEPS = config.price_move_steps
EXIT_TIMING_PCT = config.exit_timing_pct
MIN_EXIT_HOURS = config.min_exit_hours
MIN_FUNDING_RATE = config.min_funding_rate
IV_CRUSH_ATM = config.iv_crush_atm
IV_CRUSH_NEAR = config.iv_crush_near
IV_CRUSH_FAR = config.iv_crush_far
IV_CRUSH_TIME_FACTOR = config.iv_crush_time_factor
DEFAULT_PM_PROBABILITY = config.default_pm_probability
DEFAULT_CRYPTO_CORRELATION = config.default_crypto_correlation
MIN_SPREAD_WIDTH_PCT = config.min_spread_width_pct
MAX_SPREAD_WIDTH_PCT = config.max_spread_width_pct
SUPPORTED_CURRENCIES = config.supported_currencies
MIN_RISK_REWARD_RATIO = config.execution.min_risk_reward_ratio

# ---- Network & API hygiene (new) ----
HTTP_TIMEOUT_SEC = float(os.getenv('HTTP_TIMEOUT_SEC', '10'))
HTTP_RETRY_MAX = int(os.getenv('HTTP_RETRY_MAX', '3'))

# Polymarket batching for /prices
POLYMARKET_PRICES_CHUNK = int(os.getenv('POLYMARKET_PRICES_CHUNK', '150'))

# ---- Polymarket filters (NEW) ----
# API-level filters:
POLYMARKET_FILTER_ACTIVE = config.polymarket.filter_active if hasattr(config, 'polymarket') else os.getenv('POLYMARKET_FILTER_ACTIVE', 'true').lower() == 'true'
POLYMARKET_FILTER_CLOSED = config.polymarket.filter_closed if hasattr(config, 'polymarket') else os.getenv('POLYMARKET_FILTER_CLOSED', 'true').lower() == 'true'
# Dailies behavior:
POLYMARKET_INCLUDE_DAILIES = config.polymarket.include_dailies if hasattr(config, 'polymarket') else os.getenv('POLYMARKET_INCLUDE_DAILIES', 'true').lower() == 'true'
POLYMARKET_DAILIES_WINDOW_HOURS = config.polymarket.dailies_window_hours if hasattr(config, 'polymarket') else float(os.getenv('POLYMARKET_DAILIES_WINDOW_HOURS', '24'))

# Derive orderbook channel constraints (per docs)
ALLOWED_ORDERBOOK_GROUPS = {1, 10, 100}
ALLOWED_ORDERBOOK_DEPTHS = {1, 10, 20, 100}

# Binance REST endpoints (preferred + fallbacks)
BINANCE_REST_PRIMARY = os.getenv('BINANCE_REST_PRIMARY', 'https://data-api.binance.vision')
BINANCE_REST_SECONDARY = os.getenv('BINANCE_REST_SECONDARY', 'https://api.binance.com')
# Additional REST hosts for rotation
BINANCE_REST_HOSTS = [
    BINANCE_REST_PRIMARY,
    BINANCE_REST_SECONDARY,
    'https://api1.binance.com',
    'https://api2.binance.com', 
    'https://api3.binance.com',
    'https://api4.binance.com',
    'https://api-gcp.binance.com'
]

# Variance configuration is already loaded at the top of the file (lines 75-96)
# OPTIONS_UNWIND_MODEL is also already defined at the top (line 73)