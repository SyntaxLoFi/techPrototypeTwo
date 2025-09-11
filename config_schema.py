# config_schema.py (Pydantic v2)
from typing import Dict, Literal, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict, conint, confloat, field_validator, model_validator


def _pct(v: float) -> float:
    """
    Accept either 0..1 or 0..100 style; normalize to 0..1 if > 1.
    Note: negative values are not rescaled (e.g., -0.05 is valid; -5.0 is invalid).
    """
    try:
        v = float(v)
    except (TypeError, ValueError):
        return v
    return v / 100.0 if v > 1.0 else v


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    to_file: bool = True
    json_logs: bool = Field(default=True, alias="json")
    filename: str = "logs/app.log"
    max_bytes: conint(gt=0) = 20 * 1024 * 1024
    backup_count: conint(ge=0) = 5
    utc: bool = True

class DebugCaptureSettings(BaseModel):
    polymarket: bool = True
    options: bool = True
    perps: bool = True
    snapshot: bool = True

class DebugLogSettings(BaseModel):
    reason_codes: bool = True
    reason_codes_level: Literal["DEBUG","INFO"] = "DEBUG"
    reason_codes_max_lines: int = 400
    per_currency_snapshot: bool = True
    ev_summary_info: bool = True

class DebugSettings(BaseModel):
    enabled: bool = False
    dump_dir: str = "debug_runs"
    capture: DebugCaptureSettings = DebugCaptureSettings()
    log: DebugLogSettings = DebugLogSettings()


class LyraSettings(BaseModel):
    api_base: str = "https://api.lyra.finance"
    ws_uri: str = "wss://api.lyra.finance/ws"
    options_currencies: Dict[str, str] = Field(default_factory=lambda: {
        'BTC': 'BTC',
        'ETH': 'ETH'
    })
    perps_currencies: Dict[str, str] = Field(default_factory=lambda: {
        'BTC': 'BTC-PERP',
        'ETH': 'ETH-PERP',
        'SOL': 'SOL-PERP',
        'XRP': 'XRP-PERP',
        'DOGE': 'DOGE-PERP'
    })

    @property
    def enabled_currencies(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.options_currencies.keys()) | set(self.perps_currencies.keys())))


class ExecutionSettings(BaseModel):
    # Risk & sizing
    risk_free_rate: confloat(ge=-0.05, le=1.0) = 0.05
    min_risk_reward_ratio: confloat(gt=0) = 1.5
    min_kelly_fraction: confloat(ge=0, le=1) = 0.01
    kelly_fraction_cap: confloat(gt=0, le=1) = 0.25

    # Position sizing
    default_position_size: confloat(gt=0) = 100.0
    max_position_size: confloat(gt=0) = 10000.0
    min_position_size: confloat(gt=0) = 100.0
    # Instrument capture / legacy alias (UI-only helpers)
    capture_instruments: bool = False
    # DEPRECATED: kept as a backwards-compatibility alias for default_position_size
    position_base_unit: Optional[confloat(gt=0)] = None

    @model_validator(mode='after')
    def _apply_legacy_aliases(self):
        """Map legacy fields to the canonical ones.
        If position_base_unit is provided *and* default_position_size was not
        explicitly provided, treat it as the default position size.
        If both are provided, default_position_size takes precedence.
        """
        try:
            fields_set = getattr(self, 'model_fields_set', set())
        except Exception:
            fields_set = set()
        # Only adopt the legacy alias when the canonical field wasn't set explicitly
        if ('position_base_unit' in fields_set) and ('default_position_size' not in fields_set):
            try:
                if self.position_base_unit:
                    object.__setattr__(self, 'default_position_size', float(self.position_base_unit))
            except Exception:
                pass
        return self

    # Spread / hedging
    hedge_combination_mode: Literal["add", "offset"] = "add"
    min_spread_threshold: confloat(ge=0) = 0.02
    max_spread_allowed: confloat(ge=0) = 0.02
    default_spread_width_pct: confloat(ge=0) = 0.02
    min_spread_width: confloat(ge=0) = 0.5
    max_spread_count: conint(ge=1) = 10000
    settle_spread_at_pm_expiry: bool = True
    # How to value option hedges at PM resolution (T_pm)
    #   sticky_strike: BS with per-strike IV and reduced T (default)
    #   intrinsic_only: intrinsic at PM time
    options_unwind_model: Literal["sticky_strike", "intrinsic_only"] = "sticky_strike"
    # Optional post-event IV drop applied by the strategy layer when computing closeout marks
    post_event_vol_drop: confloat(ge=0, le=1) = 0.0

    # Expected value filtering
    min_expected_value: confloat(ge=0) = 0.10
    min_sharpe_ratio: confloat(ge=0) = 0.5
    max_acceptable_loss: confloat(gt=0) = 100.0
    transaction_cost_rate: confloat(ge=0) = 0.002

    # Normalize percent-like inputs (before validation constraints run)
    @field_validator(
        'risk_free_rate',
        'min_kelly_fraction',
        'kelly_fraction_cap',
        'min_spread_threshold',
        'max_spread_allowed',
        'default_spread_width_pct',
        'transaction_cost_rate',
        mode='before'
    )
    @classmethod
    def _normalize_pct(cls, v):
        return _pct(v)

    @model_validator(mode='after')
    def _bounds(self):
        if self.min_kelly_fraction > self.kelly_fraction_cap:
            raise ValueError("min_kelly_fraction must be <= kelly_fraction_cap")
        if self.min_spread_threshold > self.max_spread_allowed:
            raise ValueError("min_spread_threshold must be <= max_spread_allowed")
        if self.min_position_size > self.max_position_size:
            raise ValueError("min_position_size must be <= max_position_size")
        if self.default_position_size > self.max_position_size:
            raise ValueError("default_position_size must be <= max_position_size")
        return self


class MarketImpactSettings(BaseModel):
    # Per-asset linear impact coefficient gamma (>0). Example: {"BTC": 0.30, "ETH": 0.25, ...}
    gamma: Dict[str, confloat(gt=0)] = Field(default_factory=lambda: {
        'BTC': 0.15,
        'ETH': 0.18,
        'SOL': 0.25,
        'XRP': 0.28,
        'DOGE': 0.30
    })


class RankingSettings(BaseModel):
    # Prob blending
    blend_mode: Literal["fixed", "precision"] = "fixed"
    fixed_weight: Optional[confloat(gt=0, le=1)] = 0.5  # used when blend_mode="fixed"

    # Risk penalties
    vega_penalty_lambda: confloat(ge=0) = 0.5
    funding_penalty_kappa: confloat(ge=0) = 1.0
    liquidity_penalty_theta: confloat(ge=0) = 0.001

    # Quality tier thresholds
    true_arbitrage_dni_threshold: confloat(ge=-1000) = 0.0
    near_arbitrage_dni_threshold: confloat(ge=-1000) = -100.0
    near_arbitrage_prob_threshold: confloat(ge=0, le=1) = 0.8
    high_probability_threshold: confloat(ge=0, le=1) = 0.7
    moderate_probability_threshold: confloat(ge=0, le=1) = 0.5

    # Probability clipping
    probability_clip_floor: confloat(ge=0, le=1) = 0.001
    probability_clip_ceiling: confloat(ge=0, le=1) = 0.999

    # Other
    min_correlation_threshold: confloat(ge=0, le=1) = 0.3
    covariance_regularization: confloat(gt=0) = 1e-6

    # Normalize percent-like inputs
    @field_validator(
        'fixed_weight',
        'near_arbitrage_prob_threshold',
        'high_probability_threshold',
        'moderate_probability_threshold',
        'probability_clip_floor',
        'probability_clip_ceiling',
        'min_correlation_threshold',
        mode='before'
    )
    @classmethod
    def _normalize_pct(cls, v):
        return _pct(v)

    @model_validator(mode='after')
    def _check(self):
        if self.blend_mode == "fixed" and self.fixed_weight is None:
            raise ValueError("fixed_weight is required when blend_mode='fixed'")
        if self.near_arbitrage_dni_threshold >= self.true_arbitrage_dni_threshold:
            raise ValueError("near_arbitrage_dni_threshold must be < true_arbitrage_dni_threshold")
        if self.high_probability_threshold <= self.moderate_probability_threshold:
            raise ValueError("high_probability_threshold must be > moderate_probability_threshold")
        if self.probability_clip_floor >= self.probability_clip_ceiling:
            raise ValueError("probability_clip_floor must be < probability_clip_ceiling")
        return self


class TimeSettings(BaseModel):
    days_per_year: confloat(gt=0) = 365.25
    min_hours_to_expiry: confloat(ge=0) = 0.5
    max_days_to_expiry: confloat(gt=0) = 180

    @property
    def seconds_per_year(self) -> float:
        return float(self.days_per_year) * 24 * 60 * 60


class DataSettings(BaseModel):
    save_detailed_data: bool = True
    save_unfiltered_opportunities: bool = True
    scan_interval_seconds: conint(gt=0) = 300
    data_retention_minutes: conint(gt=0) = 60
    orderbook_depth: conint(gt=0) = 10


class PolymarketConfig(BaseModel):
    include_dailies: bool = True
    dailies_window_hours: float = 24.0
    filter_closed: bool = True
    filter_active: bool = True


class QuadraticHedgingSettings(BaseModel):
    # Objective parameters
    regularization: confloat(gt=0) = 1e-8
    cost_budget: confloat(ge=0) = 0.0
    include_bonds: bool = True
    include_forwards: bool = True
    
    # Sparse helper
    use_sparse_helper: bool = True
    max_legs: conint(gt=0) = 6
    
    # Bond basis name
    bond_basis_name: str = "bond_T1"
    
    # Contract sizing
    contract_increment_by_asset: Dict[str, confloat(gt=0)] = Field(default_factory=lambda: {
        'BTC': 0.01,
        'ETH': 0.01
    })
    min_contracts_by_asset: Dict[str, confloat(gt=0)] = Field(default_factory=lambda: {
        'BTC': 0.01,
        'ETH': 0.01
    })
    min_notional: confloat(gt=0) = 25.0


class VarianceHedgingSettings(BaseModel):
    # Objective parameters
    regularization: confloat(gt=0) = 1e-8
    cost_budget: confloat(ge=0) = 0.0
    include_bonds: bool = True
    include_forwards: bool = True

    # Sparse helper
    use_sparse_helper: bool = True
    max_legs: conint(gt=0) = 6

    # Bond basis name
    bond_basis_name: str = 'bond_T1'

    # Contract sizing (same defaults as quadratic)
    contract_increment_by_asset: Dict[str, confloat(gt=0)] = Field(default_factory=lambda: {
        'BTC': 0.01,
        'ETH': 0.01
    })
    min_contracts_by_asset: Dict[str, confloat(gt=0)] = Field(default_factory=lambda: {
        'BTC': 0.01,
        'ETH': 0.01
    })
    min_notional: confloat(gt=0) = 25.0

    # Variance-swap-specific toggles
    pm_anchored_static_replication: bool = True
    cost_recovery_tol: confloat(ge=0, le=0.2) = 0.02
    digital_width_bps: confloat(ge=0) = 50.0
    clean_quotes: bool = True
    no_arb_enforce: bool = True
    
    # Digital-bounds gate (PM binary vs option-implied digital)
    use_pm_digital_bounds: bool = True
    enforce_pm_digital_bounds: bool = False  # default to SOFT (do not prune)
    pm_bounds_inclusive: bool = False        # exclusive by default
    pm_bounds_slack_abs: float = 0.02        # absolute slack in price (e.g., 2 cents)
    pm_bounds_slack_rel: float = 0.01        # relative slack (1% of bound mid)
    pm_bounds_use_tradable: bool = True      # use bid/ask call-spread bounds when possible
    pm_bounds_require_two_sided: bool = False
    pm_bounds_max_rel_width: float = 0.02    # max Δ/K when choosing neighbor strikes
    pm_gate_use_pm_strike: bool = True       # gate at PM strike, not K0

    # ---------- Expiry selection policy ----------
    # Which expiries to consider relative to PM resolution and how strict to be.
    # 'allow_far_with_unwind' requires the execution unwind model to not be 'intrinsic_only'.
    expiry_policy: Literal["nearest_on_or_after", "allow_far_with_unwind"] = "allow_far_with_unwind"
    max_expiry_gap_days: conint(ge=0, le=60) = 60
    max_expiries_considered: conint(ge=1, le=10) = 10
    require_live_quotes_for_trades: bool = True
    strike_proximity_window: confloat(ge=0.05, le=0.50) = 0.25
    min_quotes_per_expiry: conint(ge=2, le=20) = 2

    # ---------- Variance strip building: liquidity-aware options ----------
    variance_forward_must_be_in_range: bool = False     # if True, reject expiry when F not in [Kmin,Kmax]
    variance_forward_clip_to_range: bool = True         # if forward is out of range, clip to nearest strike and continue

    # Zero-bid truncation behavior at each wing:
    #   "vix"        -> truncate after two consecutive zero-bid strikes (Cboe practice)
    #   "one_sided"  -> truncate only after two fully dead strikes (bid==ask==0 or both missing)
    variance_zero_bid_truncation_mode: str = "vix"
    variance_zero_bid_streak: int = 2                   # how many consecutive zeros trigger truncation

    # Wing coverage:
    variance_require_both_wings: bool = False           # if False, allow one-sided expiries with penalties
    variance_min_wing_quotes: int = 1                   # minimum OTM quotes required per included wing

    # One-sided synthetic mid construction (when only bid or only ask is present)
    variance_one_sided_rel_spread: float = 0.05         # 5% of price; used half each side to form a mid
    variance_one_sided_abs_spread: float = 0.00         # absolute add-on (e.g., $0.01 ticks); used half

    # Liquidity penalties (bps added to execution cost when building EV)
    variance_penalty_bps_per_one_sided: float = 5.0     # per strike priced with one-sided synthetic mid
    variance_penalty_bps_missing_wing: float = 25.0     # once-off penalty if only one wing present
    variance_penalty_bps_per_day_gap: float = 2.0       # per calendar day between PM resolution and option expiry


class HedgingSettings(BaseModel):
    quadratic: QuadraticHedgingSettings = QuadraticHedgingSettings()
    variance: VarianceHedgingSettings = VarianceHedgingSettings()


class AppConfig(BaseModel):
    env: Literal["dev", "staging", "prod"] = "dev"
    time: TimeSettings = TimeSettings()
    logging: LoggingSettings = LoggingSettings()
    debug: DebugSettings = DebugSettings()
    lyra: LyraSettings = LyraSettings()
    execution: ExecutionSettings = ExecutionSettings()
    market_impact: MarketImpactSettings = MarketImpactSettings()
    ranking: RankingSettings = RankingSettings()
    data: DataSettings = DataSettings()
    polymarket: PolymarketConfig = PolymarketConfig()
    hedging: HedgingSettings = HedgingSettings()

    # Pydantic v2 config
    model_config = ConfigDict(
        extra='forbid',    # reject unknown keys
        frozen=True        # freeze after construction
        # NOTE: we intentionally do NOT set populate_by_name=True
        # to preserve v1 behavior of requiring the alias when one is defined.
    )

    @model_validator(mode='after')
    def _cross_validate(self):
        # If we allow far expiries, we must support unwinding with a model richer than intrinsic-only.
        if (
            getattr(self.hedging.variance, "expiry_policy", "nearest_on_or_after") == "allow_far_with_unwind"
            and getattr(self.execution, "options_unwind_model", "sticky_strike") == "intrinsic_only"
        ):
            raise ValueError(
                "hedging.variance.expiry_policy='allow_far_with_unwind' requires "
                "execution.options_unwind_model != 'intrinsic_only'"
            )
        return self