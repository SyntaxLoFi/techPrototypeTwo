# ranking_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from config_loader import get_config


@dataclass(frozen=True)
class RankingConfigView:
    # Probability blending
    probability_blend_mode: str
    probability_blend_weight: float  # read from config.ranking.fixed_weight

    # Penalties
    vega_penalty_lambda: float
    funding_penalty_kappa: float
    liquidity_penalty_theta: float

    # Quality tiers
    true_arbitrage_dni_threshold: float
    near_arbitrage_dni_threshold: float
    near_arbitrage_prob_threshold: float
    high_probability_threshold: float
    moderate_probability_threshold: float

    # Clipping
    probability_clip_floor: float
    probability_clip_ceiling: float

    # Other
    min_correlation_threshold: float
    covariance_regularization: float

    # Cross-section from execution (used in ranking)
    transaction_cost_rate: float
    kelly_fraction_cap: float

    # Liquidity / market impact scaling used by ProbabilityRanker
    market_impact_coefficient: float

    # ---------------- Alias (UPPER_SNAKE_CASE) read-only properties ----------------
    @property
    def PROBABILITY_BLEND_MODE(self) -> str: return self.probability_blend_mode
    @property
    def PROBABILITY_BLEND_WEIGHT(self) -> float: return self.probability_blend_weight
    @property
    def VEGA_PENALTY_LAMBDA(self) -> float: return self.vega_penalty_lambda
    @property
    def FUNDING_PENALTY_KAPPA(self) -> float: return self.funding_penalty_kappa
    @property
    def LIQUIDITY_PENALTY_THETA(self) -> float: return self.liquidity_penalty_theta
    @property
    def TRUE_ARBITRAGE_DNI_THRESHOLD(self) -> float: return self.true_arbitrage_dni_threshold
    @property
    def NEAR_ARBITRAGE_DNI_THRESHOLD(self) -> float: return self.near_arbitrage_dni_threshold
    @property
    def NEAR_ARBITRAGE_PROB_THRESHOLD(self) -> float: return self.near_arbitrage_prob_threshold
    @property
    def HIGH_PROBABILITY_THRESHOLD(self) -> float: return self.high_probability_threshold
    @property
    def MODERATE_PROBABILITY_THRESHOLD(self) -> float: return self.moderate_probability_threshold
    @property
    def PROBABILITY_CLIP_FLOOR(self) -> float: return self.probability_clip_floor
    @property
    def PROBABILITY_CLIP_CEILING(self) -> float: return self.probability_clip_ceiling
    @property
    def MIN_CORRELATION_THRESHOLD(self) -> float: return self.min_correlation_threshold
    @property
    def COVARIANCE_REGULARIZATION(self) -> float: return self.covariance_regularization
    @property
    def TRANSACTION_COST_RATE(self) -> float: return self.transaction_cost_rate
    @property
    def KELLY_FRACTION_CAP(self) -> float: return self.kelly_fraction_cap
    @property
    def MARKET_IMPACT_COEFFICIENT(self) -> float: return self.market_impact_coefficient

    # ---------------- Lifecycle: normalize + validate ----------------
    def __post_init__(self) -> None:
        # normalize blend mode
        object.__setattr__(self, "probability_blend_mode", str(self.probability_blend_mode).strip().lower())
        self._validate()

    def _validate(self) -> None:
        # weights/thresholds in [0,1]
        if not (0.0 <= self.probability_blend_weight <= 1.0):
            raise ValueError(f"probability_blend_weight must be in [0,1], got {self.probability_blend_weight}")
        for name in ("near_arbitrage_prob_threshold", "high_probability_threshold", "moderate_probability_threshold"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {v}")

        # clip floor/ceiling ordered and inside [0,1]
        if not (0.0 <= self.probability_clip_floor < self.probability_clip_ceiling <= 1.0):
            raise ValueError(
                "probability_clip_floor must be >= 0 and < probability_clip_ceiling <= 1.0 "
                f"(got floor={self.probability_clip_floor}, ceiling={self.probability_clip_ceiling})"
            )

        # correlation in [-1,1]
        if not (-1.0 <= self.min_correlation_threshold <= 1.0):
            raise ValueError(f"min_correlation_threshold must be in [-1,1], got {self.min_correlation_threshold}")

        # non-negatives
        for name in ("covariance_regularization", "vega_penalty_lambda", "funding_penalty_kappa",
                     "liquidity_penalty_theta", "transaction_cost_rate", "market_impact_coefficient"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")

        # kelly cap in (0,1]
        if not (0.0 < self.kelly_fraction_cap <= 1.0):
            raise ValueError(f"kelly_fraction_cap must be in (0,1], got {self.kelly_fraction_cap}")

        if not self.probability_blend_mode:
            raise ValueError("probability_blend_mode cannot be empty")

    # ---------------- Construction ----------------
    @classmethod
    def from_app(cls) -> "RankingConfigView":
        cfg = get_config()
        r = cfg.get("ranking", {}) if isinstance(cfg, dict) else getattr(cfg, "ranking", {})
        e = cfg.get("execution", {}) if isinstance(cfg, dict) else getattr(cfg, "execution", {})
        def rget(k, default=None): return (r or {}).get(k, default)

        return cls(
            probability_blend_mode=str(rget("blend_mode", "fixed")),
            probability_blend_weight=float(rget("fixed_weight", 0.5)),

            # penalties (defaults aligned to baseline.yaml)
            vega_penalty_lambda=float(rget("vega_penalty_lambda", 0.5)),
            funding_penalty_kappa=float(rget("funding_penalty_kappa", 1.0)),
            liquidity_penalty_theta=float(rget("liquidity_penalty_theta", 0.001)),

            # tiers
            true_arbitrage_dni_threshold=float(rget("true_arbitrage_dni_threshold", 0.0)),
            near_arbitrage_dni_threshold=float(rget("near_arbitrage_dni_threshold", -100.0)),
            near_arbitrage_prob_threshold=float(rget("near_arbitrage_prob_threshold", 0.8)),
            high_probability_threshold=float(rget("high_probability_threshold", 0.7)),
            moderate_probability_threshold=float(rget("moderate_probability_threshold", 0.5)),

            # clipping
            probability_clip_floor=float(rget("probability_clip_floor", 0.001)),
            probability_clip_ceiling=float(rget("probability_clip_ceiling", 0.999)),

            # other
            min_correlation_threshold=float(rget("min_correlation_threshold", 0.3)),
            covariance_regularization=float(rget("covariance_regularization", 1e-6)),

            # execution cross-overs
            transaction_cost_rate=float((e or {}).get("transaction_cost_rate", 0.002)),
            kelly_fraction_cap=float((e or {}).get("kelly_fraction_cap", 0.25)),

            # used by ProbabilityRanker, add to YAML ranking.* as needed
            market_impact_coefficient=float(rget("market_impact_coefficient", 0.1)),
        )


# Keep the existing API surface
ranking_config = RankingConfigView.from_app()

def get_ranking_config() -> RankingConfigView:
    return ranking_config