"""
Variance Swap Hedging Strategy

This strategy uses variance swaps to hedge prediction market exposure,
particularly when the prediction depends on volatility regimes.

Mathematical foundation:
- Variance swap payoff: N_var * (σ²_realized - σ²_strike)
- Static replication: K_var = (2e^(rT)/T) * (∫P(K)/K² dK + ∫C(K)/K² dK)
- Hedges probability shifts due to volatility movements

Academic references:
- Demeterfi et al. (1999): "More Than You Ever Wanted to Know About Volatility Swaps"
  https://emanuelerman.com/media/volatility-swaps.pdf
  Complete derivation of variance swap replication formula
  
- Carr & Madan (1998): "Towards a Theory of Volatility Trading"
  https://math.nyu.edu/~carrp/papers/voltrading.pdf
  Log contract and volatility derivatives theory
"""

from typing import Dict, List, Optional, Tuple, Any, Sequence
import numpy as np
from datetime import datetime, timezone, date
from math import erf, log, sqrt
import math, os, logging, warnings
import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from utils.instrument_capture import BASE_UNIT
from utils.validation_audit import emit as _audit_emit
from pm.contract import PMContract, collect_unique_expiries as pm_collect_unique_expiries
from filters.option_expiry import filter_options_by_expiry

logger = logging.getLogger("VarianceSwapStrategy")

#
# LOGGING GATE (requested): only emit extra quote/basis samples when LOG_OPTIONS_DEBUG=1
#
LOG_OPTIONS_DEBUG = os.getenv("LOG_OPTIONS_DEBUG", "0") == "1"

# Config defaults (may also be set via external config/constructor)
# - If not set here, getattr(...) defaults inside the code will be used.
# self.max_forward_pm_gap = 0.05
# self.exit_spread_pct_default = 0.01
# self.require_cost_recovery = True
# Feature flags & tolerances (toggle here)
PM_ANCHORED_STATIC_REPLICATION = True     # default ON
COST_RECOVERY_TOL = 0.02                  # start at 2%, configurable
CLEAN_QUOTES = True
NO_ARB_ENFORCE = True
DIGITAL_WIDTH_BPS = float(os.getenv("DIGITAL_WIDTH_BPS", "0"))  # used by digital bounds below

# --- Base class import with robust fallback for tests/standalone ---
try:
    from .base_options_strategy import BaseOptionsStrategy  # type: ignore
except Exception:
    try:
        from base_options_strategy import BaseOptionsStrategy  # type: ignore
    except Exception:
        class BaseOptionsStrategy:
            """Minimal stub so this module can be imported and tested without external deps."""
            def __init__(self, risk_free_rate: float = 0.0) -> None:
                self.risk_free_rate = float(risk_free_rate)
                self.logger = logging.getLogger(self.__class__.__name__)
                self.n_simulations = 2048
                self.use_greeks_haircut = False
                self.exit_horizon_seconds = 24 * 3600
                self.delta_sigma_quantile = 0.84
                self.slippage_bps = 0.0
                self.forward_fee_bps = 0.0
            def get_strategy_name(self) -> str:
                return self.__class__.__name__
            def validate_inputs(self, polymarket_contract, current_spot) -> bool:
                return True
            def validate_cost_recovery_at_strike(self, *args, **kwargs) -> bool:
                return True
            def filter_options_by_expiry(self, options: List[Dict], pm_days_to_expiry: float, *, inclusive: bool = True) -> List[Dict]:
                """
                Build a list of instruments from up to N **valid** expiries on/after PM.
                Valid expiry requires: not synthetic/flagged, has >= min_quotes_per_expiry instruments with **both** bid>0 and ask>0,
                and (for additional expiries) within max_expiry_gap_days when far expiries are allowed.
                Far expiries allowed only if (expiry_policy == 'allow_far_with_unwind') and (options_unwind_model != 'intrinsic_only').
                """
                # TEST-ONLY breadcrumb (guarded): prove we reached expiry selection
                if os.getenv("VALIDATION_AUDIT_ENABLE", "0") == "1":
                    try:
                        path = os.getenv("VALIDATION_AUDIT_PATH", "analysis/validation_audit.jsonl")
                        Path(path).parent.mkdir(parents=True, exist_ok=True)
                        with open(path, "a") as fh:
                            fh.write(json.dumps({
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "event": "entered_filter_options_by_expiry",
                                "pm_days_to_expiry": float(pm_days_to_expiry or 0.0),
                                "num_options": int(len(options or []))
                            }) + "\n")
                    except Exception:
                        pass
                if not options:
                    return []
                cutoff_days = float(pm_days_to_expiry or 0.0)
                # Group by expiry and keep only groups on/after the PM date
                by_expiry: Dict[str, List[Dict]] = {}
                for o in options:
                    try:
                        d = float(o.get("days_to_expiry", float("nan")))
                    except Exception:
                        d = float("nan")
                    if (inclusive and not math.isnan(d) and d + 1e-9 >= cutoff_days) or \
                       ((not inclusive) and not math.isnan(d) and d > cutoff_days + 1e-9):
                        exp = o.get("expiry_date") or o.get("expiration") or o.get("expiry")
                        if exp:
                            by_expiry.setdefault(str(exp), []).append(o)
                if not by_expiry: return []
                # Sort expiries by proximity to PM date (ascending days_to_expiry)
                exp_ranks = []
                for exp, group in by_expiry.items():
                    ds = [float(g.get('days_to_expiry') or float('inf')) for g in group if g.get('days_to_expiry') is not None]
                    rep_days = min(ds) if ds else float('inf')
                    exp_ranks.append((exp, rep_days))
                exp_ranks.sort(key=lambda x: x[1])

                policy = str(getattr(self, 'expiry_policy', 'nearest_on_or_after')).lower()
                allow_far = (policy == 'allow_far_with_unwind') and (str(getattr(self, 'options_unwind_model', OPTIONS_UNWIND_MODEL)).lower() != 'intrinsic_only')
                max_n = int(getattr(self, 'max_expiries_considered', 1) or 1)
                max_gap = float(getattr(self, 'max_expiry_gap_days', 14) or 14)

                chosen, out = [], []
                for exp, rep_days in exp_ranks:
                    if len(chosen) >= max_n: break
                    if not allow_far and len(chosen) >= 1: break
                    if len(chosen) >= 1 and allow_far:
                        # enforce gap for additional expiries
                        if rep_days - exp_ranks[0][1] > max_gap + 1e-9:
                            continue
                    group = by_expiry[exp]
                    # Skip synthetic / flagged expiries
                    if any((g.get('has_live_prices') is False) or (g.get('skip_for_execution') is True) for g in group):
                        continue
                    # Require min # of two-sided quotes
                    live2 = 0
                    for g in group:
                        bid = float(g.get('bid') or 0.0)
                        ask = float(g.get('ask') or 0.0)
                        if (bid > 0.0) and (ask > 0.0):
                            live2 += 1
                    if live2 < int(getattr(self, 'min_quotes_per_expiry', 4)):
                        continue
                    out.extend(group)
                    chosen.append(exp)
                return out

# --- Option type normalizer with fallback ---
try:
    from .utils.opt_keys import normalize_opt_type  # type: ignore
except Exception:
    def normalize_opt_type(t):
        if t is None:
            return None
        s = str(t).strip().lower()
        if s in ("c", "call"):
            return "call"
        if s in ("p", "put"):
            return "put"
        return s

# --- Config constants with fallback defaults ---
try:
    from config_manager import (
        SLIPPAGE_BPS,
        SECONDS_PER_YEAR,
        OPTIONS_UNWIND_MODEL,
        VARIANCE_EXPIRY_POLICY,
        VARIANCE_MAX_EXPIRY_GAP_DAYS,
        VARIANCE_MAX_EXPIRIES_CONSIDERED,
        VARIANCE_REQUIRE_LIVE_QUOTES_FOR_TRADES,
        VARIANCE_STRIKE_PROXIMITY_WINDOW,
        VARIANCE_MIN_QUOTES_PER_EXPIRY,
    )  # type: ignore
except Exception:
    SLIPPAGE_BPS = 0.0
    SECONDS_PER_YEAR = 365 * 24 * 3600
    OPTIONS_UNWIND_MODEL = "intrinsic_only"
    VARIANCE_EXPIRY_POLICY = "nearest_on_or_after"
    VARIANCE_MAX_EXPIRY_GAP_DAYS = 14
    VARIANCE_MAX_EXPIRIES_CONSIDERED = 1
    VARIANCE_REQUIRE_LIVE_QUOTES_FOR_TRADES = True
    VARIANCE_STRIKE_PROXIMITY_WINDOW = 0.20
    VARIANCE_MIN_QUOTES_PER_EXPIRY = 4
try:
    from utils.instrument_capture import CAPTURE_ENABLED as _CAPTURE_ENABLED, record_option_leg as _record_option_leg
except Exception:
    _CAPTURE_ENABLED = False
    def _record_option_leg(*args, **kwargs): return

# --- add near the other imports ---
try:
    # your helper from: sparse-static-hedge/src/sparse_static_hedge/omp_hedge.py
    from sparse_static_hedge import (
        select_sparse_quadratic,
        round_options_and_repair_budget,
        refit_bases_given_fixed_options,
    )
    SPARSE_HEDGE_AVAILABLE = True
except Exception:
    SPARSE_HEDGE_AVAILABLE = False
    # Deterministic, minimal fallbacks (no external deps), used only in tests/standalone.
    def select_sparse_quadratic(*, X, y, price_vec, names, max_option_legs, l2, budget,
                                always_on_bases=(), candidate_idx=None, initial_active_idx=None, verbose=False):
        import numpy as _np
        n = len(names)
        w = _np.zeros(n, float)
        cand = [i for i in (candidate_idx or range(n)) if str(names[i]).startswith(("call_", "put_"))]
        cand.sort(key=lambda i: (float(price_vec[i]), str(names[i])))
        active = cand[: int(max_option_legs)]
        if not active:
            return w, {"mse_final": float(_np.mean((X @ w - y) ** 2)) if getattr(X, "shape", None) else 0.0}
        total_price = sum(float(price_vec[i]) for i in active) or 1.0
        unit = float(budget) / float(total_price)
        for i in active:
            w[i] = unit
        return w, {"mse_final": float(_np.mean((X @ w - y) ** 2)) if getattr(X, "shape", None) else 0.0}
    def round_options_and_repair_budget(names, w_trade, price_vec, min_contracts_by_asset=None,
                                        contract_increment_by_asset=None, contract_increment=0.01,
                                        base_names=(), min_notional_local=0.0, max_legs=None):
        step = Decimal(str(contract_increment or 0.01))
        out = []
        for nm, w in zip(names, w_trade):
            if str(nm).startswith(("call_", "put_")):
                q = (Decimal(str(w)) / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                out.append(float(q * step))
            else:
                out.append(float(w))
        return out
    def refit_bases_given_fixed_options(*, X, y, price_vec, names, fixed_idx, l2, budget, base_names=()):
        return price_vec * 0.0


# --- Deprecation helper ---
_warned: set[str] = set()
def _deprecated(name: str, details: str = "") -> None:
    if name not in _warned:
        warnings.warn(f"{name} is deprecated and unused. {details}".strip(), DeprecationWarning, stacklevel=2)
        _warned.add(name)

# --- Deterministic rounding helper to avoid float drift ---
def round_notional(value: float | Decimal, step: float | Decimal = 0.01, mode: str = "half_away_from_zero") -> float:
    """
    Round value to the nearest multiple of step using Decimal.
    Parameters
    ----------
    value : float | Decimal
    step : float | Decimal
        Quantization step (e.g., 0.01 contracts).
    mode : {"half_away_from_zero", "half_even"}
    """
    v = Decimal(str(value))
    s = Decimal(str(step))
    if s <= 0:
        return float(v)
    rounding = ROUND_HALF_UP if mode != "half_even" else ROUND_HALF_EVEN
    q = (v / s).quantize(Decimal("1"), rounding=rounding)
    return float(q * s)


class VarianceSwapStrategy(BaseOptionsStrategy):
    # Router binding: parent "options", child "variance_swap"
    TAGS = ("options.variance_swap",)
    CATEGORY = "options"
    """
    Implements variance swap hedging for volatility-sensitive predictions.
    
    Key features:
    - Replicates variance swaps using options weighted by 1/K²
    - Hedges volatility-linked probability movements
    - Useful when predictions correlate with volatility regimes
    - Can combine with directional hedges
    """
    # --- PM digital-bounds gate (toggle) ---
    # Turn ON/OFF the bid-ask no-arbitrage bounds check for the PM contract at K0.
    # When enforced, opportunities are returned only when PM YES price lies OUTSIDE the bounds.
    use_pm_digital_bounds: bool = True
    enforce_pm_digital_bounds: bool = True
    pm_bounds_slack: float = 0.00  # optional slack around bounds (e.g., 0.01)
    pm_bounds_inclusive: bool = True  # treat boundary equality as inside
    # --- Feature flags & tolerances (toggle here) ---
    PM_ANCHORED_STATIC_REPLICATION = True
    COST_RECOVERY_TOL = 0.02                  # start at 2%, configurable
    CLEAN_QUOTES = True
    NO_ARB_ENFORCE = True

    
    def _normalize_option_record(self, opt: dict) -> dict:
        t = (opt.get("type")
             or opt.get("option_type")
             or (opt.get("metadata") or {}).get("type"))
        nt = normalize_opt_type(t)
        if nt:
            opt["type"] = nt
        return opt
    
    def __init__(
        self,
        cfg,
        logger,
        *args,
        risk_free_rate: float = 0.05,
        min_strikes_required: int = 6,
        otm_cutoff: float = 0.50,
        volatility_sensitivity_threshold: float = 0.0,
        **kwargs
    ):
        import sys
        import datetime
        print(f"\n{'*'*80}", file=sys.stderr)
        print(f"VARIANCE SWAP STRATEGY INITIALIZED!", file=sys.stderr)
        print(f"{'*'*80}\n", file=sys.stderr)
        
        # Also write to a dedicated debug file
        debug_log_path = "debug_runs/variance_swap_debug.log"
        try:
            with open(debug_log_path, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"[{datetime.datetime.now().isoformat()}] VARIANCE SWAP STRATEGY INITIALIZED\n")
                f.write(f"Config: expiry_policy={getattr(cfg, 'hedging', {}).get('variance', {}).get('expiry_policy', 'unknown')}\n")
                f.write(f"{'='*80}\n")
        except Exception as e:
            print(f"Failed to write to debug log: {e}", file=sys.stderr)
        
        super().__init__(risk_free_rate)
        self._sparse_logged = False
        self._logged_option_examples = False  # ensures we only dump sample quotes once per run
        self.min_strikes_required = int(min_strikes_required)
        self.otm_cutoff = float(otm_cutoff)
        self.volatility_sensitivity_threshold = float(volatility_sensitivity_threshold)
        # === Strategy defaults / toggles ===
        # Anchor K0 at the PM strike by default (no snapping)
        self.PM_ANCHORED_STATIC_REPLICATION = PM_ANCHORED_STATIC_REPLICATION
        self.COST_RECOVERY_TOL = float(os.getenv('COST_RECOVERY_TOL', COST_RECOVERY_TOL))
        self.CLEAN_QUOTES = CLEAN_QUOTES
        self.NO_ARB_ENFORCE = NO_ARB_ENFORCE
        # NOTE: PM‑digital proxy width is deterministic from the strike ladder; no width knobs.
        # Anchor mode derived from feature flag:
        self.variance_anchor_mode = 'pm' if bool(self.PM_ANCHORED_STATIC_REPLICATION) else 'forward'     # anchor K0 at PM strike unless disabled
        # Direction-aware cost-recovery gate
        self.SIGN_AWARE_COST_RECOVERY = True
        self.use_greeks_haircut = True       # apply Δ/Γ/Θ haircut in cost-recovery by default
        self.exit_horizon_seconds = 45.0     # short liquidation horizon (seconds)
        self.delta_sigma_quantile = 1.0      # 1σ; raise to 1.645 for ~95% one-sided

        # ---- Sparse selection & rounding knobs (same semantics as quadratic) ----
        self.use_sparse_helper = True
        self.max_legs = 6
        self.regularization = 1.0e-8
        self.cost_budget = 0.0
        self.include_bonds = True
        self.include_forwards = True
        self.always_on_bases = ("bond_T1", "S_T")
        self.bond_basis_name = "bond_T1"

        # rounding (venue/asset aware); same defaults as quadratic
        self.contract_increment_by_asset = {"BTC": 0.01, "ETH": 0.01}
        self.min_contracts_by_asset = {"BTC": 0.01, "ETH": 0.01}
        self.min_notional = 25.0
        self.contract_increment = 0.01
        self.min_contracts = 0.01
        self.slippage_bps = 10.0
        self.forward_fee_bps = 0

        self.n_simulations = 50_000  # state grid size for L2 hedging
        self._load_sparse_hedge_config()

        # --- New: cost-recovery & PM-bounds controls ---
        # Tolerance for near-threshold accepts (helps with rounding/lotting noise)
        self.COST_RECOVERY_TOL = float(self.COST_RECOVERY_TOL)  # class-default; can be overridden
        # Toggle sign-aware comparison for cost recovery (recommended: True)
        self.SIGN_AWARE_COST_RECOVERY = True
        # Read from config (with safe fallbacks)
        hedging_cfg = getattr(cfg, "hedging", None)
        s = getattr(hedging_cfg, "variance", cfg) if hedging_cfg else cfg
        self.use_pm_digital_bounds = bool(getattr(s, "use_pm_digital_bounds", True))
        self.enforce_pm_digital_bounds = bool(getattr(s, "enforce_pm_digital_bounds", False))
        self.pm_bounds_inclusive = bool(getattr(s, "pm_bounds_inclusive", False))
        self.pm_bounds_slack_abs = float(getattr(s, "pm_bounds_slack_abs", 0.02))
        self.pm_bounds_slack_rel = float(getattr(s, "pm_bounds_slack_rel", 0.01))
        self.pm_bounds_use_tradable = bool(getattr(s, "pm_bounds_use_tradable", True))
        self.pm_bounds_require_two_sided = bool(getattr(s, "pm_bounds_require_two_sided", False))
        self.pm_bounds_max_rel_width = float(getattr(s, "pm_bounds_max_rel_width", 0.02))
        # Minimum strikes required post-filter; allow override via config (hedging.variance.min_strikes_required)
        self.min_strikes_required = int(getattr(s, "min_strikes_required", self.min_strikes_required))
        self.pm_gate_use_pm_strike = bool(getattr(s, "pm_gate_use_pm_strike", True))
        # Liquidity-aware knobs
        self.var_F_in_range = bool(getattr(s, "variance_forward_must_be_in_range", False))
        self.var_F_clip = bool(getattr(s, "variance_forward_clip_to_range", True))
        self.var_zero_mode = str(getattr(s, "variance_zero_bid_truncation_mode", "vix")).lower()
        self.var_zero_streak = int(getattr(s, "variance_zero_bid_streak", 2))
        self.var_require_both_wings = bool(getattr(s, "variance_require_both_wings", False))
        self.var_min_wing_quotes = int(getattr(s, "variance_min_wing_quotes", 1))
        self.var_one_sided_rel_spread = float(getattr(s, "variance_one_sided_rel_spread", 0.05))
        self.var_one_sided_abs_spread = float(getattr(s, "variance_one_sided_abs_spread", 0.00))
        self.var_pen_bps_one_sided = float(getattr(s, "variance_penalty_bps_per_one_sided", 5.0))
        self.var_pen_bps_missing_wing = float(getattr(s, "variance_penalty_bps_missing_wing", 25.0))
        self.var_pen_bps_per_day_gap = float(getattr(s, "variance_penalty_bps_per_day_gap", 2.0))

        # Strategy-local overrides (if present) fall back to repo-level config constants
        self.expiry_policy = str(getattr(s, "expiry_policy", VARIANCE_EXPIRY_POLICY)).lower()
        # 'nearest_on_or_after' (default) or 'allow_far_with_unwind'
        self.max_expiry_gap_days = int(getattr(s, "max_expiry_gap_days", VARIANCE_MAX_EXPIRY_GAP_DAYS))
        self.max_expiries_considered = int(getattr(s, "max_expiries_considered", VARIANCE_MAX_EXPIRIES_CONSIDERED))
        self.require_live_quotes = bool(getattr(s, "require_live_quotes_for_trades", VARIANCE_REQUIRE_LIVE_QUOTES_FOR_TRADES))
        # Strike preselection window for sparse: fraction around K0 (e.g., 0.2 => [0.8K0, 1.2K0])
        self.strike_proximity_window = float(getattr(s, "strike_proximity_window", VARIANCE_STRIKE_PROXIMITY_WINDOW))
        # Minimum instruments with *two-sided* quotes required to admit an expiry
        self.min_quotes_per_expiry = int(getattr(s, "min_quotes_per_expiry", VARIANCE_MIN_QUOTES_PER_EXPIRY))
        # allow tests/callers to flip unwind gating without re-importing config
        self.options_unwind_model = str(OPTIONS_UNWIND_MODEL).lower()
        
        self.logger.info(
            "[VARCFG] unwind=%s policy=%s max_gap_days=%s max_expiries=%s live_quotes=%s strike_window=%.2f min_quotes_per_expiry=%s",
            OPTIONS_UNWIND_MODEL, self.expiry_policy, self.max_expiry_gap_days,
            self.max_expiries_considered, self.require_live_quotes,
            self.strike_proximity_window, self.min_quotes_per_expiry
        )
    
    def get_strategy_name(self) -> str:
        return "Variance Swap Hedge"

    def validate_cost_recovery_at_strike(
        self,
        strike: float | None = None,
        *,
        portfolio: object | None = None,
        **kwargs: Any,
    ) -> bool:
        """
        Adapter over BaseOptionsStrategy.validate_cost_recovery_at_strike with back-compat.
        Accepts both `strike` and `pm_strike`, and ignores unknown kwargs (e.g., `portfolio`).
        Returns a boolean pass/fail (or True if the base impl is unavailable).
        """
        pm_strike = kwargs.pop("pm_strike", None)
        strike = float(strike if strike is not None else pm_strike if pm_strike is not None else 0.0)
        kwargs.pop("portfolio", None)  # drop extra kwarg if base doesn't accept it
        try:
            return super().validate_cost_recovery_at_strike(pm_strike=strike, **kwargs)  # type: ignore[arg-type]
        except TypeError:
            try:
                return super().validate_cost_recovery_at_strike(strike=strike, **kwargs)  # type: ignore[arg-type]
            except TypeError:
                try:
                    return super().validate_cost_recovery_at_strike(strike)  # type: ignore[misc]
                except Exception:
                    if not getattr(self, "_warned_validate_sig", False):
                        self.logger.warning("variance.validate_cost_recovery: falling back (signature mismatch)")
                        setattr(self, "_warned_validate_sig", True)
                    return True

    
    def _load_sparse_hedge_config(self) -> None:
        import os, yaml
        try:
            cfg = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'baseline.yaml')
            if os.path.exists(cfg):
                with open(cfg, 'r') as f:
                    conf = yaml.safe_load(f) or {}
            else:
                return

            h = (conf.get('hedging') or {})
            v = h.get('variance') or h.get('quadratic') or {}
            def set_if(k): 
                if k in v: setattr(self, k, v[k])

            for k in (
                'use_sparse_helper','max_legs','regularization','cost_budget',
                'include_bonds','include_forwards','bond_basis_name','always_on_bases',
                'contract_increment_by_asset','min_contracts_by_asset','min_notional',
                'contract_increment','min_contracts','slippage_bps','forward_fee_bps'
            ):
                set_if(k)
            if not self._sparse_logged:
                self.logger.info(f"[variance sparse] use_sparse={self.use_sparse_helper}, max_legs={self.max_legs}")
                self._sparse_logged = True
        except Exception as e:
            self.logger.debug(f"Could not load sparse hedge config for variance: {e}")
    
    def evaluate_opportunities(
        self,
        polymarket_contract: Dict,
        hedge_instruments: Dict,
        current_spot: float,
        position_size: float = 100
    ) -> List[Dict]:
        """
        Evaluate variance swap hedging opportunities.
        """
        # Debug logging
        import datetime
        debug_log_path = "debug_runs/variance_swap_debug.log"
        try:
            with open(debug_log_path, "a") as f:
                f.write(f"\n[{datetime.datetime.now().isoformat()}] evaluate_opportunities CALLED\n")
                f.write(f"  - PM Market: {polymarket_contract.get('question', 'N/A')}\n")
                f.write(f"  - Currency: {polymarket_contract.get('currency', 'N/A')}\n")
                f.write(f"  - Current spot: {current_spot}\n")
                f.write(f"  - Options available: {len(hedge_instruments.get('options', []))}\n")
        except Exception:
            pass
            
        if not self.validate_inputs(polymarket_contract, current_spot):
            try:
                with open(debug_log_path, "a") as f:
                    f.write(f"  - VALIDATION FAILED\n")
            except Exception:
                pass
            return []
        
        options_data = [self._normalize_option_record(o) for o in (hedge_instruments.get('options') or [])]
        # DEBUG: summarize quote sources to ensure live data is used
        try:
            total = len(options_data)
            by_src = {"WS": 0, "REST": 0, "EST": 0, "UNKNOWN": 0}
            for q in options_data:
                src = str(q.get("source") or ("EST" if q.get("price_estimated") else None) or "UNKNOWN").upper()
                if src not in by_src:
                    by_src["UNKNOWN"] += 1
                else:
                    by_src[src] += 1
            self.logger.debug("[HEDGE] Using LIVE options quotes: %d legs, source mix: WS:%d REST:%d EST:%d UNKNOWN:%d",
                              total, by_src["WS"], by_src["REST"], by_src["EST"], by_src["UNKNOWN"])
        except Exception:
            pass
        if not options_data:
            self.logger.debug("No options data available")
            return []
        
        # Volatility-sensitivity heuristic disabled by policy (no text scanning).
        vol_sensitivity = 1.0
        
        opportunities = []
        
        # Extract PM contract details
        pm_strike = polymarket_contract.get('strike_price')
        yes_price = polymarket_contract.get('yes_price')
        no_price = polymarket_contract.get('no_price')
        if no_price is None:
            # Do NOT infer 1 - YES. Only use real NO book if present; otherwise drop.
            try:
                nb = float(polymarket_contract.get('no_best_bid') or 0.0)
                na = float(polymarket_contract.get('no_best_ask') or 0.0)
            except Exception:
                nb, na = 0.0, 0.0
            if nb > 0.0 and na > 0.0:
                no_price = 0.5 * (nb + na)
                self.logger.debug("[variance] filled missing NO price from book mid: %.6f", no_price)
            elif nb > 0.0:
                no_price = nb
                self.logger.debug("[variance] filled missing NO price from best bid: %.6f", no_price)
            elif na > 0.0:
                no_price = na
                self.logger.debug("[variance] filled missing NO price from best ask: %.6f", no_price)
            else:
                self.logger.info("[variance] skip: NO price unavailable and NO book empty")
                return []
        # carry PM context for audit breadcrumb
        try:
            self._last_pm_market_id = (polymarket_contract.get('id')
                or polymarket_contract.get('question_id') or polymarket_contract.get('slug'))
        except Exception:
            self._last_pm_market_id = None
        days_to_expiry = polymarket_contract.get('days_to_expiry')
        
        # Filter options that expire after PM
        suitable_options = self.filter_options_by_expiry(options_data, days_to_expiry)
        
        if not suitable_options:
            self.logger.debug("No options with suitable expiry found")
            return []
        
        # Group by expiry
        options_by_expiry = self._group_by_expiry(suitable_options)
        
        # Evaluate strategies for each expiry
        for expiry_date, expiry_options in options_by_expiry.items():
            self.logger.debug(f"[variance] expiry={expiry_date}: raw options={len(expiry_options)}, unique strikes={len(set(o['strike'] for o in expiry_options))}")
            # Let the portfolio builder enforce grid adequacy; don't pre-filter here
            
            # Evaluate YES position with variance hedge
            yes_opportunity = self._evaluate_variance_hedge(
                'YES', yes_price, pm_strike,
                expiry_date, expiry_options, current_spot,
                position_size, polymarket_contract, vol_sensitivity
            )
            if yes_opportunity:
                opportunities.append(yes_opportunity)
            
            # Evaluate NO position with variance hedge
            no_opportunity = self._evaluate_variance_hedge(
                'NO', no_price, pm_strike,
                expiry_date, expiry_options, current_spot,
                position_size, polymarket_contract, vol_sensitivity
            )
            if no_opportunity:
                opportunities.append(no_opportunity)
        
        return opportunities
    
    def _assess_volatility_sensitivity(self, polymarket_contract: Dict) -> float:
        """
        Deactivated: per policy we do not inspect title/description.
        Keep the method for interface compatibility.
        """
        return 1.0
    
    def _evaluate_variance_hedge(
        self,
        pm_side: str,
        pm_price: float,
        pm_strike: float,
        expiry_date: str,
        expiry_options: List[Dict],
        current_spot: float,
        position_size: float,
        polymarket_contract: Dict,
        vol_sensitivity: float,
        pm_entry_cost: Optional[float] = None,
        pm_market_side: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Evaluate variance swap hedge for PM position.
        """
        try:
            # Treat expiry as end-of-day UTC for stability; 'now' in UTC
            option_expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            now_utc = datetime.now(timezone.utc)
            expiry_dt_utc = datetime.combine(option_expiry_date, datetime.max.time()).replace(tzinfo=timezone.utc)
            seconds_to_option_expiry = max(0.0, (expiry_dt_utc - now_utc).total_seconds())
            days_to_option_expiry = seconds_to_option_expiry / (24 * 3600)
            
            # Skip expired or too-close expiries (e.g., < 0.5 trading days)
            if seconds_to_option_expiry <= 0 or days_to_option_expiry < 0.5:
                return None
                
            time_to_expiry = max(seconds_to_option_expiry / SECONDS_PER_YEAR, 1e-8)
            
            # Build variance swap replication portfolio
            var_swap_portfolio = self._build_variance_swap_portfolio(
                expiry_options, current_spot, time_to_expiry, pm_strike=pm_strike
            )
            
            if not var_swap_portfolio:
                self.logger.debug(f"[variance] expiry={expiry_date}: portfolio build failed (None)")
                return None
            self.logger.debug(f"[variance] expiry={expiry_date}: portfolio built with puts={len(var_swap_portfolio.get('puts',[]))}, calls={len(var_swap_portfolio.get('calls',[]))}")
            
            # Calculate variance swap parameters
            var_swap_params = self._calculate_variance_swap_params(
                var_swap_portfolio, current_spot, time_to_expiry
            )
            
            # Determine variance exposure based on PM position
            var_exposure = self._determine_variance_exposure(
                pm_side, polymarket_contract, vol_sensitivity
            )
            
            # --- NEW: build <= max_legs sparse legs that replicate the log contract ---
            sigma_for_states = float(var_swap_params.get("strike_volatility", 0.0))
            var_notional = float(var_exposure) * float(position_size)   # same scale you show elsewhere

            # Prepare data for sparse variance legs
            if SPARSE_HEDGE_AVAILABLE and self.use_sparse_helper and int(self.max_legs) > 0:
                F  = float(var_swap_portfolio['forward'])
                K0 = float(var_swap_portfolio['k0'])
                r  = float(self.risk_free_rate)
                
                # Candidate strike universe = observed strikes (quotes cleaned in portfolio)
                strikes = sorted({ float(o['strike']) for o in var_swap_portfolio.get('options', []) })
                
                # State grid (risk‑neutral) and L2 target: log‑contract payoff
                S_T = self._make_state_grid(current_spot, time_to_expiry, max(1e-8, float(sigma_for_states)),
                                            n=self.n_simulations, r=r, seed=42)
                y = -2.0 * np.log(np.maximum(S_T, 1e-12) / max(F, 1e-12))
                
                # Basis and prices aligned with names
                X, names = self._build_payoff_basis_for_sparse(S_T, strikes,
                                                               include_bond=self.include_bonds,
                                                               include_forward=self.include_forwards)
                p_vec = self._price_basis_from_quotes(
                    spot=current_spot, r=r, T_years=time_to_expiry, strikes=strikes, names=names,
                    portfolio=var_swap_portfolio, slippage_bps=self.slippage_bps, forward_fee_bps=self.forward_fee_bps
                )
                
                # Restrict candidates to OTM side only (classical variance replication sign pattern)

                # Restrict candidates to OTM side only (classical variance replication sign pattern)
                # AND keep them within a proximity window around K0 to avoid far‑wing junk.
                candidate_idx = []
                win = float(self.strike_proximity_window)
                K_low, K_high = K0*(1.0 - win), K0*(1.0 + win)
                puts_cand = sum(1 for nm,K in zip(names, strikes) if nm.startswith("put_") and (K <= K0) and (K >= K_low))
                calls_cand = sum(1 for nm,K in zip(names, strikes) if nm.startswith("call_") and (K >= K0) and (K <= K_high))
                self.logger.info("[SPARSE] K0=%.2f window=±%.0f%% → candidates: puts=%d calls=%d",
                                 K0, win*100, puts_cand, calls_cand)
                for i, nm in enumerate(names):
                    if nm.startswith("put_"):
                        K = float(nm.split("_",1)[1])
                        if (K <= K0) and (K >= K_low):
                            candidate_idx.append(i)
                    elif nm.startswith("call_"):
                        K = float(nm.split("_",1)[1])
                        if (K >= K0) and (K <= K_high):
                            candidate_idx.append(i)
                    # bond_T1 / S_T are handled by always_on_bases
                
                sparse_legs, sparse_diag = self._sparse_variance_legs(
                    var_swap_portfolio, var_notional, X, y, p_vec, names, candidate_idx, sigma_for_states
                )
            else:
                sparse_legs, sparse_diag = [], {"reason": "sparse_helper_disabled"}
            
            self.logger.info(f"[variance] Sparse legs returned: {len(sparse_legs) if sparse_legs else 0} legs, diag: {sparse_diag}")
            
            # Calculate hedge cost
            hedge_cost = self._calculate_variance_hedge_cost(
                var_swap_portfolio, var_exposure, position_size
            )
            
            # GATE: skip if forward/PM gap is large when using forward anchoring
            F = float(var_swap_portfolio['forward'])
            try:
                pm_gap = abs(F - float(pm_strike)) / max(1e-12, float(pm_strike))
            except Exception:
                pm_gap = None
            max_gap = getattr(self, "max_forward_pm_gap", 0.05)  # 5% default
            if (var_swap_portfolio.get('anchor_mode', 'forward') == 'forward') and (pm_gap is not None) and (pm_gap > max_gap):
                self.logger.info(f"[variance] skip: forward/PM gap {pm_gap:.2%} exceeds {max_gap:.2%}")
                return None
            
            # Calculate PM payoffs
            shares, pm_payoff_true, pm_payoff_false = self.calculate_pm_payoffs(
                pm_side, pm_price, position_size
            )
            
            # Simulate payoffs under different volatility scenarios
            payoff_scenarios = self._simulate_volatility_scenarios(
                pm_payoffs={'true': pm_payoff_true, 'false': pm_payoff_false},
                var_swap_params=var_swap_params,
                var_exposure=var_exposure,
                position_size=position_size,
                vol_sensitivity=vol_sensitivity,
                pm_strike=pm_strike,
                time_to_expiry=time_to_expiry  # keep consistent with all other calculations
            )
            
            # Calculate P&L metrics
            max_profit = max(payoff_scenarios.values()) - hedge_cost
            max_loss = min(payoff_scenarios.values()) - hedge_cost
            expected_profit = np.mean(list(payoff_scenarios.values())) - hedge_cost
            
            # Probability of profit (net per scenario)
            if len(payoff_scenarios) > 0:
                prob_profit = sum(1 for p in payoff_scenarios.values() if (p - hedge_cost) > 0) / len(payoff_scenarios)
            else:
                prob_profit = 0.0
            
            # Get is_above from polymarket contract
            is_above = polymarket_contract.get('is_above', True)
            
            # Entry cost & legs (centralized entry-side handling)
            hedge_cost_new, hedge_positions = self._compute_variance_hedge_entry_cost_and_legs(
                portfolio=var_swap_portfolio,
                var_exposure=float(var_exposure),
                position_size=float(position_size),
            )
            
            # Expected profit numbers (unchanged)
            max_profit = max(payoff_scenarios.values()) - hedge_cost_new
            max_loss = min(payoff_scenarios.values()) - hedge_cost_new
            expected_profit = np.mean(list(payoff_scenarios.values())) - hedge_cost_new
            prob_profit = (sum(1 for p in payoff_scenarios.values() if (p - hedge_cost_new) > 0) / len(payoff_scenarios)) if payoff_scenarios else 0.0
            
            # 4) Directional, tolerance-aware cost recovery at PM strike (S = K0)
            # BaseOptionsStrategy API expects days_to_pm_expiry and entry cost (no 'portfolio' kw)
            _days_to_pm_expiry = float(time_to_expiry) * (SECONDS_PER_YEAR / (24.0*3600.0))
            # Robustly validate cost recovery; tolerate signature mismatches
            try:
                recovery_check = self.validate_cost_recovery_at_strike(
                    pm_strike=pm_strike,
                    hedge_positions=sparse_legs,
                    hedge_entry_cost=hedge_cost_new,
                    days_to_pm_expiry=_days_to_pm_expiry,
                    current_spot=current_spot,
                    default_exit_spread_pct=None,  # use per-quote bid/ask by default
                    apply_greeks_haircut=self.use_greeks_haircut,
                    haircut_dt_seconds=self.exit_horizon_seconds,
                    delta_sigma_quantile=self.delta_sigma_quantile,
                    portfolio=var_swap_portfolio,
                )
            except TypeError as _sig_e:
                if not getattr(self, "_warned_validate_sig", False):
                    self.logger.warning(
                        "variance.validate_cost_recovery: adapter used; proceeding",
                        extra={"diag": {"error": str(_sig_e)}},
                    )
                    setattr(self, "_warned_validate_sig", True)
                try:
                    recovery_check = super().validate_cost_recovery_at_strike(
                        pm_strike=pm_strike,
                        hedge_positions=sparse_legs,
                        hedge_entry_cost=hedge_cost_new,
                        days_to_pm_expiry=_days_to_pm_expiry,
                        current_spot=current_spot,
                    )
                except Exception:
                    recovery_check = True
            recovery_check = recovery_check or {}
            # Sign-aware comparison: align "recovered" and "needed" with hedge direction
            direction = 1.0 if float(var_exposure) >= 0.0 else -1.0
            needed_raw    = float(recovery_check.get("total_cost_to_recover", 0.0))
            recovered_raw = float(recovery_check.get("hedge_value_at_strike", 0.0))
            if self.SIGN_AWARE_COST_RECOVERY:
                needed_dir    = direction * needed_raw
                recovered_dir = direction * recovered_raw
            else:
                needed_dir, recovered_dir = needed_raw, recovered_raw
            passes_recovery = (recovered_dir + 1e-12) >= (1.0 - float(self.COST_RECOVERY_TOL)) * needed_dir
            if not passes_recovery:
                self.logger.info(
                    "[variance] reject: cost recovery at PM strike %s fails: recovers %+.2f vs needed %+.2f "
                    "(dir=%+d, aligned: %.2f vs %.2f, tol=%.2f%%)",
                    f"{pm_strike:.1f}",
                    recovered_raw, needed_raw, int(direction),
                    recovered_dir, needed_dir, 100.0*float(self.COST_RECOVERY_TOL),
                )
                return None
            else:
                self.logger.info(
                    "[variance] accept: cost recovery OK at PM strike %s: recovers %+.2f vs needed %+.2f "
                    "(dir=%+d, aligned: %.2f vs %.2f, tol=%.2f%%)",
                    f"{pm_strike:.1f}",
                    recovered_raw, needed_raw, int(direction),
                    recovered_dir, needed_dir, 100.0*float(self.COST_RECOVERY_TOL),
                )

            # 5) PM digital-bounds gate at the PM strike (K_PM), with tradable bounds & slack
            pm_digital_bounds = None  # annotations for UI/debug; set below when available
            if self.use_pm_digital_bounds:
                K_pm = self._infer_pm_strike(polymarket_contract, var_swap_portfolio)
                if K_pm is not None:
                    bounds = self._digital_bounds_at_strike(
                        K_pm, var_swap_portfolio,
                        use_tradable=self.pm_bounds_use_tradable,
                        require_two_sided=self.pm_bounds_require_two_sided,
                        max_rel_width=self.pm_bounds_max_rel_width
                    )
                    lo, hi = bounds.get("lower"), bounds.get("upper")
                    pm_yes = self._pm_yes_price(polymarket_contract)
                    if lo is not None and hi is not None and pm_yes is not None:
                        # add slack (absolute and relative)
                        midb = 0.5 * (lo + hi)
                        slack = max(self.pm_bounds_slack_abs, self.pm_bounds_slack_rel * max(1e-12, midb))
                        lo_eff = max(0.0, lo - slack)
                        hi_eff = min(1.0, hi + slack)
                        inside = (lo_eff <= pm_yes <= hi_eff) if self.pm_bounds_inclusive else (lo_eff < pm_yes < hi_eff)
                        pm_digital_bounds = {
                            'lower': lo, 'upper': hi,
                            'lower_eff': lo_eff, 'upper_eff': hi_eff,
                            'pm_yes': pm_yes, 'inside': bool(inside)
                        }
                        # SOFT mode: annotate; HARD mode: prune
                        if self.enforce_pm_digital_bounds and inside:
                            self.logger.info(f"[variance] prune: PM YES within digital no-arb band at K_PM={K_pm:.4f} (pm={pm_yes:.4f}, band=[{lo:.4f},{hi:.4f}], slack={slack:.4f})")
                            return None
                    else:
                        self.logger.debug("[variance] skip digital gate (insufficient bounds or PM price)")
                        pm_digital_bounds = bounds  # record whatever we computed (may be None)
                else:
                    self.logger.debug("[variance] skip digital gate (no PM strike)")
                    pm_digital_bounds = None

            # 6) Build opportunity
            opportunity = {
                'strategy': self.get_strategy_name(),
                'description': self._build_hedge_description(var_swap_portfolio, var_exposure),
                'pm_side': pm_side,
                'pm_strike': pm_strike,
                'option_expiry': expiry_date,
                'days_to_option_expiry': days_to_option_expiry,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'expected_profit': expected_profit,
                'probability_profit': prob_profit,
                'cash_premium': hedge_cost_new,  # positive = cash outlay, negative = cash received
                'estimated_cash_needed': max(0.0, hedge_cost_new),
                'hedge_cost': hedge_cost_new,
                'variance_strike': var_swap_params['strike_variance'],
                'current_implied_vol': np.sqrt(var_swap_params['strike_variance']),
                'variance_notional': var_exposure * position_size,
                'volatility_sensitivity': vol_sensitivity,
                # robust to portfolio without a flat 'options' list
                'num_options_used': len(var_swap_portfolio.get('options') or ((var_swap_portfolio.get('puts') or []) + (var_swap_portfolio.get('calls') or []))),
                'pm_strike': pm_strike,
                'current_spot': current_spot,
                'cost_recovery': recovery_check,
                'pm_digital_bounds': pm_digital_bounds,
                # Standardized profit fields based on PM resolution
                # For variance swaps, use expected profit as hedge performance varies with volatility
                'profit_if_yes': (pm_payoff_true + expected_profit) if is_above else (pm_payoff_false + expected_profit),
                'profit_if_no': (pm_payoff_false + expected_profit) if is_above else (pm_payoff_true + expected_profit)
            }
            
            # Add standardized Lyra info with all strikes used (robust)
            _opts_for_strikes = var_swap_portfolio.get('options') or ((var_swap_portfolio.get('puts') or []) + (var_swap_portfolio.get('calls') or []))
            all_strikes = sorted({float(opt['strike']) for opt in _opts_for_strikes})
            opportunity['lyra'] = self.create_lyra_info(
                expiry_date=expiry_date,
                strikes=all_strikes[:2] if len(all_strikes) >= 2 else all_strikes + [0],  # Show first 2 strikes
                cost=hedge_cost,
                days_to_expiry=days_to_option_expiry
            )
            
            # Add detailed variance swap portfolio info
            opportunity['variance_swap_portfolio'] = {
                'puts': [{'strike': p['strike'], 'weight': p.get('weight', 0)} for p in var_swap_portfolio['puts']],
                'calls': [{'strike': c['strike'], 'weight': c.get('weight', 0)} for c in var_swap_portfolio['calls']],
                'atm_strike': var_swap_params.get('k0', var_swap_params.get('atm_strike')),
                'total_options': len(var_swap_portfolio['options'])
            }
            
            # --- Build a dense list of all non-zero-weight OTM options used in the replication ---
            dense_legs = []
            try:
                for q in (var_swap_portfolio.get("puts") or []) + (var_swap_portfolio.get("calls") or []):
                    w = float(q.get("weight", 0.0) or 0.0)
                    if w == 0.0:
                        continue
                    dense_legs.append({
                        "type": (q.get("type") or "").upper(),                          # "CALL" / "PUT"
                        "strike": float(q.get("strike", 0.0) or 0.0),
                        "action": "BUY" if w > 0 else "SELL",
                        # IMPORTANT: for dense academic replication we expose *weights*, not contracts
                        "weight": abs(w),
                        "expiry": expiry_date,
                        "bid": float(q.get("bid", 0.0) or 0.0),
                        "ask": float(q.get("ask", 0.0) or 0.0),
                        "instrument_id": f"Lyra:{'C' if ((q.get('type') or '').lower()=='call') else 'P'}:{int(float(q.get('strike',0) or 0))}",
                        "venue": q.get("venue", "Lyra"),
                        "symbol": polymarket_contract.get("currency", ""),
                        "fee_bps": float(getattr(self, "slippage_bps", 0.0) or 0.0),
                        "slippage_bps": float(getattr(self, "slippage_bps", 0.0) or 0.0),
                    })
            except Exception as _e:
                self.logger.debug(f"[variance] Could not build dense replication legs: {_e}")

            # Prefer sparse/executable legs for the UI summary; DO NOT show dense academic weights as contracts
            if sparse_legs:
                opportunity.setdefault("detailed_strategy", {})["required_options"] = sparse_legs
                self.logger.debug(f"[variance] UI summary legs = sparse ({len(sparse_legs)})")
            else:
                opportunity.setdefault("detailed_strategy", {})["required_options"] = []
                self.logger.debug(f"[variance] UI summary legs = empty (sparse helper returned 0 legs)")
            
            # Also surface expiry for fallback renderers
            opportunity.setdefault('option_expiry', expiry_date)
            
            # Add currency to opportunity
            opportunity['currency'] = polymarket_contract.get('currency', 'BTC')
            
            # Populate the fields Streamlit expects (mirrors quadratic), robust to missing 'options'
            if sparse_legs:
                _opts_src = var_swap_portfolio.get("options") or ((var_swap_portfolio.get("puts") or []) + (var_swap_portfolio.get("calls") or []))
                opportunity['hedge_instruments'] = [
                    {
                        "type": (q.get("type") or "").upper(),
                        "strike": float(q.get("strike", 0.0) or 0.0),
                        "expiry_date": expiry_date,
                        "bid": float(q.get("bid", 0.0) or 0.0),
                        "ask": float(q.get("ask", 0.0) or 0.0),
                        "has_live_prices": (q.get("bid") is not None and q.get("ask") is not None),
                        "instrument_name": f"Lyra:{'C' if ((q.get('type') or '').lower()=='call') else 'P'}:{int(float(q.get('strike',0) or 0))}",
                    }
                    for q in _opts_src
                ]
                opportunity["sparse_variance_fit"] = sparse_diag
            
            # 5) Capture only SPARSE legs (never capture dense fallback)
            if _CAPTURE_ENABLED:
                # wipe any earlier capture on this opp to avoid merging/duplicates
                opportunity.setdefault('execution_details', {})['options_legs'] = []
                if sparse_legs and len(sparse_legs) > 0:
                    for leg in sparse_legs:
                        _record_option_leg(
                            opportunity,
                            venue=leg.get("venue","Lyra"),
                            symbol=leg.get("symbol", polymarket_contract.get("currency","")),
                            type_=leg.get("type","CALL"),
                            action=leg.get("action","BUY"),
                            contracts=float(leg.get("contracts") or 0.0),
                            weight=float(leg.get("contracts") or 0.0),
                            expiry=str(leg.get("expiry")),
                            strike=float(leg.get("strike") or 0.0),
                            quote={"bid": leg.get("bid"), "ask": leg.get("ask"), "mid": leg.get("mid")},
                            greeks={},
                            fee_bps=float(leg.get("fee_bps") or 0.0),
                            slippage_bps=float(leg.get("slippage_bps") or 0.0),
                            instrument_id=leg.get("instrument_id"),
                        )
                else:
                    self.logger.info("[variance] sparse helper returned 0 legs; capture skipped (dense fallback not captured)")
            
            # --- Expose exact instruments to the UI / downstream execution ---
            try:
                ed = opportunity.setdefault("execution_details", {})
                meta = ed.get("_meta") or {}
                # Base notional unit so 'contracts' can be interpreted (e.g., weights scaled per $1,000)
                meta["base_unit"] = float(BASE_UNIT)
                ed["_meta"] = meta
                # Prefer sparse/executable legs only; DO NOT fall back to dense academic weights downstream
                ed["options_legs"] = (sparse_legs if (sparse_legs and len(sparse_legs) > 0) else []) or []
                which = "sparse" if (sparse_legs and len(sparse_legs) > 0) else "empty"
                self.logger.debug(f"[variance] execution_details.options_legs <- {which} ({len(ed['options_legs'])})")
            except Exception as _e:
                self.logger.debug(f"[variance] Could not populate execution_details.options_legs: {_e}")

            # One-line INFO summary with structured diag
            try:
                diag = dict(sparse_diag or {})
                diag.update({
                    "strategy": self.get_strategy_name(),
                    "contract": (polymarket_contract.get("slug") or polymarket_contract.get("id") or ""),
                    "pm_side": pm_side,
                    "expiry": str(expiry_date),
                    "legs_post_round": diag.get("legs_post_round", len(hedge_positions or [])),
                    "cost_scaled_pre_round": diag.get("cost_scaled_pre_round"),
                    "achieved_var_notional": diag.get("achieved_var_notional"),
                })
                self.logger.info("variance", extra={"diag": diag})
            except Exception:
                pass
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Error in variance swap hedge: {str(e)}")
            return None
    
    def _build_variance_swap_portfolio(
        self,
        options: List[Dict],
        current_spot: float,
        time_to_expiry: float,
        pm_strike: Optional[float] = None,
        pm_resolve_dt=None
    ) -> Optional[Dict]:
        """
        Builds the OTM strip around K0 with liquidity-aware behavior:
          • Forward outside range: clip if allowed.
          • Truncation: 'vix' (two zero-bid bids) or 'one_sided' (two fully-dead strikes).
          • One-sided quotes: allowed via synthetic mids; record penalty.
          • Missing wing: allowed (conservative lower bound) + penalty.
        Returns (portfolio, metrics) or (None, reason).
        """
        # Diagnostics
        self.logger.debug(f"[_build_var_port] inputs: raw options={len(options)} T={time_to_expiry:.6g}")
        
        T = max(float(time_to_expiry), 1e-12)
        r = float(self.risk_free_rate)

        # --- normalize & sanitize quotes (centralized) ---
        valid = self.normalize_options(options)

        if len(valid) < int(self.min_strikes_required):
            return None

        # --- forward via parity (weighted), fallback to S*e^{rT} ---
        byK = {}
        for q in valid:
            byK.setdefault(q['strike'], {}).setdefault(q['type'], q)

        F_candidates, W = [], []
        for K, qc in byK.items():
            p, c = qc.get('put'), qc.get('call')
            if not p or not c:
                continue
            put_mid  = 0.5*(p['bid'] + p['ask'])
            call_mid = 0.5*(c['bid'] + c['ask'])
            if put_mid <= 0.0 or call_mid <= 0.0:
                continue
            Fk = K + np.exp(r*T)*(call_mid - put_mid)      # from C-P = e^{-rT}(F-K)
            spread = (c['ask']-c['bid']) + (p['ask']-p['bid'])
            # higher weight for near-ATM (small |C-P|) and tight spreads
            w = 1.0 / (abs(call_mid - put_mid) + 0.1*spread + 1e-8)
            F_candidates.append(Fk); W.append(w)

        F = float(np.average(F_candidates, weights=W)) if F_candidates else float(current_spot)*float(np.exp(r*T))

        # --- forward window (optional) ---
        if getattr(self, 'otm_cutoff', None):
            lo, hi = F*(1.0 - float(self.otm_cutoff)), F*(1.0 + float(self.otm_cutoff))
            filtered = [q for q in valid if lo <= q['strike'] <= hi]
            # Only apply the cutoff if it still leaves a usable grid; otherwise keep the unfiltered set
            if len(filtered) >= int(self.min_strikes_required):
                valid = filtered
            else:
                self.logger.debug(f"[variance] OTM window would leave {len(filtered)} < {self.min_strikes_required}; keeping unfiltered quotes")

        valid.sort(key=lambda x: x['strike'])
        try:
            self.logger.debug(f"[_build_var_port] stage=otm_window n={len(valid)} F={float(F):.6g} cutoff={self.otm_cutoff}")
        except Exception:
            self.logger.debug(f"[_build_var_port] stage=otm_window n={len(valid)} F=? cutoff={self.otm_cutoff}")
        self.logger.debug(f"[_build_var_port] sanitized valid={len(valid)}")
        strikes_all = [q['strike'] for q in valid]

        # Guard against empty strike lists after filtering/deduplication
        if not strikes_all:
            self.logger.warning("[variance] no valid strikes after filtering; skipping expiry")
            return None

        # Guard: forward inside range (soft clamp instead of dropping the expiry)
        if len(strikes_all) < 2 or not (strikes_all[0] < F < strikes_all[-1]):
            if self.var_F_in_range and not self.var_F_clip:
                self.logger.info(
                    f"[_build_var_port] abort: forward {F:.6g} outside strike range "
                    f"[{strikes_all[0]:.6g}, {strikes_all[-1]:.6g}]"
                )
                return None
            if self.var_F_clip:
                F = max(min(F, strikes_all[-1]), strikes_all[0])
                self.logger.debug(f"[variance] clipped forward into range at {F:.6f}")

        # Anchoring mode: "forward" (Demeterfi) or "pm" (event-centric)
        anchor_mode = getattr(self, "variance_anchor_mode", "pm")
        if anchor_mode == "pm" and pm_strike is not None:
            # Keep K0 exactly at the PM strike; do not snap to the options grid.
            K0 = float(pm_strike)
        else:
            # Forward-anchored split (fallback): use F itself as the OTM divider.
            K0 = float(F)
        try:
            self.logger.debug(f"[_build_var_port] anchor={anchor_mode} K0={float(K0):.6g} pm_strike={pm_strike} F={float(F):.6g}")
        except Exception:
            self.logger.debug(f"[_build_var_port] anchor={anchor_mode} K0={K0} pm_strike={pm_strike}")
        
        # Ensure strikes on both sides of K0
        strikes_below = [k for k in strikes_all if k < K0]
        strikes_above = [k for k in strikes_all if k > K0]
        if not strikes_below or not strikes_above:
            # Throttle noisy logs when the chain has only one wing around the PM-anchored K0
            self._var_miss_both_sides_total = getattr(self, "_var_miss_both_sides_total", 0) + 1
            if self._var_miss_both_sides_total <= 5 or self._var_miss_both_sides_total % 100 == 0:
                self.logger.warning(
                    f"[variance] cannot build replication around K0={K0}: "
                    f"need strikes on both sides (suppressed; total={self._var_miss_both_sides_total})"
                )
            return None

        # helper: detect "zero" according to mode
        def _is_zero_bid(opt):
            b = opt.get("bid")
            try:
                b = float(b) if b is not None else 0.0
            except Exception:
                b = 0.0
            return b <= 0.0
        def _is_fully_dead(opt):
            b = opt.get("bid"); a = opt.get("ask")
            try: b = float(b) if b is not None else 0.0
            except: b = 0.0
            try: a = float(a) if a is not None else 0.0
            except: a = 0.0
            return b <= 0.0 and a <= 0.0
        def _zero_hit(opt):
            return _is_zero_bid(opt) if self.var_zero_mode == "vix" else _is_fully_dead(opt)

        # helper: effective mid handling one-sided quotes
        def _eff_mid(opt):
            b, a = opt.get("bid"), opt.get("ask")
            try: b = float(b) if b is not None else None
            except: b = None
            try: a = float(a) if a is not None else None
            except: a = None
            if b is not None and a is not None and b > 0 and a > 0:
                return 0.5 * (b + a), 0  # mid, extra_penalty_count
            # one-sided synthetic
            pen = 1
            rs, as_ = self.var_one_sided_rel_spread, self.var_one_sided_abs_spread
            if b is not None and b > 0:
                # synth ask from bid
                mid = max(0.0, b * (1.0 + 0.5*rs) + 0.5*as_)
                return mid, pen
            if a is not None and a > 0:
                # synth bid from ask
                mid = max(0.0, a * (1.0 - 0.5*rs) - 0.5*as_)
                return mid, pen
            return 0.0, 0  # fully dead

        # collect OTM wings with truncation
        K0f = float(K0)
        left = sorted([p for p in valid if p['type']=='put'  and float(p["strike"]) <  K0f], key=lambda x: float(x["strike"]), reverse=True)
        right= sorted([c for c in valid if c['type']=='call' and float(c["strike"]) >  K0f], key=lambda x: float(x["strike"]))

        def _truncate(seq):
            streak = 0
            out = []
            for opt in seq:
                if _zero_hit(opt):
                    streak += 1
                else:
                    streak = 0
                if streak >= self.var_zero_streak:
                    break
                out.append(opt)
            return out

        puts_trunc  = _truncate(left)
        calls_trunc = _truncate(right)
        self.logger.debug(f"[_build_var_port] after zero-bid trunc: puts={len(puts_trunc)}, calls={len(calls_trunc)}")
        try:
            self.logger.debug(f"[_build_var_port] stage=zero_bid puts={len(puts_trunc)} calls={len(calls_trunc)} K0={float(K0):.6g}")
        except Exception:
            self.logger.debug(f"[_build_var_port] stage=zero_bid puts={len(puts_trunc)} calls={len(calls_trunc)} K0=?")
        at_k0       = [q for q in valid if q['strike']==K0]  # can contain both P and C

        # require some coverage on both wings to avoid pathological integrals
        missing_wing = (len(puts_trunc) < self.var_min_wing_quotes) or (len(calls_trunc) < self.var_min_wing_quotes)
        if self.var_require_both_wings and missing_wing:
            self.logger.debug("Insufficient OTM coverage on one wing after zero-bid truncation")
            self._var_wing_coverage_total = getattr(self, "_var_wing_coverage_total", 0) + 1
            if self._var_wing_coverage_total <= 5 or self._var_wing_coverage_total % 100 == 0:
                self.logger.info(
                    f"[_build_var_port] result=none reason=wing_coverage (suppressed; total={self._var_wing_coverage_total})"
                )
            return None

        # --- assemble final, de-duplicated option list ---
        # Keep only true OTM on each wing; K0 enters only via at_k0
        puts_clean  = [q for q in puts_trunc  if float(q['strike']) < float(K0)]
        calls_clean = [q for q in calls_trunc if float(q['strike']) > float(K0)]

        candidate = []
        candidate.extend(puts_clean)
        candidate.extend(calls_clean)
        candidate.extend(at_k0)  # exactly one put and one call at K0

        # De-duplicate by (type, strike); merge to best market: highest bid, lowest ask
        dedup = {}
        for q in candidate:
            typ = (q.get('type') or q.get('option_type') or '').strip().lower()
            K   = float(q.get('strike', 0.0) or 0.0)
            bid = float(q.get('bid', 0.0) or 0.0)
            ask = float(q.get('ask', 0.0) or 0.0)
            if ask <= 0.0:
                # cannot trade at zero/neg ask; skip
                continue
            bid = max(0.0, bid)
            key = (typ, K)

            prev = dedup.get(key)
            if prev is None:
                dedup[key] = {'type': typ, 'strike': K, 'bid': bid, 'ask': ask}
            else:
                prev['bid'] = max(prev['bid'], bid)
                prev['ask'] = ask if (prev['ask'] <= 0.0 or (ask > 0.0 and ask < prev['ask'])) else prev['ask']

        # Final sorted list
        valid = sorted(dedup.values(), key=lambda x: (float(x['strike']), x['type']))
        self.logger.debug(f"[_build_var_port] deduped valid={len(valid)} (unique strikes={len(set(v['strike'] for v in valid))})")
        if len(valid) < int(self.min_strikes_required):
            return None

        # Sanity: at most one put and one call exactly at K0
        k0_rows = [q for q in valid if abs(float(q['strike']) - float(K0)) <= 1e-12]
        assert sum(1 for q in k0_rows if q['type'] == 'put') <= 1, "Duplicate PUT at K0"
        assert sum(1 for q in k0_rows if q['type'] == 'call') <= 1, "Duplicate CALL at K0"

        # --- unique strike grid for ΔK (avoid K0 double counting) ---
        grid = sorted({q['strike'] for q in valid})

        def deltaK(i):
            if i == 0:
                return grid[1] - grid[0]
            if i == len(grid) - 1:
                return grid[-1] - grid[-2]
            return 0.5*(grid[i+1] - grid[i-1])

        weights = {}
        scale = (2.0*np.exp(r*T))/T
        for i, K in enumerate(grid):
            dK = float(deltaK(i))
            if dK <= 0.0:
                self.logger.debug(f"Nonpositive ΔK at K={K:.6g}; skipping strike")
                continue
            weights[K] = scale*(dK/(K*K))
            # sparse wing diagnostic (raised to 25% as per VIX convention)
            if (dK / K) > 0.25:
                self.logger.debug(f"Sparse wing at K={K:.6g} (ΔK/K={dK/K:.1%}) — variance may be underestimated")

        # Build discrete approximation of the variance integral using effective mids
        one_sided_count = 0
        def _contrib(opts, side):
            nonlocal one_sided_count
            contrib = []
            prevK = None
            for opt in opts:
                K = float(opt["strike"])
                mid, pen = _eff_mid(opt)
                one_sided_count += pen
                if prevK is not None:
                    dK = abs(K - prevK)
                else:
                    dK = 0.0
                # store for later integration weighting 2*mid/dK/K^2 (Demeterfi discretization)
                contrib.append({"K": K, "mid": mid})
                prevK = K
            return contrib

        left_c  = _contrib(puts_trunc, "put")
        right_c = _contrib(calls_trunc,"call")

        # If a wing is entirely missing, proceed with conservative lower bound (treat missing wing as zero contrib)
        # but mark a liquidity penalty to EV.
        if not left_c and not right_c:
            return None

        # Compute liquidity penalties
        penalty_bps = 0.0
        penalty_bps += one_sided_count * self.var_pen_bps_one_sided
        if missing_wing:
            penalty_bps += self.var_pen_bps_missing_wing
        if pm_resolve_dt is not None and hasattr(options[0] if options else None, 'expiry'):
            try:
                import datetime as _dt
                expiry = options[0].expiry if hasattr(options[0], 'expiry') else options[0].get('expiry_date')
                if expiry:
                    if isinstance(expiry, str):
                        expiry_dt = _dt.datetime.strptime(expiry, '%Y-%m-%d')
                    else:
                        expiry_dt = expiry
                    days_gap = max(0.0, (expiry_dt - pm_resolve_dt).total_seconds() / 86400.0)
                    penalty_bps += days_gap * self.var_pen_bps_per_day_gap
            except Exception:
                pass

        # Continue with existing logic for building portfolio
        # --- split OTM sets (keep K0, but mark ½ weight if both sides present) ---
        puts, calls = [], []
        for q in valid:
            K = q['strike']
            w = weights.get(K, 0.0)
            row = dict(q)
            if K < K0 and q['type']=='put':
                row['weight'] = w
                puts.append(row)
            elif K > K0 and q['type']=='call':
                row['weight'] = w
                calls.append(row)
            elif K == K0:
                # keep both, but set half-weight to be explicit (not used for cost since we skip K0)
                row['weight'] = 0.5*w
                if q['type']=='put':
                    puts.append(row)
                elif q['type']=='call':
                    calls.append(row)

        self.logger.debug(f"Variance replication: {len(puts)} puts, {len(calls)} calls, K0={K0:.6g}, F={F:.6g}, T={T:.6g}")

        portfolio = {
            'forward': float(F),
            'k0': float(K0),
            'replication_weights': {float(k): float(v) for k, v in weights.items()},
            'options': valid,  # sanitized list of option quotes with bid/ask/iv
            'puts': puts,
            'calls': calls,
            'liquidity_penalty_bps': penalty_bps,
            'one_sided_count': one_sided_count,
            'missing_wing': missing_wing,
            'zero_mode': self.var_zero_mode,
        }
        # Attach anchor info for diagnostics
        portfolio['anchor_mode'] = anchor_mode
        portfolio['pm_strike'] = float(pm_strike) if pm_strike is not None else None
        portfolio['strikes_all'] = strikes_all
        return portfolio
    
    def _calculate_variance_swap_params(self, portfolio: Dict, current_spot: float, time_to_expiry: float) -> Dict:
        """
        Discrete Demeterfi/CBOE replication:
            σ² = (2 e^{rT} / T) * Σ[ΔK/K² * Q(K)]  −  (1/T) * ((F/K0 − 1)²)
        Q(K) = put(K) for K<K0; call(K) for K>K0; ½(put+call) at K0.
        """
        T = max(float(time_to_expiry), 1e-12)
        F = float(portfolio['forward'])
        K0 = float(portfolio['k0'])

        # unique strike grid and precomputed weights
        grid = sorted({float(k) for k in portfolio.get('replication_weights', {}).keys()})
        self.logger.debug(f"[_calc_params] grid_size={len(grid)} K0={K0:.6g} F={F:.6g}")
        weights = {float(k): float(v) for k, v in portfolio.get('replication_weights', {}).items()}

        # mid quotes on each side
        put_mid  = {float(o['strike']): 0.5*(float(o['bid']) + float(o['ask'])) for o in portfolio.get('puts', [])}
        call_mid = {float(o['strike']): 0.5*(float(o['bid']) + float(o['ask'])) for o in portfolio.get('calls', [])}

        # basic sanity around K0
        if not grid or not (grid[0] <= K0 <= grid[-1]):
            return {'strike_variance': 0.0, 'strike_volatility': 0.0, 'forward': F, 'k0': K0}

        summation = 0.0
        for K in grid:
            if K < K0:
                Q = put_mid.get(K, 0.0)
            elif K > K0:
                Q = call_mid.get(K, 0.0)
            else:
                p = put_mid.get(K, 0.0); c = call_mid.get(K, 0.0)
                Q = 0.5*(p + c) if (p > 0.0 or c > 0.0) else 0.0

            if Q <= 0.0:
                continue  # positivity guard on integrand

            summation += weights[K] * Q

        sigma2 = max(0.0, float(summation - (1.0/T)*((F/K0 - 1.0)**2)))
        return {
            'strike_variance': sigma2,
            'strike_volatility': float(np.sqrt(sigma2)),
            'forward': F,
            'k0': K0,
        }
    
    def _determine_variance_exposure(
        self,
        pm_side: str,
        polymarket_contract: Dict,
        vol_sensitivity: float
    ) -> float:
        """
        Determine variance swap exposure based on PM position.
        
        Returns notional as fraction of position size.
        """
        # Base exposure on volatility sensitivity
        base_exposure = vol_sensitivity
        
        # Adjust based on PM position
        is_above = polymarket_contract.get('is_above', True)
        
        if pm_side == 'YES':
            if is_above:
                # YES on "above" - short variance (bet on stability)
                exposure = -base_exposure
            else:
                # YES on "below" - long variance (bet on volatility)
                exposure = base_exposure
        else:  # NO
            if is_above:
                # NO on "above" - long variance
                exposure = base_exposure
            else:
                # NO on "below" - short variance
                exposure = -base_exposure
        
        # Scale exposure
        return exposure * 0.5  # Conservative sizing
    
    def _calculate_variance_hedge_cost(self, portfolio, var_exposure: float, position_size: float) -> float:
        """
        Cash cost for the static replication:
          qty_i = |var_exposure| * position_size * weight[K_i]
        Long variance -> buy OTM options at ask; short variance -> sell at bid.
        At K0 we hold 1/2 weight in the put and 1/2 weight in the call.

        This function also consolidates any duplicate rows by (type, strike),
        keeping the highest bid and lowest ask per key so we don't double-count.
        """
        try:
            notional = abs(float(var_exposure)) * float(position_size)
        except Exception:
            return 0.0
        if notional == 0.0:
            return 0.0

        # optional slippage (in bps) via config; safe default = 0
        try:
            from config_manager import SLIPPAGE_BPS as _SLIP
            slippage = float(_SLIP) * 1e-4
        except Exception:
            slippage = 0.0

        direction = 1.0 if var_exposure > 0 else -1.0

        # weights & K0
        weights = {float(k): float(v) for k, v in (portfolio.get('replication_weights') or {}).items()}
        K0 = float(portfolio.get('k0', 0.0) or 0.0)

        # --- consolidate duplicates by (type, strike) ---
        consolidated = {}
        for row in (portfolio.get('options') or []):
            typ = (row.get('type') or '').strip().lower()
            try:
                K = float(row.get('strike', 0.0) or 0.0)
            except Exception:
                continue
            bid = float(row.get('bid', 0.0) or 0.0)
            ask = float(row.get('ask', 0.0) or 0.0)
            if ask <= 0.0:
                continue
            bid = max(0.0, bid)
            key = (typ, K)
            prev = consolidated.get(key)
            if prev is None:
                consolidated[key] = {'type': typ, 'strike': K, 'bid': bid, 'ask': ask}
            else:
                prev['bid'] = max(prev['bid'], bid)
                prev['ask'] = ask if (prev['ask'] <= 0.0 or (ask > 0.0 and ask < prev['ask'])) else prev['ask']

        # --- price legs ---
        total = 0.0
        for (_, K), row in consolidated.items():
            w = float(weights.get(K, 0.0))
            if w <= 0.0:
                continue
            # split K0 across the two legs (call & put)
            if abs(K - K0) <= 1e-12:
                w *= 0.5

            qty = notional * w

            bid = row['bid']; ask = row['ask']
            mid = 0.5 * (bid + ask) if (bid > 0.0 and ask > 0.0) else max(bid, ask, 0.0)

            if direction > 0:  # buy
                px = ask if ask > 0.0 else (mid * (1.0 + slippage))
            else:              # sell
                px = bid if bid > 0.0 else (mid * (1.0 - slippage))

            if px <= 0.0:
                continue

            total += direction * qty * px

        return float(total)

    
    def _simulate_volatility_scenarios(
        self,
        pm_payoffs: Dict[str, float],
        var_swap_params: Dict,
        var_exposure: float,
        position_size: float,
        vol_sensitivity: float,
        pm_strike: float,
        time_to_expiry: float
    ) -> Dict[str, float]:
        """
        Simulate P&L under different volatility scenarios using lognormal distribution.
        
        For binary PMs tied to a threshold at expiry, compute:
        P(S_T >= H) = N(d2), where d2 = [ln(S0/H) + (r - σ²/2)T] / (σ√T)
        """
        scenarios = {}
        
        # Define volatility scenarios
        current_vol = var_swap_params['strike_volatility']
        vol_scenarios = {
            'low_vol': current_vol * 0.5,
            'normal_vol': current_vol,
            'high_vol': current_vol * 1.5,
            'extreme_vol': current_vol * 2.0
        }
        
        # Recover S0 from forward price
        S0 = var_swap_params['forward'] * np.exp(-self.risk_free_rate * time_to_expiry)
        H = pm_strike
        
        for scenario_name, realized_vol in vol_scenarios.items():
            # Variance swap payoff
            realized_var = realized_vol ** 2
            strike_var = var_swap_params['strike_variance']
            var_payoff = var_exposure * position_size * (realized_var - strike_var)
            
            # Calculate probability under lognormal diffusion
            sigma = max(1e-8, realized_vol)
            d2 = (np.log(S0 / H) + (self.risk_free_rate - 0.5 * sigma**2) * time_to_expiry) / (sigma * np.sqrt(time_to_expiry))
            prob_true = float(0.5 * (1.0 + erf(d2 / np.sqrt(2.0))))  # N(d2)
            
            # Weight PM payoffs by calculated probabilities
            expected_pm = prob_true * pm_payoffs['true'] + (1 - prob_true) * pm_payoffs['false']
            
            # Total P&L
            total_pnl = expected_pm + var_payoff
            scenarios[scenario_name] = total_pnl
        
        return scenarios
    
    def _build_hedge_description(self, portfolio: Dict, var_exposure: float) -> str:
        """Build description of variance swap hedge."""
        direction = "LONG" if var_exposure > 0 else "SHORT"
        num_puts = len(portfolio['puts'])
        num_calls = len(portfolio['calls'])
        
        return f"{direction} variance swap via {num_puts} puts + {num_calls} calls (1/K² weighted)"
    
    
    def filter_options_by_expiry(self, options, pm_days_to_expiry, *, inclusive: bool = True):
        """
        Return a list of instruments drawn from up to N **valid** expiries on/after the PM date.
        Valid expiry requires:
          - not synthetic / not flagged skip_for_execution
          - if require_live_quotes: bid>0 and ask>0 and has_live_prices=True
          - at least `min_quotes_per_expiry` such instruments for the expiry to count
          - additional expiries allowed only if:
              expiry_policy == 'allow_far_with_unwind'
              and OPTIONS_UNWIND_MODEL != 'intrinsic_only'
              and (expiry_dte - pm_dte) <= max_expiry_gap_days
        Steps:
          1) group options by expiry_date (accept keys: 'expiry_date' | 'expiry' | 'expiration')
          2) drop expiries strictly before PM (or <= if inclusive=False)
          3) sort by |expiry_dte - pm_dte|
          4) scan forward until we have at most `max_expiries_considered` valid expiries
        """
        # --- AGGRESSIVE DEBUG: START ---
        import sys
        import datetime
        print(f"\n\n{'='*80}", file=sys.stderr)
        print(f"VARIANCE SWAP: filter_options_by_expiry CALLED!", file=sys.stderr)
        print(f"Options count: {len(options) if options else 0}", file=sys.stderr)
        print(f"PM days to expiry: {pm_days_to_expiry}", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        
        # Write to dedicated debug file
        debug_log_path = "debug_runs/variance_swap_debug.log"
        try:
            with open(debug_log_path, "a") as f:
                f.write(f"\n[{datetime.datetime.now().isoformat()}] filter_options_by_expiry CALLED\n")
                f.write(f"  - Options count: {len(options) if options else 0}\n")
                f.write(f"  - PM days to expiry: {pm_days_to_expiry}\n")
                f.write(f"  - Inclusive: {inclusive}\n")
                if options and len(options) > 0:
                    # Sample first few options
                    f.write(f"  - Sample options (first 3):\n")
                    for i, opt in enumerate(options[:3]):
                        f.write(f"    Option {i}: expiry={opt.get('expiry_date', 'N/A')}, dte={opt.get('days_to_expiry', 'N/A')}\n")
        except Exception as e:
            print(f"Failed to write to debug log: {e}", file=sys.stderr)
        
        # --- DEBUG: inventory BEFORE selection ---
        import json, os, math, datetime as _dt
        _audit_path = os.environ.get("EXPIRY_DEBUG_PATH", "debug_runs/expiry_debug.jsonl")
        try:
            os.makedirs(os.path.dirname(_audit_path), exist_ok=True)
        except Exception:
            pass

        try:
            inv = {}
            for o in (options or []):
                exp = o.get("expiry_date") or o.get("expiration") or o.get("expiry")
                dte = o.get("days_to_expiry")
                key = (str(exp), None if dte is None else float(dte))
                inv[key] = inv.get(key, 0) + 1
            row = {
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "stage": "pre_filter_inventory",
                "pm_days_to_expiry": float(pm_days_to_expiry or 0.0),
                "pre_filter_expiries": [
                    {"expiry": k[0], "rep_days_to_expiry": k[1], "count": c}
                    for k, c in sorted(inv.items(), key=lambda kv: (kv[0][1] if kv[0][1] is not None else 1e9))
                ],
            }
            with open(_audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"DEBUG: Wrote expiry debug to {_audit_path}", file=sys.stderr)
            
            # Also note in our debug log
            with open(debug_log_path, "a") as f:
                f.write(f"  - Wrote expiry inventory to {_audit_path}\n")
                f.write(f"  - Found {len(inv)} unique expiries\n")
        except Exception as e:
            print(f"DEBUG: Failed to write expiry debug: {e}", file=sys.stderr)
            with open(debug_log_path, "a") as f:
                f.write(f"  - ERROR writing expiry debug: {e}\n")

        from math import inf
        pm_dte = float(pm_days_to_expiry or 0.0)
        allow_far = (self.expiry_policy == "allow_far_with_unwind" and OPTIONS_UNWIND_MODEL != "intrinsic_only")
        maxN = int(self.max_expiries_considered or 1)
        gap = float(self.max_expiry_gap_days or 0.0)

        # Group options by expiry (normalize keys)
        by_exp = {}
        for o in options or []:
            exp = o.get("expiry_date") or o.get("expiry") or o.get("expiration")
            if not exp:
                # skip silently; we can log rare keyless entries at debug level if needed
                continue
            try:
                dte = float(o.get("days_to_expiry", inf))
            except Exception:
                dte = inf
            # Keep original record plus normalized fields
            rec = dict(o)
            rec["_expiry_norm"] = exp
            rec["_dte"] = dte
            by_exp.setdefault(exp, []).append(rec)

        # Build candidate expiries on/after PM date
        def _is_two_sided(x):
            bid = float(x.get("bid") or 0.0)
            ask = float(x.get("ask") or 0.0)
            if self.require_live_quotes:
                return (bid > 0.0) and (ask > 0.0) and bool(x.get("has_live_prices", False)) and not bool(x.get("skip_for_execution", False))
            else:
                return not bool(x.get("skip_for_execution", False))

        # Prepare a sortable list: (distance, dte, exp_str, instruments[])
        rows = []
        for exp, arr in by_exp.items():
            # select instruments with (optional) live, two-sided quotes
            twos = [x for x in arr if _is_two_sided(x)]
            # expiry must be on/after the PM date
            if inclusive:
                on_after = [x for x in twos if (x["_dte"] >= pm_dte)]
            else:
                on_after = [x for x in twos if (x["_dte"] > pm_dte)]
            if not on_after:
                continue
            # distance from PM date, using the first item's dte (all same expiry)
            dte_val = min(x["_dte"] for x in on_after)
            dist = abs(dte_val - pm_dte)
            rows.append((dist, dte_val, exp, on_after))

        rows.sort(key=lambda r: (r[0], r[1]))

        # Scan forward, honoring far-expiry policy and gap, and min_quotes_per_expiry
        selected, reasons = [], {"pre_pm": 0, "far_not_allowed": 0, "gap_exceeded": 0, "insufficient_two_sided": 0, "ok": 0}
        for _, dte_val, exp, on_after in rows:
            if dte_val < pm_dte or (not inclusive and dte_val <= pm_dte):
                reasons["pre_pm"] += 1
                continue
            if len(on_after) < int(self.min_quotes_per_expiry or 1):
                reasons["insufficient_two_sided"] += 1
                continue
            # first valid expiry is always ok; subsequent ones are gated
            if selected:
                if not allow_far:
                    reasons["far_not_allowed"] += 1
                    continue
                if (dte_val - pm_dte) > gap:
                    reasons["gap_exceeded"] += 1
                    continue
            reasons["ok"] += 1
            selected.append((exp, dte_val, on_after))
            if len(selected) >= maxN:
                break

        # Flatten instruments from selected expiries
        out = []
        for _, _, arr in selected:
            out.extend(arr)

        self.logger.info(
            "[EXPSEL] pm_dte=%.2f allow_far=%s gap=%d maxN=%d → selected=%s reasons=%s",
            pm_dte, allow_far, int(gap), int(maxN), [exp for exp, _, _ in selected], reasons
        )
        return out

    def _group_by_expiry(self, options: List[Dict]) -> Dict[str, List[Dict]]:
        """Group options by expiry date."""
        grouped = {}
        for option in options:
            expiry = option.get('expiry_date')
            if expiry:
                if expiry not in grouped:
                    grouped[expiry] = []
                grouped[expiry].append(option)
        return grouped
    
    def _make_state_grid(self, spot: float, T_years: float, sigma: float, *, 
                         n: int = None, r: float = 0.0, seed: Optional[int] = 42) -> np.ndarray:
        """Risk‑neutral GBM terminals; used for L2 static hedging target."""
        n = int(n or self.n_simulations)
        if T_years <= 0.0:
            return np.full(n, float(spot), float)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n)
        return float(spot) * np.exp((r - 0.5 * sigma * sigma) * T_years + sigma * math.sqrt(T_years) * z)

    def _build_payoff_basis_for_sparse(self, S_T: np.ndarray, strikes: List[float],
                                       include_bond: bool = True, include_forward: bool = True
                                      ) -> Tuple[np.ndarray, List[str]]:
        """Columns: [1 (bond), S_T (forward), (S_T-K)^+ for calls, (K-S_T)^+ for puts]."""
        cols, names = [], []
        if include_bond:
            cols.append(np.ones_like(S_T, float)); names.append("bond_T1")
        if include_forward:
            cols.append(np.array(S_T, float));    names.append("S_T")
        for K in strikes:
            cols.append(np.maximum(np.array(S_T)-K, 0.0)); names.append(f"call_{K:g}")
            cols.append(np.maximum(K-np.array(S_T), 0.0)); names.append(f"put_{K:g}")
        return np.column_stack(cols), names

    def _price_basis_from_quotes(self, *, spot: float, r: float, T_years: float,
                                 strikes: List[float], names: List[str],
                                 portfolio: Dict, slippage_bps: float = 0.0, forward_fee_bps: float = 0.0
                                ) -> np.ndarray:
        """
        Price each basis column at t=0 using MARKET QUOTES (mid) for options,
        e^{-rT} for bond, and spot (± fees) for the forward basis S_T.
        Names MUST align with _build_payoff_basis_for_sparse(...).
        """
        # Map quotes
        call_mid = { float(c['strike']): 0.5*(float(c['bid'])+float(c['ask'])) 
                     for c in portfolio.get('calls',[]) if c.get('bid') is not None and c.get('ask') is not None }
        put_mid  = { float(p['strike']): 0.5*(float(p['bid'])+float(p['ask'])) 
                     for p in portfolio.get('puts' ,[]) if p.get('bid') is not None and p.get('ask') is not None }

        # Map per-strike contract multipliers if provided; default to 1.0 (USD-quoted per coin).
        call_cm = { float(c.get('strike')): float(c.get('contract_multiplier', 1.0))
                    for c in portfolio.get('calls', []) if c.get('strike') is not None }
        put_cm  = { float(p.get('strike')): float(p.get('contract_multiplier', 1.0))
                    for p in portfolio.get('puts',  []) if p.get('strike') is not None }

        # Optional debug: log a few quotes and a sample basis conversion
        if os.getenv('LOG_OPTIONS_DEBUG', '0') == '1':
            try:
                _samples = (portfolio.get('calls', []) + portfolio.get('puts', []))[:3]
                for i, q in enumerate(_samples):
                    b = float(q.get('bid') or 0.0); a = float(q.get('ask') or 0.0)
                    cm = float(q.get('contract_multiplier', 1.0))
                    mid_local = 0.5*(b+a) if (b > 0 and a > 0) else 0.0
                    mid_usd = mid_local * cm
                    _sz = q.get('size') or q.get('best_bid_qty') or q.get('best_ask_qty') or q.get('qty')
                    self.logger.info(f"[opts_debug] ex{i+1} K={q.get('strike')} bid={b} ask={a} size={_sz} mid_usd≈{mid_usd:.6g}")
                # Sample one K for the triplet
                sampleK = None
                if call_mid: sampleK = next(iter(call_mid.keys()))
                elif put_mid: sampleK = next(iter(put_mid.keys()))
                if sampleK is not None:
                    sample_mid = call_mid.get(sampleK, put_mid.get(sampleK, 0.0))
                    sample_cm  = call_cm.get(sampleK, put_cm.get(sampleK, 1.0))
                    sample_mid_usd = sample_mid * sample_cm
                    self.logger.info(f"[_price_basis] sample_mid={sample_mid:.6g} (venue units), spot={float(spot):.6g}, sample_mid_usd≈{sample_mid_usd:.6g}")
            except Exception:
                pass

        # After building call_mid/put_mid maps:
        sampleK = next(iter(call_mid.keys() or put_mid.keys() or []), None)
        if sampleK is not None:
            sample_mid = call_mid.get(sampleK, put_mid.get(sampleK, 0.0))
            try:
                self.logger.info(f"[_price_basis] sample_mid={sample_mid:.6g} (quoted units), spot={spot:.6g}, sample_mid_usd≈{sample_mid*float(spot):.2f}")
            except Exception:
                self.logger.info(f"[_price_basis] sample_mid={sample_mid} (quoted units), spot={spot}")

        p = []
        for nm in names:
            if nm == "bond_T1":
                p.append(math.exp(-r*T_years))
            elif nm == "S_T":
                p.append(float(spot) * (1.0 + float(forward_fee_bps)*1e-4))
            elif nm.startswith("call_"):
                K = float(nm.split("_",1)[1]); mid = call_mid.get(K, 0.0)
                cm = call_cm.get(K, 1.0)
                p.append(mid * cm * (1.0 + float(slippage_bps)*1e-4))
            elif nm.startswith("put_"):
                K = float(nm.split("_",1)[1]); mid = put_mid.get(K, 0.0)
                cm = put_cm.get(K, 1.0)
                p.append(mid * cm * (1.0 + float(slippage_bps)*1e-4))
            else:
                p.append(0.0)
        return np.array(p, float)
    
    def _representative_mid_usd_from_quotes(self, options: List[Dict[str, Any]], spot: float) -> Optional[float]:
        """
        Compute a representative option mid-price in **USD** terms for cost basis.
        Unit convention (assumption consistent with options_data_enricher / chain collector):
          - bid/ask are quoted in currency units **per coin** (e.g., USD/coin).
          - contract_multiplier scales per‑contract pricing. For most venues this is 1.
        Therefore: mid_usd_per_contract = mid_local * contract_multiplier.
        Critically, we do **not** multiply by spot here (that would double‑count notional).
        """
        if not options:
            return None

        # Take a representative at/near‑the‑money quote for scale
        sample = options[0]
        bid = float(sample.get("bid", 0.0) or 0.0)
        ask = float(sample.get("ask", 0.0) or 0.0)
        if bid <= 0.0 and ask <= 0.0:
            return None
        if ask > 0.0 and bid > ask:
            # crossed or bad quote
            return None

        mid_local = 0.5 * ((bid if bid > 0.0 else 0.0) + (ask if ask > 0.0 else 0.0))
        if mid_local <= 0.0:
            return None

        # Correct unit conversion:
        contract_multiplier = float(sample.get("contract_multiplier", 1.0) or 1.0)
        sample_mid_usd = mid_local * contract_multiplier

        # Optional one‑time debug dump of a few quotes and the computed basis triplet
        if LOG_OPTIONS_DEBUG and not self._logged_option_examples:
            try:
                qdump = []
                for q in options[:3]:
                    qb = float(q.get("bid", 0.0) or 0.0)
                    qa = float(q.get("ask", 0.0) or 0.0)
                    qmul = float(q.get("contract_multiplier", 1.0) or 1.0)
                    qmid_local = 0.5 * (max(qb, 0.0) + max(qa, 0.0))
                    qmid_usd = qmid_local * qmul
                    qdump.append(
                        {
                            "strike": q.get("strike"),
                            "type": q.get("type"),
                            "bid": qb,
                            "ask": qa,
                            "size": q.get("size") or q.get("bid_size") or q.get("ask_size"),
                            "mid_usd": qmid_usd,
                        }
                    )
                self.logger.debug("[variance.debug] sample_quotes=%s", qdump)
                self.logger.debug("[variance.debug] price_basis_triplet: sample_mid_local=%.6f, spot=%.6f, sample_mid_usd=%.6f",
                             mid_local, spot, sample_mid_usd)
            finally:
                self._logged_option_examples = True

        return sample_mid_usd
    
    def _sparse_variance_legs(self, portfolio, var_notional, X, y, p_vec, names, candidate_idx, sigma_for_states):
        # Determine a sensible unit budget. If config gives 0.0, infer from dense replication.
        configured_budget = float(self.cost_budget)
        if abs(configured_budget) <= 1e-12:
            sign = 1.0 if float(var_notional) >= 0.0 else -1.0
            try:
                unit_budget = float(self._calculate_variance_hedge_cost(
                    portfolio, var_exposure=sign, position_size=1.0
                ))
            except Exception:
                unit_budget = 0.0
        else:
            unit_budget = configured_budget

        try:
            self.logger.debug(f"[_sparse_variance_legs] solve: budget_input={configured_budget:.6g} unit_budget={unit_budget:.6g}")
        except Exception:
            self.logger.debug("[_sparse_variance_legs] solve: budget_input=?, unit_budget=?")

        # Solve sparse L2 with budget (same API as quadratic)
        w_unit, info = select_sparse_quadratic(
            X=X, y=y, price_vec=p_vec, names=names,
            max_option_legs=int(self.max_legs),
            l2=float(self.regularization),
            budget=float(unit_budget),
            always_on_bases=tuple(self.always_on_bases),
            candidate_idx=candidate_idx,
            initial_active_idx=None,
            verbose=False,
        )
        # Enforce same-sign option weights for variance strips:
        # For long variance (var_notional >= 0), all option weights must be >= 0.
        # For short variance (var_notional < 0), all option weights must be <= 0.
        try:
            before_clip = sum(1 for i,nm in enumerate(names) if nm.startswith(("call_","put_")) and w_unit[i] != 0.0)
            opt_idx = [i for i, nm in enumerate(names) if nm.startswith(("call_", "put_"))]
            if float(var_notional) >= 0.0:
                for i in opt_idx:
                    if w_unit[i] < 0.0:
                        w_unit[i] = 0.0
            else:
                for i in opt_idx:
                    if w_unit[i] > 0.0:
                        w_unit[i] = 0.0
            after_clip = sum(1 for i in opt_idx if w_unit[i] != 0.0)
            self.logger.info("[STRIP] same-sign enforced: weights_nonzero before=%d after=%d", before_clip, after_clip)
        except Exception:
            pass
        try:
            mse_unit = info.get('mse_final', float(np.mean((X @ w_unit - y) ** 2)))
        except Exception:
            mse_unit = float(np.mean((X @ w_unit - y) ** 2))
        try:
            self.logger.debug(f"[_sparse_variance_legs] unit_solution: mse={mse_unit:.3e} cost_unit={float(p_vec @ w_unit):.6g}")
        except Exception:
            self.logger.debug(f"[_sparse_variance_legs] unit_solution: mse={mse_unit} cost_unit=?")

        # Scale to desired notional exposure (var_notional may be +/-)
        w_trade = w_unit * float(var_notional)
        cost_scaled = float(p_vec @ w_trade)
        # ---------- Quantization-aware global scaling (literature: linear replication) ----------
        # We may have a correct continuous solution whose option weights are too small to survive
        # exchange lot/min-notional constraints. Because the variance (log-contract) replication
        # is linear in option payoffs and prices (Carr–Lee 2009; Demeterfi et al. 1999),
        # we can multiply all option weights by a single scalar α without changing the payoff shape.
        # This ensures at least one leg clears both the contract floor and min notional before quantization.
        # Convert USD min_notional to "price units" when quotes are in underlying units.
        F = float(portfolio.get("forward") or 0.0)  # forward/spot in USD
        quote_in_underlying = bool(portfolio.get("quote_in_underlying", True))
        min_notional_local = float(self.min_notional)
        if quote_in_underlying and F > 0.0:
            min_notional_local = float(self.min_notional) / F
        # Debug: report min_notional_local for first solve
        if os.getenv('LOG_OPTIONS_DEBUG','0')=='1':
            try:
                self.logger.info(f"[opts_debug] min_notional_local≈{min_notional_local:.6g}")
            except Exception: pass

        # Compute α so that max option contract and value clear thresholds.
        opt_idx = [i for i, nm in enumerate(names) if nm.startswith(("call_", "put_"))]
        scale_applied = 1.0
        if opt_idx:
            floor_default = float(getattr(self, "min_contracts", 0.01))
            # Use defaults here; the rounder will still apply per-asset overrides.
            max_abs_w = max(abs(float(w_trade[i])) for i in opt_idx) if opt_idx else 0.0
            max_abs_val = max(abs(float(w_trade[i]) * float(p_vec[i])) for i in opt_idx) if opt_idx else 0.0
            need_floor = (max_abs_w > 0.0) and (max_abs_w < floor_default - 1e-12)
            need_value = (max_abs_val > 0.0) and (max_abs_val < float(min_notional_local) - 1e-12)
            if need_floor or need_value:
                s_floor = (floor_default / max_abs_w) if max_abs_w > 0.0 else float("inf")
                s_value = (float(min_notional_local) / max_abs_val) if max_abs_val > 0.0 else float("inf")
                scale_applied = max(1.0, s_floor, s_value) * (1.0 + 1e-6)  # small epsilon to clear thresholds
                w_trade *= scale_applied
                cost_scaled *= scale_applied
                try:
                    self.logger.debug(
                        f"[_sparse_variance_legs] scale_up_to_tradeable "
                        f"max|w|={max_abs_w:.6g} floor={floor_default:.6g} "
                        f"max|w|*price={max_abs_val:.6g} min_notional_local={min_notional_local:.6g} "
                        f"alpha={scale_applied:.3f}"
                    )
                except Exception:
                    pass
        # Guard: ensure at least min_legs will survive rounding
        min_legs = int(getattr(self, "min_legs", 8) or 8)
        if opt_idx and len(opt_idx) >= min_legs:
            abs_w = sorted([abs(float(w_trade[i])) for i in opt_idx], reverse=True)
            abs_val = sorted([abs(float(w_trade[i]) * float(p_vec[i])) for i in opt_idx], reverse=True)
            kth_w = abs_w[min_legs - 1] if len(abs_w) >= min_legs else 0.0
            kth_val = abs_val[min_legs - 1] if len(abs_val) >= min_legs else 0.0
            s_floor_k = (floor_default / kth_w) if kth_w > 0.0 else float("inf")
            s_val_k = (float(min_notional_local) / kth_val) if kth_val > 0.0 else float("inf")
            extra_scale = max(1.0, s_floor_k, s_val_k)
            if extra_scale > 1.0 + 1e-12:
                scale_applied *= extra_scale
                w_trade *= extra_scale
                cost_scaled *= extra_scale
                self.logger.debug(f"[_sparse_variance_legs] scale_extra_for_min_legs: k={min_legs} alpha={extra_scale:.3f}")
        try:
            self.logger.debug(f"[_sparse_variance_legs] pre_round: cost_scaled={cost_scaled:.6f} min_notional={self.min_notional} floor=0.01 step=0.01")
        except Exception:
            self.logger.debug("[_sparse_variance_legs] pre_round: cost_scaled=?, min_notional=?, floor=0.01, step=0.01")

        # Round to exchange constraints and min notional.
        # IMPORTANT: our option prices (p_vec) are in underlying units on venues like Lyra/Deribit.
        # min_notional is configured in USD, so convert it to "price units" before calling the rounder.
        F = float(portfolio.get("forward") or 0.0)  # forward/spot in USD
        quote_in_underlying = bool(portfolio.get("quote_in_underlying", True))
        min_notional_local = float(self.min_notional)
        if quote_in_underlying and F > 0.0:
            min_notional_local = float(self.min_notional) / F
        try:
            self.logger.debug(
                f"[_sparse_variance_legs] using min_notional_local={min_notional_local:.8f} "
                f"(price units) from USD_floor={self.min_notional}"
            )
        except Exception:
            pass

        w_round = round_options_and_repair_budget(
            names, w_trade, p_vec,
            underlyings=getattr(self, "basis_underlyings", None),
            asset_resolver=getattr(self, "asset_resolver", None),
            step_by_asset=self.contract_increment_by_asset,
            floor_by_asset=self.min_contracts_by_asset,
            default_step=float(self.contract_increment),
            default_floor=float(self.min_contracts),
            min_notional=float(min_notional_local),
            bond_name=str(self.bond_basis_name),
        )

        # Post-round diagnostics; surface common culling reason clearly
        nonzero_legs = [(nm, float(w)) for nm, w in zip(names, w_round)
                        if (nm.startswith("call_") or nm.startswith("put_")) and abs(float(w)) > 0.0]
        try:
            self.logger.debug(f"[_sparse_variance_legs] post_round: legs={len(nonzero_legs)} examples={nonzero_legs[:4]}")
        except Exception:
            self.logger.debug(f"[_sparse_variance_legs] post_round: legs={len(nonzero_legs)}")
        if not nonzero_legs:
            self.logger.debug("[_sparse_variance_legs] culled_by=rounding hint=budget_unit≈0 or per_leg_cost<min_notional")

        # Refit base weights (e.g., bond/forward) to preserve budget after rounding
        try:
            w_final = refit_bases_given_fixed_options(
                X, y, p_vec, names, w_round,
                l2=float(self.regularization),
                budget=cost_scaled,
                base_names=tuple(self.always_on_bases),
            )
        except Exception:
            w_final = w_round
        w_round = w_final
        # Deterministic quantization of option weights using Decimal to avoid float drift
        step = float(getattr(self, "contract_increment", 0.01) or 0.01)
        w_q = []
        for nm, w in zip(names, w_round):
            if nm.startswith("call_") or nm.startswith("put_"):
                w_q.append(round_notional(float(w), step=step, mode="half_away_from_zero"))
            else:
                w_q.append(float(w))
        w_round = np.array(w_q, float)

        # Build legs from rounded/refit weights
        legs = []
        for nm, w in zip(names, w_round):
            if not (nm.startswith("call_") or nm.startswith("put_")):
                continue
            if abs(float(w)) <= 0.0:
                continue
            # translate nm into leg dict (action, type, strike, expiry, contracts, etc.)
            leg = self._name_to_leg(nm, float(w), portfolio)  # assuming you already have a helper
            if leg:
                legs.append(leg)

        diagnostics = {
            "sigma_used": float(sigma_for_states),
            "mse_unit": float(np.mean((X @ w_unit - y) ** 2)),
            "budget_input": float(self.cost_budget),
            "unit_budget": float(unit_budget),
            "scale_applied": float(scale_applied),
            "min_notional_local": float(min_notional_local),
            "achieved_var_notional": float(var_notional) * float(scale_applied),
            "cost_unit": float(p_vec @ w_unit),
            "cost_scaled_pre_round": float(cost_scaled),
            "legs_post_round": int(len([1 for nm, w in zip(names, w_round)
                                        if (nm.startswith('call_') or nm.startswith('put_')) and abs(float(w)) > 0])),
            "total_candidates": len(candidate_idx),
            "total_evaluated": int(len(portfolio.get('options', []))),
        }
        # Validate legs: (a) no inverted put verticals; (b) two-sided quotes if required
        def _has_inverted_put_vertical(legs: List[Dict]) -> bool:
            puts = [l for l in legs if (l.get("type") == "PUT" and l.get("expiry"))]
            # Build map by expiry
            by_exp = {}
            for l in puts:
                by_exp.setdefault(l['expiry'], []).append(l)
            for exp, arr in by_exp.items():
                buys  = [l for l in arr if (l.get("action") or "").upper() == "BUY"]
                sells = [l for l in arr if (l.get("action") or "").upper() == "SELL"]
                for b in buys:
                    for s in sells:
                        if float(b.get("strike", 0.0)) < float(s.get("strike", 0.0)):
                            return True
            return False

        if _has_inverted_put_vertical(legs):
            self.logger.warning("[variance sparse] [REJECT] inverted put vertical detected")
            return [], {"culled_by": "inverted_put_vertical"}

        if getattr(self, "require_live_quotes", True):
            for l in legs:
                bid = float(l.get("bid") or 0.0)
                ask = float(l.get("ask") or 0.0)
                # Require two-sided quotes for any leg to be tradable
                if (not (bid > 0.0 and ask > 0.0)) or (l.get("has_live_quotes") is False) or (l.get("skip_for_execution") is True):
                    self.logger.warning("[variance sparse] [REJECT] missing live quotes for %s %s@%s bid=%.4f ask=%.4f", l.get("type"), l.get("strike"), l.get("expiry"), bid, ask)
                    return [], {"culled_by": "missing_live_quotes"}

        return legs, diagnostics
    
    def _name_to_leg(self, nm: str, w: float, portfolio: Dict) -> Optional[Dict]:
        """Convert a name like 'call_95000' and weight to a leg dictionary."""
        if not (nm.startswith("call_") or nm.startswith("put_")):
            return None
            
        if abs(float(w)) <= 0.0:
            return None
            
        # Extract type and strike from name
        is_call = nm.startswith("call_")
        K = float(nm.split("_", 1)[1])
        
        # Find the option data in portfolio
        options_list = portfolio.get('options', [])
        if is_call:
            options_list = portfolio.get('calls', options_list)
        else:
            options_list = portfolio.get('puts', options_list)
            
        opt_data = None
        for opt in options_list:
            if abs(float(opt.get('strike', 0)) - K) < 1e-6:
                opt_data = opt
                break
                
        if not opt_data:
            return None
            
        return {
            "action": "BUY" if w > 0 else "SELL",
            "type": "CALL" if is_call else "PUT",
            "strike": K,
            "expiry": opt_data.get('expiry_date', ''),
            "contracts": float(abs(w)),
            "fee_bps": float(self.slippage_bps),
            "slippage_bps": float(self.slippage_bps),
            "bid": float(opt_data.get("bid", 0.0) or 0.0),
            "ask": float(opt_data.get("ask", 0.0) or 0.0),
            "mid": float(opt_data.get("mid", 0.0) or 0.0),
            "has_live_quotes": bool(opt_data.get("has_live_prices", False)),
            "instrument_id": f"Lyra:{'C' if is_call else 'P'}:{int(K)}",
            "skip_for_execution": bool(opt_data.get("skip_for_execution", False)),
            "venue": opt_data.get("venue", "Lyra"),
            "symbol": opt_data.get("currency", ""),
        }

    # ------------------------------------------------------------------
    # NEW: Convert replication weights -> concrete hedge legs with qty/quotes
    # ------------------------------------------------------------------
    def _build_variance_replication_legs(self, portfolio: dict, var_exposure: float, position_size: float) -> List[Dict]:
        """
        Convert replication weights into a list of option positions with quantities and quotes.
        Assumes:
          portfolio['replication_weights'] : { strike(float) -> weight(float) }
          portfolio['k0']                  : float
          portfolio['options']             : list of option quote dicts with fields:
                                             'type' ('call'|'put'), 'strike', 'bid','ask','iv' (optional)
        Quantity convention:
          • Long variance (var_exposure > 0): long OTM options with given weights.
          • Short variance (var_exposure < 0): short OTM options with given weights.
        """
        weights = {float(k): float(v) for k, v in (portfolio.get('replication_weights') or {}).items()}
        K0 = float(portfolio.get('k0', 0.0) or 0.0)
        direction = 1.0 if float(var_exposure) > 0 else -1.0

        # Index best quotes by (type, strike)
        by_key: Dict = {}
        for row in (portfolio.get('options') or []):
            typ = (row.get('type') or row.get('option_type') or '').strip().lower()
            K = float(row.get('strike', 0.0) or 0.0)
            key = (typ, K)
            bid = float(row.get('bid', 0.0) or 0.0)
            ask = float(row.get('ask', 0.0) or 0.0)
            iv  = float(row.get('iv', 0.0)  or 0.0)
            prev = by_key.get(key)
            if prev is None:
                by_key[key] = {'type': typ, 'strike': K, 'bid': bid, 'ask': ask, 'iv': iv}
            else:
                prev['bid'] = max(prev['bid'], bid)
                prev['ask'] = (ask if (prev['ask'] == 0.0 or (ask > 0 and ask < prev['ask'])) else prev['ask'])
                prev['iv']  = max(prev['iv'], iv)

        legs: List[Dict] = []
        for K, w in sorted(weights.items()):
            if w <= 0.0:
                continue
            qty = abs(float(var_exposure)) * float(position_size) * float(w)
            if K < K0:
                ref = by_key.get(('put', K))
                if ref:
                    legs.append({**ref, 'quantity': direction * qty})
            elif K > K0:
                ref = by_key.get(('call', K))
                if ref:
                    legs.append({**ref, 'quantity': direction * qty})
            else:
                for typ in ('put', 'call'):
                    ref = by_key.get((typ, K))
                    if ref:
                        legs.append({**ref, 'quantity': direction * (0.5 * qty)})
        return legs

    # ------------------------------------------------------------------
    # NEW: Compute entry cost AND return legs (entry uses bid/ask as per side)
    # ------------------------------------------------------------------
    def _compute_variance_hedge_entry_cost_and_legs(self, portfolio: dict, var_exposure: float, position_size: float) -> tuple:
        """
        Returns: (hedge_entry_cost: float, hedge_positions: List[Dict])
        Entry conventions:
          • Buying longs at ask (cash outflow),
          • Selling shorts at bid (cash inflow).
        """
        legs = self._build_variance_replication_legs(portfolio, var_exposure, position_size)
        total_cash = 0.0
        for leg in legs:
            q   = float(leg['quantity'])
            bid = float(leg.get('bid', 0.0) or 0.0)
            ask = float(leg.get('ask', 0.0) or 0.0)
            mid = 0.5*(bid+ask) if (bid > 0.0 and ask > 0.0) else max(bid, ask)
            if q > 0:
                px = ask if ask > 0.0 else mid  # buy long at ask
                total_cash -= q * px
            else:
                px = bid if bid > 0.0 else mid  # sell short at bid
                total_cash -= q * px  # q is negative: adds positive cash
        hedge_entry_cost = -total_cash  # positive number = cost
        return float(hedge_entry_cost), legs

    # --- Deterministic PM‑digital proxy helpers (strike‑ladder based) ---
    # Use the immediate bracketing strikes around the PM strike to approximate e^{-rT}·Q(S_T≥K0)
    # via a call‑spread finite difference; no configurable width.
    def _bracket_strikes(self, strikes: Sequence[float], k0: float) -> Optional[tuple[float, float]]:
        if not strikes:
            return None
        s = sorted(set(float(k) for k in strikes))
        # find first strike >= k0
        j = next((i for i, K in enumerate(s) if K >= k0), None)
        if j is None:
            return None
        if s[j] == k0:
            if j + 1 < len(s):
                return (s[j], s[j+1])
            return None
        if j - 1 >= 0:
            return (s[j-1], s[j])
        return None

    def _pm_digital_proxy_price_at_k0(self,
                                      calls_by_strike: dict[float, dict],
                                      k0: float,
                                      r: float,
                                      T_years: float) -> Optional[float]:
        """
        Approximate PV of a $1 digital call at K0 using the vertical [K-, K+]:
            e^{-rT}·Q(S_T≥K0)  ≈  (C(K-) - C(K+)) / (K+ - K-)
        using *mid* call prices (same numeraire as hedge costs).
        """
        Ks = list(calls_by_strike.keys())
        br = self._bracket_strikes(Ks, k0)
        if br is None:
            return None
        k_lo, k_hi = br
        c_lo = (calls_by_strike.get(k_lo) or {}).get("mid")
        c_hi = (calls_by_strike.get(k_hi) or {}).get("mid")
        if c_lo is None or c_hi is None:
            return None
        dk = float(k_hi) - float(k_lo)
        if dk <= 0:
            return None
        # Finite difference; prices are already PVs, so discounting is consistent with hedge PV units.
        pv_yes = (float(c_lo) - float(c_hi)) / dk
        # Clip tiny negatives from noise
        return pv_yes if pv_yes >= -1e-6 else None

    # ---------- Helpers: PM strike, PM YES price, digital bounds ----------
    def _infer_pm_strike(self, contract, opt_chain):
        """
        Use the PM market's explicit threshold if available; otherwise try to parse.
        Fall back to the variance split strike only if pm_gate_use_pm_strike=False.
        """
        if self.pm_gate_use_pm_strike:
            for k in ("pm_strike", "threshold", "strike", "level", "target", "strike_price"):
                v = contract.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        pass
            # last-resort: regex from question/title like "≥ 3000" / "above 3,000"
            q = (contract.get("question") or contract.get("title") or "")
            import re
            m = re.search(r"([<>]=?)?\s*\$?\s*([0-9][0-9,]*\.?[0-9]*)", q)
            if m:
                try:
                    return float(m.group(2).replace(",", ""))
                except Exception:
                    pass
        # fallback: use variance K0 if allowed
        return opt_chain.get("K0") if isinstance(opt_chain, dict) else getattr(opt_chain, "K0", None)

    def _pm_yes_price(self, contract):
        # prefer mid, then best-ask/bid midpoint, then stored 'yes_price'
        for k in ("yes_price_mid", "yes_mid", "yes_price"):
            v = contract.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        bb = contract.get("yes_best_bid")
        ba = contract.get("yes_best_ask")
        try:
            if bb and ba:
                return 0.5 * (float(bb) + float(ba))
        except Exception:
            pass
        return None

    def _digital_bounds_at_strike(self, K, opt_chain, *, use_tradable=True, require_two_sided=False, max_rel_width=0.02):
        """
        Compute lower/upper bounds for the digital call at strike K using the same expiry as opt_chain.
        Prefer tradable one-sided call-spread bounds (bid/ask). Fall back to mid-price finite differences.
        """
        # opt_chain is the portfolio dictionary with 'options' or 'calls'/'puts'
        calls = opt_chain.get("calls") or []
        options = opt_chain.get("options") or []
        # If no calls but we have options, filter for calls
        if not calls and options:
            calls = [o for o in options if (o.get("type") or o.get("option_type") or "").lower().startswith("c")]
        if not calls:
            return {"lower": None, "upper": None}

        # find nearest lower and upper strikes
        strikes = sorted([float(c["strike"]) for c in calls if c.get("strike") is not None])
        import bisect
        i = bisect.bisect_left(strikes, K)
        if i == 0 or i == len(strikes):
            return {"lower": None, "upper": None}
        Kl, Ku = strikes[i-1], strikes[i]
        # sanity on width
        if max_rel_width and (Ku - Kl) / max(K, 1e-6) > max_rel_width:
            self.logger.debug(f"[digital-bounds] neighbor gap too wide: [{Kl},{Ku}] around K={K}")
            return {"lower": None, "upper": None}

        def _pick(strike):
            for c in calls:
                if float(c["strike"]) == strike:
                    return c
            return None
        Cl, Ck, Cu = _pick(Kl), _pick(K), _pick(Ku)
        if Ck is None or Cl is None or Cu is None:
            return {"lower": None, "upper": None}

        def _num(x):
            try: return float(x) if x is not None else None
            except: return None

        # tradable bounds via one-sided call-spreads (ask for longs, bid for shorts)
        lb = ub = None
        if use_tradable:
            askK  = _num(Ck.get("ask"));   bidK  = _num(Ck.get("bid"))
            askKu = _num(Cu.get("ask"));   bidKu = _num(Cu.get("bid"))
            askKl = _num(Cl.get("ask"));   bidKl = _num(Cl.get("bid"))
            # Lower: (long C(K) @ ask) - (short C(Ku) @ bid) over width (Ku-K)
            if askK is not None and bidKu is not None:
                lb = max(0.0, min(1.0, (askK - bidKu) / (Ku - K)))
            # Upper: (long C(Kl) @ ask) - (short C(K) @ bid) over width (K-Kl)
            if askKl is not None and bidK is not None:
                ub = max(0.0, min(1.0, (askKl - bidK) / (K - Kl)))
            if require_two_sided and (lb is None or ub is None):
                return {"lower": None, "upper": None}

        # fallback to mid-price finite differences if needed
        if lb is None or ub is None:
            mid = lambda c: _num(c.get("mid")) or ( ((_num(c.get("bid")) or 0.0) + (_num(c.get("ask")) or 0.0)) / 2.0 )
            mKl, mK, mKu = mid(Cl), mid(Ck), mid(Cu)
            if (lb is None) and (mK is not None and mKu is not None):
                lb = max(0.0, min(1.0, -(mKu - mK) / (Ku - K)))
            if (ub is None) and (mKl is not None and mK is not None):
                ub = max(0.0, min(1.0, -(mK - mKl) / (K - Kl)))

        return {"lower": lb, "upper": ub}


# --- Synthetic BS surface sanity check for the replication ---
def _bs_mid(S, K, r, sigma, T, kind):
    import math
    if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
        return max(0.0, (S-K) if kind=='call' else (K-S))
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    from math import erf, sqrt
    N = lambda x: 0.5*(1.0 + erf(x/sqrt(2.0)))
    if kind == 'call':
        return S*N(d1) - K*math.exp(-r*T)*N(d2)
    else:
        return K*math.exp(-r*T)*N(-d2) - S*N(-d1)

def variance_replication_self_test():
    S0, r, vol, T = 100.0, 0.00, 0.22, 0.5
    def chain(n):
        import numpy as np
        Ks = np.linspace(40.0, 160.0, n)
        out = []
        for K in Ks:
            c = _bs_mid(S0, K, r, vol, T, 'call')
            p = _bs_mid(S0, K, r, vol, T, 'put')
            out.append({'strike': float(K), 'type': 'call', 'bid': c, 'ask': c})
            out.append({'strike': float(K), 'type': 'put', 'bid': p, 'ask': p})
        return out

    vs = VarianceSwapStrategy(risk_free_rate=r)
    errs = []
    for n in (15, 25, 41, 61, 101, 161):
        port = vs._build_variance_swap_portfolio(chain(n), S0, T)
        if not port: 
            raise RuntimeError("portfolio build failed")
        params = vs._calculate_variance_swap_params(port, S0, T)
        errs.append(abs(params['strike_variance'] - vol*vol))
    print("max abs error:", max(errs))
    print("errors by n:", {n: err for n, err in zip((15, 25, 41, 61, 101, 161), errs)})
    # With discrete approximation, 0.5% error is reasonable for coarse grids
    assert max(errs) < 5e-3, f"Variance replication didn't converge (max err={max(errs):.3e})"