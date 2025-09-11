# config_loader.py (production-ready revision)
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from pydantic import ValidationError

# NOTE: this module expects a `config_schema.py` that defines `AppConfig`
from config_schema import AppConfig


# --------------------------------------------------------------------------------------
# Paths & discovery
# --------------------------------------------------------------------------------------

# Environment variable names that can override the config root directory.
_CONFIG_ROOT_ENV_VARS: Tuple[str, ...] = ("APP_CONFIG_ROOT", "CONFIG_ROOT")

# Default relative locations (under the resolved config root).
_BASELINE_FILENAME = "baseline.yaml"
_BASELINE_LOCK_FILENAME = "baseline.lock"
_SCHEMA_FILENAME = "schema.json"

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _sha256_file(p: Path) -> str:
    return _sha256_bytes(p.read_bytes())

def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _resolve_config_root(explicit_root: Optional[Path] = None) -> Path:
    """
    Resolve the configuration *root* directory using the following priority:
      1) explicit_root argument (if provided)
      2) env var APP_CONFIG_ROOT or CONFIG_ROOT
      3) a 'config/' folder found by walking up from CWD
      4) a 'config/' folder next to this file
      5) fallback to '<CWD>/config'
    """
    if explicit_root is not None:
        return Path(explicit_root).expanduser().resolve()

    for name in _CONFIG_ROOT_ENV_VARS:
        val = os.getenv(name)
        if val:
            return Path(val).expanduser().resolve()

    cwd = Path.cwd().resolve()
    for base in (cwd,) + tuple(cwd.parents):
        candidate = base / "config"
        if candidate.exists() and candidate.is_dir():
            return candidate

    here = Path(__file__).resolve().parent
    candidate = here / "config"
    if candidate.exists() and candidate.is_dir():
        return candidate

    return Path.cwd().resolve() / "config"


def _paths_for(root: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """
    Compute key paths under the given config root:
      - baseline.yaml
      - baseline.lock
      - schema.json
    """
    root_dir = _resolve_config_root(root)
    baseline = root_dir / _BASELINE_FILENAME
    lock = root_dir / _BASELINE_LOCK_FILENAME
    schema = root_dir / _SCHEMA_FILENAME
    return baseline, lock, schema


# --------------------------------------------------------------------------------------
# Env overrides
# --------------------------------------------------------------------------------------

def _apply_env_overrides(effective: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal, explicit env override map to preserve clarity.
    Add entries as needed; pydantic will coerce types.
    """
    mapping: Dict[str, Tuple[str, ...]] = {
        "LOG_LEVEL": ("logging", "level"),
        "LOG_TO_FILE": ("logging", "to_file"),
        "LOG_JSON": ("logging", "json"),
        "LOG_FILENAME": ("logging", "filename"),
        # Debugging
        "DEBUG": ("debug", "enabled"),
        "DEBUG_DUMP_DIR": ("debug", "dump_dir"),
        "DEBUG_CAPTURE_POLYMARKET": ("debug", "capture", "polymarket"),
        "DEBUG_CAPTURE_OPTIONS": ("debug", "capture", "options"),
        "DEBUG_CAPTURE_PERPS": ("debug", "capture", "perps"),
        "DEBUG_CAPTURE_SNAPSHOT": ("debug", "capture", "snapshot"),
        "DEBUG_REASON_CODES": ("debug", "log", "reason_codes"),
        "DEBUG_REASON_LEVEL": ("debug", "log", "reason_codes_level"),
        "DEBUG_REASON_MAX_LINES": ("debug", "log", "reason_codes_max_lines"),
        "DEBUG_PER_CCY_SNAPSHOT": ("debug", "log", "per_currency_snapshot"),
        "DEBUG_EV_SUMMARY_INFO": ("debug", "log", "ev_summary_info"),
        "RISK_FREE_RATE": ("execution", "risk_free_rate"),
        "MIN_RISK_REWARD_RATIO": ("execution", "min_risk_reward_ratio"),
        "MIN_KELLY_FRACTION": ("execution", "min_kelly_fraction"),
        "KELLY_FRACTION_CAP": ("execution", "kelly_fraction_cap"),
        "HEDGE_COMBINATION_MODE": ("execution", "hedge_combination_mode"),
        "MIN_SPREAD_THRESHOLD": ("execution", "min_spread_threshold"),
        "MAX_SPREAD_ALLOWED": ("execution", "max_spread_allowed"),
        "DEFAULT_SPREAD_WIDTH_PCT": ("execution", "default_spread_width_pct"),
        "MIN_SPREAD_WIDTH": ("execution", "min_spread_width"),
        "MAX_SPREAD_COUNT": ("execution", "max_spread_count"),
        "SETTLE_SPREAD_AT_PM_EXPIRY": ("execution", "settle_spread_at_pm_expiry"),
        "OPTIONS_UNWIND_MODEL": ("execution", "options_unwind_model"),
        "POST_EVENT_VOL_DROP": ("execution", "post_event_vol_drop"),
        "DEFAULT_POSITION_SIZE": ("execution", "default_position_size"),
        "POSITION_BASE_UNIT": ("execution", "default_position_size"),  # legacy alias

        "MAX_POSITION_SIZE": ("execution", "max_position_size"),
        "MIN_POSITION_SIZE": ("execution", "min_position_size"),
        "MIN_EXPECTED_VALUE": ("execution", "min_expected_value"),
        "MIN_SHARPE_RATIO": ("execution", "min_sharpe_ratio"),
        "MAX_ACCEPTABLE_LOSS": ("execution", "max_acceptable_loss"),
        "TRANSACTION_COST_RATE": ("execution", "transaction_cost_rate"),
        "DAYS_PER_YEAR": ("time", "days_per_year"),
        "MIN_HOURS_TO_EXPIRY": ("time", "min_hours_to_expiry"),
        "MAX_DAYS_TO_EXPIRY": ("time", "max_days_to_expiry"),
        "LYRA_API_BASE": ("lyra", "api_base"),
        "LYRA_WS_URI": ("lyra", "ws_uri"),
        "SAVE_DETAILED_DATA": ("data", "save_detailed_data"),
        "SAVE_UNFILTERED_OPPORTUNITIES": ("data", "save_unfiltered_opportunities"),
        "SCAN_INTERVAL_SECONDS": ("data", "scan_interval_seconds"),
        "DATA_RETENTION_MINUTES": ("data", "data_retention_minutes"),
        "ORDERBOOK_DEPTH": ("data", "orderbook_depth"),
        # Ranking parameters
        "PROBABILITY_BLEND_MODE": ("ranking", "blend_mode"),
        "PROBABILITY_BLEND_WEIGHT": ("ranking", "fixed_weight"),
        "VEGA_PENALTY_LAMBDA": ("ranking", "vega_penalty_lambda"),
        "FUNDING_PENALTY_KAPPA": ("ranking", "funding_penalty_kappa"),
        "LIQUIDITY_PENALTY_THETA": ("ranking", "liquidity_penalty_theta"),
        "TRUE_ARB_DNI_THRESHOLD": ("ranking", "true_arbitrage_dni_threshold"),
        "NEAR_ARB_DNI_THRESHOLD": ("ranking", "near_arbitrage_dni_threshold"),
        "NEAR_ARB_PROB_THRESHOLD": ("ranking", "near_arbitrage_prob_threshold"),
        "HIGH_PROB_THRESHOLD": ("ranking", "high_probability_threshold"),
        "MODERATE_PROB_THRESHOLD": ("ranking", "moderate_probability_threshold"),
        "PROBABILITY_CLIP_FLOOR": ("ranking", "probability_clip_floor"),
        "PROBABILITY_CLIP_CEILING": ("ranking", "probability_clip_ceiling"),
        "MIN_CORRELATION_THRESHOLD": ("ranking", "min_correlation_threshold"),
        "COV_REGULARIZATION": ("ranking", "covariance_regularization"),
        # Back-compat synonyms (you already use some of these names in your .env)
        "MARKET_IMPACT_COEFFICIENT": ("ranking", "market_impact_coefficient"),
        "TRUE_ARBITRAGE_DNI_THRESHOLD": ("ranking", "true_arbitrage_dni_threshold"),
        "NEAR_ARBITRAGE_DNI_THRESHOLD": ("ranking", "near_arbitrage_dni_threshold"),
        "NEAR_ARBITRAGE_PROB_THRESHOLD": ("ranking", "near_arbitrage_prob_threshold"),
        "HIGH_PROBABILITY_THRESHOLD": ("ranking", "high_probability_threshold"),
        "COVARIANCE_REGULARIZATION": ("ranking", "covariance_regularization"),
        # Hedging variance (moved from defaults.yaml)
        "VARIANCE_MAX_EXPIRIES_CONSIDERED": ("hedging", "variance", "max_expiries_considered"),
        "VARIANCE_MAX_EXPIRY_GAP_DAYS": ("hedging", "variance", "max_expiry_gap_days"),
        "VARIANCE_REQUIRE_LIVE_QUOTES_FOR_TRADES": ("hedging", "variance", "require_live_quotes_for_trades"),
        "VARIANCE_STRIKE_PROXIMITY_WINDOW": ("hedging", "variance", "strike_proximity_window"),
        "VARIANCE_MIN_QUOTES_PER_EXPIRY": ("hedging", "variance", "min_quotes_per_expiry"),
        "EXPIRY_POLICY": ("hedging", "variance", "expiry_policy"),
        # Execution capture for automated trading (lives in baseline.yaml)
        "CAPTURE_INSTRUMENTS": ("execution", "capture_instruments"),
        # Polymarket parameters
        "POLYMARKET_INCLUDE_DAILIES": ("polymarket", "include_dailies"),
        "POLYMARKET_DAILIES_WINDOW_HOURS": ("polymarket", "dailies_window_hours"),
        "POLYMARKET_FILTER_CLOSED": ("polymarket", "filter_closed"),
        "POLYMARKET_FILTER_ACTIVE": ("polymarket", "filter_active"),
    }

    result: Dict[str, Any] = dict(effective)

    for env_name, path in mapping.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue

        node = result
        for key in path[:-1]:
            node = node.setdefault(key, {})  # type: ignore[assignment]

        # Let pydantic coerce types for everything except booleans
        if env_name in {
            "LOG_TO_FILE", "LOG_JSON", "SETTLE_SPREAD_AT_PM_EXPIRY",
            "SAVE_DETAILED_DATA", "SAVE_UNFILTERED_OPPORTUNITIES",
            "VARIANCE_REQUIRE_LIVE_QUOTES_FOR_TRADES", "CAPTURE_INSTRUMENTS",
            "POLYMARKET_INCLUDE_DAILIES", "POLYMARKET_FILTER_CLOSED", "POLYMARKET_FILTER_ACTIVE",
            "DEBUG", "DEBUG_CAPTURE_POLYMARKET", "DEBUG_CAPTURE_OPTIONS",
            "DEBUG_CAPTURE_PERPS", "DEBUG_CAPTURE_SNAPSHOT",
            "DEBUG_REASON_CODES", "DEBUG_PER_CCY_SNAPSHOT", "DEBUG_EV_SUMMARY_INFO"
        }:
            node[path[-1]] = _parse_bool(raw)
        elif env_name in {"DEBUG_REASON_MAX_LINES"}:
            try:
                node[path[-1]] = int(raw)
            except Exception:
                pass
        else:
            node[path[-1]] = raw

    return result

# --------------------------------------------------------------------------------------
# Load / validate / freeze
# --------------------------------------------------------------------------------------

def load_config(
    config_root: Optional[Path] = None,
    baseline_filename: str = _BASELINE_FILENAME,
    lock_filename: str = _BASELINE_LOCK_FILENAME,
) -> AppConfig:
    """
    Load YAML, apply env overrides, validate, and (if APP_ENV==prod) enforce baseline lock.
    Respects a root-path fallback and env override via APP_CONFIG_ROOT/CONFIG_ROOT.
    """
    baseline_path, lock_path, _ = _paths_for(config_root)
    baseline_path = baseline_path.with_name(baseline_filename)
    lock_path = lock_path.with_name(lock_filename)

    # If baseline doesn't exist, create an empty dict (will use all defaults)
    if baseline_path.exists():
        raw_yaml = baseline_path.read_text(encoding="utf-8")
        raw_data = yaml.safe_load(raw_yaml) or {}
        if not isinstance(raw_data, dict):
            raise ValueError("Top-level YAML must be a mapping/object.")
    else:
        raw_yaml = ""
        raw_data = {}

    # Apply explicit env overrides
    effective_dict = _apply_env_overrides(dict(raw_data))

    try:
        cfg = AppConfig(**effective_dict)
    except ValidationError as ve:
        errs = ve.errors()
        raise SystemExit("Config validation failed:\n" + json.dumps(errs, indent=2))

    # In prod, freeze against a locked baseline SHA to avoid drift
    app_env = os.getenv("APP_ENV", getattr(cfg, "env", None) or "dev")
    if app_env == "prod":
        if not baseline_path.exists():
            raise RuntimeError(f"{baseline_path} required in production mode")
        if not lock_path.exists():
            raise RuntimeError(
                f"{lock_path} missing. Run `python freeze_baseline.py` after approving baseline.yaml."
            )
        expected_sha = lock_path.read_text(encoding="utf-8").strip()
        actual_sha = _sha256_bytes(raw_yaml.encode("utf-8"))
        if actual_sha != expected_sha:
            raise RuntimeError(
                "baseline.yaml changed but lock does not match.\n"
                f"  expected: {expected_sha}\n"
                f"  actual:   {actual_sha}\n"
                "Refuse to start in prod. Re‑freeze after an explicit review."
            )

    return cfg

def export_json_schema(
    path: Optional[Path] = None,
    config_root: Optional[Path] = None,
) -> Path:
    """
    Generate a JSON Schema for external validation / tooling.
    Uses Pydantic v2 `model_json_schema()` if available; falls back to v1 `.schema()`.
    Writes to <config_root>/schema.json by default.
    Returns the written path.
    """
    _, _, default_schema_path = _paths_for(config_root)
    out_path = Path(path) if path is not None else default_schema_path

    # Prefer v2
    if hasattr(AppConfig, "model_json_schema"):
        schema = AppConfig.model_json_schema()  # type: ignore[attr-defined]
    else:
        schema = AppConfig.schema()  # type: ignore[call-arg]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return out_path


# Create a convenience function to get config (module-level cache)
_config_instance: Optional[AppConfig] = None


def get_config(**kwargs: Any) -> AppConfig:
    """Get a cached config instance (call `load_config(**kwargs)` on first use)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config(**kwargs)
    return _config_instance


if __name__ == "__main__":
    # Tiny CLI for convenience: dump schema
    import argparse

    parser = argparse.ArgumentParser(description="Config loader helpers")
    parser.add_argument("--schema", action="store_true", help="Write JSON schema to <config_root>/schema.json")
    parser.add_argument("--root", type=str, default=None, help="Override config root (same as APP_CONFIG_ROOT)")
    parser.add_argument("--out", type=str, default=None, help="Explicit schema.json output path")
    args = parser.parse_args()

    root = Path(args.root).expanduser() if args.root else None
    if args.schema:
        written = export_json_schema(path=Path(args.out) if args.out else None, config_root=root)
        print(f"Wrote schema to: {written}")