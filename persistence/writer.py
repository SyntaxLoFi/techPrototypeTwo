from __future__ import annotations
from typing import Protocol, Iterable, Dict, Any, Optional, Mapping, List
import logging
import json
import os
from datetime import datetime

# Numpy-safe encoder (no hard dependency on numpy)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore
        if np is not None:
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

class Writer(Protocol):
    def save(self, opportunities: Iterable[Dict[str, Any]]) -> None: ...

class DefaultWriter:
    """
    Serialization + file output for opportunities.
    Preserves the legacy JSON shape enough for existing tests & dashboards.
    """
    def __init__(self, config: Any = None, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    # Minimal Writer interface used by Orchestrator
    def save(self, opportunities: Iterable[Dict[str, Any]]) -> None:
        dummy = type('ScannerShim', (), {'logger': self.logger, 'scanners': {}, 'opportunities': list(opportunities)})()
        self.save_from_scanner(dummy, filename=None, detailed_mode=True)

    # Back-compat helper used by tests that still call ArbitrageScanner.save_opportunities(...)
    def save_from_scanner(self, self_obj, filename: Optional[str] = None, detailed_mode: bool = True):
        opps = list(self_obj.opportunities or [])
        if not opps:
            self.logger.info("No opportunities to save.")
            return

        os.makedirs('results', exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if filename:
            path = os.path.join('results', filename)
        else:
            path = os.path.join('results', f"detailed_opportunities_{ts}.json")

        # Build serializable payload
        data: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "total_opportunities": len(opps),
            "opportunities": [],
        }

        for opp in opps:
            metrics = (opp.get("metrics") or {})
            probs   = (opp.get("probabilities") or {})
            dni_raw = metrics.get("distance_to_no_arb", metrics.get("dni"))

            try:
                dni = float(dni_raw) if dni_raw is not None else 0.0
            except Exception:
                dni = 0.0

            pm_block: Dict[str, Any] = {
                "prob_of_profit": float(metrics.get("prob_of_profit", 0.0) or 0.0),
                "expected_value": float(metrics.get("expected_value", 0.0) or 0.0),
                "adjusted_ev": float(metrics.get("adjusted_ev", 0.0) or 0.0),
                "edge_per_downside": float(metrics.get("edge_per_downside", 0.0) or 0.0),
                "distance_to_no_arb": dni,
            }
            if "is_true_arbitrage" in metrics:
                pm_block["is_true_arbitrage"] = bool(metrics.get("is_true_arbitrage"))

            row = {
                "rank": opp.get("rank", 0),
                "quality_tier": opp.get("quality_tier", "UNKNOWN"),
                "currency": opp.get("currency", opp.get("symbol", "Unknown")),
                "hedge_type": opp.get("hedge_type", opp.get("strategy_type", "Unknown")),
                "strategy": opp.get("strategy", opp.get("strategy_name", "Unknown")),
                "max_profit": float(opp.get("max_profit", 0.0) or 0.0),
                "max_loss": float(opp.get("max_loss", 0.0) or 0.0),
                "probabilities": probs,
                "probability_metrics": pm_block,
                "polymarket": (opp.get("polymarket_contract") or opp.get("polymarket") or {}),
            }

            # Attach concise options details when present
            if opp.get("required_options") or opp.get("detailed_strategy"):
                row["detailed_strategy"] = {
                    **(opp.get("detailed_strategy") or {}),
                    "required_options": opp.get("required_options", []),
                    "greeks": (opp.get("detailed_strategy") or {}).get("greeks", {}),
                }

            data["opportunities"].append(row)

        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2, cls=NumpyEncoder)
        os.replace(tmp, path)
        self.logger.info("Saved %d opportunities → %s", len(opps), path)

        # Also emit a compact ranked companion for quick viewing when detailed_mode is False
        if not detailed_mode:
            ranked_min = {
                "timestamp": data["timestamp"],
                "total_opportunities": data["total_opportunities"],
                "opportunities": [
                    {
                        "currency": o.get("currency"),
                        "strategy": o.get("strategy"),
                        "probability_metrics": o.get("probability_metrics", {}),
                        "polymarket": o.get("polymarket", {}),
                        "rank": idx + 1,
                    }
                    for idx, o in enumerate(data["opportunities"])
                ],
            }
            small_path = os.path.join("results", f"ranked_opportunities_{ts}.json")
            with open(small_path, "w") as fh:
                json.dump(ranked_min, fh, indent=2, cls=NumpyEncoder)
            self.logger.info("Saved ranked summary → %s", small_path)
