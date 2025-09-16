from __future__ import annotations

from typing import Protocol, Mapping, Iterable, Any, Dict, Optional, List
import logging
from utils.debug_recorder import get_recorder  # type: ignore
from utils.log_gate import per_currency_snapshot_enabled  # type: ignore
from utils.step_debugger import get_step_debugger  # type: ignore

# Small interfaces to make dependencies injectable
class DataRefresher(Protocol):
    async def fetch_all(self) -> Mapping[str, Any]: ...

class HedgeBuilder(Protocol):
    def build(self, market_snapshot: Mapping[str, Any]) -> Iterable[Dict[str, Any]]: ...

class Writer(Protocol):
    def save(self, opportunities: Iterable[Dict[str, Any]]) -> None: ...

class Orchestrator:
    """Coordinates data refresh -> hedge construction -> persistence.

    This class is intentionally tiny. Real logic lives behind the injected interfaces.
    """
    def __init__(self, refresher: DataRefresher, hedge: HedgeBuilder, writer: Writer,
                 config: Any, logger: logging.Logger) -> None:
        self.refresher = refresher
        self.hedge = hedge
        self.writer = writer
        self.config = config
        self.logger = logger

    def _save_unfiltered_opportunities(self, opportunities: List[Dict[str, Any]]) -> None:
        """Save unfiltered opportunities to a separate file"""
        import json
        import os
        from datetime import datetime
        from persistence.writer import NumpyEncoder
        
        os.makedirs('results', exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join('results', f'unfiltered_opportunities_{ts}.json')
        
        data = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "total_opportunities": len(opportunities),
            "opportunities": opportunities,
            "note": "These are unfiltered opportunities before ExpectedValueFilter"
        }
        
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2, cls=NumpyEncoder)
        os.replace(tmp, path)
        self.logger.info("Saved %d unfiltered opportunities → %s", len(opportunities), path)
    
    async def run_once(self) -> None:
        debugger = get_step_debugger()
        
        snapshot = await self.refresher.fetch_all()
        
        # Checkpoint: Market snapshot after refresh
        debugger.checkpoint("market_snapshot_refreshed", snapshot, 
                          {"currencies": list(snapshot.keys()),
                           "has_spot_prices": {k: v.get("current_spot") is not None for k, v in snapshot.items()}})
        # One-line per-currency summary (INFO) to avoid log bloat
        try:
            for ccy, s in (snapshot or {}).items():
                try:
                    # Accept both 'contracts' and legacy 'polymarket_contracts'
                    contracts = s.get("contracts") or s.get("polymarket_contracts") or []
                    oc = s.get("options_collector")
                    n_opts = len(oc.get_all_options() or []) if oc else 0
                    has_perps = bool(s.get("perps_collector"))
                    if per_currency_snapshot_enabled():
                        self.logger.info("Snapshot[%s]: PM markets=%d, options=%d, perps=%s",
                                         ccy, len(contracts), n_opts, "yes" if has_perps else "no")
                except Exception:
                    continue
        except Exception:
            pass
        # Persist snapshot once on debug runs
        try:
            get_recorder(self.config).dump_json("snapshot/scanners.json", snapshot, category="snapshot")
        except Exception:
            pass
        
        # Checkpoint: Before hedge building
        debugger.checkpoint("pre_hedge_build", snapshot,
                          {"total_contracts": sum(len(s.get("contracts", [])) for s in snapshot.values())})
        
        opportunities = list(self.hedge.build(snapshot))
        
        # Checkpoint: After hedge building
        debugger.checkpoint("post_hedge_build", opportunities,
                          {"count": len(opportunities),
                           "currencies": list(set(o.get("currency") for o in opportunities))})
        # Save and optionally dump the final opps
        try:
            get_recorder(self.config).dump_json("snapshot/opportunities.json", opportunities, category="snapshot")
        except Exception:
            pass
        if not opportunities:
            self.logger.warning("Pipeline produced 0 opportunities – check upstream counts above.")
            # Checkpoint: Final empty result
            debugger.checkpoint("final_opportunities_empty", opportunities, {"reason": "all_filtered_out"})
        else:
            # Checkpoint: Final opportunities
            debugger.checkpoint("final_opportunities", opportunities, {"count": len(opportunities)})
        
        self.writer.save(opportunities)
        
        # Save unfiltered opportunities if configured and available
        try:
            if self.config and self.config.data.save_unfiltered_opportunities:
                unfiltered_opps = getattr(self.hedge, 'unfiltered_opportunities', [])
                if unfiltered_opps:
                    self.logger.info("Saving %d unfiltered opportunities", len(unfiltered_opps))
                    self._save_unfiltered_opportunities(unfiltered_opps)
        except Exception as e:
            self.logger.debug("Failed to save unfiltered opportunities: %s", e)
