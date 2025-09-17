from __future__ import annotations

from datetime import datetime, date
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ... existing imports ...
from pm.contract import PMContract, collect_unique_expiries as pm_collect_unique_expiries
from filters.option_expiry import filter_options_by_expiry
try:
    from core.expiry_window import enumerate_expiries, pm_default_cutoff_for_date_only
except Exception:
    enumerate_expiries = None
    pm_default_cutoff_for_date_only = None


class VarianceSwapStrategy:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("VARIANCE SWAP STRATEGY INITIALIZED")

    def evaluate_opportunities(
        self,
        pm_market: str,
        currency: str,
        options: Sequence[dict],
        pm_days_to_expiry: Optional[float] = None,
        inclusive: bool = True,
        tz: str = "UTC",
        max_expiries_per_contract: int = 5,
    ) -> Dict[str, Any]:
        """
        Existing behavior preserved.
        When pm_days_to_expiry is provided, we match the closest expiry (legacy).
        Otherwise, parse pm_market to support multiple expiries.
        """
        self.logger.info("evaluate_opportunities CALLED")
        self.logger.info(f" - PM Market: {pm_market}")
        self.logger.info(f" - Currency: {currency}")
        self.logger.info(f" - Options available: {len(options)}")

        selected_expiries: List[date] = []

        # Choose selection path:
        # 1) If legacy param supplied, do the old single-expiry selection
        if pm_days_to_expiry is not None:
            filtered, selected_expiries = filter_options_by_expiry(
                options,
                pm_days_to_expiry=pm_days_to_expiry,
                inclusive=inclusive,
                pm_expiries=None,
                window=None,
            )
        else:
            # 2) New path: parse the market text and select multiple expiries
            avail = pm_collect_unique_expiries(options)
            pm = PMContract.from_market_text(pm_market, tz=tz, max_expiries=max_expiries_per_contract)

            # Prefer shared enumerator (0..60d window, liquidity gates) if available
            if enumerate_expiries is not None and pm.point is not None:
                pm_ts = pm.point
                # For date-only "on <date>" cases parsed as 00:00, downstream can bump to 23:59:59Z
                if pm_default_cutoff_for_date_only:
                    pm_ts = pm_default_cutoff_for_date_only(pm_ts)
                enum = enumerate_expiries(pm_ts, options, policy_cfg=None)
                selected_expiries = [c.date for c in enum]
                filtered, _ = filter_options_by_expiry(
                    options,
                    pm_days_to_expiry=None,
                    inclusive=inclusive,
                    pm_expiries=selected_expiries,
                    window=None,
                )
            else:
                # Fallback: legacy candidate selection by nearest days
                selected_expiries = pm.candidate_expiries(avail)
                filtered, _ = filter_options_by_expiry(
                    options,
                    pm_days_to_expiry=None,
                    inclusive=inclusive,
                    pm_expiries=selected_expiries,
                    window=None,
                )

        self.logger.info(f" - Selected expiries: {[str(d) for d in selected_expiries]}")

        # ... continue with pricing/greeks/hedge using 'filtered' set ...
        # Return shape preserved; include selected expiries for observability.
        return {
            "selected_expiries": [str(d) for d in selected_expiries],
            "options_filtered": filtered,
        }