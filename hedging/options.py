from __future__ import annotations

from typing import Protocol, Mapping, Any, Dict, Iterable, Optional, List, Tuple
import os
import logging
from dataclasses import dataclass
from strategies.tag_router import StrategyTagRouter
from digital_hedge_builder import build_digital_vertical_at_K
try:
    from config_manager import RISK_FREE_RATE, get_config  # type: ignore
except Exception:  # pragma: no cover
    RISK_FREE_RATE = 0.0
    def get_config():
        return None
from utils.log_gate import reason_debug  # type: ignore
from utils.step_debugger import get_step_debugger  # type: ignore
from utils.debug_recorder import get_recorder  # type: ignore
from utils.validation_audit import emit as _audit_emit

# Optional deps — degrade gracefully when missing
try:
    from probability_ranker import ProbabilityRanker  # type: ignore
except Exception:  # pragma: no cover
    class ProbabilityRanker:  # type: ignore
        def rank_opportunities(self, opportunities):  # pass-through
            return opportunities

try:
    from expected_value_filter import ExpectedValueFilter  # type: ignore
except Exception:  # pragma: no cover
    class ExpectedValueFilter:  # type: ignore
        def filter_opportunities(self, opportunities):
            return opportunities

try:
    from market_data_analyzer import MarketDataAnalyzer  # type: ignore
except Exception:  # pragma: no cover
    class MarketDataAnalyzer:  # type: ignore
        def analyze(self, currency: str, **kwargs):
            # conservative defaults
            return {
                "pm_max_usd": 500.0,
                "options_max_usd": 500.0,
                "perp_max_usd": 500.0,
            }

from config_manager import (
    LIQUIDITY_SAFETY_FACTOR,
    MIN_POSITION_SIZE,
    MAX_POSITION_SIZE,
)

class HedgeBuilder(Protocol):
    def build(self, market_snapshot: Mapping[str, Any]) -> Iterable[Dict[str, Any]]: ...

@dataclass
class _Quote:
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]

class OptionHedgeBuilder:
    """
    Bridge from refreshed market data (scanners) → evaluated opportunities.

    Implements the essential pieces from the original main_scanner:
      - PM liquidity checks and YES/NO pairing
      - Dynamic position sizing from MarketDataAnalyzer
      - Anchored digital vertical construction when options data is available
      - Ranking & filtering via ProbabilityRanker + ExpectedValueFilter
    """
    def __init__(self,
                 scanners: Dict[str, Dict[str, Any]],
                 market_analyzer: Optional[Any] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        self.scanners = scanners or {}
        self.market_analyzer = market_analyzer or MarketDataAnalyzer()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        # Write one consumer-side checkpoint only (single JSON per run)
        self._consumer_checkpoint_written = False

    # ----- public API -----
    def build(self, _market_snapshot: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
        opportunities: List[Dict[str, Any]] = []
        debugger = get_step_debugger()

        for currency, scanner in self.scanners.items():
            contracts = scanner.get('contracts') or []
            if not contracts:
                continue

            for contract in contracts:
                if not self._pm_has_liquidity(contract):
                    continue

                # ---- Strategy routing by tags (options.XXX) ----
                # Build instrument universe once per contract
                hedge_instruments: Dict[str, Any] = {}
                oc = scanner.get('options_collector')
                if scanner.get('has_options') and oc:
                    try:
                        all_options = oc.get_all_options() or []
                        hedge_instruments["options"] = all_options
                        try:
                            _audit_emit({
                                "run_id": os.getenv("APP_RUN_ID", "unknown"),
                                "stage": "pre_strategy",
                                "pm_market_id": contract.get("id") or contract.get("question_id") or contract.get("slug"),
                                "pm_question": contract.get("question"),
                                "pm_ticker": contract.get("ticker") or contract.get("symbol") or contract.get("asset") or contract.get("pm_asset"),
                                "pm_resolution_iso": contract.get("end_date") or contract.get("endDate"),
                                "pm_resolution_date": (contract.get("end_date") or contract.get("endDate")),
                                "options_count_seen": len(all_options or []),
                                "fields_checked": ["currency","strike_price","yes_price","no_price","days_to_expiry"],
                                "raw_currency_field": contract.get("currency"),
                            })
                        except Exception:
                            pass
                        # -- Consumer-side checkpoint (single JSON) --
                        # Right after options are made available to the hedge builder,
                        # record a compact summary proving whether options are present
                        # and what the hedge logic will "see". Write once per run.
                        try:
                            if not self._consumer_checkpoint_written:
                                rec = get_recorder(get_config())
                                if getattr(rec, "enabled", False):
                                    expiries = []
                                    strike_min = None
                                    strike_max = None
                                    # Cap iteration to keep this lightweight
                                    for o in (all_options or [])[:2000]:
                                        ex = (o.get("expiry_date")
                                              or o.get("expiry")
                                              or o.get("expiration"))
                                        if ex:
                                            expiries.append(str(ex))
                                        try:
                                            k = (o.get("strike")
                                                 or o.get("strike_price")
                                                 or o.get("k"))
                                            if k is not None:
                                                kf = float(k)
                                                strike_min = kf if strike_min is None else min(strike_min, kf)
                                                strike_max = kf if strike_max is None else max(strike_max, kf)
                                        except Exception:
                                            pass
                                    contract_hint = (contract.get("question_id")
                                                     or contract.get("slug")
                                                     or contract.get("id")
                                                     or contract.get("marketSlug")
                                                     or contract.get("title"))
                                    summary = {
                                        "currency": currency,
                                        "contract_hint": str(contract_hint),
                                        "has_options_flag": bool(scanner.get("has_options")),
                                        "collector_present": bool(oc),
                                        "num_options": len(all_options or []),
                                        "unique_expiries_count": len(set(expiries)),
                                        "sample_expiries": sorted(set(expiries))[:5],
                                        "strike_min": strike_min,
                                        "strike_max": strike_max,
                                    }
                                    rec.dump_json(
                                        "OptionsHedgeBuilder_checkpoints/consumer_summary.json",
                                        summary,
                                        category="checkpoint",
                                        overwrite=True,
                                    )
                                    self._consumer_checkpoint_written = True
                        except Exception:
                            # Never allow debugging to impact flow
                            pass
                    except Exception:
                        pass
                if scanner.get('has_perps') and scanner.get('perps_collector'):
                    try:
                        hedge_instruments["perps"] = scanner['perps_collector'].get_perp_data(currency) or {}
                    except Exception:
                        pass

                # Derive dynamic sizing
                position_size = self._calculate_liquidity_based_position_size(contract, hedge_instruments)

                # Collect hierarchical strategy tags; fallback to legacy names
                tags: List[str] = list(contract.get("strategyTags") or [])
                if not tags:
                    legacy = [str(s) for s in (contract.get("strategies") or [])]
                    if "variance_swap" in legacy:
                        tags.append("options.variance_swap")

                # Gate options.variance_swap to SINGLE_THRESHOLD markets only
                if "options.variance_swap" in tags and str(contract.get("marketClass")).upper() != "SINGLE_THRESHOLD":
                    tags = [t for t in tags if t != "options.variance_swap"]

                if tags:
                    router = StrategyTagRouter()
                    strategy_instances = router.instantiate_for_tags(
                        [t for t in tags if t.startswith("options.")],
                        risk_free_rate=RISK_FREE_RATE,
                        cfg=get_config(),
                    )
                else:
                    strategy_instances = []

                produced_by_strategy = False
                if strategy_instances:
                    current_spot = float(scanner.get('current_spot') or 0.0)
                    for strat in strategy_instances:
                        try:
                            opps = strat.evaluate_opportunities(
                                polymarket_contract=contract,
                                hedge_instruments=hedge_instruments,
                                current_spot=current_spot,
                                position_size=position_size,
                            ) or []
                        except Exception as e:
                            self.logger.debug("Strategy %s failed: %s", getattr(strat, "__class__", type(strat)).__name__, e)
                            continue

                        for opp in opps:
                            # Standard metadata
                            opp.setdefault("currency", currency)
                            opp.setdefault("hedge_type", "options")
                            opp.setdefault("strategy", getattr(strat, "__class__", type(strat)).__name__)
                            # PM spend and payoff normalization
                            pm_price = contract.get('yes_price') if contract.get('is_above', True) else contract.get('no_price')
                            try:
                                pm_price = float(pm_price) if pm_price is not None else None
                            except Exception:
                                pm_price = None
                            opp.setdefault("position_size_usd", float(position_size))
                            opp.setdefault("pm_cash_out", float(position_size))
                            # Fallback payoff fields if a strategy forgot to set them
                            if "profit_if_yes" not in opp or "profit_if_no" not in opp:
                                try:
                                    shares = (float(position_size) / float(pm_price)) if pm_price else None
                                    if shares is not None:
                                        if bool(contract.get("is_above", True)):
                                            pm_yes = shares * 1.0 - float(position_size)
                                            pm_no  = -float(position_size)
                                        else:
                                            pm_yes = -float(position_size)
                                            pm_no  = shares * 1.0 - float(position_size)
                                        costs   = opp.get("costs") or {}
                                        debit   = float(costs.get("option_entry_debit", 0.0))
                                        credit  = float(costs.get("option_entry_credit", 0.0))
                                        opp.setdefault("profit_if_yes", pm_yes - debit + credit)
                                        opp.setdefault("profit_if_no",  pm_no  - debit + credit)
                                        opp.setdefault("upfront_cashflow", -(float(position_size) + debit - credit))
                                except Exception:
                                    pass

                            # Normalize Polymarket block (and compute days if missing)
                            pmc = contract or {}
                            days = pmc.get("days_to_expiry")
                            if days is None:
                                from datetime import datetime, timezone
                                end = pmc.get("endDate") or pmc.get("end_date")
                                if end:
                                    try:
                                        end_dt = datetime.fromisoformat(str(end).replace("Z","+00:00"))
                                        days = max(0.0, (end_dt - datetime.now(timezone.utc)).total_seconds() / 86400.0)
                                    except Exception:
                                        days = None
                            opp["polymarket"] = {
                                "question": pmc.get("question"),
                                "strike": pmc.get("strike_price"),
                                "yes_price": pmc.get("yes_price"),
                                "no_price": pmc.get("no_price"),
                                "is_above": pmc.get("is_above", True),
                                "end_date": pmc.get("end_date") or pmc.get("endDate"),
                                "days_to_expiry": days,
                            }
                            opportunities.append(opp)
                            produced_by_strategy = True

                if produced_by_strategy:
                    # Already produced opportunities via strategy routing; skip generic digital construction
                    continue

                base = {
                    "currency": currency,
                    "hedge_type": "options",
                    "strategy": "digital_vertical",
                    "polymarket_contract": contract,
                }
                for opp in self._build_pm_pairs(base, contract):
                    # Build instrument universe per contract
                    hedge_instruments: Dict[str, Any] = {}
                    oc = scanner.get('options_collector')
                    if scanner.get('has_options') and oc:
                        try:
                            # Use pre-fetched options list if available
                            all_options = oc.get_all_options() or []
                            hedge_instruments["options"] = all_options
                        except Exception:
                            pass

                    if scanner.get('has_perps') and scanner.get('perps_collector'):
                        try:
                            hedge_instruments["perps"] = scanner['perps_collector'].get_perp_data(currency) or {}
                        except Exception:
                            pass

                    # Dynamic position size
                    position_size = self._calculate_liquidity_based_position_size(contract, hedge_instruments)
                    opp["position_size_usd"] = float(position_size)

                    # Construct vertical(s). When EVALUATE_ALL_EXPIRIES=1, build one opp per valid expiry.
                    detail_sets: List[Dict[str, Any]] = []
                    if os.getenv("EVALUATE_ALL_EXPIRIES") == "1":
                        try:
                            detail_sets = self._construct_options_hedges(opp, contract, scanner) or []
                        except Exception:
                            detail_sets = []
                    else:
                        details = self._construct_options_hedge(opp, contract, scanner)
                        if details:
                            detail_sets = [details]

                    # ---- Standardize payoffs & cashflows (ranker requires these) ----
                    try:
                        def _finalize(opp_like: Dict[str, Any]) -> Dict[str, Any]:
                            # PM spend
                            opp_like.setdefault("pm_cash_out", float(opp_like.get("position_size_usd") or 0.0))
                            # Fallback profit fields if missing
                            if ("profit_if_yes" not in opp_like) or ("profit_if_no" not in opp_like):
                                pm_side_yes = bool(contract.get("is_above", True))
                                pm_price = contract.get('yes_price') if pm_side_yes else contract.get('no_price')
                                try:
                                    pm_price = float(pm_price) if pm_price is not None else None
                                except Exception:
                                    pm_price = None
                                pos = float(opp_like.get("position_size_usd") or 0.0)
                                shares = (pos / pm_price) if (pm_price and pm_price > 0) else None
                                if shares is not None:
                                    if pm_side_yes:
                                        pm_yes = shares * 1.0 - pos
                                        pm_no = -pos
                                    else:
                                        pm_yes = -pos
                                        pm_no = shares * 1.0 - pos
                                    costs = opp_like.get("costs") or {}
                                    debit = float(costs.get("option_entry_debit", 0.0))
                                    credit = float(costs.get("option_entry_credit", 0.0))
                                    opp_like.setdefault("profit_if_yes", pm_yes - debit + credit)
                                    opp_like.setdefault("profit_if_no", pm_no - debit + credit)
                                    opp_like.setdefault("upfront_cashflow", -(pos + debit - credit))
                            # Normalize PM block for downstream ranker
                            pmc = contract or {}
                            opp_like["polymarket"] = {
                                "question": pmc.get("question"),
                                "strike": pmc.get("strike_price"),
                                "yes_price": pmc.get("yes_price"),
                                "no_price": pmc.get("no_price"),
                                "is_above": pmc.get("is_above", True),
                                "end_date": pmc.get("end_date") or pmc.get("endDate"),
                                "days_to_expiry": pmc.get("days_to_expiry"),
                            }
                            return opp_like
                    except Exception:
                        # Keep going; ranking will drop only the rare item still missing fields
                        pass

                    # Derive days_to_expiry from endDate when missing
                    try:
                        if contract.get("endDate") and not contract.get("days_to_expiry"):
                            from datetime import datetime, timezone
                            end_dt = datetime.fromisoformat(str(contract["endDate"]).replace("Z", "+00:00"))
                            days = max(0.0, (end_dt - datetime.now(timezone.utc)).total_seconds() / 86400.0)
                            contract["days_to_expiry"] = days
                    except Exception:
                        pass

                    if not detail_sets:
                        # still push the PM-only opp for visibility in diagnostics
                        opportunities.append(_finalize(dict(opp)))
                    else:
                        for ds in detail_sets:
                            opp_i = dict(opp)
                            opp_i.update(ds)
                            opportunities.append(_finalize(opp_i))

        # Checkpoint: Raw hedge opportunities before ranking
        debugger.checkpoint("hedge_opportunities_raw", opportunities,
                          {"count": len(opportunities),
                           "currencies": list(set(o.get("currency") for o in opportunities)),
                           "strategies": list(set(o.get("strategy") for o in opportunities))})

        # ---- Diagnostic bypass: return everything (no ranking, no EV filter) ----
        if os.getenv("RETURN_ALL_OPPS") == "1":
            debugger.checkpoint("return_all_opps_bypass", opportunities, {"count": len(opportunities)})
            return opportunities

        # Rank + filter
        if opportunities:
            # Checkpoint: Before ranking
            debugger.checkpoint("pre_probability_ranking", opportunities,
                              {"count": len(opportunities),
                               "has_payoffs": sum(1 for o in opportunities if "profit_if_yes" in o and "profit_if_no" in o)})
            
            try:
                ranker = ProbabilityRanker()
                opportunities = ranker.rank_opportunities(opportunities)  # type: ignore
                
                # Checkpoint: After ranking
                debugger.checkpoint("post_probability_ranking", opportunities,
                                  {"count": len(opportunities),
                                   "has_metrics": sum(1 for o in opportunities if "metrics" in o)})
            except Exception as e:
                self.logger.warning("ProbabilityRanker failed (%s); continuing without ranking", e)
            # Checkpoint: Before EV filter
            pre_ev_count = len(opportunities)
            debugger.checkpoint("pre_ev_filter", opportunities,
                              {"count": pre_ev_count})
            
            try:
                if os.getenv("SKIP_EV_FILTER") == "1":
                    pass
                else:
                    # Optional ENV overrides for thresholds
                    def _f(env_name: str, default: Optional[float]) -> Optional[float]:
                        try:
                            v = os.getenv(env_name)
                            return float(v) if v is not None and v != "" else default
                        except Exception:
                            return default
                    filt = ExpectedValueFilter(
                        min_ev=_f("MIN_EV", None) or 0.10,
                        min_sharpe=_f("MIN_SHARPE", None) or 0.50,
                        min_kelly=_f("MIN_KELLY", None) or 0.01,
                    )
                    opportunities = filt.filter_opportunities(opportunities)  # type: ignore
                
                # Checkpoint: After EV filter
                debugger.checkpoint("post_ev_filter", opportunities,
                                  {"count": len(opportunities),
                                   "filtered_out": pre_ev_count - len(opportunities)})
            except Exception as e:
                self.logger.warning("ExpectedValueFilter failed (%s); continuing without EV filter", e)

        # Checkpoint: Final hedge opportunities with expiry information
        try:
            # Collect expiry statistics
            expiry_counts = {}
            opportunities_with_expiry = 0
            for opp in opportunities:
                if opp.get('option_expiry'):
                    opportunities_with_expiry += 1
                    expiry = opp['option_expiry']
                    expiry_counts[expiry] = expiry_counts.get(expiry, 0) + 1
                    
            debugger.checkpoint("hedge_opportunities_with_expiry", opportunities,
                              {"count": len(opportunities),
                               "with_expiry": opportunities_with_expiry,
                               "unique_expiries": len(expiry_counts),
                               "expiry_distribution": expiry_counts})
        except Exception:
            pass

        return opportunities

    # ----- helpers -----

    def _pm_has_liquidity(self, contract: Dict[str, Any]) -> bool:
        """
        Minimal PM liquidity gate: require either YES or NO top-of-book size > 0.
        Now robust to books that have only bids or only asks.
        """
        try:
            ysz = float(contract.get('yes_size') or contract.get('yes_qty') or 0.0)
            nsz = float(contract.get('no_size')  or contract.get('no_qty')  or 0.0)
            # Consider explicit bid/ask sizes if size fields are unset
            y_bid = float(contract.get('yes_bid_qty') or 0.0)
            y_ask = float(contract.get('yes_ask_qty') or 0.0)
            n_bid = float(contract.get('no_bid_qty')  or 0.0)
            n_ask = float(contract.get('no_ask_qty')  or 0.0)
            ysz = max(ysz, y_bid, y_ask)
            nsz = max(nsz, n_bid, n_ask)
        except Exception:
            ysz = nsz = 0.0
        ok = (ysz > 0.0) or (nsz > 0.0)
        # Fallback for diagnostics: allow markets with sane prices even if size fields are missing
        if not ok:
            try:
                yp = float(contract.get('yes_price') or 0.0)
                np_ = float(contract.get('no_price') or 0.0)
                price_ok = (0.0 < yp < 1.0) or (0.0 < np_ < 1.0)
                ok = ok or price_ok
            except Exception:
                pass
        if not ok:
            reason_debug(
                self.logger,
                "REPL PM_NO_LIQUIDITY question=%s yes_sz=%.3f no_sz=%.3f",
                str(contract.get('question') or contract.get('id') or '-'), ysz, nsz
            )
        return ok

    def _build_pm_pairs(self, base: Dict[str, Any], contract: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Duplicate each PM market into (YES, short digital) and (NO, long digital).
        """
        is_above = bool(contract.get("is_above", True))

        yes = dict(base)
        yes["pm_side"] = "YES"
        yes["short_digital"] = True if is_above else False
        if "pm_price" not in yes and contract.get("yes_price") is not None:
            yes["pm_price"] = float(contract["yes_price"])

        no = dict(base)
        no["pm_side"] = "NO"
        no["short_digital"] = False if is_above else True
        if "pm_price" not in no and contract.get("no_price") is not None:
            no["pm_price"] = float(contract["no_price"])

        # Keep an explicit copy of the PM block for UIs that expect it on the root too
        for o in (yes, no):
            o["polymarket_contract"] = contract

        return [yes, no]

    def _calculate_liquidity_based_position_size(self, contract: Dict[str, Any], hedge_instruments: Dict[str, Any]) -> float:
        """
        Recreate the dynamic sizing from the old scanner using MarketDataAnalyzer outputs,
        then clamp with LIQUIDITY_SAFETY_FACTOR / MIN_ / MAX_ bounds.
        """
        currency = (contract.get("currency") or contract.get("asset") or "ETH").upper()

        try:
            constraints = self.market_analyzer.analyze(currency=currency)  # type: ignore
        except Exception:
            constraints = {"pm_max_usd": 500.0, "options_max_usd": 500.0, "perp_max_usd": 500.0}

        pm_max = float(constraints.get("pm_max_usd", 500.0))
        opt_max = float(constraints.get("options_max_usd", pm_max))
        perp_max = float(constraints.get("perp_max_usd", pm_max))

        # Prefer options gating when present; otherwise perps; fallback to PM
        if hedge_instruments.get("options"):
            gate = opt_max
        elif hedge_instruments.get("perps"):
            gate = perp_max
        else:
            gate = pm_max

        size = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, gate)) * float(LIQUIDITY_SAFETY_FACTOR or 1.0)
        return float(size)

    def _transform_options_to_chain(self, options: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
        """
        Transform flat options list to nested structure required by digital_hedge_builder.
        Returns: {expiry: {strike: {'call': {...}, 'put': {...}}}}
        """
        chains_by_expiry = {}
        
        for opt in options or []:
            # Get expiry date
            expiry = opt.get('expiry_date') or opt.get('expiry') or opt.get('expiration')
            if not expiry:
                continue
                
            # Get strike as string float
            try:
                strike = str(float(opt.get('strike', 0)))
            except (ValueError, TypeError):
                continue
                
            # Normalize option type
            opt_type_raw = str(opt.get('type', '')).upper()
            if opt_type_raw in ('C', 'CALL'):
                opt_type = 'call'
            elif opt_type_raw in ('P', 'PUT'):
                opt_type = 'put'
            else:
                continue
                
            # Build nested structure
            if expiry not in chains_by_expiry:
                chains_by_expiry[expiry] = {}
            if strike not in chains_by_expiry[expiry]:
                chains_by_expiry[expiry][strike] = {}
            
            chains_by_expiry[expiry][strike][opt_type] = opt
            
        return chains_by_expiry
    
    def _select_best_expiry(self, chains_by_expiry: Dict[str, Any], pm_days_to_expiry: float) -> Optional[str]:
        """
        Select the most appropriate expiry for digital hedge construction.
        Currently selects the nearest expiry that is on or after PM expiry.
        """
        if not chains_by_expiry:
            return None
            
        valid_expiries = []
        
        for expiry, chain in chains_by_expiry.items():
            # Check if this expiry has sufficient liquidity
            # Count instruments with two-sided quotes
            liquid_strikes = 0
            for strike, types in chain.items():
                call = types.get('call', {})
                put = types.get('put', {})
                
                # Check if either call or put has two-sided quotes
                call_liquid = (float(call.get('bid', 0)) > 0 and float(call.get('ask', 0)) > 0)
                put_liquid = (float(put.get('bid', 0)) > 0 and float(put.get('ask', 0)) > 0)
                
                if call_liquid or put_liquid:
                    liquid_strikes += 1
                    
            # Require at least 2 liquid strikes
            if liquid_strikes >= 2:
                valid_expiries.append(expiry)
                
        if not valid_expiries:
            return None
            
        # For now, return the first valid expiry (could be enhanced to sort by days to expiry)
        return valid_expiries[0]

    def _construct_options_hedges(self, opportunity: Dict[str, Any], contract: Dict[str, Any], scanner: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Build a list of digital verticals across *all* valid expiries (diagnostic mode).
        Controlled via EVALUATE_ALL_EXPIRIES=1.
        """
        all_options: List[Dict[str, Any]] = (scanner.get("options_collector").get_all_options()  # type: ignore
                                             if scanner.get("options_collector") else [])
        if not all_options:
            reason_debug(self.logger, "REPL NO_OPTIONS_DATA currency=%s", str(opportunity.get('currency') or (opportunity.get('polymarket') or {}).get('asset') or '-'))
            return None
        try:
            K = float(contract.get("strike_price"))
        except Exception:
            reason_debug(self.logger, "REPL MISSING_PM_STRIKE question=%s",
                         str((contract or {}).get('question') or (opportunity.get('polymarket') or {}).get('question') or '-'))
            return None
        is_above = bool(contract.get("is_above", True))
        chains_by_expiry = self._transform_options_to_chain(all_options)
        if not chains_by_expiry:
            reason_debug(self.logger, "REPL NO_VALID_CHAINS currency=%s", opportunity.get('currency'))
            return None
        # Reuse same liquidity rule as _select_best_expiry, but enumerate
        valid_expiries = []
        for expiry, chain in chains_by_expiry.items():
            liquid_strikes = 0
            for strike, types in chain.items():
                call = types.get('call', {}); put = types.get('put', {})
                call_liquid = (float(call.get('bid', 0)) > 0 and float(call.get('ask', 0)) > 0)
                put_liquid = (float(put.get('bid', 0)) > 0 and float(put.get('ask', 0)) > 0)
                if call_liquid or put_liquid:
                    liquid_strikes += 1
            if liquid_strikes >= 2:
                valid_expiries.append(expiry)
        if not valid_expiries:
            reason_debug(self.logger, "REPL NO_SUITABLE_EXPIRY currency=%s", opportunity.get('currency'))
            return None
        out: List[Dict[str, Any]] = []
        for ex in valid_expiries:
            chain_for_expiry = chains_by_expiry.get(ex) or {}
            digital_result = build_digital_vertical_at_K(
                is_above=is_above,
                K=K,
                expiry=ex,
                chain_for_expiry=chain_for_expiry
            )
            if not digital_result:
                reason_debug(self.logger, "REPL DIGITAL_BUILD_FAILED K=%s expiry=%s", K, ex)
                continue
            width = digital_result['width']
            k_low = digital_result['k_low']
            k_high = digital_result['k_high']
            pm_side = str(opportunity.get("pm_side","YES")).upper()
            side_price = opportunity.get("pm_price")
            if side_price is None:
                side_price = contract.get("yes_price") if pm_side == "YES" else contract.get("no_price")
            try:
                side_price = float(side_price) if side_price is not None else None
            except Exception:
                side_price = None
            pos_usd = float(opportunity.get("position_size_usd", 0.0))
            if side_price is None or pos_usd <= 0.0:
                reason_debug(self.logger, "REPL POSITION_SIZING_UNAVAILABLE side=%s side_price=%s pos_usd=%s",
                             pm_side, str(side_price), str(opportunity.get('position_size_usd')))
                continue
            pm_shares = pos_usd / max(1e-12, side_price)
            n = pm_shares / max(1e-12, width)
            n = max(0.01, round(n / 0.01) * 0.01)
            short_digital = bool(opportunity.get("short_digital", False))
            legs = digital_result['legs_shortD'] if short_digital else digital_result['legs_longD']
            digital_price = digital_result['digital_sell_per_1'] if short_digital else digital_result['digital_buy_per_1']
            required_legs = []
            for leg in legs:
                scaled_leg = dict(leg); scaled_leg['contracts'] = n
                required_legs.append(scaled_leg)
            if short_digital:
                option_entry_credit = abs(digital_price * width * n); option_entry_debit = 0.0
            else:
                option_entry_debit = abs(digital_price * width * n); option_entry_credit = 0.0
            pm_cash_out = float(opportunity.get("pm_cash_out", 0.0))
            upfront_cashflow = - (pm_cash_out + option_entry_debit - option_entry_credit)
            max_profit = width * n - option_entry_debit if not short_digital else option_entry_credit
            max_loss = option_entry_debit if short_digital else max(0.0, option_entry_credit)
            out.append({
                "required_options": required_legs,
                "digital_width": width,
                "spread_contracts": n,
                "short_vertical": short_digital,
                "costs": {
                    "pm_cash_out": pm_cash_out,
                    "option_entry_debit": option_entry_debit,
                    "option_entry_credit": option_entry_credit,
                    "upfront_cashflow": upfront_cashflow,
                },
                "required_capital": max(0.0, pm_cash_out + option_entry_debit - option_entry_credit),
                "max_profit": float(max_profit),
                "max_loss": float(-abs(max_loss)),
                "option_expiry": ex,
                "hedge": {
                    "type": "digital_vertical",
                    "instrument_type": "options",
                    "legs": required_legs,
                    "expiry": ex,
                    "k_low": k_low,
                    "k_high": k_high,
                    "width": width,
                    "has_exact_k": digital_result.get('has_exact_k', False)
                }
            })
        return out

    # --- anchored digital vertical construction (best-effort) ---

    def _get_quote(self, items: List[Dict[str, Any]], otype: str, strike: float, expiry: Optional[str]) -> _Quote:
        """Extract a robust (bid, ask, mid) from raw option items."""
        otype = (otype or "").upper()
        best: Optional[Dict[str, Any]] = None
        for o in items or []:
            try:
                if str(o.get("type","")).upper() != otype:
                    continue
                if float(o.get("strike")) != float(strike):
                    continue
                if expiry and o.get("expiry_date") not in (expiry,):
                    continue
                best = o; break
            except Exception:
                continue
        if not best:
            reason_debug(self.logger, "REPL NO_OPTION_MATCH type=%s strike=%s expiry=%s", otype, str(strike), str(expiry))
            return _Quote(None, None, None)
        bid = best.get("bid"); ask = best.get("ask"); mid = best.get("mid")
        try:
            bid = float(bid) if bid is not None else None
            ask = float(ask) if ask is not None else None
            if mid is None and bid is not None and ask is not None and bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
            mid = float(mid) if mid is not None else None
        except Exception:
            bid = ask = mid = None
        # DEBUG: mark one‑sided quotes
        if best is not None and (bid is None or ask is None):
            reason_debug(self.logger, "REPL ONE_SIDED_QUOTE type=%s strike=%s has_bid=%s has_ask=%s",
                         otype, str(strike), str(bid is not None), str(ask is not None))
        return _Quote(bid, ask, mid)

    def _nearest_vertical(self, all_options: List[Dict[str, Any]], K: float, is_above: bool) -> Optional[Tuple[str, float, float]]:
        """
        Find a small-width vertical around K using available strikes.
        Returns (otype, k_low, k_high) where otype in {"CALL","PUT"}.
        """
        strikes = sorted({float(o.get("strike")) for o in all_options if o.get("strike") is not None})
        if not strikes:
            reason_debug(self.logger, "REPL NO_OPTION_STRIKES")
            return None
        # Find nearest lower & higher strikes around K
        lower = max((s for s in strikes if s <= K), default=None)
        higher = min((s for s in strikes if s >= K and s != lower), default=None)
        if lower is None or higher is None or higher == lower:
            reason_debug(self.logger, "REPL INSUFFICIENT_STRIKES_AROUND_K K=%s", str(K))
            return None
        otype = "CALL" if is_above else "PUT"
        return (otype, float(lower), float(higher))

    def _construct_options_hedge(self, opportunity: Dict[str, Any], contract: Dict[str, Any], scanner: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build a digital vertical centered at the PM strike using live options when available.
        Uses digital_hedge_builder to construct the actual hedge with proper expiry selection.
        """
        all_options: List[Dict[str, Any]] = (scanner.get("options_collector").get_all_options()  # type: ignore
                                             if scanner.get("options_collector") else [])
        if not all_options:
            reason_debug(self.logger, "REPL NO_OPTIONS_DATA currency=%s",
                         str(opportunity.get('currency') or (opportunity.get('polymarket') or {}).get('asset') or '-'))
            return None

        try:
            K = float(contract.get("strike_price"))
        except Exception:
            reason_debug(self.logger, "REPL MISSING_PM_STRIKE question=%s",
                         str((contract or {}).get('question') or (opportunity.get('polymarket') or {}).get('question') or '-'))
            return None

        is_above = bool(contract.get("is_above", True))
        
        # Transform options to required structure
        chains_by_expiry = self._transform_options_to_chain(all_options)
        if not chains_by_expiry:
            reason_debug(self.logger, "REPL NO_VALID_CHAINS currency=%s", opportunity.get('currency'))
            return None
            
        # Select best expiry
        pm_days_to_expiry = contract.get('days_to_expiry', 0)
        selected_expiry = self._select_best_expiry(chains_by_expiry, pm_days_to_expiry)
        if not selected_expiry:
            reason_debug(self.logger, "REPL NO_SUITABLE_EXPIRY currency=%s", opportunity.get('currency'))
            return None
            
        # Build digital vertical using digital_hedge_builder
        chain_for_expiry = chains_by_expiry[selected_expiry]
        digital_result = build_digital_vertical_at_K(
            is_above=is_above,
            K=K,
            expiry=selected_expiry,
            chain_for_expiry=chain_for_expiry
        )
        
        if not digital_result:
            reason_debug(self.logger, "REPL DIGITAL_BUILD_FAILED K=%s expiry=%s", K, selected_expiry)
            return None
            
        # Extract results from digital_hedge_builder
        width = digital_result['width']
        k_low = digital_result['k_low']
        k_high = digital_result['k_high']

        # Contract count n = pm_shares / width; derive pm_shares from USD and side price when missing
        pm_side = str(opportunity.get("pm_side","YES")).upper()
        side_price = opportunity.get("pm_price")
        if side_price is None:
            side_price = contract.get("yes_price") if pm_side == "YES" else contract.get("no_price")
        try:
            side_price = float(side_price) if side_price is not None else None
        except Exception:
            side_price = None

        pos_usd = float(opportunity.get("position_size_usd", 0.0))
        if side_price is None or pos_usd <= 0.0:
            reason_debug(self.logger, "REPL POSITION_SIZING_UNAVAILABLE side=%s side_price=%s pos_usd=%s",
                         pm_side, str(side_price), str(opportunity.get('position_size_usd')))
            return None
        pm_shares = pos_usd / max(1e-12, side_price)
        n = pm_shares / max(1e-12, width)

        # Round contracts to a reasonable step (0.01)
        n = max(0.01, round(n / 0.01) * 0.01)
        
        short_digital = bool(opportunity.get("short_digital", False))

        # Get legs from digital_result and scale by n contracts
        if short_digital:
            legs = digital_result['legs_shortD']
            digital_price = digital_result['digital_sell_per_1']
        else:
            legs = digital_result['legs_longD']
            digital_price = digital_result['digital_buy_per_1']
            
        # Scale legs by contract count
        required_legs = []
        for leg in legs:
            scaled_leg = dict(leg)
            scaled_leg['contracts'] = n
            required_legs.append(scaled_leg)
            
        # Calculate entry costs
        if short_digital:
            option_entry_credit = abs(digital_price * width * n)
            option_entry_debit = 0.0
        else:
            option_entry_debit = abs(digital_price * width * n) 
            option_entry_credit = 0.0

        pm_cash_out = float(opportunity.get("pm_cash_out", 0.0))  # leave as-is if previously set
        upfront_cashflow = - (pm_cash_out + option_entry_debit - option_entry_credit)

        # Rough risk envelope (used by ranker when present)
        max_profit = width * n - option_entry_debit if not short_digital else option_entry_credit
        max_loss   = option_entry_debit if short_digital else max(0.0, option_entry_credit)  # conservative

        return {
            "required_options": required_legs,
            "digital_width": width,
            "spread_contracts": n,
            "short_vertical": short_digital,
            "costs": {
                "pm_cash_out": pm_cash_out,
                "option_entry_debit": option_entry_debit,
                "option_entry_credit": option_entry_credit,
                "upfront_cashflow": upfront_cashflow,
            },
            "required_capital": max(0.0, pm_cash_out + option_entry_debit - option_entry_credit),
            "max_profit": float(max_profit),
            "max_loss": float(-abs(max_loss)),  # negative for loss
            "option_expiry": selected_expiry,
            "hedge": {
                "type": "digital_vertical",
                "instrument_type": "options", 
                "legs": required_legs,
                "expiry": selected_expiry,
                "k_low": k_low,
                "k_high": k_high,
                "width": width,
                "has_exact_k": digital_result.get('has_exact_k', False)
            }
        }