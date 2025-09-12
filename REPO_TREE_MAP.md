diff --git a/hedging/options.py b/hedging/options.py
index 0000000..0000001 100644
--- a/hedging/options.py
+++ b/hedging/options.py
@@
-from __future__ import annotations
-from typing import Protocol, Mapping, Any, Dict, Iterable, Optional, List, Tuple
+from __future__ import annotations
+from typing import Protocol, Mapping, Any, Dict, Iterable, Optional, List, Tuple
+import os
 import logging
 from dataclasses import dataclass
@@
 class OptionHedgeBuilder:
@@
-    def build(self, _market_snapshot: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
+    def build(self, _market_snapshot: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
         opportunities: List[Dict[str, Any]] = []
         debugger = get_step_debugger()
@@
-            if scanner.get('has_perps') and scanner.get('perps_collector'):
+            if scanner.get('has_perps') and scanner.get('perps_collector'):
                 try:
                     hedge_instruments["perps"] = scanner['perps_collector'].get_perp_data(currency) or {}
                 except Exception:
                     pass
@@
-            if tags:
+            if tags:
                 router = StrategyTagRouter()
                 strategy_instances = router.instantiate_for_tags(
                     [t for t in tags if t.startswith("options.")],
                     risk_free_rate=RISK_FREE_RATE,
                     cfg=get_config(),
                 )
             else:
                 strategy_instances = []
@@
-        if produced_by_strategy:
-            # Already produced opportunities via strategy routing; skip generic digital construction
-            continue
+        if produced_by_strategy:
+            # Already produced opportunities via strategy routing; skip generic digital construction
+            continue
@@
-        for opp in self._build_pm_pairs(base, contract):
+        for opp in self._build_pm_pairs(base, contract):
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
@@
-            # Construct a vertical (when possible)
-            details = self._construct_options_hedge(opp, contract, scanner)
-            if details:
-                opp.update(details)
-            # ---- Standardize payoffs & cashflows (ranker requires these) ----
+            # Construct vertical(s). When EVALUATE_ALL_EXPIRIES=1, build one opp per valid expiry.
+            detail_sets: List[Dict[str, Any]] = []
+            if os.getenv("EVALUATE_ALL_EXPIRIES") == "1":
+                try:
+                    detail_sets = self._construct_options_hedges(opp, contract, scanner) or []
+                except Exception:
+                    detail_sets = []
+            else:
+                details = self._construct_options_hedge(opp, contract, scanner)
+                if details:
+                    detail_sets = [details]
+
+            # ---- Standardize payoffs & cashflows (ranker requires these) ----
             try:
-                # PM spend
-                opp.setdefault("pm_cash_out", float(opp.get("position_size_usd") or 0.0))
-                # Fallback profit fields if missing
-                if ("profit_if_yes" not in opp) or ("profit_if_no" not in opp):
-                    # Orientation: yes-side if is_above else no-side
-                    pm_side_yes = bool(contract.get("is_above", True))
-                    pm_price = contract.get('yes_price') if pm_side_yes else contract.get('no_price')
-                    try:
-                        pm_price = float(pm_price) if pm_price is not None else None
-                    except Exception:
-                        pm_price = None
-                    pos = float(opp.get("position_size_usd") or 0.0)
-                    shares = (pos / pm_price) if (pm_price and pm_price > 0) else None
-                    if shares is not None:
-                        if pm_side_yes:
-                            pm_yes = shares * 1.0 - pos
-                            pm_no = -pos
-                        else:
-                            pm_yes = -pos
-                            pm_no = shares * 1.0 - pos
-                        costs = opp.get("costs") or {}
-                        debit = float(costs.get("option_entry_debit", 0.0))
-                        credit = float(costs.get("option_entry_credit", 0.0))
-                        opp.setdefault("profit_if_yes", pm_yes - debit + credit)
-                        opp.setdefault("profit_if_no", pm_no - debit + credit)
-                        opp.setdefault("upfront_cashflow", -(pos + debit - credit))
+                def _finalize(opp_like: Dict[str, Any]) -> Dict[str, Any]:
+                    # PM spend
+                    opp_like.setdefault("pm_cash_out", float(opp_like.get("position_size_usd") or 0.0))
+                    # Fallback profit fields if missing
+                    if ("profit_if_yes" not in opp_like) or ("profit_if_no" not in opp_like):
+                        pm_side_yes = bool(contract.get("is_above", True))
+                        pm_price = contract.get('yes_price') if pm_side_yes else contract.get('no_price')
+                        try:
+                            pm_price = float(pm_price) if pm_price is not None else None
+                        except Exception:
+                            pm_price = None
+                        pos = float(opp_like.get("position_size_usd") or 0.0)
+                        shares = (pos / pm_price) if (pm_price and pm_price > 0) else None
+                        if shares is not None:
+                            if pm_side_yes:
+                                pm_yes = shares * 1.0 - pos
+                                pm_no = -pos
+                            else:
+                                pm_yes = -pos
+                                pm_no = shares * 1.0 - pos
+                            costs = opp_like.get("costs") or {}
+                            debit = float(costs.get("option_entry_debit", 0.0))
+                            credit = float(costs.get("option_entry_credit", 0.0))
+                            opp_like.setdefault("profit_if_yes", pm_yes - debit + credit)
+                            opp_like.setdefault("profit_if_no", pm_no - debit + credit)
+                            opp_like.setdefault("upfront_cashflow", -(pos + debit - credit))
+                    # Normalize PM block for downstream ranker
+                    pmc = contract or {}
+                    opp_like["polymarket"] = {
+                        "question": pmc.get("question"),
+                        "strike": pmc.get("strike_price"),
+                        "yes_price": pmc.get("yes_price"),
+                        "no_price": pmc.get("no_price"),
+                        "is_above": pmc.get("is_above", True),
+                        "end_date": pmc.get("end_date") or pmc.get("endDate"),
+                        "days_to_expiry": pmc.get("days_to_expiry"),
+                    }
+                    return opp_like
             except Exception:
                 # Keep going; ranking will drop only the rare item still missing fields
                 pass
-            # Derive days_to_expiry from endDate when missing
-            try:
-                if contract.get("endDate") and not contract.get("days_to_expiry"):
-                    from datetime import datetime, timezone
-                    end_dt = datetime.fromisoformat(str(contract["endDate"]).replace("Z", "+00:00"))
-                    days = max(0.0, (end_dt - datetime.now(timezone.utc)).total_seconds() / 86400.0)
-                    contract["days_to_expiry"] = days
-            except Exception:
-                pass
-            # Normalize PM block for downstream ranker
-            pmc = contract or {}
-            opp["polymarket"] = {
-                "question": pmc.get("question"),
-                "strike": pmc.get("strike_price"),
-                "yes_price": pmc.get("yes_price"),
-                "no_price": pmc.get("no_price"),
-                "is_above": pmc.get("is_above", True),
-                "end_date": pmc.get("end_date") or pmc.get("endDate"),
-                "days_to_expiry": pmc.get("days_to_expiry"),
-            }
-            opportunities.append(opp)
+
+            # Derive days_to_expiry from endDate when missing
+            try:
+                if contract.get("endDate") and not contract.get("days_to_expiry"):
+                    from datetime import datetime, timezone
+                    end_dt = datetime.fromisoformat(str(contract["endDate"]).replace("Z", "+00:00"))
+                    days = max(0.0, (end_dt - datetime.now(timezone.utc)).total_seconds() / 86400.0)
+                    contract["days_to_expiry"] = days
+            except Exception:
+                pass
+
+            if not detail_sets:
+                # still push the PM-only opp for visibility in diagnostics
+                opportunities.append(_finalize(dict(opp)))
+            else:
+                for ds in detail_sets:
+                    opp_i = dict(opp)
+                    opp_i.update(ds)
+                    opportunities.append(_finalize(opp_i))
@@
         # Checkpoint: Raw hedge opportunities before ranking
         debugger.checkpoint("hedge_opportunities_raw", opportunities, {"count": len(opportunities), "currencies": list(set(o.get("currency") for o in opportunities)), "strategies": list(set(o.get("strategy") for o in opportunities))})
 
+        # ---- Diagnostic bypass: return everything (no ranking, no EV filter) ----
+        if os.getenv("RETURN_ALL_OPPS") == "1":
+            debugger.checkpoint("return_all_opps_bypass", opportunities, {"count": len(opportunities)})
+            return opportunities
+
         # Rank + filter
         if opportunities:
             # Checkpoint: Before ranking
             debugger.checkpoint("pre_probability_ranking", opportunities, {"count": len(opportunities), "has_payoffs": sum(1 for o in opportunities if "profit_if_yes" in o and "profit_if_no" in o)})
             try:
                 ranker = ProbabilityRanker()
                 opportunities = ranker.rank_opportunities(opportunities)  # type: ignore
                 # Checkpoint: After ranking
                 debugger.checkpoint("post_probability_ranking", opportunities, {"count": len(opportunities), "has_metrics": sum(1 for o in opportunities if "metrics" in o)})
             except Exception as e:
                 self.logger.warning("ProbabilityRanker failed (%s); continuing without ranking", e)
 
             # Checkpoint: Before EV filter
-            debugger.checkpoint("pre_ev_filter", opportunities, {"count": len(opportunities)})
+            pre_ev_count = len(opportunities)
+            debugger.checkpoint("pre_ev_filter", opportunities, {"count": pre_ev_count})
             try:
-                filt = ExpectedValueFilter()
-                opportunities = filt.filter_opportunities(opportunities)  # type: ignore
+                if os.getenv("SKIP_EV_FILTER") == "1":
+                    pass
+                else:
+                    # Optional ENV overrides for thresholds
+                    def _f(env_name: str, default: Optional[float]) -> Optional[float]:
+                        try:
+                            v = os.getenv(env_name)
+                            return float(v) if v is not None and v != "" else default
+                        except Exception:
+                            return default
+                    filt = ExpectedValueFilter(
+                        min_ev=_f("MIN_EV", None) or 0.10,
+                        min_sharpe=_f("MIN_SHARPE", None) or 0.50,
+                        min_kelly=_f("MIN_KELLY", None) or 0.01,
+                    )
+                    opportunities = filt.filter_opportunities(opportunities)  # type: ignore
                 # Checkpoint: After EV filter
-                debugger.checkpoint("post_ev_filter", opportunities, {"count": len(opportunities), "filtered_out": len(debugger.checkpoints[-2]["stats"]["count"]) - len(opportunities) if debugger.checkpoints else 0})
+                debugger.checkpoint("post_ev_filter", opportunities, {"count": len(opportunities), "filtered_out": pre_ev_count - len(opportunities)})
             except Exception as e:
                 self.logger.warning("ExpectedValueFilter failed (%s); continuing without EV filter", e)
@@
-    def _pm_has_liquidity(self, contract: Dict[str, Any]) -> bool:
+    def _pm_has_liquidity(self, contract: Dict[str, Any]) -> bool:
         """ Minimal PM liquidity gate: require either YES or NO top-of-book size > 0.
         Now robust to books that have only bids or only asks.
         """
         try:
             ysz = float(contract.get('yes_size') or contract.get('yes_qty') or 0.0)
             nsz = float(contract.get('no_size') or contract.get('no_qty') or 0.0)
             # Consider explicit bid/ask sizes if size fields are unset
             y_bid = float(contract.get('yes_bid_qty') or 0.0)
             y_ask = float(contract.get('yes_ask_qty') or 0.0)
             n_bid = float(contract.get('no_bid_qty') or 0.0)
             n_ask = float(contract.get('no_ask_qty') or 0.0)
             ysz = max(ysz, y_bid, y_ask)
             nsz = max(nsz, n_bid, n_ask)
         except Exception:
             ysz = nsz = 0.0
-        ok = (ysz > 0.0) or (nsz > 0.0)
+        ok = (ysz > 0.0) or (nsz > 0.0)
+        # Fallback for diagnostics: allow markets with sane prices even if size fields are missing
+        if not ok:
+            try:
+                yp = float(contract.get('yes_price') or 0.0)
+                np_ = float(contract.get('no_price') or 0.0)
+                price_ok = (0.0 < yp < 1.0) or (0.0 < np_ < 1.0)
+                ok = ok or price_ok
+            except Exception:
+                pass
         if not ok:
             reason_debug(
                 self.logger,
                 "REPL PM_NO_LIQUIDITY question=%s yes_sz=%.3f no_sz=%.3f",
                 str(contract.get('question') or contract.get('id') or '-'),
                 ysz, nsz
             )
         return ok
@@
-    def _select_best_expiry(self, chains_by_expiry: Dict[str, Any], pm_days_to_expiry: float) -> Optional[str]:
+    def _select_best_expiry(self, chains_by_expiry: Dict[str, Any], pm_days_to_expiry: float) -> Optional[str]:
         """ Select the most appropriate expiry for digital hedge construction.
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
+
+    def _construct_options_hedges(self, opportunity: Dict[str, Any], contract: Dict[str, Any], scanner: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
+        """
+        Build a list of digital verticals across *all* valid expiries (diagnostic mode).
+        Controlled via EVALUATE_ALL_EXPIRIES=1.
+        """
+        all_options: List[Dict[str, Any]] = (scanner.get("options_collector").get_all_options()  # type: ignore
+                                             if scanner.get("options_collector") else [])
+        if not all_options:
+            reason_debug(self.logger, "REPL NO_OPTIONS_DATA currency=%s", str(opportunity.get('currency') or (opportunity.get('polymarket') or {}).get('asset') or '-'))
+            return None
+        try:
+            K = float(contract.get("strike_price"))
+        except Exception:
+            reason_debug(self.logger, "REPL MISSING_PM_STRIKE question=%s",
+                         str((contract or {}).get('question') or (opportunity.get('polymarket') or {}).get('question') or '-'))
+            return None
+        is_above = bool(contract.get("is_above", True))
+        chains_by_expiry = self._transform_options_to_chain(all_options)
+        if not chains_by_expiry:
+            reason_debug(self.logger, "REPL NO_VALID_CHAINS currency=%s", opportunity.get('currency'))
+            return None
+        # Reuse same liquidity rule as _select_best_expiry, but enumerate
+        valid_expiries = []
+        for expiry, chain in chains_by_expiry.items():
+            liquid_strikes = 0
+            for strike, types in chain.items():
+                call = types.get('call', {}); put = types.get('put', {})
+                call_liquid = (float(call.get('bid', 0)) > 0 and float(call.get('ask', 0)) > 0)
+                put_liquid = (float(put.get('bid', 0)) > 0 and float(put.get('ask', 0)) > 0)
+                if call_liquid or put_liquid:
+                    liquid_strikes += 1
+            if liquid_strikes >= 2:
+                valid_expiries.append(expiry)
+        if not valid_expiries:
+            reason_debug(self.logger, "REPL NO_SUITABLE_EXPIRY currency=%s", opportunity.get('currency'))
+            return None
+        out: List[Dict[str, Any]] = []
+        for ex in valid_expiries:
+            chain_for_expiry = chains_by_expiry.get(ex) or {}
+            digital_result = build_digital_vertical_at_K(
+                is_above=is_above,
+                K=K,
+                expiry=ex,
+                chain_for_expiry=chain_for_expiry
+            )
+            if not digital_result:
+                reason_debug(self.logger, "REPL DIGITAL_BUILD_FAILED K=%s expiry=%s", K, ex)
+                continue
+            width = digital_result['width']
+            k_low = digital_result['k_low']
+            k_high = digital_result['k_high']
+            pm_side = str(opportunity.get("pm_side","YES")).upper()
+            side_price = opportunity.get("pm_price")
+            if side_price is None:
+                side_price = contract.get("yes_price") if pm_side == "YES" else contract.get("no_price")
+            try:
+                side_price = float(side_price) if side_price is not None else None
+            except Exception:
+                side_price = None
+            pos_usd = float(opportunity.get("position_size_usd", 0.0))
+            if side_price is None or pos_usd <= 0.0:
+                reason_debug(self.logger, "REPL POSITION_SIZING_UNAVAILABLE side=%s side_price=%s pos_usd=%s",
+                             pm_side, str(side_price), str(opportunity.get('position_size_usd')))
+                continue
+            pm_shares = pos_usd / max(1e-12, side_price)
+            n = pm_shares / max(1e-12, width)
+            n = max(0.01, round(n / 0.01) * 0.01)
+            short_digital = bool(opportunity.get("short_digital", False))
+            legs = digital_result['legs_shortD'] if short_digital else digital_result['legs_longD']
+            digital_price = digital_result['digital_sell_per_1'] if short_digital else digital_result['digital_buy_per_1']
+            required_legs = []
+            for leg in legs:
+                scaled_leg = dict(leg); scaled_leg['contracts'] = n
+                required_legs.append(scaled_leg)
+            if short_digital:
+                option_entry_credit = abs(digital_price * width * n); option_entry_debit = 0.0
+            else:
+                option_entry_debit = abs(digital_price * width * n); option_entry_credit = 0.0
+            pm_cash_out = float(opportunity.get("pm_cash_out", 0.0))
+            upfront_cashflow = - (pm_cash_out + option_entry_debit - option_entry_credit)
+            max_profit = width * n - option_entry_debit if not short_digital else option_entry_credit
+            max_loss = option_entry_debit if short_digital else max(0.0, option_entry_credit)
+            out.append({
+                "required_options": required_legs,
+                "digital_width": width,
+                "spread_contracts": n,
+                "short_vertical": short_digital,
+                "costs": {
+                    "pm_cash_out": pm_cash_out,
+                    "option_entry_debit": option_entry_debit,
+                    "option_entry_credit": option_entry_credit,
+                    "upfront_cashflow": upfront_cashflow,
+                },
+                "required_capital": max(0.0, pm_cash_out + option_entry_debit - option_entry_credit),
+                "max_profit": float(max_profit),
+                "max_loss": float(-abs(max_loss)),
+                "option_expiry": ex,
+                "hedge": {
+                    "type": "digital_vertical",
+                    "instrument_type": "options",
+                    "legs": required_legs,
+                    "expiry": ex,
+                    "k_low": k_low,
+                    "k_high": k_high,
+                    "width": width,
+                    "has_exact_k": digital_result.get('has_exact_k', False)
+                }
+            })
+        return out
