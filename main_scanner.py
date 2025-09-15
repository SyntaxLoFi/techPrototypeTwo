from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from utils.timebox import compute_days_to_expiry
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.data_collection.polymarket_client import PolymarketClient  # for CLOB books/prices

logger = logging.getLogger(__name__)
try:
    from logger_config import set_request_id  # type: ignore
except Exception:
    def set_request_id(*args, **kwargs):  # type: ignore
        pass
from utils.debug_recorder import get_recorder  # type: ignore
from utils.log_gate import configure_from_config  # type: ignore
from utils.step_debugger import get_step_debugger  # type: ignore

# ---- new: minimal scanner initialization mirroring the original ----

def _dump_polymarket_clob_debug(
    markets: List[Dict[str, Any]], *, recorder=None, logger: Optional[logging.Logger] = None
) -> None:
    """
    Collect CLOB artifacts (order books + prices) for all tokenIds found in `markets`
    and persist them to debug_runs/<run_id>/polymarket/*.json when debug is enabled.
    Safe to call multiple times; DebugRecorder deduplicates per-file writes.
    """
    try:
        rec = recorder or get_recorder()
    except Exception:
        rec = None
    # Fallback: create a recorder if debug is on but no recorder active
    try:
        if (not rec) or (not getattr(rec, "enabled", False)):
            from utils.debug_recorder import RawDataRecorder  # type: ignore
            rec = RawDataRecorder(enabled=True)
    except Exception:
        pass

    log = logger or logging.getLogger("PolymarketCLOB")

    # Collect token IDs from markets (supports both YES/NO and UP/DOWN)
    token_ids: List[str] = []
    for m in markets or []:
        tids = m.get("tokenIds") if isinstance(m, dict) else None
        if isinstance(tids, dict):
            for key in ("yes", "no", "up", "down", "YES", "NO", "UP", "DOWN"):
                v = tids.get(key)
                if v:
                    s = str(v).strip()
                    if s:
                        token_ids.append(s)
        elif isinstance(tids, list):
            for v in tids:
                s = str(v).strip()
                if s:
                    token_ids.append(s)
    token_ids = sorted(set(token_ids))
    if not token_ids:
        log.info("No tokenIds present in markets; skipping CLOB fetch.")
        return

    client = PolymarketClient(logger=log)
    # Order books
    try:
        books = client.get_books(token_ids) or {}
    except Exception as e:
        log.warning("Error fetching CLOB books: %s", e)
        books = {}
    # Prices
    try:
        prices = client.get_multiple_prices(token_ids) or {}
    except Exception as e:
        log.warning("Error fetching CLOB prices: %s", e)
        prices = {}
    # Persist
    try:
        if rec:
            rec.dump_json("polymarket/token_ids.json", token_ids, category="polymarket")
            rec.dump_json("polymarket/books.json", books, category="polymarket", overwrite=True)
            rec.dump_json("polymarket/prices.json", prices, category="polymarket", overwrite=True)
    except Exception:
        pass
    log.info("Dumped Polymarket CLOB artifacts: %d tokens, books=%d, prices=%d", len(token_ids), len(books), len(prices))

def build_scanners(logger: Optional[logging.Logger] = None, *, recorder=None) -> Dict[str, Dict[str, Any]]:
    """
    Build per-currency scanners with orderbook/spot handlers, options/perps collectors,
    and attach Polymarket contracts when available.
    """
    logger = logger or logging.getLogger("ScannerInit")
    scanners: Dict[str, Dict[str, Any]] = {}

    try:
        from config_manager import ARBITRAGE_ENABLED_CURRENCIES, LYRA_OPTIONS_CURRENCIES, LYRA_PERPS_CURRENCIES
        from scripts.data_collection.orderbook_handler import OrderbookHandler
        from scripts.data_collection.spot_feed_handler import SpotFeedHandler
        from scripts.data_collection.options_chain_collector import OptionsChainCollector
        from scripts.data_collection.perps_data_collector import PerpsDataCollector
        from scripts.data_collection.polymarket_fetcher import PolymarketFetcher
        # --- NEW: v2 fetcher & offline tagger (optional) ---
        try:
            # Uses your new scripts/data_collection/polymarket_fetcher_v2.py
            from scripts.data_collection.polymarket_fetcher_v2 import PolymarketFetcher as PolymarketFetcherV2  # type: ignore
        except Exception:
            PolymarketFetcherV2 = None  # type: ignore
        try:
            # Uses your new scripts/data_collection/pm_ingest.py
            from scripts.data_collection.pm_ingest import tag_from_local_markets  # type: ignore
        except Exception:
            tag_from_local_markets = None  # type: ignore
    except Exception as e:
        logger.error("Failed imports during scanner init: %s", e)
        return scanners

    # Fetch Polymarket contracts once
    contracts_by_ccy: Dict[str, List[Dict[str, Any]]] = {}
    try:
        # --- NEW: prefer v2 (tagged) fetcher; fallback to legacy + offline tagging ---
        markets: List[Dict[str, Any]] = []
        # 1) Try the new v2 fetcher which returns pre‑tagged markets
        try:
            if PolymarketFetcherV2 is not None:
                pf2 = PolymarketFetcherV2(debug_dir="debug_runs")
                markets = pf2.fetch_tagged_markets_live(include_closed=False)
                try:
                    rec = recorder or get_recorder()
                    rec.dump_json("polymarket/pm_tagged_live.json", markets, category="polymarket")
                except Exception:
                    pass
                # NEW: persist CLOB artifacts (books + prices) for the same token set
                try:
                    _dump_polymarket_clob_debug(
                        markets,
                        recorder=recorder or get_recorder(),
                        logger=logging.getLogger("PolymarketCLOB"),
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Polymarket v2 fetch failed: %s", e)

        # 2) Fallback to legacy fetcher if v2 not present or failed
        if not markets:
            pf = PolymarketFetcher()
            # Prefer markets method when available; otherwise events
            try:
                markets = pf.fetch_crypto_markets()  # type: ignore[attr-defined]
            except Exception:
                # Fallback: fetch events and parse into standardized contract dicts
                events = pf.fetch_crypto_events()
                try:
                    markets = pf.parse_contracts(events)
                except Exception:
                    # Last resort: leave empty; we prefer no data to malformed mapping
                    markets = []

            # 3) If we got raw markets, offline‑tag them using your new tagger
            if markets and tag_from_local_markets is not None:
                try:
                    tagged = tag_from_local_markets(markets)
                    if tagged:
                        markets = tagged
                        try:
                            rec = recorder or get_recorder()
                            rec.dump_json("polymarket/pm_tagged_offline.json", markets, category="polymarket")
                        except Exception:
                            pass
                except Exception:
                    pass

        # Debug dump + summary log for whichever path succeeded
        try:
            rec = recorder or get_recorder()
            rec.dump_json("polymarket/markets.json", markets, category="polymarket")
            logger.info("Polymarket fetcher working (%d crypto markets found)", len(markets))
            # Checkpoint: Raw Polymarket data
            debugger = get_step_debugger(rec)
            debugger.checkpoint("polymarket_markets_raw", markets, {"source": "v2" if markets else "legacy"})
        except Exception:
            pass
        for m in markets or []:
            # Robust symbol extraction – normalize to BTC/ETH/SOL/XRP/DOGE
            raw = (
                # --- NEW: prefer classified asset symbols first ---
                m.get("asset")
                or m.get("pm_asset")
                # legacy fallbacks:
                or m.get("currency")
                or m.get("ticker")
                or m.get("symbol")
                or m.get("underlying")
                or m.get("base")
                or ""
            )
            if not raw:
                # Fallback: try to infer from question/title
                q = f"{m.get('question','')} {m.get('event_title','')}".upper()
                for sym in ("BTC", "ETH", "SOL", "XRP", "DOGE"):
                    if sym in q:
                        raw = sym
                        break
            ccy = str(raw).upper().strip()
            if not ccy:
                continue
            # NEW: ensure currency stamped onto the *contract dict* used by strategies
            try:
                m.setdefault("currency", ccy)
            except Exception:
                pass
            # NEW: compute days_to_expiry on the same dict reaching validation
            try:
                if m.get("days_to_expiry") is None:
                    end_val = m.get("end_date") or m.get("endDate") or m.get("endDateIso") or m.get("resolution_ts")
                    dte = compute_days_to_expiry(end_val)
                    if dte is not None:
                        m["days_to_expiry"] = float(dte)
            except Exception:
                # Safe: leave absent if we cannot parse; strategy gates remain unchanged
                pass
            contracts_by_ccy.setdefault(ccy, []).append(m)
    except Exception as e:
        logger.warning("Polymarket fetch failed: %s", e)

    # Checkpoint: Contracts grouped by currency
    try:
        debugger = get_step_debugger(recorder)
        debugger.checkpoint("contracts_by_currency", contracts_by_ccy, {"currencies": list(contracts_by_ccy.keys())})
    except Exception:
        pass

    for ccy in set(getattr(__import__('config_manager'), 'ARBITRAGE_ENABLED_CURRENCIES', ())):
        has_options = bool(getattr(__import__('config_manager'), 'LYRA_OPTIONS_CURRENCIES', {}).get(ccy))
        has_perps = bool(getattr(__import__('config_manager'), 'LYRA_PERPS_CURRENCIES', {}).get(ccy))

        ob = OrderbookHandler()
        spot = SpotFeedHandler(ccy)
        oc = OptionsChainCollector() if has_options else None
        pc = PerpsDataCollector() if has_perps else None
        if oc:
            try:
                oc.fetch_all_options(ccy)
                try:
                    rec = recorder or get_recorder()
                    rec.dump_json(f"options/{ccy}_options.json", oc.get_all_options(), category="options")
                    logger.info("Options chain collection successful (%d %s options)", len(oc.get_all_options() or []), ccy)
                except Exception:
                    pass
            except Exception:
                pass
        if pc:
            try:
                pc.fetch_perp_info(ccy, fetch_funding_history=True)
                try:
                    rec = recorder or get_recorder()
                    data = getattr(pc, "perps_data", {}).get(ccy)
                    if data:
                        rec.dump_json(f"perps/{ccy}_perps.json", data, category="perps")
                        logger.info("Perps data collection successful for %s", ccy)
                except Exception:
                    pass
            except Exception:
                pass

        scanners[ccy] = {
            "currency": ccy,
            "orderbook_handler": ob,
            "spot_handler": spot,
            "options_collector": oc,
            "perps_collector": pc,
            "has_options": has_options,
            "has_perps": has_perps,
            "current_spot": None,
            "contracts": contracts_by_ccy.get(ccy, []),
        }

    # Checkpoint: Completed scanners with all data
    try:
        debugger = get_step_debugger(recorder)
        debugger.checkpoint("scanners_built", scanners, 
                          {"currencies": list(scanners.keys()),
                           "total_contracts": sum(len(s.get("contracts", [])) for s in scanners.values())})
    except Exception:
        pass

    return scanners


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore
        if np is not None:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


class ArbitrageScanner:
    """
    Back-compat façade: exposes `save_opportunities(...)` so existing tests
    and callers keep working. Full runtime orchestration lives in `main()`.
    """
    def __init__(self, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.scanners: Dict[str, Any] = {}
        self.opportunities: List[Dict[str, Any]] = []

    def save_opportunities(self, filename: Optional[str] = None, detailed_mode: bool = True) -> None:
        if not getattr(self, "opportunities", None):
            return

        if filename:
            path = Path(filename); path.parent.mkdir(parents=True, exist_ok=True)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = Path("results"); outdir.mkdir(parents=True, exist_ok=True)
            path = outdir / f"detailed_opportunities_{ts}.json"

        data: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "total_opportunities": len(self.opportunities),
            "opportunities": [],
        }

        for opp in self.opportunities:
            metrics: Dict[str, Any] = (opp.get("metrics") or {})
            probs: Dict[str, Any] = (opp.get("probabilities") or {})

            dni_raw = metrics.get("dni", None)
            if dni_raw is None or (isinstance(dni_raw, str) and len(str(dni_raw).strip()) == 0):
                distance_to_no_arb: Optional[float] = None
            else:
                try:
                    distance_to_no_arb = float(dni_raw)
                except Exception:
                    distance_to_no_arb = None

            if "is_true_arbitrage" in metrics:
                is_true_arb: Optional[bool] = bool(metrics.get("is_true_arbitrage"))
            elif distance_to_no_arb is not None:
                is_true_arb = bool(distance_to_no_arb >= 0.0)
            else:
                is_true_arb = None

            pm_block: Dict[str, Any] = {
                "prob_of_profit": float(metrics.get("prob_of_profit", 0.0) or 0.0),
                "expected_value": float(metrics.get("expected_value", 0.0) or 0.0),
                "adjusted_ev": float(metrics.get("adjusted_ev", 0.0) or 0.0),
                "edge_per_downside": float(metrics.get("edge_per_downside", 0.0) or 0.0),
                "distance_to_no_arb": distance_to_no_arb,
            }
            if is_true_arb is not None:
                pm_block["is_true_arbitrage"] = is_true_arb

            data["opportunities"].append({
                "rank": opp.get("rank", 0),
                "quality_tier": opp.get("quality_tier", "UNKNOWN"),
                "currency": opp.get("currency", opp.get("symbol", "Unknown")),
                "hedge_type": opp.get("hedge_type", opp.get("strategy_type", "Unknown")),
                "strategy": opp.get("strategy", opp.get("strategy_name", "Unknown")),
                "max_profit": opp.get("max_profit", 0.0),
                "max_loss": opp.get("max_loss", 0.0),
                "probabilities": probs,
                "probability_metrics": pm_block,
                "polymarket": (opp.get("polymarket_contract") or opp.get("polymarket") or {}),
            })

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as fh:
            json.dump(data, fh, indent=2, cls=NumpyEncoder)
        os.replace(tmp_path, path)
        self.logger.info("Saved %d opportunities to %s", len(self.opportunities), str(path))


async def main() -> None:
    try:
        from logger_config import setup_logging  # type: ignore
        setup_logging()
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log = logging.getLogger("main_scanner")
    log.info("🚀 Polymarket–Lyra Arbitrage Scanner")

    try:
        from orchestrator import Orchestrator  # type: ignore
        from data_refresh import DataRefresh  # type: ignore
        from hedging.options import OptionHedgeBuilder  # type: ignore
        from persistence.writer import DefaultWriter  # type: ignore

        # Stable run id added to all logs + debug dumps
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        os.environ.setdefault("APP_RUN_ID", run_id)
        try:
            set_request_id(run_id)
        except Exception:
            pass

        try:
            from config_loader import get_config  # type: ignore
            app_cfg = get_config()
        except Exception:
            app_cfg = None
        try:
            configure_from_config(app_cfg, run_id=run_id)
        except Exception:
            pass

        rec = get_recorder(app_cfg)
        debugger = get_step_debugger(rec)
        try:
            logging.getLogger("Main").info(
                "DebugRecorder status: enabled=%s dir=%s run_id=%s",
                getattr(rec, "enabled", False),
                getattr(rec, "dump_dir", "debug_runs"),
                getattr(rec, "run_id", os.getenv("APP_RUN_ID", "unknown")),
            )
        except Exception:
            pass
        scanners = build_scanners(logger=logging.getLogger("ScannerInit"), recorder=rec)
        refresher = DataRefresh(scanners=scanners, logger=logging.getLogger("DataRefresh"))
        hedge = OptionHedgeBuilder(scanners=scanners, logger=logging.getLogger("OptionHedgeBuilder"))
        writer = DefaultWriter(logger=logging.getLogger("Writer"))

        orch = Orchestrator(refresher=refresher, hedge=hedge, writer=writer,
                            config=app_cfg, logger=logging.getLogger("Orchestrator"))

        # run_once is async; await directly
        await asyncio.wait_for(orch.run_once(), timeout=60 * 15)
        
        # Save debugging summary at the end
        debugger.save_summary()

    except ImportError as e:
        log.error("Refactored modules not found (%s). Add:\n  orchestrator.py\n  data_refresh.py\n  hedging/options.py\n  persistence/writer.py", e)
        raise
    except Exception as e:
        log.exception("Fatal error during scan: %s", e)
        raise


if __name__ == "__main__":
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for t in pending: t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.run_until_complete(loop.shutdown_asyncgens()); loop.close()
    except KeyboardInterrupt:
        logger.info("⚠️  Scanner interrupted by user")
