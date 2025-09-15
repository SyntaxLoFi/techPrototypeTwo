# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .pm_ingest import tag_from_local_markets, fetch_and_tag_live
from .polymarket_client import PolymarketClient
from typing import Any
try:
    from utils.debug_recorder import get_recorder  # type: ignore
except Exception:
    def get_recorder(*args: Any, **kwargs: Any):
        class _N:
            enabled = False
            def dump_json(self, *a, **k):  # no-op
                return None
        return _N()


class PolymarketFetcher:
    def __init__(self, *, debug_dir: str = "debug_runs", logger: Optional[logging.Logger] = None, top_of_book_only: Optional[bool] = None):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("PolymarketFetcher")
        # Fallback: operate using top-of-book only (from /prices or derived) when set or when depth missing
        if top_of_book_only is None:
            env = os.getenv("TOP_OF_BOOK_ONLY", "").strip().lower()
            top_of_book_only = env in {"1","true","t","yes","on"}
        self.top_of_book_only = bool(top_of_book_only)

    # -------------------- public API --------------------
    def tag_from_local_debug(self, filename: str = "markets.json") -> List[dict]:
        """
        Offline path: read debug_runs/markets.json and tag those entries.
        """
        path = self.debug_dir / filename
        with path.open("r", encoding="utf-8") as f:
            markets = json.load(f)
        tagged = tag_from_local_markets(markets)
        # No orderbook enrichment on offline path
        self._dump_json("pm_tagged_offline.json", tagged)
        self._dump_json("pm_tagged_offline_counts.json", self._summarize(tagged))
        return tagged

    def fetch_tagged_markets_live(self, include_closed: bool = False) -> List[dict]:
        """
        Live path via Gamma API; writes pm_tagged_live*.json in debug_runs.
        """
        client = PolymarketClient(logger=self.logger)
        tagged = fetch_and_tag_live(client, include_closed=include_closed)
        # 1) First, attach tokenIds from the CLOB markets catalog by market_slug
        try:
            tagged = self._attach_token_ids_via_clob(client, tagged)
        except Exception as e:
            self.logger.warning("CLOB token attach failed: %s", e)
        # 2) Next, enrich with CLOB orderbook so liquidity gate can pass
        try:
            tagged = self._enrich_with_orderbooks(client, tagged)
        except Exception as e:
            self.logger.warning("Orderbook enrichment failed: %s", e)
        self._dump_json("pm_tagged_live.json", tagged)
        self._dump_json("pm_tagged_live_counts.json", self._summarize(tagged))
        return tagged

    # -------------------- helpers --------------------
    def _dump_json(self, name: str, obj: object) -> None:
        path = self.debug_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        self.logger.info("Wrote %s", path)

    def _summarize(self, tagged: List[dict]) -> Dict[str, object]:
        c_assets = Counter([t.get("asset") for t in tagged if t.get("asset")])
        c_classes = Counter([t.get("marketClass") for t in tagged if t.get("marketClass")])
        by_asset = defaultdict(Counter)
        for t in tagged:
            a = t.get("asset")
            if not a:
                continue
            by_asset[a][t.get("marketClass")] += 1
        return {
            "assets_count": dict(c_assets),
            "class_count": dict(c_classes),
            "by_asset": {a: dict(ct) for a, ct in by_asset.items()},
            "total_markets_tagged": len(tagged),
        }

    # -------------------- NEW: attach token IDs from CLOB markets --------------------
    def _attach_token_ids_via_clob(self, client: PolymarketClient, markets: List[dict]) -> List[dict]:
        """
        Use the CLOB /markets catalog to map each tagged market's slug to its YES/NO (or UP/DOWN)
        token IDs. This fixes the "0 tokenIds in snapshot" issue and enables /books enrichment.
        Docs: GET /<clob-endpoint>/markets returns tokens[2] with {token_id, outcome}.  :contentReference[oaicite:5]{index=5}
        """
        # Build slug -> token map
        slug_to_tokens: Dict[str, Dict[str, str]] = {}
        count = 0
        for cm in client.iter_clob_markets():
            count += 1
            slug = (cm.get("market_slug") or cm.get("slug") or "").strip().lower()
            toks = cm.get("tokens") or []
            tokmap: Dict[str, str] = {}
            for t in toks:
                outcome = str(t.get("outcome") or "").strip().upper()
                tid = str(t.get("token_id") or "").strip()
                if not outcome or not tid:
                    continue
                # Normalize outcome keys to what downstream expects
                if outcome in ("YES", "NO", "UP", "DOWN"):
                    tokmap[outcome] = tid
            if slug and tokmap:
                slug_to_tokens[slug] = tokmap
        self.logger.info("CLOB markets scanned=%d, token maps built=%d", count, len(slug_to_tokens))

        # Attach into tagged rows by slug
        attached = 0
        out: List[dict] = []
        for m in markets:
            mm = dict(m)
            slug = (mm.get("marketSlug") or mm.get("market_slug") or mm.get("slug") or "").strip().lower()
            if slug and not mm.get("tokenIds"):
                tokmap = slug_to_tokens.get(slug)
                if tokmap:
                    # use the common 'tokenIds' container the rest of the pipeline reads
                    mm["tokenIds"] = {k.upper(): v for k, v in tokmap.items()}
                    attached += 1
            out.append(mm)
        self.logger.info("CLOB tokenIds attached: %d/%d markets", attached, len(out))
        return out

    # -------------------- orderbook enrichment --------------------
    def _enrich_with_orderbooks(self, client: PolymarketClient, markets: List[dict]) -> List[dict]:
        """
        Attach best bid/ask and simple quantities for YES/NO (or UP/DOWN) tokens so
        downstream liquidity gates like _pm_has_liquidity() can pass.

        Fields added per market (when tokenIds are present):
          yes_bid, yes_bid_qty, yes_ask, yes_ask_qty, yes_mid, yes_qty, yes_size
          no_bid,  no_bid_qty,  no_ask,  no_ask_qty,  no_mid,  no_qty,  no_size
        We also set yes_price/no_price to the best ask (buy) for downstream pm_price.
        """
        # Gather token ids
        id_to_slots: Dict[str, List[tuple]] = {}
        for idx, m in enumerate(markets):
            toks = (m.get("tokenIds") or {}) if isinstance(m.get("tokenIds"), dict) else {}
            # Normalize keys to upper-case
            toks = {str(k).upper(): str(v) for k, v in toks.items() if v}
            for key in ("YES", "NO", "UP", "DOWN"):
                tid = toks.get(key)
                if not tid:
                    continue
                id_to_slots.setdefault(str(tid), []).append((idx, key))

        if not id_to_slots:
            return markets

        token_ids = list(id_to_slots.keys())
        if not token_ids:
            return markets

        # tiny: if we somehow missed tokenIds attach, log once for visibility
        if not token_ids:
            self.logger.info("Orderbook enrichment skipped: no tokenIds present")
            return markets

        # From docs: batch POST /books with [{"token_id": "..."}, ...]  :contentReference[oaicite:6]{index=6}
        # client.get_books implements this.

        # NEW: Also capture direct {BUY,SELL} from /prices for same tokens
        try:
            token_ids = list(id_to_slots.keys())
            prices = client.get_prices_by_request(token_ids) or {}
            rec = get_recorder()
            if getattr(rec, "enabled", False):
                rec.dump_json("polymarket/prices.json", prices, category="polymarket")
            # Fallback to class debug_dir as well
            try:
                self._dump_json("polymarket/prices.json", prices)
            except Exception:
                pass
        except Exception:
            pass

        # Fetch in manageable batches
        def chunked(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i:i+n]

        books: Dict[str, dict] = {}

        for chunk in chunked(token_ids, 200):
            try:
                books.update(client.get_books(chunk) or {})
            except Exception as e:
                self.logger.warning("get_books(%d tokens) failed: %s", len(chunk), e)

        # Persist the collected books for offline inspection
        try:
            rec = get_recorder()
            if getattr(rec, "enabled", False):
                rec.dump_json("polymarket/books.json", books, category="polymarket", overwrite=True)
                # Fallback write to class debug_dir for visibility in non-recorder envs
                try:
                    self._dump_json("polymarket/books.json", books)
                except Exception:
                    pass
        except Exception:
            pass

        # Build top-of-book snapshot from whichever source we have (books preferred)
        try:
            from decimal import Decimal
            books_best: Dict[str, Dict[str, str]] = {}
            if books:
                for _tid, _book in (books or {}).items():
                    bids = _book.get("bids") or []
                    asks = _book.get("asks") or []
                    # best SELL = highest bid, best BUY = lowest ask
                    best_ask = None; ask_qty = Decimal("0")
                    for lvl in asks:
                        try:
                            p = Decimal(str(lvl.get("price"))); q = Decimal(str(lvl.get("size") or "0"))
                        except Exception:
                            continue
                        if best_ask is None or p < best_ask:
                            best_ask = p; ask_qty = q
                    best_bid = None; bid_qty = Decimal("0")
                    for lvl in bids:
                        try:
                            p = Decimal(str(lvl.get("price"))); q = Decimal(str(lvl.get("size") or "0"))
                        except Exception:
                            continue
                        if best_bid is None or p > best_bid:
                            best_bid = p; bid_qty = q
                    if best_ask is None and best_bid is None:
                        continue
                    books_best[str(_tid)] = {
                        "BUY": str(best_ask) if best_ask is not None else None,
                        "SELL": str(best_bid) if best_bid is not None else None,
                        "BUY_QTY": str(ask_qty),
                        "SELL_QTY": str(bid_qty),
                    }
            else:
                # Fallback: synthesize books_best from /prices when no depth is available
                for _tid, pv in (prices or {}).items():
                    if not isinstance(pv, dict):
                        continue
                    buy = pv.get("BUY"); sell = pv.get("SELL")
                    if buy is None and sell is None:
                        continue
                    books_best[str(_tid)] = {
                        "BUY": str(buy) if buy is not None else None,
                        "SELL": str(sell) if sell is not None else None,
                        "BUY_QTY": "0",
                        "SELL_QTY": "0",
                    }
            rec = get_recorder()
            if getattr(rec, "enabled", False):
                rec.dump_json("polymarket/books_best.json", books_best, category="polymarket")
            try:
                self._dump_json("polymarket/books_best.json", books_best)
            except Exception:
                pass
        except Exception:
            pass

        # Helper(s) to read top-of-book
        def _coerce_float(x):
            try:
                return float(x) if x is not None else None
            except Exception:
                return None
        def _read_level(level: dict) -> tuple:
            px = level.get("price") or level.get("px") or level.get("p") or level.get("mid") or 0
            qty = level.get("size") or level.get("qty") or level.get("quantity") or level.get("q") or level.get("amount") or 0
            try:
                return float(px), float(qty)
            except Exception:
                return (None, 0.0)

        def _best(levels):
            if isinstance(levels, list) and levels:
                return _read_level(levels[0])
            if isinstance(levels, dict):
                return _read_level(levels)
            return (None, 0.0)

        def _extract(book: dict) -> dict:
            # Accept both formats: {best_bid:{}, best_ask:{}} OR arrays bids/asks
            bb_px, bb_qty = _best(book.get("best_bid") or book.get("bids") or [])
            ba_px, ba_qty = _best(book.get("best_ask") or book.get("asks") or [])
            mid = book.get("mid") or book.get("mid_price") or None
            try:
                mid = float(mid) if mid is not None else None
            except Exception:
                mid = None
            return {
                "bid": bb_px, "bid_qty": bb_qty,
                "ask": ba_px, "ask_qty": ba_qty,
                "mid": mid,
            }

        # Apply to markets
        for tid, slots in id_to_slots.items():
            book = books.get(str(tid)) or {}
            info = _extract(book) if (book and not getattr(self, "top_of_book_only", False)) else (
                (
                    {"bid": _coerce_float(prices.get(str(tid), {}).get("SELL")), "bid_qty": 0.0,
                     "ask": _coerce_float(prices.get(str(tid), {}).get("BUY")),  "ask_qty": 0.0, "mid": None}
                    if prices.get(str(tid)) else {"bid": None, "bid_qty": 0.0, "ask": None, "ask_qty": 0.0, "mid": None}
                )
            )
            for idx, key in slots:
                m = markets[idx]
                if key in ("YES", "UP"):
                    m["yes_bid"] = info["bid"];      m["yes_bid_qty"] = info["bid_qty"]
                    m["yes_ask"] = info["ask"];      m["yes_ask_qty"] = info["ask_qty"]
                    m["yes_mid"] = info["mid"]
                    # Back-compat fields for liquidity gate and pricing
                    # Do not overwrite Gamma prices if already stamped
                    if "yes_price" not in m:
                        m["yes_price"] = info["ask"]
                    # Use *either* side for liquidity gates; some books have only bids or only asks
                    qty = float(max(info.get("ask_qty") or 0.0, info.get("bid_qty") or 0.0))
                    m["yes_qty"] = qty
                    m["yes_size"] = qty
                elif key in ("NO", "DOWN"):
                    m["no_bid"] = info["bid"];       m["no_bid_qty"] = info["bid_qty"]
                    m["no_ask"] = info["ask"];       m["no_ask_qty"] = info["ask_qty"]
                    m["no_mid"] = info["mid"]
                    if "no_price" not in m:
                        m["no_price"] = info["ask"]
                    # Use *either* side for liquidity gates; some books have only bids or only asks
                    qty = float(max(info.get("ask_qty") or 0.0, info.get("bid_qty") or 0.0))
                    m["no_qty"] = qty
                    m["no_size"] = qty

        # If a market only had UP/DOWN tokens, map them into YES/NO slots for compatibility
        for m in markets:
            if ("yes_qty" not in m or m.get("yes_qty", 0) == 0) and "yes_ask_qty" not in m:
                # ensure quantities for gate if UP mapped above
                if "yes_qty" not in m and "yes_ask_qty" in m:
                    qty = float(m.get("yes_ask_qty") or 0.0)
                    m["yes_qty"] = qty; m["yes_size"] = qty
            if ("no_qty" not in m or m.get("no_qty", 0) == 0) and "no_ask_qty" not in m:
                if "no_qty" not in m and "no_ask_qty" in m:
                    qty = float(m.get("no_ask_qty") or 0.0)
                    m["no_qty"] = qty; m["no_size"] = qty

        return markets