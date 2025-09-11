# -*- coding: utf-8 -*-
"""
Polymarket data access layer (Gamma + CLOB).
Adds multi-token pricing helpers for debug capture.

This module is intentionally small and dependency-light. It provides:
  - HTTP helpers with basic retries and sane timeouts
  - Gamma read-only endpoints:
      * iter_events
      * iter_markets
      * get_event
      * get_market
  - CLOB read endpoints:
      * get_book
      * get_books
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional

import requests

# Conservative batch size for CLOB /prices to avoid HTTP 400 on large payloads
MAX_CLOB_PRICE_BATCH = 100

DEFAULT_USER_AGENT = "pm-refactor/1.0 (+https://polymarket.com)"

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# (connect timeout, read timeout)
REQ_TIMEOUT = (6.1, 20.0)
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.35  # seconds


@dataclass
class HttpResult:
    ok: bool
    status: int
    json: Optional[object]
    error: Optional[str] = None


class PolymarketClient:
    """
    Thin HTTP client for Polymarket Gamma + CLOB APIs.
    """

    def __init__(self, session: Optional[requests.Session] = None, logger: Optional[logging.Logger] = None, timeout: float = 12.0, backoff: float = 0.25):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
        self.logger = logger or logging.getLogger("PolymarketClient")
        self.timeout = timeout
        self.backoff = backoff

    # -------------------- HTTP helpers --------------------
    

    @staticmethod
    def _chunked(seq, n):
        n = max(int(n or 1), 1)
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    def _get(self, url: str, params: Optional[dict] = None) -> HttpResult:
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = self.session.get(url, params=params, timeout=REQ_TIMEOUT)
                if r.status_code == 200:
                    return HttpResult(True, r.status_code, r.json())
                else:
                    last_err = f"HTTP {r.status_code}: {r.text[:240]}"
                    self.logger.warning("GET %s failed: %s", getattr(r, "url", url), last_err)
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                self.logger.warning("GET %s exception: %s", url, e)
            time.sleep(SLEEP_BETWEEN * attempt)
        return HttpResult(False, 0, None, last_err)

    def _post_json(self, url: str, payload: object) -> HttpResult:
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = self.session.post(url, json=payload, timeout=REQ_TIMEOUT)
                if r.status_code == 200:
                    return HttpResult(True, r.status_code, r.json())
                else:
                    last_err = f"HTTP {r.status_code}: {r.text[:240]}"
                    self.logger.warning("POST %s failed: %s", getattr(r, "url", url), last_err)
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                self.logger.warning("POST %s exception: %s", url, e)
            time.sleep(SLEEP_BETWEEN * attempt)
        return HttpResult(False, 0, None, last_err)

    # -------------------- Gamma --------------------
    def iter_events(self, *, limit: int = 250, include_closed: bool = False, **filters) -> Iterable[dict]:
        """
        Iterator over events with offset-based pagination.
        Accepts arbitrary filter params supported by Gamma (e.g., tag_id, start_date_min, etc.)
        """
        offset = 0
        while True:
            params = {"limit": limit, "offset": offset}
            if not include_closed:
                params["closed"] = "false"
            params.update(filters or {})
            res = self._get(f"{GAMMA_BASE}/events", params)
            if not res.ok:
                self.logger.error("iter_events failed: %s", res.error)
                break
            data = res.json
            if not isinstance(data, list) or not data:
                break
            for item in data:
                yield item
            if len(data) < limit:
                break
            offset += limit

    def iter_markets(self, *, limit: int = 250, include_closed: bool = False, **filters) -> Iterable[dict]:
        """
        Iterator over markets with offset-based pagination.
        """
        offset = 0
        while True:
            params = {"limit": limit, "offset": offset}
            if not include_closed:
                params["closed"] = "false"
            params.update(filters or {})
            res = self._get(f"{GAMMA_BASE}/markets", params)
            if not res.ok:
                self.logger.error("iter_markets failed: %s", res.error)
                break
            data = res.json
            if not isinstance(data, list) or not data:
                break
            for item in data:
                yield item
            if len(data) < limit:
                break
            offset += limit

    def get_event(self, event_id: str) -> Optional[dict]:
        res = self._get(f"{GAMMA_BASE}/events/{event_id}")
        return res.json if res.ok else None

    def get_market(self, market_id: str) -> Optional[dict]:
        res = self._get(f"{GAMMA_BASE}/markets/{market_id}")
        return res.json if res.ok else None

    # -------------------- NEW: CLOB Markets iterator --------------------
    def iter_clob_markets(self, *, start_cursor: str = "") -> Iterator[Dict[str, Any]]:
        """
        Iterate CLOB markets with token pairs (Token[2] -> token_id/outcome).
        Docs: GET /<clob-endpoint>/markets?next_cursor=... . The response contains
        'data': [{ 'market_slug': ..., 'question_id': ..., 'tokens': [{'token_id':..., 'outcome':...}, ...]}].
        See Polymarket docs.  :contentReference[oaicite:1]{index=1}
        """
        cursor = start_cursor or ""
        while True:
            params = {}
            if cursor:
                params["next_cursor"] = cursor
            url = f"{CLOB_BASE}/markets"
            r = requests.get(url, params=params, timeout=self.timeout)
            if r.status_code != 200:
                self.logger.warning("CLOB /markets HTTP %s body=%s", r.status_code, r.text[:300])
                break
            payload = r.json() or {}
            data = payload.get("data") or []
            for m in data:
                yield m
            cursor = (payload.get("next_cursor") or "") if isinstance(payload.get("next_cursor"), str) else ""
            # The docs note next_cursor == 'LTE=' means end-of-stream. :contentReference[oaicite:2]{index=2}
            if not cursor or cursor == "LTE=":
                break

    # -------------------- CLOB books: single + batch --------------------
    def get_book(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order book summary for a single token.
        Docs: GET /book?token_id=...  :contentReference[oaicite:3]{index=3}
        """
        try:
            url = f"{CLOB_BASE}/book"
            r = requests.get(url, params={"token_id": token_id}, timeout=self.timeout)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def get_books(self, token_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """
        Batch get order book summaries for multiple tokens.
        Docs: POST /books with body: [{ "token_id": "<id>"}, ...]  :contentReference[oaicite:4]{index=4}
        Returns a dict[token_id] -> book (response uses asset_id as the key field).
        """
        ids = [str(t).strip() for t in token_ids if t]
        out: Dict[str, Dict[str, Any]] = {}
        if not ids:
            return out
        # chunk conservatively to avoid payload bloat
        for i in range(0, len(ids), 200):
            chunk = ids[i:i+200]
            try:
                payload = [{"token_id": tid} for tid in chunk]
                r = requests.post(f"{CLOB_BASE}/books", json=payload, timeout=self.timeout)
                if r.status_code != 200:
                    self.logger.warning("CLOB /books HTTP %s for %d ids (e.g. %s...)", r.status_code, len(chunk), chunk[:1])
                    time.sleep(self.backoff)
                    continue
                arr = r.json() or []
                # Response objects are keyed by "asset_id" (docs), with bids/asks arrays.
                for book in arr:
                    tid = str(book.get("asset_id") or book.get("token_id") or "").strip()
                    if tid:
                        out[tid] = book
            except Exception as e:
                self.logger.warning("CLOB /books failed for %d ids: %s", len(chunk), e)
                time.sleep(self.backoff)
                continue
        return out

    # -------------------- Pricing helpers --------------------
    def get_prices_by_request(self, token_ids: Iterable[str], sides: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, str]]:
        """
        POST /prices with an ARRAY body (batched safely).
        Returns: {token_id: {"BUY": "...", "SELL": "..."}}
        """
        ids = [str(t).strip() for t in token_ids if t]
        if not ids:
            return {}
        side_list = [str(s).upper() for s in (sides or ("BUY","SELL"))]
        out: Dict[str, Dict[str, str]] = {}
        total = len(ids); ok_batches=0; fail_batches=0
        for chunk in self._chunked(ids, MAX_CLOB_PRICE_BATCH):
            payload = []
            for tid in chunk:
                for s in side_list:
                    payload.append({"token_id": tid, "side": s})
            try:
                r = self.session.post(f"{CLOB_BASE}/prices", json=payload, timeout=self.timeout)
                if r.status_code != 200:
                    fail_batches += 1
                    self.logger.warning("CLOB /prices HTTP %s for %d tokens", r.status_code, len(chunk))
                    continue
                data = r.json() or {}
                if isinstance(data, dict):
                    for k, v in data.items():
                        tid = str(k)
                        if isinstance(v, dict):
                            out.setdefault(tid, {})
                            for kk, vv in v.items():
                                out[tid][str(kk).upper()] = str(vv)
                ok_batches += 1
            except Exception as e:
                fail_batches += 1
                self.logger.warning("CLOB /prices failed for batch of %d: %s", len(chunk), e)
                continue
        self.logger.info(
            "clob_prices: requested=%d, batch_size=%d, batches=%d, ok_batches=%d, failed_batches=%d, prices_returned=%d",
            total, MAX_CLOB_PRICE_BATCH, (total + MAX_CLOB_PRICE_BATCH - 1)//MAX_CLOB_PRICE_BATCH, ok_batches, fail_batches, len(out)
        )
        return out

    def get_all_prices(self) -> Dict[str, Dict[str, str]]:
        """
        GET /prices -> ENTIRE map of token_id -> {BUY, SELL}. Potentially large.
        Docs: https://docs.polymarket.com/api-reference/pricing/get-multiple-market-prices
        """
        try:
            r = self.session.get(f"{CLOB_BASE}/prices", timeout=self.timeout)
            if r.status_code != 200:
                self.logger.warning("CLOB GET /prices HTTP %s", r.status_code)
                return {}
            data = r.json() or {}
            if isinstance(data, dict):
                return {str(k): {str(sk).upper(): str(sv) for sk, sv in (v or {}).items()} for k, v in data.items()}
            return {}
        except Exception as e:
            self.logger.warning("CLOB GET /prices failed: %s", e)
            return {}

    # -------------------- Pricing: multiple market prices (batched) --------------------
    def get_multiple_prices(self, token_ids: Iterable[str], sides: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, str]]:
        """
        Fetch best BUY/SELL prices for a set of token_ids using POST /prices with batching.
        Docs: https://docs.polymarket.com/api-reference/pricing/get-multiple-market-prices-by-request
        Returns: {token_id: {"BUY": "x.xx", "SELL": "y.yy"}}
        """
        tids = [str(t).strip() for t in token_ids if str(t).strip()]
        if not tids:
            return {}
        side_list = list(sides) if sides else ["BUY", "SELL"]
        total = len(tids)
        ok_batches = 0; fail_batches = 0
        out: Dict[str, Dict[str, str]] = {}
        for chunk in self._chunked(tids, MAX_CLOB_PRICE_BATCH):
            req: list[dict] = []
            for tid in chunk:
                for s in side_list:
                    req.append({"token_id": tid, "side": s})
            try:
                r = self.session.post(f"{CLOB_BASE}/prices", json=req, timeout=self.timeout)
                if r.status_code != 200:
                    fail_batches += 1
                    self.logger.warning("CLOB /prices HTTP %s for %d tokens", r.status_code, len(chunk))
                    continue
                data = r.json() or {}
                if isinstance(data, dict):
                    for k, v in data.items():
                        tid = str(k)
                        if isinstance(v, dict):
                            out.setdefault(tid, {})
                            for kk, vv in v.items():
                                out[tid][str(kk).upper()] = str(vv)
                ok_batches += 1
            except Exception as e:
                fail_batches += 1
                self.logger.warning("CLOB /prices failed for batch of %d: %s", len(chunk), e)
                continue
        self.logger.info(
            "clob_prices: requested=%d, batch_size=%d, batches=%d, ok_batches=%d, failed_batches=%d, prices_returned=%d",
            total, MAX_CLOB_PRICE_BATCH, (total + MAX_CLOB_PRICE_BATCH - 1)//MAX_CLOB_PRICE_BATCH, ok_batches, fail_batches, len(out)
        )
        return out