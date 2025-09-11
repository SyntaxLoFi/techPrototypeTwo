# -*- coding: utf-8 -*-
"""
End-to-end ingestion & tagging for Polymarket crypto price markets.

Two entry points:
  - tag_from_local_markets(markets: Iterable[dict]) -> List[dict]
  - fetch_and_tag_live(client: PolymarketClient, ...) -> List[dict]
"""
from __future__ import annotations

import logging
from typing import Iterable, List

from .pm_classifier import classify_market, is_crypto_text

try:
    # Only used in type hints; safe if PolymarketClient is not present at import time in tests
    from .polymarket_client import PolymarketClient  # type: ignore
except Exception:  # noqa: BLE001
    PolymarketClient = object  # type: ignore

logger = logging.getLogger("pm_ingest")


def _synthesize_event_from_market(m: dict) -> dict:
    """
    Market-only records sometimes include event references; normalize what we can.
    """
    return {
        "id": m.get("eventId") or m.get("event_id") or m.get("event") or "",
        "title": m.get("eventTitle") or m.get("eventName") or m.get("event") or m.get("question") or "",
        "description": m.get("eventDescription") or "",
    }


def tag_from_local_markets(markets: Iterable[dict]) -> List[dict]:
    """
    Markets-only dataset → produce standalone TaggedMarket rows.
    We keep only crypto price markets (BTC/ETH/SOL/XRP/DOGE by text).
    """
    out: List[dict] = []
    for m in markets:
        event = _synthesize_event_from_market(m)
        blob = f"{event.get('title','')} {event.get('description','')} {m.get('question','')} {m.get('description','')}"
        if not is_crypto_text(blob):
            continue
        out.append(classify_market(event, m))
    return out


def fetch_and_tag_live(client: "PolymarketClient", *, limit_per_page: int = 250, include_closed: bool = False) -> List[dict]:
    """
    Live path via Gamma:
      Strategy: iterate markets to avoid relying on whether events embed markets in response.
      - Do minimal, safe filtering (closed=false by default)
      - For each market, synthesize an event container and classify
    """
    tagged: List[dict] = []
    count = 0
    for m in client.iter_markets(limit=limit_per_page, include_closed=include_closed):
        count += 1
        # Fast filter: crypto words present somewhere
        text = f"{m.get('question','')} {m.get('description','')} {m.get('eventTitle','')}"
        if not is_crypto_text(text):
            continue
        event = m.get("event") if isinstance(m.get("event"), dict) else _synthesize_event_from_market(m)
        tagged.append(classify_market(event, m))
    logger.info("fetch_and_tag_live: scanned %d markets, tagged %d", count, len(tagged))
    return tagged