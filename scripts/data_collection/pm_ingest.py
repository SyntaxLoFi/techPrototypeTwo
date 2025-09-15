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
from datetime import datetime, timezone
from typing import Optional
import os
from typing import List, Dict, Any
from market_data.polymarket_gamma import load_polymarket_gamma_normalized
import json

try:
    # Only used in type hints; safe if PolymarketClient is not present at import time in tests
    from .polymarket_client import PolymarketClient  # type: ignore
except Exception:  # noqa: BLE001
    PolymarketClient = object  # type: ignore

logger = logging.getLogger("pm_ingest")

def compute_days_to_expiry(end_iso_or_ts, now: Optional[datetime]=None) -> Optional[float]:
    """Return non-negative days from now (UTC) to end date, or None if invalid."""
    if not end_iso_or_ts:
        return None
    try:
        s = str(end_iso_or_ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)  # handles offset form
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = now or datetime.now(timezone.utc)
        delta = (dt.astimezone(timezone.utc) - now).total_seconds() / 86400.0
        return max(delta, 0.0)
    except Exception:
        return None


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


def load_polymarket_markets(*args, **kwargs) -> List[Dict[str, Any]]:
    """
    Gamma-only ingestion wrapper.
    Pulls markets from Gamma List-markets and returns normalized dicts with:
      pm_market_id, conditionId, clobTokenIds, endDate, days_to_expiry,
      yes_price, no_price, price_source="gamma_outcome",
      pm_last_trade_price, pm_best_bid, pm_best_ask
    """
    # Default: strictly Gamma
    use_clob = os.getenv("PM_USE_CLOB", "0") == "1"
    if use_clob:
        # Diagnostic-only, but we NEVER assign prices from CLOB here.
        # (Keep any legacy code behind this gate. Do not overwrite y/n prices.)
        pass
    markets = load_polymarket_gamma_normalized()
    return markets


def write_polymarket_snapshot(out_path: str = "debug_runs/latest/polymarket/pm_tagged_live.json") -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    normalized = load_polymarket_markets()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    return out_path


def fetch_and_tag_live(client: "PolymarketClient", *, limit_per_page: int = 250, include_closed: bool = False) -> List[dict]:
    """
    Live path via Gamma:
      Strategy: iterate markets to avoid relying on whether events embed markets in response.
      - Do minimal, safe filtering (closed=false by default)
      - For each market, synthesize an event container and classify
    """
    tagged: List[dict] = []
    count = 0
    for gm in client.iter_markets(limit=limit_per_page, include_closed=include_closed):
        count += 1
        # Fast filter: crypto words present somewhere
        text = f"{gm.get('question','')} {gm.get('description','')} {gm.get('eventTitle','')}"
        if not is_crypto_text(text):
            continue
        event = gm.get("event") if isinstance(gm.get("event"), dict) else _synthesize_event_from_market(gm)
        m_out = classify_market(event, gm)
        # ---- Canonical stamping from Gamma ----
        m_out["pm_market_id"] = gm.get("id")
        m_out["conditionId"]   = gm.get("conditionId")
        ctids = gm.get("clobTokenIds")
        m_out["clobTokenIds"]  = ctids if isinstance(ctids, list) else ([ctids] if ctids else [])
        m_out["pm_question"]   = gm.get("question")
        m_out["endDate"]       = gm.get("endDate") or gm.get("endDateIso") or gm.get("end_date")
        dte = compute_days_to_expiry(m_out.get("endDate"))
        if dte is not None:
            m_out["days_to_expiry"] = float(dte)
        # ---- Preferred prices from Gamma outcomePrices ----
        try:
            ops = gm.get("outcomePrices") or []
            outs = gm.get("outcomes") or []
            # Map common YES/NO ordering; be conservative if shapes mismatch
            if isinstance(ops, list) and len(ops) >= 2:
                # assume outcomes[0] ↔ YES, outcomes[1] ↔ NO when labeled, else treat [0]=YES,[1]=NO
                yes_idx = 0
                no_idx  = 1 if len(ops) > 1 else 0
                m_out.setdefault("yes_price", float(ops[yes_idx]))
                m_out.setdefault("no_price",  float(ops[no_idx]))
                m_out["price_source"] = "gamma_outcome"
        except Exception:
            pass
        tagged.append(m_out)
    logger.info("fetch_and_tag_live: scanned %d markets, tagged %d", count, len(tagged))
    return tagged