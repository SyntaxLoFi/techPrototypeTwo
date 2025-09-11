#!/usr/bin/env python3
"""
diagnose_clob_tokens.py

Quick, focused Polymarket CLOB diagnostic.

It verifies three things:
  1) Token ID resolution (YES/NO present for each market)
  2) Orderbook coverage from CLOB /books and /book (prices + sizes)
  3) Whether markets are two-sided (optionally with min size / notional)

Usage:
  python predictionMarketVarianceStrategy/scripts/tools/diagnose_clob_tokens.py --sample 20
  # or as a module (if namespace package works on your env):
  python -m predictionMarketVarianceStrategy.scripts.tools.diagnose_clob_tokens --sample 20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# Ensure repo root is on PYTHONPATH when running as a script
ROOT = Path(__file__).resolve().parents[2]  # predictionMarketVarianceStrategy/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logger_config import setup_logging  # type: ignore
from scripts.data_collection.polymarket_fetcher import PolymarketFetcher  # type: ignore

try:
    from tabulate import tabulate
except Exception:
    tabulate = None

@dataclass
class Coverage:
    markets_total: int = 0
    with_yes_no_ids: int = 0
    tokens_total: int = 0
    tokens_with_quotes: int = 0
    tokens_with_sizes: int = 0
    markets_two_sided_price: int = 0
    markets_two_sided_sized: int = 0

def _two_sided_prices(c: Dict, side: str) -> bool:
    bb = float(c.get(f"{side}_best_bid") or 0)
    ba = float(c.get(f"{side}_best_ask") or 0)
    return bb > 0 and ba > 0

def _two_sided_sized(c: Dict, side: str, min_size: float = 0.0, min_notional: float = 0.0) -> bool:
    bb = float(c.get(f"{side}_best_bid") or 0)
    ba = float(c.get(f"{side}_best_ask") or 0)
    bq = float(c.get(f"{side}_best_bid_size") or 0)
    aq = float(c.get(f"{side}_best_ask_size") or 0)
    if not (bb > 0 and ba > 0):
        return False
    if not (bq > min_size and aq > min_size):
        return False
    if min_notional > 0.0 and ((bb*bq) < min_notional or (ba*aq) < min_notional):
        return False
    return True

def _has_ids(c: Dict) -> bool:
    return bool(c.get("yes_token_id") or c.get("no_token_id") or c.get("clob_token_ids"))

def summarize(contracts: List[Dict], *, min_size: float, min_notional: float, limit: int | None = None) -> Coverage:
    cov = Coverage()
    cov.markets_total = len(contracts)
    for c in (contracts if limit is None else contracts[:limit]):
        if _has_ids(c):
            cov.with_yes_no_ids += 1

        # Count tokens with quotes and sizes
        for side in ("yes", "no"):
            bb = float(c.get(f"{side}_best_bid") or 0)
            ba = float(c.get(f"{side}_best_ask") or 0)
            bq = float(c.get(f"{side}_best_bid_size") or 0)
            aq = float(c.get(f"{side}_best_ask_size") or 0)
            cov.tokens_total += 1
            if bb > 0 or ba > 0:
                cov.tokens_with_quotes += 1
            if ((bb > 0 or ba > 0) and (bq > 0 or aq > 0)):
                cov.tokens_with_sizes += 1

        # Two-sided tests (per market)
        price_yes = _two_sided_prices(c, "yes")
        price_no  = _two_sided_prices(c, "no")
        size_yes  = _two_sided_sized(c, "yes", min_size=min_size, min_notional=min_notional)
        size_no   = _two_sided_sized(c, "no",  min_size=min_size, min_notional=min_notional)
        if price_yes and price_no:
            cov.markets_two_sided_price += 1
        if size_yes and size_no:
            cov.markets_two_sided_sized += 1
    return cov

def main():
    parser = argparse.ArgumentParser(description="Polymarket CLOB diagnostic")
    parser.add_argument("--sample", type=int, default=12, help="How many markets to print in the sample table")
    parser.add_argument("--min-size", type=float, default=0.0, help="Minimum size per side to count as sized two-sided")
    parser.add_argument("--min-notional", type=float, default=0.0, help="Minimum notional per side to count as sized two-sided")
    parser.add_argument("--limit", type=int, default=0, help="Limit how many markets to process (0 = all)")
    parser.add_argument("--json", action="store_true", help="Print JSON for the sample rows instead of a table")
    args = parser.parse_args()

    # Logging
    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"), to_file=False, json_logs=False)
    log = logging.getLogger("clob.diagnostic")

    # Show which endpoints we are about to hit
    api_base = os.getenv("POLYMARKET_API_BASE", "https://gamma-api.polymarket.com")
    clob_base = os.getenv("POLYMARKET_CLOB_BASE", "https://clob.polymarket.com")
    log.info(f"Using endpoints: POLYMARKET_API_BASE={api_base} POLYMARKET_CLOB_BASE={clob_base}")

    f = PolymarketFetcher()

    # 1) Fetch events and parse contracts
    events = f.fetch_crypto_events()
    log.info(f"Fetched {len(events)} crypto events")

    contracts = f.parse_contracts(events)
    log.info(f"Parsed {len(contracts)} contracts")

    # Optional limit
    if args.limit and args.limit > 0:
        contracts = contracts[: args.limit]

    # 2) Enrich with books (adds best_bid/best_ask and sizes per YES/NO)
    contracts = f.enrich_with_books(contracts)

    # 3) Summarize coverage
    cov = summarize(contracts, min_size=args.min_size, min_notional=args.min_notional)

    print("\n=== CLOB Coverage Summary ===")
    print(f"Markets total:             {cov.markets_total}")
    print(f"Markets with YES/NO IDs:   {cov.with_yes_no_ids}")
    print(f"Tokens total (2 per mkt):  {cov.tokens_total}")
    print(f"Tokens w/ quotes:          {cov.tokens_with_quotes}")
    print(f"Tokens w/ sizes:           {cov.tokens_with_sizes}")
    print(f"Markets two-sided (price): {cov.markets_two_sided_price}")
    print(f"Markets two-sided (sized): {cov.markets_two_sided_sized}")

    # 4) Print a sample table
    N = max(1, args.sample)
    sample = contracts[:N]
    rows: List[List[object]] = []
    for c in sample:
        rows.append([
            (c.get("event_title") or "")[:28],
            (c.get("question") or "")[:40],
            c.get("currency") or "",
            c.get("market_id") or c.get("id") or "",
            c.get("yes_token_id") or "",
            c.get("no_token_id") or "",
            float(c.get("yes_best_bid") or 0),
            float(c.get("yes_best_ask") or 0),
            float(c.get("no_best_bid") or 0),
            float(c.get("no_best_ask") or 0),
            float(c.get("yes_best_bid_size") or 0),
            float(c.get("yes_best_ask_size") or 0),
            float(c.get("no_best_bid_size") or 0),
            float(c.get("no_best_ask_size") or 0),
        ])

    headers = [
        "event", "question", "currency", "market_id",
        "YES_token", "NO_token",
        "YES_bid", "YES_ask", "NO_bid", "NO_ask",
        "YES_bsz", "YES_asz", "NO_bsz", "NO_asz",
    ]

    print("\n=== Sample Top-of-Book (first {} markets) ===".format(N))
    if args.json or (tabulate is None):
        print(json.dumps({"headers": headers, "rows": rows}, indent=2))
    else:
        print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f"))

    # Exit code hints for CI runs
    if cov.markets_total == 0:
        log.error("No markets were parsed. This usually means the upstream schema changed.")
        sys.exit(2)
    if cov.with_yes_no_ids == 0:
        log.error("No markets have YES/NO token IDs. Token resolution is broken.")
        sys.exit(3)
    if cov.tokens_with_quotes == 0:
        log.error("No tokens have quotes. CLOB /books or /book may be failing.")
        sys.exit(4)

if __name__ == "__main__":
    main()