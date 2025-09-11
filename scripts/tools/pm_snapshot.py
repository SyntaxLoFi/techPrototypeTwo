#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI to run local tagging against debug_runs/markets.json and write:
  - debug_runs/pm_tagged.json
  - debug_runs/pm_summary.json
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import List

from scripts.data_collection.pm_ingest import tag_from_local_markets

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEBUG_DIR = os.path.join(REPO_ROOT, "debug_runs")


def _load_markets() -> List[dict]:
    path = os.path.join(DEBUG_DIR, "markets.json")
    if not os.path.exists(path):
        raise SystemExit(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main() -> None:
    markets = _load_markets()
    tagged = tag_from_local_markets(markets)

    out_path = os.path.join(DEBUG_DIR, "pm_tagged.json")
    _write_json(out_path, tagged)

    c_assets = Counter([t.get("asset") for t in tagged if t.get("asset")])
    c_classes = Counter([t.get("marketClass") for t in tagged if t.get("marketClass")])
    by_asset = defaultdict(Counter)
    for t in tagged:
        a = t.get("asset")
        if a:
            by_asset[a][t.get("marketClass")] += 1

    summary = {
        "assets_count": dict(c_assets),
        "class_count": dict(c_classes),
        "by_asset": {a: dict(ct) for a, ct in by_asset.items()},
        "total_markets_tagged": len(tagged),
    }
    sum_path = os.path.join(DEBUG_DIR, "pm_summary.json")
    _write_json(sum_path, summary)

    print("Wrote:", out_path)
    print("Wrote:", sum_path)
    print("Counts by asset:", dict(c_assets))
    print("Counts by class:", dict(c_classes))


if __name__ == "__main__":
    main()