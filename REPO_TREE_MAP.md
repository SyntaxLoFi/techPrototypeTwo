# techPrototypeTwo — Repository Tree & Interaction Map

**Last inspected:** 2025-09-11 21:19 UTC

This document summarizes the repository structure and, more importantly, how the scripts interact at runtime. It uses a GitHub‑renderable Mermaid diagram for the execution flow and a concise tree for orientation.

---

## 1) High‑level runtime flow

```mermaid
flowchart TD
    A[main_scanner.py] --> B[Orchestrator]
    A -->|build_scanners()| S1[Scanners per currency]
    S1 -->|attach| C1[OrderbookHandler\n(scripts/data_collection)]
    S1 -->|attach| C2[SpotFeedHandler\n(scripts/data_collection)]
    S1 -->|optional| C3[OptionsChainCollector\n(scripts/data_collection)]
    S1 -->|optional| C4[PerpsDataCollector\n(scripts/data_collection)]
    S1 -->|add| C5[Polymarket contracts\n(PolymarketFetcher / v2 + pm_ingest)]

    B[orchestrator.py\nOrchestrator] --> C[data_refresh.py\nDataRefresh.fetch_all()]
    C -->|updates| C1
    C -->|updates| C2
    C -->|updates| C3
    C -->|updates| C4

    B --> D[hedging/options.py\nOptionHedgeBuilder.build()]
    D -->|strategy assembly| DHB[digital_hedge_builder.py\nbuild_digital_vertical_at_K]
    D -->|ranking| PR[probability_ranker.py]
    D -->|filtering| EVF[expected_value_filter.py]
    D -->|market metrics| MDA[market_data_analyzer.py]

    B --> E[persistence/writer.py\nDefaultWriter.save()]
    E -->|writes| R[(results/*.json)]

    subgraph Utils
      U1[utils.debug_recorder]:::util
      U2[utils.step_debugger]:::util
      U3[utils.log_gate]:::util
      U4[utils.http_client]:::util
    end

    classDef util fill:#f7f7f7,stroke:#999,color:#333

    A -. uses .-> U1
    A -. uses .-> U2
    A -. uses .-> U3
    C -. REST/ws .-> U4
    PR -. config .-> MDA
```

**What it shows:** `main_scanner.py` builds per‑currency scanner objects, then hands control to the `Orchestrator`. The orchestrator refreshes market data, asks the hedger to construct and score opportunities, and finally persists ranked results. Utilities capture debug artifacts and rate‑limit external calls.

---

## 2) Repository tree (selected)

```
techPrototypeTwo/
├─ config/
├─ debug_runs/                 # debug artifacts written at runtime
├─ docs/
├─ hedging/
│  └─ options.py               # OptionHedgeBuilder (hedge construction)
├─ market_data/
│  ├─ options_repository.py    # options snapshot store (read by collectors/hedger)
│  └─ options_quotes_cache.py  # live quotes cache
├─ persistence/
│  └─ writer.py                # DefaultWriter for results/*.json
├─ scripts/
│  └─ data_collection/
│     ├─ polymarket_client.py          # Gamma + CLOB HTTP client
│     ├─ polymarket_fetcher.py         # legacy Polymarket fetch
│     ├─ polymarket_fetcher_v2.py      # tagged fetch (preferred when present)
│     ├─ pm_ingest.py                  # offline tagging of markets
│     ├─ orderbook_handler.py          # order book accumulation/cleanup
│     ├─ spot_feed_handler.py          # spot (delegates to Binance integration)
│     ├─ options_chain_collector.py    # fetch + organize Lyra option chains
│     └─ perps_data_collector.py       # Lyra perpetuals + funding stats
├─ sparse-static-hedge/
├─ strategies/
│  ├─ tag_router.py                    # tag→strategy class resolver
│  └─ options/
│     └─ utils/opt_keys.py             # canonical option keys/helpers
├─ tests/
├─ utils/
│  ├─ debug_recorder.py
│  ├─ http_client.py
│  ├─ log_gate.py
│  └─ step_debugger.py
├─ black_scholes_greeks.py
├─ config_loader.py
├─ config_manager.py
├─ config_schema.py
├─ data_refresh.py
├─ digital_hedge_builder.py
├─ execution_pricing.py
├─ expected_value_filter.py
├─ logger_config.py
├─ main_scanner.py
├─ market_data_analyzer.py
├─ orchestrator.py
├─ probability_ranker.py
├─ ranking_config.py
├─ requirements.txt
└─ simple_websocket_manager.py
```

> **Note:** Tree focuses on modules that participate in the core run path.

---

## 3) Who calls whom (quick map)

- **main_scanner.py**
  - Builds per‑currency `scanners` with: `OrderbookHandler`, `SpotFeedHandler`, optional `OptionsChainCollector`, optional `PerpsDataCollector`, and attaches Polymarket contracts (via `PolymarketFetcher` (v1/v2) with optional `pm_ingest` tagging).
  - Launches the pipeline via `Orchestrator`.

- **orchestrator.py → Orchestrator.run_once()**
  - Calls `DataRefresh.fetch_all()` to update scanner state (spot, orderbooks, options/perps snapshots).
  - Calls `OptionHedgeBuilder.build()` to construct candidate hedges and score them.
  - Calls `DefaultWriter.save()` to persist results into `results/*.json`.

- **data_refresh.py → DataRefresh.fetch_all()**
  - Fills `scanner['current_spot']` (prefers Binance spot).
  - Collects a short websocket snapshot (spot + orderbooks) and feeds messages to `OrderbookHandler.process_message(...)`.
  - If options present, uses their instrument list to subscribe to option orderbooks.

- **hedging/options.py → OptionHedgeBuilder.build()**
  - Iterates attached Polymarket contracts per currency.
  - If option chains available, exposes them to the strategy engine; constructs anchored digital verticals with `digital_hedge_builder.build_digital_vertical_at_K(...)`.
  - Ranks and filters using `ProbabilityRanker` and `ExpectedValueFilter`; enriches with `MarketDataAnalyzer` signals.

- **persistence/writer.py → DefaultWriter.save()**
  - Serializes ranked opportunities (with metrics and required options) to `results/detailed_opportunities_*.json` and a compact ranked summary when configured.

- **scripts/data_collection/**
  - **polymarket_client.py**: Thin HTTP client for Gamma/CLOB (events/markets, books, prices).
  - **polymarket_fetcher(_v2).py** + **pm_ingest.py**: Fetch crypto markets and (optionally) tag markets offline.
  - **orderbook_handler.py**/**spot_feed_handler.py**/**options_chain_collector.py**/**perps_data_collector.py**: Venue adapters that maintain in‑process state for scanners.

- **utils/**
  - `debug_recorder.py` writes one‑per‑run JSON artifacts under `debug_runs/<RUN_ID>/...`
  - `step_debugger.py` checkpoints major pipeline stages (with counts/keys) into `debug_runs/.../checkpoints/*`
  - `log_gate.py` provides gated reason‑code logging and per‑currency snapshot switches
  - `http_client.py` centralizes REST session, retries, and TPS limiting

---

## 4) Data & artifacts written during a run

- Per‑source raw snapshots under `debug_runs/<RUN_ID>/`:
  - `polymarket/*` — contracts, CLOB books/prices (when enabled)
  - `options/*` — Lyra option snapshots
  - `perps/*` — perps snapshots + funding stats
  - `checkpoints/*` — step debugger JSONs + summary
  - `snapshot/` — complete scanner snapshot and post‑hedge opportunities
- Final outputs under `results/`:
  - `detailed_opportunities_YYYYMMDD_HHMMSS.json` (+ optional `ranked_opportunities_*.json`)

---

### Appendix — Notes on configuration

- Logging/setup comes from `logger_config.py` (JSON or text; request‑id injected).
- Config values are unified in `config_manager.py` (and loaded via `config_loader.py` / `config_schema.py`); ranking‑specific knobs in `ranking_config.py`.
