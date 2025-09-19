# Main Scanner Data Flow Documentation

This document describes the expected input and output data structures for each method/checkpoint in the main_scanner pipeline. Each section shows the data transformation that occurs at each stage.

## 1. Polymarket Markets Fetch
**Checkpoint**: `polymarket_markets_raw`  
**Function**: `PolymarketFetcherV2.fetch_tagged_markets_live()` or legacy fetcher

### Output Structure
```json
[
  {
    "eventId": null,
    "eventTitle": "Will Bitcoin reach $1,000,000 by December 31, 2025?",
    "eventDescription": "",
    "marketId": "516861",
    "marketSlug": "will-bitcoin-reach-1000000-by-december-31-2025",
    "question": "Will Bitcoin reach $1,000,000 by December 31, 2025?",
    "marketDescription": "...",
    "endDate": "2025-12-31T12:00:00Z",
    "tokenIds": {
      "YES": "112540911653160777059655478391259433595972605218365763034134019729862917878641",
      "NO": "72957845969259179114974336105989648762775384471357386872640167050913336248574"
    },
    "asset": "BTC",                          // Extracted cryptocurrency
    "marketClass": "SINGLE_THRESHOLD",       // Market type classification
    "binanceSymbol": null,
    "relation": ">=",                        // Threshold relation
    "threshold": 1000000.0,                  // Raw threshold value
    "strike_price": 1000000.0,               // Normalized strike price
    "is_above": true,                        // Direction flag
    "strategies": ["variance_swap", "perpetuals", "options_vanilla"],
    "strategyCategories": {
      "perpetuals": true,
      "options": true,
      "hybrid": true
    },
    "strategyEligibility": {
      "options": {"variance_swap": true},
      "perpetuals": {},
      "hybrid": {}
    },
    "strategyTags": ["options.variance_swap"],  // Hierarchical strategy tags
    "yes_bid": 0.001,
    "yes_bid_qty": 1170419.0,
    "yes_ask": 0.999,
    "yes_ask_qty": 20507195.0,
    "yes_mid": null,
    "yes_price": 0.999,
    "yes_qty": 20507195.0,
    "yes_size": 20507195.0,
    "no_bid": 0.001,
    "no_bid_qty": 20507195.0,
    "no_ask": 0.999,
    "no_ask_qty": 1170419.0,
    "no_mid": null,
    "no_price": 0.999,
    "no_qty": 1170419.0,
    "no_size": 1170419.0
  }
]
```


#### Notes
- When available from the v2 fetcher, top-of-book fields (e.g., `yes_bid/ask`, `no_bid/ask`) may be present inline on each market object.
- Additionally, `_dump_polymarket_clob_debug(...)` persists **debug artifacts** for the current run:
  - `polymarket/token_ids.json`
  - `polymarket/books.json`
  - `polymarket/prices.json`

## 2. Contracts by Currency Grouping
**Checkpoint**: `contracts_by_currency`  
**Function**: `build_scanners()` - grouping logic

### Input
Array of Polymarket contracts (as above)

### Output Structure
```json
{
  "BTC": [
    // Array of BTC-related Polymarket contracts
  ],
  "ETH": [
    // Array of ETH-related Polymarket contracts
  ],
  "SOL": [
    // Array of SOL-related Polymarket contracts
  ],
  "XRP": [
    // Array of XRP-related Polymarket contracts
  ],
  "DOGE": [
    // Array of DOGE-related Polymarket contracts
  ]
}
```

## 3. Scanner Building
**Checkpoint**: `scanners_built`  
**Function**: `build_scanners()` - scanner construction

### Input
- Contracts grouped by currency

### Internals
- `build_scanners()` **instantiates and populates** an `OptionsChainCollector` and a `PerpsDataCollector` for each currency; these are **not upstream inputs**.
- Handlers for spot feeds and order books are also created here.

### Output Structure
```json
{
  "BTC": {
    "currency": "BTC",
    "orderbook_handler": "<OrderbookHandler object>",
    "spot_handler": "<SpotFeedHandler object>",
    "options_collector": "<OptionsChainCollector object>",  // null if no options
    "perps_collector": "<PerpsDataCollector object>",      // null if no perps
    "has_options": true,
    "has_perps": true,
    "current_spot": null,  // Not yet populated
    "contracts": [/* Array of Polymarket contracts for BTC */]
  },
  "ETH": { /* Similar structure */ },
  "SOL": { /* Similar structure */ },
  "XRP": { /* Similar structure */ },
  "DOGE": { /* Similar structure */ }
}
```

## 4. Market Snapshot After Refresh
**Checkpoint**: `market_snapshot_refreshed`  
**Function**: `DataRefresh.fetch_all()`

### Input
Scanners dictionary (as above)

### Output Structure
```json
{
  "BTC": {
    "currency": "BTC",
    "orderbook_handler": "<OrderbookHandler with populated orderbooks>",
    "spot_handler": "<SpotFeedHandler object>",
    "options_collector": "<OptionsChainCollector with data>",
    "perps_collector": "<PerpsDataCollector with data>",
    "has_options": true,
    "has_perps": true,
    "current_spot": 112543.22,  // NOW POPULATED with current spot price
    "contracts": [/* Polymarket contracts */]
  },
  // ... other currencies with current_spot populated
}
```

### Options Collector Data Structure
```json
{
  "all_options": [
    {
      "instrument_type": "option",
      "instrument_name": "ETH-20260327-5400-P",
      "base_currency": "ETH",
      "quote_currency": "USDC",
      "expiry_date": "2026-03-27",
      "strike": 5400.0,
      "type": "put",  // or "call"
      "bid": 123.45,  // May be populated later
      "ask": 124.55,  // May be populated later
      "mid": 124.00,  // May be populated later
      "option_details": {
        "index": "ETH-USD",
        "expiry": 1774598400,
        "strike": "5400",
        "option_type": "P"
      }
    }
  ]
}
```

### Perps Collector Data Structure
```json
{
  "BTC": {
    "instrument_name": "BTC-PERP",
    "currency": "BTC",
    "mark_price": 112543.22,
    "index_price": 112421.4,
    "best_bid": 112517.2,
    "best_ask": 112554.6,
    "spread": 37.4,
    "spread_pct": 0.0332,
    "funding_rate": 0.0000565,  // Current hourly funding
    "funding_rate_8h": 0.000452,
    "avg_funding_rate": 0.0000230,
    "funding_rate_std": 0.0000234
  }
}
```

## 5. Pre-Hedge Build
**Checkpoint**: `pre_hedge_build`  
**Function**: Before `HedgeOpportunityBuilder.build()`

### Input/Output
Same as market snapshot (no transformation, just a checkpoint)

## 6. Post-Hedge Build (Raw Opportunities)
**Checkpoint**: `post_hedge_build` and `hedge_opportunities_raw`  
**Function**: `HedgeOpportunityBuilder.build()`

### Input
Market snapshot with populated spot prices

### Output Structure
```json
[
  {
    "currency": "BTC",
    "hedge_type": "options",
    "strategy": "digital_vertical",  // or strategy class name
    "pm_side": "YES",               // or "NO"
    "short_digital": false,         // true if shorting the digital
    "position_size_usd": 500.0,     // Dynamic position sizing
    "pm_cash_out": 500.0,           // PM investment
    "pm_price": 0.45,               // YES or NO price used
    
    // Payoff structure (CRITICAL - often missing)
    "profit_if_yes": 611.11,        // Profit if YES wins
    "profit_if_no": -500.0,         // Loss if NO wins
    "upfront_cashflow": -850.0,     // Total initial outlay
    
    // Polymarket contract reference
    "polymarket_contract": { /* Full PM contract */ },
    "polymarket": {
      "question": "Will Bitcoin reach $120,000?",
      "strike": 120000.0,
      "yes_price": 0.45,
      "no_price": 0.55,
      "is_above": true,
      "end_date": "2025-01-31T12:00:00Z",
      "days_to_expiry": 21.5
    },
    
    // Options hedge details (if constructed)
    "required_options": [
      {
        "type": "CALL",
        "strike": 119000,
        "contracts": 0.05,
        "action": "BUY"
      },
      {
        "type": "CALL", 
        "strike": 121000,
        "contracts": 0.05,
        "action": "SELL"
      }
    ],
    "digital_width": 2000.0,
    "spread_contracts": 0.05,
    "short_vertical": false,
    "costs": {
      "pm_cash_out": 500.0,
      "option_entry_debit": 350.0,
      "option_entry_credit": 0.0,
      "upfront_cashflow": -850.0
    },
    "required_capital": 850.0,
    "max_profit": 1111.11,
    "max_loss": -850.0
  }
]
```

## 7. Pre-Probability Ranking
**Checkpoint**: `pre_probability_ranking`  
**Function**: Before `ProbabilityRanker.rank_opportunities()`

### Input/Output
Same as post-hedge opportunities (checkpoint only)

## 8. Post-Probability Ranking
**Checkpoint**: `post_probability_ranking`  
**Function**: `ProbabilityRanker.rank_opportunities()`

### Input
Raw opportunities array

### Output Structure (opportunities with metrics added)
```json
[
  {
    // ... all previous fields plus:
    "rank": 1,
    "quality_tier": "HIGH_PROBABILITY",  // or "MODERATE", "LOW", etc.
    "metrics": {
      "prob_of_profit": 0.65,
      "expected_value": 125.50,
      "adjusted_ev": 118.25,         // After transaction costs
      "edge_per_downside": 0.15,
      "dni": -50.0,                  // Distance to no-arbitrage
      "sharpe_ratio": 1.25,
      "kelly_fraction": 0.08,
      "is_true_arbitrage": false
    },
    "probabilities": {
      "pm_yes": 0.45,
      "pm_no": 0.55,
      "options_implied": 0.48,       // If available
      "weighted": 0.465              // Blended probability
    },
    "risk_metrics": {
      "vega_exposure": 125.50,
      "gamma_exposure": -0.05,
      "theta_exposure": -15.25
    }
  }
]
```

## 9. Pre-EV Filter
**Checkpoint**: `pre_ev_filter` and `ev_filter_input`  
**Function**: Before `ExpectedValueFilter.filter_opportunities()`

### Input/Output
Same as ranked opportunities (checkpoint only)

## 10. Post-EV Filter
**Checkpoint**: `post_ev_filter` and `ev_filter_output`  
**Function**: `ExpectedValueFilter.filter_opportunities()`

### Input
Ranked opportunities with metrics

### Output Structure
```json
[
  // Only opportunities that pass filter criteria
  // Each opportunity has additional filter metadata:
  {
    // ... all previous fields plus:
    "metrics": {
      // ... previous metrics plus:
      "filter_code": "PASS",  // or DROP_DOWNSIDE_TOO_LARGE, DROP_EV_BELOW_MIN, etc.
      "inclusion_reason": "Positive EV after costs: $118.25",
      "worst_case": -850.0,
      "best_case": 1111.11,
      "transaction_costs": 7.25,
      "downside_threshold": 1000.0,
      "position_size_used": 500.0,
      "prob_of_loss": 0.55,
      "missing_fields": []  // Track any missing required fields
    }
  }
]
```

### Filter Drop Reasons
- `DROP_NO_PAYOFF_DATA`: Missing profit_if_yes or profit_if_no
- `DROP_DOWNSIDE_TOO_LARGE`: Max loss exceeds threshold
- `DROP_EV_BELOW_MIN`: Expected value too low
- `DROP_SHARPE_BELOW_MIN`: Risk-adjusted return too low
- `DROP_KELLY_BELOW_MIN`: Kelly criterion suggests no position
- `DROP_TRUE_ARB_LOW_PROFIT`: True arbitrage but profit < costs

## 11. Final Opportunities
**Checkpoint**: `final_opportunities` or `final_opportunities_empty`  
**Function**: End of `Orchestrator.run_once()`

### Input/Output
Filtered opportunities array (often empty due to strict filtering)


## 12. Writer Summary Output (Back-Compat)
**Function**: `ArbitrageScanner.save_opportunities(...)`

When final opportunities exist, a compact **summary JSON** is written for downstream consumers and quick inspection. The schema is:

```json
{
  "timestamp": "2025-09-10T12:34:56Z",
  "total_opportunities": 3,
  "opportunities": [
    {
      "rank": 1,
      "quality_tier": "HIGH_PROBABILITY",
      "currency": "BTC",
      "hedge_type": "options",
      "strategy": "digital_vertical",
      "max_profit": 1111.11,
      "max_loss": -850.0,
      "probabilities": {
        "pm_yes": 0.45,
        "pm_no": 0.55,
        "weighted": 0.465
      },
      "probability_metrics": {
        "prob_of_profit": 0.65,
        "expected_value": 125.5,
        "adjusted_ev": 118.25,
        "edge_per_downside": 0.15,
        "distance_to_no_arb": -50.0,
        "is_true_arbitrage": false
      },
      "polymarket": {
        // Derived from the selected Polymarket contract
      }
    }
  ]
}
```

## Common Data Issues Leading to Empty Results

1. **Missing Spot Prices**: `current_spot` is null, causing vega calculation failures
2. **Missing Payoff Fields**: `profit_if_yes` and `profit_if_no` not properly calculated
3. **Overly Strict Filtering**: All opportunities rejected due to:
   - Downside thresholds too restrictive
   - EV requirements too high
   - Missing position size data for transaction cost calculations
4. **Data Type Issues**: String values where floats expected (e.g., "0.45" vs 0.45)

## Debugging Recommendations

1. Check checkpoint files in order to identify where data is lost
2. Verify spot prices are populated after market refresh
3. Ensure payoff calculations include all required fields
4. Review filter thresholds in config/baseline.yaml
5. Track missing_fields in EV filter output