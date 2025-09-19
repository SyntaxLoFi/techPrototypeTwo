# Complete Pipeline Dependency Tree - main_scanner.py

## Overview
This document provides a comprehensive multi-level dependency tree showing ALL scripts called throughout the pipeline, including subdependencies of subdependencies. Every module's imports and what those imports subsequently call are fully mapped to the deepest level.

## Legend
- `→` Direct import/call
- `├──` Branch in dependency tree
- `│` Continuation of tree structure
- `[External]` Third-party library
- `[Built-in]` Python standard library
- `[Optional]` Import with fallback/try-except

## Entry Point: main_scanner.py

```
main_scanner.py
├── [Built-in] asyncio, json, logging, os, datetime, timezone, pathlib, typing
│
├── utils.timebox
│   └── compute_days_to_expiry()
│       └── [Built-in] datetime parsing
│
├── scripts.data_collection.polymarket_client
│   └── PolymarketClient
│       ├── [External] requests
│       │   └── urllib3 (HTTP connection pooling)
│       ├── [Built-in] time, logging, dataclasses
│       └── API calls to Polymarket CLOB endpoints
│
├── scripts.data_collection.pm_ingest
│   ├── load_polymarket_markets()
│   ├── tag_from_local_markets()
│   └── Dependencies:
│       ├── .pm_classifier
│       │   ├── classify_market()
│       │   │   ├── is_crypto_text()
│       │   │   │   └── [Built-in] re.search() with crypto patterns
│       │   │   ├── find_asset()
│       │   │   ├── detect_binance_symbol()
│       │   │   ├── extract_token_ids()
│       │   │   ├── _assign_strategy_categories()
│       │   │   ├── _assign_strategy_eligibility()
│       │   │   └── polymarket_classifier.classify_market_title [Optional]
│       │   └── [Built-in] json, typing
│       ├── market_data.polymarket_gamma
│       │   └── normalize_gamma_market()
│       │       └── market_data.polymarket_price.derive_yes_price_from_gamma()
│       │           └── Gamma outcome price parsing
│       └── [Built-in] datetime, os, json
│
├── strategies.options.expiry_layer (auto-wires multi-expiry)
│   ├── core.expiry_window
│   │   ├── enumerate_expiries()
│   │   └── pm_default_cutoff_for_date_only()
│   │       └── [Built-in] datetime, timezone
│   └── config_loader
│       └── load_config()
│           ├── [External] pydantic (configuration validation)
│           └── [External] yaml (config file parsing)
│
├── logger_config [Optional]
│   ├── setup_logging()
│   │   └── [Built-in] logging configuration
│   └── set_request_id()
│       └── Thread-local storage
│
├── utils.debug_recorder
│   ├── get_recorder()
│   └── RawDataRecorder
│       ├── [Built-in] os, json, threading, logging, datetime, pathlib
│       └── logger_config.set_request_id [Optional]
│
├── utils.log_gate
│   └── configure_from_config()
│       └── [Built-in] logging, os
│
├── utils.step_debugger
│   └── get_step_debugger()
│       └── StepDebugger
│           ├── utils.debug_recorder
│           │   ├── get_recorder()
│           │   └── RawDataRecorder
│           └── [Built-in] json, datetime, pathlib
│
├── config_manager
│   ├── ARBITRAGE_ENABLED_CURRENCIES
│   ├── LYRA_OPTIONS_CURRENCIES
│   ├── LYRA_PERPS_CURRENCIES
│   └── Dependencies:
│       ├── config_loader
│       │   ├── get_config() (cached singleton)
│       │   └── load_config()
│       │       ├── _resolve_config_root()
│       │       │   └── Path traversal logic
│       │       ├── _apply_env_overrides()
│       │       │   └── 180+ environment variable mappings
│       │       ├── [External] yaml.safe_load()
│       │       └── [External] pydantic validation
│       ├── config_schema
│       │   └── AppConfig (root model)
│       │       ├── LoggingSettings
│       │       ├── DebugSettings
│       │       │   ├── DebugCaptureSettings
│       │       │   └── DebugLogSettings
│       │       ├── LyraSettings
│       │       ├── ExecutionSettings
│       │       ├── RankingSettings
│       │       ├── TimeSettings
│       │       ├── DataSettings
│       │       ├── PolymarketConfig
│       │       ├── HedgingSettings
│       │       │   ├── VarianceHedgingSettings
│       │       │   └── QuadraticHedgingSettings
│       │       └── [External] pydantic v2 models
│       │           ├── BaseModel
│       │           ├── Field descriptors
│       │           ├── ConfigDict
│       │           ├── conint, confloat validators
│       │           └── field_validator, model_validator
│       ├── [External] python-dotenv
│       │   └── load_dotenv()
│       └── [External] yaml [Optional]
│           └── yaml.safe_load()
│
└── Main Components (instantiated in main())
    ├── orchestrator.Orchestrator
    ├── data_refresh.DataRefresh
    ├── hedging.opportunity_builder.HedgeOpportunityBuilder
    └── persistence.writer.DefaultWriter
```

## Scanner Building Dependencies (build_scanners function)

```
build_scanners()
├── scripts.data_collection.orderbook_handler
│   └── OrderbookHandler
│       ├── config_manager
│       │   ├── ORDERBOOK_RETENTION_SECONDS
│       │   └── AGGRESSIVE_VALIDATION
│       └── [Built-in] time, logging, collections.OrderedDict, decimal.Decimal
│
├── scripts.data_collection.spot_feed_handler
│   └── SpotFeedHandler
│       └── scripts.data_collection.binance_spot_integration
│           └── get_spot_price()
│               ├── BinanceSpotPriceFeed.get_current_price()
│               └── [Built-in] logging
│
├── scripts.data_collection.options_chain_collector
│   └── OptionsChainCollector
│       ├── config_manager
│       │   ├── LYRA_MAINNET_BASE_URL
│       │   ├── LYRA_EXPIRY_BUFFER_HOURS
│       │   └── MAX_OPTIONS_QUEUE_SIZE
│       ├── utils.http_client
│       │   ├── get()
│       │   │   ├── [External] requests.Session
│       │   │   └── urllib3.util.retry.Retry
│       │   └── post()
│       │       └── Same as get()
│       ├── utils.debug_recorder
│       │   └── get_recorder()
│       ├── market_data.options_quotes_cache
│       │   └── OptionsQuotesCache
│       │       └── [Built-in] threading.Lock
│       ├── market_data.options_repository
│       │   └── OptionsRepository (singleton)
│       │       ├── [Built-in] threading.Lock
│       │       └── [Built-in] json serialization
│       └── black_scholes_greeks
│           └── BlackScholesGreeks
│               ├── [External] scipy.stats.norm
│               └── [Built-in] math functions
│
├── scripts.data_collection.perps_data_collector
│   └── PerpsDataCollector
│       ├── config_manager (various settings)
│       └── utils.http_client
│           └── get(), post()
│
├── scripts.data_collection.polymarket_fetcher
│   └── PolymarketFetcher
│       └── [External] requests
│
└── scripts.data_collection.polymarket_fetcher_v2 [Optional]
    └── PolymarketFetcher (v2)
        └── Tagged market fetching
```

## Orchestrator Dependencies

```
orchestrator.py (Orchestrator class)
├── [Built-in] typing (Protocol definitions)
├── [Built-in] logging
├── [Built-in] json, os, datetime (for save_unfiltered_opportunities)
│
├── strategies.options.expiry_layer
│   └── (Same as main_scanner.py import)
│
├── utils.debug_recorder
│   └── get_recorder()
│       └── (Same as main_scanner.py import)
│
├── utils.log_gate
│   └── per_currency_snapshot_enabled()
│       └── Configuration checking
│
├── utils.step_debugger
│   └── get_step_debugger()
│       └── (Same as main_scanner.py import)
│
└── persistence.writer
    └── NumpyEncoder
        └── [External] numpy [Optional]
            └── Type conversion for numpy arrays
```

## Data Refresh Dependencies

```
data_refresh.py (DataRefresh class)
├── [Built-in] asyncio, logging, time, typing
│
├── scripts.data_collection.derive_ws_client [Optional]
│   └── collect_data_snapshot()
│       └── DeriveWSClient
│           ├── simple_websocket_manager
│           │   └── SimpleWebSocketManager
│           │       ├── [External] websockets
│           │       ├── [External] ssl, certifi
│           │       └── utils.websocket_tracker [Optional]
│           │           ├── track_websocket()
│           │           └── untrack_websocket()
│           └── [Built-in] asyncio, json, re, sys
│
├── simple_websocket_manager [Optional fallback]
│   └── collect_data_snapshot()
│       └── (Same as derive_ws_client dependencies)
│
└── scripts.data_collection.binance_spot_integration
    └── get_spot_price()
        └── BinanceSpotPriceFeed
            ├── [External] websockets
            ├── [External] pandas
            ├── [External] requests
            ├── [External] ssl, certifi
            ├── config_manager
            │   ├── BINANCE_API_KEY
            │   ├── BINANCE_WEBSOCKET_URL
            │   └── Other Binance settings
            ├── utils.websocket_tracker [Optional]
            └── utils.http_client
                └── get(), post()
```

## Hedge Building Dependencies

```
hedging/opportunity_builder.py (HedgeOpportunityBuilder class)
├── [Built-in] os, logging, dataclasses, typing
│
├── strategies.tag_router
│   └── StrategyTagRouter
│       ├── strategies.base_strategy [Optional]
│       │   └── BaseStrategy
│       │       ├── [Built-in] abc.ABC
│       │       └── utils.validation_audit
│       │           └── emit()
│       │               └── [Built-in] json, threading
│       └── strategies.strategy_loader [Optional]
│           └── StrategyLoader
│               ├── [Built-in] importlib, inspect
│               └── Dynamic strategy loading
│
├── digital_hedge_builder
│   └── build_digital_vertical_at_K()
│       ├── core.expiry_window
│       │   ├── enumerate_expiries() (detailed above)
│       │   └── pm_date_to_default_cutoff_utc()
│       │       └── [Built-in] datetime with timezone
│       ├── config_loader
│       │   └── load_config() (detailed above)
│       ├── strategies.options.utils.opt_keys [Optional]
│       │   ├── key_for_is_above()
│       │   └── chain_slice()
│       └── execution_pricing
│           └── option_exec_price()
│               ├── option_apply_fee()
│               │   └── config_manager fee constants
│               ├── black_scholes_greeks.BlackScholesGreeks
│               │   └── call_price(), put_price()
│               ├── [Built-in] math.isfinite
│               └── config_manager execution settings
│
├── config_manager
│   ├── RISK_FREE_RATE
│   ├── LIQUIDITY_SAFETY_FACTOR
│   ├── MIN_POSITION_SIZE
│   ├── MAX_POSITION_SIZE
│   └── get_config()
│
├── utils.log_gate
│   └── reason_debug()
│       └── Conditional logging
│
├── utils.validation_audit
│   └── emit()
│       └── (Same as strategies.base_strategy)
│
├── probability_ranker [Optional]
│   └── ProbabilityRanker
│       ├── rank_opportunities()
│       │   ├── _extract_payoff_profile()
│       │   ├── _get_pm_probability()
│       │   ├── _get_options_implied_probability()
│       │   │   └── black_scholes_greeks.BlackScholesGreeks
│       │   │       ├── [External] scipy.stats.norm
│       │   │       └── [External] numpy
│       │   ├── _blend_probabilities()
│       │   │   └── [External] scipy.interpolate.interp1d
│       │   └── _calculate_metrics()
│       │       └── [External] numpy array operations
│       ├── config_loader.get_config()
│       │   └── load_config()
│       │       ├── [External] yaml.safe_load()
│       │       ├── [External] pydantic.ValidationError
│       │       └── config_schema.AppConfig
│       │           └── Nested pydantic models
│       ├── market_data_analyzer.MarketDataAnalyzer
│       └── [Built-in] json, os, dataclasses, math, logging
│
├── expected_value_filter [Optional]
│   └── ExpectedValueFilter
│       ├── filter_opportunities()
│       │   ├── calculate_expected_value()
│       │   │   └── [External] numpy operations
│       │   ├── calculate_binary_sharpe_ratio()
│       │   │   └── [External] numpy.sqrt
│       │   ├── calculate_kelly_fraction()
│       │   │   └── Mathematical calculations
│       │   └── should_include_opportunity()
│       ├── utils.log_gate.reason_debug()
│       │   └── Conditional logging based on config
│       ├── utils.step_debugger.get_step_debugger()
│       │   └── StepDebugger checkpointing
│       ├── config_manager constants (with fallbacks)
│       └── [Built-in] collections.Counter, logging
│
└── market_data_analyzer [Optional]
    └── MarketDataAnalyzer
        ├── analyze() → calculate_dynamic_liquidity_constraints()
        │   ├── get_market_implied_volatility()
        │   │   └── black_scholes_greeks.BlackScholesGreeks
        │   ├── _analyze_orderbook_liquidity()
        │   ├── _analyze_options_liquidity()
        │   └── [External] numpy statistics
        ├── calculate_implied_volatility_from_prices()
        │   └── Volatility surface fitting
        ├── config_manager constants
        └── [Built-in] datetime, logging, typing
```

## Strategy System Dependencies

```
Strategy Loading and Execution
├── strategies.tag_router.StrategyTagRouter
│   └── instantiate_for_tags()
│       └── strategies.strategy_loader.StrategyLoader
│           └── load_strategies()
│               ├── [Built-in] importlib.import_module()
│               └── Dynamic loading of strategy modules
│
├── strategies.base_strategy.BaseStrategy
│   ├── [Built-in] abc.ABC, abstractmethod
│   └── utils.validation_audit.emit()
│
└── Strategy Implementations
    ├── strategies.variance_swap.VarianceSwapStrategy
    │   ├── pm.contract
    │   │   ├── PMContract.from_market_text()
    │   │   │   └── pm.parsing.parse_market_expiry()
    │   │   │       ├── [Built-in] re (regex matching)
    │   │   │       ├── [Built-in] datetime, timedelta, timezone
    │   │   │       └── Month/timezone mapping dictionaries
    │   │   ├── PMContract.candidate_expiries()
    │   │   └── collect_unique_expiries()
    │   │       └── Date parsing from various formats
    │   ├── filters.option_expiry
    │   │   └── filter_options_by_expiry()
    │   │       ├── collect_unique_expiries()
    │   │       ├── [Built-in] datetime, timedelta, timezone
    │   │       └── Debug logging to JSONL files
    │   └── core.expiry_window
    │       └── enumerate_expiries()
    │           ├── ExpiryCandidate dataclass
    │           ├── [Built-in] dataclasses.asdict
    │           ├── [Built-in] json, os
    │           └── Policy-based expiry filtering
    │
    └── strategies.options.variance_swap_strategy.VarianceSwapStrategy
        ├── strategies.options.base_options_strategy [Optional]
        │   └── BaseOptionsStrategy
        ├── utils.instrument_capture
        │   └── BASE_UNIT
        ├── utils.validation_audit.emit()
        │   └── [Built-in] json, os, threading, datetime
        ├── pm.contract (same as above)
        ├── filters.option_expiry (same as above)
        ├── [External] numpy
        │   └── Array operations, mathematical functions
        └── [Built-in] math, decimal.Decimal, json, pathlib
```

## Persistence Dependencies

```
persistence/writer.py (DefaultWriter class)
├── [Built-in] json, os, datetime, logging
│
└── NumpyEncoder
    └── [External] numpy [Optional]
        └── Handles numpy type conversion
```

## Market Data Dependencies

```
Market Data Modules
├── market_data.options_repository
│   └── OptionsRepository (singleton)
│       ├── [Built-in] dataclasses
│       ├── [Built-in] threading.Lock
│       └── [Built-in] json, os, time
│
├── market_data.options_quotes_cache
│   └── OptionsQuotesCache
│       └── [Built-in] threading.Lock
│
├── market_data.polymarket_gamma
│   └── normalize_gamma_market()
│       └── [Built-in] json parsing
│
└── market_data.polymarket_price
    └── derive_yes_price_from_gamma()
        └── [Built-in] json, typing
```

## Utility Module Dependencies

```
Utility Modules
├── utils.http_client
│   ├── RateLimiter
│   │   └── [Built-in] threading.Lock, time
│   ├── get(), post()
│   │   ├── [External] requests.Session
│   │   └── [External] urllib3.util.retry.Retry
│   └── Session management
│       └── [Built-in] collections.defaultdict
│
├── utils.debug_recorder
│   └── RawDataRecorder
│       ├── [Built-in] os, json, threading.Lock
│       ├── [Built-in] datetime, pathlib
│       └── logger_config.set_request_id [Optional]
│
├── utils.step_debugger
│   └── StepDebugger
│       ├── utils.debug_recorder
│       │   └── get_recorder()
│       └── [Built-in] json, datetime, pathlib
│
├── utils.validation_audit
│   └── emit()
│       └── [Built-in] json, os, threading, datetime
│
├── utils.log_gate
│   ├── configure_from_config()
│   ├── per_currency_snapshot_enabled()
│   └── reason_debug()
│       └── [Built-in] logging, os
│
└── utils.timebox
    └── compute_days_to_expiry()
        └── [Built-in] datetime parsing
```

## External Library Dependencies

```
Third-Party Libraries (Full Dependency Chain)
├── Network & HTTP
│   ├── requests
│   │   └── urllib3
│   │       └── SSL/TLS handling
│   ├── websockets
│   │   └── asyncio integration
│   └── ssl, certifi
│       └── Certificate validation
│
├── Data Processing
│   ├── pandas
│   │   └── numpy (optional)
│   ├── numpy
│   │   └── C extensions for numerical operations
│   └── scipy
│       └── numpy, mathematical functions
│
├── Configuration
│   ├── pydantic
│   │   └── Type validation and settings management
│   ├── yaml
│   │   └── YAML file parsing
│   └── python-dotenv
│       └── Environment variable loading
│
└── Development Tools
    └── typing_extensions
        └── Enhanced type hints
```

## Execution Flow Summary

1. **Startup Phase**
   ```
   main_scanner.py
   ├── Logger configuration
   ├── Configuration loading (yaml, env, pydantic)
   └── Debug recorder initialization
   ```

2. **Scanner Building Phase**
   ```
   build_scanners()
   ├── Polymarket data fetching (v1/v2)
   ├── Market classification and tagging
   ├── Options chain collection
   └── Perps data collection
   ```

3. **Main Processing Loop**
   ```
   Orchestrator.run_once()
   ├── DataRefresh.fetch_all()
   │   ├── Binance spot prices
   │   └── WebSocket orderbook snapshots
   ├── HedgeOpportunityBuilder.build()
   │   ├── Strategy instantiation
   │   ├── Digital hedge construction
   │   ├── Probability ranking
   │   └── Expected value filtering
   └── DefaultWriter.save()
   ```

4. **Data Flow**
   - **Input**: Polymarket contracts → Lyra options → Binance spot prices
   - **Processing**: Strategy evaluation → Hedge construction → Ranking/filtering
   - **Output**: JSON opportunities file with detailed metrics

## Deepest Level Dependencies Summary

### Core Mathematical Libraries
- **scipy**: Used for normal distribution (norm.cdf, norm.pdf) in Black-Scholes calculations and interpolation
- **numpy**: Array operations, mathematical functions throughout probability and pricing calculations
- **math**: Basic mathematical operations, isfinite checks

### Data Validation & Configuration
- **pydantic v2**: Complete configuration validation with nested models, field validators, constraints
- **yaml**: Configuration file parsing with safe_load
- **python-dotenv**: Environment variable loading

### Standard Library (Deepest Usage)
- **datetime/timezone**: Timestamp handling, expiry calculations throughout
- **json**: Data serialization, debug output, JSONL logging
- **threading.Lock**: Thread-safe singleton patterns, concurrent access protection
- **re**: Regular expressions for market classification, pattern matching
- **dataclasses**: Data structures with asdict for serialization
- **pathlib**: File system operations, path resolution
- **decimal.Decimal**: Precise decimal arithmetic for financial calculations
- **collections**: Counter for statistics, OrderedDict for orderbooks
- **abc**: Abstract base classes for strategy framework

## Key Observations

1. **No External Script Executions**: The entire pipeline is Python-based with no subprocess calls
2. **Robust Fallbacks**: Most imports have try-except blocks with fallback implementations  
3. **Thread Safety**: Critical sections use threading.Lock for concurrent access
4. **Debug Infrastructure**: Comprehensive debugging with JSONL output and step-by-step checkpointing
5. **Modular Architecture**: Clear separation between data collection, processing, and strategy layers
6. **Configuration-Driven**: 180+ environment variables mapped through pydantic validation
7. **Type Safety**: Heavy use of type hints and runtime validation with pydantic v2
8. **Mathematical Rigor**: Extensive use of scipy/numpy for options pricing and probability calculations
9. **Deep Nesting**: Some dependencies go 6+ levels deep (e.g., main → orchestrator → hedge builder → strategy → pm.contract → parsing → regex)
10. **Singleton Patterns**: Configuration and market data repositories use module-level caching

This dependency tree represents the COMPLETE call graph of the techPrototypeTwo pipeline, showing ALL levels of script dependencies from entry point to the deepest mathematical operations and standard library calls.