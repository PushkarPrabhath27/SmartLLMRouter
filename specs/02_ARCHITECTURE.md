# SmartRoute — System Architecture

## Design Principles

1. **Async-first**: Every I/O boundary is `async`. No blocking calls in the public API.
2. **Immutable data flow**: Classifier outputs and routing decisions are pure data structures. Side effects are isolated to storage and provider calls.
3. **Fail-open**: If classification fails, route to the safest (highest-tier) model. If a provider fails, fallback to the next tier. Never crash the host app.
4. **Zero external dependencies for core logic**: Only provider SDKs are external. Classifier, router, and signal detection use stdlib + `tiktoken`.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Application                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐      │
│  │  complete() │    │   stream()  │    │  report() CLI   │      │
│  └──────┬──────┘    └──────┬──────┘    └─────────────────┘      │
└─────────┼──────────────────┼───────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SmartRoute Library                       │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │   Config     │───▶│  Classifier  │───▶│  RoutingEngine  │    │
│  │  (YAML +     │    │  (Heuristics)│    │  (Decision +    │    │
│  │   Override)  │    │              │    │   Reputation)   │    │
│  └──────────────┘    └──────────────┘    └────────┬────────┘    │
│                                                    │             │
│  ┌──────────────┐    ┌──────────────┐    ┌────────▼────────┐    │
│  │   Storage    │◀───│   Signal     │◀───│   Provider      │    │
│  │  (SQLite)    │    │  Collector   │    │   Dispatcher    │    │
│  │              │    │              │    │  (OpenAI/       │    │
│  └──────────────┘    └──────────────┘    │   Anthropic/    │    │
│                                           │   Groq)         │    │
│                                           └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow (Single Request)

```
User calls router.complete(prompt, context)
    │
ConfigLoader resolves: override_hook → YAML rules → defaults
    │
Classifier.extract_features(prompt) → FeatureVector
    │
Classifier.classify(FeatureVector) → ClassificationResult
    │
RoutingEngine.decide(ClassificationResult, Config, ReputationTable)
    │   ├── Explicit override? → use it
    │   ├── YAML rule match? → use it
    │   ├── Reputation score < threshold? → bump tier
    │   └── Default tier for (task_type, complexity_bucket)
    │
ProviderDispatcher.call(model, prompt) → RawResponse
    │
SignalCollector.observe(prompt, model, context) → stores decision
    │
Router returns RoutingResult(text, meta)
```

## Post-Response Learning Flow

```
User app calls router.report_signal(signal_type, decision_id, ...)
OR SignalCollector auto-detects from subsequent calls
    │
SignalCollector.update_reputation(bucket, model_tier, signal_value)
    │
Reputation EMA updated in SQLite
    │
If EMA crosses threshold → bucket auto-bumped (logged, not acted on retroactively)
```

## Module Structure

```
smartroute/
├── __init__.py              # Public exports: Router, RoutingResult, StreamChunk
├── router.py                # Main Router class, public API
├── config/
│   ├── __init__.py
│   ├── loader.py            # YAML parsing, env var interpolation, validation
│   ├── schema.py            # Pydantic models for config
│   └── presets/             # Built-in domain presets (YAML files)
│       ├── web_dev.yaml
│       ├── data_science.yaml
│       └── general.yaml
├── classifier/
│   ├── __init__.py
│   ├── features.py          # Feature extraction (token count, code ratio, etc.)
│   ├── classifier.py        # Heuristic scoring + confidence
│   └── domain_keywords.py   # Keyword → task_type mappings
├── routing/
│   ├── __init__.py
│   ├── engine.py            # Decision hierarchy, reputation integration
│   ├── reputation.py        # EMA scoring, threshold checking
│   └── explainability.py    # "Why" string generation
├── providers/
│   ├── __init__.py
│   ├── base.py              # Abstract provider interface
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── groq_provider.py
├── signals/
│   ├── __init__.py
│   ├── collector.py         # Signal detection + storage
│   ├── detectors.py         # Hard regen, soft regen, explicit correction
│   └── reputation_updater.py # EMA update logic
├── storage/
│   ├── __init__.py
│   ├── connection.py        # SQLite connection manager
│   ├── schema.py            # Table definitions
│   ├── decisions.py         # Decision CRUD
│   └── reputation.py        # Reputation CRUD
├── types.py                 # Shared type definitions (dataclasses/enums)
└── exceptions.py            # Custom exceptions
```

## Dependency Graph

```
router.py
├── config/loader.py
├── classifier/classifier.py
├── routing/engine.py
├── providers/*.py
├── signals/collector.py
└── storage/connection.py

config/loader.py
└── config/schema.py

classifier/classifier.py
├── classifier/features.py
└── classifier/domain_keywords.py

routing/engine.py
├── routing/reputation.py
├── routing/explainability.py
└── storage/reputation.py

signals/collector.py
├── signals/detectors.py
├── signals/reputation_updater.py
└── storage/decisions.py
```

## Threading & Concurrency Model

- **SQLite**: Uses `aiosqlite` for async access. All storage operations are async.
- **Provider calls**: Each provider runs its own `httpx.AsyncClient` (or native SDK async client).
- **Router instance**: Thread-safe for reads. Config reload is atomic (swaps reference).
- **Signal collection**: Fire-and-forget async task. Never blocks response delivery.

## Error Handling Strategy

| Failure Mode | Behavior | Log Level |
|--------------|----------|-----------|
| Config file missing | Create default `general` preset | WARNING |
| Config validation error | Raise `ConfigError` on init | ERROR |
| Classifier feature extraction fails | Default to `complexity=0.5, task_type="general"` | WARNING |
| Provider API error | Try next provider in tier. If all fail, raise `ProviderError` | ERROR |
| Signal storage fails | Silently drop (don't crash user app). Log to stderr. | WARNING |
| Reputation update fails | Degrade gracefully. Next request uses stale reputation. | WARNING |

## Extension Points (For V2+)

1. **Custom classifier**: Implement `BaseClassifier`, register in config.
2. **Custom provider**: Implement `BaseProvider`, register in config.
3. **Custom signal detector**: Implement `BaseDetector`, register in config.
4. **Custom storage backend**: Implement `BaseStorage`, swap SQLite for Postgres/Redis.
