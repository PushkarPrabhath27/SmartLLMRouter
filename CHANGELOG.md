# Changelog

All notable changes to SmartRoute are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-09-25

### Added

- **Phase 0 — Scaffolding & Tooling**:
  - Initial repository layout adhering to systems engineering standards.
  - Modern build configuration with Hatchling (`pyproject.toml`) and `uv`.
  - Continuous integration workflows via GitHub Actions for Python 3.10–3.12.
  - Tooling configuration: `ruff` (linter and formatter), `mypy` strict type checking, and `pytest` with `pytest-cov` and `pytest-asyncio`.

- **Phase 1 — Core Infrastructure, SQLite Storage & Configuration**:
  - Shared domain models and enums: `TaskType`, `ComplexityBucket`, `ClassificationResult`, `RoutingMeta`, `RoutingResult`, `StreamChunk`, `ConversationContext`, and `ProjectReport`.
  - Comprehensive exception hierarchy: `SmartRouteError`, `ConfigError`, `ProviderError`, `ClassificationError`, `StorageError`.
  - Asynchronous SQLite storage engine (`Storage`) with WAL mode, foreign keys, and drop-and-recreate migration strategy.
  - Complete schema tables: `decisions`, `signals`, `reputation`, `adaptations`, and `config_overrides`.
  - Pydantic configuration schemas with environment variable interpolation (`${VAR}` and `$VAR`), validation rules, and built-in domain presets (`general`, `web_dev`, `data_science`).

- **Phase 2 — Heuristic Prompt Classification**:
  - Sub-10ms local heuristic classifier (`HeuristicClassifier`) without external embeddings or network requests.
  - 9 deterministic feature extractors: token counting via `tiktoken`, fenced code ratio, question/instruction detection, multi-step markers, hedge-word ambiguity, domain keyword hints, file path references, urgency markers, and instruction verb density.
  - Precompiled domain keyword sets covering Code, Reasoning, Creative, Summarization, and Translation.
  - Continuous complexity scoring in $[0.0, 1.0]$ with calibrated tier thresholds (`low` < 0.33, `medium` < 0.66, `high` >= 0.66).

- **Phase 3 — Routing Engine, Reputation Math & Explainability**:
  - 4-level routing decision hierarchy: Programmatic override hook $\rightarrow$ YAML rules (exact, contains, regex, path) $\rightarrow$ Adaptive reputation score $\rightarrow$ Default complexity tier mapping.
  - Reputation exponential moving average (EMA) formula with auto-bump threshold checking, minimum call limits, and cooldown tracking.
  - Deterministic explainability reason formatters producing human-readable audit trails for every decision.
  - Pre-flight token cost estimation based on standard provider pricing tables.

- **Phase 4 — Provider Integration & Unified Dispatcher**:
  - Provider abstraction (`BaseProvider`) and async implementations for **OpenAI**, **Anthropic**, and **Groq** using `httpx`.
  - Robust error normalization across HTTP status codes, malformed JSON, rate limits, and network timeouts.
  - `ProviderDispatcher` supporting fallback chains, attempt tracking, and stream failover before the first token.

- **Phase 5 — Implicit Feedback Signals & Reputation Updates**:
  - Multi-pattern implicit feedback detectors:
    - `hard_regen` (-0.3): exact prompt repeated within 30 seconds.
    - `soft_regen` (-0.1): >80% token overlap within 60 seconds.
    - `explicit_correction` (-0.2): short follow-up messages with negative feedback regex patterns across 6 languages (English, Spanish, French, German, Chinese, Japanese).
    - `acceptance` (+0.05): natural conversation continuation.
  - Signal priority resolution guaranteeing strongest signal selection.
  - Asynchronous fire-and-forget `SignalCollector` with background task tracking and failure isolation.

- **Phase 6 — Public API (`Router`), Top-Level Exports & Examples**:
  - Public `Router` class orchestrating the entire lifecycle (`complete`, `stream`, `report`, `reset_reputation`, `report_signal`, `reload_config`).
  - Privacy-preserving decision persistence (storing SHA-256 hashes and 100-character previews; raw prompt text is never stored).
  - Clean top-level package exports in `smartroute/__init__.py`.
  - Comprehensive runnable examples covering basic usage, FastAPI streaming, custom overrides, domain presets, report inspection, manual signal reporting, and reputation resetting.

- **Phase 7 — Release Readiness & Packaging**:
  - Packaged wheel and source distribution bundling `py.typed` and preset YAML data files.
  - 100% test coverage on public APIs and 97% overall test suite coverage across 459 test cases.
