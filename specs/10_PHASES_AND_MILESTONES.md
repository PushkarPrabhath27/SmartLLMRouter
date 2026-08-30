# SmartRoute — Build Phases & Milestones

> **Total estimated time:** 3-4 weeks for a solo developer working 3-4 hours/day.
> **Rule:** Do not start Phase N+1 until Phase N exit criteria are met.

---

## Phase 0: Project Scaffolding (Days 1-2)

### Goal
A clean, professional Python package that passes CI before any business logic exists.

### Tasks
- [ ] Initialize `pyproject.toml` with:
  - Package name: `smartroute`
  - Version: `0.1.0`
  - Python requirement: `>=3.10`
  - Dependencies: `aiosqlite`, `tiktoken`, `pydantic`, `httpx`
  - Dev dependencies: `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `pytest-cov`
- [ ] Create directory structure per `ARCHITECTURE.md`
- [ ] Set up `ruff` config (line length 100, strict imports)
- [ ] Set up `mypy --strict` config
- [ ] Create GitHub Actions workflow:
  - Run on Python 3.10, 3.11, 3.12
  - `ruff check .`
  - `ruff format --check .`
  - `mypy smartroute/`
  - `pytest --cov=smartroute --cov-report=xml`
- [ ] Write `README.md` skeleton (install, quickstart, config reference)
- [ ] Write `CONTRIBUTING.md` (for future contributors)
- [ ] Add `.gitignore`, `LICENSE` (MIT)

### Exit Criteria
- [ ] `git push` passes CI green on all Python versions
- [ ] `pip install -e .` works in a fresh venv
- [ ] `pytest` runs (even if zero tests exist yet)

---

## Phase 1: Core Infrastructure (Days 3-5)

### Goal
Storage, config, types, and exceptions are fully implemented and tested.

### Tasks
- [ ] Implement `smartroute/types.py` — all dataclasses and enums
- [ ] Implement `smartroute/exceptions.py` — exception hierarchy
- [ ] Implement `smartroute/storage/schema.py` — SQL DDL
- [ ] Implement `smartroute/storage/connection.py` — aiosqlite wrapper
- [ ] Implement `smartroute/storage/decisions.py` — DecisionRecord CRUD
- [ ] Implement `smartroute/storage/reputation.py` — ReputationRecord CRUD
- [ ] Implement `smartroute/config/schema.py` — Pydantic Config model
- [ ] Implement `smartroute/config/loader.py` — YAML parsing, env interpolation, preset merging
- [ ] Implement `smartroute/config/presets/` — general.yaml, web_dev.yaml, data_science.yaml
- [ ] Write unit tests for all storage operations (use `:memory:` SQLite)
- [ ] Write unit tests for config loading and validation

### Exit Criteria
- [ ] All storage CRUD operations tested with >90% coverage
- [ ] Config validation catches all invalid configs with clear error messages
- [ ] Preset merging works correctly (user overrides win)
- [ ] `Router()` can initialize with no arguments (creates default preset)

---

## Phase 2: Classifier (Days 6-8)

### Goal
Heuristic classifier extracts 9 features and produces accurate task_type + complexity.

### Tasks
- [ ] Implement `smartroute/classifier/domain_keywords.py` — keyword dictionaries
- [ ] Implement `smartroute/classifier/features.py` — all 9 feature extractors
- [ ] Implement `smartroute/classifier/classifier.py` — scoring algorithm, confidence
- [ ] Create hand-labeled test set: 50 prompts with expected task_type and complexity_bucket
- [ ] Write unit tests for each feature extractor
- [ ] Write integration test: classifier accuracy >70% on task_type, >60% on complexity_bucket
- [ ] Test fail-open behavior (simulate tiktoken failure)

### Exit Criteria
- [ ] Classifier runs in <10ms for 1000-token prompts
- [ ] Accuracy on hand-labeled set meets thresholds
- [ ] No external network calls during classification
- [ ] All edge cases handled (empty string, only code, mixed language)

---

## Phase 3: Routing Engine (Days 9-11)

### Goal
Decision hierarchy, reputation system, and explainability are fully functional.

### Tasks
- [ ] Implement `smartroute/routing/reputation.py` — EMA update, threshold checking, cooldown
- [ ] Implement `smartroute/routing/explainability.py` — reason string generation
- [ ] Implement `smartroute/routing/engine.py` — full decision hierarchy
- [ ] Implement provider fallback chain logic
- [ ] Write unit tests for each decision level (override, YAML, reputation, default)
- [ ] Write integration test: reputation bump after 3 negative signals
- [ ] Test explainability strings for all scenarios

### Exit Criteria
- [ ] Override hook fires correctly and is first priority
- [ ] YAML rules match all 4 match types
- [ ] EMA updates correctly after signals
- [ ] Auto-bump fires when EMA < 0.3 and call_count >= 10
- [ ] Cooldown prevents double-bump within 5 minutes
- [ ] Explainability strings are human-readable and accurate

---

## Phase 4: Provider Integration (Days 12-14)

### Goal
Unified provider interface with 3 providers, async streaming, and fallback.

### Tasks
- [ ] Implement `smartroute/providers/base.py` — Abstract base class
- [ ] Implement `smartroute/providers/openai_provider.py`
- [ ] Implement `smartroute/providers/anthropic_provider.py`
- [ ] Implement `smartroute/providers/groq_provider.py`
- [ ] Implement `smartroute/providers/dispatcher.py` — fallback chain execution
- [ ] Write mock provider for testing (returns deterministic responses)
- [ ] Write integration tests with mocked providers
- [ ] Test fallback: primary fails -> secondary succeeds
- [ ] Test streaming: chunks arrive in order, final chunk has meta

### Exit Criteria
- [ ] All 3 providers implement `complete()` and `stream()`
- [ ] Fallback chain works with mocked failures
- [ ] Streaming produces correct `StreamChunk` sequence
- [ ] Latency is recorded for every successful call
- [ ] Cost estimation is within 2x of actual (acceptable for v1)

---

## Phase 5: Signal Collection (Days 15-17)

### Goal
Implicit feedback loop is closed: signals are detected, stored, and update reputation.

### Tasks
- [ ] Implement `smartroute/signals/detectors.py` — hard_regen, soft_regen, explicit_correction
- [ ] Implement `smartroute/signals/reputation_updater.py` — EMA update logic
- [ ] Implement `smartroute/signals/collector.py` — orchestration
- [ ] Implement `router.report_signal()` manual API
- [ ] Write unit tests for each detector (10 positive, 10 negative cases each)
- [ ] Write integration test: full loop from prompt -> signal -> reputation update
- [ ] Test fire-and-forget: storage failure does not block response

### Exit Criteria
- [ ] Hard regeneration detected within 30s window
- [ ] Soft regeneration detected with >80% overlap
- [ ] Explicit correction detected in all 5 languages
- [ ] EMA converges to <0.3 after 10 hard_regens
- [ ] Signal storage is fire-and-forget (never blocks)

---

## Phase 6: Public API & Examples (Days 18-20)

### Goal
`Router` class is complete with all public methods, examples, and documentation.

### Tasks
- [ ] Implement `smartroute/router.py` — `Router.complete()`, `Router.stream()`, `Router.report()`, `Router.reset_reputation()`
- [ ] Implement `smartroute/__init__.py` — clean public exports
- [ ] Write `examples/basic_usage.py`
- [ ] Write `examples/fastapi_streaming.py`
- [ ] Write `examples/custom_override.py`
- [ ] Write `examples/domain_preset.py`
- [ ] Write `examples/report_reading.py`
- [ ] Complete `README.md` with all examples
- [ ] Add docstrings to all public methods (Google style)

### Exit Criteria
- [ ] All examples run without errors
- [ ] `Router` API matches `API_SPEC.md` exactly
- [ ] README is copy-paste runnable
- [ ] All public methods have docstrings

---

## Phase 7: Polish & Ship (Days 21-24)

### Goal
Production-ready package with clean code, full test coverage, and no tech debt.

### Tasks
- [ ] Run `ruff check .` — fix all warnings
- [ ] Run `ruff format .`
- [ ] Run `mypy --strict smartroute/` — fix all type errors
- [ ] Achieve >90% test coverage
- [ ] Write `CHANGELOG.md` for v0.1.0
- [ ] Tag `v0.1.0` release
- [ ] Build and verify wheel: `python -m build`
- [ ] Test install from wheel in fresh venv
- [ ] Post on relevant forums (Reddit r/MachineLearning, HN, Twitter)

### Exit Criteria
- [ ] CI passes green
- [ ] Coverage >90%
- [ ] `pip install smartroute` works from TestPyPI
- [ ] All P0 features from `V1_SCOPE.md` are implemented
- [ ] No P1/P2 code exists in the repo

---

## Post-V1 Roadmap (For Reference Only)

### V1.5 (Month 2-3)
- Shadow mode
- Pre-flight cost estimation
- Latency-aware routing
- Structured output detection
- HTML dashboard generator
- More providers (Google, Cohere)

### V2 (Month 4-6)
- Embedding-based classifier
- Semantic caching
- Team-wide routing profiles
- Drift detection
- Hosted/cloud option
