# SmartRoute — Step-by-Step Implementation Guide

## Pre-Flight Checklist

Before writing any code:
- [ ] Python 3.10+ installed
- [ ] `uv` or `pip` available
- [ ] Git repo initialized
- [ ] All spec files read and understood

## Step 1: Project Bootstrap (Phase 0)

```bash
mkdir smartroute && cd smartroute
git init

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "smartroute"
version = "0.1.0"
description = "Adaptive LLM routing library with implicit feedback learning"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
dependencies = [
    "aiosqlite>=0.20.0",
    "tiktoken>=0.7.0",
    "pydantic>=2.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "ruff>=0.5.0",
    "mypy>=1.10",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
EOF

# Create directory structure
mkdir -p smartroute/{config/presets,classifier,routing,providers,signals,storage}
mkdir -p tests/data examples

# Create __init__.py files
touch smartroute/__init__.py
touch smartroute/{config,classifier,routing,providers,signals,storage}/__init__.py

# Create CI
mkdir -p .github/workflows
cat > .github/workflows/ci.yml << 'EOF'
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv pip install -e ".[dev]"
      - run: ruff check .
      - run: ruff format --check .
      - run: mypy --strict smartroute/
      - run: pytest --cov=smartroute --cov-report=xml --cov-fail-under=90
EOF

git add .
git commit -m "chore(project): initial scaffolding"
```

## Step 2: Types & Exceptions (Phase 1)

Create `smartroute/types.py`:

```python
"""Shared type definitions for SmartRoute."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskType(Enum):
    CODE = "code"
    CREATIVE = "creative"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    GENERAL = "general"


class ComplexityBucket(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ClassificationResult:
    task_type: TaskType
    complexity: float
    confidence: float
    features: dict[str, Any]


@dataclass(frozen=True)
class RoutingMeta:
    model: str
    task_type: str
    complexity: float
    complexity_bucket: str
    confidence: float
    reason: str
    reputation_score: float
    was_adapted: bool
    override_applied: Optional[str]
    estimated_cost_usd: float
    latency_ms: int
    decision_id: str


@dataclass(frozen=True)
class RoutingResult:
    text: str
    meta: RoutingMeta


@dataclass(frozen=True)
class StreamChunk:
    text: str
    is_finished: bool
    meta: Optional[RoutingMeta] = None


@dataclass
class ConversationContext:
    """Lightweight conversation state. V1 stores but does not escalate."""
    conversation_id: Optional[str] = None
    turn_number: int = 0
    previous_decision_ids: list[str] = field(default_factory=list)


@dataclass
class ProjectReport:
    total_decisions: int
    total_cost_usd: float
    average_latency_ms: float
    model_distribution: dict[str, int]
    bucket_distribution: dict[str, int]
    adapted_buckets: list[dict]
    recent_decisions: list[dict]
```

Create `smartroute/exceptions.py`:

```python
"""Custom exceptions for SmartRoute."""


class SmartRouteError(Exception):
    """Base exception for all SmartRoute errors."""
    pass


class ConfigError(SmartRouteError):
    """Configuration is missing or invalid."""
    pass


class ProviderError(SmartRouteError):
    """All providers failed."""

    def __init__(self, message: str, attempts: list[dict]):
        self.attempts = attempts
        super().__init__(message)


class ClassificationError(SmartRouteError):
    """Classifier failed and fail-open could not recover."""
    pass


class StorageError(SmartRouteError):
    """SQLite operation failed."""
    pass
```

## Step 3: Storage Layer (Phase 1)

Implement in this order:
1. `smartroute/storage/schema.py` — SQL DDL strings
2. `smartroute/storage/connection.py` — `Storage` class with `connect()` and `close()`
3. `smartroute/storage/decisions.py` — DecisionRecord dataclass + CRUD
4. `smartroute/storage/reputation.py` — ReputationRecord + CRUD

Key implementation detail: Use `aiosqlite`. Enable WAL mode in `connect()`:

```python
await self._conn.execute("PRAGMA journal_mode=WAL")
await self._conn.execute("PRAGMA foreign_keys=ON")
```

## Step 4: Config Layer (Phase 1)

Implement:
1. `smartroute/config/schema.py` — Pydantic models
2. `smartroute/config/loader.py` — YAML + env interpolation + validation
3. `smartroute/config/presets/*.yaml` — Three preset files

Env interpolation regex:

```python
import re
ENV_PATTERN = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
```

## Step 5: Classifier (Phase 2)

Implement:
1. `smartroute/classifier/domain_keywords.py` — Dictionaries
2. `smartroute/classifier/features.py` — 9 extractors
3. `smartroute/classifier/classifier.py` — Orchestrator

Performance target: Use `tiktoken` caching:

```python
import tiktoken

_encoders: dict[str, tiktoken.Encoding] = {}

def get_encoder(model_name: str = "gpt-4o-mini") -> tiktoken.Encoding:
    if model_name not in _encoders:
        _encoders[model_name] = tiktoken.encoding_for_model(model_name)
    return _encoders[model_name]
```

## Step 6: Routing Engine (Phase 3)

Implement:
1. `smartroute/routing/reputation.py` — EMA math
2. `smartroute/routing/explainability.py` — String templates
3. `smartroute/routing/engine.py` — Decision hierarchy

Explainability template:

```python
REASON_TEMPLATES = {
    "override_hook": "Programmatic override: forced to {model}",
    "yaml_exact": "YAML rule matched: exact string '{match}' -> {model}",
    "yaml_contains": "YAML rule matched: prompt contains '{match}' -> {model}",
    "yaml_regex": "YAML rule matched: regex /{match}/ -> {model}",
    "yaml_path": "YAML rule matched: path '{match}' -> {model}",
    "adaptation": "Adaptive: reputation {score:.2f} below threshold, bumped to {tier}",
    "default": "Default routing: {task} task, {bucket} complexity ({complexity:.2f}) -> {model}",
    "fallback": "Primary provider failed, fallback to {model} after {error}",
}
```

## Step 7: Providers (Phase 4)

Implement:
1. `smartroute/providers/base.py` — Abstract class
2. `smartroute/providers/openai_provider.py`
3. `smartroute/providers/anthropic_provider.py`
4. `smartroute/providers/groq_provider.py`
5. `smartroute/providers/dispatcher.py` — Fallback execution

Streaming pattern:

```python
async def stream(self, prompt: str, model: str) -> AsyncIterator[StreamChunk]:
    start_time = time.perf_counter()
    # ... API call ...
    async for chunk in api_response:
        yield StreamChunk(text=chunk.text, is_finished=False, meta=None)

    yield StreamChunk(
        text="",
        is_finished=True,
        meta=RoutingMeta(..., latency_ms=int((time.perf_counter() - start_time) * 1000))
    )
```

## Step 8: Signal Collection (Phase 5)

Implement:
1. `smartroute/signals/detectors.py` — 4 detectors
2. `smartroute/signals/reputation_updater.py` — EMA update
3. `smartroute/signals/collector.py` — Orchestration

Fire-and-forget pattern:

```python
async def _store_signal_safe(self, signal: SignalRecord) -> None:
    try:
        await self.storage.store_signal(signal)
    except Exception as e:
        logger.warning(f"Signal storage failed: {e}")

# In the hot path:
asyncio.create_task(self._store_signal_safe(signal))
```

## Step 9: Public API (Phase 6)

Implement `smartroute/router.py`:

```python
class Router:
    def __init__(self, ...):
        self._config = ...
        self._storage = ...
        self._classifier = ...
        self._engine = ...
        self._collector = ...
        self._providers = ...

    async def complete(self, prompt: str, context=None) -> RoutingResult:
        # 1. Detect signals from context
        # 2. Classify
        # 3. Route
        # 4. Call provider
        # 5. Store decision
        # 6. Return result

    async def stream(self, prompt: str, context=None) -> AsyncIterator[StreamChunk]:
        # Same as complete but yield chunks
```

## Step 10: Polish (Phase 7)

Run in this exact order:

```bash
ruff check smartroute/ tests/
ruff format smartroute/ tests/
mypy --strict smartroute/
pytest --cov=smartroute --cov-report=html --cov-fail-under=90
```

Fix until all pass.

## Common Pitfalls

1. SQLite in async: Always use `aiosqlite`, never `sqlite3` in async context.
2. Streaming meta: Only the final chunk has `meta`. Intermediate chunks have `meta=None`.
3. Provider keys: Internal keys are `"openai"`, `"anthropic"`, `"groq"`. Model strings are `"openai/gpt-4o-mini"`.
4. EMA initial value: 0.5 (neutral), not 0.0.
5. Config search path: Check cwd first, then home directory.
6. Prompt hashing: Use `hashlib.sha256(prompt.encode()).hexdigest()` for privacy.
