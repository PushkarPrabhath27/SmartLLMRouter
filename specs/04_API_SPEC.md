# SmartRoute — Public API Specification

## Core Types

```python
from dataclasses import dataclass
from typing import Optional, AsyncIterator, Callable, Any
from enum import Enum

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
    complexity: float        # 0.0 to 1.0
    confidence: float        # 0.0 to 1.0
    features: dict[str, Any] # raw feature vector for debugging

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

@dataclass(frozen=True)
class RoutingResult:
    text: str
    meta: RoutingMeta

@dataclass(frozen=True)
class StreamChunk:
    text: str
    is_finished: bool
    meta: Optional[RoutingMeta]  # only present on is_finished=True

@dataclass
class ConversationContext:
    """Lightweight conversation state. V1 stores but does not escalate."""
    conversation_id: Optional[str] = None
    turn_number: int = 0
    previous_decision_ids: list[str] = None

    def __post_init__(self):
        if self.previous_decision_ids is None:
            self.previous_decision_ids = []

@dataclass
class ProjectReport:
    total_decisions: int
    total_cost_usd: float
    average_latency_ms: float
    model_distribution: dict[str, int]
    bucket_distribution: dict[str, int]
    adapted_buckets: list[dict]  # which buckets were bumped and when
    recent_decisions: list[dict]  # last 10 decisions with meta

# Override hook type
OverrideHook = Callable[[str, Optional[ConversationContext]], Optional[str]]
```

## Router Class

```python
class Router:
    def __init__(
        self,
        config_path: Optional[str] = None,
        override_hook: Optional[OverrideHook] = None,
        storage_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the router.

        Args:
            config_path: Path to smartroute.yaml. If None, searches cwd then ~/.smartroute/
            override_hook: Optional function for programmatic routing overrides
            storage_path: Path to SQLite db. If None, uses .smartroute/db.sqlite in cwd

        Raises:
            ConfigError: If config file is invalid or missing required fields
            StorageError: If SQLite cannot be initialized
        """
        ...

    async def complete(
        self,
        prompt: str,
        context: Optional[ConversationContext] = None,
    ) -> RoutingResult:
        """
        Complete a prompt with automatic routing.

        Flow:
        1. Classify prompt -> task_type + complexity + confidence
        2. Apply overrides -> rules -> reputation -> defaults
        3. Call selected provider
        4. Store decision + metadata
        5. Return result with full explainability

        Args:
            prompt: The user prompt string
            context: Optional conversation context for signal detection

        Returns:
            RoutingResult with text and metadata

        Raises:
            ProviderError: If all providers in tier + fallback tiers fail
            ClassificationError: If classifier fails and fail-open also fails
        """
        ...

    async def stream(
        self,
        prompt: str,
        context: Optional[ConversationContext] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a response with automatic routing.

        The routing decision is made before streaming begins (same as complete).
        The first chunk may have a small delay while classification + routing happens.
        The final chunk contains the full RoutingMeta.

        Args:
            prompt: The user prompt string
            context: Optional conversation context

        Yields:
            StreamChunk objects. The last chunk always has is_finished=True.

        Raises:
            ProviderError: If all providers fail
        """
        ...

    async def report_signal(
        self,
        decision_id: str,
        signal_type: str,  # "hard_regen", "soft_regen", "explicit_correction", "acceptance"
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Report an explicit signal for a past decision.

        This is the manual API for signal reporting. Most signals should be
        auto-detected via the SignalCollector, but this allows apps with
        explicit feedback UI (thumbs up/down) to feed into the system.

        Args:
            decision_id: The UUID returned in meta.decision_id
            signal_type: One of the known signal types
            metadata: Optional extra context

        Returns:
            None (fire-and-forget, failures are logged not raised)
        """
        ...

    async def report(self) -> ProjectReport:
        """
        Generate a project health report.

        Returns:
            ProjectReport with stats, distributions, and recent adaptations
        """
        ...

    async def reset_reputation(self, bucket_key: Optional[str] = None) -> None:
        """
        Reset reputation scores. Useful for testing or when user wants to
        clear learned preferences.

        Args:
            bucket_key: If provided, reset only this bucket. If None, reset all.
        """
        ...
```

## Usage Examples

### Basic Usage
```python
import asyncio
from smartroute import Router

async def main():
    router = Router()
    result = await router.complete("Explain Python decorators")
    print(result.text)
    print(f"Routed to: {result.meta.model} because: {result.meta.reason}")

asyncio.run(main())
```

### Streaming with FastAPI
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from smartroute import Router

app = FastAPI()
router = Router()

@app.post("/chat")
async def chat(prompt: str):
    async def event_generator():
        async for chunk in router.stream(prompt):
            yield f"data: {chunk.text}\n\n"
            if chunk.is_finished:
                yield f"event: meta\ndata: {chunk.meta.reason}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

### Custom Override Hook
```python
def legal_override(prompt: str, context) -> Optional[str]:
    if "contract" in prompt.lower() or "/legal" in prompt.lower():
        return "anthropic"  # Force Anthropic for legal content
    return None

router = Router(override_hook=legal_override)
```

### With Conversation Context
```python
from smartroute import ConversationContext

context = ConversationContext(
    conversation_id="conv_123",
    turn_number=3,
)
result = await router.complete("Refactor that function", context=context)
```

### Reading the Report
```python
report = await router.report()
print(f"Total cost so far: ${report.total_cost_usd:.4f}")
for bucket in report.adapted_buckets:
    print(f"Bucket {bucket['key']} bumped to {bucket['new_tier']} at {bucket['timestamp']}")
```

## Error Types

```python
class SmartRouteError(Exception):
    """Base exception."""
    pass

class ConfigError(SmartRouteError):
    """Configuration is missing or invalid."""
    pass

class ProviderError(SmartRouteError):
    """All providers failed."""
    def __init__(self, message: str, attempts: list[dict]):
        self.attempts = attempts  # [{provider, model, error}, ...]
        super().__init__(message)

class ClassificationError(SmartRouteError):
    """Classifier failed and fail-open could not recover."""
    pass

class StorageError(SmartRouteError):
    """SQLite operation failed."""
    pass
```

## Configuration File Search Path

1. `config_path` argument to `Router()`
2. `SMARTROUTE_CONFIG` environment variable
3. `./smartroute.yaml` (current working directory)
4. `~/.smartroute/config.yaml`
5. Built-in `general` preset (copied to `~/.smartroute/` on first run)

## Environment Variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI provider |
| `ANTHROPIC_API_KEY` | Anthropic provider |
| `GROQ_API_KEY` | Groq provider |
| `SMARTROUTE_CONFIG` | Override config file path |
| `SMARTROUTE_STORAGE` | Override SQLite db path |
| `SMARTROUTE_LOG_LEVEL` | DEBUG, INFO, WARNING, ERROR |
