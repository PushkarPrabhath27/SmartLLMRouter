# SmartRoute

**Local-first, embeddable Python library that classifies LLM prompts and adaptively routes them across providers — learning from implicit user feedback over time.**

SmartRoute is not an external proxy, an API gateway, or a managed cloud service. It is simply `import smartroute` in your backend: zero infrastructure, zero dashboard, zero telemetry, and zero vendor lock-in.

```
prompt → classify (<10ms) → route → respond → capture signal → update reputation EMA → future routes adapt
```

All learning happens locally inside a SQLite file (`.smartroute/db.sqlite`). Prompts are privacy-preserved using SHA-256 hashes and short previews — full prompt text is never persisted.

---

## ✨ Features (V1)

- **⚡ Fast Heuristic Classifier (<10ms)**: Zero network calls, zero embeddings. Evaluates 9 deterministic features (token count, code ratio, multi-step markers, domain hints, imperative verbs, etc.) to assign a semantic task type and complexity score in $[0.0, 1.0]$.
- **🧠 Adaptive Reputation Engine**: Maintains per-project exponential moving average (EMA) reputation scores for each `(task_type, complexity_bucket, model_tier)`. When a tier consistently fails or triggers negative feedback, it automatically auto-bumps to higher capability models.
- **🔍 100% Deterministic Explainability**: Every response includes `RoutingMeta` detailing the exact `model` chosen, `task_type`, `complexity_bucket`, `estimated_cost_usd`, `latency_ms`, and human-readable `reason` explaining *why* that model was selected.
- **🛡️ Provider Fallback & Dispatch**: Unified async communication with **OpenAI**, **Anthropic**, and **Groq** supporting both single completions and token streaming with automatic fallback chains.
- **🤫 Implicit Feedback Detection**: Observes user interaction patterns without intrusive surveys:
  - `hard_regen` (`-0.3`): Same exact prompt repeated within 30 seconds.
  - `soft_regen` (`-0.1`): Similar prompt (>80% token overlap) sent within 60 seconds.
  - `explicit_correction` (`-0.2`): Short follow-up containing negative feedback across 6 languages (English, Spanish, French, German, Chinese, Japanese).
  - `acceptance` (`+0.05`): Natural conversation turn continuation or closure.
- **🔒 Local-First Privacy**: Stored in a local SQLite file with WAL mode. Zero cloud sync. Zero telemetry.

---

## 📦 Installation

```bash
pip install smartroute
```

---

## 🚀 Quickstart

### Zero-Config (Using Environment Variables)

Set your provider API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."
```

Run completion with automatic tier selection:

```python
import asyncio
from smartroute import Router


async def main() -> None:
    # Zero-config Router defaults to the built-in "general" preset and .smartroute/db.sqlite
    async with Router() as router:
        result = await router.complete("Explain Python decorators with a clean code example")

        print("Response:\n", result.text)
        print("\nExplainability:")
        print(f"  Model Used:         {result.meta.model}")
        print(f"  Task Type:          {result.meta.task_type}")
        print(
            f"  Complexity Bucket:  {result.meta.complexity_bucket} ({result.meta.complexity:.2f})"
        )
        print(f"  Reason:             {result.meta.reason}")
        print(f"  Latency:            {result.meta.latency_ms}ms")
        print(f"  Estimated Cost:     ${result.meta.estimated_cost_usd:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🌊 Streaming with FastAPI

SmartRoute streams tokens directly from providers while resolving routing pre-flight. The final chunk contains `is_finished=True` and the full `RoutingMeta`:

```python
from collections.abc import AsyncIterator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from smartroute import Router

app = FastAPI(title="SmartRoute Streaming API")
router = Router()


@app.post("/chat")
async def chat(prompt: str) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        async for chunk in router.stream(prompt):
            if chunk.is_finished:
                if chunk.meta:
                    yield f"event: meta\ndata: {chunk.meta.reason}\n\n"
                break
            yield f"data: {chunk.text}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
```

---

## 🎯 Programmatic Overrides & Conversation Context

You can provide custom business rules using an `override_hook` or track conversation turns:

```python
import asyncio
from smartroute import ConversationContext, Router


def legal_override(prompt: str, context: ConversationContext | None) -> str | None:
    """Force Anthropic for sensitive legal queries or extended turn sequences."""
    prompt_lower = prompt.lower()
    if any(k in prompt_lower for k in ["contract", "legal", "liability", "terms of service"]):
        return "anthropic"
    if context and context.turn_number > 5:
        return "anthropic"
    return None


async def main() -> None:
    router = Router(override_hook=legal_override)
    context = ConversationContext(conversation_id="conv_123")

    # Turn 1
    res1 = await router.complete(
        "Draft an indemnification clause for a SaaS contract", context=context
    )
    print(f"Turn 1 Routed to: {res1.meta.model} (Override: {res1.meta.override_applied})")

    # Turn 2 (implicit feedback observation automatically tracks continuity)
    res2 = await router.complete("Now add a mutual limitation of liability", context=context)
    print(f"Turn 2 Routed to: {res2.meta.model}")

    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📁 Domain Presets & Custom Configuration

SmartRoute includes three tuned domain presets:
1. `general` (Default): Balanced cost and capability.
2. `web_dev`: Optimized for full-stack engineering, framework refactoring, and code review.
3. `data_science`: Tuned for analytics, SQL generation, and machine learning scripts.

### Using `smartroute.yaml`:

Create a `smartroute.yaml` in your project root:

```yaml
preset: "web_dev"

providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o-mini"
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-sonnet"
  groq:
    api_key: "${GROQ_API_KEY}"
    model: "llama-3.1-8b"

routing:
  low_complexity: "groq"
  medium_complexity: "openai"
  high_complexity: "anthropic"
  fallback:
    low: ["openai", "anthropic"]
    medium: ["anthropic"]
  overrides:
    - match_type: "contains"
      match: "production incident"
      model: "anthropic"
      description: "Force high-tier on production incidents"

adaptation:
  enabled: true
  bump_threshold: 0.3
  cooldown_minutes: 5
  ema_alpha: 0.3
  min_calls_before_bump: 10
```

Load your configuration:

```python
router = Router(config_path="smartroute.yaml")
```

---

## 📊 Analytics & Health Reports

Inspect how your routing choices perform over time:

```python
report = await router.report()

print(f"Total Decisions:   {report.total_decisions}")
print(f"Total Cost:        ${report.total_cost_usd:.4f}")
print(f"Average Latency:   {report.average_latency_ms:.1f}ms")
print("Model Distribution:", report.model_distribution)
print("Adapted Buckets:   ", report.adapted_buckets)
```

To reset learned weights:

```python
# Reset single bucket
await router.reset_reputation(bucket_key="code_low")

# Or reset all learned reputations
await router.reset_reputation()
```

---

## ⚙️ Configuration Reference

### Search Path Precedence

1. **Configuration**: `config_path` argument $\rightarrow$ `SMARTROUTE_CONFIG` env var $\rightarrow$ `./smartroute.yaml` $\rightarrow$ `~/.smartroute/config.yaml` $\rightarrow$ built-in defaults.
2. **Storage**: `storage_path` argument $\rightarrow$ `SMARTROUTE_STORAGE` env var $\rightarrow$ `./.smartroute/db.sqlite` $\rightarrow$ `~/.smartroute/db.sqlite`.

### Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API authentication key |
| `ANTHROPIC_API_KEY` | Anthropic API authentication key |
| `GROQ_API_KEY` | Groq API authentication key |
| `SMARTROUTE_CONFIG` | Path to `smartroute.yaml` configuration file |
| `SMARTROUTE_STORAGE` | Path to SQLite storage file (e.g. `:memory:` for testing) |

---

## 🧪 Development & Testing

```bash
git clone https://github.com/PushkarPrabhath27/SmartLLMRouter.git
cd SmartLLMRouter

# Install with development dependencies
uv sync --all-extras

# Run full test suite with coverage
uv run pytest --cov=smartroute --cov-report=term-missing -q

# Static analysis and linting
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict smartroute/

# Build wheel package
uv build
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).
