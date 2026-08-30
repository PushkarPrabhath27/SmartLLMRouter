# SmartRoute — Usage Examples

## Example 1: Basic Usage

```python
import asyncio
from smartroute import Router

async def main():
    router = Router()
    result = await router.complete("Explain Python decorators in simple terms")

    print(result.text)
    print(f"Model: {result.meta.model}")
    print(f"Why: {result.meta.reason}")
    print(f"Complexity: {result.meta.complexity:.2f}")
    print(f"Confidence: {result.meta.confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Example 2: Streaming with FastAPI

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
            if chunk.is_finished:
                yield f"event: meta\ndata: {chunk.meta.reason}\n\n"
                break
            yield f"data: {chunk.text}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

## Example 3: Custom Override Hook

```python
from typing import Optional
from smartroute import Router, ConversationContext

def legal_override(prompt: str, context: Optional[ConversationContext]) -> Optional[str]:
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ["contract", "legal", "lawyer", "liability"]):
        return "anthropic"
    if context and context.turn_number > 5:
        return "anthropic"  # Long conversations get best model
    return None

async def main():
    router = Router(override_hook=legal_override)
    result = await router.complete("Draft a software liability clause")
    print(result.meta.model)  # anthropic/claude-3-sonnet
    print(result.meta.override_applied)  # programmatic_hook

if __name__ == "__main__":
    asyncio.run(main())
```

## Example 4: Domain Preset

```python
from smartroute import Router

async def main():
    # Use web_dev preset for a coding project
    router = Router(config_path="web_dev_preset.yaml")

    # This will likely route to anthropic due to "refactor" override in preset
    result = await router.complete("Refactor this React component to use hooks")
    print(result.meta.reason)

if __name__ == "__main__":
    asyncio.run(main())
```

`web_dev_preset.yaml`:

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
```

## Example 5: Reading the Report

```python
from smartroute import Router

async def main():
    router = Router()

    # Run some prompts...
    for _ in range(5):
        await router.complete("Generate a random fact")

    # Get report
    report = await router.report()
    print(f"Total decisions: {report.total_decisions}")
    print(f"Total cost: ${report.total_cost_usd:.4f}")
    print(f"Avg latency: {report.average_latency_ms}ms")

    print("\nModel distribution:")
    for model, count in report.model_distribution.items():
        print(f"  {model}: {count}")

    print("\nAdapted buckets:")
    for adaptation in report.adapted_buckets:
        print(f"  {adaptation['key']}: {adaptation['old_tier']} -> {adaptation['new_tier']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Example 6: Manual Signal Reporting

```python
from smartroute import Router

async def main():
    router = Router()

    # Your app has a thumbs-down button
    result = await router.complete("Some prompt")
    decision_id = result.meta.decision_id

    # User clicks thumbs down
    await router.report_signal(
        decision_id=decision_id,
        signal_type="explicit_correction"
    )

    # This immediately affects the reputation EMA for this bucket

if __name__ == "__main__":
    asyncio.run(main())
```

## Example 7: Resetting Learned Data

```python
from smartroute import Router

async def main():
    router = Router()

    # Reset all reputation (start fresh)
    await router.reset_reputation()

    # Or reset just one bucket
    await router.reset_reputation(bucket_key="code_low")

if __name__ == "__main__":
    asyncio.run(main())
```

## Example 8: Inspecting Routing Decisions

```python
from smartroute import Router

async def main():
    router = Router()

    result = await router.complete("Write a Python function to sort a list")

    meta = result.meta
    print(f"""
    Routing Decision:
    -----------------
    Model: {meta.model}
    Task Type: {meta.task_type}
    Complexity: {meta.complexity:.2f} ({meta.complexity_bucket})
    Confidence: {meta.confidence:.2f}
    Reason: {meta.reason}
    Reputation Score: {meta.reputation_score:.2f}
    Was Adapted: {meta.was_adapted}
    Estimated Cost: ${meta.estimated_cost_usd:.6f}
    Actual Latency: {meta.latency_ms}ms
    """)

if __name__ == "__main__":
    asyncio.run(main())
```
