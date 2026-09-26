# SmartRoute

[![CI](https://github.com/PushkarPrabhath27/SmartLLMRouter/actions/workflows/ci.yml/badge.svg)](https://github.com/PushkarPrabhath27/SmartLLMRouter/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/smartroute-ai.svg)](https://pypi.org/project/smartroute-ai/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://pypi.org/project/smartroute-ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

An in-process, embeddable Python library that classifies LLM prompt complexity in <10ms and adaptively routes queries across providers using implicit feedback loops and local SQLite persistence.

```bash
pip install smartroute-ai
```

---

## Quickstart

```python
import asyncio
from smartroute import Router


async def main() -> None:
    async with Router() as router:
        result = await router.complete("Explain Python decorators in simple terms")
        print(f"Model:    {result.meta.model}")
        print(f"Decision: {result.meta.reason}")
        print(f"Latency:  {result.meta.latency_ms}ms")
        print(f"Cost:     ${result.meta.estimated_cost_usd:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
```

Output:

```text
Model:    groq/llama-3.1-8b
Decision: Default routing: general task, low complexity (0.20) -> groq
Latency:  312ms
Cost:     $0.000016
```

---

## How It Works

SmartRoute routes queries across models (e.g., Groq Llama 3, OpenAI GPT-4o-mini, Anthropic Claude 3.5 Sonnet) without running auxiliary classification models or proxy processes.

```text
Prompt
  │
  ▼
[Heuristic Classifier] ──────────── <10ms, single-pass feature extraction
  │
  ▼
[4-Tier Decision Hierarchy] ─────── Evaluates overrides, reputation, and defaults
  │
  ▼
[Provider Dispatcher] ───────────── Dispatches with fallback chain (OpenAI / Anthropic / Groq)
  │
  ▼
[Signal Collector] ──────────────── Detects hard/soft regens, corrections, acceptance
  │ (background async)
  ▼
[SQLite WAL Storage] ────────────── Updates EMA reputation scores; hashes prompt to SHA-256
```

### 1. Zero-LLM Heuristic Classifier (<10ms)

Instead of incurring a 50–200ms latency penalty and dollar cost calling a classification model (e.g., GPT-4o-mini as a router), SmartRoute extracts 9 structural and lexical features in a single pass over the input text:

- **Token Length**: Binned into sub-ranges (`<50`, `50–200`, `200–500`, `>500`).
- **Code Ratio**: Density of syntax tokens (`{`, `}`, `def `, `class `, `import `, indentation).
- **Multi-Step Markers**: Sequential reasoning cues (`first`, `then`, `step 1`, `finally`).
- **Wh-Questions**: Query framing markers (`what`, `why`, `how`, `explain`).
- **Domain Dictionaries**: Keyword matching for technical domains (databases, systems, math, legal).
- **Imperative Density**: Instruction-to-context ratio.

These features compute a deterministic complexity score in $[0.0, 1.0]$, mapped to a discrete tier (`low`, `medium`, `high`) and semantic task category (`code`, `reasoning`, `general`). In local benchmarks, classification completes in `<8ms` for typical prompts (~1000 tokens).

### 2. Deterministic 4-Tier Decision Hierarchy

Routing decisions follow an explicit evaluation pipeline:

| Precedence | Level | Evaluation Mechanism | Fallthrough Condition |
|---|---|---|---|
| 1 (Highest) | Programmatic Override | User-supplied `Callable[[str, Context], str \| None]` | Returns `None` |
| 2 | Configuration Rules | Pattern matching (`exact`, `contains`, `regex`, `path`) | No rule matches |
| 3 | Adaptive Reputation | Checks whether target tier EMA $< 0.30$ outside cooldown | Score healthy ($\ge 0.30$) or in cooldown |
| 4 (Lowest) | Default Tier Mapping | Maps classified `(task_type, complexity_bucket)` to default provider | None (terminates evaluation) |

### 3. Implicit Feedback Loop (EMA Reputation)

SmartRoute updates model reputations automatically by observing downstream user interactions rather than requiring explicit survey dialogs.

When a signal $s$ is captured, the bucket's Exponential Moving Average (EMA) updates:

$$\text{EMA}_t = \alpha \cdot s + (1 - \alpha) \cdot \text{EMA}_{t-1} \quad (\alpha = 0.3)$$

The updated EMA is clamped to $[0.0, 1.0]$.

| Signal Type | Numerical Value ($s$) | Detection Heuristic |
|---|---|---|
| `hard_regen` | `-0.30` | The exact same prompt submitted within 30 seconds of previous output. |
| `explicit_correction` | `-0.20` | Next conversational message (<200 tokens) matches negative feedback expressions across 6 supported languages (EN, ES, FR, DE, ZH, JA). |
| `soft_regen` | `-0.10` | Jaccard token overlap between sequential prompts exceeds 80% within 60 seconds. |
| `acceptance` | `+0.05` | Normal conversation continuation or clean closure without negative signals. |

#### Auto-Bump Condition
A tier auto-escalates (`low` $\rightarrow$ `medium` $\rightarrow$ `high`) when all three criteria are met:
1. `call_count >= min_calls_before_bump` (default: 10 calls recorded).
2. `ema_score < bump_threshold` (default: 0.30).
3. Cooldown expired: `(now - last_bumped_at) >= cooldown_minutes` (default: 5 minutes).

---

## Architecture & Data Invariants

### In-Process Architecture
SmartRoute runs entirely within the hosting Python process (`import smartroute`). It is not an HTTP sidecar proxy (e.g., LiteLLM proxy, Portkey, Envoy). This architecture guarantees:
- Zero additional network serialization hops or socket overhead.
- No background daemon processes or external containers to maintain.
- Direct propagation of provider exceptions and native asynchronous generator streaming.

### Privacy & Storage Contract
All persistent state resides in a local SQLite file (default: `.smartroute/db.sqlite`):
- **WAL Mode**: Initialized with `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000` to allow non-blocking concurrent reads during background writes.
- **Zero Full-Text Prompt Storage**: Prompts are hashed via SHA-256 (`prompt_hash`). Only the hash and an administrative 100-character prefix (`prompt_preview`) are persisted. Raw prompt bodies are never stored to disk.
- **Zero Telemetry**: No outbound network requests are dispatched other than direct completions to configured model providers.

---

## Streaming & Multi-Turn Usage

### FastAPI SSE Streaming

```python
from collections.abc import AsyncIterator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from smartroute import Router

app = FastAPI()
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

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### Multi-Turn Context Tracking

Pass `ConversationContext` to correlate signals across turns in a conversation session:

```python
import asyncio
from smartroute import ConversationContext, Router


async def main() -> None:
    router = Router()
    context = ConversationContext(conversation_id="session_01")

    # Turn 1: Initial query
    res1 = await router.complete("Write an SQL query to calculate user churn", context=context)
    print(f"Turn 1 ({res1.meta.model}):\n{res1.text}\n")

    # Turn 2: User provides negative feedback -> triggers explicit_correction (-0.2)
    res2 = await router.complete("That's wrong, calculate it over 30 days instead", context=context)
    print(f"Turn 2 ({res2.meta.model}):\n{res2.text}\n")

    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Configuration Reference

Configuration files use standard YAML syntax (`smartroute.yaml`). Values support `${VAR}` environment variable interpolation.

```yaml
preset: "general"

providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o-mini"
    timeout: 30.0
    max_retries: 2
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-sonnet"
    timeout: 30.0
    max_retries: 2
  groq:
    api_key: "${GROQ_API_KEY}"
    model: "llama-3.1-8b"
    timeout: 30.0
    max_retries: 2

routing:
  low_complexity: "groq"
  medium_complexity: "openai"
  high_complexity: "anthropic"
  fallback:
    low: ["openai"]
    medium: ["anthropic"]
    high: ["openai"]
  overrides:
    - match: "incident"
      match_type: "contains"
      model: "anthropic"

adaptation:
  enabled: true
  bump_threshold: 0.30
  cooldown_minutes: 5
  ema_alpha: 0.30
  min_calls_before_bump: 10
```

### Parameter Reference

| Parameter | Type | Default | Behavior |
|---|---|---|---|
| `preset` | string | `"general"` | Base template (`general`, `web_dev`, `data_science`). |
| `providers.<name>.api_key` | string | Required | Provider secret key; raises `ConfigError` if unset. |
| `providers.<name>.model` | string | Provider default | Upstream model identifier passed to completion API. |
| `providers.<name>.timeout` | float | `30.0` | Socket read/write timeout in seconds. |
| `providers.<name>.max_retries` | int | `2` | Number of retry attempts on transient network errors. |
| `routing.low_complexity` | string | `"groq"` | Provider key assigned to low complexity tier. |
| `routing.medium_complexity` | string | `"openai"` | Provider key assigned to medium complexity tier. |
| `routing.high_complexity` | string | `"anthropic"` | Provider key assigned to high complexity tier. |
| `routing.fallback.<tier>` | list[str] | `[]` | Ordered failover provider list on request errors. |
| `adaptation.enabled` | bool | `true` | Enables implicit signal detection and EMA updates. |
| `adaptation.bump_threshold` | float | `0.30` | EMA boundary triggering tier auto-escalation. |
| `adaptation.cooldown_minutes`| int | `5` | Required quiet period between consecutive auto-bumps. |
| `adaptation.ema_alpha` | float | `0.30` | Smoothing factor applied to incoming feedback signals. |
| `adaptation.min_calls_before_bump` | int | `10` | Minimum sample size required prior to evaluating bumps. |

### Built-in Presets

Presets bootstrap configuration without boilerplate:
- `general`: Balanced cost-to-capability allocation across generic queries.
- `web_dev`: Biases refactoring and debugging patterns toward high-tier models.
- `data_science`: Enforces code-oriented reasoning routing on analytical and SQL keywords.

---

## Verification & Benchmarks

The test suite enforces full test coverage, deterministic error handling, and strict typing across all modules:

```bash
# Run test suite with coverage
uv run pytest --cov=smartroute --cov-report=term-missing -q

# Static type analysis and linting
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict smartroute/
```

- **Test Suite**: 459 passing tests.
- **Coverage**: 97% branch and statement coverage.
- **Type Safety**: Fully typed with `mypy --strict` compliance.
- **Classification Latency**: <8ms single-pass execution.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
