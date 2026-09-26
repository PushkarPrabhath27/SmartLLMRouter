# SmartRoute

<p align="center">
  <a href="https://pypi.org/project/smartroute-ai/"><img src="https://img.shields.io/pypi/v/smartroute-ai.svg?color=blue" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/smartroute-ai/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python Versions" /></a>
  <a href="https://github.com/PushkarPrabhath27/SmartLLMRouter/actions/workflows/ci.yml"><img src="https://github.com/PushkarPrabhath27/SmartLLMRouter/actions/workflows/ci.yml/badge.svg" alt="CI Status" /></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Code style: ruff" /></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/types-Mypy%20Strict-blue.svg" alt="Type checked: mypy" /></a>
  <a href="https://github.com/PushkarPrabhath27/SmartLLMRouter"><img src="https://img.shields.io/badge/coverage-97%25-brightgreen.svg" alt="Test Coverage" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" /></a>
</p>

An in-process, embeddable Python library that classifies LLM prompt complexity in under 10ms and adaptively routes traffic across model providers based on implicit user feedback signals.

---

## 1. Overview

LLM routing typically introduces operational complexity: managing standalone proxy containers (e.g. LiteLLM, Portkey) or routing prompt payloads through third-party hosted routers.

SmartRoute solves this at the application layer. It executes entirely in-process (`import smartroute`), requiring no external sidecars, no cloud dashboards, and no background daemon processes. It inspects prompts deterministically, directs traffic across providers (**Groq**, **OpenAI**, **Anthropic**), and continuously tracks model performance per task type using an Exponential Moving Average (EMA) stored in an embedded SQLite database.

```
prompt ──> classify (<10ms) ──> evaluate hierarchy ──> dispatch with fallback ──> observe signals ──> update EMA
```

---

## 2. Architectural Comparison

| Dimension | Gateway Sidecars | Hosted Cloud Routers | SmartRoute (`smartroute-ai`) |
|---|---|---|---|
| **Runtime Model** | Separate daemon / Docker container | Third-party cloud service | **In-process Python library** |
| **Classification Latency** | 50–150ms (or secondary LLM call) | 200–500ms network round-trip | **< 10ms deterministic heuristic** |
| **Telemetry & Privacy** | Centralized proxy logs | Plaintext prompt leaves boundary | **Zero telemetry; SHA-256 hashed locally** |
| **State Storage** | Redis / PostgreSQL required | Proprietary cloud database | **SQLite WAL mode embedded in repository** |
| **Adaptation Mechanism** | Static routing weights | Proprietary heuristics | **Explicit EMA updated on implicit signals** |
| **Type Guarantees** | JSON schema / REST | Remote API contracts | **Strict type hints (`mypy --strict`)** |

---

## 3. Installation

Install via `pip` or `uv`:

```bash
pip install smartroute-ai
```

The package installs as `smartroute-ai` and exposes the top-level namespace `smartroute`.

---

## 4. Minimal Runnable Example

```python
import asyncio
from smartroute import Router


async def main() -> None:
    async with Router() as router:
        result = await router.complete("Explain Python decorators with a concise code example")

        print(result.text)
        print(f"Model:      {result.meta.model}")
        print(f"Latency:    {result.meta.latency_ms:.1f}ms")
        print(f"Estimated:  ${result.meta.estimated_cost_usd:.6f}")
        print(f"Rationale:  {result.meta.reason}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Streaming Execution

SmartRoute supports native asynchronous streaming across all providers. The routing decision is resolved pre-flight; intermediate chunks yield text deltas, and the terminating chunk delivers complete `RoutingMeta` metadata:

```python
from collections.abc import AsyncIterator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from smartroute import Router

app = FastAPI(title="SmartRoute Gateway")
router = Router()


@app.post("/stream")
async def stream_completion(prompt: str) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        async for chunk in router.stream(prompt):
            if chunk.is_finished:
                if chunk.meta:
                    yield f"event: meta\ndata: {chunk.meta.reason}\n\n"
                break
            yield f"data: {chunk.text}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 6. Decision Hierarchy Specification

When resolving a routing decision, SmartRoute executes a deterministic 4-level evaluation hierarchy:

```
[Level 1: Programmatic Override Hook]
        │
        ├── None ────────────────────────────────────────────────────────┐
        ▼                                                                │
[Level 2: Config Override Rules (Exact, Regex, Contains, Path)]          │ Match Found
        │                                                                ▼
        ├── No Match ───────────────────────────────────────────> [Execute Route]
        ▼                                                                ▲
[Level 3: Adaptive Reputation Check]                                     │
        │  - Check EMA score vs bump_threshold                           │
        │  - If degraded & cooldown expired: Escalate Tier               │
        │                                                                │
        ▼                                                                │
[Level 4: Default Complexity Mapping] ───────────────────────────────────┘
```

1. **Level 1 — Programmatic Override Hook**: User-supplied function `Callable[[str, ConversationContext | None], str | None]` evaluated first. Returns provider key (e.g. `"anthropic"`) or `None`.
2. **Level 2 — Configuration Override Rules**: Ordered rule evaluations defined in YAML or presets supporting `exact`, `contains`, `regex`, and `path` matches.
3. **Level 3 — Adaptive Reputation Evaluation**: Checks whether the target complexity tier has dropped below `bump_threshold` with sufficient sample size (`min_calls_before_bump`) and outside `cooldown_minutes`. If triggered, auto-escalates tier (`low` $\rightarrow$ `medium` $\rightarrow$ `high`).
4. **Level 4 — Default Complexity Mapping**: Deterministic assignment mapping classified `(task_type, complexity_bucket)` to the configured baseline provider.

---

## 7. Configuration Specification

Configuration follows layered inheritance: **Built-in Defaults $\rightarrow$ Preset File $\rightarrow$ User YAML File**.

### Search Path Precedence
1. **Config File**: `config_path` parameter $\rightarrow$ `SMARTROUTE_CONFIG` environment variable $\rightarrow$ `./smartroute.yaml` $\rightarrow$ `~/.smartroute/config.yaml` $\rightarrow$ built-in defaults.
2. **Storage File**: `storage_path` parameter $\rightarrow$ `SMARTROUTE_STORAGE` environment variable $\rightarrow$ `./.smartroute/db.sqlite` $\rightarrow$ `~/.smartroute/db.sqlite`.

### Configuration Schema Reference

```yaml
preset: "general" # "general" | "web_dev" | "data_science"

providers:
  openai:
    api_key: "${OPENAI_API_KEY}" # Supports ${VAR} environment interpolation
    model: "gpt-4o-mini"
    base_url: null # Optional custom proxy/endpoint
    timeout: 30 # Request timeout in seconds
    max_retries: 2
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-sonnet"
    timeout: 30
    max_retries: 2
  groq:
    api_key: "${GROQ_API_KEY}"
    model: "llama-3.1-8b"
    timeout: 30
    max_retries: 2

routing:
  low_complexity: "groq"
  medium_complexity: "openai"
  high_complexity: "anthropic"
  fallback:
    low: ["openai", "anthropic"]
    medium: ["anthropic"]
    high: ["openai"]

adaptation:
  enabled: true
  bump_threshold: 0.3 # EMA score below this triggers auto-bump
  cooldown_minutes: 5 # Minimum interval between consecutive tier bumps
  ema_alpha: 0.3 # Exponential moving average weight factor
  min_calls_before_bump: 10 # Sample size threshold required before escalation

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `preset` | string | `"general"` | Base template (`general`, `web_dev`, `data_science`) |
| `providers.<name>.api_key` | string | Environment variable | API authentication key (interpolated at load time) |
| `providers.<name>.model` | string | Provider default | Upstream model identifier |
| `providers.<name>.timeout` | float | `30.0` | HTTP request timeout in seconds |
| `routing.low_complexity` | string | `"groq"` | Provider key for low complexity queries |
| `routing.medium_complexity`| string | `"openai"` | Provider key for medium complexity queries |
| `routing.high_complexity` | string | `"anthropic"` | Provider key for high complexity queries |
| `adaptation.enabled` | bool | `true` | Enables implicit signal tracking & reputation updates |
| `adaptation.bump_threshold`| float | `0.3` | Score threshold below which a tier auto-escalates |
| `adaptation.cooldown_minutes` | int | `5` | Required cooldown duration between bump actions |
| `adaptation.ema_alpha` | float | `0.3` | Weight of new signal: $\text{EMA}_t = \alpha S + (1 - \alpha)\text{EMA}_{t-1}$ |
| `adaptation.min_calls_before_bump` | int | `10` | Minimum recorded calls before an auto-bump can occur |

---

## 8. Telemetry & Privacy Contract

SmartRoute enforces strict local-first privacy invariants:

1. **Zero External Telemetry**: The library initiates zero network requests other than direct outbound LLM calls to your configured providers. There are no tracking pings, metrics collectors, or external analytics endpoints.
2. **SHA-256 Prompt Hashing**: Full prompt text is never written to disk. The embedded SQLite storage records:
   - `prompt_hash`: SHA-256 digest of the prompt string for deduplication and regeneration detection.
   - `prompt_preview`: The first 100 characters of the prompt string for debugging identification.
3. **SQLite WAL Mode**: Operational state (routing decisions, implicit feedback signals, and reputation EMA values) is persisted to a local SQLite database configured with Write-Ahead Logging (`PRAGMA journal_mode=WAL`) and `PRAGMA busy_timeout=5000` to ensure non-blocking concurrent reads.

---

## 9. Implicit Feedback Signals

SmartRoute detects user interaction signals asynchronously without blocking response completion:

| Signal Type | Value | Trigger Condition |
|---|---|---|
| `hard_regen` | `-0.30` | The exact same prompt is re-submitted within 30 seconds. |
| `soft_regen` | `-0.10` | Jaccard token overlap between consecutive prompts exceeds 80% within 60 seconds. |
| `explicit_correction` | `-0.20` | Short follow-up message (<200 words) matching negative sentiment phrases across 6 languages (EN, ES, FR, DE, ZH, JA). |
| `acceptance` | `+0.05` | Normal conversational continuation or session conclusion without negative markers. |

---

## 10. Analytics & Reporting

Inspect project-level routing performance and cost metrics programmatically:

```python
report = await router.report()

print(f"Total Routed Decisions: {report.total_decisions}")
print(f"Aggregated Cost:        ${report.total_cost_usd:.4f}")
print(f"Mean Latency:           {report.average_latency_ms:.1f}ms")
print(f"Model Distribution:     {report.model_distribution}")
print(f"Adapted Buckets:        {report.adapted_buckets}")
```

To reset learned weights back to baseline:

```python
# Reset single bucket
await router.reset_reputation(bucket_key="code_low")

# Reset all reputation states
await router.reset_reputation()
```

---

## 11. Development & Test Suite

SmartRoute maintains a full unit and integration test suite with mock transports:

```bash
# Clone repository
git clone https://github.com/PushkarPrabhath27/SmartLLMRouter.git
cd SmartLLMRouter

# Install with development dependencies
uv sync --all-extras

# Run test suite with code coverage
uv run pytest --cov=smartroute --cov-report=term-missing -q

# Code formatting and static type checking
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict smartroute/
```

---

## 12. License

This project is licensed under the [MIT License](LICENSE).
