# SmartRoute

**Local-first, embeddable Python library that classifies LLM prompts and adaptively routes them across providers — learning from implicit user feedback over time.**

SmartRoute is not a gateway and not a proxy. It is `import smartroute` in your backend: zero infrastructure, zero dashboard, zero telemetry.

```
prompt → classify → route → respond → capture signal → update reputation → future routes improve
```

This adaptive loop is the entire product. Everything else exists to feed it or explain it.

## Features (V1)

- **Heuristic classifier** — 9 features (token count, code ratio, multi-step markers, domain hints, …) → task type + complexity, no embeddings, no network calls.
- **Adaptive reputation** — per-project EMA reputation per `(task_type, complexity_bucket, model_tier)`; failing tiers get auto-bumped after repeated negative implicit signals.
- **Explainability** — every routing decision ships a human-readable `reason`, `was_adapted`, `override_applied`, and more in `RoutingMeta`.
- **Local-first privacy** — all learning happens in a local SQLite file (`.smartroute/db.sqlite`). No cloud. No telemetry.
- **3 providers, one interface** — OpenAI, Anthropic, Groq, with automatic tier fallback.

## Installation

```bash
pip install smartroute
```

## Quickstart

```python
import asyncio

from smartroute import Router


async def main() -> None:
    router = Router()  # uses the default "general" preset
    result = await router.complete("Refactor this function to use pathlib: ...")
    print(result.text)
    print(result.meta.why)  # why this model was chosen


asyncio.run(main())
```

> The quickstart above becomes runnable when the `Router` public API lands (Phase 6). Configuration, examples, and the full API reference will be filled in here as phases complete — see the [roadmap](https://github.com/PushkarPrabhath27/SmartLLMRouter/blob/main/CHANGELOG.md).

## Configuration

SmartRoute is configured via `smartroute.yaml` (or the `SMARTROUTE_CONFIG` environment variable), with optional `general`, `web_dev`, and `data_science` presets. The full config reference will be documented here in Phase 6.

## Development

```bash
git clone https://github.com/PushkarPrabhath27/SmartLLMRouter.git
cd SmartLLMRouter
python -m venv .venv && source .venv/Scripts/activate  # Windows Git Bash
pip install -e .[dev]

pytest              # run tests
ruff check .        # lint
ruff format --check .
mypy smartroute/    # strict type check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and code standards.

## License

[MIT](LICENSE)
