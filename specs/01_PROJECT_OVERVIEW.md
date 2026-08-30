# SmartRoute — Project Overview

## Elevator Pitch

SmartRoute is a **local-first, embeddable Python library** that automatically classifies LLM prompts by task type and complexity, routes them to the optimal model tier, and **learns from implicit user feedback** to improve routing decisions per project over time.

It is not a gateway. It is not a proxy. It is `import smartroute` in your backend, zero infrastructure, zero dashboard, zero telemetry.

## The Core Adaptive Loop

```
prompt → classify → route → respond → capture signal → update reputation → future routes improve
```

This loop is the entire product. Everything else exists to feed it or explain it.

## Positioning vs. OmniRoute

| Dimension | OmniRoute | SmartRoute |
|-----------|-----------|------------|
| Form factor | Server/gateway (npm install, run daemon) | Library (`pip install`, `import`) |
| Routing logic | Provider health / cost / quota scoring | Task-type + complexity + per-project adaptive reputation |
| Target user | CLI tool / IDE integrator | Backend developer building AI features |
| Learning | None (static strategies) | Per-project implicit feedback loop |
| Surface area | 339 providers, dashboard, MCP, A2A | 3 providers, zero UI, code-only |
| Philosophy | Cover everything | Do one thing perfectly |

## Key Differentiators

1. **Automatic classification by task semantics** — not just "which provider is cheapest right now" but "this prompt is a 0.7 complexity code-refactoring task."
2. **Per-project adaptive reputation** — the library learns that *your* project's "code" prompts under 50 tokens fail on the cheap model and quietly bumps them up.
3. **Explainability as a trust layer** — every routing decision tells you *why*, so developers trust auto-mode enough to not override everything on day one.
4. **Local-first privacy** — all learning happens in a local SQLite file. No cloud. No telemetry.

## Non-Goals (Explicitly Out of Scope for V1)

- No semantic caching
- No MCP / A2A protocol support
- No proxy / stealth / request interception
- No hosted dashboard or web UI
- No provider catalog beyond OpenAI, Anthropic, Groq
- No embedding-based classifier (heuristics only for v1)
- No multi-turn conversation state escalation
- No latency-as-first-class routing dimension
- No shadow mode / A/B testing
- No pre-flight cost estimation UI

## Success Criteria

A developer can:
1. `pip install smartroute` in 30 seconds
2. Run 20 prompts through it without touching config
3. Watch the library auto-bump at least one complexity bucket after detecting 3-4 implicit negative signals
4. Read `result.meta.why` and understand every routing decision
5. Override a single rule in YAML and see it respected immediately

## Target Persona

**Primary:** Solo backend developers building AI features into their app (FastAPI, Django, Next.js backend). They want smart routing without running a sidecar.

**Secondary:** Small teams with multiple AI features (support bot + contract analyzer) who want different quality/cost profiles per feature without managing multiple API keys manually.
