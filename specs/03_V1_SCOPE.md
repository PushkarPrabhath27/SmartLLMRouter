# SmartRoute — V1 Scope (Frozen)

> **Rule:** Nothing in the P0 list changes without a written justification. P1/P2 items are explicitly shelved. If a feature is not on this list, it does not exist in v1.

## P0 — Must Ship (The V1)

### 1. Async Public API
- `Router.complete(prompt, context=None) -> RoutingResult`
- `Router.stream(prompt, context=None) -> AsyncIterator[StreamChunk]`
- Both support optional `ConversationContext` for multi-turn awareness (storage only, no escalation logic)
- `Router.report()` → returns `ProjectReport` dataclass with current stats

### 2. Heuristic Classifier (9 Features)
- Token count (`tiktoken`)
- Code block ratio (``` fenced blocks / total tokens)
- Question vs. instruction detection
- Multi-step marker count ("then", "after that", "step 1", etc.)
- Ambiguity score (hedge words / total words)
- Domain hint via keyword matching (code, creative, reasoning, summarization, translation)
- File path reference detection
- Urgency keyword detection
- Instruction verb density
- Output: `{task_type: str, complexity: float[0-1], confidence: float[0-1]}`

### 3. Routing Decision Engine
- Decision hierarchy (strict order):
  1. Programmatic override hook (if returns non-None)
  2. YAML rule match (exact string, regex, or path contains)
  3. Adaptive reputation score (if EMA < 0.3, bump tier)
  4. Default tier for (task_type, complexity_bucket)
- Complexity buckets: `low` (0.0-0.33), `medium` (0.33-0.66), `high` (0.66-1.0)
- Default tier mapping:
  - low → groq/llama-3.1-8b
  - medium → openai/gpt-4o-mini
  - high → anthropic/claude-3-sonnet

### 4. Reputation-Based Adaptation
- EMA (exponential moving average) per `(task_type, complexity_bucket, model_tier)`
- Alpha = 0.3
- Signals:
  - Hard regeneration (exact same prompt within 30s): -0.3
  - Explicit correction (next message matches negative keyword list): -0.2
  - Soft regeneration (>80% token overlap within 60s): -0.1
  - Acceptance (conversation continues, no negative signal): +0.05
- Auto-bump when EMA < 0.3 over last 10+ calls
- Cooldown: 5 minutes between bumps for same bucket
- Visible log: `"low_complexity/code bucket bumped to med_complexity tier"`

### 5. Implicit Signal Detection
- Hard regeneration: exact string match + timestamp window
- Explicit correction: regex keyword list (English + 5 common languages: Spanish, French, German, Chinese, Japanese)
- Soft regeneration: token set overlap ratio (cheap, no embeddings)
- All signals stored in SQLite with decision_id foreign key

### 6. Local SQLite Storage
- Database path: `.smartroute/db.sqlite` (relative to cwd, configurable)
- Tables:
  - `decisions`: id, timestamp, prompt_hash, task_type, complexity, model_used, cost, latency_ms
  - `signals`: id, decision_id, signal_type, value, timestamp
  - `reputation`: id, bucket_key, model_tier, ema_score, call_count, last_updated
  - `config`: key, value (for runtime overrides)
- `aiosqlite` for async access
- Auto-migration on version change (simple: drop and recreate if schema mismatch)

### 7. YAML Configuration + Programmatic Override
- Config file: `smartroute.yaml` (or env var `SMARTROUTE_CONFIG`)
- Schema:
  ```yaml
  preset: "general"  # or "web_dev", "data_science"
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
    overrides:
      - match: "path contains /legal"
        model: "anthropic"
      - match: "prompt contains 'contract'"
        model: "anthropic"
  adaptation:
    enabled: true
    bump_threshold: 0.3
    cooldown_minutes: 5
  ```
- Programmatic override hook signature:
  ```python
  def my_override(prompt: str, context: ConversationContext) -> Optional[str]:
      # Return model key like "anthropic" or None to fall through
      pass
  ```

### 8. Explainability Metadata
Every `RoutingResult` and `StreamChunk` includes:
```python
@dataclass
class RoutingMeta:
    model: str                    # e.g., "openai/gpt-4o-mini"
    task_type: str                # e.g., "code"
    complexity: float             # 0.0-1.0
    complexity_bucket: str        # "low", "medium", "high"
    confidence: float             # classifier confidence
    reason: str                   # human-readable explanation
    reputation_score: float       # current EMA for this bucket
    was_adapted: bool             # true if reputation caused bump
    override_applied: Optional[str]  # which override rule fired
    estimated_cost_usd: float     # pre-flight cost estimate
    latency_ms: int                # actual latency
```

### 9. Domain Presets (Cold Start Fix)
- `general.yaml`: Balanced defaults for mixed tasks
- `web_dev.yaml`: Code-heavy defaults, lower threshold for code→medium tier
- `data_science.yaml`: Python/data keywords, routes structured output to stronger models
- Presets ship in `smartroute/config/presets/` and are copied on first run

### 10. Provider Integration (3 Providers)
- OpenAI (gpt-4o-mini)
- Anthropic (claude-3-sonnet)
- Groq (llama-3.1-8b)
- Unified interface: `BaseProvider.complete()` and `BaseProvider.stream()`
- Automatic fallback: if primary provider in tier fails, try next provider in same tier, then escalate to next tier

## P1 — Shelved for V1.5 (Post-First-Users)

- Shadow mode (call expensive model in background for comparison)
- Pre-flight cost estimation with token counting
- Latency-aware routing (track rolling average latency per model)
- Structured output detection (JSON/XML routing)
- Conversation-state escalation (turn count, accumulated frustration)
- HTML dashboard generator (`.smartroute/dashboard.html`)
- Cross-project reputation transfer (opt-in)
- More providers (Google, Cohere, local models)

## P2 — Shelved for V2 (Post-Product-Market-Fit)

- Embedding-based classifier
- Semantic caching
- MCP / A2A protocol support
- Multi-provider catalog (>10)
- Hosted/cloud version
- Team-wide routing profiles
- Drift detection (model behavior change over time)

## Definition of Done for V1

- [ ] All P0 features implemented and unit-tested
- [ ] Integration test: 20 synthetic prompts, classifier accuracy >70% on hand-labeled test set
- [ ] Integration test: reputation system bumps a bucket after 3 negative signals
- [ ] Example: FastAPI endpoint using `router.stream()`
- [ ] Example: CLI script using `router.complete()` with custom override
- [ ] README with install, quickstart, and config reference
- [ ] `pyproject.toml` with proper metadata, no dependency bloat
- [ ] CI passes (pytest, ruff, mypy --strict)
- [ ] No P1/P2 code in the repository (not even stubs that suggest future features)
