# SmartRoute — Routing Engine Specification

## Overview

The RoutingEngine is the decision-making core. It takes a `ClassificationResult`, consults the configuration, checks reputation scores, and produces a routing decision with full explainability.

## Decision Hierarchy (Strict Priority Order)

The engine evaluates rules in this exact order. The first match wins. No merging, no voting.

### Level 1: Programmatic Override Hook
- **Input:** `prompt: str`, `context: Optional[ConversationContext]`
- **Source:** `override_hook` passed to `Router.__init__()`
- **Behavior:** If hook returns a non-None string (provider key like "anthropic" or "openai" or "groq"), use that provider immediately.
- **Explainability:** `override_applied="programmatic_hook"`
- **Performance:** Must complete in <5ms. If it raises, log error and fall through.

### Level 2: YAML Rule Match
- **Input:** `prompt: str`
- **Source:** `config.routing.overrides` list
- **Rule Types:**
  1. **Exact match:** `prompt == rule.match`
  2. **Contains match:** `rule.match in prompt` (case-insensitive)
  3. **Regex match:** `re.search(rule.match, prompt, re.IGNORECASE)`
  4. **Path match:** `rule.match` is a file path pattern, matched against any file paths detected in prompt
- **Behavior:** First matching rule determines provider. If no rule matches, fall through.
- **Explainability:** `override_applied="yaml_rule:<rule_description>"`

### Level 3: Adaptive Reputation Score
- **Input:** `(task_type, complexity_bucket)`
- **Source:** SQLite `reputation` table
- **Lookup Key:** `"{task_type.value}_{complexity_bucket.value}"` (e.g., "code_low")
- **Behavior:**
  1. Look up EMA score for this bucket at current default tier
  2. If EMA < `config.adaptation.bump_threshold` (default 0.3) AND call_count >= 10:
     - Bump to next tier (low -> medium -> high)
     - Record adaptation event
     - Set `was_adapted=True`
  3. If EMA >= threshold, use default tier
- **Explainability:** `reason="Reputation score {ema:.2f} below threshold, bumped to {tier}"`
- **Cooldown:** After a bump, same bucket cannot bump again for `cooldown_minutes` (default 5).

### Level 4: Default Tier Mapping
- **Input:** `complexity_bucket`
- **Source:** `config.routing.{low|medium|high}_complexity`
- **Behavior:** Direct mapping from bucket to provider key.
- **Explainability:** `reason="Default tier for {bucket} complexity {task_type}"`

## Reputation System (EMA)

### EMA Formula

```python
def update_ema(old_ema: float, signal_value: float, alpha: float = 0.3) -> float:
    return alpha * signal_value + (1 - alpha) * old_ema
```

- `alpha = 0.3` (configurable, but fixed in v1)
- Initial EMA for all buckets = 0.5 (neutral)
- Signal values:
  - Hard regeneration: -0.3
  - Explicit correction: -0.2
  - Soft regeneration: -0.1
  - Acceptance: +0.05

### Auto-Bump Logic

```python
def should_bump(ema: float, call_count: int, threshold: float, last_bump_time: Optional[datetime]) -> bool:
    if call_count < 10:
        return False  # Not enough data
    if ema >= threshold:
        return False  # Score is acceptable
    if last_bump_time and (now - last_bump_time) < timedelta(minutes=cooldown_minutes):
        return False  # In cooldown
    return True
```

### Bucket Key Schema

```
{task_type}_{complexity_bucket}

Examples:
- "code_low"
- "code_medium"
- "code_high"
- "creative_low"
- "reasoning_medium"
- "general_high"
```

Total possible buckets: 6 task types * 3 buckets = 18. In practice, most projects will only use 3-8 actively.

## Provider Fallback Chain

When a provider call fails:
1. **Same-tier fallback:** Try other providers in the same complexity tier (if configured)
2. **Next-tier fallback:** If all same-tier providers fail, escalate to next complexity tier
3. **Final fallback:** If all tiers fail, raise `ProviderError` with full attempt log

Example chain for medium tier:
1. Primary: `openai/gpt-4o-mini`
2. Same-tier fallback: `groq/llama-3.1-8b` (if configured as medium fallback)
3. Next-tier fallback: `anthropic/claude-3-sonnet` (high tier)

## Explainability String Generation

The `reason` field in `RoutingMeta` must be a human-readable sentence:

| Scenario | Reason String |
|---|---|
| Override hook | "Programmatic override: forced to anthropic" |
| YAML exact match | "YAML rule matched: exact string 'debug this' -> anthropic" |
| YAML contains | "YAML rule matched: prompt contains 'contract' -> anthropic" |
| YAML regex | "YAML rule matched: regex /legal/ -> anthropic" |
| Reputation bump | "Adaptive: reputation 0.25 below threshold, bumped from low to medium" |
| Default | "Default routing: code task, medium complexity (0.45) -> openai" |
| Fallback | "Primary provider failed, fallback to groq after openai error: Timeout" |

## Cost Estimation (Pre-Flight)

Before routing, estimate cost using provider pricing table (hardcoded in v1):

```python
PRICING = {
    "groq/llama-3.1-8b": {"input": 0.05, "output": 0.08},      # per 1M tokens
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-3-sonnet": {"input": 3.00, "output": 15.00},
}

def estimate_cost(prompt_tokens: int, model: str, avg_output_ratio: float = 0.5) -> float:
    pricing = PRICING[model]
    estimated_output = int(prompt_tokens * avg_output_ratio)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (estimated_output / 1_000_000) * pricing["output"]
    return input_cost + output_cost
```

This is stored in `meta.estimated_cost_usd` for observability. It is NOT used for routing decisions in v1.

## Latency Tracking

After each successful provider call:
- Record actual latency in `decisions.latency_ms`
- Update rolling average per model in memory (not persisted to SQLite in v1)
- Include actual latency in `meta.latency_ms`

Latency is tracked but NOT used for routing in v1. It is for explainability and future v1.5 latency-aware routing.

## Thread Safety

- Config is read-only after init (immutable dataclass)
- Reputation reads are from SQLite (aiosqlite handles concurrency)
- Reputation writes are async and fire-and-forget
- The engine itself is stateless; all state is in storage

## Error Handling

| Error | Behavior |
|---|---|
| Config missing routing section | Use hardcoded defaults (low=groq, med=openai, high=anthropic) |
| Reputation table empty | Use defaults, start EMA at 0.5 |
| Reputation read fails | Log warning, use defaults |
| Provider not found in config | Skip that provider in fallback chain |
| All providers fail | Raise `ProviderError` with attempt log |

## Testing Requirements

- Unit test each level of decision hierarchy independently
- Mock storage to test reputation bump logic
- Test fallback chain with mocked failing providers
- Test explainability strings for all scenarios
- Edge case: EMA exactly at threshold (should NOT bump)
- Edge case: Cooldown active (should NOT bump even if EMA is low)
