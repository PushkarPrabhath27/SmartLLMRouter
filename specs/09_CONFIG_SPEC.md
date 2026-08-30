# SmartRoute — Configuration Specification

## Overview

Configuration is layered: defaults -> preset -> user YAML -> environment variables -> programmatic override. Each layer can override the previous.

## Config File Format (YAML)

```yaml
# smartroute.yaml

# Optional: bootstrap from a preset
preset: "general"  # "general" | "web_dev" | "data_science"

# Provider credentials and model selection
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"        # Supports env var interpolation
    model: "gpt-4o-mini"
    base_url: null                      # Optional: for proxies/Azure
    timeout: 30                         # Seconds
    max_retries: 2

  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-sonnet"
    base_url: null
    timeout: 30
    max_retries: 2

  groq:
    api_key: "${GROQ_API_KEY}"
    model: "llama-3.1-8b"
    base_url: null
    timeout: 30
    max_retries: 2

# Routing defaults
routing:
  low_complexity: "groq"                # Provider key for low tier
  medium_complexity: "openai"           # Provider key for medium tier
  high_complexity: "anthropic"          # Provider key for high tier

  # Fallback chain within each tier (optional)
  fallback:
    low: ["groq"]
    medium: ["openai", "groq"]
    high: ["anthropic", "openai"]

  # Explicit override rules
  overrides:
    # Rule 1: Exact match
    - match: "debug this function"
      match_type: "exact"
      model: "anthropic"
      description: "Debugging always gets best model"

    # Rule 2: Contains substring
    - match: "contract"
      match_type: "contains"
      model: "anthropic"
      description: "Legal content"

    # Rule 3: Regex
    - match: "\\b(legal|law|contract|liability)\\b"
      match_type: "regex"
      model: "anthropic"
      description: "Legal keywords"

    # Rule 4: File path pattern
    - match: "/legal/"
      match_type: "path"
      model: "anthropic"
      description: "Legal directory"

# Adaptation settings
adaptation:
  enabled: true
  bump_threshold: 0.3                   # EMA below this triggers bump
  cooldown_minutes: 5                   # Minimum time between bumps
  ema_alpha: 0.3                        # EMA smoothing factor
  min_calls_before_bump: 10             # Need at least N calls to bump

# Logging
logging:
  level: "INFO"                         # DEBUG | INFO | WARNING | ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Advanced (v1.5+)
# latency_aware: false
# shadow_mode: false
# cost_estimation: true
```

## Preset Files

### `general.yaml` (Default)

```yaml
routing:
  low_complexity: "groq"
  medium_complexity: "openai"
  high_complexity: "anthropic"

  overrides: []

adaptation:
  enabled: true
  bump_threshold: 0.3
  cooldown_minutes: 5
  ema_alpha: 0.3
  min_calls_before_bump: 10
```

### `web_dev.yaml`

```yaml
routing:
  low_complexity: "groq"
  medium_complexity: "openai"
  high_complexity: "anthropic"

  overrides:
    - match: "refactor"
      match_type: "contains"
      model: "anthropic"
      description: "Refactoring needs strong model"

    - match: "debug"
      match_type: "contains"
      model: "anthropic"
      description: "Debugging needs strong model"

adaptation:
  enabled: true
  bump_threshold: 0.35                  # Slightly higher threshold (code is often fine on cheap)
  cooldown_minutes: 5
  ema_alpha: 0.3
  min_calls_before_bump: 10
```

### `data_science.yaml`

```yaml
routing:
  low_complexity: "groq"
  medium_complexity: "openai"
  high_complexity: "anthropic"

  overrides:
    - match: "\\b(pandas|numpy|scipy|sklearn|matplotlib|seaborn|plotly)\\b"
      match_type: "regex"
      model: "openai"
      description: "Data science libraries -> structured output needs GPT"

    - match: "dataframe"
      match_type: "contains"
      model: "openai"

adaptation:
  enabled: true
  bump_threshold: 0.3
  cooldown_minutes: 5
  ema_alpha: 0.3
  min_calls_before_bump: 10
```

## Environment Variable Interpolation

Values wrapped in `${VAR}` or `$VAR` are replaced from environment variables at load time.

```yaml
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
```

If the variable is not set:
- For `api_key`: raise `ConfigError` (required)
- For other fields: leave as literal string `${VAR}`

## Validation Rules

1. **Required fields:** `providers` must have at least one provider with `api_key` and `model`
2. **Provider keys:** Must be one of `openai`, `anthropic`, `groq` (v1)
3. **Routing tiers:** `low_complexity`, `medium_complexity`, `high_complexity` must map to valid provider keys
4. **Override rules:** `match_type` must be one of `exact`, `contains`, `regex`, `path`
5. **Regex rules:** Must compile successfully with `re.compile()`
6. **No circular fallbacks:** A tier cannot fallback to itself

## ConfigLoader Implementation

```python
class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = self._resolve_path(config_path)

    def load(self) -> Config:
        """Load and validate configuration."""
        raw = self._read_yaml()
        raw = self._apply_env_interpolation(raw)
        raw = self._merge_preset(raw)
        config = self._parse_and_validate(raw)
        return config

    def _resolve_path(self, config_path: Optional[str]) -> str:
        """Search path resolution."""
        ...

    def _read_yaml(self) -> dict:
        ...

    def _apply_env_interpolation(self, raw: dict) -> dict:
        """Replace ${VAR} and $VAR with os.environ values."""
        ...

    def _merge_preset(self, raw: dict) -> dict:
        """If preset specified, merge preset defaults under user overrides."""
        ...

    def _parse_and_validate(self, raw: dict) -> Config:
        """Pydantic validation."""
        ...
```

## Programmatic Override Hook

```python
from typing import Optional
from smartroute import ConversationContext

def my_override(prompt: str, context: Optional[ConversationContext]) -> Optional[str]:
    """
    Return a provider key ("openai", "anthropic", "groq") to force routing.
    Return None to let the normal decision hierarchy proceed.

    This runs BEFORE YAML rules, so it has highest priority.
    """
    if "contract" in prompt.lower():
        return "anthropic"
    if context and context.turn_number > 5:
        return "anthropic"  # Long conversations get better model
    return None

router = Router(override_hook=my_override)
```

## Hot Reload

Config is loaded once at Router initialization. To reload:

```python
router.reload_config()  # Async, re-reads YAML and rebuilds Config object
```

This is NOT automatic (no file watchers in v1).

## Testing

- Test all 4 match types (exact, contains, regex, path)
- Test env var interpolation with missing vars
- Test preset merging (user overrides should win)
- Test validation: missing api_key, invalid provider key, bad regex
- Test override hook: returning None, returning invalid key, raising exception
