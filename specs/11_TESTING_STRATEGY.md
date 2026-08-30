# SmartRoute — Testing Strategy

## Philosophy

Test behavior, not implementation. Every test should answer: "If I change the internal code, does the external promise still hold?"

## Test Pyramid

```
    /\
   /  \     E2E (3 tests)
  /----\
 /      \   Integration (15 tests)
/--------\
/          \ Unit (80+ tests)
/------------\
```

## Unit Tests (80% of suite)

### Naming Convention
`test_{module}_{function}_{scenario}`

Examples:
- `test_features_token_count_empty_string_returns_zero`
- `test_engine_override_hook_takes_priority_over_yaml`
- `test_reputation_ema_converges_after_negative_signals`

### Required Unit Test Coverage

| Module | Min Tests | Key Scenarios |
|--------|-----------|---------------|
| `classifier/features.py` | 20 | Each feature extractor: empty, normal, edge case, unicode, very long |
| `classifier/classifier.py` | 10 | Complexity monotonicity, confidence bounds, fail-open |
| `routing/engine.py` | 15 | Each decision level, fallback chain, tie-breaking |
| `routing/reputation.py` | 10 | EMA math, threshold logic, cooldown, boundary values |
| `routing/explainability.py` | 8 | All reason string templates |
| `signals/detectors.py` | 15 | Each detector: 10 positive, 10 negative, boundary timing |
| `storage/*` | 20 | CRUD, foreign keys, concurrent access, `:memory:` vs file |
| `config/loader.py` | 10 | Env interpolation, preset merge, validation errors |

### Mocking Rules

- **Mock providers:** Never hit real APIs in tests. Use `MockProvider` that returns deterministic responses.
- **Mock storage:** For routing tests, mock `Storage` to return preset reputation values.
- **Mock time:** Use `freezegun` or manual `datetime` injection for signal timing tests.
- **Do NOT mock:** The classifier (test the real heuristics), the config loader (test with real YAML files in `tmp_path`).

## Integration Tests (15% of suite)

### Test Fixtures

Create `conftest.py` with:

```python
@pytest.fixture
async def router(tmp_path):
    config_path = tmp_path / "smartroute.yaml"
    config_path.write_text("""
providers:
  openai:
    api_key: "test-key"
    model: "gpt-4o-mini"
  anthropic:
    api_key: "test-key"
    model: "claude-3-sonnet"
  groq:
    api_key: "test-key"
    model: "llama-3.1-8b"
routing:
  low_complexity: "groq"
  medium_complexity: "openai"
  high_complexity: "anthropic"
""")
    return Router(config_path=str(config_path), storage_path=str(tmp_path / "test.db"))

@pytest.fixture
def mock_provider():
    class MockProvider(BaseProvider):
        async def complete(self, prompt, model):
            return "mock response"
        async def stream(self, prompt, model):
            yield StreamChunk(text="mock", is_finished=True, meta=None)
    return MockProvider()
```

### Required Integration Tests

1. End-to-end happy path: Prompt -> classify -> route -> respond -> store decision
2. Reputation bump: Send 10 prompts, inject 3 hard_regen signals, verify next prompt routes to higher tier
3. Fallback chain: Primary provider fails, secondary succeeds, latency recorded
4. Streaming: 5 chunks arrive in order, final chunk has meta
5. Config reload: Change YAML, call `reload_config()`, new rules apply
6. Override hook: Hook returns "anthropic", verify anthropic is called regardless of complexity
7. YAML regex rule: Prompt matches regex, verify correct provider
8. Signal detection: Two identical prompts within 20s, verify hard_regen stored
9. Multilingual correction: Spanish "no funciona" triggers explicit_correction
10. Report generation: After 5 decisions, report shows correct stats

## Property-Based Tests (5% of suite)

Use `hypothesis` for:

```python
from hypothesis import given, strategies as st

@given(st.text(), st.text())
def test_classifier_complexity_is_normalized(prompt, extra):
    """Adding arbitrary text should not break complexity bounds."""
    result = classify(prompt + extra)
    assert 0.0 <= result.complexity <= 1.0
    assert 0.0 <= result.confidence <= 1.0

@given(st.integers(min_value=0, max_value=1000))
def test_ema_never_exceeds_bounds(signal_count):
    """EMA stays within [-1, 1] regardless of signal sequence."""
    ema = 0.5
    for _ in range(signal_count):
        ema = update_ema(ema, random.choice([-0.3, -0.2, -0.1, 0.05]))
        assert -1.0 <= ema <= 1.0
```

## Performance Tests (Not in CI, run manually)

```python
@pytest.mark.benchmark
def test_classifier_1000_tokens_under_10ms():
    prompt = "x " * 1000
    start = time.perf_counter()
    classify(prompt)
    assert time.perf_counter() - start < 0.010
```

## Coverage Requirements

- Line coverage: >= 90%
- Branch coverage: >= 85%
- Exclusions: `if __name__ == "__main__":` blocks, `examples/`, `config/presets/`

## CI Configuration

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv pip install -e ".[dev]"
      - run: ruff check .
      - run: ruff format --check .
      - run: mypy --strict smartroute/
      - run: pytest --cov=smartroute --cov-report=xml --cov-fail-under=90
      - uses: codecov/codecov-action@v4
```

## Test Data

Create `tests/data/`:
- `labeled_prompts.json` — 50 hand-labeled prompts for classifier accuracy testing
- `negative_keywords.json` — Multilingual negative keywords for detector testing
- `sample_conversations.json` — Mock conversation histories for signal testing
