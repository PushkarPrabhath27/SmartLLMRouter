"""Unit and integration tests for smartroute.routing.engine (Phase 3 Module 3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from smartroute.config.schema import (
    AdaptationConfig,
    Config,
    OverrideRule,
    ProviderConfig,
    RoutingConfig,
)
from smartroute.routing.engine import (
    RoutingEngine,
    estimate_cost,
)
from smartroute.storage.connection import Storage
from smartroute.types import (
    ClassificationResult,
    DomainHint,
    FeatureVector,
    TaskType,
)


def _features(tokens: int = 1000) -> FeatureVector:
    """Helper to construct dummy FeatureVector for testing."""
    return FeatureVector(
        token_count=tokens,
        code_block_ratio=0.0,
        is_question=False,
        is_instruction=False,
        instruction_verb_count=0,
        multi_step_count=0,
        ambiguity_score=0.0,
        domain_hint=DomainHint(domain="", match_count=0, match_ratio=0.0),
        file_path_count=0,
        urgency_count=0,
        instruction_verb_density=0.0,
    )


def _classification(
    task_type: TaskType = TaskType.CODE,
    complexity: float = 0.5,
    tokens: int = 1000,
) -> ClassificationResult:
    """Helper to construct dummy ClassificationResult."""
    return ClassificationResult(
        task_type=task_type,
        complexity=complexity,
        confidence=0.8,
        features=_features(tokens=tokens),
    )


@pytest.fixture
def mock_config() -> Config:
    """Standard config fixture with all 3 providers and fallback chains."""
    providers = {
        "groq": ProviderConfig(api_key="k-groq", model="llama-3.1-8b"),
        "openai": ProviderConfig(api_key="k-openai", model="gpt-4o-mini"),
        "anthropic": ProviderConfig(api_key="k-anthropic", model="claude-3-sonnet"),
    }
    routing = RoutingConfig(
        low_complexity="groq",
        medium_complexity="openai",
        high_complexity="anthropic",
        fallback={
            "low": ["groq", "openai"],
            "medium": ["openai", "groq", "anthropic"],
            "high": ["anthropic", "openai"],
        },
        overrides=[
            OverrideRule(
                match="debug this exact",
                match_type="exact",
                model="anthropic",
                description="exact_debug",
            ),
            OverrideRule(
                match="contract",
                match_type="contains",
                model="anthropic",
                description="legal_contains",
            ),
            OverrideRule(
                match=r"\burgent\s+bug\b",
                match_type="regex",
                model="openai",
                description="urgent_bug_regex",
            ),
            OverrideRule(
                match="*.py",
                match_type="path",
                model="anthropic",
                description="python_path",
            ),
        ],
    )
    adaptation = AdaptationConfig(
        enabled=True,
        bump_threshold=0.3,
        cooldown_minutes=5,
        ema_alpha=0.3,
        min_calls_before_bump=10,
    )
    return Config(providers=providers, routing=routing, adaptation=adaptation)


@pytest.fixture
async def in_memory_storage() -> Storage:
    storage = Storage(":memory:")
    await storage.connect()
    try:
        yield storage
    finally:
        await storage.close()


class TestCostEstimation:
    def test_estimate_cost_openai(self) -> None:
        """1000 tokens for openai/gpt-4o-mini:
        input: 1000 / 1e6 * 0.15 = 0.00015
        output: 500 / 1e6 * 0.60 = 0.00030
        total: 0.00045
        """
        cost = estimate_cost(1000, "openai/gpt-4o-mini")
        assert pytest.approx(cost, abs=1e-7) == 0.00045

    def test_estimate_cost_groq(self) -> None:
        """1000 tokens for groq/llama-3.1-8b:
        input: 1000 / 1e6 * 0.05 = 0.00005
        output: 500 / 1e6 * 0.08 = 0.00004
        total: 0.00009
        """
        cost = estimate_cost(1000, "groq/llama-3.1-8b")
        assert pytest.approx(cost, abs=1e-7) == 0.00009

    def test_estimate_cost_anthropic(self) -> None:
        """1000 tokens for anthropic/claude-3-sonnet:
        input: 1000 / 1e6 * 3.00 = 0.003
        output: 500 / 1e6 * 15.00 = 0.0075
        total: 0.0105
        """
        cost = estimate_cost(1000, "anthropic/claude-3-sonnet")
        assert pytest.approx(cost, abs=1e-7) == 0.0105

    def test_unknown_model_returns_zero(self) -> None:
        cost = estimate_cost(1000, "unknown-model-xyz")
        assert cost == 0.0


class TestLevel1ProgrammaticHook:
    @pytest.mark.asyncio
    async def test_hook_returns_valid_provider(self, mock_config: Config) -> None:
        engine = RoutingEngine(
            config=mock_config,
            override_hook=lambda prompt, ctx: "anthropic",
        )
        decision = await engine.route("test prompt", _classification())
        assert decision.provider_key == "anthropic"
        assert decision.model == "anthropic/claude-3-sonnet"
        assert decision.override_applied == "programmatic_hook"
        assert decision.reason == "Programmatic override: forced to anthropic"
        assert decision.fallback_chain[0] == "anthropic"

    @pytest.mark.asyncio
    async def test_hook_returns_none_falls_through(self, mock_config: Config) -> None:
        engine = RoutingEngine(
            config=mock_config,
            override_hook=lambda prompt, ctx: None,
        )
        decision = await engine.route("test prompt", _classification(complexity=0.1))
        # Falls through to Level 4 (low complexity -> groq)
        assert decision.provider_key == "groq"
        assert decision.override_applied is None

    @pytest.mark.asyncio
    async def test_hook_raises_exception_falls_through(self, mock_config: Config) -> None:
        def bad_hook(p: str, c: object) -> str:
            raise RuntimeError("hook error")

        engine = RoutingEngine(config=mock_config, override_hook=bad_hook)
        decision = await engine.route("test prompt", _classification(complexity=0.1))
        assert decision.provider_key == "groq"
        assert decision.override_applied is None

    @pytest.mark.asyncio
    async def test_hook_returns_unknown_provider_falls_through(self, mock_config: Config) -> None:
        engine = RoutingEngine(
            config=mock_config,
            override_hook=lambda prompt, ctx: "unconfigured-provider",
        )
        decision = await engine.route("test prompt", _classification(complexity=0.5))
        assert decision.provider_key == "openai"
        assert decision.override_applied is None


class TestLevel2YAMLOverrides:
    @pytest.mark.asyncio
    async def test_exact_match(self, mock_config: Config) -> None:
        engine = RoutingEngine(config=mock_config)
        decision = await engine.route("debug this exact", _classification())
        assert decision.provider_key == "anthropic"
        assert decision.override_applied == "yaml_rule:exact_debug"
        assert decision.reason == "YAML rule matched: exact string 'debug this exact' -> anthropic"

    @pytest.mark.asyncio
    async def test_contains_match_case_insensitive(self, mock_config: Config) -> None:
        engine = RoutingEngine(config=mock_config)
        decision = await engine.route("Please review this CONTRACT today", _classification())
        assert decision.provider_key == "anthropic"
        assert decision.override_applied == "yaml_rule:legal_contains"
        assert decision.reason == "YAML rule matched: prompt contains 'contract' -> anthropic"

    @pytest.mark.asyncio
    async def test_regex_match(self, mock_config: Config) -> None:
        engine = RoutingEngine(config=mock_config)
        decision = await engine.route("fix this urgent bug in prod", _classification())
        assert decision.provider_key == "openai"
        assert decision.override_applied == "yaml_rule:urgent_bug_regex"
        assert decision.reason == r"YAML rule matched: regex /\burgent\s+bug\b/ -> openai"

    @pytest.mark.asyncio
    async def test_path_match(self, mock_config: Config) -> None:
        engine = RoutingEngine(config=mock_config)
        decision = await engine.route("refactor main.py please", _classification())
        assert decision.provider_key == "anthropic"
        assert decision.override_applied == "yaml_rule:python_path"
        assert decision.reason == "YAML rule matched: path '*.py' -> anthropic"

    @pytest.mark.asyncio
    async def test_first_match_wins_ordering(self, mock_config: Config) -> None:
        # Prompt contains both "contract" (rule 2) and "*.py" (rule 4)
        engine = RoutingEngine(config=mock_config)
        decision = await engine.route("contract in helper.py", _classification())
        assert decision.override_applied == "yaml_rule:legal_contains"


class TestLevel3ReputationBump:
    @pytest.mark.asyncio
    async def test_disabled_adaptation_falls_through(
        self, mock_config: Config, in_memory_storage: Storage
    ) -> None:
        disabled_config = mock_config.model_copy(
            update={"adaptation": AdaptationConfig(enabled=False)}
        )
        engine = RoutingEngine(config=disabled_config, storage=in_memory_storage)
        decision = await engine.route("plain prompt", _classification(complexity=0.1))
        assert decision.provider_key == "groq"
        assert decision.was_adapted is False

    @pytest.mark.asyncio
    async def test_storage_none_falls_through(self, mock_config: Config) -> None:
        engine = RoutingEngine(config=mock_config, storage=None)
        decision = await engine.route("plain prompt", _classification(complexity=0.1))
        assert decision.provider_key == "groq"
        assert decision.was_adapted is False

    @pytest.mark.asyncio
    async def test_call_count_below_min_calls_does_not_bump(
        self, mock_config: Config, in_memory_storage: Storage
    ) -> None:
        # Seed reputation with low score (0.1) but only 5 calls (<10)
        await in_memory_storage.update_reputation(
            bucket_key="code_low", model_tier="low", ema=0.1, call_count=5
        )
        engine = RoutingEngine(config=mock_config, storage=in_memory_storage)
        decision = await engine.route("plain prompt", _classification(complexity=0.1))
        assert decision.provider_key == "groq"
        assert decision.was_adapted is False
        assert decision.reputation_score == 0.1

    @pytest.mark.asyncio
    async def test_successful_reputation_bump(
        self, mock_config: Config, in_memory_storage: Storage
    ) -> None:
        # Seed reputation with low score (0.25) and 15 calls (>=10)
        await in_memory_storage.update_reputation(
            bucket_key="code_low", model_tier="low", ema=0.25, call_count=15
        )
        engine = RoutingEngine(config=mock_config, storage=in_memory_storage)
        decision = await engine.route("plain prompt", _classification(complexity=0.1))
        # Bumped from low (groq) to medium (openai)
        assert decision.provider_key == "openai"
        assert decision.was_adapted is True
        assert (
            decision.reason
            == "Adaptive: reputation 0.25 below threshold, bumped from low to medium"
        )
        # Verify adaptation record was written to storage
        adaptations = await in_memory_storage.get_adaptations(bucket_key="code_low")
        assert len(adaptations) == 1
        assert adaptations[0].old_tier == "low"
        assert adaptations[0].new_tier == "medium"

    @pytest.mark.asyncio
    async def test_active_cooldown_prevents_bump(
        self, mock_config: Config, in_memory_storage: Storage
    ) -> None:
        now = datetime.now(timezone.utc)
        recent = now - timedelta(minutes=2)
        # Seed reputation with low score and recent bump
        await in_memory_storage.update_reputation(
            bucket_key="code_low",
            model_tier="low",
            ema=0.2,
            call_count=15,
            last_bumped_at=recent,
        )
        engine = RoutingEngine(config=mock_config, storage=in_memory_storage)
        decision = await engine.route("plain prompt", _classification(complexity=0.1))
        assert decision.provider_key == "groq"
        assert decision.was_adapted is False


class TestLevel4DefaultMapping:
    @pytest.mark.asyncio
    async def test_low_complexity_maps_to_groq(self, mock_config: Config) -> None:
        engine = RoutingEngine(config=mock_config)
        decision = await engine.route("simple query", _classification(complexity=0.15))
        assert decision.provider_key == "groq"
        assert decision.model == "groq/llama-3.1-8b"
        assert decision.fallback_chain == ["groq", "openai"]
        assert decision.reason == "Default routing: code task, low complexity (0.15) -> groq"
        assert decision.was_adapted is False

    @pytest.mark.asyncio
    async def test_medium_complexity_maps_to_openai(self, mock_config: Config) -> None:
        engine = RoutingEngine(config=mock_config)
        decision = await engine.route("medium task", _classification(complexity=0.5))
        assert decision.provider_key == "openai"
        assert decision.model == "openai/gpt-4o-mini"
        assert decision.fallback_chain == ["openai", "groq", "anthropic"]
        assert decision.reason == "Default routing: code task, medium complexity (0.50) -> openai"

    @pytest.mark.asyncio
    async def test_high_complexity_maps_to_anthropic(self, mock_config: Config) -> None:
        engine = RoutingEngine(config=mock_config)
        decision = await engine.route("complex architecture", _classification(complexity=0.85))
        assert decision.provider_key == "anthropic"
        assert decision.model == "anthropic/claude-3-sonnet"
        assert decision.fallback_chain == ["anthropic", "openai"]
        assert decision.reason == "Default routing: code task, high complexity (0.85) -> anthropic"

    @pytest.mark.asyncio
    async def test_storage_exception_during_read_falls_through_gracefully(
        self, mock_config: Config, monkeypatch: pytest.MonkeyPatch, in_memory_storage: Storage
    ) -> None:
        async def mock_get_rep(*args: object, **kwargs: object) -> None:
            raise RuntimeError("disk failure")

        monkeypatch.setattr(in_memory_storage, "get_reputation", mock_get_rep)
        engine = RoutingEngine(config=mock_config, storage=in_memory_storage)
        decision = await engine.route("simple query", _classification(complexity=0.15))
        assert decision.provider_key == "groq"
        assert decision.was_adapted is False

    @pytest.mark.asyncio
    async def test_storage_exception_during_record_adaptation_does_not_crash(
        self, mock_config: Config, monkeypatch: pytest.MonkeyPatch, in_memory_storage: Storage
    ) -> None:
        await in_memory_storage.update_reputation(
            bucket_key="code_low", model_tier="low", ema=0.25, call_count=15
        )

        async def mock_rec_adapt(*args: object, **kwargs: object) -> None:
            raise RuntimeError("write failure")

        monkeypatch.setattr(in_memory_storage, "record_adaptation", mock_rec_adapt)
        engine = RoutingEngine(config=mock_config, storage=in_memory_storage)
        decision = await engine.route("simple query", _classification(complexity=0.15))
        assert decision.provider_key == "openai"
        assert decision.was_adapted is True

    def test_estimate_cost_bare_model(self) -> None:
        cost = estimate_cost(1000, "gpt-4o-mini")
        assert pytest.approx(cost, abs=1e-7) == 0.00045
