"""Comprehensive unit and integration tests for the Router public API (Phase 6)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from smartroute.config.schema import (
    ProviderConfig,
)
from smartroute.exceptions import ProviderError
from smartroute.providers.base import BaseProvider
from smartroute.router import Router, _resolve_storage_path
from smartroute.types import (
    ConversationContext,
    RoutingResult,
    StreamChunk,
)


class MockProvider(BaseProvider):
    """Configurable mock provider for testing Router interactions."""

    def __init__(
        self,
        config: ProviderConfig,
        response_text: str = "mock response",
        fail_complete: bool = False,
        fail_stream: bool = False,
        chunks: list[str] | None = None,
    ) -> None:
        super().__init__(config)
        self.response_text = response_text
        self.fail_complete = fail_complete
        self.fail_stream = fail_stream
        self.chunks = chunks if chunks is not None else ["chunk1 ", "chunk2"]
        self.closed = False

    async def complete(self, prompt: str, model: str) -> str:
        if self.fail_complete:
            raise ProviderError(f"Simulated failure for {model}", attempts=[])
        return self.response_text

    async def stream(self, prompt: str, model: str) -> AsyncIterator[StreamChunk]:
        if self.fail_stream:
            raise ProviderError(f"Simulated stream failure for {model}", attempts=[])
        for chunk in self.chunks:
            yield StreamChunk(text=chunk, is_finished=False, meta=None)
        yield StreamChunk(text="", is_finished=True, meta=None)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def mock_providers_dict() -> dict[str, BaseProvider]:
    p_groq = MockProvider(
        ProviderConfig(api_key="k1", model="llama-3.1-8b"),
        response_text="Groq low complexity response",
    )
    p_openai = MockProvider(
        ProviderConfig(api_key="k2", model="gpt-4o-mini"),
        response_text="OpenAI medium response",
    )
    p_anthropic = MockProvider(
        ProviderConfig(api_key="k3", model="claude-3-sonnet"),
        response_text="Anthropic high response",
    )
    return {
        "groq": p_groq,
        "openai": p_openai,
        "anthropic": p_anthropic,
    }


@pytest.fixture
def test_yaml_config(tmp_path: Path) -> str:
    cfg = tmp_path / "smartroute.yaml"
    cfg.write_text(
        """
providers:
  groq:
    api_key: "dummy_groq"
    model: "llama-3.1-8b"
  openai:
    api_key: "dummy_openai"
    model: "gpt-4o-mini"
  anthropic:
    api_key: "dummy_anthropic"
    model: "claude-3-sonnet"
routing:
  low_complexity: "groq"
  medium_complexity: "openai"
  high_complexity: "anthropic"
  fallback:
    low: ["openai", "anthropic"]
    medium: ["anthropic"]
adaptation:
  enabled: true
  bump_threshold: 0.3
  cooldown_minutes: 5
  min_calls_before_bump: 2
""",
        encoding="utf-8",
    )
    return str(cfg)


@pytest.mark.asyncio
class TestRouterComplete:
    async def test_complete_happy_path(
        self, test_yaml_config: str, mock_providers_dict: dict[str, BaseProvider]
    ) -> None:
        router = Router(
            config_path=test_yaml_config,
            storage_path=":memory:",
            providers=mock_providers_dict,
        )
        async with router:
            result = await router.complete("What is the capital of France?")
            assert isinstance(result, RoutingResult)
            assert result.text == "Groq low complexity response"

            # Verify RoutingMeta fields
            meta = result.meta
            assert meta.model == "groq/llama-3.1-8b"
            assert meta.task_type in (
                "general",
                "reasoning",
                "creative",
                "summarization",
                "translation",
                "code",
            )
            assert meta.complexity_bucket == "low"
            assert meta.complexity >= 0.0
            assert meta.confidence > 0.0
            assert "Default routing" in meta.reason or "groq" in meta.reason
            assert meta.latency_ms >= 0
            assert meta.estimated_cost_usd >= 0.0
            assert meta.decision_id is not None
            assert len(meta.decision_id) > 10

            # Verify decision record persisted in SQLite
            decisions = await router.storage.get_recent_decisions(limit=5)
            assert len(decisions) == 1
            assert decisions[0].id == meta.decision_id
            assert decisions[0].model_used == meta.model
            assert decisions[0].prompt_preview == "What is the capital of France?"
            # Invariant: prompt hash is stored, full prompt is not accepted
            assert len(decisions[0].prompt_hash) == 64

    async def test_complete_with_conversation_context(
        self, test_yaml_config: str, mock_providers_dict: dict[str, BaseProvider]
    ) -> None:
        router = Router(
            config_path=test_yaml_config,
            storage_path=":memory:",
            providers=mock_providers_dict,
        )
        async with router:
            context = ConversationContext(conversation_id="conv_100", turn_number=0)
            res1 = await router.complete("Hello", context=context)
            assert context.turn_number == 1
            assert len(context.previous_decision_ids) == 1
            assert context.previous_decision_ids[0] == res1.meta.decision_id

            res2 = await router.complete("Follow-up question", context=context)
            assert context.turn_number == 2
            assert len(context.previous_decision_ids) == 2
            assert context.previous_decision_ids[1] == res2.meta.decision_id

    async def test_complete_fallback_reason(self, test_yaml_config: str) -> None:
        # groq fails, openai succeeds
        failing_groq = MockProvider(
            ProviderConfig(api_key="k1", model="llama-3.1-8b"),
            fail_complete=True,
        )
        working_openai = MockProvider(
            ProviderConfig(api_key="k2", model="gpt-4o-mini"),
            response_text="OpenAI fallback reply",
        )
        providers = {"groq": failing_groq, "openai": working_openai}

        router = Router(
            config_path=test_yaml_config,
            storage_path=":memory:",
            providers=providers,
        )
        async with router:
            result = await router.complete("Simple question")
            assert result.text == "OpenAI fallback reply"
            assert result.meta.model == "openai/gpt-4o-mini"
            assert "fallback to openai" in result.meta.reason


@pytest.mark.asyncio
class TestRouterStream:
    async def test_stream_happy_path(
        self, test_yaml_config: str, mock_providers_dict: dict[str, BaseProvider]
    ) -> None:
        router = Router(
            config_path=test_yaml_config,
            storage_path=":memory:",
            providers=mock_providers_dict,
        )
        async with router:
            chunks: list[StreamChunk] = []
            async for chunk in router.stream("Explain quicksort"):
                chunks.append(chunk)

            assert len(chunks) >= 2
            # All chunks except the last should have is_finished=False and meta=None
            for intermediate in chunks[:-1]:
                assert not intermediate.is_finished
                assert intermediate.meta is None

            # Last chunk must have is_finished=True and meta populated
            last_chunk = chunks[-1]
            assert last_chunk.is_finished
            assert last_chunk.text == ""
            assert last_chunk.meta is not None
            assert last_chunk.meta.decision_id is not None

            # Reassembled text
            assembled = "".join(c.text for c in chunks)
            assert "chunk1 chunk2" in assembled

            # Stored in SQLite
            decisions = await router.storage.get_recent_decisions(limit=1)
            assert len(decisions) == 1
            assert decisions[0].id == last_chunk.meta.decision_id

    async def test_stream_fallback(self, test_yaml_config: str) -> None:
        failing_groq = MockProvider(
            ProviderConfig(api_key="k1", model="llama-3.1-8b"),
            fail_stream=True,
        )
        working_openai = MockProvider(
            ProviderConfig(api_key="k2", model="gpt-4o-mini"),
            chunks=["streamed ", "from ", "openai"],
        )
        providers = {"groq": failing_groq, "openai": working_openai}

        router = Router(
            config_path=test_yaml_config,
            storage_path=":memory:",
            providers=providers,
        )
        async with router:
            chunks = [chunk async for chunk in router.stream("Simple question")]
            assert len(chunks) >= 2
            final_chunk = chunks[-1]
            assert final_chunk.is_finished
            assert final_chunk.meta is not None
            assert final_chunk.meta.model == "openai/gpt-4o-mini"
            assert "fallback" in final_chunk.meta.reason


@pytest.mark.asyncio
class TestRouterSignalsAndAdaptiveLearning:
    async def test_implicit_hard_regen_detection(
        self, test_yaml_config: str, mock_providers_dict: dict[str, BaseProvider]
    ) -> None:
        router = Router(
            config_path=test_yaml_config,
            storage_path=":memory:",
            providers=mock_providers_dict,
        )
        async with router:
            # First prompt
            r1 = await router.complete("Explain Python decorators")
            d1 = r1.meta.decision_id

            # Exact same prompt repeated within 30s -> triggers hard_regen
            r2 = await router.complete("Explain Python decorators")
            d2 = r2.meta.decision_id
            assert d1 != d2

            # Flush pending background tasks
            await router._collector.wait_pending()

            # Verify signals were persisted for d1
            signals = await router.storage.get_signals_for_decision(d1)
            assert len(signals) == 1
            assert signals[0].signal_type == "hard_regen"
            assert signals[0].signal_value == -0.3

    async def test_multi_turn_explicit_correction(
        self, test_yaml_config: str, mock_providers_dict: dict[str, BaseProvider]
    ) -> None:
        router = Router(
            config_path=test_yaml_config,
            storage_path=":memory:",
            providers=mock_providers_dict,
        )
        async with router:
            context = ConversationContext(conversation_id="conv_turn")
            r1 = await router.complete("Write quicksort in python", context=context)
            d1 = r1.meta.decision_id

            # User responds in next turn with negative feedback
            await router.complete("No, that's completely wrong and broken", context=context)
            await router._collector.wait_pending()

            signals = await router.storage.get_signals_for_decision(d1)
            assert len(signals) == 1
            assert signals[0].signal_type == "explicit_correction"


@pytest.mark.asyncio
class TestRouterManagementAndReporting:
    async def test_report_generation(
        self, test_yaml_config: str, mock_providers_dict: dict[str, BaseProvider]
    ) -> None:
        router = Router(
            config_path=test_yaml_config,
            storage_path=":memory:",
            providers=mock_providers_dict,
        )
        async with router:
            await router.complete("Query 1")
            await router.complete("Query 2")

            report = await router.report()
            assert report.total_decisions == 2
            assert report.average_latency_ms >= 0.0
            assert "groq/llama-3.1-8b" in report.model_distribution
            assert len(report.recent_decisions) == 2

    async def test_reset_reputation(
        self, test_yaml_config: str, mock_providers_dict: dict[str, BaseProvider]
    ) -> None:
        router = Router(
            config_path=test_yaml_config,
            storage_path=":memory:",
            providers=mock_providers_dict,
        )
        async with router:
            # Set reputation score
            await router.storage.update_reputation("code_low", "low", ema=0.2, call_count=5)
            rep = await router.storage.get_reputation("code_low", "low")
            assert rep is not None

            # Reset
            await router.reset_reputation("code_low")
            rep2 = await router.storage.get_reputation("code_low", "low")
            assert rep2 is None

    async def test_report_manual_signal(
        self, test_yaml_config: str, mock_providers_dict: dict[str, BaseProvider]
    ) -> None:
        router = Router(
            config_path=test_yaml_config,
            storage_path=":memory:",
            providers=mock_providers_dict,
        )
        async with router:
            res = await router.complete("Testing manual rating")
            dec_id = res.meta.decision_id

            await router.report_signal(dec_id, signal_type="thumbs_down")
            await router._collector.wait_pending()

            signals = await router.storage.get_signals_for_decision(dec_id)
            assert len(signals) == 1
            assert signals[0].signal_type == "explicit_correction"
            assert signals[0].detection_method == "manual"

    async def test_reload_config(
        self, test_yaml_config: str, mock_providers_dict: dict[str, BaseProvider]
    ) -> None:
        router = Router(
            config_path=test_yaml_config,
            storage_path=":memory:",
            providers=mock_providers_dict,
        )
        assert router.config.adaptation.bump_threshold == 0.3
        router.reload_config()
        assert router.config.adaptation.bump_threshold == 0.3


class TestRouterStorageResolution:
    def test_storage_path_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _resolve_storage_path(":memory:") == ":memory:"

        monkeypatch.setenv("SMARTROUTE_STORAGE", "/custom/db.sqlite")
        assert _resolve_storage_path(None) == "/custom/db.sqlite"

        monkeypatch.delenv("SMARTROUTE_STORAGE", raising=False)
        resolved = _resolve_storage_path(None)
        assert ".smartroute" in resolved
        assert "db.sqlite" in resolved

    def test_default_init_with_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env_openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env_anthropic")
        monkeypatch.setenv("GROQ_API_KEY", "env_groq")
        test_db = str(tmp_path / "env_db.sqlite")
        monkeypatch.setenv("SMARTROUTE_STORAGE", test_db)

        router = Router()
        assert router.config.providers["openai"].api_key == "env_openai"
        assert router.config.providers["anthropic"].api_key == "env_anthropic"
        assert router.config.providers["groq"].api_key == "env_groq"
        assert router.storage_path == test_db

    @pytest.mark.asyncio
    async def test_auto_connect_without_context_manager(
        self, test_yaml_config: str, mock_providers_dict: dict[str, BaseProvider]
    ) -> None:
        router = Router(
            config_path=test_yaml_config,
            storage_path=":memory:",
            providers=mock_providers_dict,
        )
        # Calling complete() directly auto-connects storage
        res = await router.complete("Hello without context manager")
        assert res.text == "Groq low complexity response"
        assert router.storage._connection is not None
        await router.close()
        assert router.storage._connection is None
