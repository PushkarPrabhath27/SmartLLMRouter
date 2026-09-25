"""Unit tests for ProviderDispatcher (Phase 4 Part 2)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from smartroute.config.schema import (
    AdaptationConfig,
    Config,
    ProviderConfig,
    RoutingConfig,
)
from smartroute.exceptions import ProviderError
from smartroute.providers.base import BaseProvider
from smartroute.providers.dispatcher import DispatchResult, ProviderDispatcher
from smartroute.types import StreamChunk


class MockProvider(BaseProvider):
    """Configurable mock provider for testing dispatcher behavior."""

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
        self.chunks = chunks if chunks is not None else ["hello", " world"]
        self.closed = False

    async def complete(self, prompt: str, model: str) -> str:
        if self.fail_complete:
            raise ProviderError(f"Simulated failure for {model}")
        return self.response_text

    async def stream(self, prompt: str, model: str) -> AsyncIterator[StreamChunk]:
        if self.fail_stream:
            raise ProviderError(f"Simulated stream error for {model}")
        for chunk in self.chunks:
            yield StreamChunk(text=chunk, is_finished=False, meta=None)
        yield StreamChunk(text="", is_finished=True, meta=None)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def mock_config() -> Config:
    providers = {
        "openai": ProviderConfig(api_key="k1", model="gpt-4o-mini"),
        "anthropic": ProviderConfig(api_key="k2", model="claude-3-sonnet"),
        "groq": ProviderConfig(api_key="k3", model="llama-3.1-8b"),
    }
    routing = RoutingConfig(
        low_complexity="groq",
        medium_complexity="openai",
        high_complexity="anthropic",
        fallback={
            "medium": ["openai", "anthropic", "groq"],
        },
    )
    return Config(
        providers=providers,
        routing=routing,
        adaptation=AdaptationConfig(),
    )


class TestProviderDispatcherComplete:
    @pytest.mark.asyncio
    async def test_primary_succeeds_immediately(self, mock_config: Config) -> None:
        openai_mock = MockProvider(mock_config.providers["openai"], response_text="OpenAI answer")
        anthropic_mock = MockProvider(
            mock_config.providers["anthropic"], response_text="Anthropic answer"
        )
        dispatcher = ProviderDispatcher(
            config=mock_config,
            providers={"openai": openai_mock, "anthropic": anthropic_mock},
        )

        result: DispatchResult = await dispatcher.complete(["openai", "anthropic"], "Hi")
        assert result.text == "OpenAI answer"
        assert result.provider_used == "openai"
        assert result.model_used == "openai/gpt-4o-mini"
        assert result.attempts == []
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_primary_fails_secondary_succeeds(self, mock_config: Config) -> None:
        openai_mock = MockProvider(mock_config.providers["openai"], fail_complete=True)
        anthropic_mock = MockProvider(
            mock_config.providers["anthropic"],
            response_text="Anthropic fallback answer",
        )
        dispatcher = ProviderDispatcher(
            config=mock_config,
            providers={"openai": openai_mock, "anthropic": anthropic_mock},
        )

        result = await dispatcher.complete(["openai", "anthropic"], "Hi")
        assert result.text == "Anthropic fallback answer"
        assert result.provider_used == "anthropic"
        assert result.model_used == "anthropic/claude-3-sonnet"
        assert len(result.attempts) == 1
        assert result.attempts[0]["provider"] == "openai"
        assert "Simulated failure" in result.attempts[0]["error"]

    @pytest.mark.asyncio
    async def test_all_providers_fail(self, mock_config: Config) -> None:
        openai_mock = MockProvider(mock_config.providers["openai"], fail_complete=True)
        anthropic_mock = MockProvider(mock_config.providers["anthropic"], fail_complete=True)
        dispatcher = ProviderDispatcher(
            config=mock_config,
            providers={"openai": openai_mock, "anthropic": anthropic_mock},
        )

        with pytest.raises(ProviderError) as exc_info:
            await dispatcher.complete(["openai", "anthropic"], "Hi")

        err = exc_info.value
        assert "All providers in fallback chain" in str(err)
        assert len(err.attempts) == 2
        assert err.attempts[0]["provider"] == "openai"
        assert err.attempts[1]["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_empty_fallback_chain_raises(self, mock_config: Config) -> None:
        dispatcher = ProviderDispatcher(config=mock_config)
        with pytest.raises(ProviderError) as exc_info:
            await dispatcher.complete([], "Hi")
        assert "No providers specified" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unconfigured_provider_skipped(self, mock_config: Config) -> None:
        openai_mock = MockProvider(mock_config.providers["openai"], response_text="OpenAI answer")
        dispatcher = ProviderDispatcher(
            config=mock_config,
            providers={"openai": openai_mock},
        )
        # "custom" is in fallback chain but not in providers map
        result = await dispatcher.complete(["custom", "openai"], "Hi")
        assert result.provider_used == "openai"
        assert len(result.attempts) == 1
        assert result.attempts[0]["provider"] == "custom"


class TestProviderDispatcherStream:
    @pytest.mark.asyncio
    async def test_stream_primary_succeeds(self, mock_config: Config) -> None:
        openai_mock = MockProvider(mock_config.providers["openai"], chunks=["A", "B", "C"])
        dispatcher = ProviderDispatcher(
            config=mock_config,
            providers={"openai": openai_mock},
        )

        chunks = [c.text async for c in dispatcher.stream(["openai"], "Hi")]
        assert chunks == ["A", "B", "C", ""]

    @pytest.mark.asyncio
    async def test_stream_primary_fails_secondary_succeeds(self, mock_config: Config) -> None:
        openai_mock = MockProvider(mock_config.providers["openai"], fail_stream=True)
        anthropic_mock = MockProvider(mock_config.providers["anthropic"], chunks=["X", "Y"])
        dispatcher = ProviderDispatcher(
            config=mock_config,
            providers={"openai": openai_mock, "anthropic": anthropic_mock},
        )

        chunks = [c.text async for c in dispatcher.stream(["openai", "anthropic"], "Hi")]
        assert chunks == ["X", "Y", ""]

    @pytest.mark.asyncio
    async def test_stream_all_fail(self, mock_config: Config) -> None:
        openai_mock = MockProvider(mock_config.providers["openai"], fail_stream=True)
        anthropic_mock = MockProvider(mock_config.providers["anthropic"], fail_stream=True)
        dispatcher = ProviderDispatcher(
            config=mock_config,
            providers={"openai": openai_mock, "anthropic": anthropic_mock},
        )

        with pytest.raises(ProviderError) as exc_info:
            async for _ in dispatcher.stream(["openai", "anthropic"], "Hi"):
                pass
        assert len(exc_info.value.attempts) == 2

    @pytest.mark.asyncio
    async def test_stream_empty_fallback_chain_raises(self, mock_config: Config) -> None:
        dispatcher = ProviderDispatcher(config=mock_config)
        with pytest.raises(ProviderError) as exc_info:
            async for _ in dispatcher.stream([], "Hi"):
                pass
        assert "No providers specified" in str(exc_info.value)


class TestProviderDispatcherLifecycle:
    def test_default_providers_initialization(self, mock_config: Config) -> None:
        dispatcher = ProviderDispatcher(config=mock_config)
        assert "openai" in dispatcher._providers
        assert "anthropic" in dispatcher._providers
        assert "groq" in dispatcher._providers

    @pytest.mark.asyncio
    async def test_stream_unconfigured_provider_skipped(self, mock_config: Config) -> None:
        openai_mock = MockProvider(mock_config.providers["openai"], chunks=["ok"])
        dispatcher = ProviderDispatcher(
            config=mock_config,
            providers={"openai": openai_mock},
        )
        chunks = [c.text async for c in dispatcher.stream(["custom", "openai"], "Hi")]
        assert chunks == ["ok", ""]

    @pytest.mark.asyncio
    async def test_stream_immediate_empty_generator(self, mock_config: Config) -> None:
        openai_mock = MockProvider(mock_config.providers["openai"], chunks=[])
        dispatcher = ProviderDispatcher(
            config=mock_config,
            providers={"openai": openai_mock},
        )
        chunks = [c.text async for c in dispatcher.stream(["openai"], "Hi")]
        assert chunks == [""]

    @pytest.mark.asyncio
    async def test_close_and_context_manager(self, mock_config: Config) -> None:
        openai_mock = MockProvider(mock_config.providers["openai"])
        anthropic_mock = MockProvider(mock_config.providers["anthropic"])

        async with ProviderDispatcher(
            config=mock_config,
            providers={"openai": openai_mock, "anthropic": anthropic_mock},
        ):
            assert not openai_mock.closed
            assert not anthropic_mock.closed

        assert openai_mock.closed
        assert anthropic_mock.closed
