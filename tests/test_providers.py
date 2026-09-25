"""Unit tests for LLM providers (Phase 4 Part 1)."""

from __future__ import annotations

import httpx
import pytest

from smartroute.config.schema import ProviderConfig
from smartroute.exceptions import ProviderError
from smartroute.providers.anthropic_provider import AnthropicProvider
from smartroute.providers.groq_provider import GroqProvider
from smartroute.providers.openai_provider import OpenAIProvider


@pytest.fixture
def openai_config() -> ProviderConfig:
    return ProviderConfig(
        api_key="sk-openai-test-key",
        model="gpt-4o-mini",
        timeout=15,
        max_retries=1,
    )


@pytest.fixture
def anthropic_config() -> ProviderConfig:
    return ProviderConfig(
        api_key="sk-ant-test-key",
        model="claude-3-sonnet",
        timeout=20,
        max_retries=1,
    )


@pytest.fixture
def groq_config() -> ProviderConfig:
    return ProviderConfig(
        api_key="gsk-test-key",
        model="llama-3.1-8b",
        timeout=10,
        max_retries=1,
    )


class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_complete_success(self, openai_config: ProviderConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == "https://api.openai.com/v1/chat/completions"
            assert request.headers["Authorization"] == "Bearer sk-openai-test-key"
            assert request.headers["Content-Type"] == "application/json"
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Hello from OpenAI!",
                            }
                        }
                    ]
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(openai_config, client=client)
        result = await provider.complete("Hi", "openai/gpt-4o-mini")
        assert result == "Hello from OpenAI!"
        await provider.close()

    @pytest.mark.asyncio
    async def test_complete_custom_base_url(self) -> None:
        config = ProviderConfig(
            api_key="test-key",
            model="gpt-4o",
            base_url="https://custom-proxy.internal/v1",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == "https://custom-proxy.internal/v1/chat/completions"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Proxy response"}}]},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(config, client=client)
        result = await provider.complete("Hi", "gpt-4o")
        assert result == "Proxy response"

    @pytest.mark.asyncio
    async def test_complete_http_error(self, openai_config: ProviderConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="Unauthorized API key")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(openai_config, client=client)
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete("Hi", "gpt-4o-mini")
        assert "401" in str(exc_info.value)
        assert "Unauthorized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_network_timeout(self, openai_config: ProviderConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("Connection timed out")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(openai_config, client=client)
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete("Hi", "gpt-4o-mini")
        assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_malformed_json(self, openai_config: ProviderConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected_payload": True})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(openai_config, client=client)
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete("Hi", "gpt-4o-mini")
        assert "Invalid OpenAI response structure" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_success(self, openai_config: ProviderConfig) -> None:
        sse_body = (
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
            b"data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == "https://api.openai.com/v1/chat/completions"
            return httpx.Response(
                200,
                content=sse_body,
                headers={"content-type": "text/event-stream"},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(openai_config, client=client)
        chunks = [c async for c in provider.stream("Hi", "gpt-4o-mini")]

        assert len(chunks) == 3
        assert chunks[0].text == "Hello"
        assert chunks[0].is_finished is False
        assert chunks[1].text == " world"
        assert chunks[1].is_finished is False
        assert chunks[2].text == ""
        assert chunks[2].is_finished is True

    @pytest.mark.asyncio
    async def test_stream_http_error(self, openai_config: ProviderConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAIProvider(openai_config, client=client)
        with pytest.raises(ProviderError) as exc_info:
            async for _ in provider.stream("Hi", "gpt-4o-mini"):
                pass
        assert "500" in str(exc_info.value)


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_complete_success(self, anthropic_config: ProviderConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == "https://api.anthropic.com/v1/messages"
            assert request.headers["x-api-key"] == "sk-ant-test-key"
            assert request.headers["anthropic-version"] == "2023-06-01"
            assert request.headers["content-type"] == "application/json"
            return httpx.Response(
                200,
                json={
                    "id": "msg_123",
                    "content": [{"type": "text", "text": "Hello from Anthropic Claude!"}],
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider(anthropic_config, client=client)
        result = await provider.complete("Hi", "anthropic/claude-3-sonnet")
        assert result == "Hello from Anthropic Claude!"
        await provider.close()

    @pytest.mark.asyncio
    async def test_complete_http_error(self, anthropic_config: ProviderConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="Forbidden key")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider(anthropic_config, client=client)
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete("Hi", "claude-3-sonnet")
        assert "403" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_network_error(self, anthropic_config: ProviderConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Read timed out")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider(anthropic_config, client=client)
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete("Hi", "claude-3-sonnet")
        assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_success(self, anthropic_config: ProviderConfig) -> None:
        sse_body = (
            b'data: {"type": "message_start", "message": {}}\n\n'
            b'data: {"type": "content_block_delta", "index": 0, '
            b'"delta": {"type": "text_delta", "text": "Claude"}}\n\n'
            b'data: {"type": "content_block_delta", "index": 0, '
            b'"delta": {"type": "text_delta", "text": " rocks"}}\n\n'
            b'data: {"type": "message_stop"}\n\n'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == "https://api.anthropic.com/v1/messages"
            return httpx.Response(
                200,
                content=sse_body,
                headers={"content-type": "text/event-stream"},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider(anthropic_config, client=client)
        chunks = [c async for c in provider.stream("Hi", "claude-3-sonnet")]

        assert len(chunks) == 3
        assert chunks[0].text == "Claude"
        assert chunks[0].is_finished is False
        assert chunks[1].text == " rocks"
        assert chunks[1].is_finished is False
        assert chunks[2].text == ""
        assert chunks[2].is_finished is True

    @pytest.mark.asyncio
    async def test_complete_malformed_json(self, anthropic_config: ProviderConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"content": "not-a-list"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider(anthropic_config, client=client)
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete("Hi", "claude-3-sonnet")
        assert "Invalid Anthropic response structure" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_http_error(self, anthropic_config: ProviderConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(529, text="Overloaded")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider(anthropic_config, client=client)
        with pytest.raises(ProviderError) as exc_info:
            async for _ in provider.stream("Hi", "claude-3-sonnet"):
                pass
        assert "529" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_network_error(self, anthropic_config: ProviderConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("Connect timeout")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider(anthropic_config, client=client)
        with pytest.raises(ProviderError) as exc_info:
            async for _ in provider.stream("Hi", "claude-3-sonnet"):
                pass
        assert "Connect timeout" in str(exc_info.value)


class TestGroqProvider:
    @pytest.mark.asyncio
    async def test_complete_success(self, groq_config: ProviderConfig) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == "https://api.groq.com/openai/v1/chat/completions"
            assert request.headers["Authorization"] == "Bearer gsk-test-key"
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Fast response from Groq!",
                            }
                        }
                    ]
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = GroqProvider(groq_config, client=client)
        result = await provider.complete("Hi", "groq/llama-3.1-8b")
        assert result == "Fast response from Groq!"
        await provider.close()

    @pytest.mark.asyncio
    async def test_stream_success(self, groq_config: ProviderConfig) -> None:
        sse_body = (
            b'data: {"choices": [{"delta": {"content": "Fast"}}\n\n'
            b'data: {"choices": [{"delta": {"content": " Groq"}}]}\n\n'
            b"data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_body,
                headers={"content-type": "text/event-stream"},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = GroqProvider(groq_config, client=client)
        chunks = [c async for c in provider.stream("Hi", "llama-3.1-8b")]

        # First line was malformed JSON in SSE, second succeeded
        assert len(chunks) == 2
        assert chunks[0].text == " Groq"
        assert chunks[0].is_finished is False
        assert chunks[1].text == ""
        assert chunks[1].is_finished is True


class TestBaseProviderLifecycle:
    @pytest.mark.asyncio
    async def test_async_context_manager(self, openai_config: ProviderConfig) -> None:
        async with OpenAIProvider(openai_config) as provider:
            client = provider._get_client()
            assert not client.is_closed
        assert client.is_closed
