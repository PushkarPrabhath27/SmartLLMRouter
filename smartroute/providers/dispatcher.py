"""Provider dispatcher: fallback chain execution and streaming failover (spec 06, spec 13).

Manages provider instances initialized from configuration and executes ordered
fallback chains with latency tracking and comprehensive attempt logging.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from smartroute.config.schema import Config
from smartroute.exceptions import ProviderError
from smartroute.providers.anthropic_provider import AnthropicProvider
from smartroute.providers.base import BaseProvider
from smartroute.providers.groq_provider import GroqProvider
from smartroute.providers.openai_provider import OpenAIProvider
from smartroute.types import StreamChunk

logger = logging.getLogger(__name__)


def _fully_qualified_model(provider_key: str, configured_model: str) -> str:
    """Ensure a model name has a provider prefix."""
    if "/" in configured_model:
        return configured_model
    return f"{provider_key}/{configured_model}"


@dataclass(frozen=True)
class DispatchResult:
    """Result of a successful provider call through ProviderDispatcher.

    Attributes:
        text: Response text returned by the successful provider.
        provider_used: Provider key that fulfilled the request (e.g. 'openai').
        model_used: Fully qualified model ID (e.g. 'openai/gpt-4o-mini').
        latency_ms: Measured elapsed latency in milliseconds.
        attempts: List of failed attempts before success, if any.
    """

    text: str
    provider_used: str
    model_used: str
    latency_ms: int
    attempts: list[dict[str, str]] = field(default_factory=list)


class ProviderDispatcher:
    """Dispatches requests across LLM providers with automatic fallback."""

    def __init__(
        self,
        config: Config,
        providers: dict[str, BaseProvider] | None = None,
    ) -> None:
        """Initialize dispatcher with configuration and provider adapters.

        Args:
            config: Validated SmartRoute configuration.
            providers: Optional explicit map of provider instances (useful for testing).
        """
        self.config = config
        if providers is not None:
            self._providers = dict(providers)
        else:
            self._providers = {}
            for key, pcfg in config.providers.items():
                if key == "openai":
                    self._providers[key] = OpenAIProvider(pcfg)
                elif key == "anthropic":
                    self._providers[key] = AnthropicProvider(pcfg)
                elif key == "groq":
                    self._providers[key] = GroqProvider(pcfg)

    async def complete(self, fallback_chain: list[str], prompt: str) -> DispatchResult:
        """Execute complete() across fallback_chain until one provider succeeds.

        Args:
            fallback_chain: Ordered list of provider keys to try.
            prompt: User prompt string.

        Returns:
            DispatchResult containing response text, provider used, and latency.

        Raises:
            ProviderError: If all providers in the fallback chain fail.
        """
        if not fallback_chain:
            raise ProviderError("No providers specified in fallback chain", attempts=[])

        attempts: list[dict[str, str]] = []

        for provider_key in fallback_chain:
            provider = self._providers.get(provider_key)
            if provider is None:
                logger.warning(
                    "provider '%s' not configured; skipping in fallback chain",
                    provider_key,
                )
                attempts.append(
                    {
                        "provider": provider_key,
                        "model": "unknown",
                        "error": f"Provider '{provider_key}' is not configured",
                    }
                )
                continue

            raw_model = self.config.providers[provider_key].model
            model = _fully_qualified_model(provider_key, raw_model)
            call_start = time.perf_counter()
            try:
                text = await provider.complete(prompt, model)
                latency_ms = int((time.perf_counter() - call_start) * 1000)
                return DispatchResult(
                    text=text,
                    provider_used=provider_key,
                    model_used=model,
                    latency_ms=latency_ms,
                    attempts=attempts,
                )
            except Exception as exc:
                logger.warning(
                    "provider '%s' failed for model '%s': %s",
                    provider_key,
                    model,
                    exc,
                )
                attempts.append(
                    {
                        "provider": provider_key,
                        "model": model,
                        "error": str(exc),
                    }
                )

        raise ProviderError(
            f"All providers in fallback chain {fallback_chain} failed",
            attempts=attempts,
        )

    async def stream(self, fallback_chain: list[str], prompt: str) -> AsyncIterator[StreamChunk]:
        """Execute stream() across fallback_chain with failover before first chunk.

        Args:
            fallback_chain: Ordered list of provider keys to try.
            prompt: User prompt string.

        Yields:
            StreamChunk instances from the succeeding provider.

        Raises:
            ProviderError: If all providers fail to initialize the stream.
        """
        if not fallback_chain:
            raise ProviderError("No providers specified in fallback chain", attempts=[])

        attempts: list[dict[str, str]] = []

        for provider_key in fallback_chain:
            provider = self._providers.get(provider_key)
            if provider is None:
                logger.warning(
                    "provider '%s' not configured; skipping in fallback chain",
                    provider_key,
                )
                attempts.append(
                    {
                        "provider": provider_key,
                        "model": "unknown",
                        "error": f"Provider '{provider_key}' is not configured",
                    }
                )
                continue

            raw_model = self.config.providers[provider_key].model
            model = _fully_qualified_model(provider_key, raw_model)

            try:
                stream_gen = provider.stream(prompt, model)
                first_chunk = await anext(stream_gen)
            except StopAsyncIteration:
                yield StreamChunk(text="", is_finished=True, meta=None)
                return
            except Exception as exc:
                logger.warning(
                    "provider '%s' stream failed to initialize for model '%s': %s",
                    provider_key,
                    model,
                    exc,
                )
                attempts.append(
                    {
                        "provider": provider_key,
                        "model": model,
                        "error": str(exc),
                    }
                )
                continue

            # Stream initialized successfully; yield first chunk and stream the rest
            yield first_chunk
            async for chunk in stream_gen:
                yield chunk
            return

        raise ProviderError(
            f"All providers in fallback chain {fallback_chain} failed",
            attempts=attempts,
        )

    async def close(self) -> None:
        """Close all initialized provider instances."""
        for provider in self._providers.values():
            await provider.close()

    async def __aenter__(self) -> ProviderDispatcher:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Async context manager exit."""
        await self.close()
