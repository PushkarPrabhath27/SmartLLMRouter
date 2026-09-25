"""Abstract base class for LLM providers (spec 02, spec 04).

Defines the unified asynchronous completion and streaming interface that all
provider adapters (OpenAI, Anthropic, Groq) implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import httpx

from smartroute.config.schema import ProviderConfig
from smartroute.types import StreamChunk


class BaseProvider(ABC):
    """Abstract base class for asynchronous LLM providers.

    All implementations manage an underlying ``httpx.AsyncClient``, handle
    provider-specific authentication and endpoints, and normalize streaming
    chunks into :class:`~smartroute.types.StreamChunk` instances.
    """

    def __init__(
        self,
        config: ProviderConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize provider with configuration.

        Args:
            config: Validated provider configuration containing api_key, model,
                timeout, max_retries, and optional base_url.
            client: Optional custom or mocked httpx.AsyncClient.
        """
        self.config = config
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        """Return or lazily create the httpx.AsyncClient instance."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=float(self.config.timeout),
                transport=httpx.AsyncHTTPTransport(retries=self.config.max_retries),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> BaseProvider:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Async context manager exit."""
        await self.close()

    @abstractmethod
    async def complete(self, prompt: str, model: str) -> str:
        """Generate a complete text response asynchronously.

        Args:
            prompt: User prompt string.
            model: Model identifier.

        Returns:
            Completed response string.

        Raises:
            ProviderError: On any HTTP, network, or provider failure.
        """
        ...

    @abstractmethod
    def stream(self, prompt: str, model: str) -> AsyncIterator[StreamChunk]:
        """Stream response chunks asynchronously.

        Args:
            prompt: User prompt string.
            model: Model identifier.

        Yields:
            StreamChunk instances with ``is_finished=False`` for text deltas,
            and a final chunk with ``is_finished=True`` and empty text.

        Raises:
            ProviderError: On any HTTP, network, or provider failure.
        """
        ...
