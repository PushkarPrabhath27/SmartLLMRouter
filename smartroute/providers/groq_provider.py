"""Groq API provider implementation (spec 02, spec 04).

Supports asynchronous completions and SSE streaming for Groq models
(e.g., llama-3.1-8b) using Groq's OpenAI-compatible completions endpoint.
"""

from __future__ import annotations

import httpx

from smartroute.config.schema import ProviderConfig
from smartroute.providers.openai_provider import OpenAIProvider

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(OpenAIProvider):
    """Asynchronous provider adapter for Groq's OpenAI-compatible API."""

    def __init__(
        self,
        config: ProviderConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize Groq provider.

        Args:
            config: Provider configuration.
            client: Optional injected httpx.AsyncClient.
        """
        super().__init__(
            config=config,
            client=client,
            default_base_url=DEFAULT_GROQ_BASE_URL,
            provider_prefix="groq",
        )
