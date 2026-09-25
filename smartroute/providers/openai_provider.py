"""OpenAI Chat Completions API provider implementation (spec 02, spec 04).

Supports asynchronous completions and SSE streaming for OpenAI models
such as gpt-4o and gpt-4o-mini.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from smartroute.config.schema import ProviderConfig
from smartroute.exceptions import ProviderError
from smartroute.providers.base import BaseProvider
from smartroute.types import StreamChunk

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _clean_model(model: str, prefix: str = "openai") -> str:
    """Remove provider namespace prefix if present (e.g. 'openai/gpt-4o' -> 'gpt-4o')."""
    if model.startswith(f"{prefix}/"):
        return model[len(prefix) + 1 :]
    return model


class OpenAIProvider(BaseProvider):
    """Asynchronous provider adapter for OpenAI's Chat Completions API."""

    def __init__(
        self,
        config: ProviderConfig,
        client: httpx.AsyncClient | None = None,
        default_base_url: str = DEFAULT_OPENAI_BASE_URL,
        provider_prefix: str = "openai",
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            config: Provider configuration.
            client: Optional injected httpx.AsyncClient.
            default_base_url: Default API base URL if not set in config.
            provider_prefix: Prefix used for model names (default 'openai').
        """
        super().__init__(config, client=client)
        base = config.base_url or default_base_url
        self.base_url = base.rstrip("/")
        self.provider_prefix = provider_prefix

    def _headers(self) -> dict[str, str]:
        """Build standard authorization and content headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    async def complete(self, prompt: str, model: str) -> str:
        """Generate a complete response using OpenAI Chat Completions.

        Args:
            prompt: User prompt string.
            model: Model identifier.

        Returns:
            Completed assistant response text.

        Raises:
            ProviderError: If the request fails or returns a non-200 status code.
        """
        client = self._get_client()
        url = f"{self.base_url}/chat/completions"
        resolved_model = _clean_model(model, self.provider_prefix)
        payload = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = await client.post(url, json=payload, headers=self._headers())
        except httpx.RequestError as exc:
            logger.warning("OpenAI network error: %s", exc)
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        if response.status_code != 200:
            logger.warning(
                "OpenAI returned non-200 status %d: %s",
                response.status_code,
                response.text,
            )
            raise ProviderError(f"OpenAI error {response.status_code}: {response.text}")

        try:
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, ValueError, TypeError, AttributeError) as exc:
            logger.warning("Failed to parse OpenAI response: %s", exc)
            raise ProviderError(f"Invalid OpenAI response structure: {exc}") from exc

    async def stream(self, prompt: str, model: str) -> AsyncIterator[StreamChunk]:
        """Stream response chunks via Server-Sent Events (SSE).

        Args:
            prompt: User prompt string.
            model: Model identifier.

        Yields:
            StreamChunk items with text deltas, terminating with an empty final chunk.

        Raises:
            ProviderError: If the connection or stream fails.
        """
        client = self._get_client()
        url = f"{self.base_url}/chat/completions"
        resolved_model = _clean_model(model, self.provider_prefix)
        payload = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        try:
            async with client.stream(
                "POST", url, json=payload, headers=self._headers()
            ) as response:
                if response.status_code != 200:
                    err_bytes = await response.aread()
                    err_msg = err_bytes.decode("utf-8", errors="replace")
                    logger.warning(
                        "OpenAI stream returned non-200 status %d: %s",
                        response.status_code,
                        err_msg,
                    )
                    raise ProviderError(f"OpenAI stream error {response.status_code}: {err_msg}")

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        choices = chunk_json.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield StreamChunk(text=content, is_finished=False, meta=None)
                    except json.JSONDecodeError:
                        continue
        except httpx.RequestError as exc:
            logger.warning("OpenAI streaming network error: %s", exc)
            raise ProviderError(f"OpenAI stream failed: {exc}") from exc

        yield StreamChunk(text="", is_finished=True, meta=None)
