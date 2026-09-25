"""Anthropic Messages API provider implementation (spec 02, spec 04).

Supports asynchronous completions and SSE streaming for Anthropic Claude models.
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

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096


def _clean_model(model: str) -> str:
    """Remove 'anthropic/' prefix if present."""
    if model.startswith("anthropic/"):
        return model[len("anthropic/") :]
    return model


class AnthropicProvider(BaseProvider):
    """Asynchronous provider adapter for Anthropic's Messages API."""

    def __init__(
        self,
        config: ProviderConfig,
        client: httpx.AsyncClient | None = None,
        default_base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
    ) -> None:
        """Initialize Anthropic provider.

        Args:
            config: Provider configuration.
            client: Optional injected httpx.AsyncClient.
            default_base_url: Default API base URL if not set in config.
        """
        super().__init__(config, client=client)
        base = config.base_url or default_base_url
        self.base_url = base.rstrip("/")

    def _headers(self) -> dict[str, str]:
        """Build Anthropic headers with API key and version."""
        return {
            "x-api-key": self.config.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }

    async def complete(self, prompt: str, model: str) -> str:
        """Generate a complete text response using Anthropic Messages API.

        Args:
            prompt: User prompt string.
            model: Model identifier.

        Returns:
            Completed assistant response text.

        Raises:
            ProviderError: If the request fails or returns a non-200 status code.
        """
        client = self._get_client()
        url = f"{self.base_url}/messages"
        resolved_model = _clean_model(model)
        payload = {
            "model": resolved_model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = await client.post(url, json=payload, headers=self._headers())
        except httpx.RequestError as exc:
            logger.warning("Anthropic network error: %s", exc)
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        if response.status_code != 200:
            logger.warning(
                "Anthropic returned non-200 status %d: %s",
                response.status_code,
                response.text,
            )
            raise ProviderError(f"Anthropic error {response.status_code}: {response.text}")

        try:
            data = response.json()
            blocks = data["content"]
            if not isinstance(blocks, list):
                raise ValueError("Anthropic 'content' field must be a list")
            text_parts = [
                str(b["text"]) for b in blocks if isinstance(b, dict) and b.get("type") == "text"
            ]
            return "".join(text_parts)
        except (KeyError, ValueError, AttributeError, TypeError) as exc:
            logger.warning("Failed to parse Anthropic response: %s", exc)
            raise ProviderError(f"Invalid Anthropic response structure: {exc}") from exc

    async def stream(self, prompt: str, model: str) -> AsyncIterator[StreamChunk]:
        """Stream response chunks via Anthropic SSE events.

        Args:
            prompt: User prompt string.
            model: Model identifier.

        Yields:
            StreamChunk items with text deltas, terminating with an empty final chunk.

        Raises:
            ProviderError: If the connection or stream fails.
        """
        client = self._get_client()
        url = f"{self.base_url}/messages"
        resolved_model = _clean_model(model)
        payload = {
            "model": resolved_model,
            "max_tokens": DEFAULT_MAX_TOKENS,
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
                        "Anthropic stream returned non-200 status %d: %s",
                        response.status_code,
                        err_msg,
                    )
                    raise ProviderError(f"Anthropic stream error {response.status_code}: {err_msg}")

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    try:
                        event_data = json.loads(data_str)
                        event_type = event_data.get("type")
                        if event_type == "content_block_delta":
                            delta = event_data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    yield StreamChunk(text=text, is_finished=False, meta=None)
                        elif event_type == "message_stop":
                            break
                    except json.JSONDecodeError:
                        continue
        except httpx.RequestError as exc:
            logger.warning("Anthropic streaming network error: %s", exc)
            raise ProviderError(f"Anthropic stream failed: {exc}") from exc

        yield StreamChunk(text="", is_finished=True, meta=None)
