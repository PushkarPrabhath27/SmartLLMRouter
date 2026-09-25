"""Example 2: Streaming with FastAPI and SmartRoute."""

from __future__ import annotations

from collections.abc import AsyncIterator

try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
except ImportError:
    FastAPI = None  # type: ignore[misc,assignment]
    StreamingResponse = None  # type: ignore[misc,assignment]

from smartroute import Router

if FastAPI is not None:
    app = FastAPI(title="SmartRoute Streaming Demo")
    router = Router()

    @app.post("/chat")
    async def chat(prompt: str) -> StreamingResponse:
        """Stream LLM responses with metadata in server-sent events."""

        async def event_generator() -> AsyncIterator[str]:
            async for chunk in router.stream(prompt):
                if chunk.is_finished:
                    if chunk.meta:
                        yield f"event: meta\ndata: {chunk.meta.reason}\n\n"
                    break
                yield f"data: {chunk.text}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )
