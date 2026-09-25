"""Example 3: Custom Programmatic Override Hook."""

from __future__ import annotations

import asyncio

from smartroute import ConversationContext, Router


def legal_override(prompt: str, context: ConversationContext | None) -> str | None:
    """Force Anthropic for sensitive legal prompts or long multi-turn sessions."""
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ["contract", "legal", "lawyer", "liability"]):
        return "anthropic"
    if context and context.turn_number > 5:
        return "anthropic"
    return None


async def main() -> None:
    router = Router(override_hook=legal_override)
    result = await router.complete("Draft a software liability clause")
    print(f"Model used: {result.meta.model}")
    print(f"Override applied: {result.meta.override_applied}")
    print(f"Reason: {result.meta.reason}")
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
