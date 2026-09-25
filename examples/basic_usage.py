"""Example 1: Basic Usage of SmartRoute Router."""

from __future__ import annotations

import asyncio

from smartroute import Router


async def main() -> None:
    router = Router()
    result = await router.complete("Explain Python decorators in simple terms")

    print(result.text)
    print(f"Model: {result.meta.model}")
    print(f"Why: {result.meta.reason}")
    print(f"Complexity: {result.meta.complexity:.2f}")
    print(f"Confidence: {result.meta.confidence:.2f}")
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
