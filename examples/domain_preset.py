"""Example 4: Using a Domain Preset."""

from __future__ import annotations

import asyncio
from pathlib import Path

from smartroute import Router


async def main() -> None:
    preset_path = str(Path(__file__).parent / "web_dev_preset.yaml")
    router = Router(config_path=preset_path)

    result = await router.complete("Refactor this React component to use hooks")
    print(f"Task type: {result.meta.task_type}")
    print(f"Complexity: {result.meta.complexity_bucket}")
    print(f"Model: {result.meta.model}")
    print(f"Reason: {result.meta.reason}")
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
