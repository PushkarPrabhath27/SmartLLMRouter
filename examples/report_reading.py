"""Example 5: Reading the Project Health Report."""

from __future__ import annotations

import asyncio

from smartroute import Router


async def main() -> None:
    router = Router()

    # Query project analytics report
    report = await router.report()
    print(f"Total decisions: {report.total_decisions}")
    print(f"Total cost: ${report.total_cost_usd:.4f}")
    print(f"Avg latency: {report.average_latency_ms:.2f}ms")

    print("\nModel distribution:")
    for model, count in report.model_distribution.items():
        print(f"  {model}: {count}")

    print("\nAdapted buckets:")
    for adaptation in report.adapted_buckets:
        print(f"  {adaptation['key']}: {adaptation['old_tier']} -> {adaptation['new_tier']}")

    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
