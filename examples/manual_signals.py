"""Example 6: Manual Feedback Signal Reporting."""

from __future__ import annotations

import asyncio

from smartroute import Router


async def main() -> None:
    router = Router()

    # Route a prompt and capture decision ID
    result = await router.complete("Explain asynchronous programming in Python")
    decision_id = result.meta.decision_id
    print(f"Response: {result.text[:60]}...")
    print(f"Decision ID: {decision_id}")

    # Explicit user feedback from UI (e.g. thumbs down)
    await router.report_signal(
        decision_id=decision_id,
        signal_type="thumbs_down",
    )
    print("Reported explicit thumbs_down feedback.")

    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
