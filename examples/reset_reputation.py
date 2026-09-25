"""Example 7: Resetting Learned Reputation Data."""

from __future__ import annotations

import asyncio

from smartroute import Router


async def main() -> None:
    router = Router()

    # Reset reputation scores for all buckets
    await router.reset_reputation()
    print("Reset reputation scores for all buckets.")

    # Or reset just one specific bucket
    await router.reset_reputation(bucket_key="code_low")
    print("Reset reputation scores for bucket 'code_low'.")

    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
