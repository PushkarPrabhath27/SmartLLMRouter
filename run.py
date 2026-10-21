"""SmartRoute CLI Runner & Interactive Playground.

Execute prompt routing and streaming completions directly from the terminal:
    uv run python run.py "Explain Python decorators in simple terms"
    uv run python run.py  # Interactive mode
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from smartroute import Router
from smartroute.types import RoutingResult


def _find_config() -> str | None:
    for candidate in ["smartroute.local.yaml", "smartroute.yaml"]:
        if Path(candidate).is_file():
            return candidate
    return None


async def run_single_prompt(router: Router, prompt: str) -> None:
    print(f"\n[Prompt]: {prompt}\n")
    print("-" * 60)
    print("Routing and streaming response...\n")

    full_text = []
    meta = None

    async for chunk in router.stream(prompt):
        if chunk.is_finished:
            meta = chunk.meta
            break
        print(chunk.text, end="", flush=True)
        full_text.append(chunk.text)

    print("\n" + "-" * 60)
    if meta:
        print(f"Model:      {meta.model}")
        print(f"Task Type:  {meta.task_type} (Bucket: {meta.complexity_bucket}, Score: {meta.complexity:.2f})")
        print(f"Rationale:  {meta.reason}")
        print(f"Latency:    {meta.latency_ms}ms")
        print(f"Est. Cost:  ${meta.estimated_cost_usd:.6f}")
    print("-" * 60 + "\n")


async def main() -> None:
    config_file = _find_config()
    router = Router(config_path=config_file)

    # If arguments provided on CLI, run once
    args = sys.argv[1:]
    if args:
        prompt = " ".join(args)
        await run_single_prompt(router, prompt)
        await router.close()
        return

    # Otherwise run demo prompt and interactive loop
    default_prompt = "Explain Python decorators in simple terms with a 3-line code example."
    print("=" * 60)
    print("SmartRoute - In-Process Adaptive LLM Prompt Router")
    if config_file:
        print(f"Loaded config: {config_file}")
    print("=" * 60)

    print("\n[Running Default Demonstration]")
    await run_single_prompt(router, default_prompt)

    # Show project stats from embedded SQLite
    report = await router.report()
    print(f"[SQLite WAL Analytics]: Total decisions={report.total_decisions}, Avg latency={report.average_latency_ms:.1f}ms")

    print("\nEnter a prompt to route (or 'exit' to quit):")
    try:
        while True:
            user_input = input("\nsmartroute> ").strip()
            if not user_input or user_input.lower() in ("exit", "quit", "q"):
                break
            await run_single_prompt(router, user_input)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await router.close()
        print("\nSession closed.")


if __name__ == "__main__":
    asyncio.run(main())
