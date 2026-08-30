"""Shared pytest fixtures for SmartRoute storage tests.

All storage tests run against aiosqlite in-memory databases per
``specs/11_TESTING_STRATEGY.md``. WAL behavior is additionally exercised
against a temp file where noted (spec 11 allows ":memory: vs file").
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiosqlite
import pytest

from smartroute.storage.schema import SCHEMA_STATEMENTS
from smartroute.types import DecisionRecord


@pytest.fixture
async def conn() -> AsyncIterator[aiosqlite.Connection]:
    """Schema-initialized in-memory connection with foreign keys enabled."""
    async with aiosqlite.connect(":memory:") as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        for statement in SCHEMA_STATEMENTS:
            await db.execute(statement)
        await db.commit()
        yield db


@pytest.fixture
def sample_decision() -> DecisionRecord:
    """A fully populated decision for CRUD tests."""
    return DecisionRecord(
        prompt_hash="a" * 64,
        prompt_preview="Write a function to parse CSV files",
        task_type="code",
        complexity=0.45,
        complexity_bucket="medium",
        confidence=0.8,
        model_used="openai/gpt-4o-mini",
        provider_key="openai",
        estimated_cost_usd=0.00012,
        actual_cost_usd=0.00011,
        latency_ms=120,
        was_adapted=False,
        override_applied=None,
        reason="Default routing: code task, medium complexity (0.45) -> openai",
        conversation_id="conv_1",
        turn_number=1,
        features_json='{"token_count": 40}',
    )
