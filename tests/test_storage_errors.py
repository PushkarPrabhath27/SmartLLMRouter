"""Error-path tests: every CRUD function wraps sqlite failures in StorageError.

Uses a schema-less in-memory connection, so every query fails with
"no such table" — exercising the StorageError branches without mocks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiosqlite
import pytest

from smartroute.exceptions import StorageError
from smartroute.storage.decisions import (
    get_decision_by_prompt_hash,
    get_decisions_by_conversation,
    get_recent_decisions,
    get_signals_for_decision,
    store_decision,
    store_signal,
)
from smartroute.storage.reputation import (
    get_adaptations,
    get_all_reputation,
    get_reputation,
    record_adaptation,
    reset_reputation,
    update_reputation,
)
from smartroute.types import DecisionRecord, SignalRecord


@pytest.fixture
async def bare_conn() -> AsyncIterator[aiosqlite.Connection]:
    """Open in-memory connection with no tables, so queries fail."""
    async with aiosqlite.connect(":memory:") as db:
        db.row_factory = aiosqlite.Row
        yield db


def _decision() -> DecisionRecord:
    return DecisionRecord(
        prompt_hash="h" * 64,
        task_type="code",
        complexity=0.5,
        complexity_bucket="medium",
        confidence=0.8,
        model_used="openai/gpt-4o-mini",
        provider_key="openai",
        reason="r",
    )


class TestDecisionErrorPaths:
    async def test_store_decision_failure(self, bare_conn: aiosqlite.Connection) -> None:
        with pytest.raises(StorageError):
            await store_decision(bare_conn, _decision())

    async def test_get_recent_decisions_failure(self, bare_conn: aiosqlite.Connection) -> None:
        with pytest.raises(StorageError):
            await get_recent_decisions(bare_conn)

    async def test_get_decisions_by_conversation_failure(
        self, bare_conn: aiosqlite.Connection
    ) -> None:
        with pytest.raises(StorageError):
            await get_decisions_by_conversation(bare_conn, "c1")

    async def test_get_decision_by_prompt_hash_failure(
        self, bare_conn: aiosqlite.Connection
    ) -> None:
        with pytest.raises(StorageError):
            await get_decision_by_prompt_hash(bare_conn, "h" * 64)

    async def test_store_signal_failure(self, bare_conn: aiosqlite.Connection) -> None:
        with pytest.raises(StorageError):
            await store_signal(
                bare_conn,
                SignalRecord(decision_id="d", signal_type="acceptance", signal_value=0.05),
            )

    async def test_get_signals_for_decision_failure(self, bare_conn: aiosqlite.Connection) -> None:
        with pytest.raises(StorageError):
            await get_signals_for_decision(bare_conn, "d1")


class TestReputationErrorPaths:
    async def test_get_reputation_failure(self, bare_conn: aiosqlite.Connection) -> None:
        with pytest.raises(StorageError):
            await get_reputation(bare_conn, "code_low", "low")

    async def test_update_reputation_failure(self, bare_conn: aiosqlite.Connection) -> None:
        with pytest.raises(StorageError):
            await update_reputation(bare_conn, "code_low", "low", 0.5, 1)

    async def test_get_all_reputation_failure(self, bare_conn: aiosqlite.Connection) -> None:
        with pytest.raises(StorageError):
            await get_all_reputation(bare_conn)

    async def test_record_adaptation_failure(self, bare_conn: aiosqlite.Connection) -> None:
        with pytest.raises(StorageError):
            await record_adaptation(bare_conn, "code_low", "low", "medium", 0.25)

    async def test_get_adaptations_failure(self, bare_conn: aiosqlite.Connection) -> None:
        with pytest.raises(StorageError):
            await get_adaptations(bare_conn)

    async def test_reset_reputation_failure(self, bare_conn: aiosqlite.Connection) -> None:
        with pytest.raises(StorageError):
            await reset_reputation(bare_conn)
