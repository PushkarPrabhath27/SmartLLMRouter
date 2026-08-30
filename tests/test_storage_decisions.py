"""Unit tests for smartroute.storage.decisions (Phase 1, module 4)."""

from datetime import timedelta

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
from smartroute.types import DecisionRecord, SignalRecord


def decision_ts(seconds_offset: int):
    """Timestamp helper: now + offset (positive = newer)."""
    from smartroute.storage.decisions import utc_now

    return utc_now() + timedelta(seconds=seconds_offset)


def _minimal_decision(**overrides: object) -> DecisionRecord:
    base: dict[str, object] = {
        "prompt_hash": "h" * 64,
        "task_type": "general",
        "complexity": 0.2,
        "complexity_bucket": "low",
        "confidence": 0.6,
        "model_used": "groq/llama-3.1-8b",
        "provider_key": "groq",
        "reason": "Default routing",
    }
    base.update(overrides)
    return DecisionRecord(**base)  # type: ignore[arg-type]


class TestStoreDecision:
    async def test_store_returns_id_and_roundtrips_all_fields(
        self, conn: aiosqlite.Connection, sample_decision: DecisionRecord
    ) -> None:
        returned = await store_decision(conn, sample_decision)
        assert returned == sample_decision.id
        recent = await get_recent_decisions(conn, limit=1)
        assert len(recent) == 1
        loaded = recent[0]
        assert loaded.id == sample_decision.id
        assert loaded.prompt_hash == sample_decision.prompt_hash
        assert loaded.prompt_preview == sample_decision.prompt_preview
        assert loaded.task_type == sample_decision.task_type
        assert loaded.complexity == sample_decision.complexity
        assert loaded.complexity_bucket == sample_decision.complexity_bucket
        assert loaded.confidence == sample_decision.confidence
        assert loaded.model_used == sample_decision.model_used
        assert loaded.provider_key == sample_decision.provider_key
        assert loaded.estimated_cost_usd == sample_decision.estimated_cost_usd
        assert loaded.actual_cost_usd == sample_decision.actual_cost_usd
        assert loaded.latency_ms == sample_decision.latency_ms
        assert loaded.was_adapted == sample_decision.was_adapted
        assert loaded.override_applied == sample_decision.override_applied
        assert loaded.reason == sample_decision.reason
        assert loaded.conversation_id == sample_decision.conversation_id
        assert loaded.turn_number == sample_decision.turn_number
        assert loaded.features_json == sample_decision.features_json
        assert loaded.timestamp == sample_decision.timestamp

    async def test_store_minimal_decision_optional_fields_default_none(
        self, conn: aiosqlite.Connection
    ) -> None:
        decision = _minimal_decision()
        await store_decision(conn, decision)
        loaded = (await get_recent_decisions(conn))[0]
        assert loaded.prompt_preview is None
        assert loaded.estimated_cost_usd is None
        assert loaded.actual_cost_usd is None
        assert loaded.latency_ms is None
        assert loaded.was_adapted is False
        assert loaded.conversation_id is None

    async def test_store_was_adapted_true_roundtrips(self, conn: aiosqlite.Connection) -> None:
        decision = _minimal_decision(was_adapted=True)
        await store_decision(conn, decision)
        loaded = (await get_recent_decisions(conn))[0]
        assert loaded.was_adapted is True

    async def test_duplicate_id_raises_storage_error(self, conn: aiosqlite.Connection) -> None:
        decision = _minimal_decision(id="dup")
        await store_decision(conn, decision)
        with pytest.raises(StorageError):
            await store_decision(conn, _minimal_decision(id="dup"))

    async def test_store_full_prompt_never_persisted(
        self, conn: aiosqlite.Connection, sample_decision: DecisionRecord
    ) -> None:
        """Privacy guarantee: only hash + preview columns exist for prompt data."""
        await store_decision(conn, sample_decision)
        async with conn.execute("SELECT * FROM decisions") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        row_dict = dict(row)
        assert "prompt_hash" in row_dict
        assert len(row["prompt_preview"]) <= 100


class TestGetRecentDecisions:
    async def test_empty_database_returns_empty_list(self, conn: aiosqlite.Connection) -> None:
        assert await get_recent_decisions(conn) == []

    async def test_orders_newest_first_and_respects_limit(self, conn: aiosqlite.Connection) -> None:
        for i in range(5):
            decision = _minimal_decision(
                id=f"d{i}",
                timestamp=decision_ts(i),
            )
            await store_decision(conn, decision)
        recent = await get_recent_decisions(conn, limit=3)
        assert [r.id for r in recent] == ["d4", "d3", "d2"]


class TestGetDecisionsByConversation:
    async def test_filters_and_orders_chronologically(self, conn: aiosqlite.Connection) -> None:
        await store_decision(
            conn, _minimal_decision(id="a", conversation_id="c1", timestamp=decision_ts(1))
        )
        await store_decision(conn, _minimal_decision(id="b", conversation_id="c2"))
        await store_decision(
            conn, _minimal_decision(id="c", conversation_id="c1", timestamp=decision_ts(3))
        )
        result = await get_decisions_by_conversation(conn, "c1")
        assert [r.id for r in result] == ["a", "c"]

    async def test_unknown_conversation_returns_empty(self, conn: aiosqlite.Connection) -> None:
        assert await get_decisions_by_conversation(conn, "missing") == []


class TestGetDecisionByPromptHash:
    async def test_finds_newest_match_within_window(self, conn: aiosqlite.Connection) -> None:
        await store_decision(
            conn, _minimal_decision(id="old", prompt_hash="h" * 64, timestamp=decision_ts(-30))
        )
        await store_decision(
            conn, _minimal_decision(id="new", prompt_hash="h" * 64, timestamp=decision_ts(-5))
        )
        found = await get_decision_by_prompt_hash(conn, "h" * 64, window_seconds=60)
        assert found is not None
        assert found.id == "new"

    async def test_ignores_match_outside_window(self, conn: aiosqlite.Connection) -> None:
        await store_decision(
            conn, _minimal_decision(id="old", prompt_hash="h" * 64, timestamp=decision_ts(-120))
        )
        assert await get_decision_by_prompt_hash(conn, "h" * 64, window_seconds=60) is None

    async def test_no_match_returns_none(self, conn: aiosqlite.Connection) -> None:
        assert await get_decision_by_prompt_hash(conn, "z" * 64) is None


class TestSignals:
    async def test_store_and_fetch_signals(self, conn: aiosqlite.Connection) -> None:
        await store_decision(conn, _minimal_decision(id="d1"))
        signal = SignalRecord(decision_id="d1", signal_type="hard_regen", signal_value=-0.3)
        await store_signal(conn, signal)
        signals = await get_signals_for_decision(conn, "d1")
        assert len(signals) == 1
        loaded = signals[0]
        assert loaded.id == signal.id
        assert loaded.signal_type == "hard_regen"
        assert loaded.signal_value == -0.3
        assert loaded.detection_method == "auto"
        assert loaded.detected_at == signal.detected_at

    async def test_signal_for_unknown_decision_raises_storage_error(
        self, conn: aiosqlite.Connection
    ) -> None:
        with pytest.raises(StorageError):
            await store_signal(
                conn,
                SignalRecord(decision_id="missing", signal_type="acceptance", signal_value=0.05),
            )

    async def test_signals_scoped_to_decision(self, conn: aiosqlite.Connection) -> None:
        await store_decision(conn, _minimal_decision(id="d1"))
        await store_decision(conn, _minimal_decision(id="d2"))
        await store_signal(
            conn, SignalRecord(decision_id="d1", signal_type="acceptance", signal_value=0.05)
        )
        assert await get_signals_for_decision(conn, "d2") == []
