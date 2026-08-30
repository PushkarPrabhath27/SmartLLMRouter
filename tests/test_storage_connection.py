"""Unit tests for smartroute.storage.connection (Phase 1, module 6)."""

from datetime import timedelta

import aiosqlite
import pytest

from smartroute.exceptions import StorageError
from smartroute.storage.connection import Storage
from smartroute.storage.decisions import utc_now
from smartroute.storage.schema import SCHEMA_VERSION
from smartroute.types import DecisionRecord, SignalRecord


def _decision(**overrides: object) -> DecisionRecord:
    base: dict[str, object] = {
        "prompt_hash": "h" * 64,
        "task_type": "code",
        "complexity": 0.45,
        "complexity_bucket": "medium",
        "confidence": 0.8,
        "model_used": "openai/gpt-4o-mini",
        "provider_key": "openai",
        "reason": "Default routing",
        "estimated_cost_usd": 0.0001,
        "actual_cost_usd": 0.0002,
        "latency_ms": 100,
    }
    base.update(overrides)
    return DecisionRecord(**base)  # type: ignore[arg-type]


class TestLifecycle:
    async def test_context_manager_connects_and_closes(self) -> None:
        async with Storage(":memory:") as storage:
            await storage.store_decision(_decision())
            assert await storage.get_recent_decisions() != []
        with pytest.raises(StorageError):
            await storage.get_recent_decisions()

    async def test_double_connect_raises(self) -> None:
        storage = Storage(":memory:")
        await storage.connect()
        with pytest.raises(StorageError, match="already connected"):
            await storage.connect()
        await storage.close()

    async def test_close_is_idempotent(self) -> None:
        storage = Storage(":memory:")
        await storage.connect()
        await storage.close()
        await storage.close()

    async def test_operations_before_connect_raise(self) -> None:
        storage = Storage(":memory:")
        with pytest.raises(StorageError, match="not connected"):
            await storage.store_decision(_decision())

    async def test_connect_creates_parent_directory(self, tmp_path) -> None:
        db_path = tmp_path / "nested" / "dir" / "db.sqlite"
        async with Storage(str(db_path)) as storage:
            await storage.store_decision(_decision())
        assert db_path.exists()


class TestPragmas:
    async def test_wal_mode_enabled_on_file_database(self, tmp_path) -> None:
        async with (
            Storage(str(tmp_path / "wal.sqlite")) as storage,
            storage.connection.execute("PRAGMA journal_mode") as cursor,
        ):
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "wal"

    async def test_foreign_keys_enforced(self) -> None:
        async with (
            Storage(":memory:") as storage,
            storage.connection.execute("PRAGMA foreign_keys") as cursor,
        ):
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1


class TestMigration:
    async def test_fresh_database_is_stamped_with_schema_version(self, tmp_path) -> None:
        async with (
            Storage(str(tmp_path / "fresh.sqlite")) as storage,
            storage.connection.execute("PRAGMA user_version") as cursor,
        ):
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == SCHEMA_VERSION

    async def test_stale_schema_is_dropped_and_recreated(self, tmp_path) -> None:
        """A database with an old/incompatible schema is rebuilt on connect."""
        db_path = tmp_path / "stale.sqlite"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("CREATE TABLE decisions (id TEXT PRIMARY KEY, old_column TEXT)")
            await conn.execute("INSERT INTO decisions (id, old_column) VALUES ('legacy', 'data')")
            await conn.commit()

        async with Storage(str(db_path)) as storage:
            async with storage.connection.execute("PRAGMA table_info(decisions)") as cursor:
                columns = {row[1] for row in await cursor.fetchall()}
            assert "old_column" not in columns
            assert "prompt_hash" in columns
            assert await storage.get_recent_decisions() == []

    async def test_matching_version_preserves_data(self, tmp_path) -> None:
        db_path = tmp_path / "keep.sqlite"
        async with Storage(str(db_path)) as storage:
            await storage.store_decision(_decision(id="kept"))

        async with Storage(str(db_path)) as storage:
            records = await storage.get_recent_decisions()
            assert [r.id for r in records] == ["kept"]


class TestFacadeRoundtrip:
    async def test_decision_crud_through_facade(self) -> None:
        async with Storage(":memory:") as storage:
            decision = _decision()
            stored_id = await storage.store_decision(decision)
            assert stored_id == decision.id
            by_conversation = await storage.get_decisions_by_conversation("missing")
            assert by_conversation == []
            assert await storage.get_decision_by_prompt_hash("z" * 64) is None

    async def test_signal_crud_through_facade(self) -> None:
        async with Storage(":memory:") as storage:
            decision_id = await storage.store_decision(_decision())
            await storage.store_signal(
                SignalRecord(decision_id=decision_id, signal_type="hard_regen", signal_value=-0.3)
            )
            signals = await storage.get_signals_for_decision(decision_id)
            assert len(signals) == 1
            assert signals[0].signal_type == "hard_regen"

    async def test_reputation_crud_through_facade(self) -> None:
        async with Storage(":memory:") as storage:
            assert await storage.get_reputation("code_low", "low") is None
            await storage.update_reputation("code_low", "low", 0.2, 5)
            record = await storage.get_reputation("code_low", "low")
            assert record is not None
            assert record.ema_score == 0.2
            await storage.record_adaptation("code_low", "low", "medium", 0.2)
            assert len(await storage.get_adaptations()) == 1
            await storage.reset_reputation()
            assert await storage.get_all_reputation() == []


class TestProjectStats:
    async def test_empty_database_returns_zeroed_stats(self) -> None:
        async with Storage(":memory:") as storage:
            stats = await storage.get_project_stats()
        assert stats["total_decisions"] == 0
        assert stats["total_cost_usd"] == 0
        assert stats["average_latency_ms"] == 0
        assert stats["model_distribution"] == {}
        assert stats["bucket_distribution"] == {}
        assert stats["adapted_buckets"] == []
        assert stats["recent_decisions"] == []

    async def test_aggregates_populated_database(self) -> None:
        async with Storage(":memory:") as storage:
            for i in range(3):
                await storage.store_decision(
                    _decision(
                        id=f"d{i}",
                        model_used="openai/gpt-4o-mini",
                        latency_ms=100 + i,
                        timestamp=utc_now() + timedelta(seconds=i),
                    )
                )
            await storage.store_decision(
                _decision(
                    id="g1",
                    model_used="groq/llama-3.1-8b",
                    provider_key="groq",
                    actual_cost_usd=None,
                    latency_ms=50,
                )
            )
            await storage.record_adaptation("code_low", "low", "medium", 0.25)
            stats = await storage.get_project_stats()
        assert stats["total_decisions"] == 4
        assert stats["model_distribution"] == {"openai/gpt-4o-mini": 3, "groq/llama-3.1-8b": 1}
        assert stats["bucket_distribution"] == {"medium": 4}
        assert stats["total_cost_usd"] == pytest.approx(3 * 0.0002 + 0.0001)
        assert stats["average_latency_ms"] == pytest.approx((303 + 50) / 4)
        assert len(stats["adapted_buckets"]) == 1
        assert len(stats["recent_decisions"]) == 4

    async def test_recent_decisions_capped_at_ten(self) -> None:
        async with Storage(":memory:") as storage:
            for i in range(15):
                await storage.store_decision(
                    _decision(id=f"d{i}", timestamp=utc_now() + timedelta(seconds=i))
                )
            stats = await storage.get_project_stats()
        assert len(stats["recent_decisions"]) == 10
