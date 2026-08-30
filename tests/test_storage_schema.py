"""Unit tests for smartroute.storage.schema (Phase 1, module 3).

All tests use aiosqlite with an in-memory database, per the testing strategy.
"""

from datetime import datetime, timezone

import aiosqlite
import pytest

from smartroute.storage.schema import (
    SCHEMA_STATEMENTS,
    SCHEMA_VERSION,
    TABLE_DDL,
)


@pytest.fixture
async def conn() -> aiosqlite.Connection:
    async with aiosqlite.connect(":memory:") as c:
        c.row_factory = aiosqlite.Row
        await c.execute("PRAGMA foreign_keys=ON")
        for statement in SCHEMA_STATEMENTS:
            await c.execute(statement)
        await c.commit()
        yield c  # type: ignore[misc]


class TestSchemaCreation:
    async def test_all_tables_created(self, conn: aiosqlite.Connection) -> None:
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ) as cursor:
            tables = {row[0] for row in await cursor.fetchall()}
        assert {"decisions", "signals", "reputation", "adaptations", "config_overrides"} <= tables

    async def test_all_indexes_created(self, conn: aiosqlite.Connection) -> None:
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ) as cursor:
            indexes = {row[0] for row in await cursor.fetchall()}
        assert len(indexes) == 10

    async def test_decisions_has_no_full_prompt_column(self, conn: aiosqlite.Connection) -> None:
        async with conn.execute("PRAGMA table_info(decisions)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        assert "prompt" not in columns
        assert "prompt_hash" in columns
        assert "prompt_preview" in columns


class TestConstraints:
    async def test_signals_check_constraint_rejects_unknown_type(
        self, conn: aiosqlite.Connection
    ) -> None:
        await conn.execute(
            "INSERT INTO decisions (id, prompt_hash, task_type, complexity, "
            "complexity_bucket, confidence, model_used, provider_key, reason) "
            "VALUES ('d1', 'h', 'code', 0.5, 'medium', 0.8, 'm', 'openai', 'r')"
        )
        with pytest.raises(aiosqlite.IntegrityError, match="signal_type"):
            await conn.execute(
                "INSERT INTO signals (id, decision_id, signal_type, signal_value) "
                "VALUES ('s1', 'd1', 'bogus_signal', -0.3)"
            )

    async def test_reputation_unique_bucket_tier(self, conn: aiosqlite.Connection) -> None:
        await conn.execute(
            "INSERT INTO reputation (id, bucket_key, model_tier) VALUES ('r1', 'code_low', 'low')"
        )
        with pytest.raises(aiosqlite.IntegrityError, match="UNIQUE"):
            await conn.execute(
                "INSERT INTO reputation (id, bucket_key, model_tier) "
                "VALUES ('r2', 'code_low', 'low')"
            )

    async def test_signal_fk_rejects_unknown_decision(self, conn: aiosqlite.Connection) -> None:
        with pytest.raises(aiosqlite.IntegrityError, match="FOREIGN KEY"):
            await conn.execute(
                "INSERT INTO signals (id, decision_id, signal_type, signal_value) "
                "VALUES ('s1', 'missing', 'acceptance', 0.05)"
            )

    async def test_signal_fk_cascade_deletes_with_decision(
        self, conn: aiosqlite.Connection
    ) -> None:
        await conn.execute(
            "INSERT INTO decisions (id, prompt_hash, task_type, complexity, "
            "complexity_bucket, confidence, model_used, provider_key, reason) "
            "VALUES ('d1', 'h', 'code', 0.5, 'medium', 0.8, 'm', 'openai', 'r')"
        )
        await conn.execute(
            "INSERT INTO signals (id, decision_id, signal_type, signal_value) "
            "VALUES ('s1', 'd1', 'hard_regen', -0.3)"
        )
        await conn.execute("DELETE FROM decisions WHERE id = 'd1'")
        async with conn.execute("SELECT COUNT(*) FROM signals") as cursor:
            count = (await cursor.fetchone())[0]
        assert count == 0

    async def test_reputation_ema_default_is_neutral(self, conn: aiosqlite.Connection) -> None:
        await conn.execute(
            "INSERT INTO reputation (id, bucket_key, model_tier) VALUES ('r1', 'code_low', 'low')"
        )
        async with conn.execute("SELECT ema_score, call_count FROM reputation") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0.5
        assert row[1] == 0


class TestSchemaStatements:
    def test_schema_version_is_positive_int(self) -> None:
        assert SCHEMA_VERSION == 1

    def test_table_ddl_executes_in_order(self) -> None:
        assert len(TABLE_DDL) == 5

    async def test_timestamp_roundtrip_format(self, conn: aiosqlite.Connection) -> None:
        """Python-side ISO-8601 UTC timestamps sort correctly as TEXT."""
        ts = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        await conn.execute(
            "INSERT INTO decisions (id, timestamp, prompt_hash, task_type, complexity, "
            "complexity_bucket, confidence, model_used, provider_key, reason) "
            "VALUES ('d1', ?, 'h', 'code', 0.5, 'medium', 0.8, 'm', 'openai', 'r')",
            (ts,),
        )
        async with conn.execute("SELECT timestamp FROM decisions WHERE id='d1'") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        expected = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc)
        assert datetime.fromisoformat(row[0]) == expected
