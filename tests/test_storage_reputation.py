"""Unit tests for smartroute.storage.reputation (Phase 1, module 5)."""

from datetime import timedelta

import aiosqlite

from smartroute.storage.decisions import utc_now
from smartroute.storage.reputation import (
    get_adaptations,
    get_all_reputation,
    get_reputation,
    record_adaptation,
    reset_reputation,
    update_reputation,
)


class TestGetReputation:
    async def test_unknown_bucket_returns_none(self, conn: aiosqlite.Connection) -> None:
        assert await get_reputation(conn, "code_low", "low") is None

    async def test_roundtrips_stored_record(self, conn: aiosqlite.Connection) -> None:
        bumped = utc_now() - timedelta(minutes=10)
        await update_reputation(conn, "code_low", "low", 0.25, 12, last_bumped_at=bumped)
        record = await get_reputation(conn, "code_low", "low")
        assert record is not None
        assert record.bucket_key == "code_low"
        assert record.model_tier == "low"
        assert record.ema_score == 0.25
        assert record.call_count == 12
        assert record.last_bumped_at == bumped


class TestUpdateReputation:
    async def test_insert_then_update(self, conn: aiosqlite.Connection) -> None:
        await update_reputation(conn, "code_low", "low", 0.5, 1)
        await update_reputation(conn, "code_low", "low", 0.35, 2)
        record = await get_reputation(conn, "code_low", "low")
        assert record is not None
        assert record.ema_score == 0.35
        assert record.call_count == 2

    async def test_update_preserves_last_bumped_at(self, conn: aiosqlite.Connection) -> None:
        bumped = utc_now() - timedelta(minutes=1)
        await update_reputation(conn, "code_low", "low", 0.4, 10, last_bumped_at=bumped)
        await update_reputation(conn, "code_low", "low", 0.38, 11)
        record = await get_reputation(conn, "code_low", "low")
        assert record is not None
        assert record.last_bumped_at == bumped

    async def test_update_sets_last_bumped_at_when_given(self, conn: aiosqlite.Connection) -> None:
        await update_reputation(conn, "code_low", "low", 0.4, 10)
        bumped = utc_now()
        await update_reputation(conn, "code_low", "low", 0.2, 11, last_bumped_at=bumped)
        record = await get_reputation(conn, "code_low", "low")
        assert record is not None
        assert record.last_bumped_at == bumped

    async def test_same_bucket_different_tiers_are_independent(
        self, conn: aiosqlite.Connection
    ) -> None:
        await update_reputation(conn, "code_low", "low", 0.2, 5)
        await update_reputation(conn, "code_low", "medium", 0.9, 3)
        low = await get_reputation(conn, "code_low", "low")
        medium = await get_reputation(conn, "code_low", "medium")
        assert low is not None and low.ema_score == 0.2
        assert medium is not None and medium.ema_score == 0.9


class TestGetAllReputation:
    async def test_empty_table_returns_empty_list(self, conn: aiosqlite.Connection) -> None:
        assert await get_all_reputation(conn) == []

    async def test_returns_all_rows_sorted(self, conn: aiosqlite.Connection) -> None:
        await update_reputation(conn, "code_medium", "medium", 0.7, 2)
        await update_reputation(conn, "code_low", "low", 0.4, 1)
        records = await get_all_reputation(conn)
        assert [(r.bucket_key, r.model_tier) for r in records] == [
            ("code_low", "low"),
            ("code_medium", "medium"),
        ]


class TestAdaptations:
    async def test_record_and_fetch(self, conn: aiosqlite.Connection) -> None:
        await record_adaptation(conn, "code_low", "low", "medium", 0.25)
        records = await get_adaptations(conn)
        assert len(records) == 1
        record = records[0]
        assert record.bucket_key == "code_low"
        assert record.old_tier == "low"
        assert record.new_tier == "medium"
        assert record.ema_at_bump == 0.25

    async def test_fetch_newest_first(self, conn: aiosqlite.Connection) -> None:
        await record_adaptation(conn, "code_low", "low", "medium", 0.25)
        await record_adaptation(conn, "creative_low", "low", "medium", 0.2)
        records = await get_adaptations(conn)
        assert [r.bucket_key for r in records] == ["creative_low", "code_low"]

    async def test_bucket_filter(self, conn: aiosqlite.Connection) -> None:
        await record_adaptation(conn, "code_low", "low", "medium", 0.25)
        await record_adaptation(conn, "creative_low", "low", "medium", 0.2)
        only_code = await get_adaptations(conn, bucket_key="code_low")
        assert [r.bucket_key for r in only_code] == ["code_low"]
        assert await get_adaptations(conn, bucket_key="general_high") == []


class TestResetReputation:
    async def test_reset_single_bucket(self, conn: aiosqlite.Connection) -> None:
        await update_reputation(conn, "code_low", "low", 0.2, 5)
        await update_reputation(conn, "creative_low", "low", 0.8, 4)
        await reset_reputation(conn, bucket_key="code_low")
        assert await get_reputation(conn, "code_low", "low") is None
        assert await get_reputation(conn, "creative_low", "low") is not None

    async def test_reset_all_buckets(self, conn: aiosqlite.Connection) -> None:
        await update_reputation(conn, "code_low", "low", 0.2, 5)
        await update_reputation(conn, "creative_low", "low", 0.8, 4)
        await reset_reputation(conn)
        assert await get_all_reputation(conn) == []
