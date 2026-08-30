"""CRUD for the ``reputation`` and ``adaptations`` tables (spec 08).

All functions operate on a raw ``aiosqlite.Connection`` and are exposed
through the ``Storage`` facade in ``smartroute.storage.connection``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

import aiosqlite

from smartroute.exceptions import StorageError
from smartroute.storage.decisions import dt_to_str, str_to_dt, utc_now
from smartroute.types import AdaptationRecord, ReputationRecord

logger = logging.getLogger(__name__)

_REPUTATION_COLUMNS = (
    "id, bucket_key, model_tier, ema_score, call_count, last_updated, last_bumped_at"
)


def _reputation_from_row(row: aiosqlite.Row) -> ReputationRecord:
    """Build a ReputationRecord from a database row."""
    last_bumped = row["last_bumped_at"]
    return ReputationRecord(
        id=row["id"],
        bucket_key=row["bucket_key"],
        model_tier=row["model_tier"],
        ema_score=row["ema_score"],
        call_count=row["call_count"],
        last_updated=str_to_dt(row["last_updated"]),
        last_bumped_at=str_to_dt(last_bumped) if last_bumped is not None else None,
    )


async def get_reputation(
    conn: aiosqlite.Connection, bucket_key: str, model_tier: str
) -> ReputationRecord | None:
    """Return the reputation record for a bucket/tier pair.

    Args:
        conn: Open aiosqlite connection.
        bucket_key: ``"{task_type}_{complexity_bucket}"``, e.g. ``"code_low"``.
        model_tier: ``"low"``, ``"medium"``, or ``"high"``.

    Returns:
        The stored record, or None if this pair has never been updated.
        Callers treat None as "start neutral at EMA 0.5, call_count 0".

    Raises:
        StorageError: If the query fails.
    """
    try:
        async with conn.execute(
            f"SELECT {_REPUTATION_COLUMNS} FROM reputation WHERE bucket_key = ? AND model_tier = ?",
            (bucket_key, model_tier),
        ) as cursor:
            row = await cursor.fetchone()
    except aiosqlite.Error as exc:
        logger.warning("get_reputation failed: %s", exc)
        raise StorageError(f"failed to fetch reputation: {exc}") from exc
    return _reputation_from_row(row) if row is not None else None


async def update_reputation(
    conn: aiosqlite.Connection,
    bucket_key: str,
    model_tier: str,
    ema: float,
    call_count: int,
    last_bumped_at: datetime | None = None,
) -> None:
    """Insert or update the reputation record for a bucket/tier pair.

    Args:
        conn: Open aiosqlite connection.
        bucket_key: Bucket whose reputation is being updated.
        model_tier: Tier whose reputation is being updated.
        ema: New EMA score.
        call_count: New cumulative signal count.
        last_bumped_at: Set only when a bump happens. When None, an existing
            row keeps its current ``last_bumped_at`` so cooldowns survive
            ordinary reputation updates.

    Raises:
        StorageError: If the upsert fails.
    """
    try:
        await conn.execute(
            "INSERT INTO reputation (id, bucket_key, model_tier, ema_score, call_count, "
            "last_updated, last_bumped_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(bucket_key, model_tier) DO UPDATE SET "
            "ema_score = excluded.ema_score, "
            "call_count = excluded.call_count, "
            "last_updated = excluded.last_updated, "
            "last_bumped_at = COALESCE(excluded.last_bumped_at, reputation.last_bumped_at)",
            (
                str(uuid.uuid4()),
                bucket_key,
                model_tier,
                ema,
                call_count,
                dt_to_str(utc_now()),
                dt_to_str(last_bumped_at) if last_bumped_at is not None else None,
            ),
        )
        await conn.commit()
    except aiosqlite.Error as exc:
        logger.warning("update_reputation failed: %s", exc)
        raise StorageError(f"failed to update reputation: {exc}") from exc


async def get_all_reputation(conn: aiosqlite.Connection) -> list[ReputationRecord]:
    """Return every reputation record, ordered by bucket key.

    Args:
        conn: Open aiosqlite connection.

    Returns:
        All records; empty list if the table is empty.

    Raises:
        StorageError: If the query fails.
    """
    try:
        async with conn.execute(
            f"SELECT {_REPUTATION_COLUMNS} FROM reputation ORDER BY bucket_key, model_tier"
        ) as cursor:
            rows = await cursor.fetchall()
    except aiosqlite.Error as exc:
        logger.warning("get_all_reputation failed: %s", exc)
        raise StorageError(f"failed to fetch reputation table: {exc}") from exc
    return [_reputation_from_row(row) for row in rows]


async def record_adaptation(
    conn: aiosqlite.Connection,
    bucket_key: str,
    old_tier: str,
    new_tier: str,
    ema_at_bump: float,
) -> None:
    """Record an auto-bump event for a bucket.

    Args:
        conn: Open aiosqlite connection.
        bucket_key: The bucket that was bumped.
        old_tier: Tier before the bump.
        new_tier: Tier after the bump.
        ema_at_bump: EMA score that triggered the bump.

    Raises:
        StorageError: If the insert fails.
    """
    record = AdaptationRecord(
        bucket_key=bucket_key,
        old_tier=old_tier,
        new_tier=new_tier,
        ema_at_bump=ema_at_bump,
    )
    try:
        await conn.execute(
            "INSERT INTO adaptations (id, bucket_key, old_tier, new_tier, "
            "triggered_at, ema_at_bump) VALUES (?,?,?,?,?,?)",
            (
                record.id,
                record.bucket_key,
                record.old_tier,
                record.new_tier,
                dt_to_str(record.triggered_at),
                record.ema_at_bump,
            ),
        )
        await conn.commit()
    except aiosqlite.Error as exc:
        logger.warning("record_adaptation failed: %s", exc)
        raise StorageError(f"failed to record adaptation: {exc}") from exc


async def get_adaptations(
    conn: aiosqlite.Connection, bucket_key: str | None = None
) -> list[AdaptationRecord]:
    """Return adaptation events, newest first, optionally for one bucket.

    Args:
        conn: Open aiosqlite connection.
        bucket_key: Restrict results to this bucket; None returns all.

    Returns:
        Adaptation records ordered by triggered_at, descending.

    Raises:
        StorageError: If the query fails.
    """
    query = "SELECT id, bucket_key, old_tier, new_tier, triggered_at, ema_at_bump FROM adaptations"
    params: tuple[str, ...] = ()
    if bucket_key is not None:
        query += " WHERE bucket_key = ?"
        params = (bucket_key,)
    query += " ORDER BY triggered_at DESC"
    try:
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
    except aiosqlite.Error as exc:
        logger.warning("get_adaptations failed: %s", exc)
        raise StorageError(f"failed to fetch adaptations: {exc}") from exc
    return [
        AdaptationRecord(
            id=row["id"],
            bucket_key=row["bucket_key"],
            old_tier=row["old_tier"],
            new_tier=row["new_tier"],
            ema_at_bump=row["ema_at_bump"],
            triggered_at=str_to_dt(row["triggered_at"]),
        )
        for row in rows
    ]


async def reset_reputation(conn: aiosqlite.Connection, bucket_key: str | None = None) -> None:
    """Delete reputation rows, either for one bucket or for all buckets.

    Args:
        conn: Open aiosqlite connection.
        bucket_key: Delete only this bucket's rows; None clears every bucket.

    Raises:
        StorageError: If the delete fails.
    """
    try:
        if bucket_key is None:
            await conn.execute("DELETE FROM reputation")
        else:
            await conn.execute("DELETE FROM reputation WHERE bucket_key = ?", (bucket_key,))
        await conn.commit()
    except aiosqlite.Error as exc:
        logger.warning("reset_reputation failed: %s", exc)
        raise StorageError(f"failed to reset reputation: {exc}") from exc
