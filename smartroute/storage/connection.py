"""SQLite connection management and the Storage facade (spec 08).

``Storage`` is the single persistence entry point for SmartRoute: it owns the
aiosqlite connection, applies pragmas (WAL, foreign keys), performs the v1
drop-and-recreate schema migration, and exposes the CRUD operations from
``smartroute.storage.decisions`` and ``smartroute.storage.reputation``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from smartroute.exceptions import StorageError
from smartroute.storage import decisions as decisions_crud
from smartroute.storage import reputation as reputation_crud
from smartroute.storage.schema import SCHEMA_STATEMENTS, SCHEMA_VERSION
from smartroute.types import (
    AdaptationRecord,
    DecisionRecord,
    ReputationRecord,
    SignalRecord,
)

logger = logging.getLogger(__name__)

_TABLE_NAMES = ("decisions", "signals", "reputation", "adaptations", "config_overrides")


class Storage:
    """Async SQLite storage for decisions, signals, and reputation.

    Single connection per instance (SQLite limitation); all operations are
    async. Use ``await storage.connect()`` once, then call the CRUD methods,
    or use as an async context manager.

    Args:
        db_path: Path to the SQLite file, or ``":memory:"`` for an in-memory
            database (used by tests). Parent directories are created on
            ``connect()`` for file paths.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the connection, apply pragmas, and run schema migration.

        Raises:
            StorageError: If already connected or SQLite cannot be opened.
        """
        if self._connection is not None:
            raise StorageError("storage is already connected")
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = await aiosqlite.connect(self.db_path)
        except aiosqlite.Error as exc:
            raise StorageError(f"failed to open database {self.db_path}: {exc}") from exc
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        self._connection = conn
        await self._migrate()

    async def close(self) -> None:
        """Close the connection. Safe to call multiple times."""
        if self._connection is None:
            return
        try:
            await self._connection.close()
        except aiosqlite.Error as exc:
            logger.warning("error while closing storage: %s", exc)
        finally:
            self._connection = None

    async def __aenter__(self) -> Storage:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @property
    def connection(self) -> aiosqlite.Connection:
        """The open aiosqlite connection.

        Raises:
            StorageError: If ``connect()`` has not been called (or ``close()``
                already ran).
        """
        if self._connection is None:
            raise StorageError("storage is not connected; call connect() first")
        return self._connection

    async def _migrate(self) -> None:
        """Drop-and-recreate schema when ``user_version`` mismatches (spec 08)."""
        conn = self.connection
        async with conn.execute("PRAGMA user_version") as cursor:
            row = await cursor.fetchone()
        version = int(row[0]) if row is not None else 0
        if version == SCHEMA_VERSION:
            return
        logger.info(
            "migrating database schema from version %s to %s (drop and recreate)",
            version,
            SCHEMA_VERSION,
        )
        try:
            for table in _TABLE_NAMES:
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
            for statement in SCHEMA_STATEMENTS:
                await conn.execute(statement)
            await conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            await conn.commit()
        except aiosqlite.Error as exc:
            raise StorageError(f"schema migration failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    async def store_decision(self, decision: DecisionRecord) -> str:
        """Store a decision, returning its id (spec 08)."""
        return await decisions_crud.store_decision(self.connection, decision)

    async def get_recent_decisions(self, limit: int = 10) -> list[DecisionRecord]:
        """Return the most recent decisions, newest first (spec 08)."""
        return await decisions_crud.get_recent_decisions(self.connection, limit)

    async def get_decisions_by_conversation(self, conversation_id: str) -> list[DecisionRecord]:
        """Return all decisions in a conversation, chronological order (spec 08)."""
        return await decisions_crud.get_decisions_by_conversation(self.connection, conversation_id)

    async def get_decision_by_prompt_hash(
        self, prompt_hash: str, window_seconds: int = 60
    ) -> DecisionRecord | None:
        """Find the newest recent decision with the same prompt hash (spec 08)."""
        return await decisions_crud.get_decision_by_prompt_hash(
            self.connection, prompt_hash, window_seconds
        )

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    async def store_signal(self, signal: SignalRecord) -> None:
        """Store an implicit feedback signal (spec 08)."""
        await decisions_crud.store_signal(self.connection, signal)

    async def get_signals_for_decision(self, decision_id: str) -> list[SignalRecord]:
        """Return all signals recorded against one decision (spec 08)."""
        return await decisions_crud.get_signals_for_decision(self.connection, decision_id)

    # ------------------------------------------------------------------
    # Reputation
    # ------------------------------------------------------------------

    async def get_reputation(self, bucket_key: str, model_tier: str) -> ReputationRecord | None:
        """Return the reputation record for a bucket/tier pair, or None (spec 08)."""
        return await reputation_crud.get_reputation(self.connection, bucket_key, model_tier)

    async def update_reputation(
        self,
        bucket_key: str,
        model_tier: str,
        ema: float,
        call_count: int,
        last_bumped_at: datetime | None = None,
    ) -> None:
        """Insert or update a reputation record (spec 08)."""
        await reputation_crud.update_reputation(
            self.connection, bucket_key, model_tier, ema, call_count, last_bumped_at
        )

    async def get_all_reputation(self) -> list[ReputationRecord]:
        """Return every reputation record (spec 08)."""
        return await reputation_crud.get_all_reputation(self.connection)

    async def reset_reputation(self, bucket_key: str | None = None) -> None:
        """Delete reputation rows for one bucket or all buckets (spec 08)."""
        await reputation_crud.reset_reputation(self.connection, bucket_key)

    # ------------------------------------------------------------------
    # Adaptations
    # ------------------------------------------------------------------

    async def record_adaptation(
        self,
        bucket_key: str,
        old_tier: str,
        new_tier: str,
        ema_at_bump: float,
    ) -> None:
        """Record an auto-bump event (spec 08)."""
        await reputation_crud.record_adaptation(
            self.connection, bucket_key, old_tier, new_tier, ema_at_bump
        )

    async def get_adaptations(self, bucket_key: str | None = None) -> list[AdaptationRecord]:
        """Return adaptation events, newest first (spec 08)."""
        return await reputation_crud.get_adaptations(self.connection, bucket_key)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    async def get_project_stats(self) -> dict[str, Any]:
        """Aggregate stats for ProjectReport (spec 08).

        Returns:
            A dict with ``total_decisions``, ``total_cost_usd`` (actual cost
            falling back to estimate), ``average_latency_ms``,
            ``model_distribution``, ``bucket_distribution``,
            ``adapted_buckets``, and ``recent_decisions``.

        Raises:
            StorageError: If any aggregate query fails.
        """
        conn = self.connection
        try:
            async with conn.execute(
                "SELECT COUNT(*) AS total, "
                "COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd, 0)), 0) AS cost, "
                "COALESCE(AVG(latency_ms), 0) AS avg_latency FROM decisions"
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            async with conn.execute(
                "SELECT model_used, COUNT(*) AS count FROM decisions "
                "GROUP BY model_used ORDER BY count DESC"
            ) as cursor:
                model_distribution = {r["model_used"]: r["count"] for r in await cursor.fetchall()}
            async with conn.execute(
                "SELECT complexity_bucket, COUNT(*) AS count FROM decisions "
                "GROUP BY complexity_bucket ORDER BY count DESC"
            ) as cursor:
                bucket_distribution = {
                    r["complexity_bucket"]: r["count"] for r in await cursor.fetchall()
                }
        except aiosqlite.Error as exc:
            logger.warning("get_project_stats failed: %s", exc)
            raise StorageError(f"failed to aggregate project stats: {exc}") from exc
        adaptations = await self.get_adaptations()
        recent = await self.get_recent_decisions(limit=10)
        return {
            "total_decisions": row["total"],
            "total_cost_usd": row["cost"],
            "average_latency_ms": row["avg_latency"],
            "model_distribution": model_distribution,
            "bucket_distribution": bucket_distribution,
            "adapted_buckets": [
                {
                    "key": a.bucket_key,
                    "old_tier": a.old_tier,
                    "new_tier": a.new_tier,
                    "ema_at_bump": a.ema_at_bump,
                    "timestamp": a.triggered_at.isoformat(),
                }
                for a in adaptations
            ],
            "recent_decisions": [
                {
                    "id": d.id,
                    "timestamp": d.timestamp.isoformat(),
                    "task_type": d.task_type,
                    "complexity": d.complexity,
                    "complexity_bucket": d.complexity_bucket,
                    "model_used": d.model_used,
                    "provider_key": d.provider_key,
                    "reason": d.reason,
                    "was_adapted": d.was_adapted,
                    "override_applied": d.override_applied,
                }
                for d in recent
            ],
        }
