"""CRUD for the ``decisions`` and ``signals`` tables (spec 08).

All functions operate on a raw ``aiosqlite.Connection`` and are exposed
through the ``Storage`` facade in ``smartroute.storage.connection``.
Timestamps are stored as ISO-8601 UTC TEXT so they sort lexicographically.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import aiosqlite

from smartroute.exceptions import StorageError
from smartroute.types import DecisionRecord, SignalRecord

logger = logging.getLogger(__name__)

_DECISION_COLUMNS = (
    "id, timestamp, prompt_hash, prompt_preview, task_type, complexity, "
    "complexity_bucket, confidence, model_used, provider_key, "
    "estimated_cost_usd, actual_cost_usd, latency_ms, was_adapted, "
    "override_applied, reason, conversation_id, turn_number, features_json"
)


def dt_to_str(value: datetime) -> str:
    """Serialize a datetime as ISO-8601 text for SQLite storage."""
    return value.isoformat()


def str_to_dt(value: str) -> datetime:
    """Parse an ISO-8601 timestamp string back into a datetime."""
    return datetime.fromisoformat(value)


def utc_now() -> datetime:
    """Current UTC time, used for storage window queries."""
    return datetime.now(timezone.utc)


def _decision_from_row(row: aiosqlite.Row) -> DecisionRecord:
    """Build a DecisionRecord from a database row."""
    return DecisionRecord(
        id=row["id"],
        timestamp=str_to_dt(row["timestamp"]),
        prompt_hash=row["prompt_hash"],
        prompt_preview=row["prompt_preview"],
        task_type=row["task_type"],
        complexity=row["complexity"],
        complexity_bucket=row["complexity_bucket"],
        confidence=row["confidence"],
        model_used=row["model_used"],
        provider_key=row["provider_key"],
        estimated_cost_usd=row["estimated_cost_usd"],
        actual_cost_usd=row["actual_cost_usd"],
        latency_ms=row["latency_ms"],
        was_adapted=bool(row["was_adapted"]),
        override_applied=row["override_applied"],
        reason=row["reason"],
        conversation_id=row["conversation_id"],
        turn_number=row["turn_number"],
        features_json=row["features_json"],
    )


def _signal_from_row(row: aiosqlite.Row) -> SignalRecord:
    """Build a SignalRecord from a database row."""
    return SignalRecord(
        id=row["id"],
        decision_id=row["decision_id"],
        signal_type=row["signal_type"],
        signal_value=row["signal_value"],
        detected_at=str_to_dt(row["detected_at"]),
        detection_method=row["detection_method"],
    )


async def store_decision(conn: aiosqlite.Connection, decision: DecisionRecord) -> str:
    """Store a decision, returning its id.

    Args:
        conn: Open aiosqlite connection.
        decision: The decision to persist. Its ``prompt_hash`` and (optional)
            ``prompt_preview`` are stored; the full prompt is never accepted.

    Returns:
        The stored decision's id.

    Raises:
        StorageError: If the insert fails (e.g. duplicate id, constraint).
    """
    try:
        await conn.execute(
            f"INSERT INTO decisions ({_DECISION_COLUMNS}) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                decision.id,
                dt_to_str(decision.timestamp),
                decision.prompt_hash,
                decision.prompt_preview,
                decision.task_type,
                decision.complexity,
                decision.complexity_bucket,
                decision.confidence,
                decision.model_used,
                decision.provider_key,
                decision.estimated_cost_usd,
                decision.actual_cost_usd,
                decision.latency_ms,
                int(decision.was_adapted),
                decision.override_applied,
                decision.reason,
                decision.conversation_id,
                decision.turn_number,
                decision.features_json,
            ),
        )
        await conn.commit()
    except aiosqlite.Error as exc:
        logger.warning("store_decision failed: %s", exc)
        raise StorageError(f"failed to store decision {decision.id}: {exc}") from exc
    return decision.id


async def get_recent_decisions(conn: aiosqlite.Connection, limit: int = 10) -> list[DecisionRecord]:
    """Return the most recent decisions, newest first.

    Args:
        conn: Open aiosqlite connection.
        limit: Maximum number of records to return (default 10).

    Returns:
        Decision records ordered by timestamp, descending. Empty if none.

    Raises:
        StorageError: If the query fails.
    """
    try:
        async with conn.execute(
            f"SELECT {_DECISION_COLUMNS} FROM decisions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    except aiosqlite.Error as exc:
        logger.warning("get_recent_decisions failed: %s", exc)
        raise StorageError(f"failed to fetch recent decisions: {exc}") from exc
    return [_decision_from_row(row) for row in rows]


async def get_decisions_by_conversation(
    conn: aiosqlite.Connection, conversation_id: str
) -> list[DecisionRecord]:
    """Return all decisions in a conversation, chronological order.

    Args:
        conn: Open aiosqlite connection.
        conversation_id: The conversation to look up.

    Returns:
        Decision records ordered by timestamp, ascending. Empty if unknown id.

    Raises:
        StorageError: If the query fails.
    """
    try:
        async with conn.execute(
            f"SELECT {_DECISION_COLUMNS} FROM decisions "
            "WHERE conversation_id = ? ORDER BY timestamp ASC",
            (conversation_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    except aiosqlite.Error as exc:
        logger.warning("get_decisions_by_conversation failed: %s", exc)
        raise StorageError(f"failed to fetch conversation decisions: {exc}") from exc
    return [_decision_from_row(row) for row in rows]


async def get_decision_by_prompt_hash(
    conn: aiosqlite.Connection, prompt_hash: str, window_seconds: int = 60
) -> DecisionRecord | None:
    """Find the newest decision with the same prompt hash inside a time window.

    Used by signal detection to link a repeated prompt to the decision it
    implicitly grades.

    Args:
        conn: Open aiosqlite connection.
        prompt_hash: SHA-256 hex digest of the prompt.
        window_seconds: How far back to search (default 60).

    Returns:
        The newest matching decision, or None if there is no match.

    Raises:
        StorageError: If the query fails.
    """
    cutoff = dt_to_str(utc_now() - timedelta(seconds=window_seconds))
    try:
        async with conn.execute(
            f"SELECT {_DECISION_COLUMNS} FROM decisions "
            "WHERE prompt_hash = ? AND timestamp >= ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (prompt_hash, cutoff),
        ) as cursor:
            row = await cursor.fetchone()
    except aiosqlite.Error as exc:
        logger.warning("get_decision_by_prompt_hash failed: %s", exc)
        raise StorageError(f"failed to fetch decision by prompt hash: {exc}") from exc
    return _decision_from_row(row) if row is not None else None


async def store_signal(conn: aiosqlite.Connection, signal: SignalRecord) -> None:
    """Store an implicit feedback signal for a decision.

    Args:
        conn: Open aiosqlite connection.
        signal: The signal to persist; its ``decision_id`` must reference an
            existing decision (enforced by foreign key).

    Raises:
        StorageError: If the insert fails (e.g. unknown decision_id).
    """
    try:
        await conn.execute(
            "INSERT INTO signals (id, decision_id, signal_type, signal_value, "
            "detected_at, detection_method) VALUES (?,?,?,?,?,?)",
            (
                signal.id,
                signal.decision_id,
                signal.signal_type,
                signal.signal_value,
                dt_to_str(signal.detected_at),
                signal.detection_method,
            ),
        )
        await conn.commit()
    except aiosqlite.Error as exc:
        logger.warning("store_signal failed: %s", exc)
        raise StorageError(f"failed to store signal {signal.id}: {exc}") from exc


async def get_signals_for_decision(
    conn: aiosqlite.Connection, decision_id: str
) -> list[SignalRecord]:
    """Return all signals recorded against one decision, newest first.

    Args:
        conn: Open aiosqlite connection.
        decision_id: The decision to look up.

    Returns:
        Signal records ordered by detected_at, descending. Empty if none.

    Raises:
        StorageError: If the query fails.
    """
    try:
        async with conn.execute(
            "SELECT id, decision_id, signal_type, signal_value, detected_at, "
            "detection_method FROM signals WHERE decision_id = ? "
            "ORDER BY detected_at DESC",
            (decision_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    except aiosqlite.Error as exc:
        logger.warning("get_signals_for_decision failed: %s", exc)
        raise StorageError(f"failed to fetch signals for decision: {exc}") from exc
    return [_signal_from_row(row) for row in rows]
