"""Reputation score calculations, auto-bump decisions, and tier helper functions (spec 06).

Stateless helper functions for reputation EMA calculations, threshold checking,
tier escalation, and bucket key construction.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from smartroute.types import ComplexityBucket, TaskType

_DEFAULT_ALPHA = 0.3
_DEFAULT_BUMP_THRESHOLD = 0.3
_DEFAULT_MIN_CALLS = 10
_DEFAULT_COOLDOWN_MINUTES = 5

_TIER_ESCALATION: dict[str, str] = {
    "low": "medium",
    "medium": "high",
    "high": "high",
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float value to [low, high]."""
    return max(low, min(high, value))


def update_ema(old_ema: float, signal_value: float, alpha: float = _DEFAULT_ALPHA) -> float:
    """Calculate the updated Exponential Moving Average (EMA) for reputation (spec 06).

    Formula: ``new_ema = alpha * signal_value + (1 - alpha) * old_ema``
    The result is clamped to [0.0, 1.0].

    Args:
        old_ema: Previous EMA score in [0.0, 1.0].
        signal_value: Feedback signal value (e.g., -0.3, -0.2, -0.1, +0.05).
        alpha: Smoothing factor in (0.0, 1.0], defaults to 0.3.

    Returns:
        Updated EMA score clamped to [0.0, 1.0].
    """
    raw_ema = alpha * signal_value + (1.0 - alpha) * old_ema
    return _clamp(raw_ema)


def should_bump(
    ema: float,
    call_count: int,
    threshold: float = _DEFAULT_BUMP_THRESHOLD,
    last_bumped_at: datetime | None = None,
    cooldown_minutes: int = _DEFAULT_COOLDOWN_MINUTES,
    min_calls: int = _DEFAULT_MIN_CALLS,
    now: datetime | None = None,
) -> bool:
    """Determine whether a complexity bucket should auto-bump to the next tier (spec 06).

    Bump conditions:
    1. ``call_count >= min_calls`` (enough data).
    2. ``ema < threshold`` (reputation score is below threshold).
    3. Not in cooldown: ``(now - last_bumped_at) >= cooldown_minutes``.

    Args:
        ema: Current EMA reputation score.
        call_count: Number of recorded signals for this bucket/tier pair.
        threshold: Bump threshold; defaults to 0.3.
        last_bumped_at: Timestamp of the last bump event, or None if never bumped.
        cooldown_minutes: Minimum minutes between bumps; defaults to 5.
        min_calls: Minimum required calls before bump; defaults to 10.
        now: Optional current timestamp override for testing; defaults to UTC now.

    Returns:
        True if all bump conditions are met; False otherwise.
    """
    if call_count < min_calls:
        return False
    if ema >= threshold:
        return False
    if last_bumped_at is not None:
        if now is None:
            now = datetime.now(timezone.utc)
        if last_bumped_at.tzinfo is None:
            last_bumped_at = last_bumped_at.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if (now - last_bumped_at) < timedelta(minutes=cooldown_minutes):
            return False
    return True


def get_next_tier(current_tier: str) -> str:
    """Escalate a model tier to the next complexity level (spec 06).

    Mapping:
    - ``"low"`` -> ``"medium"``
    - ``"medium"`` -> ``"high"``
    - ``"high"`` -> ``"high"`` (capped at high)

    Args:
        current_tier: Current complexity tier name (``"low"``, ``"medium"``, or ``"high"``).

    Returns:
        The next tier name string.
    """
    return _TIER_ESCALATION.get(current_tier, current_tier)


def get_bucket_key(
    task_type: TaskType | str,
    complexity_bucket: ComplexityBucket | str,
) -> str:
    """Build canonical bucket key string for reputation lookup (spec 06).

    Example: ``"code_low"``, ``"reasoning_medium"``.

    Args:
        task_type: TaskType enum or string value (e.g., TaskType.CODE or "code").
        complexity_bucket: ComplexityBucket enum or string value
            (e.g., ComplexityBucket.LOW or "low").

    Returns:
        Formatted string ``"{task_type}_{complexity_bucket}"``.
    """
    tt_str = task_type.value if isinstance(task_type, TaskType) else str(task_type)
    cb_str = (
        complexity_bucket.value
        if isinstance(complexity_bucket, ComplexityBucket)
        else str(complexity_bucket)
    )
    return f"{tt_str}_{cb_str}"
