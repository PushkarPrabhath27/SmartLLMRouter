"""Unit tests for smartroute.routing.reputation (Phase 3 Module 1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from smartroute.routing.reputation import (
    get_bucket_key,
    get_next_tier,
    should_bump,
    update_ema,
)
from smartroute.types import ComplexityBucket, TaskType


class TestUpdateEMA:
    def test_default_alpha_calculation(self) -> None:
        """alpha=0.3: 0.3 * (-0.3) + 0.7 * 0.5 = -0.09 + 0.35 = 0.26"""
        result = update_ema(0.5, -0.3)
        assert pytest.approx(result, abs=1e-5) == 0.26

    def test_acceptance_signal_increases_ema(self) -> None:
        """alpha=0.3: 0.3 * 0.05 + 0.7 * 0.5 = 0.015 + 0.35 = 0.365"""
        result = update_ema(0.5, 0.05)
        assert pytest.approx(result, abs=1e-5) == 0.365

    def test_custom_alpha(self) -> None:
        """alpha=0.5: 0.5 * 1.0 + 0.5 * 0.2 = 0.6"""
        result = update_ema(0.2, 1.0, alpha=0.5)
        assert pytest.approx(result, abs=1e-5) == 0.6

    def test_clamping_lower_bound(self) -> None:
        """Negative values clamp to 0.0."""
        result = update_ema(0.0, -1.0, alpha=0.5)
        assert result == 0.0

    def test_clamping_upper_bound(self) -> None:
        """Values exceeding 1.0 clamp to 1.0."""
        result = update_ema(1.0, 2.0, alpha=0.5)
        assert result == 1.0


class TestShouldBump:
    def test_returns_false_if_call_count_below_min_calls(self) -> None:
        assert should_bump(ema=0.1, call_count=9, min_calls=10) is False

    def test_returns_false_if_ema_above_threshold(self) -> None:
        assert should_bump(ema=0.4, call_count=15, threshold=0.3) is False

    def test_returns_false_if_ema_exactly_at_threshold(self) -> None:
        """Boundary condition: ema == threshold must return False."""
        assert should_bump(ema=0.3, call_count=15, threshold=0.3) is False

    def test_returns_true_when_all_conditions_met_without_previous_bump(self) -> None:
        assert should_bump(ema=0.29, call_count=10, threshold=0.3) is True

    def test_returns_false_when_cooldown_is_active(self) -> None:
        now = datetime.now(timezone.utc)
        recent_bump = now - timedelta(minutes=3)
        assert (
            should_bump(
                ema=0.2,
                call_count=15,
                threshold=0.3,
                last_bumped_at=recent_bump,
                cooldown_minutes=5,
                now=now,
            )
            is False
        )

    def test_returns_true_when_cooldown_has_expired(self) -> None:
        now = datetime.now(timezone.utc)
        old_bump = now - timedelta(minutes=6)
        assert (
            should_bump(
                ema=0.2,
                call_count=15,
                threshold=0.3,
                last_bumped_at=old_bump,
                cooldown_minutes=5,
                now=now,
            )
            is True
        )

    def test_handles_naive_and_aware_timestamps(self) -> None:
        now_aware = datetime.now(timezone.utc)
        naive_bump = (now_aware - timedelta(minutes=10)).replace(tzinfo=None)
        assert (
            should_bump(
                ema=0.2,
                call_count=15,
                last_bumped_at=naive_bump,
                now=now_aware,
            )
            is True
        )

    def test_handles_omitted_now_and_naive_now(self) -> None:
        naive_now = datetime.now()
        naive_bump = naive_now - timedelta(minutes=10)
        assert (
            should_bump(
                ema=0.2,
                call_count=15,
                last_bumped_at=naive_bump,
                now=naive_now,
            )
            is True
        )
        # Test omitted now with explicit past last_bumped_at
        past_bump = datetime.now(timezone.utc) - timedelta(minutes=10)
        assert (
            should_bump(
                ema=0.2,
                call_count=15,
                last_bumped_at=past_bump,
            )
            is True
        )


class TestGetNextTier:
    def test_escalates_low_to_medium(self) -> None:
        assert get_next_tier("low") == "medium"

    def test_escalates_medium_to_high(self) -> None:
        assert get_next_tier("medium") == "high"

    def test_high_remains_high(self) -> None:
        assert get_next_tier("high") == "high"

    def test_unknown_tier_fallback(self) -> None:
        assert get_next_tier("custom") == "custom"


class TestGetBucketKey:
    def test_with_enums(self) -> None:
        key = get_bucket_key(TaskType.CODE, ComplexityBucket.LOW)
        assert key == "code_low"

    def test_with_strings(self) -> None:
        key = get_bucket_key("reasoning", "medium")
        assert key == "reasoning_medium"

    def test_with_mixed_types(self) -> None:
        key = get_bucket_key(TaskType.TRANSLATION, "high")
        assert key == "translation_high"
