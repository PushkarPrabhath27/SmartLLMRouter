"""Signal collector: orchestration, detection hooks, and fire-and-forget persistence (spec 07).

Coordinates observation of user prompts and conversation turns, detects implicit
feedback (hard/soft regeneration, explicit corrections, acceptance), and updates
reputation asynchronously in fire-and-forget tasks without blocking user responses.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from datetime import datetime, timezone
from typing import Any

from smartroute.config.schema import Config
from smartroute.signals.detectors import (
    ACCEPTANCE_VALUE,
    EXPLICIT_CORRECTION_VALUE,
    HARD_REGEN_VALUE,
    SIGNAL_VALUES,
    SOFT_REGEN_VALUE,
    HistoryItem,
    detect_acceptance,
    detect_explicit_correction,
    detect_hard_regen,
    detect_soft_regen,
)
from smartroute.signals.reputation_updater import apply_signal
from smartroute.storage.connection import Storage
from smartroute.types import ConversationContext, Signal, SignalRecord

logger = logging.getLogger(__name__)

_CANONICAL_SIGNALS = {"hard_regen", "soft_regen", "explicit_correction", "acceptance"}

_MANUAL_SIGNAL_MAPPING: dict[str, tuple[str, float]] = {
    "thumbs_up": ("acceptance", ACCEPTANCE_VALUE),
    "thumbs_down": ("explicit_correction", EXPLICIT_CORRECTION_VALUE),
    "hard_regen": ("hard_regen", HARD_REGEN_VALUE),
    "soft_regen": ("soft_regen", SOFT_REGEN_VALUE),
    "explicit_correction": ("explicit_correction", EXPLICIT_CORRECTION_VALUE),
    "acceptance": ("acceptance", ACCEPTANCE_VALUE),
}


class SignalCollector:
    """Orchestrates implicit feedback detection and async reputation updates."""

    def __init__(self, storage: Storage, config: Config) -> None:
        """Initialize the signal collector.

        Args:
            storage: Storage facade for decisions and reputation.
            config: SmartRoute configuration.
        """
        self.storage = storage
        self.config = config
        self._history: list[HistoryItem] = []
        self._decision_buckets: dict[str, tuple[str, str]] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()

    def register_prompt(
        self,
        decision_id: str,
        prompt: str,
        task_type: str,
        complexity_bucket: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a completed routing decision for future signal correlation.

        Args:
            decision_id: ID of the decision.
            prompt: User prompt.
            task_type: Classified task type (e.g. 'code').
            complexity_bucket: Bucket tier (e.g. 'low').
            timestamp: Optional UTC timestamp (defaults to now).
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        elif timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        self._history.append(
            HistoryItem(decision_id=decision_id, prompt=prompt, timestamp=timestamp)
        )
        if len(self._history) > 20:
            self._history = self._history[-20:]
        self._decision_buckets[decision_id] = (task_type, complexity_bucket)

    def _dispatch_background(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Dispatch a coroutine as a tracked fire-and-forget task."""

        async def _safe_run() -> None:
            try:
                await coro
            except Exception as exc:
                logger.warning("background signal processing failed: %s", exc)

        task = asyncio.create_task(_safe_run())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _process_signal(self, signal: Signal, detection_method: str = "auto") -> None:
        """Fetch decision metadata, persist signal record, and apply to reputation."""
        bucket_info = self._decision_buckets.get(signal.decision_id)
        decision = await self.storage.get_decision(signal.decision_id)
        if decision is not None and bucket_info is None:
            bucket_info = (decision.task_type, decision.complexity_bucket)

        if decision is not None:
            rec = SignalRecord(
                decision_id=signal.decision_id,
                signal_type=signal.signal_type,
                signal_value=signal.value,
                detection_method=detection_method,
            )
            await self.storage.store_signal(rec)
        else:
            logger.warning(
                "decision_id '%s' not found in storage; skipping signal persistence",
                signal.decision_id,
            )

        if bucket_info is not None:
            task_type, complexity_bucket = bucket_info
            bucket_key = f"{task_type}_{complexity_bucket}"
            await apply_signal(
                bucket_key=bucket_key,
                model_tier=complexity_bucket,
                signal=signal,
                storage=self.storage,
                config=self.config,
            )

    async def on_new_prompt(
        self,
        prompt: str,
        context: ConversationContext | None = None,
        now: datetime | None = None,
    ) -> Signal | None:
        """Check if incoming prompt is a hard or soft regeneration (spec 07).

        Args:
            prompt: Incoming user prompt.
            context: Optional conversation context.
            now: Optional current timestamp override for testing.

        Returns:
            The detected Signal, or None.
        """
        signal = detect_hard_regen(prompt, self._history, now=now)
        if signal is None:
            signal = detect_soft_regen(prompt, self._history, now=now)

        if signal is not None:
            self._dispatch_background(self._process_signal(signal, detection_method="auto"))
        return signal

    async def on_conversation_turn(
        self,
        conversation_id: str,
        turn_number: int,
        last_decision_id: str,
        next_message: str,
    ) -> Signal | None:
        """Check for explicit corrections or natural acceptance in a turn.

        Args:
            conversation_id: Conversation identifier.
            turn_number: Turn index.
            last_decision_id: Decision ID of the turn being responded to.
            next_message: New message text from the user.

        Returns:
            The detected Signal.
        """
        signal = detect_explicit_correction(next_message, last_decision_id)
        if signal is None:
            signal = detect_acceptance(last_decision_id)

        self._dispatch_background(self._process_signal(signal, detection_method="auto"))
        return signal

    async def on_conversation_close(
        self,
        conversation_id: str,
        last_decision_id: str,
    ) -> Signal | None:
        """Record implicit acceptance when a conversation concludes.

        Args:
            conversation_id: Conversation identifier.
            last_decision_id: Decision ID of the last turn.

        Returns:
            Acceptance Signal.
        """
        signal = detect_acceptance(last_decision_id)
        self._dispatch_background(self._process_signal(signal, detection_method="auto"))
        return signal

    async def report_manual_signal(
        self,
        decision_id: str,
        signal_type: str,
        value: float | None = None,
    ) -> None:
        """Record an explicit feedback signal from a user interface (e.g. thumbs up/down).

        Args:
            decision_id: Decision ID being rated.
            signal_type: Signal identifier (e.g. 'thumbs_up', 'thumbs_down', 'hard_regen').
            value: Optional explicit numeric value (defaults to spec values).
        """
        if signal_type in _MANUAL_SIGNAL_MAPPING:
            mapped_type, default_val = _MANUAL_SIGNAL_MAPPING[signal_type]
            actual_type = mapped_type
            if value is None:
                value = default_val
        else:
            actual_type = signal_type
            if value is None:
                value = SIGNAL_VALUES.get(signal_type, ACCEPTANCE_VALUE)

        signal = Signal(
            signal_type=actual_type,
            value=value,
            decision_id=decision_id,
        )
        self._dispatch_background(self._process_signal(signal, detection_method="manual"))

    async def wait_pending(self) -> None:
        """Wait for all pending background tasks to complete (useful in tests)."""
        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
