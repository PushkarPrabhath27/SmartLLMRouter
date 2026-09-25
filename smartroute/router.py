"""Public interface for SmartRoute: the Router class (spec 04).

High-level entry point that ties together heuristic classification, adaptive
reputation-based routing, unified provider dispatch with fallback, and implicit
feedback signal collection.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smartroute.classifier.classifier import HeuristicClassifier
from smartroute.config.loader import ConfigLoader
from smartroute.config.schema import Config
from smartroute.providers.base import BaseProvider
from smartroute.providers.dispatcher import ProviderDispatcher
from smartroute.routing.engine import RoutingEngine
from smartroute.routing.explainability import format_fallback_reason
from smartroute.signals.collector import SignalCollector
from smartroute.storage.connection import Storage
from smartroute.types import (
    ConversationContext,
    DecisionRecord,
    OverrideHook,
    ProjectReport,
    RoutingMeta,
    RoutingResult,
    StreamChunk,
)

logger = logging.getLogger(__name__)


def _resolve_storage_path(storage_path: str | None) -> str:
    """Resolve the SQLite database path per spec 04 and spec 08.

    Search order:
    1. Explicit ``storage_path`` argument
    2. ``SMARTROUTE_STORAGE`` environment variable
    3. Existing ``~/.smartroute/db.sqlite`` if present and cwd lacks one
    4. ``.smartroute/db.sqlite`` in the current working directory
    """
    if storage_path is not None:
        return storage_path
    env_path = os.environ.get("SMARTROUTE_STORAGE")
    if env_path:
        return env_path
    cwd_db = Path.cwd() / ".smartroute" / "db.sqlite"
    home_db = Path.home() / ".smartroute" / "db.sqlite"
    if not cwd_db.exists() and home_db.exists():
        return str(home_db)
    return str(cwd_db)


class Router:
    """Primary public interface for SmartRoute prompt routing.

    Orchestrates the entire routing lifecycle:
    1. Collects implicit signals (regeneration, turns) from previous decisions
    2. Classifies prompt complexity and task domain using local heuristics (<10ms)
    3. Evaluates routing rules, adaptive reputation scores, and default tiers
    4. Dispatches to LLM providers with automatic fallback chains
    5. Persists decision records (privacy-preserving: SHA-256 hash only)
    6. Returns responses with complete explainability metadata

    Args:
        config_path: Path to smartroute.yaml. If None, searches standard paths.
        override_hook: Optional callable for programmatic routing overrides.
        storage_path: Path to SQLite DB. If None, resolves per spec search path.
        providers: Optional pre-configured provider instances (for testing/mocking).
    """

    def __init__(
        self,
        config_path: str | None = None,
        override_hook: OverrideHook | None = None,
        storage_path: str | None = None,
        providers: dict[str, BaseProvider] | None = None,
    ) -> None:
        self.config_path = config_path
        self.override_hook = override_hook
        self.storage_path = _resolve_storage_path(storage_path)

        self._config: Config = ConfigLoader(config_path).load()
        self._storage: Storage = Storage(self.storage_path)
        self._classifier: HeuristicClassifier = HeuristicClassifier()
        self._engine: RoutingEngine = RoutingEngine(self._config, self._storage, self.override_hook)
        self._dispatcher: ProviderDispatcher = ProviderDispatcher(self._config, providers=providers)
        self._collector: SignalCollector = SignalCollector(self._storage, self._config)

    @property
    def config(self) -> Config:
        """The currently active configuration."""
        return self._config

    @property
    def storage(self) -> Storage:
        """The underlying storage facade."""
        return self._storage

    async def _ensure_connected(self) -> None:
        """Ensure storage is connected before database operations."""
        if self._storage._connection is None:
            await self._storage.connect()

    async def __aenter__(self) -> Router:
        """Async context manager entry."""
        await self._ensure_connected()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Flush pending background tasks and release network and database resources."""
        await self._collector.wait_pending()
        await self._dispatcher.close()
        await self._storage.close()

    async def complete(
        self,
        prompt: str,
        context: ConversationContext | None = None,
    ) -> RoutingResult:
        """Complete a prompt with automatic intelligent routing.

        Args:
            prompt: User prompt text.
            context: Optional conversation context for turn tracking and signal detection.

        Returns:
            RoutingResult containing generated text and explainability metadata.

        Raises:
            ProviderError: If all providers in the fallback chain fail.
            ClassificationError: If heuristic classification fails unrecoverably.
        """
        await self._ensure_connected()

        # Step A: Signal detection hooks
        await self._collector.on_new_prompt(prompt, context)
        if context and context.previous_decision_ids:
            last_decision_id = context.previous_decision_ids[-1]
            await self._collector.on_conversation_turn(
                conversation_id=context.conversation_id or "default",
                turn_number=context.turn_number,
                last_decision_id=last_decision_id,
                next_message=prompt,
            )

        # Step B: Heuristic classification (<10ms)
        classification = self._classifier.classify(prompt)

        # Step C: Decision hierarchy evaluation
        decision = await self._engine.route(prompt, classification, context)

        # Step D: Unified provider dispatch with fallback execution
        dispatch_res = await self._dispatcher.complete(decision.fallback_chain, prompt)

        # Determine explainability reason
        if dispatch_res.provider_used != decision.provider_key:
            err_msg = (
                dispatch_res.attempts[0].get("error", "primary provider failed")
                if dispatch_res.attempts
                else "primary provider failed"
            )
            reason = format_fallback_reason(
                decision.provider_key, dispatch_res.provider_used, err_msg
            )
        else:
            reason = decision.reason

        # Step E: Privacy-preserving decision persistence (hash only, never full prompt)
        decision_id = str(uuid.uuid4())
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompt_preview = prompt[:100]
        now = datetime.now(timezone.utc)

        record = DecisionRecord(
            id=decision_id,
            timestamp=now,
            prompt_hash=prompt_hash,
            prompt_preview=prompt_preview,
            task_type=classification.task_type.value,
            complexity=classification.complexity,
            complexity_bucket=classification.complexity_bucket.value,
            confidence=classification.confidence,
            model_used=dispatch_res.model_used,
            provider_key=dispatch_res.provider_used,
            estimated_cost_usd=decision.estimated_cost_usd,
            actual_cost_usd=None,
            latency_ms=dispatch_res.latency_ms,
            was_adapted=decision.was_adapted,
            override_applied=decision.override_applied,
            reason=reason,
            conversation_id=context.conversation_id if context else None,
            turn_number=context.turn_number if context else None,
            features_json=None,
        )
        await self._storage.store_decision(record)

        # Register prompt for ongoing signal correlation
        self._collector.register_prompt(
            decision_id=decision_id,
            prompt=prompt,
            task_type=classification.task_type.value,
            complexity_bucket=classification.complexity_bucket.value,
            timestamp=now,
        )

        if context is not None:
            context.previous_decision_ids.append(decision_id)
            context.turn_number += 1

        # Step F: Assemble and return result
        meta = RoutingMeta(
            model=dispatch_res.model_used,
            task_type=classification.task_type.value,
            complexity=classification.complexity,
            complexity_bucket=classification.complexity_bucket.value,
            confidence=classification.confidence,
            reason=reason,
            reputation_score=decision.reputation_score,
            was_adapted=decision.was_adapted,
            override_applied=decision.override_applied,
            estimated_cost_usd=decision.estimated_cost_usd,
            latency_ms=dispatch_res.latency_ms,
            decision_id=decision_id,
        )
        return RoutingResult(text=dispatch_res.text, meta=meta)

    async def stream(
        self,
        prompt: str,
        context: ConversationContext | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response with automatic intelligent routing.

        Routing decision is evaluated before streaming starts. Intermediate chunks
        contain text deltas with is_finished=False. The final chunk contains
        is_finished=True and full explainability RoutingMeta.

        Args:
            prompt: User prompt text.
            context: Optional conversation context.

        Yields:
            StreamChunk instances.

        Raises:
            ProviderError: If all providers fail to stream.
        """
        await self._ensure_connected()

        # Step A: Signal detection hooks
        await self._collector.on_new_prompt(prompt, context)
        if context and context.previous_decision_ids:
            last_decision_id = context.previous_decision_ids[-1]
            await self._collector.on_conversation_turn(
                conversation_id=context.conversation_id or "default",
                turn_number=context.turn_number,
                last_decision_id=last_decision_id,
                next_message=prompt,
            )

        # Step B: Classification
        classification = self._classifier.classify(prompt)

        # Step C: Route
        decision = await self._engine.route(prompt, classification, context)

        # Step D: Stream dispatch
        start_time = time.perf_counter()
        stream_gen = self._dispatcher.stream(decision.fallback_chain, prompt)

        accumulated_text: list[str] = []
        async for chunk in stream_gen:
            if chunk.is_finished:
                if chunk.text:
                    accumulated_text.append(chunk.text)
                    yield StreamChunk(text=chunk.text, is_finished=False, meta=None)
                break
            accumulated_text.append(chunk.text)
            yield StreamChunk(text=chunk.text, is_finished=False, meta=None)

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Identify actual provider and model used during streaming
        provider_used = self._dispatcher.last_stream_provider or decision.provider_key
        model_used = self._dispatcher.last_stream_model or decision.model

        if provider_used != decision.provider_key:
            reason = format_fallback_reason(
                decision.provider_key,
                provider_used,
                "primary provider stream failed to initialize",
            )
        else:
            reason = decision.reason

        # Persist decision
        decision_id = str(uuid.uuid4())
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompt_preview = prompt[:100]
        now = datetime.now(timezone.utc)

        record = DecisionRecord(
            id=decision_id,
            timestamp=now,
            prompt_hash=prompt_hash,
            prompt_preview=prompt_preview,
            task_type=classification.task_type.value,
            complexity=classification.complexity,
            complexity_bucket=classification.complexity_bucket.value,
            confidence=classification.confidence,
            model_used=model_used,
            provider_key=provider_used,
            estimated_cost_usd=decision.estimated_cost_usd,
            actual_cost_usd=None,
            latency_ms=latency_ms,
            was_adapted=decision.was_adapted,
            override_applied=decision.override_applied,
            reason=reason,
            conversation_id=context.conversation_id if context else None,
            turn_number=context.turn_number if context else None,
            features_json=None,
        )
        await self._storage.store_decision(record)

        self._collector.register_prompt(
            decision_id=decision_id,
            prompt=prompt,
            task_type=classification.task_type.value,
            complexity_bucket=classification.complexity_bucket.value,
            timestamp=now,
        )

        if context is not None:
            context.previous_decision_ids.append(decision_id)
            context.turn_number += 1

        # Final termination chunk with metadata
        final_meta = RoutingMeta(
            model=model_used,
            task_type=classification.task_type.value,
            complexity=classification.complexity,
            complexity_bucket=classification.complexity_bucket.value,
            confidence=classification.confidence,
            reason=reason,
            reputation_score=decision.reputation_score,
            was_adapted=decision.was_adapted,
            override_applied=decision.override_applied,
            estimated_cost_usd=decision.estimated_cost_usd,
            latency_ms=latency_ms,
            decision_id=decision_id,
        )
        yield StreamChunk(text="", is_finished=True, meta=final_meta)

    async def report(self) -> ProjectReport:
        """Generate a project health and routing analytics report.

        Returns:
            ProjectReport containing cost, latency, distributions, and adaptations.
        """
        await self._ensure_connected()
        stats = await self._storage.get_project_stats()
        return ProjectReport(
            total_decisions=stats["total_decisions"],
            total_cost_usd=stats["total_cost_usd"],
            average_latency_ms=stats["average_latency_ms"],
            model_distribution=stats["model_distribution"],
            bucket_distribution=stats["bucket_distribution"],
            adapted_buckets=stats["adapted_buckets"],
            recent_decisions=stats["recent_decisions"],
        )

    async def reset_reputation(self, bucket_key: str | None = None) -> None:
        """Reset learned reputation scores and adaptation history.

        Args:
            bucket_key: Specific bucket key (e.g. 'code_low') to reset. If None,
                resets all reputation scores across all buckets.
        """
        await self._ensure_connected()
        await self._storage.reset_reputation(bucket_key)

    async def report_signal(
        self,
        decision_id: str,
        signal_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Report an explicit feedback signal for a past routing decision.

        Enables user interface feedback (such as thumbs-up/thumbs-down) to directly
        and adaptively update reputation scores.

        Args:
            decision_id: UUID of the decision being rated.
            signal_type: Signal name ('thumbs_up', 'thumbs_down', 'hard_regen', etc.).
            metadata: Optional extra feedback metadata.
        """
        await self._ensure_connected()
        await self._collector.report_manual_signal(decision_id=decision_id, signal_type=signal_type)

    def reload_config(self) -> None:
        """Reload configuration from disk and re-initialize routing subsystems."""
        self._config = ConfigLoader(self.config_path).load()
        self._engine = RoutingEngine(self._config, self._storage, self.override_hook)
        self._dispatcher = ProviderDispatcher(self._config)
        self._collector.config = self._config
