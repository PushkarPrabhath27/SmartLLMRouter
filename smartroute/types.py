"""Shared type definitions for SmartRoute.

All dataclasses and enums shared across the classifier, routing engine,
providers, storage, and signal collectors live here. Storage record types
mirror the SQLite schema in ``specs/08_STORAGE_SPEC.md``; public API types
mirror ``specs/04_API_SPEC.md``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> datetime:
    """Current UTC time; the single source of record timestamps."""
    return datetime.now(timezone.utc)


class TaskType(Enum):
    """Semantic task categories a prompt can be classified into."""

    CODE = "code"
    CREATIVE = "creative"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    GENERAL = "general"


class ComplexityBucket(Enum):
    """Complexity buckets derived from the 0.0-1.0 complexity score."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class DomainHint:
    """Result of domain keyword matching (classifier feature F6).

    Attributes:
        domain: Best-matching task type name (empty string if no match).
        match_count: Number of domain keywords found in the prompt.
        match_ratio: Keyword matches divided by total words, in [0.0, 1.0].
    """

    domain: str
    match_count: int
    match_ratio: float


@dataclass(frozen=True)
class FeatureVector:
    """Raw feature vector extracted from a prompt (classifier features F1-F9).

    Attributes:
        token_count: F1 — tiktoken token count of the prompt.
        code_block_ratio: F2 — fraction of tokens inside fenced code blocks.
        is_question: F3 — prompt ends with "?" or starts with a wh-word.
        is_instruction: F3 — prompt contains imperative verbs.
        instruction_verb_count: F3 — number of imperative verbs found.
        multi_step_count: F4 — count of multi-step transition markers.
        ambiguity_score: F5 — hedge-word ratio in [0.0, 1.0].
        domain_hint: F6 — best domain keyword match.
        file_path_count: F7 — count of unique file path references.
        urgency_count: F8 — count of urgency keywords (explainability only).
        instruction_verb_density: F9 — imperative verbs / total words.
    """

    token_count: int
    code_block_ratio: float
    is_question: bool
    is_instruction: bool
    instruction_verb_count: int
    multi_step_count: int
    ambiguity_score: float
    domain_hint: DomainHint
    file_path_count: int
    urgency_count: int
    instruction_verb_density: float


@dataclass(frozen=True)
class ClassificationResult:
    """Output of the heuristic classifier for a single prompt.

    Attributes:
        task_type: Classified task category.
        complexity: Complexity score in [0.0, 1.0].
        confidence: Classifier confidence in [0.0, 1.0].
        features: Raw feature vector used to produce the scores.

    Properties:
        complexity_bucket: LOW (< 0.33), MEDIUM (< 0.66), otherwise HIGH.
    """

    task_type: TaskType
    complexity: float
    confidence: float
    features: FeatureVector

    @property
    def complexity_bucket(self) -> ComplexityBucket:
        """Map the complexity score onto its bucket using spec boundaries."""
        if self.complexity < 0.33:
            return ComplexityBucket.LOW
        if self.complexity < 0.66:
            return ComplexityBucket.MEDIUM
        return ComplexityBucket.HIGH


@dataclass(frozen=True)
class RoutingMeta:
    """Explainability metadata attached to every routing decision.

    Attributes:
        model: Fully qualified model id, e.g. ``"openai/gpt-4o-mini"``.
        task_type: Classified task type name, e.g. ``"code"``.
        complexity: Complexity score in [0.0, 1.0].
        complexity_bucket: Bucket name: ``"low"``, ``"medium"``, or ``"high"``.
        confidence: Classifier confidence in [0.0, 1.0].
        reason: Human-readable explanation of why this model was chosen.
        reputation_score: Current EMA reputation for the bucket at decision time.
        was_adapted: True if the reputation system bumped the tier.
        override_applied: Which override fired (``"programmatic_hook"`` or a
            YAML rule description), or None.
        estimated_cost_usd: Pre-flight cost estimate for this call.
        latency_ms: Actual provider call latency in milliseconds.
        decision_id: UUID of the stored decision row (for signal reporting).

    Properties:
        why: Alias for ``reason`` (the documented quick-access name).
    """

    model: str
    task_type: str
    complexity: float
    complexity_bucket: str
    confidence: float
    reason: str
    reputation_score: float
    was_adapted: bool
    override_applied: str | None
    estimated_cost_usd: float
    latency_ms: int
    decision_id: str

    @property
    def why(self) -> str:
        """Human-readable explanation of the routing decision."""
        return self.reason


@dataclass(frozen=True)
class RoutingResult:
    """Result of ``Router.complete()``.

    Attributes:
        text: The completed response text.
        meta: Explainability metadata for the routing decision.
    """

    text: str
    meta: RoutingMeta


@dataclass(frozen=True)
class StreamChunk:
    """One chunk of a streamed response from ``Router.stream()``.

    Attributes:
        text: Chunk text delta (empty string on the final chunk).
        is_finished: True on the final chunk.
        meta: RoutingMeta, only present on the final chunk (otherwise None).
    """

    text: str
    is_finished: bool
    meta: RoutingMeta | None = None


@dataclass
class ConversationContext:
    """Lightweight conversation state. V1 stores but does not escalate.

    Attributes:
        conversation_id: Optional conversation identifier.
        turn_number: 0-based turn counter within the conversation.
        previous_decision_ids: Decision ids of earlier turns in this
            conversation (used by signal detection).
    """

    conversation_id: str | None = None
    turn_number: int = 0
    previous_decision_ids: list[str] = field(default_factory=list)


@dataclass
class ProjectReport:
    """Aggregated project health stats returned by ``Router.report()``.

    Attributes:
        total_decisions: Number of stored routing decisions.
        total_cost_usd: Sum of actual cost (falling back to estimate).
        average_latency_ms: Mean provider latency across decisions.
        model_distribution: model id -> decision count.
        bucket_distribution: complexity bucket name -> decision count.
        adapted_buckets: Adaptation events (bucket, old/new tier, timestamp).
        recent_decisions: Last 10 decisions rendered as dicts.
    """

    total_decisions: int
    total_cost_usd: float
    average_latency_ms: float
    model_distribution: dict[str, int]
    bucket_distribution: dict[str, int]
    adapted_buckets: list[dict[str, Any]]
    recent_decisions: list[dict[str, Any]]


OverrideHook = Callable[[str, ConversationContext | None], str | None]


@dataclass(frozen=True)
class Signal:
    """A detected implicit feedback signal awaiting storage (spec 07).

    Attributes:
        signal_type: One of ``hard_regen``, ``soft_regen``,
            ``explicit_correction``, ``acceptance``.
        value: Numeric signal value applied to the reputation EMA.
        decision_id: The past decision this signal applies to.
    """

    signal_type: str
    value: float
    decision_id: str


def _new_id() -> str:
    """Generate a fresh UUID v4 string for record primary keys."""
    return str(uuid.uuid4())


@dataclass
class DecisionRecord:
    """Row of the ``decisions`` table (spec 08).

    Privacy: contains only the SHA-256 prompt hash and an optional
    100-character preview. The full prompt is never persisted.

    Attributes:
        id: UUID v4 primary key (generated automatically).
        timestamp: UTC decision time (defaults to now).
        prompt_hash: SHA-256 hex digest of the prompt.
        prompt_preview: First 100 chars of the prompt, for debugging only.
        task_type: Classified task type name.
        complexity: Complexity score in [0.0, 1.0].
        complexity_bucket: Bucket name at decision time.
        confidence: Classifier confidence in [0.0, 1.0].
        model_used: Fully qualified model id, e.g. ``"openai/gpt-4o-mini"``.
        provider_key: Provider key, e.g. ``"openai"``.
        estimated_cost_usd: Pre-flight cost estimate, if computed.
        actual_cost_usd: Cost reported by the provider, if any.
        latency_ms: Provider latency in milliseconds, if recorded.
        was_adapted: True if the reputation system bumped the tier.
        override_applied: Which override fired, if any.
        reason: Human-readable routing explanation.
        conversation_id: Conversation this decision belongs to, if any.
        turn_number: Turn number within the conversation, if any.
        features_json: JSON blob of the full feature vector.
    """

    prompt_hash: str
    task_type: str
    complexity: float
    complexity_bucket: str
    confidence: float
    model_used: str
    provider_key: str
    reason: str
    id: str = field(default_factory=_new_id)
    timestamp: datetime = field(default_factory=lambda: _utcnow())
    prompt_preview: str | None = None
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    latency_ms: int | None = None
    was_adapted: bool = False
    override_applied: str | None = None
    conversation_id: str | None = None
    turn_number: int | None = None
    features_json: str | None = None


@dataclass
class SignalRecord:
    """Row of the ``signals`` table (spec 08).

    Attributes:
        decision_id: The decision this signal applies to (FK).
        signal_type: One of the four known signal types.
        signal_value: Numeric value applied to the reputation EMA.
        id: UUID v4 primary key (generated automatically).
        detected_at: UTC detection time (defaults to now).
        detection_method: ``"auto"`` or ``"manual"``.
    """

    decision_id: str
    signal_type: str
    signal_value: float
    id: str = field(default_factory=_new_id)
    detected_at: datetime = field(default_factory=lambda: _utcnow())
    detection_method: str | None = "auto"


@dataclass
class ReputationRecord:
    """Row of the ``reputation`` table (spec 08).

    Attributes:
        bucket_key: ``"{task_type}_{complexity_bucket}"``, e.g. ``"code_low"``.
        model_tier: Tier the EMA applies to: ``low``, ``medium``, or ``high``.
        id: UUID v4 primary key (generated automatically).
        ema_score: EMA reputation in [0.0, 1.0]; 0.5 is neutral.
        call_count: Number of signals applied to this bucket/tier pair.
        last_updated: UTC time of the last EMA update (defaults to now).
        last_bumped_at: UTC time of the last auto-bump, if any.
    """

    bucket_key: str
    model_tier: str
    id: str = field(default_factory=_new_id)
    ema_score: float = 0.5
    call_count: int = 0
    last_updated: datetime = field(default_factory=lambda: _utcnow())
    last_bumped_at: datetime | None = None


@dataclass
class AdaptationRecord:
    """Row of the ``adaptations`` table (spec 08): one auto-bump event.

    Attributes:
        bucket_key: Bucket that was bumped.
        old_tier: Tier before the bump.
        new_tier: Tier after the bump.
        ema_at_bump: EMA score that triggered the bump.
        id: UUID v4 primary key (generated automatically).
        triggered_at: UTC bump time (defaults to now).
    """

    bucket_key: str
    old_tier: str
    new_tier: str
    ema_at_bump: float
    id: str = field(default_factory=_new_id)
    triggered_at: datetime = field(default_factory=lambda: _utcnow())
