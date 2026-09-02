"""Heuristic classifier: complexity scoring, confidence, and fail-open (spec 05).

Stateless and thread-safe: no mutable shared state, no network calls, no
disk I/O during classification.
"""

from __future__ import annotations

import logging

from smartroute.classifier.features import extract_features
from smartroute.types import (
    ClassificationResult,
    DomainHint,
    FeatureVector,
    TaskType,
)

logger = logging.getLogger(__name__)

_LOW_TOKEN_BAND = 50
_MEDIUM_TOKEN_BAND = 200
_HIGH_TOKEN_BAND = 500


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp value into [low, high]."""
    return max(low, min(high, value))


def compute_complexity(features: FeatureVector) -> float:
    """Compute the 0.0-1.0 complexity score from a feature vector (spec 05).

    Args:
        features: The extracted feature vector.

    Returns:
        Complexity score clamped to [0.0, 1.0].
    """
    tokens = features.token_count
    if tokens < _LOW_TOKEN_BAND:
        score = 0.1
    elif tokens < _MEDIUM_TOKEN_BAND:
        score = 0.3
    elif tokens < _HIGH_TOKEN_BAND:
        score = 0.5
    else:
        score = 0.7

    score += features.code_block_ratio * 0.2
    score += min(features.multi_step_count * 0.05, 0.3)
    if features.ambiguity_score > 0.1:
        score += 0.15
    score += min(features.file_path_count * 0.02, 0.1)
    if features.instruction_verb_density > 0.1:
        score += 0.1
    if features.is_question and not features.is_instruction:
        score -= 0.15

    return _clamp(score)


def compute_confidence(features: FeatureVector, task_type: TaskType) -> float:
    """Compute the 0.0-1.0 classifier confidence (spec 05).

    Args:
        features: The extracted feature vector.
        task_type: The assigned task type (reserved for scoring nuances).

    Returns:
        Confidence score clamped to [0.0, 1.0].
    """
    confidence = 0.5
    if features.domain_hint.match_ratio > 0.3:
        confidence += 0.3
    elif features.domain_hint.match_ratio > 0.1:
        confidence += 0.15
    else:
        confidence -= 0.2
    if features.code_block_ratio > 0.5:
        confidence += 0.1
    if features.is_question and not features.is_instruction:
        confidence += 0.1
    if features.is_question and features.is_instruction:
        confidence -= 0.15
    return _clamp(confidence)


def _task_type_from_hint(hint: DomainHint) -> TaskType:
    """Map a domain hint to a TaskType; empty hint -> GENERAL."""
    if not hint.domain:
        return TaskType.GENERAL
    return TaskType(hint.domain)


def _default_features(prompt: str) -> FeatureVector:
    """Fail-open default feature vector (spec 05).

    Args:
        prompt: The prompt whose classification failed.

    Returns:
        All-zero vector except ``token_count=len(prompt.split())``.
    """
    return FeatureVector(
        token_count=len(prompt.split()),
        code_block_ratio=0.0,
        is_question=False,
        is_instruction=False,
        instruction_verb_count=0,
        multi_step_count=0,
        ambiguity_score=0.0,
        domain_hint=DomainHint(domain="", match_count=0, match_ratio=0.0),
        file_path_count=0,
        urgency_count=0,
        instruction_verb_density=0.0,
    )


class HeuristicClassifier:
    """Classify prompts by task type, complexity, and confidence (spec 05).

    The classifier is heuristic and stateless: regex features, simple
    statistics, and tiktoken token counts. Any internal failure falls open
    to a safe default result rather than raising (spec 05 "Fail-Open
    Behavior").
    """

    def classify(self, prompt: str) -> ClassificationResult:
        """Classify a prompt into task type, complexity, and confidence.

        Args:
            prompt: The raw prompt string.

        Returns:
            A ClassificationResult. On any internal failure this is the
            fail-open default: GENERAL, complexity 0.5, confidence 0.3.
        """
        try:
            features = extract_features(prompt)
            task_type = _task_type_from_hint(features.domain_hint)
            return ClassificationResult(
                task_type=task_type,
                complexity=compute_complexity(features),
                confidence=compute_confidence(features, task_type),
                features=features,
            )
        except Exception as exc:  # noqa: BLE001 - fail-open is the spec contract
            logger.warning("classification failed, using fail-open default: %s", exc)
            return ClassificationResult(
                task_type=TaskType.GENERAL,
                complexity=0.5,
                confidence=0.3,
                features=_default_features(prompt),
            )


def classify_prompt(prompt: str) -> ClassificationResult:
    """Module-level convenience wrapper around :class:`HeuristicClassifier`."""
    return HeuristicClassifier().classify(prompt)
