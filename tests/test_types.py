"""Unit tests for smartroute.types (Phase 1, module 1)."""

import json
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timezone

import pytest

from smartroute.types import (
    AdaptationRecord,
    ClassificationResult,
    ComplexityBucket,
    ConversationContext,
    DecisionRecord,
    DomainHint,
    FeatureVector,
    ReputationRecord,
    RoutingMeta,
    RoutingResult,
    Signal,
    SignalRecord,
    StreamChunk,
    TaskType,
)


def _feature_vector(complexity_irrelevant: float = 0.0) -> FeatureVector:
    return FeatureVector(
        token_count=10,
        code_block_ratio=0.0,
        is_question=False,
        is_instruction=True,
        instruction_verb_count=2,
        multi_step_count=0,
        ambiguity_score=0.0,
        domain_hint=DomainHint(domain="code", match_count=3, match_ratio=0.2),
        file_path_count=0,
        urgency_count=0,
        instruction_verb_density=0.1,
    )


class TestFeatureVectorSerialization:
    def test_feature_vector_serializes_to_json_without_ceremony(self) -> None:
        """features_json storage path: plain dataclasses.asdict + json.dumps."""
        as_dict = asdict(_feature_vector())
        serialized = json.dumps(as_dict)
        restored = json.loads(serialized)
        expected_hint = {"domain": "code", "match_count": 3, "match_ratio": 0.2}
        assert restored["domain_hint"] == expected_hint
        nested = DomainHint(**restored["domain_hint"])
        rebuilt = FeatureVector(**{**restored, "domain_hint": nested})
        assert rebuilt == _feature_vector()


def _classification(complexity: float) -> ClassificationResult:
    return ClassificationResult(
        task_type=TaskType.CODE,
        complexity=complexity,
        confidence=0.8,
        features=_feature_vector(),
    )


class TestEnums:
    def test_task_type_values_match_spec(self) -> None:
        assert {t.value for t in TaskType} == {
            "code",
            "creative",
            "reasoning",
            "summarization",
            "translation",
            "general",
        }

    def test_complexity_bucket_values_match_spec(self) -> None:
        assert {b.value for b in ComplexityBucket} == {"low", "medium", "high"}


class TestClassificationResult:
    @pytest.mark.parametrize(
        ("complexity", "expected"),
        [
            (0.0, ComplexityBucket.LOW),
            (0.32, ComplexityBucket.LOW),
            (0.33, ComplexityBucket.MEDIUM),
            (0.5, ComplexityBucket.MEDIUM),
            (0.659, ComplexityBucket.MEDIUM),
            (0.66, ComplexityBucket.HIGH),
            (1.0, ComplexityBucket.HIGH),
        ],
    )
    def test_complexity_bucket_boundaries(
        self, complexity: float, expected: ComplexityBucket
    ) -> None:
        assert _classification(complexity).complexity_bucket is expected

    def test_classification_result_is_frozen(self) -> None:
        result = _classification(0.5)
        with pytest.raises(FrozenInstanceError):
            result.complexity = 0.9  # type: ignore[misc]


class TestRoutingMeta:
    def test_reason_is_the_only_explanation_field(self) -> None:
        """Ruling on spec tension 04 vs 01: `reason` is canonical, no `why` alias."""
        meta = _meta()
        assert meta.reason == "Default routing: code task"
        assert not hasattr(meta, "why")

    def test_routing_meta_is_frozen(self) -> None:
        meta = _meta()
        with pytest.raises(FrozenInstanceError):
            meta.model = "groq/llama-3.1-8b"  # type: ignore[misc]

    def test_routing_result_carries_meta(self) -> None:
        meta = _meta()
        result = RoutingResult(text="hello", meta=meta)
        assert result.text == "hello"
        assert result.meta is meta


def _meta() -> RoutingMeta:
    return RoutingMeta(
        model="openai/gpt-4o-mini",
        task_type="code",
        complexity=0.45,
        complexity_bucket="medium",
        confidence=0.8,
        reason="Default routing: code task",
        reputation_score=0.5,
        was_adapted=False,
        override_applied=None,
        estimated_cost_usd=0.0001,
        latency_ms=120,
        decision_id="abc-123",
    )


class TestStreamChunk:
    def test_intermediate_chunk_has_no_meta(self) -> None:
        chunk = StreamChunk(text="hello", is_finished=False)
        assert chunk.meta is None
        assert not chunk.is_finished

    def test_final_chunk_has_meta(self) -> None:
        meta = _meta()
        chunk = StreamChunk(text="", is_finished=True, meta=meta)
        assert chunk.meta is meta
        assert chunk.is_finished


class TestConversationContext:
    def test_defaults_are_empty(self) -> None:
        ctx = ConversationContext()
        assert ctx.conversation_id is None
        assert ctx.turn_number == 0
        assert ctx.previous_decision_ids == []

    def test_previous_decision_ids_not_shared_between_instances(self) -> None:
        a = ConversationContext()
        b = ConversationContext()
        a.previous_decision_ids.append("d1")
        assert b.previous_decision_ids == []


class TestSignal:
    def test_signal_fields(self) -> None:
        signal = Signal(signal_type="hard_regen", value=-0.3, decision_id="d1")
        assert signal.signal_type == "hard_regen"
        assert signal.value == -0.3
        assert signal.decision_id == "d1"


class TestDecisionRecord:
    def test_privacy_no_full_prompt_field(self) -> None:
        fields = {f.name for f in DecisionRecord.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        assert "prompt" not in fields
        assert "prompt_hash" in fields
        assert "prompt_preview" in fields

    def test_ids_and_timestamps_auto_generated_and_distinct(self) -> None:
        a = DecisionRecord(
            prompt_hash="h1",
            task_type="code",
            complexity=0.5,
            complexity_bucket="medium",
            confidence=0.8,
            model_used="openai/gpt-4o-mini",
            provider_key="openai",
            reason="r",
        )
        b = DecisionRecord(
            prompt_hash="h1",
            task_type="code",
            complexity=0.5,
            complexity_bucket="medium",
            confidence=0.8,
            model_used="openai/gpt-4o-mini",
            provider_key="openai",
            reason="r",
        )
        assert a.id != b.id
        assert a.timestamp.tzinfo is timezone.utc
        assert isinstance(a.timestamp, datetime)

    def test_defaults(self) -> None:
        record = DecisionRecord(
            prompt_hash="h",
            task_type="code",
            complexity=0.1,
            complexity_bucket="low",
            confidence=0.5,
            model_used="groq/llama-3.1-8b",
            provider_key="groq",
            reason="r",
        )
        assert record.was_adapted is False
        assert record.prompt_preview is None
        assert record.override_applied is None


class TestSignalRecord:
    def test_defaults(self) -> None:
        record = SignalRecord(decision_id="d1", signal_type="acceptance", signal_value=0.05)
        assert record.detection_method == "auto"
        assert record.detected_at.tzinfo is timezone.utc


class TestReputationRecord:
    def test_neutral_defaults(self) -> None:
        record = ReputationRecord(bucket_key="code_low", model_tier="low")
        assert record.ema_score == 0.5
        assert record.call_count == 0
        assert record.last_bumped_at is None


class TestAdaptationRecord:
    def test_defaults(self) -> None:
        record = AdaptationRecord(
            bucket_key="code_low", old_tier="low", new_tier="medium", ema_at_bump=0.25
        )
        assert record.triggered_at.tzinfo is timezone.utc
