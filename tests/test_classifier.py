"""Unit tests for smartroute.classifier.classifier (Phase 2, module 3)."""

import pytest

from smartroute.classifier import classifier as classifier_module
from smartroute.classifier.classifier import (
    HeuristicClassifier,
    classify_prompt,
    compute_complexity,
    compute_confidence,
)
from smartroute.classifier.features import extract_features, get_encoder
from smartroute.types import DomainHint, FeatureVector, TaskType

get_encoder()


def _vector(**overrides: object) -> FeatureVector:
    base: dict[str, object] = {
        "token_count": 100,
        "code_block_ratio": 0.0,
        "is_question": False,
        "is_instruction": True,
        "instruction_verb_count": 2,
        "multi_step_count": 0,
        "ambiguity_score": 0.0,
        "domain_hint": DomainHint(domain="code", match_count=5, match_ratio=0.2),
        "file_path_count": 0,
        "urgency_count": 0,
        "instruction_verb_density": 0.2,
    }
    base.update(overrides)
    return FeatureVector(**base)  # type: ignore[arg-type]


class TestComputeComplexity:
    @pytest.mark.parametrize(
        ("token_count", "expected_base"),
        [(10, 0.1), (60, 0.3), (250, 0.5), (600, 0.7)],
    )
    def test_token_bands(self, token_count: int, expected_base: float) -> None:
        features = _vector(
            token_count=token_count,
            is_instruction=False,
            instruction_verb_density=0.0,
            domain_hint=DomainHint("", 0, 0.0),
        )
        assert compute_complexity(features) == pytest.approx(expected_base)

    def test_code_ratio_contributes(self) -> None:
        base = compute_complexity(_vector(code_block_ratio=0.0))
        with_code = compute_complexity(_vector(code_block_ratio=0.5))
        assert with_code == pytest.approx(base + 0.1)

    def test_multi_step_capped_at_03(self) -> None:
        at_cap = compute_complexity(_vector(multi_step_count=6))
        over_cap = compute_complexity(_vector(multi_step_count=20))
        assert at_cap == over_cap  # 6 * 0.05 = 0.3 already at cap

    def test_ambiguity_only_above_01(self) -> None:
        at_boundary = compute_complexity(_vector(ambiguity_score=0.1))
        above = compute_complexity(_vector(ambiguity_score=0.11))
        assert above == pytest.approx(at_boundary + 0.15)

    def test_file_refs_capped_at_01(self) -> None:
        at_cap = compute_complexity(_vector(file_path_count=5))
        over_cap = compute_complexity(_vector(file_path_count=50))
        assert at_cap == over_cap  # 5 * 0.02 = 0.1 already at cap

    def test_question_penalty(self) -> None:
        base = compute_complexity(_vector(is_question=False))
        penalized = compute_complexity(_vector(is_question=True, is_instruction=False))
        assert penalized == pytest.approx(base - 0.15)

    def test_mixed_question_and_instruction_no_penalty(self) -> None:
        base = compute_complexity(_vector(is_question=False))
        mixed = compute_complexity(_vector(is_question=True, is_instruction=True))
        assert mixed == pytest.approx(base)

    def test_result_never_exceeds_bounds(self) -> None:
        extreme = _vector(
            token_count=10_000,
            code_block_ratio=1.0,
            multi_step_count=50,
            ambiguity_score=1.0,
            file_path_count=100,
            instruction_verb_density=1.0,
            is_question=False,
        )
        assert 0.0 <= compute_complexity(extreme) <= 1.0

    def test_question_only_prompt_stays_non_negative(self) -> None:
        features = _vector(
            token_count=10,
            is_question=True,
            is_instruction=False,
            instruction_verb_density=0.0,
        )
        assert compute_complexity(features) >= 0.0


class TestComputeConfidence:
    def test_clear_domain_boosts(self) -> None:
        assert compute_confidence(_vector(), TaskType.CODE) > 0.5

    def test_unclear_domain_reduces(self) -> None:
        features = _vector(domain_hint=DomainHint("", 0, 0.0))
        assert compute_confidence(features, TaskType.GENERAL) < 0.5

    def test_strong_code_signal_boosts(self) -> None:
        features = _vector(code_block_ratio=0.9)
        base = compute_confidence(_vector(), TaskType.CODE)
        assert compute_confidence(features, TaskType.CODE) == pytest.approx(base + 0.1)

    def test_pure_question_boosts(self) -> None:
        features = _vector(is_question=True, is_instruction=False)
        base = compute_confidence(_vector(), TaskType.CODE)
        assert compute_confidence(features, TaskType.CODE) == pytest.approx(base + 0.1)

    def test_mixed_signals_reduce(self) -> None:
        features = _vector(is_question=True, is_instruction=True)
        base = compute_confidence(_vector(), TaskType.CODE)
        assert compute_confidence(features, TaskType.CODE) == pytest.approx(base - 0.15)

    def test_bounds(self) -> None:
        assert 0.0 <= compute_confidence(_vector(), TaskType.CODE) <= 1.0


class TestClassify:
    def test_code_prompt_classified_as_code(self) -> None:
        result = classify_prompt("Refactor this function in main.py to use async await properly")
        assert result.task_type == TaskType.CODE
        assert 0.0 <= result.complexity <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.features.token_count > 0

    def test_creative_prompt_classified_as_creative(self) -> None:
        result = classify_prompt("Write a short story with a strong narrative and vivid characters")
        assert result.task_type == TaskType.CREATIVE

    def test_translation_prompt_classified(self) -> None:
        result = classify_prompt("Translate this paragraph from English to French")
        assert result.task_type == TaskType.TRANSLATION

    def test_generic_prompt_defaults_to_general(self) -> None:
        result = classify_prompt("Tell me about your weekend plans sometime")
        assert result.task_type == TaskType.GENERAL

    def test_empty_string(self) -> None:
        result = classify_prompt("")
        assert result.task_type == TaskType.GENERAL
        assert 0.0 <= result.complexity <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_complexity_bucket_property_consistent(self) -> None:
        result = classify_prompt("debug this stack trace error in the function")
        assert result.complexity_bucket is not None

    def test_classifier_is_stateless(self) -> None:
        clf = HeuristicClassifier()
        first = clf.classify("write a poem about the sea")
        second = clf.classify("debug the runtime error in the module")
        assert first.task_type == TaskType.CREATIVE
        assert second.task_type == TaskType.CODE


class TestFailOpen:
    def test_extract_features_crash_falls_open(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def boom(prompt: str) -> FeatureVector:
            raise RuntimeError("simulated extraction failure")

        monkeypatch.setattr(classifier_module, "extract_features", boom)
        with caplog.at_level("WARNING"):
            result = HeuristicClassifier().classify("any prompt at all")
        assert result.task_type == TaskType.GENERAL
        assert result.complexity == 0.5
        assert result.confidence == 0.3
        assert result.features.token_count == 4  # "any prompt at all"
        assert any("fail-open" in record.message for record in caplog.records)

    def test_real_tiktoken_failure_falls_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Break the real tiktoken path (not a mock of our own code)."""
        import tiktoken

        def broken_encoding_for_model(name: str) -> tiktoken.Encoding:
            raise KeyError(f"no encoding for {name}")

        monkeypatch.setattr(tiktoken, "encoding_for_model", broken_encoding_for_model)
        monkeypatch.setattr(classifier_module, "_ENCODERS", {}, raising=False)
        # features.py holds its own cache reference; clear it via the module.
        import smartroute.classifier.features as features_module

        monkeypatch.setattr(features_module, "_ENCODERS", {})
        result = HeuristicClassifier().classify("debug the error please")
        assert result.task_type == TaskType.GENERAL
        assert result.complexity == 0.5
        assert result.confidence == 0.3

    def test_fail_open_features_default_vector(self) -> None:
        result = classify_prompt("one two three four five")
        # Sanity: normal path does not produce the fail-open signature.
        assert not (
            result.complexity == 0.5
            and result.confidence == 0.3
            and result.task_type == TaskType.GENERAL
        )


class TestMonotonicity:
    def test_complexity_monotonic_with_token_count(self) -> None:
        """Longer prompts never land in a lower token band (spec 11 property)."""
        short = compute_complexity(extract_features("short prompt"))
        medium = compute_complexity(extract_features("explain this concept " * 8))
        long = compute_complexity(extract_features("explain this concept " * 100))
        assert long > medium > short
