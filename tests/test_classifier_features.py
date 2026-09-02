"""Unit tests for smartroute.classifier.features (Phase 2, module 2).

Warms the tiktoken encoder once at module import so no test performs a
network fetch (tiktoken caches BPE files on disk after first load).
"""

import pytest
import tiktoken

from smartroute.classifier.features import (
    extract_features,
    get_encoder,
)

# Warm the encoder before any timing/network-sensitive test runs.
get_encoder()


class TestGetEncoder:
    def test_encoder_is_cached(self) -> None:
        first = get_encoder()
        second = get_encoder()
        assert first is second

    def test_encoder_is_real_tiktoken_encoding(self) -> None:
        assert isinstance(get_encoder(), tiktoken.Encoding)


class TestF1TokenCount:
    def test_empty_string_is_zero(self) -> None:
        assert extract_features("").token_count == 0

    def test_simple_prompt_counts_tokens(self) -> None:
        features = extract_features("hello world, how are you?")
        assert features.token_count > 0

    def test_longer_prompt_counts_more_tokens(self) -> None:
        short = extract_features("hello world").token_count
        longer = extract_features("hello world " * 20).token_count
        assert longer > short


class TestF2CodeBlockRatio:
    def test_no_code_blocks_is_zero(self) -> None:
        assert extract_features("plain text only").code_block_ratio == 0.0

    def test_only_code_block_is_near_one(self) -> None:
        prompt = "```\nprint('hello world')\n```"
        features = extract_features(prompt)
        assert features.code_block_ratio > 0.8

    def test_partial_code_block_is_fractional(self) -> None:
        prompt = "please explain this code:\n```\nx = 1\n```"
        features = extract_features(prompt)
        assert 0.0 < features.code_block_ratio < 1.0

    def test_unclosed_fence_counts_nothing(self) -> None:
        features = extract_features("```\nunclosed block")
        assert features.code_block_ratio == 0.0


class TestF3QuestionVsInstruction:
    def test_ends_with_question_mark(self) -> None:
        features = extract_features("What is a decorator?")
        assert features.is_question is True

    def test_starts_with_wh_word(self) -> None:
        features = extract_features("how do I parse JSON in Python")
        assert features.is_question is True

    def test_imperative_prompt_is_instruction(self) -> None:
        features = extract_features("Refactor this function to use pathlib")
        assert features.is_instruction is True
        assert features.instruction_verb_count >= 1
        assert features.is_question is False

    def test_instruction_verb_count_counts_occurrences(self) -> None:
        # F3 verb list: add, update (x2) count; "import" is not an F3 verb.
        features = extract_features("add the import, update the class, add tests")
        assert features.instruction_verb_count == 3

    def test_neither_question_nor_instruction(self) -> None:
        features = extract_features("The quick brown fox jumps over the lazy dog")
        assert features.is_question is False
        assert features.is_instruction is False


class TestF4MultiStepMarkers:
    def test_no_markers_is_zero(self) -> None:
        assert extract_features("do the thing").multi_step_count == 0

    def test_counts_each_marker(self) -> None:
        prompt = "First do this, then do that, finally clean up"
        assert extract_features(prompt).multi_step_count == 3

    def test_step_number_marker(self) -> None:
        assert extract_features("step 1: install. step 2: run").multi_step_count == 2

    def test_phrases_match(self) -> None:
        prompt = "after that, followed by a final check"
        assert extract_features(prompt).multi_step_count == 2

    def test_word_boundary_no_substring_match(self) -> None:
        # "firstly" contains "first" but must not match it.
        assert extract_features("firstly, do it").multi_step_count == 0


class TestF5Ambiguity:
    def test_no_hedges_is_zero(self) -> None:
        assert extract_features("write the function now").ambiguity_score == 0.0

    def test_counts_hedges_over_total_words(self) -> None:
        # 4 hedge hits ("maybe", "I think", "might", "probably") in 7 words.
        features = extract_features("maybe I think this might work, probably")
        assert features.ambiguity_score == pytest.approx(4 / 7)

    def test_literal_either_or_phrase(self) -> None:
        features = extract_features("either...or whatever you prefer")
        assert features.ambiguity_score > 0.0

    def test_empty_prompt_is_zero(self) -> None:
        assert extract_features("").ambiguity_score == 0.0


class TestF6DomainHint:
    def test_code_hint_populated(self) -> None:
        features = extract_features("debug the function and fix the syntax error")
        assert features.domain_hint.domain == "code"

    def test_no_domain_returns_empty_string(self) -> None:
        features = extract_features("gibberish words only here")
        assert features.domain_hint.domain == ""


class TestF7FilePaths:
    def test_code_file_reference(self) -> None:
        assert extract_features("look at main.py").file_path_count == 1

    def test_posix_path_reference(self) -> None:
        assert extract_features("edit /usr/local/bin/app.cfg please").file_path_count >= 1

    def test_windows_path_reference(self) -> None:
        features = extract_features(r"check C:\Users\dev\config.yaml")
        assert features.file_path_count >= 1

    def test_known_directory_reference(self) -> None:
        assert extract_features("add tests to tests/ directory").file_path_count >= 1

    def test_unique_counting(self) -> None:
        features = extract_features("main.py and main.py again")
        assert features.file_path_count == 1

    def test_no_references_is_zero(self) -> None:
        assert extract_features("nothing to see here").file_path_count == 0


class TestF8Urgency:
    def test_counts_urgency_keywords(self) -> None:
        # urgent, production, broken, ASAP are all spec urgency keywords.
        features = extract_features("urgent: production is broken, fix ASAP")
        assert features.urgency_count == 4

    def test_no_urgency_is_zero(self) -> None:
        assert extract_features("whenever you have time").urgency_count == 0


class TestF9InstructionDensity:
    def test_density_is_verbs_over_words(self) -> None:
        features = extract_features("write and test the function")
        assert features.instruction_verb_density == pytest.approx(2 / 5)

    def test_no_verbs_is_zero(self) -> None:
        assert extract_features("the sunset was beautiful").instruction_verb_density == 0.0

    def test_empty_prompt_is_zero(self) -> None:
        assert extract_features("").instruction_verb_density == 0.0


class TestEdgeCases:
    def test_empty_string_produces_valid_vector(self) -> None:
        features = extract_features("")
        assert features.token_count == 0
        assert features.code_block_ratio == 0.0
        assert features.is_question is False
        assert features.is_instruction is False

    def test_only_question_mark(self) -> None:
        features = extract_features("?")
        assert features.is_question is True

    def test_only_code_block(self) -> None:
        features = extract_features("```\ndef f(): pass\n```")
        assert features.code_block_ratio > 0.5
        assert features.is_instruction is False

    def test_mixed_language_prompt(self) -> None:
        features = extract_features("translate こんにちは to English please")
        assert features.token_count > 0

    def test_very_long_prompt(self) -> None:
        features = extract_features("explain this concept " * 2000)
        assert features.token_count > 4000

    def test_unicode_prompt(self) -> None:
        features = extract_features("décor 🚀 naïve résumé")
        assert features.token_count > 0

    def test_vector_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        features = extract_features("hello")
        with pytest.raises(FrozenInstanceError):
            features.token_count = 5  # type: ignore[misc]
