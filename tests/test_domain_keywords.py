"""Unit tests for smartroute.classifier.domain_keywords (Phase 2, module 1)."""

import pytest

from smartroute.classifier.domain_keywords import (
    CODE_KEYWORDS,
    CREATIVE_KEYWORDS,
    DOMAIN_KEYWORDS,
    DOMAIN_PATTERNS,
    REASONING_KEYWORDS,
    SUMMARIZATION_KEYWORDS,
    TRANSLATION_KEYWORDS,
    best_domain_match,
)
from smartroute.types import TaskType


class TestDictionaries:
    @pytest.mark.parametrize(
        ("keywords", "expected_count"),
        [
            (CODE_KEYWORDS, 37),
            (CREATIVE_KEYWORDS, 33),
            (REASONING_KEYWORDS, 32),
            (SUMMARIZATION_KEYWORDS, 16),
            (TRANSLATION_KEYWORDS, 32),
        ],
    )
    def test_dictionary_sizes_match_spec(
        self, keywords: tuple[str, ...], expected_count: int
    ) -> None:
        assert len(keywords) == expected_count

    def test_all_five_task_types_have_dictionaries(self) -> None:
        assert set(DOMAIN_KEYWORDS) == {
            TaskType.CODE,
            TaskType.CREATIVE,
            TaskType.REASONING,
            TaskType.SUMMARIZATION,
            TaskType.TRANSLATION,
        }

    def test_punctuation_keywords_present_verbatim(self) -> None:
        assert "TL;DR" in SUMMARIZATION_KEYWORDS
        assert "CI/CD" in CODE_KEYWORDS
        assert "stack trace" in CODE_KEYWORDS
        assert "pros and cons" in REASONING_KEYWORDS


class TestBestDomainMatch:
    def test_code_prompt_wins(self) -> None:
        hint = best_domain_match("debug this function and fix the syntax error", 9)
        assert hint.domain == "code"
        assert hint.match_count == 4  # debug, function, syntax, error
        assert hint.match_ratio == pytest.approx(4 / 9)

    def test_creative_prompt_wins(self) -> None:
        hint = best_domain_match("write a story with dialogue and a plot twist", 9)
        assert hint.domain == "creative"

    def test_reasoning_prompt_wins(self) -> None:
        hint = best_domain_match("analyze the evidence and evaluate the hypothesis logically", 9)
        assert hint.domain == "reasoning"

    def test_summarization_prompt_wins(self) -> None:
        hint = best_domain_match("summarize the key points and main ideas briefly", 9)
        assert hint.domain == "summarization"

    def test_translation_prompt_wins(self) -> None:
        hint = best_domain_match("translate this language text from English to Japanese", 9)
        assert hint.domain == "translation"

    def test_case_insensitive_matching(self) -> None:
        hint = best_domain_match("DEBUG the FUNCTION", 3)
        assert hint.domain == "code"
        assert hint.match_count == 2

    def test_repeated_keyword_counts_each_occurrence(self) -> None:
        hint = best_domain_match("debug debug debug", 3)
        assert hint.match_count == 3

    def test_word_boundaries_prevent_substring_matches(self) -> None:
        # "classical" must not match "class"; "nextjs" style suffixes neither.
        hint = best_domain_match("classical music appreciation", 3)
        assert hint.domain == ""

    def test_ratio_below_threshold_returns_empty(self) -> None:
        # 1 match ("write") in 23 words = 0.043 <= 0.05 -> GENERAL
        prompt = (
            "write something nice about a very long and detailed topic "
            "covering many aspects and considerations of the modern world "
            "for readers everywhere today"
        )
        assert len(prompt.split()) == 23
        hint = best_domain_match(prompt, 23)
        assert hint.domain == ""
        assert hint.match_ratio == 0.0

    def test_ratio_just_above_threshold_wins(self) -> None:
        # 2 matches in 20 words = 0.10 > 0.05
        prompt = "write a story about " + "things " * 15
        hint = best_domain_match(prompt.strip(), 19)
        assert hint.domain == "creative"

    def test_empty_prompt_returns_empty_hint(self) -> None:
        hint = best_domain_match("", 0)
        assert hint.domain == ""
        assert hint.match_count == 0
        assert hint.match_ratio == 0.0

    def test_highest_ratio_wins_over_other_domains(self) -> None:
        # code: 2/8 = 0.25, translation: 1/8 = 0.125 -> code wins
        hint = best_domain_match("refactor the function for language support", 7)
        assert hint.domain == "code"

    def test_tie_broken_by_match_count(self) -> None:
        # "write" is creative; "summarize" and "summary" are summarization.
        # Constructed so both domains hit the same ratio, summarization wins
        # on count via two distinct keywords vs one repeated creative hit.
        prompt = "write the summary and overview"  # creative 1/5, summ 2/5
        hint = best_domain_match(prompt, 5)
        assert hint.domain == "summarization"


class TestPatternCompilation:
    def test_every_keyword_compiles_into_its_domain_pattern(self) -> None:
        for task_type, keywords in DOMAIN_KEYWORDS.items():
            pattern = DOMAIN_PATTERNS[task_type]
            for keyword in keywords:
                assert pattern.search(keyword) is not None, (task_type, keyword)

    def test_patterns_are_case_insensitive(self) -> None:
        assert DOMAIN_PATTERNS[TaskType.CODE].search("DOCKER") is not None
