"""Domain keyword dictionaries and matching (spec 05, feature F6).

Each task type has a keyword dictionary. Matching counts keyword
*occurrences* (case-insensitive, word-bounded) per domain; the domain with
the highest match ratio (occurrences / total words) wins, provided the
ratio exceeds 0.05 — otherwise the prompt defaults to GENERAL.

Patterns are compiled once at import so classification stays well under
the 10ms performance budget.
"""

from __future__ import annotations

import re

from smartroute.types import DomainHint, TaskType

CODE_KEYWORDS: tuple[str, ...] = (
    "function",
    "class",
    "variable",
    "debug",
    "refactor",
    "compile",
    "runtime",
    "syntax",
    "error",
    "stack trace",
    "import",
    "module",
    "package",
    "API",
    "endpoint",
    "database",
    "query",
    "SQL",
    "regex",
    "algorithm",
    "data structure",
    "OOP",
    "functional",
    "async",
    "await",
    "decorator",
    "lambda",
    "list comprehension",
    "git",
    "commit",
    "branch",
    "merge",
    "pull request",
    "unit test",
    "CI/CD",
    "Docker",
    "Kubernetes",
)

CREATIVE_KEYWORDS: tuple[str, ...] = (
    "story",
    "poem",
    "essay",
    "blog",
    "write",
    "draft",
    "creative",
    "imaginative",
    "fiction",
    "narrative",
    "character",
    "plot",
    "setting",
    "dialogue",
    "tone",
    "style",
    "metaphor",
    "simile",
    "analogy",
    "brainstorm",
    "ideate",
    "concept",
    "slogan",
    "tagline",
    "brand",
    "marketing copy",
    "ad",
    "campaign",
    "social media",
    "tweet",
    "thread",
    "email",
    "newsletter",
)

REASONING_KEYWORDS: tuple[str, ...] = (
    "analyze",
    "compare",
    "contrast",
    "evaluate",
    "assess",
    "pros and cons",
    "trade-off",
    "decision",
    "strategy",
    "plan",
    "framework",
    "model",
    "theory",
    "hypothesis",
    "evidence",
    "proof",
    "logical",
    "rational",
    "deduce",
    "infer",
    "conclude",
    "solve",
    "puzzle",
    "riddle",
    "math",
    "equation",
    "calculate",
    "statistics",
    "probability",
    "optimize",
    "maximize",
    "minimize",
)

SUMMARIZATION_KEYWORDS: tuple[str, ...] = (
    "summarize",
    "summary",
    "TL;DR",
    "key points",
    "main ideas",
    "condense",
    "shorten",
    "brief",
    "overview",
    "recap",
    "digest",
    "abstract",
    "extract",
    "highlight",
    "bullet points",
    "executive summary",
)

TRANSLATION_KEYWORDS: tuple[str, ...] = (
    "translate",
    "translation",
    "language",
    "English",
    "Spanish",
    "French",
    "German",
    "Chinese",
    "Japanese",
    "Korean",
    "Arabic",
    "Hindi",
    "Portuguese",
    "Russian",
    "Italian",
    "Dutch",
    "Swedish",
    "Polish",
    "Turkish",
    "Vietnamese",
    "Thai",
    "Indonesian",
    "Malay",
    "Filipino",
    "Hebrew",
    "Greek",
    "Czech",
    "Romanian",
    "Hungarian",
    "Finnish",
    "Norwegian",
    "Danish",
)

DOMAIN_KEYWORDS: dict[TaskType, tuple[str, ...]] = {
    TaskType.CODE: CODE_KEYWORDS,
    TaskType.CREATIVE: CREATIVE_KEYWORDS,
    TaskType.REASONING: REASONING_KEYWORDS,
    TaskType.SUMMARIZATION: SUMMARIZATION_KEYWORDS,
    TaskType.TRANSLATION: TRANSLATION_KEYWORDS,
}

# One alternation regex per domain, precompiled: word-bounded, escaped,
# case-insensitive occurrences of any keyword in the dictionary.
DOMAIN_PATTERNS: dict[TaskType, re.Pattern[str]] = {
    task_type: re.compile(
        "|".join(rf"\b{re.escape(keyword)}\b" for keyword in keywords),
        re.IGNORECASE,
    )
    for task_type, keywords in DOMAIN_KEYWORDS.items()
}

# Spec 05 F6: a domain hint only counts when its match ratio exceeds this.
DOMAIN_RATIO_THRESHOLD = 0.05


def _count_matches(pattern: re.Pattern[str], prompt: str) -> int:
    """Count keyword occurrences for one domain (case-insensitive)."""
    return len(pattern.findall(prompt))


def best_domain_match(prompt: str, total_words: int) -> DomainHint:
    """Find the best-matching domain for a prompt (spec 05 F6).

    Args:
        prompt: The raw prompt string.
        total_words: Total word count of the prompt (shared denominator for
            every domain's ratio).

    Returns:
        A DomainHint whose ``domain`` is the winning TaskType *value* string
        (e.g. ``"code"``), or ``""`` when no domain's ratio exceeds 0.05
        (including zero-word prompts). ``match_count`` is the occurrence
        count for the winning domain; ``match_ratio`` is
        ``match_count / total_words`` (0.0 for empty prompts).
    """
    best_type: TaskType | None = None
    best_count = 0
    best_ratio = 0.0
    for task_type, pattern in DOMAIN_PATTERNS.items():
        count = _count_matches(pattern, prompt)
        if total_words == 0:
            continue
        ratio = count / total_words
        if ratio > best_ratio or (
            ratio == best_ratio and best_type is not None and count > best_count
        ):
            best_type = task_type
            best_count = count
            best_ratio = ratio
    if best_type is None or best_ratio <= DOMAIN_RATIO_THRESHOLD:
        return DomainHint(domain="", match_count=0, match_ratio=0.0)
    return DomainHint(domain=best_type.value, match_count=best_count, match_ratio=best_ratio)
