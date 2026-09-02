"""Feature extraction for the heuristic classifier (spec 05, F1-F9).

All patterns are precompiled at import time and all matching is local —
no network calls and no disk I/O beyond tiktoken's one-time encoder load.
The public entry point is :func:`extract_features`.
"""

from __future__ import annotations

import re

import tiktoken

from smartroute.classifier.domain_keywords import best_domain_match
from smartroute.types import DomainHint, FeatureVector

_ENCODERS: dict[str, tiktoken.Encoding] = {}
DEFAULT_ENCODER_MODEL = "gpt-4o-mini"


def get_encoder(model_name: str = DEFAULT_ENCODER_MODEL) -> tiktoken.Encoding:
    """Return a cached tiktoken encoder for the given model (spec 13).

    Args:
        model_name: Model to load the encoding for; defaults to gpt-4o-mini.

    Returns:
        The tiktoken encoding, built once per model name per process.

    Raises:
        tiktoken errors propagate to the caller, which implements fail-open.
    """
    if model_name not in _ENCODERS:
        _ENCODERS[model_name] = tiktoken.encoding_for_model(model_name)
    return _ENCODERS[model_name]


# ----------------------------------------------------------------------
# F2: code block ratio
# ----------------------------------------------------------------------

_FENCED_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)

# ----------------------------------------------------------------------
# F3: question vs instruction
# ----------------------------------------------------------------------

_WH_WORDS = ("what", "why", "how", "when", "where", "who", "which")
_INSTRUCTION_VERBS_F3: frozenset[str] = frozenset(
    {
        "create",
        "build",
        "write",
        "refactor",
        "fix",
        "implement",
        "add",
        "remove",
        "update",
        "delete",
    }
)
_INSTRUCTION_VERB_PATTERN_F3 = re.compile(
    r"\b(" + "|".join(sorted(_INSTRUCTION_VERBS_F3)) + r")\b", re.IGNORECASE
)

# ----------------------------------------------------------------------
# F4: multi-step markers
# ----------------------------------------------------------------------

_MULTI_STEP_PATTERN = re.compile(
    r"\b(then|after that|next|step|first|second|third|finally|subsequently"
    r"|followed by)\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# F5: ambiguity / hedge words ("either...or" is a literal substring)
# ----------------------------------------------------------------------

_HEDGE_WORDS = (
    "maybe",
    "perhaps",
    "possibly",
    "might",
    "could",
    "not sure",
    "uncertain",
    "unclear",
    "vague",
    "or something",
    "I think",
    "probably",
    "sort of",
    "kind of",
)
_HEDGE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _HEDGE_WORDS) + r")\b",
    re.IGNORECASE,
)
_HEDGE_LITERAL = "either...or"

# ----------------------------------------------------------------------
# F7: file path references (unique matches across all patterns)
# ----------------------------------------------------------------------

_CODE_EXTENSIONS = (
    "py",
    "js",
    "ts",
    "tsx",
    "jsx",
    "go",
    "rs",
    "java",
    "kt",
    "swift",
    "cpp",
    "c",
    "h",
    "rb",
    "php",
    "md",
    "json",
    "yaml",
    "yml",
    "toml",
    "xml",
    "sql",
    "sh",
    "bat",
    "ps1",
)
_EXT_FILE_PATTERN = re.compile(
    r"\b[\w-]+(?:\." + "|".join(_CODE_EXTENSIONS) + r")\b", re.IGNORECASE
)
_WINDOWS_PATH_PATTERN = re.compile(r"[A-Za-z]:\\(?:[\w.-]+\\)*[\w.-]+")
_POSIX_PATH_PATTERN = re.compile(r"(?:/[\w.-]+){2,}")
_KNOWN_DIR_PATTERN = re.compile(r"\b(?:src|lib|app|components|models|tests|docs)/", re.IGNORECASE)

# ----------------------------------------------------------------------
# F8: urgency keywords
# ----------------------------------------------------------------------

_URGENCY_PATTERN = re.compile(
    r"\b(urgent|ASAP|immediately|critical|blocking|broken|down|outage"
    r"|emergency|hotfix|production|prod|live|customer waiting|deadline"
    r"|due today|due tomorrow|EOD|end of day|priority|P0|P1|severity"
    r"|high severity|regression|rollback|revert)\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# F9: instruction verb density (deduplicated imperative verb list)
# ----------------------------------------------------------------------

_INSTRUCTION_VERBS_F9: frozenset[str] = frozenset(
    [
        "create",
        "build",
        "write",
        "generate",
        "produce",
        "make",
        "construct",
        "develop",
        "design",
        "implement",
        "code",
        "program",
        "script",
        "automate",
        "configure",
        "set",
        "up",
        "deploy",
        "publish",
        "release",
        "fix",
        "debug",
        "refactor",
        "rewrite",
        "optimize",
        "improve",
        "enhance",
        "upgrade",
        "update",
        "modify",
        "change",
        "adjust",
        "tweak",
        "edit",
        "correct",
        "resolve",
        "solve",
        "handle",
        "manage",
        "process",
        "parse",
        "validate",
        "verify",
        "check",
        "test",
        "run",
        "execute",
        "add",
        "remove",
        "delete",
        "insert",
        "merge",
        "split",
        "join",
        "extract",
        "filter",
        "sort",
        "search",
        "find",
        "organize",
        "format",
        "convert",
        "transform",
        "translate",
        "explain",
        "describe",
        "document",
        "log",
        "track",
        "analyze",
        "evaluate",
        "compare",
        "calculate",
        "estimate",
        "predict",
        "simulate",
        "model",
        "scaffold",
        "import",
        "export",
        "define",
        "declare",
        "initialize",
        "allocate",
        "read",
        "save",
        "load",
        "fetch",
        "send",
        "upload",
        "download",
        "sync",
        "backup",
        "migrate",
        "install",
        "package",
        "encrypt",
        "decrypt",
        "hash",
        "encode",
        "decode",
        "serialize",
        "compile",
        "lint",
        "prettify",
    ]
)
_INSTRUCTION_VERB_PATTERN_F9 = re.compile(
    r"\b(" + "|".join(sorted(_INSTRUCTION_VERBS_F9)) + r")\b", re.IGNORECASE
)


def _count_tokens(encoder: tiktoken.Encoding, prompt: str) -> int:
    """Count tiktoken tokens in the prompt (F1)."""
    return len(encoder.encode(prompt))


def _code_block_ratio(encoder: tiktoken.Encoding, prompt: str) -> float:
    """Fraction of tokens inside fenced code blocks (F2)."""
    total = len(encoder.encode(prompt))
    if total == 0:
        return 0.0
    inside = sum(len(encoder.encode(block)) for block in _FENCED_BLOCK.findall(prompt))
    return min(inside / total, 1.0)


def _is_question(prompt: str) -> bool:
    """True when the prompt ends with '?' or starts with a wh-word (F3)."""
    stripped = prompt.strip()
    if stripped.endswith("?"):
        return True
    first = stripped.lower().split(" ")[0] if stripped else ""
    return first.rstrip(",") in _WH_WORDS


def _instruction_matches(prompt: str) -> list[str]:
    """Imperative verb occurrences for F3 (small spec verb list)."""
    return _INSTRUCTION_VERB_PATTERN_F3.findall(prompt)


def _multi_step_count(prompt: str) -> int:
    """Count transition/sequence markers (F4)."""
    return len(_MULTI_STEP_PATTERN.findall(prompt))


def _ambiguity_score(prompt: str, total_words: int) -> float:
    """Hedge-word occurrences divided by total words (F5)."""
    if total_words == 0:
        return 0.0
    hedge_count = len(_HEDGE_PATTERN.findall(prompt))
    hedge_count += prompt.count(_HEDGE_LITERAL)
    return hedge_count / total_words


def _file_path_count(prompt: str) -> int:
    """Count unique file path references across all patterns (F7)."""
    references: set[str] = set()
    for pattern in (
        _EXT_FILE_PATTERN,
        _WINDOWS_PATH_PATTERN,
        _POSIX_PATH_PATTERN,
        _KNOWN_DIR_PATTERN,
    ):
        references.update(match.group(0) for match in pattern.finditer(prompt))
    return len(references)


def _urgency_count(prompt: str) -> int:
    """Count urgency keyword occurrences (F8; explainability only)."""
    return len(_URGENCY_PATTERN.findall(prompt))


def _instruction_verb_density(prompt: str, total_words: int) -> float:
    """Imperative verb occurrences divided by total words (F9)."""
    if total_words == 0:
        return 0.0
    return len(_INSTRUCTION_VERB_PATTERN_F9.findall(prompt)) / total_words


def extract_features(prompt: str) -> FeatureVector:
    """Extract the full 9-feature vector from a prompt (spec 05).

    Args:
        prompt: The raw prompt string.

    Returns:
        A frozen FeatureVector with every feature populated.

    Raises:
        Exception: Any extractor failure (e.g. tiktoken unavailable)
            propagates; ``smartroute.classifier.classifier`` implements the
            fail-open contract on top of this function.
    """
    encoder = get_encoder()
    total_words = len(prompt.split())
    token_count = _count_tokens(encoder, prompt)
    instruction_matches = _instruction_matches(prompt)
    domain_hint: DomainHint = best_domain_match(prompt, total_words)
    return FeatureVector(
        token_count=token_count,
        code_block_ratio=_code_block_ratio(encoder, prompt),
        is_question=_is_question(prompt),
        is_instruction=bool(instruction_matches),
        instruction_verb_count=len(instruction_matches),
        multi_step_count=_multi_step_count(prompt),
        ambiguity_score=_ambiguity_score(prompt, total_words),
        domain_hint=domain_hint,
        file_path_count=_file_path_count(prompt),
        urgency_count=_urgency_count(prompt),
        instruction_verb_density=_instruction_verb_density(prompt, total_words),
    )
