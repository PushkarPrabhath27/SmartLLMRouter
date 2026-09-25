"""Implicit feedback signal detectors (spec 07).

Detects user feedback signals from interaction patterns:
- hard_regen: exact same prompt repeated within 30 seconds (-0.3)
- soft_regen: >80% token overlap within 60 seconds (-0.1)
- explicit_correction: negative feedback phrase in short follow-up (-0.2)
- acceptance: normal conversation continuation or close (+0.05)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from smartroute.types import Signal

# Signal numerical impact values per spec 07
HARD_REGEN_VALUE = -0.3
SOFT_REGEN_VALUE = -0.1
EXPLICIT_CORRECTION_VALUE = -0.2
ACCEPTANCE_VALUE = 0.05

SIGNAL_VALUES: dict[str, float] = {
    "hard_regen": HARD_REGEN_VALUE,
    "explicit_correction": EXPLICIT_CORRECTION_VALUE,
    "soft_regen": SOFT_REGEN_VALUE,
    "acceptance": ACCEPTANCE_VALUE,
}

SIGNAL_PRIORITY: dict[str, int] = {
    "hard_regen": 4,
    "explicit_correction": 3,
    "soft_regen": 2,
    "acceptance": 1,
}

# ----------------------------------------------------------------------
# Multilingual negative correction patterns (spec 07)
# ----------------------------------------------------------------------

# English
ENGLISH_PHRASES = [
    "no",
    "wrong",
    "incorrect",
    "bad",
    "terrible",
    "awful",
    "sucks",
    "not right",
    "doesn't work",
    "didn't work",
    "not working",
    "redo",
    "do it again",
    "try again",
    "fix this",
    "that's wrong",
    "you're wrong",
    "not what I asked",
    "missed the point",
    "off topic",
    "irrelevant",
    "useless",
    "garbage",
    "nonsense",
    "can you fix",
    "please fix",
    "needs fixing",
    "correction:",
    "actually,",
    "wait,",
    "hold on,",
    "scratch that",
]

# Spanish
SPANISH_PHRASES = [
    "no",
    "incorrecto",
    "malo",
    "horrible",
    "no funciona",
    "no sirve",
    "intenta de nuevo",
    "hazlo de nuevo",
    "corrige",
    "estas equivocado",
    "no es lo que pedi",
    "fuera de tema",
    "inutil",
    "basura",
]

# French
FRENCH_PHRASES = [
    "non",
    "faux",
    "incorrect",
    "mauvais",
    "nul",
    "ca ne marche pas",
    "refais",
    "corrige",
    "tu te trompes",
    "ce n'est pas ce que j'ai demande",
    "inutile",
    "a cote de la plaque",
]

# German
GERMAN_PHRASES = [
    "nein",
    "falsch",
    "inkorrekt",
    "schlecht",
    "funktioniert nicht",
    "mach nochmal",
    "korrigiere",
    "das ist falsch",
    "nicht was ich wollte",
    "nutzlos",
    "muell",
    "quatsch",
]

# Chinese (simplified)
CHINESE_PHRASES = [
    "不对",
    "错了",
    "不好",
    "不行",
    "没用",
    "垃圾",
    "重做",
    "再试一次",
    "修正",
    "你错了",
    "不是我想要的",
    "跑题了",
    "毫无意义",
]

# Japanese
JAPANESE_PHRASES = [
    "違う",
    "間違い",
    "悪い",
    "ダメ",
    "役に立たない",
    "ゴミ",
    "やり直し",
    "修正して",
    "違います",
    "求めたものではない",
]

# Build precompiled regex: word boundaries for Latin alphabets, direct for CJK
_LATIN_PHRASES = ENGLISH_PHRASES + SPANISH_PHRASES + FRENCH_PHRASES + GERMAN_PHRASES
_CJK_PHRASES = CHINESE_PHRASES + JAPANESE_PHRASES

_LATIN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(_LATIN_PHRASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_CJK_PATTERN = re.compile(
    "|".join(re.escape(p) for p in sorted(_CJK_PHRASES, key=len, reverse=True))
)


@dataclass(frozen=True)
class HistoryItem:
    """A past prompt submission for regeneration detection.

    Attributes:
        decision_id: UUID of the routing decision.
        prompt: Raw prompt text.
        timestamp: Time the prompt was submitted (UTC).
    """

    decision_id: str
    prompt: str
    timestamp: datetime


def _tokenize(text: str) -> set[str]:
    """Simple whitespace tokenization + lowercasing (spec 07)."""
    return {token.lower() for token in text.split()}


def detect_hard_regen(
    prompt: str,
    history: list[HistoryItem],
    now: datetime | None = None,
) -> Signal | None:
    """Detect if the exact same prompt was repeated within 30 seconds (spec 07).

    Args:
        prompt: The newly submitted prompt.
        history: List of recent prompt submissions.
        now: Optional current timestamp (defaults to UTC now).

    Returns:
        A hard_regen Signal (-0.3) or None.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Check last 5 decisions in reverse order
    for item in reversed(history[-5:]):
        item_time = (
            item.timestamp
            if item.timestamp.tzinfo is not None
            else item.timestamp.replace(tzinfo=timezone.utc)
        )
        if item.prompt == prompt and (now - item_time) < timedelta(seconds=30):
            return Signal(
                signal_type="hard_regen",
                value=HARD_REGEN_VALUE,
                decision_id=item.decision_id,
            )
    return None


def detect_soft_regen(
    prompt: str,
    history: list[HistoryItem],
    now: datetime | None = None,
    overlap_threshold: float = 0.8,
) -> Signal | None:
    """Detect if a substantially similar prompt (>80% overlap) was sent within 60s.

    Args:
        prompt: The newly submitted prompt.
        history: List of recent prompt submissions.
        now: Optional current timestamp (defaults to UTC now).
        overlap_threshold: Minimum Jaccard token overlap ratio (default 0.8).

    Returns:
        A soft_regen Signal (-0.1) or None.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    prompt_tokens = _tokenize(prompt)
    if not prompt_tokens:
        return None

    for item in reversed(history[-5:]):
        item_time = (
            item.timestamp
            if item.timestamp.tzinfo is not None
            else item.timestamp.replace(tzinfo=timezone.utc)
        )
        record_tokens = _tokenize(item.prompt)
        if not record_tokens:
            continue

        overlap = len(prompt_tokens & record_tokens) / max(len(prompt_tokens), len(record_tokens))
        if overlap > overlap_threshold and (now - item_time) < timedelta(seconds=60):
            return Signal(
                signal_type="soft_regen",
                value=SOFT_REGEN_VALUE,
                decision_id=item.decision_id,
            )
    return None


def detect_explicit_correction(next_message: str, decision_id: str) -> Signal | None:
    """Detect if a short follow-up message contains explicit negative feedback.

    Args:
        next_message: Follow-up message text.
        decision_id: Decision ID of the turn being evaluated.

    Returns:
        An explicit_correction Signal (-0.2) or None.
    """
    # False positive mitigation: only trigger if message is short (<200 tokens/words)
    tokens = next_message.split()
    if len(tokens) >= 200 or not tokens:
        return None

    msg = next_message.strip()
    if _LATIN_PATTERN.search(msg) or _CJK_PATTERN.search(msg):
        return Signal(
            signal_type="explicit_correction",
            value=EXPLICIT_CORRECTION_VALUE,
            decision_id=decision_id,
        )
    return None


def detect_acceptance(decision_id: str) -> Signal:
    """Generate an implicit acceptance signal (+0.05) for a decision.

    Args:
        decision_id: Decision ID of the accepted turn.

    Returns:
        An acceptance Signal (+0.05).
    """
    return Signal(
        signal_type="acceptance",
        value=ACCEPTANCE_VALUE,
        decision_id=decision_id,
    )


def resolve_strongest_signal(signals: list[Signal]) -> Signal | None:
    """Select the single strongest (most negative) signal from a list (spec 07).

    Priority: hard_regen (4) > explicit_correction (3) > soft_regen (2) > acceptance (1).

    Args:
        signals: List of detected candidate signals.

    Returns:
        The highest priority signal, or None if the list is empty.
    """
    if not signals:
        return None
    return max(signals, key=lambda s: SIGNAL_PRIORITY.get(s.signal_type, 0))
