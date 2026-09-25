"""Unit tests for implicit feedback signal detectors (Phase 5)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from smartroute.signals.detectors import (
    ACCEPTANCE_VALUE,
    EXPLICIT_CORRECTION_VALUE,
    HARD_REGEN_VALUE,
    SOFT_REGEN_VALUE,
    HistoryItem,
    detect_acceptance,
    detect_explicit_correction,
    detect_hard_regen,
    detect_soft_regen,
    resolve_strongest_signal,
)
from smartroute.types import Signal


class TestHardRegenDetector:
    def test_positive_cases(self) -> None:
        now = datetime.now(timezone.utc)
        history = [
            HistoryItem(
                decision_id="d1",
                prompt="write quicksort in python",
                timestamp=now - timedelta(seconds=10),
            ),
            HistoryItem(
                decision_id="d2",
                prompt="explain quantum computing",
                timestamp=now - timedelta(seconds=5),
            ),
        ]
        # Exact match within 30s
        signal = detect_hard_regen("explain quantum computing", history, now=now)
        assert signal is not None
        assert signal.signal_type == "hard_regen"
        assert signal.value == HARD_REGEN_VALUE
        assert signal.decision_id == "d2"

    @pytest.mark.parametrize(
        "prompt_text",
        [
            "fix bug in auth",
            "SELECT * FROM users",
            "what is the capital of France?",
            "how to use asyncio.gather?",
            "docker run -p 8080:80 nginx",
            "regex for email address",
            "git cherry-pick commit_sha",
            "def factorial(n):",
            "translate to Spanish: hello",
            "summarize this article",
        ],
    )
    def test_ten_positive_matches(self, prompt_text: str) -> None:
        now = datetime.now(timezone.utc)
        history = [
            HistoryItem(
                decision_id="dec_pos", prompt=prompt_text, timestamp=now - timedelta(seconds=15)
            )
        ]
        signal = detect_hard_regen(prompt_text, history, now=now)
        assert signal is not None
        assert signal.decision_id == "dec_pos"
        assert signal.value == -0.3

    @pytest.mark.parametrize(
        "prompt_text",
        [
            "completely different query",
            "write quicksort in java",
            "SELECT id FROM users",
            "what is the capital of Germany?",
            "how to use asyncio.create_task?",
            "docker-compose up -d",
            "regex for phone number",
            "git merge main",
            "def fibonacci(n):",
            "summarize this book",
        ],
    )
    def test_ten_negative_matches(self, prompt_text: str) -> None:
        now = datetime.now(timezone.utc)
        history = [
            HistoryItem(
                decision_id="d0",
                prompt="initial unrelated prompt",
                timestamp=now - timedelta(seconds=5),
            )
        ]
        signal = detect_hard_regen(prompt_text, history, now=now)
        assert signal is None

    def test_time_window_boundary_29s_vs_31s(self) -> None:
        now = datetime.now(timezone.utc)
        # 29s: within 30s window -> matches
        history_29s = [
            HistoryItem(decision_id="d29", prompt="hello", timestamp=now - timedelta(seconds=29))
        ]
        assert detect_hard_regen("hello", history_29s, now=now) is not None

        # 31s: expired -> None
        history_31s = [
            HistoryItem(decision_id="d31", prompt="hello", timestamp=now - timedelta(seconds=31))
        ]
        assert detect_hard_regen("hello", history_31s, now=now) is None


class TestSoftRegenDetector:
    def test_positive_overlap(self) -> None:
        now = datetime.now(timezone.utc)
        p1 = "write a clean python function to parse json safely"
        p2 = "write a clean python function to parse json"
        history = [
            HistoryItem(decision_id="d_soft", prompt=p1, timestamp=now - timedelta(seconds=20))
        ]
        signal = detect_soft_regen(p2, history, now=now)
        assert signal is not None
        assert signal.signal_type == "soft_regen"
        assert signal.value == SOFT_REGEN_VALUE
        assert signal.decision_id == "d_soft"

    @pytest.mark.parametrize(
        ("p1", "p2"),
        [
            (
                "create fastapi endpoint with pydantic model",
                "create a fastapi endpoint with pydantic model",
            ),
            ("implement binary search tree in python", "implement a binary search tree in python"),
            ("how to dockerize a nodejs web application", "how to dockerize a nodejs web app"),
            (
                "explain transformer self attention mechanism clearly",
                "explain transformer self attention mechanism very clearly",
            ),
            (
                "write postgresql query to calculate monthly revenue",
                "write a postgresql query to calculate monthly revenue",
            ),
            (
                "configure nginx reverse proxy for https ssl",
                "configure nginx reverse proxy for https",
            ),
            (
                "refactor react component using custom hooks",
                "refactor this react component using custom hooks",
            ),
            (
                "best way to handle authentication in nextjs",
                "best way to handle user authentication in nextjs",
            ),
            (
                "calculate standard deviation of numpy array",
                "calculate the standard deviation of numpy array",
            ),
            ("parse csv file line by line in golang", "parse a csv file line by line in golang"),
        ],
    )
    def test_ten_positive_soft_regen_cases(self, p1: str, p2: str) -> None:
        now = datetime.now(timezone.utc)
        history = [
            HistoryItem(decision_id="d_pair", prompt=p1, timestamp=now - timedelta(seconds=30))
        ]
        signal = detect_soft_regen(p2, history, now=now)
        assert signal is not None
        assert signal.value == -0.1

    @pytest.mark.parametrize(
        ("p1", "p2"),
        [
            ("write a python script", "show me a recipe for chocolate cake"),
            ("fix authentication bug in jwt parser", "deploy kubernetes cluster to aws eks"),
            ("how to center a div in css", "what is the speed of light in vacuum"),
            (
                "unit test for database transaction rollback",
                "generate marketing copy for running shoes",
            ),
            ("install tailwind css in vite react project", "summarize the quarterly earnings call"),
            ("optimize sql query with composite index", "translate this greeting into japanese"),
            (
                "setup github actions ci workflow for pytest",
                "analyze the pros and cons of remote work",
            ),
            ("solve two sum problem with hash map", "write a haiku about autumn leaves"),
            ("implement rate limiter with redis token bucket", "recommend tourist spots in kyoto"),
            ("convert markdown document to html string", "explain quantum entanglement to a child"),
        ],
    )
    def test_ten_negative_soft_regen_cases(self, p1: str, p2: str) -> None:
        now = datetime.now(timezone.utc)
        history = [
            HistoryItem(decision_id="d_pair", prompt=p1, timestamp=now - timedelta(seconds=10))
        ]
        signal = detect_soft_regen(p2, history, now=now)
        assert signal is None

    def test_time_window_boundary_59s_vs_61s(self) -> None:
        now = datetime.now(timezone.utc)
        p1 = "write quicksort algorithm in python with comments"
        p2 = "write quicksort algorithm in python with detailed comments"

        history_59s = [
            HistoryItem(decision_id="d59", prompt=p1, timestamp=now - timedelta(seconds=59))
        ]
        assert detect_soft_regen(p2, history_59s, now=now) is not None

        history_61s = [
            HistoryItem(decision_id="d61", prompt=p1, timestamp=now - timedelta(seconds=61))
        ]
        assert detect_soft_regen(p2, history_61s, now=now) is None


class TestExplicitCorrectionDetector:
    @pytest.mark.parametrize(
        "msg",
        [
            # English
            "no, that's wrong",
            "doesn't work at all",
            "not what I asked for",
            "redo this function please",
            "can you fix this error?",
            # Spanish
            "no, no funciona",
            "estas equivocado, hazlo de nuevo",
            # French
            "non, c'est faux",
            "ca ne marche pas",
            # German
            "nein, das ist falsch",
            "funktioniert nicht",
            # Chinese
            "不对，再试一次",
            "不是我想要的",
            # Japanese
            "違う、やり直し",
            "ダメです、修正して",
        ],
    )
    def test_multilingual_positive_corrections(self, msg: str) -> None:
        signal = detect_explicit_correction(msg, decision_id="dec_multilang")
        assert signal is not None
        assert signal.signal_type == "explicit_correction"
        assert signal.value == EXPLICIT_CORRECTION_VALUE
        assert signal.decision_id == "dec_multilang"

    @pytest.mark.parametrize(
        "msg",
        [
            "Thank you, that worked perfectly!",
            "Great, now let's write the tests for this.",
            "Can you explain line 15 in more detail?",
            "What if we use a set instead of a list?",
            "Looks awesome! How should we deploy it?",
            "Understood, thank you very much.",
            "Now let's add support for another format.",
            "Merci beaucoup, c'est parfait!",
            "Muchas gracias, funcionó bien.",
            "谢谢，很有帮助！",
        ],
    )
    def test_ten_negative_explicit_corrections(self, msg: str) -> None:
        signal = detect_explicit_correction(msg, decision_id="dec_neg")
        assert signal is None

    def test_message_length_cutoff_over_200_words_ignored(self) -> None:
        long_message = "no " + "word " * 210
        signal = detect_explicit_correction(long_message, decision_id="d_long")
        assert signal is None


class TestAcceptanceDetector:
    def test_detect_acceptance(self) -> None:
        signal = detect_acceptance("dec_accept")
        assert signal.signal_type == "acceptance"
        assert signal.value == ACCEPTANCE_VALUE
        assert signal.decision_id == "dec_accept"


class TestSignalPriorityResolution:
    def test_hard_regen_beats_all(self) -> None:
        s1 = Signal("acceptance", 0.05, "d1")
        s2 = Signal("soft_regen", -0.1, "d1")
        s3 = Signal("hard_regen", -0.3, "d1")
        s4 = Signal("explicit_correction", -0.2, "d1")

        best = resolve_strongest_signal([s1, s2, s3, s4])
        assert best is not None
        assert best.signal_type == "hard_regen"

    def test_explicit_correction_beats_soft_regen_and_acceptance(self) -> None:
        s1 = Signal("acceptance", 0.05, "d1")
        s2 = Signal("soft_regen", -0.1, "d1")
        s4 = Signal("explicit_correction", -0.2, "d1")

        best = resolve_strongest_signal([s1, s2, s4])
        assert best is not None
        assert best.signal_type == "explicit_correction"

    def test_empty_signals_list_returns_none(self) -> None:
        assert resolve_strongest_signal([]) is None


class TestDetectorEdgeCases:
    def test_detect_hard_regen_empty_history(self) -> None:
        assert detect_hard_regen("test", []) is None

    def test_detect_soft_regen_empty_history(self) -> None:
        assert detect_soft_regen("test", []) is None

    def test_detect_soft_regen_empty_prompt(self) -> None:
        now = datetime.now(timezone.utc)
        history = [HistoryItem("d1", "some prompt", now)]
        assert detect_soft_regen("", history, now=now) is None

    def test_detect_explicit_correction_empty_or_whitespace(self) -> None:
        assert detect_explicit_correction("", "d1") is None
        assert detect_explicit_correction("   \n\t  ", "d1") is None
