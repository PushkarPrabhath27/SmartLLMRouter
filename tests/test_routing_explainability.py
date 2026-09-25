"""Unit tests for smartroute.routing.explainability (Phase 3 Module 2)."""

from __future__ import annotations

from smartroute.routing.explainability import (
    format_default_routing_reason,
    format_fallback_reason,
    format_override_hook_reason,
    format_reputation_bump_reason,
    format_yaml_contains_reason,
    format_yaml_exact_reason,
    format_yaml_path_reason,
    format_yaml_regex_reason,
    format_yaml_rule_reason,
)


class TestExplainabilityFormatting:
    def test_override_hook_reason(self) -> None:
        reason = format_override_hook_reason("anthropic")
        assert reason == "Programmatic override: forced to anthropic"

    def test_yaml_exact_reason(self) -> None:
        reason = format_yaml_exact_reason("debug this", "anthropic")
        assert reason == "YAML rule matched: exact string 'debug this' -> anthropic"

    def test_yaml_contains_reason(self) -> None:
        reason = format_yaml_contains_reason("contract", "anthropic")
        assert reason == "YAML rule matched: prompt contains 'contract' -> anthropic"

    def test_yaml_regex_reason(self) -> None:
        reason = format_yaml_regex_reason("legal", "anthropic")
        assert reason == "YAML rule matched: regex /legal/ -> anthropic"

    def test_yaml_path_reason(self) -> None:
        reason = format_yaml_path_reason("src/*.py", "anthropic")
        assert reason == "YAML rule matched: path 'src/*.py' -> anthropic"

    def test_yaml_rule_reason_dispatcher(self) -> None:
        assert (
            format_yaml_rule_reason("exact", "hello", "openai")
            == "YAML rule matched: exact string 'hello' -> openai"
        )
        assert (
            format_yaml_rule_reason("contains", "foo", "groq")
            == "YAML rule matched: prompt contains 'foo' -> groq"
        )
        assert (
            format_yaml_rule_reason("regex", r"\d+", "anthropic")
            == r"YAML rule matched: regex /\d+/ -> anthropic"
        )
        assert (
            format_yaml_rule_reason("path", "*.ts", "openai")
            == "YAML rule matched: path '*.ts' -> openai"
        )
        assert (
            format_yaml_rule_reason("custom", "val", "groq")
            == "YAML rule matched: custom 'val' -> groq"
        )

    def test_reputation_bump_reason_spec_example(self) -> None:
        reason = format_reputation_bump_reason(0.25, "low", "medium")
        assert reason == "Adaptive: reputation 0.25 below threshold, bumped from low to medium"

    def test_reputation_bump_float_formatting(self) -> None:
        reason = format_reputation_bump_reason(0.2500001, "low", "medium")
        assert reason == "Adaptive: reputation 0.25 below threshold, bumped from low to medium"

    def test_default_routing_reason_spec_example(self) -> None:
        reason = format_default_routing_reason("code", "medium", 0.45, "openai")
        assert reason == "Default routing: code task, medium complexity (0.45) -> openai"

    def test_default_routing_float_formatting(self) -> None:
        reason = format_default_routing_reason("general", "low", 0.123456, "groq")
        assert reason == "Default routing: general task, low complexity (0.12) -> groq"

    def test_fallback_reason_spec_example(self) -> None:
        reason = format_fallback_reason("openai", "groq", "Timeout")
        assert reason == "Primary provider failed, fallback to groq after openai error: Timeout"

    def test_fallback_reason_long_message(self) -> None:
        long_err = "Connection reset by peer while waiting for 10.0s response from remote server"
        reason = format_fallback_reason("anthropic", "openai", long_err)
        assert (
            reason
            == f"Primary provider failed, fallback to openai after anthropic error: {long_err}"
        )
