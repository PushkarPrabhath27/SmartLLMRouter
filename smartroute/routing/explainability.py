"""Explainability string formatting helpers for routing decisions (spec 06).

Provides deterministic human-readable explanations for all routing scenarios:
programmatic override hooks, YAML override rules, adaptive reputation bumps,
default tier routing, and provider fallback events.
"""

from __future__ import annotations


def format_override_hook_reason(provider: str) -> str:
    """Format explanation for programmatic hook override (spec 06).

    Args:
        provider: Provider key forced by the hook (e.g., 'anthropic').

    Returns:
        Human-readable reason string.
    """
    return f"Programmatic override: forced to {provider}"


def format_yaml_exact_reason(match: str, provider: str) -> str:
    """Format explanation for YAML exact string match rule (spec 06).

    Args:
        match: The exact match string.
        provider: Provider key routed to.

    Returns:
        Human-readable reason string.
    """
    return f"YAML rule matched: exact string '{match}' -> {provider}"


def format_yaml_contains_reason(match: str, provider: str) -> str:
    """Format explanation for YAML substring contains rule (spec 06).

    Args:
        match: The substring pattern.
        provider: Provider key routed to.

    Returns:
        Human-readable reason string.
    """
    return f"YAML rule matched: prompt contains '{match}' -> {provider}"


def format_yaml_regex_reason(pattern: str, provider: str) -> str:
    """Format explanation for YAML regex pattern rule (spec 06).

    Args:
        pattern: The regex pattern string.
        provider: Provider key routed to.

    Returns:
        Human-readable reason string.
    """
    return f"YAML rule matched: regex /{pattern}/ -> {provider}"


def format_yaml_path_reason(pattern: str, provider: str) -> str:
    """Format explanation for YAML path pattern rule (spec 06).

    Args:
        pattern: The file path pattern.
        provider: Provider key routed to.

    Returns:
        Human-readable reason string.
    """
    return f"YAML rule matched: path '{pattern}' -> {provider}"


def format_yaml_rule_reason(match_type: str, match: str, provider: str) -> str:
    """Format explanation for any YAML override rule (spec 06).

    Args:
        match_type: One of 'exact', 'contains', 'regex', 'path'.
        match: The pattern or string matched.
        provider: Provider key routed to.

    Returns:
        Human-readable reason string.
    """
    if match_type == "exact":
        return format_yaml_exact_reason(match, provider)
    if match_type == "contains":
        return format_yaml_contains_reason(match, provider)
    if match_type == "regex":
        return format_yaml_regex_reason(match, provider)
    if match_type == "path":
        return format_yaml_path_reason(match, provider)
    return f"YAML rule matched: {match_type} '{match}' -> {provider}"


def format_reputation_bump_reason(score: float, old_tier: str, new_tier: str) -> str:
    """Format explanation for adaptive reputation tier bump (spec 06).

    Args:
        score: Current EMA reputation score that triggered the bump.
        old_tier: Previous complexity tier name.
        new_tier: Escalated complexity tier name.

    Returns:
        Human-readable reason string.
    """
    return f"Adaptive: reputation {score:.2f} below threshold, bumped from {old_tier} to {new_tier}"


def format_default_routing_reason(
    task_type: str, bucket: str, complexity: float, provider: str
) -> str:
    """Format explanation for default complexity tier routing (spec 06).

    Args:
        task_type: Classified task type name (e.g., 'code').
        bucket: Complexity bucket name ('low', 'medium', or 'high').
        complexity: Computed complexity score in [0.0, 1.0].
        provider: Provider key mapped to this bucket.

    Returns:
        Human-readable reason string.
    """
    return (
        f"Default routing: {task_type} task, {bucket} complexity ({complexity:.2f}) -> {provider}"
    )


def format_fallback_reason(primary_provider: str, fallback_provider: str, error_msg: str) -> str:
    """Format explanation for provider fallback events (spec 06).

    Args:
        primary_provider: Original provider key that failed.
        fallback_provider: Fallback provider key used instead.
        error_msg: Error message or reason for the primary failure.

    Returns:
        Human-readable reason string.
    """
    return (
        f"Primary provider failed, fallback to {fallback_provider} "
        f"after {primary_provider} error: {error_msg}"
    )
