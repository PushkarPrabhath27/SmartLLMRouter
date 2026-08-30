"""Unit tests for smartroute.exceptions (Phase 1, module 2)."""

import pytest

from smartroute.exceptions import (
    ClassificationError,
    ConfigError,
    ProviderError,
    SmartRouteError,
    StorageError,
)


@pytest.mark.parametrize(
    "exc_type",
    [ConfigError, ProviderError, ClassificationError, StorageError],
)
def test_all_errors_derive_from_base(exc_type: type[Exception]) -> None:
    assert issubclass(exc_type, SmartRouteError)
    assert issubclass(exc_type, Exception)


def test_base_error_is_catchable_as_smartroute_error() -> None:
    with pytest.raises(SmartRouteError):
        raise ConfigError("bad yaml")


def test_provider_error_carries_attempt_log() -> None:
    attempts = [
        {"provider": "openai", "model": "openai/gpt-4o-mini", "error": "Timeout"},
        {"provider": "groq", "model": "groq/llama-3.1-8b", "error": "429"},
    ]
    err = ProviderError("all providers failed", attempts)
    assert err.attempts == attempts
    assert str(err) == "all providers failed"
    assert isinstance(err, SmartRouteError)


def test_provider_error_attempts_default_empty() -> None:
    err = ProviderError("no providers configured")
    assert err.attempts == []


def test_config_error_message_preserved() -> None:
    err = ConfigError("invalid regex in override rule: [")
    assert "invalid regex" in str(err)
