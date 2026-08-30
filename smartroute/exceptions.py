"""Custom exception hierarchy for SmartRoute.

Every error the library raises derives from :class:`SmartRouteError`, so host
applications can catch all SmartRoute failures with a single except clause.
"""

from __future__ import annotations


class SmartRouteError(Exception):
    """Base exception for all SmartRoute errors."""


class ConfigError(SmartRouteError):
    """Configuration is missing or invalid."""


class ProviderError(SmartRouteError):
    """All providers in the fallback chain failed.

    Attributes:
        attempts: The attempt log, one dict per provider tried, each with
            ``provider``, ``model``, and ``error`` keys.
    """

    def __init__(self, message: str, attempts: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.attempts: list[dict[str, str]] = attempts if attempts is not None else []


class ClassificationError(SmartRouteError):
    """Classifier failed and fail-open could not recover."""


class StorageError(SmartRouteError):
    """SQLite operation failed."""
