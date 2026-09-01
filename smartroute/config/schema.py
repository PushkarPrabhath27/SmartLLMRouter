"""Pydantic models for SmartRoute configuration (spec 09).

All models are frozen and reject unknown fields, so invalid or misspelled
keys fail loudly at load time with clear messages.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

KNOWN_PROVIDERS = frozenset({"openai", "anthropic", "groq"})
KNOWN_MATCH_TYPES = ("exact", "contains", "regex", "path")
KNOWN_PRESETS = ("general", "web_dev", "data_science")
KNOWN_TIERS = ("low", "medium", "high")


class _FrozenModel(BaseModel):
    """Base for all config models: immutable, unknown keys rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProviderConfig(_FrozenModel):
    """Configuration for a single LLM provider (spec 09).

    Attributes:
        api_key: Provider API key; may reference environment variables
            (``${VAR}`` / ``$VAR``) which the loader resolves.
        model: Model identifier used for this provider.
        base_url: Optional override for proxies or Azure-style endpoints.
        timeout: Request timeout in seconds.
        max_retries: Retries per request on transient failures.
    """

    api_key: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: str | None = None
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=2, ge=0)


class OverrideRule(_FrozenModel):
    """A YAML routing override rule (spec 09).

    Attributes:
        match: The pattern to match against the prompt (or detected paths
            for ``match_type="path"``).
        match_type: One of ``exact``, ``contains``, ``regex``, ``path``.
        model: Provider key to route to when the rule matches.
        description: Optional human-readable explanation, surfaced in
            ``RoutingMeta.override_applied``.
    """

    match: str = Field(min_length=1)
    match_type: Literal["exact", "contains", "regex", "path"]
    model: str
    description: str | None = None

    @field_validator("model")
    @classmethod
    def _model_is_known_provider(cls, value: str) -> str:
        if value not in KNOWN_PROVIDERS:
            raise ValueError(
                f"unknown provider key '{value}' (must be one of {sorted(KNOWN_PROVIDERS)})"
            )
        return value

    @model_validator(mode="after")
    def _regex_must_compile(self) -> OverrideRule:
        if self.match_type == "regex":
            try:
                re.compile(self.match)
            except re.error as exc:
                raise ValueError(
                    f"invalid regex in override rule: pattern '{self.match}' "
                    f"failed to compile ({exc})"
                ) from exc
        return self


class RoutingConfig(_FrozenModel):
    """Routing tier mapping, fallback chains, and override rules (spec 09).

    Attributes:
        low_complexity: Provider key for the low complexity tier.
        medium_complexity: Provider key for the medium complexity tier.
        high_complexity: Provider key for the high complexity tier.
        fallback: Optional per-tier provider chains (``low``/``medium``/
            ``high`` -> ordered provider keys, primary first).
        overrides: Ordered override rules; the first match wins.
    """

    low_complexity: str
    medium_complexity: str
    high_complexity: str
    fallback: dict[str, list[str]] | None = None
    overrides: list[OverrideRule] = []

    @model_validator(mode="after")
    def _validate_tiers(self) -> RoutingConfig:
        for tier in KNOWN_TIERS:
            provider_key = getattr(self, f"{tier}_complexity")
            if provider_key not in KNOWN_PROVIDERS:
                raise ValueError(
                    f"{tier}_complexity must map to a valid provider key "
                    f"{sorted(KNOWN_PROVIDERS)}, got '{provider_key}'"
                )
        return self

    @model_validator(mode="after")
    def _validate_fallback(self) -> RoutingConfig:
        if self.fallback is None:
            return self
        unknown_keys = set(self.fallback) - set(KNOWN_TIERS)
        if unknown_keys:
            raise ValueError(
                f"fallback keys must be among {list(KNOWN_TIERS)}, got {sorted(unknown_keys)}"
            )
        for tier in KNOWN_TIERS:
            chain = self.fallback.get(tier)
            if not chain:
                continue
            unknown_entries = set(chain) - KNOWN_PROVIDERS
            if unknown_entries:
                raise ValueError(
                    f"fallback chain for '{tier}' contains unknown provider keys "
                    f"{sorted(unknown_entries)} (must be one of {sorted(KNOWN_PROVIDERS)})"
                )
            if len(set(chain)) != len(chain):
                raise ValueError(
                    f"fallback chain for '{tier}' contains duplicate providers: {chain}"
                )
            primary = getattr(self, f"{tier}_complexity")
            if primary in chain[1:]:
                raise ValueError(
                    f"circular fallback: tier '{tier}' lists its own primary "
                    f"provider '{primary}' after position 0"
                )
        return self


class AdaptationConfig(_FrozenModel):
    """Adaptive reputation settings (spec 09).

    Attributes:
        enabled: Master switch for the adaptive loop.
        bump_threshold: EMA below this (with enough calls) triggers a bump.
        cooldown_minutes: Minimum minutes between bumps for one bucket.
        ema_alpha: EMA smoothing factor in (0, 1].
        min_calls_before_bump: Minimum recorded signals before a bump.
    """

    enabled: bool = True
    bump_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=5, ge=0)
    ema_alpha: float = Field(default=0.3, gt=0.0, le=1.0)
    min_calls_before_bump: int = Field(default=10, ge=0)


class LoggingConfig(_FrozenModel):
    """Logging settings (spec 09)."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Config(_FrozenModel):
    """Fully validated SmartRoute configuration (spec 09).

    Attributes:
        preset: Optional preset name the config was bootstrapped from.
        providers: Provider configurations keyed by provider key.
        routing: Tier mapping and override rules.
        adaptation: Adaptive reputation settings.
        logging: Logging settings.
    """

    preset: Literal["general", "web_dev", "data_science"] | None = None
    providers: dict[str, ProviderConfig]
    routing: RoutingConfig
    adaptation: AdaptationConfig = AdaptationConfig()
    logging: LoggingConfig = LoggingConfig()

    @field_validator("providers")
    @classmethod
    def _validate_providers(cls, value: dict[str, ProviderConfig]) -> dict[str, ProviderConfig]:
        if not value:
            raise ValueError("providers must contain at least one provider (spec rule 1)")
        unknown = set(value) - KNOWN_PROVIDERS
        if unknown:
            raise ValueError(
                f"unknown provider keys {sorted(unknown)} (must be one of "
                f"{sorted(KNOWN_PROVIDERS)})"
            )
        return value

    @model_validator(mode="after")
    def _validate_tiers_and_overrides(self) -> Config:
        for tier in KNOWN_TIERS:
            provider_key = getattr(self.routing, f"{tier}_complexity")
            if provider_key not in self.providers:
                raise ValueError(
                    f"{tier}_complexity maps to provider '{provider_key}' which "
                    "is not configured in providers"
                )
        for rule in self.routing.overrides:
            if rule.model not in self.providers:
                raise ValueError(
                    f"override rule '{rule.match}' routes to provider "
                    f"'{rule.model}' which is not configured in providers"
                )
        return self
