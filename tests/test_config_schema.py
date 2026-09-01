"""Unit tests for smartroute.config.schema (Phase 1, module 7)."""

import pytest
from pydantic import ValidationError

from smartroute.config.schema import (
    AdaptationConfig,
    Config,
    LoggingConfig,
    OverrideRule,
    ProviderConfig,
    RoutingConfig,
)


def _providers(
    remove: tuple[str, ...] = (), **overrides: ProviderConfig
) -> dict[str, ProviderConfig]:
    providers = {
        "openai": ProviderConfig(api_key="k-openai", model="gpt-4o-mini"),
        "anthropic": ProviderConfig(api_key="k-anthropic", model="claude-3-sonnet"),
        "groq": ProviderConfig(api_key="k-groq", model="llama-3.1-8b"),
    }
    for provider_key in remove:
        del providers[provider_key]
    providers.update(overrides)
    return providers


def _routing(**overrides: object) -> RoutingConfig:
    base: dict[str, object] = {
        "low_complexity": "groq",
        "medium_complexity": "openai",
        "high_complexity": "anthropic",
    }
    base.update(overrides)
    return RoutingConfig(**base)  # type: ignore[arg-type]


def _config(**overrides: object) -> Config:
    base: dict[str, object] = {
        "providers": _providers(),
        "routing": _routing(),
    }
    base.update(overrides)
    return Config(**base)  # type: ignore[arg-type]


class TestValidConfig:
    def test_minimal_valid_config(self) -> None:
        config = _config()
        assert config.adaptation.bump_threshold == 0.3
        assert config.adaptation.ema_alpha == 0.3
        assert config.adaptation.cooldown_minutes == 5
        assert config.adaptation.min_calls_before_bump == 10
        assert config.adaptation.enabled is True
        assert config.routing.overrides == []
        assert config.logging.level == "INFO"

    def test_config_with_all_match_types(self) -> None:
        config = _config(
            routing=_routing(
                overrides=[
                    {"match": "debug this function", "match_type": "exact", "model": "anthropic"},
                    {"match": "contract", "match_type": "contains", "model": "anthropic"},
                    {"match": r"\b(legal|law)\b", "match_type": "regex", "model": "anthropic"},
                    {"match": "/legal/", "match_type": "path", "model": "anthropic"},
                ]
            )
        )
        assert len(config.routing.overrides) == 4
        assert {o.match_type for o in config.routing.overrides} == {
            "exact",
            "contains",
            "regex",
            "path",
        }

    def test_config_is_frozen(self) -> None:
        config = _config()
        with pytest.raises(ValidationError):
            config.preset = "web_dev"  # type: ignore[misc]

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="shadow_mode"):
            _config(shadow_mode=True)


class TestProviderValidation:
    def test_empty_providers_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one provider"):
            _config(providers={})

    def test_unknown_provider_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown provider keys"):
            _config(
                providers={
                    "cohere": ProviderConfig(api_key="k", model="m"),
                    "openai": ProviderConfig(api_key="k", model="gpt-4o-mini"),
                }
            )

    def test_missing_api_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(model="gpt-4o-mini")  # type: ignore[call-arg]

    def test_empty_api_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(api_key="", model="gpt-4o-mini")

    def test_unknown_override_model_rejected_at_config_level(self) -> None:
        with pytest.raises(ValidationError, match="not configured"):
            _config(
                providers=_providers(remove=("groq",)),
                routing=_routing(
                    overrides=[{"match": "x", "match_type": "contains", "model": "groq"}]
                ),
            )


class TestRoutingValidation:
    def test_tier_mapping_to_unconfigured_provider_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not configured"):
            _config(providers=_providers(remove=("anthropic",)))

    @pytest.mark.parametrize("field", ["low_complexity", "medium_complexity", "high_complexity"])
    def test_tier_mapping_to_unknown_provider_key_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            _routing(**{field: "cohere"})

    def test_invalid_regex_rejected_with_pattern_in_message(self) -> None:
        with pytest.raises(ValidationError, match=r"invalid regex.*\[bad"):
            OverrideRule(match="[bad", match_type="regex", model="anthropic")

    def test_invalid_regex_only_checked_for_regex_rules(self) -> None:
        rule = OverrideRule(match="[not a regex", match_type="contains", model="anthropic")
        assert rule.match == "[not a regex"

    def test_unknown_match_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OverrideRule(match="x", match_type="fuzzy", model="anthropic")  # type: ignore[arg-type]

    def test_unknown_fallback_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="fallback keys"):
            _routing(fallback={"ultra": ["groq"]})

    def test_duplicate_in_fallback_chain_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            _routing(fallback={"medium": ["openai", "groq", "groq"]})

    def test_circular_fallback_rejected(self) -> None:
        with pytest.raises(ValidationError, match="circular"):
            _routing(fallback={"medium": ["groq", "openai"]})

    def test_primary_at_position_zero_is_allowed(self) -> None:
        routing = _routing(fallback={"medium": ["openai", "groq"]})
        assert routing.fallback is not None
        assert routing.fallback["medium"] == ["openai", "groq"]

    def test_fallback_entries_must_be_known_providers(self) -> None:
        with pytest.raises(ValidationError):
            _routing(fallback={"medium": ["openai", "cohere"]})


class TestAdaptationValidation:
    def test_bump_threshold_out_of_bounds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AdaptationConfig(bump_threshold=1.5)

    def test_ema_alpha_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AdaptationConfig(ema_alpha=0.0)
        with pytest.raises(ValidationError):
            AdaptationConfig(ema_alpha=1.5)
        assert AdaptationConfig(ema_alpha=1.0).ema_alpha == 1.0

    def test_negative_cooldown_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AdaptationConfig(cooldown_minutes=-1)


class TestPresetValidation:
    def test_unknown_preset_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _config(preset="startup")

    @pytest.mark.parametrize("preset", ["general", "web_dev", "data_science"])
    def test_known_presets_accepted(self, preset: str) -> None:
        assert _config(preset=preset).preset == preset


class TestLoggingConfig:
    def test_default_format(self) -> None:
        assert "%(levelname)s" in LoggingConfig().format

    def test_invalid_level_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LoggingConfig(level="VERBOSE")  # type: ignore[arg-type]
