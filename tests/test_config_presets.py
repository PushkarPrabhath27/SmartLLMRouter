"""Unit tests for the built-in domain presets (Phase 1, module 9).

Preset files are validated against the Pydantic config models directly; the
loader's preset-merge behavior is tested in test_config_loader.py.
"""

from importlib import resources
from typing import Any

import pytest
import yaml

from smartroute.config.schema import Config, ProviderConfig

PRESET_NAMES = ("general", "web_dev", "data_science")


def _load_preset(name: str) -> dict[str, Any]:
    text = (resources.files("smartroute.config.presets") / f"{name}.yaml").read_text(
        encoding="utf-8"
    )
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    return data


def _providers() -> dict[str, ProviderConfig]:
    return {
        "openai": {"api_key": "k", "model": "gpt-4o-mini"},
        "anthropic": {"api_key": "k", "model": "claude-3-sonnet"},
        "groq": {"api_key": "k", "model": "llama-3.1-8b"},
    }


def _config_with_preset(name: str) -> Config:
    raw = {**_load_preset(name), "preset": name, "providers": _providers()}
    return Config.model_validate(raw)


class TestPresetFiles:
    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_preset_parses_and_validates_against_schema(self, name: str) -> None:
        config = _config_with_preset(name)
        assert config.preset == name
        assert config.routing.low_complexity == "groq"
        assert config.routing.medium_complexity == "openai"
        assert config.routing.high_complexity == "anthropic"
        assert config.adaptation.ema_alpha == 0.3
        assert config.adaptation.cooldown_minutes == 5
        assert config.adaptation.min_calls_before_bump == 10

    def test_general_has_no_overrides_and_balanced_defaults(self) -> None:
        config = _config_with_preset("general")
        assert config.routing.overrides == []
        assert config.adaptation.bump_threshold == 0.3

    def test_web_dev_routes_refactor_and_debug_to_anthropic(self) -> None:
        config = _config_with_preset("web_dev")
        assert config.adaptation.bump_threshold == 0.35
        assert [(o.match, o.match_type, o.model) for o in config.routing.overrides] == [
            ("refactor", "contains", "anthropic"),
            ("debug", "contains", "anthropic"),
        ]

    def test_data_science_overrides_use_regex_and_contains(self) -> None:
        config = _config_with_preset("data_science")
        rules = config.routing.overrides
        assert len(rules) == 2
        assert rules[0].match_type == "regex"
        assert rules[0].model == "openai"
        assert rules[1].match == "dataframe"
        assert rules[1].match_type == "contains"
        assert rules[1].description is not None
