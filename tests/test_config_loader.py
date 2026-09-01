"""Unit tests for smartroute.config.loader (Phase 1, module 8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from smartroute.config.loader import ConfigLoader
from smartroute.exceptions import ConfigError

PROVIDER_ENV_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY")


@pytest.fixture(autouse=True)
def clean_config_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate tests from real env vars, SMARTROUTE_CONFIG, and cwd config files."""
    for var in (*PROVIDER_ENV_VARS, "SMARTROUTE_CONFIG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_config(path: Path, raw: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _full_user_config(**extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "providers": {
            "openai": {"api_key": "k-openai", "model": "gpt-4o-mini"},
            "anthropic": {"api_key": "k-anthropic", "model": "claude-3-sonnet"},
            "groq": {"api_key": "k-groq", "model": "llama-3.1-8b"},
        },
        "routing": {
            "low_complexity": "groq",
            "medium_complexity": "openai",
            "high_complexity": "anthropic",
        },
    }
    config.update(extra)
    return config


def _set_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.setenv(var, f"test-{var.lower()}")


class TestEnvInterpolation:
    def test_braced_env_var_set_is_replaced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-live-123")
        raw = _full_user_config()
        raw["providers"]["openai"]["api_key"] = "${OPENAI_API_KEY}"
        path = _write_config(tmp_path / "sr.yaml", raw)
        config = ConfigLoader(str(path)).load()
        assert config.providers["openai"].api_key == "sk-live-123"

    def test_bare_env_var_set_is_replaced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-bare-456")
        raw = _full_user_config()
        raw["providers"]["openai"]["api_key"] = "$OPENAI_API_KEY"
        config = ConfigLoader(str(_write_config(tmp_path / "sr.yaml", raw))).load()
        assert config.providers["openai"].api_key == "sk-bare-456"

    def test_unset_api_key_raises_config_error(self, tmp_path: Path) -> None:
        raw = _full_user_config()
        raw["providers"]["openai"]["api_key"] = "${OPENAI_API_KEY}"
        path = _write_config(tmp_path / "sr.yaml", raw)
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            ConfigLoader(str(path)).load()

    def test_unset_api_key_error_names_the_provider(self, tmp_path: Path) -> None:
        raw = _full_user_config()
        raw["providers"]["groq"]["api_key"] = "${GROQ_API_KEY}"
        path = _write_config(tmp_path / "sr.yaml", raw)
        with pytest.raises(ConfigError, match="'groq'"):
            ConfigLoader(str(path)).load()

    def test_unset_non_api_key_var_left_as_literal(self, tmp_path: Path) -> None:
        raw = _full_user_config()
        raw["providers"]["openai"]["model"] = "$MY_CUSTOM_MODEL"
        config = ConfigLoader(str(_write_config(tmp_path / "sr.yaml", raw))).load()
        assert config.providers["openai"].model == "$MY_CUSTOM_MODEL"

    def test_multiple_vars_in_one_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        raw = _full_user_config()
        raw["providers"]["openai"]["base_url"] = "${DB_HOST}:${DB_PORT}/v1"
        config = ConfigLoader(str(_write_config(tmp_path / "sr.yaml", raw))).load()
        assert config.providers["openai"].base_url == "localhost:5432/v1"

    def test_literal_key_without_env_reference_loads_as_is(self, tmp_path: Path) -> None:
        raw = _full_user_config()
        raw["providers"]["openai"]["api_key"] = "sk-literal-key"
        config = ConfigLoader(str(_write_config(tmp_path / "sr.yaml", raw))).load()
        assert config.providers["openai"].api_key == "sk-literal-key"


class TestSearchPath:
    def test_explicit_path_used(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path / "custom.yaml", _full_user_config())
        loader = ConfigLoader(str(path))
        assert loader.config_path == path
        assert loader.load().preset is None

    def test_smartroute_config_env_var_used(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = _write_config(tmp_path / "envpointed.yaml", _full_user_config())
        monkeypatch.setenv("SMARTROUTE_CONFIG", str(path))
        loader = ConfigLoader()
        assert loader.config_path == path

    def test_cwd_smartroute_yaml_used(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path / "smartroute.yaml", _full_user_config())
        loader = ConfigLoader()
        assert loader.config_path == path

    def test_no_config_anywhere_falls_back_to_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_provider_keys(monkeypatch)
        config = ConfigLoader().load()
        assert config.preset is None
        assert set(config.providers) == {"openai", "anthropic", "groq"}
        assert config.routing.low_complexity == "groq"
        assert config.routing.medium_complexity == "openai"
        assert config.routing.high_complexity == "anthropic"
        assert config.adaptation.bump_threshold == 0.3

    def test_defaults_resolve_api_keys_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_provider_keys(monkeypatch)
        config = ConfigLoader().load()
        assert config.providers["openai"].api_key == "test-openai_api_key"

    def test_missing_explicit_path_warns_and_uses_defaults(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _set_provider_keys(monkeypatch)
        with caplog.at_level("WARNING"):
            config = ConfigLoader(str(tmp_path / "nope.yaml")).load()
        assert any("not found" in record.message for record in caplog.records)
        assert config.routing.high_complexity == "anthropic"

    def test_invalid_yaml_raises_config_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("providers: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid YAML"):
            ConfigLoader(str(path)).load()

    def test_non_mapping_root_raises_config_error(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must be a mapping"):
            ConfigLoader(str(path)).load()

    def test_empty_yaml_file_loads_defaults(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_provider_keys(monkeypatch)
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        config = ConfigLoader(str(path)).load()
        assert config.adaptation.ema_alpha == 0.3

    def test_non_mapping_provider_entry_rejected(self, tmp_path: Path) -> None:
        raw = _full_user_config()
        raw["providers"]["openai"] = "not-a-mapping"
        path = _write_config(tmp_path / "sr.yaml", raw)
        with pytest.raises(ConfigError, match="invalid configuration"):
            ConfigLoader(str(path)).load()

    def test_validation_failure_wrapped_as_config_error(self, tmp_path: Path) -> None:
        raw = _full_user_config()
        raw["routing"]["overrides"] = [
            {"match": "[bad", "match_type": "regex", "model": "anthropic"}
        ]
        path = _write_config(tmp_path / "sr.yaml", raw)
        with pytest.raises(ConfigError, match="invalid configuration"):
            ConfigLoader(str(path)).load()


class TestPresetMerging:
    def test_preset_loaded_via_user_yaml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_provider_keys(monkeypatch)
        raw = _full_user_config(preset="web_dev")
        config = ConfigLoader(str(_write_config(tmp_path / "sr.yaml", raw))).load()
        assert config.preset == "web_dev"
        assert config.adaptation.bump_threshold == 0.35
        assert [o.match for o in config.routing.overrides] == ["refactor", "debug"]

    def test_user_overrides_beat_preset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_provider_keys(monkeypatch)
        raw = _full_user_config(preset="web_dev", adaptation={"bump_threshold": 0.4})
        config = ConfigLoader(str(_write_config(tmp_path / "sr.yaml", raw))).load()
        assert config.adaptation.bump_threshold == 0.4
        assert config.adaptation.ema_alpha == 0.3  # inherited from preset

    def test_user_overrides_replace_preset_rules(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_provider_keys(monkeypatch)
        raw = _full_user_config(
            preset="web_dev",
            routing={
                "low_complexity": "groq",
                "medium_complexity": "openai",
                "high_complexity": "anthropic",
                "overrides": [{"match": "deploy", "match_type": "contains", "model": "groq"}],
            },
        )
        config = ConfigLoader(str(_write_config(tmp_path / "sr.yaml", raw))).load()
        assert [o.match for o in config.routing.overrides] == ["deploy"]

    def test_unknown_preset_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _set_provider_keys(monkeypatch)
        raw = _full_user_config(preset="startup")
        with pytest.raises(ConfigError, match="unknown preset"):
            ConfigLoader(str(_write_config(tmp_path / "sr.yaml", raw))).load()


class TestProviderReplacement:
    def test_user_providers_replaced_not_merged_with_defaults(self, tmp_path: Path) -> None:
        """INTENTIONAL CONTRACT: the user `providers` section replaces the
        built-in defaults wholesale; omitted providers are NOT resurrected.

        A user who configures only one provider must end up with exactly one
        provider, or tier/override validation against "configured providers"
        would silently accept providers the user deliberately removed. Do
        not "fix" this by deep-merging providers; this test pins the approved
        behavior (Phase 1 ruling #5).
        """
        raw = {
            "providers": {"openai": {"api_key": "k-openai", "model": "gpt-4o-mini"}},
            "routing": {
                "low_complexity": "openai",
                "medium_complexity": "openai",
                "high_complexity": "openai",
            },
        }
        config = ConfigLoader(str(_write_config(tmp_path / "sr.yaml", raw))).load()
        # A deep merge would have kept anthropic and groq from the defaults.
        assert set(config.providers) == {"openai"}

        # Corollary: routing to a removed provider must now fail validation,
        # proving the removed provider is truly gone.
        bad = dict(raw)
        bad["routing"] = {
            "low_complexity": "openai",
            "medium_complexity": "openai",
            "high_complexity": "anthropic",
        }
        with pytest.raises(ConfigError, match="not configured"):
            ConfigLoader(str(_write_config(tmp_path / "sr2.yaml", bad))).load()

    def test_user_provider_fields_deep_merge(self, tmp_path: Path) -> None:
        """Within one provider, partial user fields merge over the entry."""
        raw = _full_user_config()
        raw["providers"]["openai"] = {"api_key": "k", "model": "gpt-4o-mini", "timeout": 15}
        config = ConfigLoader(str(_write_config(tmp_path / "sr.yaml", raw))).load()
        assert config.providers["openai"].timeout == 15
        assert config.providers["openai"].max_retries == 2

    def test_no_preset_yaml_still_gets_adaptation_defaults(self, tmp_path: Path) -> None:
        config = ConfigLoader(str(_write_config(tmp_path / "sr.yaml", _full_user_config()))).load()
        assert config.adaptation.min_calls_before_bump == 10
        assert config.logging.level == "INFO"
