"""YAML configuration loading, env interpolation, and preset merging (spec 09).

Layering order (lowest to highest): built-in defaults -> preset -> user YAML.
The user layer always wins. See ``specs/09_CONFIG_SPEC.md``.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from importlib import resources
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from smartroute.config.schema import KNOWN_PRESETS, Config
from smartroute.exceptions import ConfigError

logger = logging.getLogger(__name__)

ENV_PATTERN = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

_REPLACE_KEYS = frozenset({"providers"})

_PRESET_PACKAGE = "smartroute.config.presets"

_BUILTIN_DEFAULTS: dict[str, Any] = {
    "providers": {
        "openai": {
            "api_key": "${OPENAI_API_KEY}",
            "model": "gpt-4o-mini",
        },
        "anthropic": {
            "api_key": "${ANTHROPIC_API_KEY}",
            "model": "claude-3-sonnet",
        },
        "groq": {
            "api_key": "${GROQ_API_KEY}",
            "model": "llama-3.1-8b",
        },
    },
    "routing": {
        "low_complexity": "groq",
        "medium_complexity": "openai",
        "high_complexity": "anthropic",
        "overrides": [],
    },
    "adaptation": {
        "enabled": True,
        "bump_threshold": 0.3,
        "cooldown_minutes": 5,
        "ema_alpha": 0.3,
        "min_calls_before_bump": 10,
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
}


class ConfigLoader:
    """Load, interpolate, merge, and validate SmartRoute configuration.

    Args:
        config_path: Explicit path to a YAML config file. If None, the search
            order is ``SMARTROUTE_CONFIG`` env var, ``./smartroute.yaml``,
            then ``~/.smartroute/config.yaml``. When nothing is found, the
            built-in defaults (general preset behavior) are used and a
            warning is logged.
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path: Path | None = self._resolve_path(config_path)
        if config_path is not None and self.config_path is None:
            logger.warning(
                "config file not found at '%s'; using built-in defaults (general preset behavior)",
                config_path,
            )

    def load(self) -> Config:
        """Load and validate the full configuration.

        Returns:
            A validated, frozen :class:`Config`.

        Raises:
            ConfigError: If the YAML is malformed, an ``api_key`` references
                an unset environment variable, or validation fails.
        """
        user_raw = self._read_yaml()
        merged = self._merge_layers(user_raw)
        merged = self._apply_env_interpolation(merged)
        self._validate_api_keys(merged)
        return self._parse_and_validate(merged)

    # ------------------------------------------------------------------
    # Path resolution and reading
    # ------------------------------------------------------------------

    def _resolve_path(self, config_path: str | None) -> Path | None:
        """Resolve the config file path per the spec 04 search order."""
        candidates: list[Path] = []
        if config_path is not None:
            candidates.append(Path(config_path))
        env_path = os.environ.get("SMARTROUTE_CONFIG")
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(Path.cwd() / "smartroute.yaml")
        candidates.append(Path.home() / ".smartroute" / "config.yaml")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _read_yaml(self) -> dict[str, Any]:
        """Read the resolved YAML file, returning {} when no file exists."""
        if self.config_path is None:
            return {}
        try:
            text = self.config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"failed to read config file {self.config_path}: {exc}") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {self.config_path}: {exc}") from exc
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(
                f"config root must be a mapping, got {type(data).__name__} in {self.config_path}"
            )
        return data

    # ------------------------------------------------------------------
    # Layer merging
    # ------------------------------------------------------------------

    def _merge_layers(self, user_raw: dict[str, Any]) -> dict[str, Any]:
        """Merge defaults <- preset <- user YAML (user wins at every level).

        The ``providers`` section is replaced (not deep-merged) by the user's
        value when present, so omitted default providers are not resurrected.
        """
        merged = copy.deepcopy(_BUILTIN_DEFAULTS)
        preset_name = user_raw.get("preset")
        if preset_name is not None:
            if preset_name not in KNOWN_PRESETS:
                raise ConfigError(
                    f"unknown preset '{preset_name}' (must be one of {list(KNOWN_PRESETS)})"
                )
            preset_raw = self._load_preset(preset_name)
            merged = _deep_merge(merged, preset_raw, _REPLACE_KEYS)
        return _deep_merge(merged, user_raw, _REPLACE_KEYS)

    def _load_preset(self, preset_name: str) -> dict[str, Any]:
        """Load a built-in preset YAML from the package."""
        path = resources.files(_PRESET_PACKAGE) / f"{preset_name}.yaml"
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, FileNotFoundError) as exc:
            raise ConfigError(f"preset file for '{preset_name}' is missing: {exc}") from exc
        data = cast("Any", yaml.safe_load(text))
        if not isinstance(data, dict):
            raise ConfigError(f"preset '{preset_name}' is not a valid mapping")
        return cast("dict[str, Any]", data)

    # ------------------------------------------------------------------
    # Environment interpolation
    # ------------------------------------------------------------------

    def _apply_env_interpolation(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Replace ``${VAR}`` / ``$VAR`` in all string values from the env.

        Unresolved variables are left as literal strings, except ``api_key``
        values, which are rejected separately by ``_validate_api_keys``.
        """
        env = os.environ

        def substitute(value: str) -> str:
            def replace(match: re.Match[str]) -> str:
                var_name = match.group(1) or match.group(2)
                return env.get(var_name, match.group(0))

            return ENV_PATTERN.sub(replace, value)

        def walk(node: Any) -> Any:
            if isinstance(node, dict):
                return {key: walk(value) for key, value in node.items()}
            if isinstance(node, list):
                return [walk(item) for item in node]
            if isinstance(node, str):
                return substitute(node)
            return node

        return cast("dict[str, Any]", walk(raw))

    def _validate_api_keys(self, raw: dict[str, Any]) -> None:
        """Reject api_key values that still reference unset env variables.

        Args:
            raw: The interpolated raw config tree.

        Raises:
            ConfigError: If a provider's api_key still contains an
                ``${VAR}``/``$VAR`` reference that the environment could not
                satisfy (spec 09: api_key is required, other fields keep the
                literal string).
        """
        providers = raw.get("providers")
        if not isinstance(providers, dict):
            return
        for provider_key, provider_cfg in providers.items():
            if not isinstance(provider_cfg, dict):
                continue
            api_key = provider_cfg.get("api_key")
            if isinstance(api_key, str) and ENV_PATTERN.search(api_key):
                raise ConfigError(
                    f"api_key for provider '{provider_key}' references unset "
                    f"environment variable(s): '{api_key}'. Set the environment "
                    "variable or provide a literal API key in the config file."
                )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _parse_and_validate(self, raw: dict[str, Any]) -> Config:
        """Validate the merged tree against the Pydantic config schema.

        Raises:
            ConfigError: With the full validation report on failure.
        """
        try:
            return Config.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(f"invalid configuration: {exc}") from exc


def _deep_merge(
    base: dict[str, Any], override: dict[str, Any], replace_keys: frozenset[str]
) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins).

    Keys listed in ``replace_keys`` are replaced wholesale instead of merged.
    """
    result = dict(base)
    for key, value in override.items():
        if (
            key in replace_keys
            or key not in result
            or not isinstance(result[key], dict)
            or not isinstance(value, dict)
        ):
            result[key] = copy.deepcopy(value)
        else:
            result[key] = _deep_merge(result[key], value, replace_keys)
    return result
