# Changelog

All notable changes to SmartRoute are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - unreleased

### Added

- Project scaffolding: package layout per architecture spec, `pyproject.toml` (hatchling build), ruff + `mypy --strict` + pytest configuration, and GitHub Actions CI for Python 3.10–3.12. (Phase 0)
- Core infrastructure (Phase 1): shared types and enums (`TaskType`, `ComplexityBucket`, API dataclasses, storage records), exception hierarchy (`SmartRouteError`, `ConfigError`, `ProviderError`, `ClassificationError`, `StorageError`).
- Local SQLite storage layer: schema DDL for `decisions`, `signals`, `reputation`, `adaptations`, `config_overrides`; aiosqlite `Storage` facade with WAL mode, foreign keys, drop-and-recreate migration, and CRUD for decisions, signals, reputation, and adaptations.
- Configuration layer: Pydantic config models enforcing all spec validation rules (provider keys, tier mapping, override match types, regex compilation, fallback chains), `ConfigLoader` with `${VAR}`/`$VAR` environment interpolation, layered defaults -> preset -> user merging, and `general`/`web_dev`/`data_science` domain presets.
- Added `pyyaml` dependency (and `types-PyYAML` for dev) required for YAML configuration.
