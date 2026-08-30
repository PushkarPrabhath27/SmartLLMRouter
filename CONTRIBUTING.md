# Contributing to SmartRoute

Thanks for your interest in contributing. This document describes how the project is built and the quality bar every change must clear.

## Development Setup

```bash
python -m venv .venv && source .venv/Scripts/activate  # Windows Git Bash
pip install -e .[dev]
```

## Quality Bar

Every contribution must pass, before it is merged to `main`:

```bash
ruff check .            # lint (line length 100, strict imports)
ruff format --check .   # formatting
mypy smartroute/        # strict typing — zero errors
pytest --cov=smartroute # tests, target >90% coverage
```

CI runs the same checks on Python 3.10, 3.11, and 3.12.

## Development Workflow

The project follows a spec-driven, phased build plan (`specs/10_PHASES_AND_MILESTONES.md`). The files in `specs/` are the single source of truth — do not invent behavior, schemas, or APIs that are not in them.

For every module, work through these modes explicitly:

1. **Architect** — design doc first, no code.
2. **Implementer** — implement exactly what the spec says.
3. **Tester** — unit tests as you build, mocking all I/O.
4. **Reviewer** — self-review against the spec; fix and re-review.

## Code Standards

- Strict typing everywhere: `from __future__ import annotations`, all parameters and returns typed. `mypy --strict` must pass with zero errors.
- Google-style docstrings on every public class and method.
- All I/O is `async`. No `time.sleep()` in production code — use `asyncio.sleep()`.
- Use `httpx.AsyncClient`, never `requests`. No bare `except:`. No `print()` in library code — use `logging.getLogger(__name__)`.
- Scope is frozen by `specs/03_V1_SCOPE.md`: no P1/P2 features, not even stubs.

## Commit Messages

Format: `type(scope): description` — for example:

```
feat(classifier): add file path detection feature
fix(routing): handle empty reputation table
test(storage): add concurrent access tests
chore(phase-0): project scaffolding
```

## Submitting Changes

1. Run the full local check suite (lint, format, types, tests).
2. Open a pull request against `main` with a clear description of what changed and which spec section it implements.
