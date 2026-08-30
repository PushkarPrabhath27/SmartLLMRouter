# SmartRoute — Claude Code Developer Guide

## How to Use This Document

This is the **master instruction set** for Claude Code. Before writing any code, Claude must read the relevant spec file for the module being built. No code without a spec.

## Sub-Agent Definitions

Claude Code should operate in distinct modes. Switch modes explicitly with a header like:

```
[MODE: architect]
```

### 1. Architect Mode
**Trigger:** When designing a new module or refactoring existing structure.

**Responsibilities:**
- Read the relevant spec file completely
- Propose file structure and class hierarchy
- Define public interfaces before implementation
- Identify edge cases and failure modes
- **Output:** A design document (3-5 bullet points) before any code

**Prompt template:**
```
You are in Architect Mode. Read {SPEC_FILE} and design the implementation for {MODULE}.
Do not write code yet. Produce:
1. Class list with responsibilities
2. Public method signatures
3. Key internal data structures
4. Edge cases to handle
5. Testing strategy for this module
```

### 2. Implementer Mode
**Trigger:** When writing production code.

**Responsibilities:**
- Implement exactly what the spec says
- Use type hints everywhere (`def foo(x: int) -> str:`)
- Follow Google-style docstrings
- Handle all error cases defined in the spec
- No P1/P2 features, even as stubs
- **Output:** Production-ready code with `# TODO` only for known v1.5 items

**Prompt template:**
```
You are in Implementer Mode. Implement {MODULE} per {SPEC_FILE}.
Requirements:
- Strict typing (mypy --strict must pass)
- All public methods have docstrings
- All error cases from the spec are handled
- No features outside V1_SCOPE.md
- Include module-level docstring explaining purpose
```

### 3. Tester Mode
**Trigger:** After implementation is complete.

**Responsibilities:**
- Write unit tests for the implemented module
- Use `pytest` and `pytest-asyncio`
- Mock external dependencies (providers, storage)
- Test happy path + all error paths
- Aim for 100% branch coverage of the module
- **Output:** `tests/test_{module}.py`

**Prompt template:**
```
You are in Tester Mode. Write comprehensive tests for {MODULE}.
Requirements:
- Use pytest-asyncio for async code
- Mock all I/O (providers, storage, file system)
- Test every public method
- Test every error path
- Use descriptive test names: test_{function}_{scenario}_{expected}
- Include one integration test if module has external deps
```

### 4. Reviewer Mode
**Trigger:** Before committing any code.

**Responsibilities:**
- Review code against spec compliance
- Check for type safety, error handling, edge cases
- Verify no scope creep (P1/P2 code)
- Check test coverage and quality
- **Output:** Review comment list (Approve / Request Changes)

**Prompt template:**
```
You are in Reviewer Mode. Review {FILE_PATH} against {SPEC_FILE}.
Check:
1. Spec compliance -- does it implement exactly what the spec says?
2. Type safety -- will mypy --strict pass?
3. Error handling -- are all failure modes covered?
4. No scope creep -- any P1/P2 features snuck in?
5. Test quality -- do tests cover edge cases?
Output a numbered review list. Be strict.
```

### 5. Docs Mode
**Trigger:** When README, examples, or docstrings need updating.

**Responsibilities:**
- Ensure all public APIs are documented
- Update examples to match current API
- Keep README copy-paste runnable
- **Output:** Updated markdown files

## Workflow

### Per-Module Workflow (Repeat for each module)
```
1. [Architect]  -> Design doc
2. [Implementer]  -> Code + inline comments
3. [Tester]       -> Tests
4. [Reviewer]     -> Review & fix
5. [Implementer]  -> Fix issues
6. [Reviewer]     -> Final approval
7. [Docs]         -> Update if public API changed
```

### Per-Phase Workflow

At the start of each phase from `PHASES_AND_MILESTONES.md`:
```
1. Read PHASES_AND_MILESTONES.md for current phase
2. Read relevant spec files
3. List all files to create/modify
4. Execute per-module workflow for each file
5. Run full test suite
6. Run ruff + mypy
7. Update CHANGELOG.md
8. Commit with message: "feat(phase-N): description"
```

## File Naming Conventions

- **Production:** `smartroute/{module}/{file}.py`
- **Tests:** `tests/test_{module}_{file}.py` or `tests/test_{module}.py`
- **Fixtures:** `tests/conftest.py`
- **Data:** `tests/data/{file}.json`
- **Examples:** `examples/{scenario}.py`

## Code Style Rules

1. **Imports:** Grouped as stdlib, third-party, local. Sorted within groups.
2. **Types:** All function parameters and returns must be typed. Use `from __future__ import annotations`.
3. **Async:** All I/O is async. No `time.sleep()` in production code.
4. **Errors:** Custom exceptions only. No bare `except:`.
5. **Logging:** Use `logging.getLogger(__name__)`. No `print()` in library code.
6. **Constants:** UPPER_SNAKE_CASE at module level.
7. **Classes:** PascalCase. Private methods prefixed with `_`.
8. **Docstrings:** Google style. Every public class and method.

## Forbidden Patterns

These will fail review immediately:

- `requests` library (use `httpx.AsyncClient`)
- `time.sleep()` in async code (use `asyncio.sleep()`)
- Bare `except:`
- `print()` in library code
- Untyped `def foo(x):`
- P1/P2 features (shadow mode, embedding classifier, etc.)
- Any cloud/telemetry code
- Sync file I/O in async context

## Commit Message Format

```
type(scope): description
```

Types:
- `feat`: new feature
- `fix`: bug fix
- `test`: adding tests
- `docs`: documentation
- `refactor`: code change that neither fixes nor adds features
- `chore`: tooling, config, deps

Examples:
- `feat(classifier): add file path detection feature`
- `fix(routing): handle empty reputation table`
- `test(storage): add concurrent access tests`

## Debugging Commands

```bash
# Run specific test
pytest tests/test_classifier.py::test_token_count -xvs

# Run with coverage
pytest --cov=smartroute --cov-report=html

# Type check
mypy --strict smartroute/

# Lint
ruff check smartroute/ tests/
ruff format smartroute/ tests/

# Run single example
python examples/basic_usage.py
```

## When to Ask for Human Input

Claude Code should pause and ask when:

1. A spec is ambiguous or contradictory
2. A proposed solution violates a V1 constraint
3. Test coverage cannot reach 90% without testing implementation details
4. A dependency conflict arises
5. The design requires a P1/P2 feature to work correctly
